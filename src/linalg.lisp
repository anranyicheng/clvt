(in-package :clvt)

(defvar *simd-matmul-2d-fn* nil
  "由 simd-matmul.lisp 注册的 SIMD 2D matmul 快速路径。
   返回 VT 表示成功，返回 NIL 表示 vt-matmul 回退到 einsum。")

(defvar *simd-batched-matmul-fn* nil
  "批量 matmul 的 SIMD 快速路径。在 vt-matmul 的 >2D 分支里尝试调用，
   返回 VT 成功，NIL 回退到 vt-einsum。")

(defvar *simd-gemm-raw-fn* nil
  "由 src/simd-matmul.lisp 注册的**裸数据** 2D GEMM（只接受 NN 布局）。

   签名：(a-data b-data c-data m k n a-off b-off c-off elem-kind)
         → T 已计算 / NIL 未接管。

   语义与 %MATMUL-*-FAST-KERNEL 家族**完全一致**：C = C + A·B，其中
   A:(m,k)、B:(k,n)、C:(m,n) 全部行主序连续、互不重叠；A-OFF/B-OFF/C-OFF
   是**元素**下标（不是字节）；ELEM-KIND 取 :FLOAT64 / :FLOAT32 /
   :INT64 / :INT32。

   为什么需要它（§16.26）：einsum 的批量 GEMM 快通道原先只调 linalg.lisp
   里的标量内核（4 路手写展开、无 SIMD、无分块），而 VT-MATMUL 走
   simd-matmul.lisp 的 AVX/FMA + 64×256×256 分块内核 —— 两条路径互不
   相通，于是同一台机器上
       (vt-einsum \"ij,jk->ik\" a b)   比   (vt-matmul a b)   慢 5.7~6.4×
   （1024² float64、单线程实测；多线程下内存带宽掩盖一部分，实测 1.78×）。
   linalg.lisp 早于 simd-matmul.lisp 编译，故沿用 *SIMD-MATMUL-2D-FN*
   的钩子模式。为 NIL 或返回 NIL 时退回标量内核，语义不变（只是慢）。")

;;; ------------------------------------------------------------------
;;; 复数 einsum 快路径钩子（由 src/complex.lisp 注册，§16.21）
;;;
;;; linalg.lisp 先于 complex.lisp 编译，故用钩子而非直接引用。两个钩子
;;; 只在「输入全为 complex64/complex128」时被调用，签名如下：
;;;   *CX-EINSUM-GEMM-FN* : (data-a data-b data-c m k n
;;;                          a-off b-off c-off
;;;                          sa-i sa-j sb-j sb-k sc-i sc-k) → C += A@B
;;;   *CX-EINSUM-DOT-FN*  : (a b) → 复数标量值（两输入已连续，不取共轭）
;;; 为 NIL 时复数走通用累加状态机（语义正确，只是慢）。
;;; ------------------------------------------------------------------
(defvar *cx-einsum-gemm-fn* nil
  "复数 einsum 的 GEMM 快路径（见 src/complex.lisp 的 %CX-GEMM-STRIDED）。")

(defvar *cx-einsum-dot-fn* nil
  "复数 einsum 的全收缩内积快路径（见 src/complex.lisp 的 %CX-EINSUM-DOT）。")

(defvar *vt-einsum-parse-cache* (make-hash-table :test 'equal))

(defvar *vt-einsum-cache-lock*
  (sb-thread:make-mutex :name "einsum-cache-lock"))

(defconstant +vt-einsum-cache-max+ 1024)

;;; 多线程 GEMM 调度
(defvar *matmul-thread-count* (- *processor-number* 1)
  "并行 GEMM 的最大工作线程数。建议设为 CPU 物理核数-1。")

(defvar *matmul-parallel-threshold* 5000000
  "当 m*k*n 低于该阈值时走单线程，避免线程创建开销。")

(defvar *enable-parallel-matmul* t
  "全局开关：nil 则强制所有 GEMM 走单线程（调试用）。")

(defmacro %with-einsum-cache-lock (&body body)
  `(sb-thread:with-mutex (*vt-einsum-cache-lock*) ,@body))

(declaim (inline get-parsed-subscripts))
(defun get-parsed-subscripts (str)
  (declare (optimize (speed 3) (safety 0)) (simple-string str))
  (%with-einsum-cache-lock
    (let ((cached (gethash str *vt-einsum-parse-cache*)))
      (if cached
          (values (the list (first cached))
                  (the list (second cached))
                  (the boolean (third cached)))
          (multiple-value-bind (inputs output explicit-p)
              (parse-subscript-tokens str)
            ;; 缓存超限时整体清空，简单有效的近似 LRU
            (when (>= (hash-table-count *vt-einsum-parse-cache*)
                      +vt-einsum-cache-max+)
              (clrhash *vt-einsum-parse-cache*))
            (setf (gethash str *vt-einsum-parse-cache*)
                  (list inputs output explicit-p))
            (values inputs output explicit-p))))))

(declaim (inline parse-subscript-tokens))
(defun parse-subscript-tokens (str)
  (declare (optimize (speed 3)) (simple-string str))
  (let ((len (length str))
	(inputs nil)
	(current-sub nil)
	(output nil)
	(i 0)
	(state :inputs))
    (declare (fixnum len i))
    (flet ((save-current-sub ()
             (when current-sub
               (let ((final-sub (nreverse current-sub)))
                 (if (eq state :inputs)
		     (push final-sub inputs)
		     (setf output final-sub)))
               (setf current-sub nil))))
      (loop while (< i len) do
        (let ((char (char str i)))
          (declare (character char))
          (cond
            ((and (char= char #\.) (< (+ i 2) len)
		  (char= (char str (+ i 1)) #\.)
		  (char= (char str (+ i 2)) #\.))
             (push :ellipsis current-sub)
	     (incf i 3))
            ((and (char= char #\-)
		  (< (1+ i) len)
		  (char= (char str (1+ i)) #\>))
             (save-current-sub)
	     (setf state :outputs)
	     (incf i 2))
            ((char= char #\,)
	     (when (eq state :inputs)
	       (save-current-sub))
	     (incf i))
            ((char= char #\space)
	     (incf i))
            ((char= char #\.)
	     (error "invalid syntax: single '.' found."))
            (t (push (the fixnum (char-code char))
		     current-sub)
	       (incf i)))))
      (save-current-sub)
      (values (nreverse inputs)
	      output
	      (if (eq state :outputs) t nil)))))

(declaim (inline expand-ellipsis))
(defun expand-ellipsis (input-subs output-sub vts)
  (declare (optimize (speed 3) (safety 0)))
  (with-float-safe
    (let ((ellipsis-ranks nil))
      (declare (list ellipsis-ranks))
      (loop for sub in input-subs for vt in vts
            for explicit-rank = (count-if #'(lambda (x)
					      (and (typep x 'fixnum)
						   (>= (the fixnum x) 0)))
					  sub)
            for has-ellipsis = (member :ellipsis sub)
            for tensor-rank = (length (vt-shape vt))
            for implicit-rank = (the fixnum
				     (if has-ellipsis
					 (- tensor-rank explicit-rank) 0))
            do (when (and has-ellipsis (< implicit-rank 0))
		 (error "subscript dimension mismatch"))
               (when (> (count :ellipsis sub) 1)
		 (error "only one ellipsis allowed"))
               (push implicit-rank ellipsis-ranks))
      (setf ellipsis-ranks (nreverse ellipsis-ranks))
      (let ((max-implicit-rank (reduce #'max ellipsis-ranks :initial-value 0)))
        (declare (fixnum max-implicit-rank))
        (let ((all-ellipsis-labels
		(loop for i from 1 to max-implicit-rank
		      collect (the fixnum (- i)))))
          (flet
	      ((expand-sub (sub implicit-rank)
                 (declare (list sub) (fixnum implicit-rank))
                 (let ((pos (position :ellipsis sub)))
                   (if (not pos) sub
                       (let* ((before (subseq sub 0 pos))
                              (after (nthcdr (1+ pos) sub))
                              (start-idx
				(the fixnum
				     (- max-implicit-rank implicit-rank)))
                              (labels-to-use
				(subseq all-ellipsis-labels start-idx)))
                         (nconc before labels-to-use after))))))
            (values (mapcar #'expand-sub input-subs ellipsis-ranks)
                    (when output-sub
		      (expand-sub output-sub max-implicit-rank)))))))))

(declaim (inline analyze-einsum))
(defun analyze-einsum (input-subs output-subs vts explicit-p)
  (declare (type list input-subs output-subs vts)
	   (type boolean explicit-p) (optimize (speed 3)))
  (with-float-safe
    (when (and explicit-p
	       output-subs
               (> (length output-subs)
                  (length (remove-duplicates output-subs :test #'eql))))
      (error "einsum: output subscript ~a repeats a label" output-subs))
    (let ((label-dims
	    (make-array 266 :element-type 'fixnum :initial-element -1))
          (label-counts
	    (make-array 266 :element-type 'fixnum :initial-element 0))
          (all-labels-list nil))
      (macrolet
	  ((to-idx (code)
	     `(the fixnum (if (< ,code 0)
			      (+ 256 ,code)
			      ,code))))
        (loop for sub list in input-subs
	      for tns in vts for shape = (vt-shape tns) do
		(unless (= (length sub)
			   (length shape))
		  (error "subscript dimension mismatch"))
		(loop for label fixnum in sub
		      for dim fixnum in shape do
			(incf (aref label-counts (to-idx label)))
			(pushnew label all-labels-list)
			(let ((old-dim (aref label-dims (to-idx label))))
			  (when (and (/= old-dim -1)
				     (/= dim 1)
				     (/= old-dim 1)
				     (/= old-dim dim))
			    (error "dimension conflict for label ~a: ~a vs ~a" 
				   (if (< label 0)
				       (format nil "ellipsis-dim(~a)" label)
				       (string (code-char label))) old-dim dim))
			  (setf (aref label-dims (to-idx label))
				(cond ((= old-dim -1) dim)
				      ((or (= old-dim 0)
					   (= dim 0)) 0)
				      (t (max old-dim dim)))))))
        (let* ((final-output-subs output-subs)
	       (explicit-mode explicit-p))
          (unless explicit-mode
            (let ((ellipsis-subs nil)
		  (normal-subs nil))
              (loop for label fixnum in all-labels-list
                    for mapped-idx = (if (< label 0)
					 (+ 256 label)
					 label)
                    when (or (< label 0)
			     (= (aref label-counts mapped-idx) 1))
                      do (if (< label 0)
			     (push label ellipsis-subs)
			     (push label normal-subs)))
              (setf final-output-subs
		    (append (sort ellipsis-subs #'>)
			    (sort normal-subs #'<)))))
          ;; 注：显式模式下「只出现在输入、不在输出」的下标（如
          ;; ij,kl->ij 的 k/l）numpy 会对该操作数对应轴求和而非报错，
          ;; 本实现语义一致，无需校验。
          (let* ((sum-labels
		   (set-difference all-labels-list final-output-subs))
                 (all-labels
		   (coerce (append final-output-subs sum-labels)
			   'simple-vector)))
            (values all-labels label-dims final-output-subs)))))))

;;; ------------------------------------------------------------------
;;; 通用并行调度器（无类型约束，rows-fn 负责全部计算）
;;; ------------------------------------------------------------------

(declaim (notinline call-parallel-gemm))
(defun call-parallel-gemm (rows-fn a-data b-data c-data m k n a-off b-off c-off)
  "把 m 行切成若干份，每份一个 worker 线程执行 rows-fn。
   rows-fn 签名: (a-data b-data c-data k n a-off b-off c-off i-start i-end)。"
  (declare (type function rows-fn)
           (type fixnum m k n a-off b-off c-off))
  (cond
    ((or (not *enable-parallel-matmul*)
         (< m 64)
         (< (* m k n) *matmul-parallel-threshold*))
     (funcall rows-fn a-data b-data c-data k n a-off b-off c-off 0 m))
    (t
     (let* ((nthreads (max 1 (min *matmul-thread-count* m)))
            (rows-per (ceiling m nthreads))
            (threads nil))
       (declare (type fixnum nthreads rows-per))
       (loop for tid of-type fixnum from 0 below nthreads
             do (let* ((i-start (the fixnum (* tid rows-per)))
                       (i-end   (the fixnum (min m (+ i-start rows-per)))))
                  (declare (type fixnum i-start i-end))
                  (when (< i-start i-end)
                    (push (sb-thread:make-thread
                           (lambda ()
                             (funcall rows-fn a-data b-data c-data k n
                                      a-off b-off c-off i-start i-end)))
                          threads))))
       (dolist (th threads)
         (sb-thread:join-thread th))
       nil))))

;;; ------------------------------------------------------------------
;;; 行范围内核 1: double-float NN
;;; ------------------------------------------------------------------
(declaim (inline %matmul-df-rows))
(defun %matmul-df-rows (a-data b-data c-data k n
                        a-off b-off c-off i-start i-end)
  "C[i-start..i-end, :] += A[i-start..i-end, :] * B[:, :]。
   A: (·,k) 行主序，B: (k,n) 行主序，C: (·,n) 行主序。"
  (declare (type (simple-array double-float (*)) a-data b-data c-data)
           (type fixnum k n a-off b-off c-off i-start i-end)
           (optimize (speed 3) (safety 0) (debug 0) (compilation-speed 0)))
  (let ((n-main (the fixnum (logand n -4))))
    (declare (type fixnum n-main))
    (loop for i of-type fixnum from i-start below i-end do
      (let ((a-row (the fixnum (+ a-off (the fixnum (* i k)))))
            (c-row (the fixnum (+ c-off (the fixnum (* i n))))))
        (declare (type fixnum a-row c-row))
        (loop for l of-type fixnum from 0 below k do
          (let ((a-val (aref a-data (the fixnum (+ a-row l))))
                (b-ptr (the fixnum (+ b-off (the fixnum (* l n)))))
                (c-ptr c-row))
            (declare (type double-float a-val)
                     (type fixnum b-ptr c-ptr))
            (loop for j of-type fixnum from 0 below n-main by 4 do
              (incf (aref c-data c-ptr)
                    (the double-float (* a-val (aref b-data b-ptr))))
              (incf (aref c-data (the fixnum (1+ c-ptr)))
                    (the double-float (* a-val (aref b-data (the fixnum (1+ b-ptr))))))
              (incf (aref c-data (the fixnum (+ c-ptr 2)))
                    (the double-float (* a-val (aref b-data (the fixnum (+ b-ptr 2))))))
              (incf (aref c-data (the fixnum (+ c-ptr 3)))
                    (the double-float (* a-val (aref b-data (the fixnum (+ b-ptr 3))))))
              (incf c-ptr 4)
              (incf b-ptr 4))
            (loop for j of-type fixnum from n-main below n do
              (incf (aref c-data c-ptr)
                    (the double-float (* a-val (aref b-data b-ptr))))
              (incf c-ptr)
              (incf b-ptr))))))))

(declaim (inline %matmul-df-fast-kernel))
(defun %matmul-df-fast-kernel (a-data b-data c-data m k n a-off b-off c-off)
  (call-parallel-gemm #'%matmul-df-rows
                      a-data b-data c-data m k n a-off b-off c-off))

;;; ------------------------------------------------------------------
;;; 行范围内核 2: single-float NN
;;; ------------------------------------------------------------------
(declaim (inline %matmul-sf-rows))
(defun %matmul-sf-rows (a-data b-data c-data k n
                        a-off b-off c-off i-start i-end)
  (declare (type (simple-array single-float (*)) a-data b-data c-data)
           (type fixnum k n a-off b-off c-off i-start i-end)
           (optimize (speed 3) (safety 0) (debug 0) (compilation-speed 0)))
  (let ((n-main (the fixnum (logand n -4))))
    (declare (type fixnum n-main))
    (loop for i of-type fixnum from i-start below i-end do
      (let ((a-row (the fixnum (+ a-off (the fixnum (* i k)))))
            (c-row (the fixnum (+ c-off (the fixnum (* i n))))))
        (declare (type fixnum a-row c-row))
        (loop for l of-type fixnum from 0 below k do
          (let ((a-val (aref a-data (the fixnum (+ a-row l))))
                (b-ptr (the fixnum (+ b-off (the fixnum (* l n)))))
                (c-ptr c-row))
            (declare (type single-float a-val)
                     (type fixnum b-ptr c-ptr))
            (loop for j of-type fixnum from 0 below n-main by 4 do
              (incf (aref c-data c-ptr)
                    (the single-float (* a-val (aref b-data b-ptr))))
              (incf (aref c-data (the fixnum (1+ c-ptr)))
                    (the single-float (* a-val (aref b-data (the fixnum (1+ b-ptr))))))
              (incf (aref c-data (the fixnum (+ c-ptr 2)))
                    (the single-float (* a-val (aref b-data (the fixnum (+ b-ptr 2))))))
              (incf (aref c-data (the fixnum (+ c-ptr 3)))
                    (the single-float (* a-val (aref b-data (the fixnum (+ b-ptr 3))))))
              (incf c-ptr 4)
              (incf b-ptr 4))
            (loop for j of-type fixnum from n-main below n do
              (incf (aref c-data c-ptr)
                    (the single-float (* a-val (aref b-data b-ptr))))
              (incf c-ptr)
              (incf b-ptr))))))))

(declaim (inline %matmul-sf-fast-kernel))
(defun %matmul-sf-fast-kernel (a-data b-data c-data m k n a-off b-off c-off)
  (call-parallel-gemm #'%matmul-sf-rows
                      a-data b-data c-data m k n a-off b-off c-off))

;;; ------------------------------------------------------------------
;;; 行范围内核 3: double-float NT (B 已转置，B: n×k 行主序)
;;; ------------------------------------------------------------------
(declaim (inline %matmul-df-nt-rows))
(defun %matmul-df-nt-rows (a-data b-data c-data k n
                           a-off b-off c-off i-start i-end)
  "C[i,j] = sum_l A[i,l] * B[j,l]。
   A: (·,k) 行主序；B: (n,k) 行主序；C: (·,n) 行主序。"
  (declare (type (simple-array double-float (*)) a-data b-data c-data)
           (type fixnum k n a-off b-off c-off i-start i-end)
           (optimize (speed 3) (safety 0) (debug 0) (compilation-speed 0)))
  (let ((k-main (the fixnum (logand k -4))))
    (declare (type fixnum k-main))
    (loop for i of-type fixnum from i-start below i-end do
      (let ((a-row (the fixnum (+ a-off (the fixnum (* i k)))))
            (c-row (the fixnum (+ c-off (the fixnum (* i n))))))
        (declare (type fixnum a-row c-row))
        (loop for j of-type fixnum from 0 below n do
          (let ((a-ptr a-row)
                (b-ptr (the fixnum (+ b-off (the fixnum (* j k)))))
                (acc0 0.0d0) (acc1 0.0d0) (acc2 0.0d0) (acc3 0.0d0))
            (declare (type fixnum a-ptr b-ptr)
                     (type double-float acc0 acc1 acc2 acc3))
            (loop for l of-type fixnum from 0 below k-main by 4 do
              (incf acc0 (* (aref a-data a-ptr)
                            (aref b-data b-ptr)))
              (incf acc1 (* (aref a-data (the fixnum (1+ a-ptr)))
                            (aref b-data (the fixnum (1+ b-ptr)))))
              (incf acc2 (* (aref a-data (the fixnum (+ a-ptr 2)))
                            (aref b-data (the fixnum (+ b-ptr 2)))))
              (incf acc3 (* (aref a-data (the fixnum (+ a-ptr 3)))
                            (aref b-data (the fixnum (+ b-ptr 3)))))
              (incf a-ptr 4)
              (incf b-ptr 4))
            (loop for l of-type fixnum from k-main below k do
              (incf acc0 (* (aref a-data a-ptr) (aref b-data b-ptr)))
              (incf a-ptr)
              (incf b-ptr))
	    (incf (aref c-data (the fixnum (+ c-row j)))
                  (+ acc0 acc1 acc2 acc3))))))))

(declaim (inline %matmul-df-nt-fast-kernel))
(defun %matmul-df-nt-fast-kernel (a-data b-data c-data m k n a-off b-off c-off)
  (call-parallel-gemm #'%matmul-df-nt-rows
                      a-data b-data c-data m k n a-off b-off c-off))

;;; ------------------------------------------------------------------
;;; 行范围内核 4: single-float NT
;;; ------------------------------------------------------------------
(declaim (inline %matmul-sf-nt-rows))
(defun %matmul-sf-nt-rows (a-data b-data c-data k n
                           a-off b-off c-off i-start i-end)
  (declare (type (simple-array single-float (*)) a-data b-data c-data)
           (type fixnum k n a-off b-off c-off i-start i-end)
           (optimize (speed 3) (safety 0) (debug 0) (compilation-speed 0)))
  (let ((k-main (the fixnum (logand k -4))))
    (declare (type fixnum k-main))
    (loop for i of-type fixnum from i-start below i-end do
      (let ((a-row (the fixnum (+ a-off (the fixnum (* i k)))))
            (c-row (the fixnum (+ c-off (the fixnum (* i n))))))
        (declare (type fixnum a-row c-row))
        (loop for j of-type fixnum from 0 below n do
          (let ((a-ptr a-row)
                (b-ptr (the fixnum (+ b-off (the fixnum (* j k)))))
                (acc0 0.0f0) (acc1 0.0f0) (acc2 0.0f0) (acc3 0.0f0))
            (declare (type fixnum a-ptr b-ptr)
                     (type single-float acc0 acc1 acc2 acc3))
            (loop for l of-type fixnum from 0 below k-main by 4 do
              (incf acc0 (* (aref a-data a-ptr)
                            (aref b-data b-ptr)))
              (incf acc1 (* (aref a-data (the fixnum (1+ a-ptr)))
                            (aref b-data (the fixnum (1+ b-ptr)))))
              (incf acc2 (* (aref a-data (the fixnum (+ a-ptr 2)))
                            (aref b-data (the fixnum (+ b-ptr 2)))))
              (incf acc3 (* (aref a-data (the fixnum (+ a-ptr 3)))
                            (aref b-data (the fixnum (+ b-ptr 3)))))
              (incf a-ptr 4)
              (incf b-ptr 4))
            (loop for l of-type fixnum from k-main below k do
              (incf acc0 (* (aref a-data a-ptr) (aref b-data b-ptr)))
              (incf a-ptr)
              (incf b-ptr))
	    (incf (aref c-data (the fixnum (+ c-row j)))
                  (+ acc0 acc1 acc2 acc3))))))))

(declaim (inline %matmul-sf-nt-fast-kernel))
(defun %matmul-sf-nt-fast-kernel (a-data b-data c-data m k n a-off b-off c-off)
  (call-parallel-gemm #'%matmul-sf-nt-rows
                      a-data b-data c-data m k n a-off b-off c-off))

;;; ------------------------------------------------------------------
;;; 行范围内核 5: int64
;;; ------------------------------------------------------------------
(declaim (inline %matmul-i64-rows))
(defun %matmul-i64-rows (a-data b-data c-data k n
                         a-off b-off c-off i-start i-end)
  "C[i-start..i-end, :] += A[i-start..i-end, :] * B[:, :] (int64)。"
  (declare (type (simple-array (signed-byte 64) (*)) a-data b-data c-data)
           (type fixnum k n a-off b-off c-off i-start i-end)
           (optimize (speed 3) (safety 0) (debug 0) (compilation-speed 0)))
  (let ((n-main (the fixnum (logand n -4))))
    (declare (type fixnum n-main))
    (loop for i of-type fixnum from i-start below i-end do
      (let ((a-row (the fixnum (+ a-off (the fixnum (* i k)))))
            (c-row (the fixnum (+ c-off (the fixnum (* i n))))))
        (declare (type fixnum a-row c-row))
        (loop for l of-type fixnum from 0 below k do
          (let ((a-val (aref a-data (the fixnum (+ a-row l))))
                (b-ptr (the fixnum (+ b-off (the fixnum (* l n)))))
                (c-ptr c-row))
            (declare (type (signed-byte 64) a-val)
                     (type fixnum b-ptr c-ptr))
            (loop for j of-type fixnum from 0 below n-main by 4 do
              (incf (aref c-data c-ptr)
                    (the (signed-byte 64)
                         (* a-val (the (signed-byte 64) (aref b-data b-ptr)))))
              (incf (aref c-data (the fixnum (1+ c-ptr)))
                    (the (signed-byte 64)
                         (* a-val (the (signed-byte 64) (aref b-data (the fixnum (1+ b-ptr)))))))
              (incf (aref c-data (the fixnum (+ c-ptr 2)))
                    (the (signed-byte 64)
                         (* a-val (the (signed-byte 64) (aref b-data (the fixnum (+ b-ptr 2)))))))
              (incf (aref c-data (the fixnum (+ c-ptr 3)))
                    (the (signed-byte 64)
                         (* a-val (the (signed-byte 64) (aref b-data (the fixnum (+ b-ptr 3)))))))
              (incf c-ptr 4)
              (incf b-ptr 4))
            (loop for j of-type fixnum from n-main below n do
              (incf (aref c-data c-ptr)
                    (the (signed-byte 64)
                         (* a-val (the (signed-byte 64) (aref b-data b-ptr)))))
              (incf c-ptr)
              (incf b-ptr))))))))

(declaim (inline %matmul-i64-fast-kernel))
(defun %matmul-i64-fast-kernel (a-data b-data c-data m k n a-off b-off c-off)
  (call-parallel-gemm #'%matmul-i64-rows
                      a-data b-data c-data m k n a-off b-off c-off))

;;; ------------------------------------------------------------------
;;; 行范围内核 5: int32
;;; ------------------------------------------------------------------
(declaim (inline %matmul-i32-rows))
(defun %matmul-i32-rows (a-data b-data c-data k n
                         a-off b-off c-off i-start i-end)
  "C[i-start..i-end, :] += A[i-start..i-end, :] * B[:, :] (int32)。"
  (declare (type (simple-array (signed-byte 32) (*)) a-data b-data c-data)
           (type fixnum k n a-off b-off c-off i-start i-end)
           (optimize (speed 3) (safety 0) (debug 0) (compilation-speed 0)))
  (let ((n-main (the fixnum (logand n -4))))
    (declare (type fixnum n-main))
    (loop for i of-type fixnum from i-start below i-end do
      (let ((a-row (the fixnum (+ a-off (the fixnum (* i k)))))
            (c-row (the fixnum (+ c-off (the fixnum (* i n))))))
        (declare (type fixnum a-row c-row))
        (loop for l of-type fixnum from 0 below k do
          (let ((a-val (aref a-data (the fixnum (+ a-row l))))
                (b-ptr (the fixnum (+ b-off (the fixnum (* l n)))))
                (c-ptr c-row))
            (declare (type (signed-byte 32) a-val)
                     (type fixnum b-ptr c-ptr))
            (loop for j of-type fixnum from 0 below n-main by 4 do
              (incf (aref c-data c-ptr)
                    (the (signed-byte 32)
                         (* a-val (the (signed-byte 32) (aref b-data b-ptr)))))
              (incf (aref c-data (the fixnum (1+ c-ptr)))
                    (the (signed-byte 32)
                         (* a-val (the (signed-byte 32) (aref b-data (the fixnum (1+ b-ptr)))))))
              (incf (aref c-data (the fixnum (+ c-ptr 2)))
                    (the (signed-byte 32)
                         (* a-val (the (signed-byte 32) (aref b-data (the fixnum (+ b-ptr 2)))))))
              (incf (aref c-data (the fixnum (+ c-ptr 3)))
                    (the (signed-byte 32)
                         (* a-val (the (signed-byte 32) (aref b-data (the fixnum (+ b-ptr 3)))))))
              (incf c-ptr 4)
              (incf b-ptr 4))
            (loop for j of-type fixnum from n-main below n do
              (incf (aref c-data c-ptr)
                    (the (signed-byte 32)
                         (* a-val (the (signed-byte 32) (aref b-data b-ptr)))))
              (incf c-ptr)
              (incf b-ptr))))))))

(declaim (inline %matmul-i32-fast-kernel))
(defun %matmul-i32-fast-kernel (a-data b-data c-data m k n a-off b-off c-off)
  (call-parallel-gemm #'%matmul-i32-rows
                      a-data b-data c-data m k n a-off b-off c-off))

;;; ==================================================================
;;; 通用 strided GEMM —— einsum 批量通道的内层（§16.26）
;;; ==================================================================
;;; 语义（与 einsum 的 incf 累加严格一致）：
;;;
;;;   C[off-c + i·so-i + k·so-k] += Σ_j A[off-a + i·sa-i + j·sa-j]
;;;                                       · B[off-b + j·sb-j + k·sb-k]
;;;
;;; 三个轴的角色由 einsum 的标签分析固定（i = A 的行、j = 收缩轴、
;;; k = B 的列），但**存储布局**任意，于是有四种组合。这里把它们归一化成
;;; 「连续 C += 连续 A·连续 B」，再进分块内核（SIMD 优先）：
;;;
;;;   ① 全连续（NN）       → 直接进内核（最快）。
;;;   ② B 按 (n,k) 存（NT）→ 先转成 (k,n) 连续缓冲再走 ①。
;;;      代价 O(n·k) 拷贝换 O(m·n·k) 的 SIMD 计算，m ≥ 2 就回本；
;;;      原实现走标量 NT 内核，512² 实测慢 12.2×（单线程）/ 1.7×（32 线程）。
;;;   ③ C 非行主序        → 先算到连续临时 C 再 scatter 累加。
;;;      不能直接让内层循环沿 so-k 走 —— 那会让最内层跨 cache line，
;;;      512² 的 "ij,jk->ki" 实测比 ① 慢 19.6×（609 ms vs 31 ms，单线程）。
;;;   ④ 其余（A 非行连续）→ 通用三重循环兜底，语义与改动前逐位一致。
;;;
;;; 为什么不写在 einsum-execute 的 macrolet 里（踩过的坑）：
;;;   einsum-execute 已有 4 类型 × 2 内核的宏展开，再塞进布局归一化后
;;;   函数体膨胀，SBCL 对**兜底路径**（一行没改）的代码生成退化 ~25%
;;;   （512² 的 "ij,jk->ki" 从 711~730 ms 变成 934~971 ms，重复 3 次稳定）。
;;;   这里每个分支只留一行独立函数调用，展开体积回到原来的量级。
;;;
;;; 数值一致性：四条路最终都调用同一个 %MATMUL-*-FAST-KERNEL 家族（或
;;; 同一 SIMD 内核）、分块顺序相同，因此与 VT-MATMUL **逐位相同**；
;;; 与旧的标量三重循环相比只差 FMA 少一次舍入（相对差 ~1e-12，更准）。

(defmacro define-einsum-gemm (fn-name strided-name ltype elem-kind kernel)
  "为 LTYPE 生成 einsum 的 strided GEMM 入口 FN-NAME 与兜底 STRIDED-NAME。

   STRIDED-NAME 就是原来的标量三重循环（逐位不变）；FN-NAME 在其上做
   布局归一化，能归一化时走 ELEM-KIND 对应的 SIMD / KERNEL 快路径。"
  ;; 临时缓冲的初值必须与元素类型同型：写 `:initial-element 0` 会让 SBCL
  ;; 报 "Constant 0 conflicts with its asserted type DOUBLE-FLOAT"（整数 0
  ;; 与浮点数组元素类型冲突），并让 COMPILE-FILE 返回 failure-p=T。
  ;;
  ;; 而且必须用 (FILL buf ZERO) 而**不是** :INITIAL-ELEMENT：对 2 MB 级别的
  ;; 大对象，SBCL 的 :INITIAL-ELEMENT 0.0d0 会走 calloc/零页（不真正写内存），
  ;; 于是首次写入由 **每个工作线程各自触发 page fault** —— 31 线程并发抢
  ;; mmap_lock，512² 实测「分配 + GEMM」从 47 ms 恶化到 111 ms。
  ;; FILL 走 memset 真写入，主线程一次 touch 完，代价 0.5 ms。
  (let ((zero (case elem-kind
                (:float64 ''0.0d0)
                (:float32 ''0.0f0)
                (t 0))))
    `(progn
     ;; ---- 兜底：任意步长的三重循环（= 改动前 einsum 的「兼容回退」）----
     (declaim (inline ,strided-name))
     (defun ,strided-name (da db dc m k n a-off b-off c-off
                           sa-i sa-j sb-j sb-k so-i so-k)
       (declare (type (simple-array ,ltype (*)) da db dc)
                (type fixnum m k n a-off b-off c-off
                      sa-i sa-j sb-j sb-k so-i so-k)
                (optimize (speed 3) (safety 0) (debug 0)))
       (loop for i-idx fixnum from 0 below m do
         (let ((a-row (the fixnum (+ a-off (the fixnum (* i-idx sa-i)))))
               (c-row (the fixnum (+ c-off (the fixnum (* i-idx so-i))))))
           (declare (type fixnum a-row c-row))
           (loop for j-idx fixnum from 0 below k do
             (let ((a-val (aref da (the fixnum
                                    (+ a-row (the fixnum (* j-idx sa-j))))))
                   (b-ptr (the fixnum (+ b-off (the fixnum (* j-idx sb-j)))))
                   (c-ptr c-row))
               (declare (type ,ltype a-val) (type fixnum b-ptr c-ptr))
               (loop for k-idx fixnum from 0 below n do
                 (incf (aref dc c-ptr)
                       (the ,ltype (* a-val (aref db b-ptr))))
                 (incf b-ptr sb-k)
                 (incf c-ptr so-k)))))))
     ;; ---- 入口：布局归一化 ----
     (defun ,fn-name (da db dc m k n a-off b-off c-off
                      sa-i sa-j sb-j sb-k so-i so-k)
       (declare (type (simple-array ,ltype (*)) da db dc)
                (type fixnum m k n a-off b-off c-off
                      sa-i sa-j sb-j sb-k so-i so-k)
                (optimize (speed 3) (safety 0) (debug 0)))
       (labels ((core (ba bb bc boff coff)
                  ;; 连续 C += 连续 A·连续 B。SIMD 分块内核优先，它自带
                  ;; 「尺寸低于阈值 / 全局关闭 → NIL」的判据。
                  (declare (type (simple-array ,ltype (*)) ba bb bc)
                           (type fixnum boff coff))
                  (unless (and *simd-gemm-raw-fn*
                               (funcall *simd-gemm-raw-fn* ba bb bc m k n
                                        a-off boff coff ,elem-kind))
                    (,kernel ba bb bc m k n a-off boff coff)))
                (b-as-row-major (bb boff)
                  ;; B 按 (n,k) 存（NT）→ 转成 (k,n) 连续缓冲。
                  ;; **必须分块转置**：朴素双重循环里 BB 的内层步长是 SB-K
                  ;; （NT 时 = k，即整行），每步跨一行 → 512² 时几乎每次
                  ;; 都是 cache miss，光这一步就把 NT 拖慢 1.8×。
                  ;; 64×64 分块后块内两侧都落在 L2。
                  (declare (type (simple-array ,ltype (*)) bb)
                           (type fixnum boff))
                  (let ((bt (make-array (the fixnum (* k n))
                                        :element-type ',ltype)))
                    (declare (type (simple-array ,ltype (*)) bt))
                    (fill bt ,zero)
                    (loop for j0 fixnum from 0 below k by 64 do
                      (let ((j1 (the fixnum (min k (the fixnum (+ j0 64))))))
                        (declare (type fixnum j1))
                        (loop for k0 fixnum from 0 below n by 64 do
                          (let ((k1 (the fixnum
                                         (min n (the fixnum (+ k0 64))))))
                            (declare (type fixnum k1))
                            (loop for jj fixnum from j0 below j1 do
                              (let ((brow (the fixnum
                                               (+ boff (the fixnum
                                                        (* jj sb-j)))))
                                    (trow (the fixnum (* jj n))))
                                (declare (type fixnum brow trow))
                                (loop for kk fixnum from k0 below k1 do
                                  (setf (aref bt (the fixnum (+ trow kk)))
                                        (aref bb
                                              (the fixnum
                                               (+ brow (the fixnum
                                                        (* kk sb-k)))))))))))))
                    bt))
                (scatter (tmp)
                  ;; 把连续临时 C 按 out 的实际步长累加回 DC。
                  ;; 同样必须**分块 + 让写侧连续**：`"ij,jk->ki"` 的
                  ;; SO-K = m（跨整行），若按「i 外层 / kk 内层」写，
                  ;; 内层每步换一条 cache line，512² 实测 scatter 单项
                  ;; 就要 50ms 以上（GEMM 本体只要 41ms）。
                  ;; 这里改成「kk 外层 / i 内层」（写侧沿 SO-I，通常 = 1
                  ;; 即连续），并 64×64 分块让读侧也留在 L2。
                  (declare (type (simple-array ,ltype (*)) tmp))
                  (loop for k0 fixnum from 0 below n by 64 do
                    (let ((k1 (the fixnum (min n (the fixnum (+ k0 64))))))
                      (declare (type fixnum k1))
                      (loop for i0 fixnum from 0 below m by 64 do
                        (let ((i1 (the fixnum
                                       (min m (the fixnum (+ i0 64))))))
                          (declare (type fixnum i1))
                          (loop for kk fixnum from k0 below k1 do
                            (let ((crow (the fixnum
                                             (+ c-off (the fixnum
                                                       (* kk so-k))))))
                              (declare (type fixnum crow))
                              (loop for i fixnum from i0 below i1 do
                                (incf (aref dc
                                            (the fixnum
                                             (+ crow (the fixnum
                                                      (* i so-i)))))
                                      (aref tmp
                                            (the fixnum
                                             (+ (the fixnum (* i n))
                                                kk)))))))))))))
         (cond
           ;; ④ A 非行连续：没有可复用的内核，直接兜底
           ((not (and (= sa-j 1) (= sa-i k)))
            (,strided-name da db dc m k n a-off b-off c-off
                           sa-i sa-j sb-j sb-k so-i so-k))
           ;; ③ 的退化版：C 非行主序但 m·n 很小，临时缓冲分配不划算
           ((and (not (and (= so-k 1) (= so-i n)))
                 (< (the fixnum (* m n)) 4096))
            (,strided-name da db dc m k n a-off b-off c-off
                           sa-i sa-j sb-j sb-k so-i so-k))
           (t
            (let ((bt nil) (boff b-off))
              (declare (type (or null (simple-array ,ltype (*))) bt)
                       (type fixnum boff))
              (cond
                ;; ①/③ B 已是行主序
                ((and (= sb-k 1) (= sb-j n)))
                ;; ②/③ B 是 NT 布局 → 转成行主序
                ((and (= sb-j 1) (= sb-k k))
                 (setf bt (b-as-row-major db b-off) boff 0))
                ;; 其它（B 也是 strided）：兜底
                (t
                 (return-from ,fn-name
                   (,strided-name da db dc m k n a-off b-off c-off
                                  sa-i sa-j sb-j sb-k so-i so-k))))
              (let ((bb (if bt bt db)))
                (declare (type (simple-array ,ltype (*)) bb))
                (if (and (= so-k 1) (= so-i n))
                    (core da bb dc boff c-off)
                    (let ((tmp (make-array (the fixnum (* m n))
                                           :element-type ',ltype)))
                      (declare (type (simple-array ,ltype (*)) tmp))
                      (fill tmp ,zero)
                      (core da bb tmp boff 0)
                      (scatter tmp))))))))))))

(define-einsum-gemm %einsum-gemm-f64 %einsum-gemm-strided-f64
    double-float :float64 %matmul-df-fast-kernel)
(define-einsum-gemm %einsum-gemm-f32 %einsum-gemm-strided-f32
    single-float :float32 %matmul-sf-fast-kernel)
(define-einsum-gemm %einsum-gemm-i64 %einsum-gemm-strided-i64
    (signed-byte 64) :int64 %matmul-i64-fast-kernel)
(define-einsum-gemm %einsum-gemm-i32 %einsum-gemm-strided-i32
    (signed-byte 32) :int32 %matmul-i32-fast-kernel)


(defun einsum-execute
    (all-labels-vec label-dims-vec output-subs input-subs vts &key out)
  (declare (type simple-vector all-labels-vec)
           (type (simple-array fixnum (*)) label-dims-vec)
           (type list output-subs input-subs vts))
  (with-float-safe
    (when out
      ;; 别名保护（§4.3）：通用累加内核用 `incf`，要求输出缓冲区初始为 0
      ;; （见下方 `(vt-fill output 0)`）。若 out 与某输入**物理区间重叠**，
      ;; 清零会先破坏输入 → 结果错误。故先快照重叠的输入。
      ;; v0.3.6 修复：原实现只比较 `(vt-data out)` 与 `(vt-data tns)` 指针，
      ;; 漏掉「同一 data 但 offset/strides 不同」的重叠视图（如 a = base[0:64]、
      ;; out = base[32:96]），改用物理区间重叠判定。
      (setf vts
            (mapcar (lambda (tns)
                      (if (%vt-views-overlap-p out tns)
                          (vt-copy tns)
                          tns))
                    vts)))
    (let* ((rank (length all-labels-vec))
           (n-vts (length vts))
           (dims-vec (make-array rank :element-type 'fixnum)))
      
      (loop for i fixnum from 0 below rank
            for label fixnum = (svref all-labels-vec i) do
              (setf (aref dims-vec i)
		    (aref label-dims-vec
			  (if (< label 0) (+ 256 label) label))))
      
      (let* ((strides-mat (make-array (list n-vts rank)
                                      :element-type 'fixnum))
             (in-data-vec (make-array n-vts))
             (in-offsets-vec (make-array n-vts :element-type 'fixnum)))
        
        (loop for sub list in input-subs
              for tns in vts
              for t-idx fixnum from 0 do
		(let ((cont-tns (vt-contiguous tns)))
		  (let ((phys-strides (vt-strides cont-tns))
			(phys-shape (vt-shape cont-tns)))
		    (setf (aref in-data-vec t-idx) (vt-data cont-tns))
		    (setf (aref in-offsets-vec t-idx) (vt-offset cont-tns))
		    (let ((p-strides-arr (coerce phys-strides 'simple-vector))
			  (p-shape-arr (coerce phys-shape 'simple-vector)))
                      (loop for lbl fixnum across all-labels-vec
			    for logical-idx fixnum from 0 do
			      (let ((p-stride-sum 0))
				(declare (fixnum p-stride-sum))
				(loop for pos fixnum from 0 below (length sub)
				      for sub-lbl fixnum = (nth pos sub)
				      when (eql sub-lbl lbl) do
					(let ((p-dim (the fixnum (svref p-shape-arr pos)))
					      (p-stride (the fixnum
							     (svref p-strides-arr pos))))
					  (incf p-stride-sum
						(if (and (= p-dim 1)
							 (> (aref dims-vec logical-idx) 1))
						    0 p-stride)))) 
				(setf (aref strides-mat t-idx logical-idx)
				      p-stride-sum)
				))))))
        
        ;; 1. 扩展类型判断：支持 4 种数据类型
        (let* ((all-f64-p (every #'(lambda (vt) (eq (vt-dtype vt) :float64)) vts))
               (all-f32-p (every #'(lambda (vt) (eq (vt-dtype vt) :float32)) vts))
               (all-i64-p (every #'(lambda (vt) (eq (vt-dtype vt) :int64)) vts))
               (all-i32-p (every #'(lambda (vt) (eq (vt-dtype vt) :int32)) vts))
               ;; 复数：缺这两支时 out-dtype 会落到 :float64，复数结果写进
               ;; 实数缓冲区 → 在 %DOUBLE-FLOAT 处报错（原「不支持」的真相）
               (all-c128-p (every #'(lambda (vt) (eq (vt-dtype vt) :complex128)) vts))
               (all-c64-p (every #'(lambda (vt) (eq (vt-dtype vt) :complex64)) vts))
               (out-dtype (cond (all-c128-p :complex128)
                                (all-c64-p :complex64)
                                (all-f64-p :float64)
                                (all-f32-p :float32)
                                (all-i64-p :int64)
                                (all-i32-p :int32)
                                (t :float64)))
               (out-shape
		 (loop for lbl fixnum across all-labels-vec
                       when (member lbl output-subs)
                         collect (the fixnum (aref label-dims-vec
                                                   (if (< lbl 0)
                                                       (+ 256 lbl)
                                                       lbl))))))
          (declare (type boolean all-f64-p all-f32-p all-i64-p all-i32-p
                                 all-c128-p all-c64-p)
		   (list out-shape))
          
          (let* ((output (or out (vt-zeros out-shape :dtype out-dtype))))
            (when out
              ;; out 硬契约（设计约定）：违反必报错，不使用 assert
              ;; （assert 可能被 (safety 0) 编译剔除，硬契约必须保证触发）。
              (unless (equal (vt-shape output) out-shape)
                (error "vt-einsum: :out 形状 ~a 与逻辑结果形状 ~a 不匹配"
                       (vt-shape output) out-shape))
              (unless (eq (vt-dtype output) out-dtype)
                (error "vt-einsum: :out dtype ~a 与结果 dtype ~a 不匹配"
                       (vt-dtype output) out-dtype))
              ;; 通用累加内核依赖零初始值（见下方 BMM fallback 的 incf / 累加循环）。
              (vt-fill output 0))
            
            (let* ((out-offset (vt-offset output))
                   (out-data (vt-data output))
                   (out-strides-vec
                     (make-array rank :element-type 'fixnum)))
              (declare (type fixnum out-offset))
              (if out      ; out 提供时使用其实际 strides，避免非连续 :out 写坏内存
                  (let ((actual-strides (coerce (vt-strides output) 'simple-vector)))
                    (loop for i fixnum from 0 below rank
                          for lbl = (svref all-labels-vec i)
                          for pos = (position lbl output-subs :test #'eql)
                          do (setf (aref out-strides-vec i)
                                   (if pos
                                       (the fixnum (svref actual-strides pos))
                                       0))))
		  (let ((acc 1))
                    (declare (fixnum acc))
                    (loop for i fixnum from (1- rank) downto 0
			  for dim = (aref dims-vec i) do
			    (if (member (svref all-labels-vec i) output-subs)
				(progn
				  (setf (aref out-strides-vec i) acc)
				  (setf acc (the fixnum (* acc dim))))
				(setf (aref out-strides-vec i) 0)))))
              
              (when (or (zerop (vt-size output))
                        (some #'zerop out-shape)
                        (some #'zerop dims-vec))
                (return-from einsum-execute output))
	      ;; ★ P1-1: 纯逐元素 einsum 路由到 vt-map
              ;; 形如 "ij,ij->ij" / "i,i->i" / "...ij,...ij->...ij"
              ;; 复数（§16.21）：走 VT-* 的复数特化元素内核。**不要**用
              ;; VT-MAP —— 后者对复数要走泛型 funcall，1M 元素实测
              ;; 138ms vs 8ms（17×）。实数维持原 VT-MAP 分支不动。
              (when (and (= n-vts 2)
                         (equal (first input-subs) (second input-subs))
                         (equal (first input-subs) output-subs)
			 (= (length (first input-subs))
			    (length (remove-duplicates (first input-subs) :test #'eql)))
                         (or all-f64-p all-f32-p all-i64-p all-i32-p
                             all-c128-p all-c64-p))
                (return-from einsum-execute
                  (if (or all-c128-p all-c64-p)
                      (vt-* (first vts) (second vts) :out output)
                      (vt-map #'* (first vts) (second vts) :out output))))
              ;; ★ P1-2b: 复数全收缩内积 "i,i->"（§16.21）
              ;; 与下面实数内核同条件，只是累加器为复数：交给复数层的
              ;; %CX-DOT-1D（不做共轭，与 numpy.dot / einsum 一致）。
              (when (and (= n-vts 2)
                         (or all-c128-p all-c64-p)
                         *cx-einsum-dot-fn*
                         (equal (first input-subs) (second input-subs))
                         (null output-subs)
                         (equal (vt-shape (first vts)) (vt-shape (second vts)))
			 (= (length (first input-subs))
			    (length (remove-duplicates (first input-subs) :test #'eql)))
                         (eq (vt-dtype (first vts)) (vt-dtype (second vts))))
                (let* ((sum (funcall *cx-einsum-dot-fn*
                                     (vt-contiguous (first vts))
                                     (vt-contiguous (second vts))))
                       (val (if (eq out-dtype :complex64)
                                (complex (coerce (realpart sum) 'single-float)
                                         (coerce (imagpart sum) 'single-float))
                                sum)))
                  (setf (aref (the (simple-array * (*)) (vt-data output))
                              (vt-offset output))
                        val)
                  (return-from einsum-execute output)))
              ;; ★ P1-2: 全收缩内积 "i,i->" / "ij,ij->" 专用内核
              ;; 条件：两输入形状完全相同、output 为空
              (when (and (= n-vts 2)
                         (equal (first input-subs) (second input-subs))
                         (null output-subs)
                         (equal (vt-shape (first vts)) (vt-shape (second vts)))
			 (= (length (first input-subs))
			    (length (remove-duplicates (first input-subs) :test #'eql)))
                         (or all-f64-p all-f32-p all-i64-p all-i32-p))
                (let* ((a-c (vt-contiguous (first vts)))
                       (b-c (vt-contiguous (second vts)))
                       (n (vt-size a-c))
                       (a-data (vt-data a-c))
                       (b-data (vt-data b-c))
                       (a-off (vt-offset a-c))
                       (b-off (vt-offset b-c))
                       (a-end (the fixnum (+ a-off n)))
                       (result
                         (cond
                           (all-f64-p
                            (let ((da (the (simple-array double-float (*)) a-data))
                                  (db (the (simple-array double-float (*)) b-data))
                                  (pa a-off) (pb b-off) (acc 0.0d0))
                              (declare (type double-float acc) (type fixnum pa pb a-end))
                              (loop while (< pa a-end) do
                                (incf acc (* (aref da pa) (aref db pb)))
                                (incf pa) (incf pb))
                              acc))
                           (all-f32-p
                            (let ((da (the (simple-array single-float (*)) a-data))
                                  (db (the (simple-array single-float (*)) b-data))
                                  (pa a-off) (pb b-off) (acc 0.0f0))
                              (declare (type single-float acc) (type fixnum pa pb a-end))
                              (loop while (< pa a-end) do
                                (incf acc (* (aref da pa) (aref db pb)))
                                (incf pa) (incf pb))
                              acc))
                           (all-i64-p
                            (let ((da (the (simple-array (signed-byte 64) (*)) a-data))
                                  (db (the (simple-array (signed-byte 64) (*)) b-data))
                                  (pa a-off) (pb b-off) (acc 0))
                              (declare (type (signed-byte 64) acc) (type fixnum pa pb a-end))
                              (loop while (< pa a-end) do
                                (incf acc (* (aref da pa) (aref db pb)))
                                (incf pa) (incf pb))
                              acc))
                           (all-i32-p
                            (let ((da (the (simple-array (signed-byte 32) (*)) a-data))
                                  (db (the (simple-array (signed-byte 32) (*)) b-data))
                                  (pa a-off) (pb b-off) (acc 0))
                              (declare (type (signed-byte 32) acc) (type fixnum pa pb a-end))
                              (loop while (< pa a-end) do
                                (incf acc (* (aref da pa) (aref db pb)))
                                (incf pa) (incf pb))
                              acc)))))
                  (setf (aref (the (simple-array * (*)) (vt-data output))
                              (vt-offset output))
                        (vt-cast result out-dtype))
                  (return-from einsum-execute output)))
              
              ;; 批量矩阵乘法 (BMM) 极速通道
              (when (and (= n-vts 2)
                         (>= (length output-subs) 2)
                         (or all-f64-p all-f32-p all-i64-p all-i32-p
                             ;; 复数（§16.21）：钩子由 complex.lisp 注册；
                             ;; 未注册时不进本通道，回落到通用累加（语义不变）。
                             (and *cx-einsum-gemm-fn*
                                  (or all-c128-p all-c64-p))))
                (let* ((s1 (first input-subs))
                       (s2 (second input-subs))
                       (shared (intersection s1 s2))
                       (j-label (car (set-difference shared output-subs)))
                       (i-label (car (set-difference s1 s2)))
                       (k-label (car (set-difference s2 s1)))
                       (batch-labels (intersection shared output-subs)))
                  (when (and j-label i-label k-label
                             (= (length (set-difference shared output-subs)) 1)
                             (= (length (set-difference s1 s2)) 1)
                             (= (length (set-difference s2 s1)) 1))
                    (let* ((pos-i (position i-label all-labels-vec :test #'eql))
                           (pos-j (position j-label all-labels-vec :test #'eql))
                           (pos-k (position k-label all-labels-vec :test #'eql))
                           (d-i (aref dims-vec pos-i))
                           (d-j (aref dims-vec pos-j))
                           (d-k (aref dims-vec pos-k))
                           (sa-i (aref strides-mat 0 pos-i))
                           (sa-j (aref strides-mat 0 pos-j))
                           (sb-j (aref strides-mat 1 pos-j))
                           (sb-k (aref strides-mat 1 pos-k))
                           (so-i (aref out-strides-vec pos-i))
                           (so-k (aref out-strides-vec pos-k))
                           (off-a-base (aref in-offsets-vec 0))
                           (off-b-base (aref in-offsets-vec 1))
                           (data-a (aref in-data-vec 0))
                           (data-b (aref in-data-vec 1)))
                      
                      ;; 4 种元素类型各自一行调用：布局归一化（NN / B 转置 /
                      ;; C 转置 / 兜底）全在 %EINSUM-GEMM-* 里（§16.26）。
                      ;; **不要**把那些判断搬回来展开：会让本函数体膨胀，
                      ;; 连带把兜底路径的代码生成拖慢 33%（见那里的注释）。
		      (macrolet ((gen-bmm-logic (lisp-type fn)
                                   `(,fn (the (simple-array ,lisp-type (*)) data-a)
                                         (the (simple-array ,lisp-type (*)) data-b)
                                         (the (simple-array ,lisp-type (*)) out-data)
                                         d-i d-j d-k off-a off-b off-c
                                         sa-i sa-j sb-j sb-k so-i so-k)))
                        
                        (labels
                            ((loop-batch (b-labels off-a off-b off-c)
                               (if (null b-labels)
				   (cond
                                     (all-f64-p (gen-bmm-logic double-float
                                                               %einsum-gemm-f64))
                                     (all-f32-p (gen-bmm-logic single-float
                                                               %einsum-gemm-f32))
                                     (all-i64-p (gen-bmm-logic (signed-byte 64)
                                                               %einsum-gemm-i64))
                                     (all-i32-p (gen-bmm-logic (signed-byte 32)
                                                               %einsum-gemm-i32))
                                     ;; 复数（§16.21）：内核在 complex.lisp，
                                     ;; 由 %CX-GEMM-STRIDED 按元素类型再分派，
                                     ;; 因此这里不需要 gen-bmm-logic 的类型展开。
                                     ((and *cx-einsum-gemm-fn*
                                           (or all-c128-p all-c64-p))
                                      (funcall *cx-einsum-gemm-fn*
                                               data-a data-b out-data
                                               d-i d-j d-k off-a off-b off-c
                                               sa-i sa-j sb-j sb-k so-i so-k))
                                     (t nil))
                                   
                                   (let* ((lbl (first b-labels))
                                          (pos-lbl (position lbl all-labels-vec :test #'eql))
                                          (dim (aref dims-vec pos-lbl))
                                          (stride-a (aref strides-mat 0 pos-lbl))
                                          (stride-b (aref strides-mat 1 pos-lbl))
                                          (stride-c (aref out-strides-vec pos-lbl)))
                                     (loop for i fixnum from 0 below dim do
                                       (loop-batch (rest b-labels)
                                                   (the fixnum (+ off-a (the fixnum (* i stride-a))))
                                                   (the fixnum (+ off-b (the fixnum (* i stride-b))))
                                                   (the fixnum (+ off-c (the fixnum (* i stride-c))))))))))
                          
                          (loop-batch batch-labels off-a-base off-b-base out-offset))
                        
                        (return-from einsum-execute output))))))
              
              ;; 通用 einsum 路径 (极简迭代状态机，零分配)
              (let ((cur-ptrs (make-array n-vts
                                          :element-type 'fixnum
                                          :initial-element 0)))
                (declare (type (simple-array fixnum (*)) cur-ptrs))
                (loop for k fixnum from 0 below n-vts do
                  (setf (aref cur-ptrs k) (aref in-offsets-vec k)))
                (let ((indices (make-array rank
                                           :element-type 'fixnum
                                           :initial-element 0))
                      (depth 0)
                      (out-ptr out-offset))
                  (declare (type (simple-array fixnum (*)) indices)
                           (type fixnum depth out-ptr))
                  (loop
                    (cond
                      ((= depth rank)
                       ;; 扩展为 4 种类型的强类型计算
                       (cond
                         (all-f64-p
                          (let ((product 1.0d0))
                            (declare (type double-float product))
                            (if (= n-vts 2)
                                (setf product
                                      (* (aref (the (simple-array double-float (*))
						    (aref in-data-vec 0))
					       (aref cur-ptrs 0))
                                         (aref (the (simple-array double-float (*))
						    (aref in-data-vec 1))
					       (aref cur-ptrs 1))))
                                (loop for k fixnum from 0 below n-vts do
                                  (setf product (* product (aref (the (simple-array double-float (*))
								      (aref in-data-vec k))
								 (aref cur-ptrs k))))))
                            (incf (the double-float
				       (aref (the (simple-array double-float (*)) out-data) out-ptr))
				  product)))
                         
                         (all-f32-p
                          (let ((product 1.0f0))
                            (declare (type single-float product))
                            (if (= n-vts 2)
                                (setf product
                                      (* (aref (the (simple-array single-float (*))
						    (aref in-data-vec 0))
					       (aref cur-ptrs 0))
                                         (aref (the (simple-array single-float (*))
						    (aref in-data-vec 1))
					       (aref cur-ptrs 1))))
                                (loop for k fixnum from 0 below n-vts do
                                  (setf product
					(* product (aref (the (simple-array single-float (*))
							      (aref in-data-vec k))
							 (aref cur-ptrs k))))))
                            (incf (the single-float
				       (aref (the (simple-array single-float (*))
						  out-data)
					     out-ptr))
				  product)))
                         
                         (all-i64-p
                          (let ((product 1))
                            (declare (type (signed-byte 64) product))
                            (if (= n-vts 2)
                                (setf product
                                      (* (the (signed-byte 64)
					      (aref (the (simple-array (signed-byte 64) (*))
							 (aref in-data-vec 0))
						    (aref cur-ptrs 0)))
                                         (the (signed-byte 64)
					      (aref (the (simple-array (signed-byte 64) (*))
							 (aref in-data-vec 1))
						    (aref cur-ptrs 1)))))
                                (loop for k fixnum from 0 below n-vts do
                                  (setf product
					(the (signed-byte 64)
					     (* product (the (signed-byte 64)
							     (aref (the (simple-array (signed-byte 64) (*))
									(aref in-data-vec k))
								   (aref cur-ptrs k))))))))
                            (incf (the (signed-byte 64)
				       (aref (the (simple-array (signed-byte 64) (*)) out-data)
					     out-ptr))
				  product)))
                         
                         (all-i32-p
                          (let ((product 1))
                            (declare (type (signed-byte 32) product))
                            (if (= n-vts 2)
                                (setf product
                                      (* (the (signed-byte 32)
					      (aref (the (simple-array (signed-byte 32) (*))
							 (aref in-data-vec 0))
						    (aref cur-ptrs 0)))
                                         (the (signed-byte 32)
					      (aref (the (simple-array (signed-byte 32) (*))
							 (aref in-data-vec 1))
						    (aref cur-ptrs 1)))))
                                (loop for k fixnum from 0 below n-vts do
                                  (setf product
					(the (signed-byte 32)
					     (* product
						(the (signed-byte 32)
						     (aref (the (simple-array (signed-byte 32) (*))
								(aref in-data-vec k))
							   (aref cur-ptrs k))))))))
                            (incf (the (signed-byte 32)
				       (aref (the (simple-array (signed-byte 32) (*))
						  out-data)
					     out-ptr))
				  product)))
                         
                         (t
                          (let ((product 1))
                            (if (= n-vts 2)
                                (setf product
                                      (* (aref (aref in-data-vec 0) (aref cur-ptrs 0))
                                         (aref (aref in-data-vec 1) (aref cur-ptrs 1))))
                                (loop for k fixnum from 0 below n-vts do
                                  (setf product
					(* product (aref (aref in-data-vec k) (aref cur-ptrs k))))))
                            (incf (aref out-data out-ptr) product))))
                       
                       (loop
                         (decf depth)
                         (when (< depth 0)
                           (return-from einsum-execute output))
                         (let ((d depth))
                           (if (< (aref indices d) (1- (aref dims-vec d)))
                               (progn
                                 (incf out-ptr (aref out-strides-vec d))
                                 (loop for k fixnum from 0 below n-vts do
                                   (incf (aref cur-ptrs k) (aref strides-mat k d)))
                                 (incf (aref indices d))
                                 (incf depth)
                                 (return))
                               (progn
                                 (decf out-ptr
				       (the fixnum (* (aref indices d) (aref out-strides-vec d))))
                                 (loop for k fixnum from 0 below n-vts do
                                   (decf (aref cur-ptrs k)
					 (the fixnum (* (aref indices d) (aref strides-mat k d)))))
                                 (setf (aref indices d) 0))))))
                      (t (incf depth)))
		    ))))))))))


(defun vt-einsum (subscripts &rest args)
  "高性能 einsum (爱因斯坦求和约定) 终极接口。
  流程：
  1. 参数提取与清洗 (parse-vt-op-args)。
  2. 类型推导与统一 (astype)。
  3. 下标解析与语义分析。
  4. 直接调用 einsum-execute (自动触发 Matmul 极速内核)。"
  (declare (optimize (speed 3) (safety 0)))
  (multiple-value-bind (tensors dtype-arg out-arg)
      (parse-vt-op-args args)    
    (let* ((clean-tensors (mapcar #'ensure-vt tensors))
           ;; 复数（complex64/complex128）也允许：类型无关的通用累加路径
           ;; 天然正确；三类快路径（逐元素/内积/GEMM）在复数层注册钩子后
           ;; 也一并对复数生效（§16.21）。钩子缺席时自动回落通用路径。
           (supported-types '(:float64 :float32 :int64 :int32
                              :complex64 :complex128)))
      ;; 校验显式 dtype
      (when (and dtype-arg (not (member dtype-arg supported-types)))
        (error "vt-einsum: 不支持的显式 dtype (~a)。允许: ~a。"
	       dtype-arg supported-types))
      ;; 校验 out 类型
      (when (and out-arg (not (member (vt-dtype out-arg) supported-types)))
        (error "vt-einsum: 不支持的 out 张量类型 (~a)。允许: ~a。"
	       (vt-dtype out-arg) supported-types))
      ;; 类型推导与统一 ===
      (let* ((final-dtype
               (cond
                 (dtype-arg dtype-arg)
                 (t (apply #'vt-promote-type
                           (mapcar #'vt-dtype clean-tensors)))))
             ;; 类型转换 (零拷贝优化)
             (cast-tensors 
               (if (every #'(lambda (vt) (eq (vt-dtype vt) final-dtype)) clean-tensors)
                   clean-tensors
                   (mapcar #'(lambda (vt) (vt-astype vt final-dtype)) clean-tensors))))        
        ;; :out dtype 契约（§4.2 H3「严格相等」）：结果 dtype 由「输入提升 +
        ;; 显式 :dtype」决定，**不由 out 决定**。einsum-execute 内部还会对
        ;; 形状（H2）做精确校验；此处提前给出带 "vt-einsum" 前缀的清晰报错。
        (when out-arg
          (unless (eq (vt-dtype out-arg) final-dtype)
            (error "vt-einsum: :out dtype ~a 与结果 dtype ~a 不匹配"
                   (vt-dtype out-arg) final-dtype)))
        (multiple-value-bind (raw-inputs raw-output explicit-p)
            (get-parsed-subscripts subscripts)          
          (unless (= (length raw-inputs) (length cast-tensors))
            (error "张量数量 ~a 与下标 ~a 不匹配" (length cast-tensors) raw-inputs))
          (multiple-value-bind (input-subs output-subs)
              (expand-ellipsis raw-inputs raw-output cast-tensors)
            (unless explicit-p (setf output-subs nil))            
            ;; 语义分析 (构建维度映射)
            (multiple-value-bind (all-labels label-dims output-subs-final)
                (analyze-einsum input-subs output-subs cast-tensors explicit-p)
              (einsum-execute 
               all-labels label-dims output-subs-final input-subs cast-tensors
               :out out-arg))))))))

(defun vt-matmul (a b &key dtype out)
  "矩阵乘法，兼容 1d 向量（对标 numpy 的 @ 运算符）。"
  (let ((ra (vt-order a))
        (rb (vt-order b)))
    (cond
      ;; 2d @ 2d → 2d（矩阵乘法）
      ((and (= ra 2) (= rb 2))
       (or (and *simd-matmul-2d-fn*
		(funcall *simd-matmul-2d-fn* a b dtype out))
	   (vt-einsum "ij,jk->ik" a b :dtype dtype :out out)))
      ;; 1d @ 1d → 标量（内积）
      ((and (= ra 1) (= rb 1))
       (vt-einsum "i,i->" a b :dtype dtype :out out))
      ;; 2d @ 1d → 1d（矩阵乘向量）
      ((and (= ra 2) (= rb 1))
       (vt-einsum "ij,j->i" a b :dtype dtype :out out))
      ;; 1d @ 2d → 1d（向量乘矩阵）
      ((and (= ra 1) (= rb 2))
       (vt-einsum "i,ij->j" a b :dtype dtype :out out))
      ;; (N≥2)d @ 1d → 末轴收缩后去掉尾维（numpy 1-D 提升语义）
      ((and (>= ra 2) (= rb 1))
       (let ((r (vt-einsum "...ij,j->...i" a b :dtype dtype)))
         (if out (progn (vt-copy-into out r) out) r)))
      ;; 1d @ (N≥2)d → 倒数第二轴收缩后去掉首维（numpy 1-D 提升语义）
      ((and (= ra 1) (>= rb 2))
       (let ((r (vt-einsum "j,...jk->...k" a b :dtype dtype)))
         (if out (progn (vt-copy-into out r) out) r)))
      ;; >2d @ >2d → 批量矩阵乘法
      (t (or (and *simd-batched-matmul-fn*
		  (funcall *simd-batched-matmul-fn* a b dtype out))
	     (vt-einsum "...ij,...jk->...ik" a b :dtype dtype :out out))))))

(defun vt-@ (vt1 vt2 &key dtype out)
  "矩阵乘法运算符别名（对标 numpy 的 @ 运算符）。等价 (vt-matmul vt1 vt2)。"
  (vt-matmul vt1 vt2 :dtype dtype :out out))

(defun vt-dot (a b &key dtype out) 
  "点积/内积，支持任意维度： 
   - 若 a,b 均为 1d → 向量内积，返回 0 维张量。 
   - 若 a 为 2d, b 为 1d → 矩阵乘向量，返回 1d 向量。 
   - 若 a 为 1d, b 为 2d → 向量乘矩阵，返回 1d 向量。 
   - 若 a,b 均为 2d → 矩阵乘法 a @ b。 
   - 若 a,b 秩均 ≥2 → 批量矩阵乘法 '...ij,...jk->...ik'。 
   其他情况请直接使用 vt-einsum。"
  (with-float-safe
    (let ((ra (length (vt-shape (ensure-vt a)))) 
          (rb (length (vt-shape (ensure-vt b)))))
      (cond ((and (= ra 0) (= rb 0)) 
             (vt-* a b :dtype dtype :out out))
            ((and (= ra 1) (= rb 1)) 
             (vt-einsum "i,i->" a b :dtype dtype :out out)) 
            ((and (= ra 2) (= rb 1)) 
             (vt-einsum "ij,j->i" a b :dtype dtype :out out)) 
            ((and (= ra 1) (= rb 2)) 
             (vt-einsum "i,ij->j" a b :dtype dtype :out out)) 
            ((and (= ra 2) (= rb 2)) 
             (vt-einsum "ij,jk->ik" a b :dtype dtype :out out)) 
            ((and (>= ra 2) (>= rb 2)) 
             (vt-einsum "...ij,...jk->...ik" a b :dtype dtype :out out)) 
            (t (error "vt-dot: unsupported dimensions (a: ~d, b: ~d).
                      use vt-einsum directly." ra rb))))))

(defun vt-outer (a b &key (flatten t) dtype out)
  "计算张量外积。
  flatten = t (默认):
    先将输入展平为一维向量，再计算外积，返回二维矩阵。
    完全兼容 numpy 的 outer 函数 (支持任意维度输入自动展平)。
  flatten = nil:
    保留输入的每个轴，将所有轴拼接形成新张量。
    例如：2d 与 3d 计算后得到 5d 张量；若包含标量(0维)，则直接广播相乘。"
  (with-float-safe
    (let* ((a-vt (ensure-vt a))
           (b-vt (ensure-vt b))
           (shape-a (vt-shape a-vt))
           (shape-b (vt-shape b-vt)))
      (if flatten
          (let ((1d-a (if (null shape-a) 
                          (vt-reshape a-vt '(1)) 
                          (vt-flatten a-vt)))
                (1d-b (if (null shape-b) 
                          (vt-reshape b-vt '(1)) 
                          (vt-flatten b-vt))))
            (vt-einsum "i,j->ij" 1d-a 1d-b :dtype dtype :out out))          
          (if (or (null shape-a) (null shape-b))
              (vt-* a-vt b-vt :dtype dtype :out out)
              (let* ((rank-a (length shape-a))
                     (rank-b (length shape-b))
                     (a-reshaped
                       (vt-reshape a-vt
                                   (append shape-a
                                           (make-list rank-b 
                                                      :initial-element 1))))
                     (b-reshaped
                       (vt-reshape b-vt
                                   (append (make-list rank-a 
                                                      :initial-element 1)
                                           shape-b))))
                (vt-* a-reshaped b-reshaped :dtype dtype :out out)))))))


(defun vt-trace (matrix &key (offset 0) axis1 axis2 dtype out)
  "矩阵迹（对标 numpy.trace）：指定对角线上元素之和。

   OFFSET 语义同 vt-diagonal —— 正数取主对角线上方（右上）的第 OFFSET 条
   对角线，负数取下方；越界（没有任何对角元素）时结果是 0（numpy 亦然）。

   AXIS1 / AXIS2 是取迹的两条轴，**必须同时给出**；缺省取最后两轴
   （与 vt-diagonal 一致）。注意 numpy 的 trace 缺省是 axis1=0/axis2=1
   —— 2 维时两者等价，batch 输入需显式给出才能复现 numpy 缺省行为。

   复数：结果保持复数 dtype（见 complex.lisp 的同名分派）。"
  (let ((rank (length (vt-shape (ensure-vt matrix)))))
    ;; 硬契约用 ERROR 而非 ASSERT：assert 可能被 (safety 0) 编译剔除
    (when (< rank 2)
      (error "vt-trace: 要求秩 >= 2，收到秩 ~a" rank)))
  (with-float-safe
    ;; vt-diagonal 把对角线轴放在**末尾**，因此沿 -1 求和即「沿对角方向收缩」，
    ;; 保留其余轴 —— 正是 numpy.trace 对高维输入的行为（3D + axis1/axis2
    ;; 得到「去掉这两轴」的张量，而不是标量）。
    (vt-sum (vt-diagonal matrix :offset offset :axis1 axis1 :axis2 axis2)
            :axis -1 :dtype dtype :out out)))

(defun vt-norm (vt &key axis keepdims dtype out)
  "l2 范数（欧几里得范数）
   - 整数输入按 NumPy 语义提升到 float64，避免整数 sqrt 截断。
   - 若提供 out，求和与开方将原地执行，零临时内存分配。"
  (with-float-safe
    (let* ((in-dtype (vt-dtype vt))
           (effective-dtype (or dtype
                                (if (vt-float-dtype-p in-dtype)
                                    in-dtype
                                    :float64)))
           (sq (if (vt-float-dtype-p in-dtype)
                   (vt-square vt)
                   (vt-square (vt-astype vt :float64)))))
      (if axis
          (let ((sum-res (vt-sum sq :axis axis :keepdims keepdims
                                    :dtype effective-dtype :out out)))
            (vt-sqrt sum-res :dtype (vt-dtype sum-res) :out sum-res))
          (let ((sum-res (vt-sum sq :dtype effective-dtype :out out)))
            (vt-sqrt sum-res :dtype (vt-dtype sum-res) :out sum-res)
            ;; v0.4.1（差分测试 R10）：keepdims 且无 axis 时对标
            ;; np.linalg.norm(x, keepdims=True) —— 全部维度收缩为 1
            ;; （1d → (1,)，2d → (1,1)），而非 0 维。
            (if (and keepdims (vt-shape vt))
                (vt-reshape sum-res (make-list (length (vt-shape vt))
                                               :initial-element 1))
                sum-res))))))

(defun vt-l1-norm (vt &key axis keepdims dtype out)
  "l1 范数"
  (with-float-safe
    (vt-sum (vt-abs vt) :axis axis :keepdims keepdims
			:dtype dtype :out out)))

(defun vt-frobenius-norm (matrix &key axis keepdims dtype out)
  "frobenius 范数 (专用于矩阵)"
  (with-float-safe
    (vt-norm matrix :axis axis :keepdims keepdims :dtype dtype :out out)))

;;; 方程求解与矩阵分析

(defun ensure-contiguous-2d-vt (vt)
  "确保矩阵在内存中是连续的."
  (with-float-safe
    (if (vt-contiguous-p vt)
	vt
	(vt-contiguous vt))))

;;; ============================================================
;;; 分解函数双路径内核：SIMD(AVX/FMA) 快速内核 + 通用内核
;;; ------------------------------------------------------------
;;; 路由模式与 simd-matmul.lisp（vt-matmul）、vt-svd 一致：
;;;   1. 快速路径（SBCL 且 *vt-linalg-simd-enabled* = t）：sb-simd
;;;      f64.4 + FMA 内核（SAP 访问，with-pinned-objects 固定数组）；
;;;   2. 通用路径（任何平台 / *vt-linalg-simd-enabled* = nil）：纯标量
;;;      内核（仅 simple-array aref，零 SIMD/SAP 依赖），即原实现的
;;;      算法与逐元素公式。
;;; 内核语义与原实现逐元素公式一致（axpy/dir 由调用方传入符号），
;;; 仅浮点求和分组不同（SIMD 分部分和，同 vt-svd 的容差约定）。
;;; ============================================================

(defvar *vt-linalg-simd-enabled* t
  "vt-lu/vt-qr/vt-solve/vt-cholesky/vt-eig 是否启用 SIMD(AVX/FMA) 内核；置 nil 强制使用通用内核（无 SIMD 依赖，同 vt-matmul 回退 einsum 的模式）。")

;;; ============ 通用内核（无 SIMD 依赖：simple-array + aref）============

(declaim (inline %ln-f64-axpy %ln-f64-dot %ln-f64-rot2))

;; y[yi+k] += a * x[xi+k]，k = 0..n-1
(defun %ln-f64-axpy (x xi y yi n a)
  (declare (type (simple-array double-float (*)) x y)
           (type fixnum xi yi n)
           (type double-float a)
           (optimize (speed 3) (safety 0) (debug 0)))
  (loop for k of-type fixnum from 0 below n
        do (incf (aref y (+ yi k)) (* a (aref x (+ xi k))))))

;; Σ_k x[xi+k] * y[yi+k]
(defun %ln-f64-dot (x xi y yi n)
  (declare (type (simple-array double-float (*)) x y)
           (type fixnum xi yi n)
           (optimize (speed 3) (safety 0) (debug 0)))
  (let ((sum 0.0d0))
    (declare (type double-float sum))
    (loop for k of-type fixnum from 0 below n
          do (incf sum (* (aref x (+ xi k)) (aref y (+ yi k)))))
    sum))

;; 双行旋转（就地）：x' = c·x + s·y ; y' = -s·x + c·y
(defun %ln-f64-rot2 (x xi y yi n c s)
  (declare (type (simple-array double-float (*)) x y)
           (type fixnum xi yi n)
           (type double-float c s)
           (optimize (speed 3) (safety 0) (debug 0)))
  (loop for k of-type fixnum from 0 below n
        do (let ((xv (aref x (+ xi k)))
                 (yv (aref y (+ yi k))))
             (setf (aref x (+ xi k)) (+ (* c xv) (* s yv))
                   (aref y (+ yi k)) (+ (* (- s) xv) (* c yv))))))

;;; ============ AVX 快速内核（SBCL + sb-simd）============
;;; 寻址约定同 %svd-f64-*-avx：f64.4-sap-ref 的下标为【元素索引】，
;;; 标量尾经 sap-ref-double 访问时才乘 8 转字节偏移。
#+sbcl
(progn
  (declaim (inline %ln-f64-axpy-avx %ln-f64-dot-avx %ln-f64-rot2-avx))

  ;; y += a·x（2×f64.4 FMA 展开打破依赖链，尾部标量补齐）
  (defun %ln-f64-axpy-avx (sx xi sy yi n a)
    (declare (type sb-sys:system-area-pointer sx sy)
             (type fixnum xi yi n)
             (type double-float a)
             (optimize (speed 3) (safety 0) (debug 0) (space 0)))
    (let ((av (sb-simd-avx:f64.4 a))
          (main8 (the fixnum (logand n -8)))
          (main4 (the fixnum (logand n -4))))
      (declare (type fixnum main8 main4))
      (do ((k 0 (the fixnum (+ k 8)))
           (i xi (the fixnum (+ i 8)))
           (j yi (the fixnum (+ j 8))))
          ((>= k main8))
        (declare (type fixnum k i j))
        (setf (sb-simd-avx:f64.4-sap-ref sy j)
              (sb-simd-fma:f64.4-fmadd av (sb-simd-avx:f64.4-sap-ref sx i)
                                       (sb-simd-avx:f64.4-sap-ref sy j))
              (sb-simd-avx:f64.4-sap-ref sy (+ j 4))
              (sb-simd-fma:f64.4-fmadd av (sb-simd-avx:f64.4-sap-ref sx (+ i 4))
                                       (sb-simd-avx:f64.4-sap-ref sy (+ j 4)))))
      (do ((k main8 (the fixnum (+ k 4)))
           (i (+ xi main8) (the fixnum (+ i 4)))
           (j (+ yi main8) (the fixnum (+ j 4))))
          ((>= k main4))
        (declare (type fixnum k i j))
        (setf (sb-simd-avx:f64.4-sap-ref sy j)
              (sb-simd-fma:f64.4-fmadd av (sb-simd-avx:f64.4-sap-ref sx i)
                                       (sb-simd-avx:f64.4-sap-ref sy j))))
      (do ((k main4 (1+ k))
           (i (+ xi main4) (1+ i))
           (j (+ yi main4) (1+ j)))
          ((>= k n))
        (declare (type fixnum k i j))
        (incf (sb-sys:sap-ref-double sy (* j 8))
              (* a (sb-sys:sap-ref-double sx (* i 8)))))))

  ;; Σ x·y（4 累加器展开，结构同 %svd-f64-dot-avx）
  (defun %ln-f64-dot-avx (sx xi sy yi n)
    (declare (type sb-sys:system-area-pointer sx sy)
             (type fixnum xi yi n)
             (optimize (speed 3) (safety 0) (debug 0) (space 0)))
    (let ((acc0 (sb-simd-avx:f64.4 0.0d0))
          (acc1 (sb-simd-avx:f64.4 0.0d0))
          (acc2 (sb-simd-avx:f64.4 0.0d0))
          (acc3 (sb-simd-avx:f64.4 0.0d0))
          (main16 (the fixnum (logand n -16)))
          (main4  (the fixnum (logand n -4)))
          (tail 0.0d0))
      (declare (type fixnum main16 main4) (type double-float tail))
      (do ((e 0 (the fixnum (+ e 16)))
           (i xi (the fixnum (+ i 16)))
           (j yi (the fixnum (+ j 16))))
          ((>= e main16))
        (declare (type fixnum e i j))
        (setf acc0 (sb-simd-fma:f64.4-fmadd
                    (sb-simd-avx:f64.4-sap-ref sx i)
                    (sb-simd-avx:f64.4-sap-ref sy j) acc0)
              acc1 (sb-simd-fma:f64.4-fmadd
                    (sb-simd-avx:f64.4-sap-ref sx (+ i 4))
                    (sb-simd-avx:f64.4-sap-ref sy (+ j 4)) acc1)
              acc2 (sb-simd-fma:f64.4-fmadd
                    (sb-simd-avx:f64.4-sap-ref sx (+ i 8))
                    (sb-simd-avx:f64.4-sap-ref sy (+ j 8)) acc2)
              acc3 (sb-simd-fma:f64.4-fmadd
                    (sb-simd-avx:f64.4-sap-ref sx (+ i 12))
                    (sb-simd-avx:f64.4-sap-ref sy (+ j 12)) acc3)))
      (do ((e main16 (the fixnum (+ e 4)))
           (i (+ xi main16) (the fixnum (+ i 4)))
           (j (+ yi main16) (the fixnum (+ j 4))))
          ((>= e main4))
        (declare (type fixnum e i j))
        (setf acc0 (sb-simd-fma:f64.4-fmadd
                    (sb-simd-avx:f64.4-sap-ref sx i)
                    (sb-simd-avx:f64.4-sap-ref sy j) acc0)))
      (do ((e main4 (1+ e))
           (i (+ xi main4) (1+ i))
           (j (+ yi main4) (1+ j)))
          ((>= e n))
        (declare (type fixnum e i j))
        (incf tail (* (sb-sys:sap-ref-double sx (* i 8))
                      (sb-sys:sap-ref-double sy (* j 8)))))
      (let ((ab (sb-simd-avx:f64.4+ acc0 acc1))
            (cd (sb-simd-avx:f64.4+ acc2 acc3)))
        (multiple-value-bind (a b c d) (sb-simd-avx:f64.4-values (sb-simd-avx:f64.4+ ab cd))
          (declare (type double-float a b c d))
          (+ tail (+ (+ a c) (+ b d)))))))

  ;; 双行旋转（2×f64.4 展开；x' = c·x + s·y ; y' = -s·x + c·y）
  (defun %ln-f64-rot2-avx (sx xi sy yi n c s)
    (declare (type sb-sys:system-area-pointer sx sy)
             (type fixnum xi yi n)
             (type double-float c s)
             (optimize (speed 3) (safety 0) (debug 0) (space 0)))
    (let ((cv (sb-simd-avx:f64.4 c))
          (sv (sb-simd-avx:f64.4 s))
          (nsv (sb-simd-avx:f64.4 (- s)))
          (main8 (the fixnum (logand n -8)))
          (main4 (the fixnum (logand n -4))))
      (declare (type fixnum main8 main4))
      (do ((k 0 (the fixnum (+ k 8)))
           (i xi (the fixnum (+ i 8)))
           (j yi (the fixnum (+ j 8))))
          ((>= k main8))
        (declare (type fixnum k i j))
        (let ((xv  (sb-simd-avx:f64.4-sap-ref sx i))
              (yv  (sb-simd-avx:f64.4-sap-ref sy j))
              (xv2 (sb-simd-avx:f64.4-sap-ref sx (+ i 4)))
              (yv2 (sb-simd-avx:f64.4-sap-ref sy (+ j 4))))
          (setf (sb-simd-avx:f64.4-sap-ref sx i)
                (sb-simd-fma:f64.4-fmadd cv xv (sb-simd-avx:f64.4* sv yv))
                (sb-simd-avx:f64.4-sap-ref sy j)
                (sb-simd-fma:f64.4-fmadd cv yv (sb-simd-avx:f64.4* nsv xv))
                (sb-simd-avx:f64.4-sap-ref sx (+ i 4))
                (sb-simd-fma:f64.4-fmadd cv xv2 (sb-simd-avx:f64.4* sv yv2))
                (sb-simd-avx:f64.4-sap-ref sy (+ j 4))
                (sb-simd-fma:f64.4-fmadd cv yv2 (sb-simd-avx:f64.4* nsv xv2)))))
      (do ((k main8 (the fixnum (+ k 4)))
           (i (+ xi main8) (the fixnum (+ i 4)))
           (j (+ yi main8) (the fixnum (+ j 4))))
          ((>= k main4))
        (declare (type fixnum k i j))
        (let ((xv (sb-simd-avx:f64.4-sap-ref sx i))
              (yv (sb-simd-avx:f64.4-sap-ref sy j)))
          (setf (sb-simd-avx:f64.4-sap-ref sx i)
                (sb-simd-fma:f64.4-fmadd cv xv (sb-simd-avx:f64.4* sv yv))
                (sb-simd-avx:f64.4-sap-ref sy j)
                (sb-simd-fma:f64.4-fmadd cv yv (sb-simd-avx:f64.4* nsv xv)))))
      (do ((k main4 (1+ k))
           (i (+ xi main4) (1+ i))
           (j (+ yi main4) (1+ j)))
          ((>= k n))
        (declare (type fixnum k i j))
        (let ((xv (sb-sys:sap-ref-double sx (* i 8)))
              (yv (sb-sys:sap-ref-double sy (* j 8))))
          (setf (sb-sys:sap-ref-double sx (* i 8)) (+ (* c xv) (* s yv))
                (sb-sys:sap-ref-double sy (* j 8)) (+ (* (- s) xv) (* c yv))))))))


;;; ============ 句柄统一访问器（aref / SAP 同签名，供宏内联）============
(declaim (inline %ln-f64-aref))
(defun %ln-f64-aref (arr idx)
  (declare (type (simple-array double-float (*)) arr)
           (type fixnum idx)
           (optimize (speed 3) (safety 0) (debug 0)))
  (aref arr idx))

(declaim (inline (setf %ln-f64-aref)))
(defun (setf %ln-f64-aref) (v arr idx)
  (declare (type double-float v)
           (type (simple-array double-float (*)) arr)
           (type fixnum idx)
           (optimize (speed 3) (safety 0) (debug 0)))
  (setf (aref arr idx) v))

#+sbcl
(progn
  (declaim (inline %ln-sap-f64-aref))
  (defun %ln-sap-f64-aref (sap idx)
    (declare (type sb-sys:system-area-pointer sap)
             (type fixnum idx)
             (optimize (speed 3) (safety 0) (debug 0)))
    (sb-sys:sap-ref-double sap (* idx 8)))

  (declaim (inline (setf %ln-sap-f64-aref)))
  (defun (setf %ln-sap-f64-aref) (v sap idx)
    (declare (type double-float v)
             (type sb-sys:system-area-pointer sap)
             (type fixnum idx)
             (optimize (speed 3) (safety 0) (debug 0)))
    (setf (sb-sys:sap-ref-double sap (* idx 8)) v)))

;;; ============ LU 消元核心（宏生成通用 / AVX 双引擎）============
;;; 迭代次序、部分选主元、行交换与消元公式与原 vt-lu 完全一致；
;;; 消元内层改用 AXPY 内核（data[i,j] -= mult·data[k,j] ≡
;;; y += (-mult)·x，IEEE 负号精确，逐元素位一致）。

(defmacro %define-vt-lu-core (name axpy-fn eref handle-type)
  `(defun ,name (wh off s0 s1 m n k-max piv)
     "部分选主元 LU 消元核心（就地修改 wh，填充 piv，返回 sign）。
wh: 连续工作矩阵句柄（通用: simple-array；AVX: SAP），行主序，s1=1。"
     (declare (type ,handle-type wh)
              (type fixnum off s0 s1 m n k-max)
              (type (simple-array fixnum (*)) piv)
              (optimize (speed 3) (safety 0) (debug 0) (space 0)))
     (let ((sign 1))
       (declare (type fixnum sign))
       (loop for k of-type fixnum from 0 below k-max
             for base-k of-type fixnum = (+ off (* k s0))
             for kk     of-type fixnum = (+ off (* k s0) (* k s1))
             do (let ((max-row k)
                      (max-val (abs (,eref wh kk))))
                  (declare (type fixnum max-row)
                           (type double-float max-val))
                  ;; ---------- 选主元（部分选主元，与原实现一致）----------
                  (loop for i of-type fixnum from (1+ k) below m
                        for p of-type fixnum = (+ off (* i s0) (* k s1))
                        for val of-type double-float = (abs (,eref wh p))
                        do (when (> val max-val)
                             (setf max-val val max-row i)))
                  (unless (< max-val 1d-300)
                    ;; ---------- 行交换 ----------
                    (unless (= max-row k)
                      (rotatef (aref piv k) (aref piv max-row))
                      (setf sign (- sign))
                      (loop for j of-type fixnum from 0 below n
                            do (rotatef (,eref wh (+ base-k j))
                                        (,eref wh (+ off (* max-row s0) j)))))
                    ;; ---------- 消元：mult 写入 ik，剩余段 AXPY ----------
                    (let ((pivot (,eref wh kk))
                          (j0    (1+ k)))
                      (declare (type double-float pivot) (type fixnum j0))
                      (loop for i of-type fixnum from j0 below m
                            for base-i of-type fixnum = (+ off (* i s0))
                            for mult of-type double-float
                              = (/ (,eref wh (+ base-i k)) pivot)
                            do (setf (,eref wh (+ base-i k)) mult)
                               ;; data[i,j0..] -= mult·data[k,j0..]
                               (,axpy-fn wh (+ base-k j0) wh (+ base-i j0)
                                         (- n j0) (- mult)))))))
       sign)))

;; 通用引擎：无 SIMD 依赖（原 vt-lu 的算法与逐元素公式，纯 aref）
(%define-vt-lu-core %vt-lu-core-generic %ln-f64-axpy %ln-f64-aref
                    (simple-array double-float (*)))

;; AVX 快速引擎：SBCL + sb-simd（内核约定同 simd-matmul.lisp）
#+sbcl
(%define-vt-lu-core %vt-lu-core-avx %ln-f64-axpy-avx %ln-sap-f64-aref
                    sb-sys:system-area-pointer)

(defun vt-lu (matrix)
  "部分选主元 LU 分解。返回 (values a piv sign)：
   a    : 原地修改后的合并 LU 矩阵（vt，float64）
   piv  : fixnum 向量（不是列表），行置换
   sign : 行交换符号（+1/-1）
   满足 P @ A = L @ U，其中 P[i, piv[i]] = 1。支持非方阵。

  实现：双路径路由（同 vt-matmul/vt-svd 模式）——
  1. 快速路径（SBCL 且 *vt-linalg-simd-enabled* = t）：消元内层 AXPY
     用 sb-simd AVX/FMA 内核（FMA 乘加，舍入分组略有差异）；
  2. 通用路径：原实现的纯标量循环，逐元素位一致。
  *vt-linalg-simd-enabled* 控制路由，两条路径共享选主元/行交换语义。"
  (let* ((a (ensure-contiguous-2d-vt (vt-astype matrix :float64)))
         (m (first (vt-shape a)))    ; 行数
         (n (second (vt-shape a)))   ; 列数
         (k-max (min m n))
         (data (vt-data a))
         (s0 (first (vt-strides a))) ; 行步长
         (s1 (second (vt-strides a))); 列步长（ensure-contiguous 保证 =1）
         (off (vt-offset a))
         (piv (make-array m :element-type 'fixnum))
         (sign 1))
    (declare (type (simple-array double-float (*)) data)
             (type fixnum m n k-max s0 s1 off sign)
             (type (simple-array fixnum (*)) piv))
    ;; 初始化置换向量
    (loop for i of-type fixnum from 0 below m
          do (setf (aref piv i) i))
    ;; 引擎分派（ensure-contiguous 保证 s1=1，AXPY 内核要求连续段）
    (cond
      #+sbcl
      ((and *vt-linalg-simd-enabled* (= s1 1))
       (setf sign (sb-sys:with-pinned-objects (data)
                    (%vt-lu-core-avx (sb-sys:vector-sap data)
                                     off s0 s1 m n k-max piv))))
      (t
       (setf sign (%vt-lu-core-generic data off s0 s1 m n k-max piv))))
    (values a piv sign)))


(defun vt-det (matrix &key out)
  "基于 lu 分解计算行列式，返回 0 维张量"
  (with-float-safe
    (let ((shape (vt-shape matrix)))
      ;; 校验方阵
      (unless (and (= (length shape) 2)
                   (= (first shape) (second shape)))
        (error "vt-det: 行列式仅支持方阵，收到形状 ~a" shape)))
    ;; :out 硬契约（§4.2）：det 结果为 0 维 float64 标量，:out 必须形状为 NIL、
    ;; dtype 为 :float64，否则 vt-fill 会静默填满任意形状 / 静默降精度写入。
    (when out
      (vt-check-out out nil :float64 :op-name "vt-det"))
    (multiple-value-bind (lu piv sign)
        (vt-lu matrix)
      (declare (ignore piv))
      (let* ((n (first (vt-shape lu)))
             (data (vt-data lu))
             (s0 (first (vt-strides lu)))
             (s1 (second (vt-strides lu)))
             (off (vt-offset lu))
             (det sign))
        (loop for i from 0 below n
              for pivot = (aref data (+ off (* i s0) (* i s1)))
              when (zerop pivot)
                do (return-from vt-det
                     (if out (vt-fill out 0.0d0)
			 (make-vt nil 0.0d0 :dtype :float64)))
              do (setf det (* det pivot)))
        (if out (vt-fill out det) (make-vt nil det :dtype :float64))))))



;;; ============ 三角替换核心（宏生成通用 / AVX 双引擎）============
;;; 置换 → 前代 → 回代，迭代次序与原 vt-solve 完全一致；
;;; 前代/回代的内层（nrhs 维连续段）改用 AXPY 内核：
;;; b[i,j] -= mult·b[k,j] ≡ y += (-mult)·x（IEEE 负号精确，逐元素
;;; 位一致；AVX 路径经 FMA，舍入分组略有差异）。
;;; vt-inv（nrhs = n，O(n³)）是最大受益者。

(defmacro %define-vt-solve-subst (name axpy-fn eref handle-type)
  `(defun ,name (lu-h lu-off lu-s0 lu-s1 n
                 b-h b-off b-s0 b-s1
                 ob-h ob-off ob-s0 ob-s1
                 piv nrhs singular-threshold)
     "行置换 + 前代 + 回代核心（就地修改 b）。
lu/b/ob: 连续行主序句柄（通用: simple-array；AVX: SAP），s1=1。"
     (declare (type ,handle-type lu-h b-h ob-h)
              (type (simple-array fixnum (*)) piv)
              (type fixnum lu-off lu-s0 lu-s1 n
                    b-off b-s0 b-s1 ob-off ob-s0 ob-s1 nrhs)
              (type double-float singular-threshold)
              (optimize (speed 3) (safety 0) (debug 0) (space 0)))
     ;; 1. 应用行置换 pb（逐元素拷贝，与原实现一致）
     (loop for i of-type fixnum from 0 below n do
       (loop for j of-type fixnum from 0 below nrhs do
         (setf (,eref b-h (+ b-off (* i b-s0) (* j b-s1)))
               (,eref ob-h (+ ob-off (* (aref piv i) ob-s0)
                             (* j b-s1))))))
     ;; 2. 前代（单位下三角）：b[i,:] -= L[i,k]·b[k,:]
     (loop for k of-type fixnum from 0 below n do
       (loop for i of-type fixnum from (1+ k) below n
             for mult of-type double-float
               = (,eref lu-h (+ lu-off (* i lu-s0) (* k lu-s1)))
             do (,axpy-fn b-h (+ b-off (* k b-s0))
                          b-h (+ b-off (* i b-s0))
                          nrhs (- mult))))
     ;; 3. 回代（上三角）：除对角，b[i,:] -= U[i,k]·b[k,:]
     (loop for k of-type fixnum from (1- n) downto 0 do
       (let ((pivot (,eref lu-h (+ lu-off (* k lu-s0) (* k lu-s1)))))
         (declare (type double-float pivot))
         (when (or (zerop pivot)
                   (< (abs pivot) singular-threshold))
           (error "LinAlgError: Singular matrix. Cannot solve or invert."))
         (loop for j of-type fixnum from 0 below nrhs do
           (setf (,eref b-h (+ b-off (* k b-s0) (* j b-s1)))
                 (/ (,eref b-h (+ b-off (* k b-s0) (* j b-s1)))
                    pivot)))
         (loop for i of-type fixnum from 0 below k
               for factor of-type double-float
                 = (,eref lu-h (+ lu-off (* i lu-s0) (* k lu-s1)))
               do (,axpy-fn b-h (+ b-off (* k b-s0))
                            b-h (+ b-off (* i b-s0))
                            nrhs (- factor)))))))

;; 通用引擎：无 SIMD 依赖（原 vt-solve 的算法与逐元素公式，纯 aref）
(%define-vt-solve-subst %vt-solve-subst-generic %ln-f64-axpy %ln-f64-aref
                        (simple-array double-float (*)))

;; AVX 快速引擎：SBCL + sb-simd
#+sbcl
(%define-vt-solve-subst %vt-solve-subst-avx %ln-f64-axpy-avx %ln-sap-f64-aref
                        sb-sys:system-area-pointer)

(defun vt-solve (a b &key out)
  "求解线性方程组 ax = b (支持多右端项)

  实现：双路径路由（同 vt-matmul/vt-svd 模式）——
  1. 快速路径（SBCL 且 *vt-linalg-simd-enabled* = t）：前代/回代的
     nrhs 维连续段走 sb-simd AVX/FMA AXPY 内核（vt-inv 多右端项
     O(n³) 主受益）；
  2. 通用路径：原实现的纯标量循环，逐元素位一致。
  *vt-linalg-simd-enabled* 控制路由，两条路径共享置换/阈值/奇异报错语义。"
  (with-float-safe
    (let* ((a-shape (vt-shape a)))
      ;; 校验系数矩阵为方阵
      (unless (and (= (length a-shape) 2)
                   (= (first a-shape) (second a-shape)))
        (error "vt-solve: 线性方程组求解要求系数矩阵为方阵，收到形状 ~a" a-shape))
      (let* ((a (ensure-contiguous-2d-vt a))
             (b-vt (ensure-vt b))
             (n (first (vt-shape a)))
             (b-shape (vt-shape b-vt)))
        ;; 逻辑层契约：b 的行数必须与系数矩阵阶数一致。
        ;; 缺失此校验时，消元循环按 n 行读写 b 的底层缓冲区，
        ;; 行数不足会越界写入（内存安全 L1/L2 违规）。
        (unless (and (>= (length b-shape) 1)
                     (= (first b-shape) n))
          (error "vt-solve: b 的行数 ~a 与系数矩阵阶数 ~a 不匹配（b 形状 ~a）"
                 (if b-shape (first b-shape) 0) n b-shape))
        (let* ((nrhs (if (> (length b-shape) 1) (second b-shape) 1))
               ;; 1D b 先 reshape 为 n×1 列向量（与原实现一致），再确保连续
               (b-copy (ensure-contiguous-2d-vt
                        (let ((b2 (vt-astype (vt-copy b-vt) :float64)))
                          (if (= nrhs 1)
                              (vt-reshape b2 (list n 1))
                              b2))))
               (orig-b (vt-copy b-copy)))
          (multiple-value-bind (lu piv sign)
              (vt-lu a)
            (declare (ignore sign))
            (let ((lu-data (vt-data lu))
                  (lu-s0 (first (vt-strides lu)))
                  (lu-s1 (second (vt-strides lu)))
                  (lu-off (vt-offset lu))
                  (b-data (vt-data b-copy))
                  (b-s0 (first (vt-strides b-copy)))
                  (b-s1 (second (vt-strides b-copy)))
                  (b-off (vt-offset b-copy))
                  (ob-data (vt-data orig-b))
                  (ob-s0 (first (vt-strides orig-b)))
                  (ob-s1 (second (vt-strides orig-b)))
                  (ob-off (vt-offset orig-b)))
              ;; 计算相对奇异阈值：基于 LU 矩阵的最大绝对值元素，
              ;; 避免绝对阈值 1e-12 对小范数矩阵误判为奇异、对大范数矩阵漏判。
              ;; 阈值 = n * eps * max|lu_ij|，n 为矩阵阶数，
              ;; eps = double-float-epsilon (≈2.22e-16)，n 倍补偿消元累积误差。
              (let ((max-abs 0.0d0))
                (declare (double-float max-abs))
                (loop for i below (length lu-data) do
                  (let ((v (abs (aref lu-data i))))
                    (when (> v max-abs)
                      (setf max-abs v))))
                (let ((singular-threshold
                        (* (max 1 n) double-float-epsilon
                           (max max-abs 1.0d0))))
                  (declare (double-float singular-threshold))
                  ;; 引擎分派（ensure-contiguous/vt-lu 保证 s1=1）
                  (cond
                    #+sbcl
                    ((and *vt-linalg-simd-enabled*
                          (= lu-s1 1) (= b-s1 1) (= ob-s1 1))
                     (sb-sys:with-pinned-objects (lu-data b-data ob-data)
                       (%vt-solve-subst-avx
                        (sb-sys:vector-sap lu-data) lu-off lu-s0 lu-s1 n
                        (sb-sys:vector-sap b-data) b-off b-s0 b-s1
                        (sb-sys:vector-sap ob-data) ob-off ob-s0 ob-s1
                        piv nrhs singular-threshold)))
                    (t
                     (%vt-solve-subst-generic
                      lu-data lu-off lu-s0 lu-s1 n
                      b-data b-off b-s0 b-s1
                      ob-data ob-off ob-s0 ob-s1
                      piv nrhs singular-threshold)))
                  (let ((res (if (= nrhs 1)
                                 (vt-reshape b-copy (list n))
                                 b-copy)))
                    (if out
                        (vt-map #'identity res :out out)
                        res)))))))))))


(defun vt-inv (matrix)
  "矩阵求逆。"
  (with-float-safe
    (let* ((n (first (vt-shape matrix)))
           (identity (vt-eye n :dtype (vt-dtype matrix))))
      (vt-solve matrix identity))))


;;; 3. 矩阵分解

;;; ============ QR 核心（宏生成通用 / AVX 双引擎）============
;;; 迭代次序、Householder 公式与原 vt-qr 完全一致。两处完善：
;;;   1. 原实现的 w[j] = Σ v[ii]·R[i+ii, i+j] 沿「列」跨行访问（stride
;;;      = r-s0），缓存命中率差——重构为按行 AXPY 累加
;;;      （w += v[ii]·R[i+ii, i..]），每个 w[j] 的逐元素求和顺序不变，
;;;      结果与原实现位一致；
;;;   2. reflector 向量不再逐级构造 vt-slice/vt-copy 对象，直接在打包
;;;      的扁平缓冲 v-buf 上操作。
;;; 消元/更新内层统一走 AXPY 内核（y -= a·x ≡ y += (-a)·x，IEEE 负号
;;; 精确，逐元素位一致；AVX 路径经 FMA，舍入分组略有差异）。
;;;
;;; 数值零列防护（两引擎共享，语义一致）：
;;;   秩亏矩阵（如常值矩阵）消元后，后续列是 ε 级噪声列。对噪声列
;;;   构造 Householder 反射时 β = 2/‖v‖² = O(1/‖x‖²) 会随噪声指数级
;;;   缩小而溢出（SIMD 路径 FMA 舍入误差更小、噪声衰减更快，先触发
;;;   β→inf，再经 inf·0 扩散为 NaN；通用路径在大尺寸下同样会触发）。
;;;   防护规则：
;;;   1. 相对阈值：‖x‖ ≤ ε·max(m,n)·max|R_ij|（原矩阵尺度）时视为
;;;      数值零列，β = 0 跳过反射。跳过只影响 ε 级噪声（backward
;;;      误差不超过 QR 固有水平），且两引擎决策一致，保证秩亏输入
;;;      下 SIMD ≡ 通用结果一致；
;;;   2. 绝对下限：‖v‖² < +vt-householder-vnorm2-min+ 时 β 必然溢出
;;;      或损失精度（极小尺度矩阵），同样视为零列跳过。

(defconstant +vt-householder-vnorm2-min+ 1d-290
  "Householder β = 2/‖v‖² 的安全下限：‖v‖² 低于该值时 β 超出
double-float 表示范围（≈1.8e308）或精度受损，列应视为数值零。")

(defmacro %define-vt-qr-core (name axpy-fn eref handle-type)
  `(defun ,name (r-h r-off r-s0 n m k
                 q-h q-off nq
                 v-buf v-offs betas w w-h)
     "Householder QR 核心（就地修改 r/q，填充 v-buf/v-offs/betas）。
r: m×n 连续行主序（s1=1）；q: m×nq 连续行主序（s1=1）。
v-offs[i]: 第 i 个 reflector 在 v-buf 中的起始下标；长度 m-i。
w: 工作缓冲（数组，两路径同构）；w-h: w 的引擎句柄（通用=w，AVX=SAP）。"
     (declare (type ,handle-type r-h q-h w-h)
              (type (simple-array double-float (*)) v-buf betas w)
              (type (simple-array fixnum (*)) v-offs)
              (type fixnum r-off r-s0 n m k q-off nq)
              (optimize (speed 3) (safety 0) (debug 0) (space 0)))
     ;; 数值零列相对阈值：ε·max(m,n)·max|R_ij|（分解前计算一次，
     ;; 两引擎输入相同 → 阈值相同 → 跳过决策一致）
     (let ((a-scale 0.0d0))
       (declare (type double-float a-scale))
       (loop for si of-type fixnum from 0 below m
             do (loop for sj of-type fixnum from 0 below n
                      do (let ((v (abs (,eref r-h (+ r-off (* si r-s0) sj)))))
                           (when (> v a-scale)
                             (setf a-scale v)))))
       (let ((a-zero-threshold
               (* a-scale (the fixnum (max 1 (max m n)))
                  double-float-epsilon)))
         (declare (type double-float a-zero-threshold))
         ;; ======== 1. 正向分解，更新 r，存 reflector ========
         (loop for i of-type fixnum from 0 below k
               do (let ((m-sub (- m i))
                        (n-sub (- n i))
                        (v-off (aref v-offs i)))
                    (declare (type fixnum m-sub n-sub v-off))
                    (if (<= m-sub 1)
                        (setf (aref betas i) 0.0d0)
                        (let ((base-r (+ r-off (* i r-s0) i))) ; R[i, i]
                          (declare (type fixnum base-r))
                          ;; ---- Householder：x = R[i..m-1, i]，stride r-s0 ----
                          (let ((norm-sq 0.0d0))
                            (declare (type double-float norm-sq))
                            (loop for ii of-type fixnum from 0 below m-sub
                                  do (let ((val (,eref r-h (+ base-r (* ii r-s0)))))
                                       (incf norm-sq (* val val))
                                       (setf (aref v-buf (+ v-off ii)) val)))
                            (let ((norm (sqrt norm-sq)))
                              (declare (type double-float norm))
                              (if (or (zerop norm)
                                      (<= norm a-zero-threshold))
                                  (setf (aref betas i) 0.0d0)
                                  (let* ((x0 (aref v-buf v-off))
                                         (sigma (if (>= x0 0.0d0) (- norm) norm))
                                         (v-norm-sq (* 2.0d0 norm (+ norm (abs x0)))))
                                    (declare (type double-float x0 sigma v-norm-sq))
                                    (if (< v-norm-sq +vt-householder-vnorm2-min+)
                                        ;; ‖v‖² 下溢：β = 2/‖v‖² 必溢出，视为零列
                                        (setf (aref betas i) 0.0d0)
                                        (progn
                                          ;; v[0] = x0 - sigma（与 compute-householder 一致）
                                          (setf (aref v-buf v-off) (- x0 sigma))
                                          (let ((beta (/ 2.0d0 v-norm-sq)))
                                            (declare (type double-float beta))
                                            (setf (aref betas i) beta)
                                            (when (> beta 0.0d0)
                                              ;; ---- w += v[ii]·R[i+ii, i..n-1]（行 AXPY）----
                                              (loop for j of-type fixnum from 0 below n-sub
                                                    do (setf (aref w j) 0.0d0))
                                              (loop for ii of-type fixnum from 0 below m-sub
                                                    do (,axpy-fn r-h
                                                                 (+ base-r (* ii r-s0))
                                                                 w-h 0 n-sub
                                                                 (aref v-buf (+ v-off ii))))
                                              ;; ---- R[i+ii, i..] -= (β·v[ii])·w ----
                                              (loop for ii of-type fixnum from 0 below m-sub
                                                    do (,axpy-fn w-h 0
                                                                 r-h (+ base-r (* ii r-s0))
                                                                 n-sub
                                                                 (- (* beta
                                                                       (aref v-buf (+ v-off ii))))))))))))))))))
         ;; ======== 2. 反向累积 q ========
         (loop for i of-type fixnum from (1- k) downto 0
               do (let ((m-sub (- m i))
                        (nq-sub nq)
                        (beta (aref betas i))
                        (v-off (aref v-offs i)))
                    (declare (type fixnum m-sub nq-sub v-off))
                    (when (> beta 0.0d0)
                      (let ((base-q (+ q-off (* i nq))))
                        (declare (type fixnum base-q))
                        ;; ---- w += v[ii]·Q[i+ii, 0..nq-1]（行 AXPY）----
                        (loop for j of-type fixnum from 0 below nq-sub
                              do (setf (aref w j) 0.0d0))
                        (loop for ii of-type fixnum from 0 below m-sub
                              do (,axpy-fn q-h (+ base-q (* ii nq)) w-h 0 nq-sub
                                           (aref v-buf (+ v-off ii))))
                        ;; ---- Q[i+ii, 0..nq-1] -= (β·v[ii])·w ----
                        (loop for ii of-type fixnum from 0 below m-sub
                              do (,axpy-fn w-h 0
                                           q-h (+ base-q (* ii nq)) nq-sub
                                           (- (* beta (aref v-buf (+ v-off ii))))))))))))))

;; 通用引擎：无 SIMD 依赖（原 vt-qr 算法 + 行 AXPY 重构，纯 aref）
(%define-vt-qr-core %vt-qr-core-generic %ln-f64-axpy %ln-f64-aref
                    (simple-array double-float (*)))

;; AVX 快速引擎：SBCL + sb-simd
#+sbcl
(%define-vt-qr-core %vt-qr-core-avx %ln-f64-axpy-avx %ln-sap-f64-aref
                    sb-sys:system-area-pointer)

(defun vt-qr (matrix &key (mode :reduced))
  "矩阵 qr 分解。
   matrix : m×n 矩阵。
   mode :reduced 返回 q(m×k), r(k×n)，k = min(m,n)。
         :full 返回 q(m×m), r(m×n)。

  实现：双路径路由（同 vt-matmul/vt-svd 模式）——
  1. 快速路径（SBCL 且 *vt-linalg-simd-enabled* = t）：AXPY 内层走
     sb-simd AVX/FMA 内核；
  2. 通用路径：原算法 + 行 AXPY 重构（修复原实现按列跨行访问的
     性能缺陷，逐元素求和顺序不变），纯标量 aref，无 SIMD 依赖。
  两条路径共享 Householder 公式、迭代次序与输出形状契约。"
  (assert (= 2 (vt-order matrix)))
  (with-float-safe
    (let* ((row (first  (vt-shape matrix)))
           (col (second (vt-shape matrix)))
           (k   (min row col))
           (need-full (eq mode :full))
           (nq   (if need-full row k))
           (r    (ensure-contiguous-2d-vt (vt-astype matrix :float64)))
           (q    (vt-eye row :cols nq :dtype :float64))
           (r-data (vt-data r))
           (r-s0   (first  (vt-strides r)))
           (r-s1   (second (vt-strides r)))
           (r-off  (vt-offset r))
           (q-data (vt-data q))
           (q-s1   (second (vt-strides q)))
           (q-off  (vt-offset q))
           (betas  (make-array k :element-type 'double-float))
           ;; 打包 reflector 缓冲：Σ_{i<k} (m-i)
           (v-total (loop for i of-type fixnum from 0 below k
                          sum (- row i) of-type fixnum))
           (v-buf  (make-array v-total :element-type 'double-float
                                       :initial-element 0.0d0))
           (v-offs (make-array k :element-type 'fixnum))
           ;; 复用的工作缓冲，长度取 max(row,col) 覆盖两种模式
           (w (make-array (max row col) :element-type 'double-float
                                      :initial-element 0.0d0)))
      (declare (type fixnum row col k r-s0 r-s1 r-off q-s1 q-off)
               (type (simple-array double-float (*)) r-data q-data betas v-buf w)
               (type (simple-array fixnum (*)) v-offs))
      (let ((acc 0))
        (declare (type fixnum acc))
        (loop for i of-type fixnum from 0 below k
              do (setf (aref v-offs i) acc)
                 (incf acc (- row i))))
      ;; 引擎分派（ensure-contiguous/vt-eye 保证 s1=1）
      (cond
        #+sbcl
        ((and *vt-linalg-simd-enabled* (= r-s1 1) (= q-s1 1))
         (sb-sys:with-pinned-objects (r-data q-data v-buf betas w)
           (%vt-qr-core-avx (sb-sys:vector-sap r-data) r-off r-s0 col row k
                            (sb-sys:vector-sap q-data) q-off nq
                            v-buf v-offs betas w (sb-sys:vector-sap w))))
        (t
         (%vt-qr-core-generic r-data r-off r-s0 col row k
                              q-data q-off nq v-buf v-offs betas w w)))
      (values q (if need-full r (vt-slice r (list 0 k) '(:all)))))))


(defun compute-householder (x)
  "给定向量 x，返回 v, beta, sigma 使得
   h = i - beta * v * v^t 满足 h*x = sigma * e1。
   其中 sigma = -sign(x[0]) * ||x||，β = 2 / ||v||²。
   当 x 为零向量，或 ||v||² 低于 +vt-householder-vnorm2-min+（β 将
   溢出/损失精度，x 数值上为零向量）时返回 beta=0, sigma=0。"
  (with-float-safe
    (let* ((x-data   (vt-data x))
           (x-stride (first  (vt-strides x)))
           (x-off    (vt-offset x))
           (size     (vt-size x)))
      (declare (type (simple-array double-float (*)) x-data)
               (type fixnum x-stride x-off size)
               (optimize (speed 3) (safety 0) (debug 0)))
      ;; 一次遍历求 ||x||²
      (let ((norm-sq 0.0d0))
        (declare (type double-float norm-sq))
        (loop for i of-type fixnum from 0 below size
              for p of-type fixnum = x-off then (+ p x-stride)
              for val of-type double-float = (aref x-data p)
              do (incf norm-sq (* val val)))
        (let ((norm (sqrt norm-sq)))
          (declare (type double-float norm))
          (if (zerop norm)
              (values (vt-copy x) 0.0d0 0.0d0)
              (let* ((x0   (aref x-data x-off))
                     (sigma (if (>= x0 0.0d0) (- norm) norm))
                     (v-norm-sq (* 2.0d0 norm (+ norm (abs x0)))))
                (declare (type double-float x0 sigma v-norm-sq))
                (if (< v-norm-sq +vt-householder-vnorm2-min+)
                    ;; ‖v‖² 下溢：β = 2/‖v‖² 必溢出，x 视为数值零向量
                    (values (vt-copy x) 0.0d0 0.0d0)
                    (let* ((beta (/ 2.0d0 v-norm-sq))
                           (v      (vt-copy x))
                           (v-data (vt-data v))
                           (v-off    (vt-offset v)))
                      (declare (type double-float beta)
                               (type (simple-array double-float (*)) v-data)
                               (type fixnum v-off))
                      (setf (aref v-data v-off) (- x0 sigma))
                      (values v beta sigma))))))))))

(defun vt-matrix-rank (matrix &optional (tol 1e-10))
  "计算矩阵的秩 (线性代数定义：线性无关的行/列数)。
   基于带部分选主元的高斯消元法，统计非零主元的数量。
   等价于 numpy.linalg.matrix_rank。"
  (assert (= 2 (length (vt-shape matrix))) 
          (matrix) "vt-matrix-rank requires a 2D tensor")
  (with-float-safe
    (let* ((m (first (vt-shape matrix)))
           (n (second (vt-shape matrix)))
           ;; 复制一份，避免破坏原矩阵
           (a (vt-astype matrix :float64))
           (a-data (vt-data a))
           (a-offset (vt-offset a))
           (s0 (first (vt-strides a))) ; 行步长
           (s1 (second (vt-strides a))); 列步长
           (rank 0)
           (row 0))
      ;; 从第一列到最后一列进行消元
      (loop for col from 0 below n
            while (< row m) do
              ;; 1. 在当前列及以下的行中，寻找绝对值最大的主元 (部分选主元)
              (let ((max-val 0.0d0)
		    (max-row row))
		(loop for i from row below m
                      for val = (abs (aref a-data (+ a-offset
						     (* i s0)
						     (* col s1))))
                      when (> val max-val)
			do (setf max-val val max-row i))
		
		;; 2. 判断主元是否足够大 (大于容差 tol)
		(if (> max-val tol)
		    (progn
                      (incf rank)
                      ;; 3. 如果最大主元不在当前行，则交换行
                      (unless (= max-row row)
			(loop for j from col below n
                              for off1 = (+ a-offset (* row s0) (* j s1))
                              for off2 = (+ a-offset (* max-row s0) (* j s1))
                              do (rotatef (aref a-data off1) (aref a-data off2))))
                      ;; 4. 消元：将当前列下方的元素清零
                      (let ((pivot (aref a-data (+ a-offset
						   (* row s0)
						   (* col s1)))))
			(loop for i from (1+ row) below m
                              for multiplier = (/ (aref a-data (+ a-offset (* i s0)
								  (* col s1)))
						  pivot)
                              do (loop for j from (1+ col) below n
                                       for off-target = (+ a-offset (* i s0) (* j s1))
                                       for off-source = (+ a-offset (* row s0) (* j s1))
                                       do (decf (aref a-data off-target)
						(* multiplier (aref a-data off-source)))))
			;; 物理上将下方元素置零，保证数值干净
			(loop for i from (1+ row) below m
                              do (setf (aref a-data (+ a-offset (* i s0)
						       (* col s1)))
				       0.0d0)))
                      ;; 5. 处理下一行
                      (incf row))
		    ;; 如果主元太小，说明该列线性相关，跳过该列，继续看下一列
		    nil)))
      rank)))

(defun extend-orthogonal-basis (u-econ &key (rng *vt-default-random-state*))
  "将 m×k 的 u_econ 通过随机向量 + gram-schmidt 补全为 m×m 正交矩阵。"
  (declare (random-state rng))
  (with-float-safe
    (let* ((m (first (vt-shape u-econ)))
           (k (second (vt-shape u-econ)))
           (extra (- m k)))
      (if (zerop extra) u-econ
          (let ((u-full (vt-zeros (list m m) :dtype :float64)))
            (dotimes (i k)
              (setf (vt-slice u-full '(:all) (list i))
                    (vt-slice u-econ '(:all) (list i))))
            (loop for col from k below m
                  for v = (vt-random (list m) :rng rng) do
                    (loop repeat 2 do
                      (dotimes (j col)
                        ;; flatten 避免 2D-1D 广播分配
                        (let* ((uj (vt-flatten
				    (vt-slice u-full '(:all) (list j))))
                               (proj (vt-ref (vt-dot uj v))))
                          (setf v (vt-- v (vt-scale uj proj))))))
                    (let ((norm (sqrt (vt-ref (vt-dot v v)))))
                      (if (> norm 1e-12)
                          (setf (vt-slice u-full '(:all) (list col))
                                (vt-scale v (/ 1.0s0 norm)))
                          (error "failed to generate orthogonal vector"))))
            u-full)))))

;;; ============================================================
;;; vt-svd 双路径：SIMD(AVX/FMA) 快速内核 + 通用内核（无 SIMD 依赖）
;;; ------------------------------------------------------------
;;; 路由模式与 simd-matmul.lisp（vt-matmul）一致：
;;;   1. 快速路径（SBCL 且 *vt-linalg-simd-enabled* = t）：转置布局
;;;      W = A^T / VT = V^T + sb-simd f64.4 FMA 内核（SAP 访问，
;;;      with-pinned-objects 固定数组，尾部标量补齐）；
;;;   2. 通用路径（任何平台 / *vt-linalg-simd-enabled* = nil）：同一
;;;      转置布局 + 纯标量内核（仅 simple-array aref，零 SIMD/SAP
;;;      依赖）。即保留原 vt-svd 的单边 Jacobi 实现，并修复其重大
;;;      性能缺陷——原实现把旋转直接作用在行主序矩阵的「列」上，
;;;      所有内层循环以行步长大步长跳跃，缓存命中率与向量化都很
;;;      差；转置后列运算全部变为连续行访问。
;;; 两条路径共享同一迭代次序、收敛判据、旋转公式、范数解析更新、
;;; 零奇异值补全与重建逻辑；仅浮点求和分组不同（SIMD 分部分和）。
;;; ============================================================

;;; ============ 通用内核（无 SIMD 依赖：simple-array + aref）============
;;; 原 vt-svd Jacobi 算法的点积 / 双行旋转 / 元素读原语，纯标量实现，
;;; 不依赖 sb-simd 与 sb-sys，任何平台可用。

(declaim (inline %svd-f64-dot %svd-f64-rot2 %svd-f64-ref))

(defun %svd-f64-dot (w i j len)
  (declare (type (simple-array double-float (*)) w)
           (type fixnum i j len)
           (optimize (speed 3) (safety 0) (debug 0)))
  (let ((sum 0.0d0))
    (declare (type double-float sum))
    (loop for r of-type fixnum from 0 below len
          do (incf sum (* (aref w (+ i r)) (aref w (+ j r)))))
    sum))

(defun %svd-f64-rot2 (w p q len c s)
  (declare (type (simple-array double-float (*)) w)
           (type fixnum p q len)
           (type double-float c s)
           (optimize (speed 3) (safety 0) (debug 0)))
  (loop for r of-type fixnum from 0 below len
        do (let ((u (aref w (+ p r)))
                 (v (aref w (+ q r))))
             (setf (aref w (+ p r)) (- (* c u) (* s v))
                   (aref w (+ q r)) (+ (* s u) (* c v))))))

(defun %svd-f64-ref (w idx)
  (declare (type (simple-array double-float (*)) w)
           (type fixnum idx)
           (optimize (speed 3) (safety 0) (debug 0)))
  (aref w idx))

;;; ============ AVX 快速内核（SBCL + sb-simd）============
#+sbcl
(progn
  (declaim (inline %svd-f64-dot-avx %svd-f64-rot2-avx %svd-f64-ref-avx))

  ;; ---- f64 连续点积（AVX/FMA，4 累加器展开打破 FMA 依赖链）----
  ;; sap 为已按字节偏移修好的句柄；i/j/长度均为元素索引
  (defun %svd-f64-dot-avx (sap i j len)
    (declare (type sb-sys:system-area-pointer sap)
             (type fixnum i j len)
             (optimize (speed 3) (safety 0) (debug 0) (space 0)))
    (let ((acc0 (sb-simd-avx:f64.4 0.0d0))
          (acc1 (sb-simd-avx:f64.4 0.0d0))
          (acc2 (sb-simd-avx:f64.4 0.0d0))
          (acc3 (sb-simd-avx:f64.4 0.0d0))
          (main16 (the fixnum (logand len -16)))
          (main4  (the fixnum (logand len -4)))
          (tail 0.0d0))
      (declare (type fixnum main16 main4) (type double-float tail))
      (do ((e 0 (the fixnum (+ e 16)))
           (i0 i (the fixnum (+ i0 16)))
           (j0 j (the fixnum (+ j0 16))))
          ((>= e main16))
        (declare (type fixnum e i0 j0))
        (setf acc0 (sb-simd-fma:f64.4-fmadd
                    (sb-simd-avx:f64.4-sap-ref sap i0)
                    (sb-simd-avx:f64.4-sap-ref sap j0) acc0)
              acc1 (sb-simd-fma:f64.4-fmadd
                    (sb-simd-avx:f64.4-sap-ref sap (+ i0 4))
                    (sb-simd-avx:f64.4-sap-ref sap (+ j0 4)) acc1)
              acc2 (sb-simd-fma:f64.4-fmadd
                    (sb-simd-avx:f64.4-sap-ref sap (+ i0 8))
                    (sb-simd-avx:f64.4-sap-ref sap (+ j0 8)) acc2)
              acc3 (sb-simd-fma:f64.4-fmadd
                    (sb-simd-avx:f64.4-sap-ref sap (+ i0 12))
                    (sb-simd-avx:f64.4-sap-ref sap (+ j0 12)) acc3)))
      (do ((e main16 (the fixnum (+ e 4)))
           (i0 (+ i main16) (the fixnum (+ i0 4)))
           (j0 (+ j main16) (the fixnum (+ j0 4))))
          ((>= e main4))
        (declare (type fixnum e i0 j0))
        (setf acc0 (sb-simd-fma:f64.4-fmadd
                    (sb-simd-avx:f64.4-sap-ref sap i0)
                    (sb-simd-avx:f64.4-sap-ref sap j0) acc0)))
      (do ((e main4 (1+ e))
           (i0 (+ i main4) (1+ i0))
           (j0 (+ j main4) (1+ j0)))
          ((>= e len))
        (declare (type fixnum e i0 j0))
        (incf tail (* (sb-sys:sap-ref-double sap (* i0 8))
                      (sb-sys:sap-ref-double sap (* j0 8)))))
      (let ((ab (sb-simd-avx:f64.4+ acc0 acc1))
            (cd (sb-simd-avx:f64.4+ acc2 acc3)))
        (multiple-value-bind (a b c d) (sb-simd-avx:f64.4-values (sb-simd-avx:f64.4+ ab cd))
          (declare (type double-float a b c d))
          (+ tail (+ (+ a c) (+ b d)))))))

  ;; ---- f64 双行旋转（就地）：p' = c·p - s·q ; q' = s·p + c·q ----
  ;; 主循环每次处理 8 个元素（两组独立 4 宽），减少循环开销
  (defun %svd-f64-rot2-avx (sap p q len c s)
    (declare (type sb-sys:system-area-pointer sap)
             (type fixnum p q len)
             (type double-float c s)
             (optimize (speed 3) (safety 0) (debug 0) (space 0)))
    (let ((cv (sb-simd-avx:f64.4 c))
          (sv (sb-simd-avx:f64.4 s))
          (main8 (the fixnum (logand len -8)))
          (main  (the fixnum (logand len -4))))
      (declare (type fixnum main8 main))
      (do ((e 0 (the fixnum (+ e 8)))
           (pp p (the fixnum (+ pp 8)))
           (qq q (the fixnum (+ qq 8))))
          ((>= e main8))
        (declare (type fixnum e pp qq))
        (let ((u0 (sb-simd-avx:f64.4-sap-ref sap pp))
              (v0 (sb-simd-avx:f64.4-sap-ref sap qq))
              (u1 (sb-simd-avx:f64.4-sap-ref sap (+ pp 4)))
              (v1 (sb-simd-avx:f64.4-sap-ref sap (+ qq 4))))
          (setf (sb-simd-avx:f64.4-sap-ref sap pp)
                (sb-simd-fma:f64.4-fmsub cv u0 (sb-simd-avx:f64.4* sv v0))
                (sb-simd-avx:f64.4-sap-ref sap qq)
                (sb-simd-fma:f64.4-fmadd sv u0 (sb-simd-avx:f64.4* cv v0))
                (sb-simd-avx:f64.4-sap-ref sap (+ pp 4))
                (sb-simd-fma:f64.4-fmsub cv u1 (sb-simd-avx:f64.4* sv v1))
                (sb-simd-avx:f64.4-sap-ref sap (+ qq 4))
                (sb-simd-fma:f64.4-fmadd sv u1 (sb-simd-avx:f64.4* cv v1)))))
      (do ((e main8 (the fixnum (+ e 4)))
           (pp (+ p main8) (the fixnum (+ pp 4)))
           (qq (+ q main8) (the fixnum (+ qq 4))))
          ((>= e main))
        (declare (type fixnum e pp qq))
        (let ((u (sb-simd-avx:f64.4-sap-ref sap pp))
              (v (sb-simd-avx:f64.4-sap-ref sap qq)))
          (setf (sb-simd-avx:f64.4-sap-ref sap pp)
                (sb-simd-fma:f64.4-fmsub cv u (sb-simd-avx:f64.4* sv v))
                (sb-simd-avx:f64.4-sap-ref sap qq)
                (sb-simd-fma:f64.4-fmadd sv u (sb-simd-avx:f64.4* cv v)))))
      (do ((e main (1+ e))
           (pp (+ p main) (1+ pp))
           (qq (+ q main) (1+ qq)))
          ((>= e len))
        (declare (type fixnum e pp qq))
        (let ((u (sb-sys:sap-ref-double sap (* pp 8)))
              (v (sb-sys:sap-ref-double sap (* qq 8))))
          (setf (sb-sys:sap-ref-double sap (* pp 8)) (- (* c u) (* s v))
                (sb-sys:sap-ref-double sap (* qq 8)) (+ (* s u) (* c v)))))))

  (defun %svd-f64-ref-avx (sap idx)
    (declare (type sb-sys:system-area-pointer sap)
             (type fixnum idx)
             (optimize (speed 3) (safety 0) (debug 0)))
    (sb-sys:sap-ref-double sap (* idx 8))))

;;; ============ Jacobi 扫描引擎（宏生成通用 / AVX 两份同构实现）============
;;; 迭代次序 / 收敛判据 / 旋转公式 / 范数解析更新与原实现一致；
;;; 仅内核实现不同（通用: aref 标量；AVX: sb-simd FMA + SAP）。

(defmacro %define-vt-svd-sweeps (name dot-fn rot-fn ref-fn handle-type)
  `(defun ,name (wh vh col-norms m n max-sweeps tol)
     "单边 Jacobi 扫描引擎（就地修改 wh/vh/col-norms）。
wh: 转置工作矩阵 W 的句柄，行 i = A 的列 i，行宽 m。
vh: VT = V^T 的句柄，行宽 n。"
     (declare (type ,handle-type wh vh)
              (type (simple-array double-float (*)) col-norms)
              (type fixnum m n max-sweeps)
              (type double-float tol)
              (optimize (speed 3) (safety 0) (debug 0) (space 0)))
     (let ((sweep 0)
           (changed t))
       (declare (type fixnum sweep))
       (loop while (and changed (< sweep max-sweeps)) do
         (setf changed nil)
         ;; ====== 重算所有行范数（每轮扫描一次，抑制浮点漂移）======
         (loop for i of-type fixnum from 0 below n
               do (let ((sum 0.0d0)
                        (base (* i m)))
                    (declare (type double-float sum)
                             (type fixnum base))
                    (loop for r of-type fixnum from 0 below m
                          do (let ((val (,ref-fn wh (+ base r))))
                               (incf sum (* val val))))
                    (setf (aref col-norms i) sum)))
         ;; ====== 本轮所有 (i, j) 列对 ======
         (loop for i of-type fixnum from 0 below (1- n) do
           (let ((base-i (* i m)))
             (declare (type fixnum base-i))
             (loop for j of-type fixnum from (1+ i) below n
                   for alpha of-type double-float = (aref col-norms i)
                   for beta  of-type double-float = (aref col-norms j)
                   ;; ---- 点积：连续内存单次遍历 ----
                   ;; 范数为 0 ⟺ 整行为零向量 ⟹ 点积必为 0，直接跳过扫描
                   for gamma of-type double-float =
                     (if (or (zerop alpha) (zerop beta))
                         0.0d0
                         (,dot-fn wh base-i (* j m) m))
                   when (> (abs gamma) (* tol (sqrt (* alpha beta)))) do
                     (setf changed t)
                     (let* ((zeta (/ (- beta alpha) (* 2.0d0 gamma)))
                            (t-abs (/ 1.0d0 (+ (abs zeta)
                                               (sqrt (+ 1.0d0 (* zeta zeta))))))
                            (t-val (if (>= zeta 0.0d0) t-abs (- t-abs)))
                            (c (/ 1.0d0 (sqrt (+ 1.0d0 (* t-val t-val)))))
                            (s (* c t-val)))
                       (declare (type double-float zeta t-abs t-val c s))
                       ;; ---- 旋转 W 的第 i, j 行（= A 的第 i, j 列）----
                       (,rot-fn wh base-i (* j m) m c s)
                       ;; ---- 旋转 VT 的第 i, j 行（= V 的第 i, j 列）----
                       (,rot-fn vh (* i n) (* j n) n c s)
                       ;; ---- 用解析公式更新缓存范数（O(1)）----
                       ;; new α = c²·α - 2cs·γ + s²·β
                       ;; new β = s²·α + 2cs·γ + c²·β
                       (setf (aref col-norms i)
                             (+ (* c c alpha)
                                (* -2.0d0 c s gamma)
                                (* s s beta)))
                       (setf (aref col-norms j)
                             (+ (* s s alpha)
                                (*  2.0d0 c s gamma)
                                (* c c beta)))))))
           (incf sweep))
       (values))))

;; 通用引擎：无 SIMD 依赖（原实现算法 + 修复跨行访问缺陷）
(%define-vt-svd-sweeps %vt-svd-jacobi-sweeps
  %svd-f64-dot %svd-f64-rot2 %svd-f64-ref
  (simple-array double-float (*)))

;; AVX 快速引擎：SBCL + sb-simd（内核约定同 simd-matmul.lisp）
#+sbcl
(%define-vt-svd-sweeps %vt-svd-jacobi-sweeps-avx
  %svd-f64-dot-avx %svd-f64-rot2-avx %svd-f64-ref-avx
  sb-sys:system-area-pointer)

(defun %vt-svd-jacobi-sweeps* (w-data vt-data col-norms m n max-sweeps tol)
  "扫描引擎分派：SIMD 启用且可用 → AVX 快速引擎（固定数组取 SAP）；
否则 → 通用引擎（纯标量，无 SIMD 依赖）。模式同 vt-matmul 快速路径回退。"
  (declare (type (simple-array double-float (*)) w-data vt-data col-norms)
           (type fixnum m n max-sweeps)
           (type double-float tol))
  #+sbcl
  (if *vt-linalg-simd-enabled*
      (sb-sys:with-pinned-objects (w-data vt-data)
        (%vt-svd-jacobi-sweeps-avx (sb-sys:vector-sap w-data)
                                   (sb-sys:vector-sap vt-data)
                                   col-norms m n max-sweeps tol))
      (%vt-svd-jacobi-sweeps w-data vt-data col-norms m n max-sweeps tol))
  #-sbcl
  (%vt-svd-jacobi-sweeps w-data vt-data col-norms m n max-sweeps tol)
  (values))

(defun %vt-svd-rebuild (w-data vt-data col-norms m n k full-matrices)
  "共享重建（AVX / 通用两条引擎路径共用）：按奇异值降序排列（stable，
并列保持列序，同原实现），填充 U、零奇异值随机正交补全（两轮
Gram-Schmidt）、构造 vt（= V^T），并按 full-matrices 分派形状。
返回 (values u s vt)。"
  (declare (type (simple-array double-float (*)) w-data vt-data col-norms)
           (type fixnum m n k))
  (let* ((s-vec (make-array k :element-type 'double-float))
         (u-k   (vt-zeros (list m k) :dtype :float64))
         (u-data (vt-data u-k))
         (pairs (stable-sort
                 (loop for col of-type fixnum from 0 below n
                       collect (cons (sqrt (aref col-norms col)) col))
                 #'> :key #'car)))
    (declare (type (simple-array double-float (*)) s-vec u-data))

    ;; 数值零奇异值阈值：k·ε·σ_max。秩亏矩阵消元后的小奇异值由舍入噪声
    ;; 决定（如常值矩阵的 σ ~ 1e-152 落在 denormal 域，解析列范数与实际
    ;; 列范数严重失真，用其作除数会得到非单位长度的 U 列，破坏正交性）。
    ;; σ 低于阈值时视为零奇异值：σ 置零并走正交补全，两条引擎路径使用
    ;; 同一阈值 → 补全决策一致。
    (let ((s-max 0.0d0))
      (declare (type double-float s-max))
      (loop for col of-type fixnum from 0 below n
            do (let ((v (sqrt (aref col-norms col))))
                 (when (> v s-max)
                   (setf s-max v))))
      (let ((zero-cut (* s-max (the fixnum (max 1 k))
                         double-float-epsilon)))
        (declare (type double-float zero-cut))

        ;; ---- 填充 U（m×k 行主序）：列 new-i = W 行 old-col / σ；零 σ 留零待补 ----
        (let ((new-i 0))
          (declare (type fixnum new-i))
          (dolist (pair pairs)
            (when (< new-i k)
              (let ((val (car pair))
                    (old-col (cdr pair)))
                (declare (type double-float val) (type fixnum old-col))
                (if (<= val zero-cut)
                    (setf (aref s-vec new-i) 0.0d0)
                    (let ((inv (/ 1.0d0 val))
                          (src (* old-col m)))
                      (declare (type double-float inv) (type fixnum src))
                      (setf (aref s-vec new-i) val)
                      (loop for r of-type fixnum from 0 below m
                            do (setf (aref u-data (+ (* r k) new-i))
                                     (* inv (aref w-data (+ src r)))))))
                (incf new-i)))))

        ;; ---- 零奇异值 → 随机正交补（两轮 Gram-Schmidt，同原实现）----
        (dotimes (i k)
          (when (zerop (aref s-vec i))
            (let* ((rv-t (vt-random (list m) :rng *vt-default-random-state*))
                   (rv   (vt-data rv-t)))
              (declare (type (simple-array double-float (*)) rv))
              (loop repeat 2 do
                (dotimes (j i)
                  (let ((proj 0.0d0))
                    (declare (type double-float proj))
                    (loop for r of-type fixnum from 0 below m
                          do (incf proj (* (aref u-data (+ (* r k) j))
                                           (aref rv r))))
                    (loop for r of-type fixnum from 0 below m
                          do (decf (aref rv r)
                                   (* proj (aref u-data (+ (* r k) j))))))))
              (let ((norm (sqrt (loop for r of-type fixnum from 0 below m
                                      sum (* (aref rv r) (aref rv r))
                                      of-type double-float))))
                (declare (type double-float norm))
                (if (> norm 1e-12)
                    (let ((inv (/ 1.0d0 norm)))
                      (declare (type double-float inv))
                      (loop for r of-type fixnum from 0 below m
                            do (setf (aref u-data (+ (* r k) i))
                                     (* (aref rv r) inv))))
                    (loop for r of-type fixnum from 0 below m
                          do (setf (aref u-data (+ (* r k) i))
                                   (aref rv r))))))))))

    ;; ---- 构造 vt（= V^T，n×n）：前 k 行取排序后的 VT 行，其余升序补齐 ----
    (let* ((vt-out (vt-zeros (list n n) :dtype :float64))
           (vt-out-data (vt-data vt-out))
           (used (make-array n :element-type 'bit :initial-element 0)))
      (declare (type (simple-array double-float (*)) vt-out-data))
      (let ((row 0))
        (declare (type fixnum row))
        (dolist (pair pairs)
          (when (< row k)
            (let ((old (cdr pair)))
              (declare (type fixnum old))
              (setf (bit used old) 1)
              (replace vt-out-data vt-data
                       :start1 (* row n) :end1 (+ (* row n) n)
                       :start2 (* old n) :end2 (+ (* old n) n))
              (incf row)))))
      (let ((row k))
        (declare (type fixnum row))
        (loop for old of-type fixnum from 0 below n
              when (zerop (bit used old))
                do (replace vt-out-data vt-data
                            :start1 (* row n) :end1 (+ (* row n) n)
                            :start2 (* old n) :end2 (+ (* old n) n))
                   (incf row)))

      ;; ---- 返回分派（形状契约同原实现）----
      (let ((s-vt (vt-from-sequence (coerce s-vec 'list) :dtype :float64)))
        (if (not full-matrices)
            (let ((vt-k (if (< k n)
                            (let ((vk (vt-zeros (list k n) :dtype :float64)))
                              (replace (vt-data vk) vt-out-data :end1 (* k n))
                              vk)
                            vt-out)))
              (values u-k s-vt vt-k))
            (cond ((= m n) (values u-k s-vt vt-out))
                  ((> m n) (values (extend-orthogonal-basis u-k) s-vt vt-out))
                  ((< m n) (values u-k s-vt vt-out))))))))

(defun vt-svd (matrix &key (full-matrices nil) (max-sweeps 50) (tol 1e-14))
  "奇异值分解 a = u s v^t。
  full-matrices : t 则返回完整尺寸 u(m×m), s(k), vt(n×n) (k = min(m,n))
                  : nil 返回经济尺寸 u(m×k), s(k), vt(k×n)
  max-sweeps : jacobi 最大扫描次数
  tol : 收敛容差

  实现：单边 Jacobi 双路径路由（同 vt-matmul 的快速路径 + 通用回退模式）：
  1. 快速路径（SBCL 且 *vt-linalg-simd-enabled* = t）：转置布局 W = A^T +
     连续内存 + sb-simd AVX/FMA 内核（内核约定参考 simd-matmul.lisp），
     点积与旋转均为连续访问；
  2. 通用路径：同一转置布局 + 纯标量内核（保留原实现的 Jacobi 算法与
     全部语义，修复其按列跨行访问的性能缺陷），无 SIMD 依赖。
  两条路径共享迭代次序、收敛判据、零奇异值补全与重建逻辑。"
  (assert (= 2 (vt-order matrix)))
  (with-float-safe
    (let* ((mat (vt-astype matrix :float64))
           (m (first  (vt-shape mat)))
           (n (second (vt-shape mat)))
           (k (min m n)))
      ;; ---- 1×1 边界 ----
      (when (and (= m 1) (= n 1))
        (let ((val (aref (vt-data mat) (vt-offset mat))))
          (return-from vt-svd
            (values (vt-const '(1 1) 1.0d0 :dtype :float64)
                    (vt-const '(1) (abs val) :dtype :float64)
                    (if (>= val 0) (vt-ones '(1 1) :dtype :float64)
                        (vt-const '(1 1) -1.0d0 :dtype :float64))))))

      (let* ((mat-data (vt-data mat))
             (mat-off  (vt-offset mat))
             (mat-s0   (first  (vt-strides mat)))
             (mat-s1   (second (vt-strides mat)))
             ;; W = A^T：n 行 × m 列连续，行 i = A 的列 i
             (w-data  (make-array (* n m) :element-type 'double-float :initial-element 0.0d0))
             ;; VT = V^T：n×n 连续，行 i = V 的列 i；初始单位阵
             (vt-data (make-array (* n n) :element-type 'double-float :initial-element 0.0d0))
             (col-norms (make-array n :element-type 'double-float :initial-element 0.0d0)))
        (declare (type (simple-array double-float (*)) mat-data w-data vt-data col-norms)
                 (type fixnum m n k mat-off mat-s0 mat-s1))

        ;; ---- 填充 W：一次转置拷贝，支持任意输入布局（含 stride=0 广播视图）----
        (loop for i of-type fixnum from 0 below n
              do (let ((src (+ mat-off (* i mat-s1)))
                       (dst (* i m)))
                   (declare (type fixnum src dst))
                   (loop for r of-type fixnum from 0 below m
                         do (setf (aref w-data dst) (aref mat-data src))
                            (incf src mat-s0)
                            (incf dst))))
        ;; ---- 填充 VT 单位阵 ----
        (loop for i of-type fixnum from 0 below n
              do (setf (aref vt-data (+ (* i n) i)) 1.0d0))

        ;; ---- Jacobi 扫描（引擎分派：SIMD 快速 / 通用，就地修改 W/VT/范数）----
        ;; tol 收敛到 double-float（兼容旧默认值 1e-14 为 single-float 的情况）
        (%vt-svd-jacobi-sweeps* w-data vt-data col-norms m n max-sweeps
                                (coerce tol 'double-float))

        ;; ---- 共享重建：排序 / 填 U / 零 σ 补全 / 构造 vt / 形状分派 ----
        (%vt-svd-rebuild w-data vt-data col-norms m n k full-matrices)))))

;;;; linalg-extensions.lisp — 扩展线性代数功能
;;; ============================================================
;;; 1. Cholesky 分解
;;; ============================================================

;;; ============ Cholesky 核心（宏生成通用 / AVX 双引擎）============
;;; 算法与原 vt-cholesky 完全一致（k 递增累加、非正定报错、upper 转置）；
;;; 两个内层 k 点积改用 dot 内核（通用路径逐元素顺序与原实现一致；
;;; AVX 路径分部分和分组不同，同 vt-svd 容差约定）。

(defmacro %define-vt-cholesky-core (name dot-fn eref handle-type)
  `(defun ,name (a-h a-off a-s0 l-h l-off n)
     "就地计算 L（下三角）。a/l: 连续行主序句柄，s1=1。"
     (declare (type ,handle-type a-h l-h)
              (type fixnum a-off a-s0 l-off n)
              (optimize (speed 3) (safety 0) (debug 0) (space 0)))
     (loop for j of-type fixnum from 0 below n do
       (let ((sum (,eref a-h (+ a-off (* j a-s0) j))))
         (declare (type double-float sum))
         ;; sum -= Σ_k l[j,k]² （k 递增，与原实现一致）
         (decf sum (,dot-fn l-h (+ l-off (* j n)) l-h (+ l-off (* j n)) j))
         (when (<= sum 0.0d0)
           (error "vt-cholesky: 矩阵不是正定的"))
         (setf (,eref l-h (+ l-off (* j n) j)) (sqrt sum)))
       (loop for i of-type fixnum from (1+ j) below n do
         (let ((sum (,eref a-h (+ a-off (* i n) j))))
           (declare (type double-float sum))
           ;; sum -= Σ_k l[i,k]·l[j,k]
           (decf sum (,dot-fn l-h (+ l-off (* i n)) l-h (+ l-off (* j n)) j))
           (setf (,eref l-h (+ l-off (* i n) j))
                 (/ sum (,eref l-h (+ l-off (* j n) j)))))))))

;; 通用引擎：无 SIMD 依赖（原 vt-cholesky 算法，纯 aref）
(%define-vt-cholesky-core %vt-cholesky-core-generic %ln-f64-dot %ln-f64-aref
                          (simple-array double-float (*)))

;; AVX 快速引擎：SBCL + sb-simd
#+sbcl
(%define-vt-cholesky-core %vt-cholesky-core-avx %ln-f64-dot-avx %ln-sap-f64-aref
                          sb-sys:system-area-pointer)

(defun vt-cholesky (matrix &key (upper nil))
  "对正定对称矩阵进行 Cholesky 分解。

  实现：双路径路由（同 vt-matmul/vt-svd 模式）——
  1. 快速路径（SBCL 且 *vt-linalg-simd-enabled* = t）：k 点积走
     sb-simd AVX/FMA 内核；
  2. 通用路径：原实现算法（逐元素顺序一致），纯标量 aref。
  *vt-linalg-simd-enabled* 控制路由，非正定报错与 upper 语义共享。"
  (assert (= 2 (vt-order matrix)))
  (let ((shape (vt-shape matrix)))
    (assert (= (first shape) (second shape))
            (matrix) "Cholesky 分解要求方阵"))
  (with-float-safe
    (let* ((n (first (vt-shape matrix)))
           (a (ensure-contiguous-2d-vt (vt-astype matrix :float64)))
           (l (vt-zeros (list n n) :dtype :float64))
           (a-data (vt-data a))
           (l-data (vt-data l))
           (a-s0 (first (vt-strides a)))
           (a-s1 (second (vt-strides a)))
           (a-off (vt-offset a))
           (l-s0 (first (vt-strides l)))
           (l-s1 (second (vt-strides l)))
           (l-off (vt-offset l)))
      (declare (type (simple-array double-float (*)) a-data l-data)
               (type fixnum n a-s0 a-s1 a-off l-s0 l-s1 l-off))
      ;; 引擎分派（ensure-contiguous/vt-zeros 保证 s1=1）
      (cond
        #+sbcl
        ((and *vt-linalg-simd-enabled* (= a-s1 1) (= l-s1 1))
         (sb-sys:with-pinned-objects (a-data l-data)
           (%vt-cholesky-core-avx (sb-sys:vector-sap a-data) a-off a-s0
                                  (sb-sys:vector-sap l-data) l-off n)))
        (t
         (%vt-cholesky-core-generic a-data a-off a-s0 l-data l-off n)))
      (if upper (vt-transpose l) l))))

;;; ============================================================
;;; 2. 对称矩阵特征值分解 (Jacobi 旋转法)
;;; ============================================================

;;; ============ Jacobi 特征值核心（通用 / AVX 双引擎）============
;;; 通用引擎：原 vt-eig 算法逐元素一致（仅 vt-ref 函数调用 → 直接
;;; aref，消除每元素 O(1) 调用与 stride 检查开销）。
;;; AVX 引擎：利用对称性改读「行」p/q（连续），双行旋转走 rot2 内核、
;;; 收敛判据走 dot 内核；V 以 V^T 形式累积（行旋转连续）；旋转后把
;;; 行值镜像写回列（保对称，O(n) 每旋转）。

(defun %vt-eig-core-generic (a-data v-data n max-iter tol)
  "Jacobi 旋转（通用路径：与原 vt-eig 逐元素一致，纯 aref）。"
  (declare (type (simple-array double-float (*)) a-data v-data)
           (type fixnum n max-iter)
           (type double-float tol)
           (optimize (speed 3) (safety 0) (debug 0) (space 0)))
  (loop for iter of-type fixnum from 0 below max-iter do
    ;; 收敛判据：上三角非对角平方和
    (let ((off-sum 0.0d0))
      (declare (type double-float off-sum))
      (loop for i of-type fixnum from 0 below n do
        (loop for j of-type fixnum from (1+ i) below n do
          (let ((val (aref a-data (+ (* i n) j))))
            (incf off-sum (* val val)))))
      (when (< (sqrt off-sum) tol) (return)))
    ;; 旋转扫描
    (loop for p of-type fixnum from 0 below (1- n) do
      (loop for q of-type fixnum from (1+ p) below n do
        (let ((apq (aref a-data (+ (* p n) q))))
          (declare (type double-float apq))
          (when (> (abs apq) tol)
            (let* ((app (aref a-data (+ (* p n) p)))
                   (aqq (aref a-data (+ (* q n) q)))
                   (theta (* 0.5d0 (atan (* 2.0d0 apq) (- app aqq))))
                   (c (cos theta))
                   (s (sin theta)))
              (declare (type double-float app aqq theta c s))
              ;; 对角与 pq 位置
              (setf (aref a-data (+ (* p n) q)) 0.0d0
                    (aref a-data (+ (* q n) p)) 0.0d0
                    (aref a-data (+ (* p n) p))
                    (+ (* c c app) (* 2.0d0 s c apq) (* s s aqq))
                    (aref a-data (+ (* q n) q))
                    (+ (* s s app) (* -2.0d0 s c apq) (* c c aqq)))
              ;; 非对角：r 行读列 p/q，写回 4 个对称位置（与原实现一致）
              (loop for r of-type fixnum from 0 below n
                    when (and (/= r p) (/= r q)) do
                      (let ((arp (aref a-data (+ (* r n) p)))
                            (arq (aref a-data (+ (* r n) q))))
                        (declare (type double-float arp arq))
                        (let ((na (+ (* c arp) (* s arq)))
                              (nb (+ (* (- s) arp) (* c arq))))
                          (declare (type double-float na nb))
                          (setf (aref a-data (+ (* r n) p)) na
                                (aref a-data (+ (* p n) r)) na
                                (aref a-data (+ (* r n) q)) nb
                                (aref a-data (+ (* q n) r)) nb))))
              ;; 特征向量：V 的 p/q 列
              (loop for r of-type fixnum from 0 below n do
                (let ((vp (aref v-data (+ (* r n) p)))
                      (vq (aref v-data (+ (* r n) q))))
                  (declare (type double-float vp vq))
                  (setf (aref v-data (+ (* r n) p))
                        (+ (* c vp) (* s vq))
                        (aref v-data (+ (* r n) q))
                        (+ (* (- s) vp) (* c vq))))))))))))

#+sbcl
(defun %vt-eig-core-avx (a-h a-data v-h v-data n max-iter tol)
  "Jacobi 旋转（AVX 路径：行对称访问 + rot2/dot 内核 + V^T 累积）。"
  (declare (type sb-sys:system-area-pointer a-h v-h)
           (type (simple-array double-float (*)) a-data v-data)
           (type fixnum n max-iter)
           (type double-float tol)
           (optimize (speed 3) (safety 0) (debug 0) (space 0)))
  (loop for iter of-type fixnum from 0 below max-iter do
    ;; 收敛判据：逐行上三角段 ‖a[i, i+1..n-1]‖²（dot 内核，行内连续）
    (let ((off-sum 0.0d0))
      (declare (type double-float off-sum))
      (loop for i of-type fixnum from 0 below n do
        (incf off-sum
              (%ln-f64-dot-avx a-h (+ (* i n) i) a-h (+ (* i n) i)
                               (- n i 1))))
      (when (< (sqrt off-sum) tol) (return)))
    ;; 旋转扫描
    (loop for p of-type fixnum from 0 below (1- n) do
      (loop for q of-type fixnum from (1+ p) below n do
        (let ((apq (aref a-data (+ (* p n) q))))
          (declare (type double-float apq))
          (when (> (abs apq) tol)
            (let* ((app (aref a-data (+ (* p n) p)))
                   (aqq (aref a-data (+ (* q n) q)))
                   (theta (* 0.5d0 (atan (* 2.0d0 apq) (- app aqq))))
                   (c (cos theta))
                   (s (sin theta)))
              (declare (type double-float app aqq theta c s))
              ;; 行 p/q 双行旋转（连续，rot2 内核；pq 处的值随后覆盖）
              (%ln-f64-rot2-avx a-h (* p n) a-h (* q n) n c s)
              ;; V^T 的 p/q 行旋转（= V 的 p/q 列，连续）
              (%ln-f64-rot2-avx v-h (* p n) v-h (* q n) n c s)
              ;; 对角与 pq 位置：覆盖 rot2 的临时值（公式与原实现一致）
              (setf (aref a-data (+ (* p n) p))
                    (+ (* c c app) (* 2.0d0 s c apq) (* s s aqq))
                    (aref a-data (+ (* q n) q))
                    (+ (* s s app) (* -2.0d0 s c apq) (* c c aqq))
                    (aref a-data (+ (* p n) q)) 0.0d0
                    (aref a-data (+ (* q n) p)) 0.0d0)
              ;; 镜像写回：行值 → 列（保对称，供后续列读与收敛扫描）
              (loop for r of-type fixnum from 0 below n
                    unless (or (= r p) (= r q)) do
                      (setf (aref a-data (+ (* r n) p))
                            (aref a-data (+ (* p n) r))
                            (aref a-data (+ (* r n) q))
                            (aref a-data (+ (* q n) r)))))))))))

(defun vt-eig (matrix &key (max-iter 200) (tol 1e-10) (uplo :lower))
  "对称矩阵特征值分解 (Jacobi 旋转法)，输入约定与 numpy.linalg.eigh 对齐。
   不校验对称性：仅使用 UPLO 指定的三角（:lower 对应 numpy 的
   UPLO='L' 默认值，:upper 对应 UPLO='U'），将其镜像覆盖另一三角后
   再做分解。对对称矩阵无任何影响；对非对称输入返回其单三角对称化
   矩阵的特征值（GIGO，与 numpy eigh 行为一致）。镜像在入口一次性
   完成，保证 AVX 与通用两条路径对任意输入使用完全相同的数据——
   修复了旧版对非对称输入两路径给出不同结果的问题。
   UPLO 仅接受 :lower / :upper（小写关键字，对应 numpy 的 'L'/'U'）。

  实现：双路径路由（同 vt-matmul/vt-svd 模式）——
  1. 快速路径（SBCL 且 *vt-linalg-simd-enabled* = t）：双行旋转与
     收敛判据走 sb-simd AVX/FMA 内核，V 以 V^T 形式累积；
  2. 通用路径：原 vt-eig 算法逐元素一致（完善：vt-ref 函数调用开销
     → 直接 aref），纯标量，无 SIMD 依赖。
  两条路径共享旋转公式、收敛判据、特征值降序排序与输出契约。"
  (assert (= 2 (vt-order matrix)))
  (let ((shape (vt-shape matrix)))
    (assert (= (first shape) (second shape))
            (matrix) "特征值分解要求方阵"))
  (with-float-safe
    (let* ((n (first (vt-shape matrix)))
           (a (ensure-contiguous-2d-vt (vt-copy (vt-astype matrix :float64))))
           (a-data (vt-data a))
           (v-tensor (vt-eye n :dtype :float64))
           (v-data (vt-data v-tensor))
           ;; AVX 模式下 v-data 存 V^T（行旋转连续）；通用模式存 V
           (avx-mode
             #+sbcl
             (and *vt-linalg-simd-enabled*
                  (= (second (vt-strides a)) 1) (= (second (vt-strides v-tensor)) 1))
             #-sbcl
             nil)
           ;; tol 收敛到 double-float（兼容旧默认值 1e-10 为 single-float 的情况）
           (tol-d (coerce tol 'double-float)))
      (declare (type (simple-array double-float (*)) a-data v-data)
               (type boolean avx-mode))
      ;; ---- numpy.linalg.eigh 对齐：单三角镜像（替代旧对称性校验）----
      ;; :lower（默认，numpy UPLO='L'）→ 下三角覆盖上三角；
      ;; :upper（numpy UPLO='U'）→ 上三角覆盖下三角。
      ;; a-data 是入口拷贝，原地镜像不影响调用方张量。
      (ecase uplo
        (:lower
         (loop for i of-type fixnum from 0 below n do
           (loop for j of-type fixnum from (1+ i) below n do
             (setf (aref a-data (+ (* i n) j))
                   (aref a-data (+ (* j n) i))))))
        (:upper
         (loop for i of-type fixnum from 0 below n do
           (loop for j of-type fixnum from (1+ i) below n do
             (setf (aref a-data (+ (* j n) i))
                   (aref a-data (+ (* i n) j)))))))
      (cond
        (avx-mode
         (sb-sys:with-pinned-objects (a-data v-data)
           (%vt-eig-core-avx (sb-sys:vector-sap a-data) a-data
                             (sb-sys:vector-sap v-data) v-data
                             n max-iter tol-d)))
        (t
         (%vt-eig-core-generic a-data v-data n max-iter tol-d)))
      ;; 特征值降序排序（stable，同原实现）+ 特征向量列重排
      (let ((pairs (loop for i of-type fixnum from 0 below n
                         collect (cons (aref a-data (+ (* i n) i)) i))))
        (setf pairs (sort pairs #'> :key #'car))
        (let ((vals (vt-from-sequence (mapcar #'car pairs) :dtype :float64))
              (vec (vt-zeros (list n n) :dtype :float64)))
          (loop for ni of-type fixnum from 0 below n
                for oi = (cdr (nth ni pairs))
                do (if avx-mode
                       ;; VT：vec 列 ni = VT 行 oi
                       (loop for r of-type fixnum from 0 below n do
                         (setf (aref (vt-data vec) (+ (* r n) ni))
                               (aref v-data (+ (* oi n) r))))
                       ;; V：vec 列 ni = V 列 oi
                       (loop for r of-type fixnum from 0 below n do
                         (setf (aref (vt-data vec) (+ (* r n) ni))
                               (aref v-data (+ (* r n) oi))))))
          (values vals vec))))))



(defun vt-pinv (matrix &key (rcond 1e-15))
  "Moore-Penrose 伪逆。"
  (assert (= 2 (vt-order matrix)))
  (with-float-safe
    (multiple-value-bind (u s vt-mat) (vt-svd matrix)
      (let* ((s-data (vt-data s))
             (s-size (vt-size s))
             (s-max (loop for i fixnum from 0 below s-size maximize (aref s-data i)))
             (cutoff (* rcond s-max))
             (s-pinv (vt-zeros (list s-size) :dtype :float64)))
        (loop for i fixnum from 0 below s-size do
          (when (> (aref s-data i) cutoff)
            (setf (vt-ref s-pinv i) (/ 1.0d0 (aref s-data i)))))
        (let* ((n (second (vt-shape vt-mat)))
               (k (vt-size s))
               (v-scaled (vt-zeros (list n k) :dtype :float64)))
          (loop for j fixnum from 0 below k do
            (let ((sc (vt-ref s-pinv j)))
              (loop for i fixnum from 0 below n do
                (setf (vt-ref v-scaled i j) (* (vt-ref vt-mat j i) sc)))))
          ;; v0.4.1（差分测试 R12）：Newton–Schulz 迭代精化
          ;; X ← X(2I − AX)，把 SVD 初始估计 ~1e-11 级的误差二次收敛到
          ;; 机器精度（对标 LAPACK 的 pinv 精度）。
          (let* ((x (vt-@ v-scaled (vt-transpose u)))
                 (m (first (vt-shape matrix)))
                 (eye2 (vt-* (vt-eye m :dtype :float64) 2.0d0)))
            (dotimes (_ 2 x)
              (setf x (vt-@ x (vt-- eye2 (vt-matmul matrix x)))))))))))

;;; ============================================================
;;; 4. 最小二乘
;;; ============================================================

(defun vt-lstsq (a b &key (rcond 1e-15))
  "最小二乘解 min ||Ax - b||_2。"
  (assert (= 2 (vt-order a)))
  (with-float-safe
    (let* ((m (first (vt-shape a)))
           (n (second (vt-shape a)))
           (b-vt (ensure-vt b))
           (b-shape (vt-shape b-vt))
           (nrhs (if (= (length b-shape) 1) 1 (second b-shape)))
           (b-mat (if (= (length b-shape) 1) (vt-reshape b-vt (list m 1)) b-vt)))
      (multiple-value-bind (u s vt-mat) (vt-svd a)
        (let* ((s-data (vt-data s))
               (k (vt-size s))
               (s-max (loop for i fixnum from 0 below k maximize (aref s-data i)))
               (cutoff (* rcond s-max))
               (rank (loop for i fixnum from 0 below k count (> (aref s-data i) cutoff)))
               (utb (vt-@ (vt-transpose u) b-mat))
               (x (vt-zeros (list k nrhs) :dtype :float64)))
          (loop for j fixnum from 0 below nrhs do
            (loop for i fixnum from 0 below rank do
              (setf (vt-ref x i j) (* (vt-ref utb i j) (/ 1.0d0 (aref s-data i))))))
          (setf x (vt-@ (vt-transpose vt-mat) x))
          (let ((res (if (and (> m n) (= rank n))
                         (vt-ref (vt-norm (vt-flatten (vt-- b-mat (vt-@ a x)))))
                         0.0d0))
                (xr (if (= nrhs 1) (vt-flatten (vt-slice x '(:all) '(0))) x)))
            (values xr res rank s)))))))
