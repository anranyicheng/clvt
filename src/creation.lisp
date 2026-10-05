;;;; creation.lisp — 张量创建

(in-package :clvt)

(defun vt-zeros (shape &key (dtype :float64))
  "创建全 0 张量。SHAPE 为维度列表（NIL 表示 0 维标量）。"
  (make-vt shape 0 :dtype dtype))

(defun vt-ones (shape &key (dtype :float64))
  "创建全 1 张量。SHAPE 为维度列表（NIL 表示 0 维标量）。"
  (make-vt shape 1 :dtype dtype))

(defun vt-const (shape value &key (dtype :float64))
  "创建以 VALUE 填充的张量。等价 numpy.full(shape, value)。"
  (make-vt shape value :dtype dtype))

(defun vt-full (shape fill-value &key (dtype :float64))
  "创建以 FILL-VALUE 填充的张量（对标 numpy.full）。"
  (make-vt shape fill-value :dtype dtype))

(defun vt-empty (shape &key (dtype :float64))
  "创建「未初始化」张量（对标 numpy.empty）。本库以 0 填充，故与 vt-zeros 等价。"
  (vt-zeros shape :dtype dtype))

(defun vt-zeros-like (vt &key dtype)
  "创建与 VT 形状/（缺省）dtype 相同的全 0 张量（对标 numpy.zeros_like）。"
  (vt-zeros (vt-shape vt) :dtype (or dtype (vt-dtype vt))))

(defun vt-ones-like (vt &key dtype)
  "创建与 VT 形状/（缺省）dtype 相同的全 1 张量（对标 numpy.ones_like）。"
  (vt-ones (vt-shape vt) :dtype (or dtype (vt-dtype vt))))

(defun vt-full-like (vt fill-value &key dtype)
  "创建与 VT 形状/（缺省）dtype 相同、以 FILL-VALUE 填充的张量（对标 numpy.full_like）。"
  (vt-full (vt-shape vt) fill-value :dtype (or dtype (vt-dtype vt))))

(defun vt-empty-like (vt &key dtype)
  "创建与 VT 形状/（缺省）dtype 相同的未初始化张量（本库以 0 填充）。"
  (vt-empty (vt-shape vt) :dtype (or dtype (vt-dtype vt))))

(defun vt-identity (n &key (dtype :float64))
  "创建 N×N 单位阵（对标 numpy.identity）。等价 (vt-eye n :dtype dtype)。

缺省 dtype 为 :float64；若显式传入，则必须是合法 dtype 符号。
修复：原先 `&key dtype` 未给缺省值时 dtype 为 NIL，直接下传 vt-eye 会
触发 `NIL fell through ECASE expression`，故此处对齐 vt-eye 的缺省值。"
  (vt-eye n :dtype dtype))

(defun vt-eye (rows &key (cols rows) (k 0) (value 1) (dtype :float64))
  "创建单位/对角矩阵，对标 NumPy np.eye。"
  (declare (type fixnum rows cols k))
  (let* ((shape (list rows cols))
         (lisp-type (vt-dtype->lisp-type dtype))
         (data (make-array (* rows cols) :element-type lisp-type
                                          :initial-element (coerce 0 lisp-type)))
         (res (%make-vt :data data :shape shape
                        :strides (vt-compute-strides shape)
                        :offset 0 :dtype dtype)))
    (let* ((row-stride (first (vt-strides res)))
           (col-stride (second (vt-strides res)))
           (r-start (max 0 (- k)))
           (c-start (max 0 k))
           (diag-len (max 0 (min (- rows r-start)
				 (- cols c-start)))))
      (when (> diag-len 0)
        (let ((start-offset (+ (* r-start row-stride)
			       (* c-start col-stride))))
          (loop for i fixnum from 0 below diag-len
                for offset fixnum = start-offset
		  then (+ offset row-stride col-stride)
                do (setf (aref data offset) (vt-cast value dtype)))))
      res)))

(defun vt-arange (total-num &key (start 0) (step 1) (dtype :float64))
  "创建包含 total-num 个元素的等差数列一维张量。

优化：按 dtype 特化填充循环，避免逐元素 vt-cast 的 ecase 分派开销。"
  (declare (fixnum total-num))
  (when (and (numberp step) (zerop step))
    (error "vt-arange: step 不能为 0"))
  (let* ((data (make-array total-num :element-type (vt-dtype->lisp-type dtype)))
         (shape (list total-num)))
    (ecase dtype
      (:float64
       (let ((s (coerce start 'double-float))
             (d (coerce step 'double-float)))
         (declare (double-float s d))
         (loop for i fixnum below total-num
               do (setf (aref data i) (+ s (* d (coerce i 'double-float)))))))
      (:float32
       (let ((s (coerce start 'single-float))
             (d (coerce step 'single-float)))
         (declare (single-float s d))
         (loop for i fixnum below total-num
               do (setf (aref data i) (+ s (* d (coerce i 'single-float)))))))
      ((:int64 :int32)
       (let ((s (truncate start)) (d (truncate step)))
         (declare (fixnum s d))
         ;; 与 int16/int8/uint 路径一致：溢出按 NumPy 语义回绕（§8.5），
         ;; 避免大参数下 (setf aref) 触发数组元素类型错误
         (loop for i fixnum below total-num
               do (setf (aref data i)
                        (if (eq dtype :int64)
                            (%wrap-int64 (+ s (* i d)))
                            (%wrap-int32 (+ s (* i d))))))))
      (:int16
       (let ((s (truncate start)) (d (truncate step)))
         (declare (fixnum s d))
         (loop for i fixnum below total-num
               do (setf (aref data i) (%wrap-int16 (+ s (* i d)))))))
      (:int8
       (let ((s (truncate start)) (d (truncate step)))
         (declare (fixnum s d))
         (loop for i fixnum below total-num
               do (setf (aref data i) (%wrap-int8 (+ s (* i d)))))))
      (:uint8
       (let ((s (truncate start)) (d (truncate step)))
         (declare (fixnum s d))
         (loop for i fixnum below total-num
               do (setf (aref data i) (%wrap-uint8 (+ s (* i d)))))))
      (:uint16
       (let ((s (truncate start)) (d (truncate step)))
         (declare (fixnum s d))
         (loop for i fixnum below total-num
               do (setf (aref data i) (%wrap-uint16 (+ s (* i d))))))))
    (%make-vt :data data :shape shape :strides (vt-compute-strides shape)
              :offset 0 :dtype dtype)))

(defun vt-linspace (start end num &key (endpoint t) (dtype :float64))
  "创建线性间隔数组，对标 numpy.linspace。

优化：按 dtype 特化填充循环，避免逐元素 vt-cast 的 ecase 分派开销。"
  (when (<= num 0) (error "num 必须大于 0，当前值为 ~d" num))
  (when (= num 1)
    (return-from vt-linspace
      (make-vt (list 1) (vt-cast start dtype) :dtype dtype)))
  (let* ((div (if endpoint (1- num) num))
         (step (/ (- end start) div)))
    (let ((data (make-array num :element-type (vt-dtype->lisp-type dtype))))
      (ecase dtype
        (:float64
         (let ((s (coerce start 'double-float))
	       (d (coerce step 'double-float)))
           (declare (double-float s d))
           (loop for i fixnum below num
                 do (setf (aref data i) (+ s (* d (coerce i 'double-float)))))))
        (:float32
         ;; 与 NumPy 一致：float32 linspace 以 double 精度计算后舍入存储，
         ;; 避免长序列的 float32 累积漂移
         (let ((s (coerce start 'double-float))
	       (d (coerce step 'double-float)))
           (declare (double-float s d))
           (loop for i fixnum below num
                 do (setf (aref data i)
                          (coerce (+ s (* d (coerce i 'double-float)))
                                  'single-float)))))
        ((:int64 :int32 :int16 :int8 :uint8 :uint16)
         (loop for i fixnum below num
               do (setf (aref data i) (vt-cast (+ start (* i step)) dtype)))))
      (when endpoint
        (setf (aref data (1- num)) (vt-cast end dtype)))
      (%make-vt :data data :shape (list num) :strides '(1) :offset 0 :dtype dtype))))

(defun vt-logspace (start stop num &key (base 10.0d0) (endpoint t) (dtype :float64))
  "创建对数间隔的一维张量。"
  (vt-map (lambda (x) (expt base x))
          (vt-linspace start stop num :endpoint endpoint :dtype dtype)))

(defun vt-from-array (arr &key (dtype nil) (fast nil))
  "从标准 CL 多维数组创建张量（保持维度）。
   fast t 意味着直接使用 coerce 转换，可能报错
        nil 则用 vt-cast 安全转换(默认)
   未显式指定 :dtype 时按数组元素类型精确推断（覆盖逻辑层 dtype 全集）：
   v0.3.5 修复——旧实现对 (signed-byte 16/8) 与 (unsigned-byte 8/16)
   数组一律静默提升为 :int32/:float64，违背 dtype 单一事实来源原则，
   现按最小精确类型推断（值域严格包含数组元素类型时才选用）。"
  (let* ((shape (array-dimensions arr))
         (size (vt-shape-to-size shape))
         (cl-etype (array-element-type arr))
         (infer (cond
                  ;; 窄类型优先：signed-byte 8 ⊂ 16 ⊂ 32 ⊂ 64，
                  ;; subtypep 判定"数组元素类型是某 dtype 值域的子集"，
                  ;; 最先匹配者即最小精确 dtype。
                  ((subtypep cl-etype 'double-float) :float64)
                      ((subtypep cl-etype 'single-float) :float32)
                  ((subtypep cl-etype '(signed-byte 8))   :int8)
                  ((subtypep cl-etype '(unsigned-byte 8))  :uint8)
                  ((subtypep cl-etype '(signed-byte 16))  :int16)
                  ((subtypep cl-etype '(unsigned-byte 16)) :uint16)
                      ((subtypep cl-etype '(signed-byte 32)) :int32)
                      ((subtypep cl-etype '(signed-byte 64)) :int64)
                      ((subtypep cl-etype 'fixnum) :int64)
                      (t :float64)))
         (final (or dtype infer))
	 (lisp-type (vt-dtype->lisp-type final))
         (data (make-array size :element-type lisp-type)))
    (if fast
	(dotimes (i size)
	  (setf (aref data i) (coerce (row-major-aref arr i) lisp-type)))
	(dotimes (i size)
	  (setf (aref data i) (vt-cast (row-major-aref arr i) final))))    
    (%make-vt :data data :shape shape :strides (vt-compute-strides shape)
              :offset 0 :dtype final)))

(defun vt-from-function (shape fn &key (dtype :float64))
  "根据函数创建张量：fn 接收索引列表并返回元素值。"
  (let* ((size (vt-shape-to-size shape))
         (data (make-array size :element-type (vt-dtype->lisp-type dtype)))
         (result (%make-vt :data data :shape shape
                           :strides (vt-compute-strides shape) :offset 0 :dtype dtype))
         (rank (length shape)))
    (labels ((recurse (depth indices flat-idx)
               (if (= depth rank)
                   (setf (aref data flat-idx) (vt-cast (funcall fn indices) dtype))
                   (let ((dim (nth depth shape))
			 (stride (nth depth (vt-strides result))))
                     (loop for i from 0 below dim
                           do (recurse (1+ depth) (append indices (list i))
                                       (+ flat-idx (* i stride))))))))
      (recurse 0 nil 0))
    result))

(defun vt-kron (a b)
  "Kronecker 积（对标 numpy.kron）。"
  (let ((a-shape (vt-shape a))
	(b-shape (vt-shape b)))
    (when (null a-shape)
      (setf a-shape '(1)))
    (when (null b-shape)
      (setf b-shape '(1)))
    (let* ((nda (length a-shape))
	   (ndb (length b-shape))
           (max-ndim (max nda ndb))
           (a-pad (append (make-list (- max-ndim nda) :initial-element 1) a-shape))
           (b-pad (append (make-list (- max-ndim ndb) :initial-element 1) b-shape))
           (a-new '()) (b-new '()) (final '()))
      (loop for da in a-pad for db in b-pad
            do (push da a-new) (push 1 a-new)
               (push 1 b-new) (push db b-new)
               (push (* da db) final))
      (setf a-new (nreverse a-new))
      (setf b-new (nreverse b-new))
      (setf final (nreverse final))
      (let ((a-r (vt-reshape a a-new))
	    (b-r (vt-reshape b b-new)))
        (vt-view (vt-* a-r b-r) final)))))

(defun vt-meshgrid (vts-list &key (indexing :xy) (sparse nil) (copy t))
  "生成坐标网格（对标 numpy.meshgrid）。"
  (dolist (v vts-list)
    (assert (= (length (vt-shape v)) 1) (v) "meshgrid 输入必须为 1d"))
  (let* ((nd (length vts-list))
         (dims (mapcar (lambda (v) (first (vt-shape v))) vts-list))
         (output-shape (if (and (eq indexing :xy) (>= nd 2))
                           (let ((sh (copy-list dims)))
			     (rotatef (first sh) (second sh)) sh)
                           dims))
         (target-axes (if (and (eq indexing :xy) (>= nd 2))
                          (let ((axes (loop for i below nd collect i)))
                            (rotatef (first axes) (second axes)) axes)
                          (loop for i below nd collect i))))
    (labels ((sparse-shape (i)
               (loop for ax from 0 below nd
		     collect (if (= ax (nth i target-axes))
				 (nth i dims)
				 1))))
      (loop for i from 0 below nd for v in vts-list
            for src = (if copy (vt-copy v) v)
            for sp = (sparse-shape i)
            collect (if sparse
			(vt-reshape src sp)
                        (vt-broadcast-to (vt-reshape src sp) output-shape))))))
