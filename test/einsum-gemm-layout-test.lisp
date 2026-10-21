;;;; einsum-gemm-layout-test.lisp — einsum 的 GEMM 快通道（布局归一化）回归网
;;;;
;;;; 覆盖 CONVENTIONS §16.26 的四条路径：
;;;;   ① NN（A 行连续 + B 行主序 + C 行主序）→ SIMD 分块内核
;;;;   ② NT（B 按 (n,k) 存，如 "ij,kj->ik"）→ 分块转置成 (k,n) 后走 ①
;;;;   ③ C 非行主序（如 "ij,jk->ki"）→ 临时连续 C + 分块 scatter
;;;;   ④ 其余（A 非行连续等）→ 通用三重循环兜底
;;;;
;;;; 断言策略（不依赖外部 numpy：本套件要能离线跑）：
;;;;   · 主断言 = 与 **vt-matmul 逐位相等**。vt-matmul 走的是另一条入口
;;;;     （*SIMD-MATMUL-2D-FN*），两者最终同一内核 → 必须逐位一致；
;;;;     这条断言同时锁住「einsum 没有偷偷改语义」和「四条路径数值一致」。
;;;;   · 形状/布局类断言用代数恒等式（转置、结合律）表达，避免手写期望值
;;;;     （CONVENTIONS §10.4）。
;;;;   · 小矩阵（低于 *SIMD-MATMUL-THRESHOLD*）走标量内核，单独回归。
;;;;
;;;; 用法：sbcl --non-interactive --load test/einsum-gemm-layout-test.lisp

(require :asdf)
(defparameter *clvt-test-dir* (make-pathname :name nil :type nil :defaults *load-truename*))
(pushnew (merge-pathnames "../" *clvt-test-dir*) asdf:*central-registry* :test #'equal)
(handler-bind ((warning #'muffle-warning)) (asdf:load-system :clvt))

(in-package :clvt)

;;; ------------------------------------------------------------------
;;; 骨架
;;; ------------------------------------------------------------------

(defvar *pass* 0)
(defvar *fail* 0)
(defvar *failures* '())

(defun expect-true (name form)
  (if form
      (incf *pass*)
      (progn (incf *fail*) (push name *failures*) (format t "~&FAIL ~a~%" name))))

(defun report (name ok &optional detail)
  (if ok
      (incf *pass*)
      (progn (incf *fail*) (push name *failures*)
             (format t "~&FAIL ~a ~@[~a~]~%" name detail))))

;;; ------------------------------------------------------------------
;;; 数据构造：确定性伪随机（不用 random，保证可重放）
;;; ------------------------------------------------------------------

(defun mk (dims seed &key (dt :float64))
  (let* ((n (reduce #'* dims :initial-value 1))
         (t1 (make-vt dims 0 :dtype dt))
         (d (vt-data t1))
         (s (mod (abs seed) 2147483647)))
    (declare (type fixnum n))
    (dotimes (i n)
      (setf s (mod (+ (* s 1103515245) 12345) 2147483648))
      (case dt
        (:float64 (setf (aref d i) (- (/ s 2147483648.0d0) 0.5d0)))
        (:float32 (setf (aref d i) (coerce (- (/ s 2147483648.0d0) 0.5d0)
                                           'single-float)))
        (:complex128 (setf (aref d i)
                           (complex (- (/ s 2147483648.0d0) 0.5d0) 0.25d0)))
        (:complex64 (setf (aref d i)
                          (complex (coerce (- (/ s 2147483648.0d0) 0.5d0)
                                           'single-float)
                                   0.25)))
        (:int64 (setf (aref d i) (- (mod s 1000) 500)))
        (:int32 (setf (aref d i) (- (mod s 1000) 500)))))
    t1))

(defun flat (t1)
  "展开成 double-float 列表（复数按 实部/虚部 交错）。"
  (let* ((c (vt-contiguous (vt-flatten t1)))
         (d (vt-data c))
         (n (length d))
         (o '()))
    (dotimes (i n)
      (let ((v (aref d i)))
        (if (complexp v)
            (progn (push (coerce (realpart v) 'double-float) o)
                   (push (coerce (imagpart v) 'double-float) o))
            (push (coerce v 'double-float) o))))
    (nreverse o)))

(defun max-abs-diff (a b)
  (if (/= (length a) (length b))
      most-positive-double-float
      (let ((w 0d0))
        (loop for x in a for y in b do (setf w (max w (abs (- x y)))))
        w)))

;;; ------------------------------------------------------------------
;;; 1) NN：einsum "ij,jk->ik" ≡ vt-matmul（逐位）
;;; ------------------------------------------------------------------

(defun run-nn ()
  (format t "~&[NN] ij,jk->ik ≡ vt-matmul（逐位）~%")
  (dolist (sz '(8 64 128 256))
    (let* ((a (mk (list sz sz) (+ sz 1)))
           (b (mk (list sz sz) (+ sz 2)))
           (got (flat (vt-einsum "ij,jk->ik" a b)))
           (want (flat (vt-matmul a b))))
      (report (format nil "NN ~ax~a 逐位相等" sz sz) (equalp got want))))
  ;; 形状保持
  (let* ((a (mk '(37 53) 1)) (b (mk '(53 29) 2))
         (r (vt-einsum "ij,jk->ik" a b)))
    (report "NN 非方阵形状" (equal (vt-shape r) '(37 29)))
    (report "NN dtype 保持" (eq (vt-dtype r) :float64))))

;;; ------------------------------------------------------------------
;;; 2) NT：ij,kj->ik ≡ a @ bᵀ
;;; ------------------------------------------------------------------

(defun run-nt ()
  (format t "~&[NT] ij,kj->ik ≡ a @ bᵀ~%")
  (dolist (sz '(8 64 128 256))
    (let* ((a (mk (list sz sz) (+ sz 3)))
           (bt (mk (list sz sz) (+ sz 4)))
           (got (flat (vt-einsum "ij,kj->ik" a bt)))
           (want (flat (vt-matmul a (vt-transpose bt)))))
      (report (format nil "NT ~ax~a 逐位相等" sz sz) (equalp got want))))
  ;; 非方阵的 NT
  (let* ((a (mk '(11 23) 5)) (bt (mk '(17 23) 6))
         (got (flat (vt-einsum "ij,kj->ik" a bt)))
         (want (flat (vt-matmul a (vt-transpose bt)))))
    (report "NT 非方阵（11x23 · (17x23)ᵀ）" (equalp got want))))

;;; ------------------------------------------------------------------
;;; 3) C 非行主序：ij,jk->ki ≡ (a @ b)ᵀ
;;; ------------------------------------------------------------------

(defun run-c-transposed ()
  (format t "~&[C 转置] ij,jk->ki ≡ (a @ b)ᵀ~%")
  (dolist (sz '(8 64 128 256))
    (let* ((a (mk (list sz sz) (+ sz 7)))
           (b (mk (list sz sz) (+ sz 8)))
           (got (flat (vt-einsum "ij,jk->ki" a b)))
           (want (flat (vt-transpose (vt-einsum "ij,jk->ik" a b)))))
      (report (format nil "C 转置 ~ax~a 逐位相等" sz sz) (equalp got want))))
  ;; 非方阵：m=n 时转置看不出错位，这里用 m≠n 锁住索引
  (let* ((a (mk '(13 7) 9)) (b (mk '(7 19) 10))
         (r (vt-einsum "ij,jk->ki" a b)))
    (report "C 转置形状 (19x13)" (equal (vt-shape r) '(19 13)))
    (report "C 转置数值"
            (equalp (flat r) (flat (vt-transpose (vt-matmul a b)))))))

;;; ------------------------------------------------------------------
;;; 4) 兜底与其它布局
;;; ------------------------------------------------------------------

(defun run-other ()
  (format t "~&[其它布局] 转置输入 / 切片 / :out / 小矩阵~%")
  (let* ((a (mk '(96 96) 11)) (b (mk '(96 96) 12)))
    ;; A 转置：sa-j ≠ 1，走 ④ 兜底
    (report "A 转置输入"
            (equalp (flat (vt-einsum "ij,jk->ik" (vt-transpose a) b))
                    (flat (vt-matmul (vt-transpose a) b))))
    ;; 切片输入（offset ≠ 0）
    (report "A 带 offset 的切片"
            (let ((s (vt-slice a '(3 96) '(0 96))))
              (equalp (flat (vt-einsum "ij,jk->ik" s b))
                      (flat (vt-matmul s b)))))
    ;; :out 预填非零 → einsum 必须覆盖而不是累加（§4.2）
    (let ((o (vt-+ (vt-zeros '(96 96) :dtype :float64) 999.0d0)))
      (vt-einsum "ij,jk->ik" a b :out o)
      (report ":out 预填 999 被覆盖" (equalp (flat o) (flat (vt-matmul a b)))))
    ;; 低于 SIMD 阈值 → 两条路都走标量内核
    (let* ((s (mk '(16 16) 13)) (t2 (mk '(16 16) 14)))
      (report "16²（低于 SIMD 阈值）"
              (equalp (flat (vt-einsum "ij,jk->ik" s t2))
                      (flat (vt-matmul s t2)))))))

;;; ------------------------------------------------------------------
;;; 5) 批量（含广播、批量 NT）
;;; ------------------------------------------------------------------

(defun run-batch ()
  (format t "~&[批量] bij,bjk->bik / 广播 / 批量 NT~%")
  ;; 单个 batch 的 m·k·n = 32768 < *SIMD-MATMUL-THRESHOLD*(50000)，本侧走
  ;; 标量内核；而 vt-matmul 的批量通道按**总工作量**（batch·m·k·n）判阈值，
  ;; 走 SIMD。两条内核的累加顺序不同 → 只能比容差，不能逐位。
  (let ((a (mk '(4 32 32) 21)) (b (mk '(4 32 32) 22)))
    (let ((d (max-abs-diff (flat (vt-einsum "bij,bjk->bik" a b))
                           (flat (vt-matmul a b)))))
      (report "bij,bjk->bik 4x32x32（容差 1e-10）" (< d 1d-10)
              (format nil "maxabs=~,3e" d))))
  ;; 每 batch 都超过 SIMD 阈值 → 两侧都走 SIMD，应逐位相等
  (let ((a (mk '(2 64 64) 27)) (b (mk '(2 64 64) 28)))
    (report "bij,bjk->bik 2x64x64（每 batch 超阈值，逐位）"
            (equalp (flat (vt-einsum "bij,bjk->bik" a b))
                    (flat (vt-matmul a b)))))
  (let ((a (mk '(3 48 48) 23)) (b (mk '(48 48) 24)))
    ;; vt-matmul 走 *SIMD-BATCHED-MATMUL-FN*（批间并行），累加顺序不同，
    ;; 故只要求相对差 ≤ 1e-10，不要求逐位。
    (let* ((got (flat (vt-einsum "bij,jk->bik" a b)))
           (want (flat (vt-matmul a b))))
      (report "bij,jk->bik 广播（容差 1e-10）"
              (< (max-abs-diff got want) 1d-10)
              (format nil "maxabs=~,3e" (max-abs-diff got want)))))
  (let ((a (mk '(2 32 32) 25)) (b (mk '(2 32 32) 26)))
    (let ((d (max-abs-diff (flat (vt-einsum "bij,bkj->bik" a b))
                           (flat (vt-matmul a (vt-transpose b '(0 2 1)))))))
      (report "bij,bkj->bik（批量 NT，容差 1e-10）" (< d 1d-10)
              (format nil "maxabs=~,3e" d)))))

;;; ------------------------------------------------------------------
;;; 6) 其它 dtype（f32 / i64 / i32 / 复数）
;;; ------------------------------------------------------------------

(defun run-dtypes ()
  (format t "~&[dtype] f32 / i64 / i32 / c128 / c64~%")
  (let ((a (mk '(64 64) 31 :dt :float32)) (b (mk '(64 64) 32 :dt :float32)))
    (report "f32 ij,jk->ik"
            (equalp (flat (vt-einsum "ij,jk->ik" a b))
                    (flat (vt-matmul a b)))))
  (let ((a (make-vt '(64 64) 3 :dtype :int64))
        (b (make-vt '(64 64) 5 :dtype :int64)))
    (report "i64 ij,jk->ik"
            (equalp (flat (vt-einsum "ij,jk->ik" a b))
                    (flat (vt-matmul a b)))))
  (let ((a (make-vt '(64 64) 3 :dtype :int32))
        (b (make-vt '(64 64) 5 :dtype :int32)))
    (report "i32 ij,jk->ik"
            (equalp (flat (vt-einsum "ij,jk->ik" a b))
                    (flat (vt-matmul a b)))))
  ;; 复数走 *CX-EINSUM-GEMM-FN*，不在本轮改动范围内，但要确认没被影响
  (let ((a (mk '(64 64) 41 :dt :complex128)) (b (mk '(64 64) 42 :dt :complex128)))
    (report "c128 ij,jk->ik"
            (equalp (flat (vt-einsum "ij,jk->ik" a b))
                    (flat (vt-matmul a b))))
    (report "c128 ij,jk->ki"
            (equalp (flat (vt-einsum "ij,jk->ki" a b))
                    (flat (vt-transpose (vt-matmul a b))))))
  (let ((a (mk '(64 64) 43 :dt :complex64)) (b (mk '(64 64) 44 :dt :complex64)))
    (report "c64 ij,jk->ik"
            (equalp (flat (vt-einsum "ij,jk->ik" a b))
                    (flat (vt-matmul a b))))))

;;; ------------------------------------------------------------------
;;; 7) SIMD 关 / 线程关 时结果不变（开关只影响速度）
;;; ------------------------------------------------------------------

(defun run-switches ()
  ;; 开关只应影响速度，不应影响结果 —— 但「逐位」不成立：SIMD 内核用 FMA
  ;; （少一次舍入），标量内核用 mul+add。这一条对 vt-matmul 本来就成立，
  ;; 因此这里断言的是「相对差 ≤ 1e-12」，而不是逐位相等（§16.26.3）。
  (format t "~&[开关] SIMD 关 / 单线程 结果不变（容差 1e-12）~%")
  (let* ((a (mk '(128 128) 51)) (b (mk '(128 128) 52))
         (ref (flat (vt-einsum "ij,jk->ik" a b)))
         (ref-t (flat (vt-einsum "ij,jk->ki" a b)))
         (ref-nt (flat (vt-einsum "ij,kj->ik" a (vt-transpose b)))))
    (flet ((same (name got want)
             (let ((d (max-abs-diff got want)))
               (report name (< d 1d-12) (format nil "maxabs=~,3e" d)))))
      (let ((*simd-matmul-enabled* nil))
        (same "SIMD 关：ij,jk->ik 不变" (flat (vt-einsum "ij,jk->ik" a b)) ref)
        (same "SIMD 关：ij,jk->ki 不变" (flat (vt-einsum "ij,jk->ki" a b)) ref-t)
        (same "SIMD 关：ij,kj->ik 不变"
              (flat (vt-einsum "ij,kj->ik" a (vt-transpose b))) ref-nt))
      (let ((*simd-matmul-thread-count* 1))
        (same "单线程：ij,jk->ik 不变" (flat (vt-einsum "ij,jk->ik" a b)) ref)
        (same "单线程：ij,jk->ki 不变" (flat (vt-einsum "ij,jk->ki" a b)) ref-t)))))

;;; ------------------------------------------------------------------
;;; 入口
;;; ------------------------------------------------------------------

(format t "~&=====================================~%")
(format t "einsum GEMM 布局归一化测试（§16.26）~%")
(format t "=====================================~%")
(run-nn)
(run-nt)
(run-c-transposed)
(run-other)
(run-batch)
(run-dtypes)
(run-switches)
(format t "~&=====================================~%")
(format t "Total: ~d / Pass: ~d / Fail: ~d~%" (+ *pass* *fail*) *pass* *fail*)
(format t "PASS=~d FAIL=~d~%" *pass* *fail*)
(when *failures*
  (format t "~&--- 失败清单 ---~%")
  (dolist (f (reverse *failures*)) (format t "  · ~a~%" f)))
(when (plusp *fail*) (sb-ext:exit :code 1))
