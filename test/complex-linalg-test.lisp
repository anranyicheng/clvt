;;;; complex-linalg-test.lisp — 复数线性代数测试（CONVENTIONS §16.19）
;;;;
;;;; 两层断言：
;;;;   1. **差分对照**：期望值全部由 test/gen_complex_linalg_expected.py
;;;;      从本机 numpy 实测生成（§10.4 禁手写期望值）；
;;;;   2. **结构恒等式**：重构 / 正交 / 酉性质。这一类断言不依赖具体
;;;;      数值约定（QR 的 R、SVD 的 U/V 都带相位自由度，逐元素比对
;;;;      没有意义），是真正能发现算法错误的检查。

(require :asdf)
(defparameter *clvt-test-dir* (make-pathname :name nil :type nil :defaults *load-truename*))
(pushnew (merge-pathnames "../" *clvt-test-dir*) asdf:*central-registry* :test #'equal)
(handler-bind ((warning #'muffle-warning)) (asdf:load-system :clvt))
(in-package :clvt)

(defvar *pass* 0)
(defvar *fail* 0)
(defvar *failures* nil)

(defun expect-true (name ok)
  (if ok
      (incf *pass*)
      (progn (incf *fail*)
             (push name *failures*)
             (format t "~&FAIL ~a~%" name))))

(defun expect-equal (name got want)
  (if (equal got want)
      (incf *pass*)
      (progn (incf *fail*)
             (push name *failures*)
             (format t "~&FAIL ~a: got ~a, want ~a~%" name got want))))

;;; ------------------------------------------------------------------
;;; 输入数据（与 gen_complex_linalg_expected.py 一一对应）
;;; ------------------------------------------------------------------

(defun z (x) (coerce x '(complex double-float)))

(defparameter cx-la-a (vt-from-sequence
                       (list (list (z #C(2 1)) (z #C(1 0)))
                             (list (z #C(1 -1)) (z #C(3 2))))
                       :dtype :complex128))
(defparameter cx-la-h (vt-from-sequence
                       (list (list (z #C(4 0)) (z #C(1 1)))
                             (list (z #C(1 -1)) (z #C(3 0))))
                       :dtype :complex128))
(defparameter cx-la-w (vt-from-sequence
                       (list (list (z #C(1 1)) (z #C(2 0)) (z #C(0 -1)))
                             (list (z #C(3 0)) (z #C(1 2)) (z #C(1 0))))
                       :dtype :complex128))
(defparameter cx-la-t (vt-transpose cx-la-w))
(defparameter cx-la-b1 (vt-from-sequence (list (z #C(1 0)) (z #C(2 0))) :dtype :complex128))
(defparameter cx-la-b2 (vt-eye 2 :dtype :complex128))
(defparameter cx-la-b3 (vt-from-sequence (list (z #C(1 0)) (z #C(2 0)) (z #C(3 0)))
                                         :dtype :complex128))
(defparameter cx-la-v1 (vt-from-sequence (list (z #C(1 2)) (z #C(3 -4)) (z #C(5 0)))
                                         :dtype :complex128))
(defparameter cx-la-v2 (vt-from-sequence (list (z #C(2 -1)) (z #C(1 1)) (z #C(0 2)))
                                         :dtype :complex128))
(defparameter cx-la-m2 (vt-from-sequence
                        (list (list (z #C(1 1)) (z #C(2 0)))
                              (list (z #C(3 0)) (z #C(0 4))))
                        :dtype :complex128))
(defparameter cx-la-sing (vt-from-sequence
                          (list (list (z #C(1 0)) (z #C(2 0)))
                                (list (z #C(2 0)) (z #C(4 0))))
                          :dtype :complex128))

;;; ------------------------------------------------------------------
;;; 期望值读取与比较
;;; ------------------------------------------------------------------

(defun normalize-real (x)
  (case x
    (:nan +vt-dfloat-nan+)
    (:inf +vt-dfloat-pos-inf+)
    (:neg-inf +vt-dfloat-neg-inf+)
    (t x)))

(defun normalize-scalar (x)
  (if (and (consp x) (eq (first x) :cx))
      (complex (normalize-real (second x))
               (normalize-real (third x)))
      (normalize-real x)))

(defun flatten-list (v)
  (cond ((null v) nil)
        ((listp v) (mapcan #'flatten-list v))
        (t (list v))))

(defun flatten-value (v)
  "期望值 → 扁平数值列表；(:CX re im) 是**标量**不是列表。"
  (cond ((null v) nil)
        ((and (consp v) (eq (first v) :cx)) (list (normalize-scalar v)))
        ((listp v) (mapcan #'flatten-value v))
        (t (list (normalize-scalar v)))))

(defun approx= (a b &optional (tol 1d-9))
  "相对容差比较（含复数、NaN、±Inf）。"
  (with-float-safe
    (cond
      ((and (numberp a) (numberp b)
            (or (and (floatp a) (/= a a)) (and (floatp b) (/= b b))))
       nil)
      ((and (or (complexp a) (complexp b)))
       (let ((aa (if (complexp a) a (complex a 0)))
             (bb (if (complexp b) b (complex b 0))))
         (and (approx= (realpart aa) (realpart bb) tol)
              (approx= (imagpart aa) (imagpart bb) tol))))
      ((and (floatp a) (floatp b) (or (> (abs a) 1d300) (> (abs b) 1d300)))
       (= (signum a) (signum b)))
      (t
       (let* ((x (coerce a 'double-float))
              (y (coerce b 'double-float))
              (scale (max 1d0 (abs y))))
         (<= (abs (- x y)) (* tol scale)))))))

(defvar *cases* nil)
(defvar *generator* nil)

(defparameter *expected-file*
  (merge-pathnames "complex-linalg-expected.sexp" cl-user::*clvt-test-dir*)
  "由 gen_complex_linalg_expected.py 生成的 numpy 实测期望值。")

(with-open-file (s *expected-file*)
  (let ((data (read s)))
    (setf *generator* (getf data :generator)
          *cases* (getf data :cases))))

(defun run-differential ()
  (format t "~&--- 差分对照（numpy ~a，~d 例）~%" *generator* (length *cases*))
  (dolist (c *cases*)
    (let ((name (getf c :name))
          (form (getf c :form))
          (value (getf c :value))
          (kind (getf c :kind))
          (tol (or (getf c :tol) 1d-9)))
      (handler-case
          (let ((got (eval (read-from-string form))))
            (if (eq kind :symbol)
                (expect-equal name got value)
                (let ((items (flatten-list (if (vt-p got) (vt-to-list got) got)))
                      (wants (flatten-value value)))
                  (cond
                    ((/= (length items) (length wants))
                     (incf *fail*)
                     (push name *failures*)
                     (format t "~&FAIL ~a: 元素数 ~d ≠ 期望 ~d~%"
                             name (length items) (length wants)))
                    (t
                     (expect-true name
                                  (every (lambda (g w) (approx= g w tol))
                                         items wants)))))))
        (error (e)
          (incf *fail*)
          (push name *failures*)
          (format t "~&FAIL ~a: 报错 ~a~%" name e))))))

;;; ------------------------------------------------------------------
;;; 结构恒等式（不依赖相位/排序约定）
;;; ------------------------------------------------------------------

(defun mabs-diff (a b)
  "两个张量逐元素最大 |差|。"
  (let* ((fa (vt-flatten (ensure-vt a)))
         (fb (vt-flatten (ensure-vt b)))
         (n (vt-size fa))
         (best 0.0d0))
    (dotimes (i n best)
      (let ((x (vt-item (vt-slice fa (list i (1+ i)))))
            (y (vt-item (vt-slice fb (list i (1+ i))))))
        (setf best (max best (abs (- x y))))))))

(defun eye-like (n &key (dtype :complex128))
  (vt-eye n :dtype dtype))

(defun diag-complex (vals)
  "1 维实数/复数张量 → 对角复矩阵。"
  (let* ((k (vt-size vals))
         (d (vt-zeros (list k k) :dtype :complex128)))
    (dotimes (i k)
      (setf (vt-ref d i i) (let ((v (vt-item (vt-slice vals (list i (1+ i))))))
                             (if (complexp v) v (complex (coerce v 'double-float) 0.0d0)))))
    d))

(defun structural-tests ()
  (format t "~&--- 结构恒等式（重构 / 正交 / 酉）~%")

  ;; ---- LU：P·A = L·U ----
  (dolist (m (list (cons "sq" cx-la-a) (cons "tall" cx-la-t) (cons "wide" cx-la-w)))
    (let* ((mat (cdr m))
           (n (second (vt-shape mat)))
           (mr (first (vt-shape mat)))
           (kmax (min mr n)))
      (multiple-value-bind (lu piv sign) (vt-lu mat)
        (declare (ignore sign))
        (let ((l (eye-like mr)) (u (vt-zeros (list mr n) :dtype :complex128)))
          (dotimes (i mr)
            (dotimes (j n)
              (cond ((> i j) (setf (vt-ref l i j) (vt-ref lu i j)))
                    ((and (= i j) (< i kmax)) (setf (vt-ref u i j) (vt-ref lu i j)))
                    ((<= i j) (setf (vt-ref u i j) (vt-ref lu i j))))))
          (let ((pa (vt-zeros (list mr n) :dtype :complex128)))
            (dotimes (i mr)
              (dotimes (j n)
                (setf (vt-ref pa i j) (vt-ref mat (aref piv i) j))))
            (expect-true (format nil "lu ~a: P·A = L·U" (car m))
                         (< (mabs-diff (vt-matmul l u) pa) 1d-12)))))))

  ;; ---- Cholesky：L·L^H = A ----
  (let ((l (vt-cholesky cx-la-h)))
    (expect-true "chol: L·L^H = A"
                 (< (mabs-diff (vt-matmul l (vt-transpose (vt-conj l))) cx-la-h) 1d-12))
    (expect-true "chol: L 是下三角"
                 (and (zerop (abs (vt-ref l 0 1)))
                      (plusp (abs (vt-ref l 1 0))))))

  ;; ---- QR：Q·R = A，Q^H·Q = I ----
  (dolist (m (list (cons "sq" cx-la-a) (cons "tall" cx-la-t) (cons "wide" cx-la-w)))
    (let* ((mat (cdr m))
           (q (nth-value 0 (vt-qr mat)))
           (r (nth-value 1 (vt-qr mat))))
      (expect-true (format nil "qr ~a: Q·R = A" (car m))
                   (< (mabs-diff (vt-matmul q r) mat) 1d-12))
      (expect-true (format nil "qr ~a: Q^H·Q = I" (car m))
                   (< (mabs-diff (vt-matmul (vt-transpose (vt-conj q)) q)
                                 (eye-like (second (vt-shape q))))
                      1d-12))))
  ;; :full 模式形状
  (let ((q (nth-value 0 (vt-qr cx-la-t :mode :full)))
        (r (nth-value 1 (vt-qr cx-la-t :mode :full))))
    (expect-equal "qr :full Q 形状" (vt-shape q) '(3 3))
    (expect-equal "qr :full R 形状" (vt-shape r) '(3 2))
    (expect-true "qr :full 重构"
                 (< (mabs-diff (vt-matmul q r) cx-la-t) 1d-12)))

  ;; ---- SVD：U·S·V^H = A，U^H·U = V^H·V = I ----
  (dolist (m (list (cons "sq" cx-la-a) (cons "tall" cx-la-t) (cons "wide" cx-la-w)))
    (let* ((mat (cdr m))
           (u (nth-value 0 (vt-svd mat)))
           (s (nth-value 1 (vt-svd mat)))
           (vh (nth-value 2 (vt-svd mat))))
      (expect-true (format nil "svd ~a: U·S·V^H = A" (car m))
                   (< (mabs-diff (vt-matmul (vt-matmul u (diag-complex s)) vh) mat)
                      1d-12))
      (expect-true (format nil "svd ~a: U^H·U = I" (car m))
                   (< (mabs-diff (vt-matmul (vt-transpose (vt-conj u)) u)
                                 (eye-like (second (vt-shape u))))
                      1d-12))
      (expect-true (format nil "svd ~a: V^H·V = I" (car m))
                   (< (mabs-diff (vt-matmul vh (vt-transpose (vt-conj vh)))
                                 (eye-like (first (vt-shape vh))))
                      1d-12))
      (expect-true (format nil "svd ~a: 奇异值降序" (car m))
                   (let ((items (flatten-list (vt-to-list s))))
                     (every #'>= items (rest items))))))

  ;; ---- 厄米特特征值：A·V = V·Λ ----
  (let* ((vals (nth-value 0 (vt-eig cx-la-h)))
         (vecs (nth-value 1 (vt-eig cx-la-h))))
    (expect-true "eig: A·V = V·Λ"
                 (< (mabs-diff (vt-matmul cx-la-h vecs)
                               (vt-matmul vecs (diag-complex vals)))
                    1d-12))
    (expect-true "eig: V^H·V = I"
                 (< (mabs-diff (vt-matmul (vt-transpose (vt-conj vecs)) vecs)
                               (eye-like 2))
                    1d-12))
    (expect-true "eig: 特征值降序"
                 (let ((items (flatten-list (vt-to-list vals))))
                   (every #'>= items (rest items)))))

  ;; ---- pinv：A·A⁺·A = A ----
  (dolist (m (list (cons "sq" cx-la-a) (cons "tall" cx-la-t) (cons "wide" cx-la-w)))
    (let* ((mat (cdr m))
           (p (vt-pinv mat)))
      (expect-true (format nil "pinv ~a: A·A⁺·A = A" (car m))
                   (< (mabs-diff (vt-matmul (vt-matmul mat p) mat) mat) 1d-12))
      ;; Moore-Penrose 四条条件（注意：A⁺ 一般**不是**厄米特 ——
      ;; 只有 A 厄米特时才如此，别把 A⁻¹ 的性质套到 A⁺ 上）
      (expect-true (format nil "pinv ~a: A⁺·A·A⁺ = A⁺" (car m))
                   (< (mabs-diff (vt-matmul (vt-matmul p mat) p) p) 1d-12))
      (expect-true (format nil "pinv ~a: (A·A⁺)^H = A·A⁺" (car m))
                   (< (mabs-diff (vt-transpose (vt-conj (vt-matmul mat p)))
                                 (vt-matmul mat p))
                      1d-12))
      (expect-true (format nil "pinv ~a: (A⁺·A)^H = A⁺·A" (car m))
                   (< (mabs-diff (vt-transpose (vt-conj (vt-matmul p mat)))
                                 (vt-matmul p mat))
                      1d-12))))

  ;; ---- solve 残差 ----
  (let* ((x (vt-solve cx-la-a cx-la-b1)))
    (expect-true "solve: A·x = b"
                 (< (mabs-diff (vt-matmul cx-la-a (vt-reshape x '(2 1)))
                               (vt-reshape cx-la-b1 '(2 1)))
                    1d-12)))

  ;; ---- einsum 与专用内核一致 ----
  (expect-true "einsum matmul = vt-matmul"
               (< (mabs-diff (vt-einsum "ij,jk->ik" cx-la-m2 cx-la-m2)
                             (vt-matmul cx-la-m2 cx-la-m2))
                  1d-12))
  (expect-true "einsum dot = vt-dot"
               (< (mabs-diff (vt-einsum "i,i->" cx-la-v1 cx-la-v2)
                             (vt-dot cx-la-v1 cx-la-v2))
                  1d-12)))

;;; ------------------------------------------------------------------
;;; 契约断言
;;; ------------------------------------------------------------------

(defun contract-tests ()
  (format t "~&--- 契约断言~%")

  ;; :out 硬契约（形状/dtype 不符必须报错）
  (expect-equal "det :out dtype 错误必须报错"
                (handler-case (progn (vt-det cx-la-a :out (vt-zeros nil :dtype :float64)) :no-error)
                  (error () :ok))
                :ok)
  (expect-equal "det :out 正确契约"
                (let ((o (vt-zeros nil :dtype :complex128)))
                  (vt-det cx-la-a :out o)
                  (type-of (vt-item o)))
                '(complex double-float))

  ;; 奇异矩阵：det→0、solve/inv→报错
  (expect-equal "奇异矩阵 solve 报错"
                (handler-case (progn (vt-solve cx-la-sing cx-la-b1) :no-error)
                  (error () :ok))
                :ok)
  (expect-equal "奇异矩阵 inv 报错"
                (handler-case (progn (vt-inv cx-la-sing) :no-error)
                  (error () :ok))
                :ok)
  (expect-true "奇异矩阵 det = 0"
               (let ((d (vt-item (vt-det cx-la-sing)))) (< (abs d) 1d-12)))

  ;; 非方阵 det 报错；非厄米正定 cholesky 报错
  (expect-equal "非方阵 det 报错"
                (handler-case (progn (vt-det cx-la-t) :no-error) (error () :ok))
                :ok)
  ;; H 若是正定，Cholesky 当然成功；这里用真正不定的厄米特矩阵
  ;; （[[1,2],[2,1]] 特征值 3、-1）
  (expect-equal "非正定 cholesky 报错"
                (handler-case
                    (progn (vt-cholesky (vt-from-sequence
                                         (list (list (z #C(1 0)) (z #C(2 0)))
                                               (list (z #C(2 0)) (z #C(1 0))))
                                         :dtype :complex128))
                           :no-error)
                  (error () :ok))
                :ok)

  ;; 结果 dtype 契约
  (expect-equal "inv dtype" (vt-dtype (vt-inv cx-la-a)) :complex128)
  (expect-equal "qr q dtype" (vt-dtype (nth-value 0 (vt-qr cx-la-a))) :complex128)
  (expect-equal "svd s dtype" (vt-dtype (nth-value 1 (vt-svd cx-la-a))) :float64)
  (expect-equal "svd u dtype" (vt-dtype (nth-value 0 (vt-svd cx-la-a))) :complex128)
  (expect-equal "eig vecs dtype" (vt-dtype (nth-value 1 (vt-eig cx-la-h))) :complex128)
  (expect-equal "pinv dtype" (vt-dtype (vt-pinv cx-la-a)) :complex128)

  ;; complex64 输入：内核统一升到 complex128 计算（numpy 同样如此）
  (expect-equal "c64 输入 svd u 升精度"
                (vt-dtype (nth-value 0 (vt-svd (vt-astype cx-la-a :complex64))))
                :complex128)

  ;; 实数路径零改动（回归网）
  (let* ((r (vt-from-sequence (list (list 1.0d0 2.0d0) (list 3.0d0 4.0d0)) :dtype :float64)))
    (expect-true "实数 det 不变"
                 (approx= (vt-item (vt-det r)) -2.0d0 1d-12))
    (expect-equal "实数 svd s dtype" (vt-dtype (nth-value 1 (vt-svd r))) :float64)
    (expect-true "实数 qr 重构"
                 (< (mabs-diff (vt-matmul (nth-value 0 (vt-qr r)) (nth-value 1 (vt-qr r))) r)
                    1d-12))
    (expect-true "实数 matrix-rank 默认 tol 生效（回归 p0109）"
                 (= (vt-matrix-rank (vt-from-sequence
                                     (list (list 1.0d0 2.0d0 3.0d0)
                                           (list 2.0d0 4.0d0 6.0d0)
                                           (list 1.0d0 1.0d0 1.0d0))
                                     :dtype :float64))
                    2))))

;;; ------------------------------------------------------------------
;;; 数值压测：随机 / 退化 / 极端尺寸（不依赖 numpy，全部用恒等式判定）
;;; ------------------------------------------------------------------
;;; 差分对照只能覆盖「我事先想到的矩阵」；随机化 + 恒等式能覆盖
;;; 「没想到的」情形（近奇异、病态、非方阵、秩亏、全零）。
;;; 恒等式比逐元素比对更强：QR 的 R / SVD 的 U,V 都带相位自由度，
;;; 但 `Q·R = A` 与 `U·S·V^H = A` + 正交性已经完全刻画了分解。

(defvar *lcg* 12345)
(defun rnd ()
  "确定性 LCG（同一颗种子 → 同一批矩阵，测试可复现）。"
  (setf *lcg* (mod (+ (* *lcg* 1103515245) 12345) 2147483648))
  (/ (float *lcg* 1d0) 2147483648d0))
(defun rnd-c () (complex (- (rnd) 0.5d0) (- (rnd) 0.5d0)))
(defun rnd-mat (m n)
  (vt-from-sequence
   (loop for i below m
         collect (loop for j below n collect (rnd-c)))
   :dtype :complex128))

(defun max-abs (x)
  (let* ((f (vt-flatten (ensure-vt x))) (n (vt-size f)) (best 0.0d0))
    (dotimes (i n best)
      (let ((v (vt-item (vt-slice f (list i (1+ i))))))
        (setf best (max best (abs (if (complexp v) v (complex v 0.0d0)))))))))

(defun rd (a b) (max-abs (vt-- a b)))

(defun stress-random (m n)
  (let* ((a (rnd-mat m n))
         (k (min m n))
         (scale (max 1d0 (max-abs a)))
         (tag (format nil "~ax~a" m n)))
    ;; SVD：重构 + 双侧正交 + σ 单调
    (multiple-value-bind (u sv vh) (vt-svd a)
      (expect-true (format nil "stress ~a svd 重构" tag)
                   (< (rd (vt-matmul (vt-matmul u (diag-complex sv)) vh) a)
                      (* 1d-11 scale)))
      (expect-true (format nil "stress ~a svd U 正交" tag)
                   (< (rd (vt-matmul (vt-transpose (vt-conj u)) u) (eye-like k)) 1d-11))
      (expect-true (format nil "stress ~a svd V 正交" tag)
                   (< (rd (vt-matmul vh (vt-transpose (vt-conj vh))) (eye-like k)) 1d-11))
      (expect-true (format nil "stress ~a svd σ 降序非负" tag)
                   (let ((items (flatten-list (vt-to-list sv))))
                     (and (every (lambda (x) (>= (float x 1d0) 0d0)) items)
                          (every #'>= items (rest items)))))
      ;; 交叉验证：σ² 应等于 A^H·A 的特征值（两条独立算法互证）
      (let* ((aha (vt-matmul (vt-transpose (vt-conj a)) a))
             (ev (flatten-list (vt-to-list (nth-value 0 (vt-eig aha)))))
             (ss (mapcar (lambda (x) (let ((v (float x 1d0))) (* v v)))
                         (flatten-list (vt-to-list sv)))))
        (expect-true (format nil "stress ~a σ² = eig(A^H A)" tag)
                     (< (loop for x in ss for y in ev
                              maximize (abs (- x (float y 1d0))))
                        (* 1d-8 scale scale)))))
    ;; QR
    (let ((q (nth-value 0 (vt-qr a))) (r (nth-value 1 (vt-qr a))))
      (expect-true (format nil "stress ~a qr 重构" tag)
                   (< (rd (vt-matmul q r) a) (* 1d-11 scale)))
      (expect-true (format nil "stress ~a qr Q 正交" tag)
                   (< (rd (vt-matmul (vt-transpose (vt-conj q)) q) (eye-like k)) 1d-11)))
    ;; pinv：Moore–Penrose 四条
    (let ((p (vt-pinv a)))
      (expect-true (format nil "stress ~a pinv A·A⁺·A=A" tag)
                   (< (rd (vt-matmul (vt-matmul a p) a) a) (* 1d-10 scale)))
      (expect-true (format nil "stress ~a pinv A⁺·A·A⁺=A⁺" tag)
                   (< (rd (vt-matmul (vt-matmul p a) p) p) (* 1d-10 scale)))
      (expect-true (format nil "stress ~a pinv (AA⁺)^H 对称" tag)
                   (< (rd (vt-transpose (vt-conj (vt-matmul a p)))
                          (vt-matmul a p))
                      1d-10))
      (expect-true (format nil "stress ~a pinv (A⁺A)^H 对称" tag)
                   (< (rd (vt-transpose (vt-conj (vt-matmul p a)))
                          (vt-matmul p a))
                      1d-10)))
    ;; 方阵：逆 / 解
    (when (= m n)
      (expect-true (format nil "stress ~a A·A⁻¹=I" tag)
                   (< (rd (vt-matmul a (vt-inv a)) (eye-like n)) 1d-8))
      (let* ((b (rnd-mat n 1))
             (x (vt-solve a (vt-flatten b))))
        (expect-true (format nil "stress ~a A·x=b" tag)
                     (< (rd (vt-matmul a (vt-reshape x (list n 1))) b)
                        (* 1d-9 scale)))))
    ;; 随机方阵应满秩
    (expect-true (format nil "stress ~a 满秩" tag)
                 (= (vt-matrix-rank a) k))))

(defun stress-static ()
  ;; 零矩阵
  (let ((z0 (vt-zeros (list 3 3) :dtype :complex128)))
    (expect-true "stress 零矩阵 det=0" (zerop (abs (vt-item (vt-det z0)))))
    (expect-true "stress 零矩阵 rank=0" (= (vt-matrix-rank z0) 0))
    (expect-true "stress 零矩阵 σ 全 0"
                 (every (lambda (x) (zerop (float x 1d0)))
                        (flatten-list (vt-to-list (nth-value 1 (vt-svd z0))))))
    (expect-true "stress 零矩阵 pinv=0" (< (max-abs (vt-pinv z0)) 1d-12))
    (expect-true "stress 零矩阵 inv 报错"
                 (handler-case (progn (vt-inv z0) nil) (error () t)))
    (expect-true "stress 零矩阵 cond=inf" (> (vt-cond z0) 1d300)))
  ;; 秩亏（两行线性相关）
  (let ((rk (vt-from-sequence (list (list (z #C(1 0)) (z #C(2 0)))
                                   (list (z #C(2 0)) (z #C(4 0))))
                             :dtype :complex128)))
    (expect-true "stress 秩亏 rank=1" (= (vt-matrix-rank rk) 1))
    (expect-true "stress 秩亏 det=0" (zerop (abs (vt-item (vt-det rk)))))
    (expect-true "stress 秩亏 pinv A·A⁺·A=A"
                 (< (rd (vt-matmul (vt-matmul rk (vt-pinv rk)) rk) rk) 1d-10))
    (expect-true "stress 秩亏 σ₂≈0"
                 (< (abs (float (second (flatten-list
                                         (vt-to-list (nth-value 1 (vt-svd rk)))))
                                1d0))
                    1d-12)))
  ;; 1×1
  (let ((one (vt-from-sequence (list (list (z #C(3 4)))) :dtype :complex128)))
    (expect-true "stress 1x1 det" (< (abs (- (vt-item (vt-det one)) (z #C(3 4)))) 1d-12))
    (expect-true "stress 1x1 inv"
                 (< (abs (- (vt-item (vt-inv one)) (/ (z #C(1 0)) (z #C(3 4))))) 1d-12))
    (expect-true "stress 1x1 svd σ=|3+4i|"
                 (< (abs (- (float (first (flatten-list
                                           (vt-to-list (nth-value 1 (vt-svd one)))))
                                  1d0)
                            5d0))
                    1d-12))))

(defun stress-spd ()
  (dolist (n '(1 2 3 6))
    (let* ((b (rnd-mat n n))
           (h (vt-+ (vt-matmul (vt-transpose (vt-conj b)) b)
                    (vt-scale (eye-like n) (float n 1d0))))
           (tag (format nil "spd~a" n)))
      (let ((l (vt-cholesky h)))
        (expect-true (format nil "stress ~a chol L·L^H=A" tag)
                     (< (rd (vt-matmul l (vt-transpose (vt-conj l))) h) 1d-10))
        (expect-true (format nil "stress ~a chol 下三角" tag)
                     (let ((ok t))
                       (dotimes (i n)
                         (dotimes (j n)
                           (when (and (> j i) (> (abs (vt-ref l i j)) 1d-12))
                             (setf ok nil))))
                       ok)))
      (multiple-value-bind (vals vecs) (vt-eig h)
        (expect-true (format nil "stress ~a eig A·V=V·Λ" tag)
                     (< (rd (vt-matmul h vecs) (vt-matmul vecs (diag-complex vals)))
                        1d-10))
        (expect-true (format nil "stress ~a eig V 正交" tag)
                     (< (rd (vt-matmul (vt-transpose (vt-conj vecs)) vecs)
                            (eye-like n))
                        1d-10))
        (expect-true (format nil "stress ~a eig 特征值全正" tag)
                     (every (lambda (x) (> (float x 1d0) 0d0))
                            (flatten-list (vt-to-list vals))))
        (expect-true (format nil "stress ~a eig 降序" tag)
                     (let ((v (flatten-list (vt-to-list vals))))
                       (every #'>= v (rest v))))))))

(defun stress-lstsq ()
  (dolist (sz '((6 3) (3 6)))
    (let* ((m (first sz)) (n (second sz))
           (a (rnd-mat m n))
           (b (rnd-mat m 1))
           (tag (format nil "lstsq~ax~a" m n)))
      (multiple-value-bind (x res rank) (vt-lstsq a (vt-flatten b))
        (declare (ignore res))
        (expect-true (format nil "stress ~a 解长度" tag) (= (vt-size x) n))
        (expect-true (format nil "stress ~a rank" tag) (= rank (min m n)))
        (when (>= m n)
          ;; 超定：法方程 A^H A x = A^H b
          (let* ((aha (vt-matmul (vt-transpose (vt-conj a)) a))
                 (ahb (vt-matmul (vt-transpose (vt-conj a)) b)))
            (expect-true (format nil "stress ~a 法方程" tag)
                         (< (rd (vt-matmul aha (vt-reshape x (list n 1))) ahb)
                            1d-8))))))))

(defun stress-tests ()
  (format t "~&--- 数值压测（随机 / 退化 / SPD / lstsq）~%")
  (dolist (sz '((1 1) (3 3) (5 5) (2 5) (5 2)))
    (stress-random (first sz) (second sz)))
  (stress-static)
  (stress-spd)
  (stress-lstsq))

;;; ------------------------------------------------------------------

(format t "~&=====================================~%")
(format t "复数线性代数测试 (CONVENTIONS §16.19)~%")
(format t "=====================================~%")
(run-differential)
(structural-tests)
(stress-tests)
(contract-tests)
(format t "~&=====================================~%")
(format t "Total: ~d / Pass: ~d / Fail: ~d~%" (+ *pass* *fail*) *pass* *fail*)
(format t "PASS=~d FAIL=~d~%" *pass* *fail*)
(when *failures*
  (format t "~&失败清单：~%")
  (dolist (f (reverse *failures*)) (format t "  - ~a~%" f)))
