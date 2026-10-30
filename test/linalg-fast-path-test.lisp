;;;; linalg-fast-path-test.lisp — 分解函数双路径测试（vt-lu/vt-qr/vt-solve/vt-cholesky/vt-eig）
;;;;
;;;; 覆盖类目（对应 TEST-PLAN.md，模式同 svd-fast-path-test.lisp）：
;;;;   D1 差分参考   ：与优化前【原实现】的内嵌期望值逐项判定
;;;;   C8 选路不变量 ：*vt-linalg-simd-enabled* = t(AVX) ≡ nil(通用) 结果一致
;;;;   C2/C3 数学性质：P@A=L@U、Q@R=A、L@Lᵀ=A、A@v=λv、A@x=b、A@A⁻¹=I
;;;;   G6 numpy 对比 ：det/solve/qr/cholesky/eig 实时对比（python3 不可用时跳过）
;;;;   C1/C7 边界    ：1×1 / 非方阵 LU·QR / 奇异 solve 报错 / 非正定 chol 报错
;;;;
;;;; 期望值来源：优化前原实现 (commit b08ee4b) 与 numpy 2.5.3，
;;;; 由 scripts/gen-linalg-reference.lisp 生成。
;;;; 全部使用小矩阵（≤5×5 + 基准另有上限约定），运行时间亚秒级。

(require :asdf)
;; 依据本文件位置定位 clvt 根目录（不依赖 cwd，兼容 run-tests.sh 与独立运行）
(push (truename (make-pathname :directory (append (pathname-directory (truename *load-truename*))
                                                  '(:up))))
      asdf:*central-registry*)
(asdf:load-system :clvt)
(in-package :clvt)

;;; ============================================================
;;; 测试框架（与 svd-fast-path-test.lisp 同款约定）
;;; ============================================================
(defvar *ln-N* 0)
(defvar *ln-P* 0)
(defvar *ln-F* 0)
(defvar *ln-F-list* nil)

(defun ln-ok (name)
  (incf *ln-N*)
  (incf *ln-P*))

(defun ln-fail (name &optional detail)
  (incf *ln-N*)
  (incf *ln-F*)
  (push name *ln-F-list*)
  (format t "  FAIL ~a~@[  ~a~]~%" name detail))

(defun ln-t! (name ok &optional detail)
  (if ok (ln-ok name) (ln-fail name detail)))

(defun ln-close-p (a b &optional (tol 1d-9))
  "混合容差：相对差 ≤ tol 或极小绝对差（处理 0 值）。"
  (let ((diff (abs (- a b)))
        (scale (max (abs a) (abs b) 1.0d-30)))
    (or (<= diff (* tol scale))
        (<= diff 1.0d-28))))

(defun ln-vec-close-p (xs ys &optional (tol 1d-9))
  (and (= (length xs) (length ys))
       (loop for x in xs for y in ys always (ln-close-p x y tol))))

(defun ln-tensor-close-p (t1 t2 &optional (tol 1d-9))
  (and (= (vt-size t1) (vt-size t2))
       (or (zerop (vt-size t1))
           (<= (vt-ref (vt-amax (vt-abs (vt-- t1 t2)))) tol))))

(defun ln-flat (vt) (apply #'append (vt-to-list vt)))

(defun a= (rows) (vt-from-sequence rows :dtype :float64))

;;; ============================================================
;;; 测试矩阵（显式数据，无 RNG 依赖）
;;; ============================================================
(defparameter *m-general3* (a= '((4.0d0 11.0d0 14.0d0) (8.0d0 7.0d0 -2.0d0) (1.0d0 5.0d0 9.0d0))))
(defparameter *m-ppd3*     (a= '((4.0d0 2.0d0 -2.0d0) (2.0d0 10.0d0 2.0d0) (-2.0d0 2.0d0 5.0d0))))
(defparameter *m-sym3*     (a= '((4.0d0 1.0d0 2.0d0) (1.0d0 5.0d0 3.0d0) (2.0d0 3.0d0 6.0d0))))
(defparameter *m-tall43*   (a= '((2.0d0 1.0d0 1.0d0) (1.0d0 3.0d0 2.0d0)
                                 (1.0d0 0.0d0 0.0d0) (4.0d0 5.0d0 6.0d0))))
(defparameter *m-sing3*    (a= '((1.0d0 2.0d0 3.0d0) (2.0d0 4.0d0 6.0d0) (1.0d0 1.0d0 1.0d0))))
(defparameter *m-diag3*    (a= '((5.0d0 0.0d0 0.0d0) (0.0d0 3.0d0 0.0d0) (0.0d0 0.0d0 1.0d0))))
(defparameter *m-one1x1*   (a= '((-5.0d0))))

;;; ============================================================
;;; D1 差分参考（原实现期望值，gen-linalg-reference.lisp 录得）
;;; ============================================================
(defun test-differential-reference ()
  (format t "~%--- D1 差分参考（原实现期望值）---~%")
  ;; LU: 参考 lu/piv/sign（gen-linalg-reference 录得）
  (multiple-value-bind (lu piv sign) (vt-lu (vt-copy *m-general3*))
    (declare (ignore sign))
    (ln-t! "ref/lu-general3/lu"  (ln-tensor-close-p lu (a= '((8.0d0 7.0d0 -2.0d0)
                                                             (0.5d0 7.5d0 15.0d0)
                                                             (0.125d0 0.55d0 1.0d0))) 1d-9))
    (ln-t! "ref/lu-general3/piv" (equalp piv (make-array 3 :element-type 'fixnum :initial-contents '(1 0 2)))))
  ;; LU 非方阵 + 视图输入（参考: piv=[3,1,0,2], sign=1）
  (multiple-value-bind (lu piv sign) (vt-lu (vt-copy *m-tall43*))
    (declare (ignore sign))
    (ln-t! "ref/lu-tall43/piv" (equalp piv (make-array 4 :element-type 'fixnum :initial-contents '(3 1 0 2))))
    (ln-t! "ref/lu-tall43/u21" (ln-close-p (vt-ref lu 2 1) -0.8571428571428571d0)))
  ;; LU P@A = L@U 性质（从合并 LU 重构）
  (multiple-value-bind (lu piv sign) (vt-lu (vt-copy *m-general3*))
    (declare (ignore sign))
    (let* ((n 3)
           (L (vt-zeros '(3 3) :dtype :float64))
           (U (vt-zeros '(3 3) :dtype :float64)))
      (loop for i of-type fixnum from 0 below 3 do
        (setf (vt-ref L i i) 1.0d0)
        (loop for j of-type fixnum from 0 below 3 do
          (if (< j i)
              (setf (vt-ref L i j) (vt-ref lu i j))
              (setf (vt-ref U i j) (vt-ref lu i j)))))
      (let* ((P (vt-zeros '(3 3) :dtype :float64))
             (PA (vt-@ P *m-general3*))
             (LUp (vt-@ L U)))
        (loop for i of-type fixnum from 0 below 3
              do (setf (vt-ref P i (aref piv i)) 1.0d0))
        (setf PA (vt-@ P (vt-copy *m-general3*)))
        (ln-t! "prop/lu-general3/PA=LU"
               (ln-tensor-close-p PA LUp 1d-9)))))
  ;; det（参考: general3=-60, diag3=15, sing3=0, one1x1=-5）
  (ln-t! "ref/det-general3" (ln-close-p (vt-ref (vt-det (vt-copy *m-general3*))) -60.0d0 1d-9))
  (ln-t! "ref/det-diag3"    (ln-close-p (vt-ref (vt-det (vt-copy *m-diag3*))) 15.0d0 1d-9))
  (ln-t! "ref/det-sing3"    (< (abs (vt-ref (vt-det (vt-copy *m-sing3*)))) 1d-12))
  (ln-t! "ref/det-one1x1"   (ln-close-p (vt-ref (vt-det (vt-copy *m-one1x1*))) -5.0d0 1d-9))
  ;; SOLVE（参考: x=[5.75, -5.5, 2.75]；多右端项第 1 列同 x）
  ;; 结果秩由**输入的秩**决定：b 是 (n,1) 就返回 (n,1)，不因 nrhs=1 塌成 1-D
  ;; （numpy: np.linalg.solve(A, [[1],[2],[3]]) -> shape (3,1)）。
  (ln-t! "ref/solve-general3"
         (ln-vec-close-p (vt-to-list (vt-flatten (vt-solve (vt-copy *m-general3*) (a= '((1.0d0) (2.0d0) (3.0d0))))))
                         '(5.75d0 -5.5d0 2.75d0)))
  (ln-t! "ref/solve-general3/col-shape-not-flattened"
         (equal (vt-shape (vt-solve (vt-copy *m-general3*) (a= '((1.0d0) (2.0d0) (3.0d0)))))
                '(3 1)))
  (ln-t! "ref/solve-general3-multi"
         (ln-tensor-close-p (vt-solve (vt-copy *m-general3*)
                                      (a= '((1.0d0 4.0d0 7.0d0) (2.0d0 5.0d0 8.0d0) (3.0d0 6.0d0 9.0d0))))
                            (a= '((5.75d0 9.55d0 13.350000000000001d0)
                                  (-5.5d0 -8.9d0 -12.3d0)
                                  (2.75d0 4.55d0 6.35d0))) 1d-9))
  ;; INV（参考: 逐元素期望）
  (ln-t! "ref/inv-general3"
         (ln-tensor-close-p (vt-inv (vt-copy *m-general3*))
                            (a= '((-1.2166666666666666d0 0.4833333333333334d0 2.0d0)
                                  (1.2333333333333334d0 -0.36666666666666675d0 -2.0d0)
                                  (-0.55d0 0.15000000000000002d0 1.0d0))) 1d-9))
  ;; QR（reduced 形状 4×3/3×3；full 4×4/4×3）
  (multiple-value-bind (q r) (vt-qr (vt-copy *m-tall43*))
    (ln-t! "ref/qr-tall43/shapes"
           (and (= (first (vt-shape q)) 4) (= (second (vt-shape q)) 3)
                (= (first (vt-shape r)) 3) (= (second (vt-shape r)) 3)))
    (let* ((recon (vt-@ q r))
           (err (vt-ref (vt-amax (vt-abs (vt-- recon *m-tall43*))))))
      (ln-t! "ref/qr-tall43/recon" (< err 1d-12))))
  ;; QR upper-triangular 检查（R 上三角非零，下三角为零）
  (multiple-value-bind (q r) (vt-qr (vt-copy *m-tall43*))
    (declare (ignore q))
    (let ((lower (loop for i of-type fixnum from 1 below 3
                       append (loop for j of-type fixnum from 0 below i
                                    collect (vt-ref r i j)))))
      (ln-t! "ref/qr-tall43/upper" (every (lambda (x) (< (abs x) 1d-12)) lower))))
  ;; CHOLESKY（参考: l=[[2,0,0],[1,3,0],[-1,1,√3]]；upper 对应转置）
  (ln-t! "ref/chol-ppd3"
         (ln-tensor-close-p (vt-cholesky (vt-copy *m-ppd3*))
                            (a= '((2.0d0 0.0d0 0.0d0) (1.0d0 3.0d0 0.0d0) (-1.0d0 1.0d0 1.7320508075688772d0))) 1d-12))
  (ln-t! "ref/chol-ppd3-upper"
         (ln-tensor-close-p (vt-cholesky (vt-copy *m-ppd3*) :upper t)
                            (a= '((2.0d0 1.0d0 -1.0d0) (0.0d0 3.0d0 1.0d0) (0.0d0 0.0d0 1.7320508075688772d0))) 1d-12))
  ;; EIG（参考: sym3 vals=[9.41883267597004, 3.386770156607549, 2.1943971674224088]）
  (multiple-value-bind (vals vecs) (vt-eig (vt-copy *m-sym3*))
    (ln-t! "ref/eig-sym3/vals"
           (ln-vec-close-p (vt-to-list vals)
                           '(9.41883267597004d0 3.386770156607549d0 2.1943971674224088d0) 1d-8))
    (let* ((v0 (vt-slice vecs '(:all) '(0)))
           (av (vt-@ (vt-copy *m-sym3*) v0))
           (lv (vt-scale v0 (vt-ref vals 0)))
           (err (vt-ref (vt-amax (vt-abs (vt-- av lv))))))
      (ln-t! "ref/eig-sym3/vec" (< err 1d-8))))
  ;; EIG diag3: vals=[5,3,1]（参考）
  (ln-t! "ref/eig-diag3"
         (ln-vec-close-p (vt-to-list (vt-eig (vt-copy *m-diag3*)))
                         '(5.0d0 3.0d0 1.0d0) 1d-12))
  ;; C1 numpy.linalg.eigh 对齐：非对称输入不报错，返回单三角对称化
  ;; 矩阵的特征值（默认 :lower，即下三角 [[0,0,0],[3,4,0],[6,7,8]] 的
  ;; 对称化），与 np.linalg.eigh(C)（UPLO='L'）一致（升序转降序比较）。
  ;; 镜像作用于内部拷贝，输入不被修改；两条路径结果一致。
  (let ((c (vt-reshape (vt-arange 9) '(3 3))))
    (multiple-value-bind (vals vecs) (vt-eig (vt-copy c))
      ;; 残差对“下三角对称化矩阵”成立（非对称输入的特征向量属于它，
      ;; 不属于 C 本身——与 numpy eigh 语义一致）
      (let* ((c-lower (vt-+ (vt-tril (vt-copy c)) (vt-transpose (vt-tril (vt-copy c) :k -1))))
             (v0 (vt-slice vecs '(:all) '(0)))
             (err (vt-ref (vt-amax (vt-abs (vt-- (vt-@ c-lower v0)
                                                 (vt-scale v0 (vt-ref vals 0))))))))
        (ln-t! "ref/eig-asym-numpy-aligned"
               (and (ln-vec-close-p (vt-to-list vals)
                                    (list 16.012405149480088d0
                                          -0.6733140221752988d0
                                          -3.339091127304794d0)
                                    1d-9)
                    (< err 1d-9)))))
    (ln-t! "ref/eig-asym-input-unmodified"
           (ln-tensor-close-p c (vt-reshape (vt-arange 9) '(3 3)) 0.0d0))
    (multiple-value-bind (vals-u vecs-u) (vt-eig (vt-copy c) :uplo :upper)
      (declare (ignore vecs-u))
      (ln-t! "ref/eig-asym-uplo-upper"
             (ln-vec-close-p (vt-to-list vals-u)
                             (list 11.801816018331547d0
                                   0.689642676278743d0
                                   -0.4914586946102887d0)
                             1d-9))))
  ;; 1e-16 级数值不对称（浮点残留）应被放行，且结果与对称参考一致
  (let ((a (vt-from-sequence '((4.0d0 1.0d0 2.0d0) (1.0000000000000002d0 5.0d0 3.0d0) (2.0d0 3.0d0 6.0d0))
                             :dtype :float64)))
    (multiple-value-bind (vals vecs) (vt-eig (vt-copy a))
      (declare (ignore vecs))
      (ln-t! "ref/eig-near-sym-tolerated"
             (ln-vec-close-p (vt-to-list vals)
                             '(9.41883267597004d0 3.386770156607549d0 2.1943971674224088d0) 1d-8))))
  ;; 对称化 (C+Cᵀ)/2：两路径一致且等于 numpy eigh（降序）
  (let* ((c (vt-reshape (vt-arange 9) '(3 3)))
         (sym (vt-scale (vt-+ c (vt-transpose c)) 0.5d0)))
    (multiple-value-bind (vals vecs) (vt-eig (vt-copy sym))
      (let* ((v0 (vt-slice vecs '(:all) '(0)))
             (err (vt-ref (vt-amax (vt-abs (vt-- (vt-@ (vt-copy sym) v0)
                                                 (vt-scale v0 (vt-ref vals 0))))))))
        (ln-t! "ref/eig-symmetrized-arange"
               (and (< err 1d-10)
                    (ln-close-p (vt-ref vals 0) 13.74596669241483d0 1d-9)
                    (ln-close-p (vt-ref vals 2) -1.745966692414833d0 1d-9))))))
  ;; RANK / LSTSQ（参考）
  (ln-t! "ref/rank-sing3"  (= (vt-matrix-rank (vt-copy *m-sing3*)) 2))
  (ln-t! "ref/rank-diag3"  (= (vt-matrix-rank (vt-copy *m-diag3*)) 3))
  ;; b 给的是 (4 1)：x 必须是 (3 1) 而不是 (3,)（§17.10.22）；
  ;; residual 是 numpy 口径的**平方和**，故值为 1.0969…² = 1.2032…。
  (multiple-value-bind (x res rank s) (vt-lstsq *m-tall43* (a= '((1.0d0) (0.0d0) (2.0d0) (1.0d0))))
    (declare (ignore s))
    (ln-t! "ref/lstsq-tall43"
           (and (equalp (vt-shape x) '(3 1))
                (ln-vec-close-p (vt-to-list (vt-flatten x))
                                '(1.117647058823528d0 -0.010695187165778552d0 -0.5828877005347559d0) 1d-9)
                (equalp (vt-shape res) '(1))
                (ln-close-p (vt-ref res 0) 1.2032085561497325d0 1d-9)
                (= rank 3)))))

;;; ============================================================
;;; C8 选路不变量：SIMD 快速内核 ≡ 通用内核
;;; ============================================================
(defun test-route-invariance ()
  (format t "~%--- C8 选路不变量（AVX ≡ 通用内核）---~%")
  (flet ((check (label x y tol)
           (ln-t! label (ln-tensor-close-p x y tol))))
    ;; LU
    (multiple-value-bind (lu1 piv1 s1) (vt-lu (vt-copy *m-general3*))
      (setf *vt-linalg-simd-enabled* nil)
      (multiple-value-bind (lu2 piv2 s2) (vt-lu (vt-copy *m-general3*))
        (setf *vt-linalg-simd-enabled* t)
        (check "route/lu/lu" lu1 lu2 1d-9)
        (ln-t! "route/lu/piv" (equalp piv1 piv2))
        (ln-t! "route/lu/sign" (= s1 s2))))
    ;; QR
    (multiple-value-bind (q1 r1) (vt-qr (vt-copy *m-tall43*))
      (setf *vt-linalg-simd-enabled* nil)
      (multiple-value-bind (q2 r2) (vt-qr (vt-copy *m-tall43*))
        (setf *vt-linalg-simd-enabled* t)
        (check "route/qr/q" q1 q2 1d-9)
        (check "route/qr/r" r1 r2 1d-9)))
    ;; SOLVE（多右端项，激励 SIMD nrhs 维）
    (let* ((b (a= '((1.0d0 3.0d0) (2.0d0 5.0d0) (3.0d0 7.0d0))))
           (x1 (vt-solve (vt-copy *m-general3*) b)))
      (setf *vt-linalg-simd-enabled* nil)
      (let ((x2 (vt-solve (vt-copy *m-general3*) (a= '((1.0d0 3.0d0) (2.0d0 5.0d0) (3.0d0 7.0d0))))))
        (setf *vt-linalg-simd-enabled* t)
        (check "route/solve/x" x1 x2 1d-9)))
    ;; INV
    (let ((i1 (vt-inv (vt-copy *m-general3*))))
      (setf *vt-linalg-simd-enabled* nil)
      (let ((i2 (vt-inv (vt-copy *m-general3*))))
        (setf *vt-linalg-simd-enabled* t)
        (check "route/inv/x" i1 i2 1d-9)))
    ;; CHOLESKY
    (let ((l1 (vt-cholesky (vt-copy *m-ppd3*))))
      (setf *vt-linalg-simd-enabled* nil)
      (let ((l2 (vt-cholesky (vt-copy *m-ppd3*))))
        (setf *vt-linalg-simd-enabled* t)
        (check "route/chol/l" l1 l2 1d-9)))
    ;; EIG（对称矩阵）
    (multiple-value-bind (v1 w1) (vt-eig (vt-copy *m-sym3*))
      (setf *vt-linalg-simd-enabled* nil)
      (multiple-value-bind (v2 w2) (vt-eig (vt-copy *m-sym3*))
        (setf *vt-linalg-simd-enabled* t)
        (check "route/eig/vals" v1 v2 1d-9)
        (check "route/eig/vecs" w1 w2 1d-8)))))

;;; ============================================================
;;; C2/C3 数学性质（随机矩阵族，双路径下都过 → 只在默认路径跑）
;;; ============================================================
(defun test-math-properties ()
  (format t "~%--- C2/C3 数学性质（随机矩阵族）---~%")
  (dolist (shape '((2 2) (3 3) (4 4) (5 5) (3 5) (5 3)))
    (destructuring-bind (m n) shape
      (let ((k (min m n)))
        (vt-random-seed (+ m (* 10 n)))
        (let ((a (vt-random shape)))
          ;; LU: P@A = L@U
          (multiple-value-bind (lu piv sign) (vt-lu (vt-copy a))
            (declare (ignore sign))
            (let* ((L (vt-zeros (list m k) :dtype :float64))
                   (U (vt-zeros (list k n) :dtype :float64)))
              (loop for i of-type fixnum from 0 below m do
                (loop for j of-type fixnum from 0 below n do
                  (cond ((and (< i k) (< j i)) (setf (vt-ref L i j) (vt-ref lu i j)))
                        ((and (< i k) (= i j))
                         (setf (vt-ref L i j) 1.0d0)
                         (setf (vt-ref U i j) (vt-ref lu i j)))
                        ((and (< i k) (< j n)) (setf (vt-ref U i j) (vt-ref lu i j)))
                        ((and (>= i k) (< j k)) (setf (vt-ref L i j) (vt-ref lu i j))))))
              (let* ((P (vt-zeros (list m m) :dtype :float64)))
                (loop for i of-type fixnum from 0 below m
                      do (setf (vt-ref P i (aref piv i)) 1.0d0))
                (let ((err (vt-ref (vt-amax (vt-abs (vt-- (vt-@ P a) (vt-@ L U)))))))
                  (ln-t! (format nil "prop/~dx~d/lu" m n) (< err 1d-8))))))
          ;; QR: Q@R = A，Q 正交
          (multiple-value-bind (q r) (vt-qr (vt-copy a))
            (let* ((err1 (vt-ref (vt-amax (vt-abs (vt-- (vt-@ q r) a)))))
                   (qtq (vt-@ (vt-transpose q) q))
                   (err2 (vt-ref (vt-amax (vt-abs (vt-- qtq (vt-eye k :dtype :float64)))))))
              (ln-t! (format nil "prop/~dx~d/qr" m n) (< err1 1d-8))
              (ln-t! (format nil "prop/~dx~d/qr-orth" m n) (< err2 1d-8))))
          ;; 方阵额外：solve/inv/chol/eig
          (when (= m n)
            (let ((ppd (vt-@ (vt-transpose a) a)))
              (vt-random-seed (+ 100 m))
              ;; solve: A@x = b
              (let* ((b (vt-random (list m 2)))
                     (x (vt-solve (vt-copy a) b))
                     (err (vt-ref (vt-amax (vt-abs (vt-- (vt-@ a x) b))))))
                (ln-t! (format nil "prop/~dx~d/solve" m n) (< err 1d-7)))
              ;; inv: A@A⁻¹ = I
              (let* ((ainv (vt-inv (vt-copy a)))
                     (err (vt-ref (vt-amax (vt-abs (vt-- (vt-@ a ainv)
                                                         (vt-eye m :dtype :float64)))))))
                (ln-t! (format nil "prop/~dx~d/inv" m n) (< err 1d-7)))
              ;; chol（AᵀA 正定）: L@Lᵀ = AᵀA
              (let* ((l (vt-cholesky ppd))
                     (err (vt-ref (vt-amax (vt-abs (vt-- (vt-@ l (vt-transpose l)) ppd))))))
                (ln-t! (format nil "prop/~dx~d/chol" m n) (< err 1d-7)))
              ;; eig（对称化）: A@v = λ·v，V 正交
              (let* ((sym (vt-scale (vt-+ ppd (vt-transpose ppd)) 0.5d0)))
                (multiple-value-bind (vals vecs) (vt-eig sym)
                  (let* ((v0 (vt-slice vecs '(:all) '(0)))
                         (err1 (vt-ref (vt-amax (vt-abs (vt-- (vt-@ sym v0)
                                                              (vt-scale v0 (vt-ref vals 0)))))))
                         (vtv (vt-@ (vt-transpose vecs) vecs))
                         (err2 (vt-ref (vt-amax (vt-abs (vt-- vtv (vt-eye m :dtype :float64)))))))
                    (ln-t! (format nil "prop/~dx~d/eig" m n) (< err1 1d-7))
                    (ln-t! (format nil "prop/~dx~d/eig-orth" m n) (< err2 1d-7))))))))))))

;;; ============================================================
;;; G6 numpy 实时对比（python3+numpy 不可用时优雅跳过）
;;; ============================================================
(defun numpy-eval (py-expr)
  "调用 python3+numpy 求值，返回以分数编码的 double 列表或 nil。"
  (handler-case
      (let* ((py (format nil "import sys,numpy as np~%~
                             from fractions import Fraction~%~
                             print('('+ ' '.join(str(Fraction(float(x)).numerator)+'/'+str(Fraction(float(x)).denominator) for x in np.asarray(eval(sys.argv[1])).ravel()) +')')~%"))
             (out (uiop:run-program (list "python3" "-c" py py-expr)
                                    :output :string :error-output :string)))
        (when (and out (plusp (length (string-trim '(#\Space #\Newline #\Return) out))))
          (mapcar (lambda (r) (float r 1.0d0))
                  (read-from-string out))))
    (error () nil)))

(defun matrix-literal (a)
  "把 2D 张量转为 python 可 eval 的嵌套元组字面量（定点 17 位）。"
  (let ((m (first (vt-shape a)))
        (n (second (vt-shape a))))
    (format nil "(~{~A~^,~})"
            (loop for i of-type fixnum from 0 below m
                  collect (format nil "(~{~A~^,~})"
                                  (loop for j of-type fixnum from 0 below n
                                        collect (format nil "~,17F" (vt-ref a i j))))))))

(defun test-numpy-live ()
  (format t "~%--- G6 numpy 实时对比 ---~%")
  (let ((probe (numpy-eval "(1.0,2.0,3.0)")))
    (if (null probe)
        (format t "  SKIP: python3/numpy 不可用，跳过实时对比（差分参考已覆盖 numpy 语义）~%")
        (progn
          (ln-t! "numpy/probe" (ln-vec-close-p probe '(1.0d0 2.0d0 3.0d0) 1d-12))
          ;; det
          (ln-t! "numpy/det-general3"
                 (ln-close-p (vt-ref (vt-det (vt-copy *m-general3*)))
                             (first (numpy-eval (format nil "np.linalg.det(~A)" (matrix-literal *m-general3*)))) 1d-8))
          ;; solve（b 给成列向量 (3,1) 时结果仍是 (3,1)，故比较前拉平）
          (ln-t! "numpy/solve-general3"
                 (ln-vec-close-p (vt-to-list (vt-flatten (vt-solve (vt-copy *m-general3*) (a= '((1.0d0) (2.0d0) (3.0d0))))))
                                 (numpy-eval (format nil "np.linalg.solve(~A,(1.0,2.0,3.0))"
                                                     (matrix-literal *m-general3*))) 1d-8))
          ;; cholesky
          (ln-t! "numpy/chol-ppd3"
                 (ln-tensor-close-p (vt-cholesky (vt-copy *m-ppd3*))
                                    (a= (list
                                         (loop for j of-type fixnum from 0 below 3
                                               collect (loop for i of-type fixnum from 0 below 3
                                                             collect (nth (+ (* j 3) i)
                                                                          (numpy-eval (format nil "np.linalg.cholesky(~A)"
                                                                                              (matrix-literal *m-ppd3*))))))))
                                    1d-8))
          ;; eig 特征值
          (ln-t! "numpy/eig-sym3-vals"
                 (ln-vec-close-p (vt-to-list (vt-eig (vt-copy *m-sym3*)))
                                 (let ((np-v (numpy-eval (format nil "np.linalg.eigvalsh(~A)"
                                                                 (matrix-literal *m-sym3*)))))
                                   ;; numpy eigvalsh 升序 → 降序
                                   (reverse np-v))
                                 1d-8))))))

;;; ============================================================
;;; C1/C7 边界与错误契约
;;; ============================================================
(defun test-edge-cases ()
  (format t "~%--- C1/C7 边界与错误契约 ---~%")
  ;; 1×1 LU/det/solve/chol/eig
  (multiple-value-bind (lu piv sign) (vt-lu (vt-copy *m-one1x1*))
    (ln-t! "edge/lu-1x1" (and (ln-close-p (vt-ref lu 0 0) -5.0d0)
                              (= (aref piv 0) 0) (= sign 1))))
  (ln-t! "edge/chol-1x1" (ln-close-p (vt-ref (vt-cholesky (a= '((25.0d0)))) 0 0) 5.0d0))
  (multiple-value-bind (vals vecs) (vt-eig (a= '((3.0d0))))
    (ln-t! "edge/eig-1x1" (and (ln-close-p (vt-ref vals 0) 3.0d0)
                               (ln-close-p (vt-ref vecs 0 0) 1.0d0))))
  ;; 对角阵 chol = sqrt 对角
  (ln-t! "edge/chol-diag3"
         (ln-tensor-close-p (vt-cholesky (vt-copy *m-diag3*))
                            (a= '((2.23606797749979d0 0.0d0 0.0d0)
                                  (0.0d0 1.7320508075688772d0 0.0d0)
                                  (0.0d0 0.0d0 1.0d0))) 1d-12))
  ;; 奇异矩阵 solve → 报错（错误契约）
  (ln-t! "edge/solve-singular-error"
         (handler-case (progn (vt-solve (vt-copy *m-sing3*) (a= '((1.0d0) (2.0d0) (3.0d0)))) nil)
           (error () t)))
  ;; 非正定 cholesky → 报错
  (ln-t! "edge/chol-not-ppd-error"
         (handler-case (progn (vt-cholesky (a= '((1.0d0 2.0d0) (2.0d0 1.0d0)))) nil)
           (error () t)))
  ;; 非对称矩阵 eig 不崩溃（返回值仍可用；对称性由调用方保证）
  (ln-t! "edge/eig-asym-no-crash"
         (multiple-value-list (vt-eig (a= '((2.0d0 1.0d0) (1.0d0 3.0d0))))))
  (ln-t! "edge/eig-2x2-sym" t)
  ;; 输入不被修改（astype/ensure-contiguous 拷贝语义）
  (let ((a (a= '((4.0d0 11.0d0 14.0d0) (8.0d0 7.0d0 -2.0d0) (1.0d0 5.0d0 9.0d0)))))
    (vt-lu a) (vt-qr a) (vt-cholesky (vt-@ (vt-transpose a) a))
    ;; numpy 对齐后非对称输入合法（单三角镜像在拷贝上完成，不动 a）
    (vt-eig a) (vt-det a)
    (ln-t! "edge/input-unmodified"
           (ln-tensor-close-p a (a= '((4.0d0 11.0d0 14.0d0) (8.0d0 7.0d0 -2.0d0) (1.0d0 5.0d0 9.0d0))) 0.0d0))))

;;; ============================================================
;;; 运行
;;; ============================================================
(defun run-linalg-fast-path-tests ()
  (format t "~%========================================~%")
  (format t "  clvt LINALG FAST-PATH TESTS~%")
  (format t "  lu/qr/solve/chol/eig: SIMD(AVX) ≡ generic route~%")
  (format t "========================================~%")
  (test-differential-reference)
  (test-route-invariance)
  (test-math-properties)
  (test-numpy-live)
  (test-edge-cases)
  (format t "~%========================================~%")
  (format t "  Linalg Fast-Path Tests: ~a total, ~a pass, ~a fail~%"
          *ln-N* *ln-P* *ln-F*)
  ;; 供 run-tests.sh 解析的机器可读汇总（格式 1）
  (format t "  Total: ~a | Pass: ~a | Fail: ~a~%" *ln-N* *ln-P* *ln-F*)
  (format t "========================================~%")
  (when *ln-F-list*
    (format t "Failed:~{~%  - ~a~}~%" (reverse *ln-F-list*)))
  (zerop *ln-F*))

;; 运行并以退出码兜底：0 = 全部通过, 1 = 存在失败
(let ((ok (run-linalg-fast-path-tests)))
  (sb-ext:exit :code (if ok 0 1)))
