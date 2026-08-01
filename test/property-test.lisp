;;;; property-test.lisp — 属性测试: 用随机输入验证数学恒等式
;;;; 不对比 numpy，而是验证数学性质是否成立

(require :asdf)
(push (truename (make-pathname :directory '(:relative :up))) asdf:*central-registry*)
(asdf:load-system :clvt)
(in-package :clvt)

;;; 测试框架
(defvar *pt-N* 0) (defvar *pt-P* 0) (defvar *pt-F* 0) (defvar *pt-F-list* nil)

(defun pt-ok (name)
  (incf *pt-N*) (incf *pt-P*))

(defun pt-fail (name &optional detail)
  (incf *pt-N*) (incf *pt-F*) (push name *pt-F-list*)
  (format t "  FAIL ~a~@[  ~a~]~%" name detail))

(defun pt-summary ()
  (format t "~%========================================~%")
  (format t "  Property Tests: ~a total, ~a pass, ~a fail~%"
          *pt-N* *pt-P* *pt-F*)
  (format t "========================================~%")
  (when *pt-F-list*
    (format t "Failed:~{~%  - ~a~}~%" (reverse *pt-F-list*)))
  (zerop *pt-F*))

(defun pt-rand (m n &optional (scale 10.0d0))
  "生成 m*n 随机矩阵。"
  (let ((arr (make-array (list m n) :element-type 'double-float)))
    (dotimes (i (* m n))
      (setf (row-major-aref arr i) (- (* scale 2.0d0 (random 1.0d0)) scale)))
    (vt-from-array arr)))

(defun pt-sym (n &optional (scale 10.0d0))
  "生成 n*n 对称矩阵。"
  (let ((a (pt-rand n n scale)))
    (vt-scale (vt-+ a (vt-transpose a)) 0.5d0)))

(defun pt-posdef (n &optional (scale 5.0d0))
  "生成 n*n 正定矩阵。"
  (let* ((a (pt-rand n n scale))
         (ata (vt-@ (vt-transpose a) a)))
    (vt-+ ata (vt-scale (vt-eye n :dtype :float64) 0.1d0))))

(defun max-abs (a b)
  "两个张量最大绝对差。"
  (vt-item (vt-amax (vt-abs (vt-- a b)))))

(defun is-identity (m &optional (tol 1e-8))
  (let ((n (first (vt-shape m))))
    (< (max-abs m (vt-eye n :dtype :float64)) tol)))

(defun is-zero (m &optional (tol 1e-8))
  (< (vt-item (vt-amax (vt-abs m))) tol))

(defmacro check (name expr &optional detail)
  `(if ,expr (pt-ok ,name) (pt-fail ,name ,detail)))


;;; ============================================================
;;; 1. 矩阵乘法属性
;;; ============================================================
(defun test-matmul ()
  (format t "~%--- 1. Matmul ---~%")
  ;; (AB)C == A(BC)
  (dotimes (i 20)
    (let* ((a (pt-rand 3 4)) (b (pt-rand 4 5)) (c (pt-rand 5 2)))
      (check (format nil "assoc ~a" i)
             (< (max-abs (vt-@ (vt-@ a b) c) (vt-@ a (vt-@ b c))) 1e-8))))
  ;; A(B+C) == AB+AC
  (dotimes (i 20)
    (let* ((a (pt-rand 3 4)) (b (pt-rand 4 5)) (c (pt-rand 4 5)))
      (check (format nil "distrib ~a" i)
             (< (max-abs (vt-@ a (vt-+ b c)) (vt-+ (vt-@ a b) (vt-@ a c))) 1e-8))))
  ;; AI == A
  (dotimes (i 10)
    (let* ((a (pt-rand 3 4)) (i4 (vt-eye 4 :dtype :float64)))
      (check (format nil "A@I=A ~a" i) (< (max-abs (vt-@ a i4) a) 1e-10))))
  ;; (AB)^T == B^T A^T
  (dotimes (i 20)
    (let* ((a (pt-rand 3 4)) (b (pt-rand 4 5)))
      (check (format nil "(AB)^T ~a" i)
             (< (max-abs (vt-transpose (vt-@ a b))
                         (vt-@ (vt-transpose b) (vt-transpose a))) 1e-8)))))


;;; ============================================================
;;; 2. 逆矩阵
;;; ============================================================
(defun test-inverse ()
  (format t "~%--- 2. Inverse ---~%")
  ;; AA^-1 == I
  (dotimes (i 20)
    (let* ((a (pt-posdef 3)) (ai (vt-inv a)))
      (check (format nil "A@Ainv=I ~a" i) (is-identity (vt-@ a ai) 1e-6))))
  ;; det(A^-1) == 1/det(A)
  (dotimes (i 20)
    (let* ((a (pt-posdef 3)) (d (vt-item (vt-det a))) (di (vt-item (vt-det (vt-inv a)))))
      (check (format nil "det(Ainv) ~a" i) (< (abs (- di (/ 1.0d0 d))) 1e-6))))
  ;; (AB)^-1 == B^-1 A^-1
  (dotimes (i 10)
    (let* ((a (pt-posdef 3)) (b (pt-posdef 3)))
      (check (format nil "(AB)inv ~a" i)
             (< (max-abs (vt-inv (vt-@ a b)) (vt-@ (vt-inv b) (vt-inv a))) 1e-4)))))


;;; ============================================================
;;; 3. SVD
;;; ============================================================
(defun test-svd ()
  (format t "~%--- 3. SVD ---~%")
  ;; U @ diag(S) @ Vt == A
  (dotimes (i 20)
    (let ((a (pt-rand 4 3)))
      (multiple-value-bind (u s vt-mat) (vt-svd a)
        (check (format nil "SVD recon ~a" i)
               (< (max-abs a (vt-@ u (vt-@ (vt-diag s) vt-mat))) 1e-8)))))
  ;; U^T U == I
  (dotimes (i 10)
    (let ((a (pt-rand 4 3)))
      (multiple-value-bind (u s vt-mat) (vt-svd a)
        (declare (ignore s vt-mat))
        (check (format nil "U^TU=I ~a" i) (is-identity (vt-@ (vt-transpose u) u) 1e-6)))))
  ;; 奇异值非负且降序
  (dotimes (i 20)
    (let ((a (pt-rand 5 4)))
      (multiple-value-bind (u s vt-mat) (vt-svd a)
        (declare (ignore u vt-mat))
        (let ((sl (vt-to-list s)))
          (check (format nil "s>=0 ~a" i) (every (lambda (x) (>= x -1e-10)) sl))
          (check (format nil "s desc ~a" i)
                 (loop for j from 0 below (1- (length sl))
                       always (>= (nth j sl) (nth (1+ j) sl) -1e-10))))))))


;;; ============================================================
;;; 4. Cholesky
;;; ============================================================
(defun test-cholesky ()
  (format t "~%--- 4. Cholesky ---~%")
  (dotimes (i 20)
    (let* ((a (pt-posdef 4)) (l (vt-cholesky a)))
      (check (format nil "L@Lt=A ~a" i) (< (max-abs a (vt-@ l (vt-transpose l))) 1e-8))
      (check (format nil "L lower ~a" i) (is-zero (vt-triu l :k 1) 1e-10)))))


;;; ============================================================
;;; 5. 特征值
;;; ============================================================
(defun test-eig ()
  (format t "~%--- 5. Eigenvalues ---~%")
  ;; V diag(lambda) V^T == A
  (dotimes (i 20)
    (let ((a (pt-sym 3)))
      (multiple-value-bind (vals vecs) (vt-eig a)
        (check (format nil "eig recon ~a" i)
               (< (max-abs a (vt-@ vecs (vt-@ (vt-diag vals) (vt-transpose vecs)))) 1e-6)))))
  ;; V^T V == I
  (dotimes (i 10)
    (let ((a (pt-sym 3)))
      (multiple-value-bind (vals vecs) (vt-eig a)
        (declare (ignore vals))
        (check (format nil "VtV=I ~a" i) (is-identity (vt-@ (vt-transpose vecs) vecs) 1e-6))))))


;;; ============================================================
;;; 6. 归约恒等式
;;; ============================================================
(defun test-reduction ()
  (format t "~%--- 6. Reduction ---~%")
  ;; sum(sum(ax)) == total
  (dotimes (i 10)
    (let* ((a (pt-rand 3 4)) (total (vt-item (vt-sum a))))
      (check (format nil "sum(sum0)=total ~a" i)
             (< (abs (- (vt-item (vt-sum (vt-sum a :axis 0))) total)) 1e-8))
      (check (format nil "sum(sum1)=total ~a" i)
             (< (abs (- (vt-item (vt-sum (vt-sum a :axis 1))) total)) 1e-8))))
  ;; mean * n == sum
  (dotimes (i 10)
    (let* ((a (pt-rand 5 4)) (n (vt-size a)))
      (check (format nil "mean*n=sum ~a" i)
             (< (abs (- (* (vt-item (vt-mean a)) n) (vt-item (vt-sum a)))) 1e-6))))
  ;; var == mean(x^2) - mean(x)^2
  (dotimes (i 10)
    (let* ((a (pt-rand 3 4))
           (v (vt-item (vt-var a)))
           (m (vt-item (vt-mean a)))
           (m2 (vt-item (vt-mean (vt-square a)))))
      (check (format nil "var id ~a" i)
             (< (abs (- v (- m2 (* m m)))) 1e-6))))
  ;; var(ddof=1) = N/(N-1) * var(ddof=0)
  (dotimes (trial 10)
    (let* ((a (pt-rand 5 4))
           (v0 (vt-item (vt-var a :ddof 0)))
           (v1 (vt-item (vt-var a :ddof 1)))
           (n (vt-size a)))
      (check (format nil "var ddof ~a" trial)
             (< (abs (- v1 (* (/ n (1- n)) v0))) 1e-6))))

  ;; prod(1..10) == 10!
  (let* ((v (vt-from-sequence (loop for i from 1 to 10 collect (float i 1.0d0))))
         (p (vt-item (vt-prod v)))
         (f (loop for i from 1 to 10 with r = 1 do (setf r (* r i)) finally (return r))))
    (check "prod=factorial" (< (abs (- p (float f 1.0d0))) 1e-6))))


;;; ============================================================
;;; 7. einsum 恒等式
;;; ============================================================
(defun test-einsum ()
  (format t "~%--- 7. einsum ---~%")
  ;; trace(AB) == trace(BA)
  (dotimes (i 20)
    (let* ((a (pt-rand 3 4)) (b (pt-rand 4 3)))
      (check (format nil "tr(AB)=tr(BA) ~a" i)
             (< (abs (- (vt-item (vt-trace (vt-@ a b)))
                        (vt-item (vt-trace (vt-@ b a))))) 1e-6))))
  ;; x^T y == y^T x
  (dotimes (i 20)
    (let* ((x (pt-rand 5 1)) (y (pt-rand 5 1)))
      (check (format nil "xTy=yTx ~a" i)
             (< (abs (- (vt-ref (vt-@ (vt-transpose x) y) 0 0)
                        (vt-ref (vt-@ (vt-transpose y) x) 0 0))) 1e-10))))
  ;; |x|^2 == x^T x
  (dotimes (i 20)
    (let* ((x (pt-rand 5 1)))
      (check (format nil "|x|^2=xTx ~a" i)
             (< (abs (- (expt (vt-item (vt-norm x)) 2)
                        (vt-ref (vt-@ (vt-transpose x) x) 0 0))) 1e-6)))))


;;; ============================================================
;;; 8. 广播一致性
;;; ============================================================
(defun test-broadcast ()
  (format t "~%--- 8. Broadcast ---~%")
  ;; A+0 == A
  (dotimes (i 10)
    (let ((a (pt-rand 3 4)))
      (check (format nil "A+0=A ~a" i) (< (max-abs (vt-+ a 0.0d0) a) 1e-10))))
  ;; A*1 == A
  (dotimes (i 10)
    (let ((a (pt-rand 3 4)))
      (check (format nil "A*1=A ~a" i) (< (max-abs (vt-* a 1.0d0) a) 1e-10)))))


;;; ============================================================
;;; 9. 转置
;;; ============================================================
(defun test-transpose ()
  (format t "~%--- 9. Transpose ---~%")
  ;; (A^T)^T == A
  (dotimes (i 20)
    (let ((a (pt-rand 3 4)))
      (check (format nil "(At)t=A ~a" i) (< (max-abs (vt-transpose (vt-transpose a)) a) 1e-10)))))


;;; ============================================================
;;; 10. 数值稳定性
;;; ============================================================
(defun test-stability ()
  (format t "~%--- 10. Stability ---~%")
  ;; 连续乘法不发散
  (let ((a (vt-from-sequence '((1.001 0.001) (0.001 0.999)) :dtype :float64))
        (r (vt-eye 2 :dtype :float64)))
    (dotimes (i 100) (setf r (vt-@ r a)))
    (check "matmul 100x stable"
           (every (lambda (x) (and (not (vt-float-nan-p x)) (not (vt-float-inf-p x)) (< (abs x) 1e30)))
                  (vt-to-list (vt-flatten r)))))
  ;; softmax 大值
  (let ((probs (vt-softmax (vt-from-sequence '(100.0 200.0 300.0 400.0 500.0)))))
    (check "softmax sum=1" (< (abs (- (vt-item (vt-sum probs)) 1.0)) 1e-5))
    (check "softmax no nan" (every (lambda (x) (not (vt-float-nan-p x))) (vt-to-list probs))))
  ;; sigmoid 极端值
  (let ((s (vt-sigmoid (vt-from-sequence '(-1000.0 -100.0 0.0 100.0 1000.0)))))
    (check "sigmoid extreme"
           (and (< (vt-ref s 0) 1e-10) (>= (vt-ref s 4) 1.0d0)
                (every (lambda (x) (not (vt-float-nan-p x))) (vt-to-list s))))))


;;; ============================================================
;;; 运行
;;; ============================================================
(defun run-property-tests ()
  (format t "~%========================================~%")
  (format t "  clvt PROPERTY-BASED TESTS~%")
  (format t "  Random input + math identity verification~%")
  (format t "========================================~%")
  (test-matmul)
  (test-inverse)
  (test-svd)
  (test-cholesky)
  (test-eig)
  (test-reduction)
  (test-einsum)
  (test-broadcast)
  (test-transpose)
  (test-stability)
  (pt-summary))

(run-property-tests)
