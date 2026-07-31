;;;; linalg-extensions.lisp — 扩展线性代数功能

(in-package :clvt)

;;; ============================================================
;;; 1. Cholesky 分解
;;; ============================================================

(defun vt-cholesky (matrix &key (upper nil))
  "对正定对称矩阵进行 Cholesky 分解。"
  (assert (= 2 (vt-order matrix)))
  (let ((shape (vt-shape matrix)))
    (assert (= (first shape) (second shape))
            (matrix) "Cholesky 分解要求方阵"))
  (with-float-safe
    (let* ((n (first (vt-shape matrix)))
           (a (vt-astype matrix :float64))
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
      (loop for j fixnum from 0 below n do
        (let ((sum (aref a-data (+ a-off (* j a-s0) (* j a-s1)))))
          (declare (type double-float sum))
          (loop for k fixnum from 0 below j do
            (let ((l-jk (aref l-data (+ l-off (* j l-s0) (* k l-s1)))))
              (decf sum (* l-jk l-jk))))
          (when (<= sum 0.0d0)
            (error "vt-cholesky: 矩阵不是正定的"))
          (setf (aref l-data (+ l-off (* j l-s0) (* j l-s1))) (sqrt sum)))
        (loop for i fixnum from (1+ j) below n do
          (let ((sum (aref a-data (+ a-off (* i a-s0) (* j a-s1)))))
            (declare (type double-float sum))
            (loop for k fixnum from 0 below j do
              (decf sum (* (aref l-data (+ l-off (* i l-s0) (* k l-s1)))
                           (aref l-data (+ l-off (* j l-s0) (* k l-s1))))))
            (setf (aref l-data (+ l-off (* i l-s0) (* j l-s1)))
                  (/ sum (aref l-data (+ l-off (* j l-s0) (* j l-s1))))))))
      (if upper (vt-transpose l) l))))

;;; ============================================================
;;; 2. 对称矩阵特征值分解 (Jacobi 旋转法)
;;; ============================================================

(defun vt-eig (matrix &key (max-iter 200) (tol 1e-10))
  "对称矩阵特征值分解 (Jacobi 旋转法)。"
  (assert (= 2 (vt-order matrix)))
  (let ((shape (vt-shape matrix)))
    (assert (= (first shape) (second shape))
            (matrix) "特征值分解要求方阵"))
  (with-float-safe
    (let* ((n (first (vt-shape matrix)))
           (a (vt-copy (vt-astype matrix :float64)))
           (v (vt-eye n :dtype :float64)))
      ;; Jacobi iterations - use local variables to avoid compiler issues
      (loop for iter fixnum from 0 below max-iter do
        ;; Check convergence
        (let ((off-sum 0.0d0))
          (declare (type double-float off-sum))
          (loop for i fixnum from 0 below n do
            (loop for j fixnum from (1+ i) below n do
              (let ((val (vt-ref a i j)))
                (incf off-sum (* val val)))))
          (when (< (sqrt off-sum) tol) (return)))
        ;; Apply rotations
        (loop for p fixnum from 0 below (1- n) do
          (loop for q fixnum from (1+ p) below n do
            (let ((apq (vt-ref a p q)))
              (when (> (abs apq) tol)
                (let* ((app (vt-ref a p p))
                       (aqq (vt-ref a q q))
                       (theta (* 0.5d0 (atan (/ (* 2.0d0 apq) (- app aqq)))))
                       (c (cos theta))
                       (s (sin theta)))
                  ;; Update A
                  (setf (vt-ref a p q) 0.0d0)
                  (setf (vt-ref a q p) 0.0d0)
                  (setf (vt-ref a p p) (+ (* c c app) (* 2.0d0 s c apq) (* s s aqq)))
                  (setf (vt-ref a q q) (+ (* s s app) (* -2.0d0 s c apq) (* c c aqq)))
                  ;; Update off-diagonal
                  (loop for r fixnum from 0 below n
                        when (and (/= r p) (/= r q)) do
                          (let ((arp (vt-ref a r p))
                                (arq (vt-ref a r q)))
                            (let ((na (+ (* c arp) (* s arq)))
                                  (nb (+ (* (- s) arp) (* c arq))))
                              (setf (vt-ref a r p) na)
                              (setf (vt-ref a p r) na)
                              (setf (vt-ref a r q) nb)
                              (setf (vt-ref a q r) nb))))
                  ;; Update eigenvectors
                  (loop for r fixnum from 0 below n do
                    (let ((vp (vt-ref v r p))
                          (vq (vt-ref v r q)))
                      (setf (vt-ref v r p) (+ (* c vp) (* s vq)))
                      (setf (vt-ref v r q) (+ (* (- s) vp) (* c vq)))))))))))
      ;; Extract and sort eigenvalues
      (let ((pairs (loop for i fixnum from 0 below n
                         collect (cons (vt-ref a i i) i))))
        (setf pairs (sort pairs #'> :key #'car))
        (let ((vals (vt-from-sequence (mapcar #'car pairs) :dtype :float64))
              (vec (vt-zeros (list n n) :dtype :float64)))
          (loop for ni fixnum from 0 below n
                for oi = (cdr (nth ni pairs))
                do (loop for r fixnum from 0 below n do
                     (setf (vt-ref vec r ni) (vt-ref v r oi))))
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
        (let* ((n (first (vt-shape vt-mat)))
               (k (vt-size s))
               (v-scaled (vt-zeros (list n k) :dtype :float64)))
          (loop for j fixnum from 0 below k do
            (let ((sc (vt-ref s-pinv j)))
              (loop for i fixnum from 0 below n do
                (setf (vt-ref v-scaled i j) (* (vt-ref vt-mat j i) sc)))))
          (vt-@ v-scaled (vt-transpose u)))))))

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
               (x (vt-zeros (list n nrhs) :dtype :float64)))
          (loop for j fixnum from 0 below nrhs do
            (loop for i fixnum from 0 below rank do
              (setf (vt-ref x i j) (* (vt-ref utb i j) (/ 1.0d0 (aref s-data i))))))
          (setf x (vt-@ (vt-transpose vt-mat) x))
          (let ((res (if (and (> m n) (= rank n))
                         (vt-ref (vt-norm (vt-flatten (vt-- b-mat (vt-@ a x)))))
                         0.0d0))
                (xr (if (= nrhs 1) (vt-flatten (vt-slice x '(:all) '(0))) x)))
            (values xr res rank s)))))))
