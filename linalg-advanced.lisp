(in-package :clvt)

(defun vt-qr (matrix &key (mode :reduced))
  "矩阵 qr 分解。
   matrix : m×n 矩阵。
   mode   :reduced 返回 q(m×k), r(k×n)，k = min(m,n)。
          :full    返回 q(m×m), r(m×n)。
   返回 (values q r)。"
  (assert (= 2 (vt-order matrix)))
  (with-float-safe
    (let* ((m (first (vt-shape matrix)))
           (n (second (vt-shape matrix)))
           (k (min m n))
           (r (vt-astype matrix 'double-float))
           (vlist (make-array k :initial-element nil))
           (betas (make-array k :element-type 'double-float)))
      ;; ---- 1. 正向分解，更新 r ----
      (loop for i from 0 below k
            for x = (vt-slice r (list i m) (list i))
	    ;; 第 i 列，i 行开始，形状 (m-i)
            do (if (<= (vt-size x) 1)
                   (setf (aref betas i) 0.0d0)
                   (multiple-value-bind (v beta sigma)
                       (compute-householder x)
                     (declare (ignore sigma))
                     (setf (aref vlist i) v
                           (aref betas i) beta)
                     ;; 对子矩阵 r(i:m , i:n) 应用反射
                     (let ((subr (vt-slice r (list i m) (list i n))))
                       (let* ((w (vt-einsum "i,ij->j" v subr))
			      ;; 长度 (n-i)
			      (wsize (vt-size w))
			      (vcol (vt-reshape v (list (vt-size v) 1)))
			      (wrow (vt-reshape w (list 1 wsize)))
			      (update (vt-outer vcol wrow)))
			 (setf (vt-slice r (list i m) (list i n))
                               (vt-- subr (vt-scale update beta))))))))
      ;; ---- 2. 反向累积 q (必须从 k-1 到 0) ----
      (let* ((need-full (eq mode :full))
             (q (if need-full
                    (vt-eye m :m m :type 'double-float)
                    (vt-eye m :m k :type 'double-float))))
	(loop for i from (1- k) downto 0  ;; 反向循环
              for beta = (aref betas i)
              for v = (aref vlist i)
              when (and v (> beta 0.0d0))
		do (let ((qsub (vt-slice q (list i m) '(:all))))
                     (let* ((w (vt-einsum "i,ij->j" v qsub))
			    (wsize (vt-size w))
                            (vcol (vt-reshape v (list (vt-size v) 1)))
                            (wrow (vt-reshape w (list 1 wsize)))
                            (update (vt-outer vcol wrow)))
                       (setf (vt-slice q (list i m) '(:all))
                             (vt-- qsub (vt-scale update beta))))))
	(values q (if need-full
                      r
                      (vt-slice r (list 0 k) '(:all))))))))

(defun compute-householder (x)
  "给定向量 x，返回三个值：v, beta, sigma。
   其中 sigma = -sign(x[0]) * ||x||，
   使得 (i - beta * v * v^t) * x = sigma * e1 。"
  (with-float-safe
    (let* ((norm (sqrt (vt-ref (vt-dot x x))))
           (sx0 (aref (vt-data x) 0))
           (sigma (if (>= sx0 0.0d0) (- norm) norm)))
      (let ((v (vt-copy x)))
	(setf (aref (vt-data v) 0)
	      (- (aref (vt-data v) 0) sigma))
	(let ((beta (/ 2.0d0 (vt-ref (vt-dot v v)))))
          (values v beta sigma))))))

(defun vt-svd (matrix &key (full-matrices nil) (max-sweeps 50)
                        (tol 1e-10))
  "奇异值分解 a = u s v^t。
   full-matrices : t 则返回完整尺寸 u(m×m), s(k), vt(n×n)  (k = min(m,n))
                  : nil 返回经济尺寸 u(m×k), s(k), vt(k×n)
   max-sweeps    : jacobi 最大扫描次数
   tol           : 收敛容差"
  (assert (= 2 (vt-order matrix)))
  (with-float-safe
    (let* ((mat (vt-astype matrix 'double-float))
           (m (first (vt-shape mat)))
           (n (second (vt-shape mat)))
           (k (min m n)))
      ;; 标量情形
      (when (and (= m 1) (= n 1))
	(let ((val (aref (vt-data mat) 0)))
          (return-from vt-svd
            (values (vt-const '(1 1) 1.0d0 :type 'double-float)
                    (vt-const '(1) (abs val) :type 'double-float)
                    (if (>= val 0)
			(vt-ones '(1 1) :type 'double-float)
			(vt-const '(1 1) -1.0d0 :type 'double-float))))))

      (flet ((col-norm-sq (m col)
               (vt-ref (vt-dot (vt-slice m '(:all) (list col))
			       (vt-slice m '(:all) (list col)))))
             (col-dot (m c1 c2)
               (vt-ref (vt-dot (vt-slice m '(:all) (list c1))
			       (vt-slice m '(:all) (list c2))))))

	(let* ((u (vt-copy mat))   ; m×n
               (v (vt-eye n :type 'double-float))
               (changed t)
               (sweep 0))
          ;; jacobi 旋转
          (loop while (and changed (< sweep max-sweeps)) do
            (setf changed nil)
            (loop for i from 0 below (1- n) do
              (loop for j from (1+ i) below n
                    for alpha = (col-norm-sq u i)
                    for beta  = (col-norm-sq u j)
                    for gamma = (col-dot u i j)
                    when (> (abs gamma) (* tol (sqrt (* alpha beta))))
                      do (setf changed t)
			 (let* ((zeta (/ (- beta alpha) (* 2 gamma)))
				(t-abs (/ 1.0d0 (+ (abs zeta) (sqrt (+ 1 (* zeta zeta))))))
				(t-val (if (>= zeta 0) t-abs (- t-abs)))
				(c (/ 1.0d0 (sqrt (+ 1 (* t-val t-val)))))
				(s (* c t-val)))
                           (let ((old-i (vt-copy (vt-slice u '(:all) (list i))))
				 (old-j (vt-copy (vt-slice u '(:all) (list j)))))
                             (setf (vt-slice u '(:all) (list i))
				   (vt-- (vt-scale old-i c) (vt-scale old-j s)))
                             (setf (vt-slice u '(:all) (list j))
				   (vt-+ (vt-scale old-i s) (vt-scale old-j c))))
                           (let ((old-i (vt-copy (vt-slice v '(:all) (list i))))
				 (old-j (vt-copy (vt-slice v '(:all) (list j)))))
                             (setf (vt-slice v '(:all) (list i))
				   (vt-- (vt-scale old-i c) (vt-scale old-j s)))
                             (setf (vt-slice v '(:all) (list j))
				   (vt-+ (vt-scale old-i s) (vt-scale old-j c)))))))
            (incf sweep))

          ;; 提取奇异值并排序
          (let* ((s-vec (make-array k :element-type 'double-float))
		 (u-k (vt-zeros (list m k) :type 'double-float))
		 (pairs (sort (loop for col from 0 below k
                                    collect (cons (sqrt (col-norm-sq u col)) col))
                              #'> :key #'car)))
            ;; 重组 u-k 和奇异值
            (dotimes (new-i k)
              (destructuring-bind (val . old-col) (nth new-i pairs)
		(setf (aref s-vec new-i) val)
		(setf (vt-slice u-k '(:all) (list new-i))
		      (vt-slice u '(:all) (list old-col)))))
            ;; 归一化 u-k 的列
            (dotimes (i k)
              (let ((inv (if (zerop (aref s-vec i)) 0.0d0 (/ 1.0d0 (aref s-vec i)))))
		(setf (vt-slice u-k '(:all) (list i))
		      (vt-scale (vt-slice u-k '(:all) (list i)) inv))))

            ;; 重排序 v 的列
            (let* ((v-sorted (vt-zeros (list n n) :type 'double-float))
                   (used-cols nil))
              (dotimes (new-i k)
		(let ((old-col (cdr (nth new-i pairs))))
                  (push old-col used-cols)
                  (setf (vt-slice v-sorted '(:all) (list new-i))
			(vt-slice v '(:all) (list old-col)))))
              (let ((rest-cols (loop for col from 0 below n
                                     unless (member col used-cols) collect col)))
		(loop for offset from 0 for col in rest-cols
                      do (setf (vt-slice v-sorted '(:all) (list (+ k offset)))
			       (vt-slice v '(:all) (list col)))))

              (let ((s-vt (vt-from-sequence (coerce s-vec 'list)
					    :type 'double-float))
                    (vt (vt-transpose v-sorted)))   ; 现在 vt 是 n×n 的 v^t
		(if (not full-matrices)
                    ;; 经济模式：取 vt 的前 k 行 -> k×n
                    (let ((vt-k (vt-slice vt (list 0 k) '(:all))))
                      (values u-k s-vt vt-k))
                    ;; 完整模式：需要补全 u 到 m×m
                    (cond
                      ((= m n)
                       (values u-k s-vt vt))
                      ((> m n)
                       (let ((u-full (extend-orthogonal-basis u-k)))
			 (values u-full s-vt vt)))
                      ((< m n)
                       (values u-k s-vt vt))))))))))))


(defun extend-orthogonal-basis (u-econ &key (rng *vt-default-random-state*))
  "将 m×k 的 u_econ 通过随机向量 + gram-schmidt 补全为 m×m 正交矩阵。"
  (declare (random-state rng))
  (with-float-safe
    (let* ((m (first (vt-shape u-econ)))
           (k (second (vt-shape u-econ)))
           (extra (- m k)))
      (if (zerop extra)
          u-econ
          (let ((u-full (vt-zeros (list m m) :type 'double-float)))
            (dotimes (i k)
              (setf (vt-slice u-full '(:all) (list i))
                    (vt-slice u-econ '(:all) (list i))))
            (loop for col from k below m
                  for v = (vt-random (list m) :rng rng)
                  do (loop repeat 2
                           do (dotimes (j col)
                                (let* ((uj (vt-slice u-full '(:all) (list j)))
                                       (proj (vt-ref (vt-dot uj v))))
                                  (setf v (vt-- v (vt-scale uj proj))))))
                     (let ((norm (sqrt (vt-ref (vt-dot v v)))))
                       (if (> norm 1e-12)
                           (setf (vt-slice u-full '(:all) (list col))
                                 (vt-scale v (/ 1.0 norm)))
                           (error "failed to generate orthogonal vector"))))
            u-full)))))
