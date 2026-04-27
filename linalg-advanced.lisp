(in-package :clvt)

(defun vt-qr (matrix &key (mode :reduced))
  "矩阵 QR 分解。
   matrix : m×n 矩阵。
   mode   :REDUCED 返回 Q(m×k), R(k×n)，k = min(m,n)。
          :FULL    返回 Q(m×m), R(m×n)。
   返回 (values Q R)。"
  (let* ((m (first (vt-shape matrix)))
         (n (second (vt-shape matrix)))
         (k (min m n))
         (R (vt-astype matrix 'double-float))
         (vlist (make-array k :initial-element nil))
         (betas (make-array k :element-type 'double-float)))
    ;; ---- 1. 正向分解，更新 R ----
    (loop for i from 0 below k
          for x = (vt-slice R (list i m) (list i))
	  ;; 第 i 列，i 行开始，形状 (m-i)
          do (if (<= (vt-size x) 1)
                 (setf (aref betas i) 0.0d0)
                 (multiple-value-bind (v beta sigma)
                     (compute-householder x)
                   (declare (ignore sigma))
                   (setf (aref vlist i) v
                         (aref betas i) beta)
                   ;; 对子矩阵 R(i:m , i:n) 应用反射
                   (let ((subR (vt-slice R (list i m) (list i n))))
                     (let* ((w (vt-einsum "i,ij->j" v subR))   ;; 长度 (n-i)
			    (wsize (vt-size w))
                            (vcol (vt-reshape v (list (vt-size v) 1)))
                            (wrow (vt-reshape w (list 1 wsize)))
                            (update (vt-outer vcol wrow)))
                       (setf (vt-slice R (list i m) (list i n))
                             (vt-- subR (vt-scale update beta))))))))
    ;; ---- 2. 反向累积 Q (必须从 k-1 到 0) ----
    (let* ((need-full (eq mode :full))
           (Q (if need-full
                  (vt-eye m :m m :type 'double-float)
                  (vt-eye m :m k :type 'double-float))))
      (loop for i from (1- k) downto 0  ;; 反向循环
            for beta = (aref betas i)
            for v = (aref vlist i)
            when (and v (> beta 0.0d0))
              do (let ((Qsub (vt-slice Q (list i m) '(:all))))
                   (let* ((w (vt-einsum "i,ij->j" v Qsub))
			  (wsize (vt-size w))
                          (vcol (vt-reshape v (list (vt-size v) 1)))
                          (wrow (vt-reshape w (list 1 wsize)))
                          (update (vt-outer vcol wrow)))
                     (setf (vt-slice Q (list i m) '(:all))
                           (vt-- Qsub (vt-scale update beta))))))
      (values Q (if need-full
                    R
                    (vt-slice R (list 0 k) '(:all)))))))

(defun compute-householder (x)
  "给定向量 x，返回三个值：v, beta, sigma。
   其中 sigma = -sign(x[0]) * ||x||，
   使得 (I - beta * v * v^T) * x = sigma * e1 。"
  (let* ((norm (sqrt (vt-dot x x)))
         (sx0 (aref (vt-data x) 0))
         (sigma (if (>= sx0 0.0d0) (- norm) norm)))
    (let ((v (vt-copy x)))
      (incf (aref (vt-data v) 0) sigma)
      (let ((beta (/ 2.0d0 (vt-dot v v))))
        (values v beta sigma)))))

(defun vt-svd (matrix &key (full-matrices nil) (max-sweeps 50)
                        (tol 1e-10))
  "奇异值分解 A = U S V^T。
   full-matrices : T 则返回完整尺寸 U(m×m), S(k), Vt(n×n)  (k = min(m,n))
                  : NIL 返回经济尺寸 U(m×k), S(k), Vt(k×n)
   max-sweeps    : Jacobi 最大扫描次数
   tol           : 收敛容差"
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

    (flet ((col-norm-sq (M col)
             (vt-dot (vt-slice M '(:all) (list col))
		     (vt-slice M '(:all) (list col))))
           (col-dot (M c1 c2)
             (vt-dot (vt-slice M '(:all) (list c1))
		     (vt-slice M '(:all) (list c2)))))

      (let* ((U (vt-copy mat))   ; m×n
             (V (vt-eye n :type 'double-float))
             (changed t)
             (sweep 0))
        ;; Jacobi 旋转
        (loop while (and changed (< sweep max-sweeps)) do
          (setf changed nil)
          (loop for i from 0 below (1- n) do
            (loop for j from (1+ i) below n
                  for alpha = (col-norm-sq U i)
                  for beta  = (col-norm-sq U j)
                  for gamma = (col-dot U i j)
                  when (> (abs gamma) (* tol (sqrt (* alpha beta))))
                    do (setf changed t)
                       (let* ((zeta (/ (- beta alpha) (* 2 gamma)))
                              (t-abs (/ 1.0d0 (+ (abs zeta) (sqrt (+ 1 (* zeta zeta))))))
                              (t-val (if (>= zeta 0) t-abs (- t-abs)))
                              (c (/ 1.0d0 (sqrt (+ 1 (* t-val t-val)))))
                              (s (* c t-val)))
                         (let ((old-i (vt-copy (vt-slice U '(:all) (list i))))
                               (old-j (vt-copy (vt-slice U '(:all) (list j)))))
                           (setf (vt-slice U '(:all) (list i))
				 (vt-- (vt-scale old-i c) (vt-scale old-j s)))
                           (setf (vt-slice U '(:all) (list j))
				 (vt-+ (vt-scale old-i s) (vt-scale old-j c))))
                         (let ((old-i (vt-copy (vt-slice V '(:all) (list i))))
                               (old-j (vt-copy (vt-slice V '(:all) (list j)))))
                           (setf (vt-slice V '(:all) (list i))
				 (vt-- (vt-scale old-i c) (vt-scale old-j s)))
                           (setf (vt-slice V '(:all) (list j))
				 (vt-+ (vt-scale old-i s) (vt-scale old-j c)))))))
          (incf sweep))

        ;; 提取奇异值并排序
        (let* ((S-vec (make-array k :element-type 'double-float))
               (U-k (vt-zeros (list m k) :type 'double-float))
               (pairs (sort (loop for col from 0 below k
                                  collect (cons (sqrt (col-norm-sq U col)) col))
                            #'> :key #'car)))
          ;; 重组 U-k 和奇异值
          (dotimes (new-i k)
            (destructuring-bind (val . old-col) (nth new-i pairs)
              (setf (aref S-vec new-i) val)
              (setf (vt-slice U-k '(:all) (list new-i))
		    (vt-slice U '(:all) (list old-col)))))
          ;; 归一化 U-k 的列
          (dotimes (i k)
            (let ((inv (if (zerop (aref S-vec i)) 0.0d0 (/ 1.0d0 (aref S-vec i)))))
              (setf (vt-slice U-k '(:all) (list i))
		    (vt-scale (vt-slice U-k '(:all) (list i)) inv))))

          ;; 重排序 V 的列
          (let* ((V-sorted (vt-zeros (list n n) :type 'double-float))
                 (used-cols nil))
            (dotimes (new-i k)
              (let ((old-col (cdr (nth new-i pairs))))
                (push old-col used-cols)
                (setf (vt-slice V-sorted '(:all) (list new-i))
		      (vt-slice V '(:all) (list old-col)))))
            (let ((rest-cols (loop for col from 0 below n
                                   unless (member col used-cols) collect col)))
              (loop for offset from 0 for col in rest-cols
                    do (setf (vt-slice V-sorted '(:all) (list (+ k offset)))
			     (vt-slice V '(:all) (list col)))))

            (let ((S-vt (vt-from-sequence (coerce S-vec 'list)
					  :type 'double-float))
                  (Vt (vt-transpose V-sorted)))   ; 现在 Vt 是 n×n 的 V^T
              (if (not full-matrices)
                  ;; 经济模式：取 Vt 的前 k 行 -> k×n
                  (let ((Vt-k (vt-slice Vt (list 0 k) '(:all))))
                    (values U-k S-vt Vt-k))
                  ;; 完整模式：需要补全 U 到 m×m
                  (cond
                    ((= m n)
                     (values U-k S-vt Vt))
                    ((> m n)
                     (let ((U-full (extend-orthogonal-basis U-k)))
                       (values U-full S-vt Vt)))
                    ((< m n)
                     (values U-k S-vt Vt)))))))))))


(defun extend-orthogonal-basis (U-econ)
  "将 m×k 的正交列矩阵 U_econ 补全为 m×m 正交矩阵，保持前 k 列完全不变。"
  (let* ((m (first (vt-shape U-econ)))
         (k (second (vt-shape U-econ)))
         (extra (- m k)))
    (if (zerop extra)
        U-econ
        (let ((U-full (vt-zeros (list m m) :type 'double-float)))
          ;; 复制前 k 列（不变）
          (dotimes (i k)
            (setf (vt-slice U-full '(:all) (list i))
		  (vt-slice U-econ '(:all) (list i))))
          ;; 对第 k..m-1 列执行带重正交化的 Gram‑Schmidt
          (loop for col from k below m
                for v = (vt-random (list m)) ;; 随机向量
                do (loop repeat 2            ;; 重复两次以增强数值稳定性
                         do (dotimes (j col)
                              (let* ((uj (vt-slice U-full '(:all) (list j)))
                                     (proj (vt-dot uj v)))
                                (setf v (vt-- v (vt-scale uj proj))))))
                   (let ((norm (sqrt (vt-dot v v))))
                     (if (> norm 1e-12)
                         (setf (vt-slice U-full '(:all) (list col))
                               (vt-scale v (/ 1.0 norm)))
                         (error "Failed to generate orthogonal vector"))))
          U-full))))
