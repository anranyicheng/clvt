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
    (loop for i from 0 below k
          for x = (vt-slice R (list i m) i)     ; 第 i 列，i 行开始，形状 (m-i)
          do (if (<= (vt-size x) 1)
                 (setf (aref betas i) 0.0d0)
                 (multiple-value-bind (v beta sigma)
                     (compute-householder x)
                   (declare (ignore sigma))
                   (setf (aref vlist i) v
                         (aref betas i) beta)
                   ;; 对子矩阵 R(i:m , i:n) 应用反射
                   (let ((subR (vt-slice R (list i m) (list i n))))
                     (let* ((w (vt-einsum "i,ij->j" v subR))   ; 长度 (n-i)
			    (wsize (vt-size w))
                            (vcol (vt-reshape v (list (vt-size v) 1)))
                            (wrow (vt-reshape w (list 1 wsize)))
                            (update (vt-outer vcol wrow)))
                       (setf (vt-slice R (list i m) (list i n))
                             (vt-- subR (vt-scale update beta))))))))
    ;; 构造 Q
    (let* ((need-full (eq mode :full))
           (Q (if need-full
                  (vt-eye m :m m :type 'double-float)
                  (vt-eye m :m k :type 'double-float))))
      (loop for i from 0 below k
            for beta = (aref betas i)
            for v = (aref vlist i)
            when (and v (> beta 0.0d0))
              do (let ((Qsub (vt-slice Q (list i m) :all)))
                   (let* ((w (vt-einsum "i,ij->j" v Qsub))
			  (wsize (vt-size w))
                          (vcol (vt-reshape v (list (vt-size v) 1)))
                          (wrow (vt-reshape w (list 1 wsize)))
                          (update (vt-outer vcol wrow)))
                     (setf (vt-slice Q (list i m) :all)
                           (vt-- Qsub (vt-scale update beta))))))
      (values Q (if need-full
                    R
                    (vt-slice R (list 0 k) :all))))))

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

(defun vt-svd (matrix &key (full-matrices nil) (max-sweeps 50) (tol 1e-10))
  "奇异值分解 A = U S V^T。
   full-matrices : T 则返回完整尺寸 U(m×m), S(k), Vt(n×n)  (k = min(m,n))
                  : NIL 返回经济尺寸 U(m×k), S(k), Vt(k×n)
   max-sweeps    : Jacobi 最大扫描次数
   tol           : 收敛容差"
  (let* ((mat (vt-astype matrix 'double-float))
         (m (first (vt-shape mat)))
         (n (second (vt-shape mat)))
         (k (min m n)))
    ;; ===== 标量情形 =====
    (when (and (= m 1) (= n 1))
      (let ((val (aref (vt-data mat) 0)))
        (return-from vt-svd
          (values (vt-const '(1 1) 1.0d0 :type 'double-float)
                  (vt-const '(1) (abs val) :type 'double-float)
                  (if (>= val 0)
                      (vt-ones '(1 1) :type 'double-float)
                      (vt-const '(1 1) -1.0d0 :type 'double-float))))))

    ;; ===== 辅助函数：列范数平方、列内积 =====
    ;; 提升到此处，使得整个函数体可见
    (flet ((col-norm-sq (M col)
             "返回矩阵 M 第 col 列的平方范数（即列与自身的点积）。"
             (vt-dot (vt-slice M :all col) (vt-slice M :all col)))
           (col-dot (M c1 c2)
             "返回矩阵 M 第 c1 列与第 c2 列的点积。"
             (vt-dot (vt-slice M :all c1) (vt-slice M :all c2))))

      ;; ===== 1. 单边 Jacobi =====
      (let* ((U (vt-copy mat))                ; m×n，迭代后前 k 列有意义
             (V (vt-eye n :type 'double-float)) ; 注意：vt-eye 只需一个参数 n
             (changed t)
             (sweep 0))
        ;; Jacobi 循环
        (loop while (and changed (< sweep max-sweeps)) do
          (setf changed nil)
          (loop for i from 0 below (1- n) do
            (loop for j from (1+ i) below n
                  for alpha = (col-norm-sq U i)
                  for beta  = (col-norm-sq U j)
                  for gamma = (col-dot U i j)
                  when (> (abs gamma) (* tol (sqrt (* alpha beta))))
                    do (setf changed t)
                       ;; 计算 Givens 旋转参数
                       (let* ((zeta (/ (- beta alpha) (* 2 gamma)))
                              (t-abs (/ 1.0d0 (+ (abs zeta)
                                                 (sqrt (+ 1 (* zeta zeta))))))
                              (t-val (if (>= zeta 0) t-abs (- t-abs)))
                              (c (/ 1.0d0 (sqrt (+ 1 (* t-val t-val)))))
                              (s (* c t-val)))
                         ;; 更新 U 的第 i, j 列
                         (let ((old-i (vt-copy (vt-slice U :all i)))
                               (old-j (vt-copy (vt-slice U :all j))))
                           (setf (vt-slice U :all i)
                                 (vt-- (vt-scale old-i c) (vt-scale old-j s))
                                 (vt-slice U :all j)
                                 (vt-+ (vt-scale old-i s) (vt-scale old-j c))))
                         ;; 更新 V 的第 i, j 列
                         (let ((old-i (vt-copy (vt-slice V :all i)))
                               (old-j (vt-copy (vt-slice V :all j))))
                           (setf (vt-slice V :all i)
                                 (vt-- (vt-scale old-i c) (vt-scale old-j s))
                                 (vt-slice V :all j)
                                 (vt-+ (vt-scale old-i s) (vt-scale old-j c)))))))
          (incf sweep))

      ;; ===== 2. 提取奇异值并排序 (此时 col-norm-sq 仍然可见) =====
      (let* ((S-vec (make-array k :element-type 'double-float))
             (U-k (vt-zeros (list m k) :type 'double-float))
             ;; 生成奇异值列表 (sqrt(col_norm_sq) . 原始列号)
             (pairs (sort (loop for col from 0 below k
                                collect (cons (sqrt (col-norm-sq U col)) col))
                          #'> :key #'car)))
        ;; 重组 U_k 并填充 S
        (dotimes (new-i k)
          (destructuring-bind (val . old-col) (nth new-i pairs)
            (setf (aref S-vec new-i) val)
            (setf (vt-slice U-k :all new-i) (vt-slice U :all old-col))))
        ;; 规范化 U_k 的列
        (dotimes (i k)
          (let ((inv (if (zerop (aref S-vec i)) 0.0d0
                         (/ 1.0d0 (aref S-vec i)))))
            (setf (vt-slice U-k :all i) (vt-scale (vt-slice U-k :all i) inv))))
        ;; 重排序 V：需要将 V 的列也按照奇异值降序排列
        (labels ((reorder-V-by-pairs (V-matrix pairs)
                   (let* ((n (second (vt-shape V-matrix)))
                          (V-sorted (vt-zeros (list n n) :type 'double-float)))
                     ;; 前 k 个主奇异向量按降序放置
                     (dotimes (new-i k)
                       (let ((old-col (cdr (nth new-i pairs))))
                         (setf (vt-slice V-sorted :all new-i)
                               (vt-slice V-matrix :all old-col))))
                     ;; 剩余 n-k 个零奇异值向量保持原顺序
                     (let ((rest (loop for col from 0 below n
                                       unless (member col (mapcar #'cdr pairs))
                                         collect col)))
                       (loop for offset from 0
                             for col in rest
                             do (setf (vt-slice V-sorted :all (+ k offset))
                                      (vt-slice V-matrix :all col))))
                     V-sorted)))
          (let ((V-sorted (reorder-V-by-pairs V pairs)))
            ;; Vt 为 n×n 矩阵
            (setf V (vt-transpose V-sorted))))

        ;; ===== 3. 根据 full-matrices 构建返回值 =====
        (if (not full-matrices)
            ;; 经济尺寸：U(m×k), S(k), Vt(k×n)
            (let ((Vt-k (vt-slice V (list 0 k) :all)))
              (values U-k S-vec Vt-k))
            ;; 全尺寸：U(m×m), S(k), Vt(n×n)
            (cond
              ((= m n)
               (values U-k S-vec V))          ; 方阵已完整
              ((> m n)
               (let ((U-full (extend-orthogonal-basis U-k)))
                 (values U-full S-vec V)))
              ((< m n)
               ;; V 已经是 n×n，U 需要 m×m：由于 k = m, U-k 已是 m×m 正交阵
               (values U-k S-vec V)))))))))

;; ---- 辅助函数：正交基扩展 ----
(defun extend-orthogonal-basis (U-econ)
  "将 m×n 的正交列矩阵 U_econ 补全为 m×m 正交矩阵。"
  (let* ((m (first (vt-shape U-econ)))
         (n (second (vt-shape U-econ)))
         (extra (- m n)))
    (if (zerop extra)
        U-econ
        (let* ((rand (vt-random (list m extra)))       ; 随机填充
               (aug (vt-concatenate 1 U-econ rand))     ; [U_econ | rand]
               (Q (vt-qr aug :mode :full)))             ; 取 Q 因子
          Q))))


;; ---- 辅助函数：正交基扩展 ----
(defun extend-orthogonal-basis (U-econ)
  "将 m×n 的正交列矩阵 U_econ 补全为 m×m 正交矩阵。"
  (let* ((m (first (vt-shape U-econ)))
         (n (second (vt-shape U-econ)))
         (extra (- m n)))
    (if (zerop extra)
        U-econ
        (let* ((rand (vt-random (list m extra)))       ; 随机填充
               (aug (vt-concatenate 1 U-econ rand))     ; [U_econ | rand]
               (Q (vt-qr aug :mode :full)))             ; 取 Q 因子
          Q))))
