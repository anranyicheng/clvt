(in-package :clvt)
;;; 1. 基础操作

(defun vt-dot (a b) 
  "点积/内积，支持任意维度： 
   - 若 a,b 均为 1d → 向量内积，返回数字。 
   - 若 a 为 2d, b 为 1d → 矩阵乘向量，返回 1d 向量。 
   - 若 a 为 1d, b 为 2d → 向量乘矩阵，返回 1d 向量。 
   - 若 a,b 均为 2d → 矩阵乘法 a @ b。 
   - 若 a,b 秩均 ≥2 → 批量矩阵乘法 '...ij,...jk->...ik'。 
   其他情况请直接使用 vt-einsum。"
  (with-float-safe
    (let ((ra (length (vt-shape a))) 
          (rb (length (vt-shape b)))) 
      (cond ((and (= ra 1) (= rb 1)) (vt-einsum "i,i->" a b)) 
            ;; === 新增以下两个分支 ===
            ((and (= ra 2) (= rb 1)) (vt-einsum "ij,j->i" a b)) 
            ((and (= ra 1) (= rb 2)) (vt-einsum "i,ij->j" a b)) 
            ;; ==========================
            ((and (= ra 2) (= rb 2)) (vt-einsum "ij,jk->ik" a b)) 
            ((and (>= ra 2) (>= rb 2)) (vt-einsum "...ij,...jk->...ik" a b)) 
            (t (error "vt-dot: unsupported dimensions (a: ~d, b: ~d).
                     use vt-einsum directly." ra rb))))))

(defun vt-outer (a b &key (flatten t))
  "计算张量外积。
   flatten = t (默认)
    : 先将输入展平为一维向量，再计算外积，返回二维矩阵。
      完全兼容 numpy 的 outer 函数 (支持任意维度输入自动展平)。
   flatten = nil
    : 保留输入的每个轴，将所有轴拼接形成新张量。
      例如 2d 与 3d → 5d 张量。
      等价于 (vt-einsum \"... ,...-> ... ...\"
       无法使用的替代显式下标写法)。"
  (with-float-safe
    (if flatten
	(let ((flat-a (vt-flatten a))
              (flat-b (vt-flatten b)))
          (vt-einsum "i,j->ij" flat-a flat-b))
	(vt-einsum "...i,...j->...ij" a b))))

(defun vt-trace (matrix)
  "矩阵迹: 对角线元素之和"
  (with-float-safe
    (vt-sum (vt-diagonal matrix))))

(defun vt-norm (vt &key (axis nil))
  "l2 范数 (欧几里得范数)"
  (with-float-safe
    (let ((sq (vt-square vt)))
      (if axis
          (vt-sqrt (vt-sum sq :axis axis))
          (vt-sqrt (vt-sum sq))))))

(defun vt-l1-norm (vt &key (axis nil))
  "l1 范数"
  (with-float-safe
    (if axis
	(vt-sum (vt-abs vt) :axis axis)
	(vt-sum (vt-abs vt)))))

(defun vt-frobenius-norm (matrix)
  "frobenius 范数 (专用于矩阵)"
  (with-float-safe
    (vt-norm matrix)))

;;; 2. 方程求解与矩阵分析

(defun ensure-contiguous-2d-vt (vt)
  "确保矩阵在内存中是连续的."
  (with-float-safe
    (if (vt-contiguous-p vt)
	vt
	(vt-contiguous vt))))

;;; 部分选主元 lu 分解 (返回 p, l, u，使 p*a = l*u)
(defun vt-lu (matrix)
  "lu 分解。返回 (p, l, u)，其中 p 为置换矩阵（由行交换向量表示）。"
  (with-float-safe
    (let* ((a (ensure-contiguous-2d-vt matrix))
           (n (first (vt-shape a)))
           (lu (vt-copy a))          ; 原地存储 l+u
           (piv (loop for i from 0 below n collect i)) ; 行交换记录
           (sign 1))
      (unless (eq (vt-element-type matrix) 'double-float)
	(setf lu (vt-astype lu 'double-float)))
      (loop for k from 0 below n
            for max-row = k
            for max-val = (abs (vt-ref lu k k))
            do (loop for i from (1+ k) below n
                     for abs-a = (abs (vt-ref lu i k))
                     when (> abs-a max-val)
                       do (setf max-val abs-a max-row i))
               (when (zerop max-val)
		 (error "矩阵奇异，无法进行 lu 分解"))
               ;; 交换行
               (unless (= max-row k)
		 (rotatef (nth k piv) (nth max-row piv))
		 (setf sign (- sign))
		 (loop for j from 0 below n
                       do (rotatef (vt-ref lu k j)
				   (vt-ref lu max-row j))))
               ;; 消元
               (let ((pivot (vt-ref lu k k)))
		 (loop for i from (1+ k) below n
                       for multiplier = (/ (vt-ref lu i k) pivot)
                       do (setf (vt-ref lu i k) multiplier)
                          (loop for j from (1+ k) below n
				do (decf (vt-ref lu i j)
					 (* multiplier (vt-ref lu k j)))))))
      (values lu piv sign))))

(defun vt-det (matrix)
  "基于 lu 分解计算行列式。"
  (with-float-safe
    (multiple-value-bind (lu piv sign)
	(vt-lu matrix)
      (declare (ignore piv))
      (let ((n (first (vt-shape lu)))
            (det sign))
	(dotimes (i n)
          (setf det (* det (vt-ref lu i i))))
	det))))

(defun vt-solve (a b)
  "求解线性方程组 ax = b（支持多右端项）。"
  (with-float-safe
    (let* ((a (ensure-contiguous-2d-vt a))
           (b-vt (ensure-vt b))
           (n (first (vt-shape a)))
           (nrhs (if (> (length (vt-shape b-vt)) 1)
                     (second (vt-shape b-vt))
                     1))
           (b-copy (if (= nrhs 1)
                       (vt-reshape (vt-copy b-vt) (list n 1))
                       (vt-copy b-vt))))
      (unless (eq (vt-element-type b-copy) 'double-float)
	(setf b-copy (vt-astype b-copy 'double-float)))
      (multiple-value-bind (lu piv sign)
	  (vt-lu a)
	(declare (ignore sign))
	
	;; ==========================================
	;; 1. 应用行置换 pb
	;; cltv的piv语义：最终lu的第i行 = 原始a的第piv[i]行
	;; 所以 pb 的第 i 行 = 原始 b 的第 piv[i] 行
	;; ==========================================
	(let ((orig-b (vt-copy b-copy)))
          (loop for i from 0 below n
		do (loop for j from 0 below nrhs
			 do (setf (vt-ref b-copy i j)
                                  (vt-ref orig-b (nth i piv) j)))))
	
	;; 2. 前代
	(loop for k from 0 below n
              do (loop for i from (1+ k) below n
                       for multiplier = (vt-ref lu i k)
                       do (loop for j from 0 below nrhs
				do (decf (vt-ref b-copy i j)
					 (* multiplier (vt-ref b-copy k j))))))
	
	;; 3. 回代
	(loop for k from (1- n) downto 0
              do (loop for j from 0 below nrhs
                       do (setf (vt-ref b-copy k j)
				(/ (vt-ref b-copy k j)
                                   (vt-ref lu k k))))
		 (loop for i from 0 below k
                       for factor = (vt-ref lu i k)
                       do (loop for j from 0 below nrhs
				do (decf (vt-ref b-copy i j)
					 (* factor (vt-ref b-copy k j))))))
	
	(if (= nrhs 1)
            (vt-reshape b-copy (list n))
            b-copy)))))

(defun vt-inv (matrix)
  "矩阵求逆。"
  (with-float-safe
    (let* ((n (first (vt-shape matrix)))
           (identity (vt-eye n :type (vt-element-type matrix))))
      (vt-solve matrix identity))))


;;; 3. 矩阵分解

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
  "给定向量 x，返回 v, beta, sigma 使得
   h = i - beta * v * v^t 满足 h*x = sigma * e1。
   其中 sigma = -sign(x[0]) * ||x||，β = 2 / ||v||²。
   当 x 为零向量时返回 beta=0, sigma=0。"
  (with-float-safe
    (let* ((norm-sq (vt-ref (vt-dot x x)))
           (norm (sqrt norm-sq)))
      (if (zerop norm)
          (values (vt-copy x) 0.0d0 0.0d0)
          (let* ((sx0 (vt-ref x 0))
                 (sigma (if (>= sx0 0.0d0) (- norm) norm))
                 (v (vt-copy x)))
            (setf (vt-ref v 0) (- sx0 sigma))
            (let ((beta (/ 2.0d0 (vt-ref (vt-dot v v)))))
              (values v beta sigma)))))))

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
           (a (vt-astype matrix 'double-float))
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
