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
    (let ((ra (length (vt-shape (ensure-vt a)))) 
          (rb (length (vt-shape (ensure-vt b)))))
      (cond ((and (= ra 0) (= rb 0)) (vt-* a b))
	    ((and (= ra 1) (= rb 1)) (vt-einsum "i,i->" a b)) 
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
  flatten = t (默认):
    先将输入展平为一维向量，再计算外积，返回二维矩阵。
    完全兼容 numpy 的 outer 函数 (支持任意维度输入自动展平)。
    例如：标量 3 和 4 -> shape (1, 1)；2d 和 3d -> shape (size_a, size_b)。
  flatten = nil:
    保留输入的每个轴，将所有轴拼接形成新张量。
    例如：2d 与 3d 计算后得到 5d 张量；若包含标量(0维)，则直接广播相乘。
    注：此模式通过在维度末尾/开头补 1 并利用广播乘法实现，无法用单一的标准 einsum 字符串直接表达。"
  (with-float-safe
    (let* ((a-vt (ensure-vt a))
           (b-vt (ensure-vt b))
           (shape-a (vt-shape a-vt))
           (shape-b (vt-shape b-vt)))
      
      (if flatten
          ;; flatten = t: 完全对标 np.outer，输出必定为 2D 矩阵
          (let ((1d-a (if (null shape-a) 
                          (vt-reshape a-vt '(1)) 
                          (vt-flatten a-vt)))
                (1d-b (if (null shape-b) 
                          (vt-reshape b-vt '(1)) 
                          (vt-flatten b-vt))))
            (vt-einsum "i,j->ij" 1d-a 1d-b))          
          ;; flatten = nil: 保留原轴拼接 (纯数学张量外积)
          (if (or (null shape-a) (null shape-b))
              ;; 只要包含标量，直接广播相乘
              (vt-* a-vt b-vt)
              ;; 均为非标量时，通过补 1 广播拼接所有轴
              (let* ((rank-a (length shape-a))
                     (rank-b (length shape-b))
                     (a-reshaped
		       (vt-reshape
			a-vt
			(append shape-a
				(make-list rank-b :initial-element 1))))
                     (b-reshaped
		       (vt-reshape
			b-vt
			(append (make-list rank-a :initial-element 1)
				shape-b))))
                (vt-* a-reshaped b-reshaped)))))))


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
  "lu 分解。返回 (p, l, u)，其中 p 为置换矩阵(由行交换向量表示)"
  (with-float-safe
    (let* ((a (ensure-contiguous-2d-vt (vt-astype matrix 'double-float)))
           (n (first (vt-shape a)))
           (data (vt-data a))
           (s0 (first (vt-strides a))) ; 行步长
           (s1 (second (vt-strides a))) ; 列步长
           (off (vt-offset a))
           (piv (loop for i from 0 below n collect i))
           (sign 1))
      (declare (type (simple-array double-float (*)) data)
               (type fixnum n s0 s1 off))
      (loop for k from 0 below n
            for max-row = k
            for max-val = (abs (aref data (+ off (* k s0) (* k s1))))
            do (loop for i from (1+ k) below n
                     for val = (abs (aref data (+ off (* i s0) (* k s1))))
                     when (> val max-val)
                       do (setf max-val val max-row i))
	       (unless (zerop max-val)
		 ;; 交换行
		 (unless (= max-row k)
                   (rotatef (nth k piv) (nth max-row piv))
                   (setf sign (- sign))
                   (loop for j from 0 below n
			 for ptr1 = (+ off (* k s0) (* j s1))
			 for ptr2 = (+ off (* max-row s0) (* j s1))
			 do (rotatef (aref data ptr1) (aref data ptr2))))
		 ;; 消元
		 (let ((pivot (aref data (+ off (* k s0) (* k s1)))))
                   (loop for i from (1+ k) below n
			 for ptr-ik = (+ off (* i s0) (* k s1))
			 for multiplier = (/ (aref data ptr-ik) pivot)
			 do (setf (aref data ptr-ik) multiplier)
                            (loop for j from (1+ k) below n
                                  for ptr-ij = (+ off (* i s0) (* j s1))
                                  for ptr-kj = (+ off (* k s0) (* j s1))
                                  do (decf (aref data ptr-ij)
					   (* multiplier (aref data ptr-kj))))))))
      (values a piv sign))))

(defun vt-det (matrix)
  "基于 lu 分解计算行列式"
  (with-float-safe
    (multiple-value-bind (lu piv sign)
        (vt-lu matrix)
      (declare (ignore piv))
      (let* ((n (first (vt-shape lu)))
             (data (vt-data lu))
             (s0 (first (vt-strides lu)))
             (s1 (second (vt-strides lu)))
             (off (vt-offset lu))
             (det sign))
        (loop for i from 0 below n
	      for pivot = (aref data (+ off
					(* i s0)
					(* i s1)))
	      when (zerop pivot)
		do (return-from vt-det 0.0d0)
		   do
		      (setf det (* det pivot)))
        det))))

(defun vt-solve (a b)
  "求解线性方程组 ax = b (支持多右端项)"
  (with-float-safe
    (let* ((a (ensure-contiguous-2d-vt a))
           (b-vt (ensure-vt b))
           (n (first (vt-shape a)))
           (b-shape (vt-shape b-vt))
           (nrhs (if (> (length b-shape) 1) (second b-shape) 1))
           (b-copy (if (= nrhs 1)
                       (vt-reshape (vt-astype (vt-copy b-vt) 'double-float)
				   (list n 1))
                       (vt-astype (vt-copy b-vt) 'double-float)))
           (orig-b (vt-copy b-copy)))
      (multiple-value-bind (lu piv sign)
          (vt-lu a)
        (declare (ignore sign))
        (let ((lu-data (vt-data lu))
              (lu-s0 (first (vt-strides lu)))
              (lu-s1 (second (vt-strides lu)))
              (lu-off (vt-offset lu))
              (b-data (vt-data b-copy))
              (b-s0 (first (vt-strides b-copy)))
              (b-s1 (second (vt-strides b-copy)))
              (b-off (vt-offset b-copy))
              (ob-data (vt-data orig-b))
              (ob-s0 (first (vt-strides orig-b)))
              (ob-s1 (second (vt-strides orig-b)))
              (ob-off (vt-offset orig-b)))          
          ;; 1. 应用行置换 pb
          (loop for i from 0 below n do
            (loop for j from 0 below nrhs do
              (setf (aref b-data (+ b-off
				    (* i b-s0)
				    (* j b-s1)))
                    (aref ob-data (+ ob-off
				     (* (nth i piv) ob-s0)
				     (* j ob-s1))))))          
          ;; 2. 前代
          (loop for k from 0 below n do
            (loop for i from (1+ k) below n
                  for mult = (aref lu-data (+ lu-off
					      (* i lu-s0)
					      (* k lu-s1)))
                  do (loop for j from 0 below nrhs do
                    (decf (aref b-data (+ b-off
					  (* i b-s0)
					  (* j b-s1)))
                          (* mult (aref b-data (+ b-off
						  (* k b-s0)
						  (* j b-s1))))))))          
          ;; 3. 回代
          (loop for k from (1- n) downto 0 do
            (let ((pivot (aref lu-data (+ lu-off
					  (* k lu-s0)
					  (* k lu-s1)))))
              (loop for j from 0 below nrhs do
                (setf (aref b-data (+ b-off
				      (* k b-s0)
				      (* j b-s1)))
                      (/ (aref b-data (+ b-off
					 (* k b-s0)
					 (* j b-s1)))
			 pivot)))
              (loop for i from 0 below k
                    for factor = (aref lu-data (+ lu-off
						  (* i lu-s0)
						  (* k lu-s1)))
                    do (loop for j from 0 below nrhs do
                      (decf (aref b-data (+ b-off
					    (* i b-s0)
					    (* j b-s1)))
                            (* factor (aref b-data (+ b-off
						      (* k b-s0)
						      (* j b-s1)))))))))
          (if (= nrhs 1)
              (vt-reshape b-copy (list n))
              b-copy))))))

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
    (let* ((x-data (vt-data x))
           (x-stride (first (vt-strides x)))
           (x-off (vt-offset x))
           (size (vt-size x))
           (norm-sq 0.0d0))
      (loop for i from 0 below size
            for ptr = (+ x-off (* i x-stride))
            for val = (aref x-data ptr)
            do (incf norm-sq (* val val)))
      (let ((norm (sqrt norm-sq)))
        (if (zerop norm)
            (values (vt-copy x) 0.0d0 0.0d0)
            (let* ((sx0 (aref x-data x-off))
                   (sigma (if (>= sx0 0.0d0) (- norm) norm))
                   (v (vt-copy x))
                   (v-data (vt-data v)))
              (setf (aref v-data 0) (- sx0 sigma))
              (let* ((beta-num 0.0d0)
                     (v-stride (first (vt-strides v)))
                     (v-off (vt-offset v)))
                (loop for i from 0 below size
                      for ptr = (+ v-off (* i v-stride))
                      for val = (aref v-data ptr)
                      do (incf beta-num (* val val)))
                (let ((beta (/ 2.0d0 beta-num)))
                  (values v beta sigma)))))))))

(defun vt-svd (matrix &key (full-matrices nil) (max-sweeps 50) (tol 1e-10))
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
        (let* ((u (vt-copy mat))
               (v (vt-eye n :type 'double-float))
               (changed t)
               (sweep 0))
          
          (let ((u-data (vt-data u))
                (u-s0 (first (vt-strides u)))
                (u-s1 (second (vt-strides u)))
                (u-off (vt-offset u))
                (v-data (vt-data v))
                (v-s0 (first (vt-strides v)))
                (v-s1 (second (vt-strides v)))
                (v-off (vt-offset v)))
            
            (loop while (and changed (< sweep max-sweeps)) do
              (setf changed nil)
              (loop for i from 0 below (1- n) do
                (loop for j from (1+ i) below n
                      for alpha = (col-norm-sq u i)
                      for beta = (col-norm-sq u j)
                      for gamma = (col-dot u i j)
                      when (> (abs gamma) (* tol (sqrt (* alpha beta)))) do
                        (setf changed t)
                        (let* ((zeta (/ (- beta alpha) (* 2 gamma)))
                               (t-abs (/ 1.0d0 (+ (abs zeta)
						  (sqrt (+ 1 (* zeta zeta))))))
                               (t-val (if (>= zeta 0) t-abs (- t-abs)))
                               (c (/ 1.0d0 (sqrt (+ 1 (* t-val t-val)))))
                               (s (* c t-val)))
                          (loop for r from 0 below m do
                            (let* ((ptr-i (+ u-off (* r u-s0) (* i u-s1)))
                                   (ptr-j (+ u-off (* r u-s0) (* j u-s1)))
                                   (ui (aref u-data ptr-i))
                                   (uj (aref u-data ptr-j)))
                              (setf (aref u-data ptr-i) (- (* ui c) (* uj s)))
                              (setf (aref u-data ptr-j) (+ (* ui s) (* uj c)))))
                          (loop for r from 0 below n do
                            (let* ((ptr-i (+ v-off (* r v-s0) (* i v-s1)))
                                   (ptr-j (+ v-off (* r v-s0) (* j v-s1)))
                                   (vi (aref v-data ptr-i))
                                   (vj (aref v-data ptr-j)))
                              (setf (aref v-data ptr-i) (- (* vi c) (* vj s)))
                              (setf (aref v-data ptr-j) (+ (* vi s) (* vj c))))))))
              (incf sweep)))
          
          (let* ((s-vec (make-array k :element-type 'double-float))
                 (u-k (vt-zeros (list m k) :type 'double-float))
                 ;; 修复1：遍历所有 n 列
                 (pairs (sort (loop for col from 0 below n
				    collect 
				    (cons (sqrt (col-norm-sq u col)) col)) 
                              #'> :key #'car)))
            
            (dotimes (new-i k)
              (destructuring-bind (val . old-col) (nth new-i pairs)
                (setf (aref s-vec new-i) val)
                (setf (vt-slice u-k '(:all) (list new-i)) 
                      (vt-slice u '(:all) (list old-col)))))
            
            ;; 修复2：零奇异值必须补全正交基，不能乘 0
            (dotimes (i k)
              (if (zerop (aref s-vec i))
                  (let ((random-v
			  (vt-random (list m) :rng *vt-default-random-state*)))
                    (loop repeat 2 do
                      (dotimes (j i)
                        (let* ((uj (vt-slice u-k '(:all) (list j)))
                               (proj (vt-ref (vt-dot uj random-v))))
                          (setf random-v (vt-- random-v (vt-scale uj proj))))))
                    (let ((norm (sqrt (vt-ref (vt-dot random-v random-v)))))
                      (if (> norm 1e-12)
                          (setf (vt-slice u-k '(:all) (list i))
				(vt-scale random-v (/ 1.0d0 norm)))
                          (setf (vt-slice u-k '(:all) (list i)) random-v))))
                  (let ((inv (/ 1.0d0 (aref s-vec i))))
                    (setf (vt-slice u-k '(:all) (list i)) 
                          (vt-scale (vt-slice u-k '(:all) (list i)) inv)))))
            
            (let* ((v-sorted (vt-zeros (list n n) :type 'double-float))
                   (used-cols nil))
              (dotimes (new-i k)
                (let ((old-col (cdr (nth new-i pairs))))
                  (push old-col used-cols)
                  (setf (vt-slice v-sorted '(:all) (list new-i)) 
                        (vt-slice v '(:all) (list old-col)))))
              
              (let ((rest-cols (loop for col from 0 below n 
                                     unless (member col used-cols) 
                                       collect col)))
                (loop for offset from 0 for col in rest-cols do
                  (setf (vt-slice v-sorted '(:all) (list (+ k offset))) 
                        (vt-slice v '(:all) (list col)))))
              
              (let ((s-vt (vt-from-sequence (coerce s-vec 'list)
					    :type 'double-float))
                    (vt (vt-transpose v-sorted)))
                (if (not full-matrices)
                    (let ((vt-k (vt-slice vt (list 0 k) '(:all))))
                      (values u-k s-vt vt-k))
                    (cond ((= m n) 
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
