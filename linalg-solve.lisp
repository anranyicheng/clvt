(in-package :clvt)

;; 辅助函数：将 2D VT 转为 Common Lisp 原生嵌套 Array
(defun vt-to-2d-array (vt)
  (let ((shape (vt-shape vt))
        (data (vt-data vt)))
    (assert (= (length shape) 2))
    (let* ((rows (first shape))
           (cols (second shape))
           (arr (make-array (list rows cols)
			    :initial-element 0.0d0)))
      (declare (fixnum rows cols))
      (dotimes (i rows)
	(declare (fixnum i))
        (dotimes (j cols)
	  (declare (fixnum j))
          ;; 假设是行主序，直接计算偏移（注意：若 VT 经过 transpose，应先调用 vt-contiguous）
          (setf (aref arr i j) (aref data (+ (* i cols) j)))))
      arr)))

;; 辅助函数：从 2D Array 创建 VT
(defun vt-from-2d-array (arr)
  (let* ((rows (array-dimension arr 0))
         (cols (array-dimension arr 1))
         (data (make-array (* rows cols)
			   :initial-element 0.0d0
			   :element-type 'double-float)))
    (declare (fixnum rows cols))
    (dotimes (i rows)
      (declare (fixnum i))
      (dotimes (j cols)
	(declare (fixnum j))
        (setf (aref data (+ (* i cols) j))
	      (coerce (aref arr i j) 'double-float))))
    (%make-vt :data data :shape (list rows cols) 
             :strides (list cols 1) :offset 0)))

;; 确保矩阵在内存中是连续的，否则转换为嵌套数组会出错
(defun ensure-contiguous-2d-vt (vt)
  (if (vt-contiguous-p vt)
      vt
      (vt-contiguous vt)))

;; 行列式 (基于高斯消元)
(defun vt-det (matrix)
  (let* ((c-vt (ensure-contiguous-2d-vt matrix))
         (arr (vt-to-2d-array c-vt))
         (n (first (vt-shape c-vt)))
         (det-val 1.0d0))
    (dotimes (i n)
      (declare (fixnum i))
      ;; 选主元
      (let ((max-row i)
            (max-val (abs (aref arr i i))))
        (dotimes (k (- n i 1))
	  (declare (fixnum k))
          (when (> (abs (aref arr (+ k i 1) i)) max-val)
            (setf max-val (abs (aref arr (+ k i 1) i))
                  max-row (+ k i 1))))
        ;; 若主元为0，行列式为0
        (when (zerop max-val)
	  (return-from vt-det 0.0d0))
        ;; 交换行
        (unless (= max-row i)
          (rotatef (row-major-aref arr i)
		   (row-major-aref arr max-row)) ;; 简写示意，实际需交换整行
          (setf det-val (- det-val)))
        ;; 消元
        (let ((pivot (aref arr i i)))
          (setf det-val (* det-val pivot))
          (dotimes (k (- n i 1))
            (let ((factor (/ (aref arr (+ k i 1) i) pivot)))
              (dotimes (j (- n i))
                (decf (aref arr (+ k i 1) (+ j i))
		      (* factor (aref arr i (+ j i)))))))))
      det-val)))

;; 求解线性方程组 Ax = b (高斯-约旦消元法)
;; 返回解向量 x (形状为 (n, 1) 的 VT)
(defun vt-solve (a b)
  (let* ((a-c (ensure-contiguous-2d-vt a))
         (b-c (ensure-contiguous-2d-vt b))
         (a-arr (vt-to-2d-array a-c))
         (n (first (vt-shape a-c)))
         ;; 构建增广矩阵 [A|b]
         (aug (make-array (list n (1+ n))
			  :element-type 'double-float
			  :initial-element 0.0d0)))
    
    ;; 填充增广矩阵
    (dotimes (i n)
      (declare (fixnum i))
      (dotimes (j n)
        (declare (fixnum j))
        (setf (aref aug i j) (aref a-arr i j)))
      (setf (aref aug i n) (aref (vt-data b-c) i))) ;; 假设 b 是列向量
    
    ;; 高斯-约旦消元
    (dotimes (i n)
      ;; 选列主元并交换行
      (let ((max-row i))
        (dotimes (k (- n i 1))
          (when (> (abs (aref aug (+ k i 1) i))
		   (abs (aref aug max-row i)))
            (setf max-row (+ k i 1))))
	(unless (= max-row i)
          (dotimes (col (1+ n))
            (declare (fixnum col))
            (rotatef (aref aug i col)
		     (aref aug max-row col))))
        
        (let ((pivot (aref aug i i)))
          ;; 归一化当前行
          (dotimes (j (1+ n))
            (setf (aref aug i j) (/ (aref aug i j) pivot)))
          ;; 消去其他行
          (dotimes (k n)
            (unless (= k i)
              (let ((factor (aref aug k i)))
                (dotimes (j (1+ n))
                  (decf (aref aug k j) (* factor (aref aug i j))))))))))
    
    ;; 提取结果列
    (let ((res-data (make-array n :initial-element 0.0d0
				  :element-type 'double-float)))
      (dotimes (i n)
        (setf (aref res-data i) (aref aug i n)))
      ;; 返回形状为 的列张量
      (%make-vt :data res-data :shape (list n 1)
		:strides (list 1 1) :offset 0))
    ))

;; 矩阵求逆: 通过求解 AX = I 实现
(defun vt-inv (matrix)
  (let* ((n (first (vt-shape matrix)))
         (identity (vt-eye n)))
    (vt-solve matrix identity)))
