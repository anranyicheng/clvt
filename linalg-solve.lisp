(in-package :clvt)

(defun ensure-contiguous-2d-vt (vt)
  "确保矩阵在内存中是连续的."
  (if (vt-contiguous-p vt)
      vt
      (vt-contiguous vt)))

;;; 部分选主元 LU 分解 (返回 P, L, U，使 P*A = L*U)
(defun vt-lu (matrix)
  "LU 分解。返回 (P, L, U)，其中 P 为置换矩阵（由行交换向量表示）。"
  (sb-int::with-float-traps-masked (:invalid)
    (let* ((a (ensure-contiguous-2d-vt matrix))
           (n (first (vt-shape a)))
           (lu (vt-copy a))          ; 原地存储 L+U
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
		 (error "矩阵奇异，无法进行 LU 分解"))
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
  "基于 LU 分解计算行列式。"
  (sb-int::with-float-traps-masked (:invalid)
    (multiple-value-bind (lu piv sign)
	(vt-lu matrix)
      (declare (ignore piv))
      (let ((n (first (vt-shape lu)))
            (det sign))
	(dotimes (i n)
          (setf det (* det (vt-ref lu i i))))
	det))))


(defun vt-solve (a b)
  "求解线性方程组 Ax = b（支持多右端项）。"
  (sb-int::with-float-traps-masked (:invalid)
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
	;; 1. 应用行置换 Pb
	;; CLTV的piv语义：最终LU的第i行 = 原始A的第piv[i]行
	;; 所以 Pb 的第 i 行 = 原始 b 的第 piv[i] 行
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
  (sb-int::with-float-traps-masked (:invalid)
    (let* ((n (first (vt-shape matrix)))
           (identity (vt-eye n :type (vt-element-type matrix))))
      (vt-solve matrix identity))))
