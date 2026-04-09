(in-package :clvt)

;; 向量内积: 纯用 einsum
(defun vt-dot (v1 v2)
  (vt-einsum "i,i->" v1 v2))

;; 向量/矩阵外积
(defun vt-outer (v1 v2)
  (vt-einsum "i,j->ij" v1 v2))

;; 矩阵迹: 对角线元素之和
(defun vt-trace (matrix)
  (vt-sum (vt-diagonal matrix)))

;; L2 范数 (欧几里得范数)
(defun vt-norm (vt &optional (axis nil))
  (let ((sq (vt-square vt)))
    (if axis
        (vt-sqrt (vt-sum sq :axis axis))
        (vt-sqrt (vt-sum sq)))))

;; L1 范数
(defun vt-l1-norm (vt &optional (axis nil))
  (if axis
      (vt-sum (vt-abs vt) :axis axis)
      (vt-sum (vt-abs vt))))

;; Frobenius 范数 (专用于矩阵)
(defun vt-frobenius-norm (matrix)
  (vt-norm matrix))
