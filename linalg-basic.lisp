(in-package :clvt)

(defun vt-dot (v1 v2)
  "向量内积: 纯用 einsum"
  (vt-einsum "i,i->" v1 v2))

(defun vt-outer (v1 v2)
  "向量/矩阵外积"
  (vt-einsum "i,j->ij" v1 v2))

(defun vt-trace (matrix)
  "矩阵迹: 对角线元素之和"
  (vt-sum (vt-diagonal matrix)))

(defun vt-norm (vt &key (axis nil))
  "L2 范数 (欧几里得范数)"
  (let ((sq (vt-square vt)))
    (if axis
        (vt-sqrt (vt-sum sq :axis axis))
        (vt-sqrt (vt-sum sq)))))

(defun vt-l1-norm (vt &key (axis nil))
  "L1 范数"
  (if axis
      (vt-sum (vt-abs vt) :axis axis)
      (vt-sum (vt-abs vt))))

(defun vt-frobenius-norm (matrix)
  "Frobenius 范数 (专用于矩阵)"
  (vt-norm matrix))
