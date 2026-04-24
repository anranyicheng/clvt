(in-package :clvt)

(defun vt-dot (a b)
  "点积/内积，支持任意维度：
   - 若 a,b 均为 1D → 向量内积，返回数字。
   - 若 a,b 均为 2D → 矩阵乘法 a @ b。
   - 若 a,b 秩均 ≥2  → 批量矩阵乘法 '...ij,...jk->...ik'。
   其他情况（如 1D 与 2D）请直接使用 vt-einsum。"
  (let ((ra (length (vt-shape a)))
        (rb (length (vt-shape b))))
    (cond
      ((and (= ra 1) (= rb 1))
       (vt-ref (vt-einsum "i,i->" a b)))
      ((and (= ra 2) (= rb 2))
       (vt-einsum "ij,jk->ik" a b))
      ((and (>= ra 2) (>= rb 2))
       (vt-einsum "...ij,...jk->...ik" a b))
      (t
       (error "vt-dot: Unsupported dimensions (a: ~D, b: ~D).
               Use vt-einsum directly."
              ra rb)))))

(defun vt-outer (a b &key (flatten t))
  "计算张量外积。
   flatten = T (默认)
    : 先将输入展平为一维向量，再计算外积，返回二维矩阵。
      完全兼容 NumPy 的 outer 函数 (支持任意维度输入自动展平)。
   flatten = NIL
    : 保留输入的每个轴，将所有轴拼接形成新张量。
      例如 2D 与 3D → 5D 张量。
      等价于 (vt-einsum \"... ,...-> ... ...\"
       无法使用的替代显式下标写法)。"
  (if flatten
      (let ((flat-a (vt-flatten a))
            (flat-b (vt-flatten b)))
        (vt-einsum "i,j->ij" flat-a flat-b))
      (vt-einsum "...i,...j->...ij" a b)))

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
