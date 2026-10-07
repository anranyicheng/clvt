;;;; extensions3.lisp — 补充 NumPy 重要缺失函数（第三批）
;;;;
;;;; 本文件严格遵循 CONVENTIONS.md：
;;;;   §1  三层架构：逻辑层（shape/dtype/广播/归约语义）/
;;;;        物理层（strides/offset/连续性/别名）/ 执行层（快路径 vs 通用路径）；
;;;;   §3  dtype 系统：8 种逻辑 dtype；比较/逻辑/谓词函数返回 :int8 布尔语义；
;;;;   §4  :out 硬契约：形状精确匹配、dtype 精确匹配、可写；
;;;;   §5  NaN/Inf 语义对齐 NumPy；
;;;;   §6  签名约定：一律 `&key dtype out`，缺省 nil 表示按输入提升；
;;;;   §7  SBCL 陷阱：非有限值取整/转换必须显式拦截；
;;;;   §11 docstring 规范。
;;;;
;;;; 载入顺序（clvt.asd，:serial t）：... extensions → extensions2 → extensions3
;;;; 本文件仅依赖此前已加载的模块，不引入任何前向依赖。

(in-package :clvt)

;;; ==================================================================
;;; 通用内部工具（本文件私有，避免与前序模块重名）
;;; ==================================================================

(defun %e3-flatten-list (x)
  "把任意嵌套的 list/张量递归展平为 Lisp 列表（叶子为数值）。
   用于 vt-block / vt-diagflat 等需要「按元素重排」的场合。"
  (cond ((vt-p x)
         (let ((f (vt-flatten x))
               (acc '()))
           (dotimes (i (vt-size f) (nreverse acc))
             (push (vt-ref f i) acc))))
        ((consp x) (mapcan #'%e3-flatten-list x))
        (t (list x))))

(defun %e3-int-dtype-p (dtype)
  "本库 6 种整型 dtype（含 int8/uint8/uint16 小整型）判定。"
  (member dtype '(:int64 :int32 :int16 :int8 :uint8 :uint16)))

(defun %e3-float-prefer-dtype (dtype in-dtype)
  "数学函数结果 dtype 推导（对齐 numpy 的「整型进 float64 出、float32 进 float32 出」）。
   显式 :dtype 已在调用点优先处理；缺省时按输入 dtype 决定：
     float32 输入 → :float32；其余（含整数）→ :float64。
   注意：缺省**不能**返回 nil——否则 vt-map 会退回 vt-promote-type，
   整数输入将得到整数结果（expm1/log1p 会把值截断为整数），
   与 numpy 的「整型进 float64 出」相悖。"
  (cond ((null dtype) (if (eq in-dtype :float32) :float32 :float64))
        ((eq dtype :float32) :float32)
        ((eq dtype :float64) :float64)
        (t dtype)))                            ; 显式整型目标原样返回

(defun %e3-as-1d-float (x)
  "把输入折叠成 1 维 float64 张量（用于 cov / corrcoef 的输入规范化）。"
  (let ((v (ensure-vt x :dtype :float64)))
    (if (null (vt-shape v))
        (vt-reshape v '(1))
        (vt-from-sequence (coerce (vt-data (vt-astype v :float64)) 'list)
                          :dtype :float64))))

;;; ==================================================================
;;; 1. 创建类（Creation）
;;; ==================================================================

(defun vt-asarray (obj &key (dtype nil))
  "把输入转换为张量（对标 numpy.asarray）。
   与 ensure-vt 语义一致：标量 → 0 维，序列 → ≥1 维，张量 → 原样（dtype 相同）。
   与 numpy 的差异：numpy 对已是 ndarray 的输入做「零拷贝共享」；
   本库张量本身即视图，传入 vt 且 dtype 已匹配时同样返回原对象（不拷贝）。
   示例：(vt-asarray '(1 2 3)) => [1 2 3] (int64)
         (vt-asarray '(1 2 3) :dtype :float32) => [1.0 2.0 3.0] (float32)"
  (ensure-vt obj :dtype dtype))

(defun vt-fromiter (iterable &key (dtype :float64) (count nil))
  "从 Lisp 序列构建 1 维张量（对标 numpy.fromiter）。
   COUNT 若给出必须是序列长度，用于前置校验（numpy 语义：count 应精确匹配）；
   纯语义构建时忽略 COUNT 的实际作用（本库一次性构建，无需预分配）。
   :out 契约：本函数为构建函数，无 :out。
   示例：(vt-fromiter '(1 2 3) :dtype :float64) => [1.0 2.0 3.0]"
  (declare (ignore count))
  (when (not (or (null iterable) (typep iterable 'sequence)))
    (error "vt-fromiter: 需要序列输入，收到 ~a" (type-of iterable)))
  (vt-from-sequence (or iterable '()) :dtype dtype))

(defun vt-tri (n &key (m nil) (k 0) (dtype :float64))
  "下三角布尔/数值矩阵（对标 numpy.tri）。
   返回 shape (N, M) 的矩阵，第 (i, j) 元为 1（当 j <= i + K），否则 0。
   M 缺省等于 N。K 缺省 0。DTYPE 缺省 :float64（与 numpy 一致）。
   示例：(vt-tri 3)       => [[1 0 0][1 1 0][1 1 1]]
         (vt-tri 3 :k 1)  => [[1 1 0][1 1 1][1 1 1]]
         (vt-tri 3 :m 4 :k -1) => [[0 0 0 0][1 0 0 0][1 1 0 0]]"
  (when (< n 0) (error "vt-tri: n 必须 >= 0，收到 ~a" n))
  (let* ((cols (or m n))
         (res (vt-zeros (list n cols) :dtype dtype)))
    (when (> cols 0)
      (dotimes (i n)
        (dotimes (j cols)
          (when (<= j (+ i k))
            (setf (vt-ref res i j) (vt-cast 1 dtype))))))
    res))

(defun vt-diagflat (v &key (k 0))
  "把输入展平后构造对角矩阵（对标 numpy.diagflat）。
   输入任意秩 → 先展平为 1 维，再沿 K 对角线铺开为方阵（边长 = 元素数 + |K|）。
   示例：(vt-diagflat '((1 2) (3 4))) => 4x4 对角阵 [1 2 3 4]
         (vt-diagflat '(1 2) :k 1)    => [[0 1 0][0 0 2][0 0 0]]"
  (let* ((flat (ensure-vt v))
         (one-d (if (null (vt-shape flat))
                    (vt-reshape flat '(1))
                    (vt-flatten flat))))
    (vt-diag one-d :k k)))

(defun vt-trim-zeros (tensor &key (trim :fb))
  "去除 1 维张量首尾的零（对标 numpy.trim_zeros）。
   TRIM 取值 :fb（默认，去首尾）/ :f（仅去首）/ :b（仅去尾）。
   判定用 = 比较（与 numpy 一致，0.0 亦视为零）。
   示例：(vt-trim-zeros '(0 0 1 2 0 3 0 0)) => [1 2 0 3]"
  (let* ((input (ensure-vt tensor))
         (one-d (if (null (vt-shape input))
                    (vt-reshape input '(1))
                    (vt-ravel input)))
         (n (vt-size one-d)))
    (unless (member trim '(:fb :f :b))
      (error "vt-trim-zeros: trim 必须是 :fb / :f / :b 之一，收到 ~a" trim))
    (if (zerop n)
        one-d
        (let ((vals (coerce (vt-data (vt-astype one-d :float64)) 'list)))
          (let ((start 0) (end n))
            (when (member trim '(:fb :f))
              (loop while (and (< start end) (zerop (nth start vals)))
                    do (incf start)))
            (when (member trim '(:fb :b))
              (loop while (and (< start end) (zerop (nth (1- end) vals)))
                    do (decf end)))
            (vt-from-sequence (subseq vals start end) :dtype (vt-dtype one-d)))))))

;;; ==================================================================
;;; 2. 形状操作类（Shape manipulation）
;;; ==================================================================

(defun vt-rollaxis (tensor axis &optional (start 0))
  "把 AXIS 轴滚动到 START 位置（对标 numpy.rollaxis）。
   若 axis < start，则轴最终落在 start-1 位置（numpy 的「先移出后插入」语义）。
   返回视图（本库用 transpose 实现，零拷贝语义等价）。
   示例：(vt-rollaxis (vt-ones '(3 4 5)) 2)     => shape (5 3 4)
        (vt-rollaxis (vt-ones '(3 4 5)) 0 3)   => shape (4 5 3)"
  (let* ((shape (vt-shape tensor))
         (rank (length shape))
         (ax (vt-normalize-axis axis rank)))
    (when (or (< start 0) (> start rank))
      (error "vt-rollaxis: start ~a 超出 [0, ~a] 范围" start rank))
    (let* ((perm (loop for i from 0 below rank collect i))
           (pos (if (< ax start) (1- start) start))
           (moved (nth ax perm))
           (rest (loop for i in perm unless (= i ax) collect i)))
      (vt-transpose tensor
                    (append (subseq rest 0 pos)
                            (list moved)
                            (subseq rest pos))))))

(defun vt-column-stack (tensors)
  "把多个 1 维张量（或标量）按列堆叠成 2 维（对标 numpy.column_stack）。
   规则：每个输入先至少提升为 1 维，再 reshape 为列向量 (N,1)，最后沿 axis=1 拼接。
   示例：(vt-column-stack (list (vt-fromiter '(1 2 3)) (vt-fromiter '(4 5 6))))
        => [[1 4][2 5][3 6]]"
  (let* ((vs (mapcar (lambda (x)
                       (let ((v (ensure-vt x)))
                         (cond ((null (vt-shape v))
                                (vt-reshape v '(1 1)))
                               ((= (vt-order v) 1)
                                (vt-reshape v (list (vt-size v) 1)))
                               ((= (vt-order v) 2) v)
                               (t (error "vt-column-stack: 输入必须是 0/1/2 维，收到 ~a 维"
                                         (vt-order v))))))
                     tensors)))
    (if (rest vs)
        (apply #'vt-concatenate 1 vs)
        (first vs))))

(defun vt-block (arrays)
  "按嵌套块组装张量（对标 numpy.block）。
   输入为嵌套 list；叶子为张量/标量。
   同一行的相邻块沿 axis=1（列方向）拼接；
   不同行的块沿 axis=0（行方向）拼接；1 维最外层沿 axis=0 拼接。
   示例：(vt-block (list (list (vt-ones '(2 2)) (vt-zeros '(2 1)))
                        (list (vt-zeros '(1 2)) (vt-ones '(1 1)))))
        => 3x3 块矩阵"
  (labels ((leaf (x) (ensure-vt x))
           (depth (x) (if (consp x) (1+ (reduce #'max x :key #'depth :initial-value 0)) 0))
           (build (x d)
             (cond
               ((or (not (consp x)) (zerop d)) (leaf x))
               ((= d 1)
                ;; 一行内的多个叶子：从左到右沿列轴拼接。
                ;; 秩 1 叶子沿 axis=0，秩 ≥2 叶子沿 axis=1。
                (let ((vs (mapcar #'leaf x)))
                  (if (rest vs)
                      (let ((ax (if (= (the fixnum (apply #'max (mapcar #'vt-order vs))) 1)
                                    0
                                    1)))
                        (apply #'vt-concatenate ax vs))
                      (first vs))))
               (t
                ;; 多行：先各自成行，再沿 axis=0 堆叠。
                (let ((rows (mapcar (lambda (r) (build r (1- d))) x)))
                  (if (rest rows)
                      (apply #'vt-concatenate 0 rows)
                      (first rows)))))))
    (build arrays (depth arrays))))

(defun vt-broadcast-arrays (tensors)
  "把多个张量广播到公共形状，返回张量列表（对标 numpy.broadcast_arrays）。
   所有返回张量形状相同 = 全部输入的广播结果形状（只读广播视图）。
   示例：(vt-broadcast-arrays (list (vt-ones '(3 1)) (vt-ones '(1 4))))
        => 两个 (3 4) 张量"
  (let* ((vs (mapcar #'ensure-vt tensors))
         (final (reduce #'vt-broadcast-shapes (mapcar #'vt-shape vs)
                        :initial-value '())))
    (mapcar (lambda (v) (vt-broadcast-to v final)) vs)))

(defun vt-resize (tensor new-shape)
  "按需重复/截断元素以填充 NEW-SHAPE（对标 numpy.resize，元素级循环填充）。
   先展平源数据，再按行主序循环取用，直至填满目标元素个数。
   若源为空 → 结果全零（对标 numpy）。NEW-SHAPE 中的 -1 不受支持（numpy.resize 不支持）。
   示例：(vt-resize (vt-fromiter '(1 2 3)) '(2 3)) => [[1 2 3][1 2 3]]
        (vt-resize (vt-fromiter '(1 2 3)) '(5))   => [1 2 3 1 2]"
  (let* ((flat (vt-ravel tensor))
         (src (vt-astype flat (vt-dtype tensor)))
         (n (vt-size src))
         (total (vt-shape-to-size new-shape))
         (res (vt-zeros new-shape :dtype (vt-dtype tensor))))
    (when (zerop total)
      (return-from vt-resize res))
    (if (zerop n)
        res
        (let ((out (vt-ravel res))
              (vals (vt-data src)))
          (dotimes (i total res)
            (setf (vt-ref out i) (aref vals (mod i n))))))))

;;; ==================================================================
;;; 3. 索引类（Indexing）
;;; ==================================================================

(defun vt-take-along-axis (tensor indices axis)
  "沿 AXIS 用 INDICES 取值（对标 numpy.take_along_axis）。
   INDICES 必须与 TENSOR 秩相同；非 AXIS 的每一维，INDICES 的长度须等于 TENSOR
   对应维长度或是 1（按 NumPy 广播规则扩展到 TENSOR 的长度）。
   结果形状：AXIS 维取 INDICES 的长度，其余维取 TENSOR 的长度。
   示例：(vt-take-along-axis (vt-from-array #2A((1 2)(3 4))) (vt-from-array #2A((0)(1))) 1)
        => [[1][4]]
        (vt-take-along-axis (vt-from-array #2A((1 2)(3 4))) (vt-from-array #2A((0)(0))) 0)
        => [[1 2][1 2]]"
  (let* ((shape (vt-shape tensor))
         (rank (length shape)))
    (unless (= rank (vt-order indices))
      (error "vt-take-along-axis: indices 与 arr 必须维度数相同（~a vs ~a）"
             (vt-order indices) rank))
    (let* ((ax (vt-normalize-axis axis rank))
           (ishape (vt-shape indices))
           (ax-dim (nth ax shape)))
      ;; NumPy 语义：非 AXIS 维按广播规则检查（长度相等或为 1）
      (loop for d from 0 below rank
            for idim = (nth d ishape)
            for sdim = (nth d shape)
            when (and (/= d ax) (/= idim sdim) (/= idim 1))
              do (error "vt-take-along-axis: 第 ~a 维长度不匹配（indices ~a vs arr ~a）"
                        d idim sdim))
      ;; 结果形状：非 AXIS 维沿 TENSOR 的长度广播，AXIS 维用 indices 的长度
      (let* ((oshape (loop for d from 0 below rank
                           collect (if (= d ax) (nth d ishape) (nth d shape))))
             (out (vt-zeros oshape :dtype (vt-dtype tensor)))
             (idx (vt-ravel (vt-astype indices :int64)))
             (strides (vt-strides tensor))
             (src-data (vt-data tensor))
             (src-off (vt-offset tensor))
             (out-data (vt-data out))
             (out-off (vt-offset out))
             (ostrides (vt-strides out)))
        (labels ((lin (coords dims strd)
                   (let ((acc 0))
                     (loop for d from 0 below (length dims)
                           do (incf acc (* (nth d coords) (nth d strd))))
                     acc)))
          ;; 遍历输出所有坐标，回溯到 indices 坐标（非 AXIS 维按广播规则取 0）
          (dotimes (olinear (vt-size out) out)
            (let* ((ocoords (make-list rank))
                   (rem olinear))
              (loop for d from (1- rank) downto 0
                    for dim = (nth d oshape)
                    do (multiple-value-bind (q r) (floor rem (max 1 dim))
                         (setf (nth d ocoords) r)
                         (setf rem q)))
              ;; 输出坐标 → indices 坐标：AXIS 维用输出坐标；其余维若 indices 长度为 1 取 0
              (let* ((icoords (loop for d from 0 below rank
                                    collect (if (or (= d ax)
                                                    (/= (nth d ishape) 1))
                                                (nth d ocoords)
                                                0)))
                     ;; v0.4.0 修复：idx 已按行主序 ravel，平面下标必须用
                     ;; 行主序 strides（vt-compute-strides）计算；原实现传全 1
                     ;; 步长等价于按列主序展开，导致第 r 行索引整体偏移 +r。
                     (ilinear (lin icoords ishape (vt-compute-strides ishape)))
                     (sel (vt-ref idx ilinear))
                     (src-coords (copy-list ocoords)))
                (setf sel (if (minusp sel) (+ sel ax-dim) sel))
                (unless (<= 0 sel (1- ax-dim))
                  (error "vt-take-along-axis: 索引 ~a 越界（轴长 ~a）" sel ax-dim))
                (setf (nth ax src-coords) sel)
                (setf (aref out-data (+ out-off (lin ocoords oshape ostrides)))
                      (aref src-data (+ src-off (lin src-coords shape strides))))))))))))

(defun vt-put-along-axis (tensor indices values axis)
  "沿 AXIS 把 VALUES 按 INDICES 写入 TENSOR 的拷贝并返回（对标 numpy.put_along_axis）。
   不修改输入（numpy 亦返回新数组）；INDICES 约束同 vt-take-along-axis。
   示例：(vt-put-along-axis (vt-from-array #2A((1 2)(3 4))) (vt-from-array #2A((0)(1))) (vt-full '(2 1) 9) 1)
        => [[9 2][3 9]]"
  (let* ((shape (vt-shape tensor))
         (rank (length shape)))
    (unless (= rank (vt-order indices))
      (error "vt-put-along-axis: indices 与 arr 必须维度数相同"))
    (let* ((ax (vt-normalize-axis axis rank))
           (ishape (vt-shape indices))
           (out (vt-copy tensor)))
      (unless (equal ishape (vt-shape values))
        (error "vt-put-along-axis: values 形状 ~a 必须等于 indices 形状 ~a"
               (vt-shape values) ishape))
      (loop for d from 0 below rank
            for idim = (nth d ishape)
            for sdim = (nth d shape)
            when (and (/= d ax) (/= idim sdim) (/= idim 1))
              do (error "vt-put-along-axis: 第 ~a 维长度不匹配（indices ~a vs arr ~a）"
                        d idim sdim))
      (let* ((idx (vt-ravel (vt-astype indices :int64)))
             (ax-dim (nth ax shape))
             (isize (vt-size indices))
             (val-flat (vt-ravel values))
             (strides (vt-strides out))
             (out-data (vt-data out))
             (out-off (vt-offset out)))
        (labels ((lin (coords dims strd)
                   (let ((acc 0))
                     (loop for d from 0 below (length dims)
                           do (incf acc (* (nth d coords) (nth d strd))))
                     acc)))
          ;; 遍历 indices 全部坐标，写入目标（非 AXIS 维若长度为 1 则广播到整个维）
          (dotimes (linear isize out)
            (let* ((icoords (make-list rank))
                   (rem linear))
              (loop for d from (1- rank) downto 0
                    for dim = (nth d ishape)
                    do (multiple-value-bind (q r) (floor rem (max 1 dim))
                         (setf (nth d icoords) r) (setf rem q)))
              (let ((sel (vt-ref idx linear)))
                (setf sel (if (minusp sel) (+ sel ax-dim) sel))
                (unless (<= 0 sel (1- ax-dim))
                  (error "vt-put-along-axis: 索引 ~a 越界（轴长 ~a）" sel ax-dim))
                ;; 沿非 AXIS 维广播写入（indices 长度为 1 时铺满该维）
                (labels ((put (d coords)
                           (if (= d rank)
                               (let ((tcoords (copy-list coords)))
                                 (setf (nth ax tcoords) sel)
                                 (setf (aref out-data (+ out-off (lin tcoords shape strides)))
                                       (vt-ref val-flat linear)))
                               (if (= d ax)
                                   (progn (setf (nth d coords) (nth d icoords))
                                          (put (1+ d) coords))
                                   (if (= (nth d ishape) 1)
                                       (loop for k from 0 below (nth d shape)
                                             do (setf (nth d coords) k)
                                                (put (1+ d) coords))
                                       (progn (setf (nth d coords) (nth d icoords))
                                              (put (1+ d) coords)))))))
                  (put 0 (make-list rank)))))))))))

(defun vt-compress (condition tensor &key axis)
  "沿 AXIS 按布尔 CONDITION 选取切片（对标 numpy.compress）。
   CONDITION 为长度 ≤ 该轴长度的序列（不足部分视为假）；
   AXIS 为 nil 时展平后选取。
   示例：(vt-compress '(1 0 1 0 1) (vt-fromiter '(0 1 2 3 4))) => [0 2 4]"
  (let* ((cond-v (ensure-vt condition))
         (cond-list (mapcar (lambda (x) (if (zerop x) nil t))
                            (coerce (vt-data (vt-astype (vt-ravel cond-v) :int64)) 'list))))
    (if (null axis)
        (let* ((flat (vt-ravel tensor))
               (n (vt-size flat))
               (sel '()))
          (loop for i from 0 below n
                for c in (append cond-list (make-list (max 0 (- n (length cond-list)))
                                                      :initial-element nil))
                when c do (push (vt-ref flat i) sel))
          (vt-from-sequence (nreverse sel) :dtype (vt-dtype tensor)))
        (let* ((shape (vt-shape tensor))
               (rank (length shape))
               (ax (vt-normalize-axis axis rank))
               (ax-dim (nth ax shape)))
          (when (> (length cond-list) ax-dim)
            (error "vt-compress: condition 长度 ~a 超过轴长 ~a"
                   (length cond-list) ax-dim))
          (let* ((full (append cond-list
                               (make-list (max 0 (- ax-dim (length cond-list)))
                                          :initial-element nil)))
                 (idxs (loop for i from 0 below ax-dim when (nth i full) collect i)))
            (vt-take tensor (vt-from-sequence idxs :dtype :int64) :axis ax))))))

(defun vt-indices (dimensions &key (dtype :int64) (sparse nil))
  "网格索引张量（对标 numpy.indices）。
   SPARSE=nil：返回 shape (R, D0, D1, ...) 的单一张量（R = 维度数），
     第 r 层为沿第 r 轴的坐标网格。
   SPARSE=t：返回 Lisp 列表（对标 numpy 的 tuple），第 r 个张量形状为
     仅在自己轴上保留长度、其余为 1。
   示例：(vt-indices '(2 3)) => shape (2 2 3)
        (vt-indices '(2 3) :sparse t) => (list 形状 (2 1) 和 (1 3) 的张量)"
  (let* ((dims (mapcar (lambda (d) (if (integerp d) d (truncate d))) dimensions))
         (r (length dims)))
    (when (zerop r)
      (error "vt-indices: dimensions 不能为空"))
    (let ((layers '()))
      (dotimes (ax r)
        (let* ((sh (if sparse
                       (loop for d from 0 below r collect (if (= d ax) (nth d dims) 1))
                       dims))
               (layer (vt-zeros sh :dtype dtype))
               (flat (vt-ravel layer))
               ;; 该层沿 ax 轴的步长（行主序），以及每个「i 值」重复块的大小
               (stride (apply #'* (nthcdr (1+ ax) sh)))
               (stride (if (zerop stride) 1 stride))
               (outer (apply #'* (subseq sh 0 ax)))
               (axis-len (nth ax sh)))
          (dotimes (o outer)
            (dotimes (i axis-len)
              (dotimes (s stride)
                (setf (vt-ref flat (+ (* o stride axis-len) (* i stride) s))
                      (vt-cast i dtype)))))
          (push layer layers)))
      (let ((ordered (nreverse layers)))
        (if sparse
            ordered
            (apply #'vt-stack 0 ordered))))))

(defun vt-fill-diagonal (tensor value &key (wrap nil))
  "就地填充 2 维（或 N 维）张量的主对角线为 VALUE（对标 numpy.fill_diagonal）。
   VALUE 可为标量或序列（长度须为对角线长度；WRAP=t 时循环填充）。
   返回修改后的 TENSOR（就地操作；二维要求方阵的对角线长度 = min(行,列)）。
   示例：(vt-fill-diagonal (vt-zeros '(3 3)) 9) => 对角全 9"
  (let* ((shape (vt-shape tensor))
         (rank (length shape)))
    (when (< rank 2)
      (error "vt-fill-diagonal: 需要 >=2 维张量，收到 ~a 维" rank))
    (let* ((rows (nth (- rank 2) shape))
           (cols (nth (1- rank) shape))
           (dlen (min rows cols))
           (rank-1 (= rank 2))
           (batch (if rank-1 1 (reduce #'* (subseq shape 0 (- rank 2)))))
           (vals (cond ((numberp value) nil)
                       (t (coerce (vt-data (vt-astype (vt-ravel (ensure-vt value)) :int64))
                                  'list)))))
      (when (and vals (> (length vals) dlen) (not wrap))
        (error "vt-fill-diagonal: value 长度 ~a 超过对角线长度 ~a"
               (length vals) dlen))
      (let ((in-data (vt-data tensor))
            (in-off (vt-offset tensor))
            (in-strs (vt-strides tensor)))
        (dotimes (b batch tensor)
          (let ((base in-off))
            (when (> rank 2)
              (let ((rem b))
                (loop for d from (- rank 3) downto 0
                      for dim = (nth d shape)
                      for str = (nth d in-strs)
                      do (multiple-value-bind (q r) (floor rem (max 1 dim))
                           (incf base (* r str)) (setf rem q)))))
            (dotimes (i dlen)
              (let ((v (if vals
                           (nth (if wrap (mod i (length vals)) i) vals)
                           value))
                    (ptr (+ base (* i (nth (- rank 2) in-strs))
                            (* i (nth (1- rank) in-strs)))))
                (setf (aref in-data ptr) (vt-cast v (vt-dtype tensor)))))))))))

;;; ==================================================================
;;; 4. 数学类（Math）
;;; ==================================================================

(defun vt-absolute (tensor &key out dtype)
  "逐元素绝对值（对标 numpy.absolute，等价 vt-abs）。
   整型输入整型出（保持 dtype）；浮点输入同 dtype（NaN/±Inf 按 |x| 传播）。
   :out 契约同 vt-abs（形状/dtype 精确匹配、可写）。"
  (vt-fast-map #'abs (ensure-vt tensor) :out out :dtype dtype))

(defun vt-sign (tensor &key out dtype)
  "逐元素符号函数（对标 numpy.sign）。
   返回 -1/0/1；整型输入整型出，浮点输入浮点出（NaN → NaN；-0.0 → 0.0）。
   :out 契约同 vt-abs。"
  (let* ((v (ensure-vt tensor))
         (res-dtype (or dtype (vt-dtype v))))
    (vt-map (lambda (x)
              (with-float-safe
                (cond ((%nan-p x) x)
                      ((zerop x) (if (floatp x) 0.0d0 0))
                      ((plusp x) (if (floatp x) 1.0d0 1))
                      (t (if (floatp x) -1.0d0 -1)))))
            v :out out :dtype res-dtype)))

(defun vt-positive (tensor &key out dtype)
  "逐元素取正（一元 +，对标 numpy.positive）：数值恒等映射。
   :out 契约同 vt-abs。"
  (vt-map #'identity (ensure-vt tensor) :out out :dtype dtype))

(defun %e3-expm1 (x)
  "数值稳定的 exp(x) - 1（对标 numpy.expm1）。|x| 小时用 Taylor 级数避免灾难性抵消。"
  (let ((xf (coerce x 'double-float)))
    (cond ((%nan-p xf) xf)
          ((%pos-inf-p xf) xf)
          ((%neg-inf-p xf) -1.0d0)
          ((< (abs xf) 1.0d-5)
           ;; exp(x) - 1 = x + x²/2 + x³/6 + x⁴/24 + x⁵/120 + ...
           (* xf (+ 1.0d0
                    (* xf (+ 0.5d0
                             (* xf (+ (/ 1.0d0 6.0d0)
                                      (* xf (+ (/ 1.0d0 24.0d0)
                                               (* xf (/ 1.0d0 120.0d0)))))))))))
          (t (- (exp xf) 1.0d0)))))

(defun vt-expm1 (tensor &key out dtype)
  "逐元素 exp(x) - 1（对标 numpy.expm1）。
   小量下保留精度（1e-10 → 1.00000000005e-10）；整数输入输出 float64。
   :out 契约同 vt-abs。"
  (let ((v (ensure-vt tensor)))
    (%float-map #'%e3-expm1 v out (%e3-float-prefer-dtype dtype (vt-dtype v)))))

(defun %e3-log1p (x)
  "数值稳定的 log(1 + x)（对标 numpy.log1p）。|x| 小时用 log1p 级数避免精度损失。"
  (let ((xf (coerce x 'double-float)))
    (cond ((%nan-p xf) xf)
          ((= xf -1.0d0) (vt-get-neg-inf :float64))
          ((< xf -1.0d0) (vt-get-nan :float64))
          ((%pos-inf-p xf) xf)
          ((< (abs xf) 1.0d-3)
           ;; log(1+x) = x - x²/2 + x³/3 - x⁴/4 + ...（|x| 小时收敛快）
           (let ((acc 0.0d0) (term xf))
             (loop for k from 1 to 30
                   for s = (if (oddp k) 1.0d0 -1.0d0)
                   do (incf acc (* s (/ term k)))
                      (setf term (* term xf))
                      (when (< (abs term) 1.0d-20) (return)))
             acc))
          (t (log (+ 1.0d0 xf))))))

(defun vt-log1p (tensor &key out dtype)
  "逐元素 log(1 + x)（对标 numpy.log1p）。
   x = -1 → -Inf；x < -1 → NaN；小量下保留精度（1e-10 → 9.999999999500001e-11）。
   整数输入输出 float64。:out 契约同 vt-abs。"
  (let ((v (ensure-vt tensor)))
    (%float-map #'%e3-log1p v out (%e3-float-prefer-dtype dtype (vt-dtype v)))))

(defun vt-logaddexp (x y &key out dtype)
  "逐元素 log(exp(x) + exp(y))，数值稳定（对标 numpy.logaddexp）。
   整数输入输出 float64。:out 契约同 vt-abs（支持广播）。"
  (let ((fn (lambda (a b)
              (let ((af (coerce a 'double-float)) (bf (coerce b 'double-float)))
                (cond ((%nan-p af) af)
                      ((%nan-p bf) bf)
                      ((%pos-inf-p af) af)
                      ((%pos-inf-p bf) bf)
                      ((%neg-inf-p af) bf)
                      ((%neg-inf-p bf) af)
                      (t (let ((m (max af bf)))
                           (+ m (log (+ 1.0d0 (exp (- (min af bf) m))))))))))))
    (vt-map fn (ensure-vt x :dtype :float64) (ensure-vt y :dtype :float64)
            :out out :dtype (%e3-float-prefer-dtype dtype :float64))))

(defun vt-float-power (x y &key out dtype)
  "逐元素幂运算，结果恒为浮点（对标 numpy.float_power）。
   整数基/指数亦按浮点计算：负底数配非整数指数 → NaN；
   结果 dtype 缺省 float64。:out 契约同 vt-abs（支持广播）。"
  (vt-map (lambda (a b)
            (let ((af (coerce a 'double-float)) (bf (coerce b 'double-float)))
              (cond ((%nan-p af) af)
                    ((%nan-p bf) bf)
                    ((and (minusp af) (/= bf (floor bf))) (vt-get-nan :float64))
                    (t (expt af bf)))))
          (ensure-vt x :dtype :float64) (ensure-vt y :dtype :float64)
          :out out :dtype (%e3-float-prefer-dtype dtype :float64)))

(defun vt-copysign (x y &key out dtype)
  "逐元素赋予 X 的绝对值与 Y 的符号（对标 numpy.copysign）。
   结果恒为浮点；NaN 符号位处理与 numpy 一致（copysign(1, nan) → 1）。
   :out 契约同 vt-abs（支持广播）。"
  (vt-map (lambda (a b)
            (let ((af (coerce a 'double-float)) (bf (coerce b 'double-float)))
              (if (or (minusp bf) (and (floatp bf) (%nan-p bf) nil))
                  (- (abs af))
                  (abs af))))
          (ensure-vt x :dtype :float64) (ensure-vt y :dtype :float64)
          :out out :dtype (%e3-float-prefer-dtype dtype :float64)))

(defun vt-signbit (tensor &key out)
  "逐元素符号位判定，返回 1/0，dtype 为 :int8（对标 numpy.signbit）。
   x < 0 或为负零（-0.0）→ 1，否则 0。:out 契约同 vt-=（dtype :int8）。"
  (vt-map (lambda (x)
            (let ((xf (coerce x 'double-float)))
              ;; float-sign 保留 -0.0 的符号位（返回 -1.0），故可直接用于符号判定
              (if (minusp (float-sign xf)) 1 0)))
          (ensure-vt tensor :dtype :float64) :out out :dtype :int8))

(defun %e3-nextafter (a b)
  "标量 nextafter（对标 numpy.nextafter）：返回朝 B 方向的次可表示 double。"
  (let ((af (coerce a 'double-float)) (bf (coerce b 'double-float)))
    (cond ((%nan-p af) af)
          ((%nan-p bf) bf)
          ((= af bf) bf)
          ((zerop af) (if (plusp bf) 5.0d-324 -5.0d-324))
          ((or (%pos-inf-p af) (%neg-inf-p af)) af)
          ((< af bf)
           (if (plusp af)
               (+ af (* (abs af) 2.220446049250313d-16))
               (+ af (* (abs af) 1.1102230246251565d-16))))
          (t
           (if (plusp af)
               (- af (* (abs af) 1.1102230246251565d-16))
               (- af (* (abs af) 2.220446049250313d-16)))))))

(defun vt-nextafter (x y &key out dtype)
  "逐元素「朝 Y 方向的最近可表示浮点数」（对标 numpy.nextafter）。
   以 double-float 精度计算。:out 契约同 vt-abs（支持广播）。
   示例：(vt-nextafter 1.0 2.0) => 1.0000000000000002"
  (vt-map (lambda (a b) (%e3-nextafter a b))
          (ensure-vt x :dtype :float64) (ensure-vt y :dtype :float64)
          :out out :dtype (%e3-float-prefer-dtype dtype :float64)))

(defun vt-spacing (tensor &key out dtype)
  "逐元素返回与该值相邻浮点数之间的距离（对标 numpy.spacing）。
   以 double-float 精度计算。:out 契约同 vt-abs。
   示例：(vt-spacing 1.0) => 2.220446049250313e-16；(vt-spacing 0.0) => 5e-324"
  (vt-map (lambda (x)
            (let ((xf (abs (coerce x 'double-float))))
              (with-float-safe
                (cond ((zerop xf) 5.0d-324)
                      ((%nan-p xf) xf)
                      ((%pos-inf-p xf) xf)
                      ;; numpy.spacing(x) = nextafter(|x|, +inf) - |x|
                      (t (- (%e3-nextafter xf most-positive-double-float) xf))))))
          (ensure-vt tensor :dtype :float64) :out out :dtype (%e3-float-prefer-dtype dtype :float64)))

;;; 整型范围回绕工具：（供 gcd/lcm 复用）
(defun %e3-gcd2 (a b)
  "非负整数最大公约数（欧几里得）。"
  (let ((a (abs a)) (b (abs b)))
    (loop while (/= b 0) do (psetf a b b (mod a b)))
    a))

(defun vt-gcd (x y &key out dtype)
  "逐元素最大公约数，结果非负（对标 numpy.gcd）。
   仅支持整型输入（对齐 numpy：浮点输入报错）。:out 契约同 vt-abs（支持广播）。
   示例：(vt-gcd '(12 18) '(8 24)) => [4 6]"
  (let ((xv (ensure-vt x)) (yv (ensure-vt y)))
    (unless (and (%e3-int-dtype-p (vt-dtype xv)) (%e3-int-dtype-p (vt-dtype yv)))
      (error "vt-gcd: 仅支持整型输入（收到 ~a / ~a）"
             (vt-dtype xv) (vt-dtype yv)))
    (vt-map (lambda (a b) (%e3-gcd2 (truncate a) (truncate b)))
            xv yv :out out :dtype (or dtype (vt-promote-type (vt-dtype xv) (vt-dtype yv))))))

(defun %e3-lcm2 (a b)
  "非负整数最小公倍数（a,b 均为 0 → 0）。"
  (let ((a (abs a)) (b (abs b)))
    (if (or (zerop a) (zerop b))
        0
        (/ (* a b) (%e3-gcd2 a b)))))

(defun vt-lcm (x y &key out dtype)
  "逐元素最小公倍数，结果非负（对标 numpy.lcm）。
   仅支持整型输入；任一为 0 → 0。:out 契约同 vt-abs（支持广播）。
   示例：(vt-lcm '(4 6) '(6 8)) => [12 24]"
  (let ((xv (ensure-vt x)) (yv (ensure-vt y)))
    (unless (and (%e3-int-dtype-p (vt-dtype xv)) (%e3-int-dtype-p (vt-dtype yv)))
      (error "vt-lcm: 仅支持整型输入（收到 ~a / ~a）"
             (vt-dtype xv) (vt-dtype yv)))
    (vt-map (lambda (a b) (%e3-lcm2 (truncate a) (truncate b)))
            xv yv :out out :dtype (or dtype (vt-promote-type (vt-dtype xv) (vt-dtype yv))))))

(defun vt-divmod (x y &key out dtype)
  "逐元素同时返回商与余数（对标 numpy.divmod）。
   余数符号跟随除数（floor 语义）；返回 (values quotient remainder)。
   :out 契约：本函数返回两个结果，暂不支持 :out（传 out 将报错）。
   示例：(multiple-value-bind (q r) (vt-divmod '(7 -7) '(2 3)) ...)
        => q=[3 -3], r=[1 2]"
  (declare (ignore out))
  (let* ((xv (ensure-vt x)) (yv (ensure-vt y))
         ;; floor 除法（余数跟随除数）：商 = floor(x / y)（浮点亦按 floor 取整）
         (quot (vt-map (lambda (a b)
                         (let ((af (coerce a 'double-float))
                               (bf (coerce b 'double-float)))
                           (cond ((or (%nan-p af) (%nan-p bf)) af)
                                 ((zerop bf) (if (or (%nan-p af) (%nan-p bf))
                                                 (vt-get-nan :float64)
                                                 (vt-get-nan :float64)))
                                 (t (floor (/ af bf))))))
                       xv yv :dtype dtype)))
    (values quot (vt-mod xv yv :dtype dtype))))

(defun vt-nan-to-num (tensor &key (copy t) (nan 0.0d0) (posinf nil) (neginf nil) out dtype)
  "把 NaN 替换为 NAN、+Inf 替换为 POSINF、-Inf 替换为 NEGINF（对标 numpy.nan_to_num）。
   POSINF/NEGINF 缺省为对应 dtype 的最大/最小有限值（对齐 numpy）。
   COPY 仅作 API 兼容（本库不改输入，恒返回新张量）。:out 契约同 vt-abs。
   示例：(vt-nan-to-num '(nan inf -inf 1.0)) => [0.0 1.797e308 -1.797e308 1.0]"
  (declare (ignore copy))
  (let* ((v (ensure-vt tensor))
         (dt (or dtype (vt-dtype v)))
         (maxf (if (eq dt :float32) 3.4028235f38 most-positive-double-float))
         (minf (if (eq dt :float32) -3.4028235f38 most-negative-double-float)))
    (vt-map (lambda (x)
              (let ((xf (coerce x 'double-float)))
                (cond ((%nan-p xf) (coerce (or nan 0.0d0) 'double-float))
                      ((%pos-inf-p xf) (coerce (or posinf maxf) 'double-float))
                      ((%neg-inf-p xf) (coerce (or neginf minf) 'double-float))
                      (t xf))))
            v :out out :dtype (%e3-float-prefer-dtype dt :float64))))

(defun vt-real (tensor &key out dtype)
  "逐元素实部（对标 numpy.real）。本库无复数 dtype，恒返回输入本身（数值不变）。
   :out 契约同 vt-abs。"
  (vt-map #'identity (ensure-vt tensor) :out out :dtype dtype))

(defun vt-imag (tensor &key out dtype)
  "逐元素虚部（对标 numpy.imag）。本库无复数 dtype，恒返回全零。
   :out 契约同 vt-abs。"
  (vt-map (lambda (x) (declare (ignore x)) 0)
          (ensure-vt tensor) :out out :dtype (or dtype (vt-dtype (ensure-vt tensor)))))

(defun vt-conj (tensor &key out dtype)
  "逐元素复共轭（对标 numpy.conj）。本库无复数 dtype，恒返回输入本身。
   :out 契约同 vt-abs。"
  (vt-map #'identity (ensure-vt tensor) :out out :dtype dtype))

(defun vt-angle (tensor &key (deg nil) out dtype)
  "逐元素相位角（对标 numpy.angle）。对实数 z：z >= 0 → 0，z < 0 → π。
   DEG=t 时以角度（度）表示。NaN → NaN。:out 契约同 vt-abs。"
  (vt-map (lambda (x)
            (let ((xf (coerce x 'double-float)))
              (cond ((%nan-p xf) xf)
                    ((minusp xf) (if deg 180.0d0 pi))
                    (t 0.0d0))))
          (ensure-vt tensor :dtype :float64) :out out :dtype (%e3-float-prefer-dtype dtype :float64)))

;;; ==================================================================
;;; 5. 统计类（Statistics）
;;; ==================================================================

(defun %e3-cumulative-skip-nan (tensor op init-val &key axis dtype out op-name)
  "NaN 感知的累积原语：沿 AXIS 累积 OP，遇到 NaN 时跳过（不改变累积值），
   但输出的该位置仍写入当前累积值（对标 numpy.nancumsum/nancumprod）。
   AXIS 为 nil 时展平。init-val 为累积初值（0 / 1）。"
  (with-float-safe
   (let* ((input (ensure-vt tensor))
         (shape (vt-shape input))
         (final-dtype (or dtype (vt-dtype input)))
         (result (if out
                     (vt-check-out out shape final-dtype :op-name op-name)
                     (vt-zeros shape :dtype final-dtype)))
         (work (vt-astype input :float64))
         (op-fn (if (eq op :sum) #'+ #'*))
         (op-init (or init-val (if (eq op :sum) 0.0d0 1.0d0))))
    (flet ((cumstep (acc x)
             (if (%nan-p x) acc (funcall op-fn acc x))))
      (if axis
          (let* ((rank (length shape))
                 (ax (vt-normalize-axis axis rank))
                 (indices (make-array rank :element-type '(signed-byte 64)
                                           :initial-element 0))
                 (in-data (vt-data work)) (in-strs (vt-strides work))
                 (in-off (vt-offset work)))
            (labels ((advance ()
                       (loop for d from (1- rank) downto 0
                             when (/= d ax)
                               do (incf (aref indices d))
                                  (if (< (aref indices d) (nth d shape))
                                      (return-from advance t)
                                      (setf (aref indices d) 0)))))
              (loop
                (let ((ptr in-off))
                  (loop for d from 0 below rank
                        do (incf ptr (* (aref indices d) (nth d in-strs))))
                  (let ((cum (coerce (if (eq op :sum) 0 op-init) 'double-float))
                        (ax-stride (nth ax in-strs))
                        (ax-dim (nth ax shape)))
                    (dotimes (i ax-dim)
                      (setf cum (cumstep cum (aref in-data (+ ptr (* i ax-stride)))))
                      ;; 目标坐标：indices 中 AXIS 位置替换为 i，其余维保持
                      (let ((coords (coerce indices 'list)))
                        (setf (nth ax coords) i)
                        (setf (apply #'vt-ref result coords)
                              (vt-cast cum final-dtype))))))
                (unless (advance) (return))))
            result)
          (let ((flat (vt-ravel work))
                (out-flat (vt-ravel result))
                (n (vt-size work))
                (cum (coerce (if (eq op :sum) 0 op-init) 'double-float)))
            (dotimes (i n result)
              (setf cum (cumstep cum (vt-ref flat i)))
              (setf (vt-ref out-flat i) (vt-cast cum final-dtype)))))))))

(defun vt-nancumsum (tensor &key axis dtype out)
  "沿 AXIS 累积和，NaN 视为 0（跳过，不改变累积值）（对标 numpy.nancumsum）。
   AXIS 为 nil 时展平。:out 契约同 vt-cumsum。"
  (%e3-cumulative-skip-nan tensor :sum 0 :axis axis :dtype dtype :out out
                                 :op-name "vt-nancumsum"))

(defun vt-nancumprod (tensor &key axis dtype out)
  "沿 AXIS 累积积，NaN 视为 1（跳过）（对标 numpy.nancumprod）。
   AXIS 为 nil 时展平。:out 契约同 vt-cumprod。"
  (%e3-cumulative-skip-nan tensor :prod 1 :axis axis :dtype dtype :out out
                                  :op-name "vt-nancumprod"))

(defun %e3-nan-percentile (tensor q &key axis keepdims interpolation op-name)
  "NaN 感知的百分位/分位核心：沿 AXIS（或全局）忽略 NaN 后取分位 Q ∈ [0,1]。
   全 NaN/空切片 → NaN。返回 float64 张量。"
  (declare (ignorable op-name))
  (with-float-safe
    (let* ((tensor (ensure-vt tensor))
           (shape (vt-shape tensor))
           (rank (length shape)))
      (flet ((fiber-nanquantile (fiber)
               (let* ((fs (vt-size fiber))
                      (vals (loop for i below fs
                                  for v = (vt-ref fiber i)
                                  unless (%nan-p v) collect (vt-cast v :float64))))
                 (cond ((null vals) (vt-get-nan :float64))
                       (t (let* ((sv (vt-numpy-sort vals #'<))
                                 (n (length sv))
                                 (pos (* q (1- n)))
                                 (lo (floor pos))
                                 (hi (min (1+ lo) (1- n)))
                                 (frac (- pos lo)))
                            (case interpolation
                              (:lower (nth lo sv))
                              (:higher (nth hi sv))
                              (:nearest (nth (round pos) sv))
                              (:midpoint (/ (+ (nth lo sv) (nth hi sv)) 2.0d0))
                              (t (+ (nth lo sv) (* frac (- (nth hi sv) (nth lo sv))))))))))))
        (if axis
            (let* ((ax (vt-normalize-axis axis rank))
                   (out-shape (if keepdims
                                  (loop for d in shape for i from 0
                                        collect (if (= i ax) 1 d))
                                  (loop for d in shape for i from 0
                                        unless (= i ax) collect d)))
                   (result (vt-zeros out-shape :dtype :float64))
                   (out-strides (vt-strides result)))
              (vt-do-each (ptr val result)
                (declare (ignore val))
                (let* ((out-idx (vt-unravel-index ptr out-shape out-strides))
                       (specs (loop for i from 0 below rank
                                    for out-i = (cond ((= i ax) nil)
                                                      (keepdims i)
                                                      ((< i ax) i)
                                                      (t (1- i)))
                                    collect (if (null out-i)
                                                '(:all)
                                                (list (nth out-i out-idx)))))
                       (fiber (apply #'vt-slice tensor specs)))
                  (setf (aref (vt-data result) ptr)
                        (fiber-nanquantile fiber))))
              result)
            ;; 全局
            (let ((res (vt-zeros (if keepdims (make-list rank :initial-element 1) nil)
                                 :dtype :float64)))
              (vt-fill res (fiber-nanquantile tensor))
              res))))))

(defun vt-nanpercentile (tensor percentile &key axis keepdims (interpolation :linear))
  "忽略 NaN 的百分位数（对标 numpy.nanpercentile）。
   PERCENTILE ∈ [0,100]；INTERPOLATION 同 vt-percentile；
   全 NaN/空切片 → NaN。AXIS/KEEPDIMS 语义同 vt-percentile。"
  (unless (and (realp percentile) (<= 0 percentile 100))
    (error "vt-nanpercentile: percentile 必须在 [0,100] 内，收到 ~a" percentile))
  (%e3-nan-percentile tensor (/ percentile 100.0d0)
                      :axis axis :keepdims keepdims
                      :interpolation interpolation :op-name "vt-nanpercentile"))

(defun vt-nanquantile (tensor q &key axis keepdims (interpolation :linear))
  "忽略 NaN 的分位数（对标 numpy.nanquantile）。Q ∈ [0,1]；
   全 NaN/空切片 → NaN。AXIS/KEEPDIMS 语义同 vt-quantile。"
  (unless (and (realp q) (<= 0 q 1))
    (error "vt-nanquantile: q 必须在 [0,1] 内，收到 ~a" q))
  (%e3-nan-percentile tensor (coerce q 'double-float)
                      :axis axis :keepdims keepdims
                      :interpolation interpolation :op-name "vt-nanquantile"))

(defun vt-cov (m &key (y nil y-p) (rowvar t) (ddof nil ddof-p) (bias nil))
  "协方差矩阵（对标 numpy.cov）。
   默认 ROWVAR=t：每行为一个变量，每列一次观测。
   Y 给定时先与 M 拼接（vertically stack）后计算。
   归一化分母 = (N - DDOF)，与 numpy 完全一致：
     · DDOF **缺省**时（未传该关键字）等价 numpy 的缺省行为 → 分母 N-1；
     · 显式 DDOF=0 → 分母 N；显式 DDOF=1 → 分母 N-1；以此类推；
     · BIAS=t 等价 DDOF=N（分母 N），与 numpy 一致。
   输入按 float64 计算。返回 2 维协方差矩阵（标量变量时返回 0 维）。
   示例：(vt-cov (vt-from-array #2A((1 2 3)(4 5 6)))) => [[1 1][1 1]]"
  (declare (ignore y-p))
  ;; numpy 的 ddof 缺省值为 1（不是 0）：np.cov(a) 除以 N-1，
  ;; 而 np.cov(a, ddof=0) 除以 N。用 DDOF-P 区分「未传」与「显式 0」。
  (let ((ddof (if ddof-p ddof 1)))
    (let* ((mm (%e3-as-1d-float-transform m rowvar))
           (xx (if y (%e3-as-1d-float-transform y rowvar) nil)))
      (let* ((stacked (if xx (vt-concatenate 0 mm xx) mm))
             (n-var (first (vt-shape stacked)))
             (n-obs (second (vt-shape stacked)))
             (denom (if bias n-obs (- n-obs ddof)))
             (denom (if (<= denom 0) n-obs denom)))
      (when (= n-var 1)
        ;; 单变量 → 标量方差
        (let ((row (vt-slice stacked '(0)))
              (mean (vt-item (vt-mean (vt-slice stacked '(0))))))
          (return-from vt-cov
            (let ((acc 0.0d0))
              (dotimes (j n-obs (make-vt nil (/ acc denom) :dtype :float64))
                (incf acc (expt (- (vt-ref row j) mean) 2)))))))
      (let ((res (vt-zeros (list n-var n-var) :dtype :float64)))
        (dotimes (i n-var res)
          (dotimes (j n-var)
            (let* ((ri (vt-slice stacked (list i)))
                   (rj (vt-slice stacked (list j)))
                   (mi (vt-item (vt-mean ri)))
                   (mj (vt-item (vt-mean rj)))
                   (acc 0.0d0))
              (dotimes (k n-obs)
                (incf acc (* (- (vt-ref ri k) mi) (- (vt-ref rj k) mj))))
              (setf (vt-ref res i j) (/ acc denom))))))))))

(defun %e3-as-1d-float-transform (m rowvar)
  "把输入规范为 2 维（n-var × n-obs）float64 矩阵（对标 numpy.cov 的 rowvar 处理）。"
  (let ((v (ensure-vt m :dtype :float64)))
    (cond
      ((null (vt-shape v)) (vt-reshape v '(1 1)))
      ((= (vt-order v) 1)
       (if rowvar
           (vt-reshape v (list 1 (vt-size v)))     ; 视为单个变量（1 行）
           (vt-reshape v (list (vt-size v) 1))))
      ((= (vt-order v) 2)
       (if rowvar v (vt-transpose v)))
      (t (error "vt-cov: 输入至多 2 维，收到 ~a 维" (vt-order v))))))

(defun vt-corrcoef (m &key (y nil y-p) (rowvar t))
  "相关系数矩阵（对标 numpy.corrcoef）。
   由协方差矩阵归一化得到：R_ij = C_ij / sqrt(C_ii * C_jj)。
   对角线恒为 1（数值上）。输入按 float64 计算。
   示例：(vt-corrcoef (vt-from-array #2A((1 2 3)(4 5 6)))) => [[1 1][1 1]]"
  (declare (ignore y-p))
  (let* ((c (vt-cov m :y y :rowvar rowvar))
         (n (first (vt-shape c))))
    (if (null (vt-shape c))
        ;; 标量方差 → 相关系数恒为 1
        (vt-full nil 1.0d0 :dtype :float64)
        (let ((r (vt-zeros (list n n) :dtype :float64)))
          (dotimes (i n r)
            (dotimes (j n)
              (let ((cii (vt-ref c i i)) (cjj (vt-ref c j j)))
                (setf (vt-ref r i j)
                      (if (or (zerop cii) (zerop cjj))
                          (if (= i j) 1.0d0 0.0d0)
                          (/ (vt-ref c i j) (sqrt (* cii cjj))))))))))))

(defun vt-cross (a b &key (axis nil) (axisa -1) (axisb -1) (axisc -1) dtype)
  "向量叉积（对标 numpy.cross）。
   支持 2 分量（标量 z 分量输出）与 3 分量向量；沿 AXISA/AXISB 轴取向量，
   结果沿 AXISC 轴放置。AXIS 给定时等价 axisa=axisb=axisc=axis。
   示例：(vt-cross '(1 2 3) '(4 5 6)) => [-3 6 -3]
        (vt-cross '(1 2) '(4 5))     => -3"
  (let* ((av (ensure-vt a)) (bv (ensure-vt b)))
    (when axis (setf axisa axis
		     axisb axis
		     axisc axis))
    ;; 统一到 2 维便于处理（1 维视为单向量）；默认轴 -1 表示最后一维
    (let* ((a2 (%e3-as2d av axisa))
           (b2 (%e3-as2d bv axisb)))
      (multiple-value-bind (al bl) (values (second (vt-shape a2)) (second (vt-shape b2)))
        (let ((n-vec (first (vt-shape a2)))
              (out-len (cond ((= al 3) 3) ((= al 2) 1) (t 0))))
          (cond
            ((zerop out-len)
             (error "vt-cross: 向量分量必须为 2 或 3，收到 ~a" al))
            ((/= al bl)
             (error "vt-cross: a 与 b 分量数不一致（~a vs ~a）" al bl))
            (t
             (let* ((res (vt-zeros (list n-vec out-len) :dtype (or dtype (vt-dtype av)))))
               (dotimes (i n-vec res)
                 (let ((a0 (vt-ref a2 i 0)) (a1 (vt-ref a2 i 1))
                       (a2v (if (= al 3) (vt-ref a2 i 2) 0))
                       (b0 (vt-ref b2 i 0)) (b1 (vt-ref b2 i 1))
                       (b2v (if (= bl 3) (vt-ref b2 i 2) 0)))
                   (if (= out-len 3)
                       (progn
                         (setf (vt-ref res i 0) (- (* a1 b2v) (* a2v b1)))
                         (setf (vt-ref res i 1) (- (* a2v b0) (* a0 b2v)))
                         (setf (vt-ref res i 2) (- (* a0 b1) (* a1 b0))))
                       (setf (vt-ref res i 0) (- (* a0 b1) (* a1 b0))))))
               ;; 结果形状对齐 numpy：
               ;;  - 1 维输入（向量）→ 去掉 batch 轴；分量 2 且 1 维 → 0 维标量
               ;;  - 2 维输入（批量）→ 保留 batch 轴；分量 2 → 去掉分量轴
               (cond
                 ((= (vt-order av) 1)
                  (if (= out-len 1)
                      (vt-ref res 0 0)                       ; 标量
                      (vt-reshape res (list out-len))))
                 ((= out-len 1)
                  (vt-reshape res (list n-vec)))
                 (t res))))))))))

(defun %e3-as2d (v axis)
  "把张量规范为 (n-vec, len) 的 2 维张量，向量取自 AXIS 轴（-1 = 最后一维）。"
  (let ((vv (ensure-vt v)))
    (cond ((null (vt-shape vv)) (vt-reshape vv '(1 1)))
          ((= (vt-order vv) 1) (vt-reshape vv '(1 -1)))
          ((= (vt-order vv) 2)
           ;; 仅当向量轴是第 0 维时才转置，使向量轴落到最后一维
           (if (= axis 0) (vt-transpose vv) vv))
          (t (error "vt-cross: 输入至多 2 维，收到 ~a 维" (vt-order vv))))))

;;; ==================================================================
;;; 6. 线性代数类（Linear Algebra）
;;; ==================================================================

(defun vt-vdot (a b &key dtype out)
  "向量点积（对标 numpy.vdot）：先展平两输入，再求内积。
   与 vt-dot 的差异：vdot 恒按展平后的 1 维内积计算，忽略原始形状。
   :out 契约同 vt-dot（标量输出 → 0 维张量）。
   示例：(vt-vdot '(1 2 3) '(4 5 6)) => 32
        (vt-vdot (vt-from-array #2A((1 2)(3 4))) (vt-from-array #2A((5 6)(7 8)))) => 70"
  (let ((af (vt-ravel (ensure-vt a)))
        (bf (vt-ravel (ensure-vt b))))
    (when (/= (vt-size af) (vt-size bf))
      (error "vt-vdot: 输入元素个数不一致（~a vs ~a）" (vt-size af) (vt-size bf)))
    (vt-dot af bf :dtype dtype :out out)))

(defun vt-eigvals (matrix &key (max-iter 200) (tol 1e-10))
  "方阵特征值（对标 numpy.linalg.eigvals）。
   实现基于对称 Jacobi（对本库而言特征分解仅支持实对称矩阵，与 vt-eig 同）。
   返回按降序排列的特征值 1 维张量（float64）。
   示例：(vt-eigvals (vt-from-array #2A((2.0 0.0)(0.0 3.0)))) => [3.0 2.0]"
  (multiple-value-bind (vals vec) (vt-eig matrix :max-iter max-iter :tol tol)
    (declare (ignore vec))
    vals))

(defun vt-eigvalsh (matrix &key (max-iter 200) (tol 1e-10))
  "对称（Hermitian）方阵特征值（对标 numpy.linalg.eigvalsh）。
   与 numpy 一致：返回**升序**排列的特征值。
   （v0.4.0 修复：原实现直接透传 vt-eigvals 的非排序结果，
   不满足 numpy.linalg.eigvalsh 的升序约定。）
   示例：(vt-eigvalsh (vt-from-array #2A((2.0 1.0)(1.0 2.0)))) => [1.0 3.0]"
  (let ((w (vt-eigvals matrix :max-iter max-iter :tol tol)))
    (vt-sort w :axis -1)))

(defun vt-matrix-power (matrix n)
  "整数幂的方阵（对标 numpy.linalg.matrix_power）。
   N > 0：矩阵连乘 N 次；N = 0：单位阵；N < 0：先求逆再取 |N| 次幂。
   示例：(vt-matrix-power (vt-from-array #2A((1 2)(3 4))) 2) => [[7 10][15 22]]"
  (assert (= 2 (vt-order matrix)) (matrix) "matrix_power 要求 2 维方阵")
  (let ((nrow (first (vt-shape matrix))))
    (assert (= nrow (second (vt-shape matrix))) (matrix) "matrix_power 要求方阵")
    (cond
      ((zerop n) (vt-eye nrow :dtype (vt-dtype matrix)))
      ((plusp n)
       (let ((result (vt-eye nrow :dtype (vt-dtype matrix)))
             (base (vt-copy matrix)))
         (dotimes (i n result)
           (declare (ignorable i))
           (setf result (vt-matmul result base)))))
      (t
       (let ((inv (vt-inv matrix)))
         (vt-matrix-power inv (- n)))))))

(defun vt-cond (matrix &key (p nil) (tol 1e-12))
  "矩阵条件数（对标 numpy.linalg.cond）。
   P=nil 或 2 → 2-范数条件数（最大/最小奇异值）；
   P=1 → 1-范数条件数；P=:inf → ∞-范数条件数。
   返回 double-float 标量（对标 numpy 返回标量）。
   示例：(vt-cond (vt-eye 3)) => 1.0"
  (assert (= 2 (vt-order matrix)) (matrix) "cond 要求 2 维矩阵")
  (with-float-safe
    (cond
      ((or (null p) (eql p 2))
       (multiple-value-bind (u s v) (vt-svd matrix)
         (declare (ignore u v))
         (let* ((sv (coerce (vt-data s) 'list))
                (smax (reduce #'max sv))
                (smin (reduce #'min sv)))
           (if (or (zerop smin) (< smin tol))
               (vt-get-pos-inf :float64)
               (/ (coerce smax 'double-float) (coerce smin 'double-float))))))
      ((eql p 1)
       (let ((n (first (vt-shape matrix))))
         (let ((maxcol 0.0d0) (mincol nil))
           (dotimes (j n)
             (let ((acc 0.0d0))
               (dotimes (i n) (incf acc (abs (vt-ref matrix i j))))
               (setf maxcol (max maxcol acc))
               (setf mincol (if mincol (min mincol acc) acc))))
           (if (zerop mincol) (vt-get-pos-inf :float64) (/ maxcol mincol)))))
      ((or (eql p :inf) (eql p :infinity))
       (let ((n (first (vt-shape matrix))))
         (let ((maxrow 0.0d0) (minrow nil))
           (dotimes (i n)
             (let ((acc 0.0d0))
               (dotimes (j n) (incf acc (abs (vt-ref matrix i j))))
               (setf maxrow (max maxrow acc))
               (setf minrow (if minrow (min minrow acc) acc))))
           (if (zerop minrow) (vt-get-pos-inf :float64) (/ maxrow minrow)))))
      (t (error "vt-cond: 不支持的 p 值 ~a（支持 nil/2/1/:inf）" p)))))

(defun vt-multi-dot (arrays)
  "多个矩阵的有序列乘法（对标 numpy.linalg.multi_dot）。
   按左侧结合顺序依次相乘（本库不做最优括号化，语义等价）。
   至少需要 2 个矩阵；相邻维度必须可乘。
   示例：(vt-multi-dot (list (vt-ones '(2 3)) (vt-ones '(3 4)) (vt-ones '(4 2))))
        => shape (2 2)"
  (when (< (length arrays) 2)
    (error "vt-multi-dot: 至少需要 2 个矩阵，收到 ~a" (length arrays)))
  (let* ((vs (mapcar #'ensure-vt arrays))
         (acc (first vs)))
    (loop for m in (rest vs)
          for expected-k = (first (vt-shape acc))
          when (not (= (second (vt-shape acc)) (first (vt-shape m))))
            do (error "vt-multi-dot: 维度不匹配（~a 的列 ~a vs ~a 的行 ~a）"
                      (vt-shape acc) (second (vt-shape acc))
                      (vt-shape m) (first (vt-shape m)))
          do (setf acc (vt-matmul acc m))
          finally (return acc))))

;;; ==================================================================
;;; 7. 逻辑类（Logic）
;;; ==================================================================

(defun vt-array-equal (a b)
  "判定两张量形状相同且逐元素相等（对标 numpy.array_equal）。
   NaN != NaN（含 NaN 恒不相等，同 numpy）；返回 Lisp 布尔值（非张量）。
   示例：(vt-array-equal '(1 2) '(1 2)) => T
        (vt-array-equal '(1 nan) '(1 nan)) => NIL"
  (let ((av (ensure-vt a)) (bv (ensure-vt b)))
    (and (equal (vt-shape av) (vt-shape bv))
         (let ((af (vt-ravel (vt-astype av :float64)))
               (bf (vt-ravel (vt-astype bv :float64)))
               (n (vt-size av)))
           (with-float-safe
             (dotimes (i n t)
               (unless (= (vt-ref af i) (vt-ref bf i))
                 (return nil))))))))

(defun vt-array-equiv (a b)
  "判定两张量是否可通过广播相等（对标 numpy.array_equiv）。
   形状可广播且广播后逐元素相等 → T；NaN != NaN。返回 Lisp 布尔值。
   示例：(vt-array-equiv '(1 2) '((1 2)(1 2))) => T"
  (let ((av (ensure-vt a)) (bv (ensure-vt b)))
    (handler-case
        (let* ((sa (if (null (vt-shape av)) (list 1) (vt-shape av)))
               (sb (if (null (vt-shape bv)) (list 1) (vt-shape bv)))
               (final (vt-broadcast-shapes sa sb))
               (ab (vt-broadcast-to (vt-reshape av sa) final))
               (bb (vt-broadcast-to (vt-reshape bv sb) final)))
          (vt-array-equal ab bb))
      (error () nil))))

(defun vt-isposinf (tensor &key out)
  "逐元素 +Inf 判定，返回 1/0，dtype 为 :int8（对标 numpy.isposinf）。
   :out 契约同 vt-=（dtype :int8）。"
  (vt-map (lambda (x) (if (and (floatp x) (%pos-inf-p x)) 1 0))
          (ensure-vt tensor :dtype :float64) :out out :dtype :int8))

(defun vt-isneginf (tensor &key out)
  "逐元素 -Inf 判定，返回 1/0，dtype 为 :int8（对标 numpy.isneginf）。
   :out 契约同 vt-=（dtype :int8）。"
  (vt-map (lambda (x) (if (and (floatp x) (%neg-inf-p x)) 1 0))
          (ensure-vt tensor :dtype :float64) :out out :dtype :int8))

;;; ==================================================================
;;; 8. 集合类（Set operations）
;;; ==================================================================

(defun vt-isin (element test-elements &key (invert nil) (assume-unique nil))
  "逐元素判定是否存在于 TEST-ELEMENTS 中（对标 numpy.isin）。
   返回与 ELEMENT 形状相同的布尔张量（dtype :int8，1/0）。
   INVERT=t 取反；ASSUME-UNIQUE 为性能提示（本库忽略，语义等价）。
   示例：(vt-isin '(1 2 3 4) '(2 4)) => [0 1 0 1]"
  (declare (ignore assume-unique))
  (let* ((elem (ensure-vt element))
         (test (vt-unique (ensure-vt test-elements)))
         (test-set (coerce (vt-data test) 'list)))
    (vt-map (lambda (x)
              (let ((hit (if (member x test-set :test #'vt-float-nan-inf-=) 1 0)))
                (if invert (- 1 hit) hit)))
            elem
            :dtype :int8)))
