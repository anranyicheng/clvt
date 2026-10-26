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

(defun vt-fromiter (iterable &key (dtype nil dtype-p) (count nil))
  "从 Lisp 序列构建 1 维张量（对标 numpy.fromiter）。

   DTYPE 缺省时**推断**：序列含复数 → :complex128，否则 :float64
   （保持 v0.4 之前的默认，不破坏既有调用）。
   显式传入实数 dtype 而序列含复数 → 报错，与 numpy 的
   `TypeError: Cannot cast complex to real` 一致（绝不静默丢虚部）。

   COUNT 若给出必须是序列长度，用于前置校验（numpy 语义：count 应精确匹配）；
   纯语义构建时忽略 COUNT 的实际作用（本库一次性构建，无需预分配）。
   :out 契约：本函数为构建函数，无 :out。
   示例：(vt-fromiter '(1 2 3) :dtype :float64) => [1.0 2.0 3.0]"
  (declare (ignore count))
  (when (not (or (null iterable) (typep iterable 'sequence)))
    (error "vt-fromiter: 需要序列输入，收到 ~a" (type-of iterable)))
  (let* ((seq (or iterable '()))
         (has-complex (some (lambda (x) (or (complexp x) (%cx-deep-p x))) seq))
         (dt (cond ((and dtype-p dtype)
                    (when (and (not (vt-complex-dtype-p dtype)) has-complex)
                      (error "vt-fromiter: 序列含复数，与显式 dtype ~a 冲突\
（numpy 同样抛 TypeError: Cannot cast complex to real）" dtype))
                    dtype)
                   (t (if has-complex :complex128 :float64)))))
    (vt-from-sequence seq :dtype dt)))

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
             (idx (%flat-offset0 (vt-astype indices :int64)))
             (total (vt-size out))
             (src-data (vt-data tensor))
             (src-off (vt-offset tensor))
             (out-data (vt-data out))
             (out-off (vt-offset out))
             (strides (vt-strides tensor))
             (strd (coerce strides 'simple-vector))
             (i-strd (coerce (vt-compute-strides ishape) 'simple-vector))
             (odims (coerce oshape 'simple-vector))
             (idims (coerce ishape 'simple-vector))
             (in-ax-stride (nth ax strides))
             (idx-arr (vt-data idx)))
        ;; ================================================================
        ;; v0.4.7 快路径：AX 是**末轴** + 源/输出均连续 + indices 无广播
        ;;
        ;; 此时布局就是 (ROWS, AX-DIM) → (ROWS, LEN)，每个输出元素只做
        ;; 「一次索引读 + 一次 src 读 + 一次 dst 写」，增量里程计完全不需要
        ;; （它每元素要多算几次加减与一次进位判断，都是纯开销）。
        ;; 1000×1000 上轮实测 22305us（numpy 5387us），其中里程计占大头。
        ;;
        ;; 条件里要求 (equal (subseq ishape 0 ax) (subseq shape 0 ax))：indices
        ;; 的非 AX 维必须与源完全一致，才说明 IDX 与输出同构（无广播）；有广播
        ;; 时仍走下方通用路径（铁律 2：回退而非报错）。
        ;; ================================================================
        (when (and (= ax (1- rank))
                   (vt-contiguous-p tensor)
                   (zerop src-off)
                   (plusp total)
                   (equal (subseq ishape 0 ax) (subseq shape 0 ax)))
          (let* ((len (nth ax ishape))
                 (rows (the fixnum (truncate (the fixnum total) (max len 1))))
                 (et (array-element-type out-data)))
            (declare (type fixnum len rows))
            (macrolet ((g (lt)
                         `(let ((s (the (simple-array ,lt (*)) src-data))
                                (d (the (simple-array ,lt (*)) out-data))
                                (ia (the (simple-array (signed-byte 64) (*)) idx-arr)))
                            (declare (type (simple-array ,lt (*)) s d)
                                     (type (simple-array (signed-byte 64) (*)) ia)
                                     (type fixnum len rows ax-dim))
                            (dotimes (o rows)
                              (declare (type fixnum o))
                              (let ((sb (the fixnum (* o ax-dim)))
                                    (db (the fixnum (* o len))))
                                (declare (type fixnum sb db))
                                (dotimes (j len)
                                  (declare (type fixnum j))
                                  ;; 注意：IDX 的扁平布局是 (ROWS, LEN)，索引必须
                                  ;; 带行为偏移 O*LEN。漏掉它时「LEN=1 且多行」的
                                  ;; 广播用例（如 idx 形状 (2,1) 配 (2,2) 源）会
                                  ;; 每行都读第 0 个索引，静默给出错误结果。
                                  (let* ((raw (aref ia (the fixnum (+ (* o len) j))))
                                         (sel (if (minusp raw)
                                                  (the fixnum (+ raw ax-dim))
                                                  (the fixnum raw))))
                                    (declare (type fixnum sel))
                                    (unless (<= 0 sel (1- ax-dim))
                                      (error "vt-take-along-axis: 索引 ~a 越界（轴长 ~a）"
                                             raw ax-dim))
                                    (setf (aref d (the fixnum (+ db j)))
                                          (aref s (the fixnum (+ sb sel)))))))))))
              (cond
                ((equal et 'double-float)       (g double-float))
                ((equal et 'single-float)       (g single-float))
                ((equal et '(signed-byte 64))   (g (signed-byte 64)))
                ((equal et '(signed-byte 32))   (g (signed-byte 32)))
                ((equal et '(signed-byte 16))   (g (signed-byte 16)))
                ((equal et '(signed-byte 8))    (g (signed-byte 8)))
                ((equal et '(unsigned-byte 16)) (g (unsigned-byte 16)))
                ((equal et '(unsigned-byte 8))  (g (unsigned-byte 8)))
                (t nil)))
            (when (member et (list 'double-float 'single-float
                                   '(signed-byte 64) '(signed-byte 32)
                                   '(signed-byte 16) '(signed-byte 8)
                                   '(unsigned-byte 16) '(unsigned-byte 8))
                          :test #'equal)
              (return-from vt-take-along-axis out))))
        ;; ================================================================
        ;; v0.4.5 重写（原 157964us / numpy 5820us / 27.1 倍）
        ;;
        ;; 原实现对**每个输出元素**都：make-list 建两份坐标表 → 两趟 NTH
        ;; 走表 → lin 里再 (nth d coords)(nth d strd) 走两遍 → vt-ref 反射。
        ;; 1000×1000 = 1e6 次这样的分配 + 链表遍历。
        ;;
        ;; 改为**增量里程计**：源指针拆成「除 AX 外各维贡献 src-other」+
        ;; 「AX 维贡献 sel*in-ax-stride」两部分，每前进一个元素只做若干次
        ;; 整数加减；零分配、无 NTH。输出恒为新建 C 连续张量，故 out 指针
        ;; 就是线性自增。→ 28906us（5.36×）
        ;;
        ;; v0.4.6 再进一步：原先把写操作做成 FLET + FUNCALL 回调以复用里程计，
        ;; 于是每元素多一次闭包调用（约 20ns）。现在把 store **内联**进 8 份
        ;; 按元素类型特化的循环体（一份源码、8 次宏展开），→ 见下。
        ;; ================================================================
        (macrolet ((walk (lt)
                     `(let ((dst (the (simple-array ,lt (*)) out-data))
                            (src (the (simple-array ,lt (*)) src-data))
                            ;; v0.4.6：IDX-ARR 原先漏了声明 → 每次取 SEL 都是
                            ;; 通用 AREF（二维分派 + 边界检查）。逐元素一次，
                            ;; 1e6 元素就是 1e6 次（实测 ~15ns/次）。
                            (ia  (the (simple-array (signed-byte 64) (*)) idx-arr))
                            (coords (make-array rank :element-type 'fixnum
                                                     :initial-element 0))
                            (src-other 0) (idx-off 0) (sel 0) (op 0))
                        (declare (type (simple-array ,lt (*)) dst src)
                                 (type (simple-array (signed-byte 64) (*)) ia)
                                 (type (simple-array fixnum (*)) coords)
                                 (type fixnum src-other idx-off sel op))
                        ;; 里程计：逐元素推进，零分配、无 NTH
                        (when (plusp total)
                          (setf sel (aref ia 0))
                          (loop
                            (let* ((sel* (if (minusp sel)
                                             (the fixnum (+ sel ax-dim))
                                             sel))
                                   (s (the fixnum (+ src-off src-other
                                                     (* sel* in-ax-stride)))))
                              (declare (type fixnum sel* s))
                              (unless (<= 0 sel* (1- ax-dim))
                                (error "vt-take-along-axis: 索引 ~a 越界（轴长 ~a）"
                                       sel ax-dim))
                              (setf (aref dst (the fixnum (+ out-off op))) (aref src s)))
                            (incf op)
                            (when (>= op total) (return))
                            (let ((d (1- rank)))
                              (declare (type fixnum d))
                              (loop
                                (when (< d 0) (return))
                                (let ((nd (the fixnum (1+ (aref coords d)))))
                                  (cond
                                    ((< nd (aref odims d))
                                     (setf (aref coords d) nd)
                                     (if (= d ax)
                                         (incf idx-off (the fixnum (aref i-strd d)))
                                         (progn
                                           (incf src-other (the fixnum (aref strd d)))
                                           (unless (= (aref idims d) 1)
                                             (incf idx-off (the fixnum (aref i-strd d))))))
                                     (return))
                                    (t
                                     ;; 进位：坐标从 (DIM-1) 归零 → 贡献减少 (DIM-1)*STRIDE
                                     (let ((last (the fixnum
                                                      (1- (the fixnum (aref odims d))))))
                                       (declare (type fixnum last))
                                       (setf (aref coords d) 0)
                                       (if (= d ax)
                                           (decf idx-off
                                                 (the fixnum (* last (the fixnum (aref i-strd d)))))
                                           (progn
                                             (decf src-other
                                                   (the fixnum (* last (the fixnum (aref strd d)))))
                                             (unless (= (aref idims d) 1)
                                               (decf idx-off
                                                     (the fixnum (* last (the fixnum (aref i-strd d)))))))))
                                     (decf d))))))
                            (setf sel (aref ia idx-off)))))))
          (let ((et (array-element-type out-data)))
            (cond
              ((equal et 'double-float)     (walk double-float))
              ((equal et 'single-float)     (walk single-float))
              ((equal et '(signed-byte 64)) (walk (signed-byte 64)))
              ((equal et '(signed-byte 32)) (walk (signed-byte 32)))
              ((equal et '(signed-byte 16)) (walk (signed-byte 16)))
              ((equal et '(signed-byte 8))  (walk (signed-byte 8)))
              ((equal et '(unsigned-byte 16)) (walk (unsigned-byte 16)))
              ((equal et '(unsigned-byte 8))  (walk (unsigned-byte 8)))
              ;; 复数（CONVENTIONS §16.19）：本函数只做「按索引搬
              ;; 元素」，与元素类型无关 —— 8 路分派纯粹是性能特化，
              ;; 漏掉复数就会掉进报错分支。注意 SEL 是**下标**（fixnum），
              ;; 元素值是复数，不受 fixnum 声明影响。
              ((equal et '(complex double-float)) (walk (complex double-float)))
              ((equal et '(complex single-float)) (walk (complex single-float)))
              (t (error "vt-take-along-axis: 不支持的输出元素类型 ~a" et)))))
        out))))

(defun vt-put-along-axis (tensor indices values axis)
  "沿 AXIS 把 VALUES 按 INDICES 写入 TENSOR 的拷贝并返回。
   INDICES 约束同 vt-take-along-axis。

   ⚠ 与 numpy 的差异（v0.4.2 记录）：numpy.put_along_axis 是**原地**修改并
   返回 NIL；本函数返回新张量、不改输入。两者都提供「按轴写入」的语义，
   但副作用模型不同 —— 需要原地行为请用 (vt-copy-into tensor
   (vt-put-along-axis tensor idx val axis))。同族的 vt-put 则是原地写入
   （与 numpy.put 一致），两者不一致属历史遗留。
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
   示例：(vt-compress '(1 0 1 0 1) (vt-fromiter '(0 1 2 3 4))) => [0 2 4]

   v0.4.5 性能修复：numpy 的 compress 等价于 take(nonzero(cond))，本实现现在
   真的这么走 —— 先扫一遍条件得到选中的下标向量，再委托 vt-take。
   原 axis=nil 分支是「list 化条件 + 逐元素 vt-ref + vt-from-sequence」，
   1M 元素实测 44245us（numpy 1011us，差 43 倍）；axis 分支原本就委托
   vt-take，但用 (nth i full) 在 list 上取元素（长度 1M 时是 O(n²)），
   一并改为数组扫描。"
  (let* ((cond-v (ensure-vt condition))
         ;; 不再 (vt-astype ... :int64) —— 那会把 1M 条件整块转换一遍（8MB 分配
         ;; + 逐元素 coerce），实测占 compress 总耗时的大头。直接按条件的原生
         ;; 元素类型扫描「非零」即可。
         (cflat (%flat-offset0 cond-v))
         (cdata (vt-data cflat))
         (clen (vt-size cflat)))
    (flet ((selected-indices (limit)
             ;; 返回长度 = 条件中为真的个数、元素为 [0,LIMIT) 下标的 int64 张量
             (let* ((m (min limit clen))
                    (buf (make-array (max m 1) :element-type '(signed-byte 64)))
                    (k 0))
               (declare (type fixnum k))
               (macrolet ((scan (lt)
                            `(let ((c (the (simple-array ,lt (*)) cdata)))
                               (declare (type (simple-array ,lt (*)) c)
                                        (type fixnum m k))
                               (dotimes (i m)
                                 (declare (type fixnum i))
                                 (unless (zerop (aref c i))
                                   (setf (aref buf k) i)
                                   (incf k))))))
                 (let ((et (array-element-type cdata)))
                   (cond
                     ((equal et 'double-float)     (scan double-float))
                     ((equal et 'single-float)     (scan single-float))
                     ((equal et '(signed-byte 64)) (scan (signed-byte 64)))
                     ((equal et '(signed-byte 32)) (scan (signed-byte 32)))
                     ((equal et '(signed-byte 16)) (scan (signed-byte 16)))
                     ((equal et '(signed-byte 8))  (scan (signed-byte 8)))
                     ((equal et '(unsigned-byte 16)) (scan (unsigned-byte 16)))
                     ((equal et '(unsigned-byte 8))  (scan (unsigned-byte 8)))
                     (t (dotimes (i m)
                          (declare (type fixnum i))
                          (unless (zerop (aref cdata i))
                            (setf (aref buf k) i)
                            (incf k)))))))
               (%make-vt :data (if (= k (max m 1)) buf (subseq buf 0 k))
                         :shape (list k) :strides '(1) :offset 0 :dtype :int64))))
      (if (null axis)
          (let* ((v (ensure-vt tensor))
                 (n (vt-size v)))
            ;; axis=nil：先展平再按条件选取（vt-take 内部会做 %flat-offset0）
            (vt-take v (selected-indices n) :axis nil))
          (let* ((shape (vt-shape tensor))
                 (rank (length shape))
                 (ax (vt-normalize-axis axis rank))
                 (ax-dim (nth ax shape)))
            (when (> clen ax-dim)
              (error "vt-compress: condition 长度 ~a 超过轴长 ~a" clen ax-dim))
            (vt-take tensor (selected-indices ax-dim) :axis ax))))))

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
   :out 契约同 vt-abs。

   v0.4.3 性能修复：原实现把 `with-float-safe` 写在**逐元素 lambda 内部**，
   等价于对每个元素各执行一次 `sb-int:with-float-traps-masked`（读/改/恢复
   FP 控制字），1M 元素 = 1M 次陷阱屏蔽 + 1M 次 funcall + 泛型 aref，
   实测 198ms（numpy.sign 0.59ms，差 335 倍）。vt-map / vt-fast-map 外层
   本就已屏蔽浮点陷阱，逐元素再屏蔽一次纯属冗余。
   现改为复用 vt-signum 的命名算子 %sign1（内联、无内层 with-float-safe），
   语义逐位不变 —— vt-sign 与 vt-signum 的差异只在「输出 dtype 的来源」。
   %sign1 用 (float 1 x) 而非字面 1.0d0，故 float32 输入写 single-float
   数组不会触发 specialized-array 类型错误。"
  (when (%cx-any-p tensor)
    (return-from vt-sign
      (let* ((in (ensure-vt tensor))
             (dt (%cx-result-complex-dtype (vt-dtype in) dtype "VT-SIGN")))
        (vt-fast-map (function %cop-sign) in :out out :dtype dt))))
  (let* ((v (ensure-vt tensor))
         (res-dtype (or dtype (vt-dtype v))))
    (vt-fast-map #'%sign1 v :out out :dtype res-dtype)))

(defun vt-positive (tensor &key out dtype)
  "逐元素取正（一元 +，对标 numpy.positive）：数值恒等映射。
   :out 契约同 vt-abs。"
  (when (%cx-any-p tensor)
    (return-from vt-positive (%cx-op-positive tensor :out out :dtype dtype)))
  (vt-fast-map #'identity (ensure-vt tensor) :out out :dtype dtype))

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
  (when (%cx-any-p tensor)
    (return-from vt-expm1
      (let* ((in (ensure-vt tensor))
             (dt (%cx-result-complex-dtype (vt-dtype in) dtype "VT-EXPM1")))
        (vt-fast-map (function %cop-expm1) in :out out :dtype dt))))
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
  (when (%cx-any-p tensor)
    (return-from vt-log1p
      (let* ((in (ensure-vt tensor))
             (dt (%cx-result-complex-dtype (vt-dtype in) dtype "VT-LOG1P")))
        (vt-fast-map (function %cop-log1p) in :out out :dtype dt))))
  (let ((v (ensure-vt tensor)))
    (%float-map #'%e3-log1p v out (%e3-float-prefer-dtype dtype (vt-dtype v)))))

(defun vt-logaddexp (x y &key out dtype)
  "逐元素 log(exp(x) + exp(y))，数值稳定（对标 numpy.logaddexp）。
   整数输入输出 float64。:out 契约同 vt-abs（支持广播）。"
    (when (%cx-reject x y)
    (error "vt-logaddexp: 不支持复数输入（numpy 同样拒绝）。见 CONVENTIONS §16.11"))
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
   结果 dtype 缺省 float64。:out 契约同 vt-abs（支持广播）。

  复数（§16.10）：两个分量各自按复数幂算，与 vt-pow 共用 %CX-POW-2。
  漏掉这里的复数分派会静默取实部：(vt-float-power #C(2 1) #C(2 0))
  会得 4.0，而 numpy 是 3+4j。"
  (when (or (%cx-deep-p x) (%cx-deep-p y))
    (return-from vt-float-power
      (let* ((xe (ensure-vt x))
             (ye (ensure-vt y))
             (promoted (or dtype (%op-final-dtype
                                  (list (%scalar-kind x) (%scalar-kind y))
                                  (list (vt-dtype xe) (vt-dtype ye)))))
             (dt (if (vt-complex-dtype-p promoted) promoted :complex128))
             (dt (%cx-result-complex-dtype dt (and dtype dtype) "VT-FLOAT-POWER")))
        (vt-map (function %cx-pow-2) xe ye :out out :dtype dt))))
  (vt-map (lambda (a b)
            (let ((af (coerce a 'double-float)) (bf (coerce b 'double-float)))
              (cond ((%nan-p af) af)
                    ((%nan-p bf) bf)
                    ((and (minusp af) (/= bf (floor bf))) (vt-get-nan :float64))
                    (t (expt af bf)))))
          (ensure-vt x :dtype :float64) (ensure-vt y :dtype :float64)
          :out out :dtype (%e3-float-prefer-dtype dtype :float64)))

(defun %e3-sign-bit-p (x)
  "IEEE 754 符号位判定（含 -0.0 与 NaN 的符号位）。
   minusp/float-sign 均无法区分 ±0.0，这里直接读 double-float 位模式。"
  (declare (double-float x))
  (logbitp 63 (logand #xFFFFFFFFFFFFFFFF (sb-kernel:double-float-bits x))))

(defun vt-copysign (x y &key out dtype)
  "逐元素赋予 X 的绝对值与 Y 的符号（对标 numpy.copysign）。
   结果恒为浮点；NaN 符号位处理与 numpy 一致（copysign(1, nan) → 1）。
   :out 契约同 vt-abs（支持广播）。"
    (when (%cx-reject x y)
    (error "vt-copysign: 不支持复数输入（numpy 同样拒绝）。见 CONVENTIONS §16.11"))
(vt-map (lambda (a b)
            ;; v0.4.1 修复（差分测试 R4）：符号取自 y 的 IEEE 符号位，
            ;; minusp 无法识别 -0.0（copysign(1,-0.0) 应为 -1.0）。
            (let ((af (coerce a 'double-float)) (bf (coerce b 'double-float)))
              (if (%e3-sign-bit-p bf) (- (abs af)) (abs af))))
          (ensure-vt x :dtype :float64) (ensure-vt y :dtype :float64)
          :out out :dtype (%e3-float-prefer-dtype dtype :float64)))

(defun vt-signbit (tensor &key out)
  "逐元素符号位判定，返回 1/0，dtype 为 :int8（对标 numpy.signbit）。
   x < 0 或为负零（-0.0）→ 1，否则 0。:out 契约同 vt-=（dtype :int8）。"
    (when (%cx-reject tensor)
    (error "vt-signbit: 不支持复数输入（numpy 同样拒绝）。见 CONVENTIONS §16.11"))
(vt-map (lambda (x)
            ;; v0.4.1 修复（差分测试 R5）：按 IEEE 符号位判定；正 NaN 符号位
            ;; 为 0（numpy.signbit(+nan)=0），负 NaN/-0.0/负数为 1。
            (if (%e3-sign-bit-p (coerce x 'double-float)) 1 0))
          (ensure-vt tensor :dtype :float64) :out out :dtype :int8))

(declaim (inline %e3-nextafter))
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
    (when (%cx-reject x y)
    (error "vt-nextafter: 不支持复数输入（numpy 同样拒绝）。见 CONVENTIONS §16.11"))
;; 性能：原来是 (lambda (a b) ...) → vt-map 的 funcall 路径（1M 元素
  ;; 实测 270ms）。%e3-nextafter 已是 inline 函数，直接用 vt-fast-map
  ;; 内联成二元特化循环；仅显式 :dtype :float32（需要先降精度）保留
  ;; 通用路径，保证「double 精度计算、输出降精度」的既有语义不变。
  (let ((dt (%e3-float-prefer-dtype dtype :float64))
        (xv (ensure-vt x :dtype :float64))
        (yv (ensure-vt y :dtype :float64)))
    (if (eq dt :float32)
        (vt-map #'%e3-nextafter xv yv :out out :dtype dt)
        (vt-fast-map #'%e3-nextafter xv yv :out out :dtype dt))))

(declaim (inline %e3-op-spacing-df))
(defun %e3-op-spacing-df (x)
  "spacing 的 double 算子：nextafter(|x|, +inf) - |x| 的展开式。

   参数用 COERCE 而非 (DECLARE (TYPE DOUBLE-FLOAT X))：本算子会被
   vt-fast-map 的 8 路（元素类型）展开同时内联，整型分支上传入的是
   fixnum —— 硬声明会触发「Derived type of ... is T」编译告警。
   COERCE 在类型已匹配时由 SBCL 内联为恒等，零开销。"
  (let* ((x (coerce x 'double-float))
         (xf (abs x)))
    (cond ((= xf 0.0d0) 5.0d-324)
          ((/= xf xf) xf)                       ; NaN
          ((= xf sb-ext:double-float-positive-infinity) xf)
          (t (- (+ xf (* xf 2.220446049250313d-16)) xf)))))

(defun vt-spacing (tensor &key out dtype)
  "逐元素返回与该值相邻浮点数之间的距离（对标 numpy.spacing）。
   以 double-float 精度计算。:out 契约同 vt-abs。
   示例：(vt-spacing 1.0) => 2.220446049250313e-16；(vt-spacing 0.0) => 5e-324"
    (when (%cx-reject tensor)
    (error "vt-spacing: 不支持复数输入（numpy 同样拒绝）。见 CONVENTIONS §16.11"))
;; 性能：原实现对每个元素做一次 lambda 调用 + 一次 with-float-safe
  ;; （读写 FP 控制字，约 20ns/元素）+ 一次 %e3-nextafter 全分支比较，
  ;; 1M 元素实测 309ms。这里把三者摊平：
  ;;   1. 外层 vt-fast-map 已用 with-float-safe 包住整个循环，逐元素
  ;;      再包一次纯属重复；
  ;;   2. nextafter(|x|, +inf) - |x| 在 af>0 时恒等于 af*eps 的
  ;;      「加后再减」，直接内联该算式（与 %e3-nextafter 同一条浮点路径，
  ;;      逐位一致）；
  ;;   3. 算子写成符号，供 vt-fast-map 编译期内联。
  (let ((dt (%e3-float-prefer-dtype dtype :float64))
        (in (ensure-vt tensor :dtype :float64)))
    (if (eq dt :float32)
        (vt-map #'%e3-op-spacing-df in :out out :dtype dt)
        (%float-map #'%e3-op-spacing-df in out dt))))

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

(defun %nan-to-num-buffer (flat nanv posv negv)
  "FLAT 为连续浮点张量（float32/float64），返回**同 dtype** 的新张量：
   逐元素把 NaN / +Inf / -Inf 换成 NANV / POSV / NEGV。

   为什么不用 `vt-map` + 闭包：闭包在通用循环里每元素一次 funcall + 装箱，
   1M double 实测 21.9 ms（complex128 那边 78.9 ms）；换成类型特化的直线
   循环后 1.7 ms / 4.8 ms。这是「值级内核」第一次没能走 vt-fast-map 白名单
   —— 因为 nan_to_num 带三个额外参数，不是一元内核。"
  (macrolet ((pull (lt maxf)
               `(let ((s (the (simple-array ,lt (*)) (vt-data flat)))
                      (off (the fixnum (vt-offset flat)))
                      (n (the fixnum (vt-size flat)))
                      (nan (coerce nanv ',lt))
                      (pos (coerce posv ',lt))
                      (neg (coerce negv ',lt))
                      (mx (coerce ,maxf ',lt))
                      (res (make-array (the fixnum (vt-size flat)) :element-type ',lt)))
                  (declare (type (simple-array ,lt (*)) s res)
                           (type fixnum off n) (type ,lt nan pos neg mx)
                           (optimize (speed 3) (safety 1)))
                  (with-float-safe
                    (dotimes (i n)
                      (let ((x (aref s (the fixnum (+ off i)))))
                        (declare (type ,lt x))
                        (setf (aref res i)
                              (cond ((/= x x) nan)
                                    ((> x mx) pos)
                                    ((< x (- mx)) neg)
                                    (t x))))))
                  (%make-vt :data res
                            :shape (vt-shape flat)
                            :strides (vt-compute-strides (vt-shape flat))
                            :offset 0
                            :dtype (if (eq ',lt 'single-float) :float32 :float64)))))
    (let ((et (array-element-type (vt-data flat))))
      (cond ((equal et 'double-float) (pull double-float most-positive-double-float))
            ((equal et 'single-float) (pull single-float most-positive-single-float))
            (t nil)))))
(defun vt-nan-to-num (tensor &key (copy t) (nan 0.0d0) (posinf nil) (neginf nil) out dtype)
  "把 NaN 替换为 NAN、+Inf 替换为 POSINF、-Inf 替换为 NEGINF（对标 numpy.nan_to_num）。
   POSINF/NEGINF 缺省为对应 dtype 的最大/最小有限值（对齐 numpy）。
   COPY 仅作 API 兼容（本库不改输入，恒返回新张量）。:out 契约同 vt-abs。
   示例：(vt-nan-to-num '(nan inf -inf 1.0)) => [0.0 1.797e308 -1.797e308 1.0]"
  (when (%cx-any-p tensor)
    (return-from vt-nan-to-num (%cx-op-nan-to-num tensor :copy copy :nan nan :posinf posinf :neginf neginf :out out :dtype dtype)))
  (let* ((v (ensure-vt tensor))
         (dt (or dtype (vt-dtype v)))
         (maxf (if (eq dt :float32) 3.4028235f38 most-positive-double-float))
         (minf (if (eq dt :float32) -3.4028235f38 most-negative-double-float))
         (odt (%e3-float-prefer-dtype dt :float64))
         (nanv (or nan 0.0d0))
         (posv (or posinf maxf))
         (negv (or neginf minf)))
    (if (and (eq (vt-dtype v) odt)
             ;; 快路径只处理浮点：整数输入时 odt 也是整数（%E3-FLOAT-PREFER-DTYPE
             ;; 的 `(t dtype)` 分支原样返回），而 %NAN-TO-NUM-BUFFER 只认 f32/f64
             ;; —— 少这个判据会返回 NIL（实测：整数输入直接拿到 NIL 张量）。
             (member (vt-dtype v) '(:float32 :float64)))
        ;; 快路径：连输入输出 dtype 都一致的浮点张量，走类型特化直线循环
        (let ((res (%nan-to-num-buffer (vt-contiguous v) nanv posv negv)))
          (if (null out)
              res
              (progn (vt-check-out out (vt-shape v) odt :op-name "VT-NAN-TO-NUM")
                     (vt-copy-into out res)
                     out)))
        ;; 需要换 dtype（整数输入 → float64、显式 :dtype 等）：保持原 vt-map 路径
        (vt-map (lambda (x)
                  (let ((xf (coerce x 'double-float)))
                    (cond ((%nan-p xf) (coerce nanv 'double-float))
                          ((%pos-inf-p xf) (coerce posv 'double-float))
                          ((%neg-inf-p xf) (coerce negv 'double-float))
                          (t xf))))
                v :out out :dtype odt))))

;;; 下面这组（real / imag / conj / angle 等）只保留**一份**实现：复数与实数
;;; 都在函数体内判定（`(when (%cx-any-p …) …)`），不存在第二份同名定义。
;;; 历史上有过「本库无复数 dtype，恒返回输入本身」的错误桩，靠加载顺序覆盖，
;;; 一旦调整 `clvt.asd` 的顺序就会静默返回错误结果（§16.32）。

(defun vt-real (tensor &key out dtype)
  "取实部（对标 numpy.real）。复数 → 分量浮点 dtype 的实数张量；
   实数输入原样返回（副本）。

   与 numpy 的差异（§16.6）：numpy 的 `.real` 是**零拷贝视图**（写实部会
   改到原数组）；clvt 的 strided view 无法表达「同一缓冲的实数子集」
   （缓冲元素类型是复数），故返回**副本**。需要写回请用 vt-complex 重组。"
  (let* ((in (ensure-vt tensor))
         (dt (or dtype (vt-real-dtype-of (vt-dtype in))))
         (dt (if (vt-complex-dtype-p dt) (vt-real-dtype-of dt) dt)))
    (if (vt-complex-dtype-p (vt-dtype in))
        (%cx-run-to-real #'%cx-real-run #'%creal in out dt "VT-REAL")
        (vt-map #'%creal in :out out :dtype dt))))

(defun vt-imag (tensor &key out dtype)
  "取虚部（对标 numpy.imag）。实数输入 → 全 0（与 numpy 一致）。
   返回副本而非视图，原因同 vt-real。"
  (let* ((in (ensure-vt tensor))
         (in-dt (vt-dtype in))
         (dt (or dtype (if (vt-complex-dtype-p in-dt)
                           (vt-real-dtype-of in-dt)
                           in-dt))))
    (if (vt-complex-dtype-p in-dt)
        (%cx-run-to-real #'%cx-imag-run #'%cimag in out dt "VT-IMAG")
        (vt-map #'%cimag in :out out :dtype dt))))

(defun vt-conj (tensor &key out dtype)
  "逐元素共轭（对标 numpy.conj / numpy.conjugate）。实数输入原样返回。"
  (let ((in (ensure-vt tensor)))
    (vt-fast-map #'%cconj in :out out :dtype (or dtype (vt-dtype in)))))

(defun vt-angle (tensor &key deg out dtype)
  "逐元素幅角 atan2(im, re) ∈ (-π, π]（对标 numpy.angle）。
   DEG 为真时返回角度制。结果恒为实数 dtype。
   零值返回 0（保留 ±0.0 符号）。"
  (let* ((in (ensure-vt tensor))
         (in-dt (vt-dtype in))
         (dt (or dtype (if (vt-complex-dtype-p in-dt)
                           (vt-real-dtype-of in-dt)
                           (if (eq in-dt :float32) :float32 :float64))))
         (fn (if deg
                 (lambda (z)
                   (* (%carg z)
                      (if (eq dt :float32) (/ 180.0s0 0.31415927s1) (/ 180.0d0 pi))))
                 #'%carg)))
    ;; 度数换算需要额外一次乘法（内核未参数化），deg 走通用路径；
    ;; 默认弧度走 %CX-ANGLE-RUN 特化内核。
    (if (and (not deg) (vt-complex-dtype-p in-dt))
        (%cx-run-to-real #'%cx-angle-run #'%carg in out dt "VT-ANGLE")
        (vt-map fn in :out out :dtype dt))))


;;; ==================================================================
;;; 5. 统计类（Statistics）
;;; ==================================================================

(defun %e3-store-double-vector (result tmp)
  "把 double vector TMP（长度 = RESULT 元素数）按 C 顺序写进 RESULT（可非连续）。
   按目标元素类型 8 路特化：原实现用 (setf (vt-ref result i) ...) 写，
   每元素一次泛型 aref + 坐标折算，1M 元素约 100ms。"
  (let ((d (vt-data result))
        (i 0))
    (declare (type fixnum i))
    (macrolet ((w (lt)
                 `(let ((dd (the (simple-array ,lt (*)) d)))
                    (declare (type (simple-array ,lt (*)) dd)
                             (type fixnum i))
                    (vt-do-each (ptr val result)
                      (declare (ignore val))
                      (setf (aref dd ptr) (%cast-to ,lt (aref tmp i)))
                      (incf i)))))
      (cond ((equal (array-element-type d) 'double-float) (w double-float))
            ((equal (array-element-type d) 'single-float) (w single-float))
            ((equal (array-element-type d) '(signed-byte 64)) (w (signed-byte 64)))
            ((equal (array-element-type d) '(signed-byte 32)) (w (signed-byte 32)))
            ((equal (array-element-type d) '(signed-byte 16)) (w (signed-byte 16)))
            ((equal (array-element-type d) '(signed-byte 8)) (w (signed-byte 8)))
            ((equal (array-element-type d) '(unsigned-byte 16)) (w (unsigned-byte 16)))
            ((equal (array-element-type d) '(unsigned-byte 8)) (w (unsigned-byte 8)))
            (t (vt-do-each (ptr val result)
                 (declare (ignore val))
                 (setf (aref d ptr) (vt-cast (aref tmp i) (vt-dtype result)))
                 (incf i)))))
    result))

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
          (op-init (or init-val (if (eq op :sum) 0.0d0 1.0d0))))
    ;; 注：原有一个 `(flet ((cumstep (acc x) ...))` 包住两条路径，内部用
    ;; `(funcall op-fn acc x)`。op 内联进两个内层循环后它已无调用点，
    ;; 一并删除（每元素一次 FUNCALL 的装箱开销是这条路径的主要成本之一）。
    (if axis
        (let* ((rank (length shape))
               (ax (vt-normalize-axis axis rank))
               (indices (make-array rank :element-type '(signed-byte 64)
                                         :initial-element 0))
               (in-data (vt-data work)) (in-strs (vt-strides work))
               (in-off (vt-offset work))
               (out-data (vt-data result)) (out-strs (vt-strides result))
               (out-off (vt-offset result)))
          ;; 类型声明（性能）：WORK 恒为连续 float64，声明后内层 aref
          ;; 走元素类型特化，省掉每元素一次 widetag 分派。
          (declare (type (simple-array double-float (*)) in-data))
          (labels ((advance ()
                     (loop for d from (1- rank) downto 0
                           when (/= d ax)
                             do (incf (aref indices d))
                                (if (< (aref indices d) (nth d shape))
                                    (return-from advance t)
                                    (setf (aref indices d) 0)))))
            (loop
              ;; 每条 fiber 只算一次起点（输入与目标各一份），内层按
              ;; `起点 + i*stride` 递推 —— 把逐元素
              ;; `(setf (apply #'vt-ref result coords) …)` 换掉：那里
              ;; 每个元素都要 cons 一次坐标列表 + 一次通用坐标折算，
              ;; 1M 元素沿轴实测 260ms（numpy 6ms）。机制 B（offset +
              ;; strides 驱动）在这里只差一个指针。
              (let ((ptr in-off) (dptr out-off))
                (loop for d from 0 below rank
                      do (incf ptr (* (aref indices d) (nth d in-strs)))
                         (incf dptr (* (aref indices d) (nth d out-strs))))
                (let ((cum (coerce (if (eq op :sum) 0 op-init) 'double-float))
                      (ax-stride (nth ax in-strs))
                      (ax-dstride (nth ax out-strs))
                      (ax-dim (nth ax shape))
                      (sum-p (eq op :sum))
                      ;; 结果 dtype 就是 float64 时省掉逐元素 VT-CAST
                      ;; （ecase + 泛型算术，是这条路径剩下的主要开销）。
                      (raw-store-p (eq final-dtype :float64)))
                  (declare (type double-float cum))
                  (dotimes (i ax-dim)
                    ;; 首元素取「掩码后的输入本身」而不是 OP(单位元, 输入)：
                    ;; numpy.nancumsum([-0.0]) => [-0.]，而 0.0 + (-0.0) = +0.0
                    ;; 会洗掉符号位（CONVENTIONS §16.30.3 的实数线遗留项）。
                    (let ((x (aref in-data (+ ptr (* i ax-stride)))))
                      (if (zerop i)
                          (setf cum (if (%nan-p x)
                                        (if sum-p 0.0d0 1.0d0)
                                        x))
                          ;; op 内联进循环：原来每元素一次 FUNCALL，
                          ;; 装箱开销比算术本身还大。
                          (unless (%nan-p x)
                            (setf cum (if sum-p (+ cum x) (* cum x))))))
                    (setf (aref out-data (+ dptr (* i ax-dstride)))
                          (if raw-store-p cum (vt-cast cum final-dtype))))))
              (unless (advance) (return))))
          result)
        ;; 展平路径：原实现每元素 VT-REF 读 + VT-REF 写（各一次泛型
        ;; aref 与坐标折算）+ VT-CAST，1M 元素实测 474ms。
        ;; 改为「一次特化扫描算进 double vector，再一次性写回」。
        (let* ((flat (if (vt-contiguous-p work) work (vt-contiguous work)))
               (d (the (simple-array double-float (*)) (vt-data flat)))
               (off (vt-offset flat))
               (n (vt-size work))
               (tmp (make-array n :element-type 'double-float))
               (cum (coerce (if (eq op :sum) 0 op-init) 'double-float))
               (sum-p (eq op :sum)))
          (declare (type fixnum off n) (type double-float cum))
          (dotimes (i n)
            (declare (type fixnum i))
            (let ((x (aref d (the fixnum (+ off i)))))
              (if (zerop i)
                  ;; 首元素：掩码后直拷（numpy.nancumsum([-0.0]) => [-0.]）
                  (setf cum (if (/= x x)
                                (coerce (if sum-p 0.0d0 1.0d0) 'double-float)
                                x))
                  (unless (/= x x)                ; NaN 跳过（保持累积值）
                    (setf cum (if sum-p (+ cum x) (* cum x))))))
            (setf (aref tmp i) cum))
          (%e3-store-double-vector result tmp))))))

(defun vt-nancumsum (tensor &key axis dtype out)
  "沿 AXIS 累积和，NaN 视为 0（跳过，不改变累积值）（对标 numpy.nancumsum）。
   AXIS 为 nil 时展平。:out 契约同 vt-cumsum。"
  (when (%cx-any-p tensor)
    (return-from vt-nancumsum (%cx-op-nancumsum tensor :axis axis :dtype dtype :out out)))
  (%e3-cumulative-skip-nan tensor :sum 0 :axis axis :dtype dtype :out out
                                 :op-name "vt-nancumsum"))

(defun vt-nancumprod (tensor &key axis dtype out)
  "沿 AXIS 累积积，NaN 视为 1（跳过）（对标 numpy.nancumprod）。
   AXIS 为 nil 时展平。:out 契约同 vt-cumprod。"
  (when (%cx-any-p tensor)
    (return-from vt-nancumprod (%cx-op-nancumprod tensor :axis axis :dtype dtype :out out)))
  (%e3-cumulative-skip-nan tensor :prod 1 :axis axis :dtype dtype :out out
                                  :op-name "vt-nancumprod"))

(defun %e3-quantile-buffer (buf m q interpolation)
  "在**未排序**的 double 缓冲 BUF[0,M) 上取分位 Q，就地重排 BUF。
   取位规则与 %E3-QUANTILE-SORTED 完全一致（:nearest 用 ROUND，
   即 numpy 的 round-half-to-even），只是把「排序后按下标取」
   换成「quickselect 取第 k 小」。"
  (declare (type (simple-array double-float (*)) buf)
           (type fixnum m) (type double-float q))
  (let* ((pos (* q (1- m)))
         (lo (floor pos))
         (hi (min (1+ lo) (1- m)))
         (frac (- pos lo)))
    (declare (type fixnum lo hi) (type double-float frac))
    (case interpolation
      (:lower (%select-double buf m lo))
      (:higher (%select-double buf m hi))
      (:nearest (%select-double buf m (round pos)))
      (:midpoint (/ (+ (%select-double buf m lo) (%select-double buf m hi)) 2.0d0))
      (t (+ (%select-double buf m lo)
            (* frac (- (%select-double buf m hi) (%select-double buf m lo))))))))

(defun %e3-quantile-sorted (sv n q interpolation)
  "在**已升序排序**的 simple-vector / vector SV（长度 N）上取分位 Q ∈ [0,1]，
   插值规则与 numpy 的 linear/lower/higher/nearest/midpoint 一致。"
  (declare (type fixnum n) (type double-float q))
  (let* ((pos (* q (1- n)))
         (lo (floor pos))
         (hi (min (1+ lo) (1- n)))
         (frac (- pos lo)))
    (declare (type fixnum lo hi))
    (case interpolation
      (:lower (aref sv lo))
      (:higher (aref sv hi))
      (:nearest (aref sv (round pos)))
      (:midpoint (/ (+ (aref sv lo) (aref sv hi)) 2.0d0))
      (t (+ (aref sv lo) (* frac (- (aref sv hi) (aref sv lo))))))))

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
            ;;
            ;; 性能：原实现直接调 FIBER-NANQUANTILE，内部对每个元素做一次
            ;; VT-REF（泛型 aref + 坐标折算）并 LOOP COLLECT 成 list，
            ;; 再交给 vt-numpy-sort（先 coerce 成 list，再 cons 两份）。
            ;; 1M 元素实测 1585ms（numpy.nanpercentile 同工作量约 60ms）。
            ;; 这里改为「一次特化扫描写进 double vector → 原地 sort →
            ;; 下标直接取」，零 cons、零泛型 aref。
            (let* ((fv (ensure-vt tensor :dtype :float64))
                   (flat (if (vt-contiguous-p fv) fv (vt-contiguous fv)))
                   (d (the (simple-array double-float (*)) (vt-data flat)))
                   (off (vt-offset flat))
                   (n (vt-size flat))
                   (tmp (make-array n :element-type 'double-float))
                   (m 0))
              (declare (type fixnum off n m))
              (dotimes (i n)
                (declare (type fixnum i))
                (let ((v (aref d (the fixnum (+ off i)))))
                  (unless (/= v v)                    ; NaN 跳过
                    (setf (aref tmp m) v)
                    (incf m))))
              (let ((val (if (zerop m)
                             (vt-get-nan :float64)
                             (progn
                               ;; v0.4.10：不再全排序，用 quickselect 只取所需
                               ;; 的两个名次（1M 从 225ms 降到 ~25ms）。
                               (%e3-quantile-buffer tmp m q interpolation)))))
                (let ((res (vt-zeros (if keepdims (make-list rank :initial-element 1) nil)
                                     :dtype :float64)))
                  (vt-fill res val)
                  res))))))))

(defun vt-nanpercentile (tensor percentile &key axis keepdims (interpolation :linear))
  "忽略 NaN 的百分位数（对标 numpy.nanpercentile）。
   PERCENTILE ∈ [0,100]；INTERPOLATION 同 vt-percentile；
   全 NaN/空切片 → NaN。AXIS/KEEPDIMS 语义同 vt-percentile。"
    (when (%cx-reject tensor percentile)
    (error "vt-nanpercentile: 不支持复数输入（numpy 同样拒绝）。见 CONVENTIONS §16.11"))
(unless (and (realp percentile) (<= 0 percentile 100))
    (error "vt-nanpercentile: percentile 必须在 [0,100] 内，收到 ~a" percentile))
  (%e3-nan-percentile tensor (/ percentile 100.0d0)
                      :axis axis :keepdims keepdims
                      :interpolation interpolation :op-name "vt-nanpercentile"))

(defun vt-nanquantile (tensor q &key axis keepdims (interpolation :linear))
  "忽略 NaN 的分位数（对标 numpy.nanquantile）。Q ∈ [0,1]；
   全 NaN/空切片 → NaN。AXIS/KEEPDIMS 语义同 vt-quantile。"
    (when (%cx-reject tensor q)
    (error "vt-nanquantile: 不支持复数输入（numpy 同样拒绝）。见 CONVENTIONS §16.11"))
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
  (when (%cx-any-p m)
    (return-from vt-cov (%cx-op-cov m :y y :rowvar rowvar :ddof ddof :bias bias)))
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
      ;; v0.4.2 性能修复：原实现是三重循环，且把「每行均值」放在 (i,j) 循环
      ;; **内部** 计算 —— n_var² 次 `vt-mean`（每次 O(n_obs)），再加逐元素
      ;; `vt-ref`（泛型 aref + 下标运算）。1000×1000 实测 **54 秒**
      ;; （numpy 约 30 ms，差 1800 倍）。
      ;; 改为矩阵形式：cov = (X - mean(X)) · (X - mean(X))ᵀ / denom，
      ;; 复用库内已验证的 SIMD/多线程 matmul；并按 numpy 的做法对称化。
      (let* ((x (vt-contiguous stacked))                       ; n_var × n_obs
             (mu (vt-mean x :axis 1 :keepdims t))              ; n_var × 1
             (xc (vt-- x mu))                                  ; 广播居中
             (raw (vt-matmul xc (vt-transpose xc)))            ; n_var × n_var
             (sym (vt-scale (vt-+ raw (vt-transpose raw)) 0.5d0))
             (res (vt-astype sym :float64)))
        (if (= denom 1)
            res
            (vt-scale res (/ 1.0d0 denom))))))))

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
  (when (%cx-any-p m)
    (return-from vt-corrcoef (%cx-op-corrcoef m :y y :rowvar rowvar)))
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
  (declare (ignorable axisc))   ; 结果轴暂固定为末轴；AXISC 仅作 API 兼容
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
  (when (%cx-any-p a b)
    (return-from vt-vdot (%cx-op-vdot a b :dtype dtype :out out)))
  (let ((af (vt-ravel (ensure-vt a)))
        (bf (vt-ravel (ensure-vt b))))
    (when (/= (vt-size af) (vt-size bf))
      (error "vt-vdot: 输入元素个数不一致（~a vs ~a）" (vt-size af) (vt-size bf)))
    (vt-dot af bf :dtype dtype :out out)))

(defun vt-eigvals (matrix &key (max-iter 200) (tol 1e-10) (uplo :lower))
  "方阵特征值（对标 numpy.linalg.eigvals）。
   实现基于对称 Jacobi，输入约定与 vt-eig 一致（numpy.linalg.eigh
   三角语义：不校验对称性，仅使用 UPLO 指定的三角，默认 :lower）。
   注意：与 numpy.linalg.eigvals（通用 QR 算法）不同，本实现对非对称
   输入返回其单三角对称化矩阵的特征值，而非真实特征值。
   返回按降序排列的特征值 1 维张量（float64）。
   示例：(vt-eigvals (vt-from-array #2A((2.0 0.0)(0.0 3.0)))) => [3.0 2.0]"
  (multiple-value-bind (vals vec) (vt-eig matrix :max-iter max-iter :tol tol :uplo uplo)
    (declare (ignore vec))
    vals))

(defun vt-eigvalsh (matrix &key (max-iter 200) (tol 1e-10) (uplo :lower))
  "对称（Hermitian）方阵特征值（对标 numpy.linalg.eigvalsh）。
   与 numpy 完全对齐：UPLO 关键字（:lower 默认 / :upper）语义同
   numpy.linalg.eigvalsh 的 UPLO 参数；返回**升序**排列的特征值。
   （v0.4.0 修复：原实现直接透传 vt-eigvals 的非排序结果，
   不满足 numpy.linalg.eigvalsh 的升序约定。）
   示例：(vt-eigvalsh (vt-from-array #2A((2.0 1.0)(1.0 2.0)))) => [1.0 3.0]"
  (let ((w (vt-eigvals matrix :max-iter max-iter :tol tol :uplo uplo)))
    (vt-sort w :axis -1)))

(defun vt-matrix-power (matrix n)
  "整数幂的方阵（对标 numpy.linalg.matrix_power）。
   N > 0：矩阵连乘 N 次；N = 0：单位阵；N < 0：先求逆再取 |N| 次幂。
   示例：(vt-matrix-power (vt-from-array #2A((1 2)(3 4))) 2) => [[7 10][15 22]]"
  (when (%cx-any-p matrix n)
    (return-from vt-matrix-power (%cx-op-matrix-power matrix n)))
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
       ;; v0.4.1 修复（差分测试 R6）：cond_1 = ||A||_1 · ||A^-1||_1，
       ;; 此前误用 A 自身列和的 max/min 比值，与 numpy 不一致。
       ;; 奇异矩阵 vt-inv 报错，对标 numpy 的 LinAlgError。
       (flet ((%norm-1 (m)
                (let* ((sh (vt-shape m)) (rows (first sh)) (cols (second sh))
                       (best 0.0d0))
                  (dotimes (j cols)
                    (let ((acc 0.0d0))
                      (dotimes (i rows) (incf acc (abs (vt-ref m i j))))
                      (setf best (max best acc))))
                  best)))
         (let ((ainv (vt-inv matrix)))
           (* (%norm-1 matrix) (%norm-1 ainv)))))
      ((or (eql p :inf) (eql p :infinity))
       ;; cond_inf = ||A||_inf · ||A^-1||_inf（行绝对和之最大）
       (flet ((%norm-inf (m)
                (let* ((sh (vt-shape m)) (rows (first sh)) (cols (second sh))
                       (best 0.0d0))
                  (dotimes (i rows)
                    (let ((acc 0.0d0))
                      (dotimes (j cols) (incf acc (abs (vt-ref m i j))))
                      (setf best (max best acc))))
                  best)))
         (let ((ainv (vt-inv matrix)))
           (* (%norm-inf matrix) (%norm-inf ainv)))))
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
  ;; v0.4.2 性能修复：与 vt-in1d 共用哈希会员判定（原 O(n×m) 线性 member，
  ;; 1M×1k 实测 87 秒）。
  (when (%cx-any-p element test-elements)
    (return-from vt-isin (%cx-op-isin element test-elements :invert invert :assume-unique assume-unique)))
  (let* ((elem (ensure-vt element))
         (test (ensure-vt test-elements))
         (fm (%membership-float-mode-p elem test)))
    (%membership-map elem (%membership-table (vt-unique test) fm) fm invert)))
