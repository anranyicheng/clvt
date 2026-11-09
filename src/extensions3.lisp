;;;; extensions3.lisp — 补充 NumPy 重要缺失函数（第三批）
;;;;
;;;; 本文件严格遵循 CONVENTIONS.md：
;;;;   §1  三层架构：逻辑层（shape/dtype/广播/归约语义）/
;;;;        物理层（strides/offset/连续性/别名）/ 执行层（快路径 vs 通用路径）；
;;;;   §3/§17  dtype 系统：9 种逻辑 dtype（含 :bool）；比较/逻辑/谓词函数返回 :bool；
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
             (push (%vt-ref-raw f i) acc))))
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

(defun %e3-double-result-dtype (dtype raw-x xo raw-y yo)
  "「以 double 计算」的那批二元函数（logaddexp / copysign / nextafter /
   float-power）的输出 dtype。

   两个维度必须分开看：
     · **计算精度**：这些函数一律先把输入提到 float64 再算（double 精度
       是它们的语义，`vt-nextafter` 的「一格之差」必须用 double 才看得见）。
     · **输出 dtype**：由 `%weak-final-dtype` 按 NEP 50 决定 —— 张量宽度
       说了算，弱标量只贡献种类（§16.31）。

   XO / YO 必须是 `(ensure-vt x)` 而**不是** `(%coerce-float-input xo :float64)`：
   把已经提到 float64 的值传进来，判定照样得 float64，等于没改。这正是
   本函数第一版踩的坑 —— 修完 dtype 一测还是 FLOAT64，因为参数给错了。

   原来的写法是 `(%e3-float-prefer-dtype dtype :float64)`，把 in-dtype
   硬编码成 float64，于是 `(vt-logaddexp f32数组 1.5d0)`、乃至
   `(vt-logaddexp f32数组 f32数组)` 都返回 float64，numpy 是 float32。
   「用 double 算」不等于「结果必须是 double」。"
  (%e3-float-prefer-dtype dtype (%weak-final-dtype raw-x xo raw-y yo)))

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
   示例：(vt-trim-zeros '(0 0 1 2 0 3 0 0)) => [1 2 0 3]

   性能（v0.4.21）：原实现把整条数据 `(COERCE … 'LIST)` 成 1M 个 cons，
   再用 `(NTH K VALS)` 逐个试探两端、`(SUBSEQ VALS START END)` 复制一段
   list，最后 `VT-FROM-SEQUENCE` 逐元素 `VT-CAST` 建结果。1M 元素实测
   28ms —— 而真正需要的只是「两端各走几步」。现在直接在类型化的
   double-float 缓冲上走指针，找到 [start, end) 后 `VT-SLICE` + `VT-CONTIGUOUS`。
   判定仍走 **float64**（与原实现一致）：复数输入按实部判零，
   而不是按 `(ZEROP #C(0 1))`（那会给「非零」），语义不变。
   结果元素直接取自原张量 —— 原实现把值经 float64 往返一遍，
   `|x| > 2^53` 的整数会被静默舍入，这里顺带修掉了。"
  (let* ((input (ensure-vt tensor))
         (one-d (if (null (vt-shape input))
                    (%flat-offset0 (vt-reshape input '(1)))
                    (%flat-offset0 input)))
         (n (vt-size one-d)))
    (unless (member trim '(:fb :f :b))
      (error "vt-trim-zeros: trim 必须是 :fb / :f / :b 之一，收到 ~a" trim))
    (if (zerop n)
        one-d
        (let* ((fv (if (eq (vt-dtype one-d) :float64)
                       one-d
                       (vt-astype one-d :float64)))
               (d (vt-data fv)))
          (declare (type (simple-array double-float (*)) d)
                   (type fixnum n))
          (let ((start 0) (end n))
            (declare (type fixnum start end))
            (when (member trim '(:fb :f))
              (loop while (and (< start end) (zerop (aref d start)))
                    do (incf start)))
            (when (member trim '(:fb :b))
              (loop while (and (< start end) (zerop (aref d (the fixnum (1- end)))))
                    do (decf end)))
            (vt-contiguous (vt-slice one-d (list start end))))))))

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
        (vt-resize (vt-fromiter '(1 2 3)) '(5))   => [1 2 3 1 2]

   填充就是「把同一段模式按周期重复」，所以按**块倍增**用 %MEMCPY 拷：
   先把源整段写入，再每次把已填好的前缀翻倍（块长不超过剩余量），
   落点恒为连续张量的线性下标 —— 没有 `(mod i n)`，也没有逐元素
   `(setf (vt-ref …))`（那要走通用 aref + 坐标折算）。"
  (let* ((flat (vt-contiguous (vt-ravel tensor)))
         (n (vt-size flat))
         (total (vt-shape-to-size new-shape))
         (res (vt-zeros new-shape :dtype (vt-dtype tensor))))
    (when (or (zerop total) (zerop n))
      (return-from vt-resize res))
    (let ((dst (vt-data res))
          (src (vt-data flat))
          (soff (vt-offset flat)))
      (declare (type fixnum soff))
      ;; 源比目标长时只取前 TOTAL 个（numpy.resize 的截断语义）
      (let ((filled (min total n)))
        (declare (type fixnum filled))
        (%memcpy dst 0 src soff filled)
        (loop while (< filled total)
              do (let ((chunk (min filled (- total filled))))
                   (declare (type fixnum chunk))
                   ;; DST 与 SRC 是同一数组且目标区间在源之后：CLHS REPLACE 对
                   ;; 「同一对象重叠」规定为「先整体取出再写回」，结果与逐元素
                   ;; 取模相同。
                   (%memcpy dst filled dst 0 chunk)
                   (incf filled chunk))))
      res)))

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
              (let ((sel (%vt-ref-raw idx linear)))
                (setf sel (if (minusp sel) (+ sel ax-dim) sel))
                (unless (<= 0 sel (1- ax-dim))
                  (error "vt-put-along-axis: 索引 ~a 越界（轴长 ~a）" sel ax-dim))
                ;; 沿非 AXIS 维广播写入（indices 长度为 1 时铺满该维）
                (labels ((put (d coords)
                           (if (= d rank)
                               (let ((tcoords (copy-list coords)))
                                 (setf (nth ax tcoords) sel)
                                 (setf (aref out-data (+ out-off (lin tcoords shape strides)))
                                       (%vt-ref-raw val-flat linear)))
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
   示例：(vt-compress '(1 0 1 0 1) (vt-fromiter '(0 1 2 3 4) :dtype :int64)) => [0 2 4]
   注意 VT-FROMITER 的 dtype 缺省是 :float64，省略 :dtype 时结果是 [0.0 2.0 4.0]。

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

(defun %e3-index-layer-fill (layer outer axis-len stride dtype)
  "把 VT-INDICES 的一层就地填好：值 = 该元素沿本层轴上的坐标 I。
   布局是「外层块（OUTER 个）× 轴长 × 连续重复（STRIDE 个）」，
   落点因此是纯线性下标 `off + o·stride·axis-len + i·stride + s`。"
  (declare (type fixnum outer axis-len stride))
  (let ((d (vt-data layer))
        (off (vt-offset layer))
        ;; `:bool` 的物理元素是 0/1 字节，而 `VT-CAST` 的**逻辑层**契约是
        ;; 返回 T/NIL（§17.10.8）。直接把 T/NIL 塞进 (unsigned-byte 8) 数组
        ;; 会抛 TYPE-ERROR；索引值只有「零 / 非零」两种，这里显式落到 0/1。
        (boolp (eq dtype :bool)))
    (declare (type fixnum off))
    ;; 逐元素写 (setf (vt-ref …)) 要为每个元素算一次坐标再走通用 aref；
    ;; 按元素类型特化后内层只剩三次加法。VT-CAST 提到 I 层（每行一次），
    ;; 与逐元素调用语义相同（下标非负，故 i 与 dtype 的转换是幂等的）。
    (flet ((val (i)
             "坐标 I 按目标 dtype 落地的**物理值**。"
             (if boolp (if (zerop i) 0 1) (vt-cast i dtype))))
      (macrolet ((walk (lt)
                 `(let ((dd (the (simple-array ,lt (*)) d)))
                    (declare (type (simple-array ,lt (*)) dd)
                             (type fixnum off outer axis-len stride))
                    ;; 待写入的值只取决于 I（≤ 轴长个），按轴长预转一次；
                    ;; 放进 O 循环里就是 OUTER × 轴长 次 VT-CAST —— 而
                    ;; VT-CAST 自己包着 WITH-FLOAT-SAFE（两次 FP 控制字
                    ;; 读写），沿**末轴**的那一层恰好 OUTER = ∏前置维，
                    ;; 会把整个函数的耗时拖到一个量级以上。
                    (let ((vls (make-array axis-len :element-type ',lt)))
                      (declare (type (simple-array ,lt (*)) vls))
                      (dotimes (i axis-len)
                        (declare (type fixnum i))
                        (setf (aref vls i) (the ,lt (val i))))
                      (dotimes (o outer)
                        (declare (type fixnum o))
                        (let ((base (the fixnum (+ off (* o stride axis-len)))))
                          (declare (type fixnum base))
                          (dotimes (i axis-len)
                            (declare (type fixnum i))
                            (let ((cv (aref vls i)))
                              (dotimes (s stride)
                                (declare (type fixnum s))
                                (setf (aref dd (the fixnum (+ base (* i stride) s)))
                                      cv)))))))))
               (walk-generic ()
                 "未知元素类型（复数等）：按线性下标写通用 AREF。
                  ★ 必须写 `(AREF D …)` 而不是 `(SETF (VT-REF LAYER …))` ——
                  LAYER 是 (R D0 D1 …) 的多维张量，VT-REF 要求索引个数等于
                  秩，给一个展平下标会直接报「索引数量 1 与维度数 3 不匹配」。
                  层由 VT-ZEROS 建成 C 连续、offset 0，故线性下标即逻辑位置。"
                 `(dotimes (o outer)
                    (declare (type fixnum o))
                    (dotimes (i axis-len)
                      (declare (type fixnum i))
                      (let ((cv (val i)))
                        (dotimes (s stride)
                          (declare (type fixnum s))
                          (setf (aref d (the fixnum (+ off (* o stride axis-len)
                                                       (* i stride) s)))
                                cv)))))))
        (let ((et (array-element-type d)))
          (cond ((equal et 'double-float)            (walk double-float))
                ((equal et 'single-float)            (walk single-float))
                ((equal et '(signed-byte 64))        (walk (signed-byte 64)))
                ((equal et '(signed-byte 32))        (walk (signed-byte 32)))
                ((equal et '(signed-byte 16))        (walk (signed-byte 16)))
                ((equal et '(signed-byte 8))         (walk (signed-byte 8)))
                ((equal et '(unsigned-byte 16))      (walk (unsigned-byte 16)))
                ((equal et '(unsigned-byte 8))       (walk (unsigned-byte 8)))
                (t (walk-generic)))))))
  layer)

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
               ;; 该层沿 AX 的步长（行主序）= AX 之后各维之积，每个「i 值」
               ;; 连续重复 STRIDE 次；OUTER 是 AX 之前各维之积。三者都单遍
               ;; 求出（`(apply #'* (subseq …))` 要为每层切一次子表）。
               (stride 1)
               (outer 1)
               (axis-len 1)
               (di 0))
          (declare (type fixnum stride outer axis-len di))
          (dolist (d sh)
            (cond ((< di ax) (setf outer (the fixnum (* outer d))))
                  ((> di ax) (setf stride (the fixnum (* stride d))))
                  (t (setf axis-len d)))
            (incf di))
          ;; STRIDE 为 0 说明该层后面有 0 长维（如 (2 0 3) 的第 0 层），
          ;; 整层没有任何元素可写。把它兜成 1 再照常写就会落到空 buffer 上；
          ;; 层大小为 0 时直接跳过填充（numpy 对 (2 0 3) 给出全空张量）。
          (when (plusp (%shape-product sh))
            (%e3-index-layer-fill layer outer axis-len
                                  (if (zerop stride) 1 stride) dtype))
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
           ;; 目标为 :bool 时 T/NIL 是**标量值**（§17.10.8）。不加这一条就会
           ;; 落进序列分支：`(ensure-vt t)` 是 0 维、`vt-ravel` 后只剩 1 个
           ;; 元素，于是 `(vt-fill-diagonal m t)` 只填对角线第一个位置。
           (scalar-p (or (numberp value)
                         (and (eq (vt-dtype tensor) :bool)
                              (or (eq value t) (null value)))))
           (vals (cond (scalar-p nil)
                       (t (coerce (vt-data (vt-astype
                                            (vt-ravel (ensure-vt
                                                       value
                                                       :dtype (and (eq (vt-dtype tensor) :bool)
                                                                   :bool)))
                                            :int64))
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

(declaim (inline %e3-logaddexp2))
(defun %e3-logaddexp2 (a b)
  "logaddexp 内核（判据与改前 lambda 逐条相同：NaN → +Inf → -Inf → 稳定式）。"
  (let ((af (coerce a 'double-float)) (bf (coerce b 'double-float)))
    (cond ((%nan-p af) af)
          ((%nan-p bf) bf)
          ((%pos-inf-p af) af)
          ((%pos-inf-p bf) bf)
          ((%neg-inf-p af) bf)
          ((%neg-inf-p bf) af)
          (t (let ((m (max af bf)))
               (+ m (log (+ 1.0d0 (exp (- (min af bf) m))))))))))

(defun vt-logaddexp (x y &key out dtype)
  "逐元素 log(exp(x) + exp(y))，数值稳定（对标 numpy.logaddexp）。
   整数输入输出 float64，float32 张量进 float32 出（弱标量不升宽，§16.31）。
   :out 契约同 vt-abs（支持广播）。"
  (when (%cx-reject x y)
    (error "vt-logaddexp: 不支持复数输入（numpy 同样拒绝）。见 CONVENTIONS §16.11"))
  (let* ((xo (ensure-vt x))
         (yo (ensure-vt y))
         (xe (%coerce-float-input xo :float64))
         (ye (%coerce-float-input yo :float64)))
    (vt-fast-map #'%e3-logaddexp2 xe ye
                 :out out :dtype (%e3-double-result-dtype dtype x xo y yo))))

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
  (let* ((xo (ensure-vt x))
         (yo (ensure-vt y))
         (xe (%coerce-float-input xo :float64))
         (ye (%coerce-float-input yo :float64)))
    (vt-map (lambda (a b)
              (let ((af (coerce a 'double-float)) (bf (coerce b 'double-float)))
                (cond ((%nan-p af) af)
                      ((%nan-p bf) bf)
                      ((and (minusp af) (/= bf (floor bf))) (vt-get-nan :float64))
                      (t (expt af bf)))))
            xe ye
            :out out :dtype (%e3-double-result-dtype dtype x xo y yo))))

(declaim (inline %e3-sign-bit-p))
(defun %e3-sign-bit-p (x)
  "IEEE 754 符号位判定（含 -0.0 与 NaN 的符号位）。
   minusp/float-sign 均无法区分 ±0.0，这里直接读 double-float 位模式。"
  (declare (double-float x))
  (logbitp 63 (logand #xFFFFFFFFFFFFFFFF (sb-kernel:double-float-bits x))))

;;; 谓词族的具名算子（v0.4.21）——理由同 elementwise1.lisp 同名小节：
;;; `vt-fast-map` 只对 `(function <符号>)` 内联，匿名 lambda 一律退回闭包路径。
;;; 这三个算子都靠 `with-float-safe`（`vt-fast-map` 外层已屏蔽），
;;; 因此对 NaN 做 `(> x most-positive-double-float)` 不会抛 INVALID-OPERATION。
(declaim (inline %pred-signbit %pred-posinf %pred-neginf))

(defun %pred-signbit (v)
  "符号位 → 0/1。入参已由调用方 `ensure-vt :dtype :float64` 收敛到 double。"
  (if (%e3-sign-bit-p (coerce v 'double-float)) 1 0))

(defun %pred-posinf (v) (if (and (floatp v) (%pos-inf-p v)) 1 0))
(defun %pred-neginf (v) (if (and (floatp v) (%neg-inf-p v)) 1 0))

(declaim (inline %e3-copysign2))
(defun %e3-copysign2 (a b)
  "copysign 内核：符号取自 b 的 IEEE 符号位（minusp 认不出 −0.0）。"
  ;; v0.4.1 修复（差分测试 R4）：符号取自 y 的 IEEE 符号位，
  ;; minusp 无法识别 -0.0（copysign(1,-0.0) 应为 -1.0）。
  (let ((af (coerce a 'double-float)) (bf (coerce b 'double-float)))
    (if (%e3-sign-bit-p bf) (- (abs af)) (abs af))))

(defun vt-copysign (x y &key out dtype)
  "逐元素赋予 X 的绝对值与 Y 的符号（对标 numpy.copysign）。
   结果恒为浮点；NaN 符号位处理与 numpy 一致（copysign(1, nan) → 1）。
   :out 契约同 vt-abs（支持广播）。"
  (when (%cx-reject x y)
    (error "vt-copysign: 不支持复数输入（numpy 同样拒绝）。见 CONVENTIONS §16.11"))
  (let* ((xo (ensure-vt x))
         (yo (ensure-vt y))
         (xe (%coerce-float-input xo :float64))
         (ye (%coerce-float-input yo :float64)))
    (vt-fast-map #'%e3-copysign2 xe ye
                 :out out :dtype (%e3-double-result-dtype dtype x xo y yo))))

(defun vt-signbit (tensor &key out)
  "逐元素符号位判定，返回 **:bool** 布尔张量（对标 numpy.signbit，§17）。
   x < 0 或为负零（-0.0）→ True，否则 False。:out 契约同 vt-=（dtype :bool）。

   性能（v0.4.21）：原实现以匿名 lambda 传 `vt-map`，命中不了
   `vt-fast-map` 的内联分支，1M float64 实测 31ms（numpy 0.9ms）。
   改为具名内联算子后走 `%mask1-loop`，语义逐位不变 ——
   `vt-map` 的写回本就是 `vt-cast-fun` 的真值化，而算子恒返 0/1。"
  (when (%cx-reject tensor)
    (error "vt-signbit: 不支持复数输入（numpy 同样拒绝）。见 CONVENTIONS §16.11"))
  (vt-fast-map #'%pred-signbit (ensure-vt tensor :dtype :float64)
               :out out :dtype :bool))

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
   示例：(vt-item (vt-nextafter 1.0d0 2.0d0)) => 1.0000000000000002d0
   打印按 6 位有效数字显示为 `1.0`（§17.10.19）：要看出一格之差的差别，
   必须取标量值而不是读打印结果。"
    (when (%cx-reject x y)
    (error "vt-nextafter: 不支持复数输入（numpy 同样拒绝）。见 CONVENTIONS §16.11"))
  ;; 性能：原来是 (lambda (a b) ...) → vt-map 的 funcall 路径（1M 元素
  ;; 实测 270ms）。%e3-nextafter 已是 inline 函数，直接用 vt-fast-map
  ;; 内联成二元特化循环；仅结果 dtype 为 :float32（需要先降精度）保留
  ;; 通用路径，保证「double 精度计算、输出降精度」的既有语义不变。
  ;; 结果 dtype 走 %e3-double-result-dtype：原先把 in-dtype 硬编码成
  ;; :float64，于是 f32 张量哪怕两侧都是 f32 也返回 float64（§16.31）。
  (let* ((xo (ensure-vt x))
         (yo (ensure-vt y))
         (xv (%coerce-float-input xo :float64))
         (yv (%coerce-float-input yo :float64))
         (dt (%e3-double-result-dtype dtype x xo y yo)))
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
   示例：(vt-item (vt-spacing 1.0d0)) => 2.220446049250313d-16
        (vt-item (vt-spacing 0.0d0)) => 4.9406564584124654d-324（最小非规格化数）
   打印形式分别是 `2.220446e-16` 与 `4.940656e-324`（§17.10.19）。"
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
            xv yv :out out
            ;; 弱标量（NEP 50，§16.31）：`(vt-gcd 2 int8数组)` 应保持 int8，
            ;; 原写法 promote 已 ensure 的 int64 0 维张量 → 升宽成 int64。
            :dtype (or dtype (%weak-final-dtype x xv y yv)))))

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
            xv yv :out out
            ;; 同 vt-gcd：弱整数标量不参与宽度判定（§16.31 NEP 50）。
            :dtype (or dtype (%weak-final-dtype x xv y yv)))))

(defun vt-divmod (x y &key out dtype)
  "逐元素同时返回商与余数（对标 numpy.divmod）。
   余数符号跟随除数（floor 语义）；返回 (values quotient remainder)。
   :out 契约：本函数返回两个结果，暂不支持 :out（传 out 将报错）。
   布尔输入（CONVENTIONS §17.10.13）：结果 dtype 取 **:int8**，与
   `vt-mod` / `vt-rem` / `vt-div` 同一条规则 —— 原实现没有这一步，商沿用
   输入的 `:bool`，`floor(0/0)` 得 NaN 再被 `%bool-flag` 收成 1，于是
   `(vt-divmod #(T NIL) #(T NIL))` 的商是 `(T T)`（numpy：int8 `[1 0]`）。
   示例：(multiple-value-bind (q r) (vt-divmod '(7 -7) '(2 3)) ...)
        => q=[3 -3], r=[1 2]"
  ;; docstring 一直写着「传 out 将报错」，实现却是 (declare (ignore out)) ——
  ;; 静默吞掉：用户给了 :out，既不报错也不写入，拿到的是两个新张量，
  ;; 而 out 原封不动。这是本轮全库 :out 契约扫描（274 次探测）里唯一的漏网，
  ;; 其余 273 次都在形状或 dtype 不符时按 §4.4 报错。
  (when out
    (error "vt-divmod: 本函数返回两个值（商、余数），不支持 :out。~%~
            需要原地写入请分别调用 vt-div / vt-mod 并各自传 :out。~%~
            见 CONVENTIONS §17.10.18"))
  (when (and (null dtype) (%all-bool-tensors-p (list x y)))
    (setf dtype :int8))
  (let* ((xv (ensure-vt x)) (yv (ensure-vt y))
         ;; 商直接调 VT-DIV —— 与余数调 VT-MOD 对称，也保证
         ;;   divmod(x,y) 的商 == vt-div(x,y)
         ;; 这条恒等式成立。原实现自带一份 floor 除法副本，两处就分了叉：
         ;;   (vt-div #(1.0 -1.0 0.0) #(0.0 0.0 0.0)) → (-Inf -Inf NaN)
         ;;   (nth-value 0 (vt-divmod ...))            → (NaN NaN NaN)
         ;; 同一个输入、同一族两个入口，三种答案（numpy 是 (Inf -Inf NaN)）。
         ;; 副本里那个 (if (or nan-p nan-p) nan nan) 两支相同，是从没写完的残迹。
         (quot (vt-div xv yv :dtype dtype)))
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
   示例：(vt-nan-to-num (vt-from-array (vector (vt-get-nan :float64) 1.0d0))) => [0.0 1.0]
        (vt-nan-to-num (vt-from-array (vector (vt-get-pos-inf :float64) 1.0d0)))
        => [1.797693e+308 1.0]（+Inf 换成该 dtype 的最大有限值）"
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
   每元素一次泛型 aref + 坐标折算，1M 元素约 100ms。

   v0.4.20 补漏：RESULT **连续**（调用点几乎都是 `vt-zeros` 造的新数组）
   时改走线性下标直写。原先不论连续与否都走 `VT-DO-EACH`，那个宏每元素要
   算一次多维坐标、过一遍 `%cast-to`，即便目标是 double→double 也是
   11 ns/元素（1M 实测 11 ms），而裸循环只要 0.5 ms。"
  (let ((d (vt-data result))
        (i 0)
        (off (vt-offset result)))
    ;; TMP 恒为 `(make-array N :element-type 'double-float)`（本文件两处调用
    ;; 点都是），声明后 `(aref tmp j)` 才走定长数组取值；不声明就是通用
    ;; AREF，每元素一次 widetag 分派，1M 实测多出 3.3 ms。
    (declare (type fixnum i off)
             (type (simple-array double-float (*)) tmp))
    (macrolet ((w (lt)
                 `(let ((dd (the (simple-array ,lt (*)) d)))
                    (declare (type (simple-array ,lt (*)) dd)
                             (type fixnum i))
                    (vt-do-each (ptr val result)
                      (declare (ignore val))
                      (setf (aref dd ptr) (%cast-to ,lt (aref tmp i)))
                      (incf i))))
               (w-bool ()
                 "布尔目标：真值化，不是截断。
                    `(unsigned-byte 8)` 同时承载 `:uint8` 与 `:bool`，
                    而两者的写回语义相反：uint8 要截断（2.5 → 2），bool 要
                    真值化（0.5 → 1）。此前这里只有截断那条，于是
                    `(vt-nancumsum #(0.5 -0.5 0.25) :dtype :bool)` 得到
                    `(0 0 0)` —— 三个真值里有两个被截成了 NIL。"
                 `(let ((dd (the (simple-array (unsigned-byte 8) (*)) d)))
                    (declare (type (simple-array (unsigned-byte 8) (*)) dd)
                             (type fixnum i))
                    (vt-do-each (ptr val result)
                      (declare (ignore val))
                      (setf (aref dd ptr) (if (zerop (aref tmp i)) 0 1))
                      (incf i))))
               (wc (lt val)
                 ;; 连续目标的线性直写：C 顺序的第 j 个元素就在基数组的
                 ;; OFF+J 处，无需坐标折算。VAL 里可以引用循环变量 J。
                 `(let ((dd (the (simple-array ,lt (*)) d)))
                    (declare (type (simple-array ,lt (*)) dd)
                             (type fixnum off))
                    (dotimes (j (vt-size result))
                      (declare (type fixnum j))
                      (setf (aref dd (the fixnum (+ off j))) ,val))
                    t))
               (wc-bool ()
                 `(let ((dd (the (simple-array (unsigned-byte 8) (*)) d)))
                    (declare (type (simple-array (unsigned-byte 8) (*)) dd)
                             (type fixnum off))
                    (dotimes (j (vt-size result))
                      (declare (type fixnum j))
                      (setf (aref dd (the fixnum (+ off j)))
                            (if (zerop (aref tmp j)) 0 1)))
                    t)))
      (let ((contig-p (vt-contiguous-p result))
            (et (array-element-type d)))
        (cond ((and (equal et '(unsigned-byte 8)) (eq (vt-dtype result) :bool))
                 (if contig-p (wc-bool) (w-bool)))
                ;; double → double 不需要任何转换，直接搬（`%cast-to`
                ;; double-float 虽然语义上是恒等，但它带着一次类型判断，
                ;; 1M 元素实测 4.3 ms vs 裸拷贝 0.5 ms）。
                ((equal et 'double-float)
                 (if contig-p (wc double-float (aref tmp j)) (w double-float)))
                ((equal et 'single-float)
                 (if contig-p (wc single-float (%cast-to single-float (aref tmp j)))
                     (w single-float)))
                ((equal et '(signed-byte 64))
                 (if contig-p (wc (signed-byte 64) (%cast-to (signed-byte 64) (aref tmp j)))
                     (w (signed-byte 64))))
                ((equal et '(signed-byte 32))
                 (if contig-p (wc (signed-byte 32) (%cast-to (signed-byte 32) (aref tmp j)))
                     (w (signed-byte 32))))
                ((equal et '(signed-byte 16))
                 (if contig-p (wc (signed-byte 16) (%cast-to (signed-byte 16) (aref tmp j)))
                     (w (signed-byte 16))))
                ((equal et '(signed-byte 8))
                 (if contig-p (wc (signed-byte 8) (%cast-to (signed-byte 8) (aref tmp j)))
                     (w (signed-byte 8))))
                ((equal et '(unsigned-byte 16))
                 (if contig-p (wc (unsigned-byte 16) (%cast-to (unsigned-byte 16) (aref tmp j)))
                     (w (unsigned-byte 16))))
                ((equal et '(unsigned-byte 8))
                 (if contig-p (wc (unsigned-byte 8) (%cast-to (unsigned-byte 8) (aref tmp j)))
                     (w (unsigned-byte 8))))
                (t (vt-do-each (ptr val result)
                     (declare (ignore val))
                     (setf (aref d ptr) (vt-cast (aref tmp i) (vt-dtype result)))
                     (incf i))))))
    result))

(defun %e3-cumulative-skip-nan (tensor op init-val &key axis dtype out op-name)
  "NaN 感知的累积原语：沿 AXIS 累积 OP，遇到 NaN 时跳过（不改变累积值），
   但输出的该位置仍写入当前累积值（对标 numpy.nancumsum/nancumprod）。
   AXIS 为 nil 时展平。init-val 为累积初值（0 / 1）。

   结果 dtype 与 `VT-CUMSUM` / `VT-CUMPROD` 共用一张提升表
   （`%CUMULATIVE-RESULT-DTYPE`）。§17.10.12：这里原先是
   `(or dtype (vt-dtype input))`，同族里只漏了这一处 —— `nancumsum`
   对 bool 返回 `:bool` 且把计数 2/3 写进 `(unsigned-byte 8)`
   （脏字节），对 int8/uint8/int16 则静默回绕（`#(100 100)` → -56）。"
  ;; 整数 / 布尔输入里不存在 NaN，所以「跳过 NaN」是恒等操作，直接委托给
  ;; 无 NaN 的同一内核 —— 顺带保住整数的精确性。
  ;; 为什么必须绕开：下面无条件 `(vt-astype input :float64)`，int64 的大值
  ;; 一旦超过 2^53 就在 double 里丢低位：
  ;;   (vt-nancumsum #(2^60 1 -2^60)) → (2^60 2^60 0)
  ;;   (vt-cumsum     #(2^60 1 -2^60)) → (2^60 2^60+1 1)   ← numpy 同
  ;; 同一库内两个入口对同一个输入给出不同答案，且错的那个还是「NaN 感知」版
  ;; ——而它压根没有 NaN 可跳。
  (let ((head (ensure-vt tensor)))
    (when (or (eq (vt-dtype head) :bool) (vt-int-dtype-p (vt-dtype head)))
      (return-from %e3-cumulative-skip-nan
        (vt-cumulative head (if (eq op :sum) #'+ #'*) init-val
                       :axis axis :dtype dtype :out out))))
  (with-float-safe
   (let* ((input (ensure-vt tensor))
          (shape (vt-shape input))
          (final-dtype (%cumulative-result-dtype input dtype))
          ;; 结果形状：axis=nil 时先展平，与 %VT-CUMULATIVE 的 RES-SHAPE 同口径
          ;; （两者的 docstring 都写着「AXIS 为 nil 时展平」，此前这里没做到）。
          ;; 原实现无论如何都沿用输入形状，于是 (vt-nancumsum 2x4) 把**展平后的**
          ;; 累积值按 (2 4) 摆回去 —— 值对、形状说谎，与 vt-cumsum / numpy 分叉。
          (res-shape (if axis shape (list (vt-size input))))
          (result (if out
                     (vt-check-out out res-shape final-dtype :op-name op-name)
                     (vt-zeros res-shape :dtype final-dtype)))
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
          ;; 空数组整体跳过（与 %VT-CUMULATIVE 同一条）：ADVANCE 先跑 body
          ;; 再进位，shape 含 0 维时会拿 ptr = 0 去读长度 0 的 buffer。
          (when (plusp (vt-size result))
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
                (unless (advance) (return)))))
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
   换成「quickselect 取第 k 小」。

   v0.4.20 补漏：`:linear` 原来把 `(%select-double buf m lo)` 算了**两遍**
   （`(- hi lo)` 里一次、外面一次），一共三次 select；而选完 Lo 之后再从
   lo=0/hi=n-1 起选 Hi 必然退化（见 `%select-two-double` 的注释）。改为
   一次 `%select-two-double` 同时取两个序位：1M 实测 193 ms → 14.9 ms。"
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
      (:midpoint (multiple-value-bind (lv hv) (%select-two-double buf m lo hi)
                   (/ (+ lv hv) 2.0d0)))
      (t (multiple-value-bind (lv hv) (%select-two-double buf m lo hi)
           (+ lv (* frac (- hv lv))))))))

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
                                  for v = (%vt-ref-raw fiber i)
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
  ;; 零观测（n_obs = 0）时 numpy 给 nan 并 continue，而 SBCL 默认把
  ;; 0/0 与 1/0 作为陷阱抛出（实测 (vt-cov (vt-zeros '(0))) 报
  ;; FLOATING-POINT-INVALID-OPERATION、(vt-cov (vt-zeros '(3 0))) 报
  ;; DIVISION-BY-ZERO）。§5.5：浮点异常一律 mask，输出 nan/inf。
  (with-float-safe
    (let ((ddof (if ddof-p ddof 1)))
      (let* ((mm (%e3-as-1d-float-transform m rowvar))
             (xx (if y (%e3-as-1d-float-transform y rowvar) nil)))
        (let* ((stacked (if xx (vt-concatenate 0 mm xx) mm))
               (n-var (first (vt-shape stacked)))
               (n-obs (second (vt-shape stacked)))
               (denom (if bias n-obs (- n-obs ddof)))
               ;; numpy 的写法是 `fact = n_obs - ddof; if fact <= 0: fact = 0.0`
               ;; 然后 `c *= 1./fact` —— 也就是角分母**不做替换**，让 1/0 变成
               ;; inf，再由 0*inf 得 nan。原先这里把 denom<=0 直接换成 n_obs，
               ;; 相当于凭空改分母：np.cov([5.]) 是 nan（0/0），本库给 0.0；
               ;; np.cov([1,2], ddof=3) 是 inf（0.5/0），本库给 0.25。
               (denom (max denom 0)))
          (when (= n-var 1)
            ;; 单变量 → 标量方差
            (let ((row (vt-slice stacked '(0)))
                  (mean (vt-item (vt-mean (vt-slice stacked '(0))))))
              (return-from vt-cov
                (let ((acc 0.0d0))
                  (dotimes (j n-obs (make-vt nil (/ acc denom) :dtype :float64))
                    (incf acc (expt (- (%vt-ref-raw row j) mean) 2)))))))
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
                (vt-scale res (/ 1.0d0 denom)))))))))

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
  ;; 零观测时 vt-cov 给出 nan 矩阵，归一化的 (/ nan nan) 同样是陷阱
  ;; （§5.5）。numpy 的 corrcoef([]) 给 nan 标量。
  (with-float-safe
    (let* ((c (vt-cov m :y y :rowvar rowvar))
           (n (first (vt-shape c))))
      (if (null (vt-shape c))
          ;; 标量方差 → R = C / sqrt(C²)。不硬编码 1.0：numpy 对单变量同样
          ;; 走这条除法，而 C 可能是 0（全等输入）或 nan（单观测 / 空输入），
          ;; 那时 0/0 与 nan/nan 都该给 nan —— np.corrcoef([5.]) = nan。
          (let ((v (vt-item c)))
            (vt-full nil (/ v (sqrt (* v v))) :dtype :float64))
          (let ((r (vt-zeros (list n n) :dtype :float64)))
            ;; 性能（v0.4.21）：原循环对每个 (i,j) 调两次 `%VT-REF-RAW` 读对角线、
            ;; 一次 `(SETF (VT-REF R I J))` 写结果 —— 后者要走完整的坐标折算
            ;; （形状/步长遍历 + 越界检查）。n=1000 即 100 万次折算，实测
            ;; 148ms 而 `VT-COV` 本身只要 16ms（numpy.corrcoef 8.1ms）。
            ;;
            ;; 现在：C 与 R 都先归为连续视图（R 由 `VT-ZEROS` 建成，恒连续；
            ;; C 由 `VT-COV` 新建，`VT-CONTIGUOUS` 在已连续时是零开销原语），
            ;; 对角线抄进一个长度 n 的缓冲，内层只剩三次 AREF 与一次 SQRT。
            ;;
            ;; **逐位不变**：`(SQRT (* CII CJJ))` 的算式与求值次序一字未动
            ;; （没有改成 `(* (SQRT CII) (SQRT CJJ))` —— 那在末位不同）。
            (let* ((cc   (vt-contiguous c))
                   (cd   (vt-data cc))
                   (co   (vt-offset cc))
                   (cols (second (vt-shape c)))
                   (rd   (vt-data r))
                   (diag (make-array (max n 1) :element-type 'double-float)))
              (declare (type (simple-array double-float (*)) cd rd diag)
                       (type fixnum n cols co))
              (dotimes (i n)
                (declare (type fixnum i))
                (setf (aref diag i) (aref cd (the fixnum (+ co (* i cols) i)))))
              (dotimes (i n r)
                (declare (type fixnum i))
                (let ((cii (aref diag i))
                      (row (* i n)))
                  (declare (type double-float cii) (type fixnum row))
                  (dotimes (j n)
                    (declare (type fixnum j))
                    (let ((cjj (aref diag j)))
                      (declare (type double-float cjj))
                      (setf (aref rd (the fixnum (+ row j)))
                            (if (or (zerop cii) (zerop cjj))
                                (if (= i j) 1.0d0 0.0d0)
                                (/ (aref cd (the fixnum (+ co (* i cols) j)))
                                   (sqrt (* cii cjj)))))))))))))))

(defun vt-cross (a b &key (axis nil) (axisa -1) (axisb -1) (axisc -1) dtype)
  "向量叉积（对标 numpy.cross）。
   支持 2 分量（标量 z 分量输出）与 3 分量向量；沿 AXISA/AXISB 轴取向量，
   结果沿 AXISC 轴放置。AXIS 给定时等价 axisa=axisb=axisc=axis。
   示例：(vt-cross '(1 2 3) '(4 5 6)) => [-3 6 -3]
        (vt-cross '(1 2) '(4 5))     => -3

   布尔输入（CONVENTIONS §17.3）：**报错**，与 `vt-sub` 同源 —— numpy 的
   `np.cross(bool,bool)` 直接抛 `TypeError: numpy boolean subtract ...`，
   因为叉积定义里含减法。原实现把结果数组建在输入 dtype 上
   （`(or dtype (vt-dtype av))`），bool 输入于是得到 **:bool**，
   `-3`/`6` 这类分量被截成 1，静默输出 `(1 1 1)` —— 一个看起来合法、
   实际毫无意义的答案。"
  (declare (ignorable axisc))   ; 结果轴暂固定为末轴；AXISC 仅作 API 兼容
  (let* ((av (ensure-vt a)) (bv (ensure-vt b)))
    (when (and (null dtype) (eq (vt-dtype av) :bool))
      (error "vt-cross: 布尔张量不支持叉积（numpy 同样拒绝：布尔减法未定义）。~%~
              请先 (vt-astype a :int64) 显式升为数值类型；~%~
              只要方向（正交性）请用 vt-logical-xor / vt-bit-and。~%~
              见 CONVENTIONS §17.3"))
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
                 (let ((a0 (%vt-ref-raw a2 i 0)) (a1 (%vt-ref-raw a2 i 1))
                       (a2v (if (= al 3) (%vt-ref-raw a2 i 2) 0))
                       (b0 (%vt-ref-raw b2 i 0)) (b1 (%vt-ref-raw b2 i 1))
                       (b2v (if (= bl 3) (%vt-ref-raw b2 i 2) 0)))
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
                      (%vt-ref-raw res 0 0)                  ; 标量
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
        (vt-vdot (vt-asarray '((1 2) (3 4))) (vt-asarray '((5 6) (7 8)))) => 70
   （嵌套 list 走 VT-ASARRAY，推断 :int64；若换成 (vt-from-array #2A((1 2)(3 4)))，
   元素类型 T 会推断出 :float64，结果是 70.0 —— 见 §17.10.19 的偏差表。）"
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
                      (dotimes (i rows) (incf acc (abs (%vt-ref-raw m i j))))
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
                      (dotimes (j cols) (incf acc (abs (%vt-ref-raw m i j))))
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

(defun %array-equal-run (x y n)
  "X / Y 是同 dtype、连续、已展平的 vt，元素数 N。返回 T / NIL。

   **逐元素用 `=` 而不是 `EQL`**：numpy 的 `==` 按数值判等，
   `0.0 == -0.0` 为真、`nan == nan` 为假 —— `EQL` 正好相反（±0.0 位模式
   不同则不等），用它会与 numpy 在符号零上分叉。浮点陷阱在此屏蔽，
   否则 `NaN` 参与 `=` 会抛 `FLOATING-POINT-INVALID-OPERATION`。

   按**存储元素类型**展开成 10 份特化循环：`(aref (simple-array double-float (*)) i)`
   是一次 `movsd`，而通用 `AREF` 每元素要查 widetag。1M f64 的
   `VT-ARRAY-EQUAL` 整体实测 83 → 2 ms（§7.4.13）。"
  (let ((xd (vt-data x))
        (yd (vt-data y))
        (xo (the fixnum (vt-offset x)))
        (yo (the fixnum (vt-offset y)))
        (et (array-element-type (vt-data x))))
    (with-float-safe
      (macrolet
          ((same (lt)
             `(let ((xs (the (simple-array ,lt (*)) xd))
                    (ys (the (simple-array ,lt (*)) yd)))
                (declare (type (simple-array ,lt (*)) xs ys)
                         (type fixnum xo yo n))
                (dotimes (i n t)
                  (declare (type fixnum i))
                  (unless (= (aref xs (the fixnum (+ xo i)))
                             (aref ys (the fixnum (+ yo i))))
                    (return nil))))))
        (cond
          ((equal et 'double-float)           (same double-float))
          ((equal et 'single-float)           (same single-float))
          ((equal et '(signed-byte 64))       (same (signed-byte 64)))
          ((equal et '(signed-byte 32))       (same (signed-byte 32)))
          ((equal et '(signed-byte 16))       (same (signed-byte 16)))
          ((equal et '(signed-byte 8))        (same (signed-byte 8)))
          ((equal et '(unsigned-byte 16))     (same (unsigned-byte 16)))
          ((equal et '(unsigned-byte 8))      (same (unsigned-byte 8)))
          ((equal et '(complex double-float)) (same (complex double-float)))
          ((equal et '(complex single-float)) (same (complex single-float)))
          (t
           (dotimes (i n t)
             (declare (type fixnum i))
             (unless (= (aref xd (the fixnum (+ xo i)))
                        (aref yd (the fixnum (+ yo i))))
               (return nil)))))))))

(defun vt-array-equal (a b)
  "判定两张量形状相同且逐元素相等（对标 numpy.array_equal）。
   NaN != NaN（含 NaN 恒不相等，同 numpy）；返回 Lisp 布尔值（非张量）。
   示例：(vt-array-equal '(1 2) '(1 2)) => T
        (vt-array-equal (vt-from-array (vector (vt-get-nan :float64)))
                        (vt-from-array (vector (vt-get-nan :float64)))) => NIL

   比较域（§7.4.13）：dtype **相同**时直接在原 dtype 上比（整数不经过浮点，
   故 2^54+1 与 2^54+2 不会被 double 抹成同一个数）；不同 dtype 时按
   `VT-PROMOTE-TYPE` 提升到公共域再比 —— 与 numpy 的 `==` 提升规则同一张表。
   此前一律先 `VT-ASTYPE … :float64`，于是
     · `#C(1.0 2.0)` 与 `#C(1.0 3.0)` 判等（虚部被 astype 丢掉）；
     · int64 上超过 2^53 的两个相邻值判等（double 只有 53 位尾数）。
   两条都是静默错值，且都不报错。"
  (let* ((av (ensure-vt a))
         (bv (ensure-vt b))
         (shape-a (vt-shape av)))
    (and (equal shape-a (vt-shape bv))
         (let ((n (vt-size av)))
           (if (zerop n)
               t
               (let* ((adt (vt-dtype av))
                      (bdt (vt-dtype bv))
                      (dt (if (eq adt bdt) adt (vt-promote-type adt bdt))))
                 (%array-equal-run
                  (if (eq adt dt)
                      (vt-contiguous (vt-ravel av))
                      (vt-astype av dt))
                  (if (eq bdt dt)
                      (vt-contiguous (vt-ravel bv))
                      (vt-astype bv dt))
                  n)))))))

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
  "逐元素 +Inf 判定，返回 **:bool** 布尔张量（对标 numpy.isposinf，§17）。
   :out 契约同 vt-=（dtype :bool）。

   性能（v0.4.21）：匿名 lambda → 具名内联算子 `%pred-posinf`，
   1M float64 实测 26.5ms → 2.6ms（`vt-isnan` 同一形状只要 1.7ms，
   差额是 `ensure-vt :dtype :float64` 的 dtype 检查本身）。"
  (vt-fast-map #'%pred-posinf (ensure-vt tensor :dtype :float64)
               :out out :dtype :bool))

(defun vt-isneginf (tensor &key out)
  "逐元素 -Inf 判定，返回 **:bool** 布尔张量（对标 numpy.isneginf，§17）。
   :out 契约同 vt-=（dtype :bool）。性能同 `vt-isposinf`。"
  (vt-fast-map #'%pred-neginf (ensure-vt tensor :dtype :float64)
               :out out :dtype :bool))

;;; ==================================================================
;;; 8. 集合类（Set operations）
;;; ==================================================================

(defun vt-isin (element test-elements &key (invert nil) (assume-unique nil))
  "逐元素判定是否存在于 TEST-ELEMENTS 中（对标 numpy.isin）。
   返回与 ELEMENT 形状相同的 **:bool** 布尔张量（§17）。
   INVERT=t 取反；ASSUME-UNIQUE 为性能提示（本库忽略，语义等价）。
   示例：(vt-isin '(1 2 3 4) '(2 4)) => [NIL T NIL T]（布尔印 T/NIL，见 §17.8）"
  ;; v0.4.2 性能修复：与 vt-in1d 共用哈希会员判定（原 O(n×m) 线性 member，
  ;; 1M×1k 实测 87 秒）。
  (when (%cx-any-p element test-elements)
    (return-from vt-isin (%cx-op-isin element test-elements :invert invert :assume-unique assume-unique)))
  (let* ((elem (ensure-vt element))
         (test (ensure-vt test-elements))
         (fm (%membership-float-mode-p elem test)))
    (%membership-map elem (%membership-table (vt-unique test) fm) fm invert)))
