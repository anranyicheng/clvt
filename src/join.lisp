;;;; join.lisp — 连接、堆叠、追加、插入、删除
;;;;
;;;; 约定基准：CONVENTIONS.md（唯一语义契约来源）。
;;;;   - 铁律 1/2：连续性只影响选路；快路径失败必须回退通用路径，
;;;;     且快路径与通用路径结果逐位一致。
;;;;   - 本文件全部为纯数据搬运（不执行浮点运算），两条路径天然逐位等价；
;;;;     需要归一 dtype 时复用 vt-astype（其 cast 语义与 vt-copy-into
;;;;     的 vt-cast-fun 一致，见 dtype.lisp）。
;;;;
;;;; 性能设计：
;;;;   1. vt-concatenate：结果缓冲一次分配（各输入块沿 axis 恰好平铺满
;;;;      [0, total)，全部元素必被写入，故无需清零预热）；连续输入按
;;;;      (outer, axis-size, inner) 三段视图直接 replace 块拷贝，
;;;;      非连续输入回退 slice 视图 + vt-copy-into 通用路径。
;;;;   2. vt-insert：flat 模式按「升序稳定排序 + 单遍装配」直接写入
;;;;      结果缓冲 O(n + k·v)，替代旧实现逐索引 concatenate 的
;;;;      O(k·n) 与中间副本；axis 模式把行切片与插入块单遍合并成
;;;;      最终切片列表 O(n + k)，替代逐次 append/subseq 的 O(k·n)。
;;;;   3. vt-delete：flat 模式按保留 run 顺序拷贝（单遍，替代
;;;;      vt-take 的索引 gather）；axis 模式把保留行折叠为连续 run，
;;;;      连续输入直接块拷贝，非连续输入以合并 run 的 narrow 视图
;;;;      走 concatenate（视图数从 O(dim) 降到 O(run)）。

(in-package :clvt)

;;; ------------------------------------------------------------------
;;; 内部原语：连续块拷贝（本文件快路径共用）
;;; ------------------------------------------------------------------

(declaim (inline %axis-outer-inner %block-copy))

(defun %block-copy (dest-data src-data dst-start src-start length)
  "同元素类型一维缓冲区间的块拷贝（等价 memcpy；length 为 0 时 no-op）。
   实现委托给 `clvt::%memcpy`（core 里的单一份按元素类型展开的表，
   含 complex128/complex64 两格）—— 本函数只保留本文件惯用的实参顺序。"
  (declare (type fixnum dst-start src-start length))
  (%memcpy dest-data dst-start src-data src-start length)
  (values))

(defun %axis-outer-inner (shape axis)
  "把 shape 沿 axis 拆为 (outer, axis-size, inner) 三段视图，返回 (values outer inner)。
   C 连续张量沿该轴的存储可组织为 outer 段长 axis-size×inner 的连续 run。"
  (declare (type fixnum axis))
  (values (the fixnum (reduce #'* (subseq shape 0 axis) :initial-value 1))
          (the fixnum (reduce #'* (subseq shape (1+ axis)) :initial-value 1))))

(defun %copy-contiguous-blocks (dest-data source axis-pos axis-size outer inner dst-stride)
  "快路径：把连续张量 source 的数据块拷入连续 dest 缓冲。
   source 沿连接轴可视为 (outer, axis-size, inner)；对每个 outer 下标，
   源与目标各是一段长 axis-size×inner 的连续 run（铁律 1：布局不同不改值）。

   两条子路径：
     · run-length 较大 → 每段一次 %BLOCK-COPY（memmove），调用开销被摊薄；
     · run-length 很小 → **逐元素内联**（8 路类型）。这不是微优化：
       vt-stack 沿新轴拼接 1M 个长度 1 的 run（1M 次 REPLACE 调用），
       实测 55.9ms；内联后同工作量降到个位数毫秒。"
  (let ((src-data (vt-data source))
        (src-offset (vt-offset source))
        (run-length (* axis-size inner)))
    (declare (type fixnum run-length dst-stride src-offset))
    (if (<= run-length 8)
        (macrolet ((sc (lt)
                     `(let ((dd (the (simple-array ,lt (*)) dest-data))
                            (ss (the (simple-array ,lt (*)) src-data)))
                        (declare (type (simple-array ,lt (*)) dd ss)
                                 (type fixnum run-length dst-stride src-offset
                                       axis-pos inner outer))
                        (dotimes (o outer)
                          (declare (type fixnum o))
                          (let ((di (the fixnum (+ (* o dst-stride) (* axis-pos inner))))
                                (si (the fixnum (+ src-offset (* o run-length)))))
                            (declare (type fixnum di si))
                            (dotimes (k run-length)
                              (declare (type fixnum k))
                              (setf (aref dd (the fixnum (+ di k)))
                                    (aref ss (the fixnum (+ si k))))))))))
          (let ((et (array-element-type dest-data)))
            (cond ((equal et 'double-float) (sc double-float))
                  ((equal et 'single-float) (sc single-float))
                  ((equal et '(signed-byte 64)) (sc (signed-byte 64)))
                  ((equal et '(signed-byte 32)) (sc (signed-byte 32)))
                  ((equal et '(signed-byte 16)) (sc (signed-byte 16)))
                  ((equal et '(signed-byte 8)) (sc (signed-byte 8)))
                  ((equal et '(unsigned-byte 16)) (sc (unsigned-byte 16)))
                  ((equal et '(unsigned-byte 8)) (sc (unsigned-byte 8)))
                  (t (dotimes (o outer)
                       (%block-copy dest-data src-data
                                    (+ (* o dst-stride) (* axis-pos inner))
                                    (+ src-offset (* o run-length))
                                    run-length))))))
        (dotimes (o outer)
          (%block-copy dest-data src-data
                       (+ (* o dst-stride) (* axis-pos inner))
                       (+ src-offset (* o run-length))
                       run-length)))))

;;; ------------------------------------------------------------------
;;; 连接
;;; ------------------------------------------------------------------

(defun vt-concatenate (axis &rest tensors)
  "沿指定轴连接多个张量（对标 numpy.concatenate / torch.cat）。

注意：axis 为首个位置参数（clvt 历史契约，测试与示例依赖此顺序），
支持负轴。各张量秩必须相同，非 axis 轴的维度必须严格一致。

结果 dtype 为各输入 dtype 的公共提升（§3.2 提升表）；
结果沿 axis 的长度 = 各输入该轴长度之和。

NaN/Inf：纯数据搬运，逐位保留输入位模式。

性能：dtype 归一后仍连续的输入走块拷贝快路径，否则逐输入回退
stride 驱动的通用路径（slice 视图 + vt-copy-into），两者逐位一致。

示例：(vt-concatenate 0 (vt-from-sequence '(1 2)) (vt-from-sequence '(3 4)))"
  (when (null tensors)
    (error "vt-concatenate: tensors 至少需要一个张量，收到 0 个"))
  (let* ((shapes (mapcar #'vt-shape tensors))
         (first-shape (car shapes))
         (rank (length first-shape))
         (ax (vt-normalize-axis axis rank)))
    ;; 逻辑层校验：秩一致 + 非 axis 轴维度一致（确定性报错，L2）
    (loop for shape in shapes
          for i from 0
          do (unless (= (length shape) rank)
               (error "vt-concatenate: 张量 ~a 秩 ~a 与首个张量秩 ~a 不匹配"
                      i (length shape) rank))
             (loop for d in shape
                   for j from 0
                   unless (or (= j ax) (= d (nth j first-shape)))
                     do (error "vt-concatenate: 张量 ~a 沿轴 ~a 的维度 ~a 与首个张量的 ~a 不匹配"
                               i j d (nth j first-shape))))
    (let* ((new-shape (copy-list first-shape))
           (total (reduce #'+ (mapcar (lambda (s) (nth ax s)) shapes)))
           (result-dtype (apply #'vt-promote-type (mapcar #'vt-dtype tensors))))
      ;; 先把连接轴长度更新为 total，再按新形状分配结果缓冲
      (setf (nth ax new-shape) total)
      (let* (;; 结果缓冲一次分配且不清零：各输入块沿 axis 平铺 [0, total)，
             ;; 两条路径都会写满全部元素（未写满即 bug，与清零版语义一致）。
             ;; 缓冲长度 = 结果完整元素数（total × inner × outer），非轴长 total。
             (result-data (make-array (vt-shape-to-size new-shape)
                                      :element-type (vt-dtype->lisp-type result-dtype)))
             (result (%make-vt :data result-data :shape new-shape
                               :strides (vt-compute-strides new-shape)
                               :offset 0 :dtype result-dtype))
             (dst-stride (* total (nth-value 1 (%axis-outer-inner first-shape ax))))
             (cursor 0))
        (declare (type fixnum cursor dst-stride))
        (dolist (tensor tensors)
          (let* ((axis-size (nth ax (vt-shape tensor)))
                 ;; dtype 归一：与结果不同时先走 vt-astype 特化转换
                 ;; （astype 输出连续，语义与通用路径的逐元素 cast 一致）
                 (source (if (eq (vt-dtype tensor) result-dtype)
                             tensor
                             (vt-astype tensor result-dtype))))
            (if (vt-contiguous-p source)
                ;; ---- 执行层快路径：连续块拷贝 ----
                (multiple-value-bind (outer inner)
                    (%axis-outer-inner (vt-shape source) ax)
                  (%copy-contiguous-blocks result-data source
                                           cursor axis-size outer inner dst-stride))
                ;; ---- 执行层通用路径：stride 驱动 ----
                (let ((slice-specs (loop for i below rank
                                         collect (if (= i ax)
                                                     (list cursor (+ cursor axis-size))
                                                     '(:all)))))
                  (setf (apply #'vt-slice result slice-specs) source)))
            (incf cursor axis-size)))
        result))))

(defun vt-concat (axis &rest tensors)
  "沿既有轴连接多个张量（对标 numpy.concatenate / torch.cat）。
vt-concatenate 的别名，签名与语义完全一致。"
  (apply #'vt-concatenate axis tensors))

;;; ------------------------------------------------------------------
;;; 堆叠
;;; ------------------------------------------------------------------

(defun vt-stack (axis &rest tensors)
  "沿新轴堆叠（对标 numpy.stack / torch.stack）。

所有张量形状必须完全一致；结果在 axis 位置插入长度为
len(tensors) 的新轴。axis 支持负值（-1 表示插在末尾之后）。

结果 dtype 为各输入 dtype 的公共提升（与 vt-concatenate 一致）。

示例：(vt-stack 0 (vt-from-sequence '(1 2)) (vt-from-sequence '(3 4)))"
  (when (null tensors)
    (error "vt-stack: tensors 至少需要一个张量，收到 0 个"))
  (let ((base-shape (vt-shape (car tensors))))
    (loop for tensor in tensors
          unless (equal (vt-shape tensor) base-shape)
            do (error "vt-stack: 张量形状 ~a 与首个张量形状 ~a 不匹配"
                      (vt-shape tensor) base-shape))
    (apply #'vt-concatenate axis
           (mapcar (lambda (tensor) (vt-expand-dims tensor axis)) tensors))))

(defun vt-vstack (&rest tensors)
  "垂直堆叠（对标 numpy.vstack / torch.vstack）。

1-D 输入提升为 (1, n)，0-d 提升为 (1, 1)，≥2-D 原样，再沿轴 0 连接。
结果 dtype 为各输入 dtype 的公共提升。

示例：(vt-vstack (vt-from-sequence '(1 1)) (vt-from-sequence '(2 2)))"
  (when (null tensors)
    (error "vt-vstack: tensors 至少需要一个张量，收到 0 个"))
  (apply #'vt-concatenate 0
         (mapcar (lambda (tensor)
                   (let ((rank (length (vt-shape tensor))))
                     (cond ((= rank 0) (vt-reshape tensor '(1 1)))
                           ((= rank 1) (vt-reshape tensor
                                                   (list 1 (first (vt-shape tensor)))))
                           (t tensor))))
                 tensors)))

(defun vt-hstack (&rest tensors)
  "水平堆叠（对标 numpy.hstack）。

1-D 输入沿轴 0 拼接，≥2-D 输入沿轴 1 拼接。
结果 dtype 为各输入 dtype 的公共提升。

示例：(vt-hstack (vt-from-sequence '(1 1)) (vt-from-sequence '(2 2)))"
  (when (null tensors)
    (error "vt-hstack: tensors 至少需要一个张量，收到 0 个"))
  (if (= (length (vt-shape (car tensors))) 1)
      (apply #'vt-concatenate 0 tensors)
      (apply #'vt-concatenate 1 tensors)))

(defun vt-dstack (&rest tensors)
  "深度堆叠（对标 numpy.dstack / torch.dstack）。

0-d 视为 (1, 1, 1)，1-D 视为 (1, n, 1)，2-D 视为 (m, n, 1)，≥3-D 原样，
再沿轴 2 连接。结果 dtype 为各输入 dtype 的公共提升。

示例：(vt-dstack (vt-from-sequence '(1 1)) (vt-from-sequence '(2 2)))"
  (when (null tensors)
    (error "vt-dstack: tensors 至少需要一个张量，收到 0 个"))
  (apply #'vt-concatenate 2
         (mapcar (lambda (tensor)
                   (let ((rank (length (vt-shape tensor))))
                     (cond ((= rank 0) (vt-reshape tensor '(1 1 1)))
                           ((= rank 1) (vt-reshape tensor
                                                   (list 1 (first (vt-shape tensor)) 1)))
                           ((= rank 2) (vt-reshape tensor (append (vt-shape tensor) '(1))))
                           (t tensor))))
                 tensors)))

;;; ------------------------------------------------------------------
;;; 追加
;;; ------------------------------------------------------------------

(defun vt-append (tensor values &key (axis nil))
  "将 values 追加到 tensor 末尾（对标 numpy.append）。

axis 为 NIL：两数组展平为一维后沿轴 0 连接，结果一维；
axis 给定：两者秩必须一致，且除 axis 外各维度必须相等，
结果沿 axis 的长度为两者之和。

结果 dtype 为两者 dtype 的公共提升（与 vt-concatenate 一致）。
纯数据搬运，NaN 逐位保留。

示例：(vt-append (vt-from-sequence '(1 2)) (vt-from-sequence '(3 4)))"
  (let ((tensor-vt (ensure-vt tensor))
        (values-vt (ensure-vt values)))
    (if (null axis)
        (vt-concatenate 0 (vt-ravel tensor-vt) (vt-ravel values-vt))
        (let* ((rank (length (vt-shape tensor-vt)))
               (ax (vt-normalize-axis axis rank)))
          (unless (= (length (vt-shape tensor-vt)) (length (vt-shape values-vt)))
            (error "vt-append: tensor 秩 ~a 与 values 秩 ~a 不匹配"
                   (length (vt-shape tensor-vt)) (length (vt-shape values-vt))))
          (loop for i below rank
                unless (or (= i ax)
                           (= (nth i (vt-shape tensor-vt))
                              (nth i (vt-shape values-vt))))
                  do (error "vt-append: 轴 ~a 维度 tensor=~a values=~a 不匹配"
                            i (nth i (vt-shape tensor-vt))
                            (nth i (vt-shape values-vt))))
          (vt-concatenate ax tensor-vt values-vt)))))

;;; ------------------------------------------------------------------
;;; 插入
;;; ------------------------------------------------------------------

(defun %bool-mask-positions (mask limit caller)
  "把 `:bool` 张量按 numpy 的**掩码**语义转成位置列表（升序）。

  numpy 对「位置参数」（positions / obj）有两套并存语义，按 dtype 分叉
  （CONVENTIONS §17.10.14）：

    · 掩码：`delete` / `insert` / `place` —— 布尔数组逐元素表示
      「这一位要不要动」，长度必须等于被索引轴的长度；
    · 索引：`take` / `put` / `choose` / `bincount` —— 布尔数组被当作
      0/1 整数索引，长度任意。

  这是 numpy 里极少数**按 dtype 而非值**分叉的地方：同样的 `#(1 0 1 1)`，
  `:uint8` 走索引、`:bool` 走掩码。本库原先两处都走索引，导致移植代码
  `np.delete(a, a > 3)` 落到索引语义上**静默删错元素**（`(0 1 2 3)` 配
  `(T NIL T T)` 得 `(2 3)`，numpy 得 `(1)`）。

  MASK 必须一维且长度等于 LIMIT（numpy 同样限制，消息也照抄其措辞）。"
  (unless (= 1 (length (vt-shape mask)))
    (error "~a: 布尔位置参数必须是一维且长度与被索引轴一致，收到形状 ~a"
           caller (vt-shape mask)))
  (let ((n (vt-size mask)))
    (unless (= n limit)
      (error "~a: 布尔位置参数长度 ~a 与被索引轴长度 ~a 不一致"
             caller n limit))
    (let* ((flat (vt-flatten mask))
           (data (vt-data flat))
           (off (vt-offset flat))
           (acc '()))
      ;; 倒序 collect 再 push，结果天然升序，%kept-runs-for 与
      ;; %insert-flat 都要求按位置升序处理。
      (loop for i downfrom (1- n) to 0
            when (/= 0 (aref data (+ off i)))
              do (push i acc))
      acc)))

(declaim (inline %insert-norm-pos))

(defun %insert-norm-pos (raw size)
  "insert 位置归一化：负索引加轴长，越界确定性报错（numpy 语义，
位置允许等于轴长，即插入到末尾）。"
  (let ((pos (if (minusp raw) (+ size raw) raw)))
    (when (or (< pos 0) (> pos size))
      (error "vt-insert: 索引 ~a 越界（轴大小 ~a）" raw size))
    pos))

(defun %insert-flat (flat flat-val insertions dtype)
  "flat 模式插入：positions 已归一化为列表（未排序）。
   flat / flat-val 均为一维连续张量；结果 dtype = dtype（values 已 coerce）。
   numpy 语义：位置升序稳定排序后依次插入，同位置多个值按 values 顺序排布。
   实现采用单遍装配：对排序后的插入计划，从左到右拷贝源区间与插入块，
   O(n + k·v)，无中间副本。"
  (let* ((size (vt-size flat))
         (val-size (vt-size flat-val))
         (num (length insertions)))
    (unless (or (= val-size num) (= val-size 1))
      (error "vt-insert: values 元素数 ~a 与索引数 ~a 不一致" val-size num))
    ;; 稳定排序保证同位置值按 values 原顺序排布
    (let* ((pairs (stable-sort
                   (if (= val-size 1)
                       (loop for raw-pos in insertions
                             collect (cons (%insert-norm-pos raw-pos size)
                                           flat-val))
                       (loop for raw-pos in insertions
                             for i from 0 below val-size
                             collect (cons (%insert-norm-pos raw-pos size)
                                           (vt-narrow flat-val 0 i (1+ i)))))
                   #'< :key #'car))
           (result-size (+ size (length pairs)))
           (result-data (make-array result-size
                                    :element-type (vt-dtype->lisp-type dtype)))
           (src-data (vt-data flat))
           (src-offset (vt-offset flat))
           (read-pos 0)
           (write-pos 0))
      (declare (type fixnum read-pos write-pos))
      (dolist (pair pairs)
        (let* ((pos (car pair))
               (block-vt (cdr pair))
               (block-size (vt-size block-vt))
               (run (- pos read-pos)))
          ;; 源区间 [read-pos, pos)
          (%block-copy result-data src-data
                       write-pos (+ src-offset read-pos) run)
          (incf write-pos run)
          ;; 插入块（flat-val 的连续窄视图或单元素张量本身）
          (%block-copy result-data (vt-data block-vt) write-pos
                       (vt-offset block-vt) block-size)
          (incf write-pos block-size)
          (setf read-pos pos)))
      ;; 末段源区间
      (%block-copy result-data src-data
                   write-pos (+ src-offset read-pos) (- size read-pos))
      (%make-vt :data result-data :shape (list result-size)
                :strides (list 1) :offset 0 :dtype dtype))))

(defun %shape-broadcastable-to-p (from to)
  "FROM 形状能否按 numpy 规则广播到 TO（右对齐，逐维相等或 FROM 维为 1）。"
  (let ((nf (length from)) (nt (length to)))
    (and (<= nf nt)
         (let ((pad (append (make-list (- nt nf) :initial-element 1) from)))
           (every (lambda (a b) (or (= a 1) (= a b))) pad to)))))

(defun %insert-axis (tensor-vt insertions values-vt axis scalar-pos)
  "axis 模式插入：沿 axis 在指定位置插入 values 块。

   完整复刻 numpy 2.x `insert` 的 axis 路径：
     values = array(values, ndmin=arr.ndim, dtype=arr.dtype)
     if <obj 为标量>:  values = np.moveaxis(values, 0, axis)
     numnew = values.shape[axis]
     new[..., index:index+numnew, ...] = values        ; 广播写入
   多索引 obj 路径不做 ndmin / 不做 moveaxis，直接把 values 广播进块。

   此前实现用 vt-reshape（行主序重排）顶替 moveaxis(values, 0, axis)。
   两者仅当 values 首轴长度为 1 时等价；values 含多行时形状校验只看
   「元素总数」，恰在元素总数相同时通过，从而**静默**产出与 numpy 不同的
   结果（实测：a(1000,1000)、obj=7、axis=1、values=(2,1000)，
   首处差异 (0,8)：clvt 1.0 vs numpy 1000.0）。"
  (let* ((shape (vt-shape tensor-vt))
         (rank (length shape))
         (ax (vt-normalize-axis axis rank))
         (ax-size (nth ax shape))
         (num (length insertions))
         (values-nd (vt-shape values-vt))
         ;; (1) ndmin：左侧补 1 轴到 rank（numpy: array(values, ndmin=arr.ndim)）
         (ranked (if (and values-nd (< (length values-nd) rank))
                     (vt-reshape values-vt
                                 (append (make-list (- rank (length values-nd))
                                                    :initial-element 1)
                                         values-nd))
                     values-vt))
         ;; (2) 单索引：标量 obj 需 moveaxis(values, 0, axis)
         (prepared (if (= num 1)
                       (let ((base (if values-nd
                                       ranked
                                       (vt-reshape values-vt
                                                   (make-list rank
                                                              :initial-element 1)))))
                         (if scalar-pos (vt-moveaxis base 0 ax) base))
                       ranked))
         ;; (3) numnew = prepared.shape[ax]
         (numnew (if (= num 1) (nth ax (vt-shape prepared)) num))
         (target-shape
           (loop for i below rank
                 collect (if (= i ax) numnew (nth i shape))))
         ;; (4) 广播 prepared 到插入块形状（numpy: new[..., sl(ax), ...] = values）
         (block (cond ((equal (vt-shape prepared) target-shape) prepared)
                      ((%shape-broadcastable-to-p (vt-shape prepared) target-shape)
                       (vt-broadcast-to prepared target-shape))
                      (t (error "vt-insert: values 形状 ~a 无法广播到插入块形状 ~a（numpy 广播语义）"
                                values-nd target-shape))))
         (row-slices
           (loop for i from 0 below ax-size
                 collect (vt-narrow tensor-vt ax i (1+ i))))
         (value-blocks
           (if (= num 1)
               (list block)
               (loop for i from 0 below num
                     collect (vt-narrow block ax i (1+ i)))))
         ;; 先归一化再稳定排序（同 flat 模式，numpy 语义）
         (pairs (stable-sort (loop for raw-pos in insertions
                                   for block in value-blocks
                                   collect (cons (%insert-norm-pos raw-pos ax-size)
                                                 block))
                             #'< :key #'car)))
    ;; 单遍合并：行切片与插入块按最终顺序一次成型
    (let ((final-slices '())
          (pending pairs))
      (loop for row below ax-size
            do (loop while (and pending (= (car (first pending)) row))
                     do (push (cdr (first pending)) final-slices)
                        (pop pending))
               (push (nth row row-slices) final-slices))
      ;; 剩余插入点位置均为 ax-size（插入到末尾）
      (dolist (p pending)
        (push (cdr p) final-slices))
      (apply #'vt-concatenate ax (nreverse final-slices)))))

(defun vt-insert (tensor positions values &key (axis nil))
  "对标 numpy.insert 在指定位置插入值。

positions：整数 / 整数列表 / 整数张量。所有位置先按当前轴尺寸
归一化（负索引）并做越界检查（位置允许等于轴长，即插入到末尾）；
多位置按升序稳定排序后依次插入，同位置多个值按 values 顺序排布。
标量 positions 在 axis 为 NIL 时按 values 元素数广播为重复位置
（np.insert(a,1,[7,8]) ≡ positions=[1,1]）。
非空浮点 positions 报错（对标 numpy TypeError/IndexError）；**空**
positions 合法，等价于原样返回（np.insert(a, [], v) 亦如此），
见 `%check-integer-index`。

axis 给定时严格复刻 numpy 2.x：
  - 单索引（obj 为标量或 [k]）：values 先经
    `array(values, ndmin=tensor.ndim)` 左侧补 1 轴；若 obj 为**标量**，
    再对 values 做 `moveaxis(0 → axis)`（这正是
    `np.insert(a,1,v)` 与 `np.insert(a,[1],v)` 结果不同的原因）。
    块长 numnew = values.shape[axis]，随后把 values 广播写入该块。
  - 多索引：不做 ndmin / moveaxis，values 直接广播进 (…, num, …) 块。

axis 为 NIL：展平成一维后插入。

结果 dtype 与 tensor 一致（values 先 coerce 到 tensor dtype）。
NaN/Inf：纯数据搬运，逐位保留。

示例：(vt-insert (vt-from-sequence '(0 1 2)) 1 (vt-from-sequence '(10 20)))"
  (let* ((tensor-vt (ensure-vt tensor))
         (values-vt (ensure-vt values :dtype (vt-dtype tensor-vt)))
         ;; positions 是否为「标量 obj」：numpy 只对 obj.ndim==0 做
         ;; moveaxis(values, 0, axis)，对 obj=[1] 不做，二者结果不同。
         (scalar-pos (cond ((integerp positions) t)
                           ((vt-p positions) (null (vt-shape positions)))
                           (t nil)))
         (insertions
           (cond ((and (vt-p positions) (eq (vt-dtype positions) :bool))
                  ;; §17.10.14：布尔位置参数是掩码，不是 0/1 索引
                  (%bool-mask-positions positions
                                        (if axis
                                            (nth (vt-normalize-axis
                                                  axis (length (vt-shape tensor-vt)))
                                                 (vt-shape tensor-vt))
                                            (vt-size tensor-vt))
                                        "vt-insert"))
                 ((vt-p positions)
                  (%check-integer-index "vt-insert 的 positions" positions)
                  (let ((flat (vt-flatten positions)))
                    (loop for i below (vt-size flat)
                          collect (truncate (aref (vt-data flat) i)))))
                 ((integerp positions)
                  (let ((n (if axis
                               1
                               (vt-size (vt-flatten values-vt)))))
                    (if (> n 1)
                        (make-list n :initial-element positions)
                        (list positions))))
                 ((listp positions)
                  (unless (every #'integerp positions)
                    (error "vt-insert: positions 列表必须全为整数，收到 ~a" positions))
                  positions)
                 (t (error "vt-insert: positions 类型无效，收到 ~a" positions)))))
    (if (null axis)
        (%insert-flat (vt-ravel tensor-vt) (vt-ravel values-vt)
                      insertions (vt-dtype tensor-vt))
        (%insert-axis tensor-vt insertions values-vt axis scalar-pos))))

;;; ------------------------------------------------------------------
;;; 删除
;;; ------------------------------------------------------------------

(declaim (inline %delete-norm-idx %delete-norm-end))

(defun %delete-norm-idx (idx dim)
  "delete 行索引归一化：负索引加轴长，越界确定性报错（0 ≤ idx < dim）。"
  (let ((n (if (minusp idx) (+ idx dim) idx)))
    (unless (and (>= n 0) (< n dim))
      (error "vt-delete: 索引 ~a 越界（轴大小 ~a）" idx dim))
    n))

(defun %delete-norm-end (idx dim)
  "delete 切片端点归一化：允许 0 ≤ end ≤ dim（闭区间端点）。"
  (let ((n (if (minusp idx) (+ idx dim) idx)))
    (unless (and (>= n 0) (<= n dim))
      (error "vt-delete: 切片端点 ~a 越界（轴大小 ~a）" idx dim))
    n))

(defun %deleted-flags (positions dim)
  "把 positions（整数列表或 (:slice start end) 内部形式）编译为
长度 dim 的位向量：1 = 删除。重复位置幂等。"
  (let ((flags (make-array dim :element-type 'bit :initial-element 0)))
    (if (and (listp positions) (eq (first positions) :slice))
        (destructuring-bind (_ s e) positions
          (declare (ignore _))
          (let ((ss (%delete-norm-end s dim))
                (ee (%delete-norm-end e dim)))
            (loop for i from ss below ee do (setf (bit flags i) 1))))
        (dolist (idx positions)
          (setf (bit flags (%delete-norm-idx idx dim)) 1)))
    flags))

(defun %kept-runs (flags dim)
  "把位向量折叠为保留区间的连续 run 列表 ((start end) ...)，end 排他，
升序返回。将 O(dim) 个行视图压缩为 O(run) 个，减少 concatenate 视图数。

  v0.4.6：补 FLAGS / DIM 的类型声明。原来两者都没声明，(bit flags i) 走
  通用 AREF（二维分派 + 边界检查），1M 元素实测 8354us —— 而整件事只是
  扫一遍位向量。"
  (declare (type (simple-array bit (*)) flags) (type fixnum dim))
  (let ((runs '())
        (run-start nil))
    (dotimes (i dim)
      (declare (type fixnum i))
      (if (zerop (sbit flags i))
          (unless run-start (setf run-start i))
          (when run-start
            (push (cons (the fixnum run-start) i) runs)
            (setf run-start nil))))
    (when run-start (push (cons (the fixnum run-start) dim) runs))
    (nreverse runs)))

(defun %kept-runs-from-positions (positions dim)
  "由**已归一化**的删除位置表直接生成保留 run（升序、end 排他）。
  复杂度 O(k log k)（k = 删除个数），不需要扫全数组。
  重复位置幂等（与位向量路径等价）。"
  (declare (type fixnum dim))
  (let ((sorted (sort (copy-list positions) #'<))
        (runs '())
        (cur 0))
    (declare (type fixnum cur))
    (dolist (p sorted)
      (let ((p (the fixnum p)))
        (when (> p cur)
          (push (cons cur p) runs))
        (when (>= p cur)
          (setf cur (the fixnum (1+ p))))))
    (when (< cur dim) (push (cons cur dim) runs))
    (nreverse runs)))

(defun %kept-runs-for (positions dim)
  "把「删除位置表」折叠为保留 run。K << DIM 时走排序位置表 O(k log k)，
  否则走位向量 O(dim)。

  v0.4.6 性能修复：原实现**无论删几个元素**都建 len=DIM 的位向量并扫一遍。
  (vt-delete 1M-元素 500000) 实测 17550us（numpy.delete 600us，差 29 倍），
  其中 8354us 全在扫位向量、15us 在建位向量 —— 删除位置越少，浪费越大。"
  (declare (type fixnum dim))
  (cond
    ((and (listp positions)
          (not (eq (first positions) :slice))
          (<= (length positions) (max 32 (ash dim -3))))
     (%kept-runs-from-positions
      (mapcar (lambda (i) (%delete-norm-idx i dim)) positions)
      dim))
    (t (%kept-runs (%deleted-flags positions dim) dim))))

(defun %delete-axis-copy (source ax runs kept-count)
  "快路径：C 连续张量沿 ax 删除后保留行的直接块拷贝。
   连续张量沿 ax 可视为 (outer, dim, inner)；每个保留 run 在每个
   outer 块内对应一段 (end-start)×inner 的连续 run。"
  (let* ((shape (vt-shape source))
         (dim (nth ax shape)))
    (multiple-value-bind (outer inner)
        (%axis-outer-inner shape ax)
      (let* ((result-size (* kept-count inner outer))
             (result-data (make-array result-size
                                      :element-type (vt-element-type source)))
             (src-data (vt-data source))
             (src-offset (vt-offset source))
             (src-outer-stride (* dim inner))
             (dst-outer-stride (* kept-count inner))
             (dst-run-base 0))
        (declare (type fixnum dst-run-base))
        (dolist (run runs)
          (let* ((len (* (- (cdr run) (car run)) inner))
                 (src-run-base (+ src-offset (* (car run) inner))))
            (dotimes (o outer)
              (%block-copy result-data src-data
                           (+ (* o dst-outer-stride) dst-run-base)
                           (+ src-run-base (* o src-outer-stride))
                           len)))
          (incf dst-run-base (* (- (cdr run) (car run)) inner)))
        (let ((new-shape (copy-list shape)))
          (setf (nth ax new-shape) kept-count)
          (%make-vt :data result-data :shape new-shape
                    :strides (vt-compute-strides new-shape)
                    :offset 0 :dtype (vt-dtype source)))))))

(defun vt-delete (tensor positions &key (axis nil))
  "对标 numpy.delete 删除元素。

positions：整数 / 整数列表 / 整数张量 / (:slice start end)（内部形式，
删除区间 [start, end)）。所有索引先按当前轴尺寸归一化（负索引）并
越界检查；重复位置幂等。

axis 为 NIL：先展平成一维再删除（对标 numpy 扁平语义）；
axis 给定：沿该轴删除对应行，其余维度不变。

NaN/Inf：纯数据搬运，逐位保留。

示例：(vt-delete (vt-from-sequence '(0 1 2)) 1)"
  (let* ((tensor-vt (ensure-vt tensor))
         (shape (vt-shape tensor-vt))
         (rank (length shape))
         (dtype (vt-dtype tensor-vt))
         (positions-list
           (cond ((and (vt-p positions) (eq (vt-dtype positions) :bool))
                  ;; §17.10.14：布尔位置参数是掩码，不是 0/1 索引
                  (%bool-mask-positions positions
                                        (if axis
                                            (nth (vt-normalize-axis axis rank) shape)
                                            (vt-size tensor-vt))
                                        "vt-delete"))
                 ((vt-p positions)
                  (let ((flat (vt-flatten positions)))
                    (loop for i below (vt-size flat)
                          collect (truncate (aref (vt-data flat) i)))))
                 ((listp positions) positions)
                 (t (list positions)))))
    (if (null axis)
        ;; ---- flat 模式：ravel 保证一维连续 → 保留 run 单遍拷贝 ----
        (let* ((flat (vt-ravel tensor-vt))
               (flat-size (vt-size flat))
               (runs (%kept-runs-for positions-list flat-size)))
          (if (null runs)
              (vt-zeros '(0) :dtype dtype)
              (let* ((kept-count (loop for run in runs sum (- (cdr run) (car run))))
                     (result-data (make-array kept-count
                                              :element-type (vt-element-type tensor-vt)))
                     (src-data (vt-data flat))
                     (src-offset (vt-offset flat))
                     (write-pos 0))
                (declare (type fixnum write-pos))
                (dolist (run runs)
                  (let ((len (- (cdr run) (car run))))
                    (%block-copy result-data src-data
                                 write-pos (+ src-offset (car run)) len)
                    (incf write-pos len)))
                (%make-vt :data result-data :shape (list kept-count)
                          :strides (list 1) :offset 0 :dtype dtype))))
        ;; ---- axis 模式：保留行折叠为 run ----
        (let* ((ax (vt-normalize-axis axis rank))
               (dim (nth ax shape))
               (runs (%kept-runs-for positions-list dim))
               (kept-count (loop for run in runs sum (- (cdr run) (car run)))))
          (if (null runs)
              (let ((empty-shape (copy-list shape)))
                (setf (nth ax empty-shape) 0)
                (vt-zeros empty-shape :dtype dtype))
              (if (vt-contiguous-p tensor-vt)
                  ;; 快路径：连续输入 → 直接块拷贝
                  (%delete-axis-copy tensor-vt ax runs kept-count)
                  ;; 通用路径：合并 run 的 narrow 视图走 concatenate
                  (apply #'vt-concatenate ax
                         (loop for run in runs
                               collect (vt-narrow tensor-vt ax (car run) (cdr run))))))))))
