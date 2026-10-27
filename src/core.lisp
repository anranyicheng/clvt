;;;; core.lisp — 张量核心：结构、步长、广播、连续判定、拷贝

(in-package :clvt)

;;; ------------------------------------------------------------------
;;; 结构定义
;;; ------------------------------------------------------------------

(defstruct (vt (:constructor %make-vt))
  "N 维张量 = 扁平缓冲区上的视图（strided view）。
  物理层四要素 + dtype：
    data    一维物理缓冲区（内存是一维的，(3,4) 张量存 12 个数）
    shape   逻辑形状
    strides 每维元素步长：沿第 d 轴前进 1，物理下标加 strides[d]；
            长度为 1 的轴 stride=0 即广播视图（虚拟重复读，语义只读）
    offset  视图起点（支持零拷贝切片）
    dtype   元素类型（同时决定物理数组的元素类型）
  逻辑层（shape/dtype/广播/归约语义）与执行层（快/慢路径选路）
  均建立在本物理描述之上；连续性只影响性能，不影响正确性。"
  (data (make-array 0) :type (simple-array *))
  (shape nil :type list)
  (strides nil :type list)
  (offset 0 :type fixnum)
  (dtype :float64 :type symbol))

(declaim (inline vt-shape vt-strides vt-offset vt-data vt-dtype
                 vt-order vt-size))
;; VT-P 不列入：结构体访问器由 SBCL 在 layout 已知时自动内联，显式
;; DECLAIM 只会得到「Proclaiming VT-P to be INLINE, but 1 call …」告警。

;;; ------------------------------------------------------------------
;;; 访问器与尺寸
;;; ------------------------------------------------------------------

(defun vt-element-type (vt)
  "返回张量底层物理数组的 Common Lisp 元素类型。"
  (vt-dtype->lisp-type (vt-dtype vt)))

(defun vt-order (vt)
  "张量的维度数（秩）。"
  (length (vt-shape vt)))

(defun vt-size (vt)
  "张量的逻辑元素总数。"
  (the fixnum (reduce #'* (vt-shape vt) :initial-value 1)))

(defun vt-shape-to-size (shape)
  "计算形状对应的总元素个数。"
  (declare (list shape))
  (reduce #'* shape :initial-value 1))

(defun vt-itemsize (vt)
  "每个元素的字节大小。"
  (vt-dtype-itemsize (vt-dtype vt)))

(defun vt-nbytes (vt)
  "张量占用的总字节数。"
  (* (vt-size vt) (vt-itemsize vt)))

;;; ------------------------------------------------------------------
;;; 步长
;;; ------------------------------------------------------------------

(defun vt-compute-strides (shape)
  "根据形状计算 C 连续（行主序）步长；标量 (nil) 返回 nil。"
  (declare (list shape) (optimize (speed 3) (safety 0)))
  (if (null shape)
      nil
      (let ((result nil)
            (stride 1))
        (declare (type fixnum stride))
        (do ((tail (reverse shape) (cdr tail)))
            ((null tail) result)
          (push stride result)
          (setf stride (the fixnum (* stride (the fixnum (car tail)))))))))

(defun vt-compute-logical-strides (shape)
  "计算给定形状的逻辑（连续内存）步长。"
  (vt-compute-strides shape))

;;; ------------------------------------------------------------------
;;; 构造
;;; ------------------------------------------------------------------

(defun make-vt (shape initial-element &key (dtype :float64))
  "创建指定形状与类型的张量，并用 initial-element 填充。"
  (let* ((size (vt-shape-to-size shape))
         (strides (vt-compute-strides shape)))
    ;; v0.4.6 性能修复：原实现把元素类型写成运行期值
    ;;   (make-array size :element-type (vt-dtype->lisp-type dtype) :initial-element ...)
    ;; SBCL 无法据此特化填充循环，只能走「通用序列 SETF + 逐元素类型检查」。
    ;; 1M double-float 实测 1632us；元素类型字面化后 945us（省 0.7ms，
    ;; 约等于一次 1M 逐元素算子总耗时的 12%）。因输出分配是**每个算子
    ;; 都要走一次**的公共路径，这里按 dtype 展开成 8 个字面类型分支。
    ;; 语义不变：仍用 vt-cast 把 INITIAL-ELEMENT 收敛到目标 dtype。
    (macrolet ((mk (lt kw)
                 `(%make-vt :data (make-array size :element-type ',lt
                                              :initial-element
                                              (the ,lt (vt-cast initial-element ,kw)))
                            :shape shape :strides strides :offset 0 :dtype dtype)))
      (ecase dtype
        (:complex128 (mk (complex double-float) :complex128))
        (:complex64  (mk (complex single-float) :complex64))
        (:float64 (mk double-float :float64))
        (:float32 (mk single-float :float32))
        (:int64   (mk (signed-byte 64) :int64))
        (:int32   (mk (signed-byte 32) :int32))
        (:int16   (mk (signed-byte 16) :int16))
        (:int8    (mk (signed-byte 8) :int8))
        (:uint16  (mk (unsigned-byte 16) :uint16))
        (:uint8   (mk (unsigned-byte 8) :uint8))))))

(defun %alloc-vt (shape dtype)
  "分配内存连续、元素内容**未定义**的张量。调用方必须随后写满每个元素。

   与 `make-vt` 的唯一差别是省掉 `:initial-element 0` 那一趟全量填充：
   1M complex128 实测 3 ms（f64 1 ms）。`vt-copy` / `vt-contiguous` /
   `%gather-contiguous` 这类「先分配、再把每个元素覆写一遍」的路径用不到零值，
   白填一趟等于白白多一次全量写（还让目标页变脏，随后的真实写入要付 RFO）。
   元素类型仍按字面展开，保持与 `make-vt` 相同的代码生成质量。"
  (let ((size (vt-shape-to-size shape)))
    (macrolet ((mk (lt) `(make-array size :element-type ',lt)))
      (%make-vt :data (ecase dtype
                        (:complex128 (mk (complex double-float)))
                        (:complex64  (mk (complex single-float)))
                        (:float64 (mk double-float))
                        (:float32 (mk single-float))
                        (:int64   (mk (signed-byte 64)))
                        (:int32   (mk (signed-byte 32)))
                        (:int16   (mk (signed-byte 16)))
                        (:int8    (mk (signed-byte 8)))
                        (:uint16  (mk (unsigned-byte 16)))
                        (:uint8   (mk (unsigned-byte 8))))
                :shape shape
                :strides (vt-compute-strides shape)
                :offset 0
                :dtype dtype))))

(defun %memcpy (dst dst-off src src-off count)
  "把 SRC[SRC-OFF, SRC-OFF+COUNT) 块拷贝到 DST[DST-OFF, …)（同 dtype、同底层数组类型）。

   为什么需要它：`replace` 只有**两端都被编译器确认为同类 specialized
   SIMPLE-ARRAY** 时才走字节级 memmove；少了类型声明就退化成逐元素泛型序列
   操作。1M 元素实测：`(replace dst src)` 8 ms vs 带声明 0.2 ms（16 字节元素
   最明显，f64 也白付一次 SAETP 分派）。本函数把「按元素类型展开 + 带 THE
   声明」这两件事收在一处，调用方不必各自复制 10 个分支。"
  (macrolet ((cp (lt)
               `(replace (the (simple-array ,lt (*)) dst)
                         (the (simple-array ,lt (*)) src)
                         :start1 dst-off :end1 (+ dst-off count)
                         :start2 src-off :end2 (+ src-off count))))
    (let ((et (array-element-type dst)))
      (cond ((equal et 'double-float)            (cp double-float))
            ((equal et 'single-float)            (cp single-float))
            ((equal et '(signed-byte 64))        (cp (signed-byte 64)))
            ((equal et '(signed-byte 32))        (cp (signed-byte 32)))
            ((equal et '(signed-byte 16))        (cp (signed-byte 16)))
            ((equal et '(signed-byte 8))         (cp (signed-byte 8)))
            ((equal et '(unsigned-byte 16))      (cp (unsigned-byte 16)))
            ((equal et '(unsigned-byte 8))       (cp (unsigned-byte 8)))
            ((equal et '(complex double-float))  (cp (complex double-float)))
            ((equal et '(complex single-float))  (cp (complex single-float)))
            ;; 未知元素类型：通用序列 replace（语义一致，只是慢）
            (t (replace dst src :start1 dst-off :end1 (+ dst-off count)
                             :start2 src-off :end2 (+ src-off count)))))
    count))

(defun %all-integer-sequence-p (seq)
  "递归判断嵌套序列的所有叶子是否都是整数。
   空序列返回 NIL（numpy 对空列表 np.array([]).dtype 实际上是 float64，
   本函数与 numpy 一致）。仅用于 ensure-vt 的 dtype 推断。"
  (cond
    ;; 叶子：整数 → T
    ((integerp seq) t)
    ;; 嵌套：list 或 vector 递归
    ((consp seq) (and (not (null seq))
                      (every #'%all-integer-sequence-p seq)))
    ((vectorp seq) (and (plusp (length seq))
                        (every #'%all-integer-sequence-p seq)))
    (t nil)))

(defun %any-complex-in-sequence-p (seq)
  "递归判断嵌套序列中是否含有复数叶子。
   用于 ensure-vt 的 dtype 推断：numpy 对「含复数的列表」推断 complex128，
   对「纯实数列表」仍推断 float64/int64（CONVENTIONS §16.2）。"
  (cond
    ((complexp seq) t)
    ((consp seq) (some #'%any-complex-in-sequence-p seq))
    ((vectorp seq) (some #'%any-complex-in-sequence-p seq))
    (t nil)))

(defun ensure-vt (obj &key (dtype nil))
  "将标量/序列/张量统一转换为张量。标量 -> 0 维张量；序列 -> 一维以上张量。
   复数（#C(re im)）：标量推断 complex64/complex128（按分量浮点宽度），
   序列中只要含一个复数叶子即整体 complex128（对标 numpy，
   见 CONVENTIONS §16.2）。"
  (etypecase obj
    (vt (if (and dtype (not (eq (vt-dtype obj) dtype)))
            (vt-copy-into (make-vt (vt-shape obj) 0 :dtype dtype) obj)
            obj))
    (number
     (let* ((infer (cond ((complexp obj)
                          (if (typep obj '(complex single-float))
                              :complex64 :complex128))
                         ((typep obj 'single-float) :float32)
                         ((typep obj 'double-float) :float64)
                         ((typep obj 'integer) :int64)
                         (t :float64)))
            (final (or dtype infer))
            (lisp-type (vt-dtype->lisp-type final)))
       (%make-vt :data (make-array 1 :element-type lisp-type
                                     :initial-element (vt-cast obj final))
                 :shape nil :strides nil :offset 0 :dtype final)))
    (sequence
     (let ((infer (cond ((%any-complex-in-sequence-p obj) :complex128)
                        ((%all-integer-sequence-p obj) :int64)
                        (t :float64))))
       (vt-from-sequence obj :dtype (or dtype infer))))))

;;; ------------------------------------------------------------------
;;; 广播
;;; ------------------------------------------------------------------

(declaim (inline vt-broadcast-shapes))

(defun vt-broadcast-shapes (shape1 shape2)
  "计算广播后的结果形状，严格对标 NumPy（右对齐，短形状左侧补 1）。"
  (declare (list shape1 shape2)
           (optimize (speed 3) (safety 0)))
  (let* ((len1 (length shape1))
         (len2 (length shape2))
         (max-len (max len1 len2))
         (result (make-list max-len)))
    (declare (type fixnum len1 len2 max-len))
    (do ((i 0 (1+ i))
         (s1 (append (make-list (- max-len len1) :initial-element 1) shape1) (cdr s1))
         (s2 (append (make-list (- max-len len2) :initial-element 1) shape2) (cdr s2))
         (r result (cdr r)))
        ((= i max-len) result)
      (declare (type fixnum i))
      (let ((dim1 (the fixnum (car s1)))
            (dim2 (the fixnum (car s2))))
        (declare (type fixnum dim1 dim2))
        (cond ((= dim1 dim2) (setf (car r) dim1))
              ((= dim1 1)    (setf (car r) dim2))
              ((= dim2 1)    (setf (car r) dim1))
              (t (error "形状 ~a 和 ~a 无法广播：维度 ~a 与 ~a 不兼容"
                        shape1 shape2 dim1 dim2)))))))

(declaim (inline vt-broadcast-strides))

(defun vt-broadcast-strides (orig-shape target-shape orig-strides)
  "计算 orig-shape 广播到 target-shape 后的步长（被广播的维度步长为 0）。"
  (declare (list orig-shape target-shape orig-strides)
           (optimize (speed 3) (safety 0)))
  (let* ((target-len (length target-shape))
         (orig-len (length orig-shape))
         (rank-diff (- target-len orig-len))
         (result (make-list target-len)))
    (declare (type fixnum target-len orig-len rank-diff))
    (when (minusp rank-diff)
      (error "vt-broadcast-strides: 原始形状 ~a 的秩大于目标形状 ~a" orig-shape target-shape))
    (let ((t-tail result)
          (t-shp target-shape))
      (dotimes (i rank-diff)
        (declare (type fixnum i))
        (setf (car t-tail) 0)
        (setf t-tail (cdr t-tail))
        (setf t-shp (cdr t-shp)))
      (do ((o-shp orig-shape (cdr o-shp))
           (o-str orig-strides (cdr o-str)))
          ((null o-shp) result)
        (let ((t-dim (the fixnum (car t-shp)))
              (o-dim (the fixnum (car o-shp))))
          (unless (or (= o-dim t-dim) (= o-dim 1))
            (error "vt-broadcast-strides: 形状不匹配! ~a vs ~a" o-dim t-dim))
          (setf (car t-tail)
                (if (= o-dim 1)
                    0
                    (the fixnum (car o-str))))
          (setf t-tail (cdr t-tail))
          (setf t-shp (cdr t-shp)))))))

;;; ------------------------------------------------------------------
;;; 轴归一化
;;; ------------------------------------------------------------------

(defun vt-normalize-axis (axis rank)
  "将负轴转换为正轴并做越界检查。axis 为 nil 时返回 nil。"
  (when axis
    (let ((ax (if (minusp axis) (+ axis rank) axis)))
      (when (or (< ax 0) (>= ax rank))
        (error "axis ~a is out of bounds for tensor of rank ~a" axis rank))
      ax)))

(defun vt-normalize-axes (axis rank)
  "将 axis (nil/整数/整数列表）归一化为排序后的正轴列表；nil 表示全局归约。"
  (if (null axis)
      nil
      (let* ((axes (if (listp axis) axis (list axis)))
             (sorted (sort (mapcar (lambda (a)
                                     (vt-normalize-axis a rank))
                                   axes)
                           #'<)))
        (loop for (a b) on sorted
              when (and b (= a b))
                do (error "vt-normalize-axes: 轴 ~a 重复" a))
        sorted)))

;;; ------------------------------------------------------------------
;;; 连续判定与落地
;;; ------------------------------------------------------------------

(defun vt-contiguous-p (vt)
  "判断张量是否为 C 连续（可安全重塑）。对标 numpy 的 c_contiguous 判定。"
  (let ((shape (vt-shape vt))
        (strides (vt-strides vt)))
    (if (or (null shape)
            (some #'zerop shape))
        t
        (let ((expected 1) (contiguous t))
          (declare (type fixnum expected))
          (loop for i fixnum from (1- (length shape)) downto 0
                for dim fixnum = (the fixnum (nth i shape))
                for stride fixnum = (the fixnum (nth i strides))
                do (cond ((= dim 1) nil)
                         ((= stride expected)
                          (setf expected (the fixnum (* expected dim))))
                         (t (setf contiguous nil))))
          contiguous))))



;;; ------------------------------------------------------------------
;;; 拷贝与类型转换
;;; ------------------------------------------------------------------
(defparameter +astype-fuse-min-run+ 16
  "`vt-astype` 对非连续视图启用「逐 run 单趟转换」的最小内层 run 长度。

   为什么有门槛：run 太短时（转置/翻转视图 run=1）逐 run 的调用开销
   远大于收益，此时改走 `%gather-contiguous` 的同 dtype 特化收集
   （一次遍历、静态元素类型），再做整数组转换。
   16 与 `vt-copy-into` 用 `%inner-run-suffix` 做块拷贝的门槛取同一个
   数：两处若各取一个值，同形视图会在两条 API 上表现出不同的快慢。")

(eval-when (:compile-toplevel :load-toplevel :execute)
  (defun %astype-cast-form (out-lt src-expr &optional in-lt)
    "生成把 src-expr 转换到 out-lt 的代码。
     整数走与 vt-cast / vt-cast-fun 一致的安全 coerce/回绕语义。
     IN-LT（源元素类型）只用于区分「复数源」，决定走哪条 §16.4 转换路径：
       · 目标为复数 + 源为复数 → 实部/虚部分别 coerce 到分量类型；
       · 目标为复数 + 源为实数 → 虚部补 0；
       · 目标为实/整 + 源为复数 → 先取实部（静默丢弃虚部，对标 numpy astype）；
       · 目标为实/整 + 源为实数 → 原规则不变。"
    (let ((in-complex (and (consp in-lt) (eq (car in-lt) 'complex))))
      (labels ((wrap (ft zero)
                 (if in-complex
                     `(complex (coerce (realpart ,src-expr) ',ft)
                               (coerce (imagpart ,src-expr) ',ft))
                     `(complex (coerce ,src-expr ',ft) ,zero)))
               (cast (form)
                 (cond ((equal out-lt '(signed-byte 64))   `(%coerce-int64  ,form))
                       ((equal out-lt '(signed-byte 32))   `(%coerce-int32  ,form))
                       ((equal out-lt '(signed-byte 16))   `(%coerce-int16  ,form))
                       ((equal out-lt '(signed-byte 8))    `(%coerce-int8   ,form))
                       ((equal out-lt '(unsigned-byte 16)) `(%coerce-uint16 ,form))
                       ((equal out-lt '(unsigned-byte 8))  `(%coerce-uint8  ,form))
                       ((subtypep out-lt 'integer)         `(truncate ,form))
                       (t                                  `(coerce ,form ',out-lt)))))
        (cond ((equal out-lt '(complex double-float)) (wrap 'double-float '0.0d0))
              ((equal out-lt '(complex single-float)) (wrap 'single-float '0.0s0))
              ;; 实/整目标：复数源先取实部（§16.4），其余逐字沿用原规则。
              (t (cast (if in-complex `(realpart ,src-expr) src-expr))))))))

;;; ------------------------------------------------------------------
;;; 内层连续 run 分解（strided 视图的块拷贝快路径原语）
;;; ------------------------------------------------------------------

(defun %inner-run-suffix (shape strides)
  "把 strided 视图从尾部向头部延伸出最长的「行主序连续后缀」。
   返回 (values cut run)：cut 为后缀起始轴下标（[0, rank]），
   run 为后缀元素数（≥1）。dim=1 的轴不参与延伸（其 stride 无意义，
   仅贡献单个元素）。完全连续视图 run = 总元素数（cut=0）；
   转置/步进视图退化为 run=1（此时块拷贝无收益，调用方应走逐元素路径）。

   铁律 1：本分解只影响选路，不改变任何语义。"
  (declare (type list shape strides))
  (let ((run 1)
        (cut (length shape)))
    (declare (type fixnum run cut))
    (loop for d from (1- (length shape)) downto 0
          for dim = (nth d shape)
          for stride = (nth d strides)
          when (> dim 1)
            do (if (= stride run)
                   (setf run (the fixnum (* run dim))
                         cut d)
                   (return)))
    (values cut run)))

(defun %map-inner-runs (shape strides offset cut run fn)
  "对视图 [0, cut) 的外层索引空间做行主序 odometer，对每个外层组合调用
   (funcall fn run-start)；run-start 为该组合对应内层连续 run 的物理起始
   下标（= offset + Σ i_d × stride_d）。外层任一维度为 0 时不调用 FN。
   各 run 物理区间互不重叠（dim>1 且 stride=0 的广播维度已在
   vt-contiguous-p / 写入侧契约中排除）。
   RUN 应与 (%inner-run-suffix shape strides) 的返回值一致。"
  (declare (type list shape strides)
           (type fixnum offset cut)
           ;; RUN 是「调用方与本函数必须对同一个内层 run 长度达成一致」的
           ;; 契约参数：本函数只在 [0,cut) 的外层 odometer 上工作，不需要它。
           ;; 保留形参是为了让调用点显式写出 run（错配时人能看见）。
           (ignore run))
  (let* ((outer-dims (coerce (subseq shape 0 cut) 'simple-vector))
         (outer-strides (coerce (subseq strides 0 cut) 'simple-vector))
         (k (length outer-dims)))
    (unless (some #'zerop outer-dims)
      (let ((idx (make-array k :element-type 'fixnum :initial-element 0))
            (p offset))
        (declare (type simple-vector outer-dims outer-strides)
                 (type (simple-array fixnum (*)) idx)
                 (type fixnum p))
        (loop
          (funcall fn p)
          (let ((d (1- k)))
            (loop
              (when (< d 0) (return-from %map-inner-runs))
              (incf (aref idx d))
              (if (< (aref idx d) (the fixnum (svref outer-dims d)))
                  (progn (incf p (the fixnum (svref outer-strides d))) (return))
                  (progn (decf p (* (1- (the fixnum (svref outer-dims d)))
                                    (the fixnum (svref outer-strides d))))
                         (setf (aref idx d) 0)
                         (decf d)))))))))
  (values))

(defun %gather-contiguous (tensor)
  "把非连续视图按 strides 收集成**同 dtype 的内存连续**新张量（行主序落点）。
   与 `vt-copy-into` 的通用 walker 语义相同，差别只在**元素类型是静态的**：
   泛型 `aref` 对 `(complex double-float)` 每个元素都要装箱/解包，
   实测 1000×1000 转置视图 16MB 拷贝 17ms，特化后 1~2ms。

   用途：`vt-astype` 的非连续回退。同 dtype 时它的返回**就是**结果
   （一次特化遍历，无第二趟）；跨 dtype 时作为中间副本，交给连续 cast 特化。

   遍历形态：**末轴用 DOTIMES 展开，其余轴用 odometer**。
   落点行主序 ⇒ 末轴在目标上连续，内层循环只需源指针按 `s[末轴]` 步进；
   odometer 的迭代次数从 `size` 降到 `size / d[末轴]`，每元素开销被摊薄。
   实测 1000×1000 转置视图：全 odometer 6ms → 末轴展开 1ms（同数据的
   手写 2 层循环也是 1ms，说明这是该访存次序的下限）。

   不按 `%inner-run-suffix` 合并 run：转置/步进视图的 run 只有 1 个元素，
   逐 run `replace` 的调用开销远大于收益。

   铁律 1：只改变搬运方式，落点（行主序连续布局）与逐元素 walker 逐位一致。"
  (let* ((shape (vt-shape tensor))
         (rank (length shape))
         (size (vt-shape-to-size shape))
         (dtype (vt-dtype tensor))
         (in-data (vt-data tensor))
         (in-strides (vt-strides tensor))
         (in-offset (vt-offset tensor))
         (out (%alloc-vt shape dtype))
         (out-data (vt-data out)))
    (declare (type fixnum rank size in-offset))
    (let* ((dims (coerce shape 'simple-vector))
           (i-strs (coerce in-strides 'simple-vector))
           (inner-dim (if (plusp rank) (the fixnum (svref dims (1- rank))) 1))
           (outer-rank (max 0 (1- rank)))
           ;; 末轴为 0 长度时 size 也是 0，无需再除
           (outer-size (if (zerop outer-rank) 1
                           (if (zerop inner-dim) 0 (/ size inner-dim))))
           (indices (make-array (max outer-rank 1)
                                :element-type 'fixnum :initial-element 0)))
      (declare (type simple-vector dims i-strs)
               (type fixnum inner-dim outer-rank outer-size)
               (type (simple-array fixnum (*)) indices))
      (macrolet ((walk (elt)
                   `(let ((src (the (simple-array ,elt (*)) in-data))
                          (dst (the (simple-array ,elt (*)) out-data))
                          (s-in (if (plusp rank)
                                    (the fixnum (svref i-strs (1- rank))) 1))
                          (i-ptr in-offset)
                          (d-ptr 0))
                      (declare (type (simple-array ,elt (*)) src dst)
                               (type fixnum s-in i-ptr d-ptr))
                      (dotimes (k outer-size)
                        ;; ---- 末轴：dst 连续、src 按 S-IN 步进 ----
                        (let ((sp i-ptr))
                          (declare (type fixnum sp))
                          (dotimes (j inner-dim)
                            (setf (aref dst (the fixnum (+ d-ptr j))) (aref src sp))
                            (incf sp s-in)))
                        (incf d-ptr inner-dim)
                        ;; ---- 其余轴：行主序 odometer（只走 outer-rank 个轴）----
                        (let ((d (the fixnum (1- outer-rank))))
                          (declare (type fixnum d))
                          (loop
                            (when (< d 0) (return))
                            (incf (aref indices d))
                            (incf i-ptr (the fixnum (svref i-strs d)))
                            (when (< (aref indices d) (the fixnum (svref dims d)))
                              (return))
                            (let ((dim (the fixnum (svref dims d))))
                              (decf i-ptr (* dim (the fixnum (svref i-strs d))))
                              (setf (aref indices d) 0)
                              (decf d))))))))
        (let ((lt (array-element-type in-data)))
          (cond ((equal lt 'double-float)            (walk double-float))
                ((equal lt 'single-float)            (walk single-float))
                ((equal lt '(signed-byte 64))        (walk (signed-byte 64)))
                ((equal lt '(signed-byte 32))        (walk (signed-byte 32)))
                ((equal lt '(signed-byte 16))        (walk (signed-byte 16)))
                ((equal lt '(signed-byte 8))         (walk (signed-byte 8)))
                ((equal lt '(unsigned-byte 16))      (walk (unsigned-byte 16)))
                ((equal lt '(unsigned-byte 8))       (walk (unsigned-byte 8)))
                ((equal lt '(complex double-float))  (walk (complex double-float)))
                ((equal lt '(complex single-float))  (walk (complex single-float)))
                ;; 未知底层元素类型：回退到库的通用 strided 拷贝（语义一致）。
                ;; **不能**回写 VT-CONTIGUOUS —— 它现在反过来调用本函数，
                ;; 那样就是一个自递归死循环。
                (t (let ((new (%alloc-vt shape dtype)))
                     (vt-copy-into new tensor)
                     new))))))
    out))

(defun vt-contiguous (vt)
  "返回内存连续的副本（若已连续则返回自身，零拷贝）。

   非连续时走 `%gather-contiguous`：它按**末轴展开 + 其余轴 odometer** 做
   同 dtype 的特化收集，避开了 `vt-copy-into` 通用 walker 每元素的
   shape/stride 重算。实测 1000×1000 转置视图（16MB 来回）：
   3.4ms → 1.2ms；stride=2 视图 1.0ms → 0.4ms。
   落点仍是行主序连续（铁律 1：连续性只影响选路，不影响结果）。"
  (if (vt-contiguous-p vt)
      vt
      (%gather-contiguous vt)))

(defun vt-astype (tensor new-dtype)
  "将张量转换为新类型（浮点转整数截断）。返回连续的新张量。

   复数语义（CONVENTIONS §16.4）：
     · 目标为复数 → 实部/虚部分别转换；实数入参虚部补 +0.0；
     · 目标为实数/整数且入参为复数 → **取实部**（静默丢弃虚部，
       对标 numpy astype 的 ComplexWarning，本库不产生警告）；
     · NaN/±Inf → 整数一律 0（§7.2）。

   执行层（§16.4.1）—— 三条落点，**共用同一张 10×10 特化表**：
     1. 连续视图：整数组一次 `%CAST-RUN`（同 dtype 走 `%memcpy`）；
     2. 非连续、内层 run ≥ `+astype-fuse-min-run+`（切片/跨步视图）：
        逐 run 直接写进目标数组 —— 单趟、无临时副本；
     3. 非连续、run 更短（转置/翻转视图）：先 `%gather-contiguous` 做
        同 dtype 特化收集（同 dtype 时它就是结果），跨 dtype 再走第 1 条。
   分配用 `make-array`（不填充零值，与 `%alloc-vt` 等价）。"
  (with-float-safe
    (let* ((shape (vt-shape tensor))
           (rank (length shape))
           (size (vt-shape-to-size shape))
           (in-data (vt-data tensor))
           (in-strides (vt-strides tensor))
           (in-offset (vt-offset tensor))
           (new-lisp-type (vt-dtype->lisp-type new-dtype))
           (new-data (make-array size :element-type new-lisp-type))
           (new (%make-vt :data new-data
                          :shape shape
                          :strides (vt-compute-strides shape)
                          :offset 0
                          :dtype new-dtype)))
      (declare (type fixnum rank size in-offset))
      ;; 10×10 特化表被两条落点共用（整数组 / 逐 run），所以提成一个局部
      ;; 函数 %CAST-RUN：(DST P Q LEN) = 把 IN-DATA 的 [P, P+LEN) 按
      ;; NEW-DTYPE 写到 DST 的 [Q, Q+LEN)。提成一处而不是各写一套，是为了
      ;; 避免「快路径与通用路径各写一套」再分叉（CONVENTIONS §7.4）。
      (macrolet
          ((spec (in-lt out-lt)
             (let ((expr (%astype-cast-form out-lt '(aref src dp) in-lt)))
               `(let ((src (the (simple-array ,in-lt (*)) in-data))
                      (d (the (simple-array ,out-lt (*)) dst))
                      (dp p)
                      (dq q))
                  (declare (type (simple-array ,in-lt (*)) src)
                           (type (simple-array ,out-lt (*)) d)
                           (type fixnum dp dq))
                  (dotimes (j len)
                    (setf (aref d dq) ,expr)
                    (incf dp)
                    (incf dq)))))
           (memcpy (lt)
             ;; 同型块拷贝统一走 %MEMCPY（库内唯一块拷贝实现，§2.2 / §16.4.1）。
             ;; LT 只让调用点自解释（写出「两端同型」这件事），实际类型由
             ;; %MEMCPY 在运行时按目的地元素类型分派 —— 这也是它能一处顶掉
             ;; 原先 10 份 `(replace (the (simple-array …)))` 的原因。
             (declare (ignore lt))
             `(%memcpy dst q in-data p len)))
        (flet ((%cast-run (dst p q len)
                 (declare (type fixnum p q len))
                 (let ((in-et (array-element-type in-data)))
                   (cond
                     ;; ============================================================
                     ;; 同型 memcpy
                     ;; ============================================================
                     ((and (equal in-et 'double-float)       (equal new-lisp-type 'double-float))
                      (memcpy double-float))
                     ((and (equal in-et 'single-float)       (equal new-lisp-type 'single-float))
                      (memcpy single-float))
                     ((and (equal in-et '(signed-byte 64))   (equal new-lisp-type '(signed-byte 64)))
                      (memcpy (signed-byte 64)))
                     ((and (equal in-et '(signed-byte 32))   (equal new-lisp-type '(signed-byte 32)))
                      (memcpy (signed-byte 32)))
                     ;; 以下 4 个为防御性：*vt-storage-dtypes* 目前不分配这些底层数组，
                     ;; 但保留可让 vt-from-array 等异常路径也走快车道。
                     ((and (equal in-et '(signed-byte 16))   (equal new-lisp-type '(signed-byte 16)))
                      (memcpy (signed-byte 16)))
                     ((and (equal in-et '(signed-byte 8))    (equal new-lisp-type '(signed-byte 8)))
                      (memcpy (signed-byte 8)))
                     ((and (equal in-et '(unsigned-byte 16)) (equal new-lisp-type '(unsigned-byte 16)))
                      (memcpy (unsigned-byte 16)))
                     ((and (equal in-et '(unsigned-byte 8))  (equal new-lisp-type '(unsigned-byte 8)))
                      (memcpy (unsigned-byte 8)))
                     ;; 复数同型：物理布局逐位相同（interleaved），一次 replace 即可，
                     ;; 不需要经过任何逐元素解包/重组（§16.4：同 dtype 不改变数值）。
                     ((and (equal in-et '(complex double-float)) (equal new-lisp-type '(complex double-float)))
                      (memcpy (complex double-float)))
                     ((and (equal in-et '(complex single-float)) (equal new-lisp-type '(complex single-float)))
                      (memcpy (complex single-float)))

                     ;; ============================================================
                     ;; double-float 源
                     ;; ============================================================
                     ((and (equal in-et 'double-float) (equal new-lisp-type '(signed-byte 64)))
                      (spec double-float (signed-byte 64)))
                     ((and (equal in-et 'double-float) (equal new-lisp-type '(signed-byte 32)))
                      (spec double-float (signed-byte 32)))
                     ((and (equal in-et 'double-float) (equal new-lisp-type '(signed-byte 16)))
                      (spec double-float (signed-byte 16)))
                     ((and (equal in-et 'double-float) (equal new-lisp-type '(signed-byte 8)))
                      (spec double-float (signed-byte 8)))
                     ((and (equal in-et 'double-float) (equal new-lisp-type '(unsigned-byte 16)))
                      (spec double-float (unsigned-byte 16)))
                     ((and (equal in-et 'double-float) (equal new-lisp-type '(unsigned-byte 8)))
                      (spec double-float (unsigned-byte 8)))
                     ((and (equal in-et 'double-float) (equal new-lisp-type 'single-float))
                      (spec double-float single-float))

                     ;; ============================================================
                     ;; single-float 源
                     ;; ============================================================
                     ((and (equal in-et 'single-float) (equal new-lisp-type 'double-float))
                      (spec single-float double-float))
                     ((and (equal in-et 'single-float) (equal new-lisp-type '(signed-byte 64)))
                      (spec single-float (signed-byte 64)))
                     ((and (equal in-et 'single-float) (equal new-lisp-type '(signed-byte 32)))
                      (spec single-float (signed-byte 32)))
                     ((and (equal in-et 'single-float) (equal new-lisp-type '(signed-byte 16)))
                      (spec single-float (signed-byte 16)))
                     ((and (equal in-et 'single-float) (equal new-lisp-type '(signed-byte 8)))
                      (spec single-float (signed-byte 8)))
                     ((and (equal in-et 'single-float) (equal new-lisp-type '(unsigned-byte 16)))
                      (spec single-float (unsigned-byte 16)))
                     ((and (equal in-et 'single-float) (equal new-lisp-type '(unsigned-byte 8)))
                      (spec single-float (unsigned-byte 8)))

                     ;; ============================================================
                     ;; int64 源
                     ;; ============================================================
                     ((and (equal in-et '(signed-byte 64)) (equal new-lisp-type 'double-float))
                      (spec (signed-byte 64) double-float))
                     ((and (equal in-et '(signed-byte 64)) (equal new-lisp-type 'single-float))
                      (spec (signed-byte 64) single-float))
                     ((and (equal in-et '(signed-byte 64)) (equal new-lisp-type '(signed-byte 32)))
                      (spec (signed-byte 64) (signed-byte 32)))
                     ((and (equal in-et '(signed-byte 64)) (equal new-lisp-type '(signed-byte 16)))
                      (spec (signed-byte 64) (signed-byte 16)))
                     ((and (equal in-et '(signed-byte 64)) (equal new-lisp-type '(signed-byte 8)))
                      (spec (signed-byte 64) (signed-byte 8)))
                     ((and (equal in-et '(signed-byte 64)) (equal new-lisp-type '(unsigned-byte 16)))
                      (spec (signed-byte 64) (unsigned-byte 16)))
                     ((and (equal in-et '(signed-byte 64)) (equal new-lisp-type '(unsigned-byte 8)))
                      (spec (signed-byte 64) (unsigned-byte 8)))

                     ;; ============================================================
                     ;; int32 源
                     ;; ============================================================
                     ((and (equal in-et '(signed-byte 32)) (equal new-lisp-type 'double-float))
                      (spec (signed-byte 32) double-float))
                     ((and (equal in-et '(signed-byte 32)) (equal new-lisp-type 'single-float))
                      (spec (signed-byte 32) single-float))
                     ((and (equal in-et '(signed-byte 32)) (equal new-lisp-type '(signed-byte 64)))
                      (spec (signed-byte 32) (signed-byte 64)))
                     ((and (equal in-et '(signed-byte 32)) (equal new-lisp-type '(signed-byte 16)))
                      (spec (signed-byte 32) (signed-byte 16)))
                     ((and (equal in-et '(signed-byte 32)) (equal new-lisp-type '(signed-byte 8)))
                      (spec (signed-byte 32) (signed-byte 8)))
                     ((and (equal in-et '(signed-byte 32)) (equal new-lisp-type '(unsigned-byte 16)))
                      (spec (signed-byte 32) (unsigned-byte 16)))
                     ((and (equal in-et '(signed-byte 32)) (equal new-lisp-type '(unsigned-byte 8)))
                      (spec (signed-byte 32) (unsigned-byte 8)))

                     ;; ============================================================
                     ;; 复数源 → 复数目标（CONVENTIONS §16.4）
                     ;; 分量各自 coerce 后重组；interleaved 存储的读写都是 16B/8B 整体，
                     ;; 因此这里不存在「实部/虚部两趟」的余地，逐元素 spec 即最优形态。
                     ;; ============================================================
                     ((and (equal in-et '(complex double-float)) (equal new-lisp-type '(complex single-float)))
                      (spec (complex double-float) (complex single-float)))
                     ((and (equal in-et '(complex single-float)) (equal new-lisp-type '(complex double-float)))
                      (spec (complex single-float) (complex double-float)))

                     ;; ============================================================
                     ;; 复数源 → 实/整目标：先取实部，再按实数规则 cast
                     ;; （§16.4 astype 语义：静默丢弃虚部，对标 numpy 的 ComplexWarning，
                     ;;  本库不产生 warning；整数目标仍是截断 + 回绕）。
                     ;; ============================================================
                     ((and (equal in-et '(complex double-float)) (equal new-lisp-type 'double-float))
                      (spec (complex double-float) double-float))
                     ((and (equal in-et '(complex double-float)) (equal new-lisp-type 'single-float))
                      (spec (complex double-float) single-float))
                     ((and (equal in-et '(complex double-float)) (equal new-lisp-type '(signed-byte 64)))
                      (spec (complex double-float) (signed-byte 64)))
                     ((and (equal in-et '(complex double-float)) (equal new-lisp-type '(signed-byte 32)))
                      (spec (complex double-float) (signed-byte 32)))
                     ((and (equal in-et '(complex double-float)) (equal new-lisp-type '(signed-byte 16)))
                      (spec (complex double-float) (signed-byte 16)))
                     ((and (equal in-et '(complex double-float)) (equal new-lisp-type '(signed-byte 8)))
                      (spec (complex double-float) (signed-byte 8)))
                     ((and (equal in-et '(complex double-float)) (equal new-lisp-type '(unsigned-byte 16)))
                      (spec (complex double-float) (unsigned-byte 16)))
                     ((and (equal in-et '(complex double-float)) (equal new-lisp-type '(unsigned-byte 8)))
                      (spec (complex double-float) (unsigned-byte 8)))
                     ((and (equal in-et '(complex single-float)) (equal new-lisp-type 'double-float))
                      (spec (complex single-float) double-float))
                     ((and (equal in-et '(complex single-float)) (equal new-lisp-type 'single-float))
                      (spec (complex single-float) single-float))
                     ((and (equal in-et '(complex single-float)) (equal new-lisp-type '(signed-byte 64)))
                      (spec (complex single-float) (signed-byte 64)))
                     ((and (equal in-et '(complex single-float)) (equal new-lisp-type '(signed-byte 32)))
                      (spec (complex single-float) (signed-byte 32)))
                     ((and (equal in-et '(complex single-float)) (equal new-lisp-type '(signed-byte 16)))
                      (spec (complex single-float) (signed-byte 16)))
                     ((and (equal in-et '(complex single-float)) (equal new-lisp-type '(signed-byte 8)))
                      (spec (complex single-float) (signed-byte 8)))
                     ((and (equal in-et '(complex single-float)) (equal new-lisp-type '(unsigned-byte 16)))
                      (spec (complex single-float) (unsigned-byte 16)))
                     ((and (equal in-et '(complex single-float)) (equal new-lisp-type '(unsigned-byte 8)))
                      (spec (complex single-float) (unsigned-byte 8)))

                     ;; ============================================================
                     ;; 实数/整数源 → 复数目标：虚部补 0（§16.4）
                     ;; ============================================================
                     ((and (equal in-et 'double-float) (equal new-lisp-type '(complex double-float)))
                      (spec double-float (complex double-float)))
                     ((and (equal in-et 'double-float) (equal new-lisp-type '(complex single-float)))
                      (spec double-float (complex single-float)))
                     ((and (equal in-et 'single-float) (equal new-lisp-type '(complex double-float)))
                      (spec single-float (complex double-float)))
                     ((and (equal in-et 'single-float) (equal new-lisp-type '(complex single-float)))
                      (spec single-float (complex single-float)))
                     ((and (equal in-et '(signed-byte 64)) (equal new-lisp-type '(complex double-float)))
                      (spec (signed-byte 64) (complex double-float)))
                     ((and (equal in-et '(signed-byte 64)) (equal new-lisp-type '(complex single-float)))
                      (spec (signed-byte 64) (complex single-float)))
                     ((and (equal in-et '(signed-byte 32)) (equal new-lisp-type '(complex double-float)))
                      (spec (signed-byte 32) (complex double-float)))
                     ((and (equal in-et '(signed-byte 32)) (equal new-lisp-type '(complex single-float)))
                      (spec (signed-byte 32) (complex single-float)))
                     ((and (equal in-et '(signed-byte 16)) (equal new-lisp-type '(complex double-float)))
                      (spec (signed-byte 16) (complex double-float)))
                     ((and (equal in-et '(signed-byte 16)) (equal new-lisp-type '(complex single-float)))
                      (spec (signed-byte 16) (complex single-float)))
                     ((and (equal in-et '(signed-byte 8)) (equal new-lisp-type '(complex double-float)))
                      (spec (signed-byte 8) (complex double-float)))
                     ((and (equal in-et '(signed-byte 8)) (equal new-lisp-type '(complex single-float)))
                      (spec (signed-byte 8) (complex single-float)))
                     ((and (equal in-et '(unsigned-byte 16)) (equal new-lisp-type '(complex double-float)))
                      (spec (unsigned-byte 16) (complex double-float)))
                     ((and (equal in-et '(unsigned-byte 16)) (equal new-lisp-type '(complex single-float)))
                      (spec (unsigned-byte 16) (complex single-float)))
                     ((and (equal in-et '(unsigned-byte 8)) (equal new-lisp-type '(complex double-float)))
                      (spec (unsigned-byte 8) (complex double-float)))
                     ((and (equal in-et '(unsigned-byte 8)) (equal new-lisp-type '(complex single-float)))
                      (spec (unsigned-byte 8) (complex single-float)))

                     ;; ============================================================
                     ;; 窄整型源（int16/int8/uint16/uint8）→ 实/整目标
                     ;;
                     ;; 这 4 种是**逻辑级** dtype，物理也是窄数组。原先它们掉进下面的
                     ;; 通用 fallback：每元素一次 `(funcall (vt-cast-fun ...))` ——
                     ;; 一次闭包调用 + 动态 dtype 派发，1M 元素实测 8ms（同规模
                     ;; f64→f32 只要 1ms）。这里按 (源, 目标) 逐个展开特化循环，
                     ;; 与上面 4 种存储级 dtype 走的是同一份 `spec` 模板。
                     ;; ============================================================
                     ((member in-et +small-int-ets+ :test #'equal)
                      (macrolet ((narrow-cases ()
                                   `(cond
                                      ,@(loop for in-lt in '((signed-byte 16) (signed-byte 8)
                                                             (unsigned-byte 16) (unsigned-byte 8))
                                              append
                                              (loop for out-lt in '(double-float single-float
                                                                    (signed-byte 64) (signed-byte 32)
                                                                    (signed-byte 16) (signed-byte 8)
                                                                    (unsigned-byte 16) (unsigned-byte 8))
                                                    collect
                                                    `((and (equal in-et ',in-lt)
                                                           (equal new-lisp-type ',out-lt))
                                                      (spec ,in-lt ,out-lt)))))))
                        (narrow-cases)))
                   ;; ============================================================
                   ;; 其他组合：通用 fallback（走 vt-cast-fun，语义已对齐）
                   ;; ============================================================
                   (t
                    (let ((conv (vt-cast-fun new-dtype))
                          (dp p)
                          (dq q))
                      (declare (type fixnum dp dq))
                      (dotimes (j len)
                        (setf (aref dst dq)
                              (funcall conv (aref in-data dp)))
                        (incf dp)
                        (incf dq))))))))
          (cond
            ;; ---- 0 维 ----
            ((zerop rank)
             (setf (aref new-data 0)
                   (funcall (vt-cast-fun new-dtype) (aref in-data in-offset))))

            ;; ---- 连续视图：整数组一趟 ----
            ((vt-contiguous-p tensor)
             (%cast-run new-data in-offset 0 size))

            ;; ---- 非连续视图 ----
            (t
             (multiple-value-bind (run-cut run-len)
                 (%inner-run-suffix shape in-strides)
               (declare (type fixnum run-cut run-len))
               (if (>= run-len +astype-fuse-min-run+)
                   ;; 内层 run 够长：逐 run 直接写进目标数组。
                   ;; 一趟读者 + 一趟写者，省掉「先收集成连续副本再转换」的
                   ;; 中间 4~16MB 分配与两趟额外流量。
                   (let ((q 0))
                     (declare (type fixnum q))
                     (%map-inner-runs shape in-strides in-offset run-cut run-len
                       (lambda (p)
                         (declare (type fixnum p))
                         (%cast-run new-data p q run-len)
                         (incf q run-len)))
                     new)
                   ;; run 太短：同 dtype 特化收集（同 dtype 时它就是结果）。
                   ;; 这两条必须 RETURN-FROM：收集结果与调用方预分配的 NEW
                   ;; 是两个不同对象，落回函数末尾的 NEW 会返回未写满的数组。
                   (let ((tmp (%gather-contiguous tensor)))
                     (if (eq (vt-dtype tensor) new-dtype)
                         (return-from vt-astype tmp)
                         (return-from vt-astype (vt-astype tmp new-dtype))))))))))
      new)))


(defun vt-copy (vt &key dtype)
  "深度拷贝：返回独立、内存连续的新张量。可选类型转换。"
  (let ((target-dtype (or dtype (vt-dtype vt))))
    (if (eq target-dtype (vt-dtype vt))
        (if (vt-contiguous-p vt)
            (let* ((shape (vt-shape vt))
                   (size (vt-shape-to-size shape))
                   (new (%alloc-vt shape target-dtype)))
              (%memcpy (vt-data new) 0 (vt-data vt) (vt-offset vt) size)
              new)
            ;; 非连续：走 %GATHER-CONTIGUOUS 的特化收集（同为行主序落点，
            ;; 但它末轴展开 + 其余轴 odometer，实测快 2~2.5 倍）。
            ;; 它自己分配新数组，故不需要上面那个 NEW。
            (%gather-contiguous vt))
        (vt-astype vt target-dtype))))
;;; ------------------------------------------------------------------
;;; 别名检测辅助（放在 vt-copy-into 之前）
;;; ------------------------------------------------------------------

(defun %vt-views-overlap-p (a b)
  "判断两个 vt 视图是否共享底层物理数组，且其访问的物理索引区间存在交集。
   目的：为 vt-copy-into 提供别名检测。当 dest 与 src 别名且区间重叠时，
   必须先对 src 做快照，否则逐元素写入会破坏尚未读取的源数据
   （即 memmove 语义的原地拷贝退化）。
   返回：
     T   —— a 与 b 共享同一底层数组，且物理访问区间相交；
     NIL —— 其他情况（含：底层数组不同、区间不相交）。
   前提与假设：
     1. 同一轴上不会同时出现正负步长（库内由 vt-flip / vt-transpose 保证）。
     2. 广播维度（dim>1 且 stride=0）不扩展区间，只贡献 offset 本身。
     3. 标量视图（shape=nil）的区间退化为 [offset, offset]。"
  (and (eq (vt-data a) (vt-data b))
       (flet ((span (v)
                "返回视图 v 访问的物理索引闭区间 [lo, hi]。
                 负步长把 lo 向低端扩展，正步长把 hi 向高端扩展，
                 因此 lo/hi 对任意步长符号都给出正确的物理边界。
                 stride=0 的维度（广播）对边界无贡献。"
                (let ((lo (vt-offset v)) (hi (vt-offset v)))
                  (declare (type fixnum lo hi))
                  (loop for d of-type fixnum in (vt-shape v)
                        for s of-type fixnum in (vt-strides v)
                        when (> d 1)
                          do (let ((end (* (1- d) s)))
                               (if (minusp end)
                                   (decf lo (- end))
                                   (incf hi end))))
                  (values lo hi))))
         (multiple-value-bind (a-lo a-hi) (span a)
           (multiple-value-bind (b-lo b-hi) (span b)
             (and (<= a-lo b-hi) (<= b-lo a-hi)))))))

(defun %copy-strided-hi (dest-data dest-dtype dest-strides dest-off
                         src-data src-strides src-off
                         shape size same-dtype)
  "rank ≥ 4 的通用里程计。
   - 预计算 (1- dim) 数组，避免每次回退重复计算
   - 内层循环用局部变量缓存 indices/dims/stride
   - 同 dtype 时直接拷贝，异 dtype 时循环外取 caster"
  (declare (type fixnum size dest-off src-off)
           (type list shape dest-strides src-strides))
  (when (zerop size) (return-from %copy-strided-hi nil))
  (let* ((rank   (length shape))
         (dims   (coerce shape 'simple-vector))
         (d-strs (coerce dest-strides 'simple-vector))
         (s-strs (coerce src-strides 'simple-vector))
         (dims-1 (make-array rank :element-type 'fixnum))
         (indices (make-array rank :element-type 'fixnum :initial-element 0))
         (caster (unless same-dtype (vt-cast-fun dest-dtype)))
         (d-ptr dest-off)
         (s-ptr src-off))
    (declare (type fixnum rank d-ptr s-ptr)
             (type simple-vector dims d-strs s-strs)
             (type (simple-array fixnum (*)) dims-1 indices)
             (type (or null function) caster))
    (loop for i fixnum from 0 below rank
          do (setf (aref dims-1 i)
                   (1- (the fixnum (svref dims i)))))
    (loop
      (setf (aref dest-data d-ptr)
            (if caster
                (funcall caster (aref src-data s-ptr))
                (aref src-data s-ptr)))
      (let ((depth (the fixnum (1- rank))))
        (declare (type fixnum depth))
        (loop
          (when (< depth 0) (return-from %copy-strided-hi nil))
          (let ((i (the fixnum (1+ (the fixnum (aref indices depth))))))
            (declare (type fixnum i))
            (if (< i (the fixnum (svref dims depth)))
                (progn
                  (setf (aref indices depth) i)
                  (incf d-ptr (the fixnum (svref d-strs depth)))
                  (incf s-ptr (the fixnum (svref s-strs depth)))
                  (return))
                (progn
                  (setf (aref indices depth) 0)
                  (decf d-ptr (* (the fixnum (svref d-strs depth))
                                 (the fixnum (aref dims-1 depth))))
                  (decf s-ptr (* (the fixnum (svref s-strs depth))
                                 (the fixnum (aref dims-1 depth))))
                  (decf depth)))))))))

(defun %copy-generic (dest-data dest-dtype dest-strides dest-off
                      src-data src-strides src-off
                      shape size rank same-dtype)
  "通用拷贝：rank 0/1/2/3 走专门化嵌套循环；rank ≥ 4 退回里程计。
   所有路径单遍遍历，无中间缓冲。

   类型声明（性能）：DEST-DATA / SRC-DATA 的元素类型由运行时 dtype 决定，
   本函数声明里没有它们的具体类型，于是下面所有 AREF/SETF 都是**泛型**
   分派（一次 aref 要判数组类型 + 拆装箱）。这是 vt-contiguous /
   vt-copy / vt-reshape(非连续) / vt-take 等基础操作的共同瓶颈：
   实测 50 万元素 stride=2 的 vt-contiguous 需 3.0ms，numpy 0.14ms。
   同 dtype 时两条数组类型相同，可以安全地按 8 种元素类型各生成一份
   「三轴游标」内核（缺失轴用 dim=1、stride=0 补位，故一份代码覆盖
   rank 0/1/2/3）；rank ≥ 4 与跨 dtype 仍走原路径，语义不变（铁律 1/2）。"
  (declare (type fixnum size dest-off src-off rank)
           (type list shape dest-strides src-strides))
  (when (zerop size) (return-from %copy-generic nil))
  (flet ((d (i) (if (> rank i) (the fixnum (nth i shape)) 1))
         (ds (i) (if (> rank i) (the fixnum (nth i dest-strides)) 0))
         (ss (i) (if (> rank i) (the fixnum (nth i src-strides)) 0)))
    (when (and same-dtype (<= rank 3))
      (macrolet ((typed-walk (lt)
                   `(let ((dd (the (simple-array ,lt (*)) dest-data))
                          (sd (the (simple-array ,lt (*)) src-data))
                          (d0 (d 0)) (d1 (d 1)) (d2 (d 2))
                          (ds0 (ds 0)) (ds1 (ds 1)) (ds2 (ds 2))
                          (ss0 (ss 0)) (ss1 (ss 1)) (ss2 (ss 2)))
                      (declare (type (simple-array ,lt (*)) dd sd)
                               (type fixnum d0 d1 d2 ds0 ds1 ds2 ss0 ss1 ss2))
                      (let ((dp dest-off) (sp src-off))
                        (declare (type fixnum dp sp))
                        (dotimes (i0 d0)
                          (declare (type fixnum i0))
                          (let ((dp1 dp) (sp1 sp))
                            (declare (type fixnum dp1 sp1))
                            (dotimes (i1 d1)
                              (declare (type fixnum i1))
                              (let ((dp2 dp1) (sp2 sp1))
                                (declare (type fixnum dp2 sp2))
                                (dotimes (i2 d2)
                                  (declare (type fixnum i2))
                                  (setf (aref dd dp2) (aref sd sp2))
                                  (incf dp2 ds2) (incf sp2 ss2))
                                (incf dp1 ds1) (incf sp1 ss1)))
                            (incf dp ds0) (incf sp ss0)))))))
        (let ((et (array-element-type dest-data)))
          (cond ((equal et 'double-float)      (typed-walk double-float))
                ((equal et 'single-float)      (typed-walk single-float))
                ((equal et '(signed-byte 64))  (typed-walk (signed-byte 64)))
                ((equal et '(signed-byte 32))  (typed-walk (signed-byte 32)))
                ((equal et '(signed-byte 16))  (typed-walk (signed-byte 16)))
                ((equal et '(signed-byte 8))   (typed-walk (signed-byte 8)))
                ((equal et '(unsigned-byte 16)) (typed-walk (unsigned-byte 16)))
                ((equal et '(unsigned-byte 8))  (typed-walk (unsigned-byte 8)))
                ;; 复数同型：interleaved 布局下与实数同构（§16.1 铁律 3），
                ;; 缺这两格时会掉到 %COPY-GENERIC-SLOW 的泛型 aref —— 每元素
                ;; 装箱/解包，实测 1000×1000 转置视图 17ms（补格后 6ms，
                ;; 同形 f64T 是 3ms，2 倍字节数对应 2 倍耗时）。
                ((equal et '(complex double-float)) (typed-walk (complex double-float)))
                ((equal et '(complex single-float)) (typed-walk (complex single-float)))
                (t (return-from %copy-generic
                     (%copy-generic-slow dest-data dest-dtype dest-strides dest-off
                                         src-data src-strides src-off
                                         shape size rank same-dtype))))
          (return-from %copy-generic nil))))
  (%copy-generic-slow dest-data dest-dtype dest-strides dest-off
                      src-data src-strides src-off
                      shape size rank same-dtype)))

(defun %copy-generic-slow (dest-data dest-dtype dest-strides dest-off
                           src-data src-strides src-off
                           shape size rank same-dtype)
  "跨 dtype / rank ≥ 4 的通用拷贝（原 %copy-generic 实现，语义基准）。"
  (declare (type fixnum size dest-off src-off rank)
           (type list shape dest-strides src-strides))
  (when (zerop size) (return-from %copy-generic-slow nil))
  (let ((caster (unless same-dtype (vt-cast-fun dest-dtype))))
    (declare (type (or null function) caster))
    (case rank
      ;; ---------- rank 0：标量 ----------
      (0 (setf (aref dest-data dest-off)
               (if caster
                   (funcall caster (aref src-data src-off))
                   (aref src-data src-off))))

      ;; ---------- rank 1 ----------
      (1 (let* ((d0  (the fixnum (first shape)))
                (ds0 (the fixnum (first dest-strides)))
                (ss0 (the fixnum (first src-strides))))
           (declare (type fixnum d0 ds0 ss0))
           (if caster
               (loop for i fixnum from 0 below d0
                     for dp fixnum = dest-off then (+ dp ds0)
                     for sp fixnum = src-off  then (+ sp ss0)
                     do (setf (aref dest-data dp)
                              (funcall caster (aref src-data sp))))
               (loop for i fixnum from 0 below d0
                     for dp fixnum = dest-off then (+ dp ds0)
                     for sp fixnum = src-off  then (+ sp ss0)
                     do (setf (aref dest-data dp) (aref src-data sp))))))

      ;; ---------- rank 2 ----------
      (2 (let* ((d0  (the fixnum (first shape)))
                (d1  (the fixnum (second shape)))
                (ds0 (the fixnum (first dest-strides)))
                (ds1 (the fixnum (second dest-strides)))
                (ss0 (the fixnum (first src-strides)))
                (ss1 (the fixnum (second src-strides))))
           (declare (type fixnum d0 d1 ds0 ds1 ss0 ss1))
           (if caster
               (loop for i0 fixnum from 0 below d0
                     for dp0 fixnum = dest-off then (+ dp0 ds0)
                     for sp0 fixnum = src-off  then (+ sp0 ss0)
                     do (loop for i1 fixnum from 0 below d1
                              for dp fixnum = dp0 then (+ dp ds1)
                              for sp fixnum = sp0 then (+ sp ss1)
                              do (setf (aref dest-data dp)
                                       (funcall caster (aref src-data sp)))))
               (loop for i0 fixnum from 0 below d0
                     for dp0 fixnum = dest-off then (+ dp0 ds0)
                     for sp0 fixnum = src-off  then (+ sp0 ss0)
                     do (loop for i1 fixnum from 0 below d1
                              for dp fixnum = dp0 then (+ dp ds1)
                              for sp fixnum = sp0 then (+ sp ss1)
                              do (setf (aref dest-data dp)
                                       (aref src-data sp)))))))

      ;; ---------- rank 3 ----------
      (3 (let* ((d0  (the fixnum (first shape)))
                (d1  (the fixnum (second shape)))
                (d2  (the fixnum (third shape)))
                (ds0 (the fixnum (first dest-strides)))
                (ds1 (the fixnum (second dest-strides)))
                (ds2 (the fixnum (third dest-strides)))
                (ss0 (the fixnum (first src-strides)))
                (ss1 (the fixnum (second src-strides)))
                (ss2 (the fixnum (third src-strides))))
           (declare (type fixnum d0 d1 d2 ds0 ds1 ds2 ss0 ss1 ss2))
           (if caster
               (loop for i0 fixnum from 0 below d0
                     for dp0 fixnum = dest-off then (+ dp0 ds0)
                     for sp0 fixnum = src-off  then (+ sp0 ss0)
                     do (loop for i1 fixnum from 0 below d1
                              for dp1 fixnum = dp0 then (+ dp1 ds1)
                              for sp1 fixnum = sp0 then (+ sp1 ss1)
                              do (loop for i2 fixnum from 0 below d2
                                       for dp fixnum = dp1 then (+ dp ds2)
                                       for sp fixnum = sp1 then (+ sp ss2)
                                       do (setf (aref dest-data dp)
                                                (funcall caster (aref src-data sp))))))
               (loop for i0 fixnum from 0 below d0
                     for dp0 fixnum = dest-off then (+ dp0 ds0)
                     for sp0 fixnum = src-off  then (+ sp0 ss0)
                     do (loop for i1 fixnum from 0 below d1
                              for dp1 fixnum = dp0 then (+ dp1 ds1)
                              for sp1 fixnum = sp0 then (+ sp1 ss1)
                              do (loop for i2 fixnum from 0 below d2
                                       for dp fixnum = dp1 then (+ dp ds2)
                                       for sp fixnum = sp1 then (+ sp ss2)
                                       do (setf (aref dest-data dp)
                                                (aref src-data sp))))))))

      ;; ---------- rank ≥ 4：退回里程计 ----------
      (otherwise
       (%copy-strided-hi dest-data dest-dtype dest-strides dest-off
                         src-data src-strides src-off
                         shape size same-dtype)))))

;;; ------------------------------------------------------------------
;;; 拷贝入口
;;; ------------------------------------------------------------------

(defun vt-copy-into (dest src)
  "将 src 拷贝到 dest（支持广播与类型转换）。返回 dest。
   语义契约：
   1. dest 形状必须能容纳 src 广播后的形状，否则报错。
   2. dest 中「dim>1 且 stride=0」的广播维度是只读的，写入会报错。
   3. 当 dest 与 src 共享底层数组且物理区间重叠时，先对 src 做快照。"
  (setf src (ensure-vt src))
  (with-float-safe
    (let ((dest-shape (vt-shape dest))
          (src-shape  (vt-shape src)))
      ;; ---- 1) 可写性检查 ------------------------------------------------
      (when (plusp (vt-size dest))
        (loop for d in dest-shape
              for s in (vt-strides dest)
              when (and (> d 1) (zerop s))
                do (error "vt-copy-into: 目标视图是只读的广播视图（维度 ~a）" d)))
      ;; ---- 2) 形状兼容性 ------------------------------------------------
      (let ((final-shape (vt-broadcast-shapes dest-shape src-shape)))
        (unless (equal final-shape dest-shape)
          (error "vt-copy-into: dest 形状 ~a 无法容纳 src 广播后 ~a"
                 dest-shape final-shape)))
      ;; ---- 2.5) 零尺寸写入是 no-op（早退）------------------------------
      ;; 下方快路径按 (offset+size) 计算平坦边界；零尺寸视图的 offset
      ;; 可能超过底层存储长度（例如 concatenate 向 (0 2) 结果的
      ;; [:, 1:2] 空切片写入时 offset=1、存储长度=0），先返回，
      ;; 避免 replace/aref 触发序列越界（v0.3.3 空 strided 问题的写入侧）。
      (when (zerop (vt-shape-to-size dest-shape))
        (return-from vt-copy-into dest))
      ;; ---- 3) 别名保护：重叠 → 深拷贝 src 作快照 ------------------------
      (when (%vt-views-overlap-p dest src)
        (setf src (vt-copy src)))
      ;; ---- 4) 参数预计算 ------------------------------------------------
      (let* ((dest-data    (vt-data dest))
             (src-data     (vt-data src))
             (dest-dtype   (vt-dtype dest))
             (src-dtype    (vt-dtype src))
             (dest-strides (vt-strides dest))
             (src-strides  (vt-broadcast-strides
                            src-shape dest-shape (vt-strides src)))
             (dest-off     (vt-offset dest))
             (src-off      (vt-offset src))
             (size         (vt-shape-to-size dest-shape))
             (rank         (length dest-shape))
             (dest-contig  (vt-contiguous-p dest))
             (src-contig   (vt-contiguous-p src))
             (same-shape   (equal dest-shape src-shape))
             (same-dtype   (eq dest-dtype src-dtype)))
        (declare (type fixnum size dest-off src-off rank))
        (cond
          ;; 快路径 1：连续 + 同形 + 同 dtype → replace (memcpy)
          ((and dest-contig src-contig same-shape same-dtype)
           (%memcpy dest-data dest-off src-data src-off size))

          ;; 快路径 1.5：dest 连续 + 同形 + 同 dtype + src 可分解为
          ;; 「内层连续 run」（rank>2 且 run≥64；rank≤2 的行主序 walker
          ;; 内层循环已是 memcpy 级，实测无需替换）→ 逐 run replace。
          ;; 与通用路径为同一数据搬运，逐位等价（铁律 1/2）。
          ((and dest-contig same-shape same-dtype (> rank 2)
                (> (nth-value 1 (%inner-run-suffix src-shape src-strides)) 64))
           (multiple-value-bind (run-cut run-len)
               (%inner-run-suffix src-shape src-strides)
             (declare (type fixnum run-cut run-len))
             (let ((q dest-off))
               (declare (type fixnum q))
               (%map-inner-runs src-shape src-strides src-off run-cut run-len
                 (lambda (p)
                   (declare (type fixnum p))
                   (%memcpy dest-data q src-data p run-len)
                   (incf q run-len))))))

          ;; 快路径 2：连续 + 同形 + 需要类型转换
          ((and dest-contig src-contig same-shape)
           (let ((caster (vt-cast-fun dest-dtype)))
             (declare (type function caster))
             (dotimes (i size)
               (setf (aref dest-data (+ dest-off i))
                     (funcall caster (aref src-data (+ src-off i)))))))
          ;; 快路径 3：src 是标量或 size=1 → 广播填充
          ((= (vt-size src) 1)
           (vt-fill dest (aref src-data src-off)))

          ;; 通用路径：按秩专门化
          (t
           (%copy-generic dest-data dest-dtype dest-strides dest-off
                          src-data src-strides src-off
                          dest-shape size rank same-dtype)))
        dest))))

;;; ------------------------------------------------------------------
;;; 填充
;;; ------------------------------------------------------------------

(defun vt-fill (vt value)
  "用标量 value 原地填充张量 vt 的所有元素（支持视图）。返回 vt。
   广播视图（dim>1 且 stride=0 的维度）在语义上只读，写入会报错。"
  (when (plusp (vt-size vt))
    (loop for d in (vt-shape vt)
          for s in (vt-strides vt)
          when (and (> d 1) (zerop s))
            do (error "vt-fill: 目标视图是只读的广播视图（维度 ~a）" d)))
  (let* ((data (vt-data vt))
         (cval (vt-cast value (vt-dtype vt)))
         (size (vt-size vt)))
    (if (vt-contiguous-p vt)
        (let ((off (vt-offset vt)))
          (macrolet ((fill-contig (lt)
                       (let ((d (gensym "D"))
                             (v (gensym "V"))
                             (p (gensym "P"))
                             (end (gensym "END")))
                         `(let ((,d (the (simple-array ,lt (*)) data))
                                (,v (the ,lt cval))
                                (,p off)
                                (,end (the fixnum (+ off size))))
                            (declare (type (simple-array ,lt (*)) ,d)
                                     (type ,lt ,v)
                                     (type fixnum ,p ,end))
                            (loop while (< ,p ,end) do
                              (setf (aref ,d ,p) ,v)
                              (incf ,p))))))
            (let ((et (array-element-type data)))
              (cond ((equal et 'double-float)       (fill-contig double-float))
                    ((equal et 'single-float)       (fill-contig single-float))
                    ((equal et '(signed-byte 64))   (fill-contig (signed-byte 64)))
                    ((equal et '(signed-byte 32))   (fill-contig (signed-byte 32)))
                    ((equal et '(unsigned-byte 64)) (fill-contig (unsigned-byte 64)))
                    ((equal et '(unsigned-byte 32)) (fill-contig (unsigned-byte 32)))
                    ((equal et '(unsigned-byte 16)) (fill-contig (unsigned-byte 16)))
                    ((equal et '(unsigned-byte 8))  (fill-contig (unsigned-byte 8)))
                    (t
                     ;; 通用类型（t / 其他）：仅去掉索引加法
                     (let ((p off)
                           (end (the fixnum (+ off size))))
                       (declare (type fixnum p end))
                       (loop while (< p end) do
                         (setf (aref data p) cval)
                         (incf p))))))))
        ;; 非连续路径：原样保留
        (let* ((dims (coerce (vt-shape vt) 'simple-vector))
               (strs (coerce (vt-strides vt) 'simple-vector))
               (rank (length dims))
               (idx (make-array rank :element-type 'fixnum :initial-element 0))
               (ptr (vt-offset vt)))
          (when (plusp size)
            (loop
              (setf (aref data ptr) cval)
              (let ((d (1- rank)))
                (loop
                  (when (< d 0) (return-from vt-fill vt))
                  (incf (aref idx d))
                  (if (< (aref idx d) (svref dims d))
                      (progn (incf ptr (svref strs d)) (return))
                      (progn (setf (aref idx d) 0)
                             (decf ptr (* (svref strs d) (1- (svref dims d))))
                             (decf d)))))))))
    vt))
