;;;; indexing.lisp — 索引、切片、选择

(in-package :clvt)

(defun %vt-ref-raw (vt &rest indices)
  "按逻辑坐标取单个元素的**物理值**（`:bool` 也返回 0/1 字节）。

   这是 `VT-REF` 的数值内核：所有内部算术/比较/累加一律走这里，既避开
   布尔逻辑视图（T/NIL 不能进 `+`/`=`），也省掉每元素一次 dtype 判断。
   外部用户应使用 `VT-REF`（§17.8 逻辑视图）。

   性能（单元素路径，1M 次调用）：
     · 索引遍历里一次 `(dolist (idx indices))` 同时做「整数校验 + 越界校验 +
       指针推进」，取代原先三层并行 FOR 子句（每层各推一次游标）；
     · 元素读取按 `(VT-DTYPE)` 走符号 `CASE`（编译成跳转表），取代原先四次
       `(EQUAL (ARRAY-ELEMENT-TYPE data) '…)` —— 后者每次调用都要取/比较类型
       说明符，sb-sprof 里占 4.6%，且 `ARRAY-ELEMENT-TYPE` 本身是一次调用。
   本函数自身耗时基本不变（~11 ms，它的校验本来就在循环里）；去掉的是
   `VT-REF` 里那趟独立的类型预扫，见那边的注释。"
  (declare (optimize (speed 3)) (dynamic-extent indices))
  (let ((shape (vt-shape vt))
        (strides (vt-strides vt))
        (ptr (vt-offset vt))
        (n 0))
    (declare (type fixnum ptr n)
             (type list shape strides))
    (when (%shape-empty-p shape) (error "cannot index into an empty tensor"))
    (dolist (idx indices)
      (unless shape
        (error "索引数量 ~a 与维度数 ~a 不匹配"
               (length indices) (+ n (length shape))))
      (unless (integerp idx)
        (error "索引 ~a 不是整数" idx))
      (let ((dim (the fixnum (car shape)))
            (stride (the fixnum (car strides)))
            (i (the integer idx)))
        (declare (type fixnum dim stride))
        (when (minusp i) (incf i dim))
        (unless (and (>= i 0) (< i dim))
          (error "索引 ~a 越界（轴大小 ~a）" idx dim))
        (incf ptr (the fixnum (* i stride)))
        (incf n)
        (setf shape (cdr shape) strides (cdr strides))))
    (when shape
      (error "索引数量 ~a 与维度数 ~a 不匹配" n (+ n (length shape))))
    (let ((data (vt-data vt)))
      (declare (simple-array data))
      (case (vt-dtype vt)
        (:float64 (aref (the (simple-array double-float (*)) data) ptr))
        (:float32 (aref (the (simple-array single-float (*)) data) ptr))
        (:int64   (aref (the (simple-array (signed-byte 64) (*)) data) ptr))
        (:int32   (aref (the (simple-array (signed-byte 32) (*)) data) ptr))
        (:int16   (aref (the (simple-array (signed-byte 16) (*)) data) ptr))
        (:int8    (aref (the (simple-array (signed-byte 8) (*)) data) ptr))
        (:uint16  (aref (the (simple-array (unsigned-byte 16) (*)) data) ptr))
        (:uint8   (aref (the (simple-array (unsigned-byte 8) (*)) data) ptr))
        (:bool    (aref (the (simple-array (unsigned-byte 8) (*)) data) ptr))
        (:complex128 (aref (the (simple-array (complex double-float) (*)) data) ptr))
        (:complex64  (aref (the (simple-array (complex single-float) (*)) data) ptr))
        (t (aref data ptr))))))

(defun vt-ref (vt &rest indices)
  "获取指定位置元素的**逻辑值**（支持负索引，带边界检查）。

   `:bool` 张量返回 Lisp 布尔 `T` / `NIL`，与 `VT-ITEM` 一致：numpy 的
   `a[0]` 给 `np.bool_`，Lisp 的对应物就是 T/NIL。写成
   `(when (vt-ref mask i) ...)` 可直接用，不必再绕 `(= 1 ...)`。

   分层契约（CONVENTIONS §17.8）：
     · 逻辑视图 —— `VT-REF` / `VT-ITEM` / 打印 → T/NIL
     · 物理视图 —— `%VT-REF-RAW` / `VT-DATA` / `VT-TO-LIST` 默认 → 0/1 字节

   代价与 `VT-ITEM` 相同：布尔结果不能直接进算术（`(+ (vt-ref b 0) 1)` 报
   类型错误）。要算数请显式 `(if (vt-ref b 0) 1 0)`，或内部用 `%VT-REF-RAW`。

   花式索引（布尔掩码 / 索引数组）不支持（§17.10.7）：`VT-REF` 的契约是
   「取一个标量」，返回拷贝会破坏它。这里给一条明确的去路，否则用户看到的
   是 `The value #<VT shape:(3) dtype:BOOL ...> is not of type FIXNUM`。"
  (declare (dynamic-extent indices))
  ;; 单遍校验：不再用 `(find-if-not #'integerp indices)` 预扫一遍 ——
  ;; 那是一次额外的整表遍历 + 每元素一次函数调用，sb-sprof 实测占
  ;; `vt-ref` 全部耗时的 32%（%FIND-POSITION-IF-NOT 22.1% + INTEGERP 6.0%
  ;; + IDENTITY 3.8%），几乎与真正的取数一样贵。这里把校验并进
  ;; `%VT-REF-RAW` 的指针遍历，只在出错时才构造消息。
  (let ((shape (vt-shape vt))
        (strides (vt-strides vt))
        (ptr (vt-offset vt))
        (n 0))
    (declare (type fixnum ptr n)
             (type list shape strides))
    (when (%shape-empty-p shape) (error "cannot index into an empty tensor"))
    (dolist (idx indices)
      (unless (integerp idx)
        (error "vt-ref: 索引必须是整数，收到 ~a (~a)。布尔掩码请用 (vt-extract mask a) ~
                或 (vt-take a (vt-nonzero mask))；索引数组请用 vt-take。"
               idx (type-of idx)))
      (unless shape
        (error "索引数量 ~a 与维度数 ~a 不匹配"
               (length indices) (+ n (length shape))))
      (let ((dim (the fixnum (car shape)))
            (stride (the fixnum (car strides)))
            (i (the integer idx)))
        (declare (type fixnum dim stride))
        (when (minusp i) (incf i dim))
        (unless (and (>= i 0) (< i dim))
          (error "索引 ~a 越界（轴大小 ~a）" idx dim))
        (incf ptr (the fixnum (* i stride)))
        (incf n)
        (setf shape (cdr shape) strides (cdr strides))))
    (when shape
      (error "索引数量 ~a 与维度数 ~a 不匹配" n (+ n (length shape))))
    (let ((data (vt-data vt)))
      (declare (simple-array data))
      (case (vt-dtype vt)
        (:float64 (aref (the (simple-array double-float (*)) data) ptr))
        (:float32 (aref (the (simple-array single-float (*)) data) ptr))
        (:int64   (aref (the (simple-array (signed-byte 64) (*)) data) ptr))
        (:int32   (aref (the (simple-array (signed-byte 32) (*)) data) ptr))
        (:int16   (aref (the (simple-array (signed-byte 16) (*)) data) ptr))
        (:int8    (aref (the (simple-array (signed-byte 8) (*)) data) ptr))
        (:uint16  (aref (the (simple-array (unsigned-byte 16) (*)) data) ptr))
        (:uint8   (aref (the (simple-array (unsigned-byte 8) (*)) data) ptr))
        (:bool    (if (zerop (aref (the (simple-array (unsigned-byte 8) (*)) data) ptr))
                      nil t))
        (:complex128 (aref (the (simple-array (complex double-float) (*)) data) ptr))
        (:complex64  (aref (the (simple-array (complex single-float) (*)) data) ptr))
        (t (aref data ptr))))))

(defun (setf vt-ref) (value vt &rest indices)
  "设置指定位置元素（支持负索引，带边界检查）。

   `:bool` 目标接受 `T`/`NIL` 与任意数值（按 `%BOOL-FLAG` 真值化：非零→1、
   零→0、NaN→1），与 `VT-FILL` / `VT-PUT` / `VT-CAST` 的布尔语义一致；
   其余 dtype 走 `COERCE`。原实现只写 `COERCE`，于是
   `(setf (vt-ref b 0) t)` 直接报 'T can't be converted to (UNSIGNED-BYTE 8)',
   布尔化 `VT-REF` 之后这会变成最自然的写法，必须一并支持。

   H4 写入契约：目标若是只读的广播视图（存在 dim>1 且 stride=0 的轴），
   禁止写入 —— 该轴上的所有逻辑位置共享同一物理单元，写入语义歧义。
   对标 numpy：np.broadcast_to 的结果只读，赋值抛 ValueError。
   注：vt-put / vt-copy-into 早已具备该保护，本函数与 (setf vt-slice)
   的标量分支此前缺失，会写穿并污染共享底层数据。"
  (declare (optimize (speed 3)) (dynamic-extent indices))
  (unless (vt-out-writable-p vt)
    (error "vt-ref: 目标是只读的广播视图（存在 dim>1 且 stride=0 的轴），禁止写入（H4 契约）"))
  (let ((shape (vt-shape vt))
        (strides (vt-strides vt))
        (ptr (vt-offset vt))
        (n 0))
    (declare (type fixnum ptr n)
             (type list shape strides))
    (when (%shape-empty-p shape) (error "cannot index into an empty tensor"))
    (dolist (idx indices)
      (unless shape
        (error "索引数量 ~a 与维度数 ~a 不匹配"
               (length indices) (+ n (length shape))))
      (let ((dim (the fixnum (car shape)))
            (stride (the fixnum (car strides)))
            (i (the integer idx)))
        (declare (type fixnum dim stride))
        (when (minusp i) (incf i dim))
        (unless (and (>= i 0) (< i dim))
          (error "索引 ~a 越界（轴大小 ~a）" idx dim))
        (incf ptr (the fixnum (* i stride)))
        (incf n)
        (setf shape (cdr shape) strides (cdr strides))))
    (when shape
      (error "索引数量 ~a 与维度数 ~a 不匹配" n (+ n (length shape))))
    (let* ((data (vt-data vt))
           (dtype (vt-dtype vt))
           ;; 慢路径只需要判定「值不是普通数（T/NIL/复数）」—— :bool 目标的
           ;; 真值化必须逐分支做，因为它与目标存储类型绑定。
           (slow (or (eq value t) (null value) (complexp value))))
      (declare (simple-array data) (type boolean slow))
      ;; 每个 dtype 分支自带 COERCE：原实现先 `(coerce value (vt-element-type vt))`
      ;; 再按 `ARRAY-ELEMENT-TYPE` 分派 —— `VT-ELEMENT-TYPE` →
      ;; `VT-DTYPE->LISP-TYPE` 里有两次 `(member dtype *vt-dtypes*)` 线性扫描
      ;; （sb-sprof：%MEMBER 15% + VT-DTYPE->LISP-TYPE 20%），而 coerce 的
      ;; 目标类型本来就已经由分派给出。
      (macrolet ((branch (ctype &optional bool-p)
                   `(flet ((store (v)
                             (setf (aref (the (simple-array ,ctype (*)) data) ptr) v)))
                      ,(if bool-p
                           `(store (%bool-flag value))
                           `(if slow (store (vt-cast value dtype))
                                 (store (coerce value ',ctype)))))))
        (case dtype
          (:float64 (branch double-float))
          (:float32 (branch single-float))
          (:int64   (branch (signed-byte 64)))
          (:int32   (branch (signed-byte 32)))
          (:int16   (branch (signed-byte 16)))
          (:int8    (branch (signed-byte 8)))
          (:uint16  (branch (unsigned-byte 16)))
          (:uint8   (branch (unsigned-byte 8)))
          (:bool    (branch (unsigned-byte 8) t))
          (:complex128 (branch (complex double-float)))
          (:complex64  (branch (complex single-float)))
          (t (setf (aref data ptr)
                   (if slow (vt-cast value dtype) (coerce value (vt-element-type vt))))))))))

(defun %vt-unbox (vt)
  "单元素张量的物理值解包；`:bool` 归一成 Lisp 布尔（§17.8）。"
  (let ((v (aref (vt-data vt) (vt-offset vt))))
    (if (eq (vt-dtype vt) :bool) (if (zerop v) nil t) v)))

(defun vt-item (x)
  "单元素张量解包为原生标量（对标 numpy 的 .item()：任意秩、size=1）。

   `:bool` 张量解包成 **Lisp 布尔 `T` / `NIL`**：numpy 的 `.item()` 返回
   Python `bool`，Lisp 的对应物就是 T/NIL。这样
   `(when (vt-item (vt-all m)) ...)` 直接可用，不必再绕 `(= 1 ...)`。
   `VT-REF` 给同样的逻辑视图；物理视图（`%VT-REF-RAW` / `VT-DATA` /
   `VT-TO-LIST` 默认）保持 0/1 字节，分层契约见 CONVENTIONS §17.8。

   v0.4.x 修复：原实现仅解包 0 维张量，(1,1) 等二维单元素张量被原样
   返回而非标量；现按 data[offset] 直取（零索引元素物理位置即 offset）。
   多元素张量报错（numpy.item 同样拒绝）；非张量原样返回。"
  (if (vt-p x)
      (progn
        (unless (= (vt-size x) 1)
          (error "vt-item: 张量含 ~d 个元素，仅 size=1 可解包" (vt-size x)))
        (%vt-unbox x))
      x))

(declaim (inline %item-number))
(defun %item-number (x)
  "`VT-ITEM` 的结果转成数值：T→1、NIL→0，其余原样。

   给「要算数但可能收到布尔标量」的消费方用（如 `VT-BINCOUNT` 取最大值：
   numpy 接受布尔输入，`np.bincount([True False])` → `[1 1]`）。
   `:bool` 的逻辑视图是 T/NIL（§17.8），算术前必须过这一层。"
  (let ((v (vt-item x)))
    (cond ((eq v t) 1) ((null v) 0) (t v))))

(defun vt-slice (vt &rest specs)
  "统一切片接口。每个 spec 为列表：
   (:all)/(t) 全选；(:newa) 新轴；(:elli) 省略号；
   (idx) 整数索引降维；(start end &optional step) 范围切片。

   注意 spec 是**列表**，不是关键字：`(vt-slice a '(0 2))` 取 [0:2]，
   `(vt-slice a '(nil nil -1))` 反转。没有 :start/:stop/:step 这种写法。"
  (declare (optimize (speed 3)))
  ;; 关键字误用单独拦：`(vt-slice a :start 0 :stop 2)` 会变成四个位置参数
  ;; (:START 0 :STOP 2)，走到下面报「索引数 4 超过秩 1」—— 用户完全无法
  ;; 从这个消息反推出「应该写成列表」。而这是从 numpy / Clojure 迁过来时
  ;; 最容易踩的一步。
  (dolist (s specs)
    (when (keywordp s)
      (error "vt-slice: 切片说明必须是**列表**，不是关键字（收到 ~s 之类）。~%~
              写法：(vt-slice vt '(start stop &optional step))，~%~
                例 (vt-slice a '(0 2)) 取 [0:2]、(vt-slice a '(nil nil -1)) 反转、~
                (vt-slice a '(:all)) 全选。不支持 :start/:stop/:step 关键字。" s)))
  (let* ((old-shape   (vt-shape vt))
         (old-strides (vt-strides vt))
         (old-offset  (vt-offset vt))
         ;; 空输入 → 输出必然也是空；此时 offset 的任何位移都会越出（长度同样
         ;; 为 0 的）缓冲区，故一律原样保留。见 §17.10.23 / vt-flip 同款注释。
         (empty-in    (%shape-empty-p old-shape))
         (old-rank    (length old-shape))
         (osv  (coerce old-shape   'simple-vector))
         (ostv (coerce old-strides 'simple-vector))
         (n-specs (length specs)))
    (let ((elli-seen nil)
          (num-new 0)
          (num-index 0))
      (declare (fixnum num-new num-index))
      (dolist (s specs)
        (when (consp s)
          (let ((k (car s)))
            (cond ((eq k :elli)
                   (when elli-seen
                     (error "vt-slice: :elli 只能出现一次"))
                   (setf elli-seen t))
                  ((eq k :newa) (incf num-new))
                  ((and (null (cdr s)) (integerp k)) (incf num-index))))))
      (let* ((consuming (- n-specs (if elli-seen 1 0) num-new))
             (out-rank (- (+ old-rank num-new) num-index)))
        (when (> consuming old-rank)
          (error "vt-slice: 索引数 ~d 超过秩 ~d" consuming old-rank))
        (let ((needed (- old-rank consuming)))
          (declare (fixnum needed))
          (let ((new-shape   (make-array out-rank :element-type 'fixnum))
                (new-strides (make-array out-rank :element-type 'fixnum)))
            (declare (type (simple-array fixnum (*)) new-shape new-strides))
            (let ((out-i 0)
                  (axis-idx 0)
                  (cur-offset old-offset))
              (declare (fixnum out-i axis-idx cur-offset))
              (labels
                  ((emit-n-all (n)
                     (declare (fixnum n))
                     (dotimes (j n)
                       (declare (ignorable j))
                       (when (>= axis-idx old-rank)
                         (error "vt-slice: axis ~d 越界" axis-idx))
                       (setf (aref new-shape   out-i) (the fixnum (svref osv  axis-idx)))
                       (setf (aref new-strides out-i) (the fixnum (svref ostv axis-idx)))
                       (incf out-i)
                       (incf axis-idx)))
                   (emit-newa ()
                     (setf (aref new-shape   out-i) 1
                           (aref new-strides out-i) 0)
                     (incf out-i))
                   (emit-all-axis ()
                     (when (>= axis-idx old-rank)
                       (error "vt-slice: axis ~d 越界" axis-idx))
                     (setf (aref new-shape   out-i) (the fixnum (svref osv  axis-idx)))
                     (setf (aref new-strides out-i) (the fixnum (svref ostv axis-idx)))
                     (incf out-i)
                     (incf axis-idx))
                   (emit-index (idx)
                     (when (>= axis-idx old-rank)
                       (error "vt-slice: axis ~d 越界" axis-idx))
                     (let ((dim    (the fixnum (svref osv  axis-idx)))
                           (stride (the fixnum (svref ostv axis-idx)))
                           (i      (the fixnum idx)))
                       (declare (fixnum dim stride i))
                       (when (minusp i) (incf i dim))
                       (unless (and (>= i 0) (< i dim))
                         (error "vt-slice: index ~d out of bounds for axis ~d (size ~d)"
                                idx axis-idx dim))
                       (incf cur-offset (the fixnum (* i stride)))
                       (incf axis-idx)))
                   (emit-range (start end step)
                     (declare (type (or null fixnum) start end)
                              (fixnum step))
                     (when (>= axis-idx old-rank)
                       (error "vt-slice: axis ~d 越界" axis-idx))
                     (when (= 0 step)
                       (error "vt-slice: slice step cannot be zero"))
                     (let ((dim    (the fixnum (svref osv  axis-idx)))
                           (stride (the fixnum (svref ostv axis-idx))))
                       (declare (fixnum dim stride))
                       (let ((s (when start
                                  (let ((s (the fixnum (truncate start))))
                                    (if (minusp s) (the fixnum (+ s dim)) s))))
                             (e (when end
                                  (let ((e (the fixnum (truncate end))))
                                    (if (minusp e) (the fixnum (+ e dim)) e)))))
                         (declare (type (or null fixnum) s e))
                         (if (plusp step)
                             (let* ((s* (if s (max 0 (min s dim)) 0))
                                    (e* (if e (max 0 (min e dim)) dim))
                                    (len (if (> e* s*)
                                             (ceiling (- e* s*) step)
                                             0)))
                               (declare (fixnum s* e* len))
                               (incf cur-offset (the fixnum (* s* stride)))
                               (setf (aref new-shape   out-i) len
                                     (aref new-strides out-i) (* stride step)))
                             (let* ((s* (if s (max -1 (min s (1- dim))) (the fixnum (1- dim))))
                                    (e* (if e (max -1 (min e (1- dim))) -1))
                                    (len (if (> s* e*)
                                             (ceiling (- s* e*) (- step))
                                             0)))
                               (declare (fixnum s* e* len))
                               ;; 空结果不动 offset：dim = 0 时 s* 被钳到 -1，
                               ;; (s* × stride) 会把 offset 推成负数，而 offset
                               ;; 的类型是 (unsigned-byte 44) —— 之后任何
                               ;; %MAKE-VT 都会在类型检查上炸（§17.10.23）。
                               (when (plusp len)
                                 (incf cur-offset (the fixnum (* s* stride))))
                               (setf (aref new-shape   out-i) len
                                     (aref new-strides out-i) (* stride step))))
                         (incf out-i)
                         (incf axis-idx)))))
                (dolist (spec specs)
                  (if (not (consp spec))
                      (error "vt-slice: invalid slice spec ~a" spec)
                      (let ((k (car spec)))
                        (cond
                          ((and (eq k :elli) (null (cdr spec)))
                           (emit-n-all needed))
                          ((and (eq k :newa) (null (cdr spec))) (emit-newa))
                          ((and (eq k :all)  (null (cdr spec))) (emit-all-axis))
                          ((and (eq k t)     (null (cdr spec))) (emit-all-axis))
                          ((and (null (cdr spec)) (integerp k))
                           (emit-index k))
                          ((<= 2 (length spec) 3)
                           (emit-range (first spec) (second spec)
                                       (if (cddr spec) (third spec) 1)))
                          (t (error "vt-slice: invalid slice spec ~a" spec))))))
                (unless elli-seen
                  (emit-n-all (- old-rank axis-idx))))
              (unless (= out-i out-rank)
                (error "vt-slice: 内部错误 out-i=~d out-rank=~d" out-i out-rank))
              (%make-vt :data (vt-data vt)
                        :shape   (coerce new-shape   'list)
                        :strides (coerce new-strides 'list)
                        :offset  (if empty-in old-offset cur-offset)
                        :dtype   (vt-dtype vt)))))))))

(defun (setf vt-slice) (value vt &rest specs)
  "设置切片区域值（value 可为标量或张量，支持广播）。

   H4 写入契约：广播视图整体只读。切片/降维都无法解除 stride-0 带来的
   物理共享（即便把该轴索引掉，剩余元素仍与其它逻辑位置共用单元），
   因此在**源张量**上先行拦截。
   注：此前仅张量分支经 vt-copy-into 间接触发只读检查，标量分支直接调
   (setf vt-ref) 而无保护，可写穿。"
  (unless (vt-out-writable-p vt)
    (error "vt-slice: 目标是只读的广播视图（存在 dim>1 且 stride=0 的轴），禁止写入（H4 契约）"))
  (let ((target-view (apply #'vt-slice vt specs)))
    (if (null (vt-shape target-view))
        (let ((scalar (cond ((numberp value) value)
                            ((vt-p value)
                             (if (null (vt-shape value))
                                 (aref (vt-data value) (vt-offset value))
                                 (error "cannot assign tensor ~a to scalar slice"
                                        (vt-shape value))))
                            (t (error "invalid value ~a for scalar slice" value)))))
          (setf (vt-ref target-view) scalar))
        (vt-copy-into target-view value))))

(defun vt-where (condition &optional (x nil x-p) (y nil y-p) &key out dtype)
  "条件选择，完全对标 numpy.where 的两种形式。

  两种用法：
    1. `(vt-where cond x y ...)` —— 三元选择（对标 np.where(cond, x, y)）：
       条件非零处取 X，否则取 Y，三者均支持广播。
       dtype 语义（§4.2 H3「严格相等」）：
         结果 dtype = 显式 :dtype，否则为 (promote x y) —— **不由 out 决定**。
         out 的 dtype/形状必须精确匹配结果，否则报错（错误信息以 \"vt-where\"
         开头）。
    2. `(vt-where cond)` —— 单参数形式（对标 np.where(cond)）：
       返回**按轴分组的索引元组**，等价于 `(vt-nonzero cond)`，
       即 `np.nonzero(cond)` —— 长度 = rank 的张量列表，第 d 个元素是
       非零位置在第 d 轴的坐标（形状 (n,)），全为零时各项形状为 (0,)。
       注意 numpy 文档明确 `np.where(cond)` 是
       `np.asarray(cond).nonzero()` 的简写，**不是** `np.argwhere`
       （后者把坐标堆叠成 (n, rank)）。
       （此前返回 (n, rank) 坐标即 argwhere 语义，与 numpy 契约不符；
       argwhere 请直接用 `vt-argwhere`。）

   只给 X 或只给 Y（参数不配对）时报错，对齐 numpy 的必填参数语义。"
  (let ((x-p x-p) (y-p y-p))
    (cond
      ;; 单参数形式：按轴分组的索引元组（= vt-nonzero / np.nonzero）
      ((and (not x-p) (not y-p))
       (vt-nonzero (ensure-vt condition)))
      ;; 参数不配对 → 报错
      ((not y-p)
       (error "vt-where: 单参数形式只接受 (vt-where cond)；\
若要三元选择必须同时提供 X 与 Y（当前缺少 Y）"))
      (t
       ;; 分支是「值位置」：`T`/`NIL` 在这里只能是布尔真/假（§17.10.8）。
       ;; 与 `ensure-vt` 的 `NIL` 二义性无关 —— 三元选择的两个分支各自
       ;; 只放一个值，`(vt-where b t nil)` 就是 numpy 的
       ;; `np.where(b, True, False)`（结果 :bool），不存在「空分支」这种读法。
       ;; RAW-X / RAW-Y 保存 branch 之前的原始实参：弱标量判定必须看它们
       ;; （§16.31），拿 ensure 之后的 dtype promote 会让
       ;; `(vt-where cond f32数组 1.5d0)` 升成 float64（numpy 给 float32）。
       ;; `T`/`NIL` 经 branch 变成 :bool 张量，此时按张量参与提升，
       ;; 与 `np.where(cond, arr, True)` 的规则一致。
       (let ((raw-x x) (raw-y y))
         (flet ((branch (v)
                  (if (or (eq v t) (null v))
                      (ensure-vt (if v 1 0) :dtype :bool)
                      (ensure-vt v))))
           (setf condition (ensure-vt condition)
                 x (branch x)
                 y (branch y)))
         (let* ((target-shape (vt-broadcast-shapes
                               (vt-shape condition)
                               (vt-broadcast-shapes (vt-shape x) (vt-shape y))))
                (promoted (%weak-final-dtype raw-x x raw-y y))
                (final-dtype (or dtype promoted))
                ;; 统一硬契约（H1–H4）：形状/dtype 严格相等/可写
                ;; 分配用不填充版：下面两条路都会写满每个元素
                (result (if out
                            (vt-check-out out target-shape final-dtype
                                          :op-name "vt-where")
                            (%alloc-vt target-shape final-dtype)))
                ;; 快路径前置：结果连续 + 每个输入「要么长度 1（广播），
                ;; 要么连续且形状与结果相同」，且 x/y/结果同元素类型、掩码为 :int8。
                ;; 不满足就退回 vt-map 的闭包路径（语义完全一致）。
                (fast-p (and (vt-contiguous-p result)
                             (flet ((ok (tv)
                                      (or (= (vt-size tv) 1)
                                          (and (vt-contiguous-p tv)
                                               (equal (vt-shape tv) target-shape)))))
                               (and (ok condition) (ok x) (ok y))))))
         (unless (and fast-p (%select-fast condition x y result))
           (vt-map (lambda (c a b) (if (/= c 0) a b))
                   condition x y :out result :dtype final-dtype))
         result))))))

(defun %flat-nonzero-indices (tensor)
  "TENSOR（任意 strides）展平后非零元素的**逻辑扁平下标**，返回
  `(simple-array (signed-byte 64) (*))`，长度 = 非零个数。

  这是 `vt-flatnonzero` / `vt-nonzero` / `vt-argwhere` / `vt-extract` 四条
  路径共用的**唯一**「找出非零位置」实现 —— 它们的差别只是拿到下标之后
  怎么组织输出（1-D 下标 / 按轴分解的坐标列表 / (n,rank) 坐标矩阵 /
  按下标采集值），所以收集这一层必须只有一份，否则四条路会各自退化。

  两遍扫描：先数、再落位。原实现一律 `(push i acc)` + `(nreverse acc)` ——
  每命中一次分配一个 cons（1M 元素全是非零时就是 1M 个 cons、约 16MB 垃圾），
  且逐元素 `(aref data ptr)` 的 DATA 没有元素类型声明，每次都要查 widetag
  走通用 AREF。这里把「按元素类型展开」的循环收在一处，命中时只写一个
  `(signed-byte 64)`，结果可直接 `%MAKE-VT` 包成张量，不必再过
  `vt-from-sequence`（那会对每个元素走一次 `vt-cast`，13.3 万个下标要 33ms，
  比扫描本身贵 60 倍）。

  落位缓冲用 `(signed-byte 64)` 而不是 fixnum：`vt-flatnonzero` 的结果
  dtype 就是 :int64（其物理类型），由 `%MAKE-VT` 直接包装，省掉一次转换。
  下标值本身恒非负且 < size，故对 fixnum 与 s64 都是安全范围。"
  (let* ((flat (vt-ravel tensor))
         (data (vt-data flat))
         (size (vt-size flat))
         (offset (vt-offset flat))
         (stride (or (first (vt-strides flat)) 1)))
    (declare (type fixnum size offset stride))
    (macrolet ((walk (lt)
                 `(let ((d (the (simple-array ,lt (*)) data)))
                    (declare (type (simple-array ,lt (*)) d)
                             (type fixnum size offset stride))
                    (let ((cnt 0))
                      (declare (type fixnum cnt))
                      (let ((ptr offset))
                        (declare (type fixnum ptr))
                        (dotimes (i size)
                          (unless (= (aref d ptr) 0) (incf cnt))
                          (incf ptr stride)))
                      (let ((acc (make-array cnt :element-type '(signed-byte 64)))
                            (k 0)
                            (ptr offset))
                        (declare (type (simple-array (signed-byte 64) (*)) acc)
                                 (type fixnum k ptr))
                        (dotimes (i size)
                          (unless (= (aref d ptr) 0)
                            (setf (aref acc k) i)
                            (incf k))
                          (incf ptr stride))
                        acc))))
               (walk-generic ()
                 "未知元素类型：不声明，保持通用 AREF（语义一致，只是慢）。"
                 `(let ((cnt 0))
                    (declare (type fixnum cnt))
                    (dotimes (i size)
                      (unless (= (aref data (the fixnum (+ offset (* i stride)))) 0)
                        (incf cnt)))
                    (let ((acc (make-array cnt :element-type '(signed-byte 64)))
                          (k 0))
                      (declare (type (simple-array (signed-byte 64) (*)) acc)
                               (type fixnum k))
                      (dotimes (i size)
                        (unless (= (aref data (the fixnum (+ offset (* i stride)))) 0)
                          (setf (aref acc k) i)
                          (incf k)))
                      acc))))
      (let ((et (array-element-type data)))
        (cond ((equal et 'double-float)            (walk double-float))
              ((equal et 'single-float)            (walk single-float))
              ((equal et '(signed-byte 64))        (walk (signed-byte 64)))
              ((equal et '(signed-byte 32))        (walk (signed-byte 32)))
              ((equal et '(signed-byte 16))        (walk (signed-byte 16)))
              ((equal et '(signed-byte 8))         (walk (signed-byte 8)))
              ((equal et '(unsigned-byte 16))      (walk (unsigned-byte 16)))
              ((equal et '(unsigned-byte 8))       (walk (unsigned-byte 8)))
              ((equal et '(complex double-float))  (walk (complex double-float)))
              ((equal et '(complex single-float))  (walk (complex single-float)))
              (t (walk-generic)))))))

(defun %coords-from-flat (flat shape)
  "把逻辑扁平下标向量 FLAT（**必须升序**，`%FLAT-NONZERO-INDICES` 的产物
  天然满足）按 SHAPE 分解成每轴一个 `(signed-byte 64)` 数组的列表。

  坐标用**里程计**推进，不做除法：原实现逐元素算
  `(MOD (FLOOR K STRIDE_D) DIM_D)`，秩 r 就是每元素 2r 次整数除法
  （1M 元素 × 秩 2 ≈ 300 万次除法，实测占 `vt-nonzero` 总耗时的七成）。
  这里改成「从上一个下标走到当前下标」：每一步只给末轴 +1，溢出才向前进位，
  所有元素合起来正好走 N 步，摊还 O(1)。

  秩 1 直接返回 FLAT 本身（坐标就是扁平下标，不必复制）。"
  (let* ((rank (length shape))
         (num (length flat)))
    (declare (type fixnum rank num))
    (cond
      ((zerop rank) '())
      ((= rank 1) (list flat))
      (t
       (let* ((dims (coerce shape 'simple-vector))
              (arrs (coerce (loop repeat rank
                                  collect (make-array num
                                                      :element-type '(signed-byte 64)))
                            'simple-vector))
              (col (make-array rank :element-type 'fixnum :initial-element 0))
              (cur 0)
              (last (1- rank)))
         (declare (type simple-vector dims arrs)
                  (type (simple-array fixnum (*)) col)
                  (type fixnum cur last))
         (dotimes (i num)
           (declare (type fixnum i))
           (let ((k (aref flat i)))
             (declare (type fixnum k))
             ;; 里程计从 CUR 推进到 K（K ≥ CUR，故总步数恰为 N）
             (loop while (< cur k)
                   do (incf cur)
                      (let ((d last))
                        (declare (type fixnum d))
                        (loop
                          (let ((nv (1+ (the fixnum (aref col d)))))
                            (declare (type fixnum nv))
                            (if (< nv (the fixnum (svref dims d)))
                                (progn (setf (aref col d) nv) (return))
                                (progn (setf (aref col d) 0)
                                       (decf d)
                                       (when (< d 0) (return))))))))
             ;; 按轴落位（每轴一次，带类型断言避免通用 AREF）
             (dotimes (d rank)
               (declare (type fixnum d))
               (setf (aref (the (simple-array (signed-byte 64) (*)) (svref arrs d)) i)
                     (the (signed-byte 64) (aref col d))))))
         (coerce arrs 'list))))))

(defun vt-argwhere (condition &key (dtype :int64))
  "返回非零元素坐标，形状 (n, rank)。"
  (let* ((final (if (member dtype '(:int32 :int64)) dtype :int64))
         (lisp-type (if (eq final :int32) '(signed-byte 32) '(signed-byte 64)))
         (condition (ensure-vt condition))
         (shape (vt-shape condition))
         (rank (length shape))
         (flat (%flat-nonzero-indices condition))
         (count (length flat)))
    ;; 性能（v0.4.21）：原实现用递归里程计逐元素遍历，命中一个就
    ;; `(VECTOR-PUSH-EXTEND …)` 逐分量追加进可调向量（可调向量扩容要反复
    ;; realloc + 整段拷贝），最后再整段拷进定长数组。1M 掩码实测 17.5ms
    ;; （numpy 0.76ms，差 23 倍）。
    ;; 现在复用 `%FLAT-NONZERO-INDICES`（两遍扫描 + 类型化循环）拿扁平下标，
    ;; 再逐轴分解；输出形状与元素值与原实现完全相同。
    (cond
      ((zerop count)
       (vt-zeros (list 0 rank) :dtype final))
      ((zerop rank)
       ;; 0 维：numpy 的 argwhere 对非零 0 维输入给 shape (1, 0)。
       (%make-vt :data (make-array 0 :element-type lisp-type)
                 :shape (list count 0) :strides '(0 1) :offset 0 :dtype final))
      (t
       (let ((arrays (%coords-from-flat flat shape))
             (total (* count rank)))
         (declare (type fixnum total))
         (macrolet ((fill-out (ot)
                      `(let ((out (make-array total :element-type ',ot)))
                         (declare (type (simple-array ,ot (*)) out)
                                  (type fixnum count rank total))
                         (dotimes (d rank)
                           (declare (type fixnum d))
                           (let ((arr (the (simple-array (signed-byte 64) (*))
                                           (nth d arrays))))
                             (declare (type (simple-array (signed-byte 64) (*)) arr))
                             (dotimes (i count)
                               (declare (type fixnum i))
                               (setf (aref out (the fixnum (+ (* i rank) d)))
                                     (aref arr i)))))
                         (%make-vt :data out
                                   :shape (list count rank) :strides (list rank 1)
                                   :offset 0 :dtype final))))
           (if (eq final :int32) (fill-out (signed-byte 32))
               (fill-out (signed-byte 64)))))))))

(defun vt-nonzero (condition)
  "返回非零元素在各维度的索引列表（对标 torch.nonzero(as_tuple=true)）。
   rank = 0（标量）时，返回 (list (vt-zeros '(0) :dtype :int64))：
   - 长度 1 的列表，表示「存在非零元素」；
   - 内含形状 (0) 的空索引张量，表示「0 维张量没有可索引的维度」。
   这样既能区分「非零标量」与「零标量」（后者走 (zerop num) 分支返回 nil），
   又与 NumPy 对 0 维输入的 (array([0]),) 语义在「有个元素被选中」这一点上对齐。

   性能（v0.4.21）：原实现用 `(PUSH LOGIC-IDX NZ)` 收集扁平下标（每个非零
   元素一个 cons，1M 非零 ≈ 16MB 垃圾），再 `(NREVERSE …)`，最后逐轴
   `(AREF (NTH D ARRAYS) I)` —— `NTH` 在最内层，每次写一个分量都要遍历一次
   LIST 且 AREF 走通用分派。1M float64 实测 39ms（numpy 2.7ms）。
   现在复用 `%FLAT-NONZERO-INDICES` + `%COORDS-FROM-FLAT`（都是类型化两遍
   扫描），并按轴外层循环写回 —— 输出与原实现逐元素相同。"
  (let* ((condition (if (numberp condition)
                        (ensure-vt condition)
                        condition))
         (shape (vt-shape condition))
         (rank (length shape)))
    (cond
      ;; 0 维：既有行为（非零/零都给 (list 空 (0)) ）
      ((zerop rank)
       (list (vt-zeros '(0) :dtype :int64)))
      (t
       (let* ((flat (%flat-nonzero-indices condition))
              (num (length flat)))
         (cond
           ((zerop num)
            (loop repeat rank collect (vt-zeros '(0) :dtype :int64)))
           (t
            (let ((arrays (%coords-from-flat flat shape)))
              (loop for arr in arrays
                    collect (%make-vt :data arr
                                      :shape (list num)
                                      :strides '(1)
                                      :offset 0
                                      :dtype :int64))))))))))

;;; ------------------------------------------------------------------
;;; 索引 dtype 契约
;;; ------------------------------------------------------------------

(defun %check-integer-index (subject index-vt)
  "整数索引契约检查（对标 numpy）。SUBJECT 为出错的入口名，如
\"vt-take\"、\"vt-insert 的 positions\"。

numpy 对**非空**浮点索引一律报错：
  - np.take / ndarray.put → TypeError: Cannot cast array data from
    dtype('float64') to dtype('int64') ...
  - np.choose / np.insert    → TypeError: 'float' object cannot be
    interpreted as an integer
此前各实现无条件 truncate，浮点索引被静默截断（0.5 → 0），返回看似
合理却错误的结果，故一律拒绝。

例外：**空索引**（元素数 0）不做检查，视为合法的 no-op。
理由有两点：
  1. 空索引没有任何元素会被截断，不存在静默错值风险；
  2. CL 侧表示「无索引」的自然写法 —— (vt-from-sequence '())、
     #()、(vt-zeros '(0)) —— 默认都是 float64，若一并拒绝会把
     「索引为空」这一正常用法误判为错误。
numpy 对空 *list* 同样放行（np.take(a, []) → 空结果、a.put([], [])
→ no-op、np.insert(a, [], v) → 原样返回）；它只对空 float64 *数组*
因 casting 规则报错，那是 casting 的副作用而非有意契约，这里统一
放宽为放行。"
  (when (and (> (vt-size index-vt) 0)
             (vt-float-dtype-p (vt-dtype index-vt)))
    (error "~a: 索引必须为整数类型，收到 ~a（对标 numpy：非空浮点索引抛 TypeError/IndexError；空索引不受此限制）"
           subject (vt-dtype index-vt))))

(defun %index->int64 (vt)
  "把索引张量摊成 (simple-array (signed-byte 64) (*))，供 gather 循环零开销读取。
   浮点索引按 truncate 收敛（调用方已由 %check-integer-index 校验过类型）。

   v0.4.6 性能修复：原实现是 vt-take 内部的 labels，且 DATA 没有元素类型，
   逐元素 (aref data (+ off i)) 走通用 AREF（二维分派 + 边界检查）——
   500k 个 int64 索引实测约 7ms，而裸类型化循环只要 0.3ms。
   另外绝大多数调用方的索引**本来就是 int64 连续**，那种情形直接零拷贝
   返回原数组（下面第一个分支），连 8MB 复制都省掉。"
  (let* ((len (vt-size vt))
         (off (vt-offset vt))
         (data (vt-data vt)))
    (if (vt-contiguous-p vt)
        (let ((et (array-element-type data)))
          (cond
            ;; ① 已是 int64 且 offset=0 → 零拷贝
            ((and (equal et '(signed-byte 64)) (zerop off))
             data)
            ;; ② 按元素类型特化地搬一遍
            (t
             (let ((arr (make-array len :element-type '(signed-byte 64))))
               (declare (type (simple-array (signed-byte 64) (*)) arr))
               (macrolet ((conv (lt form)
                            `(let ((d (the (simple-array ,lt (*)) data)))
                               (declare (type (simple-array ,lt (*)) d)
                                        (type (simple-array (signed-byte 64) (*)) arr)
                                        (type fixnum len off))
                               (dotimes (i len)
                                 (declare (type fixnum i))
                                 (setf (aref arr i) ,form)))))
                 (cond
                   ((equal et 'double-float)      (conv double-float (truncate (aref d (+ off i)))))
                   ((equal et 'single-float)      (conv single-float (truncate (aref d (+ off i)))))
                   ((equal et '(signed-byte 64))  (conv (signed-byte 64) (aref d (+ off i))))
                   ((equal et '(signed-byte 32))  (conv (signed-byte 32) (aref d (+ off i))))
                   ((equal et '(signed-byte 16))  (conv (signed-byte 16) (aref d (+ off i))))
                   ((equal et '(signed-byte 8))   (conv (signed-byte 8) (aref d (+ off i))))
                   ((equal et '(unsigned-byte 64)) (conv (unsigned-byte 64) (aref d (+ off i))))
                   ((equal et '(unsigned-byte 32)) (conv (unsigned-byte 32) (aref d (+ off i))))
                   ((equal et '(unsigned-byte 16)) (conv (unsigned-byte 16) (aref d (+ off i))))
                   ((equal et '(unsigned-byte 8))  (conv (unsigned-byte 8) (aref d (+ off i))))
                   ;; 防御：其他元素类型（含字符型）→ 通用路径
                   (t (dotimes (i len)
                        (declare (type fixnum i))
                        (let ((v (aref data (+ off i))))
                          (setf (aref arr i)
                                (if (floatp v) (truncate v) v)))))))
               arr))))
        ;; 通用路径：stride 驱动游走（非连续索引）
        (let ((arr (make-array len :element-type '(signed-byte 64)))
              (shape (vt-shape vt))
              (strides (vt-strides vt))
              (rank (length (vt-shape vt))))
          (declare (type (simple-array (signed-byte 64) (*)) arr)
                   (type fixnum rank off))
          (labels ((walk (depth ptr out-i)
                     (declare (type fixnum depth ptr out-i))
                     (if (= depth rank)
                         (let ((v (aref data ptr)))
                           (setf (aref arr out-i)
                                 (if (floatp v) (truncate v) v))
                           (the fixnum (1+ out-i)))
                         (let ((dim (nth depth shape))
                               (str (nth depth strides)))
                           (dotimes (i dim)
                             (setf out-i (walk (1+ depth) (+ ptr (* i str)) out-i)))
                           out-i))))
            (walk 0 off 0))
          arr))))

(defun vt-take (tensor indices &key axis)
  "按索引取值（axis 为 nil 时展平取值，否则沿指定轴 gather）。

   INDICES 必须为整数：非空浮点索引报错，对标 numpy
   （IndexError: arrays used as indices must be of integer
   (or boolean) type）。原实现无条件 truncate，浮点索引会被静默
   截断（如 0.9 → 0）并返回看似合理却错误的结果。
   空索引（如 '() / (vt-from-sequence '()) / (vt-zeros '(0))）合法，
   返回空结果（见 `%check-integer-index`）。"
  (let ((idx-vt (ensure-vt indices)))
    (%check-integer-index "vt-take" idx-vt))
    (if (null axis)
        (let* ((idx-vt (ensure-vt indices))
               ;; v0.4.5：用 %flat-offset0 而不是 vt-ravel —— 后者对连续输入
               ;; 返回**保留原 offset** 的视图，下面的裸 AREF 会读到基数组
               ;; 开头（实测 slice[3:6] + take(0 2) 返回 (100 300) 而非 (5 9)）。
               (flat (%flat-offset0 tensor))
               (size (vt-size flat)))
          (if (null (vt-shape idx-vt))
              (let* ((raw (truncate (aref (vt-data idx-vt) (vt-offset idx-vt))))
                     (idx (if (minusp raw) (+ raw size) raw)))
                (unless (<= 0 idx (1- size)) (error "索引 ~d 越界" idx))
                (make-vt nil (aref (vt-data flat) idx) :dtype (vt-dtype tensor)))
              (let* ((idx-shape (vt-shape idx-vt))
                     (out (vt-zeros idx-shape :dtype (vt-dtype tensor)))
                     (idx-arr (%index->int64 idx-vt))
                     (idx-len (vt-size idx-vt))
                     (flat-data (vt-data flat))
                     (out-data (vt-data out)))
                ;; v0.4.5 性能修复：按输出元素类型特化的扁平 gather。
                ;; 原来 OUT-DATA / FLAT-DATA 都没有元素类型声明，(aref ...)
                ;; 与 (setf (aref ...)) 都走泛型二维分派 + 装箱算术，
                ;; 100k 索引实测 1376us（numpy.take 87us，差 15.8 倍）。
                (macrolet ((g (lt)
                             `(let ((src (the (simple-array ,lt (*)) flat-data))
                                    (dst (the (simple-array ,lt (*)) out-data))
                                    ;; v0.4.6：IDX-ARR 原先没有声明，于是
                                    ;; (aref idx-arr i) 走通用 AREF（二维分派 +
                                    ;; 边界检查）。500k 索引 gather 实测 6900us
                                    ;; （约 13.8ns/元素，机器码应在 2ns 量级）。
                                    (ia  (the (simple-array (signed-byte 64) (*))
                                              idx-arr)))
                                (declare (type (simple-array ,lt (*)) src dst)
                                         (type (simple-array (signed-byte 64) (*)) ia)
                                         (type fixnum idx-len size))
                                (dotimes (i idx-len)
                                  (declare (type fixnum i))
                                  (let* ((raw (aref ia i))
                                         (ix (if (minusp raw)
                                                 (the fixnum (+ raw size))
                                                 (the fixnum raw))))
                                    (declare (type fixnum ix))
                                    (unless (<= 0 ix (1- size))
                                      (error "索引 ~d 越界" raw))
                                    (setf (aref dst i) (aref src ix)))))))
                  (cond
                    ((equal (array-element-type out-data) 'double-float)     (g double-float))
                    ((equal (array-element-type out-data) 'single-float)     (g single-float))
                    ((equal (array-element-type out-data) '(signed-byte 64)) (g (signed-byte 64)))
                    ((equal (array-element-type out-data) '(signed-byte 32)) (g (signed-byte 32)))
                    ((equal (array-element-type out-data) '(signed-byte 16)) (g (signed-byte 16)))
                    ((equal (array-element-type out-data) '(signed-byte 8))  (g (signed-byte 8)))
                    ((equal (array-element-type out-data) '(unsigned-byte 16)) (g (unsigned-byte 16)))
                    ((equal (array-element-type out-data) '(unsigned-byte 8))  (g (unsigned-byte 8)))
                    ;; 防御：未知元素类型 → 通用路径
                    (t (let ((ia (the (simple-array (signed-byte 64) (*)) idx-arr)))
                         (declare (type (simple-array (signed-byte 64) (*)) ia))
                         (dotimes (i idx-len)
                           (declare (type fixnum i))
                           (let* ((raw (aref ia i))
                                  (ix (if (minusp raw) (+ raw size) raw)))
                             (unless (<= 0 ix (1- size)) (error "索引 ~d 越界" raw))
                             (setf (aref out-data i) (aref flat-data ix))))))))
                out)))
        (let* ((shape (vt-shape tensor))
               (rank (length shape))
               (ax (vt-normalize-axis axis rank))
               (ax-dim (nth ax shape))
               (idx-vt (ensure-vt indices))
               (idx-shape (vt-shape idx-vt))
               (idx-rank (length idx-shape))
               (out-shape (append (subseq shape 0 ax)
                                  idx-shape (subseq shape (1+ ax))))
               (out (vt-zeros out-shape :dtype (vt-dtype tensor)))
               (idx-arr (%index->int64 idx-vt))
               (idx-strides (vt-compute-strides idx-shape))
               (in-strides (vt-strides tensor))
               (out-strides (vt-strides out))
               (in-off (vt-offset tensor))
               (in-data (vt-data tensor)) (out-data (vt-data out)))
          (if (vt-contiguous-p tensor)
              ;; ---- 快路径：C 连续张量按 (outer, idx, inner) 行块 gather ----
              ;; 连续时每个 (外层组合, 索引) 对应一段长 inner 的连续行块，
              ;; 直接块拷贝；idx-arr 为行主序扁平数组，与结果 idx 块的
              ;; 行主序布局一一对应，顺序遍历即可。与通用路径（逐元素
              ;; 递归游走）为同一数据搬运，逐位一致（铁律 1/2）。
              ;; 非连续张量自动落入下方通用递归（铁律 2：回退而非报错）。
              (let* ((outer (reduce #'* (subseq shape 0 ax) :initial-value 1))
                     (inner (reduce #'* (subseq shape (1+ ax)) :initial-value 1))
                     (count (vt-size idx-vt))
                     (out-et (array-element-type out-data)))
                (declare (type fixnum outer inner count))
                (macrolet
                    ((gather (lt)
                       `(let ((src (the (simple-array ,lt (*)) in-data))
                              (dst (the (simple-array ,lt (*)) out-data))
                              (ia  (the (simple-array (signed-byte 64) (*)) idx-arr)))
                          (declare (type (simple-array ,lt (*)) src dst)
                                   (type (simple-array (signed-byte 64) (*)) ia))
                          ;; v0.4.5：SRC-ROW / DST-ROW / J / IDX 原先都没有声明，
                          ;; (+ dst-row j) 之类全部走装箱整数算术（每元素约 10ns）。
                          ;; 1000×1000 沿轴 take 实测 5304us（numpy 303us）。
                          (if (= inner 1)
                              ;; inner=1：逐元素直拷（避免每元素一次 replace 调用）
                              (dotimes (o outer)
                                (declare (type fixnum o))
                                (let ((src-row (the fixnum (+ in-off (* o ax-dim))))
                                      (dst-row (the fixnum (* o count))))
                                  (declare (type fixnum src-row dst-row))
                                  (dotimes (j count)
                                    (declare (type fixnum j))
                                    (let* ((raw (aref ia j))
                                           (idx (if (minusp raw)
                                                    (the fixnum (+ raw ax-dim))
                                                    (the fixnum raw))))
                                      (declare (type fixnum idx))
                                      (unless (<= 0 idx (1- ax-dim))
                                        (error "索引 ~d 越界，轴 ~d 大小 ~d" raw ax ax-dim))
                                      (setf (aref dst (the fixnum (+ dst-row j)))
                                            (aref src (the fixnum (+ src-row idx))))))))
                              ;; inner>1：行块 replace
                              (dotimes (o outer)
                                (declare (type fixnum o))
                                (let ((src-row (the fixnum (+ in-off (* o ax-dim inner))))
                                      (dst-row (the fixnum (* o count inner))))
                                  (declare (type fixnum src-row dst-row))
                                  (dotimes (j count)
                                    (declare (type fixnum j))
                                    (let* ((raw (aref ia j))
                                           (idx (if (minusp raw)
                                                    (the fixnum (+ raw ax-dim))
                                                    (the fixnum raw))))
                                      (declare (type fixnum idx))
                                      (unless (<= 0 idx (1- ax-dim))
                                        (error "索引 ~d 越界，轴 ~d 大小 ~d" raw ax ax-dim))
                                      (replace dst src
                                               :start1 (the fixnum (+ dst-row (* j inner)))
                                               :end1 (the fixnum (+ dst-row (* (1+ j) inner)))
                                               :start2 (the fixnum (+ src-row (* idx inner)))
                                               :end2 (the fixnum (+ src-row (* (1+ idx) inner))))))))))))
                  (cond
                    ((equal out-et 'double-float)       (gather double-float))
                    ((equal out-et 'single-float)       (gather single-float))
                    ((equal out-et '(signed-byte 64))   (gather (signed-byte 64)))
                    ((equal out-et '(signed-byte 32))   (gather (signed-byte 32)))
                    ((equal out-et '(signed-byte 16))   (gather (signed-byte 16)))
                    ((equal out-et '(signed-byte 8))    (gather (signed-byte 8)))
                    ((equal out-et '(unsigned-byte 16)) (gather (unsigned-byte 16)))
                    ((equal out-et '(unsigned-byte 8))  (gather (unsigned-byte 8)))
                    ;; 防御：未知底层元素类型 → 通用逐行 replace
                    (t
                     (let ((ia (the (simple-array (signed-byte 64) (*)) idx-arr)))
                       (declare (type (simple-array (signed-byte 64) (*)) ia))
                       (dotimes (o outer)
                         (let ((src-row (+ in-off (* o ax-dim inner)))
                               (dst-row (* o count inner)))
                           (dotimes (j count)
                             (let* ((raw (aref ia j))
                                  (idx (if (minusp raw) (+ raw ax-dim) raw)))
                             (unless (<= 0 idx (1- ax-dim))
                               (error "索引 ~d 越界，轴 ~d 大小 ~d" raw ax ax-dim))
                             (replace out-data in-data
                                      :start1 (+ dst-row (* j inner))
                                      :end1 (+ dst-row (* (1+ j) inner))
                                        :start2 (+ src-row (* idx inner))
                                        :end2 (+ src-row (* (1+ idx) inner)))))))))))
                out)
              ;; ---- 通用路径：stride 驱动递归（非连续张量）----
              (progn
                (labels ((recurse (depth in-ptr out-ptr)
                           (if (= depth rank)
                               (setf (aref out-data out-ptr) (aref in-data in-ptr))
                               (if (= depth ax)
                                   (walk-idx 0 0 in-ptr out-ptr)
                                   (let ((dim (nth depth shape))
                                         (in-str (nth depth in-strides))
                                         (out-str (nth (if (< depth ax)
                                                           depth
                                                           (+ depth idx-rank -1))
                                                       out-strides)))
                                     (dotimes (i dim)
                                       (recurse (1+ depth)
                                                (+ in-ptr (* i in-str))
                                                (+ out-ptr (* i out-str))))))))
                         (walk-idx (d idx-off in-ptr out-base)
                           (if (= d idx-rank)
                               (let* ((raw (aref idx-arr idx-off))
                                      (idx (if (minusp raw) (+ raw ax-dim) raw)))
                                 (unless (<= 0 idx (1- ax-dim))
                                   (error "索引 ~d 越界，轴 ~d 大小 ~d" raw ax ax-dim))
                                 (recurse (1+ ax)
                                          (+ in-ptr (* idx (nth ax in-strides)))
                                          out-base))
                               (let ((dim (nth d idx-shape))
                                     (idx-str (nth d idx-strides))
                                     (out-str (nth (+ ax d) out-strides)))
                                 (dotimes (i dim)
                                   (walk-idx (1+ d) (+ idx-off (* i idx-str)) in-ptr
                                             (+ out-base (* i out-str))))))))
                  (recurse 0 in-off 0))
                out)))))



(defun %put-single-value (values)
  "把 VALUES 归纳成**单个值**（成功时返回 (values V T)，否则 (values NIL NIL)）。
   与 `vt-put` 的 VAL-LIST 首元素同一个值，只是不为此摊开整条序列。"
  (cond
    ;; T / NIL 是单个布尔值，必须先于 LISTP 判断（NIL 既是假又是空表）。
    ((or (eq values t) (null values)) (values (if values 1 0) t))
    ((numberp values) (values values t))
    ((vt-p values)
     (let ((flat (vt-flatten values)))   ; VT-FLATTEN 恒为 offset=0 的副本
       (if (= 1 (vt-size flat))
           (values (aref (vt-data flat) 0) t)
           (values nil nil))))
    (t (values nil nil))))

(defun %put-flat-contiguous (tensor indices value mode)
  "`VT-PUT` 的快路径：目标行主序连续 + 只有一个值。
   返回 T 表示已写入；NIL 表示目标元素类型不认识，交回通用路径。

   连续目标上「逻辑线性下标 ≡ 物理偏移」，于是通用路径里三样东西都能提到
   循环外：多维分解（它每个元素都 `(reverse shape)` 分配两个 list）、
   `vt-cast`（每个元素一次 ECASE 分派）、以及下标/值的 LIST。语义与通用
   路径逐位一致：同一个 VT-CAST、同一套越界判定与 MODE 语义。"
  (let* ((total (vt-size tensor))
         (data (vt-data tensor))
         (offset (vt-offset tensor))
         (cval (vt-cast value (vt-dtype tensor)))
         (ia (%index->int64 (ensure-vt indices)))
         (n (length ia)))
    (declare (type fixnum total offset n))
    (when (zerop n) (return-from %put-flat-contiguous t))
    (macrolet ((writer (lt)
                 (let ((d (gensym "D")) (v (gensym "V")) (ar (gensym "IA"))
                       (i (gensym "I")) (raw (gensym "RAW")) (idx (gensym "IDX")))
                   `(let ((,d (the (simple-array ,lt (*)) data))
                          (,v (the ,lt cval))
                          (,ar (the (simple-array (signed-byte 64) (*)) ia)))
                      (declare (type (simple-array ,lt (*)) ,d)
                               (type ,lt ,v)
                               (type (simple-array (signed-byte 64) (*)) ,ar)
                               (type fixnum total offset n))
                      (dotimes (,i n)
                        (declare (type fixnum ,i))
                        (let* ((,raw (aref ,ar ,i))
                               (,idx (cond ((and (>= ,raw 0) (< ,raw total)) ,raw)
                                           ((eq mode :clip)
                                            (max 0 (min ,raw (1- total))))
                                           ((eq mode :wrap) (mod ,raw total))
                                           ((and (minusp ,raw) (>= ,raw (- total)))
                                            (+ ,raw total))
                                           (t (error "索引 ~d 超出范围 [0, ~d)"
                                                     ,raw total)))))
                          (declare (type fixnum ,idx))
                          (setf (aref ,d (the fixnum (+ offset ,idx))) ,v)))
                      t))))
      (let ((et (array-element-type data)))
        (cond ((equal et 'double-float) (writer double-float))
              ((equal et 'single-float) (writer single-float))
              ((equal et '(signed-byte 64)) (writer (signed-byte 64)))
              ((equal et '(signed-byte 32)) (writer (signed-byte 32)))
              ((equal et '(signed-byte 16)) (writer (signed-byte 16)))
              ((equal et '(signed-byte 8)) (writer (signed-byte 8)))
              ((equal et '(unsigned-byte 16)) (writer (unsigned-byte 16)))
              ((equal et '(unsigned-byte 8)) (writer (unsigned-byte 8)))
              (t nil))))))

(defun %put-flat-multi-1to1 (tensor idxs vals mode)
  "`VT-PUT` 的「连续目标 + 索引与值一一对应」快路径。已写入返回 T。
   识别不了目标元素类型时返回 NIL，交回通用路径。

   IDXS 是 `%INDEX->INT64` 产出的 (SIGNED-BYTE 64) 数组（负值尚未折算），
   VALS 是与 IDXS **等长**的 simple-vector。

   与通用路径逐位一致：同一个 `VT-CAST`、同一套越界判定与 MODE 语义，
   连判定顺序都照抄（`:clip` 必须先于负数折算，见通用路径的说明）。
   省掉的是「多维分解」—— 行主序连续目标上逻辑线性下标就等于物理偏移，
   不需要每个元素再 `(reverse shape)` 一遍做逐维除法。"

  (let* ((total (vt-size tensor))
         (data (vt-data tensor))
         (offset (vt-offset tensor))
         (dt (vt-dtype tensor))
         (n (length idxs)))
    (declare (type fixnum total offset n))
    (when (zerop n) (return-from %put-flat-multi-1to1 t))
    (macrolet ((writer (lt)
                 (let ((d (gensym "D")) (ar (gensym "AR")) (va (gensym "VA"))
                       (i (gensym "I")) (raw (gensym "RAW")) (idx (gensym "IDX")))
                   `(let ((,d (the (simple-array ,lt (*)) data))
                          (,ar (the (simple-array (signed-byte 64) (*)) idxs))
                          (,va (the simple-vector vals)))
                      (declare (type (simple-array ,lt (*)) ,d)
                               (type (simple-array (signed-byte 64) (*)) ,ar)
                               (type simple-vector ,va)
                               (type fixnum total offset n))
                      ;; 浮点陷阱屏蔽提到循环外：逐元素调 VT-CAST 时那层
                      ;; WITH-FLOAT-TRAPS-MASKED 要读写 FPU 控制字，实测
                      ;; 216 ns/次，会让整条写回退化成「控制字搬运」。
                      (with-float-safe
                        (dotimes (,i n)
                          (declare (type fixnum ,i))
                          (let* ((,raw (aref ,ar ,i))
                                 (,idx (cond ((and (>= ,raw 0) (< ,raw total)) ,raw)
                                             ((eq mode :clip)
                                              (max 0 (min ,raw (1- total))))
                                             ((eq mode :wrap) (mod ,raw total))
                                             ((and (minusp ,raw)
                                                   (>= ,raw (- total)))
                                              (+ ,raw total))
                                             (t (error "索引 ~d 超出范围 [0, ~d)"
                                                       ,raw total)))))
                            (declare (type fixnum ,idx))
                            (setf (aref ,d (the fixnum (+ offset ,idx)))
                                  (%vt-cast-raw (aref ,va ,i) dt))))
                        t)))))
      (let ((et (array-element-type data)))
        (cond ((equal et 'double-float) (writer double-float))
              ((equal et 'single-float) (writer single-float))
              ((equal et '(signed-byte 64)) (writer (signed-byte 64)))
              ((equal et '(signed-byte 32)) (writer (signed-byte 32)))
              ((equal et '(signed-byte 16)) (writer (signed-byte 16)))
              ((equal et '(signed-byte 8)) (writer (signed-byte 8)))
              ((equal et '(unsigned-byte 16)) (writer (unsigned-byte 16)))
              ((equal et '(unsigned-byte 8)) (writer (unsigned-byte 8)))
              (t nil))))))

(defun vt-put (tensor indices values &key (mode :raise))
  "按展平索引就地写入 values。

   INDICES 必须为整数（整数标量 / 整数张量）：非空浮点会报错，对标
   numpy（'float' object cannot be interpreted as an integer）。原实现
   用 round/truncate 静默截断浮点索引。
   空索引（如 '() / (vt-from-sequence '()) / (vt-zeros '(0))）合法，
   为空操作（见 `%check-integer-index`）。

   性能（v0.4.20 补漏）：通用路径把下标逐个 truncate 收进 LIST、值也收进
   LIST，再对每个元素做多维分解 + vt-cast —— 1M 索引实测 238 ms
   （numpy 1.2 ms）。连续目标 + 单值时走 `%put-flat-contiguous` 的线性下标
   内核，1M 实测 2.3 ms。"
  (cond
    ((and (numberp indices) (floatp indices))
     (error "vt-put: 索引必须为整数，收到 ~a（对标 numpy）" indices))
    ((not (numberp indices))
     (%check-integer-index "vt-put" (ensure-vt indices))))
  (let* ((total (vt-size tensor))
         (val-vec (cond
                     ;; T / NIL 是**单个**布尔值，必须先于 LISTP 判断：
                     ;; NIL 既是布尔假又是空表，落到下面两支会得到空
                     ;; VAL-VEC（N-VALUES=0 → VT-PUT 直接返回，静默空操作）。
                     ((or (eq values t) (null values))
                      (vector (if values 1 0)))
                     ((numberp values) (vector values))
                     ((or (listp values) (arrayp values))
                      ;; 目标为 :bool 时按布尔叶子展平：默认规则把 NIL 当空
                      ;; 子序列，`(list t nil t)` 会被压成 `(T T)`，值列表
                      ;; 短一截后由 (mod vi n-values) 循环复用 → 静默错值。
                      (coerce (vt-flatten-sequence
                               values :bool (eq (vt-dtype tensor) :bool))
                              'simple-vector))
                     ((vt-p values)
                      ;; 收进 vector 而不是 list：通用路径按 (mod vi n-values)
                      ;; 逐个取值，LIST 上是 O(n) 的 NTH —— 整条循环因此退化成
                      ;; O(n²)（1M 索引约 9 分钟）。vector 上是 O(1)。
                      (let* ((flat (vt-flatten (ensure-vt values)))
                             (m (vt-size flat))
                             (src (vt-data flat))
                             (v (make-array m)))
                        (declare (type fixnum m))
                        (dotimes (i m v)
                          (setf (aref v i) (aref src i)))))))
         (n-values (length val-vec)) (shape (vt-shape tensor))
         (strides (vt-strides tensor)) (offset (vt-offset tensor)) (data (vt-data tensor)))
    (declare (type simple-vector val-vec))
    ;; H4 写入契约（v0.4.x 修复）：dim>1 且 stride=0 的广播维度禁止就地写入，
    ;; 否则一个逻辑位置会经零步长污染整片共享底层数据。
    ;; 改用 vt-out-writable-p 而非「任一 stride=0」：vt-slice 的 (:newa)
    ;; 产生 size=1 且 stride=0 的轴，它不构成广播、写入无歧义
    ;; （对标 numpy 允许 a[np.newaxis].put(...)）；原判定会误拒该情形。
    (unless (vt-out-writable-p tensor)
      (error "vt-put: 目标包含 dim>1 且 stride=0 的广播维度，禁止写入（H4 契约）"))
    (when (zerop n-values) (return-from vt-put tensor))
    ;; ---- 快路径：连续目标 + 单值 ----
    (when (and (not (numberp indices)) (vt-contiguous-p tensor))
      (multiple-value-bind (v single-p) (%put-single-value values)
        (when single-p
          ;; 值的循环复用规则是 (mod vi n-values)，n-values = 1 时恒取同一个
          ;; 值 —— 与 VAL-VEC 首元素等价，但不必为此摊开整条序列。
          (when (%put-flat-contiguous tensor indices v mode)
            (return-from vt-put tensor)))))
    (let* ((idxs (if (numberp indices)
                     (let ((a (make-array 1 :element-type '(signed-byte 64))))
                       (setf (aref a 0) (round indices))
                       a)
                     (%index->int64 (ensure-vt indices))))
           (n-idx (length idxs)))
      (declare (type (simple-array (signed-byte 64) (*)) idxs)
               (type fixnum n-idx))
      ;; ---- 快路径：连续目标 + 一一对应 ----
      ;; 逻辑线性下标 ≡ 物理偏移，多维分解整段省掉。不满足一一对应时仍走
      ;; 通用路径 —— 那里的 (mod vi n-values) 复用才是正确语义。
      (when (and (= n-idx n-values) (vt-contiguous-p tensor))
        (when (%put-flat-multi-1to1 tensor idxs val-vec mode)
          (return-from vt-put tensor)))
      (dotimes (vi n-idx)
        (let* ((raw (aref idxs vi))
               (val (svref val-vec (mod vi n-values)))
               (idx raw))
          ;; v0.4.x 修复：负数折算（-1 → n-1）仅适用于 :raise 语义；
          ;; 原实现对所有模式先折算，导致 :clip 的 -1 被 wrap 到
          ;; n-1 而非裁剪到 0（numpy.put(clip): 负索引裁剪到 0）。
          ;; :wrap 的 (mod idx total) 对负数与折算结果一致。
          (cond ((and (>= idx 0) (< idx total)) nil)
                ((eq mode :clip) (setq idx (max 0 (min idx (1- total)))))
                ((eq mode :wrap) (setq idx (mod idx total)))
                ((and (minusp idx) (>= idx (- total))) (incf idx total))
                (t (error "索引 ~d 超出范围 [0, ~d)" raw total)))
          (let ((rem idx) (phys offset))
            (loop for dim in (reverse shape)
                  for stride in (reverse strides)
                  do (multiple-value-bind (q r) (floor rem dim)
                       (incf phys (* r stride)) (setf rem q)))
            (setf (aref data phys) (vt-cast val (vt-dtype tensor))))))
      tensor)))

(defun vt-choose (choices indices &key (mode :raise))
  "根据索引数组从多个数组中选择值。对标 numpy.choose。
   choices : 候选张量列表，长度为 n。
   indices : 整数索引张量（浮点类型报错，对标 numpy TypeError）。
   mode    : 越界索引的处理方式，对标 np.choose 的 mode 参数：
             :raise (默认) — 负数或 >= n 都报错；
             :wrap         — 索引按 n 取模（-1 -> n-1）；
             :clip         — 索引裁剪到 [0, n-1]（-1 -> 0）。
   广播规则（与 NumPy 一致）：
     indices 与所有 choices 一起参与广播，输出形状为
       broadcast(indices.shape, choices[0].shape, choices[1].shape, ...)
   输出 dtype 为所有 choices 的提升类型。
   示例：
     (vt-choose (list c0 c1) idx)
       c0 = ((1 2) (3 4)), c1 = ((10 20) (30 40)), idx = ((1 0) (1 1))
       => shape (2 2), 值 ((10 2) (30 40))
     (vt-choose (list c0 c1) idx)   ; idx = ((0) (1)) shape (2,1)
       => shape (2 2), 值 ((1 2) (30 40))   ; 与 NumPy 一致"

  (let ((n (length choices)))
    (when (zerop n)
      (error "vt-choose: choices 不能为空"))

    (let* ((idx-vt      (ensure-vt indices))
           (choice-vts  (mapcar #'ensure-vt choices))
           (final-shape (reduce #'vt-broadcast-shapes
                                (cons (vt-shape idx-vt)
                                      (mapcar #'vt-shape choice-vts))))
           (idx-b       (vt-contiguous (vt-broadcast-to idx-vt final-shape)))
           (flat-choices
             (mapcar (lambda (c)
                       (vt-contiguous (vt-broadcast-to c final-shape)))
                     choice-vts))
           (final-size  (vt-shape-to-size final-shape))
           (idx-data    (vt-data idx-b))
           (idx-off     (vt-offset idx-b))
           (result-type (apply #'vt-promote-type
                               (mapcar #'vt-dtype flat-choices)))
           (result-data (make-array final-size
                                    :element-type
                                    (vt-dtype->lisp-type result-type))))
      ;; numpy 对非空浮点 indices 抛 TypeError: 'float' object cannot be
      ;; interpreted as an integer；此前用 truncate 静默截断（0.5 → 0），
      ;; 会静默产生错误结果。空索引合法（no-op），见 `%check-integer-index`。
      (%check-integer-index "vt-choose" idx-vt)
      ;; ------------------------------------------------------------------
      ;; 性能（v0.4.21）：原循环每元素取一次 `(nth idx flat-choices)`（列表
      ;; 遍历）、两次 `(vt-data …)`/`(vt-offset …)` 访问器，以及一次
      ;; `VT-CAST` —— 而 `VT-CAST` 内部包着 `WITH-FLOAT-SAFE`（两次 FP 控制字
      ;; 读写）+ 按 dtype 的 ECASE 分派。1M 元素实测 264ms（numpy 9.5ms，
      ;; 差 28 倍），三个选择张量都是 float64，转换本身是恒等映射。
      ;;
      ;; 现在分三步，每步只对它自己的类型维度展开，**不做 in×out 全组合**
      ;; （那是 10×10 份循环体）：
      ;;   1. 索引归一：按索引元素类型展开，把 [0,n) 的结果写进 FIXNUM 数组；
      ;;   2. 取值：按**结果**元素类型展开，候选与其同型时是纯 AREF 直传；
      ;;   3. 候选与结果类型不同（混合 dtype 提升）时，退回原逐元素 VT-CAST。
      ;; 索引数据只读一遍、候选数据只读一遍，没有额外的 dtype 分派。
      ;; ------------------------------------------------------------------
      (let ((datas (coerce (mapcar #'vt-data flat-choices) 'simple-vector))
            (offs  (coerce (mapcar #'vt-offset flat-choices) 'simple-vector))
            (rlt   (vt-dtype->lisp-type result-type)))
        (declare (type simple-vector datas offs))
        (flet ((norm (raw)
                 "索引归一（:wrap/:clip）并做越界判定；返回恒在 [0, n)。"
                 (let ((idx (ecase mode
                              (:raise raw)
                              (:wrap  (mod raw n))
                              (:clip  (max 0 (min (1- n) raw))))))
                   (when (or (< idx 0) (>= idx n))
                     (error "vt-choose: 索引 ~a 越界 (choices 数量 ~a)" raw n))
                   idx)))
          (declare (inline norm))
          (if (and (> final-size 0)
                   (every (lambda (tv)
                            (equal (array-element-type (vt-data tv)) rlt))
                          flat-choices))
              (let ((nidx (make-array final-size :element-type 'fixnum)))
                (declare (type (simple-array fixnum (*)) nidx))
                ;; --- 第 1 步：索引归一（按索引元素类型展开） ---
                (macrolet ((idx-walk (ilt)
                             `(let ((idat (the (simple-array ,ilt (*)) idx-data)))
                                (declare (type (simple-array ,ilt (*)) idat)
                                         (type fixnum final-size idx-off))
                                (dotimes (i final-size)
                                  (declare (type fixnum i))
                                  (setf (aref nidx i)
                                        (norm (aref idat (the fixnum (+ idx-off i))))))))
                           (idx-generic ()
                             `(dotimes (i final-size)
                                (declare (type fixnum i))
                                (setf (aref nidx i)
                                      (norm (aref idx-data (+ idx-off i)))))))
                  (let ((iet (array-element-type idx-data)))
                    (cond ((equal iet '(signed-byte 64))   (idx-walk (signed-byte 64)))
                          ((equal iet '(signed-byte 32))   (idx-walk (signed-byte 32)))
                          ((equal iet '(signed-byte 16))   (idx-walk (signed-byte 16)))
                          ((equal iet '(signed-byte 8))    (idx-walk (signed-byte 8)))
                          ((equal iet '(unsigned-byte 16)) (idx-walk (unsigned-byte 16)))
                          ((equal iet '(unsigned-byte 8))  (idx-walk (unsigned-byte 8)))
                          (t (idx-generic)))))
                ;; --- 第 2 步：取值（按结果元素类型展开，无 VT-CAST） ---
                (macrolet ((walk (lt)
                             `(let ((od (the (simple-array ,lt (*)) result-data)))
                                (declare (type (simple-array ,lt (*)) od)
                                         (type fixnum final-size))
                                (dotimes (i final-size)
                                  (declare (type fixnum i))
                                  (let* ((idx (aref nidx i))
                                         (src (the (simple-array ,lt (*))
                                                   (svref datas idx)))
                                         (so  (the fixnum (svref offs idx))))
                                    (declare (type fixnum idx so))
                                    (setf (aref od i)
                                          (aref src (the fixnum (+ so i)))))))))
                  (cond ((equal rlt 'double-float)            (walk double-float))
                        ((equal rlt 'single-float)            (walk single-float))
                        ((equal rlt '(signed-byte 64))        (walk (signed-byte 64)))
                        ((equal rlt '(signed-byte 32))        (walk (signed-byte 32)))
                        ((equal rlt '(signed-byte 16))        (walk (signed-byte 16)))
                        ((equal rlt '(signed-byte 8))         (walk (signed-byte 8)))
                        ((equal rlt '(unsigned-byte 16))      (walk (unsigned-byte 16)))
                        ((equal rlt '(unsigned-byte 8))       (walk (unsigned-byte 8)))
                        ((equal rlt '(complex double-float))  (walk (complex double-float)))
                        ((equal rlt '(complex single-float))  (walk (complex single-float)))
                        (t (error "vt-choose: 不认识的元素类型 ~a" rlt)))))
              ;; 混合 dtype：逐元素 VT-CAST（正确性兜底）
              (loop for i from 0 below final-size
                    for idx = (norm (aref idx-data (+ idx-off i)))
                    do (setf (aref result-data i)
                             (vt-cast (aref (svref datas idx)
                                            (the fixnum (+ (svref offs idx) i)))
                                      result-type))))))
      (%make-vt :data result-data
                :shape final-shape
                :strides (vt-compute-strides final-shape)
                :offset 0
                :dtype result-type))))

(defun vt-select (condlist choicelist &key (default 0))
  "根据多个条件从多个数组中选择（对标 numpy.select）。
   result = default
   for i from n-1 downto 0:
       result = vt-where(cond_i, choice_i, result)
   condlist / choicelist 的元素可为张量或标量：标量经 ensure-vt 归一为
   0 维张量后参与广播（对标 numpy.select 允许 choicelist 含标量，
   condlist 含布尔标量），支持广播和非连续视图。
   （此前对元素直接调 vt-shape/vt-dtype，标量元素会抛出
   \"The value 99 is not of type CLVT:VT\"。）"
  (let ((n (length condlist)))
    (unless (= n (length choicelist))
      (error "condlist and choicelist must have same length"))
    (when (zerop n)
      (error "condlist and choicelist cannot be empty"))

    ;; condlist / choicelist 元素统一归一（标量 → 0 维张量），
    ;; 使标量 choice / 标量条件参与广播（对标 numpy.select）。
    (let* ((conds   (mapcar #'ensure-vt condlist))
           (choices (mapcar #'ensure-vt choicelist))
           (broadcast-shape (apply #'vt-broadcast-shapes
                                   (append (mapcar #'vt-shape conds)
                                           (mapcar #'vt-shape choices))))
           ;; 默认值类型：整数按 int64，浮点按 float64（足够安全）
           (default-dtype (if (integerp default) :int64 :float64))
           ;; 所有可能值（choicelist + 默认值）的类型提升
           (result-dtype (apply #'vt-promote-type
                                (append (mapcar #'vt-dtype choices)
                                        (list default-dtype))))
           (result (vt-full broadcast-shape default :dtype result-dtype)))
      (loop for i from (1- n) downto 0
            for cond = (nth i conds)
            for choice = (nth i choices)
            do (setf result (vt-where cond choice result)))
      result)))

(defun %gather-at-flat (flat idx limit)
  "按逻辑扁平下标 IDX 从 FLAT 采集元素，返回新 1-D 张量（dtype 同 FLAT）。
  FLAT 必须已过 `%FLAT-OFFSET0`（offset 0、stride 1）；超过 LIMIT 的下标按
  LIMIT 裁剪（numpy `take(..., mode='clip')`）。IDX 为 `(signed-byte 64)` 数组。

  为什么单独提出来：`vt-extract` 的两条分支（同形 / 需裁剪）差别只在
  下标来源，采集这一层必须只有一份，否则两条路会各自退化。
  布尔源按 `%BOOL-FLAG` 真值化写回（`:bool` 的物理类型也是 ub8，但语义是
  0/1，不能原样搬运 `:uint8` 的数据，见 §17.1）。"
  (let* ((dt (vt-dtype flat))
         (boolp (eq dt :bool))
         (n (length idx))
         (data (vt-data flat))
         (res-data (make-array n :element-type (vt-dtype->lisp-type dt))))
    (declare (type (simple-array (signed-byte 64) (*)) idx)
             (type fixnum n limit))
    (macrolet ((walk (lt &optional bool-write)
                 `(let ((d  (the (simple-array ,lt (*)) data))
                        (od (the (simple-array ,lt (*)) res-data)))
                    (declare (type (simple-array ,lt (*)) d od)
                             (type fixnum n limit))
                    (dotimes (i n)
                      (declare (type fixnum i))
                      (let ((k (aref idx i)))
                        (declare (type fixnum k))
                        (when (> k limit) (setf k limit))
                        (setf (aref od i)
                              ,(if bool-write
                                   `(the ,lt (%bool-flag (aref d k)))
                                   `(aref d k))))))))
      (let ((et (array-element-type data)))
        (cond ((equal et 'double-float)            (walk double-float))
              ((equal et 'single-float)            (walk single-float))
              ((equal et '(signed-byte 64))        (walk (signed-byte 64)))
              ((equal et '(signed-byte 32))        (walk (signed-byte 32)))
              ((equal et '(signed-byte 16))        (walk (signed-byte 16)))
              ((equal et '(signed-byte 8))         (walk (signed-byte 8)))
              ((equal et '(unsigned-byte 16))      (walk (unsigned-byte 16)))
              ((equal et '(unsigned-byte 8))
               (if boolp (walk (unsigned-byte 8) t) (walk (unsigned-byte 8))))
              ((equal et '(complex double-float))  (walk (complex double-float)))
              ((equal et '(complex single-float))  (walk (complex single-float)))
              (t (error "vt-extract: 不认识的元素类型 ~a" et)))))
    (%make-vt :data res-data :shape (list n) :strides '(1) :offset 0 :dtype dt)))

(defun vt-extract (condition tensor)
  "根据条件提取元素（对标 numpy.extract）。
   CONDITION 可为：
     - 标量（number 或 0 维 vt）：非零 → 返回 TENSOR 全部元素（展平 1-D）；
                                    为零 → 返回形状 (0) 的空张量。
     - 与 TENSOR 同形的张量：逐元素判断，提取真值位置的元素（按行主序展平）。
     - 可广播到 TENSOR 形状的张量：先广播到 TENSOR 形状，再逐元素判断。
   TENSOR 可为任意形状张量或标量（number），内部统一经 ensure-vt 归一。
   返回值：1-D 张量，dtype 与 TENSOR 相同，元素顺序为行主序。"
  (setf tensor    (ensure-vt tensor))
  (setf condition (ensure-vt condition))
  (let ((cshape (vt-shape condition))
        (tshape (vt-shape tensor)))
    (cond
      ((null cshape)
       ;; 不能写成 (if (vt-item condition) ...)：VT-ITEM 对 :bool 现在给
       ;; T/NIL，但对 :int8 掩码仍给 1/0 —— 而 CL 里 0 是真值，
       ;; (if 0 'a 'b) 会走 A。%TRUTHY-P 把两种形态统一成 Lisp 布尔（§17.8）。
       (if (%truthy-p (vt-item condition))
           (vt-flatten tensor)
           (vt-zeros (list 0) :dtype (vt-dtype tensor))))
      ((equal cshape tshape)
       (let* ((flat-cond (%flat-offset0 condition))
              (flat-data (%flat-offset0 tensor))
              (idx       (%flat-nonzero-indices flat-cond)))
         (%gather-at-flat flat-data idx (1- (vt-size flat-data)))))
      (t
       ;; numpy 语义：arr.take(condition.nonzero()[0], mode='clip')（展平）。
       ;; v0.4.x 修复：原实现把 condition 广播到 TENSOR 形状，与 numpy 的
       ;; 「条件展平非零索引 + clip 采集」不一致（如 (2,1) 条件配 (3,) 数据，
       ;; numpy 取 arr[0]，原实现广播失败报错）。
       (let* ((flat-cond (%flat-offset0 condition))
              (flat-data (%flat-offset0 tensor))
              (tsize     (vt-size flat-data))
              (idx       (%flat-nonzero-indices flat-cond)))
         (when (and (> (length idx) 0) (zerop tsize))
           (error "vt-extract: 条件含真值但 TENSOR 为空（numpy IndexError）"))
         (%gather-at-flat flat-data idx (max 0 (1- tsize))))))))

;;; ------------------------------------------------------------------
;;; searchsorted 的类型特化二分内核（v0.4.4 性能修复）
;;;
;;; 原实现是一个**没有任何类型声明**的二分循环：DATA / V-DATA 都是未声明
;;; 的数组变量，于是 (aref data mid) 走泛型二维分派 + 边界检查，比较
;;; (>= ...) 走泛型数值分派。1M 个查询 × 17 次探测 ≈ 1.7e7 次这样的
;;; 探测，实测 248ms（numpy.searchsorted 127ms，差 2.0 倍）。
;;;
;;; 改法：先把「查找表」与「查询值」归一到同一元素类型（双整型 → int64，
;;; 其余含浮点 → double-float，与 numpy 的类型提升规则一致），再进一个
;;; 全声明的特化循环 —— 每次探测就是一次带符号比较 + 两次整数位运算。
;;;
;;; 为什么不用 8×8 的元素类型组合：二分探测是**随机访存**，耗时的主体是
;;; cache miss 而不是类型分派，两种归一化已经足够；组合爆炸只会让编译期
;;; 代码量按平方增长（第 1 轮踩过：8×8 展开直接 Heap exhausted）。
;;; 精度：int64 与 int64 比较时**不**转 double，避免 >2^53 的精度丢失
;;; （numpy 此时同样按 int64 比较）。
;;; ------------------------------------------------------------------
(defun %elt-coerce-array (tensor lt)
  "把 TENSOR（须已 flatten+contiguous，offset=0）的底层数据转成 LT 元素的
   simple-array；已是该类型则原样返回（零拷贝）。"
  (let ((d (vt-data tensor)))
    (if (equal (array-element-type d) lt)
        d
        (let* ((n (vt-size tensor))
               (buf (make-array n :element-type lt)))
          (dotimes (i n)
            (declare (type fixnum i))
            (setf (aref buf i) (coerce (aref d i) lt)))
          buf))))

(macrolet ((def-searcher (name lt)
             `(defun ,name (d v nd nv out right-p)
                "对 D[0,ND) 中每个 V[0,NV) 做二分，结果写入 OUT。
                 NaN 语义对标 numpy：numpy 的二分用
                 LT(x,y) = (x < y) || (isnan(y) && !isnan(x))，
                 即把 NaN 当作**最大**元素。排好序的表里 NaN 必然只在末尾，
                 所以这里先求出「非 NaN 前缀长度 K」，只在 D[0,K) 上二分；
                 查询值为 NaN 时：left 给 K（cmp(nan,nan)=0，落在第一个 NaN 处），
                 right 给 ND（cmp(nan,任意)=+1 且 <=0 判定使其越过所有 NaN）。
                 表内无 NaN 时 K=ND，两侧与「NaN 最大」的直觉一致。"
                (declare (type (simple-array ,lt (*)) d v)
                         (type (simple-array (signed-byte 64) (*)) out)
                         (type fixnum nd nv))
                (let ((k nd))
                  (declare (type fixnum k))
                  ;; 收缩到非 NaN 前缀（整型表里 (floatp w) 恒假，编译器会消掉）
                  (loop while (and (plusp k)
                                   (let ((w (aref d (1- k))))
                                     (and (floatp w) (%nan-p w))))
                        do (decf k))
                  (if right-p
                      (dotimes (i nv)
                        (declare (type fixnum i))
                        (let ((val (aref v i)) (lo 0) (hi k))
                          (declare (type fixnum lo hi) (type ,lt val))
                          (if (and (floatp val) (%nan-p val))
                              (setf (aref out i) nd)
                              (progn
                                (loop while (< lo hi) do
                                  (let ((mid (the fixnum (ash (+ lo hi) -1))))
                                    (declare (type fixnum mid))
                                    (if (> (aref d mid) val)
                                        (setf hi mid)
                                        (setf lo (the fixnum (1+ mid))))))
                                (setf (aref out i) lo)))))
                      (dotimes (i nv)
                        (declare (type fixnum i))
                        (let ((val (aref v i)) (lo 0) (hi k))
                          (declare (type fixnum lo hi) (type ,lt val))
                          (if (and (floatp val) (%nan-p val))
                              ;; left：NaN 查询落在**第一个 NaN** 处（cmp(nan,nan)=0）
                              (setf (aref out i) k)
                              (progn
                                (loop while (< lo hi) do
                                  (let ((mid (the fixnum (ash (+ lo hi) -1))))
                                    (declare (type fixnum mid))
                                    (if (>= (aref d mid) val)
                                        (setf hi mid)
                                        (setf lo (the fixnum (1+ mid))))))
                                (setf (aref out i) lo))))))))))
  (def-searcher %searchsorted-df double-float)
  (def-searcher %searchsorted-i64 (signed-byte 64)))

(defun vt-searchsorted (tensor values &key (side :left))
  "在有序数组中查找插入点。返回形状与 VALUES 相同的 int64 张量（NumPy 语义）。
   SIDE 为 :left（默认，返回第一个 >= val 的位置）或 :right（第一个 > val）。
   v0.4.4 性能修复：改为按元素类型特化的二分内核（原 248ms → 见下）。"
  (when (%cx-any-p tensor values)
    (return-from vt-searchsorted (%cx-op-searchsorted tensor values :side side)))
  (let* ((flat (vt-contiguous (vt-flatten tensor)))
         (val-vt (ensure-vt values))
         (out-shape (vt-shape val-vt))
         (v-flat (vt-contiguous (vt-flatten val-vt)))
         (nd (vt-size flat))
         (nv (vt-size v-flat))
         (etd (array-element-type (vt-data flat)))
         (etv (array-element-type (vt-data v-flat)))
         (result (make-array nv :element-type '(signed-byte 64)))
         (right (eq side :right)))
    ;; v0.4.4 逻辑修复：查找表含 NaN 时原先直接崩 —— 原实现没有任何
    ;; with-float-safe，而 (>= nan val) 会 signal FLOATING-POINT-INVALID-OPERATION
    ;; （实测 base 版 (vt-searchsorted [1 3 NaN] [2 4]) 必崩，numpy 返回 [1 2]）。
    ;; 屏蔽陷阱后 IEEE 语义即是 numpy 语义：NaN 与任何数比较都为「无序」，
    ;; 于是 NaN 落在末尾，插入点等于其下标。
    (multiple-value-bind (lt d v)
        (if (and (subtypep etd 'integer) (subtypep etv 'integer))
            ;; 双整型：保持 int64，避免 >2^53 精度丢失
            (values '(signed-byte 64)
                    (%elt-coerce-array flat '(signed-byte 64))
                    (%elt-coerce-array v-flat '(signed-byte 64)))
            ;; 任一侧浮点：统一 double-float（与 numpy 提升规则一致）
            (values 'double-float
                    (%elt-coerce-array flat 'double-float)
                    (%elt-coerce-array v-flat 'double-float)))
      (with-float-safe
        (if (equal lt 'double-float)
            (%searchsorted-df d v nd nv result right)
            (%searchsorted-i64 d v nd nv result right))))
    (%make-vt :data result
              :shape out-shape
              :strides (vt-compute-strides out-shape)
              :offset 0 :dtype :int64)))

(defun %bincount-weighted (x weights minlength)
  "VT-BINCOUNT 的 WEIGHTS 分支。对标 numpy.bincount(x, weights=w)：
   结果 dtype **恒为 float64**，即使 weights 本身是整型或布尔也如此 ——
   numpy 文档明说「result is stored as a float64 array if weights is
   given」，累加的是权重而不是计数。WEIGHTS 与 X 的元素数必须一致，
   负数权重允许（加权直方图常见）。
   复数 weights 报错：numpy 在这里直接 TypeError（无法 cast 到 float64），
   本库同样拒绝而不是给出一个 numpy 给不出的结果。"
  (let* ((flat (vt-flatten x))
         (wf (vt-flatten (ensure-vt weights)))
         (n (vt-size flat))
         (maxval (if (zerop n) -1 (%item-number (vt-amax flat))))
         (size (max (1+ maxval) minlength)))
    (unless (= n (vt-size wf))
      (error "vt-bincount: weights 的元素数 (~a) 与 x 的元素数 (~a) 不一致（numpy 同样要求同形）"
             (vt-size wf) n))
    (when (member (vt-dtype wf) '(:complex64 :complex128))
      (error "vt-bincount: weights 不支持复数（numpy 同样报 TypeError：无法 cast 到 float64）。见 CONVENTIONS §16.11"))
    (let ((res (make-array size :element-type 'double-float :initial-element 0d0))
          (wdata (vt-data (vt-astype wf :float64))))
      (let ((i 0))
        (vt-do-each (ptr val flat)
          (declare (ignore ptr))
          (let ((idx (truncate (%item-number val))))
            ;; 不声明 FIXNUM：TRUNCATE 对大浮点可以给 BIGNUM，而声明冲突会
            ;; 把「越界」报成「The value … is not of type FIXNUM」。
            (when (or (minusp idx) (>= idx size))
              (error "vt-bincount: 索引 ~a 越界（须 0 <= x < ~a）" idx size))
            (incf (aref res idx) (aref wdata i)))
          (incf i)))
      (%make-vt :data res :shape (list size) :strides '(1)
                :offset 0 :dtype :float64))))

(defun vt-bincount (x &key (minlength 0) weights)
  "统计非负整数出现次数。

   WEIGHTS 给定时按权重累加而非计数（对标 numpy.bincount 的 weights 参数）：
   结果 dtype 恒为 :float64（即使权重本身是整型或布尔，numpy 也是如此）；
   负权重合法（加权直方图常见）；复数权重**报错** —— numpy 在这里同样抛
   TypeError（无法 cast 到 float64），本库不「出于好意」给出 complex128。
   WEIGHTS 的元素数必须等于 X 的元素数。

   示例：(vt-bincount (vt-from-sequence '(1 1 3) :dtype :int64))          ; => (0 2 0 1)
        (vt-bincount (vt-from-sequence '(1 1 3) :dtype :int64)
                     :weights (vt-from-sequence '(0.5 0.25 2.0) :dtype :float64))
        ; => (0.0 0.75 0.0 2.0)"
  (when (%cx-reject x)
    (error "vt-bincount: 不支持复数输入（numpy 同样拒绝）。见 CONVENTIONS §16.11"))
  (when weights
    (return-from vt-bincount (%bincount-weighted x weights minlength)))
  (let* ((flat (vt-flatten x))
         ;; :bool 的 VT-ITEM 给 T/NIL（§17.8），这里要的是数值：numpy 接受
         ;; 布尔输入（np.bincount([True False]) → [1 1]），走 %ITEM-NUMBER。
         (maxval (if (zerop (vt-size flat))
                     -1
                     (%item-number (vt-amax flat))))
         (size (max (1+ maxval) minlength))
         (result (make-array size :element-type '(signed-byte 64)
                                  :initial-element 0)))
    ;; v0.4.8 性能修复：原实现用 VT-DO-EACH 遍历（通用迭代器，每元素一次
    ;; 闭包/宏展开的通用取值 + TRUNCATE 泛型调用）。1M 元素实测 28277us
    ;; （numpy.bincount 3757us）。FLAT 已是 1-D 连续副本，直接按元素类型
    ;; 特化计数即可（RESULT 本来就是字面 (signed-byte 64)，无需再声明）。
    (let ((src-data (vt-data flat))
          (nsize (vt-size flat)))
      (declare (type fixnum nsize))
      (macrolet ((count-it (lt conv)
                   `(let ((src (the (simple-array ,lt (*)) src-data)))
                      (declare (type (simple-array ,lt (*)) src))
                      (dotimes (i nsize)
                        (declare (type fixnum i))
                        (let ((idx ,conv))
                          (declare (type fixnum idx))
                          (when (or (minusp idx) (>= idx size))
                            (error "vt-bincount: 索引 ~a 越界" idx))
                          (incf (aref result idx)))))))
        (let ((et (array-element-type src-data)))
          (cond
            ((equal et '(signed-byte 64))   (count-it (signed-byte 64) (aref src i)))
            ((equal et '(signed-byte 32))   (count-it (signed-byte 32) (aref src i)))
            ((equal et '(signed-byte 16))   (count-it (signed-byte 16) (aref src i)))
            ((equal et '(signed-byte 8))    (count-it (signed-byte 8) (aref src i)))
            ((equal et '(unsigned-byte 16)) (count-it (unsigned-byte 16) (aref src i)))
            ((equal et '(unsigned-byte 8))  (count-it (unsigned-byte 8) (aref src i)))
            ((equal et 'double-float)       (count-it double-float (truncate (aref src i))))
            ((equal et 'single-float)       (count-it single-float (truncate (aref src i))))
            ;; 防御：其他元素类型 → 原通用路径
            (t (vt-do-each (ptr val flat)
                 (declare (ignore ptr))
                 (let ((idx (truncate val)))
                   (when (or (minusp idx) (>= idx size))
                     (error "vt-bincount: 索引 ~a 越界" idx))
                   (incf (aref result idx)))))))))
    (%make-vt :data result :shape (list size) :strides '(1)
              :offset 0 :dtype :int64)))

(defun %bins-monotonicity (data n)
  "numpy._monotonicity：返回 1（单调不减）/ -1（单调不增）/ 0（都不是）。
   全相等、长度 0 或 1 均视为 1（与 numpy 实测一致：np.digitize 不报错）。"
  (let ((inc t) (dec t))
    (loop for i from 1 below n
          for prev = (aref data (1- i))
          for cur = (aref data i)
          do (cond ((< prev cur) (setf dec nil))
                   ((> prev cur) (setf inc nil))))
    (cond (inc 1) (dec -1) (t 0))))

;;; ------------------------------------------------------------------
;;; digitize 的二分内核（v0.4.20 性能修复）
;;;
;;; 原实现每元素一次 `(%digitize-searchsorted bin-data nbins …)'，而
;;; BIN-DATA 是 (simple-array t)，所以每次比较都要：查 widetag → 走通用
;;; AREF → 拆箱 → 泛型数值比较。1M 元素 / 1000 bins 实测 138 ms
;;; （138 ns/元素，全库最慢的一格），而同样的二分在 C 里是 3 ms 量级。
;;;
;;; 拆成两条**按比较域特化**的内核：bins 与 x 中只要有浮点就走 double
;;; 比较域（float32 → double 无损，整型 → double 与 numpy 的 result_type
;;; 提升一致），否则走 (signed-byte 64) 精确整数域。两条内核的分支结构
;;; 与原来的 `%digitize-searchsorted' 逐字相同（同 side、同 reverse、
;;; 同 `>` / `>=' 方向），只把「泛型取值」换成「定长数组 AREF」。
;;; ------------------------------------------------------------------
(declaim (inline %bisect-double %bisect-s64))
(defun %bisect-double (bins nbins v reverse-p side-right-p)
  "在 (simple-array double-float) 上二分，返回插入点（0..NBINS）。"
  (declare (type (simple-array double-float (*)) bins)
           (type fixnum nbins)
           (type double-float v)
           (type boolean reverse-p side-right-p))
  (let ((lo 0) (hi nbins))
    (declare (type fixnum lo hi))
    (loop while (< lo hi) do
      (let* ((mid (ash (+ lo hi) -1))
             (mv (aref bins (if reverse-p (- nbins 1 mid) mid))))
        (if (if side-right-p (> mv v) (>= mv v))
            (setf hi mid)
            (setf lo (1+ mid)))))
    lo))

(defun %bisect-s64 (bins nbins v reverse-p side-right-p)
  "在 (simple-array (signed-byte 64)) 上二分，比较式与 `%bisect-double' 同构。"
  (declare (type (simple-array (signed-byte 64) (*)) bins)
           (type fixnum nbins)
           (type (signed-byte 64) v)
           (type boolean reverse-p side-right-p))
  (let ((lo 0) (hi nbins))
    (declare (type fixnum lo hi))
    (loop while (< lo hi) do
      (let* ((mid (ash (+ lo hi) -1))
             (mv (aref bins (if reverse-p (- nbins 1 mid) mid))))
        (if (if side-right-p (> mv v) (>= mv v))
            (setf hi mid)
            (setf lo (1+ mid)))))
    lo))

(defun %digitize-run-double (x-data xoff xstride n bins nbins reverse-p side-right-p result)
  "双精度域：按 X-DATA 的 STRIDE 遍历，逐元素二分后写 RESULT[0..N)。"
  (declare (type (simple-array double-float (*)) x-data bins)
           (type (simple-array (signed-byte 64) (*)) result)
           (type fixnum xoff xstride n nbins)
           (type boolean reverse-p side-right-p))
  (let ((ptr xoff))
    (declare (type fixnum ptr))
    (dotimes (i n)
      (let ((ss (%bisect-double bins nbins (aref x-data ptr) reverse-p side-right-p)))
        (setf (aref result i) (if reverse-p (- nbins ss) ss)))
      (incf ptr xstride)))
  result)

(defun %digitize-run-s64 (x-data xoff xstride n bins nbins reverse-p side-right-p result)
  "整数域：同 `%digitize-run-double'，比较在 (signed-byte 64) 上精确进行。"
  (declare (type (simple-array (signed-byte 64) (*)) x-data bins)
           (type (simple-array (signed-byte 64) (*)) result)
           (type fixnum xoff xstride n nbins)
           (type boolean reverse-p side-right-p))
  (let ((ptr xoff))
    (declare (type fixnum ptr))
    (dotimes (i n)
      (let ((ss (%bisect-s64 bins nbins (aref x-data ptr) reverse-p side-right-p)))
        (setf (aref result i) (if reverse-p (- nbins ss) ss)))
      (incf ptr xstride)))
  result)

(defun %digitize-simple-contiguous-p (vt)
  "VT 是否为「偏移 0 + 一维步长 1」的连续视图（可直接当定长数组读）。"
  (and (zerop (vt-offset vt))
       (= 1 (or (first (vt-strides vt)) 1))))

(defun vt-digitize (x bins &key (right nil))
  "返回输入值在 bins 中的索引（对标 numpy.digitize）。

   X 可为任意形状：结果形状与 X **完全一致**（多维输入不展平）。
   BINS 接受张量或可直接构造为张量的序列（list / vector），
   需为一维且单调（单调不减或单调不增，numpy 语义；否则报错）。

   算法与 numpy 一致（numpy 文档：digitize 由 searchsorted 实现）：
     side = 'left' if right else 'right'
     bins 单调不减 → searchsorted(bins, x, side)
     bins 单调不增 → len(bins) - searchsorted(bins[::-1], x, side)
   即 right=False 时区间为 `bins[i-1] <= x < bins[i]`，
   right=True 时为 `bins[i-1] < x <= bins[i]`（递增情形）。

   性能（v0.4.20 补漏）：比较域按 dtype 一次选定（有浮点 → double，
   全整型 → (signed-byte 64)），X 与 BINS 各转一次到该域，随后是
   「步长遍历 + 定长数组二分」，不再逐元素走泛型 AREF。1M 元素 / 1000
   bins：138 ms → 16.0 ms（8.6 倍）。

   （此前返回 (vt-shape flat-x)，多维输入被展平为 1-D；bins 直接
   vt-flatten，传 Lisp list 会抛 \"The value (4 8) is not of type CLVT:VT\"；
   且线性扫描把递减 bins 当作递增处理，结果与 numpy 不符。
   另：原实现把结果写到 `(aref result ptr)'，PTR 是**物理**偏移 ——
   步进视图（如 (vt-slice a :step 2)）会在 size 之外写、并漏写前半段。
   现在一律按逻辑序号写，与 X 的形状严格对应。）"
  (when (%cx-any-p x bins)
    (error "vt-digitize: 复数不可分箱（numpy 对复数同样报 TypeError: x may not be complex）"))
  (let* ((x-vt (ensure-vt x))
         (bins-vt (ensure-vt bins))
         (flat-x (vt-flatten x-vt))
         (flat-bins (vt-flatten bins-vt))
         (nbins (vt-size flat-bins))
         (xn (vt-size flat-x))
         (x-shape (vt-shape x-vt))
         (mono (%bins-monotonicity (vt-data flat-bins) nbins))
         ;; numpy: side='left' if right else 'right'
         (side-right-p (not right))
         (reverse-p (= mono -1))
         (float-domain (or (vt-float-dtype-p (vt-dtype flat-x))
                           (vt-float-dtype-p (vt-dtype flat-bins))))
         (result (make-array xn :element-type '(signed-byte 64))))
    (when (zerop mono)
      (error "vt-digitize: bins 必须单调递增或递减（numpy 语义）"))
    (with-float-safe
     (ecase (if float-domain :double :s64)
      (:double
       (let* ((xs (if (eq (vt-dtype flat-x) :float64)
                      flat-x
                      (vt-astype flat-x :float64)))
              (bs (if (and (eq (vt-dtype flat-bins) :float64)
                           (%digitize-simple-contiguous-p flat-bins))
                      flat-bins
                      (vt-astype flat-bins :float64))))
         (%digitize-run-double
          (the (simple-array double-float (*)) (vt-data xs))
          (vt-offset xs)
          (or (first (vt-strides xs)) 1)
          xn
          (the (simple-array double-float (*)) (vt-data bs))
          nbins reverse-p side-right-p result)))
      (:s64
       (let* ((xs (if (eq (vt-dtype flat-x) :int64)
                      flat-x
                      (vt-astype flat-x :int64)))
              (bs (if (and (eq (vt-dtype flat-bins) :int64)
                           (%digitize-simple-contiguous-p flat-bins))
                      flat-bins
                      (vt-astype flat-bins :int64))))
         (%digitize-run-s64
          (the (simple-array (signed-byte 64) (*)) (vt-data xs))
          (vt-offset xs)
          (or (first (vt-strides xs)) 1)
          xn
          (the (simple-array (signed-byte 64) (*)) (vt-data bs))
          nbins reverse-p side-right-p result)))))
    ;; 形状取 X 的原始形状（而非展平后的 (n)）
    (%make-vt :data result :shape x-shape
              :strides (vt-compute-strides x-shape)
              :offset 0 :dtype :int64)))
