;;;; indexing.lisp — 索引、切片、选择

(in-package :clvt)

(defun vt-ref (vt &rest indices)
  "获取指定位置元素（支持负索引，带边界检查）。"
  (declare (optimize (speed 3)))
  (let* ((shape (vt-shape vt))
         (rank (length shape)))
    (declare (list shape))
    (when (some #'zerop shape) (error "cannot index into an empty tensor"))
    (unless (= (length indices) rank)
      (error "索引数量 ~a 与维度数 ~a 不匹配" (length indices) rank))
    (let ((data (vt-data vt))
          (ptr (vt-offset vt)))
      (declare (fixnum ptr)
               (simple-array data))
      (loop for idx fixnum in indices
            for dim fixnum in shape
            for stride fixnum in (vt-strides vt)
            do (let ((i (if (minusp idx) (+ idx dim) idx)))
                 (unless (and (>= i 0) (< i dim))
                   (error "索引 ~a 越界（轴大小 ~a）" idx dim))
                 (incf ptr (the fixnum (* i stride)))))
      (macrolet ((ref (lt)
                   `(aref (the (simple-array ,lt (*)) data) ptr)))
        (cond ((equal (array-element-type data) 'double-float) (ref double-float))
              ((equal (array-element-type data) 'single-float) (ref single-float))
              ((equal (array-element-type data) '(signed-byte 64)) (ref (signed-byte 64)))
              ((equal (array-element-type data) '(signed-byte 32)) (ref (signed-byte 32)))
              (t (aref data ptr)))))))

(defun (setf vt-ref) (value vt &rest indices)
  "设置指定位置元素（支持负索引，带边界检查）。

   H4 写入契约：目标若是只读的广播视图（存在 dim>1 且 stride=0 的轴），
   禁止写入 —— 该轴上的所有逻辑位置共享同一物理单元，写入语义歧义。
   对标 numpy：np.broadcast_to 的结果只读，赋值抛 ValueError。
   注：vt-put / vt-copy-into 早已具备该保护，本函数与 (setf vt-slice)
   的标量分支此前缺失，会写穿并污染共享底层数据。"
  (declare (optimize (speed 3)))
  (unless (vt-out-writable-p vt)
    (error "vt-ref: 目标是只读的广播视图（存在 dim>1 且 stride=0 的轴），禁止写入（H4 契约）"))
  (let* ((shape (vt-shape vt))
         (rank (length shape)))
    (when (some #'zerop shape) (error "cannot index into an empty tensor"))
    (unless (= (length indices) rank)
      (error "索引数量 ~a 与维度数 ~a 不匹配" (length indices) rank))
    (let ((data (vt-data vt))
          (ptr (vt-offset vt)))
      (declare (fixnum ptr)
               (simple-array data))
      (loop for idx fixnum in indices
            for dim fixnum in shape
            for stride fixnum in (vt-strides vt)
            do (let ((i (if (minusp idx) (+ idx dim) idx)))
                 (unless (and (>= i 0) (< i dim))
                   (error "索引 ~a 越界（轴大小 ~a）" idx dim))
                 (incf ptr (the fixnum (* i stride)))))
      (let ((cval (coerce value (vt-element-type vt))))
        (macrolet ((set-ref (lt)
                     `(setf (aref (the (simple-array ,lt (*)) data) ptr) cval)))
          (cond ((equal (array-element-type data) 'double-float) (set-ref double-float))
                ((equal (array-element-type data) 'single-float) (set-ref single-float))
                ((equal (array-element-type data) '(signed-byte 64)) (set-ref (signed-byte 64)))
                ((equal (array-element-type data) '(signed-byte 32)) (set-ref (signed-byte 32)))
                (t (setf (aref data ptr) cval))))))))

(defun vt-item (x)
  "单元素张量解包为原生数字（对标 numpy 的 .item()：任意秩、size=1）。
   v0.4.x 修复：原实现仅解包 0 维张量，(1,1) 等二维单元素张量被原样
   返回而非标量；现按 data[offset] 直取（零索引元素物理位置即 offset）。
   多元素张量报错（numpy.item 同样拒绝）；非张量原样返回。"
  (cond ((and (vt-p x) (null (vt-shape x))) (vt-ref x))
        ((vt-p x)
         (unless (= (vt-size x) 1)
           (error "vt-item: 张量含 ~d 个元素，仅 size=1 可解包" (vt-size x)))
         (aref (vt-data x) (vt-offset x)))
        (t x)))

(defun vt-slice (vt &rest specs)
  "统一切片接口。每个 spec 为列表：
   (:all)/(t) 全选；(:newa) 新轴；(:elli) 省略号；
   (idx) 整数索引降维；(start end &optional step) 范围切片。"
  (declare (optimize (speed 3)))
  (let* ((old-shape   (vt-shape vt))
         (old-strides (vt-strides vt))
         (old-offset  (vt-offset vt))
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
                               (incf cur-offset (the fixnum (* s* stride)))
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
                        :offset  cur-offset
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
       (setf condition (ensure-vt condition)
             x (ensure-vt x)
             y (ensure-vt y))
       (let* ((target-shape (vt-broadcast-shapes
                             (vt-shape condition)
                             (vt-broadcast-shapes (vt-shape x) (vt-shape y))))
              (promoted (vt-promote-type (vt-dtype x) (vt-dtype y)))
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
         result)))))

(defun vt-argwhere (condition &key (dtype :int64))
  "返回非零元素坐标，形状 (n, rank)。"
  (let* ((final (if (member dtype '(:int32 :int64)) dtype :int64))
         (lisp-type (if (eq final :int32) '(signed-byte 32) '(signed-byte 64)))
         (shape (vt-shape condition))
         (rank (length shape))
         (data (vt-data condition))
         (offset (vt-offset condition))
         (shape-vec (coerce shape 'simple-vector))
         (strides-vec (coerce (vt-strides condition) 'simple-vector))
         (result (make-array 64 :element-type '(signed-byte 64)
                                :fill-pointer 0 :adjustable t))
         (coord (make-array rank :element-type '(signed-byte 64))))
    (labels ((recurse (depth ptr)
               (if (= depth rank)
                   (when (/= (aref data ptr) 0)
                     (if (zerop rank)
                         (vector-push-extend 0 result)
                         (loop for c across coord
                               do (vector-push-extend c result))))
                   (let ((dim (svref shape-vec depth))
                         (stride (svref strides-vec depth)))
                     (loop for i from 0 below dim do
                       (setf (aref coord depth) i)
                       (recurse (1+ depth) (+ ptr (* i stride))))))))
      (recurse 0 offset))
    (let* ((total (length result))
           (count (if (zerop rank) total (floor total rank))))
      (cond ((zerop count)
             (vt-zeros (list 0 rank) :dtype final))
            ((> rank 0)
             (let ((final-data (make-array total :element-type lisp-type)))
               (loop for i from 0 below total
                     for idx across result
                     do (setf (aref final-data i) idx))
               (%make-vt :data final-data :shape (list count rank) :strides (list rank 1)
                         :offset 0 :dtype final)))
            (t (%make-vt :data (make-array 0 :element-type lisp-type)
                         :shape (list count 0) :strides '(0 1) :offset 0 :dtype final))))))

(defun vt-nonzero (condition)
  "返回非零元素在各维度的索引列表（对标 torch.nonzero(as_tuple=true)）。
   rank = 0（标量）时，返回 (list (vt-zeros '(0) :dtype :int64))：
   - 长度 1 的列表，表示「存在非零元素」；
   - 内含形状 (0) 的空索引张量，表示「0 维张量没有可索引的维度」。
   这样既能区分「非零标量」与「零标量」（后者走 (zerop num) 分支返回 nil），
   又与 NumPy 对 0 维输入的 (array([0]),) 语义在「有个元素被选中」这一点上对齐。"
  (let* ((condition (if (numberp condition)
                        (ensure-vt condition)
                        condition))
         (shape (vt-shape condition))
         (rank (length shape))
         (data (vt-data condition))
         (offset (vt-offset condition))
         (in-strides (vt-strides condition))
         (logic-strides (vt-compute-strides shape))
         (nz '()))
    (labels ((recurse (depth logic-idx ptr)
               (if (= depth rank)
                   (unless (zerop (aref data ptr)) (push logic-idx nz))
                   (let ((dim (nth depth shape))
                         (in-stride (nth depth in-strides))
                         (logic-stride (nth depth logic-strides)))
                     (loop for i from 0 below dim do
                       (recurse (1+ depth)
                                (+ logic-idx (* i logic-stride))
                                (+ ptr (* i in-stride))))))))
      (recurse 0 0 offset))
    (setf nz (nreverse nz))
    (let ((num (length nz)))
      (cond
        ((zerop rank)
         (list (vt-zeros '(0) :dtype :int64)))
        ((zerop num)
         (loop repeat rank collect (vt-zeros '(0) :dtype :int64)))
        (t
         (let ((arrays (loop repeat rank
                             collect (make-array num
                                                 :element-type '(signed-byte 64)))))
           (loop for flat in nz for i from 0 do
             (let ((rem flat))
               (loop for d from 0 below rank
                     for stride in logic-strides do
                 (let ((coord (floor rem stride)))
                   (setf (aref (nth d arrays) i) coord)
                   (decf rem (* coord stride))))))
           (loop for arr in arrays
                 collect (%make-vt :data arr
                                   :shape (list num)
                                   :strides '(1)
                                   :offset 0
                                   :dtype :int64))))))))

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



(defun vt-put (tensor indices values &key (mode :raise))
  "按展平索引就地写入 values。

   INDICES 必须为整数（整数标量 / 整数张量）：非空浮点会报错，对标
   numpy（'float' object cannot be interpreted as an integer）。原实现
   用 round/truncate 静默截断浮点索引。
   空索引（如 '() / (vt-from-sequence '()) / (vt-zeros '(0))）合法，
   为空操作（见 `%check-integer-index`）。"
  (cond
    ((and (numberp indices) (floatp indices))
     (error "vt-put: 索引必须为整数，收到 ~a（对标 numpy）" indices))
    ((not (numberp indices))
     (%check-integer-index "vt-put" (ensure-vt indices))))
  (let* ((total (vt-size tensor))
         (idx-list (if (numberp indices) (list (round indices))
                       (let ((flat (vt-flatten (ensure-vt indices))))
                         (loop for i below (vt-size flat)
                               collect (truncate (aref (vt-data flat) i))))))
         (val-list (cond ((numberp values) (list values))
                         ((or (listp values) (arrayp values))
                          (vt-flatten-sequence values))
                         ((vt-p values)
                          (let ((flat (vt-flatten (ensure-vt values))))
                            (loop for i below (vt-size flat)
                                  collect (aref (vt-data flat) i))))))
         (n-values (length val-list)) (shape (vt-shape tensor))
         (strides (vt-strides tensor)) (offset (vt-offset tensor)) (data (vt-data tensor)))
    ;; H4 写入契约（v0.4.x 修复）：dim>1 且 stride=0 的广播维度禁止就地写入，
    ;; 否则一个逻辑位置会经零步长污染整片共享底层数据。
    ;; 改用 vt-out-writable-p 而非「任一 stride=0」：vt-slice 的 (:newa)
    ;; 产生 size=1 且 stride=0 的轴，它不构成广播、写入无歧义
    ;; （对标 numpy 允许 a[np.newaxis].put(...)）；原判定会误拒该情形。
    (unless (vt-out-writable-p tensor)
      (error "vt-put: 目标包含 dim>1 且 stride=0 的广播维度，禁止写入（H4 契约）"))
    (when (zerop n-values) (return-from vt-put tensor))
    (loop for raw in idx-list
          for vi from 0
          for val = (nth (mod vi n-values) val-list)
          do (let ((idx raw))
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
    tensor))

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
      (loop for i from 0 below final-size
            for raw = (truncate (aref idx-data (+ idx-off i)))
            for idx = (ecase mode
                        (:raise raw)
                        (:wrap  (mod raw n))
                        (:clip  (max 0 (min (1- n) raw))))
            do (when (or (< idx 0) (>= idx n))
                 (error "vt-choose: 索引 ~a 越界 (choices 数量 ~a)" raw n))
               (let* ((cv (nth idx flat-choices))
                      (cd (vt-data cv))
                      (co (vt-offset cv)))
                 (setf (aref result-data i)
                       (vt-cast (aref cd (+ co i)) result-type))))
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
       (if (/= (vt-item condition) 0)
           (vt-flatten tensor)
           (vt-zeros (list 0) :dtype (vt-dtype tensor))))
      ((equal cshape tshape)
       (let* ((flat-cond   (vt-flatten condition))
              (flat-data   (vt-flatten tensor))
              (cond-data   (vt-data flat-cond))
              (tensor-data (vt-data flat-data))
              (size        (vt-size flat-data))
              (result      '()))
         (loop for i from 0 below size
               when (/= (aref cond-data i) 0)
                 do (push (aref tensor-data i) result))
         (vt-from-sequence (nreverse result) :dtype (vt-dtype tensor))))
      (t
       ;; numpy 语义：arr.take(condition.nonzero()[0], mode='clip')（展平）。
       ;; v0.4.x 修复：原实现把 condition 广播到 TENSOR 形状，与 numpy 的
       ;; 「条件展平非零索引 + clip 采集」不一致（如 (2,1) 条件配 (3,) 数据，
       ;; numpy 取 arr[0]，原实现广播失败报错）。
       (let* ((flat-cond (vt-flatten condition))
              (flat-data (vt-flatten tensor))
              (cd (vt-data flat-cond))
              (td (vt-data flat-data))
              (csize (vt-size flat-cond))
              (tsize (vt-size flat-data))
              (result '()))
         (loop for i from 0 below csize
               when (/= (aref cd i) 0)
                 do (if (zerop tsize)
                        (error "vt-extract: 条件含真值但 TENSOR 为空（numpy IndexError）")
                        (let ((idx (max 0 (min i (1- tsize)))))
                          (push (aref td idx) result))))
         (vt-from-sequence (nreverse result) :dtype (vt-dtype tensor)))))))

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

(defun vt-bincount (x &key (minlength 0))
  "统计非负整数出现次数。"
      (when (%cx-reject x)
    (error "vt-bincount: 不支持复数输入（numpy 同样拒绝）。见 CONVENTIONS §16.11"))
(when (%cx-reject x)
    (error "vt-bincount: 不支持复数输入（numpy 同样拒绝）。见 CONVENTIONS §16.11"))
(let* ((flat (vt-flatten x))
         (maxval (if (zerop (vt-size flat))
                     -1
                     (vt-item (vt-amax flat))))
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

(defun %digitize-searchsorted (data n reverse-p side-right-p val)
  "在单调序列 data[0,n) 上二分。REVERSE-P 为 T 时逻辑序列取 data 的逆序；
   SIDE-RIGHT-P 为 T 等价 numpy searchsorted(side='right')，否则 'left'。"
  (let ((lo 0) (hi n))
    (loop while (< lo hi) do
      (let* ((mid (ash (+ lo hi) -1))
             (mv (aref data (if reverse-p (- n 1 mid) mid))))
        (if (if side-right-p (> mv val) (>= mv val))
            (setf hi mid)
            (setf lo (1+ mid)))))
    lo))

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

   （此前返回 (vt-shape flat-x)，多维输入被展平为 1-D；bins 直接
   vt-flatten，传 Lisp list 会抛 \"The value (4 8) is not of type CLVT:VT\"；
   且线性扫描把递减 bins 当作递增处理，结果与 numpy 不符。）"
  (when (%cx-any-p x bins)
    (error "vt-digitize: 复数不可分箱（numpy 对复数同样报 TypeError: x may not be complex）"))
  (let* ((x-vt (ensure-vt x))
         (bins-vt (ensure-vt bins))
         (flat-x (vt-flatten x-vt))
         (flat-bins (vt-flatten bins-vt))
         (bin-data (vt-data flat-bins))
         (nbins (vt-size flat-bins))
         (x-shape (vt-shape x-vt))
         (mono (%bins-monotonicity bin-data nbins))
         ;; numpy: side='left' if right else 'right'
         (side-right-p (not right))
         (reverse-p (= mono -1))
         (result (make-array (vt-size flat-x) :element-type '(signed-byte 64))))
    (when (zerop mono)
      (error "vt-digitize: bins 必须单调递增或递减（numpy 语义）"))
    (vt-do-each (ptr val flat-x)
      (let ((ss (%digitize-searchsorted bin-data nbins reverse-p
                                        side-right-p val)))
        (setf (aref result ptr) (if reverse-p (- nbins ss) ss))))
    ;; 形状取 X 的原始形状（而非展平后的 (n)）
    (%make-vt :data result :shape x-shape
              :strides (vt-compute-strides x-shape)
              :offset 0 :dtype :int64)))
