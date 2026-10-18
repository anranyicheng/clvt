;;;; manip.lisp — 形状操作、视图、翻转、滚动、三角、填充

(in-package :clvt)

;;; ------------------------------------------------------------------
;;; 重塑 / 视图 / 转置
;;; ------------------------------------------------------------------
(defun %resolve-minus-one (new-shape total-size)
  "解析形状中的 -1 占位符。
   - 形状中最多允许出现一个 -1，多个 -1 报错。
   - 若形状不含 -1，原样返回。
   - 若含一个 -1：
       * 已知维度乘积 known ≠ 0：要求 total-size 能被 known 整除，
         该位置解析为 total-size / known；
       * known = 0 且 total-size = 0：该位置解析为 0（与 NumPy 一致，
         例如 shape=(0 -1)、total=0 → (0 0)）；
       * known = 0 且 total-size ≠ 0：报错（无法确定未定维度）。"
  (declare (list new-shape)
           (type integer total-size))
  (let ((n-neg (count -1 new-shape)))
    (cond
      ((zerop n-neg)
       new-shape)
      ((> n-neg 1)
       (error "形状 ~a 中 -1 至多出现一次，实际出现 ~a 次"
              new-shape n-neg))
      (t
       (let* ((neg-idx (position -1 new-shape))
              (known   (reduce #'* (remove -1 new-shape) :initial-value 1))
              (shape   (copy-list new-shape)))
         (declare (type fixnum neg-idx)
                  (type integer known))
         (cond
           ((zerop known)
            (if (zerop total-size)
                (progn (setf (nth neg-idx shape) 0) shape)
                (error "无法将元素总数 ~a 重塑为形状 ~a（已知维度乘积为 0，-1 无法确定）"
                       total-size new-shape)))
           ((not (zerop (rem total-size known)))
            (error "无法将元素总数 ~a 重塑为形状 ~a（不能被已知维度乘积 ~a 整除）"
                   total-size new-shape known))
           (t
            (setf (nth neg-idx shape) (truncate total-size known))
            shape)))))))

(defun vt-view (vt new-shape)
  "零拷贝重塑视图（对标 pytorch tensor.view），要求输入连续。"
  (let* ((new-shape (%resolve-minus-one
                     new-shape
                     (vt-shape-to-size (vt-shape vt))))
         (old-size (vt-shape-to-size (vt-shape vt)))
         (new-size (vt-shape-to-size new-shape)))
    (unless (= old-size new-size)
      (error "view 失败: 元素总数不一致 (旧 ~a, 新 ~a)" old-size new-size))
    (unless (vt-contiguous-p vt)
      (error "view 失败: 张量内存不连续，请使用 vt-reshape"))
    (%make-vt :data (vt-data vt) :shape new-shape
              :strides (vt-compute-strides new-shape)
              :offset (vt-offset vt) :dtype (vt-dtype vt))))

(defun vt-reshape (vt new-shape)
  "重塑形状：连续则零拷贝视图，否则落地后重塑。"
  (let* ((new-shape (%resolve-minus-one
                     new-shape
                     (vt-shape-to-size (vt-shape vt))))
         (old-size (vt-shape-to-size (vt-shape vt)))
         (new-size (vt-shape-to-size new-shape)))
    (unless (= old-size new-size)
      (error "重塑失败: 元素总数不一致 (旧 ~a, 新 ~a)" old-size new-size))
    (if (vt-contiguous-p vt)
        (%make-vt :data (vt-data vt) :shape new-shape
                  :strides (vt-compute-strides new-shape)
                  :offset (vt-offset vt) :dtype (vt-dtype vt))
        (let ((cont (vt-contiguous vt)))
          (%make-vt :data (vt-data cont) :shape new-shape
                    :strides (vt-compute-strides new-shape)
                    :offset 0 :dtype (vt-dtype cont))))))

(defun vt-transpose (vt &optional (perm nil))
  "零拷贝转置。perm 为轴排列列表；缺省时反转所有轴。"
  (let* ((shape (vt-shape vt))
         (strides (vt-strides vt))
         (rank (length shape)))
    (unless perm
      (setf perm (loop for i from (1- rank) downto 0 collect i)))
    (unless (= (length perm) rank)
      (error "perm 长度 ~a 必须等于秩 ~a" (length perm) rank))
    (let ((seen (make-array rank :element-type 'bit :initial-element 0)))
      (dolist (p perm)
        (unless (and (integerp p) (<= 0 p (1- rank)))
          (error "perm 值 ~a 超出范围 [0, ~a]" p (1- rank)))
        (when (= (aref seen p) 1) (error "perm 中存在重复轴索引: ~a" p))
        (setf (aref seen p) 1)))
    (%make-vt :data (vt-data vt)
              :shape (mapcar (lambda (p) (nth p shape)) perm)
              :strides (mapcar (lambda (p) (nth p strides)) perm)
              :offset (vt-offset vt) :dtype (vt-dtype vt))))

(defun vt-squeeze (vt &key axis)
  "移除长度为 1 的维度（axis 指定单轴，缺省移除全部）。"
  (let ((shape (vt-shape vt))
        (strides (vt-strides vt))
        (offset (vt-offset vt)))
    (if axis
        (let* ((ax (vt-normalize-axis axis (length shape))))
          (unless (= (nth ax shape) 1)
            (error "无法挤压非单例维度: axis ~a 大小为 ~a" axis (nth ax shape)))
          (%make-vt :data (vt-data vt)
                    :shape (append (subseq shape 0 ax) (subseq shape (1+ ax)))
                    :strides (append (subseq strides 0 ax) (subseq strides (1+ ax)))
                    :offset offset :dtype (vt-dtype vt)))
        (let ((ns '()) (nst '()))
          (loop for d in shape for s in strides
                when (/= d 1) do (push d ns) (push s nst))
          (%make-vt :data (vt-data vt) :shape (nreverse ns)
                    :strides (nreverse nst) :offset offset :dtype (vt-dtype vt))))))

(defun vt-expand-dims (vt axis)
  "在指定位置插入长度为 1 的新轴。"
  (let* ((shape (vt-shape vt))
         (strides (vt-strides vt))
         (rank (length shape))
         (ax (if (< axis 0) (+ rank axis 1) axis)))
    (when (or (< ax 0) (> ax rank)) (error "轴 ~a 越界 (秩 ~a)" axis rank))
    (%make-vt :data (vt-data vt)
              :shape (append (subseq shape 0 ax) '(1) (subseq shape ax))
              :strides (append (subseq strides 0 ax) '(0) (subseq strides ax))
              :offset (vt-offset vt) :dtype (vt-dtype vt))))

(defun vt-unsqueeze (vt axis)
  "在 AXIS 处插入长度为 1 的轴（对标 torch.unsqueeze / numpy.expand_dims）。"
  (vt-expand-dims vt axis))

(defun vt-swapaxes (vt axis1 axis2)
  "交换两个轴（对标 numpy.swapaxes）。返回视图（共享底层内存）。"
  (let* ((rank (length (vt-shape vt)))
         (ax1 (vt-normalize-axis axis1 rank))
         (ax2 (vt-normalize-axis axis2 rank)))
    (vt-transpose vt (loop for i from 0 below rank
                           collect
                           (cond ((= i ax1) ax2)
                                 ((= i ax2) ax1)
                                 (t i))))))

(defun vt-moveaxis (tensor source destination)
  "移动轴。source/destination 可为整数或整数列表。"
  (let* ((shape (vt-shape tensor))
         (rank (length shape))
         (src (if (listp source) source (list source)))
         (dst (if (listp destination) destination (list destination))))
    (unless (= (length src) (length dst))
      (error "vt-moveaxis: source 与 destination 长度必须一致"))
    (setf src (mapcar (lambda (s) (vt-normalize-axis s rank)) src))
    (setf dst (mapcar (lambda (d) (vt-normalize-axis d rank)) dst))
    (unless (= (length src) (length (remove-duplicates src)))
      (error "vt-moveaxis: source 归一化后 ~a 存在重复轴" src))
    (unless (= (length dst) (length (remove-duplicates dst)))
      (error "vt-moveaxis: destination 归一化后 ~a 存在重复轴" dst))
    (let ((pairs '()))
      (loop for s in src
            for d in dst
            do (push (cons s d) pairs))
      (let ((remaining (loop for i below rank
                             unless (member i src) collect i))
            (free (loop for i below rank
                        unless (member i dst) collect i)))
        (loop for r in remaining
              for f in free
              do (push (cons r f) pairs)))
      (setf pairs (sort pairs #'< :key #'cdr))
      (vt-transpose tensor (mapcar #'car pairs)))))

(defun vt-rot90 (tensor &key (k 1) (axes '(0 1)))
  "在 axes 平面内旋转 90 度 k 次。"
  (let* ((rank (length (vt-shape tensor)))
         (ax0 (vt-normalize-axis (first axes) rank))
         (ax1 (vt-normalize-axis (second axes) rank)))
    (when (< rank 2) (error "vt-rot90: 秩必须 >= 2"))
    (when (= ax0 ax1) (error "vt-rot90: axes 必须不同"))
    (let ((k (mod k 4)))
      (loop with result = tensor repeat k
            do (setf result (vt-flip (vt-swapaxes result ax0 ax1) :axis ax0))
            finally (return result)))))

;;; ------------------------------------------------------------------
;;; 展平
;;; ------------------------------------------------------------------

(defun vt-flatten (vt)
  "展平为一维（返回副本）。"
  (vt-view (vt-copy vt) (list (vt-size vt))))

(defun vt-ravel (vt)
  "展平视图（连续则零拷贝，否则副本）。"
  (if (vt-contiguous-p vt)
      (vt-view vt (list (vt-size vt)))
      (vt-flatten vt)))

;;; ------------------------------------------------------------------
;;; 扁平化 + 强制 offset=0（v0.4.5 修复一类静默错误）
;;;
;;; vt-ravel 对**连续**输入返回的是 vt-view，它**保留原 offset**。于是所有
;;; 「先展平、再 (aref (vt-data flat) i) 按扁平下标直接取数」的代码，一旦
;;; 遇到带 offset 的切片视图（base[a:b]），读到的都是**基数组开头**的数据
;;; —— 不报错，只给出错误答案。
;;;
;;; 合并态实测受害者：
;;;   (vt-take (vt-slice [100 200 300 5 1 9] 3 6) '(0 2) :axis nil)
;;;     => (100.0d0 300.0d0)    numpy: (5.0d0 9.0d0)
;;; 第 5 轮的 vt-argsort(:axis nil) 是同一根因（已在那边单独修掉）。
;;;
;;; 约定：凡是「扁平下标 + 裸 AREF」的路径一律先过本函数；用 vt-ref 的路径
;;; （自带 offset 计算）不受影响，无需改动。
;;; ------------------------------------------------------------------
(defun %flat-offset0 (tensor)
  "展平成 offset=0、stride=1 的一维张量；能零拷贝就零拷贝。
   VT-FLATTEN 恒返回副本（offset=0），故非连续/带 offset 的情况直接用它。"
  (cond
    ((and (vt-contiguous-p tensor) (zerop (vt-offset tensor)))
     (if (= (length (vt-shape tensor)) 1)
         tensor
         (vt-view tensor (list (vt-size tensor)))))
    (t (vt-flatten tensor))))

(defun vt-broadcast-to (vt new-shape)
  "广播到新形状（零拷贝视图）。"
  (let ((bs (vt-broadcast-shapes (vt-shape vt) new-shape)))
    (unless (equal bs new-shape)
      (error "形状 ~a 不能广播到 ~a" (vt-shape vt) new-shape))
    (%make-vt :data (vt-data vt) :shape new-shape
              :strides (vt-broadcast-strides
                        (vt-shape vt) new-shape (vt-strides vt))
              :offset (vt-offset vt) :dtype (vt-dtype vt))))

;;; ------------------------------------------------------------------
;;; 翻转与滚动
;;; ------------------------------------------------------------------

(defun vt-flip (vt &key axis)
  "沿指定轴翻转（零拷贝视图）；axis 为 nil 时全部翻转。"
  (let ((shape (vt-shape vt)) (strides (vt-strides vt)))
    (if (null axis)
        (%make-vt :data (vt-data vt) :shape shape
                  :strides (mapcar #'- strides)
                  :offset (+ (vt-offset vt)
                             (loop for d in shape
                                   for s in strides
                                   sum (* (1- d) s)))
                  :dtype (vt-dtype vt))
        (let* ((rank (length shape))
               (ax (vt-normalize-axis axis rank))
               (strides (copy-list strides))
               (dim (nth ax shape))
               (old-stride (nth ax strides)))
          (setf (nth ax strides) (- old-stride))
          (%make-vt :data (vt-data vt) :shape shape :strides strides
                    :offset (+ (vt-offset vt) (* (1- dim) old-stride))
                    :dtype (vt-dtype vt))))))

(defun vt-roll (vt shift &key axis)
  "滚动元素。语义对标 NumPy：
   - shift 为整数, axis 为整数   → 沿该轴滚动
   - shift 为列表, axis 为列表   → 长度必须相同，逐轴滚动
   - shift 为列表, axis 为 nil   → 累加所有 shift 后展平滚动
   - shift 为整数, axis 为 nil   → 展平滚动

   返回值始终是**新张量**、不与输入共享底层内存（对标 numpy.roll 总是
   返回新数组）。此前在 shift 取模后为 0（含 shift=0、shift 为轴长
   整数倍）或轴长为 0 时直接返回入参 `vt` 本身，调用者对结果的就地写入
   会写穿到输入张量。"
  (let ((shift-list (if (listp shift) shift (list shift))))
    (cond
      ;; axis 为列表：逐轴滚动
      ((and axis (listp axis))
       (let ((shifts (if (listp shift) shift
                         (make-list (length axis) :initial-element shift))))
         (unless (= (length shifts) (length axis))
           (error "shift 和 axis 的长度必须一致"))
         (loop with result = (vt-copy vt)
               for s in shifts for ax in axis
               do (setf result (vt-roll result s :axis ax))
               finally (return result))))
      ;; axis 为整数：只允许单个 shift
      (axis
       (unless (= (length shift-list) 1)
         (error "vt-roll: axis 为整数时 shift 必须为单个整数（或长度 1 的列表），收到 ~a"
                shift-list))
       (let* ((sh (vt-shape vt)) (ax (vt-normalize-axis axis (length sh)))
                                 (n (nth ax sh)))
         (if (zerop n) (vt-copy vt)
             (let ((s (mod (car shift-list) n)))
               (if (zerop s) (vt-copy vt)
                   (vt-concatenate
                    ax
                    (apply #'vt-slice vt (loop for i below (length sh)
                                               collect
                                               (if (= i ax)
                                                   (list (- n s) n)
                                                   '(:all))))
                    (apply #'vt-slice vt (loop for i below (length sh)
                                               collect
                                               (if (= i ax)
                                                   (list 0 (- n s))
                                                   '(:all))))))))))
      ;; axis 为 nil：累加所有 shift 后展平滚动
      (t
       (let* ((flat (vt-flatten vt)) (n (vt-size flat)))
         (if (zerop n) (vt-reshape flat (vt-shape vt))
             (let ((s (mod (reduce #'+ shift-list) n)))
               (if (zerop s)
                   (vt-reshape flat (vt-shape vt))
                   (vt-reshape (vt-concatenate
                                0
                                (vt-slice flat (list (- n s) n))
                                (vt-slice flat (list 0 (- n s))))
                               (vt-shape vt))))))))))

;;; ------------------------------------------------------------------
;;; 窄切片与分割
;;; ------------------------------------------------------------------

(defun %require-equal-division (op-name dim sections)
  "numpy 的 split / vsplit / hsplit / dsplit 在**整数段数**形式下要求整除，
   否则抛 ValueError(\"array split does not result in an equal division\")。
   整数列表形式（indices）无需整除，调用方不应走这里。
   注意：本库的 vt-split 刻意对标 numpy.array_split（允许不整除），
   故仅 vsplit / hsplit / dsplit 需要本检查。
   （此前 hsplit / dsplit 直接转交 vt-split，沿用了 array_split 的余数
   语义，与 numpy 的 hsplit/dsplit 契约不符。）"
  (when (and (integerp sections) (plusp sections))
    (unless (and (plusp dim) (zerop (mod dim sections)))
      (error "~a: array split does not result in an equal division（该轴长度 ~a ÷ ~a 段不整除；numpy 的 split/vsplit/hsplit/dsplit 对整数段数要求整除，需不等分请改用 vt-split）"
             op-name dim sections))))

(defun vt-narrow (vt axis start end)
  "零拷贝切片（等价 pytorch narrow）。

start/end 支持 numpy 切片负索引语义：负值先加轴长（-1 → 末尾）；
规约后自动截断到 [0, dim-size]（对标 numpy 切片对越界的 clip 行为，
v0.4.2 起支持，此前负索引/越界直接报错）。规约后 end < start 仍报错。"
  (let* ((shape (copy-list (vt-shape vt)))
         (rank (length shape))
         (ax (vt-normalize-axis axis rank))
         (strides (vt-strides vt))
         (dim-size (nth ax shape))
         ;; numpy 切片语义：负索引 +轴长，再 clip 到 [0, dim-size]
         (start (max 0 (min (+ start (if (< start 0) dim-size 0)) dim-size)))
         (end   (max 0 (min (+ end   (if (< end 0)   dim-size 0)) dim-size))))
    (when (< end start)
      (error "vt-narrow: 规约后 end (~a) 必须不小于 start (~a)" end start))
    (setf (nth ax shape) (- end start))
    (%make-vt :data (vt-data vt) :shape shape :strides strides
              :offset (+ (vt-offset vt) (* start (nth ax strides)))
              :dtype (vt-dtype vt))))

(defun vt-split (tensor indices-or-sections &key (axis 0))
  "沿轴分割（对标 numpy array_split）。"
  (let* ((shape (vt-shape tensor))
         (rank (length shape))
         (ax (vt-normalize-axis axis rank))
         (dim-size (nth ax shape)))
    (cond
      ((integerp indices-or-sections)
       (let* ((n indices-or-sections)
              (base (floor dim-size n))
              (rem (rem dim-size n)))
         (loop with start = 0
               for i from 0 below n
               for chunk = (if (< i rem) (1+ base) base)
               collect (prog1 (vt-narrow tensor ax start (+ start chunk))
                         (incf start chunk)))))
      ((listp indices-or-sections)
       (let ((points (append '(0) indices-or-sections (list dim-size))))
         (setf points (mapcar (lambda (p) (if (minusp p) (+ p dim-size) p)) points))
         (setf points (mapcar (lambda (p) (max 0 (min p dim-size))) points))
         (loop for (s e) on points by #'cdr
               while e
               collect (vt-narrow tensor ax s e))))
      (t (error "indices-or-sections 必须是整数或整数列表")))))

(defun vt-vsplit (vt indices-or-sections)
  "垂直分割（对标 numpy.vsplit）。等价 split 沿轴 0，但 numpy.vsplit
   对整数段数要求严格整除（不整除报错）；
   索引列表形式（np.vsplit(a, [1,3])）无需整除，保持支持。"
  (let* ((sh (vt-shape vt))
         (rows (if sh (first sh) (vt-size vt))))
    (%require-equal-division "vt-vsplit" rows indices-or-sections))
  (vt-split vt indices-or-sections :axis 0))

(defun vt-hsplit (vt indices-or-sections)
  "水平均分（对标 numpy.hsplit）。1D 时等价 split 沿轴 0，否则沿轴 1。

   numpy.hsplit 等价 numpy.split（非 array_split）：对**整数段数**
   要求该轴整除，不整除抛 ValueError；整数列表形式无需整除。
   （此前直接转交 vt-split，采用了 array_split 的余数语义，例如
   (2,5) 按 2 段会得到 (2,3)+(2,2)，numpy 会报错。）"
  (let ((rank (length (vt-shape vt))))
    (when (zerop rank)
      (error "vt-hsplit: hsplit only works on arrays of 1 or more dimensions（numpy 语义）"))
    (let ((ax (if (<= rank 1) 0 1)))
      (%require-equal-division "vt-hsplit" (nth ax (vt-shape vt)) indices-or-sections)
      (vt-split vt indices-or-sections :axis ax))))

(defun vt-dsplit (vt indices-or-sections)
  "深度均分（对标 numpy.dsplit）。沿轴 2 均分，要求秩 >= 3。

   numpy.dsplit 等价 numpy.split（非 array_split）：对**整数段数**
   要求轴 2 整除，不整除抛 ValueError；整数列表形式无需整除。
   （此前直接转交 vt-split，采用了 array_split 的余数语义，例如轴长 4
   按 3 段会得到 2+1+1，numpy 会报错。）"
  (let ((rank (length (vt-shape vt))))
    (when (< rank 3)
      (error "vt-dsplit: dsplit only works on arrays of 3 or more dimensions（当前秩 ~a，numpy 语义）"
             rank))
    (%require-equal-division "vt-dsplit" (nth 2 (vt-shape vt)) indices-or-sections)
    (vt-split vt indices-or-sections :axis 2)))

;;; ------------------------------------------------------------------
;;; 三角与对角
;;; ------------------------------------------------------------------

(defun vt-triu (tensor &key (k 0))
  "上三角矩阵（支持 batch）。"
  (let* ((res (vt-copy tensor))
         (res-data (vt-data res))
         (rank (length (vt-shape res)))
         (shape-vec (coerce (vt-shape res) 'simple-vector))
         (strs-vec (coerce (vt-strides res) 'simple-vector))
         (zero (coerce 0 (vt-element-type res))))
    (when (< rank 2) (error "vt-triu 要求秩 >= 2"))
    (labels ((recurse (depth ptr)
               (if (= depth (- rank 2))
                   (let ((rows (svref shape-vec depth))
                         (cols (svref shape-vec (1+ depth)))
                         (str-r (svref strs-vec depth))
                         (str-c (svref strs-vec (1+ depth))))
                     ;; 同上：清零 [0, r+k) 这一段连续区间
                     (loop for r from 0 below rows
                           for rp = ptr then (+ rp str-r)
                           do (if (= str-c 1)
                                  (let ((c1 (min cols (max 0 (+ r k)))))
                                    (when (plusp c1)
                                      (fill res-data zero :start rp :end (+ rp c1))))
                                  (loop for c from 0 below cols
                                        when (< c (+ r k))
                                          do (setf (aref res-data (+ rp (* c str-c)))
                                                   zero)))))
                   (let ((dim (svref shape-vec depth))
                         (stride (svref strs-vec depth)))
                     (loop for i from 0 below dim do
                       (recurse (1+ depth) ptr) (incf ptr stride))))))
      (recurse 0 (vt-offset res)))
    res))

(defun vt-tril (tensor &key (k 0))
  "下三角矩阵（支持 batch）。"
  (let* ((res (vt-copy tensor))
         (res-data (vt-data res))
         (rank (length (vt-shape res)))
         (shape-vec (coerce (vt-shape res) 'simple-vector))
         (strs-vec (coerce (vt-strides res) 'simple-vector))
         (zero (coerce 0 (vt-element-type res))))
    (when (< rank 2) (error "vt-tril 要求秩 >= 2"))
    (labels ((recurse (depth ptr)
               (if (= depth (- rank 2))
                   (let ((rows (svref shape-vec depth))
                         (cols (svref shape-vec (1+ depth)))
                         (str-r (svref strs-vec depth))
                         (str-c (svref strs-vec (1+ depth))))
                     ;; v0.4.5 性能修复：每行要清零的是**连续一段**（res 由
                     ;; vt-copy 得到，恒为 C 连续 → str-c = 1），用 FILL 一次
                     ;; 写完，避免逐元素泛型 (setf (aref ...))（原 9485us /
                     ;; numpy 725us）。非连续（str-c ≠ 1）退回逐元素循环。
                     (loop for r from 0 below rows
                           for rp = ptr then (+ rp str-r)
                           do (if (= str-c 1)
                                  (let ((c0 (max 0 (1+ (+ r k)))))
                                    (when (< c0 cols)
                                      (fill res-data zero
                                            :start (+ rp c0)
                                            :end (+ rp cols))))
                                  (loop for c from 0 below cols
                                        when (> c (+ r k))
                                          do (setf (aref res-data (+ rp (* c str-c)))
                                                   zero)))))
                   (let ((dim (svref shape-vec depth))
                         (stride (svref strs-vec depth)))
                     (loop for i from 0 below dim do
                       (recurse (1+ depth) ptr) (incf ptr stride))))))
      (recurse 0 (vt-offset res)))
    res))

(defun vt-diagonal (tensor &key (offset 0) axis1 axis2)
  "提取对角线（支持 batch）。

   AXIS1 / AXIS2 指定取对角线的两条轴（支持负索引）；结果形状 =
   「去掉这两轴后其余各维（升序）」+「对角线长度」（对角线轴置于末尾），
   与 numpy.diagonal / torch.diagonal 一致。

   两者必须**同时**给出（只给其一无法确定对角平面）。

   **默认（都不给）取最后两轴** —— pytorch / torch.diagonal 风格，
   等价于 `np.diagonal(a, axis1=-2, axis2=-1)`；显式给出 axis1/axis2
   则得到 numpy 语义。注意 numpy 的 `np.diagonal(a)` 缺省是
   axis1=0 / axis2=1 的**历史特例**，batch 下与本函数默认行为不同，
   需显式传 `axis1=0 axis2=1` 才能复现。

   （此前只有 offset，且固定取最后两轴，无法表达 numpy 的 axis1/axis2。）"
  (let* ((in-shape (vt-shape tensor))
         (rank (length in-shape)))
    (when (< rank 2) (error "vt-diagonal 要求秩 >= 2"))
    (unless (eq (null axis1) (null axis2))
      (error "vt-diagonal: :axis1 与 :axis2 必须同时给出（只给其一无法确定对角平面）"))
    (let* ((a1 (if axis1 (vt-normalize-axis axis1 rank) (- rank 2)))
           (a2 (if axis2 (vt-normalize-axis axis2 rank) (1- rank))))
      (when (= a1 a2)
        (error "vt-diagonal: :axis1 与 :axis2 不能指向同一轴（~a）" a1))
      (let* ((rows (nth a1 in-shape))
             (cols (nth a2 in-shape))
             (r-init (if (> offset 0) 0 (- offset)))
             (c-init (if (> offset 0) offset 0))
             (diag-len (max 0 (min (- rows r-init) (- cols c-init))))
             ;; 保留轴（升序）：输出形状 = 保留各维 + 对角线维（末尾）
             (keep (loop for i below rank unless (or (= i a1) (= i a2)) collect i))
             (keep-dims (mapcar (lambda (i) (nth i in-shape)) keep))
             (out-shape (append keep-dims (list diag-len)))
             (res (vt-zeros out-shape :dtype (vt-dtype tensor)))
             (res-data (vt-data res))
             (in-data (vt-data tensor))
             (in-strs (vt-strides tensor))
             (batch-size (reduce #'* keep-dims :initial-value 1))
             (str-r (nth a1 in-strs))
             (str-c (nth a2 in-strs))
             (in-offset (vt-offset tensor))
             (out-idx 0))
        (if (or (zerop diag-len) (zerop batch-size))
            res
            (progn
              (dotimes (batch batch-size)
                (let ((in-ptr in-offset)
                      (rem batch))
                  ;; 从最后一个保留轴反向解码坐标，匹配 row-major 输出
                  (loop for d from (1- (length keep)) downto 0
                        for ax = (nth d keep)
                        for dim = (nth ax in-shape)
                        for str = (nth ax in-strs)
                        do (multiple-value-bind (q r) (floor rem dim)
                             (incf in-ptr (* r str))
                             (setf rem q)))
                  (loop for i from 0 below diag-len
                        for src = (+ in-ptr (* (+ r-init i) str-r)
                                     (* (+ c-init i) str-c))
                        do (setf (aref res-data out-idx) (aref in-data src))
                           (incf out-idx))))
              res))))))

(defun vt-diag (tensor &key (k 0))
  "提取对角线或构造对角矩阵（对标 numpy.diag）。"
  (let* ((shape (vt-shape tensor)) (rank (length shape)))
    (cond
      ((= rank 1)
       (let* ((n (first shape))
              (dim (+ n (abs k)))
              (res (vt-zeros (list dim dim) :dtype (vt-dtype tensor)))
              (res-data (vt-data res))
              (in-data (vt-data tensor))
              (in-offset (vt-offset tensor))
              (in-stride (first (vt-strides tensor)))
              (row-stride (first (vt-strides res)))
              (col-stride (second (vt-strides res)))
              (start (if (> k 0) (* k col-stride) (* (- k) row-stride))))
         (loop for i from 0 below n
               for src = (+ in-offset (* i in-stride))
               for dst = start then (+ dst row-stride col-stride)
               do (setf (aref res-data dst) (aref in-data src)))
         res))
      ((= rank 2) (vt-diagonal tensor :offset k))
      (t (error "vt-diag 仅支持 1D 或 2D 输入，当前维度 ~a" rank)))))

;;; ------------------------------------------------------------------
;;; 重复与平铺
;;; ------------------------------------------------------------------
;;; ------------------------------------------------------------------
;;; 沿轴重复的填充内核（vt-repeat 的轴分支用）
;;;
;;; v0.4.6 性能修复：vt-repeat 的轴分支里 SD / RD 都取自 (vt-data ...)，
;;; 没有元素类型声明，于是 `(aref sd ...)` 与 `(setf (aref rd ...) ...)`
;;; 逐元素走通用 AREF（二维分派 + 边界检查）与装箱算术。
;;; 500×2000 沿轴重复 ×2 实测 45778us（numpy.repeat 2643us）。
;;;
;;; 处理方式是「每元素沿轴重复 REPEATS[J] 次」：把一条 fiber 的写入
;;; 抽成本函数，内部按元素类型展开成 8 份特化循环。里程计与外层指针
;;; 推进仍留在 vt-repeat 里（它们只跟秩有关，与元素类型无关），
;;; 所以这里只负责「从 IP 读一条 fiber，按 REPS 写到 OP 起的位置」，
;;; 不回写任何指针 —— 调用方用固定的 OP 作为起点即可。
;;; ------------------------------------------------------------------
(defun %repeat-flat-scalar (sd soff rd size r)
  "把 SD[SOFF, SOFF+SIZE) 的每个元素连续复制 R 份，写入 RD（从 0 开始）。
   仅用于 vt-repeat 的「标量重复 + axis=nil」快路径。"
  (declare (type fixnum soff size r))
  (macrolet ((dup (lt)
               `(let ((s (the (simple-array ,lt (*)) sd))
                      (d (the (simple-array ,lt (*)) rd)))
                  (declare (type (simple-array ,lt (*)) s d)
                           (type fixnum soff size r))
                  (when (plusp r)
                    (dotimes (i size)
                      (declare (type fixnum i))
                      (let ((v (aref s (the fixnum (+ soff i))))
                            (p (the fixnum (* i r))))
                        (declare (type ,lt v) (type fixnum p))
                        (dotimes (k r)
                          (declare (type fixnum k))
                          (setf (aref d (the fixnum (+ p k))) v))))))))
    (let ((et (array-element-type rd)))
      (cond
        ((equal et 'double-float)       (dup double-float))
        ((equal et 'single-float)       (dup single-float))
        ((equal et '(signed-byte 64))   (dup (signed-byte 64)))
        ((equal et '(signed-byte 32))   (dup (signed-byte 32)))
        ((equal et '(signed-byte 16))   (dup (signed-byte 16)))
        ((equal et '(signed-byte 8))    (dup (signed-byte 8)))
        ((equal et '(unsigned-byte 16)) (dup (unsigned-byte 16)))
        ((equal et '(unsigned-byte 8))  (dup (unsigned-byte 8)))
        (t (error "vt-repeat: 不支持的输出元素类型 ~a" et))))))

(defun %repeat-counts (repeats ax-size ax)
  "把 repeats（标量或列表，可含 0/负数）归一成**长度 AX-SIZE 的非负 fixnum 向量**。
   AX 只用于报错信息。列表形态要求长度恰好等于轴长（与原实现一致）。"
  (declare (type fixnum ax-size ax))
  (let ((v (make-array (max ax-size 1) :element-type 'fixnum)))
    (cond
      ((listp repeats)
       (unless (= (length repeats) ax-size)
         (error "vt-repeat: repeats 长度 ~a 与轴 ~a 长度 ~a 不匹配"
                (length repeats) ax ax-size))
       (loop for r in repeats
             for i fixnum from 0
             do (setf (aref v i) (max (the integer r) 0))))
      (t (let ((r (max (the integer repeats) 0)))
           (declare (type fixnum r))
           (dotimes (i ax-size)
             (declare (type fixnum i))
             (setf (aref v i) r)))))
    v))

(defun %repeat-fiber (sd rd ip op reps n in-ax-str out-ax-str)
  "把 SD 中从 IP 起、步长 IN-AX-STR 的 N 个元素，按 REPS[J] 次逐元素重复写入 RD。
   OP 为写入起点；每写一份前进 OUT-AX-STR。
   REPS 是长度 N 的 fixnum 向量（非负），标量/列表/负值已由调用方归一化。
   要求 RD / SD 的元素类型相同且属于 8 种存储类型之一。"
  (declare (type fixnum ip op n in-ax-str out-ax-str)
           (type (simple-array fixnum (*)) reps))
  (macrolet ((dup (lt)
               `(let ((s (the (simple-array ,lt (*)) sd))
                      (d (the (simple-array ,lt (*)) rd))
                      (rv (the (simple-array fixnum (*)) reps)))
                  (declare (type (simple-array ,lt (*)) s d)
                           (type (simple-array fixnum (*)) rv)
                           (type fixnum ip op n in-ax-str out-ax-str))
                  (dotimes (j n)
                    (declare (type fixnum j))
                    (let ((v (aref s (the fixnum (+ ip (the fixnum (* j in-ax-str))))))
                          (rr (aref rv j)))
                      (declare (type ,lt v) (type fixnum rr))
                      (when (plusp rr)
                        (dotimes (k rr)
                          (declare (type fixnum k))
                          (setf (aref d op) v)
                          (incf op out-ax-str))))))))
    (let ((et (array-element-type rd)))
      (cond
        ((equal et 'double-float)       (dup double-float))
        ((equal et 'single-float)       (dup single-float))
        ((equal et '(signed-byte 64))   (dup (signed-byte 64)))
        ((equal et '(signed-byte 32))   (dup (signed-byte 32)))
        ((equal et '(signed-byte 16))   (dup (signed-byte 16)))
        ((equal et '(signed-byte 8))    (dup (signed-byte 8)))
        ((equal et '(unsigned-byte 16)) (dup (unsigned-byte 16)))
        ((equal et '(unsigned-byte 8))  (dup (unsigned-byte 8)))
        (t (error "vt-repeat: 不支持的输出元素类型 ~a" et))))))

(defun vt-repeat (vt repeats &key axis)
  "重复元素。repeats 可为标量或列表；若为列表，其长度必须等于被重复轴的长度。

  v0.4.2 修复（原实现两处严重缺陷）：
    1) axis=nil 时对每个元素 `(make-vt (list rep) ...)` 造一个独立张量，
       再 `(apply #'vt-concatenate 0 parts)` —— 1M 元素即 1M 次小分配，
       且 apply 要压 1M 个实参，实测直接 **Control stack exhausted**
       （数据大一点就崩，不是慢的问题）；axis 分支同样用 apply 拼 ax-size 个切片。
    2) 现在两条分支都改为「先算输出形状、再直接填充」：内存分配 1 次、
       循环 O(输出元素数)、无 apply、无中间视图。"
  ;; ================================================================
  ;; v0.4.6 快路径：**标量** repeats + axis=nil
  ;;
  ;; 原实现无条件把标量展开成 (make-list size :initial-element repeats)：
  ;; 1M 元素即 1M 个 cons（约 16MB 垃圾），随后还要 (loop for r in reps)
  ;; 遍历整张表取 r。实测 20892us（numpy.repeat 2636us，差 7.9 倍）。
  ;; 标量情形根本不需要表 —— 每个元素连续复制 R 份即可。
  ;; 独立成 (when ... (return-from ...)) 是有意的：不改动下方既有分支
  ;; 的嵌套层次，回退/审查都只需看这一段。
  ;; ================================================================
  (when (and (null axis) (integerp repeats))
    (let* ((flat (vt-contiguous (vt-flatten vt)))
           (size (vt-size flat))
           (r (max (the integer repeats) 0))
           (total (the fixnum (* size r)))
           (res (make-vt (list total) 0 :dtype (vt-dtype vt))))
      (%repeat-flat-scalar (vt-data flat) (vt-offset flat) (vt-data res) size r)
      (return-from vt-repeat res)))
  (if (null axis)
      ;; ---------------- 扁平重复：一维结果，线性填充 ----------------
      (let* ((flat (vt-contiguous (vt-flatten vt)))
             (size (vt-size flat))
             (reps (if (listp repeats)
                       repeats
                       (make-list size :initial-element repeats))))
        (unless (= (length reps) size)
          (error "vt-repeat: repeats 长度 ~a 与张量元素总数 ~a 不匹配"
                 (length reps) size))
        (let* ((total (loop for r in reps sum (max r 0)))
               (res (make-vt (list total) 0 :dtype (vt-dtype vt)))
               (rd (vt-data res))
               (sd (vt-data flat))
               (soff (the fixnum (vt-offset flat)))
               (pos 0))
          (declare (type fixnum soff pos))
          ;; v0.4.5 性能修复：RD / SD 原来都没有元素类型声明，逐元素
          ;; (setf (aref rd pos) v) 走泛型二维分派 + 装箱（1M×2 实测
          ;; 23273us / numpy 1869us）。改为按元素类型特化的写循环。
          ;; 注：这里**不**用 FILL —— 每段只有 REPEATS 个元素，1M 次 FILL
          ;; 调用本身的开销比逐元素写还大（实测反而慢 6%）。
          (macrolet ((dup (lt)
                       `(let ((d (the (simple-array ,lt (*)) rd))
                              (sc (the (simple-array ,lt (*)) sd)))
                          (declare (type (simple-array ,lt (*)) d sc)
                                   (type fixnum soff pos))
                          (loop for r in reps
                                for i fixnum from 0
                                do (let ((v (aref sc (the fixnum (+ soff i))))
                                         (rr (max r 0)))
                                     (declare (type ,lt v) (type fixnum rr))
                                     (dotimes (k rr)
                                       (declare (type fixnum k))
                                       (setf (aref d pos) v)
                                       (incf pos)))))))
            (let ((et (array-element-type rd)))
              (cond
                ((equal et 'double-float)     (dup double-float))
                ((equal et 'single-float)     (dup single-float))
                ((equal et '(signed-byte 64)) (dup (signed-byte 64)))
                ((equal et '(signed-byte 32)) (dup (signed-byte 32)))
                ((equal et '(signed-byte 16)) (dup (signed-byte 16)))
                ((equal et '(signed-byte 8))  (dup (signed-byte 8)))
                ((equal et '(unsigned-byte 16)) (dup (unsigned-byte 16)))
                ((equal et '(unsigned-byte 8))  (dup (unsigned-byte 8)))
                (t (loop for r in reps
                         for i fixnum from 0
                         do (let ((v (aref sd (the fixnum (+ soff i))))
                                  (rr (max r 0)))
                              (dotimes (k rr)
                                (declare (ignore k))
                                (setf (aref rd pos) v)
                                (incf pos))))))))
          res))
      ;; ---------------- 沿指定轴重复 ----------------
      (let* ((sh (vt-shape vt))
             (rank (length sh))
             (ax (vt-normalize-axis axis rank))
             (ax-size (nth ax sh))
             ;; v0.4.6：归一成 fixnum 向量（非负、长度 = 轴长），长度校验也搬进去。
             ;; 原来这里既可能是用户给的 list 也可能是临时 make-list，内层循环
             ;; 于是逐元素做一次 (max r 0) 的**泛型**算术 —— 1e6 次。
             (reps (%repeat-counts repeats ax-size ax)))
        (let* ((out-ax (reduce #'+ reps :initial-value 0))
               (out-shape (loop for d in sh for i fixnum from 0
                                collect (if (= i ax) out-ax d)))
               (res (make-vt out-shape 0 :dtype (vt-dtype vt)))
               (src (vt-contiguous vt))
               (sd (vt-data src)) (rd (vt-data res))
               (in-str (vt-strides src)) (out-str (vt-strides res))
               (soff (the fixnum (vt-offset src)))
               (in-ax-str (the fixnum (nth ax in-str)))
               (out-ax-str (the fixnum (nth ax out-str)))
               ;; 除 AX 之外的外层轴（源与目标同构下标）
               (o-dims (loop for d in sh for i fixnum from 0
                             unless (= i ax) collect d))
               (o-in (loop for st in in-str for i fixnum from 0
                           unless (= i ax) collect st))
               (o-out (loop for st in out-str for i fixnum from 0
                            unless (= i ax) collect st))
               (no (length o-dims)))
          (if (zerop out-ax)
              res
              (let ((idx (make-list no :initial-element 0))
                    (ip soff)
                    (op 0))
                (declare (type fixnum ip op))
                (loop
                  ;; 对当前外层组合：沿 AX 逐元素重复（REPEATS 是「每个元素
                  ;; 重复几次」，不是「整条切片重复几次」—— 后者是 tile 语义）
                  ;; v0.4.6：本段原来是「逐元素 (max r 0) + 未声明 AREF」的内层
                  ;; 循环（500×2000 沿轴 ×2 实测 42773us / numpy 2643us）。
                  ;; 现在把「按 REPS 把一条 fiber 写出去」交给类型化内核。
                  (%repeat-fiber sd rd ip op reps ax-size in-ax-str out-ax-str)
                  ;; 推进外层里程计（行主序，最后一维最快）
                  (let ((d (1- no)))
                    (declare (type fixnum d))
                    (loop
                      (when (< d 0) (return-from vt-repeat res))
                      (let ((nd (1+ (nth d idx))))
                        (cond ((< nd (nth d o-dims))
                               (setf (nth d idx) nd)
                               (incf ip (nth d o-in))
                               (incf op (nth d o-out))
                               (return))
                              (t (decf ip (the fixnum (* (nth d idx) (nth d o-in))))
                                 (decf op (the fixnum (* (nth d idx) (nth d o-out))))
                                 (setf (nth d idx) 0)
                                 (decf d)))))))))))))

(defun vt-tile (vt reps)
  "平铺构造新数组（对标 numpy.tile）。
   REPS 可为整数或整数列表：
     - 列表长度 < VT 秩：左侧补 1 后再逐轴相乘；
     - 列表长度 > VT 秩：VT 左侧补 1 维后再逐轴相乘；
     - REPS 中含 0：结果为空张量，形状对应维度为 0（对标 numpy）；
     - REPS 中含负数：报错（numpy 同样报错）。
   示例（对标 numpy）：
     (vt-tile x 2)          ≡ np.tile(x, 2)
     (vt-tile x '(2 3))     ≡ np.tile(x, (2, 3))
     (vt-tile x '(0 2))     → 空张量 shape=(0 2*cols)（不报错）
     (vt-tile x '(2 -1))    → 报错：reps 不能为负"
  (let* ((sh (vt-shape vt))
         (reps-list (if (listp reps) reps (list reps)))
         (ndim (max (length sh) (length reps-list)))
         (padded-sh    (append (make-list (- ndim (length sh))
                                          :initial-element 1)
                               sh))
         (padded-reps  (append (make-list (- ndim (length reps-list))
                                          :initial-element 1)
                               reps-list))
         (final-shape  (mapcar #'* padded-sh padded-reps)))
    (unless (every #'integerp padded-reps)
      (error "vt-tile: reps 必须全为整数，收到 ~a" reps))
    (when (some #'minusp padded-reps)
      (error "vt-tile: reps 不能为负数，收到 ~a" reps))
    (when (some #'zerop padded-reps)
      (return-from vt-tile
        (vt-zeros final-shape :dtype (vt-dtype vt))))
    (let ((result (vt-reshape vt padded-sh)))
      (loop for axis from 0 below ndim
            for rep = (nth axis padded-reps)
            when (> rep 1)
              do (setf result (apply #'vt-concatenate
                                     axis
                                     (loop repeat rep collect result))))
      (vt-view result final-shape))))
;;; ------------------------------------------------------------------
;;; 填充
;;; ------------------------------------------------------------------

(defun %normalize-pad-width (pad-width rank)
  (labels ((intp (x)
             (and (integerp x) (>= x 0)))
           (pairp (x)
             (and (consp x) (= (length x) 2) (every #'integerp x)
                  (>= (first x) 0) (>= (second x) 0)))
           (norm (x)
             (cond ((intp x) (list x x)) ((pairp x) x)
                   (t (error "invalid pad-width element: ~a" x)))))
    (cond ((integerp pad-width)
           (make-list rank :initial-element (norm pad-width)))
          ((and (listp pad-width)
                (= (length pad-width) 1)
                (integerp (first pad-width)))
           (make-list rank :initial-element (norm (first pad-width))))
          ((and (listp pad-width)
                (= (length pad-width) 2)
                (every #'integerp pad-width))
           (make-list rank :initial-element (norm pad-width)))
          ((and (listp pad-width)
                (= (length pad-width) 1)
                (consp (first pad-width))
                (= (length (first pad-width)) 2)
                (every #'integerp (first pad-width)))
           (make-list rank :initial-element (norm (first pad-width))))
          ((and (listp pad-width)
                (= (length pad-width) rank))
           (mapcar #'norm pad-width))
          (t (error "pad-width 长度 ~a 与秩 ~a 不匹配" (length pad-width) rank)))))

(defun %pad-map (mode dist sk side)
  "计算 padding 时，把越界坐标 dist 映射回原始轴 [0, sk) 内的索引。
   dist  > 0，表示越界距离（从最近边界算起）。
   sk    = 原轴长度（调用方保证 sk >= 0）。
   side  = :left（左/上越界）或 :right（右/下越界）。
   前置条件：dist > 0；sk > 0 或 mode 为 :constant（由调用方处理）。"
  (when (and (<= sk 0) (not (eq mode :constant)))
    (error "无法用 mode ~a 扩展空轴 (sk=~a)" mode sk))
  (ecase mode
    ;; ---- :edge：越界一律取最近端点 ----
    (:edge
     (if (eq side :left)
         0
         (1- sk)))
    ;; ---- :wrap：按 sk 取模循环 ----
    (:wrap
     (mod (if (eq side :left)
              (- sk dist)
              (1- dist)) sk))
    ;; ---- :reflect：以边界为轴心反射，周期 = 2*(sk-1) ----
    ;; sk = 1 时无法构成有效周期（period = 0），与 NumPy / scipy 一致报错。
    (:reflect
     (when (< sk 2)
       (error "无法用 mode :reflect 扩展长度为 ~a 的轴（需 >= 2）" sk))
     (let* ((period (* 2 (1- sk)))
            (x   (if (eq side :left)
                     dist
                     (- sk 1 dist)))
            (idx (mod x period)))
       (if (< idx sk)
           idx
           (- period idx))))
    ;; ---- :symmetric：以边界元素为轴心镜像，周期 = 2*sk ----
    ;; sk = 1 时退化为常值映射到索引 0，天然安全，无需特殊处理。
    (:symmetric
     (let* ((period (* 2 sk))
            (x   (if (eq side :left)
                     (- dist)
                     (+ sk dist -1)))
            (idx (mod x period)))
       (if (< idx sk)
           idx
           (- period idx 1))))))


(defun vt-pad (vt pad-width &key (mode :constant) (constant-values 0))
  "对张量填充。mode: :constant/:edge/:wrap/:reflect/:symmetric。

   constant-values 支持：
   - 标量：所有维度左右填充相同值；
   - (left right)：所有维度使用同一对前/后值；
   - ((l0 r0) (l1 r1) ...)：每个维度指定前/后值（长度须等于 rank）。
   角点取值符合 NumPy 逐轴填充顺序：后处理的轴覆盖先处理的轴。"
  (let* ((shape (vt-shape vt))
         (rank (length shape))
         (pad (%normalize-pad-width pad-width rank))

         ;; 规范化 constant-values
         (cv-norm
           (cond
             ((numberp constant-values)
              (make-list rank :initial-element (list constant-values constant-values)))
             ((and (listp constant-values)
                   (= (length constant-values) 2)
                   (every #'numberp constant-values))
              (make-list rank :initial-element constant-values))
             ((and (listp constant-values)
                   (= (length constant-values) rank)
                   (every (lambda (x)
                            (and (listp x)
                                 (= (length x) 2)
                                 (every #'numberp x)))
                          constant-values))
              constant-values)
             (t
              (error "invalid constant-values: ~a" constant-values))))

         (c-lefts (coerce (mapcar (lambda (cv) (vt-cast (first cv) (vt-dtype vt)))
                                  cv-norm)
                          'simple-vector))
         (c-rights (coerce (mapcar (lambda (cv) (vt-cast (second cv) (vt-dtype vt)))
                                   cv-norm)
                           'simple-vector))

         (new-shape (loop for s in shape
                          for (b a) in pad
                          collect (+ s b a)))
         (out (vt-zeros new-shape :dtype (vt-dtype vt)))
         (out-data (vt-data out))
         (in-data (vt-data vt))
         (in-offset (vt-offset vt))
         (out-strides (coerce (vt-compute-strides new-shape) 'simple-vector))
         (in-shape (coerce shape 'simple-vector))
         (in-strides (coerce (vt-strides vt) 'simple-vector))
         (pad-before (coerce (mapcar #'first pad) 'simple-vector))
         (out-size (vt-size out)))

    ;; ================================================================
    ;; v0.4.6 性能修复（:constant 快路径）
    ;;
    ;; 原实现是「对每个输出元素做 rank 次 FLOOR 分解 + 泛型 AREF」：
    ;; 1M 元素 × (1 次除法 + 2 次泛型访问)，实测 19966us
    ;; （numpy.pad 225us，差 87 倍）。但 :constant 模式下输出被
    ;; 「边界 + 内部盒」完整铺满，而**内部盒就是源的一次整块拷贝**：
    ;;
    ;;   1. 逐轴填充前后两条边界板（d 升序 → 后面的轴覆盖前面的轴，
    ;;      与下方循环「最后一个越界轴胜出」的角点语义一致）；
    ;;   2. 内部盒 = out 的居中视图，用 vt-copy-into 一次拷入源
    ;;      （源任意步长都支持；dest 为 strided 视图时亦然）。
    ;;
    ;; 边界体积是 O(表面积) 而非 O(体积)：1M 元素 pad (10,10) 只有 20 个
    ;; 边界元素，其余全走 memcpy。:edge/:wrap/:reflect/:symmetric 需要
    ;; 逐点映射源下标，仍走下方通用循环（语义完全不变）。
    ;; ================================================================
    (when (eq mode :constant)
      (loop for d from 0 below rank do
        (let* ((pb (first (nth d pad)))
               (pa (second (nth d pad))))
          (when (plusp pb)
            (vt-fill (apply #'vt-slice out
                            (loop for e from 0 below rank
                                  collect (if (= e d) (list 0 pb) '(:all))))
                     (svref c-lefts d)))
          (when (plusp pa)
            (vt-fill (apply #'vt-slice out
                            (loop for e from 0 below rank
                                  collect (if (= e d)
                                              (list (+ (nth e shape) pb)
                                                    (nth e new-shape))
                                              '(:all))))
                     (svref c-rights d)))))
      (vt-copy-into (apply #'vt-slice out
                           (loop for e from 0 below rank
                                 collect (list (first (nth e pad))
                                               (+ (first (nth e pad))
                                                  (nth e shape)))))
                    vt)
      (return-from vt-pad out))

    ;; ================================================================
    ;; v0.4.7 快路径：:edge / :wrap / :reflect / :symmetric
    ;;
    ;; 原实现是「对每个输出元素做 rank 次 FLOOR 分解 + 泛型 AREF + 一次
    ;; %PAD-MAP」，1M 元素 pad (10,10) 实测 37–41ms，1000×1000 pad (10,10)
    ;; 达 68ms（numpy.pad 同等工作量 0.6–1.4ms）。
    ;;
    ;; 但边界体积是 O(表面积)：1M 元素 pad (10,10) 只有 20 个元素越界。
    ;; 于是照 :constant 的骨架来，只是把「填常量」换成「按映射索引 gather」：
    ;;   1. 内部盒 = 源的一次整块拷贝（与 :constant 相同）；
    ;;   2. 逐轴（升序）把该轴的映射索引用 vt-take 取出，写进前/后板。
    ;;
    ;; 角点语义与 numpy 的「逐轴 concat」逐位一致：写第 D 轴的板时，其它轴
    ;; e<D 取**已扩展**的全范围（:all），e>D 取尚未扩展的居中区间
    ;; [pb_e, pb_e+shape_e)。因此后处理的轴覆盖先处理的轴 —— 正是 numpy 的
    ;; 「最后一个越界轴胜出」。
    ;;
    ;; 注：gather 的索引是「相对内部盒」的，而 out 的 D 轴带有 PB 偏移，
    ;; 故索引要加 PB；这是本段唯一容易写错的地方。
    ;; ================================================================
    (when (plusp rank)
      (flet ((other-specs (d d-spec)
               ;; 除 D 之外的轴向：e < D 已扩展 → :all；e > D 未扩展 → 居中区间
               (loop for e from 0 below rank
                     collect (if (= e d)
                                 d-spec
                                 (if (< e d)
                                     '(:all)
                                     (list (first (nth e pad))
                                           (+ (first (nth e pad)) (nth e shape)))))))
             (idx-tensor (lo-hi n)
               ;; 构造长度为 N 的 int64 索引张量（零拷贝包装，见 %make-vt）
               (%make-vt :data lo-hi :shape (list n) :strides '(1)
                         :offset 0 :dtype :int64)))
        ;; 1) 内部盒一次拷入
        (vt-copy-into (apply #'vt-slice out
                             (loop for e from 0 below rank
                                   collect (list (first (nth e pad))
                                                 (+ (first (nth e pad))
                                                    (nth e shape)))))
                      vt)
        ;; 2) 逐轴前/后板
        (loop for d from 0 below rank do
          (let* ((pb (first (nth d pad)))
                 (pa (second (nth d pad)))
                 (sk (nth d shape)))
            (when (plusp pb)
              (let ((iv (make-array pb :element-type '(signed-byte 64))))
                (dotimes (o pb)
                  (setf (aref iv o)
                        (the fixnum (+ pb (%pad-map mode (- pb o) sk :left)))))
                (vt-copy-into (apply #'vt-slice out (other-specs d (list 0 pb)))
                              (vt-take (apply #'vt-slice out (other-specs d '(:all)))
                                       (idx-tensor iv pb)
                                       :axis d))))
            (when (plusp pa)
              (let ((iv (make-array pa :element-type '(signed-byte 64))))
                (dotimes (o pa)
                  (setf (aref iv o)
                        (the fixnum (+ pb (%pad-map mode (1+ o) sk :right)))))
                (vt-copy-into (apply #'vt-slice out
                                     (other-specs d (list (+ sk pb) (nth d new-shape))))
                              (vt-take (apply #'vt-slice out (other-specs d '(:all)))
                                       (idx-tensor iv pa)
                                       :axis d)))))))
      (return-from vt-pad out))

    (loop for out-ptr fixnum from 0 below out-size do
      (let ((rem out-ptr)
            (in-ptr in-offset)
            (is-const nil)
            (const-val 0))                     ; 初始值任意，is-const 为 nil 时不使用
        (loop for d from 0 below rank do
          (multiple-value-bind (out-idx r) (floor rem (svref out-strides d))
            (setf rem r)
            (let* ((bk (svref pad-before d))
                   (sk (svref in-shape d))
                   (offset (- out-idx bk))
                   (src-idx 0))
              (cond
                ;; 左/上越界
                ((< offset 0)
                 (if (eq mode :constant)
                     ;; 关键修复：无条件赋值，确保后处理的轴覆盖先处理的轴
                     (setf is-const t const-val (svref c-lefts d))
                     (setf src-idx (%pad-map mode (- offset) sk :left))))
                ;; 右/下越界
                ((>= offset sk)
                 (if (eq mode :constant)
                     (setf is-const t const-val (svref c-rights d))
                     (setf src-idx (%pad-map mode (- offset sk -1) sk :right))))
                ;; 内部元素
                (t
                 (setf src-idx offset)))
              (incf in-ptr (* src-idx (svref in-strides d))))))

        (setf (aref out-data out-ptr)
              (if is-const
                  const-val
                  (aref in-data in-ptr)))))
    out))

