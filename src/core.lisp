;;;; core.lisp — 张量核心：结构、步长、广播、连续判定、拷贝

(in-package :clvt)

;;; ------------------------------------------------------------------
;;; 结构定义
;;; ------------------------------------------------------------------

(defstruct (vt (:constructor %make-vt))
  "N 维张量。data 为一维物理数组，shape/strides 描述逻辑视图，offset 支持零拷贝切片。"
  (data (make-array 0) :type (simple-array *))
  (shape nil :type list)
  (strides nil :type list)
  (offset 0 :type fixnum)
  (dtype :float64 :type symbol))

(declaim (inline vt-shape vt-strides vt-offset vt-data vt-dtype vt-p
		 vt-order vt-size))

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
         (lisp-type (vt-dtype->lisp-type dtype))
         (data (make-array size :element-type lisp-type
                                :initial-element
				(if initial-element
				    (coerce initial-element lisp-type)
				    (coerce 0 lisp-type)))))
    (%make-vt :data data
              :shape shape
              :strides (vt-compute-strides shape)
              :offset 0
              :dtype dtype)))

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

(defun ensure-vt (obj &key (dtype nil))
  "将标量/序列/张量统一转换为张量。标量 -> 0 维张量；序列 -> 一维以上张量。"
  (etypecase obj
    (vt (if (and dtype (not (eq (vt-dtype obj) dtype)))
            (vt-copy-into (make-vt (vt-shape obj) 0 :dtype dtype) obj)
            obj))
    (number
     (let* ((infer (cond ((typep obj 'single-float) :float32)
                         ((typep obj 'double-float) :float64)
                         ((typep obj 'integer) :int64)
                         (t :float64)))
            (final (or dtype infer))
            (lisp-type (vt-dtype->lisp-type final)))
       (%make-vt :data (make-array 1 :element-type lisp-type
                                     :initial-element (vt-cast obj final))
                 :shape nil :strides nil :offset 0 :dtype final)))
    (sequence
     (let ((infer (if (%all-integer-sequence-p obj) :int64 :float64)))
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

(defun vt-contiguous (vt)
  "返回内存连续的副本（若已连续则返回自身）。"
  (if (vt-contiguous-p vt)
      vt
      (let* ((new (make-vt (vt-shape vt) 0 :dtype (vt-dtype vt))))
        (vt-copy-into new vt)
        new)))

;;; ------------------------------------------------------------------
;;; 拷贝与类型转换
;;; ------------------------------------------------------------------
(eval-when (:compile-toplevel :load-toplevel :execute)
  (defun %astype-cast-form (out-lt src-expr)
    "生成把 src-expr 转换到 out-lt 的代码。
     整数走与 vt-cast / vt-cast-fun 一致的安全 coerce/回绕语义。"
    (cond ((equal out-lt '(signed-byte 64))    `(%coerce-int64  ,src-expr))
          ((equal out-lt '(signed-byte 32))    `(%coerce-int32  ,src-expr))
          ((equal out-lt '(signed-byte 16))    `(%coerce-int16  ,src-expr))
          ((equal out-lt '(signed-byte 8))     `(%coerce-int8   ,src-expr))
          ((equal out-lt '(unsigned-byte 16))  `(%coerce-uint16 ,src-expr))
          ((equal out-lt '(unsigned-byte 8))   `(%coerce-uint8  ,src-expr))
          ((subtypep out-lt 'integer)          `(truncate ,src-expr))
          (t                                   `(coerce ,src-expr ',out-lt)))))

(defun vt-astype (tensor new-dtype)
  "将张量转换为新类型（浮点转整数截断）。返回连续的新张量。"
  (declare (optimize (speed 3) (safety 0)))
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
      (cond
        ;; ---- 标量 ----
        ((zerop rank)
         (setf (aref new-data 0)
               (funcall (vt-cast-fun new-dtype) (aref in-data in-offset))))

        ;; ---- 连续视图 ----
        ((vt-contiguous-p tensor)
         (let ((in-et (array-element-type in-data)))
           (macrolet
	       ((spec (in-lt out-lt)
                  (let ((expr (%astype-cast-form out-lt '(aref src p))))
                    `(let ((src (the (simple-array ,in-lt (*)) in-data))
                           (dst (the (simple-array ,out-lt (*)) new-data))
                           (p in-offset))
                       (declare (type (simple-array ,in-lt (*)) src)
                                (type (simple-array ,out-lt (*)) dst)
                                (type fixnum p))
                       (dotimes (i size)
                         (setf (aref dst i) ,expr)
                         (incf p)))))
                (memcpy (lt)
                  `(replace (the (simple-array ,lt (*)) new-data)
                            (the (simple-array ,lt (*)) in-data)
                            :start1 0 :end1 size
                            :start2 in-offset
                            :end2 (the fixnum (+ in-offset size)))))
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
               ;; 其他组合：通用 fallback（走 vt-cast-fun，语义已对齐）
               ;; ============================================================
               (t
                (let ((conv (vt-cast-fun new-dtype))
                      (p in-offset))
                  (declare (type fixnum p))
                  (dotimes (i size)
                    (setf (aref new-data i)
                          (funcall conv (aref in-data p)))
                    (incf p))))))))

        ;; ---- 非连续视图 ----
        (t
         (let ((converter (vt-cast-fun new-dtype))
               (dims (coerce shape 'simple-vector))
               (i-strs (coerce in-strides 'simple-vector))
               (indices (make-array rank :element-type 'fixnum :initial-element 0))
               (i-ptr in-offset))
           (declare (type simple-vector dims i-strs)
                    (type (simple-array fixnum (*)) indices)
                    (type fixnum i-ptr))
           (dotimes (k size)
             (setf (aref new-data k)
                   (funcall converter (aref in-data i-ptr)))
             (let ((d (the fixnum (1- rank))))
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
      new)))


(defun vt-copy (vt &key dtype)
  "深度拷贝：返回独立、内存连续的新张量。可选类型转换。"
  (let ((target-dtype (or dtype (vt-dtype vt))))
    (if (eq target-dtype (vt-dtype vt))
        (let* ((shape (vt-shape vt))
               (size (vt-shape-to-size shape))
               (new (make-vt shape 0 :dtype target-dtype)))
          (if (vt-contiguous-p vt)
              (replace (vt-data new) (vt-data vt)
                       :start2 (vt-offset vt) :end2 (+ (vt-offset vt) size))
              (vt-copy-into new vt))
          new)
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
           (type list shape dest-strides src-strides)
           (optimize (speed 3) (safety 0)))
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
   所有路径单遍遍历，无中间缓冲。"
  (declare (type fixnum size dest-off src-off rank)
           (type list shape dest-strides src-strides)
           (optimize (speed 3) (safety 0)))
  (when (zerop size) (return-from %copy-generic nil))
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
         (replace dest-data src-data
                  :start1 dest-off :end1 (+ dest-off size)
                  :start2 src-off  :end2 (+ src-off size)))

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
      dest)))

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
