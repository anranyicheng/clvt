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

(defun %make-vt-uninit (shape dtype)
  "创建一个内容未定义的张量；调用方必须完整写入所有 size 个元素。
   用于逐元素映射等『先分配、后全覆盖』的场景，避免 make-array 的零填充。
   与 make-vt 的唯一区别：不传 :initial-element，跳过全量写零。"
  (let* ((size (vt-shape-to-size shape))
         (lisp-type (vt-dtype->lisp-type dtype))
         (data (make-array size :element-type lisp-type)))
    (%make-vt :data data
              :shape shape
              :strides (vt-compute-strides shape)
              :offset 0
              :dtype dtype)))

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
    (sequence (vt-from-sequence obj :dtype (or dtype :float64)))))

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

;;; ------------------------------------------------------------------
;;; 拷贝入口
;;; ------------------------------------------------------------------

(defun vt-copy-into (dest src)
  "将 src 拷贝到 dest（支持广播与类型转换）。返回 dest。

   语义契约：
   1. dest 形状必须能容纳 src 广播后的形状，否则报错。
   2. dest 中「dim>1 且 stride=0」的广播维度是只读的，写入会报错。
   3. 当 dest 与 src 共享底层数组且物理区间重叠时，先对 src 做快照，
      使拷贝具有 memmove 语义（等价于 numpy 对重叠视图的处理），
      调用方无需关心 dest 与 src 是否别名。"
  (setf src (ensure-vt src))
  (let ((dest-shape (vt-shape dest))
        (src-shape  (vt-shape src)))
    ;; ---- 1) dest 可写性检查 ----------------------------------------
    (loop for d in dest-shape
          for s in (vt-strides dest)
          when (and (> d 1) (zerop s))
            do (error "vt-copy-into: 目标视图是只读的广播视图（维度 ~a）" d))
    ;; ---- 2) 形状兼容性检查 ------------------------------------------
    (let ((final-shape (vt-broadcast-shapes dest-shape src-shape)))
      (unless (equal final-shape dest-shape)
        (error "vt-copy-into: dest 形状 ~a 无法容纳 src 广播后 ~a"
               dest-shape final-shape)))
    ;; ---- 3) 别名保护：重叠 → 用 vt-do-each 快照 src ----------------
    ;; 快照后 src 是独立连续的 vt，后续所有分支都不再与 dest 别名。
    (when (%vt-views-overlap-p dest src)
      (let* ((shape (vt-shape src))
             (size  (vt-shape-to-size shape))
             (data  (vt-data src))
             (et    (array-element-type data))
             (tmp   (make-array size :element-type et))
             (i     0))
        (declare (type fixnum i size))
        (vt-do-each (p v src)
          (declare (ignore p))
          (setf (aref tmp i) v)
          (incf i))
        (setf src (%make-vt :data tmp
                            :shape shape
                            :strides (vt-compute-strides shape)
                            :offset 0
                            :dtype (vt-dtype src)))))
    ;; ---- 4) 实际拷贝 -----------------------------------------------
    (let* ((dest-data   (vt-data dest))
           (src-data    (vt-data src))
           (dest-dtype  (vt-dtype dest))
           (src-dtype   (vt-dtype src))
           (src-strides (vt-broadcast-strides src-shape dest-shape
                                              (vt-strides src)))
           (size        (vt-shape-to-size dest-shape)))
      (declare (type fixnum size))
      (cond
        ;; 极速：连续 + 同形 + 同型 → 底层 replace
        ((and (vt-contiguous-p dest)
              (vt-contiguous-p src)
              (equal dest-shape src-shape)
              (equal dest-dtype src-dtype))
         (replace dest-data src-data
                  :start1 (vt-offset dest) :end1 (+ (vt-offset dest) size)
                  :start2 (vt-offset src)  :end2 (+ (vt-offset src)  size)))

        ;; 中速：连续 + 同形 → 单层类型转换循环
        ((and (vt-contiguous-p dest)
              (vt-contiguous-p src)
              (equal dest-shape src-shape))
         (let ((d-off  (vt-offset dest))
               (s-off  (vt-offset src))
               (caster (vt-cast-fun dest-dtype)))
           (declare (type fixnum d-off s-off)
                    (type function caster))
           (dotimes (i size)
             (setf (aref dest-data (+ d-off i))
                   (funcall caster (aref src-data (+ s-off i)))))))

        ;; 慢速：非连续 / 广播 → 通用 strided 迭代
        (t
         (%copy-strided dest-data dest-dtype (vt-strides dest) (vt-offset dest)
                        src-data src-dtype src-strides (vt-offset src)
                        dest-shape size)))

      dest)))

(defun %copy-strided (dest-data dest-dtype dest-strides dest-offset
                      src-data src-dtype src-strides src-offset shape size)
  "通用按步长/广播拷贝（里程表迭代，零动态分配）。

优化：当 src-dtype 与 dest-dtype 相同时，跳过逐元素 vt-cast 的 ecase
分派，直接原值拷贝；异型时在循环外用 vt-cast-fun 取出特化转换函数，
避免每元素重复 ecase。"
  (let* ((rank (length shape))
         (dims (coerce shape 'simple-vector))
         (d-strs (coerce dest-strides 'simple-vector))
         (s-strs (coerce src-strides 'simple-vector))
         (indices (make-array rank :element-type 'fixnum :initial-element 0))
         (d-ptr dest-offset)
         (s-ptr src-offset)
         ;; 同型时为 nil（直接拷贝原值），异型时为转换函数
         (caster (unless (equal dest-dtype src-dtype)
                   (vt-cast-fun dest-dtype))))
    (declare (type simple-vector dims d-strs s-strs)
             (type (simple-array fixnum (*)) indices)
             (type fixnum d-ptr s-ptr rank)
             (type (or null function) caster))
    (when (zerop size)
      (return-from %copy-strided nil))
    (loop
      (setf (aref dest-data d-ptr)
            (if caster
                (funcall caster (aref src-data s-ptr))
                (aref src-data s-ptr)))
      (let ((depth (1- rank)))
        (loop
          (when (< depth 0)
	    (return-from %copy-strided nil))
          (incf (aref indices depth))
          (if (< (aref indices depth) (svref dims depth))
              (progn (incf d-ptr (svref d-strs depth))
                     (incf s-ptr (svref s-strs depth))
                     (return))
              (progn (setf (aref indices depth) 0)
                     (decf d-ptr (* (svref d-strs depth) (1- (svref dims depth))))
                     (decf s-ptr (* (svref s-strs depth) (1- (svref dims depth))))
                     (decf depth))))))))

;;; ------------------------------------------------------------------
;;; 填充
;;; ------------------------------------------------------------------

(defun vt-fill (vt value)
  "用标量 value 原地填充张量 vt 的所有元素（支持视图）。返回 vt。
   广播视图（dim>1 且 stride=0 的维度）在语义上只读，写入会报错。"
  (loop for d in (vt-shape vt)
        for s in (vt-strides vt)
        when (and (> d 1) (zerop s))
          do (error "vt-fill: 目标视图是只读的广播视图（维度 ~a）" d))
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
