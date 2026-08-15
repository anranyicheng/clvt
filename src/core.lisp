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

(declaim (inline vt-shape vt-strides vt-offset vt-data vt-dtype vt-p))

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
  (reduce #'* (vt-shape vt) :initial-value 1))

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
      (let ((result nil) (stride 1))
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
                                 :initial-element (coerce initial-element lisp-type))))
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
                                     :initial-element (coerce obj lisp-type))
                 :shape nil :strides nil :offset 0 :dtype final)))
    (sequence (vt-from-sequence obj :dtype (or dtype :float64)))))

;;; ------------------------------------------------------------------
;;; 广播
;;; ------------------------------------------------------------------

(declaim (inline vt-broadcast-shapes))

(defun vt-broadcast-shapes (shape1 shape2)
  "计算广播后的结果形状，严格对标 NumPy。"
  (declare (list shape1 shape2) (optimize (speed 3) (safety 0)))
  (let* ((len1 (length shape1))
         (len2 (length shape2))
         (max-len (max len1 len2))
         (result (make-list max-len)))
    (declare (type fixnum len1 len2 max-len))
    (do ((i 0 (1+ i))
         (s1 (nthcdr (- max-len len1) shape1) (cdr s1))
         (s2 (nthcdr (- max-len len2) shape2) (cdr s2))
         (r result (cdr r)))
        ((= i max-len) result)
      (declare (type fixnum i))
      (let ((dim1 (if s1 (the fixnum (car s1)) 1))
            (dim2 (if s2 (the fixnum (car s2)) 1)))
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
    (let ((t-tail result) (t-shp target-shape))
      (dotimes (i rank-diff)
        (declare (type fixnum i))
        (setf (car t-tail) 0
              t-tail (cdr t-tail)
              t-shp (cdr t-shp)))
      (do ((o-shp orig-shape (cdr o-shp))
           (o-str orig-strides (cdr o-str)))
          ((null o-shp) result)
        (let ((t-dim (the fixnum (car t-shp)))
              (o-dim (the fixnum (car o-shp))))
          (unless (or (= o-dim t-dim) (= o-dim 1))
            (error "vt-broadcast-strides: 形状不匹配! ~a vs ~a" o-dim t-dim))
          (setf (car t-tail) (if (= o-dim 1) 0 (the fixnum (car o-str)))
                t-tail (cdr t-tail)
                t-shp (cdr t-shp)))))))

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
  "将 axis（nil/整数/整数列表）归一化为排序后的正轴列表；nil 表示全局归约。"
  (if (null axis)
      nil
      (let ((axes (if (listp axis) axis (list axis))))
        (sort (mapcar (lambda (a) (vt-normalize-axis a rank)) axes) #'<))))

;;; ------------------------------------------------------------------
;;; 连续判定与落地
;;; ------------------------------------------------------------------

(defun vt-contiguous-p (vt)
  "判断张量是否为 C 连续（可安全重塑）。对标 numpy 的 c_contiguous 判定。"
  (let ((shape (vt-shape vt)) (strides (vt-strides vt)))
    (if (or (null shape) (some #'zerop shape))
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

(defun vt-astype (tensor new-dtype)
  "将张量转换为新类型（浮点转整数截断）。返回连续的新张量。"
  (let* ((shape (vt-shape tensor))
         (new (make-vt shape 0 :dtype new-dtype))
         (new-data (vt-data new))
         (in-data (vt-data tensor))
         (in-strides (vt-strides tensor))
         (in-offset (vt-offset tensor))
         (rank (length shape))
         (converter (vt-cast-fun new-dtype)))
    (labels ((copy-rec (depth in-ptr out-ptr)
               (if (= depth rank)
                   (setf (aref new-data out-ptr)
                         (funcall converter (aref in-data in-ptr)))
                   (let ((dim (nth depth shape))
                         (in-stride (nth depth in-strides))
                         (out-stride (nth depth (vt-strides new))))
                     (loop for i fixnum from 0 below dim do
                       (copy-rec (1+ depth) in-ptr out-ptr)
                       (incf in-ptr in-stride)
                       (incf out-ptr out-stride))))))
      (copy-rec 0 in-offset 0))
    new))

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

(defun vt-copy-into (dest src)
  "将 src 拷贝到 dest（支持广播与类型转换）。返回 dest。"
  (setf src (ensure-vt src))
  (let ((dest-shape (vt-shape dest)) (src-shape (vt-shape src)))
    (loop for d in dest-shape for s in (vt-strides dest)
          when (and (> d 1) (zerop s))
            do (error "vt-copy-into: 目标视图是只读的广播视图（维度 ~a）" d))
    (let ((final-shape (vt-broadcast-shapes dest-shape src-shape)))
      (unless (equal final-shape dest-shape)
        (error "vt-copy-into: dest 形状 ~a 无法容纳 src 广播后 ~a" dest-shape final-shape))
      (let* ((dest-data (vt-data dest))
             (src-data (vt-data src))
             (dest-dtype (vt-dtype dest))
             (src-dtype (vt-dtype src))
             (src-strides (vt-broadcast-strides src-shape dest-shape (vt-strides src)))
             (size (vt-shape-to-size dest-shape)))
        (cond
          ;; 极速：连续 + 同型 -> memcpy
          ((and (vt-contiguous-p dest) (vt-contiguous-p src)
                (equal dest-shape src-shape) (equal dest-dtype src-dtype))
           (replace dest-data src-data
                    :start1 (vt-offset dest) :end1 (+ (vt-offset dest) size)
                    :start2 (vt-offset src) :end2 (+ (vt-offset src) size)))
          ;; 中速：连续 + 同形 -> 单层类型转换循环
          ((and (vt-contiguous-p dest) (vt-contiguous-p src)
                (equal dest-shape src-shape))
           (let ((d-off (vt-offset dest)) (s-off (vt-offset src)))
             (declare (type fixnum d-off s-off size))
             (dotimes (i size)
               (setf (aref dest-data (+ d-off i))
                     (vt-cast (aref src-data (+ s-off i)) dest-dtype)))))
          ;; 慢速：非连续 / 广播 -> 通用 strided 迭代
          (t
           (%copy-strided dest-data dest-dtype (vt-strides dest) (vt-offset dest)
                          src-data src-strides (vt-offset src)
                          dest-shape size)))
        dest))))

(defun %copy-strided (dest-data dest-dtype dest-strides dest-offset
                      src-data src-strides src-offset shape size)
  "通用按步长/广播拷贝（里程表迭代，零动态分配）。"
  (let* ((rank (length shape))
         (dims (coerce shape 'simple-vector))
         (d-strs (coerce dest-strides 'simple-vector))
         (s-strs (coerce src-strides 'simple-vector))
         (indices (make-array rank :element-type 'fixnum :initial-element 0))
         (d-ptr dest-offset)
         (s-ptr src-offset))
    (declare (type simple-vector dims d-strs s-strs)
             (type (simple-array fixnum (*)) indices)
             (type fixnum d-ptr s-ptr rank))
    (when (zerop size) (return-from %copy-strided nil))
    (loop
      (setf (aref dest-data d-ptr) (vt-cast (aref src-data s-ptr) dest-dtype))
      (let ((depth (1- rank)))
        (loop
          (when (< depth 0) (return-from %copy-strided nil))
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
  "用标量 value 原地填充张量 vt 的所有元素（支持视图）。返回 vt。"
  (let* ((data (vt-data vt))
         (cval (vt-cast value (vt-dtype vt)))
         (size (vt-size vt)))
    (if (vt-contiguous-p vt)
        (let ((off (vt-offset vt)))
          (dotimes (i size)
            (setf (aref data (+ off i)) cval)))
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
