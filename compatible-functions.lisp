;;;; 扩展功能:对标 NumPy 函数
(in-package :clvt)

;;; =========================================================================
;;; 0. 数组属性补充
;;; =========================================================================

(defun vt-itemsize (vt)
  "返回每个元素的字节大小."
  (let ((element-type (vt-element-type vt)))
    (cond ((subtypep element-type 'double-float) 8)
          ((subtypep element-type 'single-float) 4)
          ((subtypep element-type 'fixnum)
	   #+sbcl sb-vm:n-word-bytes
	   #-sbcl 8)
          ((subtypep element-type '(unsigned-byte 8)) 1)
          ((subtypep element-type '(signed-byte 8)) 1)
          ((subtypep element-type '(unsigned-byte 16)) 2)
          ((subtypep element-type '(signed-byte 16)) 2)
          ((subtypep element-type '(unsigned-byte 32)) 4)
          ((subtypep element-type '(signed-byte 32)) 4)
          ((subtypep element-type '(unsigned-byte 64)) 8)
          ((subtypep element-type '(signed-byte 64)) 8)
          (t 8)))) ; 默认8字节

(defun vt-nbytes (vt)
  "返回张量占用的总字节数."
  (* (vt-size vt) (vt-itemsize vt)))


;;; ===========================================
;;; 1. 数组创建扩展
;;; ===========================================

(defun vt-linspace
    (start end num &key (endpoint t) (type 'double-float))
  "创建线性间隔数组.
  start: 起始值
  end: 结束值
  num: 元素个数
  endpoint: 是否包含结束值(默认 T)
  返回: 一维张量"
  (declare (type number start end)
           (type fixnum num)
           (type boolean endpoint))
  (when (<= num 0)
    (error "num 必须大于 0"))
  (if (= num 1)
      ;; 单点情况：直接返回 [start]
      (let ((data (make-array 1 :element-type type
                                :initial-element (coerce start type))))
        (%make-vt :data data :shape '(1) :strides '(1) :offset 0))
      (let* ((divisor (if endpoint (1- num) num))
             (step (/ (- end start) divisor))
             (data (make-array num :element-type type)))
        (loop for i from 0 below num
              do (setf (aref data i)
                       (coerce (+ start (* i step)) type)))
        (%make-vt :data data
                  :shape (list num)
                  :strides '(1)
                  :offset 0))))

(defun vt-full (shape fill-value &key (type 'double-float))
  "创建指定值填充的数组.
  shape: 形状列表
  fill-value: 填充值
  返回: 张量"
  (make-vt shape fill-value :type type))

(defun vt-empty (shape &key (type 'double-float))
  "创建未初始化数组(实际用 0 填充,避免垃圾数据).
  shape: 形状列表
  返回: 张量"
  (vt-zeros shape :type type))

(defun vt-zeros-like (vt &key (type nil))
  "创建与给定数组形状相同的全零数组.
  vt: 输入张量
  type: 数据类型(默认与输入相同)
  返回: 张量"
  (vt-zeros (vt-shape vt)
	    :type (or type (vt-element-type vt))))

(defun vt-ones-like (vt &key (type nil))
  "创建与给定数组形状相同的全一数组."
  (vt-ones (vt-shape vt)
	   :type (or type (vt-element-type vt))))

(defun vt-full-like (vt fill-value &key (type nil))
  "创建与给定数组形状相同的填充数组."
  (vt-full (vt-shape vt)
	   fill-value
	   :type (or type (vt-element-type vt))))

(defun vt-empty-like (vt &key (type nil))
  "创建与给定数组形状相同的未初始化数组."
  (vt-empty (vt-shape vt)
	    :type (or type (vt-element-type vt))))


(defun vt-identity (n &key (type 'double-float))
  "创建 n×n 单位矩阵。"
  (vt-eye n :type type))

(defun vt-logspace
    (start stop num &key (base 10.0d0) (endpoint t)
		      (type 'double-float))
  "返回对数间隔的 1D 张量。"
  (vt-map (lambda (x) (expt base x))
          (vt-linspace start stop num :endpoint endpoint
				      :type type)))

(defun vt-kron (a b)
  "计算任意维张量的 Kronecker 积，行为与 NumPy 完全一致。"
  (let ((a-shape (vt-shape a))
        (b-shape (vt-shape b)))
    ;; 标量视为形状 (1)
    (when (null a-shape) (setf a-shape '(1)))
    (when (null b-shape) (setf b-shape '(1)))
    (let* ((ndim-a (length a-shape))
           (ndim-b (length b-shape))
           (max-ndim (max ndim-a ndim-b))
           ;; 维度对齐：在前面补 1 使秩相等
           (a-shape-padded (append (make-list (- max-ndim ndim-a)
					      :initial-element 1)
                                   a-shape))
           (b-shape-padded (append (make-list (- max-ndim ndim-b)
					      :initial-element 1)
                                   b-shape))
           (a-new-shape nil)
           (b-new-shape nil)
           (final-shape nil))
      ;; 构建交错形状：A: (d1, 1, d2, 1, ...)，B: (1, e1, 1, e2, ...)
      (loop for da in a-shape-padded
            for db in b-shape-padded
            do (push da a-new-shape)   ; A: da, 1
               (push 1 a-new-shape)
               (push 1 b-new-shape)   ; B: 1, db
               (push db b-new-shape)
               (push (* da db) final-shape))
      (setf a-new-shape (nreverse a-new-shape))
      (setf b-new-shape (nreverse b-new-shape))
      (setf final-shape (nreverse final-shape))
      ;; 重塑并广播相乘，最后 reshape 到目标形状
      (let ((a-reshaped (vt-reshape a a-new-shape))
            (b-reshaped (vt-reshape b b-new-shape)))
        (vt-reshape (vt-* a-reshaped b-reshaped) final-shape)))))

(defun vt-random-int (low high &key (size nil) (type 'fixnum))
  "创建随机整数数组.
  low: 下界(包含)
  high: 上界(不包含)
  size: 形状(nil 表示标量)
  返回: 张量"
  (let ((range (- high low)))
    (if size
        (vt-astype
	 (vt-map (lambda (x)
		   (declare (ignore x))
                   (+ low (random range)))
                 (vt-zeros size :type type))
	 type)
        (coerce (+ low (random range)) type))))

(defun vt-from-function (shape fn &key (type 'double-float))
  "根据函数创建数组.
  shape: 形状列表
  fn: 接受索引列表并返回值的函数
  返回: 张量"
  (let* ((size (vt-shape-to-size shape))
         (data (make-array size :element-type type))
         (result (%make-vt :data data :shape shape 
                           :strides (vt-compute-strides shape)
			   :offset 0))
         (rank (length shape)))
    ;; 递归填充
    (labels
	((recurse (depth indices flat-idx)
           (declare (type fixnum depth flat-idx))
           (if (= depth rank)
               (setf (aref data flat-idx) 
                     (coerce (funcall fn indices) type))
               (let ((dim (nth depth shape))
                     (stride (nth depth (vt-strides result))))
                 (loop for i from 0 below dim
                       do (recurse (1+ depth) 
                                   (append indices (list i))
                                   (+ flat-idx
				      (* i stride))))))))
      (recurse 0 nil 0))
    result))

(defun vt-meshgrid (vts-list &key (indexing :xy) (sparse nil) (copy t))
  "生成坐标矩阵，完全兼容 NumPy 的 meshgrid 语义。
   vts-list : 一维张量列表。
   indexing : :ij（矩阵索引）或 :xy（笛卡尔坐标，默认 :xy）。
   sparse   : 若为 T，返回稀疏网格（形状中非对应轴为 1 的广播视图），否则返回全尺寸网格。
   copy     : 若为 T，强制复制数据；否则尽量返回视图。
   返回     : 张量列表，长度与 vts 相同。"
  (declare (list vts-list))
  (dolist (v vts-list)
    (assert (= (length (vt-shape v)) 1)
            (v) "All inputs to meshgrid must be 1-D"))
  (let* ((nd (length vts-list))
         (dims (mapcar (lambda (v) (first (vt-shape v))) vts-list))
         ;; 计算输出形状（xy 模式下交换前两个维度）
         (output-shape
           (if (and (eq indexing :xy) (>= nd 2))
               (let ((sh (copy-list dims)))
                 (rotatef (first sh) (second sh))
                 sh)
               dims))
         ;; 每个输入在输出形状中对应的轴
         (target-axes
           (if (and (eq indexing :xy) (>= nd 2))
               (let ((axes (loop for i below nd collect i)))
                 (rotatef (first axes) (second axes))
                 axes)
               (loop for i below nd collect i))))
    ;; 生成第 i 个输出的稀疏形状（基于 output-shape）
    (labels ((make-sparse-shape (i)
               (loop for ax from 0 below nd
                     collect (if (= ax (nth i target-axes))
				 (nth i dims)
				 1))))
      (loop for i from 0 below nd
            for v in vts-list
            for src = (if copy (vt-copy v) v)
            for sp = (make-sparse-shape i)
            collect (if sparse
			(vt-reshape src sp)
			(let ((sp-view (vt-reshape src sp)))
			  (vt-broadcast-to sp-view output-shape)))))))

;;; ===========================================
;;; 2. 形状操作扩展
;;; ===========================================

(defun vt-flatten (vt)
  "将多维数组降为一维(返回副本).
  vt: 输入张量
  返回: 一维张量"
  (vt-reshape vt (list (vt-size vt))))

(defun vt-ravel (vt)
  "返回展平后的视图(尽量不复制数据).
  如果输入是连续的,返回视图；否则返回副本."
  (if (vt-contiguous-p vt)
      ;; 连续内存:直接创建视图
      (%make-vt :data (vt-data vt)
                :shape (list (vt-size vt))
                :strides '(1)
                :offset (vt-offset vt))
      ;; 非连续:创建副本
      (vt-flatten vt)))

(defun vt-squeeze (vt &key axis)
  "移除长度为1的维度。支持负轴。"
  (let* ((old-shape (vt-shape vt))
         (old-strides (vt-strides vt))
         (old-offset (vt-offset vt)))
    (if axis
        (let* ((ax (vt-normalize-axis axis (length old-shape)))  ; 规范化
               (axis-size (nth ax old-shape)))
          (unless (= axis-size 1)
            (error "无法挤压非单例维度: axis ~A 大小为 ~A" axis axis-size))
          (%make-vt :data (vt-data vt)
                    :shape (append (subseq old-shape 0 ax)
                                   (subseq old-shape (1+ ax)))
                    :strides (append (subseq old-strides 0 ax)
                                     (subseq old-strides (1+ ax)))
                    :offset old-offset))
        ;; 移除所有长度为1的维度（保持不变）
        (let ((new-shape '())
              (new-strides '()))
          (loop for dim in old-shape
                for stride in old-strides
                when (> dim 1)
                  do (push dim new-shape)
                     (push stride new-strides))
          (when (null new-shape)
            (setf new-shape nil
                  new-strides nil))
          (%make-vt :data (vt-data vt)
                    :shape (nreverse new-shape)
                    :strides (nreverse new-strides)
                    :offset old-offset)))))

(defun vt-expand-dims (vt axis)
  "在指定位置插入新轴.
  axis: 插入位置(0表示最前面)
  返回: 张量视图"
  (let* ((old-shape (vt-shape vt))
         (old-strides (vt-strides vt))
         (old-offset (vt-offset vt))
         (rank (length old-shape))
         (actual-axis (if (< axis 0) (+ rank axis 1) axis)))
    (when (or (< actual-axis 0) (> actual-axis rank))
      (error "轴 ~A 越界(秩 ~A)" axis rank))
    (let ((new-shape
	    (append (subseq old-shape 0 actual-axis)
                    (list 1)
                    (subseq old-shape actual-axis)))
          (new-strides
	    (append (subseq old-strides 0 actual-axis)
                    (list 0)  ; 新轴步长为0
                    (subseq old-strides actual-axis))))
      (%make-vt :data (vt-data vt)
                :shape new-shape
                :strides new-strides
                :offset old-offset))))

(defun vt-unsqueeze (vt axis)
  "在指定位置插入新轴"
  (vt-expand-dims vt axis))

(defun vt-swapaxes (vt axis1 axis2)
  "交换数组的两个轴。axis1 和 axis2 支持负数索引。
   返回：张量视图。"
  (let* ((rank (length (vt-shape vt)))
         (ax1 (vt-normalize-axis axis1 rank))
         (ax2 (vt-normalize-axis axis2 rank)))
    (vt-transpose
     vt
     (loop for i from 0 below rank
           collect (cond ((= i ax1) ax2)
                         ((= i ax2) ax1)
                         (t i))))))

(defun vt-broadcast-to (vt new-shape)
  "将张量广播到新形状，返回零拷贝视图"
  (let* ((old-shape (vt-shape vt))
         (broadcast-shape
	   (vt-broadcast-shapes old-shape new-shape)))
    (unless (equal broadcast-shape new-shape)
      (error "形状 ~A 不能广播到 ~A" old-shape new-shape))
    (%make-vt :data (vt-data vt)
              :shape new-shape
              :strides (vt-broadcast-strides
			old-shape new-shape (vt-strides vt))
              :offset (vt-offset vt))))

(defun vt-repeat (vt repeats &key axis)
  "重复数组元素。重写为切片拼接方式。
   修正：正确处理rep=0、apply调用、空结果情形。"
  (if (null axis)
      ;; 展平后重复
      (let* ((flat (vt-flatten vt))
             (size (vt-size flat))
             (reps (if (listp repeats)
		       repeats
		       (make-list size :initial-element repeats)))
             (parts
	       (loop for i from 0 below size
                     for rep in reps
                     for val = (vt-ref flat i)
                     when (> rep 0)
                       collect (make-vt (list rep) val
					:type (vt-element-type vt)))))
        (if parts
            (apply #'vt-concatenate 0 parts)
            (vt-zeros (list 0) :type (vt-element-type vt))))
      ;; 沿轴重复
      (let* ((sh (vt-shape vt))
             (ax (vt-normalize-axis axis (length sh)))
             (ax-size (nth ax sh))
             (reps (if (listp repeats)
                       repeats
                       (make-list ax-size :initial-element repeats)))
             (slices
	       (loop for i from 0 below ax-size
                     for rep in reps
                     for part = (apply #'vt-slice vt
                                       (loop for d from 0 below (length sh)
                                             collect
					     (if (= d ax)
                                                 `(,i ,(1+ i)) '(:all))))
                     when (> rep 0)
                       collect
		       (if (= rep 1)
                           part
                           (apply #'vt-concatenate ax
                                  (loop repeat rep collect part))))))
        (if slices
            (apply #'vt-concatenate ax slices)
            (let ((zero-shape (copy-list sh)))
              (setf (nth ax zero-shape) 0)
              (vt-zeros zero-shape :type (vt-element-type vt)))))))

(defun vt-tile (vt reps)
  "重复数组构造新数组.
  vt: 输入张量
  reps: 每个维度的重复次数列表
  返回: 新张量"
  (let* ((sh (vt-shape vt))
         (reps (if (listp reps) reps (list reps)))
         (ndim (max (length sh) (length reps)))
         (full-sh (append (make-list (- ndim (length sh))
				     :initial-element 1) sh))
         (full-reps (append (make-list (- ndim (length reps))
				       :initial-element 1)
			    reps))
         (result vt))
    (loop for axis from 0 below ndim
          for rep = (nth axis full-reps)
          when (> rep 1)
            do (setf result (vt-repeat result rep :axis axis)))
    (vt-reshape result (mapcar #'* full-sh full-reps))))

;;; ===========================================
;;; 3. 数组连接与分割
;;; ===========================================

(defun vt-concatenate (axis &rest vts)
  "沿指定轴连接数组.
  axis: 连接轴 (支持 -1 等负数表示)
  vts: 张量列表
  返回: 新张量"
  (when (null vts)
    (error "vt-concatenate 至少需要一个张量"))  
  (let* ((shapes (mapcar #'vt-shape vts))
         (rank (length (car shapes)))
         ;; 处理负轴
         (real-axis (vt-normalize-axis axis rank)))
    ;; 验证形状
    (loop for shape in shapes
          for i from 0
          do (unless (= (length shape) rank)
               (error "张量 ~A 的秩不匹配" i))
             (loop for dim in shape
                   for j from 0
                   unless (or (= j real-axis)
                              (= dim (nth j (car shapes))))
                     do (error "形状不匹配")))
    ;; 计算新形状
    (let* ((new-shape (copy-list (car shapes)))
           (total-axis-size
             (reduce #'+ (mapcar
                          (lambda (s)
                            (nth real-axis s))
                          shapes)))
           (result-type (vt-element-type (car vts))))
      (setf (nth real-axis new-shape) total-axis-size)
      (let ((result (vt-zeros new-shape :type result-type))
            (cur-offset 0))
        ;; 沿轴拷贝数据
        (dolist (vt vts)
          (let* ((axis-size
                   (nth real-axis (vt-shape vt)))
		 (slice-args
                   (loop for i from 0 below rank
                         append
                         (if (= i real-axis)
                             ;; 当前拼接轴: 提供范围
                             (list (list cur-offset
                                         (+ cur-offset
                                            axis-size)))
                             ;; 其他轴: 全部选中
                             (list '(:all))))))
            ;; 将源张量直接写入目标切片
            (setf (apply #'vt-slice result slice-args) vt)
            (incf cur-offset axis-size)))
        result))))

(defun vt-concat (axis &rest vts)
  "沿指定轴连接数组."
  (apply #'vt-concatenate axis vts))

(defun vt-stack (axis &rest vts)
  "沿新轴连接数组.
  axis: 新轴位置
  vts: 张量列表
  返回: 新张量"
  (when (null vts)
    (error "vt-stack 至少需要一个张量"))
  
  (let* ((shapes (mapcar #'vt-shape vts))
         (base-shape (car shapes)))
    ;; 验证形状一致
    (loop for shape in shapes
          for i from 0
          unless (equal shape base-shape)
            do (error "张量 ~A 形状不匹配" i))
    
    ;; 沿 axis 扩展每个数组维度
    (let ((expanded-vts
	    (mapcar (lambda (vt) (vt-expand-dims vt axis))
		    vts)))
      (apply #'vt-concatenate axis expanded-vts))))

(defun vt-vstack (&rest vts)
  "垂直堆叠(沿轴0).
  等价于 concatenate axis=0"
  (apply #'vt-concatenate 0 vts))

(defun vt-hstack (&rest vts)
  "水平堆叠(沿轴1,对于1D数组沿轴0).
  对于1D数组,等价于 concatenate"
  (let ((rank (length (vt-shape (car vts)))))
    (if (= rank 1)
        (apply #'vt-concatenate 0 vts)
        (apply #'vt-concatenate 1 vts))))

(defun vt-dstack (&rest vts)
  "深度堆叠(沿轴2)"
  (apply #'vt-concatenate 2 vts))

(defun vt-vsplit (vt indices-or-sections)
  "垂直分割(沿轴0)"
  (vt-split vt indices-or-sections :axis 0))

(defun vt-hsplit (vt indices-or-sections)
  "水平分割(沿轴1)"
  (vt-split vt indices-or-sections :axis 1))

(defun vt-dsplit (vt indices-or-sections)
  "深度分割(沿轴2)"
  (vt-split vt indices-or-sections :axis 2))

;;; ===========================================
;;; 4. 统计扩展
;;; ===========================================

(defun vt-prod (tensor &key axis)
  "求积.
  axis: 归约轴(nil 表示全局)"
  (nth-value 0 (vt-reduce tensor axis 1
                          (lambda (acc val)
                            (declare (type number acc val))
                            (values (* acc val) nil))
                          :return-arg nil)))

(defun vt-cumsum (tensor &key axis)
  "累积和.
   axis 支持负数.
   返回新张量"
  (if axis
      (let* ((shape (vt-shape tensor))
             (rank (length shape))
             (ax (vt-normalize-axis axis rank))          
             (result (vt-copy tensor)))
        (labels
	    ((recurse (depth in-ptr out-ptr)
               (declare (type fixnum depth in-ptr out-ptr))
               (if (= depth ax)                      
                   ;; 沿累积轴处理
                   (let* ((dim (nth depth shape))
                          (in-stride (nth depth (vt-strides tensor)))
                          (out-stride (nth depth (vt-strides result)))
                          (sum (coerce 0 (vt-element-type tensor))))
                     (loop for i from 0 below dim
                           do (incf sum (aref (vt-data tensor) in-ptr))
                              (setf (aref (vt-data result) out-ptr) sum)
                              (incf in-ptr in-stride)
                              (incf out-ptr out-stride)))
                   ;; 其他维度
                   (when (< depth rank)
                     (let ((dim (nth depth shape))
                           (in-stride (nth depth (vt-strides tensor)))
                           (out-stride (nth depth (vt-strides result))))
                       (loop for i from 0 below dim
                             do (recurse
				 (1+ depth)
                                 (+ in-ptr (* i in-stride))
                                 (+ out-ptr (* i out-stride)))))))))
          (recurse 0 (vt-offset tensor) 0))
        result)
      ;; 展平后累积
      (let* ((flat (vt-flatten tensor))
             (size (vt-size flat))
             (result (vt-zeros (list size) :type (vt-element-type tensor)))
             (sum (coerce 0 (vt-element-type tensor))))
        (loop for i from 0 below size
              do (incf sum (aref (vt-data flat) i))
                 (setf (aref (vt-data result) i) sum))
        result)))

(defun vt-cumprod (tensor &key axis)
  "累积积.
   axis 支持负数.
    返回新张量"
  (if axis
      (let* ((shape (vt-shape tensor))
             (rank (length shape))
             (ax (vt-normalize-axis axis rank))   
             (result (vt-copy tensor)))
        (labels
	    ((recurse (depth in-ptr out-ptr)
               (declare (type fixnum depth in-ptr out-ptr))
               (if (= depth ax)               
                   (let* ((dim (nth depth shape))
                          (in-stride (nth depth (vt-strides tensor)))
                          (out-stride (nth depth (vt-strides result)))
                          (prod (coerce 1 (vt-element-type tensor))))
                     (loop for i from 0 below dim
                           do (setf prod
				    (* prod (aref (vt-data tensor) in-ptr)))
                              (setf (aref (vt-data result) out-ptr) prod)
                              (incf in-ptr in-stride)
                              (incf out-ptr out-stride)))
                   (when (< depth rank)
                     (let ((dim (nth depth shape))
                           (in-stride (nth depth (vt-strides tensor)))
                           (out-stride (nth depth (vt-strides result))))
                       (loop for i from 0 below dim
                             do (recurse
				 (1+ depth)
                                 (+ in-ptr (* i in-stride))
                                 (+ out-ptr (* i out-stride)))))))))
          (recurse 0 (vt-offset tensor) 0))
        result)
      (let* ((flat (vt-flatten tensor))
             (size (vt-size flat))
             (result (vt-zeros (list size) :type (vt-element-type tensor)))
             (prod (coerce 1 (vt-element-type tensor))))
        (loop for i from 0 below size
              do (setf prod (* prod (aref (vt-data flat) i)))
                 (setf (aref (vt-data result) i) prod))
        result)))

(defun vt-unravel-index (offset shape strides)
  "将一维物理偏移 offset 转换为多维逻辑索引（基于 shape 和 strides，行优先序）。"
  (declare (type fixnum offset))
  (loop with rem = offset
        for dim in shape
        for stride in strides
        collect (multiple-value-bind (idx r)
		    (floor rem stride)
                  (setf rem r)
                  idx)))

(defun vt-median (tensor &key axis)
  "中位数。axis 支持负数 (nil 表示全局)。"
  (if axis
      (let* ((shape (vt-shape tensor))
             (rank (length shape))
             (ax (vt-normalize-axis axis rank))
             (out-shape (loop for d in shape for i from 0
                              unless (= i ax) collect d))
             (result (vt-zeros out-shape :type 'double-float))
             (out-strides (vt-strides result)))
        (vt-do-each (ptr val result)
          (declare (ignore val))
          ;; 根据输出指针 ptr 生成输出逻辑索引
          (let ((out-idx (vt-unravel-index ptr out-shape out-strides)))
            ;; 构建切片规格：在归约轴用 :all，其他轴用单元素索引列表
            (let* ((specs (loop for i from 0 below rank
                                if (= i ax) collect '(:all)
                                  else collect (list (pop out-idx))))
                   (fiber (apply #'vt-slice tensor specs))
                   (fiber-size (vt-size fiber)))
              ;; 收集纤维元素，必须通过 vt-ref 读取以处理视图偏移/步长
              (let ((vals (sort (loop for i below fiber-size
                                      collect (vt-ref fiber i))
                                #'<)))
                (setf (aref (vt-data result) ptr)
                      (if (oddp fiber-size)
                          (coerce (nth (floor fiber-size 2) vals)
				  'double-float)
                          (/ (+ (coerce (nth (1- (floor fiber-size 2)) vals)
					'double-float)
                                (coerce (nth (floor fiber-size 2) vals)
					'double-float))
                             2.0d0)))))))
        result)
      ;; 全局中位数
      (let* ((flat (vt-flatten tensor))
             (size (vt-size flat))
             (vals (sort (loop for i below size
                               collect (aref (vt-data flat) i))
                         #'<)))
        (if (oddp size)
            (coerce (nth (floor size 2) vals) 'double-float)
            (/ (+ (coerce (nth (1- (floor size 2)) vals) 'double-float)
                  (coerce (nth (floor size 2) vals) 'double-float))
               2.0d0)))))


(defun percent-from-sorted (sorted q interpolation)
  "从已排序列表 SORTED 中，按分数 Q (0..1) 和插值方法 INTERPOLATION 计算百分位值。
   INTERPOLATION 可选 :LINEAR, :LOWER, :HIGHER, :MIDPOINT, :NEAREST。"
  (let* ((N (length sorted))
         (idx (* q (1- N)))
         (lower (floor idx))
         (upper (min (ceiling idx) (1- N)))
         (frac (- idx lower)))
    (case interpolation
      (:linear
       (if (= lower upper)
           (coerce (nth lower sorted) 'double-float)
           (+ (* (- 1 frac) (coerce (nth lower sorted) 'double-float))
              (* frac (coerce (nth upper sorted) 'double-float)))))
      (:lower
       (coerce (nth lower sorted) 'double-float))
      (:higher
       (coerce (nth upper sorted) 'double-float))
      (:midpoint
       (/ (+ (coerce (nth lower sorted) 'double-float)
             (coerce (nth upper sorted) 'double-float))
          2.0d0))
      (:nearest
       (let ((nearest-idx (if (<= frac 0.5d0) lower upper)))
	 (coerce (nth nearest-idx sorted) 'double-float)))
      (otherwise
       (error "Unknown interpolation method ~A" interpolation)))))

(defun vt-percentile (tensor percentile &key axis (interpolation :linear))
  "计算百分位数。
   PERCENTILE : 0~100 的百分位数值。
   AXIS       : 归约轴，支持负数 (NIL 表示全局)。
   INTERPOLATION : :LINEAR, :LOWER, :HIGHER, :MIDPOINT,
                   :NEAREST (默认 :LINEAR)。"
  (let ((q (/ percentile 100.0d0)))
    (if axis
        (let* ((shape (vt-shape tensor))
               (rank (length shape))
               (ax (vt-normalize-axis axis rank))
               (out-shape (loop for d in shape for i from 0
                                unless (= i ax) collect d))
               (result (vt-zeros out-shape :type 'double-float))
               (out-strides (vt-strides result)))
          (vt-do-each (ptr val result)
            (declare (ignore val))
            (let ((out-idx (vt-unravel-index ptr out-shape out-strides)))
              (let* ((specs (loop for i from 0 below rank
                                  if (= i ax) collect '(:all)
                                    else collect (list (pop out-idx))))
                     (fiber (apply #'vt-slice tensor specs))
                     (fiber-size (vt-size fiber))
                     (vals (sort (loop for i below fiber-size
                                       collect (vt-ref fiber i))
                                 #'<)))
                (setf (aref (vt-data result) ptr)
                      (percent-from-sorted vals q interpolation)))))
          result)
        ;; 全局百分位
        (let* ((flat (vt-flatten tensor))
               (size (vt-size flat))
               (vals (sort (loop for i below size
                                 collect (aref (vt-data flat) i))
                           #'<)))
          (percent-from-sorted vals q interpolation)))))

(defun vt-quantile (tensor q &key axis (interpolation :linear))
  "计算分位数.
  q: 分位数(0-1)"
  (vt-percentile tensor (* q 100) :axis axis
				  :interpolation interpolation))

(defun vt-ptp (tensor &key axis)
  "峰-峰值(最大值 - 最小值)"
  (if axis
      (vt-- (vt-amax tensor :axis axis)
	    (vt-amin tensor :axis axis))
      (- (vt-amax tensor)
	 (vt-amin tensor))))

(defun vt-histogram (tensor &key bins range density)
  "计算直方图。
   bins : bin 数量（默认 10）。
   range : (min, max) 范围。
   density : 若为 T，返回概率密度直方图（总面积=1）。
   返回：(hist, bin-edges) 两个一维张量。"
  (declare (list range))
  (let* ((flat (vt-flatten tensor))
         (data (vt-data flat))
         (size (vt-size flat))
         (bins (or bins 10))
         (data-min (if range (car range) (vt-amin tensor)))
         (data-max (if range (cadr range) (vt-amax tensor)))
         (bin-width (/ (- data-max data-min) bins))
         (hist (make-array bins :element-type 'double-float
                                :initial-element 0.0d0))
         (bin-edges (make-array (1+ bins) :element-type 'double-float)))

    ;; 生成 bin edges
    (loop for i from 0 to bins
          do (setf (aref bin-edges i) (+ data-min (* i bin-width))))

    ;; 统计计数（左闭右开，但最后一个 bin 包含右边界）
    (loop for i from 0 below size
          for val = (aref data i)
          for bin-idx = (cond ((< val data-min) nil)
                              ((>= val data-max) (1- bins))
                              (t (floor (- val data-min) bin-width)))
          when bin-idx
            do (incf (aref hist bin-idx)))

    ;; 密度归一化
    (when density
      (let ((total (reduce #'+ hist)))
        (if (zerop total)
            ;; 无数据时，返回平坦密度以避免除零（或全零）
            (loop for i from 0 below bins
                  do (setf (aref hist i) 0.0d0))
            (loop for i from 0 below bins
                  do (setf (aref hist i)
                           (/ (aref hist i) (* total bin-width)))))))

    ;; 返回两个 VT
    (values (vt-from-sequence (coerce hist 'list)
			      :type 'double-float)
            (vt-from-sequence (coerce bin-edges 'list)
			      :type 'double-float))))

;;; ===========================================
;;; 5. 逻辑运算扩展
;;; ===========================================

(defun vt-logical-and (t1 t2)
  "逻辑与"
  (vt-map (lambda (a b)
	    (if (and (/= a 0)
		     (/= b 0))
		1.0d0 0.0d0))
	  t1 t2))

(defun vt-logical-or (t1 t2)
  "逻辑或"
  (vt-map (lambda (a b)
	    (if (or (/= a 0)
		    (/= b 0))
		1.0d0 0.0d0))
	  t1 t2))

(defun vt-logical-not (vt)
  "逻辑非"
  (vt-map (lambda (a)
	    (if (= a 0)
		1.0d0 0.0d0))
	  vt))

(defun vt-logical-xor (t1 t2)
  "逻辑异或"
  (vt-map (lambda (a b) 
            (if (/= (if (= a 0) 0 1)
		    (if (= b 0) 0 1))
		1.0d0
		0.0d0)) 
          t1 t2))

(defun vt-all (condition &key axis)
  "检查所有元素是否为真"
  (nth-value
   0 (vt-reduce condition axis 1.0d0
                (lambda (acc val)
                  (declare (type number acc val))
                  (values (if (and (/= acc 0) (/= val 0))
			      1.0d0 0.0d0)
			  nil))
                :return-arg nil)))

(defun vt-any (condition &key axis)
  "检查是否存在任一元素为真"
  (nth-value
   0 (vt-reduce condition axis 0.0d0
                (lambda (acc val)
                  (declare (type number acc val))
                  (values (if (or (/= acc 0) (/= val 0))
			      1.0d0 0.0d0)
			  nil))
                :return-arg nil)))

(defun vt-isclose (t1 t2 &key (rtol 1e-5) (atol 1e-8))
  "判断两个数组元素是否在容差范围内接近"
  (vt-map (lambda (a b)
            (let ((diff (abs (- a b))))
              (if (<= diff (+ atol
			      (* rtol
				 (max (abs a)
				      (abs b)))))
                  1.0d0 0.0d0)))
          t1 t2))

(defun vt-allclose (t1 t2 &key (rtol 1e-5) (atol 1e-8))
  "判断两个数组整体是否在容差范围内接近"
  (= (vt-all (vt-isclose t1 t2 :rtol rtol :atol atol)) 1.0d0))

(defun vt-isfinite (vt)
  "检查是否为有限值"
  (vt-map (lambda (x) 
            (declare (type double-float x))
            (if (and (< x most-positive-double-float)
                     (> x most-negative-double-float))
                1.0d0 0.0d0))
          vt))

(defun vt-isinf (vt)
  "检查是否为无穷"
  (vt-map (lambda (x)
            (declare (type double-float x))
            (if (or (= x most-positive-double-float)
                    (= x most-negative-double-float))
                1.0d0 0.0d0))
          vt))

(defun vt-isnan (vt)
  "检查是否为 NaN"
  (vt-map (lambda (x)
            (declare (type double-float x))
            (if (/= x x)  ; NaN 不等于自身
                1.0d0 0.0d0))
          vt))



;;; ===========================================
;;; 6. 排序与搜索
;;; ===========================================

(defun vt-sort (tensor &key (axis -1))
  "沿指定轴排序；axis 为 nil 时展平后排序。axis 支持负数。"
  (if axis
      (let* ((shape (vt-shape tensor))
             (rank (length shape))
             ;; 关键：归一化负数轴
             (ax (vt-normalize-axis axis rank))
             (ax-dim (nth ax shape))
             (in-strides (vt-strides tensor))
             (in-offset (vt-offset tensor))
             (in-data (vt-data tensor))
             ;; 结果张量完全复制输入（形状、类型、数据）
             (result (vt-copy tensor))
             (out-strides (vt-strides result))
             (out-data (vt-data result)))
        (labels
	    ((recurse (depth in-ptr out-ptr)
               (cond
                 ((= depth ax)
                  ;; 抵达排序轴：收集该“纤维”上的值，排序后写回
                  (let* ((in-stride (nth ax in-strides))
                         (out-stride (nth ax out-strides))
                         ;; 收集 (值 . 原索引) 对，用于稳定排序
                         (pairs
                           (loop for i of-type fixnum from 0 below ax-dim
                                 for off = (+ in-ptr (* i in-stride))
                                 collect (cons (aref in-data off) i))))
                    ;; 稳定排序（按值排序，相等时保持原顺序）
                    (setq pairs (stable-sort pairs #'< :key #'car))
                    ;; 将排序后的值按顺序写回结果张量
                    (loop for (val . nil) in pairs
                          for off = out-ptr then (+ off out-stride)
                          do (setf (aref out-data off) val))))
                 
                 ((< depth rank)
                  ;; 非排序轴：遍历该维度，递归进入下一深度
                  (let ((dim (nth depth shape))
                        (in-stride (nth depth in-strides))
                        (out-stride (nth depth out-strides)))
                    (loop for i of-type fixnum from 0 below dim do
                      (recurse (1+ depth)
                               (+ in-ptr (* i in-stride))
                               (+ out-ptr (* i out-stride))))))
                 
                 (t
                  ;; depth == rank 不会发生，若发生则忽略
                  nil))))
          (recurse 0 in-offset 0))
        result)
      ;; axis = nil：展平为一维后排序
      (let* ((flat (vt-flatten tensor))
             (data (coerce (vt-data flat) 'list))
             (sorted (stable-sort data #'<)))
        (vt-from-sequence sorted :type (vt-element-type tensor)))))

(defun vt-argsort (tensor &key (axis -1))
  "返回沿指定轴的排序索引张量，形状与输入相同。
   axis 默认 -1（最后一轴），支持负数；axis = nil 时展平排序。"
  (if (null axis)
      ;; 展平排序（保持不变）
      (let* ((flat (vt-ravel tensor))
             (n (vt-size flat))
             (data (vt-data flat))
             (pairs (loop for i from 0 below n
                          collect (cons i (aref data i))))
             (sorted (if (fboundp 'stable-sort)
                         (stable-sort pairs #'< :key #'cdr)
                         (sort pairs #'< :key #'cdr))))
        (%make-vt :data (make-array n :element-type 'fixnum
                                      :initial-contents
				      (mapcar #'car sorted))
                  :shape (list n)
                  :strides '(1)
                  :offset 0))
      ;; 沿特定轴排序（修复版本）
      (let* ((shape (vt-shape tensor))
             (rank (length shape))
             (ax (vt-normalize-axis axis rank))
             (ax-dim (nth ax shape))
             (in-strides (vt-strides tensor))
             (in-offset (vt-offset tensor))
             (in-data (vt-data tensor))
             (result (vt-zeros shape :type 'fixnum))
             (out-strides (vt-strides result))
             (out-data (vt-data result)))
        (labels
            ((recurse (depth in-ptr out-ptr)
               (cond
                 ((= depth rank) nil)
                 ((< depth ax)
                  (let ((dim (nth depth shape))
                        (in-stride (nth depth in-strides))
                        (out-stride (nth depth out-strides)))
                    (dotimes (i dim)
                      (recurse (1+ depth)
                               (+ in-ptr (* i in-stride))
                               (+ out-ptr (* i out-stride))))))
                 ((= depth ax)
                  (let* ((in-stride (nth ax in-strides))
                         (out-stride (nth ax out-strides))
                         (tail-dims (subseq shape (1+ ax)))
                         (tail-rank (length tail-dims))
                         (tail-strides (subseq out-strides (1+ ax)))
                         (tail-size
			   (reduce #'* tail-dims :initial-value 1)))
                    ;; 预计算尾部每个维度在原始步长中的偏移
                    (let ((tail-in-strides
                            (loop for idx from 0 below tail-rank
                                  collect (nth (+ ax 1 idx) in-strides))))
                      (dotimes (tail-i tail-size)
                        (let ((rem tail-i)
                              (extra-in-off 0)
                              (extra-out-off 0))
                          ;; 计算当前组合的额外偏移（正确乘以 r）
                          (loop for idx from 0 below tail-rank
                                for dim in tail-dims
                                for in-strd in tail-in-strides
                                for out-strd in tail-strides
                                do (multiple-value-bind (q r)
				       (floor rem dim)
                                     (incf extra-in-off (* r in-strd))
                                     (incf extra-out-off (* r out-strd))
                                     (setf rem q)))
                          ;; 收集当前纤维上所有元素（值 + 原始位置）
                          (let ((pairs nil))
                            (dotimes (pos ax-dim)
                              (let* ((in-off (+ in-ptr
						(* pos in-stride)
						extra-in-off))
                                     (val (aref in-data in-off)))
                                (push (cons pos val) pairs)))
                            (setf pairs (stable-sort (nreverse pairs)
						     #'<
						     :key #'cdr))
                            ;; 写回排序后的索引
                            (dotimes (pos ax-dim)
                              (let* ((out-off (+ out-ptr
						 (* pos out-stride)
						 extra-out-off))
                                     (src-idx (car (nth pos pairs))))
                                (setf (aref out-data out-off)
				      src-idx))))))))))))
          (recurse 0 in-offset 0))
        result)))

(defun vt-unique (tensor &key return-index return-inverse return-counts)
  "返回展平后张量中的唯一元素，自动排序（升序）。
   支持以下关键字（默认均为 nil）：
   :return-index    - 若为真，返回首次出现索引（与唯一值对应的一维 fixnum 张量）
   :return-inverse  - 若为真，返回逆索引（一维 fixnum 张量，长度等于展平后元素数，
                       元素为对应位置在唯一值数组中的下标）
   :return-counts   - 若为真，返回每个唯一值的出现次数（一维 fixnum 张量）
   返回值规则：
   - 若所有关键字均为 nil，只返回唯一值张量。
   - 若任一关键字为真，则按顺序返回：唯一值、索引（若请求）、逆索引（若请求）、计数（若请求）。
   未请求的项在返回值中不会被提供。
   注意：逆索引始终保持一维，对应展平后的顺序。"
  (let* ((flat (vt-flatten tensor)) ;; 处理视图，获取扁平数据
         (n (vt-size flat))
         (src-data (vt-data flat))
         (elem-type (vt-element-type flat))
         ;; 初始化索引向量 0..n-1
         (sorted-idx (make-array n :element-type 'fixnum
                                   :initial-contents
				   (loop for i below n collect i))))
    (declare (type (simple-array * (*)) src-data)
             (type (simple-array fixnum (*)) sorted-idx)
             (type fixnum n))
    ;; 稳定排序索引，键为对应的元素值
    (stable-sort sorted-idx #'< :key (lambda (i) (aref src-data i)))
    ;; 动态数组收集结果
    (let ((unique-vals (make-array 0 :element-type elem-type
                                     :adjustable t :fill-pointer t))
          (first-idx   (make-array 0 :element-type 'fixnum
                                     :adjustable t :fill-pointer t))
          (cnts        (make-array 0 :element-type 'fixnum
                                     :adjustable t :fill-pointer t))
          (inverse     (make-array n :element-type 'fixnum
				     :initial-element 0)))
      (loop with pos = 0
            while (< pos n)
            for uniq-num from 0
            for idx0 = (aref sorted-idx pos)
            for val = (aref src-data idx0)
            for start = pos
            do (vector-push-extend val unique-vals)
               (vector-push-extend idx0 first-idx)
               ;; 处理所有等于当前值的元素并填充逆索引
               (loop while (and (< pos n)
				(= val
				   (aref src-data
					 (aref sorted-idx pos))))
                     for orig = (aref sorted-idx pos)
                     do (setf (aref inverse orig) uniq-num)
                        (incf pos))
               (vector-push-extend (- pos start) cnts))
      ;; 转换为 VT
      (let ((uniq-vt (vt-from-sequence (coerce unique-vals 'list)
				       :type elem-type))
            (idx-vt  (when return-index
                       (vt-from-sequence (coerce first-idx 'list)
					 :type 'fixnum)))
            (inv-vt  (when return-inverse
                       (let ((v (make-array n :element-type 'fixnum)))
                         (dotimes (i n) (setf (aref v i)
					      (aref inverse i)))
                         (%make-vt :data v
				   :shape (list n)
				   :strides '(1)
				   :offset 0))))
            (cnt-vt  (when return-counts
                       (vt-from-sequence (coerce cnts 'list)
					 :type 'fixnum))))
        (if (or return-index return-inverse return-counts)
            (values uniq-vt
                    (when return-index idx-vt)
                    (when return-inverse inv-vt)
                    (when return-counts cnt-vt))
            uniq-vt)))))

(defun vt-nonzero (condition)
  "返回非零元素的索引(已存在 vt-where)"
  (vt-where condition))

(defun vt-extract (condition tensor)
  "根据条件从数组中提取元素"
  (let* ((flat-cond (vt-flatten condition))
         (flat-data (vt-flatten tensor))
         (cond-data (vt-data flat-cond))
         (tensor-data (vt-data flat-data))
         (size (vt-size flat-data))
         (result-list '()))
    (loop for i from 0 below size
          when (/= (aref cond-data i) 0)
            do (push (aref tensor-data i) result-list))
    (vt-from-sequence (nreverse result-list))))

(defun vt-searchsorted (tensor values &key side)
  "在有序数组中查找插入点"
  (declare (ignore side))
  (let* ((flat (vt-flatten tensor))
         (data (vt-data flat))
         (size (vt-size flat))
         (values-flat (vt-flatten values))
         (v-data (vt-data values-flat))
         (v-size (vt-size values-flat))
         (result (make-array v-size :element-type 'fixnum)))
    (loop for i from 0 below v-size
          for val = (aref v-data i)
          do (setf (aref result i)
                   (loop for j from 0 below size
                         when (>= (aref data j) val)
                           return j
                         finally (return size))))
    (%make-vt :data result :shape (list v-size)
	      :strides '(1) :offset 0)))

;;; ===========================================
;;; 7. 数据类型转换
;;; ===========================================

(defun vt-astype (tensor new-type)
  "将张量转换为新类型。浮点数→整数时使用截断 (truncate)，其余使用 coerce。"
  (let* ((shape (vt-shape tensor))
	 (type (if (eq new-type 'fixnum) 'fixnum 'double-float))
         (new (vt-zeros shape :type type))
         (new-data (vt-data new))
         (offset (vt-offset new))
         (in-data (vt-data tensor))
         (in-strides (vt-strides tensor))
         (in-offset (vt-offset tensor))
         (rank (length shape))
         (converter (if (eq new-type 'fixnum)
                        #'truncate
                        (lambda (x) (coerce x type)))))
    (labels ((copy-rec (depth in-ptr out-ptr)
               (if (= depth rank)
                   (setf (aref new-data out-ptr)
			 (funcall converter (aref in-data in-ptr)))
                   (let ((dim (nth depth shape))
                         (in-stride (nth depth in-strides))
                         (out-stride (nth depth (vt-strides new))))
                     (loop for i below dim do
                       (copy-rec (1+ depth) in-ptr out-ptr)
                       (incf in-ptr in-stride)
                       (incf out-ptr out-stride))))))
      (copy-rec 0 in-offset offset))
    new))

;;; ===========================================
;;; 8. 集合操作
;;; ===========================================

(defun vt-intersect1d (t1 t2)
  "交集"
  (let* ((u1 (vt-unique t1))
         (u2 (vt-unique t2))
         (u2-set (coerce (vt-data u2) 'list)))
    (let ((result '()))
      (vt-do-each (ptr val u1)
	(declare (ignore ptr))
	(when (member val u2-set)
	  (push val result)))
      (vt-from-sequence (sort result #'<)))))

(defun vt-union1d (t1 t2)
  "并集"
  (let* ((u1 (vt-unique t1))
         (u2 (vt-unique t2)))
    (vt-unique (vt-concatenate 0 u1 u2))))

(defun vt-setdiff1d (t1 t2)
  "差集(在 t1 但不在 t2)"
  (let* ((u1 (vt-unique t1))
         (u2 (vt-unique t2))
         (u2-set (coerce (vt-data u2) 'list)))
    (let ((result '()))
      (vt-do-each (ptr val u1)
	(declare (ignore ptr))
	(unless (member val u2-set)
	  (push val result)))
      (vt-from-sequence (sort result #'<)))))

(defun vt-setxor1d (t1 t2)
  "对称差集"
  (let* ((u1 (vt-unique t1))
         (u2 (vt-unique t2))
         (u1-set (coerce (vt-data u1) 'list))
         (u2-set (coerce (vt-data u2) 'list)))
    (let ((result '()))
      (vt-do-each (ptr val u1)
	(declare (ignore ptr))
	(unless (member val u2-set)
	  (push val result)))
      (vt-do-each (ptr val u2)
	(declare (ignore ptr))
	(unless (member val u1-set)
	  (push val result)))
      (vt-from-sequence (sort result #'<)))))

(defun vt-in1d (t1 t2)
  "检查数组元素是否在另一个数组中"
  (let ((t2-set (coerce (vt-data (vt-unique t2)) 'list)))
    (vt-map (lambda (x) (if (member x t2-set) 1.0d0 0.0d0))
	    t1)))

;;; ===========================================
;;; 9. 其他数学运算
;;; ===========================================

(defun vt-clip (tensor min max)
  "将数值限制在指定范围内"
  (vt-map (lambda (x) (max min (min max x))) tensor))

(defun vt-gradient (tensor &key (spacing 1.0d0) axis)
  "计算张量的梯度（沿指定轴的二阶中心差分）。
   axis  : nil → 全部轴, 整数或整数列表 → 指定轴（支持负数）。
   spacing: 标量（统一间距）、列表（按轴间距，长度需与轴数一致）、
            或单轴配合 1D 张量作为坐标数组。
   返回：若 axis 为 nil 或列表，返回梯度张量的列表；若 axis 为单个整数，返回单个张量。"
  (let* ((shape (vt-shape tensor))
         (rank (length shape))
         (axes
	   (cond ((null axis)
		  (loop for i below rank collect i))
                 ((integerp axis)
		  (list (vt-normalize-axis axis rank)))
                 ((listp axis)
		  (mapcar (lambda (a) (vt-normalize-axis a rank))
			  axis))
                 (t (error "axis must be nil, integer or list of integers.")))))
    (assert (every (lambda (ax) (<= 0 ax (1- rank))) axes))
    (let ((spacings
            (cond ((numberp spacing)
                   (make-list (length axes) :initial-element spacing))
                  ((listp spacing)
                   (assert (= (length spacing) (length axes)))
                   spacing)
                  ((typep spacing 'vt)
                   (when (> (length axes) 1)
                     (error "When spacing is a tensor, axis must be a single axis."))
                   (list spacing))
                  (t (error "spacing must be a number, list or 1D tensor.")))))
      (labels
	  ((slice-specs (ax start end)
	     "生成沿轴 AX 的切片规格：从 start (包含) 到 end (不包含)。"
	     (loop for d from 0 below rank
                   collect (if (= d ax) (list start end) '(:all))))
           (grad-along-axis (ax sp)
	     (let* ((n (nth ax shape)))
               (when (< n 2)
                 (error "Tensor size ~D along axis ~D is too small for gradient." n ax))
               (if (numberp sp)
                   ;; ---------- 等间距 ----------
                   (if (= n 2)
                       ;; 轴长度=2，仅用一阶差分（左右边界相同）
                       (let ((edge
			       (vt-/ (vt-- (apply #'vt-slice tensor (slice-specs ax 1 2))
                                           (apply #'vt-slice tensor (slice-specs ax 0 1)))
                                     sp)))
                         (vt-concatenate ax edge edge))
                       ;; 正常长度（n≥3），边界一阶差分 + 内部中心差分
                       (let* ((left-edge  (vt-/ (vt-- (apply #'vt-slice tensor (slice-specs ax 1 2))
                                                      (apply #'vt-slice tensor (slice-specs ax 0 1)))
                                                sp))
                              (inner      (vt-/ (vt-- (apply #'vt-slice tensor (slice-specs ax 2 n))
                                                      (apply #'vt-slice tensor (slice-specs ax 0 (- n 2))))
                                                (* 2.0d0 sp)))
                              (right-edge (vt-/ (vt-- (apply #'vt-slice tensor (slice-specs ax (1- n) n))
                                                      (apply #'vt-slice tensor (slice-specs ax (- n 2) (1- n))))
                                                sp)))
                         (vt-concatenate ax left-edge inner right-edge)))
                   ;; ---------- 非均匀间距（坐标数组） ----------
                   (let* ((h-vt (ensure-vt sp))
                          (n-h (vt-size h-vt)))
		     (assert (= n n-h) (sp) "Spacing array must have same length as axis.")
		     (if (= n 2)
                         ;; 轴长度=2，一阶差分重复
                         (let* ((h-diff (vt-- (apply #'vt-slice h-vt (list (list 1 2)))
                                              (apply #'vt-slice h-vt (list (list 0 1)))))
                                (df     (vt-- (apply #'vt-slice tensor (slice-specs ax 1 2))
                                              (apply #'vt-slice tensor (slice-specs ax 0 1))))
                                (grad   (vt-/ df h-diff)))
                           (vt-concatenate ax grad grad))
                         ;; 正常长度（n≥3）
                         (let* ((h-diff-left  (vt-- (apply #'vt-slice h-vt (list (list 1 2)))
						    (apply #'vt-slice h-vt (list (list 0 1)))))
                                (h-diff-right (vt-- (apply #'vt-slice h-vt (list (list (1- n) n)))
						    (apply #'vt-slice h-vt (list (list (- n 2) (1- n))))))
                                (h-diff-inner (vt-- (apply #'vt-slice h-vt (list (list 2 n)))
						    (apply #'vt-slice h-vt (list (list 0 (- n 2))))))
                                (df-left  (vt-- (apply #'vt-slice tensor (slice-specs ax 1 2))
                                                (apply #'vt-slice tensor (slice-specs ax 0 1))))
                                (df-right (vt-- (apply #'vt-slice tensor (slice-specs ax (1- n) n))
                                                (apply #'vt-slice tensor (slice-specs ax (- n 2) (1- n)))))
                                (df-inner (vt-- (apply #'vt-slice tensor (slice-specs ax 2 n))
                                                (apply #'vt-slice tensor (slice-specs ax 0 (- n 2))))))
                           (vt-concatenate ax
					   (vt-/ df-left h-diff-left)
					   (vt-/ df-inner h-diff-inner)
					   (vt-/ df-right h-diff-right)))))))))
        ;; 对所有轴计算
        (let ((results (loop for ax in axes
                             for sp in spacings
                             collect (grad-along-axis ax sp))))
          (if (and (integerp axis) (null (cdr axes)))  ; 单轴直接返回张量
              (car results)
              results))))))


;;; ===========================================
;;; 10. 索引高级功能
;;; ===========================================

(defun vt-take (tensor indices &key axis)
  "从张量中按索引取值。支持多维 indices 和负数 axis。
   当 axis=nil 且 indices 为标量数字时，直接返回数值；
   否则返回张量。"
  ;; 帮助函数：将任意形式的 indices 转为 fixnum 简单数组
  (labels
      ((ensure-fixnum-1d (vt)
         (let* ((len (vt-size vt))
                (arr (make-array len :element-type 'fixnum)))
           (if (eq (vt-element-type vt) 'double-float)
               (dotimes (i len)
                 (setf (aref arr i) (truncate (aref (vt-data vt) i))))
               (dotimes (i len)
                 (setf (aref arr i) (aref (vt-data vt) i))))
           (values arr len))))

    (if (null axis)
        ;; ======== axis = nil ========
        (let ((idx-vt (ensure-vt indices)))
          (if (null (vt-shape idx-vt))
              ;; 标量索引：直接返回数值
              (let* ((flat (vt-ravel tensor))
                     (size (vt-size flat))
                     (idx (if (eq (vt-element-type idx-vt) 'double-float)
                              (truncate (aref (vt-data idx-vt) 0))
                              (aref (vt-data idx-vt) 0))))
                (unless (<= 0 idx (1- size))
                  (error "索引 ~D 越界，展平大小 ~D" idx size))
                (aref (vt-data flat) idx))
              ;; 非标量索引：保留形状
              (let* ((flat (vt-ravel tensor))
                     (size (vt-size flat))
                     (idx-shape (vt-shape idx-vt))
                     (out (vt-zeros idx-shape
				    :type (vt-element-type tensor)))
                     (idx-len (vt-size idx-vt)))
                (multiple-value-bind (idx-arr _)
                    (ensure-fixnum-1d idx-vt)
                  (declare (ignore _))
                  (let ((flat-data (vt-data flat))
                        (out-data (vt-data out)))
                    (dotimes (i idx-len)
                      (let ((idx (aref idx-arr i)))
                        (unless (<= 0 idx (1- size))
                          (error "索引 ~D 越界，展平大小 ~D" idx size))
                        (setf (aref out-data i) (aref flat-data idx)))))
                  out))))

	;; ======== axis 整数 ========
	(let* ((shape (vt-shape tensor))
               (rank (length shape))
               (ax (vt-normalize-axis axis rank))
               (ax-dim (nth ax shape))
               (idx-vt (ensure-vt indices))
               (idx-shape (vt-shape idx-vt))
               (idx-len (length idx-shape))
               (idx-size (vt-size idx-vt))
               (out-shape (append (subseq shape 0 ax)
                                  idx-shape
                                  (subseq shape (1+ ax))))
               (out (vt-zeros out-shape :type (vt-element-type tensor)))
               (idx-strides (vt-compute-strides idx-shape))
               (idx-arr (make-array idx-size :element-type 'fixnum)))
          ;; 填充 idx-arr
          (let ((raw (vt-data (vt-flatten idx-vt))))
            (if (eq (vt-element-type idx-vt) 'double-float)
		(dotimes (i idx-size)
                  (setf (aref idx-arr i) (truncate (aref raw i))))
		(dotimes (i idx-size)
                  (setf (aref idx-arr i) (aref raw i)))))
          (let ((in-strides (vt-strides tensor))
		(out-strides (vt-strides out))
		(in-off (vt-offset tensor))
		(out-off 0))
            (labels
		((recurse (depth in-ptr out-ptr)
                   (if (= depth rank)
                       (setf (aref (vt-data out) out-ptr)
			     (aref (vt-data tensor) in-ptr))
                       (if (= depth ax)
                           (walk-idx 0 0 in-ptr out-ptr)
                           (let ((dim (nth depth shape))
                                 (in-str (nth depth in-strides))
                                 (out-str
				   (let ((out-idx
					   (if (< depth ax)
                                               depth
                                               (+ depth idx-len -1))))
                                     (nth out-idx out-strides))))
			     (dotimes (i dim)
                               (recurse (1+ depth)
                                        (+ in-ptr (* i in-str))
                                        (+ out-ptr (* i out-str))))))))
                 (walk-idx (d idx-off in-ptr out-base)
                   (if (= d idx-len)
                       (let ((idx (aref idx-arr idx-off)))
                         (unless (<= 0 idx (1- ax-dim))
                           (error "索引 ~D 越界，轴 ~D 大小 ~D"
				  idx ax ax-dim))
                         (recurse (1+ ax)
                                  (+ in-ptr (* idx (nth ax in-strides)))
                                  out-base))
                       (let ((dim (nth d idx-shape))
			     (idx-str (nth d idx-strides))
			     (out-str (nth (+ ax d) out-strides)))
                         (dotimes (i dim)
                           (walk-idx (1+ d)
				     (+ idx-off (* i idx-str))
				     in-ptr
				     (+ out-base (* i out-str))))))))
              (recurse 0 in-off out-off))
            out)))))


(defun vt-put (tensor indices values)
  "将 values 按 indices 指定的展平（行优先）索引放入 tensor。
   tensor 被原地修改并返回。
   indices: 整数、一维张量或列表。
   values: 标量、一维张量或列表，长度不足时循环使用。
   注意：索引必须是 [0, 总元素数) 内的非负整数。"
  (let* ((total-size (vt-size tensor))
         ;; 将 indices 统一为整数列表
         (idx-list
	   (if (numberp indices)
               (list (round indices))
               (let ((flat (vt-flatten (ensure-vt indices))))
                 (loop for i below (vt-size flat)
                       collect (truncate (aref (vt-data flat) i))))))
         ;; 将 values 统一为列表
         (val-list (cond ((numberp values) (list values))
			 ((or (listp values)
			      (arrayp values))
			  (vt-flatten-sequence values))
			 ((vt-p values)
			  (let ((flat (vt-flatten (ensure-vt values))))
                            (loop for i below (vt-size flat)
				  collect (aref (vt-data flat) i))))))
         (n-values (length val-list))
         (shape (vt-shape tensor))
         (strides (vt-strides tensor))
         (offset (vt-offset tensor))
         (data (vt-data tensor))
         (elem-type (vt-element-type tensor)))
    
    (loop for idx in idx-list
          for i from 0
          for val = (nth (mod i n-values) val-list)
          do (unless (and (integerp idx)
			  (<= 0 idx)
			  (< idx total-size))
               (error "索引 ~D 超出范围 [0, ~D)" idx total-size))
             ;; 一维逻辑索引 → 多维坐标 → 物理偏移
             (let ((remaining idx)
                   (phys-offset offset))
               (loop for dim in (reverse shape)
                     for stride in (reverse strides)
                     do (multiple-value-bind (q r)
			    (floor remaining dim)
                          (incf phys-offset (* r stride))
                          (setf remaining q)))
               ;; 直接写入底层数组，自动进行必要的类型转换
               (setf (aref data phys-offset)
                     (coerce val elem-type))))
    tensor))

(defun vt-choose (choices indices)
  "根据索引数组从多个数组中选择值"
  (let* ((n-choices (length choices))
         (idx-flat (vt-flatten indices))
         (idx-data (vt-data idx-flat))
         (idx-size (vt-size idx-flat))
         (result-data (make-array idx-size
				  :element-type 'double-float)))
    (loop for i from 0 below idx-size
          for idx = (aref idx-data i)
          do (when (or (< idx 0) (>= idx n-choices))
               (error "选择索引 ~A 越界" idx))
             (setf (aref result-data i)
                   (aref (vt-data (nth idx choices)) i)))
    (%make-vt :data result-data :shape (list idx-size)
              :strides '(1) :offset 0)))

(defun vt-select (condlist choicelist &key default)
  "根据多个条件从多个数组中选择值"
  (declare (ignore default))
  (let* ((n-conds (length condlist))
         (shape (vt-shape (car condlist)))
         (result (vt-zeros shape)))
    (vt-do-each (ptr val result)
      (declare (ignorable  val))
      (loop for i from 0 below n-conds
	    when (/= (aref (vt-data (nth i condlist)) ptr) 0)
	      do (setf (aref (vt-data result) ptr)
		       (aref (vt-data (nth i choicelist)) ptr))
		 (return)))
    result))

(defun vt-flip (vt &key axis)
  "沿指定轴翻转，返回零拷贝视图"
  (if (null axis)
      ;; 全部翻转
      (let* ((strides (vt-strides vt))
             (shape (vt-shape vt))
             (new-strides (mapcar #'- strides))
             (new-offset (+ (vt-offset vt)
                            (loop for dim in shape
                                  for stride in strides
                                  sum (* (1- dim) stride)))))
        (%make-vt :data (vt-data vt)
                  :shape shape
                  :strides new-strides
                  :offset new-offset))
      (let* ((rank (length (vt-shape vt)))
             (ax (vt-normalize-axis axis rank))
             (strides (copy-list (vt-strides vt)))
             (dim (nth ax (vt-shape vt)))
             (old-stride (nth ax strides)))
        (setf (nth ax strides) (- old-stride))
        (%make-vt :data (vt-data vt)
                  :shape (vt-shape vt)
                  :strides strides
                  :offset (+ (vt-offset vt)
			     (* (1- dim) old-stride))))))

(defun vt-roll (vt shift &key axis)
  "滚动元素。shift 可为整数或列表，axis 可为整数、列表或 nil。
   多轴模式下 shift 与 axis 长度需相等。"
  ;; 统一 shift 为列表
  (let ((shift-list (if (listp shift) shift (list shift))))
    (cond
      ;; 多轴模式：axis 为列表
      ((and axis (listp axis))
       (let ((result (vt-copy vt)))
         (loop for s in shift-list
               for ax in axis
               do (setf result (vt-roll result s :axis ax)))
         result))
      ;; 单轴模式：axis 为整数
      (axis
       (let* ((sh (vt-shape vt))
              (ax (vt-normalize-axis axis (length sh)))
              (n (nth ax sh))
              (s (mod (car shift-list) n)))
         (if (zerop s)
             vt
             (vt-concatenate
	      ax
              (apply #'vt-slice vt
                     (loop for i from 0 below (length sh)
                           collect (if (= i ax)
                                       `(,(- n s) ,n) '(:all))))
              (apply #'vt-slice vt
                     (loop for i from 0 below (length sh)
                           collect (if (= i ax)
                                       `(0 ,(- n s)) '(:all))))))))
      ;; axis = nil：展平后滚动并恢复形状
      (t
       (let* ((flat (vt-flatten vt))
              (n (vt-size flat))
              (s (mod (car shift-list) n)))
         (if (zerop s)
             (vt-reshape flat (vt-shape vt))
             (vt-reshape (vt-concatenate
			  0
                          (vt-slice flat `(,(- n s) ,n))
                          (vt-slice flat `(0 ,(- n s))))
                         (vt-shape vt))))))))


(defun normalize-pad-width (pad-width rank)
  "将 pad-width 标准化为 列表。"
  (let ((pw-list (if (integerp pad-width)
                     (make-list rank :initial-element pad-width)
                     pad-width)))
    (unless (= (length pw-list) rank)
      (error "pad-width length ~A does not match tensor rank ~A"
	     (length pw-list) rank))
    (mapcar (lambda (x)
              (cond ((integerp x) 
                     (if (< x 0)
			 (error "Negative padding is not supported")
			 (list x x)))
                    ((and (consp x) (= (length x) 2))
                     (if (or (< (first x) 0) (< (second x) 0))
                         (error "Negative padding is not supported")
                         x))
                    (t (error "Invalid pad-width element: ~A" x))))
            pw-list)))

(defun apply-pad-mode (mode dist sk side)
  "纯坐标映射：dist 表示距离边缘的距离 (1, 2, 3...)。
   sk 是源张量维度大小。"
  (ecase mode
    (:edge (if (eq side :left) 0 (1- sk)))
    
    (:wrap 
     ;; wrap 左侧起点 sk-1 向左递减；右侧起点 0 向右递增
     (let ((x (if (eq side :left) (- sk dist) (1- dist))))
       (mod x sk)))
    
    (:reflect
     ;; reflect 左侧起点 1 向左递增；右侧起点 sk-2 向右递减
     (when (<= sk 1)
       (error "Cannot reflect on dimension with size <= 1"))
     (let* ((period (* 2 (1- sk)))
            (x (if (eq side :left) dist (- sk 1 dist)))
            (idx (mod x period)))
       (if (< idx sk) idx (- period idx))))
    
    (:symmetric
     ;; symmetric 左侧起点 0 向左递增；右侧起点 sk-1 向右递减
     (let* ((period (* 2 sk))
            (x (if (eq side :left) (- dist) (+ sk dist -1)))
            (idx (mod x period)))
       (if (< idx sk) idx (- period idx 1))))))

(defun vt-pad (vt pad-width &key (mode :constant) (constant-values 0))
  "对张量进行填充，返回新张量。
   pad-width ：整数（所有轴相同）或长度为 rank 的列表，每个元素可为整数或 对。
   mode :constant（默认）, :edge, :wrap, :reflect, :symmetric。
   constant-values ：常数填充值，可用标量或 列表。"
  (let* ((shape (vt-shape vt))
         (rank (length shape))
         (pad (normalize-pad-width pad-width rank))
         
         (cv-list (if (listp constant-values)
		      constant-values
		      (list constant-values constant-values)))
         (c-left (first cv-list))
         (c-right (second cv-list))
         
         (new-shape (loop for s in shape for (b a) in pad
			  collect (+ s b a))))
    
    (when (and (listp constant-values) (not (= (length cv-list) 2)))
      (error "constant-values must be a scalar or a list of exactly 2 elements"))
    
    (vt-from-function 
     new-shape
     (lambda (out-idxs)
       (let ((const-val nil)
             (src-idxs (make-list rank)))
         (loop
	   for d from 0 below rank
           for ok = (nth d out-idxs)
           for (bk ak) in pad
           for sk = (nth d shape)
           do (let ((offset (- ok bk)))
                (declare (fixnum offset))
                (cond
                  ;; 左侧填充区：将负数 offset 转化为正向距离 dist
                  ((< offset 0)
                   (if (eq mode :constant)
                       (unless const-val (setf const-val c-left))
                       (setf (nth d src-idxs) 
                             (apply-pad-mode mode (- offset) sk :left))))
                  ;; 右侧填充区：将越界 offset 转化为正向距离 dist
                  ((>= offset sk)
                   (if (eq mode :constant)
                       (unless const-val (setf const-val c-right))
                       (setf (nth d src-idxs) 
                             ;; CL中 (- a b -1) 等价于 a - b + 1
                             (apply-pad-mode
			      mode (- offset sk -1) sk :right))))
                  ;; 原始数据区
                  (t
                   (setf (nth d src-idxs) offset)))))
         
         (if const-val
             const-val
             (apply #'vt-ref vt src-idxs))))
     :type (vt-element-type vt))))

(defun vt-diff (vt &key (axis -1) (n 1))
  "沿指定轴计算 n 阶差分，与 numpy.diff 兼容。"
  (let ((result vt))
    (loop repeat n
	  do
	     (let* ((sh (vt-shape result))
		    (ax (vt-normalize-axis axis (length sh)))
		    (len (nth ax sh)))
               ;; 当轴长度 <= 1 时，无法做差分，返回空张量
               (when (< len 2)
		 (let ((out-shape (append (subseq sh 0 ax)
					  (list 0)
					  (subseq sh (1+ ax)))))
		   (return-from vt-diff
		     (vt-zeros out-shape
			       :type (vt-element-type result)))))
               ;; left: result[..., 1:, ...] ; right: result[..., :-1, ...]
               (let ((left  (vt-narrow result ax 1 len))
		     (right (vt-narrow result ax 0 (1- len))))
		 (setf result (vt-- left right))))
	  finally (return result))))

(defun vt-bincount (x &key (minlength 0))
  "统计非负整数数组中每个值出现的次数。"
  (let* ((flat (vt-flatten x))
         (maxval (if (zerop (vt-size flat)) -1 (vt-amax flat)))
         (size (max (1+ maxval) minlength))
         (result (make-array size :element-type 'fixnum
				  :initial-element 0)))
    (vt-do-each (ptr val flat)
      (declare (ignore ptr))
      (incf (aref result (truncate val))))
    (%make-vt :data result :shape (list size)
	      :strides '(1) :offset 0)))

(defun vt-digitize (x bins &key (right nil))
  "返回输入值在 bins 中的索引。"
  (let* ((flat-x (vt-flatten x))
         (flat-bins (vt-flatten bins))
         (bin-data (vt-data flat-bins))
         (nbins (vt-size flat-bins))
         (result (make-array (vt-size flat-x)
			     :element-type 'fixnum)))
    (vt-do-each (ptr val flat-x)
      (let ((idx (loop for i from 0 below nbins
                       until (if right
				 (< val (aref bin-data i))
                                 (<= val (aref bin-data i)))
                       finally (return i))))
        (setf (aref result ptr) idx)))
    (%make-vt :data result
	      :shape (vt-shape flat-x)
	      :strides '(1)
	      :offset 0)))

(defun vt-correlate (a v &key (mode :full))
  "一维互相关（不翻转 v），与 numpy.correlate 完全一致。"
  (let* ((a-flat (vt-flatten a))
         (v-flat (vt-flatten v))
         (n (vt-size a-flat))
         (m (vt-size v-flat))
         (a-data (vt-data a-flat))
         (v-data (vt-data v-flat)))
    (flet ((compute-one (k)
             (let ((sum 0.0d0))
               (loop for j from (max 0 (- k)) below (min m (- n k))
                     do (incf sum (* (aref a-data (+ j k))
				     (aref v-data j))))
               sum)))
      ;; 取 full 的长度和偏移
      (let* ((full-len (+ n m -1))
             (offset  (1- m))          ; k=0 对应 full 数组的索引 offset
             (full (make-array full-len :element-type 'double-float)))
        ;; 计算完整的 full 互相关
        (loop for k from (- offset) below n
              for i from 0
              do (setf (aref full i) (compute-one k)))
        (ecase mode
          (:full
           (%make-vt :data full
		     :shape (list full-len)
		     :strides '(1)
		     :offset 0))
          (:valid
           (let* ((len (max 0 (1+ (- n m))))     ; n - m + 1
                  (start offset))
             (let ((data (make-array len :element-type 'double-float)))
               (loop for i from 0 below len
                     for idx from start
                     do (setf (aref data i) (aref full idx)))
               (%make-vt :data data
			 :shape (list len)
			 :strides '(1)
			 :offset 0))))
          (:same
           (let* ((out-len (max n m))
                  (start (floor (- full-len out-len) 2)))
             (let ((data (make-array out-len :element-type 'double-float)))
               (loop for i from 0 below out-len
                     for idx from start
                     do (setf (aref data i) (aref full idx)))
               (%make-vt :data data
			 :shape (list out-len)
			 :strides '(1)
			 :offset 0)))))))))

(defun vt-convolve (a v &key (mode :full))
  "一维卷积。"
  (vt-correlate a (vt-flip v) :mode mode))

(defun vt-trapz (y &key (x nil) (dx 1.0d0) (axis -1))
  "使用梯形法则沿指定轴积分，与 numpy.trapz 一致。"
  (let* ((sh (vt-shape y))
         (ax (vt-normalize-axis axis (length sh)))
         (n (nth ax sh)))
    ;; 轴长度不足 → 返回全零，形状删除该轴
    (when (< n 2)
      (let ((out-shape (append (subseq sh 0 ax)
			       (subseq sh (1+ ax)))))
        (return-from vt-trapz
	  (vt-zeros out-shape :type (vt-element-type y)))))
    ;; 步长向量（长度 = n-1）
    (let ((h (if x
                 (vt-diff (ensure-vt x))          ; 1D,长度 n-1
                 (make-vt (list (1- n)) dx
                          :type (vt-element-type y)))))
      ;; 将 h 的外形扩展为 [1,…,1,n-1,1,…,1] 以便广播
      (let ((h-broadcast-shape
              (append (make-list ax :initial-element 1)
                      (list (1- n))
                      (make-list (- (length sh) ax 1)
				 :initial-element 1))))
        (setf h (vt-reshape h h-broadcast-shape)))
      ;; 梯形平均：0.5 * (y_[:-1] + y_[1:]) * h
      (let* ((left  (vt-narrow y ax 0 (1- n)))   ; 相当于 y[...,:-1,:...]
             (right (vt-narrow y ax 1 n))        ; 相当于 y[...,1:,:...]
             (integrand (vt-map (lambda (l r h)
				  (* 0.5d0 (+ l r) h))
                                left right h)))
        (vt-sum integrand :axis ax)))))

(defun vt-interp (x xp fp &key (left nil) (right nil))
  "一维线性插值。x, xp, fp 均为 1D 张量。"
  (let* ((xp-data (vt-data xp))
         (fp-data (vt-data fp))
         (n (vt-size xp)))
    (assert (= (vt-size fp) n))
    (vt-map
     (lambda (xi)
       (if (<= xi (aref xp-data 0))
           (if left left (aref fp-data 0))
           (if (>= xi (aref xp-data (1- n)))
               (if right right (aref fp-data (1- n)))
               (loop for i from 0 below (1- n)
                     when (and (>= xi (aref xp-data i))
                               (<= xi (aref xp-data (1+ i))))
                       return
		       (+ (aref fp-data i)
			  (* (- (aref fp-data (1+ i))
				(aref fp-data i))
                             (/ (- xi (aref xp-data i))
                                (- (aref xp-data (1+ i))
				   (aref xp-data i)))))))))
     x)))

(defun vt-hypot (t1 &optional t2)
  "计算 (|t1|^2 + |t2|^2)^0.5。支持广播。"
  (vt-sqrt (vt-+ (vt-square t1) (vt-square (if t2 t2 0)))))

(defun vt-sinc (tensor)
  "计算 sinc(x) = sin(pi*x) / (pi*x)，对 x=0 返回 1。"
  (let* ((x*pi (vt-scale tensor pi)))
    (vt-map (lambda (x)
              (if (zerop x) 1.0d0 (/ (sin x) x)))
            x*pi)))

(defun vt-deg2rad (tensor)
  "角度转弧度。"
  (vt-scale tensor (/ pi 180.0d0)))

(defun vt-rad2deg (tensor)
  "弧度转角度。"
  (vt-scale tensor (/ 180.0d0 pi)))

;;; ===========================================
;;; 12. 随机数扩展
;;; ===========================================

(defun get-random-normal ()
  "返回标准正态分布随机数。线程安全，无内部缓存。"
  (let ((u1 (random 1.0d0))
        (u2 (random 1.0d0)))
    (* (sqrt (* -2.0d0 (log u1)))
       (cos (* 2.0d0 pi u2)))))

(defun vt-random-seed (seed)
  "设置随机种子。seed 应为 nil, t, 或 random-state 对象。"
  (setq *random-state*
        (make-random-state
         (if (random-state-p seed)
             seed
             (make-random-state t)))))

(defun vt-random-uniform (shape &key (low 0.0d0) (high 1.0d0))
  "生成均匀分布随机张量。"
  (vt-map (lambda (x)
            (declare (ignore x))
            (+ low (* (- high low) (random 1.0d0))))
          (vt-zeros shape :type 'double-float)))

(defun vt-random-normal (shape &key (mean 0.0d0) (std 1.0d0))
  "生成正态分布随机张量。"
  (let ((res (vt-zeros shape :type 'double-float)))
    (vt-do-each (ptr val res)
      (declare (ignore val))
      (setf (aref (vt-data res) ptr)
            (+ mean (* std (get-random-normal)))))
    res))

(defun vt-random-integers (low high &key (size nil) (type 'fixnum))
  "生成随机整数张量。"
  (vt-random-int low high :size size :type type))

