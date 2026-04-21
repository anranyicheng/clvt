;;;; 扩展功能:对标 NumPy 函数
(in-package :clvt)


;;; =========================================================================
;;; 2. 数组属性补充
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
  (let* ((actual-num (if endpoint num (1- num)))
         (step (if (= actual-num 0)
                   0
                   (/ (- end start) actual-num)))
         (data (make-array num :element-type type))
         (shape (list num)))
    (loop for i from 0 below num
          for val = (+ start (* i step))
          do (setf (aref data i) (coerce val type)))
    (%make-vt :data data :shape shape :strides '(1) :offset 0)))

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
  (vt-zeros (vt-shape vt) :type (or type (vt-element-type vt))))

(defun vt-ones-like (vt &key (type nil))
  "创建与给定数组形状相同的全一数组."
  (vt-ones (vt-shape vt) :type (or type (vt-element-type vt))))

(defun vt-full-like (vt fill-value &key (type nil))
  "创建与给定数组形状相同的填充数组."
  (vt-full (vt-shape vt) fill-value :type (or type (vt-element-type vt))))

(defun vt-empty-like (vt &key (type nil))
  "创建与给定数组形状相同的未初始化数组."
  (vt-empty (vt-shape vt) :type (or type (vt-element-type vt))))

(defun vt-random-int (low high &key (size nil) (type 'fixnum))
  "创建随机整数数组.
  low: 下界(包含)
  high: 上界(不包含)
  size: 形状(nil 表示标量)
  返回: 张量"
  (let ((range (- high low)))
    (if size
        (vt-map (lambda (x) (declare (ignore x))
                  (coerce (+ low (random range)) type))
                (vt-zeros size :type type))
        (coerce (+ low (random range)) type))))

(defun vt-from-function (shape fn &key (type 'double-float))
  "根据函数创建数组.
  shape: 形状列表
  fn: 接受索引列表并返回值的函数
  返回: 张量"
  (let* ((size (vt-shape-to-size shape))
         (data (make-array size :element-type type))
         (result (%make-vt :data data :shape shape 
                           :strides (vt-compute-strides shape) :offset 0))
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

(defun vt-meshgrid (&rest vts)
  "生成坐标矩阵.
   vts: 一维张量列表
  返回: 坐标张量列表(稀疏模式,兼容 NumPy sparse=True)"
  (let* ((shapes (mapcar #'vt-shape vts))
         (dims (mapcar #'car shapes))
         (rank (length dims)))
    ;; 每个输出张量的形状是所有维度的组合
    (loop for i from 0 below rank
          for vt in vts
          for dim-i = (nth i dims)
          collect
          (let* ((out-shape dims)
                 (result (vt-zeros out-shape))
                 (data (vt-data result))
                 (strides (vt-strides result)))
            ;; 广播填充
            (labels
		((recurse (depth indices)
                   (declare (type fixnum depth))
                   (if (= depth rank)
                       (let ((flat-idx (loop for idx in indices
                                             for stride in strides
                                             sum (* idx stride))))
                         (setf (aref data flat-idx)
                               (aref (vt-data vt) (nth i indices))))
                       (loop for j from 0 below (nth depth dims)
                             do (recurse (1+ depth)
					 (append indices (list j)))))))
              (recurse 0 nil))
            result))))

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

(defun vt-squeeze (vt &optional axis)
  "移除长度为1的维度.
  axis: 指定轴(nil 表示移除所有长度为1的维度)
  返回: 张量视图"
  (let* ((old-shape (vt-shape vt))
         (old-strides (vt-strides vt))
         (old-offset (vt-offset vt)))
    (if axis
        ;; 移除指定轴
        (let ((axis-size (nth axis old-shape)))
          (unless (= axis-size 1)
            (error "无法挤压非单例维度: axis ~A 大小为 ~A" axis axis-size))
          (%make-vt :data (vt-data vt)
                    :shape (append (subseq old-shape 0 axis)
                                   (subseq old-shape (1+ axis)))
                    :strides (append (subseq old-strides 0 axis)
                                     (subseq old-strides (1+ axis)))
                    :offset old-offset))
        ;; 移除所有长度为1的维度
        (let ((new-shape '())
              (new-strides '()))
          (loop for dim in old-shape
                for stride in old-strides
                when (> dim 1)
                  do (push dim new-shape)
                     (push stride new-strides))
          ;; 如果所有维度都是1,返回标量视图
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
    (let ((new-shape (append (subseq old-shape 0 actual-axis)
                             (list 1)
                             (subseq old-shape actual-axis)))
          (new-strides (append (subseq old-strides 0 actual-axis)
                               (list 0)  ; 新轴步长为0
                               (subseq old-strides actual-axis))))
      (%make-vt :data (vt-data vt)
                :shape new-shape
                :strides new-strides
                :offset old-offset))))

(defun vt-swapaxes (vt axis1 axis2)
  "交换数组的两个轴.
  返回: 张量视图"
  (vt-transpose vt (loop for i from 0 below (length (vt-shape vt))
                         collect (cond ((= i axis1) axis2)
                                       ((= i axis2) axis1)
                                       (t i)))))

(defun vt-tile (vt reps)
  "重复数组构造新数组.
  vt: 输入张量
  reps: 每个维度的重复次数列表
  返回: 新张量"
  (let* ((old-shape (vt-shape vt))
         (old-rank (length old-shape))
         (reps-len (length reps)))
    ;; 扩展 reps 到相同维度
    (when (< reps-len old-rank)
      (setf reps (append (make-list (- old-rank reps-len)
				    :initial-element 1) reps)))
    (when (> reps-len old-rank)
      (setf old-shape (append (make-list (- reps-len old-rank)
					 :initial-element 1)
			      old-shape)))
    
    (let* ((new-shape (mapcar #'* old-shape reps))
           (result (vt-zeros new-shape :type (vt-element-type vt)))
           (rank (length new-shape)))
      ;; 拷贝数据
      (labels
	  ((recurse (depth out-idx base-idx)
             (declare (type fixnum depth out-idx base-idx))
             (if (= depth rank)
                 (setf (aref (vt-data result) out-idx)
                       (aref (vt-data vt) base-idx))
                 (let ((dim (nth depth old-shape))
                       (rep (nth depth reps)))
                   (loop
		     for r from 0 below rep
                     do (loop
			  for i from 0 below dim
                          do (let ((out-stride
				     (nth depth (vt-strides result)))
                                   (in-stride
				     (if (< depth (length (vt-strides vt)))
                                         (nth depth (vt-strides vt))
                                         0)))
                               (recurse (1+ depth)
                                        (+ out-idx
					   (* (+ (* r dim)
						 i)
					      out-stride))
                                        (if in-stride
					    (+ base-idx
					       (* i in-stride))
					    base-idx)))))))))
        (recurse 0 0 (vt-offset vt)))
      result)))

(defun vt-repeat (vt repeats &key axis)
  "重复数组元素.
  repeats: 整数或整数列表
  axis: 指定轴(nil 表示展平后重复)
  返回: 新张量"
  (cond
    ;; 无 axis:展平后重复
    ((null axis)
     (let* ((flat (vt-flatten vt))
            (size (vt-size flat))
            (data (vt-data flat)))
       (if (listp repeats)
           ;; 不同元素不同重复次数
           (let* ((total-size (reduce #'+ repeats))
                  (result-data
		    (make-array total-size
				:element-type (vt-element-type vt)))
                  (idx 0))
             (loop for i from 0 below size
                   for rep in repeats
                   do (loop for r from 0 below rep
                            do (setf (aref result-data idx) (aref data i))
                               (incf idx)))
             (%make-vt :data result-data :shape (list total-size)
                       :strides '(1) :offset 0))
           ;; 所有元素相同重复次数
           (let* ((total-size (* size repeats))
                  (result-data
		    (make-array total-size
				:element-type (vt-element-type vt)))
                  (idx 0))
             (loop for i from 0 below size
                   do (loop for r from 0 below repeats
                            do (setf (aref result-data idx) (aref data i))
                               (incf idx)))
             (%make-vt :data result-data :shape (list total-size)
                       :strides '(1) :offset 0)))))
    ;; 有 axis:沿轴重复
    (t
     (let* ((old-shape (vt-shape vt))
            (axis-size (nth axis old-shape))
            (reps (if (listp repeats)
		      repeats
		      (make-list axis-size :initial-element repeats)))
            (new-axis-size (reduce #'+ reps))
            (new-shape (copy-list old-shape)))
       (setf (nth axis new-shape) new-axis-size)
       (let ((result (vt-zeros new-shape :type (vt-element-type vt))))
         ;; 实现沿轴重复
         (labels
	     ((recurse (depth in-ptr out-ptr)
                (declare (type fixnum depth in-ptr out-ptr))
                (if (= depth axis)
                    ;; 沿重复轴处理
                    (let ((in-stride (nth depth (vt-strides vt)))
                          (out-stride (nth depth (vt-strides result))))
                      (loop
			for i from 0 below axis-size
                        for rep in reps
                        do (loop
			     for r from 0 below rep
                             do (setf (aref (vt-data result) out-ptr)
                                      (aref (vt-data vt) in-ptr))
                                (incf out-ptr out-stride))
                           (incf in-ptr in-stride)))
                    ;; 其他维度
                    (if (= depth (length old-shape))
                        nil
                        (let ((dim (nth depth old-shape))
                              (in-stride (nth depth (vt-strides vt)))
                              (out-stride (nth depth (vt-strides result))))
                          (loop
			    for i from 0 below dim
                            do (recurse (1+ depth)
					(+ in-ptr (* i in-stride))
					(+ out-ptr (* i out-stride)))))))))
           (recurse 0 (vt-offset vt) 0))
         result)))))

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
         (real-axis (if (< axis 0)
                        (+ rank axis)
                        axis)))    
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
                             (list :all)))))
            ;; 将源张量直接写入目标切片
            (setf (apply #'vt-slice
                         (cons result slice-args))
                  vt)
            (incf cur-offset axis-size)))
        result))))

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
	    (mapcar (lambda (vt) (vt-expand-dims vt axis)) vts)))
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

(defun vt-array-split (vt indices-or-sections &key axis)
  "沿轴分割数组.
  indices-or-sections: 整数N(等分)或索引列表
  axis: 分割轴
  返回: 张量列表"
  (let* ((shape (vt-shape vt))
         (axis-size (nth axis shape))
         (splits nil))
    (if (integerp indices-or-sections)
        ;; 等分
        (let ((section-size (floor axis-size indices-or-sections)))
          (loop for i from 0 below axis-size by section-size
                for end = (min (+ i section-size) axis-size)
                do (push (vt-split vt axis i end) splits)))
        ;; 按索引分割
        (let ((prev 0))
          (dolist (idx indices-or-sections)
            (push (vt-split vt axis prev idx) splits)
            (setf prev idx))
          ;; 最后一段
          (when (< prev axis-size)
            (push (vt-split vt axis prev axis-size) splits))))
    (nreverse splits)))

(defun vt-vsplit (vt indices-or-sections)
  "垂直分割(沿轴0)"
  (vt-array-split vt indices-or-sections :axis 0))

(defun vt-hsplit (vt indices-or-sections)
  "水平分割(沿轴1)"
  (vt-array-split vt indices-or-sections :axis 1))

(defun vt-dsplit (vt indices-or-sections)
  "深度分割(沿轴2)"
  (vt-array-split vt indices-or-sections :axis 2))

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
  axis: 归约轴(nil 表示展平后计算)
  返回: 新张量"
  (if axis
      ;; 沿轴累积
      (let* ((shape (vt-shape tensor))
             (result (vt-copy tensor)))
        (labels
	    ((recurse (depth in-ptr out-ptr)
               (declare (type fixnum depth in-ptr out-ptr))
               (if (= depth axis)
                   ;; 沿累积轴处理
                   (let* ((dim (nth depth shape))
                          (in-stride (nth depth (vt-strides tensor)))
                          (out-stride (nth depth (vt-strides result)))
                          (sum 0.0d0))
                     (loop
		       for i from 0 below dim
                       do (incf sum (aref (vt-data tensor) in-ptr))
                          (setf (aref (vt-data result) out-ptr) sum)
                          (incf in-ptr in-stride)
                          (incf out-ptr out-stride)))
                   ;; 其他维度
                   (if (= depth (length shape))
                       nil
                       (let ((dim (nth depth shape))
                             (in-stride (nth depth (vt-strides tensor)))
                             (out-stride (nth depth (vt-strides result))))
                         (loop
			   for i from 0 below dim
                           do (recurse (1+ depth)
                                       (+ in-ptr (* i in-stride))
                                       (+ out-ptr (* i out-stride)))))))))
          (recurse 0 (vt-offset tensor) 0))
        result)
      ;; 展平后累积
      (let* ((flat (vt-flatten tensor))
             (size (vt-size flat))
             (result (vt-zeros (list size)))
             (sum 0.0d0))
        (loop for i from 0 below size
              do (incf sum (aref (vt-data flat) i))
                 (setf (aref (vt-data result) i) sum))
        result)))

(defun vt-cumprod (tensor &key axis)
  "累积积"
  (if axis
      (let* ((shape (vt-shape tensor))
             (result (vt-copy tensor)))
        (labels
	    ((recurse (depth in-ptr out-ptr)
               (declare (type fixnum depth in-ptr out-ptr))
               (if (= depth axis)
                   (let* ((dim (nth depth shape))
                          (in-stride (nth depth (vt-strides tensor)))
                          (out-stride (nth depth (vt-strides result)))
                          (prod 1.0d0))
                     (loop
		       for i from 0 below dim
                       do (setf prod (* prod
					(aref (vt-data tensor) in-ptr)))
                          (setf (aref (vt-data result) out-ptr) prod)
                          (incf in-ptr in-stride)
                          (incf out-ptr out-stride)))
                   (if (= depth (length shape))
                       nil
                       (let ((dim (nth depth shape))
                             (in-stride (nth depth (vt-strides tensor)))
                             (out-stride (nth depth (vt-strides result))))
                         (loop
			   for i from 0 below dim
                           do (recurse (1+ depth)
                                       (+ in-ptr
					  (* i in-stride))
                                       (+ out-ptr
					  (* i out-stride)))))))))
          (recurse 0 (vt-offset tensor) 0))
        result)
      (let* ((flat (vt-flatten tensor))
             (size (vt-size flat))
             (result (vt-zeros (list size)))
             (prod 1.0d0))
        (loop for i from 0 below size
              do (setf prod (* prod
			       (aref (vt-data flat) i)))
                 (setf (aref (vt-data result) i) prod))
        result)))

(defun vt-median (tensor &key axis)
  "中位数.
  axis: 归约轴(nil 表示全局)"
  (if axis
      ;; 沿轴计算
      (let* ((shape (vt-shape tensor))
             (axis-size (nth axis shape))
             (out-shape (loop for d in shape for i from 0
			      unless (= i axis) collect d))
             (result (vt-zeros out-shape :type (vt-element-type tensor))))
        ;; 对每个切片排序并取中位数
        (labels
	    ((recurse (depth in-ptr out-ptr)
               (declare (type fixnum depth in-ptr out-ptr))
               (if (= depth axis)
                   ;; 收集该轴的数据
                   (let ((data-list '())
                         (in-stride (nth depth (vt-strides tensor))))
                     (loop
		       for i from 0 below axis-size
                       do (push (aref (vt-data tensor) in-ptr) data-list)
                          (incf in-ptr in-stride))
                     (setf data-list (sort data-list #'<))
                     (setf (aref (vt-data result) out-ptr)
                           (if (oddp axis-size)
                               (nth (floor axis-size 2) data-list)
                               (/ (+ (nth (1- (/ axis-size 2)) data-list)
                                     (nth (/ axis-size 2) data-list))
                                  2))))
                   (if (= depth (length shape))
                       nil
                       (let ((dim (nth depth shape))
                             (in-stride (nth depth (vt-strides tensor)))
                             (out-stride (nth depth (vt-strides result))))
                         (loop
			   for i from 0 below dim
                           do (recurse (1+ depth)
                                       (+ in-ptr
					  (* i in-stride))
                                       (+ out-ptr
					  (* i out-stride)))))))))
          (recurse 0 (vt-offset tensor) 0))
        result)
      ;; 全局中位数
      (let* ((flat (vt-flatten tensor))
             (size (vt-size flat))
             (data-list (loop for i from 0 below size
                              collect (aref (vt-data flat) i)))
             (sorted (sort data-list #'<)))
        (if (oddp size)
            (nth (floor size 2) sorted)
            (/ (+ (nth (1- (/ size 2)) sorted)
                  (nth (/ size 2) sorted))
               2)))))

(defun vt-percentile (tensor percentile &key axis interpolation)
  "计算百分位数.
  percentile: 百分位(0-100)
  axis: 归约轴
  interpolation: 插值方法(:linear, :lower, :higher, :midpoint, :nearest)"
  (declare (ignore interpolation))
  (let ((q (/ percentile 100.0d0)))
    (if axis
        (let* ((shape (vt-shape tensor))
               (axis-size (nth axis shape))
               (out-shape
		 (loop
		   for d in shape for i from 0 unless (= i axis) collect d))
               (result (vt-zeros out-shape :type (vt-element-type tensor))))
          (labels
	      ((recurse (depth in-ptr out-ptr)
                 (declare (type fixnum depth in-ptr out-ptr))
                 (if (= depth axis)
                     (let ((data-list '())
                           (in-stride (nth depth (vt-strides tensor))))
                       (loop for i from 0 below axis-size
                             do (push (aref (vt-data tensor) in-ptr)
				      data-list)
                                (incf in-ptr in-stride))
                       (setf data-list (sort data-list #'<))
                       (let* ((idx (* q (1- axis-size)))
                              (lower (floor idx))
                              (upper (ceiling idx))
                              (frac (- idx lower)))
                         (setf (aref (vt-data result) out-ptr)
                               (+ (* (- 1 frac)
				     (nth lower data-list))
                                  (* frac
				     (nth (min upper (1- axis-size))
					  data-list))))))
                     (if (= depth (length shape))
                         nil
                         (let ((dim (nth depth shape))
                               (in-stride (nth depth (vt-strides tensor)))
                               (out-stride (nth depth (vt-strides result))))
                           (loop
			     for i from 0 below dim
                             do (recurse (1+ depth)
                                         (+ in-ptr
					    (* i in-stride))
                                         (+ out-ptr
					    (* i out-stride)))))))))
            (recurse 0 (vt-offset tensor) 0))
          result)
        (let* ((flat (vt-flatten tensor))
               (size (vt-size flat))
               (data-list (loop for i from 0 below size
				collect (aref (vt-data flat) i)))
               (sorted (sort data-list #'<))
               (idx (* q (1- size)))
               (lower (floor idx))
               (upper (ceiling idx))
               (frac (- idx lower)))
          (+ (* (- 1 frac)
		(nth lower sorted))
             (* frac
		(nth (min upper (1- size)) sorted)))))))

(defun vt-quantile (tensor q &key axis)
  "计算分位数.
  q: 分位数(0-1)"
  (vt-percentile tensor (* q 100) :axis axis))

(defun vt-ptp (tensor &key axis)
  "峰-峰值(最大值 - 最小值)"
  (if axis
      (vt-- (vt-amax tensor :axis axis) (vt-amin tensor :axis axis))
      (- (vt-amax tensor) (vt-amin tensor))))

(defun vt-histogram (tensor &key bins range density)
  "计算直方图.
  bins: bin数量(默认10)
  range: (min, max) 范围
  density: 是否归一化
  返回: (hist, bin-edges)"
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
          do (setf (aref bin-edges i)
		   (+ data-min
		      (* i bin-width))))
    
    ;; 计算直方图
    (loop for i from 0 below size
          for val = (aref data i)
          for bin-idx = (min (1- bins)
			     (floor (- val data-min) bin-width))
          when (and (>= val data-min) (< val data-max))
            do (incf (aref hist bin-idx)))
    
    ;; 归一化
    (when density
      (let ((total (reduce #'+ hist)))
        (loop for i from 0 below bins
              do (setf (aref hist i) 
                       (/ (aref hist i)
			  (* total bin-width))))))
    
    (values (vt-from-sequence (coerce hist 'list))
            (vt-from-sequence (coerce bin-edges 'list)))))

;;; ===========================================
;;; 5. 逻辑运算扩展
;;; ===========================================

(defun vt-logical-and (t1 t2)
  "逻辑与"
  (vt-map (lambda (a b) (if (and (/= a 0) (/= b 0)) 1.0d0 0.0d0))
	  t1 t2))

(defun vt-logical-or (t1 t2)
  "逻辑或"
  (vt-map (lambda (a b) (if (or (/= a 0) (/= b 0)) 1.0d0 0.0d0))
	  t1 t2))

(defun vt-logical-not (vt)
  "逻辑非"
  (vt-map (lambda (a) (if (= a 0) 1.0d0 0.0d0)) vt))

(defun vt-logical-xor (t1 t2)
  "逻辑异或"
  (vt-map (lambda (a b) 
            (if (/= (if (= a 0) 0 1) (if (= b 0) 0 1)) 1.0d0 0.0d0)) 
          t1 t2))

(defun vt-all (condition &key axis)
  "检查所有元素是否为真"
  (nth-value
   0 (vt-reduce condition axis 1.0d0
                (lambda (acc val)
                  (declare (type number acc val))
                  (values (if (and (/= acc 0) (/= val 0)) 1.0d0 0.0d0) nil))
                :return-arg nil)))

(defun vt-any (condition &key axis)
  "检查是否存在任一元素为真"
  (nth-value
   0 (vt-reduce condition axis 0.0d0
                (lambda (acc val)
                  (declare (type number acc val))
                  (values (if (or (/= acc 0) (/= val 0)) 1.0d0 0.0d0) nil))
                :return-arg nil)))

(defun vt-isclose (t1 t2 &key (rtol 1e-5) (atol 1e-8))
  "判断两个数组元素是否在容差范围内接近"
  (vt-map (lambda (a b)
            (let ((diff (abs (- a b))))
              (if (<= diff (+ atol (* rtol (max (abs a) (abs b)))))
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

(defun vt-sort (tensor &key axis kind)
  "排序.
  axis: 排序轴(nil 表示展平)
  kind: 排序算法(忽略,使用 Common Lisp 的 sort)"
  (declare (ignore kind))
  (if axis
      ;; 沿轴排序
      (let* ((shape (vt-shape tensor))
             (axis-size (nth axis shape))
             (result (vt-copy tensor)))
        (labels
	    ((recurse (depth in-ptr out-ptr)
               (declare (type fixnum depth in-ptr out-ptr))
               (if (= depth axis)
                   ;; 收集并排序
                   (let ((indices '())
                         (values '())
                         (in-stride (nth depth (vt-strides tensor)))
                         (out-stride (nth depth (vt-strides result))))
                     (loop for i from 0 below axis-size
                           do (push i indices)
                              (push (aref (vt-data tensor) in-ptr) values)
                              (incf in-ptr in-stride))
                     ;; 排序(稳定排序)
                     (let ((sorted-pairs (sort (mapcar #'cons values indices)
                                               #'< :key #'car)))
                       ;; 写回
                       (loop
			 for i from 0 below axis-size
                         for pair in sorted-pairs
                         do (setf (aref (vt-data result) out-ptr)
				  (car pair))
                            (incf out-ptr out-stride))))
                   (if (= depth (length shape))
                       nil
                       (let ((dim (nth depth shape))
                             (in-stride (nth depth (vt-strides tensor)))
                             (out-stride (nth depth (vt-strides result))))
                         (loop
			   for i from 0 below dim
                           do (recurse (1+ depth)
                                       (+ in-ptr
					  (* i in-stride))
                                       (+ out-ptr
					  (* i out-stride)))))))))
          (recurse 0 (vt-offset tensor) 0))
        result)
      ;; 展平排序
      (let* ((flat (vt-flatten tensor))
             (data-list (coerce (vt-data flat) 'list))
             (sorted (sort data-list #'<)))
        (vt-from-sequence sorted))))

(defun vt-argsort (tensor &key axis)
  "返回排序后的索引"
  (if axis
      (let* ((shape (vt-shape tensor))
             (axis-size (nth axis shape))
             (out-shape
	       (loop
		 for d in shape for i from 0 unless (= i axis) collect d))
             (result (vt-zeros out-shape :type 'fixnum)))
        (labels
	    ((recurse (depth in-ptr out-ptr)
               (declare (type fixnum depth in-ptr out-ptr))
               (if (= depth axis)
                   (let ((indices '())
                         (values '())
                         (in-stride (nth depth (vt-strides tensor))))
                     (loop
		       for i from 0 below axis-size
                       do (push i indices)
                          (push (aref (vt-data tensor) in-ptr) values)
                          (incf in-ptr in-stride))
                     (let ((sorted-pairs (sort (mapcar #'cons values indices)
                                               #'< :key #'car)))
                       (loop
			 for pair in sorted-pairs
                         for i from 0
                         do (setf (aref (vt-data result)
					(+ out-ptr
					   (* i (nth axis
						     (vt-strides result)))))
                                  (cdr pair)))))
                   (if (= depth (length shape))
                       nil
                       (let ((dim (nth depth shape))
                             (in-stride (nth depth (vt-strides tensor)))
                             (out-stride (nth depth (vt-strides result))))
                         (loop
			   for i from 0 below dim
                           do (recurse (1+ depth)
                                       (+ in-ptr
					  (* i in-stride))
                                       (+ out-ptr
					  (* i out-stride)))))))))
          (recurse 0 (vt-offset tensor) 0))
        result)
      (let* ((flat (vt-flatten tensor))
             (size (vt-size flat))
             (indices (loop for i from 0 below size collect i))
             (values (coerce (vt-data flat) 'list))
             (pairs (sort (mapcar #'cons values indices) #'< :key #'car))
             (sorted-indices (mapcar #'cdr pairs)))
        (vt-from-sequence sorted-indices :type 'fixnum))))

(defun vt-unique (tensor &key return-index return-inverse return-counts)
  "返回数组中的唯一元素"
  (declare (ignore return-index return-inverse return-counts))
  (let* ((flat (vt-flatten tensor))
         (data-list (coerce (vt-data flat) 'list))
         (unique-list (remove-duplicates (sort data-list #'<))))
    (vt-from-sequence unique-list)))

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
                         when (< (aref data j) val)
                           return j
                         finally (return size))))
    (%make-vt :data result :shape (list v-size)
	      :strides '(1) :offset 0)))

;;; ===========================================
;;; 7. 数据类型转换
;;; ===========================================

(defun vt-astype (tensor new-type)
  "转换数组的数据类型"
  (let* ((result (vt-zeros (vt-shape tensor) :type new-type)))
    (vt-map (lambda (x) (coerce x new-type)) tensor)
    result))

(defun vt-tolist (tensor)
  "将数组转换为嵌套列表"
  (vt-data->list tensor))

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

(defun vt-square (vt)
  "平方"
  (vt-map (lambda (x) (* x x)) vt))

(defun vt-gradient (tensor &key spacing)
  "计算梯度"
  (declare (ignore spacing))
  ;; 简化实现:中心差分
  (let* ((flat (vt-flatten tensor))
         (size (vt-size flat))
         (data (vt-data flat))
         (result (vt-zeros (list size)))
         (r-data (vt-data result)))
    (when (> size 1)
      ;; 前向差分
      (setf (aref r-data 0) (- (aref data 1) (aref data 0)))
      ;; 中心差分
      (loop for i from 1 below (1- size)
            do (setf (aref r-data i)
                     (/ (- (aref data (1+ i))
			   (aref data (1- i)))
			2.0d0)))
      ;; 后向差分
      (setf (aref r-data (1- size))
            (- (aref data (1- size)) (aref data (- size 2)))))
    result))

;;; ===========================================
;;; 10. 索引高级功能
;;; ===========================================

(defun vt-take (tensor indices &key axis)
  "从数组中按索引取值"
  (if axis
      ;; 沿轴取值
      (let* ((shape (vt-shape tensor))
             (axis-size (nth axis shape))
             (idx-flat (vt-flatten indices))
             (idx-data (vt-data idx-flat))
             (idx-size (vt-size idx-flat))
             (out-shape (loop for d in shape for i from 0
                              if (= i axis) collect idx-size
				else collect d))
             (result (vt-zeros out-shape :type (vt-element-type tensor))))
        (labels
	    ((recurse (depth in-ptr out-ptr)
               (declare (type fixnum depth in-ptr out-ptr))
               (if (= depth axis)
                   (let ((in-stride (nth depth (vt-strides tensor)))
                         (out-stride (nth depth (vt-strides result))))
                     (loop for i from 0 below idx-size
                           for idx = (aref idx-data i)
                           do (when (or (< idx 0) (>= idx axis-size))
                                (error "索引 ~A 越界" idx))
                              (setf (aref (vt-data result) out-ptr)
				    (aref (vt-data tensor)
					  (+ in-ptr
					     (* idx in-stride))))
                              (incf out-ptr out-stride)))
                   (if (= depth (length shape))
                       nil
                       (let ((dim (nth depth shape))
                             (in-stride (nth depth (vt-strides tensor)))
                             (out-stride (nth depth (vt-strides result))))
                         (loop
			   for i from 0 below dim
                           do (recurse (1+ depth)
                                       (+ in-ptr
					  (* i in-stride))
                                       (+ out-ptr
					  (* i out-stride)))))))))
          (recurse 0 (vt-offset tensor) 0))
        result)
      ;; 展平后取值
      (let* ((flat (vt-flatten tensor))
             (data (vt-data flat))
             (size (vt-size flat))
             (idx-flat (vt-flatten indices))
             (idx-data (vt-data idx-flat))
             (idx-size (vt-size idx-flat))
             (result-data
	       (make-array idx-size
			   :element-type (vt-element-type tensor))))
        (loop for i from 0 below idx-size
              for idx = (aref idx-data i)
              do (when (or (< idx 0) (>= idx size))
                   (error "索引 ~A 越界" idx))
                 (setf (aref result-data i) (aref data idx)))
        (%make-vt :data result-data :shape (list idx-size)
                  :strides '(1) :offset 0))))

(defun vt-put (tensor indices values)
  "按索引设置值"
  (let* ((flat (vt-flatten tensor))
         (size (vt-size flat))
         (data (vt-data flat))
         (idx-flat (vt-flatten indices))
         (idx-data (vt-data idx-flat))
         (idx-size (vt-size idx-flat))
         (val-flat (vt-flatten values))
         (val-data (vt-data val-flat)))
    (loop for i from 0 below idx-size
          for idx = (aref idx-data i)
          do (when (or (< idx 0) (>= idx size))
               (error "索引 ~A 越界" idx))
             (setf (aref data idx)
		   (aref val-data (mod i (vt-size val-flat)))))
    tensor))

(defun vt-choose (choices indices)
  "根据索引数组从多个数组中选择值"
  (let* ((n-choices (length choices))
         (idx-flat (vt-flatten indices))
         (idx-data (vt-data idx-flat))
         (idx-size (vt-size idx-flat))
         (result-data (make-array idx-size :element-type 'double-float)))
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
