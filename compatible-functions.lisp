;;;; 扩展功能:对标 numpy 函数
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

(defun vt-linspace (start end num &key (endpoint t) (type 'double-float))
  "创建线性间隔数组，与 numpy linspace 完全兼容。
   start     : 起始值
   end       : 结束值
   num       : 元素个数（>0 整数）
   endpoint  : 是否包含终止值（默认 t）
   type      : 元素类型（默认 double-float）
   返回      : 一维张量"
  (declare (type number start end)
           (type fixnum num)
           (type boolean endpoint))
  (when (<= num 0)
    (error "num 必须大于 0，当前值为 ~d" num))
  (with-float-safe
    ;; 单元素特殊情况
    (when (= num 1)
      (return-from vt-linspace
	(make-vt (list 1) (coerce start type) :type type)))
    ;; 计算步长与元素个数
    (let* ((div (if endpoint (1- num) num))
           (step (/ (- end start) div))
           (elements (loop for i fixnum from 0 below num
                           collect (+ start (* i step)))))
      (if (eq type 'fixnum)
	  (setf elements (mapcar #'truncate elements)))
      (vt-from-sequence elements :type type))))

(defun vt-full (shape fill-value &key (type 'double-float))
  "创建指定值填充的数组.
  shape: 形状列表
  fill-value: 填充值
  返回: 张量"
  (with-float-safe
    (make-vt shape fill-value :type type)))

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
  (with-float-safe
    (vt-full (vt-shape vt)
	     fill-value
	     :type (or type (vt-element-type vt)))))

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
  "返回对数间隔的 1d 张量。"
  (with-float-safe
    (vt-map (lambda (x) (expt base x))
            (vt-linspace start stop num :endpoint endpoint
					:type type))))

(defun vt-kron (a b)
  "计算任意维张量的 kronecker 积，行为与 numpy 完全一致。"
  (with-float-safe
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
	;; 构建交错形状：a: (d1, 1, d2, 1, ...)，b: (1, e1, 1, e2, ...)
	(loop for da in a-shape-padded
              for db in b-shape-padded
              do (push da a-new-shape)   ; a: da, 1
		 (push 1 a-new-shape)
		 (push 1 b-new-shape)   ; b: 1, db
		 (push db b-new-shape)
		 (push (* da db) final-shape))
	(setf a-new-shape (nreverse a-new-shape))
	(setf b-new-shape (nreverse b-new-shape))
	(setf final-shape (nreverse final-shape))
	;; 重塑并广播相乘，最后 reshape 到目标形状
	(let ((a-reshaped (vt-reshape a a-new-shape))
              (b-reshaped (vt-reshape b b-new-shape)))
          (vt-reshape (vt-* a-reshaped b-reshaped) final-shape))))))

(defun vt-from-function (shape fn &key (type 'double-float))
  "根据函数创建数组.
  shape: 形状列表
  fn: 接受索引列表并返回值的函数
  返回: 张量"
  (with-float-safe
    (let* ((size (vt-shape-to-size shape))
           (data (make-array size :element-type type))
           (result (%make-vt :data data
			     :shape shape 
                             :strides (vt-compute-strides shape)
			     :offset 0
			     :etype type))
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
      result)))

(defun vt-meshgrid (vts-list &key (indexing :xy) (sparse nil) (copy t))
  "生成坐标矩阵，完全兼容 numpy 的 meshgrid 语义。
   vts-list : 一维张量列表。
   indexing : :ij（矩阵索引）或 :xy（笛卡尔坐标，默认 :xy）。
   sparse   : 若为 t，返回稀疏网格（形状中非对应轴为 1 的广播视图），否则返回全尺寸网格。
   copy     : 若为 t，强制复制数据；否则尽量返回视图。
   返回     : 张量列表，长度与 vts 相同。"
  (declare (list vts-list))
  (with-float-safe
    (dolist (v vts-list)
      (assert (= (length (vt-shape v)) 1)
              (v) "all inputs to meshgrid must be 1-d"))
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
			    (vt-broadcast-to sp-view output-shape))))))))

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
                :offset (vt-offset vt)
		:etype (vt-element-type vt))
      ;; 非连续:创建副本
      (vt-flatten vt)))

(defun vt-squeeze (vt &key axis)
  "移除长度为1的维度。支持负轴。"
  (with-float-safe
    (let* ((old-shape (vt-shape vt))
           (old-strides (vt-strides vt))
           (old-offset (vt-offset vt)))
      (if axis
          (let* ((ax (vt-normalize-axis axis (length old-shape)))  ; 规范化
		 (axis-size (nth ax old-shape)))
            (unless (= axis-size 1)
              (error "无法挤压非单例维度: axis ~a 大小为 ~a" axis axis-size))
            (%make-vt :data (vt-data vt)
                      :shape (append (subseq old-shape 0 ax)
                                     (subseq old-shape (1+ ax)))
                      :strides (append (subseq old-strides 0 ax)
                                       (subseq old-strides (1+ ax)))
                      :offset old-offset
		      :etype (vt-element-type vt)))
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
                      :offset old-offset
		      :etype (vt-element-type vt)))))))

(defun vt-expand-dims (vt axis)
  "在指定位置插入新轴.
  axis: 插入位置(0表示最前面)
  返回: 张量视图"
  (with-float-safe
    (let* ((old-shape (vt-shape vt))
           (old-strides (vt-strides vt))
           (old-offset (vt-offset vt))
           (rank (length old-shape))
           (actual-axis (if (< axis 0) (+ rank axis 1) axis)))
      (when (or (< actual-axis 0) (> actual-axis rank))
	(error "轴 ~a 越界(秩 ~a)" axis rank))
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
                  :offset old-offset
		  :etype (vt-element-type vt))))))

(defun vt-unsqueeze (vt axis)
  "在指定位置插入新轴"
  (vt-expand-dims vt axis))

(defun vt-swapaxes (vt axis1 axis2)
  "交换数组的两个轴。axis1 和 axis2 支持负数索引。
   返回：张量视图。"
  (with-float-safe
    (let* ((rank (length (vt-shape vt)))
           (ax1 (vt-normalize-axis axis1 rank))
           (ax2 (vt-normalize-axis axis2 rank)))
      (vt-transpose
       vt
       (loop for i from 0 below rank
             collect (cond ((= i ax1) ax2)
                           ((= i ax2) ax1)
                           (t i)))))))

(defun vt-rot90 (tensor &key (k 1) (axes '(0 1)))
  "将张量在 axes 指定的平面内旋转 90 度 k 次。
    axes : 长度为 2 的列表，指定旋转平面（默认 (0 1)）。
    k    : 旋转次数（正整数为逆时针，负数为顺时针），自动模 4。
   
   返回零拷贝视图（当底层操作均为视图时）。
   行为与 NumPy 的 rot90 完全一致。"
  (let* ((shape (vt-shape tensor))
         (rank (length shape)))
    (when (< rank 2)
      (error "vt-rot90: tensor rank must be >= 2, but got rank ~D" rank))
    (let ((ax0 (first axes))
          (ax1 (second axes)))
      ;; 支持负数轴
      (setf ax0 (vt-normalize-axis ax0 rank))
      (setf ax1 (vt-normalize-axis ax1 rank))
      (when (= ax0 ax1)
        (error "vt-rot90: axes must be different, but got ~D and ~D" ax0 ax1))
      ;; k 规范化到 0~3
      (let ((k (mod k 4)))
        (when (minusp k)    ; 保证非负
          (incf k 4))
        ;; 重复 k 次：先转置，再翻转第一个轴（与 NumPy 一致）
        (loop with result = tensor
              repeat k
              do (setf result (vt-swapaxes result ax0 ax1))
                 (setf result (vt-flip result :axis ax0))
              finally (return result))))))

(defun vt-rotate (tensor angle &key (center nil) (order 1) (cval 0.0))
   "将二维张量按指定角度旋转（对标 scipy.ndimage.rotate）。
   
   **旋转中心**：
   - 默认 center = NIL，旋转中心位于张量的几何中心，即 (rows/2, cols/2)。
   - 若提供 center 列表 (cy, cx)，则绕该点旋转。例如 '(0 0) 表示绕左上角原点旋转。
   - 注意：行坐标对应 cy，列坐标对应 cx，与数组索引顺序一致（行在前，列在后）。

   **旋转方向**：
   - angle 为正时，逆时针旋转；为负时，顺时针旋转（与 scipy 一致）。
   - 内部通过逆向映射实现：从目标张量的每个像素出发，通过旋转矩阵反推其在原始张量中的对应坐标，然后采样。这保证了旋转后的图像没有空洞，并且不会产生“值变负”的现象（像素值仅为重新定位，不改变自身的数值）。

   **插值方式**：
   - order = 0 : 最近邻插值（直接取整，速度快，边缘有锯齿）。
   - order = 1 : 双线性插值（平滑，但计算量稍大）。
   - 注意：目前仅支持 order 0 和 1，其他值未实现。

   **边界处理**：
   - 当目标像素映射到原始张量边界之外时，使用 cval 填充。cval 默认为 0.0。
   - 这意味着旋转后原本在图像角落的像素若被转到图像外，就会“消失”并被填充值替代。例如绕原点旋转 180° 时，除原点外的所有像素都会移到负坐标而被 cval 覆盖，仅原点处的像素保留。

   **返回值**：
   - 返回一个与输入 tensor 形状完全相同的新张量（非视图）。数据类型固定被强制执行并返回双精度浮点数。

   **注意**：
   - 该函数为通用任意角度旋转，与仅支持 90° 倍数的 `vt-rot90` 不同，它涉及插值且可能产生边界丢失。
   - 若要实现类似 `scipy.ndimage.rotate` 的 `reshape=True`（自动扩大画布以包含全部内容），当前函数未提供该选项，需外部手动处理。

   示例：
   ;; 绕图像中心旋转 45°
   (let ((m (clvt:vt-from-sequence '((1 0 0) (0 1 0) (0 0 1)))))
     (vt-rotate m (/ pi 4) :order 0))

   ;; 绕左上角原点旋转 30°
   (let ((m (clvt:vt-from-sequence '((1 0) (0 1)))))
     (vt-rotate m (/ pi 6) :order 0))
"
  (let* ((shape (vt-shape tensor))
         (rows (first shape))
         (cols (second shape))
         (cos-a (cos angle))
         (sin-a (sin angle))
         (cy (if center (first center) (/ rows 2.0)))
         (cx (if center (second center) (/ cols 2.0))))
    (vt-from-function
     shape
     (lambda (coords)
       (destructuring-bind (ny nx) coords   ; 行, 列
         (let* ((dx (- nx cx))
                (dy (- ny cy))
                ;; 逆旋转变换：源坐标相对中心的偏移
                (src-rel-x (+ (* dx cos-a) (* dy sin-a)))
                (src-rel-y (- (* dy cos-a) (* dx sin-a)))
                ;; 回到绝对坐标
                (src-x (+ src-rel-x cx))
                (src-y (+ src-rel-y cy)))
           (if (= order 0)
               ;; 最近邻
               (let ((ix (round src-x))
                     (iy (round src-y)))
                 (if (and (>= ix 0) (< ix cols)
                          (>= iy 0) (< iy rows))
                     (vt-ref tensor iy ix)   ; 注意：行在前
                     cval))
               ;; 双线性插值（同样使用修正后的 src-x, src-y）
               (let* ((x0 (floor src-x))
                      (y0 (floor src-y))
                      (x1 (1+ x0))
                      (y1 (1+ y0))
                      (fx (- src-x x0))
                      (fy (- src-y y0))
                      (fx1 (- 1.0 fx))
                      (fy1 (- 1.0 fy)))
                 (if (or (< x0 0) (>= x1 cols)
                         (< y0 0) (>= y1 rows))
                     cval
                     (let ((pxy (vt-ref tensor y0 x0))
                           (px1y (if (< x1 cols)
				     (vt-ref tensor y0 x1)
				     0.0))
                           (pxy1 (if (< y1 rows)
				     (vt-ref tensor y1 x0)
				     0.0))
                           (px1y1 (if (and (< x1 cols) (< y1 rows))
                                      (vt-ref tensor y1 x1)
                                      0.0)))
                       (+ (* fx1 fy1 pxy)
                          (* fx fy1 px1y)
                          (* fx1 fy pxy1)
                          (* fx fy px1y1)))))))))
     :type 'double-float)))

(defun vt-rotate-origin (tensor angle &key (order 0))
  "绕左上角原点 (0,0) 旋转张量，是 vt-rotate 的便捷版本。
   等价于 (vt-rotate tensor angle :center '(0 0) :order order)。
   该函数将张量的 (0,0) 点视为旋转中心，逆时针旋转 angle 弧度。
   像素值仅位置发生变化，不改变数值。
   超出原张量边界的部分用 0 填充。"
  (vt-rotate tensor angle :center '(0 0) :order order))

(defun vt-broadcast-to (vt new-shape)
  "将张量广播到新形状，返回零拷贝视图"
  (with-float-safe
    (let* ((old-shape (vt-shape vt))
           (broadcast-shape
	     (vt-broadcast-shapes old-shape new-shape)))
      (unless (equal broadcast-shape new-shape)
	(error "形状 ~a 不能广播到 ~a" old-shape new-shape))
      (%make-vt :data (vt-data vt)
		:shape new-shape
		:strides (vt-broadcast-strides
			  old-shape new-shape (vt-strides vt))
		:offset (vt-offset vt)
		:etype (vt-element-type vt)))))

(defun vt-repeat (vt repeats &key axis)
  "重复数组元素。重写为切片拼接方式。
   修正：正确处理rep=0、apply调用、空结果情形。"
  (with-float-safe
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
		(vt-zeros zero-shape :type (vt-element-type vt))))))))

(defun vt-tile (vt reps)
  "重复数组构造新数组（与 numpy tile 一致）
   vt: 输入张量
   reps: 每个维度的重复次数列表
   返回: 新张量"
  (with-float-safe
    (let* ((sh (vt-shape vt))
           (reps-list (if (listp reps) reps (list reps)))
           (ndim (max (length sh) (length reps-list)))
           ;; 在前面补 1 使秩等于 ndim
           (padded-sh (append (make-list (- ndim (length sh))
					 :initial-element 1) sh))
           (padded-reps (append (make-list (- ndim (length reps-list))
					   :initial-element 1)
				reps-list))
           ;; 先将 vt 重塑为 ndim 维
           (result (vt-reshape vt padded-sh)))
      (loop for axis from 0 below ndim
            for rep = (nth axis padded-reps)
            when (> rep 1)
              do (let ((parts (loop repeat rep collect result)))
                   (setf result (apply #'vt-concatenate axis parts))))
      ;; 最终形状
      (vt-reshape result (mapcar #'* padded-sh padded-reps)))))

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
  (with-float-safe
    (let* ((shapes (mapcar #'vt-shape vts))
           (rank (length (car shapes)))
           ;; 处理负轴
           (real-axis (vt-normalize-axis axis rank)))
      ;; 验证形状
      (loop for shape in shapes
            for i from 0
            do (unless (= (length shape) rank)
		 (error "张量 ~a 的秩不匹配" i))
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
          result)))))

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
  (with-float-safe
    (let* ((shapes (mapcar #'vt-shape vts))
           (base-shape (car shapes)))
      ;; 验证形状一致
      (loop for shape in shapes
            for i from 0
            unless (equal shape base-shape)
              do (error "张量 ~a 形状不匹配" i))
      
      ;; 沿 axis 扩展每个数组维度
      (let ((expanded-vts
	      (mapcar (lambda (vt) (vt-expand-dims vt axis))
		      vts)))
	(apply #'vt-concatenate axis expanded-vts)))))

(defun vt-vstack (&rest vts)
  "垂直堆叠(沿轴0).
  等价于 concatenate axis=0"
  (apply #'vt-concatenate 0 vts))

(defun vt-hstack (&rest vts)
  "水平堆叠(沿轴1,对于1d数组沿轴0).
  对于1d数组,等价于 concatenate"
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
;;; 数组追加、插入、删除
;;; ===========================================

(defun vt-append (arr values &key (axis nil))
  "将 values 附加到 arr 末尾。
   arr    : 输入张量
   values : 要追加的值（张量、标量或序列）
   axis   : 连接轴（none 表示展平后追加；整数表示沿该轴连接）
   返回   : 新张量"
  (with-float-safe
    (let ((arr-vt (ensure-vt arr))
          (val-vt (ensure-vt values)))
      (if (null axis)
          ;; 展平模式
          (let ((flat-arr (vt-flatten arr-vt))
		(flat-val (vt-flatten val-vt)))
            (vt-concatenate 0 flat-arr flat-val))
          ;; 沿轴连接
          (let* ((rank (length (vt-shape arr-vt)))
		 (ax (vt-normalize-axis axis rank))
		 (arr-shape (vt-shape arr-vt))
		 (val-shape (vt-shape val-vt)))
            ;; 形状校验：除连接轴外，其余维度必须相等
            (when (/= (length arr-shape) (length val-shape))
              (error "vt-append: dimensions mismatch: arr has rank ~a, values has rank ~a"
                     (length arr-shape) (length val-shape)))
            (loop for i from 0 below (length arr-shape)
                  unless (or (= i ax)
                             (= (nth i arr-shape) (nth i val-shape)))
                    do (error "vt-append: shape mismatch at axis ~a: arr ~a vs values ~a"
                              i (nth i arr-shape) (nth i val-shape)))
            (vt-concatenate ax arr-vt val-vt))))))

(defun vt-insert (arr obj values &key (axis nil))
  "完全兼容 numpy 的 np.insert。
   obj可以是：
     - 整数：插入单个位置。
     - 整数列表：插入多个位置（值块按原始顺序与位置一一对应）。
   values : 要插入的标量、序列或张量。
   axis   : nil 表示展平后插入；
            整数表示沿指定轴插入。"
  (with-float-safe
    (let* ((arr-vt (ensure-vt arr))
          (values-vt (ensure-vt values :type (vt-element-type arr-vt))))
      (if (null axis)
          ;; ---------- 展平模式 ----------
          (let* ((flat-arr (vt-flatten arr-vt))
                 (flat-val (vt-flatten values-vt))
                 (val-size (vt-size flat-val)))
            ;; 情况1：整数索引 → 整个 values 插入到该位置
            (when (integerp obj)
              (let ((current-size (vt-size flat-arr)))
                (when (or (< obj 0) (> obj current-size))
                  (error "索引 ~A 越界" obj))
                (if (zerop val-size)
                    flat-arr
                    (let ((left (if (> obj 0)
                                    (vt-slice flat-arr (list 0 obj))
                                    (vt-zeros (list 0) :type (vt-element-type flat-arr))))
                          (right (if (< obj current-size)
                                     (vt-slice flat-arr (list obj current-size))
                                     (vt-zeros (list 0) :type (vt-element-type flat-arr)))))
                      (return-from vt-insert
                        (vt-concatenate 0 left flat-val right))))))
            ;; 情况2：列表索引
            (let ((num-insert (length obj)))
              ;; 检查 values 长度：必须为1（广播）或等于索引数
              (unless (or (= val-size num-insert) (= val-size 1))
                (error "FLAT mode: values size ~A ≠ number of indices ~A" val-size num-insert))
              ;; 构建 (索引 . 值块) 对，并降序排序，从后往前插入避免偏移
              (let ((sorted-pairs
                      (stable-sort
                       (if (= val-size 1)
                           ;; 单值广播：每个位置都插入同一个 flat-val
                           (loop for idx in obj collect (cons idx flat-val))
                           ;; 一一对应：拆分 flat-val 为单个元素
                           (loop for idx in obj
                                 for i from 0 below val-size
                                 collect (cons idx (vt-narrow flat-val 0 i (1+ i)))))
                       #'> :key #'car))
                    (result flat-arr))
                (dolist (pair sorted-pairs)
                  (let* ((pos (car pair))
                         (val (cdr pair))
                         (current-size (vt-size result)))
                    (when (or (< pos 0) (> pos current-size))
                      (error "索引 ~A 越界" pos))
                    (let ((left (if (> pos 0)
                                    (vt-slice result (list 0 pos))
                                    (vt-zeros (list 0) :type (vt-element-type result))))
                          (right (if (< pos current-size)
                                     (vt-slice result (list pos current-size))
                                     (vt-zeros (list 0) :type (vt-element-type result)))))
                      (setf result (vt-concatenate 0 left val right)))))
                result)))
          ;; ---------- 沿轴插入 ----------
          (let* ((arr-shape (vt-shape arr-vt))
                 (rank (length arr-shape))
                 (ax (vt-normalize-axis axis rank))
                 (arr-axis-size (nth ax arr-shape))
                 (obj-list (if (listp obj) obj (list obj)))
                 (num-insert (length obj-list))
                 (target-shape (loop for i below rank
                                     collect (if (= i ax)
						 num-insert
						 (nth i arr-shape))))
		 ;; 标量广播：若 values 是标量，复制填充至目标形状
                 (values-vt-broadcast
                   (if (null (vt-shape values-vt))
                       (vt-full target-shape (vt-ref values-vt)
                                :type (vt-element-type values-vt))
                       values-vt))
                 (values-reshaped (vt-reshape values-vt-broadcast target-shape))
                 (arr-slices
                   (loop for i from 0 below arr-axis-size
                         collect (vt-narrow arr-vt ax i (1+ i))))
                 (value-blocks
                   (if (= num-insert 1)
                       (list values-reshaped)
                       (loop for i from 0 below num-insert
                             collect (vt-narrow values-reshaped ax i (1+ i)))))
                 ;; 配对并按位置降序排序
                 (pairs (loop for pos in obj-list
                              for block in value-blocks
                              collect (cons pos block)))
                 (sorted-pairs (stable-sort pairs #'> :key #'car))
                 ;; 构造最终切片列表
                 (result-slices
                   (let ((slices (copy-list arr-slices)))
                     (dolist (pair sorted-pairs slices)
                       (let ((pos (car pair))
                             (block (cdr pair)))
                         (setf slices
                               (append (subseq slices 0 pos)
                                       (list block)
                                       (subseq slices pos))))))))
            (apply #'vt-concatenate ax result-slices))))))

(defun vt-delete (arr obj &key (axis nil))
  "完全兼容 numpy 的 np.delete。
   obj 可以是：
     - 整数：删除单个索引。
     - 整数列表：删除多个独立索引（例如 `(0 2)` 删除索引 0 和 2）。
     - 形如 `(:slice start end)` 的列表：删除从 start 到 end-1 的连续切片。
   axis 为 nil 时展平后删除，否则沿指定轴删除。"
  (with-float-safe
    (let* ((tensor (ensure-vt arr))
           (sh (vt-shape tensor))
           (rank (length sh))
           (etype (vt-element-type tensor)))
      (labels ((normalize-index (idx dim)
                 "将负索引转为非负，并检查是否在 [0, dim-1] 内。"
                 (let ((nidx (if (minusp idx) (+ idx dim) idx)))
                   (unless (<= 0 nidx (1- dim))
                     (error "索引 ~D 越界（轴大小 ~D）" idx dim))
                   nidx))
               (normalize-slice-endpoint (idx dim)
                 "将负索引转为非负，并检查是否在 [0, dim] 内（用于切片端点）。"
                 (let ((nidx (if (minusp idx) (+ idx dim) idx)))
                   (unless (<= 0 nidx dim)
                     (error "切片端点 ~D 越界（轴大小 ~D）" idx dim))
                   nidx))
               (obj->keep-indices (obj dim)
                 "根据 obj 返回保留的索引列表（严格递增）。"
                 (let ((del-set (make-hash-table :test 'eql)))
                   (flet ((add-del (i)
                            (setf (gethash i del-set) t)))
                     (cond ((integerp obj)
                            (add-del (normalize-index obj dim)))
                           ((and (listp obj) (eq (first obj) :slice))
                            (destructuring-bind (_ start end) obj
                              (declare (ignore _))
                              (let ((s (normalize-slice-endpoint start dim))
                                    (e (normalize-slice-endpoint end dim)))
                                (when (> s e)
                                  (error "切片起始 ~D 不能大于终止 ~D" s e))
                                (loop for i from s below e do (add-del i)))))
                           ((listp obj)
                            (dolist (idx obj)
                              (add-del (normalize-index idx dim))))
                           (t (error "无效的 obj 类型 ~A" obj))))
                   ;; 收集所有未被删除的索引
                   (loop for i from 0 below dim
                         unless (gethash i del-set) collect i))))
        (if (null axis)
            ;; 展平模式：先展平为 1D，然后按保留索引提取
            (let* ((flat (vt-ravel tensor))          ; 零拷贝视图或连续副本
                   (flat-size (vt-size flat))
                   (keep (obj->keep-indices obj flat-size)))
              (if keep
                  (vt-take flat (vt-from-sequence keep :type 'fixnum))
                  (vt-zeros (list 0) :type etype)))
            ;; 沿轴删除
            (let* ((ax (vt-normalize-axis axis rank))
                   (ax-dim (nth ax sh))
                   (keep (obj->keep-indices obj ax-dim)))
              (if keep
                  (let ((slices (loop for i in keep
                                      collect (vt-narrow tensor ax i (1+ i)))))
                    (apply #'vt-concatenate ax slices))
                  ;; 轴大小变为 0
                  (let ((new-sh (copy-list sh)))
                    (setf (nth ax new-sh) 0)
                    (vt-zeros new-sh :type etype)))))))))

;;; ===========================================
;;; 4. 统计扩展
;;; ===========================================

(defun vt-prod (tensor &key axis)
  "求积.
  axis: 归约轴(nil 表示全局)"
  (with-float-safe
    (nth-value 0 (vt-reduce tensor axis 1
                            (lambda (acc val)
                              (declare (type number acc val))
                              (values (* acc val) nil))
                            :return-arg nil))))

(defun vt-unravel-index (offset shape strides)
  "将一维物理偏移 offset 转换为多维逻辑索引（基于 shape 和 strides，行优先序）。"
  (declare (type fixnum offset))
  (with-float-safe
    (loop with rem = offset
          for dim in shape
          for stride in strides
          collect (multiple-value-bind (idx r)
		      (floor rem stride)
                    (setf rem r)
                    idx))))

(defun vt-cumsum (tensor &key axis)
  "累积和。axis 支持负数，返回与输入同类型的新张量。"
  (with-float-safe
    (if axis
	(let* ((shape (vt-shape tensor))
               (rank (length shape))
               (ax (vt-normalize-axis axis rank))
               (axis-size (nth ax shape))
               ;; 创建输出张量，初始化为零
               (result (vt-zeros shape :type (vt-element-type tensor)))
               (out-strides (vt-strides result)))
          (vt-do-each (ptr val result)
            (declare (ignore val))
            (let ((full-idx (vt-unravel-index ptr shape out-strides)))
              ;; 仅处理每条纤维的起点，避免重复计算
              (when (= (nth ax full-idx) 0)
		(let* ((specs (loop for i from 0 below rank
                                    if (= i ax) collect '(:all)
                                      else collect (list (nth i full-idx))))
                       (fiber (apply #'vt-slice tensor specs))
                       (cum (coerce 0 (vt-element-type tensor))))
                  (loop for i from 0 below axis-size
			for val = (vt-ref fiber i)
			do (setf cum (+ cum val))
                           ;; 写入输出对应位置
                           (let ((pos-idx (copy-list full-idx)))
                             (setf (nth ax pos-idx) i)
                             (setf (apply #'vt-ref result pos-idx) cum)))))))
          result)
	;; axis = nil：展平后续累积和
	(let* ((flat (vt-flatten tensor))
               (size (vt-size flat))
               (result (vt-zeros (list size) :type (vt-element-type tensor)))
               (in-data (vt-data flat))
               (out-data (vt-data result))
               (cum (coerce 0 (vt-element-type tensor))))
          (loop for i fixnum from 0 below size
		do (setf cum (+ cum (aref in-data i)))
                   (setf (aref out-data i) cum))
          result))))

(defun vt-cumprod (tensor &key axis)
  "累积积。axis 支持负数，返回与输入同类型的新张量。"
  (with-float-safe
    (if axis
	(let* ((shape (vt-shape tensor))
               (rank (length shape))
               (ax (vt-normalize-axis axis rank))
               (axis-size (nth ax shape))
               (result (vt-zeros shape :type (vt-element-type tensor))) ; 实际会被重写
               (out-strides (vt-strides result)))
          (vt-do-each (ptr val result)
            (declare (ignore val))
            (let ((full-idx (vt-unravel-index ptr shape out-strides)))
              (when (= (nth ax full-idx) 0)
		(let* ((specs (loop for i from 0 below rank
                                    if (= i ax) collect '(:all)
                                      else collect (list (nth i full-idx))))
                       (fiber (apply #'vt-slice tensor specs))
                       (cum (coerce 1 (vt-element-type tensor)))) ; 初始值为 1
                  (loop for i from 0 below axis-size
			for val = (vt-ref fiber i)
			do (setf cum (* cum val))
                           (let ((pos-idx (copy-list full-idx)))
                             (setf (nth ax pos-idx) i)
                             (setf (apply #'vt-ref result pos-idx) cum)))))))
          result)
	(let* ((flat (vt-flatten tensor))
               (size (vt-size flat))
               (result (vt-zeros (list size) :type (vt-element-type tensor)))
               (in-data (vt-data flat))
               (out-data (vt-data result))
               (cum (coerce 1 (vt-element-type tensor))))
          (loop for i fixnum from 0 below size
		do (setf cum (* cum (aref in-data i)))
                   (setf (aref out-data i) cum))
          result))))

(defun vt-median (tensor &key axis)
  "中位数。axis 支持负数 (nil 表示全局)。"
  (with-float-safe
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
		(let ((vals (vt-numpy-sort (loop for i below fiber-size
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
               (vals (vt-numpy-sort (loop for i below size
					  collect (aref (vt-data flat) i))
				    #'<)))
	   (if (oddp size)
               (make-vt nil (coerce (nth (floor size 2) vals) 'double-float)
			:type 'double-float)
            (make-vt nil (/ (+ (coerce (nth (1- (floor size 2)) vals) 'double-float)
                               (coerce (nth (floor size 2) vals) 'double-float))
                            2.0d0)
                     :type 'double-float))))))


(defun percent-from-sorted (sorted q interpolation)
  "从已排序列表 sorted 中，按分数 q (0..1) 和插值方法 interpolation 计算百分位值。
   interpolation 可选 :linear, :lower, :higher, :midpoint, :nearest。"
  (with-float-safe
    (let* ((n (length sorted))
           (idx (* q (1- n)))
           (lower (floor idx))
           (upper (min (ceiling idx) (1- n)))
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
	 (error "unknown interpolation method ~a" interpolation))))))

(defun vt-percentile (tensor percentile &key axis (interpolation :linear))
  "计算百分位数。
   percentile : 0~100 的百分位数值。
   axis       : 归约轴，支持负数 (nil 表示全局)。
   interpolation : :linear, :lower, :higher, :midpoint,
                   :nearest (默认 :linear)。"
  (with-float-safe
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
                       (vals (vt-numpy-sort (loop for i below fiber-size
						  collect (vt-ref fiber i))
					    #'<)))
                  (setf (aref (vt-data result) ptr)
			(percent-from-sorted vals q interpolation)))))
            result)
          ;; 全局百分位
          (let* ((flat (vt-flatten tensor))
		 (size (vt-size flat))
		 (vals (vt-numpy-sort (loop for i below size
					    collect (aref (vt-data flat) i))
				      #'<)))
	    (make-vt nil (percent-from-sorted vals q interpolation)
		     :type 'double-float))))))

(defun vt-quantile (tensor q &key axis (interpolation :linear))
  "计算分位数.
  q: 分位数(0-1)"
  (with-float-safe
    (vt-percentile tensor (* q 100) :axis axis
				    :interpolation interpolation)))

(defun vt-ptp (tensor &key axis)
  "峰-峰值(最大值 - 最小值)"
  (with-float-safe
    (if axis
	(vt-- (vt-amax tensor :axis axis)
	      (vt-amin tensor :axis axis))
	(- (vt-item (vt-amax tensor))
	   (vt-item (vt-amin tensor))))))

(defun vt-histogram (tensor &key bins range density)
  "计算直方图。
   bins   : bin 数量（默认 10）。
   range  : (min, max) 列表。若未提供，要求数据中不含 nan 或 inf。
   density: 若为 t，返回概率密度直方图（总面积=1）。
   返回   : (hist, bin-edges) 两个一维张量。

   行为与 numpy 一致：
   - 不指定 range 时，数据必须全部有限，否则报错。
   - 指定 range 后，nan 和超出 [min, max] 的值会被忽略。
   - 注意：bin-edges 包含右边界，最后一个 bin 是闭区间 [edges[-2], edges[-1]]。"
  (with-float-safe
    (let* ((flat (vt-flatten tensor))
           (data (vt-data flat))
           (size (vt-size flat))
           (bins (or bins 10)))
      ;; ---------- 自动确定范围 ----------
      (let ((data-min nil)
            (data-max nil))
        (if range
            (setf data-min (first range)
                  data-max (second range))
            (progn
              ;; 检查是否包含非有限值
              (let ((finite-check (vt-all (vt-isfinite tensor))))
		  (unless (= (vt-item finite-check) 1.0d0)
                  (error "automatic bin range determination requires finite input.~
                       ~%  please provide explicit 'range' argument, or remove nan/inf values.")))
              ;; 数据全为有限值，安全地计算最小最大值
              (setf data-min (vt-item (vt-amin tensor))
                    data-max (vt-item (vt-amax tensor)))))
        ;; 处理范围相等边界情况（所有值相同）
        (when (= data-min data-max)
          (setf data-min (- data-min 0.5)
                data-max (+ data-max 0.5)))
        (let* ((bin-width (/ (- data-max data-min) bins))
               (hist (make-array bins :initial-element 0))
               (bin-edges (make-array (1+ bins) :element-type t)))

          ;; 生成 bin edges
          (loop for i from 0 to bins
                do (setf (aref bin-edges i)
                         (+ data-min (* i bin-width))))
	  ;; 统计计算
	  (loop for i from 0 below size
                for val = (aref data i)
                when (and (>= val data-min) (<= val data-max))
                  do (let ((bin-idx (if (= val data-max)
                                        (1- bins)
                                        (floor (- val data-min)
					       bin-width))))
                       (incf (aref hist bin-idx))))

          ;; 密度归一化
          (when density
            (let ((total (reduce #'+ hist)))
              (if (zerop total)
                  (loop for i from 0 below bins
                        do (setf (aref hist i) 0.0d0))
                  (loop for i from 0 below bins
                        do (setf (aref hist i)
                                 (/ (aref hist i)
                                    (* total bin-width)))))))

          ;; 返回
          (values (vt-from-sequence
		   (coerce hist 'list) :type 'double-float)
                  (vt-from-sequence
		   (coerce bin-edges 'list) :type 'double-float)))))))


;;; ===========================================
;;; 5. 逻辑运算扩展
;;; ===========================================

(defun vt-logical-and (t1 t2)
  "逻辑与"
  (with-float-safe
    (vt-map (lambda (a b)
	      (if (and (/= a 0)
		       (/= b 0))
		  1.0d0 0.0d0))
	    t1 t2)))

(defun vt-logical-or (t1 t2)
  "逻辑或"
  (with-float-safe
    (vt-map (lambda (a b)
	      (if (or (/= a 0)
		      (/= b 0))
		  1.0d0 0.0d0))
	    t1 t2)))

(defun vt-logical-not (vt)
  "逻辑非"
  (with-float-safe
    (vt-map (lambda (a)
	      (if (= a 0)
		  1.0d0 0.0d0))
	    vt)))

(defun vt-logical-xor (t1 t2)
  "逻辑异或"
  (with-float-safe
    (vt-map (lambda (a b) 
              (if (/= (if (= a 0) 0 1)
		      (if (= b 0) 0 1))
		  1.0d0
		  0.0d0)) 
            t1 t2)))

(defun vt-all (condition &key axis)
  "检查所有元素是否为真"
  (with-float-safe
    (nth-value
     0 (vt-reduce condition axis 1.0d0
                  (lambda (acc val)
                    (declare (type number acc val))
                    (values (if (and (/= acc 0) (/= val 0))
				1.0d0 0.0d0)
			    nil))
                  :return-arg nil))))

(defun vt-any (condition &key axis)
  "检查是否存在任一元素为真"
  (with-float-safe
    (nth-value
     0 (vt-reduce condition axis 0.0d0
                  (lambda (acc val)
                    (declare (type number acc val))
                    (values (if (or (/= acc 0) (/= val 0))
				1.0d0 0.0d0)
			    nil))
                  :return-arg nil))))

(defun vt-isclose (t1 t2 &key (rtol 1e-5) (atol 1e-8))
  "判断两个数组元素是否在容差范围内接近。
   注意：此实现采用对称容差 (使用 max(|a|, |b|))，与 pytorch 一致。"
  (with-float-safe
    (vt-map (lambda (a b)
              ;; 1. 利用 cl 的 = 运算符：inf == inf 为 t，nan == nan 为 nil
              ;; 这一步直接拦截了无穷大相等的情况，并加速了完全相同的元素
              (if (vt-float-nan-inf-= a b)
                  1.0d0
                  ;; 2. 对于不相等的有限浮点数，进行容差比较
                  ;; (如果包含 nan，- 运算会得到 nan，随后的 <= 会返回 nil，逻辑依然正确)
                  (let ((diff (abs (- a b))))
                    (if (<= diff (+ atol (* rtol (max (abs a) (abs b)))))
			1.0d0 
			0.0d0))))
            t1 t2)))


(defun vt-allclose (t1 t2 &key (rtol 1e-5) (atol 1e-8))
  "判断两个数组整体是否在容差范围内接近"
  (with-float-safe
    (= (vt-item
	(vt-all
	 (vt-isclose t1 t2 :rtol rtol :atol atol)))
       1.0d0)))

(defun vt-isfinite (vt)
  "检查是否为有限值（非 nan，非无穷）。跨平台兼容实现。"
  (with-float-safe
    (let ((pos-inf (vt-float-pos-inf))
          (neg-inf (vt-float-neg-inf)))
      (vt-map (lambda (x)
		(if (and (vt-float-nan-= x x) ; 排除 nan
			 (not (vt-float-inf-= x pos-inf))
			 (not (vt-float-inf-= x neg-inf)))
                    1.0d0 0.0d0))
              vt))))

(defun vt-isinf (vt)
  "检查是否为无穷。"
  (with-float-safe
    (let ((pos-inf (vt-float-pos-inf))
          (neg-inf (vt-float-neg-inf)))
      (vt-map (lambda (x)
		(if (or (vt-float-nan-= x pos-inf)
			(vt-float-inf-= x neg-inf))
                    1.0d0 0.0d0))
              vt))))

(defun vt-isnan (vt)
  "检查是否为 nan"
  (with-float-safe
    (vt-map (lambda (x)
              (if (not (vt-float-nan-= x x))  ; nan 不等于自身
                  1.0d0 0.0d0))
            vt)))

(defun vt-sort (tensor &key (axis -1))
  "沿指定轴排序；axis 为 nil 时展平后排序。axis 支持负数。"
  (with-float-safe
    (if axis
        (let* ((shape (vt-shape tensor))
               (rank (length shape))
               (ax (vt-normalize-axis axis rank))
               (ax-dim (nth ax shape))
               (in-strides (vt-strides tensor))
               (in-offset (vt-offset tensor))
               (in-data (vt-data tensor))
               ;; 结果张量完全复制输入（形状、类型、数据）
               (result (vt-copy tensor))
               (out-strides (vt-strides result))
               (out-data (vt-data result)))
          (labels ((recurse (depth in-ptr out-ptr)
                     (cond
                       ;; 抵达排序轴：收集该“纤维”上的值，排序后写回
                       ((= depth ax)
                        (let* ((in-stride (nth ax in-strides))
                               (out-stride (nth ax out-strides))
                               ;; 收集值列表
                               (vals (loop for i fixnum from 0 below ax-dim
                                           for off = (+ in-ptr (* i in-stride))
                                           collect (aref in-data off)))
                               ;; 使用 nan 安全的 vt-numpy-sort 排序
                               (sorted-vals (vt-numpy-sort vals #'<)))
                          ;; 将排序后的值按顺序写回结果张量
                          (loop for val in sorted-vals
                                for off = out-ptr then (+ off out-stride)
                                do (setf (aref out-data off) val))))
                       ;; 非排序轴：遍历该维度，递归进入下一深度
                       ((< depth rank)
                        (let ((dim (nth depth shape))
                              (in-stride (nth depth in-strides))
                              (out-stride (nth depth out-strides)))
                          (loop for i fixnum from 0 below dim do
                            (recurse (1+ depth)
                                     (+ in-ptr (* i in-stride))
                                     (+ out-ptr (* i out-stride))))))
                       (t nil))))
            (recurse 0 in-offset 0))
          result)
        ;; axis = nil：展平为一维后排序
        (let* ((flat (vt-flatten tensor))
               (data (coerce (vt-data flat) 'list))
               (sorted (vt-numpy-sort data #'<)))
          (vt-from-sequence sorted :type (vt-element-type tensor))))))

(defun vt-argsort (tensor &key (axis -1))
  "返回沿指定轴的排序索引张量，形状与输入相同。
   axis 默认 -1（最后一轴），支持负数；axis = nil 时展平排序。"
  (with-float-safe
    (if (null axis)
        ;; ========== 展平排序 ==========
        (let* ((flat (vt-ravel tensor))
               (n (vt-size flat))
               (in-data (vt-data flat))
               ;; 构造 (值 . 原始索引) 对
               (pairs (loop for i fixnum from 0 below n
                            collect (cons (aref in-data i) i))))
          ;; 分离 nan 和非 nan，保持各自原始顺序
          (let ((non-nans '())
                (nans '()))
            (dolist (p pairs)
              (if (vt-float-nan-p (car p))
                  (push p nans)
                  (push p non-nans)))
            (setq non-nans (nreverse non-nans)
                  nans      (nreverse nans))
            ;; 非 nan 部分按值稳定排序
            (setq non-nans (stable-sort non-nans #'< :key #'car))
            ;; 拼接：升序时 nan 在末尾
            (let ((sorted-indices (mapcar #'cdr (append non-nans nans))))
              (%make-vt :data (make-array n :element-type 'fixnum
                                            :initial-contents sorted-indices)
                        :shape (list n)
                        :strides '(1)
                        :offset 0
                        :etype 'fixnum))))

        ;; ========== 沿轴排序 ==========
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
                   ;; 递归终止（实际上不会到达）
                   ((= depth rank) nil)
                   
                   ;; 未到达目标轴：继续沿当前维度遍历
                   ((< depth ax)
                    (let ((dim (nth depth shape))
                          (in-stride (nth depth in-strides))
                          (out-stride (nth depth out-strides)))
                      (dotimes (i dim)
                        (recurse (1+ depth)
                                 (+ in-ptr (* i in-stride))
                                 (+ out-ptr (* i out-stride))))))
                   
                   ;; 到达目标轴：处理该轴上一根纤维
                   ((= depth ax)
                    (let* ((in-stride (nth ax in-strides))
                           (out-stride (nth ax out-strides))
                           ;; 目标轴之后的维度（尾部）
                           (tail-dims (subseq shape (1+ ax)))
                           (tail-rank (length tail-dims))
                           (tail-strides (subseq out-strides (1+ ax)))
                           (tail-size (reduce #'* tail-dims :initial-value 1))
                           (tail-in-strides
                             (loop for idx from 0 below tail-rank
                                   collect (nth (+ ax 1 idx) in-strides))))
                      ;; 遍历每根纤维（通过线性索引 tail-i）
                      (dotimes (tail-i tail-size)
                        ;; ---------- 计算尾部偏移 ----------
                        (let ((extra-in-off 0)
                              (extra-out-off 0)
                              (rem tail-i))
                          ;; 行优先分解：从最后一维向前取模
                          (loop for idx from (1- tail-rank) downto 0
                                for dim = (nth idx tail-dims)
                                for in-strd = (nth idx tail-in-strides)
                                for out-strd = (nth idx tail-strides)
                                do (multiple-value-bind (q r) (floor rem dim)
                                     ;; r 是该维度的坐标，乘上步长累加到偏移
                                     (incf extra-in-off (* r in-strd))
                                     (incf extra-out-off (* r out-strd))
                                     (setf rem q)))
                          
                          ;; ---------- 收集纤维的 (值 . 位置) 对 ----------
                          (let ((pairs
                                  (loop for pos fixnum from 0 below ax-dim
                                        for in-off = (+ in-ptr (* pos in-stride)
                                                        extra-in-off)
                                        collect (cons (aref in-data in-off) pos))))
                            ;; 分离 nan 和非 nan
                            (let ((non-nans '())
                                  (nans '()))
                              (dolist (p pairs)
                                (if (vt-float-nan-p (car p))
                                    (push p nans)
                                    (push p non-nans)))
                              (setq non-nans (nreverse non-nans)
                                    nans      (nreverse nans))
                              ;; 对非 nan 稳定排序
                              (setq non-nans (stable-sort non-nans #'< :key #'car))
                              ;; 写回索引（cdr 即原始位置）
                              (loop for (val . pos) in (append non-nans nans)
                                    for out-off = out-ptr then (+ out-off out-stride)
                                    do (setf (aref out-data (+ out-off extra-out-off))
                                             pos))))))))
                   
                   (t nil))))
            (recurse 0 in-offset 0))
          result))))

(defun vt-unique (tensor &key return-index return-inverse return-counts)
  "返回展平后张量中的唯一元素，自动排序（升序），所有 nan 视作同一个值。
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
  (with-float-safe
    (let* ((flat (vt-flatten tensor))
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

      ;; nan 安全的排序：将 nan 排在末尾且连续
      (setf sorted-idx
	    (sort sorted-idx
		  (lambda (a b)
		    (let ((va (aref src-data a))
			  (vb (aref src-data b)))
		      (cond ((vt-float-nan-p va) nil)   ; nan 最大
			    ((vt-float-nan-p vb) t)     ; 其他 < nan
			    (t (< va vb)))))))

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
		 ;; 分组：nan 与普通值分开处理
		 (if (vt-float-nan-p val)
		     ;; nan 分组：连续 nan 视为同一个值
		     (loop while (and (< pos n)
				      (vt-float-nan-p
                                       (aref src-data (aref sorted-idx pos))))
			   for orig = (aref sorted-idx pos)
			   do (setf (aref inverse orig) uniq-num)
			      (incf pos))
		     ;; 普通数值分组：依赖数值相等
		     (loop while (and (< pos n)
				      (= val
					 (aref src-data (aref sorted-idx pos))))
			   for orig = (aref sorted-idx pos)
			   do (setf (aref inverse orig) uniq-num)
			      (incf pos)))
		 (vector-push-extend (- pos start) cnts))
	;; 转换为 vt
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
				     :offset 0
				     :etype 'fixnum))))
              (cnt-vt  (when return-counts
			 (vt-from-sequence (coerce cnts 'list)
					   :type 'fixnum))))
          (if (or return-index return-inverse return-counts)
              (values uniq-vt
                      (when return-index idx-vt)
                      (when return-inverse inv-vt)
                      (when return-counts cnt-vt))
              uniq-vt))))))

(defun vt-nonzero (condition)
  "返回非零元素的索引(已存在 vt-where)"
  (vt-where condition))

(defun vt-extract (condition tensor)
  "根据条件从数组中提取元素"
  (with-float-safe
    (let* ((flat-cond (vt-flatten condition))
           (flat-data (vt-flatten tensor))
           (cond-data (vt-data flat-cond))
           (tensor-data (vt-data flat-data))
           (size (vt-size flat-data))
           (result-list '()))
      (loop for i from 0 below size
            when (/= (aref cond-data i) 0)
              do (push (aref tensor-data i) result-list))
      (vt-from-sequence (nreverse result-list)))))

(defun vt-searchsorted (tensor values &key side)
  "在有序数组中查找插入点"
  (declare (ignore side))
  (with-float-safe
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
      (%make-vt :data result
		:shape (list v-size)
		:strides '(1)
		:offset 0
		:etype 'fixnum))))

;;; ===========================================
;;; 7. 数据类型转换
;;; ===========================================

(defun vt-astype (tensor new-type)
  "将张量转换为新类型。浮点数→整数时使用截断 (truncate)，其余使用 coerce。"
  (with-float-safe
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
      new)))

;;; ===========================================
;;; 8. 集合操作
;;; ===========================================

(defun vt-intersect1d (t1 t2)
  "交集"
  (with-float-safe
    (let* ((u1 (vt-unique t1))
           (u2 (vt-unique t2))
           (u2-set (coerce (vt-data u2) 'list)))
      (let ((result '()))
	(vt-do-each (ptr val u1)
	  (declare (ignore ptr))
	  (when (member val u2-set)
	    (push val result)))
	(vt-from-sequence (vt-numpy-sort result #'<)
			  :type (vt-element-type t1))))))

(defun vt-union1d (t1 t2)
  "并集"
  (with-float-safe
    (let* ((u1 (vt-unique t1))
           (u2 (vt-unique t2)))
      (vt-unique (vt-concatenate 0 u1 u2)))))

(defun vt-setdiff1d (t1 t2)
  "差集(在 t1 但不在 t2)"
  (with-float-safe
    (let* ((u1 (vt-unique t1))
           (u2 (vt-unique t2))
           (u2-set (coerce (vt-data u2) 'list)))
      (let ((result '()))
	(vt-do-each (ptr val u1)
	  (declare (ignore ptr))
	  (unless (member val u2-set)
	    (push val result)))
	(vt-from-sequence (vt-numpy-sort result #'<)
			  :type (vt-element-type t1))))))

(defun vt-setxor1d (t1 t2)
  "对称差集"
  (with-float-safe
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
	(vt-from-sequence (vt-numpy-sort result #'<)
			  :type (vt-element-type t1))))))

(defun vt-in1d (t1 t2)
  "检查数组元素是否在另一个数组中"
  (with-float-safe
    (let ((t2-set (coerce (vt-data (vt-unique t2)) 'list)))
      (vt-map (lambda (x) (if (member x t2-set) 1.0d0 0.0d0))
	      t1))))

;;; ===========================================
;;; 9. 其他数学运算
;;; ===========================================

(defun vt-clip (tensor min max)
  "将数值限制在指定范围内"
  (with-float-safe
    (vt-map (lambda (x) (max min (min max x))) tensor)))

(defun vt-gradient (tensor &key (spacing 1.0d0) axis)
  "计算张量的梯度（沿指定轴的二阶中心差分）。
   axis  : nil → 全部轴, 整数或整数列表 → 指定轴（支持负数）。
   spacing: 标量（统一间距）、列表（按轴间距，长度需与轴数一致）、
            或单轴配合 1d 张量作为坐标数组。
   返回：若 axis 为 nil 或列表，返回梯度张量的列表；若 axis 为单个整数，返回单个张量。"
  (with-float-safe
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
                       (error "when spacing is a tensor, axis must be a single axis."))
                     (list spacing))
                    (t (error "spacing must be a number, list or 1d tensor.")))))
	(labels
	    ((slice-specs (ax start end)
	       "生成沿轴 ax 的切片规格：从 start (包含) 到 end (不包含)。"
	       (loop for d from 0 below rank
                     collect (if (= d ax) (list start end) '(:all))))
             (grad-along-axis (ax sp)
	       (let* ((n (nth ax shape)))
		 (when (< n 2)
                   (error "tensor size ~d along axis ~d is too small for gradient." n ax))
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
		       (assert (= n n-h) (sp) "spacing array must have same length as axis.")
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
		results)))))))


;;; ===========================================
;;; 10. 索引高级功能
;;; ===========================================

(defun vt-take (tensor indices &key axis)
  "从张量中按索引取值。支持多维 indices 和负数 axis。
   当 axis=nil 且 indices 为标量数字时，直接返回数值；
   否则返回张量。"
  ;; 帮助函数：将任意形式的 indices 转为 fixnum 简单数组
  (with-float-safe
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
                    (error "索引 ~d 越界，展平大小 ~d" idx size))
		  (make-vt nil (aref (vt-data flat) idx)
			   :type (vt-element-type tensor)))
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
                            (error "索引 ~d 越界，展平大小 ~d" idx size))
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
                             (error "索引 ~d 越界，轴 ~d 大小 ~d"
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
              out))))))

(defun vt-put (tensor indices values &key (mode :raise))
  "将 values 按 indices 指定的展平（行优先）索引放入 tensor。
   tensor 被原地修改并返回。
   indices: 整数、一维张量或列表。
   values: 标量、一维张量或列表，长度不足时循环使用。
   mode: :raise (越界报错，默认), :wrap (越界取模), :clip (越界截断)。"
  (with-float-safe
    (let* ((total-size (vt-size tensor))
           ;; 统一 indices 为整数列表
           (idx-list
             (if (numberp indices)
		 (list (round indices))
		 (let ((flat (vt-flatten (ensure-vt indices))))
                   (loop for i below (vt-size flat)
			 collect (truncate (aref (vt-data flat) i))))))
           ;; 统一 values 为列表
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
      
      (loop for raw-idx in idx-list
            for val-idx from 0 
            ;; 优化：用 mod 递增代替 (nth (mod i n) list)，保证 o(1) 取值
            for val = (nth (mod val-idx n-values) val-list)
            do (let ((idx raw-idx))
		 ;; 处理越界模式
		 (cond
                   ((and (>= idx 0) (< idx total-size)) nil) ; 合法，什么都不做
                   ((eq mode :clip) (setq idx (max 0 (min idx (1- total-size)))))
                   ((eq mode :wrap) (setq idx (mod idx total-size)))
                   (t (error "索引 ~d 超出范围 [0, ~d)" raw-idx total-size)))
		 
		 ;; 核心算法：1d逻辑索引 → 多维坐标 → 物理偏移
		 (let ((remaining idx)
                       (phys-offset offset))
                   (loop for dim in (reverse shape)
			 for stride in (reverse strides)
			 do (multiple-value-bind (q r) (floor remaining dim)
                              (incf phys-offset (* r stride))
                              (setf remaining q)))
                   ;; 写入底层物理内存
                   (setf (aref data phys-offset)
			 (coerce val elem-type)))))
      tensor)))

(defun vt-choose (choices indices)
  "根据索引数组从多个数组中选择值"
  (with-float-safe
    (let* ((n-choices (length choices))
           (idx-flat (vt-flatten indices))
           (idx-data (vt-data idx-flat))
           (idx-size (vt-size idx-flat))
           (result-data (make-array idx-size
				    :element-type 'double-float)))
      (loop for i from 0 below idx-size
            for idx = (aref idx-data i)
            do (when (or (< idx 0) (>= idx n-choices))
		 (error "选择索引 ~a 越界" idx))
               (setf (aref result-data i)
                     (aref (vt-data (nth idx choices)) i)))
      (%make-vt :data result-data
		:shape (list idx-size)
		:strides '(1)
		:offset 0
		:etype 'double-float))))

(defun vt-select (condlist choicelist &key default)
  "根据多个条件从多个数组中选择值"
  (declare (ignore default))
  (with-float-safe   
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
      result)))

(defun vt-flip (vt &key axis)
  "沿指定轴翻转，返回零拷贝视图"
  (with-float-safe
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
                    :offset new-offset
		    :etype (vt-element-type vt)))
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
			       (* (1- dim) old-stride))
		    :etype (vt-element-type vt))))))

(defun vt-roll (vt shift &key axis)
  "滚动元素。shift 可为整数或列表，axis 可为整数、列表或 nil。
   多轴模式下 shift 与 axis 长度需相等。"
  ;; 统一 shift 为列表
  (with-float-safe
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
                           (vt-shape vt)))))))))


(defun normalize-pad-width (pad-width rank)
  "将 pad-width 标准化为 列表。"
  (with-float-safe
    (let ((pw-list (if (integerp pad-width)
                       (make-list rank :initial-element pad-width)
                       pad-width)))
      (unless (= (length pw-list) rank)
	(error "pad-width length ~a does not match tensor rank ~a"
	       (length pw-list) rank))
      (mapcar (lambda (x)
		(cond ((integerp x) 
                       (if (< x 0)
			   (error "negative padding is not supported")
			   (list x x)))
                      ((and (consp x) (= (length x) 2))
                       (if (or (< (first x) 0) (< (second x) 0))
                           (error "negative padding is not supported")
                           x))
                      (t (error "invalid pad-width element: ~a" x))))
              pw-list))))

(defun apply-pad-mode (mode dist sk side)
  "纯坐标映射：dist 表示距离边缘的距离 (1, 2, 3...)。
   sk 是源张量维度大小。"
  (with-float-safe
    (ecase mode
      (:edge (if (eq side :left) 0 (1- sk)))
      
      (:wrap 
       ;; wrap 左侧起点 sk-1 向左递减；右侧起点 0 向右递增
       (let ((x (if (eq side :left) (- sk dist) (1- dist))))
	 (mod x sk)))
      
      (:reflect
       ;; reflect 左侧起点 1 向左递增；右侧起点 sk-2 向右递减
       (when (<= sk 1)
	 (error "cannot reflect on dimension with size <= 1"))
       (let* ((period (* 2 (1- sk)))
              (x (if (eq side :left) dist (- sk 1 dist)))
              (idx (mod x period)))
	 (if (< idx sk) idx (- period idx))))
      
      (:symmetric
       ;; symmetric 左侧起点 0 向左递增；右侧起点 sk-1 向右递减
       (let* ((period (* 2 sk))
              (x (if (eq side :left) (- dist) (+ sk dist -1)))
              (idx (mod x period)))
	 (if (< idx sk) idx (- period idx 1)))))))

(defun vt-pad (vt pad-width &key (mode :constant) (constant-values 0))
  "对张量进行填充，返回新张量。
   pad-width ：整数（所有轴相同）或长度为 rank 的列表，每个元素可为整数或 对。
   mode :constant（默认）, :edge, :wrap, :reflect, :symmetric。
   constant-values ：常数填充值，可用标量或 列表。"
  (with-float-safe
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
                               ;; cl中 (- a b -1) 等价于 a - b + 1
                               (apply-pad-mode
				mode (- offset sk -1) sk :right))))
                    ;; 原始数据区
                    (t
                     (setf (nth d src-idxs) offset)))))
           
           (if const-val
               const-val
               (apply #'vt-ref vt src-idxs))))
       :type (vt-element-type vt)))))

(defun vt-diff (vt &key (axis -1) (n 1))
  "沿指定轴计算 n 阶差分，与 numpy.diff 兼容。"
  (with-float-safe
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
	    finally (return result)))))

(defun vt-bincount (x &key (minlength 0))
  "统计非负整数数组中每个值出现的次数。"
  (with-float-safe
    (let* ((flat (vt-flatten x))
           (maxval (if (zerop (vt-size flat))
		       -1
		       (vt-item (vt-amax flat))))
           (size (max (1+ maxval) minlength))
           (result (make-array size :element-type 'fixnum
				    :initial-element 0)))
      (vt-do-each (ptr val flat)
	(declare (ignore ptr))
	(incf (aref result (truncate val))))
      (%make-vt :data result
		:shape (list size)
		:strides '(1)
		:offset 0
		:etype 'fixnum))))

(defun vt-digitize (x bins &key (right nil))
  "返回输入值在 bins 中的索引。"
  (with-float-safe
    (let* ((flat-x (vt-flatten x))
           (flat-bins (vt-flatten bins))
           (bin-data (vt-data flat-bins))
           (nbins (vt-size flat-bins))
           (result (make-array (vt-size flat-x)
			       :element-type 'fixnum)))
      (vt-do-each (ptr val flat-x)
	(let ((idx (loop for i from 0 below nbins
			 until (if right
                                   (<= val (aref bin-data i))
				   (< val (aref bin-data i)))
			 finally (return i))))
          (setf (aref result ptr) idx)))
      (%make-vt :data result
		:shape (vt-shape flat-x)
		:strides '(1)
		:offset 0
		:etype 'fixnum))))

(defun vt-correlate (a v &key (mode :full))
  "一维互相关（不翻转 v），与 numpy.correlate 完全一致。"
  (with-float-safe
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
		       :offset 0
		       :etype 'double-float))
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
			   :offset 0
			   :etype 'double-float))))
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
			   :offset 0
			   :etype 'double-float))))))))))

(defun vt-convolve (a v &key (mode :full))
  "一维卷积。"
  (with-float-safe
    (vt-correlate a (vt-flip v) :mode mode)))

(defun vt-trapz (y &key (x nil) (dx 1.0d0) (axis -1))
  "使用梯形法则沿指定轴积分，与 numpy.trapz 一致。"
  (with-float-safe
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
                   (vt-diff (ensure-vt x))          ; 1d,长度 n-1
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
          (vt-sum integrand :axis ax))))))

(defun vt-interp (x xp fp &key (left nil) (right nil))
  "一维线性插值。x, xp, fp 均为 1d 张量。"
  (with-float-safe
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
       x))))

(defun vt-hypot (t1 &optional t2)
  "计算 (|t1|^2 + |t2|^2)^0.5。支持广播。"
  (with-float-safe
    (vt-sqrt (vt-+ (vt-square t1) (vt-square (if t2 t2 0))))))

(defun vt-sinc (tensor)
  "计算 sinc(x) = sin(pi*x) / (pi*x)，对 x=0 返回 1。"
  (with-float-safe
    (let* ((x*pi (vt-scale tensor pi)))
      (vt-map (lambda (x)
		(if (zerop x) 1.0d0 (/ (sin x) x)))
              x*pi))))

(defun vt-deg2rad (tensor)
  "角度转弧度。"
  (with-float-safe
    (vt-scale tensor (/ pi 180.0d0))))

(defun vt-rad2deg (tensor)
  "弧度转角度。"
  (with-float-safe
    (vt-scale tensor (/ 180.0d0 pi))))

;;; ===========================================
;;; 12. 随机数扩展
;;; ===========================================

(defvar *vt-default-random-state* (make-random-state *random-state*)
  "clvt 内部使用的默认随机状态。可由 vt-random-seed 修改或通过 let 覆盖。")

(defun vt-make-random-state (&optional seed)
  "创建并返回一个新的 random-state 对象。seed 同 make-random-state 参数。
   示例：
     (vt-make-random-state)     => 使用 t 随机初始化
     (vt-make-random-state nil) => 从当前 *random-state* 复制
     (vt-make-random-state 42)  => 不可移植地使用整数初始化"
  #+sbcl
  (sb-ext::seed-random-state seed)
  #-sbcl
  (typecase seed
    (null (make-random-state nil)) ; 保持与 cl 一致：nil → 复制当前全局状态
    (random-state (make-random-state seed))
    (t (make-random-state seed)))  ; 实现依赖，通常可为整数加载
  )

(defun vt-random-seed (seed)
  "修改 *vt-default-random-state* 并返回新的状态。seed 含义同 vt-make-random-state。
   例：(vt-random-seed 42)  后，所有使用默认 rng 的函数将产生可复现序列。"
  (setf *vt-default-random-state* (vt-make-random-state seed)))

(declaim (inline %uniform-rand %normal-rand))

(defun %uniform-rand (state)
  "生成 [0,1) 均匀分布随机数。"
  (declare (random-state state))
  (random 1.0d0 state))

(defun %normal-rand (state)
  "box-muller 方法生成标准正态分布随机数。"
  (declare (random-state state))
  (let ((u1 (random 1.0d0 state))
        (u2 (random 1.0d0 state)))
    (* (sqrt (* -2.0d0 (log u1)))
       (cos (* 2.0d0 pi u2)))))

(defun vt-random (shape &key (type 'double-float)
			  (rng *vt-default-random-state*))
  "返回形状为 shape 的张量，元素独立同分布于 [0,1) 均匀分布。"
  (declare (list shape) (random-state rng))
  (vt-map (lambda (x)
            (declare (ignore x))
            (coerce (%uniform-rand rng) type))
          (vt-zeros shape :type type)))

(defun vt-random-uniform
    (shape &key (low 0.0d0) (high 1.0d0) (type 'double-float)
	     (rng *vt-default-random-state*))
  "返回 [low, high) 均匀分布随机张量。"
  (declare (list shape) (random-state rng))
  (let ((range (- high low)))
    (vt-map (lambda (x)
              (declare (ignore x))
              (coerce (+ low (* range (%uniform-rand rng))) type))
            (vt-zeros shape :type type))))

(defun vt-random-normal (shape &key (mean 0.0d0) (std 1.0d0)
				 (rng *vt-default-random-state*))
  "返回正态分布随机张量。"
  (declare (list shape) (random-state rng))
  (let ((res (vt-zeros shape :type 'double-float)))
    (vt-do-each (ptr val res)
      (declare (ignore val))
      (setf (aref (vt-data res) ptr)
            (+ mean (* std (%normal-rand rng)))))
    res))

(defun vt-random-int (low high &key (size nil) (type 'fixnum)
				 (rng *vt-default-random-state*))
  "创建随机整数数组.
  low: 下界(包含)
  high: 上界(不包含)
  size: 形状(nil 表示标量)
  返回: 张量"
  (declare (random-state rng))
  (let ((range (- high low)))
    (if size
        (vt-astype (vt-map (lambda (x)
                             (declare (ignore x))
                             (+ low (random range rng)))
                           (vt-zeros size :type type))
                   type)
	 (make-vt nil (+ low (random range rng)) :type type))))

(defun vt-random-integers (low high &key (size nil) (type 'fixnum)
				      (rng *vt-default-random-state*))
  "同 vt-random-int。"
  (vt-random-int low high :size size :type type :rng rng))

