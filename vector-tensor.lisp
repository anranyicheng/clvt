(in-package #:clvt)

(defstruct (vt (:constructor %make-vt))
  "n维张量结构.
   data: 存储数据的一维简单数组.
   shape: 形状列表.
   strides: 步长列表.
   offset: 起始偏移量,支持切片视图不复制数据.
   dtype: 元素数据类型, 如 (:int32, :int64, :float32, :float64) 等"
  (data (make-array 0) :type (simple-array *))
  (shape nil :type list)
  (strides nil :type list)
  (offset 0 :type fixnum)
  (dtype :float64 :type symbol))

(defun vt-itemsize (vt)
  "返回每个元素的字节大小."
  (let ((dtype (vt-dtype vt)))
    (cond ((eq dtype :float64) 8)
          ((eq dtype :float32) 4)
          ((eq dtype :int64)
           #+sbcl sb-vm:n-word-bytes #-sbcl 8)
          ((eq dtype :int32) 4)
          ((eq dtype :uint8) 1)
          ((eq dtype :int8) 1)
          ((eq dtype :uint16) 2)
          ((eq dtype :int16) 2)
          (t 8)))) ; 默认8字节

(defun vt-nbytes (vt)
  "返回张量占用的总字节数."
  (* (vt-size vt) (vt-itemsize vt)))

(declaim (inline vt-element-type
		 vt-shape
		 vt-offset
		 vt-strides
		 vt-data
		 vt-reshape
		 vt-dtype
		 make-vt
		 vt-shape-to-size))

(defun vt-dtype->lisp-type (dtype)
  "将内部 dtype 符号映射为 CL 数组元素类型。"
  (ecase dtype
    (:float64 'double-float)
    (:float32 'single-float)
    (:int64 '(signed-byte 64))
    (:int32 '(signed-byte 32))
    (t 'double-float)))

(defun vt-element-type (vt)
  "返回张量底层物理数组的 Lisp 元素类型。"
  (vt-dtype->lisp-type (vt-dtype vt)))

(defun vt-order (vt)
  "张量的维度大小"
  (length (vt-shape vt)))

(defun vt-size (vt)
  "张量的数据大小"
  (reduce #'* (vt-shape vt) :initial-value 1))

(defun vt-shape-to-size (shape)
  "计算形状对应的总元素个数."
  (declare (list shape))
  (reduce #'* shape :initial-value 1))

(defun vt-promote-type (&rest dtypes)
  "推断运算结果类型。严格对标 NumPy 类型提升规则。
   覆盖 :int32, :int64, :float32, :float64。"
  (let ((has-f64 (member :float64 dtypes))
        (has-f32 (member :float32 dtypes))
        (has-i64 (member :int64 dtypes))
        (has-i32 (member :int32 dtypes)))
    (cond
      ;; 1. 任何 float64 参与运算，结果必为 float64
      (has-f64 :float64)
      ;; 2. float32 与 int64 运算，NumPy 提升为 float64 防止精度丢失
      ((and has-f32 has-i64) :float64)
      ;; 3. 纯 float32 运算
      (has-f32 :float32)
      ;; 4. 整数运算提升链: i64 -> i32
      (has-i64 :int64)
      (has-i32 :int32)
      ;; 5. 默认回退 (如空列表)：对标 NumPy 默认浮点类型
      (t :float64))))

(defun vt-cast (val dtype)
  "安全类型转换。对标 NumPy，浮点转整数时执行截断 (truncate)。"
  (ecase dtype
    (:int32 (truncate val))
    (:int64 (truncate val))
    (:float32 (coerce val 'single-float))
    (:float64 (coerce val 'double-float))))

(defun vt-cast-fun (dtype)
  "安全类型转换。对标 NumPy，浮点转整数时执行截断 (truncate)。"
  (ecase dtype
    (:int32 #'truncate)
    (:int64 #'truncate)
    (:float32 (lambda (val) (coerce val 'single-float)))
    (:float64 (lambda (val) (coerce val 'double-float)))))

(defun make-vt (shape initial-element &key (dtype :float64))
  "创建一个指定形状和类型的张量，并用初始值填充。"
  (let* ((size (vt-shape-to-size shape))
         (lisp-type (vt-dtype->lisp-type dtype))
         (data (make-array size
                           :element-type lisp-type
                           :initial-element
			   (coerce initial-element lisp-type)))
         (strides (vt-compute-strides shape)))
    (%make-vt :data data
              :shape shape
              :strides strides
              :offset 0
              :dtype dtype)))


(defun ensure-vt (obj &key (dtype nil))
  "将标量转换为 0 维张量。显式 dtype 优先，否则基于 Lisp 原生类型推断。"
  (etypecase obj
    (vt
     (if (and dtype (not (eq (vt-dtype obj) dtype)))
	 (vt-copy-into (make-vt (vt-shape obj) 0 :dtype dtype) obj) ; 强制转换
	 obj))
    (number
     (let* ((infer-dtype (cond
                           ((typep obj 'single-float) :float32)
                           ((typep obj 'double-float) :float64)
                           ((typep obj 'integer)      :int64)
                           (t                         :float64)))
            (final-dtype (or dtype infer-dtype))
            (lisp-type (vt-dtype->lisp-type final-dtype)))
       (%make-vt :data (make-array 1
                                   :initial-element (coerce obj lisp-type)
                                   :element-type lisp-type)
                 :shape nil
                 :strides nil
                 :offset 0
                 :dtype final-dtype
                 )))
    (sequence
     (let ((final-dtype (or dtype :float64)))
       (vt-from-sequence obj :dtype final-dtype)))))

(defun vt-flatten-sequence (seq)
  "深度优先遍历 seq 及其嵌套序列,返回所有原子元素的列表.
   支持列表、向量、多维数组、位数组等序列类型,按行优先顺序处理."
  (with-float-safe
    (labels ((sequence-p (obj)
               (or (listp obj) (arrayp obj))))
      (if (not (sequence-p seq))
          ;; 输入不是序列,直接包装为列表返回
          (list seq)
          (let ((result '())
		(stack (list (cons seq (if (listp seq) seq 0)))))
            (loop
              (unless stack (return))
              (let* ((frame (pop stack))
                     (s (car frame))
                     (state (cdr frame)))
		(cond
                  ((listp s)
                   ;; 列表处理:state 是当前 cons 单元
                   (if (null state)
                       nil  ; 列表结束
                       (let ((elt (car state))
                             (new-state (cdr state)))
			 ;; 保存当前帧的后续状态
			 (push (cons s new-state) stack)
			 ;; 处理元素
			 (if (sequence-p elt)
                             (push (cons elt (if (listp elt) elt 0))
				   stack)
                             (push elt result)))))
                  ((arrayp s)
                   ;; 数组处理:state 是当前索引 (0..total-size-1)
                   (let ((len (array-total-size s)))
                     (if (>= state len)
			 nil  ; 数组结束
			 (let ((elt (row-major-aref s state))
                               (new-state (1+ state)))
                           (push (cons s new-state) stack)
                           (if (sequence-p elt)
                               (push (cons elt (if (listp elt) elt 0))
				     stack)
                               (push elt result))))))
                  (t (error "~s 不是序列" s)))))
            (nreverse result))))))

(defun vt-from-sequence (contents &key (dtype :float64))
  "从嵌套序列创建张量。支持任意维度的规则嵌套列表或向量。
   例如 (vt-from-sequence '((1 2) (3 4))) 返回形状为 (2 2) 的张量。
   若传入一维序列，返回一维张量。
   注意: 空序列 () 或 #() 将被视为形状 (0) 的一维张量。
   不规则嵌套报错 如: (vt-from-sequence  #(1 '(2) 3 4))"
  (with-float-safe
    (labels
        ((infer-shape (seq)
           (typecase seq
             (list
              (if (null seq)
                  (list 0)
                  (let* ((first (car seq))
                         (rest-shape (typecase first
                                       (list (infer-shape first))
                                       (vector (infer-shape first))
                                       (t nil))))
                    (if rest-shape
                        ;; 存在嵌套，校验后续元素形状
                        (cons (length seq)
                              (loop for sub in (cdr seq)
                                    unless (equal (infer-shape sub) rest-shape)
                                      do (error "不规则嵌套")
                                    finally (return rest-shape)))
                        ;; 无嵌套，确保后续元素也不是序列
                        (progn
                          (loop for sub in (cdr seq)
                                when (or (listp sub) (typep sub 'vector))
                                  do (error "不规则嵌套"))
                          (list (length seq)))))))
             (vector
              (let ((len (length seq)))
                (if (zerop len) (list 0)
                    (let* ((first (aref seq 0))
                           (rest-shape (typecase first
                                         (list (infer-shape first))
                                         (vector (infer-shape first))
                                         (t nil))))
                      (if rest-shape
                          (cons len
                                (loop for i from 1 below len
                                      for sub = (aref seq i)
                                      unless (equal (infer-shape sub) rest-shape)
                                        do (error "不规则嵌套")
                                      finally (return rest-shape)))
                          (progn
                            (loop for i from 1 below len
                                  for sub = (aref seq i)
                                  when (or (listp sub) (typep sub 'vector))
                                    do (error "不规则嵌套"))
                            (list len)))))))
             (t (error "无法创建张量"))))
         
         ;; 保持原有 5 个参数签名不变，内部优化遍历逻辑
         (fill-tensor (data seq shape strides flat-idx)
           (declare (type list shape strides)
                    (type fixnum flat-idx))
           (if (null shape)
               ;; 叶子节点：直接赋值并强制类型转换
               (setf (aref data flat-idx)
                     (coerce seq (vt-dtype->lisp-type dtype)))
               ;; 非叶子节点：遍历当前维度
               (let ((stride (the fixnum (first strides)))
                     (current-offset flat-idx))
                 (declare (type fixnum stride current-offset))
                 (typecase seq
                   (list
                    (loop for elem in seq do
                      (fill-tensor data elem (rest shape) (rest strides) current-offset)
                      ;; 增量累加偏移量，消除循环内的乘法
                      (incf current-offset stride)))
                   (vector
                    (loop for elem across seq do
                      (fill-tensor data elem (rest shape) (rest strides) current-offset)
                      (incf current-offset stride)))
                   (t (error "fill-tensor: 不支持的序列类型")))))))
      
      (let* ((shape (infer-shape contents))
             (size (reduce #'* shape :initial-value 1))
             (lisp-type (vt-dtype->lisp-type dtype))
             (data (make-array size
                               :element-type lisp-type
                               :initial-element (coerce 0 lisp-type)))
             (strides (vt-compute-strides shape)))
        (fill-tensor data contents shape strides 0)
        (%make-vt :data data
                  :shape shape
                  :strides strides
                  :offset 0
                  :dtype dtype)))))

(defun vt-from-array (arr &key (dtype nil))
  "从标准 CL 多维数组创建张量，保持维度不变。
   自动推断元素类型，支持通过 :dtype 强制覆盖。
   若发生类型不兼容转换 (如浮点强转整数)，将直接抛出类型错误快速失败。"
  (declare (type array arr) (optimize (safety 1)))
  (let* ((shape (array-dimensions arr))
         (size (reduce #'* shape :initial-value 1))
         (cl-etype (array-element-type arr))
         ;; 映射 CL 数组类型到内部 dtype
         (infer-dtype (cond
                        ((subtypep cl-etype 'double-float) :float64)
                        ((subtypep cl-etype 'single-float) :float32)
                        ((subtypep cl-etype '(signed-byte 32)) :int32)
                        ((subtypep cl-etype '(signed-byte 64)) :int64)
			((subtypep cl-etype '(unsigned-byte 8)) :int32)
			((subtypep cl-etype 'fixnum) :int64)
                        ;; 兜底未知或通用类型
                        (t :float64)))
         (final-dtype (or dtype infer-dtype))
         (lisp-type (vt-dtype->lisp-type final-dtype))
         (data (make-array size :element-type lisp-type))
         (strides (vt-compute-strides shape)))
    (declare (type fixnum size))
    ;; 逐元素拷贝并强制类型转换
    (dotimes (i size)
      (setf (aref data i)
            (vt-cast (row-major-aref arr i) final-dtype)))
    (%make-vt :data data
              :shape shape
              :strides strides
              :offset 0
              :dtype final-dtype)))

(defun vt-flatten-to-nested (dims data)
  "将按行主序存储的一维向量 data 转换为符合 dims 维度的嵌套列表.
   (vt-flatten-to-nested (vt-shape vt) (vt-data vt))"
  (with-float-safe
    (let ((total-size (reduce #'* dims))    ;; 总元素数
	  (idx 0))                          ;; 当前读取位置
      (assert (= total-size (length data)) (data)
              "数据长度与维度乘积不匹配")
      (labels
	  ((recurse (dims block-size)
             (if (null dims)    ;; 叶子节点
                 (prog1 (aref data idx) (incf idx))
                 (let* ((n (first dims))
                        (sub-dims (rest dims))
                        (sub-block-size
			  (if (zerop n)
			      0
			      (/ block-size n)))
                        result)
                   (dotimes (i n)    ;; 遍历当前维度的每个块
		     (declare (fixnum i)
			      (optimize (speed 3)))
                     (push (recurse sub-dims sub-block-size) result))
                   (nreverse result)))))   ;; 反转得到正确顺序
	(recurse dims total-size)))))

(defun vt-to-list (vt)
  "将张量转换为嵌套列表，正确处理任意 strides / offset 的视图。"
  (with-float-safe
    (labels
	((build (depth shape strides offset data)
           (declare (type fixnum depth offset)
                    (type list shape strides)
                    (type (simple-array * (*)) data))
           (if (null shape)
               ;; 叶子：直接取标量
               (aref data offset)
               ;; 内部节点：遍历当前维度并递归
               (let ((dim (first shape))
                     (stride (first strides))
                     (rest-shape (rest shape))
                     (rest-strides (rest strides))
                     (result nil))
                 (loop for i fixnum from (1- dim) downto 0
                       for sub-offset fixnum = (+ offset (* i stride))
                       do (push (build (1+ depth) rest-shape rest-strides
                                       sub-offset data)
                                result))
                 result))))
      (let ((shape (vt-shape vt))
            (strides (vt-strides vt))
            (offset (vt-offset vt))
            (data (vt-data vt)))
	(if shape
            (build 0 shape strides offset data)
            ;; 0 维张量（标量）直接返回值
            (aref data offset))))))

(defun vt-to-array (vt &key dtype)
  "将张量转换为原生多维数组。零中间 List 分配，直接进行内存级搬运。
   - vt: 输入张量。
   - dtype: 指定输出数组的元素类型。
            如果提供，张量会先转换为该类型，再转换为数组。"
  (declare (type vt vt) (optimize (safety 0)))
  (if dtype (setf vt (vt-astype vt dtype)))
  (let ((shape (vt-shape vt)))
    (macrolet
	((gen-to-array (lisp-type)
           `(let ((arr (make-array shape :element-type ',lisp-type))
                  (in-data (the (simple-array ,lisp-type (*)) (vt-data vt)))
                  (out-strides (vt-compute-strides shape)))
	      (declare (type (simple-array ,lisp-type) arr)           
                       (type (simple-array ,lisp-type (*)) in-data))
              (labels ((recurse (depth in-ptr out-ptr)
                         (declare (type fixnum depth in-ptr out-ptr))
                         (if (= depth (length shape))
                             (setf (row-major-aref arr out-ptr)
                                   (aref in-data in-ptr))
                             (let ((dim (the fixnum (nth depth shape)))
                                   (in-stride (the fixnum (nth depth (vt-strides vt))))
                                   (out-stride (the fixnum (nth depth out-strides)))
                                   (cur-in in-ptr)
                                   (cur-out out-ptr))
                               (declare (type fixnum dim in-stride out-stride cur-in cur-out))
                               (loop for i fixnum from 0 below dim do
                                 (recurse (1+ depth) cur-in cur-out)
                                 (incf cur-in in-stride)
                                 (incf cur-out out-stride))))))
                (recurse 0 (the fixnum (vt-offset vt)) 0))
              arr)))
      (etypecase (vt-data vt)
        ((simple-array double-float (*))
         (gen-to-array double-float))
        ((simple-array single-float (*))
         (gen-to-array single-float))
        ((simple-array (signed-byte 64) (*))
         (gen-to-array (signed-byte 64)))
        ((simple-array (signed-byte 32) (*))
         (gen-to-array (signed-byte 32)))
	((simple-array (unsigned-byte 8) (*))
         (gen-to-array (unsigned-byte 8)))
        (simple-vector
         (let ((arr (make-array shape :element-type t))
               (in-data (vt-data vt))
               (out-strides (vt-compute-strides shape)))
           (declare (type simple-vector in-data))
           (labels ((recurse (depth in-ptr out-ptr)
                      (declare (type fixnum depth in-ptr out-ptr))
                      (if (= depth (length shape))
                          (setf (row-major-aref arr out-ptr)
                                (aref in-data in-ptr))
                          (let ((dim (the fixnum (nth depth shape)))
                                (in-stride (the fixnum (nth depth (vt-strides vt))))
                                (out-stride (the fixnum (nth depth out-strides)))
                                (cur-in in-ptr)
                                (cur-out out-ptr))
                            (declare (type fixnum dim in-stride out-stride cur-in cur-out))
                            (loop for i fixnum from 0 below dim do
                              (recurse (1+ depth) cur-in cur-out)
                              (incf cur-in in-stride)
                              (incf cur-out out-stride))))))
             (recurse 0 (the fixnum (vt-offset vt)) 0))
           arr))))))



(defun vt-arange (n &key (start 0) (step 1) (dtype :float64))
  "创建一个包含范围值的一维张量."
  (declare (fixnum n))
  (let* ((data (make-array n :element-type (vt-dtype->lisp-type dtype)))
         (shape (list n)))
    (loop for i fixnum below n
          do (setf (aref data i)
		   (vt-cast (+ start (* i step)) dtype)))
    (%make-vt :data data
	      :shape shape
	      :strides (vt-compute-strides shape)
	      :dtype dtype)))

(defun vt-zeros (shape &key (dtype :float64))
  "创建全0张量."
  (declare (list shape))
  (make-vt shape 0 :dtype dtype))

(defun vt-ones (shape &key (dtype :float64))
  "创建全1张量."
  (declare (list shape))
  (make-vt shape 1 :dtype dtype))

(defun vt-const (shape value &key (dtype :float64))
  (declare (list shape))
  (make-vt shape value :dtype dtype))

(defun vt-full (shape fill-value &key (dtype :float64))
  "创建指定值填充的数组.
  shape: 形状列表
  fill-value: 填充值
  返回: 张量"
  (with-float-safe
    (make-vt shape fill-value :dtype dtype)))

(defun vt-empty (shape &key (dtype :float64))
  "创建未初始化数组(实际用 0 填充,避免垃圾数据).
  shape: 形状列表
  返回: 张量"
  (vt-zeros shape :dtype dtype))

(defun vt-zeros-like (vt &key dtype)
  "创建与给定数组形状相同的全零数组.
  vt: 输入张量
  dtype: 数据类型(默认与输入相同)
  返回: 张量"
  (vt-zeros (vt-shape vt)
	    :dtype (or dtype (vt-dtype vt))))

(defun vt-ones-like (vt &key dtype)
  "创建与给定数组形状相同的全一数组."
  (vt-ones (vt-shape vt)
	   :dtype (or dtype (vt-dtype vt))))

(defun vt-full-like (vt fill-value &key dtype)
  "创建与给定数组形状相同的填充数组."
  (with-float-safe
    (vt-full (vt-shape vt)
	     fill-value
	     :dtype (or dtype (vt-dtype vt)))))

(defun vt-empty-like (vt &key dtype)
  "创建与给定数组形状相同的未初始化数组."
  (vt-empty (vt-shape vt)
	    :dtype (or dtype (vt-dtype vt))))


(defun vt-identity (n &key dtype)
  "创建 n×n 单位矩阵。"
  (vt-eye n :dtype dtype))

(defun vt-eye (rows &key (cols rows) (value 1) (dtype :float64))
  "创建单位矩阵或矩形对角矩阵。
   rows: 行数。
   cols: 列数 (默认为 rows)。
   value: 对角线填充值 (默认 1)。"
  (declare (type fixnum rows cols))
  (let* ((shape (list rows cols))
         (lisp-type (vt-dtype->lisp-type dtype))
         ;; 直接分配底层连续物理数组，避免 vt-zeros 二次解析
         (data (make-array (* rows cols)
                           :element-type lisp-type
                           :initial-element (coerce 0 lisp-type)))
         (strides (vt-compute-strides shape))
         (res (%make-vt :data data
                        :shape shape
                        :strides strides
                        :offset 0
                        :dtype dtype
                        )))
    (declare (type (simple-array * (*)) data)
             (type list strides))
    ;; 对于行优先，data[i, i] 的偏移是 i * row_stride + i * col_stride
    (let ((row-stride (first strides))
          (col-stride (second strides))
          (diagonal-len (min rows cols)))
      (declare (type fixnum row-stride col-stride diagonal-len))
      ;; 沿对角线极速填充
      (loop for i fixnum from 0 below diagonal-len
            for offset fixnum = 0 then (+ offset row-stride col-stride)
            do (setf (aref data offset)
                     (coerce value lisp-type))))
    res))

(defun vt-diag (tensor &key (k 0))
  "提取对角线或构造对角矩阵。严格对标 NumPy 的 np.diag。
  - 输入 1D (n): 返回 2D (n+|k|, n+|k|) 对角矩阵，k 指定偏移。
  - 输入 2D (m, n): 返回 1D (min(m, n)-|k|) 对角线向量，k 指定偏移。
  k: 对角线偏移量，正数表示主对角线之上，负数表示之下。默认为 0。
  "
  (declare (type fixnum k))
  (let* ((shape (vt-shape tensor))
         (rank (length shape)))
    (cond
      ;; === 1D -> 2D: 构造对角矩阵 ===
      ((= rank 1)
       (let* ((n (the fixnum (first shape)))
              (dim (the fixnum (+ n (abs k)))) ; 矩阵维度需扩大以容纳偏移
              (dtype (vt-dtype tensor))
              (res (vt-zeros (list dim dim) :dtype dtype))
              (res-data (vt-data res))
              (res-strides (vt-strides res))
              (in-data (vt-data tensor))
              (in-offset (vt-offset tensor))
              (in-stride (first (vt-strides tensor))))
         (declare (type fixnum dim n in-offset in-stride))
         (let ((row-stride (first res-strides))
               (col-stride (second res-strides)))
           (declare (type fixnum row-stride col-stride))
           ;; 计算偏移后的起始物理地址
           ;; k > 0: 右上方，起点在第 0 行，第 k 列
           ;; k < 0: 左下方，起点在第 |k| 行，第 0 列
           (let ((start-offset (the fixnum 
                                    (if (> k 0) 
                                        (* k col-stride) 
                                        (* (- k) row-stride)))))
             (declare (type fixnum start-offset))
             ;; 基于步长极速填充对角线
             (loop for i fixnum from 0 below n
                   for src-off fixnum = (+ in-offset (* i in-stride))
                   for dst-off fixnum = start-offset
		     then (+ dst-off row-stride col-stride)
                   do (setf (aref res-data dst-off)
			    (aref in-data src-off))))
           res)))
      
      ;; === 2D -> 1D: 提取对角线 ===
      ((= rank 2)
       ;; 复用已有的高效提取函数 vt-diagonal
       (vt-diagonal tensor :offset k))
      
      ;; === 其他维度报错 ===
      (t (error "vt-diag 仅支持 1D 或 2D 张量输入，当前维度为: ~a" rank)))))


(defun vt-triu (tensor &key (k 0))
  "返回上三角矩阵。
   k: 对角线偏移 (0=主对角线, 1=主对角线之上)。
   支持 batch (rank >= 2)。"
  (let* ((res (vt-copy tensor))
         (res-data (vt-data res))
         (rank (length (vt-shape res)))
         ;; 预提取为 simple-vector，消灭循环内的 nth 开销
         (shape-vec (coerce (vt-shape res) 'simple-vector))
         (strs-vec (coerce (vt-strides res) 'simple-vector))
         ;; 预先将 0 转换为目标类型，避免循环内反复 coerce
         (zero-val (coerce 0 (vt-element-type res))))
    (declare (type simple-vector shape-vec strs-vec)
             (type fixnum rank))
    (when (< rank 2)
      (error "vt-triu requires rank >= 2"))
    (labels
        ((recurse (depth ptr)
           (declare (type fixnum depth ptr))
           (if (= depth (- rank 2))
               ;; === 底层矩阵操作 ===
               (let* ((rows (svref shape-vec depth))
                      (cols (svref shape-vec (1+ depth)))
                      (str-r (svref strs-vec depth))
                      (str-c (svref strs-vec (1+ depth))))
                 (declare (type fixnum rows cols str-r str-c))
                 (loop for r fixnum from 0 below rows
                       for row-ptr fixnum = ptr then (+ row-ptr str-r) do
                         (loop for c fixnum from 0 below cols do
                           ;; 列号 >= 行号 + k
                           (when (< c (+ r k))
                             (setf (aref res-data
                                         (+ row-ptr (* c str-c)))
                                   zero-val)))))
               ;; === 高维递归 ===
               (let ((dim (svref shape-vec depth))
                     (stride (svref strs-vec depth)))
                 (declare (type fixnum dim stride))
                 (loop for i fixnum from 0 below dim do
                   (recurse (1+ depth) ptr)
                   (incf ptr stride))))))
      (recurse 0 (vt-offset res)))
    res))

(defun vt-tril (tensor &key (k 0))
  "返回下三角矩阵。
   k: 对角线偏移 (0=主对角线, -1=主对角线之下)。
   支持 batch。"
  (let* ((res (vt-copy tensor))
         (res-data (vt-data res))
         (rank (length (vt-shape res)))
         (shape-vec (coerce (vt-shape res) 'simple-vector))
         (strs-vec (coerce (vt-strides res) 'simple-vector))
         (zero-val (coerce 0 (vt-element-type res))))
    (declare (type simple-vector shape-vec strs-vec)
             (type fixnum rank))
    (when (< rank 2)
      (error "vt-tril requires rank >= 2"))
    (labels
        ((recurse (depth ptr)
           (declare (type fixnum depth ptr))
           (if (= depth (- rank 2))
               ;; === 底层矩阵操作 ===
               (let* ((rows (svref shape-vec depth))
                      (cols (svref shape-vec (1+ depth)))
                      (str-r (svref strs-vec depth))
                      (str-c (svref strs-vec (1+ depth))))
                 (declare (type fixnum rows cols str-r str-c))
                 (loop for r fixnum from 0 below rows
                       for row-ptr fixnum = ptr then (+ row-ptr str-r) do
                         (loop for c fixnum from 0 below cols do
                           ;; 列号 <= 行号 + k
                           (when (> c (+ r k))
                             (setf (aref res-data
                                         (+ row-ptr (* c str-c)))
                                   zero-val)))))
               ;; === 高维递归 ===
               (let ((dim (svref shape-vec depth))
                     (stride (svref strs-vec depth)))
                 (declare (type fixnum dim stride))
                 (loop for i fixnum from 0 below dim do
                   (recurse (1+ depth) ptr)
                   (incf ptr stride))))))
      (recurse 0 (vt-offset res)))
    res))

(defun vt-diagonal (tensor &key (offset 0))
  "提取对角线元素。
   返回一个 1d 张量 (向量), 如果是 batch 输入则返回 2d 张量。
   offset: 对角线偏移。"
  (let* ((in-shape (vt-shape tensor))
         (rank (length in-shape)))
    (declare (type fixnum rank))
    (when (< rank 2)
      (error "vt-diagonal requires rank >= 2"))
    (let* ((rows (nth (- rank 2) in-shape))
           (cols (nth (- rank 1) in-shape))
           (diag-len (max 0 (- (min rows cols) (abs offset))))
           (dtype (vt-dtype tensor))
           (out-shape (append (subseq in-shape 0 (- rank 2))
                              (list diag-len)))
           (res (vt-zeros out-shape :dtype dtype))
           (res-data (vt-data res))
           (in-data (vt-data tensor))
           (in-strs (coerce (vt-strides tensor) 'simple-vector))
           (out-strs (coerce (vt-strides res) 'simple-vector))
           (in-shp-vec (coerce in-shape 'simple-vector))
           ;; 预计算起点偏移量
           (r-init (if (> offset 0) 0 (- offset)))
           (c-init (if (> offset 0) offset 0)))
      (declare (type simple-vector in-strs out-strs in-shp-vec)
               (type fixnum r-init c-init))
      (labels
          ((recurse (depth in-ptr out-ptr)
             (declare (type fixnum depth in-ptr out-ptr))
             (if (= depth (- rank 2))
                 ;; === 底层: 提取对角线 ===
                 (let ((str-r (svref in-strs depth))
                       (str-c (svref in-strs (1+ depth))))
                   (declare (type fixnum str-r str-c))
                   (loop for i fixnum from 0 below diag-len
                         for src-off fixnum = (+ in-ptr
                                                 (* (+ r-init i) str-r)
                                                 (* (+ c-init i) str-c))
                         do (setf (aref res-data out-ptr)
                                  (aref in-data src-off))
                            (incf out-ptr)))
                 ;; === 高维递归 ===
                 (let ((dim (svref in-shp-vec depth))
                       (in-str (svref in-strs depth))
                       (out-str (svref out-strs depth)))
                   (declare (type fixnum dim in-str out-str))
                   (loop for i fixnum from 0 below dim do
                     (recurse (1+ depth) in-ptr out-ptr)
                     (incf in-ptr in-str)
                     (incf out-ptr out-str))))))
        (recurse 0 (vt-offset tensor) 0))
      res)))


;; 1. 选用正确的步长计算函数
(defun vt-compute-strides (shape)
  "根据形状计算连续内存步长.
   shape 为 nil (标量) -> 返回 nil.
   shape 为 (10) -> 返回 (1)."
  (if (null shape)
      nil
      (loop with stride = 1
            for dim in (reverse shape)
            collect stride into strides
            do (setf stride (* stride dim))
            finally (return (reverse strides)))))

(defun vt-contiguous-p (vt)
  "检查张量是否在内存中连续 只有连续的张量才能安全地重塑为任意形状.
   (对标 numpy 的 c_contiguous 判定逻辑)"
  (let ((shape (vt-shape vt))
        (strides (vt-strides vt)))
    (if (null shape)
        t ; 0 维张量（标量）必然连续
        ;; 零尺寸张量一律视为连续 (对标 numpy 行为)
        (if (some #'zerop shape)
            t
            (let ((expected-stride 1)
                  (contiguous t))
              (declare (type fixnum expected-stride))
              (loop for i fixnum from (1- (length shape)) downto 0
                    for dim fixnum = (the fixnum (nth i shape))
                    for stride fixnum = (the fixnum (nth i strides))
                    do (cond
                         ((= dim 1)
                          nil) ; 维度大小为 1，步长无意义，跳过
                         ((= stride expected-stride)
                          (setf expected-stride
				(the fixnum (* expected-stride dim))))
                         (t
                          (setf contiguous nil))))
              contiguous)))))

(defun vt-contiguous (vt)
  "返回一个内存连续的副本.如果原张量已连续,可能返回自身或副本.
   这是视图操作后的“落地”操作."
  (if (vt-contiguous-p vt)
      vt
      (let* ((new-size (vt-shape-to-size (vt-shape vt)))
             (new-data (make-array new-size 
				   :element-type (vt-element-type vt)))
             (new-vt (%make-vt
                      :data new-data
                      :shape (vt-shape vt)
                      :strides (vt-compute-strides (vt-shape vt))
                      :offset 0
		      :dtype (vt-dtype vt))))
        ;; 高效拷贝:利用迭代器填充新数据
        (vt-copy-into new-vt vt)
        new-vt)))

(defun vt-view (vt new-shape)
  "零拷贝重塑视图。对标 pytorch 的 tensor.view()。
  要求输入张量必须是内存连续的，否则报错。
  用于确保操作不会产生隐式的数据拷贝。"
  
  (let ((neg-idx (position -1 new-shape)))
    (when neg-idx
      (let ((known-size (reduce #'* (remove -1 new-shape) :initial-value 1))
            (total-size (vt-shape-to-size (vt-shape vt))))
	(cond
          ;; 情况 1: 已知维度乘积为 0
          ((zerop known-size)
           ;; 如果总元素也是 0，将 -1 替换为 0 (如 (0,3) -> (-1, 0) => (0, 0))
           (if (zerop total-size)
               (setf new-shape (substitute 0 -1 new-shape))
               ;; 如果总元素非 0 却要求已知维度乘积为 0，则是非法请求
               (error "无法将形状 ~A 重塑为含 -1 的 ~A" (vt-shape vt) new-shape)))
          ;; 情况 2: 无法整除
          ((not (zerop (rem total-size known-size)))
           (error "无法将形状 ~A 重塑为含 -1 的 ~A" (vt-shape vt) new-shape))
          ;; 情况 3: 正常计算
          (t
           (setf new-shape
		 (substitute (/ total-size known-size) -1 new-shape)))))))


  (let ((old-size (vt-shape-to-size (vt-shape vt)))
        (new-size (vt-shape-to-size new-shape)))
    (unless (= old-size new-size)
      (error "view 失败: 元素总数不一致 (旧: ~a, 新: ~a)" old-size new-size))
    (unless (vt-contiguous-p vt)
      (error "view 失败: 张量内存不连续! 请先调用 或使用 柔性重塑。"))
    (%make-vt :data (vt-data vt)
              :shape new-shape
              :strides (vt-compute-strides new-shape)
              :offset (vt-offset vt)
              :dtype (vt-dtype vt))))

(defun vt-reshape (vt new-shape)
  "重塑形状.
   如果张量是连续的,则创建视图(零拷贝)；
   否则自动创建连续副本后重塑.
   这解决了原代码在非连续内存上重塑导致的数据错误问题."
  (let ((neg-idx (position -1 new-shape)))
    (when neg-idx
      (let ((known-size (reduce #'* (remove -1 new-shape) :initial-value 1))
            (total-size (vt-shape-to-size (vt-shape vt))))
	(cond
          ;; 情况 1: 已知维度乘积为 0
          ((zerop known-size)
           ;; 如果总元素也是 0，将 -1 替换为 0 (如 (0,3) -> (-1, 0) => (0, 0))
           (if (zerop total-size)
               (setf new-shape (substitute 0 -1 new-shape))
               ;; 如果总元素非 0 却要求已知维度乘积为 0，则是非法请求
               (error "无法将形状 ~A 重塑为含 -1 的 ~A" (vt-shape vt) new-shape)))
          ;; 情况 2: 无法整除
          ((not (zerop (rem total-size known-size)))
           (error "无法将形状 ~A 重塑为含 -1 的 ~A" (vt-shape vt) new-shape))
          ;; 情况 3: 正常计算
          (t
           (setf new-shape
		 (substitute (/ total-size known-size) -1 new-shape)))))))

  (let ((old-size (vt-shape-to-size (vt-shape vt)))
        (new-size (vt-shape-to-size new-shape)))
    (unless (= old-size new-size)
      (error "重塑失败:元素总数不一致 (旧: ~a, 新: ~a)" old-size new-size))    
    (if (vt-contiguous-p vt)
        ;; 零拷贝路径:直接复用数据
        (%make-vt
         :data (vt-data vt)
         :shape new-shape
         :strides (vt-compute-strides new-shape)
         :offset (vt-offset vt)
	 :dtype (vt-dtype vt))
        ;; 安全路径:先内存整理,再重塑
        (let ((cont-vt (vt-contiguous vt)))
          (%make-vt
           :data (vt-data cont-vt)
           :shape new-shape
           :strides (vt-compute-strides new-shape)
           :offset 0
	   :dtype (vt-dtype cont-vt))))))

(defun vt-transpose (vt &optional (perm nil))
  "零拷贝转置.仅交换步长和形状维度,不移动数据.
   perm: 排列列表,例如 (1 0) 表示交换维度0和1.
   如果不提供 perm，默认反转所有轴 (对标 numpy 的 np.transpose 默认行为)。"
  (let* ((shape (vt-shape vt))
         (strides (vt-strides vt))
         (rank (length shape)))    
    ;; 1. 若未提供 perm，默认反转所有轴
    (unless perm
      (setf perm (loop for i from (1- rank) downto 0 collect i)))
    ;; 2. 检查长度是否匹配
    (unless (= (length perm) rank)
      (error "perm 的长度 ~a 必须等于张量维度数 ~a" (length perm) rank))
    ;; 3. 检查范围合法性与去重
    (let ((seen (make-array rank :element-type 'bit :initial-element 0)))
      (dolist (p perm)
        (unless (and (integerp p) (>= p 0) (< p rank))
          (error "perm 中的值 ~a 超出合法范围 [0, ~a]" p (1- rank)))
        (when (= (aref seen p) 1)
          (error "perm 中存在重复的轴索引: ~a" p))
        (setf (aref seen p) 1)))    
    ;; 4. 执行转置逻辑 (零拷贝视图操作)
    (let ((new-shape (mapcar #'(lambda (p) (nth p shape)) perm))
          (new-strides (mapcar #'(lambda (p) (nth p strides)) perm)))
      (%make-vt :data (vt-data vt)
                :shape new-shape
                :strides new-strides
                :offset (vt-offset vt)
                :dtype (vt-dtype vt)))))

(defun vt-normalize-axis (axis rank)
  "将可能为负数的 axis 转换为严格正数，并进行越界检查。
   例如：rank=4, axis=-1 -> 3 ; axis=-2 -> 2"
  (if axis
      (let ((ax (if (minusp axis)
		    (+ axis rank)
		    axis)))
	(when (or (< ax 0) (>= ax rank))
	  (error "axis ~a is out of bounds for tensor of rank ~a"
		 axis rank))
	ax)))

(defun vt-narrow (vt axis start end)
  "零拷贝切片.沿指定轴切片,调整偏移量和形状. (等价于 pytorch 的 narrow)"
  (with-float-safe
    (let* ((shape (copy-list (vt-shape vt)))
           (rank (length shape))
           (ax (vt-normalize-axis axis rank)) 
           (strides (vt-strides vt))
           (dim-size (nth ax shape)))    
      (when (or (< start 0) (> end dim-size))
	(error "切片索引 [~a, ~a) 越界，当前轴大小为 ~a"
	       start end dim-size))
      (when (< end start)
	(error "vt-narrow: end (~a) must be greater than start (~a)"
	       end start))
      (let ((new-offset (+ (vt-offset vt)
                           (* start (nth ax strides))))
            (new-shape (progn (setf (nth ax shape) (- end start)) shape)))
	(%make-vt
	 :data (vt-data vt)
	 :shape new-shape
	 :strides strides 
	 :offset new-offset
	 :dtype (vt-dtype vt))))))

(defun vt-split (tensor indices-or-sections &key (axis 0))
  "沿指定轴分割张量 (对标 numpy 的 array_split)。
  indices-or-sections:
  - 整数 n: 分割成 n 个块。如果无法整除，前 (dim % n) 块会比后面的块多 1 个元素。
  - 列表: 按照列表中的索引位置进行切分。
  axis: 分割轴，支持负数。
  返回: 张量列表"
  (with-float-safe
    (let* ((shape (vt-shape tensor))
           (rank (length shape))
           (ax (vt-normalize-axis axis rank))
           (dim-size (nth ax shape)))
      (cond
        ;; ========== 情况 a：分成 n 个块 (允许不均等) ==========
        ((integerp indices-or-sections)
         (let ((n indices-or-sections))
           (when (<= n 0)
             (error "分割块数 n 必须是正整数: ~a" n))
           ;; 如果 n >= dim-size，退化为每块最多1个元素
           (let ((base (floor dim-size n))
                 (rem (rem dim-size n))
                 (cur-start 0)
                 (result nil))
             (declare (type fixnum base rem cur-start))
             (dotimes (i n (nreverse result))
               ;; 前 rem 块大小为 base + 1，其余为 base
               (let* ((chunk (the fixnum (if (< i rem) (1+ base) base)))
                      (end (the fixnum (+ cur-start chunk))))
                 (push (vt-narrow tensor ax cur-start end) result)
                 (setf cur-start end))))))        
        ;; ========== 情况 b：指定位置下刀 ==========
        ((listp indices-or-sections)
         (let* ((raw-points
		  (append (list 0) indices-or-sections (list dim-size)))
		(normalized-points (mapcar (lambda (p)
					     (if (minusp p) (+ p dim-size) p))
					   raw-points))
		(clamped-points (mapcar (lambda (p)
					  (max 0 (min p dim-size)))
					normalized-points)))
           (loop for (start end) on clamped-points by #'cdr
                 while end
                 collect (vt-narrow tensor ax start end))))
        (t (error "indices-or-sections 必须是整数或整数列表"))))))


(defun vt-slice (vt &rest specs)
  "通用切片函数（统一列表接口）。
   每个 spec 必须是一个列表：
     - (:all) 或 (t)   : 保留整个维度
     - (:newa)         : 插入一个长度为1的新轴 newaxis
     - (:elli)         : 省略号 ellipsis（最多一个），自动展开为多个 :all
     - (idx)           : 整数索引，降维
     - (start end &optional step) : 范围切片，start/end 可用 nil 省略
   自动补齐尾部缺失的 :all；支持负索引与负步长。
   返回零拷贝视图。
编号  numpy 表达式          含义                        新 vt-slice 写法                                   输出形状
1     b[1, 2]              单个元素                   (vt-slice b '(1) '(2))                             标量 (降维)
2     b[1, :]              第 1 行                    (vt-slice b '(1) '(:all))                          (5)
3     b[:, 2]              第 2 列                    (vt-slice b '(:all) '(2))                          (4)
4     b[0:2, 1:4]          子矩阵                     (vt-slice b '(0 2) '(1 4))                         (2,3)
5     b[:2, 2:]            前两行，第 2 列起           (vt-slice b '(nil 2) '(2 nil))                     (2,3)
6     b[:, :]              整个矩阵                   (vt-slice b '(:all) '(:all)) 或 (vt-slice b)       (4,5)
7     b[::-1, :]           行逆序                     (vt-slice b '(nil nil -1) '(:all))                 (4,5)
8     b[:, ::-1]           列逆序                     (vt-slice b '(:all) '(nil nil -1))                 (4,5)
9     b[-1, :]             最后一行                   (vt-slice b '(-1) '(:all))                         (5)
10    b[:, -2:]            最后两列                   (vt-slice b '(:all) '(-2 nil))                     (4,2)
11    b[1, -2]             固定行列 (降维)             (vt-slice b '(1) '(-2))                            标量
12    b[1:3, :]            行切片                     (vt-slice b '(1 3) '(:all))                        (2,5)
13    b[1:, :-2]           第 1 行起，不含最后两列      (vt-slice b '(1 nil) '(nil -2))                    (3,3)
14    b[::2, 1::2]         隔行，第 1 列隔列取         (vt-slice b '(nil nil 2) '(1 nil 2))               (2,2)
15    b[none,:,none,0]     插入新轴并选取              (vt-slice b '(:newa) '(:all) '(:newa) '(0))        (1,4,1)
16    b[..., :2]           省略号，切最后一维前两列     (vt-slice b '(:elli) '(nil 2))                     (4,2)
17    b[:, -1]             最后一列 (降维)             (vt-slice b '(:all) '(-1))                         (4)
18    b[:, [0,2]]          花式索引 (暂未支持)         可考虑用 vt-take                                   —
"
  (with-float-safe
    (let* ((old-shape (vt-shape vt))
           (old-strides (vt-strides vt))
           (old-offset (vt-offset vt))
           (old-rank (length old-shape))
           (expanded-specs '()))

      ;; ========== 1. 展开省略号，确保原轴数正确 ==========
      (let* ((ellipsis-pos (position '(:elli) specs :test #'equal))
             (num-newaxis (count '(:newa) specs :test #'equal))
             (explicit-original (- (length specs) (if ellipsis-pos 1 0) num-newaxis)))
	(when (> explicit-original old-rank)
          (error "too many indices for array of rank ~d" old-rank))
	(let ((needed-all (- old-rank explicit-original)))
          (dolist (s specs)
            (if (equal s '(:elli))
		(dotimes (i needed-all)
                  (push '(:all) expanded-specs))
		(push s expanded-specs)))
	  (unless ellipsis-pos
            (let ((remaining (- old-rank (- (length specs) num-newaxis))))
              (dotimes (i remaining)
		(push '(:all) expanded-specs))))                     
          (setf expanded-specs (nreverse expanded-specs))))
      
      ;; ========== 2. 遍历每个规格 ==========
      (let ((new-shape '())
            (new-strides '())
            (cur-offset old-offset)
            (axis-idx 0))
	(dolist (spec expanded-specs)
          (unless (listp spec)
            (error "invalid slice spec: ~a, expected a list" spec))
          (cond
            ;; --- 新轴 ---
            ((equal spec '(:newa))
             (push 1 new-shape)
             (push 0 new-strides))

            ;; --- 全选（:all 或 t） ---
            ((or (equal spec '(:all)) (equal spec '(t)))
             (when (>= axis-idx old-rank)
               (error "axis index out of bounds"))
             (let ((dim (nth axis-idx old-shape))
                   (stride (nth axis-idx old-strides)))
               (push dim new-shape)
               (push stride new-strides)
               (incf axis-idx)))

            ;; --- 范围切片 (start end &optional step) ---
            ((and (listp spec) (<= 2 (length spec) 3))
             (destructuring-bind (start end &optional (step 1)) spec
               (when (>= axis-idx old-rank)
		 (error "axis index out of bounds"))
               (let ((dim (nth axis-idx old-shape))
                     (stride (nth axis-idx old-strides)))
		 (when (and (numberp start) (< start 0))
                   (incf start dim))
		 (when (and (numberp end)   (< end 0))
                   (incf end dim))
		 (when (zerop step)
                   (error "slice step cannot be zero"))
		 (cond
                   ((> step 0)
                    (setf start (if (null start) 0
                                    (max 0 (min start dim))))
                    (setf end   (if (null end) dim
                                    (max 0 (min end dim)))))
                   ((< step 0)
                    (setf start (if (null start) (1- dim)
                                    (max -1 (min start (1- dim)))))
                    (setf end   (if (null end) -1
                                    (max -1 (min end (1- dim)))))))
		 (let ((slice-dim 0))
                   (cond
                     ((> step 0)
                      (when (> end start)
			(setf slice-dim (ceiling (- end start) step))))
                     ((< step 0)
                      (when (> start end)
			(setf slice-dim (ceiling (- start end) (- step))))))
                   (incf cur-offset (* start stride))
                   (push slice-dim new-shape)
                   (push (* stride step) new-strides)))
               (incf axis-idx)))

            ;; --- 整数索引 (降维) ---
            ((and (listp spec) (= (length spec) 1) (integerp (first spec)))
             (let ((idx (first spec)))
               (when (>= axis-idx old-rank)
		 (error "axis index out of bounds"))
               (let ((dim (nth axis-idx old-shape))
                     (stride (nth axis-idx old-strides)))
		 (when (< idx 0) (incf idx dim))
		 (unless (and (>= idx 0) (< idx dim))
                   (error "index ~d out of bounds for axis ~d (size ~d)"
			  idx axis-idx dim))
		 (incf cur-offset (* idx stride))
		 (incf axis-idx))))

            ;; --- 其他情况报错 ---
            (t (error "invalid slice spec: ~a" spec))))
	(%make-vt :data (vt-data vt)
                  :shape (nreverse new-shape)
                  :strides (nreverse new-strides)
                  :offset cur-offset
		  :dtype (vt-dtype vt))))))

(defun (setf vt-slice) (value vt &rest specs)
  "设置切片区域的值。value 可以是数字或张量。
   specs 遵循新 vt-slice 的统一列表接口。
   示例:
     (let ((m (vt-ones '(4 5) :type 'fixnum)))
     ;; 将第一行全部置 0
     (setf (vt-slice m '(0) '(:all)) 8)
     ;; 将子矩阵赋值为另一个张量
     (setf (vt-slice m '(0 2) '(0 2)) (vt-slice m '(2 4) '(2 4))))"
  (with-float-safe
    (let ((target-view (apply #'vt-slice vt specs)))
      (if (null (vt-shape target-view))
          ;; 目标为标量：直接将 value 转换为标量数值并写入
          (let ((scalar-value
                  (cond ((numberp value) value)
			((vt-p value)
			 (if (null (vt-shape value))
                             (aref (vt-data value) (vt-offset value))
                             (error "cannot assign tensor of shape ~a to scalar slice"
                                    (vt-shape value))))
			(t (error "invalid value ~a for scalar slice" value)))))
            (setf (vt-ref target-view) scalar-value))
          ;; 目标为非标量：使用通用拷贝（支持广播）
          (vt-copy-into target-view value)))))

(defun vt-astype (tensor new-dtype)
  "将张量转换为新类型。浮点数→整数时使用截断 (truncate)，其余使用 coerce。"
  (with-float-safe
    (let* ((shape (vt-shape tensor))
           (new (vt-zeros shape :dtype new-dtype))
           (new-data (vt-data new))
           (offset (vt-offset new))
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
                       (loop for i below dim do
			 (copy-rec (1+ depth) in-ptr out-ptr)
			 (incf in-ptr in-stride)
			 (incf out-ptr out-stride))))))
	(copy-rec 0 in-offset offset))
      new)))

(defun vt-copy (vt &key dtype)
  "深度拷贝张量.
   返回一个全新的、内存连续的张量副本.
   副本与原张量完全独立,修改副本不会影响原张量.
   自动处理视图:如果原张量是切片或转置,返回的是落地后的连续数据.
   Parameters:
   - dtype: (可选) 指定新张量的数据类型。如果与原张量不同，将进行类型转换。"
  (with-float-safe
    (let* ((target-dtype (or dtype (vt-dtype vt))))
      (declare (type (member :int32 :int64 :float32 :float64) target-dtype))
      
      (if (eq target-dtype (vt-dtype vt))
          ;; === 场景 1: 类型未变 (包括未指定 dtype) ===
          (let* ((shape (vt-shape vt))
                 (src-data (vt-data vt))
                 (element-type (vt-element-type vt))
                 (size (vt-shape-to-size shape))
                 (new-data (make-array size :element-type element-type))
                 (new-vt (%make-vt :data new-data
                                   :shape (copy-list shape)
                                   :strides (vt-compute-strides shape)
                                   :offset 0
                                   :dtype target-dtype)))
            (declare (type (simple-array *) src-data new-data)
                     (type fixnum size))
            
            (if (vt-contiguous-p vt)
                ;; 快速路径: 内存块拷贝
                (replace new-data src-data
                         :start2 (vt-offset vt)
                         :end2 (+ (vt-offset vt) size))
                ;; 通用路径: 处理非连续视图
                (vt-copy-into new-vt vt))
            new-vt)
          ;; === 场景 2: 需要类型转换 ===
          (vt-astype vt target-dtype)))))

(defun vt-copy-into (dest src)
  "将 src 的数据拷贝到 dest (支持广播和类型转换).
   极致优化:连续内存走 replace (memcpy),非连续走展开循环。"
  (setf src (ensure-vt src))  
  (let ((dest-shape (vt-shape dest))
        (src-shape (vt-shape src)))     
    ;; 1. 安全检查：防止写入 0 步长的广播视图
    (loop for d in dest-shape
          for s in (vt-strides dest)
          when (and (> d 1) (zerop s))
            do (error "vt-copy-into: 目标视图是只读的广播视图 (含有大小为 ~a 的广播维度)" d))            
    (let ((final-shape (vt-broadcast-shapes dest-shape src-shape)))
      (unless (equal final-shape dest-shape)
        (error "copy-into 失败: dest 形状 ~a 无法容纳 src 广播后的形状 ~a"
               dest-shape final-shape))               
      (let* ((dest-data (vt-data dest))
             (src-data (vt-data src))
             (dest-etype (array-element-type dest-data))
             (src-etype (array-element-type src-data))
             ;; 预计算广播后的源步长 (不分配新张量,零开销)
             (src-strides
	       (vt-broadcast-strides src-shape dest-shape (vt-strides src)))
             (size (vt-shape-to-size dest-shape)))        
        (cond
          ;; 极速路径: 完全连续 且 类型一致
          ;; 直接调用底层内存块拷贝 (等价于 C 的 memcpy)
          ((and (vt-contiguous-p dest)
		(vt-contiguous-p src)
		(equal dest-shape src-shape)
		(equal dest-etype src-etype))
           (replace dest-data src-data
                    :start1 (vt-offset dest)
                    :end1 (+ (vt-offset dest) size)
                    :start2 (vt-offset src)
                    :end2 (+ (vt-offset src) size)))
          ;; 中速路径: 连续内存 但类型不同 (单层线性循环)
          ((and (vt-contiguous-p dest)
		(vt-contiguous-p src)
		(equal dest-shape src-shape))
           (let ((d-off (vt-offset dest))
                 (s-off (vt-offset src)))
             (declare (type fixnum d-off s-off size))
             (loop for i fixnum from 0 below size do
               (setf (aref dest-data (+ d-off i))
                     (coerce (aref src-data (+ s-off i)) dest-etype)))))
          ;; 慢速路径: 非连续/需要广播 (N维指针推进)
          (t
           (let* ((shape-vec (coerce dest-shape 'simple-vector))
                  (d-strs-vec (coerce (vt-strides dest) 'simple-vector))
                  (s-strs-vec (coerce src-strides 'simple-vector))
                  (rank (length dest-shape)))
             (declare (type simple-vector shape-vec d-strs-vec s-strs-vec)
                      (type fixnum rank))
             (labels ((recurse (depth d-ptr s-ptr)
			(declare (type fixnum depth d-ptr s-ptr))
			(if (= depth rank)
                            (setf (aref dest-data d-ptr)
                                  (coerce (aref src-data s-ptr) dest-etype))
                            (let ((dim (svref shape-vec depth))
                                  (d-str (svref d-strs-vec depth))
                                  (s-str (svref s-strs-vec depth)))
                              (declare (type fixnum dim d-str s-str))
                              (loop for i fixnum from 0 below dim do
				(recurse (1+ depth) d-ptr s-ptr)
				(incf d-ptr d-str)
				(incf s-ptr s-str))))))
               (recurse 0 (vt-offset dest) (vt-offset src))))))
        
        dest))))

(declaim (inline vt-ref))
(defun vt-ref (vt &rest indices)
  "获取张量指定位置的元素 (支持高维索引，支持负索引。无深度越界检查以换取极致性能)。"
  ;; 1. 防崩检查：拦截空张量访问
  (when (zerop (vt-size vt))
    (error "cannot index into an empty tensor (size 0)"))
  (let* ((shape (vt-shape vt))
         (strides (vt-strides vt))
         (offset (vt-offset vt)))
    ;; 2. 防崩检查：确保维度数量匹配，避免底层 nil 算术运算导致崩溃
    (unless (= (length indices) (length shape))
      (error "索引数量 ~a 与张量维度数 ~a 不匹配"
	     (length indices) (length shape)))
    (let ((ptr offset))
      (declare (type fixnum ptr))
      ;; 3. 引入 dim 用于处理负索引
      (loop for idx in indices
            for dim in shape
            for stride in strides
            do (setf ptr
		     (the fixnum 
                          (+ ptr 
                             (the fixnum 
                                  (* (the fixnum (if (minusp idx)
						     (+ idx dim) idx)) 
				     (the fixnum stride)))))))
      (aref (vt-data vt) ptr))))

(defun (setf vt-ref) (value vt &rest indices)
  "设置张量元素(支持高维索引，支持负索引)。不包含越界检查以换取极致性能。"
  (with-float-safe
    (when (zerop (vt-size vt))
      (error "cannot index into an empty tensor (size 0)"))    
    (let* ((shape (vt-shape vt))
           (strides (vt-strides vt))
           (data (vt-data vt))
           (flat-idx (vt-offset vt)))
      (declare (type fixnum flat-idx))
      (unless (= (length indices) (length shape))
        (error "索引数量 ~a 与张量维度数 ~a 不匹配"
               (length indices) (length shape)))
      
      (loop for idx in indices
            for dim in shape
            for stride in strides
            do (incf flat-idx
                     (the fixnum
                          (* (the fixnum
                                  (if (minusp idx)
                                      (+ idx dim)
                                      idx))
                             (the fixnum stride)))))
      ;; 强制类型转换写入底层内存
      (setf (aref data flat-idx)
            (coerce value (vt-element-type vt))))))

(declaim (inline vt-item))
(defun vt-item (x)
  "解包 0 维张量为原生数字，其他原样返回。专为 repl 和条件判断设计。"
  (if (and (vt-p x)
	   (null (vt-shape x)))
      (vt-ref x)
      x))

(declaim (inline vt-broadcast-shapes))
(defun vt-broadcast-shapes (shape1 shape2)
  "计算广播结果形状，严格对标 numpy 规范（完美支持 0 尺寸维度）。"
  (declare (list shape1 shape2) (optimize (speed 3)))
  (let ((len1 (length shape1))
        (len2 (length shape2))
        (result '()))
    (declare (fixnum len1 len2))
    (loop for i fixnum from 1 to (max len1 len2)
          for dim1 fixnum = (if (<= i len1) (nth (- len1 i) shape1) 1)
          for dim2 fixnum = (if (<= i len2) (nth (- len2 i) shape2) 1)
          do (cond ((= dim1 dim2) (push dim1 result)) ;; 规则1: 相等则取其一 (0=0 -> 0)
                   ((= dim1 1) (push dim2 result))    ;; 规则2: dim1为1，取dim2 (1,0->0; 1,3->3)
                   ((= dim2 1) (push dim1 result))    ;; 规则2: dim2为1，取dim1 (0,1->0; 3,1->3)
                   (t (error "形状 ~a 和 ~a 无法进行广播，维度 ~a 和 ~a 不兼容" 
                             shape1 shape2 dim1 dim2)))) ;; 规则3: 其他报错 (0,3报错)
    (the list result)))

(declaim (inline vt-broadcast-strides))
(defun vt-broadcast-strides (orig-shape target-shape orig-strides)
  "计算广播后的步长. 添加形状兼容性校验，避免静默越界。"
  (declare (list orig-shape target-shape orig-strides))
  (declare (optimize (speed 3) (safety 1))) ; 开启基础安全检查
  (let ((rank-diff (- (length target-shape) (length orig-shape))))
    (declare (fixnum rank-diff))
    (when (minusp rank-diff)
      (error "vt-broadcast-strides: orig-shape ~a rank > target-shape ~a rank" 
             orig-shape target-shape))
    
    ;; 预分配固定数组，避免 cons 分配 (可进一步优化为 simple-vector)
    (let ((result (make-list (length target-shape))))
      (loop
	for i fixnum from 0 below (length target-shape)
        for tail-ptr = (nthcdr i result)
        do (setf (car tail-ptr)
                 (cond
                   ((< i rank-diff) 0)
                   (t (let* ((orig-idx
			       (the fixnum (- i rank-diff)))
                             (orig-dim
			       (the fixnum (nth orig-idx orig-shape)))
                             (target-dim
			       (the fixnum (nth i target-shape))))
                        (unless (or (= orig-dim target-dim)
				    (= orig-dim 1))
                          (error "vt-broadcast-strides: shape mismatch! orig-dim ~a vs target-dim ~a at axis ~a"
                                 orig-dim target-dim i))
                        (if (= orig-dim 1)
			    0
			    (nth orig-idx orig-strides)))))))
      result)))


;; 核心宏:高性能广播迭代 
(defmacro with-broadcasting-ptrs ((t1 t2 result-vt) &body body)
  "高性能广播宏. 指针随着循环正确递增."
  (with-float-safe
    (let ((idx-var (gensym "idx"))
          (rank-var (gensym "rank"))
          (res-shape (gensym "res-shape"))
          ;; 指针变量
          (p1 (gensym "p1"))
          (p2 (gensym "p2"))
          (pr (gensym "pr"))
          ;; 步长变量
          (s1 (gensym "s1"))
          (s2 (gensym "s2"))
          (sr (gensym "sr"))
          (dim (gensym "dim"))
          (data1 (gensym "data1"))
          (data2 (gensym "data2"))
          (data-res (gensym "data-res")))
      `(let* ((,res-shape (vt-shape ,result-vt))
              (,rank-var (length ,res-shape))
              (,data1 (vt-data ,t1))
              (,data2 (vt-data ,t2))
              (,data-res (vt-data ,result-vt))
              ;; 预计算广播后的步长
              (strides1-list (vt-broadcast-strides
			      (vt-shape ,t1) ,res-shape (vt-strides ,t1)))
              (strides2-list (vt-broadcast-strides
			      (vt-shape ,t2) ,res-shape (vt-strides ,t2)))
              (strides-res-list (vt-strides ,result-vt)))
	 
	 (labels ((recurse (depth ,p1 ,p2 ,pr)
                    (if (= depth ,rank-var)
			;; 最内层:直接访问内存
			(let ((val1 (aref ,data1 ,p1))
                              (val2 (aref ,data2 ,p2)))
                          (setf (aref ,data-res ,pr) (progn ,@body)))                      
			;; 外层:循环并移动指针
			(let* ((,dim (nth depth ,res-shape))
                               (,s1 (nth depth strides1-list))
                               (,s2 (nth depth strides2-list))
                               (,sr (nth depth strides-res-list)))                        
                          ;; === 使用局部变量迭代指针 ===
                          (let ((cur-p1 ,p1)
				(cur-p2 ,p2)
				(cur-pr ,pr))
                            (loop for ,idx-var from 0 below ,dim do
                              (recurse (1+ depth) cur-p1 cur-p2 cur-pr)
                              ;; 手动递增指针
                              (incf cur-p1 ,s1)
                              (incf cur-p2 ,s2)
                              (incf cur-pr ,sr)))))))         
           ;; 初始调用,加上各自的 offset
           (recurse 0
                    (vt-offset ,t1)
                    (vt-offset ,t2)
                    (vt-offset ,result-vt)))))))
