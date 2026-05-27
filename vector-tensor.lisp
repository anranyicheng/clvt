(in-package #:clvt)

(defmacro with-float-safe (&body body)
  `(sb-int:with-float-traps-masked
       (:invalid :divide-by-zero :overflow :underflow)
     ,@body))

(defun fixnump (num)
  "判断数字是不是fixnum类型"
  (typep num 'fixnum))

(defun double-float-p (num)
  "判断数字是不是double-float类型"
  (typep num 'double-float))

(defstruct (vt (:constructor %make-vt))
  "n维张量结构.
   data: 存储数据的一维简单数组.
   shape: 形状列表.
   strides: 步长列表.
   offset: 起始偏移量,支持切片视图不复制数据.
   etype: 元素类型符号, 如 'double-float, 'fixnum 等"
  (data (make-array 0) :type (simple-array *))
  (shape nil :type list)
  (strides nil :type list)
  (offset 0 :type fixnum)
  (etype 'double-float :type symbol))

(declaim (inline vt-element-type
		 vt-shape
		 vt-offset
		 vt-strides
		 vt-data
		 vt-reshape
		 vt-etype
		 make-vt
		 vt-zeros
		 vt-shape-to-size))

(defun vt-element-type (vt)
  "返回数组元素的类型"
  (vt-etype vt))

(defun vt-shape-to-size (shape)
  "计算形状对应的总元素个数."
  (declare (list shape))
  (reduce #'* shape :initial-value 1))

(defun make-vt (shape initial-element &key (type 'double-float))
  "创建一个指定形状的张量,填充初始值."
  (declare (list shape))
  (with-float-safe
    (let* ((size (vt-shape-to-size shape))
	   (tmp-type (if (eq type 'fixnum)
			 'fixnum
			 'double-float))
           (data (make-array size :element-type tmp-type
				  :initial-element
				  (coerce initial-element tmp-type)))
           (strides (vt-compute-strides shape)))
      (%make-vt :data data
		:shape shape
		:strides strides
		:offset 0
		:etype tmp-type))))


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

(defun vt-from-sequence (contents &key (type 'double-float))
  "从嵌套序列创建张量。支持任意维度的规则嵌套列表或向量。
   例如 (vt-from-sequence '((1 2) (3 4))) 返回形状为 (2 2) 的张量。
   若传入一维序列，返回一维张量。"
  ;; 辅助函数：递归推断形状并收集元素
  (with-float-safe
    (labels
	((infer-shape (seq)
           "返回形状列表，同时检查是否规则。"
           (if (or (listp seq)
		   (typep seq 'vector))
	       (let ((len (length seq)))
		 (if (zerop len)
                     (list 0)   ; 空序列作为一维0尺寸
                     (let* ((first (elt seq 0))
                            (rest-shape (when (or (listp first)
						  (typep first 'vector))
                                          (infer-shape first))))
		       (if rest-shape
                           ;; 嵌套：所有子序列必须形状一致
                           (cons
			    len
			    (loop for i from 1 below len
                                  for sub = (elt seq i)
                                  for sub-shape = (infer-shape sub)
                                  unless (equal sub-shape rest-shape)
                                    do (error "不规则嵌套: 期望 ~a 但得到 ~a"
					      rest-shape sub-shape)
                                  finally (return rest-shape)))
                           ;; 叶子：一维
                           (list len)))))
	       ;; 不是序列（但这种情况不应该发生，因为顶层是序列）
	       (error "无法从 ~a 创建张量" seq)))
	 (fill-tensor (data seq shape strides depth flat-idx)
           (if (null shape)
	       ;; 到达叶子：写入元素
	       (setf (aref data flat-idx) (coerce seq type))
	       (let ((len (first shape))
                     (stride (first strides)))
		 (loop for i from 0 below len
		       for elem = (if (and (or (listp seq)
					       (typep seq 'vector))
                                           (< i (length seq)))
				      (elt seq i)
				      (error "序列长度不匹配"))
		       for offset = (+ flat-idx (* i stride))
		       do (fill-tensor data elem (rest shape) (rest strides)
				       (1+ depth) offset))))))
      (let* ((shape (infer-shape contents))
             (size (reduce #'* shape :initial-value 1))
             (data (make-array size :element-type type
				    :initial-element (coerce 0 type)))
             (strides (vt-compute-strides shape)))
	(fill-tensor data contents shape strides 0 0)
	(%make-vt :data data
		  :shape shape
		  :strides strides
		  :offset 0
		  :etype type)))))

(defun vt-from-array (arr)
  "从数组转vt, 维度不变"
  (declare (array arr))
  (with-float-safe
    (let* ((shape (array-dimensions arr))
	   (type (if (eq (array-element-type arr) 'fixnum)
		     'fixnum
		     'double-float))
	   (vt (vt-zeros shape :type type)))
      (dotimes (idx (reduce #'* shape))
	(declare (fixnum idx))
	(setf (row-major-aref (vt-data vt) idx)
	      (coerce (row-major-aref arr idx) type)))
      vt)))

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
                        (sub-block-size (/ block-size n))
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

(defun vt-to-array (vt)
  "将张量转化为同维度的数组"
  (make-array (vt-shape vt)
	      :element-type (vt-element-type vt)
	      :initial-contents (vt-to-list vt)))

(defun vt-order (vt)
  "张量的维度大小"
  (length (vt-shape vt)))

(defun vt-size (vt)
  "张量的数据大小"
  (reduce #'* (vt-shape vt) :initial-value 1))

(defun vt-arange (n &key (start 0) (step 1) (type 'double-float))
  "创建一个包含范围值的一维张量."
  (declare (fixnum n))
  (let* ((data (make-array n :element-type type))
         (shape (list n)))
    (loop for i fixnum below n
          do (setf (aref data i) (coerce (+ start (* i step)) type)))
    (%make-vt :data data
	      :shape shape
	      :strides (vt-compute-strides shape)
	      :etype type)))

(defun vt-zeros (shape &key (type 'double-float))
  "创建全0张量."
  (declare (list shape))
  (make-vt shape 0 :type type))

(defun vt-ones (shape &key (type 'double-float))
  "创建全1张量."
  (declare (list shape))
  (make-vt shape 1 :type type))

(defun vt-const (shape value &key (type 'double-float))
  (declare (list shape))
  (make-vt shape value :type type))

(defun vt-eye (n &key (m n) (value 1) (type 'double-float))
  "创建单位矩阵 或矩形矩阵.
   n: 行数.
   m: 列数 (默认为 n).
   value: 对角线填充值 (默认 1.0)."
  (with-float-safe
    (let* ((rows n)
           (cols m)
           ;; 简单情况:2d 矩阵
           (shape (list rows cols))
           (res (vt-zeros shape :type type))
           (data (vt-data res))
           (strides (vt-strides res)))    
      (declare (type (simple-array * (*)) data)
               (type list strides))
      ;; 对于行优先,data[i, i] 的偏移是 i * row_stride + i * col_stride
      ;; row_stride 通常是 cols (对于密集矩阵), col_stride 是 1
      (let ((row-stride (first strides)) ;; 第一维步长
					 (col-stride (second strides)) ;; 第二维步长
					 (diagonal-len (min rows cols)))      
	(declare (type fixnum row-stride col-stride diagonal-len))
	;; 沿对角线填充
	(loop for i fixnum from 0 below diagonal-len
              for offset fixnum = 0 then (+ offset row-stride col-stride)
              do (setf (aref data offset) (coerce value type))))
      res)))

(defun vt-diag (vector-tensor)
  "从 1d 张量 (向量) 创建方阵对角矩阵.
   输入: 形状为 的张量.
   输出: 形状为 的张量,非对角线为 0."
  (with-float-safe
    (let* ((in-data (vt-data vector-tensor))
           (n (car (vt-shape vector-tensor)))
           (res (make-vt (list n n) 0
			 :type (vt-element-type vector-tensor)))
           (res-data (vt-data res))
           (res-strides (vt-strides res)))    
      (declare (type fixnum n))
      (let ((row-stride (first res-strides))
            (col-stride (second res-strides)))      
	(declare (type fixnum row-stride col-stride))
	;; 填充对角线
	(loop for i fixnum from 0 below n
              for src-off fixnum from 0
              for dst-off fixnum = 0 then (+ dst-off row-stride col-stride)
              do (setf (aref res-data dst-off) (aref in-data src-off))))    
      res)))

(defun vt-triu (tensor &key (k 0) (in-place nil))
  "返回上三角矩阵.
   k: 对角线偏移 (0=主对角线, 1=主对角线之上).
   in-place: 如果为 t,直接修改原张量；否则返回副本.
   支持 batch (rank >= 2)."
  (with-float-safe
    (let* ((res (if in-place tensor (vt-copy tensor)))
	   ;; vt-copy 需自行实现或用 vt-map identity
	   (res-data (vt-data res))
	   (res-shape (vt-shape res))
	   (res-strides (vt-strides res))
	   (rank (length res-shape)))    
      (when (< rank 2) (error "vt-triu requires rank >= 2"))
      (labels
	  ((recurse (depth ptr)
             (declare (type fixnum depth ptr))
             (if (= depth (- rank 2))
		 ;; === 底层矩阵操作 ===
		 (let* ((rows (nth depth res-shape))
			(cols (nth (1+ depth) res-shape))
			(stride-row (nth depth res-strides))
			(stride-col (nth (1+ depth) res-strides)))
                   (declare (type fixnum rows cols stride-row stride-col))                 
                   (loop
		     for r fixnum from 0 below rows
                     for row-ptr fixnum = ptr then (+ row-ptr stride-row) do
                       (loop for c fixnum from 0 below cols do
			 ;; 判断条件:列号 >= 行号 + k
			 (when (< c (+ r k))
                           ;; 置零
                           (setf (aref res-data 
				       (+ row-ptr (* c stride-col)))
				 (coerce 0 (vt-element-type tensor)))))))               
		 ;; === 高维递归 ===
		 (let ((dim (nth depth res-shape))
                       (stride (nth depth res-strides)))
                   (loop for i fixnum from 0 below dim do
                     (recurse (1+ depth) ptr)
                     (incf ptr stride))))))      
	(recurse 0 (vt-offset res)))    
      res)))

(defun vt-tril (tensor &key (k 0) (in-place nil))
  "返回下三角矩阵.
   k: 对角线偏移 (0=主对角线, -1=主对角线之下).
   支持 batch."
  (with-float-safe
    (let* ((res (if in-place tensor (vt-copy tensor)))
           (res-data (vt-data res))
           (res-shape (vt-shape res))
           (res-strides (vt-strides res))
           (rank (length res-shape)))    
      (when (< rank 2) (error "vt-tril requires rank >= 2"))
      (labels 
	  ((recurse (depth ptr)
             (declare (type fixnum depth ptr))
             (if (= depth (- rank 2))
		 ;; === 底层矩阵操作 ===
		 (let* ((rows (nth depth res-shape))
			(cols (nth (1+ depth) res-shape))
			(stride-row (nth depth res-strides))
			(stride-col (nth (1+ depth) res-strides)))
                   (declare (type fixnum rows cols stride-row stride-col))                 
                   (loop
		     for r fixnum from 0 below rows
                     for row-ptr fixnum = ptr then (+ row-ptr stride-row) do
                       (loop for c fixnum from 0 below cols do
			 ;; 判断条件:列号 <= 行号 + k
			 (when (> c (+ r k))
                           ;; 置零
                           (setf (aref res-data 
				       (+ row-ptr (* c stride-col)))
				 (coerce 0 (vt-element-type tensor)))))))               
		 ;; === 高维递归 ===
		 (let ((dim (nth depth res-shape))
                       (stride (nth depth res-strides)))
                   (loop for i fixnum from 0 below dim do
                     (recurse (1+ depth) ptr)
                     (incf ptr stride))))))      
	(recurse 0 (vt-offset res)))    
      res)))

(defun vt-diagonal (tensor &key (offset 0))
  "提取对角线元素.
   返回一个 1d 张量 (向量),如果是 batch 输入则返回 2d 张量.
   offset: 对角线偏移."
  (with-float-safe
    (let* ((in-shape (vt-shape tensor))
           (rank (length in-shape)))    
      (when (< rank 2) (error "vt-diagonal requires rank >= 2"))
      (let* ((rows (nth (- rank 2) in-shape))
             (cols (nth (- rank 1) in-shape))
             ;; 计算对角线长度
             (diag-len (max 0 (- (min rows cols) (abs offset))))
	     (type (vt-element-type tensor))
             ;; 确定输出形状:batch 维度 + 对角线长度
             (out-shape
	       (append (subseq in-shape 0 (- rank 2))
		       (list diag-len)))
             (res (vt-zeros out-shape :type type))
             (res-data (vt-data res))
             (in-data (vt-data tensor))
             (in-strides (vt-strides tensor)))
	(labels 
	    ((recurse (depth in-ptr out-ptr)
               (declare (type fixnum depth in-ptr out-ptr))
               (if (= depth (- rank 2))
                   ;; === 底层:提取对角线 ===
                   (let ((stride-row (nth depth in-strides))
			 (stride-col (nth (1+ depth) in-strides)))
                     (declare (type fixnum stride-row stride-col))
                     (loop
		       for i fixnum from 0 below diag-len
                       ;; 根据偏移量计算起点和步长
                       ;; offset > 0: 向右偏移,起点 (0, offset)
                       ;; offset < 0: 向下偏移,起点
                       for r fixnum = (if (> offset 0) 0 (- offset))
                       for c fixnum = (if (> offset 0) offset 0)
                       ;; 计算指针位置
                       for src-off fixnum = (+ in-ptr 
                                               (* (+ r i) stride-row) 
                                               (* (+ c i) stride-col))
                       do (setf (aref res-data out-ptr)
				(aref in-data src-off))
                          (incf out-ptr)))
                   ;; === 高维递归 ===
                   (let ((dim (nth depth in-shape))
			 (stride (nth depth in-strides))
			 (out-stride
			   (nth depth (vt-strides res)))) ;; res strides 对应 batch 维
                     (loop for i fixnum from 0 below dim do
                       (recurse (1+ depth) in-ptr out-ptr)
                       (incf in-ptr stride)
                       (incf out-ptr out-stride))))))
          (recurse 0 (vt-offset tensor) 0))      
	res))))

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
  "检查张量是否在内存中连续.
   只有连续的张量才能安全地重塑为任意形状."
  (equal (vt-strides vt) (vt-compute-strides (vt-shape vt))))

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
		      :etype (vt-element-type vt))))
        ;; 高效拷贝:利用迭代器填充新数据
        (vt-copy-into new-vt vt)
        new-vt)))

(defun vt-view (vt new-shape)
  "零拷贝重塑视图。对标 PyTorch 的 tensor.view()。
  要求输入张量必须是内存连续的，否则报错。
  用于确保操作不会产生隐式的数据拷贝。"
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
              :etype (vt-element-type vt))))

(defun vt-reshape (vt new-shape)
  "重塑形状.
   如果张量是连续的,则创建视图(零拷贝)；
   否则自动创建连续副本后重塑.
   这解决了原代码在非连续内存上重塑导致的数据错误问题."
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
	 :etype (vt-element-type vt))
        ;; 安全路径:先内存整理,再重塑
        (let ((cont-vt (vt-contiguous vt)))
          (%make-vt
           :data (vt-data cont-vt)
           :shape new-shape
           :strides (vt-compute-strides new-shape)
           :offset 0
	   :etype (vt-element-type cont-vt))))))

(defun vt-transpose (vt &optional (perm nil))
  "零拷贝转置.仅交换步长和形状维度,不移动数据.
   perm: 排列列表,例如 (1 0) 表示交换维度0和1."
  (let* ((rank (length (vt-shape vt)))
         (perm-real
	   (or perm
	       (loop for i from (1- rank) downto 0 collect i))))
    (unless (= (length perm-real) rank)
      (error "置换维度数量与张量秩不匹配"))
    (%make-vt
     :data (vt-data vt)
     :shape (loop for p in perm-real collect (nth p (vt-shape vt)))
     :strides (loop for p in perm-real collect (nth p (vt-strides vt)))
     :offset (vt-offset vt)
     :etype (vt-element-type vt))))

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
	 :etype (vt-element-type vt))))))

(defun vt-split (tensor indices-or-sections &key (axis 0))
  "模仿 numpy 的 split，支持负数 axis
   沿轴分割张量.
   indices-or-sections: 整数n(等分)或索引列表
   axis: 分割轴
   返回: 张量列表"
  (with-float-safe
    (let* ((shape (vt-shape tensor))
           (rank (length shape))
           (ax (vt-normalize-axis axis rank))
           (dim-size (nth ax shape)))    
      (cond
	;; ========== 情况 a：均分 ==========
	((integerp indices-or-sections)
	 (let* ((n indices-or-sections)
		(chunk-size (floor dim-size n)))
           (unless (zerop (rem dim-size n))
             (error "数组沿轴 ~a 的大小 ~a 不能被 ~a 整除" ax dim-size n))
           (loop for i from 0 below n
		 collect (vt-narrow tensor ax 
                                    (* i chunk-size) 
                                    (* (1+ i) chunk-size)))))      
	;; ========== 情况 b：指定位置下刀 ==========
	((listp indices-or-sections)
	 (let ((points (append (list 0) indices-or-sections (list dim-size))))
           (loop for (start end) on points by #'cdr
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
          (setf expanded-specs (nreverse expanded-specs))
          (unless ellipsis-pos
            (let ((remaining (- old-rank (- (length expanded-specs) num-newaxis))))
              (dotimes (i remaining)
		(setf expanded-specs (append expanded-specs '((:all)))))))))
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
		  :etype (vt-element-type vt))))))

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

(defun vt-copy (vt)
  "深度拷贝张量.
   返回一个全新的、内存连续的张量副本.
   副本与原张量完全独立,修改副本不会影响原张量.
   自动处理视图:如果原张量是切片或转置,返回的是落地后的连续数据."
  (with-float-safe
    (let* ((shape (vt-shape vt))
           (src-data (vt-data vt))
           (element-type (vt-element-type vt))
           (size (vt-shape-to-size shape))
           ;; 分配全新的内存空间
           (new-data (make-array size :element-type element-type))
           ;; 构建新张量结构 (总是连续的)
           (new-vt (%make-vt :data new-data
                             :shape (copy-list shape) ; 复制列表防止共享结构
                             :strides (vt-compute-strides shape)
                             :offset 0
			     :etype (vt-element-type vt))))
      (declare (type (simple-array *) src-data new-data)
               (type fixnum size)
	       (type vt new-vt))
      ;; 核心拷贝逻辑:分情况优化
      (if (vt-contiguous-p vt)
          ;; === 优化路径:连续内存 ===
          ;; 直接使用 replace 进行内存块拷贝,速度最快
          (replace new-data src-data
		   :start2 (vt-offset vt)
		   :end2 (+ (vt-offset vt) size))
          ;; === 通用路径:非连续视图 ===
          ;; 使用已有的迭代拷贝逻辑
          (vt-copy-into new-vt vt))    
      new-vt)))

(declaim (inline vt-ref))
(defun vt-ref (vt &rest indices)
  "获取张量元素(支持高维索引，支持负索引)。不包含越界检查以换取极致性能。"
  (let* ((shape (vt-shape vt))
         (strides (vt-strides vt))
         (data (vt-data vt))
         (flat-idx (vt-offset vt)))
    (declare (type fixnum flat-idx))
    (when (zerop (vt-size vt))
      (error "Cannot index into an empty tensor (size 0)"))
    ;; 平行遍历索引、维度和步长，累加物理偏移
    (loop for idx in indices
          for dim in shape
          for stride in strides
          ;; 负索引规范化：如果是负数就加上维度长度，否则原样使用
          do (incf flat-idx
		   (the fixnum
			(* (the fixnum
				(if (minusp idx)
				    (+ idx dim)
				    idx)) 
                           (the fixnum stride)))))
    (aref data flat-idx)))

(defun (setf vt-ref) (value vt &rest indices)
  "设置张量元素(支持高维索引，支持负索引)。不包含越界检查以换取极致性能。"
  (with-float-safe
    (let* ((shape (vt-shape vt))
           (strides (vt-strides vt))
           (data (vt-data vt))
           (flat-idx (vt-offset vt)))
      (declare (type fixnum flat-idx))
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

;;; --- 高效迭代逻辑  ---
(defmacro vt-do-each ((ptr-var val-var vt) &body body)
  "高效遍历宏。
   ptr-var: 当前元素的物理索引。
   val-var: 当前元素的值。"
  (with-float-safe
    (let ((rank-sym (gensym "rank"))
          (shape-sym (gensym "shape"))
          (strides-sym (gensym "strides"))
          (data-sym (gensym "data"))
          (offset-sym (gensym "offset"))
          (depth-sym (gensym "depth")))
      `(let* ((,shape-sym (vt-shape ,vt))
              (,strides-sym (vt-strides ,vt))
              (,data-sym (vt-data ,vt))
              (,offset-sym (vt-offset ,vt))
              (,rank-sym (length ,shape-sym)))
	 (labels ((recurse (,depth-sym current-ptr)
                    (if (= ,depth-sym ,rank-sym)
			;; 到达最内层：绑定为普通变量，允许 declare ignore
			(let ((,ptr-var current-ptr)
                              (,val-var (aref ,data-sym current-ptr)))
                          (declare (ignorable ,ptr-var ,val-var))
                          ,@body)
			;; 中间维度：循环并累加指针
			(let ((dim (nth ,depth-sym ,shape-sym))
                              (stride (nth ,depth-sym ,strides-sym)))
                          (loop for i from 0 below dim do
                            (recurse (1+ ,depth-sym) current-ptr)
                            (incf current-ptr stride))))))
           (recurse 0 ,offset-sym))))))

(defun vt-copy-into (dest src)
  "将 src 的数据拷贝到 dest (支持广播和类型转换).
 src: 源张量或标量数字.
 dest: 目标张量视图.
 注意：dest 不能是带有 0 步长的广播视图(维度>1且步长=0)，否则物理上无法写入不同数据。"
  (setf src (ensure-vt src))  
  (with-float-safe
    (let ((dest-shape (vt-shape dest))
          (src-shape (vt-shape src))) 
      (loop for d in dest-shape
            for s in (vt-strides dest)
            when (and (> d 1) (zerop s))
              do  (error "vt-copy-into: Assignment destination is read-only: 无法对广播产生的视图 (含有大小为 ~a 的广播维度) 进行赋值。若需修改，请先对源张量进行操作或使用 (vt-copy) 生成独立副本。" d))
      (let ((final-shape (vt-broadcast-shapes dest-shape src-shape)))
        (unless (equal final-shape dest-shape)
          (error "copy-into 失败: dest 形状 ~a 无法容纳 src 广播后的形状 ~a"
		 dest-shape final-shape))
        (let* ((dest-data (vt-data dest))
               (src-data (vt-data src))
               (src-strides
		 (vt-broadcast-strides
		  src-shape dest-shape (vt-strides src)))
               (dest-strides (vt-strides dest))
               (dest-offset (vt-offset dest))
               (src-offset (vt-offset src))
               (rank (length dest-shape))
               (element-type (array-element-type dest-data)))
          
          (labels ((recurse (depth d-ptr s-ptr)
                     (declare (type fixnum depth d-ptr s-ptr))
                     (if (= depth rank)
                         (setf (aref dest-data d-ptr)
			       (coerce (aref src-data s-ptr) element-type))
                         (let ((dim (nth depth dest-shape))
                               (d-stride (nth depth dest-strides))
                               (s-stride (nth depth src-strides))
                               (cur-d-ptr d-ptr)
                               (cur-s-ptr s-ptr))
                           (declare (type fixnum dim d-stride s-stride))
                           (loop for i fixnum from 0 below dim do
                             (recurse (1+ depth) cur-d-ptr cur-s-ptr)
                             (incf cur-d-ptr d-stride)
                             (incf cur-s-ptr s-stride))))))
            (recurse 0 dest-offset src-offset))
          dest)))))

;; 快速类型转换
(declaim (inline ensure-vt))
(defun ensure-vt (obj &key (type nil))
  (with-float-safe
    (etypecase obj
      (vt obj)
      (number
       (if (null type)
	   (setf type
		 (cond ((fixnump obj) 'fixnum)
		       (t 'double-float))))
       (%make-vt :data (make-array 1 :initial-element (coerce obj type)
				     :element-type type)
                 :shape nil
		 :strides nil
		 :offset 0
		 :etype type))
      (sequence
       (if (null type)
	   (let* ((data (vt-flatten-sequence obj)))
	     (if (every #'fixnump data)
		 (setf type 'fixnum)
		 (setf type 'double-float))))       
       (vt-from-sequence obj :type type)))))

(declaim (inline vt-broadcast-shapes))
(defun vt-broadcast-shapes (shape1 shape2)
  "计算广播结果形状，严格对标 NumPy 规范（完美支持 0 尺寸维度）。"
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
  "高性能广播宏.
   修复了指针迭代逻辑,确保指针随着循环正确递增."
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

(defun vt-map (fn &rest tensors)
   "高效映射函数.
   支持标量、列表、张量混合输入.
   特性:自动类型转换、广播、指针算术优化、双目运算内联
   极致优化：试探性推断输出类型与布尔语义，统一遍历骨架，将类型分发延迟到叶子节点
   输入读取不再做类型强转假设，彻底杜绝跨类型内存乱读。"
  (declare (function fn) (optimize (safety 0)))
  (with-float-safe
    (let* ((clean-tensors (mapcar #'ensure-vt tensors))
           (n-tensors (length clean-tensors)))
      (when (= n-tensors 0)
        (error "vt-map requires at least one tensor"))
      
      (let* ((out-shape (reduce #'vt-broadcast-shapes
				(mapcar #'vt-shape clean-tensors)))
             (rank (length out-shape))
             (size (vt-shape-to-size out-shape))
             (ins-data (map 'vector #'vt-data clean-tensors))
             (ins-offs (map 'vector #'vt-offset clean-tensors))
             (ins-strides
	       (map 'vector (lambda (vt)
			      (vt-broadcast-strides
			       (vt-shape vt)
			       out-shape
			       (vt-strides vt)))
		    clean-tensors))
             (cur-ptrs (make-array n-tensors :element-type 'fixnum
					     :initial-element 0)))
        (declare (type (simple-array t (*)) ins-data)
                 (type (simple-array t (*)) ins-strides)
                 (type (simple-array fixnum (*)) cur-ptrs)
                 (type list out-shape)
                 (type fixnum rank n-tensors size))
        
        (loop for i fixnum from 0 below n-tensors do
          (setf (aref cur-ptrs i) (aref ins-offs i)))
        
        ;; ==============================================================      
        ;; 阶段 1：冷路径 - 试探性推断累加器类型
        ;; ==============================================================      
        (multiple-value-bind (effective-out-type is-boolean-fn)
            (if (zerop size)
                (let ((pt (apply #'vt-promote-type
				 (mapcar #'vt-element-type clean-tensors))))
                  (values (if (eq pt 'fixnum)
			      'fixnum
			      'double-float)
			  nil))
                (let* ((first-vals
			 (loop for k fixnum from 0 below n-tensors
                               collect
			       (aref (the simple-array (aref ins-data k))
				     (aref cur-ptrs k)))))
                  (let ((first-result (apply fn first-vals)))
                    (cond
                      ((or (eq first-result t)
			   (null first-result))
		       (values 'double-float t))
                      ((and (integerp first-result)
			    (<= most-negative-fixnum
				first-result
				most-positive-fixnum))
                       (values 'fixnum nil))
                      (t (values 'double-float nil))))))
          
          (let* ((res (vt-zeros out-shape :type effective-out-type))
                 (res-data (vt-data res))
                 (res-strides (vt-strides res)))
            
            ;; ==============================================================      
            ;; 阶段 2：热路径 - 统一遍历骨架 (代码只生成一次！)
            ;; ==============================================================      
            ;; 宏的作用仅限于生成递归骨架，不再包含业务逻辑分支
            (macrolet
		((gen-unified-loop (&body leaf-body)
                   `(labels
			((recurse (depth out-ptr)
                           (declare (type fixnum depth out-ptr))
                           (if (= depth rank)
                               (progn ,@leaf-body) ; <--- 只有这里不同
                               (let* ((dim (the fixnum (nth depth out-shape)))
                                      (res-stride
					(the fixnum (nth depth res-strides)))
                                      (strides-at-depth
					(loop for k fixnum from 0 below n-tensors
                                              collect
					      (the fixnum
						   (nth depth (aref ins-strides k))))))
                                 (declare (type fixnum dim res-stride)
                                          (dynamic-extent strides-at-depth))
                                 (loop for i fixnum from 0 below dim do
                                   (recurse (1+ depth) out-ptr)
                                   (incf out-ptr res-stride)
                                   (loop for k fixnum from 0 below n-tensors
                                         for s fixnum in strides-at-depth do
                                           (incf (aref cur-ptrs k) s)))
                                 ;; 指针回溯
                                 (decf out-ptr (the fixnum (* dim res-stride)))
                                 (loop for k fixnum from 0 below n-tensors
                                       for s fixnum in strides-at-depth do
                                         (decf (aref cur-ptrs k)
					       (the fixnum (* dim s))))))))
                      (recurse 0 0))))
              
              ;; 调用宏，传入统一的叶子节点逻辑
              (gen-unified-loop
               ;; ==============================================================      
               ;; 统一的叶子节点逻辑：先求值，再根据预推断的类型写入
               ;; ==============================================================      
               (let ((raw-result
                       ;; 1. 统一读取输入并计算
                       (case n-tensors
                         (1 (funcall
			     fn
			     (aref (the simple-array (aref ins-data 0))
				   (aref cur-ptrs 0))))
                         (2 (funcall
			     fn 
                             (aref (the simple-array (aref ins-data 0))
				   (aref cur-ptrs 0))
                             (aref (the simple-array (aref ins-data 1))
				   (aref cur-ptrs 1))))
                         (otherwise 
                          (apply
			   fn
			   (loop for k fixnum from 0 below n-tensors
                                 collect
				 (aref (the simple-array (aref ins-data k))
				       (aref cur-ptrs k))))))))
                  
                  ;; 2. 根据冷路径推断的类型，将结果安全写入
                  ;; 此处 ecase 分支预测命中率达 100%，零性能损耗
                  (ecase effective-out-type
                    (double-float
                     (if is-boolean-fn
                         (setf (aref (the (simple-array double-float (*))
					  res-data)
				     out-ptr)
                               (the double-float
				    (if raw-result 1.0d0 0.0d0)))
                         (setf (aref (the (simple-array double-float (*))
					  res-data)
				     out-ptr)
                               (the double-float
				    (coerce raw-result 'double-float)))))
                    (fixnum
                     (setf (aref (the (simple-array fixnum (*))
				      res-data)
				 out-ptr)
                           (the fixnum
                                (if (and (integerp raw-result)
                                         (<= most-negative-fixnum
					     raw-result
					     most-positive-fixnum))
                                    raw-result
                                    ;; 饱和截断保护
                                    (max most-negative-fixnum 
                                         (min most-positive-fixnum
					      (truncate raw-result)))))))))))
            
            res))))))


(defun vt-ensure-shape-compatible (shape axis)
  "检查 axis 是否合法. 支持负数轴."
  (let* ((rank (length shape))
         (real-axis (if (< axis 0) (+ rank axis) axis)))
    (when (or (< real-axis 0) (>= real-axis rank))
      (error "axis ~a 越界,张量秩为 ~a" axis rank))))

(declaim (inline get-reduction-identity))
(defun get-reduction-identity (op element-type)
  "根据操作类型和元素类型,返回正确的初始值."
  (case op
    (:sum
     (coerce 0 element-type))
    (:max
     (cond ((eq element-type 'double-float) most-negative-double-float)
           ((eq element-type 'fixnum) most-negative-fixnum)
           (t 0)))
    (:min
     (cond ((eq element-type 'double-float) most-positive-double-float)
           ((eq element-type 'fixnum) most-positive-fixnum)
           (t 0)))))

(declaim (inline vt-reduce))
(defun vt-reduce
    (tensor axis init-val reducer-fn &key return-arg keepdims)
  "通用归约核心.
   特性:
   1. 使用列表递归 (经测试速度更优).
   2. 冷路径推断累加器类型，热路径特化，消除 coerce 开销.
   3. 防御 Fixnum 溢出与布尔崩溃，(safety 0) 下绝对安全."
  (declare (type vt tensor)
           (type (or null fixnum) axis)
           (type function reducer-fn)
           (optimize (safety 0)))
  (with-float-safe
    (let* ((in-shape (vt-shape tensor))
           (element-type (vt-element-type tensor))
           (rank (length in-shape))
	   (real-axis (vt-normalize-axis axis rank))
           (is-global-reduction (null real-axis))
	   ;; 1. 输出形状
	   (out-shape
	     (cond
               ((and is-global-reduction (not keepdims)) nil)
               (is-global-reduction (make-list rank :initial-element 1))
               ((not keepdims)
		(loop for d in in-shape
		      for i from 0
		      unless (= i real-axis)
			collect d))
               (t (loop for d in in-shape
			for i from 0
		        collect (if (= i real-axis) 1 d)))))
	   ;; 2. 输入属性
	   (in-data (vt-data tensor))
	   (in-strides (vt-strides tensor))
	   (in-offset (vt-offset tensor))           
	   ;; 3. 索引步长映射 (用于计算 argmax/argmin 的逻辑位置)
	   (arg-strides
	     (if return-arg 
		 (if is-global-reduction
		     (vt-compute-strides in-shape)
		     (loop for i from 0 below rank
			   if (= i real-axis) collect 1
			     else collect 0))
		 (make-list rank :initial-element 0)))
	   ;; 4. 归约轴大小
	   (axis-size
	     (if real-axis
                 (the fixnum (nth real-axis in-shape))
                 (the fixnum (reduce #'* in-shape :initial-value 1)))))      
      
      (declare (type (simple-array * (*)) in-data)
               (type list in-strides arg-strides in-shape))

      ;; 当归约轴大小为 0 时，直接返回
      (when (zerop axis-size)
        (let ((empty-result-val
                (cond ((eq init-val 0) (coerce 0 element-type))
                      ((or (eq init-val most-positive-double-float)
                           (eq init-val most-negative-double-float))
                       (vt-float-nan))
                      (t (coerce init-val element-type)))))
          (return-from vt-reduce
            (values (make-vt out-shape empty-result-val :type element-type)
                    (when return-arg (make-vt out-shape 0 :type 'fixnum))))))

      ;; ==============================================================
      ;; 阶段 1：冷路径 - 试探性推断累加器类型
      ;; ==============================================================
      (let* ((first-val (aref in-data 0))
             (first-result (funcall reducer-fn init-val first-val))
             (effective-out-type
               (cond
                 ((or (eq first-result t)
		      (null first-result))
		  'double-float)
                 ((and (integerp first-result)
                       (<= most-negative-fixnum first-result most-positive-fixnum))
                  'fixnum)
                 (t 'double-float)))
             ;; 安全的初始值转换
             (safe-init-val
               (if (eq effective-out-type 'fixnum)
                   (if (and (integerp init-val)
                            (<= most-negative-fixnum init-val most-positive-fixnum))
                       init-val
                       (truncate init-val))
                   (coerce init-val 'double-float))))
        
        ;; ==============================================================
        ;; 阶段 2：分配结果与步长映射，执行热路径
        ;; ==============================================================
        (let* ((res (make-vt out-shape safe-init-val :type effective-out-type))
               (res-data (vt-data res))
               (res-offset (vt-offset res))           
               (res-idx (when return-arg (make-vt out-shape 0 :type 'fixnum)))
               (res-idx-data (when res-idx (vt-data res-idx)))
               ;; 关键修正: out-strides-map 必须在 res 分配之后计算
               (out-strides-map
                 (if is-global-reduction
                     (make-list rank :initial-element 0)
                     (loop for i from 0 below rank
                           if (= i real-axis) collect 0
                             else collect
                                  (let ((out-idx (if keepdims
						     i
						     (if (< i real-axis)
							 i
							 (1- i)))))
                                    (nth out-idx (vt-strides res)))))))

          (declare (type (simple-array * (*)) res-data)
                   (type list out-strides-map))

          ;; 宏消除递归冗余
          (macrolet
              ((gen-reduce (leaf-body)
		 `(labels
                      ((recurse (depth in-ptr out-ptr out-idx-ptr current-arg-val)
			 (declare (type fixnum depth in-ptr out-ptr out-idx-ptr current-arg-val))
			 (if (= depth rank)
                             ,leaf-body
                             (let* ((dim (nth depth in-shape))
                                    (in-stride (nth depth in-strides))
                                    (out-stride (nth depth out-strides-map))
                                    (arg-stride (nth depth arg-strides)))
                               (declare (type fixnum dim in-stride out-stride arg-stride))
                               (loop for i fixnum from 0 below dim do
				 (recurse (1+ depth)
                                          in-ptr
                                          out-ptr
                                          out-idx-ptr
                                          (+ current-arg-val (* i arg-stride)))
				 (incf in-ptr in-stride)
				 (incf out-ptr out-stride)
				 (when return-arg
                                   (incf out-idx-ptr out-stride)))))))
                    (recurse 0 in-offset res-offset (if res-idx (vt-offset res-idx) 0) 0))))

            ;; 热路径特化
            (ecase effective-out-type
              (double-float
               (gen-reduce
                (let ((val (aref (the simple-array in-data) in-ptr))
                      (cur-acc (aref (the (simple-array double-float (*)) res-data) out-ptr)))
                  (multiple-value-bind (new-acc do-update-idx)
                      (funcall reducer-fn cur-acc val)
                    (setf (aref (the (simple-array double-float (*)) res-data) out-ptr)
                          (the double-float (coerce new-acc 'double-float)))
                    (when (and return-arg do-update-idx res-idx-data)
                      (setf (aref (the (simple-array fixnum (*)) res-idx-data) out-idx-ptr)
                            current-arg-val))))))

              (fixnum
               (gen-reduce
                (let ((val (aref (the simple-array in-data) in-ptr))
                      (cur-acc (aref (the (simple-array fixnum (*)) res-data) out-ptr)))
                  (multiple-value-bind (new-acc do-update-idx)
                      (funcall reducer-fn cur-acc val)
                    (setf (aref (the (simple-array fixnum (*)) res-data) out-ptr)
                          (the fixnum
                               (if (and (integerp new-acc)
					(<= most-negative-fixnum new-acc most-positive-fixnum))
                                   new-acc
                                   (truncate new-acc))))
                    (when (and return-arg do-update-idx res-idx-data)
                      (setf (aref (the (simple-array fixnum (*)) res-idx-data) out-idx-ptr)
                            current-arg-val))))))))

          (values res res-idx))))))

(defun vt-matmul (a b)
  "矩阵乘法，兼容 1d 向量（对标 numpy 的 @ 运算符）。"
  (let ((ra (vt-order a))
        (rb (vt-order b)))
    (cond
      ;; 2d @ 2d → 2d（矩阵乘法）
      ((and (= ra 2) (= rb 2))
       (vt-matmul-df a b))
      ;; 1d @ 1d → 标量（内积）
      ((and (= ra 1) (= rb 1)) (vt-einsum "i,i->" a b))
      ;; 2d @ 1d → 1d（矩阵乘向量）
      ((and (= ra 2) (= rb 1)) (vt-einsum "ij,j->i" a b))
      ;; 1d @ 2d → 1d（向量乘矩阵）
      ((and (= ra 1) (= rb 2)) (vt-einsum "i,ij->j" a b))
      ;; >2d @ >2d → 批量矩阵乘法
      (t (vt-einsum "...ij,...jk->...ik" a b)))))


(defun vt-@ (vt1 vt2)
  (vt-matmul vt1 vt2))

(defun vt-= (t1 t2)
  "逐元素相等比较.
   支持广播.返回布尔张量(1.0或0.0)."
  (with-float-safe
    (let* ((result-element-type (vt-element-type t1))
	   (true-result (coerce 1 result-element-type))
	   (false-result (coerce 0 result-element-type)))
      (vt-map (lambda (a b)
		(if (= a b) true-result false-result))
	      t1 t2))))

(defun vt-/= (t1 t2)
  "逐元素不相等比较.
   支持广播.返回布尔张量(1.0或0.0)."
  (with-float-safe
    (let* ((result-element-type (vt-element-type t1))
	   (true-result (coerce 1 result-element-type))
	   (false-result (coerce 0 result-element-type)))
      (vt-map (lambda (a b)
		(if (/= a b) true-result false-result))
	      t1 t2))))

(defun vt-< (t1 t2)
  "逐元素小于比较.
   支持广播.返回布尔张量(1.0或0.0)."
  (with-float-safe
    (let* ((result-element-type (vt-element-type t1))
	   (true-result (coerce 1 result-element-type))
	   (false-result (coerce 0 result-element-type)))
      (vt-map (lambda (a b)
		(if (< a b) true-result false-result))
	      t1 t2))))

(defun vt-<= (t1 t2)
  "逐元素小于等于比较.
   支持广播.返回布尔张量(1.0或0.0)."
  (with-float-safe
    (let* ((result-element-type (vt-element-type t1))
	   (true-result (coerce 1 result-element-type))
	   (false-result (coerce 0 result-element-type)))
      (vt-map (lambda (a b)
		(if (<= a b) true-result false-result))
	      t1 t2))))

(defun vt-> (t1 t2)
  "逐元素大于比较.
   支持广播.返回布尔张量(1.0或0.0)."
  (with-float-safe
    (let* ((result-element-type (vt-element-type t1))
	   (true-result (coerce 1 result-element-type))
	   (false-result (coerce 0 result-element-type)))
      (vt-map (lambda (a b)
		(if (> a b) true-result false-result))
	      t1 t2))))

(defun vt->= (t1 t2)
  "逐元素大于等于比较.
   支持广播.返回布尔张量(1.0或0.0)."
  (with-float-safe
    (let* ((result-element-type (vt-element-type t1))
	   (true-result (coerce 1 result-element-type))
	   (false-result (coerce 0 result-element-type)))
      (vt-map (lambda (a b)
		(if (>= a b) true-result false-result))
	      t1 t2))))

(defun vt-promote-type (&rest types)
  "推断运算结果类型。对标 NumPy: 只要有浮点数就提升为浮点数"
  (cond
    ((member 'double-float types) 'double-float)
    ((member 'single-float types) 'single-float)
    (t 'fixnum)))

(defun vt-where (condition x y)
  "三元条件选择，对标 PyTorch 的 torch.where 和 NumPy 的 np.where(cond, x, y)。
   根据 condition 的元素真假，从 x 或 y 中选择对应元素组成新张量。
   condition, x, y 会自动广播到同一形状。
   注意：如需查找非零元素的索引，请使用 vt-nonzero"
  (with-float-safe    
    ;; 统一转换为张量
    (setf condition (ensure-vt condition))
    (setf x (ensure-vt x))
    (setf y (ensure-vt y))
    ;; 计算广播形状和步长
    (let* ((target-shape (vt-broadcast-shapes
			  (vt-shape condition)
                          (vt-broadcast-shapes
			   (vt-shape x)
			   (vt-shape y))))
	   (element-type (vt-promote-type (vt-element-type x)
					  (vt-element-type y)))
           (result (vt-zeros target-shape :type element-type))
           (cond-strides
	     (vt-broadcast-strides
	      (vt-shape condition) target-shape (vt-strides condition)))
           (x-strides (vt-broadcast-strides
		       (vt-shape x) target-shape (vt-strides x)))
           (y-strides (vt-broadcast-strides
		       (vt-shape y) target-shape (vt-strides y)))
           (res-strides (vt-strides result))
           (rank (length target-shape)))
      
      (labels
	  ((recurse (depth c-ptr x-ptr y-ptr r-ptr)
             (if (= depth rank)
		 ;; 叶节点：根据 condition 决定取 x 还是 y
		 (setf (aref (vt-data result) r-ptr)
                       (coerce (if (not (zerop (aref (vt-data condition)
						     c-ptr)))
                                   (aref (vt-data x) x-ptr)
                                   (aref (vt-data y) y-ptr))
                               element-type))
		 ;; 分支节点：递归遍历
		 (let ((dim (nth depth target-shape))
                       (c-stride (nth depth cond-strides))
                       (x-stride (nth depth x-strides))
                       (y-stride (nth depth y-strides))
                       (r-stride (nth depth res-strides)))
                   (loop for i fixnum from 0 below dim do
                     (recurse (1+ depth)
                              (+ c-ptr (* i c-stride))
                              (+ x-ptr (* i x-stride))
                              (+ y-ptr (* i y-stride))
                              (+ r-ptr (* i r-stride))))))))
	(recurse 0
		 (vt-offset condition)
		 (vt-offset x)
		 (vt-offset y)
		 (vt-offset result)))
      result)))

(defun vt-argwhere (condition)
  "查找非零元素的坐标.
   condition: 条件张量.
   返回: 形状为 (n, rank) 的二维张量,每一行是一个非零元素的完整坐标.
   示例: (vt-argwhere tensor) -> [[0, 1], [2, 3], ...]"
  (declare (type vt condition))
  (with-float-safe
    (let* ((in-shape (vt-shape condition))
           (rank (length in-shape))
           (in-data (vt-data condition))
           (in-strides (vt-strides condition))
           (in-offset (vt-offset condition))
           ;; 使用动态数组收集结果坐标
           (result-indices (make-array 0 :element-type 'fixnum 
					 :fill-pointer t 
					 :adjustable t)))
      
      (declare (type (simple-array double-float (*)) in-data)
               (type list in-shape in-strides)
               (type (vector fixnum) result-indices))
      
      ;; 使用固定大小的坐标缓冲区,避免递归中频繁分配列表
      (let ((coord-buffer (make-array rank :element-type 'fixnum)))
	(declare (type (simple-array fixnum (*)) coord-buffer))
	(labels
	    ((recurse (depth current-ptr)
               (declare (type fixnum depth current-ptr))
               (if (= depth rank)
                   ;; 叶节点:检查值
                   (when (/= (aref in-data current-ptr) 0.0d0)
                     ;; 记录当前坐标缓冲区的内容
                     (loop for c across coord-buffer do
                       (vector-push-extend c result-indices)))
                   ;; 中间节点:遍历维度
                   (let ((dim (nth depth in-shape))
			 (stride (nth depth in-strides)))
                     (declare (type fixnum dim stride))
                     (loop for i fixnum from 0 below dim do
                       ;; 更新当前层坐标
                       (setf (aref coord-buffer depth) i)
                       (recurse (1+ depth)
				(+ current-ptr (* i stride))))))))
          (recurse 0 in-offset)))    
      ;; 结果封装
      (let* ((total-indices (length result-indices))
             (count (if (zerop rank)
			total-indices (floor total-indices rank))))
	;; 如果没有找到元素,返回空张量
	(if (zerop count)
            (vt-zeros (list 0 rank))
            (let ((final-data
		    (make-array total-indices :element-type 'fixnum)))
              ;; 转换类型
              (loop for i from 0 below total-indices
                    for idx across result-indices
                    do (setf (aref final-data i) idx))
              ;; 返回 (n, rank) 形状的张量
              (%make-vt :data final-data
			:shape (list count rank)
			:strides (list rank 1) ; 连续内存
			:offset 0
			:etype (array-element-type final-data))))))))


;;;; 关于nan的相关定义
(eval-when (:compile-toplevel :load-toplevel :execute)
  (defun vt-float-nan ()
    (with-float-safe
      (/ 0.0d0 0.0d0)))

  (defun vt-float-nan-p (x)
    (sb-kernel::float-nan-p (coerce x 'double-float)))

  (defun vt-float-nan-= (nan-a nan-b)
    (with-float-safe
      (= nan-a nan-b)))

  (defun vt-float-pos-inf ()
    (with-float-safe
      (* sb-kernel::double-float-positive-infinity 2.0d0)))

  (defun vt-float-neg-inf ()
    (with-float-safe
      (* sb-kernel::double-float-negative-infinity 2.0d0)))

  (defun vt-float-inf-p (x)
    (sb-kernel::float-infinity-p (coerce x 'double-float)))

  (defun vt-float-inf-= (inf-a inf-b)
    (with-float-safe
      (= inf-a inf-b)))

  (defun vt-float-nan-inf-= (a b)
    (with-float-safe
      (= a b))))

(defconstant +vt-float-nan+
  (load-time-value (vt-float-nan)))

(defconstant +vt-float-pos-inf+
  (load-time-value (vt-float-pos-inf)))

(defconstant +vt-float-neg-inf+
  (load-time-value (vt-float-neg-inf)))

(defun vt-numpy-sort (sequence &optional (predicate #'<))
  "对实数序列（列表或可转为列表的向量）进行排序(默认升序)，
   将 nan 放在末尾，保持多个 nan 的原始相对顺序。
   返回新列表。"
  (declare (type (or list vector) sequence))
  (assert (or (eq predicate #'<)
	      (eq predicate '<)
	      (eq predicate #'>)
	      (eq predicate '>)))
  (with-float-safe
    (let ((sequence (coerce sequence 'list))    ; 统一为列表
          non-nans
          nans)
      ;; 分离 nan 和非 nan，按原始顺序
      (dolist (x sequence (setq non-nans (nreverse non-nans)
				nans      (nreverse nans)))
	(if (vt-float-nan-p x)
            (push x nans)
            (push x non-nans)))
      ;; 只在有限数（含 inf）上做标准稳定排序
      (setq non-nans (stable-sort non-nans predicate))
      ;; 拼接：有限数在前，nan 在后
      (cond ((or (eq predicate #'<)
		 (eq predicate '<))
	     (append non-nans nans))
	    ((or (eq predicate #'>)
		 (eq predicate '>))
	     (append nans (nreverse non-nans)))))))
