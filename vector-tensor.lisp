(in-package #:clvt)

(defstruct (vt (:constructor %make-vt))
  "N维张量结构.
   data: 存储数据的一维简单数组.
   shape: 形状列表.
   strides: 步长列表.
   offset: 起始偏移量,支持切片视图不复制数据."
  (data (make-array 0) :type (simple-array *))
  (shape nil :type list)
  (strides nil :type list)
  (offset 0 :type fixnum))

(declaim (inline vt-element-type
		 vt-shape
		 vt-offset
		 vt-strides
		 vt-data
		 vt-reshape
		 make-vt
		 vt-zeros
		 vt-shape-to-size))

(defun vt-element-type (vt)
  "返回数组元素的类型"
  (array-element-type (vt-data vt)))

(defun vt-shape-to-size (shape)
  "计算形状对应的总元素个数."
  (declare (list shape))
  (reduce #'* shape :initial-value 1))

(defun make-vt (shape initial-element &key (type 'double-float))
  "创建一个指定形状的张量,填充初始值."
  (declare (list shape))
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
	      :strides strides)))

(defun vt-flatten-sequence (seq)
  "深度优先遍历 SEQ 及其嵌套序列,返回所有原子元素的列表.
   支持列表、向量、多维数组、位数组等序列类型,按行优先顺序处理."
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
                (t (error "~S 不是序列" s)))))
          (nreverse result)))))

(defun vt-from-sequence (contents &key (type 'double-float))
  "从嵌套序列创建张量。支持任意维度的规则嵌套列表或向量。
   例如 (vt-from-sequence '((1 2) (3 4))) 返回形状为 (2 2) 的张量。
   若传入一维序列，返回一维张量。"
  ;; 辅助函数：递归推断形状并收集元素
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
                                  do (error "不规则嵌套: 期望 ~A 但得到 ~A"
					    rest-shape sub-shape)
                                finally (return rest-shape)))
                         ;; 叶子：一维
                         (list len)))))
             ;; 不是序列（但这种情况不应该发生，因为顶层是序列）
             (error "无法从 ~A 创建张量" seq)))
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
		:offset 0))))

(defun vt-from-array (arr)
  "从数组转vt, 维度不变"
  (declare (array arr))
  (let* ((shape (array-dimensions arr))
	 (type (if (eq (array-element-type arr) 'fixnum)
		   'fixnum
		   'double-float))
	 (vt (vt-zeros shape :type type)))
    (dotimes (idx (reduce #'* shape))
      (declare (fixnum idx))
      (setf (row-major-aref (vt-data vt) idx)
	    (coerce (row-major-aref arr idx) type)))
    vt))

(defun vt-flatten-to-nested (dims data)
  "将按行主序存储的一维向量 data 转换为符合 dims 维度的嵌套列表.
   (vt-flatten-to-nested (vt-shape vt) (vt-data vt))"
  (let ((total-size (reduce #'* dims))    ;; 总元素数
        (idx 0))                          ;; 当前读取位置
    (assert (= total-size (length data)) (data)
            "数据长度与维度乘积不匹配")
    (labels ((recurse (dims block-size)
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
      (recurse dims total-size))))

(defun vt-data->list (vt)
  "张量数据按照维度结构导出为列表形式,多维则是嵌套列表"
  (vt-flatten-to-nested (vt-shape vt) (vt-data vt)))

(defun vt-order (vt)
  "张量的维度大小"
  (length (vt-shape vt)))

(defun vt-size (vt)
  "张量的数据大小"
  (reduce #'* (vt-shape vt)))

(defun vt-arange (n &key (start 0) (step 1) (type 'double-float))
  "创建一个包含范围值的一维张量."
  (declare (fixnum n))
  (let* ((data (make-array n :element-type type))
         (shape (list n)))
    (loop for i fixnum below n
          do (setf (aref data i) (coerce (+ start (* i step)) type)))
    (%make-vt :data data
	      :shape shape
	      :strides (vt-compute-strides shape))))

(defun vt-random (shape &key (type 'double-float))
  "创建随机数张量"
  (declare (list shape))
  (vt-map (lambda (x) (setf x (coerce (random 1.0) type)))
	  (vt-zeros shape)))

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

(defun vt-eye (n &key (m n) (value 1.0d0) (type 'double-float))
  "创建单位矩阵 或矩形矩阵.
   N: 行数.
   M: 列数 (默认为 N).
   Value: 对角线填充值 (默认 1.0).
   支持高维批处理:如果传入 shape (b1 b2 ... n m),则创建批量单位矩阵."
  (let* ((rows n)
         (cols m)
         ;; 简单情况:2D 矩阵
         (shape (list rows cols))
         (res (make-vt shape 0 :type type))
         (data (vt-data res))
         (strides (vt-strides res)))    
    (declare (type (simple-array * (*)) data)
             (type list strides))    
    ;; 优化:直接计算对角线步长
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
    
    res))

(defun vt-diag (vector-tensor)
  "从 1D 张量 (向量) 创建方阵对角矩阵.
   输入: 形状为 的张量.
   输出: 形状为 的张量,非对角线为 0."
  (let* ((in-data (vt-data vector-tensor))
         (n (car (vt-shape vector-tensor)))
         (res (make-vt (list n n) 0))
         (res-data (vt-data res))
         (res-strides (vt-strides res)))    
    (declare (type fixnum n)
             (type (simple-array double-float (*)) in-data res-data))
    (let ((row-stride (first res-strides))
          (col-stride (second res-strides)))      
      (declare (type fixnum row-stride col-stride))
      ;; 填充对角线
      (loop for i fixnum from 0 below n
            for src-off fixnum from 0
            for dst-off fixnum = 0 then (+ dst-off row-stride col-stride)
            do (setf (aref res-data dst-off) (aref in-data src-off))))    
    res))

(defun vt-triu (tensor &key (k 0) (in-place nil))
  "返回上三角矩阵.
   K: 对角线偏移 (0=主对角线, 1=主对角线之上).
   In-Place: 如果为 T,直接修改原张量；否则返回副本.
   支持 Batch (Rank >= 2)."
  (let* ((res (if in-place tensor (vt-copy tensor))) ;; vt-copy 需自行实现或用 vt-map identity
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
			       0.0d0)))))               
               ;; === 高维递归 ===
               (let ((dim (nth depth res-shape))
                     (stride (nth depth res-strides)))
                 (loop for i fixnum from 0 below dim do
                   (recurse (1+ depth) ptr)
                   (incf ptr stride))))))      
      (recurse 0 (vt-offset res)))    
    res))

(defun vt-tril (tensor &key (k 0) (in-place nil))
  "返回下三角矩阵.
   K: 对角线偏移 (0=主对角线, -1=主对角线之下).
   支持 Batch."
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
			       0.0d0)))))               
               ;; === 高维递归 ===
               (let ((dim (nth depth res-shape))
                     (stride (nth depth res-strides)))
                 (loop for i fixnum from 0 below dim do
                   (recurse (1+ depth) ptr)
                   (incf ptr stride))))))      
      (recurse 0 (vt-offset res)))    
    res))

(defun vt-diagonal (tensor &key (offset 0))
  "提取对角线元素.
   返回一个 1D 张量 (向量),如果是 Batch 输入则返回 2D 张量.
   Offset: 对角线偏移."
  (let* ((in-shape (vt-shape tensor))
         (rank (length in-shape)))    
    (when (< rank 2) (error "vt-diagonal requires rank >= 2"))
    (let* ((rows (nth (- rank 2) in-shape))
           (cols (nth (- rank 1) in-shape))
           ;; 计算对角线长度
           (diag-len (max 0 (- (min rows cols) (abs offset))))
           ;; 确定输出形状:Batch 维度 + 对角线长度
           (out-shape
	     (append (subseq in-shape 0 (- rank 2))
		     (list diag-len)))
           (res (make-vt out-shape 0))
           (res-data (vt-data res))
           (in-data (vt-data tensor))
           (in-strides (vt-strides tensor)))      
      (declare (type (simple-array double-float (*)) in-data res-data))
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
      res)))

;; 1. 选用正确的步长计算函数
(defun vt-compute-strides (shape)
  "根据形状计算连续内存步长.
   Shape 为 nil (标量) -> 返回 nil.
   Shape 为 (10) -> 返回 (1)."
  (if (null shape)
      nil
      (loop with stride = 1
            for dim in (reverse shape)
            collect stride into strides
            do (setf stride (* stride dim))
            finally (return (reverse strides)))))

(defun vt-reshape (vt new-shape)
  "重塑形状.
   如果张量是连续的,则创建视图(零拷贝)；
   否则自动创建连续副本后重塑.
   这解决了原代码在非连续内存上重塑导致的数据错误问题."
  (let ((old-size (vt-shape-to-size (vt-shape vt)))
        (new-size (vt-shape-to-size new-shape)))
    (unless (= old-size new-size)
      (error "重塑失败:元素总数不一致 (旧: ~A, 新: ~A)" old-size new-size))    
    (if (vt-contiguous-p vt)
        ;; 零拷贝路径:直接复用数据
        (%make-vt
         :data (vt-data vt)
         :shape new-shape
         :strides (vt-compute-strides new-shape)
         :offset (vt-offset vt))
        ;; 安全路径:先内存整理,再重塑
        (let ((cont-vt (vt-contiguous vt)))
          (%make-vt
           :data (vt-data cont-vt)
           :shape new-shape
           :strides (vt-compute-strides new-shape)
           :offset 0)))))

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
     :offset (vt-offset vt))))

(defun vt-normalize-axis (axis rank)
  "将可能为负数的 axis 转换为严格正数，并进行越界检查。
   例如：rank=4, axis=-1 -> 3 ; axis=-2 -> 2"
  (if axis
      (let ((ax (if (minusp axis)
		    (+ axis rank)
		    axis)))
	(when (or (< ax 0) (>= ax rank))
	  (error "Axis ~A is out of bounds for tensor of rank ~A" axis rank))
	ax)))

(defun vt-narrow (vt axis start end)
  "零拷贝切片.沿指定轴切片,调整偏移量和形状. (等价于 PyTorch 的 narrow)"
  (let* ((shape (copy-list (vt-shape vt)))
         (rank (length shape))
         (ax (vt-normalize-axis axis rank)) 
         (strides (vt-strides vt))
         (dim-size (nth ax shape)))    
    (when (or (< start 0) (> end dim-size))
      (error "切片索引 [~A, ~A) 越界，当前轴大小为 ~A" start end dim-size))
    (let ((new-offset (+ (vt-offset vt)
                         (* start (nth ax strides))))
          (new-shape (progn (setf (nth ax shape) (- end start)) shape)))
      (%make-vt
       :data (vt-data vt)
       :shape new-shape
       :strides strides 
       :offset new-offset))))

(defun vt-split (tensor indices-or-sections &key (axis 0))
  "模仿 NumPy 的 split，支持负数 axis
   沿轴分割张量.
   indices-or-sections: 整数N(等分)或索引列表
   axis: 分割轴
   返回: 张量列表"
  (let* ((shape (vt-shape tensor))
         (rank (length shape))
         (ax (vt-normalize-axis axis rank))
         (dim-size (nth ax shape)))    
    (cond
      ;; ========== 情况 A：均分 ==========
      ((integerp indices-or-sections)
       (let* ((n indices-or-sections)
              (chunk-size (floor dim-size n)))
         (unless (zerop (rem dim-size n))
           (error "数组沿轴 ~A 的大小 ~A 不能被 ~A 整除" ax dim-size n))
         (loop for i from 0 below n
               collect (vt-narrow tensor ax 
                                  (* i chunk-size) 
                                  (* (1+ i) chunk-size)))))      
      ;; ========== 情况 B：指定位置下刀 ==========
      ((listp indices-or-sections)
       (let ((points (append (list 0) indices-or-sections (list dim-size))))
         (loop for (start end) on points by #'cdr
               while end
               collect (vt-narrow tensor ax start end))))      
      (t (error "indices-or-sections 必须是整数或整数列表")))))

(defun vt-slice (vt &rest specs)
  "通用切片函数 (NumPy风格,零拷贝).
   参数:
     vt: 张量
     specs: 切片规格列表,每个元素可以是:
       - 整数: 选取指定索引,该维度消失.
       - 列表: 切片,保留维度.
       - 关键字 :all 或 t: 选取整个维度.
       - nil: 同 :all.
   示例: 区间左闭右开
     ;; 1. 截取子矩阵 (行0-2, 列1-4)
     (vt-slice mat '(0 2) '(1 4))
     ;; 2. 提取特定行 (结果降为1维)
     (vt-slice mat 1)
     ;; 3. 提取特定列 (保留行维度)
     (vt-slice mat :all 2)
     ;; 4. 带步长切片
     (vt-slice mat '(0 10 2)) ; 每2行取一行
     ;; 5. 负索引支持
     (vt-slice mat '(-2 -1)) ; 最后两行"
  
  (let* ((old-shape (vt-shape vt))
         (old-strides (vt-strides vt))
         (old-offset (vt-offset vt))
         (old-rank (length old-shape))
         (n-specs (length specs)))    
    ;; 1. 参数预处理:如果 specs 少于维度,用 :all 补齐
    (when (< n-specs old-rank)
      (setf specs (append specs (make-list (- old-rank n-specs)
					   :initial-element :all))))    
    (when (> (length specs) old-rank)
      (error "切片参数过多: ~A > 张量秩 ~A" (length specs) old-rank))    
    (let ((new-shape '())
          (new-strides '())
          (cur-offset old-offset))      
      ;; 2. 遍历每个维度规格
      (loop for spec in specs
            for dim in old-shape
            for stride in old-strides
            do (cond
		 ;; --- 情况 A: 整数索引 (降维) ---
		 ((integerp spec)
		  (let ((idx spec))
                    ;; 处理负索引
                    (when (< idx 0) (incf idx dim))
                    (unless (and (>= idx 0) (< idx dim))
                      (error "索引 ~A 越界 (维度大小: ~A)" spec dim))
                    ;; 仅移动偏移量,不添加到 shape/strides
                    (incf cur-offset (* idx stride))))
		 ;; --- 情况 B: 全选 ---
		 ((member spec '(:all t nil))
		  (push dim new-shape)
		  (push stride new-strides))              
		 ;; --- 情况 C: 列表切片 ---
		 ((and (listp spec) (<= 2 (length spec) 3))
		  (destructuring-bind (start end &optional (step 1)) spec
                    ;; 1. 规范化负索引
                    (when (< start 0) (incf start dim))
                    (when (< end 0) (incf end dim))                 
                    ;; 2. 边界钳位 (类似 NumPy)
                    ;; start: clamp(0, dim)
                    ;; end: clamp(0, dim)
                    ;; 注意:如果 start > end 且 step > 0,切片为空
                    (setf start (max 0 (min start dim)))
                    (setf end (max 0 (min end dim)))
                    ;; 3. 计算新维度大小和新偏移量
                    (let ((slice-dim 0))
                      (cond
			;; 正向步长
			((> step 0)
			 (when (< start end)
                           ;; 长度 = ceil((end - start) / step)
                           (setf slice-dim
				 (floor (+ (- end start) step -1) step))))
			;; 反向步长
			((< step 0)
			 ;; 允许反向切片,例如 (5 -1 -1) 或 (5 0 -1)
			 ;; 调整边界逻辑以允许从大到小
			 ;; NumPy 处理 start/end 的方式较复杂,这里简化处理:
			 ;; 假设用户给出的 start/end 在逻辑路径上是有效的
			 ;; 简单起见,若 step < 0,我们计算元素个数
			 ;; count = floor((start - end - 1) / abs(step))
			 (when (> start end)
                           (setf slice-dim (floor (+ (- start end)
						     (abs step) -1)
						  (abs step)))))
			(t (error "步长不能为0")))
                      ;; 更新状态
                      (incf cur-offset (* start stride))
                      (push slice-dim new-shape)
                      (push (* stride step) new-strides))))
		 (t (error "无效的切片规格: ~A" spec))))
      
      ;; 3. 构建结果
      (%make-vt :data (vt-data vt)
                :shape (nreverse new-shape)
                :strides (nreverse new-strides)
                :offset cur-offset))))

(defun (setf vt-slice) (value vt &rest specs)
  "设置切片区域的值.
   value: 可以是数字或另一个张量.
   specs: 切片参数 (同 vt-slice).
   示例:
     ;; 将第一行全部置 0
     (setf (vt-slice m 0 :all) 0.0)
     ;; 将子矩阵赋值为另一个张量
     (setf (vt-slice m '(0 2) '(0 2)) another-matrix)"
  ;; 1. 获取目标切片的视图
  ;; apply 调用我们之前实现的 vt-slice
  (let ((target-view (apply #'vt-slice vt specs)))
    ;; 2. 执行拷贝 (自动处理标量->广播 或 张量->形状匹配)
    (vt-copy-into target-view value)))

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
                      :offset 0)))
        ;; 高效拷贝:利用迭代器填充新数据
        (vt-copy-into new-vt vt)
        new-vt)))

(defun vt-copy (vt)
  "深度拷贝张量.
   返回一个全新的、内存连续的张量副本.
   副本与原张量完全独立,修改副本不会影响原张量.
   自动处理视图:如果原张量是切片或转置,返回的是落地后的连续数据."
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
                           :offset 0)))
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
    new-vt))

(defun vt-ref (vt &rest indices)
  "获取张量元素(支持高维索引)."
  (let* ((strides (vt-strides vt))
         (data (vt-data vt))
         (base-offset (vt-offset vt))
         (idx-sum (loop for idx in indices
                        for stride in strides
                        sum (* idx stride))))
    (aref data (+ base-offset idx-sum))))

(defun (setf vt-ref) (value vt &rest indices)
  "设置张量元素."
  (let* ((strides (vt-strides vt))
         (data (vt-data vt))
         (base-offset (vt-offset vt))
         (flat-idx (+ base-offset
                      (loop for idx in indices
                            for stride in strides
                            sum (* idx stride)))))
    (setf (aref data flat-idx)
	  (coerce value (vt-element-type vt)))))

;;; --- 高效迭代逻辑  ---
(defmacro vt-do-each ((ptr-var val-var vt) &body body)
  "高效遍历宏。
   ptr-var: 当前元素的物理索引。
   val-var: 当前元素的值。"
  (let ((rank-sym (gensym "RANK"))
        (shape-sym (gensym "SHAPE"))
        (strides-sym (gensym "STRIDES"))
        (data-sym (gensym "DATA"))
        (offset-sym (gensym "OFFSET"))
        (depth-sym (gensym "DEPTH")))
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
         (recurse 0 ,offset-sym)))))

(defun vt-copy-into (dest src)
  "将 SRC 的数据拷贝到 DEST (支持广播和类型转换).
   SRC: 源张量或标量数字.
   DEST: 目标张量视图."  
  ;; === 支持标量输入
  (when (numberp src)
    (setf src (vt-from-sequence (list src))))  
  (let ((dest-shape (vt-shape dest))
        (src-shape (vt-shape src)))
    ;; 确保 Dest 的形状能够容纳 Src 广播后的形状
    (let ((final-shape (vt-broadcast-shapes dest-shape src-shape)))
      (unless (equal final-shape dest-shape)
        (error "Copy-Into 失败: Dest 形状 ~A 无法容纳 Src 广播后的形状 ~A" 
               dest-shape final-shape))      
      ;; 准备数据
      (let* ((dest-data (vt-data dest))
             (src-data (vt-data src))
             ;; 使用辅助函数计算广播步长 逻辑更清晰
             (src-strides (vt-broadcast-strides 
                           src-shape dest-shape (vt-strides src)))
             (dest-strides (vt-strides dest))
             (dest-offset (vt-offset dest))
             (src-offset (vt-offset src))
             (rank (length dest-shape))
             (element-type (array-element-type dest-data)))        
        (labels
	    ((recurse (depth d-ptr s-ptr)
               (declare (type fixnum depth d-ptr s-ptr))
               (if (= depth rank)
                   ;; 叶节点:写入并进行类型转换
                   (setf (aref dest-data d-ptr)
                         (coerce (aref src-data s-ptr) element-type))                   
                   ;; 递归层
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
        dest))))

;; 快速类型转换
(declaim (inline ensure-vt))
(defun ensure-vt (obj)
  (etypecase obj
    (vt obj)
    (number 
     (let ((val (coerce obj 'double-float)))
       (%make-vt :data (make-array 1 :initial-element val
				     :element-type 'double-float)
                 :shape nil :strides nil :offset 0)))
    (sequence (vt-from-sequence obj))))

(declaim (inline vt-broadcast-shapes))
(defun vt-broadcast-shapes (shape1 shape2)
  "计算广播结果形状."
  (declare (list shape1 shape2)
	   (optimize (speed 3)))
  (let ((len1 (length shape1))
        (len2 (length shape2))
        (result '()))
    (declare (fixnum len1 len2))
    (loop for i fixnum from 1 to (max len1 len2)
          for dim1 fixnum = (if (<= i len1) (nth (- len1 i) shape1) 1)
          for dim2 fixnum = (if (<= i len2) (nth (- len2 i) shape2) 1)
          do (cond ((= dim1 dim2) (push dim1 result))
                   ((or (= dim1 1) (= dim2 1))
		    (push (max dim1 dim2) result))
                   (t (error "形状 ~A 和 ~A 无法进行广播" shape1 shape2))))
    (the list result)))


(declaim (inline vt-broadcast-strides))
(defun vt-broadcast-strides (orig-shape target-shape orig-strides)
  "计算广播后的步长.总是返回列表."
  (declare (list orig-shape target-shape orig-strides))
  (declare (optimize (speed 3)))
  (let ((rank-diff (- (length target-shape) (length orig-shape))))
    (declare (fixnum rank-diff))
    (loop for i fixnum from 0 below (length target-shape)
          ;; 规则:如果该维度是广播出来的(前面补0),或者原维度为1(被广播),
          ;; 步长为0.否则使用原步长.
          collect (cond
                    ;; 1. 原维度不够长,前面补的维度,步长为 0
                    ((< i rank-diff) 0)
                    (t
                     (let* ((orig-idx (- i rank-diff))
                            (orig-dim (nth orig-idx orig-shape)))
		       (declare (fixnum orig-dim orig-idx))
                       ;; 2. 原维度为1,被广播扩展,步长为 0
                       (if (= orig-dim 1)
                           0
                           ;; 3. 正常维度,使用原步长
                           (nth orig-idx orig-strides))))))))

;; 核心宏:高性能广播迭代 
(defmacro with-broadcasting-ptrs ((t1 t2 result-vt) &body body)
  "高性能广播宏.
   修复了指针迭代逻辑,确保指针随着循环正确递增."
  (let ((idx-var (gensym "IDX"))
        (rank-var (gensym "RANK"))
        (res-shape (gensym "RES-SHAPE"))
        ;; 指针变量
        (p1 (gensym "P1"))
        (p2 (gensym "P2"))
        (pr (gensym "PR"))
        ;; 步长变量
        (s1 (gensym "S1"))
        (s2 (gensym "S2"))
        (sr (gensym "SR"))
        (dim (gensym "DIM"))
        (data1 (gensym "DATA1"))
        (data2 (gensym "DATA2"))
        (data-res (gensym "DATA-RES")))
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
                  (vt-offset ,result-vt))))))

(defun vt-map (fn &rest tensors)
  "高效映射函数.
   支持标量、列表、张量混合输入.
   特性:自动类型转换、广播、指针算术优化、双目运算内联."
  (declare (function fn))
  ;; 1. [无缝操作] 统一转换输入为 VT
  (let* ((clean-tensors (mapcar #'ensure-vt tensors))
         (n-tensors (length clean-tensors)))    
    (when (= n-tensors 0) (error "vt-map requires at least one tensor"))
    (let* (;; 2. 确定输出形状 (广播合并)
           (out-shape (reduce #'vt-broadcast-shapes
			      (mapcar #'vt-shape clean-tensors)))
           (rank (length out-shape))           
           ;; 3. 分配结果张量, 统一返回double-float类型
           (res (vt-zeros out-shape))
           (res-data (vt-data res))
           (res-strides (vt-strides res))           
           ;; 4. 准备输入张量的底层数据、偏移量与广播后的步长
           (ins-data (map 'vector #'vt-data clean-tensors))
           (ins-offs (map 'vector #'vt-offset clean-tensors))
           (ins-strides (map 'vector 
                             (lambda (vt) 
                               (vt-broadcast-strides
				(vt-shape vt) out-shape (vt-strides vt)))
                             clean-tensors))
           
           ;; 5. 迭代器状态:当前指针数组 (用于可变状态遍历)
           (cur-ptrs (make-array n-tensors :element-type
				 'fixnum :initial-element 0)))
      (declare (type (simple-array double-float (*)) res-data)
               (type (simple-array t (*)) ins-data)
               (type (simple-array fixnum (*)) cur-ptrs)
               (type list res-strides out-shape)
               (type fixnum rank n-tensors))      
      ;; 初始化指针
      (loop for i fixnum from 0 below n-tensors do
        (setf (aref cur-ptrs i) (aref ins-offs i)))      
      ;; 递归迭代核心
      (labels
	  ((recurse (depth out-ptr)
             (declare (type fixnum depth out-ptr))
             (if (= depth rank)
                 ;; --- 最内层:执行计算 ---
                 (let ((d0 (aref (aref ins-data 0) 
                                 (aref cur-ptrs 0))))
                   (setf (aref res-data out-ptr)
                         (the double-float
                              (case n-tensors
                                (1 (coerce (funcall fn d0) 'double-float))
                                (2 (let ((d1 (aref (aref ins-data 1) 
                                                   (aref cur-ptrs 1))))
                                     (coerce (funcall fn d0 d1)
					     'double-float)))
                                (otherwise 
                                 (let ((args
					 (loop
					   for k fixnum from 0 below n-tensors
                                           collect
					   (aref (aref ins-data k)
                                                 (aref cur-ptrs k)))))
                                   (declare (dynamic-extent args))
                                   (coerce (apply fn args)
					   'double-float)))))))                 
                 ;; --- 外层:遍历维度 ---
                 (let* ((dim (the fixnum (nth depth out-shape)))
                        (res-stride (the fixnum (nth depth res-strides)))
                        (strides-at-depth 
                          (loop for k fixnum from 0 below n-tensors
                                collect
				(the fixnum
				     (nth depth (aref ins-strides k))))))
                   (declare (type fixnum dim res-stride)
                            (dynamic-extent strides-at-depth))                   
                   (loop for i fixnum from 0 below dim do
                     (recurse (1+ depth) out-ptr)                     
                     ;; 更新指针 (移动一个步长)
                     (incf out-ptr res-stride)
                     (loop for k fixnum from 0 below n-tensors
                           for s fixnum in strides-at-depth do
                             (incf (aref cur-ptrs k) s)))
                   ;; 指针回溯 (为了上层循环的正确性)
                   (decf out-ptr (the fixnum (* dim res-stride)))
                   (loop for k fixnum from 0 below n-tensors
                         for s fixnum in strides-at-depth do
                           (decf (aref cur-ptrs k)
				 (the fixnum (* dim s))))))))
        (recurse 0 0))
      res)))

(defun vt-ensure-shape-compatible (shape axis)
  "检查 axis 是否合法. 支持负数轴."
  (let* ((rank (length shape))
         (real-axis (if (< axis 0) (+ rank axis) axis)))
    (when (or (< real-axis 0) (>= real-axis rank))
      (error "Axis ~A 越界,张量秩为 ~A" axis rank))))

(defun vt-compute-logical-strides (shape)
  "计算逻辑步长(用于计算扁平化索引).
   例如 shape (3 4) -> strides (4 1).
   这意味着第0轴每走一步,线性索引+4；第1轴每走一步,线性索引+1."
  (if (null shape)
      nil
      (loop with rank = (length shape)
            with strides = (make-list rank)
            for i from (1- rank) downto 0
            for acc = 1 then (* acc (nth (1+ i) shape))
            do (setf (nth i strides) acc)
            finally (return strides))))

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
   2. 自动适配输入张量的数据类型.
   3. 强制类型转换保证不会因类型不匹配报错."
  (declare (type vt tensor)
           (type (or null fixnum) axis)
           (type function reducer-fn))  
  (let* ((in-shape (vt-shape tensor))
         ;; 关键:获取实际类型
         (element-type (vt-element-type tensor))
         (rank (length in-shape))
	 (real-axis (vt-normalize-axis axis rank))
         (is-global-reduction (null real-axis))
	 ;; 1. 输出形状
	 (out-shape
	   (cond
             ;; 全局归约 且 不保留维度 -> nil (标量)
             ((and is-global-reduction (not keepdims)) nil)
             ;; 全局归约 且 保留维度 -> 例如 (3,4) 变 (1,1)
             (is-global-reduction (make-list rank :initial-element 1))
             ;; 轴向归约 且 不保留维度 -> 剔除该轴 (原逻辑)
             ((not keepdims)
	      (loop for d in in-shape for i from 0 unless (= i real-axis)
		    collect d))
             ;; 轴向归约 且 保留维度 -> 该轴置为 1，例如 (3,4) real-axis=1 变 (3,1)
             (t (loop for d in in-shape for i from 0
		      collect (if (= i real-axis) 1 d)))))
	 ;; 2. 初始值安全转换
	 (safe-init-val (coerce init-val element-type))           
	 ;; 3. 分配结果张量
	 (res (make-vt out-shape safe-init-val :type element-type))
	 (res-data (vt-data res))
	 (res-offset (vt-offset res))           
	 ;; 4. 索引张量 (类型始终为 fixnum)
	 (res-idx (when return-arg (make-vt out-shape 0 :type 'fixnum)))
	 (res-idx-data (when res-idx (vt-data res-idx)))
	 ;; 5. 输入属性
	 (in-data (vt-data tensor))
	 (in-strides (vt-strides tensor))
	 (in-offset (vt-offset tensor))           
	 ;; 6. 步长映射 (保持为列表,不做数组特化)
	 (out-strides-map
	   (if is-global-reduction
               (make-list rank :initial-element 0)
               ;; 轴向归约：根据 keepdims 决定索引偏移
	       ;; keepdims=T: 形状没少，输入第 i 轴直接对应输出第 i 轴
	       ;; keepdims=nil: 形状少了一维，跨过归约轴后要 -1
               (loop
		 for i from 0 below rank
                 if (= i real-axis) collect 0
                   else collect
			(let ((out-idx
				(if keepdims                                  
                                    i                                  
                                    (if (< i real-axis)
					i
					(1- i)))))
                          (nth out-idx (vt-strides res))))))
	 ;; 索引步长 (用于计算 argmax/argmin 的逻辑位置)
	 (arg-strides
	   (if return-arg 
               (if is-global-reduction
		   (vt-compute-strides in-shape)
		   (loop for i from 0 below rank
			 if (= i real-axis)
			   collect 1
			 else collect 0))
               (make-list rank :initial-element 0))))      
    (declare (type (simple-array * (*)) in-data res-data)
             (type list out-strides-map in-strides arg-strides in-shape))
    (labels
	((recurse (depth in-ptr out-ptr out-idx-ptr current-arg-val)
           (declare (type fixnum depth in-ptr out-ptr out-idx-ptr
			  current-arg-val))           
           (if (= depth rank)
               ;; --- 叶节点:执行归约 ---
               (let ((val (aref in-data in-ptr))
                     (cur-acc (aref res-data out-ptr)))
                 (multiple-value-bind (new-acc do-update-idx) 
                     (funcall reducer-fn cur-acc val)
                   ;; 写入前强制转换回原类型
                   ;; 防止 fixnum + fixnum -> overflow 或比较函数返回非预期类型
                   (setf (aref res-data out-ptr) 
                         (coerce new-acc element-type))                   
                   (when (and return-arg do-update-idx res-idx-data)
                     (setf (aref res-idx-data out-idx-ptr)
			   current-arg-val))))               
               ;; --- 分支节点:列表遍历 ---
               (let* ((dim (nth depth in-shape))
                      (in-stride (nth depth in-strides))
                      (out-stride (nth depth out-strides-map))
                      (arg-stride (nth depth arg-strides)))
                 (declare
		  (type fixnum dim in-stride out-stride arg-stride))
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
      (recurse 0
	       in-offset
	       res-offset 
               (if res-idx (vt-offset res-idx) 0) 
               0))
    (if (and is-global-reduction (not keepdims))
        (let ((final-val (aref res-data res-offset)))
          (if return-arg
	      (values final-val (aref res-idx-data (vt-offset res-idx)))
	      final-val))
        ;; 其余情况（轴向归约，或 keepdims=T），均返回 VT 结构体
        (values res res-idx))))

(defun vt-matmul (vt1 vt2)
  (cond ((= 2
	    (vt-order vt1)
	    (vt-order vt2))
	 (vt-matmul-df vt1 vt2))
	(t 
	 (vt-einsum "...ij,...jk->...ik" vt1 vt2))))

(defun vt-@ (vt1 vt2)
  (vt-matmul vt1 vt2))

(defun vt-= (t1 t2)
  "逐元素相等比较.
   支持广播.返回布尔张量(1.0或0.0)."
  (let* ((result-element-type (vt-element-type t1))
	 (true-result (coerce 1 result-element-type))
	 (false-result (coerce 0 result-element-type)))
    (vt-map (lambda (a b)
	      (if (= a b) true-result false-result))
	    t1 t2)))

(defun vt-/= (t1 t2)
  "逐元素不相等比较.
   支持广播.返回布尔张量(1.0或0.0)."
  (let* ((result-element-type (vt-element-type t1))
	 (true-result (coerce 1 result-element-type))
	 (false-result (coerce 0 result-element-type)))
    (vt-map (lambda (a b)
	      (if (/= a b) true-result false-result))
	    t1 t2)))

(defun vt-< (t1 t2)
  "逐元素小于比较.
   支持广播.返回布尔张量(1.0或0.0)."
  (let* ((result-element-type (vt-element-type t1))
	 (true-result (coerce 1 result-element-type))
	 (false-result (coerce 0 result-element-type)))
    (vt-map (lambda (a b)
	      (if (< a b) true-result false-result))
	    t1 t2)))

(defun vt-<= (t1 t2)
  "逐元素小于等于比较.
   支持广播.返回布尔张量(1.0或0.0)."
  (let* ((result-element-type (vt-element-type t1))
	 (true-result (coerce 1 result-element-type))
	 (false-result (coerce 0 result-element-type)))
    (vt-map (lambda (a b)
	      (if (<= a b) true-result false-result))
	    t1 t2)))

(defun vt-> (t1 t2)
  "逐元素大于比较.
   支持广播.返回布尔张量(1.0或0.0)."
  (let* ((result-element-type (vt-element-type t1))
	 (true-result (coerce 1 result-element-type))
	 (false-result (coerce 0 result-element-type)))
    (vt-map (lambda (a b)
	      (if (> a b) true-result false-result))
	    t1 t2)))

(defun vt->= (t1 t2)
  "逐元素大于等于比较.
   支持广播.返回布尔张量(1.0或0.0)."
  (let* ((result-element-type (vt-element-type t1))
	 (true-result (coerce 1 result-element-type))
	 (false-result (coerce 0 result-element-type)))
    (vt-map (lambda (a b)
	      (if (>= a b) true-result false-result))
	    t1 t2)))

(defun vt-where (condition &optional x y)
  "NumPy 风格的 Where 函数.
   模式 1 (查找索引): (vt-where condition)
     返回满足条件的索引.返回一个张量列表,每个张量对应一个维度的索引.
     示例: (vt-where cond) -> (list rows-tensor cols-tensor ...)
   模式 2 (三元选择): (vt-where condition x y)
     根据 condition 从 x 或 y 中选择元素.支持广播.
     示例: (vt-where cond t1 t2) -> 新张量"
  (cond
    ;; === 模式 2: 三元选择 (condition, x, y) ===
    ((and x y)
     (vt-map (lambda (c val-x val-y)
               (if (/= c 0) val-x val-y))
             condition x y))    
    ;; === 模式 1: 查找索引 ===
    ((and (not x) (not y))
     (let* ((shape (vt-shape condition))
            (rank (length shape))
            (data (vt-data condition))
            (strides (vt-strides condition))
            (offset (vt-offset condition))
            ;; 为每个维度准备一个动态数组收集索引
            (buffers
	      (loop for i from 0 below rank
                    collect (make-array 0 :element-type 'fixnum 
                                          :fill-pointer t 
                                          :adjustable t))))
       
       (declare (type (simple-array double-float (*)) data)
                (type list shape strides buffers))       
       (labels
	   ((recurse (depth current-ptr coords)
              (declare (type fixnum depth current-ptr))
              (if (= depth rank)
                  ;; 叶节点:如果非零,记录坐标
                  (when (/= (aref data current-ptr) 0.0d0)
                    (loop for c in coords
                          for buf in buffers
                          do (vector-push-extend c buf)))                      
                  ;; 递归层
                  (let ((dim (nth depth shape))
                        (stride (nth depth strides)))
                    (declare (type fixnum dim stride))
                    (loop for i fixnum from 0 below dim do
                      (recurse (1+ depth)
                               (+ current-ptr (* i stride))
                               (append coords (list i))))))))
         (recurse 0 offset nil))
       ;; 将收集到的索引数组转换为张量列表
       (loop for buf in buffers
             collect (let* ((len (length buf))
                            (final-data (make-array
					 len :element-type 'fixnum)))
                       (loop for i from 0 below len
                             for idx across buf
                             do (setf (aref final-data i) idx))
                       (%make-vt :data final-data
                                 :shape (list len)
                                 :strides '(1)
                                 :offset 0)))))    
    (t (error "VT-WHERE 参数错误: 必须只传入 condition,或者同时传入 condition, x, y"))))


(defun vt-argwhere (condition)
  "查找非零元素的坐标.
   condition: 条件张量.
   返回: 形状为 (N, Rank) 的二维张量,每一行是一个非零元素的完整坐标.
   示例: (vt-argwhere tensor) -> [[0, 1], [2, 3], ...]"
  (declare (type vt condition))  
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
            ;; 返回 (N, Rank) 形状的张量
            (%make-vt :data final-data
                      :shape (list count rank)
                      :strides (list rank 1) ; 连续内存
                      :offset 0))))))
