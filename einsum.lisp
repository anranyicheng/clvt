(in-package :clvt)

;; Einsum 引擎 (适配 VT 结构与视图)

(declaim (inline parse-subscript-tokens))
(defun parse-subscript-tokens (str)
  "将下标字符串解析为 Token 列表.
   支持:
   1. 标准字符 (如 ij, jk)
   2. 省略号 ... (解析为 :ELLIPSIS)
   3. 显式输出 (->)
   4. 隐式输出 (无 ->)
   返回:
   1. inputs: 输入下标列表的列表
   2. output: 输出下标列表
   3. explicit-p: 是否显式指定输出"
  
  (let ((len (length str))
        (inputs nil)          ;; 存储所有输入张量的下标列表
        (current-sub nil)     ;; 当前正在构建的下标列表
        (output nil)          ;; 输出下标列表
        (i 0)
        (state :inputs))      ;; 状态机:当前处于输入部分还是输出部分
    (flet ((save-current-sub ()
             "将当前累积的 current-sub 保存到 inputs 或 output 中"
             (when current-sub
               ;; current-sub 是反序构建的,需要反转
               (let ((final-sub (nreverse current-sub)))
                 (if (eq state :inputs)
                     (push final-sub inputs)
                     (setf output final-sub)))
               (setf current-sub nil))))      
      (loop while (< i len) do
        (let ((char (char str i)))
          (cond
            ;; 1. 处理省略号 "..."
            ((and (char= char #\.)
                  (< (+ i 2) len)
                  (char= (char str (+ i 1)) #\.)
                  (char= (char str (+ i 2)) #\.))
             ;; 将 :ellipsis 标记加入当前列表
             (push :ellipsis current-sub)
             (incf i 3))
            ;; 2. 处理箭头 "->"
            ((and (char= char #\-)
                  (< (1+ i) len)
                  (char= (char str (1+ i)) #\>))
             ;; 遇到箭头,意味着:
             ;; A. 结束最后一个输入张量的解析
             (save-current-sub)
             ;; B. 切换状态到输出模式
             (setf state :outputs)
             (incf i 2))
            ;; 3. 处理逗号 ","
            ((char= char #\,)
             ;; 只有在输入模式下,逗号才作为分隔符
             (when (eq state :inputs)
               (save-current-sub))
             ;; 如果在输出模式下遇到逗号,通常是语法错误,
             ;; 但为了兼容性或特定扩展,这里暂时忽略或报错
             ;; NumPy 标准输出不支持逗号(只有一个输出)
             (incf i))
            ;; 4. 处理空格
            ((char= char #\Space)
             (incf i))
            ;; 5. 错误检查:单点 "."
            ((char= char #\.)
             (error "Invalid syntax: single '.' found. Use '...' for ellipsis."))
            ;; 6. 处理普通字符 (a-z, A-Z)
            (t (push char current-sub)
               (incf i)))))
      
      ;; 循环结束后,处理最后剩余的 current-sub
      (save-current-sub)
      ;; 返回结果
      ;; inputs 是反序压栈的,需要反转
      ;; output 不需要反转(只有一个)
      (values (nreverse inputs) 
              output 
              (if (eq state :outputs) t nil)))))

(declaim (inline expand-ellipsis))
(defun expand-ellipsis (input-subs output-sub vts)
  "将解析出的 :ellipsis."
  ;; 1. 计算维度
  (let ((ellipsis-ranks nil))
    (loop for sub in input-subs
          for vt in vts
          for explicit-rank = (count-if #'characterp sub)
          for has-ellipsis = (member :ellipsis sub)
          for tensor-rank = (length (vt-shape vt))
          for implicit-rank = (if has-ellipsis (- tensor-rank explicit-rank) 0) do
	    (when (and has-ellipsis (< implicit-rank 0))
              (error "Subscript dimension mismatch"))
	    (when (> (count :ellipsis sub) 1)
              (error "Only one ellipsis allowed"))
	    (push implicit-rank ellipsis-ranks))
    
    (setf ellipsis-ranks (nreverse ellipsis-ranks))    
    ;; 2. 生成标签
    (let* ((max-implicit-rank (reduce #'max ellipsis-ranks :initial-value 0))
           ;; 使用特殊的内部字符,避免与用户输入冲突
           (ellipsis-labels (loop for i from 0 below max-implicit-rank
                                  collect (code-char (+ (char-code #\?) i)))))      
      (flet ((expand-sub (sub implicit-rank)
               "高效展开:使用 nconc 代替 append,避免额外拷贝"
               (let ((pos (position :ellipsis sub)))
                 (if (not pos)
                     sub
                     ;; 优化点:直接拼接列表结构
                     (let* ((before (subseq sub 0 pos))
                            (after (nthcdr (1+ pos) sub)) ; nthcdr 不创建新列表,只是移动指针
                            (labels-to-use (subseq ellipsis-labels 
                                                   (- max-implicit-rank implicit-rank))))
                       ;; nconc 是破坏性的,但因为 before 是 subseq 新建的,所以安全
                       (nconc before labels-to-use after))))))        
        (values (mapcar #'expand-sub input-subs ellipsis-ranks)
                (when output-sub
                  (expand-sub output-sub max-implicit-rank)))))))

;; 智能重排:优化缓存命中率
(declaim (inline smart-reorder-labels))
(defun smart-reorder-labels (raw-labels output-subscripts sum-labels)
  (declare (type list raw-labels output-subscripts sum-labels)
	   (optimize (speed 3)))
  (if (and (= (length output-subscripts) 2) (= (length sum-labels) 1))
      (let ((dim-i (first output-subscripts))
            (dim-k (second output-subscripts))
            (dim-j (first sum-labels)))
        (list dim-i dim-j dim-k))
      raw-labels))

;; 构建视图感知的步长映射
(declaim (inline build-view-strides))
(defun build-view-strides (vt subscript all-labels label-dims)
  "将 VT 的物理步长映射到全局标签顺序,处理广播和视图."
  (let* ((phys-shape (vt-shape vt))
         (phys-strides (vt-strides vt))
         (strides (make-list (length all-labels) :initial-element 0)))
    (declare (type list strides phys-shape phys-strides))
    ;; 遍历下标,建立 标签 -> 物理步长 的映射
    (loop for label in subscript
          for dim in phys-shape
          for stride in phys-strides
          for pos = (position label all-labels :test #'char=)
          do (let ((res-dim (gethash label label-dims 0)))
               ;; 广播逻辑:如果物理维度为1但结果维度>1,步长强制为0
               ;; 否则使用物理步长 (支持非连续视图)
               (when (and (= dim 1) (> res-dim 1))
                 (setf stride 0))
               ;; 支持对角线:如果标签重复,步长累加
               (incf (nth pos strides) stride)))
    strides))

(declaim (inline analyze-einsum))
(defun analyze-einsum (input-subscripts output-subscripts explicit-mode vts)
  (declare (type list input-subscripts vts)
	   (optimize (speed 1) (safety 1)))
  (let ((label-dims (make-hash-table))
        (label-counts (make-hash-table))
        (all-labels-set nil))    
    ;; 1. 收集维度信息
    (loop for sub in input-subscripts
          for tns in vts
          for shape = (vt-shape tns)
          do (unless (= (length sub) (length shape))
               (error "子脚标 ~A 与张量形状 ~A 维度不匹配" sub shape))
             (loop for label in sub
                   for dim in shape
                   do (incf (gethash label label-counts 0))
                      (pushnew label all-labels-set)
                      (multiple-value-bind (old-dim found-p)
			  (gethash label label-dims)
			(if found-p
                            ;; 维度一致性检查 (允许广播: 1扩N)
                            (unless (or (= old-dim dim) (= dim 1) (= old-dim 1))
                              (error "标签 ~A 维度冲突: ~A vs ~A" label old-dim dim))
                            (setf (gethash label label-dims) dim))
			;; 更新 label-dims 为广播后的最大维度
			(setf (gethash label label-dims)
			      (max (gethash label label-dims 0) dim)))))

    ;; 2. 确定输出下标
    (when (not explicit-mode)
      (setf output-subscripts
            (sort (loop for label being the hash-key of label-counts
                        when (= (gethash label label-counts) 1)
                          collect label)
                  #'char<)))

    (let* ((sum-labels (set-difference all-labels-set output-subscripts))
           (raw-labels (append output-subscripts sum-labels))
           (all-labels (smart-reorder-labels
			raw-labels output-subscripts sum-labels)))
      
      ;; 3. 构建步长矩阵 
      (let ((vt-global-strides
	      (loop for sub in input-subscripts
                    for tns in vts
                    collect
		    (build-view-strides tns sub all-labels label-dims))))
        (values all-labels label-dims vt-global-strides output-subscripts)))))

(declaim (inline build-dim-vector))
(defun build-dim-vector (labels dim-hash)
  (declare (type list labels)
	   (type hash-table dim-hash)
	   (optimize (speed 3) (safety 1)))
  (let ((vec (make-array (length labels) :element-type 'fixnum)))
    (loop for label in labels for i from 0 do
      (setf (aref vec i) (the fixnum (gethash label dim-hash 0))))
    vec))

(declaim (inline build-strides-matrix))
(defun build-strides-matrix (all-labels vts vt-global-strides)
  (declare (type list vts all-labels vt-global-strides)
	   (optimize (speed 1) (safety 1)))
  (let* ((n-vts (length vts))
         (n-labels (length all-labels))
         (res (make-array n-vts :element-type t)))
    (loop for strides in vt-global-strides for t-idx from 0 do
      (let ((vec (make-array n-labels :element-type 'fixnum)))
        (loop for s in strides for l-idx from 0 do
	  (setf (aref vec l-idx) (the fixnum s)))
        (setf (aref res t-idx) vec)))
    res))

(declaim (inline build-output-strides))
(defun build-output-strides (output-subscripts all-labels label-dims)
  "计算输出张量的步长(输出总是连续的)."
  (let ((local-stride-map (make-hash-table))
        (acc 1))
    (declare (type fixnum acc))
    ;; 反向计算连续步长
    (loop for label in (reverse all-labels)
          for dim = (gethash label label-dims 0)
          when (member label output-subscripts) ; 只计算输出维度
            do (setf (gethash label local-stride-map) acc)
               (setf acc (the fixnum (* acc dim))))
    ;; 映射到 all-labels 顺序
    (loop for label in all-labels
          collect (if (member label output-subscripts)
                      (gethash label local-stride-map 0)
                      0))))

(declaim (inline einsum-execute-fast))
(defun einsum-execute-fast
    (all-labels dim-sizes vt-global-strides output-subscripts vts)
  (declare (type list all-labels output-subscripts vts)
           (type hash-table dim-sizes)
           (optimize (speed 1) (safety 1) (debug 1)))
  
  (let* ((rank (length all-labels))
         (n-vts (length vts))
         (dims-vec (build-dim-vector all-labels dim-sizes))
         (strides-mat
	   (build-strides-matrix all-labels vts vt-global-strides))
         
         ;; 输出张量创建
         (out-shape (mapcar (lambda (l)
			      (gethash l dim-sizes))
			    output-subscripts))
         (output (vt-zeros out-shape)) ; 默认 double-float
         (out-data (vt-data output))
         
         ;; 输出步长 (连续)
         (out-strides-list (build-output-strides
			    output-subscripts all-labels dim-sizes))
         (out-strides-vec (make-array rank :element-type 'fixnum))
         
         ;; 输入数据与偏移
         (in-data-vec (map 'vector #'vt-data vts))
         (offsets-vec (map 'vector #'vt-offset vts)))

    (declare (type (simple-array fixnum (*)) dims-vec out-strides-vec)
             (type (simple-array (simple-array fixnum (*)) (*)) strides-mat)
             (type (simple-array double-float (*)) out-data) ;; 假设 double-float
             (type (simple-array t (*)) in-data-vec offsets-vec))
    
    (loop for s in out-strides-list for i from 0 do
      (setf (aref out-strides-vec i) s))

    ;; === 优化路径:矩阵乘法 (I, J, K) ===
    ;; 仅当 2 个输入,秩为 3 时启用
    (if (and (= n-vts 2) (= rank 3))
        (let ((labels-vec (coerce all-labels 'vector)))
          (declare (type (simple-array t (*)) labels-vec)
		   (ignorable labels-vec))
          ;; 动态查找 I, J, K 位置,确保通用性
          (let* ((label-i (first output-subscripts))
                 (label-k (second output-subscripts))
                 (label-j (car (set-difference all-labels output-subscripts)))
                 (pos-i (position label-i all-labels))
                 (pos-j (position label-j all-labels))
                 (pos-k (position label-k all-labels)))
            
            (when (and pos-i pos-j pos-k)
              (let ((data-A (the (simple-array double-float (*))
				 (aref in-data-vec 0)))
                    (data-B (the (simple-array double-float (*))
				 (aref in-data-vec 1)))
                    (strides-A (aref strides-mat 0))
                    (strides-B (aref strides-mat 1))
                    (offset-A (the fixnum (aref offsets-vec 0)))
                    (offset-B (the fixnum (aref offsets-vec 1))))
                
                (let ((d-i (aref dims-vec pos-i))
                      (d-j (aref dims-vec pos-j))
                      (d-k (aref dims-vec pos-k))
                      (sA-i (aref strides-A pos-i))
		      (sA-j (aref strides-A pos-j))
                      (sB-j (aref strides-B pos-j))
		      (sB-k (aref strides-B pos-k))
                      (sO-i (aref out-strides-vec pos-i))
		      (sO-k (aref out-strides-vec pos-k)))
                  
                  (declare (type fixnum d-i d-j d-k sA-i sA-j sB-j
				 sB-k sO-i sO-k offset-A offset-B))
                  
                  ;; 执行优化的三重循环
                  (let ((ptr-A-base offset-A)
                        (ptr-O-base 0))
                    (declare (type fixnum ptr-A-base ptr-O-base))
                    (loop for i-fixnum from 0 below d-i do
                      (let ((ptr-A-cur ptr-A-base)
                            (ptr-B-row offset-B))
                        (declare (type fixnum ptr-A-cur ptr-B-row))
                        (loop for j-fixnum from 0 below d-j do
                          (let ((val-a (aref data-A ptr-A-cur)))
                            (declare (type double-float val-a))
                            (let ((ptr-o ptr-O-base)
                                  (ptr-b ptr-B-row))
                              (declare (type fixnum ptr-o ptr-b))
                              (loop for k-fixnum from 0 below d-k do
                                (incf (aref out-data ptr-o)
				      (* val-a (aref data-B ptr-b)))
                                (incf ptr-o sO-k)
                                (incf ptr-b sB-k))))
                          (incf ptr-A-cur sA-j)
                          (incf ptr-B-row sB-j))
                        (incf ptr-A-base sA-i)
                        (incf ptr-O-base sO-i)))
                    
                    ;; 如果优化路径执行成功,直接返回
                    (return-from einsum-execute-fast output)))))))
	
	;; === 通用路径 ===
	(let ((cur-ptrs (make-array n-vts :element-type 'fixnum
					  :initial-element 0))
              (cur-out-ptr 0))
	  (declare (type (simple-array fixnum (*)) cur-ptrs))
	  ;; 初始化指针偏移
	  (loop for k from 0 below n-vts do
	    (setf (aref cur-ptrs k)
		  (the fixnum (aref offsets-vec k))))
	  
	  (labels
	      ((recurse (depth)
                 (declare (type fixnum depth))
                 (if (= depth rank)
		     ;; 叶节点:计算
		     (let ((product 1.0d0)) ;; double-float
		       (declare (type double-float product))
		       (loop for k fixnum from 0 below n-vts
			     for data across in-data-vec
			     for ptr fixnum = (aref cur-ptrs k)
			     do (setf product (* product (aref data ptr))))
		       (incf (aref out-data cur-out-ptr) product))		       
		     ;; 分支:遍历
		     (let* ((dim (aref dims-vec depth))
			    (out-stride (aref out-strides-vec depth)))
		       (declare (type fixnum dim out-stride))
		       (loop for i fixnum from 0 below dim do
                         (recurse (1+ depth))
                         ;; 指针后移
                         (loop for k fixnum from 0 below n-vts
			       for strides across strides-mat
			       do (incf (aref cur-ptrs k)
					(aref strides depth)))
                         (incf cur-out-ptr out-stride))
		       ;; 指针回溯
		       (loop for k fixnum from 0 below n-vts
			     for strides across strides-mat
			     do (decf (aref cur-ptrs k)
				      (the fixnum (* dim (aref strides depth)))))
		       (decf cur-out-ptr (the fixnum (* dim out-stride)))))))
            (recurse 0))))
    output))


(declaim (inline vt-einsum))
(defun vt-einsum (subscripts &rest vts)
  "爱因斯坦求和约定 (增强版,支持标准 '...' 省略号).
   示例:
     (vt-einsum \"...ij,...jk->...ik\" A B) ; 批量矩阵乘法
     (vt-einsum \"...ii->...i\" A)          ; 批量提取对角线
  "
  (multiple-value-bind (raw-inputs raw-output explicit-p)
      (parse-subscript-tokens subscripts)    
    (unless (= (length raw-inputs) (length vts))
      (error "提供的张量数量 ~A 与子脚标 ~A 不匹配" (length vts) raw-inputs))
    ;; 1. 展开 Ellipsis
    (multiple-value-bind (input-subs output-subs)
        (expand-ellipsis raw-inputs raw-output vts)
      ;; 2. 如果没有显式输出,自动推导
      ;; 注意:analyze-einsum 内部也会推导,但我们需要先展开 ellipsis
      ;; 这里我们简化,直接传递 nil 让 analyze-einsum 推导
      (unless explicit-p
        (setf output-subs nil))
      (multiple-value-bind (all-labels dim-sizes global-strides output-subs-final)
          (analyze-einsum input-subs output-subs explicit-p vts)        
        (einsum-execute-fast all-labels dim-sizes global-strides output-subs-final vts)))))
