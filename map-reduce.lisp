(in-package :clvt)

(defun vt-map (fn &rest args)
   "高效映射函数. 支持标量、列表、张量混合输入，并自动进行多维广播.
  极致性能与安全优化:
  1. 编译期特化: 针对递增的 1-6 个参数在宏展开期生成专属调用，彻底消灭运行时 apply 和列表分配.
  2. 双轨制内存: 
     - Fast Path: 严格校验输入输出的连续性 (vt-contiguous-p)，连续内存直接走 1D 平坦遍历.
     - Slow Path: 非连续内存 (如转置视图) 自动走 N-D 递归，精确依据 strides 进行多维跳跃寻址.
  3. 零开销类型转换: 引入编译期局部宏，在展开期静态推断输出类型，生成无分支的 coerce/truncate 代码.
     既保证了向类型化数组写入时的绝对类型安全 (杜绝内存损坏)，又消除了运行时类型检查的开销.
  4. 寄存器优化: 同类型快速路径中提取数组指针并声明临时变量类型，最大化利用 CPU 寄存器，避免浮点装箱."
  (declare (function fn) (optimize (safety 0)))
  (with-float-safe
    (multiple-value-bind (tensors dtype out)
	(parse-vt-op-args args)
      (when (null tensors)
        (error "vt-map requires at least one tensor"))
      (let* ((clean-tensors (mapcar #'ensure-vt tensors))
             (n-tensors (length clean-tensors))
             (out-shape (reduce #'vt-broadcast-shapes
                                (mapcar #'vt-shape clean-tensors))))	
        (let* ((final-dtype (cond
                              ((and out dtype (not (eq (vt-dtype out) dtype)))
                               (error "vt-map: :out 的类型 (~a) 与 :dtype (~a) 冲突"
                                      (vt-dtype out) dtype))
                              (out (vt-dtype out))
                              (dtype dtype)
                              (t (apply #'vt-promote-type
                                        (mapcar #'vt-dtype clean-tensors))))) 
               (res (or out (vt-zeros out-shape :dtype final-dtype))))
          
          (when out
            (unless (equal (vt-shape res) out-shape)
              (error "vt-map: :out 张量形状 ~a 与广播结果 ~a 不匹配"
                     (vt-shape res) out-shape)))
          (let* ((res-data (vt-data res))
                 (res-strides (vt-strides res))
                 (ins-data (coerce (mapcar #'vt-data clean-tensors)
                                   'simple-vector))
                 (ins-strides (coerce (mapcar (lambda (vt)
                                                (vt-broadcast-strides
                                                 (vt-shape vt) out-shape
                                                 (vt-strides vt)))
                                              clean-tensors)
				      'simple-vector))
                 (cur-ptrs (make-array n-tensors :element-type 'fixnum
                                                 :initial-element 0))
                 (rank (length out-shape))
                 (size (vt-shape-to-size out-shape))
		 (is-fast (and (vt-contiguous-p res)
			       (every #'(lambda (ts)
					  (or (and (equal (vt-shape ts) out-shape)
                                                   (vt-contiguous-p ts))
					      (= (vt-size ts) 1)))
				      clean-tensors)))
                 (all-same-type (every #'(lambda (ts)
                                           (eq (vt-dtype ts) final-dtype))
                                       clean-tensors)))
            
            (declare (type simple-vector ins-data ins-strides)
                     (type (simple-array fixnum (*)) cur-ptrs)
                     (type list out-shape res-strides)
                     (type fixnum rank n-tensors size)
                     (boolean is-fast all-same-type))

            (loop for vt in clean-tensors
                  for i fixnum from 0
                  do (setf (aref cur-ptrs i) (vt-offset vt))) 

            (macrolet ((cast-to-out (val lisp-type)
                         `(if ,(subtypep lisp-type 'integer)
                              (truncate ,val)
                              (coerce ,val ',lisp-type))))
              
              (macrolet
                  ((gen-dispatch (lisp-type)
                     (let ((calls-same
                             (loop for n from 1 to 6
                                   collect
				    `(,n (funcall
					  fn
					  ,@(loop for k from 0 below n
						  collect
						  `(aref (the (simple-array ,lisp-type (*))
							      (aref ins-data ,k))
							 (aref cur-ptrs ,k)))))))
                           (calls-diff
                             (loop for n from 1 to 6
                                   collect
				    `(,n (funcall
					  fn
					  ,@(loop for k from 0 below n
						  collect `(aref (the simple-array (aref ins-data ,k))
								 (aref cur-ptrs ,k))))))))
                     `(let ((out-data (the (simple-array ,lisp-type (*)) res-data)))
                        (with-float-safe
                          (if is-fast
                              ;; ================ FAST PATH ================
                              (let ((out-ptr (vt-offset res))
				    (steps (let ((arr (make-array n-tensors :element-type 'fixnum)))
					     (loop for k from 0 below n-tensors
						   for ts in clean-tensors
						   do (setf (aref arr k)
							    (if (equal (vt-shape ts) out-shape) 1 0))) 
					     arr)))
                                (declare (type fixnum out-ptr)
                                         (type (simple-array fixnum (*)) steps))
                                (if all-same-type
                                    (cond
                                      ((= n-tensors 1)
                                       (let ((d0 (the (simple-array ,lisp-type (*)) (aref ins-data 0)))
                                             (p0 (aref cur-ptrs 0)))
                                         (declare (type (simple-array ,lisp-type (*)) d0)
						  (type fixnum p0))
                                         (loop for i fixnum from 0 below size do
					   (let ((v (funcall fn (aref d0 p0))))
                                             ;; 使用宏替换
                                             (setf (aref out-data out-ptr) (cast-to-out v ,lisp-type)))
					   (incf out-ptr)
					   (incf p0 (aref steps 0))))) 
                                      ((= n-tensors 2)
                                       (let ((d0 (the (simple-array ,lisp-type (*)) (aref ins-data 0)))
                                             (p0 (aref cur-ptrs 0))
                                             (d1 (the (simple-array ,lisp-type (*)) (aref ins-data 1)))
                                             (p1 (aref cur-ptrs 1)))
                                         (declare (type (simple-array ,lisp-type (*)) d0 d1)
						  (type fixnum p0 p1))
                                         (loop for i fixnum from 0 below size do
					   (let ((v (funcall fn (aref d0 p0) (aref d1 p1))))
                                             (setf (aref out-data out-ptr) (cast-to-out v ,lisp-type)))
					   (incf out-ptr)
					   (incf p0 (aref steps 0))
					   (incf p1 (aref steps 1)))))
                                      (t
                                       (loop for i fixnum from 0 below size do
                                         (let ((v (case n-tensors
                                                    ,@calls-same
                                                    (otherwise
						     (apply fn (loop for k fixnum from 0 below n-tensors
                                                                     collect
								     (aref (the (simple-array ,lisp-type (*))
										(aref ins-data k))
									   (aref cur-ptrs k))))))))
					   (setf (aref out-data out-ptr) (cast-to-out v ,lisp-type)))
                                         (incf out-ptr)
                                         (loop for k fixnum from 0 below n-tensors do
					   (incf (aref cur-ptrs k) (aref steps k))))))
                                    (cond
                                      ((= n-tensors 1)
                                       (let ((d0 (the simple-array (aref ins-data 0)))
                                             (p0 (aref cur-ptrs 0)))
                                         (declare (type simple-array d0)
						  (type fixnum p0))
                                         (loop for i fixnum from 0 below size do
					   (let ((v (funcall fn (aref d0 p0))))
                                             (setf (aref out-data out-ptr) (cast-to-out v ,lisp-type)))
					   (incf out-ptr)
					   (incf p0 (aref steps 0))))) 
                                      ((= n-tensors 2)
                                       (let ((d0 (the simple-array (aref ins-data 0)))
                                             (p0 (aref cur-ptrs 0))
                                             (d1 (the simple-array (aref ins-data 1)))
                                             (p1 (aref cur-ptrs 1)))
                                         (declare (type simple-array d0 d1)
						  (type fixnum p0 p1))
                                         (loop for i fixnum from 0 below size do
					   (let ((v (funcall fn (aref d0 p0) (aref d1 p1))))
                                             (setf (aref out-data out-ptr) (cast-to-out v ,lisp-type)))
					   (incf out-ptr)
					   (incf p0 (aref steps 0))
					   (incf p1 (aref steps 1))))) 
                                      (t
                                       (loop for i fixnum from 0 below size do
                                         (let ((v (case n-tensors
                                                    ,@calls-diff
                                                    (otherwise
						     (apply fn (loop for k fixnum
								     from 0 below n-tensors
                                                                     collect
								     (aref (the simple-array (aref ins-data k))
									   (aref cur-ptrs k))))))))
					   (setf (aref out-data out-ptr) (cast-to-out v ,lisp-type)))
                                         (incf out-ptr)
                                         (loop for k fixnum from 0 below n-tensors do
					   (incf (aref cur-ptrs k) (aref steps k))))))))
                              
                              ;; ================ SLOW PATH (N-D 递归) ================
                              (labels
                                  ((recurse (depth out-ptr)
                                     (declare (type fixnum depth out-ptr))
                                     (if (= depth rank)
					 (let ((raw-result
						 (case n-tensors
                                                   ,@calls-diff
                                                   (otherwise
						    (apply fn (loop for k fixnum from 0 below n-tensors
                                                                    collect
								    (aref (the simple-array (aref ins-data k))
									  (aref cur-ptrs k))))))))
                                           (setf (aref out-data out-ptr)
						 (cast-to-out raw-result ,lisp-type)))
					 (let* ((dim (the fixnum (nth depth out-shape)))
						(res-stride (the fixnum (nth depth res-strides))))
                                           (declare (type fixnum dim res-stride))
                                           (loop for i fixnum from 0 below dim do
                                             (recurse (1+ depth) out-ptr)
                                             (incf out-ptr res-stride)
                                             (loop for k fixnum from 0 below n-tensors do
					       (incf (aref cur-ptrs k)
                                                     (the fixnum (nth depth (aref ins-strides k))))))
                                           (loop for k fixnum from 0 below n-tensors do
                                             (decf (aref cur-ptrs k)
                                                   (the fixnum (* dim
                                                                  (the fixnum
								       (nth depth (aref ins-strides k))))))))))) 
                                (recurse 0 (vt-offset res)))))))))
              
              (etypecase res-data
                ((simple-array double-float (*))
                 (gen-dispatch double-float))
                ((simple-array single-float (*))
                 (gen-dispatch single-float))
                ((simple-array (signed-byte 64) (*))
                 (gen-dispatch (signed-byte 64)))
                ((simple-array (signed-byte 32) (*))
                 (gen-dispatch (signed-byte 32))))) 
            
            res)))))))


(defun vt-binary (fn t1 t2 &key out dtype)
  "二元张量操作基石. 
  专为 2 个输入张量极致优化，零中间分配，支持算术内联.
  API 设计对标 NumPy: (vt-binary #'+ t1 t2 :out out :dtype dtype)
  "
  (declare (function fn) (optimize (safety 0)))
  (with-float-safe
    (let* ((clean-t1 (ensure-vt t1))
           (clean-t2 (ensure-vt t2))
           (out-shape
	     (vt-broadcast-shapes (vt-shape clean-t1) (vt-shape clean-t2))))
      
      ;; 1. 确定输出张量及其类型 (严格校验)
      (let* ((final-dtype (cond
                            ((and out dtype (not (eq (vt-dtype out) dtype)))
                             (error "vt-binary: :out 的类型 (~a) 与 :dtype (~a) 冲突"
                                    (vt-dtype out) dtype))
                            (out (vt-dtype out))
                            (dtype dtype)
                            (t (vt-promote-type (vt-dtype clean-t1) (vt-dtype clean-t2))))) 
             (res (or out (vt-zeros out-shape :dtype final-dtype))))
        
        (when out
          (unless (equal (vt-shape res) out-shape)
            (error "vt-binary: :out 张量形状 ~a 与广播结果 ~a 不匹配"
                   (vt-shape res) out-shape)))

        (let* ((res-data (vt-data res))
               (res-strides (vt-strides res))
               (d0 (vt-data clean-t1))
               (d1 (vt-data clean-t2))
               (strides-0 (vt-broadcast-strides
			   (vt-shape clean-t1) out-shape (vt-strides clean-t1)))
               (strides-1 (vt-broadcast-strides
			   (vt-shape clean-t2) out-shape (vt-strides clean-t2)))
               (p0 (vt-offset clean-t1))
               (p1 (vt-offset clean-t2))
               (rank (length out-shape))
               (size (vt-shape-to-size out-shape))
	       (is-fast (and (vt-contiguous-p res)
			     (or (equal (vt-shape clean-t1) out-shape)
				 (= (vt-size clean-t1) 1))
			     (or (equal (vt-shape clean-t2) out-shape)
				 (= (vt-size clean-t2) 1))))

               (all-same-type (and (eq (vt-dtype clean-t1) final-dtype)
                                   (eq (vt-dtype clean-t2) final-dtype))))
          
          (declare (type list out-shape res-strides strides-0 strides-1)
                   (type fixnum p0 p1 rank size)
                   (boolean is-fast all-same-type))

          ;; 2. 核心类型派发宏：二元特化版
          (macrolet
	      ((gen-dispatch (lisp-type)
                 `(let ((out-data (the (simple-array ,lisp-type (*)) res-data)))
		    (with-float-safe
                      (if is-fast
			  ;; ================ FAST PATH ================
			  (let ((out-ptr (vt-offset res))
				(step0 (if (equal (vt-shape clean-t1) out-shape) 1 0))
				(step1 (if (equal (vt-shape clean-t2) out-shape) 1 0)))
                            (declare (type fixnum out-ptr step0 step1))
                            (if all-same-type
                                ;; --- 输入输出类型完全一致：零装箱，算术内联 ---
                                (let ((arr0 (the (simple-array ,lisp-type (*)) d0))
                                      (arr1 (the (simple-array ,lisp-type (*)) d1)))
				  (declare (type (simple-array ,lisp-type (*)) arr0 arr1))
				  (loop for i fixnum from 0 below size do
                                    (let ((v (funcall fn (aref arr0 p0) (aref arr1 p1))))
                                      (declare (type ,lisp-type v))
                                      (setf (aref out-data out-ptr) v))
                                    (incf out-ptr)
                                    (incf p0 step0)
                                    (incf p1 step1)))
                                ;; --- 输入输出类型不一致：通用数组读取 ---
                                (progn
				  (loop for i fixnum from 0 below size do
                                    (let ((v (funcall fn (aref d0 p0) (aref d1 p1))))
                                      (setf (aref out-data out-ptr)
                                            (if ,(subtypep lisp-type 'integer)
						(truncate v)
						(coerce v ',lisp-type))))
                                    (incf out-ptr)
                                    (incf p0 step0)
                                    (incf p1 step1)))))
			  
			  ;; ================ SLOW PATH (N-D 递归) ================
			  (labels
			      ((recurse (depth out-ptr)
                                 (declare (type fixnum depth out-ptr))
                                 (if (= depth rank)
                                     (let ((raw-result (funcall fn (aref d0 p0) (aref d1 p1))))
				       (setf (aref out-data out-ptr)
                                             (if ,(subtypep lisp-type 'integer)
						 (truncate raw-result) (coerce raw-result ',lisp-type))))
                                     (let* ((dim (the fixnum (nth depth out-shape)))
                                            (res-stride (the fixnum (nth depth res-strides)))
                                            (str0 (the fixnum (nth depth strides-0)))
                                            (str1 (the fixnum (nth depth strides-1))))
				       (declare (type fixnum dim res-stride str0 str1))
				       (loop for i fixnum from 0 below dim do
                                         (recurse (1+ depth) out-ptr)
                                         (incf out-ptr res-stride)
                                         (incf p0 str0)
                                         (incf p1 str1))
				       (decf p0 (the fixnum (* dim str0)))
				       (decf p1 (the fixnum (* dim str1)))))))
                            (recurse 0 (vt-offset res))))))))
            
            ;; 触发宏展开，生成 4 份特化代码
            (etypecase res-data
              ((simple-array double-float (*))
               (gen-dispatch double-float))
              ((simple-array single-float (*))
               (gen-dispatch single-float))
              ((simple-array (signed-byte 64) (*))
               (gen-dispatch (signed-byte 64)))
              ((simple-array (signed-byte 32) (*))
               (gen-dispatch (signed-byte 32))))))
        
        res))))

(declaim (inline get-reduction-identity))
(defun get-reduction-identity (op element-type)
  "根据操作类型和元素类型,返回正确的初始值。
使用 ieee 无穷大作为浮点数归约初始值，以支持输入中的 inf。
精确匹配 (signed-byte 32/64) 并返回正确的极值边界。"
  (with-float-safe
    (case op
      (:sum (coerce 0 element-type))
      (:max
       (cond
         ((eq element-type 'double-float)
          #+sbcl sb-ext:double-float-negative-infinity
          #-sbcl (/ -1.0d0 0.0d0))
         ((eq element-type 'single-float)
          #+sbcl sb-ext:single-float-negative-infinity
          #-sbcl (/ -1.0f0 0.0f0))
         ;; 修复: 使用 equal 匹配列表类型，并返回真实的 64/32 位最小值
         ((equal element-type '(signed-byte 64)) #.(- (expt 2 63)))
         ((equal element-type '(signed-byte 32)) #.(- (expt 2 31)))
         ;; 兜底: 未知整数类型退回 fixnum 极值
         ((subtypep element-type 'integer) most-negative-fixnum)
         (t 0)))
      (:min
       (cond
         ((eq element-type 'double-float)
          #+sbcl sb-ext:double-float-positive-infinity
          #-sbcl (/ 1.0d0 0.0d0))
         ((eq element-type 'single-float)
          #+sbcl sb-ext:single-float-positive-infinity
          #-sbcl (/ 1.0f0 0.0f0))
         ;; 修复: 使用 equal 匹配列表类型，并返回真实的 64/32 位最大值
         ((equal element-type '(signed-byte 64)) #.(1- (expt 2 63)))
         ((equal element-type '(signed-byte 32)) #.(1- (expt 2 31)))
         ;; 兜底: 未知整数类型退回 fixnum 极值
         ((subtypep element-type 'integer) most-positive-fixnum)
         (t 0))))))


(defun vt-reduce (tensor axis init-val reducer-fn &key out dtype keepdims return-arg)
  "通用归约核心. 支持多维指定轴(可多轴)归约与全局归约.
  API 对标主流库:
    (vt-reduce tensor axis init-val reducer-fn :out out :dtype :float64 :keepdims t)
  axis: 可以是 nil (全局), 单个整数, 或整数列表 (如 '(0 2))
  "
  (declare (type vt tensor)
           (type (or null fixnum list) axis)
           (type function reducer-fn)
           (optimize (safety 0)))
  (with-float-safe
    (let* ((in-shape (vt-shape tensor))
           (rank (length in-shape))
           ;; 规范化 axis 为排序列表
           (raw-axes (if (listp axis) axis (list axis)))
           (real-axes (when raw-axes
			(sort (mapcar (lambda (a)
					(vt-normalize-axis a rank))
				      raw-axes)
			      #'<)))
           (is-global-reduction (null real-axes))
           (out-shape
             (cond
               ((and is-global-reduction (not keepdims)) nil)
               (is-global-reduction (make-list rank :initial-element 1))
               ((not keepdims)
                (loop for d in in-shape
                      for i from 0
                      unless (member i real-axes) collect d))
               (t (loop for d in in-shape
                        for i from 0
                        collect (if (member i real-axes) 1 d)))))
           (in-strides (vt-strides tensor))
           (in-offset (vt-offset tensor))
	   (arg-strides
	     (if return-arg
		 (if is-global-reduction
		     (vt-compute-strides in-shape)
		     (loop for i from 0 below rank
			   if (member i real-axes) collect 1
			     else collect 0))
		 (make-list rank :initial-element 0)))
           (axis-size
             (if real-axes
                 (the fixnum (reduce #'* (mapcar (lambda (a)
						   (nth a in-shape))
						 real-axes)
				     :initial-value 1))
                 (the fixnum (reduce #'* in-shape :initial-value 1)))))      
      (declare (type list in-strides arg-strides in-shape))
      
      ;; 空张量防御
      (when (or (zerop axis-size) (zerop (vt-size tensor)))
        (let ((empty-dtype (or dtype (and out (vt-dtype out)) (vt-dtype tensor))))
          (return-from vt-reduce
            (values (make-vt out-shape 0 :dtype empty-dtype)
                    (when return-arg (make-vt out-shape 0 :dtype :int32))))))
      
      ;; 1. 确定输出类型 (对标 NumPy 严格校验)
      (let* ((final-dtype (cond
                            ((and out dtype (not (eq (vt-dtype out) dtype)))
                             (error "vt-reduce: :out 的类型 (~a) 与 :dtype (~a) 冲突"
                                    (vt-dtype out) dtype))
                            (out (vt-dtype out))
                            (dtype dtype)
                            ((and init-val (or (floatp init-val)
					       (vt-float-inf-p init-val)))
			     :float64)
                            (t (vt-dtype tensor))))
             (res (or out (make-vt out-shape 0 :dtype final-dtype)))
             (res-data (vt-data res))
             (res-offset (vt-offset res))
             (res-idx (when return-arg (make-vt out-shape 0 :dtype :int32)))
             (res-idx-data (when res-idx (vt-data res-idx)))
             (out-strides-map
               (if is-global-reduction
                   (make-list rank :initial-element 0)
                   (loop for i from 0 below rank
                         if (member i real-axes) collect 0
                           else collect
				(let ((out-idx (if keepdims 
                                                   i 
                                                   (count-if-not
						    (lambda (x) (member x real-axes)) 
                                                    (loop for j below i collect j)))))
                                  (nth out-idx (vt-strides res)))))))
	;; 2. 核心类型派发宏：归约特化版
	(macrolet
	    ((gen-reduce (out-type)
               `(let ((out-data (the (simple-array ,out-type (*)) res-data))
                      (in-data (the (simple-array * (*)) (vt-data tensor)))
                      (idx-data
			(when res-idx (the (simple-array (signed-byte 32) (*))
					   res-idx-data))))
                  (declare (type (simple-array ,out-type (*)) out-data)
                           (type (simple-array * (*)) in-data))
                  (with-float-safe
                    ;; === 修复1：安全初始化输出视图，防止越界覆盖共享内存 ===
                    (let ((init-num (or init-val (error "vt-reduce: 必须提供 init-val")))
                          (res-strides-vec (coerce (vt-strides res) 'simple-vector))
                          (out-shape-vec (coerce out-shape 'simple-vector))
                          (out-rank (length out-shape)))
                      (declare (type fixnum out-rank))
                      (labels ((init-view (depth out-ptr)
                                 (declare (type fixnum depth out-ptr))
                                 (if (= depth out-rank)
                                     (setf (aref out-data out-ptr)
                                           ,(if (subtypep out-type 'integer)
                                                `(truncate init-num)
                                                `(coerce init-num ',out-type)))
                                     (let ((dim (the fixnum (svref out-shape-vec depth)))
                                           (stride (the fixnum (svref res-strides-vec depth))))
                                       (declare (type fixnum dim stride))
                                       (loop for i fixnum from 0 below dim do
                                         (init-view (1+ depth) out-ptr)
                                         (incf out-ptr stride))))))
                        (init-view 0 res-offset)))
                    
                    (labels
			((recurse (depth in-ptr out-ptr out-idx-ptr current-arg-val)
                           (declare (type fixnum depth in-ptr out-ptr out-idx-ptr current-arg-val))
                           (if (= depth rank)
                               (let* ((val (aref in-data in-ptr))
                                      (raw-acc (aref out-data out-ptr)))
                                 (multiple-value-bind (new-acc do-update-idx)
                                     (funcall reducer-fn raw-acc val)
                                   (setf (aref out-data out-ptr)
                                         ,(if (subtypep out-type 'integer)
                                              `(truncate new-acc)
                                              `(coerce new-acc ',out-type)))
                                   (when (and return-arg do-update-idx idx-data)
                                     (setf (aref idx-data out-idx-ptr) current-arg-val))))
                               (let* ((dim (the fixnum (nth depth in-shape)))
                                      (in-stride (the fixnum (nth depth in-strides)))
                                      (out-stride (the fixnum (nth depth out-strides-map)))
                                      (arg-stride (the fixnum (nth depth arg-strides))))
                                 (declare (type fixnum dim in-stride out-stride arg-stride))
                                 (loop for i fixnum from 0 below dim do
                                   (recurse (1+ depth)
					    in-ptr out-ptr
					    out-idx-ptr
					    (+ current-arg-val (* i arg-stride)))
                                   (incf in-ptr in-stride)
                                   (incf out-ptr out-stride)
                                   (when return-arg (incf out-idx-ptr out-stride)))))))
                      (recurse 0 in-offset res-offset (if res-idx (vt-offset res-idx) 0) 0))))))
          ;; 触发宏展开，仅生成 4 份特化代码，清爽且高效
          (etypecase res-data
            ((simple-array double-float (*))
	     (gen-reduce double-float))
            ((simple-array single-float (*))
	     (gen-reduce single-float))
            ((simple-array (signed-byte 64) (*))
	     (gen-reduce (signed-byte 64)))
            ((simple-array (signed-byte 32) (*))
	     (gen-reduce (signed-byte 32)))))
	(values res res-idx)))))


;; 底层统一迭代器宏
(defmacro vt-foreach ((out-spec &rest in-specs) &body body)
  "高效的多张量遍历基石宏。
Fast-Path: 连续内存极速线性循环。
Slow-Path: 通用 N 维非连续遍历 (编译期展开，零分配指针推进)。
支持从 body 中自动提取 declare 语句并置于合法作用域。"

  (let* ((out-tens (first out-spec))
         (p-out (second out-spec))
         (in-tens (mapcar #'first in-specs))
         (in-vars (mapcar #'second in-specs))
         ;; 提取 body 中的 declare 语句
         (decls (loop for form in body
                      while (and (consp form) (eq (car form) 'declare))
                      collect form))
         (real-body (nthcdr (length decls) body)))
    (let ((size (gensym "SIZE"))
          (is-fast (gensym "IS-FAST"))
          (all-tens (gensym "ALL-TENS"))
          (dims-vec (gensym "DIMS-VEC"))
          (rank (gensym "RANK"))
          (out-strs-vec (gensym "OUT-STRS-VEC"))
          (in-strs-vecs (loop for v in in-vars
                              collect (gensym
                                       (concatenate 'string
                                                    (symbol-name v)
                                                    "-STRS-VEC"))))
          (in-stride-vars (loop for v in in-vars
                                collect (gensym
                                         (concatenate 'string
                                                      (symbol-name v)
                                                      "-STRIDE")))))
      `(let* ((,all-tens (list ,out-tens ,@in-tens))
              (,size (vt-size ,out-tens))
              (,is-fast (and (every #'vt-contiguous-p ,all-tens)
                             (loop for ts in ,all-tens
                                   always (or (equal (vt-shape ts)
                                                     (vt-shape ,out-tens))
                                              (= (vt-size ts) 1))))))
	 (with-float-safe
           (if ,is-fast
               ;; Fast-Path: 连续内存极速线性循环
               (let* ((,p-out (vt-offset ,out-tens))
                      ,@(loop for v in in-vars
                              for ts in in-tens
                              collect `(,v (vt-offset ,ts)))
                      ,@(loop for s in in-stride-vars
                              for ts in in-tens
                              collect 
			      `(,s (if (equal (vt-shape ,ts)
					      (vt-shape ,out-tens))
				       1 0))
			      ))
		 (declare (type fixnum ,p-out ,@in-vars ,@in-stride-vars))
		 ,@decls
		 (loop for i fixnum from 0 below ,size
                       do (progn
                            ,@real-body
                            (incf ,p-out)
                            ,@(loop for v in in-vars
                                    for s in in-stride-vars
                                    collect `(incf ,v ,s)))))

               ;; Slow-Path: 通用 N 维非连续遍历
               (let* ((,dims-vec (coerce (vt-shape ,out-tens) 'simple-vector))
                      (,rank (length ,dims-vec))
                      (,out-strs-vec (coerce (vt-strides ,out-tens) 'simple-vector))
                      ,@(loop for vec in in-strs-vecs
			      for ts in in-tens collect
                              `(,vec (coerce (vt-broadcast-strides (vt-shape ,ts)
								   (vt-shape ,out-tens)
								   (vt-strides ,ts))
					     'simple-vector)))
                      (,p-out (vt-offset ,out-tens))
                      ,@(loop for v in in-vars
			      for ts in in-tens
			      collect `(,v (vt-offset ,ts))))
		 (declare (type fixnum ,p-out ,@in-vars)
                          (type simple-vector ,dims-vec ,out-strs-vec
				,@in-strs-vecs))
		 ,@decls
		 (labels
		     ((recurse (depth)
			(declare (type fixnum depth))
			(if (= depth ,rank)
                            (progn ,@real-body)
                            (let ((dim (svref ,dims-vec depth)))
                              (loop for i fixnum from 0 below dim do
				(recurse (1+ depth))
				(incf ,p-out (svref ,out-strs-vec depth))
                                    ,@(loop for v in in-vars
                                            for vec in in-strs-vecs
                                            collect
					    `(incf ,v
                                                   (svref ,vec depth))))))))
                   (recurse 0)))))))))



;;; --- 高效迭代逻辑  ---
(defmacro vt-do-each ((ptr-var val-var vt) &body body)
  "高效遍历宏 (支持非连续内存视图)。
   ptr-var: 当前元素的物理索引。
   val-var: 当前元素的值。"
  (let ((shape-sym (gensym "shape"))
        (strides-sym (gensym "strides"))
        (data-sym (gensym "data"))
        (offset-sym (gensym "offset"))
        (rank-sym (gensym "rank"))
        (dims-vec (gensym "dims-vec"))
        (strs-vec (gensym "strs-vec"))
        (depth-sym (gensym "depth")))
    `(let* ((,shape-sym (vt-shape ,vt))
            (,strides-sym (vt-strides ,vt))
            (,data-sym (vt-data ,vt))
            (,offset-sym (vt-offset ,vt))
            (,rank-sym (length ,shape-sym))
            ;; 预提取为简单向量，使用 svref 达到 O(1) 访问速度
            (,dims-vec (coerce ,shape-sym 'simple-vector))
            (,strs-vec (coerce ,strides-sym 'simple-vector)))
       (labels
	   ((recurse (,depth-sym current-ptr)
              (if (= ,depth-sym ,rank-sym)
                  ;; 到达最内层：绑定为普通变量，允许 declare ignore
                  (let ((,ptr-var current-ptr)
                        (,val-var (aref (the simple-array ,data-sym) current-ptr)))
                    (declare (ignorable ,ptr-var ,val-var))
                    ,@body)
                  ;; 中间维度：循环并累加指针
                  (let ((dim (svref ,dims-vec ,depth-sym))
                        (stride (svref ,strs-vec ,depth-sym)))
                    (loop for i from 0 below dim do
                      (recurse (1+ ,depth-sym) current-ptr)
                      (incf current-ptr stride))))))
         (recurse 0 ,offset-sym)))))

(defun vt-where (condition x y &key out dtype)
  "三元条件选择，对标 pytorch 的 torch.where 和 numpy 的 np.where(cond, x, y)。
   根据 condition 的真假，从 x 或 y 中选择元素。支持自动广播。
   
   Parameters:
   - condition: 条件张量。
   - x, y: 选择源张量。
   - out: (可选) 输出张量。如果提供，结果将写入该张量。
   - dtype: (可选) 输出张量的数据类型。如果未提供，则由 x 和 y 推断。
           如果提供了 out，则 dtype 必须与 out 的类型匹配。"
  (setf condition (ensure-vt condition))
  (setf x (ensure-vt x))
  (setf y (ensure-vt y))
  (with-float-safe
    (let* ((target-shape (vt-broadcast-shapes
                          (vt-shape condition)
                          (vt-broadcast-shapes (vt-shape x) (vt-shape y))))
           ;; 推断 x 和 y 的自然提升类型
           (promoted-dtype (vt-promote-type (vt-dtype x) (vt-dtype y)))
           
           ;; 1. 确定 final-dtype (优先级: out > 显式 dtype > 自动推断)
           (final-dtype (cond 
                          ;; 如果同时提供 out 和 dtype，必须一致
                          ((and out dtype) 
                           (unless (eq (vt-dtype out) dtype)
                             (error "vt-where: :out 的 dtype (~a) 与 :dtype (~a) 冲突" 
                                    (vt-dtype out) dtype))
                           (vt-dtype out))
                          ;; 如果有 out，以 out 类型为准
                          (out (vt-dtype out))
                          ;; 如果有 dtype，以显式 dtype 为准
                          (dtype dtype)
                          ;; 否则自动推断
                          (t promoted-dtype)))
           
           (lisp-type (vt-dtype->lisp-type final-dtype))
           
           ;; 2. 确定输出张量 result
           (result (if out
                       (progn
                         ;; 校验 out 的形状是否合法
                         (unless (equal (vt-shape out) target-shape)
                           (error "vt-where: :out 的形状 ~a 与广播结果 ~a 不匹配" 
                                  (vt-shape out) target-shape))
                         out)
                       ;; 如果没有 out，创建新张量
                       (vt-zeros target-shape :dtype final-dtype)))
           
           (size (vt-shape-to-size target-shape))
           
           ;; 3. 快速路径判断
           ;; 只有当所有相关张量均为连续内存，且形状符合广播规则时才走快速路径
           ;; 注意：result (out) 也必须是连续的才能走快速路径 (步长必须为1)
           (is-fast (and (vt-contiguous-p result)
                         (vt-contiguous-p condition)
                         (vt-contiguous-p x)
                         (vt-contiguous-p y)
                         ;; 形状匹配 target-shape 或者是标量广播 (size=1)
                         (or (equal (vt-shape condition)
				    target-shape)
			     (= (vt-size condition) 1))
                         (or (equal (vt-shape x)
				    target-shape)
			     (= (vt-size x) 1))
                         (or (equal (vt-shape y)
				    target-shape)
			     (= (vt-size y) 1)))))
      
      (let ((c-data (vt-data condition))
            (x-data (vt-data x))
            (y-data (vt-data y))
            (r-data (vt-data result))) ; 可能是 out 的 data，也可能是新分配的 data
        
        (if is-fast
            ;; === 极速路径: 连续内存极速线性循环 ===
            (let ((c-ptr (vt-offset condition))
                  (x-ptr (vt-offset x))
                  (y-ptr (vt-offset y))
                  (r-ptr (vt-offset result))
                  ;; 计算步长：如果形状完全匹配则为 1，否则（标量广播）为 0
                  (c-step (if (equal (vt-shape condition) target-shape) 1 0))
                  (x-step (if (equal (vt-shape x) target-shape) 1 0))
                  (y-step (if (equal (vt-shape y) target-shape) 1 0)))
              (declare (type fixnum c-ptr x-ptr y-ptr r-ptr c-step x-step y-step))
              (loop for i fixnum from 0 below size do
                ;; 使用 /= 0 判断真值，兼容整数类型
                (setf (aref r-data r-ptr)
                      (coerce (if (/= (aref c-data c-ptr) 0)
                                  (aref x-data x-ptr)
                                  (aref y-data y-ptr))
                              lisp-type))
                (incf c-ptr c-step)
                (incf x-ptr x-step)
                (incf y-ptr y-step)
                (incf r-ptr)))
            
            ;; === 慢速路径: 通用 N 维非连续/广播遍历 ===
            (let* ((rank (length target-shape))
                   (dims-vec (coerce target-shape 'simple-vector))
                   ;; 计算广播后的源张量步长
                   (c-strs-vec (coerce (vt-broadcast-strides
                                        (vt-shape condition)
                                        target-shape
                                        (vt-strides condition))
                                       'simple-vector))
                   (x-strs-vec (coerce (vt-broadcast-strides
                                        (vt-shape x) target-shape (vt-strides x))
                                       'simple-vector))
                   (y-strs-vec (coerce (vt-broadcast-strides
                                        (vt-shape y) target-shape (vt-strides y))
                                       'simple-vector))
                   ;; 获取 result (out) 的实际步长，支持非连续输出
                   (r-strs-vec (coerce (vt-strides result) 'simple-vector)))
              (declare (type fixnum rank)
                       (type simple-vector dims-vec c-strs-vec x-strs-vec y-strs-vec r-strs-vec))
              
              (labels ((recurse (depth c-ptr x-ptr y-ptr r-ptr)
                         (declare (type fixnum depth c-ptr x-ptr y-ptr r-ptr)
                                  (optimize (safety 0)))
                         (if (= depth rank)
                             ;; 叶节点：根据 condition 决定取 x 还是 y
                             (setf (aref r-data r-ptr)
                                   (coerce (if (/= (aref c-data c-ptr) 0)
                                               (aref x-data x-ptr)
                                               (aref y-data y-ptr))
                                           lisp-type))
                             ;; 分支节点：递归遍历
                             (let ((dim (svref dims-vec depth))
                                   (c-str (svref c-strs-vec depth))
                                   (x-str (svref x-strs-vec depth))
                                   (y-str (svref y-strs-vec depth))
                                   (r-str (svref r-strs-vec depth)))
                               (declare (type fixnum dim c-str x-str y-str r-str))
                               (loop for i fixnum from 0 below dim do
                                 (recurse (1+ depth) c-ptr x-ptr y-ptr r-ptr)
                                 ;; 指针递增
                                 (incf c-ptr c-str)
                                 (incf x-ptr x-str)
                                 (incf y-ptr y-str)
                                 (incf r-ptr r-str))))))
                (recurse 0
                         (vt-offset condition)
                         (vt-offset x)
                         (vt-offset y)
                         (vt-offset result))))))
      result)))

(defun vt-argwhere (condition &key (dtype :int64))
  "查找非零元素的坐标。
condition: 条件张量。
返回: 形状为 (n, rank) 的二维张量，每一行是一个非零元素的完整坐标。
示例: (vt-argwhere tensor) -> [[0, 1], [2, 3], ...]

注意 (对标 pytorch/numpy):
如果输入是 0 维张量 (标量), rank 为 0。
- 若标量非零: 返回形状为 (1, 0) 的张量。
- 若标量为零: 返回形状为 (0, 0) 的张量。

:dtype 选项: 仅支持 :int32 或 :int64 (默认)。"
  (declare (type vt condition)
	   (type (member nil :int32 :int64) dtype))
  (let ((final-dtype (or dtype :int64)))
    ;; 严格校验：只允许整数类型
    (unless (member final-dtype '(:int32 :int64))
      (setf final-dtype :int64))
    
    (let* ((lisp-type (if (eq final-dtype :int32)
                          '(signed-byte 32)
                          '(signed-byte 64)))
           (in-shape (vt-shape condition))
           (rank (length in-shape))
           (in-data (vt-data condition))
           (in-offset (vt-offset condition))
           (shape-vec (coerce in-shape 'simple-vector))
           (strides-vec (coerce (vt-strides condition) 'simple-vector))
           ;; 中间收集器，使用 64 位防止遍历时越界
           (result-indices (make-array 128
                                       :element-type '(signed-byte 64)
                                       :fill-pointer 0
                                       :adjustable t))
           (coord-buffer (make-array rank 
                                     :element-type '(signed-byte 64))))
      (declare (type simple-vector shape-vec strides-vec)
               (type fixnum rank in-offset))

      (macrolet
          ((gen-recurse (test-fn)
             `(labels
                  ((recurse (depth current-ptr)
                     (declare (type fixnum depth current-ptr))
                     (with-float-safe
                       (if (= depth rank)
                           (when (funcall ,test-fn in-data current-ptr)
                             (if (zerop rank)
                                 (vector-push-extend 0 result-indices)
                                 (loop for c across coord-buffer do
                                   (vector-push-extend c result-indices))))
                           (let ((dim (svref shape-vec depth))
                                 (stride (svref strides-vec depth)))
                             (declare (type fixnum dim stride))
                             (loop for i fixnum from 0 below dim
                                   do (setf (aref coord-buffer depth) i)
                                      (recurse (1+ depth)
                                               (+ current-ptr
                                                  (* i stride)))))))))
                (recurse 0 in-offset))))

        (etypecase in-data
          ((simple-array double-float (*))
           (gen-recurse
            (lambda (data ptr)
              (declare (type (simple-array double-float (*)) data)
                       (type fixnum ptr))
              (/= (aref data ptr) 0.0d0))))
          ((simple-array fixnum (*))
           (gen-recurse
            (lambda (data ptr)
              (declare (type (simple-array fixnum (*)) data)
                       (type fixnum ptr))
              (/= (the fixnum (aref data ptr)) 0))))
          ((simple-array single-float (*))
           (gen-recurse
            (lambda (data ptr)
              (declare (type (simple-array single-float (*)) data)
                       (type fixnum ptr))
              (/= (aref data ptr) 0.0f0))))
          ((simple-array (signed-byte 64) (*))
           (gen-recurse
            (lambda (data ptr)
              (declare (type (simple-array (signed-byte 64) (*)) data)
                       (type fixnum ptr))
              (/= (the (signed-byte 64) (aref data ptr)) 0))))
          ((simple-array (signed-byte 32) (*))
           (gen-recurse
            (lambda (data ptr)
              (declare (type (simple-array (signed-byte 32) (*)) data)
                       (type fixnum ptr))
              (/= (the (signed-byte 32) (aref data ptr)) 0)))))

        (let* ((total-indices (length result-indices))
               (count (if (zerop rank)
                          total-indices
                          (floor total-indices rank))))
          (cond
            ((zerop count)
             (vt-zeros (list 0 rank) :dtype final-dtype))
            ((> rank 0)
             ;; 严格按照用户指定的 lisp-type 创建最终底层数组
             (let ((final-data (make-array total-indices
                                            :element-type lisp-type)))
               (loop for i from 0 below total-indices
                     for idx across result-indices
                     do (setf (aref final-data i) idx))
               (%make-vt :data final-data
                         :shape (list count rank)
                         :strides (list rank 1)
                         :offset 0
                         :dtype final-dtype)))
            (t
             (%make-vt :data (make-array 0 
                                          :element-type lisp-type)
                       :shape (list count 0)
                       :strides (list 0 1)
                       :offset 0
                       :dtype final-dtype))))))))
