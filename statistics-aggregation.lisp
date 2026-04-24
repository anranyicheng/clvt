(in-package :clvt)

(defun vt-maximum (t1 t2)
  "逐元素取两数组中较大者"
  (vt-map (lambda (a b) (max a b)) t1 t2))

(defun vt-minimum (t1 t2)
  "逐元素取两数组中较小者"
  (vt-map (lambda (a b) (min a b)) t1 t2))

(defun vt-sum (tensor &key axis keepdims)
  "求和. 自动适配 int/float 类型."
  (let ((element-type (array-element-type (vt-data tensor))))
    ;; 获取类型匹配的初始值 0
    (nth-value
     0 
     (vt-reduce tensor axis (get-reduction-identity :sum element-type)
                (lambda (acc val)
                  (declare (type number acc val))
                  ;; 简单的加法,类型转换由 vt-reduce 内部处理
                  (values (+ acc val) nil))
                :return-arg nil
		:keepdims keepdims))))

(defun vt-amax (tensor &key axis keepdims)
  "最大值. 自动适配类型."
  (let ((element-type (array-element-type (vt-data tensor))))
    ;; 获取类型匹配的最小值 (如 most-negative-fixnum)
    (nth-value
     0 
     (vt-reduce tensor axis (get-reduction-identity :max element-type)
                (lambda (acc val)
                  (declare (type number acc val))
                  (if (> val acc)
                      (values val t) ; 返回新值,标记更新索引
                      (values acc nil)))
                :return-arg nil
		:keepdims keepdims))))

(defun vt-amin (tensor &key axis keepdims)
  "最小值. 自动适配类型."
  (let ((element-type (array-element-type (vt-data tensor))))
    ;; 获取类型匹配的最大值
    (nth-value
     0 
     (vt-reduce tensor axis (get-reduction-identity :min element-type)
                (lambda (acc val)
                  (declare (type number acc val))
                  (if (< val acc)
                      (values val t)
                      (values acc nil)))
                :return-arg nil
		:keepdims keepdims))))

(defun vt-argmax (tensor &key axis)
  "最大值索引."
  (let ((element-type (array-element-type (vt-data tensor))))
    ;; 初始值同样需要是最大下界
    (nth-value
     1 
     (vt-reduce tensor axis (get-reduction-identity :max element-type)
                (lambda (acc val)
                  (declare (type number acc val))
                  (if (> val acc)
                      (values val t) ; 发现更大值,更新值,并通知更新索引
                      (values acc nil)))
                :return-arg t))))

(defun vt-argmin (tensor &key axis)
  "最小值索引."
  (let ((element-type (array-element-type (vt-data tensor))))
    (nth-value
     1 
     (vt-reduce tensor axis (get-reduction-identity :min element-type)
                (lambda (acc val)
                  (declare (type number acc val))
                  (if (< val acc)
                      (values val t)
                      (values acc nil)))
                :return-arg t))))

(defun vt-mean (tensor &key axis keepdims)
  "计算平均值. axis: nil (全局) 或 fixnum (支持负数).
    返回: double-float 或 VT 张量."
  (let* ((shape (vt-shape tensor))
         (rank (length shape))
         (real-axis (vt-normalize-axis axis rank))
         (sum-result
	   (vt-sum tensor :axis real-axis :keepdims keepdims))
         (element-type (vt-element-type tensor))
         (count (if real-axis
                    (the fixnum (nth real-axis shape))
                    (the fixnum (reduce #'* shape)))))
    ;; 避免除以 0
    (when (= count 0)
      (return-from vt-mean 0.0d0))
    ;; 执行除法
    (if (vt-p sum-result)
        (vt-map (lambda (s)
		  (/ s (coerce count element-type)))
		sum-result)
        (/ sum-result (coerce count element-type)))))

(defun vt-average (tensor weights &key axis keepdims)
  "计算加权平均值.
   tensor: 输入张量.
   weights: 权重张量 (形状必须可与 tensor 广播).
   axis: 归约轴.
   keepdims: 是否保持维度."
  (declare (vt tensor weights))
  (let* ((weighted-sum (vt-sum (vt-map #'* tensor weights)
			       :axis axis
			       :keepdims keepdims))
         (sum-weights (vt-sum weights :axis axis
				      :keepdims keepdims)))
    (if (vt-p weighted-sum)
        (vt-map (lambda (s w) 
                  (if (= w 0) 0.0d0 (/ s w))) 
                weighted-sum 
                sum-weights)
        (if (= sum-weights 0)
            0.0d0 
            (/ weighted-sum sum-weights)))))

(defun vt-var (tensor &key axis keepdims)
  "计算方差."
  (let* ((mean-val (vt-mean tensor :axis axis
				   :keepdims keepdims))
         (diff (vt-map #'- tensor mean-val))  
         (sq-diff (vt-map (lambda (x) (* x x)) diff)))
    (vt-mean sq-diff :axis axis :keepdims keepdims)))

(defun vt-std (tensor &key axis keepdims)
  "计算标准差."
  (let ((variance (vt-var tensor :axis axis
				 :keepdims keepdims)))
    (if (vt-p variance)
        (vt-map #'sqrt variance)
        (sqrt variance))))
