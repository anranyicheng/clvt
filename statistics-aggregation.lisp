(in-package :clvt)

(defun vt-sum (tensor &key axis)
  "求和.自动适配 int/float 类型."
  (let ((element-type (array-element-type (vt-data tensor))))
    ;; 获取类型匹配的初始值 0
    (nth-value
     0 
     (vt-reduce tensor axis (get-reduction-identity :sum element-type)
                (lambda (acc val)
                  (declare (type number acc val))
                  ;; 简单的加法,类型转换由 vt-reduce 内部处理
                  (values (+ acc val) nil))
                :return-arg nil))))

(defun vt-amax (tensor &key axis)
  "最大值.自动适配类型."
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
                :return-arg nil))))

(defun vt-amin (tensor &key axis)
  "最小值.自动适配类型."
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
                :return-arg nil))))

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

(defun vt-mean (tensor &key axis)
  "计算平均值.
   axis: nil (全局) 或 fixnum (轴向).
   返回: double-float 或 VT 张量."
  (let* ((sum-result (vt-sum tensor :axis axis))
         (element-type (vt-element-type tensor))
         ;; 计算归约维度的元素个数
         (count (if axis
                    ;; 轴向归约:除数是该轴的长度
                    (the fixnum (nth axis (vt-shape tensor)))
                    ;; 全局归约:除数是总元素数
                    (the fixnum (reduce #'* (vt-shape tensor))))))
    
    ;; 避免除以 0 (虽然理论上 shape 维度不会为 0)
    (when (= count 0) (return-from vt-mean 0.0d0))
    ;; 执行除法
    ;; 如果 sum-result 是张量 (axis 模式),vt-map 会自动广播标量 count
    ;; 如果 sum-result 是数值 (全局模式),直接除
    (if (vt-p sum-result)
        (vt-map (lambda (s) (/ s (coerce count element-type)))
		sum-result)
        (/ sum-result (coerce count element-type)))))

(defun vt-average (tensor weights &key axis)
  "计算加权平均值.
   tensor: 输入张量.
   weights: 权重张量 (形状必须可与 tensor 广播).
   axis: 归约轴."
  (declare (vt tensor weights))
  ;; 1. 计算加权和: Sum(X * W)
  ;; vt-map 会自动处理广播,生成中间张量,然后归约
  (let* ((weighted-sum (vt-sum (vt-map #'* tensor weights) :axis axis))
         ;; 2. 计算权重和: Sum(W)
         ;; 必须对 weights 进行同样方式的归约
         (sum-weights (vt-sum weights :axis axis)))
    
    ;; 3. 相除: Sum(X*W) / Sum(W)
    ;; 这里的除法利用 vt-map 自动广播,处理张量/张量的点对点除法
    (if (vt-p weighted-sum)
        (vt-map (lambda (s w) 
                  (if (= w 0) 0.0d0 (/ s w))) 
                weighted-sum 
                sum-weights)
        (if (= sum-weights 0)
	    0.0d0 
	    (/ weighted-sum sum-weights)))))

(defun vt-var (tensor &key axis)
  "计算方差."
  (let* ((mean-val (vt-mean tensor :axis axis)))
    ;; 1. 计算差值
    ;; 注意:mean-val 在 axis 模式下是张量,vt-map 会自动广播
    (let ((diff (vt-map #'- tensor mean-val)))  
      ;; 2. 计算差值的平方
      (let ((sq-diff (vt-map (lambda (x) (* x x)) diff)))
        ;; 3. 对平方求均值 (方差即差平方的均值)
        ;; 复用 vt-mean 避免重复写除法逻辑
        (vt-mean sq-diff :axis axis)))))

(defun vt-std (tensor &key axis)
  "计算标准差."
  (let ((variance (vt-var tensor :axis axis)))
    ;; 对方差开根号
    (if (vt-p variance)
        (vt-map #'sqrt variance)
        (sqrt variance))))

