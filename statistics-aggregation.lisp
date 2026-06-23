(in-package :clvt)

(defun vt-maximum (t1 t2)
  "逐元素取两数组中较大者。NaN 会传播。"
  (with-float-safe 
    (vt-map (lambda (a b)
              (cond ((vt-float-nan-p a) a) ; 如果 a 是 NaN，返回 NaN
                    ((vt-float-nan-p b) b) ; 如果 b 是 NaN，返回 NaN
                    (t (max a b))))
            t1 t2)))

(defun vt-minimum (t1 t2)
  "逐元素取两数组中较小者。NaN 会传播。"
  (with-float-safe 
    (vt-map (lambda (a b)
              (cond ((vt-float-nan-p a) a) ; 如果 a 是 NaN，返回 NaN
                    ((vt-float-nan-p b) b) ; 如果 b 是 NaN，返回 NaN
                    (t (min a b))))
            t1 t2)))

(defun vt-sum (tensor &key axis keepdims)
  "求和. 自动适配 int/float 类型."
  (with-float-safe
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
		  :keepdims keepdims)))))

(defun vt-amax (tensor &key axis keepdims)
  "最大值，若包含 nan 则结果为 nan（索引指向第一个 nan）。
   类型自动适配。"
  (with-float-safe
    (let ((element-type (array-element-type (vt-data tensor))))
      (nth-value 0 
        (vt-reduce tensor axis (get-reduction-identity :max element-type)
                   (lambda (acc val)
                     (declare (type number acc val))
                     (cond ((vt-float-nan-p val)
                            ;; 遇到 nan
                            (if (vt-float-nan-p acc)
                                (values acc nil) ;; 已为 nan，保持原索引
                                (values val t))) ;; 首次 nan，更新值并标记更新索引
                           ((vt-float-nan-p acc)
                            ;; acc 已是 nan，保持
                            (values acc nil))
                           (t
                            ;; 正常数值比较
                            (if (> val acc)
				(values val t)
				(values acc nil)))))
                   :return-arg nil :keepdims keepdims)))))

(defun vt-amin (tensor &key axis keepdims)
  "最小值，若包含 nan 则结果为 nan（索引指向第一个 nan）。"
  (with-float-safe
    (let ((element-type (array-element-type (vt-data tensor))))
      (nth-value 0 
        (vt-reduce tensor axis (get-reduction-identity :min element-type)
                   (lambda (acc val)
                     (declare (type number acc val))
                     (cond ((vt-float-nan-p val)
                            (if (vt-float-nan-p acc)
                                (values acc nil)
                                (values val t)))
                           ((vt-float-nan-p acc)
                            (values acc nil))
                           (t
                            (if (< val acc)
				(values val t)
				(values acc nil)))))
                   :return-arg nil :keepdims keepdims)))))

(defun vt-argmax (tensor &key axis)
  "返回最大值索引，如果存在 nan 则返回第一个 nan 的索引。"
  (with-float-safe
    (let ((element-type (array-element-type (vt-data tensor))))
      (nth-value 1 
        (vt-reduce tensor axis (get-reduction-identity :max element-type)
                   (lambda (acc val)
                     (declare (type number acc val))
                     (cond ((vt-float-nan-p val)
                            (if (vt-float-nan-p acc)
                                (values acc nil)
                                (values val t)))
                           ((vt-float-nan-p acc)
                            (values acc nil))
                           (t
                            (if (> val acc)
				(values val t)
				(values acc nil)))))
                   :return-arg t)))))

(defun vt-argmin (tensor &key axis)
  "返回最小值索引，如果存在 nan 则返回第一个 nan 的索引。"
  (with-float-safe
    (let ((element-type (array-element-type (vt-data tensor))))
      (nth-value 1 
        (vt-reduce tensor axis (get-reduction-identity :min element-type)
                   (lambda (acc val)
                     (declare (type number acc val))
                     (cond ((vt-float-nan-p val)
                            (if (vt-float-nan-p acc)
                                (values acc nil)
                                (values val t)))
                           ((vt-float-nan-p acc)
                            (values acc nil))
                           (t
                            (if (< val acc)
				(values val t)
				(values acc nil)))))
                   :return-arg t)))))

(defun vt-mean (tensor &key axis keepdims)
  "计算平均值. axis: nil (全局) 或 fixnum (支持负数).
    返回: double-float 或 vt 张量."
  (with-float-safe
    (let* ((shape (vt-shape tensor))
           (rank (length shape))
           (real-axis (vt-normalize-axis axis rank))
           (sum-result
	     (vt-sum tensor :axis real-axis :keepdims keepdims))
           (count (if real-axis
                      (the fixnum (nth real-axis shape))
                      (the fixnum (reduce #'* shape)))))
      ;; 避免除以 0
      (when (= count 0)
	(return-from vt-mean
	  (vt-full (vt-shape sum-result)
		   +vt-float-nan+ :type 'double-float)))
      ;; 执行除法
      (vt-map (lambda (s)
		(/ s (coerce count 'double-float)))
	      sum-result))))

(defun vt-average (tensor weights &key axis keepdims)
  "计算加权平均值.
  严格对标 NumPy:
  - axis 为整数时: weights 必须为 1D 且长度等于 a.shape[axis]，沿 axis 广播
  - axis 为 nil 时: weights 必须与 tensor 形状完全一致
  - 权重和为零或 NaN 时：返回 NaN（数值库惯例）或抛出错误。"
  (declare (vt tensor weights))
  (with-float-safe
    (let ((a-shape (vt-shape tensor))
          (w-shape (vt-shape weights)))
      (if axis
          ;; === 分支 1: 沿指定轴 ===
          (let* ((rank (length a-shape))
                 (ax (vt-normalize-axis axis rank))
                 (ax-size (nth ax a-shape)))
            ;; 形状校验
            (unless (and (= (length w-shape) 1) (= (first w-shape) ax-size))
              (error "ValueError: 1D weights expected when axis is specified.~@
                      weights shape ~a does not match a.shape[~a] = ~a"
                     w-shape ax ax-size))            
            (let* ((expanded-shape (loop for i below rank collect (if (= i ax) ax-size 1)))
                   (expanded-w (vt-reshape weights expanded-shape))
                   ;; 计算加权和
                   (weighted-sum (vt-sum (vt-map #'* tensor expanded-w) :axis ax :keepdims keepdims))
                   ;; 计算权重和 (标量)
                   (sum-weights (vt-item (vt-sum weights))))              
              (cond
                ;; 情况 1: 权重和为 NaN -> 结果传播 NaN
                ((vt-float-nan-p sum-weights)
                 (vt-full (vt-shape weighted-sum) +vt-float-nan+ :type 'double-float))
                ;; 情况 2: 权重和为 0 -> 报错 (NumPy 标准行为)
                ((zerop sum-weights)
                 (error "ZeroDivisionError: Weights sum to zero, can't be normalized"))
                ;; 情况 3: 正常计算
                (t
                 (vt-map (lambda (s) (/ s sum-weights 1.0d0)) weighted-sum)))))          
          ;; === 分支 2: 全局 (axis=nil) ===
          (progn
            ;; 形状校验
            (unless (equal w-shape a-shape)
              (error "ValueError: weights must have the same shape as a when axis is None.~@
                      weights shape ~a vs a shape ~a" w-shape a-shape))            
            (let* ((weighted-sum (vt-sum (vt-map #'* tensor weights) :keepdims keepdims))
                   (sum-weights (vt-item (vt-sum weights))))
              (cond
                ((vt-float-nan-p sum-weights)
                 (vt-full (vt-shape weighted-sum) +vt-float-nan+ :type 'double-float))
                ((zerop sum-weights)
                 (error "ZeroDivisionError: Weights sum to zero, can't be normalized"))
                (t
                 (vt-map (lambda (s) (/ s sum-weights 1.0d0)) weighted-sum)))
              ))))))


(defun vt-var (tensor &key axis keepdims (ddof 0))
  "计算方差。
   ddof : delta degrees of freedom（自由度修正值）。默认为 0（总体方差）。
   设为 1 则计算无偏样本方差。
   当 n - ddof <= 0 时，遵循 numpy 规范返回 nan。"
  (with-float-safe
    (let* ((shape (vt-shape tensor))
           (rank (length shape))
           (real-axis (when axis (vt-normalize-axis axis rank)))
           ;; 1. 计算均值（强制 keepdims 以便后续广播）
           (mean-val (vt-mean tensor :axis real-axis :keepdims t))
           ;; 2. 计算偏差平方
           (sq-diff (vt-square (vt-- tensor mean-val)))
           ;; 3. 计算平方和
           (sum-sq (vt-sum sq-diff :axis real-axis :keepdims keepdims))
           ;; 4. 计算元素个数 n
           (n (if real-axis 
                  (the fixnum (nth real-axis shape))
                  (the fixnum (reduce #'* shape))))
           ;; 5. 计算除数
           (divisor (- n ddof)))
      ;; 6. 判断除数是否合法
      (if (<= divisor 0)
          ;; 分母不合法（<=0），直接将 sum-sq 映射为全 nan 张量
          ;; vt-map 天然支持 0 维张量，完美契合 PyTorch 规范
          (vt-map (lambda (s)
                    (declare (ignore s))
                    +vt-float-nan+)
                  sum-sq)
          ;; 分母合法，执行正常除法
          (vt-/ sum-sq divisor)))))


(defun vt-std (tensor &key axis keepdims (ddof 0))
  "计算标准差。
   ddof : 自由度修正值，默认为 0。设为 1 计算无偏样本标准差。"
  (with-float-safe
    (let ((variance (vt-var tensor :axis axis
				   :keepdims keepdims
				   :ddof ddof)))
      (vt-sqrt variance))))

(defun nan-stats-helpers (tensor &key axis keepdims)
  "返回两个值：将 nan 转为 0 的张量 (形状同 tensor) 和有效元素计数 (标量或张量)。"
  (with-float-safe
    (let* ((mask (vt-isnan tensor))
           ;; 非 nan 位置为 1，nan 位置为 0
           (not-nan (vt-logical-not mask))
           ;; clean: nan 处填 0，其余不变
           (clean (vt-where mask 0.0d0 tensor))
           ;; 有效计数
           (count (vt-sum not-nan :axis axis :keepdims keepdims)))
      (values clean count))))

(defun vt-nansum (tensor &key axis keepdims)
  "忽略 nan 的元素求和。axis 和 keepdims 行为等同于 numpy。"
  (with-float-safe
    (let ((clean (nan-stats-helpers tensor :axis axis :keepdims nil)))
      (vt-sum clean :axis axis :keepdims keepdims))))

(defun vt-nanmean (tensor &key axis keepdims)
  "忽略 nan 计算均值。axis 和 keepdims 行为等同于 numpy。"
  (with-float-safe
    (multiple-value-bind (clean count)
	(nan-stats-helpers tensor :axis axis :keepdims keepdims)
      (let ((sum (vt-sum clean :axis axis :keepdims keepdims)))
	;; 防止除零，当 count 为 0 时返回 nan
	(labels ((safe-div (s c)
                   (if (zerop c)
                       (vt-float-nan)  ;; 产生 nan
                       (/ s c))))
              (vt-map #'safe-div sum count))))))

(defun vt-nanvar (tensor &key axis keepdims (ddof 0))
  "忽略 nan 计算方差 (对标 PyTorch，归约始终返回张量)。
   ddof ：自由度修正（默认 0，总体方差；1=样本方差）。
   axis, keepdims 同 PyTorch。"
  (with-float-safe
    (let* ((mask (vt-isnan tensor))
           (not-nan (vt-logical-not mask))  ;; 非 nan 为 1，nan 为 0
           (clean (vt-where mask (vt-zeros-like tensor) tensor))
           (count (vt-sum not-nan :axis axis :keepdims keepdims))
           (mean (vt-nanmean tensor :axis axis :keepdims t))
           (squared-diff (vt-* (vt-map (lambda (c m) (* (- c m) (- c m)))
                                       clean mean)
                               not-nan))
           (sum2 (vt-sum squared-diff :axis axis :keepdims keepdims))
           (divisor (vt-map (lambda (c) (max 0 (- c ddof))) count)))
      (vt-map (lambda (s d)
                (if (<= d 0)
                    (vt-float-nan)
                    (/ s d)))
              sum2 divisor))))

(defun vt-nanstd (tensor &key axis keepdims (ddof 0))
  "忽略 nan 计算标准差。参数同 vt-nanvar。"
  (with-float-safe
    (let ((var (vt-nanvar tensor :axis axis :keepdims keepdims
				 :ddof ddof)))
      (vt-map #'sqrt var))))

(defun vt-nanmax (tensor &key axis keepdims)
  "忽略 nan 的最大值。对标 NumPy: 若沿轴全为 nan 则返回 nan。"
  (if (eq (vt-element-type tensor) 'fixnum)
      (vt-amax tensor :axis axis :keepdims keepdims)
      (with-float-safe
        (let* ((mask (vt-isnan tensor))
               (clean (vt-where mask +vt-float-neg-inf+ tensor))
               (result (vt-amax clean :axis axis :keepdims keepdims)))
          (vt-where (vt-all mask :axis axis :keepdims keepdims)
                    +vt-float-nan+
                    result)))))

(defun vt-nanmin (tensor &key axis keepdims)
  "忽略 nan 的最小值。对标 NumPy: 若沿轴全为 nan 则返回 nan。"
  (if (eq (vt-element-type tensor) 'fixnum)
      (vt-amin tensor :axis axis :keepdims keepdims)
      (with-float-safe
        (let* ((mask (vt-isnan tensor))
               (clean (vt-where mask +vt-float-pos-inf+ tensor))
               (result (vt-amin clean :axis axis :keepdims keepdims)))
          (vt-where (vt-all mask :axis axis :keepdims keepdims)
                    +vt-float-nan+
                    result)))))
