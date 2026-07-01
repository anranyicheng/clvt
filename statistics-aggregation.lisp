(in-package :clvt)

(defun vt-maximum (t1 t2 &key out dtype)
  "逐元素取两数组中较大者。NaN 会传播。"
  (with-float-safe
    (vt-map (lambda (a b)
              (cond ((vt-float-nan-p a) a)
                    ((vt-float-nan-p b) b)
                    (t (max a b))))
            t1 t2 :out out :dtype dtype)))

(defun vt-minimum (t1 t2 &key out dtype)
  "逐元素取两数组中较小者。NaN 会传播。"
  (with-float-safe
    (vt-map (lambda (a b)
              (cond ((vt-float-nan-p a) a)
                    ((vt-float-nan-p b) b)
                    (t (min a b))))
            t1 t2 :out out :dtype dtype)))

(defun vt-get-axes-and-count (axis rank shape)
  "返回 (values real-axes count)。
   real-axes 为排序后的列表，count 为归约轴上的元素总乘积。"
  (let* ((raw-axes (if (null axis)
                       nil
                       (if (listp axis) axis (list axis))))
         (real-axes (when raw-axes
                      (sort (mapcar (lambda (a)
                                      (vt-normalize-axis a rank))
                                    raw-axes)
                            #'<)))
         (count (if real-axes
                    (reduce #'* (mapcar (lambda (a)
                                          (the fixnum (nth a shape)))
                                        real-axes)
                            :initial-value 1)
                    (the fixnum (reduce #'* shape :initial-value 1)))))
    (values real-axes count)))

(defun vt-sum (tensor &key axis keepdims dtype out)
  "求和. 自动适配 int/float 类型. 支持多轴归约."
  (with-float-safe
    (let ((element-type (array-element-type (vt-data tensor))))
      (nth-value
       0
       (vt-reduce tensor axis
                  (get-reduction-identity :sum element-type)
                  (lambda (acc val)
                    (declare (type number acc val))
                    (values (+ acc val) nil))
                  :out out :dtype dtype
                  :return-arg nil :keepdims keepdims)))))

(defun vt-amax (tensor &key axis keepdims dtype out)
  "最大值，若包含 nan 则结果为 nan（索引指向第一个 nan）。
   类型自动适配。支持多轴归约。"
  (with-float-safe
    (let ((element-type (array-element-type (vt-data tensor))))
      (nth-value
       0
       (vt-reduce tensor axis
                  (get-reduction-identity :max element-type)
                  (lambda (acc val)
                    (declare (type number acc val))
                    (cond ((vt-float-nan-p val)
                           (if (vt-float-nan-p acc)
                               (values acc nil)
                               (values val t)))
                          ((vt-float-nan-p acc) (values acc nil))
                          (t (if (> val acc)
                                 (values val t)
                                 (values acc nil)))))
                  :out out :dtype dtype
                  :return-arg nil :keepdims keepdims)))))

(defun vt-amin (tensor &key axis keepdims dtype out)
  "最小值，若包含 nan 则结果为 nan（索引指向第一个 nan）。支持多轴归约。"
  (with-float-safe
    (let ((element-type (array-element-type (vt-data tensor))))
      (nth-value
       0
       (vt-reduce tensor axis
                  (get-reduction-identity :min element-type)
                  (lambda (acc val)
                    (declare (type number acc val))
                    (cond ((vt-float-nan-p val)
                           (if (vt-float-nan-p acc)
                               (values acc nil)
                               (values val t)))
                          ((vt-float-nan-p acc) (values acc nil))
                          (t (if (< val acc)
                                 (values val t)
                                 (values acc nil)))))
                  :out out :dtype dtype
                  :return-arg nil :keepdims keepdims)))))

(defun vt-argmax (tensor &key axis out)
  "返回最大值索引，如果存在 nan 则返回第一个 nan 的索引。"
  (with-float-safe
    (let ((element-type (array-element-type (vt-data tensor))))
      (nth-value
       1
       (vt-reduce tensor axis
                  (get-reduction-identity :max element-type)
                  (lambda (acc val)
                    (declare (type number acc val))
                    (cond ((vt-float-nan-p val)
                           (if (vt-float-nan-p acc)
                               (values acc nil)
                               (values val t)))
                          ((vt-float-nan-p acc) (values acc nil))
                          (t (if (> val acc)
                                 (values val t)
                                 (values acc nil)))))
                  :out out :return-arg t)))))

(defun vt-argmin (tensor &key axis out)
  "返回最小值索引，如果存在 nan 则返回第一个 nan 的索引。"
  (with-float-safe
    (let ((element-type (array-element-type (vt-data tensor))))
      (nth-value
       1
       (vt-reduce tensor axis
                  (get-reduction-identity :min element-type)
                  (lambda (acc val)
                    (declare (type number acc val))
                    (cond ((vt-float-nan-p val)
                           (if (vt-float-nan-p acc)
                               (values acc nil)
                               (values val t)))
                          ((vt-float-nan-p acc) (values acc nil))
                          (t (if (< val acc)
                                 (values val t)
                                 (values acc nil)))))
                  :out out :return-arg t)))))

(defun vt-mean (tensor &key axis keepdims dtype out)
  "计算平均值. 支持多轴归约。
   优化: 透传 dtype 和 out，底层自动处理类型校验与原地写入。"
  (with-float-safe
    (let* ((shape (vt-shape tensor))
           (rank (length shape)))
      (multiple-value-bind (real-axes count)
          (vt-get-axes-and-count axis rank shape)
        ;; 仅在 dtype 和 out 缺失时推导默认浮点类型
        (let ((final-dtype (or dtype
                               (and out (vt-dtype out))
                               (if (eq (vt-dtype tensor) :float32)
                                   :float32 :float64))))
          (when (= count 0)
            (let ((nan-val (vt-get-nan final-dtype))
                  (out-shape (if keepdims
                                 (loop for d in shape for i below rank
                                       collect (if (or (null real-axes)
                                                       (member i real-axes))
                                                   1 d))
                                 (loop for d in shape for i below rank
                                       unless (or (null real-axes)
                                                  (member i real-axes))
                                       collect d))))
              (return-from vt-mean
                (if out
                    (vt-map (lambda (x)
                              (declare (ignore x))
                              nan-val)
                            out :dtype final-dtype :out out)
                    (vt-full out-shape nan-val :dtype final-dtype)))))

          (let* ((sum-result (vt-sum tensor :axis real-axes
                                     :keepdims keepdims
                                     :dtype final-dtype :out out))
                 (div (coerce count (if (eq final-dtype :float32)
                                        'single-float 'double-float))))
            (vt-map (lambda (s) (/ s div))
                    sum-result :dtype final-dtype :out sum-result)))))))

(defun vt-var (tensor &key axis keepdims (ddof 0) dtype out)
  "计算方差。支持多轴归约。
   优化: 透传 dtype 和 out，底层自动处理类型校验与原地写入。"
  (with-float-safe
    (let* ((shape (vt-shape tensor))
           (rank (length shape)))
      (multiple-value-bind (real-axes n)
          (vt-get-axes-and-count axis rank shape)
        (let* ((divisor (- n ddof))
               (final-dtype (or dtype
                                (and out (vt-dtype out))
                                (if (eq (vt-dtype tensor) :float32)
                                    :float32 :float64))))
          (if (<= divisor 0)
              (vt-map (lambda (s)
                        (declare (ignore s))
                        (vt-get-nan final-dtype))
                      (vt-sum tensor :axis real-axes :keepdims keepdims)
                      :dtype final-dtype :out out)
              (let* ((mean-val (vt-mean tensor :axis real-axes
                                        :keepdims t :dtype final-dtype))
                     (sq-diff (vt-square (vt-- tensor mean-val
                                               :dtype final-dtype)
                                         :dtype final-dtype))
                     (sum-sq (vt-sum sq-diff :axis real-axes
                                     :keepdims keepdims
                                     :dtype final-dtype :out out)))
                (vt-/ sum-sq divisor :dtype final-dtype :out sum-sq))))))))

(defun vt-std (tensor &key axis keepdims (ddof 0) dtype out)
  "计算标准差。支持多轴归约。"
  (with-float-safe
    (let* ((final-dtype (or dtype
                            (and out (vt-dtype out))
                            (if (eq (vt-dtype tensor) :float32)
                                :float32 :float64)))
           (variance (vt-var tensor :axis axis :keepdims keepdims
                            :ddof ddof :dtype final-dtype :out out)))
      (vt-sqrt variance :dtype final-dtype :out variance))))

(defun nan-stats-helpers (tensor &key axis keepdims)
  "返回两个值：将 nan 转为 0 的张量和有效元素计数。支持多轴。"
  (with-float-safe
    (let* ((mask (vt-isnan tensor))
           (not-nan (vt-logical-not mask))
           (zero-val (if (eq (vt-dtype tensor) :float32) 0.0s0 0.0d0))
           (clean (vt-where mask zero-val tensor))
           (count (vt-sum not-nan :axis axis :keepdims keepdims
                          :dtype :int64)))
      (values clean count))))

(defun vt-nansum (tensor &key axis keepdims dtype out)
  "忽略 nan 的元素求和。支持多轴。"
  (with-float-safe
    (let ((clean (nan-stats-helpers tensor :axis axis :keepdims nil)))
      (vt-sum clean :axis axis :keepdims keepdims
              :dtype dtype :out out))))

(defun vt-nanmean (tensor &key axis keepdims dtype out)
  "忽略 nan 计算均值。支持多轴。"
  (with-float-safe
    (multiple-value-bind (clean count)
        (nan-stats-helpers tensor :axis axis :keepdims keepdims)
      (let* ((final-dtype (or dtype
                              (and out (vt-dtype out))
                              (if (eq (vt-dtype tensor) :float32)
                                  :float32 :float64)))
             (nan-val (vt-get-nan final-dtype))
             (sum (vt-sum clean :axis axis :keepdims keepdims
                          :dtype final-dtype :out out)))
        (vt-map (lambda (s c) (if (zerop c) nan-val (/ s c)))
                sum count :dtype final-dtype :out sum)))))

(defun vt-nanvar (tensor &key axis keepdims (ddof 0) dtype out)
  "忽略 nan 计算方差。支持多轴。"
  (with-float-safe
    (let* ((mask (vt-isnan tensor))
           (not-nan (vt-logical-not mask))
           (final-dtype (or dtype
                            (and out (vt-dtype out))
                            (if (eq (vt-dtype tensor) :float32)
                                :float32 :float64)))
           (nan-val (vt-get-nan final-dtype))
           (zero-val (if (eq final-dtype :float32) 0.0s0 0.0d0))
           (clean (vt-where mask zero-val tensor))
           (count (vt-sum not-nan :axis axis :keepdims keepdims
                          :dtype :int64))
           (mean (vt-nanmean tensor :axis axis :keepdims t
                             :dtype final-dtype))
           (squared-diff (vt-* (vt-map (lambda (c m) (* (- c m) (- c m)))
                                       clean mean :dtype final-dtype)
                               not-nan :dtype final-dtype))
           (sum2 (vt-sum squared-diff :axis axis :keepdims keepdims
                         :dtype final-dtype :out out))
           (divisor (vt-map (lambda (c) (max 0 (- c ddof)))
                            count :dtype :int64)))
      (vt-map (lambda (s d) (if (<= d 0) nan-val (/ s d)))
              sum2 divisor :dtype final-dtype :out sum2))))

(defun vt-nanstd (tensor &key axis keepdims (ddof 0) dtype out)
  "忽略 nan 计算标准差。支持多轴。"
  (with-float-safe
    (let* ((final-dtype (or dtype
                            (and out (vt-dtype out))
                            (if (eq (vt-dtype tensor) :float32)
                                :float32 :float64)))
           (var (vt-nanvar tensor :axis axis :keepdims keepdims
                           :ddof ddof :dtype final-dtype :out out)))
      (vt-sqrt var :dtype final-dtype :out var))))

(defun vt-nanmax (tensor &key axis keepdims out)
  "忽略 nan 的最大值。对标 NumPy: 若沿轴全为 nan 则返回 nan。
   支持多轴。"
  (if (member (vt-dtype tensor) '(:int32 :int64))
      (vt-amax tensor :axis axis :keepdims keepdims :out out)
      (with-float-safe
        (let* ((infer-dtype (vt-dtype tensor))
               (neg-inf (vt-get-neg-inf infer-dtype))
               (nan-val (vt-get-nan infer-dtype))
               (mask (vt-isnan tensor))
               (clean (vt-where mask neg-inf tensor))
               (result (vt-amax clean :axis axis :keepdims keepdims
                                :out out)))
          (vt-where (vt-all mask :axis axis :keepdims keepdims)
                    nan-val result :out result)))))

(defun vt-nanmin (tensor &key axis keepdims out)
  "忽略 nan 的最小值。对标 NumPy: 若沿轴全为 nan 则返回 nan。
   支持多轴。"
  (if (member (vt-dtype tensor) '(:int32 :int64))
      (vt-amin tensor :axis axis :keepdims keepdims :out out)
      (with-float-safe
        (let* ((infer-dtype (vt-dtype tensor))
               (pos-inf (vt-get-pos-inf infer-dtype))
               (nan-val (vt-get-nan infer-dtype))
               (mask (vt-isnan tensor))
               (clean (vt-where mask pos-inf tensor))
               (result (vt-amin clean :axis axis :keepdims keepdims
                                :out out)))
          (vt-where (vt-all mask :axis axis :keepdims keepdims)
                    nan-val result :out result)))))
