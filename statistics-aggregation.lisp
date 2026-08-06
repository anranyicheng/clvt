(in-package :clvt)

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

(defun vt-average (tensor weights &key axis keepdims dtype out)
  "计算加权平均值. 严格对标 NumPy np.average，并扩展支持 :dtype 和 :out 参数。"
  (declare (type vt tensor weights))
  (with-float-safe
    (let ((a-shape (vt-shape tensor))
          (w-shape (vt-shape weights))
          (effective-weights weights))
      ;; 1. 形状校验与权重广播准备
      (cond (axis (let* ((rank (length a-shape))
                         (ax (vt-normalize-axis axis rank))
                         (ax-size (nth ax a-shape)))
                    (unless (and (= (length w-shape) 1)
				 (= (first w-shape) ax-size))
                      (error "ValueError: 1D weights expected when axis is specified.~@ weights shape ~a does not match a.shape[~a] = ~a" w-shape ax ax-size))
                    (let ((expanded-shape (loop for i below rank
						collect (if (= i ax) ax-size 1))))
                      (setf effective-weights (vt-reshape weights expanded-shape)))))
            (t (unless (equal w-shape a-shape)
                 (error "ValueError: weights must have the same shape as a when axis is None.~@ weights shape ~a vs a shape ~a" w-shape a-shape))))
      
      ;; 2. 计算加权和与权重和
      (let* ((prod (vt-map #'* tensor effective-weights))
             (weighted-sum (vt-sum prod :axis axis :keepdims keepdims))
             (sum-weights (vt-item (vt-sum weights)))
             (in-dtype (vt-dtype weighted-sum))
             (need-promote (member in-dtype '(:int32 :int64 :int16 :int8 :uint8 :uint16)))
             
             (exec-dtype (cond
                           ((and out dtype (not (eq (vt-dtype out) dtype)))
                            (error "vt-average: :out 的类型 (~a) 与 :dtype (~a) 冲突" (vt-dtype out) dtype))
                           (out (vt-dtype out))
                           (dtype dtype)
                           (need-promote :float64)
                           (t in-dtype)))             
             (is-float32 (eq exec-dtype :float32))
             (nan-val (if is-float32
			  +vt-sfloat-nan+
			  +vt-dfloat-nan+))
             (scalar-divisor (coerce sum-weights (if is-float32
						     'single-float 'double-float)))
             (map-dtype
	       (cond (out nil)
		     (dtype dtype)
		     (need-promote :float64)
		     (t nil))))
        ;; 校验浮点合法性
        (unless (member exec-dtype '(:float32 :float64))
          (error "vt-average: 计算结果必须为浮点类型，收到 ~a" exec-dtype))
        (when out
          (unless (equal (vt-shape out)
			 (vt-shape weighted-sum))
            (error "vt-average: :out 张量形状 ~a 与期望结果形状 ~a 不匹配" (vt-shape out) (vt-shape weighted-sum))))
        
        (cond ((vt-float-nan-p sum-weights)
               (if out (progn (vt-map (lambda (x)
					(declare (ignorable x)) nan-val)
				      weighted-sum :out out :dtype dtype)
			      out)
                   (vt-full (vt-shape weighted-sum) nan-val :dtype exec-dtype)))
              ((zerop sum-weights)
               (error "ZeroDivisionError: Weights sum to zero, can't be normalized"))
              (t (vt-map (lambda (s) (/ s scalar-divisor))
			 weighted-sum :dtype map-dtype :out out)))))))

(defun vt-mean (tensor &key axis keepdims dtype out)
  "计算平均值. 支持多轴归约。"
  (with-float-safe
    (let* ((shape (vt-shape tensor))
           (rank (length shape)))
      (multiple-value-bind (real-axes count)
	  (vt-get-axes-and-count axis rank shape)
        (let* ((final-dtype (cond
                              ((and out dtype (not (eq (vt-dtype out) dtype)))
                               (error "vt-mean: :out 的类型 (~a) 与 :dtype (~a) 冲突" (vt-dtype out) dtype))
                              (out (vt-dtype out))
                              (dtype dtype)
                              (t (if (eq (vt-dtype tensor) :float32) :float32 :float64)))))
          (unless (member final-dtype '(:float32 :float64))
            (error "vt-mean: 均值结果必须为浮点类型，收到 ~a" final-dtype))
          
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
              (return-from vt-mean (if out
                                       (vt-map (lambda (x)
						 (declare (ignore x)) nan-val)
					       out :dtype final-dtype :out out)
                                       (vt-full out-shape nan-val :dtype final-dtype)))))
          
          (let* ((sum-result
		   (vt-sum tensor :axis real-axes :keepdims keepdims :dtype final-dtype :out out))
                 (div (coerce count (if (eq final-dtype :float32)
					'single-float 'double-float))))
            (vt-map (lambda (s) (/ s div))
		    sum-result :dtype final-dtype :out sum-result)))))))

(defun vt-var (tensor &key axis keepdims (ddof 0) dtype out)
  "计算方差。支持多轴归约。"
  (with-float-safe
    (let* ((shape (vt-shape tensor))
           (rank (length shape)))
      (multiple-value-bind (real-axes n)
	  (vt-get-axes-and-count axis rank shape)
        (let* ((divisor (- n ddof))
               (final-dtype (cond
                              ((and out dtype (not (eq (vt-dtype out) dtype)))
                               (error "vt-var: :out 的类型 (~a) 与 :dtype (~a) 冲突" (vt-dtype out) dtype))
                              (out (vt-dtype out))
                              (dtype dtype)
                              (t (if (eq (vt-dtype tensor) :float32)
				     :float32 :float64)))))
          (unless (member final-dtype '(:float32 :float64))
            (error "vt-var: 方差结果必须为浮点类型，收到 ~a" final-dtype))
          
          (if (<= divisor 0)
              (vt-map (lambda (s)
			(declare (ignore s))
			(vt-get-nan final-dtype))
                      (vt-sum tensor :axis real-axes :keepdims keepdims)
		      :dtype final-dtype :out out)
              (let* ((mean-val
		       (vt-mean tensor :axis real-axes :keepdims t :dtype final-dtype))
                     (sq-diff
		       (vt-square (vt-- tensor mean-val :dtype final-dtype)
				  :dtype final-dtype))
                     (sum-sq
		       (vt-sum sq-diff :axis real-axes :keepdims keepdims
				       :dtype final-dtype :out out)))
                (vt-/ sum-sq divisor :dtype final-dtype :out sum-sq))))))))

(defun vt-std (tensor &key axis keepdims (ddof 0) dtype out)
  "计算标准差。支持多轴归约。"
  (with-float-safe
    (let* ((final-dtype (cond
                          ((and out dtype (not (eq (vt-dtype out) dtype)))
                           (error "vt-std: :out 的类型 (~a) 与 :dtype (~a) 冲突" (vt-dtype out) dtype))
                          (out (vt-dtype out))
                          (dtype dtype)
                          (t (if (eq (vt-dtype tensor) :float32) :float32 :float64)))))
      (unless (member final-dtype '(:float32 :float64))
        (error "vt-std: 标准差结果必须为浮点类型，收到 ~a" final-dtype))
      (let ((variance
	      (vt-var tensor :axis axis :keepdims keepdims :ddof ddof
			     :dtype final-dtype :out out)))
        (vt-sqrt variance :dtype final-dtype :out variance)))))



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
      (let* ((final-dtype (cond
                            ((and out dtype (not (eq (vt-dtype out) dtype)))
                             (error "vt-nanmean: :out 与 :dtype 冲突"))
                            (out (vt-dtype out))
                            (dtype dtype)
                            (t (if (eq (vt-dtype tensor) :float32) :float32 :float64))))
             (nan-val (vt-get-nan final-dtype))
             (sum
	       (vt-sum clean :axis axis :keepdims keepdims :dtype final-dtype :out out)))
        (unless (member final-dtype '(:float32 :float64))
          (error "vt-nanmean: 结果必须为浮点类型"))
        (vt-map (lambda (s c)
		  (if (zerop c) nan-val (/ s c)))
		sum count :dtype final-dtype :out sum)))))

(defun vt-nanvar (tensor &key axis keepdims (ddof 0) dtype out)
  "忽略 nan 计算方差。支持多轴。"
  (with-float-safe
    (let* ((mask (vt-isnan tensor))
           (not-nan (vt-logical-not mask))
           (final-dtype (cond
                          ((and out dtype (not (eq (vt-dtype out) dtype)))
                           (error "vt-nanvar: :out 与 :dtype 冲突"))
                          (out (vt-dtype out))
                          (dtype dtype)
                          (t (if (eq (vt-dtype tensor) :float32) :float32 :float64))))
           (nan-val (vt-get-nan final-dtype))
           (zero-val (if (eq final-dtype :float32) 0.0s0 0.0d0))
           (clean
	     (vt-where mask zero-val tensor))
           (count
	     (vt-sum not-nan :axis axis :keepdims keepdims :dtype :int64))
           (mean
	     (vt-nanmean tensor :axis axis :keepdims t :dtype final-dtype))
           (squared-diff
	     (vt-* (vt-map (lambda (c m) (* (- c m) (- c m)))
			   clean mean :dtype final-dtype)
		   not-nan :dtype final-dtype))
           (sum2
	     (vt-sum squared-diff :axis axis :keepdims keepdims
				  :dtype final-dtype :out out))
           (divisor (vt-map (lambda (c)
			      (max 0 (- c ddof)))
			    count :dtype :int64)))
      (unless (member final-dtype '(:float32 :float64))
        (error "vt-nanvar: 结果必须为浮点类型"))
      (vt-map (lambda (s d)
		(if (<= d 0)
		    nan-val (/ s d)))
	      sum2 divisor :dtype final-dtype :out sum2))))

(defun vt-nanstd (tensor &key axis keepdims (ddof 0) dtype out)
  "忽略 nan 计算标准差。支持多轴。"
  (with-float-safe
    (let* ((final-dtype (cond
                          ((and out dtype (not (eq (vt-dtype out) dtype)))
                           (error "vt-nanstd: :out 与 :dtype 冲突"))
                          (out (vt-dtype out))
                          (dtype dtype)
                          (t (if (eq (vt-dtype tensor) :float32) :float32 :float64))))
           (var
	     (vt-nanvar tensor :axis axis :keepdims keepdims :ddof ddof
			       :dtype final-dtype :out out)))
      (unless (member final-dtype '(:float32 :float64))
        (error "vt-nanstd: 结果必须为浮点类型"))
      (vt-sqrt var :dtype final-dtype :out var))))

(defun vt-nanmax (tensor &key axis keepdims dtype out)
  "忽略 nan 的最大值。对标 NumPy: 若沿轴全为 nan 则返回 nan。"
  (if (member (vt-dtype tensor) '(:int32 :int64))
      (vt-amax tensor :axis axis :keepdims keepdims :dtype dtype :out out)
      (with-float-safe
        (let* ((infer-dtype (cond
                              ((and out dtype (not (eq (vt-dtype out) dtype)))
                               (error "vt-nanmax: :out 与 :dtype 冲突"))
                              (out (vt-dtype out))
                              (dtype dtype)
                              (t (vt-dtype tensor))))
               (neg-inf (vt-get-neg-inf infer-dtype))
               (nan-val (vt-get-nan infer-dtype))
               (mask (vt-isnan tensor))
               ;; 传入 dtype 保证 clean 张量类型与后续 out 匹配
               (clean
		 (vt-where mask neg-inf tensor :dtype infer-dtype))
               (result
		 (vt-amax clean :axis axis :keepdims keepdims
				:dtype infer-dtype :out out)))
          (vt-where (vt-all mask :axis axis :keepdims keepdims)
		    nan-val result :dtype infer-dtype :out result)))))

(defun vt-nanmin (tensor &key axis keepdims dtype out)
  "忽略 nan 的最小值。对标 NumPy: 若沿轴全为 nan 则返回 nan。"
  (if (member (vt-dtype tensor) '(:int32 :int64))
      (vt-amin tensor :axis axis :keepdims keepdims :dtype dtype :out out)
      (with-float-safe
        (let* ((infer-dtype (cond
                              ((and out dtype (not (eq (vt-dtype out) dtype)))
                               (error "vt-nanmin: :out 与 :dtype 冲突"))
                              (out (vt-dtype out))
                              (dtype dtype)
                              (t (vt-dtype tensor))))
               (pos-inf (vt-get-pos-inf infer-dtype))
               (nan-val (vt-get-nan infer-dtype))
               (mask (vt-isnan tensor))
               (clean
		 (vt-where mask pos-inf tensor :dtype infer-dtype))
               (result
		 (vt-amin clean :axis axis :keepdims keepdims
				:dtype infer-dtype :out out)))
          (vt-where (vt-all mask :axis axis :keepdims keepdims)
		    nan-val result :dtype infer-dtype :out result)))))

(defun vt-nanargmax (tensor &key axis out)
  "返回最大值索引，忽略 NaN。对标 NumPy 的 np.nanargmax。"
  (with-float-safe
    (if (member (vt-dtype tensor) '(:int32 :int64))
        (vt-argmax tensor :axis axis :out out)
        (let* ((mask (vt-isnan tensor))
               (neg-inf (vt-get-neg-inf (vt-dtype tensor)))
               (clean (vt-where mask neg-inf tensor)))
          (vt-argmax clean :axis axis :out out)))))

(defun vt-nanargmin (tensor &key axis out)
  "返回最小值索引，忽略 NaN。对标 NumPy 的 np.nanargmin。"
  (with-float-safe
    (if (member (vt-dtype tensor) '(:int32 :int64))
        (vt-argmin tensor :axis axis :out out)
        (let* ((mask (vt-isnan tensor))
               (pos-inf (vt-get-pos-inf (vt-dtype tensor)))
               (clean (vt-where mask pos-inf tensor)))
          (vt-argmin clean :axis axis :out out)))))

(defun vt-nanprod (tensor &key axis keepdims dtype out)
  "忽略 NaN 的乘积。对标 NumPy 的 np.nanprod。"
  (with-float-safe
    (if (member (vt-dtype tensor) '(:int32 :int64))
        (vt-prod tensor :axis axis :keepdims keepdims :dtype dtype :out out)
        (let* ((mask (vt-isnan tensor))
               (one-val (if (eq (vt-dtype tensor) :float32) 1.0s0 1.0d0))
               (clean (vt-where mask one-val tensor)))
          (vt-prod clean :axis axis :keepdims keepdims :dtype dtype :out out)))))

(defun vt-nanmedian (tensor &key axis keepdims out)
  "忽略 NaN 计算中位数。对标 NumPy 的 np.nanmedian。"
  (with-float-safe
    (if (member (vt-dtype tensor) '(:int32 :int64))
        (vt-median tensor :axis axis :keepdims keepdims :out out)
        (let* ((nan-val (vt-get-nan :float64))
               (in-data (vt-data tensor))
               (in-strides (vt-strides tensor))
               (in-offset (vt-offset tensor))
               (in-shape (vt-shape tensor))
               (rank (length in-shape)))
          (cond
            ;; Global median
            ((null axis)
             (let ((vals '()))
               (vt-do-each (ptr val tensor)
                 (unless (vt-float-nan-p val)
                   (push val vals)))
               (setf vals (sort vals #'<))
               (let ((result
                       (cond
                         ((null vals) nan-val)
                         ((oddp (length vals)) (coerce (nth (floor (length vals) 2) vals) 'double-float))
                         (t (coerce (/ (+ (nth (1- (/ (length vals) 2)) vals)
                                          (nth (/ (length vals) 2) vals))
                                       2.0d0) 'double-float)))))
                 (if out
                     (progn (vt-fill out result) out)
                     (make-vt nil result :dtype :float64)))))
            ;; Axis reduction: iterate over output, collect along axis
            (t
             (let* ((ax (vt-normalize-axis axis rank))
                    (ax-size (nth ax in-shape))
                    (ax-stride (nth ax in-strides))
                    ;; Build output shape
                    (out-shape (if keepdims
                                   (loop for d in in-shape
                                         for i from 0
                                         collect (if (= i ax) 1 d))
                                   (loop for d in in-shape
                                         for i from 0
                                         unless (= i ax) collect d)))
                    (res (vt-zeros out-shape :dtype :float64))
                    (res-data (vt-data res))
                    (res-offset (vt-offset res))
                    ;; Map: for each output dimension, which input dimension it corresponds to
                    ;; and what stride to use
                    (out-rank (length out-shape))
                    (out-dims (coerce out-shape 'simple-vector))
                    (out-strs (coerce (vt-strides res) 'simple-vector))
                    ;; For each output dim, the corresponding input dim index
                    (in-dim-map (let ((map (make-array out-rank :element-type 'fixnum))
                                      (k 0))
                                  (loop for i from 0 below rank do
                                    (unless (= i ax)
                                      (setf (aref map k) i)
                                      (incf k)))
                                  map))
                    (in-dim-strs (coerce (loop for i below out-rank
                                               collect (nth (aref in-dim-map i) in-strides))
                                         'simple-vector)))
               (declare (type simple-vector out-dims out-strs in-dim-strs)
                        (type (simple-array fixnum (*)) in-dim-map))
               ;; Recurse over output dimensions
               (labels ((compute-med (depth in-ptr out-ptr)
                          (declare (type fixnum depth in-ptr out-ptr))
                          (if (= depth out-rank)
                              ;; Leaf: collect non-NaN values along axis
                              (let ((vals '()))
                                (loop for i fixnum from 0 below ax-size
                                      for ptr fixnum = in-ptr then (+ ptr ax-stride)
                                      for v = (aref in-data ptr)
                                      unless (vt-float-nan-p v)
                                        do (push (coerce v 'double-float) vals))
                                (setf vals (nreverse vals))
                                (setf (aref res-data out-ptr)
                                      (cond
                                        ((null vals) nan-val)
                                        ((oddp (length vals))
                                         (nth (floor (length vals) 2) vals))
                                        (t (/ (+ (nth (1- (/ (length vals) 2)) vals)
                                                 (nth (/ (length vals) 2) vals))
                                              2.0d0)))))
                              ;; Loop over this output dimension
                              (let ((dim (the fixnum (svref out-dims depth)))
                                    (out-str (the fixnum (svref out-strs depth)))
                                    (in-str (the fixnum (svref in-dim-strs depth))))
                                (declare (type fixnum dim out-str in-str))
                                (loop for i fixnum from 0 below dim do
                                  (compute-med (1+ depth) in-ptr out-ptr)
                                  (incf in-ptr in-str)
                                  (incf out-ptr out-str))))))
                 (compute-med 0 in-offset res-offset))
               (if out
                   (progn (vt-copy-into out res) out)
                   res))))))))
