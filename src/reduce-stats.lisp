;;;; reduce-stats.lisp — 归约、统计、排序、nan 感知统计、信号类函数

(in-package :clvt)

(defun vt-unravel-index (offset shape strides)
  "将物理偏移还原为逻辑坐标列表。"
  (loop with rem = offset
	for dim in shape
	for stride in strides
        collect (multiple-value-bind (idx r)
		    (floor rem stride)
		  (setf rem r) idx)))

;;; ============================================================
;;; 编译期算子描述
;;; ============================================================

(eval-when (:compile-toplevel :load-toplevel :execute)

  (defparameter +kernel-lts+
    '(double-float single-float (signed-byte 64) (signed-byte 32))
    "内核支持的 Lisp 元素类型。")

  (defun %op-arg-p (op)
    "是否为 arg 类归约（输出索引）。"
    (member op '(:argmax :argmin :nanargmax :nanargmin)))

  (defun %op-base (op)
    (ecase op
      ((:max :nanmax :argmax :nanargmax) :max)
      ((:min :nanmin :argmin :nanargmin) :min)
      ((:sum :nansum) :sum)
      ((:prod :nanprod) :prod)
      ((:all) :all)
      ((:any) :any)))

  (defun %op-nan-skip-p (op)
    (member op '(:nanmax :nanmin :nansum :nanprod :nanargmax :nanargmin)))

  (defun %op-nan-prop-p (op)
    (member op '(:max :min :argmax :argmin)))

  (defun %op-requires-float-p (op)
    (member op '(:nanmax :nanmin :nanargmax :nanargmin)))

  (defun %dtype->lt (dtype)
    (case dtype
      (:float64 'double-float)
      (:float32 'single-float)
      (:int64   '(signed-byte 64))
      (:int32   '(signed-byte 32))
      (t nil)))

  (defun %et->lt (in-et)
    (cond ((equal in-et 'double-float)     'double-float)
          ((equal in-et 'single-float)     'single-float)
          ((equal in-et '(signed-byte 64)) '(signed-byte 64))
          ((equal in-et '(signed-byte 32)) '(signed-byte 32))
          (t nil)))

  (defun %lt-rank (lt)
    (cond ((eq lt 'double-float) 5)
          ((eq lt 'single-float) 4)
          ((equal lt '(signed-byte 64)) 3)
          ((equal lt '(signed-byte 32)) 2)
          (t -1)))

  (defun %op-acc-lt (op lt res-lt)
    "累加器 Lisp 类型：sum/prod 提升到 lt/res-lt 中较宽者；
     max/min/arg 用 lt；all/any 用 int64。"
    (cond
      ((member (%op-base op) '(:all :any)) '(signed-byte 64))
      ((%op-arg-p op) lt)
      (t (if (>= (%lt-rank lt) (%lt-rank res-lt)) lt res-lt))))

  (defun %cast-form (lt form)
    (cond ((eq lt 'double-float) `(coerce ,form 'double-float))
          ((eq lt 'single-float) `(coerce ,form 'single-float))
          ((member lt '((signed-byte 64) (signed-byte 32)) :test #'equal)
           `(truncate ,form))
          (t form)))

  (defun %op-init (op lt res-lt)
    "归约初始值（编译期常量，类型与 acc-lt 匹配）。"
    (let* ((acc-lt (%op-acc-lt op lt res-lt))
           (skip   (%op-nan-skip-p op)))
      (ecase (%op-base op)
        (:sum  (cond ((eq acc-lt 'double-float) 0.0d0)
                     ((eq acc-lt 'single-float) 0.0s0)
                     (t 0)))
        (:prod (cond ((eq acc-lt 'double-float) 1.0d0)
                     ((eq acc-lt 'single-float) 1.0s0)
                     (t 1)))
        (:all 1)
        (:any 0)
        (:max  (cond ((and skip (eq acc-lt 'double-float)) '(vt-get-nan :float64))
                     ((and skip (eq acc-lt 'single-float)) '(vt-get-nan :float32))
                     ((eq acc-lt 'double-float) '+vt-dfloat-neg-inf+)
                     ((eq acc-lt 'single-float) '+vt-sfloat-neg-inf+)
                     ((equal acc-lt '(signed-byte 64)) -9223372036854775808)
                     ((equal acc-lt '(signed-byte 32)) -2147483648)
                     (t 0)))
        (:min  (cond ((and skip (eq acc-lt 'double-float)) '(vt-get-nan :float64))
                     ((and skip (eq acc-lt 'single-float)) '(vt-get-nan :float32))
                     ((eq acc-lt 'double-float) '+vt-dfloat-pos-inf+)
                     ((eq acc-lt 'single-float) '+vt-sfloat-pos-inf+)
                     ((equal acc-lt '(signed-byte 64)) 9223372036854775807)
                     ((equal acc-lt '(signed-byte 32)) 2147483647)
                     (t 0))))))

  (defun %op-out-dtype (op in-dtype)
    (case op
      ((:all :any) :int64)
      ((:argmax :argmin :nanargmax :nanargmin) :int32)
      (t in-dtype)))

  ) ; end eval-when

;;; ============================================================
;;; 归约步宏
;;; ============================================================

(defmacro %op-step (op acc val lt acc-lt)
  "生成一次 acc ← f(acc, val) 更新。val 类型 lt，acc 类型 acc-lt。"
  (let* ((base (%op-base op))
         (skip (%op-nan-skip-p op))
         (prop (%op-nan-prop-p op))
         (float (subtypep lt 'float))
         (casted (if (equal lt acc-lt) val (%cast-form acc-lt val))))
    (ecase base
      (:sum  (if (and skip float)
                 `(when (not (%nan-p ,val)) (incf ,acc ,casted))
                 `(incf ,acc ,casted)))
      (:prod (if (and skip float)
                 `(when (not (%nan-p ,val)) (setf ,acc (* ,acc ,casted)))
                 `(setf ,acc (* ,acc ,casted))))
      (:all  `(when (zerop ,val) (setf ,acc 0)))
      (:any  `(when (/= ,val 0) (setf ,acc 1)))
      (:max  (cond ((and skip float)
                    `(cond ((%nan-p ,val) nil)
                           ((%nan-p ,acc) (setf ,acc ,casted))
                           ((> ,val ,acc) (setf ,acc ,casted))))
                   ((and prop float)
                    `(cond ((%nan-p ,acc) nil)
                           ((%nan-p ,val) (setf ,acc ,casted))
                           ((> ,val ,acc) (setf ,acc ,casted))))
                   (t `(when (> ,val ,acc) (setf ,acc ,casted)))))
      (:min  (cond ((and skip float)
                    `(cond ((%nan-p ,val) nil)
                           ((%nan-p ,acc) (setf ,acc ,casted))
                           ((< ,val ,acc) (setf ,acc ,casted))))
                   ((and prop float)
                    `(cond ((%nan-p ,acc) nil)
                           ((%nan-p ,val) (setf ,acc ,casted))
                           ((< ,val ,acc) (setf ,acc ,casted))))
                   (t `(when (< ,val ,acc) (setf ,acc ,casted))))))))

(defmacro %op-arg-step (op acc best-i r val lt acc-lt)
  "生成一次 arg 更新：acc ← val，best-i ← r。acc-lt 对 arg 恒等于 lt。"
  (declare (ignore acc-lt))
  (let* ((base (%op-base op))
         (skip (%op-nan-skip-p op))
         (prop (%op-nan-prop-p op))
         (float (subtypep lt 'float)))
    (ecase base
      (:max (cond ((and skip float)
                   `(cond ((%nan-p ,val) nil)
                          ((%nan-p ,acc) (setf ,acc ,val ,best-i ,r))
                          ((> ,val ,acc) (setf ,acc ,val ,best-i ,r))))
                  ((and prop float)
                   `(cond ((%nan-p ,val)
                           (when (not (%nan-p ,acc))
                             (setf ,acc ,val ,best-i ,r)))
                          ((%nan-p ,acc) nil)
                          ((> ,val ,acc) (setf ,acc ,val ,best-i ,r))))
                  (t `(when (> ,val ,acc) (setf ,acc ,val ,best-i ,r)))))
      (:min (cond ((and skip float)
                   `(cond ((%nan-p ,val) nil)
                          ((%nan-p ,acc) (setf ,acc ,val ,best-i ,r))
                          ((< ,val ,acc) (setf ,acc ,val ,best-i ,r))))
                  ((and prop float)
                   `(cond ((%nan-p ,val)
                           (when (not (%nan-p ,acc))
                             (setf ,acc ,val ,best-i ,r)))
                          ((%nan-p ,acc) nil)
                          ((< ,val ,acc) (setf ,acc ,val ,best-i ,r))))
                  (t `(when (< ,val ,acc) (setf ,acc ,val ,best-i ,r))))))))

;;; ============================================================
;;; 三条内核路径宏
;;; ============================================================

(defmacro %kernel-global (op lt res-lt)
  "路径 1：连续 + 全局。"
  (let* ((acc-lt (%op-acc-lt op lt res-lt))
         (init   (%op-init   op lt res-lt))
         (arg-p  (%op-arg-p  op))
         (nan-skip-arg (and arg-p (%op-nan-skip-p op) (subtypep lt 'float))))
    `(let ((d (the (simple-array ,lt (*)) in-data)))
       (let ((acc (the ,acc-lt ,init))
             ,@(when arg-p `((best-i 0)))
             (p   (the fixnum in-off))
             (end (the fixnum (+ in-off in-size))))
         (declare (type fixnum p end)
                  ,@(when arg-p '((type fixnum best-i))))
         (loop while (< p end) do
           (let ((v (the ,lt (aref d p))))
             ,(if arg-p
                  `(%op-arg-step ,op acc best-i (- p in-off) v ,lt ,acc-lt)
                  `(%op-step ,op acc v ,lt ,acc-lt)))
           (incf p))
         ,@(when nan-skip-arg
             `((when (%nan-p acc)
                 (error "~a: All-NaN slice encountered" ',op))))
         (setf (aref (the (simple-array ,res-lt (*)) res-data) res-off)
               ,(%cast-form res-lt (if arg-p 'best-i 'acc)))))))

(defmacro %kernel-single-axis (op lt res-lt)
  "路径 2：连续 + 单轴。"
  (let* ((acc-lt (%op-acc-lt op lt res-lt))
         (init   (%op-init   op lt res-lt))
         (arg-p  (%op-arg-p  op))
         (nan-skip-arg (and arg-p (%op-nan-skip-p op) (subtypep lt 'float))))
    `(let ((d  (the (simple-array ,lt (*)) in-data))
           (od (the (simple-array ,res-lt (*)) res-data))
           (init-v (the ,acc-lt ,init)))
       (declare (type fixnum outer red inner))
       (dotimes (o outer)
         (let ((in-base (the fixnum (+ in-off (* o red inner)))))
           (declare (type fixnum in-base))
           (dotimes (i inner)
             (let ((acc init-v)
                   ,@(when arg-p `((best-i 0))))
               (declare ,@(when arg-p '((type fixnum best-i))))
               (dotimes (r red)
                 (let ((v (the ,lt (aref d (the fixnum (+ in-base (* r inner) i))))))
                   ,(if arg-p
                        `(%op-arg-step ,op acc best-i r v ,lt ,acc-lt)
                        `(%op-step ,op acc v ,lt ,acc-lt))))
               ,@(when nan-skip-arg
                   `((when (%nan-p acc)
                       (error "~a: All-NaN slice encountered" ',op))))
               (setf (aref od (the fixnum (+ res-off (* o inner) i)))
                     ,(%cast-form res-lt (if arg-p 'best-i 'acc))))))))))

(defmacro %kernel-general (op lt res-lt)
  "路径 3：通用回退。"
  (let* ((acc-lt (%op-acc-lt op lt res-lt))
         (init   (%op-init   op lt res-lt))
         (arg-p  (%op-arg-p  op))
         (nan-skip-arg (and arg-p (%op-nan-skip-p op) (subtypep lt 'float))))
    `(let ((d  (the (simple-array ,lt (*)) in-data))
           (od (the (simple-array ,res-lt (*)) res-data)))
       (declare (type fixnum n-red n-non red-size non-size))
       (let ((non-idx (make-array (max n-non 1) :element-type 'fixnum :initial-element 0))
             (red-idx (make-array (max n-red 1) :element-type 'fixnum :initial-element 0))
             (init-v  (the ,acc-lt ,init)))
         (declare (type (simple-array fixnum (*)) non-idx red-idx))
         (dotimes (out-pos non-size)
           (let ((base in-off))
             (declare (type fixnum base))
             (dotimes (dd n-non)
               (incf base (* (aref non-idx dd) (svref non-strides dd))))
             (fill red-idx 0)
             (let ((acc init-v)
                   ,@(when arg-p `((best-i 0))))
               (declare (type ,acc-lt acc)
                        ,@(when arg-p '((type fixnum best-i))))
               (dotimes (r red-size)
                 (let ((offset 0) (linear 0) (mult 1))
                   (declare (type fixnum offset linear mult))
                   (loop for dd fixnum from (1- n-red) downto 0 do
                     (incf offset (* (aref red-idx dd) (svref red-strides-v dd)))
                     (incf linear (* (aref red-idx dd) mult))
                     (setf mult (* mult (svref red-sizes dd))))
                   (let ((v (the ,lt (aref d (the fixnum (+ base offset))))))
                     ,(if arg-p
                          `(%op-arg-step ,op acc best-i linear v ,lt ,acc-lt)
                          `(%op-step ,op acc v ,lt ,acc-lt))))
                 (when (plusp n-red)
                   (let ((dd (1- n-red)))
                     (declare (type fixnum dd))
                     (loop
                       (incf (aref red-idx dd))
                       (when (< (aref red-idx dd) (svref red-sizes dd)) (return))
                       (setf (aref red-idx dd) 0)
                       (decf dd)
                       (when (< dd 0) (return))))))
               ,@(when nan-skip-arg
                   `((when (%nan-p acc)
                       (error "~a: All-NaN slice encountered" ',op))))
               (setf (aref od (the fixnum (+ res-off out-pos)))
                     ,(%cast-form res-lt (if arg-p 'best-i 'acc))))
             (when (plusp n-non)
               (let ((dd (1- n-non)))
                 (declare (type fixnum dd))
                 (loop
                   (incf (aref non-idx dd))
                   (when (< (aref non-idx dd) (svref non-sizes dd)) (return))
                   (setf (aref non-idx dd) 0)
                   (decf dd)
                   (when (< dd 0) (return)))))))))))

;;; ============================================================
;;; 统一生成宏
;;; ============================================================

(defmacro %def-vt-reduce (name op)
  "为算子 op 生成函数 vt-<name>。对 in-et × res-lt 两级分派到类型特化内核。"
  (let ((fn-name (intern (format nil "VT-~a" name)))
        (arg-p   (%op-arg-p op))
        (requires-float (%op-requires-float-p op)))
    (flet ((dispatch (kernel-macro)
             `(cond
                ,@(loop for lt in +kernel-lts+
                        unless (and requires-float (subtypep lt 'integer))
                        collect
                        `((equal in-et ',lt)
                          (cond
                            ,@(loop for rlt in +kernel-lts+
                                    collect
                                    `((equal res-lt ',rlt)
                                      (,kernel-macro ,op ,lt ,rlt)))
                            (t (error "unsupported output dtype ~a" res-lt)))))
                (t (error "unsupported input dtype ~a" in-et)))))
      `(defun ,fn-name (tensor &key axis keepdims dtype out)
         (declare (type vt tensor)
                  (type (or null fixnum list) axis)
                  (type (or null vt) out))
         (with-float-safe
           (let* ((in-shape (vt-shape tensor))
                  (rank (length in-shape))
                  (axes (vt-normalize-axes axis rank))
                  (global (null axes))
                  (single-axis-p (and (= (length axes) 1) (not global)))
                  (out-shape
                    (cond ((and global (not keepdims)) nil)
                          (global (make-list rank :initial-element 1))
                          ((not keepdims)
                           (loop for d in in-shape for i fixnum from 0
                                 unless (member i axes) collect d))
                          (t (loop for d in in-shape for i fixnum from 0
                                   collect (if (member i axes) 1 d))))))
             ;; ---- :out 形状校验 ----
             (when (and out (not (equal (vt-shape out) out-shape)))
               (error "vt-~a: :out shape ~a does not match expected ~a"
                      ',name (vt-shape out) out-shape))
             ;; ---- :out 与 :dtype 冲突校验 ----
             (when (and out dtype (not (eq (vt-dtype out) dtype)))
               (error "vt-~a: :out dtype ~a conflicts with :dtype ~a"
                      ',name (vt-dtype out) dtype))
             (let* ((axis-size (if axes
                                   (reduce #'* (mapcar (lambda (a) (nth a in-shape)) axes)
                                           :initial-value 1)
                                   (reduce #'* in-shape :initial-value 1)))
                    (in-data (vt-data tensor))
                    (in-off  (vt-offset tensor))
                    (in-et   (array-element-type in-data))
                    (in-size (vt-size tensor))
                    (final-out-dtype
                      (or dtype
                          (and out (vt-dtype out))
                          (%op-out-dtype ,op (vt-dtype tensor))))
                    (res-lt (%dtype->lt final-out-dtype))
                    (res (or out (make-vt out-shape 0 :dtype final-out-dtype)))
                    (res-data (vt-data res))
                    (res-off  (vt-offset res)))
               (declare (fixnum rank axis-size in-size))
               (unless res-lt
                 (error "vt-~a: unsupported output dtype ~a" ',name final-out-dtype))
               ,@(when requires-float
                   `((when (subtypep in-et 'integer)
                       (error "~a: NaN-aware max/min/arg op on integer input" ',name))))
               ;; ---- 空输入 ----
               (when (or (zerop axis-size) (zerop in-size))
                 ,@(when (and arg-p (%op-nan-skip-p op))
                     `((error "~a: empty slice or All-NaN encountered" ',name)))
                 ,(if arg-p
                      `(progn (vt-fill res 0)
                              (return-from ,fn-name res))
                      `(progn
                         (let ((lt (%et->lt in-et)))
                           (vt-fill res
                                    (cond
                                      ,@(loop for rlt in +kernel-lts+
                                              collect
                                              `((equal res-lt ',rlt)
                                                (cond
                                                  ,@(loop for l in +kernel-lts+
                                                          collect
                                                          `((equal lt ',l)
                                                            ,(%op-init op l rlt)))
                                                  (t 0))))
                                      (t 0))))
                         (return-from ,fn-name res))))
               ;; ---- 主分派 ----
               (cond
                 ;; 路径 1：连续 + 全局
                 ((and global (vt-contiguous-p tensor))
                  ,(dispatch '%kernel-global))
                 ;; 路径 2：连续 + 单轴
                 ((and single-axis-p
                       (vt-contiguous-p tensor)
                       (vt-contiguous-p res))
                  (let* ((ax (first axes))
                         (outer (reduce #'* in-shape :end ax :initial-value 1))
                         (red   (nth ax in-shape))
                         (inner (reduce #'* in-shape :start (1+ ax) :initial-value 1)))
                    (declare (fixnum ax outer red inner))
                    ,(dispatch '%kernel-single-axis)))
                 ;; 路径 3：通用回退
                 (t
                  (when (and out (not (vt-contiguous-p res)))
                    (error "vt-~a: :out must be contiguous in fallback path" ',name))
                  (let* ((global-red  (null axes))
                         (eff-red-axes (if global-red
                                           (loop for i below rank collect i)
                                           axes))
                         (eff-non-axes (if global-red
                                           nil
                                           (loop for i below rank
                                                 unless (member i axes) collect i)))
                         (in-strides (vt-strides tensor))
                         (in-shape-vec   (coerce in-shape 'simple-vector))
                         (in-strides-vec (coerce in-strides 'simple-vector))
                         (red-axes (coerce eff-red-axes 'simple-vector))
                         (non-axes (coerce eff-non-axes 'simple-vector))
                         (n-red (length eff-red-axes))
                         (n-non (- rank n-red))
                         (red-sizes (map 'vector (lambda (a) (svref in-shape-vec a))
                                         red-axes))
                         (red-strides-v (map 'vector (lambda (a) (svref in-strides-vec a))
                                             red-axes))
                         (non-sizes (map 'vector (lambda (a) (svref in-shape-vec a))
                                         non-axes))
                         (non-strides (map 'vector (lambda (a) (svref in-strides-vec a))
                                           non-axes))
                         (red-size (reduce #'* red-sizes :initial-value 1))
                         (non-size (reduce #'* non-sizes :initial-value 1)))
                    (declare (fixnum n-red n-non red-size non-size))
                    ,(dispatch '%kernel-general))))
               res)))))))

;;; ============================================================
;;; 归约族定义
;;; ============================================================

(%def-vt-reduce sum  :sum)
(%def-vt-reduce prod :prod)
(%def-vt-reduce amax :max)
(%def-vt-reduce amin :min)
(%def-vt-reduce all  :all)
(%def-vt-reduce any  :any)

(%def-vt-reduce nansum  :nansum)
(%def-vt-reduce nanprod :nanprod)
(%def-vt-reduce nanmax  :nanmax)
(%def-vt-reduce nanmin  :nanmin)

(%def-vt-reduce argmax    :argmax)
(%def-vt-reduce argmin    :argmin)
(%def-vt-reduce nanargmax :nanargmax)
(%def-vt-reduce nanargmin :nanargmin)


(defun vt-isclose (t1 t2 &key (rtol 1e-5) (atol 1e-8) out)
  (vt-map (lambda (a b)
            (cond ((or (%nan-p a) (%nan-p b)) 0.0d0)
                  ((or (%inf-p a) (%inf-p b)) (if (= a b) 1.0d0 0.0d0))
                  (t (if (<= (abs (- a b))
			     (+ atol (* rtol (max (abs a) (abs b)))))
			 1.0d0 0.0d0))))
          t1 t2 :dtype :float64 :out out))

(defun vt-allclose (t1 t2 &key (rtol 1e-5) (atol 1e-8))
  (= (vt-item (vt-all (vt-isclose t1 t2 :rtol rtol :atol atol))) 1.0d0))

(defun vt-isfinite (vt &key out)
  (vt-map (lambda (x) (if (and (not (%nan-p x)) (not (%inf-p x))) 1.0d0 0.0d0))
          vt :dtype :float64 :out out))

(defun vt-isinf (vt &key out)
  (vt-map (lambda (x) (if (%inf-p x) 1.0d0 0.0d0)) vt :dtype :float64 :out out))

(defun vt-isnan (vt &key out)
  (vt-map (lambda (x) (if (%nan-p x) 1.0d0 0.0d0)) vt :dtype :float64 :out out))

;;; ------------------------------------------------------------------
;;; 均值 / 方差 / 标准差
;;; ------------------------------------------------------------------

(defun %get-axes-count (axis rank shape)
  (let* ((axes (vt-normalize-axes axis rank))
         (count (if axes (reduce #'* (mapcar (lambda (a)
					       (nth a shape))
					     axes)
				 :initial-value 1)
                    (reduce #'* shape :initial-value 1))))
    (values axes count)))

(defun vt-average (tensor weights &key axis keepdims dtype out)
  (with-float-safe
    (let ((a-shape (vt-shape tensor))
	  (w-shape (vt-shape weights))
          (eff weights))
      (cond (axis (let* ((rank (length a-shape))
			 (ax (vt-normalize-axis axis rank))
			 (ax-size (nth ax a-shape)))
                    (unless (and (= (length w-shape) 1)
				 (= (first w-shape) ax-size))
                      (error "1D weights expected when axis is specified"))
                    (setf eff (vt-reshape weights (loop for i below rank
							collect (if (= i ax) ax-size 1))))))
            (t (unless (equal w-shape a-shape)
		 (error "weights must have same shape as a when axis is nil"))))
      (let* ((prod (vt-map #'* tensor eff))
             (weighted-sum (vt-sum prod :axis axis :keepdims keepdims))
             (sum-weights (vt-item (vt-sum weights)))
             (in-dtype (vt-dtype weighted-sum))
             (need-promote (member in-dtype '(:int32 :int64 :int16 :int8 :uint8 :uint16)))
             (exec-dtype (cond ((and out dtype (not (eq (vt-dtype out) dtype)))
				(error "vt-average: :out 与 :dtype 冲突"))
                               (out (vt-dtype out)) (dtype dtype)
                               (need-promote :float64) (t in-dtype)))
             (nan-val (vt-get-nan exec-dtype))
             (scalar-divisor (coerce sum-weights (if (eq exec-dtype :float32)
						     'single-float 'double-float)))
             (map-dtype (cond (out nil) (dtype dtype) (need-promote :float64) (t nil))))
	(cond ((%nan-p sum-weights)
               (if out (progn (vt-map (lambda (x)
					(declare (ignore x))
					nan-val)
				      weighted-sum :out out :dtype dtype)
			      out)
                   (vt-full (vt-shape weighted-sum) nan-val :dtype exec-dtype)))
              ((zerop sum-weights) (error "Weights sum to zero"))
              (t (vt-map (lambda (s)
			   (/ s scalar-divisor))
			 weighted-sum :dtype map-dtype :out out)))))))

(defun vt-mean (tensor &key axis keepdims dtype out)
  (let* ((shape (vt-shape tensor))
	 (rank (length shape)))
    (multiple-value-bind (axes count) (%get-axes-count axis rank shape)
      (let* ((final-dtype (cond ((and out dtype (not (eq (vt-dtype out) dtype)))
                                 (error "vt-mean: :out 与 :dtype 冲突"))
                                (out (vt-dtype out)) (dtype dtype)
                                (t (if (eq (vt-dtype tensor) :float32)
				       :float32 :float64)))))
        (when (= count 0)
          (let ((nan (vt-get-nan final-dtype))
                (out-shape (if keepdims
                               (loop for d in shape
				     for i below rank
                                     collect (if (or (null axes) (member i axes)) 1 d))
                               (loop for d in shape
				     for i below rank
                                     unless (or (null axes) (member i axes))
				       collect d))))
            (return-from vt-mean
              (if out (progn (vt-map (lambda (x)
				       (declare (ignore x)) nan)
				     out :dtype final-dtype :out out)
			     out)
                  (vt-full out-shape nan :dtype final-dtype)))))
        (let* ((sum-result (vt-sum tensor :axis axes :keepdims keepdims
					  :dtype final-dtype :out out))
               (div (coerce count (if (eq final-dtype :float32)
				      'single-float 'double-float))))
          (vt-map (lambda (s) (/ s div))
		  sum-result :dtype final-dtype :out sum-result))))))

(defun vt-var (tensor &key axis keepdims (ddof 0) dtype out)
  (let* ((shape (vt-shape tensor))
	 (rank (length shape)))
    (multiple-value-bind (axes n) (%get-axes-count axis rank shape)
      (let* ((divisor (- n ddof))
             (final-dtype (cond ((and out dtype (not (eq (vt-dtype out) dtype)))
                                 (error "vt-var: :out 与 :dtype 冲突"))
                                (out (vt-dtype out)) (dtype dtype)
                                (t (if (eq (vt-dtype tensor) :float32)
				       :float32 :float64)))))
        (if (<= divisor 0)
            (vt-map (lambda (s)
		      (declare (ignore s))
		      (vt-get-nan final-dtype))
                    (vt-sum tensor :axis axes :keepdims keepdims)
		    :dtype final-dtype :out out)
            (let* ((mean-val (vt-mean tensor :axis axes :keepdims t :dtype final-dtype))
                   (sq-diff (vt-square (vt-- tensor mean-val :dtype final-dtype)
				       :dtype final-dtype))
                   (sum-sq (vt-sum sq-diff :axis axes :keepdims keepdims
					   :dtype final-dtype :out out)))
              (vt-/ sum-sq divisor :dtype final-dtype :out sum-sq)))))))

(defun vt-std (tensor &key axis keepdims (ddof 0) dtype out)
  (let* ((final-dtype (cond ((and out dtype (not (eq (vt-dtype out) dtype)))
                             (error "vt-std: :out 与 :dtype 冲突"))
                            (out (vt-dtype out)) (dtype dtype)
                            (t (if (eq (vt-dtype tensor) :float32)
				   :float32 :float64)))))
    (let ((variance (vt-var tensor :axis axis :keepdims keepdims :ddof ddof
				   :dtype final-dtype :out out)))
      (vt-sqrt variance :dtype final-dtype :out variance))))

;;; ------------------------------------------------------------------
;;; 累积
;;; ------------------------------------------------------------------

(defun vt-cumulative (tensor op init-val &key axis dtype out)
  (let* ((shape (vt-shape tensor))
	 (rank (length shape))
         (final-dtype (cond ((and out dtype (not (eq (vt-dtype out) dtype)))
                             (error "类型冲突: :out (~a) vs :dtype (~a)" (vt-dtype out) dtype))
                            (dtype dtype) (out (vt-dtype out)) (t (vt-dtype tensor))))
         (lisp-type (vt-dtype->lisp-type final-dtype))
         (result (or out (vt-zeros shape :dtype final-dtype)))
         (in-data (vt-data tensor)) (out-data (vt-data result))
         (in-strides (vt-strides tensor)) (in-offset (vt-offset tensor))
         (out-strides (vt-strides result)) (out-offset (vt-offset result)))
    (if axis
        (let* ((ax (vt-normalize-axis axis rank))
	       (ax-dim (nth ax shape))
               (indices (make-array rank :element-type '(signed-byte 64)
					 :initial-element 0)))
          (labels ((advance ()
                     (loop for d from (1- rank) downto 0
                           when (/= d ax)
                             do (incf (aref indices d))
                                (if (< (aref indices d) (nth d shape))
                                    (return-from advance t)
                                    (setf (aref indices d) 0)))))
            (loop
              (let ((in-ptr in-offset) (out-ptr out-offset))
                (loop for d from 0 below rank do
                  (incf in-ptr (* (aref indices d)
				  (nth d in-strides)))
                  (incf out-ptr (* (aref indices d)
				   (nth d out-strides))))
                (let ((in-stride (nth ax in-strides))
		      (out-stride (nth ax out-strides))
                      (cum (coerce init-val lisp-type)))
                  (loop for i from 0 below ax-dim do
                    (setf cum (funcall op cum (aref in-data (+ in-ptr (* i in-stride)))))
                    (setf (aref out-data (+ out-ptr (* i out-stride)))
			  (vt-cast cum final-dtype)))))
              (unless (advance) (return)))))
        (let* ((flat (vt-ravel tensor))
	       (flat-in (vt-data flat))
               (flat-offset (vt-offset flat))
	       (cum (coerce init-val lisp-type))
               (out-shape-vec (coerce (vt-shape result) 'simple-vector))
               (out-strs-vec (coerce out-strides 'simple-vector))
               (out-rank (length out-shape-vec)))
          (labels ((recurse (depth out-ptr flat-idx)
                     (if (= depth out-rank)
                         (progn (setf cum (funcall op cum (aref flat-in (+ flat-offset flat-idx))))
                                (setf (aref out-data out-ptr)
				      (vt-cast cum final-dtype))
                                (1+ flat-idx))
                         (let ((dim (svref out-shape-vec depth))
			       (stride (svref out-strs-vec depth)))
                           (loop for i from 0 below dim
                                 for cur = out-ptr then (+ cur stride)
                                 do (setf flat-idx (recurse (1+ depth) cur flat-idx)))))))
            (recurse 0 out-offset 0))))
    result))

(defun vt-cumsum (tensor &key axis dtype out)
  (vt-cumulative tensor #'+ 0 :axis axis :dtype dtype :out out))

(defun vt-cumprod (tensor &key axis dtype out)
  (vt-cumulative tensor #'* 1 :axis axis :dtype dtype :out out))

;;; ------------------------------------------------------------------
;;; 中位数 / 百分位 / 直方图
;;; ------------------------------------------------------------------

(defun vt-median (tensor &key axis)
  (with-float-safe
    (if axis
	(let* ((shape (vt-shape tensor))
	       (rank (length shape))
	       (ax (vt-normalize-axis axis rank))
               (out-shape (loop for d in shape
				for i from 0
				unless (= i ax)
				  collect d))
               (result (vt-zeros out-shape :dtype :float64))
	       (out-strides (vt-strides result)))
          (vt-do-each (ptr val result)
            (declare (ignore val))
            (let ((out-idx (vt-unravel-index ptr out-shape out-strides)))
              (let* ((specs (loop for i from 0 below rank
                                  if (= i ax) collect '(:all)
                                    else collect (list (pop out-idx))))
                     (fiber (apply #'vt-slice tensor specs)) (fs (vt-size fiber)))
		(setf (aref (vt-data result) ptr)
                      (if (zerop fs) (vt-get-nan :float64)
                          (let ((vals (loop for i below fs collect (vt-ref fiber i))))
                            (if (some #'%nan-p vals) (vt-get-nan :float64)
				(let ((sv (vt-numpy-sort vals #'<)))
                                  (if (oddp fs) (vt-cast (nth (floor fs 2) sv) :float64)
                                      (/ (+ (vt-cast (nth (1- (floor fs 2)) sv) :float64)
                                            (vt-cast (nth (floor fs 2) sv) :float64))
					 2.0d0))))))))))
          result)
	(let* ((flat (vt-flatten tensor))
	       (size (vt-size flat)))
          (if (zerop size)
	      (make-vt nil (vt-get-nan :float64) :dtype :float64)
              (let ((vals (loop for i below size collect (aref (vt-data flat) i))))
		(if (some #'%nan-p vals)
		    (make-vt nil (vt-get-nan :float64) :dtype :float64)
                    (let ((sv (vt-numpy-sort vals #'<)))
                      (if (oddp size)
			  (make-vt nil (vt-cast (nth (floor size 2) sv) :float64) :dtype :float64)
                          (make-vt nil (/ (+ (vt-cast (nth (1- (floor size 2)) sv) :float64)
                                             (vt-cast (nth (floor size 2) sv) :float64))
					  2.0d0)
                                   :dtype :float64))))))))))

(defun %percent-from-sorted (sorted q interpolation)
  (let* ((n (length sorted))
	 (idx (* q (1- n)))
         (lower (floor idx))
	 (upper (min (ceiling idx) (1- n)))
	 (frac (- idx lower)))
    (case interpolation
      (:linear (if (= lower upper) (vt-cast (nth lower sorted) :float64)
                   (+ (* (- 1 frac) (vt-cast (nth lower sorted) :float64))
                      (* frac (vt-cast (nth upper sorted) :float64)))))
      (:lower (vt-cast (nth lower sorted) :float64))
      (:higher (vt-cast (nth upper sorted) :float64))
      (:midpoint (/ (+ (vt-cast (nth lower sorted) :float64)
		       (vt-cast (nth upper sorted) :float64))
		    2.0d0))
      (:nearest (vt-cast (nth (if (<= frac 0.5d0) lower upper) sorted) :float64)))))

(defun vt-percentile (tensor percentile &key axis (interpolation :linear))
  (with-float-safe
    (let ((q (/ percentile 100.0d0)))
      (if axis
          (let* ((shape (vt-shape tensor))
		 (rank (length shape))
		 (ax (vt-normalize-axis axis rank))
		 (out-shape (loop for d in shape
				  for i from 0 unless (= i ax) collect d))
		 (result (vt-zeros out-shape :dtype :float64))
		 (out-strides (vt-strides result)))
            (vt-do-each (ptr val result)
              (declare (ignore val))
              (let ((out-idx (vt-unravel-index ptr out-shape out-strides)))
		(let* ((specs (loop for i from 0 below rank
                                    if (= i ax) collect '(:all)
                                      else collect (list (pop out-idx))))
                       (fiber (apply #'vt-slice tensor specs)) (fs (vt-size fiber)))
                  (setf (aref (vt-data result) ptr)
			(if (zerop fs) (vt-get-nan :float64)
                            (let ((raw (loop for i below fs collect (vt-ref fiber i))))
                              (if (some #'%nan-p raw) (vt-get-nan :float64)
                                  (%percent-from-sorted
				   (vt-numpy-sort raw #'<) q interpolation))))))))
            result)
          (let* ((flat (vt-flatten tensor))
		 (size (vt-size flat)))
            (if (zerop size)
		(make-vt nil (vt-get-nan :float64) :dtype :float64)
		(let ((raw (loop for i below size collect (aref (vt-data flat) i))))
                  (if (some #'%nan-p raw)
		      (make-vt nil (vt-get-nan :float64) :dtype :float64)
                      (make-vt nil (%percent-from-sorted
				    (vt-numpy-sort raw #'<) q interpolation)
			       :dtype :float64)))))))))

(defun vt-quantile (tensor q &key axis (interpolation :linear))
  (vt-percentile tensor (* q 100) :axis axis :interpolation interpolation))

(defun vt-ptp (tensor &key axis)
  (if axis (vt-- (vt-amax tensor :axis axis) (vt-amin tensor :axis axis))
      (- (vt-item (vt-amax tensor)) (vt-item (vt-amin tensor)))))

(defun vt-histogram (tensor &key bins range density)
  (with-float-safe
    (let* ((flat (vt-flatten tensor))
	   (data (vt-data flat))
	   (size (vt-size flat))
           (bins (or bins 10))
	   (data-min nil)
	   (data-max nil))
      (if range
	  (setf data-min (first range) data-max (second range))
          (progn (unless (= (vt-item (vt-all (vt-isfinite tensor))) 1.0d0)
                   (error "自动确定 bin 范围需要有限输入"))
                 (setf data-min (vt-item (vt-amin tensor))
		       data-max (vt-item (vt-amax tensor)))))
      (when (= data-min data-max)
	(setf data-min (- data-min 0.5) data-max (+ data-max 0.5)))
      (let* ((bin-width (/ (- data-max data-min) bins))
             (hist (make-array bins :initial-element 0))
             (edges (make-array (1+ bins) :element-type t)))
        (loop for i from 0 to bins do
	  (setf (aref edges i) (+ data-min (* i bin-width))))
        (loop for i from 0 below size
	      for val = (aref data i)
              when (and (>= val data-min) (<= val data-max))
                do (let ((bi (if (= val data-max)
				 (1- bins)
				 (floor (- val data-min) bin-width))))
                     (incf (aref hist bi))))
        (when density
          (let ((total (reduce #'+ hist)))
            (if (zerop total)
                (loop for i from 0 below bins do
		  (setf (aref hist i) 0.0d0))
                (loop for i from 0 below bins do
		  (setf (aref hist i)
			(/ (aref hist i)
			   (* total bin-width)))))))
        (values (vt-from-sequence hist :dtype :float64)
		(vt-from-sequence edges :dtype :float64))))))

;;; ------------------------------------------------------------------
;;; 排序
;;; ------------------------------------------------------------------

(defun vt-sort (tensor &key (axis -1))
  (if axis
      (let* ((shape (vt-shape tensor))
	     (rank (length shape))
	     (ax (vt-normalize-axis axis rank))
             (ax-dim (nth ax shape))
	     (in-strides (vt-strides tensor))
             (in-offset (vt-offset tensor))
	     (in-data (vt-data tensor))
             (result (vt-copy tensor))
	     (out-strides (vt-strides result))
	     (out-data (vt-data result)))
        (labels ((recurse (depth in-ptr out-ptr)
                   (cond ((= depth ax)
                          (let* ((in-stride (nth ax in-strides))
				 (out-stride (nth ax out-strides))
                                 (vals (loop for i from 0 below ax-dim
                                             for off = (+ in-ptr (* i in-stride))
                                             collect (aref in-data off)))
                                 (sv (vt-numpy-sort vals #'<)))
                            (loop for val in sv
				  for off = out-ptr then (+ off out-stride)
                                  do (setf (aref out-data off) val))))
                         ((< depth rank)
                          (let ((dim (nth depth shape))
				(in-stride (nth depth in-strides))
                                (out-stride (nth depth out-strides)))
                            (loop for i from 0 below dim do
                              (recurse (1+ depth)
				       (+ in-ptr (* i in-stride))
				       (+ out-ptr (* i out-stride))))))
                         (t nil))))
          (recurse 0 in-offset 0))
        result)
      (let* ((flat (vt-flatten tensor))
             (data (coerce (vt-data flat) 'list)))
        (vt-from-sequence (vt-numpy-sort data #'<) :dtype (vt-dtype tensor)))))

(defun vt-argsort (tensor &key (axis -1))
  (with-float-safe
    (if (null axis)
	(let* ((flat (vt-ravel tensor))
	       (n (vt-size flat))
	       (in-data (vt-data flat))
               (pairs (loop for i from 0 below n
			    collect (cons (aref in-data i) i)))
               (non-nans '())
	       (nans '()))
          (dolist (p pairs) (if (%nan-p (car p)) (push p nans) (push p non-nans)))
          (setf non-nans (stable-sort (nreverse non-nans) #'< :key #'car)
		nans (nreverse nans))
          (%make-vt :data (make-array n :element-type '(signed-byte 64)
					:initial-contents (mapcar #'cdr (append non-nans nans)))
                    :shape (list n) :strides '(1) :offset 0 :dtype :int64))
	(let* ((shape (vt-shape tensor))
	       (rank (length shape))
	       (ax (vt-normalize-axis axis rank))
               (ax-dim (nth ax shape))
	       (in-strides (vt-strides tensor))
               (in-offset (vt-offset tensor))
	       (in-data (vt-data tensor))
               (result (vt-zeros shape :dtype :int64))
	       (out-strides (vt-strides result))
               (out-data (vt-data result)))
          (labels ((recurse (depth in-ptr out-ptr)
                     (cond ((< depth ax)
                            (let ((dim (nth depth shape))
				  (in-stride (nth depth in-strides))
                                  (out-stride (nth depth out-strides)))
                              (dotimes (i dim)
				(recurse (1+ depth)
					 (+ in-ptr (* i in-stride))
					 (+ out-ptr (* i out-stride))))))
                           ((= depth ax)
                            (let* ((in-stride (nth ax in-strides))
				   (out-stride (nth ax out-strides))
                                   (tail-dims (subseq shape (1+ ax)))
                                   (tail-size (reduce #'* tail-dims :initial-value 1))
                                   (tail-in-strides (subseq in-strides (1+ ax)))
                                   (tail-out-strides (subseq out-strides (1+ ax))))
                              (dotimes (tail-i tail-size)
				(let ((extra-in 0)
				      (extra-out 0)
				      (rem tail-i))
                                  (loop for idx from (1- (length tail-dims)) downto 0
					for dim = (nth idx tail-dims)
					for is = (nth idx tail-in-strides)
					for os = (nth idx tail-out-strides)
					do (multiple-value-bind (q r) (floor rem dim)
                                             (incf extra-in (* r is))
					     (incf extra-out (* r os)) (setf rem q)))
                                  (let ((pairs (loop for pos from 0 below ax-dim
                                                     for off = (+ in-ptr (* pos in-stride) extra-in)
                                                     collect (cons (aref in-data off) pos)))
					(non-nans '()) (nans '()))
                                    (dolist (p pairs)
				      (if (%nan-p (car p))
					  (push p nans)
					  (push p non-nans)))
                                    (setf non-nans (stable-sort (nreverse non-nans) #'< :key #'car)
                                          nans (nreverse nans))
                                    (loop for (val . pos) in (append non-nans nans)
                                          for off = out-ptr then (+ off out-stride)
                                          do (setf (aref out-data (+ off extra-out)) pos)))))))
                           (t nil))))
            (recurse 0 in-offset 0))
          result))))

;;; ------------------------------------------------------------------
;;; nan 感知统计
;;; ------------------------------------------------------------------

(defun vt-nanmean (tensor &key axis keepdims dtype out)
  (let* ((mask (vt-isnan tensor))
	 (not-nan (vt-logical-not mask))
         (final-dtype (cond ((and out dtype (not (eq (vt-dtype out) dtype)))
			     (error "vt-nanmean: :out 与 :dtype 冲突"))
                            (out (vt-dtype out)) (dtype dtype)
                            (t (if (eq (vt-dtype tensor) :float32) :float32 :float64))))
         (zero (if (eq final-dtype :float32) 0.0s0 0.0d0))
         (clean (vt-where mask zero tensor :dtype final-dtype))
         ;; count 使用与最终结果相同的浮点dtype，避免float->int强制转换错误
         (count (vt-sum not-nan :axis axis :keepdims keepdims :dtype final-dtype))
         (nan (vt-get-nan final-dtype))
         (sum (vt-sum clean :axis axis :keepdims keepdims :dtype final-dtype :out out)))
    (vt-map (lambda (s c)
	      (if (<= c zero) nan (/ s c)))
	    sum count :dtype final-dtype :out sum)))

(defun vt-nanvar (tensor &key axis keepdims (ddof 0) dtype out)
  (let* ((mask (vt-isnan tensor))
	 (not-nan (vt-logical-not mask))
	 (final-dtype (cond ((and out dtype (not (eq (vt-dtype out) dtype)))
			     (error "vt-nanvar: :out 与 :dtype 冲突"))
                            (out (vt-dtype out)) (dtype dtype)
                            (t (if (eq (vt-dtype tensor) :float32) :float32 :float64))))
         (nan (vt-get-nan final-dtype))
	 (zero (if (eq final-dtype :float32) 0.0s0 0.0d0))
         (clean (vt-where mask zero tensor :dtype final-dtype))
         ;; count 使用浮点dtype
         (count (vt-sum not-nan :axis axis :keepdims keepdims :dtype final-dtype))
         (mean (vt-nanmean tensor :axis axis :keepdims t :dtype final-dtype))
         (sq-diff (vt-* (vt-map (lambda (c m)
				  (* (- c m) (- c m)))
				clean mean :dtype final-dtype)
                        not-nan :dtype final-dtype))
         (sum2 (vt-sum sq-diff :axis axis :keepdims keepdims :dtype final-dtype :out out))
         ;; divisor 使用浮点dtype
         (ddof-f (coerce ddof (vt-dtype->lisp-type final-dtype)))
         (divisor (vt-map (lambda (c)
			    (if (< c ddof-f) zero (- c ddof-f)))
			  count :dtype final-dtype)))
    (vt-map (lambda (s d)
	      (if (<= d zero) nan (/ s d)))
	    sum2 divisor :dtype final-dtype :out sum2)))

(defun vt-nanstd (tensor &key axis keepdims (ddof 0) dtype out)
  (let* ((final-dtype (cond ((and out dtype (not (eq (vt-dtype out) dtype)))
			     (error "vt-nanstd: :out 与 :dtype 冲突"))
                            (out (vt-dtype out)) (dtype dtype)
                            (t (if (eq (vt-dtype tensor) :float32)
				   :float32 :float64))))
         (var (vt-nanvar tensor :axis axis :keepdims keepdims :ddof ddof
				:dtype final-dtype :out out)))
    (vt-sqrt var :dtype final-dtype :out var)))

(defun vt-nanmedian (tensor &key axis keepdims out)
  (with-float-safe
    (if (member (vt-dtype tensor) '(:int32 :int64))
	(vt-median tensor :axis axis)
	(let* ((nan (vt-get-nan :float64))
	       (in-data (vt-data tensor))
               (in-strides (vt-strides tensor))
	       (in-offset (vt-offset tensor))
               (in-shape (vt-shape tensor))
	       (rank (length in-shape)))
          (if (null axis)
              (let ((vals '()))
		(vt-do-each (ptr val tensor)
                  (declare (ignore ptr))
                  (unless (%nan-p val) (push val vals)))
		(setf vals (sort vals #'<))
		(let ((result (cond ((null vals) nan)
                                    ((oddp (length vals))
				     (coerce (nth (floor (length vals) 2) vals) 'double-float))
                                    (t (/ (+ (nth (1- (/ (length vals) 2)) vals)
					     (nth (/ (length vals) 2) vals))
					  2.0d0)))))
                  (if out (progn (vt-fill out result) out)
		      (make-vt nil result :dtype :float64))))
              (let* ((ax (vt-normalize-axis axis rank))
		     (ax-size (nth ax in-shape))
                     (ax-stride (nth ax in-strides))
                     (out-shape (if keepdims
                                    (loop for d in in-shape for i from 0
					  collect (if (= i ax) 1 d))
                                    (loop for d in in-shape for i from 0
					  unless (= i ax) collect d)))
                     (res (vt-zeros out-shape :dtype :float64))
                     (res-data (vt-data res)) (res-offset (vt-offset res))
                     (out-rank (length out-shape))
                     (out-dims (coerce out-shape 'simple-vector))
                     (out-strs (coerce (vt-strides res) 'simple-vector))
                     (in-map (let ((m (make-array out-rank :element-type 'fixnum)) (k 0))
                               (loop for i from 0 below rank unless (= i ax)
                                     do (setf (aref m k) i) (incf k))
                               m))
                     (in-strs (coerce (loop for i below out-rank
					    collect (nth (aref in-map i) in-strides))
                                      'simple-vector)))
		(labels ((compute (depth in-ptr out-ptr)
                           (if (= depth out-rank)
                               (let ((vals '()))
				 (loop for i from 0 below ax-size
                                       for ptr = in-ptr then (+ ptr ax-stride)
                                       for v = (aref in-data ptr)
                                       unless (%nan-p v)
					 do (push (coerce v 'double-float) vals))
				 (setf vals (nreverse vals))
				 (setf (aref res-data out-ptr)
                                       (cond ((null vals) nan)
                                             ((oddp (length vals))
					      (nth (floor (length vals) 2) vals))
                                             (t (/ (+ (nth (1- (/ (length vals) 2)) vals)
						      (nth (/ (length vals) 2) vals))
						   2.0d0)))))
                               (let ((dim (svref out-dims depth))
				     (out-str (svref out-strs depth))
                                     (in-str (svref in-strs depth)))
				 (loop for i from 0 below dim do
                                   (compute (1+ depth) in-ptr out-ptr)
                                   (incf in-ptr in-str) (incf out-ptr out-str))))))
                  (compute 0 in-offset res-offset))
		(if out (progn (vt-copy-into out res) out) res)))))))

;;; ------------------------------------------------------------------
;;; 差分 / 积分 / 相关 / 卷积 / 插值 / 梯度
;;; ------------------------------------------------------------------

(defun vt-diff (vt &key (axis -1) (n 1))
  (let ((result vt))
    (loop repeat n do
      (let* ((sh (vt-shape result))
	     (ax (vt-normalize-axis axis (length sh)))
	     (len (nth ax sh)))
        (when (< len 2)
          (return-from vt-diff (vt-zeros (append (subseq sh 0 ax) '(0) (subseq sh (1+ ax)))
                                         :dtype (vt-dtype result))))
        (setf result (vt-- (vt-narrow result ax 1 len) (vt-narrow result ax 0 (1- len)))))
	  finally (return result))))

(defun vt-trapz (y &key (x nil) (dx 1.0d0) (axis -1))
  (let* ((sh (vt-shape y))
	 (ax (vt-normalize-axis axis (length sh)))
	 (n (nth ax sh)))
    (when (< n 2)
      (return-from vt-trapz
	(vt-zeros (append (subseq sh 0 ax) (subseq sh (1+ ax))) :dtype (vt-dtype y))))
    (let ((h (if x
		 (vt-diff (ensure-vt x))
		 (make-vt (list (1- n)) dx :dtype (vt-dtype y)))))
      (setf h (vt-reshape h (append (make-list ax :initial-element 1) (list (1- n))
                                    (make-list (- (length sh) ax 1) :initial-element 1))))
      (let* ((left (vt-narrow y ax 0 (1- n)))
	     (right (vt-narrow y ax 1 n))
	     (integrand (vt-map (lambda (l r hh)
				  (* 0.5d0 (+ l r) hh))
				left right h)))
        (vt-sum integrand :axis ax)))))

(defun %normalize-mode-keyword (mode)
  "将 \"full\", :full 归一化为关键字 :full，方便 numpy 用户习惯。"
  (etypecase mode
    (keyword mode)
    (string (intern (string-upcase mode) "KEYWORD"))))

(defun vt-correlate (a v &key (mode :full))
  "1D 互相关，对标 np.correlate。mode 支持关键字(:full/:valid/:same)或字符串。"
  (let* ((mode-kw (%normalize-mode-keyword mode))
         (a-flat (vt-contiguous (vt-flatten a)))
	 (v-flat (vt-contiguous (vt-flatten v)))
         (n (vt-size a-flat))
	 (m (vt-size v-flat))
         (a-data (vt-data a-flat))
	 (v-data (vt-data v-flat)))
    (flet ((compute (k)
             (let ((sum 0.0d0))
               (loop for j from (max 0 (- k)) below (min m (- n k))
                     do (incf sum (* (aref a-data (+ j k)) (aref v-data j))))
               sum)))
      (let* ((full-len (+ n m -1)) (offset (1- m))
				   (full (make-array full-len :element-type 'double-float)))
        (loop for k from (- offset) below n for i from 0
              do (setf (aref full i) (compute k)))
        (ecase mode-kw
          (:full (%make-vt :data full :shape (list full-len) :strides '(1)
			   :offset 0 :dtype :float64))
          (:valid (let* ((len (max 0 (1+ (- n m)))) (start offset)
						    (data (make-array len :element-type 'double-float)))
                    (loop for i from 0 below len do
		      (setf (aref data i) (aref full (+ start i))))
                    (%make-vt :data data :shape (list len) :strides '(1)
			      :offset 0 :dtype :float64)))
          (:same (let* ((out-len (max n m))
			(start (floor (- full-len out-len) 2))
                        (data (make-array out-len :element-type 'double-float)))
                   (loop for i from 0 below out-len do
		     (setf (aref data i) (aref full (+ start i))))
                   (%make-vt :data data :shape (list out-len) :strides '(1)
			     :offset 0 :dtype :float64))))))))

(defun vt-convolve (a v &key (mode :full))
  "1D 卷积，对标 np.convolve。mode 支持关键字(:full/:valid/:same)或字符串。"
  (vt-correlate (vt-contiguous a)
		(vt-contiguous (vt-flip v))
		:mode (%normalize-mode-keyword mode)))

(defun vt-interp (x xp fp &key (left nil) (right nil))
  (let* ((xp-vt (if (eq (vt-dtype xp) :float64)
		    xp (vt-astype xp :float64)))
         (fp-vt (if (eq (vt-dtype fp) :float64)
		    fp (vt-astype fp :float64)))
         (x-vt (ensure-vt x :dtype :float64))
         (xp-data (vt-data xp-vt))
	 (fp-data (vt-data fp-vt))
	 (n (vt-size xp-vt))
         (x-data (vt-data x-vt))
	 (x-size (vt-size x-vt))
         (out (vt-zeros (vt-shape x-vt) :dtype :float64))
	 (out-data (vt-data out))
         (xp0 (aref xp-data 0))
	 (fp0 (aref fp-data 0))
         (xp-end (aref xp-data (1- n)))
	 (fp-end (aref fp-data (1- n)))
         (left-val (if left (vt-cast left :float64) fp0))
         (right-val (if right (vt-cast right :float64) fp-end)))
    (loop for i from 0 below x-size
	  for xi = (aref x-data i) do
	    (setf (aref out-data i)
		  (cond ((<= xi xp0) left-val)
			((>= xi xp-end) right-val)
			(t (let ((lo 0) (hi (- n 2)))
			     (loop while (< lo hi) do
                               (let ((mid (ash (+ lo hi 1) -1)))
				 (if (<= (aref xp-data mid) xi)
				     (setf lo mid)
				     (setf hi (1- mid)))))
			     (let* ((xl (aref xp-data lo))
				    (xr (aref xp-data (1+ lo)))
				    (yl (aref fp-data lo))
				    (yr (aref fp-data (1+ lo)))
				    (denom (- xr xl)))
                               (if (zerop denom)
				   yl
				   (+ yl (* (- yr yl)
					    (/ (- xi xl) denom))))))))))
    out))

(defun vt-gradient (tensor &key (spacing 1.0d0) axis)
  (let* ((shape (vt-shape tensor))
	 (rank (length shape))
         (axes (cond ((null axis)
		      (loop for i below rank collect i))
                     ((integerp axis) (list (vt-normalize-axis axis rank)))
                     ((listp axis) (mapcar (lambda (a) (vt-normalize-axis a rank)) axis))
                     (t (error "axis 必须是 nil、整数或整数列表"))))
         (spacings (cond ((numberp spacing) (make-list (length axes) :initial-element spacing))
                         ((listp spacing) spacing)
                         ((vt-p spacing) (list spacing))
                         (t (error "spacing 必须是数字、列表或 1d 张量")))))
    (labels ((slice-specs (ax s e)
               (loop for d from 0 below rank
		     collect (if (= d ax) (list s e) '(:all))))
             (grad-along (ax sp)
               (let ((n (nth ax shape)))
                 (when (< n 2) (error "轴 ~a 长度 ~a 太小" ax n))
                 (if (numberp sp)
                     (if (= n 2)
                         (let ((edge (vt-/ (vt-- (apply #'vt-slice tensor (slice-specs ax 1 2))
                                                 (apply #'vt-slice tensor (slice-specs ax 0 1)))
					   sp)))
                           (vt-concatenate ax edge edge))
                         (let ((left (vt-/ (vt-- (apply #'vt-slice tensor (slice-specs ax 1 2))
                                                 (apply #'vt-slice tensor (slice-specs ax 0 1)))
					   sp))
                               (inner (vt-/ (vt-- (apply #'vt-slice tensor (slice-specs ax 2 n))
                                                  (apply #'vt-slice tensor (slice-specs ax 0 (- n 2))))
					    (* 2.0d0 sp)))
                               (right (vt-/ (vt-- (apply #'vt-slice tensor (slice-specs ax (1- n) n))
                                                  (apply #'vt-slice tensor (slice-specs ax (- n 2) (1- n))))
					    sp)))
                           (vt-concatenate ax left inner right)))
                     (let* ((h (ensure-vt sp)) (nh (vt-size h)))
                       (assert (= n nh) (sp) "spacing 数组长度必须与轴一致")
                       (if (= n 2)
                           (let* ((hd (vt-- (vt-slice h '(1 2)) (vt-slice h '(0 1))))
                                  (df (vt-- (apply #'vt-slice tensor (slice-specs ax 1 2))
                                            (apply #'vt-slice tensor (slice-specs ax 0 1))))
                                  (g (vt-/ df hd)))
                             (vt-concatenate ax g g))
                           (let ((hl (vt-- (vt-slice h '(1 2)) (vt-slice h '(0 1))))
                                 (hr (vt-- (vt-slice h (list (1- n) n)) (vt-slice h (list (- n 2) (1- n)))))
                                 (hi (vt-- (vt-slice h (list 2 n)) (vt-slice h (list 0 (- n 2)))))
                                 (dl (vt-- (apply #'vt-slice tensor (slice-specs ax 1 2))
                                           (apply #'vt-slice tensor (slice-specs ax 0 1))))
                                 (dr (vt-- (apply #'vt-slice tensor (slice-specs ax (1- n) n))
                                           (apply #'vt-slice tensor (slice-specs ax (- n 2) (1- n)))))
                                 (di (vt-- (apply #'vt-slice tensor (slice-specs ax 2 n))
                                           (apply #'vt-slice tensor (slice-specs ax 0 (- n 2))))))
                             (vt-concatenate ax (vt-/ dl hl) (vt-/ di hi) (vt-/ dr hr)))))))))
      (let ((results (loop for ax in axes for sp in spacings
			   collect (grad-along ax sp))))
        (if (and (or (null axis) (integerp axis))
		 (null (cdr results)))
	    (car results) results)))))

