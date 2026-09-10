;;;; map-reduce.lisp — 逐元素映射与归约核心

(in-package :clvt)

;;; ------------------------------------------------------------------
;;; 逐元素映射：vt-map
;;; ------------------------------------------------------------------

(defun vt-map (fn &rest args)
  "高效逐元素映射：支持标量/列表/张量混合输入并自动广播。"
  (declare (function fn) (optimize (speed 3) (safety 0)))
  (with-float-safe
    (multiple-value-bind (tensors dtype out) (parse-vt-op-args args)
      (when (null tensors)
	(error "vt-map 至少需要一个输入张量"))
      (let* ((inputs (mapcar #'ensure-vt tensors))
             (out-shape (reduce #'vt-broadcast-shapes (mapcar #'vt-shape inputs)))
             (final-dtype (cond
                            ((and out dtype (not (eq (vt-dtype out) dtype)))
                             (error "vt-map: :out 类型 (~a) 与 :dtype (~a) 冲突"
                                    (vt-dtype out) dtype))
                            (out (vt-dtype out))
                            (dtype dtype)
                            (t (apply #'vt-promote-type (mapcar #'vt-dtype inputs)))))
             (res (or out (%make-vt-uninit out-shape final-dtype))))
	(when out
          (unless (equal (vt-shape res) out-shape)
            (error "vt-map: :out 形状 ~a 与广播结果 ~a 不匹配" (vt-shape res) out-shape)))
	(%vt-map-run fn inputs res out-shape)
	res))))

(defun %vt-map-run (fn inputs res out-shape)
  (declare (optimize (speed 3) (safety 0))
           (list inputs out-shape))
  (let* ((n (length inputs))
         (res-data (vt-data res))
         (res-dtype (vt-dtype res))
         (res-strides-vec (coerce (vt-strides res) 'simple-vector))
         (dims (coerce out-shape 'simple-vector))
         (rank (length out-shape))
         (size (vt-shape-to-size out-shape))
         (in-datas (coerce (mapcar #'vt-data inputs) 'simple-vector))
         (in-offsets (coerce (mapcar #'vt-offset inputs) 'simple-vector))
         (in-sizes (coerce (mapcar #'vt-size inputs) 'simple-vector))
         (in-strides (coerce (mapcar (lambda (in)
                                       (coerce (vt-broadcast-strides
                                                (vt-shape in) out-shape (vt-strides in))
                                               'simple-vector))
                                     inputs)
                             'simple-vector))
         (contig (and (vt-contiguous-p res)
                      (every (lambda (in)
                               (or (= (vt-size in) 1)
                                   (and (equal (vt-shape in) out-shape)
                                        (vt-contiguous-p in))))
                             inputs)))
         (all-same (every (lambda (in) (eq (vt-dtype in) res-dtype)) inputs))
         (res-off (vt-offset res)))
    (declare (type simple-vector res-strides-vec dims in-datas in-offsets in-sizes in-strides)
             (type fixnum n rank size res-off))

    (labels ((gather (ptrs vals)
               (loop for k fixnum from 0 below n
                     do (setf (aref vals k) (aref (svref in-datas k) (aref ptrs k)))))
             (call-fn (vals)
               (case n
                 (1 (funcall fn (aref vals 0)))
                 (2 (funcall fn (aref vals 0) (aref vals 1)))
                 (3 (funcall fn (aref vals 0) (aref vals 1) (aref vals 2)))
                 (4 (funcall fn (aref vals 0) (aref vals 1) (aref vals 2) (aref vals 3)))
                 (5 (funcall fn (aref vals 0) (aref vals 1) (aref vals 2) (aref vals 3) (aref vals 4)))
                 (otherwise (apply fn (coerce vals 'list))))))
      (macrolet ((cast-to (lt form)
                   (if (subtypep lt 'integer)
                       `(truncate ,form)
                       `(the ,lt (coerce ,form ',lt))))
                 ;; 快路径：连续 + 输入/输出同类型 + n=k（k=1..5）
                 ;; 展开后与原代码逐字等价，只是把 5 份重复合并
                 (same-k (lt k)
                   (let ((ds (loop for i below k collect (gensym "D")))
                         (ps (loop for i below k collect (gensym "P")))
                         (ss (loop for i below k collect (gensym "S"))))
                     `(let ((od (the (simple-array ,lt (*)) res-data))
                            ,@(loop for d in ds for i below k
                                    collect `(,d (the (simple-array ,lt (*))
                                                      (svref in-datas ,i))))
                            ,@(loop for p in ps for i below k
                                    collect `(,p (svref in-offsets ,i)))
                            ,@(loop for s in ss for i below k
                                    collect `(,s (if (= (svref in-sizes ,i) 1) 0 1)))
                            (op res-off))
                        (declare (type (simple-array ,lt (*)) od ,@ds)
                                 (type fixnum ,@ps ,@ss op))
                        (dotimes (i size)
                          (setf (aref od op)
                                (cast-to ,lt
                                         (funcall fn
                                                  ,@(loop for d in ds for p in ps
                                                          collect `(aref ,d ,p)))))
                          (incf op)
                          ,@(loop for p in ps for s in ss collect `(incf ,p ,s))))))
                 ;; 连续但类型不同 / n>5：保留 od + cast-to
                 (contig-any (lt)
                   `(let ((od (the (simple-array ,lt (*)) res-data))
                          (ptrs (copy-seq in-offsets))
                          (vals (make-array n))
                          (steps (coerce (loop for k below n
                                               collect (if (= (svref in-sizes k) 1) 0 1))
                                         'simple-vector))
                          (op res-off))
                      (declare (type (simple-array ,lt (*)) od)
                               (type simple-vector ptrs vals steps)
                               (type fixnum op))
                      (dotimes (i size)
                        (gather ptrs vals)
                        (setf (aref od op) (cast-to ,lt (call-fn vals)))
                        (incf op)
                        (loop for k fixnum below n do
                          (incf (aref ptrs k) (aref steps k))))))
                 ;; 非连续：保留 od + cast-to
                 (noncontig (lt)
                   `(let ((od (the (simple-array ,lt (*)) res-data))
                          (ptrs (copy-seq in-offsets))
                          (vals (make-array n))
                          (indices (make-array rank :element-type 'fixnum :initial-element 0))
                          (op res-off))
                      (declare (type (simple-array ,lt (*)) od)
                               (type simple-vector ptrs vals)
                               (type (simple-array fixnum (*)) indices)
                               (type fixnum op))
                      (dotimes (i size)
                        (gather ptrs vals)
                        (setf (aref od op) (cast-to ,lt (call-fn vals)))
                        (let ((d (1- rank)))
                          (loop
                            (when (< d 0) (return))
                            (incf (aref indices d))
                            (if (< (aref indices d) (svref dims d))
                                (progn (incf op (svref res-strides-vec d))
                                       (loop for k fixnum from 0 below n do
                                         (incf (aref ptrs k)
                                               (aref (svref in-strides k) d)))
                                       (return))
                                (progn (setf (aref indices d) 0)
                                       (decf op (* (svref res-strides-vec d)
                                                   (1- (svref dims d))))
                                       (loop for k fixnum from 0 below n do
                                         (decf (aref ptrs k)
                                               (* (aref (svref in-strides k) d)
                                                  (1- (svref dims d)))))
                                       (decf d))))))))
		 (gen (lt)
		   `(cond
		      ((and contig all-same (= n 1)) (same-k ,lt 1))
		      ((and contig all-same (= n 2)) (same-k ,lt 2))
		      ((and contig all-same (= n 3)) (same-k ,lt 3))
		      ((and contig all-same (= n 4)) (same-k ,lt 4))
		      ((and contig all-same (= n 5)) (same-k ,lt 5))
		      (contig (contig-any ,lt))
		      (t (noncontig ,lt))))
		 )
        (cond ((equal (array-element-type res-data) 'double-float) (gen double-float))
              ((equal (array-element-type res-data) 'single-float) (gen single-float))
              ((equal (array-element-type res-data) '(signed-byte 64)) (gen (signed-byte 64)))
              ((equal (array-element-type res-data) '(signed-byte 32)) (gen (signed-byte 32)))
              (t
               ;; 非特化类型：保持原实现（无 cast，通用 aref）
               (let ((ptrs (copy-seq in-offsets))
                     (vals (make-array n))
                     (indices (make-array rank :element-type 'fixnum :initial-element 0))
                     (op res-off))
                 (declare (type simple-vector ptrs vals)
                          (type (simple-array fixnum (*)) indices)
                          (type fixnum op))
                 (dotimes (i size)
                   (gather ptrs vals)
                   (setf (aref res-data op) (call-fn vals))
                   (let ((d (1- rank)))
                     (loop
                       (when (< d 0) (return))
                       (incf (aref indices d))
                       (if (< (aref indices d) (svref dims d))
                           (progn (incf op (svref res-strides-vec d))
                                  (loop for k fixnum from 0 below n do
                                    (incf (aref ptrs k)
                                          (aref (svref in-strides k) d)))
                                  (return))
                           (progn (setf (aref indices d) 0)
                                  (decf op (* (svref res-strides-vec d)
                                              (1- (svref dims d))))
                                  (loop for k fixnum from 0 below n do
                                    (decf (aref ptrs k)
                                          (* (aref (svref in-strides k) d)
                                             (1- (svref dims d)))))
                                  (decf d))))))))))
      res)))

;;; ------------------------------------------------------------------
;;; 二元特化（内部，供算术层使用）
;;; ------------------------------------------------------------------

(defun vt-binary (fn t1 t2 &key out dtype)
  "二元逐元素运算（内部入口，等价于两参数 vt-map）。"
  (declare (optimize (speed 3) (safety 0)))
  (apply #'vt-map fn (ensure-vt t1) (ensure-vt t2) :out out :dtype dtype))

;;; ------------------------------------------------------------------
;;; 编译期内联逐元素映射（对标原 vt-fast-map，消除 funcall 装箱）
;;; ------------------------------------------------------------------

(defmacro %cast-to (lt form)
  "内联类型转换：整数截断，浮点 coerce。"
  (declare (optimize (speed 3) (safety 0)))
  `(if ,(subtypep lt 'integer) (truncate ,form) (coerce ,form ',lt)))

(defmacro %inline1-loop (lt op a res)
  (declare (optimize (speed 3) (safety 0)))
  "一元内联循环（lt 为类型，op 为算子符号）。"
  `(let ((od (the (simple-array ,lt (*)) (vt-data ,res)))
         (d0 (the (simple-array ,lt (*)) (vt-data ,a)))
         (p0 (vt-offset ,a))
         (s0 (if (= (vt-size ,a) 1) 0 1))
         (op (vt-offset ,res)))
     (declare (type (simple-array ,lt (*)) od d0) (type fixnum p0 s0 op))
     (loop for i fixnum from 0 below (vt-size ,res) do
       (setf (aref od op) (%cast-to ,lt (,op (aref d0 p0))))
       (incf op) (incf p0 s0))))

(defmacro %inline2-loop (lt op a b res)
  "二元内联循环（lt 为类型，op 为算子符号）。"
  (declare (optimize (speed 3) (safety 0)))
  `(let ((od (the (simple-array ,lt (*)) (vt-data ,res)))
         (d0 (the (simple-array ,lt (*)) (vt-data ,a)))
         (d1 (the (simple-array ,lt (*)) (vt-data ,b)))
         (p0 (vt-offset ,a)) (p1 (vt-offset ,b))
         (s0 (if (= (vt-size ,a) 1) 0 1))
         (s1 (if (= (vt-size ,b) 1) 0 1))
         (op (vt-offset ,res)))
     (declare (type (simple-array ,lt (*)) od d0 d1) (type fixnum p0 p1 s0 s1 op))
     (loop for i fixnum from 0 below (vt-size ,res) do
       (setf (aref od op) (%cast-to ,lt (,op (aref d0 p0) (aref d1 p1))))
       (incf op) (incf p0 s0) (incf p1 s1))))

(defmacro %vt-inline1-fast (op a res)
  (declare (optimize (speed 3) (safety 0)))
  "一元连续快路径（按输出类型派发，内联 op）。"
  `(let ((rd (vt-data ,res)))
     (cond ((equal (array-element-type rd) 'double-float) (%inline1-loop double-float ,op ,a ,res))
           ((equal (array-element-type rd) 'single-float) (%inline1-loop single-float ,op ,a ,res))
           ((equal (array-element-type rd) '(signed-byte 64)) (%inline1-loop (signed-byte 64) ,op ,a ,res))
           ((equal (array-element-type rd) '(signed-byte 32)) (%inline1-loop (signed-byte 32) ,op ,a ,res))
           (t (vt-map (function ,op) ,a :out ,res)))))

(defmacro %vt-inline2-fast (op a b res)
  "二元连续快路径（按输出类型派发，内联 op）。"
  (declare (optimize (speed 3) (safety 0)))
  `(let ((rd (vt-data ,res)))
     (cond ((equal (array-element-type rd) 'double-float) (%inline2-loop double-float ,op ,a ,b ,res))
           ((equal (array-element-type rd) 'single-float) (%inline2-loop single-float ,op ,a ,b ,res))
           ((equal (array-element-type rd) '(signed-byte 64)) (%inline2-loop (signed-byte 64) ,op ,a ,b ,res))
           ((equal (array-element-type rd) '(signed-byte 32)) (%inline2-loop (signed-byte 32) ,op ,a ,b ,res))
           (t (vt-map (function ,op) ,a ,b :out ,res)))))

(defmacro vt-fast-map (fn &rest args)
  "编译期内联已知算子的逐元素映射（一元/二元）；否则回退到 vt-map。"
  (declare (optimize (speed 3) (safety 0)))
  (let ((op (and (consp fn) (eq (car fn) 'function) (symbolp (cadr fn)) (cadr fn))))
    (if (null op)
        `(apply #'vt-map ,fn ,@args)
        (multiple-value-bind (tensors dtype out)
	    (parse-vt-op-args args)
	  (declare (list tensors))
          (let* ((n (length tensors))
                 (tvs (loop repeat n collect (gensym "TV"))))
            (if (not (or (= n 1) (= n 2))) 
                `(apply #'vt-map ,fn ,@args)
                `(let (,@(loop for tv in tvs for tf in tensors
                               collect `(,tv (ensure-vt ,tf))))
                   (let* ((out-shape (reduce #'vt-broadcast-shapes
                                             (mapcar #'vt-shape (list ,@tvs))))
                          (final-dtype (cond ((and ,out ,dtype
                                                   (not (eq (vt-dtype ,out) ,dtype)))
                                              (error "类型冲突: :out (~a) vs :dtype (~a)"
                                                     (vt-dtype ,out) ,dtype))
                                             (,out (vt-dtype ,out))
                                             (,dtype ,dtype)
                                             (t (apply #'vt-promote-type
                                                       (mapcar #'vt-dtype (list ,@tvs))))))
                          (res (or ,out (make-vt out-shape 0 :dtype final-dtype))))
                     (when ,out
                       (unless (equal (vt-shape res) out-shape)
                         (error ":out 形状 ~a 与广播结果 ~a 不匹配"
                                (vt-shape res) out-shape)))
                     ,(if (= n 1)
                          `(if (and (vt-contiguous-p res)
                                    (or (= (vt-size ,(first tvs)) 1)
                                        (and (equal (vt-shape ,(first tvs)) out-shape)
                                             (vt-contiguous-p ,(first tvs))))
                                    (eq (vt-dtype ,(first tvs)) (vt-dtype res)))
                               (%vt-inline1-fast ,op ,(first tvs) res)
                               (vt-map (function ,op) ,(first tvs) :out res))
                          `(if (and (vt-contiguous-p res)
                                    (or (= (vt-size ,(first tvs)) 1)
                                        (and (equal (vt-shape ,(first tvs)) out-shape)
                                             (vt-contiguous-p ,(first tvs))))
                                    (or (= (vt-size ,(second tvs)) 1)
                                        (and (equal (vt-shape ,(second tvs)) out-shape)
                                             (vt-contiguous-p ,(second tvs))))
                                    (eq (vt-dtype ,(first tvs)) (vt-dtype res))
                                    (eq (vt-dtype ,(second tvs)) (vt-dtype res)))
                               (%vt-inline2-fast ,op ,(first tvs) ,(second tvs) res)
                               (vt-map (function ,op) ,(first tvs) ,(second tvs) :out res)))
                     res))))))))


;;; ------------------------------------------------------------------
;;; 归约核心：vt-reduce
;;; ------------------------------------------------------------------

(defun get-reduction-identity (op element-type)
  "返回指定归约操作在给定元素类型下的初始值。"
  (declare (optimize (speed 3) (safety 0)))
  (case op
    (:sum (coerce 0 element-type))
    (:max (cond ((eq element-type 'double-float) +vt-dfloat-neg-inf+)
                ((eq element-type 'single-float) +vt-sfloat-neg-inf+)
                ((equal element-type '(signed-byte 64)) (- (expt 2 63)))
                ((equal element-type '(signed-byte 32)) (- (expt 2 31)))
                ((equal element-type '(unsigned-byte 64)) 0)
                ((equal element-type '(unsigned-byte 32)) 0)
                ((equal element-type '(unsigned-byte 16)) 0)
                ((equal element-type '(unsigned-byte 8)) 0)
                ((subtypep element-type 'integer) most-negative-fixnum)
                (t 0)))
    (:min (cond ((eq element-type 'double-float) +vt-dfloat-pos-inf+)
                ((eq element-type 'single-float) +vt-sfloat-pos-inf+)
                ((equal element-type '(signed-byte 64)) (1- (expt 2 63)))
                ((equal element-type '(signed-byte 32)) (1- (expt 2 31)))
                ((equal element-type '(unsigned-byte 64)) (1- (expt 2 64)))
                ((equal element-type '(unsigned-byte 32)) (1- (expt 2 32)))
                ((equal element-type '(unsigned-byte 16)) (1- (expt 2 16)))
                ((equal element-type '(unsigned-byte 8)) (1- (expt 2 8)))
                ((subtypep element-type 'integer) most-positive-fixnum)
                (t 0)))))

(defun vt-reduce (tensor axis init-val reducer-fn &key out dtype keepdims return-arg)
  (declare (type vt tensor)
           (type (or null fixnum list) axis)
           (type function reducer-fn))
  (with-float-safe
    (let* ((in-shape (vt-shape tensor))
           (rank (length in-shape))
           (axes (vt-normalize-axes axis rank))
           (global (null axes))
           (out-shape (cond ((and global (not keepdims)) nil)
                            (global (make-list rank :initial-element 1))
                            ((not keepdims)
                             (loop for d in in-shape
                                   for i fixnum from 0
                                   unless (member i axes) collect d))
                            (t (loop for d in in-shape
                                     for i fixnum from 0
                                     collect (if (member i axes) 1 d)))))
           (axis-size (if axes
                          (reduce #'* (mapcar (lambda (a) (nth a in-shape)) axes)
                                  :initial-value 1)
                          (reduce #'* in-shape :initial-value 1))))
      (declare (fixnum rank axis-size))

      ;; 空输入
      (when (or (zerop axis-size) (zerop (vt-size tensor)))
        (let ((empty-dtype (or dtype (and out (vt-dtype out)) (vt-dtype tensor))))
          (return-from vt-reduce
            (values (make-vt out-shape (or init-val 0) :dtype empty-dtype)
                    (when return-arg (make-vt out-shape 0 :dtype :int32))))))

      (let* ((final-dtype (cond
                            ((and out dtype (not (eq (vt-dtype out) dtype)))
                             (error "vt-reduce: :out type (~a) 与 :dtype (~a) 冲突"
                                    (vt-dtype out) dtype))
                            (out (vt-dtype out))
                            (dtype dtype)
                            ((and init-val (or (floatp init-val) (%inf-p init-val))) :float64)
                            (t (vt-dtype tensor))))
             (res (or out (%make-vt-uninit out-shape final-dtype))))

        ;; ================================================================
        ;; 快路径 1：连续 + 全局归约（axis=nil, 不 keepdims, 无 arg）
        ;; ================================================================
        (when (and global (not keepdims) (not return-arg)
                   (numberp init-val)
                   (vt-contiguous-p tensor))
          (let* ((size (vt-size tensor))
                 (in-data (vt-data tensor))
                 (in-off (vt-offset tensor))
                 (in-et (array-element-type in-data))
                 (res-data (vt-data res))
                 (res-off (vt-offset res)))
            (declare (type fixnum size in-off res-off))
            (labels ((store-fast (val)
                       (setf (aref (the (simple-array * (*)) res-data) res-off)
                             (vt-cast val final-dtype))
                       (return-from vt-reduce (values res nil))))
              (cond
                ((equal in-et 'double-float)
                 (let ((d (the (simple-array double-float (*)) in-data)))
                   (cond
                     ((and (typep init-val 'double-float) (zerop init-val))
                      (let ((acc 0.0d0) (p in-off) (end (the fixnum (+ in-off size))))
                        (declare (type double-float acc) (type fixnum p end))
                        (loop while (< p end) do (incf acc (aref d p)) (incf p))
                        (store-fast acc)))
                     ((= init-val +vt-dfloat-neg-inf+)
                      (let ((acc +vt-dfloat-neg-inf+) (p in-off) (end (the fixnum (+ in-off size))))
                        (declare (type double-float acc) (type fixnum p end))
                        (loop while (< p end) do
                          (let ((v (aref d p))) (when (> v acc) (setf acc v)))
                          (incf p))
                        (store-fast acc)))
                     ((= init-val +vt-dfloat-pos-inf+)
                      (let ((acc +vt-dfloat-pos-inf+) (p in-off) (end (the fixnum (+ in-off size))))
                        (declare (type double-float acc) (type fixnum p end))
                        (loop while (< p end) do
                          (let ((v (aref d p))) (when (< v acc) (setf acc v)))
                          (incf p))
                        (store-fast acc))))))
                ((equal in-et 'single-float)
                 (let ((d (the (simple-array single-float (*)) in-data)))
                   (when (and (typep init-val 'single-float) (zerop init-val))
                     (let ((acc 0.0s0) (p in-off) (end (the fixnum (+ in-off size))))
                       (declare (type single-float acc) (type fixnum p end))
                       (loop while (< p end) do (incf acc (aref d p)) (incf p))
                       (store-fast acc)))))
                ((equal in-et '(signed-byte 32))
                 (when (eql init-val 0)
                   (let ((d (the (simple-array (signed-byte 32) (*)) in-data))
                         (acc 0) (p in-off) (end (the fixnum (+ in-off size))))
                     (declare (type (signed-byte 32) acc) (type fixnum p end))
                     (loop while (< p end) do (incf acc (aref d p)) (incf p))
                     (store-fast acc))))
                ((equal in-et '(signed-byte 64))
                 (when (eql init-val 0)
                   (let ((d (the (simple-array (signed-byte 64) (*)) in-data))
                         (acc 0) (p in-off) (end (the fixnum (+ in-off size))))
                     (declare (type (signed-byte 64) acc) (type fixnum p end))
                     (loop while (< p end) do (incf acc (aref d p)) (incf p))
                     (store-fast acc))))))))

        ;; ================================================================
        ;; 快路径 2：连续 + 末尾连续归约轴（无 arg）——消除 odometer
        ;; ================================================================
        (when (and (not global) (not return-arg)
                   (numberp init-val)
                   (vt-contiguous-p tensor)
                   (vt-contiguous-p res)
                   (loop for a in axes
                         for i from (- rank (length axes))
                         always (= a i)))
          (let* ((inner (the fixnum axis-size))
                 (outer (the fixnum (truncate (the fixnum (vt-size tensor)) inner)))
                 (in-data (vt-data tensor))
                 (in-off (vt-offset tensor))
                 (in-et (array-element-type in-data))
                 (res-data (vt-data res))
                 (res-off (vt-offset res))
                 (res-et (array-element-type res-data)))
            (macrolet ((cast-to (lt form)
                         (if (subtypep lt 'integer)
                             `(truncate ,form)
                             `(coerce ,form ',lt)))
                       (acc-init (lt form)
                         (if (subtypep lt 'integer)
                             `(the ,lt (truncate ,form))
                             `(the ,lt (coerce ,form ',lt))))
                       ;; out-lt: 输出元素类型；in-lt: 输入元素类型；
                       ;; acc-lt: 累加器类型（必须是 in-lt 与 out-lt 的提升类型）
                       ;; init-acc 在 outer 循环外只算一次（原版每次外层迭代都算一次）
                       (blk (out-lt in-lt acc-lt op)
                         `(progn
                            (let ((ip in-off) (opos res-off)
                                  (init-acc (acc-init ,acc-lt init-val)))
                              (declare (type fixnum ip opos)
                                       (type ,acc-lt init-acc))
                              (dotimes (o outer)
                                (let ((acc init-acc))
                                  (declare (type ,acc-lt acc))
                                  (dotimes (i inner)
                                    (let ((v (aref (the (simple-array ,in-lt (*)) in-data) ip)))
                                      ,(ecase op
                                         (:sum `(incf acc v))
                                         (:max `(when (> v acc)
                                                  (setf acc (cast-to ,acc-lt v))))
                                         (:min `(when (< v acc)
                                                  (setf acc (cast-to ,acc-lt v))))))
                                    (incf ip))
                                  (setf (aref (the (simple-array ,out-lt (*)) res-data) opos)
                                        (cast-to ,out-lt acc))
                                  (incf opos))))
                            (return-from vt-reduce (values res nil)))))
              (cond
                ;; ---- SUM ----
                ((eq reducer-fn #'+)
                 (cond
                   ((and (equal res-et 'double-float) (equal in-et 'double-float))
                    (blk double-float double-float double-float :sum))
                   ((and (equal res-et 'double-float) (equal in-et 'single-float))
                    (blk double-float single-float double-float :sum))
                   ((and (equal res-et 'double-float) (equal in-et '(signed-byte 64)))
                    (blk double-float (signed-byte 64) double-float :sum))
                   ((and (equal res-et 'double-float) (equal in-et '(signed-byte 32)))
                    (blk double-float (signed-byte 32) double-float :sum))
                   ((and (equal res-et 'single-float) (equal in-et 'single-float))
                    (blk single-float single-float single-float :sum))
                   ((and (equal res-et 'single-float) (equal in-et 'double-float))
                    (blk single-float double-float double-float :sum))
                   ((and (equal res-et '(signed-byte 64)) (equal in-et '(signed-byte 64)))
                    (blk (signed-byte 64) (signed-byte 64) (signed-byte 64) :sum))
                   ((and (equal res-et '(signed-byte 32)) (equal in-et '(signed-byte 32)))
                    (blk (signed-byte 32) (signed-byte 32) (signed-byte 32) :sum))))
                ;; ---- MAX ----
                ((eq reducer-fn #'max)
                 (cond
                   ((and (equal res-et 'double-float) (equal in-et 'double-float))
                    (blk double-float double-float double-float :max))
                   ((and (equal res-et 'double-float) (equal in-et 'single-float))
                    (blk double-float single-float double-float :max))
                   ((and (equal res-et 'single-float) (equal in-et 'single-float))
                    (blk single-float single-float single-float :max))
                   ((and (equal res-et 'single-float) (equal in-et 'double-float))
                    (blk single-float double-float double-float :max))
                   ((and (equal res-et '(signed-byte 64)) (equal in-et '(signed-byte 64)))
                    (blk (signed-byte 64) (signed-byte 64) (signed-byte 64) :max))
                   ((and (equal res-et '(signed-byte 32)) (equal in-et '(signed-byte 32)))
                    (blk (signed-byte 32) (signed-byte 32) (signed-byte 32) :max))))
                ;; ---- MIN ----
                ((eq reducer-fn #'min)
                 (cond
                   ((and (equal res-et 'double-float) (equal in-et 'double-float))
                    (blk double-float double-float double-float :min))
                   ((and (equal res-et 'double-float) (equal in-et 'single-float))
                    (blk double-float single-float double-float :min))
                   ((and (equal res-et 'single-float) (equal in-et 'single-float))
                    (blk single-float single-float single-float :min))
                   ((and (equal res-et 'single-float) (equal in-et 'double-float))
                    (blk single-float double-float double-float :min))
                   ((and (equal res-et '(signed-byte 64)) (equal in-et '(signed-byte 64)))
                    (blk (signed-byte 64) (signed-byte 64) (signed-byte 64) :min))
                   ((and (equal res-et '(signed-byte 32)) (equal in-et '(signed-byte 32)))
                    (blk (signed-byte 32) (signed-byte 32) (signed-byte 32) :min))))))))

        ;; ================================================================
        ;; 通用路径：任意轴 / keepdims / return-arg / 非连续
        ;; ================================================================
        (let* ((res-data (vt-data res))
               (res-offset (vt-offset res))
               (res-strides (vt-strides res))
               (res-idx (when return-arg (make-vt out-shape 0 :dtype :int32)))
               (res-idx-data (when res-idx (vt-data res-idx)))
               (res-idx-offset (when res-idx (vt-offset res-idx)))
               (res-idx-strides (when res-idx (vt-strides res-idx)))
               (in-data (vt-data tensor))
               (in-strides (vt-strides tensor))
               (in-offset (vt-offset tensor))
               (in-shape-vec (coerce in-shape 'simple-vector))
               (in-strides-vec (coerce in-strides 'simple-vector))
               (out-strides-map
                 (if global
                     (make-list rank :initial-element 0)
                     (loop for i from 0 below rank
                           if (member i axes) collect 0
                             else collect
                             (let ((out-idx (if keepdims i
                                                (count-if-not (lambda (x) (member x axes))
                                                              (loop for j below i collect j)))))
                               (nth out-idx res-strides)))))
               (idx-strides-map
                 (if (or (not return-arg) global)
                     (make-list rank :initial-element 0)
                     (loop for i from 0 below rank
                           if (member i axes) collect 0
                             else collect
                             (let ((out-idx (if keepdims i
                                                (count-if-not (lambda (x) (member x axes))
                                                              (loop for j below i collect j)))))
                               (nth out-idx res-idx-strides)))))
               (arg-strides
                 (if return-arg
                     (if global
                         (vt-compute-strides in-shape)
                         (let* ((red-shape (mapcar (lambda (a) (nth a in-shape)) axes))
                                (red-strides (vt-compute-strides red-shape))
                                (k -1))
                           (declare (fixnum k))
                           (loop for i fixnum below rank
                                 if (member i axes)
                                   collect (progn (incf k) (nth k red-strides))
                                 else collect 0)))
                     (make-list rank :initial-element 0)))
               (out-strides-vec (coerce out-strides-map 'simple-vector))
               (idx-strides-vec (coerce idx-strides-map 'simple-vector))
               (arg-strides-vec (coerce arg-strides 'simple-vector))
               (out-et (array-element-type res-data))
               (in-et (array-element-type in-data)))

          ;; 优化：res 为新建（make-vt 已零填充）且 init-val=0 时，跳过全量填充
          ;; res-idx 恒为新建零填充，无需再 fill
          (unless (and (null out) (numberp init-val) (zerop init-val))
            (vt-fill res init-val))

          (when (= rank 0)
            (let ((val (aref in-data in-offset)))
              (multiple-value-bind (new-acc do-update-arg)
                  (funcall reducer-fn init-val val)
                (setf (aref res-data res-offset) (vt-cast new-acc final-dtype))
                (if return-arg
                    (progn
                      (when do-update-arg
                        (setf (aref res-idx-data res-idx-offset) 0))
                      (return-from vt-reduce (values res res-idx)))
                    (return-from vt-reduce (values res nil))))))

          (macrolet
              ((cast-to (lt form)
                 (cond ((null lt) form)
                       ((subtypep lt 'integer) `(truncate ,form))
                       (t `(coerce ,form ',lt))))
               (define-loop (out-lt in-lt red-type with-arg)
                 (let ((out-array (if out-lt
                                      `(the (simple-array ,out-lt (*)) res-data)
                                      'res-data))
                       (in-array (if in-lt
                                     `(the (simple-array ,in-lt (*)) in-data)
                                     'in-data)))
                   `(let ((indices (make-array rank :element-type 'fixnum :initial-element 0)))
                      (declare (type (simple-array fixnum (*)) indices))
                      (let ((in-ptr in-offset)
                            (out-ptr res-offset)
                            ,@(when with-arg `((arg-ptr res-idx-offset) (arg-val 0))))
                        (declare (type fixnum in-ptr out-ptr
                                       ,@(when with-arg '(arg-ptr arg-val))))
                        (block outer-loop
                          (loop
                            (let* ((val (aref ,in-array in-ptr))
                                   (raw-acc (aref ,out-array out-ptr)))
                              ,(ecase red-type
                                 (:sum `(setf (aref ,out-array out-ptr)
                                              (cast-to ,out-lt (+ raw-acc val))))
                                 (:max `(when (> val raw-acc)
                                          (setf (aref ,out-array out-ptr)
                                                (cast-to ,out-lt val))
                                          ,(when with-arg
                                             `(setf (aref res-idx-data arg-ptr) arg-val))))
                                 (:min `(when (< val raw-acc)
                                          (setf (aref ,out-array out-ptr)
                                                (cast-to ,out-lt val))
                                          ,(when with-arg
                                             `(setf (aref res-idx-data arg-ptr) arg-val))))
                                 (:custom
                                  (if with-arg
                                      `(multiple-value-bind (new-acc do-update-arg)
                                           (funcall reducer-fn raw-acc val)
                                         (setf (aref ,out-array out-ptr)
                                               (cast-to ,out-lt new-acc))
                                         (when do-update-arg
                                           (setf (aref res-idx-data arg-ptr) arg-val)))
                                      `(multiple-value-bind (new-acc)
                                           (funcall reducer-fn raw-acc val)
                                         (setf (aref ,out-array out-ptr)
                                               (cast-to ,out-lt new-acc)))))))
                            (let ((d (1- rank)))
                              (loop
                                (incf (aref indices d))
                                (incf in-ptr (svref in-strides-vec d))
                                (incf out-ptr (svref out-strides-vec d))
                                ,@(when with-arg
                                    `((incf arg-ptr (svref idx-strides-vec d))
                                      (incf arg-val (svref arg-strides-vec d))))
                                (when (< (aref indices d) (svref in-shape-vec d))
                                  (return))
                                (let ((dim (svref in-shape-vec d)))
                                  (decf in-ptr (* dim (svref in-strides-vec d)))
                                  (decf out-ptr (* dim (svref out-strides-vec d)))
                                  ,@(when with-arg
                                      `((decf arg-ptr (* dim (svref idx-strides-vec d)))
                                        (decf arg-val (* dim (svref arg-strides-vec d)))))
                                  (setf (aref indices d) 0)
                                  (decf d)
                                  (when (< d 0)
                                    (return-from outer-loop)))))))))))
               (generate-typed-cond (with-arg)
                 `(cond
                    ((eq reducer-fn #'+)
                     (cond
                       ((equal out-et 'double-float)
                        (cond ((equal in-et 'double-float) (define-loop double-float double-float :sum ,with-arg))
                              ((equal in-et 'single-float) (define-loop double-float single-float :sum ,with-arg))
                              ((equal in-et '(signed-byte 64)) (define-loop double-float (signed-byte 64) :sum ,with-arg))
                              ((equal in-et '(signed-byte 32)) (define-loop double-float (signed-byte 32) :sum ,with-arg))
                              (t (define-loop double-float nil :sum ,with-arg))))
                       ((equal out-et 'single-float)
                        (cond ((equal in-et 'single-float) (define-loop single-float single-float :sum ,with-arg))
                              ((equal in-et 'double-float) (define-loop single-float double-float :sum ,with-arg))
                              ((equal in-et '(signed-byte 64)) (define-loop single-float (signed-byte 64) :sum ,with-arg))
                              ((equal in-et '(signed-byte 32)) (define-loop single-float (signed-byte 32) :sum ,with-arg))
                              (t (define-loop single-float nil :sum ,with-arg))))
                       ((equal out-et '(signed-byte 64))
                        (cond ((equal in-et '(signed-byte 64)) (define-loop (signed-byte 64) (signed-byte 64) :sum ,with-arg))
                              ((equal in-et '(signed-byte 32)) (define-loop (signed-byte 64) (signed-byte 32) :sum ,with-arg))
                              ((equal in-et 'double-float)     (define-loop (signed-byte 64) double-float     :sum ,with-arg))
                              ((equal in-et 'single-float)     (define-loop (signed-byte 64) single-float     :sum ,with-arg))
                              (t (define-loop (signed-byte 64) nil :sum ,with-arg))))
                       ((equal out-et '(signed-byte 32))
                        (cond ((equal in-et '(signed-byte 32)) (define-loop (signed-byte 32) (signed-byte 32) :sum ,with-arg))
                              ((equal in-et '(signed-byte 64)) (define-loop (signed-byte 32) (signed-byte 64) :sum ,with-arg))
                              ((equal in-et 'double-float)     (define-loop (signed-byte 32) double-float     :sum ,with-arg))
                              ((equal in-et 'single-float)     (define-loop (signed-byte 32) single-float     :sum ,with-arg))
                              (t (define-loop (signed-byte 32) nil :sum ,with-arg))))
                       (t (define-loop nil nil :sum ,with-arg))))
                    ((eq reducer-fn #'max)
                     (cond
                       ((equal out-et 'double-float)
                        (cond ((equal in-et 'double-float) (define-loop double-float double-float :max ,with-arg))
                              ((equal in-et 'single-float) (define-loop double-float single-float :max ,with-arg))
                              ((equal in-et '(signed-byte 64)) (define-loop double-float (signed-byte 64) :max ,with-arg))
                              ((equal in-et '(signed-byte 32)) (define-loop double-float (signed-byte 32) :max ,with-arg))
                              (t (define-loop double-float nil :max ,with-arg))))
                       ((equal out-et 'single-float)
                        (cond ((equal in-et 'single-float) (define-loop single-float single-float :max ,with-arg))
                              ((equal in-et 'double-float) (define-loop single-float double-float :max ,with-arg))
                              ((equal in-et '(signed-byte 64)) (define-loop single-float (signed-byte 64) :max ,with-arg))
                              ((equal in-et '(signed-byte 32)) (define-loop single-float (signed-byte 32) :max ,with-arg))
                              (t (define-loop single-float nil :max ,with-arg))))
                       ((equal out-et '(signed-byte 64))
                        (cond ((equal in-et '(signed-byte 64)) (define-loop (signed-byte 64) (signed-byte 64) :max ,with-arg))
                              ((equal in-et '(signed-byte 32)) (define-loop (signed-byte 64) (signed-byte 32) :max ,with-arg))
                              ((equal in-et 'double-float)     (define-loop (signed-byte 64) double-float     :max ,with-arg))
                              ((equal in-et 'single-float)     (define-loop (signed-byte 64) single-float     :max ,with-arg))
                              (t (define-loop (signed-byte 64) nil :max ,with-arg))))
                       ((equal out-et '(signed-byte 32))
                        (cond ((equal in-et '(signed-byte 32)) (define-loop (signed-byte 32) (signed-byte 32) :max ,with-arg))
                              ((equal in-et '(signed-byte 64)) (define-loop (signed-byte 32) (signed-byte 64) :max ,with-arg))
                              ((equal in-et 'double-float)     (define-loop (signed-byte 32) double-float     :max ,with-arg))
                              ((equal in-et 'single-float)     (define-loop (signed-byte 32) single-float     :max ,with-arg))
                              (t (define-loop (signed-byte 32) nil :max ,with-arg))))
                       (t (define-loop nil nil :max ,with-arg))))
                    ((eq reducer-fn #'min)
                     (cond
                       ((equal out-et 'double-float)
                        (cond ((equal in-et 'double-float) (define-loop double-float double-float :min ,with-arg))
                              ((equal in-et 'single-float) (define-loop double-float single-float :min ,with-arg))
                              ((equal in-et '(signed-byte 64)) (define-loop double-float (signed-byte 64) :min ,with-arg))
                              ((equal in-et '(signed-byte 32)) (define-loop double-float (signed-byte 32) :min ,with-arg))
                              (t (define-loop double-float nil :min ,with-arg))))
                       ((equal out-et 'single-float)
                        (cond ((equal in-et 'single-float) (define-loop single-float single-float :min ,with-arg))
                              ((equal in-et 'double-float) (define-loop single-float double-float :min ,with-arg))
                              ((equal in-et '(signed-byte 64)) (define-loop single-float (signed-byte 64) :min ,with-arg))
                              ((equal in-et '(signed-byte 32)) (define-loop single-float (signed-byte 32) :min ,with-arg))
                              (t (define-loop single-float nil :min ,with-arg))))
                       ((equal out-et '(signed-byte 64))
                        (cond ((equal in-et '(signed-byte 64)) (define-loop (signed-byte 64) (signed-byte 64) :min ,with-arg))
                              ((equal in-et '(signed-byte 32)) (define-loop (signed-byte 64) (signed-byte 32) :min ,with-arg))
                              ((equal in-et 'double-float)     (define-loop (signed-byte 64) double-float     :min ,with-arg))
                              ((equal in-et 'single-float)     (define-loop (signed-byte 64) single-float     :min ,with-arg))
                              (t (define-loop (signed-byte 64) nil :min ,with-arg))))
                       ((equal out-et '(signed-byte 32))
                        (cond ((equal in-et '(signed-byte 32)) (define-loop (signed-byte 32) (signed-byte 32) :min ,with-arg))
                              ((equal in-et '(signed-byte 64)) (define-loop (signed-byte 32) (signed-byte 64) :min ,with-arg))
                              ((equal in-et 'double-float)     (define-loop (signed-byte 32) double-float     :min ,with-arg))
                              ((equal in-et 'single-float)     (define-loop (signed-byte 32) single-float     :min ,with-arg))
                              (t (define-loop (signed-byte 32) nil :min ,with-arg))))
                       (t (define-loop nil nil :min ,with-arg))))
                    (t
                     (cond
                       ((equal out-et 'double-float)
                        (cond ((equal in-et 'double-float) (define-loop double-float double-float :custom ,with-arg))
                              ((equal in-et 'single-float) (define-loop double-float single-float :custom ,with-arg))
                              ((equal in-et '(signed-byte 64)) (define-loop double-float (signed-byte 64) :custom ,with-arg))
                              ((equal in-et '(signed-byte 32)) (define-loop double-float (signed-byte 32) :custom ,with-arg))
                              (t (define-loop double-float nil :custom ,with-arg))))
                       ((equal out-et 'single-float)
                        (cond ((equal in-et 'single-float) (define-loop single-float single-float :custom ,with-arg))
                              ((equal in-et 'double-float) (define-loop single-float double-float :custom ,with-arg))
                              ((equal in-et '(signed-byte 64)) (define-loop single-float (signed-byte 64) :custom ,with-arg))
                              ((equal in-et '(signed-byte 32)) (define-loop single-float (signed-byte 32) :custom ,with-arg))
                              (t (define-loop single-float nil :custom ,with-arg))))
                       ((equal out-et '(signed-byte 64))
                        (cond ((equal in-et '(signed-byte 64)) (define-loop (signed-byte 64) (signed-byte 64) :custom ,with-arg))
                              ((equal in-et '(signed-byte 32)) (define-loop (signed-byte 64) (signed-byte 32) :custom ,with-arg))
                              ((equal in-et 'double-float)     (define-loop (signed-byte 64) double-float     :custom ,with-arg))
                              ((equal in-et 'single-float)     (define-loop (signed-byte 64) single-float     :custom ,with-arg))
                              (t (define-loop (signed-byte 64) nil :custom ,with-arg))))
                       ((equal out-et '(signed-byte 32))
                        (cond ((equal in-et '(signed-byte 32)) (define-loop (signed-byte 32) (signed-byte 32) :custom ,with-arg))
                              ((equal in-et '(signed-byte 64)) (define-loop (signed-byte 32) (signed-byte 64) :custom ,with-arg))
                              ((equal in-et 'double-float)     (define-loop (signed-byte 32) double-float     :custom ,with-arg))
                              ((equal in-et 'single-float)     (define-loop (signed-byte 32) single-float     :custom ,with-arg))
                              (t (define-loop (signed-byte 32) nil :custom ,with-arg))))
                       (t (define-loop nil nil :custom ,with-arg)))))))
            (if return-arg
                (generate-typed-cond t)
                (generate-typed-cond nil)))
          (values res res-idx))))))
