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
           (res (or out (make-vt out-shape 0 :dtype final-dtype))))
      (when out
        (unless (equal (vt-shape res) out-shape)
          (error "vt-map: :out 形状 ~a 与广播结果 ~a 不匹配" (vt-shape res) out-shape)))
      (%vt-map-run fn inputs res out-shape)
      res))))

(defun %vt-map-run (fn inputs res out-shape)
  (let* ((n (length inputs))
         (res-data (vt-data res))
         (res-dtype (vt-dtype res))
         (res-strides-vec (coerce (vt-strides res) 'simple-vector))
         (dims (coerce out-shape 'simple-vector))
         (rank (length out-shape))
         (size (vt-shape-to-size out-shape))
         (in-datas (coerce (mapcar #'vt-data inputs) 'simple-vector))
         (in-offsets (coerce (mapcar #'vt-offset inputs) 'simple-vector))
         (in-strides (coerce (mapcar (lambda (in)
                                       (coerce (vt-broadcast-strides (vt-shape in) out-shape (vt-strides in))
                                               'simple-vector))
                                     inputs)
                             'simple-vector))
         (vals (make-array n))
         (contig (and (vt-contiguous-p res)
                      (every (lambda (in)
                               (or (= (vt-size in) 1)
                                   (and (equal (vt-shape in) out-shape) (vt-contiguous-p in))))
                             inputs)))
         (all-same (every (lambda (in) (eq (vt-dtype in) res-dtype)) inputs)))
    (declare (type simple-vector res-strides-vec dims in-datas in-offsets in-strides vals)
             (type fixnum n rank size))
    (labels ((gather (ptrs)
               (loop for k fixnum from 0 below n
                     do (setf (aref vals k) (aref (svref in-datas k) (aref ptrs k)))))
             (call-fn ()
               (case n
                 (1 (funcall fn (aref vals 0)))
                 (2 (funcall fn (aref vals 0) (aref vals 1)))
                 (3 (funcall fn (aref vals 0) (aref vals 1) (aref vals 2)))
                 (4 (funcall fn (aref vals 0) (aref vals 1) (aref vals 2) (aref vals 3)))
                 (5 (funcall fn (aref vals 0) (aref vals 1) (aref vals 2) (aref vals 3) (aref vals 4)))
                 (otherwise (apply fn (coerce vals 'list))))))
      (macrolet ((cast-to (lt form)
                   `(if ,(subtypep lt 'integer) (truncate ,form) (coerce ,form ',lt)))
                 (gen (lt)
                   `(let ((od (the (simple-array ,lt (*)) res-data)))
                      (cond
                        ((and contig all-same (= n 1))
                         (let ((d0 (the (simple-array ,lt (*)) (svref in-datas 0)))
                               (p0 (svref in-offsets 0))
                               (s0 (if (= (vt-size (first inputs)) 1) 0 1))
                               (op (vt-offset res)))
                           (declare (type (simple-array ,lt (*)) d0) (type fixnum p0 s0 op))
                           (dotimes (i size)
                             (let ((v (funcall fn (aref d0 p0))))
                               (setf (aref od op) (cast-to ,lt v)))
                             (incf op)
                             (incf p0 s0))))
                        ((and contig all-same (= n 2))
                         (let ((d0 (the (simple-array ,lt (*)) (svref in-datas 0)))
                               (d1 (the (simple-array ,lt (*)) (svref in-datas 1)))
                               (p0 (svref in-offsets 0)) (p1 (svref in-offsets 1))
                               (s0 (if (= (vt-size (first inputs)) 1) 0 1))
                               (s1 (if (= (vt-size (second inputs)) 1) 0 1))
                               (op (vt-offset res)))
                           (declare (type (simple-array ,lt (*)) d0 d1) (type fixnum p0 p1 s0 s1 op))
                           (dotimes (i size)
                             (let ((v (funcall fn (aref d0 p0) (aref d1 p1))))
                               (setf (aref od op) (cast-to ,lt v)))
                             (incf op)
                             (incf p0 s0)
                             (incf p1 s1))))
                        (contig
                         (let ((ptrs (copy-seq in-offsets))
                               (steps (coerce (mapcar (lambda (in) (if (= (vt-size in) 1) 0 1)) inputs) 'simple-vector))
                               (op (vt-offset res)))
                           (declare (type simple-vector ptrs steps) (type fixnum op))
                           (dotimes (i size)
                             (gather ptrs)
                             (setf (aref od op) (cast-to ,lt (call-fn)))
                             (incf op)
                             (loop for k fixnum from 0 below n do
                               (incf (aref ptrs k) (aref steps k))))))
                        (t
                         (let ((ptrs (copy-seq in-offsets))
                               (indices (make-array rank :element-type 'fixnum :initial-element 0))
                               (op (vt-offset res)))
                           (declare (type simple-vector ptrs)
                                    (type (simple-array fixnum (*)) indices)
                                    (type fixnum op))
                           (dotimes (i size)
                             (gather ptrs)
                             (setf (aref od op) (cast-to ,lt (call-fn)))
                             (let ((d (1- rank)))
                               (loop
                                 (when (< d 0) (return))
                                 (incf (aref indices d))
                                 (if (< (aref indices d) (svref dims d))
                                     (progn (incf op (svref res-strides-vec d))
                                            (loop for k fixnum from 0 below n do
                                              (incf (aref ptrs k) (aref (svref in-strides k) d)))
                                            (return))
                                     (progn (setf (aref indices d) 0)
                                            (decf op (* (svref res-strides-vec d) (1- (svref dims d))))
                                            (loop for k fixnum from 0 below n do
                                              (decf (aref ptrs k) (* (aref (svref in-strides k) d) (1- (svref dims d)))))
                                            (decf d))))))))))))
        (cond ((equal (array-element-type res-data) 'double-float) (gen double-float))
              ((equal (array-element-type res-data) 'single-float) (gen single-float))
              ((equal (array-element-type res-data) '(signed-byte 64)) (gen (signed-byte 64)))
              ((equal (array-element-type res-data) '(signed-byte 32)) (gen (signed-byte 32)))
              (t
               (let ((ptrs (copy-seq in-offsets)) (op (vt-offset res)))
                 (declare (type simple-vector ptrs) (type fixnum op))
                 (dotimes (i size)
                   (gather ptrs)
                   (setf (aref res-data op) (call-fn))
                   (incf op)
                   (loop for k fixnum from 0 below n do (incf (aref ptrs k) 1))))))))
    res))

;;; ------------------------------------------------------------------
;;; 二元特化（内部，供算术层使用）
;;; ------------------------------------------------------------------

(defun vt-binary (fn t1 t2 &key out dtype)
  "二元逐元素运算（内部入口，等价于两参数 vt-map）。"
  (apply #'vt-map fn (ensure-vt t1) (ensure-vt t2) :out out :dtype dtype))

;;; ------------------------------------------------------------------
;;; 编译期内联逐元素映射（对标原 vt-fast-map，消除 funcall 装箱）
;;; ------------------------------------------------------------------

(defmacro %cast-to (lt form)
  "内联类型转换：整数截断，浮点 coerce。"
  `(if ,(subtypep lt 'integer) (truncate ,form) (coerce ,form ',lt)))

(defmacro %inline1-loop (lt op a res)
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
  "一元连续快路径（按输出类型派发，内联 op）。"
  `(let ((rd (vt-data ,res)))
     (cond ((equal (array-element-type rd) 'double-float) (%inline1-loop double-float ,op ,a ,res))
           ((equal (array-element-type rd) 'single-float) (%inline1-loop single-float ,op ,a ,res))
           ((equal (array-element-type rd) '(signed-byte 64)) (%inline1-loop (signed-byte 64) ,op ,a ,res))
           ((equal (array-element-type rd) '(signed-byte 32)) (%inline1-loop (signed-byte 32) ,op ,a ,res))
           (t (vt-map (function ,op) ,a :out ,res)))))

(defmacro %vt-inline2-fast (op a b res)
  "二元连续快路径（按输出类型派发，内联 op）。"
  `(let ((rd (vt-data ,res)))
     (cond ((equal (array-element-type rd) 'double-float) (%inline2-loop double-float ,op ,a ,b ,res))
           ((equal (array-element-type rd) 'single-float) (%inline2-loop single-float ,op ,a ,b ,res))
           ((equal (array-element-type rd) '(signed-byte 64)) (%inline2-loop (signed-byte 64) ,op ,a ,b ,res))
           ((equal (array-element-type rd) '(signed-byte 32)) (%inline2-loop (signed-byte 32) ,op ,a ,b ,res))
           (t (vt-map (function ,op) ,a ,b :out ,res)))))

(defmacro vt-fast-map (fn &rest args)
  "编译期内联已知算子的逐元素映射（一元/二元）；否则回退到 vt-map。"
  (let ((op (and (consp fn) (eq (car fn) 'function) (symbolp (cadr fn)) (cadr fn))))
    (if (null op)
        `(apply #'vt-map ,fn ,@args)
        (multiple-value-bind (tensors dtype out) (parse-vt-op-args args)
          (let* ((n (length tensors))
                 (tvs (loop repeat n collect (gensym "TV"))))
            `(let (,@(loop for tv in tvs for tf in tensors collect `(,tv (ensure-vt ,tf))))
               (let* ((out-shape (reduce #'vt-broadcast-shapes (mapcar #'vt-shape (list ,@tvs))))
                      (final-dtype (cond ((and ,out ,dtype (not (eq (vt-dtype ,out) ,dtype)))
                                          (error "类型冲突: :out (~a) vs :dtype (~a)" (vt-dtype ,out) ,dtype))
                                         (,out (vt-dtype ,out))
                                         (,dtype ,dtype)
                                         (t (apply #'vt-promote-type (mapcar #'vt-dtype (list ,@tvs))))))
                      (res (or ,out (make-vt out-shape 0 :dtype final-dtype))))
                 (when ,out
                   (unless (equal (vt-shape res) out-shape)
                     (error ":out 形状 ~a 与广播结果 ~a 不匹配" (vt-shape res) out-shape)))
                 ,(case n
                    (1 `(if (and (vt-contiguous-p res)
                                 (or (= (vt-size ,(first tvs)) 1)
                                     (and (equal (vt-shape ,(first tvs)) out-shape)
                                          (vt-contiguous-p ,(first tvs))))
                                 (eq (vt-dtype ,(first tvs)) (vt-dtype res)))
                            (%vt-inline1-fast ,op ,(first tvs) res)
                            (vt-map (function ,op) ,(first tvs) :out res)))
                    (2 `(if (and (vt-contiguous-p res)
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
                    (t `(apply #'vt-map ,fn ,@args)))
                 res)))))))

;;; ------------------------------------------------------------------
;;; 归约核心：vt-reduce
;;; ------------------------------------------------------------------

(defun get-reduction-identity (op element-type)
  "返回指定归约操作在给定元素类型下的初始值。"
  (case op
    (:sum (coerce 0 element-type))
    (:max (cond ((eq element-type 'double-float) +vt-dfloat-neg-inf+)
                ((eq element-type 'single-float) +vt-sfloat-neg-inf+)
                ((equal element-type '(signed-byte 64)) (- (expt 2 63)))
                ((equal element-type '(signed-byte 32)) (- (expt 2 31)))
                ((subtypep element-type 'integer) most-negative-fixnum)
                (t 0)))
    (:min (cond ((eq element-type 'double-float) +vt-dfloat-pos-inf+)
                ((eq element-type 'single-float) +vt-sfloat-pos-inf+)
                ((equal element-type '(signed-byte 64)) (1- (expt 2 63)))
                ((equal element-type '(signed-byte 32)) (1- (expt 2 31)))
                ((subtypep element-type 'integer) most-positive-fixnum)
                (t 0)))))

(defun vt-reduce (tensor axis init-val reducer-fn &key out dtype keepdims return-arg)
  "通用归约核心。axis 可为 nil/整数/整数列表。
   reducer-fn 接收 (acc val)，返回 (values new-acc update-arg-p)。
   Returns: (values result arg-result)"
  (declare (type vt tensor)
           (type (or null fixnum list) axis)
           (type function reducer-fn)
           (optimize (speed 3) (safety 0)))
  (with-float-safe
    (let* ((in-shape (vt-shape tensor))
           (rank (length in-shape))
           (axes (vt-normalize-axes axis rank))
           (global (null axes))
           (out-shape (cond ((and global (not keepdims)) nil)
                            (global (make-list rank :initial-element 1))
                            ((not keepdims)
                             (loop for d in in-shape for i from 0
                                   unless (member i axes) collect d))
                            (t (loop for d in in-shape for i from 0
                                     collect (if (member i axes) 1 d)))))
           (axis-size (if axes
                          (reduce #'* (mapcar (lambda (a) (nth a in-shape)) axes)
                                  :initial-value 1)
                          (reduce #'* in-shape :initial-value 1))))
      (when (or (zerop axis-size) (zerop (vt-size tensor)))
        (let ((empty-dtype (or dtype (and out (vt-dtype out)) (vt-dtype tensor))))
          (return-from vt-reduce
            (values (make-vt out-shape (or init-val 0) :dtype empty-dtype)
                    (when return-arg (make-vt out-shape 0 :dtype :int32))))))
      (let* ((final-dtype (cond
                            ((and out dtype (not (eq (vt-dtype out) dtype)))
                             (error "vt-reduce: :out 类型 (~a) 与 :dtype (~a) 冲突"
                                    (vt-dtype out) dtype))
                            (out (vt-dtype out))
                            (dtype dtype)
                            ((and init-val (or (floatp init-val) (%inf-p init-val))) :float64)
                            (t (vt-dtype tensor))))
             (res (or out (make-vt out-shape 0 :dtype final-dtype)))
             (res-data (vt-data res))
             (res-offset (vt-offset res))
             (res-strides (vt-strides res))
             (res-idx (when return-arg (make-vt out-shape 0 :dtype :int32)))
             (res-idx-data (when res-idx (vt-data res-idx)))
             (in-data (vt-data tensor))
             (in-strides (vt-strides tensor))
             (in-offset (vt-offset tensor))
             (arg-strides
               (if return-arg
                   (if global
                       (vt-compute-strides in-shape)
                       (let* ((red-shape (mapcar (lambda (a) (nth a in-shape)) axes))
                              (red-strides (vt-compute-strides red-shape))
                              (k -1))
                         (loop for i below rank
                               if (member i axes)
                                 collect (progn (incf k) (nth k red-strides))
                               else collect 0)))
                   (make-list rank :initial-element 0)))
             (out-strides-map
               (if global
                   (make-list rank :initial-element 0)
                   (loop for i from 0 below rank
                         if (member i axes) collect 0
                           else collect
                                (let ((out-idx (if keepdims i
                                                   (count-if-not (lambda (x) (member x axes))
                                                                 (loop for j below i collect j)))))
                                  (nth out-idx res-strides))))))
        (vt-fill res init-val)
        (when res-idx (vt-fill res-idx 0))
        (let ((in-shp-vec (coerce in-shape 'simple-vector))
              (in-str-vec (coerce in-strides 'simple-vector))
              (osm-vec (coerce out-strides-map 'simple-vector))
              (arg-str-vec (coerce arg-strides 'simple-vector)))
          (labels ((recurse (depth in-ptr out-ptr arg-ptr arg-val)
                     (declare (type fixnum depth in-ptr out-ptr arg-ptr arg-val))
                     (if (= depth rank)
                         (let* ((val (aref in-data in-ptr))
                                (raw-acc (aref res-data out-ptr)))
                           (multiple-value-bind (new-acc do-update-arg)
                               (funcall reducer-fn raw-acc val)
                             (setf (aref res-data out-ptr) (vt-cast new-acc final-dtype))
                             (when (and return-arg do-update-arg res-idx-data)
                               (setf (aref res-idx-data arg-ptr) arg-val))))
                         (let* ((dim (svref in-shp-vec depth))
                                (in-stride (svref in-str-vec depth))
                                (out-stride (svref osm-vec depth))
                                (arg-stride (svref arg-str-vec depth)))
                           (declare (type fixnum dim in-stride out-stride arg-stride))
                           (loop for i fixnum from 0 below dim do
                             (recurse (1+ depth) in-ptr out-ptr arg-ptr
                                      (+ arg-val (* i arg-stride)))
                             (incf in-ptr in-stride)
                             (incf out-ptr out-stride)
                             (when return-arg (incf arg-ptr out-stride)))))))
            (recurse 0 in-offset res-offset (if res-idx (vt-offset res-idx) 0) 0)))
        (values res res-idx)))))
