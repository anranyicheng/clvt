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
         (in-strides (coerce (mapcar (lambda (in)
                                       (coerce (vt-broadcast-strides (vt-shape in) out-shape (vt-strides in))
                                               'simple-vector))
                                     inputs)
                             'simple-vector))
         (contig (and (vt-contiguous-p res)
                      (every (lambda (in)
                               (or (= (vt-size in) 1)
                                   (and (equal (vt-shape in) out-shape) (vt-contiguous-p in))))
                             inputs)))
         (all-same (every (lambda (in) (eq (vt-dtype in) res-dtype)) inputs)))
    (declare (type simple-vector res-strides-vec dims in-datas in-offsets in-strides)
             (type fixnum n rank size))

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
                   `(if ,(subtypep lt 'integer) (truncate ,form) (coerce ,form ',lt)))
                 ;; 为特定类型生成展开代码，避免复杂反引号嵌套
                 (gen (lt)
                   `(let ((od (the (simple-array ,lt (*)) res-data)))
                      (cond
                        ;; 连续且类型相同，n=1
                        ((and contig all-same (= n 1))
                         (let* ((d0 (the (simple-array ,lt (*)) (svref in-datas 0)))
                                (p0 (svref in-offsets 0))
                                (s0 (if (= (vt-size (first inputs)) 1) 0 1))
                                (op (vt-offset res)))
                           (declare (type (simple-array ,lt (*)) d0)
                                    (type fixnum p0 s0 op))
                           (dotimes (i size)
                             (let ((v (funcall fn (aref d0 p0))))
                               (setf (aref od op) (cast-to ,lt v)))
                             (incf op)
                             (incf p0 s0))))
                        ;; 连续且类型相同，n=2
                        ((and contig all-same (= n 2))
                         (let* ((d0 (the (simple-array ,lt (*)) (svref in-datas 0)))
                                (d1 (the (simple-array ,lt (*)) (svref in-datas 1)))
                                (p0 (svref in-offsets 0))
                                (p1 (svref in-offsets 1))
                                (s0 (if (= (vt-size (first inputs)) 1) 0 1))
                                (s1 (if (= (vt-size (second inputs)) 1) 0 1))
                                (op (vt-offset res)))
                           (declare (type (simple-array ,lt (*)) d0 d1)
                                    (type fixnum p0 p1 s0 s1 op))
                           (dotimes (i size)
                             (let ((v (funcall fn (aref d0 p0) (aref d1 p1))))
                               (setf (aref od op) (cast-to ,lt v)))
                             (incf op)
                             (incf p0 s0)
                             (incf p1 s1))))
                        ;; 连续且类型相同，n=3
                        ((and contig all-same (= n 3))
                         (let* ((d0 (the (simple-array ,lt (*)) (svref in-datas 0)))
                                (d1 (the (simple-array ,lt (*)) (svref in-datas 1)))
                                (d2 (the (simple-array ,lt (*)) (svref in-datas 2)))
                                (p0 (svref in-offsets 0))
                                (p1 (svref in-offsets 1))
                                (p2 (svref in-offsets 2))
                                (s0 (if (= (vt-size (first inputs)) 1) 0 1))
                                (s1 (if (= (vt-size (second inputs)) 1) 0 1))
                                (s2 (if (= (vt-size (third inputs)) 1) 0 1))
                                (op (vt-offset res)))
                           (declare (type (simple-array ,lt (*)) d0 d1 d2)
                                    (type fixnum p0 p1 p2 s0 s1 s2 op))
                           (dotimes (i size)
                             (let ((v (funcall fn (aref d0 p0) (aref d1 p1) (aref d2 p2))))
                               (setf (aref od op) (cast-to ,lt v)))
                             (incf op)
                             (incf p0 s0)
                             (incf p1 s1)
                             (incf p2 s2))))
                        ;; 连续且类型相同，n=4
                        ((and contig all-same (= n 4))
                         (let* ((d0 (the (simple-array ,lt (*)) (svref in-datas 0)))
                                (d1 (the (simple-array ,lt (*)) (svref in-datas 1)))
                                (d2 (the (simple-array ,lt (*)) (svref in-datas 2)))
                                (d3 (the (simple-array ,lt (*)) (svref in-datas 3)))
                                (p0 (svref in-offsets 0))
                                (p1 (svref in-offsets 1))
                                (p2 (svref in-offsets 2))
                                (p3 (svref in-offsets 3))
                                (s0 (if (= (vt-size (first inputs)) 1) 0 1))
                                (s1 (if (= (vt-size (second inputs)) 1) 0 1))
                                (s2 (if (= (vt-size (third inputs)) 1) 0 1))
                                (s3 (if (= (vt-size (fourth inputs)) 1) 0 1))
                                (op (vt-offset res)))
                           (declare (type (simple-array ,lt (*)) d0 d1 d2 d3)
                                    (type fixnum p0 p1 p2 p3 s0 s1 s2 s3 op))
                           (dotimes (i size)
                             (let ((v (funcall fn (aref d0 p0)
					       (aref d1 p1)
					       (aref d2 p2)
					       (aref d3 p3))))
                               (setf (aref od op) (cast-to ,lt v)))
                             (incf op)
                             (incf p0 s0)
                             (incf p1 s1)
                             (incf p2 s2)
                             (incf p3 s3))))
                        ;; 连续且类型相同，n=5
                        ((and contig all-same (= n 5))
                         (let* ((d0 (the (simple-array ,lt (*)) (svref in-datas 0)))
                                (d1 (the (simple-array ,lt (*)) (svref in-datas 1)))
                                (d2 (the (simple-array ,lt (*)) (svref in-datas 2)))
                                (d3 (the (simple-array ,lt (*)) (svref in-datas 3)))
                                (d4 (the (simple-array ,lt (*)) (svref in-datas 4)))
                                (p0 (svref in-offsets 0))
                                (p1 (svref in-offsets 1))
                                (p2 (svref in-offsets 2))
                                (p3 (svref in-offsets 3))
                                (p4 (svref in-offsets 4))
                                (s0 (if (= (vt-size (first inputs)) 1) 0 1))
                                (s1 (if (= (vt-size (second inputs)) 1) 0 1))
                                (s2 (if (= (vt-size (third inputs)) 1) 0 1))
                                (s3 (if (= (vt-size (fourth inputs)) 1) 0 1))
                                (s4 (if (= (vt-size (fifth inputs)) 1) 0 1))
                                (op (vt-offset res)))
                           (declare (type (simple-array ,lt (*)) d0 d1 d2 d3 d4)
                                    (type fixnum p0 p1 p2 p3 p4 s0 s1 s2 s3 s4 op))
                           (dotimes (i size)
                             (let ((v (funcall fn (aref d0 p0)
					       (aref d1 p1)
					       (aref d2 p2)
					       (aref d3 p3)
					       (aref d4 p4))))
                               (setf (aref od op) (cast-to ,lt v)))
                             (incf op)
                             (incf p0 s0)
                             (incf p1 s1)
                             (incf p2 s2)
                             (incf p3 s3)
                             (incf p4 s4))))
                        ;; 连续但类型不同或 n>5，使用通用路径
                        (contig
                         (let ((ptrs (copy-seq in-offsets))
                               (vals (make-array n))
                               (steps (coerce (mapcar (lambda (in)
                                                        (if (= (vt-size in) 1) 0 1))
                                                      inputs)
                                              'simple-vector))
                               (op (vt-offset res)))
                           (declare (type simple-vector ptrs vals steps)
                                    (type fixnum op))
                           (dotimes (i size)
                             (gather ptrs vals)
                             (setf (aref od op) (cast-to ,lt (call-fn vals)))
                             (incf op)
                             (loop for k fixnum from 0 below n do
                               (incf (aref ptrs k) (aref steps k))))))
                        ;; 非连续路径
                        (t
                         (let ((ptrs (copy-seq in-offsets))
                               (vals (make-array n))
                               (indices (make-array rank :element-type 'fixnum :initial-element 0))
                               (op (vt-offset res)))
                           (declare (type simple-vector ptrs vals)
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
                                            (decf d))))))))))))
        (cond ((equal (array-element-type res-data) 'double-float) (gen double-float))
              ((equal (array-element-type res-data) 'single-float) (gen single-float))
              ((equal (array-element-type res-data) '(signed-byte 64)) (gen (signed-byte 64)))
              ((equal (array-element-type res-data) '(signed-byte 32)) (gen (signed-byte 32)))
              (t
               ;; 非特化类型，保持原有通用循环
               (let ((ptrs (copy-seq in-offsets))
                     (vals (make-array n))
                     (indices (make-array rank :element-type 'fixnum :initial-element 0))
                     (op (vt-offset res)))
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
  "通用归约核心。axis 可为 nil/整数/整数列表。
   reducer-fn 接收 (当前累积值, 当前元素值)，返回 (新累积值, 是否更新arg索引)。
   返回 (values result arg-result)。"
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

      ;; 空输入处理
      (when (or (zerop axis-size) (zerop (vt-size tensor)))
        (let ((empty-dtype (or dtype (and out (vt-dtype out)) (vt-dtype tensor))))
          (return-from vt-reduce
            (values (make-vt out-shape (or init-val 0) :dtype empty-dtype)
                    (when return-arg (make-vt out-shape 0 :dtype :int32))))))

      ;; ================================================================
      ;; 快速路径：连续张量 + 全局归约 (axis=nil, 不keepdims, 不需要arg)
      ;; ================================================================
      (when (and global (not keepdims) (not return-arg)
                 (vt-contiguous-p tensor))
        (let* ((final-dtype (cond ((and out dtype (not (eq (vt-dtype out) dtype)))
                                   (error "vt-reduce: :out type (~a) 与 :dtype (~a) 冲突"
                                          (vt-dtype out) dtype))
                                  (out (vt-dtype out))
                                  (dtype dtype)
                                  ((and init-val (or (floatp init-val) (%inf-p init-val))) :float64)
                                  (t (vt-dtype tensor))))
               (size (vt-size tensor))
               (in-data (vt-data tensor))
               (in-off (vt-offset tensor))
               (in-et (array-element-type in-data))
               (out-lisp-type (vt-dtype->lisp-type final-dtype))
               (zero-val (coerce 0 out-lisp-type)))
          (labels ((make-result-fast (val)
                     (let ((res (or out (make-vt nil zero-val :dtype final-dtype))))
                       (setf (aref (the (simple-array * (*)) (vt-data res)) (vt-offset res))
                             (vt-cast val final-dtype))
                       (return-from vt-reduce (values res nil)))))
            (cond
              ;; double-float 输入
              ((equal in-et 'double-float)
               (let ((d (the (simple-array double-float (*)) in-data)))
                 (declare (type (simple-array double-float (*)) d))
                 (cond
                   ((and (typep init-val 'double-float) (zerop init-val))
                    (let ((acc 0.0d0))
                      (declare (type double-float acc))
                      (loop for i fixnum from 0 below size do
                        (incf acc (aref d (+ in-off i))))
                      (make-result-fast acc)))
                   ((= init-val +vt-dfloat-neg-inf+)
                    (let ((acc +vt-dfloat-neg-inf+))
                      (declare (type double-float acc))
                      (loop for i fixnum from 0 below size do
                        (let ((v (aref d (+ in-off i))))
                          (when (> v acc) (setf acc v))))
                      (make-result-fast acc)))
                   ((= init-val +vt-dfloat-pos-inf+)
                    (let ((acc +vt-dfloat-pos-inf+))
                      (declare (type double-float acc))
                      (loop for i fixnum from 0 below size do
                        (let ((v (aref d (+ in-off i))))
                          (when (< v acc) (setf acc v))))
                      (make-result-fast acc)))
                   (t nil))))
              ;; single-float 输入
              ((equal in-et 'single-float)
               (let ((d (the (simple-array single-float (*)) in-data)))
                 (declare (type (simple-array single-float (*)) d))
                 (cond
                   ((and (typep init-val 'single-float) (zerop init-val))
                    (let ((acc 0.0s0))
                      (declare (type single-float acc))
                      (loop for i fixnum from 0 below size do (incf acc (aref d (+ in-off i))))
                      (make-result-fast acc))))))
              ;; int32 输入 (sum)
              ((equal in-et '(signed-byte 32))
               (when (eql init-val 0)
                 (let ((d (the (simple-array (signed-byte 32) (*)) in-data))
                       (acc 0))
                   (declare (type (simple-array (signed-byte 32) (*)) d)
                            (type (signed-byte 32) acc))
                   (loop for i fixnum from 0 below size do (incf acc (aref d (+ in-off i))))
                   (make-result-fast acc))))
              ;; int64 输入 (sum)
              ((equal in-et '(signed-byte 64))
               (when (eql init-val 0)
                 (let ((d (the (simple-array (signed-byte 64) (*)) in-data))
                       (acc 0))
                   (declare (type (simple-array (signed-byte 64) (*)) d)
                            (type (signed-byte 64) acc))
                   (loop for i fixnum from 0 below size do (incf acc (aref d (+ in-off i))))
                   (make-result-fast acc))))))))

      ;; ================================================================
      ;; 通用路径：任意轴归约、keepdims、return-arg
      ;; 优化：增量指针更新 + 内联常见 reducer + 消除运行时分支
      ;; ================================================================
      (let* ((final-dtype (cond
                            ((and out dtype (not (eq (vt-dtype out) dtype)))
                             (error "vt-reduce: :out type (~a) 与 :dtype (~a) 冲突"
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
             (res-idx-offset (when res-idx (vt-offset res-idx)))
             (res-idx-strides (when res-idx (vt-strides res-idx)))
             (in-data (vt-data tensor))
             (in-strides (vt-strides tensor))
             (in-offset (vt-offset tensor))
             (in-shape-vec (coerce in-shape 'simple-vector))
             (in-strides-vec (coerce in-strides 'simple-vector))
             ;; 预计算输出和arg的步长映射
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
             ;; 归约轴步长（用于计算arg索引）
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
        
        ;; 初始化结果
        (vt-fill res init-val)
        (when res-idx (vt-fill res-idx 0))

        ;; 标量特殊处理（rank = 0）
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

        ;; 宏定义：根据 out-lt, in-lt, red-type, with-arg 生成循环
        (macrolet
            ((cast-to (lt form)
               (cond
                 ((null lt) form)  ; 无类型声明，直接使用原值
                 ((subtypep lt 'integer)
                  `(truncate ,form))
                 (t
                  `(coerce ,form ',lt))))
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
                          ,@(when with-arg
                              `((arg-ptr res-idx-offset)
                                (arg-val 0))))
                      (declare (type fixnum in-ptr out-ptr
                                     ,@(when with-arg '(arg-ptr arg-val))))
                      (block outer-loop
                        (loop
                          ;; --- 处理当前元素 ---
                          (let* ((val (aref ,in-array in-ptr))
                                 (raw-acc (aref ,out-array out-ptr)))
                            ,(ecase red-type
                               (:sum
                                `(setf (aref ,out-array out-ptr)
                                       (cast-to ,out-lt (+ raw-acc val))))
                               (:max
                                `(when (> val raw-acc)
                                   (setf (aref ,out-array out-ptr)
                                         (cast-to ,out-lt val))
                                   ,(when with-arg
                                      `(setf (aref res-idx-data arg-ptr) arg-val))))
                               (:min
                                `(when (< val raw-acc)
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
                          
                          ;; --- 递增索引并更新指针 ---
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
                              ;; 进位：当前维度归零，回退指针
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
             ;; 新增：生成类型选择的 cond 表达式
             (generate-typed-cond (with-arg)
               `(cond
                  ;; SUM
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
                            (t (define-loop (signed-byte 64) nil :sum ,with-arg))))
                     ((equal out-et '(signed-byte 32))
                      (cond ((equal in-et '(signed-byte 32)) (define-loop (signed-byte 32) (signed-byte 32) :sum ,with-arg))
                            (t (define-loop (signed-byte 32) nil :sum ,with-arg))))
                     (t (define-loop nil nil :sum ,with-arg))))
                  ;; MAX
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
                            (t (define-loop (signed-byte 64) nil :max ,with-arg))))
                     ((equal out-et '(signed-byte 32))
                      (cond ((equal in-et '(signed-byte 32)) (define-loop (signed-byte 32) (signed-byte 32) :max ,with-arg))
                            (t (define-loop (signed-byte 32) nil :max ,with-arg))))
                     (t (define-loop nil nil :max ,with-arg))))
                  ;; MIN
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
                            (t (define-loop (signed-byte 64) nil :min ,with-arg))))
                     ((equal out-et '(signed-byte 32))
                      (cond ((equal in-et '(signed-byte 32)) (define-loop (signed-byte 32) (signed-byte 32) :min ,with-arg))
                            (t (define-loop (signed-byte 32) nil :min ,with-arg))))
                     (t (define-loop nil nil :min ,with-arg))))
                  ;; CUSTOM
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
                            (t (define-loop (signed-byte 64) nil :custom ,with-arg))))
                     ((equal out-et '(signed-byte 32))
                      (cond ((equal in-et '(signed-byte 32)) (define-loop (signed-byte 32) (signed-byte 32) :custom ,with-arg))
                            (t (define-loop (signed-byte 32) nil :custom ,with-arg))))
                     (t (define-loop nil nil :custom ,with-arg)))))))
          
          ;; 根据 return-arg 选择展开
          (if return-arg
              (generate-typed-cond t)
              (generate-typed-cond nil)))
        
        (values res res-idx)))))
