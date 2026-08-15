;;;; creation.lisp — 张量创建

(in-package :clvt)

(defun vt-zeros (shape &key (dtype :float64))
  (make-vt shape 0 :dtype dtype))

(defun vt-ones (shape &key (dtype :float64))
  (make-vt shape 1 :dtype dtype))

(defun vt-const (shape value &key (dtype :float64))
  (make-vt shape value :dtype dtype))

(defun vt-full (shape fill-value &key (dtype :float64))
  (make-vt shape fill-value :dtype dtype))

(defun vt-empty (shape &key (dtype :float64))
  (vt-zeros shape :dtype dtype))

(defun vt-zeros-like (vt &key dtype)
  (vt-zeros (vt-shape vt) :dtype (or dtype (vt-dtype vt))))

(defun vt-ones-like (vt &key dtype)
  (vt-ones (vt-shape vt) :dtype (or dtype (vt-dtype vt))))

(defun vt-full-like (vt fill-value &key dtype)
  (vt-full (vt-shape vt) fill-value :dtype (or dtype (vt-dtype vt))))

(defun vt-empty-like (vt &key dtype)
  (vt-empty (vt-shape vt) :dtype (or dtype (vt-dtype vt))))

(defun vt-identity (n &key dtype)
  (vt-eye n :dtype dtype))

(defun vt-eye (rows &key (cols rows) (k 0) (value 1) (dtype :float64))
  "创建单位/对角矩阵，对标 NumPy np.eye。"
  (declare (type fixnum rows cols k))
  (let* ((shape (list rows cols))
         (lisp-type (vt-dtype->lisp-type dtype))
         (data (make-array (* rows cols) :element-type lisp-type
                                          :initial-element (coerce 0 lisp-type)))
         (res (%make-vt :data data :shape shape
                        :strides (vt-compute-strides shape)
                        :offset 0 :dtype dtype)))
    (let* ((row-stride (first (vt-strides res)))
           (col-stride (second (vt-strides res)))
           (r-start (max 0 (- k)))
           (c-start (max 0 k))
           (diag-len (max 0 (min (- rows r-start) (- cols c-start)))))
      (when (> diag-len 0)
        (let ((start-offset (+ (* r-start row-stride) (* c-start col-stride))))
          (loop for i fixnum from 0 below diag-len
                for offset fixnum = start-offset then (+ offset row-stride col-stride)
                do (setf (aref data offset) (vt-cast value dtype)))))
      res)))

(defun vt-arange (total-num &key (start 0) (step 1) (dtype :float64))
  "创建包含 total-num 个元素的等差数列一维张量。"
  (declare (fixnum total-num))
  (when (and (numberp step) (zerop step))
    (error "vt-arange: step 不能为 0"))
  (let* ((data (make-array total-num :element-type (vt-dtype->lisp-type dtype)))
         (shape (list total-num)))
    (loop for i fixnum below total-num
          do (setf (aref data i) (vt-cast (+ start (* i step)) dtype)))
    (%make-vt :data data :shape shape :strides (vt-compute-strides shape)
              :offset 0 :dtype dtype)))

(defun vt-linspace (start end num &key (endpoint t) (dtype :float64))
  "创建线性间隔数组，对标 numpy.linspace。"
  (when (<= num 0) (error "num 必须大于 0，当前值为 ~d" num))
  (when (= num 1)
    (return-from vt-linspace (make-vt (list 1) (vt-cast start dtype) :dtype dtype)))
  (let* ((div (if endpoint (1- num) num))
         (step (/ (- end start) div)))
    (let ((data (make-array num :element-type (vt-dtype->lisp-type dtype))))
      (loop for i fixnum from 0 below num
            do (setf (aref data i) (vt-cast (+ start (* i step)) dtype)))
      (%make-vt :data data :shape (list num) :strides '(1) :offset 0 :dtype dtype))))

(defun vt-logspace (start stop num &key (base 10.0d0) (endpoint t) (dtype :float64))
  "创建对数间隔的一维张量。"
  (vt-map (lambda (x) (expt base x))
          (vt-linspace start stop num :endpoint endpoint :dtype dtype)))

(defun vt-from-array (arr &key (dtype nil))
  "从标准 CL 多维数组创建张量（保持维度）。"
  (let* ((shape (array-dimensions arr))
         (size (vt-shape-to-size shape))
         (cl-etype (array-element-type arr))
         (infer (cond ((subtypep cl-etype 'double-float) :float64)
                      ((subtypep cl-etype 'single-float) :float32)
                      ((subtypep cl-etype '(signed-byte 32)) :int32)
                      ((subtypep cl-etype '(signed-byte 64)) :int64)
                      ((subtypep cl-etype 'fixnum) :int64)
                      (t :float64)))
         (final (or dtype infer))
         (data (make-array size :element-type (vt-dtype->lisp-type final))))
    (dotimes (i size)
      (setf (aref data i) (vt-cast (row-major-aref arr i) final)))
    (%make-vt :data data :shape shape :strides (vt-compute-strides shape)
              :offset 0 :dtype final)))

(defun vt-from-function (shape fn &key (dtype :float64))
  "根据函数创建张量：fn 接收索引列表并返回元素值。"
  (let* ((size (vt-shape-to-size shape))
         (data (make-array size :element-type (vt-dtype->lisp-type dtype)))
         (result (%make-vt :data data :shape shape
                           :strides (vt-compute-strides shape) :offset 0 :dtype dtype))
         (rank (length shape)))
    (labels ((recurse (depth indices flat-idx)
               (if (= depth rank)
                   (setf (aref data flat-idx) (vt-cast (funcall fn indices) dtype))
                   (let ((dim (nth depth shape)) (stride (nth depth (vt-strides result))))
                     (loop for i from 0 below dim
                           do (recurse (1+ depth) (append indices (list i))
                                       (+ flat-idx (* i stride))))))))
      (recurse 0 nil 0))
    result))

(defun vt-kron (a b)
  "Kronecker 积（对标 numpy.kron）。"
  (let ((a-shape (vt-shape a)) (b-shape (vt-shape b)))
    (when (null a-shape) (setf a-shape '(1)))
    (when (null b-shape) (setf b-shape '(1)))
    (let* ((nda (length a-shape)) (ndb (length b-shape))
           (max-ndim (max nda ndb))
           (a-pad (append (make-list (- max-ndim nda) :initial-element 1) a-shape))
           (b-pad (append (make-list (- max-ndim ndb) :initial-element 1) b-shape))
           (a-new '()) (b-new '()) (final '()))
      (loop for da in a-pad for db in b-pad
            do (push da a-new) (push 1 a-new)
               (push 1 b-new) (push db b-new)
               (push (* da db) final))
      (setf a-new (nreverse a-new) b-new (nreverse b-new) final (nreverse final))
      (let ((a-r (vt-reshape a a-new)) (b-r (vt-reshape b b-new)))
        (vt-view (vt-* a-r b-r) final)))))

(defun vt-meshgrid (vts-list &key (indexing :xy) (sparse nil) (copy t))
  "生成坐标网格（对标 numpy.meshgrid）。"
  (dolist (v vts-list)
    (assert (= (length (vt-shape v)) 1) (v) "meshgrid 输入必须为 1d"))
  (let* ((nd (length vts-list))
         (dims (mapcar (lambda (v) (first (vt-shape v))) vts-list))
         (output-shape (if (and (eq indexing :xy) (>= nd 2))
                           (let ((sh (copy-list dims))) (rotatef (first sh) (second sh)) sh)
                           dims))
         (target-axes (if (and (eq indexing :xy) (>= nd 2))
                          (let ((axes (loop for i below nd collect i)))
                            (rotatef (first axes) (second axes)) axes)
                          (loop for i below nd collect i))))
    (labels ((sparse-shape (i)
               (loop for ax from 0 below nd collect (if (= ax (nth i target-axes)) (nth i dims) 1))))
      (loop for i from 0 below nd for v in vts-list
            for src = (if copy (vt-copy v) v)
            for sp = (sparse-shape i)
            collect (if sparse (vt-reshape src sp)
                        (vt-broadcast-to (vt-reshape src sp) output-shape))))))
