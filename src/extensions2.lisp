;;;; extensions2.lisp — 补充 NumPy/PyTorch 重要缺失函数
(in-package :clvt)

;;; ------------------------------------------------------------------
;;; 1. 方便别名：fliplr / flipud
;;; ------------------------------------------------------------------
(defun vt-fliplr (vt)
  "左右翻转（沿最后一个轴），对标 np.fliplr。"
  (let ((rank (length (vt-shape vt))))
    (when (< rank 2) (error "vt-fliplr requires at least 2D tensor"))
    (vt-flip vt :axis (1- rank))))

(defun vt-flipud (vt)
  "上下翻转（沿第0轴），对标 np.flipud。"
  (let ((rank (length (vt-shape vt))))
    (when (< rank 1) (error "vt-flipud requires at least 1D tensor"))
    (vt-flip vt :axis 0)))

;;; ------------------------------------------------------------------
;;; 2. ediff1d — 相邻元素差（对标 np.ediff1d）
;;; ------------------------------------------------------------------
(defun vt-ediff1d (vt &key to-end to-beginning)
  "1D张量相邻元素差，对标 np.ediff1d。"
  (let* ((flat (vt-flatten vt))
         (n (vt-size flat))
         (diff (if (<= n 1)
                   (vt-zeros (list 0) :dtype (vt-dtype flat))
                   (let ((src (vt-slice flat `(0 ,(1- n))))
                         (dst (vt-slice flat `(1 ,n))))
                     (vt-- dst src))))
         (parts (list diff)))
    (when to-beginning
      (push (ensure-vt to-beginning :dtype (vt-dtype diff)) parts))
    (when to-end
      (setf parts (append parts (list (ensure-vt to-end :dtype (vt-dtype diff))))))
    (if (rest parts) (apply #'vt-concatenate 0 parts) diff)))

;;; ------------------------------------------------------------------
;;; 3. geomspace — 等比数列（对标 np.geomspace）
;;; ------------------------------------------------------------------
(defun vt-geomspace (start stop num &key (dtype :float64) (endpoint t))
  "等比数列（对数尺度等间距），对标 np.geomspace。start/stop 必须同号且非零。"
  (declare (fixnum num))
  (when (< num 1) (error "vt-geomspace: num must be >= 1"))
  (when (= num 1)
    (return-from vt-geomspace (vt-full (list 1) start :dtype dtype)))
  (let* ((s (coerce start 'double-float))
         (e (coerce stop 'double-float))
         (log-s (log (abs s)))
         (log-e (log (abs e)))
         (sign-s (if (minusp s) -1d0 1d0))
         (div (if endpoint (1- num) num))
         (result (vt-zeros (list num) :dtype :float64))
         (rdata (vt-data result)))
    (declare (fixnum div) (double-float log-s log-e sign-s))
    (dotimes (i num)
      (let* ((frac (if (= div 0) 0d0 (/ (coerce i 'double-float) (coerce div 'double-float))))
             (lv (+ log-s (* frac (- log-e log-s))))
             (v (* sign-s (exp lv))))
        (setf (aref rdata i) v)))
    (if (eq dtype :float32) (vt-astype result :float32) result)))

;;; ------------------------------------------------------------------
;;; 4. ravel-multi-index
;;; ------------------------------------------------------------------
(defun vt-ravel-multi-index (multi-index shape)
  "多维索引→扁平索引，对标 np.ravel_multi_index。支持标量和批量模式。"
  (if (every #'numberp multi-index)
      (let ((strides (let ((s 1) (lst nil))
                       (loop for d in (reverse shape) do (push s lst) (setf s (* s d)))
                       lst)))
        (reduce #'+ (mapcar #'* multi-index strides)))
      (let* ((n (length (car multi-index)))
             (strides (let ((s 1) (lst nil))
                        (loop for d in (reverse shape) do (push s lst) (setf s (* s d))) lst))
             (result (vt-zeros n :dtype :int64))
             (rdata (vt-data result)))
        (loop for i fixnum below n do
          (let ((idx 0))
            (declare (type (signed-byte 64) idx))
            (loop for mi in multi-index for str in strides do
              (incf idx (* (the fixnum (nth i mi)) (the fixnum str))))
            (setf (aref rdata i) idx)))
        result)))

;;; ------------------------------------------------------------------
;;; 5. triu_indices / tril_indices
;;; ------------------------------------------------------------------
(defun vt-tril-indices (n &key (k 0) m)
  "返回下三角索引 (rows, cols)，对标 np.tril_indices。"
  (declare (fixnum n k))
  (let ((cols (or m n)) (rows '()) (cl '()))
    (declare (fixnum cols))
    (loop for i fixnum below n do
      (loop for j fixnum below cols do
        (when (<= (- j i) k)
          (push i rows) (push j cl))))
    (values (vt-from-array (make-array (length rows) :element-type '(signed-byte 64)
                                       :initial-contents (nreverse rows)) :dtype :int64)
            (vt-from-array (make-array (length cl) :element-type '(signed-byte 64)
                                       :initial-contents (nreverse cl)) :dtype :int64))))

(defun vt-triu-indices (n &key (k 0) m)
  "返回上三角索引 (rows, cols)，对标 np.triu_indices。"
  (declare (fixnum n k))
  (let ((cols (or m n)) (rows '()) (cl '()))
    (declare (fixnum cols))
    (loop for i fixnum below n do
      (loop for j fixnum below cols do
        (when (>= (- j i) k)
          (push i rows) (push j cl))))
    (values (vt-from-array (make-array (length rows) :element-type '(signed-byte 64)
                                       :initial-contents (nreverse rows)) :dtype :int64)
            (vt-from-array (make-array (length cl) :element-type '(signed-byte 64)
                                       :initial-contents (nreverse cl)) :dtype :int64))))

;;; ------------------------------------------------------------------
;;; 6. vander
;;; ------------------------------------------------------------------
(defun vt-vander (x &key n (increasing nil))
  "范德蒙德矩阵，对标 np.vander。"
  (let* ((xv (vt-contiguous (vt-flatten x)))
         (len (vt-size xv))
         (ncols (or n len))
         (result (vt-zeros (list len ncols) :dtype (vt-dtype xv)))
         (xdata (vt-data xv)) (rdata (vt-data result))
         (one (coerce 1 (array-element-type (vt-data xv)))))
    (declare (fixnum len ncols))
    (dotimes (i len)
      (let ((xi (aref xdata i)))
        (dotimes (j ncols)
          (let* ((pow (if increasing j (- ncols 1 j)))
                 (base one))
            (declare (fixnum pow))
            (dotimes (k pow) (setf base (* base xi)))
            (setf (aref rdata (+ (* i ncols) j)) base)))))
    result))

;;; ------------------------------------------------------------------
;;; 7. one-hot
;;; ------------------------------------------------------------------
(defun vt-one-hot (x num-classes &key (dtype :float64))
  "one-hot编码，对标 torch.nn.functional.one_hot。"
  (let* ((x-vt (if (eq (vt-dtype x) :int64) x (vt-astype x :int64)))
         (in-shape (vt-shape x-vt))
         (out-shape (append in-shape (list num-classes)))
         (result (vt-zeros out-shape :dtype dtype))
         (xdata (vt-data x-vt))
         (total (reduce #'* in-shape :initial-value 1))
         (one-val (coerce 1 (vt-dtype->lisp-type dtype))))
    (declare (fixnum total num-classes))
    (dotimes (i total)
      (let ((cls (aref xdata i)))
        (when (and (>= cls 0) (< cls num-classes))
          (setf (row-major-aref (vt-data result) (+ (* i num-classes) cls)) one-val))))
    result))

;;; ------------------------------------------------------------------
;;; 8. standardize (z-score)
;;; ------------------------------------------------------------------
(defun vt-standardize (vt &key axis (ddof 0) dtype out)
  "Z-score 标准化：(x - mean) / std，对标 sklearn.preprocessing.scale。"
  (let* ((m (vt-mean vt :axis axis :keepdims t :dtype dtype))
         (s (vt-std vt :axis axis :keepdims t :ddof ddof :dtype dtype))
         (centered (vt-- vt m :dtype dtype)))
    (vt-map (lambda (x sd) (if (< (abs (float sd 1d0)) 1d-12)
                               (coerce 0 (vt-dtype->lisp-type (or dtype (vt-dtype vt))))
                               (/ x sd)))
            centered s :dtype dtype :out out)))

;;; ------------------------------------------------------------------
;;; 9. Layer Normalization
;;; ------------------------------------------------------------------
(defun vt-layer-norm (vt normalized-shape &key (eps 1d-5) gamma beta dtype out)
  "Layer Normalization，对标 torch.nn.functional.layer_norm。"
  (let* ((norm-axes (let ((rank (length (vt-shape vt)))
                          (ndim (length normalized-shape)))
                      (loop for i from (- rank ndim) below rank collect i)))
         (mean (vt-mean vt :axis norm-axes :keepdims t :dtype dtype))
         (var (vt-var vt :axis norm-axes :keepdims t :dtype dtype))
         (eps-typed (coerce eps (if (and dtype (eq dtype :float32)) 'single-float 'double-float)))
         (std (vt-sqrt (vt-+ var eps-typed :dtype dtype) :dtype dtype))
         (normed (vt-/ (vt-- vt mean :dtype dtype) std :dtype dtype)))
    (let ((result (if gamma (vt-* normed gamma :dtype dtype :out out)
                      (if out (progn (vt-copy-into normed out) out) normed))))
      (if beta (vt-+ result beta :dtype dtype :out result) result))))

;;; ------------------------------------------------------------------
;;; 10. apply-along-axis
;;; ------------------------------------------------------------------
(defun vt-apply-along-axis (func axis vt)
  "沿指定轴对1D切片应用函数，对标 np.apply_along_axis（支持1D→scalar和1D→1D）。"
  (let* ((vt-c (vt-contiguous vt))
         (shape (vt-shape vt-c))
         (rank (length shape))
         (ax (vt-normalize-axis axis rank))
         (ax-dim (nth ax shape))
         (ltype (vt-dtype->lisp-type (vt-dtype vt-c)))
         (before (subseq shape 0 ax))
         (after (subseq shape (1+ ax)))
         (outer (reduce #'* before :initial-value 1))
         (inner (reduce #'* after :initial-value 1)))
    (declare (fixnum ax-dim outer inner))
    ;; 探测输出形状
    (let* ((probe (make-array ax-dim :element-type ltype :initial-element (coerce 0 ltype)))
           (probe-vt (vt-from-array probe :dtype (vt-dtype vt-c)))
           (sample (funcall func probe-vt)))
      (cond
        ;; 1D -> scalar
        ((or (numberp sample) (and (vt-p sample) (null (vt-shape sample))))
         (let* ((out-shape (append before after))
                (result (vt-zeros out-shape :dtype (vt-dtype vt-c)))
                (rdata (vt-data result))
                (data (vt-data vt-c)))
           (dotimes (bi outer)
             (dotimes (ii inner)
               (let ((arr (make-array ax-dim :element-type ltype)))
                 (dotimes (k ax-dim)
                   (setf (aref arr k) (aref data (+ (* bi ax-dim inner) (* k inner) ii))))
                 (let* ((sl (vt-from-array arr :dtype (vt-dtype vt-c)))
                        (rv (funcall func sl)))
                   (setf (aref rdata (+ (* bi inner) ii))
                         (coerce (if (vt-p rv) (vt-item rv) rv) ltype))))))
           result))
        ;; 1D -> 1D
        ((and (vt-p sample) (= (length (vt-shape sample)) 1))
         (let* ((new-dim (car (vt-shape sample)))
                (out-shape (append before (list new-dim) after))
                (result (vt-zeros out-shape :dtype (vt-dtype vt-c)))
                (rdata (vt-data result))
                (data (vt-data vt-c)))
           (declare (fixnum new-dim))
           (dotimes (bi outer)
             (dotimes (ii inner)
               (let ((arr (make-array ax-dim :element-type ltype)))
                 (dotimes (k ax-dim)
                   (setf (aref arr k) (aref data (+ (* bi ax-dim inner) (* k inner) ii))))
                 (let* ((sl (vt-from-array arr :dtype (vt-dtype vt-c)))
                        (ov (funcall func sl))
                        (odata (vt-data ov)))
                   (dotimes (k new-dim)
                     (setf (aref rdata (+ (* bi new-dim inner) (* k inner) ii))
                           (aref odata k)))))))
           result))
        (t (error "vt-apply-along-axis: func must return scalar or 1D VT"))))))
