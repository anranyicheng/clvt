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
(defun %as-1d (x dtype)
  "将任意输入（标量、列表、任意秩张量）折叠为 1 维张量。
   标量 → shape (1)；其余 → 展平为一维。"
  (let ((tv (ensure-vt x :dtype dtype)))
    (if (null (vt-shape tv))
        (vt-reshape tv '(1))
        (vt-flatten tv))))

(defun vt-ediff1d (vt &key to-end to-beginning)
  "1D张量相邻元素差，对标 np.ediff1d。
   to-beginning / to-end 可为标量、列表或张量，统一按 1 维拼接。"
  (with-float-safe
    (let* ((flat (vt-flatten vt))
           (n (vt-size flat))
           (diff (if (<= n 1)
                     (vt-zeros (list 0) :dtype (vt-dtype flat))
                     (let ((src (vt-slice flat `(0 ,(1- n))))
                           (dst (vt-slice flat `(1 ,n))))
                       (vt-- dst src))))
           (parts (list diff)))
      (when to-beginning
        (push (%as-1d to-beginning (vt-dtype diff)) parts))
      (when to-end
        (setf parts (append parts
                            (list (%as-1d to-end (vt-dtype diff))))))
      (if (rest parts) (apply #'vt-concatenate 0 parts) diff))))

;;; ------------------------------------------------------------------
;;; 3. geomspace — 等比数列（对标 np.geomspace）
;;; ------------------------------------------------------------------

(defun vt-geomspace (start stop num &key (dtype :float64) (endpoint t))
  "等比数列（对数尺度等间距），对标 np.geomspace。
   前置条件（否则报错）：
   - num >= 1
   - start 和 stop 均非零
   - start 和 stop 同号
   符号处理：结果为全正或全负，符号取自 start（= stop 的符号）。
   示例：
     (vt-geomspace 1 1000 4)    => [1, 10, 100, 1000]
     (vt-geomspace -1 -1000 4)  => [-1, -10, -100, -1000]
     (vt-geomspace 1 -100 5)    => error（异号）
     (vt-geomspace 0 100 5)     => error（含零）"
  (declare (fixnum num))
  (when (< num 1)
    (error "vt-geomspace: num 必须 >= 1，当前为 ~a" num))

  (let ((s (coerce start 'double-float))
        (e (coerce stop  'double-float)))
    ;; ---- 前置校验：与 NumPy 一致 ----
    (when (or (zerop s) (zerop e))
      (error "vt-geomspace: start (~a) 和 stop (~a) 均不能为零" start stop))
    (when (not (eql (minusp s) (minusp e)))
      (error "vt-geomspace: start (~a) 和 stop (~a) 必须同号" start stop))

    (when (= num 1)
      (return-from vt-geomspace
        (vt-full (list 1) s :dtype dtype)))

    (let* ((log-s (log (abs s)))
           (log-e (log (abs e)))
           (sign  (if (minusp s) -1d0 1d0))
           (div   (if endpoint (1- num) num))
           (result (vt-zeros (list num) :dtype :float64))
           (rdata  (vt-data result)))
      (declare (type fixnum div)
               (type double-float log-s log-e sign))
      (dotimes (i num)
        (let* ((frac (/ (coerce i 'double-float)
                        (coerce div 'double-float)))
               (lv (+ log-s (* frac (- log-e log-s))))
               (v  (* sign (exp lv))))
          (setf (aref rdata i) v)))
      (if (eq dtype :float32) (vt-astype result :float32) result))))

;;; ------------------------------------------------------------------
;;; 4. ravel-multi-index
;;; ------------------------------------------------------------------
(defun vt-ravel-multi-index (multi-index shape)
  "多维索引→扁平索引，对标 np.ravel_multi_index。支持标量和批量模式。"
  (let ((strides (let ((s 1) (lst nil))
                   (loop for d in (reverse shape)
                         do (push s lst) (setf s (* s d)))
                   lst)))
    (if (every #'numberp multi-index)
        (reduce #'+ (mapcar #'* multi-index strides))
        (let* ((n (length (first multi-index)))
               (result (vt-zeros (list n) :dtype :int64))
               (rdata (vt-data result)))
          (loop for i fixnum below n do
            (let ((idx 0))
              (declare (type (signed-byte 64) idx))
              (loop for mi in multi-index
                    for str in strides
                    do (incf idx (* (truncate (nth i mi)) (the fixnum str))))
              (setf (aref rdata i) idx)))
          result))))

;;; ------------------------------------------------------------------
;;; 5. triu_indices / tril_indices
;;; ------------------------------------------------------------------
(defun vt-tril-indices (n &key (k 0) m)
  "返回下三角索引 (rows, cols)，对标 np.tril_indices。"
  (declare (fixnum n k))
  (when (minusp n)
    (error "vt-tril-indices: n (~a) 不能为负" n))
  (let ((cols (or m n)) (rows '()) (cl '()))
    (declare (fixnum cols))
    (when (minusp cols)
      (error "vt-tril-indices: m (~a) 不能为负" m))
    (loop for i fixnum below n do
      (loop for j fixnum below cols do
        (when (<= (- j i) k)
          (push i rows) (push j cl))))
    (values (vt-from-array
             (make-array (length rows) :element-type '(signed-byte 64)
                                       :initial-contents (nreverse rows))
             :dtype :int64)
            (vt-from-array
             (make-array (length cl) :element-type '(signed-byte 64)
                                     :initial-contents (nreverse cl))
             :dtype :int64))))

(defun vt-triu-indices (n &key (k 0) m)
  "返回上三角索引 (rows, cols)，对标 np.triu_indices。"
  (declare (fixnum n k))
  (when (minusp n)
    (error "vt-triu-indices: n (~a) 不能为负" n))
  (let ((cols (or m n)) (rows '()) (cl '()))
    (declare (fixnum cols))
    (when (minusp cols)
      (error "vt-triu-indices: m (~a) 不能为负" m))
    (loop for i fixnum below n do
      (loop for j fixnum below cols do
        (when (>= (- j i) k)
          (push i rows) (push j cl))))
    (values (vt-from-array
	     (make-array (length rows) :element-type '(signed-byte 64)
                                       :initial-contents (nreverse rows))
             :dtype :int64)
            (vt-from-array
	     (make-array (length cl) :element-type '(signed-byte 64)
                                     :initial-contents (nreverse cl))
             :dtype :int64))))

;;; ------------------------------------------------------------------
;;; 6. vander
;;; ------------------------------------------------------------------
(defun vt-vander (x &key n (increasing nil))
  "范德蒙德矩阵，对标 np.vander。
   - increasing = t : 第 j 列为 x^j
   - increasing = nil : 第 j 列为 x^(ncols-1-j)
   实现用一行内 O(ncols) 递推，避免每个元素重复做幂运算。"
  (let* ((xv (vt-contiguous (vt-flatten x)))
         (len (vt-size xv))
         (ncols (or n len))
         (result (vt-zeros (list len ncols) :dtype (vt-dtype xv)))
         (xdata (vt-data xv))
         (rdata (vt-data result))
         (one (vt-cast 1 (vt-dtype xv))))
    (declare (fixnum len ncols))
    (dotimes (i len)
      (let* ((xi (aref xdata i))
             (base one)
             (row-start (* i ncols)))
        (declare (type fixnum row-start))
        (if increasing
            (dotimes (j ncols)
              (setf (aref rdata (+ row-start j)) base)
              (setf base (* base xi)))
            ;; 递减：j=0 写入最高次幂 xi^(ncols-1)，j=ncols-1 写入 xi^0
            (dotimes (j ncols)
              (setf (aref rdata (+ row-start (- ncols 1 j))) base)
              (setf base (* base xi))))))
    result))

;;; ------------------------------------------------------------------
;;; 7. one-hot
;;; ------------------------------------------------------------------
(defun vt-one-hot (x num-classes &key (dtype :float64))
  "one-hot 编码，对标 torch.nn.functional.one_hot。
   x 可为标量、列表或任意形状张量；非连续视图会先落地为连续副本，
   索引超出 [0, num-classes) 时按 PyTorch 语义报错。"
  (declare (fixnum num-classes))
  (when (< num-classes 1)
    (error "vt-one-hot: num-classes (~a) 必须为正整数" num-classes))
  (let* ((x-contig (vt-contiguous (vt-astype (ensure-vt x) :int64)))
         (in-shape (vt-shape x-contig))
         (out-shape (append in-shape (list num-classes)))
         (result (vt-zeros out-shape :dtype dtype))
         (xdata (vt-data x-contig))
         (x-off (vt-offset x-contig))
         (rdata (vt-data result))
         (total (reduce #'* in-shape :initial-value 1))
         (one-val (coerce 1 (vt-dtype->lisp-type dtype))))
    (declare (type fixnum total))
    (dotimes (i total)
      (let ((cls (aref xdata (+ x-off i))))
        (unless (and (>= cls 0) (< cls num-classes))
          (error "vt-one-hot: 类别 ~a 越界 [0, ~a)" cls num-classes))
        (setf (aref rdata (+ (* i num-classes) cls)) one-val)))
    result))

;;; ------------------------------------------------------------------
;;; 8. standardize (z-score)
;;; ------------------------------------------------------------------
(defun vt-standardize (vt &key axis (ddof 0) dtype out)
  "Z-score 标准化：(x - mean) / std，对标 sklearn.preprocessing.scale。
   结果 dtype 永远是浮点：显式传入整数 dtype 会报错，
   未指定时按输入 dtype 提升到 :float32 / :float64。"
  (when (and dtype (member dtype '(:int8 :int16 :int32 :int64
                                   :uint8 :uint16)))
    (error "vt-standardize: 标准化结果必须为浮点 dtype，收到 ~a" dtype))
  (let* ((final-dtype (or dtype
                          (if (eq (vt-dtype vt) :float32) :float32 :float64)))
         (m (vt-mean vt :axis axis :keepdims t :dtype final-dtype))
         (s (vt-std vt :axis axis :keepdims t :ddof ddof :dtype final-dtype))
         (centered (vt-- vt m :dtype final-dtype))
         (zero (if (eq final-dtype :float32) 0.0s0 0.0d0)))
    (vt-map (lambda (x sd)
              (if (< (abs (float sd 1d0)) 1d-12)
                  zero
                  (/ x sd)))
            centered s :dtype final-dtype :out out)))

;;; ------------------------------------------------------------------
;;; 9. Layer Normalization
;;; ------------------------------------------------------------------
(defun vt-layer-norm (vt normalized-shape &key (eps 1d-5) gamma beta dtype out)
  "Layer Normalization，对标 torch.nn.functional.layer_norm。
   normalized-shape 是输入张量的尾部形状，要求：
   - 其秩 <= 输入张量的秩；
   - 与输入张量的尾部维度逐一相等。
   例如 vt-shape=(2,3,4) 时 normalized-shape 可为 (4)、(3,4)、(2,3,4)。
   gamma / beta 的形状必须严格等于 normalized-shape。"
  (let* ((in-shape (vt-shape vt))
         (rank (length in-shape))
         (ndim (length normalized-shape)))

    ;; 两层前置校验，避免误导性的 \"轴重复\" 报错
    (when (> ndim rank)
      (error "vt-layer-norm: normalized-shape 秩 (~a) 不能大于输入张量秩 (~a)"
             ndim rank))
    (let ((tail (subseq in-shape (- rank ndim))))
      (unless (equal tail normalized-shape)
        (error "vt-layer-norm: normalized-shape ~a 与输入尾部维度 ~a 不匹配"
               normalized-shape tail)))

    ;; gamma / beta 形状校验
    (when gamma
      (unless (equal (vt-shape gamma) normalized-shape)
        (error "vt-layer-norm: gamma 形状 ~a 与 normalized-shape ~a 不匹配"
               (vt-shape gamma) normalized-shape)))
    (when beta
      (unless (equal (vt-shape beta) normalized-shape)
        (error "vt-layer-norm: beta 形状 ~a 与 normalized-shape ~a 不匹配"
               (vt-shape beta) normalized-shape)))

    (let* ((norm-axes (loop for i from (- rank ndim) below rank collect i))
           (mean (vt-mean vt :axis norm-axes :keepdims t :dtype dtype))
           (var (vt-var vt :axis norm-axes :keepdims t :dtype dtype))
           (final-dtype (or dtype
                            (if (eq (vt-dtype vt) :float32) :float32 :float64)))
           (eps-typed (coerce eps (if (eq final-dtype :float32)
                                      'single-float
                                      'double-float)))
           (std (vt-sqrt (vt-+ var eps-typed :dtype final-dtype)
                         :dtype final-dtype))
           (normed (vt-/ (vt-- vt mean :dtype final-dtype)
                         std :dtype final-dtype)))
      (let ((result (if gamma
                        (vt-* normed gamma :dtype final-dtype :out out)
                        (if out
                            (progn (vt-copy-into normed out) out)
                            normed))))
        (if beta
            (vt-+ result beta :dtype final-dtype :out result)
            result)))))

;;; ------------------------------------------------------------------
;;; 10. apply-along-axis
;;; ------------------------------------------------------------------
(defun vt-apply-along-axis (func axis vt)
  "沿指定轴对 1D 切片应用函数，对标 np.apply_along_axis。
   支持 func 返回标量（1D→scalar）或 1D 张量（1D→1D）。
   注意：func 会被额外调用一次用于探测输出形状，因此不应有副作用。"
  (with-float-safe
    (let* ((vt-c (vt-contiguous vt))
           (shape (vt-shape vt-c))
           (rank (length shape))
           (ax (vt-normalize-axis axis rank))
           (ax-dim (nth ax shape))
           (ltype (vt-dtype->lisp-type (vt-dtype vt-c)))
           (before (subseq shape 0 ax))
           (after (subseq shape (1+ ax)))
           (outer (reduce #'* before :initial-value 1))
           (inner (reduce #'* after :initial-value 1))
           (data (vt-data vt-c))
           (data-off (vt-offset vt-c)))
      (declare (fixnum ax-dim outer inner))
      ;; 探测输出形状
      (let* ((probe (make-array ax-dim :element-type ltype
                                         :initial-element (coerce 0 ltype)))
             (probe-vt (vt-from-array probe :dtype (vt-dtype vt-c)))
             (sample (funcall func probe-vt)))
        (cond
          ;; 1D -> scalar
          ((or (numberp sample) (and (vt-p sample) (null (vt-shape sample))))
           (let* ((out-shape (append before after))
                  (result (vt-zeros out-shape :dtype (vt-dtype vt-c)))
                  (rdata (vt-data result)))
             (dotimes (bi outer)
               (dotimes (ii inner)
                 (let ((arr (make-array ax-dim :element-type ltype)))
                   (dotimes (k ax-dim)
                     (setf (aref arr k)
                           (aref data (+ data-off
                                         (* bi ax-dim inner)
                                         (* k inner) ii))))
                   (let* ((sl (vt-from-array arr :dtype (vt-dtype vt-c)))
                          (rv (funcall func sl))
                          (raw (if (vt-p rv) (vt-item rv) rv)))
                     (setf (aref rdata (+ (* bi inner) ii))
                           (vt-cast raw (vt-dtype vt-c)))))))
             result))
          ;; 1D -> 1D
          ((and (vt-p sample) (= (length (vt-shape sample)) 1))
           (let* ((new-dim (car (vt-shape sample)))
                  (out-shape (append before (list new-dim) after))
                  (result (vt-zeros out-shape :dtype (vt-dtype vt-c)))
                  (rdata (vt-data result)))
             (declare (fixnum new-dim))
             (dotimes (bi outer)
               (dotimes (ii inner)
                 (let ((arr (make-array ax-dim :element-type ltype)))
                   (dotimes (k ax-dim)
                     (setf (aref arr k)
                           (aref data (+ data-off
                                         (* bi ax-dim inner)
                                         (* k inner) ii))))
                   (let* ((sl (vt-from-array arr :dtype (vt-dtype vt-c)))
                          (ov (funcall func sl)))
                     ;; 用 vt-ref 逐元素读取，兼容非连续/视图返回值
                     (dotimes (k new-dim)
                       (setf (aref rdata (+ (* bi new-dim inner)
                                            (* k inner) ii))
                             (vt-ref ov k)))))))
             result))
          (t (error "vt-apply-along-axis: func must return scalar or 1D VT")))))))
