;;;; manip.lisp — 形状操作、视图、翻转、滚动、三角、填充

(in-package :clvt)

;;; ------------------------------------------------------------------
;;; 重塑 / 视图 / 转置
;;; ------------------------------------------------------------------

(defun %resolve-minus-one (new-shape total-size)
  "解析形状中的 -1 占位符。"
  (let ((neg-idx (position -1 new-shape)))
    (if (null neg-idx)
        new-shape
        (let* ((known (reduce #'* (remove -1 new-shape) :initial-value 1))
               (shape (copy-list new-shape)))
          (cond ((zerop known)
                 (if (zerop total-size)
                     (progn (setf (nth neg-idx shape) 0) shape)
                     (error "无法将形状 ~a 重塑为含 -1 的 ~a" nil new-shape)))
                ((not (zerop (rem total-size known)))
                 (error "无法将形状重塑为含 -1 的 ~a" new-shape))
                (t (setf (nth neg-idx shape) (/ total-size known)) shape))))))

(defun vt-view (vt new-shape)
  "零拷贝重塑视图（对标 pytorch tensor.view），要求输入连续。"
  (let* ((new-shape (%resolve-minus-one new-shape (vt-shape-to-size (vt-shape vt))))
         (old-size (vt-shape-to-size (vt-shape vt)))
         (new-size (vt-shape-to-size new-shape)))
    (unless (= old-size new-size)
      (error "view 失败: 元素总数不一致 (旧 ~a, 新 ~a)" old-size new-size))
    (unless (vt-contiguous-p vt)
      (error "view 失败: 张量内存不连续，请使用 vt-reshape"))
    (%make-vt :data (vt-data vt) :shape new-shape
              :strides (vt-compute-strides new-shape)
              :offset (vt-offset vt) :dtype (vt-dtype vt))))

(defun vt-reshape (vt new-shape)
  "重塑形状：连续则零拷贝视图，否则落地后重塑。"
  (let* ((new-shape (%resolve-minus-one new-shape (vt-shape-to-size (vt-shape vt))))
         (old-size (vt-shape-to-size (vt-shape vt)))
         (new-size (vt-shape-to-size new-shape)))
    (unless (= old-size new-size)
      (error "重塑失败: 元素总数不一致 (旧 ~a, 新 ~a)" old-size new-size))
    (if (vt-contiguous-p vt)
        (%make-vt :data (vt-data vt) :shape new-shape
                  :strides (vt-compute-strides new-shape)
                  :offset (vt-offset vt) :dtype (vt-dtype vt))
        (let ((cont (vt-contiguous vt)))
          (%make-vt :data (vt-data cont) :shape new-shape
                    :strides (vt-compute-strides new-shape)
                    :offset 0 :dtype (vt-dtype cont))))))

(defun vt-transpose (vt &optional (perm nil))
  "零拷贝转置。perm 为轴排列列表；缺省时反转所有轴。"
  (let* ((shape (vt-shape vt)) (strides (vt-strides vt)) (rank (length shape)))
    (unless perm
      (setf perm (loop for i from (1- rank) downto 0 collect i)))
    (unless (= (length perm) rank)
      (error "perm 长度 ~a 必须等于秩 ~a" (length perm) rank))
    (let ((seen (make-array rank :element-type 'bit :initial-element 0)))
      (dolist (p perm)
        (unless (and (integerp p) (<= 0 p (1- rank)))
          (error "perm 值 ~a 超出范围 [0, ~a]" p (1- rank)))
        (when (= (aref seen p) 1) (error "perm 中存在重复轴索引: ~a" p))
        (setf (aref seen p) 1)))
    (%make-vt :data (vt-data vt)
              :shape (mapcar (lambda (p) (nth p shape)) perm)
              :strides (mapcar (lambda (p) (nth p strides)) perm)
              :offset (vt-offset vt) :dtype (vt-dtype vt))))

(defun vt-squeeze (vt &key axis)
  "移除长度为 1 的维度（axis 指定单轴，缺省移除全部）。"
  (let ((shape (vt-shape vt)) (strides (vt-strides vt)) (offset (vt-offset vt)))
    (if axis
        (let* ((ax (vt-normalize-axis axis (length shape))))
          (unless (= (nth ax shape) 1)
            (error "无法挤压非单例维度: axis ~a 大小为 ~a" axis (nth ax shape)))
          (%make-vt :data (vt-data vt)
                    :shape (append (subseq shape 0 ax) (subseq shape (1+ ax)))
                    :strides (append (subseq strides 0 ax) (subseq strides (1+ ax)))
                    :offset offset :dtype (vt-dtype vt)))
        (let ((ns '()) (nst '()))
          (loop for d in shape for s in strides
                when (> d 1) do (push d ns) (push s nst))
          (%make-vt :data (vt-data vt) :shape (nreverse ns)
                    :strides (nreverse nst) :offset offset :dtype (vt-dtype vt))))))

(defun vt-expand-dims (vt axis)
  "在指定位置插入长度为 1 的新轴。"
  (let* ((shape (vt-shape vt)) (strides (vt-strides vt))
         (rank (length shape))
         (ax (if (< axis 0) (+ rank axis 1) axis)))
    (when (or (< ax 0) (> ax rank)) (error "轴 ~a 越界 (秩 ~a)" axis rank))
    (%make-vt :data (vt-data vt)
              :shape (append (subseq shape 0 ax) '(1) (subseq shape ax))
              :strides (append (subseq strides 0 ax) '(0) (subseq strides ax))
              :offset (vt-offset vt) :dtype (vt-dtype vt))))

(defun vt-unsqueeze (vt axis)
  (vt-expand-dims vt axis))

(defun vt-swapaxes (vt axis1 axis2)
  (let* ((rank (length (vt-shape vt)))
         (ax1 (vt-normalize-axis axis1 rank))
         (ax2 (vt-normalize-axis axis2 rank)))
    (vt-transpose vt (loop for i from 0 below rank
                           collect (cond ((= i ax1) ax2) ((= i ax2) ax1) (t i))))))

(defun vt-moveaxis (tensor source destination)
  "移动轴。source/destination 可为整数或整数列表。"
  (let* ((shape (vt-shape tensor)) (rank (length shape))
         (src (if (listp source) source (list source)))
         (dst (if (listp destination) destination (list destination))))
    (unless (= (length src) (length dst))
      (error "vt-moveaxis: source 与 destination 长度必须一致"))
    (setf src (mapcar (lambda (s) (vt-normalize-axis s rank)) src))
    (setf dst (mapcar (lambda (d) (vt-normalize-axis d rank)) dst))
    (let ((pairs '()))
      (loop for s in src for d in dst do (push (cons s d) pairs))
      (let ((remaining (loop for i below rank unless (member i src) collect i))
            (free (loop for i below rank unless (member i dst) collect i)))
        (loop for r in remaining for f in free do (push (cons r f) pairs)))
      (setf pairs (sort pairs #'< :key #'cdr))
      (vt-transpose tensor (mapcar #'car pairs)))))

(defun vt-rot90 (tensor &key (k 1) (axes '(0 1)))
  "在 axes 平面内旋转 90 度 k 次。"
  (let* ((rank (length (vt-shape tensor)))
         (ax0 (vt-normalize-axis (first axes) rank))
         (ax1 (vt-normalize-axis (second axes) rank)))
    (when (< rank 2) (error "vt-rot90: 秩必须 >= 2"))
    (when (= ax0 ax1) (error "vt-rot90: axes 必须不同"))
    (let ((k (mod k 4)))
      (loop with result = tensor repeat k
            do (setf result (vt-flip (vt-swapaxes result ax0 ax1) :axis ax0))
            finally (return result)))))

;;; ------------------------------------------------------------------
;;; 展平
;;; ------------------------------------------------------------------

(defun vt-flatten (vt)
  "展平为一维（返回副本）。"
  (vt-view (vt-copy vt) (list (vt-size vt))))

(defun vt-ravel (vt)
  "展平视图（连续则零拷贝，否则副本）。"
  (if (vt-contiguous-p vt)
      (vt-view vt (list (vt-size vt)))
      (vt-flatten vt)))

(defun vt-broadcast-to (vt new-shape)
  "广播到新形状（零拷贝视图）。"
  (let ((bs (vt-broadcast-shapes (vt-shape vt) new-shape)))
    (unless (equal bs new-shape)
      (error "形状 ~a 不能广播到 ~a" (vt-shape vt) new-shape))
    (%make-vt :data (vt-data vt) :shape new-shape
              :strides (vt-broadcast-strides (vt-shape vt) new-shape (vt-strides vt))
              :offset (vt-offset vt) :dtype (vt-dtype vt))))

;;; ------------------------------------------------------------------
;;; 翻转与滚动
;;; ------------------------------------------------------------------

(defun vt-flip (vt &key axis)
  "沿指定轴翻转（零拷贝视图）；axis 为 nil 时全部翻转。"
  (let ((shape (vt-shape vt)) (strides (vt-strides vt)))
    (if (null axis)
        (%make-vt :data (vt-data vt) :shape shape
                  :strides (mapcar #'- strides)
                  :offset (+ (vt-offset vt)
                             (loop for d in shape for s in strides sum (* (1- d) s)))
                  :dtype (vt-dtype vt))
        (let* ((rank (length shape))
               (ax (vt-normalize-axis axis rank))
               (strides (copy-list strides))
               (dim (nth ax shape)) (old-stride (nth ax strides)))
          (setf (nth ax strides) (- old-stride))
          (%make-vt :data (vt-data vt) :shape shape :strides strides
                    :offset (+ (vt-offset vt) (* (1- dim) old-stride))
                    :dtype (vt-dtype vt))))))

(defun vt-roll (vt shift &key axis)
  "滚动元素。shift/axis 可为整数或列表。"
  (let ((shift-list (if (listp shift) shift (list shift))))
    (cond
      ((and axis (listp axis))
       (let ((shifts (if (listp shift) shift
                         (make-list (length axis) :initial-element shift))))
         (unless (= (length shifts) (length axis))
           (error "shift 和 axis 的长度必须一致"))
         (loop with result = (vt-copy vt)
               for s in shifts for ax in axis
               do (setf result (vt-roll result s :axis ax))
               finally (return result))))
      (axis
       (let* ((sh (vt-shape vt)) (ax (vt-normalize-axis axis (length sh)))
              (n (nth ax sh)))
         (if (zerop n) vt
             (let ((s (mod (car shift-list) n)))
               (if (zerop s) vt
                   (vt-concatenate ax
                     (apply #'vt-slice vt (loop for i below (length sh)
                                                collect (if (= i ax) (list (- n s) n) '(:all))))
                     (apply #'vt-slice vt (loop for i below (length sh)
                                                collect (if (= i ax) (list 0 (- n s)) '(:all))))))))))
      (t
       (let* ((flat (vt-flatten vt)) (n (vt-size flat)))
         (if (zerop n) (vt-reshape flat (vt-shape vt))
             (let ((s (mod (car shift-list) n)))
               (if (zerop s) (vt-reshape flat (vt-shape vt))
                   (vt-reshape (vt-concatenate 0
                                 (vt-slice flat (list (- n s) n))
                                 (vt-slice flat (list 0 (- n s))))
                               (vt-shape vt))))))))))

;;; ------------------------------------------------------------------
;;; 窄切片与分割
;;; ------------------------------------------------------------------

(defun vt-narrow (vt axis start end)
  "零拷贝切片（等价 pytorch narrow）。"
  (let* ((shape (copy-list (vt-shape vt))) (rank (length shape))
         (ax (vt-normalize-axis axis rank)) (strides (vt-strides vt))
         (dim-size (nth ax shape)))
    (when (or (< start 0) (> end dim-size))
      (error "切片索引 [~a, ~a) 越界，轴大小 ~a" start end dim-size))
    (when (< end start)
      (error "vt-narrow: end (~a) 必须大于 start (~a)" end start))
    (setf (nth ax shape) (- end start))
    (%make-vt :data (vt-data vt) :shape shape :strides strides
              :offset (+ (vt-offset vt) (* start (nth ax strides)))
              :dtype (vt-dtype vt))))

(defun vt-split (tensor indices-or-sections &key (axis 0))
  "沿轴分割（对标 numpy array_split）。"
  (let* ((shape (vt-shape tensor)) (rank (length shape))
         (ax (vt-normalize-axis axis rank)) (dim-size (nth ax shape)))
    (cond
      ((integerp indices-or-sections)
       (let* ((n indices-or-sections)
              (base (floor dim-size n)) (rem (rem dim-size n)))
         (loop with start = 0 for i from 0 below n
               for chunk = (if (< i rem) (1+ base) base)
               collect (prog1 (vt-narrow tensor ax start (+ start chunk))
                         (incf start chunk)))))
      ((listp indices-or-sections)
       (let ((points (append '(0) indices-or-sections (list dim-size))))
         (setf points (mapcar (lambda (p) (if (minusp p) (+ p dim-size) p)) points))
         (setf points (mapcar (lambda (p) (max 0 (min p dim-size))) points))
         (loop for (s e) on points by #'cdr while e
               collect (vt-narrow tensor ax s e))))
      (t (error "indices-or-sections 必须是整数或整数列表")))))

(defun vt-vsplit (vt indices-or-sections) (vt-split vt indices-or-sections :axis 0))
(defun vt-hsplit (vt indices-or-sections)
  (if (<= (length (vt-shape vt)) 1)
      (vt-split vt indices-or-sections :axis 0)
      (vt-split vt indices-or-sections :axis 1)))
(defun vt-dsplit (vt indices-or-sections) (vt-split vt indices-or-sections :axis 2))

;;; ------------------------------------------------------------------
;;; 三角与对角
;;; ------------------------------------------------------------------

(defun vt-triu (tensor &key (k 0))
  "上三角矩阵（支持 batch）。"
  (let* ((res (vt-copy tensor)) (res-data (vt-data res))
         (rank (length (vt-shape res)))
         (shape-vec (coerce (vt-shape res) 'simple-vector))
         (strs-vec (coerce (vt-strides res) 'simple-vector))
         (zero (coerce 0 (vt-element-type res))))
    (when (< rank 2) (error "vt-triu 要求秩 >= 2"))
    (labels ((recurse (depth ptr)
               (if (= depth (- rank 2))
                   (let ((rows (svref shape-vec depth)) (cols (svref shape-vec (1+ depth)))
                         (str-r (svref strs-vec depth)) (str-c (svref strs-vec (1+ depth))))
                     (loop for r from 0 below rows
                           for rp = ptr then (+ rp str-r)
                           do (loop for c from 0 below cols
                                    when (< c (+ r k))
                                      do (setf (aref res-data (+ rp (* c str-c))) zero))))
                   (let ((dim (svref shape-vec depth)) (stride (svref strs-vec depth)))
                     (loop for i from 0 below dim do
                       (recurse (1+ depth) ptr) (incf ptr stride))))))
      (recurse 0 (vt-offset res)))
    res))

(defun vt-tril (tensor &key (k 0))
  "下三角矩阵（支持 batch）。"
  (let* ((res (vt-copy tensor)) (res-data (vt-data res))
         (rank (length (vt-shape res)))
         (shape-vec (coerce (vt-shape res) 'simple-vector))
         (strs-vec (coerce (vt-strides res) 'simple-vector))
         (zero (coerce 0 (vt-element-type res))))
    (when (< rank 2) (error "vt-tril 要求秩 >= 2"))
    (labels ((recurse (depth ptr)
               (if (= depth (- rank 2))
                   (let ((rows (svref shape-vec depth)) (cols (svref shape-vec (1+ depth)))
                         (str-r (svref strs-vec depth)) (str-c (svref strs-vec (1+ depth))))
                     (loop for r from 0 below rows
                           for rp = ptr then (+ rp str-r)
                           do (loop for c from 0 below cols
                                    when (> c (+ r k))
                                      do (setf (aref res-data (+ rp (* c str-c))) zero))))
                   (let ((dim (svref shape-vec depth)) (stride (svref strs-vec depth)))
                     (loop for i from 0 below dim do
                       (recurse (1+ depth) ptr) (incf ptr stride))))))
      (recurse 0 (vt-offset res)))
    res))

(defun vt-diagonal (tensor &key (offset 0))
  "提取对角线（支持 batch）。"
  (let* ((in-shape (vt-shape tensor)) (rank (length in-shape)))
    (when (< rank 2) (error "vt-diagonal 要求秩 >= 2"))
    (let* ((rows (nth (- rank 2) in-shape)) (cols (nth (1- rank) in-shape))
           (r-init (if (> offset 0) 0 (- offset)))
           (c-init (if (> offset 0) offset 0))
           (diag-len (max 0 (min (- rows r-init) (- cols c-init))))
           (out-shape (append (subseq in-shape 0 (- rank 2)) (list diag-len)))
           (res (vt-zeros out-shape :dtype (vt-dtype tensor)))
           (res-data (vt-data res))
           (in-data (vt-data tensor)) (in-strs (vt-strides tensor))
           (out-idx 0)
           (batch-size (reduce #'* (subseq in-shape 0 (- rank 2)) :initial-value 1))
           (str-r (nth (- rank 2) in-strs)) (str-c (nth (1- rank) in-strs))
           (in-offset (vt-offset tensor)))
      (dotimes (batch batch-size)
        (let ((in-ptr in-offset) (rem batch))
          (loop for d from 0 below (- rank 2) do
            (let ((dim (nth d in-shape)) (str (nth d in-strs)))
              (multiple-value-bind (q r) (floor rem dim)
                (incf in-ptr (* r str)) (setf rem q))))
          (loop for i from 0 below diag-len
                for src = (+ in-ptr (* (+ r-init i) str-r) (* (+ c-init i) str-c))
                do (setf (aref res-data out-idx) (aref in-data src)) (incf out-idx))))
      res)))

(defun vt-diag (tensor &key (k 0))
  "提取对角线或构造对角矩阵（对标 numpy.diag）。"
  (let* ((shape (vt-shape tensor)) (rank (length shape)))
    (cond
      ((= rank 1)
       (let* ((n (first shape)) (dim (+ n (abs k)))
              (res (vt-zeros (list dim dim) :dtype (vt-dtype tensor)))
              (res-data (vt-data res))
              (in-data (vt-data tensor)) (in-offset (vt-offset tensor))
              (in-stride (first (vt-strides tensor)))
              (row-stride (first (vt-strides res))) (col-stride (second (vt-strides res)))
              (start (if (> k 0) (* k col-stride) (* (- k) row-stride))))
         (loop for i from 0 below n
               for src = (+ in-offset (* i in-stride))
               for dst = start then (+ dst row-stride col-stride)
               do (setf (aref res-data dst) (aref in-data src)))
         res))
      ((= rank 2) (vt-diagonal tensor :offset k))
      (t (error "vt-diag 仅支持 1D 或 2D 输入，当前维度 ~a" rank)))))

;;; ------------------------------------------------------------------
;;; 重复与平铺
;;; ------------------------------------------------------------------

(defun vt-repeat (vt repeats &key axis)
  "重复元素。"
  (if (null axis)
      (let* ((flat (vt-flatten vt)) (size (vt-size flat))
             (reps (if (listp repeats) repeats (make-list size :initial-element repeats)))
             (parts (loop for i from 0 below size for rep in reps
                          when (> rep 0)
                            collect (make-vt (list rep) (vt-ref flat i) :dtype (vt-dtype vt)))))
        (if parts (apply #'vt-concatenate 0 parts) (vt-zeros '(0) :dtype (vt-dtype vt))))
      (let* ((sh (vt-shape vt)) (ax (vt-normalize-axis axis (length sh)))
             (ax-size (nth ax sh))
             (reps (if (listp repeats) repeats (make-list ax-size :initial-element repeats)))
             (slices (loop for i from 0 below ax-size for rep in reps
                           for part = (apply #'vt-slice vt
                                             (loop for d below (length sh)
                                                   collect (if (= d ax) (list i (1+ i)) '(:all))))
                           when (> rep 0)
                             collect (if (= rep 1) part
                                         (apply #'vt-concatenate ax (loop repeat rep collect part))))))
        (if slices (apply #'vt-concatenate ax slices)
            (let ((zero-shape (copy-list sh)))
              (setf (nth ax zero-shape) 0)
              (vt-zeros zero-shape :dtype (vt-dtype vt)))))))

(defun vt-tile (vt reps)
  "平铺构造新数组（对标 numpy.tile）。"
  (let* ((sh (vt-shape vt))
         (reps-list (if (listp reps) reps (list reps)))
         (ndim (max (length sh) (length reps-list)))
         (padded-sh (append (make-list (- ndim (length sh)) :initial-element 1) sh))
         (padded-reps (append (make-list (- ndim (length reps-list)) :initial-element 1) reps-list))
         (result (vt-reshape vt padded-sh)))
    (loop for axis from 0 below ndim for rep = (nth axis padded-reps)
          when (> rep 1)
            do (setf result (apply #'vt-concatenate axis (loop repeat rep collect result))))
    (vt-view result (mapcar #'* padded-sh padded-reps))))

;;; ------------------------------------------------------------------
;;; 填充
;;; ------------------------------------------------------------------

(defun %normalize-pad-width (pad-width rank)
  (labels ((intp (x) (and (integerp x) (>= x 0)))
           (pairp (x) (and (consp x) (= (length x) 2) (every #'integerp x)
                           (>= (first x) 0) (>= (second x) 0)))
           (norm (x) (cond ((intp x) (list x x)) ((pairp x) x)
                           (t (error "invalid pad-width element: ~a" x)))))
    (cond ((integerp pad-width) (make-list rank :initial-element (norm pad-width)))
          ((and (listp pad-width) (= (length pad-width) 1) (integerp (first pad-width)))
           (make-list rank :initial-element (norm (first pad-width))))
          ((and (listp pad-width) (= (length pad-width) 2) (every #'integerp pad-width))
           (make-list rank :initial-element (norm pad-width)))
          ((and (listp pad-width) (= (length pad-width) 1) (consp (first pad-width))
                (= (length (first pad-width)) 2) (every #'integerp (first pad-width)))
           (make-list rank :initial-element (norm (first pad-width))))
          ((and (listp pad-width) (= (length pad-width) rank)) (mapcar #'norm pad-width))
          (t (error "pad-width 长度 ~a 与秩 ~a 不匹配" (length pad-width) rank)))))

(defun %pad-map (mode dist sk side)
  (when (and (<= sk 0) (not (eq mode :constant)))
    (error "无法用 mode ~a 扩展空轴" mode))
  (ecase mode
    (:edge (if (eq side :left) 0 (1- sk)))
    (:wrap (mod (if (eq side :left) (- sk dist) (1- dist)) sk))
    (:reflect (let* ((period (* 2 (1- sk)))
                     (x (if (eq side :left) dist (- sk 1 dist)))
                     (idx (mod x period)))
                (if (< idx sk) idx (- period idx))))
    (:symmetric (let* ((period (* 2 sk))
                       (x (if (eq side :left) (- dist) (+ sk dist -1)))
                       (idx (mod x period)))
                  (if (< idx sk) idx (- period idx 1))))))

(defun vt-pad (vt pad-width &key (mode :constant) (constant-values 0))
  "对张量填充。mode: :constant/:edge/:wrap/:reflect/:symmetric。"
  (let* ((shape (vt-shape vt)) (rank (length shape))
         (pad (%normalize-pad-width pad-width rank))
         (cv (if (listp constant-values) constant-values (list constant-values constant-values)))
         (c-left (vt-cast (first cv) (vt-dtype vt)))
         (c-right (vt-cast (second cv) (vt-dtype vt)))
         (new-shape (loop for s in shape for (b a) in pad collect (+ s b a)))
         (out (vt-zeros new-shape :dtype (vt-dtype vt)))
         (out-data (vt-data out)) (in-data (vt-data vt))
         (in-offset (vt-offset vt))
         (out-strides (coerce (vt-compute-strides new-shape) 'simple-vector))
         (in-shape (coerce shape 'simple-vector))
         (in-strides (coerce (vt-strides vt) 'simple-vector))
         (pad-before (coerce (mapcar #'first pad) 'simple-vector))
         (out-size (vt-size out)))
    (loop for out-ptr fixnum from 0 below out-size do
      (let ((rem out-ptr) (in-ptr in-offset) (is-const nil) (const-val c-left))
        (loop for d from 0 below rank do
          (multiple-value-bind (out-idx r) (floor rem (svref out-strides d))
            (setf rem r)
            (let* ((bk (svref pad-before d)) (sk (svref in-shape d))
                   (offset (- out-idx bk)) (src-idx 0))
              (cond ((< offset 0)
                     (if (eq mode :constant)
                         (unless is-const (setf is-const t const-val c-left))
                         (setf src-idx (%pad-map mode (- offset) sk :left))))
                    ((>= offset sk)
                     (if (eq mode :constant)
                         (unless is-const (setf is-const t const-val c-right))
                         (setf src-idx (%pad-map mode (- offset sk -1) sk :right))))
                    (t (setf src-idx offset)))
              (incf in-ptr (* src-idx (svref in-strides d))))))
        (setf (aref out-data out-ptr) (if is-const const-val (aref in-data in-ptr)))))
    out))
