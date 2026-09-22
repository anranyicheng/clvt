;;;; random.lisp — 随机数生成

(in-package :clvt)

(defvar *vt-default-random-state* (make-random-state *random-state*)
  "clvt 内部默认随机状态。")

(defun vt-make-random-state (&optional seed)
 (if (null seed)
	     (make-random-state nil)
	     (sb-ext::seed-random-state seed)))

(defvar *vt-random-state-lock*
  (sb-thread:make-mutex :name "vt-random-state"))

(defun vt-random-seed (seed)
  (sb-thread:with-mutex (*vt-random-state-lock*)
    (setf *vt-default-random-state* (vt-make-random-state seed))))

(declaim (inline %copy-default-rng))
(defun %copy-default-rng ()
  "在锁保护下返回默认随机状态的拷贝，避免多线程直接共享 mutable random-state。"
  (sb-thread:with-mutex (*vt-random-state-lock*)
    (make-random-state *vt-default-random-state*)))

  (declaim (inline %uniform-rand %normal-rand))
(defun %uniform-rand (state)
  (random 1.0d0 state))

(defun %normal-rand (state)
  (let ((u1 (max least-positive-double-float (random 1.0d0 state)))
        (u2 (random 1.0d0 state)))
    (* (sqrt (* -2.0d0 (log u1)))
       (cos (* 2.0d0 pi u2)))))

(defun vt-random (shape &key (dtype :float64) (rng nil))
  (declare (list shape))
  (setf rng (or rng (%copy-default-rng)))
  (vt-map (lambda (x)
	    (declare (ignore x))
	    (vt-cast (%uniform-rand rng) dtype))
          (vt-zeros shape :dtype dtype)))

(defun vt-random-uniform
    (shape &key (low 0.0d0) (high 1.0d0) (dtype :float64) (rng nil))
  (declare (list shape))
  (assert (< low high) (low high))
  (setf rng (or rng (%copy-default-rng)))
  (let ((range (- high low)))
    (vt-map (lambda (x)
	      (declare (ignore x))
	      (vt-cast (+ low (* range (%uniform-rand rng))) dtype))
            (vt-zeros shape :dtype dtype))))

(defun vt-random-normal
    (shape &key (mean 0.0d0) (std 1.0d0) (dtype :float64) (rng nil))
  (declare (list shape))
  (setf rng (or rng (%copy-default-rng)))
  (let ((res (vt-zeros shape :dtype dtype)))
    (vt-do-each (ptr val res)
      (declare (ignore val))
      (setf (aref (vt-data res) ptr)
	    (vt-cast (+ mean (* std (%normal-rand rng))) dtype)))
    res))

(defun vt-random-int
    (low high &key (size nil) (dtype :int64) (rng nil))
  (setf rng (or rng (%copy-default-rng)))
  (let ((range (- high low)))
    (assert (>= range 0) (high low))
    (if (zerop range)
        (if size
	    (vt-full size low :dtype dtype)
	    (make-vt nil low :dtype dtype))
        (if size
            (vt-astype (vt-map (lambda (x)
				 (declare (ignore x))
				 (+ low (random range rng)))
                               (vt-zeros size :dtype dtype))
                       dtype)
            (make-vt nil (+ low (random range rng)) :dtype dtype)))))

(defun vt-random-integers
    (low high &key (size nil) (dtype :int64) (rng nil))
  (setf rng (or rng (%copy-default-rng)))
  (vt-random-int low high :size size :dtype dtype :rng rng))


(defun vt-random-choice
    (a &key (size nil) (replace t) (p nil) (dtype nil) (rng nil))
  "从一维数组 a 中抽样。
   a        : 整数 n（表示从 [0, n) 抽样）或一维张量。
   size     : nil → 返回 0 维张量；否则返回形状为 size 的张量。
              整数 → 1D 张量；list → 多维张量。
   replace  : t（默认）有放回；nil 无放回。
   p        : 权重列表（长度 = n，非负，总和 > 0）。
   dtype    : 输出 dtype，默认与 a 一致。
   rng      : 随机状态。
   对标 numpy.random.choice / torch.multinomial。"
  (setf rng (or rng (%copy-default-rng)))  
  (let* ((source (if (integerp a)
                     (progn (assert (> a 0) (a))
                            (vt-arange a :dtype (or dtype :int64)))
                     (ensure-vt a)))
         (n (vt-size source))
         (out-dtype (or dtype (vt-dtype source)))
         (src-data (vt-data source))
         (src-offset (vt-offset source))
         (size-shape (when size
                       (if (listp size) size (list size))))
         (size-total (if size-shape
                         (reduce #'* size-shape :initial-value 1)
                         1)))
    (assert (> n 0) (n))
    ;; --- 概率向量：归一化 CDF（用于有放回） ---
    (let ((cdf (when p
                 (assert (= (length p) n) (p n))
                 (assert (every (lambda (w) (>= w 0)) p) ())
                 (let ((total (reduce #'+ p)))
                   (assert (> total 0) ())
                   (let ((cum 0.0d0))
                     (coerce (mapcar (lambda (w)
                                       (incf cum (/ w total)) cum)
                                     p)
                             'vector))))))
      ;; --- 无放回：size 上限校验 ---
      (when (and (not replace) size)
        (when (> size-total n)
          (error "vt-random-choice: replace=nil 时 size 总数 (~a) 不能大于 n (~a)"
                 size-total n)))
      (labels
          ((sample-uniform-replace ()
             (aref src-data (+ src-offset (random n rng))))

           (sample-weighted-replace ()
             (let ((r (random 1.0d0 rng)))
               (loop for i from 0 below n
                     when (<= r (aref cdf i))
                       return (aref src-data (+ src-offset i))
                     finally (return (aref src-data (+ src-offset (1- n)))))))

           (sample-uniform-no-replace-batch (k)
             (let ((idx (make-array n :element-type 'fixnum
                                      :initial-contents (loop for i below n collect i))))
               (loop for i from 0 below k do
                 (let ((j (+ i (random (- n i) rng))))
                   (rotatef (aref idx i) (aref idx j))))
               (loop for i from 0 below k
                     collect (aref src-data (+ src-offset (aref idx i))))))

           (sample-weighted-no-replace-batch (k)
             (let* ((weights (coerce (or p (make-list n :initial-element 1.0d0))
                                     'vector))
                    (result (make-array k)))
               (dotimes (slot k)
                 (let ((total 0.0d0))
                   (dotimes (i n) (incf total (aref weights i)))
                   (when (<= total 0.0d0)
                     (error "vt-random-choice: 无放回抽样权重耗尽"))
                   (let* ((r (* (random 1.0d0 rng) total))
                          (cum 0.0d0)
                          (chosen -1))
                     (dotimes (i n)
                       (when (>= chosen 0) (return))
                       (incf cum (aref weights i))
                       (when (<= r cum) (setf chosen i)))
                     (when (minusp chosen)
                       (dotimes (i n)
                         (when (plusp (aref weights i))
                           (setf chosen i) (return))))
                     (when (minusp chosen)
                       (error "vt-random-choice: 无放回抽样失败"))
                     (setf (aref weights chosen) 0.0d0)
                     (setf (aref result slot)
                           (aref src-data (+ src-offset chosen))))))
               (coerce result 'list))))

        (cond
          ;; ---- size = nil：返回 0 维张量 ----
          ((null size)
           (make-vt nil
                    (vt-cast (if cdf (sample-weighted-replace)
                                 (sample-uniform-replace))
                             out-dtype)
                    :dtype out-dtype))

          ;; ---- 无放回批量 ----
          ((not replace)
           (let* ((vals (if cdf
                            (sample-weighted-no-replace-batch size-total)
                            (sample-uniform-no-replace-batch size-total)))
                  (result (vt-zeros size-shape :dtype out-dtype))
                  (rdata (vt-data result)))
             (loop for v in vals for i from 0
                   do (setf (aref rdata i) (vt-cast v out-dtype)))
             result))

          ;; ---- 有放回批量 ----
          (t
           (let ((result (vt-zeros size-shape :dtype out-dtype)))
             (vt-do-each (ptr val result)
               (declare (ignore val))
               (setf (aref (vt-data result) ptr)
                     (vt-cast (if cdf (sample-weighted-replace)
                                  (sample-uniform-replace))
                              out-dtype)))
             result)))))))

(defun vt-random-permutation (n &key (rng nil))
  (setf rng (or rng (%copy-default-rng)))
  (when (and (integerp n) (<= n 1))
    (return-from vt-random-permutation
      (vt-arange (if (zerop n) 0 1) :dtype :int64)))
  (if (integerp n)
      (let* ((arr (vt-arange n :dtype :int64))
	     (data (vt-data arr)))
        (loop for i from (1- n) downto 1 do
          (let ((j (random (1+ i) rng)))
	    (rotatef (aref data i) (aref data j))))
        arr)
      (let* ((tensor (ensure-vt n)) (result (vt-copy tensor))
             (first-dim (first (vt-shape result))))
        (loop for i from (1- first-dim) downto 1 do
          (let* ((j (random (1+ i) rng))
                 (si (vt-slice result (list i)))
		 (sj (vt-slice result (list j))))
            (let ((tmp (vt-copy si)))
	      (vt-copy-into si sj)
	      (vt-copy-into sj tmp))))
        result)))

(defun vt-random-shuffle (tensor &key (axis 0) (rng nil))
  (setf rng (or rng (%copy-default-rng)))
  (let* ((ax (vt-normalize-axis axis (length (vt-shape tensor))))
         (dim (nth ax (vt-shape tensor))))
    (when (<= dim 1) (return-from vt-random-shuffle tensor))
    (loop for i from (1- dim) downto 1 do
      (let* ((j (random (1+ i) rng))
             (si (apply #'vt-slice tensor
                        (loop for d below (length (vt-shape tensor))
                              collect (if (= d ax) (list i) '(:all)))))
             (sj (apply #'vt-slice tensor
                        (loop for d below (length (vt-shape tensor))
                              collect (if (= d ax) (list j) '(:all))))))
        (let ((tmp (vt-copy si))) (vt-copy-into si sj) (vt-copy-into sj tmp))))
    tensor))
(defun vt-random-multinomial (n pvals &key (size nil) (rng nil))
  (setf rng (or rng (%copy-default-rng)))
  (let* ((probs (if (vt-p pvals) (vt-to-list pvals) (coerce pvals 'list)))
         (k (length probs)))
    (assert (> k 0) () "pvals 不能为空")
    (assert (every (lambda (p) (>= p 0)) probs) ())
    (let ((total (reduce #'+ probs)))
      (assert (> total 0) ())
      ;; 计算 CDF，用双精度，强制最后一项为 1.0 以防浮点误差
      (let ((cdf (make-array k :element-type 'double-float :initial-element 0.0d0)))
        (let ((cum 0.0d0))
          (loop for i from 0 below k
                for p in probs
                do (incf cum (/ p total))
                   (setf (aref cdf i) cum)))
        ;; 显式设置最后一项为 1.0，同时保留回退逻辑更保险
        (setf (aref cdf (1- k)) 1.0d0)

        (let* ((out-shape (if size
			      (append (if (listp size) size (list size))
				      (list k))
			      (list k)))
               (result (vt-zeros out-shape :dtype :int64))
               (res-data (vt-data result))
               (total-trials (if size
				 (reduce #'* (if (listp size) size (list size)))
				 1)))
          (dotimes (trial total-trials)
            (let ((counts (make-array k :element-type '(signed-byte 64) :initial-element 0)))
              (dotimes (_ n)
                (let ((r (random 1.0d0 rng)))
                  ;; 二分查找第一个 CDF >= r 的索引
                  (let ((idx (binary-search-cdf cdf r)))
                    (incf (aref counts idx)))))
              (loop for i from 0 below k
                    do (setf (aref res-data (+ (* trial k) i))
			     (aref counts i)))))
          result)))))

;; 辅助函数：二分查找
(defun binary-search-cdf (cdf r)
  (let ((lo 0) (hi (length cdf)))
    (loop while (< lo hi) do
      (let ((mid (floor (+ lo hi) 2)))
        (if (< (aref cdf mid) r)
            (setf lo (1+ mid))
            (setf hi mid))))
    ;; 回退：如果 r 大于所有 cdf，返回最后一个索引
    (min lo (1- (length cdf)))))
