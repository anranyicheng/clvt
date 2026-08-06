(in-package :clvt)

;;; 随机数扩展
(defvar *vt-default-random-state* (make-random-state *random-state*)
  "clvt 内部使用的默认随机状态。可由 vt-random-seed 修改或通过 let 覆盖。")

(defun vt-make-random-state (&optional seed)
  "创建并返回一个新的 random-state 对象。seed 同 make-random-state 参数。
   示例：
     (vt-make-random-state)     => 使用 t 随机初始化
     (vt-make-random-state nil) => 从当前 *random-state* 复制
     (vt-make-random-state 42)  => 不可移植地使用整数初始化"
  #+sbcl
  (if (null seed)
      (make-random-state nil)  ; nil => copy current state
      (sb-ext::seed-random-state seed))
  #-sbcl
  (typecase seed
    (null (make-random-state nil)) ; 保持与 cl 一致：nil → 复制当前全局状态
    (random-state (make-random-state seed))
    (t (make-random-state seed)))  ; 实现依赖，通常可为整数加载
  )

(defun vt-random-seed (seed)
  "修改 *vt-default-random-state* 并返回新的状态。seed 含义同 vt-make-random-state。
   例：(vt-random-seed 42)  后，所有使用默认 rng 的函数将产生可复现序列。"
  (setf *vt-default-random-state* (vt-make-random-state seed)))

(declaim (inline %uniform-rand %normal-rand))

(defun %uniform-rand (state)
  "生成 [0,1) 均匀分布随机数。"
  (declare (random-state state))
  (random 1.0d0 state))

(defun %normal-rand (state)
  "box-muller 方法生成标准正态分布随机数。"
  (declare (random-state state))
  (let ((u1 (max least-positive-double-float (random 1.0d0 state)))
        (u2 (random 1.0d0 state)))
    (* (sqrt (* -2.0d0 (log u1)))
       (cos (* 2.0d0 pi u2)))))

(defun vt-random (shape &key (dtype :float64)
			  (rng *vt-default-random-state*))
  "返回形状为 shape 的张量，元素独立同分布于 [0,1) 均匀分布。"
  (declare (list shape) (random-state rng))
  (vt-map (lambda (x)
            (declare (ignore x))
            (vt-cast (%uniform-rand rng) dtype))
          (vt-zeros shape :dtype dtype)))

(defun vt-random-uniform
    (shape &key (low 0.0d0) (high 1.0d0) (dtype :float64)
	     (rng *vt-default-random-state*))
  "返回 [low, high) 均匀分布随机张量。"
  (declare (list shape) (random-state rng))
  (assert (< low high) (low high)
          "vt-random-uniform: low (~a) must be less than high (~a)" low high)
  (let ((range (- high low)))
    (vt-map (lambda (x)
              (declare (ignore x))
              (vt-cast (+ low (* range (%uniform-rand rng))) dtype))
            (vt-zeros shape :dtype dtype))))

(defun vt-random-normal (shape &key (mean 0.0d0) (std 1.0d0)
				 (dtype :float64)
				 (rng *vt-default-random-state*))
  "返回正态分布随机张量。支持 :dtype 指定输出类型。"
  (declare (list shape) (random-state rng))
  (let ((res (vt-zeros shape :dtype dtype)))
    (vt-do-each (ptr val res)
      (declare (ignore val))
      (setf (aref (vt-data res) ptr)
            (vt-cast (+ mean (* std (%normal-rand rng))) dtype)))
    res))

(defun vt-random-int (low high &key (size nil) (dtype :int64)
				 (rng *vt-default-random-state*))
  "创建随机整数数组.
  low: 下界(包含)
  high: 上界(不包含)
  size: 形状(nil 表示标量)
  返回: 张量"
  (declare (random-state rng))
  (let ((range (- high low)))
    (assert (>= range 0)
	    (high low)
	    "high: ~a less than low: ~a" high low)
    ;; When range is 0 (high == low), (random 0) would signal an error
    ;; because CL's random requires a positive argument. Return constant low.
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

(defun vt-random-integers (low high &key (size nil) (dtype :int64)
				      (rng *vt-default-random-state*))
  "同 vt-random-int。"
  (vt-random-int low high :size size :dtype dtype :rng rng))

(defun vt-random-choice (a &key (size nil) (replace t) (p nil)
				 (dtype nil) (rng *vt-default-random-state*))
  "从张量 a 中随机抽取元素。对标 NumPy 的 np.random.choice。
   a: 整数 (视为 arange(a)) 或张量。
   size: 输出形状 (nil 表示标量)。
   replace: 是否可重复抽取 (默认 t)。
   p: 每个元素的概率权重 (必须与 a 等长，总和为 1)。
   dtype: 输出类型 (默认与 a 相同)。
   rng: 随机状态。"
  (declare (random-state rng))
  (let* ((source (if (integerp a)
                     (progn
                       (assert (> a 0) (a)
                               "vt-random-choice: integer a must be positive, got ~a" a)
                       (vt-arange a :dtype (or dtype :int64)))
                     (ensure-vt a)))
         (n (vt-size source))
         (out-dtype (or dtype (vt-dtype source)))
         (src-data (vt-data source))
         (src-offset (vt-offset source)))
    (assert (> n 0) (n) "vt-random-choice: source must have at least one element")
    ;; Build CDF from probability weights
    (let ((cdf (if p
                   (progn
                     (assert (= (length p) n) (p n)
                             "vt-random-choice: p length (~a) must match source size (~a)"
                             (length p) n)
                     (assert (every (lambda (w) (>= w 0)) p) ()
                             "vt-random-choice: all probabilities must be non-negative")
                     (let ((total (reduce #'+ p)))
                       (assert (> total 0) ()
                               "vt-random-choice: probabilities sum to zero")
                       (let ((cum 0.0d0))
                         (coerce (mapcar (lambda (w)
                                           (incf cum (/ w total))
                                           cum)
                                         p)
                                 'vector))))
                   nil)))
      ;; Sampling function
      (labels ((sample-one ()
                 (cond
                   ;; With replacement + no weights: simple random index
                   ((and replace (null p))
                    (aref src-data (+ src-offset (random n rng))))
                   ;; With replacement + weights: use CDF
                   ((and replace p)
                    (let ((r (random 1.0d0 rng)))
                      (loop for i from 0 below n
                            when (<= r (aref cdf i))
                              return (aref src-data (+ src-offset i))
                            finally (return (aref src-data (+ src-offset (1- n)))))))
                   ;; Without replacement: Fisher-Yates partial shuffle
                   (t
                    (error "vt-random-choice: replace=nil 暂不支持，请使用 vt-random-permutation")))))
        (if size
            (let ((result (vt-zeros size :dtype out-dtype)))
              (vt-do-each (ptr val result)
                (declare (ignore val))
                (setf (aref (vt-data result) ptr)
                      (vt-cast (sample-one) out-dtype)))
              result)
            (make-vt nil (vt-cast (sample-one) out-dtype) :dtype out-dtype))))))

(defun vt-random-permutation (n &key (rng *vt-default-random-state*))
  "返回 0..n-1 的随机排列。对标 NumPy 的 np.random.permutation。
   n: 整数 (返回 arange(n) 的排列) 或张量 (返回其副本的随机排列)。"
  (declare (random-state rng))
  (when (and (integerp n) (<= n 1))
    (return-from vt-random-permutation
      (if (zerop n) (vt-arange 0 :dtype :int64) (vt-arange 1 :dtype :int64))))
  (if (integerp n)
      ;; Integer case: return shuffled arange
      (let* ((arr (vt-arange n :dtype :int64))
             (data (vt-data arr)))
        ;; Fisher-Yates shuffle
        (loop for i from (1- n) downto 1 do
          (let ((j (random (1+ i) rng)))
            (rotatef (aref data i) (aref data j))))
        arr)
      ;; Tensor case: shuffle a copy along first axis
      (let* ((tensor (ensure-vt n))
             (result (vt-copy tensor))
             (first-dim (first (vt-shape result)))
             (rest-stride (if (> (length (vt-shape result)) 1)
                              (reduce #'* (rest (vt-shape result)))
                              1)))
        ;; Shuffle slices along first axis
        (loop for i from (1- first-dim) downto 1 do
          (let* ((j (random (1+ i) rng))
                 (slice-i (vt-slice result (list i)))
                 (slice-j (vt-slice result (list j))))
            ;; Swap by copying data
            (let ((tmp (vt-copy slice-i)))
              (vt-copy-into slice-i slice-j)
              (vt-copy-into slice-j tmp))))
        result)))

(defun vt-random-shuffle (tensor &key (axis 0) (rng *vt-default-random-state*))
  "就地随机打乱张量沿指定轴的顺序。对标 NumPy 的 np.random.shuffle。
   注意：就地修改！返回修改后的张量。"
  (declare (random-state rng))
  (let* ((ax (vt-normalize-axis axis (length (vt-shape tensor))))
         (dim (nth ax (vt-shape tensor))))
    (when (<= dim 1) (return-from vt-random-shuffle tensor))
    ;; Fisher-Yates shuffle along the axis
    (loop for i from (1- dim) downto 1 do
      (let* ((j (random (1+ i) rng))
             (slice-i (apply #'vt-slice tensor
                             (loop for d below (length (vt-shape tensor))
                                   collect (if (= d ax) (list i) '(:all)))))
             (slice-j (apply #'vt-slice tensor
                             (loop for d below (length (vt-shape tensor))
                                   collect (if (= d ax) (list j) '(:all))))))
        (let ((tmp (vt-copy slice-i)))
          (vt-copy-into slice-i slice-j)
          (vt-copy-into slice-j tmp))))
    tensor))

(defun vt-random-multinomial (n pvals &key (size nil) (rng *vt-default-random-state*))
  "从多项分布中抽取样本。对标 NumPy 的 np.random.multinomial。
   n: 试验次数。
   pvals: 概率权重列表或向量 (不需要归一化，会自动归一化)。
   size: 输出形状 (默认为单次试验)。
   返回: 形状为 (*size, len(pvals)) 的整数张量。"
  (declare (random-state rng))
  (let* ((probs (if (vt-p pvals)
                    (vt-to-list pvals)
                    (coerce pvals 'list)))
         (k (length probs))
         (dummy1 (assert (> k 0) () "vt-random-multinomial: pvals must not be empty"))
         (dummy1 (assert (every (lambda (p) (>= p 0)) probs) ()
                    "vt-random-multinomial: all probabilities must be non-negative"))
         (total (reduce #'+ probs))
         (dummy2 (assert (> total 0) () "vt-random-multinomial: probabilities sum to zero"))
         ;; Normalize and build CDF
         (cdf (let ((cum 0.0d0))
                (coerce (mapcar (lambda (p)
                                  (incf cum (/ p total))
                                  cum)
                                probs)
                        'vector)))
         ;; Output shape
         (out-shape (if size
                        (append (if (listp size) size (list size)) (list k))
                        (list k)))
         (result (vt-zeros out-shape :dtype :int64))
         (res-data (vt-data result)))
    ;; For each trial set, draw n samples and count
    (let ((total-trials (if size
                            (reduce #'* (if (listp size) size (list size)))
                            1)))
      (dotimes (trial total-trials)
        (let ((counts (make-array k :element-type '(signed-byte 64) :initial-element 0)))
          ;; Draw n samples
          (dotimes (_ n)
            (let ((r (random 1.0d0 rng)))
              (loop for i from 0 below k
                    when (<= r (aref cdf i))
                      do (incf (aref counts i))
                         (return))))
          ;; Write counts to result
          (loop for i from 0 below k do
            (setf (aref res-data (+ (* trial k) i))
                  (aref counts i))))))
    result))

