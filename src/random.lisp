;;;; random.lisp — 随机数生成（NumPy 风格）
;;;;
;;;; 三层设计（对齐 NumPy）：
;;;;   1. vt-seed-sequence —— 处理任意 seed，支持 spawn 派生独立子序列
;;;;   2. vt-generator     —— 显式随机数实例，多线程安全的载体
;;;;   3. 全局状态          —— 向后兼容，单线程便利入口
;;;;
;;;; 三种使用模式：
;;;;   A. 全局（简用）：  (vt-random-seed 42) + vt-random-*
;;;;   B. 作用域隔离：    (with-seed (42) ...)
;;;;   C. 多线程并行：    (spawn-generators 42 N) + (with-generator (g) ...)

(in-package :clvt)

(defvar *vt-default-random-state* (make-random-state *random-state*)
  "clvt 内部默认随机状态。
   单线程可直接使用；多线程请用 spawn-generators + with-generator。")

(defvar *vt-random-state-lock*
  (sb-thread:make-mutex :name "vt-random-state")
  "保护 *vt-default-random-state* 的锁。
   仅 %copy-default-rng 内部使用。")

;;; NumPy 的 SeedSequence 核心功能：从任意 seed 派生 N 个独立子序列。

(defstruct (vt-seed-sequence (:constructor %make-vt-seed-sequence))
  (entropy 0 :type integer :read-only t))

(defun make-seed-sequence (seed)
  "构造 SeedSequence。
   SEED 可以是：
     整数                  → 直接用
     字节向量              → 直接用（作为 entropy 传给 SBCL）
     整数列表              → 打包成字节向量
     已有 SeedSequence     → 拷贝
   最终 entropy 一定是 SBCL 能接受的类型（整数或字节向量）。"
  (etypecase seed
    (integer          (%make-vt-seed-sequence :entropy seed))
    ((simple-array (unsigned-byte 8) (*))
     (%make-vt-seed-sequence :entropy seed))
    (list
     (let ((bytes (make-array (length seed)
                              :element-type '(unsigned-byte 8))))
       (loop for v in seed for i from 0
             do (setf (aref bytes i) (logand v #xFF)))
       (%make-vt-seed-sequence :entropy bytes)))
    (vt-seed-sequence (%make-vt-seed-sequence
                       :entropy (vt-seed-sequence-entropy seed)))))

(defun %seed-mix (entropy tag)
  "从 (ENTROPY, TAG) 派生确定性子 seed。splitmix64 finalizer。
   TAG 可以是整数或符号。返回 (unsigned-byte 32)。"
  (let* ((tag-int (typecase tag
                    (integer tag)
                    (symbol (logand (sxhash tag) #xFFFFFFFF))
                    (t (logand (sxhash (princ-to-string tag)) #xFFFFFFFF))))
         (x (logand (logxor (logand entropy #xFFFFFFFF)
                            tag-int
                            #x9E3779B9)
                    #xFFFFFFFF)))
    (setf x (logand (* (logxor x (ash x -16)) #x85EBCA6B) #xFFFFFFFF))
    (setf x (logand (* (logxor x (ash x -13)) #xC2B2AE35) #xFFFFFFFF))
    (setf x (logxor x (ash x -16)))
    (logand x #xFFFFFFFF)))

(defun seed-sequence-spawn (ss n)
  "派生 N 个独立子 SeedSequence。
   并行随机流的核心——每个 worker 一个子序列，随机流不重叠。"
  (loop for i below n
        collect (%make-vt-seed-sequence
                 :entropy (%seed-mix (vt-seed-sequence-entropy ss) i))))

(defun seed-sequence-generate-state (ss)
  "从 SeedSequence 生成一个 random-state。"
  (sb-ext:seed-random-state (vt-seed-sequence-entropy ss)))

;;; 显式随机数实例——多个 Generator 互不干扰。

(defstruct (vt-generator (:constructor %make-vt-generator (state)))
  (state nil :type random-state))

(defun make-generator (&optional seed)
  "构造 Generator。
   SEED 为 nil 时用系统熵源；为整数时严格可复现。"
  (if seed
      (%make-vt-generator (sb-ext:seed-random-state seed))
      (%make-vt-generator (make-random-state t))))

(defun generator-from-seed-sequence (ss)
  "从 SeedSequence 构造 Generator。"
  (%make-vt-generator (seed-sequence-generate-state ss)))

(defun spawn-generators (master-seed n)
  "从 MASTER-SEED 派生 N 个独立 Generator。
   等价于 NumPy: [default_rng(s) for s in SeedSequence(seed).spawn(n)]。

   用法（多线程）：
     (let ((workers (spawn-generators 42 4)))
       (loop for w in workers do
         (make-thread (lambda ()
                        (with-generator (w)
                          (do-work))))))"
  (let ((root (make-seed-sequence master-seed)))
    (mapcar #'generator-from-seed-sequence
            (seed-sequence-spawn root n))))

;;; rng 参数标准化
;;; 所有 vt-random-* 的 :rng 参数统一走这里——
;;; 接受 nil / random-state / vt-generator 三种形式。

(defun %ensure-random-state (rng)
  "把 :rng 参数标准化为 random-state。
   NIL          → 取全局快照（%copy-default-rng）
   random-state → 直接用
   vt-generator → 取内部的 state"
  (etypecase rng
    (null         (%copy-default-rng))
    (random-state rng)
    (vt-generator (vt-generator-state rng))))

(defmacro with-seed ((seed) &body body)
  "在 SEED 指定的随机状态下执行 BODY，离开后自动恢复全局。
   用法：(with-seed (42) (train-ppo :episodes 500))"
  `(let ((*vt-default-random-state* (vt-make-random-state ,seed)))
     ,@body))

(defmacro with-generator ((gen) &body body)
  "在 GENERATOR 指定的随机状态下执行 BODY。
   用法：
     (with-generator (my-gen)
       (vt-random-normal '(3)))"
  `(let ((*vt-default-random-state* (%ensure-random-state ,gen)))
     ,@body))

(defun vt-make-random-state (&optional seed)
  "从 SEED 构造 random-state。语义对齐 CL 的 MAKE-RANDOM-STATE：

   nil              → 复制当前 *random-state*（CL 默认行为）
   t                → 从系统熵源新建（不可复现）
   random-state     → 拷贝一份（不复用——避免共享可变状态）
   整数             → 从整数种子构造（SBCL 专有，严格可复现）
   vt-seed-sequence → 从其 entropy 构造（clvt 扩展）

   不支持的 SEED 类型报 type-error。"
  (typecase seed
    (null          (make-random-state nil))
    ((eql t)       (make-random-state t))
    (random-state  (make-random-state seed))       ; 拷贝，不共享
    (integer       (sb-ext:seed-random-state seed))
    (vt-seed-sequence
     (sb-ext:seed-random-state (vt-seed-sequence-entropy seed)))
    (t (error 'type-error
              :datum seed
              :expected-type '(or null (eql t) random-state integer
                               vt-seed-sequence)))))

(defun vt-random-seed (seed)
  "永久设置全局随机状态。
   SEED 接受 vt-make-random-state 支持的所有类型（透传 SBCL 契约 + SeedSequence）。
   ⚠ 推荐用 (with-seed (SEED) ...) 做作用域隔离。
   ⚠ 多线程下不要调用——用 spawn-generators + with-generator。"
  (sb-thread:with-mutex (*vt-random-state-lock*)
    (setf *vt-default-random-state* (vt-make-random-state seed)))
  seed)

(declaim (inline %copy-default-rng))
(defun %copy-default-rng ()
  "返回全局随机状态的快照，并推进全局状态一步。
   保证连续调用返回不同序列（避免同 seed 下连续 vt-random-* 返回相同结果）。"
  (sb-thread:with-mutex (*vt-random-state-lock*)
    (let ((snapshot (make-random-state *vt-default-random-state*)))
      (random 1.0d0 *vt-default-random-state*)
      snapshot)))


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
  (setf rng (%ensure-random-state rng))
  (vt-map (lambda (x)
            (declare (ignore x))
            (vt-cast (%uniform-rand rng) dtype))
          (vt-zeros shape :dtype dtype)))

(defun vt-random-uniform
    (shape &key (low 0.0d0) (high 1.0d0) (dtype :float64) (rng nil))
  (declare (list shape))
  (assert (< low high) (low high))
  (setf rng (%ensure-random-state rng))
  (let ((range (- high low)))
    (vt-map (lambda (x)
              (declare (ignore x))
              (vt-cast (+ low (* range (%uniform-rand rng))) dtype))
            (vt-zeros shape :dtype dtype))))

(defun vt-random-normal
    (shape &key (mean 0.0d0) (std 1.0d0) (dtype :float64) (rng nil))
  (declare (list shape))
  (setf rng (%ensure-random-state rng))
  (let ((res (vt-zeros shape :dtype dtype)))
    (vt-do-each (ptr val res)
      (declare (ignore val))
      (setf (aref (vt-data res) ptr)
            (vt-cast (+ mean (* std (%normal-rand rng))) dtype)))
    res))

(defun vt-random-int
    (low high &key (size nil) (dtype :int64) (rng nil))
  (setf rng (%ensure-random-state rng))
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
  (setf rng (%ensure-random-state rng))
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
   rng      : nil / random-state / vt-generator。
   对标 numpy.random.choice / torch.multinomial。"
  (setf rng (%ensure-random-state rng))
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
                                      :initial-contents
				      (loop for i below n collect i))))
               (loop for i from 0 below k do
                 (let ((j (+ i (random (- n i) rng))))
                   (rotatef (aref idx i) (aref idx j))))
               (loop for i from 0 below k
                     collect (aref src-data (+ src-offset (aref idx i))))))

           (sample-weighted-no-replace-batch (k)
             (let* ((weights
		      (coerce (or p (make-list n :initial-element 1.0d0))
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
          ((null size)
           (make-vt nil
                    (vt-cast (if cdf (sample-weighted-replace)
                                 (sample-uniform-replace))
                             out-dtype)
                    :dtype out-dtype))

          ((not replace)
           (let* ((vals (if cdf
                            (sample-weighted-no-replace-batch size-total)
                            (sample-uniform-no-replace-batch size-total)))
                  (result (vt-zeros size-shape :dtype out-dtype))
                  (rdata (vt-data result)))
             (loop for v in vals for i from 0
                   do (setf (aref rdata i) (vt-cast v out-dtype)))
             result))

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
  (setf rng (%ensure-random-state rng))
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
      (let* ((tensor (ensure-vt n))
	     (result (vt-copy tensor))
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
  (setf rng (%ensure-random-state rng))
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
        (let ((tmp (vt-copy si)))
	  (vt-copy-into si sj)
	  (vt-copy-into sj tmp))))
    tensor))

(defun vt-random-multinomial (n pvals &key (size nil) (rng nil))
  (setf rng (%ensure-random-state rng))
  (let* ((probs (if (vt-p pvals) (vt-to-list pvals) (coerce pvals 'list)))
         (k (length probs)))
    (assert (> k 0) () "pvals 不能为空")
    (assert (every (lambda (p) (>= p 0)) probs) ())
    (let ((total (reduce #'+ probs)))
      (assert (> total 0) ())
      (let ((cdf (make-array k :element-type 'double-float
			       :initial-element 0.0d0)))
        (let ((cum 0.0d0))
          (loop for i from 0 below k
                for p in probs
                do (incf cum (/ p total))
                   (setf (aref cdf i) cum)))
        (setf (aref cdf (1- k)) 1.0d0)

        (let* ((out-shape (if size
                              (append (if (listp size) size (list size))
                                      (list k))
                              (list k)))
               (result (vt-zeros out-shape :dtype :int64))
               (res-data (vt-data result))
               (total-trials (if size
                                 (reduce #'* (if (listp size)
						 size (list size)))
                                 1)))
          (dotimes (trial total-trials)
            (let ((counts (make-array k :element-type '(signed-byte 64)
					:initial-element 0)))
              (dotimes (_ n)
                (let ((r (random 1.0d0 rng)))
                  (let ((idx (binary-search-cdf cdf r)))
                    (incf (aref counts idx)))))
              (loop for i from 0 below k
                    do (setf (aref res-data (+ (* trial k) i))
                             (aref counts i)))))
          result)))))

(defun binary-search-cdf (cdf r)
  (let ((lo 0) (hi (length cdf)))
    (loop while (< lo hi) do
      (let ((mid (floor (+ lo hi) 2)))
        (if (< (aref cdf mid) r)
            (setf lo (1+ mid))
            (setf hi mid))))
    (min lo (1- (length cdf)))))

