;;;; random.lisp — 随机数生成

(in-package :clvt)

(defvar *vt-default-random-state* (make-random-state *random-state*)
  "clvt 内部默认随机状态。")

(defun vt-make-random-state (&optional seed)
  #+sbcl (if (null seed) (make-random-state nil) (sb-ext::seed-random-state seed))
  #-sbcl (typecase seed
           (null (make-random-state nil))
           (random-state (make-random-state seed))
           (t (make-random-state seed))))

(defun vt-random-seed (seed)
  (setf *vt-default-random-state* (vt-make-random-state seed)))

(declaim (inline %uniform-rand %normal-rand))
(defun %uniform-rand (state)
  (random 1.0d0 state))
(defun %normal-rand (state)
  (let ((u1 (max least-positive-double-float (random 1.0d0 state)))
        (u2 (random 1.0d0 state)))
    (* (sqrt (* -2.0d0 (log u1))) (cos (* 2.0d0 pi u2)))))

(defun vt-random (shape &key (dtype :float64) (rng *vt-default-random-state*))
  (declare (list shape) (random-state rng))
  (vt-map (lambda (x) (declare (ignore x)) (vt-cast (%uniform-rand rng) dtype))
          (vt-zeros shape :dtype dtype)))

(defun vt-random-uniform (shape &key (low 0.0d0) (high 1.0d0) (dtype :float64)
                                  (rng *vt-default-random-state*))
  (declare (list shape) (random-state rng))
  (assert (< low high) (low high))
  (let ((range (- high low)))
    (vt-map (lambda (x) (declare (ignore x)) (vt-cast (+ low (* range (%uniform-rand rng))) dtype))
            (vt-zeros shape :dtype dtype))))

(defun vt-random-normal (shape &key (mean 0.0d0) (std 1.0d0) (dtype :float64)
                                 (rng *vt-default-random-state*))
  (declare (list shape) (random-state rng))
  (let ((res (vt-zeros shape :dtype dtype)))
    (vt-do-each (ptr val res)
      (declare (ignore val))
      (setf (aref (vt-data res) ptr) (vt-cast (+ mean (* std (%normal-rand rng))) dtype)))
    res))

(defun vt-random-int (low high &key (size nil) (dtype :int64) (rng *vt-default-random-state*))
  (declare (random-state rng))
  (let ((range (- high low)))
    (assert (>= range 0) (high low))
    (if (zerop range)
        (if size (vt-full size low :dtype dtype) (make-vt nil low :dtype dtype))
        (if size
            (vt-astype (vt-map (lambda (x) (declare (ignore x)) (+ low (random range rng)))
                               (vt-zeros size :dtype dtype))
                       dtype)
            (make-vt nil (+ low (random range rng)) :dtype dtype)))))

(defun vt-random-integers (low high &key (size nil) (dtype :int64) (rng *vt-default-random-state*))
  (vt-random-int low high :size size :dtype dtype :rng rng))

(defun vt-random-choice (a &key (size nil) (replace t) (p nil) (dtype nil)
                             (rng *vt-default-random-state*))
  (declare (random-state rng))
  (let* ((source (if (integerp a)
                     (progn (assert (> a 0) (a)) (vt-arange a :dtype (or dtype :int64)))
                     (ensure-vt a)))
         (n (vt-size source)) (out-dtype (or dtype (vt-dtype source)))
         (src-data (vt-data source)) (src-offset (vt-offset source)))
    (assert (> n 0) (n))
    (let ((cdf (if p
                   (progn (assert (= (length p) n) (p n))
                          (assert (every (lambda (w) (>= w 0)) p) ())
                          (let ((total (reduce #'+ p)))
                            (assert (> total 0) ())
                            (let ((cum 0.0d0))
                              (coerce (mapcar (lambda (w) (incf cum (/ w total)) cum) p) 'vector))))
                   nil)))
      (labels ((sample-one ()
                 (cond ((and replace (null p)) (aref src-data (+ src-offset (random n rng))))
                       ((and replace p)
                        (let ((r (random 1.0d0 rng)))
                          (loop for i from 0 below n
                                when (<= r (aref cdf i)) return (aref src-data (+ src-offset i))
                                finally (return (aref src-data (+ src-offset (1- n)))))))
                       (t (error "vt-random-choice: replace=nil 暂不支持")))))
        (if size
            (let ((result (vt-zeros size :dtype out-dtype)))
              (vt-do-each (ptr val result)
                (declare (ignore val))
                (setf (aref (vt-data result) ptr) (vt-cast (sample-one) out-dtype)))
              result)
            (make-vt nil (vt-cast (sample-one) out-dtype) :dtype out-dtype))))))

(defun vt-random-permutation (n &key (rng *vt-default-random-state*))
  (declare (random-state rng))
  (when (and (integerp n) (<= n 1))
    (return-from vt-random-permutation (vt-arange (if (zerop n) 0 1) :dtype :int64)))
  (if (integerp n)
      (let* ((arr (vt-arange n :dtype :int64)) (data (vt-data arr)))
        (loop for i from (1- n) downto 1 do
          (let ((j (random (1+ i) rng))) (rotatef (aref data i) (aref data j))))
        arr)
      (let* ((tensor (ensure-vt n)) (result (vt-copy tensor))
             (first-dim (first (vt-shape result))))
        (loop for i from (1- first-dim) downto 1 do
          (let* ((j (random (1+ i) rng))
                 (si (vt-slice result (list i))) (sj (vt-slice result (list j))))
            (let ((tmp (vt-copy si))) (vt-copy-into si sj) (vt-copy-into sj tmp))))
        result)))

(defun vt-random-shuffle (tensor &key (axis 0) (rng *vt-default-random-state*))
  (declare (random-state rng))
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
(defun vt-random-multinomial (n pvals &key (size nil) (rng *vt-default-random-state*))
  (declare (random-state rng))
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

        (let* ((out-shape (if size (append (if (listp size) size (list size)) (list k)) (list k)))
               (result (vt-zeros out-shape :dtype :int64))
               (res-data (vt-data result))
               (total-trials (if size (reduce #'* (if (listp size) size (list size))) 1)))
          (dotimes (trial total-trials)
            (let ((counts (make-array k :element-type '(signed-byte 64) :initial-element 0)))
              (dotimes (_ n)
                (let ((r (random 1.0d0 rng)))
                  ;; 二分查找第一个 CDF >= r 的索引
                  (let ((idx (binary-search-cdf cdf r)))
                    (incf (aref counts idx)))))
              (loop for i from 0 below k
                    do (setf (aref res-data (+ (* trial k) i)) (aref counts i)))))
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
