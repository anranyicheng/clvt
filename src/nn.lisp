;;;; nn.lisp — 激活函数、损失函数与 softmax

(in-package :clvt)

;;; ------------------------------------------------------------------
;;; 激活函数 — 优化版：优先使用特化快路径避免通用lambda funcall开销
;;; ------------------------------------------------------------------

(declaim (inline %relu-double %relu-single %sigmoid-double %sigmoid-single))

;; ReLU：用 (< x 0.0d0) 判断。
;;   - x < 0  → 返回 0
;;   - x >= 0 → 返回 x
;;   - x = NaN → (< NaN 0.0d0) 为 nil（陷阱已屏蔽），返回 x 本身 = NaN ✔
;; 注意：NaN 参与比较要求调用方已屏蔽浮点陷阱。
(defun %relu-double (x)
  (declare (double-float x) (optimize (speed 3) (safety 0)))
  (if (< x 0.0d0) 0.0d0 x))

(defun %relu-single (x)
  (declare (single-float x) (optimize (speed 3) (safety 0)))
  (if (< x 0.0s0) 0.0s0 x))

;; Sigmoid：x = NaN 时走 else 分支，exp(NaN)=NaN，除法结果仍为 NaN，语义正确；
;;   但 NaN 参与 (>= x 0.0d0) 比较同样需要陷阱已屏蔽。
;;   数值稳定版的分支写法保持不变即可。
(defun %sigmoid-double (x)
  (declare (double-float x) (optimize (speed 3) (safety 0)))
  (if (>= x 0.0d0)
      (/ 1.0d0 (+ 1.0d0 (exp (- x))))
      (let ((e (exp x))) (/ e (+ 1.0d0 e)))))

(defun %sigmoid-single (x)
  (declare (single-float x) (optimize (speed 3) (safety 0)))
  (if (>= x 0.0s0)
      (/ 1.0s0 (+ 1.0s0 (exp (- x))))
      (let ((e (exp x))) (/ e (+ 1.0s0 e)))))

(defun vt-sigmoid (vt &key dtype out)
  "数值稳定 sigmoid。连续 float 张量走特化快路径。

  dtype 语义（§4.2 H3「严格相等」）：结果 dtype = 显式 :dtype，否则按输入
  浮点提升（非浮点 → :float64）。out 必须精确匹配结果 shape/dtype。"
  (with-float-safe
    (let* ((dt (or dtype (if (eq (vt-dtype vt) :float32) :float32 :float64)))
           (a (ensure-vt vt))
           ;; 统一硬契约前置：形状/dtype 严格相等/可写（非连续 out 允许，
           ;; 由下方 vt-map 通用路径正确处理 strides）
           (res (if out
                    (vt-check-out out (vt-shape a) dt :op-name "vt-sigmoid")
                    (make-vt (vt-shape a) 0 :dtype dt))))
      (if (and (vt-contiguous-p res)
	       (vt-contiguous-p a)
               (eq (vt-dtype a) dt)
	       (eq (vt-dtype res) dt)          
               (equal (vt-shape res) (vt-shape a))
               (member dt '(:float64 :float32)))
          (let* ((rd (vt-data res))
		 (ad (vt-data a))
		 (size (vt-size a))
		 (rop (vt-offset res))
		 (aop (vt-offset a)))
            (declare (fixnum size rop aop) (optimize (speed 3) (safety 0)))
            (if (eq dt :float64)
		(let ((r (the (simple-array double-float (*)) rd))
                      (d (the (simple-array double-float (*)) ad)))
                  (dotimes (i size)
                    (setf (aref r (+ rop i)) (%sigmoid-double (aref d (+ aop i))))))
		(let ((r (the (simple-array single-float (*)) rd))
                      (d (the (simple-array single-float (*)) ad)))
                  (dotimes (i size)
                    (setf (aref r (+ rop i)) (%sigmoid-single (aref d (+ aop i)))))))
            res)
          (vt-map (lambda (x) (if (>= x 0) (/ 1.0d0 (+ 1.0d0 (exp (- x))))
                                  (let ((e (exp x))) (/ e (+ 1.0d0 e)))))
                  a :dtype dt :out res)))))

(defun vt-relu (vt &key dtype out)
  "ReLU 激活。连续 float 张量走特化快路径（消除 funcall 开销）。
  dtype 语义同 vt-sigmoid（§4.2 H3「严格相等」）。"
  (with-float-safe
    (let* ((dt (or dtype (if (eq (vt-dtype vt) :float32) :float32 :float64)))
           (a (ensure-vt vt))
           (res (if out
                    (vt-check-out out (vt-shape a) dt :op-name "vt-relu")
                    (make-vt (vt-shape a) 0 :dtype dt))))
      (if (and (vt-contiguous-p res)
	       (vt-contiguous-p a)
               (eq (vt-dtype a) dt)
	       (eq (vt-dtype res) dt)
               (equal (vt-shape res) (vt-shape a))
               (member dt '(:float64 :float32)))
          (let* ((rd (vt-data res))
		 (ad (vt-data a))
		 (size (vt-size a))
		 (rop (vt-offset res))
		 (aop (vt-offset a)))
            (declare (fixnum size rop aop) (optimize (speed 3) (safety 0)))
            (if (eq dt :float64)
		(let ((r (the (simple-array double-float (*)) rd))
                      (d (the (simple-array double-float (*)) ad)))
                  (dotimes (i size)
                    (setf (aref r (+ rop i)) (%relu-double (aref d (+ aop i))))))
		(let ((r (the (simple-array single-float (*)) rd))
                      (d (the (simple-array single-float (*)) ad)))
                  (dotimes (i size)
                    (setf (aref r (+ rop i)) (%relu-single (aref d (+ aop i)))))))
            res)
	  (vt-map (if (eq dt :float64) #'%relu-double #'%relu-single)
		  a :dtype dt :out res)))))

(defun vt-leaky-relu (vt &key (alpha 0.01d0) dtype out)
  "Leaky ReLU 激活：x > 0 时为 x，否则为 alpha * x（ALPHA 缺省 0.01）。"
  (vt-map (lambda (x) (if (> x 0.0d0)
			  x
			  (* alpha x)))
	  vt :dtype dtype :out out))

(defun vt-swish (vt &key dtype out)
  "Swish / SiLU 激活：x * sigmoid(x)。"
  (let ((sig (vt-sigmoid vt :dtype dtype)))
    (vt-map #'* vt sig :dtype dtype :out out)))

(defun vt-softplus (vt &key dtype out)
  "Softplus 激活：log(1 + exp(x))。x > 20 时直接返回 x 以避免 exp 溢出。"
  (vt-map (lambda (x)
	    (if (> x 20.0d0)
		x
		(log (+ 1.0d0 (exp x)))))
	  vt :dtype dtype :out out))

(defun vt-gelu (vt &key dtype out)
  "GELU 激活（tanh 近似式，对标 PyTorch F.gelu 默认近似）。"
  (let ((c (sqrt (/ 2.0d0 (coerce pi 'double-float)))))
    (vt-map (lambda (x)
              (let* ((x3 (* x x x))
		     (inner (+ x (* 0.044715d0 x3)))
                     (tanh-val (tanh (* c inner))))
                (* 0.5d0 x (+ 1.0d0 tanh-val))))
            vt :dtype dtype :out out)))

(defun vt-mish (vt &key dtype out)
  "Mish 激活：x * tanh(softplus(x))。"
  (let ((sp (vt-softplus vt :dtype dtype)))
    (vt-* vt (vt-tanh sp) :dtype dtype :out out)))

(defun vt-hard-tanh (vt &key dtype out)
  "HardTanh 激活：把输入截断到 [-1, 1]。"
  (vt-clip vt -1.0d0 1.0d0 :dtype dtype :out out))

(defun vt-hard-sigmoid (vt &key dtype out)
  "HardSigmoid 激活：clip(x/5 + 0.5, 0, 1)。"
  (let ((scaled (vt-+ (vt-scale vt 0.2d0) 0.5d0 :dtype dtype)))
    (vt-clip scaled 0.0d0 1.0d0 :dtype dtype :out out)))

(defun vt-softmax (vt &key (axis -1) dtype out)
  "softmax（沿 axis，默认最后一维）。数值稳定化：先减去该轴最大值再 exp。
  注意（对标 PyTorch）：若某行全为 -Inf，则 max = -Inf，exp(-Inf - (-Inf)) = NaN，
  即全 -Inf 行的 softmax 结果为 NaN——这是 IEEE 754 下减去最大值稳定化的固有语义。"
  (let* ((max-val (vt-amax vt :axis axis :keepdims t :dtype dtype))
         (exp-vt (vt-exp (vt-- vt max-val :dtype dtype) :dtype dtype))
         (sum-exp (vt-sum exp-vt :axis axis :keepdims t :dtype dtype)))
    (vt-/ exp-vt sum-exp :dtype dtype :out out)))

(defun vt-log-softmax (vt &key (axis -1) dtype out)
  "log-softmax = x - logsumexp(x)，沿 axis（默认最后一维）。
  数值稳定化同 vt-softmax；结果 dtype 由输入提升 + 显式 :dtype 决定（§4.2 H3）。"
  (let* ((max-val (vt-amax vt :axis axis :keepdims t :dtype dtype))
         (shifted (vt-- vt max-val :dtype dtype))
         (lse (vt-log (vt-sum (vt-exp shifted :dtype dtype)
			      :axis axis :keepdims t :dtype dtype)
                      :dtype dtype)))
    (vt-- shifted lse :dtype dtype :out out)))

(defun vt-mean-squared-error (y-true y-pred &key dtype out)
  "均方误差：mean((y_true - y_pred)²)。结果为标量（float64）。"
  (vt-mean (vt-square (vt-- y-true y-pred :dtype dtype) :dtype dtype)
	   :dtype dtype :out out))

(defun vt-binary-cross-entropy (y-true y-pred &key (eps 1.0d-7) dtype out)
  "二元交叉熵（对标 torch.nn.BCELoss）。p 含 NaN 时损失为 NaN；EPS 用于裁剪 log 的自变量避免 log(0)。"
  ;; NaN 语义对标 torch.BCELoss：p 含 NaN 时损失必须为 NaN，
  ;; 裁剪前显式判断（min/max 在 NaN 上比较恒 nil 会把 NaN 静默替换为 1-eps）。
  (vt-mean (vt-map (lambda (y p)
                     (let* ((pc (if (and (floatp p) (not (= p p)))
                                    p
                                    (max eps (min (- 1.0d0 eps) p))))
                            (omp (if (and (floatp pc) (not (= pc pc)))
                                     pc
                                     (max eps (- 1.0d0 pc)))))
                       (- (+ (* y (log pc)) (* (- 1.0d0 y) (log omp))))))
                   y-true y-pred :dtype dtype)
           :dtype dtype :out out))

(defun vt-cross-entropy (y-true y-pred &key (eps 1.0d-7) dtype out)
  "交叉熵（对标 torch.nn.CrossEntropyLoss）。逐样本取 -Σ y·log(p) 后取均值；EPS 同 vt-binary-cross-entropy。"
  (let* ((p-clipped (vt-clip y-pred eps (- 1.0d0 eps) :dtype dtype))
         (log-prob (vt-log p-clipped :dtype dtype))
         (loss-per-sample (vt-- (vt-sum (vt-* y-true log-prob :dtype dtype)
					:axis -1 :dtype dtype)
                                :dtype dtype)))
    (vt-mean loss-per-sample :dtype dtype :out out)))
