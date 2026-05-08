(in-package :clvt)

(defun vt-sigmoid (vt)
  "sigmoid: 1 / (1 + exp(-x))"
  (with-float-safe
    (vt-map (lambda (x)
	      (/ 1.0d0 (+ 1.0d0 (exp (- x)))))
	    vt)))

(defun vt-relu (vt)
  "relu: max(0, x)"
  (with-float-safe
    (vt-map (lambda (x) (max 0.0d0 x)) vt)))

(defun vt-leaky-relu (vt &key (alpha 0.01d0))
  "leaky relu: max(alpha * x, x)"
  (with-float-safe
    (vt-map (lambda (x)
	      (if (> x 0.0d0) x (* alpha x)))
	    vt)))

(defun vt-swish (vt)
  "swish (silu): x * sigmoid(x)"
  (with-float-safe
    (let ((sig (vt-sigmoid vt)))
      (vt-map #'* vt sig))))

(defun vt-softplus (vt)
  "softplus: log(1 + exp(x))
 加入数值稳定性处理: 当 x > 20 时，直接返回 x，避免 exp 溢出"
  (with-float-safe
    (vt-map (lambda (x) 
              (if (> x 20.0d0) 
                  x 
                  (log (+ 1.0d0 (exp x)))))
            vt)))

(defun vt-gelu (vt)
  "gelu (高斯误差线性单元): 使用 tanh 近似，pytorch 默认方式
   0.5 * x * (1 + tanh( sqrt(2/pi) * (x + 0.044715 * x^3)))"
  (with-float-safe
    (let* ((c (sqrt (/ 2.0d0 (coerce pi 'double-float))))
           (x-cubed (vt-expt vt 3.0d0))
           (inner (vt-+ vt (vt-scale x-cubed 0.044715d0)))
           (tanh-val (vt-tanh (vt-scale inner c))))
      (vt-scale (vt-* vt
		      (vt-+ (vt-ones-like vt) tanh-val))
		0.5d0))))

(defun vt-mish (vt)
  "mish: x * tanh(softplus(x))"
  (with-float-safe
    (let ((sp (vt-softplus vt)))
      (vt-* vt (vt-tanh sp)))))

(defun vt-hard-tanh (vt)
  "hard tanh: clamp(x, -1, 1)"
  (with-float-safe
    (vt-clip vt -1.0d0 1.0d0)))

(defun vt-hard-sigmoid (vt)
  "hard sigmoid: 快速分段线性近似"
  (with-float-safe
    (let* ((scaled (vt-+ (vt-scale vt 0.2d0) 0.5d0))
           (clipped (vt-clip scaled 0.0d0 1.0d0)))
      clipped)))
