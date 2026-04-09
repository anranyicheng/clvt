(in-package :clvt)

;; Sigmoid: 1 / (1 + exp(-x))
(defun vt-sigmoid (vt)
  (vt-map (lambda (x) (/ 1.0d0 (+ 1.0d0 (exp (- x))))) vt))

;; ReLU: max(0, x)
(defun vt-relu (vt)
  (vt-map (lambda (x) (max 0.0d0 x)) vt))

;; Leaky ReLU: max(alpha * x, x)
(defun vt-leaky-relu (vt &optional (alpha 0.01d0))
  (vt-map (lambda (x) (if (> x 0.0d0) x (* alpha x))) vt))

;; Swish (SiLU): x * sigmoid(x)
(defun vt-swish (vt)
  (let ((sig (vt-sigmoid vt)))
    (vt-map #'* vt sig)))

;; Softplus: log(1 + exp(x))
;; 加入数值稳定性处理：当 x > 20 时，直接返回 x，避免 exp 溢出
(defun vt-softplus (vt)
  (vt-map (lambda (x) 
            (if (> x 20.0d0) 
                x 
                (log (+ 1.0d0 (exp x)))))
          vt))

;; GELU (高斯误差线性单元): 使用 Tanh 近似，PyTorch 默认方式
;; 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
(defun vt-gelu (vt)
  (let* ((c (sqrt (/ 2.0d0 (coerce pi 'double-float))))
         (x-cubed (vt-expt vt 3.0d0))
         (inner (vt-+ vt (vt-scale x-cubed 0.044715d0)))
         (tanh-val (vt-tanh (vt-scale inner c))))
    (vt-scale (vt-* vt (vt-+ (vt-ones-like vt) tanh-val)) 0.5d0)))

;; Mish: x * tanh(softplus(x))
(defun vt-mish (vt)
  (let ((sp (vt-softplus vt)))
    (vt-* vt (vt-tanh sp))))

;; Hard Tanh: clamp(x, -1, 1)
(defun vt-hard-tanh (vt)
  (vt-clip vt -1.0d0 1.0d0))

;; Hard Sigmoid: 快速分段线性近似
(defun vt-hard-sigmoid (vt)
  (let* ((scaled (vt-+ (vt-scale vt 0.2d0) 0.5d0))
         (clipped (vt-clip scaled 0.0d0 1.0d0)))
    clipped))
