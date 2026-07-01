(in-package :clvt)

(defun vt-sigmoid (vt &key dtype out)
  "sigmoid: 1 / (1 + exp(-x)) —— 数值稳定版"
  (with-float-safe 
    (vt-map (lambda (x)
              (if (>= x 0)
                  (/ 1.0d0 (+ 1.0d0 (exp (- x))))
                  (let ((exp-x (exp x)))
                    (/ exp-x (+ 1.0d0 exp-x)))))
            vt :dtype dtype :out out)))

(defun vt-relu (vt &key dtype out)
  "relu: max(0, x)"
  (with-float-safe
    (vt-map (lambda (x) (max 0.0d0 x)) vt :dtype dtype :out out)))

(defun vt-leaky-relu (vt &key (alpha 0.01d0) dtype out)
  "leaky relu: max(alpha * x, x)"
  (with-float-safe
    (vt-map (lambda (x)
              (if (> x 0.0d0) (* x 1.0d0) (* alpha x)))
            vt :dtype dtype :out out)))

(defun vt-swish (vt &key dtype out)
  "swish (silu): x * sigmoid(x)"
  (with-float-safe
    (let ((sig (vt-sigmoid vt :dtype dtype)))
      (vt-map #'* vt sig :dtype dtype :out out))))

(defun vt-softplus (vt &key dtype out)
  "softplus: log(1 + exp(x))
   加入数值稳定性处理: 当 x > 20 时，直接返回 x，避免 exp 溢出"
  (with-float-safe
    (vt-map (lambda (x) 
              (if (> x 20.0d0) 
                  (* x 1.0d0)
                  (* 1.0d0 (log (+ 1.0d0 (exp x))))))
            vt :dtype dtype :out out)))

(defun vt-gelu (vt &key dtype out)
  "gelu (高斯误差线性单元): 使用 tanh 近似，pytorch 默认方式
   0.5 * x * (1 + tanh( sqrt(2/pi) * (x + 0.044715 * x^3)))"
  (let ((c (sqrt (/ 2.0d0 (coerce pi 'double-float)))))
    (with-float-safe
      (vt-map (lambda (x)
                (let* ((x3 (* x x x))
                       (inner (+ x (* 0.044715d0 x3)))
                       (tanh-val (tanh (* c inner))))
                  (* 0.5d0 x (+ 1.0d0 tanh-val))))
              vt :dtype dtype :out out))))

(defun vt-mish (vt &key dtype out)
  "mish: x * tanh(softplus(x))"
  (with-float-safe
    (let ((sp (vt-softplus vt :dtype dtype)))
      (vt-* vt (vt-tanh sp) :dtype dtype :out out))))

(defun vt-hard-tanh (vt &key dtype out)
  "hard tanh: clamp(x, -1, 1)"
  (with-float-safe
    (vt-clip vt -1.0d0 1.0d0 :dtype dtype :out out)))

(defun vt-hard-sigmoid (vt &key dtype out)
  "hard sigmoid: 快速分段线性近似"
  (with-float-safe
    (let* ((scaled (vt-+ (vt-scale vt 0.2d0) 0.5d0 :dtype dtype)))
      (vt-clip scaled 0.0d0 1.0d0 :dtype dtype :out out))))
