;;;; 算术运算

(in-package #:clvt)

(defun vt-+ (&rest args)
  "逐元素加法. 支持标量、列表、张量混合.
   自动优化: 多参数时单次遍历."
  (apply #'vt-map #'+ args))

(defun vt-* (&rest args)
  "逐元素乘法. 支持标量、列表、张量混合."
  (apply #'vt-map #'* args))

(defun vt-- (vt &rest args)
  "逐元素减法.
   单参数: 取反.
   多参数: a - b - c ... "
  (let ((first-vt (ensure-vt vt)))
    (if (null args)
        (vt-map #'- first-vt)
        ;; 将 first-vt 放在最前面,一次性传入
        (apply #'vt-map #'- first-vt args))))

(defun vt-/ (vt &rest args)
  "逐元素除法.
   单参数: 倒数.
   多参数: a / b / c ... "
  (let ((first-vt (ensure-vt vt)))
    (if (null args)
        (vt-map (lambda (v) (/ 1.0d0 v)) first-vt)
        (apply #'vt-map #'/ first-vt args))))

(defun vt-scale (a b)
  "将张量 TENSOR 的所有元素乘以标量 SCALAR, 返回新的张量."
  (vt-* a b))

(defun vt-sin (vt)
  "计算逐元素正弦值."
  (vt-map #'sin vt))

(defun vt-cos (vt)
  "计算逐元素余弦值."
  (vt-map #'cos vt))
(defun vt-tan (vt)
  "计算逐元素正切值."
  (vt-map #'tan vt))

(defun vt-asin (vt)
  "计算逐元素反正弦值."
  (vt-map #'asin vt))

(defun vt-acos (vt)
  "计算逐元素反余弦值."
  (vt-map #'acos vt))

(defun vt-atan (vt)
  "计算逐元素反正切值."
  (vt-map #'atan vt))

(defun vt-sinh (vt)
  "计算逐元素双曲正弦值."
  (vt-map #'sinh vt))

(defun vt-cosh (vt)
  "计算逐元素双曲余弦值."
  (vt-map #'cosh vt))

(defun vt-tanh (vt)
  "计算逐元素双曲正切值."
  (vt-map #'tanh vt))

(defun vt-exp (vt)
  "计算逐元素 e^vt."
  (vt-map #'exp vt))

(defun vt-sqrt (vt)
  "计算逐元素平方根."
  (vt-map #'sqrt vt))

(defun vt-abs (vt)
  "计算逐元素绝对值."
  (vt-map #'abs vt))

(defun vt-signum (vt)
  "计算逐元素符号."
  (vt-map #'signum vt))


(defun vt-positive-p (vt)
  "判断是否大于零. 返回 1.0 (True) 或 0.0 (False)."
  (vt-map (lambda (v) (if (> v 0.0d0) 1.0d0 0.0d0)) vt))

(defun vt-negative-p (vt)
  "判断是否小于零. 返回 1.0 或 0.0."
  (vt-map (lambda (v) (if (< v 0.0d0) 1.0d0 0.0d0)) vt))

(defun vt-zero-p (vt)
  "判断是否为零. 返回 1.0 或 0.0."
  (vt-map (lambda (v) (if (zerop v) 1.0d0 0.0d0)) vt))

(defun vt-nonzero-p (vt)
  "判断是否非零. 返回 1.0 或 0.0."
  (vt-map (lambda (v) (if (zerop v) 0.0d0 1.0d0)) vt))

(defun vt-even-p (vt)
  "判断是否为偶数. 返回 1.0 或 0.0."
  ;; 注意:浮点数判断偶数需先取整
  (vt-map (lambda (v) (if (evenp (floor v)) 1.0d0 0.0d0)) vt))

(defun vt-odd-p (vt)
  "判断是否为奇数. 返回 1.0 或 0.0."
  (vt-map (lambda (v) (if (oddp (floor v)) 1.0d0 0.0d0)) vt))

(defun vt-expt (vt-base power-num)
  "逐元素幂运算."
  (vt-map (lambda (x) (expt x power-num)) vt-base))

(defun vt-mod (vt divisor)
  "逐元素取模."
  (vt-map (lambda (x) (mod x divisor)) vt))

(defun vt-rem (vt divisor)
  "逐元素取余数."
  (vt-map (lambda (x) (rem x divisor)) vt))

(defun vt-atan2 (vty x)
  "逐元素计算 atan2(y, x). 支持广播."
  (vt-map (lambda (y) (atan y x)) vty))

(defun vt-floor (vt &optional (divisor 1))
  "向下取整."
  (vt-map (lambda (x) (floor x divisor)) vt))

(defun vt-ceiling (vt &optional (divisor 1))
  "向上取整."
  (vt-map (lambda (x) (ceiling x divisor)) vt))

(defun vt-round (vt &optional (divisor 1))
  "四舍五入."
  (vt-map (lambda (x) (round x divisor)) vt))

(defun vt-rint (vt)
  "四舍五入到最接近的整数 (浮点数返回值)."
  ;; Common Lisp 没有直接的 rint,使用 round 然后 float 化
  (vt-map (lambda (x) (float (round x) x)) vt))

(defun vt-trancate (vt &optional (divisor 1))
  "向0取整"
  (vt-map (lambda (x) (truncate x divisor))))

(defun vt-log (vt &optional base)
  "以 base 为底的对数."
  (if base
      (vt-map (lambda (val)
		(log val base))
	      vt)
      (vt-map (lambda (val)
		(log val))
              vt)))

(defun vt-log10 (vt)
  "以 10 为底的对数."
  (vt-log vt 10.0d0))

(defun vt-log2 (vt)
  "以 2 为底的对数."
  (vt-log vt 2.0d0))
