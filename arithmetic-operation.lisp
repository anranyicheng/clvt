;;;; 算术运算

(in-package #:clvt)

(defun vt-+ (&rest args)
  "逐元素加法. 支持标量、列表、张量混合.
   自动优化: 多参数时单次遍历."
  (with-float-safe
    (apply #'vt-map #'+ args)))

(defun vt-add (&rest args)
  (apply #'vt-+ args))

(defun vt-* (&rest args)
  "逐元素乘法. 支持标量、列表、张量混合."
  (with-float-safe
    (apply #'vt-map #'* args)))

(defun vt-mul (&rest args)
  (apply #'vt-* args))

(defun vt-- (vt &rest args)
  "逐元素减法.
   单参数: 取反.
   多参数: a - b - c ... "
  (with-float-safe
    (let ((first-vt (ensure-vt vt)))
      (if (null args)
          (vt-map #'- first-vt)
          (apply #'vt-map #'- first-vt args)))))

(defun vt-sub (vt &rest args)
  (apply #'vt-- vt args))

(defun vt-/ (vt &rest args)
  "逐元素除法. 单参数: 倒数. 多参数: a / b / c ... "
  (with-float-safe
    (let ((first-vt (ensure-vt vt)))
      (if (null args)
          (vt-map (lambda (v) (/ 1.0d0 v)) first-vt)
          (reduce (lambda (acc x) (vt-map #'/ acc (ensure-vt x))) 
                  args :initial-value first-vt)))))

(defun vt-div (vt &rest args)
  (apply #'vt-/ vt args))

(defun vt-scale (a b)
  "将张量 tensor 的所有元素乘以标量 scalar, 返回新的张量."
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

(defun vt-pow (vt power)
  "计算逐元素幂次方"
  (with-float-safe
    (vt-map (lambda (x) (expt x power)) vt)))

(defun vt-square (vt)
  "计算逐元素平方"
  (vt-pow vt 2))

(defun vt-sqrt (vt)
  "计算逐元素平方根."
   (vt-map (lambda (x) 
            (if (minusp x) 
                +vt-float-nan+
                (sqrt x)))
	   vt))

(defun vt-abs (vt)
  "计算逐元素绝对值."
  (vt-map #'abs vt))

(defun vt-signum (vt)
  "计算逐元素符号."
  (vt-map #'signum vt))


(defun vt-positive-p (vt)
  "判断是否大于零. 返回 1.0 (true) 或 0.0 (false)."
  (with-float-safe
    (vt-map (lambda (v) (if (> v 0.0d0) 1.0d0 0.0d0)) vt)))

(defun vt-negative-p (vt)
  "判断是否小于零. 返回 1.0 或 0.0."
  (with-float-safe
    (vt-map (lambda (v) (if (< v 0.0d0) 1.0d0 0.0d0)) vt)))

(defun vt-zero-p (vt)
  "判断是否为零. 返回 1.0 或 0.0."
  (with-float-safe
    (vt-map (lambda (v) (if (zerop v) 1.0d0 0.0d0)) vt)))

(defun vt-nonzero-p (vt)
  "判断是否非零. 返回 1.0 或 0.0."
  (with-float-safe
    (vt-map (lambda (v) (if (zerop v) 0.0d0 1.0d0)) vt)))

(defun vt-even-p (vt)
  "判断是否为偶数. 返回 1.0 或 0.0."
  ;; 注意:浮点数判断偶数需先取整
  (with-float-safe
    (vt-map (lambda (v) (if (evenp (floor v)) 1.0d0 0.0d0)) vt)))

(defun vt-odd-p (vt)
  "判断是否为奇数. 返回 1.0 或 0.0."
  (with-float-safe
    (vt-map (lambda (v) (if (oddp (floor v)) 1.0d0 0.0d0)) vt)))

(defun vt-expt (vt power-num)
  "逐元素幂运算."
  (with-float-safe
    (vt-map (lambda (x) (expt x power-num)) vt)))

(defun vt-mod (vt divisor)
  "逐元素取模."
  (with-float-safe
    (vt-map (lambda (x) (mod x divisor)) vt)))

(defun vt-rem (vt divisor)
  "逐元素取余数."
  (with-float-safe
    (vt-map (lambda (x) (rem x divisor)) vt)))

(defun vt-atan2 (vty vtx)
  "逐元素计算 atan2(y, x). 支持广播.
   参数 vty 代表 y 轴坐标，参数 x 代表 x 轴坐标。"
  (with-float-safe 
    (vt-map (lambda (y-coord x-coord) 
              (atan y-coord x-coord)) 
            vty vtx)))

(defun vt-floor (vt &optional (divisor 1))
  "向下取整."
  (with-float-safe 
    (vt-map (lambda (x) 
              (let ((res (nth-value 0 (floor x divisor))))
                (if (floatp x)
		    (float res 1.0d0)
		    res)))
	    vt)))

(defun vt-ceiling (vt &optional (divisor 1))
  "向上取整."
  (with-float-safe 
    (vt-map (lambda (x) 
              (let ((res (nth-value 0 (ceiling x divisor))))
                (if (floatp x)
		    (float res 1.0d0)
		    res)))
	    vt)))

(defun vt-round (vt &optional (divisor 1))
  "四舍五入."
  (with-float-safe 
    (vt-map (lambda (x) 
              (let ((res (nth-value 0 (round x divisor))))
                (if (floatp x)
		    (float res 1.0d0)
		    res)))
	    vt)))

(defun vt-truncate (vt &optional (divisor 1))
  "向0取整"
  (with-float-safe 
    (vt-map (lambda (x) 
              (let ((res (nth-value 0 (truncate x divisor))))
                (if (floatp x)
		    (float res 1.0d0)
		    res)))
	    vt)))

(defun vt-rint (vt)
  "四舍五入到最接近的整数 (浮点数返回值)."
  (with-float-safe 
    (vt-map (lambda (x) 
              (let ((res (nth-value 0 (round x))))
                (if (floatp x)
		    (float res 1.0d0)
		    res))) 
            vt)))

(defun vt-log (vt &optional base)
  "以 base 为底的对数. (非正数返回 nan)"
  (with-float-safe 
    (if base
        (vt-map (lambda (val)
		  (if (<= val 0)
		      +vt-float-nan+
		      (* 1.0d0 (log val base))))
		vt)
        (vt-map (lambda (val)
		  (if (<= val 0)
		      +vt-float-nan+
		      (* 1.0d0 (log val))))
		vt))))


(defun vt-log10 (vt)
  "以 10 为底的对数."
  (with-float-safe
    (vt-log vt 10.0d0)))

(defun vt-log2 (vt)
  "以 2 为底的对数."
  (with-float-safe
    (vt-log vt 2.0d0)))
