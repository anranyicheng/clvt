;;;; 算术运算

(in-package #:clvt)

(defun vt-+ (&rest args)
  "逐元素加法。支持标量、列表、张量混合。
   自动优化：多参数时单次遍历。"
  (apply #'vt-map #'+ args))

(defun vt-* (&rest args)
  "逐元素乘法。支持标量、列表、张量混合。"
  (apply #'vt-map #'* args))

(defun vt-- (vt &rest args)
  "逐元素减法。
   单参数: 取反。
   多参数: a - b - c ... "
  (let ((first-vt (ensure-vt vt)))
    (if (null args)
        (vt-map #'- first-vt)
        ;; 将 first-vt 放在最前面，一次性传入
        (apply #'vt-map #'- first-vt args))))

(defun vt-/ (vt &rest args)
  "逐元素除法。
   单参数: 倒数。
   多参数: a / b / c ... "
  (let ((first-vt (ensure-vt vt)))
    (if (null args)
        (vt-map (lambda (v) (/ 1.0d0 v)) first-vt)
        (apply #'vt-map #'/ first-vt args))))

(defun vt-sin (vt) "计算逐元素正弦值。" (vt-map #'sin vt))
(defun vt-cos (vt) "计算逐元素余弦值。" (vt-map #'cos vt))
(defun vt-tan (vt) "计算逐元素正切值。" (vt-map #'tan vt))

(defun vt-asin (vt) "计算逐元素反正弦值。" (vt-map #'asin vt))
(defun vt-acos (vt) "计算逐元素反余弦值。" (vt-map #'acos vt))
(defun vt-atan (vt) "计算逐元素反正切值。" (vt-map #'atan vt))

(defun vt-sinh (vt) "计算逐元素双曲正弦值。" (vt-map #'sinh vt))
(defun vt-cosh (vt) "计算逐元素双曲余弦值。" (vt-map #'cosh vt))
(defun vt-tanh (vt) "计算逐元素双曲正切值。" (vt-map #'tanh vt))

(defun vt-exp (vt) "计算逐元素 e^vt。" (vt-map #'exp vt))
(defun vt-sqrt (vt) "计算逐元素平方根。" (vt-map #'sqrt vt))

(defun vt-abs (vt) "计算逐元素绝对值。" (vt-map #'abs vt))
(defun vt-signum (vt) "计算逐元素符号。" (vt-map #'signum vt))


(defun vt-positive-p (vt)
  "判断是否大于零。返回 1.0 (True) 或 0.0 (False)。"
  (vt-map (lambda (v) (if (> v 0.0d0) 1.0d0 0.0d0)) vt))

(defun vt-negative-p (vt)
  "判断是否小于零。返回 1.0 或 0.0。"
  (vt-map (lambda (v) (if (< v 0.0d0) 1.0d0 0.0d0)) vt))

(defun vt-zero-p (vt)
  "判断是否为零。返回 1.0 或 0.0。"
  (vt-map (lambda (v) (if (zerop v) 1.0d0 0.0d0)) vt))

(defun vt-nonzero-p (vt)
  "判断是否非零。返回 1.0 或 0.0。"
  (vt-map (lambda (v) (if (zerop v) 0.0d0 1.0d0)) vt))

(defun vt-even-p (vt)
  "判断是否为偶数。返回 1.0 或 0.0。"
  ;; 注意：浮点数判断偶数需先取整
  (vt-map (lambda (v) (if (evenp (floor v)) 1.0d0 0.0d0)) vt))

(defun vt-odd-p (vt)
  "判断是否为奇数。返回 1.0 或 0.0。"
  (vt-map (lambda (v) (if (oddp (floor v)) 1.0d0 0.0d0)) vt))

(defun vt-expt (vt-base power-num)
  "逐元素幂运算"
  (vt-map (lambda (x) (expt x power-num)) vt-base))

(defun vt-mod (vt divisor)
  "逐元素取模"
  (vt-map (lambda (x) (mod x divisor)) vt))

(defun vt-rem (vt divisor)
  "逐元素取余数"
  (vt-map (lambda (x) (rem x divisor)) vt))

(defun vt-atan2 (vt-y x)
  "逐元素计算 atan2(y, x)。支持广播。"
  (vt-map (lambda (y) (atan y x)) vt-y))

(defun vt-floor (vt divisor)
  "向下取整。"
  (vt-map (lambda (x) (floor x divisor)) vt))

(defun vt-ceiling (vt divisor)
  "向上取整。"
  (vt-map (lambda (x) (ceiling x divisor)) vt))

(defun vt-round (vt divisor)
  "四舍五入。"
  (vt-map (lambda (x) (round x divisor)) vt))

(defun vt-rint (vt)
  "四舍五入到最接近的整数 (浮点数返回值)。"
  ;; Common Lisp 没有直接的 rint，使用 round 然后 float 化
  (vt-map (lambda (x) (float (round x) x)) vt))


;;; 带清洗功能的 Log 函数
;; 辅助函数：检查是否存在非正数
(defun vt-any-nonpositive-p (vt)
  "快速检查张量中是否存在小于等于 0 的元素。"
  (let ((data (vt-data vt))
        (offset (vt-offset vt))
        (strides (vt-strides vt))
        (shape (vt-shape vt)))
    (labels
	((recurse (axis current-idx)
           (if (= axis (length shape))
               (let ((val (aref data current-idx)))
                 (<= val 0.0d0)) ; 发现非正数即返回 T
               (let ((dim (nth axis shape))
                     (stride (nth axis strides))
                     (found nil))
                 (loop for i from 0 below dim
                       while (not found) ; 只要找到一个就停止
                       do (setf found (recurse (1+ axis) 
                                               (+ current-idx
						  (* i stride)))))
                 found))))
      (recurse 0 offset))))


(defun vt-log-clean (x &optional base)
  "安全的对数计算：
   1. 检查数据是否全部大于0。
   2. 如果存在非正数，发出三个顺序警告。
   3. 将小于等于0的数替换为0进行计算。"
  (assert (or (null base) (>= base 0)))
  (let ((has-invalid (vt-any-nonpositive-p x)))
    ;; 步骤 2: 发出警告
    (when has-invalid
      (warn "警告 (1/3): 检测到输入数据中包含小于等于 0 的值。")
      (warn "警告 (2/3): 这些非法值将被自动替换为 0 进行计算。")
      (warn "警告 (3/3: 正在执行计算，对应的输出结果将为0。"))
    (if base
        (vt-map (lambda (val)
                  (let ((clean-val (if (> val 0.0d0) val 0.0d0)))
                    (log clean-val base)))
                x)
        (vt-map (lambda (val)
                  (let ((clean-val (if (> val 0.0d0) val 0.0d0)))
		    clean-val))
                x))))


(defun vt-log10 (vt)
  "以 10 为底的对数。"
  (vt-log-clean vt 10.0d0))

(defun vt-log2 (vt)
  "以 2 为底的对数。"
  (vt-log-clean vt 2.0d0))
