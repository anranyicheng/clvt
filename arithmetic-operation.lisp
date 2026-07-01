;;;; 算术运算

(in-package #:clvt)

;; === 1. 基础算术运算 (多参数单趟遍历) ===
(defun vt-+ (&rest args)
  "逐元素加法. 支持标量、列表、张量混合. 多参数时单次遍历."
  (with-float-safe (apply #'vt-map #'+ args)))

(defun vt-add (&rest args) (apply #'vt-+ args))

(defun vt-* (&rest args)
  "逐元素乘法. 支持标量、列表、张量混合. 多参数时单次遍历."
  (with-float-safe (apply #'vt-map #'* args)))

(defun vt-mul (&rest args) (apply #'vt-* args))

(defun vt-- (vt &rest args)
  "逐元素减法. 单参数取反. 多参数: a - b - c ..."
  (with-float-safe 
    (let ((first-vt (ensure-vt vt)))
      (if (null args)
          (vt-map #'- first-vt)
          (apply #'vt-map #'- first-vt args)))))

(defun vt-sub (vt &rest args) (apply #'vt-- vt args))

(defun vt-/ (vt &rest args)
  "逐元素除法. 单参数取倒数. 多参数: a / b / c ... (单趟遍历优化)"
  (with-float-safe 
    (let ((first-vt (ensure-vt vt)))
      (if (null args)
          (vt-map (lambda (v) (/ 1.0d0 v)) first-vt)
          (apply #'vt-map #'/ first-vt args)))))

(defun vt-div (vt &rest args) (apply #'vt-/ vt args))

(defun vt-scale (a b &key out dtype)
  "将张量乘以标量."
  (vt-map #'* a b :out out :dtype dtype))

;; ===== 2. 三角与双曲函数 (一元特化) =====
;; 修复: 整数输入必须强制提升为浮点数类型，防止小数被截断为 0
(defun vt-sin (vt &key out dtype)
  (vt-map #'sin vt :out out 
		   :dtype (or dtype (if (eq (vt-dtype vt) :float32)
					:float32 :float64))))

(defun vt-cos (vt &key out dtype)
  (vt-map #'cos vt :out out 
		   :dtype (or dtype (if (eq (vt-dtype vt) :float32)
					:float32 :float64))))

(defun vt-tan (vt &key out dtype)
  (vt-map #'tan vt :out out 
		   :dtype (or dtype (if (eq (vt-dtype vt) :float32)
					:float32 :float64))))

(defun vt-asin (vt &key out dtype)
  (vt-map #'asin vt :out out 
		    :dtype (or dtype (if (eq (vt-dtype vt) :float32)
					 :float32 :float64))))

(defun vt-acos (vt &key out dtype)
  (vt-map #'acos vt :out out 
		    :dtype (or dtype (if (eq (vt-dtype vt) :float32)
					 :float32 :float64))))

(defun vt-atan (vt &key out dtype)
  (vt-map #'atan vt :out out 
		    :dtype (or dtype (if (eq (vt-dtype vt) :float32)
					 :float32 :float64))))

(defun vt-sinh (vt &key out dtype)
  (vt-map #'sinh vt :out out 
		    :dtype (or dtype (if (eq (vt-dtype vt) :float32)
					 :float32 :float64))))

(defun vt-cosh (vt &key out dtype)
  (vt-map #'cosh vt :out out 
		    :dtype (or dtype (if (eq (vt-dtype vt) :float32)
					 :float32 :float64))))

(defun vt-tanh (vt &key out dtype)
  (vt-map #'tanh vt :out out 
		    :dtype (or dtype (if (eq (vt-dtype vt) :float32)
					 :float32 :float64))))

;; ========== 3. 指数与对数运算 ==========
(defun vt-exp (vt &key out dtype)
  (vt-map #'exp vt :out out 
		   :dtype (or dtype (if (eq (vt-dtype vt) :float32)
					:float32 :float64))))

(defun vt-pow (vt power &key out dtype)
  "计算逐元素幂次方"
  (with-float-safe 
    (vt-map (lambda (x) (expt x power)) vt :out out :dtype dtype)))

(defun vt-expt (vt power-num &key out dtype)
  "vt-pow 的别名. 逐元素幂运算."
  (vt-pow vt power-num :out out :dtype dtype))

(defun vt-square (vt &key out dtype)
  "计算逐元素平方"
  (vt-pow vt 2 :out out :dtype dtype))


(defun vt-sqrt (vt &key out dtype)
  "计算逐元素平方根. 负数返回对应类型的 NaN."
  (let* ((infer-dtype (or dtype (if (eq (vt-dtype vt) :float32)
				    :float32 :float64)))
         ;; 预先绑定正确类型的 NaN，消灭循环内的 if 判断
         (nan-val (vt-get-nan dtype)))
    (vt-map (lambda (x) (if (minusp x) nan-val (sqrt x))) 
            vt :out out :dtype infer-dtype)))

(defun vt-log (vt &key base out dtype)
  "对数运算. 非正数返回对应类型的 nan."
  (with-float-safe 
    (let* ((infer-dtype (or dtype (if (eq (vt-dtype vt) :float32)
				      :float32 :float64)))
           (nan-val (vt-get-nan dtype)))
      (if base
          (vt-map (lambda (val) (if (<= val 0) nan-val (log val base))) 
                  vt :out out :dtype infer-dtype)
          (vt-map (lambda (val) (if (<= val 0) nan-val (log val))) 
                  vt :out out :dtype infer-dtype)))))

(defun vt-log10 (vt &key out dtype)
  "以 10 为底的对数."
  (vt-log vt :base 10.0d0 :out out :dtype dtype))

(defun vt-log2 (vt &key out dtype)
  "以 2 为底的对数."
  (vt-log vt :base 2.0d0 :out out :dtype dtype))

;; ========== 4. 符号与绝对值运算 ==========
(defun vt-abs (vt &key out dtype)
  (vt-map #'abs vt :out out :dtype dtype))

(defun vt-signum (vt &key out dtype)
  (vt-map #'signum vt :out out :dtype dtype))

;; ============ 5. 布尔判断运算 ============
(defun vt-positive-p (vt &key out (dtype :float64))
  "判断是否大于零. 返回 1.0 (true) 或 0.0 (false)."
  (with-float-safe 
    (vt-map (lambda (v) (if (> v 0.0d0) 1.0d0 0.0d0))
	    vt :out out :dtype dtype)))

(defun vt-negative-p (vt &key out (dtype :float64))
  "判断是否小于零. 返回 1.0 或 0.0."
  (with-float-safe 
    (vt-map (lambda (v) (if (< v 0.0d0) 1.0d0 0.0d0))
	    vt :out out :dtype dtype)))

(defun vt-zero-p (vt &key out (dtype :float64))
  "判断是否为零. 返回 1.0 或 0.0."
  (with-float-safe 
    (vt-map (lambda (v) (if (zerop v) 1.0d0 0.0d0))
	    vt :out out :dtype dtype)))

(defun vt-nonzero-p (vt &key out (dtype :float64))
  "判断是否非零. 返回 1.0 或 0.0."
  (with-float-safe 
    (vt-map (lambda (v) (if (zerop v) 0.0d0 1.0d0))
	    vt :out out :dtype dtype)))

(defun vt-even-p (vt &key out (dtype :float64))
  "判断是否为偶数. 返回 1.0 或 0.0."
  (with-float-safe 
    (vt-map (lambda (v) (if (evenp (floor v)) 1.0d0 0.0d0))
	    vt :out out :dtype dtype)))

(defun vt-odd-p (vt &key out (dtype :float64))
  "判断是否为奇数. 返回 1.0 或 0.0."
  (with-float-safe 
    (vt-map (lambda (v) (if (oddp (floor v)) 1.0d0 0.0d0))
	    vt :out out :dtype dtype)))

;; ============ 6. 取整与取余运算 ============
(defun vt-mod (vt divisor &key out dtype)
  "逐元素取模."
  (with-float-safe 
    (vt-map (lambda (x) (mod x divisor))
	    vt :out out :dtype dtype)))

(defun vt-rem (vt divisor &key out dtype)
  "逐元素取余数."
  (with-float-safe 
    (vt-map (lambda (x) (rem x divisor))
	    vt :out out :dtype dtype)))

(defun vt-atan2 (vty vtx &key out dtype)
  "逐元素计算 atan2(y, x). 支持广播."
  (with-float-safe 
    (vt-map #'atan vty vtx :out out :dtype dtype)))

;; 修复: 使用输入值 x 作为浮点基准，防止将 float32 强制提升为 float64
(defun vt-floor (vt &key (divisor 1) out dtype)
  "向下取整."
  (with-float-safe 
    (vt-map (lambda (x) 
              (let ((res (nth-value 0 (floor x divisor))))
                (if (floatp x) (float res x) res))) 
            vt :out out :dtype dtype)))

(defun vt-ceiling (vt &key (divisor 1) out dtype)
  "向上取整."
  (with-float-safe 
    (vt-map (lambda (x) 
              (let ((res (nth-value 0 (ceiling x divisor))))
                (if (floatp x) (float res x) res))) 
            vt :out out :dtype dtype)))

(defun vt-round (vt &key (divisor 1) out dtype)
  "四舍五入."
  (with-float-safe 
    (vt-map (lambda (x) 
              (let ((res (nth-value 0 (round x divisor))))
                (if (floatp x) (float res x) res))) 
            vt :out out :dtype dtype)))

(defun vt-truncate (vt &key (divisor 1) out dtype)
  "向0取整"
  (with-float-safe 
    (vt-map (lambda (x) 
              (let ((res (nth-value 0 (truncate x divisor))))
                (if (floatp x) (float res x) res))) 
            vt :out out :dtype dtype)))

(defun vt-rint (vt &key out dtype)
  "四舍五入到最接近的整数 (浮点数返回值)."
  (with-float-safe 
    (vt-map (lambda (x) 
              (let ((res (nth-value 0 (round x))))
                (if (floatp x) (float res x) res))) 
            vt :out out :dtype dtype)))

;; ============ 7. 比较运算 ============
(defun vt-= (t1 t2 &key (dtype :float64) out)
  "逐元素相等比较. 支持广播.返回布尔张量(1.0或0.0)."
  (with-float-safe 
    (vt-map (lambda (a b) (if (= a b) 1.0d0 0.0d0)) 
            (ensure-vt t1) (ensure-vt t2) :dtype dtype :out out)))

(defun vt-/= (t1 t2 &key (dtype :float64) out)
  "逐元素不相等比较. 支持广播.返回布尔张量(1.0或0.0)."
  (with-float-safe 
    (vt-map (lambda (a b) (if (/= a b) 1.0d0 0.0d0)) 
            (ensure-vt t1) (ensure-vt t2) :dtype dtype :out out)))

(defun vt-< (t1 t2 &key (dtype :float64) out)
  "逐元素小于比较. 支持广播.返回布尔张量(1.0或0.0)."
  (with-float-safe 
    (vt-map (lambda (a b) (if (< a b) 1.0d0 0.0d0)) 
            (ensure-vt t1) (ensure-vt t2) :dtype dtype :out out)))

(defun vt-<= (t1 t2 &key (dtype :float64) out)
  "逐元素小于等于比较. 支持广播.返回布尔张量(1.0或0.0)."
  (with-float-safe 
    (vt-map (lambda (a b) (if (<= a b) 1.0d0 0.0d0)) 
            (ensure-vt t1) (ensure-vt t2) :dtype dtype :out out)))

(defun vt-> (t1 t2 &key (dtype :float64) out)
  "逐元素大于比较. 支持广播.返回布尔张量(1.0或0.0)."
  (with-float-safe 
    (vt-map (lambda (a b) (if (> a b) 1.0d0 0.0d0)) 
            (ensure-vt t1) (ensure-vt t2) :dtype dtype :out out)))

(defun vt->= (t1 t2 &key (dtype :float64) out)
  "逐元素大于等于比较. 支持广播.返回布尔张量(1.0或0.0)."
  (with-float-safe 
    (vt-map (lambda (a b) (if (>= a b) 1.0d0 0.0d0)) 
            (ensure-vt t1) (ensure-vt t2) :dtype dtype :out out)))


;; ========== 8. 补充: 双曲反函数 ==========
;; 修复逻辑: acosh 输入 < 1 会产生复数，需安全返回 NaN
(defun vt-asinh (vt &key out dtype)
  "反双曲正弦."
  (vt-map #'asinh vt :out out 
		     :dtype (or dtype (if (eq (vt-dtype vt) :float32)
					  :float32 :float64))))

(defun vt-acosh (vt &key out dtype)
  "反双曲余弦. 输入 < 1 返回 NaN."
  (let ((infer-dtype (or dtype (if (eq (vt-dtype vt) :float32)
				   :float32 :float64))))
    (vt-map (lambda (x) 
              (if (< x 1.0d0) 
                  (if (eq infer-dtype :float32)
		      +vt-sfloat-nan+
		      +vt-dfloat-nan+)
                  (acosh x))) 
            vt :out out :dtype infer-dtype)))

(defun vt-atanh (vt &key out dtype)
  "反双曲正切. 输入绝对值 >= 1 返回 NaN."
  (let ((infer-dtype (or dtype (if (eq (vt-dtype vt) :float32)
				   :float32 :float64))))
    (vt-map (lambda (x) 
              (if (>= (abs x) 1.0d0) 
                  (if (eq infer-dtype :float32)
		      +vt-sfloat-nan+
		      +vt-dfloat-nan+)
                  (atanh x))) 
            vt :out out :dtype infer-dtype)))

;; ========== 9. 补充: 角度与弧度转换 ==========
(defun vt-degrees (vt &key out dtype)
  "弧度转角度."
  (let ((factor (float (/ 180.0 pi) 1.0d0)))
    (vt-map (lambda (x) (* x factor))
	    vt
	    :out out 
	    :dtype (or dtype (if (eq (vt-dtype vt) :float32)
				 :float32 :float64)))))

(defun vt-radians (vt &key out dtype)
  "角度转弧度."
  (let ((factor (float (/ pi 180.0) 1.0d0)))
    (vt-map (lambda (x) (* x factor))
	    vt
	    :out out 
	    :dtype (or dtype (if (eq (vt-dtype vt) :float32)
				 :float32 :float64)))))

;; ========== 10. 补充: 逐元素极值运算 ==========
(defun vt-maximum (t1 t2 &key out dtype)
  "逐元素取两数组中较大者。NaN 会传播。"
  (with-float-safe
    (vt-map (lambda (a b)
              (cond ((vt-float-nan-p a) a)
                    ((vt-float-nan-p b) b)
                    (t (max a b))))
            t1 t2 :out out :dtype dtype)))

(defun vt-minimum (t1 t2 &key out dtype)
  "逐元素取两数组中较小者。NaN 会传播。"
  (with-float-safe
    (vt-map (lambda (a b)
              (cond ((vt-float-nan-p a) a)
                    ((vt-float-nan-p b) b)
                    (t (min a b))))
            t1 t2 :out out :dtype dtype)))

(defun vt-fmax (t1 t2 &key out dtype)
  "逐元素取最大值，忽略 NaN (NaN 仅在两个输入均为 NaN 时返回)."
  (with-float-safe 
    (vt-map (lambda (a b) 
              (cond ((and (vt-float-nan-p a) (vt-float-nan-p b)) a)
                    ((vt-float-nan-p a) b)
                    ((vt-float-nan-p b) a)
                    (t (max a b))))
            (ensure-vt t1) (ensure-vt t2) :dtype dtype :out out)))

(defun vt-fmin (t1 t2 &key out dtype)
  "逐元素取最小值，忽略 NaN."
  (with-float-safe 
    (vt-map (lambda (a b) 
              (cond ((and (vt-float-nan-p a) (vt-float-nan-p b)) a)
                    ((vt-float-nan-p a) b)
                    ((vt-float-nan-p b) a)
                    (t (min a b))))
            (ensure-vt t1) (ensure-vt t2) :dtype dtype :out out)))

;; ========== 11. 补充: 逐元素逻辑运算 (返回 1.0 或 0.0) ==========
(defun vt-logical-and (t1 t2 &key out (dtype :float64))
  "逻辑与. 非零为真."
  (with-float-safe 
    (vt-map (lambda (a b)
	      (if (and (not (zerop a)) (not (zerop b))) 1.0d0 0.0d0))
            (ensure-vt t1) (ensure-vt t2) :dtype dtype :out out)))

(defun vt-logical-or (t1 t2 &key out (dtype :float64))
  "逻辑或."
  (with-float-safe 
    (vt-map (lambda (a b)
	      (if (or (not (zerop a)) (not (zerop b))) 1.0d0 0.0d0))
            (ensure-vt t1) (ensure-vt t2) :dtype dtype :out out)))

(defun vt-logical-not (vt &key out (dtype :float64))
  "逻辑非."
  (with-float-safe 
    (vt-map (lambda (v)
	      (if (zerop v) 1.0d0 0.0d0))
	    vt :dtype dtype :out out)))

(defun vt-logical-xor (t1 t2 &key out (dtype :float64))
  "逻辑异或."
  (with-float-safe 
    (vt-map (lambda (a b)
	      (if (not (eq (not (zerop a)) (not (zerop b)))) 1.0d0 0.0d0))
            (ensure-vt t1) (ensure-vt t2) :dtype dtype :out out)))

;; ========== 12. 补充: 逐元素位运算 (仅适用于整数张量) ==========
;; 注意: 如果对浮点张量使用，底层 Lisp 会抛出类型错误，符合"不给上层擦屁股"原则
(defun vt-bit-and (t1 t2 &key out dtype)
  "逐元素按位与."
  (vt-map #'logand (ensure-vt t1) (ensure-vt t2) :dtype dtype :out out))

(defun vt-bit-ior (t1 t2 &key out dtype)
  "逐元素按位或."
  (vt-map #'logior (ensure-vt t1) (ensure-vt t2) :dtype dtype :out out))

(defun vt-bit-xor (t1 t2 &key out dtype)
  "逐元素按位异或."
  (vt-map #'logxor (ensure-vt t1) (ensure-vt t2) :dtype dtype :out out))

(defun vt-bit-not (vt &key out dtype)
  "逐元素按位取反."
  (vt-map #'lognot vt :dtype dtype :out out))

(defun vt-left-shift (vt shift &key out dtype)
  "逐元素左移."
  (vt-map (lambda (x) (ash x shift)) vt :dtype dtype :out out))

(defun vt-right-shift (vt shift &key out dtype)
  "逐元素右移 (算术右移)."
  (vt-map (lambda (x) (ash x (- shift))) vt :dtype dtype :out out))

;; ========== 13. 补充: 数值限制与插值 ==========
(defun vt-clip (vt min-val max-val &key out dtype)
  "将张量元素限制在 [min-val, max-val] 范围内."
  (with-float-safe 
    (vt-map (lambda (x) (max min-val (min max-val x))) 
            vt :dtype dtype :out out)))

(defun vt-lerp (start end weight &key out dtype)
  "线性插值: start + (end - start) * weight. 支持广播."
  (with-float-safe 
    (vt-map (lambda (s e w) (+ s (* (- e s) w)))
            (ensure-vt start) (ensure-vt end) (ensure-vt weight) 
            :dtype dtype :out out)))

;; ========== 14. 补充: 特殊数学函数 ==========
(defun vt-cbrt (vt &key out dtype)
  "计算逐元素立方根."
  ;; 修复: Lisp 的 不支持负数，需特殊处理
  (vt-map (lambda (x) 
            (if (minusp x) 
                (- (expt (- x) (/ 3.0d0))) 
                (expt x (/ 3.0d0)))) 
          vt
	  :out out 
          :dtype (or dtype (if (eq (vt-dtype vt) :float32)
			       :float32 :float64))))

(defun vt-hypot (t1 t2 &key out dtype)
  "计算直角三角形斜边 sqrt(a^2 + b^2). 防溢出实现. 支持广播."
  (with-float-safe 
    (vt-map (lambda (a b) 
              (let ((abs-a (abs a)) (abs-b (abs b)))
                (cond ((zerop abs-a) abs-b)
                      ((zerop abs-b) abs-a)
                      (t (* (max abs-a abs-b) 
                            (sqrt (+ 1.0d0 (expt (/ (min abs-a abs-b)
						    (max abs-a abs-b)) 2))))))))
            (ensure-vt t1) (ensure-vt t2) :out out 
            :dtype (or dtype (if (or (eq (vt-dtype t1) :float32) 
                                     (eq (vt-dtype t2) :float32)) 
                                 :float32 :float64)))))

(defun vt-reciprocal (vt &key out dtype)
  "逐元素取倒数 1/x. (相当于单参数的 vt-/)"
  (vt-map (lambda (v) (/ 1.0d0 v))
	  vt :out out 
	  :dtype (or dtype (if (eq (vt-dtype vt) :float32)
			       :float32 :float64))))

(defun vt-negative (vt &key out dtype)
  "显式逐元素取反. (相当于单参数的 vt--)"
  (vt-map #'- vt :dtype dtype :out out))
