;;;; elementwise.lisp — 逐元素算术 / 数学 / 比较 / 逻辑 / 位运算

(in-package :clvt)

(defun vt-+ (&rest args)
  (with-float-safe
    (multiple-value-bind (tensors dtype out) (parse-vt-op-args args)
      (case (length tensors)
        (1 (vt-fast-map #'+ (first tensors)
                            :dtype dtype :out out))
        (2 (vt-fast-map #'+ (first tensors)
                            (second tensors)
                            :dtype dtype :out out))
        (3 (vt-fast-map #'+ (first tensors)
                            (second tensors)
                            (third tensors)
                            :dtype dtype :out out))
        (t (apply #'vt-map #'+ args))))))

(defun vt-* (&rest args)
  (with-float-safe
    (multiple-value-bind (tensors dtype out) (parse-vt-op-args args)
      (case (length tensors)
        (1 (vt-fast-map #'* (first tensors)
                            :dtype dtype :out out))
        (2 (vt-fast-map #'* (first tensors)
                            (second tensors)
                            :dtype dtype :out out))
        (3 (vt-fast-map #'* (first tensors)
                            (second tensors)
                            (third tensors)
                            :dtype dtype :out out))
        (t (apply #'vt-map #'* args))))))

(defun vt-- (vt &rest args)
  "一元：-vt；二元及以上：vt - arg1 - arg2 - ...
   VT 是第一个张量，其余从 ARGS 解析。"
  (with-float-safe
    (multiple-value-bind (tensors dtype out) (parse-vt-op-args args)
      (let ((all (cons (ensure-vt vt) tensors)))
        (case (length all)
          (1 (vt-fast-map #'- (first all)
                              :dtype dtype :out out))
          (2 (vt-fast-map #'- (first all) (second all)
                              :dtype dtype :out out))
          (3 (vt-fast-map #'- (first all) (second all) (third all)
                              :dtype dtype :out out))
          (t (apply #'vt-map #'- all)))))))

(defun vt-/ (vt &rest args)
  "一元：1 / vt（倒数）；二元及以上：vt / arg1 / arg2 / ...
   VT 是第一个张量，其余从 ARGS 解析。
   整型输入默认提升到 float64，与 numpy 对齐。"
  (with-float-safe
    (multiple-value-bind (tensors dtype out) (parse-vt-op-args args)
      (let* ((all (cons (ensure-vt vt) tensors))
             (effective-dtype
               (cond (dtype dtype)
                     (out   (vt-dtype out))
                     ((every (lambda (x) (vt-int-dtype-p (vt-dtype x))) all)
                      :float64)
                     (t nil))))
        (case (length all)
          (1 (vt-fast-map #'/ (make-vt nil 1 :dtype (vt-dtype (first all)))
                              (first all)
                              :dtype effective-dtype :out out))
          (2 (vt-fast-map #'/ (first all) (second all)
                              :dtype effective-dtype :out out))
          (3 (vt-fast-map #'/ (first all) (second all) (third all)
                              :dtype effective-dtype :out out))
          (t (apply #'vt-map #'/ all)))))))

(defun vt-add (a b &key dtype out)
  (vt-fast-map #'+ a b :dtype dtype :out out))

(defun vt-sub (a b &key dtype out)
  (vt-fast-map #'- a b :dtype dtype :out out))

(defun vt-mul (a b &key dtype out)
  (vt-fast-map #'* a b :dtype dtype :out out))

(defun vt-div (a b &key dtype out)
  (vt-fast-map #'/ a b :dtype dtype :out out))

(defun vt-scale (a b &key out dtype)
  (vt-fast-map #'* a b :out out :dtype dtype))

(defun %infer-float-dtype (vt dtype)
  (or dtype (if (eq (vt-dtype vt) :float32) :float32 :float64)))

(defun vt-sin (vt &key out dtype)
  (vt-fast-map #'sin vt :out out :dtype (%infer-float-dtype vt dtype)))

(defun vt-cos (vt &key out dtype)
  (vt-fast-map #'cos vt :out out :dtype (%infer-float-dtype vt dtype)))

(defun vt-tan (vt &key out dtype)
  (vt-fast-map #'tan vt :out out :dtype (%infer-float-dtype vt dtype)))

(defun vt-atan (vt &key out dtype)
  (vt-fast-map #'atan vt :out out :dtype (%infer-float-dtype vt dtype)))

(defun vt-sinh (vt &key out dtype)
  (vt-fast-map #'sinh vt :out out :dtype (%infer-float-dtype vt dtype)))

(defun vt-cosh (vt &key out dtype)
  (vt-fast-map #'cosh vt :out out :dtype (%infer-float-dtype vt dtype)))

(defun vt-tanh (vt &key out dtype)
  (vt-fast-map #'tanh vt :out out :dtype (%infer-float-dtype vt dtype)))

(defun vt-asin (vt &key out dtype)
  (let* ((dt (%infer-float-dtype vt dtype))
	 (nan (vt-get-nan dt)))
    (vt-map (lambda (x)
	      (if (> (abs x) 1.0d0) nan (asin x)))
	    vt :out out :dtype dt)))

(defun vt-acos (vt &key out dtype)
  (let* ((dt (%infer-float-dtype vt dtype))
	 (nan (vt-get-nan dt)))
    (vt-map (lambda (x)
	      (if (> (abs x) 1.0d0) nan (acos x)))
	    vt :out out :dtype dt)))

(defun vt-asinh (vt &key out dtype)
  (vt-fast-map #'asinh vt :out out :dtype (%infer-float-dtype vt dtype)))

(defun vt-acosh (vt &key out dtype)
  (let* ((dt (%infer-float-dtype vt dtype))
	 (nan (vt-get-nan dt)))
    (vt-map (lambda (x)
	      (if (< x 1.0d0) nan (acosh x)))
	    vt :out out :dtype dt)))

(defun vt-atanh (vt &key out dtype)
  (let* ((dt (%infer-float-dtype vt dtype))
	 (nan (vt-get-nan dt)))
    (vt-map (lambda (x)
	      (if (>= (abs x) 1.0d0) nan (atanh x)))
	    vt :out out :dtype dt)))

(defun vt-exp (vt &key out dtype)
  (vt-fast-map #'exp vt :out out :dtype (%infer-float-dtype vt dtype)))

(defun vt-pow (vt power &key out dtype)
  (let* ((dt (%infer-float-dtype vt dtype))
         (nan (vt-get-nan dt)))
    (cond
      ((and (integerp power) (plusp power))
       (vt-map (lambda (x) (expt x power)) vt :out out :dtype dt))
      (t
       (vt-map (lambda (x)
                 (let ((result (handler-case (expt x power)
                                 (error () nan))))
                   (if (realp result) result nan)))
               vt :out out :dtype dt)))))

(defun vt-expt (vt power &key out dtype)
  (vt-pow vt power :out out :dtype dtype))

(defun vt-square (vt &key out dtype)
  "逐元素平方。特化为 vt-fast-map #'* 以避免 vt-pow 的 handler-case/expt/realp 开销。"
  (vt-fast-map #'* vt vt :out out :dtype dtype))

(defun vt-sqrt (vt &key out dtype)
  (let* ((dt (%infer-float-dtype vt dtype))
	 (nan (vt-get-nan dt)))
    (vt-map (lambda (x) (if (minusp x) nan (sqrt x)))
	    vt :out out :dtype dt)))

(defun vt-log (vt &key base out dtype)
  (let* ((dt (%infer-float-dtype vt dtype))
         (nan (vt-get-nan dt))
	 (neginf (vt-get-neg-inf dt))
	 (posinf (vt-get-pos-inf dt)))
    (cond
      ((and base (or (<= base 0) (= base 1)))
       (vt-map (lambda (x)
		 (declare (ignore x)) nan)
	       vt :out out :dtype dt))
      ((null base)
       (vt-map (lambda (x)
		 (if (> x 0)
		     (log x)
		     (if (zerop x)
			 neginf nan)))
	       vt :out out :dtype dt))
      (t
       (let ((zero-result (if (plusp (log base)) neginf posinf)))
         (vt-map (lambda (x)
		   (if (> x 0)
		       (log x base)
		       (if (zerop x)
			   zero-result nan)))
                 vt :out out :dtype dt))))))

(defun vt-log10 (vt &key out dtype)
  (vt-log vt :base 10.0d0 :out out :dtype dtype))

(defun vt-log2 (vt &key out dtype)
  (vt-log vt :base 2.0d0 :out out :dtype dtype))

(defun vt-abs (vt &key out dtype)
  (vt-fast-map #'abs vt :out out :dtype dtype))

(defun vt-signum (vt &key out dtype)
  (vt-map (lambda (x) (if (%nan-p x) x (signum x)))
          vt :out out :dtype dtype))

(defun vt-positive-p (vt &key out (dtype :float64))
  (vt-map (lambda (v)
	    (if (> v 0.0d0) 1.0d0 0.0d0))
	  vt :out out :dtype dtype))

(defun vt-negative-p (vt &key out (dtype :float64))
  (vt-map (lambda (v)
	    (if (< v 0.0d0) 1.0d0 0.0d0))
	  vt :out out :dtype dtype))

(defun vt-zero-p (vt &key out (dtype :float64))
  (vt-map (lambda (v)
	    (if (zerop v) 1.0d0 0.0d0))
	  vt :out out :dtype dtype))

(defun vt-nonzero-p (vt &key out (dtype :float64))
  (vt-map (lambda (v)
	    (if (zerop v) 0.0d0 1.0d0))
	  vt :out out :dtype dtype))

(defun vt-even-p (vt &key out (dtype :float64))
  (vt-map (lambda (v)
	    (if (or (%nan-p v) (%inf-p v)) 0.0d0 (if (evenp (floor v)) 1.0d0 0.0d0)))
	  vt :out out :dtype dtype))

(defun vt-odd-p (vt &key out (dtype :float64))
  (vt-map (lambda (v)
	    (if (or (%nan-p v) (%inf-p v)) 0.0d0 (if (oddp (floor v)) 1.0d0 0.0d0)))
	  vt :out out :dtype dtype))

(defun %mod-nan-or-inf-p (x)
  "取模语境下的非有限值判定（需已屏蔽浮点陷阱）。"
  (and (floatp x) (%nan-or-inf-p x)))

(declaim (inline %mod-float-nan))
(defun %mod-float-nan (x)
  "取模遇 NaN 时的返回值。写入按输出 dtype 转换，恒返 double NaN 即可。"
  (declare (ignore x))
  +vt-dfloat-nan+)

(defun vt-mod (vt divisor &key out dtype)
  "逐元素取模。DIVISOR 可为标量或张量（自动广播）。
   语义对标 numpy.remainder：结果与除数同号；
   除数为 0 时返回 0（库约定，对标 np.mod 的整型零除）；
   NaN 任一侧出现 → NaN；被除数为 ±Inf → NaN；除数为 ±Inf → 被除数本身。"
  (if (numberp divisor)
  (vt-map (lambda (x)
                (cond ((%mod-nan-or-inf-p x) (%mod-float-nan x))
                      ((%mod-nan-or-inf-p divisor) (%mod-float-nan x))
                      ((and (numberp divisor) (zerop divisor)) 0)
                      ((and (floatp divisor) (%inf-p divisor)) x)
                      (t (mod x divisor))))
              vt :out out :dtype dtype)
      (vt-map (lambda (x y)
                (cond ((%mod-nan-or-inf-p x) (%mod-float-nan x))
                      ((%mod-nan-or-inf-p y) (%mod-float-nan x))
                      ((zerop y) 0)
                      ((and (floatp y) (%inf-p y)) x)
                      (t (mod x y))))
              vt divisor :out out :dtype dtype)))

(defun vt-rem (vt divisor &key out dtype)
  "逐元素余数。DIVISOR 可为标量或张量（自动广播）。
   CL:rem 为截断除法余数（与被除数同号），对标 numpy.fmod
   （docstring 曾误标 numpy.remainder，现已更正）。
   除数为 0 时返回 0（库约定）；NaN 任一侧 → NaN；
   被除数为 ±Inf → NaN；除数为 ±Inf → 被除数本身。"
  (if (numberp divisor)
  (vt-map (lambda (x)
                (cond ((%mod-nan-or-inf-p x) (%mod-float-nan x))
                      ((%mod-nan-or-inf-p divisor) (%mod-float-nan x))
                      ((zerop divisor) 0)
                      ((and (floatp divisor) (%inf-p divisor)) x)
                      (t (rem x divisor))))
              vt :out out :dtype dtype)
      (vt-map (lambda (x y)
                (cond ((%mod-nan-or-inf-p x) (%mod-float-nan x))
                      ((%mod-nan-or-inf-p y) (%mod-float-nan x))
                      ((zerop y) 0)
                      ((and (floatp y) (%inf-p y)) x)
                      (t (rem x y))))
              vt divisor :out out :dtype dtype)))

(defun vt-atan2 (vty vtx &key out dtype)
  (vt-fast-map #'atan vty vtx :out out :dtype dtype))

;; floor 族对 NaN/±Inf 统一传播本身（对标 numpy：floor/ceil/trunc/round(nan)=nan，
;; floor(±inf)=±inf）。SBCL 对非有限值直接调用 floor/round 会 signal
;; FLOATING-POINT-INVALID-OPERATION（trap 屏蔽也无法避免），必须显式拦截。
(defmacro %floor-family-body (x op)
  `(if (and (floatp ,x) (%nan-or-inf-p ,x))
       ,x
       (let ((res (nth-value 0 (,op ,x divisor))))
         (if (floatp ,x) (float res ,x) res))))

(defun vt-floor (vt &key (divisor 1) out dtype)
  (vt-map (lambda (x) (%floor-family-body x floor))
	  vt :out out :dtype dtype))

(defun vt-ceiling (vt &key (divisor 1) out dtype)
  (vt-map (lambda (x) (%floor-family-body x ceiling))
	  vt :out out :dtype dtype))

(defun vt-round (vt &key (divisor 1) out dtype)
  (vt-map (lambda (x) (%floor-family-body x round))
	  vt :out out :dtype dtype))

(defun vt-truncate (vt &key (divisor 1) out dtype)
  (vt-map (lambda (x) (%floor-family-body x truncate))
	  vt :out out :dtype dtype))

(defun vt-rint (vt &key out dtype)
  (vt-map (lambda (x)
            (if (and (floatp x) (%nan-or-inf-p x))
                x
	    (let ((res (nth-value 0 (round x))))
                  (if (floatp x) (float res x) res))))
	  vt :out out :dtype dtype))

(declaim (inline %op-eq %op-ne %op-lt %op-le %op-gt %op-ge))
(defun %op-eq (a b) (if (=  a b) 1.0d0 0.0d0))

(defun %op-ne (a b) (if (/= a b) 1.0d0 0.0d0))

(defun %op-lt (a b) (if (<  a b) 1.0d0 0.0d0))

(defun %op-le (a b) (if (<= a b) 1.0d0 0.0d0))

(defun %op-gt (a b) (if (>  a b) 1.0d0 0.0d0))

(defun %op-ge (a b) (if (>= a b) 1.0d0 0.0d0))

(defun vt-= (t1 t2 &key (dtype :float64) out)
  (vt-fast-map #'%op-eq (ensure-vt t1) (ensure-vt t2)
               :dtype dtype :out out))

(defun vt-/= (t1 t2 &key (dtype :float64) out)
  (vt-fast-map #'%op-ne (ensure-vt t1) (ensure-vt t2)
               :dtype dtype :out out))

(defun vt-< (t1 t2 &key (dtype :float64) out)
  (vt-fast-map #'%op-lt (ensure-vt t1) (ensure-vt t2)
               :dtype dtype :out out))

(defun vt-<= (t1 t2 &key (dtype :float64) out)
  (vt-fast-map #'%op-le (ensure-vt t1) (ensure-vt t2)
               :dtype dtype :out out))

(defun vt-> (t1 t2 &key (dtype :float64) out)
  (vt-fast-map #'%op-gt (ensure-vt t1) (ensure-vt t2)
               :dtype dtype :out out))

(defun vt->= (t1 t2 &key (dtype :float64) out)
  (vt-fast-map #'%op-ge (ensure-vt t1) (ensure-vt t2)
               :dtype dtype :out out))

(defun vt-rad2deg (vt &key out dtype)
  (let* ((dt (%infer-float-dtype vt dtype))
         (factor (if (eq dt :float32) (/ 180.0s0 (coerce pi 'single-float))
                     (/ 180.0d0 pi))))
    (vt-map (lambda (x) (* x factor)) vt :out out :dtype dt)))

(defun vt-deg2rad (vt &key out dtype)
  (let* ((dt (%infer-float-dtype vt dtype))
         (factor (if (eq dt :float32) (/ (coerce pi 'single-float) 180.0s0)
                     (/ pi 180.0d0))))
    (vt-map (lambda (x) (* x factor)) vt :out out :dtype dt)))

(defun vt-maximum (t1 t2 &key out dtype)
  (vt-map (lambda (a b)
	    (cond ((%nan-p a) a)
		  ((%nan-p b) b)
		  (t (max a b))))	  
          t1 t2 :out out :dtype dtype))
(defun vt-minimum (t1 t2 &key out dtype)
  (vt-map (lambda (a b)
	    (cond ((%nan-p a) a)
		  ((%nan-p b) b)
		  (t (min a b))))
          t1 t2 :out out :dtype dtype))

(defun vt-fmax (t1 t2 &key out dtype)
  (vt-map (lambda (a b)
	    (cond ((and (%nan-p a) (%nan-p b)) a)
                  ((%nan-p a) b) ((%nan-p b) a)
		  (t (max a b))))
          (ensure-vt t1) (ensure-vt t2) :dtype dtype :out out))

(defun vt-fmin (t1 t2 &key out dtype)
  (vt-map (lambda (a b)
	    (cond ((and (%nan-p a) (%nan-p b)) a)
                  ((%nan-p a) b) ((%nan-p b) a)
		  (t (min a b))))
          (ensure-vt t1) (ensure-vt t2) :dtype dtype :out out))

(defun vt-logical-and (t1 t2 &key out (dtype :float64))
  (vt-map (lambda (a b)
	    (if (and (not (zerop a)) (not (zerop b)))
		1.0d0 0.0d0))
          (ensure-vt t1) (ensure-vt t2) :dtype dtype :out out))

(defun vt-logical-or (t1 t2 &key out (dtype :float64))
  (vt-map (lambda (a b)
	    (if (or (not (zerop a)) (not (zerop b)))
		1.0d0 0.0d0))
          (ensure-vt t1) (ensure-vt t2) :dtype dtype :out out))

(defun vt-logical-not (vt &key out (dtype :float64))
  (vt-map (lambda (v)
	    (if (zerop v) 1.0d0 0.0d0))
	  vt :dtype dtype :out out))

(defun vt-logical-xor (t1 t2 &key out (dtype :float64))
  (vt-map (lambda (a b)
	    (if (not (eq (not (zerop a)) (not (zerop b))))
		1.0d0 0.0d0))
          (ensure-vt t1) (ensure-vt t2) :dtype dtype :out out))

(defun vt-bit-and (t1 t2 &key out dtype)
  (vt-map #'logand (ensure-vt t1) (ensure-vt t2) :dtype dtype :out out))

(defun vt-bit-ior (t1 t2 &key out dtype)
  (vt-map #'logior (ensure-vt t1) (ensure-vt t2) :dtype dtype :out out))

(defun vt-bit-xor (t1 t2 &key out dtype)
  (vt-map #'logxor (ensure-vt t1) (ensure-vt t2) :dtype dtype :out out))

(defun vt-bit-not (vt &key out dtype)
  (vt-map #'lognot vt :dtype dtype :out out))

(defun vt-left-shift (vt shift &key out dtype)
  (vt-map (lambda (x) (ash x shift))
	  vt :dtype dtype :out out))

(defun vt-right-shift (vt shift &key out dtype)
  (vt-map (lambda (x) (ash x (- shift)))
	  vt :dtype dtype :out out))

(declaim (inline %op-clip))
(defun %op-clip (x minv maxv)
  "三元 clip：x 先和 maxv 比较，再和 minv 比较。
   与 (min max-val (max min-val x)) 语义一致。"
  (let ((x (if (> x maxv) maxv x)))
    (if (< x minv) minv x)))

(defun vt-clip (vt min-val max-val &key out dtype)
  "逐元素裁剪。MIN-VAL / MAX-VAL 是标量（不是张量）。
   语义：clip(x, min, max) = min(max, max(min, x))"
  (vt-fast-map #'%op-clip vt min-val max-val :dtype dtype :out out))

(defun vt-lerp (start end weight &key out dtype)
  (vt-map (lambda (s e w)
	    (+ s (* (- e s) w)))
	  (ensure-vt start) (ensure-vt end) (ensure-vt weight)
          :dtype dtype :out out))

(defun vt-cbrt (vt &key out dtype)
  "逐元素立方根。"
  (let* ((dt (%infer-float-dtype vt dtype))
         (third (if (eq dt :float32)
		    (/ 3.0s0)
		    (/ 3.0d0))))
    (vt-map (lambda (x) (* (signum x) (expt (abs x) third)))
            vt :out out :dtype dt)))

(defun vt-hypot (t1 t2 &key out dtype)
  (let* ((dt (or dtype (if (or (eq (vt-dtype (ensure-vt t1)) :float32)
                               (eq (vt-dtype (ensure-vt t2)) :float32))
                           :float32 :float64)))
         (one (if (eq dt :float32) 1.0s0 1.0d0)))
    (vt-map (lambda (a b)
              (let ((abs-a (abs a)) (abs-b (abs b)))
                (cond ((or (%inf-p abs-a) (%inf-p abs-b))
                       ;; NumPy hypot：任一参数 ±Inf → +Inf（即便另一参数为 NaN）
                       (vt-get-pos-inf dt))
                      ((%nan-p abs-a) abs-a)
                      ((%nan-p abs-b) abs-b)
                      ((zerop abs-a) abs-b)
                      ((zerop abs-b) abs-a)
                      (t (let* ((mx (max abs-a abs-b))
                                (mn (min abs-a abs-b))
                                (r  (/ mn mx)))
                           (* mx (sqrt (+ one (* r r)))))))))
            (ensure-vt t1) (ensure-vt t2) :out out :dtype dt)))

(defun vt-reciprocal (vt &key out dtype)
  "逐元素倒数 (1/x)。"
  (let ((dt (%infer-float-dtype vt dtype)))
    (if (eq dt :float32)
        (vt-map (lambda (v) (/ 1.0s0 v)) vt :out out :dtype dt)
        (vt-map (lambda (v) (/ 1.0d0 v)) vt :out out :dtype dt))))

(defun vt-negative (vt &key out dtype)
  (vt-fast-map #'- vt :dtype dtype :out out))

(defun vt-sinc (tensor &key out dtype)
  (let* ((dt (%infer-float-dtype tensor dtype))
	 (x-pi (vt-scale tensor pi :dtype dt)))
    (vt-map (lambda (x)
	      (if (zerop x) 1.0d0 (/ (sin x) x)))
	    x-pi :out out :dtype dt)))
