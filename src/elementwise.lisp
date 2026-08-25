;;;; elementwise.lisp — 逐元素算术 / 数学 / 比较 / 逻辑 / 位运算

(in-package :clvt)

(defun vt-+ (&rest args)
  (with-float-safe
    (case (length args)
      (1 (vt-fast-map #'+ (first args)))
      (2 (vt-fast-map #'+ (first args) (second args)))
      (t (reduce #'vt-+ args)))))

(defun vt-* (&rest args)
  (with-float-safe
    (case (length args)
      (1 (vt-fast-map #'* (first args)))
      (2 (vt-fast-map #'* (first args) (second args)))
      (t (reduce #'vt-* args)))))

(defun vt-- (vt &rest args)
  (with-float-safe
    (let ((first (ensure-vt vt)))
      (cond ((null args) (vt-fast-map #'- first))
            ((null (cdr args)) (vt-fast-map #'- first (first args)))
            (t (reduce #'vt-- (cons first args)))))))

(defun vt-/ (vt &rest args)
  (with-float-safe
    (let ((first (ensure-vt vt)))
      (cond ((null args) (vt-map (lambda (v) (/ 1.0d0 v)) first))
            ((null (cdr args)) (vt-fast-map #'/ first (first args)))
            (t (reduce #'vt-/ (cons first args)))))))

(defun vt-add (a b &key dtype out) (vt-fast-map #'+ a b :dtype dtype :out out))
(defun vt-sub (a b &key dtype out) (vt-fast-map #'- a b :dtype dtype :out out))
(defun vt-mul (a b &key dtype out) (vt-fast-map #'* a b :dtype dtype :out out))
(defun vt-div (a b &key dtype out) (vt-fast-map #'/ a b :dtype dtype :out out))
(defun vt-scale (a b &key out dtype) (vt-fast-map #'* a b :out out :dtype dtype))

(defun %infer-float-dtype (vt dtype)
  (or dtype (if (eq (vt-dtype vt) :float32) :float32 :float64)))

(defun vt-sin (vt &key out dtype) (vt-fast-map #'sin vt :out out :dtype (%infer-float-dtype vt dtype)))
(defun vt-cos (vt &key out dtype) (vt-fast-map #'cos vt :out out :dtype (%infer-float-dtype vt dtype)))
(defun vt-tan (vt &key out dtype) (vt-fast-map #'tan vt :out out :dtype (%infer-float-dtype vt dtype)))
(defun vt-atan (vt &key out dtype) (vt-fast-map #'atan vt :out out :dtype (%infer-float-dtype vt dtype)))
(defun vt-sinh (vt &key out dtype) (vt-fast-map #'sinh vt :out out :dtype (%infer-float-dtype vt dtype)))
(defun vt-cosh (vt &key out dtype) (vt-fast-map #'cosh vt :out out :dtype (%infer-float-dtype vt dtype)))
(defun vt-tanh (vt &key out dtype) (vt-fast-map #'tanh vt :out out :dtype (%infer-float-dtype vt dtype)))

(defun vt-asin (vt &key out dtype)
  (let* ((dt (%infer-float-dtype vt dtype)) (nan (vt-get-nan dt)))
    (vt-map (lambda (x) (if (> (abs x) 1.0d0) nan (asin x))) vt :out out :dtype dt)))

(defun vt-acos (vt &key out dtype)
  (let* ((dt (%infer-float-dtype vt dtype)) (nan (vt-get-nan dt)))
    (vt-map (lambda (x) (if (> (abs x) 1.0d0) nan (acos x))) vt :out out :dtype dt)))

(defun vt-asinh (vt &key out dtype) (vt-fast-map #'asinh vt :out out :dtype (%infer-float-dtype vt dtype)))

(defun vt-acosh (vt &key out dtype)
  (let* ((dt (%infer-float-dtype vt dtype)) (nan (vt-get-nan dt)))
    (vt-map (lambda (x) (if (< x 1.0d0) nan (acosh x))) vt :out out :dtype dt)))

(defun vt-atanh (vt &key out dtype)
  (let* ((dt (%infer-float-dtype vt dtype)) (nan (vt-get-nan dt)))
    (vt-map (lambda (x) (if (>= (abs x) 1.0d0) nan (atanh x))) vt :out out :dtype dt)))

(defun vt-exp (vt &key out dtype) (vt-fast-map #'exp vt :out out :dtype (%infer-float-dtype vt dtype)))

(defun vt-pow (vt power &key out dtype)
  (let* ((dt (%infer-float-dtype vt dtype))
         (nan (vt-get-nan dt)))
    (vt-map (lambda (x)
              (let ((result (handler-case
                                (expt x power)
                              (error () nan))))
                (if (realp result)
                    result
                    nan)))
            vt :out out :dtype dt)))

(defun vt-expt (vt power &key out dtype) (vt-pow vt power :out out :dtype dtype))
(defun vt-square (vt &key out dtype) (vt-pow vt 2 :out out :dtype dtype))

(defun vt-sqrt (vt &key out dtype)
  (let* ((dt (%infer-float-dtype vt dtype)) (nan (vt-get-nan dt)))
    (vt-map (lambda (x) (if (minusp x) nan (sqrt x))) vt :out out :dtype dt)))

(defun vt-log (vt &key base out dtype)
  (let* ((dt (%infer-float-dtype vt dtype))
         (nan (vt-get-nan dt)) (neginf (vt-get-neg-inf dt)) (posinf (vt-get-pos-inf dt)))
    (cond
      ((and base (or (<= base 0) (= base 1)))
       (vt-map (lambda (x) (declare (ignore x)) nan) vt :out out :dtype dt))
      ((null base)
       (vt-map (lambda (x) (if (> x 0) (log x) (if (zerop x) neginf nan))) vt :out out :dtype dt))
      (t
       (let ((zero-result (if (plusp (log base)) neginf posinf)))
         (vt-map (lambda (x) (if (> x 0) (log x base) (if (zerop x) zero-result nan)))
                 vt :out out :dtype dt))))))

(defun vt-log10 (vt &key out dtype) (vt-log vt :base 10.0d0 :out out :dtype dtype))
(defun vt-log2 (vt &key out dtype) (vt-log vt :base 2.0d0 :out out :dtype dtype))

(defun vt-abs (vt &key out dtype) (vt-fast-map #'abs vt :out out :dtype dtype))
(defun vt-signum (vt &key out dtype) (vt-fast-map #'signum vt :out out :dtype dtype))

(defun vt-positive-p (vt &key out (dtype :float64))
  (vt-map (lambda (v) (if (> v 0.0d0) 1.0d0 0.0d0)) vt :out out :dtype dtype))
(defun vt-negative-p (vt &key out (dtype :float64))
  (vt-map (lambda (v) (if (< v 0.0d0) 1.0d0 0.0d0)) vt :out out :dtype dtype))
(defun vt-zero-p (vt &key out (dtype :float64))
  (vt-map (lambda (v) (if (zerop v) 1.0d0 0.0d0)) vt :out out :dtype dtype))
(defun vt-nonzero-p (vt &key out (dtype :float64))
  (vt-map (lambda (v) (if (zerop v) 0.0d0 1.0d0)) vt :out out :dtype dtype))
(defun vt-even-p (vt &key out (dtype :float64))
  (vt-map (lambda (v) (if (evenp (floor v)) 1.0d0 0.0d0)) vt :out out :dtype dtype))
(defun vt-odd-p (vt &key out (dtype :float64))
  (vt-map (lambda (v) (if (oddp (floor v)) 1.0d0 0.0d0)) vt :out out :dtype dtype))

(defun vt-mod (vt divisor &key out dtype) (vt-map (lambda (x) (mod x divisor)) vt :out out :dtype dtype))
(defun vt-rem (vt divisor &key out dtype) (vt-map (lambda (x) (rem x divisor)) vt :out out :dtype dtype))
(defun vt-atan2 (vty vtx &key out dtype) (vt-fast-map #'atan vty vtx :out out :dtype dtype))

(defun vt-floor (vt &key (divisor 1) out dtype)
  (vt-map (lambda (x) (let ((res (nth-value 0 (floor x divisor))))
                        (if (floatp x) (float res x) res))) vt :out out :dtype dtype))
(defun vt-ceiling (vt &key (divisor 1) out dtype)
  (vt-map (lambda (x) (let ((res (nth-value 0 (ceiling x divisor))))
                        (if (floatp x) (float res x) res))) vt :out out :dtype dtype))
(defun vt-round (vt &key (divisor 1) out dtype)
  (vt-map (lambda (x) (let ((res (nth-value 0 (round x divisor))))
                        (if (floatp x) (float res x) res))) vt :out out :dtype dtype))
(defun vt-truncate (vt &key (divisor 1) out dtype)
  (vt-map (lambda (x) (let ((res (nth-value 0 (truncate x divisor))))
                        (if (floatp x) (float res x) res))) vt :out out :dtype dtype))
(defun vt-rint (vt &key out dtype)
  (vt-map (lambda (x) (let ((res (nth-value 0 (round x))))
                        (if (floatp x) (float res x) res))) vt :out out :dtype dtype))

(defun vt-= (t1 t2 &key (dtype :float64) out)
  (vt-map (lambda (a b) (if (= a b) 1.0d0 0.0d0)) (ensure-vt t1) (ensure-vt t2) :dtype dtype :out out))
(defun vt-/= (t1 t2 &key (dtype :float64) out)
  (vt-map (lambda (a b) (if (/= a b) 1.0d0 0.0d0)) (ensure-vt t1) (ensure-vt t2) :dtype dtype :out out))
(defun vt-< (t1 t2 &key (dtype :float64) out)
  (vt-map (lambda (a b) (if (< a b) 1.0d0 0.0d0)) (ensure-vt t1) (ensure-vt t2) :dtype dtype :out out))
(defun vt-<= (t1 t2 &key (dtype :float64) out)
  (vt-map (lambda (a b) (if (<= a b) 1.0d0 0.0d0)) (ensure-vt t1) (ensure-vt t2) :dtype dtype :out out))
(defun vt-> (t1 t2 &key (dtype :float64) out)
  (vt-map (lambda (a b) (if (> a b) 1.0d0 0.0d0)) (ensure-vt t1) (ensure-vt t2) :dtype dtype :out out))
(defun vt->= (t1 t2 &key (dtype :float64) out)
  (vt-map (lambda (a b) (if (>= a b) 1.0d0 0.0d0)) (ensure-vt t1) (ensure-vt t2) :dtype dtype :out out))

(defun vt-rad2deg (vt &key out dtype)
  (let ((factor (float (/ 180.0 pi) 1.0d0)))
    (vt-map (lambda (x) (* x factor)) vt :out out :dtype (%infer-float-dtype vt dtype))))
(defun vt-deg2rad (vt &key out dtype)
  (let ((factor (float (/ pi 180.0) 1.0d0)))
    (vt-map (lambda (x) (* x factor)) vt :out out :dtype (%infer-float-dtype vt dtype))))

(defun vt-maximum (t1 t2 &key out dtype)
  (vt-map (lambda (a b) (cond ((%nan-p a) a) ((%nan-p b) b) (t (max a b))))
          t1 t2 :out out :dtype dtype))
(defun vt-minimum (t1 t2 &key out dtype)
  (vt-map (lambda (a b) (cond ((%nan-p a) a) ((%nan-p b) b) (t (min a b))))
          t1 t2 :out out :dtype dtype))
(defun vt-fmax (t1 t2 &key out dtype)
  (vt-map (lambda (a b) (cond ((and (%nan-p a) (%nan-p b)) a)
                              ((%nan-p a) b) ((%nan-p b) a) (t (max a b))))
          (ensure-vt t1) (ensure-vt t2) :dtype dtype :out out))
(defun vt-fmin (t1 t2 &key out dtype)
  (vt-map (lambda (a b) (cond ((and (%nan-p a) (%nan-p b)) a)
                              ((%nan-p a) b) ((%nan-p b) a) (t (min a b))))
          (ensure-vt t1) (ensure-vt t2) :dtype dtype :out out))

(defun vt-logical-and (t1 t2 &key out (dtype :float64))
  (vt-map (lambda (a b) (if (and (not (zerop a)) (not (zerop b))) 1.0d0 0.0d0))
          (ensure-vt t1) (ensure-vt t2) :dtype dtype :out out))
(defun vt-logical-or (t1 t2 &key out (dtype :float64))
  (vt-map (lambda (a b) (if (or (not (zerop a)) (not (zerop b))) 1.0d0 0.0d0))
          (ensure-vt t1) (ensure-vt t2) :dtype dtype :out out))
(defun vt-logical-not (vt &key out (dtype :float64))
  (vt-map (lambda (v) (if (zerop v) 1.0d0 0.0d0)) vt :dtype dtype :out out))
(defun vt-logical-xor (t1 t2 &key out (dtype :float64))
  (vt-map (lambda (a b) (if (not (eq (not (zerop a)) (not (zerop b)))) 1.0d0 0.0d0))
          (ensure-vt t1) (ensure-vt t2) :dtype dtype :out out))

(defun vt-bit-and (t1 t2 &key out dtype) (vt-map #'logand (ensure-vt t1) (ensure-vt t2) :dtype dtype :out out))
(defun vt-bit-ior (t1 t2 &key out dtype) (vt-map #'logior (ensure-vt t1) (ensure-vt t2) :dtype dtype :out out))
(defun vt-bit-xor (t1 t2 &key out dtype) (vt-map #'logxor (ensure-vt t1) (ensure-vt t2) :dtype dtype :out out))
(defun vt-bit-not (vt &key out dtype) (vt-map #'lognot vt :dtype dtype :out out))
(defun vt-left-shift (vt shift &key out dtype) (vt-map (lambda (x) (ash x shift)) vt :dtype dtype :out out))
(defun vt-right-shift (vt shift &key out dtype) (vt-map (lambda (x) (ash x (- shift))) vt :dtype dtype :out out))

(defun vt-clip (vt min-val max-val &key out dtype)
  (vt-map (lambda (x) (max min-val (min max-val x))) vt :dtype dtype :out out))

(defun vt-lerp (start end weight &key out dtype)
  (vt-map (lambda (s e w) (+ s (* (- e s) w))) (ensure-vt start) (ensure-vt end) (ensure-vt weight)
          :dtype dtype :out out))

(defun vt-cbrt (vt &key out dtype)
  (let* ((dt (%infer-float-dtype vt dtype)))
    (vt-map (lambda (x) (if (minusp x) (- (expt (- x) (/ 3.0d0))) (expt x (/ 3.0d0))))
            vt :out out :dtype dt)))

(defun vt-hypot (t1 t2 &key out dtype)
  (let ((dt (or dtype (if (or (eq (vt-dtype t1) :float32) (eq (vt-dtype t2) :float32)) :float32 :float64))))
    (vt-map (lambda (a b)
              (let ((abs-a (abs a)) (abs-b (abs b)))
                (cond ((zerop abs-a) abs-b) ((zerop abs-b) abs-a)
                      (t (* (max abs-a abs-b) (sqrt (+ 1.0d0 (expt (/ (min abs-a abs-b) (max abs-a abs-b)) 2))))))))
            (ensure-vt t1) (ensure-vt t2) :out out :dtype dt)))

(defun vt-reciprocal (vt &key out dtype)
  (vt-map (lambda (v) (/ 1.0d0 v)) vt :out out :dtype (%infer-float-dtype vt dtype)))

(defun vt-negative (vt &key out dtype)
  (vt-fast-map #'- vt :dtype dtype :out out))

(defun vt-sinc (tensor &key out dtype)
  (let* ((dt (%infer-float-dtype tensor dtype)) (x-pi (vt-scale tensor pi :dtype dt)))
    (vt-map (lambda (x) (if (zerop x) 1.0d0 (/ (sin x) x))) x-pi :out out :dtype dt)))
