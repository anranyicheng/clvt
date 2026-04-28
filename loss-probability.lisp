(in-package :clvt)

(defun vt-softmax (vt &key (axis -1))
"Softmax: exp(x - max) / sum(exp(x - max))
 关键: 减去 max 保证数值稳定性; keepdims 保证广播正确"
  (let* ((max-val (vt-amax vt :axis axis :keepdims t))
         (exp-vt (vt-exp (vt-- vt max-val)))
         (sum-exp (vt-sum exp-vt :axis axis :keepdims t)))
    (vt-/ exp-vt sum-exp)))

(defun vt-log-softmax (vt &key (axis -1))
  "Log-Softmax: log(softmax(x))，比分开算更数值稳定"
  (let* ((max-val (vt-amax vt :axis axis :keepdims t))
         (shifted (vt-- vt max-val))
         (log-sum-exp (vt-log (vt-sum (vt-exp shifted)
				      :axis axis :keepdims t))))
    (vt-- shifted log-sum-exp)))

(defun vt-mean-squared-error (y-true y-pred)
  "均方误差"
  (vt-mean (vt-square (vt-- y-true y-pred))))

(defun vt-binary-cross-entropy (y-true y-pred &key (eps 1e-7))
  "二元交叉熵: -[y*log(p) + (1-y)*log(1-p)]"
  (let* ((p-clipped (vt-clip y-pred eps (- 1.0d0 eps)))
         (one-minus-p (vt-clip (vt-- 1.0d0 p-clipped) eps 1.0d0))
         (term1 (vt-* y-true (vt-log p-clipped)))
         (term2 (vt-* (vt-- 1.0d0 y-true)
		      (vt-log one-minus-p)))
         (loss (vt-- (vt-+ term1 term2))))
    (vt-mean loss)))

(defun vt-cross-entropy (y-true y-pred &key (eps 1e-7))
  "多分类交叉熵: -sum(y_true * log(y_pred))"
  (let* ((p-clipped (vt-clip y-pred eps (- 1.0d0 eps)))
         (log-prob (vt-log p-clipped))
         (loss-per-sample (vt-- (vt-sum (vt-* y-true log-prob) :axis -1))))
    (vt-mean loss-per-sample)))
