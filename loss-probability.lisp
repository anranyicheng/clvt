(in-package :clvt)

(defun vt-softmax (vt &key (axis -1) dtype out)
  "softmax: exp(x - max) / sum(exp(x - max))
   减去 max 保证数值稳定性; keepdims 保证广播正确"
  (with-float-safe
    (let* ((max-val (vt-amax vt :axis axis :keepdims t :dtype dtype))
           (exp-vt (vt-exp (vt-- vt max-val :dtype dtype) :dtype dtype))
           (sum-exp (vt-sum exp-vt :axis axis :keepdims t :dtype dtype)))
      (vt-/ exp-vt sum-exp :dtype dtype :out out))))

(defun vt-log-softmax (vt &key (axis -1) dtype out)
  "log-softmax: log(softmax(x))，比分开算更数值稳定"
  (with-float-safe
    (let* ((max-val (vt-amax vt :axis axis :keepdims t :dtype dtype))
           (shifted (vt-- vt max-val :dtype dtype))
           (log-sum-exp (vt-log (vt-sum (vt-exp shifted :dtype dtype)
                                        :axis axis :keepdims t :dtype dtype)
                                :dtype dtype)))
      (vt-- shifted log-sum-exp :dtype dtype :out out))))

(defun vt-mean-squared-error (y-true y-pred &key dtype out)
  "均方误差"
  (with-float-safe
    (vt-mean (vt-square (vt-- y-true y-pred :dtype dtype) :dtype dtype)
             :dtype dtype :out out)))

(defun vt-binary-cross-entropy (y-true y-pred &key (eps 1.0d-7) dtype out)
  "二元交叉熵: -[y*log(p) + (1-y)*log(1-p)]"
  (with-float-safe
    (vt-mean (vt-map (lambda (y p)
                       (let* ((pc (max eps (min (- 1.0d0 eps) p)))
                              (omp (max eps (- 1.0d0 pc))))
                         (- (+ (* y (log pc))
                               (* (- 1.0d0 y) (log omp))))))
                     y-true y-pred :dtype dtype)
             :dtype dtype :out out)))

(defun vt-cross-entropy (y-true y-pred &key (eps 1.0d-7) dtype out)
  "多分类交叉熵: -sum(y_true * log(y_pred))"
  (with-float-safe
    (let* ((p-clipped (vt-clip y-pred eps (- 1.0d0 eps) :dtype dtype))
           (log-prob (vt-log p-clipped :dtype dtype))
           (loss-per-sample (vt-- (vt-sum (vt-* y-true log-prob :dtype dtype)
                                          :axis -1 :dtype dtype)
                                  :dtype dtype)))
      (vt-mean loss-per-sample :dtype dtype :out out))))

