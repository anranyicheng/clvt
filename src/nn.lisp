;;;; nn.lisp — 激活函数、损失函数与 softmax

(in-package :clvt)

(defun vt-sigmoid (vt &key dtype out)
  (vt-map (lambda (x) (if (>= x 0) (/ 1.0d0 (+ 1.0d0 (exp (- x))))
                          (let ((e (exp x))) (/ e (+ 1.0d0 e)))))
          vt :dtype dtype :out out))

(defun vt-relu (vt &key dtype out)
  (vt-map (lambda (x) (max 0.0d0 x)) vt :dtype dtype :out out))

(defun vt-leaky-relu (vt &key (alpha 0.01d0) dtype out)
  (vt-map (lambda (x) (if (> x 0.0d0) x (* alpha x))) vt :dtype dtype :out out))

(defun vt-swish (vt &key dtype out)
  (let ((sig (vt-sigmoid vt :dtype dtype)))
    (vt-map #'* vt sig :dtype dtype :out out)))

(defun vt-softplus (vt &key dtype out)
  (vt-map (lambda (x) (if (> x 20.0d0) x (log (+ 1.0d0 (exp x))))) vt :dtype dtype :out out))

(defun vt-gelu (vt &key dtype out)
  (let ((c (sqrt (/ 2.0d0 (coerce pi 'double-float)))))
    (vt-map (lambda (x)
              (let* ((x3 (* x x x)) (inner (+ x (* 0.044715d0 x3)))
                     (tanh-val (tanh (* c inner))))
                (* 0.5d0 x (+ 1.0d0 tanh-val))))
            vt :dtype dtype :out out)))

(defun vt-mish (vt &key dtype out)
  (let ((sp (vt-softplus vt :dtype dtype)))
    (vt-* vt (vt-tanh sp) :dtype dtype :out out)))

(defun vt-hard-tanh (vt &key dtype out)
  (vt-clip vt -1.0d0 1.0d0 :dtype dtype :out out))

(defun vt-hard-sigmoid (vt &key dtype out)
  (let ((scaled (vt-+ (vt-scale vt 0.2d0) 0.5d0 :dtype dtype)))
    (vt-clip scaled 0.0d0 1.0d0 :dtype dtype :out out)))

(defun vt-softmax (vt &key (axis -1) dtype out)
  (let* ((max-val (vt-amax vt :axis axis :keepdims t :dtype dtype))
         (exp-vt (vt-exp (vt-- vt max-val :dtype dtype) :dtype dtype))
         (sum-exp (vt-sum exp-vt :axis axis :keepdims t :dtype dtype)))
    (vt-/ exp-vt sum-exp :dtype dtype :out out)))

(defun vt-log-softmax (vt &key (axis -1) dtype out)
  (let* ((max-val (vt-amax vt :axis axis :keepdims t :dtype dtype))
         (shifted (vt-- vt max-val :dtype dtype))
         (lse (vt-log (vt-sum (vt-exp shifted :dtype dtype) :axis axis :keepdims t :dtype dtype)
                      :dtype dtype)))
    (vt-- shifted lse :dtype dtype :out out)))

(defun vt-mean-squared-error (y-true y-pred &key dtype out)
  (vt-mean (vt-square (vt-- y-true y-pred :dtype dtype) :dtype dtype) :dtype dtype :out out))

(defun vt-binary-cross-entropy (y-true y-pred &key (eps 1.0d-7) dtype out)
  (vt-mean (vt-map (lambda (y p)
                     (let* ((pc (max eps (min (- 1.0d0 eps) p)))
                            (omp (max eps (- 1.0d0 pc))))
                       (- (+ (* y (log pc)) (* (- 1.0d0 y) (log omp))))))
                   y-true y-pred :dtype dtype)
           :dtype dtype :out out))

(defun vt-cross-entropy (y-true y-pred &key (eps 1.0d-7) dtype out)
  (let* ((p-clipped (vt-clip y-pred eps (- 1.0d0 eps) :dtype dtype))
         (log-prob (vt-log p-clipped :dtype dtype))
         (loss-per-sample (vt-- (vt-sum (vt-* y-true log-prob :dtype dtype) :axis -1 :dtype dtype)
                                :dtype dtype)))
    (vt-mean loss-per-sample :dtype dtype :out out)))
