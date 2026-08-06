;;;; nan-random-test.lisp — 测试新增的 nan 和 random 函数
;;;; Usage: sbcl --noinform --non-interactive --eval '(require :asdf)' --eval '(push #p"./" asdf:*central-registry*)' --eval '(asdf:load-system :clvt)' --eval '(load "test/nan-random-test.lisp")'

(require :asdf)
(push (truename (make-pathname :directory '(:relative :up))) asdf:*central-registry*)
(asdf:load-system :clvt)
(in-package :clvt)

;;; ============================================================
;;; 测试框架
;;; ============================================================

(defvar *test-count* 0)
(defvar *pass-count* 0)
(defvar *fail-count* 0)
(defvar *failures* '())

(defmacro test-assert (name expected actual &optional (tol 1e-10))
  `(progn
     (incf *test-count*)
     (let* ((e ,expected)
            (a ,actual)
            (ok (cond
                  ((and (numberp e) (numberp a))
                   (if (and (floatp e) (floatp a))
                       (if (or (vt-float-nan-p e) (vt-float-nan-p a))
                           (and (vt-float-nan-p e) (vt-float-nan-p a))
                           (< (abs (- e a)) ,tol))
                       (equal e a)))
                  ((and (listp e) (listp a))
                   (and (= (length e) (length a))
                        (every (lambda (x y)
                                 (cond
                                   ((and (floatp x) (floatp y))
                                    (if (or (vt-float-nan-p x) (vt-float-nan-p y))
                                        (and (vt-float-nan-p x) (vt-float-nan-p y))
                                        (< (abs (- x y)) ,tol)))
                                   ((and (numberp x) (numberp y)) (equal x y))
                                   ((and (listp x) (listp y))
                                    (and (= (length x) (length y))
                                         (every (lambda (a b)
                                                  (if (and (floatp a) (floatp b))
                                                      (< (abs (- a b)) ,tol)
                                                      (equal a b)))
                                                x y)))
                                   (t (equal x y))))
                               e a)))
                  ((and (vt-p a) (listp e))
                   (let ((al (vt-to-list a)))
                     (and (= (length e) (length al))
                          (every (lambda (x y)
                                   (if (and (floatp x) (floatp y))
                                       (if (or (vt-float-nan-p x) (vt-float-nan-p y))
                                           (and (vt-float-nan-p x) (vt-float-nan-p y))
                                           (< (abs (- x y)) ,tol))
                                       (equal x y)))
                                 e al))))
                  (t (equal e a)))))
       (if ok
           (incf *pass-count*)
           (progn
             (incf *fail-count*)
             (push (format nil "FAIL [~a]: expected ~a, got ~a" ,name e a) *failures*)
             (format t "  ❌ ~a~%" ,name))))))

(defmacro test-assert-float (name expected actual &optional (tol 1e-6))
  `(test-assert ,name ,expected ,actual ,tol))

(defun test-summary ()
  (format t "~%========================================~%")
  (format t "  Total: ~a  Pass: ~a  Fail: ~a~%" *test-count* *pass-count* *fail-count*)
  (format t "========================================~%")
  (when *failures*
    (format t "~%Failures:~%")
    (dolist (f (reverse *failures*))
      (format t "  ~a~%" f)))
  (zerop *fail-count*))

;;; ============================================================
;;; 主测试
;;; ============================================================

(defun run-nan-random-tests ()
  (format t "~%========================================~%")
  (format t "  clvt nan/random Test Suite~%")
  (format t "========================================~%~%")

  ;; === 1. nan/inf predicates ===
  (format t "~%--- 1. nan/inf predicates ---~%")
  (test-assert "nan-p on nan" t (vt-float-nan-p +vt-float-nan+))
  (test-assert "nan-p on 1.0" nil (vt-float-nan-p 1.0d0))
  (test-assert "nan-p on pos-inf" nil (vt-float-nan-p +vt-float-pos-inf+))
  (test-assert "nan-p on neg-inf" nil (vt-float-nan-p +vt-float-neg-inf+))
  (test-assert "pos-inf-p on pos-inf" t (vt-float-pos-inf-p +vt-float-pos-inf+))
  (test-assert "pos-inf-p on neg-inf" nil (vt-float-pos-inf-p +vt-float-neg-inf+))
  (test-assert "pos-inf-p on nan" nil (vt-float-pos-inf-p +vt-float-nan+))
  (test-assert "pos-inf-p on 1.0" nil (vt-float-pos-inf-p 1.0d0))
  (test-assert "neg-inf-p on neg-inf" t (vt-float-neg-inf-p +vt-float-neg-inf+))
  (test-assert "neg-inf-p on pos-inf" nil (vt-float-neg-inf-p +vt-float-pos-inf+))
  (test-assert "neg-inf-p on nan" nil (vt-float-neg-inf-p +vt-float-nan+))
  (test-assert "neg-inf-p on -1.0" nil (vt-float-neg-inf-p -1.0d0))
  (test-assert "sfloat-nan-p" t (vt-float-nan-p +vt-sfloat-nan+))
  (test-assert "sfloat-pos-inf-p" t (vt-float-pos-inf-p +vt-sfloat-pos-inf+))
  (test-assert "sfloat-neg-inf-p" t (vt-float-neg-inf-p +vt-sfloat-neg-inf+))

  ;; vt-isnan on tensor
  (let ((a (vt-from-sequence (list 1.0 +vt-float-nan+ 3.0 +vt-float-nan+) :dtype :float64)))
    (test-assert "isnan tensor" '(0.0 1.0 0.0 1.0) (vt-to-list (vt-isnan a))))

  (let ((a (vt-from-sequence (list 1.0 +vt-float-pos-inf+ +vt-float-neg-inf+ 3.0) :dtype :float64)))
    (test-assert "isinf tensor" '(0.0 1.0 1.0 0.0) (vt-to-list (vt-isinf a))))

  (let ((a (vt-from-sequence (list 1.0 +vt-float-nan+ +vt-float-pos-inf+ 3.0) :dtype :float64)))
    (test-assert "isfinite tensor" '(1.0 0.0 0.0 1.0) (vt-to-list (vt-isfinite a))))

  ;; === 2. vt-nanargmax / vt-nanargmin ===
  (format t "~%--- 2. vt-nanargmax / vt-nanargmin ---~%")

  (let ((a (vt-from-sequence (list 1.0 +vt-float-nan+ 3.0 2.0) :dtype :float64)))
    (test-assert "nanargmax global" 2 (vt-item (vt-nanargmax a)))
    (test-assert "nanargmin global" 0 (vt-item (vt-nanargmin a))))

  (let ((a (vt-from-sequence (list (list 1.0 +vt-float-nan+ 3.0) (list 2.0 5.0 +vt-float-nan+)) :dtype :float64)))
    (test-assert "nanargmax axis=1" '(2 1) (vt-to-list (vt-nanargmax a :axis 1)))
    (test-assert "nanargmin axis=1" '(0 0) (vt-to-list (vt-nanargmin a :axis 1)))
    (test-assert "nanargmax axis=0" '(1 1 0) (vt-to-list (vt-nanargmax a :axis 0)))
    (test-assert "nanargmin axis=0" '(0 1 0) (vt-to-list (vt-nanargmin a :axis 0))))

  ;; === 3. vt-nanprod ===
  (format t "~%--- 3. vt-nanprod ---~%")

  (let ((a (vt-from-sequence (list 2.0 +vt-float-nan+ 3.0 4.0) :dtype :float64)))
    (test-assert-float "nanprod global" 24.0 (vt-item (vt-nanprod a))))

  (let ((a (vt-from-sequence (list (list 2.0 +vt-float-nan+) (list 3.0 4.0)) :dtype :float64)))
    (test-assert-float "nanprod axis=0" '(6.0 4.0) (vt-to-list (vt-nanprod a :axis 0)))
    (test-assert-float "nanprod axis=1" '(2.0 12.0) (vt-to-list (vt-nanprod a :axis 1))))

  (let ((a (vt-from-sequence (list +vt-float-nan+ +vt-float-nan+) :dtype :float64)))
    (test-assert "nanprod all-nan" 1.0 (vt-item (vt-nanprod a))))

  ;; === 4. vt-nanmedian ===
  (format t "~%--- 4. vt-nanmedian ---~%")

  (let ((a (vt-from-sequence (list 1.0 2.0 +vt-float-nan+ 3.0 4.0) :dtype :float64)))
    (test-assert-float "nanmedian global" 2.5 (vt-item (vt-nanmedian a))))

  (let ((a (vt-from-sequence (list 1.0 +vt-float-nan+ 3.0) :dtype :float64)))
    (test-assert-float "nanmedian odd" 2.0 (vt-item (vt-nanmedian a))))

  (let ((a (vt-from-sequence (list +vt-float-nan+ +vt-float-nan+) :dtype :float64)))
    (test-assert "nanmedian all-nan" t (vt-float-nan-p (vt-item (vt-nanmedian a)))))

  (let ((a (vt-from-sequence (list (list 1.0 +vt-float-nan+ 3.0) (list 2.0 5.0 +vt-float-nan+)) :dtype :float64)))
    (test-assert-float "nanmedian axis=1" '(2.0 3.5) (vt-to-list (vt-nanmedian a :axis 1)))
    (test-assert-float "nanmedian axis=0" '(1.5 5.0 3.0) (vt-to-list (vt-nanmedian a :axis 0))))

  ;; === 5. existing nan stats (verify) ===
  (format t "~%--- 5. existing nan stats (verify) ---~%")

  (let ((a (vt-from-sequence (list 1.0 +vt-float-nan+ 3.0 4.0) :dtype :float64)))
    (test-assert-float "nanmean" 2.6666666666666665 (vt-item (vt-nanmean a)))
    (test-assert-float "nansum" 8.0 (vt-item (vt-nansum a)))
    (test-assert-float "nanmax" 4.0 (vt-item (vt-nanmax a)))
    (test-assert-float "nanmin" 1.0 (vt-item (vt-nanmin a))))

  ;; === 6. vt-random-seed ===
  (format t "~%--- 6. vt-random-seed ---~%")

  (vt-random-seed 42)
  (let ((a (vt-random '(3) :dtype :float64)))
    (vt-random-seed 42)
    (let ((b (vt-random '(3) :dtype :float64)))
      (test-assert "random-seed reproducibility" t
                   (equal (vt-to-list a) (vt-to-list b)))))

  ;; === 7. vt-random-int ===
  (format t "~%--- 7. vt-random-int ---~%")

  (vt-random-seed 123)
  (let ((a (vt-random-int 0 10 :size '(100) :dtype :int64)))
    (test-assert "random-int shape" '(100) (vt-shape a))
    (test-assert "random-int range" t
                 (every (lambda (x) (and (>= x 0) (< x 10)))
                        (vt-to-list a))))

  (let ((a (vt-random-int 5 10)))
    (test-assert "random-int scalar" t
                 (and (integerp (vt-item a))
                      (>= (vt-item a) 5)
                      (< (vt-item a) 10))))

  (let ((a (vt-random-int 5 5 :size '(3))))
    (test-assert "random-int zero-range" '(5 5 5) (vt-to-list a)))

  ;; === 8. vt-random-choice ===
  (format t "~%--- 8. vt-random-choice ---~%")

  (vt-random-seed 42)
  (let ((a (vt-random-choice 5 :size '(20) :dtype :int64)))
    (test-assert "choice-from-int shape" '(20) (vt-shape a))
    (test-assert "choice-from-int range" t
                 (every (lambda (x) (and (>= x 0) (< x 5)))
                        (vt-to-list a))))

  (let* ((src (vt-from-sequence '(10 20 30 40 50) :dtype :int64))
         (a (vt-random-choice src :size '(10))))
    (test-assert "choice-from-tensor shape" '(10) (vt-shape a))
    (test-assert "choice-from-tensor values" t
                 (every (lambda (x) (member x '(10 20 30 40 50)))
                        (vt-to-list a))))

  (vt-random-seed 99)
  (let* ((p '(0.7 0.1 0.1 0.05 0.05))
         (a (vt-random-choice 5 :size '(1000) :p p :dtype :int64))
         (counts (make-array 5 :initial-element 0)))
    (dolist (x (vt-to-list a))
      (incf (aref counts x)))
    (test-assert "choice-weighted bias" t
                 (> (aref counts 0) 500)))

  ;; === 9. vt-random-permutation ===
  (format t "~%--- 9. vt-random-permutation ---~%")

  (vt-random-seed 42)
  (let ((p (vt-random-permutation 10)))
    (test-assert "permutation shape" '(10) (vt-shape p))
    (test-assert "permutation is permutation" t
                 (equal (sort (copy-list (vt-to-list p)) #'<)
                        '(0 1 2 3 4 5 6 7 8 9))))

  (vt-random-seed 42)
  (let* ((a (vt-from-sequence '(10 20 30 40 50) :dtype :int64))
         (p (vt-random-permutation a)))
    (test-assert "permutation-tensor shape" '(5) (vt-shape p))
    (test-assert "permutation-tensor elements" t
                 (equal (sort (copy-list (vt-to-list p)) #'<)
                        '(10 20 30 40 50))))

  ;; === 10. vt-random-shuffle ===
  (format t "~%--- 10. vt-random-shuffle ---~%")

  (vt-random-seed 42)
  (let ((a (vt-from-sequence '(1 2 3 4 5 6 7 8 9 10) :dtype :int64)))
    (vt-random-shuffle a)
    (test-assert "shuffle preserves elements" t
                 (equal (sort (copy-list (vt-to-list a)) #'<)
                        '(1 2 3 4 5 6 7 8 9 10)))
    (test-assert "shuffle changes order" t
                 (not (equal (vt-to-list a) '(1 2 3 4 5 6 7 8 9 10)))))

  ;; === 11. vt-random-multinomial ===
  (format t "~%--- 11. vt-random-multinomial ---~%")

  (vt-random-seed 42)
  (let ((result (vt-random-multinomial 100 '(0.5 0.3 0.2))))
    (test-assert "multinomial shape" '(3) (vt-shape result))
    (test-assert "multinomial sum" 100 (vt-item (vt-sum result)))
    (test-assert "multinomial bias" t
                 (let ((vals (vt-to-list result)))
                   (> (first vals) (second vals)))))

  (vt-random-seed 42)
  (let ((result (vt-random-multinomial 10 '(0.5 0.5) :size '(3))))
    (test-assert "multinomial with size" '(3 2) (vt-shape result)))

  ;; === 12. vt-random-uniform / vt-random-normal ===
  (format t "~%--- 12. vt-random-uniform / vt-random-normal ---~%")

  (vt-random-seed 42)
  (let ((a (vt-random-uniform '(1000) :low 0.0d0 :high 1.0d0)))
    (test-assert "uniform shape" '(1000) (vt-shape a))
    (test-assert "uniform range" t
                 (every (lambda (x) (and (>= x 0.0d0) (< x 1.0d0)))
                        (vt-to-list a)))
    (test-assert-float "uniform mean ~0.5" 0.5d0 (vt-item (vt-mean a)) 0.05))

  (vt-random-seed 42)
  (let ((a (vt-random-normal '(10000) :mean 0.0d0 :std 1.0d0)))
    (test-assert "normal shape" '(10000) (vt-shape a))
    (test-assert-float "normal mean ~0" 0.0d0 (vt-item (vt-mean a)) 0.1)
    (test-assert-float "normal std ~1" 1.0d0 (vt-item (vt-std a)) 0.1))

  ;; === 13. Edge cases ===
  (format t "~%--- 13. Edge cases ---~%")

  ;; float32 nan operations
  (let ((a (vt-from-sequence (list 1.0s0 +vt-sfloat-nan+ 3.0s0) :dtype :float32)))
    (test-assert-float "nanmean-float32" 2.0s0 (vt-item (vt-nanmean a)) 1e-5)
    (test-assert-float "nansum-float32" 4.0s0 (vt-item (vt-nansum a)) 1e-5))

  ;; Integer tensor with nan functions (should pass through)
  (let ((a (vt-from-sequence '(1 2 3 4 5) :dtype :int64)))
    (test-assert "nanargmax-int" 4 (vt-item (vt-nanargmax a)))
    (test-assert "nanargmin-int" 0 (vt-item (vt-nanargmin a)))
    (test-assert "nanprod-int" 120 (vt-item (vt-nanprod a))))

  ;; === 14. Single-float inf predicates ===
  (format t "~%--- 14. Single-float inf predicates ---~%")
  (test-assert "sfloat inf-p on pos-inf" t (vt-float-inf-p +vt-sfloat-pos-inf+))
  (test-assert "sfloat inf-p on neg-inf" t (vt-float-inf-p +vt-sfloat-neg-inf+))
  (test-assert "sfloat inf-p on nan" nil (vt-float-inf-p +vt-sfloat-nan+))
  (test-assert "sfloat inf-p on 1.0" nil (vt-float-inf-p 1.0s0))
  (test-assert "sfloat pos-inf-p" t (vt-float-pos-inf-p +vt-sfloat-pos-inf+))
  (test-assert "sfloat neg-inf-p" t (vt-float-neg-inf-p +vt-sfloat-neg-inf+))
  (test-assert "sfloat pos-inf-p on neg" nil (vt-float-pos-inf-p +vt-sfloat-neg-inf+))
  (test-assert "sfloat neg-inf-p on pos" nil (vt-float-neg-inf-p +vt-sfloat-pos-inf+))

  ;; float-inf-= edge cases
  (test-assert "inf-= two pos-inf" t (vt-float-inf-= +vt-float-pos-inf+ +vt-float-pos-inf+))
  (test-assert "inf-= pos vs neg" nil (vt-float-inf-= +vt-float-pos-inf+ +vt-float-neg-inf+))
  (test-assert "inf-= nan vs inf" nil (vt-float-inf-= +vt-float-nan+ +vt-float-pos-inf+))
  (test-assert "inf-= nan vs nan" nil (vt-float-inf-= +vt-float-nan+ +vt-float-nan+))

  ;; === 15. Random validation tests ===
  (format t "~%--- 15. Random validation ---~%")

  ;; vt-random-uniform: low >= high should error
  (let ((caught nil))
    (handler-case (vt-random-uniform '(3) :low 5.0d0 :high 3.0d0)
      (error () (setf caught t)))
    (test-assert "uniform low>=high error" t caught))

  ;; vt-random-choice: p length mismatch should error
  (let ((caught nil))
    (handler-case (vt-random-choice 5 :size '(3) :p '(0.5 0.5))
      (error () (setf caught t)))
    (test-assert "choice p-length mismatch error" t caught))

  ;; vt-random-choice: negative probability should error
  (let ((caught nil))
    (handler-case (vt-random-choice 3 :size '(3) :p '(0.5 -0.2 0.7))
      (error () (setf caught t)))
    (test-assert "choice negative-p error" t caught))

  ;; vt-random-choice: zero sum should error
  (let ((caught nil))
    (handler-case (vt-random-choice 3 :size '(3) :p '(0.0 0.0 0.0))
      (error () (setf caught t)))
    (test-assert "choice zero-sum-p error" t caught))

  ;; vt-random-multinomial: empty pvals should error
  (let ((caught nil))
    (handler-case (vt-random-multinomial 10 '())
      (error () (setf caught t)))
    (test-assert "multinomial empty-pvals error" t caught))

  ;; vt-random-multinomial: negative prob should error
  (let ((caught nil))
    (handler-case (vt-random-multinomial 10 '(0.5 -0.1 0.6))
      (error () (setf caught t)))
    (test-assert "multinomial negative-p error" t caught))

  ;; vt-random-normal with dtype :float32
  (vt-random-seed 42)
  (let ((a (vt-random-normal '(100) :dtype :float32)))
    (test-assert "normal float32 dtype" :float32 (vt-dtype a)))

  ;; vt-random-permutation edge: n=0
  (let ((p (vt-random-permutation 0)))
    (test-assert "permutation n=0 shape" '(0) (vt-shape p)))

  ;; vt-random-permutation edge: n=1
  (let ((p (vt-random-permutation 1)))
    (test-assert "permutation n=1" '(0) (vt-to-list p)))

  ;; === Summary ===
  (test-summary))

;; Run
(run-nan-random-tests)
