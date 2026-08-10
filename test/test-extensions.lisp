;;;; test-extensions.lisp — 测试新增的扩展函数

(load "~/quicklisp/setup.lisp")
(ql:quickload :clvt)

(in-package :clvt)

(defvar *pass* 0)
(defvar *fail* 0)

(defmacro check (name expr expected)
  `(let ((result ,expr))
     (if (equal result ,expected)
         (progn (incf *pass*) (format t "[PASS] ~a~%" ,name))
         (progn (incf *fail*)
                (format t "[FAIL] ~a: expected ~a, got ~a~%" ,name ,expected result)))))

(defmacro check-approx (name expr expected &optional (tol 1e-6))
  `(let ((result ,expr))
     (if (< (abs (- result ,expected)) ,tol)
         (progn (incf *pass*) (format t "[PASS] ~a~%" ,name))
         (progn (incf *fail*)
                (format t "[FAIL] ~a: expected ~a, got ~a~%" ,name ,expected result)))))

(format t "~%=== Testing New Extension Functions ===~%")

;; 1. vt-count-nonzero
(let ((m (vt-from-sequence (list (list 0 1 0) (list 3 0 5)) :dtype :float64)))
  (check "count-nonzero global"
         (vt-item (vt-count-nonzero m))
         3)
  (check "count-nonzero axis=1"
         (vt-to-list (vt-count-nonzero m :axis 1))
         '(1 2))
  (check "count-nonzero axis=0"
         (vt-to-list (vt-count-nonzero m :axis 0))
         '(1 1 1)))

;; 2. vt-moveaxis
(let ((m (vt-zeros (list 2 3 4) :dtype :float64)))
  (check "moveaxis 0->2 shape"
         (vt-shape (vt-moveaxis m 0 2))
         '(3 4 2))
  (check "moveaxis 2->0 shape"
         (vt-shape (vt-moveaxis m 2 0))
         '(4 2 3))
  (check "moveaxis (0 2)->(2 0) shape"
         (vt-shape (vt-moveaxis m '(0 2) '(2 0)))
         '(4 3 2)))

;; 3. vt-inner
(let ((a (vt-from-sequence (list 1 2 3) :dtype :float64))
      (b (vt-from-sequence (list 4 5 6) :dtype :float64)))
  (check-approx "inner 1D"
         (vt-item (vt-inner a b))
         32.0d0))

(let ((a (vt-from-sequence (list (list 1 2 3) (list 4 5 6)) :dtype :float64))
      (b (vt-from-sequence (list (list 7 8 9) (list 10 11 12)) :dtype :float64)))
  (check "inner 2D shape"
         (vt-shape (vt-inner a b))
         '(2 2))
  (check-approx "inner 2D [0,0]"
         (vt-ref (vt-inner a b) 0 0)
         50.0d0))

;; 4. vt-tensordot
(let ((a (vt-from-sequence (list (list 1 2) (list 3 4)) :dtype :float64))
      (b (vt-from-sequence (list (list 5 6) (list 7 8)) :dtype :float64)))
  (check-approx "tensordot axes=2 (scalar)"
         (vt-item (vt-tensordot a b :axes 2))
         70.0d0)
  (check "tensordot axes=1 shape"
         (vt-shape (vt-tensordot a b :axes 1))
         '(2 2)))

;; 5. vt-topk
(let ((m (vt-from-sequence (list 3 1 4 1 5 9 2 6) :dtype :float64)))
  (multiple-value-bind (vals idxs) (vt-topk m 3)
    (check "topk 3 vals"
           (vt-to-list vals)
           '(9.0d0 6.0d0 5.0d0))
    (check "topk 3 idxs"
           (vt-to-list idxs)
           '(5 7 4)))
  (multiple-value-bind (vals idxs) (vt-topk m 3 :largest nil)
    (check "topk 3 smallest vals"
           (vt-to-list vals)
           '(1.0d0 1.0d0 2.0d0))))

;; 6. vt-flatnonzero
(let ((m (vt-from-sequence (list 0 1 0 3 0 5) :dtype :float64)))
  (check "flatnonzero"
         (vt-to-list (vt-flatnonzero m))
         '(1 3 5)))

;; 7. vt-count
(let ((m (vt-from-sequence (list 1 2 3 2 2 4) :dtype :float64)))
  (check "count 2"
         (vt-item (vt-count m 2))
         3)
  (check "count 5 (not found)"
         (vt-item (vt-count m 5))
         0))

;; 8. vt-clip-tensor
(let ((m (vt-from-sequence (list 1 5 10 15 20) :dtype :float64)))
  (check "clip-tensor"
         (vt-to-list (vt-clip-tensor m 3 12))
         '(3.0d0 5.0d0 10.0d0 12.0d0 12.0d0)))

;; 9. vt-set-print-options / vt-get-print-options
(vt-set-print-options :threshold 500 :precision 2)
(check "print-options"
       (vt-get-print-options)
       '(500 2 1))
;; Restore
(vt-set-print-options :threshold 3 :precision 6 :indent-step 1)

(format t "~%=== Results: ~a PASS, ~a FAIL ===~%" *pass* *fail*)
(when (> *fail* 0)
  (sb-ext:exit :code 1))
(sb-ext:exit :code 0)
