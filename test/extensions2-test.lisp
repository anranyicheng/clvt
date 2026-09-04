;;;; extensions2-test.lisp — 测试新增函数（输出与run_all_tests兼容的标准格式）
(require :asdf)
(push (truename (make-pathname :directory '(:relative :up))) asdf:*central-registry*)
(asdf:load-system :clvt)
(in-package :clvt)

(defvar *N* 0) (defvar *P* 0) (defvar *F* 0) (defvar *F-list* nil)

(defun ->list (x)
  (cond ((vt-p x) (->list (vt-to-list x)))
        ((numberp x) (float x 1d0))
        ((consp x) (mapcar #'->list x))
        (t x)))

(defun approx (e a &optional (tol 1d-5))
  (let ((el (->list e)) (al (->list a)))
    (cond ((and (numberp el) (numberp al)) (< (abs (- el al)) tol))
          ((and (consp el) (consp al))
           (and (= (length el) (length al)) (every (lambda (x y) (approx x y tol)) el al)))
          (t (equalp el al)))))

(defun check (name expected actual &optional (tol 1d-5))
  (incf *N*)
  (if (approx expected actual tol) (incf *P*)
      (progn (incf *F*) (push name *F-list*)
             (format t "  ❌ ~a~%     exp: ~a~%     got: ~a~%~%" name
                     (let ((e (->list expected))) (if (consp e) (subseq e 0 (min 8 (length e))) e))
                     (let ((g (->list actual))) (if (consp g) (subseq g 0 (min 8 (length g))) g))))))

(defun mk (data)
  (labels ((f (x) (cond ((numberp x) (float x 1d0)) ((consp x) (mapcar #'f x)) (t x)))
           (sh (x) (if (consp x) (cons (length x) (sh (car x))) nil)))
    (let* ((fd (f data)) (s (sh fd)))
      (vt-from-array (make-array s :element-type 'double-float :initial-contents fd) :dtype :float64))))

(defun mki (data)
  (labels ((sh (x) (if (consp x) (cons (length x) (sh (car x))) nil)))
    (let ((s (sh data)))
      (vt-from-array (make-array s :element-type '(signed-byte 64) :initial-contents data) :dtype :int64))))

(defun summary ()
  (format t "~%============================================================~%")
  (format t "  Total: ~a | Pass: ~a | Fail: ~a | Skip: 0~%" *N* *P* *F*)
  (format t "============================================================~%")
  (when *F-list*
    (format t "~%Failed:~{~%  - ~a~}~%" (reverse *F-list*)))
  (zerop *F*))

(format t "~%=== Extensions2 新增函数测试 ===~%~%")

;;; fliplr/flipud
(let ((m (mk '((1d0 2d0 3d0) (4d0 5d0 6d0)))))
  (check "fliplr" '((3d0 2d0 1d0) (6d0 5d0 4d0)) (vt-fliplr m))
  (check "flipud" '((4d0 5d0 6d0) (1d0 2d0 3d0)) (vt-flipud m)))

;;; ediff1d
(check "ediff1d" '(1d0 2d0 3d0) (vt-ediff1d (mk '(1d0 2d0 4d0 7d0))))
(check "ediff1d-2elem" '(1d0) (vt-ediff1d (mk '(1d0 2d0))))
(check "ediff1d-1elem" 0 (vt-size (vt-ediff1d (mk '(5d0)))))

;;; geomspace
(let ((g (vt-geomspace 1d0 1000d0 4)))
  (check "geomspace(1,1000,4)" '(1d0 10d0 100d0 1000d0) g 1d-3)
  (check "geomspace shape" '(4) (vt-shape g)))
(check "geomspace(100,25,3)" '(100d0 50d0 25d0) (vt-geomspace 100d0 25d0 3) 1d-3)
(check "geomspace(-1,-1000,4)" '(-1d0 -10d0 -100d0 -1000d0) (vt-geomspace -1d0 -1000d0 4) 1d-3)

;;; ravel-multi-index
(check "ravel (1,2)/(3,4)" 6 (vt-ravel-multi-index '(1 2) '(3 4)))
(check "ravel (0,0)/(3,4)" 0 (vt-ravel-multi-index '(0 0) '(3 4)))
(check "ravel (2,3)/(3,4)" 11 (vt-ravel-multi-index '(2 3) '(3 4)))
(check "ravel (0,0,0)/(2,3,4)" 0 (vt-ravel-multi-index '(0 0 0) '(2 3 4)))
(check "ravel (1,2,3)/(2,3,4)" 23 (vt-ravel-multi-index '(1 2 3) '(2 3 4)))

;;; tril/triu indices
(multiple-value-bind (r c) (vt-tril-indices 3)
  (check "tril_indices(3) rows" '(0 1 1 2 2 2) r)
  (check "tril_indices(3) cols" '(0 0 1 0 1 2) c))
(multiple-value-bind (r c) (vt-triu-indices 3)
  (check "triu_indices(3) rows" '(0 0 0 1 1 2) r)
  (check "triu_indices(3) cols" '(0 1 2 1 2 2) c))
(multiple-value-bind (r c) (vt-tril-indices 3 :k -1)
  (check "tril k=-1 count" 3 (length (vt-to-list r))))

;;; vander
(check "vander decreasing"
       '((1d0 1d0 1d0) (4d0 2d0 1d0) (9d0 3d0 1d0))
       (vt-vander (mk '(1d0 2d0 3d0))))
(check "vander increasing"
       '((1d0 1d0 1d0) (1d0 2d0 4d0) (1d0 3d0 9d0))
       (vt-vander (mk '(1d0 2d0 3d0)) :increasing t))
(check "vander N=2" '((1d0 1d0) (2d0 1d0) (3d0 1d0))
       (vt-vander (mk '(1d0 2d0 3d0)) :n 2))

;;; one-hot
(let ((oh (vt-one-hot (mki '(0 2 1)) 3)))
  (check "one-hot 1D shape" '(3 3) (vt-shape oh))
  (check "one-hot [0,2,1]" '((1d0 0d0 0d0) (0d0 0d0 1d0) (0d0 1d0 0d0)) oh))
(let ((oh2 (vt-one-hot (mki '((0 1) (2 0))) 3)))
  (check "one-hot 2D shape" '(2 2 3) (vt-shape oh2))
  (check "one-hot [0,0]=1" 1d0 (vt-ref oh2 0 0 0) 1d-6))

;;; standardize
(let ((s (vt-standardize (mk '(1d0 2d0 3d0)))))
  (check "standardize mean≈0" 0d0 (vt-mean s) 1d-6)
  (check "standardize std≈1" 1d0 (vt-std s) 1d-4))
(let ((m (mk '((1d0 2d0 3d0) (4d0 5d0 6d0)))))
  (let ((s (vt-standardize m :axis 1)))
    (check "standardize axis=1 row means≈0" '(0d0 0d0) (vt-mean s :axis 1) 1d-4)
    (check "standardize axis=1 row std≈1" '(1d0 1d0) (vt-std s :axis 1) 1d-3)))

;;; layer-norm
(let ((m (mk '((1d0 2d0 3d0) (4d0 5d0 6d0)))))
  (let ((ln (vt-layer-norm m '(3) :eps 1d-12)))
    (check "layer-norm shape" '(2 3) (vt-shape ln))
    (check "layer-norm per-row mean=0" '(0d0 0d0) (vt-mean ln :axis 1) 1d-4)
    (check "layer-norm per-row std=1" '(1d0 1d0) (vt-std ln :axis 1) 1d-3)))

;;; apply-along-axis
(let ((m (mk '((1d0 2d0 3d0) (4d0 5d0 6d0)))))
  (check "apply-along sum axis=1" '(6d0 15d0) (vt-apply-along-axis #'vt-sum 1 m) 1d-4)
  (check "apply-along mean axis=0" '(2.5d0 3.5d0 4.5d0) (vt-apply-along-axis #'vt-mean 0 m) 1d-4)
  (check "apply-along cumsum shape" '(2 3)
	 (vt-shape (vt-apply-along-axis #'vt-cumsum 1 m)))
  (check "apply-along cumsum content"
	 '((1d0 3d0 6d0) (4d0 9d0 15d0))
	 (vt-apply-along-axis #'vt-cumsum 1 m) 1d-4))

(summary)
