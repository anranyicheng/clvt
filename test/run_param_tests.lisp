;;;; run_param_tests.lisp — 参数化 3D+ 多轴测试
;;;; 实时调用 Python (ref_compute.py) 生成 NumPy 参考值，不依赖静态 JSON
;;;; Usage: sbcl --noinform --non-interactive --load test/run_param_tests.lisp

(require :asdf)
(push (truename (make-pathname :directory '(:relative :up))) asdf:*central-registry*)
(asdf:load-system :clvt)
(in-package :clvt)

;;; ============================================================
;;; 通过子进程实时调用 ref_compute.py 获取 NumPy 参考值 + 内置JSON解析器
;;; ============================================================
(defun parse-json-string (s)
  (let ((pos 0) (len (length s)))
    (labels ((skip () (loop while (and (< pos len) (member (char s pos) '(#\Space #\Tab #\Newline #\Return))) do (incf pos)))
             (rd () (skip) (when (>= pos len) (return-from rd nil))
               (let ((c (char s pos)))
                 (cond
                   ((char= c #\") (incf pos) (let ((st pos)) (loop while (and (< pos len) (char/= (char s pos) #\")) do (when (char= (char s pos) #\\) (incf pos)) (incf pos)) (prog1 (subseq s st pos) (incf pos))))
                   ((char= c #\[) (incf pos) (skip) (let ((a nil)) (loop while (and (< pos len) (char/= (char s pos) #\])) do (push (rd) a) (skip) (when (and (< pos len) (char= (char s pos) #\,)) (incf pos) (skip))) (incf pos) (nreverse a)))
                   ((char= c #\{) (incf pos) (skip) (let ((h (make-hash-table :test 'equal))) (loop while (and (< pos len) (char/= (char s pos) #\})) do (let ((k (rd))) (skip) (when (and (< pos len) (char= (char s pos) #\:)) (incf pos)) (setf (gethash k h) (rd))) (skip) (when (and (< pos len) (char= (char s pos) #\,)) (incf pos) (skip))) (incf pos) h))
                   ((or (char<= #\0 c #\9) (char= c #\-) (char= c #\+)) (let ((st pos)) (when (member c '(#\- #\+)) (incf pos)) (loop while (and (< pos len) (or (char<= #\0 (char s pos) #\9) (char= (char s pos) #\.) (member (char s pos) '(#\e #\E #\d #\D #\- #\+)))) do (incf pos)) (let* ((s0 (subseq s st pos)) (s1 (substitute #\e #\d (substitute #\e #\D s0)))) (if (or (find #\. s1) (find #\e s1) (find #\E s1)) (let ((*read-eval* nil)) (read-from-string s1)) (parse-integer s1)))))
                   ((and (< (+ pos 4) len) (string= (subseq s pos (min len (+ pos 4))) "true")) (incf pos 4) t)
                   ((and (< (+ pos 5) len) (string= (subseq s pos (min len (+ pos 5))) "false")) (incf pos 5) nil)
                   ((and (< (+ pos 4) len) (string= (subseq s pos (min len (+ pos 4))) "null")) (incf pos 4) nil)
                   (t (incf pos) nil)))))
      (rd))))

(defun load-expected ()
  "实时调用 python3 test/ref_compute.py 生成参考值，解析为 hash-table。"
  (let* ((script (namestring (merge-pathnames "ref_compute.py" *load-truename*)))
         (output (uiop:run-program (list "python3" script) :output :string
                                   :error-output :interactive)))
    (parse-json-string output)))

;;; ============================================================
;;; 测试框架
;;; ============================================================
(defvar *N* 0) (defvar *P* 0) (defvar *F* 0) (defvar *F-list* nil)

(defun approx (e a &optional (tol 1e-5))
  (cond ((and (null e) (null a)) t)
        ((and (numberp e) (numberp a))
         (let ((ev (if (realp e) (float e 1.0d0) e))
               (av (if (realp a) (float a 1.0d0) a)))
           (cond ((or (and (floatp ev) (not (< ev ev)) (floatp av) (not (< av av)))) t)
                 ((and (floatp ev) (floatp av))
                  (< (abs (- ev av)) (+ tol (* 0.001 (abs ev)))))
                 (t (= (round ev) (round av))))))
        ((and (listp e) (listp a))
         (and (= (length e) (length a)) (every (lambda (x y) (approx x y tol)) e a)))
        ((and (vectorp e) (vectorp a))
         (approx (coerce e 'list) (coerce a 'list) tol))
        (t (equal e a))))

(defun E (J key)
  "从JSON hash-table中取出key对应的值。"
  (let ((v (gethash key J)))
    (cond ((hash-table-p v)
           (or (gethash "scalar" v) (gethash "data" v) (gethash "v" v) v))
          (t v))))

(defun T! (name expected actual &optional (tol 1e-5))
  (incf *N*)
  (if (approx expected actual tol)
      (incf *P*)
      (progn (incf *F*) (push (cons name expected) *F-list*)
             (format t "  ✗ ~a: expected ~a... got ~a...~%" name
                     (if (listp expected) (subseq (write-to-string expected) 0 (min 60 (length (write-to-string expected)))) expected)
                     (if (listp actual) (subseq (write-to-string actual) 0 (min 60 (length (write-to-string actual)))) actual)))))

(defun summary ()
  (format t "~%=== Parametric Tests: Total ~a | Pass ~a | Fail ~a ===~%" *N* *P* *F*)
  (when *F-list*
    (format t "Failed:~%")
    (dolist (f (reverse *F-list*)) (format t "  - ~a~%" (car f))))
  (zerop *F*))

;;; ============================================================
;;; 主测试体
;;; ============================================================
(defun run-param-tests ()
  (format t "~%=== clvt PARAMETRIC TESTS (3D+, multi-axis, 实时NumPy参考) ===~%")
  (let ((J (load-expected))
        (m (vt-astype (vt-reshape (vt-arange 12 :dtype :int64) '(3 4)) :float64))
        (mi (vt-reshape (vt-arange 12 :dtype :int64) '(3 4)))
        (t3 (vt-astype (vt-reshape (vt-arange 24 :dtype :int64) '(2 3 4)) :float64))
        (ti (vt-reshape (vt-arange 24 :dtype :int64) '(2 3 4)))
        (t222 (vt-from-sequence '(((1.0 2.0) (3.0 4.0)) ((5.0 6.0) (7.0 8.0)))))
        (t222i (vt-from-sequence '(((1 2) (3 4)) ((5 6) (7 8))) :dtype :int64)))

    ;; --- Reduction 2D ---
    (T! "sum_2d_all"     (E J "sum_2d_all")     (vt-item (vt-sum m)) 1e-6)
    (T! "sum_2d_ax0"     (E J "sum_2d_ax0")     (vt-to-list (vt-sum m :axis 0)) 1e-6)
    (T! "sum_2d_ax1"     (E J "sum_2d_ax1")     (vt-to-list (vt-sum m :axis 1)) 1e-6)
    (T! "sum_2d_ax-1"    (E J "sum_2d_ax_neg1") (vt-to-list (vt-sum m :axis -1)) 1e-6)
    (T! "sum_2d_kd0"     (E J "sum_2d_ax0_kd")  (vt-to-list (vt-sum m :axis 0 :keepdims t)) 1e-6)
    (T! "mean_2d_all"    (E J "mean_2d_all")    (vt-item (vt-mean m)) 1e-6)
    (T! "mean_2d_ax0"    (E J "mean_2d_ax0")    (vt-to-list (vt-mean m :axis 0)) 1e-6)
    (T! "mean_2d_ax1"    (E J "mean_2d_ax1")    (vt-to-list (vt-mean m :axis 1)) 1e-6)
    (T! "max_2d_all"     (E J "amax_2d_all")    (vt-item (vt-amax m)) 1e-6)
    (T! "max_2d_ax0"     (E J "amax_2d_ax0")    (vt-to-list (vt-amax m :axis 0)) 1e-6)
    (T! "min_2d_ax1"     (E J "amin_2d_ax1")    (vt-to-list (vt-amin m :axis 1)) 1e-6)
    (T! "argmax_2d_ax0"  (E J "argmax_2d_ax0")  (vt-to-list (vt-argmax m :axis 0)) 0)
    (T! "argmax_2d_ax1"  (E J "argmax_2d_ax1")  (vt-to-list (vt-argmax m :axis 1)) 0)
    (T! "argmin_2d_ax0"  (E J "argmin_2d_ax0")  (vt-to-list (vt-argmin m :axis 0)) 0)
    (T! "argmin_2d_ax-1" (E J "argmin_2d_ax_neg1") (vt-to-list (vt-argmin m :axis -1)) 0)
    (T! "std_2d_all"     (E J "std_2d_all")     (vt-item (vt-std m)) 1e-5)
    (T! "var_2d_all"     (E J "var_2d_all")     (vt-item (vt-var m)) 1e-5)
    (T! "cumsum_2d_ax0"  (E J "cumsum_2d_ax0")  (vt-to-list (vt-cumsum mi :axis 0)) 0)
    (T! "cumsum_2d_ax1"  (E J "cumsum_2d_ax1")  (vt-to-list (vt-cumsum mi :axis 1)) 0)
    (T! "cumprod_2d_ax0" (E J "cumprod_2d_ax0") (vt-to-list (vt-cumprod mi :axis 0)) 0)
    (T! "median_2d_ax0"  (E J "median_2d_ax0")  (vt-to-list (vt-median m :axis 0)) 1e-6)
    (T! "median_2d_ax1"  (E J "median_2d_ax1")  (vt-to-list (vt-median m :axis 1)) 1e-6)
    (T! "ptp_2d_ax0"     (E J "ptp_2d_ax0")     (vt-to-list (vt-ptp m :axis 0)) 1e-6)
    (T! "ptp_2d_ax1"     (E J "ptp_2d_ax1")     (vt-to-list (vt-ptp m :axis 1)) 1e-6)
    (T! "sort_2d_ax0"    (E J "sort_2d_ax0")    (vt-to-list (vt-sort m :axis 0)) 1e-6)
    (T! "sort_2d_ax1"    (E J "sort_2d_ax1")    (vt-to-list (vt-sort m :axis 1)) 1e-6)
    (T! "argsort_2d_ax0" (E J "argsort_2d_ax0") (vt-to-list (vt-argsort m :axis 0)) 0)
    (T! "argsort_2d_ax1" (E J "argsort_2d_ax1") (vt-to-list (vt-argsort m :axis 1)) 0)
    (T! "diff_2d_ax0"    (E J "diff_2d_ax0")    (vt-to-list (vt-diff m :axis 0)) 1e-6)
    (T! "diff_2d_ax1"    (E J "diff_2d_ax1")    (vt-to-list (vt-diff m :axis 1)) 1e-6)

    ;; --- Reduction 3D ---
    (T! "sum_3d_all"  (E J "sum_3d_all")  (vt-item (vt-sum t3)) 1e-6)
    (T! "sum_3d_ax0"  (E J "sum_3d_ax0")  (vt-to-list (vt-sum t3 :axis 0)) 1e-6)
    (T! "sum_3d_ax1"  (E J "sum_3d_ax1")  (vt-to-list (vt-sum t3 :axis 1)) 1e-6)
    (T! "sum_3d_ax2"  (E J "sum_3d_ax2")  (vt-to-list (vt-sum t3 :axis 2)) 1e-6)
    (T! "sum_3d_ax-1" (E J "sum_3d_ax_neg1") (vt-to-list (vt-sum t3 :axis -1)) 1e-6)
    (T! "sum_3d_ax-2" (E J "sum_3d_ax_neg2") (vt-to-list (vt-sum t3 :axis -2)) 1e-6)
    (T! "sum_3d_ax01" (E J "sum_3d_ax01") (vt-to-list (vt-sum t3 :axis '(0 1))) 1e-6)
    (T! "sum_3d_ax12" (E J "sum_3d_ax12") (vt-to-list (vt-sum t3 :axis '(1 2))) 1e-6)
    (T! "sum_3d_kd0"  (E J "sum_3d_ax0_kd") (vt-to-list (vt-sum t3 :axis 0 :keepdims t)) 1e-6)
    (T! "sum_3d_kd2"  (E J "sum_3d_ax2_kd") (vt-to-list (vt-sum t3 :axis 2 :keepdims t)) 1e-6)
    (T! "mean_3d_all" (E J "mean_3d_all") (vt-item (vt-mean t3)) 1e-6)
    (T! "mean_3d_ax0" (E J "mean_3d_ax0") (vt-to-list (vt-mean t3 :axis 0)) 1e-6)
    (T! "mean_3d_ax1" (E J "mean_3d_ax1") (vt-to-list (vt-mean t3 :axis 1)) 1e-6)
    (T! "mean_3d_ax2" (E J "mean_3d_ax2") (vt-to-list (vt-mean t3 :axis 2)) 1e-6)
    (T! "mean_3d_ax-1" (E J "mean_3d_ax_neg1") (vt-to-list (vt-mean t3 :axis -1)) 1e-6)
    (T! "mean_3d_ax01" (E J "mean_3d_ax01") (vt-to-list (vt-mean t3 :axis '(0 1))) 1e-6)
    (T! "mean_3d_kd0" (E J "mean_3d_ax0_kd") (vt-to-list (vt-mean t3 :axis 0 :keepdims t)) 1e-6)
    (T! "max_3d_all"  (E J "amax_3d_all")  (vt-item (vt-amax t3)) 1e-6)
    (T! "max_3d_ax0"  (E J "amax_3d_ax0")  (vt-to-list (vt-amax t3 :axis 0)) 1e-6)
    (T! "max_3d_ax2"  (E J "amax_3d_ax2")  (vt-to-list (vt-amax t3 :axis 2)) 1e-6)
    (T! "min_3d_ax1"  (E J "amin_3d_ax1")  (vt-to-list (vt-amin t3 :axis 1)) 1e-6)
    (T! "argmax_3d_ax0" (E J "argmax_3d_ax0") (vt-to-list (vt-argmax t3 :axis 0)) 0)
    (T! "argmax_3d_ax2" (E J "argmax_3d_ax2") (vt-to-list (vt-argmax t3 :axis 2)) 0)
    (T! "argmin_3d_ax0" (E J "argmin_3d_ax0") (vt-to-list (vt-argmin t3 :axis 0)) 0)
    (T! "argmin_3d_ax1" (E J "argmin_3d_ax1") (vt-to-list (vt-argmin t3 :axis 1)) 0)
    (T! "std_3d_ax0"  (E J "std_3d_ax0")  (vt-to-list (vt-std t3 :axis 0)) 1e-5)
    (T! "var_3d_ax2"  (E J "var_3d_ax2")  (vt-to-list (vt-var t3 :axis 2)) 1e-5)
    (T! "cumsum_3d_ax0" (E J "cumsum_3d_ax0") (vt-to-list (vt-cumsum ti :axis 0)) 0)
    (T! "cumsum_3d_ax1" (E J "cumsum_3d_ax1") (vt-to-list (vt-cumsum ti :axis 1)) 0)
    (T! "cumsum_3d_ax2" (E J "cumsum_3d_ax2") (vt-to-list (vt-cumsum ti :axis 2)) 0)
    (T! "cumprod_3d_ax2" (E J "cumprod_3d_ax2") (vt-to-list (vt-cumprod ti :axis 2)) 0)
    (T! "median_3d_ax0" (E J "median_3d_ax0") (vt-to-list (vt-median t3 :axis 0)) 1e-6)
    (T! "median_3d_ax2" (E J "median_3d_ax2") (vt-to-list (vt-median t3 :axis 2)) 1e-6)
    (T! "ptp_3d_ax0"    (E J "ptp_3d_ax0")    (vt-to-list (vt-ptp t3 :axis 0)) 1e-6)
    (T! "ptp_3d_ax2"    (E J "ptp_3d_ax2")    (vt-to-list (vt-ptp t3 :axis 2)) 1e-6)
    (T! "sort_3d_ax0"   (E J "sort_3d_ax0")   (vt-to-list (vt-sort t3 :axis 0)) 1e-6)
    (T! "sort_3d_ax2"   (E J "sort_3d_ax2")   (vt-to-list (vt-sort t3 :axis 2)) 1e-6)
    (T! "argsort_3d_ax0" (E J "argsort_3d_ax0") (vt-to-list (vt-argsort t3 :axis 0)) 0)
    (T! "argsort_3d_ax2" (E J "argsort_3d_ax2") (vt-to-list (vt-argsort t3 :axis 2)) 0)
    (T! "diff_3d_ax0"   (E J "diff_3d_ax0")   (vt-to-list (vt-diff t3 :axis 0)) 1e-6)
    (T! "diff_3d_ax2"   (E J "diff_3d_ax2")   (vt-to-list (vt-diff t3 :axis 2)) 1e-6)
    (T! "maximum_3d"    (E J "maximum_3d")    (vt-to-list (vt-maximum t3 (vt-flip t3 :axis 0))) 1e-6)
    (T! "minimum_3d"    (E J "minimum_3d")    (vt-to-list (vt-minimum t3 (vt-flip t3 :axis 0))) 1e-6)

    ;; --- Percentile 2D+3D ---
    (dolist (p '(25 50 75 90))
      (T! (format nil "pct~a_2d_ax0" p) (E J (format nil "pct~a_2d_ax0" p))
          (vt-to-list (vt-percentile m p :axis 0)) 1e-4)
      (T! (format nil "pct~a_2d_ax1" p) (E J (format nil "pct~a_2d_ax1" p))
          (vt-to-list (vt-percentile m p :axis 1)) 1e-4)
      (T! (format nil "pct~a_3d_ax0" p) (E J (format nil "pct~a_3d_ax0" p))
          (vt-to-list (vt-percentile t3 p :axis 0)) 1e-4)
      (T! (format nil "pct~a_3d_ax2" p) (E J (format nil "pct~a_3d_ax2" p))
          (vt-to-list (vt-percentile t3 p :axis 2)) 1e-4))

    ;; --- Shape 3D ---
    (T! "trans_3d_021" (E J "trans_3d_021") (vt-to-list (vt-transpose t3 '(0 2 1))) 1e-6)
    (T! "trans_3d_102" (E J "trans_3d_102") (vt-to-list (vt-transpose t3 '(1 0 2))) 1e-6)
    (T! "trans_3d_210" (E J "trans_3d_210") (vt-to-list (vt-transpose t3 '(2 1 0))) 1e-6)
    (T! "squeeze_134"  (E J "squeeze_3d_134") (vt-to-list (vt-squeeze (vt-reshape (vt-arange 12 :dtype :int64) '(1 3 4)))) 0)
    (T! "squeeze_314"  (E J "squeeze_3d_314") (vt-to-list (vt-squeeze (vt-reshape (vt-arange 12 :dtype :int64) '(3 1 4)))) 0)
    (T! "squeeze_341"  (E J "squeeze_3d_341") (vt-to-list (vt-squeeze (vt-reshape (vt-arange 12 :dtype :int64) '(3 4 1)))) 0)
    (T! "expand_3d_ax0" (E J "expand_3d_ax0_shape") (vt-shape (vt-expand-dims t3 0)))
    (T! "expand_3d_ax2" (E J "expand_3d_ax2_shape") (vt-shape (vt-expand-dims t3 2)))
    (T! "expand_3d_ax3" (E J "expand_3d_ax3_shape") (vt-shape (vt-expand-dims t3 3)))
    (T! "concat_3d_ax0" (E J "concat_3d_ax0") (vt-to-list (vt-concatenate 0 t3 (vt-+ t3 100.0))) 1e-6)
    (T! "flip_3d_ax0" (E J "flip_3d_ax0") (vt-to-list (vt-flip t3 :axis 0)) 1e-6)
    (T! "flip_3d_ax1" (E J "flip_3d_ax1") (vt-to-list (vt-flip t3 :axis 1)) 1e-6)
    (T! "flip_3d_ax2" (E J "flip_3d_ax2") (vt-to-list (vt-flip t3 :axis 2)) 1e-6)
    (T! "roll_3d_ax0" (E J "roll_3d_ax0") (vt-to-list (vt-roll ti 1 :axis 0)) 0)
    (T! "roll_3d_ax2" (E J "roll_3d_ax2") (vt-to-list (vt-roll ti 1 :axis 2)) 0)
    (T! "triu_3d" (E J "triu_3d") (vt-to-list (vt-triu t3)) 1e-6)
    (T! "tril_3d" (E J "tril_3d") (vt-to-list (vt-tril t3)) 1e-6)
    (T! "swapaxes_3d_01" (E J "swapaxes_3d_01") (vt-to-list (vt-swapaxes t3 0 1)) 1e-6)
    (T! "swapaxes_3d_02" (E J "swapaxes_3d_02") (vt-to-list (vt-swapaxes t3 0 2)) 1e-6)
    (T! "swapaxes_3d_12" (E J "swapaxes_3d_12") (vt-to-list (vt-swapaxes t3 1 2)) 1e-6)
    (T! "narrow_3d_ax0" (E J "narrow_3d_ax0") (vt-to-list (vt-narrow t3 0 0 1)) 1e-6)
    (T! "narrow_3d_ax1" (E J "narrow_3d_ax1") (vt-to-list (vt-narrow t3 1 1 3)) 1e-6)
    (T! "narrow_3d_ax2" (E J "narrow_3d_ax2") (vt-to-list (vt-narrow t3 2 1 4)) 1e-6)
    (T! "tile_3d_121" (E J "tile_3d_121") (vt-to-list (vt-tile t222i '(1 2 1))) 0)
    (T! "tile_3d_212" (E J "tile_3d_212") (vt-to-list (vt-tile t222i '(2 1 2))) 0)
    (T! "repeat_3d_ax0" (E J "repeat_3d_ax0") (vt-to-list (vt-repeat t222i 2 :axis 0)) 0)
    (T! "repeat_3d_ax2" (E J "repeat_3d_ax2") (vt-to-list (vt-repeat t222i 2 :axis 2)) 0)
    (T! "pad_3d_const" (E J "pad_3d_const") (vt-to-list (vt-pad t222 1 :mode :constant :constant-values 0)) 1e-6)
    (T! "pad_3d_edge"  (E J "pad_3d_edge")  (vt-to-list (vt-pad t222 1 :mode :edge)) 1e-6)
    (T! "broadcast_1x3x4" (E J "broadcast_1x3x4_to_2x3x4")
        (vt-to-list (vt-broadcast-to (vt-reshape (vt-arange 12 :dtype :int64) '(1 3 4)) '(2 3 4))) 0)

    ;; --- Arithmetic 3D ---
    (T! "add_3d_scalar" (E J "add_3d_scalar") (vt-to-list (vt-+ t3 10.0)) 1e-6)
    (T! "mul_3d_scalar" (E J "mul_3d_scalar") (vt-to-list (vt-* t3 2.0)) 1e-6)
    (T! "add_3d_3d"     (E J "add_3d_3d")     (vt-to-list (vt-+ t3 t3)) 1e-6)
    (T! "mul_3d_3d"     (E J "mul_3d_3d")     (vt-to-list (vt-* t3 t3)) 1e-6)
    (T! "abs_3d"        (E J "abs_3d")        (vt-to-list (vt-abs (vt-- t3 12.0))) 1e-6)
    (T! "square_3d"     (E J "square_3d")     (vt-to-list (vt-square t3)) 1e-6)
    (T! "sqrt_3d"       (E J "sqrt_3d")       (vt-to-list (vt-sqrt (vt-abs t3))) 1e-6)
    (T! "exp_3d"        (E J "exp_3d")        (vt-to-list (vt-exp (vt-scale t3 0.1))) 1e-5)
    (T! "log_3d"        (E J "log_3d")        (vt-to-list (vt-log (vt-+ (vt-abs t3) 1.0))) 1e-5)
    (T! "clip_3d"       (E J "clip_3d")       (vt-to-list (vt-clip t3 5.0 15.0)) 1e-6)

    ;; --- Trig 3D ---
    (T! "sin_3d"  (E J "sin_3d")  (vt-to-list (vt-sin (vt-scale t3 0.1))) 1e-5)
    (T! "cos_3d"  (E J "cos_3d")  (vt-to-list (vt-cos (vt-scale t3 0.1))) 1e-5)
    (T! "tanh_3d" (E J "tanh_3d") (vt-to-list (vt-tanh (vt-scale (vt-- t3 12.0) 0.1))) 1e-5)

    ;; --- Comparison 3D ---
    (T! "lt_3d"       (E J "lt_3d")       (vt-to-list (vt-< t3 12.0)) 0)
    (T! "gt_3d"       (E J "gt_3d")       (vt-to-list (vt-> t3 12.0)) 0)
    (T! "isfinite_3d" (E J "isfinite_3d") (vt-to-list (vt-isfinite t3)) 0)
    (T! "all_3d_ax0"  (E J "all_3d_ax0")  (vt-to-list (vt-all (vt-> t3 0.0) :axis 0)) 0)
    (T! "any_3d_ax2"  (E J "any_3d_ax2")  (vt-to-list (vt-any (vt-> t3 10.0) :axis 2)) 0)

    ;; --- einsum advanced ---
    (T! "einsum_trace_2d" (E J "einsum_trace_2d")
        (vt-item (vt-einsum "ii->" (vt-reshape (vt-arange 4 :dtype :int64) '(2 2)))) 0)
    (T! "einsum_3d_reduce" (E J "einsum_3d_reduce")
        (vt-to-list (vt-einsum "ijk->i" t3)) 1e-5)
    (T! "einsum_3d_ax2" (E J "einsum_3d_reduce_ax2")
        (vt-to-list (vt-einsum "ijk->ij" t3)) 1e-5)

    ;; --- Where 3D ---
    (T! "where_3d" (E J "where_3d")
        (vt-to-list (vt-where (vt-> t3 12.0) t3 (vt-zeros '(2 3 4)))) 1e-6)

    ;; --- Activation 3D ---
    (T! "sigmoid_3d" (E J "sigmoid_3d") (vt-to-list (vt-sigmoid t3)) 1e-5)
    (T! "relu_3d"    (E J "relu_3d")    (vt-to-list (vt-relu (vt-- t3 12.0))) 1e-6)

    ;; --- Diagonal 3D ---
    (T! "diag_3d" (E J "diag_3d") (vt-to-list (vt-diagonal t3)) 1e-6)
    (T! "convolve_valid" (E J "convolve_valid")
        (vt-to-list (vt-convolve (vt-from-sequence '(1.0 2.0 3.0 4.0 5.0))
                                 (vt-from-sequence '(1.0 0.0 -1.0)) :mode :valid)) 1e-6)
    (T! "convolve_full" (E J "convolve_full")
        (vt-to-list (vt-convolve (vt-from-sequence '(1.0 2.0 3.0 4.0 5.0))
                                 (vt-from-sequence '(1.0 0.0 -1.0)) :mode :full)) 1e-6)

    ;; --- Eigenvalue decomposition ---
    (let ((a (vt-from-sequence '((4.0 2.0 1.0) (2.0 5.0 3.0) (1.0 3.0 6.0)) :dtype :float64)))
      (multiple-value-bind (vals vecs) (vt-eig a :max-iter 500)
        (declare (ignore vecs))
        (T! "eig_3x3_vals" (E J "eig_3x3_vals") (vt-to-list (vt-sort vals)) 1e-4)))

    ;; --- Stack 3D (shape checks) ---
    (T! "stack_3d_ax0" (E J "stack_3d_ax0") (vt-shape (vt-stack 0 t3 (vt-+ t3 100.0))))
    (T! "stack_3d_ax3" (E J "stack_3d_ax3") (vt-shape (vt-stack 3 t3 (vt-+ t3 100.0))))

    ;; --- Split 3D ---
    (let ((parts (vt-split t3 2 :axis 0)))
      (T! "split_3d_ax0" (E J "split_3d_ax0") (mapcar #'vt-to-list parts) 1e-6))
    (let ((parts (vt-split t3 2 :axis 2)))
      (T! "split_3d_ax2" (E J "split_3d_ax2") (mapcar #'vt-to-list parts) 1e-6))

    ;; --- Floor/Round 3D ---
    (T! "floor_3d" (E J "floor_3d") (vt-to-list (vt-floor (vt-scale t3 0.7))) 1e-4)
    (T! "round_3d" (E J "round_3d") (vt-to-list (vt-round (vt-scale t3 0.7))) 1e-4)

    ;; --- Diagonal 3D (PyTorch convention duplicate) ---
    (T! "diag_3d_pytorch" (E J "diag_3d_pytorch") (vt-to-list (vt-diagonal t3)) 1e-6)
    )
  (summary))

;; Run
(let ((ret (run-param-tests)))
  (sb-ext:exit :code (if ret 0 1)))
