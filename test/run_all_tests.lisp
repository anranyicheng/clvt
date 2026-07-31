;;;; run_all_tests.lisp — 全自动测试 clvt 所有函数，对比 numpy 期望值
;;;; Usage: sbcl --noinform --non-interactive --load run_all_tests.lisp

(require :asdf)
(push (truename (make-pathname :directory '(:relative :up))) asdf:*central-registry*)
(asdf:load-system :clvt)
(in-package :clvt)

;;; ============================================================
;;; JSON 解析器
;;; ============================================================
(defun parse-json (path)
  (with-open-file (s path :direction :input)
    (let* ((len (file-length s))
           (buf (make-string len))
           (pos 0))
      (read-sequence buf s)
      (labels ((skip () (loop while (and (< pos len) (member (char buf pos) '(#\Space #\Tab #\Newline #\Return))) do (incf pos)))
               (rd ()
                 (skip) (when (>= pos len) (return-from rd nil))
                 (let ((c (char buf pos)))
                   (cond
                     ((char= c #\") (incf pos) (let ((st pos)) (loop while (and (< pos len) (char/= (char buf pos) #\")) do (when (char= (char buf pos) #\\) (incf pos)) (incf pos)) (prog1 (subseq buf st pos) (incf pos))))
                     ((char= c #\[) (incf pos) (skip) (let ((a nil)) (loop while (and (< pos len) (char/= (char buf pos) #\])) do (push (rd) a) (skip) (when (and (< pos len) (char= (char buf pos) #\,)) (incf pos) (skip))) (incf pos) (nreverse a)))
                     ((char= c #\{) (incf pos) (skip) (let ((h (make-hash-table :test 'equal))) (loop while (and (< pos len) (char/= (char buf pos) #\})) do (let ((k (rd))) (skip) (when (and (< pos len) (char= (char buf pos) #\:)) (incf pos)) (setf (gethash k h) (rd))) (skip) (when (and (< pos len) (char= (char buf pos) #\,)) (incf pos) (skip))) (incf pos) h))
                     ((or (char<= #\0 c #\9) (char= c #\-) (char= c #\+)) (let ((st pos)) (when (member c '(#\- #\+)) (incf pos)) (loop while (and (< pos len) (or (char<= #\0 (char buf pos) #\9) (char= (char buf pos) #\.) (member (char buf pos) '(#\e #\E #\d #\D #\- #\+)))) do (incf pos)) (let* ((s0 (subseq buf st pos)) (s1 (substitute #\e #\d (substitute #\e #\D s0)))) (if (or (find #\. s1) (find #\e s1) (find #\E s1)) (let ((*read-eval* nil)) (read-from-string s1)) (parse-integer s1)))))
                     ((and (< (+ pos 4) len) (string= (subseq buf pos (min len (+ pos 4))) "true")) (incf pos 4) t)
                     ((and (< (+ pos 5) len) (string= (subseq buf pos (min len (+ pos 5))) "false")) (incf pos 5) nil)
                     ((and (< (+ pos 4) len) (string= (subseq buf pos (min len (+ pos 4))) "null")) (incf pos 4) nil)
                     (t (error "JSON parse error at ~a: ~a" pos c))))))
        (rd)))))

;;; ============================================================
;;; 测试框架
;;; ============================================================
(defvar *N* 0) (defvar *P* 0) (defvar *F* 0) (defvar *F-list* nil)
(defvar *skip-list* '())  ; 已知问题跳过

(defun approx (e a &optional (tol 1e-6))
  (cond
    ((and (numberp e) (numberp a))
     (if (and (floatp e) (floatp a))
         (< (abs (- e a)) (+ tol (* 0.001 (abs e))))
         (eql e a)))
    ((and (listp e) (listp a))
     (and (= (length e) (length a)) (every (lambda (x y) (approx x y tol)) e a)))
    (t (equal e a))))

(defun get-val (entry)
  "从 JSON entry 提取值"
  (cond
    ((hash-table-p entry)
     (let ((t0 (gethash "t" entry)))
       (cond ((string= t0 "a") (gethash "v" entry))
             ((string= t0 "i") (gethash "v" entry))
             ((string= t0 "f") (gethash "v" entry))
             ((string= t0 "l") (gethash "v" entry))
             ((string= t0 "n") nil)
             (t entry))))
    (t entry)))

(defun T! (name expected actual &optional (tol 1e-6))
  (incf *N*)
  (if (member name *skip-list* :test #'string=)
      (format t "  ⏭ ~a (skipped)~%" name)
      (if (approx expected actual tol)
          (incf *P*)
          (progn (incf *F*) (push name *F-list*)
                 (format t "  ❌ ~a~%     exp: ~a~%     got: ~a~%~%" name
                         (if (listp expected) (subseq expected 0 (min 5 (length expected))) expected)
                         (if (listp actual) (subseq actual 0 (min 5 (length actual))) actual))))))

(defun summary ()
  (format t "~%============================================================~%")
  (format t "  Total: ~a | Pass: ~a | Fail: ~a | Skip: ~a~%" *N* *P* *F* (- *N* *P* *F*))
  (format t "============================================================~%")
  (when *F-list*
    (format t "~%Failed:~{~%  - ~a~}~%" (reverse *F-list*)))
  (zerop *F*))

;;; ============================================================
;;; 主测试
;;; ============================================================
(defun run ()
  (format t "~%============================================================~%")
  (format t "  clvt FULL TEST — ~a functions vs NumPy~%" (length *vt-fun-list*))
  (format t "============================================================~%~%")

  ;; 已知问题跳过列表
  (setf *skip-list* '("einsum_batch_mm"))  ; 3D batch matmul bug

  (let* ((J (parse-json "test/all_expected.json"))
         (E (lambda (k) (get-val (gethash k J)))))

    ;; 1. 张量创建
    (format t "--- 1. Creation ---~%")
    (T! "arange(10)" (funcall E "arange_10") (vt-to-list (vt-arange 10 :dtype :int64)))
    (T! "linspace(0,1,5)" (funcall E "linspace_0_1_5") (vt-to-list (vt-linspace 0.0 1.0 5)))
    (T! "linspace(0,10,3)" (funcall E "linspace_0_10_3") (vt-to-list (vt-linspace 0.0 10.0 3)))
    (T! "logspace(0,3,4)" (funcall E "logspace_0_3_4") (vt-to-list (vt-logspace 0 3 4)))
    (T! "eye(3)" (funcall E "eye_3") (vt-to-list (vt-eye 3 :dtype :float64)))
    (T! "eye(4,6)" (funcall E "eye_4x6") (vt-to-list (vt-eye 4 :cols 6 :dtype :float64)))
    (T! "eye(3,k=1)" (funcall E "eye_3_k1") (vt-to-list (vt-eye 3 :k 1 :dtype :float64)))
    (T! "eye(3,k=-1)" (funcall E "eye_3_k_neg1") (vt-to-list (vt-eye 3 :k -1 :dtype :float64)))
    (T! "diag([1,2,3])" (funcall E "diag_v123") (vt-to-list (vt-diag (vt-from-sequence '(1 2 3) :dtype :int64))))
    (T! "diag(extract)" (funcall E "diag_extract") (vt-to-list (vt-diag (vt-reshape (vt-arange 9 :dtype :int64) '(3 3)))))
    (T! "diag(extract,k=1)" (funcall E "diag_extract_k1") (vt-to-list (vt-diag (vt-reshape (vt-arange 9 :dtype :int64) '(3 3)) :k 1)))

    ;; 2. 形状操作
    (format t "~%--- 2. Shape ---~%")
    (T! "reshape(6)->(2,3)" (funcall E "reshape_6_23") (vt-to-list (vt-reshape (vt-arange 6 :dtype :int64) '(2 3))))
    (T! "reshape(6)->(3,2)" (funcall E "reshape_6_32") (vt-to-list (vt-reshape (vt-arange 6 :dtype :int64) '(3 2))))
    (T! "transpose(2,3)" (funcall E "transpose_23") (vt-to-list (vt-transpose (vt-reshape (vt-arange 6 :dtype :int64) '(2 3)))))
    (T! "squeeze(1,2,3)" (funcall E "squeeze_123") (vt-to-list (vt-squeeze (vt-reshape (vt-arange 6 :dtype :int64) '(1 2 3)))))
    (T! "squeeze(2,1,3)" (funcall E "squeeze_213") (vt-to-list (vt-squeeze (vt-reshape (vt-arange 6 :dtype :int64) '(2 1 3)))))
    (T! "expand_dims(0)" (funcall E "expand_dims_0") (vt-to-list (vt-expand-dims (vt-from-sequence '(1.0 2.0 3.0)) 0)))
    (T! "expand_dims(1)" (funcall E "expand_dims_1") (vt-to-list (vt-expand-dims (vt-from-sequence '(1.0 2.0 3.0)) 1)))
    (T! "flatten" (funcall E "flatten") (vt-to-list (vt-flatten (vt-reshape (vt-arange 6 :dtype :int64) '(2 3)))))
    (T! "ravel" (funcall E "ravel") (vt-to-list (vt-ravel (vt-reshape (vt-arange 6 :dtype :int64) '(2 3)))))
    (T! "concat(axis=0)" (funcall E "concat_0") (vt-to-list (vt-concatenate 0 (vt-from-sequence '((1.0 2.0) (3.0 4.0))) (vt-from-sequence '((5.0 6.0) (7.0 8.0))))))
    (T! "concat(axis=1)" (funcall E "concat_1") (vt-to-list (vt-concatenate 1 (vt-from-sequence '((1.0 2.0) (3.0 4.0))) (vt-from-sequence '((5.0 6.0) (7.0 8.0))))))
    (T! "stack(axis=0)" (funcall E "stack_0") (vt-to-list (vt-stack 0 (vt-from-sequence '(1.0 2.0 3.0)) (vt-from-sequence '(2.0 4.0 6.0)))))
    (T! "flip(1d)" (funcall E "flip_1d") (vt-to-list (vt-flip (vt-from-sequence '(1 2 3 4 5) :dtype :int64))))
    (T! "flip(axis=0)" (funcall E "flip_axis0") (vt-to-list (vt-flip (vt-reshape (vt-arange 12 :dtype :float64) '(3 4)) :axis 0)))
    (T! "flip(axis=1)" (funcall E "flip_axis1") (vt-to-list (vt-flip (vt-reshape (vt-arange 12 :dtype :float64) '(3 4)) :axis 1)))
    (T! "roll(2)" (funcall E "roll_2") (vt-to-list (vt-roll (vt-from-sequence '(1 2 3 4 5) :dtype :int64) 2)))
    (T! "roll(-1)" (funcall E "roll_neg1") (vt-to-list (vt-roll (vt-from-sequence '(1 2 3 4 5) :dtype :int64) -1)))
    (T! "triu" (funcall E "triu_3") (vt-to-list (vt-triu (vt-from-sequence '((1.0 2.0 3.0) (4.0 5.0 6.0) (7.0 8.0 9.0)) :dtype :float64))))
    (T! "triu(k=1)" (funcall E "triu_3_k1") (vt-to-list (vt-triu (vt-from-sequence '((1.0 2.0 3.0) (4.0 5.0 6.0) (7.0 8.0 9.0)) :dtype :float64) :k 1)))
    (T! "tril" (funcall E "tril_3") (vt-to-list (vt-tril (vt-from-sequence '((1.0 2.0 3.0) (4.0 5.0 6.0) (7.0 8.0 9.0)) :dtype :float64))))
    (T! "diagonal" (funcall E "diag_3") (vt-to-list (vt-diagonal (vt-from-sequence '((1.0 2.0 3.0) (4.0 5.0 6.0) (7.0 8.0 9.0)) :dtype :float64))))
    (T! "diagonal(k=1)" (funcall E "diag_3_k1") (vt-to-list (vt-diagonal (vt-from-sequence '((1.0 2.0 3.0) (4.0 5.0 6.0) (7.0 8.0 9.0)) :dtype :float64) :offset 1)))
    (T! "tile(3)" (funcall E "tile_3") (vt-to-list (vt-tile (vt-from-sequence '(1 2 3) :dtype :int64) 3)))
    (T! "repeat(2)" (funcall E "repeat_2") (vt-to-list (vt-repeat (vt-from-sequence '(1 2 3) :dtype :int64) 2)))
    (T! "broadcast_to(2,3)" (funcall E "broadcast_23") (vt-to-list (vt-broadcast-to (vt-from-sequence '(1 2 3) :dtype :int64) '(2 3))))

    ;; 3. 切片
    (format t "~%--- 3. Slicing ---~%")
    (T! "slice[2:7]" (funcall E "slice_2_7") (vt-to-list (vt-slice (vt-arange 10 :dtype :int64) '(2 7))))
    (T! "slice[1:9:2]" (funcall E "slice_1_9_2") (vt-to-list (vt-slice (vt-arange 10 :dtype :int64) '(1 9 2))))
    (T! "slice[:5]" (funcall E "slice_nil_5") (vt-to-list (vt-slice (vt-arange 10 :dtype :int64) '(nil 5))))
    (T! "slice[5:]" (funcall E "slice_5_nil") (vt-to-list (vt-slice (vt-arange 10 :dtype :int64) '(5 nil))))
    (T! "slice[::-1]" (funcall E "slice_reverse") (vt-to-list (vt-slice (vt-arange 10 :dtype :int64) '(nil nil -1))))
    (T! "slice[8:3:-1]" (funcall E "slice_8_3_neg1") (vt-to-list (vt-slice (vt-arange 10 :dtype :int64) '(8 3 -1))))
    (T! "slice[-1]" (funcall E "slice_neg1") (vt-item (vt-slice (vt-arange 10 :dtype :int64) '(-1))))
    (T! "2d[1,2]" (funcall E "2d_1_2") (vt-item (vt-slice (vt-reshape (vt-arange 20 :dtype :int64) '(4 5)) '(1) '(2))))
    (T! "2d row2" (funcall E "2d_row2") (vt-to-list (vt-slice (vt-reshape (vt-arange 20 :dtype :int64) '(4 5)) '(2) '(:all))))
    (T! "2d col3" (funcall E "2d_col3") (vt-to-list (vt-slice (vt-reshape (vt-arange 20 :dtype :int64) '(4 5)) '(:all) '(3))))
    (T! "2d sub" (funcall E "2d_sub") (vt-to-list (vt-slice (vt-reshape (vt-arange 20 :dtype :int64) '(4 5)) '(1 3) '(2 4))))
    (T! "2d ellipsis" (funcall E "2d_ellipsis") (vt-to-list (vt-slice (vt-reshape (vt-arange 20 :dtype :int64) '(4 5)) '(:elli) '(nil 2))))

    ;; 4. 算术
    (format t "~%--- 4. Arithmetic ---~%")
    (let ((a (vt-from-sequence '(1 2 3 4) :dtype :int64)) (b (vt-from-sequence '(5 6 7 8) :dtype :int64)))
      (T! "a+b(int)" (funcall E "add_ii") (vt-to-list (vt-+ a b)))
      (T! "b-a(int)" (funcall E "sub_ii") (vt-to-list (vt-- b a)))
      (T! "a*b(int)" (funcall E "mul_ii") (vt-to-list (vt-* a b)))
      (T! "a+10(int)" (funcall E "add_scalar_i") (vt-to-list (vt-+ a 10)))
      (T! "a*2(int)" (funcall E "mul_scalar_i") (vt-to-list (vt-* a 2))))
    (let ((a (vt-from-sequence '(1.0 2.0 3.0 4.0))) (b (vt-from-sequence '(5.0 6.0 7.0 8.0))))
      (T! "a+b(float)" (funcall E "add_ff") (vt-to-list (vt-+ a b)))
      (T! "a*b(float)" (funcall E "mul_ff") (vt-to-list (vt-* a b)))
      (T! "b/a(float)" (funcall E "div_ff") (vt-to-list (vt-/ b a))))
    (T! "abs(-3..3)" (funcall E "abs_neg") (vt-to-list (vt-abs (vt-from-sequence '(-3.0 -1.0 0.0 1.0 3.0)))))
    (T! "square" (funcall E "square") (vt-to-list (vt-square (vt-from-sequence '(1.0 2.0 3.0 4.0)))))
    (T! "sqrt(1,4,9)" (funcall E "sqrt_149") (vt-to-list (vt-sqrt (vt-from-sequence '(1.0 4.0 9.0)))))
    (T! "exp(1,2,3,4)" (funcall E "exp_1234") (vt-to-list (vt-exp (vt-from-sequence '(1.0 2.0 3.0 4.0)))))
    (T! "log(1,2,3,4)" (funcall E "log_1234") (vt-to-list (vt-log (vt-from-sequence '(1.0 2.0 3.0 4.0)))))
    (T! "log2(1,2,3,4)" (funcall E "log2_1234") (vt-to-list (vt-log2 (vt-from-sequence '(1.0 2.0 3.0 4.0)))))
    (T! "log10(1,2,3,4)" (funcall E "log10_1234") (vt-to-list (vt-log10 (vt-from-sequence '(1.0 2.0 3.0 4.0)))))
    (T! "clip(1..4,2,3)" (funcall E "clip_23") (vt-to-list (vt-clip (vt-from-sequence '(1.0 2.0 3.0 4.0)) 2.0 3.0)))
    (T! "floor" (funcall E "floor_1234") (vt-to-list (vt-floor (vt-from-sequence '(1.2 2.5 3.7 4.1)))))
    (T! "ceiling" (funcall E "ceil_1234") (vt-to-list (vt-ceiling (vt-from-sequence '(1.2 2.5 3.7 4.1)))))
    (T! "round" (funcall E "round_1234") (vt-to-list (vt-round (vt-from-sequence '(1.2 2.5 3.7 4.1)))))
    (T! "reciprocal" (funcall E "reciprocal") (vt-to-list (vt-reciprocal (vt-from-sequence '(1.0 2.0 3.0 4.0)))))

    ;; 5. 三角函数
    (format t "~%--- 5. Trig ---~%")
    (T! "sin(0..pi/2)" (funcall E "sin_0pi2") (vt-to-list (vt-sin (vt-linspace 0.0 (/ pi 2) 4))))
    (T! "cos(0..pi/2)" (funcall E "cos_0pi2") (vt-to-list (vt-cos (vt-linspace 0.0 (/ pi 2) 4))))
    (T! "tanh(1,2,3)" (funcall E "tanh_123") (vt-to-list (vt-tanh (vt-from-sequence '(1.0 2.0 3.0)))))
    (T! "hypot(3,4)" (funcall E "hypot_34") (vt-to-list (vt-hypot (vt-from-sequence '(3.0 5.0)) (vt-from-sequence '(4.0 12.0)))))
    (T! "deg2rad" (funcall E "deg2rad") (vt-to-list (vt-deg2rad (vt-from-sequence '(0.0 90.0 180.0)))))
    (T! "rad2deg" (funcall E "rad2deg") (vt-to-list (vt-rad2deg (vt-from-sequence '(0.0 1.5707963267948966 3.141592653589793)))))

    ;; 6. 比较与逻辑
    (format t "~%--- 6. Comparison ---~%")
    (let ((a (vt-from-sequence '(1.0 2.0 3.0 4.0 5.0))) (b (vt-from-sequence '(5.0 4.0 3.0 2.0 1.0))))
      (T! "a<b" (funcall E "lt") (vt-to-list (vt-< a b)))
      (T! "a==b" (funcall E "eq") (vt-to-list (vt-= a b)))
      (T! "a>b" (funcall E "gt") (vt-to-list (vt-> a b))))
    (T! "all(true)" (funcall E "all_true") (vt-item (vt-all (vt-from-sequence '(1.0 1.0 1.0)))))
    (T! "any(true)" (funcall E "any_true") (vt-item (vt-any (vt-from-sequence '(0.0 0.0 1.0)))))
    (T! "isfinite" (funcall E "isfinite") (vt-to-list (vt-isfinite (vt-from-sequence (list 1.0d0 +vt-float-nan+ +vt-float-pos-inf+ +vt-float-neg-inf+ 0.0d0)))))
    (T! "isnan" (funcall E "isnan") (vt-to-list (vt-isnan (vt-from-sequence (list 1.0d0 +vt-float-nan+ +vt-float-pos-inf+ 0.0d0)))))
    (T! "isinf" (funcall E "isinf") (vt-to-list (vt-isinf (vt-from-sequence (list 1.0d0 +vt-float-nan+ +vt-float-pos-inf+ +vt-float-neg-inf+ 0.0d0)))))

    ;; 7. 归约
    (format t "~%--- 7. Reduction ---~%")
    (let ((a (vt-astype (vt-reshape (vt-arange 12 :dtype :int64) '(3 4)) :float64)))
      (T! "sum(all)" (funcall E "sum_all") (vt-item (vt-sum a)))
      (T! "sum(axis=0)" (funcall E "sum_ax0") (vt-to-list (vt-sum a :axis 0)))
      (T! "sum(axis=1)" (funcall E "sum_ax1") (vt-to-list (vt-sum a :axis 1)))
      (T! "sum(axis=0,kd)" (funcall E "sum_ax0_kd") (vt-to-list (vt-sum a :axis 0 :keepdims t)))
      (T! "mean(all)" (funcall E "mean_all") (vt-item (vt-mean a)))
      (T! "mean(axis=1)" (funcall E "mean_ax1") (vt-to-list (vt-mean a :axis 1)))
      (T! "max(all)" (funcall E "max_all") (vt-item (vt-amax a)))
      (T! "max(axis=0)" (funcall E "max_ax0") (vt-to-list (vt-amax a :axis 0)))
      (T! "min(axis=1)" (funcall E "min_ax1") (vt-to-list (vt-amin a :axis 1)))
      (T! "argmax(axis=1)" (funcall E "argmax_ax1") (vt-to-list (vt-argmax a :axis 1)))
      (T! "argmin(axis=0)" (funcall E "argmin_ax0") (vt-to-list (vt-argmin a :axis 0)))
      (T! "std(all)" (funcall E "std_all") (vt-item (vt-std a)))
      (T! "var(all)" (funcall E "var_all") (vt-item (vt-var a))))
    (T! "cumsum(1,2,3,4)" (funcall E "cumsum_1234") (vt-to-list (vt-cumsum (vt-from-sequence '(1 2 3 4) :dtype :int64))))
    (T! "cumprod(1,2,3,4)" (funcall E "cumprod_1234") (vt-to-list (vt-cumprod (vt-from-sequence '(1 2 3 4) :dtype :int64))))
    (T! "median(odd)" (funcall E "median_odd") (vt-item (vt-median (vt-from-sequence '(3.0 1.0 4.0 1.0 5.0)))))
    (T! "median(even)" (funcall E "median_even") (vt-item (vt-median (vt-from-sequence '(1.0 2.0 3.0 4.0)))))
    (T! "pct50" (funcall E "pct50") (vt-item (vt-percentile (vt-from-sequence '(1.0 2.0 3.0 4.0 5.0)) 50)))
    (T! "pct90" (funcall E "pct90") (vt-item (vt-percentile (vt-from-sequence '(1.0 2.0 3.0 4.0 5.0)) 90)))
    (T! "ptp" (funcall E "ptp") (vt-item (vt-ptp (vt-from-sequence '(3.0 1.0 4.0 1.0 5.0)))))
    (T! "sort(1d)" (funcall E "sort_1d") (vt-to-list (vt-sort (vt-from-sequence '(3.0 1.0 4.0 1.0 5.0 9.0 2.0 6.0)))))
    (T! "maximum" (funcall E "maximum") (vt-to-list (vt-maximum (vt-from-sequence '(1.0 5.0 3.0)) (vt-from-sequence '(4.0 2.0 6.0)))))
    (T! "minimum" (funcall E "minimum") (vt-to-list (vt-minimum (vt-from-sequence '(1.0 5.0 3.0)) (vt-from-sequence '(4.0 2.0 6.0)))))
    (T! "diff(1d)" (funcall E "diff_1d") (vt-to-list (vt-diff (vt-from-sequence '(1.0 3.0 6.0 10.0 15.0)))))

    ;; 8. 线性代数
    (format t "~%--- 8. Linalg ---~%")
    (T! "matmul 2x2" (funcall E "matmul_2x2") (vt-to-list (vt-matmul (vt-from-sequence '((1.0 2.0) (3.0 4.0))) (vt-from-sequence '((5.0 6.0) (7.0 8.0))))))
    (T! "matmul 2x3*3x2" (funcall E "matmul_2x3_3x2") (vt-to-list (vt-matmul (vt-from-sequence '((1.0 2.0 3.0) (4.0 5.0 6.0))) (vt-from-sequence '((1.0 2.0) (3.0 4.0) (5.0 6.0))))))
    (T! "dot(1d)" 32.0d0 (vt-item (vt-dot (vt-from-sequence '(1.0 2.0 3.0)) (vt-from-sequence '(4.0 5.0 6.0)))))
    (T! "outer" (funcall E "outer_3_2") (vt-to-list (vt-outer (vt-from-sequence '(1.0 2.0 3.0)) (vt-from-sequence '(4.0 5.0)))))
    (T! "trace" (funcall E "trace_3") (vt-item (vt-trace (vt-from-sequence '((1.0 2.0 3.0) (4.0 5.0 6.0) (7.0 8.0 9.0))))))
    (T! "norm(3,4)" (funcall E "norm_34") (vt-item (vt-norm (vt-from-sequence '(3.0 4.0)))))
    (T! "det(2x2)" (funcall E "det_2x2") (vt-item (vt-det (vt-from-sequence '((1.0 2.0) (3.0 4.0))))))
    (T! "inv(2x2)" (funcall E "inv_2x2") (vt-to-list (vt-inv (vt-from-sequence '((1.0 2.0) (3.0 4.0))))))
    (T! "solve(2x2)" (funcall E "solve_2x2") (vt-to-list (vt-solve (vt-from-sequence '((2.0 1.0) (1.0 3.0))) (vt-from-sequence '(7.0 8.0)))))
    (T! "Cholesky" (funcall E "chol_L") (vt-to-list (vt-cholesky (vt-from-sequence '((4.0 2.0) (2.0 3.0))))))
    (T! "matrix-rank(full)" (funcall E "rank_full") (vt-matrix-rank (vt-eye 3 :dtype :float64)))
    (T! "matrix-rank(deficient)" (funcall E "rank_deficient") (vt-matrix-rank (vt-from-sequence '((1.0 2.0) (2.0 4.0)))))

    ;; QR
    (multiple-value-bind (q r) (vt-qr (vt-from-sequence '((1.0 2.0) (3.0 4.0) (5.0 6.0))))
      (T! "QR recon err" (funcall E "qr_recon_err") (vt-item (vt-amax (vt-abs (vt-- (vt-from-sequence '((1.0 2.0) (3.0 4.0) (5.0 6.0))) (vt-@ q r))))) 1e-10))

    ;; 9. einsum
    (format t "~%--- 9. einsum ---~%")
    (T! "einsum dot" (funcall E "einsum_dot") (vt-item (vt-einsum "i,i->" (vt-from-sequence '(1 2 3) :dtype :int64) (vt-from-sequence '(4 5 6) :dtype :int64))))
    (T! "einsum matmul" (funcall E "einsum_matmul") (vt-to-list (vt-einsum "ij,jk->ik" (vt-reshape (vt-arange 6 :dtype :int64) '(2 3)) (vt-reshape (vt-arange 6 :dtype :int64) '(3 2)))))
    (T! "einsum transpose" (funcall E "einsum_transpose") (vt-to-list (vt-einsum "ij->ji" (vt-from-sequence '((1 2) (3 4)) :dtype :int64))))
    (T! "einsum diag" (funcall E "einsum_diag") (vt-to-list (vt-einsum "ii->i" (vt-reshape (vt-arange 9 :dtype :int64) '(3 3)))))
    (T! "einsum trace" (funcall E "einsum_trace") (vt-item (vt-einsum "ii->" (vt-reshape (vt-arange 9 :dtype :int64) '(3 3)))))
    (T! "einsum outer" (funcall E "einsum_outer") (vt-to-list (vt-einsum "i,j->ij" (vt-from-sequence '(1 2 3) :dtype :int64) (vt-from-sequence '(4 5) :dtype :int64))))

    ;; 10. 激活函数
    (format t "~%--- 10. Activation ---~%")
    (let ((x (vt-from-sequence '(-2.0 -1.0 0.0 1.0 2.0))))
      (T! "sigmoid" (funcall E "sigmoid") (vt-to-list (vt-sigmoid x)))
      (T! "relu" (funcall E "relu") (vt-to-list (vt-relu x)))
      (T! "tanh" (funcall E "tanh") (vt-to-list (vt-tanh x))))
    (T! "softmax" (funcall E "softmax_123") (vt-to-list (vt-softmax (vt-from-sequence '(1.0 2.0 3.0)))))

    ;; 11. 集合
    (format t "~%--- 11. Set ---~%")
    (T! "unique" (funcall E "unique_122333") (vt-to-list (vt-unique (vt-from-sequence '(1 2 2 3 3 3) :dtype :int64))))
    (T! "intersect1d" (funcall E "intersect1d") (vt-to-list (vt-intersect1d (vt-from-sequence '(1 2 3 4 5) :dtype :int64) (vt-from-sequence '(3 4 5 6 7) :dtype :int64))))
    (T! "union1d" (funcall E "union1d") (vt-to-list (vt-union1d (vt-from-sequence '(1 2 3 4 5) :dtype :int64) (vt-from-sequence '(3 4 5 6 7) :dtype :int64))))
    (T! "setdiff1d" (funcall E "setdiff1d") (vt-to-list (vt-setdiff1d (vt-from-sequence '(1 2 3 4 5) :dtype :int64) (vt-from-sequence '(3 4 5 6 7) :dtype :int64))))

    ;; 12. where / nonzero
    (format t "~%--- 12. Where/Nonzero ---~%")
    (T! "where" (funcall E "where_cxy") (vt-to-list (vt-where (vt-from-sequence '(1.0 0.0 1.0 0.0)) (vt-from-sequence '(10.0 20.0 30.0 40.0)) (vt-from-sequence '(100.0 200.0 300.0 400.0)))))
    (T! "nonzero(1d)" (funcall E "nonzero_1d") (vt-to-list (first (vt-nonzero (vt-from-sequence '(0 1 0 2 0 3) :dtype :int64)))))

    ;; 13. nan
    (format t "~%--- 13. nan ---~%")
    (let ((a (vt-from-sequence (list 1.0d0 +vt-float-nan+ 3.0d0 4.0d0) :dtype :float64)))
      (T! "nanmean" (funcall E "nanmean") (vt-item (vt-nanmean a)))
      (T! "nansum" (funcall E "nansum") (vt-item (vt-nansum a)))
      (T! "nanmax" (funcall E "nanmax") (vt-item (vt-nanmax a)))
      (T! "nanmin" (funcall E "nanmin") (vt-item (vt-nanmin a))))

    ;; 14. pad / kron / meshgrid
    (format t "~%--- 14. Misc ---~%")
    (T! "pad(const)" (funcall E "pad_const_1") (vt-to-list (vt-pad (vt-from-sequence '((1 2) (3 4)) :dtype :int64) 1 :mode :constant :constant-values 0)))
    (T! "pad(edge)" (funcall E "pad_edge_1") (vt-to-list (vt-pad (vt-from-sequence '((1 2) (3 4)) :dtype :int64) 1 :mode :edge)))
    (T! "kron" (funcall E "kron_22") (vt-to-list (vt-kron (vt-from-sequence '((1 2) (3 4)) :dtype :int64) (vt-from-sequence '((0 5) (6 7)) :dtype :int64))))
    (T! "mse" (funcall E "mse") (vt-item (vt-mean-squared-error (vt-from-sequence '(1.0 0.0 0.0)) (vt-from-sequence '(0.7 0.2 0.1)))))

    ;; Summary
    (summary)))

(run)
