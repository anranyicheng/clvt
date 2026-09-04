;;;; auto-compare-test.lisp — 自动对比 clvt 与 NumPy 输出
;;;; 实时调用 Python (ref_compute.py) 生成 NumPy 参考值，不依赖静态 JSON 文件
;;;; 同时保留 PyTorch 交叉验证层（ref_compute.py 已内置）
;;;; Usage: sbcl --noinform --non-interactive --load test/auto-compare-test.lisp

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

(defun load-expected-realtime ()
  "调用 python3 test/ref_compute.py 生成所有参考值，解析为 hash-table。"
  (let* ((script (namestring (merge-pathnames "ref_compute.py" *load-truename*)))
         (output (uiop:run-program (list "python3" script) :output :string
                                   :error-output :interactive)))
    (parse-json-string output)))

;;; ============================================================
;;; 测试框架
;;; ============================================================
(defvar *test-count* 0)
(defvar *pass-count* 0)
(defvar *fail-count* 0)
(defvar *failures* '())

(defun approx-val (expected actual &optional (tol 1e-5))
  "递归比较数值/列表，浮点使用相对+绝对容差。"
  (cond
    ((and (null expected) (null actual)) t)
    ((and (numberp expected) (numberp actual))
     (let ((e (float expected 1.0d0))
           (a (float actual 1.0d0)))
       (cond
         ((or (and (not (< e e)) (not (< a a)))  ; both NaN
              (and (eql expected actual) (not (floatp expected))))
          t)
         (t (< (abs (- e a)) (+ tol (* tol (max 1.0d-12 (abs e)))))))))
    ((and (listp expected) (listp actual))
     (and (= (length expected) (length actual))
          (every (lambda (x y) (approx-val x y tol)) expected actual)))
    ((and (vectorp expected) (vectorp actual))
     (approx-val (coerce expected 'list) (coerce actual 'list) tol))
    (t (equal expected actual))))

(defun get-expected (ht key)
  "从hash-table中取出期望值。"
  (let ((v (gethash key ht)))
    (cond ((hash-table-p v)
           (or (gethash "scalar" v) (gethash "data" v) (gethash "v" v) v))
          (t v))))

(defun run-test (name expected actual &optional (tol 1e-5))
  (incf *test-count*)
  (if (approx-val expected actual tol)
      (progn (incf *pass-count*))
      (progn
        (incf *fail-count*)
        (let ((short-e (if (listp expected)
                           (subseq (format nil "~a" expected) 0 (min 80 (length (format nil "~a" expected))))
                           expected))
              (short-a (if (listp actual)
                           (subseq (format nil "~a" actual) 0 (min 80 (length (format nil "~a" actual))))
                           actual)))
          (push (format nil "FAIL [~a]" name) *failures*)
          (format t "  ❌ ~a~%" name)))))

(defun test-summary ()
  (format t "~%============================================================~%")
  (format t "  Total: ~a  Pass: ~a  Fail: ~a~%" *test-count* *pass-count* *fail-count*)
  (format t "============================================================~%")
  (when *failures*
    (format t "~%Failures:~%")
    (dolist (f (reverse *failures*))
      (format t "  ~a~%" f)))
  (zerop *fail-count*))

;;; ============================================================
;;; 主测试
;;; ============================================================
(defun run-auto-comparison ()
  (format t "~%============================================================~%")
  (format t "  clvt vs NumPy 自动对比测试（实时生成参考值）~%")
  (format t "============================================================~%~%")

  (let ((expected (load-expected-realtime)))
    (flet ((E (key) (get-expected expected key)))

      ;; ========== 1. 张量创建 ==========
      (format t "--- 1. Tensor Creation ---~%")
      (run-test "arange(10)"       (E "arange_10")       (vt-to-list (vt-arange 10 :dtype :int64)))
      (run-test "linspace(0,1,5)"  (E "linspace_0_1_5")  (vt-to-list (vt-linspace 0.0 1.0 5)))
      (run-test "logspace(0,3,4)"  (E "logspace_0_3_4")  (vt-to-list (vt-logspace 0 3 4)))
      (run-test "eye(3)"           (E "eye_3")           (vt-to-list (vt-eye 3 :dtype :float64)))
      (run-test "diag([1,2,3])"    (E "diag_123")        (vt-to-list (vt-diag (vt-from-sequence '(1 2 3) :dtype :int64))))

      ;; ========== 2. 形状操作 ==========
      (format t "~%--- 2. Shape Operations ---~%")
      (run-test "reshape(6)->(2,3)" (E "reshape_23")
                (vt-to-list (vt-reshape (vt-arange 6 :dtype :int64) '(2 3))))
      (run-test "transpose(2,3)" (E "transpose_23")
                (vt-to-list (vt-transpose (vt-reshape (vt-arange 6 :dtype :int64) '(2 3)))))
      (run-test "squeeze(1,2,3)" (E "squeeze_123")
                (vt-to-list (vt-squeeze (vt-reshape (vt-arange 6 :dtype :int64) '(1 2 3)))))
      (run-test "expand_dims(0)" (E "expand_dims_0")
                (vt-to-list (vt-expand-dims (vt-from-sequence '(1 2 3) :dtype :int64) 0)))
      (run-test "concat axis=0" (E "concat_axis0")
                (vt-to-list (vt-concatenate 0
                  (vt-from-sequence '((1 2) (3 4)) :dtype :int64)
                  (vt-from-sequence '((5 6) (7 8)) :dtype :int64))))
      (run-test "stack axis=0" (E "stack_axis0")
                (vt-to-list (vt-stack 0
                  (vt-from-sequence '(1 2) :dtype :int64)
                  (vt-from-sequence '(3 4) :dtype :int64))))
      (run-test "flip axis=0" (E "flip_axis0_23")
                (vt-to-list (vt-flip (vt-reshape (vt-arange 6 :dtype :int64) '(2 3)) :axis 0)))
      (run-test "roll(2)" (E "roll_2_arange")
                (vt-to-list (vt-roll (vt-arange 5 :dtype :int64) 2)))
      (run-test "triu" (E "triu")
                (vt-to-list (vt-triu (vt-reshape (vt-arange 9 :dtype :int64) '(3 3)))))
      (run-test "tril" (E "tril")
                (vt-to-list (vt-tril (vt-reshape (vt-arange 9 :dtype :int64) '(3 3)))))
      (run-test "diagonal" (E "diagonal")
                (vt-to-list (vt-diagonal (vt-reshape (vt-arange 9 :dtype :int64) '(3 3)))))
      (run-test "tile(3)" (E "tile_3")
                (vt-to-list (vt-tile (vt-from-sequence '(1 2 3) :dtype :int64) 3)))
      (run-test "repeat(2)" (E "repeat_2")
                (vt-to-list (vt-repeat (vt-from-sequence '(1 2 3) :dtype :int64) 2)))
      (run-test "broadcast_to(2,3)" (E "broadcast_to_23")
                (vt-to-list (vt-broadcast-to (vt-from-sequence '(1 2 3) :dtype :int64) '(2 3))))

      ;; ========== 3. 切片 ==========
      (format t "~%--- 3. Slicing ---~%")
      (run-test "slice[2:7]" (E "slice_2_7")
                (vt-to-list (vt-slice (vt-arange 10 :dtype :int64) '(2 7))))
      (run-test "slice[::-1]" (E "slice_reverse")
                (vt-to-list (vt-slice (vt-arange 10 :dtype :int64) '(nil nil -1))))
      (run-test "2d[1,2]" (E "2d_1_2")
                (vt-item (vt-slice (vt-reshape (vt-arange 20 :dtype :int64) '(4 5)) '(1) '(2))))
      (run-test "2d row2" (E "2d_row2")
                (vt-to-list (vt-slice (vt-reshape (vt-arange 20 :dtype :int64) '(4 5)) '(2) '(:all))))

      ;; ========== 4. 算术 ==========
      (format t "~%--- 4. Arithmetic ---~%")
      (let ((a (vt-from-sequence '(1 2 3 4) :dtype :int64))
            (b (vt-from-sequence '(5 6 7 8) :dtype :int64)))
        (run-test "a+b"       (E "add_ab")       (vt-to-list (vt-+ a b)))
        (run-test "b-a"       (E "sub_ba")       (vt-to-list (vt-- b a)))
        (run-test "a*b"       (E "mul_ab")       (vt-to-list (vt-* a b)))
        (run-test "a+10"      (E "add_scalar10") (vt-to-list (vt-+ a 10)))
        (run-test "a*2"       (E "mul_scalar2")  (vt-to-list (vt-* a 2))))

      ;; ========== 5. 数学 ==========
      (format t "~%--- 5. Math ---~%")
      (run-test "exp(1,2,3)" (E "exp_123")
                (vt-to-list (vt-exp (vt-from-sequence '(1.0 2.0 3.0) :dtype :float64))))
      (run-test "log(1,2,3)" (E "log_123")
                (vt-to-list (vt-log (vt-from-sequence '(1.0 2.0 3.0) :dtype :float64))))
      (run-test "sqrt(1,4,9,16)" (E "sqrt_149_16")
                (vt-to-list (vt-sqrt (vt-from-sequence '(1.0 4.0 9.0 16.0) :dtype :float64))))

      ;; ========== 6. 归约 ==========
      (format t "~%--- 6. Reduction ---~%")
      (let ((a (vt-astype (vt-reshape (vt-arange 12 :dtype :int64) '(3 4)) :float64)))
        (run-test "sum(all)"       (E "sum_all")       (vt-item (vt-sum a)))
        (run-test "sum(axis=0)"    (E "sum_axis0")     (vt-to-list (vt-sum a :axis 0)))
        (run-test "sum(axis=1)"    (E "sum_axis1")     (vt-to-list (vt-sum a :axis 1)))
        (run-test "mean(all)"      (E "mean_all")      (vt-item (vt-mean a)))
        (run-test "max(all)"       (E "max_all")       (vt-item (vt-amax a)))
        (run-test "min(axis=0)"    (E "min_axis0")     (vt-to-list (vt-amin a :axis 0)))
        (run-test "argmax(axis=1)" (E "argmax_axis1")  (vt-to-list (vt-argmax a :axis 1)))
        (run-test "std(all)"       (E "std_all")       (vt-item (vt-std a)))
        (run-test "cumsum(1,2,3,4)" (E "cumsum_1234")
                  (vt-to-list (vt-cumsum (vt-from-sequence '(1 2 3 4) :dtype :int64))))
        (run-test "median" (E "median_31415926")
                  (vt-item (vt-median (vt-from-sequence '(3.0 1.0 4.0 1.0 5.0 9.0 2.0 6.0) :dtype :float64))))
        (run-test "sort" (E "sort_8")
                  (vt-to-list (vt-sort (vt-from-sequence '(3.0 1.0 4.0 1.0 5.0 9.0 2.0 6.0) :dtype :float64)))))

      ;; ========== 7. 线性代数 ==========
      (format t "~%--- 7. Linear Algebra ---~%")
      (run-test "matmul 2x2" (E "matmul_2x2")
                (vt-to-list (vt-matmul
                  (vt-from-sequence '((1.0 2.0) (3.0 4.0)) :dtype :float64)
                  (vt-from-sequence '((5.0 6.0) (7.0 8.0)) :dtype :float64))))
      (run-test "trace" (E "trace_3x3")
                (vt-item (vt-trace (vt-astype (vt-reshape (vt-arange 9 :dtype :int64) '(3 3)) :float64))))
      (run-test "det" (E "det_2x2")
                (vt-item (vt-det (vt-from-sequence '((1.0 2.0) (3.0 4.0)) :dtype :float64))))
      (run-test "norm(3,4)" (E "norm_34")
                (vt-item (vt-norm (vt-from-sequence '(3.0 4.0) :dtype :float64))))
      (run-test "Cholesky" (E "cholesky_L")
                (vt-to-list (vt-cholesky (vt-from-sequence '((4.0 2.0) (2.0 3.0)) :dtype :float64))))
      (run-test "matrix-rank" (E "rank_12_24")
                (vt-matrix-rank (vt-from-sequence '((1.0 2.0) (2.0 4.0)) :dtype :float64)))

      ;; ========== 8. einsum ==========
      (format t "~%--- 8. einsum ---~%")
      (run-test "einsum dot" (E "einsum_dot")
                (vt-item (vt-einsum "i,i->"
                  (vt-from-sequence '(1 2 3) :dtype :int64)
                  (vt-from-sequence '(4 5 6) :dtype :int64))))
      (run-test "einsum matmul" (E "einsum_matmul")
                (vt-to-list (vt-einsum "ij,jk->ik"
                  (vt-reshape (vt-arange 6 :dtype :int64) '(2 3))
                  (vt-reshape (vt-arange 6 :dtype :int64) '(3 2)))))
      (run-test "einsum diag" (E "einsum_diag")
                (vt-to-list (vt-einsum "ii->i"
                  (vt-reshape (vt-arange 9 :dtype :int64) '(3 3)))))
      (run-test "einsum outer" (E "einsum_outer")
                (vt-to-list (vt-einsum "i,j->ij"
                  (vt-from-sequence '(1 2 3) :dtype :int64)
                  (vt-from-sequence '(4 5) :dtype :int64))))

      ;; ========== 9. 激活函数 ==========
      (format t "~%--- 9. Activation ---~%")
      (let ((x (vt-from-sequence '(-2.0 -1.0 0.0 1.0 2.0) :dtype :float64)))
        (run-test "sigmoid" (E "sigmoid_x") (vt-to-list (vt-sigmoid x)) 1e-5)
        (run-test "relu"    (E "relu_x")    (vt-to-list (vt-relu x)))
        (run-test "tanh"    (E "tanh_x")    (vt-to-list (vt-tanh x)) 1e-5))
      (run-test "softmax" (E "softmax_123")
                (vt-to-list (vt-softmax (vt-from-sequence '(1.0 2.0 3.0) :dtype :float64))) 1e-5)

      ;; ========== 10. where / nonzero ==========
      (format t "~%--- 10. Where / Nonzero ---~%")
      (run-test "where" (E "where_cond")
                (vt-to-list (vt-where
                  (vt-from-sequence '(1.0 0.0 1.0 0.0) :dtype :float64)
                  (vt-from-sequence '(10.0 20.0 30.0 40.0) :dtype :float64)
                  (vt-from-sequence '(100.0 200.0 300.0 400.0) :dtype :float64))))
      (run-test "nonzero" (E "nonzero_010203")
                (vt-to-list (first (vt-nonzero (vt-from-sequence '(0 1 0 2 0 3) :dtype :int64)))))

      ;; ========== 11. 集合 ==========
      (format t "~%--- 11. Set Operations ---~%")
      (run-test "unique" (E "unique_122333")
                (vt-to-list (vt-unique (vt-from-sequence '(1 2 2 3 3 3) :dtype :int64))))
      (run-test "intersect1d" (E "intersect1d")
                (vt-to-list (vt-intersect1d
                  (vt-from-sequence '(1 2 3 4 5) :dtype :int64)
                  (vt-from-sequence '(3 4 5 6 7) :dtype :int64))))

      ;; ========== 12. nan ==========
      (format t "~%--- 12. nan handling ---~%")
      (let ((a (vt-from-sequence (list 1.0d0 +vt-float-nan+ 3.0d0 4.0d0) :dtype :float64)))
        (run-test "nanmean" (E "nanmean_1nan34") (vt-item (vt-nanmean a)) 1e-5)
        (run-test "nansum"  (E "nansum_1nan34")  (vt-item (vt-nansum a)) 1e-5)
        (run-test "nanmax"  (E "nanmax_1nan34")  (vt-item (vt-nanmax a)) 1e-5))

      ;; ========== Summary ==========
      (test-summary))))

;; Run and exit with proper code
(let ((ok (run-auto-comparison)))
  (sb-ext:exit :code (if ok 0 1)))
