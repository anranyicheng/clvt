;;;; numpy-compare-test.lisp — 运行时调用 numpy 生成参考并对比（无pytorch依赖）
;;;; 用法: 在项目根目录运行（或由 run-tests.sh 调用）

(require :asdf)
(push (truename (make-pathname :directory '(:relative :up))) asdf:*central-registry*)
(asdf:load-system :clvt)
(in-package :clvt)

;;; ------------------------------------------------------------------
;;; 简易 JSON 解析器（从字符串解析）
;;; ------------------------------------------------------------------
(defun parse-json-string (s)
  (let ((pos 0) (len (length s)))
    (labels ((skip ()
               (loop while (and (< pos len) (member (char s pos) '(#\Space #\Tab #\Newline #\Return)))
                     do (incf pos)))
             (rd ()
               (skip)
               (when (>= pos len) (return-from rd nil))
               (let ((c (char s pos)))
                 (cond
                   ((char= c #\") (incf pos)
                    (let ((st pos))
                      (loop while (and (< pos len) (char/= (char s pos) #\"))
                            do (when (char= (char s pos) #\\) (incf pos)) (incf pos))
                      (prog1 (subseq s st pos) (incf pos))))
                   ((char= c #\[) (incf pos) (skip)
                    (let ((a '()))
                      (loop while (and (< pos len) (char/= (char s pos) #\]))
                            do (push (rd) a) (skip)
                               (when (and (< pos len) (char= (char s pos) #\,)) (incf pos) (skip)))
                      (incf pos) (nreverse a)))
                   ((char= c #\{) (incf pos) (skip)
                    (let ((h (make-hash-table :test 'equal)))
                      (loop while (and (< pos len) (char/= (char s pos) #\}))
                            do (let ((k (rd))) (skip)
                                 (when (and (< pos len) (char= (char s pos) #\:)) (incf pos))
                                 (setf (gethash k h) (rd)))
                               (skip)
                               (when (and (< pos len) (char= (char s pos) #\,)) (incf pos) (skip)))
                      (incf pos) h))
                   ((or (char<= #\0 c #\9) (char= c #\-) (char= c #\+))
                    (let ((st pos))
                      (when (member c '(#\- #\+)) (incf pos))
                      (loop while (and (< pos len)
                                       (or (char<= #\0 (char s pos) #\9)
                                           (char= (char s pos) #\.)
                                           (member (char s pos) '(#\e #\E #\- #\+))))
                            do (incf pos))
                      (let ((ss (subseq s st pos)))
                        (if (or (find #\. ss) (find #\e ss) (find #\E ss))
                            (let ((*read-eval* nil)) (read-from-string ss))
                            (parse-integer ss)))))
                   (t (error "JSON 解析错误 @~a: ~a" pos c))))))
      (rd))))

;;; ------------------------------------------------------------------
;;; 框架
;;; ------------------------------------------------------------------
(defvar *n* 0) (defvar *p* 0) (defvar *f* 0) (defvar *fails* '())

(defun approx-val (e a &optional (tol 1e-5))
  (cond
    ((and (numberp e) (numberp a))
     ;; 对于所有数字比较都用容差，不严格要求类型匹配
     (let ((ee (float e 1.0d0)) (aa (float a 1.0d0)))
       (< (abs (- ee aa)) (+ tol (* tol (max (abs ee) 1.0d0))))))
    ((and (listp e) (listp a))
     (and (= (length e) (length a))
          (every (lambda (x y) (approx-val x y tol)) e a)))
    (t (equal e a))))

(defun t! (name expected actual &optional (tol 1e-5))
  (incf *n*)
  (if (approx-val expected actual tol)
      (incf *p*)
      (progn (incf *f*)
             (push name *fails*)
             (format t "  ❌ ~a~%     exp: ~a~%     got: ~a~%" name expected actual))))

;;; ------------------------------------------------------------------
;;; 调用 python3 获取参考结果
;;; ------------------------------------------------------------------
(defun run-reference ()
  (let* ((here (or *load-truename* *compile-file-truename* *default-pathname-defaults*))
         (script (namestring (merge-pathnames "ref_compute.py" here))))
    (format t "  🔄 实时调用numpy生成参考值...~%")
    (uiop:run-program (list "python3" script) :output :string)))

;;; ------------------------------------------------------------------
;;; 参考值与 clvt 结果对比
;;; ------------------------------------------------------------------
(defun run-numpy-compare ()
  (format t "~%============================================================~%")
  (format t "  clvt vs numpy 实时对比测试 (无pytorch依赖)~%")
  (format t "============================================================~%~%")
  (let ((ref (parse-json-string (run-reference))))
    (flet ((E (k) (gethash k ref))
           (vt (x) (vt-to-list x))
           (vi (x) (vt-item x)))

      ;; 创建
      (t! "arange" (E "arange") (vt (vt-arange 10 :dtype :int64)))
      (t! "linspace" (E "linspace") (vt (vt-linspace 0.0 1.0 5)))
      (t! "logspace" (E "logspace") (vt (vt-logspace 0 3 4)))
      (t! "eye" (E "eye") (vt (vt-eye 3 :dtype :float64)))
      (t! "diag" (E "diag") (vt (vt-diag (vt-from-sequence '(1 2 3) :dtype :int64))))

      ;; 逐元素
      (let ((a (vt-from-sequence '(1.0 2.0 3.0 4.0)))
            (b (vt-from-sequence '(5.0 6.0 7.0 8.0)))
            (x (vt-from-sequence '(-2.0 -1.0 0.0 1.0 2.0))))
        (t! "add" (E "add") (vt (vt-+ a b)))
        (t! "sub" (E "sub") (vt (vt-- b a)))
        (t! "mul" (E "mul") (vt (vt-* a b)))
        (t! "div" (E "div") (vt (vt-/ b a)))
        (t! "sin" (E "sin") (vt (vt-sin x)))
        (t! "cos" (E "cos") (vt (vt-cos x)))
        (t! "tanh" (E "tanh") (vt (vt-tanh x)))
        (t! "exp" (E "exp") (vt (vt-exp x)))
        (t! "log" (E "log") (vt (vt-log (vt-from-sequence '(1.0 2.0 3.0 4.0)))))
        (t! "sqrt" (E "sqrt") (vt (vt-sqrt (vt-from-sequence '(1.0 4.0 9.0 16.0)))))
        (t! "abs" (E "abs") (vt (vt-abs (vt-from-sequence '(-3.0 -1.0 0.0 1.0 3.0)))))
        (t! "clip" (E "clip") (vt (vt-clip (vt-from-sequence '(1.0 2.0 3.0 4.0)) 2.0 3.0)))
        (t! "maximum" (E "max_ab") (vt (vt-maximum a b)))
        (t! "minimum" (E "min_ab") (vt (vt-minimum a b)))
        (t! "pow" (E "pow") (vt (vt-pow a 2.0)))
        (t! "reciprocal" (E "reciprocal") (vt (vt-reciprocal a)))
        (t! "cbrt" (E "cbrt") (vt (vt-cbrt (vt-from-sequence '(-8.0 -1.0 0.0 1.0 8.0))))))

      ;; 归约 / 统计
      (let ((m (vt-reshape (vt-arange 12 :dtype :float64) '(3 4))))
        (t! "sum_all" (E "sum_all") (vi (vt-sum m)))
        (t! "sum_axis0" (E "sum_axis0") (vt (vt-sum m :axis 0)))
        (t! "sum_axis1" (E "sum_axis1") (vt (vt-sum m :axis 1)))
        (t! "mean_all" (E "mean_all") (vi (vt-mean m)))
        (t! "mean_axis1" (E "mean_axis1") (vt (vt-mean m :axis 1)))
        (t! "var_all" (E "var_all") (vi (vt-var m)))
        (t! "std_all" (E "std_all") (vi (vt-std m)))
        (t! "max_axis0" (E "max_axis0") (vt (vt-amax m :axis 0)))
        (t! "min_axis1" (E "min_axis1") (vt (vt-amin m :axis 1)))
        (t! "argmax_axis1" (E "argmax_axis1") (vt (vt-argmax m :axis 1)))
        (t! "argmin_axis0" (E "argmin_axis0") (vt (vt-argmin m :axis 0)))
        (t! "prod_all" (E "prod_all") (vi (vt-prod m))))
      (t! "cumsum" (E "cumsum") (vt (vt-cumsum (vt-from-sequence '(1.0 2.0 3.0 4.0)))))
      (t! "cumprod" (E "cumprod") (vt (vt-cumprod (vt-from-sequence '(1.0 2.0 3.0 4.0)))))
      (t! "median" (E "median") (vi (vt-median (vt-from-sequence '(3.0 1.0 4.0 1.0 5.0 9.0 2.0 6.0)))))
      (t! "percentile" (E "percentile") (vi (vt-percentile (vt-from-sequence '(1.0 2.0 3.0 4.0 5.0)) 90)))
      (t! "ptp" (E "ptp") (vi (vt-ptp (vt-from-sequence '(3.0 1.0 4.0 1.0 5.0)))))
      (t! "sort" (E "sort") (vt (vt-sort (vt-from-sequence '(3.0 1.0 4.0 1.0 5.0 9.0 2.0 6.0)) :axis 0)))
      (t! "argsort" (E "argsort") (vt (vt-argsort (vt-from-sequence '(3.0 1.0 4.0 1.0 5.0 9.0 2.0 6.0)) :axis 0)))
      (t! "diff" (E "diff") (vt (vt-diff (vt-from-sequence '(1.0 3.0 6.0 10.0 15.0)))))

      ;; 线性代数
      (let ((M (vt-from-sequence '((1.0 2.0) (3.0 4.0))))
            (N (vt-from-sequence '((5.0 6.0) (7.0 8.0)))))
        (t! "matmul" (E "matmul") (vt (vt-matmul M N)))
        (t! "det" (E "det") (vi (vt-det M)))
        (t! "inv" (E "inv") (vt (vt-inv M)))
        (t! "cholesky" (E "cholesky") (vt (vt-cholesky (vt-from-sequence '((4.0 2.0) (2.0 3.0)))))))
      (t! "dot" (E "dot") (vi (vt-dot (vt-from-sequence '(1.0 2.0 3.0)) (vt-from-sequence '(4.0 5.0 6.0)))))
      (t! "outer" (E "outer") (vt (vt-outer (vt-from-sequence '(1.0 2.0 3.0)) (vt-from-sequence '(4.0 5.0)))))
      (t! "trace" (E "trace") (vi (vt-trace (vt-reshape (vt-arange 9 :dtype :float64) '(3 3)))))
      (t! "norm" (E "norm") (vi (vt-norm (vt-from-sequence '(3.0 4.0)))))
      (t! "solve" (E "solve") (vt (vt-solve (vt-from-sequence '((2.0 1.0) (1.0 3.0))) (vt-from-sequence '(7.0 8.0)))))
      (t! "matrix_rank" (E "matrix_rank") (vt-matrix-rank (vt-from-sequence '((1.0 2.0) (2.0 4.0)))))
      (t! "eigvals" (E "eigvals") (vt (nth-value 0 (vt-eig (vt-from-sequence '((2.0 1.0) (1.0 3.0)))))))
      (t! "einsum_dot" (E "einsum_dot") (vi (vt-einsum "i,i->" (vt-from-sequence '(1.0 2.0 3.0)) (vt-from-sequence '(4.0 5.0 6.0)))))
      (t! "einsum_outer" (E "einsum_outer") (vt (vt-einsum "i,j->ij" (vt-from-sequence '(1.0 2.0 3.0)) (vt-from-sequence '(4.0 5.0)))))
      (t! "einsum_trace" (E "einsum_trace") (vi (vt-einsum "ii->" (vt-reshape (vt-arange 9 :dtype :float64) '(3 3)))))

      ;; 神经网络（numpy实现）
      (let ((x (vt-from-sequence '(-2.0 -1.0 0.0 1.0 2.0))))
        (t! "sigmoid" (E "sigmoid") (vt (vt-sigmoid x)))
        (t! "relu" (E "relu") (vt (vt-relu x)))
        (t! "gelu" (E "gelu") (vt (vt-gelu x))))
      (t! "softmax" (E "softmax") (vt (vt-softmax (vt-from-sequence '(1.0 2.0 3.0)))))
      (t! "log_softmax" (E "log_softmax") (vt (vt-log-softmax (vt-from-sequence '(1.0 2.0 3.0)))))

      ;; 集合
      (t! "unique" (E "unique") (vt (vt-unique (vt-from-sequence '(1 2 2 3 3 3) :dtype :int64))))
      (t! "intersect1d" (E "intersect1d") (vt (vt-intersect1d (vt-from-sequence '(1 2 3 4 5) :dtype :int64)
                                                               (vt-from-sequence '(3 4 5 6 7) :dtype :int64))))

      ;; 形状
      (t! "reshape" (E "reshape") (vt (vt-reshape (vt-arange 6 :dtype :int64) '(2 3))))
      (t! "transpose" (E "transpose") (vt (vt-transpose (vt-reshape (vt-arange 6 :dtype :int64) '(2 3)))))
      (t! "concatenate" (E "concatenate") (vt (vt-concatenate 0
                                                 (vt-from-sequence '((1 2) (3 4)) :dtype :int64)
                                                 (vt-from-sequence '((5 6) (7 8)) :dtype :int64))))
      (t! "broadcast" (E "broadcast") (vt (vt-broadcast-to (vt-from-sequence '(1 2 3) :dtype :int64) '(2 3))))

      ;; topk
      (multiple-value-bind (vals idxs)
          (vt-topk (vt-from-sequence '((1.0 5.0 3.0) (7.0 2.0 9.0))) 2)
        (t! "topk_vals" (E "topk_vals") (vt vals))
        (t! "topk_idx" (E "topk_idx") (vt idxs)))))

  (format t "~%============================================================~%")
  (format t "  Total: ~a | Pass: ~a | Fail: ~a~%" *n* *p* *f*)
  (format t "============================================================~%")
  (when *fails*
    (format t "Failed:~{~%  - ~a~}~%" (reverse *fails*)))
  (zerop *f*))

(run-numpy-compare)
