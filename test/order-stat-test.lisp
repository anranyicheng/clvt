;;;; order-stat-test.lisp — 顺序统计量（median / percentile / nanmedian）的
;;;;                        正确性 + 退化护栏
;;;;
;;;; 背景（CONVENTIONS §16.35.5）：_select-double_ 是 median / percentile /
;;;; quantile / nan* 家族共用的 quickselect。它有两类真实退化，都会让
;;;; 「200k 元素的取中位」从毫秒级变成秒级：
;;;;   1. 两分区在大量重复值上 O(n²)（全零数组 10.7 s）；
;;;;   2. 三分区后，整体降序且 k = n/2-1 时每趟只削掉极窄一段（142 ms，
;;;;      30.2M 次访问）。
;;;; 本套件两件事：
;;;;   A. 正确性：与「先排序再按位置取」的独立参照逐点一致（不经过 quickselect）；
;;;;   B. 护栏：重复值密集 / 降序 / 升序等退化输入相对随机输入的倍率上限
;;;;      （倍率而非绝对毫秒，避免与机器速度耦合）。
;;;;
;;;; 用法：sbcl --dynamic-space-size 4096 --non-interactive --load test/order-stat-test.lisp

(require :asdf)
(defparameter cl-user::*clvt-test-dir*
  (make-pathname :name nil :type nil :defaults *load-truename*))
(pushnew (merge-pathnames "../" cl-user::*clvt-test-dir*) asdf:*central-registry* :test #'equal)
(handler-bind ((warning #'muffle-warning)) (asdf:load-system :clvt))

(in-package :clvt)

;;; ------------------------------------------------------------------
;;; 骨架
;;; ------------------------------------------------------------------

(defvar *pass* 0)
(defvar *fail* 0)
(defvar *failures* '())

(defun expect-true (name form)
  (if form
      (incf *pass*)
      (progn (incf *fail*)
             (push name *failures*)
             (format t "~&FAIL ~a~%" name))))

(defun expect-equal (name got want)
  (if (equal got want)
      (incf *pass*)
      (progn (incf *fail*)
             (push (format nil "~a: 期望 ~s 实得 ~s" name want got) *failures*)
             (format t "~&FAIL ~a: 期望 ~s 实得 ~s~%" name want got))))

(defun expect-close (name got want tol)
  (let ((ok (cond ((and (floatp got) (sb-ext:float-nan-p got))
                   (and (floatp want) (sb-ext:float-nan-p want)))
                  ((and (floatp got) (floatp want)) (<= (abs (- got want)) tol))
                  (t (equal got want)))))
    (if ok
        (incf *pass*)
        (progn (incf *fail*)
               (push (format nil "~a: 期望 ~s 实得 ~s" name want got) *failures*)
               (format t "~&FAIL ~a: 期望 ~s 实得 ~s~%" name want got)))))

(defun scalar-of (x)
  "从 0 维张量取值（median/percentile 的结果是 0 维）。"
  (if (zerop (vt-order x)) (vt-ref x) (vt-ref x 0)))

(defun ms-of (thunk &key (runs 3))
  (let ((best most-positive-fixnum))
    (dotimes (_ runs)
      (sb-ext:gc :full t)
      (let ((t0 (get-internal-real-time)))
        (funcall thunk)
        (setf best (min best (- (get-internal-real-time) t0)))))
    (/ (* 1000.0d0 best) internal-time-units-per-second)))

;;; ------------------------------------------------------------------
;;; 数据构造
;;; ------------------------------------------------------------------

(defun make-data (kind n)
  "构造 KIND 型的 1-D float64 张量（N 元素）。"
  (ecase kind
    (:random (vt-random-uniform (list n) :low 0d0 :high 1d0 :dtype :float64))
    (:const  (vt-+ (vt-zeros (list n)) 1.5d0))
    (:two    (vt-astype (vt-> (vt-random-uniform (list n) :low 0d0 :high 1d0) 0.5d0)
                        :float64))
    (:few    (vt-astype (vt-mod (vt-astype (vt-* (vt-random-uniform (list n) :low 0d0 :high 1d0)
                                                1000d0)
                                          :int64)
                               7)
                        :float64))
    (:asc    (vt-sort (make-data :random n)))
    (:desc   (vt-flip (make-data :asc n) :axis 0))
    ;; 全部为 NaN（应传播 NaN）
    (:nan    (vt-astype (vt-zeros (list n)) :float64))))

;;; ------------------------------------------------------------------
;;; A. 正确性：与「排序 + 位置公式」参照一致
;;; ------------------------------------------------------------------

(defun ref-median (x)
  "参照实现：排序后取中位（偶数取两中位均值）。独立于 quickselect。"
  (let* ((s (vt-sort (vt-flatten x)))
         (n (vt-size s))
         (v (vt-data s)))
    (declare (type (simple-array double-float (*)) v))
    (if (evenp n)
        (/ (+ (aref v (1- (floor n 2))) (aref v (floor n 2))) 2.0d0)
        (aref v (floor n 2)))))

(defun ref-percentile (x q)
  "参照实现：排序后按 numpy 的线性插值位置公式取分位点。"
  (let* ((s (vt-sort (vt-flatten x)))
         (n (vt-size s))
         (v (vt-data s))
         (pos (* q (1- n)))
         (lo (floor pos))
         (hi (min (ceiling pos) (1- n)))
         (frac (- pos lo)))
    (declare (type (simple-array double-float (*)) v)
             (type fixnum lo hi) (type double-float frac))
    (if (= lo hi)
        (aref v lo)
        (+ (* (- 1.0d0 frac) (aref v lo)) (* frac (aref v hi))))))

(defun run-correctness ()
  (format t "~&--- A. 正确性（vs 排序参照）---~%")
  (dolist (kind '(:random :const :two :few :asc :desc))
    (dolist (n '(1 2 3 7 8 100 1001))
      (let ((x (make-data kind n)))
        (expect-close (format nil "median ~a n=~d" kind n)
                      (scalar-of (vt-median x)) (ref-median x) 1d-12)
        (dolist (q '(25 50 75 0 100))
          (expect-close (format nil "percentile(~a) ~a n=~d" q kind n)
                        (scalar-of (vt-percentile x q))
                        (ref-percentile x (/ q 100.0d0))
                        1d-12)))))
  ;; nanmedian 必须跳过 NaN，与「过滤后取中位」一致
  (let* ((x (vt-from-sequence (list 5.0d0 +vt-dfloat-nan+ 1.0d0 3.0d0 +vt-dfloat-nan+)
                              :dtype :float64)))
    (expect-close "nanmedian 跳过 NaN" (scalar-of (vt-nanmedian x)) 3.0d0 1d-12))
  ;; 全 NaN 传播
  (let ((x (vt-const '(5) +vt-dfloat-nan+ :dtype :float64)))
    (expect-true "median 全 NaN 传播" (sb-ext:float-nan-p (scalar-of (vt-median x))))
    (expect-true "nanmedian 全 NaN 传播" (sb-ext:float-nan-p (scalar-of (vt-nanmedian x))))))

;;; ------------------------------------------------------------------
;;; B. 退化护栏：重复值密集 / 已排序 / 逆序，倍率相对随机输入
;;; ------------------------------------------------------------------

(defparameter *perf-n* 200000)

(defun run-guards ()
  (format t "~&--- B. 退化护栏（n=~d，相对随机输入的倍率）---~%" *perf-n*)
  (let* ((n *perf-n*)
         ;; 基准与被测数据都必须**先构造好**：把 make-data 放进计时区间会把
         ;; 随机数生成（200k ≈ 40ms）算进去，倍率全部变成 1.0x（踩过）。
         (rand (make-data :random n))
         (base (ms-of (lambda () (vt-median rand))))
         (base-pct (ms-of (lambda () (vt-percentile rand 30)))))
    (format t "~&    基准：median(随机) ~,2f ms | percentile(随机) ~,2f ms~%"
            base base-pct)
    (dolist (kind '(:const :two :few :asc :desc))
      (let* ((x (make-data kind n))
             (t-med (ms-of (lambda () (vt-median x))))
             (t-pct (ms-of (lambda () (vt-percentile x 30))))
             (r-med (max (/ t-med (max base 0.01d0)) 1.0001d0))
             (r-pct (max (/ t-pct (max base-pct 0.01d0)) 1.0001d0)))
        (format t "~&    ~6a median ~7,2f ms (~5,1fx) | percentile ~7,2f ms (~5,1fx)~%"
                kind t-med r-med t-pct r-pct)
        ;; 倍率上限 8：修复前 const 是 5000x、desc 是 18x
        (expect-true (format nil "median ~a ≤ 8× 随机（实测 ~,1fx）" kind r-med)
                     (<= r-med 8.0d0))
        (expect-true (format nil "percentile ~a ≤ 8× 随机（实测 ~,1fx）" kind r-pct)
                     (<= r-pct 8.0d0))))
    ;; 绝对下界：全等数组是两分区退化的极端（曾 10.7 s/200k），给一条硬上限
    (let ((t-const (ms-of (lambda () (vt-median (make-data :const n))))))
      (format t "~&    全等数组 median 绝对值 ~,2f ms（修复前 10700 ms 级）~%" t-const)
      (expect-true (format nil "全等数组 median ≤ 200 ms（实测 ~,1f ms）" t-const)
                   (<= t-const 200.0d0)))))

;;; ------------------------------------------------------------------
;;; 入口
;;; ------------------------------------------------------------------

(format t "~&=====================================~%")
(format t "顺序统计量：正确性 + 退化护栏 (CONVENTIONS §16.35.5)~%")
(format t "=====================================~%")
(run-correctness)
(run-guards)
(format t "~&=====================================~%")
(format t "Total: ~d / Pass: ~d / Fail: ~d~%" (+ *pass* *fail*) *pass* *fail*)
(format t "PASS=~d FAIL=~d~%" *pass* *fail*)
(when *failures*
  (format t "~&--- 失败清单 ---~%")
  (dolist (f (reverse *failures*)) (format t "  · ~a~%" f)))
(when (plusp *fail*) (sb-ext:exit :code 1))
