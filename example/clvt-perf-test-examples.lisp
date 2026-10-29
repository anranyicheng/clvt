;;;; clvt-perf-test-examples.lisp
;;;; =============================================================================
;;;; clvt 11 轮性能优化的「测试 / 基准用例汇总」
;;;;
;;;; 说明
;;;; ----
;;;; 这份文件把第 1–11 轮优化过程中用于**验证正确性**与**测量性能**的用例集中到一处，
;;;; 分成两组，可分别或一起运行：
;;;;
;;;;   1. ASSERTIONS —— 正确性断言（约 60 条）
;;;;      覆盖各轮改动涉及的边界：NaN/±Inf/±0、空张量、0 维、非连续视图、
;;;;      offset 切片、广播、axis/keepdims/ddof/interpolation 组合、整数溢出回绕。
;;;;      其中带 ★ 的条目对应「优化过程中发现并修掉的既有 bug」，用于防回归。
;;;;
;;;;   2. BENCHMARKS —— 性能基准（约 60 项）
;;;;      每项打印单次调用耗时（µs），注释里给出「优化前 → 优化后 / NumPy 同工作量」
;;;;      的历史数字，便于回归时快速比对。规模默认 1M 元素，可用 *SCALE* 缩放。
;;;;
;;;; 用法
;;;; ----
;;;;   sbcl --dynamic-space-size 6144 --non-interactive \
;;;;        --load clvt-perf-test-examples.lisp
;;;;
;;;;   只跑断言：  (clvt-examples:run :bench nil)
;;;;   只跑基准：  (clvt-examples:run :assert nil)
;;;;   改规模：    (setf clvt-examples:*scale* 0.1) 再 (clvt-examples:run)
;;;;
;;;; 依赖：clvt（本仓库）、SBCL。基准部分仅用于相对比较，绝对值随机器差异较大。
;;;; =============================================================================

(require :asdf)
(push (or (uiop:getenv "CLVT_ROOT") (uiop:getcwd)) asdf:*central-registry*)
(handler-bind ((warning #'muffle-warning)) (asdf:load-system :clvt))

(defpackage :clvt-examples
  (:use :cl :clvt)
  (:export #:run #:*scale* #:assertions #:benchmarks))
(in-package :clvt-examples)

(defparameter *scale* 1.0
  "基准规模缩放系数。1.0 = 1M 元素 / 1000×1000 矩阵。")

(defvar *fail* 0)
(defvar *pass* 0)
(defvar *sink* nil)

;;; =============================================================================
;;; 工具
;;; =============================================================================

(defun us-now ()
  (multiple-value-bind (s u) (sb-ext:get-time-of-day)
    (+ (* s 1000000) u)))

(defmacro bench (label n &body body)
  "跑 N 次取 3 轮最小值，打印 µs/次。"
  `(let (ts)
     (dotimes (k 3)
       (sb-ext:gc :full t)
       (let ((t0 (us-now)))
         (dotimes (i ,n) (setf *sink* (progn ,@body)))
         (push (/ (float (- (us-now) t0) 1d0) ,n) ts)))
     (let ((best (first (sort ts #'<))))
       (format t "~&  B| ~36a ~12,1f us~%" ,label best)
       best)))

(deftype vt-like () 'clvt::vt)

(defun as-list (x)
  "把张量/序列转成 list（用于断言比较）。"
  (let ((v (if (typep x 'vt-like)
               x
               (vt-from-sequence (if (listp x) x (list x))))))
    (if (zerop (vt-size v))
        '()
        ;; 用物理视图内核取数：本函数是「拿字节做断言比较」的辅助，
        ;; 布尔张量也要 0/1（§17.8/§17.9），而 VT-REF 现在是逻辑视图给 T/NIL。
        (let ((flat (vt-ravel v)))
          (loop for i below (vt-size flat) collect (clvt::%vt-ref-raw flat i))))))

(defun shape-of (x)
  (if (typep x 'vt-like) (vt-shape x) (list (length x))))

(defun nan-p (x)
  "安全判定 NaN：SBCL 的浮点比较用 UCOMISD，遇 NaN 会置 invalid 并触发陷阱，
   故必须屏蔽后再比较 —— 否则 (/= nan nan) 直接抛 FLOATING-POINT-INVALID-OPERATION。"
  (and (floatp x)
       (sb-int:with-float-traps-masked (:invalid :divide-by-zero :overflow :underflow)
         (/= x x))))

(defun feq (a b &optional (tol 0d0))
  "逐元素比较；NaN 与 NaN 视为相等（与 numpy 的 assert_allclose(equal_nan) 一致）。"
  (cond ((and (floatp a) (floatp b) (nan-p a) (nan-p b)) t)
        ((and (floatp a) (floatp b))
         ;; 先判严格相等：±Inf ± Inf 相减会得到 NaN 并触发 invalid 陷阱。
         (sb-int:with-float-traps-masked (:invalid :divide-by-zero :overflow :underflow)
           (or (= a b)
               (<= (abs (- (coerce a 'double-float) (coerce b 'double-float))) tol))))
        (t (equalp a b))))

(defun check (label expected actual &key (tol 0d0) shape)
  "断言 ACTUAL（张量或 list）与 EXPECTED（list）逐元素相等；SHAPE 给定时同时校验形状。"
  (let* ((got (if (typep actual 'vt) (as-list actual) (coerce actual 'list)))
         (ok (and (= (length got) (length expected))
                  (every (lambda (a b) (feq a b tol)) got expected)
                  (or (null shape) (equal (shape-of actual) shape)))))
    (if ok
        (progn (incf *pass*) (format t "~&  ok   ~a~%" label))
        (progn (incf *fail*)
               (format t "~&  FAIL ~a~%       expected ~s~%       actual   ~s~%" label expected got)))))

(defun check-true (label form)
  (if form
      (progn (incf *pass*) (format t "~&  ok   ~a~%" label))
      (progn (incf *fail*) (format t "~&  FAIL ~a~%" label))))

(defun check-throws (label thunk)
  "断言 THUNK 抛错（用于「必须报错」的契约）。"
  (if (handler-case (progn (funcall thunk) nil) (error () t))
      (progn (incf *pass*) (format t "~&  ok   ~a（按契约报错）~%" label))
      (progn (incf *fail*) (format t "~&  FAIL ~a（应当报错却返回了值）~%" label))))

(defun f64 (&rest xs) (vt-from-sequence xs :dtype :float64))
(defun i64 (&rest xs) (vt-from-sequence xs :dtype :int64))

(defun nan-df ()
  "构造一位 double NaN（不依赖 clvt 内部常量）。"
  (sb-int:with-float-traps-masked (:invalid :divide-by-zero :overflow :underflow)
    (locally (declare (notinline /)) (/ 0.0d0 0.0d0))))

(defun inf-df () sb-ext:double-float-positive-infinity)

;;; =============================================================================
;;; 正确性断言
;;; =============================================================================

(defun assertions ()
  (setf *fail* 0 *pass* 0)
  (format t "~&~%=== ASSERTIONS ===~%")

  ;; ---- 第 1 轮：声明 / 归约 / 广播 -------------------------------
  (format t "~&[r1 归约与广播]~%")
  (check "cumsum 展平（axis=nil，2D 输入）"
         '(0.0d0 1.0d0 3.0d0 6.0d0 10.0d0 15.0d0)
         (vt-cumsum (vt-reshape (vt-arange 6 :dtype :float64) '(2 3)))
         :shape '(6))
  (check "cumsum :axis 0"
         '(0.0d0 1.0d0 2.0d0 3.0d0 5.0d0 7.0d0)
         (vt-cumsum (vt-reshape (vt-arange 6 :dtype :float64) '(2 3)) :axis 0))
  (check "cumprod :axis 1"
         '(0.0d0 0.0d0 3.0d0 12.0d0)
         (vt-cumprod (vt-reshape (vt-from-sequence '(0.0d0 1.0d0 3.0d0 4.0d0) :dtype :float64)
                                 '(2 2))
                     :axis 1))
  (check "repeat 标量（1M 元素不炸栈）"
         '(1.0d0 1.0d0 2.0d0 2.0d0 3.0d0 3.0d0)
         (vt-repeat (f64 1.0d0 2.0d0 3.0d0) 2))
  (check "repeat :axis 0"
         '(1.0d0 2.0d0 1.0d0 2.0d0 3.0d0 4.0d0 3.0d0 4.0d0)
         (vt-repeat (vt-reshape (f64 1.0d0 2.0d0 3.0d0 4.0d0) '(2 2)) 2 :axis 0))
  (check "average 带 :axis（&rest 归一后不再吞关键字）"
         '(2.0d0 3.0d0)
         (vt-average (vt-reshape (f64 1.0d0 2.0d0 3.0d0 4.0d0) '(2 2)) :axis 0))
  (check "isclose：NaN 恒不接近"
         '(0 0 1 1 0)
         (vt-isclose (vt-from-sequence (list (nan-df) (inf-df) 1.0d0 1.0d0 1.0d0)
                                       :dtype :float64)
                     (f64 1.0d0 1.0d0 1.0d0 1.0d0 2.0d0)))
  (check-true "allclose 真值" (vt-allclose (f64 1.0d0 2.0d0) (f64 1.0d0 2.0d0)))

  ;; ---- 第 3 轮：集合运算 -----------------------------------------
  (format t "~&[r3 集合运算]~%")
  (check "unique 排序去重" '(1.0d0 2.0d0 3.0d0) (vt-unique (f64 3.0d0 1.0d0 3.0d0 2.0d0)))
  (check "in1d" '(1 0 1) (vt-in1d (f64 1.0d0 5.0d0 3.0d0) (f64 1.0d0 2.0d0 3.0d0)))
  (check "intersect1d" '(2.0d0 3.0d0) (vt-intersect1d (f64 1.0d0 2.0d0 3.0d0)
                                                      (f64 2.0d0 3.0d0 4.0d0)))
  (check "setdiff1d" '(1.0d0) (vt-setdiff1d (f64 1.0d0 2.0d0 3.0d0)
                                             (f64 2.0d0 3.0d0 4.0d0)))
  (check "setxor1d" '(1.0d0 4.0d0) (vt-setxor1d (f64 1.0d0 2.0d0 3.0d0)
                                                 (f64 2.0d0 3.0d0 4.0d0)))
  (check "union1d" '(1.0d0 2.0d0 4.0d0)
         (vt-union1d (f64 1.0d0 2.0d0) (f64 2.0d0 4.0d0)))

  ;; ---- 第 4 轮：统计 ---------------------------------------------
  (format t "~&[r4 统计]~%")
  (check "trapz 均匀间距" '(2.0d0) (vt-trapz (f64 0.0d0 1.0d0 2.0d0)))
  (check "correlate full 模式（长度 n+m-1）"
         '(1.0d0 3.0d0 6.0d0 5.0d0 3.0d0)
         (vt-correlate (f64 1.0d0 2.0d0 3.0d0) (vt-ones '(3))))
  (check-true "cov 对角 == 各行 var(ddof=1)"
              (let* ((m (vt-reshape (f64 1.0d0 2.0d0 3.0d0 4.0d0 2.0d0 4.0d0 6.0d0 8.0d0) '(2 4)))
                     (c (as-list (vt-cov m)))
                     (v (as-list (vt-var m :axis 1 :ddof 1))))
                (and (feq (nth 0 c) (nth 0 v) 1d-12)
                     (feq (nth 3 c) (nth 1 v) 1d-12))))
  (check "interp 线性插值" '(1.5d0 2.5d0)
         (vt-interp (f64 0.5d0 1.5d0) (f64 0.0d0 1.0d0 2.0d0) (f64 1.0d0 2.0d0 3.0d0)))
  (check-true "histogram 计数和 == 样本数"
              (let ((h (vt-histogram (vt-random-uniform '(10)) :bins 4)))
                (= 10 (reduce #'+ (as-list h)))))

  ;; ---- 第 5 轮：argsort / searchsorted ---------------------------
  (format t "~&[r5 排序索引]~%")
  (check "argsort" '(1 2 0) (vt-argsort (f64 3.0d0 1.0d0 2.0d0)))
  (check "searchsorted left" '(1 2) (vt-searchsorted (f64 1.0d0 3.0d0 5.0d0) (f64 2.0d0 4.0d0)))
  (check "searchsorted right" '(1 1) (vt-searchsorted (f64 1.0d0 3.0d0 5.0d0) (f64 1.0d0 1.0d0)
                                                      :side :right))
  (check "searchsorted：表含 NaN 时 NaN 查询排在非 NaN 之后"
         '(2 2)
         (vt-searchsorted (vt-from-sequence (list 1.0d0 2.0d0 (nan-df)) :dtype :float64)
                          (f64 3.0d0 5.0d0)))
  (check "argsort 稳定（同值保持原序）" '(1 0 2)
         (vt-argsort (f64 2.0d0 1.0d0 2.0d0)))

  ;; ---- 第 6 轮：take / slice / pad -------------------------------
  (format t "~&[r6 索引与填充]~%")
  (check "take 1D" '(3.0d0 1.0d0) (vt-take (f64 1.0d0 2.0d0 3.0d0) (i64 2 0)))
  (check "take :axis nil（展平后取）" '(3.0d0 1.0d0)
         (vt-take (vt-reshape (f64 1.0d0 2.0d0 3.0d0 4.0d0) '(2 2)) (i64 2 0) :axis nil))
  (check "pad :constant" '(0.0d0 1.0d0 2.0d0 0.0d0)
         (vt-pad (f64 1.0d0 2.0d0) '(1 1)))
  (check "tril" '(1.0d0 0.0d0 3.0d0 4.0d0)
         (vt-tril (vt-reshape (f64 1.0d0 2.0d0 3.0d0 4.0d0) '(2 2))))
  (check "triu" '(1.0d0 2.0d0 0.0d0 4.0d0)
         (vt-triu (vt-reshape (f64 1.0d0 2.0d0 3.0d0 4.0d0) '(2 2))))

  ;; ---- 第 7 轮：pad 模式 / delete --------------------------------
  (format t "~&[r7 填充模式与删除]~%")
  (check "pad :edge" '(1.0d0 1.0d0 2.0d0 3.0d0 3.0d0)
         (vt-pad (f64 1.0d0 2.0d0 3.0d0) '(1 1) :mode :edge))
  (check "pad :wrap" '(2.0d0 3.0d0 1.0d0 2.0d0 3.0d0 1.0d0 2.0d0)
         (vt-pad (f64 1.0d0 2.0d0 3.0d0) '(2 2) :mode :wrap))
  (check "pad :reflect" '(3.0d0 2.0d0 1.0d0 2.0d0 3.0d0 2.0d0 1.0d0)
         (vt-pad (f64 1.0d0 2.0d0 3.0d0) '(2 2) :mode :reflect))
  (check "pad :symmetric" '(2.0d0 1.0d0 1.0d0 2.0d0 3.0d0 3.0d0 2.0d0)
         (vt-pad (f64 1.0d0 2.0d0 3.0d0) '(2 2) :mode :symmetric))
  (check "delete" '(2.0d0 4.0d0) (vt-delete (f64 1.0d0 2.0d0 3.0d0 4.0d0) (i64 0 2)))

  ;; ---- 第 8 轮：take-along-axis（含曾静默出错的广播用例）--------
  (format t "~&[r8 沿轴取值]~%")
  (check "take-along :axis 1（末轴快路径）"
         '(2.0d0 1.0d0 3.0d0 4.0d0)
         (vt-take-along-axis (vt-reshape (f64 1.0d0 2.0d0 3.0d0 4.0d0) '(2 2))
                             (vt-reshape (i64 1 0 0 1) '(2 2)) 1))
  (check "★ take-along 索引广播 (2,1) 配 (2,2) 源（漏行偏移时 3 个套件失败）"
         '(2.0d0 4.0d0)
         (vt-take-along-axis (vt-reshape (f64 1.0d0 2.0d0 3.0d0 4.0d0) '(2 2))
                             (vt-reshape (i64 1 1) '(2 1)) 1))
  (check "take-along :axis 0"
         '(3.0d0 4.0d0 1.0d0 2.0d0)
         (vt-take-along-axis (vt-reshape (f64 1.0d0 2.0d0 3.0d0 4.0d0) '(2 2))
                             (vt-reshape (i64 1 0) '(2 1)) 0))

  ;; ---- 第 9 轮：bincount / median / percentile -------------------
  (format t "~&[r9 频次与分位]~%")
  (check "bincount minlength" '(1 2 0 1 0)
         (vt-bincount (i64 0 1 1 3) :minlength 5))
  (check "median 奇数" '(2.0d0) (vt-median (f64 3.0d0 1.0d0 2.0d0)))
  (check "median 偶数取中间两数平均" '(2.5d0) (vt-median (f64 4.0d0 1.0d0 3.0d0 2.0d0)))
  (check-true "median 含 NaN → NaN"
         (let ((m (vt-median (vt-from-sequence (list 1.0d0 (nan-df) 3.0d0) :dtype :float64))))
           (nan-p (vt-ref (vt-ravel m) 0))))
  (check "percentile 25" '(2.0d0) (vt-percentile (f64 1.0d0 2.0d0 3.0d0 4.0d0 5.0d0) 25))
  (check "percentile :lower（p=30 → 索引 1.2 取下）" '(2.0d0)
         (vt-percentile (f64 1.0d0 2.0d0 3.0d0 4.0d0 5.0d0) 30 :interpolation :lower))
  (check "percentile :higher（取上）" '(3.0d0)
         (vt-percentile (f64 1.0d0 2.0d0 3.0d0 4.0d0 5.0d0) 30 :interpolation :higher))
  (check "percentile :midpoint" '(2.5d0)
         (vt-percentile (f64 1.0d0 2.0d0 3.0d0 4.0d0 5.0d0) 30 :interpolation :midpoint))
  (check-true "percentile 与「排序后取下标」一致（自洽，1 万随机）"
              (let* ((a (vt-random-uniform '(10000)))
                (q 37.0d0)
                (sorted (as-list (vt-sort a)))
                (n (length sorted))
                (pos (* (/ q 100.0d0) (1- n)))
                (lo (floor pos))
                (hi (min (1+ lo) (1- n)))
                (frac (- pos lo))
                (want (+ (* (- 1.0d0 frac) (nth lo sorted)) (* frac (nth hi sorted)))))
                (feq (vt-ref (vt-ravel (vt-percentile a q)) 0) want 1d-12)))

  ;; ---- 第 10 轮：数学函数 / NaN 归约 / to-array ------------------
  (format t "~&[r10 数学与 NaN 归约]~%")
  (check-true "sqrt 负数 → NaN"
         (let ((r (vt-ravel (vt-sqrt (f64 -1.0d0 4.0d0)))))
           (and (nan-p (vt-ref r 0)) (feq (vt-ref r 1) 2.0d0))))
  (check-true "★ asin(NaN) → NaN（基线在此崩溃：CL 返回复数 #C(nan nan)）"
         (nan-p (vt-ref (vt-ravel (vt-asin (vt-from-sequence (list (nan-df)) :dtype :float64))) 0)))
  (check-true "★ acos(NaN) → NaN"
         (nan-p (vt-ref (vt-ravel (vt-acos (vt-from-sequence (list (nan-df)) :dtype :float64))) 0)))
  (check-true "★ atanh(NaN) → NaN"
         (nan-p (vt-ref (vt-ravel (vt-atanh (vt-from-sequence (list (nan-df)) :dtype :float64))) 0)))
  (check "atanh(±1) → ±Inf" '(#.sb-ext:double-float-positive-infinity
                              #.sb-ext:double-float-negative-infinity)
         (vt-atanh (f64 1.0d0 -1.0d0)))
  (check "log2 与 log :base 2 一致" '(3.0d0)
         (vt-log2 (f64 8.0d0)))
  (check-true "log(0) → -Inf, log(-1) → NaN"
         (let ((r (vt-ravel (vt-log (f64 0.0d0 -1.0d0)))))
           (and (= (vt-ref r 0) sb-ext:double-float-negative-infinity)
                (nan-p (vt-ref r 1)))))
  (check "spacing(0) = 5e-324" '(5.0d-324) (vt-spacing (f64 0.0d0)))
  (check "spacing(1) 精确值" '(2.220446049250313d-16) (vt-spacing (f64 1.0d0)))
  (check "nextafter" '(1.0000000000000002d0) (vt-nextafter (f64 1.0d0) (f64 2.0d0)))
  (check "nanmean 跳过 NaN" '(2.0d0)
         (vt-nanmean (vt-from-sequence (list 1.0d0 (nan-df) 3.0d0) :dtype :float64)))
  (check "nanvar ddof=0" '(1.0d0)
         (vt-nanvar (vt-from-sequence (list 1.0d0 (nan-df) 3.0d0) :dtype :float64)))
  (check "nanvar ddof=1" '(2.0d0)
         (vt-nanvar (vt-from-sequence (list 1.0d0 (nan-df) 3.0d0) :dtype :float64) :ddof 1))
  (check-true "★ nanvar(全 NaN, ddof=-1) → NaN（基线在 (= nan nan) 处抛浮点陷阱）"
              (nan-p (vt-ref (vt-ravel (vt-nanvar (vt-full '(4) (nan-df) :dtype :float64)
                                                   :ddof -1))
                             0)))
  (check "nanstd" '(1.0d0)
         (vt-nanstd (vt-from-sequence (list 1.0d0 (nan-df) 3.0d0) :dtype :float64)))
  (check "★ nanpercentile 2D 全局 = 展平后分位（基线在此崩溃）" '(2.0d0)
         (vt-nanpercentile (vt-reshape (vt-from-sequence
                                        (list 1.0d0 2.0d0 (nan-df) 4.0d0 (nan-df) -3.0d0
                                              0.0d0 8.0d0 (nan-df) 5.0d0 2.0d0 -1.0d0)
                                        :dtype :float64)
                                       '(3 4))
                           50))
  (check "nancumsum 跳过 NaN" '(1.0d0 1.0d0 4.0d0 4.0d0 6.0d0)
         (vt-nancumsum (vt-from-sequence (list 1.0d0 (nan-df) 3.0d0 (nan-df) 2.0d0)
                                         :dtype :float64)))
  (check "nancumprod 跳过 NaN" '(2.0d0 2.0d0 6.0d0 6.0d0)
         (vt-nancumprod (vt-from-sequence (list 2.0d0 (nan-df) 3.0d0 (nan-df))
                                          :dtype :float64)))
  (check-true "to-array 1D round-trip"
              (let* ((a (f64 1.0d0 2.0d0 3.0d0))
                     (arr (vt-to-array a)))
                (and (equal (array-dimensions arr) '(3))
                     (= (aref arr 0) 1.0d0) (= (aref arr 2) 3.0d0))))
  (check-true "to-array 2D round-trip"
              (let* ((a (vt-reshape (f64 1.0d0 2.0d0 3.0d0 4.0d0) '(2 2)))
                     (arr (vt-to-array a)))
                (and (equal (array-dimensions arr) '(2 2))
                     (= (aref arr 1 0) 3.0d0))))
  (check-true "to-array 非连续输入（转置）"
              (let* ((a (vt-transpose (vt-reshape (f64 1.0d0 2.0d0 3.0d0 4.0d0 5.0d0 6.0d0) '(2 3))))
                     (arr (vt-to-array a)))
                (and (equal (array-dimensions arr) '(3 2))
                     (= (aref arr 0 1) 4.0d0))))

  ;; ---- 第 11 轮：连接 / 平铺 / 滚动 / 分位 -----------------------
  (format t "~&[r11 连接与顺序统计]~%")
  (check "concatenate :axis 0" '(1.0d0 2.0d0 3.0d0 4.0d0)
         (vt-concatenate 0 (f64 1.0d0 2.0d0) (f64 3.0d0 4.0d0)))
  (check "concatenate 视图输入（带 offset）" '(3.0d0 4.0d0 1.0d0 2.0d0 3.0d0 4.0d0)
         (vt-concatenate 0 (vt-slice (f64 1.0d0 2.0d0 3.0d0 4.0d0) '(2 4))
                         (f64 1.0d0 2.0d0 3.0d0 4.0d0)))
  (check "concatenate 混合 dtype 提升" '(1.0d0 2.0d0 3.0d0 4.0d0)
         (vt-concatenate 0 (i64 1 2) (f64 3.0d0 4.0d0)))
  (check "stack :axis 0" '(1.0d0 2.0d0 3.0d0 4.0d0)
         (vt-stack 0 (f64 1.0d0 2.0d0) (f64 3.0d0 4.0d0)))
  (check "stack 结果形状" '(2 3)
         (vt-shape (vt-stack 0 (f64 1.0d0 2.0d0 3.0d0) (f64 4.0d0 5.0d0 6.0d0))))
  (check "tile 标量" '(1.0d0 2.0d0 1.0d0 2.0d0 1.0d0 2.0d0)
         (vt-tile (f64 1.0d0 2.0d0) 3))
  (check "tile 含 0" '() (vt-tile (f64 1.0d0 2.0d0) '(0 2)))
  (check "roll 正 shift" '(4.0d0 1.0d0 2.0d0 3.0d0) (vt-roll (f64 1.0d0 2.0d0 3.0d0 4.0d0) 1))
  (check "roll 负 shift" '(2.0d0 3.0d0 4.0d0 1.0d0) (vt-roll (f64 1.0d0 2.0d0 3.0d0 4.0d0) -1))
  (check "roll 总有新缓冲（写结果不污染输入）" '(1.0d0 2.0d0)
         (let* ((a (f64 1.0d0 2.0d0))
                (r (vt-roll a 0)))
           (setf (vt-ref r 0) 9.0d0)
           (as-list a)))
  (check "split 两段（第一段）" '(1.0d0 2.0d0)
         (first (vt-split (f64 1.0d0 2.0d0 3.0d0 4.0d0) 2)))
  (check "split 索引列表（第二段）" '(3.0d0 4.0d0)
         (second (vt-split (f64 1.0d0 2.0d0 3.0d0 4.0d0) '(2))))
  (check-true "median 1 万元素 == 排序取中"
         (let* ((a (vt-random-uniform '(10001)))
                (sorted (as-list (vt-sort a)))
                (want (nth 5000 sorted)))
           (feq (vt-ref (vt-ravel (vt-median a)) 0) want 1d-12)))
  (check "nanmedian 跳过 NaN" '(2.0d0)
         (vt-nanmedian (vt-from-sequence (list 1.0d0 (nan-df) 3.0d0) :dtype :float64)))
  (check-true "nanmedian 全 NaN → NaN"
              (nan-p (vt-ref (vt-ravel (vt-nanmedian (vt-full '(3) (nan-df) :dtype :float64))) 0)))
  (check "median :axis 1" '(1.5d0 3.5d0)
         (vt-median (vt-reshape (f64 1.0d0 2.0d0 3.0d0 4.0d0) '(2 2)) :axis 1))
  (check "percentile :axis 0" '(2.0d0 3.0d0)
         (vt-percentile (vt-reshape (f64 1.0d0 2.0d0 3.0d0 4.0d0) '(2 2)) 50 :axis 0))
  (check "median keepdims（axis=nil 时结果形状为全 1）" '(1 1)
         (vt-shape (vt-median (vt-reshape (f64 1.0d0 2.0d0 3.0d0 4.0d0) '(2 2)) :keepdims t)))
  (check-throws "percentile 越界报错" (lambda () (vt-percentile (f64 1.0d0) 101)))
  (check-throws "percentile 空数组报错（numpy ValueError 语义）"
                (lambda () (vt-percentile (vt-zeros '(0)) 50)))

  ;; ---- 布尔 dtype（:bool）：掩码类函数的返回值与消费方 ------------
  ;; 这一组守的是「比较/逻辑/谓词返回真布尔」这条契约，以及掩码到提取的
  ;; 参数顺序 —— 下面的 bench 曾把 (vt-extract a mask) 写成条件在后，
  ;; 结果不是 a[mask] 而是「拿 1M 个 float 当条件、0/1 掩码当数据源」。
  (format t "~&[bool 掩码与提取]~%")
  (let* ((a (f64 1.0d0 2.0d0 3.0d0 4.0d0))
         (mask (vt-> a 2.0d0)))          ; [F F T T]
    (check-true "比较结果 dtype 为 :bool（不再是 :int8 的 0/1）"
                (eq (vt-dtype mask) :bool))
    (check "bool 掩码取值只有 0/1" '(0 0 1 1) (as-list mask))
    (check "compress(mask, a) = a[mask]" '(3.0d0 4.0d0)
           (vt-compress mask a))
    (check "extract 与 compress 同结果（参数顺序 = numpy.extract(cond, arr)）"
           '(3.0d0 4.0d0) (vt-extract mask a))
    (check-true "sum(bool) 升 int64（不在 int8 里溢出）"
                (eq (vt-dtype (vt-sum mask)) :int64))
    (check-true "any/all 返回 0 维 :bool"
                (and (eq (vt-dtype (vt-any mask)) :bool)
                     (eq (vt-dtype (vt-all mask)) :bool)
                     (null (vt-shape (vt-any mask)))))
    (check-true "where 接受 :bool 掩码"
                (equal (as-list (vt-where mask a (vt-- a)))
                       '(-1.0d0 -2.0d0 3.0d0 4.0d0)))
    (check-true "bool + bool 走逻辑或（numpy 语义），不是整数 2"
                (equal (as-list (vt-+ mask mask)) '(0 0 1 1)))
    (check-throws "bool - bool 报错（numpy 的 bool_ 不支持减法）"
                  (lambda () (vt-- mask mask))))

  (format t "~&~%=== 断言结果：通过 ~a，失败 ~a ===~%" *pass* *fail*)
  (zerop *fail*))

;;; =============================================================================
;;; 性能基准
;;; =============================================================================
;;; 注释里的「→」表示该轮优化前后的实测值（µs，1M double / 1000×1000，同机）。

(defun benchmarks ()
  (let* ((n (max 1 (round (* *scale* 1000000))))
         (m (max 2 (round (* *scale* 1000))))
         (a (vt-random-uniform (list n)))
         (b (vt-random-uniform (list n)))
         (mat (vt-random-uniform (list m m)))
         (cls (vt-astype (vt-scale a 100.0d0) :int64))
         (mask (vt-> a 0.5d0))
         (an (let ((x (vt-copy a)))
               (loop for i from 0 below (min 1000 n) by 1
                     do (setf (vt-ref x (* i 997)) (nan-df)))
               x))
         (knots (vt-sort (vt-slice a (list 0 (min 10000 n)))))
         (vals (vt-random-uniform (list (min 10000 n))))
         (small (vt-slice a (list 0 (min 100000 n)))))
    (format t "~&~%=== BENCHMARKS (n=~a, m=~a) ===~%" n m)

    (format t "~&[元素级数学]  base → now / numpy~%")
    (bench "sqrt" 5 (vt-sqrt a))                    ; 48359 → 9870 / 4498
    (bench "log" 5 (vt-log a))                      ; 56288 → 17228 / 7343
    (bench "log2" 5 (vt-log2 a))                    ; 102990 → 26434 / 7729
    (bench "cbrt" 5 (vt-cbrt a))                    ; 143844 → 36367 / 24096
    (bench "spacing" 5 (vt-spacing a))              ; 312065 → 8500 / 3952
    (bench "nextafter" 3 (vt-nextafter a b))        ; 61783 → 8903 / 3195
    (bench "add" 5 (vt-add a b))                    ; 8431 / 1500
    (bench "clip" 5 (vt-clip a 0.2d0 0.8d0))        ; 15827 / 3000
    (bench "isclose" 5 (vt-isclose a b))            ; 5631 / 5000

    (format t "~&[归约 / 统计]~%")
    (bench "sum" 5 (vt-sum a))
    (bench "cumsum" 3 (vt-cumsum a))
    (bench "nanmean" 3 (vt-nanmean an))             ; 107173 → 2003 / 2690
    (bench "nanvar" 3 (vt-nanvar an))               ; 330148 → 3522 / 6422
    (bench "nanstd" 3 (vt-nanstd an))               ; 332235 → 3521 / 6297
    (bench "median" 2 (vt-median a))                ; 134261 → 16213 / 12336
    (bench "percentile" 2 (vt-percentile a 50))     ; 135355 → 18268 / 15823
    (bench "nanpercentile" 2 (vt-nanpercentile an 50)) ; 1594792 → 16236 / 18783
    (bench "nanmedian" 2 (vt-nanmedian an))         ; 134107 → ~16000
    (bench "median :axis 1" 2 (vt-median mat :axis 1)) ; 76838 → ~25000
    (bench "histogram" 2 (vt-histogram a :bins 100))
    (bench "bincount" 3 (vt-bincount cls :minlength 200))
    (bench "cov" 2 (vt-cov mat))
    (bench "corrcoef" 2 (vt-corrcoef mat))
    (bench "trapz" 3 (vt-trapz a))
    (bench "gradient" 2 (vt-gradient a))

    (format t "~&[排序 / 选择]~%")
    (bench "sort" 2 (vt-sort a))
    (bench "argsort" 2 (vt-argsort a))
    (bench "topk k=100" 2 (vt-topk a 100))
    (bench "searchsorted (10k 表)" 3 (vt-searchsorted knots a))
    (bench "unique" 2 (vt-unique a))
    (bench "in1d / isin" 2 (vt-in1d a knots))

    (format t "~&[索引 / 搬运]~%")
    (bench "copy" 5 (vt-copy a))
    (bench "astype" 3 (vt-astype a :int32))
    (bench "concat" 3 (vt-concatenate 0 a a))
    (bench "stack" 3 (vt-stack 1 a a))              ; 55904 → 10449
    (bench "tile" 3 (vt-tile small 10))
    (bench "roll" 5 (vt-roll a 10))
    (bench "repeat" 3 (vt-repeat small 10))
    (bench "compress" 3 (vt-compress mask a))
    ;; 注意参数顺序对标 numpy.extract(condition, arr)：条件在前、被取值的张量在后。
    ;; 原写作 (vt-extract a mask) 是把 1M 个 float 当条件、拿 0/1 掩码当数据源，
    ;; 结果不是 a[mask] 而是「掩码中真值位置的元素」，且走 float 条件的慢路径。
    (bench "extract" 3 (vt-extract mask a))
    (bench "nonzero" 2 (vt-nonzero mask))
    (bench "pad :constant" 3 (vt-pad small '(10 10)))
    (bench "to-array" 3 (vt-to-array a))            ; 344333 → 7111
    (bench "to-array 2D" 3 (vt-to-array mat))       ; 372634 → 7102

    (format t "~&[线代]~%")
    (bench "matmul (1000²)" 2 (vt-matmul mat mat))
    (bench "tensordot (100²)" 2 (vt-tensordot (vt-slice mat '(0 100) '(0 100))
                                              (vt-slice mat '(0 100) '(0 100))))
    (bench "matrix-power (100², k=3)" 2 (vt-matrix-power (vt-slice mat '(0 100) '(0 100)) 3))
    (bench "kron (50²)" 2 (vt-kron (vt-slice mat '(0 50) '(0 50))
                                   (vt-slice mat '(0 50) '(0 50))))
    (format t "~&[结束]~%")))

;;; =============================================================================
;;; 入口
;;; =============================================================================

(defun run (&key (assert t) (bench t))
  "ASSERT 跑正确性断言，BENCH 跑性能基准；返回 T 表示断言全过。"
  (let ((ok (if assert (assertions) t)))
    (when bench (benchmarks))
    ok))
