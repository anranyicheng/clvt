;;;; reduce-global-narrow-perf-test.lisp — 窄整型全归约走「塌轴」路径的回归测试
;;;;
;;;; 背景（v0.4.20）：`%CONTIG-AXIS-TEST` 原来以 `(consp axes)` 起头，把**全归约**
;;;; 排除在路径 2 之外，理由是「全归约走路径 1」。但路径 1 只服务 4 种存储级
;;;; dtype（`+KERNEL-LTS+`），于是 bool / int8 / uint8 / int16 / uint16 的全归约
;;;; 一律掉进路径 3 的通用里程计（`%KERNEL-SMALL-GENERAL`：逐元素 `(aref idx d)`
;;;; 取坐标 + 三次 `svref` 取步长 + 指针回卷），1M 元素实测 2.8~9.5 ms。
;;;;
;;;; 全归约的归约轴 [0, rank) 本来就「连成一段」，OUTER=1 / RED=size / INNER=1
;;;; 时 `%KERNEL-SINGLE-AXIS` 退化成一条扁平指针递增循环。放开之后同一份数据
;;;; 的代价与**秩无关**，也不再比存储级 dtype 慢一个量级。
;;;;
;;;; 因此本套件的核心断言不是「抄期望值」，而是四条**不变量**：
;;;;   A. 值等于一份独立重写的朴素参照（按定义在逻辑值列表上 reduce，
;;;;      含 int64 二进制回绕 —— numpy 的 prod 溢出就是补码回绕，不建模就对不上）；
;;;;   B. 同一逻辑数据的 6 种物理形态（连续、转置、翻转、负步长、步长 2、reshape）
;;;;      给出**逐位相同**的结果 —— 连续性只影响选路（§1 铁律 1）；
;;;;   C. dtype 契约：sum/prod/count-nonzero → int64，max/min → 输入 dtype，
;;;;      all/any → :bool（这条与路径无关，钉住它才能保证选路改动没顺手改语义）；
;;;;   D. 退化输入（空 / 单元素 / 0 维）的单位元与报错口径。
;;;;
;;;; 第二层是**机器无关的倍率护栏**（§16.16 第 7 条：不写绝对毫秒）：
;;;; 全归约一旦退回里程计，结果照样正确、A~D 照样全绿，只是慢 3~17 倍。
;;;; 所以就同一进程、同一台机器量四条比值：
;;;;   B1 秩不变性：2D/1D ≤ 2.0（改前 3.2×）、3D/1D ≤ 2.0（改前 3.3×）
;;;;   B2 窄整型 vs 存储级：int32 ÷ narrow-2D ≥ 0.4，即 narrow ≤ 2.5×（改前 6.2×）
;;;;   B3 全局 vs 沿轴：沿轴 ÷ 全局 ≥ 0.5，即全局 ≤ 2.0×（改前 24×）
;;;;   B4 同内核兄弟：sum ÷ amax ≥ 0.34，即 amax ≤ 3.0×（改前 2.6×，改动后 2.0×）

(require :asdf)
#+quicklisp (ql:quickload :clvt :silent t)
(asdf:load-system :clvt)

(defpackage :clvt-global-reduce-test
  (:use :cl :clvt)
  (:export #:run))

(in-package :clvt-global-reduce-test)

;;; ==================================================================
;;; 0. 断言与工具
;;; ==================================================================
(defvar *n* 0)
(defvar *pass* 0)
(defvar *fail* 0)
(defvar *failures* nil)

(defun check (label ok &optional detail)
  (incf *n*)
  (if ok
      (progn (incf *pass*) (format t "  [ OK ] ~a~%" label))
      (progn (incf *fail*) (push label *failures*)
             (format t "  ❌ ~a~@[ —— ~a~]~%" label detail))))

(defun check-eq (label expected actual)
  (check label (eqv expected actual)
         (format nil "期望 ~s，实得 ~s" expected actual)))

(defun check-at-most (label ratio limit what)
  "RATIO 必须 ≤ LIMIT（倍率护栏的统一形态：不写绝对毫秒，只写比值）。"
  (incf *n*)
  (if (and (numberp ratio) (<= ratio limit))
      (progn (incf *pass*)
             (format t "  [ OK ] ~a（~a = ~,2f× ≤ ~,1f×）~%" label what (float ratio) limit))
      (progn (incf *fail*) (push label *failures*)
             (format t "  ❌ ~a（~a = ~,2f× > ~,1f×）~%" label what (float ratio) limit))))

(defun nan-p (x)
  (and (floatp x) (sb-int:with-float-traps-masked (:invalid) (not (= x x)))))

(defun eqv (a b)
  "逐位比较：整数 EQL；浮点 EQL（−0.0 ≠ +0.0 会被抓出来）+ NaN 互认。"
  (cond ((and (floatp a) (floatp b))
         (cond ((and (nan-p a) (nan-p b)) t)
               ((or (nan-p a) (nan-p b)) nil)
               ((and (zerop a) (zerop b)) (eql (float-sign a) (float-sign b)))
               (t (eql a b))))
        ((and (null a) (null b)) t)
        ((or (null a) (null b)) nil)
        (t (eql a b))))

(defun as-num (x) (declare (inline)) (cond ((eq x t) 1) ((null x) 0) (t x)))

(defun logical-values (v)
  "按 C 序读出**逻辑值**列表（用 vt-ref 而不是 vt-to-list —— 后者对 :bool 默认
   给 1/0，前者给 T/NIL，正好能验证布尔视图的一致性，见 §17.8）。"
  (let ((f (vt-ravel v))
        (n (vt-size v)))
    (loop for i below n collect (vt-ref f i))))

(defun scalar-of (v) (vt-item v))

;;; ==================================================================
;;; 1. 独立参照：按定义在逻辑值上归约
;;; ==================================================================
(defun wrap-i64 (x)
  "numpy 的整型溢出是**补码回绕**：int16 的乘积在 int64 里溢出后按 2⁶⁴ 取模
   再按有符号解释（实测 `(vt-prod uint16-9元素)` = −5670685387920506880）。
   不建模这一步，参照实现就会在 prod 上对不上 —— 这是本库既有语义，原样钉住。"
  (let ((m (ldb (byte 64 0) x)))
    (if (logbitp 63 m) (- m (ash 1 64)) m)))

(defun ref-sum (nums) (wrap-i64 (reduce #'+ nums :initial-value 0)))
(defun ref-prod (nums) (wrap-i64 (reduce #'* nums :initial-value 1)))
(defun ref-max (nums) (reduce #'max nums))
(defun ref-min (nums) (reduce #'min nums))
(defun ref-all (nums) (every (lambda (x) (/= x 0)) nums))
(defun ref-any (nums) (some (lambda (x) (/= x 0)) nums))
(defun ref-count-nonzero (nums) (count-if (lambda (x) (/= x 0)) nums))

(defun ref-op (op nums)
  (ecase op
    (:sum (ref-sum nums)) (:prod (ref-prod nums))
    (:max (ref-max nums)) (:min (ref-min nums))
    (:all (ref-all nums)) (:any (ref-any nums))
    (:count-nonzero (ref-count-nonzero nums))))

(defun ref-value (op nums in-dtype)
  "参照值，且表示形态与实得一致：all/any 与「布尔数组的 max/min」都是 T/NIL。"
  (let ((v (ref-op op nums)))
    (case op
      ((:all :any) (if v t nil))
      ((:max :min) (if (eq in-dtype :bool) (if (= v 1) t nil) v))
      (t v))))

(defun expected-dtype (op in-dtype)
  "dtype 契约（与执行路径无关）：整型（含 bool 与 4 种窄整型）的 sum/prod
   提升到 int64（numpy 在 64 位平台上同样如此），浮点保持原宽度；
   count-nonzero 恒 int64；max/min 保持输入 dtype；all/any 恒 :bool。"
  (ecase op
    ((:sum :prod)
     (if (member in-dtype '(:bool :int8 :uint8 :int16 :uint16 :int32 :int64))
         :int64
         in-dtype))
    (:count-nonzero :int64)
    ((:max :min) in-dtype)
    ((:all :any) :bool)))

;;; ==================================================================
;;; 2. 数据构造
;;; ==================================================================
(defun make-flat (dtype n &optional (seed 1))
  (loop for i below n
        collect (let ((v (mod (* (+ i seed) 37) 97)))
                  (case dtype
                    (:bool    (oddp v))
                    (:int8    (- v 48))
                    (:uint8   v)
                    (:int16   (- (* v 300) 14000))
                    (:uint16  (* v 300))
                    (:int32   (- (* v 100000) 4000000))
                    (:int64   (- (* v 100000000) 4000000000))
                    (:float32 (coerce (/ (- v 48) 7.0) 'single-float))
                    (:float64 (coerce (/ (- v 48) 7.0d0) 'double-float))))))

(defun make-tensor (dtype shape &optional (seed 1))
  (let* ((n (reduce #'* shape :initial-value 1))
         (v (vt-from-sequence (make-flat dtype n seed) :dtype dtype)))
    (if (equal shape (list n)) v (vt-reshape v shape))))

(defun call-op (op v)
  (ecase op
    (:sum (vt-sum v)) (:prod (vt-prod v))
    (:max (vt-amax v)) (:min (vt-amin v))
    (:all (vt-all v)) (:any (vt-any v))
    (:count-nonzero (vt-count-nonzero v))))

(defparameter *narrow-dtypes* '(:bool :int8 :uint8 :int16 :uint16))
(defparameter *control-dtypes* '(:int32 :int64 :float32 :float64))
(defparameter *ops* '(:sum :prod :max :min :all :any :count-nonzero))
(defparameter *shapes* '((9) (3 4) (2 3 5)))

;;; ==================================================================
;;; 3. A 段：值 ≡ 朴素参照（dtype × 形状 × 算子）
;;; ==================================================================
(defun t-values ()
  (format t "~%=== A1 值 ≡ 朴素参照（窄整型 5 dtype × 3 形状 × 7 算子）===~%")
  (dolist (dt *narrow-dtypes*)
    (dolist (shape *shapes*)
      (let* ((v (make-tensor dt shape))
             (nums (mapcar #'as-num (logical-values v))))
        (dolist (op *ops*)
          (let* ((res (call-op op v))
                 (want (ref-value op nums dt))
                 (got (scalar-of res)))
            (check (format nil "~(~a~) ~s ~(~a~) 值" dt shape op)
                   (eqv want got) (format nil "期望 ~s，实得 ~s" want got))
            (check (format nil "~(~a~) ~s ~(~a~) dtype/shape" dt shape op)
                   (and (eq (vt-dtype res) (expected-dtype op dt))
                        (null (vt-shape res))
                        (= 1 (vt-size res)))
                   (format nil "期望 dtype=~a shape=NIL，实得 dtype=~a shape=~s"
                           (expected-dtype op dt) (vt-dtype res) (vt-shape res)))))))))

(defun t-values-control ()
  (format t "~%=== A2 存储级 dtype 不得因本轮改动而变（对照组）===~%")
  (dolist (dt *control-dtypes*)
    (dolist (shape '((9) (2 3 5)))
      (let* ((v (make-tensor dt shape))
             (nums (mapcar #'as-num (logical-values v))))
        (dolist (op *ops*)
          (let ((res (call-op op v)))
            (check (format nil "~(~a~) ~s ~(~a~) dtype" dt shape op)
                   (eq (vt-dtype res) (expected-dtype op dt))
                   (format nil "期望 ~a，实得 ~a" (expected-dtype op dt) (vt-dtype res)))
            ;; 浮点的 sum/prod 有累加顺序契约（§7.3），不与「列表 reduce」比；
            ;; max/min/all/any/count-nonzero 与顺序无关，逐位比。
            (unless (and (member dt '(:float32 :float64)) (member op '(:sum :prod)))
              (let ((want (ref-value op nums dt))
                    (got (scalar-of res)))
                (check (format nil "~(~a~) ~s ~(~a~) 值" dt shape op)
                       (eqv want got) (format nil "期望 ~s，实得 ~s" want got))))))))))

;;; ==================================================================
;;; 4. B 段：物理形态无关（同一逻辑数据 6 种视图 → 逐位相同）
;;; ==================================================================
(defun views-of (v)
  "6 种**元素集合相同**的物理形态（归约算子全部对元素顺序不敏感，所以只比
   聚合结果；若某个视图漏掉/重复了元素，sum/count 立刻对不上）：
     0 原样（连续，(6 5)）
     1 转置 → (5 6)，两个 stride 互换
     2 翻转轴 0 → (6 5)，负 stride
     3 双轴全反 → (6 5)，两条负 stride
     4 reshape 成 (5 6) 再转置 → (6 5)，stride 为 (1 6)（与 0 的 (5 1) 不同）
     5 reshape 成 (2 3 5) 再置换 `(2 0 1)` → (5 2 3)，三维非连续"
  (list v
        (vt-transpose v)
        (vt-flip v :axis 0)
        (vt-slice v '(5 nil -1) '(4 nil -1))
        (vt-transpose (vt-reshape v '(5 6)))
        (vt-transpose (vt-reshape v '(2 3 5)) '(2 0 1))))

(defun t-physical ()
  (format t "~%=== A3 物理形态无关性（6 种视图 × 5 窄 dtype × 7 算子）===~%")
  (dolist (dt *narrow-dtypes*)
    (let* ((base (make-tensor dt '(6 5)))
           (views (views-of base))
           (nums (mapcar #'as-num (logical-values base))))
      (dolist (op *ops*)
        (let ((ref (call-op op base)))
          (check (format nil "~(~a~) ~(~a~) 连续形态 ≡ 参照" dt op)
                 (eqv (ref-value op nums dt) (scalar-of ref))
                 (format nil "期望 ~s，实得 ~s" (ref-value op nums dt) (scalar-of ref)))
          (loop for vw in views for k from 0 do
            (let ((got (call-op op vw)))
              (check (format nil "~(~a~) ~(~a~) 视图~a（shape ~s）逐位相同"
                             dt op k (vt-shape vw))
                     (and (eqv (scalar-of ref) (scalar-of got))
                          (eq (vt-dtype ref) (vt-dtype got)))
                     (format nil "连续 ~s（~a）vs 视图 ~s（~a）"
                             (scalar-of ref) (vt-dtype ref)
                             (scalar-of got) (vt-dtype got))))))))))

;;; ==================================================================
;;; 5. C 段：退化输入
;;; ==================================================================
(defun t-degenerate ()
  (format t "~%=== A4 退化输入（空 / 单元素 / 0 维）===~%")
  (dolist (dt *narrow-dtypes*)
    (let ((empty  (vt-reshape (vt-from-sequence nil :dtype dt) '(0)))
          (single (vt-from-sequence (list (if (eq dt :bool) t 5)) :dtype dt))
          (zero-d (vt-reshape (vt-from-sequence (list (if (eq dt :bool) nil 0)) :dtype dt)
                              '())))
      (check (format nil "~(~a~) 0 维构造 shape" dt) (null (vt-shape zero-d))
             (format nil "实得 ~s" (vt-shape zero-d)))
      ;; 空：有单位元的算子给单位元（numpy 语义）
      (dolist (pair '((:sum . 0) (:prod . 1) (:all . t) (:any . nil)))
        (let ((got (scalar-of (call-op (car pair) empty))))
          (check (format nil "~(~a~) 空 ~(~a~) 单位元" dt (car pair))
                 (eqv (cdr pair) got) (format nil "期望 ~s，实得 ~s" (cdr pair) got))
          (check (format nil "~(~a~) 空 ~(~a~) dtype" dt (car pair))
                 (eq (vt-dtype (call-op (car pair) empty))
                     (expected-dtype (car pair) dt))
                 (format nil "实得 ~a" (vt-dtype (call-op (car pair) empty))))))
      ;; 空：无单位元的算子必须报错（numpy 语义）
      (dolist (op '(:max :min))
        (check (format nil "~(~a~) 空 ~(~a~) 报错" dt op)
               (handler-case (progn (call-op op empty) nil) (error () t))))
      ;; 单元素 / 0 维
      (dolist (op *ops*)
        (let ((v1 (scalar-of (call-op op single)))
              (v0 (scalar-of (call-op op zero-d)))
              (want1 (ref-value op (list (as-num (vt-ref single 0))) dt))
              (want0 (ref-value op (list (as-num (vt-ref zero-d))) dt)))
          (check (format nil "~(~a~) 单元素 ~(~a~)" dt op)
                 (eqv want1 v1) (format nil "期望 ~s，实得 ~s" want1 v1))
          (check (format nil "~(~a~) 0 维 ~(~a~)" dt op)
                 (eqv want0 v0) (format nil "期望 ~s，实得 ~s" want0 v0)))))))

;;; ==================================================================
;;; 6. D 段：机器无关的倍率护栏
;;; ==================================================================
(defun %calibrate (thunk)
  (let ((inner 1))
    (loop
      (let ((t0 (get-internal-real-time)))
        (loop repeat inner do (funcall thunk))
        (let ((dt (- (get-internal-real-time) t0)))
          (cond ((>= dt 20000) (return inner))
                ((zerop dt) (setf inner (* inner 8)))
                (t (setf inner (max (1+ inner)
                                    (ceiling (* inner 30000) dt))))))))))

(defun min-us (thunk &key (reps 5))
  (funcall thunk)
  (let* ((inner (%calibrate thunk))
         (best most-positive-single-float))
    (loop repeat reps
          do (let ((t0 (get-internal-real-time)))
               (loop repeat inner do (funcall thunk))
               (let ((us (/ (float (- (get-internal-real-time) t0)) inner)))
                 (when (< us best) (setf best us)))))
    best))

(defun ns-per-el (thunk n &key (reps 5))
  (/ (* 1000.0 (min-us thunk :reps reps)) n))

(defparameter *big* 1000000)

(defun t-ratios ()
  (format t "~%=== B 段 机器无关倍率护栏（1M 元素，同进程同机）===~%")
  (let* ((b1 (vt-astype (vt-arange *big* :start 0 :step 1 :dtype :int64) :bool))
         (b2 (vt-reshape b1 (list 1000 1000)))
         (b3 (vt-reshape b1 (list 100 100 100)))
         (i8-1 (vt-astype (vt-arange *big* :start 0 :step 1 :dtype :int64) :int8))
         (i8-2 (vt-reshape i8-1 (list 1000 1000)))
         (i32-2 (vt-reshape (vt-astype (vt-arange *big* :start 0 :step 1 :dtype :int64) :int32)
                            (list 1000 1000))))
    (progn (setf (vt-ref b1 0) t) nil)
    ;; B1 秩不变性：塌轴之后 OUTER/RED/INNER 与秩无关
    (let* ((t1 (ns-per-el (lambda () (vt-sum b1)) *big*))
           (t2 (ns-per-el (lambda () (vt-sum b2)) *big*))
           (t3 (ns-per-el (lambda () (vt-sum b3)) *big*)))
      (check-at-most "B1a 秩不变性 sum(bool) 2D/1D" (/ t2 t1) 2.0 "2D/1D")
      (check-at-most "B1b 秩不变性 sum(bool) 3D/1D" (/ t3 t1) 2.0 "3D/1D"))
    ;; B2 窄整型不得比存储级（路径 1 的扁平循环）慢一个量级
    (let ((tn (ns-per-el (lambda () (vt-sum i8-2)) *big*))
          (ts (ns-per-el (lambda () (vt-sum i32-2)) *big*)))
      (check-at-most "B2 窄整型 2D ÷ int32 2D" (/ tn ts) 2.5 "narrow/int32"))
    ;; B3 全局与沿轴同源：同一条扁平扫描
    (let ((tn (ns-per-el (lambda () (vt-all b2)) *big*))
          (ta (ns-per-el (lambda () (vt-all b2 :axis 0)) *big*)))
      (check-at-most "B3 全局 ÷ 沿轴 all(bool)" (/ tn ta) 2.0 "全局/沿轴"))
    ;; B4 同一条内核的兄弟算子（max 多一条分支，不该差一个量级）
    (let ((ts (ns-per-el (lambda () (vt-sum i8-1)) *big*))
          (tm (ns-per-el (lambda () (vt-amax i8-1)) *big*)))
      (check-at-most "B4 amax(int8) ÷ sum(int8)" (/ tm ts) 3.0 "amax/sum"))))

;;; ==================================================================
;;; 主流程
;;; ==================================================================
(defun run ()
  (format t "~%============================================================~%")
  (format t "  窄整型全归约「塌轴」路径 回归测试（v0.4.20）~%")
  (format t "============================================================~%")
  (t-values)
  (t-values-control)
  (t-physical)
  (t-degenerate)
  (t-ratios)
  (format t "~%============================================================~%")
  (format t "  Total: ~a | Pass: ~a | Fail: ~a | Skip: 0~%" *n* *pass* *fail*)
  (format t "============================================================~%")
  (when *failures*
    (format t "~%Failed:~{~%  - ~a~}~%" (reverse *failures*)))
  (zerop *fail*))

(sb-ext:exit :code (if (run) 0 1))
