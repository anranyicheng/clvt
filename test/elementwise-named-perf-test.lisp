;;;; elementwise-named-perf-test.lisp — 逐元素算子「具名内联」一轮优化的回归测试
;;;;
;;;; 背景（v0.4.20）：`vt-fast-map` 只对 `(function <符号>)` 形式的内联，
;;;; 传 `(lambda (x) …)` 会直接 `(apply #'vt-map …)` 回退成「逐元素 funcall +
;;;; 泛型 aref + 逐元素 VT-CAST」。取整族 / 谓词族 / hypot / cbrt 早年都已
;;;; 按这条规律改过（见 elementwise1/2.lisp 的注释与 CHANGELOG 五十七），
;;;; 但有八个算子一直漏着：
;;;;
;;;;   elementwise1.lisp  vt-rint
;;;;   elementwise2.lisp  vt-sinc / vt-reciprocal / vt-deg2rad / vt-rad2deg / vt-lerp
;;;;   extensions3.lisp   vt-copysign / vt-logaddexp
;;;;
;;;; 本轮只把算子提成**具名 + inline** 的函数（算子体逐字不动），交给
;;;; `vt-fast-map` 编进元素类型特化循环。改的是执行层，逻辑层一字未动。
;;;;
;;;; 所以本套件的核心断言是**逐位等价**：新实现 ≡ 一份逐字复刻改动前的实现
;;;; （`old-*`，仍然走 `vt-map` 的闭包路径），比对 shape / dtype / 每个元素，
;;;; 覆盖 ±0.0 的符号位、NaN、±Inf、整数输入、float32、非连续视图、:out 与广播。
;;;; 不写手抄期望值 —— 抄期望值只能证明「现在是这样」，
;;;; 挡不住「改对了主体、改错了 −0.0 或 NaN」这类偏移，而本轮踩的正好是符号位。
;;;;
;;;; 第二层是**机器无关的倍率护栏**（§16.16 第 7 条：不写绝对毫秒）：
;;;; 一旦有人把具名算子改回 lambda，结果照样正确、A 段全绿，只是慢 2~11 倍。
;;;; 所以每个算子再配一条「old ÷ new」的比值下界，两边同进程同机跑。

(require :asdf)
#+quicklisp (ql:quickload :clvt :silent t)
(asdf:load-system :clvt)

(defpackage :clvt-ew-named-test
  (:use :cl :clvt)
  (:export #:run))

(in-package :clvt-ew-named-test)

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

(defun check-at-least (label ratio limit what)
  (incf *n*)
  (if (and (numberp ratio) (>= ratio limit))
      (progn (incf *pass*)
             (format t "  [ OK ] ~a（~a = ~,2f× ≥ ~,1f×）~%" label what (float ratio) limit))
      (progn (incf *fail*) (push label *failures*)
             (format t "  ❌ ~a（~a = ~,2f× < ~,1f×）~%" label what (float ratio) limit))))

(defun nan-p (x)
  (and (floatp x) (sb-int:with-float-traps-masked (:invalid) (not (= x x)))))

(defun eqv (a b)
  "逐位比较：浮点 EQL（−0.0 ≠ +0.0 会被抓出来）+ NaN 互认；整数 EQL；
   复数递归；T/NIL 相等。"
  (cond ((and (complexp a) (complexp b))
         (and (eqv (realpart a) (realpart b)) (eqv (imagpart a) (imagpart b))))
        ((or (complexp a) (complexp b)) nil)
        ((and (floatp a) (floatp b))
         (cond ((and (nan-p a) (nan-p b)) t)
               ((or (nan-p a) (nan-p b)) nil)
               ((and (zerop a) (zerop b)) (eql (float-sign a) (float-sign b)))
               (t (eql a b))))
        ((and (null a) (null b)) t)
        ((or (null a) (null b)) nil)
        (t (eql a b))))

(defun flat (v)
  "按 C 序读出逻辑值（不经 vt-to-list，避免它对 :bool 的 1/0 折叠）。"
  (let ((f (vt-ravel v)) (n (vt-size v)))
    (loop for i below n collect (vt-ref f i))))

(defun capture (thunk)
  "跑一个 thunk，返回 (:ok v) 或 (:error 类型名)。允许「两边都报错」的比较。"
  (handler-case (list :ok (funcall thunk))
    (error (e) (list :error (type-of e)))))

(defun check-same-as-old (label new-thunk old-thunk)
  "新实现与改动前实现必须逐位相同（含报错与否、shape、dtype、每个元素）。"
  (let ((a (capture new-thunk))
        (b (capture old-thunk)))
    (cond
      ((and (eq (first a) :error) (eq (first b) :error))
       (check (format nil "~a（两边都报错，类型一致 ~a）" label (second a))
              (eq (second a) (second b))
              (format nil "新 ~a vs 旧 ~a" (second a) (second b))))
      ((eq (first a) :error)
       (check (format nil "~a（新实现报错，旧实现没有）" label) nil
              (format nil "~a" (second a))))
      ((eq (first b) :error)
       (check (format nil "~a（旧实现报错，新实现没有）" label) nil
              (format nil "~a" (second b))))
      (t
       (let* ((nv (second a)) (ov (second b))
              (nl (flat nv)) (ol (flat ov)))
         (check (format nil "~a" label)
                (and (equal (vt-shape nv) (vt-shape ov))
                     (eq (vt-dtype nv) (vt-dtype ov))
                     (= (length nl) (length ol))
                     (every #'eqv nl ol))
                (format nil "新 shape=~s dtype=~a ~s / 旧 shape=~s dtype=~a ~s"
                        (vt-shape nv) (vt-dtype nv) (subseq nl 0 (min 6 (length nl)))
                        (vt-shape ov) (vt-dtype ov) (subseq ol 0 (min 6 (length ol))))))))))

;;; ==================================================================
;;; 1. 改动前的实现（逐字复刻，作为「旧」基准）
;;; ==================================================================
(defun old-rint (vt &key out dtype)
  (vt-map (lambda (x)
            (if (and (floatp x) (clvt::%nan-or-inf-p x))
                x
                (let ((res (nth-value 0 (round x))))
                  (if (floatp x)
                      (if (and (zerop res) (minusp (float-sign x)))
                          (- (float 0.0 x))
                          (float res x))
                      res))))
          vt :out out :dtype dtype))

(defun old-sinc (tensor &key out dtype)
  (let* ((dt (clvt::%infer-float-dtype tensor dtype))
         (x-pi (vt-scale tensor pi :dtype dt)))
    (vt-map (lambda (x) (if (zerop x) 1.0d0 (/ (sin x) x))) x-pi :out out :dtype dt)))

(defun old-reciprocal (vt &key out dtype)
  (let* ((in-dt (vt-dtype (clvt::ensure-vt vt)))
         (dt (or dtype (if (eq in-dt :bool) :int8 in-dt))))
    (if (clvt::vt-int-dtype-p dt)
        (vt-map (lambda (v) (if (zerop v)
                                (error "vt-reciprocal: 整数零除")
                                (truncate (/ 1 v))))
                vt :out out :dtype dt)
        (let ((one (if (eq dt :float32) 1.0s0 1.0d0)))
          (vt-map (lambda (v) (/ one v)) vt :out out :dtype dt)))))

(defun old-deg2rad (vt &key out dtype)
  (let* ((dt (clvt::%infer-float-dtype vt dtype))
         (factor (if (eq dt :float32) (/ (coerce pi 'single-float) 180.0s0)
                     (/ pi 180.0d0))))
    (vt-map (lambda (x) (* x factor)) vt :out out :dtype dt)))

(defun old-rad2deg (vt &key out dtype)
  (let* ((dt (clvt::%infer-float-dtype vt dtype))
         (factor (if (eq dt :float32) (/ 180.0s0 (coerce pi 'single-float))
                     (/ 180.0d0 pi))))
    (vt-map (lambda (x) (* x factor)) vt :out out :dtype dt)))

(defun old-copysign (x y &key out dtype)
  (let* ((xo (clvt::ensure-vt x)) (yo (clvt::ensure-vt y))
         (xe (clvt::%coerce-float-input xo :float64))
         (ye (clvt::%coerce-float-input yo :float64)))
    (vt-map (lambda (a b)
              (let ((af (coerce a 'double-float)) (bf (coerce b 'double-float)))
                (if (clvt::%e3-sign-bit-p bf) (- (abs af)) (abs af))))
            xe ye :out out
            :dtype (clvt::%e3-double-result-dtype dtype x xo y yo))))

(defun old-logaddexp (x y &key out dtype)
  (let* ((xo (clvt::ensure-vt x)) (yo (clvt::ensure-vt y))
         (xe (clvt::%coerce-float-input xo :float64))
         (ye (clvt::%coerce-float-input yo :float64)))
    (vt-map (lambda (a b)
              (let ((af (coerce a 'double-float)) (bf (coerce b 'double-float)))
                (cond ((clvt::%nan-p af) af)
                      ((clvt::%nan-p bf) bf)
                      ((clvt::%pos-inf-p af) af)
                      ((clvt::%pos-inf-p bf) bf)
                      ((clvt::%neg-inf-p af) bf)
                      ((clvt::%neg-inf-p bf) af)
                      (t (let ((m (max af bf)))
                           (+ m (log (+ 1.0d0 (exp (- (min af bf) m))))))))))
            xe ye :out out
            :dtype (clvt::%e3-double-result-dtype dtype x xo y yo))))

(defun old-lerp (start end weight &key out dtype)
  (vt-map (lambda (s e w) (+ s (* (- e s) w)))
          (clvt::ensure-vt start) (clvt::ensure-vt end) (clvt::ensure-vt weight)
          :dtype dtype :out out))

;;; ==================================================================
;;; 2. 测试语料
;;; ==================================================================
(defun vec (values dtype)
  (vt-from-sequence values :dtype dtype))

(defparameter *special*
  (list 0.0d0 -0.0d0 1.0d0 -1.0d0 0.5d0 -0.5d0 1.5d0 -1.5d0 2.5d0 -2.5d0
        0.49999999999999994d0 1.0d-300 -1.0d-300 1.0d300 -1.0d300
        (clvt:vt-get-pos-inf :float64) (clvt:vt-get-neg-inf :float64)
        (clvt:vt-get-nan :float64) 180.0d0 -90.0d0 123.456d0 -123.456d0))

(defparameter *special2*
  (list 1.0d0 -2.0d0 3.0d0 -4.0d0 0.5d0 -0.5d0 1.0d0 -1.0d0 2.0d0 -2.0d0
        0.25d0 2.0d-300 -2.0d-300 2.0d300 -2.0d300
        (clvt:vt-get-neg-inf :float64) (clvt:vt-get-pos-inf :float64)
        (clvt:vt-get-nan :float64) 0.0d0 -0.0d0 3.75d0 -3.75d0))

(defun f32 (values)
  (mapcar (lambda (x) (if (or (nan-p x) (and (floatp x) (> (abs x) most-positive-single-float)))
                          (cond ((nan-p x) (clvt:vt-get-nan :float32))
                                ((plusp x) (clvt:vt-get-pos-inf :float32))
                                (t (clvt:vt-get-neg-inf :float32)))
                          (coerce x 'single-float)))
          values))

(defun corpus-1d ()
  "一元算子的语料：f64 特殊值 / f32 / 整数 / 转置视图 / 广播后的 2D。"
  (list (cons "f64 特殊值" (vec *special* :float64))
        (cons "f32 特殊值" (vec (f32 *special*) :float32))
        (cons "单元素" (vec (list 2.5d0) :float64))
        (cons "0 维" (vt-reshape (vec (list -0.0d0) :float64) '()))
        (cons "转置视图" (vt-transpose (vt-reshape (vec *special* :float64) '(11 2))))
        (cons "翻转视图" (vt-flip (vec *special* :float64) :axis 0))
        (cons "空" (vt-reshape (vec nil :float64) '(0)))))

;;; ==================================================================
;;; 3. A 段：新实现 ≡ 改动前实现（逐位）
;;; ==================================================================
(defun t-rint ()
  (format t "~%=== A1 vt-rint ≡ 改动前实现 ===~%")
  (dolist (pair (corpus-1d))
    (check-same-as-old (format nil "vt-rint ~a" (car pair))
                       (lambda () (vt-rint (cdr pair)))
                       (lambda () (old-rint (cdr pair)))))
  ;; 整数输入：必须保持整数 dtype（本库有意偏差，numpy 是 float64）
  (dolist (dt '(:int8 :int16 :uint8 :uint16 :int32 :int64))
    (let ((v (vt-astype (vt-arange 9 :start -4 :step 1 :dtype :int64) dt)))
      (check-same-as-old (format nil "vt-rint ~(~a~) 输入" dt)
                         (lambda () (vt-rint v))
                         (lambda () (old-rint v)))
      (check (format nil "vt-rint ~(~a~) 保持整数 dtype" dt)
             (eq (vt-dtype (vt-rint v)) dt)
             (format nil "实得 ~a" (vt-dtype (vt-rint v))))))
  ;; :dtype 覆盖
  (check-same-as-old "vt-rint :dtype :float32"
                     (lambda () (vt-rint (vec *special* :float64) :dtype :float32))
                     (lambda () (old-rint (vec *special* :float64) :dtype :float32)))
  ;; :out
  (check-same-as-old "vt-rint :out"
                     (lambda () (let ((o (vt-zeros (list (length *special*)) :dtype :float64)))
                                  (vt-rint (vec *special* :float64) :out o)))
                     (lambda () (let ((o (vt-zeros (list (length *special*)) :dtype :float64)))
                                  (old-rint (vec *special* :float64) :out o))))
  ;; rint 的定义性质：最近整数、平局取偶、符号零保留
  (let ((v (vec (list 0.5d0 1.5d0 2.5d0 -0.5d0 -1.5d0 -2.5d0 3.5d0) :float64))
        (want (list 0.0d0 2.0d0 2.0d0 -0.0d0 -2.0d0 -2.0d0 4.0d0)))
    (check "vt-rint 平局取偶（含 −0.0 符号位）"
           (equal want (flat (vt-rint v)))
           (format nil "期望 ~s，实得 ~s" want (flat (vt-rint v))))))

(defun t-unary (label new-fn old-fn)
  (format t "~%=== ~a ≡ 改动前实现 ===~%" label)
  (dolist (pair (corpus-1d))
    (check-same-as-old (format nil "~a ~a" label (car pair))
                       (lambda () (funcall new-fn (cdr pair)))
                       (lambda () (funcall old-fn (cdr pair)))))
  (check-same-as-old (format nil "~a :dtype :float32" label)
                     (lambda () (funcall new-fn (vec *special* :float64) :dtype :float32))
                     (lambda () (funcall old-fn (vec *special* :float64) :dtype :float32)))
  (check-same-as-old (format nil "~a :out" label)
                     (lambda () (funcall new-fn (vec *special* :float64)
                                         :out (vt-zeros (list (length *special*)) :dtype :float64)))
                     (lambda () (funcall old-fn (vec *special* :float64)
                                         :out (vt-zeros (list (length *special*)) :dtype :float64)))))

(defun t-reciprocal ()
  (format t "~%=== A3 vt-reciprocal ≡ 改动前实现 ===~%")
  (dolist (pair (corpus-1d))
    (check-same-as-old (format nil "vt-reciprocal ~a" (car pair))
                       (lambda () (vt-reciprocal (cdr pair)))
                       (lambda () (old-reciprocal (cdr pair)))))
  ;; 整数：保持整数 dtype，零除报错（numpy 给 0 + Warning，本库拒绝静默错值）
  (dolist (dt '(:int8 :int16 :uint8 :uint16 :int32 :int64))
    (let ((v (vt-astype (vt-arange 6 :start 1 :step 1 :dtype :int64) dt)))
      (check-same-as-old (format nil "vt-reciprocal ~(~a~) 输入" dt)
                         (lambda () (vt-reciprocal v))
                         (lambda () (old-reciprocal v)))
      (check (format nil "vt-reciprocal ~(~a~) 保持整数 dtype" dt)
             (eq (vt-dtype (vt-reciprocal v)) dt)
             (format nil "实得 ~a" (vt-dtype (vt-reciprocal v))))))
  (check "vt-reciprocal 整数零除报错"
         (handler-case (progn (vt-reciprocal (vec (list 0) :int64)) nil) (error () t)))
  ;; 布尔 → :int8（§17.10.13）。注意 NIL 会变成 0 → 整数零除报错，
  ;; 这是既有语义（numpy 给 0 + RuntimeWarning，本库拒绝静默错值）：
  ;; dtype 断言用全 T 的数组，零除契约单独断言。
  (let ((b (vt-from-sequence (list t t t) :dtype :bool))
        (bz (vt-from-sequence (list t nil t) :dtype :bool)))
    (check "vt-reciprocal :bool → :int8"
           (eq (vt-dtype (vt-reciprocal b)) :int8)
           (format nil "实得 ~a" (vt-dtype (vt-reciprocal b))))
    (check-same-as-old "vt-reciprocal :bool 值"
                       (lambda () (vt-reciprocal b))
                       (lambda () (old-reciprocal b)))
    (check "vt-reciprocal :bool 含 NIL → 整数零除报错"
           (and (handler-case (progn (vt-reciprocal bz) nil) (error () t))
                (handler-case (progn (old-reciprocal bz) nil) (error () t)))))
  ;; 1/x 的定义性质（浮点）：含 ±0.0 的 ±Inf
  (let ((got (flat (vt-reciprocal (vec (list 2.0d0 -2.0d0 0.5d0 0.0d0 -0.0d0) :float64)))))
    (check "vt-reciprocal 浮点 1/x（含 ±0.0 → ±Inf）"
           (and (eqv 0.5d0 (nth 0 got)) (eqv -0.5d0 (nth 1 got))
                (eqv 2.0d0 (nth 2 got))
                (eqv (clvt:vt-get-pos-inf :float64) (nth 3 got))
                (eqv (clvt:vt-get-neg-inf :float64) (nth 4 got)))
           (format nil "实得 ~s" got))))

(defun t-binary (label new-fn old-fn)
  (format t "~%=== ~a ≡ 改动前实现 ===~%" label)
  (let ((a (vec *special* :float64))
        (b (vec *special2* :float64)))
    (check-same-as-old (format nil "~a f64 特殊值" label)
                       (lambda () (funcall new-fn a b))
                       (lambda () (funcall old-fn a b)))
    (check-same-as-old (format nil "~a f32 特殊值" label)
                       (lambda () (funcall new-fn (vec (f32 *special*) :float32)
                                           (vec (f32 *special2*) :float32)))
                       (lambda () (funcall old-fn (vec (f32 *special*) :float32)
                                           (vec (f32 *special2*) :float32))))
    (check-same-as-old (format nil "~a 第二操作数为标量" label)
                       (lambda () (funcall new-fn a 2.5d0))
                       (lambda () (funcall old-fn a 2.5d0)))
    (check-same-as-old (format nil "~a 第一操作数为标量" label)
                       (lambda () (funcall new-fn -2.5d0 b))
                       (lambda () (funcall old-fn -2.5d0 b)))
    (check-same-as-old (format nil "~a 广播 (11 2)∘(11 1)" label)
                       (lambda () (funcall new-fn (vt-reshape a '(11 2))
                                           (vt-reshape (vec (subseq *special2* 0 11) :float64)
                                                       '(11 1))))
                       (lambda () (funcall old-fn (vt-reshape a '(11 2))
                                           (vt-reshape (vec (subseq *special2* 0 11) :float64)
                                                       '(11 1)))))
    (check-same-as-old (format nil "~a 转置视图" label)
                       (lambda () (funcall new-fn (vt-transpose (vt-reshape a '(11 2))) b))
                       (lambda () (funcall old-fn (vt-transpose (vt-reshape a '(11 2))) b)))
    (check-same-as-old (format nil "~a 整数输入" label)
                       (lambda () (funcall new-fn (vec (list 1 2 3 -4) :int64)
                                           (vec (list 4 -3 2 1) :int64)))
                       (lambda () (funcall old-fn (vec (list 1 2 3 -4) :int64)
                                           (vec (list 4 -3 2 1) :int64))))
    (check-same-as-old (format nil "~a :out" label)
                       (lambda () (funcall new-fn a b
                                           :out (vt-zeros (list (length *special*)) :dtype :float64)))
                       (lambda () (funcall old-fn a b
                                           :out (vt-zeros (list (length *special*)) :dtype :float64))))))

(defun t-lerp ()
  (format t "~%=== A6 vt-lerp ≡ 改动前实现 ===~%")
  (let ((a (vec *special* :float64))
        (b (vec *special2* :float64)))
    (dolist (w (list 0.0d0 0.25d0 0.5d0 1.0d0 -1.0d0 (clvt:vt-get-nan :float64)))
      (check-same-as-old (format nil "vt-lerp 权重 ~s" w)
                         (lambda () (vt-lerp a b w))
                         (lambda () (old-lerp a b w))))
    (check-same-as-old "vt-lerp 三元广播 (11 2)×(11 2)×标量"
                       (lambda () (vt-lerp (vt-reshape a '(11 2)) (vt-reshape b '(11 2)) 0.25d0))
                       (lambda () (old-lerp (vt-reshape a '(11 2)) (vt-reshape b '(11 2)) 0.25d0)))
    (check-same-as-old "vt-lerp f32"
                       (lambda () (vt-lerp (vec (f32 *special*) :float32)
                                           (vec (f32 *special2*) :float32) 0.5d0))
                       (lambda () (old-lerp (vec (f32 *special*) :float32)
                                            (vec (f32 *special2*) :float32) 0.5d0)))
    (check-same-as-old "vt-lerp :out"
                       (lambda () (vt-lerp a b 0.5d0
                                           :out (vt-zeros (list (length *special*)) :dtype :float64)))
                       (lambda () (old-lerp a b 0.5d0
                                            :out (vt-zeros (list (length *special*)) :dtype :float64))))
    ;; 端点性质：只用**二进制可精确表示**的有限值。
    ;; 不拿 *SPECIAL* 来测的原因：`a + (b−a)·t` 在 t=0 时是 `a + 0·(b−a)`，
    ;; 于是 (a = −0.0) 会给 +0.0（符号位变了）、(b−a) 为 ±Inf/NaN 时会给 NaN ——
    ;; 那是这个定义的固有行为（numpy 的插值公式同样如此），不是实现缺陷。
    (let* ((aa (vec (list 1.0d0 2.5d0 -3.75d0 0.125d0 1024.0d0) :float64))
           (bb (vec (list -4.0d0 0.5d0 7.25d0 -0.5d0 3.0d0) :float64))
           (r0 (flat (vt-lerp aa bb 0.0d0)))
           (r1 (flat (vt-lerp aa bb 1.0d0)))
           (r5 (flat (vt-lerp aa bb 0.5d0))))
      (check "vt-lerp t=0 → a、t=1 → b（逐位）"
             (and (every #'eqv (flat aa) r0) (every #'eqv (flat bb) r1))
             (format nil "t=0 差异 ~a 个 / t=1 差异 ~a 个"
                     (loop for x in (flat aa) for y in r0 count (not (eqv x y)))
                     (loop for x in (flat bb) for y in r1 count (not (eqv x y)))))
      (check "vt-lerp t=0.5 是中点（可精确表示的值）"
             (every (lambda (a b m) (eqv (/ (+ a b) 2.0d0) m))
                    (flat aa) (flat bb) r5)
             (format nil "实得 ~s" r5)))))

;;; ==================================================================
;;; 4. B 段：机器无关的倍率护栏
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

(defparameter *big* 1000000)

(defun big-f64 (&optional (offset 1.0d0))
  (vt-+ (vt-astype (vt-arange *big* :start 0 :step 1 :dtype :float64) :float64) offset))

(defun big-f32 () (vt-astype (big-f64) :float32))
(defun big-i64 () (vt-arange *big* :start 1 :step 1 :dtype :int64))

(defmacro check-ratio (label limit new old)
  `(let ((fast (min-us (lambda () ,new)))
         (slow (min-us (lambda () ,old) :reps 3)))
     (check-at-least ,label (/ slow fast) ,limit "old/new")))

(defun t-ratios ()
  (format t "~%=== B 段 机器无关倍率护栏（1M 元素，同进程同机）===~%")
  (let ((f (big-f64)) (g (big-f64 2.0d0)) (s (big-f32)) (i (big-i64)))
    (check-ratio "B01 rint f64" 4.0 (vt-rint f) (old-rint f))
    (check-ratio "B02 rint f32" 3.0 (vt-rint s) (old-rint s))
    (check-ratio "B03 rint int64" 2.0 (vt-rint i) (old-rint i))
    (check-ratio "B04 sinc f64" 1.4 (vt-sinc f) (old-sinc f))
    (check-ratio "B05 reciprocal f64" 2.0 (vt-reciprocal f) (old-reciprocal f))
    (check-ratio "B06 reciprocal f32" 2.0 (vt-reciprocal s) (old-reciprocal s))
    (check-ratio "B07 deg2rad f64" 2.5 (vt-deg2rad f) (old-deg2rad f))
    (check-ratio "B08 deg2rad f32" 2.5 (vt-deg2rad s) (old-deg2rad s))
    (check-ratio "B09 rad2deg f64" 2.5 (vt-rad2deg f) (old-rad2deg f))
    (check-ratio "B10 rad2deg f32" 2.5 (vt-rad2deg s) (old-rad2deg s))
    (check-ratio "B11 copysign f64" 2.0 (vt-copysign f g) (old-copysign f g))
    (check-ratio "B12 logaddexp f64" 1.3 (vt-logaddexp f g) (old-logaddexp f g))
    (check-ratio "B13 lerp f64" 3.0 (vt-lerp f g 0.5d0) (old-lerp f g 0.5d0))))

;;; ==================================================================
;;; 主流程
;;; ==================================================================
(defun run ()
  (format t "~%============================================================~%")
  (format t "  逐元素算子「具名内联」优化 回归测试（v0.4.20）~%")
  (format t "============================================================~%")
  (t-rint)
  (t-unary "vt-sinc" #'vt-sinc #'old-sinc)
  (t-reciprocal)
  (t-unary "vt-deg2rad" #'vt-deg2rad #'old-deg2rad)
  (t-unary "vt-rad2deg" #'vt-rad2deg #'old-rad2deg)
  (t-binary "vt-copysign" #'vt-copysign #'old-copysign)
  (t-binary "vt-logaddexp" #'vt-logaddexp #'old-logaddexp)
  (t-lerp)
  (t-ratios)
  (format t "~%============================================================~%")
  (format t "  Total: ~a | Pass: ~a | Fail: ~a | Skip: 0~%" *n* *pass* *fail*)
  (format t "============================================================~%")
  (when *failures*
    (format t "~%Failed:~{~%  - ~a~}~%" (reverse *failures*)))
  (zerop *fail*))

(sb-ext:exit :code (if (run) 0 1))
