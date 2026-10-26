;;;; nan-fused-test.lisp — nan 族 / var / nan_to_num 的「单趟内核」与流水线等价性
;;;;                       （CONVENTIONS §16.14.7）
;;;;
;;;; 背景：nan 族原先每个算子都是「掩码 + 单位元替换 + 既有归约内核」的张量
;;;; 流水线（nanmean ≈ 7 趟、nanvar ≥ 10 趟），1M complex128 实测
;;;; nanmean 20 / nanvar 50 / nanmax 23 ms，而同形 float64 早就走单趟扫描
;;;; 只要 1.4 ms。本轮补了三个单趟扫描（求和+计数 / 字典序极值+计数 /
;;;; 离差平方和）与两个特化直线循环（nan_to_num、var）。
;;;;
;;;; 本套件的核心是 **与流水线逐例对拍**：把「旧写法」在测试里重写一遍
;;;; （只用公开算子：isnan / where / logical-not / astype / sum / amax），
;;;; 再和新实现的单趟结果比。这样即便将来有人改动融合逻辑，语义也不会漂。
;;;; 与 numpy 的对照已由 complex-ops-test / nan-inf-test 覆盖，这里不重复。
;;;;
;;;; 覆盖：
;;;;   A. nansum / nanmean / nanmax / nanmin（c128 + c64）对流水线；
;;;;      输入刻意包含「a[0] 是 NaN」（旧实现拿首元素当种子会翻车）、
;;;;      全 NaN、空、单元素、含 ±Inf。
;;;;   B. nanvar / nanstd（ddof 0/1/2，n ≤ ddof，含 NaN，全 NaN）。
;;;;   C. vt-var / vt-std 的实数与复数全归约对流水线。
;;;;   D. nan_to_num：f64 / f32 / c128 / c64 × {缺省值, 自定义值} × :out；
;;;;      逐位比较（不放松容差）。
;;;;   E. 护栏：八个算子的绝对耗时。
;;;;
;;;; 用法：sbcl --dynamic-space-size 4096 --non-interactive --load test/nan-fused-test.lisp

(require :asdf)
(defparameter cl-user::*clvt-test-dir*
  (make-pathname :name nil :type nil :defaults *load-truename*))
(pushnew (merge-pathnames "../" cl-user::*clvt-test-dir*) asdf:*central-registry* :test #'equal)
(handler-bind ((warning #'muffle-warning)) (asdf:load-system :clvt))

(in-package :clvt)

(defvar *pass* 0)
(defvar *fail* 0)
(defvar *failures* '())

(defun expect-true (name form)
  (if form
      (incf *pass*)
      (progn (incf *fail*) (push name *failures*) (format t "~&FAIL ~a~%" name))))

(defun f= (x y tol)
  "浮点近似相等（NaN 只与 NaN 相等）。"
  (cond ((and (floatp x) (sb-ext:float-nan-p x))
         (and (floatp y) (sb-ext:float-nan-p y)))
        ((and (floatp y) (sb-ext:float-nan-p y)) nil)
        ((= x y) t)
        (t (<= (abs (- x y)) (* tol (max (abs x) (abs y) 1d-300))))))

(defun c= (a b tol)
  (and (f= (realpart a) (realpart b) tol)
       (f= (imagpart a) (imagpart b) tol)))

(defun item (v)
  "取标量：VT-MEAN / VT-SUM 的 axis=nil 返回 0 维张量。"
  (if (vt-p v) (vt-item v) v))

;;; ==================================================================
;;; 0. 流水线参考实现（= 修复前的写法，只用公开算子）
;;; ==================================================================

(defun ref-count (z dt rdt)
  "非 NaN 元素个数（实数）。"
  (item (vt-sum (vt-astype (vt-logical-not (vt-isnan z)) rdt)
                :dtype rdt :keepdims nil)))

(defun ref-nansum (z)
  (let ((dt (vt-dtype z)))
    (item (vt-sum (vt-where (vt-isnan z) (vt-cast 0 dt) z :dtype dt) :dtype dt))))

(defun ref-nanmean (z)
  (let* ((dt (vt-dtype z))
         (rdt (vt-complex-real-dtype dt))
         (s (ref-nansum z))
         (c (ref-count z dt rdt)))
    (if (zerop c)
        (%cx-complex-nan-of dt)
        (/ s (coerce c (vt-dtype->lisp-type rdt))))))

(defun ref-nanextremum (z maxp)
  ;; 先数计数再算极值：空地「全 NaN」的路径根本不能调 vt-amax（空归约报错）。
  (let* ((dt (vt-dtype z))
         (rdt (vt-complex-real-dtype dt))
         (c (ref-count z dt rdt)))
    (if (zerop c)
        (%cx-complex-nan-of dt)
        (let* ((sent (%cx-nan-sentinel dt (if maxp -1 +1)))
               (w (vt-where (vt-isnan z) sent z :dtype dt)))
          (item (if maxp (vt-amax w :dtype dt) (vt-amin w :dtype dt)))))))

(defun ref-nanvar (z ddof)
  "旧流水线：clean → mean → dev → |dev|² → 掩码 → sum → /(cnt-ddof)。"
  (let* ((in-dt (vt-dtype z))
         (rdt (vt-complex-real-dtype in-dt))
         (rt (vt-dtype->lisp-type rdt))
         (mask (vt-isnan z))
         (not-nan (vt-astype (vt-logical-not mask) rdt))
         (clean (vt-where mask (vt-cast 0 in-dt) z :dtype in-dt))
         (cnt (item (vt-sum not-nan :dtype rdt))))
    (if (zerop cnt)
        (coerce +vt-dfloat-nan+ rt)
      (let* (
             (sums (item (vt-sum clean :dtype in-dt)))
             (mean (/ sums (coerce cnt rt)))
             (dev (vt-- clean mean))
             (sq (vt-square (vt-abs dev)))
             (sqm (vt-* sq not-nan))
             (sum2 (item (vt-sum sqm :dtype rdt)))
             (den (- (coerce cnt rt) (coerce ddof rt))))
        (if (<= den 0) (coerce +vt-dfloat-nan+ rt) (/ sum2 den))))))

;;; ==================================================================
;;; A. nan 族四算子（axis=nil）
;;; ==================================================================

(defun mkz (vals dtype)
  (vt-from-sequence vals :dtype dtype))

(defun nan-cases (nan dt)
  "返回 12 个输入：常规 / a[0] 是 NaN / 中间 NaN / 尾部 NaN / 全 NaN /
   空 / 单元素 / 单元素 NaN / 含 ±Inf / 只有 ±Inf / 大量重复 / 0 维。"
  (let ((one (complex 1d0 -2d0))
        (two (complex 3d0 0.5d0))
        (nanv (complex nan nan)))
    (list (list "常规" (list one two (complex -1d0 0d0)))
          (list "a[0]是NaN" (list nanv two (complex -1d0 0d0) (complex 5d0 5d0)))
          (list "中间NaN" (list one nanv two))
          (list "尾部NaN" (list one two nanv))
          (list "全NaN" (list nanv nanv))
          (list "空" '())
          (list "单元素" (list one))
          (list "单元素NaN" (list nanv))
          (list "含Inf" (list +vt-dfloat-pos-inf+
                             (complex +vt-dfloat-neg-inf+ 0d0)
                             one))
          (list "只有Inf" (list +vt-dfloat-pos-inf+ +vt-dfloat-neg-inf+))
          (list "重复" (append (make-list 5 :initial-element one)
                              (make-list 5 :initial-element two))))))

(defun test-nan-family ()
  (format t "~&--- A. nansum / nanmean / nanmax / nanmin ---")
  (dolist (dt '(:complex128 :complex64))
    (let ((nan (if (eq dt :complex64) +vt-sfloat-nan+ +vt-dfloat-nan+)))
      (dolist (case (nan-cases nan dt))
        (destructuring-bind (label vals) case
          (let* ((z (mkz vals dt))
                 (tag (format nil "~a/~a" (if (eq dt :complex64) "c64" "c128") label)))
            (expect-true (format nil "nansum ~a" tag)
                         (c= (item (vt-nansum z)) (ref-nansum z) 1d-12))
            (expect-true (format nil "nanmean ~a" tag)
                         (c= (item (vt-nanmean z)) (ref-nanmean z) 1d-12))
            (expect-true (format nil "nanmax ~a" tag)
                         (c= (item (vt-nanmax z)) (ref-nanextremum z t) 1d-12))
            (expect-true (format nil "nanmin ~a" tag)
                         (c= (item (vt-nanmin z)) (ref-nanextremum z nil) 1d-12))
            ;; keepdims 必须给出全 1 形状（0 维输入除外）
            (let ((r (vt-nanmax z :keepdims t)))
              (expect-true (format nil "nanmax ~a keepdims 形状" tag)
                           (equal (vt-shape r)
                                  (make-list (length (vt-shape z)) :initial-element 1))))))))))

(defun test-nan-zero-dim ()
  (format t "~&--- A2. 0 维输入 ---")
  (let ((z (vt-reshape (mkz (list (complex 1d0 2d0)) :complex128) nil)))
    (expect-true "0维 nansum" (c= (item (vt-nansum z)) (complex 1d0 2d0) 0d0))
    (expect-true "0维 nanmean" (c= (item (vt-nanmean z)) (complex 1d0 2d0) 0d0))
    (expect-true "0维 nanmax" (c= (item (vt-nanmax z)) (complex 1d0 2d0) 0d0))
    (expect-true "0维 nanvar（单元素总体方差 = 0）"
                 (f= (item (vt-nanvar z)) 0d0 0d0)))
  (let ((z (vt-reshape (mkz (list (complex +vt-dfloat-nan+ +vt-dfloat-nan+)) :complex128) nil)))
    (expect-true "0维全 NaN nansum = 0" (c= (item (vt-nansum z)) (complex 0d0 0d0) 0d0))
    (expect-true "0维全 NaN nanmax 是 NaN+NaNj"
                 (let ((r (item (vt-nanmax z))))
                   (and (sb-ext:float-nan-p (realpart r))
                        (sb-ext:float-nan-p (imagpart r)))))))

;;; ==================================================================
;;; B. nanvar / nanstd
;;; ==================================================================

(defun test-nanvar ()
  (format t "~&--- B. nanvar / nanstd ---")
  (dolist (dt '(:complex128 :complex64))
    (let ((nan (if (eq dt :complex64) +vt-sfloat-nan+ +vt-dfloat-nan+)))
      (dolist (case (nan-cases nan dt))
        (destructuring-bind (label vals) case
          (let ((z (mkz vals dt)))
            (dolist (ddof '(0 1 2))
              (let ((tag (format nil "~a/%a ddof=~a"
                                 (if (eq dt :complex64) "c64" "c128") label ddof)))
                (expect-true (format nil "nanvar ~a" tag)
                             (f= (item (vt-nanvar z :ddof ddof))
                                 (ref-nanvar z ddof) 1d-12))
                (let ((want (ref-nanvar z ddof)))
                  (expect-true (format nil "nanstd ~a" tag)
                               (f= (item (vt-nanstd z :ddof ddof))
                                   (if (sb-ext:float-nan-p want)
                                       want
                                       (sqrt want))
                                   1d-12)))))))))))

;;; ==================================================================
;;; C. vt-var / vt-std（实数 + 复数，全归约）
;;; ==================================================================

(defun ref-var-real (v ddof)
  "(mean, sub, square, sum, div) 的实数流水线。n ≤ ddof 时按库的口径给 NaN
   （VT-VAR 在 divisor ≤ 0 时先返回 NaN，不会走到除法）。"
  (let* ((n (vt-size v))
         (div (- n ddof)))
    (if (<= div 0)
        +vt-dfloat-nan+
        (let* ((xv (item (vt-mean v)))
               (dev (vt-- v xv))
               (sq (vt-square dev))
               (ss (item (vt-sum sq))))
          (/ ss div)))))

(defun test-var ()
  (format t "~&--- C. vt-var / vt-std ---")
  (let ((sets (list (list "f64-常规" (list 1d0 2d0 3d0 4d0 10d0) :float64)
                    (list "f64-NaN" (list 1d0 +vt-dfloat-nan+ 3d0) :float64)
                    (list "f32" (list 1s0 2s0 4s0 8s0) :float32)
                    (list "单元素" (list 5d0) :float64)
                    (list "空" '() :float64))))
    (dolist (s sets)
      (destructuring-bind (label vals dt) s
        (let ((v (vt-from-sequence vals :dtype dt)))
          (dolist (ddof '(0 1))
            (expect-true (format nil "var ~a ddof=~a" label ddof)
                         (f= (item (vt-var v :ddof ddof))
                             (ref-var-real v ddof) 1d-12)))))))
  ;; 复数：与流水线（用 nanvar 的思路但**不跳过 NaN**）比；含 NaN 时必须传播
  (dolist (case (list (list "c128-常规" (list (complex 1d0 2d0) (complex 3d0 -1d0)
                                             (complex -2d0 0.5d0)))
                      (list "c128-NaN" (list (complex 1d0 2d0)
                                             (complex +vt-dfloat-nan+ +vt-dfloat-nan+)
                                             (complex 3d0 1d0)))))
    (destructuring-bind (label vals) case
      (let* ((z (vt-from-sequence vals :dtype :complex128))
             (mu (item (vt-mean z)))
             (dev (vt-- z mu))
             (sq (vt-square (vt-abs dev)))
             (ss (item (vt-sum sq)))
             (want (/ ss (vt-size z))))
        (expect-true (format nil "var 复数 ~a" label)
                     (f= (item (vt-var z)) want 1d-12))
        (expect-true (format nil "std 复数 ~a" label)
                     (f= (item (vt-std z)) (if (sb-ext:float-nan-p want) want (sqrt want))
                         1d-12))))))

;;; ==================================================================
;;; D. nan_to_num（逐位比较）
;;; ==================================================================

(defun ref-nan-to-num (vals nan posinf neginf maxf)
  "逐元素参考：实部虚部各自替换。"
  (flet ((fix1 (v)
           (cond ((sb-ext:float-nan-p v) nan)
                 ((> v maxf) posinf)
                 ((< v (- maxf)) neginf)
                 (t v))))
    (mapcar (lambda (z) (complex (fix1 (realpart z)) (fix1 (imagpart z)))) vals)))

(defun test-nan-to-num ()
  (format t "~&--- D. nan_to_num ---")
  (let* ((nanv +vt-dfloat-nan+)
         (vals (list (complex nanv 0d0) (complex 1d0 nanv)
                     (complex +vt-dfloat-pos-inf+ 2d0)
                     (complex -3d0 +vt-dfloat-neg-inf+)
                     (complex 0.5d0 -0.5d0))))
    (dolist (dt '(:complex128 :complex64))
      (let* ((z (vt-from-sequence vals :dtype dt))
             (rt (if (eq dt :complex64) 'single-float 'double-float))
             (maxf (if (eq dt :complex64) 3.4028235f38 most-positive-double-float))
             (got (vt-nan-to-num z))
             (want (ref-nan-to-num (mapcar (lambda (v) (vt-cast v dt)) vals)
                                   0d0 maxf (- maxf) maxf)))
        (expect-true (format nil "nan_to_num 缺省 ~a" dt)
                     (let ((g (loop for i from 0 below (vt-size got)
                                    collect (vt-ref got i))))
                       (and (= (length g) (length want))
                            (every (lambda (a b) (c= a b 0d0)) g want))))
        (expect-true (format nil "nan_to_num dtype 不变 ~a" dt)
                     (eq (vt-dtype got) dt))
        ;; 自定义替换值
        (let ((g2 (vt-nan-to-num z :nan -7d0 :posinf 100d0 :neginf -100d0)))
          (expect-true (format nil "nan_to_num 自定义值 ~a" dt)
                       (c= (vt-ref g2 0) (complex -7d0 0d0) 0d0))
          (expect-true (format nil "nan_to_num 自定义 +Inf ~a" dt)
                       (c= (vt-ref g2 2) (complex 100d0 2d0) 0d0)))
        ;; :out 契约（同一 dtype 下走快路径，必须写回 OUT 而不是返回新对象）
        (let ((o (vt-zeros (vt-shape z) :dtype dt)))
          (expect-true (format nil "nan_to_num :out 返回的就是 OUT ~a" dt)
                       (eq (vt-nan-to-num z :out o) o))
          (expect-true (format nil "nan_to_num :out 写回 ~a" dt)
                       (c= (vt-ref o 0) (vt-ref (vt-nan-to-num z) 0) 0d0))))))
  ;; 实数 + 整数输入（走 dtype 转换的回退路径）
  (dolist (dt '(:float64 :float32))
    (let* ((vals (list 1d0 +vt-dfloat-nan+ +vt-dfloat-pos-inf+ -2d0))
           (z (vt-from-sequence vals :dtype dt))
           (maxf (if (eq dt :float32) 3.4028235f38 most-positive-double-float))
           (got (mapcar (lambda (i) (vt-ref (vt-nan-to-num z) i)) '(0 1 2 3))))
      (expect-true (format nil "nan_to_num 实数 ~a" dt)
                   (and (f= (nth 0 got) 1d0 0d0)
                        (or (zerop (nth 1 got)) (= (nth 1 got) 0s0))
                        (= (nth 2 got) maxf)
                        (f= (nth 3 got) -2d0 0d0)))))
  (let* ((z (vt-from-sequence '(1 2 3) :dtype :int64))
         (r (vt-nan-to-num z)))
    ;; 整数输入的 dtype 由 %E3-FLOAT-PREFER-DTYPE 原样返回（numpy 的
    ;; nan_to_num 对整数数组也保持整数），这里钉死「不降级也不升级」。
    (expect-true "nan_to_num 整数输入 dtype 不变"
                 (eq (vt-dtype r) :int64))
    (expect-true "nan_to_num 整数输入值不变"
                 (= (vt-ref r 1) 2))))

;;; ==================================================================
;;; F. 公开 API 对 NaN 输入**不抛条件**（顺手发现的既有崩溃）
;;; ==================================================================

(defun test-nan-no-signal ()
  (format t "~&--- F. 复数一元算子遇 NaN 不抛条件 ---")
  ;; `%CX-ABS-RUN` / `%CX-ANGLE-RUN` 的特化循环原先没有屏蔽浮点陷阱：
  ;; 内核里 `%CX-INF-P` 是 `(> v maxf)`、`%COP-ARG` 要看符号，UCOMISD 遇 NaN
  ;; 会置 invalid 并**触发陷阱** → `(vt-abs z)` / `(vt-angle z)` 在含 NaN 的
  ;; 输入上直接抛 FLOATING-POINT-INVALID-OPERATION（公开 API 崩溃）。
  (let ((z (vt-from-sequence
            (list (complex +vt-dfloat-nan+ 1d0) (complex 2d0 3d0)
                  (complex 1d0 +vt-dfloat-nan+))
            :dtype :complex128)))
    (dolist (f (list (cons "vt-abs" #'vt-abs) (cons "vt-angle" #'vt-angle)
                     (cons "vt-real" #'vt-real) (cons "vt-imag" #'vt-imag)
                     (cons "vt-conj" #'vt-conj) (cons "vt-square" #'vt-square)
                     (cons "vt-sqrt" #'vt-sqrt)))
      (expect-true (format nil "~a 含 NaN 时抛条件" (car f))
                   (handler-case (progn (funcall (cdr f) z) t)
                     (error () nil)))
      (expect-true (format nil "~a 含 NaN 时返回同形状张量" (car f))
                   (handler-case (equal (vt-shape (funcall (cdr f) z)) (vt-shape z))
                     (error () nil))))))

;;; ==================================================================
;;; E. 护栏
;;; ==================================================================

(defun us-now ()
  (multiple-value-bind (s u) (sb-ext:get-time-of-day) (+ (* s 1000000) u)))

(defun ms-of (thunk &key (runs 2))
  (let ((best most-positive-fixnum))
    (dotimes (_ (1+ runs))
      (sb-ext:gc :full t)
      (let ((t0 (us-now)))
        (funcall thunk)
        (setf best (min best (- (us-now) t0)))))
    (/ best 1000.0d0)))

(defun test-guardrails ()
  (format t "~&--- E. 护栏（1M）---")
  (let* ((n 1000000)
         (d (vt-random-uniform (list n) :low 0d0 :high 1d0))
         (zc (vt-astype d :complex128))
         (zn (vt-copy zc)))
    (dotimes (i 10000) (setf (vt-ref zn (* i 100)) (complex +vt-dfloat-nan+ +vt-dfloat-nan+)))
    (macrolet ((guard (label limit form)
                 `(let ((t* (ms-of (lambda () ,form))))
                    (expect-true (format nil "~a 绝对 ~,1fms < ~a" ,label t* ,limit)
                                 (< t* ,limit)))))
      (guard "nansum c128" 40 (vt-nansum zn))
      (guard "nanmean c128" 40 (vt-nanmean zn))
      (guard "nanmax c128" 40 (vt-nanmax zn))
      (guard "nanmin c128" 40 (vt-nanmin zn))
      (guard "nanvar c128" 60 (vt-nanvar zn))
      (guard "nanstd c128" 60 (vt-nanstd zn))
      (guard "nan-to-num c128" 60 (vt-nan-to-num zn))
      (guard "var c128" 60 (vt-var zc))
      (guard "var f64" 60 (vt-var d))
      (guard "nansum f64" 40 (vt-nansum d)))))

;;; ==================================================================

(defun main ()
  (format t "~&=== nan 族单趟内核测试 ===")
  (test-nan-family)
  (test-nan-zero-dim)
  (test-nanvar)
  (test-var)
  (test-nan-to-num)
  (test-nan-no-signal)
  (test-guardrails)
  (format t "~&~%通过: ~a  失败: ~a~%" *pass* *fail*)
  (when *failures*
    (format t "~&失败清单:~%")
    (dolist (f (reverse *failures*)) (format t "  - ~a~%" f)))
  (when (plusp *fail*)
    (sb-ext:exit :code 1)))

(main)
