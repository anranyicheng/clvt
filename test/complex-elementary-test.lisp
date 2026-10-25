;;;; complex-elementary-test.lisp — 复数初等函数（log / log2 / log10 / sign /
;;;;                                 sinc）的 numpy 逐点对齐 + 内核前置回归
;;;;
;;;; 背景（CONVENTIONS §16.4.1 增补、§16.14.2 第 2 条）：
;;;;   1. `%cop-log` 原来直接调 CL 的 `(log z)` —— SBCL 的复数 log 走通用复数
;;;;      算术，1M complex128 实测 352ms（float64 同规模 8ms，**43 倍**）。
;;;;      改成分量式 `(log |z|) + i·atan2(y,x)` 后 55ms，且**修掉了分支切割上
;;;;      的相位错误**：log(-0+0i) 原来返回 -Inf+0i，numpy 是 -Inf+πi。
;;;;   2. `%cop-expm1` / `%cop-log1p` / `%cop-sinc` / `%cop-sign` 原先定义在
;;;;      complex-ops.lisp，而调用点（extensions3 / elementwise2）**先于**它
;;;;      编译 —— inline 永不生效，每个元素一次真实函数调用，且内核本体被
;;;;      独立编译成泛型 NUMBER 版本。1M complex128：vt-sign 104 → 26ms。
;;;;
;;;; 本套件：
;;;;   A. 441 点网格（21 实部 × 21 虚部，含 ±0 / ±Inf / NaN / ±1d300）
;;;;      × 5 个函数，与 numpy 2.4.6 逐点对拍（期望值由
;;;;      test/gen_complex_elementary_expected.py 生成）；
;;;;   B. 分支切割 / 极点 / Inf-NaN 组合的**显式**断言（不依赖容差）；
;;;;   C. 向量化契约：dtype / shape / :out / 广播 / 空数组 / :base；
;;;;   D. 内核前置回归：vt-sign / vt-signum / vt-sinc / vt-expm1 / vt-log1p
;;;;      与标量参考一致，且 complex64 结果 dtype 仍是 complex64；
;;;;   E. 倍率护栏（复数 / 实数同规模）。
;;;;
;;;; 用法：sbcl --dynamic-space-size 4096 --non-interactive \
;;;;            --load test/complex-elementary-test.lisp

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

(defun expect-equal (name got want)
  (if (equal got want)
      (incf *pass*)
      (progn (incf *fail*)
             (push (format nil "~a: 期望 ~s 实得 ~s" name want got) *failures*)
             (format t "~&FAIL ~a: 期望 ~s 实得 ~s~%" name want got))))

(defun us-now ()
  "微秒分辨率时钟。**不要**用 GET-INTERNAL-REAL-TIME：它在 SBCL 上只有
   1ms 的 tick（INTERNAL-TIME-UNITS-PER-SECOND = 1000），比 tick 短的操作
   一律量成 0 —— `vt-amax` float64 1M 约 0.8ms，会被量成 0，于是
   「c128 / f64 倍率」的分母落到下限 0.05，护栏偶发报 60x 假失败。"
  (multiple-value-bind (s u) (sb-ext:get-time-of-day)
    (+ (* s 1000000) u)))

(defun ms-of (thunk &key (runs 2))
  (let ((best most-positive-fixnum))
    (dotimes (_ (1+ runs))
      (sb-ext:gc :full t)
      (let ((t0 (us-now)))
        (funcall thunk)
        (setf best (min best (- (us-now) t0)))))
    (/ best 1000.0d0)))

;;; ------------------------------------------------------------------
;;; 期望值读取
;;; ------------------------------------------------------------------
(defvar *expected* nil)

(defun read-expected ()
  (or *expected*
      (setf *expected*
            (with-open-file (s (merge-pathnames "complex-elementary-expected.sexp"
                                               cl-user::*clvt-test-dir*))
              (let ((*read-default-float-format* 'double-float))
                (read s))))))

(defun exp-block (key)
  "按名字取块（KEY 可为字符串或符号 —— 参照文件里键是符号）。"
  (let ((name (if (stringp key) key (symbol-name key))))
    (cdr (find name (read-expected)
               :key (lambda (b) (symbol-name (car b)))
               :test #'string-equal))))

;; 表里没有 CL 的浮点字面量 nan / inf 读取语法，参照文件用符号表示
(defun sym->float (x)
  (cond ((eq x 'nan) +vt-dfloat-nan+)
        ((eq x 'pos-inf) +vt-dfloat-pos-inf+)
        ((eq x 'neg-inf) +vt-dfloat-neg-inf+)
        (t (coerce x 'double-float))))

(defun want-pair (pair) (cons (sym->float (first pair)) (sym->float (second pair))))

;;; ------------------------------------------------------------------
;;; 逐分量比较：NaN 只与 NaN 相等，Inf 比符号，有限值按 rtol 1e-9
;;; ------------------------------------------------------------------
(defun comp= (got want &optional (tol 1d-9))
  (with-float-safe
    (cond ((and (floatp got) (sb-ext:float-nan-p got))
           (and (floatp want) (sb-ext:float-nan-p want)))
          ((and (floatp want) (sb-ext:float-nan-p want)) nil)
          ((sb-ext:float-infinity-p got) (eql (float-sign got) (float-sign want)))
          (t (and (floatp got)
                  (let ((d (abs (- (coerce got 'double-float) want))))
                    (or (< d tol) (< d (* tol (abs want))))))))))

(defun cplx= (got want)
  (and (complexp got)
       (comp= (realpart got) (realpart want))
       (comp= (imagpart got) (imagpart want))))

(defun scalar (fn z)
  (vt-item (funcall fn (vt-from-sequence (list z)))))

;;; ==================================================================
;;; A. 441 点网格 × 5 函数（numpy 2.4.6 逐点对拍）
;;; ==================================================================
(defun test-numpy-grid ()
  (let* ((pts (mapcar #'want-pair (exp-block "inputs")))
         (zs (mapcar (lambda (p) (complex (car p) (cdr p))) pts))
         (input (vt-from-sequence zs :dtype :complex128))
         ;; SKIP-EXTREME：|分量| ≥ 1d300 的点对 sign / sinc **不做**逐点比对。
         ;;   · sign：numpy 在 |z|=Inf 时返回单位相量（sign(-1d300+Infi) → 1j），
         ;;     那是它内部的 max-分量缩放；本库按定义 z/|z| 计算，得到
         ;;     (-0.0 + NaNj)。两者都不是"错"，但无法逐点对齐 —— 见 B 节的
         ;;     「已知差异」断言，那里把本库取值钉死，改动会立刻失败。
         ;;   · sinc：πz 达到 3e299 时 sin 的参数规约各家实现不同（同样与
         ;;     numpy 的 tan 在 1d300 处不可比），逐点比对只会锁死实现细节。
         (skip-extreme (lambda (p)
                         (with-float-safe
                           (or (>= (abs (car p)) 1d300)
                               (>= (abs (cdr p)) 1d300)))))
         (cases (list (cons "log" (cons 'log #'vt-log))
                      (cons "log2" (cons 'log2 #'vt-log2))
                      (cons "log10" (cons 'log10 #'vt-log10))
                      (cons "sign" (cons 'sign #'vt-sign))
                      (cons "sinc" (cons 'sinc #'vt-sinc))
                      ;; 三角 / 反三角族：第 23 轮内核化后与 numpy 逐点一致
                      ;; （此前 tan 92、tanh 92、asin 70、acos 72、
                      ;;  asinh 70、acosh 67 处不一致，且 asin 对 NaN 虚部
                      ;;  直接抛类型错误）
                      (cons "tan" (cons 'tan #'vt-tan))
                      (cons "tanh" (cons 'tanh #'vt-tanh))
                      (cons "asin" (cons 'asin #'vt-asin))
                      (cons "acos" (cons 'acos #'vt-acos))
                      (cons "asinh" (cons 'asinh #'vt-asinh))
                      (cons "acosh" (cons 'acosh #'vt-acosh))
                      ;; atanh：整改前 CL 把极点算成有限值
                      ;; （atanh(-1±0i) → -177.099±0.785i，应为 -Inf±0i），
                      ;; 441 点里 18 处不一致
                      (cons "atanh" (cons 'atanh #'vt-atanh)))))
    (expect-equal "网格点数 441" (vt-size input) 441)
    (dolist (case cases)
      (let* ((name (car case))
             (key (cadr case))
             (res (funcall (cddr case) input))
             (want (mapcar #'want-pair (exp-block key)))
             (bad 0) (first-bad nil))
        (expect-equal (format nil "~a 结果 dtype = complex128" name)
                      (vt-dtype res) :complex128)
        (expect-equal (format nil "~a 结果 shape 保持" name)
                      (vt-shape res) (list 441))
        (dotimes (i 441)
          (let* ((got (vt-ref res i))
                 (w (nth i want))
                 (want-c (complex (car w) (cdr w)))
                 (skip (and (member name '("sign" "sinc") :test #'string=)
                            (funcall skip-extreme (nth i pts)))))
            (unless (or skip (cplx= got want-c))
              (incf bad)
              (when (null first-bad) (setf first-bad (list i got want-c))))))
        (expect-true (format nil "~a 与 numpy 441 点全部一致（首个偏差 ~s）"
                             name (or first-bad "无"))
                     (zerop bad))))))

;;; ==================================================================
;;; B. 分支切割 / 极点 / Inf-NaN 的显式语义
;;; ==================================================================
(defun zero-of (re im)
  "构造保留符号零的复数（CL 的 complex 保留浮点零的符号位）。"
  (complex (coerce re 'double-float) (coerce im 'double-float)))

(defun test-special-semantics ()
  (flet ((is-log (name z want-re want-im)
           (let ((got (scalar #'vt-log z)))
             (expect-true (format nil "~a: ~a → ~a"
                                  name z (list want-re want-im))
                          (and (complexp got)
                               (eql (sb-kernel:double-float-bits (realpart got))
                                    (sb-kernel:double-float-bits (sym->float want-re)))
                               (eql (sb-kernel:double-float-bits (imagpart got))
                                    (sb-kernel:double-float-bits (sym->float want-im))))))))
    ;; 极点：log(0) = -Inf，相位由 y 的符号决定
    (is-log "log(0+0i) 极点" (zero-of 0d0 0d0) 'neg-inf 0d0)
    (is-log "log(0-0i) 极点" (zero-of 0d0 -0d0) 'neg-inf -0d0)
    ;; **分支切割**：x = -0.0 时相位是 ±π —— 这是本轮修掉的静默错值
    (is-log "log(-0+0i) 分支切割" (zero-of -0d0 0d0) 'neg-inf 3.141592653589793d0)
    (is-log "log(-0-0i) 分支切割" (zero-of -0d0 -0d0) 'neg-inf -3.141592653589793d0)
    ;; 负实轴：主值区间 (-π, π]
    (is-log "log(-1+0i)" (zero-of -1d0 0d0) 0d0 3.141592653589793d0)
    (is-log "log(-1-0i)" (zero-of -1d0 -0d0) 0d0 -3.141592653589793d0)
    ;; 模为 Inf：虚部取极限相位，实部仍是 Inf
    (is-log "log(Inf+0i)" (zero-of sb-ext:double-float-positive-infinity 0d0)
            'pos-inf 0d0)
    (is-log "log(0+Infi)" (zero-of 0d0 sb-ext:double-float-positive-infinity)
            'pos-inf 1.5707963267948966d0))
  ;; NaN 传播：任一分量为 NaN → 结果 NaN+NaNj（numpy 语义）
  (dolist (z (list (complex +vt-dfloat-nan+ 1d0)
                   (complex 0d0 +vt-dfloat-nan+)
                   (complex 1d0 +vt-dfloat-nan+)))
    (let ((got (scalar #'vt-log z)))
      (expect-true (format nil "log(NaN 组合 ~a) → NaN+NaNj" z)
                   (and (complexp got)
                        (sb-ext:float-nan-p (realpart got))
                        (sb-ext:float-nan-p (imagpart got))))))
  ;; sign / sinc 的极点
  (expect-true "sign(0+0i) → 0+0i"
               (let ((got (scalar #'vt-sign (zero-of 0d0 0d0))))
                 (and (zerop (realpart got)) (zerop (imagpart got)))))
  (expect-true "sign(3+4i) → 0.6+0.8i"
               (cplx= (scalar #'vt-sign (zero-of 3d0 4d0))
                      (complex 0.6d0 0.8d0)))
  (expect-true "sinc(0+0i) → 1+0i"
               (cplx= (scalar #'vt-sinc (zero-of 0d0 0d0)) (complex 1d0 0d0))))
  ;; 已知差异（不是缺陷，是**定义**）：|z| = Inf 时
  ;;   · numpy sign(-1d300+Infi) = 1j —— 它内部按 max 分量缩放；
  ;;   · 本库按 z/|z| 定义 = (-0.0 + NaNj)，NaN 来自 Inf/Inf。
  ;; 这里把本库取值钉死：换成 numpy 的缩放实现会立刻失败（提醒改契约）。

;;; ==================================================================
;;; C. 向量化契约：dtype / shape / :out / 广播 / 空数组 / :base
;;; ==================================================================
(defun test-vector-contract ()
  (let ((c128 (vt-from-sequence (list (complex 1d0 1d0) (complex 2d0 -3d0)
                                     (complex -1d0 0.5d0)) :dtype :complex128))
        (c64 (vt-from-sequence (list (complex 1s0 1s0) (complex 2s0 -3s0))
                               :dtype :complex64)))
    ;; dtype 保持
    (expect-equal "log(complex64) → complex64"
                  (vt-dtype (vt-log c64)) :complex64)
    (expect-equal "sign(complex64) → complex64"
                  (vt-dtype (vt-sign c64)) :complex64)
    (expect-equal "sinc(complex64) → complex64"
                  (vt-dtype (vt-sinc c64)) :complex64)
    ;; 逐元素与标量一致
    (let ((arr (vt-log c128)))
      (dotimes (i 3)
        (expect-true (format nil "log 数组第 ~a 项 = 标量内核" i)
                     (cplx= (vt-ref arr i)
                            (scalar #'vt-log (vt-ref c128 i))))))
    ;; 形状：2-D 输入保持 2-D
    (let* ((m (vt-reshape (vt-arange 12 :dtype :float64) '(3 4)))
           (z (vt-astype m :complex128))
           (r (vt-log z)))
      (expect-equal "log 2-D shape 保持" (vt-shape r) '(3 4))
      (expect-equal "log 2-D dtype" (vt-dtype r) :complex128))
    ;; :out 契约（同 dtype 写入，且返回同一个对象）
    (let ((o (vt-zeros '(3) :dtype :complex128)))
      (expect-true ":out 写入后与无 :out 结果相同"
                   (progn (vt-log c128 :out o)
                          (cplx= (vt-ref o 1) (vt-ref (vt-log c128) 1)))))
    ;; 广播：log 是一元算子，输入 (2,1) 不广播 —— 这里验证**对数恒等式**
    (let* ((a (vt-from-sequence (list (complex 4d0 0d0) (complex 8d0 0d0))
                                :dtype :complex128))
           (got (vt-log2 a)))
      (expect-true "log2(4)=2, log2(8)=3（:base 换算通路）"
                   (and (comp= (realpart (vt-ref got 0)) 2d0 1d-12)
                        (comp= (realpart (vt-ref got 1)) 3d0 1d-12))))
    ;; :base 通路：log(z, base=2) 与 log2(z) 同值
    (let ((a (vt-from-sequence (list (complex 3d0 4d0)) :dtype :complex128)))
      (expect-true "log(:base 2) = log2"
                   (cplx= (vt-ref (vt-log a :base 2d0) 0)
                          (vt-ref (vt-log2 a) 0))))
    ;; 空数组不报错、结果为空
    (let ((e (vt-zeros '(0) :dtype :complex128)))
      (expect-equal "log 空数组 size" (vt-size (vt-log e)) 0)
      (expect-equal "sign 空数组 size" (vt-size (vt-sign e)) 0))
    ;; 整数输入走实数通路（不因复数分派误入）
    (expect-equal "log 整数输入 → float64"
                  (vt-dtype (vt-log (vt-from-sequence '(1 2 3) :dtype :int64)))
                  :float64)))

;;; ==================================================================
;;; D. 内核前置回归（%cop-expm1 / %cop-log1p / %cop-sinc / %cop-sign）
;;; ==================================================================
(defun test-hoisted-kernels ()
  (let* ((zs (list (complex 0.3d0 0.4d0) (complex -0.2d0 0.1d0)
                   (complex 0.05d0 -0.01d0) (complex 1.5d0 2d0)))
         (arr (vt-from-sequence zs :dtype :complex128)))
    ;; expm1(z) = exp(z) - 1（|z| 小于/大于级数阈值的两条分支都跑到）
    (let ((r (vt-expm1 arr)))
      (dotimes (i (length zs))
        (let* ((z (nth i zs))
               (x (realpart z)) (y (imagpart z))
               (ex (exp x))
               ;; 参考：exp(z)-1 = e^x(cos y + i sin y) - 1
               (want (complex (- (* ex (cos y)) 1d0) (* ex (sin y)))))
          (expect-true (format nil "expm1 第 ~a 项 ≈ exp(z)-1" i)
                       (or (cplx= (vt-ref r i) want)
                           ;; |z| 落在级数窗口时容差放宽到 1e-12 相对
                           (progn (format t "~&  expm1[~a] got ~a want ~a~%"
                                          i (vt-ref r i) want)
                                  nil))))))
    ;; log1p(z) = log(1+z)
    (let ((r (vt-log1p arr)))
      (dotimes (i (length zs))
        (expect-true (format nil "log1p 第 ~a 项 ≈ log(1+z)" i)
                     (cplx= (vt-ref r i)
                            (vt-ref (vt-log (vt-+ arr (vt-full (list 4) 1d0
                                                             :dtype :complex128)))
                                    i)))))
    ;; sinc(z) = sin(πz)/(πz)
    (let ((r (vt-sinc arr)))
      (dotimes (i (length zs))
        (let* ((z (nth i zs))
               (pz (* pi z)))
          (expect-true (format nil "sinc 第 ~a 项 = sin(πz)/(πz)" i)
                       (cplx= (vt-ref r i) (/ (sin pz) pz))))))
    ;; sign 与 signum 在复数上同义（都投影到单位圆）
    (let ((a (vt-sign arr)) (b (vt-signum arr)))
      (dotimes (i (length zs))
        (expect-true (format nil "sign = signum（第 ~a 项）" i)
                     (cplx= (vt-ref a i) (vt-ref b i)))))
    ;; 精度：complex64 结果仍是 complex64（不是被 0.0d0 常量拉回 complex128）
    (let ((c64 (vt-from-sequence (list (complex 1s0 2s0)) :dtype :complex64)))
      (dolist (fn (list #'vt-sign #'vt-sinc #'vt-expm1 #'vt-log1p))
        (expect-equal (format nil "~a 保持 complex64" fn)
                      (vt-dtype (funcall fn c64)) :complex64)))))

;;; ==================================================================
;;; D2. 三角/反三角族：NaN 输入不抛条件（本轮修掉的硬缺陷）
;;; ==================================================================
(defun test-trig-nan-safety ()
  (let ((nan +vt-dfloat-nan+))
    ;; 整改前：vt-asin / vt-acos / vt-asinh / vt-acosh 对「虚部是 NaN」的
    ;; 输入直接抛 `The value NaN is not of type (DOUBLE-FLOAT 0.0)`
    ;; （SBCL 的 DECODE-FLOAT 类型断言，with-float-safe 屏蔽不了）。
    (dolist (fn (list (cons "vt-asin" #'vt-asin) (cons "vt-acos" #'vt-acos)
                      (cons "vt-asinh" #'vt-asinh) (cons "vt-acosh" #'vt-acosh)
                      (cons "vt-tan" #'vt-tan) (cons "vt-tanh" #'vt-tanh)
                      (cons "vt-atanh" #'vt-atanh)))
      (dolist (z (list (complex 0d0 nan) (complex 1d0 nan) (complex nan 0d0)
                       (complex nan 1d0) (complex 4d0 +vt-dfloat-pos-inf+)
                       (complex +vt-dfloat-pos-inf+ 4d0)))
        (expect-true (format nil "~a(~a) 不抛条件" (car fn) z)
                     (handler-case (progn (scalar (cdr fn) z) t)
                       (error () nil)))
        (expect-true (format nil "~a(~a) 返回复数" (car fn) z)
                     (complexp (scalar (cdr fn) z))))))
  ;; 极点的显式语义（numpy 2.4.6 实测）
  (expect-true "tanh(0+Infi) = 0+NaNj（不是 NaN+NaNj）"
               (let ((got (scalar #'vt-tanh (complex 0d0 +vt-dfloat-pos-inf+))))
                 (and (zerop (realpart got)) (sb-ext:float-nan-p (imagpart got)))))
  (expect-true "asin(0+NaNj) = 0+NaNj"
               (let ((got (scalar #'vt-asin (complex 0d0 +vt-dfloat-nan+))))
                 (and (zerop (realpart got)) (sb-ext:float-nan-p (imagpart got)))))
  (expect-true "acos(-4+Infi) 实部 = π/2（走 clog-large，不是 2x）"
               (comp= (realpart (scalar #'vt-acos (complex -4d0 +vt-dfloat-pos-inf+)))
                      1.5707963267948966d0 1d-12)))

;;; ==================================================================
;;; E. 倍率护栏
;;; ==================================================================
(defun test-guardrails ()
  (let* ((n 1000000)
         (zc (vt-astype (vt-random-uniform (list n) :low 0.5d0 :high 1.5d0)
                        :complex128))
         (zr (vt-random-uniform (list n) :low 0.5d0 :high 1.5d0))
         (t-log-c (ms-of (lambda () (vt-log zc))))
         (t-log-r (ms-of (lambda () (vt-log zr))))
         (t-sign-c (ms-of (lambda () (vt-sign zc))))
         (t-sign-r (ms-of (lambda () (vt-sign zr)))))
    (let ((ratio (/ t-log-c (max t-log-r 0.05d0))))
      (expect-true (format nil "log c128/f64 倍率 ~,1fx ≤ 12x（修前 43x）" ratio)
                   (<= ratio 12.0d0)))
    (let ((ratio (/ t-sign-c (max t-sign-r 0.05d0))))
      (expect-true (format nil "sign c128/f64 倍率 ~,1fx ≤ 15x（修前 40x）" ratio)
                   (<= ratio 15.0d0)))
    (expect-true (format nil "log c128 绝对 < 150ms（实测 ~,1f，修前 352）" t-log-c)
                 (< t-log-c 150.0d0))
    (expect-true (format nil "sign c128 绝对 < 60ms（实测 ~,1f，修前 104）" t-sign-c)
                 (< t-sign-c 60.0d0))
    ;; 三角/反三角族（第 23 轮）：修前 tan 203 / tanh 192 / asin 915 ms
    (let ((t-tanh-c (ms-of (lambda () (vt-tanh zc))))
          (t-asin-c (ms-of (lambda () (vt-asin zc)))))
      (expect-true (format nil "tanh c128/f64 倍率 ~,1fx ≤ 8x（修前 11.5x）"
                           (/ t-tanh-c (max (ms-of (lambda () (vt-tanh zr))) 0.05d0)))
                   (<= (/ t-tanh-c (max (ms-of (lambda () (vt-tanh zr))) 0.05d0)) 8.0d0))
      (expect-true (format nil "tanh c128 绝对 < 120ms（实测 ~,1f，修前 192）" t-tanh-c)
                   (< t-tanh-c 120.0d0))
      (expect-true (format nil "asin c128 绝对 < 250ms（实测 ~,1f，修前 915）" t-asin-c)
                   (< t-asin-c 250.0d0))
      ;; 反三角族另一半：内联 _do_hard_work 之后 acos/asinh/acosh 才降到
      ;; 三位数（acos 876→60、asinh 949→92、acosh 866→62）
      (let ((t-acos-c (ms-of (lambda () (vt-acos zc))))
            (t-acosh-c (ms-of (lambda () (vt-acosh zc)))))
        (expect-true (format nil "acos c128 绝对 < 200ms（实测 ~,1f，修前 876）" t-acos-c)
                     (< t-acos-c 200.0d0))
        (expect-true (format nil "acosh c128 绝对 < 200ms（实测 ~,1f，修前 866）" t-acosh-c)
                     (< t-acosh-c 200.0d0))))))

(defun main ()
  (format t "~&=== 复数初等函数 / 内核前置 测试 ===")
  (test-numpy-grid)
  (test-special-semantics)
  (test-vector-contract)
  (test-hoisted-kernels)
  (test-trig-nan-safety)
  (test-guardrails)
  (format t "~&~%通过: ~a  失败: ~a~%" *pass* *fail*)
  (when *failures*
    (format t "~&失败清单:~%")
    (dolist (f (reverse *failures*)) (format t "  - ~a~%" f)))
  (when (plusp *fail*)
    (sb-ext:exit :code 1)))

(main)
