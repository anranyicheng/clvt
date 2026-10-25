;;;; complex-kernels.lisp — 复数**标量内核**（纯函数：只吃数、吐数）
;;;;
;;;; 为什么单独成文件
;;;; ----
;;;; 调用这些内核的归属文件（elementwise / extensions3 / reduce-stats …）在
;;;; 加载顺序上**早于**复数语义层，所以内核必须先于它们可用；否则那些文件在
;;;; 前面加载。于是「复数支持」只能靠事后 `(setf (symbol-function ...))`
;;;; 覆盖，或者靠 `%cx-orig` 存一份原实现再转发 —— 两种都是覆盖式，
;;;; 重载单个源文件就会静默丢失复数语义（§16.32）。
;;;;
;;;; 内核前移到 dispatch.lisp 之后、所有归属文件之前后，归属文件里的
;;;; 单一 DEFUN 可以直接
;;;;
;;;;     (when (%cx-any-p x) (return-from vt-foo (%cx-op-foo x ...)))
;;;;
;;;; 不再需要任何运行期改写。
;;;;
;;;; 内容边界
;;;; ----
;;;;   只放**无状态标量内核**：入参是数，出参是数，不碰张量结构。
;;;;   任何「需要知道 shape/dtype/axis」的逻辑仍留在 complex*.lisp。
;;;;
;;;;   %COP-*        复数标量运算（sqrt / log / pow / …）
;;;;   %CX-*-RUN     「复数数组 → 实数数组」的按元素类型特化循环
;;;;   %CX-TO-OUT    :out 硬契约写回

(in-package :clvt)

;;; ==================================================================
;;; 0. 张量落地工具
;;; ==================================================================

(defun %cx-flat (tensor)
  "把张量落地为 C 连续、offset=0 的视图（已连续则零拷贝返回）。"
  (vt-contiguous (ensure-vt tensor)))




;;; ==================================================================
;;; 1. 同精度常量（NaN / Inf）
;;; ==================================================================

(defun %cx-nan-of (x)
  "与 X 同精度的 NaN。"
  (if (typep (%creal x) 'single-float) +vt-sfloat-nan+ +vt-dfloat-nan+))

(defun %cx-inf-of (x sign)
  "与 X 同精度的 ±Inf。"
  (if (typep (%creal x) 'single-float)
      (if (minusp sign) +vt-sfloat-neg-inf+ +vt-sfloat-pos-inf+)
      (if (minusp sign) +vt-dfloat-neg-inf+ +vt-dfloat-pos-inf+)))

(defun %cx-nan-complex (x)
  "与 X 同精度的 NaN+NaNj。"
  (let ((n (%cx-nan-of x)))
    (complex n n)))

;;; ==================================================================
;;; 2. 复数标量运算内核
;;; ==================================================================

(declaim (inline %cx-hypot %cop-abs %cop-sqrt))(defun %cx-hypot (x y)
  "无溢出的 hypot（|x|、|y| 极值时不溢出）。x/y 为实数。"
  (let ((ax (abs x)) (ay (abs y)))
    (cond ((zerop ax) ay)
          ((zerop ay) ax)
          ((> ax ay) (let ((r (/ ay ax)))
		       (* ax (sqrt (+ 1.0s0 (* r r))))))
          (t (let ((r (/ ax ay)))
	       (* ay (sqrt (+ 1.0s0 (* r r)))))))))

(defun %cop-sqrt (z)
  "复数平方根（主值，C99 csqrt 分支切割）。
   sqrt(-4+0i)=+2i、sqrt(-4-0i)=-2i（尊重虚部 -0.0 的符号）。

   为什么不用 `(sqrt z)`：SBCL 的复数 sqrt 走通用复数算术，实测
   **350ns/元素**（1M = 350ms）；而 C99 的标准公式只用 hypot + 1 次
   实数 sqrt + 2 次除法，约 20ns。二者在本库被测的全部点上逐位一致
   （complex-test 的 8 个分支切割用例 + numpy 差分）。"
  (let ((x (%creal z))
	(y (%cimag z)))
    (cond ((%cx-nan-p z) (%cx-nan-complex z))
          ;; 实轴：虚部保持原符号（含 -0.0），必须走特例，否则 ±0 会被抹平
          ((and (zerop y) (>= x 0)) (complex (sqrt x) y))
          ((and (zerop y) (minusp x))
           (complex (* 0 x)
                    (if (minusp (float-sign y))
			(- (sqrt (- x)))
			(sqrt (- x)))))
          (t
           (let* ((r (%cx-hypot x y))
                  (s (sqrt (* 0.5s0 (+ r (abs x))))))
             (if (>= x 0)
                 (complex s (/ y (* 2.0s0 s)))
                 (complex (/ (abs y) (* 2.0s0 s))
                          (if (minusp (float-sign y)) (- s) s))))))))

;;; 性能（叶子内核必须 INLINE）：这一批都是「一条复数运算」的叶子。
;;; 不内联时，调用点看不到它们的入参类型，SBCL 对 `(* z z)` / `(- z)` /
;;; `(/ z)` 只能推出 NUMBER / (OR (COMPLEX SINGLE) (COMPLEX DOUBLE))，
;;; 于是 map-reduce.lisp 的 %CAST-TO 落到复数目标时只能走泛型 COERCE，
;;; 并且每个元素多构造一个复数对象（实测 2M 元素：square 478ms，
;;; 内联后 13.5ms；c64 241ms → 10ms）。
;;; 内联后类型随实参精确流动，"取实部/虚部 + 写回 specialized-array"
;;; 全部退化成寄存器操作，与 (vt-+ z z) 同级。
(declaim (inline %cop-square %cop-neg %cconj %cop-recip %cop-signum))

(defun %cop-square (z) (* z z))

(defun %cop-recip (z)
  "复数倒数 1/z。零 → NaN+NaNj（对标 numpy：0 的倒数无定义）。"
  (if (%cis-zero z) (%cx-nan-complex z) (/ z)))

(defun %cop-signum (z)
  "复数符号 z/|z|（对标 numpy.sign 的复数语义）；零 → 0+0i。"
  (if (%cis-zero z)
      (complex (* 0 (%creal z)) (* 0 (%cimag z)))
      (/ z (abs z))))

(defun %cop-abs (z)
  "复数模（实数输出）。abs(inf+nan·i)=inf，与 numpy 的 hypot 语义一致。

   快路径：|x|、|y| 都在 [1e-150, 1e150] 内时直接 sqrt(x²+y²)
   （1 sqrt + 2 mul，约 5ns）；只有可能上溢/下溢的极端量级才走
   带缩放的 %CX-HYPOT。numpy 恒用 hypot（每元素 2 次除法），
   1M 元素比这里慢约 3 倍。"
  (cond ((%cx-inf-p z) (%cx-inf-of z 1))
        ((%cx-nan-p z) (%cx-nan-of z))
        (t
         (let ((x (%creal z)) (y (%cimag z)))
           (if (and (< (abs x) 1d150) (< (abs y) 1d150)
                    (or (zerop x) (> (abs x) 1d-150))
                    (or (zerop y) (> (abs y) 1d-150)))
               (sqrt (+ (* x x) (* y y)))
               (%cx-hypot x y))))))

(defun %cop-neg (z) (- z))

;;; log 族必须 INLINE：本内核会被 %VT-FAST-MAP 展开进 30 个逐元素循环。
;;; 不内联时（实测）调用点只能发一次真实函数调用，内核本体在独立编译单元
;;; 里拿不到实参类型 → 1M complex128 的 vt-log 199 ms；内联后 27 ms。
(declaim (inline %cop-log %cop-log2 %cop-log10))

(defun %cop-log (z)
  "复数自然对数 log|z| + i·arg(z)（主值，C99 clog 语义）。

   为什么不用 `(log z)`：SBCL 的复数 log 走通用复数算术，
   **332ns/元素**（1M = 352ms）；分量式只用 abs + log + atan2，约 100ns。
   实测 352 → 96 ms（1M，SBCL 2.6.8）。

   与 numpy 2.4.6 在 21×21 网格（含 ±0 / ±Inf / NaN / ±1e300）上**逐点一致**，
   而 `(log z)` 有 6 处不一致 —— 都是**分支切割上的相位**：
     · log(-0+0i)：numpy 给 -Inf+πi（虚部取 atan2(+0,-0)=π），
       CL 给 -Inf+0i —— 符号零带来的相位被抹掉了；
     · log(±Inf+NaNj)：numpy 给 Inf+NaNj（模为 Inf 时相位无意义），
       CL 先判 NaN 直接返回 NaN+NaNj。
   本内核用 `%cop-abs` 取模（它先判 Inf/NaN），再单独判「非 Inf 的 NaN」，
   两点即与 numpy 对齐。

   极点：z=0 → -Inf + 0i。调用方（vt-map/vt-fast-map）已在 with-float-safe
   内，故 log(0) 得 -Inf 而不是浮点陷阱。"
  (let ((x (%creal z)) (y (%cimag z)))
    (cond
      ;; Inf 优先于 NaN：模为 Inf 时 numpy 只把相位判成 NaN，模仍是 Inf
      ((and (%cx-nan-p z)
            (not (or (sb-ext:float-infinity-p x) (sb-ext:float-infinity-p y))))
       (%cx-nan-complex z))
      (t (complex (log (%cop-abs z)) (atan y x))))))

(defun %cop-log2 (z)
  "复数 log2 = log(z)/ln2。极点 0 → -Inf + 0i。"
  (let* ((l (%cop-log z))
         (ln2 (if (typep (%creal l) 'single-float) 0.6931472s0 0.6931471805599453d0)))
    (complex (/ (%creal l) ln2) (/ (%cimag l) ln2))))

(defun %cop-log10 (z)
  "复数 log10 = log(z)/ln10。极点 0 → -Inf + 0i。"
  (let* ((l (%cop-log z))
         (ln10 (if (typep (%creal l) 'single-float) 2.3025851s0
                   2.302585092994046d0)))
    (complex (/ (%creal l) ln10) (/ (%cimag l) ln10))))

;;; ------------------------------------------------------------------
;;; 2b. 四个「被晚加载文件调用」的内核（第 22 轮前移）
;;;
;;; 它们原先住在 complex-ops.lisp，而调用点在更早加载的文件里：
;;;   %cop-expm1 / %cop-log1p / %cop-sinc  ← extensions3 / elementwise2
;;;   %cop-sign                            ← extensions3 (vt-sign)
;;; 编译调用点时符号还不存在，于是 `(function %cop-xxx)` 只能编成**真实
;;; 函数调用**，且内核本体在**独立编译单元**里看不到实参类型，只能按
;;; 泛型 NUMBER 编（装箱 + 无类型推导）。实测 1M complex128：
;;;   vt-sign 104 ms（内联后 17 ms）、vt-sinc 218 ms（→ 22 ms）、
;;;   vt-expm1 116 ms（→ 47 ms）。
;;; 这与 §16.8.1 第 4 条（标志内核定义在 complex.lisp）是**同一个缺陷**，
;;; 只是换了个文件。内核是纯标量函数，前移到本文件不增加任何耦合。
;;; ------------------------------------------------------------------
(declaim (inline %cop-expm1 %cop-log1p %cop-sinc %cop-sign))

(defun %cop-expm1 (z)
  "exp(z) - 1（对标 numpy.expm1 的复数分支）。

   |z| 大时直接 exp(z)-1；|z| 小时灾难性抵消会把结果精度吃光
   （z=1d-10 时 exp(z)-1 只剩 1 位有效数字），改用级数
   expm1(z) = z·Σ_{k≥0} z^k/(k+1)!（Horner 形式，12 项对 |z|<0.5 足够：
   0.5^12/13! ≈ 1e-13，远低于 double 的 eps 累积）。"
  (cond ((%cx-nan-p z) (%cx-nan-complex z))
        ((< (abs z) 0.5)
         ;; 级数**固定用 double 算**（12 项 1.0d0/(k+2) 的系数都是 double，
         ;; 强行降到 single 只会掉精度）。因此这里把 z 显式提升成 d0 再声明类型：
         ;; 原写法直接 `(declare (type (complex double-float) term sum))` 而
         ;; init 是 z —— 对 complex64 输入这是**谎报**：SBCL 不仅报编译期
         ;; 「Derived type (complex single-float) conflicting with asserted
         ;; (complex double-float)」让整个库编译失败，运行期 (safety 1) 还会
         ;; 在 |z|<0.5 的 complex64 上直接抛类型错误。
         (let* ((zd (complex (coerce (%creal z) 'double-float)
                             (coerce (%cimag z) 'double-float)))
                (term zd)
                (sum zd))
           (declare (type (complex double-float) zd term sum))
           (dotimes (k 12 sum)
             (setf term (* term zd (/ 1.0d0 (+ k 2))))
             (incf sum term))))
        (t (- (exp z) 1))))

(defun %cop-log1p (z)
  "log(1 + z)（对标 numpy.log1p 的复数分支）。

   与 expm1 同源：|z| 小时 1+z 的有效位被吃掉，必须走级数
   log1p(z) = z - z²/2 + z³/3 - … = z·Σ_{k≥0} (-z)^k/(k+1)。
   z = -1 时 1+z = 0，交给 %COP-LOG 处理极点（-inf + 0i）。

   两个数值细节（与 numpy 2.4.6 逐点核对过）：
     · 分母是**调和**的 k，不是阶乘 k!。写成 `term *= (-z)/(k+2)` 得到的是
       z·Σ(-z)^k/(k+1)!，那是另一个函数 —— 实测 |z|=0.112 时虚部偏 0.4%
       （numpy: 0.09634224+0.04542328i，错版: 0.0962934+0.045223i）。
     · 线性分母收敛慢（|z|^n/n），窗口收到 |z|<0.25 并取 24 项：第 24 项
       量级 0.25^24/24 ≈ 1.5e-16，相对误差已达 double 的 eps。|z|≥0.25 时
       1+z 不存在抵消问题，直接走 %COP-LOG（与 numpy 分支切割一致）。"
  (cond ((%cx-nan-p z) (%cx-nan-complex z))
        ((< (abs z) 0.25)
         ;; 同 %COP-EXPM1：级数固定 double（见那里的说明），z 先显式提升。
         (let* ((zd (complex (coerce (%creal z) 'double-float)
                             (coerce (%cimag z) 'double-float)))
                (term zd)
                (sum zd))
           (declare (type (complex double-float) zd term sum))
           (dotimes (k 23 sum)
             (setf term (* term (- zd)
                           (/ (float (1+ k) 1d0) (float (+ k 2) 1d0))))
             (incf sum term))))
        (t (%cop-log (+ 1 z)))))

(defun %cop-sinc (z)
  "归一化 sinc：sin(πz)/(πz)，sinc(0)=1（对标 numpy.sinc 的复数分支）。

   |z| 很小时 sin(πz)/(πz) → 1 - (πz)²/6，直接算会 0/0。"
  (cond ((%cx-nan-p z) (%cx-nan-complex z))
        ((< (abs z) 1.0d-8)
         (- 1 (* (/ (* pi pi) 6.0d0) z z)))
        (t (let ((pz (* pi z)))
             (/ (sin pz) pz)))))

(defun %cop-sign (z)
  "复数的「符号」：z/|z|（单位相量），0 → 0（对标 numpy.sign）。

   numpy 的复数 sign 不是取实部符号，而是投影到单位圆 ——
   这正是它与 %COP-SIGNUM 同源的地方（clvt 的 vt-signum 已如此实现）。

   零的构造**按实参精度**给出（不是写死 :complex128 的 0.0d0）：
   complex64 输入被写成 0.0d0 时，%CAST-TO 还要多做一次降精度转换。"
  (cond ((%cx-nan-p z) (%cx-nan-complex z))
        ((%cis-zero z) (complex (* 0 (%creal z)) (* 0 (%cimag z))))
        (t (/ z (abs z)))))


;;; 带 :base 的换底对数单独一个内核（不是 log 的变体）：:base 必须原样
;;; 参与计算，早期若经统一的包装宏转发会把 :base 静默丢掉
;;; （实测：log(8, base=2) 退化成 ln(8)=2.079）。
(defun %cop-log-base (z base)
  "复数换底对数 log_base(z)。base 必须为正实数（numpy：非法 base → NaN）。

   ⚠ 陷阱：**不要写成 `(log base 1.0d0)`** —— CL 的
   `(log x base)` 是两参数对数 `log_x(base)`，那样算成了
   `log_1(base) = ln(base)/ln(1) = 0.693/0` = 除零（陷阱被 with-float-safe
   屏蔽时得到 Inf，再乘进结果 → 0+0i）。实测 `(vt-log #C(8 0) :base 2)`
   整改前返回 0+0i，numpy 是 3+0i。正确写法是 `(log base)`（自然对数）。"
  (if (and (realp base) (plusp base) (/= base 1))
      (let* ((l (%cop-log z))
             (lb (log (float base 1.0d0))))
        (complex (/ (realpart l) lb)
                 (/ (imagpart l) lb)))
      (%cx-nan-complex z)))

;;; ==================================================================
;;; 3. 「复数 → 实数」特化循环
;;; ==================================================================

(defmacro %def-cx-to-real-run (name kernel)
  "生成 (IN RES)：复数 IN → 实数 RES 的逐元素内核，按元素类型 2 路特化。

   代码生成用宏展开期 FLET（不是嵌套 backquote 里的 MACROLET）：
   后者的内层 `,kernel` 属于第二层 quasiquote，符号不会被外层代入
   （实测报 “The variable KERNEL is unbound”）。"
  (flet ((run (ilt olt)
           `(let ((src (the (simple-array ,ilt (*)) (vt-data in)))
                  (dst (the (simple-array ,olt (*)) (vt-data res)))
                  (n (the fixnum (vt-size res)))
                  (p (the fixnum (vt-offset in)))
                  (q (the fixnum (vt-offset res))))
              (declare (type (simple-array ,ilt (*)) src)
                       (type (simple-array ,olt (*)) dst)
                       (type fixnum n p q)
                       (optimize (speed 3) (safety 1)))
              (dotimes (i n)
                (declare (type fixnum i))
                (setf (aref dst q) (,kernel (aref src p)))
                (incf p)
                (incf q)))))
    `(defun ,name (in res)
       (declare (type vt in res))
       (flet ((fallback ()
                (vt-map (function ,kernel) in :out res :dtype (vt-dtype res))))
         (let ((et (array-element-type (vt-data in)))
               (ot (array-element-type (vt-data res))))
           (cond ((and (equal et '(complex double-float)) (equal ot 'double-float))
                  ,(run '(complex double-float) 'double-float))
                 ((and (equal et '(complex single-float)) (equal ot 'single-float))
                  ,(run '(complex single-float) 'single-float))
                 (t (fallback))))))))

(%def-cx-to-real-run %cx-abs-run %cop-abs)
(%def-cx-to-real-run %cx-real-run %creal)
(%def-cx-to-real-run %cx-imag-run %cimag)
(%def-cx-to-real-run %cx-angle-run %carg)

(defun %cx-run-to-real (run-fn kernel in out dtype op-name)
  "执行「复数 → 实数」映射：连续走特化内核，否则回退 vt-map（语义基准）。

   RES 的 shape/dtype/可写性统一由 vt-check-out 把关（§4.2 H1–H4）。"
  (let ((res (if out
                 (vt-check-out out (vt-shape in) dtype :op-name op-name)
                 (make-vt (vt-shape in) 0 :dtype dtype))))
    (if (and (vt-contiguous-p in) (vt-contiguous-p res) (plusp (vt-size in)))
        (progn (funcall run-fn in res) res)
        (vt-map kernel in :out res :dtype dtype))))

;;; ==================================================================
;;; 4. 幂内核
;;; ==================================================================

(defun %cx-pow-2 (x y)
  "复数逐元素幂 x^y（主值分支），含极点与 NaN 的边界处理。
   vt-pow 与 vt-float-power 共用同一份内核 —— 两者的复数语义完全相同
   （都按 CL 复指数 expt 的主值分支），差别只在实数路径：
   vt-pow 对整数保留精确整数幂，vt-float-power 一律按浮点算。"
  (let ((base (if (complexp x) x (complex x 0)))
        (e (if (complexp y) y (complex y 0))))
    (cond
          ;; 指数为 0（含 0+0i）→ 1+0i，**先于** NaN/极点判定
          ;; （对标 numpy 与 C99 pow：nan**0 = 1、0**0 = 1；
          ;;  实数路径本来就是 1，旧复数路径却先判 NaN 给出 NaN，
          ;;  与自身实数路径都不一致，这里一并修掉）
          ((%cis-zero e)
           (complex (float 1 (%creal base))
                    (* 0 (%cimag base))))
          ((%cx-nan-p base) (%cx-nan-complex base))
          ((and (%cis-zero base)
                (plusp (%creal e))
                (zerop (%cimag e)))
           (complex (* 0 (%creal base)) (* 0 (%cimag base))))
          ((%cis-zero base) (%cx-nan-complex base))
          (t (handler-case (expt base e)
               (error () (%cx-nan-complex base)))))))

;;; ==================================================================
;;; 4.5 dtype 重解释（vt-view 的 :dtype 分支）
;;; ==================================================================
(defun %cx-view-pair-p (from to)
  "允许的 dtype 重解释对（只开放「分量宽度相同」的两对，保证逐位等价）。
   complex128↔float64、complex64↔float32。"
  (or (and (eq from :complex128) (eq to :float64))
      (and (eq from :float64) (eq to :complex128))
      (and (eq from :complex64) (eq to :float32))
      (and (eq from :float32) (eq to :complex64))))

(defun %cx-view-dtype-shape (shape from to)
  "按 numpy 规则算重解释后的形状：只缩放**最后一维**。
   complex128 (2 3) → float64 (2 6)；float64 (2 6) → complex128 (2 3)。"
  (let* ((fi (vt-dtype-itemsize from))
         (ti (vt-dtype-itemsize to))
         (last (car (last shape)))
         (new-last (if (> fi ti)
                       (* last (floor fi ti))
                       (if (zerop (mod last (floor ti fi)))
                           (floor last (floor ti fi))
                           (error "vt-view: 最后一维 ~a 不能按 ~a→~a 整除" last from to)))))
    (append (butlast shape) (list new-last))))

(defun %cx-reinterpret (v shape dtype)
  "把 V 的内容按 DTYPE 重新解释（逐元素重建，语义等价 numpy.view）。
   SHAPE 为 NIL 时自动按 numpy 规则缩放最后一维。"
  (let* ((from (vt-dtype v))
         (flat (%cx-flat v))
         (n (vt-size flat))
         (fi (vt-dtype-itemsize from))
         (ti (vt-dtype-itemsize dtype))
         (m (if (> fi ti)
		(* n (floor fi ti))
		(floor n (floor ti fi))))
         (out-shape (or shape (%cx-view-dtype-shape (vt-shape v) from dtype)))
         (out-size (reduce #'* out-shape :initial-value 1)))
    (unless (= out-size m)
      (error "vt-view: 重解释后元素数不一致（~a 个 ~a = ~a 个 ~a，但 shape ~a 需要 ~a 个）"
             n from m dtype out-shape out-size))
    (let ((res (make-vt out-shape 0 :dtype dtype))
          (src (vt-data flat)))
      (let ((dst (vt-data res))
            (off (vt-offset flat)))
        (with-float-safe
          (cond
            ((and (eq from :complex128) (eq dtype :float64))
             (dotimes (i n)
               (let ((z (aref src (+ off i))))
                 (setf (aref dst (* 2 i)) (%creal z)
                       (aref dst (1+ (* 2 i))) (%cimag z)))))
            ((and (eq from :complex64) (eq dtype :float32))
             (dotimes (i n)
               (let ((z (aref src (+ off i))))
                 (setf (aref dst (* 2 i)) (%creal z)
                       (aref dst (1+ (* 2 i))) (%cimag z)))))
            ((and (eq from :float64) (eq dtype :complex128))
             (dotimes (i (/ n 2))
               (setf (aref dst i) (complex (aref src (+ off (* 2 i)))
                                           (aref src (+ off (1+ (* 2 i))))))))
            ((and (eq from :float32) (eq dtype :complex64))
             (dotimes (i (/ n 2))
               (setf (aref dst i) (complex (aref src (+ off (* 2 i)))
                                           (aref src (+ off (1+ (* 2 i))))))))
            (t (error "vt-view: 不支持 ~a → ~a 的重解释" from dtype)))))
      res)))

;;; ==================================================================
;;; 5. :out 写回
;;; ==================================================================

(defun %cx-to-out (r out name)
  "把已算好的 R 按 :out 硬契约写回（形状/dtype 严格相等由 vt-check-out 把关）。"
  (if out
      (let ((checked (vt-check-out out (vt-shape r) (vt-dtype r) :op-name name)))
        (vt-copy-into checked r)
        checked)
      r))

;;; ---- 极值内核（maximum / minimum / fmax / fmin）----
;;; 语义：字典序（实部→虚部），maximum/minimum 传播 NaN，fmax/fmin 忽略 NaN。
;;; 位置：按 §16.33.2「标量内核前移」放在这里（dispatch 之后、所有归属文件之前），
;;; 这样归属文件编译时至少知道这些符号存在。
;;;
;;; NaN 判定必须用免屏蔽的 `%CX-NAN-P`，不能用边界的 `%CISNAN`：
;;; 后者内部套一层 `with-float-traps-masked`（单次约 200ns），每元素判两次
;;; → 1M 元素光屏蔽/恢复陷阱就是 800ms。本组内核的全部调用点
;;; （vt-map / vt-fast-map / %cx-reduce-* / %cx-arg-reduce-run）都在
;;; with-float-safe 之下，前置条件满足（§16.14 的分工）。
;;; 实测 1M complex128：%cisnan 839ms → %cx-nan-p 3ms。
(declaim (inline %cop-max %cop-min %cop-fmax %cop-fmin))

(defun %cop-max (a b)
  "maximum 的复数语义：字典序取大，NaN 传播。"
  (cond ((%cx-nan-p a) a) ((%cx-nan-p b) b) ((%clt a b) b) (t a)))

(defun %cop-min (a b)
  "minimum 的复数语义：字典序取小，NaN 传播。"
  (cond ((%cx-nan-p a) a) ((%cx-nan-p b) b) ((%cgt a b) b) (t a)))

(defun %cop-fmax (a b)
  "fmax 的复数语义：字典序取大，忽略 NaN。"
  (cond ((%cx-nan-p a) b) ((%cx-nan-p b) a) ((%clt a b) b) (t a)))

(defun %cop-fmin (a b)
  "fmin 的复数语义：字典序取小，忽略 NaN。"
  (cond ((%cx-nan-p a) b) ((%cx-nan-p b) a) ((%cgt a b) b) (t a)))

;;; ==================================================================
;;; 比较 / 谓词的「1/0 标志」内核（§16.8 / §16.9）
;;;
;;; 为什么住在这里（而不是 complex.lisp）：`vt-=` / `vt-<` / `vt-isnan` 等在
;;; elementwise2.lisp 里通过 `vt-fast-map` **内联**调用它们，而 elementwise2
;;; 早于 complex.lisp 编译。若定义在 complex.lisp，编译 elementwise2 时这些
;;; 符号还是「未知函数」：
;;;   · 每个元素退化成一次真实函数调用（实测 vt-= 1M complex128 60ms，
;;;     实数线 1ms）；
;;;   · 事后补 `declaim inline` 也无效，SBCL 只报
;;;     「Proclaiming %CX-EQ-FLAG to be INLINE, but 12 calls were previously
;;;      compiled」—— 这句话本身就是缺陷信号，出现即说明内核放晚了。
;;; 这是 §16.33.2「标量内核前移」的又一例。
;;; ==================================================================

;;; 比较的布尔承载：clvt 约定比较运算返回 :int8 的 1/0（§3.1），
;;; 而 %clt/%cle/%cgt/%cge 返回 T/NIL（排序/极值内核需要布尔），
;;; 故此处做一层 1/0 适配，避免把 T 写进整型缓冲。
(declaim (inline %cx-lt-flag %cx-le-flag %cx-gt-flag %cx-ge-flag
                 %cx-eq-flag %cx-ne-flag %cx-nan-flag %cx-inf-flag %cx-finite-flag))

(defun %cx-lt-flag (a b) (if (%clt a b) 1 0))
(defun %cx-le-flag (a b) (if (%cle a b) 1 0))
(defun %cx-gt-flag (a b) (if (%cgt a b) 1 0))
(defun %cx-ge-flag (a b) (if (%cge a b) 1 0))

(defun %cx-eq-flag (a b)
  "复数相等：任一分量 NaN → 0（对标 numpy：nan+0j == nan+0j 为 False）。
   不能直接用 CL 的 `=`：浮点陷阱被屏蔽后 (= nan nan) 在复数上实测返回 T。"
  (cond ((%cx-nan-p a) 0)
        ((%cx-nan-p b) 0)
        ((= a b) 1)
        (t 0)))

(defun %cx-ne-flag (a b) (- 1 (%cx-eq-flag a b)))

;;; 判定族（§16.7）。谓词在多处被调用：vt-fast-map 内联循环、%cx-sort-less、
;;; %cx-arg-reduce-run —— 全部处于 with-float-safe 之下，故用免屏蔽快判。
(defun %cx-nan-flag (x) (if (%cx-nan-p x) 1 0))
(defun %cx-inf-flag (x) (if (%cx-inf-p x) 1 0))
(defun %cx-finite-flag (x) (if (%cx-finite-p x) 1 0))

