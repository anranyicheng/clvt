;;;; complex-kernels.lisp — 复数**标量内核**（第十三轮抽出）
;;;;
;;;; 为什么单独成文件
;;;; ----
;;;; 第十三轮之前这些内核住在 complex.lisp（:serial t 的最后几个文件之一），
;;;; 而调用它们的归属文件（elementwise / extensions3 / reduce-stats …）在
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

(declaim (inline %cx-hypot %cop-abs %cop-sqrt))
(defun %cx-hypot (x y)
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

(defun %cop-log (z)
  "复数自然对数 log|z| + i·arg(z)。极点 log(0) → -Inf + 0i（对标 numpy）。"
  (let ((x (%creal z)) (y (%cimag z)))
    (cond ((%cx-nan-p z) (%cx-nan-complex z))
          ((and (zerop x) (zerop y))
           (complex (%cx-inf-of z -1) y))
          (t (log z)))))

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

;;; 带 :base 的换底对数单独一个内核（不是 log 的变体）：:base 必须原样
;;; 参与计算，早期若经统一的包装宏转发会把 :base 静默丢掉
;;; （实测：log(8, base=2) 退化成 ln(8)=2.079）。
(defun %cop-log-base (z base)
  "复数换底对数 log_base(z)。base 必须为正实数（numpy：非法 base → NaN）。

   ⚠ 第十三轮修复：原实现写 `(log base 1.0d0)` —— CL 的
   `(log x base)` 是两参数对数 `log_x(base)`，于是这里算成了
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
