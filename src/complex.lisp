;;;; complex.lisp — 复数语义层（第一轮进化）
;;;;
;;;; 设计原则（CONVENTIONS §16）：
;;;;
;;;;  1) **物理层零改动**：复数沿用既有 strided view 模型 —— data 是
;;;;     (complex double-float) / (complex single-float) 特化数组，
;;;;     strides/offset 仍以「元素」为单位，实部/虚部 interleaved 存储。
;;;;     于是切片、转置、广播、非连续写入、别名快照全部自动正确。
;;;;
;;;;  2) **实数路径零改动**：本文件对每个受影响的公开函数在加载时捕获
;;;;     原实现（%cx-orig），只把「输入含复数」的调用接进复数内核；
;;;;     其余调用原样转发。实数语义与性能不受影响（§16.12）。
;;;;
;;;;  3) **未覆盖即显式报错**：第一轮不支持的函数（LU/SVD/百分位/位运算等）
;;;;     在入口处给出确定性错误，绝不静默丢虚部（§16.11）。
;;;;
;;;;  4) **单一语义出口**：所有判定/比较/取模经由 dtype.lisp 的
;;;;     %creal/%cimag/%cabs/%carg/%cisnan/%clt/… 实现，不在各处散写。

(in-package :clvt)

;;; ==================================================================
;;; 0. 原实现登记与复数参数探测
;;; ==================================================================

(defvar *cx-originals* (make-hash-table :test #'eq)
  "实数路径原实现表。首次加载时捕获；重复加载不会自我覆盖
   （否则包装器会转发给自己，造成无限递归）。")

(defun %cx-orig (name)
  "取 NAME 的**原实现**；首次调用时捕获并登记。"
  (or (gethash name *cx-originals*)
      (setf (gethash name *cx-originals*) (symbol-function name))))

;;; 加载期**预捕获**：用显式 defun 重新定义的那些函数（vt-abs / vt-sort /
;;; vt-matmul / vt-mean …）在函数体里通过 (%cx-orig 'name) 取原实现，
;;; 若等到运行时才捕获，此时 symbol-function 已经是新定义本身 ——
;;; 包装器转发给自己，形成无限递归（实测：实数路径 vt-sort 直接挂死）。
;;; 因此必须在**本文件任何 defun 之前**把所有将被覆盖的名字登记一次。
(eval-when (:load-toplevel :execute)
  (dolist (name '(vt-abs vt-real vt-imag vt-conj vt-angle
                  vt-round
                  vt-pow vt-expt vt-negative
                  vt-mean vt-var vt-std vt-average
                  vt-cumsum vt-cumprod
                  vt-sort vt-argsort vt-median
                  vt-norm vt-matmul vt-@ vt-dot vt-outer vt-trace
                  vt-argmax vt-argmin
                  ;; 宏包装的同名函数：这里先抓一遍，表里已有记录时
                  ;; %cx-orig 直接返回缓存，不会覆盖。
                  vt-sqrt vt-exp vt-log vt-log2 vt-log10
                  vt-sin vt-cos vt-tan vt-atan vt-sinh vt-cosh vt-tanh
                  vt-asin vt-acos vt-asinh vt-acosh vt-atanh
                  vt-square vt-reciprocal vt-signum
                  vt-= vt-/= vt-< vt-<= vt-> vt->=
                  vt-maximum vt-minimum vt-fmax vt-fmin
                  vt-isnan vt-isinf vt-isfinite
                  vt-sum vt-prod vt-amax vt-amin))
    (%cx-orig name)))

(defun %cx-deep-p (x)
  "递归判定 X 是否「含复数」：vt → 看 dtype；数 → complexp；容器 → 递归。
   这是全部复数分派的唯一判据（§16.12）。"
  (cond
    ((vt-p x) (not (null (vt-complex-dtype-p (vt-dtype x)))))
    ((complexp x) t)
    ((consp x) (some #'%cx-deep-p x))
    ((vectorp x) (some #'%cx-deep-p x))
    (t nil)))

(defun %cx-tensor-p (x)
  "判定 X（vt 或标量）是否为复数张量/复数标量。"
  (cond ((vt-p x) (not (null (vt-complex-dtype-p (vt-dtype x)))))
        (t (complexp x))))

;;; ------------------------------------------------------------------
;;; 通用小工具：C 连续落地 / 步长 / fiber 起点
;;; ------------------------------------------------------------------

(defun %cx-flat (tensor)
  "把张量落地为 C 连续、offset=0 的视图（已连续则零拷贝返回）。"
  (vt-contiguous (ensure-vt tensor)))

(defun %cx-strides-vec (shape)
  "C 连续布局的步长向量。"
  (coerce (vt-compute-strides shape) 'simple-vector))

(defun %cx-fiber-starts (shape ax)
  "枚举「沿 AX 的 fiber 起始线性偏移」。

   返回 (values starts stride-ax dim-ax)：
     starts     simple-vector，每个 fiber 的起始偏移（C 连续布局、offset=0）
     stride-ax  沿 AX 前进 1 的线性步长
     dim-ax     AX 轴长度

   实现：C 连续布局下线性下标 k 的 AX 坐标 = (mod (floor k stride-ax) dim-ax)，
   该坐标为 0 即 fiber 起点。O(size) 一次扫描，无 odometer 嵌套。"
  (let* ((strides (%cx-strides-vec shape))
         (ax-stride (svref strides ax))
         (dim-ax (nth ax shape))
         (total (reduce #'* shape :initial-value 1))
         (starts '()))
    (when (plusp total)
      (if (<= dim-ax 0)
          (setf starts '())
          (dotimes (k total)
            (when (zerop (mod (floor k ax-stride) dim-ax))
              (push k starts)))))
    (values (coerce (nreverse starts) 'simple-vector) ax-stride dim-ax)))

;;; ==================================================================
;;; 1. 结果 dtype 规则（§16.5）
;;; ==================================================================

(defun %cx-result-complex-dtype (in-dtype dtype op-name)
  "复数输入的数学函数结果 dtype：缺省保持输入复数 dtype；
   显式非复数 dtype 报错（对标 numpy ufunc 的 safe-casting 拒绝 complex→real）。"
  (cond ((null dtype) in-dtype)
        ((vt-complex-dtype-p dtype) dtype)
        (t (error "~a: 复数输入的结果 dtype 必须是复数类型，收到 ~a（CONVENTIONS §16.5）"
                  op-name dtype))))

(defun %cx-map1 (fn tensor out dtype op-name)
  "复数一元逐元素映射（通用 strides 路径，支持任意视图）。"
  (let* ((in (ensure-vt tensor))
         (dt (%cx-result-complex-dtype (vt-dtype in) dtype op-name)))
    (vt-map fn in :out out :dtype dt)))

;;; ==================================================================
;;; 2. 复数内核（CL 的复数实现已与 numpy 分支切割逐位一致，§16.10）
;;; ==================================================================
;;; 实测对照（SBCL 2.6.8 vs numpy 2.4.6）：
;;;   (sqrt #C(-4 -0.0)) = #C(0 -2)   ↔ np.sqrt(-4-0j)  = -2j
;;;   (log  #C(-1 -0.0)) = -πi        ↔ np.log(-1-0j)   = -πj
;;;   (atan -0.0 -1.0)   = -π         ↔ np.angle(-1-0j) = -π
;;; 故 sqrt/log/三角/双曲/反函数直接复用 CL，只需补齐极点与非有限值
;;; 的返回约定（CL 在这些点抛条件，numpy 返回 Inf/NaN）。

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

(declaim (inline %cx-hypot %cop-abs %cop-sqrt))
(defun %cx-hypot (x y)
  "无溢出的 hypot（|x|、|y| 极值时不溢出）。x/y 为实数。"
  (let ((ax (abs x)) (ay (abs y)))
    (cond ((zerop ax) ay)
          ((zerop ay) ax)
          ((> ax ay) (let ((r (/ ay ax))) (* ax (sqrt (+ 1.0s0 (* r r))))))
          (t (let ((r (/ ax ay))) (* ay (sqrt (+ 1.0s0 (* r r)))))))))

(defun %cop-sqrt (z)
  "复数平方根（主值，C99 csqrt 分支切割）。
   sqrt(-4+0i)=+2i、sqrt(-4-0i)=-2i（尊重虚部 -0.0 的符号）。

   为什么不用 `(sqrt z)`：SBCL 的复数 sqrt 走通用复数算术，实测
   **350ns/元素**（1M = 350ms）；而 C99 的标准公式只用 hypot + 1 次
   实数 sqrt + 2 次除法，约 20ns。二者在本库被测的全部点上逐位一致
   （complex-test 的 8 个分支切割用例 + numpy 差分）。"
  (let ((x (%creal z)) (y (%cimag z)))
    (cond ((%cx-nan-p z) (%cx-nan-complex z))
          ;; 实轴：虚部保持原符号（含 -0.0），必须走特例，否则 ±0 会被抹平
          ((and (zerop y) (>= x 0)) (complex (sqrt x) y))
          ((and (zerop y) (minusp x))
           (complex (* 0 x)
                    (if (minusp (float-sign y)) (- (sqrt (- x))) (sqrt (- x)))))
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

;;; ==================================================================
;;; 3. 一元函数分派包装
;;; ==================================================================

(defmacro %cx-def-unary (name &optional (kernel name))
  "为 (vt &key out dtype) 形式的一元函数生成复数分派包装。

   性能（v1.1）：复数分支**必须**走 `vt-fast-map` 宏而不是 `vt-map` 函数。
   二者语义相同，但只有宏在编译期拿得到 `(function <sym>)` 字面量，从而
   展开成「按元素类型分派 + 类型声明驱动」的内联循环；`vt-map` 是函数，
   `fn` 到手时已是函数对象，只能逐元素 `funcall` + 泛型 `aref`（实测
   1M 元素 sqrt：泛型 453ms → 内联 12ms）。"
  `(let ((orig (%cx-orig ',name)))
     (setf (symbol-function ',name)
           (lambda (tensor &rest keys &key (out nil) (dtype nil) &allow-other-keys)
             (declare (ignore keys))
             (if (%cx-tensor-p tensor)
                 (let* ((in (ensure-vt tensor))
                        (dt (%cx-result-complex-dtype (vt-dtype in) dtype ,(string name))))
                   (vt-fast-map (function ,kernel) in :out out :dtype dt))
                 (funcall orig tensor :out out :dtype dtype))))))

(%cx-def-unary vt-sqrt %cop-sqrt)
(%cx-def-unary vt-exp exp)
;;; vt-log 不能用 %cx-def-unary：它多一个 :base 关键字，宏只转发 out/dtype
;;; 会把 :base 静默丢掉（实测：log(8, base=2) 退化成 ln(8)=2.079）。
(defun %cop-log-base (z base)
  "复数换底对数 log_base(z)。base 必须为正实数（numpy：非法 base → NaN）。"
  (if (and (realp base) (plusp base) (/= base 1))
      (let ((l (%cop-log z)))
        (complex (/ (realpart l) (log base 1.0d0))
                 (/ (imagpart l) (log base 1.0d0))))
      (%cx-nan-complex z)))

(let ((orig (%cx-orig 'vt-log)))
  (setf (symbol-function 'vt-log)
        (lambda (tensor &rest keys &key (base nil) (out nil) (dtype nil)
                                 &allow-other-keys)
          (declare (ignore keys))
          (if (%cx-tensor-p tensor)
              (let* ((in (ensure-vt tensor))
                     (dt (%cx-result-complex-dtype (vt-dtype in) dtype "VT-LOG")))
                (if base
                    (vt-map (lambda (z) (%cop-log-base z base)) in :out out :dtype dt)
                    (vt-map #'%cop-log in :out out :dtype dt)))
              (funcall orig tensor :base base :out out :dtype dtype)))))
(%cx-def-unary vt-log2 %cop-log2)
(%cx-def-unary vt-log10 %cop-log10)
(%cx-def-unary vt-sin sin)
(%cx-def-unary vt-cos cos)
(%cx-def-unary vt-tan tan)
(%cx-def-unary vt-atan atan)
(%cx-def-unary vt-sinh sinh)
(%cx-def-unary vt-cosh cosh)
(%cx-def-unary vt-tanh tanh)
(%cx-def-unary vt-asin asin)
(%cx-def-unary vt-acos acos)
(%cx-def-unary vt-asinh asinh)
(%cx-def-unary vt-acosh acosh)
(%cx-def-unary vt-atanh atanh)
(%cx-def-unary vt-square %cop-square)
(%cx-def-unary vt-reciprocal %cop-recip)
(%cx-def-unary vt-signum %cop-signum)
(%cx-def-unary vt-negative %cop-neg)

;;; ==================================================================
;;; 4. 实部 / 虚部 / 共轭 / 幅角 / 模（§16.6）
;;; ==================================================================

;;; ------------------------------------------------------------------
;;; 「复数 → 实数」专用内核（abs / real / imag / angle）
;;;
;;; 为什么不能复用 vt-fast-map：那套特化要求「输入 dtype == 输出 dtype」
;;; （dtype-check），而 abs/real/imag/angle 是 complex→float 的**跨类**映射，
;;; 只能落回 vt-map 的泛型 walker（1M 元素实测 875ms）。这里按元素类型
;;; 直接生成「读复数槽位、写浮点槽位」的循环。
;;;
;;; 优化声明策略（CONVENTIONS §16.14）：
;;;   (speed 3) (safety 1) —— **刻意保留边界检查**。索引来自 offset/size，
;;;   本可证明不越界，但这里是普通函数（不是宏展开的热循环），保留检查的
;;;   实测代价 < 8%，换来「越界/类型错配立刻报错」而不是静默踩内存。
;;;   只有 map-reduce 里那些宏展开的内联循环才用 safety 0（那里索引由
;;;   循环上界逐元素推进，编译器可完全证明）。
;;; ------------------------------------------------------------------

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

(defun vt-abs (vt &key out dtype)
  "逐元素绝对值 / 复数模。

   实数：整数保持整数 dtype、浮点保精度（对标 numpy.abs）。
   复数（§16.6）：结果**恒为实数** —— complex128 → float64、
   complex64 → float32（对标 numpy：abs 的 dtype 取实部分量类型）。
   与 numpy 的差异：numpy 的 abs 是 SIMD ufunc；本实现对复数走通用
   stride 路径（无 SIMD 特化），语义一致、仅性能不同。

   :out 契约同 vt-+。"
  (let ((in (ensure-vt vt)))
    (if (vt-complex-dtype-p (vt-dtype in))
        (%cx-run-to-real #'%cx-abs-run #'%cop-abs in out
                         (or dtype (vt-real-dtype-of (vt-dtype in)))
                         "VT-ABS")
        (funcall (%cx-orig 'vt-abs) vt :out out :dtype dtype))))

(defun vt-real (tensor &key out dtype)
  "取实部（对标 numpy.real）。复数 → 分量浮点 dtype 的实数张量；
   实数输入原样返回（副本）。

   与 numpy 的差异（§16.6）：numpy 的 `.real` 是**零拷贝视图**（写实部会
   改到原数组）；clvt 的 strided view 无法表达「同一缓冲的实数子集」
   （缓冲元素类型是复数），故返回**副本**。需要写回请用 vt-complex 重组。"
  (let* ((in (ensure-vt tensor))
         (dt (or dtype (vt-real-dtype-of (vt-dtype in))))
         (dt (if (vt-complex-dtype-p dt) (vt-real-dtype-of dt) dt)))
    (if (vt-complex-dtype-p (vt-dtype in))
        (%cx-run-to-real #'%cx-real-run #'%creal in out dt "VT-REAL")
        (vt-map #'%creal in :out out :dtype dt))))

(defun vt-imag (tensor &key out dtype)
  "取虚部（对标 numpy.imag）。实数输入 → 全 0（与 numpy 一致）。
   返回副本而非视图，原因同 vt-real。"
  (let* ((in (ensure-vt tensor))
         (in-dt (vt-dtype in))
         (dt (or dtype (if (vt-complex-dtype-p in-dt)
                           (vt-real-dtype-of in-dt)
                           in-dt))))
    (if (vt-complex-dtype-p in-dt)
        (%cx-run-to-real #'%cx-imag-run #'%cimag in out dt "VT-IMAG")
        (vt-map #'%cimag in :out out :dtype dt))))

(defun vt-conj (tensor &key out dtype)
  "逐元素共轭（对标 numpy.conj / numpy.conjugate）。实数输入原样返回。"
  (let ((in (ensure-vt tensor)))
    (vt-fast-map #'%cconj in :out out :dtype (or dtype (vt-dtype in)))))

(defun vt-conjugate (tensor &key out dtype)
  "vt-conj 的 numpy 全名别名。"
  (vt-conj tensor :out out :dtype dtype))

(defun vt-angle (tensor &key deg out dtype)
  "逐元素幅角 atan2(im, re) ∈ (-π, π]（对标 numpy.angle）。
   DEG 为真时返回角度制。结果恒为实数 dtype。
   零值返回 0（保留 ±0.0 符号）。"
  (let* ((in (ensure-vt tensor))
         (in-dt (vt-dtype in))
         (dt (or dtype (if (vt-complex-dtype-p in-dt)
                           (vt-real-dtype-of in-dt)
                           (if (eq in-dt :float32) :float32 :float64))))
         (fn (if deg
                 (lambda (z)
                   (* (%carg z)
                      (if (eq dt :float32) (/ 180.0s0 0.31415927s1) (/ 180.0d0 pi))))
                 #'%carg)))
    ;; 度数换算需要额外一次乘法（内核未参数化），deg 走通用路径；
    ;; 默认弧度走 %CX-ANGLE-RUN 特化内核。
    (if (and (not deg) (vt-complex-dtype-p in-dt))
        (%cx-run-to-real #'%cx-angle-run #'%carg in out dt "VT-ANGLE")
        (vt-map fn in :out out :dtype dt))))

(defun vt-iscomplexobj (x)
  "判定 X 是否含复数（对标 numpy.iscomplexobj）。"
  (if (%cx-deep-p x) t nil))

(defun vt-isrealobj (x)
  "判定 X 是否不含复数（对标 numpy.isrealobj）。"
  (not (%cx-deep-p x)))

(defun vt-complex (real imag &key dtype)
  "由实部 / 虚部构造复数张量（对标 numpy 的 `re + 1j*im` 惯用式）。

   REAL / IMAG 可为标量、序列或张量，按广播规则组合。
   结果 dtype 缺省 complex128；显式 :dtype 支持 :complex64 / :complex128。

   示例：
     (vt-complex 1.0d0 2.0d0)          ; 0 维，值 #C(1.0d0 2.0d0)
     (vt-complex (vt-arange 3) 0.0d0)  ; (3,) complex128"
  (let ((dt (or dtype :complex128)))
    (unless (vt-complex-dtype-p dt)
      (error "vt-complex: :dtype 必须是复数类型（:complex64/:complex128），收到 ~a" dt))
    (let* ((cmp (vt-complex-real-dtype dt))
           (re (ensure-vt real))
           (im (ensure-vt imag))
           (re (if (eq (vt-dtype re) cmp) re (vt-astype re cmp)))
           (im (if (eq (vt-dtype im) cmp) im (vt-astype im cmp))))
      (vt-map #'complex re im :dtype dt))))

;;; ==================================================================
;;; 5. 幂（§16.10）
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

(defun vt-pow (vt power &key out dtype)
  "逐元素幂 VT**POWER（对标 numpy.power）。

   复数（§16.10）：任一侧为复数即按复数幂计算（CL 复指数 expt），
   结果 dtype 按提升规则取向；显式非复数 :dtype 报错。
   幂的极点：0 ** 非正实数指数 → NaN+NaNj（对标 numpy，CL 会抛条件，此处拦截）。
   实数路径完全不变（整数基 + 非负整数指数仍走精确整数幂）。"
  (let ((in (ensure-vt vt)))
    (if (or (vt-complex-dtype-p (vt-dtype in)) (%cx-deep-p power))
        (let* ((pw (if (vt-p power) power (ensure-vt power)))
               (promoted (or dtype (vt-promote-type (vt-dtype in) (vt-dtype pw))))
               (dt (if (vt-complex-dtype-p promoted) promoted :complex128))
               (dt (%cx-result-complex-dtype dt (and dtype dtype) "VT-POW")))
          (vt-map (function %cx-pow-2) in pw :out out :dtype dt))
        (funcall (%cx-orig 'vt-pow) vt power :out out :dtype dtype))))

(defun vt-expt (vt power &key out dtype)
  "vt-pow 的别名（CL 习惯命名）。"
  (vt-pow vt power :out out :dtype dtype))

;;; ------------------------------------------------------------------
;;; 5.1 vt-float-power（numpy.float_power）
;;;
;;; 实数路径的实现是 (coerce a 'double-float)/(coerce b 'double-float) 后
;;; expt —— 对复数入参就是**静默取实部**（实测 (vt-float-power #C(2 1)
;;; #C(2 0)) 得 4.0，而 numpy 给 3+4j）。复数分支直接复用 %CX-POW-2：
;;; numpy.float_power 与 numpy.power 在复数域是同一个 pow 主值分支
;;; （实测 (2+1j)**(0.5+0j) 两者都是 1.4553+0.3436j），只是 float_power
;;; 对实数也强制浮点。
;;; ------------------------------------------------------------------
(let ((orig (%cx-orig 'vt-float-power)))
  (setf (symbol-function 'vt-float-power)
        (lambda (x y &rest keys &key (out nil) (dtype nil) &allow-other-keys)
          (declare (ignore keys))
          (if (or (%cx-deep-p x) (%cx-deep-p y))
              (let* ((xe (ensure-vt x))
                     (ye (ensure-vt y))
                     (promoted (or dtype (vt-promote-type (vt-dtype xe)
                                                          (vt-dtype ye))))
                     (dt (if (vt-complex-dtype-p promoted) promoted :complex128))
                     (dt (%cx-result-complex-dtype dt (and dtype dtype)
                                                  "VT-FLOAT-POWER")))
                (vt-map (function %cx-pow-2) xe ye :out out :dtype dt))
              (funcall orig x y :out out :dtype dtype)))))

;;; ==================================================================
;;; 6. 比较与极值（字典序，§16.8）
;;; ==================================================================

(defmacro %cx-def-compare (name kernel)
  "生成比较函数的复数分派包装（结果 :int8，1/0）。"
  `(let ((orig (%cx-orig ',name)))
     (setf (symbol-function ',name)
           (lambda (t1 t2 &key (dtype :int8) out)
             (if (or (%cx-tensor-p t1) (%cx-tensor-p t2))
                 ;; vt-fast-map：复数输入 + int8 输出命中「窄整型输出」快路径
                 ;; （%vt-mask*-fast 的复数输入格），避免泛型 walker
                 (vt-fast-map (function ,kernel) (ensure-vt t1) (ensure-vt t2)
                              :out out :dtype dtype)
                 (funcall orig t1 t2 :dtype dtype :out out))))))

;; 性能：这一组在比较/谓词内核里每元素调用一次，必须内联。
;; 注意 declaim 必须出现在 **defun 之前**：SBCL 只在定义时可见
;; inline 声明才记录可内联展开，事后补 declaim 不生效（实测 78ns/次）。
(declaim (inline %cx-lt-flag %cx-le-flag %cx-gt-flag %cx-ge-flag
                 %cx-eq-flag %cx-ne-flag))

;;; 比较的布尔承载：clvt 约定比较运算返回 :int8 的 1/0（§3.1），
;;; 而 %clt/%cle/%cgt/%cge 返回 T/NIL（排序/极值内核需要布尔），
;;; 故此处做一层 1/0 适配，避免把 T 写进整型缓冲。
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

(%cx-def-compare vt-= %cx-eq-flag)
(%cx-def-compare vt-/= %cx-ne-flag)
(%cx-def-compare vt-< %cx-lt-flag)
(%cx-def-compare vt-<= %cx-le-flag)
(%cx-def-compare vt-> %cx-gt-flag)
(%cx-def-compare vt->= %cx-ge-flag)

(defun %cop-max (a b)
  "maximum 的复数语义：字典序取大，NaN 传播。"
  (cond ((%cisnan a) a) ((%cisnan b) b) ((%clt a b) b) (t a)))

(defun %cop-min (a b)
  "minimum 的复数语义：字典序取小，NaN 传播。"
  (cond ((%cisnan a) a) ((%cisnan b) b) ((%cgt a b) b) (t a)))

(defun %cop-fmax (a b)
  "fmax 的复数语义：字典序取大，忽略 NaN。"
  (cond ((%cisnan a) b) ((%cisnan b) a) ((%clt a b) b) (t a)))

(defun %cop-fmin (a b)
  "fmin 的复数语义：字典序取小，忽略 NaN。"
  (cond ((%cisnan a) b) ((%cisnan b) a) ((%cgt a b) b) (t a)))

(defmacro %cx-def-minmax (name kernel)
  "生成 maximum/minimum/fmax/fmin 的复数分派包装（结果保持复数 dtype）。"
  `(let ((orig (%cx-orig ',name)))
     (setf (symbol-function ',name)
           (lambda (t1 t2 &key out dtype)
             (if (or (%cx-tensor-p t1) (%cx-tensor-p t2))
                 (let* ((a (ensure-vt t1)) (b (ensure-vt t2))
                        (p (vt-promote-type (vt-dtype a) (vt-dtype b)))
                        (dt (or dtype p)))
                   (unless (vt-complex-dtype-p dt)
                     (error "~a: 复数输入的结果 dtype 必须是复数类型，收到 ~a"
                            ,(string name) dt))
                   (vt-map (function ,kernel) a b :out out :dtype dt))
                 (funcall orig t1 t2 :out out :dtype dtype))))))

(%cx-def-minmax vt-maximum %cop-max)
(%cx-def-minmax vt-minimum %cop-min)
(%cx-def-minmax vt-fmax %cop-fmax)
(%cx-def-minmax vt-fmin %cop-fmin)

;;; ==================================================================
;;; 7. 判定族 isnan / isinf / isfinite（§16.7）
;;; ==================================================================

(declaim (inline %cx-nan-flag %cx-inf-flag %cx-finite-flag))
; 谓词在三处被调用：vt-fast-map 内联循环（已屏蔽陷阱）、%cx-sort-less、
;; %cx-arg-reduce-run。全部处于 with-float-safe 之下，故用免屏蔽快判。
(defun %cx-nan-flag (x) (if (%cx-nan-p x) 1 0))
(defun %cx-inf-flag (x) (if (%cx-inf-p x) 1 0))
(defun %cx-finite-flag (x) (if (%cx-finite-p x) 1 0))

(defmacro %cx-def-predicate (name kernel)
  "生成 isnan/isinf/isfinite 的复数分派包装（结果 :int8，1/0）。"
  `(let ((orig (%cx-orig ',name)))
     (setf (symbol-function ',name)
           (lambda (tensor &key out)
             (if (%cx-tensor-p tensor)
                 (vt-fast-map (function ,kernel) (ensure-vt tensor)
                              :out out :dtype :int8)
                 (funcall orig tensor :out out))))))

(%cx-def-predicate vt-isnan %cx-nan-flag)
(%cx-def-predicate vt-isinf %cx-inf-flag)
(%cx-def-predicate vt-isfinite %cx-finite-flag)

;;; ==================================================================
;;; 8. 归约内核（sum / prod / max / min）
;;;
;;; 为什么不复用 def-vt-reduce 的内核：那套内核按「存储级 dtype」做编译期
;;; 特化（+kernel-lts+ 只有 4 种实数类型），复数落不进去；硬塞会把
;;; (the (simple-array double-float (*)) …) 套到复数缓冲区上（safety 0 下
;;; 直接段错误）。因此这里写一个**独立、通用、stride 驱动**的复数归约：
;;; 先 vt-contiguous 落地，再按行主序 odometer 遍历（正确性优先，
;;; 连续化只影响性能 —— §1 铁律 1）。
;;; ==================================================================

;;; ------------------------------------------------------------------
;;; 8.1 类型声明驱动的归约内核
;;;
;;; 旧版只有一个「泛型 odometer」：data 是 (simple-array * (*))，每次
;;; (aref data off) 都要走通用数组分派 + 装箱，1M 元素 sum = 92ms
;;; （实数 1ms）。这里按元素类型把缓冲区绑定成
;;; (simple-array (complex double-float) (*))，让 SBCL 直接生成
;;; 「读两个 double、加、写回」的指令序列。
;;;
;;; 声明策略：`(speed 3) (safety 1)` — 保留边界检查。索引由 head/dim/
;;; ax-stride 推出，理论上可证不越界，但这里**刻意不关检查**：这是普通
;;; 函数（非宏展开热循环），检查成本实测 < 5%，而越界/形状算错会立刻
;;; 报错而不是静默踩内存。CONVENTIONS §16.14 记录该分级。
;;; ------------------------------------------------------------------

(defmacro %cx-dispatch (data arr zero &body body)
  "把 DATA 按复数元素类型绑定为 ARR（simple-array）与 ZERO（该精度复数零），
   在 BODY 中执行。调用方保证 DATA 是复数缓冲区。

   **嵌套即同精度**（本宏的关键约定）：BODY 里再写 (%cx-dispatch data2 …)
   时，内层**不再二次分派**，而是被就地改写成「按外层已选定的精度绑定」。
   原因是 717990d 之后出现的编译失败：三缓冲区内核（matmul）展开成 2^3=8
   条分支，其中 6 条精度互不一致（例如把 complex-double 的累加结果写进
   complex-single 缓冲区）。这些分支**运行期永不成立**（上游已保证同精度），
   但编译期它们是实打实的代码：SBCL 推出 `(+ cs cs)` 为 (complex
   double-float)，再撞上目标缓冲区的 (complex single-float) 元素类型断言，
   于是每条死分支都产出一条 WARNING，COMPILE-FILE 返回 failure-p=T，
   ASDF 报 COMPILE-FILE-ERROR —— 整个库编译失败。

   改为「分派一次 + 其余 THE 断言」后：
     · 分支数 2^N → 2，代码量、编译时间、镜像体积同步下降；
     · 不留静态不一致的死代码，WARNING 归零；
     · 精度不一致从「静默读错内存」变成 (safety 1) 下 THE 的即时类型错误。"
  (flet ((bind (ctype zero-form)
           ;; 内层 %CX-DISPATCH 的同精度替身：只绑定，不再 COND。
           `(macrolet ((%cx-dispatch (data2 arr2 zero2 &body body2)
                         `(let ((,arr2 (the (simple-array ,',ctype (*)) ,data2))
                                (,zero2 ,',zero-form))
                            (declare (type (simple-array ,',ctype (*)) ,arr2)
                                     (optimize (speed 3) (safety 1)))
                            ,@body2)))
              ,@body)))
    `(let ((et (array-element-type ,data)))
       (cond ((equal et '(complex double-float))
              (let ((,arr (the (simple-array (complex double-float) (*)) ,data))
                    (,zero (complex 0.0d0 0.0d0)))
                (declare (type (simple-array (complex double-float) (*)) ,arr)
                         (optimize (speed 3) (safety 1)))
                ,(bind '(complex double-float) '(complex 0.0d0 0.0d0))))
             ((equal et '(complex single-float))
              (let ((,arr (the (simple-array (complex single-float) (*)) ,data))
                    (,zero (complex 0.0s0 0.0s0)))
                (declare (type (simple-array (complex single-float) (*)) ,arr)
                         (optimize (speed 3) (safety 1)))
                ,(bind '(complex single-float) '(complex 0.0s0 0.0s0))))
             (t (error "%cx-dispatch: 非复数缓冲区 ~a" et))))))

(defun %cx-fiber-geom (shape ax)
  "C 连续布局下沿 AX 的几何量：返回 (values head dim-ax ax-stride)：
   共 HEAD 条 fiber，每条长 DIM-AX，相邻元素线性距离 AX-STRIDE。"
  (let* ((dims (coerce shape 'simple-vector))
         (rank (length dims))
         (ax-stride 1)
         (head 1))
    (declare (type fixnum ax-stride head))
    (loop for d of-type fixnum from (1+ ax) below rank
          do (setf ax-stride (the fixnum (* ax-stride (the fixnum (svref dims d))))))
    (loop for d of-type fixnum from 0 below ax
          do (setf head (the fixnum (* head (the fixnum (svref dims d))))))
    (values head (the fixnum (svref dims ax)) ax-stride)))

(defmacro %def-cx-reducer (name step &key (init-form :zero))
  "生成归约内核 NAME(DATA SHAPE AX)：AX=NIL 走全局单循环，否则走 fiber 循环。

   STEP 是含 ACC 与 V 两个自由符号的模板（如 (+ acc v)），宏把 V 替换成
   具体取元素形式。INIT-FORM：
     :zero（缺省）→ 该精度的复数零；
     具体表达式    → 原样使用（如 prod 的 (+ zero 1)）；
     NIL          → 字面 nil，表示「首元素播种」（max/min 必须如此）。"
  (let ((init-code (if (eq init-form :zero) 'zero init-form)))
    (flet ((step-at (v) (subst v 'v step)))
      `(defun ,name (data shape ax)
         (%cx-dispatch data d zero
           (let ((init ,init-code))
             (if (null ax)
                 (let ((acc init)
                       (n (the fixnum (length d))))
                   (declare (type fixnum n))
                   (dotimes (i n)
                     (declare (type fixnum i))
                     (setf acc ,(step-at '(aref d i))))
                   (vector acc))
                 (multiple-value-bind (head dim ax-stride)
                     (%cx-fiber-geom shape ax)
                   (declare (type fixnum head dim ax-stride))
                   (let ((res (make-array (the fixnum (* head ax-stride)))))
                     (dotimes (h head)
                       (declare (type fixnum h))
                       (let* ((hb (the fixnum (* h dim ax-stride)))
                              (row (the fixnum (* h ax-stride))))
                         (declare (type fixnum hb row))
                         (dotimes (ti ax-stride)
                           (declare (type fixnum ti))
                           (let ((acc init)
                                 (p (the fixnum (+ hb ti))))
                             (declare (type fixnum p))
                             (dotimes (j dim)
                               (declare (type fixnum j))
                               (setf acc ,(step-at '(aref d p)))
                               (incf p ax-stride))
                             (setf (aref res (the fixnum (+ row ti))) acc)))))
                     res)))))))))

(%def-cx-reducer %cx-reduce-sum (+ acc v))
(%def-cx-reducer %cx-reduce-prod (* acc v) :init-form (+ zero 1))
(%def-cx-reducer %cx-reduce-max (if (null acc) v (%cop-max acc v)) :init-form nil)
(%def-cx-reducer %cx-reduce-min (if (null acc) v (%cop-min acc v)) :init-form nil)

(defun %cx-arg-reduce-run (data shape ax op)
  "argmax/argmin 的连续内核。返回 fixnum 向量。
   OP ∈ :argmax :argmin。AX=NIL 表示展平（结果为单元素向量）。"
  (%cx-dispatch data d zero
    (declare (ignore zero))
    (flet ((better (acc v)
             (cond ((null acc) t)
                   ((%cx-nan-p acc) nil)
                   ((%cx-nan-p v) t)
                   ((eq op :argmax) (%clt acc v))
                   (t (%cgt acc v)))))
      (if (null ax)
          (let ((n (the fixnum (length d)))
                (acc nil)
                (best 0))
            (declare (type fixnum n best))
            (dotimes (i n)
              (declare (type fixnum i))
              (let ((v (aref d i)))
                (when (better acc v) (setf acc v best i))))
            (vector best))
          (multiple-value-bind (head dim ax-stride) (%cx-fiber-geom shape ax)
            (declare (type fixnum head dim ax-stride))
            (let ((res (make-array (the fixnum (* head ax-stride)) :element-type 'fixnum)))
              (declare (type (simple-array fixnum (*)) res))
              (dotimes (h head)
                (declare (type fixnum h))
                (let ((hb (the fixnum (* h dim ax-stride))))
                  (declare (type fixnum hb))
                  (dotimes (ti ax-stride)
                    (declare (type fixnum ti))
                    (let ((acc nil)
                          (best 0)
                          (p (the fixnum (+ hb ti))))
                      (declare (type fixnum best p))
                      (dotimes (j dim)
                        (declare (type fixnum j))
                        (let ((v (aref d p)))
                          (when (better acc v) (setf acc v best j))))
                      (setf (aref res (the fixnum (+ (the fixnum (* h ax-stride)) ti)))
                            best)))))
              res))))))

(defun %cx-reduce-values (data shape axes op)
  "在连续 DATA（行主序、offset=0）上沿 AXES 归约，返回 results 向量。

   AXES 为 NIL 表示**归约全部轴**（numpy 的 axis=None 语义）。
   OP ∈ :sum :prod :max :min。空归约区（n-red=0）：sum→0、prod→1；
   max/min → 报错（对标 numpy 的 zero-size 归约错误）。

   选路：全局 / 单轴 → 类型特化连续内核；多轴 → 通用 odometer 兜底
   （语义完全一致，多轴归约属罕见路径）。"
  (let* ((rank (length shape))
         (axes* (sort (if axes (copy-list axes) (loop for d below rank collect d)) #'<))
         (non-axes (loop for d from 0 below rank unless (member d axes*) collect d))
         (n-red (reduce #'* (mapcar (lambda (d) (nth d shape)) axes*) :initial-value 1))
         (n-out (reduce #'* (mapcar (lambda (d) (nth d shape)) non-axes)
                        :initial-value 1)))
    (when (and (zerop n-red) (member op '(:max :min)) (plusp n-out))
      (error "zero-size array to reduction operation ~(~a~) which has no identity" op))
    (cond
      ((or (null axes) (= 1 (length axes*)))
       (let ((ax (when axes (car axes*))))
         (case op
           (:sum (%cx-reduce-sum data shape ax))
           (:prod (%cx-reduce-prod data shape ax))
           (:max (%cx-reduce-max data shape ax))
           (:min (%cx-reduce-min data shape ax)))))
      (t (%cx-reduce-values-generic data shape axes op)))))

(defun %cx-reduce-values-generic (data shape axes op)
  "多轴归约的通用 odometer 实现（语义基准；复数多轴归约的正确性兜底）。"
  (let* ((rank (length shape))
         (strides (%cx-strides-vec shape))
         (axes* (sort (copy-list axes) #'<))
         (non-axes (loop for d from 0 below rank unless (member d axes*) collect d))
         (out-shape (loop for d in non-axes collect (nth d shape)))
         (red-dims (loop for d in axes* collect (nth d shape)))
         (red-strides (loop for d in axes* collect (svref strides d)))
         (n-red (reduce #'* red-dims :initial-value 1))
         (n-out (reduce #'* out-shape :initial-value 1))
         (results (make-array (max n-out 1)))
         (red-idx (make-array (max 1 (length axes*)) :element-type 'fixnum
                                                     :initial-element 0))
         (non-idx (make-array (max 1 (length non-axes)) :element-type 'fixnum
                                                     :initial-element 0)))
    (declare (type fixnum n-red n-out)
             (type (simple-array fixnum (*)) red-idx non-idx))
    (dotimes (k n-out)
      (let ((base 0))
        (loop for j of-type fixnum from 0 below (length non-axes)
              do (incf base (the fixnum (* (aref non-idx j)
                                           (svref strides (nth j non-axes))))))
        (fill red-idx 0)
        (let ((acc (case op (:sum 0) (:prod 1) (t nil))))
          (dotimes (j n-red)
            (let ((off base))
              (loop for a of-type fixnum from 0 below (length axes*)
                    do (incf off (the fixnum (* (aref red-idx a)
                                                (nth a red-strides)))))
              (let ((v (aref data off)))
                (case op
                  (:sum (setf acc (+ acc v)))
                  (:prod (setf acc (* acc v)))
                  (:max (setf acc (if (null acc) v (%cop-max acc v))))
                  (:min (setf acc (if (null acc) v (%cop-min acc v)))))))
            (let ((d (1- (length axes*))))
              (loop
                (when (< d 0) (return))
                (incf (aref red-idx d))
                (when (< (aref red-idx d) (the fixnum (nth d red-dims))) (return))
                (setf (aref red-idx d) 0)
                (decf d))))
          (setf (aref results k)
                (if (null acc) (case op (:sum 0) (:prod 1) (t 0)) acc))))
      (let ((d (1- (length non-axes))))
        (loop
          (when (< d 0) (return))
          (incf (aref non-idx d))
          (when (< (aref non-idx d) (the fixnum (nth d out-shape))) (return))
          (setf (aref non-idx d) 0)
          (decf d))))
    results))

(defun %cx-keepdims-shape (in-shape axes keepdims)
  "按 keepdims 计算归约结果形状。"
  (if (null axes)
      (if keepdims (make-list (length in-shape) :initial-element 1) nil)
      (if keepdims
          (loop for d in in-shape for i from 0
                collect (if (member i axes) 1 d))
          (loop for d in in-shape for i from 0
                unless (member i axes) collect d))))

(defun %cx-write-flat (res values dtype)
  "把 VALUES（行主序）写进 RES（任意 strides，机制 B：offset+strides 驱动）。"
  (let* ((strides (coerce (vt-strides res) 'simple-vector))
         (dims (coerce (vt-shape res) 'simple-vector))
         (rank (length dims))
         (idx (make-array (max 1 rank) :element-type 'fixnum :initial-element 0))
         (data (vt-data res))
         (n (length values)))
    (declare (type (simple-array fixnum (*)) idx))
    (dotimes (k n)
      (let ((off (vt-offset res)))
        (loop for d of-type fixnum from 0 below rank
              do (incf off (the fixnum (* (aref idx d) (svref strides d)))))
        (setf (aref data off) (vt-cast (aref values k) dtype)))
      (let ((d (1- rank)))
        (loop
          (when (< d 0) (return))
          (incf (aref idx d))
          (when (< (aref idx d) (the fixnum (svref dims d))) (return))
          (setf (aref idx d) 0)
          (decf d))))
    res))

(defun %cx-reduce (name tensor axis keepdims dtype out op)
  "复数归约公共外壳：轴规范化 + 结果 dtype + :out 硬契约 + keepdims。"
  (with-float-safe
   (let* ((in (ensure-vt tensor))
          (in-shape (vt-shape in))
          (rank (length in-shape))
          (axes (vt-normalize-axes axis rank))
          (flat (%cx-flat in))
          (res-dtype (or dtype (vt-dtype in)))
          (out-shape (%cx-keepdims-shape in-shape axes keepdims))
          (res (if out
                   (vt-check-out out out-shape res-dtype :op-name name)
                   (make-vt out-shape 0 :dtype res-dtype)))
          (values (%cx-reduce-values (vt-data flat) in-shape axes op)))
     (%cx-write-flat res values res-dtype))))

(defmacro %cx-def-reduce (name op)
  "生成归约函数的复数分派包装。"
  `(let ((orig (%cx-orig ',name)))
     (setf (symbol-function ',name)
           (lambda (tensor &rest keys &key (axis nil) (keepdims nil)
                                    (dtype nil) (out nil) &allow-other-keys)
             (declare (ignore keys))
             (if (%cx-tensor-p tensor)
                 (%cx-reduce ,(string name) tensor axis keepdims dtype out ,op)
                 (funcall orig tensor :axis axis :keepdims keepdims
                          :dtype dtype :out out))))))

(%cx-def-reduce vt-sum :sum)
(%cx-def-reduce vt-prod :prod)
(%cx-def-reduce vt-amax :max)
(%cx-def-reduce vt-amin :min)

;;; argmax / argmin（单轴或展平；多轴与 numpy 一样不支持）---------------

(defun %cx-arg-reduce (name tensor axis keepdims out op)
  "复数 argmax/argmin。返回 :int64 下标张量。

   性能：连续数据走 %CX-ARG-REDUCE-RUN 的类型特化循环（原来用泛型
   aref + 每元素 2 次带屏蔽的 %CISNAN，1M 元素 912ms）。"
  (with-float-safe
    (let* ((in (ensure-vt tensor))
           (in-shape (vt-shape in))
           (rank (length in-shape))
           (flat (%cx-flat in))
           (axes (vt-normalize-axes axis rank)))
      (when (cdr axes)
        (error "~a: 不支持多轴（对标 numpy：arg 归约只接受单个整数 axis）" name))
      (when (zerop (vt-size flat))
        (error "attempt to get ~(~a~) of an empty sequence" name))
      (let* ((ax (car axes))
             (out-shape (%cx-keepdims-shape in-shape axes keepdims))
             (res (if out (vt-check-out out out-shape :int64 :op-name name)
                      (make-vt out-shape 0 :dtype :int64)))
             (vals (%cx-arg-reduce-run (vt-data flat) in-shape ax op)))
        (%cx-write-flat res vals :int64)))))

(defun vt-argmax (tensor &key axis keepdims dtype out)
  "最大值的下标（对标 numpy.argmax）。复数按字典序（§16.8）。
   NaN 分量的元素胜出（首个 NaN 下标）；空序列报错。"
  (declare (ignore dtype))
  (if (%cx-tensor-p tensor)
      (%cx-arg-reduce "VT-ARGMAX" tensor axis keepdims out :argmax)
      (funcall (%cx-orig 'vt-argmax) tensor :axis axis :keepdims keepdims
               :dtype nil :out out)))

(defun vt-argmin (tensor &key axis keepdims dtype out)
  "最小值的下标（对标 numpy.argmin）。复数按字典序（§16.8）。"
  (declare (ignore dtype))
  (if (%cx-tensor-p tensor)
      (%cx-arg-reduce "VT-ARGMIN" tensor axis keepdims out :argmin)
      (funcall (%cx-orig 'vt-argmin) tensor :axis axis :keepdims keepdims
               :dtype nil :out out)))

;;; ==================================================================
;;; 9. 统计量（mean / var / std / average）
;;; ==================================================================

(defun vt-mean (tensor &key axis keepdims dtype out)
  "沿 AXIS 求均值（对标 numpy.mean）。

   复数（§16.7）：求和后除以计数，**结果保持复数 dtype**
   （complex64 进 complex64 出）。空归约 → NaN+NaNj（对标 numpy）。
   :out 契约同 vt-sum。"
  (let ((in (ensure-vt tensor)))
    (unless (vt-complex-dtype-p (vt-dtype in))
      (return-from vt-mean
        (funcall (%cx-orig 'vt-mean) tensor :axis axis :keepdims keepdims
                 :dtype dtype :out out)))
    (let* ((rank (length (vt-shape in)))
           (axes (vt-normalize-axes axis rank))
           (cnt (if axes
                    (reduce #'* (mapcar (lambda (a) (nth a (vt-shape in))) axes))
                    (reduce #'* (vt-shape in) :initial-value 1)))
           (dt (or dtype (vt-dtype in))))
      (unless (vt-complex-dtype-p dt)
        (error "vt-mean: 复数输入的 :dtype 必须是复数类型，收到 ~a" dt))
      (if (zerop cnt)
          (let ((shape (%cx-keepdims-shape (vt-shape in) axes keepdims)))
            (let ((r (if out (vt-check-out out shape dt :op-name "VT-MEAN")
                         (make-vt shape (%cx-nan-complex (complex 0.0d0 0.0d0))
                                  :dtype dt))))
              (vt-fill r (%cx-nan-complex (if (eq (vt-complex-real-dtype dt) :float32)
                                              (complex 0.0s0 0.0s0)
                                              (complex 0.0d0 0.0d0))))
              r))
          (let ((sums (%cx-reduce "VT-MEAN" in axis keepdims nil nil :sum))
                (inv (if (eq (vt-complex-real-dtype dt) :float32)
                         (/ 1.0s0 (float cnt 1.0s0))
                         (/ 1.0d0 (float cnt 1.0d0)))))
            (vt-map (lambda (s) (* s inv)) sums :out out :dtype dt))))))

(defun vt-var (tensor &key axis keepdims (ddof 0) dtype out)
  "沿 AXIS 求方差（对标 numpy.var）。

   复数（§16.7）：方差 = mean(|x - mean|²)，**结果恒为实数**
   （complex128 → float64、complex64 → float32）——与 numpy 一致。
   注意：不能用 vt-mean(vt-square(x)) 手算（那会得到复数）。
   :out 契约同 vt-sum。"
  (let ((in (ensure-vt tensor)))
    (unless (vt-complex-dtype-p (vt-dtype in))
      (return-from vt-var
        (funcall (%cx-orig 'vt-var) tensor :axis axis :keepdims keepdims
                 :ddof ddof :dtype dtype :out out)))
    (let* ((rdt (vt-real-dtype-of (vt-dtype in)))
           (dt (or dtype rdt)))
      (when (vt-complex-dtype-p dt)
        (error "vt-var: 复数方差的结果 dtype 必须是实数类型，收到 ~a" dt))
      (let* ((rank (length (vt-shape in)))
             (axes (vt-normalize-axes axis rank))
             (n (if axes
                    (reduce #'* (mapcar (lambda (a) (nth a (vt-shape in))) axes))
                    (reduce #'* (vt-shape in) :initial-value 1)))
             (den (max (- n ddof) 1))
             (mu (vt-mean in :axis axis :keepdims t))
             (dev (vt-- in mu))
             (sq (vt-square (vt-abs dev)))
             (ss (funcall (%cx-orig 'vt-sum) sq :axis axis :keepdims keepdims
                                                  :dtype dt))
             (inv (if (eq dt :float32)
                      (/ 1.0s0 (float den 1.0s0))
                      (/ 1.0d0 (float den 1.0d0)))))
        (vt-map (lambda (s) (* s inv)) ss :out out :dtype dt)))))

(defun vt-std (tensor &key axis keepdims (ddof 0) dtype out)
  "沿 AXIS 求标准差（对标 numpy.std）。复数 → 实数结果（同 vt-var）。"
  (let ((in (ensure-vt tensor)))
    (unless (vt-complex-dtype-p (vt-dtype in))
      (return-from vt-std
        (funcall (%cx-orig 'vt-std) tensor :axis axis :keepdims keepdims
                 :ddof ddof :dtype dtype :out out)))
    (let* ((dt (or dtype (vt-real-dtype-of (vt-dtype in))))
           (v (vt-var in :axis axis :keepdims keepdims :ddof ddof :dtype dt)))
      (vt-sqrt v :out out :dtype dt))))

(defun vt-average (tensor &rest args)
  "加权平均（对标 numpy.average）。复数走 vt-mean 语义；
   第一轮**不支持 weights**（显式报错，避免静默忽略权重）。"
  (let ((in (ensure-vt tensor)))
    (if (and (vt-complex-dtype-p (vt-dtype in)) args)
        (error "vt-average: 复数路径第一轮不支持参数 ~a（CONVENTIONS §16.11）" args)
        ;; 注意：CL 的 (apply f (list x) args) 会把**列表 (x) 本身**当作第一个
        ;; 实参传给 f（apply 只展开最后一个实参），必须用 cons 拼接。
        (apply (%cx-orig 'vt-average) (cons (if (vt-p tensor) tensor in) args)))))

(defun vt-round (tensor &key (divisor 1) out dtype)
  "逐元素四舍五入（对标 numpy.round）。

   复数（§16.5）：**两个分量各自**按「最近偶数」取整（numpy 的 round 对复数
   即分量取整），dtype 保持复数。`:divisor` 只接受**实数**（复数除数无定义，
   报错）；语义为 `round(分量/divisor)*divisor`，与实数路径一致。
   NaN / ±Inf 原样传播。  :out 契约同 vt-+。"
  (if (not (%cx-tensor-p tensor))
      (funcall (%cx-orig 'vt-round) tensor :divisor divisor :out out :dtype dtype)
      (let ((in (ensure-vt tensor))
            (dt (or dtype (vt-dtype tensor))))
        (when (complexp divisor)
          (error "vt-round: 复数路径的 :divisor 必须是实数（CONVENTIONS §16.5）"))
        (let ((d (abs divisor)))
          (vt-map (lambda (z)
                    (if (or (%cx-nan-p z) (%cx-inf-p z))
                        z
                        (let ((re (realpart z)) (im (imagpart z)))
                          (complex (* (round (/ re d)) d)
                                   (* (round (/ im d)) d)))))
                  in :out out :dtype dt)))))

;;; ==================================================================
;;; 10. 扫描（cumsum / cumprod）
;;; ==================================================================

(defun %cx-scan (tensor axis op out dtype op-name)
  "沿 AXIS（nil=展平）的前缀扫描，op ∈ :sum :prod。

   类型特化：原来对 (simple-array * (*)) 做泛型 aref + 每元素 vt-cast
   （1M 元素 cumsum = 288ms）。现在按元素类型绑定成 simple-array，
   内核直接读写复数槽位；写回用一次 vt-copy-into（非连续 out 才发生）。"
  (with-float-safe
    (let* ((in (ensure-vt tensor))
           (rank (length (vt-shape in)))
           (flat (%cx-flat in))
           (data (vt-data flat))
           (shape (vt-shape flat))
           (dt (or dtype (vt-dtype in)))
           ;; AXIS=NIL 时 numpy 语义是「先展平再累积」，结果必须是 1-D；
           ;; v1.2 修复：此前沿用 (vt-shape in)，对 2-D 输入会返回同形结果
           ;; （数值仍是展平顺序的前缀和，但形状错了 → 与 numpy 不兼容）。
           (out-shape (if axis (vt-shape in) (list (vt-size in))))
           (res (if out (vt-check-out out out-shape dt :op-name op-name)
                    (make-vt out-shape 0 :dtype dt)))
           (res-flat (%cx-flat res))
           (ax (if axis (vt-normalize-axis axis rank) nil))
           (same-type (eq (array-element-type data)
                          (array-element-type (vt-data res-flat)))))
      (if same-type
          (progn
            (%cx-scan-run data (vt-data res-flat) shape ax op)
            (unless (eq res-flat res) (vt-copy-into res res-flat)))
          ;; dtype 不同（用户显式要 c64/f64 另一精度）：通用回退
          (let ((rdata (vt-data res-flat)))
            (if (null ax)
                (let ((n (the fixnum (vt-size flat))))
                  (declare (type fixnum n))
                  (when (plusp n)
                    (let ((acc (aref data 0)))
                      (setf (aref rdata 0) (vt-cast acc dt))
                      (dotimes (i (the fixnum (1- n)))
                        (declare (type fixnum i))
                        (let ((p (the fixnum (1+ i))))
                          (declare (type fixnum p))
                          (setf acc (if (eq op :sum)
                                        (+ acc (aref data p))
                                        (* acc (aref data p))))
                          (setf (aref rdata p) (vt-cast acc dt)))))))
                (multiple-value-bind (starts ax-stride dim-ax)
                    (%cx-fiber-starts shape ax)
                  (loop for st across starts
                        do (when (plusp dim-ax)
                             (let ((acc (aref data st))
                                   (p (the fixnum st)))
                               (declare (type fixnum p))
                               (setf (aref rdata p) (vt-cast acc dt))
                               (dotimes (j (the fixnum (1- dim-ax)))
                                 (declare (ignore j))
                                 (incf p ax-stride)
                                 (setf acc (if (eq op :sum)
                                               (+ acc (aref data p))
                                               (* acc (aref data p))))
                                 (setf (aref rdata p) (vt-cast acc dt))))))))
            (unless (eq res-flat res) (vt-copy-into res res-flat))))
      res)))

(defun %cx-scan-run (src dst shape ax op)
  "类型特化的前缀扫描：SRC/DST 元素类型一致（complex64 或 complex128）。

   语义（对标 numpy 的 accumulate）：**每条 fiber 的首元素直接拷贝**，从第二个
   元素起才做 OP。别图省事从单位元起算 —— 会引入两处静默差异：
     · cumsum：`0 + (-0.0)` 得 `+0.0`，把输入的 -0.0 洗成正零
       （numpy.cumsum([-0.]) = [-0.]）；
     · cumprod：`1 * (inf+0i)` 走复数乘法公式 `(1·inf − 0·0, 1·0 + 0·inf)`
       = `inf+nan i`，凭空多出一个虚部 NaN，且该 NaN 会污染后续全部元素
       （numpy.cumprod([inf+0j, 2+0j]) = [inf+0j, inf+nanj]，首元素是干净的）。
   注意这与**归约**方向相反：numpy 的 sum/prod 确实从单位元起算
   （np.prod([inf+0j]) = inf+nanj），那条路径在 %CX-REDUCE 里，不要一起改。"
  (%cx-dispatch src s zero
    (%cx-dispatch dst d zero2
      (declare (ignore zero2)
               (ignorable zero))
      (if (null ax)
          (let ((n (the fixnum (length s))))
            (declare (type fixnum n))
            (when (plusp n)
              ;; 首元素直拷（理由见函数注释），从第二个起才累加
              (let ((acc (aref s 0)))
                (setf (aref d 0) acc)
                (dotimes (i (the fixnum (1- n)))
                  (declare (type fixnum i))
                  (let ((p (the fixnum (1+ i))))
                    (declare (type fixnum p))
                    (setf acc (if (eq op :sum)
                                  (+ acc (aref s p))
                                  (* acc (aref s p))))
                    (setf (aref d p) acc))))))
          (multiple-value-bind (head dim ax-stride) (%cx-fiber-geom shape ax)
            (declare (type fixnum head dim ax-stride))
            (dotimes (h head)
              (declare (type fixnum h))
              ;; H 个「前导块」，每块 DIM*AX-STRIDE 个元素（前导维已全展开）
              (let ((hb (the fixnum (* h dim ax-stride))))
                (declare (type fixnum hb))
                ;; ⚠ 必须再遍历 AX 轴之后的 AX-STRIDE 个位置 —— 它们是**另外
                ;; AX-STRIDE 条**互不相交的 fiber（如 axis=0 的其余各列）。
                ;; v0.5.x 修复：原实现少了这一层，axis < rank-1 时只算了第一条
                ;; fiber，其余位置保持 make-vt 的 0 → 静默错值（§16.29）。
                ;; 归约内核（%def-cx-reducer / nanarg / sort-axis / nanmedian）
                ;; 一直是对的，它们都有这层 TI 循环，只有扫描内核漏了。
                (dotimes (ti ax-stride)
                  (declare (type fixnum ti))
                  (let ((p (the fixnum (+ hb ti))))
                    (declare (type fixnum p))
                    (when (plusp dim)
                      ;; 首元素直拷（理由见函数注释），此后沿轴累加
                      (let ((acc (aref s p)))
                        (setf (aref d p) acc)
                        (incf p ax-stride)
                        (dotimes (j (the fixnum (1- dim)))
                          (declare (ignore j))
                          (setf acc (if (eq op :sum)
                                        (+ acc (aref s p))
                                        (* acc (aref s p))))
                          (setf (aref d p) acc)
                          (incf p ax-stride)))))))))))))

(defun vt-cumsum (tensor &key axis dtype out)
  "沿 AXIS 的累积和（对标 numpy.cumsum）。AXIS=NIL 展平扫描。
   复数保持复数 dtype。:out 契约同 vt-+。"
  (if (%cx-tensor-p tensor)
      (%cx-scan tensor axis :sum out dtype "VT-CUMSUM")
      (funcall (%cx-orig 'vt-cumsum) tensor :axis axis :dtype dtype :out out)))

(defun vt-cumprod (tensor &key axis dtype out)
  "沿 AXIS 的累积积（对标 numpy.cumprod）。复数保持复数 dtype。"
  (if (%cx-tensor-p tensor)
      (%cx-scan tensor axis :prod out dtype "VT-CUMPROD")
      (funcall (%cx-orig 'vt-cumprod) tensor :axis axis :dtype dtype :out out)))

;;; ==================================================================
;;; 11. 排序 / argsort / 中位数（字典序 + NaN 末尾，§16.8）
;;; ==================================================================

(declaim (inline %cx-sort-less))

(defun %cx-sort-less (a b)
  "复数排序比较：NaN 一律排末尾；其余按字典序（实部 → 虚部）。"
  (cond ((%cx-nan-p a) nil)
        ((%cx-nan-p b) t)
        (t (%clt a b))))

(defun %cx-sort-axis (tensor axis &key index)
  "沿 AXIS 排序（INDEX 为真时返回下标）。返回新的 C 连续张量。
   AXIS=NIL 时展平（与 clvt 既有 vt-sort :axis nil 语义一致）。

   外层统一屏蔽浮点陷阱：比较器 %CX-SORT-LESS 用的是**免屏蔽**快判
   （%CX-NAN-P），必须由调用方保证陷阱已屏蔽，否则 NaN 参与比较会抛
   FLOATING-POINT-INVALID-OPERATION（complex-test 的 sort-nan 用例即此）。"
  (with-float-safe
    (%cx-sort-axis-inner tensor axis :index index)))

(defmacro %def-cx-fiber-sorter (name lt)
  "生成原地升序排序内核 NAME(V BASE N AUX)：V/AUX 同类型，升序 = 字典序 + NaN 末尾。

   相比旧实现（每元素 cons 一个 (值 . 下标) 再 STABLE-SORT）：
   零 cons、零 funcall —— 比较器 %CX-SORT-LESS 直接内联进归并循环，
   1M 元素实测 1292ms → 见 test/complex-bench.lisp。"
  `(defun ,name (v base n aux)
     (declare (type (simple-array ,lt (*)) v aux)
              (type fixnum base n)
              (optimize (speed 3) (safety 1)))
     (cond
       ((<= n 1) nil)
       ;; 小 fiber：插入排序（免 scratch 开销）
       ((<= n 16)
        (loop for ii fixnum from 1 below n do
          (let ((x (aref v (+ base ii)))
                (j (1- ii)))
            (declare (type fixnum j))
            (loop while (and (>= j 0)
                             (%cx-sort-less x (aref v (+ base j))))
                  do (setf (aref v (+ base (1+ j))) (aref v (+ base j)))
                     (decf j))
            (setf (aref v (+ base (1+ j))) x))))
       (t
        ;; 自底向上稳定归并（无递归、顺序访存）
        (let ((w 1))
          (declare (type fixnum w))
          (loop while (< w n) do
            (let ((lo 0))
              (declare (type fixnum lo))
              (loop while (< lo n) do
                (let* ((mid (min (+ lo w) n))
                       (hi (min (+ lo w w) n))
                       (i lo)
                       (j mid)
                       (k lo))
                  (declare (type fixnum mid hi i j k))
                  (loop while (and (< i mid) (< j hi)) do
                    (if (%cx-sort-less (aref v (+ base j)) (aref v (+ base i)))
                        (progn (setf (aref aux k) (aref v (+ base j))) (incf j))
                        (progn (setf (aref aux k) (aref v (+ base i))) (incf i)))
                    (incf k))
                  (loop while (< i mid) do
                    (setf (aref aux k) (aref v (+ base i))) (incf i) (incf k))
                  (loop while (< j hi) do
                    (setf (aref aux k) (aref v (+ base j))) (incf j) (incf k))
                  (replace v aux :start1 (+ base lo) :end1 (+ base hi)
                             :start2 lo :end2 hi))
                (incf lo (* 2 w))))
            (setf w (* 2 w))))))))

(%def-cx-fiber-sorter %cx-sort-fiber-f64 (complex double-float))
(%def-cx-fiber-sorter %cx-sort-fiber-f32 (complex single-float))

(defun %cx-sort-fiber (data base n)
  "按元素类型分派到原地排序内核（非复数缓冲区不会到这里）。"
  (let ((et (array-element-type data)))
    (cond ((equal et '(complex double-float))
           (let ((aux (make-array (max n 1) :element-type '(complex double-float))))
             (%cx-sort-fiber-f64 data base n aux)))
          ((equal et '(complex single-float))
           (let ((aux (make-array (max n 1) :element-type '(complex single-float))))
             (%cx-sort-fiber-f32 data base n aux)))
          (t (error "%cx-sort-fiber: 非复数缓冲区 ~a" et)))))

(defun %cx-sort-fiber-strided (data base stride n)
  "对 DATA 中「起点 BASE、步长 STRIDE、长度 N」的一条 fiber 原地排序。

   连续副本 + 归并内核复用：先把 strided fiber 收进临时连续数组，
   排完再写回（临时数组按元素类型分配，与 %CX-SORT-FIBER 同源）。"
  (let ((et (array-element-type data)))
    (cond
      ((equal et '(complex double-float))
       (let ((tmp (make-array (max n 1) :element-type '(complex double-float))))
         (dotimes (i n) (setf (aref tmp i) (aref data (+ base (* i stride)))))
         (%cx-sort-fiber tmp 0 n)
         (dotimes (i n) (setf (aref data (+ base (* i stride))) (aref tmp i)))))
      ((equal et '(complex single-float))
       (let ((tmp (make-array (max n 1) :element-type '(complex single-float))))
         (dotimes (i n) (setf (aref tmp i) (aref data (+ base (* i stride)))))
         (%cx-sort-fiber tmp 0 n)
         (dotimes (i n) (setf (aref data (+ base (* i stride))) (aref tmp i)))))
      (t (error "%cx-sort-fiber-strided: 非复数缓冲区 ~a" et)))))

(defun %cx-sort-axis-inner (tensor axis &key index)
  "排序实现体（陷阱已由 %CX-SORT-AXIS 屏蔽）。

   选路（两条都是「先落成连续副本，再按 fiber 处理」）：
     · sort（无 INDEX）：fiber 直接用归并内核原地排（零 cons / 零 funcall）；
     · argsort（INDEX=T）：fiber 建 fixnum 下标向量后 STABLE-SORT，
       比较器直读数据（不再是每元素一个 (值 . 下标) cons 对）。"
  (let* ((in (ensure-vt tensor))
         (flat (%cx-flat in))
         (shape (vt-shape flat))
         (rank (length shape))
         (dt (vt-dtype in))
         (ax (if axis (vt-normalize-axis axis rank) nil)))
    (if (null ax)
        ;; ---------- 展平 ----------
        (let* ((n (vt-size flat))
               (src (vt-data flat)))
          (if index
              (let* ((res (make-vt (list n) 0 :dtype :int64))
                     (rd (vt-data res))
                     (idx (make-array n :element-type 'fixnum)))
                (declare (type (simple-array fixnum (*)) idx))
                (dotimes (i n) (setf (aref idx i) i))
                (stable-sort idx
                             (lambda (a b)
                               (%cx-sort-less (aref src a) (aref src b))))
                (dotimes (i n) (setf (aref rd i) (aref idx i)))
                res)
              (let* ((res (%cx-flat (vt-copy flat)))
                     (rd (vt-data res)))
                (%cx-sort-fiber rd 0 n)
                (setf (vt-shape res) (list n)
                      (vt-strides res) (list 1)
                      (vt-offset res) 0)
                res)))
        ;; ---------- 单轴 ----------
        (multiple-value-bind (head dim ax-stride) (%cx-fiber-geom shape ax)
          (declare (type fixnum head dim ax-stride))
          (if index
              (let* ((res (make-vt shape 0 :dtype :int64))
                     (rd (vt-data res))          ; res 是新建连续张量，offset=0
                     (src (vt-data flat))
                     (idx (make-array (max dim 1) :element-type 'fixnum)))
                (declare (type (simple-array fixnum (*)) idx))
                (dotimes (h head)
                  (declare (type fixnum h))
                  (let ((row (the fixnum (* h dim ax-stride))))
                    (declare (type fixnum row))
                    (dotimes (t2 ax-stride)
                      (declare (type fixnum t2))
                      (dotimes (j dim) (setf (aref idx j) j))
                      (stable-sort idx
                                   (lambda (a b)
                                     (%cx-sort-less
                                      (aref src (+ row t2 (the fixnum (* a ax-stride))))
                                      (aref src (+ row t2 (the fixnum (* b ax-stride)))))))
                      (dotimes (j dim)
                        (setf (aref rd (+ row t2 (the fixnum (* j ax-stride))))
                              (aref idx j))))))
                res)
              (let* ((res (%cx-flat (vt-copy flat)))
                     (rd (vt-data res)))
                (dotimes (h head)
                  (declare (type fixnum h))
                  (let ((row (the fixnum (* h dim ax-stride))))
                    (declare (type fixnum row))
                    (dotimes (t2 ax-stride)
                      (declare (type fixnum t2))
                      (%cx-sort-fiber-strided rd (+ row t2) ax-stride dim))))
                res))))))

(defun vt-sort (tensor &key (axis -1))
  "升序排序（对标 numpy.sort）。AXIS 缺省 -1；AXIS=NIL 展平。

   复数（§16.8）：numpy 2.x 对复数采用**字典序**（先实部、再虚部），
   NaN 排末尾；本函数与之一致。"
  (if (%cx-tensor-p tensor)
      (%cx-sort-axis (ensure-vt tensor) axis)
      (funcall (%cx-orig 'vt-sort) tensor :axis axis)))

(defun vt-argsort (tensor &key (axis -1))
  "返回升序排序后的下标（对标 numpy.argsort）。复数按字典序（同 vt-sort）。"
  (if (%cx-tensor-p tensor)
      (%cx-sort-axis (ensure-vt tensor) axis :index t)
      (funcall (%cx-orig 'vt-argsort) tensor :axis axis)))

(declaim (inline %cx-elem-nan-p))
(defun %cx-elem-nan-p (z)
  "复数元素是否含 NaN（任一分量为 NaN 即为真）。与 numpy 一致：复数里
   只要有一个分量是 NaN，整个元素按「NaN」对待（median 要传播、nanmedian 要跳过）。"
  (or (%cx-nan-p (%creal z)) (%cx-nan-p (%cimag z))))

(defun %cx-has-nan-p (data off stride count)
  "沿 DATA 从 OFF 起、步长 STRIDE 取 COUNT 个元素，判断是否有 NaN。
   ⚠ 内部用 `(= x x)` 判 NaN（%CX-NAN-P），调用方**必须**在
   WITH-FLOAT-SAFE 里调用，否则开着 FP 陷阱时会抛 FLOAT-INVALID。"
  (declare (type fixnum off stride count))
  (with-float-safe
    (loop for i of-type fixnum below count
          for p of-type fixnum = (+ off (* i stride))
          thereis (%cx-elem-nan-p (aref data p)))))

(defun vt-median (tensor &key axis keepdims)
  "中位数（对标 numpy.median）。复数按字典序排序后取中位；
   偶数长度取两中位的复数均值（与 numpy 一致）。

   AXIS=NIL 时与实数路径 / numpy 一致：**先展平再取全局中位**（0 维标量），
   KEEPDIMS 时形状为全 1（3×4 → (1 1)）。
   轴长（或总元素数）为 0 → NaN+NaN i（numpy 同）。

   v1.3 修复两处与实数路径的分叉（都是静默错值，不报错）：
   1. AXIS=NIL 时原实现取 `(1- rank)`，即偷偷按**末轴**归约 ——
      (3 4) 输入返回 3 个中位、形状 (3)，而实数路径与 numpy 都返回 1 个标量；
   2. 空 fiber 时 `lo=(floor -1 2)=-1`，配合 (safety 1) 静默读出 0，
      返回 0+0i 而非 NaN（实数路径这边是显式的 nanv）。"
  (if (not (%cx-tensor-p tensor))
      (funcall (%cx-orig 'vt-median) tensor :axis axis :keepdims keepdims)
      ;; 与实数实现保持一致：整段包 with-float-safe。-0.0/inf 参与的中位数会
      ;; 触发 FLOAT-INVALID（nan 判定、inf + -inf），用户开着 FP 陷阱时不能抛。
      (with-float-safe
       (let* ((in (ensure-vt tensor))
              (rank (length (vt-shape in)))
              (dt (vt-dtype in)))
        (labels ((nan-elem ()
                   ;; 与输入同精度的 NaN+NaNj
                   (let ((nv (coerce +vt-dfloat-nan+
                                     (vt-dtype->lisp-type
                                      (vt-complex-real-dtype dt)))))
                     (complex nv nv))))
          (if (null axis)
              ;; ---------------- 全局归约（axis=nil ≡ 先 ravel）----------------
              (let* ((flat (vt-ravel in))
                     (n (the fixnum (vt-size flat)))
                     (out-shape (when keepdims (make-list rank :initial-element 1)))
                     (res (make-vt out-shape 0 :dtype dt)))
                (let ((sd (vt-data flat))
                      (so (the fixnum (vt-offset flat))))
                  (if (or (zerop n) (%cx-has-nan-p sd so 1 n))
                      ;; numpy：整表含 NaN → 结果 NaN（不忽略）；本库实数路径同样返回 NaN
                      (setf (aref (vt-data res) (vt-offset res)) (nan-elem))
                      (let* ((sorted (vt-sort flat))
                             (data (vt-data sorted))
                             (o (the fixnum (vt-offset sorted)))
                             (lo (the fixnum (floor (1- n) 2)))
                             (hi (the fixnum (floor n 2))))
                        ;; 平均值在 double 下算完再 cast：c64 的 res 元素类型是
                      ;; (complex single-float)，直接 setf 会撞类型断言。
                      (setf (aref (vt-data res) (vt-offset res))
                              (if (= lo hi)
                                  (aref data (+ o lo))
                                  (vt-cast (* 0.5d0 (+ (aref data (+ o lo))
                                                       (aref data (+ o hi))))
                                           dt))))))
                res)
              ;; ---------------- 轴归约 ----------------
              (let* ((ax (vt-normalize-axis axis rank))
                     (sorted (vt-sort in :axis ax))
                     (n (nth ax (vt-shape sorted)))
                     (out-shape (%cx-keepdims-shape (vt-shape in) (list ax) keepdims))
                     (res (make-vt out-shape 0 :dtype dt))
                     (flat (%cx-flat sorted))
                     (data (vt-data flat))
                     (shape (vt-shape flat)))
                (if (zerop n)
                    ;; 空 fiber：numpy 给 NaN+NaN i，不是 0
                    (let ((rd (vt-data res))
                          (ro (the fixnum (vt-offset res)))
                          (nanv (nan-elem)))
                      (dotimes (k (vt-size res))
                        (setf (aref rd (+ ro k)) nanv))
                      res)
                    (multiple-value-bind (starts ax-stride dim-ax)
                        (%cx-fiber-starts shape ax)
                      (declare (ignore dim-ax))
                      (let ((vals (make-array (length starts)))
                            (lo (the fixnum (floor (1- n) 2)))
                            (hi (the fixnum (floor n 2))))
                        (let ((nanv (nan-elem)))
                          (loop for s across starts
                                for i from 0
                                do (setf (aref vals i)
                                         (if (%cx-has-nan-p data s ax-stride n)
                                             nanv   ; numpy：该 fiber 含 NaN → 该处 NaN
                                             (if (= lo hi)
                                                 (aref data (+ s (* lo ax-stride)))
                                                 (* 0.5d0
                                                    (+ (aref data (+ s (* lo ax-stride)))
                                                       (aref data (+ s (* hi ax-stride))))))))))
                        (%cx-write-flat res vals dt)))))))))))

;;; ==================================================================
;;; 11.5 矩阵乘法 / 点积 / 外积（专用内核）
;;;
;;; 为什么不走 vt-einsum：einsum 的 supported-types 只有 4 种实数类型，
;;; 其结果缓冲区按这 4 种分配 —— 复数输入会得到 double-float 缓冲并
;;; 在写回时报 "is not of type DOUBLE-FLOAT"。第一轮为 matmul 家族
;;; 写一个独立的通用内核（先 contiguous 落地，再三重/二重循环），
;;; 语义与 einsum 的 `...ij,...jk->...ik` 完全一致（§16.13）。
;;; ==================================================================

(defun %cx-matmul-2d (a b)
  "复数 (m,k)@(k,n) → (m,n)，A/B 均为 C 连续。

   类型特化三重循环（i-k-j 次序，内层连续访存）。实测 256x256：
   泛型版 1748ms → 特化版见 test/complex-bench.lisp。

   ⚠ A/B 的 **offset 必须透传**（第五轮 §16.22 修复）：
   VT-CONTIGUOUS 对「本身已连续」的切片视图是零拷贝，offset 保留原值
   （如 (vt-slice (vt-ravel a) i j) 得到的 1D 片段）。原实现按扁平下标
   从 0 起算，批量 matmul 的每个 batch 切片都会读到基数组开头 → 结果
   静默错误（einsum 与 matmul 差 ~3.3）。同一根因见 manip.lisp 的
   「扁平化 + 强制 offset=0」一节。"
  (let* ((m (first (vt-shape a)))
         (k (second (vt-shape a)))
         (n (second (vt-shape b)))
         (res (make-vt (list m n) 0 :dtype (vt-dtype a))))
    (when (plusp (* m k n))
      (%cx-matmul-run (vt-data a) (vt-data b) (vt-data res)
                      (the fixnum (vt-offset a))
                      (the fixnum (vt-offset b))
                      (the fixnum (vt-offset res))
                      m k n))
    res))

;;; ------------------------------------------------------------------
;;; 11.5.1 复数 GEMM 的「行内核」（对照 linalg.lisp 的 %MATMUL-DF-ROWS）
;;;
;;; 旧内核（第一轮）是 i-p-j 三重循环，但每个元素都由 (i,j,p) **重算**
;;; 扁平下标，并且整段落在 %CX-DISPATCH 的 (safety 1) 里。实测（n=256，
;;; 单线程，只计内核本身、不含 make-vt/contiguous，见 §16.27 消融表）：
;;;
;;;     旧内核（safety 1 + 下标重算 + 无展开）        24.2 ms
;;;     + safety 0（索引可证不越界，§16.14.1 第 2 类）  22.0 ms
;;;     + 指针递推（行起点只算一次，内层只 incf）        14.1 ms
;;;     + j 维 4 路展开                                13.5 ms
;;;
;;; 三项收益互不重叠，且**全部来自代码形态**，与「复数」无关 —— 实数
;;; 内核 %MATMUL-DF-ROWS 一开始就是这三件事（safety 0 / 指针递推 /
;;; 4 路展开），差距的根因就在这里，而不是复数算术本身。
;;;
;;; 为什么**不**做 SoA（拆成 re/im 两个 double 数组）或 B 打包：
;;; 实测更慢（SoA+打包 22.9ms、SoA 单累加器 75.2ms，均劣于 13.5ms）。
;;; SBCL 的复数 * / + 是开码的：specialized 数组里的 (complex double-float)
;;; 按 16 字节就地读写，不装箱、不 GC 分配；反倒是拆出来的索引运算更贵。
;;; 因此这里保留「boxed complex 缓冲区 + * 算子」，只对齐循环形态。
;;;
;;; 索引安全性：a-row/c-row 由 i 维推出，b-ptr 由 p 维推出，c-ptr/b-ptr
;;; 与 j 维同步 +1 —— 上界分别是 a-off+m*k / c-off+m*n / b-off+k*n，
;;; 与缓冲区长度完全一致，故可用 (safety 0)（与 %MATMUL-DF-ROWS 同级）。
;;; ⚠ 本函数**不清零也不重置** C：调用方负责，语义是「C += A@B」。
;;;   这样 vt-matmul（先清零）与 einsum 的增量累加（§16.21）能共用一份内核。
;;; ------------------------------------------------------------------

(defmacro %def-cx-matmul-rows (name ctype)
  "生成一个复数 GEMM 行内核 NAME（元素类型 CTYPE，如 (complex double-float)）。
   签名与 %MATMUL-*-ROWS 一致：(A B C K N A-OFF B-OFF C-OFF I-START I-END)。"
  `(progn
     (declaim (inline ,name))
     (defun ,name (a b c k n a-off b-off c-off i-start i-end)
       (declare (type (simple-array ,ctype (*)) a b c)
                (type fixnum k n a-off b-off c-off i-start i-end)
                (optimize (speed 3) (safety 0) (debug 0) (compilation-speed 0)))
       (let ((n4 (the fixnum (* (floor n 4) 4))))
         (declare (type fixnum n4))
         (loop for i of-type fixnum from i-start below i-end do
           (let ((a-row (the fixnum (+ a-off (the fixnum (* i k)))))
                 (c-row (the fixnum (+ c-off (the fixnum (* i n))))))
             (declare (type fixnum a-row c-row))
             (loop for p of-type fixnum from 0 below k do
               (let ((av (aref a (the fixnum (+ a-row p))))
                     (b-ptr (the fixnum (+ b-off (the fixnum (* p n)))))
                     (c-ptr c-row))
                 (declare (type ,ctype av) (type fixnum b-ptr c-ptr))
                 ;; 主循环：j 维 4 路展开，c-ptr/b-ptr 与 j 同步 +1（不重算下标）
                 (loop for j of-type fixnum from 0 below n4 by 4 do
                   (incf (aref c c-ptr)
                         (the ,ctype (* av (aref b b-ptr))))
                   (incf (aref c (the fixnum (1+ c-ptr)))
                         (the ,ctype (* av (aref b (the fixnum (1+ b-ptr))))))
                   (incf (aref c (the fixnum (+ c-ptr 2)))
                         (the ,ctype (* av (aref b (the fixnum (+ b-ptr 2))))))
                   (incf (aref c (the fixnum (+ c-ptr 3)))
                         (the ,ctype (* av (aref b (the fixnum (+ b-ptr 3))))))
                   (incf c-ptr 4)
                   (incf b-ptr 4))
                 ;; 尾部（n 不是 4 的倍数）
                 (loop for j of-type fixnum from n4 below n do
                   (incf (aref c c-ptr)
                         (the ,ctype (* av (aref b b-ptr))))
                   (incf c-ptr)
                   (incf b-ptr))))))))))

(%def-cx-matmul-rows %cx-matmul-cdd-rows (complex double-float))
(%def-cx-matmul-rows %cx-matmul-css-rows (complex single-float))

(defun %cx-matmul-run (ad bd rd a-off b-off c-off m k n)
  "C[m,n] = A[m,k] @ B[k,n]（C 连续、复数同精度；三者的 offset 由调用方给出）。

   三步：① 按元素类型选定行内核；② 清零 C；③ 按 *ENABLE-PARALLEL-MATMUL*
   与 *MATMUL-PARALLEL-THRESHOLD* 决定单线程还是按行切块多线程 ——
   这段调度**直接复用实数线的 CALL-PARALLEL-GEMM**（linalg.lisp），
   因此复数与实数的并行语义（阈值、线程数、行切法）完全一致。

   清零与累加**刻意分成两趟**：实测「把 p=0 折进内核直接写 C」反而更慢
   （n=256: 14.6ms vs 13.5ms），因为写了 4 份 j 展开的初始化代码、
   挤掉了内层循环的寄存器分配。"
  (%cx-dispatch ad a zero
    (%cx-dispatch bd b zero2
      (%cx-dispatch rd c zero3
        (declare (ignorable zero2 zero3))
        ;; ① 清零（整块一趟，连续访存）
        (dotimes (i m)
          (declare (type fixnum i))
          (let ((c-row (the fixnum (+ c-off (the fixnum (* i n))))))
            (declare (type fixnum c-row))
            (dotimes (j n)
              (declare (type fixnum j))
              (setf (aref c (the fixnum (+ c-row j))) zero))))
        ;; ②③ 选内核 + 调度
        (if (equal (array-element-type ad) '(complex double-float))
            (call-parallel-gemm #'%cx-matmul-cdd-rows
                                a b c m k n a-off b-off c-off)
            (call-parallel-gemm #'%cx-matmul-css-rows
                                a b c m k n a-off b-off c-off))))))

(defun %cx-dot-1d (a b)
  "复数向量点积 → 0 维张量（不取共轭，对标 numpy.dot）。"
  (let* ((n (vt-size a))
         (ad (vt-data a)) (bd (vt-data b))
         ;; ⚠ 必须带 offset：VT-CONTIGUOUS 对已连续视图是零拷贝，
         ;; 其 offset 未必为 0（如 a[3:] 的切片）。
         (ao (the fixnum (vt-offset a)))
         (bo (the fixnum (vt-offset b)))
         (acc (vt-dtype-default-value (vt-dtype a))))
    (dotimes (i n)
      (setf acc (+ acc (* (aref ad (the fixnum (+ ao i)))
                          (aref bd (the fixnum (+ bo i)))))))
    (make-vt nil acc :dtype (vt-dtype a))))

;;; 11.6 einsum 复数 GEMM / 内积快路径（§16.21）
;;;
;;; 复数 einsum 的正确性由通用累加状态机保证，但那条路径每个元素都要
;;; 走「补码递推 + 逐张量 aref」，256×256 的 `ij,jk->ik` 实测 2526ms，
;;; 而 vt-matmul 只要 44ms（≈57×）。这里给 einsum 提供两个内核：
;;;
;;;   %CX-GEMM-STRIDED —— 带 strides 的 A@B 增量累加（einsum 的 BMM 通道）
;;;   %CX-EINSUM-DOT   —— 全收缩内积 "i,i->"
;;;
;;; 两者都由 complex.lisp 在加载期注册到 linalg.lisp 的钩子上，
;;; 因此 linalg.lisp（先编译）不需要知道复数层存在（§16.12 分派约定）。
;;; ==================================================================

(defun %cx-gemm-strided (ad bd rd m k n a-off b-off c-off
                         sa-i sa-j sb-j sb-k sc-i sc-k)
  "C += A @ B（复数，增量累加；C 由调用方清零）。

   A: m×k，行步长 SA-I、列步长 SA-J；B: k×n，SB-J/SB-K；
   C: m×n，SC-I/SC-K。语义 = einsum 的 `...ij,...jk->...ik`（不取共轭）。

   SA-J、SB-K、SC-K 全为 1 时（A 行内连续 + C 连续，即 `ij,jk->ik`
   的常规情形）走 i-p-j 特化循环，内层连续访存 —— 与 vt-matmul 同款
   内核；否则（转置 / 广播 / 非连续 :out）走通用 strided 循环，
   仍是元素类型特化代码，比通用状态机快一个数量级。

   若布局**进一步**与「连续 NN」逐项相同（SA-I=k、SB-J=n、SC-I=n，
   即 A 行距 = k、B 行距 = n、C 行距 = n），则直接委托给
   %CX-MATMUL-*-ROWS —— 与 vt-matmul 用的是**同一份行内核**（4 路展开
   + 指针递推）。这样两边的累加次序（i→p→j 递增）逐位一致，
   einsum 与 vt-matmul 的结果仍然 bit-for-bit 相等（§16.21.5 的验收项）。"
  (%cx-dispatch ad a zero
    (%cx-dispatch bd b zero2
      (%cx-dispatch rd c zero3
        (declare (ignorable zero2 zero3))
        (cond
          ;; (a) 连续 NN：与 vt-matmul 共用行内核（最快）
          ((and (= sa-j 1) (= sb-k 1) (= sc-k 1)
                (= sa-i k) (= sb-j n) (= sc-i n))
           (if (equal (array-element-type ad) '(complex double-float))
               (%cx-matmul-cdd-rows a b c k n a-off b-off c-off 0 m)
               (%cx-matmul-css-rows a b c k n a-off b-off c-off 0 m)))
          ;; (b) A 行内连续 + C 连续，但行距不同（切片/子块/非连续 :out）
          ((and (= sa-j 1) (= sb-k 1) (= sc-k 1))
           (dotimes (ii m)
             (declare (type fixnum ii))
             (let ((a-row (the fixnum (+ a-off (the fixnum (* ii sa-i)))))
                   (c-row (the fixnum (+ c-off (the fixnum (* ii sc-i))))))
               (declare (type fixnum a-row c-row))
               (dotimes (pp k)
                 (declare (type fixnum pp))
                 (let ((av (aref a (the fixnum (+ a-row pp))))
                       (b-row (the fixnum (+ b-off (the fixnum (* pp sb-j))))))
                   (declare (type fixnum b-row))
                   (dotimes (jj n)
                     (declare (type fixnum jj))
                     (incf (aref c (the fixnum (+ c-row jj)))
                           (* av (aref b (the fixnum (+ b-row jj)))))))))))
          ;; (c) 通用 strided（转置 / 广播）
          (t
           (dotimes (ii m)
             (declare (type fixnum ii))
             (let ((a-row (the fixnum (+ a-off (the fixnum (* ii sa-i)))))
                   (c-row (the fixnum (+ c-off (the fixnum (* ii sc-i))))))
               (declare (type fixnum a-row c-row))
               (dotimes (pp k)
                 (declare (type fixnum pp))
                 (let ((av (aref a (the fixnum (+ a-row (the fixnum (* pp sa-j))))))
                       (b-row (the fixnum (+ b-off (the fixnum (* pp sb-j))))))
                   (declare (type fixnum b-row))
                   (dotimes (jj n)
                     (declare (type fixnum jj))
                     (incf (aref c (the fixnum (+ c-row (the fixnum (* jj sc-k)))))
                           (* av (aref b (the fixnum (+ b-row
                                                        (the fixnum (* jj sb-k))))))))))))))))))

(defun %cx-einsum-dot (a b)
  "einsum \"i,i->\" 的复数全收缩内积，返回复数**标量**（不取共轭）。
   A/B 由调用方保证已连续（这里仍按 offset 索引，见 %CX-DOT-1D 的注记）。"
  (let* ((n (vt-size a))
         (ad (vt-data a)) (bd (vt-data b))
         (ao (the fixnum (vt-offset a)))
         (bo (the fixnum (vt-offset b))))
    (%cx-dispatch ad aa zero
      (%cx-dispatch bd bb zero2
        (declare (ignore zero2))
        (let ((acc zero)
              (pa ao)
              (pb bo))
          (declare (type fixnum pa pb))
          (dotimes (i n)
            (declare (type fixnum i))
            (setf acc (+ acc (* (aref aa pa) (aref bb pb))))
            (incf pa)
            (incf pb))
          acc)))))

(setf *cx-einsum-gemm-fn* #'%cx-gemm-strided
      *cx-einsum-dot-fn*  #'%cx-einsum-dot)

(defun %cx-batched-matmul (a b)
  "批量矩阵乘 (...,m,k)@(...,k,n) → (...,m,n)，支持 numpy 风格 batch 广播。

   逐 batch 调 %CX-MATMUL-RUN（带 offset 的 2D 内核），偏移量由 batch 轴的
   strides 递推得到 —— 因此下面两类此前的静默错误一并消失：
     · **offset 丢失**：原实现用 (vt-slice (vt-ravel a) ...) 切 batch，切片
       视图带着非零 offset，而 2D 内核从 0 起算 → 每个 batch 都读基数组开头，
       einsum 与 matmul 结果差 ~3.3（§16.22）。
     · **不支持广播**：(1,4,5)@(3,5,6) 原实现直接 reshape 失败；现在按
       维长 1 → stride 0 处理，与实数路径一致。"
  (let* ((sa (vt-shape a)) (sb (vt-shape b))
         (ra (length sa)) (rb (length sb))
         (ba (subseq sa 0 (- ra 2)))
         (bb (subseq sb 0 (- rb 2)))
         (bn (max (length ba) (length bb)))
         (npad-a (- bn (length ba)))
         (npad-b (- bn (length bb)))
         ;; numpy 语义：batch 维右对齐，缺的左侧补 1
         (ba* (append (make-list npad-a :initial-element 1) ba))
         (bb* (append (make-list npad-b :initial-element 1) bb))
         (batch (mapcar (lambda (x y)
                          (cond ((= x y) x)
                                ((= x 1) y)
                                ((= y 1) x)
                                (t (error "vt-matmul: 批量维不可广播：~a vs ~a"
                                          ba bb))))
                        ba* bb*))
         (m (nth (- ra 2) sa))
         (k (nth (1- ra) sa))
         (n (nth (1- rb) sb))
         (res (make-vt (append batch (list m n)) 0 :dtype (vt-dtype a)))
         (ad (vt-data a)) (bd (vt-data b)) (rd (vt-data res))
         (cs (vt-strides res)))
    (flet ((batch-strides (orig-dims orig-strs pad)
             "把 ORIG-DIMS/ORIG-STRS 右对齐到 BN 个 batch 轴（左侧补 0），
              并把「维长 1 但结果维 >1」的轴 stride 置 0（广播）。"
             (append (make-list pad :initial-element 0)
                     (loop for d in orig-dims
                           for s in orig-strs
                           for bd in (nthcdr pad batch)
                           collect (if (and (= d 1) (> bd 1)) 0 s)))))
      (let ((a-strs (batch-strides ba (subseq (vt-strides a) 0 (- ra 2)) npad-a))
            (b-strs (batch-strides bb (subseq (vt-strides b) 0 (- rb 2)) npad-b))
            (c-strs (subseq cs 0 bn))
            (a-off (the fixnum (vt-offset a)))
            (b-off (the fixnum (vt-offset b)))
            (c-off (the fixnum (vt-offset res))))
        (when (plusp (* m k n (reduce #'* batch :initial-value 1)))
          ;; 混合基数遍历 batch（零分配，递归深度 = batch 秩）
          (labels ((walk-batch (ax oa ob oc)
                     (cond
                       ((= ax bn)
                        (%cx-matmul-run ad bd rd oa ob oc m k n))
                       (t
                        (let ((dim (nth ax batch))
                              (stra (nth ax a-strs))
                              (strb (nth ax b-strs))
                              (strc (nth ax c-strs)))
                          (declare (type fixnum dim))
                          (dotimes (i dim)
                            (walk-batch (1+ ax)
                                (+ oa (* i stra))
                                (+ ob (* i strb))
                                (+ oc (* i strc)))))))))
            (walk-batch 0 a-off b-off c-off)))))
    res))

(defun %cx-matmul (a b)
  "复数 matmul 分派（语义同 numpy.matmul）。"
  (let* ((a (ensure-vt a)) (b (ensure-vt b))
         (ra (length (vt-shape a))) (rb (length (vt-shape b)))
         (dt (vt-promote-type (vt-dtype a) (vt-dtype b)))
         (ca (vt-contiguous a)) (cb (vt-contiguous b))
         (res
           (cond
             ((and (= ra 1) (= rb 1)) (%cx-dot-1d ca cb))
             ((and (= ra 2) (= rb 2)) (%cx-matmul-2d ca cb))
             ((and (= ra 2) (= rb 1))
              (let ((m (first (vt-shape ca))) (k (second (vt-shape ca))))
                (%cx-matmul-2d ca (vt-reshape cb (list k 1)))))
             ((and (= ra 1) (= rb 2))
              (let ((k (vt-size ca)) (n (second (vt-shape cb))))
                (vt-reshape (%cx-matmul-2d (vt-reshape ca (list 1 k)) cb) (list n))))
             (t (%cx-batched-matmul ca cb)))))
    (if (eq (vt-dtype res) dt) res (vt-astype res dt))))

(defun vt-matmul (a b &key dtype out)
  "矩阵乘法 / 批量矩阵乘法（对标 numpy.matmul）。

   复数（§16.13）：走本层专用内核（不做共轭，与 numpy.matmul 一致：
   共轭点积要用 vt-vdot 语义，即 (vt-dot (vt-conj a) b)）。
   实数路径不变（einsum/SIMD 内核）。"
  (if (or (%cx-deep-p a) (%cx-deep-p b))
      (let ((r (%cx-matmul a b)))
        (when dtype (setf r (vt-astype r dtype)))
        (if out (progn (vt-copy-into out r) out) r))
      (funcall (%cx-orig 'vt-matmul) a b :dtype dtype :out out)))

(defun vt-@ (a b &key dtype out)
  "vt-matmul 的运算符别名。"
  (vt-matmul a b :dtype dtype :out out))

(defun vt-dot (a b &key dtype out)
  "点积 / 内积（对标 numpy.dot）。复数：同 vt-matmul 语义，不取共轭。"
  (if (or (%cx-deep-p a) (%cx-deep-p b))
      (let ((r (%cx-matmul a b)))
        (when dtype (setf r (vt-astype r dtype)))
        (if out (progn (vt-copy-into out r) out) r))
      (funcall (%cx-orig 'vt-dot) a b :dtype dtype :out out)))

(defun vt-outer (a b &key (flatten t) dtype out)
  "外积（对标 numpy.outer）。复数：全部按分量乘法，dtype 保持复数。

   实现就是 `a.reshape(-1,1) * b.reshape(1,-1)`（广播乘法），
   不调用任何实数内核 —— 因此与元素类型无关。"
  (if (or (%cx-deep-p a) (%cx-deep-p b))
      (let* ((av (ensure-vt a))
             (bv (ensure-vt b))
             (af (if flatten (vt-ravel av) av))
             (bf (if flatten (vt-ravel bv) bv))
             (dt (or dtype (vt-promote-type (vt-dtype av) (vt-dtype bv))))
             (prod (vt-* (vt-reshape af (append (vt-shape af) (list 1)))
                         (vt-reshape bf (cons 1 (vt-shape bf)))))
             ;; 显式 :dtype 与自然提升不一致时才转换（避免多余拷贝）
             (r (if (eq (vt-dtype prod) dt) prod (vt-astype prod dt))))
        ;; ⚠ 原实现写成 (vt-reshape ... :dtype dt)：vt-reshape 只接受
        ;; (vt shape) 两个参数，多出的 :dtype 会变成第三个实参 →
        ;; "invalid number of arguments: 3"。复数外积**从来没跑通过**，
        ;; 而实数路径走原实现所以一直正常（§16.19 第四轮修复）。
        (if out
            (progn (vt-copy-into (vt-check-out out (vt-shape r) dt
                                               :op-name "vt-outer")
                                 r)
                   out)
            r))
      (funcall (%cx-orig 'vt-outer) a b :flatten flatten :dtype dtype :out out)))

(defun vt-trace (matrix &key (offset 0) axis1 axis2 dtype out)
  "矩阵迹（对角线之和，对标 numpy.trace）。复数保持复数 dtype。

   OFFSET / AXIS1 / AXIS2 语义见实数层同名函数（本层只做 dtype 保持）。"
  (if (%cx-deep-p matrix)
      (let* ((m (ensure-vt matrix))
             (dt (or dtype (vt-dtype m)))
             (d (vt-diagonal m :offset offset :axis1 axis1 :axis2 axis2))
             ;; 同实数层：对角线轴在末尾，沿 -1 收缩 → 保留其余轴
             (r (vt-sum d :axis -1 :dtype dt)))
        (if out (progn (vt-copy-into out r) out) r))
      (funcall (%cx-orig 'vt-trace) matrix
               :offset offset :axis1 axis1 :axis2 axis2
               :dtype dtype :out out)))

;;; ==================================================================
;;; 12. 范数（必然实数，§16.7）
;;; ==================================================================

(defun vt-norm (vt &key axis keepdims dtype out)
  "l2 范数（欧几里得范数）。

   复数（§16.7）：范数 = sqrt(Σ|z|²)，**结果恒为实数**
   （complex128 → float64、complex64 → float32）。
   实数路径不变（整数 → float64）。:out 契约同 vt-+。"
  (let ((in (ensure-vt vt)))
    (unless (vt-complex-dtype-p (vt-dtype in))
      (return-from vt-norm
        (funcall (%cx-orig 'vt-norm) vt :axis axis :keepdims keepdims
                 :dtype dtype :out out)))
    (let* ((dt (or dtype (vt-real-dtype-of (vt-dtype in)))))
      (when (vt-complex-dtype-p dt)
        (error "vt-norm: 复数范数的结果 dtype 必须是实数类型，收到 ~a" dt))
      (let* ((sq (vt-square (vt-abs in)))
             (ss (funcall (%cx-orig 'vt-sum) sq :axis axis :keepdims keepdims
                                                  :dtype dt)))
        (vt-sqrt ss :out out :dtype dt)))))

;;; ==================================================================
;;; 13. 未覆盖函数的确定性报错（§16.11）
;;; ==================================================================

(defmacro %cx-def-unsupported (names reason)
  "为 NAMES 中的函数生成「含复数入参即报错」的包装，其余调用原样转发。"
  `(dolist (name (list ,@(mapcar (lambda (n) `',n) names)))
     (let ((orig (%cx-orig name)))
       (setf (symbol-function name)
             (lambda (&rest args)
               (when (some (lambda (a) (or (%cx-deep-p a) (%cx-tensor-p a))) args)
                 (error "~a: 第一轮不支持复数输入（~a）。见 CONVENTIONS §16.11"
                        name ,reason))
               (apply orig args))))))

(%cx-def-unsupported (vt-floor vt-ceiling vt-truncate vt-rint)
  "复数没有分量取整的全序，numpy 同样不提供 floor/ceil/trunc")

(%cx-def-unsupported (vt-div vt-mod vt-rem)
  "floor_divide / remainder 对复数无定义（numpy 抛 TypeError）")

(%cx-def-unsupported (vt-bit-and vt-bit-ior vt-bit-xor vt-bit-not
                     vt-left-shift vt-right-shift)
  "位运算对复数无定义（numpy 抛 TypeError）")

(%cx-def-unsupported (vt-atan2)
  "atan2 只接受实数（numpy 抛 TypeError）；复数请用 vt-angle")

(%cx-def-unsupported (vt-cbrt)
  "numpy：ufunc 'cbrt' 不接受复数；复数立方根请用 (vt-pow z 1/3)。旧实现会把复数入参静默取实部后返回实数（(vt-cbrt #C(8 0)) → 2.0）")

(%cx-def-unsupported (vt-even-p vt-odd-p)
  "奇偶性对复数无定义")

(%cx-def-unsupported (vt-percentile vt-quantile)
  "分位数需要实数全序（numpy: a must be an array of real numbers）")

;; 复数线代（det/inv/solve/lu/cholesky/qr/svd/eig/pinv/lstsq/
;; matrix-rank/matrix-power）已由 src/complex-linalg.lisp 实现（第三轮）。
;; vt-cond 复用 vt-svd + vt-inv，随之自动可用，故也不再列入拒绝清单。
;; 详见 CONVENTIONS §16.19。

;; vt-einsum 已支持复数（第三轮白名单放开 complex64/complex128）。
;; 第五轮（§16.21）把三类快路径也接上复数：
;;   · 逐元素 "ij,ij->ij"    → vt-map（元素类型无关）
;;   · 全收缩 "i,i->"        → %CX-EINSUM-DOT
;;   · GEMM/BMM "...ij,...jk->...ik" → %CX-GEMM-STRIDED
;; 全部通过 *CX-EINSUM-GEMM-FN* / *CX-EINSUM-DOT-FN* 钩子注册（linalg.lisp
;; 先编译，不能直接引用复数层）。钩子为 NIL 时自动回落到通用累加状态机。

(%cx-def-unsupported (vt-sigmoid vt-relu vt-leaky-relu vt-swish vt-softplus
                     vt-gelu vt-mish vt-hard-tanh vt-hard-sigmoid)
  "神经网络激活函数定义在实数域；复数激活需先取模或实部")

;;; ==================================================================
;;; 14. 加载期自检（§10.1 L1/L2）
;;; ==================================================================

(defun %cx-self-check ()
  "复数层加载后自检：dtype 视图、提升规则、核心语义三点回归网。
   失败即中断加载（基础设施失效比功能失效危险）。"
  (flet ((is (got want what)
           (unless (equal got want)
             (error "complex self-check failed: ~a => ~a, want ~a" what got want))))
    (is (vt-dtype->lisp-type :complex128) '(complex double-float) "lisp-type c128")
    (is (vt-itemsize (make-vt nil 0 :dtype :complex64)) 8 "itemsize c64")
    (is (vt-promote-type :complex64 :int16) :complex64 "promote c64+i16")
    (is (vt-promote-type :complex64 :int32) :complex128 "promote c64+i32")
    (is (vt-promote-type :complex64 :float64) :complex128 "promote c64+f64")
    (is (vt-promote-type :complex64 :complex128) :complex128 "promote c64+c128")
    (is (vt-to-list (vt-abs (ensure-vt (list (complex 3.0d0 4.0d0)))))
        (list 5.0d0) "abs")
    (is (vt-dtype (vt-abs (ensure-vt (list (complex 3.0d0 4.0d0))))) :float64 "abs dtype")
    (is (vt-dtype (vt-sum (ensure-vt (list (complex 3.0d0 4.0d0))))) :complex128 "sum dtype")
    (is (vt-to-list (vt-< (ensure-vt (list (complex 1.0d0 2.0d0)))
                          (ensure-vt (list (complex 2.0d0 -1.0d0)))))
        (list 1) "< lex"))
  t)

(eval-when (:load-toplevel :execute)
  (%cx-self-check))
