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

(defun %cop-sqrt (z)
  "复数平方根（主值）。sqrt(-4+0i)=+2i、sqrt(-4-0i)=-2i（尊重虚部 -0.0）。"
  (let ((x (%creal z)) (y (%cimag z)))
    (cond ((%cisnan z) (%cx-nan-complex z))
          ((and (zerop y) (>= x 0)) (complex (sqrt x) y))
          (t (sqrt z)))))

(defun %cop-log (z)
  "复数自然对数 log|z| + i·arg(z)。极点 log(0) → -Inf + 0i（对标 numpy）。"
  (let ((x (%creal z)) (y (%cimag z)))
    (cond ((%cisnan z) (%cx-nan-complex z))
          ((and (zerop x) (zerop y))
           (complex (%cx-inf-of z -1) y))
          (t (log z)))))

(defun %cop-log2 (z)
  "复数 log2 = log(z)/ln2。极点 0 → -Inf + 0i。"
  (let* ((l (%cop-log z))
         (ln2 (if (typep (%creal l) 'single-float) 0.6931472 0.6931471805599453d0)))
    (complex (/ (%creal l) ln2) (/ (%cimag l) ln2))))

(defun %cop-log10 (z)
  "复数 log10 = log(z)/ln10。极点 0 → -Inf + 0i。"
  (let* ((l (%cop-log z))
         (ln10 (if (typep (%creal l) 'single-float) 2.3025851
                   2.302585092994046d0)))
    (complex (/ (%creal l) ln10) (/ (%cimag l) ln10))))

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
  "复数模（实数输出）。abs(inf+nan·i)=inf，与 numpy 的 hypot 语义一致。"
  (cond ((%cisnan z) (%cx-nan-of z))
        ((%cisinf z) (%cx-inf-of z 1))
        (t (abs z))))

(defun %cop-neg (z) (- z))

;;; ==================================================================
;;; 3. 一元函数分派包装
;;; ==================================================================

(defmacro %cx-def-unary (name &optional (kernel name))
  "为 (vt &key out dtype) 形式的一元函数生成复数分派包装。"
  `(let ((orig (%cx-orig ',name)))
     (setf (symbol-function ',name)
           (lambda (tensor &rest keys &key (out nil) (dtype nil) &allow-other-keys)
             (declare (ignore keys))
             (if (%cx-tensor-p tensor)
                 (%cx-map1 (function ,kernel) tensor out dtype ,(string name))
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
        (vt-map #'%cop-abs in :out out
                :dtype (or dtype (vt-real-dtype-of (vt-dtype in))))
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
    (vt-map #'%creal in :out out :dtype dt)))

(defun vt-imag (tensor &key out dtype)
  "取虚部（对标 numpy.imag）。实数输入 → 全 0（与 numpy 一致）。
   返回副本而非视图，原因同 vt-real。"
  (let* ((in (ensure-vt tensor))
         (in-dt (vt-dtype in))
         (dt (or dtype (if (vt-complex-dtype-p in-dt)
                           (vt-real-dtype-of in-dt)
                           in-dt))))
    (vt-map #'%cimag in :out out :dtype dt)))

(defun vt-conj (tensor &key out dtype)
  "逐元素共轭（对标 numpy.conj / numpy.conjugate）。实数输入原样返回。"
  (let ((in (ensure-vt tensor)))
    (vt-map #'%cconj in :out out :dtype (or dtype (vt-dtype in)))))

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
                      (if (eq dt :float32) (/ 180.0 0.31415927e1) (/ 180.0d0 pi))))
                 #'%carg)))
    (vt-map fn in :out out :dtype dt)))

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
          (vt-map (lambda (x y)
                    (let ((base (if (complexp x) x (complex x 0)))
                          (e (if (complexp y) y (complex y 0))))
                      (cond ((%cisnan base) (%cx-nan-complex base))
                            ;; 指数为 0（含 0+0i）→ 1+0i，先于极点判定
                            ;; （对标 numpy：0**0 = 1、nan**0 = 1）
                            ((%cis-zero e)
                             (complex (float 1 (%creal base))
                                      (* 0 (%cimag base))))
                            ((and (%cis-zero base)
                                  (plusp (%creal e))
                                  (zerop (%cimag e)))
                             (complex (* 0 (%creal base)) (* 0 (%cimag base))))
                            ((%cis-zero base) (%cx-nan-complex base))
                            (t (handler-case (expt base e)
                                 (error () (%cx-nan-complex base)))))))
                  in pw :out out :dtype dt))
        (funcall (%cx-orig 'vt-pow) vt power :out out :dtype dtype))))

(defun vt-expt (vt power &key out dtype)
  "vt-pow 的别名（CL 习惯命名）。"
  (vt-pow vt power :out out :dtype dtype))

;;; ==================================================================
;;; 6. 比较与极值（字典序，§16.8）
;;; ==================================================================

(defmacro %cx-def-compare (name kernel)
  "生成比较函数的复数分派包装（结果 :int8，1/0）。"
  `(let ((orig (%cx-orig ',name)))
     (setf (symbol-function ',name)
           (lambda (t1 t2 &key (dtype :int8) out)
             (if (or (%cx-tensor-p t1) (%cx-tensor-p t2))
                 (vt-map (function ,kernel) (ensure-vt t1) (ensure-vt t2)
                         :out out :dtype dtype)
                 (funcall orig t1 t2 :dtype dtype :out out))))))

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
  (cond ((%cisnan a) 0)
        ((%cisnan b) 0)
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

(defun %cx-nan-flag (x) (if (%cisnan x) 1 0))
(defun %cx-inf-flag (x) (if (%cisinf x) 1 0))
(defun %cx-finite-flag (x) (if (%cisfinite x) 1 0))

(defmacro %cx-def-predicate (name kernel)
  "生成 isnan/isinf/isfinite 的复数分派包装（结果 :int8，1/0）。"
  `(let ((orig (%cx-orig ',name)))
     (setf (symbol-function ',name)
           (lambda (tensor &key out)
             (if (%cx-tensor-p tensor)
                 (vt-map (function ,kernel) (ensure-vt tensor) :out out :dtype :int8)
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

(defun %cx-reduce-values (data shape axes op)
  "在连续 DATA（行主序、offset=0）上沿 AXES 归约，返回 results 向量。

   AXES 为 NIL 表示**归约全部轴**（numpy 的 axis=None 语义），不是「不归约」。
   OP ∈ :sum :prod :max :min。空归约区（n-red=0）：
     sum→0、prod→1；max/min → 报错（对标 numpy 的 zero-size 归约错误）。"
  (let* ((rank (length shape))
         (shp (coerce shape 'simple-vector))
         (strides (%cx-strides-vec shape))
         (axes* (sort (if axes (copy-list axes) (loop for d below rank collect d)) #'<))
         (non-axes (loop for d from 0 below rank unless (member d axes*) collect d))
         (out-shape (loop for d in non-axes collect (svref shp d)))
         (red-dims (loop for d in axes* collect (svref shp d)))
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
    (when (and (zerop n-red) (member op '(:max :min)) (plusp n-out))
      (error "zero-size array to reduction operation ~(~a~) which has no identity" op))
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
    (%cx-write-flat res values res-dtype)))

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
  "复数 argmax/argmin。返回 :int64 下标张量。"
  (let* ((in (ensure-vt tensor))
         (in-shape (vt-shape in))
         (rank (length in-shape))
         (flat (%cx-flat in))
         (data (vt-data flat))
         (axes (vt-normalize-axes axis rank)))
    (when (cdr axes)
      (error "~a: 不支持多轴（对标 numpy：arg 归约只接受单个整数 axis）" name))
    (cond
      ((null axes)
       ;; 展平 → 0 维下标（numpy 语义）
       (let* ((n (vt-size flat))
              (res (if out (vt-check-out out nil :int64 :op-name name)
                       (make-vt nil 0 :dtype :int64))))
         (when (zerop n)
           (error "attempt to get ~(~a~) of an empty sequence" name))
         (let ((acc (aref data 0)) (best 0))
           (dotimes (i n)
             (let ((v (aref data i)))
               (cond ((%cisnan acc) nil)
                     ((%cisnan v) (setf acc v best i))
                     ((eq op :argmax) (when (%clt acc v) (setf acc v best i)))
                     (t (when (%cgt acc v) (setf acc v best i))))))
           (%cx-write-flat res (vector best) :int64))))
      (t
       (let* ((ax (car axes))
              (ax-dim (nth ax in-shape))
              (out-shape (%cx-keepdims-shape in-shape axes keepdims))
              (res (if out (vt-check-out out out-shape :int64 :op-name name)
                       (make-vt out-shape 0 :dtype :int64)))
              (total (reduce #'* in-shape :initial-value 1))
              (vals (make-array (max 1 total) :element-type 'fixnum
                                               :initial-element 0))
              (cnt 0))
         (when (zerop ax-dim)
           (error "attempt to get ~(~a~) of an empty sequence" name))
         (multiple-value-bind (starts ax-stride dim-ax)
             (%cx-fiber-starts in-shape ax)
           (declare (ignore dim-ax))
           (loop for s across starts
                 do (let ((acc (aref data s)) (best 0))
                      (dotimes (j ax-dim)
                        (let ((v (aref data (+ s (* j ax-stride)))))
                          (cond ((%cisnan acc) nil)
                                ((%cisnan v) (setf acc v best j))
                                ((eq op :argmax) (when (%clt acc v) (setf acc v best j)))
                                (t (when (%cgt acc v) (setf acc v best j))))))
                      (setf (aref vals cnt) best)
                      (incf cnt))))
         (%cx-write-flat res vals :int64))))))

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
                                              (complex 0.0 0.0)
                                              (complex 0.0d0 0.0d0))))
              r))
          (let ((sums (%cx-reduce "VT-MEAN" in axis keepdims nil nil :sum))
                (inv (if (eq (vt-complex-real-dtype dt) :float32)
                         (/ 1.0 (float cnt 1.0))
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
                      (/ 1.0 (float den 1.0))
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
                    (if (or (%cisnan z) (%cisinf z))
                        z
                        (let ((re (realpart z)) (im (imagpart z)))
                          (complex (* (round (/ re d)) d)
                                   (* (round (/ im d)) d)))))
                  in :out out :dtype dt)))))

;;; ==================================================================
;;; 10. 扫描（cumsum / cumprod）
;;; ==================================================================

(defun %cx-scan (tensor axis op out dtype op-name)
  "沿 AXIS（nil=展平）的前缀扫描，op ∈ :sum :prod。"
  (let* ((in (ensure-vt tensor))
         (rank (length (vt-shape in)))
         (flat (%cx-flat in))
         (data (vt-data flat))
         (shape (vt-shape flat))
         (dt (or dtype (vt-dtype in)))
         (res (if out (vt-check-out out (vt-shape in) dt :op-name op-name)
                  (make-vt (vt-shape in) 0 :dtype dt)))
         (res-flat (%cx-flat res))
         (rdata (vt-data res-flat))
         (ax (if axis (vt-normalize-axis axis rank) nil)))
    (if (null ax)
        (let ((acc (case op (:sum 0) (:prod 1)))
              (n (vt-size flat)))
          (dotimes (i n)
            (setf acc (if (eq op :sum) (+ acc (aref data i)) (* acc (aref data i))))
            (setf (aref rdata i) (vt-cast acc dt))))
        (multiple-value-bind (starts ax-stride dim-ax)
            (%cx-fiber-starts shape ax)
          (loop for s across starts
                do (let ((acc (case op (:sum 0) (:prod 1))))
                     (dotimes (j dim-ax)
                       (let ((v (aref data (+ s (* j ax-stride)))))
                         (setf acc (if (eq op :sum) (+ acc v) (* acc v)))
                         (setf (aref rdata (+ s (* j ax-stride)))
                               (vt-cast acc dt))))))))
    ;; res-flat 可能是副本：非连续 out 时需要写回
    (unless (eq res-flat res)
      (vt-copy-into res res-flat))
    res))

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

(defun %cx-sort-less (a b)
  "复数排序比较：NaN 一律排末尾；其余按字典序（实部 → 虚部）。"
  (cond ((%cisnan a) nil)
        ((%cisnan b) t)
        (t (%clt a b))))

(defun %cx-sort-axis (tensor axis &key index)
  "沿 AXIS 排序（INDEX 为真时返回下标）。返回新的 C 连续张量。
   AXIS=NIL 时展平（与 clvt 既有 vt-sort :axis nil 语义一致）。"
  (let* ((in (ensure-vt tensor))
         (flat (%cx-flat in))
         (shape (vt-shape flat))
         (data (vt-data flat))
         (rank (length shape))
         (ax (if axis (vt-normalize-axis axis rank) nil)))
    (if (null ax)
        (let* ((n (vt-size flat))
               (pairs (loop for i from 0 below n collect (cons (aref data i) i)))
               (sorted (stable-sort pairs (lambda (a b) (%cx-sort-less (car a) (car b))))))
          (if index
              (let ((r (make-vt (list n) 0 :dtype :int64)))
                (dotimes (i n)
                  (setf (aref (vt-data r) i) (cdr (nth i sorted))))
                r)
              (let ((r (make-vt (list n) 0 :dtype (vt-dtype in))))
                (dotimes (i n)
                  (setf (aref (vt-data r) i) (car (nth i sorted))))
                r)))
        (let* ((res (if index (make-vt shape 0 :dtype :int64)
                        (make-vt shape 0 :dtype (vt-dtype in))))
               (rdata (vt-data res)))
          (multiple-value-bind (starts ax-stride dim-ax)
              (%cx-fiber-starts shape ax)
            (loop for s across starts
                  do (let* ((pairs (loop for j from 0 below dim-ax
                                         collect (cons (aref data (+ s (* j ax-stride))) j)))
                            (sorted (stable-sort pairs
                                                 (lambda (a b)
                                                   (%cx-sort-less (car a) (car b))))))
                       (loop for j from 0 below dim-ax
                             for p in sorted
                             do (setf (aref rdata (+ s (* j ax-stride)))
                                      (if index (cdr p) (car p)))))))
          res))))

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

(defun vt-median (tensor &key axis keepdims)
  "中位数（对标 numpy.median）。复数按字典序排序后取中位；
   偶数长度取两中位的复数均值（与 numpy 一致）。"
  (if (not (%cx-tensor-p tensor))
      (funcall (%cx-orig 'vt-median) tensor :axis axis :keepdims keepdims)
      (let* ((in (ensure-vt tensor))
             (rank (length (vt-shape in)))
             (ax (if axis (vt-normalize-axis axis rank) (1- rank)))
             (sorted (vt-sort in :axis ax))
             (n (nth ax (vt-shape sorted)))
             (lo (floor (1- n) 2))
             (hi (floor n 2))
             (out-shape (%cx-keepdims-shape (vt-shape in) (list ax) keepdims))
             (res (make-vt out-shape 0 :dtype (vt-dtype in)))
             (flat (%cx-flat sorted))
             (data (vt-data flat))
             (shape (vt-shape flat)))
        (multiple-value-bind (starts ax-stride dim-ax)
            (%cx-fiber-starts shape ax)
          (declare (ignore dim-ax))
          (let ((vals (make-array (length starts))))
            (loop for s across starts
                  for i from 0
                  do (setf (aref vals i)
                           (if (= lo hi)
                               (aref data (+ s (* lo ax-stride)))
                               (* 0.5d0 (+ (aref data (+ s (* lo ax-stride)))
                                           (aref data (+ s (* hi ax-stride))))))))
            (%cx-write-flat res vals (vt-dtype in))))
        res)))

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
  "复数 (m,k)@(k,n) → (m,n)，A/B 均为 C 连续。"
  (let* ((m (first (vt-shape a)))
         (k (second (vt-shape a)))
         (n (second (vt-shape b)))
         (ad (vt-data a)) (bd (vt-data b))
         (res (make-vt (list m n) 0 :dtype (vt-dtype a)))
         (rd (vt-data res)))
    (dotimes (i m)
      (dotimes (j n)
        (let ((acc (vt-dtype-default-value (vt-dtype a))))
          (dotimes (p k)
            (setf acc (+ acc (* (aref ad (+ (* i k) p))
                                (aref bd (+ (* p n) j))))))
          (setf (aref rd (+ (* i n) j)) acc))))
    res))

(defun %cx-dot-1d (a b)
  "复数向量点积 → 0 维张量（不取共轭，对标 numpy.dot）。"
  (let* ((n (vt-size a))
         (ad (vt-data a)) (bd (vt-data b))
         (acc (vt-dtype-default-value (vt-dtype a))))
    (dotimes (i n)
      (setf acc (+ acc (* (aref ad i) (aref bd i)))))
    (make-vt nil acc :dtype (vt-dtype a))))

(defun %cx-batched-matmul (a b)
  "批量矩阵乘 (...,m,k)@(...,k,n) → (...,m,n)：逐 batch 调 2D 内核。"
  (let* ((sa (vt-shape a)) (sb (vt-shape b))
         (ba (subseq sa 0 (- (length sa) 2)))
         (bb (subseq sb 0 (- (length sb) 2)))
         (batch (if (> (length ba) (length bb)) ba bb))
         (m (nth (- (length sa) 2) sa))
         (k (nth (1- (length sa)) sa))
         (n (nth (1- (length sb)) sb))
         (size (reduce #'* batch :initial-value 1))
         (res (make-vt (append batch (list m n)) 0 :dtype (vt-dtype a)))
         (rd (vt-data res)))
    (dotimes (i size)
      (let* ((a-slice (vt-contiguous
                       (vt-reshape (vt-slice (vt-ravel a) (list (* i m k) (* (1+ i) m k)))
                                   (list m k))))
             (b-slice (vt-contiguous
                       (vt-reshape (vt-slice (vt-ravel b) (list (* i k n) (* (1+ i) k n)))
                                   (list k n))))
             (part (%cx-matmul-2d a-slice b-slice))
             (pd (vt-data part)))
        (replace rd pd :start1 (* i m n) :end1 (* (1+ i) m n))))
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
  "外积（对标 numpy.outer）。复数：全部按分量乘法，dtype 保持复数。"
  (if (or (%cx-deep-p a) (%cx-deep-p b))
      (let* ((av (ensure-vt a)) (bv (ensure-vt b))
             (af (if flatten (vt-ravel av) av))
             (bf (if flatten (vt-ravel bv) bv))
             (dt (or dtype (vt-promote-type (vt-dtype av) (vt-dtype bv))))
             (r (vt-reshape (vt-* (vt-reshape af (append (vt-shape af) (list 1)))
                                  (vt-reshape bf (cons 1 (vt-shape bf))))
                            :dtype dt)))
        (if out (progn (vt-copy-into out r) out) r))
      (funcall (%cx-orig 'vt-outer) a b :flatten flatten :dtype dtype :out out)))

(defun vt-trace (matrix &key dtype out)
  "矩阵迹（对角线之和，对标 numpy.trace）。复数保持复数 dtype。"
  (if (%cx-deep-p matrix)
      (let* ((m (ensure-vt matrix))
             (dt (or dtype (vt-dtype m)))
             (d (vt-diagonal m))
             (r (vt-sum d :dtype dt)))
        (if out (progn (vt-copy-into out r) out) r))
      (funcall (%cx-orig 'vt-trace) matrix :dtype dtype :out out)))

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

(%cx-def-unsupported (vt-even-p vt-odd-p)
  "奇偶性对复数无定义")

(%cx-def-unsupported (vt-percentile vt-quantile)
  "分位数需要实数全序（numpy: a must be an array of real numbers）")

(%cx-def-unsupported (vt-det vt-inv vt-solve vt-svd vt-eig vt-pinv
                     vt-cholesky vt-qr vt-lu vt-cond)
  "复矩阵分解（LU/SVD/特征值）需共轭转置与酉变换，第二轮实现")

(%cx-def-unsupported (vt-einsum)
  "einsum 的 dtype 白名单只含 4 种实数类型，复数结果缓冲区会错配；
   第一轮请用 vt-matmul / vt-dot / vt-outer（已提供复数专用内核），
   einsum 复数路径第二轮实现")

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
