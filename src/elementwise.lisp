;;;; elementwise.lisp — 逐元素算术 / 数学 / 比较 / 逻辑 / 位运算

(in-package :clvt)

(defun vt-+ (&rest args)
  "逐元素加法（一元恒等 / 二元 / N 元，支持广播）。

   参数契约：VARGS = 任意个张量/标量 + 可选 :dtype / :out。
     :dtype  结果 dtype；缺省按 numpy 类型提升规则由全部输入决定。
     :out    输出张量。**严格相等契约**：dtype 必须精确等于结果 dtype、
             形状必须精确等于广播结果形状、必须可写。
   输入个数 ≤ 3 走编译期内联快路径，> 3 回退 vt-map（语义基准）。"
  (with-float-safe
    (multiple-value-bind (tensors dtype out) (parse-vt-op-args args)
      (case (length tensors)
        (1 (vt-fast-map #'+ (first tensors)
                            :dtype dtype :out out))
        (2 (vt-fast-map #'+ (first tensors)
                            (second tensors)
                            :dtype dtype :out out))
        (3 (vt-fast-map #'+ (first tensors)
                            (second tensors)
                            (third tensors)
                            :dtype dtype :out out))
        (t (apply #'vt-map #'+ args))))))

(defun vt-* (&rest args)
  "逐元素乘法（一元恒等 / 二元 / N 元，支持广播）。

   参数契约同 vt-+：:dtype 决定结果 dtype（缺省按提升规则），
   :out 必须精确匹配结果形状与 dtype 且可写。"
  (with-float-safe
    (multiple-value-bind (tensors dtype out) (parse-vt-op-args args)
      (case (length tensors)
        (1 (vt-fast-map #'* (first tensors)
                            :dtype dtype :out out))
        (2 (vt-fast-map #'* (first tensors)
                            (second tensors)
                            :dtype dtype :out out))
        (3 (vt-fast-map #'* (first tensors)
                            (second tensors)
                            (third tensors)
                            :dtype dtype :out out))
        (t (apply #'vt-map #'* args))))))

(defun vt-- (vt &rest args)
  "逐元素减法（一元取负 / 二元 / N 元左结合）。
   一元：-vt；二元及以上：vt - arg1 - arg2 - ...
   VT 是第一个张量，其余从 ARGS 解析。

   参数契约同 vt-+：:dtype 决定结果 dtype（缺省按提升规则），
   :out 必须精确匹配结果形状与 dtype 且可写。"
  (with-float-safe
    (multiple-value-bind (tensors dtype out) (parse-vt-op-args args)
      (let ((all (cons (ensure-vt vt) tensors)))
        (case (length all)
          (1 (vt-fast-map #'- (first all)
                              :dtype dtype :out out))
          (2 (vt-fast-map #'- (first all) (second all)
                              :dtype dtype :out out))
          (3 (vt-fast-map #'- (first all) (second all) (third all)
                              :dtype dtype :out out))
          (t (apply #'vt-map #'- all)))))))

(defun vt-/ (vt &rest args)
  "一元：1 / vt（倒数）；二元及以上：vt / arg1 / arg2 / ...
   VT 是第一个张量，其余从 ARGS 解析。
   语义对标 numpy.true_divide（v0.3.6 起）：
   除法恒以浮点语义计算——整型输入先提升为浮点再除，
   零除按 IEEE 754 返回 ±Inf/NaN（CL 的整数除零错误不再出现）；
   显式整型 :dtype 除外（保持截断语义，零除确定性报错）。

   参数契约（§4）：
     :dtype  结果 dtype。缺省时：整型输入 → 提升到 float64/float32
             （true_divide 语义）；显式给出则完全以它为准。
     :out    输出张量。**严格相等契约**：其 dtype 必须精确等于
             上面算出的结果 dtype，否则报错；形状必须精确等于
             广播结果形状。不可写（stride-0 广播视图）时报错。
             非连续 out 由统一原语按真实 strides 写入。"
  (with-float-safe
    (multiple-value-bind (tensors dtype out) (parse-vt-op-args args)
      (let* ((all (mapcar #'ensure-vt (cons vt tensors)))
             ;; true_divide 的「除法恒为浮点」规则：仅当用户**未显式指定**
             ;; :dtype 时才生效。这里只算出 promote 会得出的结果 dtype，
             ;; 绝不读 out 的 dtype 来决定结果（§4.2 H3 严格相等契约：
             ;; 结果 dtype 由输入提升 + 显式 :dtype 决定，不由 out 决定）。
             (float-target
               (cond (dtype
                      ;; 显式整型目标：保持截断语义，零除确定性报错
                      nil)
                     ((and (null out)
                           (every (lambda (x) (vt-int-dtype-p (vt-dtype x))) all))
                      :float64)
                     ((and (null out)
                           (some (lambda (x) (vt-int-dtype-p (vt-dtype x))) all))
                      (if (every (lambda (x)
                                   (or (eq (vt-dtype x) :float32)
                                       (vt-int-dtype-p (vt-dtype x))))
                                 all)
                          :float32
                          :float64))
                     ((and (null out)
                           (some (lambda (x) (eq (vt-dtype x) :float32)) all)
                           (every (lambda (x)
                                    (member (vt-dtype x) '(:float32 :float64)))
                                  all))
                      :float32)))
             (effective-dtype (or dtype float-target)))
        ;; 计算前把整型输入预转换为浮点（numpy true_divide 语义）：
        ;; 避免 CL 整数除零信号 division-by-zero（numpy 为 ±Inf），
        ;; 也避免 (/ int 0) 混合类型除法在浮点目标下的类型错误。
        ;; 顺带修复：标量除数（如 (vt-/ t 0)）先经 ensure-vt 转为 0 维张量，
        ;; 不再因 vt-dtype 作用于数字而直接类型崩溃。
        (when (and float-target
                   (some (lambda (x) (vt-int-dtype-p (vt-dtype x))) all))
          (setf all (mapcar (lambda (x)
                              (if (vt-int-dtype-p (vt-dtype x))
                                  (vt-astype x float-target)
                                  x))
                            all)))
        (case (length all)
          (1 (vt-fast-map #'/ (make-vt nil 1 :dtype (vt-dtype (first all)))
                              (first all)
                              :dtype effective-dtype :out out))
          (2 (vt-fast-map #'/ (first all) (second all)
                              :dtype effective-dtype :out out))
          (3 (vt-fast-map #'/ (first all) (second all) (third all)
                              :dtype effective-dtype :out out))
          ;; > 3 个输入：走语义基准路径。注意 vt-map 不接受 nil dtype，
          ;; 无显式 dtype 时让 vt-map 自行按提升规则推断。
          (t (if effective-dtype
                 (apply #'vt-map #'/ :dtype effective-dtype :out out all)
                 (apply #'vt-map #'/ :out out all))))))))

(defun vt-add (a b &key dtype out)
  "逐元素加法（二元特化入口），等价于 (vt-+ a b)。
      :dtype 决定结果 dtype（缺省按提升规则），:out 契约同 vt-+。"
  (vt-fast-map #'+ a b :dtype dtype :out out))

(defun vt-sub (a b &key dtype out)
  "逐元素减法（二元特化入口），等价于 (vt-- a b)。
      :dtype 决定结果 dtype（缺省按提升规则），:out 契约同 vt-+。"
  (vt-fast-map #'- a b :dtype dtype :out out))

(defun vt-mul (a b &key dtype out)
  "逐元素乘法（二元特化入口），等价于 (vt-* a b)。
      :dtype 决定结果 dtype（缺省按提升规则），:out 契约同 vt-+。"
  (vt-fast-map #'* a b :dtype dtype :out out))

(defun vt-div (a b &key dtype out)
  "逐元素除法（二元特化入口），等价于 (vt-/ a b)。
      注意：此入口不套用 vite_divide 的整型→浮点提升，
      结果为 CL 原生 / 语义；需要 numpy true_divide 语义请用 vt-/。"
  (vt-fast-map #'/ a b :dtype dtype :out out))

(defun vt-scale (a b &key out dtype)
  "按标量缩放（等价 vt-* a b），NaN/Inf 按 IEEE 754 正常传播。
      :dtype 决定结果 dtype，:out 契约同 vt-+。"
  (vt-fast-map #'* a b :out out :dtype dtype))

(defun %infer-float-dtype (vt dtype)
  (or dtype (if (eq (vt-dtype vt) :float32) :float32 :float64)))

(defun vt-sin (vt &key out dtype)
  "逐元素正弦。整数输入输出按 :dtype 或 float64（对标 numpy，
      整数进 float64 出）；float32 进 float32 出。:out 契约同 vt-+。"
  (vt-fast-map #'sin vt :out out :dtype (%infer-float-dtype vt dtype)))

(defun vt-cos (vt &key out dtype)
  "逐元素余弦。整数输入输出 float64，float32 进 float32 出。
      :out 契约同 vt-+。"
  (vt-fast-map #'cos vt :out out :dtype (%infer-float-dtype vt dtype)))

(defun vt-tan (vt &key out dtype)
  "逐元素正切。整数输入输出 float64，float32 进 float32 出。
      :out 契约同 vt-+。"
  (vt-fast-map #'tan vt :out out :dtype (%infer-float-dtype vt dtype)))

(defun vt-atan (vt &key out dtype)
  "逐元素反正切。整数输入输出 float64，float32 进 float32 出。
      :out 契约同 vt-+。"
  (vt-fast-map #'atan vt :out out :dtype (%infer-float-dtype vt dtype)))

(defun vt-sinh (vt &key out dtype)
  "逐元素双曲正弦。整数输入输出 float64，float32 进 float32 出。
      :out 契约同 vt-+。"
  (vt-fast-map #'sinh vt :out out :dtype (%infer-float-dtype vt dtype)))

(defun vt-cosh (vt &key out dtype)
  "逐元素双曲余弦。整数输入输出 float64，float32 进 float32 出。
      :out 契约同 vt-+。"
  (vt-fast-map #'cosh vt :out out :dtype (%infer-float-dtype vt dtype)))

(defun vt-tanh (vt &key out dtype)
  "逐元素双曲正切。整数输入输出 float64，float32 进 float32 出。
      :out 契约同 vt-+。"
  (vt-fast-map #'tanh vt :out out :dtype (%infer-float-dtype vt dtype)))

(defun vt-asin (vt &key out dtype)
  "逐元素反正弦。|x|>1 返回 NaN（对标 numpy，不抛条件）。
      整数输入输出 float64，float32 进 float32 出。:out 契约同 vt-+。"
  (let* ((dt (%infer-float-dtype vt dtype))
	 (nan (vt-get-nan dt)))
    (vt-map (lambda (x)
	      (if (> (abs x) 1.0d0) nan (asin x)))
	    vt :out out :dtype dt)))

(defun vt-acos (vt &key out dtype)
  "逐元素反余弦。|x|>1 返回 NaN（对标 numpy，不抛条件）。
      整数输入输出 float64，float32 进 float32 出。:out 契约同 vt-+。"
  (let* ((dt (%infer-float-dtype vt dtype))
	 (nan (vt-get-nan dt)))
    (vt-map (lambda (x)
	      (if (> (abs x) 1.0d0) nan (acos x)))
	    vt :out out :dtype dt)))

(defun vt-asinh (vt &key out dtype)
  "逐元素反双曲正弦。整数输入输出 float64，float32 进 float32 出。
      :out 契约同 vt-+。"
  (vt-fast-map #'asinh vt :out out :dtype (%infer-float-dtype vt dtype)))

(defun vt-acosh (vt &key out dtype)
  "逐元素反双曲余弦。x<1 返回 NaN（对标 numpy）。
      整数输入输出 float64，float32 进 float32 出。:out 契约同 vt-+。"
  (let* ((dt (%infer-float-dtype vt dtype))
	 (nan (vt-get-nan dt)))
    (vt-map (lambda (x)
	      (if (< x 1.0d0) nan (acosh x)))
	    vt :out out :dtype dt)))

(defun vt-atanh (vt &key out dtype)
  "逐元素反双曲正切。|x|>=1 返回 NaN（对标 numpy）。
      整数输入输出 float64，float32 进 float32 出。:out 契约同 vt-+。"
  (let* ((dt (%infer-float-dtype vt dtype))
	 (nan (vt-get-nan dt)))
    (vt-map (lambda (x)
	      (if (>= (abs x) 1.0d0) nan (atanh x)))
	    vt :out out :dtype dt)))

(defun vt-exp (vt &key out dtype)
  "逐元素指数。整数输入输出 float64，float32 进 float32 出。
      :out 契约同 vt-+。"
  (vt-fast-map #'exp vt :out out :dtype (%infer-float-dtype vt dtype)))

(defun vt-pow (vt power &key out dtype)
  "逐元素幂 VT**POWER。POWER 为标量指数。
      整数 + 正指数走精确整数幂；其余（负指数/非整数/浮点）
      走浮点幂并对非实数结果返回 NaN（对标 numpy.power）。
      :out 契约同 vt-+。"
  (let* ((dt (%infer-float-dtype vt dtype))
         (nan (vt-get-nan dt)))
    (cond
      ((and (integerp power) (plusp power))
       (vt-map (lambda (x) (expt x power)) vt :out out :dtype dt))
      (t
       (vt-map (lambda (x)
                 (let ((result (handler-case (expt x power)
                                 (error () nan))))
                   (if (realp result) result nan)))
               vt :out out :dtype dt)))))

(defun vt-expt (vt power &key out dtype)
  "vt-pow 的别名（CL 习惯命名）。语义与参数契约见 vt-pow。"
  (vt-pow vt power :out out :dtype dtype))

(defun vt-square (vt &key out dtype)
  "逐元素平方。特化为 vt-fast-map #'* 以避免 vt-pow 的 handler-case/expt/realp 开销。"
  (vt-fast-map #'* vt vt :out out :dtype dtype))
  "逐元素平方。特化为 vt-fast-map #'* 以避免 vt-pow 的
      handler-case/expt/realp 开销。:out 契约同 vt-+。"

(defun vt-sqrt (vt &key out dtype)
  "逐元素平方根。负数输入返回 NaN（对标 numpy，不抛条件）。
      整数输入输出 float64，float32 进 float32 出。:out 契约同 vt-+。"
  (let* ((dt (%infer-float-dtype vt dtype))
	 (nan (vt-get-nan dt)))
    (vt-map (lambda (x) (if (minusp x) nan (sqrt x)))
	    vt :out out :dtype dt)))

(defun vt-log (vt &key base out dtype)
  "逐元素自然对数；:base 给定时为换底对数。
      语义对标 numpy.log：x>0 正常；x=0 → -Inf（或 log(base)<0 时 +Inf）；
      x<0 → NaN；非法 base（<=0 或 =1）→ 全 NaN。
      整数输入输出 float64，float32 进 float32 出。:out 契约同 vt-+。"
  (let* ((dt (%infer-float-dtype vt dtype))
         (nan (vt-get-nan dt))
	 (neginf (vt-get-neg-inf dt))
	 (posinf (vt-get-pos-inf dt)))
    (cond
      ((and base (or (<= base 0) (= base 1)))
       (vt-map (lambda (x)
		 (declare (ignore x)) nan)
	       vt :out out :dtype dt))
      ((null base)
       (vt-map (lambda (x)
		 (if (> x 0)
		     (log x)
		     (if (zerop x)
			 neginf nan)))
	       vt :out out :dtype dt))
      (t
       (let ((zero-result (if (plusp (log base)) neginf posinf)))
         (vt-map (lambda (x)
		   (if (> x 0)
		       (log x base)
		       (if (zerop x)
			   zero-result nan)))
                 vt :out out :dtype dt))))))

(defun vt-log10 (vt &key out dtype)
  "逐元素常用对数（底 10）。语义等价 (vt-log vt :base 10)。
      :out 契约同 vt-+。"
  (vt-log vt :base 10.0d0 :out out :dtype dtype))

(defun vt-log2 (vt &key out dtype)
  "逐元素二进制对数（底 2）。语义等价 (vt-log vt :base 2)。
      :out 契约同 vt-+。"
  (vt-log vt :base 2.0d0 :out out :dtype dtype))

(defun vt-abs (vt &key out dtype)
  "逐元素绝对值。整数保持整数 dtype（对标 numpy.abs）。
      :out 契约同 vt-+。"
  (vt-fast-map #'abs vt :out out :dtype dtype))

(defun vt-signum (vt &key out dtype)
  "逐元素符号函数（-1 / 0 / +1）。整数保持整数 dtype，
      NaN 原样传播。:out 契约同 vt-+。"
  (vt-map (lambda (x) (if (%nan-p x) x (signum x)))
          vt :out out :dtype dtype))

(defun vt-positive-p (vt &key out (dtype :float64))
  "逐元素正数判定，返回 1.0/0.0（dtype 默认 float64）。
      对标 numpy.positive 的**符号过滤**语义（不是 numpy 的
      elementwise_ispos 类型判定）。NaN 比较恒假 → 0.0。"
  (vt-map (lambda (v)
	    (if (> v 0.0d0) 1.0d0 0.0d0))
	  vt :out out :dtype dtype))

(defun vt-negative-p (vt &key out (dtype :float64))
  "逐元素负数判定，返回 1.0/0.0（dtype 默认 float64）。
      NaN 比较恒假 → 0.0。"
  (vt-map (lambda (v)
	    (if (< v 0.0d0) 1.0d0 0.0d0))
	  vt :out out :dtype dtype))

(defun vt-zero-p (vt &key out (dtype :float64))
  "逐元素零判定，返回 1.0/0.0（dtype 默认 float64）。"
  (vt-map (lambda (v)
	    (if (zerop v) 1.0d0 0.0d0))
	  vt :out out :dtype dtype))

(defun vt-nonzero-p (vt &key out (dtype :float64))
  "逐元素非零判定，返回 1.0/0.0（dtype 默认 float64）。"
  (vt-map (lambda (v)
	    (if (zerop v) 0.0d0 1.0d0))
	  vt :out out :dtype dtype))

(defun vt-even-p (vt &key out (dtype :float64))
  "逐元素偶数判定，返回 1.0/0.0（dtype 默认 float64）。
      NaN/±Inf 视为非偶数 → 0.0（对标 numpy）。"
  (vt-map (lambda (v)
	    (if (or (%nan-p v) (%inf-p v)) 0.0d0 (if (evenp (floor v)) 1.0d0 0.0d0)))
	  vt :out out :dtype dtype))

(defun vt-odd-p (vt &key out (dtype :float64))
  "逐元素奇数判定，返回 1.0/0.0（dtype 默认 float64）。
      NaN/±Inf 视为非奇数 → 0.0（对标 numpy）。"
  (vt-map (lambda (v)
	    (if (or (%nan-p v) (%inf-p v)) 0.0d0 (if (oddp (floor v)) 1.0d0 0.0d0)))
	  vt :out out :dtype dtype))

(defun %mod-nan-or-inf-p (x)
  "取模语境下的非有限值判定（需已屏蔽浮点陷阱）。"
  (and (floatp x) (%nan-or-inf-p x)))

(declaim (inline %mod-float-nan))
(defun %mod-float-nan (x)
  "取模遇 NaN 时的返回值。写入按输出 dtype 转换，恒返 double NaN 即可。"
  (declare (ignore x))
  +vt-dfloat-nan+)

(defun vt-mod (vt divisor &key out dtype)
  "逐元素取模。DIVISOR 可为标量或张量（自动广播）。
   语义对标 numpy.remainder：结果与除数同号；
   除数为 0 时按 numpy dtype 语义（v0.3.6 起）：
   浮点语境（任一操作数为浮点）→ NaN；整型语境 → 0；
   NaN 任一侧出现 → NaN；被除数为 ±Inf → NaN；除数为 ±Inf → 被除数本身。"
  (flet ((zero-div (x)
           ;; numpy 语义：浮点语境零除 → NaN，整型语境 → 0
           (if (floatp x) (%mod-float-nan x) 0)))
    (if (numberp divisor)
  (vt-map (lambda (x)
                (cond ((%mod-nan-or-inf-p x) (%mod-float-nan x))
                      ((%mod-nan-or-inf-p divisor) (%mod-float-nan x))
                      ((and (numberp divisor) (zerop divisor)) (zero-div x))
                      ((and (floatp divisor) (%inf-p divisor)) x)
                      (t (mod x divisor))))
              vt :out out :dtype dtype)
      (vt-map (lambda (x y)
                (cond ((%mod-nan-or-inf-p x) (%mod-float-nan x))
                      ((%mod-nan-or-inf-p y) (%mod-float-nan x))
                      ((zerop y) (zero-div x))
                      ((and (floatp y) (%inf-p y)) x)
                      (t (mod x y))))
              vt divisor :out out :dtype dtype))))

(defun vt-rem (vt divisor &key out dtype)
  "逐元素余数。DIVISOR 可为标量或张量（自动广播）。
   CL:rem 为截断除法余数（与被除数同号），对标 numpy.fmod
   （docstring 曾误标 numpy.remainder，现已更正）。
   除数为 0 时按 numpy dtype 语义（v0.3.6 起）：
   浮点语境（任一操作数为浮点）→ NaN；整型语境 → 0；
   NaN 任一侧 → NaN；被除数为 ±Inf → NaN；除数为 ±Inf → 被除数本身。"
  (flet ((zero-div (x)
           ;; numpy 语义：浮点语境零除 → NaN，整型语境 → 0
           (if (floatp x) (%mod-float-nan x) 0)))
    (if (numberp divisor)
  (vt-map (lambda (x)
                (cond ((%mod-nan-or-inf-p x) (%mod-float-nan x))
                      ((%mod-nan-or-inf-p divisor) (%mod-float-nan x))
                      ((zerop divisor) (zero-div x))
                      ((and (floatp divisor) (%inf-p divisor)) x)
                      (t (rem x divisor))))
              vt :out out :dtype dtype)
      (vt-map (lambda (x y)
                (cond ((%mod-nan-or-inf-p x) (%mod-float-nan x))
                      ((%mod-nan-or-inf-p y) (%mod-float-nan x))
                      ((zerop y) (zero-div x))
                      ((and (floatp y) (%inf-p y)) x)
                      (t (rem x y))))
              vt divisor :out out :dtype dtype))))

(defun vt-atan2 (vty vtx &key out dtype)
  "逐元素双参数反正切 atan2(y, x)。参数顺序 VTY, VTX。
      整数输入输出 float64，float32 进 float32 出。:out 契约同 vt-+。"
  (vt-fast-map #'atan vty vtx :out out :dtype dtype))

;; floor 族对 NaN/±Inf 统一传播本身（对标 numpy：floor/ceil/trunc/round(nan)=nan，
;; floor(±inf)=±inf）。SBCL 对非有限值直接调用 floor/round 会 signal
;; FLOATING-POINT-INVALID-OPERATION（trap 屏蔽也无法避免），必须显式拦截。
(defmacro %floor-family-body (x op)
  `(if (and (floatp ,x) (%nan-or-inf-p ,x))
       ,x
       (let ((res (nth-value 0 (,op ,x divisor))))
         (if (floatp ,x) (float res ,x) res))))

(defun vt-floor (vt &key (divisor 1) out dtype)
  "逐元素向下取整（可除 DIVISOR）。
      NaN/±Inf 原样传播（对标 numpy.floor）；整数输入保持整数 dtype。
      :out 契约同 vt-+。"
  (vt-map (lambda (x) (%floor-family-body x floor))
	  vt :out out :dtype dtype))

(defun vt-ceiling (vt &key (divisor 1) out dtype)
  "逐元素向上取整（可除 DIVISOR）。
      NaN/±Inf 原样传播（对标 numpy.ceil）；整数输入保持整数 dtype。
      :out 契约同 vt-+。"
  (vt-map (lambda (x) (%floor-family-body x ceiling))
	  vt :out out :dtype dtype))

(defun vt-round (vt &key (divisor 1) out dtype)
  "逐元素四舍五入取整（可除 DIVISOR）。
      NaN/±Inf 原样传播（对标 numpy.round）；整数输入保持整数 dtype。
      注意：舍入策略为「最近偶数」，与 CL round 一致。"
  (vt-map (lambda (x) (%floor-family-body x round))
	  vt :out out :dtype dtype))

(defun vt-truncate (vt &key (divisor 1) out dtype)
  "逐元素向零取整（可除 DIVISOR）。
      NaN/±Inf 原样传播（对标 numpy.trunc）；整数输入保持整数 dtype。
      :out 契约同 vt-+。"
  (vt-map (lambda (x) (%floor-family-body x truncate))
	  vt :out out :dtype dtype))

(defun vt-rint (vt &key out dtype)
  "逐元素取整到最近偶数（对标 numpy.rint）。
      NaN/±Inf 原样传播；整数输入保持整数 dtype。:out 契约同 vt-+。"
  (vt-map (lambda (x)
            (if (and (floatp x) (%nan-or-inf-p x))
                x
	    (let ((res (nth-value 0 (round x))))
                  (if (floatp x) (float res x) res))))
	  vt :out out :dtype dtype))

(declaim (inline %op-eq %op-ne %op-lt %op-le %op-gt %op-ge))
(defun %op-eq (a b) (if (=  a b) 1.0d0 0.0d0))

(defun %op-ne (a b) (if (/= a b) 1.0d0 0.0d0))

(defun %op-lt (a b) (if (<  a b) 1.0d0 0.0d0))

(defun %op-le (a b) (if (<= a b) 1.0d0 0.0d0))

(defun %op-gt (a b) (if (>  a b) 1.0d0 0.0d0))

(defun %op-ge (a b) (if (>= a b) 1.0d0 0.0d0))

(defun vt-= (t1 t2 &key (dtype :float64) out)
  "逐元素相等判定，返回 1.0/0.0（dtype 默认 float64）。
      NaN 与任何值比较恒假 → 0.0（含 NaN == NaN）。"
  (vt-fast-map #'%op-eq (ensure-vt t1) (ensure-vt t2)
               :dtype dtype :out out))

(defun vt-/= (t1 t2 &key (dtype :float64) out)
  "逐元素不等判定，返回 1.0/0.0（dtype 默认 float64）。
      NaN 与任何值比较恒真 → 1.0（含 NaN != NaN）。"
  (vt-fast-map #'%op-ne (ensure-vt t1) (ensure-vt t2)
               :dtype dtype :out out))

(defun vt-< (t1 t2 &key (dtype :float64) out)
  "逐元素小于判定，返回 1.0/0.0（dtype 默认 float64）。
      NaN 参与的比较恒假 → 0.0。"
  (vt-fast-map #'%op-lt (ensure-vt t1) (ensure-vt t2)
               :dtype dtype :out out))

(defun vt-<= (t1 t2 &key (dtype :float64) out)
  "逐元素小于等于判定，返回 1.0/0.0（dtype 默认 float64）。
      NaN 参与的比较恒假 → 0.0。"
  (vt-fast-map #'%op-le (ensure-vt t1) (ensure-vt t2)
               :dtype dtype :out out))

(defun vt-> (t1 t2 &key (dtype :float64) out)
  "逐元素大于判定，返回 1.0/0.0（dtype 默认 float64）。
      NaN 参与的比较恒假 → 0.0。"
  (vt-fast-map #'%op-gt (ensure-vt t1) (ensure-vt t2)
               :dtype dtype :out out))

(defun vt->= (t1 t2 &key (dtype :float64) out)
  "逐元素大于等于判定，返回 1.0/0.0（dtype 默认 float64）。
      NaN 参与的比较恒假 → 0.0。"
  (vt-fast-map #'%op-ge (ensure-vt t1) (ensure-vt t2)
               :dtype dtype :out out))

(defun vt-rad2deg (vt &key out dtype)
  "弧度转角度（×180/π）。整数输入输出 float64，
      float32 进 float32 出。:out 契约同 vt-+。"
  (let* ((dt (%infer-float-dtype vt dtype))
         (factor (if (eq dt :float32) (/ 180.0s0 (coerce pi 'single-float))
                     (/ 180.0d0 pi))))
    (vt-map (lambda (x) (* x factor)) vt :out out :dtype dt)))

(defun vt-deg2rad (vt &key out dtype)
  "角度转弧度（×π/180）。整数输入输出 float64，
      float32 进 float32 出。:out 契约同 vt-+。"
  (let* ((dt (%infer-float-dtype vt dtype))
         (factor (if (eq dt :float32) (/ (coerce pi 'single-float) 180.0s0)
                     (/ pi 180.0d0))))
    (vt-map (lambda (x) (* x factor)) vt :out out :dtype dt)))

(defun vt-maximum (t1 t2 &key out dtype)
  "逐元素取较大值，**NaN 传播**（任一为 NaN → NaN，且优先取左）
      对标 numpy.maximum。:out 契约同 vt-+。"
  (vt-map (lambda (a b)
	    (cond ((%nan-p a) a)
		  ((%nan-p b) b)
		  (t (max a b))))	  
          t1 t2 :out out :dtype dtype))
(defun vt-minimum (t1 t2 &key out dtype)
  "逐元素取较小值，**NaN 传播**（任一为 NaN → NaN，且优先取左）
      对标 numpy.minimum。:out 契约同 vt-+。"
  (vt-map (lambda (a b)
	    (cond ((%nan-p a) a)
		  ((%nan-p b) b)
		  (t (min a b))))
          t1 t2 :out out :dtype dtype))

(defun vt-fmax (t1 t2 &key out dtype)
  "逐元素取较大值，**忽略 NaN**（仅两侧皆 NaN 时返回 NaN）
      对标 numpy.fmax。:out 契约同 vt-+。"
  (vt-map (lambda (a b)
	    (cond ((and (%nan-p a) (%nan-p b)) a)
                  ((%nan-p a) b) ((%nan-p b) a)
		  (t (max a b))))
          (ensure-vt t1) (ensure-vt t2) :dtype dtype :out out))

(defun vt-fmin (t1 t2 &key out dtype)
  "逐元素取较小值，**忽略 NaN**（仅两侧皆 NaN 时返回 NaN）
      对标 numpy.fmin。:out 契约同 vt-+。"
  (vt-map (lambda (a b)
	    (cond ((and (%nan-p a) (%nan-p b)) a)
                  ((%nan-p a) b) ((%nan-p b) a)
		  (t (min a b))))
          (ensure-vt t1) (ensure-vt t2) :dtype dtype :out out))

(defun vt-logical-and (t1 t2 &key out (dtype :float64))
  "逐元素逻辑与（非零视为真），返回 1.0/0.0（dtype 默认 float64）。"
  (vt-map (lambda (a b)
	    (if (and (not (zerop a)) (not (zerop b)))
		1.0d0 0.0d0))
          (ensure-vt t1) (ensure-vt t2) :dtype dtype :out out))

(defun vt-logical-or (t1 t2 &key out (dtype :float64))
  "逐元素逻辑或（非零视为真），返回 1.0/0.0（dtype 默认 float64）。"
  (vt-map (lambda (a b)
	    (if (or (not (zerop a)) (not (zerop b)))
		1.0d0 0.0d0))
          (ensure-vt t1) (ensure-vt t2) :dtype dtype :out out))

(defun vt-logical-not (vt &key out (dtype :float64))
  "逐元素逻辑非，返回 1.0/0.0（dtype 默认 float64）。"
  (vt-map (lambda (v)
	    (if (zerop v) 1.0d0 0.0d0))
	  vt :dtype dtype :out out))

(defun vt-logical-xor (t1 t2 &key out (dtype :float64))
  "逐元素逻辑异或，返回 1.0/0.0（dtype 默认 float64）。"
  (vt-map (lambda (a b)
	    (if (not (eq (not (zerop a)) (not (zerop b))))
		1.0d0 0.0d0))
          (ensure-vt t1) (ensure-vt t2) :dtype dtype :out out))

(defun vt-bit-and (t1 t2 &key out dtype)
  "逐元素按位与 logand。输入应为整数 dtype；:out 契约同 vt-+。"
  (vt-map #'logand (ensure-vt t1) (ensure-vt t2) :dtype dtype :out out))

(defun vt-bit-ior (t1 t2 &key out dtype)
  "逐元素按位或 logior。输入应为整数 dtype；:out 契约同 vt-+。"
  (vt-map #'logior (ensure-vt t1) (ensure-vt t2) :dtype dtype :out out))

(defun vt-bit-xor (t1 t2 &key out dtype)
  "逐元素按位异或 logxor。输入应为整数 dtype；:out 契约同 vt-+。"
  (vt-map #'logxor (ensure-vt t1) (ensure-vt t2) :dtype dtype :out out))

(defun vt-bit-not (vt &key out dtype)
  "逐元素按位取反 lognot。输入应为整数 dtype；:out 契约同 vt-+。"
  (vt-map #'lognot vt :dtype dtype :out out))

(defun vt-left-shift (vt shift &key out dtype)
  "逐元素左移 SHIFT 位（ash x shift）。SHIFT 为整数标量。
      输入应为整数 dtype；:out 契约同 vt-+。"
  (vt-map (lambda (x) (ash x shift))
	  vt :dtype dtype :out out))

(defun vt-right-shift (vt shift &key out dtype)
  "逐元素右移 SHIFT 位（ash x -shift，算术右移）。SHIFT 为整数标量。
      输入应为整数 dtype；:out 契约同 vt-+。"
  (vt-map (lambda (x) (ash x (- shift)))
	  vt :dtype dtype :out out))

(declaim (inline %op-clip))
(defun %op-clip (x minv maxv)
  "三元 clip：x 先和 maxv 比较，再和 minv 比较。
   与 (min max-val (max min-val x)) 语义一致。"
  (let ((x (if (> x maxv) maxv x)))
    (if (< x minv) minv x)))

(defun vt-clip (vt min-val max-val &key out dtype)
  "逐元素裁剪。MIN-VAL / MAX-VAL 是标量（不是张量）。
   语义：clip(x, min, max) = min(max, max(min, x))"
  (vt-fast-map #'%op-clip vt min-val max-val :dtype dtype :out out))

(defun vt-lerp (start end weight &key out dtype)
  "逐元素线性插值 start + (end - start) * weight，
      对标 numpy 的 a + (b-a)*t 形式（三者均支持广播）。
      :dtype 决定结果 dtype（缺省按提升规则），:out 契约同 vt-+。"
  (vt-map (lambda (s e w)
	    (+ s (* (- e s) w)))
	  (ensure-vt start) (ensure-vt end) (ensure-vt weight)
          :dtype dtype :out out))

(defun vt-cbrt (vt &key out dtype)
  "逐元素立方根。"
  (let* ((dt (%infer-float-dtype vt dtype))
         (third (if (eq dt :float32)
		    (/ 3.0s0)
		    (/ 3.0d0))))
    (vt-map (lambda (x) (* (signum x) (expt (abs x) third)))
            vt :out out :dtype dt)))
  "逐元素立方根。定义为 signum(x)*|x|^(1/3)，因此负数返回实数值
      （对标 numpy.cbrt，不返回 NaN）。整数输入输出 float64。"

(defun vt-hypot (t1 t2 &key out dtype)
  "逐元素直角三角斜边 sqrt(x² + y²)（对标 numpy.hypot）。任一输入为 ±Inf → 结果 +Inf；NaN 传播。"
  (let* ((dt (or dtype (if (or (eq (vt-dtype (ensure-vt t1)) :float32)
                               (eq (vt-dtype (ensure-vt t2)) :float32))
                           :float32 :float64)))
         (one (if (eq dt :float32) 1.0s0 1.0d0)))
    (vt-map (lambda (a b)
              (let ((abs-a (abs a)) (abs-b (abs b)))
                (cond ((or (%inf-p abs-a) (%inf-p abs-b))
                       ;; NumPy hypot：任一参数 ±Inf → +Inf（即便另一参数为 NaN）
                       (vt-get-pos-inf dt))
                      ((%nan-p abs-a) abs-a)
                      ((%nan-p abs-b) abs-b)
                      ((zerop abs-a) abs-b)
                      ((zerop abs-b) abs-a)
                      (t (let* ((mx (max abs-a abs-b))
                                (mn (min abs-a abs-b))
                                (r  (/ mn mx)))
                           (* mx (sqrt (+ one (* r r)))))))))
            (ensure-vt t1) (ensure-vt t2) :out out :dtype dt)))

(defun vt-reciprocal (vt &key out dtype)
  "逐元素倒数 (1/x)。

   dtype 语义（对标 numpy.reciprocal，v0.3.6 起）：
     整数输入 → **保持整数 dtype**（numpy 1//x 整数除法，1/2 → 0）；
     浮点输入 → 保持原浮点 dtype（float32 进 float32 出）；
     显式 :dtype 优先。
   零除：浮点 → ±Inf（IEEE 754）；整数 → 确定性报错，与 numpy
     整数零除行为一致。"
  (let* ((in-dt (vt-dtype (ensure-vt vt)))
         (dt (or dtype in-dt)))
    (if (vt-int-dtype-p dt)
        ;; 整数倒数：保持整数语义（1//x），零除交给 CL 整数除零错误
        (vt-map (lambda (v) (if (zerop v) (error "vt-reciprocal: 整数零除") (truncate (/ 1 v))))
                vt :out out :dtype dt)
        (let ((one (if (eq dt :float32) 1.0s0 1.0d0)))
          (vt-map (lambda (v) (/ one v)) vt :out out :dtype dt)))))

(defun vt-negative (vt &key out dtype)
  "逐元素取负（-x）。这是一元 vt-- 的具名入口，
      NaN/±Inf 原样传播（对标 numpy.negative）。:out 契约同 vt-+。"
  (vt-fast-map #'- vt :dtype dtype :out out))

(defun vt-sinc (tensor &key out dtype)
  "逐元素归一化 sinc 函数 sin(πx)/(πx)，且 sinc(0)=1（对标 numpy.sinc）。
      整数输入输出 float64，float32 进 float32 出。:out 契约同 vt-+。"
  (let* ((dt (%infer-float-dtype tensor dtype))
	 (x-pi (vt-scale tensor pi :dtype dt)))
    (vt-map (lambda (x)
	      (if (zerop x) 1.0d0 (/ (sin x) x)))
	    x-pi :out out :dtype dt)))
