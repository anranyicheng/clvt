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
             ;; 整型族判定（v0.4.x 修复 BUG：vt-int-dtype-p 不含无符号类型，
             ;; 导致 uint8/uint 等真除法落入整数除法分支返回整型结果）。
             (int-family-p
               (lambda (x)
                 (or (vt-int-dtype-p (vt-dtype x))
                     (member (vt-dtype x) '(:uint8 :uint16)))))
             (dts (mapcar #'vt-dtype all))
             (float-target
               (cond (dtype
                      ;; 显式整型目标：保持截断语义，零除确定性报错
                      nil)
                     ((every int-family-p all)
                      ;; 整型（含无符号）→ float64（true_divide 恒浮点）
                      :float64)
                     ((some int-family-p all)
                      ;; numpy promote：int8/int16/uint8/uint16 与 float32 混合
                      ;; → float32；int32/int64 或 float64 参与 → float64
                      ;; （v0.4.x 修复：原实现对 int32/int64+float32 误返 float32）
                      (if (every (lambda (dt)
                                   (member dt '(:float32 :int8 :int16
                                                :uint8 :uint16)))
                                 dts)
                          :float32
                          :float64))
                     ;; 纯浮点混合：自然提升（float64 优先于 float32，对齐
                     ;; numpy promote(float32,float64)=float64。v0.4.x 修复：
                     ;; 原实现强行取 float32，精度被降级）
                     (t nil)))
             (effective-dtype (or dtype float-target)))
        ;; 计算前把整型输入预转换为浮点（numpy true_divide 语义）：
        ;; 避免 CL 整数除零信号 division-by-zero（numpy 为 ±Inf），
        ;; 也避免 (/ int 0) 混合类型除法在浮点目标下的类型错误。
        ;; 顺带修复：标量除数（如 (vt-/ t 0)）先经 ensure-vt 转为 0 维张量，
        ;; 不再因 vt-dtype 作用于数字而直接类型崩溃。
        (when (and float-target (some int-family-p all))
          (setf all (mapcar (lambda (x)
                              (if (funcall int-family-p x)
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

(defun %floor-div-op (x y)
  "floor_divide 的单元素语义（对标 numpy.floor_divide）。
   v0.4.x 修复：原实现浮点分支误用真除 (/ x y)，未做 floor（0.864/0.15
   返回 5.76 而非 5.0）。现统一 floor 语义，并按 IEEE 补齐非有限值处理：
   NaN 传播；±Inf/有限 → ±Inf；有限/±Inf → ±0.0；有限/±0.0 → 与商
   同号的 ±Inf；±0.0/±0.0 → NaN（均与 numpy 一致）。"
  (cond ((and (integerp x) (integerp y) (zerop y)) 0)
        ((and (integerp x) (integerp y)) (floor x y))
        ((and (floatp x) (%nan-p x)) x)
        ((and (floatp y) (%nan-p y)) y)
        ((and (floatp x) (%inf-p x)) x)
        ((and (floatp y) (%inf-p y)) (nth-value 0 (floor x y)))
        ((and (numberp y) (zerop y))
         (if (zerop x) +vt-dfloat-nan+
             (if (or (and (plusp x) (plusp y))
                     (and (minusp x) (minusp y)))
                 (vt-get-pos-inf :float64)
                 (- (vt-get-pos-inf :float64)))))
        (t (nth-value 0 (floor x y)))))

(defun vt-div (a b &key dtype out)
  "逐元素整除（二元特化入口），floor 除法（对标 numpy.floor_divide）。
      整数输入 floor 除，除数为 0 时返回 0（对标 numpy：静默返回 0）；
      浮点输入同样做 floor（v0.4.x 修复：原实现误用真除，未 floor），
      除数 ±0.0 按 IEEE 得 ±Inf/NaN。
      需要 numpy true_divide 语义（恒浮点、不 floor）请用 vt-/。"
  (vt-map #'%floor-div-op
          (ensure-vt a) (ensure-vt b)
          :dtype dtype :out out))

(defun %int-fit-p (dt n)
  "整数 n 是否能被整型 dtype 精确容纳（弱标量语义判定用）。"
  (case dt
    (:int8   (<= -128 n 127))
    (:uint8  (typep n '(unsigned-byte 8)))
    (:int16  (<= -32768 n 32767))
    (:uint16 (typep n '(unsigned-byte 16)))
    (:int32  (typep n '(signed-byte 32)))
    (:int64  (typep n '(signed-byte 64)))
    (t nil)))

(defun vt-scale (a b &key out dtype)
  "按标量缩放（等价 vt-* a b），NaN/Inf 按 IEEE 754 正常传播。
      标量遵循 numpy 2.x 弱标量（NEP 50）语义（v0.4.x 修复：
      原实现 float32 数组×标量误升级为 float64、整型数组×标量误升级
      为 int64 且不回绕）：
        浮点数组 × 标量        → 保持数组 dtype（float32 不升级）；
        整型数组 × 可容纳整数  → 保持整型 dtype（C 语义回绕，
                                  uint8 249×3 → 235，与 numpy 一致）；
        整型数组 × 浮点标量    → float64；
        整数超出整型范围       → 退回自然提升（int64）。
      B 亦可传入张量（此时按 vt-* 广播提升语义）。
      :dtype 显式给出则完全以其为准；:out 契约同 vt-+。"
  (if (and (null dtype) (numberp b))
      (let ((dt (vt-dtype (ensure-vt a))))
        (vt-fast-map #'* a b :dtype
                     (cond ((member dt '(:float32 :float64)) dt)
                           ((and (integerp b) (%int-fit-p dt b)) dt)
                           ((floatp b) :float64)
                           (t nil))
                     :out out))
      (vt-fast-map #'* a b :out out :dtype dtype)))

(defun %infer-float-dtype (vt dtype)
  "推导逐元素数学函数的结果 dtype（对标 numpy「整型进 float64 出、
   float32 进 float32 出」）。"
  (or dtype (if (eq (vt-dtype vt) :float32) :float32 :float64)))

(defun %coerce-float-input (vt dt)
  "若 VT 已是指定浮点 dtype 则原样返回，否则转换为 DT。
   整数/无符号输入必须经过此步，否则 SBCL 会把 fixnum 元素交给
   `(sin 1)` 这类 CL 数学函数——对整数参数 CL 只返回 **single-float**
   精度（如 (sin 1) = 0.84147096，而 (sin 1d0) = 0.8414709848078965d0），
   造成 dtype 标称 float64 但数值仅 float32 精度的静默精度损失。"
  (if (eq (vt-dtype vt) dt) vt (vt-astype vt dt)))

(defmacro %float-map (fn vt out dt)
  "逐元素数学函数的统一入口：**内联快路径 + dtype 提升** 二者兼得。

   背景（性能契约）：`vt-fast-map` 是宏，编译期把字面算子（如 #'sin）
   内联成特化循环（`%vt-inline1-fast` 等），比运行时经 `funcall` 的
   `vt-map` 快约 1.6–1.8×。而 P1-1 精度修复要求「整数输入必须提升到
   浮点再算」，这个提升只能在运行时判断（输入 dtype 编译期未知）。

   若写成普通函数并委托 `vt-map`（此前 `%float-map-fn` 的做法），
   所有调用——**包括本就无需提升的连续 float64/float32 输入**——
   都会被拖到慢路径上，造成 vt-sin/vt-exp 等常见用例约 2× 的
   性能回归。本宏把 dtype 判断外提为运行时分枝，但**两条分支都在
   编译期展开为 `vt-fast-map` 内联循环**：

     (if (eq (vt-dtype in) <dt>)          ; 已是目标浮点：零拷贝直通
         (vt-fast-map #'op in  :out out :dtype <dt>)
         (vt-fast-map #'op (%coerce-float-input in <dt>) :out out :dtype <dt>))

   快路径（本库绝大多数调用）与改动前完全等价，只有整数输入多付
   一次 astype——这正是精度修复的必要代价。

   FN 必须是 `(function <symbol>)` 形式的字面算子（`#'sin` / `#'asinh`
   / 本文件内定义的一元算子如 `#'%op-eq`）。`vt-fast-map` 只对
   `(function <symbol>)` 做内联；**lambda 算子无法内联**，请改用
   `%float-map-fn`。"
  (check-type fn cons)
  (unless (and (eq (car fn) 'function) (symbolp (cadr fn)))
    (error "%%float-map: FN 必须是 (function <symbol>) 字面算子（如 #'sin）；\
lambda 算子请使用 %%float-map-fn"))
  `(let* ((%in (ensure-vt ,vt))
          (%dt ,dt))
     (if (eq (vt-dtype %in) %dt)
         (vt-fast-map (function ,(cadr fn)) %in :out ,out :dtype %dt)
         (vt-fast-map (function ,(cadr fn))
                      (%coerce-float-input %in %dt)
                      :out ,out :dtype %dt))))

(defun %float-map-fn (fn vt out dt)
  "数学函数的**兜底**慢路径：接受任意 FN（含 lambda），内部走 `vt-map`。
   仅用于「无法在编译期内联算子」的场景（本库为 vt-asin/vt-acos/
   vt-sqrt 等需按元素返回 NaN 的 lambda 分支）。能在编译期拿到字面
   算子的一律用 `%float-map` 宏走内联快路径。"
  (vt-map fn (%coerce-float-input (ensure-vt vt) dt) :out out :dtype dt))

(defmacro %float-map-lambda (fn vt out dt)
  "lambda 算子的数学函数入口：与 `%float-map` 同构（含 dtype 提升
   运行时分枝），同样保证「整数输入先提升再计算」的 P1-1 精度契约。

   与 `%float-map` 的唯一区别在快路径：`vt-fast-map` 只内联
   `(function <symbol>)` 字面算子，**无法内联 lambda**，故这里两条
   分支都走 `vt-map`。好处是：(1) 避免 `%float-map-fn` 那种「调用方
   自己先 astype、再交给 vt-map」在整数输入上少走一次提升的漏洞；
   (2) 已浮点输入零拷贝直通，不做多余 astype。"
  `(let* ((%in (ensure-vt ,vt))
          (%dt ,dt))
     (if (eq (vt-dtype %in) %dt)
         (vt-map ,fn %in :out ,out :dtype %dt)
         (vt-map ,fn (%coerce-float-input %in %dt) :out ,out :dtype %dt))))

(defun vt-sin (vt &key out dtype)
  "逐元素正弦。整数输入输出按 :dtype 或 float64（对标 numpy，
      整数进 float64 出）；float32 进 float32 出。:out 契约同 vt-+。"
  (%float-map #'sin vt out (%infer-float-dtype vt dtype)))


(defun vt-cos (vt &key out dtype)
  "逐元素余弦。整数输入输出 float64，float32 进 float32 出。
      :out 契约同 vt-+。"
  (%float-map #'cos vt out (%infer-float-dtype vt dtype)))


(defun vt-tan (vt &key out dtype)
  "逐元素正切。整数输入输出 float64，float32 进 float32 出。
      :out 契约同 vt-+。"
  (%float-map #'tan vt out (%infer-float-dtype vt dtype)))


(defun vt-atan (vt &key out dtype)
  "逐元素反正切。整数输入输出 float64，float32 进 float32 出。
      :out 契约同 vt-+。"
  (%float-map #'atan vt out (%infer-float-dtype vt dtype)))


(defun vt-sinh (vt &key out dtype)
  "逐元素双曲正弦。整数输入输出 float64，float32 进 float32 出。
      :out 契约同 vt-+。"
  (%float-map #'sinh vt out (%infer-float-dtype vt dtype)))


(defun vt-cosh (vt &key out dtype)
  "逐元素双曲余弦。整数输入输出 float64，float32 进 float32 出。
      :out 契约同 vt-+。"
  (%float-map #'cosh vt out (%infer-float-dtype vt dtype)))


(defun vt-tanh (vt &key out dtype)
  "逐元素双曲正切。整数输入输出 float64，float32 进 float32 出。
      :out 契约同 vt-+。"
  (%float-map #'tanh vt out (%infer-float-dtype vt dtype)))


(defun vt-asin (vt &key out dtype)
  "逐元素反正弦。|x|>1 返回 NaN（对标 numpy，不抛条件）。
      整数输入输出 float64，float32 进 float32 出。:out 契约同 vt-+。"
  (let* ((dt (%infer-float-dtype vt dtype))
         (nan (vt-get-nan dt)))
    (%float-map-lambda (lambda (x)
                            (if (> (abs x) 1.0d0) nan (asin x)))
                          vt out dt)))

(defun vt-acos (vt &key out dtype)
  "逐元素反余弦。|x|>1 返回 NaN（对标 numpy，不抛条件）。
      整数输入输出 float64，float32 进 float32 出。:out 契约同 vt-+。"
  (let* ((dt (%infer-float-dtype vt dtype))
         (nan (vt-get-nan dt)))
    (%float-map-lambda (lambda (x)
                            (if (> (abs x) 1.0d0) nan (acos x)))
                          vt out dt)))

(defun vt-asinh (vt &key out dtype)
  "逐元素反双曲正弦。整数输入输出 float64，float32 进 float32 出。
      :out 契约同 vt-+。"
  (%float-map #'asinh vt out (%infer-float-dtype vt dtype)))


(defun vt-acosh (vt &key out dtype)
  "逐元素反双曲余弦。x<1 返回 NaN（对标 numpy）。
      整数输入输出 float64，float32 进 float32 出。:out 契约同 vt-+。"
  (let* ((dt (%infer-float-dtype vt dtype))
         (nan (vt-get-nan dt)))
    (%float-map-lambda (lambda (x)
                            (if (< x 1.0d0) nan (acosh x)))
                          vt out dt)))

(defun vt-atanh (vt &key out dtype)
  "逐元素反双曲正切，对标 numpy.arctanh。
      |x|>1 返回 NaN；x=+1 返回 +Inf、x=-1 返回 -Inf（IEEE 极限，非 NaN）。
      整数输入输出 float64，float32 进 float32 出。:out 契约同 vt-+。"
  (let* ((dt (%infer-float-dtype vt dtype))
         (nan (vt-get-nan dt))
         (pos-inf (vt-get-pos-inf dt))
         (neg-inf (vt-get-neg-inf dt)))
    (%float-map-lambda (lambda (x)
                         (cond ((> (abs x) 1.0d0) nan)
                               ((= x 1.0d0) pos-inf)
                               ((= x -1.0d0) neg-inf)
                               (t (atanh x))))
                       vt out dt)))

(defun vt-exp (vt &key out dtype)
  "逐元素指数。整数输入输出 float64，float32 进 float32 出。
      :out 契约同 vt-+。"
  (%float-map #'exp vt out (%infer-float-dtype vt dtype)))


(defun %pow-odd-int-p (y)
  "y 是否为奇整数值（支持浮点表示的整数，如 -3.0d0）。"
  (cond ((integerp y) (oddp y))
        ((and (floatp y) (= y (ftruncate y))) (oddp (truncate y)))
        (t nil)))

(defun %pow-int-valued-p (y)
  "y 是否为整数值（整数或整值浮点）。"
  (or (integerp y)
      (and (floatp y) (= y (ftruncate y)))))

(defun %pow-op (x y nan)
  "逐元素幂的 C99 pow 特例表（对标 numpy.power 实数语义）。
   覆盖：pow(x,±0)=1、pow(±1,y)=1、pow(±0,y)=±0/±Inf（奇偶）、
   pow(±Inf,y)=±Inf/±0/NaN（整值奇偶）、pow(负底数,非整指数)=NaN。
   x/y 可为整数或浮点（vt-map 原始元素）；结果由写入层按 dtype 转换。"
  (labels ((pos-inf () sb-ext:double-float-positive-infinity)
           (neg-inf () sb-ext:double-float-negative-infinity))
    (cond
      ;; 1) pow(x, ±0) = 1（C99，含 NaN 底数与 ±Inf 底数）
      ((and (numberp y) (zerop y)) 1)
      ;; 2) NaN 底数
      ((and (floatp x) (%nan-p x)) x)
      ;; 3) pow(1, y) = 1（C99：含 y=NaN）
      ((and (numberp x) (= x 1)) 1)
      ;; 4) NaN 指数
      ((and (floatp y) (%nan-p y)) y)
      ;; 5) ±Inf 底数（y 非零非 NaN）
      ((and (floatp x) (%inf-p x))
       (cond ((not (%pow-int-valued-p y))
              (if (plusp x) (pos-inf) nan))       ; -Inf**非整数 → NaN
             ((minusp y)
              (if (and (minusp x) (%pow-odd-int-p y)) -0.0d0 0.0d0))
             (t
              (if (and (minusp x) (%pow-odd-int-p y)) (neg-inf) (pos-inf)))))
      ;; 6) ±Inf 指数（x 有限；x=±1 已被 3) 拦截）
      ((and (floatp y) (%inf-p y))
       (if (< (abs x) 1)
           (if (plusp y) 0.0d0 (pos-inf))
           (if (plusp y) (pos-inf) 0.0d0)))
      ;; 7) x = ±0（y 非零非 NaN）：正指数 → ±0，负指数 → ±Inf（奇偶定号）
      ((and (numberp x) (zerop x))
       (if (plusp y)
           (if (and (minusp x) (%pow-odd-int-p y)) -0.0d0 0.0d0)
           (if (and (minusp x) (%pow-odd-int-p y)) (neg-inf) (pos-inf))))
      ;; 8) 负底数 + 非整值指数 → NaN
      ((and (minusp x) (not (%pow-int-valued-p y))) nan)
      ;; 9) 常规幂（整数指数 → 精确；负整数指数的整数对由调用方处理）
      (t (expt x y)))))

(defun vt-pow (vt power &key out dtype)
  "逐元素幂 VT**POWER。POWER 为标量指数或张量（张量时逐元素广播幂，
      对标 numpy.power，v0.4.x 起支持）。
      整数 + 非负整数指数 → 精确整数幂，保持整数 dtype；负整数指数按
      numpy 整型语义返回 0。其余走浮点幂并按 C99 pow 特例表处理
      非有限值（pow(-8,1/3)=NaN、pow(0,-1)=+Inf、pow(1,NaN)=1 等）。
      :out 契约同 vt-+。"
  (let ((vt (ensure-vt vt)))
    (if (vt-p power)
        ;; ---- 张量指数：逐元素广播幂（自然提升，与 numpy.power 一致）----
        (let* ((promoted (if dtype
                             dtype
                             (vt-promote-type (vt-dtype vt) (vt-dtype power))))
               (int-result (member promoted
                                   '(:int8 :int16 :int32 :int64 :uint8 :uint16)))
               (nan (vt-get-nan :float64)))
          (if int-result
              ;; numpy 整型幂：负整数指数 → 0（C99 pow 后截断为整型）
              (vt-map (lambda (x y) (if (minusp y) 0 (expt x y)))
                      vt power :out out :dtype dtype)
              (vt-map (lambda (x y) (%pow-op x y nan))
                      vt power :out out :dtype dtype)))
        ;; ---- 标量指数：原路径 ----
        (let* ((in-dt (vt-dtype vt))
               (int-pow-p (and (integerp power) (plusp power)))
               ;; 整数基 + 正整数指数 → 保持整数 dtype（对标 numpy）；
               ;; 其余一律浮点（float32 进 float32 出，整数→float64）。
               (dt (cond (dtype dtype)
                         ((and int-pow-p (member in-dt '(:int8 :int16 :int32 :int64
                                                         :uint8 :uint16)))
                          in-dt)
                         ((eq in-dt :float32) :float32)
                         (t :float64)))
               (nan (vt-get-nan (if (member dt '(:float64 :float32)) dt :float64))))
          (cond
            (int-pow-p
             (vt-map (lambda (x) (expt x power)) vt :out out :dtype dt))
            (t
             (let ((vf (if (member dt '(:float64 :float32)) vt (vt-astype vt dt))))
               (vt-map (lambda (x)
                         (let ((result (handler-case (expt x power)
                                         (error () nan))))
                           (if (realp result) result nan)))
                       vf :out out :dtype dt))))))))

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
    (%float-map-lambda (lambda (x) (if (minusp x) nan (sqrt x))) vt out dt)))

(defun vt-log (vt &key base out dtype)
  "逐元素自然对数；:base 给定时为换底对数。
      语义对标 numpy.log：x>0 正常；x=0 → -Inf（或 log(base)<0 时 +Inf）；
      x<0 → NaN；非法 base（<=0 或 =1）→ 全 NaN。
      整数输入输出 float64，float32 进 float32 出。:out 契约同 vt-+。"
  (let* ((dt (%infer-float-dtype vt dtype))
         (nan (vt-get-nan dt))
         (neginf (vt-get-neg-inf dt))
         (posinf (vt-get-pos-inf dt)))
    (let ((af (%coerce-float-input (ensure-vt vt) dt)))
      (cond
        ((and base (or (<= base 0) (= base 1)))
         (vt-map (lambda (x)
                   (declare (ignore x)) nan)
                 af :out out :dtype dt))
        ((null base)
         (vt-map (lambda (x)
                   (if (> x 0)
                       (log x)
                       (if (zerop x)
                           neginf nan)))
                 af :out out :dtype dt))
        (t
         (let ((zero-result (if (plusp (log base)) neginf posinf)))
           (vt-map (lambda (x)
                     (if (> x 0)
                         (log x base)
                         (if (zerop x)
                             zero-result nan)))
                   af :out out :dtype dt)))))))

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
      NaN 原样传播。:out 契约同 vt-+。
      v0.4.x 修复：signum(-0.0) 原 CL 语义返回 -0.0，现对浮点零
      统一返回 +0.0（对标 numpy.sign：sign(-0.0)=0.0）。"
  (vt-map (lambda (x)
            (cond ((%nan-p x) x)
                  ((and (floatp x) (zerop x)) 0.0d0)
                  (t (signum x))))
          vt :out out :dtype dtype))

(defun vt-positive-p (vt &key out (dtype :int8))
  "逐元素正数判定，返回值 1/0，dtype 默认 :int8（承载布尔语义）。
      对标 numpy 的符号过滤语义（>0）。NaN 比较恒假 → 0。:out 契约同 vt-=。"
  (vt-map (lambda (v)
            (if (> v 0.0d0) 1 0))
          vt :out out :dtype dtype))

(defun vt-negative-p (vt &key out (dtype :int8))
  "逐元素负数判定，返回值 1/0，dtype 默认 :int8（承载布尔语义）。
      NaN 比较恒假 → 0。:out 契约同 vt-=。"
  (vt-map (lambda (v)
            (if (< v 0.0d0) 1 0))
          vt :out out :dtype dtype))

(defun vt-zero-p (vt &key out (dtype :int8))
  "逐元素零判定，返回值 1/0，dtype 默认 :int8（承载布尔语义）。
      :out 契约同 vt-=。"
  (vt-map (lambda (v)
            (if (zerop v) 1 0))
          vt :out out :dtype dtype))

(defun vt-nonzero-p (vt &key out (dtype :int8))
  "逐元素非零判定，返回值 1/0，dtype 默认 :int8（承载布尔语义）。
      :out 契约同 vt-=。"
  (vt-map (lambda (v)
            (if (zerop v) 0 1))
          vt :out out :dtype dtype))

(defun vt-even-p (vt &key out (dtype :int8))
  "逐元素偶数判定，返回值 1/0，dtype 默认 :int8（承载布尔语义）。
      NaN/±Inf 视为非偶数 → 0；非整数值视为非偶数 → 0
      （对标模拟基准 v%2==0）。:out 契约同 vt-=。"
  (vt-map (lambda (v)
            (cond ((or (%nan-p v) (%inf-p v)) 0)
                  ((and (floatp v) (/= v (floor v))) 0)
                  (t (if (evenp (floor v)) 1 0))))
          vt :out out :dtype dtype))

(defun vt-odd-p (vt &key out (dtype :int8))
  "逐元素奇数判定，返回值 1/0，dtype 默认 :int8（承载布尔语义）。
      NaN/±Inf 视为非奇数 → 0；非整数值视为非奇数 → 0
      （对标模拟基准 v%2==1）。:out 契约同 vt-=。"
  (vt-map (lambda (v)
            (cond ((or (%nan-p v) (%inf-p v)) 0)
                  ((and (floatp v) (/= v (floor v))) 0)
                  (t (if (oddp (floor v)) 1 0))))
          vt :out out :dtype dtype))

(defun %mod-nan-or-inf-p (x)
  "取模语境下的非有限值判定（需已屏蔽浮点陷阱）。"
  (and (floatp x) (%nan-or-inf-p x)))

(declaim (inline %mod-float-nan))
(defun %mod-float-nan (x)
  "取模遇 NaN 时的返回值。写入按输出 dtype 转换，恒返 double NaN 即可。"
  (declare (ignore x))
  +vt-dfloat-nan+)

(defun %float-exact-rational (x)
  "浮点的精确有理数值（integer-decode-float）。
   注意不能用 rationalize：它取区间内最简分数而非精确值。"
  (multiple-value-bind (sig exp sign) (integer-decode-float x)
    (* sig (expt 2 exp) sign)))

(defun %float-fmod-exact (x y)
  "精确 fmod（IEEE 754：fmod 结果必然可被输入格式精确表示，coerce 无舍入）。
   返回与 x 同号。仅用于 float32 语境（任一操作数为 single-float）；
   float64 语境的 (mod double double) 误差 ≤ 1ulp，走容差路径。"
  (let ((rx (%float-exact-rational x)) (ry (%float-exact-rational y)))
    (multiple-value-bind (q r) (truncate rx ry)
      (declare (ignore q))
      (coerce r (if (or (typep x 'double-float) (typep y 'double-float))
                    'double-float 'single-float)))))

(defun %exact-float-mod (x y)
  "浮点 mod（结果与除数同号，对标 numpy.remainder）。
   v0.4.x 修复：实测 SBCL 的 (mod single single) 存在数十 ulp 舍入误差，
   且 numpy 的 float32 remainder 并非精确 fmod——其算法为：
     r = fmod(a, b)（精确）
     若 r ≠ 0 且 sign(r) ≠ sign(b)：r = r + b（float32 算术，含一次舍入）
     修正后若符号仍与 b 不符（加法进位所致）→ 0
   本函数逐算法复刻该语义；float64 语境沿用 (mod double double)
   （误差 ≤ 1ulp，通过容差比对）。混合整型操作数沿用 CL 语义。"
  (if (and (floatp x) (floatp y)
           (or (typep x 'single-float) (typep y 'single-float)))
      (let ((r (%float-fmod-exact x y)))
        (cond ((= r 0.0f0) r)
              ((or (and (plusp r) (minusp y))
                   (and (minusp r) (plusp y)))
               (let ((r2 (+ r y)))
                 (if (or (and (plusp r2) (minusp y))
                         (and (minusp r2) (plusp y)))
                     0.0f0
                     r2)))
              (t r)))
      (mod x y)))

(defun %exact-float-rem (x y)
  "浮点 rem（结果与被除数同号，对标 numpy.fmod = C fmodf，精确）。"
  (if (and (floatp x) (floatp y)
           (or (typep x 'single-float) (typep y 'single-float)))
      (%float-fmod-exact x y)
      (rem x y)))

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
                      (t (%exact-float-mod x divisor))))
              vt :out out :dtype dtype)
      (vt-map (lambda (x y)
                (cond ((%mod-nan-or-inf-p x) (%mod-float-nan x))
                      ((%mod-nan-or-inf-p y) (%mod-float-nan x))
                      ((zerop y) (zero-div x))
                      ((and (floatp y) (%inf-p y)) x)
                      (t (%exact-float-mod x y))))
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
                      (t (%exact-float-rem x divisor))))
              vt :out out :dtype dtype)
      (vt-map (lambda (x y)
                (cond ((%mod-nan-or-inf-p x) (%mod-float-nan x))
                      ((%mod-nan-or-inf-p y) (%mod-float-nan x))
                      ((zerop y) (zero-div x))
                      ((and (floatp y) (%inf-p y)) x)
                      (t (%exact-float-rem x y))))
              vt divisor :out out :dtype dtype))))

(defun vt-atan2 (vty vtx &key out dtype)
  "逐元素双参数反正切 atan2(y, x)。参数顺序 VTY, VTX。
      整数输入输出 float64，float32 进 float32 出。:out 契约同 vt-+。"
  (vt-fast-map #'atan vty vtx :out out :dtype dtype))

;; floor 族对 NaN/±Inf 统一传播本身（对标 numpy：floor/ceil/trunc/round(nan)=nan，
;; floor(±inf)=±inf）。SBCL 对非有限值直接调用 floor/round 会 signal
;; FLOATING-POINT-INVALID-OPERATION（trap 屏蔽也无法避免），必须显式拦截。
(defmacro %floor-family-body (x op)
  ;; 对标 numpy：结果为 ±0 时符号跟随输入（floor/trunc(-0.5)=-0.0，
  ;; ceil(-0.5)=-0.0）；CL floor 族对 (-0.5,0) 区间返回泛化整数 0（恒 +0），
  ;; 需显式恢复负零符号。
  `(if (and (floatp ,x) (%nan-or-inf-p ,x))
       ,x
       (let ((res (nth-value 0 (,op ,x divisor))))
         (if (floatp ,x)
             (if (and (zerop res) (minusp (float-sign ,x)))
                 (- (float 0.0 ,x))
                 (float res ,x))
             res))))

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
                  (if (floatp x)
                      (if (and (zerop res) (minusp (float-sign x)))
                          (- (float 0.0 x))
                          (float res x))
                      res))))
          vt :out out :dtype dtype))

(declaim (inline %op-eq %op-ne %op-lt %op-le %op-gt %op-ge))
(defun %op-eq (a b) (if (=  a b) 1.0d0 0.0d0))

(defun %op-ne (a b) (if (/= a b) 1.0d0 0.0d0))

(defun %op-lt (a b) (if (<  a b) 1.0d0 0.0d0))

(defun %op-le (a b) (if (<= a b) 1.0d0 0.0d0))

(defun %op-gt (a b) (if (>  a b) 1.0d0 0.0d0))

(defun %op-ge (a b) (if (>= a b) 1.0d0 0.0d0))

(defun vt-= (t1 t2 &key (dtype :int8) out)
  "逐元素相等判定，返回值 1/0，dtype 默认 :int8（承载布尔语义，CONVENTIONS §3.1）。
      NaN 与任何值比较恒假 → 0（含 NaN == NaN）。
      :out dtype 必须精确等于结果 dtype（§4.2 H3，默认 :int8）。"
  (vt-fast-map #'%op-eq (ensure-vt t1) (ensure-vt t2)
               :dtype dtype :out out))

(defun vt-/= (t1 t2 &key (dtype :int8) out)
  "逐元素不等判定，返回值 1/0，dtype 默认 :int8（承载布尔语义）。
      NaN 与任何值比较恒真 → 1（含 NaN != NaN）。:out 契约同 vt-=。"
  (vt-fast-map #'%op-ne (ensure-vt t1) (ensure-vt t2)
               :dtype dtype :out out))

(defun vt-< (t1 t2 &key (dtype :int8) out)
  "逐元素小于判定，返回值 1/0，dtype 默认 :int8（承载布尔语义）。
      NaN 参与的比较恒假 → 0。:out 契约同 vt-=。"
  (vt-fast-map #'%op-lt (ensure-vt t1) (ensure-vt t2)
               :dtype dtype :out out))

(defun vt-<= (t1 t2 &key (dtype :int8) out)
  "逐元素小于等于判定，返回值 1/0，dtype 默认 :int8（承载布尔语义）。
      NaN 参与的比较恒假 → 0。:out 契约同 vt-=。"
  (vt-fast-map #'%op-le (ensure-vt t1) (ensure-vt t2)
               :dtype dtype :out out))

(defun vt-> (t1 t2 &key (dtype :int8) out)
  "逐元素大于判定，返回值 1/0，dtype 默认 :int8（承载布尔语义）。
      NaN 参与的比较恒假 → 0。:out 契约同 vt-=。"
  (vt-fast-map #'%op-gt (ensure-vt t1) (ensure-vt t2)
               :dtype dtype :out out))

(defun vt->= (t1 t2 &key (dtype :int8) out)
  "逐元素大于等于判定，返回值 1/0，dtype 默认 :int8（承载布尔语义）。
      NaN 参与的比较恒假 → 0。:out 契约同 vt-=。"
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

(defun vt-logical-and (t1 t2 &key out (dtype :int8))
  "逐元素逻辑与（非零视为真），返回 1/0。
   dtype 默认 :int8——与 §3.1「比较/逻辑运算返回 :int8」一致，
   语义等价 numpy 的 bool（本库不引入独立 :bool dtype）。"
  (vt-map (lambda (a b)
            (if (and (not (zerop a)) (not (zerop b))) 1 0))
          (ensure-vt t1) (ensure-vt t2) :dtype dtype :out out))

(defun vt-logical-or (t1 t2 &key out (dtype :int8))
  "逐元素逻辑或（非零视为真），返回 1/0。dtype 默认 :int8（同 §3.1）。"
  (vt-map (lambda (a b)
            (if (or (not (zerop a)) (not (zerop b))) 1 0))
          (ensure-vt t1) (ensure-vt t2) :dtype dtype :out out))

(defun vt-logical-not (vt &key out (dtype :int8))
  "逐元素逻辑非，返回 1/0。dtype 默认 :int8（同 §3.1）。"
  (vt-map (lambda (v)
            (if (zerop v) 1 0))
          vt :dtype dtype :out out))

(defun vt-logical-xor (t1 t2 &key out (dtype :int8))
  "逐元素逻辑异或，返回 1/0。dtype 默认 :int8（同 §3.1）。"
  (vt-map (lambda (a b)
            (if (not (eq (not (zerop a)) (not (zerop b)))) 1 0))
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
  "三元 clip：先下限后上限，即 (min maxv (max minv x))。
   v0.4.0 修复：原实现先上限后下限，当 minv > maxv 时结果为 minv，
   而 numpy.clip(min>max) 的结果是 maxv（clamp 顺序以 max 收尾）。"
  (let ((x (if (< x minv) minv x)))
    (if (> x maxv) maxv x)))

(defun vt-clip (vt &optional (min-val nil min-p) (max-val nil max-p)
                   &key out dtype)
  "逐元素裁剪，完全对标 numpy.clip 的签名与边界语义。

  签名对齐 numpy 2.3.5 的 `clip(a, a_min=None, a_max=None)`：
    - `(vt-clip a)`              两个边界均省略 → 返回 a 的副本（不裁剪）；
    - `(vt-clip a nil max)`      仅给上限（MIN-VAL 传 nil）→ min(x, max)；
    - `(vt-clip a min nil)`      仅给下限（MAX-VAL 传 nil）→ max(x, min)；
    - `(vt-clip a min max)`      双侧裁剪 → min(max, max(min, x))；
    - `(vt-clip a min)`          仅给 MIN-VAL 而没有 MAX-VAL → **报错**
      （对齐 numpy 的 `TypeError: clip() missing 1 required
      positional argument: 'a_max'`；numpy 不允许只给下限）。

  MIN-VAL / MAX-VAL 是标量，不是张量。:dtype / :out 契契约同 vt-map
  （结果 dtype 由输入提升或显式 :dtype 决定，out 必须精确匹配）。"
  (when (and min-p (not max-p))
    (error "vt-clip: 只提供 MIN-VAL 而未提供 MAX-VAL（numpy 语义：clip() \
missing a_max；若只要下限请显式传 MAX-VAL 为 nil）"))
  (cond
    ;; 不裁剪（两边界均省略）：直接返回副本，保持 dtype/形状不变
    ((and (null min-val) (null max-val))
     (let ((r (if out
                  (vt-check-out out (vt-shape vt) (or dtype (vt-dtype vt))
                                :op-name "vt-clip")
                  (vt-zeros (vt-shape vt) :dtype (or dtype (vt-dtype vt))))))
       (vt-copy-into r vt)
       r))
    ;; 仅上限：min(x, max)
    ((null min-val)
     (vt-map (lambda (x) (min x max-val)) vt :dtype dtype :out out))
    ;; 仅下限：max(x, min)
    ((null max-val)
     (vt-map (lambda (x) (max x min-val)) vt :dtype dtype :out out))
    ;; 双侧：min(max, max(min, x))，与 %op-clip 同义
    (t (vt-fast-map #'%op-clip vt min-val max-val :dtype dtype :out out))))

(defun vt-lerp (start end weight &key out dtype)
  "逐元素线性插值 start + (end - start) * weight，
      对标 numpy 的 a + (b-a)*t 形式（三者均支持广播）。
      :dtype 决定结果 dtype（缺省按提升规则），:out 契约同 vt-+。"
  (vt-map (lambda (s e w)
            (+ s (* (- e s) w)))
          (ensure-vt start) (ensure-vt end) (ensure-vt weight)
          :dtype dtype :out out))

(defun vt-cbrt (vt &key out dtype)
  "逐元素立方根。定义为 signum(x)*|x|^(1/3)，因此负数返回实数值
      （对标 numpy.cbrt，不返回 NaN）。整数输入输出 float64。
      v0.4.x 修复：cbrt(NaN) 原经 (expt NaN 1/3) 泄漏复数 NaN 导致
      类型错误，现显式拦截：NaN → NaN、±Inf → ±Inf（numpy 对齐）。"
  (let* ((dt (%infer-float-dtype vt dtype))
         (third (if (eq dt :float32)
                    (/ 3.0s0)
                    (/ 3.0d0))))
    (vt-map (lambda (x)
              (cond ((%nan-p x) x)
                    ((%inf-p x) x)
                    (t (* (signum x) (expt (abs x) third)))))
            vt :out out :dtype dt)))

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
