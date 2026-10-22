;;;; elementwise.lisp — 逐元素算术 / 数学 / 比较 / 逻辑 / 位运算

#+sbcl (sb-ext::gc :full t) ;; 在编译期间垃圾回收内存，防止爆炸

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
             ;; 弱标量（NEP 50）：Lisp 标量只贡献「种类」，不参与宽度判定。
             ;; 于是 (vt-/ f32-数组 2) 保持 float32（numpy 同），而
             ;; (vt-/ int-数组 2) 仍是 float64（数组本身是整型）。
             (tensor-dts (loop for raw in (cons vt tensors) for x in all
                               unless (%scalar-kind raw) collect (vt-dtype x)))
             ;; dtype 版谓词（上面那个 INT-FAMILY-P 吃的是张量对象）
             (dt-int-p (lambda (dt)
                         (or (vt-int-dtype-p dt) (member dt '(:uint8 :uint16)))))
             (float-target
               (cond (dtype
                      ;; 显式整型目标：保持截断语义，零除确定性报错
                      nil)
                     ((or (null tensor-dts) (every dt-int-p tensor-dts))
                      ;; 整型（含无符号）张量 → float64（true_divide 恒浮点）；
                      ;; 全是标量时也落 float64（numpy: 1/2 = 0.5）
                      :float64)
                     ((some dt-int-p tensor-dts)
                      ;; numpy promote：int8/int16/uint8/uint16 与 float32 混合
                      ;; → float32；int32/int64 或 float64 参与 → float64
                      ;; （v0.4.x 修复：原实现对 int32/int64+float32 误返 float32）
                      (if (every (lambda (dt)
                                   (member dt '(:float32 :int8 :int16
                                                :uint8 :uint16)))
                                 tensor-dts)
                          :float32
                          :float64))
                     ;; 纯浮点混合：自然提升（float64 优先于 float32，对齐
                     ;; numpy promote(float32,float64)=float64。v0.4.x 修复：
                     ;; 原实现强行取 float32，精度被降级）
                     (t nil)))
             ;; 自然提升（含弱标量语义）。必须在这里算定并显式下传：
             ;; 下面的 (first all)/(second all) 是**已 ensure-vt** 的值，
             ;; 再交给 vt-fast-map 时「谁是标量」已经认不出来了。
             (natural-dtype (%op-final-dtype
                             (mapcar #'%scalar-kind (cons vt tensors))
                             (mapcar #'vt-dtype all)))
             (effective-dtype (or dtype float-target natural-dtype)))
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
    (when (%cx-reject a b)
    (error "vt-div: 不支持复数输入（numpy 同样拒绝）。见 CONVENTIONS §16.11"))
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
                           ;; 复数数组 × 实数标量：numpy 保持复数 dtype
                           ;; （complex64×2.0 → complex64）。缺这一支时会把
                           ;; 结果 dtype 判成 :float64，复数结果写不进实数
                           ;; 缓冲，直接在 %DOUBLE-FLOAT 处报错 —— 实测
                           ;; vt-scale 是 vt-cov 复数路径的第一个绊脚石。
                           ((member dt '(:complex64 :complex128)) dt)
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
  (when (%cx-any-p vt)
    (return-from vt-sin
      (let* ((in (ensure-vt vt))
             (dt (%cx-result-complex-dtype (vt-dtype in) dtype "VT-SIN")))
        (vt-fast-map (function sin) in :out out :dtype dt))))
  (%float-map #'sin vt out (%infer-float-dtype vt dtype)))

;;; ==================================================================
;;; 一元数学函数的符号算子（供 vt-fast-map 编译期内联）
;;;
;;; 为什么需要这一组：`vt-fast-map` 只在算子写成 `(function <symbol>)`
;;; 字面量时才做内联展开；写成 `(lambda ...)` 一律回退到 `vt-map` 的
;;; `funcall` + 类型未知的通用循环。1M double 实测（v0.4.8）：
;;;   sqrt 48ms / log 57ms / log2 103ms / cbrt 143ms，
;;; 而同为 libm 调用的 sin（已是符号算子）只有 22ms —— 差距全在
;;; 每元素一次 funcall 与泛型 aref 上。
;;;
;;; 为什么单/双精度各写一份：原 lambda 路径里 single-float 输入走的是
;;; **单精度** libm（(sqrt 2.0s0) ≠ (float (sqrt 2d0) 1.0s0)），
;;; 统一升双再降回会差 1 ulp，破坏「快路径与通用路径逐位可比」。
;;; ==================================================================

;;; 单/双精度算子分派宏：两条分支都展开为 vt-fast-map 内联循环。
(defmacro %float-map-pair (dfn sfn vt out dt)
  (let ((d (gensym "DT")))
    `(let ((,d ,dt))
       (if (eq ,d :float32)
           (%float-map (function ,sfn) ,vt ,out ,d)
           (%float-map (function ,dfn) ,vt ,out ,d)))))

(declaim (inline %op-sqrt-df %op-sqrt-sf))
(defun %op-sqrt-df (x)
  (let ((x (coerce x 'double-float)))
    (if (minusp x) +vt-dfloat-nan+ (sqrt x))))
(defun %op-sqrt-sf (x)
  (let ((x (coerce x 'single-float)))
    (if (minusp x) +vt-sfloat-nan+ (sqrt x))))

(declaim (inline %op-asin-df %op-asin-sf %op-acos-df %op-acos-sf))
(defun %op-asin-df (x)
  (let ((x (coerce x 'double-float)))
    ;; NaN 必须前置拦截：CL 的 (asin nan) 返回**复数** #C(nan nan)，
    ;; 写回 double-float 数组时会抛 "is not of type REAL"。
    ;; 原 lambda 实现同样漏了这一步（asin/acos/atanh 传 NaN 直接崩），
    ;; numpy 的对应函数返回 nan —— 此处一并修正。
    (cond ((/= x x) x)
          ((> (abs x) 1.0d0) +vt-dfloat-nan+)
          (t (asin x)))))
(defun %op-asin-sf (x)
  (let ((x (coerce x 'single-float)))
    (cond ((/= x x) x)
          ((> (abs x) 1.0s0) +vt-sfloat-nan+)
          (t (asin x)))))
(defun %op-acos-df (x)
  (let ((x (coerce x 'double-float)))
    (cond ((/= x x) x)
          ((> (abs x) 1.0d0) +vt-dfloat-nan+)
          (t (acos x)))))
(defun %op-acos-sf (x)
  (let ((x (coerce x 'single-float)))
    (cond ((/= x x) x)
          ((> (abs x) 1.0s0) +vt-sfloat-nan+)
          (t (acos x)))))

(declaim (inline %op-acosh-df %op-acosh-sf))
(defun %op-acosh-df (x)
  (let ((x (coerce x 'double-float)))
    (if (< x 1.0d0) +vt-dfloat-nan+ (acosh x))))
(defun %op-acosh-sf (x)
  (let ((x (coerce x 'single-float)))
    (if (< x 1.0s0) +vt-sfloat-nan+ (acosh x))))

(declaim (inline %op-atanh-df %op-atanh-sf))
(defun %op-atanh-df (x)
  (let ((x (coerce x 'double-float)))
    (cond ((/= x x) x)
          ((> (abs x) 1.0d0) +vt-dfloat-nan+)
          ((= x 1.0d0) +vt-dfloat-pos-inf+)
          ((= x -1.0d0) +vt-dfloat-neg-inf+)
          (t (atanh x)))))
(defun %op-atanh-sf (x)
  (let ((x (coerce x 'single-float)))
    (cond ((/= x x) x)
          ((> (abs x) 1.0s0) +vt-sfloat-nan+)
          ((= x 1.0s0) +vt-sfloat-pos-inf+)
          ((= x -1.0s0) +vt-sfloat-neg-inf+)
          (t (atanh x)))))

;;; 对数族。LOG2/LOG10 的底保持与原实现同型的常量：原代码是
;;; `(log x base)`，base 为 double-float 变量 —— 对 single 输入会先把 x
;;; 提升到 double 再算，故 sf 版算子直接返回 double，由内联循环的
;;; %cast-to 降回 single，与原路径逐位一致。

(declaim (inline %op-log-df %op-log-sf))
(defun %op-log-df (x)
  (let ((x (coerce x 'double-float)))
    (if (> x 0.0d0)
        (log x)
        (if (zerop x) +vt-dfloat-neg-inf+ +vt-dfloat-nan+))))
(defun %op-log-sf (x)
  (let ((x (coerce x 'single-float)))
    (if (> x 0.0s0)
        (log x)
        (if (zerop x) +vt-sfloat-neg-inf+ +vt-sfloat-nan+))))

(declaim (inline %op-log2-df %op-log2-sf %op-log10-df %op-log10-sf))
(defun %op-log2-df (x)
  (let ((x (coerce x 'double-float)))
    (if (> x 0.0d0)
        (log x 2.0d0)
        (if (zerop x) +vt-dfloat-neg-inf+ +vt-dfloat-nan+))))
(defun %op-log2-sf (x)
  (let ((x (coerce x 'single-float)))
    (if (> x 0.0s0)
        (log x 2.0d0)
        (if (zerop x) +vt-sfloat-neg-inf+ +vt-sfloat-nan+))))
(defun %op-log10-df (x)
  (let ((x (coerce x 'double-float)))
    (if (> x 0.0d0)
        (log x 10.0d0)
        (if (zerop x) +vt-dfloat-neg-inf+ +vt-dfloat-nan+))))
(defun %op-log10-sf (x)
  (let ((x (coerce x 'single-float)))
    (if (> x 0.0s0)
        (log x 10.0d0)
        (if (zerop x) +vt-sfloat-neg-inf+ +vt-sfloat-nan+))))

;;; 立方根：保持 (expt |x| third) 的原算式，third 按输入精度取
;;; （single 用 1/3 的 single 表示，故结果与原来的 single 分支一致）。

(declaim (inline %op-cbrt-df %op-cbrt-sf))
(defun %op-cbrt-df (x)
  (let ((x (coerce x 'double-float)))
   (cond ((sb-ext:float-nan-p x) x)
        ((or (= x sb-ext:double-float-positive-infinity)
             (= x sb-ext:double-float-negative-infinity)) x)
         (t (* (signum x) (expt (abs x) (/ 3.0d0)))))))
(defun %op-cbrt-sf (x)
  (let ((x (coerce x 'single-float)))
   (cond ((sb-ext:float-nan-p x) x)
        ((or (= x sb-ext:single-float-positive-infinity)
             (= x sb-ext:single-float-negative-infinity)) x)
         (t (* (signum x) (expt (abs x) (/ 3.0s0)))))))


(defun vt-cos (vt &key out dtype)
  "逐元素余弦。整数输入输出 float64，float32 进 float32 出。
      :out 契约同 vt-+。"
  (when (%cx-any-p vt)
    (return-from vt-cos
      (let* ((in (ensure-vt vt))
             (dt (%cx-result-complex-dtype (vt-dtype in) dtype "VT-COS")))
        (vt-fast-map (function cos) in :out out :dtype dt))))
  (%float-map #'cos vt out (%infer-float-dtype vt dtype)))


(defun vt-tan (vt &key out dtype)
  "逐元素正切。整数输入输出 float64，float32 进 float32 出。
      :out 契约同 vt-+。"
  (when (%cx-any-p vt)
    (return-from vt-tan
      (let* ((in (ensure-vt vt))
             (dt (%cx-result-complex-dtype (vt-dtype in) dtype "VT-TAN")))
        (vt-fast-map (function tan) in :out out :dtype dt))))
  (%float-map #'tan vt out (%infer-float-dtype vt dtype)))


(defun vt-atan (vt &key out dtype)
  "逐元素反正切。整数输入输出 float64，float32 进 float32 出。
      :out 契约同 vt-+。"
  (when (%cx-any-p vt)
    (return-from vt-atan
      (let* ((in (ensure-vt vt))
             (dt (%cx-result-complex-dtype (vt-dtype in) dtype "VT-ATAN")))
        (vt-fast-map (function atan) in :out out :dtype dt))))
  (%float-map #'atan vt out (%infer-float-dtype vt dtype)))


(defun vt-sinh (vt &key out dtype)
  "逐元素双曲正弦。整数输入输出 float64，float32 进 float32 出。
      :out 契约同 vt-+。"
  (when (%cx-any-p vt)
    (return-from vt-sinh
      (let* ((in (ensure-vt vt))
             (dt (%cx-result-complex-dtype (vt-dtype in) dtype "VT-SINH")))
        (vt-fast-map (function sinh) in :out out :dtype dt))))
  (%float-map #'sinh vt out (%infer-float-dtype vt dtype)))


(defun vt-cosh (vt &key out dtype)
  "逐元素双曲余弦。整数输入输出 float64，float32 进 float32 出。
      :out 契约同 vt-+。"
  (when (%cx-any-p vt)
    (return-from vt-cosh
      (let* ((in (ensure-vt vt))
             (dt (%cx-result-complex-dtype (vt-dtype in) dtype "VT-COSH")))
        (vt-fast-map (function cosh) in :out out :dtype dt))))
  (%float-map #'cosh vt out (%infer-float-dtype vt dtype)))


(defun vt-tanh (vt &key out dtype)
  "逐元素双曲正切。整数输入输出 float64，float32 进 float32 出。
      :out 契约同 vt-+。"
  (when (%cx-any-p vt)
    (return-from vt-tanh
      (let* ((in (ensure-vt vt))
             (dt (%cx-result-complex-dtype (vt-dtype in) dtype "VT-TANH")))
        (vt-fast-map (function tanh) in :out out :dtype dt))))
  (%float-map #'tanh vt out (%infer-float-dtype vt dtype)))


(defun vt-asin (vt &key out dtype)
  "逐元素反正弦。|x|>1 返回 NaN（对标 numpy，不抛条件）。
      整数输入输出 float64，float32 进 float32 出。:out 契约同 vt-+。"
  (when (%cx-any-p vt)
    (return-from vt-asin
      (let* ((in (ensure-vt vt))
             (dt (%cx-result-complex-dtype (vt-dtype in) dtype "VT-ASIN")))
        (vt-fast-map (function asin) in :out out :dtype dt))))
(let ((dt (%infer-float-dtype vt dtype)))
    (%float-map-pair %op-asin-df %op-asin-sf vt out dt)))

(defun vt-acos (vt &key out dtype)
  "逐元素反余弦。|x|>1 返回 NaN（对标 numpy，不抛条件）。
      整数输入输出 float64，float32 进 float32 出。:out 契约同 vt-+。"
  (when (%cx-any-p vt)
    (return-from vt-acos
      (let* ((in (ensure-vt vt))
             (dt (%cx-result-complex-dtype (vt-dtype in) dtype "VT-ACOS")))
        (vt-fast-map (function acos) in :out out :dtype dt))))
(let ((dt (%infer-float-dtype vt dtype)))
    (%float-map-pair %op-acos-df %op-acos-sf vt out dt)))

(defun vt-asinh (vt &key out dtype)
  "逐元素反双曲正弦。整数输入输出 float64，float32 进 float32 出。
      :out 契约同 vt-+。"
  (when (%cx-any-p vt)
    (return-from vt-asinh
      (let* ((in (ensure-vt vt))
             (dt (%cx-result-complex-dtype (vt-dtype in) dtype "VT-ASINH")))
        (vt-fast-map (function asinh) in :out out :dtype dt))))
  (%float-map #'asinh vt out (%infer-float-dtype vt dtype)))


(defun vt-acosh (vt &key out dtype)
  "逐元素反双曲余弦。x<1 返回 NaN（对标 numpy）。
      整数输入输出 float64，float32 进 float32 出。:out 契约同 vt-+。"
  (when (%cx-any-p vt)
    (return-from vt-acosh
      (let* ((in (ensure-vt vt))
             (dt (%cx-result-complex-dtype (vt-dtype in) dtype "VT-ACOSH")))
        (vt-fast-map (function acosh) in :out out :dtype dt))))
(let ((dt (%infer-float-dtype vt dtype)))
    (%float-map-pair %op-acosh-df %op-acosh-sf vt out dt)))

(defun vt-atanh (vt &key out dtype)
  "逐元素反双曲正切，对标 numpy.arctanh。
      |x|>1 返回 NaN；x=+1 返回 +Inf、x=-1 返回 -Inf（IEEE 极限，非 NaN）。
      整数输入输出 float64，float32 进 float32 出。:out 契约同 vt-+。"
  (when (%cx-any-p vt)
    (return-from vt-atanh
      (let* ((in (ensure-vt vt))
             (dt (%cx-result-complex-dtype (vt-dtype in) dtype "VT-ATANH")))
        (vt-fast-map (function atanh) in :out out :dtype dt))))
(let ((dt (%infer-float-dtype vt dtype)))
    (%float-map-pair %op-atanh-df %op-atanh-sf vt out dt)))

(defun vt-exp (vt &key out dtype)
  "逐元素指数。整数输入输出 float64，float32 进 float32 出。
      :out 契约同 vt-+。"
  (when (%cx-any-p vt)
    (return-from vt-exp
      (let* ((in (ensure-vt vt))
             (dt (%cx-result-complex-dtype (vt-dtype in) dtype "VT-EXP")))
        (vt-fast-map (function exp) in :out out :dtype dt))))
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
  (when (%cx-any-p vt)
    (return-from vt-square
      (let* ((in (ensure-vt vt))
             (dt (%cx-result-complex-dtype (vt-dtype in) dtype "VT-SQUARE")))
        (vt-fast-map (function %cop-square) in :out out :dtype dt))))
  (vt-fast-map #'* vt vt :out out :dtype dtype))
  "逐元素平方。特化为 vt-fast-map #'* 以避免 vt-pow 的
      handler-case/expt/realp 开销。:out 契约同 vt-+。"

(defun vt-sqrt (vt &key out dtype)
  "逐元素平方根。负数输入返回 NaN（对标 numpy，不抛条件）。
      整数输入输出 float64，float32 进 float32 出。:out 契约同 vt-+。"
  (when (%cx-any-p vt)
    (return-from vt-sqrt
      (let* ((in (ensure-vt vt))
             (dt (%cx-result-complex-dtype (vt-dtype in) dtype "VT-SQRT")))
        (vt-fast-map (function %cop-sqrt) in :out out :dtype dt))))
(let ((dt (%infer-float-dtype vt dtype)))
    (%float-map-pair %op-sqrt-df %op-sqrt-sf vt out dt)))

(defun vt-log (vt &key base out dtype)
  "逐元素自然对数；:base 给定时为换底对数。
      语义对标 numpy.log：x>0 正常；x=0 → -Inf（或 log(base)<0 时 +Inf）；
      x<0 → NaN；非法 base（<=0 或 =1）→ 全 NaN。
      整数输入输出 float64，float32 进 float32 出。:out 契约同 vt-+。

  性能：base 缺省 / base=2.0d0 / base=10.0d0 三种最常见情形走
  `%float-map-pair` 的编译期内联循环（原实现对每种情形都用 lambda，
  一律回退 vt-map 的 funcall 路径，1M 元素实测 log 57ms / log2 103ms）；
  任意其它 base 仍是 lambda 路径（base 是运行期变量，无法内联）。"
  (let ((dt (%infer-float-dtype vt dtype)))
    (cond
      ((null base)
       (%float-map-pair %op-log-df %op-log-sf vt out dt))
      ((eql base 2.0d0)
       (%float-map-pair %op-log2-df %op-log2-sf vt out dt))
      ((eql base 10.0d0)
       (%float-map-pair %op-log10-df %op-log10-sf vt out dt))
      ((or (<= base 0) (= base 1))
       (let ((nan (vt-get-nan dt)))
         (vt-map (lambda (x) (declare (ignore x)) nan)
                 (%coerce-float-input (ensure-vt vt) dt) :out out :dtype dt)))
      (t
       (let* ((nan (vt-get-nan dt))
              (neginf (vt-get-neg-inf dt))
              (posinf (vt-get-pos-inf dt))
              (zero-result (if (plusp (log base)) neginf posinf)))
         (vt-map (lambda (x)
                   (if (> x 0)
                       (log x base)
                       (if (zerop x) zero-result nan)))
                 (%coerce-float-input (ensure-vt vt) dt) :out out :dtype dt))))))

(defun vt-log10 (vt &key out dtype)
  "逐元素常用对数（底 10）。语义等价 (vt-log vt :base 10)。
      :out 契约同 vt-+。"
  (when (%cx-any-p vt)
    (return-from vt-log10
      (let* ((in (ensure-vt vt))
             (dt (%cx-result-complex-dtype (vt-dtype in) dtype "VT-LOG10")))
        (vt-fast-map (function %cop-log10) in :out out :dtype dt))))
  (%float-map-pair %op-log10-df %op-log10-sf vt out
                   (%infer-float-dtype vt dtype)))

(defun vt-log2 (vt &key out dtype)
  "逐元素二进制对数（底 2）。语义等价 (vt-log vt :base 2)。
      :out 契约同 vt-+。"
  (when (%cx-any-p vt)
    (return-from vt-log2
      (let* ((in (ensure-vt vt))
             (dt (%cx-result-complex-dtype (vt-dtype in) dtype "VT-LOG2")))
        (vt-fast-map (function %cop-log2) in :out out :dtype dt))))
  (%float-map-pair %op-log2-df %op-log2-sf vt out
                   (%infer-float-dtype vt dtype)))

(defun vt-abs (vt &key out dtype)
  "逐元素绝对值。整数保持整数 dtype（对标 numpy.abs）。
      :out 契约同 vt-+。"
  (vt-fast-map #'abs vt :out out :dtype dtype))

(declaim (inline %sign1))
(defun %sign1 (x)
  "符号函数：-1/0/+1，NaN 传播，浮点零一律返回 +0.0（对标 numpy.sign）。
   浮点分支用 (float 1 x) 而非字面 1.0d0 —— 后者写进 single-float 数组
   会触发 specialized-array 类型错误。"
  (cond ((%nan-p x) x)
        ((zerop x)    (if (floatp x) (float 0.0 x) 0))
        ((plusp x)    (if (floatp x) (float 1 x) 1))
        (t            (if (floatp x) (float -1 x) -1))))

(defun vt-signum (vt &key out dtype)
  "逐元素符号函数（-1 / 0 / +1）。整数保持整数 dtype，
      NaN 原样传播。:out 契约同 vt-+。
      v0.4.x 修复：signum(-0.0) 原 CL 语义返回 -0.0，现对浮点零
      统一返回 +0.0（对标 numpy.sign：sign(-0.0)=0.0）。
      v0.4.3：改用内联字面算子 %sign1（原匿名 lambda 让 vt-map 退回
      funcall + 泛型 aref，1M 元素 69ms）。"
  (when (%cx-any-p vt)
    (return-from vt-signum
      (let* ((in (ensure-vt vt))
             (dt (%cx-result-complex-dtype (vt-dtype in) dtype "VT-SIGNUM")))
        (vt-fast-map (function %cop-signum) in :out out :dtype dt))))
  (vt-fast-map #'%sign1 vt :out out :dtype dtype))

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
    (when (%cx-reject vt)
    (error "vt-even-p: 不支持复数输入（numpy 同样拒绝）。见 CONVENTIONS §16.11"))
(vt-map (lambda (v)
            (cond ((or (%nan-p v) (%inf-p v)) 0)
                  ((and (floatp v) (/= v (floor v))) 0)
                  (t (if (evenp (floor v)) 1 0))))
          vt :out out :dtype dtype))

(defun vt-odd-p (vt &key out (dtype :int8))
  "逐元素奇数判定，返回值 1/0，dtype 默认 :int8（承载布尔语义）。
      NaN/±Inf 视为非奇数 → 0；非整数值视为非奇数 → 0
      （对标模拟基准 v%2==1）。:out 契约同 vt-=。"
    (when (%cx-reject vt)
    (error "vt-odd-p: 不支持复数输入（numpy 同样拒绝）。见 CONVENTIONS §16.11"))
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
    (when (%cx-reject vt divisor)
    (error "vt-mod: 不支持复数输入（numpy 同样拒绝）。见 CONVENTIONS §16.11"))
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
    (when (%cx-reject vt divisor)
    (error "vt-rem: 不支持复数输入（numpy 同样拒绝）。见 CONVENTIONS §16.11"))
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
    (when (%cx-reject vty vtx)
    (error "vt-atan2: 不支持复数输入（numpy 同样拒绝）。见 CONVENTIONS §16.11"))
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

;;; ------------------------------------------------------------------
;;; 取整族的 DIVISOR=1 特化（v0.4.3 性能修复）
;;;
;;; 原实现四个函数统一写成
;;;     (vt-map (lambda (x) (%floor-family-body x op)) vt ...)
;;; 闭包会让 vt-fast-map 的算子内联**彻底失效** —— 该宏只认 #'symbol
;;; 形式字面算子，遇到 (lambda ...) 直接 `(apply #'vt-map ...)` 回退，
;;; 于是循环里既没有数组元素类型声明、又是逐元素 funcall，还要为每个
;;; 元素求值一次 (%nan-or-inf-p ...)。实测 1M 元素 69ms，而同类的一元
;;; 逐元素 vt-abs（用 #'abs 走内联）只要 1.7ms —— 差 40 倍；
;;; numpy.floor 是 0.19ms，差 370 倍。
;;;
;;; 改法：把 DIVISOR=1（默认值，也是 numpy 唯一存在的形态）固化成
;;; 单参**命名**函数，交给 vt-fast-map 内联进元素类型循环；DIVISOR≠1
;;; 是本库扩展（numpy 无对应物），仍走原闭包路径，语义逐位不变。
;;; ------------------------------------------------------------------
(defmacro %floor-family-body1 (x op)
  "DIVISOR 恒为 1 时 %floor-family-body 的等价展开（供内联快路径使用）。"
  `(if (and (floatp ,x) (%nan-or-inf-p ,x))
       ,x
       (let ((res (nth-value 0 (,op ,x))))
         (if (floatp ,x)
             (if (and (zerop res) (minusp (float-sign ,x)))
                 (- (float 0.0 ,x))
                 (float res ,x))
             res))))

(declaim (inline %floor1 %ceiling1 %round1 %truncate1))
(defun %floor1 (x)    (%floor-family-body1 x floor))
(defun %ceiling1 (x)  (%floor-family-body1 x ceiling))
(defun %round1 (x)    (%floor-family-body1 x round))
(defun %truncate1 (x) (%floor-family-body1 x truncate))

(defun vt-floor (vt &key (divisor 1) out dtype)
  "逐元素向下取整（可除 DIVISOR）。
      NaN/±Inf 原样传播（对标 numpy.floor）；整数输入保持整数 dtype。
      :out 契约同 vt-+。"
    (when (%cx-reject vt)
    (error "vt-floor: 不支持复数输入（numpy 同样拒绝）。见 CONVENTIONS §16.11"))
(if (eql divisor 1)
      (vt-fast-map #'%floor1 vt :out out :dtype dtype)
      (vt-map (lambda (x) (%floor-family-body x floor))
              vt :out out :dtype dtype)))

(defun vt-ceiling (vt &key (divisor 1) out dtype)
  "逐元素向上取整（可除 DIVISOR）。
      NaN/±Inf 原样传播（对标 numpy.ceil）；整数输入保持整数 dtype。
      :out 契约同 vt-+。"
    (when (%cx-reject vt)
    (error "vt-ceiling: 不支持复数输入（numpy 同样拒绝）。见 CONVENTIONS §16.11"))
(if (eql divisor 1)
      (vt-fast-map #'%ceiling1 vt :out out :dtype dtype)
      (vt-map (lambda (x) (%floor-family-body x ceiling))
              vt :out out :dtype dtype)))

(defun vt-round (vt &key (divisor 1) out dtype)
  "逐元素四舍五入取整（可除 DIVISOR）。
      NaN/±Inf 原样传播（对标 numpy.round）；整数输入保持整数 dtype。
      注意：舍入策略为「最近偶数」，与 CL round 一致。"
  (if (eql divisor 1)
      (vt-fast-map #'%round1 vt :out out :dtype dtype)
      (vt-map (lambda (x) (%floor-family-body x round))
              vt :out out :dtype dtype)))

(defun vt-truncate (vt &key (divisor 1) out dtype)
  "逐元素向零取整（可除 DIVISOR）。
      NaN/±Inf 原样传播（对标 numpy.trunc）；整数输入保持整数 dtype。
      :out 契约同 vt-+。"
    (when (%cx-reject vt)
    (error "vt-truncate: 不支持复数输入（numpy 同样拒绝）。见 CONVENTIONS §16.11"))
(if (eql divisor 1)
      (vt-fast-map #'%truncate1 vt :out out :dtype dtype)
      (vt-map (lambda (x) (%floor-family-body x truncate))
              vt :out out :dtype dtype)))

(defun vt-rint (vt &key out dtype)
  "逐元素取整到最近偶数（对标 numpy.rint）。
      NaN/±Inf 原样传播；整数输入保持整数 dtype。:out 契约同 vt-+。"
    (when (%cx-reject vt)
    (error "vt-rint: 不支持复数输入（numpy 同样拒绝）。见 CONVENTIONS §16.11"))
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
  (when (%cx-any-p t1 t2)
    (return-from vt-=
      (vt-fast-map (function %cx-eq-flag) (ensure-vt t1) (ensure-vt t2)
                   :out out :dtype dtype)))
  (vt-fast-map #'%op-eq (ensure-vt t1) (ensure-vt t2)
               :dtype dtype :out out))

(defun vt-/= (t1 t2 &key (dtype :int8) out)
  "逐元素不等判定，返回值 1/0，dtype 默认 :int8（承载布尔语义）。
      NaN 与任何值比较恒真 → 1（含 NaN != NaN）。:out 契约同 vt-=。"
  (when (%cx-any-p t1 t2)
    (return-from vt-/=
      (vt-fast-map (function %cx-ne-flag) (ensure-vt t1) (ensure-vt t2)
                   :out out :dtype dtype)))
  (vt-fast-map #'%op-ne (ensure-vt t1) (ensure-vt t2)
               :dtype dtype :out out))

(defun vt-< (t1 t2 &key (dtype :int8) out)
  "逐元素小于判定，返回值 1/0，dtype 默认 :int8（承载布尔语义）。
      NaN 参与的比较恒假 → 0。:out 契约同 vt-=。"
  (when (%cx-any-p t1 t2)
    (return-from vt-<
      (vt-fast-map (function %cx-lt-flag) (ensure-vt t1) (ensure-vt t2)
                   :out out :dtype dtype)))
  (vt-fast-map #'%op-lt (ensure-vt t1) (ensure-vt t2)
               :dtype dtype :out out))

(defun vt-<= (t1 t2 &key (dtype :int8) out)
  "逐元素小于等于判定，返回值 1/0，dtype 默认 :int8（承载布尔语义）。
      NaN 参与的比较恒假 → 0。:out 契约同 vt-=。"
  (when (%cx-any-p t1 t2)
    (return-from vt-<=
      (vt-fast-map (function %cx-le-flag) (ensure-vt t1) (ensure-vt t2)
                   :out out :dtype dtype)))
  (vt-fast-map #'%op-le (ensure-vt t1) (ensure-vt t2)
               :dtype dtype :out out))

(defun vt-> (t1 t2 &key (dtype :int8) out)
  "逐元素大于判定，返回值 1/0，dtype 默认 :int8（承载布尔语义）。
      NaN 参与的比较恒假 → 0。:out 契约同 vt-=。"
  (when (%cx-any-p t1 t2)
    (return-from vt->
      (vt-fast-map (function %cx-gt-flag) (ensure-vt t1) (ensure-vt t2)
                   :out out :dtype dtype)))
  (vt-fast-map #'%op-gt (ensure-vt t1) (ensure-vt t2)
               :dtype dtype :out out))

(defun vt->= (t1 t2 &key (dtype :int8) out)
  "逐元素大于等于判定，返回值 1/0，dtype 默认 :int8（承载布尔语义）。
      NaN 参与的比较恒假 → 0。:out 契约同 vt-=。"
  (when (%cx-any-p t1 t2)
    (return-from vt->=
      (vt-fast-map (function %cx-ge-flag) (ensure-vt t1) (ensure-vt t2)
                   :out out :dtype dtype)))
  (vt-fast-map #'%op-ge (ensure-vt t1) (ensure-vt t2)
               :dtype dtype :out out))

(defun vt-rad2deg (vt &key out dtype)
  "弧度转角度（×180/π）。整数输入输出 float64，
      float32 进 float32 出。:out 契约同 vt-+。"
    (when (%cx-reject vt)
    (error "vt-rad2deg: 不支持复数输入（numpy 同样拒绝）。见 CONVENTIONS §16.11"))
(let* ((dt (%infer-float-dtype vt dtype))
         (factor (if (eq dt :float32) (/ 180.0s0 (coerce pi 'single-float))
                     (/ 180.0d0 pi))))
    (vt-map (lambda (x) (* x factor)) vt :out out :dtype dt)))

(defun vt-deg2rad (vt &key out dtype)
  "角度转弧度（×π/180）。整数输入输出 float64，
      float32 进 float32 出。:out 契约同 vt-+。"
    (when (%cx-reject vt)
    (error "vt-deg2rad: 不支持复数输入（numpy 同样拒绝）。见 CONVENTIONS §16.11"))
(let* ((dt (%infer-float-dtype vt dtype))
         (factor (if (eq dt :float32) (/ (coerce pi 'single-float) 180.0s0)
                     (/ pi 180.0d0))))
    (vt-map (lambda (x) (* x factor)) vt :out out :dtype dt)))

;;; ------------------------------------------------------------------
;;; 极值 / 逻辑算子（具名 + 内联）
;;;
;;; 类型声明（性能）：这些算子此前全部以 `(vt-map (lambda (a b) ...))` 形式
;;; 传入，`vt-map` 只能在运行时经 funcall 调用闭包，循环体拿不到任何类型
;;; 信息 —— aref 走泛型分派、结果装箱、写回再做一次泛型 cast。
;;; 实测 1M float64 的 vt-maximum 需 22ms（numpy 0.24ms，慢 93 倍）。
;;; 改成「具名函数 + inline 声明」后，`vt-fast-map` 能在编译期把算子
;;; 内联进 %inline{1,2,3}-loop 的特化循环，读写两端都是特化 aref。
;;; 语义与原来的 lambda 逐字相同（同样的 NaN 分支顺序）。
;;; ------------------------------------------------------------------
(declaim (inline %op-maximum %op-minimum %op-fmax %op-fmin
                 %op-logical-and %op-logical-or %op-logical-not
                 %op-logical-xor))

(defun %op-maximum (a b)
  "NaN 传播取大（对标 numpy.maximum）：任一为 NaN 即取该侧（左优先）。"
  (cond ((%nan-p a) a)
        ((%nan-p b) b)
        (t (max a b))))

(defun %op-minimum (a b)
  "NaN 传播取小（对标 numpy.minimum）。"
  (cond ((%nan-p a) a)
        ((%nan-p b) b)
        (t (min a b))))

(defun %op-fmax (a b)
  "忽略 NaN 取大（对标 numpy.fmax）：仅两侧皆 NaN 时返回 NaN。"
  (cond ((and (%nan-p a) (%nan-p b)) a)
        ((%nan-p a) b)
        ((%nan-p b) a)
        (t (max a b))))

(defun %op-fmin (a b)
  "忽略 NaN 取小（对标 numpy.fmin）。"
  (cond ((and (%nan-p a) (%nan-p b)) a)
        ((%nan-p a) b)
        ((%nan-p b) a)
        (t (min a b))))

(defun %op-logical-and (a b) (if (and (not (zerop a)) (not (zerop b))) 1 0))

(defun %op-logical-or (a b) (if (or (not (zerop a)) (not (zerop b))) 1 0))

(defun %op-logical-not (v) (if (zerop v) 1 0))

(defun %op-logical-xor (a b) (if (not (eq (not (zerop a)) (not (zerop b)))) 1 0))

(defun vt-maximum (t1 t2 &key out dtype)
  "逐元素取较大值，**NaN 传播**（任一为 NaN → NaN，且优先取左）
      对标 numpy.maximum。:out 契约同 vt-+。"
  (when (%cx-any-p t1 t2)
    (return-from vt-maximum
      (let* ((a (ensure-vt t1)) (b (ensure-vt t2))
             (p (%op-final-dtype (list (%scalar-kind t1) (%scalar-kind t2))
                                 (list (vt-dtype a) (vt-dtype b))))
             (dt (or dtype p)))
        (unless (%cx-complex-dtype-p* dt)
          (error "vt-maximum: 复数输入的结果 dtype 必须是复数类型，收到 ~a"
                 dt))
        (vt-map (function %cop-max) a b :out out :dtype dt))))
  (vt-fast-map #'%op-maximum t1 t2 :out out :dtype dtype))

(defun vt-minimum (t1 t2 &key out dtype)
  "逐元素取较小值，**NaN 传播**（任一为 NaN → NaN，且优先取左）
      对标 numpy.minimum。:out 契约同 vt-+。"
  (when (%cx-any-p t1 t2)
    (return-from vt-minimum
      (let* ((a (ensure-vt t1)) (b (ensure-vt t2))
             (p (%op-final-dtype (list (%scalar-kind t1) (%scalar-kind t2))
                                 (list (vt-dtype a) (vt-dtype b))))
             (dt (or dtype p)))
        (unless (%cx-complex-dtype-p* dt)
          (error "vt-minimum: 复数输入的结果 dtype 必须是复数类型，收到 ~a"
                 dt))
        (vt-map (function %cop-min) a b :out out :dtype dt))))
  (vt-fast-map #'%op-minimum t1 t2 :out out :dtype dtype))

(defun vt-fmax (t1 t2 &key out dtype)
  "逐元素取较大值，**忽略 NaN**（仅两侧皆 NaN 时返回 NaN）
      对标 numpy.fmax。:out 契约同 vt-+。"
  (when (%cx-any-p t1 t2)
    (return-from vt-fmax
      (let* ((a (ensure-vt t1)) (b (ensure-vt t2))
             (p (%op-final-dtype (list (%scalar-kind t1) (%scalar-kind t2))
                                 (list (vt-dtype a) (vt-dtype b))))
             (dt (or dtype p)))
        (unless (%cx-complex-dtype-p* dt)
          (error "vt-fmax: 复数输入的结果 dtype 必须是复数类型，收到 ~a"
                 dt))
        (vt-map (function %cop-fmax) a b :out out :dtype dt))))
  (vt-fast-map #'%op-fmax t1 t2 :out out :dtype dtype))

(defun vt-fmin (t1 t2 &key out dtype)
  "逐元素取较小值，**忽略 NaN**（仅两侧皆 NaN 时返回 NaN）
      对标 numpy.fmin。:out 契约同 vt-+。"
  (when (%cx-any-p t1 t2)
    (return-from vt-fmin
      (let* ((a (ensure-vt t1)) (b (ensure-vt t2))
             (p (%op-final-dtype (list (%scalar-kind t1) (%scalar-kind t2))
                                 (list (vt-dtype a) (vt-dtype b))))
             (dt (or dtype p)))
        (unless (%cx-complex-dtype-p* dt)
          (error "vt-fmin: 复数输入的结果 dtype 必须是复数类型，收到 ~a"
                 dt))
        (vt-map (function %cop-fmin) a b :out out :dtype dt))))
  (vt-fast-map #'%op-fmin t1 t2 :out out :dtype dtype))

(defun vt-logical-and (t1 t2 &key out (dtype :int8))
  "逐元素逻辑与（非零视为真），返回 1/0。
   dtype 默认 :int8——与 §3.1「比较/逻辑运算返回 :int8」一致，
   语义等价 numpy 的 bool（本库不引入独立 :bool dtype）。"
  (vt-fast-map #'%op-logical-and t1 t2 :dtype dtype :out out))

(defun vt-logical-or (t1 t2 &key out (dtype :int8))
  "逐元素逻辑或（非零视为真），返回 1/0。dtype 默认 :int8（同 §3.1）。"
  (vt-fast-map #'%op-logical-or t1 t2 :dtype dtype :out out))

(defun vt-logical-not (vt &key out (dtype :int8))
  "逐元素逻辑非，返回 1/0。dtype 默认 :int8（同 §3.1）。"
  (vt-fast-map #'%op-logical-not vt :dtype dtype :out out))

(defun vt-logical-xor (t1 t2 &key out (dtype :int8))
  "逐元素逻辑异或，返回 1/0。dtype 默认 :int8（同 §3.1）。"
  (vt-fast-map #'%op-logical-xor t1 t2 :dtype dtype :out out))

(defun vt-bit-and (t1 t2 &key out dtype)
  "逐元素按位与 logand。输入应为整数 dtype；:out 契约同 vt-+。"
    (when (%cx-reject t1 t2)
    (error "vt-bit-and: 不支持复数输入（numpy 同样拒绝）。见 CONVENTIONS §16.11"))
(vt-map #'logand (ensure-vt t1) (ensure-vt t2) :dtype dtype :out out))

(defun vt-bit-ior (t1 t2 &key out dtype)
  "逐元素按位或 logior。输入应为整数 dtype；:out 契约同 vt-+。"
    (when (%cx-reject t1 t2)
    (error "vt-bit-ior: 不支持复数输入（numpy 同样拒绝）。见 CONVENTIONS §16.11"))
(vt-map #'logior (ensure-vt t1) (ensure-vt t2) :dtype dtype :out out))

(defun vt-bit-xor (t1 t2 &key out dtype)
  "逐元素按位异或 logxor。输入应为整数 dtype；:out 契约同 vt-+。"
    (when (%cx-reject t1 t2)
    (error "vt-bit-xor: 不支持复数输入（numpy 同样拒绝）。见 CONVENTIONS §16.11"))
(vt-map #'logxor (ensure-vt t1) (ensure-vt t2) :dtype dtype :out out))

(defun vt-bit-not (vt &key out dtype)
  "逐元素按位取反 lognot。输入应为整数 dtype；:out 契约同 vt-+。"
    (when (%cx-reject vt)
    (error "vt-bit-not: 不支持复数输入（numpy 同样拒绝）。见 CONVENTIONS §16.11"))
(vt-map #'lognot vt :dtype dtype :out out))

(defun vt-left-shift (vt shift &key out dtype)
  "逐元素左移 SHIFT 位（ash x shift）。SHIFT 为整数标量。
      输入应为整数 dtype；:out 契约同 vt-+。"
    (when (%cx-reject vt shift)
    (error "vt-left-shift: 不支持复数输入（numpy 同样拒绝）。见 CONVENTIONS §16.11"))
(vt-map (lambda (x) (ash x shift))
          vt :dtype dtype :out out))

(defun vt-right-shift (vt shift &key out dtype)
  "逐元素右移 SHIFT 位（ash x -shift，算术右移）。SHIFT 为整数标量。
      输入应为整数 dtype；:out 契约同 vt-+。"
    (when (%cx-reject vt shift)
    (error "vt-right-shift: 不支持复数输入（numpy 同样拒绝）。见 CONVENTIONS §16.11"))
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
      类型错误，现显式拦截：NaN → NaN、±Inf → ±Inf（numpy 对齐）。
  性能：改用符号算子 %OP-CBRT-Df/SF，编译期内联（原 lambda 走 vt-map
  的 funcall 路径，1M 元素实测 143ms）。"
    (when (%cx-reject vt)
    (error "vt-cbrt: 不支持复数输入（numpy 同样拒绝）。见 CONVENTIONS §16.11"))
(let ((dt (%infer-float-dtype vt dtype)))
    (%float-map-pair %op-cbrt-df %op-cbrt-sf vt out dt)))

;;; v0.4.3 性能修复：vt-hypot / vt-lerp / vt-cbrt 等原本一律写成
;;;     (vt-map (lambda (a b) ...) ...)
;;; 闭包让 vt-fast-map 的算子内联失效（该宏只认 #'symbol 形式），循环里
;;; 既无数组元素类型声明、又是逐元素 funcall —— hypot 1M 元素实测 100ms
;;; （numpy 5.9ms，差 17 倍）。改成命名算子 + 按 dtype 分派即可内联。
(declaim (inline %hypot-df %hypot-sf))
(defun %hypot-df (a b)
  "hypot 的 double-float 内核。"
  (let ((abs-a (abs a)) (abs-b (abs b)))
    (cond ((or (%inf-p abs-a) (%inf-p abs-b))
           ;; NumPy hypot：任一参数 ±Inf → +Inf（即便另一参数为 NaN）
           (vt-get-pos-inf :float64))
          ((%nan-p abs-a) abs-a)
          ((%nan-p abs-b) abs-b)
          ((zerop abs-a) abs-b)
          ((zerop abs-b) abs-a)
          (t (let* ((mx (max abs-a abs-b))
                    (mn (min abs-a abs-b))
                    (r  (/ mn mx)))
               (* mx (sqrt (+ 1.0d0 (* r r)))))))))

(defun %hypot-sf (a b)
  "hypot 的 single-float 内核（常量与结果都保持 single，避免写入
   specialized-array 时的类型错误）。"
  (let ((abs-a (abs a)) (abs-b (abs b)))
    (cond ((or (%inf-p abs-a) (%inf-p abs-b))
           (vt-get-pos-inf :float32))
          ((%nan-p abs-a) abs-a)
          ((%nan-p abs-b) abs-b)
          ((zerop abs-a) abs-b)
          ((zerop abs-b) abs-a)
          (t (let* ((mx (max abs-a abs-b))
                    (mn (min abs-a abs-b))
                    (r  (/ mn mx)))
               (* mx (sqrt (+ 1.0s0 (* r r)))))))))

(defun vt-hypot (t1 t2 &key out dtype)
  "逐元素直角三角斜边 sqrt(x² + y²)（对标 numpy.hypot）。任一输入为 ±Inf → 结果 +Inf；NaN 传播。"
    (when (%cx-reject t1 t2)
    (error "vt-hypot: 不支持复数输入（numpy 同样拒绝）。见 CONVENTIONS §16.11"))
(let* ((v1 (ensure-vt t1))
         (v2 (ensure-vt t2))
         (dt (or dtype (if (or (eq (vt-dtype v1) :float32)
                               (eq (vt-dtype v2) :float32))
                           :float32 :float64))))
    (if (eq dt :float32)
        (vt-fast-map #'%hypot-sf v1 v2 :out out :dtype dt)
        (vt-fast-map #'%hypot-df v1 v2 :out out :dtype dt))))

(defun vt-reciprocal (vt &key out dtype)
  "逐元素倒数 (1/x)。

   dtype 语义（对标 numpy.reciprocal，v0.3.6 起）：
     整数输入 → **保持整数 dtype**（numpy 1//x 整数除法，1/2 → 0）；
     浮点输入 → 保持原浮点 dtype（float32 进 float32 出）；
     显式 :dtype 优先。
   零除：浮点 → ±Inf（IEEE 754）；整数 → 确定性报错，与 numpy
     整数零除行为一致。"
  (when (%cx-any-p vt)
    (return-from vt-reciprocal
      (let* ((in (ensure-vt vt))
             (dt (%cx-result-complex-dtype (vt-dtype in) dtype "VT-RECIPROCAL")))
        (vt-fast-map (function %cop-recip) in :out out :dtype dt))))
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
  (when (%cx-any-p vt)
    (return-from vt-negative
      (let* ((in (ensure-vt vt))
             (dt (%cx-result-complex-dtype (vt-dtype in) dtype "VT-NEGATIVE")))
        (vt-fast-map (function %cop-neg) in :out out :dtype dt))))
  (vt-fast-map #'- vt :dtype dtype :out out))

(defun vt-sinc (tensor &key out dtype)
  "逐元素归一化 sinc 函数 sin(πx)/(πx)，且 sinc(0)=1（对标 numpy.sinc）。
      整数输入输出 float64，float32 进 float32 出。:out 契约同 vt-+。"
  (when (%cx-any-p tensor)
    (return-from vt-sinc
      (let* ((in (ensure-vt tensor))
             (dt (%cx-result-complex-dtype (vt-dtype in) dtype "VT-SINC")))
        (vt-fast-map (function %cop-sinc) in :out out :dtype dt))))
  (let* ((dt (%infer-float-dtype tensor dtype))
         (x-pi (vt-scale tensor pi :dtype dt)))
    (vt-map (lambda (x)
              (if (zerop x) 1.0d0 (/ (sin x) x)))
            x-pi :out out :dtype dt)))

;; 在编译期间垃圾回收内存，防止爆炸
#+sbcl (sb-ext::gc :full t)
