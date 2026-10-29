;;;; elementwise.lisp — 逐元素算术 / 数学 / 比较 / 逻辑 / 位运算

#+sbcl (sb-ext::gc :full t) ;; 在编译期间垃圾回收内存，防止爆炸

(in-package :clvt)

;;; ------------------------------------------------------------------
;;; 布尔算术改派（CONVENTIONS §17.3）
;;;
;;; numpy 的 `bool_` 参与算术时**不按整数算**：
;;;   True + True  → True   （逻辑或，不是 2）
;;;   True * True  → True   （逻辑与）
;;;   True - True  → TypeError（布尔减法未定义）
;;; 若按整数算，`(vt-+ mask mask)` 会把 2 写进 `:bool` 数组，而
;;; `(safety 0)` 的写回路径不会拦下越界值 —— 「布尔数组只含 0/1」
;;; 的不变量就此破掉（后续 `vt-where` 拿到的掩码、npy 落盘的字节都会失真）。
;;;
;;; 这两个内核与 elementwise2 的 `%op-logical-or` / `%op-logical-and` 语义
;;; 相同，但必须定义在本文件：elementwise2 在本文件**之后**编译，
;;; 这里调不到它（会触发 undefined-function 编译告警）。
;;; ------------------------------------------------------------------
(declaim (inline %bool-or-op %bool-and-op))
(defun %bool-or-op (a b) (if (or (/= a 0) (/= b 0)) 1 0))
(defun %bool-and-op (a b) (if (and (/= a 0) (/= b 0)) 1 0))

(defmacro %bool-fold (fn tensors out)
  "把一串 `:bool` 张量按 FN（逻辑与/或内核）折叠成一个布尔张量。
   一元时返回自身的副本（`or(b,b) = b`）；OUT 只作用在最后一步，
   避免中间结果多分配一次。

   FN 必须是**字面量函数名**而不是函数值：`vt-fast-map` 只在
   `(function <symbol>)` 形式下才走编译期内联，传函数值会退化成
   `apply #'vt-map`（200+ 展开点的性能约定，见 map-reduce.lisp 的说明）。"
  (let ((acc (gensym "ACC"))
        (cells (gensym "CELLS")))
    `(let ((,acc (first ,tensors))
           (,cells (rest ,tensors)))
       (if (null ,cells)
           (vt-fast-map (function ,fn) ,acc ,acc :dtype :bool :out ,out)
           (progn
             (loop while ,cells
                   do (setf ,acc (vt-fast-map (function ,fn) ,acc (pop ,cells)
                                              :dtype :bool
                                              :out (when (null ,cells) ,out))))
             ,acc)))))

(defun %bool-arith-p (dtype raws)
  "是否该把 RAW 这批操作数按布尔语义处理：结果 dtype 落在 `:bool`
   （未指定时，全布尔输入按提升规则必然得 `:bool`）且操作数全为布尔张量。"
  (and (or (null dtype) (eq dtype :bool))
       (%all-bool-tensors-p raws)))

(defun vt-+ (&rest args)
  "逐元素加法（一元恒等 / 二元 / N 元，支持广播）。

   参数契约：VARGS = 任意个张量/标量 + 可选 :dtype / :out。
     :dtype  结果 dtype；缺省按 numpy 类型提升规则由全部输入决定。
     :out    输出张量。**严格相等契约**：dtype 必须精确等于结果 dtype、
             形状必须精确等于广播结果形状、必须可写。
   输入个数 ≤ 3 走编译期内联快路径，> 3 回退 vt-map（语义基准）。

   布尔输入（CONVENTIONS §17.3）：全 `:bool` 操作数按 numpy 的 `bool_`
   语义算**逻辑或**（`True + True = True`），不按整数相加 —— 后者会
   把 2 写进布尔数组。显式 `:dtype :int8` 等数值目标仍走整数路径。"
  (with-float-safe
    (multiple-value-bind (tensors dtype out) (parse-vt-op-args args)
      (when (%bool-arith-p dtype tensors)
        (return-from vt-+ (%bool-fold %bool-or-op tensors out)))
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
   :out 必须精确匹配结果形状与 dtype 且可写。

   布尔输入（CONVENTIONS §17.3）：全 `:bool` 操作数按 numpy 的 `bool_`
   语义算**逻辑与**（`True * True = True`），与整数相乘逐位巧合相同，
   但显式走逻辑内核才能保证结果只取 0/1。"
  (with-float-safe
    (multiple-value-bind (tensors dtype out) (parse-vt-op-args args)
      (when (%bool-arith-p dtype tensors)
        (return-from vt-* (%bool-fold %bool-and-op tensors out)))
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
   :out 必须精确匹配结果形状与 dtype 且可写。

   布尔输入（CONVENTIONS §17.3）：**报错**，与 numpy 一致
   （`np.array([True]) - np.array([True])` 抛 TypeError：布尔减法未定义，
   否定请用 `vt-logical-not`，异或请用 `vt-logical-xor`）。
   按整数相减会得到 0/-1，写进 `:bool` 数组即在 0/1 之外又造出 255。
   显式 `:dtype :int8` 等数值目标仍走整数路径（向后兼容）。"
  (with-float-safe
    (multiple-value-bind (tensors dtype out) (parse-vt-op-args args)
      (when (%bool-arith-p dtype (cons vt tensors))
        (error "vt--: 布尔张量不支持减法/取负（numpy 同样拒绝：布尔减法未定义）。~%  否定请用 vt-logical-not，异或请用 vt-logical-xor；~%  若确实想要整数减法，请显式给出 :dtype :int8 等数值类型。见 CONVENTIONS §17.3"))
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
                     (member (vt-dtype x) '(:uint8 :uint16 :bool)))))
             ;; 弱标量（NEP 50）：Lisp 标量只贡献「种类」，不参与宽度判定。
             ;; 于是 (vt-/ f32-数组 2) 保持 float32（numpy 同），而
             ;; (vt-/ int-数组 2) 仍是 float64（数组本身是整型）。
             (tensor-dts (loop for raw in (cons vt tensors) for x in all
                               unless (%scalar-kind raw) collect (vt-dtype x)))
             ;; dtype 版谓词（上面那个 INT-FAMILY-P 吃的是张量对象）
             ;; :bool 也归入此族：numpy 的 true_divide 对布尔输入恒给 float64
             ;; （`np.array([True,False])/np.array([True,False])` → [1., nan]），
             ;; 若留整数语义则 `0/0` 直接撞 CL 的 division-by-zero。§17.3
             (dt-int-p (lambda (dt)
                         (or (vt-int-dtype-p dt)
                             (member dt '(:uint8 :uint16 :bool)))))
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
      :dtype 决定结果 dtype（缺省按提升规则），:out 契约同 vt-+。
      布尔操作数按逻辑或处理（CONVENTIONS §17.3，同 vt-+）。"
  (when (%bool-arith-p dtype (list a b))
    (return-from vt-add
      (vt-fast-map #'%bool-or-op a b :dtype :bool :out out)))
  (vt-fast-map #'+ a b :dtype dtype :out out))

(defun vt-sub (a b &key dtype out)
  "逐元素减法（二元特化入口），等价于 (vt-- a b)。
      :dtype 决定结果 dtype（缺省按提升规则），:out 契约同 vt-+。
      布尔操作数报错（CONVENTIONS §17.3，同 vt--）。"
  (when (%bool-arith-p dtype (list a b))
    (error "vt-sub: 布尔张量不支持减法（numpy 同样拒绝）。否定请用 vt-logical-not，异或请用 vt-logical-xor；若确实想要整数减法，请显式给出 :dtype :int8 等数值类型。见 CONVENTIONS §17.3"))
  (vt-fast-map #'- a b :dtype dtype :out out))

(defun vt-mul (a b &key dtype out)
  "逐元素乘法（二元特化入口），等价于 (vt-* a b)。
      :dtype 决定结果 dtype（缺省按提升规则），:out 契约同 vt-+。
      布尔操作数按逻辑与处理（CONVENTIONS §17.3，同 vt-*）。"
  (when (%bool-arith-p dtype (list a b))
    (return-from vt-mul
      (vt-fast-map #'%bool-and-op a b :dtype :bool :out out)))
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
      需要 numpy true_divide 语义（恒浮点、不 floor）请用 vt-/。

      布尔输入（CONVENTIONS §17.3）：结果 dtype 取 **:int8** 而不是 :bool ——
      商不再是布尔量（numpy：`np.array([True,False]) // np.array([True,False])`
      → int8 [1, 0]），留在 :bool 会让 1//1 这类结果冒充布尔真值。"
  (when (%cx-reject a b)
    (error "vt-div: 不支持复数输入（numpy 同样拒绝）。见 CONVENTIONS §16.11"))
  (when (and (null dtype) (%all-bool-tensors-p (list a b)))
    (setf dtype :int8))
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
        (vt-fast-map (function %cop-tan) in :out out :dtype dt))))
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
        (vt-fast-map (function %cop-tanh) in :out out :dtype dt))))
  (%float-map #'tanh vt out (%infer-float-dtype vt dtype)))


(defun vt-asin (vt &key out dtype)
  "逐元素反正弦。|x|>1 返回 NaN（对标 numpy，不抛条件）。
      整数输入输出 float64，float32 进 float32 出。:out 契约同 vt-+。"
  (when (%cx-any-p vt)
    (return-from vt-asin
      (let* ((in (ensure-vt vt))
             (dt (%cx-result-complex-dtype (vt-dtype in) dtype "VT-ASIN")))
        (vt-fast-map (function %cop-asin) in :out out :dtype dt))))
(let ((dt (%infer-float-dtype vt dtype)))
    (%float-map-pair %op-asin-df %op-asin-sf vt out dt)))

(defun vt-acos (vt &key out dtype)
  "逐元素反余弦。|x|>1 返回 NaN（对标 numpy，不抛条件）。
      整数输入输出 float64，float32 进 float32 出。:out 契约同 vt-+。"
  (when (%cx-any-p vt)
    (return-from vt-acos
      (let* ((in (ensure-vt vt))
             (dt (%cx-result-complex-dtype (vt-dtype in) dtype "VT-ACOS")))
        (vt-fast-map (function %cop-acos) in :out out :dtype dt))))
(let ((dt (%infer-float-dtype vt dtype)))
    (%float-map-pair %op-acos-df %op-acos-sf vt out dt)))

(defun vt-asinh (vt &key out dtype)
  "逐元素反双曲正弦。整数输入输出 float64，float32 进 float32 出。
      :out 契约同 vt-+。"
  (when (%cx-any-p vt)
    (return-from vt-asinh
      (let* ((in (ensure-vt vt))
             (dt (%cx-result-complex-dtype (vt-dtype in) dtype "VT-ASINH")))
        (vt-fast-map (function %cop-asinh) in :out out :dtype dt))))
  (%float-map #'asinh vt out (%infer-float-dtype vt dtype)))


(defun vt-acosh (vt &key out dtype)
  "逐元素反双曲余弦。x<1 返回 NaN（对标 numpy）。
      整数输入输出 float64，float32 进 float32 出。:out 契约同 vt-+。"
  (when (%cx-any-p vt)
    (return-from vt-acosh
      (let* ((in (ensure-vt vt))
             (dt (%cx-result-complex-dtype (vt-dtype in) dtype "VT-ACOSH")))
        (vt-fast-map (function %cop-acosh) in :out out :dtype dt))))
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
        (vt-fast-map (function %cop-atanh) in :out out :dtype dt))))
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

;;; ==================================================================
;;; 本文件按「编译期内存峰值」拆成三段（elementwise → elementwise1 → elementwise2，
;;; 顺序见 clvt.asd），此处的分界点就在 vt-exp 之后：
;;;
;;;   elementwise.lisp  四则/标量广播/浮点映射框架/三角与双曲/exp
;;;   elementwise1.lisp 幂与超几何（pow/log/sqrt/abs/signum、mod/rem、
;;;                     atan2、floor 族）
;;;   elementwise2.lisp 比较与逻辑/位运算/极值/clip/lerp/cbrt/hypot/
;;;                     reciprocal/negative/sinc
;;;
;;; 拆分动机：整份文件（约 1530 行、133 个顶层定义）在 SBCL 2.6.8 下
;;; 冷编译时 IR1 阶段会吃满 4 GB 动态空间直接 "Heap exhausted, game over"，
;;; 而 run-tests.sh 的默认值正是 4096 MB。
;;;
;;; 铁律：**每一段都只允许定义自己那一批符号**。曾经出现过「复制出去但没从
;;; 原文件删掉」的版本，结果是 85 个 DEFUN 各存在两份，后加载者胜 —— 与
;;; CONVENTIONS §16.32/§16.33 花两轮消灭的「覆盖式分派」是同一个病根，
;;; 只是这次发生在文件之间。test/complex-dispatch-test.lisp 的
;;; 「跨文件同名 defun = 0」断言会把它抓出来。
;;; ==================================================================
