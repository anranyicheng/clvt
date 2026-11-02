;;;; nn.lisp — 激活函数、损失函数与 softmax

(in-package :clvt)

;;; ------------------------------------------------------------------
;;; 激活函数 — 优化版：优先使用特化快路径避免通用lambda funcall开销
;;; ------------------------------------------------------------------

(declaim (inline %relu-double %relu-single %sigmoid-double %sigmoid-single))

;;; ------------------------------------------------------------------
;;; 整数/非浮点输入的 dtype 提升辅助（激活函数的统一精度契约）
;;; ------------------------------------------------------------------
;;; 对标 numpy「整型进 float64 出、float32 进 float32 出」：
;;;   vt-softplus / vt-gelu 等激活函数若直接把 lambda 交给 vt-map 且
;;;   `dtype=nil`，vt-map 会走 vt-promote-type，对整数输入得到**整数**
;;;   结果 dtype——lambda 内部算出的浮点值被静默截断写回整数存储。
;;;   例：(vt-softplus (vt-asarray '(0))) → 0（应为 0.693…）；
;;;       (vt-gelu (vt-asarray '(1 2 3))) → (0 1 2)（应为 0.841/1.955/2.996）。
;;; 修复方式与 elementwise.lisp 的 %infer-float-dtype 同构：缺省 dtype
;;; 时按输入推导（float32 → :float32，其余含整数 → :float64）。
(defun %nn-float-dtype (x dtype)
  "推导激活函数的浮点结果 dtype。X 可为张量或标量。
   显式 DTYPE 优先；否则 float32 输入 → :float32，其余（含整数）→ :float64。"
  (or dtype
      (let ((dt (if (vt-p x) (vt-dtype x) nil)))
        (if (eq dt :float32) :float32 :float64))))

(defun %nn-as-float (x dtype)
  "把 X 规范到目标浮点 DTYPE（已是则原样返回，避免多余拷贝）。
   整数/无符号输入必须经过此步，否则浮点结果会被截断写回整数存储。"
  (let ((v (ensure-vt x)))
    (if (eq (vt-dtype v) dtype) v (vt-astype v dtype))))

;; ReLU：用 (< x 0.0d0) 判断。
;;   - x < 0  → 返回 0
;;   - x >= 0 → 返回 x
;;   - x = NaN → (< NaN 0.0d0) 为 nil（陷阱已屏蔽），返回 x 本身 = NaN ✔
;; 注意：NaN 参与比较要求调用方已屏蔽浮点陷阱。
(defun %relu-double (x)
  (declare (double-float x) (optimize (speed 3) (safety 0)))
  (if (< x 0.0d0) 0.0d0 x))

(defun %relu-single (x)
  (declare (single-float x) (optimize (speed 3) (safety 0)))
  (if (< x 0.0s0) 0.0s0 x))

;; Sigmoid：x = NaN 时走 else 分支，exp(NaN)=NaN，除法结果仍为 NaN，语义正确；
;;   但 NaN 参与 (>= x 0.0d0) 比较同样需要陷阱已屏蔽。
;;   数值稳定版的分支写法保持不变即可。
(defun %sigmoid-double (x)
  (declare (double-float x) (optimize (speed 3) (safety 0)))
  (if (>= x 0.0d0)
      (/ 1.0d0 (+ 1.0d0 (exp (- x))))
      (let ((e (exp x))) (/ e (+ 1.0d0 e)))))

(defun %sigmoid-single (x)
  (declare (single-float x) (optimize (speed 3) (safety 0)))
  (if (>= x 0.0s0)
      (/ 1.0s0 (+ 1.0s0 (exp (- x))))
      (let ((e (exp x))) (/ e (+ 1.0s0 e)))))

(defun vt-sigmoid (vt &key dtype out)
  "数值稳定 sigmoid。连续 float 张量走特化快路径。

  dtype 语义（§4.2 H3「严格相等」）：结果 dtype = 显式 :dtype，否则按输入
  浮点提升（非浮点 → :float64）。out 必须精确匹配结果 shape/dtype。"
  (when (%cx-reject vt)
    (error "vt-sigmoid: 不支持复数输入（numpy 同样拒绝）。见 CONVENTIONS §16.11"))
(with-float-safe
    (let* ((dt (or dtype (if (eq (vt-dtype vt) :float32) :float32 :float64)))
           (a (ensure-vt vt))
           ;; 统一硬契约前置：形状/dtype 严格相等/可写（非连续 out 允许，
           ;; 由下方 vt-map 通用路径正确处理 strides）
           (res (if out
                    (vt-check-out out (vt-shape a) dt :op-name "vt-sigmoid")
                    (make-vt (vt-shape a) 0 :dtype dt))))
      (if (and (vt-contiguous-p res)
	       (vt-contiguous-p a)
               (eq (vt-dtype a) dt)
	       (eq (vt-dtype res) dt)          
               (equal (vt-shape res) (vt-shape a))
               (member dt '(:float64 :float32)))
          (let* ((rd (vt-data res))
		 (ad (vt-data a))
		 (size (vt-size a))
		 (rop (vt-offset res))
		 (aop (vt-offset a)))
            (declare (fixnum size rop aop) (optimize (speed 3) (safety 0)))
            (if (eq dt :float64)
		(let ((r (the (simple-array double-float (*)) rd))
                      (d (the (simple-array double-float (*)) ad)))
                  (dotimes (i size)
                    (setf (aref r (+ rop i)) (%sigmoid-double (aref d (+ aop i))))))
		(let ((r (the (simple-array single-float (*)) rd))
                      (d (the (simple-array single-float (*)) ad)))
                  (dotimes (i size)
                    (setf (aref r (+ rop i)) (%sigmoid-single (aref d (+ aop i)))))))
            res)
          (vt-map (lambda (x) (if (>= x 0) (/ 1.0d0 (+ 1.0d0 (exp (- x))))
                                  (let ((e (exp x))) (/ e (+ 1.0d0 e)))))
                  a :dtype dt :out res)))))

(defun vt-relu (vt &key dtype out)
  "ReLU 激活。连续 float 张量走特化快路径（消除 funcall 开销）。
  v0.4.1（差分测试 R8）：整数输入保持整数 dtype（对标 numpy.maximum
  与 torch.relu）；浮点输入保持自身精度；显式 :dtype 仍生效。"
  (when (%cx-reject vt)
    (error "vt-relu: 不支持复数输入（numpy 同样拒绝）。见 CONVENTIONS §16.11"))
(with-float-safe
    (let* ((in-dt (vt-dtype vt))
           (dt (or dtype
                   (if (member in-dt '(:int8 :int16 :int32 :int64 :uint8 :uint16))
                       in-dt
                       (if (eq in-dt :float32) :float32 :float64))))
           (a (ensure-vt vt))
           (res (if out
                    (vt-check-out out (vt-shape a) dt :op-name "vt-relu")
                    (make-vt (vt-shape a) 0 :dtype dt))))
      (if (and (vt-contiguous-p res)
	       (vt-contiguous-p a)
               (eq (vt-dtype a) dt)
	       (eq (vt-dtype res) dt)
               (equal (vt-shape res) (vt-shape a))
               (member dt '(:float64 :float32)))
          (let* ((rd (vt-data res))
		 (ad (vt-data a))
		 (size (vt-size a))
		 (rop (vt-offset res))
		 (aop (vt-offset a)))
            (declare (fixnum size rop aop) (optimize (speed 3) (safety 0)))
            (if (eq dt :float64)
		(let ((r (the (simple-array double-float (*)) rd))
                      (d (the (simple-array double-float (*)) ad)))
                  (dotimes (i size)
                    (setf (aref r (+ rop i)) (%relu-double (aref d (+ aop i))))))
		(let ((r (the (simple-array single-float (*)) rd))
                      (d (the (simple-array single-float (*)) ad)))
                  (dotimes (i size)
                    (setf (aref r (+ rop i)) (%relu-single (aref d (+ aop i)))))))
            res)
	  ;; 通用路径：先把输入规范为目标浮点 dtype，再调用带 (double-float x)
	  ;; (safety 0) 声明的内联核函数。
	  ;; 若把 int64 等非浮点元素直接交给 %relu-double，其类型声明会被违背，
	  ;; 在 (safety 0) 下 fixnum 被当作 boxed double 解引用非法地址 → 段错误。
	  ;; 用无类型假设的 lambda 包裹（与 vt-sigmoid 通用路径同构）即可安全提升。
	  (let ((af (if (eq (vt-dtype a) dt) a (vt-astype a dt))))
	    (vt-map (if (eq dt :float64)
			(lambda (x) (%relu-double (coerce x 'double-float)))
			(lambda (x) (%relu-single (coerce x 'single-float))))
		    af :dtype dt :out res))))))

;;; ------------------------------------------------------------------
;;; 激活/损失的具名算子（性能契约）
;;;
;;; `vt-fast-map` 只在 `(function <符号>)` 这种字面形式下展开内联循环；
;;; 传 `(lambda (x) ...)` 会退化到 `vt-map` 的闭包路径 —— 每元素一次
;;; funcall + 一次泛型 aref + 一次装箱。1M float64 实测：gelu 65ms（闭包）
;;; vs 12ms（具名算子内联），softplus 同量级。
;;;
;;; 表达式体与本文件原先的 lambda **逐字相同**（连字面量的精度后缀都不改），
;;; 因此结果的舍入链不变 —— 只是不再为每个元素建立一次函数调用。
;;;
;;; 带参数的算子（elu/leaky-relu/bce 的 ALPHA、EPS）把参数绑在特殊变量上：
;;; `vt-fast-map` 要求算子必须是符号，闭包传不进去。SBCL 的特殊变量绑定
;;; 是线程局部的，因此并发调用互不干扰（与 `with-float-safe` 的绑定同一机制）。
;;; ------------------------------------------------------------------
(defvar *nn-act-param* 1.0d0
  "带参激活算子的当前参数（ALPHA）。由调用方 let 绑定。")

;;; 表达式体只写一次，按元素类型分别实例化成两个**已知参数类型**的算子。
;;;
;;; 为什么必须让编译器知道参数类型：`vt-fast-map` 内联的是「循环 + 直接
;;; 调用」，而算子自身的算术仍按它自己的编译结果走。参数类型未知时算子体
;;; 退化成泛型算术（每元素一次装箱 + 泛型分派），1M gelu 实测 86ms —— 比
;;; 原来的闭包路径还慢。见 `%def-nn-kernel`：现在用 `(coerce x 'lt)` 而不是
;;; `(declare (type lt x))` 拿到这个类型信息，理由写在宏的 docstring 里。
;;;
;;; 为什么 f32 版本**不**先把 x 提升成 double：字面量（0.044715d0 / 0.5d0）
;;; 本身就把传染后的中间结果带到 double，而 `(* x x x)` 在 single 上先舍入
;;; 两次。先提升会改变 x³ 的舍入，结果与旧实现不一致 —— 声明的唯一作用是
;;; 让编译器知道入参类型，不是改精度。
;;;
;;; 为什么还要 `(declaim (ftype ...))`：`%softplus-form` 是
;;; `(if (> x 20) x (log (+ 1 (exp x))))`，SBCL 由它推导出
;;; `%softplus-df` 的返回类型是 `(double-float 0.0d0)`（softplus 恒非负，
;;; 推导本身没错），随后按「可信派生类型」在**每个调用点**插一条返回值
;;; 检查。NaN 不落在这个区间里，于是 `(vt-softplus #(… nan …))` 在通用
;;; 路径的 `funcall` 之后抛 `TYPE-ERROR … (DOUBLE-FLOAT 0.0d0)` —— 语义上
;;; 完全合法的 NaN 被拒。显式宣告 ftype 后派生类型以声明为准，不再收窄。
(defmacro %def-nn-kernel (name args lt ret body)
  "把 BODY 固化成 ARGS 全为 LT 的算子（参数归一 + safety 0 + 显式 ftype）。

   RET 是**真实的**返回类型（由 `%simple-fun-type` 实测得来），不是猜的：
   声明必须比编译器推出来的**宽**，且必须覆盖 NaN。`(function (t) RET)` 的
   形参位置一律写 `t`，见下。

   参数用 `(coerce x ',lt)` 重绑定，而不是 `(declare (type ,lt ,@args))`：
   `vt-fast-map` 在编译期会把同一个算子编进**全部候选分支**（窄整型掩码、
   各宽度整型内联循环……），运行期才按 dtype 选。那些分支里的
   `(aref (simple-array (unsigned-byte 8) (*)) i)` 与 double-float 形参的
   断言冲突，SBCL 会为每个调用点各报一条 derived-type WARNING，而
   「零 WARNING 编译」是本库门禁。
   热路径上参数本来就是 LT（内联循环的数组已声明元素类型），
   `coerce` 被常量折叠成恒等，不产生任何指令。

   形参位置同样只能写 `t`：写成 LT 等于在每个调用点给实参加了一条类型
   断言，窄整型分支的 aref 立刻报「derived type 是 (signed-byte 16)，
   与断言 SINGLE-FLOAT 冲突」（同一个理由，第二次踩）。RET 也不能写成
   LT：`%gelu-sf` 的字面量是 double，返回值本来就是 double-float，硬说
   single-float 会报「derived type DOUBLE-FLOAT 与声明冲突」。"
  `(progn
     (declaim (ftype (function (t) ,ret) ,name))
     (defun ,name ,args
       (declare (optimize (speed 3) (safety 0)))
       (let ,(mapcar (lambda (a) `(,a (coerce ,a ',lt))) args)
         ,body))))

(defmacro %nn-act-map (df sf x dt out)
  "按结果 dtype 选 DF / SF 算子做内联映射（两条分支都是内联循环）。"
  `(if (eq ,dt :float64)
       (vt-fast-map (function ,df) ,x :dtype ,dt :out ,out)
       (vt-fast-map (function ,sf) ,x :dtype ,dt :out ,out)))

(defmacro %leaky-relu-form (x)
  `(if (> ,x 0.0d0) ,x (* *nn-act-param* ,x)))

(defmacro %softplus-form (x)
  `(if (> ,x 20.0d0) ,x (log (+ 1.0d0 (exp ,x)))))

(defconstant +gelu-sqrt-2-over-pi+
  (sqrt (/ 2.0d0 (coerce pi 'double-float)))
  "GELU tanh 近似的常数 √(2/π)。提成常量而不是在算子体内重算：
   算子按**每元素**调用一次，体内求值等于给每个元素加一次 sqrt。")

(defmacro %gelu-form (x)
  `(let* ((x3 (* ,x ,x ,x))
          (inner (+ ,x (* 0.044715d0 x3)))
          (tanh-val (tanh (* +gelu-sqrt-2-over-pi+ inner))))
     (* 0.5d0 ,x (+ 1.0d0 tanh-val))))

;;; RET 取自 `%simple-fun-type` 的实测值（去掉区间收窄的那部分）：
;;;   leaky-relu 的 `(* *nn-act-param* x)` 里 ALPHA 是**无类型声明的特殊变量**
;;;   （用户可以传 single/整数），所以乘法在编译器看来可能落进 complex ——
;;;   声明如实照抄，不为了「好看」把它写成 double-float（那会让传整型
;;;   ALPHA 的调用变成未定义行为）。
;;;   gelu 的两个版本都返回 DOUBLE-FLOAT：体里的字面量是 double，且刻意
;;;   不改成 single（见上文「f32 版本不先提升」那段）。
(%def-nn-kernel %leaky-relu-df (x) double-float
  (or double-float (complex double-float)) (%leaky-relu-form x))
(%def-nn-kernel %leaky-relu-sf (x) single-float
  (or float (complex single-float) (complex double-float)) (%leaky-relu-form x))
(%def-nn-kernel %softplus-df  (x) double-float
  double-float (%softplus-form x))
(%def-nn-kernel %softplus-sf  (x) single-float
  (or single-float double-float) (%softplus-form x))
(%def-nn-kernel %gelu-df       (x) double-float double-float (%gelu-form x))
(%def-nn-kernel %gelu-sf       (x) single-float double-float (%gelu-form x))

(defun vt-leaky-relu (vt &key (alpha 0.01d0) dtype out)
  "Leaky ReLU 激活：x > 0 时为 x，否则为 alpha * x（ALPHA 缺省 0.01）。
   dtype 语义：缺省按输入推导——float32 进 float32 出、其余（含整数）
   进 float64 出。整数输入必须提升，否则 alpha*x 的浮点结果会被截断
   写回整数存储（如 (-2 3 -4) 会静默变成 (0 3 0)）。"
  (when (%cx-reject vt)
    (error "vt-leaky-relu: 不支持复数输入（numpy 同样拒绝）。见 CONVENTIONS §16.11"))
(let ((dt (%nn-float-dtype vt dtype)))
    (let ((*nn-act-param* alpha))
      (%nn-act-map %leaky-relu-df %leaky-relu-sf (%nn-as-float vt dt) dt out))))

(defun vt-swish (vt &key dtype out)
  "Swish / SiLU 激活：x * sigmoid(x)。"
  (when (%cx-reject vt)
    (error "vt-swish: 不支持复数输入（numpy 同样拒绝）。见 CONVENTIONS §16.11"))
(let ((sig (vt-sigmoid vt :dtype dtype)))
    (vt-map #'* vt sig :dtype dtype :out out)))

(defun vt-softplus (vt &key dtype out)
  "Softplus 激活：log(1 + exp(x))。x > 20 时直接返回 x 以避免 exp 溢出。
   dtype 语义（对标 numpy）: 缺省按输入推导——float32 进 float32 出、
   其余（含整数）进 float64 出。"
  (when (%cx-reject vt)
    (error "vt-softplus: 不支持复数输入（numpy 同样拒绝）。见 CONVENTIONS §16.11"))
(let ((dt (%nn-float-dtype vt dtype)))
    (%nn-act-map %softplus-df %softplus-sf (%nn-as-float vt dt) dt out)))

(defun vt-gelu (vt &key dtype out)
  "GELU 激活（tanh 近似式，对标 PyTorch F.gelu 默认近似）。
   dtype 语义（对标 numpy）: 缺省按输入推导——float32 进 float32 出、
   其余（含整数）进 float64 出。"
  (when (%cx-reject vt)
    (error "vt-gelu: 不支持复数输入（numpy 同样拒绝）。见 CONVENTIONS §16.11"))
(let ((dt (%nn-float-dtype vt dtype)))
    (%nn-act-map %gelu-df %gelu-sf (%nn-as-float vt dt) dt out)))

(defun vt-mish (vt &key dtype out)
  "Mish 激活：x * tanh(softplus(x))。"
  (when (%cx-reject vt)
    (error "vt-mish: 不支持复数输入（numpy 同样拒绝）。见 CONVENTIONS §16.11"))
(let ((sp (vt-softplus vt :dtype dtype)))
    (vt-* vt (vt-tanh sp) :dtype dtype :out out)))

(defun vt-hard-tanh (vt &key dtype out)
  "HardTanh 激活：把输入截断到 [-1, 1]。"
  (when (%cx-reject vt)
    (error "vt-hard-tanh: 不支持复数输入（numpy 同样拒绝）。见 CONVENTIONS §16.11"))
(vt-clip vt -1.0d0 1.0d0 :dtype dtype :out out))

(defun vt-hard-sigmoid (vt &key dtype out)
  "HardSigmoid 激活：clip(x/5 + 0.5, 0, 1)。"
  (when (%cx-reject vt)
    (error "vt-hard-sigmoid: 不支持复数输入（numpy 同样拒绝）。见 CONVENTIONS §16.11"))
(let ((scaled (vt-+ (vt-scale vt 0.2d0) 0.5d0 :dtype dtype)))
    (vt-clip scaled 0.0d0 1.0d0 :dtype dtype :out out)))

(defun %nn-empty-result (in dtype out op-name)
  "空输入时 NN 族的统一出口：同形状空张量（float64/float32 随输入）。
   OUT 非 NIL 时按 §4.2 的 out 硬契约先校验再写入，返回 OUT。"
  (let* ((dt (or dtype (if (eq (vt-dtype in) :float32) :float32 :float64)))
         (shape (vt-shape in))
         (empty (vt-zeros shape :dtype dt)))
    (if out
        (progn (vt-check-out out shape dt :op-name op-name)
               (vt-copy-into out empty)
               out)
        empty)))

(defun vt-softmax (vt &key (axis -1) dtype out)
  "softmax（沿 axis，默认最后一维）。数值稳定化：先减去该轴最大值再 exp。
  注意（对标 PyTorch）：若某行全为 -Inf，则 max = -Inf，exp(-Inf - (-Inf)) = NaN，
  即全 -Inf 行的 softmax 结果为 NaN——这是 IEEE 754 下减去最大值稳定化的固有语义。"
  (let ((in (ensure-vt vt)))
    ;; 空输入 → 空输出。没有元素就没有「该轴最大值」可归约，落到 vt-amax 会
    ;; 抛它自己的「zero-size array」错误 —— 那是实现细节，不是本函数的契约
    ;; （§17.10.23）。
    (when (zerop (vt-size in))
      (return-from vt-softmax
        (%nn-empty-result in dtype out "vt-softmax")))
    (let* ((max-val (vt-amax in :axis axis :keepdims t :dtype dtype))
           ;; 减最大值与 exp 保持**两趟**：`max-val` 的形状是 (…, 1) 的列广播，
           ;; `(vt-- in max-val)` 落在 `%sew2` 的 :col 向量循环上（1M 约 1 ms），
           ;; 而把两者融合成一个标量算子只能走 odometer（exp 无法向量化，
           ;; 融合后又必须逐元素）、实测 20 ms。多一趟 8 MB 中转换 10 ms+。
           ;; 表达式与分步写法逐字相同，舍入链不变。
           (shifted (vt-- in max-val :dtype dtype))
           (exp-vt (vt-exp shifted :dtype dtype))
           (sum-exp (vt-sum exp-vt :axis axis :keepdims t :dtype dtype)))
      (vt-/ exp-vt sum-exp :dtype dtype :out out))))

(defun vt-log-softmax (vt &key (axis -1) dtype out)
  "log-softmax = x - logsumexp(x)，沿 axis（默认最后一维）。
  数值稳定化同 vt-softmax；结果 dtype 由输入提升 + 显式 :dtype 决定（§4.2 H3）。"
  (let ((in (ensure-vt vt)))
    (when (zerop (vt-size in))
      (return-from vt-log-softmax
        (%nn-empty-result in dtype out "vt-log-softmax")))
    (let* ((max-val (vt-amax in :axis axis :keepdims t :dtype dtype))
           (shifted (vt-- in max-val :dtype dtype))
           ;; 与 vt-softmax 同一取舍：减法走 :col 向量循环、exp 单独一趟。
           (exp-vt (vt-exp shifted :dtype dtype))
           (lse (vt-log (vt-sum exp-vt :axis axis :keepdims t :dtype dtype)
                        :dtype dtype)))
      (vt-- shifted lse :dtype dtype :out out))))

(defun vt-mean-squared-error (y-true y-pred &key dtype out)
  "均方误差：mean((y_true - y_pred)²)。结果为标量（float64）。"
  (vt-mean (vt-square (vt-- y-true y-pred :dtype dtype) :dtype dtype)
	   :dtype dtype :out out))

(defun vt-binary-cross-entropy (y-true y-pred &key (eps 1.0d-7) dtype out)
  "二元交叉熵（对标 torch.nn.BCELoss）。p 含 NaN 时损失为 NaN；EPS 用于裁剪 log 的自变量避免 log(0)。"
  ;; NaN 语义对标 torch.BCELoss：p 含 NaN 时损失必须为 NaN，
  ;; 裁剪前显式判断（min/max 在 NaN 上比较恒 nil 会把 NaN 静默替换为 1-eps）。
  (vt-mean (vt-map (lambda (y p)
                     (let* ((pc (if (and (floatp p) (not (= p p)))
                                    p
                                    (max eps (min (- 1.0d0 eps) p))))
                            (omp (if (and (floatp pc) (not (= pc pc)))
                                     pc
                                     (max eps (- 1.0d0 pc)))))
                       (- (+ (* y (log pc)) (* (- 1.0d0 y) (log omp))))))
                   y-true y-pred :dtype dtype)
           :dtype dtype :out out))

(defun vt-cross-entropy (y-true y-pred &key (eps 1.0d-7) dtype out)
  "多类交叉熵（**概率/one-hot 输入**）。逐样本取 -Σ y·log(p) 后取均值；
   EPS 用于裁剪 p 避免 log(0)，语义同 vt-binary-cross-entropy。

   注意：本函数接收**概率** y-pred（通常来自 vt-softmax）与同形状的
   one-hot（或软标签）y-true，不做 softmax、也不接受类别索引。
   若要在 **raw logits + 整数类别索引** 上做分类交叉熵
   （即 torch.nn.CrossEntropyLoss 的标准用法），请用
   `vt-cross-entropy-logits`。

   历史说明：本函数早期把「对标 torch.nn.CrossEntropyLoss」写进文档，
   但 torch 的该 API 实为 logits+索引语义，二者并不一致。现已把文档
   修正为本函数真实语义，对标需求由 vt-cross-entropy-logits 承担。"
  (let* ((p-clipped (vt-clip y-pred eps (- 1.0d0 eps) :dtype dtype))
         (log-prob (vt-log p-clipped :dtype dtype))
         (loss-per-sample (vt-- (vt-sum (vt-* y-true log-prob :dtype dtype)
					:axis -1 :dtype dtype)
                                :dtype dtype)))
    (vt-mean loss-per-sample :dtype dtype :out out)))

(defun %ce-flatten (x)
  "把嵌套 list / 张量摊平成标量列表（用于类别索引越界检查）。"
  (cond ((vt-p x) (%ce-flatten (vt-to-list x)))
        ((consp x) (mapcan #'%ce-flatten x))
        (t (list x))))

(defun vt-cross-entropy-logits (logits labels &key (axis -1) (reduction :mean) dtype out)
  "分类交叉熵（**raw logits + 整数类别索引**，对标 torch.nn.CrossEntropyLoss）。

   LOGITS  未归一化的得分张量，沿 AXIS 为类别维（缺省最后一维）。
   LABELS  整数类别索引，形状 = LOGITS 去掉 AXIS 维后的形状
           （如 LOGITS 为 (N, C) 时 LABELS 为 (N,)）。
   AXIS    类别轴，缺省 -1。支持任意秩：LOGITS (N1,N2,...,C) 配
           LABELS (N1,N2,...)。也支持无批次情形（LOGITS 为 (C,)、
           LABELS 为 0 维或长度 1）。
   REDUCTION  :mean（缺省，按样本均值）/ :sum / :none（逐样本损失）。

   数值稳定：内部用 log-softmax（先减该轴最大值再取 log-sum-exp），
   因此大 logits 不溢出，且**不做 [eps, 1-eps] 裁剪**——logits 本就
   可为负、可远大于 1，裁剪会破坏语义。类别索引越界即报错。

   与 vt-cross-entropy 的区别：后者收概率（通常 softmax 之后）+
   one-hot 标签；本函数直接吃 logits + 索引，一步到位且更数值稳定
   （无需先 softmax 再取 log，避免 log(softmax(·)) 的精度损失）。"
  (let* ((lg (ensure-vt logits))
         (lb (ensure-vt labels))
         (rank (vt-order lg))
         (ax (vt-normalize-axis axis rank))
         (dt (or dtype (if (eq (vt-dtype lg) :float32) :float32 :float64)))
         (c (nth ax (vt-shape lg)))
         (lb-int (if (member (vt-dtype lb) '(:int8 :int16 :int32 :int64
                                             :uint8 :uint16))
                     lb
                     (error "vt-cross-entropy-logits: LABELS 必须是整数类别索引，实得 dtype ~a"
                            (vt-dtype lb))))
         (exp-shape (append (subseq (vt-shape lg) 0 ax)
                            (subseq (vt-shape lg) (1+ ax))))
         (lb-shape (vt-shape lb-int)))
    ;; 形状校验：LABELS 须为 LOGITS 去掉类别轴后的形状。
    ;; 容忍无批次写法：LOGITS 为 (C,) 时 exp-shape = NIL，此时 LABELS
    ;; 允许 0 维标量或长度 1 的 (1,)（都表示「一个样本，类别为该索引」）——
    ;; 后者需 reshape 成 () 以与 logits 的标量索引对齐。
    (unless (equal lb-shape exp-shape)
      (if (and (null exp-shape)
               (or (null lb-shape)
                   (and (= 1 (length lb-shape)) (= 1 (car lb-shape)))))
          (setf lb-int (vt-reshape lb-int nil))
          (error "vt-cross-entropy-logits: LABELS 形状 ~a 与期望 ~a 不符（应为 LOGITS ~a 去掉第 ~a 轴）"
                 lb-shape exp-shape (vt-shape lg) ax)))
    ;; 类别索引越界检查（对标 numpy/torch 的 IndexError）
    (dolist (i (%ce-flatten lb-int))
      (unless (and (integerp i) (<= 0 i) (< i c))
        (error "vt-cross-entropy-logits: 类别索引 ~a 越界（类别数 ~a）" i c)))
    ;; 1) log-softmax：shifted - log(sum(exp(shifted)))
    (let* ((maxv (vt-amax lg :axis ax :keepdims t :dtype dt))
           (shifted (vt-- lg maxv :dtype dt))
           (lse (vt-log (vt-sum (vt-exp shifted :dtype dt)
                                :axis ax :keepdims t :dtype dt)
                        :dtype dt))
           (logp (vt-- shifted lse :dtype dt))
           ;; 2) 逐样本取 -logp[..., labels, ...] 对应类别。
           ;; 不用 vt-take-along-axis（它要求 indices 与 arr 同秩，而 labels
           ;; 比 logits 少一维）。改用 one-hot 掩码：把 labels 在类别轴上
           ;; 扩一维成 (...,1,...)，与 arange(C) 比较即得 one-hot；再与 logp
           ;; 逐元素相乘后沿类别轴求和，等价于按索引取值。
           (labels-keep (vt-expand-dims lb-int ax))
           (arange-v (vt-arange c :dtype :int64))
           ;; arange 形状 (C,) → reshape 成在 ax 处为 C、其余为 1 的形状，
           ;; 以便与 labels-keep 广播比较
           (arange-shape (loop for d from 0 below rank
                               collect (if (= d ax) c 1)))
           (arange-r (vt-reshape arange-v arange-shape))
           (onehot (vt-= labels-keep arange-r))
           ;; onehot 是 int8 布尔，转成 logp 的浮点 dtype 再相乘
           (mask (vt-astype onehot dt))
           (sel (vt-sum (vt-* mask logp :dtype dt) :axis ax :dtype dt))
           (per-sample (vt-negative sel :dtype dt)))
      (case reduction
        (:none per-sample)
        (:sum (vt-sum per-sample :dtype dt :out out))
        (:mean (vt-mean per-sample :dtype dt :out out))
        (t (error "vt-cross-entropy-logits: 未知 REDUCTION ~s（:mean/:sum/:none）"
                  reduction))))))
