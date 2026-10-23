;;;; dispatch.lisp — 复数分派的单一入口
;;;;
;;;; 背景
;;;; ----
;;;; 历史教训（详见 HISTORY.md §H.2）：367 个导出里曾有 159 个（43%）被运行期替换：
;;;;   三个文件用 %CX-DEF-* 宏 `(setf (symbol-function ...))` 装匿名包装，
;;;;   另有 65 个用「同名 defun 再写一遍 + (%cx-orig 'name) 转发」。
;;;;   后果（实测）：
;;;;     * 只重载 src/elementwise.lisp，(vt-sqrt #C(-1 0)) 从 complex128 的
;;;;       #C(-0.0d0 1.0d0) 静默变成 float64 的 NaN，且不报错；
;;;;     * 函数对象是匿名 LAMBDA，`declaim inline` 对调用点失效；
;;;;     * 哪些符号被覆盖无法静态枚举（vt-argmax 的实数实现是宏生成的，
;;;;       源码里没有字面 DEFUN），只能问运行时注册表。
;;;;
;;;; 现在的约定（本文件是唯一判据）
;;;; ----
;;;;   每个公开符号只有一个 DEFUN，在它的归属文件里，体内第一句分派：
;;;;
;;;;     (defun vt-sqrt (tensor &key out dtype)
;;;;       "…"
;;;;       (when (%cx-any-p tensor)                       ; ← 唯一新增
;;;;         (return-from vt-sqrt (%cx-sqrt tensor out dtype)))
;;;;       …原实数实现，逐字不动…)
;;;;
;;;;   为什么用 WHEN + RETURN-FROM 而不是把 body 包进 IF：
;;;;     1. DEFUN 自带同名 block，RETURN-FROM 天然可用，原 body 的缩进与
;;;;        括号完全不动 —— 159 处改造可以脚本化，diff 只增不减；
;;;;     2. 单一 DEFUN / 单一 lambda-list / 单一 docstring，不存在
;;;;        「两处 lambda-list 漂移」（`&allow-other-keys` 那种静默丢参）；
;;;;     3. 实数路径的代价是一次内联的 dtype 判断，比原来「funcall 包装
;;;;        lambda + 同样一次判断」净减少。
;;;;
;;;;   COMPLEX.LISP / COMPLEX-OPS.LISP / COMPLEX-LINALG.LISP 由此退化为
;;;;   **纯内核库**：只放 %CX-* / %COP-* 内部函数，不再定义公开符号。
;;;;   *CX-ORIGINALS*、%CX-ORIG、以及三个文件里的 EVAL-WHEN 预捕获清单
;;;;   全部删除（见 CONVENTIONS §16.32）。

(in-package :clvt)

;;; ==================================================================
;;; 1. 分派判据
;;; ==================================================================
;;; 这两个谓词原在 complex.lisp（:serial t 下倒数第三），导致 elementwise /
;;; reduce-stats / linalg 这些**先加载**的归属文件用不上它们 —— 那正是
;;; 「只能事后覆盖」的结构性原因。现在前移到 core 之后、所有归属文件之前。

(declaim (inline %cx-complex-dtype-p*))

(defun %cx-complex-dtype-p* (dtype)
  "DTYPE 是否复数。内联壳，省掉每次调用 VT-COMPLEX-DTYPE-P 的函数开销。"
  (member dtype '(:complex64 :complex128) :test #'eq))

(declaim (inline %cx-tensor-p))

(defun %cx-tensor-p (x)
  "单操作数**快判**：X 是复数张量 / 复数标量。
   ⚠ 对 list/vector 一律返回 NIL —— 分派请用 %CX-ANY-P。
   这个名字在 complex*.lisp 内部被用了几百次（实参已是 VT），故保留。"
  (cond ((vt-p x) (not (null (%cx-complex-dtype-p* (vt-dtype x)))))
        (t (complexp x))))

(defun %cx-deep-p (x)
  "递归判定 X 是否「含复数」：vt → dtype；数 → complexp；容器 → 递归。
   实参可能是 list/vector（(vt-cbrt (list #C(8 0))) 就是既有测试用例）。"
  (cond
    ((vt-p x) (not (null (%cx-complex-dtype-p* (vt-dtype x)))))
    ((complexp x) t)
    ((consp x) (some #'%cx-deep-p x))
    ((vectorp x) (some #'%cx-deep-p x))
    (t nil)))

(declaim (inline %cx-arg-complex-p))

(defun %cx-arg-complex-p (x)
  "单个实参是否需要走复数路径。
   热路径（VT）走内联的两步判断；只有容器才进递归 —— 不给每次调用加函数开销。
   这是**分派**的正确判据：用它而不是 %CX-TENSOR-P，否则 list 入参会走错分支。"
  (cond ((vt-p x) (not (null (%cx-complex-dtype-p* (vt-dtype x)))))
        ((complexp x) t)
        ((or (consp x) (vectorp x)) (%cx-deep-p x))
        (t nil)))

;;; 多实参版本做成**宏**而不是 (&rest args) 函数：后者每次调用都要 cons 一个
;;; rest 列表，而这些 op 在热循环里会被调上百万次。宏展开成短路 OR，
;;; 零分配，单实参时就是一次内联判断。
(defmacro %cx-any-p (&rest args)
  "任一实参需要走复数路径即为真（含 list/vector 入参）。"
  `(or ,@(mapcar (lambda (a) `(%cx-arg-complex-p ,a)) args)))

(defmacro %cx-reject (&rest args)
  "拒绝型守卫的判据，与 %CX-ANY-P 同源 —— 保留别名只为让调用点意图自明。"
  `(%cx-any-p ,@args))

;;; ==================================================================
;;; 2. 复数结果 dtype 的规则
;;; ==================================================================
;;; 放在这里而不是 complex.lisp：它是 24 个一元分派点的公共依赖，必须在
;;; elementwise / extensions3 **编译之前**可见（否则每处都留一个前向引用）。

(defun %cx-result-complex-dtype (in-dtype dtype op-name)
  "复数输入的数学函数结果 dtype：缺省保持输入复数 dtype；
   显式非复数 dtype 报错（对标 numpy ufunc 的 safe-casting 拒绝 complex→real）。"
  (cond ((null dtype) in-dtype)
        ((%cx-complex-dtype-p* dtype) dtype)
        (t (error "~a: 复数输入的结果 dtype 必须是复数类型，收到 ~a（CONVENTIONS §16.5）"
                  op-name dtype))))

;;; ==================================================================
;;; 3. 归约族的分派
;;; ==================================================================
;;; 归约的公开名由 DEF-VT-REDUCE-FACADE 生成（reduce-stats.lisp），实现函数是
;;; %REDUCE-<NAME>。这里按算子把复数分支路由到对应内核 —— 一个入口覆盖
;;; 14 个符号（sum/prod/max/min、nan* 四兄弟、all/any、argmax 四兄弟）。
;;;
;;; ALL/ANY 特殊：它们的复数语义是「任一分量非零即为真」，实现方式是先把
;;; 复数转成 int8 真值掩码，再喂回**同一个实现函数**。这正是必须把实现
;;; 与门面拆开的原因：单一定义下没有 %CX-ORIG 可以回调，只能调实现。
(defun %cx-reduce-dispatch (op label impl tensor axis keepdims dtype out)
  (case op
    ((:sum :prod :max :min)
     (%cx-reduce label tensor axis keepdims dtype out op))
    ((:nansum :nanprod :nanmax :nanmin)
     (%cx-nan-reduce label tensor axis keepdims dtype out
                     (ecase op (:nansum :sum) (:nanprod :prod)
                               (:nanmax :max) (:nanmin :min))))
    ((:argmax :argmin)
     (%cx-arg-reduce label tensor axis keepdims out op))
    ((:nanargmax :nanargmin)
     (%cx-nan-arg label tensor axis keepdims out op))
    ((:all :any)
     (funcall impl (%cx-bool-mask tensor) :axis axis :keepdims keepdims
              :dtype dtype :out out))
    (t (error "~a: 复数归约尚未实现的算子 ~a（CONVENTIONS §16.32）" label op))))

;;; ==================================================================
;;; 4. 后加载实现的**前向声明**
;;; ==================================================================
;;; 归属文件（elementwise / setops / reduce-stats / linalg）先于 complex*.lisp
;;; 加载，但它们的复数分支要调后者定义的内核。SBCL 在编译期看到未定义的
;;; 函数会发 UNDEFINED-FUNCTION style-warning（编译仍成功，运行期也能解析），
;;; 但那会让「编译无警告」这条基线失效 —— 以后真正的拼写错误就淹在噪声里。
;;;
;;; 这里显式声明：这些名字**保证**由 complex*.lisp 提供。声明同时起到
;;; 「契约清单」的作用：任何人删除或改名内核，都能在这份清单上对账。
(declaim (ftype function
                 %cx-op-unique        ; setops     → complex-ops
                 %cx-op-nanmedian     ; reduce-stats → complex-ops
                 %cx-op-clip          ; elementwise  → complex-ops
                 %cx-op-matrix-rank)) ; linalg       → complex-linalg
