;;;; map-reduce.lisp — 逐元素映射与归约核心
;;;
;;; 执行层选路规则（对每个核心操作提供至少两条路径）：
;;;   快路径   输入/输出连续、dtype 匹配、形状对齐 → 编译期内联循环/SIMD/并行
;;;   通用路径 strides 遍历（%inline*-strided 系列，支持任意 strides）→ 正确性兜底
;;;   vt-map   任意 dtype/形状组合 → funcall 闭包，语义基准
;;; 三条路径的输出必须逐位相同；路径选择基于运行时检查，不基于编译期假设。
;;; 连续性只影响性能，绝不影响正确性。

(in-package :clvt)

(eval-when (:compile-toplevel :load-toplevel :execute)
  (defparameter +vt-elem-types+
    '(double-float single-float
      (signed-byte 64) (signed-byte 32)
      (signed-byte 16) (signed-byte 8)
      (unsigned-byte 16) (unsigned-byte 8))
    "全部 8 种 dtype 对应的 CL 数组元素类型（顺序：宽 → 窄，命中概率降序）。")

  (defparameter +vt-mask-out-types+
    '((signed-byte 8))
    "窄整型输出类型矩阵。按 CONVENTIONS §3.1「比较/逻辑运算返回 :int8，
     取值 0/1」，布尔掩码的规范输出只有 :int8 一种；int16/uint8/uint16
     输出仅在用户显式指定 :dtype 时出现，属罕见路径，仍由 vt-map 兜底。
     矩阵保持单元素是有意的：每多一格就多一份完整内联循环，而
     vt-fast-map 在库内有 200+ 展开点，盲扩会让编译期代码量指数级放大
     （实测 8x8 全展开会直接 Heap exhausted）。")

  (defparameter +vt-strided-elem-types+
    '(double-float single-float (signed-byte 64) (signed-byte 32))
    "非连续/广播快路径支持的元素类型。窄整型的非连续运算属罕见路径，
     走 vt-map 兜底（语义不变，仅性能差异），以控制展开体积。")

  (defparameter +vt-mask-in-types+
    '(double-float single-float (signed-byte 64) (signed-byte 32))
    "窄整型输出时可配的输入类型（数据源；窄整型输入仍走同类型路径）。")

  ;; ------------------------------------------------------------------
  ;; 复数元素类型（CONVENTIONS §16.14 执行层）
  ;;
  ;; 为什么不把复数塞进 +vt-elem-types+ 的 8 路广播：vt-fast-map 在库内
  ;; 有 200+ 展开点，每个展开点都复制一份完整循环体；把 8 路扩成 10 路
  ;; 会让编译期代码量 +25%、镜像膨胀（历史实测 8x8 全展开直接 Heap exhausted）。
  ;;
  ;; 折中：**只对热算子**（+ - * / 与一元数学核）追加 2 个复数分支，
  ;; 由 +cx-inline-ops+ 白名单在宏展开期决定。实测约 30 个展开点受影响，
  ;; 换来复数算术从「泛型 walker」降到「类型声明驱动的普通循环」。
  ;; ------------------------------------------------------------------
  (defparameter +cx-elem-types+
    '((complex double-float) (complex single-float))
    "(complex128 / complex64) 的 CL 元素类型。")

  (defparameter +cx-inline-ops+
    '(+ - * / abs sqrt exp log log2 log10 sin cos tan asin acos atan
      sinh cosh tanh asinh acosh atanh square reciprocal signum
      %cop-sqrt %cop-log %cop-log2 %cop-log10 %cop-square %cop-recip
      %cop-signum %cop-abs %cconj %cop-neg
      ;; v1.3：补齐算子（复数的 expm1 / log1p / sinc / sign）
      %cop-expm1 %cop-log1p %cop-sinc %cop-sign
      %cx-eq-flag %cx-ne-flag %cx-lt-flag %cx-le-flag %cx-gt-flag %cx-ge-flag
      %cx-nan-flag %cx-inf-flag %cx-finite-flag)
    "允许生成复数内联分支的算子白名单。

     只有**语义与元素类型无关**的算子才可列入：它们在内联循环里被直接
     作用在 (complex double-float) 槽位上。未列入的算子仍走 vt-map 兜底
     （语义相同，仅性能差异，铁律 2）。")

  (defun %cx-inline-op-p (op)
    (and op (symbolp op) (member op +cx-inline-ops+ :test #'eq)))

  (defun %elem-type-cond (res-form kernel-name args &key complex)
    "生成按 RES 元素类型分派的 COND 子句列表（连续快路径）。

     COMPLEX 为真时追加 2 个复数分支（顺序在实数之后：复数命中率低，
     且实数分支更短、预测更友好）。"
    (append
     (loop for lt in +vt-elem-types+
           collect `((equal (array-element-type ,res-form) ',lt)
                     (,kernel-name ,lt ,@args)))
     (when complex
       (loop for lt in +cx-elem-types+
             collect `((equal (array-element-type ,res-form) ',lt)
                       (,kernel-name ,lt ,@args))))))

  (defun %elem-type-cond-strided (res-form kernel-name args &key complex)
    "生成按 RES 元素类型分派的 COND 子句列表（非连续/广播快路径）。"
    (append
     (loop for lt in +vt-strided-elem-types+
           collect `((equal (array-element-type ,res-form) ',lt)
                     (,kernel-name ,lt ,@args)))
     (when complex
       (loop for lt in +cx-elem-types+
             collect `((equal (array-element-type ,res-form) ',lt)
                       (,kernel-name ,lt ,@args)))))))

;;; ------------------------------------------------------------------
;;; 逐元素映射：vt-map
;;; ------------------------------------------------------------------

(defun vt-map (fn &rest args)
  "高效逐元素映射：支持标量/列表/张量混合输入并自动广播。

   参数契约（§4）：
     fn       逐元素函数，接收 n 个标量、返回一个标量（语义基准路径）
     其余参数  任意个张量/标量/嵌套序列 + 可选 :dtype / :out
     :dtype   结果 dtype；缺省按输入提升规则
     :out     输出张量。硬契约：形状与广播结果**精确相等**、
              dtype 与结果**精确相等**、必须可写（非 stride-0 广播视图）。
              违反立即报错，不静默截断。
   别名语义：out 与任一输入共享底层存储且物理区间重叠时，
     先对重叠输入做快照，保证读-写顺序与 numpy 一致（如 x+x->x 与
     滑窗重叠场景）。"
  (declare (function fn) (optimize (speed 3) (safety 0)))
  (with-float-safe
    (multiple-value-bind (tensors dtype out) (parse-vt-op-args args)
      (when (null tensors)
	(error "vt-map 至少需要一个输入张量"))
      ;; H5：:dtype 与 :out 同时给出时先查一致性
      (vt-check-out-dtype-consistency "vt-map" dtype out)
      (let* ((inputs-1 (mapcar #'ensure-vt tensors))
             ;; 别名保护：重叠输入统一先快照（快路径与通用路径共用）
             (inputs (vt-out-snapshot out inputs-1))
             (out-shape (reduce #'vt-broadcast-shapes (mapcar #'vt-shape inputs)))
             ;; 结果 dtype 由「输入提升 + 显式 :dtype」决定，**不由 out 决定**；
             ;; out 的 dtype 必须与之精确相等（§4.2 H3 严格相等契约）。
             (final-dtype (or dtype
                              (apply #'vt-promote-type
                                     (mapcar #'vt-dtype inputs))))
             ;; 硬契约：形状 + dtype + 可写（含只读广播视图）
             (res (if out
                      (vt-check-out out out-shape final-dtype :op-name "vt-map")
                      (make-vt out-shape 0 :dtype final-dtype))))
	(%vt-map-run fn inputs res out-shape)
	res))))

(defun %vt-map-run (fn inputs res out-shape)
  (declare (list inputs out-shape)
	   (optimize (speed 3)))
  (let* ((n (length inputs))
         (res-data (vt-data res))
         (res-dtype (vt-dtype res))
         (res-strides-vec (coerce (vt-strides res) 'simple-vector))
         (dims (coerce out-shape 'simple-vector))
         (rank (length out-shape))
         (size (vt-shape-to-size out-shape))
         (in-datas (coerce (mapcar #'vt-data inputs) 'simple-vector))
         (in-offsets (coerce (mapcar #'vt-offset inputs) 'simple-vector))
         (in-sizes (coerce (mapcar #'vt-size inputs) 'simple-vector))
         (in-strides (coerce (mapcar (lambda (in)
                                       (coerce (vt-broadcast-strides
                                                (vt-shape in) out-shape (vt-strides in))
                                               'simple-vector))
                                     inputs)
                             'simple-vector))
         (contig (and (vt-contiguous-p res)
                      (every (lambda (in)
                               (or (= (vt-size in) 1)
                                   (and (equal (vt-shape in) out-shape)
                                        (vt-contiguous-p in))))
                             inputs)))
         (all-same (every (lambda (in) (eq (vt-dtype in) res-dtype)) inputs))
         (res-off (vt-offset res)))
    (declare (type simple-vector res-strides-vec dims in-datas in-offsets in-sizes in-strides)
             (type fixnum n rank size res-off))

    (labels ((gather (ptrs vals)
               (loop for k fixnum from 0 below n
                     do (setf (aref vals k) (aref (svref in-datas k) (aref ptrs k)))))
             (call-fn (vals)
               (case n
                 (1 (funcall fn (aref vals 0)))
                 (2 (funcall fn (aref vals 0) (aref vals 1)))
                 (3 (funcall fn (aref vals 0) (aref vals 1) (aref vals 2)))
                 (4 (funcall fn (aref vals 0) (aref vals 1) (aref vals 2) (aref vals 3)))
                 (5 (funcall fn (aref vals 0) (aref vals 1) (aref vals 2) (aref vals 3) (aref vals 4)))
                 (otherwise (apply fn (coerce vals 'list))))))
      (macrolet ((cast-to (lt form)
		   ;; 类型声明（性能）：6 种整数存储类型必须全部走 %coerce-*，
		   ;; 否则窄整型（int8/int16/uint8/uint16）落到 (truncate REAL)
		   ;; 的泛型路径，写回 specialized-array 前还要装箱。
		   (cond ((and (consp lt) (equal lt '(signed-byte 64)))
                          `(%coerce-int64 ,form))
                         ((and (consp lt) (equal lt '(signed-byte 32)))
                          `(%coerce-int32 ,form))
                         ((and (consp lt) (equal lt '(signed-byte 16)))
                          `(%coerce-int16 ,form))
                         ((and (consp lt) (equal lt '(signed-byte 8)))
                          `(%coerce-int8 ,form))
                         ((and (consp lt) (equal lt '(unsigned-byte 16)))
                          `(%coerce-uint16 ,form))
                         ((and (consp lt) (equal lt '(unsigned-byte 8)))
                          `(%coerce-uint8 ,form))
                         ((subtypep lt 'integer)
                          `(truncate ,form))
                         (t
                          `(the ,lt (coerce ,form ',lt)))))
                 ;; 快路径：连续 + 输入/输出同类型 + n=k（k=1..5）
                 ;; 展开后与原代码逐字等价，只是把 5 份重复合并
                 (same-k (lt k)
                   (let ((ds (loop for i below k collect (gensym "D")))
                         (ps (loop for i below k collect (gensym "P")))
                         (ss (loop for i below k collect (gensym "S"))))
                     `(let ((od (the (simple-array ,lt (*)) res-data))
                            ,@(loop for d in ds for i below k
                                    collect `(,d (the (simple-array ,lt (*))
                                                      (svref in-datas ,i))))
                            ,@(loop for p in ps for i below k
                                    collect `(,p (svref in-offsets ,i)))
                            ,@(loop for s in ss for i below k
                                    collect `(,s (if (= (svref in-sizes ,i) 1) 0 1)))
                            (op res-off))
                        (declare (type (simple-array ,lt (*)) od ,@ds)
                                 (type fixnum ,@ps ,@ss op))
                        (dotimes (i size)
                          (setf (aref od op)
                                (cast-to ,lt
                                         (funcall fn
                                                  ,@(loop for d in ds for p in ps
                                                          collect `(aref ,d ,p)))))
                          (incf op)
                          ,@(loop for p in ps for s in ss collect `(incf ,p ,s))))))
                 ;; 连续但类型不同 / n>5：保留 od + cast-to
                 (contig-any (lt)
                   `(let ((od (the (simple-array ,lt (*)) res-data))
                          (ptrs (copy-seq in-offsets))
                          (vals (make-array n))
                          (steps (coerce (loop for k below n
                                               collect (if (= (svref in-sizes k) 1) 0 1))
                                         'simple-vector))
                          (op res-off))
                      (declare (type (simple-array ,lt (*)) od)
                               (type simple-vector ptrs vals steps)
                               (type fixnum op))
                      (dotimes (i size)
                        (gather ptrs vals)
                        (setf (aref od op) (cast-to ,lt (call-fn vals)))
                        (incf op)
                        (loop for k fixnum below n do
                          (incf (aref ptrs k) (aref steps k))))))
                 ;; 非连续：保留 od + cast-to
                 (noncontig (lt)
                   `(let ((od (the (simple-array ,lt (*)) res-data))
                          (ptrs (copy-seq in-offsets))
                          (vals (make-array n))
                          (indices (make-array rank :element-type 'fixnum :initial-element 0))
                          (op res-off))
                      (declare (type (simple-array ,lt (*)) od)
                               (type simple-vector ptrs vals)
                               (type (simple-array fixnum (*)) indices)
                               (type fixnum op))
                      (dotimes (i size)
                        (gather ptrs vals)
                        (setf (aref od op) (cast-to ,lt (call-fn vals)))
                        (let ((d (1- rank)))
                          (loop
                            (when (< d 0) (return))
                            (incf (aref indices d))
                            (if (< (aref indices d) (svref dims d))
                                (progn (incf op (svref res-strides-vec d))
                                       (loop for k fixnum from 0 below n do
                                         (incf (aref ptrs k)
                                               (aref (svref in-strides k) d)))
                                       (return))
                                (progn (setf (aref indices d) 0)
                                       (decf op (* (svref res-strides-vec d)
                                                   (1- (svref dims d))))
                                       (loop for k fixnum from 0 below n do
                                         (decf (aref ptrs k)
                                               (* (aref (svref in-strides k) d)
                                                  (1- (svref dims d)))))
                                       (decf d))))))))
		 (gen (lt)
		   `(cond
		      ((and contig all-same (= n 1)) (same-k ,lt 1))
		      ((and contig all-same (= n 2)) (same-k ,lt 2))
		      ((and contig all-same (= n 3)) (same-k ,lt 3))
		      ((and contig all-same (= n 4)) (same-k ,lt 4))
		      ((and contig all-same (= n 5)) (same-k ,lt 5))
		      (contig (contig-any ,lt))
		      (t (noncontig ,lt))))
		 )
        ;; 结果缓冲区按**实际**元素类型 8 路派发（先前只枚举了 4 种物理存储
        ;; 类型，窄整型结果会掉进下面那个泛型 aref 兜底分支）。
        (cond ((equal (array-element-type res-data) 'double-float)     (gen double-float))
              ((equal (array-element-type res-data) 'single-float)     (gen single-float))
              ((equal (array-element-type res-data) '(signed-byte 64)) (gen (signed-byte 64)))
              ((equal (array-element-type res-data) '(signed-byte 32)) (gen (signed-byte 32)))
              ((equal (array-element-type res-data) '(signed-byte 16)) (gen (signed-byte 16)))
              ((equal (array-element-type res-data) '(signed-byte 8))  (gen (signed-byte 8)))
              ((equal (array-element-type res-data) '(unsigned-byte 16)) (gen (unsigned-byte 16)))
              ((equal (array-element-type res-data) '(unsigned-byte 8))  (gen (unsigned-byte 8)))
              (t
	       ;; 兜底：res 底层数组类型不在 8 种之内（防御性分支，正常不可达）。
               ;; 按 dtype 回绕后写入，避免溢出写入 specialized-array 崩溃
               (let ((ptrs (copy-seq in-offsets))
                     (vals (make-array n))
                     (indices (make-array rank :element-type 'fixnum :initial-element 0))
		     (conv (vt-cast-fun res-dtype))
                     (op res-off))
                 (declare (type simple-vector ptrs vals)
                          (type (simple-array fixnum (*)) indices)
                          (type fixnum op))
                 (dotimes (i size)
                   (gather ptrs vals)
		   (setf (aref res-data op) (funcall conv (call-fn vals)))
                   (let ((d (1- rank)))
                     (loop
                       (when (< d 0) (return))
                       (incf (aref indices d))
                       (if (< (aref indices d) (svref dims d))
                           (progn (incf op (svref res-strides-vec d))
                                  (loop for k fixnum from 0 below n do
                                    (incf (aref ptrs k)
                                          (aref (svref in-strides k) d)))
                                  (return))
                           (progn (setf (aref indices d) 0)
                                  (decf op (* (svref res-strides-vec d)
                                              (1- (svref dims d))))
                                  (loop for k fixnum from 0 below n do
                                    (decf (aref ptrs k)
                                          (* (aref (svref in-strides k) d)
                                             (1- (svref dims d)))))
                                  (decf d))))))))))
      res)))

;;; ------------------------------------------------------------------
;;; 二元特化（内部，供算术层使用）
;;; ------------------------------------------------------------------

(defun vt-binary (fn t1 t2 &key out dtype)
  "二元逐元素运算（内部入口，等价于两参数 vt-map）。"
  (declare (optimize (speed 3) (safety 0)))
  (apply #'vt-map fn (ensure-vt t1) (ensure-vt t2) :out out :dtype dtype))

;;; ------------------------------------------------------------------
;;; 编译期内联逐元素映射（对标原 vt-fast-map，消除 funcall 装箱）
;;; ------------------------------------------------------------------

(defmacro %cast-to (lt form)
  ;; "内联类型转换：整数截断，浮点 coerce。"
  "内联类型转换：整数类型走安全 coerce/回绕，浮点 coerce。

   类型声明（性能）：必须覆盖**全部 6 种**整数存储类型。此前只特化
   (signed-byte 64)/32，其余整数（int8/int16/uint8/uint16）落到
   `(truncate ,form)` —— 参数是 REAL 时 `truncate` 只能走 GENERIC-TRUNCATE，
   返回 INTEGER 后还要装箱才能写回 specialized-array。这是比较运算
   （结果 :int8 布尔掩码）比同类算术慢 20x 的根因。改为走
   dtype.lisp 里已声明返回类型的 %coerce-* 后，写回点可直接证出
   元素类型，bounds/范围检查与装箱一并消失。"
  (cond ((equal lt '(signed-byte 64))   `(%coerce-int64  ,form))
        ((equal lt '(signed-byte 32))   `(%coerce-int32  ,form))
        ((equal lt '(signed-byte 16))   `(%coerce-int16  ,form))
        ((equal lt '(signed-byte 8))    `(%coerce-int8   ,form))
        ((equal lt '(unsigned-byte 16)) `(%coerce-uint16 ,form))
        ((equal lt '(unsigned-byte 8))  `(%coerce-uint8  ,form))
        ((subtypep lt 'integer)         `(truncate ,form))
        (t                              `(coerce ,form ',lt))))

(defmacro %inline1-loop (lt op a res)
  (declare (optimize (speed 3) (safety 0)))
  "一元内联循环（lt 为类型，op 为算子符号）。"
  `(let ((od (the (simple-array ,lt (*)) (vt-data ,res)))
         (d0 (the (simple-array ,lt (*)) (vt-data ,a)))
         (p0 (vt-offset ,a))
         (s0 (if (= (vt-size ,a) 1) 0 1))
         (op (vt-offset ,res)))
     (declare (type (simple-array ,lt (*)) od d0)
	      (type fixnum p0 s0 op))
     (loop for i fixnum from 0 below (vt-size ,res) do
       (setf (aref od op) (%cast-to ,lt (,op (aref d0 p0))))
       (incf op) (incf p0 s0))))

(defmacro %inline2-loop (lt op a b res)
  "二元内联循环（lt 为类型，op 为算子符号）。"
  (declare (optimize (speed 3) (safety 0)))
  `(let ((od (the (simple-array ,lt (*)) (vt-data ,res)))
         (d0 (the (simple-array ,lt (*)) (vt-data ,a)))
         (d1 (the (simple-array ,lt (*)) (vt-data ,b)))
         (p0 (vt-offset ,a)) (p1 (vt-offset ,b))
         (s0 (if (= (vt-size ,a) 1) 0 1))
         (s1 (if (= (vt-size ,b) 1) 0 1))
         (op (vt-offset ,res)))
     (declare (type (simple-array ,lt (*)) od d0 d1)
	      (type fixnum p0 p1 s0 s1 op))
     (loop for i fixnum from 0 below (vt-size ,res) do
       (setf (aref od op) (%cast-to ,lt (,op (aref d0 p0) (aref d1 p1))))
       (incf op) (incf p0 s0) (incf p1 s1))))

;;; ------------------------------------------------------------------
;;; 非连续 + 广播内联路径
;;; 与 %inline1/2-loop 的区别：用 strides 遍历，支持非连续/广播输入
;;; 与 vt-map noncontig 的区别：op 是宏展开的符号，无 funcall 装箱
;;; ------------------------------------------------------------------

(defmacro %inline1-strided (lt op a res)
  "一元 strides 遍历（内联 op）。支持任意 strides：输入按广播规则取 stride（长度 1 的轴 stride=0 虚拟重复读），
  结果张量 RES 按其自身 strides 逐元素写入——连续与非连续 RES 均正确，连续 RES 仅因访存局部性而更快。
  优化路径与通用路径（vt-map）的输出必须逐位相同；连续性只影响选路，绝不影响正确性。"
  `(let* ((od (the (simple-array ,lt (*)) (vt-data ,res)))
          (d0 (the (simple-array ,lt (*)) (vt-data ,a)))
          (rs (coerce (vt-strides ,res) 'simple-vector))
          (s0 (coerce (vt-broadcast-strides
                       (vt-shape ,a) (vt-shape ,res) (vt-strides ,a))
                      'simple-vector))
          (dims (coerce (vt-shape ,res) 'simple-vector))
          (rank (length dims))
          (p0 (vt-offset ,a))
          (op (vt-offset ,res)))
     (declare (type (simple-array ,lt (*)) od d0)
              (type simple-vector rs s0 dims)
              (type fixnum rank p0 op))
     (if (= rank 0)
         (setf (aref od op) (%cast-to ,lt (,op (aref d0 p0))))
         (unless (= (the fixnum (vt-size ,res)) 0) ; 0-size 结果（如 (0) op (1) 广播）直接跳过，禁止先读后进
           (let ((idx (make-array rank :element-type 'fixnum :initial-element 0)))
           (declare (type (simple-array fixnum (*)) idx))
           (block outer
             (loop
               (setf (aref od op) (%cast-to ,lt (,op (aref d0 p0))))
               (let ((d (1- rank)))
                 (loop
                   (incf (aref idx d))
                   (incf p0 (svref s0 d))
                   (incf op (svref rs d))
                   (when (< (aref idx d) (svref dims d)) (return))
                   (let ((dim (svref dims d)))
                     (decf p0 (* dim (svref s0 d)))
                     (decf op (* dim (svref rs d)))
                     (setf (aref idx d) 0)
                     (decf d)
                     (when (< d 0) (return-from outer))))))))))))

(defmacro %inline2-strided (lt op a b res)
  "二元 strides 遍历（内联 op）。支持任意 strides：两个输入按广播规则取 stride（长度 1 的轴 stride=0），
  结果张量 RES 按其自身 strides 逐元素写入——连续与非连续 RES 均正确，连续 RES 仅因访存局部性而更快。
  优化路径与通用路径（vt-map）的输出必须逐位相同；连续性只影响选路，绝不影响正确性。"
  `(let* ((od (the (simple-array ,lt (*)) (vt-data ,res)))
          (d0 (the (simple-array ,lt (*)) (vt-data ,a)))
          (d1 (the (simple-array ,lt (*)) (vt-data ,b)))
          (rs (coerce (vt-strides ,res) 'simple-vector))
          (s0 (coerce (vt-broadcast-strides
                       (vt-shape ,a) (vt-shape ,res) (vt-strides ,a))
                      'simple-vector))
          (s1 (coerce (vt-broadcast-strides
                       (vt-shape ,b) (vt-shape ,res) (vt-strides ,b))
                      'simple-vector))
          (dims (coerce (vt-shape ,res) 'simple-vector))
          (rank (length dims))
          (p0 (vt-offset ,a)) (p1 (vt-offset ,b))
          (op (vt-offset ,res)))
     (declare (type (simple-array ,lt (*)) od d0 d1)
              (type simple-vector rs s0 s1 dims)
              (type fixnum rank p0 p1 op))
     (if (= rank 0)
         (setf (aref od op) (%cast-to ,lt (,op (aref d0 p0) (aref d1 p1))))
         (unless (= (the fixnum (vt-size ,res)) 0) ; 0-size 结果（如 (0) op (1) 广播）直接跳过
           (let ((idx (make-array rank :element-type 'fixnum :initial-element 0)))
           (declare (type (simple-array fixnum (*)) idx))
           (block outer
             (loop
               (setf (aref od op)
                     (%cast-to ,lt (,op (aref d0 p0) (aref d1 p1))))
               (let ((d (1- rank)))
                 (loop
                   (incf (aref idx d))
                   (incf p0 (svref s0 d))
                   (incf p1 (svref s1 d))
                   (incf op (svref rs d))
                   (when (< (aref idx d) (svref dims d)) (return))
                   (let ((dim (svref dims d)))
                     (decf p0 (* dim (svref s0 d)))
                     (decf p1 (* dim (svref s1 d)))
                     (decf op (* dim (svref rs d)))
                     (setf (aref idx d) 0)
                     (decf d)
                     (when (< d 0) (return-from outer))))))))))))

(defmacro %inline3-loop (lt op a b c res)
  "三元连续内联循环。"
  `(let ((od (the (simple-array ,lt (*)) (vt-data ,res)))
         (d0 (the (simple-array ,lt (*)) (vt-data ,a)))
         (d1 (the (simple-array ,lt (*)) (vt-data ,b)))
         (d2 (the (simple-array ,lt (*)) (vt-data ,c)))
         (p0 (vt-offset ,a)) (p1 (vt-offset ,b)) (p2 (vt-offset ,c))
         (s0 (if (= (vt-size ,a) 1) 0 1))
         (s1 (if (= (vt-size ,b) 1) 0 1))
         (s2 (if (= (vt-size ,c) 1) 0 1))
         (op (vt-offset ,res)))
     (declare (type (simple-array ,lt (*)) od d0 d1 d2)
              (type fixnum p0 p1 p2 s0 s1 s2 op))
     (loop for i fixnum from 0 below (vt-size ,res) do
       (setf (aref od op)
             (%cast-to ,lt (,op (aref d0 p0) (aref d1 p1) (aref d2 p2))))
       (incf op) (incf p0 s0) (incf p1 s1) (incf p2 s2))))

(defmacro %inline3-strided (lt op a b c res)
  ;; 三元 strides 遍历：同 %inline1/2-strided，支持任意 strides（含非连续 RES）。
  "三元 strides 遍历。"
  `(let* ((od (the (simple-array ,lt (*)) (vt-data ,res)))
          (d0 (the (simple-array ,lt (*)) (vt-data ,a)))
          (d1 (the (simple-array ,lt (*)) (vt-data ,b)))
          (d2 (the (simple-array ,lt (*)) (vt-data ,c)))
          (rs (coerce (vt-strides ,res) 'simple-vector))
          (s0 (coerce (vt-broadcast-strides
                       (vt-shape ,a) (vt-shape ,res) (vt-strides ,a))
                      'simple-vector))
          (s1 (coerce (vt-broadcast-strides
                       (vt-shape ,b) (vt-shape ,res) (vt-strides ,b))
                      'simple-vector))
          (s2 (coerce (vt-broadcast-strides
                       (vt-shape ,c) (vt-shape ,res) (vt-strides ,c))
                      'simple-vector))
          (dims (coerce (vt-shape ,res) 'simple-vector))
          (rank (length dims))
          (p0 (vt-offset ,a)) (p1 (vt-offset ,b)) (p2 (vt-offset ,c))
          (op (vt-offset ,res)))
     (declare (type (simple-array ,lt (*)) od d0 d1 d2)
              (type simple-vector rs s0 s1 s2 dims)
              (type fixnum rank p0 p1 p2 op))
     (if (= rank 0)
         (setf (aref od op)
               (%cast-to ,lt (,op (aref d0 p0) (aref d1 p1) (aref d2 p2))))
         (unless (= (the fixnum (vt-size ,res)) 0) ; 0-size 结果直接跳过
           (let ((idx (make-array rank :element-type 'fixnum :initial-element 0)))
           (declare (type (simple-array fixnum (*)) idx))
           (block outer
             (loop
               (setf (aref od op)
                     (%cast-to ,lt (,op (aref d0 p0) (aref d1 p1) (aref d2 p2))))
               (let ((d (1- rank)))
                 (loop
                   (incf (aref idx d))
                   (incf p0 (svref s0 d))
                   (incf p1 (svref s1 d))
                   (incf p2 (svref s2 d))
                   (incf op (svref rs d))
                   (when (< (aref idx d) (svref dims d)) (return))
                   (let ((dim (svref dims d)))
                     (decf p0 (* dim (svref s0 d)))
                     (decf p1 (* dim (svref s1 d)))
                     (decf p2 (* dim (svref s2 d)))
                     (decf op (* dim (svref rs d)))
                     (setf (aref idx d) 0)
                     (decf d)
                     (when (< d 0) (return-from outer))))))))))))

;;; ==================================================================
;;; 元素类型派发表（编译期）
;;;
;;; 全部 8 种逻辑 dtype 的物理元素类型。这是「声明驱动」优化的中枢：
;;; 只有把 (the (simple-array <lt> (*)) ...) 真正落到每个数组变量上，
;;; SBCL 才会生成特化的 AREF 而不是泛型路径（实测泛型路径慢 20x）。
;;; 此前只列举了 4 种物理存储类型，凡是结果 dtype 为
;;; int8/int16/uint8/uint16 的运算（比较运算的布尔掩码、布尔逻辑、
;;; isnan/isfinite 族）都会掉到最后那个 (t ...) 分支回到 vt-map。
;;; ==================================================================

(defmacro %vt-inline1-strided-fast (op a res)
  "一元非连续/广播快路径（实数 4 路 + 复数 2 路派发，内联 op）。"
  (let ((rd (gensym "RD")))
    `(let ((,rd (vt-data ,res)))
       (cond ,@(%elem-type-cond-strided rd '%inline1-strided (list op a res)
                                        :complex (%cx-inline-op-p op))
             (t (vt-map (function ,op) ,a :out ,res :dtype (vt-dtype ,res)))))))

(defmacro %vt-inline2-strided-fast (op a b res)
  "二元非连续/广播快路径（实数 4 路 + 复数 2 路派发，内联 op）。"
  (let ((rd (gensym "RD")))
    `(let ((,rd (vt-data ,res)))
       (cond ,@(%elem-type-cond-strided rd '%inline2-strided (list op a b res)
                                        :complex (%cx-inline-op-p op))
             (t (vt-map (function ,op) ,a ,b :out ,res :dtype (vt-dtype ,res)))))))

(defmacro %vt-inline3-fast (op a b c res)
  "三元连续快路径（实数 8 路 + 复数 2 路派发，内联 op）。"
  (let ((rd (gensym "RD")))
    `(let ((,rd (vt-data ,res)))
       (cond ,@(%elem-type-cond rd '%inline3-loop (list op a b c res)
                                :complex (%cx-inline-op-p op))
             (t (vt-map (function ,op) ,a ,b ,c :out ,res :dtype (vt-dtype ,res)))))))

(defmacro %vt-inline3-strided-fast (op a b c res)
  "三元非连续/广播快路径（实数 4 路 + 复数 2 路派发，内联 op）。"
  (let ((rd (gensym "RD")))
    `(let ((,rd (vt-data ,res)))
       (cond ,@(%elem-type-cond-strided rd '%inline3-strided (list op a b c res)
                                        :complex (%cx-inline-op-p op))
             (t (vt-map (function ,op) ,a ,b ,c :out ,res :dtype (vt-dtype ,res)))))))

(defmacro %vt-inline1-fast (op a res)
  "一元连续快路径（实数 8 路 + 复数 2 路派发，内联 op）。"
  (let ((rd (gensym "RD")))
    `(let ((,rd (vt-data ,res)))
       (cond ,@(%elem-type-cond rd '%inline1-loop (list op a res)
                                :complex (%cx-inline-op-p op))
             (t (vt-map (function ,op) ,a :out ,res :dtype (vt-dtype ,res)))))))

(defmacro %vt-inline2-fast (op a b res)
  "二元连续快路径（实数 8 路 + 复数 2 路派发，内联 op）。"
  (let ((rd (gensym "RD")))
    `(let ((,rd (vt-data ,res)))
       (cond ,@(%elem-type-cond rd '%inline2-loop (list op a b res)
                                :complex (%cx-inline-op-p op))
             (t (vt-map (function ,op) ,a ,b :out ,res :dtype (vt-dtype ,res)))))))

;;; ==================================================================
;;; 混合类型快路径：窄整型输出（布尔掩码）
;;;
;;; 为什么必须单独一组：`%vt-inline*-fast` 的前置条件是
;;; 「全部输入与输出的元素类型相同」（dtype-check），而比较/判定类函数
;;; 天然是 float64/int64 进、int8 出 —— 永远进不去那条路，只能回退
;;; vt-map 的 funcall + 泛型 aref。实测该回退在 1M float64 上
;;; vt-< 达 51ms、vt-isclose 达 445ms（比 numpy 慢 270x）。
;;;
;;; 本组把「输入类型」与「输出类型」两组类型声明并置，读写两端同时特化；
;;; 矩阵限制在真实会出现的组合（4 种数据源 × 4 种窄输出 = 16 格），
;;; 避免 8×8 全展开把编译期代码量放大数十倍。
;;; 无法命中时仍回退 vt-map，语义不变（铁律 2）。
;;; ==================================================================

(defmacro %mask1-loop (out-lt in-lt op a res)
  "窄整型输出的一元连续内联循环：读 IN-LT → 算 OP → 按 OUT-LT 回绕写回。"
  `(let ((od (the (simple-array ,out-lt (*)) (vt-data ,res)))
         (d0 (the (simple-array ,in-lt (*)) (vt-data ,a)))
         (p0 (vt-offset ,a))
         (s0 (if (= (vt-size ,a) 1) 0 1))
         (n (the fixnum (vt-size ,res)))
         (op (vt-offset ,res)))
     (declare (type (simple-array ,out-lt (*)) od)
              (type (simple-array ,in-lt (*)) d0)
              (type fixnum p0 s0 n op))
     (dotimes (i n)
       (declare (type fixnum i))
       (setf (aref od op) (%cast-to ,out-lt (,op (aref d0 p0))))
       (incf op) (incf p0 s0))))

(defmacro %mask2-loop (out-lt in-lt op a b res)
  "窄整型输出的二元连续内联循环。"
  `(let ((od (the (simple-array ,out-lt (*)) (vt-data ,res)))
         (d0 (the (simple-array ,in-lt (*)) (vt-data ,a)))
         (d1 (the (simple-array ,in-lt (*)) (vt-data ,b)))
         (p0 (vt-offset ,a)) (p1 (vt-offset ,b))
         (s0 (if (= (vt-size ,a) 1) 0 1))
         (s1 (if (= (vt-size ,b) 1) 0 1))
         (n (the fixnum (vt-size ,res)))
         (op (vt-offset ,res)))
     (declare (type (simple-array ,out-lt (*)) od)
              (type (simple-array ,in-lt (*)) d0 d1)
              (type fixnum p0 p1 s0 s1 n op))
     (dotimes (i n)
       (declare (type fixnum i))
       (setf (aref od op) (%cast-to ,out-lt (,op (aref d0 p0) (aref d1 p1))))
       (incf op) (incf p0 s0) (incf p1 s1))))

(defmacro %mask2-strided (out-lt in-lt op a b res)
  "窄整型输出 + 任意 strides（含广播）的二元内联循环。"
  `(let* ((od (the (simple-array ,out-lt (*)) (vt-data ,res)))
          (d0 (the (simple-array ,in-lt (*)) (vt-data ,a)))
          (d1 (the (simple-array ,in-lt (*)) (vt-data ,b)))
          (rs (coerce (vt-strides ,res) 'simple-vector))
          (s0 (coerce (vt-broadcast-strides
                       (vt-shape ,a) (vt-shape ,res) (vt-strides ,a))
                      'simple-vector))
          (s1 (coerce (vt-broadcast-strides
                       (vt-shape ,b) (vt-shape ,res) (vt-strides ,b))
                      'simple-vector))
          (dims (coerce (vt-shape ,res) 'simple-vector))
          (rank (the fixnum (length dims)))
          (p0 (vt-offset ,a)) (p1 (vt-offset ,b))
          (op (vt-offset ,res)))
     (declare (type (simple-array ,out-lt (*)) od)
              (type (simple-array ,in-lt (*)) d0 d1)
              (type simple-vector rs s0 s1 dims)
              (type fixnum rank p0 p1 op))
     (if (= rank 0)
         (setf (aref od op) (%cast-to ,out-lt (,op (aref d0 p0) (aref d1 p1))))
         (unless (= (the fixnum (vt-size ,res)) 0)
           (let ((idx (make-array rank :element-type 'fixnum :initial-element 0)))
             (declare (type (simple-array fixnum (*)) idx))
             (block outer
               (loop
                 (setf (aref od op)
                       (%cast-to ,out-lt (,op (aref d0 p0) (aref d1 p1))))
                 (let ((d (1- rank)))
                   (declare (type fixnum d))
                   (loop
                     (incf (aref idx d))
                     (incf p0 (svref s0 d))
                     (incf p1 (svref s1 d))
                     (incf op (svref rs d))
                     (when (< (aref idx d) (svref dims d)) (return))
                     (let ((dim (svref dims d)))
                       (decf p0 (* dim (svref s0 d)))
                       (decf p1 (* dim (svref s1 d)))
                       (decf op (* dim (svref rs d)))
                       (setf (aref idx d) 0)
                       (decf d)
                       (when (< d 0) (return-from outer))))))))))))

(defmacro %vt-mask1-fast (op a res)
  "一元「窄整型输出」快路径（4 种实数输入 + 2 种复数输入）。"
  (let ((rd (gensym "RD")) (ea (gensym "EA")))
    `(let* ((,rd (vt-data ,res))
            (,ea (array-element-type (vt-data ,a))))
       (cond
         ,@(loop for olt in +vt-mask-out-types+
                 nconc (loop for ilt in (if (%cx-inline-op-p op)
                                             (append +vt-mask-in-types+ +cx-elem-types+)
                                             +vt-mask-in-types+)
                             collect
                             `((and (equal (array-element-type ,rd) ',olt)
                                    (equal ,ea ',ilt))
                               (%mask1-loop ,olt ,ilt ,op ,a ,res))))
         (t (vt-map (function ,op) ,a :out ,res :dtype (vt-dtype ,res)))))))

(defmacro %vt-mask2-fast (op a b res)
  "二元「窄整型输出」连续快路径（4 种实数输入 + 2 种复数输入）。"
  (let ((rd (gensym "RD")) (ea (gensym "EA")) (eb (gensym "EB")))
    `(let* ((,rd (vt-data ,res))
            (,ea (array-element-type (vt-data ,a)))
            (,eb (array-element-type (vt-data ,b))))
       (cond
         ,@(loop for olt in +vt-mask-out-types+
                 nconc (loop for ilt in (if (%cx-inline-op-p op)
                                             (append +vt-mask-in-types+ +cx-elem-types+)
                                             +vt-mask-in-types+)
                             collect
                             `((and (equal (array-element-type ,rd) ',olt)
                                    (equal ,ea ',ilt)
                                    (equal ,eb ',ilt))
                               (%mask2-loop ,olt ,ilt ,op ,a ,b ,res))))
         (t (vt-map (function ,op) ,a ,b :out ,res :dtype (vt-dtype ,res)))))))

(defmacro %vt-mask2-strided-fast (op a b res)
  "二元「窄整型输出」+ 非连续/广播快路径（4 种实数输入 + 2 种复数输入）。"
  (let ((rd (gensym "RD")) (ea (gensym "EA")) (eb (gensym "EB")))
    `(let* ((,rd (vt-data ,res))
            (,ea (array-element-type (vt-data ,a)))
            (,eb (array-element-type (vt-data ,b))))
       (cond
         ,@(loop for olt in +vt-mask-out-types+
                 nconc (loop for ilt in (if (%cx-inline-op-p op)
                                             (append +vt-mask-in-types+ +cx-elem-types+)
                                             +vt-mask-in-types+)
                             collect
                             `((and (equal (array-element-type ,rd) ',olt)
                                    (equal ,ea ',ilt)
                                    (equal ,eb ',ilt))
                               (%mask2-strided ,olt ,ilt ,op ,a ,b ,res))))
         (t (vt-map (function ,op) ,a ,b :out ,res :dtype (vt-dtype ,res)))))))

(defmacro vt-fast-map (fn &rest args)
  "编译期内联已知算子的逐元素映射（一元/二元/三元）。
   优先走内联路径，任何维度/连续性组合都有对应快路径；
   仅在 dtype 不匹配或参数个数 > 3 时回退到 vt-map。

   参数契约：与 vt-map 完全一致（§4），校验逻辑统一委托给
   vt-check-out / vt-out-snapshot / vt-check-out-dtype-consistency，
   本宏内**不再**自行拼装检查（避免多份实现漂移）。"
  (declare (optimize (speed 3) (safety 0)))
  (let ((op (and (consp fn)
                 (eq (car fn) 'function)
                 (symbolp (cadr fn))
                 (cadr fn))))
    (if (null op)
        `(apply #'vt-map ,fn ,@args)
        (multiple-value-bind (tensors dtype out)         
            (parse-vt-op-args args)
          (declare (list tensors))
          (let* ((n (length tensors))
                 (tvs (loop repeat n collect (gensym "TV"))))
            (if (not (or (= n 1) (= n 2) (= n 3)))
                `(apply #'vt-map ,fn ,@args)
                `(with-float-safe                                 
                   (let (,@(loop for tv in tvs for tf in tensors
                                 collect `(,tv (ensure-vt ,tf))))
                     ;; H5：:dtype 与 :out 冲突（统一入口）
                     (vt-check-out-dtype-consistency "vt-fast-map" ,dtype ,out)
                     (let* ((out-shape (reduce #'vt-broadcast-shapes
                                               (mapcar #'vt-shape (list ,@tvs))))
                            ;; 结果 dtype 一律由「输入提升 + 显式 :dtype」决定，
                            ;; **不由 out 决定**。这样 out 的 dtype 必须精确匹配，
                            ;; 杜绝 float64 结果静默降精度写入 float32 out
                            ;; （§4.2 H3「严格相等」契约）。
                            (final-dtype (or ,dtype
                                             (apply #'vt-promote-type
                                                    (mapcar #'vt-dtype
                                                            (list ,@tvs)))))
                            ;; 硬契约 H1–H4：形状/dtype/可写（统一入口）
                            (res (if ,out
                                     (vt-check-out ,out out-shape final-dtype
                                                   :op-name "vt-fast-map")
                                     (make-vt out-shape 0 :dtype final-dtype))))
                       ;; 别名保护统一走 vt-out-snapshot；重叠时退回 vt-map 全局快照路径
		       (if (and ,out
                                (some (lambda (in) (%vt-views-overlap-p ,out in))
                                      (list ,@tvs)))
                           (vt-map (function ,op) ,@tvs :out ,out :dtype final-dtype)
			   ,(let* ((dtype-check
                                     `(and ,@(loop for tv in tvs
						   collect `(eq (vt-dtype ,tv) (vt-dtype res)))))
				   ;; 「可内联连续循环」的判定（类型声明驱动）。
				   ;; 注意 (or (= (vt-size ,tv) 1) ...)：标量/0 维/单元素输入
				   ;; 走的是 stride=0 的虚拟重复读，`%inline*-loop` 已按
				   ;; (if (= (vt-size a) 1) 0 1) 显式支持。
				   ;; 此前要求「输入形状必须等于输出形状」，于是
				   ;; (vt-+ 1M张量 2.5d0) 这种最常见的标量广播被判定为
				   ;; 不连续，掉进 strides/odometer 路径，实测慢 20x。
				   (simple-check
                                     `(and (vt-contiguous-p res)
					   ,@(loop for tv in tvs
						   collect `(or (= (vt-size ,tv) 1)
								 (and (vt-contiguous-p ,tv)
								      (equal (vt-shape ,tv) out-shape))))))
				   (maskable
                                     `(member (vt-dtype res)
					      '(:int8 :int16 :uint8 :uint16))))
                              (cond
				((= n 1)
				 `(cond
				    (,dtype-check
				     (if ,simple-check
					 (%vt-inline1-fast ,op ,(first tvs) res)
					 (%vt-inline1-strided-fast ,op ,(first tvs) res)))
				    ((and ,simple-check ,maskable)
				     (%vt-mask1-fast ,op ,(first tvs) res))
				    (t (vt-map (function ,op) ,(first tvs)
					       :out res :dtype final-dtype))))
				((= n 2)
				 `(cond
				    (,dtype-check
				     (if ,simple-check
					 (%vt-inline2-fast ,op ,(first tvs) ,(second tvs) res)
					 (%vt-inline2-strided-fast ,op ,(first tvs) ,(second tvs) res)))
				    (,maskable
				     (if ,simple-check
					 (%vt-mask2-fast ,op ,(first tvs) ,(second tvs) res)
					 (%vt-mask2-strided-fast ,op ,(first tvs) ,(second tvs) res)))
				    (t (vt-map (function ,op) ,(first tvs) ,(second tvs)
					       :out res :dtype final-dtype))))
				(t
				 `(if ,dtype-check
                                      (if ,simple-check
					  (%vt-inline3-fast ,op ,(first tvs) ,(second tvs) ,(third tvs) res)
					  (%vt-inline3-strided-fast ,op ,(first tvs) ,(second tvs) ,(third tvs) res))
                                      (vt-map (function ,op) ,(first tvs) ,(second tvs) ,(third tvs) :out res :dtype final-dtype))))))
                       res)))))))))

;;; ------------------------------------------------------------------
;;; 归约核心：vt-reduce
;;; ------------------------------------------------------------------

(defun get-reduction-identity (op element-type)
  "返回指定归约操作在给定元素类型下的初始值。"
  (declare (optimize (speed 3) (safety 0)))
  (case op
    (:sum (coerce 0 element-type))
    (:max (cond ((eq element-type 'double-float) +vt-dfloat-neg-inf+)
                ((eq element-type 'single-float) +vt-sfloat-neg-inf+)
                ((equal element-type '(signed-byte 64)) (- (expt 2 63)))
                ((equal element-type '(signed-byte 32)) (- (expt 2 31)))
                ((equal element-type '(unsigned-byte 64)) 0)
                ((equal element-type '(unsigned-byte 32)) 0)
                ((equal element-type '(unsigned-byte 16)) 0)
                ((equal element-type '(unsigned-byte 8)) 0)
                ((subtypep element-type 'integer) most-negative-fixnum)
                (t 0)))
    (:min (cond ((eq element-type 'double-float) +vt-dfloat-pos-inf+)
                ((eq element-type 'single-float) +vt-sfloat-pos-inf+)
                ((equal element-type '(signed-byte 64)) (1- (expt 2 63)))
                ((equal element-type '(signed-byte 32)) (1- (expt 2 31)))
                ((equal element-type '(unsigned-byte 64)) (1- (expt 2 64)))
                ((equal element-type '(unsigned-byte 32)) (1- (expt 2 32)))
                ((equal element-type '(unsigned-byte 16)) (1- (expt 2 16)))
                ((equal element-type '(unsigned-byte 8)) (1- (expt 2 8)))
                ((subtypep element-type 'integer) most-positive-fixnum)
                (t 0)))))

(defun vt-reduce (tensor axis init-val reducer-fn &key out dtype keepdims return-arg)
  "沿 AXIS 归约的核心原语（内部入口，语义基准）。

   参数个数与默认值（任务3 §6）：
     tensor      必需，张量/标量/序列，经 ensure-vt 归一
     axis        必需，nil=全局归约；整数或整数列表；负值按 numpy 规则折算
     init-val    必需，累加初值（必须与 reducer-fn 的语义匹配，如 sum→0，
                 max→-Inf/-most-negative，min→+Inf/+most-positive）
     reducer-fn  必需，二元归约函数；返回 (values acc update-arg-p)
     :out        可选。**硬契约**：形状必须精确等于归约结果形状（含
                 keepdims 后的形状）、dtype 必须精确等于结果 dtype、
                 必须可写。非连续 out 由本函数按真实 strides 写入。
     :dtype      可选。结果 dtype；与 :out 同时给出时必须一致（H5）。
     :keepdims   可选，保留被归约轴为长度 1。
     :return-arg 可选，同时返回 argmax/argmin 索引张量（dtype int32，本内部参数专用）。

   结果 dtype 由「输入 dtype + 显式 :dtype + init-val 类型」决定，
   **不由 out 决定**（§4.2 H3 严格相等契约）。"
  (declare (type vt tensor)
           (type (or null fixnum list) axis)
           (type function reducer-fn))
  (setf tensor (ensure-vt tensor))
  ;; H5：:dtype 与 :out 同时给出时必须一致；此处提前拦截，
  ;;     避免后续各分支重复实现（原实现仅在通用路径里检查一次）。
  (vt-check-out-dtype-consistency "vt-reduce" dtype out)
  (with-float-safe
    (let* ((in-shape (vt-shape tensor))
           (rank (length in-shape))
           (axes (vt-normalize-axes axis rank))
           (global (null axes))
           (out-shape (cond ((and global (not keepdims)) nil)
                            (global (make-list rank :initial-element 1))
                            ((not keepdims)
                             (loop for d in in-shape
                                   for i fixnum from 0
                                   unless (member i axes) collect d))
                            (t (loop for d in in-shape
                                     for i fixnum from 0
                                     collect (if (member i axes) 1 d)))))
           (axis-size (if axes
                          (reduce #'* (mapcar (lambda (a) (nth a in-shape)) axes)
                                  :initial-value 1)
                          (reduce #'* in-shape :initial-value 1))))
      (declare (fixnum rank axis-size))

      ;; 空输入：整个归约退化为「用 init-val 填充结果」。
      ;; 此处同样必须尊重 :out 契约（形状/dtype/可写），否则空输入会成为
      ;; 绕过硬契约的后门（原实现直接 make-vt、静默丢弃用户传入的 out）。
      (when (or (zerop axis-size) (zerop (vt-size tensor)))
        (let* ((empty-dtype (or dtype (vt-dtype tensor)))
               (empty-res (cond ((null out)
                                 (make-vt out-shape (or init-val 0) :dtype empty-dtype))
                                ;; out 已给出：校验后原地填充（支持非连续视图）
                                (t (vt-check-out out out-shape empty-dtype
                                                 :op-name "vt-reduce")
                                    (vt-fill out (or init-val 0))
                                    out))))
          (return-from vt-reduce
            (values empty-res
                    (when return-arg (make-vt out-shape 0 :dtype :int32))))))

      (let* ((final-dtype (cond
                            (dtype dtype)
                            ((and init-val (or (floatp init-val) (%inf-p init-val)))
                             :float64)
                            (t (vt-dtype tensor))))
	     ;; 硬契约：形状/dtype/可写统一由 vt-check-out 把关，
	     ;; 不再在此处散落 (error "vt-reduce: ...") 检查。
	     (res (if out
	              (vt-check-out out out-shape final-dtype :op-name "vt-reduce")
	              (make-vt out-shape 0 :dtype final-dtype))))

        ;; ================================================================
        ;; 快路径 1：连续 + 全局归约（axis=nil, 不 keepdims, 无 arg）
        ;; ================================================================
        (when (and global (not keepdims) (not return-arg)
                   (numberp init-val)
                   (vt-contiguous-p tensor)
                   ;; v0.4.1 修复（差分测试 R1）：自定义 reducer-fn 必须走
                   ;; 通用路径 —— 此前快路径 1 按 init-val 分派内建
                   ;; sum/max/min，任意 fn 的全局归约都被静默替换为求和。
                   ;; 仅当 fn 与内建语义严格对应时才允许快路径。
                   (or (and (eq reducer-fn #'+) (zerop init-val))
                       (and (eq reducer-fn #'max)
                            (= init-val +vt-dfloat-neg-inf+))
                       (and (eq reducer-fn #'min)
                            (= init-val +vt-dfloat-pos-inf+))))
          (let* ((size (vt-size tensor))
                 (in-data (vt-data tensor))
                 (in-off (vt-offset tensor))
                 (in-et (array-element-type in-data))
                 (res-data (vt-data res))
                 (res-off (vt-offset res)))
            (declare (type fixnum size in-off res-off))
            (labels ((store-fast (val)
                       (setf (aref (the (simple-array * (*)) res-data) res-off)
                             (vt-cast val final-dtype))
                       (return-from vt-reduce (values res nil))))
              (cond
                ((equal in-et 'double-float)
                 (let ((d (the (simple-array double-float (*)) in-data)))
                   (cond
                     ((and (typep init-val 'double-float) (zerop init-val))
                      (let ((acc 0.0d0)
			    (p in-off)
			    (end (the fixnum (+ in-off size))))
                        (declare (type double-float acc) (type fixnum p end))
                        (loop while (< p end) do (incf acc (aref d p)) (incf p))
                        (store-fast acc)))
                     ((= init-val +vt-dfloat-neg-inf+)
                      (let ((acc +vt-dfloat-neg-inf+)
			    (p in-off)
			    (end (the fixnum (+ in-off size))))
                        (declare (type double-float acc) (type fixnum p end))
                        (loop while (< p end) do                      
			  (let ((v (aref d p)))
                            (cond ((%nan-p v) (setf acc v) (return))
                                  ((> v acc) (setf acc v))))
                          (incf p))			
                        (store-fast acc)))
		     ((= init-val +vt-dfloat-pos-inf+)
		      (let ((acc +vt-dfloat-pos-inf+)
			    (p in-off)
			    (end (the fixnum (+ in-off size))))
			(declare (type double-float acc) (type fixnum p end))
			(loop while (< p end) do
			  (let ((v (aref d p)))
			    (cond ((%nan-p v) (setf acc v) (return))
				  ((< v acc) (setf acc v))))
			  (incf p))
			(store-fast acc))))))
                ((equal in-et 'single-float)
                 (let ((d (the (simple-array single-float (*)) in-data)))
                   (when (and (typep init-val 'single-float) (zerop init-val))
                     (let ((acc 0.0s0)
			   (p in-off)
			   (end (the fixnum (+ in-off size))))
                       (declare (type single-float acc) (type fixnum p end))
                       (loop while (< p end)
			     do (incf acc (aref d p))
				(incf p))
                       (store-fast acc)))))
                ((equal in-et '(signed-byte 32))
                 (when (eql init-val 0)
                   (let ((d (the (simple-array (signed-byte 32) (*)) in-data))
                         (acc 0)
			 (p in-off)
			 (end (the fixnum (+ in-off size))))
                     (declare (type (signed-byte 32) acc) (type fixnum p end))
                     (loop while (< p end)
			   do (incf acc (aref d p))
			      (incf p))
                     (store-fast acc))))
                ((equal in-et '(signed-byte 64))
                 (when (eql init-val 0)
                   (let ((d (the (simple-array (signed-byte 64) (*)) in-data))
                         (acc 0)
			 (p in-off)
			 (end (the fixnum (+ in-off size))))
                     (declare (type (signed-byte 64) acc) (type fixnum p end))
                     (loop while (< p end)
			   do (incf acc (aref d p))
			      (incf p))
                     (store-fast acc))))))))

        ;; ================================================================
        ;; 快路径 2：连续 + 末尾连续归约轴（无 arg）——消除 odometer
        ;; ================================================================
        (when (and (not global) (not return-arg)
                   (numberp init-val)
                   (vt-contiguous-p tensor)
                   (vt-contiguous-p res)
                   (loop for a in axes
                         for i from (- rank (length axes))
                         always (= a i)))
          (let* ((inner (the fixnum axis-size))
                 (outer (the fixnum (truncate (the fixnum (vt-size tensor)) inner)))
                 (in-data (vt-data tensor))
                 (in-off (vt-offset tensor))
                 (in-et (array-element-type in-data))
                 (res-data (vt-data res))
                 (res-off (vt-offset res))
                 (res-et (array-element-type res-data)))
            (macrolet ((cast-to (lt form)
                         (if (subtypep lt 'integer)
                             `(truncate ,form)
                             `(coerce ,form ',lt)))
                       (acc-init (lt form)
                         (if (subtypep lt 'integer)
                             `(the ,lt (truncate ,form))
                             `(the ,lt (coerce ,form ',lt))))
                       ;; out-lt: 输出元素类型；in-lt: 输入元素类型；
                       ;; acc-lt: 累加器类型（必须是 in-lt 与 out-lt 的提升类型）
                       ;; init-acc 在 outer 循环外只算一次（原版每次外层迭代都算一次）
                       (blk (out-lt in-lt acc-lt op)
                         `(progn
                            (let ((ip in-off) (opos res-off)
					      (init-acc (acc-init ,acc-lt init-val)))
                              (declare (type fixnum ip opos)
                                       (type ,acc-lt init-acc))
                              (dotimes (o outer)
                                (let ((acc init-acc))
                                  (declare (type ,acc-lt acc))
                                  (dotimes (i inner)
                                    (let ((v (aref (the (simple-array ,in-lt (*)) in-data) ip)))
                                      ,(ecase op
					 (:sum `(incf acc v))
					 (:max `(cond ((%nan-p v)
                                                       (setf acc (cast-to ,acc-lt v)))
                                                      ((> v acc)
                                                       (setf acc (cast-to ,acc-lt v)))))
					 (:min `(cond ((%nan-p v)
                                                       (setf acc (cast-to ,acc-lt v)))
                                                      ((< v acc)
                                                       (setf acc (cast-to ,acc-lt v)))))))
                                    (incf ip))
                                  (setf (aref (the (simple-array ,out-lt (*)) res-data) opos)
                                        (cast-to ,out-lt acc))
                                  (incf opos))))
                            (return-from vt-reduce (values res nil)))))
              (cond
                ;; ---- SUM ----
                ((eq reducer-fn #'+)
                 (cond
                   ((and (equal res-et 'double-float) (equal in-et 'double-float))
                    (blk double-float double-float double-float :sum))
                   ((and (equal res-et 'double-float) (equal in-et 'single-float))
                    (blk double-float single-float double-float :sum))
                   ((and (equal res-et 'double-float) (equal in-et '(signed-byte 64)))
                    (blk double-float (signed-byte 64) double-float :sum))
                   ((and (equal res-et 'double-float) (equal in-et '(signed-byte 32)))
                    (blk double-float (signed-byte 32) double-float :sum))
                   ((and (equal res-et 'single-float) (equal in-et 'single-float))
                    (blk single-float single-float single-float :sum))
                   ((and (equal res-et 'single-float) (equal in-et 'double-float))
                    (blk single-float double-float double-float :sum))
                   ((and (equal res-et '(signed-byte 64)) (equal in-et '(signed-byte 64)))
                    (blk (signed-byte 64) (signed-byte 64) (signed-byte 64) :sum))
                   ((and (equal res-et '(signed-byte 32)) (equal in-et '(signed-byte 32)))
                    (blk (signed-byte 32) (signed-byte 32) (signed-byte 32) :sum))))
                ;; ---- MAX ----
                ((eq reducer-fn #'max)
                 (cond
                   ((and (equal res-et 'double-float) (equal in-et 'double-float))
                    (blk double-float double-float double-float :max))
                   ((and (equal res-et 'double-float) (equal in-et 'single-float))
                    (blk double-float single-float double-float :max))
                   ((and (equal res-et 'single-float) (equal in-et 'single-float))
                    (blk single-float single-float single-float :max))
                   ((and (equal res-et 'single-float) (equal in-et 'double-float))
                    (blk single-float double-float double-float :max))
                   ((and (equal res-et '(signed-byte 64)) (equal in-et '(signed-byte 64)))
                    (blk (signed-byte 64) (signed-byte 64) (signed-byte 64) :max))
                   ((and (equal res-et '(signed-byte 32)) (equal in-et '(signed-byte 32)))
                    (blk (signed-byte 32) (signed-byte 32) (signed-byte 32) :max))))
                ;; ---- MIN ----
                ((eq reducer-fn #'min)
                 (cond
                   ((and (equal res-et 'double-float) (equal in-et 'double-float))
                    (blk double-float double-float double-float :min))
                   ((and (equal res-et 'double-float) (equal in-et 'single-float))
                    (blk double-float single-float double-float :min))
                   ((and (equal res-et 'single-float) (equal in-et 'single-float))
                    (blk single-float single-float single-float :min))
                   ((and (equal res-et 'single-float) (equal in-et 'double-float))
                    (blk single-float double-float double-float :min))
                   ((and (equal res-et '(signed-byte 64)) (equal in-et '(signed-byte 64)))
                    (blk (signed-byte 64) (signed-byte 64) (signed-byte 64) :min))
                   ((and (equal res-et '(signed-byte 32)) (equal in-et '(signed-byte 32)))
                    (blk (signed-byte 32) (signed-byte 32) (signed-byte 32) :min))))))))

        ;; ================================================================
        ;; 通用路径：任意轴 / keepdims / return-arg / 非连续
        ;; ================================================================
        (let* ((res-data (vt-data res))
               (res-offset (vt-offset res))
               (res-strides (vt-strides res))
               (res-idx (when return-arg (make-vt out-shape 0 :dtype :int32)))
               (res-idx-data (when res-idx (vt-data res-idx)))
               (res-idx-offset (when res-idx (vt-offset res-idx)))
               (res-idx-strides (when res-idx (vt-strides res-idx)))
               (in-data (vt-data tensor))
               (in-strides (vt-strides tensor))
               (in-offset (vt-offset tensor))
               (in-shape-vec (coerce in-shape 'simple-vector))
               (in-strides-vec (coerce in-strides 'simple-vector))
               (out-strides-map
                 (if global
                     (make-list rank :initial-element 0)
                     (loop for i from 0 below rank
                           if (member i axes) collect 0
                             else collect
				  (let ((out-idx (if keepdims i
                                                     (count-if-not (lambda (x) (member x axes))
								   (loop for j below i collect j)))))
				    (nth out-idx res-strides)))))
               (idx-strides-map
                 (if (or (not return-arg) global)
                     (make-list rank :initial-element 0)
                     (loop for i from 0 below rank
                           if (member i axes) collect 0
                             else collect
				  (let ((out-idx (if keepdims i
                                                     (count-if-not (lambda (x) (member x axes))
								   (loop for j below i collect j)))))
				    (nth out-idx res-idx-strides)))))
               (arg-strides
                 (if return-arg
                     (if global
                         (vt-compute-strides in-shape)
                         (let* ((red-shape (mapcar (lambda (a) (nth a in-shape)) axes))
                                (red-strides (vt-compute-strides red-shape))
                                (k -1))
                           (declare (fixnum k))
                           (loop for i fixnum below rank
                                 if (member i axes)
                                   collect (progn (incf k) (nth k red-strides))
                                 else collect 0)))
                     (make-list rank :initial-element 0)))
               (out-strides-vec (coerce out-strides-map 'simple-vector))
               (idx-strides-vec (coerce idx-strides-map 'simple-vector))
               (arg-strides-vec (coerce arg-strides 'simple-vector))
               (out-et (array-element-type res-data))
               (in-et (array-element-type in-data)))

          ;; 优化：res 为新建（make-vt 已零填充）且 init-val=0 时，跳过全量填充
          ;; res-idx 恒为新建零填充，无需再 fill
          (unless (and (null out) (numberp init-val) (zerop init-val))
            (vt-fill res init-val))

          (when (= rank 0)
            (let ((val (aref in-data in-offset)))
              (multiple-value-bind (new-acc do-update-arg)
                  (funcall reducer-fn init-val val)
                (setf (aref res-data res-offset) (vt-cast new-acc final-dtype))
                (if return-arg
                    (progn
                      (when do-update-arg
                        (setf (aref res-idx-data res-idx-offset) 0))
                      (return-from vt-reduce (values res res-idx)))
                    (return-from vt-reduce (values res nil))))))

          (macrolet
              ((cast-to (lt form)
                 (cond ((null lt) form)
                       ((subtypep lt 'integer) `(truncate ,form))
                       (t `(coerce ,form ',lt))))
               (define-loop (out-lt in-lt red-type with-arg)
                 (let ((out-array (if out-lt
                                      `(the (simple-array ,out-lt (*)) res-data)
                                      'res-data))
                       (in-array (if in-lt
                                     `(the (simple-array ,in-lt (*)) in-data)
                                     'in-data)))
                   `(let ((indices (make-array rank :element-type 'fixnum :initial-element 0)))
                      (declare (type (simple-array fixnum (*)) indices))
                      (let ((in-ptr in-offset)
                            (out-ptr res-offset)
                            ,@(when with-arg `((arg-ptr res-idx-offset) (arg-val 0))))
                        (declare (type fixnum in-ptr out-ptr
                                       ,@(when with-arg '(arg-ptr arg-val))))
                        (block outer-loop
                          (loop
                            (let* ((val (aref ,in-array in-ptr))
                                   (raw-acc (aref ,out-array out-ptr)))
                              ,(ecase red-type
                                 (:sum `(setf (aref ,out-array out-ptr)
                                              (cast-to ,out-lt (+ raw-acc val))))
				 (:max `(cond ((%nan-p val)
                                               (setf (aref ,out-array out-ptr)
                                                     (cast-to ,out-lt val))
                                               ,(when with-arg
                                                  `(setf (aref res-idx-data arg-ptr) arg-val)))
                                              ((%nan-p raw-acc) nil)
                                              ((> val raw-acc)
                                               (setf (aref ,out-array out-ptr)
                                                     (cast-to ,out-lt val))
                                               ,(when with-arg
                                                  `(setf (aref res-idx-data arg-ptr) arg-val)))))
                                 (:min `(cond ((%nan-p val)
                                               (setf (aref ,out-array out-ptr)
                                                     (cast-to ,out-lt val))
                                               ,(when with-arg
                                                  `(setf (aref res-idx-data arg-ptr) arg-val)))
                                              ((%nan-p raw-acc) nil)
                                              ((< val raw-acc)
                                               (setf (aref ,out-array out-ptr)
                                                     (cast-to ,out-lt val))
                                               ,(when with-arg
                                                  `(setf (aref res-idx-data arg-ptr) arg-val)))))
                                 (:custom
                                  (if with-arg
                                      `(multiple-value-bind (new-acc do-update-arg)
                                           (funcall reducer-fn raw-acc val)
                                         (setf (aref ,out-array out-ptr)
                                               (cast-to ,out-lt new-acc))
                                         (when do-update-arg
                                           (setf (aref res-idx-data arg-ptr) arg-val)))
                                      `(multiple-value-bind (new-acc)
                                           (funcall reducer-fn raw-acc val)
                                         (setf (aref ,out-array out-ptr)
                                               (cast-to ,out-lt new-acc)))))))
                            (let ((d (1- rank)))
                              (loop
                                (incf (aref indices d))
                                (incf in-ptr (svref in-strides-vec d))
                                (incf out-ptr (svref out-strides-vec d))
                                ,@(when with-arg
                                    `((incf arg-ptr (svref idx-strides-vec d))
                                      (incf arg-val (svref arg-strides-vec d))))
                                (when (< (aref indices d) (svref in-shape-vec d))
                                  (return))
                                (let ((dim (svref in-shape-vec d)))
                                  (decf in-ptr (* dim (svref in-strides-vec d)))
                                  (decf out-ptr (* dim (svref out-strides-vec d)))
                                  ,@(when with-arg
                                      `((decf arg-ptr (* dim (svref idx-strides-vec d)))
                                        (decf arg-val (* dim (svref arg-strides-vec d)))))
                                  (setf (aref indices d) 0)
                                  (decf d)
                                  (when (< d 0)
                                    (return-from outer-loop)))))))))))
               (generate-typed-cond (with-arg)
                 `(cond
                    ((eq reducer-fn #'+)
                     (cond
                       ((equal out-et 'double-float)
                        (cond ((equal in-et 'double-float) (define-loop double-float double-float :sum ,with-arg))
                              ((equal in-et 'single-float) (define-loop double-float single-float :sum ,with-arg))
                              ((equal in-et '(signed-byte 64)) (define-loop double-float (signed-byte 64) :sum ,with-arg))
                              ((equal in-et '(signed-byte 32)) (define-loop double-float (signed-byte 32) :sum ,with-arg))
                              (t (define-loop double-float nil :sum ,with-arg))))
                       ((equal out-et 'single-float)
                        (cond ((equal in-et 'single-float) (define-loop single-float single-float :sum ,with-arg))
                              ((equal in-et 'double-float) (define-loop single-float double-float :sum ,with-arg))
                              ((equal in-et '(signed-byte 64)) (define-loop single-float (signed-byte 64) :sum ,with-arg))
                              ((equal in-et '(signed-byte 32)) (define-loop single-float (signed-byte 32) :sum ,with-arg))
                              (t (define-loop single-float nil :sum ,with-arg))))
                       ((equal out-et '(signed-byte 64))
                        (cond ((equal in-et '(signed-byte 64)) (define-loop (signed-byte 64) (signed-byte 64) :sum ,with-arg))
                              ((equal in-et '(signed-byte 32)) (define-loop (signed-byte 64) (signed-byte 32) :sum ,with-arg))
                              ((equal in-et 'double-float)     (define-loop (signed-byte 64) double-float     :sum ,with-arg))
                              ((equal in-et 'single-float)     (define-loop (signed-byte 64) single-float     :sum ,with-arg))
                              (t (define-loop (signed-byte 64) nil :sum ,with-arg))))
                       ((equal out-et '(signed-byte 32))
                        (cond ((equal in-et '(signed-byte 32)) (define-loop (signed-byte 32) (signed-byte 32) :sum ,with-arg))
                              ((equal in-et '(signed-byte 64)) (define-loop (signed-byte 32) (signed-byte 64) :sum ,with-arg))
                              ((equal in-et 'double-float)     (define-loop (signed-byte 32) double-float     :sum ,with-arg))
                              ((equal in-et 'single-float)     (define-loop (signed-byte 32) single-float     :sum ,with-arg))
                              (t (define-loop (signed-byte 32) nil :sum ,with-arg))))
                       (t (define-loop nil nil :sum ,with-arg))))
                    ((eq reducer-fn #'max)
                     (cond
                       ((equal out-et 'double-float)
                        (cond ((equal in-et 'double-float) (define-loop double-float double-float :max ,with-arg))
                              ((equal in-et 'single-float) (define-loop double-float single-float :max ,with-arg))
                              ((equal in-et '(signed-byte 64)) (define-loop double-float (signed-byte 64) :max ,with-arg))
                              ((equal in-et '(signed-byte 32)) (define-loop double-float (signed-byte 32) :max ,with-arg))
                              (t (define-loop double-float nil :max ,with-arg))))
                       ((equal out-et 'single-float)
                        (cond ((equal in-et 'single-float) (define-loop single-float single-float :max ,with-arg))
                              ((equal in-et 'double-float) (define-loop single-float double-float :max ,with-arg))
                              ((equal in-et '(signed-byte 64)) (define-loop single-float (signed-byte 64) :max ,with-arg))
                              ((equal in-et '(signed-byte 32)) (define-loop single-float (signed-byte 32) :max ,with-arg))
                              (t (define-loop single-float nil :max ,with-arg))))
                       ((equal out-et '(signed-byte 64))
                        (cond ((equal in-et '(signed-byte 64)) (define-loop (signed-byte 64) (signed-byte 64) :max ,with-arg))
                              ((equal in-et '(signed-byte 32)) (define-loop (signed-byte 64) (signed-byte 32) :max ,with-arg))
                              ((equal in-et 'double-float)     (define-loop (signed-byte 64) double-float     :max ,with-arg))
                              ((equal in-et 'single-float)     (define-loop (signed-byte 64) single-float     :max ,with-arg))
                              (t (define-loop (signed-byte 64) nil :max ,with-arg))))
                       ((equal out-et '(signed-byte 32))
                        (cond ((equal in-et '(signed-byte 32)) (define-loop (signed-byte 32) (signed-byte 32) :max ,with-arg))
                              ((equal in-et '(signed-byte 64)) (define-loop (signed-byte 32) (signed-byte 64) :max ,with-arg))
                              ((equal in-et 'double-float)     (define-loop (signed-byte 32) double-float     :max ,with-arg))
                              ((equal in-et 'single-float)     (define-loop (signed-byte 32) single-float     :max ,with-arg))
                              (t (define-loop (signed-byte 32) nil :max ,with-arg))))
                       (t (define-loop nil nil :max ,with-arg))))
                    ((eq reducer-fn #'min)
                     (cond
                       ((equal out-et 'double-float)
                        (cond ((equal in-et 'double-float) (define-loop double-float double-float :min ,with-arg))
                              ((equal in-et 'single-float) (define-loop double-float single-float :min ,with-arg))
                              ((equal in-et '(signed-byte 64)) (define-loop double-float (signed-byte 64) :min ,with-arg))
                              ((equal in-et '(signed-byte 32)) (define-loop double-float (signed-byte 32) :min ,with-arg))
                              (t (define-loop double-float nil :min ,with-arg))))
                       ((equal out-et 'single-float)
                        (cond ((equal in-et 'single-float) (define-loop single-float single-float :min ,with-arg))
                              ((equal in-et 'double-float) (define-loop single-float double-float :min ,with-arg))
                              ((equal in-et '(signed-byte 64)) (define-loop single-float (signed-byte 64) :min ,with-arg))
                              ((equal in-et '(signed-byte 32)) (define-loop single-float (signed-byte 32) :min ,with-arg))
                              (t (define-loop single-float nil :min ,with-arg))))
                       ((equal out-et '(signed-byte 64))
                        (cond ((equal in-et '(signed-byte 64)) (define-loop (signed-byte 64) (signed-byte 64) :min ,with-arg))
                              ((equal in-et '(signed-byte 32)) (define-loop (signed-byte 64) (signed-byte 32) :min ,with-arg))
                              ((equal in-et 'double-float)     (define-loop (signed-byte 64) double-float     :min ,with-arg))
                              ((equal in-et 'single-float)     (define-loop (signed-byte 64) single-float     :min ,with-arg))
                              (t (define-loop (signed-byte 64) nil :min ,with-arg))))
                       ((equal out-et '(signed-byte 32))
                        (cond ((equal in-et '(signed-byte 32)) (define-loop (signed-byte 32) (signed-byte 32) :min ,with-arg))
                              ((equal in-et '(signed-byte 64)) (define-loop (signed-byte 32) (signed-byte 64) :min ,with-arg))
                              ((equal in-et 'double-float)     (define-loop (signed-byte 32) double-float     :min ,with-arg))
                              ((equal in-et 'single-float)     (define-loop (signed-byte 32) single-float     :min ,with-arg))
                              (t (define-loop (signed-byte 32) nil :min ,with-arg))))
                       (t (define-loop nil nil :min ,with-arg))))
                    (t
                     (cond
                       ((equal out-et 'double-float)
                        (cond ((equal in-et 'double-float) (define-loop double-float double-float :custom ,with-arg))
                              ((equal in-et 'single-float) (define-loop double-float single-float :custom ,with-arg))
                              ((equal in-et '(signed-byte 64)) (define-loop double-float (signed-byte 64) :custom ,with-arg))
                              ((equal in-et '(signed-byte 32)) (define-loop double-float (signed-byte 32) :custom ,with-arg))
                              (t (define-loop double-float nil :custom ,with-arg))))
                       ((equal out-et 'single-float)
                        (cond ((equal in-et 'single-float) (define-loop single-float single-float :custom ,with-arg))
                              ((equal in-et 'double-float) (define-loop single-float double-float :custom ,with-arg))
                              ((equal in-et '(signed-byte 64)) (define-loop single-float (signed-byte 64) :custom ,with-arg))
                              ((equal in-et '(signed-byte 32)) (define-loop single-float (signed-byte 32) :custom ,with-arg))
                              (t (define-loop single-float nil :custom ,with-arg))))
                       ((equal out-et '(signed-byte 64))
                        (cond ((equal in-et '(signed-byte 64)) (define-loop (signed-byte 64) (signed-byte 64) :custom ,with-arg))
                              ((equal in-et '(signed-byte 32)) (define-loop (signed-byte 64) (signed-byte 32) :custom ,with-arg))
                              ((equal in-et 'double-float)     (define-loop (signed-byte 64) double-float     :custom ,with-arg))
                              ((equal in-et 'single-float)     (define-loop (signed-byte 64) single-float     :custom ,with-arg))
                              (t (define-loop (signed-byte 64) nil :custom ,with-arg))))
                       ((equal out-et '(signed-byte 32))
                        (cond ((equal in-et '(signed-byte 32)) (define-loop (signed-byte 32) (signed-byte 32) :custom ,with-arg))
                              ((equal in-et '(signed-byte 64)) (define-loop (signed-byte 32) (signed-byte 64) :custom ,with-arg))
                              ((equal in-et 'double-float)     (define-loop (signed-byte 32) double-float     :custom ,with-arg))
                              ((equal in-et 'single-float)     (define-loop (signed-byte 32) single-float     :custom ,with-arg))
                              (t (define-loop (signed-byte 32) nil :custom ,with-arg))))
                       (t (define-loop nil nil :custom ,with-arg)))))))
            (if return-arg
                (generate-typed-cond t)
                (generate-typed-cond nil)))
          (values res res-idx))))))
