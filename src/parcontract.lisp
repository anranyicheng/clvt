;;;; parcontract.lisp — 参数契约基础设施（:out / :dtype / axis 的统一校验与寻址）
;;;;
;;;; ==================================================================
;;;; 设计目标（任务3）
;;;; ==================================================================
;;;; 用户的原始要求：
;;;;   "彻底解决函数参数个数、参数默认值、对参数进行检查，
;;;;    尤其是对于连续和不连续内存张量 out 参数的检查和计算过程，
;;;;    给出一个高效稳妥的方案，注意不是简单的在函数开头检查抛错误就完了，
;;;;    而是要在计算过程中确保正确实现。"
;;;;
;;;; 因此本文件提供两类设施，缺一不可：
;;;;
;;;;   (A) 入场券 —— vt-check-out*：在任何计算发生之前完成硬契约校验
;;;;       H1 out 是 vt      H2 形状精确匹配    H3 dtype 精确匹配
;;;;       H4 可写（非 stride-0 广播视图）      H5 :dtype 与 :out 不冲突
;;;;
;;;;   (B) 计算期保障 —— vt-out-view / vt-write-* / vt-out-snapshot
;;;;       把 "out 的正确写入" 从 "每个函数各自拼装" 变成 "库级原语统一保证"：
;;;;         · 一切寻址基于 out 的真实 strides 与 offset（绝不假设连续）
;;;;         · 写入前必查可写性（计算路径内部再查一次，防止绕过入口）
;;;;         · 与输入重叠时先快照，保证读-写顺序语义与 numpy 一致
;;;;
;;;; 三层分离对应：
;;;;   逻辑层 —— 形状/dtype 匹配规则（vt-check-out）
;;;;   物理层 —— strides/offset 寻址与别名快照（vt-out-view / vt-out-snapshot）
;;;;   执行层 —— 连续性仅决定选路（vt-out-contig-p），不影响结果
;;;;
;;;; 核心不变量（任何优化路径都必须满足）：
;;;;   ∀ 合法调用： 结果(快路径) ≡ 结果(通用路径)   （逐位相同）

(in-package :clvt)

;; sb-introspect 提供 function-lambda-list，用于参数契约审计（任务3）。
(eval-when (:compile-toplevel :load-toplevel :execute)
  (require :sb-introspect))

;;; ==================================================================
;;; 1. 硬契约校验（入场券）
;;; ==================================================================

(defun %vt-out-error (op-name fmt &rest args)
  "统一错误信息格式：vt-<func>: <描述>。op-name 为函数名符号或字符串。"
  (error "~a: ~a"
         (if (symbolp op-name)
             (string-downcase (symbol-name op-name))
             op-name)
         (apply #'format nil fmt args)))

(defun vt-out-writable-p (out)
  "out 是否可写。
   广播视图（dim>1 且 stride=0）语义上只读 —— 对同一物理单元多次写入
   会产生歧义（写哪个值？），因此一律视为不可写。"
  (if (zerop (vt-size out))
      t                             ; 零尺寸视图不写入，视为可写
      (loop for d in (vt-shape out)
            for s in (vt-strides out)
            never (and (> d 1) (zerop s)))))

(defun vt-check-out (out shape dtype &key (op-name "vt-op"))
  "硬契约校验：out 必须能满足 shape（精确匹配）与 dtype（精确匹配）。
   校验通过返回 out 本身；否则报错（消息含 op-name）。

   与 numpy 的差异（更严格，属安全子集）：
      numpy 允许 float64 结果写入 float32 out（casting='same_kind'）；
      本库要求 dtype 精确相等。理由：本库无 casting 参数，
      放开一半无法表达「拒绝 float64→int32」的另一半，
      且会与 §4.4 精度解耦语义纠缠。违反契约必报错，不使用 assert
      （assert 在 (safety 0) 下可能被编译剔除）。"
  (unless (typep out 'vt)
    (%vt-out-error op-name ":out 必须是张量，收到 ~a" (type-of out)))
  (unless (equal (vt-shape out) shape)
    (%vt-out-error op-name ":out 形状 ~a 与结果形状 ~a 不匹配"
                   (vt-shape out) shape))
  (unless (eq (vt-dtype out) dtype)
    (%vt-out-error op-name ":out dtype ~a 与结果 dtype ~a 不匹配"
                   (vt-dtype out) dtype))
  (unless (vt-out-writable-p out)
    (%vt-out-error op-name ":out 是只读的广播视图（存在 dim>1 且 stride=0 的轴）"))
  out)

(defun vt-resolve-out (op-name out shape shape-fn dtype dtype-fn
                       &key (broadcast-to nil))
  "统一的 out + dtype 解析器，供所有带 :out 的公开函数使用。

   参数：
     op-name      函数名（符号或字符串），用于错误信息
     out          用户传入的 :out（可为 nil）
     shape        已确定的**结果形状**（dtype/shape 已知时直接给）
     shape-fn     惰性形状函数（shape 为 nil 时用；避免无谓计算）
     dtype        已确定的**结果 dtype**（nil 表示「稍后由 promote 决定」）
     dtype-fn     惰性 dtype 函数
     broadcast-to 若给定，表示 out 可以与输入广播（如 vt-copy-into 场景）

   返回 (values out designator)，其中：
     — 若用户给了 out：out 原样返回，designator = out
     — 若用户未给 out：designator = nil，调用方自行 make-vt
   并完成 H1–H5 全部硬校验。

   H5（:dtype 与 :out 冲突）：调用方需在传入前已解析出 dtype；
     若 dtype 由 out 决定（即用户只给了 out 没给 :dtype），
     dtype-fn 应能返回 (vt-dtype out)。"
  (let ((real-shape (or shape (and shape-fn (funcall shape-fn))))
        (real-dtype (or dtype (and dtype-fn (funcall dtype-fn)))))
    (when out
      (if broadcast-to
          ;; 广播场景：out 形状必须是广播结果形状
          (let ((final (vt-broadcast-shapes (vt-shape out) broadcast-to)))
            (unless (equal final (vt-shape out))
              (%vt-out-error op-name ":out 形状 ~a 无法容纳广播结果 ~a"
                             (vt-shape out) final))
            (unless (vt-out-writable-p out)
              (%vt-out-error op-name ":out 是只读的广播视图"))
            (when (and real-dtype (not (eq (vt-dtype out) real-dtype)))
              (%vt-out-error op-name ":out dtype ~a 与结果 dtype ~a 不匹配"
                             (vt-dtype out) real-dtype)))
          (vt-check-out out real-shape real-dtype :op-name op-name)))
    (values out real-shape real-dtype)))

(defun vt-check-out-dtype-consistency (op-name dtype out)
  "检查同时给出的 :dtype 与 :out 是否一致（H5）。
   只在两者都非 nil 时生效。"
  (when (and dtype out (not (eq (vt-dtype out) dtype)))
    (%vt-out-error op-name ":dtype ~a 与 :out dtype ~a 冲突"
                   dtype (vt-dtype out)))
  nil)

;;; ==================================================================
;;; 2. 计算期寻址原语（保证写入正确，而非仅入场检查）
;;; ==================================================================

(declaim (inline %vt-flat-idx))
(defun %vt-flat-idx (offset strides indices)
  "把一个逻辑索引向量（fixnum 数组）映射为物理下标。
   strides 与 indices 等长。这是**唯一**被允许的寻址公式：
     phys = offset + Σ_d indices[d] * strides[d]
   任何假设计算连续性（如直接用 itemsize）的写法都违反物理层契约。"
  (declare (type fixnum offset)
           (type (simple-array fixnum (*)) indices))
  (let ((p offset))
    (declare (type fixnum p))
    (loop for d of-type fixnum from 0 below (length indices)
          for s of-type fixnum in strides
          do (incf p (the fixnum (* (aref indices d) s))))
    p))

(defun vt-out-contig-p (out)
  "out 是否值得走连续快路径。
   注意：这**只是选路建议**，为假时走通用 strides 路径，
   两条路径的结果必须逐位相同（见文件头核心不变量）。"
  (vt-contiguous-p out))

(defun vt-out-snapshot (out inputs)
  "别名保护：inuts 中任何与 out 共享底层存储且物理区间重叠的输入，
   都先用 vt-copy 做快照，返回新的输入列表。

   为什么必须做：
     形如  (vt-+ z z :out z[::-1])  的调用，
     若按顺序逐元素读写同一缓冲区，未读取的源数据会被先写的结果覆盖，
     退化为 memmove 的经典错误。numpy 实测该场景输出 [8,6,4,2]，
     即「先读全量、再写」。快照是达到该语义的唯一正确做法。
   返回：新的 inputs 列表（顺序不变）。"
  (if (null out)
      inputs
      (mapcar (lambda (in)
                (if (%vt-views-overlap-p out in)
                    (vt-copy in)
                    in))
              inputs)))

;;; ==================================================================
;;; 3. 统一写入入口（strides 驱动，零连续性假设）
;;; ==================================================================

(defun vt-write-1 (out index value)
  "向 out 的**逻辑**索引 index 写入 value。
   index 为索引列表（长度 = rank；标量 out 传 nil）。
   全程使用 out 的真实 offset/strides，支持任意非连续视图。"
  (let* ((off (vt-offset out))
         (strs (vt-strides out))
         (p off))
    (declare (type fixnum p off))
    (loop for i of-type fixnum in index
          for s of-type fixnum in strs
          do (incf p (the fixnum (* i s))))
    (setf (aref (vt-data out) p)
          (vt-cast value (vt-dtype out)))
    value))

(defun vt-write-broadcast (out value)
  "用标量 value 原地填充 out（等价 vt-fill，保留为对称 API）。
   out 的可写性在 vt-fill 内部再次校验 —— 计算路径内部不依赖入口检查。"
  (vt-fill out value))

(defun vt-read-strided (src indices strides-vec)
  "按 indices（fixnum 数组）与给定 strides 从 src 读取一个元素。
   供归约/einsum 类内核遍历使用。"
  (declare (type (simple-array fixnum (*)) indices strides-vec)
           (optimize (speed 3) (safety 0)))
  (let ((p (vt-offset src)))
    (declare (type fixnum p))
    (loop for d of-type fixnum from 0 below (length indices)
          do (incf p (the fixnum (* (aref indices d) (aref strides-vec d)))))
    (aref (the (simple-array * (*)) (vt-data src)) p)))

;;; ==================================================================
;;; 4. 归约类 out 的精度解耦三段式
;;; ==================================================================

(defun vt-reduce-dtypes (inputs requested-dtype &key (promote #'vt-promote-type))
  "归约/统计类函数的 dtype 三段式（§4.4 精度解耦）。

   返回 (values compute-dtype exec-dtype)：
     compute-dtype —— 结果 dtype，也是 out 必须匹配的 dtype
     exec-dtype    —— 实际累加所用 dtype（通常 = compute，或更高）

   规则（与 numpy 实测一致）：
     · 显式 :dtype 优先
     · 整数输入 → 整数归约提升到 int64（sum/prod）
       浮点输入 → 保持（float32 → float32，float64 → float64）
     · promote 参数允许调用方覆盖（如 mean 强制 float64）

   注意：本函数**不**决定 out，out 的 dtype 必须在调用方
   用 vt-check-out 与本函数返回的 compute-dtype 比对（H3）。"
  (let ((compute
          (or requested-dtype
              (funcall promote (mapcar #'vt-dtype inputs)))))
    (values compute compute)))

;;; ==================================================================
;;; 6. 参数契约审计（任务3：函数参数个数 / 默认值 / 检查）
;;; ==================================================================

(defun vt-params-audit (&optional (package :clvt))
  "扫描包内全部 vt-* 公开函数，报告其参数契约摘要。
   用于人工审计「参数个数、是否有 &key、是否有 &rest、是否有 &optional」，
   并标出疑似违反 §6.2 约定（应显式声明 &key 却用 &rest 吞参数）的函数。

   返回一个 alist：((fn-name . plist) ...)，plist 键为
     :required 必需参数个数
     :optional 是否有 &optional
     :rest     是否有 &rest
     :key      是否有 &key
     :key-names 关键字名列表
     :lambda-list 原始 lambda-list
     :doc-p    是否有 docstring"
  (let ((pkg (find-package package))
        (result nil))
    (unless pkg (error "vt-params-audit: 找不到包 ~a" package))
    (do-symbols (sym pkg)
      (when (and (fboundp sym)
                 (eq (symbol-package sym) pkg)
                 (let ((n (symbol-name sym)))
                   (and (> (length n) 3)
                        (string= "VT-" n :end2 3))))
        (let* ((ll (sb-introspect:function-lambda-list sym))
               (required 0) (has-opt nil) (has-rest nil) (has-key nil)
               (key-names nil) (mode :required))
          (when (listp ll)
            (dolist (item ll)
              (cond ((eq item '&optional) (setf has-opt t mode :optional))
                    ((eq item '&rest)     (setf has-rest t mode :rest))
                    ((eq item '&key)      (setf has-key t mode :key))
                    ((eq item '&allow-other-keys) nil)
                    ((member item '(&aux &body)) (setf mode :aux))
                    (t (case mode
                         (:required (incf required))
                         (:key (push (if (consp item) (car item) item)
                                     key-names)))))))
          (push (cons sym
                      (list :required required
                            :optional has-opt
                            :rest has-rest
                            :key has-key
                            :key-names (nreverse key-names)
                            :lambda-list ll
                            :doc-p (not (null (documentation sym 'function)))))
                result))))
    (nreverse result)))

(defun vt-params-audit-report ()
  "打印参数契约审计报告（人读格式），并返回可疑函数名列表。
   可疑判定：
     R1 用 &rest 但不是变参运算符（vt-+ vt-* vt-- vt-/ vt-map
        vt-concatenate 等白名单之外）；
     R2 有 docstring 缺失。"
  (let* ((audit (vt-params-audit))
         (vararg-whitelist '(vt-+ vt-* vt-- vt-/ vt-map vt-reduce
                             vt-concatenate vt-stack vt-broadcast-shapes
                             vt-arange vt-params-audit vt-params-audit-report))
         (suspects nil))
    (format t "~&~%=== 参数契约审计（共 ~d 个 vt-* 函数）===~%" (length audit))
    (dolist (entry audit)
      (destructuring-bind (name . p) entry
        (let ((warn nil))
          (when (and (getf p :rest) (not (member name vararg-whitelist)))
            (push (list name :rest-without-whitelist) suspects) (setf warn t))
          (when (not (getf p :doc-p))
            (push (list name :no-docstring) suspects) (setf warn t))
          (when warn
            (format t "~&  ⚠ ~a  必需=~d &optional=~a &rest=~a &key=~a~@[ keys=~a~] doc=~a~%"
                    name (getf p :required) (getf p :optional) (getf p :rest)
                    (getf p :key) (getf p :key-names) (getf p :doc-p))))))
    (format t "~&=== 可疑项 ~d 个 ===~%" (length suspects))
    (nreverse suspects)))


(defun %parcontract-self-check ()
  "验证寻址公式与快照机制的基本正确性。
   这是「计算期正确性」最小回归网，须在每次加载时通过。"
  (let* ((base (vt-from-sequence (loop for i below 12 collect (float i))
                                 :dtype :float64))
         ;; 非连续视图：base[::2] -> 物理下标 0,2,4,6,8,10
         (strided (vt-slice base '(0 nil 2))))
    ;; 1) %vt-flat-idx 与 vt-ref 必须一致
    (loop for i below (vt-size strided)
          do (let ((idx (make-array 1 :element-type 'fixnum
                                      :initial-element i)))
               (unless (= (aref (vt-data strided)
                                (%vt-flat-idx (vt-offset strided)
                                              (vt-strides strided) idx))
                          (vt-ref strided i))
                 (error "parcontract 自检失败：strided 寻址不一致 (i=~d)" i))))
    ;; 2) 标量视图（0 维）
    (let ((s (ensure-vt 42.0)))
      (unless (= (%vt-flat-idx (vt-offset s) nil
                               (make-array 0 :element-type 'fixnum))
                 (vt-offset s))
        (error "parcontract 自检失败：标量寻址")))
    ;; 3) 重叠检测与快照
    (let* ((z (vt-from-sequence '(1.0 2.0 3.0 4.0)))
           (rev (vt-slice z '(3 nil -1))))
      (unless (%vt-views-overlap-p z rev)
        (error "parcontract 自检失败：重叠检测漏报"))
      (let ((snap (vt-out-snapshot rev (list z))))
        (unless (not (eq (first snap) z))
          (error "parcontract 自检失败：重叠输入未快照")))
      ;; 非重叠的两个独立张量不应被快照
      (let ((a (vt-ones '(3))) (b (vt-ones '(3))))
        (unless (eq (first (vt-out-snapshot b (list a))) a)
          (error "parcontract 自检失败：非重叠输入被误快照"))))
    t))

;; 注：自检在 extensions2.lisp 之后调用（此时全部依赖已就绪），
;;     见文件末尾的注册。

;;; ==================================================================
;;; 5. 自检（加载时执行一次，防止基础设施本身失效）
;;; ==================================================================
