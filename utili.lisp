(in-package :clvt)

(defmacro with-float-safe (&body body)
  `(sb-int:with-float-traps-masked
       (:invalid :divide-by-zero :overflow :underflow)
     ,@body))


(defun vt-coerce-to-tensor-type (val type-spec)
  "安全地将值转换为张量指定的元素类型。
   对于整数类型，执行向0截断（与 c 语言 (int)val 行为一致）。"
  (cond
    ((subtypep type-spec 'double-float) (coerce val 'double-float))
    ((subtypep type-spec 'single-float) (coerce val 'single-float))
    ((subtypep type-spec 'integer)
     (if (floatp val) (truncate val) val))
    (t val)))

(defun parse-vt-args (args allowed-keys)
  "通用参数解析核心.
  Args:
    args: 原始 &rest 参数列表.
    allowed-keys: 允许的关键字列表 (如 '(:dtype :out)).
  Returns:
    (values tensors-list kw-alist)
    - tensors-list: 按顺序提取的非关键字参数（张量候选者）.
    - kw-alist: 关键字参数的关联列表 ((:key . value) ...).
  Errors:
    - 遇到未知关键字立即报错.
    - 关键字参数缺值报错.
    - 关键字参数重复报错.
  "
  (let ((tensors nil)
        (kw-alist nil)
        (seen-keys nil))
    (loop with iter = args
          while iter
          for arg = (pop iter) ; 每次取出一个元素，推进列表
          do
             (cond
               ;; === 情况 A: 遇到关键字 ===
               ((keywordp arg)
		;; 1. 检查关键字是否合法
		(unless (member arg allowed-keys)
		  (error "参数解析错误: 未知关键字参数 ~S。允许的关键字为: ~S。" arg allowed-keys))             
		;; 2. 检查关键字是否重复
		(when (member arg seen-keys)
		  (error "参数解析错误: 关键字参数 ~S 重复出现。" arg))             
		;; 3. 检查关键字后是否有值 (防止以关键字结尾的情况)
		(unless iter
		  (error "参数解析错误: 关键字参数 ~S 缺少对应的值。" arg))             
		;; 4. 提取值并记录
		(let ((val (pop iter)))
		  (push (cons arg val) kw-alist)
		  (push arg seen-keys)))            
               ;; === 情况 B: 非关键字 (视为张量/位置参数) ===
               (t 
		(push arg tensors))))    
    ;; 返回结果：张量列表需要 nreverse 恢复原始顺序，kw-alist 保持倒序即可 (assoc 查找不依赖顺序)
    (values (nreverse tensors) kw-alist)))

(defun parse-vt-op-args (args)
  "专为张量运算设计的参数提取器。
  自动处理 :dtype 和 :out，返回标准格式。
  Returns:
    (values tensors-list dtype-value out-value)
  "
  (multiple-value-bind (tensors kws) 
      (parse-vt-args args '(:dtype :out))
    ;; 从关联列表中提取特定关键字，未找到则为 nil
    (let ((dtype (cdr (assoc :dtype kws)))
          (out   (cdr (assoc :out kws))))
      (values tensors dtype out))))

(defun fixnum-p (num)
  "判断数字是不是fixnum类型"
  (typep num 'fixnum))

(defun double-float-p (num)
  "判断数字是不是double-float类型"
  (typep num 'double-float))


(defun vt-numpy-sort (sequence &optional (predicate #'<))
  "对实数序列（列表或可转为列表的向量）进行排序(默认升序)。  
  nan 处理语义（严格对标 numpy）：
  - 升序 (#'<)：有限数升序排列，nan 放在末尾。
  - 降序 (#'>)：等价于 numpy 的 np.sort(arr)[::-1]。
    由于是对升序结果进行整体反转，nan 会出现在开头。
  保持多个 nan 的原始相对顺序。返回新列表。"
  (declare (type (or list vector) sequence))
  (assert (or (eq predicate #'<) (eq predicate '<)
              (eq predicate #'>) (eq predicate '>)))
  (with-float-safe
    (let ((sequence (coerce sequence 'list)) ; 统一为列表
          non-nans nans)
      ;; 分离 nan 和非 nan，按原始顺序
      (dolist (x sequence (setq non-nans (nreverse non-nans)
                                nans (nreverse nans)))
        (if (vt-float-nan-p x)
            (push x nans)
            (push x non-nans)))
      ;; 只在有限数（含 inf）上做标准稳定排序
      (setq non-nans (stable-sort non-nans predicate))
      ;; 拼接逻辑：
      ;; 对标 numpy 行为，numpy 的降序是通过升序后反转 [::-1] 实现的，
      ;; 因此降序时 nan 会被反转到开头。
      (cond 
        ;; 升序：有限数在前，nan 在后
        ((or (eq predicate #'<) (eq predicate '<)) 
         (append non-nans nans))
        ;; 降序：对标 numpy 的 [::-1] 反转，nan 在前，有限数降序在后
        ((or (eq predicate #'>) (eq predicate '>)) 
         (append nans non-nans))))))
