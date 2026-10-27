;;;; util.lisp — 通用辅助（浮点陷阱屏蔽、参数解析、排序）

(in-package :clvt)

;; 另外必须 INLINE：这些谓词的热点在归约/选择的逐元素循环里（每元素 1~2 次），
;; 而它们的形参类型是 T —— 不内联时函数体只能按泛型数值编译（`=` 走
;; TWO-ARG-= 的运行时派发），实测一次调用 ≈ 20~30 个周期。
;; 内联后调用点已知元素类型，`(floatp x)` 折成常量 T、`(= x x)` 折成一条
;; COMISD，整个判定降到 1~2 个周期。
;; 实测（1M double、axis=0 归约）：amax 14.6ms → 1.5ms，nanmax 14.5ms → 2.1ms。
(declaim (inline %nan-p %nan-or-inf-p))

;; 这两个谓词定义在 util.lisp 而不是 nan.lisp：它们必须**先于所有调用点**
;; 定义，否则调用点编译时拿不到源码、内联失败（SBCL 会给
;; "could not be inlined because its source code was not saved"）。
;; util.lisp 是第一个含代码的组件，nan.lisp 在它之后。
(declaim (ftype (function (t) boolean) %nan-p %nan-or-inf-p))

(defun %nan-p (x)
  "快速 NaN 判定：IEEE 754 中 NaN != NaN。调用方需屏蔽浮点陷阱。"
  (and (floatp x) (not (= x x))))

(defun %nan-or-inf-p (x)
  "浮点 NaN 或 ±Inf 判定。调用方需屏蔽浮点陷阱。"
  (and (floatp x)
       (or (not (= x x))
	   (> (abs x) most-positive-double-float))))



(defvar *processor-number*
  (or (ignore-errors
       (sb-alien:alien-funcall
	(sb-alien:extern-alien
	 "sysconf"
	 (function sb-alien:long sb-alien:int))
	(or sb-unix::sc-nprocessors-onln 84)))
      4)
  "系统cpu核心数量")

(defmacro with-float-safe (&body body)
  "屏蔽执行 body时，0/0、溢出等产生 nan/inf陷阱"
  `(sb-int:with-float-traps-masked
       (:invalid :divide-by-zero :overflow :underflow)
     ,@body))  

;;; ------------------------------------------------------------------
;;; 关键字参数解析
;;; ------------------------------------------------------------------

(defun parse-vt-args (args allowed-keys)
  "通用参数解析核心。
   Returns: (values tensors-list kw-alist)
   - tensors-list: 按顺序出现的非关键字参数。
   - kw-alist: ((:key . value) ...) 关联列表。
   遇到未知/重复/缺值关键字时报错。"
  (let ((tensors nil) (kw-alist nil) (seen-keys nil))
    (loop with iter = args
          while iter
          for arg = (pop iter)
          do (cond
               ((keywordp arg)
		(unless (member arg allowed-keys)
		  (error "参数解析错误（完整参数: ~S）: 未知关键字参数 ~S。允许: ~S。"
			 args arg allowed-keys))
		(when (member arg seen-keys)
		  (error "参数解析错误（完整参数: ~S）: 关键字参数 ~S 重复出现。" args arg))
		(unless iter
		  (error "参数解析错误（完整参数: ~S）: 关键字参数 ~S 缺少对应的值。" args arg))
                (push (cons arg (pop iter)) kw-alist)
                (push arg seen-keys))
               (t (push arg tensors))))
    (values (nreverse tensors) kw-alist)))

(defun parse-vt-op-args (args)
  "张量运算参数提取器：自动处理 :dtype 与 :out。
   Returns: (values tensors dtype out)"
  (multiple-value-bind (tensors kws) (parse-vt-args args '(:dtype :out))
    (values tensors (cdr (assoc :dtype kws)) (cdr (assoc :out kws)))))

;;; ------------------------------------------------------------------
;;; NaN 感知排序（严格对标 numpy）
;;; ------------------------------------------------------------------
(defun vt-numpy-sort (sequence &optional (predicate #'<))
  "对实数序列排序（默认升序）。语义严格对标 numpy：
   升序 = np.sort(arr)（有限数升序，nan 按原相对顺序排在末尾）；
   降序 = np.sort(arr)[::-1]（nan 在开头，且 nan 间相对顺序为升序的逆序）。"
  (declare (type (or list vector) sequence))
  (assert (or (eq predicate #'<)
	      (eq predicate '<)
	      (eq predicate #'>)
	      (eq predicate '>)))
  (with-float-safe
    (let* ((seq (coerce sequence 'list))
           (non-nans (loop for x in seq unless (%nan-p x) collect x))
           (nans     (loop for x in seq when   (%nan-p x) collect x))
           (asc (append (stable-sort non-nans #'<) nans)))
      (if (or (eq predicate #'<) (eq predicate '<))
          asc
          (nreverse asc)))))
