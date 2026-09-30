;;;; util.lisp — 通用辅助（浮点陷阱屏蔽、参数解析、排序）

(in-package :clvt)

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
