;;;; util.lisp — 通用辅助（浮点陷阱屏蔽、参数解析、排序）

(in-package :clvt)

(defmacro with-float-safe (&body body)
  "在 IEEE 754 陷阱被屏蔽的环境下执行 body，使 0/0、溢出等产生 nan/inf 而非报错。"
  #+sbcl `(sb-int:with-float-traps-masked
              (:invalid :divide-by-zero :overflow :underflow)
            ,@body)
  #-sbcl `(locally ,@body))

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
                  (error "参数解析错误: 未知关键字参数 ~S。允许: ~S。" arg allowed-keys))
                (when (member arg seen-keys)
                  (error "参数解析错误: 关键字参数 ~S 重复出现。" arg))
                (unless iter
                  (error "参数解析错误: 关键字参数 ~S 缺少对应的值。" arg))
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
  "对实数序列排序（默认升序）。nan 处理语义对标 numpy：
   - 升序: 有限数升序，nan 放末尾。
   - 降序: 等价于 numpy 的 np.sort(arr)[::-1]，nan 出现在开头。"
  (declare (type (or list vector) sequence))
  (assert (or (eq predicate #'<) (eq predicate '<)
              (eq predicate #'>) (eq predicate '>)))
  (with-float-safe
    (let ((sequence (coerce sequence 'list)) non-nans nans)
      (dolist (x sequence)
        (if (vt-float-nan-p x) (push x nans) (push x non-nans)))
      (setf non-nans (stable-sort (nreverse non-nans) predicate)
            nans (nreverse nans))
      (if (or (eq predicate #'<) (eq predicate '<))
          (append non-nans nans)
          (append nans non-nans)))))
