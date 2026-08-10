;;;; extensions.lisp — 补充重要的缺失函数
;;;; 对标 NumPy/PyTorch 常用功能

(in-package :clvt)

;;; ============================================================
;;; 1. vt-count-nonzero — 统计非零元素个数
;;; 对标 numpy.count_nonzero
;;; ============================================================

(defun vt-count-nonzero (tensor &key axis keepdims (dtype :int64))
  "统计张量中非零元素的个数。
   axis: nil 表示全局统计，整数或整数列表表示沿指定轴统计。
   keepdims: 是否保持归约维度。
   dtype: 输出类型，默认 :int64。
   对标 numpy.count_nonzero。"
  (with-float-safe
    (let* ((mask (vt-nonzero-p tensor :dtype :float64)))
      (vt-sum mask :axis axis :keepdims keepdims :dtype dtype))))

;;; ============================================================
;;; 2. vt-moveaxis — 移动轴到新位置
;;; 对标 numpy.moveaxis
;;; ============================================================

(defun vt-moveaxis (tensor source destination)
  "将张量的轴从 source 移动到 destination，返回零拷贝视图。
   source: 整数或整数列表（原始轴位置）。
   destination: 整数或整数列表（目标位置）。
   对标 numpy.moveaxis。"
  (with-float-safe
    (let* ((shape (vt-shape tensor))
           (rank (length shape))
           (src-list (if (listp source) source (list source)))
           (dst-list (if (listp destination) destination (list destination))))
      ;; 校验长度一致
      (unless (= (length src-list) (length dst-list))
        (error "vt-moveaxis: source 和 destination 长度必须一致"))
      ;; 规范化负索引
      (setf src-list (mapcar (lambda (s) (vt-normalize-axis s rank)) src-list))
      (setf dst-list (mapcar (lambda (d) (vt-normalize-axis d rank)) dst-list))
      ;; 构建排列：先收集未被移动的轴（按原始顺序），再按目标位置插入
      (let ((remaining (loop for i below rank
                             unless (member i src-list) collect i))
            (perm (make-list rank)))
        ;; 将未移动的轴按顺序填入非目标位置
        (let ((dst-set (coerce dst-list 'simple-vector))
              (src-map (let ((ht (make-hash-table)))
                         (loop for s in src-list for d in dst-list
                               do (setf (gethash s ht) d))
                         ht)))
          ;; 简单方法：构建 (original-pos . new-pos) 对
          (let ((pairs '()))
            ;; 被移动的轴
            (loop for s in src-list for d in dst-list
                  do (push (cons s d) pairs))
            ;; 未被移动的轴：按顺序填入剩余位置
            (let ((used-dsts (sort (copy-list dst-list) #'<))
                  (free-dsts (loop for i below rank
                                   unless (member i dst-list) collect i)))
              (loop for r in remaining for f in free-dsts
                    do (push (cons r f) pairs)))
            ;; 按目标位置排序，取原始位置构成 perm
            (setf pairs (sort pairs #'< :key #'cdr))
            (vt-transpose tensor (mapcar #'car pairs))))))))

;;; ============================================================
;;; 3. vt-inner — 内积（更通用的版本）
;;; 对标 numpy.inner
;;; ============================================================

(defun vt-inner (a b &key dtype out)
  "计算内积。对标 numpy.inner。
   - 1D × 1D → 标量（向量内积）
   - 高维：沿最后一个轴求和，返回 shape = (..., ...)
   例如：shape (2,3) × shape (4,3) → shape (2,4)"
  (with-float-safe
    (let* ((a-vt (ensure-vt a))
           (b-vt (ensure-vt b))
           (a-shape (vt-shape a-vt))
           (b-shape (vt-shape b-vt))
           (a-rank (length a-shape))
           (b-rank (length b-shape)))
      (cond
        ;; 1D × 1D → 标量
        ((and (= a-rank 1) (= b-rank 1))
         (vt-einsum "i,i->" a-vt b-vt :dtype dtype :out out))
        ;; 通用情况：沿最后一个轴收缩
        ;; 为每对输入分配唯一标签，最后一个轴共享
        (t
         (let* ((a-free (1- a-rank))
                (b-free (1- b-rank))
                ;; 使用 ASCII 字母分配标签
                (a-labels (loop for i below a-free collect (code-char (+ #.(char-code #\a) i))))
                (b-labels (loop for i below b-free collect (code-char (+ #.(char-code #\a) (+ a-free i)))))
                (contract-label #\z)
                (a-sub (append a-labels (list contract-label)))
                (b-sub (append b-labels (list contract-label)))
                (out-sub (append a-labels b-labels))
                (sub-str (format nil "~{~a~},~{~a~}->~{~a~}"
                                 (mapcar #'string a-sub)
                                 (mapcar #'string b-sub)
                                 (mapcar #'string out-sub))))
           (vt-einsum sub-str a-vt b-vt :dtype dtype :out out)))))))

;;; ============================================================
;;; 4. vt-tensordot — 张量缩并
;;; 对标 numpy.tensordot
;;; ============================================================

(defun vt-tensordot (a b &key (axes 2))
  "张量缩并。对标 numpy.tensordot。
   axes 可以是：
   - 整数 n：对 a 的最后 n 个轴和 b 的前 n 个轴求和。
   - 列表 (a-axes b-axes)：指定要收缩的轴对。
   返回缩并后的张量。"
  (with-float-safe
    (let* ((a-vt (ensure-vt a))
           (b-vt (ensure-vt b))
           (a-rank (length (vt-shape a-vt)))
           (b-rank (length (vt-shape b-vt))))
      (cond
        ;; 整数模式
        ((integerp axes)
         (let* ((n axes)
                (a-free (- a-rank n))
                (b-free (- b-rank n))
                ;; 预分配所有标签
                (all (loop for i below (+ a-free n b-free) collect (code-char (+ #.(char-code #\a) i))))
                (a-free-labels (subseq all 0 a-free))
                (contract-labels (subseq all a-free (+ a-free n)))
                (b-free-labels (subseq all (+ a-free n)))
                (a-sub (append a-free-labels contract-labels))
                (b-sub (append contract-labels b-free-labels))
                (out-sub (append a-free-labels b-free-labels))
                (sub-str (format nil "~{~a~},~{~a~}->~{~a~}"
                                 (mapcar #'string a-sub)
                                 (mapcar #'string b-sub)
                                 (mapcar #'string out-sub))))
           (vt-einsum sub-str a-vt b-vt)))
        ;; 列表模式
        ((and (listp axes) (= (length axes) 2))
         (let* ((a-axes (first axes))
                (b-axes (second axes))
                (a-axes-list (if (listp a-axes) a-axes (list a-axes)))
                (b-axes-list (if (listp b-axes) b-axes (list b-axes)))
                (n (length a-axes-list)))
           (unless (= n (length b-axes-list))
             (error "vt-tensordot: axes 的两个子列表长度必须一致"))
           ;; 用 einsum 实现：为每个轴分配唯一标签
           (let* ((a-free-count (- a-rank n))
                  (b-free-count (- b-rank n))
                  (all (loop for i below (+ a-free-count n b-free-count)
                             collect (code-char (+ #.(char-code #\a) i))))
                  ;; a 的自由轴标签
                  (a-free-labels (loop for i below a-rank
                                       unless (member i a-axes-list)
                                         collect (pop all)))
                  ;; 收缩轴标签（a 和 b 共享）
                  (contract-labels (loop repeat n collect (pop all)))
                  ;; b 的自由轴标签
                  (b-free-labels (loop for i below b-rank
                                       unless (member i b-axes-list)
                                         collect (pop all)))
                  ;; 构建 a 的下标：按轴顺序，自由轴用 free-labels，收缩轴用 contract-labels
                  (a-sub (let ((fi 0) (ci 0)
                               (result (make-list a-rank)))
                           (loop for ax from 0 below a-rank
                                 do (setf (nth ax result)
                                          (if (member ax a-axes-list)
                                              (nth ci contract-labels)
                                              (nth (prog1 fi (incf fi)) a-free-labels))))
                           result))
                  ;; 构建 b 的下标
                  (b-sub (let ((fi 0) (ci 0)
                               (result (make-list b-rank)))
                           (loop for ax from 0 below b-rank
                                 do (setf (nth ax result)
                                          (if (member ax b-axes-list)
                                              (nth (prog1 ci (incf ci)) contract-labels)
                                              (nth (prog1 fi (incf fi)) b-free-labels))))
                           result))
                  (out-sub (append a-free-labels b-free-labels))
                  (sub-str (format nil "~{~a~},~{~a~}->~{~a~}"
                                   (mapcar #'string a-sub)
                                   (mapcar #'string b-sub)
                                   (mapcar #'string out-sub))))
             (vt-einsum sub-str a-vt b-vt))))
        (t (error "vt-tensordot: axes 必须是整数或两个整数列表"))))))

;;; ============================================================
;;; 5. vt-topk — 获取前 k 个最大/最小值
;;; 对标 torch.topk
;;; ============================================================

(defun vt-topk (tensor k &key (axis -1) (largest t) (sorted t))
  "沿指定轴获取前 k 个最大（或最小）的值及其索引。
   k: 要返回的元素数量。
   axis: 操作轴（默认 -1，最后一维）。
   largest: t 取最大值，nil 取最小值。
   sorted: 是否按降序（或升序）排列。
   返回两个值：值张量和索引张量（dtype :int64）。
   对标 torch.topk。"
  (with-float-safe
    (let* ((shape (vt-shape tensor))
           (rank (length shape))
           (ax (vt-normalize-axis axis rank))
           (ax-dim (nth ax shape)))
      (when (> k ax-dim)
        (error "vt-topk: k (~a) 不能大于轴大小 (~a)" k ax-dim))
      ;; 沿轴排序（升序）
      (let* ((sorted-tensor (vt-sort tensor :axis ax))
             (sorted-indices (vt-argsort tensor :axis ax)))
        ;; 取 k 个元素
        (let ((vals (if largest
                        ;; 最大值：取最后 k 个（升序的尾部），再翻转为降序
                        (vt-flip (vt-narrow sorted-tensor ax (- ax-dim k) ax-dim) :axis ax)
                        ;; 最小值：取前 k 个
                        (vt-narrow sorted-tensor ax 0 k)))
              (idxs (if largest
                        (vt-flip (vt-narrow sorted-indices ax (- ax-dim k) ax-dim) :axis ax)
                        (vt-narrow sorted-indices ax 0 k))))
          ;; 如果不要排序
          (unless sorted
            ;; 随机打乱？这里简单返回不额外处理
            nil)
          (values vals idxs))))))

;;; ============================================================
;;; 6. vt-set-print-options — 设置打印选项
;;; ============================================================

(defun vt-set-print-options (&key threshold precision indent-step)
  "设置 clvt 的张量打印选项。
   threshold: 触发省略输出的元素阈值（默认 1000）。
   precision: 浮点数打印精度（默认 4）。
   indent-step: 缩进步长（默认 2）。"
  (when threshold
    (setf *vt-print-threshold* threshold))
  (when precision
    (setf *vt-print-precision* precision))
  (when indent-step
    (setf *vt-indent-step* indent-step))
  (values))

;;; ============================================================
;;; 7. vt-get-print-options — 获取打印选项
;;; ============================================================

(defun vt-get-print-options ()
  "返回当前打印选项的列表：(threshold precision indent-step)。"
  (list *vt-print-threshold* *vt-print-precision* *vt-indent-step*))

;;; ============================================================
;;; 8. vt-flatnonzero — 展平后返回非零元素索引
;;; 对标 numpy.flatnonzero
;;; ============================================================

(defun vt-flatnonzero (tensor &key (dtype :int64))
  "返回张量展平后非零元素的索引。对标 numpy.flatnonzero。"
  (with-float-safe
    (let* ((flat (vt-ravel tensor))
           (data (vt-data flat))
           (size (vt-size flat))
           (offset (vt-offset flat))
           (stride (first (vt-strides flat)))
           (result-indices '()))
      (loop for i fixnum from 0 below size
            for ptr fixnum = (+ offset (* i stride))
            when (/= (aref data ptr) 0)
              do (push i result-indices))
      (setf result-indices (nreverse result-indices))
      (if result-indices
          (vt-from-sequence result-indices :dtype dtype)
          (vt-zeros '(0) :dtype dtype)))))

;;; ============================================================
;;; 9. vt-argmax/argmin with keepdims support
;;; ============================================================

(defun vt-count (tensor value &key axis keepdims (dtype :int64))
  "统计张量中等于 value 的元素个数。
   axis: nil 表示全局统计。
   对标 numpy 中的 (arr == val).sum()。"
  (with-float-safe
    (let ((mask (vt-= tensor value :dtype :float64)))
      (vt-sum mask :axis axis :keepdims keepdims :dtype dtype))))

;;; ============================================================
;;; 10. vt-clip with tensor bounds
;;; ============================================================

(defun vt-clip-tensor (tensor vmin vmax &key out dtype)
  "将张量元素限制在 [vmin, vmax] 范围内。
   vmin 和 vmax 可以是标量或张量（支持广播）。
   这是 vt-clip 的张量边界扩展版本。"
  (with-float-safe
    (let* ((t-vmin (ensure-vt vmin))
           (t-vmax (ensure-vt vmax)))
      (vt-map (lambda (x lo hi) (max lo (min hi x)))
              (ensure-vt tensor) t-vmin t-vmax
              :dtype dtype :out out))))
