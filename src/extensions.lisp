;;;; extensions.lisp — 补充扩展功能

(in-package :clvt)

(defun vt-count-nonzero (tensor &key axis keepdims (dtype :int64))
  "统计非零元素个数。"
  (vt-sum (vt-nonzero-p tensor :dtype :float64)
	  :axis axis :keepdims keepdims :dtype dtype))

(defun vt-count (tensor value &key axis keepdims (dtype :int64))
  "统计等于 value 的元素个数。"
  (vt-sum (vt-= tensor value :dtype :float64)
	  :axis axis :keepdims keepdims :dtype dtype))

(defun vt-clip-tensor (tensor vmin vmax &key out dtype)
  "将元素限制在 [vmin, vmax] 内，vmin/vmax 可为标量或张量。"
  (vt-map (lambda (x lo hi) (min hi (max lo x)))
          (ensure-vt tensor)
	  (ensure-vt vmin)
	  (ensure-vt vmax)
	  :dtype dtype :out out))

;;; ------------------------------------------------------------------
;;; pytorch 风格别名
;;; ------------------------------------------------------------------

(defun vt-clamp (tensor vmin vmax &key out dtype)
  "vt-clip 的 pytorch 风格别名。"
  (vt-clip tensor vmin vmax :out out :dtype dtype))

(defun vt-copy-to! (dst src)
  "pytorch 风格原地拷贝：将 src 拷贝到 dst，返回 dst。"
  (vt-copy-into dst src)
  dst)

(defun vt-flatnonzero (tensor &key (dtype :int64))
  "返回展平后非零元素的索引。"
  (let* ((flat (vt-ravel tensor))
	 (data (vt-data flat))
	 (size (vt-size flat))
         (offset (vt-offset flat))
	 (stride (first (vt-strides flat)))
	 (result '()))
    (loop for i from 0 below size
	  for ptr = (+ offset (* i stride))
          when (/= (aref data ptr) 0)
	    do (push i result))
    (setf result (nreverse result))
    (if result
	(vt-from-sequence result :dtype dtype)
	(vt-zeros '(0) :dtype dtype))))

(defparameter *vt-einsum-label-pool*
  (let ((chars '()))
    (loop for code from 33 to 126         
          for ch = (code-char code)
          unless (member ch '(#\. #\, #\- #\>))
	    do (push ch chars))
    (coerce (nreverse chars) 'simple-vector))
  "einsum 标签池。排除 . , - > 和空白后约 90 个唯一字符。
   vt-einsum 解析器把每个非特殊字符视作一个独立标签，
   因此本池容量即单次 einsum 允许的最大维度数上限。")

(defun %vt-einsum-pool-size ()
  "标签池容量。"
  (length *vt-einsum-label-pool*))

(defun %vt-einsum-labels (n &optional (start 0))
  "从标签池的第 start 个位置开始取 n 个唯一标签。
   若 n + start 超过池容量则报错。返回字符列表。"
  (declare (type fixnum n start)
           (optimize (speed 3) (safety 1)))
  (let* ((pool *vt-einsum-label-pool*)
         (size (length pool))
         (end  (+ start n)))
    (declare (type fixnum size end))
    (when (> end size)
      (error "einsum 标签池不足：需要 ~a 个标签（起始位置 ~a），池容量仅 ~a。
              请降低参与 einsum 的张量维度。"
             n start size))
    (loop for i fixnum from start below end
          collect (aref pool i))))

(defun %vt-einsum-string (input-subs output-sub)
  "把若干下标字符列表拼成 vt-einsum 接受的字符串，
   例如 input-subs = ((#\\a #\\b) (#\\b #\\c)), output-sub = (#\\a #\\c)
        返回 \"ab,bc->ac\"。"
  (with-output-to-string (s)
    (loop for sub in input-subs
          for first = t then nil
          do (unless first (write-char #\, s))
             (dolist (ch sub) (write-char ch s)))
    (write-string "->" s)
    (dolist (ch output-sub) (write-char ch s))))

(defun vt-inner (a b &key dtype out)
  "内积（对标 numpy.inner）。
   - 1D × 1D：标量内积。
   - ND × MD：a 的最后一维与 b 的最后一维收缩，
     输出形状 = a.shape[:-1] ++ b.shape[:-1]。"
  (let* ((a-vt (ensure-vt a))
         (b-vt (ensure-vt b))
         (ar (length (vt-shape a-vt)))
         (br (length (vt-shape b-vt))))
    (cond
      ((and (= ar 1) (= br 1))
       (vt-einsum "i,i->" a-vt b-vt :dtype dtype :out out))
      (t
       (let* ((af (1- ar))
              (bf (1- br))
              (all     (%vt-einsum-labels (+ af bf 1)))
              (a-free  (subseq all 0 af))            
              (b-free  (subseq all af (+ af bf)))    
              (c-label (nth (+ af bf) all))          
              (a-sub   (append a-free (list c-label)))
              (b-sub   (append b-free (list c-label)))
              (out-sub (append a-free b-free))
              (sub     (%vt-einsum-string (list a-sub b-sub) out-sub)))
         (vt-einsum sub a-vt b-vt :dtype dtype :out out))))))

(defun vt-tensordot (a b &key (axes 2))
  "张量缩并（对标 numpy.tensordot）。
   axes 为整数 n：a 的最后 n 维与 b 的前 n 维收缩。
   axes 为 (a-axes b-axes)：按指定轴列表收缩（两列表长度必须相等）。"
  (let* ((a-vt (ensure-vt a))
         (b-vt (ensure-vt b))
         (ar (length (vt-shape a-vt)))
         (br (length (vt-shape b-vt))))
    (cond
      ((integerp axes)
       (let* ((n axes)
              (af (- ar n))                      
              (bf (- br n))                      
              (all      (%vt-einsum-labels (+ af n bf)))
              (a-free   (subseq all 0 af))       
              (contract (subseq all af (+ af n)))
              (b-free   (subseq all (+ af n)))   
              (sub (%vt-einsum-string
                    (list (append a-free contract)
                          (append contract b-free))
                    (append a-free b-free))))
         (vt-einsum sub a-vt b-vt)))
      ((and (listp axes) (= (length axes) 2))
       (let* ((a-axes (if (listp (first axes))  (first axes)  (list (first axes))))
              (b-axes (if (listp (second axes)) (second axes) (list (second axes))))
              (n (length a-axes)))
         (unless (= n (length b-axes))
           (error "axes 子列表长度必须一致"))
         (let* ((af (- ar n))
                (bf (- br n))
                (all (%vt-einsum-labels (+ af n bf)))
                (a-free   (loop for i below ar
                                unless (member i a-axes)
                                  collect (pop all)))
                (contract (loop repeat n collect (pop all)))
                (b-free   (loop for i below br
                                unless (member i b-axes)
                                  collect (pop all)))
                (a-sub (let ((fi 0) (ci 0) (res (make-list ar)))
                         (loop for ax from 0 below ar do
                           (setf (nth ax res)
                                 (if (member ax a-axes)
                                     (nth ci (prog1 contract (incf ci)))
                                     (nth fi (prog1 a-free   (incf fi))))))
                         res))
                (b-sub (let ((fi 0) (ci 0) (res (make-list br)))
                         (loop for ax from 0 below br do
                           (setf (nth ax res)
                                 (if (member ax b-axes)
                                     (nth (prog1 ci (incf ci)) contract)
                                     (nth (prog1 fi (incf fi)) b-free))))
                         res))
                (sub (%vt-einsum-string (list a-sub b-sub)
                                        (append a-free b-free))))
           (vt-einsum sub a-vt b-vt))))
      (t (error "axes 必须是整数或两个整数列表")))))

(defun %gather-along-axis (source indices axis)
  "沿 axis 用 indices 中的整数作为索引从 source 中取值（PyTorch gather 语义）。
   要求 source 和 indices 形状相同；结果形状 = source.shape。
   indices 只沿 axis 取整数，其余维度与 source 一一对应。"
  (let* ((shape (vt-shape source))
         (rank (length shape))
         (ax (vt-normalize-axis axis rank))
         (src-c (vt-contiguous source))
         (idx-c (vt-contiguous indices))
         (src-data (vt-data src-c))
         (idx-data (vt-data idx-c))
         (result-data (make-array (vt-size source)
                                  :element-type (array-element-type src-data))))
    (let* ((outer (reduce #'* (subseq shape 0 ax) :initial-value 1))
           (inner (reduce #'* (subseq shape (1+ ax)) :initial-value 1))
           (k     (nth ax shape))
           (total (* outer k inner)))
      (declare (fixnum outer inner k total))
      (dotimes (flat total)
        (let* ((i  (mod flat inner))
               (oj (floor flat inner))
               (o  (floor oj k))
               (gidx (truncate (aref idx-data flat)))
               (src-flat (+ (* o k inner) (* gidx inner) i)))
          (setf (aref result-data flat) (aref src-data src-flat))))
      (%make-vt :data result-data
                :shape shape
                :strides (vt-compute-strides shape)
                :offset 0
                :dtype (vt-dtype source)))))

(defun vt-topk (tensor k &key (axis -1) (largest t) (sorted t))
  "沿轴取前 k 个最大/最小值及其索引。
   largest = t (默认)：取最大的 k 个。
   largest = nil：取最小的 k 个。
   sorted = t (默认)：
     返回结果按值排列（largest=t 时降序，largest=nil 时升序）。
   sorted = nil：
     返回结果按原始位置（轴索引）升序排列，
     即先选出 top-k 的元素，再按它们在原张量轴上的出现顺序输出。
     与 PyTorch torch.topk(..., sorted=False) 的「顺序未定义」不同，
     clvt 给出确定的位置序，便于复现。
   返回两个值：(values, indices)，都沿 axis 大小为 k。
   示例：
     x = [3, 1, 4, 1, 5, 9, 2, 6]
     (vt-topk x 3)             => values=[9, 6, 5], indices=[5, 7, 4]
     (vt-topk x 3 :sorted nil) => values=[5, 9, 6], indices=[4, 5, 7]
     (vt-topk x 3 :largest nil)
                               => values=[1, 1, 2], indices=[1, 3, 6]"
  (let* ((shape (vt-shape tensor))
         (rank (length shape))
         (ax (vt-normalize-axis axis rank))
         (ax-dim (nth ax shape)))
    (when (minusp k)
      (error "vt-topk: k (~a) 必须非负" k))
    (when (> k ax-dim)
      (error "vt-topk: k (~a) 不能大于轴大小 (~a)" k ax-dim))
    (let* ((sorted-tensor  (vt-sort    tensor :axis ax))
           (sorted-indices (vt-argsort tensor :axis ax))
           (vals (if largest
                     (vt-flip (vt-narrow sorted-tensor  ax (- ax-dim k) ax-dim) :axis ax)
                     (vt-narrow sorted-tensor  ax 0 k)))
           (idxs (if largest
                     (vt-flip (vt-narrow sorted-indices ax (- ax-dim k) ax-dim) :axis ax)
                     (vt-narrow sorted-indices ax 0 k))))
      (if sorted
          (values vals idxs)
          (let* ((perm (vt-argsort idxs :axis ax)))
            (values (%gather-along-axis vals perm ax)
                    (%gather-along-axis idxs perm ax)))))))
