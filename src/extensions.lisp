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
  (vt-map (lambda (x lo hi) (max lo (min hi x)))
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

(defun vt-inner (a b &key dtype out)
  "内积（对标 numpy.inner）。"
  (let* ((a-vt (ensure-vt a))
	 (b-vt (ensure-vt b))
         (ar (length (vt-shape a-vt)))
	 (br (length (vt-shape b-vt))))
    (cond ((and (= ar 1) (= br 1))
	   (vt-einsum "i,i->" a-vt b-vt :dtype dtype :out out))
          (t (let* ((af (1- ar)) (bf (1- br))
				 (a-labels (loop for i below af
						 collect (code-char (+ #.(char-code #\a) i))))
				 (b-labels (loop for i below bf
						 collect (code-char (+ #.(char-code #\a) (+ af i)))))
				 (c-label #\z)
				 (sub (format nil "~{~a~},~{~a~}->~{~a~}"
					      (append a-labels (list c-label))
					      (append b-labels (list c-label))
					      (append a-labels b-labels))))
               (vt-einsum sub a-vt b-vt :dtype dtype :out out))))))

(defun vt-tensordot (a b &key (axes 2))
  "张量缩并（对标 numpy.tensordot）。"
  (let* ((a-vt (ensure-vt a))
	 (b-vt (ensure-vt b))
         (ar (length (vt-shape a-vt)))
	 (br (length (vt-shape b-vt))))
    (cond
      ((integerp axes)
       (let* ((n axes)
	      (af (- ar n))
	      (bf (- br n))
              (all (loop for i below (+ af n bf)
			 collect (code-char (+ #.(char-code #\a) i))))
              (a-free (subseq all 0 af))
	      (contract (subseq all af (+ af n)))
              (b-free (subseq all (+ af n)))
              (sub (format nil "~{~a~},~{~a~}->~{~a~}"
                           (append a-free contract)
			   (append contract b-free)
			   (append a-free b-free))))
         (vt-einsum sub a-vt b-vt)))
      ((and (listp axes) (= (length axes) 2))
       (let* ((a-axes (if (listp (first axes))
			  (first axes)
			  (list (first axes))))
              (b-axes (if (listp (second axes))
			  (second axes)
			  (list (second axes))))
              (n (length a-axes)))
         (unless (= n (length b-axes)) (error "axes 子列表长度必须一致"))
         (let* ((all (loop for i below (+ (- ar n) n (- br n))
			   collect (code-char (+ #.(char-code #\a) i))))
                (a-free (loop for i below ar
			      unless (member i a-axes)
				collect (pop all)))
                (contract (loop repeat n collect (pop all)))
                (b-free (loop for i below br
			      unless (member i b-axes)
				collect (pop all)))
                (a-sub (let ((fi 0)
			     (ci 0)
			     (res (make-list ar)))
                         (loop for ax from 0 below ar do
                           (setf (nth ax res) (if (member ax a-axes)
						  (nth ci (prog1 contract (incf ci)))
                                                  (nth fi (prog1 a-free (incf fi))))))
			 res))
                (b-sub (let ((fi 0) (ci 0) (res (make-list br)))
                         (loop for ax from 0 below br do
                           (setf (nth ax res)
				 (if (member ax b-axes)
				     (nth (prog1 ci (incf ci)) contract)
				     (nth (prog1 fi (incf fi)) b-free))))
                         res))
                (sub (format nil "~{~a~},~{~a~}->~{~a~}"
			     a-sub b-sub (append a-free b-free))))
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
