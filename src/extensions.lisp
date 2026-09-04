;;;; extensions.lisp — 补充扩展功能

(in-package :clvt)

(defun vt-count-nonzero (tensor &key axis keepdims (dtype :int64))
  "统计非零元素个数。"
  (vt-sum (vt-nonzero-p tensor :dtype :float64) :axis axis :keepdims keepdims :dtype dtype))

(defun vt-count (tensor value &key axis keepdims (dtype :int64))
  "统计等于 value 的元素个数。"
  (vt-sum (vt-= tensor value :dtype :float64) :axis axis :keepdims keepdims :dtype dtype))

(defun vt-clip-tensor (tensor vmin vmax &key out dtype)
  "将元素限制在 [vmin, vmax] 内，vmin/vmax 可为标量或张量。"
  (vt-map (lambda (x lo hi) (max lo (min hi x)))
          (ensure-vt tensor) (ensure-vt vmin) (ensure-vt vmax) :dtype dtype :out out))

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
  (let* ((flat (vt-ravel tensor)) (data (vt-data flat)) (size (vt-size flat))
         (offset (vt-offset flat)) (stride (first (vt-strides flat))) (result '()))
    (loop for i from 0 below size for ptr = (+ offset (* i stride))
          when (/= (aref data ptr) 0) do (push i result))
    (setf result (nreverse result))
    (if result (vt-from-sequence result :dtype dtype) (vt-zeros '(0) :dtype dtype))))

(defun vt-inner (a b &key dtype out)
  "内积（对标 numpy.inner）。"
  (let* ((a-vt (ensure-vt a)) (b-vt (ensure-vt b))
         (ar (length (vt-shape a-vt))) (br (length (vt-shape b-vt))))
    (cond ((and (= ar 1) (= br 1)) (vt-einsum "i,i->" a-vt b-vt :dtype dtype :out out))
          (t (let* ((af (1- ar)) (bf (1- br))
                    (a-labels (loop for i below af collect (code-char (+ #.(char-code #\a) i))))
                    (b-labels (loop for i below bf collect (code-char (+ #.(char-code #\a) (+ af i)))))
                    (c-label #\z)
                    (sub (format nil "~{~a~},~{~a~}->~{~a~}"
                                 (append a-labels (list c-label))
                                 (append b-labels (list c-label))
                                 (append a-labels b-labels))))
               (vt-einsum sub a-vt b-vt :dtype dtype :out out))))))

(defun vt-tensordot (a b &key (axes 2))
  "张量缩并（对标 numpy.tensordot）。"
  (let* ((a-vt (ensure-vt a)) (b-vt (ensure-vt b))
         (ar (length (vt-shape a-vt))) (br (length (vt-shape b-vt))))
    (cond
      ((integerp axes)
       (let* ((n axes) (af (- ar n)) (bf (- br n))
              (all (loop for i below (+ af n bf) collect (code-char (+ #.(char-code #\a) i))))
              (a-free (subseq all 0 af)) (contract (subseq all af (+ af n)))
              (b-free (subseq all (+ af n)))
              (sub (format nil "~{~a~},~{~a~}->~{~a~}"
                           (append a-free contract) (append contract b-free) (append a-free b-free))))
         (vt-einsum sub a-vt b-vt)))
      ((and (listp axes) (= (length axes) 2))
       (let* ((a-axes (if (listp (first axes)) (first axes) (list (first axes))))
              (b-axes (if (listp (second axes)) (second axes) (list (second axes))))
              (n (length a-axes)))
         (unless (= n (length b-axes)) (error "axes 子列表长度必须一致"))
         (let* ((all (loop for i below (+ (- ar n) n (- br n)) collect (code-char (+ #.(char-code #\a) i))))
                (a-free (loop for i below ar unless (member i a-axes) collect (pop all)))
                (contract (loop repeat n collect (pop all)))
                (b-free (loop for i below br unless (member i b-axes) collect (pop all)))
                (a-sub (let ((fi 0) (ci 0) (res (make-list ar)))
                         (loop for ax from 0 below ar do
                           (setf (nth ax res) (if (member ax a-axes) (nth ci (prog1 contract (incf ci)))
                                                  (nth fi (prog1 a-free (incf fi)))))) res))
                (b-sub (let ((fi 0) (ci 0) (res (make-list br)))
                         (loop for ax from 0 below br do
                           (setf (nth ax res) (if (member ax b-axes) (nth (prog1 ci (incf ci)) contract)
                                                  (nth (prog1 fi (incf fi)) b-free))))
                         res))
                (sub (format nil "~{~a~},~{~a~}->~{~a~}" a-sub b-sub (append a-free b-free))))
           (vt-einsum sub a-vt b-vt))))
      (t (error "axes 必须是整数或两个整数列表")))))

(defun vt-topk (tensor k &key (axis -1) (largest t) (sorted t))
  "沿轴取前 k 个最大/最小值及其索引。"
  (let* ((shape (vt-shape tensor)) (rank (length shape))
         (ax (vt-normalize-axis axis rank)) (ax-dim (nth ax shape)))
    (when (> k ax-dim) (error "vt-topk: k (~a) 不能大于轴大小 (~a)" k ax-dim))
    (let* ((sorted-tensor (vt-sort tensor :axis ax))
           (sorted-indices (vt-argsort tensor :axis ax))
           (vals (if largest
                     (vt-flip (vt-narrow sorted-tensor ax (- ax-dim k) ax-dim) :axis ax)
                     (vt-narrow sorted-tensor ax 0 k)))
           (idxs (if largest
                     (vt-flip (vt-narrow sorted-indices ax (- ax-dim k) ax-dim) :axis ax)
                     (vt-narrow sorted-indices ax 0 k))))
      (declare (ignore sorted))
      (values vals idxs))))
