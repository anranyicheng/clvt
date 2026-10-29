;;;; extensions.lisp — 补充扩展功能

(in-package :clvt)

(defun vt-count-nonzero (tensor &key axis keepdims (dtype :int64))
  "统计非零元素个数（对标 numpy.count_nonzero）。
   掩码用 :bool（§17），sum 再升到 DTYPE —— 此前把掩码转成 :float64 再求和，
   对「布尔→浮点」的隐式转换只是为了绕过 :int8 掩码的整数语义。"
  (vt-sum (vt-nonzero-p tensor)
	  :axis axis :keepdims keepdims :dtype dtype))

(defun vt-count (tensor value &key axis keepdims (dtype :int64))
  "统计等于 VALUE 的元素个数（对标 (a == v).sum()）。
   比较结果是 :bool（§17），sum 再升到 DTYPE。"
  (vt-sum (vt-= tensor value)
	  :axis axis :keepdims keepdims :dtype dtype))

(defun vt-clip-tensor (tensor vmin vmax &key out dtype)
  "将元素限制在 [vmin, vmax] 内，vmin/vmax 可为标量或张量。"
  (vt-map (lambda (x lo hi)
	    (if (> x hi)
		hi
		(if (< x lo) lo x)))
          (ensure-vt tensor)
	  (ensure-vt vmin)
	  (ensure-vt vmax)
	  :dtype dtype :out out))

;;; ------------------------------------------------------------------
;;; pytorch 风格别名
;;; ------------------------------------------------------------------

(defun vt-clamp (tensor &optional (vmin nil vmin-p) (vmax nil vmax-p)
                         &key out dtype)
  "vt-clip 的 pytorch 风格别名（对标 torch.clamp）。

   与 vt-clip 的差异（pytorch 语义）：**允许只给下限** ——
   `(vt-clamp a 1)` 等价 `max(x, 1)`，而 `(vt-clip a 1)` 报错
   （numpy 的 clip() missing a_max）。实现上总是把两个位置参数填满，
   于是 clip 看到的是「显式给了 MAX-VAL=nil」，走「仅下限」分支。

   签名用 &OPTIONAL 而不是 (tensor vmin vmax &key …)：后者把 vmin/vmax
   定为必填，`(vt-clamp a)` 会报 invalid number of arguments。
   复数侧没有第二份同名实现：复数分支走 complex-ops.lisp 的 %CX-OP-CLAMP，
   本函数是唯一入口（§16.32 的单一 DEFUN 约束）。

   `:out` / `:dtype` 契约同 vt-clip（复数走字典序整元素夹取）。"
  (declare (ignore vmin-p vmax-p))
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
  (declare (type fixnum n start))
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
   - 0d × 任意：广播逐元素乘（numpy.inner 对 0 维输入按乘法处理）。
   - 1D × 1D：标量内积。
   - ND × MD：a 的最后一维与 b 的最后一维收缩，
     输出形状 = a.shape[:-1] ++ b.shape[:-1]。"
  (when (%cx-any-p a b)
    (return-from vt-inner (%cx-op-inner a b :dtype dtype :out out)))
  (let* ((a-vt (ensure-vt a))
         (b-vt (ensure-vt b))
         (ar (length (vt-shape a-vt)))
         (br (length (vt-shape b-vt))))
    (cond
      ((or (zerop ar) (zerop br))
       ;; 0 维输入无收缩轴：直接广播相乘（对标 numpy.inner）
       (vt-* a-vt b-vt :dtype dtype :out out))
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
   axes 为 (a-axes b-axes)：按指定轴列表收缩（两列表长度必须相等，
   支持负轴，按各自秩归一化——v0.3.5 修复：旧实现不归一化负轴，
   负轴被当作自由轴处理，产生错误结果或内部错误）。"
  (when (%cx-any-p a b)
    (return-from vt-tensordot (%cx-op-tensordot a b :axes axes)))
  (let* ((a-vt (ensure-vt a))
         (b-vt (ensure-vt b))
         (ar (length (vt-shape a-vt)))
         (br (length (vt-shape b-vt))))
    (flet ((norm-axes (axes rank which)
             (loop for x in axes
                   collect (let ((ax (if (minusp x) (+ x rank) x)))
                             (unless (and (>= ax 0) (< ax rank))
                               (error "vt-tensordot: ~a 的轴 ~a 越界（秩 ~a）"
                                      which x rank))
                             ax))))
    (cond
      ((integerp axes)
         (let ((n axes))
           (unless (and (>= n 0) (<= n ar) (<= n br))
             (error "vt-tensordot: axes ~a 超出可收缩范围（a 秩 ~a，b 秩 ~a）" n ar br))
           (let* ((af (- ar n))
              (bf (- br n))                      
              (all      (%vt-einsum-labels (+ af n bf)))
              (a-free   (subseq all 0 af))       
              (contract (subseq all af (+ af n)))
              (b-free   (subseq all (+ af n)))   
              (sub (%vt-einsum-string
                    (list (append a-free contract)
                          (append contract b-free))
                    (append a-free b-free))))
             (vt-einsum sub a-vt b-vt))))
      ((and (listp axes) (= (length axes) 2))
         (let* ((a-axes (norm-axes
                         (if (listp (first axes)) (first axes) (list (first axes)))
                         ar 'a))
                (b-axes (norm-axes
                         (if (listp (second axes)) (second axes) (list (second axes)))
                         br 'b))
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
        (t (error "axes 必须是整数或两个整数列表"))))))


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
         ;; 同 %GATHER-AXIS-K：VT-CONTIGUOUS 可能原样返回带 offset 的视图
         (src-off (vt-offset src-c))
         (idx-off (vt-offset idx-c))
         (result-data (make-array (vt-size source)
                                  :element-type (array-element-type src-data))))
    (let* ((outer (reduce #'* (subseq shape 0 ax) :initial-value 1))
           (inner (reduce #'* (subseq shape (1+ ax)) :initial-value 1))
           (k     (nth ax shape))
           (total (* outer k inner)))
      (declare (fixnum outer inner k total src-off idx-off))
      (dotimes (flat total)
        (let* ((i  (mod flat inner))
               (oj (floor flat inner))
               (o  (floor oj k))
               (gidx (truncate (aref idx-data (+ idx-off flat))))
               (src-flat (+ src-off (* o k inner) (* gidx inner) i)))
          (setf (aref result-data flat) (aref src-data src-flat))))
      (%make-vt :data result-data
                :shape shape
                :strides (vt-compute-strides shape)
                :offset 0
                :dtype (vt-dtype source)))))

(defun %gather-axis-k (source indices axis)
  "沿 AXIS 用 INDICES 的整数下标从 SOURCE 取值，**允许两侧沿 AXIS 长度不同**
   （`%gather-along-axis` 要求形状完全相同，topk 只要 k 个位置，用不了）。
   其余维度必须一一对应；结果形状 = INDICES 的形状、dtype = SOURCE 的 dtype。

   语义：(i0…, j, i…) 位置取 SOURCE 的 (i0…, INDICES[i0…,j,i…], i…)。
   内部先把两侧都落成连续布局（INDICES 常是 NARROW/FLIP 出来的视图）。"
  (let* ((shape (vt-shape source))
         (ishape (vt-shape indices))
         (rank (length shape))
         (ax (vt-normalize-axis axis rank))
         (src-c (vt-contiguous source))
         (idx-c (vt-contiguous indices))
         (src-data (vt-data src-c))
         (idx-data (vt-data idx-c))
         ;; `vt-contiguous` 对「已经连续」的输入原样返回 —— 而 NARROW/FLIP 出来的
         ;; 视图可能是 offset 非零、strides 也满足连续判定的（例如单元素轴翻转后
         ;; 只剩 offset=7）。因此**必须**把两边的 offset 加成基址，不能假设为 0。
         (src-off (vt-offset src-c))
         (idx-off (vt-offset idx-c))
         (result-data (make-array (vt-size indices)
                                  :element-type (array-element-type src-data))))
    (let* ((outer (reduce #'* (subseq shape 0 ax) :initial-value 1))
           (inner (reduce #'* (subseq shape (1+ ax)) :initial-value 1))
           (k (nth ax shape))
           (nk (nth ax ishape))
           (total (* outer nk inner)))
      (declare (fixnum outer inner k nk total src-off idx-off))
      (dotimes (flat total)
        (let* ((i (mod flat inner))
               (oj (floor flat inner))
               (o (floor oj nk))
               (gidx (truncate (aref idx-data (+ idx-off flat))))
               (src-flat (+ src-off (* o k inner) (* gidx inner) i)))
          (setf (aref result-data flat) (aref src-data src-flat))))
      (%make-vt :data result-data
                :shape ishape
                :strides (vt-compute-strides ishape)
                :offset 0
                :dtype (vt-dtype source)))))

;;; ==================================================================
;;; topk 的有界堆内核（第 26 轮）
;;;
;;; 原实现是 `vt-argsort` 把整份数据排完 O(N log N) 再切前 k 个 —— 1M double
;;; 取 k=10 要 **174 ms**，而「只要 k 个」本可以 O(N log k)：实测堆版 **2.2 ms**
;;; （79 倍）；k=1000 时 183 → 2.9 ms；k=100000 时 194 → 54 ms
;;; （k 越大优势越小，超过轴长 1/8 就退回 argsort，见 VT-TOPK 的选路）。
;;;
;;; **硬约束：结果必须与 argsort 路径逐位一致**（值 + 下标，含并列与 NaN）。
;;; 所以比较器不重新定义语义，而是复刻 `vt-sort` 用的那个全序：
;;;   · 非 NaN 按数值升序；
;;;   · **NaN 名次最高**（`%sort-fibers-*` 把 NaN 抽出来压到末尾），NaN 之间
;;;     视为并列；
;;;   · 并列时 `vt-argsort` 稳定 ⇒ 升序下按原下标升序。
;;; 于是：
;;;   largest  → 取「名次高」的 k 个，按 (名次降, 下标降) 输出
;;;              ＝ flip(narrow(argsort, N-k, N)) 的次序；
;;;   smallest → 取「名次低」的 k 个，按 (名次升, 下标升) 输出
;;;              ＝ narrow(argsort, 0, k) 的次序。
;;; 回归测试用「同一份输入分别走堆与 argsort，比较值序列和下标序列」，
;;; 而不是与 numpy 比 —— 后者测不出并列序的漂移。
;;; ==================================================================

(declaim (inline %topk-rank-df %topk-rank-sf %topk-rank-i64 %topk-rank-i32
                 %topk-rank-cx128 %topk-rank-cx64))

(defun %topk-rank-df (a b)
  "double-float 的**排序名次**比较：-1/0/1（A 的名次低于 B）。NaN 名次最高。"
  (declare (type double-float a b))
  (let ((na (not (= a a))) (nb (not (= b b))))
    (cond ((and na nb) 0)
          (na 1)
          (nb -1)
          ((< a b) -1)
          ((> a b) 1)
          (t 0))))

(defun %topk-rank-sf (a b)
  (declare (type single-float a b))
  (let ((na (not (= a a))) (nb (not (= b b))))
    (cond ((and na nb) 0)
          (na 1)
          (nb -1)
          ((< a b) -1)
          ((> a b) 1)
          (t 0))))

(defun %topk-rank-i64 (a b)
  (declare (type (signed-byte 64) a b))
  (cond ((< a b) -1) ((> a b) 1) (t 0)))

(defun %topk-rank-i32 (a b)
  (declare (type (signed-byte 32) a b))
  (cond ((< a b) -1) ((> a b) 1) (t 0)))

(defun %topk-rank-cx128 (a b)
  "complex double-float 的排序名次：任一分量 NaN 视作 NaN；否则字典序。
   等价于 `%CX-SORT-LESS`（NaN 末尾 + 实部→虚部），只是改写成 -1/0/1。"
  (declare (type (complex double-float) a b))
  (let ((ar (realpart a)) (ai (imagpart a))
        (br (realpart b)) (bi (imagpart b)))
    (declare (type double-float ar ai br bi))
    (let ((na (or (not (= ar ar)) (not (= ai ai))))
          (nb (or (not (= br br)) (not (= bi bi)))))
      (cond ((and na nb) 0)
            (na 1)
            (nb -1)
            ((< ar br) -1)
            ((> ar br) 1)
            ((< ai bi) -1)
            ((> ai bi) 1)
            (t 0)))))

(defun %topk-rank-cx64 (a b)
  (declare (type (complex single-float) a b))
  (let ((ar (realpart a)) (ai (imagpart a))
        (br (realpart b)) (bi (imagpart b)))
    (declare (type single-float ar ai br bi))
    (let ((na (or (not (= ar ar)) (not (= ai ai))))
          (nb (or (not (= br br)) (not (= bi bi)))))
      (cond ((and na nb) 0)
            (na 1)
            (nb -1)
            ((< ar br) -1)
            ((> ar br) 1)
            ((< ai bi) -1)
            ((> ai bi) 1)
            (t 0)))))

(defmacro %def-topk-heap (name ctype ranker largest)
  "生成有界堆内核 NAME(V BASE STRIDE N K) → (values VAL-ARR IDX-ARR)。

   VAL-ARR 是长度 K 的 ',CTYPE 数组、IDX-ARR 是长度 K 的 FIXNUM 数组，
   两者都按「先优后劣」排列（LARGEST 为编译期常量决定「优」的方向）。
   V 中每条 fiber 由 (BASE + i·STRIDE) 给出，i ∈ [0,N)。"
  `(defun ,name (v base stride n k)
     (declare (type (simple-array ,ctype (*)) v)
              (type fixnum base stride n k)
              (optimize (speed 3) (safety 1)))
     (let ((hv (make-array k :element-type ',ctype))
           (hi (make-array k :element-type 'fixnum))
           (sz 0))
       (declare (type (simple-array ,ctype (*)) hv)
                (type (simple-array fixnum (*)) hi)
                (type fixnum sz))
       ;; LABELS 而非 FLET：SIFT 要调用同层的 WORSE/BEATS-ROOT
       (labels ((worse (i j)
                ;; HV[I] 比 HV[J] 更该被淘汰
                (let ((c (,ranker (aref hv i) (aref hv j))))
                  ,(if largest
                       `(or (minusp c) (and (zerop c) (< (aref hi i) (aref hi j))))
                       `(or (plusp c) (and (zerop c) (> (aref hi i) (aref hi j)))))))
              (beats-root (x ix)
                (let ((c (,ranker x (aref hv 0))))
                  ,(if largest
                       `(or (plusp c) (and (zerop c) (> ix (aref hi 0))))
                       `(or (minusp c) (and (zerop c) (< ix (aref hi 0)))))))
              (sift (start)
                (declare (type fixnum start))
                (let ((pos start))
                  (declare (type fixnum pos))
                  (loop
                    (let* ((l (the fixnum (1+ (* 2 pos))))
                           (r (the fixnum (1+ l)))
                           (m pos))
                      (declare (type fixnum l r m))
                      (when (and (< l sz) (worse l m)) (setf m l))
                      (when (and (< r sz) (worse r m)) (setf m r))
                      (when (= m pos) (return))
                      (rotatef (aref hv pos) (aref hv m))
                      (rotatef (aref hi pos) (aref hi m))
                      (setf pos m))))))
         (declare (inline worse beats-root sift))
         ;; ---- 建堆 + 扫描 ----
         (dotimes (i n)
           (let ((x (aref v (the fixnum (+ base (* i stride))))))
             (cond
               ((< sz k)
                (setf (aref hv sz) x (aref hi sz) i)
                (incf sz)
                ;; 上浮（新元素只可能比父节点更差）
                (let ((pos (the fixnum (1- sz))))
                  (declare (type fixnum pos))
                  (loop while (> pos 0)
                        do (let ((par (the fixnum (floor (1- pos) 2))))
                             (declare (type fixnum par))
                             (if (worse pos par)
                                 (progn (rotatef (aref hv pos) (aref hv par))
                                        (rotatef (aref hi pos) (aref hi par))
                                        (setf pos par))
                                 (return))))))
               ((beats-root x i)
                (setf (aref hv 0) x (aref hi 0) i)
                (sift 0)))))
         ;; ---- 按「先优后劣」弹出 ----
         (let ((ov (make-array k :element-type ',ctype))
               (oi (make-array k :element-type 'fixnum)))
           (declare (type (simple-array ,ctype (*)) ov)
                    (type (simple-array fixnum (*)) oi))
           ;; 根是**最差**的那个：依次弹出得到「劣→优」，所以要倒着填
           (dotimes (j k)
             (let ((slot (the fixnum (- k 1 j))))
               (setf (aref ov slot) (aref hv 0))
               (setf (aref oi slot) (aref hi 0)))
             (decf sz)
             (when (plusp sz)
               (setf (aref hv 0) (aref hv sz) (aref hi 0) (aref hi sz))
               (sift 0)))
           (values ov oi))))))

(%def-topk-heap %topk-heap-max-df double-float %topk-rank-df t)
(%def-topk-heap %topk-heap-min-df double-float %topk-rank-df nil)
(%def-topk-heap %topk-heap-max-sf single-float %topk-rank-sf t)
(%def-topk-heap %topk-heap-min-sf single-float %topk-rank-sf nil)
(%def-topk-heap %topk-heap-max-cx128 (complex double-float) %topk-rank-cx128 t)
(%def-topk-heap %topk-heap-min-cx128 (complex double-float) %topk-rank-cx128 nil)
(%def-topk-heap %topk-heap-max-cx64 (complex single-float) %topk-rank-cx64 t)
(%def-topk-heap %topk-heap-min-cx64 (complex single-float) %topk-rank-cx64 nil)
(%def-topk-heap %topk-heap-max-i64 (signed-byte 64) %topk-rank-i64 t)
(%def-topk-heap %topk-heap-min-i64 (signed-byte 64) %topk-rank-i64 nil)
(%def-topk-heap %topk-heap-max-i32 (signed-byte 32) %topk-rank-i32 t)
(%def-topk-heap %topk-heap-min-i32 (signed-byte 32) %topk-rank-i32 nil)

(defun %topk-heap-kernel (flat largest)
  "按 FLAT 的元素类型挑一个堆内核；不支持的类型返回 NIL（调用方回退 argsort）。"
  (let ((et (array-element-type (vt-data flat))))
    (cond ((equal et 'double-float)
           (if largest #'%topk-heap-max-df #'%topk-heap-min-df))
          ((equal et 'single-float)
           (if largest #'%topk-heap-max-sf #'%topk-heap-min-sf))
          ((equal et '(signed-byte 64))
           (if largest #'%topk-heap-max-i64 #'%topk-heap-min-i64))
          ((equal et '(signed-byte 32))
           (if largest #'%topk-heap-max-i32 #'%topk-heap-min-i32))
          ((equal et '(complex double-float))
           (if largest #'%topk-heap-max-cx128 #'%topk-heap-min-cx128))
          ((equal et '(complex single-float))
           (if largest #'%topk-heap-max-cx64 #'%topk-heap-min-cx64))
          (t nil))))

(defun %topk-heap-path (tensor ax k kern out-shape sorted)
  "沿 AX 逐 fiber 跑有界堆，返回 (values 值 下标)。
   输出按行主序直接写：fiber 的 (outer, t) 与输出下标 j 的组合位置是
   ((outer·K + j)·ax-stride + t) —— 先算 fiber 再转置会多一趟。"
  (with-float-safe
    (let* ((flat (vt-contiguous (ensure-vt tensor)))
           (shape (vt-shape flat))
           (dim (nth ax shape))
           (ax-stride (nth ax (vt-strides flat)))
           (total (vt-size flat))
           (blk (the fixnum (* dim ax-stride)))
           (off (the fixnum (vt-offset flat)))
           (src (vt-data flat))
           (vres (make-vt out-shape 0 :dtype (vt-dtype tensor)))
           (ires (make-vt out-shape 0 :dtype :int64))
           (vd (vt-data vres))
           (id (vt-data ires)))
      (loop for base from 0 below total by blk
            for outer from 0
            do (dotimes (ti ax-stride)
                 (multiple-value-bind (vv ii)
                     (funcall kern src (+ off base ti) ax-stride dim k)
                   (dotimes (j k)
                     (let ((pos (the fixnum
                                     (+ (* (the fixnum (+ (* outer k) j)) ax-stride) ti))))
                       (setf (aref vd pos) (aref vv j))
                       (setf (aref id pos) (aref ii j)))))))
      (if sorted
          (values vres ires)
          (let ((perm (vt-argsort ires :axis ax)))
            (values (%gather-along-axis vres perm ax)
                    (%gather-along-axis ires perm ax)))))))


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
    ;; 选路：k 远小于轴长时走有界堆 O(N log k)，否则维持整份 argsort。
    ;; 阈值取轴长/8 —— 实测 1M double：k=10 174→1.1 ms、k=1000 183→1.7、
    ;; k=100000（轴长 1/10）194→32；再大就该排序了。
    (let* ((heap-kern (when (and (plusp k) (<= (* 8 k) ax-dim))
                        (%topk-heap-kernel (vt-contiguous (ensure-vt tensor)) largest))))
      (when heap-kern
        (return-from vt-topk
          (%topk-heap-path tensor ax k heap-kern
                           (let ((s (copy-list shape)))
                             (setf (nth ax s) k)
                             s)
                           sorted))))
    (let* ((sorted-indices (vt-argsort tensor :axis ax))
           ;; 只算**一次**序：不要同时 vt-sort 与 vt-argsort（两趟 O(n log n)）
           ;; —— topk 只要 k 个元素，值可以直接按下标 gather（%GATHER-AXIS-K）。
           (idxs (if largest
                     (vt-flip (vt-narrow sorted-indices ax (- ax-dim k) ax-dim) :axis ax)
                     (vt-narrow sorted-indices ax 0 k)))
           (vals (%gather-axis-k tensor idxs ax)))
      (if sorted
          (values vals idxs)
          (let* ((perm (vt-argsort idxs :axis ax)))
            (values (%gather-along-axis vals perm ax)
                    (%gather-along-axis idxs perm ax)))))))
