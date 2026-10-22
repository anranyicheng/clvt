;;;; complex-ops.lisp — 复数「数据搬运层」（第二轮进化）
;;;;
;;;; 第一轮（src/complex.lisp）解决的是**逐元素算术与归约**；
;;;; 这一轮补的是**结构层**：转换 / 拼接 / 剪切 / 集合 / 缺失值 / 缩并。
;;;;
;;;; 设计约束（延续第一轮，CONVENTIONS §16）：
;;;;
;;;;  1) **实数路径零改动**：每个被覆盖的函数在加载时先用 %CX-ORIG 捕获
;;;;     原实现，只有「含复数」的调用进新内核，其余原样转发（性能与语义
;;;;     与第一轮之前完全一致）。
;;;;
;;;;  2) **顺序即语义**：复数的全序是**字典序**（实部优先，再比虚部），
;;;;     NaN 恒排末尾（§16.8）。sort / argsort / unique / searchsorted /
;;;;     set 运算共用同一个比较器 %CX-SORT-LESS，避免「各处各写一套序」
;;;;     导致 unique 与 sort 结果互相矛盾。
;;;;
;;;;  3) **哈希键唯一化**：所有「按值查表」的语义（in1d/isin/unique 的
;;;;     inverse）统一走 %CX-KEY —— 任一分量为 NaN 折叠为同一哨兵、
;;;;     其余规范化成 double-float 复数，保证 1.0s0 与 1.0d0 同键。
;;;;
;;;;  4) **不允许静默丢虚部**：nanmean/nanvar/nanstd 之前会经
;;;;     `vt-astype → :float64` 取实部（CONVENTIONS §16.5 的降级规则），
;;;;     于是 nanmean 给出「实数均值」—— 一个**看起来正常但错**的答案。
;;;;     本轮把它们全部改成复数语义（§16.18 记录）。

(in-package :clvt)

;;; ==================================================================
;;; 0. 原实现预捕获
;;; ==================================================================
;;; 与 complex.lisp 同一机制：必须在**本文件任何 defun 之前**登记，
;;; 否则运行时捕获到的是包装器自身（无限递归）。
(eval-when (:load-toplevel :execute)
  (dolist (name '(vt-nan-to-num vt-view vt-repeat
                  vt-unique vt-in1d vt-isin
                  vt-intersect1d vt-setdiff1d vt-union1d vt-setxor1d
                  vt-searchsorted
                  vt-nansum vt-nanprod vt-nanmax vt-nanmin
                  vt-nanargmax vt-nanargmin
                  vt-nanmean vt-nanvar vt-nanstd vt-nanmedian
                  vt-correlate vt-convolve vt-inner vt-tensordot
                  vt-vdot vt-nancumsum vt-nancumprod vt-digitize
                  ;; v1.3：补齐 numpy 支持复数的算子 + 明确拒绝清单
                  vt-expm1 vt-log1p vt-sinc vt-sign
                  vt-clip vt-clamp vt-standardize
                  vt-cov vt-corrcoef vt-fromiter
                  vt-nextafter vt-spacing vt-logaddexp vt-hypot vt-copysign
                  vt-signbit vt-deg2rad vt-rad2deg vt-bincount vt-histogram
                  vt-nanpercentile vt-nanquantile vt-interp vt-positive
                  ;; 第四轮：any / all 的复数真值语义
                  vt-any vt-all))
    (%cx-orig name)))

;;; ==================================================================
;;; 1. 值级小工具
;;; ==================================================================

(declaim (inline %cx-num= %cx-key %cx-abs2 %cx-all-nan-p))

(defun %cx-num= (a b)
  "复数**数值相等**：NaN 与 NaN 视为相等（numpy.unique / set 语义）。
   前置条件：调用方已在 with-float-safe 内（`=` 遇 NaN 会触发 invalid 陷阱）。"
  (if (or (%cx-nan-p a) (%cx-nan-p b))
      (and (%cx-nan-p a) (%cx-nan-p b))
      (and (= (%creal a) (%creal b)) (= (%cimag a) (%cimag b)))))

(defun %cx-key (z)
  "按值查表的哈希键：NaN（任一分量）→ 单一哨兵；否则规范化成
   (complex double-float)，使 1.0s0 与 1.0d0、1 与 1.0d0 同键。
   EQL 哈希表对 complex 按其分量做 eql 比较（SBCL 实测可用）。"
  (if (%cx-nan-p z)
      :%cx-nan
      (complex (coerce (%creal z) 'double-float)
               (coerce (%cimag z) 'double-float))))

(defun %cx-abs2 (z)
  "|z|²（实数）。比 (abs z) 少一次开方，用于方差类计算。"
  (let ((re (%creal z)) (im (%cimag z)))
    (+ (* re re) (* im im))))

(defun %cx-all-nan-p (z)
  "%CX-NAN-P 的别名（语义读起来更清楚：该元素整体是缺失值）。"
  (%cx-nan-p z))

(defun %cx-complex-zero-of (dtype)
  "DTYPE 对应精度的 0+0i。"
  (if (eq (vt-complex-real-dtype dtype) :float32)
      (complex 0.0s0 0.0s0)
      (complex 0.0d0 0.0d0)))

(defun %cx-complex-nan-of (dtype)
  "DTYPE 对应精度的 NaN+NaNj。"
  (let ((rdt (vt-complex-real-dtype dtype)))
    (if (eq rdt :float32)
        (complex +vt-sfloat-nan+ +vt-sfloat-nan+)
        (complex +vt-dfloat-nan+ +vt-dfloat-nan+))))

;;; ==================================================================
;;; 2. 转换：nan_to_num（复数按分量替换）
;;; ==================================================================

(defun %cx-nan-to-num-element (z nan posinf neginf)
  "单个复数元素的 nan_to_num：实部/虚部**各自**替换（对标 numpy）。
   调用方保证在 with-float-safe 内。"
  (flet ((fix (v)
           (cond ((%cx-nan-p v) nan)
                 ((%cx-inf-p v) (if (plusp v) posinf neginf))
                 (t v))))
    (complex (fix (%creal z)) (fix (%cimag z)))))

(defun vt-nan-to-num (tensor &key (copy t) (nan 0.0d0) (posinf nil) (neginf nil) out dtype)
  "把 NaN/±Inf 替换掉（对标 numpy.nan_to_num）。

   复数（§16.18）：**实部与虚部分别处理**，替换值 NAN/POSINF/NEGINF 是实数，
   同时作用于两个分量；POSINF/NEGINF 缺省为该分量精度的最大/最小有限值
   （complex128 → ±1.797e308，complex64 → ±3.4028e38）。
   结果 dtype 默认与输入相同（complex64 进 complex64 出）。

   例：(vt-nan-to-num #C(nan-inf i)) 的参数化版本见 test/complex-ops-test.lisp。"
  (declare (ignore copy))
  (let ((in (ensure-vt tensor)))
    (unless (vt-complex-dtype-p (vt-dtype in))
      (return-from vt-nan-to-num
        (funcall (%cx-orig 'vt-nan-to-num) tensor
                 :copy copy :nan nan :posinf posinf :neginf neginf
                 :out out :dtype dtype)))
    (let* ((dt (or dtype (vt-dtype in)))
           (rdt (vt-complex-real-dtype dt))
           (lt (vt-dtype->lisp-type rdt))
           (maxf (if (eq rdt :float32) 3.4028235f38 most-positive-double-float))
           (minf (if (eq rdt :float32) -3.4028235f38 most-negative-double-float))
           (nanv (coerce (or nan 0.0d0) lt))
           (posv (coerce (or posinf maxf) lt))
           (negv (coerce (or neginf minf) lt)))
      (unless (vt-complex-dtype-p dt)
        (error "vt-nan-to-num: 复数输入的 :dtype 必须是复数类型，收到 ~a" dt))
      (with-float-safe
        (vt-map (lambda (z) (%cx-nan-to-num-element z nanv posv negv))
                in :out out :dtype dt)))))

;;; ==================================================================
;;; 3. 转换：vt-view 的 dtype 重解释（complex ↔ float）
;;; ==================================================================
;;; numpy 的 `a.view(np.float64)` 把 complex128 的每个元素摊成两个 double，
;;; 是**零拷贝**的（同一块内存换个 dtype 看）。
;;;
;;; Common Lisp 的数组**无法**跨元素类型做位移数组（
;;; (make-array :displaced-to ...) 要求元素类型一致，SBCL 会直接报错），
;;; 所以这里按分量逐元素重建 —— 数值完全等价，但**不共享内存**。
;;; 这一偏差登记在 CONVENTIONS §16.18「已知差异 D-C7」，并写进 docstring，
;;; 避免使用者以为改视图能改原数组。

(defun %cx-view-pair-p (from to)
  "允许的 dtype 重解释对（只开放「分量宽度相同」的两对，保证逐位等价）。
   complex128↔float64、complex64↔float32。"
  (or (and (eq from :complex128) (eq to :float64))
      (and (eq from :float64) (eq to :complex128))
      (and (eq from :complex64) (eq to :float32))
      (and (eq from :float32) (eq to :complex64))))

(defun %cx-view-dtype-shape (shape from to)
  "按 numpy 规则算重解释后的形状：只缩放**最后一维**。
   complex128 (2 3) → float64 (2 6)；float64 (2 6) → complex128 (2 3)。"
  (let* ((fi (vt-dtype-itemsize from))
         (ti (vt-dtype-itemsize to))
         (last (car (last shape)))
         (new-last (if (> fi ti)
                       (* last (floor fi ti))
                       (if (zerop (mod last (floor ti fi)))
                           (floor last (floor ti fi))
                           (error "vt-view: 最后一维 ~a 不能按 ~a→~a 整除" last from to)))))
    (append (butlast shape) (list new-last))))

(defun %cx-reinterpret (v shape dtype)
  "把 V 的内容按 DTYPE 重新解释（逐元素重建，语义等价 numpy.view）。
   SHAPE 为 NIL 时自动按 numpy 规则缩放最后一维。"
  (let* ((from (vt-dtype v))
         (flat (%cx-flat v))
         (n (vt-size flat))
         (fi (vt-dtype-itemsize from))
         (ti (vt-dtype-itemsize dtype))
         (m (if (> fi ti) (* n (floor fi ti)) (floor n (floor ti fi))))
         (out-shape (or shape (%cx-view-dtype-shape (vt-shape v) from dtype)))
         (out-size (reduce #'* out-shape :initial-value 1)))
    (unless (= out-size m)
      (error "vt-view: 重解释后元素数不一致（~a 个 ~a = ~a 个 ~a，但 shape ~a 需要 ~a 个）"
             n from m dtype out-shape out-size))
    (let ((res (make-vt out-shape 0 :dtype dtype))
          (src (vt-data flat)))
      (let ((dst (vt-data res))
            (off (vt-offset flat)))
        (with-float-safe
          (cond
            ((and (eq from :complex128) (eq dtype :float64))
             (dotimes (i n)
               (let ((z (aref src (+ off i))))
                 (setf (aref dst (* 2 i)) (%creal z)
                       (aref dst (1+ (* 2 i))) (%cimag z)))))
            ((and (eq from :complex64) (eq dtype :float32))
             (dotimes (i n)
               (let ((z (aref src (+ off i))))
                 (setf (aref dst (* 2 i)) (%creal z)
                       (aref dst (1+ (* 2 i))) (%cimag z)))))
            ((and (eq from :float64) (eq dtype :complex128))
             (dotimes (i (/ n 2))
               (setf (aref dst i) (complex (aref src (+ off (* 2 i)))
                                           (aref src (+ off (1+ (* 2 i))))))))
            ((and (eq from :float32) (eq dtype :complex64))
             (dotimes (i (/ n 2))
               (setf (aref dst i) (complex (aref src (+ off (* 2 i)))
                                           (aref src (+ off (1+ (* 2 i))))))))
            (t (error "vt-view: 不支持 ~a → ~a 的重解释" from dtype)))))
      res)))

(defun vt-view (vt &rest args)
  "零拷贝重塑视图（对标 pytorch tensor.view）；给出 :DTYPE 时按 numpy 语义重解释。

  调用形式：
    (vt-view a shape)                 ; 既有行为：零拷贝、元素总数必须一致
    (vt-view a shape :dtype dt)       ; 重解释到 dt（形状需按字节数匹配）
    (vt-view a :dtype dt)             ; 省略 shape：自动缩放最后一维
                                      ; （对标 numpy 的 a.view(dt)）

  :DTYPE 与输入不同时**重解释**：complex128 ↔ float64、complex64 ↔ float32
  （只开放「分量宽度相同」的两对，保证逐位等价）。

  ⚠ 与 numpy 的差异（§16.18 D-C7）：CL 数组无法跨元素类型做位移数组，
  因此重解释返回**独立副本**，不共享内存；numpy 的 view 是共享的。

  为什么用 &REST 而不是 (&optional shape &key dtype)：CL 无法在可选位置上
  区分「关键字」与「形状」，`(vt-view a :dtype dt)` 会被解析成
  shape=:DTYPE 然后报 odd number of &KEY arguments。这里手工分流：
  首参是关键字 → 视为只有关键字；否则首参是形状。"
  (let ((v (ensure-vt vt))
        (shape nil)
        (dtype nil))
    (cond
      ((null args) (setf shape nil))
      ((keywordp (first args)) (setf dtype (getf args :dtype)))
      (t (setf shape (first args)
               dtype (getf (rest args) :dtype))))
    (cond
      ((or (null dtype) (eq dtype (vt-dtype v)))
       (funcall (%cx-orig 'vt-view) v shape))
      ((not (%cx-view-pair-p (vt-dtype v) dtype))
       (error "vt-view: 不支持 ~a → ~a 的重解释（仅 complex128↔float64、complex64↔float32）"
              (vt-dtype v) dtype))
      (t (%cx-reinterpret v shape dtype)))))



;;; ==================================================================
;;; 4. 转换：repeat（复数）
;;; ==================================================================
;;; 实现直接复用 manip.lisp 的两条特化内核，只在它的 8 路类型分派里
;;; 追加 2 个复数分支（见 src/manip.lisp 的 dup 宏）。此处只补一层
;;; 「标量 repeats + axis=nil」的完整复数路径，避免 list 展开的额外开销。

(defun %cx-repeat-complex (v repeats axis)
  "VT-REPEAT 的复数路径：REPEATS 为标量或与轴等长的列表。"
  (let* ((in (ensure-vt v))
         (flat (%cx-flat in))
         (shape (vt-shape in))
         (rank (length shape))
         (dt (vt-dtype in)))
    (if (null axis)
        (let* ((n (vt-size flat))
               (reps (if (listp repeats)
                         (progn (unless (= (length repeats) n)
                                  (error "vt-repeat: repeats 长度 ~a 与元素总数 ~a 不匹配"
                                         (length repeats) n))
                                (mapcar (lambda (r) (max (the integer r) 0)) repeats))
                         (make-list n :initial-element (max (the integer repeats) 0))))
               (total (reduce #'+ reps :initial-value 0))
               (res (make-vt (list total) 0 :dtype dt)))
          (%cx-dispatch (vt-data flat) src zero
            (declare (ignore zero))
            (let ((dst (vt-data res))
                  (off (vt-offset flat))
                  (pos 0))
              (declare (type fixnum off pos))
              (loop for r of-type fixnum in reps
                    for i of-type fixnum from 0
                    do (let ((val (aref src (+ off i))))
                         (dotimes (k r)
                           (declare (type fixnum k))
                           (setf (aref dst pos) val)
                           (incf pos))))))
          res)
        ;; 沿轴重复：外层下标走里程计，内层 fiber 交给 %REPEAT-FIBER
        (let* ((ax (vt-normalize-axis axis rank))
               (ax-size (nth ax shape))
               (reps (%repeat-counts repeats ax-size ax))
               (out-ax (reduce #'+ reps :initial-value 0))
               (out-shape (loop for d in shape for i fixnum from 0
                                collect (if (= i ax) out-ax d)))
               (res (make-vt out-shape 0 :dtype dt))
               (src (vt-contiguous in))
               (in-str (vt-strides src))
               (out-str (vt-strides res))
               (in-ax-str (nth ax in-str))
               (out-ax-str (nth ax out-str))
               (o-dims (loop for d in shape for i fixnum from 0
                             unless (= i ax) collect d))
               (o-in (loop for st in in-str for i fixnum from 0
                           unless (= i ax) collect st))
               (o-out (loop for st in out-str for i fixnum from 0
                            unless (= i ax) collect st)))
          (if (zerop out-ax)
              res
              (let ((idx (make-list (length o-dims) :initial-element 0))
                    (ip (vt-offset src))
                    (op 0))
                (declare (type fixnum ip op))
                (loop
                  (%repeat-fiber (vt-data src) (vt-data res)
                                 ip op reps ax-size in-ax-str out-ax-str)
                  (let ((d (1- (length o-dims))))
                    (loop
                      (when (< d 0) (return-from %cx-repeat-complex res))
                      (incf (nth d idx))
                      (when (< (nth d idx) (nth d o-dims)) (return))
                      (setf (nth d idx) 0)
                      (decf d)))
                  (setf ip (vt-offset src) op 0)
                  (loop for i fixnum from 0 below (length o-dims)
                        do (incf ip (* (nth i idx) (nth i o-in)))
                           (incf op (* (nth i idx) (nth i o-out)))))))))))

(defun vt-repeat (vt repeats &key axis)
  "重复元素（对标 numpy.repeat）。复数走复数路径，实数转发原实现。"
  (if (vt-complex-dtype-p (vt-dtype (ensure-vt vt)))
      (%cx-repeat-complex vt repeats axis)
      (funcall (%cx-orig 'vt-repeat) vt repeats :axis axis)))

;;; ==================================================================
;;; 5. 集合运算（unique / in1d / isin / intersect / setdiff / setxor）
;;; ==================================================================
;;; 复数的全序是字典序（§16.8），NaN 恒排末尾且**折叠为一个**（numpy 语义：
;;; np.unique 把多个 NaN 当同一值）。所有集合运算都建立在
;;;   · %CX-SORT-FIBER（原地归并，NaN 末尾）
;;;   · %CX-KEY（EQL 哈希键，NaN 折叠）
;;; 这两个原语之上，因此 unique 的顺序与 in1d 的判等**必然自洽**。

(defun %cx-unique-sorted (flat dt)
  "展平、排序（字典序 + NaN 末尾）后做 run-length。
   返回 (values 唯一值数组 计数 fixnum 向量 唯一数)。"
  (let* ((copy (vt-copy flat))
         (data (vt-data copy))
         (n (vt-size flat)))
    (%cx-sort-fiber data 0 n)
    (%cx-dispatch data arr zero
      (declare (ignore zero))
      (let ((uniq (make-array (max n 1) :element-type (array-element-type data)))
            (cnts (make-array (max n 1) :element-type '(signed-byte 64)))
            (u 0) (pos 0))
        (declare (type fixnum u pos))
        (loop while (< pos n) do
          (let* ((val (aref arr pos))
                 (start pos))
            (declare (type fixnum start))
            (loop while (and (< pos n) (%cx-num= val (aref arr pos)))
                  do (incf pos))
            (setf (aref uniq u) val)
            (setf (aref cnts u) (the fixnum (- pos start)))
            (incf u)))
        (values (if (= u n) uniq (subseq uniq 0 u))
                (if (= u n) cnts (subseq cnts 0 u))
                u)))))

(defun %cx-unique (tensor return-index return-inverse return-counts)
  "复数版 vt-unique（展平 + 字典序 + NaN 折叠）。"
  (with-float-safe
    (let* ((flat (%cx-flat tensor))
           (n (vt-size flat))
           (dt (vt-dtype tensor)))
      (multiple-value-bind (uniq cnts u) (%cx-unique-sorted flat dt)
        (let ((uniq-vt (%make-vt :data uniq :shape (list u) :strides '(1)
                                 :offset 0 :dtype dt))
              (cnt-vt (%make-vt :data cnts :shape (list u) :strides '(1)
                                :offset 0 :dtype :int64))
              (idx-vt nil) (inv-vt nil))
          ;; 首次下标 / 逆映射：只在被请求时建哈希（O(n+m)）
          (when (or return-index return-inverse)
            (let ((h (make-hash-table :test #'eql))
                  (first-idx (make-array u :element-type '(signed-byte 64)
                                           :initial-element -1))
                  (inv (when return-inverse
                         (make-array n :element-type '(signed-byte 64))))
                  (src (vt-data flat))
                  (off (vt-offset flat)))
              (dotimes (i u)
                (setf (gethash (%cx-key (aref uniq i)) h) i))
              (dotimes (i n)
                (let* ((v (aref src (+ off i)))
                       (uidx (gethash (%cx-key v) h)))
                  (when (= (aref first-idx uidx) -1)
                    (setf (aref first-idx uidx) i))
                  (when inv (setf (aref inv i) uidx))))
              (when return-index
                (setf idx-vt (%make-vt :data first-idx :shape (list u)
                                       :strides '(1) :offset 0 :dtype :int64)))
              (when return-inverse
                (setf inv-vt (%make-vt :data inv :shape (list n)
                                       :strides '(1) :offset 0 :dtype :int64)))))
          (if (or return-index return-inverse return-counts)
              (values uniq-vt
                      (when return-index idx-vt)
                      (when return-inverse inv-vt)
                      (when return-counts cnt-vt))
              uniq-vt))))))

(defun vt-unique (tensor &key return-index return-inverse return-counts axis)
  "返回展平后唯一元素（升序、NaN 折叠为一个）。复数按字典序（§16.8）。

   AXIS 给定时按轴去重（§16.23）。该路径的元素比较原语（%AXIS-ELEM-LESSP /
   %AXIS-ELEM-EQ）本身与元素类型无关，复数按 (实部, 虚部) 字典序参与比较，
   因此直接复用实数层的 %UNIQUE-AXIS，无需第二份实现。"
  (let* ((in (ensure-vt tensor))
         (rank (length (vt-shape in))))
    (cond ((and axis (> rank 1))
           (%unique-axis in (vt-normalize-axis axis rank)
                         return-index return-inverse return-counts))
          ((vt-complex-dtype-p (vt-dtype in))
           (%cx-unique in return-index return-inverse return-counts))
          (t (funcall (%cx-orig 'vt-unique) in
                      :return-index return-index
                      :return-inverse return-inverse
                      :return-counts return-counts
                      :axis axis)))))

(defun %cx-key-table (test)
  "把 TEST 的元素装进 EQL 哈希表（键为 %CX-KEY）。"
  (let ((h (make-hash-table :test #'eql))
        (flat (%cx-flat test)))
    (with-float-safe
      (%cx-dispatch (vt-data flat) arr zero
        (declare (ignore zero))
        (let ((off (vt-offset flat))
              (n (vt-size flat)))
          (dotimes (i n)
            (setf (gethash (%cx-key (aref arr (+ off i))) h) t)))))
    h))

(defun %cx-lookup-flags (element table invert)
  "逐元素查表 → :int8 的 0/1（INVERT 取反）。形状与 ELEMENT 相同。"
  (vt-map (lambda (z)
            (let ((hit (if (gethash (%cx-key z) table) 1 0)))
              (if invert (- 1 hit) hit)))
          (ensure-vt element) :dtype :int8))

(defun vt-in1d (t1 t2)
  "逐元素判定是否属于 T2（对标 numpy.in1d）。复数按值相等（NaN 与 NaN 视为相等）。"
  (if (or (vt-complex-dtype-p (vt-dtype (ensure-vt t1)))
          (vt-complex-dtype-p (vt-dtype (ensure-vt t2))))
      ;; 与实数路径一致：结果形状 = 输入形状（不是展平）
      (%cx-lookup-flags (ensure-vt t1) (%cx-key-table t2) nil)
      (funcall (%cx-orig 'vt-in1d) t1 t2)))

(defun vt-isin (element test-elements &key (invert nil) (assume-unique nil))
  "逐元素判定是否属于 TEST-ELEMENTS（对标 numpy.isin）。结果形状同 ELEMENT。"
  (declare (ignore assume-unique))
  (if (or (vt-complex-dtype-p (vt-dtype (ensure-vt element)))
          (vt-complex-dtype-p (vt-dtype (ensure-vt test-elements))))
      (%cx-lookup-flags element (%cx-key-table test-elements) invert)
      (funcall (%cx-orig 'vt-isin) element test-elements :invert invert)))

(defun %cx-set-filter (u table invert)
  "按 KEEP-TABLE 过滤已升序的复数唯一值 U（保持顺序）。"
  (let* ((n (vt-size u))
         (dt (vt-dtype u))
         (res (make-vt (list n) 0 :dtype dt))
         (k 0))
    (with-float-safe
      (%cx-dispatch (vt-data (vt-copy u)) src zero
        (declare (ignore zero))
        (let ((dst (vt-data res)))
          (dotimes (i n)
            (let ((v (aref src i)))
              (when (if invert
                        (not (gethash (%cx-key v) table))
                        (gethash (%cx-key v) table))
                (setf (aref dst k) v)
                (incf k))))))
      (%make-vt :data (subseq (vt-data res) 0 k)
                :shape (list k) :strides '(1) :offset 0 :dtype dt))))

(defun vt-intersect1d (t1 t2)
  "两个 1D 复数张量的交集，结果升序去重（对标 numpy.intersect1d）。"
  (if (or (vt-complex-dtype-p (vt-dtype (ensure-vt t1)))
          (vt-complex-dtype-p (vt-dtype (ensure-vt t2))))
      (%cx-set-filter (vt-unique (ensure-vt t1))
                      (%cx-key-table (vt-unique (ensure-vt t2))) nil)
      (funcall (%cx-orig 'vt-intersect1d) t1 t2)))

(defun vt-setdiff1d (t1 t2)
  "差集 t1 \\ t2，结果升序去重（对标 numpy.setdiff1d）。"
  (if (or (vt-complex-dtype-p (vt-dtype (ensure-vt t1)))
          (vt-complex-dtype-p (vt-dtype (ensure-vt t2))))
      (%cx-set-filter (vt-unique (ensure-vt t1))
                      (%cx-key-table (vt-unique (ensure-vt t2))) t)
      (funcall (%cx-orig 'vt-setdiff1d) t1 t2)))

(defun vt-union1d (t1 t2)
  "并集，结果升序去重（对标 numpy.union1d）。"
  (if (or (vt-complex-dtype-p (vt-dtype (ensure-vt t1)))
          (vt-complex-dtype-p (vt-dtype (ensure-vt t2))))
      (vt-unique (vt-concatenate 0 (vt-unique (ensure-vt t1))
                                   (vt-unique (ensure-vt t2))))
      (funcall (%cx-orig 'vt-union1d) t1 t2)))

(defun vt-setxor1d (t1 t2)
  "对称差，结果升序去重（对标 numpy.setxor1d）。"
  (if (or (vt-complex-dtype-p (vt-dtype (ensure-vt t1)))
          (vt-complex-dtype-p (vt-dtype (ensure-vt t2))))
      (let* ((a (ensure-vt t1)) (b (ensure-vt t2))
             (u1 (vt-unique a)) (u2 (vt-unique b))
             (only1 (%cx-set-filter u1 (%cx-key-table u2) t))
             (only2 (%cx-set-filter u2 (%cx-key-table u1) t)))
        (vt-unique (vt-concatenate 0 only1 only2)))
      (funcall (%cx-orig 'vt-setxor1d) t1 t2)))

;;; ==================================================================
;;; 6. searchsorted（字典序二分）
;;; ==================================================================

(defun %cx-searchsorted-run (tbl tvals right out)
  "在已升序的复数表 TBL 中对每个 TVALS 二分。RIGHT 为 T 时找第一个 > 的位置。"
  (%cx-dispatch tbl arr zero
    (declare (ignore zero))
    (%cx-dispatch tvals varr zero2
      (declare (ignore zero2))
      (let ((n (length arr))
            (nv (length varr)))
        (declare (type fixnum n nv))
        (dotimes (k nv)
          (declare (type fixnum k))
          (let ((v (aref varr k))
                (lo 0)
                (hi n))
            (declare (type fixnum lo hi))
            (loop while (< lo hi) do
              (let ((mid (the fixnum (ash (+ lo hi) -1))))
                (declare (type fixnum mid))
                ;; 两种 side 的「收缩方向」不同，不能共用一个谓词：
                ;;   :left  → arr[mid] <  v 时 lo 右移，否则 hi 左移（停在第 1 个 >= v）
                ;;   :right → v < arr[mid]  时 hi 左移，否则 lo 右移（停在第 1 个 >  v）
                (if right
                    (if (%clt v (aref arr mid))
                        (setf hi mid)
                        (setf lo (1+ mid)))
                    (if (%clt (aref arr mid) v)
                        (setf lo (1+ mid))
                        (setf hi mid)))))
            (setf (aref out k) lo)))))))

(defun vt-searchsorted (tensor values &key (side :left))
  "在有序张量中查找插入点（对标 numpy.searchsorted）。复数按字典序比较。

   :left  → 第一个 >= v 的位置；:right → 第一个 > v 的位置。
   NaN 参与比较时视为「不可比」：落在表尾 NaN 区之前（与排序的 NaN 末尾一致）。"
  (let ((tv (ensure-vt tensor))
        (vv (ensure-vt values)))
    (if (or (vt-complex-dtype-p (vt-dtype tv))
            (vt-complex-dtype-p (vt-dtype vv)))
        (let* ((dt (vt-promote-type (vt-dtype tv) (vt-dtype vv)))
               (dt (if (vt-complex-dtype-p dt) dt :complex128))
               (tbl (%cx-flat (vt-astype tv dt)))
               (val (%cx-flat (vt-astype vv dt)))
               (nv (vt-size val))
               (out (make-array nv :element-type '(signed-byte 64)))
               (right (eq side :right)))
          (with-float-safe
            (%cx-searchsorted-run (vt-data tbl) (vt-data val) right out))
          (%make-vt :data out :shape (vt-shape vv)
                    :strides (vt-compute-strides (vt-shape vv))
                    :offset 0 :dtype :int64))
        (funcall (%cx-orig 'vt-searchsorted) tensor values :side side))))

;;; ==================================================================
;;; 7. 缺失值归约（nan 族）
;;; ==================================================================
;;; numpy 语义（2.4.6 实测，见 §16.18 表）：
;;;   · nansum/nanprod 跳过 NaN，空归约区取单位元（0 / 1）；
;;;   · nanmax/nanmin 跳过 NaN，**全 NaN 切片 → NaN+0j**（不报错）；
;;;   · nanargmax/nanargmin 跳过 NaN，全 NaN 切片 → 报错；
;;;   · nanmean 保复数、nanvar/nanstd **恒实数**（|x-mean|² 的均值）；
;;;   · nanmedian 取字典序中位数（偶数个取中间两个的算术平均），
;;;     全 NaN → NaN+0j。
;;;
;;; 实现策略：用「掩码 + 单位元替换 + 既有复数归约内核」的向量化写法，
;;; 一次拿到所有 axis/keepdims 组合，且**复用第一轮已优化的内核**
;;; （vt-sum / vt-amax 的复数快路径），不再写第二套 fiber 循环。

(defun %cx-to-out (r out name)
  "把已算好的 R 按 :out 硬契约写回（形状/dtype 严格相等由 vt-check-out 把关）。"
  (if out
      (let ((checked (vt-check-out out (vt-shape r) (vt-dtype r) :op-name name)))
        (vt-copy-into checked r)
        checked)
      r))

(defun %cx-nan-sentinel (dt sign)
  "全 ±Inf 的复数哨兵：SIGN 为 +1 → (+inf,+inf)，-1 → (-inf,-inf)。
   它比任何有限值都「大/小」（字典序下实部就分出胜负），
   因此可以被 amax/amin 当作「已被排除的 NaN」占位。"
  (let* ((rdt (vt-complex-real-dtype dt))
         (v (if (eq rdt :float32)
                (if (plusp sign) +vt-sfloat-pos-inf+ +vt-sfloat-neg-inf+)
                (if (plusp sign) +vt-dfloat-pos-inf+ +vt-dfloat-neg-inf+))))
    (complex v v)))

(defun %cx-nan-reduce (name tensor axis keepdims dtype out op)
  "nan 族归约的公共外壳。OP ∈ :sum :prod :max :min :mean。"
  (with-float-safe
    (let* ((in (ensure-vt tensor))
           (in-dt (vt-dtype in))
           (rdt (vt-complex-real-dtype in-dt))
           (dt (or dtype in-dt))
           (mask (vt-isnan in))
           (one (vt-cast 1 dt))
           (zero (vt-cast 0 dt))
           (raw
             (case op
               (:sum (vt-sum (vt-where mask zero in :dtype dt)
                             :axis axis :keepdims keepdims :dtype dt))
               (:prod (vt-prod (vt-where mask one in :dtype dt)
                               :axis axis :keepdims keepdims :dtype dt))
               (:max (vt-amax (vt-where mask (%cx-nan-sentinel dt -1) in :dtype dt)
                              :axis axis :keepdims keepdims :dtype dt))
               (:min (vt-amin (vt-where mask (%cx-nan-sentinel dt +1) in :dtype dt)
                              :axis axis :keepdims keepdims :dtype dt))
               (:mean (vt-sum (vt-where mask zero in :dtype dt)
                              :axis axis :keepdims keepdims :dtype dt)))))
      (unless (vt-complex-dtype-p dt)
        (error "~a: 复数输入的 :dtype 必须是复数类型，收到 ~a" name dt))
      (cond
        ;; sum / prod：单位元语义天然正确（全 NaN → 0 / 1），无需修补
        ((member op '(:sum :prod))
         (%cx-to-out raw out name))
        ;; max / min：全 NaN 切片用 count==0 检测 → NaN+0j
        ((member op '(:max :min))
         (let* ((cnt (vt-sum (vt-astype (vt-logical-not mask) rdt)
                             :axis axis :keepdims keepdims :dtype rdt))
                (empty (vt-= cnt (vt-cast 0 rdt)))
                (nanv (complex (%cx-nan-of raw) 0)))
           (%cx-to-out (vt-where empty nanv raw :dtype dt) out name)))
        ;; mean：求和 / 计数；count==0 → NaN+NaNj
        (t
         (let* ((cnt (vt-sum (vt-astype (vt-logical-not mask) rdt)
                             :axis axis :keepdims keepdims :dtype rdt))
                (empty (vt-= cnt (vt-cast 0 rdt)))
                (avg (vt-/ raw cnt))
                (nanv (%cx-complex-nan-of dt)))
           (%cx-to-out (vt-where empty nanv avg :dtype dt) out name)))))))

;;; ---- nanmean ------------------------------------------------------
(defun vt-nanmean (tensor &key axis keepdims dtype out)
  "忽略 NaN 的算术平均（对标 numpy.nanmean）。

   复数（§16.18）：结果为**复数**（complex64 → complex64）；
   全 NaN 切片 → NaN+NaNj（与 numpy 一致）。"
  (if (%cx-tensor-p tensor)
      (%cx-nan-reduce "VT-NANMEAN" tensor axis keepdims dtype out :mean)
      (funcall (%cx-orig 'vt-nanmean) tensor :axis axis :keepdims keepdims
               :dtype dtype :out out)))

;;; ---- nanvar / nanstd（恒实数）--------------------------------------
(defun %cx-nan-var (name tensor axis keepdims ddof dtype out std-p)
  "复数 nanvar/nanstd：mean(|x-mean|²) 跳过 NaN，结果恒实数（§16.7 同理）。"
  (with-float-safe
    (let* ((in (ensure-vt tensor))
           (in-dt (vt-dtype in))
           (rdt (vt-complex-real-dtype in-dt))
           (dt (or dtype rdt))
           (mask (vt-isnan in))
           (not-nan (vt-astype (vt-logical-not mask) rdt))
           (clean (vt-where mask (vt-cast 0 in-dt) in :dtype in-dt))
           (cnt (vt-sum not-nan :axis axis :keepdims keepdims :dtype rdt))
           (cnt-b (vt-sum not-nan :axis axis :keepdims t :dtype rdt))
           (sums (vt-sum clean :axis axis :keepdims t :dtype in-dt))
           (mean (vt-/ sums cnt-b))
           (dev (vt-- clean mean))
           (sq (vt-square (vt-abs dev)))          ; |x-mean|²，实数
           (sqm (vt-* sq not-nan))
           (sum2 (vt-sum sqm :axis axis :keepdims keepdims :dtype rdt))
           (ddofv (vt-cast ddof rdt))
           (den (vt-- cnt ddofv))
           (bad (vt-<= cnt ddofv))
           (nanv (%cx-nan-of in))
           (var (vt-/ sum2 den)))
      (when (vt-complex-dtype-p dt)
        (error "~a: 复数方差的结果 dtype 必须是实数类型，收到 ~a" name dt))
      (let ((r (vt-where bad nanv var :dtype rdt)))
        (%cx-to-out (if std-p (vt-sqrt r) r) out name)))))

(defun vt-nanvar (tensor &key axis keepdims (ddof 0) dtype out)
  "忽略 NaN 的方差（对标 numpy.nanvar）。复数结果为**实数**（|x-mean|² 的均值）。"
  (if (%cx-tensor-p tensor)
      (%cx-nan-var "VT-NANVAR" tensor axis keepdims ddof dtype out nil)
      (funcall (%cx-orig 'vt-nanvar) tensor :axis axis :keepdims keepdims
               :ddof ddof :dtype dtype :out out)))

(defun vt-nanstd (tensor &key axis keepdims (ddof 0) dtype out)
  "忽略 NaN 的标准差（对标 numpy.nanstd）。复数结果为**实数**。"
  (if (%cx-tensor-p tensor)
      (%cx-nan-var "VT-NANSTD" tensor axis keepdims ddof dtype out t)
      (funcall (%cx-orig 'vt-nanstd) tensor :axis axis :keepdims keepdims
               :ddof ddof :dtype dtype :out out)))

;;; ---- nanargmax / nanargmin ----------------------------------------
(defun %cx-nan-arg-run (data shape ax op)
  "跳过 NaN 的 argmax/argmin 内核。返回 fixnum 向量。
   全 NaN 切片 → 报错（对标 numpy 的 ValueError: All-NaN slice encountered）。"
  (%cx-dispatch data d zero
    (declare (ignore zero))
    (flet ((better (acc v)
             (or (null acc)
                 (if (eq op :nanargmax) (%clt acc v) (%cgt acc v)))))
      (if (null ax)
          (let ((n (the fixnum (length d)))
                (acc nil) (best 0) (cnt 0))
            (declare (type fixnum n best cnt))
            (dotimes (i n)
              (declare (type fixnum i))
              (let ((v (aref d i)))
                (unless (%cx-nan-p v)
                  (incf cnt)
                  (when (better acc v) (setf acc v best i)))))
            (when (zerop cnt)
              (error "vt-~(~a~): All-NaN slice encountered" op))
            (vector best))
          (multiple-value-bind (head dim ax-stride) (%cx-fiber-geom shape ax)
            (declare (type fixnum head dim ax-stride))
            (let ((res (make-array (the fixnum (* head ax-stride))
                                   :element-type 'fixnum)))
              (declare (type (simple-array fixnum (*)) res))
              (dotimes (h head)
                (declare (type fixnum h))
                (let ((hb (the fixnum (* h dim ax-stride)))
                      (row (the fixnum (* h ax-stride))))
                  (declare (type fixnum hb row))
                  (dotimes (ti ax-stride)
                    (declare (type fixnum ti))
                    (let ((acc nil) (best 0) (cnt 0)
                          (p (the fixnum (+ hb ti))))
                      (declare (type fixnum best cnt p))
                      (dotimes (j dim)
                        (declare (type fixnum j))
                        (let ((v (aref d p)))
                          (unless (%cx-nan-p v)
                            (incf cnt)
                            (when (better acc v) (setf acc v best j))))
                        (incf p ax-stride))
                      (when (zerop cnt)
                        (error "vt-~(~a~): All-NaN slice encountered" op))
                      (setf (aref res (the fixnum (+ row ti))) best)))))
              res))))))

(defun %cx-nan-arg (name tensor axis keepdims out op)
  (with-float-safe
    (let* ((in (ensure-vt tensor))
           (shape (vt-shape in))
           (rank (length shape))
           (axes (vt-normalize-axes axis rank)))
      (when (cdr axes)
        (error "~a: 不支持多轴（对标 numpy：arg 归约只接受单个整数 axis）" name))
      (when (zerop (vt-size in))
        (error "~a: attempt to get arg of an empty sequence" name))
      (let* ((flat (%cx-flat in))
             (ax (car axes))
             (out-shape (%cx-keepdims-shape shape axes keepdims))
             (res (if out (vt-check-out out out-shape :int64 :op-name name)
                      (make-vt out-shape 0 :dtype :int64)))
             (vals (%cx-nan-arg-run (vt-data flat) shape ax op)))
        (%cx-write-flat res vals :int64)))))

;;; ---- nanmedian ----------------------------------------------------
(defun %cx-nan-median-fiber (data base stride n dt)
  "一条 fiber 的中位数（跳过 NaN，字典序）。全 NaN → NaN+0j（numpy 语义）。"
  (let ((tmp (make-array (max n 1) :element-type (array-element-type data)))
        (cnt 0))
    (declare (type fixnum cnt))
    (dotimes (i n)
      (let ((v (aref data (+ base (* i stride)))))
        (unless (%cx-nan-p v)
          (setf (aref tmp cnt) v)
          (incf cnt))))
    (cond
      ((zerop cnt) (complex (%cx-nan-of (vt-cast 0 dt)) 0))
      (t
       (%cx-sort-fiber tmp 0 cnt)
       (if (oddp cnt)
           (aref tmp (floor cnt 2))
           (let ((a (aref tmp (1- (floor cnt 2))))
                 (b (aref tmp (floor cnt 2))))
             (* 0.5s0 (+ a b))))))))

(defun vt-nanmedian (tensor &key axis keepdims out dtype)
  "忽略 NaN 的中位数（对标 numpy.nanmedian）。复数按字典序；偶数个取中间两值均值。
   全 NaN 切片 → NaN+0j（numpy 语义）。"
  (let ((in (ensure-vt tensor)))
    (unless (vt-complex-dtype-p (vt-dtype in))
      (return-from vt-nanmedian
        (funcall (%cx-orig 'vt-nanmedian) tensor :axis axis :keepdims keepdims
                 :out out)))
    (with-float-safe
      (let* ((shape (vt-shape in))
             (rank (length shape))
             (dt (or dtype (vt-dtype in)))
             (axes (vt-normalize-axes axis rank))
             (flat (%cx-flat in))
             (data (vt-data flat))
             (src-off (vt-offset flat))
             (out-shape (%cx-keepdims-shape shape axes keepdims))
             (res (if out (vt-check-out out out-shape dt :op-name "VT-NANMEDIAN")
                      (make-vt out-shape 0 :dtype dt))))
        (cond
          ;; 展平（含 0 维）：整张表一个中位数
          ((or (null axes) (null shape))
           (setf (aref (vt-data res) (vt-offset res))
                 (vt-cast (%cx-nan-median-fiber data src-off 1 (vt-size flat) dt) dt))
           res)
          ;; 单轴：逐 fiber 算进 values 向量，再按 :out 契约统一写回
          ((null (cdr axes))
           (multiple-value-bind (head dim ax-stride) (%cx-fiber-geom shape (car axes))
             (declare (type fixnum head dim ax-stride))
             (let ((vals (make-array (the fixnum (* head ax-stride))))
                   (k 0))
               (declare (type fixnum k))
               (dotimes (h head)
                 (declare (type fixnum h))
                 (let ((row (the fixnum (* h dim ax-stride))))
                   (declare (type fixnum row))
                   (dotimes (ti ax-stride)
                     (declare (type fixnum ti))
                     (setf (aref vals k)
                           (%cx-nan-median-fiber data (+ src-off row ti)
                                                 ax-stride dim dt))
                     (incf k))))
               (%cx-write-flat res vals dt)
               res)))
          (t (error "vt-nanmedian: 暂不支持多轴归约（numpy 支持；此处用 axis=nil 或单轴）")))))))

;;; ==================================================================
;;; 8. 卷积 / 互相关（复数）
;;; ==================================================================
;;; 原实现的结果缓冲是 (simple-array double-float)，复数值写不进去
;;; （TYPE-ERROR）。这里给复数单独一条内核：累加器是复数，三种 mode
;;; 的切片规则与实数路径**逐字相同**，只是元素类型换成复数。

(defun %cx-correlate-full (a v conj-p)
  "完整互相关数组（长度 n+m-1）。A、V 均为复数连续副本、offset=0。

   CONJ-P 为真时对第二个操作数取共轭 —— 这是 **numpy.correlate 的定义**
   （c[k] = Σ a[n+k]·conj(v[n])），而 numpy.convolve 用的是不取共轭的
   内核（它等价于 correlate(a, v[::-1]) 但**不**共轭）。
   实数输入下二者等价，因此这条区分只在复数上可观测。"
  (let* ((ad (vt-data a))
         (vd (vt-data v))
         (n (length ad))
         (m (length vd))
         (full-len (the fixnum (+ n m -1)))
         (full (make-array full-len :element-type (array-element-type ad))))
    (declare (type fixnum n m full-len))
    (with-float-safe
      (%cx-dispatch ad s zero
        (%cx-dispatch vd w zero2
          (declare (ignore zero2))
          (let ((offset (the fixnum (1- m))))
            (loop for k fixnum from (- offset) below n
                  for i fixnum from 0
                  do (let ((sum zero))
                       (loop for j fixnum from (the fixnum (max 0 (- k)))
                             below (the fixnum (min m (- n k)))
                             do (incf sum (* (aref s (the fixnum (+ j k)))
                                             (if conj-p (%cconj (aref w j)) (aref w j)))))
                       (setf (aref full i) sum)))))))
    full))

(defun %cx-correlate (a v mode-kw &key (conj-p t))
  "复数 1D 互相关（三 mode）。CONJ-P 见 %CX-CORRELATE-FULL。"
  (let* ((av (ensure-vt a))
         (bv (ensure-vt v))
         (dt (vt-promote-type (vt-dtype av) (vt-dtype bv)))
         (dt (if (vt-complex-dtype-p dt) dt :complex128))
         (af (%cx-flat (vt-copy (vt-astype av dt))))
         (vf (%cx-flat (vt-copy (vt-astype bv dt))))
         (n (vt-size af))
         (m (vt-size vf))
         (full (%cx-correlate-full af vf conj-p))
         (full-len (length full))
         (offset (1- m)))
    (declare (type fixnum n m full-len offset))
    (flet ((slice (len start)
             (%make-vt :data (subseq full start (min full-len (+ start len)))
                       :shape (list (max 0 len)) :strides '(1)
                       :offset 0 :dtype dt)))
      (ecase mode-kw
        (:full (%make-vt :data full :shape (list full-len) :strides '(1)
                         :offset 0 :dtype dt))
        (:valid (slice (max 0 (1+ (- n m))) offset))
        (:same (slice (max n m) (floor (- full-len (max n m)) 2)))))))

(defun vt-correlate (a v &key (mode :full))
  "1D 互相关（对标 numpy.correlate）。复数走专用内核（累加器为复数）。

   复数语义（与 numpy 一致）：c[k] = Σ a[n+k]·conj(v[n])，即**第二个操作数取共轭**。
   注意默认值是 :full（与 numpy 的 :valid 不同，见原实现文档）。"
  (if (or (vt-complex-dtype-p (vt-dtype (ensure-vt a)))
          (vt-complex-dtype-p (vt-dtype (ensure-vt v))))
      (let ((mk (if (null mode) :full (%normalize-mode-keyword mode))))
        (%cx-correlate a v mk :conj-p t))
      (funcall (%cx-orig 'vt-correlate) a v :mode mode)))

(defun vt-convolve (a v &key (mode :full))
  "1D 卷积（对标 numpy.convolve）。复数：c[k] = Σ a[n]·v[k-n]，**不取共轭**。

   实数路径转发原实现；复数复用互相关内核（对 flip(v) 不共轭地相关），
   与 numpy 的 convolve/correlate 分工一致。"
  (if (or (vt-complex-dtype-p (vt-dtype (ensure-vt a)))
          (vt-complex-dtype-p (vt-dtype (ensure-vt v))))
      (let ((mk (if (null mode) :full (%normalize-mode-keyword mode))))
        (%cx-correlate a (vt-flip (ensure-vt v)) mk :conj-p nil))
      (funcall (%cx-orig 'vt-convolve) a v :mode mode)))

;;; ==================================================================
;;; 9. 张量缩并（tensordot / inner）
;;; ==================================================================
;;; 实数路径靠 vt-einsum 实现，而 einsum 的 dtype 白名单只含 4 种实数
;;; 类型（§16.11 的「第一轮不支持」清单）。复数这里绕开 einsum：
;;;   transpose（把收缩轴挪到 a 末尾 / b 开头）→ reshape 成 2D →
;;;   **复数专用 matmul 内核** → reshape 回自由轴形状。
;;; 数学上与 einsum 等价，且完全复用第一轮已优化的矩阵乘。

(defun %cx-tensordot-axes (axes ar br)
  "把 AXES 规范化成 (values a-axes b-axes)（负轴按各自秩归一化）。"
  (cond
    ((integerp axes)
     (unless (and (>= axes 0) (<= axes ar) (<= axes br))
       (error "vt-tensordot: axes ~a 超出可收缩范围（a 秩 ~a，b 秩 ~a）" axes ar br))
     (values (loop for i from (- ar axes) below ar collect i)
             (loop for i below axes collect i)))
    ((and (listp axes) (= 2 (length axes)))
     (flet ((norm (xs rank which)
              (mapcar (lambda (x)
                        (let ((ax (if (minusp x) (+ x rank) x)))
                          (unless (and (>= ax 0) (< ax rank))
                            (error "vt-tensordot: ~a 的轴 ~a 越界（秩 ~a）" which x rank))
                          ax))
                      (if (listp xs) xs (list xs)))))
       (values (norm (first axes) ar "a")
               (norm (second axes) br "b"))))
    (t (error "vt-tensordot: axes 必须是整数或 (a-axes b-axes)，收到 ~a" axes))))

(defun %cx-tensordot (a b axes dtype out)
  "复数张量缩并：transpose → reshape(2D) → matmul → reshape。"
  (let* ((av (ensure-vt a))
         (bv (ensure-vt b))
         (ar (length (vt-shape av)))
         (br (length (vt-shape bv))))
    (multiple-value-bind (a-axes b-axes) (%cx-tensordot-axes axes ar br)
      (unless (= (length a-axes) (length b-axes))
        (error "vt-tensordot: 两侧收缩轴数不等（~a vs ~a）"
               (length a-axes) (length b-axes)))
      (let* ((a-shape (vt-shape av))
             (b-shape (vt-shape bv))
             (a-dims (mapcar (lambda (i) (nth i a-shape)) a-axes))
             (b-dims (mapcar (lambda (i) (nth i b-shape)) b-axes)))
        (unless (equal a-dims b-dims)
          (error "vt-tensordot: 收缩轴长度不一致（a ~a vs b ~a）" a-dims b-dims))
        (let* ((a-free (loop for i below ar unless (member i a-axes) collect i))
               (b-free (loop for i below br unless (member i b-axes) collect i))
               (free-a-dims (mapcar (lambda (i) (nth i a-shape)) a-free))
               (free-b-dims (mapcar (lambda (i) (nth i b-shape)) b-free))
               (m (reduce #'* free-a-dims :initial-value 1))
               (n (reduce #'* free-b-dims :initial-value 1))
               (k (reduce #'* a-dims :initial-value 1))
               (at (vt-transpose av (append a-free a-axes)))
               (bt (vt-transpose bv (append b-axes b-free)))
               (a2 (vt-reshape at (list m k)))
               (b2 (vt-reshape bt (list k n)))
               (r2 (vt-matmul a2 b2))
               (r (vt-reshape r2 (append free-a-dims free-b-dims))))
          (%cx-to-out (if dtype (vt-astype r dtype) r) out "VT-TENSORDOT"))))))

(defun vt-tensordot (a b &key (axes 2))
  "张量缩并（对标 numpy.tensordot）。复数走 transpose+reshape+matmul 专用路径。"
  (if (or (vt-complex-dtype-p (vt-dtype (ensure-vt a)))
          (vt-complex-dtype-p (vt-dtype (ensure-vt b))))
      (%cx-tensordot a b axes nil nil)
      (funcall (%cx-orig 'vt-tensordot) a b :axes axes)))

(defun vt-inner (a b &key dtype out)
  "内积（对标 numpy.inner）。复数**不取共轭**（numpy 亦如此；共轭点积见 vt-vdot）。"
  (if (or (vt-complex-dtype-p (vt-dtype (ensure-vt a)))
          (vt-complex-dtype-p (vt-dtype (ensure-vt b))))
      (let* ((av (ensure-vt a))
             (bv (ensure-vt b))
             (ar (length (vt-shape av)))
             (br (length (vt-shape bv))))
        (cond
          ;; 0 维：numpy.inner 退化为广播乘法
          ((or (zerop ar) (zerop br))
           (%cx-to-out (vt-* av bv :dtype dtype) out "VT-INNER"))
          (t
           (%cx-tensordot av bv (list (list (1- ar)) (list (1- br))) dtype out))))
      (funcall (%cx-orig 'vt-inner) a b :dtype dtype :out out)))

;;; ==================================================================
;;; 10. 缺失值累积 与 共轭点积（补第一轮的漏网）
;;; ==================================================================

(defun %cx-nan-fill (in fill)
  "把 NaN 元素替换成 FILL（复数标量），其余保持。用于 nancumsum/nancumprod。"
  (let ((dt (vt-dtype in)))
    (with-float-safe
      (vt-where (vt-isnan in) (vt-cast fill dt) in :dtype dt))))

(defun vt-nancumsum (tensor &key axis dtype out)
  "沿 AXIS 累积和，NaN 视为 0（对标 numpy.nancumsum）。

   复数：原实现走 `(vt-astype input :float64)`（取实部），会**静默丢掉虚部**。
   这里改为「NaN→0+0i 后调用复数 cumsum」，语义与 numpy 完全相同，
   且复用第一轮已优化的扫描内核。"
  (if (vt-complex-dtype-p (vt-dtype (ensure-vt tensor)))
      (vt-cumsum (%cx-nan-fill tensor 0) :axis axis :dtype dtype :out out)
      (funcall (%cx-orig 'vt-nancumsum) tensor :axis axis :dtype dtype :out out)))

(defun vt-nancumprod (tensor &key axis dtype out)
  "沿 AXIS 累积积，NaN 视为 1（对标 numpy.nancumprod）。复数同 vt-nancumsum。"
  (if (vt-complex-dtype-p (vt-dtype (ensure-vt tensor)))
      (vt-cumprod (%cx-nan-fill tensor 1) :axis axis :dtype dtype :out out)
      (funcall (%cx-orig 'vt-nancumprod) tensor :axis axis :dtype dtype :out out)))

(defun vt-vdot (a b &key dtype out)
  "向量点积（对标 numpy.vdot）。复数：**第一个操作数取共轭**
   （vdot(a,b) = Σ conj(a)·b，与 numpy 一致；vt-dot 不取共轭）。"
  (if (or (vt-complex-dtype-p (vt-dtype (ensure-vt a)))
          (vt-complex-dtype-p (vt-dtype (ensure-vt b))))
      (vt-dot (vt-conj (ensure-vt a)) (ensure-vt b) :dtype dtype :out out)
      (funcall (%cx-orig 'vt-vdot) a b :dtype dtype :out out)))

;;; ------------------------------------------------------------------
;;; 11. 明确拒绝（与 numpy 一致，但给出可读错误）
;;; ------------------------------------------------------------------

(defun vt-digitize (x bins &key (right nil))
  "分箱（对标 numpy.digitize）。**复数不支持** —— numpy 自己也报
   `TypeError: x may not be complex`（分箱要求全序，而复数的字典序不是数值序）。
   这里给出确定性错误，取代原先从类型声明里冒出来的 TYPE-ERROR。"
  (if (or (vt-complex-dtype-p (vt-dtype (ensure-vt x)))
          (vt-complex-dtype-p (vt-dtype (ensure-vt bins))))
      (error "vt-digitize: 复数不可分箱（numpy 对复数同样报 TypeError: x may not be complex）")
      (funcall (%cx-orig 'vt-digitize) x bins :right right)))

;;; ==================================================================
;;; 12. 补齐 numpy **支持**复数的算子（v1.3）
;;; ==================================================================
;;; 这一节修的是第二类缺陷：输入含复数时**静默只算实部**。它们既没报错、
;;; 也没给出正确结果，是三轮里最危险的一类（用户完全看不出来）。
;;; 判据统一为 numpy 2.4.6 实测：numpy 支持的，我们实现；numpy 拒绝的，
;;; 我们给出确定性错误（见第 13 节）。

;;; ---- 12.1 值级内核 -------------------------------------------------
;;; 全部运行在 vt-fast-map 的内联循环里（外层已 with-float-safe），
;;; 因此内部只用免屏蔽快判 %CX-NAN-P。白名单登记在
;;; map-reduce.lisp 的 +cx-inline-ops+（未登记则走 vt-map 兜底，语义相同）。

(declaim (inline %cop-expm1 %cop-log1p %cop-sinc %cop-sign))

(defun %cop-expm1 (z)
  "exp(z) - 1（对标 numpy.expm1 的复数分支）。

   |z| 大时直接 exp(z)-1；|z| 小时灾难性抵消会把结果精度吃光
   （z=1d-10 时 exp(z)-1 只剩 1 位有效数字），改用级数
   expm1(z) = z·Σ_{k≥0} z^k/(k+1)!（Horner 形式，12 项对 |z|<0.5 足够：
   0.5^12/13! ≈ 1e-13，远低于 double 的 eps 累积）。"
  (cond ((%cx-nan-p z) (%cx-nan-complex z))
        ((< (abs z) 0.5)
         ;; 级数**固定用 double 算**（12 项 1.0d0/(k+2) 的系数都是 double，
         ;; 强行降到 single 只会掉精度）。因此这里把 z 显式提升成 d0 再声明类型：
         ;; 原写法直接 `(declare (type (complex double-float) term sum))` 而
         ;; init 是 z —— 对 complex64 输入这是**谎报**：SBCL 不仅报编译期
         ;; 「Derived type (complex single-float) conflicting with asserted
         ;; (complex double-float)」让整个库编译失败，运行期 (safety 1) 还会
         ;; 在 |z|<0.5 的 complex64 上直接抛类型错误。
         (let* ((zd (complex (coerce (realpart z) 'double-float)
                             (coerce (imagpart z) 'double-float)))
                (term zd)
                (sum zd))
           (declare (type (complex double-float) zd term sum))
           (dotimes (k 12 sum)
             (setf term (* term zd (/ 1.0d0 (+ k 2))))
             (incf sum term))))
        (t (- (exp z) 1))))

(defun %cop-log1p (z)
  "log(1 + z)（对标 numpy.log1p 的复数分支）。

   与 expm1 同源：|z| 小时 1+z 的有效位被吃掉，必须走级数
   log1p(z) = z - z²/2 + z³/3 - … = z·Σ_{k≥0} (-z)^k/(k+1)。
   z = -1 时 1+z = 0，交给 %COP-LOG 处理极点（-inf + 0i）。

   两个数值细节（与 numpy 2.4.6 逐点核对过）：
     · 分母是**调和**的 k，不是阶乘 k!。写成 `term *= (-z)/(k+2)` 得到的是
       z·Σ(-z)^k/(k+1)!，那是另一个函数 —— 实测 |z|=0.112 时虚部偏 0.4%
       （numpy: 0.09634224+0.04542328i，错版: 0.0962934+0.045223i）。
     · 线性分母收敛慢（|z|^n/n），窗口收到 |z|<0.25 并取 24 项：第 24 项
       量级 0.25^24/24 ≈ 1.5e-16，相对误差已达 double 的 eps。|z|≥0.25 时
       1+z 不存在抵消问题，直接走 %COP-LOG（与 numpy 分支切割一致）。"
  (cond ((%cx-nan-p z) (%cx-nan-complex z))
        ((< (abs z) 0.25)
         ;; 同 %COP-EXPM1：级数固定 double（见那里的说明），z 先显式提升。
         (let* ((zd (complex (coerce (realpart z) 'double-float)
                             (coerce (imagpart z) 'double-float)))
                (term zd)
                (sum zd))
           (declare (type (complex double-float) zd term sum))
           (dotimes (k 23 sum)
             (setf term (* term (- zd)
                           (/ (float (1+ k) 1d0) (float (+ k 2) 1d0))))
             (incf sum term))))
        (t (%cop-log (+ 1 z)))))

(defun %cop-sinc (z)
  "归一化 sinc：sin(πz)/(πz)，sinc(0)=1（对标 numpy.sinc 的复数分支）。

   |z| 很小时 sin(πz)/(πz) → 1 - (πz)²/6，直接算会 0/0。"
  (cond ((%cx-nan-p z) (%cx-nan-complex z))
        ((< (abs z) 1.0d-8)
         (- 1 (* (/ (* pi pi) 6.0d0) z z)))
        (t (let ((pz (* pi z)))
             (/ (sin pz) pz)))))

(defun %cop-sign (z)
  "复数的「符号」：z/|z|（单位相量），0 → 0（对标 numpy.sign）。

   numpy 的复数 sign 不是取实部符号，而是投影到单位圆 ——
   这正是它与 %COP-SIGNUM 同源的地方（clvt 的 vt-signum 已如此实现）。"
  (cond ((%cx-nan-p z) (%cx-nan-complex z))
        ((zerop z) (%cx-complex-zero-of :complex128))
        (t (/ z (abs z)))))

;;; ---- 12.2 clip / clamp：字典序夹取（numpy 语义）-------------------
;;; numpy 的 clip 实现是 `np.minimum(np.maximum(a, lo), hi)`，
;;; 而 maximum/minimum 对复数走字典序（§16.8）—— 所以复数的 clip
;;; **不是**按分量裁剪，而是整元素按字典序夹取。实测：
;;;   v=[1+2j, 3-4j, 5+0j], clip(0, 2) → [1+2j, 2+0j, 2+0j]
;;; 第一项虚部 2 超出 2 却没被裁，正是因为比较的是 (1,2) 整体而不是 2。

(defun vt-clip (vt &optional (min-val nil min-p) (max-val nil max-p)
                   &key out dtype)
  "逐元素裁剪（对标 numpy.clip）。复数按**字典序**整元素夹取，不按分量。"
  (let ((v (ensure-vt vt)))
    (if (not (vt-complex-dtype-p (vt-dtype v)))
        ;; 实数路径原样转发。
        ;; ⚠ 不能写 `(funcall orig vt :out out :dtype dtype)`：orig 的签名是
        ;; `(&optional min max &key out dtype)` —— **&optional 在前**，关键字
        ;; 会被当成 min-val 的位置实参（`min-p = T, min-val = :OUT`），
        ;; 于是「无界参数」这条最常用的路径整天报错。必须把两个位置参数
        ;; 显式填满（nil nil 与「不传」在 orig 里等价：都走 copy 分支）。
        (cond ((and min-p max-p)
               (funcall (%cx-orig 'vt-clip) vt min-val max-val :out out :dtype dtype))
              (min-p
               ;; 只给下限：orig 会立刻报错（numpy 语义），不必带关键字
               (funcall (%cx-orig 'vt-clip) vt min-val))
              (t
               (funcall (%cx-orig 'vt-clip) vt nil nil :out out :dtype dtype)))
        (progn
          (when (and min-p (not max-p))
            (error "vt-clip: 只提供 MIN-VAL 而未提供 MAX-VAL（numpy 语义：\
clip() missing a_max；若只要下限请显式传 MAX-VAL 为 nil）"))
          (let* ((dt (%cx-result-complex-dtype (vt-dtype v) dtype "VT-CLIP"))
                 (r (if (and (null min-val) (null max-val))
                        (vt-copy v)
                        (let ((acc v))
                          (when min-val
                            (setf acc (vt-maximum acc (vt-cast min-val dt) :dtype dt)))
                          (when max-val
                            (setf acc (vt-minimum acc (vt-cast max-val dt) :dtype dt)))
                          acc))))
            (%cx-to-out r out "VT-CLIP"))))))

(defun vt-clamp (vt &optional (min-val nil min-p) (max-val nil max-p)
                    &key out dtype)
  "vt-clip 的别名（对标 torch.clamp）。复数语义同 vt-clip（字典序）。"
  (let ((v (ensure-vt vt)))
    ;; VT-CLIP 的 lambda list 是 (&optional min-val max-val &key out dtype)：
    ;; &OPTIONAL 在 &KEY 之前，转发时**必须把位置参数填满到 max-val**，否则
    ;; 第一个关键字会被当成 max-val 绑走（SBCL 报 "called with odd number of
    ;; keyword arguments"，运行期直接报错；这也是编译期 WARNING → ASDF 编译失败
    ;; 的一条来源）。缺少 max-val 时显式填 NIL 占位。
    (if (vt-complex-dtype-p (vt-dtype v))
        (cond ((and min-p max-p) (vt-clip v min-val max-val :out out :dtype dtype))
              (min-p (vt-clip v min-val nil :out out :dtype dtype))
              (t (vt-clip v nil nil :out out :dtype dtype)))
        ;; 实数分支同样处理；顺带修 min-p 分支静默丢弃 :out/:dtype 的 bug。
        (cond ((and min-p max-p)
               (funcall (%cx-orig 'vt-clamp) vt min-val max-val :out out :dtype dtype))
              (min-p
               (funcall (%cx-orig 'vt-clamp) vt min-val nil :out out :dtype dtype))
              (t
               (funcall (%cx-orig 'vt-clamp) vt nil nil :out out :dtype dtype))))))

;;; ---- 12.3 standardize：按复数均值居中、按实标准差缩放 --------------
;;; 复数的 std 恒为实数（§16.7），所以 (x-mean)/std 是「复数 / 实数」，
;;; 结果仍是复数。原实现把 final-dtype 定成 float64，导致 vt-mean 收到
;;; 复数输入 + 实数 dtype 而报错（我们第一版就撞上了这个）。

(defun vt-standardize (vt &key axis (ddof 0) dtype out)
  "Z-score 标准化 (x - mean) / std（对标 sklearn.preprocessing.scale）。
   复数：均值是复数、标准差是实数（|x-mean|² 的均值开方），结果保复数。"
  (let ((v (ensure-vt vt)))
    (if (not (vt-complex-dtype-p (vt-dtype v)))
        (funcall (%cx-orig 'vt-standardize) vt :axis axis :ddof ddof
                 :dtype dtype :out out)
        (let* ((rdt (vt-complex-real-dtype (vt-dtype v)))
               (cdt (%cx-result-complex-dtype (vt-dtype v) dtype "VT-STANDARDIZE"))
               (mu (vt-mean v :axis axis :keepdims t :dtype cdt))
               (sd (vt-std v :axis axis :keepdims t :ddof ddof :dtype rdt))
               (centered (vt-- v mu :dtype cdt))
               ;; sd 为 0（常量切片）时按 numpy/sklearn 惯例输出 0 而不是 NaN
               (zero-sd (vt-< (vt-abs sd) (vt-cast 1d-12 rdt)))
               (safe-sd (vt-where zero-sd (vt-cast 1 rdt) sd :dtype rdt))
               (r (vt-/ centered safe-sd))
               (r (vt-where zero-sd (vt-cast 0 cdt) r :dtype cdt)))
          (%cx-to-out r out "VT-STANDARDIZE")))))

;;; ---- 12.4 cov / corrcoef：厄米特协方差 ---------------------------
;;; numpy 的 np.cov 对复数用 X @ X.conj().T（不是 X @ X.T），
;;; 得到厄米特矩阵；corrcoef 再由它归一化。原实现
;;; （%e3-as-1d-float-transform）第一行就 `ensure-vt :dtype :float64`，
;;; 复数的虚部在此被整体丢弃 —— 输出 FLOAT64，静默错答案。

(defun %cx-cov-matrix (m rowvar what)
  "把输入规范成 n-var × n-obs 的**复数**矩阵（对标 numpy.cov 的 rowvar 处理）。"
  (let ((v (ensure-vt m)))
    (when (not (vt-complex-dtype-p (vt-dtype v)))
      (setf v (vt-astype v :complex128)))
    (when (not (eq (vt-dtype v) :complex128))
      (setf v (vt-astype v :complex128)))
    (cond
      ((null (vt-shape v)) (vt-reshape v '(1 1)))
      ((= (vt-order v) 1)
       (if rowvar
           (vt-reshape v (list 1 (vt-size v)))
           (vt-reshape v (list (vt-size v) 1))))
      ((= (vt-order v) 2) (if rowvar v (vt-transpose v)))
      (t (error "~a: 输入至多 2 维，收到 ~a 维" what (vt-order v))))))

(defun %cx-conj-t (x)
  "共轭转置（厄米特转置）X^H。复数线代里所有「转置」都应该是它。"
  (vt-transpose (vt-conj x)))

(defun %cx-cov (m y rowvar ddof bias)
  "复数协方差矩阵（numpy 语义：X·X^H / denom，并厄米特对称化）。"
  (let* ((mm (%cx-cov-matrix m rowvar "vt-cov"))
         (xx (when y (%cx-cov-matrix y rowvar "vt-cov")))
         (stacked (if xx (vt-concatenate 0 mm xx) mm))
         (n-var (first (vt-shape stacked)))
         (n-obs (second (vt-shape stacked)))
         (denom (if bias n-obs (- n-obs ddof)))
         (denom (if (<= denom 0) n-obs denom)))
    (if (= n-var 1)
        ;; 单变量 → 标量方差（实数，但保持复数 dtype 以对齐 numpy 的返回类型）
        (let* ((row (vt-slice stacked '(0)))
               (mu (vt-mean row))
               (dev (vt-- row mu))
               (acc (vt-item (vt-sum (vt-* dev (vt-conj dev))))))
          (make-vt nil (/ acc denom) :dtype :complex128))
        (let* ((x (vt-contiguous stacked))
               (mu (vt-mean x :axis 1 :keepdims t))
               (xc (vt-- x mu))
               (raw (vt-matmul xc (%cx-conj-t xc)))
               ;; 厄米特对称化：消掉浮点误差造成的 C_ji ≠ conj(C_ij)
               (sym (vt-scale (vt-+ raw (%cx-conj-t raw)) 0.5d0)))
          (if (= denom 1)
              sym
              (vt-scale sym (/ 1.0d0 denom)))))))

(defun vt-cov (m &key (y nil y-p) (rowvar t) (ddof nil ddof-p) (bias nil))
  "协方差矩阵（对标 numpy.cov）。复数得到**厄米特**矩阵（X·X^H）。"
  (declare (ignore y-p))
  (let ((ddof (if ddof-p ddof 1)))
    (if (or (%cx-deep-p m) (%cx-deep-p y))
        (%cx-cov m y rowvar ddof bias)
        (funcall (%cx-orig 'vt-cov) m :y y :rowvar rowvar
                 :ddof (if ddof-p ddof 1) :bias bias))))

(defun vt-corrcoef (m &key (y nil y-p) (rowvar t))
  "相关系数矩阵（对标 numpy.corrcoef）。复数结果仍是厄米特矩阵，
   对角为 1.0+0.0j（实数 1）。"
  (declare (ignore y-p))
  (if (or (%cx-deep-p m) (%cx-deep-p y))
      (let* ((c (%cx-cov m y rowvar 1 nil))
             (shape (vt-shape c)))
        (if (null shape)
            (vt-full nil (complex 1.0d0 0.0d0) :dtype :complex128)
            (let* ((n (first shape))
                   (diag (vt-diagonal c))
                   ;; 方差在对角上是实数（虚部只有浮点噪声），取模开方
                   (sd (vt-sqrt (vt-abs diag)))
                   (den (vt-outer sd sd))
                   (zero-d (vt-= den (vt-cast 0 (vt-dtype den))))
                   (safe (vt-where zero-d (vt-cast 1 (vt-dtype den)) den
                                   :dtype (vt-dtype den)))
                   (r0 (vt-/ c safe))
                   (ident (vt-eye n :dtype :complex128)))
              (vt-where zero-d ident r0 :dtype :complex128))))
      (funcall (%cx-orig 'vt-corrcoef) m :y y :rowvar rowvar)))

;;; ---- 12.5 fromiter：默认 dtype 推断含复数 -------------------------
;;; 原实现默认 :float64，且显式传 :float64 时复数会被静默取实部。
;;; numpy 的 fromiter 必须显式给 dtype，给实数 dtype 遇复数直接 TypeError。

(defun vt-fromiter (iterable &key (dtype nil dtype-p) (count nil))
  "从 Lisp 序列构建 1 维张量（对标 numpy.fromiter）。

   dtype 缺省时**推断**：序列含复数 → :complex128，否则保持既有的
   :float64（兼容旧调用）。显式传入实数 dtype 而序列含复数 → 报错，
   与 numpy 的 TypeError 一致（绝不静默丢虚部）。"
  (when (not (or (null iterable) (typep iterable 'sequence)))
    (error "vt-fromiter: 需要序列输入，收到 ~a" (type-of iterable)))
  (let ((seq (or iterable '())))
    (cond
      ((and dtype-p dtype)
       (when (and (not (vt-complex-dtype-p dtype))
                  (some (lambda (x) (or (complexp x) (%cx-deep-p x))) seq))
         (error "vt-fromiter: 序列含复数，与显式 dtype ~a 冲突\
（numpy 同样抛 TypeError: Cannot cast complex to real）" dtype))
       (funcall (%cx-orig 'vt-fromiter) seq :dtype dtype :count count))
      (t
       (funcall (%cx-orig 'vt-fromiter) seq
                :dtype (if (some (lambda (x) (or (complexp x) (%cx-deep-p x))) seq)
                           :complex128
                           :float64)
                :count count)))))

(defun vt-positive (tensor &key out dtype)
  "逐元素取正（一元 +，对标 numpy.positive）。复数恒等映射，保复数 dtype。"
  (let ((v (ensure-vt tensor)))
    (if (not (vt-complex-dtype-p (vt-dtype v)))
        (funcall (%cx-orig 'vt-positive) tensor :out out :dtype dtype)
        (vt-fast-map #'identity v :out out
                     :dtype (%cx-result-complex-dtype (vt-dtype v) dtype "VT-POSITIVE")))))

;;; ==================================================================
;;; 13. 明确拒绝：numpy 同样不做，但当前实现会**静默丢虚部**
;;; ==================================================================
;;; 这一节的函数在 numpy 2.4.6 上全部抛 TypeError（逐项实测见 §16.19）。
;;; 我们此前要么把复数当实数算（静默错答案），要么漏出裸 TYPE-ERROR
;;; （错误信息指向内部实现，用户看不懂）。统一改为确定性错误。
;;;
;;; 判据：只要**任一参数**含复数就拒绝 —— 与 %CX-DEF-UNSUPPORTED 一致。

;;; ---- 12.6 any / all：复数按「非零」判定真值 ----------------------
;;; numpy 对复数的布尔语义就是「任一分量非零即为真」（`bool(0+0j)` 为 False，
;;; `bool(nan+0j)` 为 True）。clvt 的 vt-any / vt-all 是 8 路实数元素类型
;;; 特化的归约，遇到复数直接抛 "unsupported input dtype"。
;;;
;;; 这里不重写归约内核 —— 先把复数转成 int8 的 0/1 真值掩码
;;; （vt-nonzero-p，已经是公共算子），再交给原实现。
;;; 语义与 numpy 完全一致，且自动继承 axis / keepdims / :out / 空输入单位元
;;; （any→0、all→1）的全部既有契约。

(defun %cx-bool-mask (v)
  "复数张量 → int8 的 0/1 真值掩码。NaN 视为真（numpy 一致）。"
  (vt-nonzero-p v))

