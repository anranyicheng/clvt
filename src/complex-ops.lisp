;;;; complex-ops.lisp — 复数「数据搬运层」：转换 / 拼接 / 剪切 / 集合 / 缺失值 / 缩并
;;;;
;;;; 定位：只放**复数分支的实现**。每个公开入口的实数实现仍在它的归属文件里，
;;;; 由该文件的单一 DEFUN 在入口处 `(when (%cx-any-p …) …)` 分流到这里的
;;;; `%cx-op-*`（CONVENTIONS §16.32/§16.33）。因此本文件不做任何运行期重定义。
;;;;
;;;; 设计约束（CONVENTIONS §16）：
;;;;
;;;;  1) **实数路径零影响**：不含复数的调用根本不进本文件，语义与性能
;;;;     由归属文件的原有实现保证。
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

(defun %cx-nan-to-num-buffer (flat nanv posv negv)
  "FLAT 为连续复数张量，返回**同 dtype** 的新张量：实部/虚部各自把
   NaN / ±Inf 换掉（对标 numpy 的逐分量替换）。

   为什么不用 `vt-map` + 闭包：闭包每元素一次 funcall + 装箱，1M complex128
   实测 78.9 ms；类型特化的直线循环 4.8 ms（对照 `vt-copy` 3.4 ms，即已接近
   纯拷贝）。nan_to_num 带三个额外参数，进不了 `vt-fast-map` 的一元内核白名单，
   只能单独写循环。"
  (macrolet ((pull (ctype rtype maxf)
               `(let* ((s (the (simple-array ,ctype (*)) (vt-data flat)))
                       (off (the fixnum (vt-offset flat)))
                       (n (the fixnum (vt-size flat)))
                       (nan (coerce nanv ',rtype))
                       (pos (coerce posv ',rtype))
                       (neg (coerce negv ',rtype))
                       (mx (coerce ,maxf ',rtype))
                       (res (make-array (the fixnum (vt-size flat)) :element-type ',ctype)))
                  (declare (type (simple-array ,ctype (*)) s res)
                           (type fixnum off n)
                           (type ,rtype nan pos neg mx)
                           (optimize (speed 3) (safety 1)))
                  (flet ((fix1 (v)
                           (declare (type ,rtype v))
                           (cond ((/= v v) nan)
                                 ((> v mx) pos)
                                 ((< v (- mx)) neg)
                                 (t v))))
                    (declare (inline fix1))
                    (with-float-safe
                      (dotimes (i n)
                        (let ((z (aref s (the fixnum (+ off i)))))
                          (setf (aref res i)
                                (complex (fix1 (%creal z)) (fix1 (%cimag z))))))))
                  (%make-vt :data res
                            :shape (vt-shape flat)
                            :strides (vt-compute-strides (vt-shape flat))
                            :offset 0
                            :dtype (vt-dtype flat)))))
    (let ((et (array-element-type (vt-data flat))))
      (cond ((equal et '(complex double-float))
             (pull (complex double-float) double-float most-positive-double-float))
            ((equal et '(complex single-float))
             (pull (complex single-float) single-float most-positive-single-float))
            (t nil)))))

(defun %cx-op-nan-to-num (tensor &key (copy t) (nan 0.0d0) (posinf nil) (neginf nil) out dtype)
  "把 NaN/±Inf 替换掉（对标 numpy.nan_to_num）。

   复数（§16.18）：**实部与虚部分别处理**，替换值 NAN/POSINF/NEGINF 是实数，
   同时作用于两个分量；POSINF/NEGINF 缺省为该分量精度的最大/最小有限值
   （complex128 → ±1.797e308，complex64 → ±3.4028e38）。
   结果 dtype 默认与输入相同（complex64 进 complex64 出）。

   例：(vt-nan-to-num #C(nan-inf i)) 的参数化版本见 test/complex-ops-test.lisp。"
  (declare (ignore copy))
  (let ((in (ensure-vt tensor)))
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
      (let ((res (%cx-nan-to-num-buffer (%cx-flat in) nanv posv negv)))
        (cond ((null out) res)
              ((eq res out) out)
              (t (vt-check-out out (vt-shape in) dt :op-name "VT-NAN-TO-NUM")
                 (vt-copy-into out res)
                 out))))))

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

(defun %cx-op-repeat (vt repeats &key axis)
  "重复元素（对标 numpy.repeat）。复数走复数路径，实数转发原实现。"
(%cx-repeat-complex vt repeats axis))


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
   返回 (values 唯一值数组 计数 fixnum 向量 唯一数)。
   DT 由调用方的 %CX-DISPATCH 决定，这里不用（保留形参以对齐签名）。"
  (declare (ignore dt))
  (let* ((copy (vt-copy flat))
         (data (vt-data copy))
         (n (vt-size flat)))
    (%cx-sort-fiber data 0 n)
    (%cx-dispatch data arr zero
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

(defun %cx-op-unique (in return-index return-inverse return-counts axis)
  "复数 vt-unique 的实现（IN 已 ENSURE-VT）。

   AXIS 给定时按轴去重（§16.23）。该路径的元素比较原语（%AXIS-ELEM-LESSP /
   %AXIS-ELEM-EQ）本身与元素类型无关，复数按 (实部, 虚部) 字典序参与比较，
   因此直接复用实数层的 %UNIQUE-AXIS，无需第二份实现。

   本函数只做复数分支；公开入口在 setops.lisp 的单一 DEFUN 里分流（§16.32）。
   若改成「事后重定义公开函数」，重载 setops.lisp 就会把复数静默送进实数内核。"
  (let ((rank (length (vt-shape in))))
    (if (and axis (> rank 1))
        (%unique-axis in (vt-normalize-axis axis rank)
                      return-index return-inverse return-counts)
        (%cx-unique in return-index return-inverse return-counts))))

(defun %cx-key-table (test)
  "把 TEST 的元素装进 EQL 哈希表（键为 %CX-KEY）。"
  (let ((h (make-hash-table :test #'eql))
        (flat (%cx-flat test)))
    (with-float-safe
      (%cx-dispatch (vt-data flat) arr zero
            (let ((off (vt-offset flat))
              (n (vt-size flat)))
          (dotimes (i n)
            (setf (gethash (%cx-key (aref arr (+ off i))) h) t)))))
    h))

(defun %cx-lookup-flags (element table invert)
  "逐元素查表 → **:bool** 张量（INVERT 取反）。形状与 ELEMENT 相同。
   复数与实数两条会员判定路径的返回值 dtype 必须一致（§17）。"
  (vt-map (lambda (z)
            (let ((hit (if (gethash (%cx-key z) table) 1 0)))
              (if invert (- 1 hit) hit)))
          (ensure-vt element) :dtype :bool))

(defun %cx-op-in1d (t1 t2)
  "逐元素判定是否属于 T2（对标 numpy.in1d）。复数按值相等（NaN 与 NaN 视为相等）。"
(%cx-lookup-flags (ensure-vt t1) (%cx-key-table t2) nil))


(defun %cx-op-isin (element test-elements &key (invert nil) (assume-unique nil))
  "逐元素判定是否属于 TEST-ELEMENTS（对标 numpy.isin）。结果形状同 ELEMENT。"
  (declare (ignore assume-unique))
(%cx-lookup-flags element (%cx-key-table test-elements) invert))


(defun %cx-set-filter (u table invert)
  "按 KEEP-TABLE 过滤已升序的复数唯一值 U（保持顺序）。"
  (let* ((n (vt-size u))
         (dt (vt-dtype u))
         (res (make-vt (list n) 0 :dtype dt))
         (k 0))
    (with-float-safe
      (%cx-dispatch (vt-data (vt-copy u)) src zero
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

(defun %cx-op-intersect1d (t1 t2)
  "两个 1D 复数张量的交集，结果升序去重（对标 numpy.intersect1d）。"
(%cx-set-filter (vt-unique (ensure-vt t1))
  (%cx-key-table (vt-unique (ensure-vt t2))) nil))


(defun %cx-op-setdiff1d (t1 t2)
  "差集 t1 \\ t2，结果升序去重（对标 numpy.setdiff1d）。"
(%cx-set-filter (vt-unique (ensure-vt t1))
  (%cx-key-table (vt-unique (ensure-vt t2))) t))


(defun %cx-op-union1d (t1 t2)
  "并集，结果升序去重（对标 numpy.union1d）。"
(vt-unique (vt-concatenate 0 (vt-unique (ensure-vt t1))
  (vt-unique (ensure-vt t2)))))


(defun %cx-op-setxor1d (t1 t2)
  "对称差，结果升序去重（对标 numpy.setxor1d）。"
(let* ((a (ensure-vt t1)) (b (ensure-vt t2))
       (u1 (vt-unique a)) (u2 (vt-unique b))
       (only1 (%cx-set-filter u1 (%cx-key-table u2) t))
       (only2 (%cx-set-filter u2 (%cx-key-table u1) t)))
  (vt-unique (vt-concatenate 0 only1 only2))))


;;; ==================================================================
;;; 6. searchsorted（字典序二分）
;;; ==================================================================

(defun %cx-searchsorted-run (tbl tvals right out)
  "在已升序的复数表 TBL 中对每个 TVALS 二分。RIGHT 为 T 时找第一个 > 的位置。"
  (%cx-dispatch tbl arr zero
    (%cx-dispatch tvals varr zero2
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

(defun %cx-op-searchsorted (tensor values &key (side :left))
  "在有序张量中查找插入点（对标 numpy.searchsorted）。复数按字典序比较。

   :left  → 第一个 >= v 的位置；:right → 第一个 > v 的位置。
   NaN 参与比较时视为「不可比」：落在表尾 NaN 区之前（与排序的 NaN 末尾一致）。"
  (let ((tv (ensure-vt tensor))
        (vv (ensure-vt values)))
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
              :offset 0 :dtype :int64))))


;;; ==================================================================
;;; 7. 缺失值归约（nan 族）
;;; ==================================================================
;;; numpy 语义（2.4.6 实测，见 §16.18 表）：
;;;   · nansum/nanprod 跳过 NaN，空归约区取单位元（0 / 1）；
;;;   · nanmax/nanmin 跳过 NaN，**全 NaN 切片 → NaN+NaNj**（不报错）；
;;;   · nanargmax/nanargmin 跳过 NaN，全 NaN 切片 → 报错；
;;;   · nanmean 保复数、nanvar/nanstd **恒实数**（|x-mean|² 的均值）；
;;;   · nanmedian 取字典序中位数（偶数个取中间两个的算术平均），
;;;     全 NaN → NaN+NaNj。
;;;
;;; 实现策略：用「掩码 + 单位元替换 + 既有复数归约内核」的向量化写法，
;;; 一次拿到所有 axis/keepdims 组合，且**复用已优化的内核**
;;; （vt-sum / vt-amax 的复数快路径），不再写第二套 fiber 循环。


(defun %cx-nan-sentinel (dt sign)
  "全 ±Inf 的复数哨兵：SIGN 为 +1 → (+inf,+inf)，-1 → (-inf,-inf)。
   它比任何有限值都「大/小」（字典序下实部就分出胜负），
   因此可以被 amax/amin 当作「已被排除的 NaN」占位。"
  (let* ((rdt (vt-complex-real-dtype dt))
         (v (if (eq rdt :float32)
                (if (plusp sign) +vt-sfloat-pos-inf+ +vt-sfloat-neg-inf+)
                (if (plusp sign) +vt-dfloat-pos-inf+ +vt-dfloat-neg-inf+))))
    (complex v v)))

;;; ==================================================================
;;; nan 族的**全归约**单趟内核
;;;
;;; 原先 nan 族的每个算子都是「掩码 + 单位元替换 + 既有归约内核」的张量
;;; 流水线：nanmean = isnan + where + sum + logical-not + astype + sum + div
;;; + where ≈ 7 趟；nanvar 再翻一倍到 10+ 趟。1M complex128 实测
;;; nanmean 20 / nanvar 50 / nanmax 23 ms，而同形 float64 走
;;; `%NAN-SUM-COUNT`/`%NAN-SUMSQ` 只要 1.4 ms —— 这个差价里没有一处是
;;; 「复数本来就该慢」。
;;;
;;; 这里补三个**单趟**扫描：求和+计数、字典序极值+计数、离差平方和。
;;; 只在 axis=nil（全归约）时启用，与实数侧的 `%NAN-FAST-P` 判据一致；
;;; 沿轴版本继续走流水线（正确性由差分测试盯着）。
;;;
;;; 为什么必须 with-float-safe 包住整个扫描：SBCL 的浮点比较用 UCOMISD，
;;; 遇到 NaN 会置 invalid 并**触发陷阱**，`(/= nan nan)` 直接抛
;;; FLOATING-POINT-INVALID-OPERATION —— 与 C 的 quiet compare 不同。
;;; 切换 FP 控制字只在进出各一次，不在逐元素路径上。
;;; ==================================================================

(defun %cx-nan-fast-p (axis)
  "全归约快路径判据（对应实数侧的 %NAN-FAST-P）。"
  (null axis))

(defun %cx-nan-sum-count (flat)
  "非 NaN 元素的复数和 + 计数（单趟，顺序累加）。"
  (macrolet ((pull (ctype rtype)
               `(let ((d (the (simple-array ,ctype (*)) (vt-data flat)))
                      (off (the fixnum (vt-offset flat)))
                      (n (the fixnum (vt-size flat))))
                  (declare (type (simple-array ,ctype (*)) d)
                           (type fixnum off n)
                           (optimize (speed 3) (safety 1)))
                  (let ((acc (complex (coerce 0 ',rtype) (coerce 0 ',rtype)))
                        (c 0))
                    (declare (type ,ctype acc) (type fixnum c) (type ,rtype))
                    (with-float-safe
                      (dotimes (i n)
                        (let ((z (aref d (the fixnum (+ off i)))))
                          (unless (%cx-nan-p z)
                            (incf acc z)
                            (incf c)))))
                    (values acc c)))))
    (let ((et (array-element-type (vt-data flat))))
      (cond ((equal et '(complex double-float)) (pull (complex double-float) double-float))
            ((equal et '(complex single-float)) (pull (complex single-float) single-float))
            (t (error "%cx-nan-sum-count: 非复数缓冲区 ~a" et))))))

(defun %cx-nan-extremum-count (flat maxp)
  "字典序极值（MAXP 真取最大）+ 计数（单趟）。COUNT=0 时返回值无意义，
   由调用方按「全 NaN」覆盖成 NaN+NaNj。比较沿用 %CX-SORT-LESS 的全序
   （`%CLT`，NaN 已在扫描里跳过）。"
  (macrolet ((pull (ctype rtype)
               `(let ((d (the (simple-array ,ctype (*)) (vt-data flat)))
                      (off (the fixnum (vt-offset flat)))
                      (n (the fixnum (vt-size flat))))
                  (declare (type (simple-array ,ctype (*)) d)
                           (type fixnum off n)
                           (optimize (speed 3) (safety 1)))
                  ;; BEST 用 FIRST 标志初始化，**不能**直接拿第一个元素当种子：
                  ;; 若 a[0] 是 NaN，`%CLT` 的比较恒假，BEST 会永远停在 NaN 上
                  ;; 而后面所有有限值都比不过它。
                  (let ((best (if (plusp n) (aref d off)
                                  (complex (coerce 0 ',rtype) (coerce 0 ',rtype))))
                        (c 0) (first t))
                    (declare (type ,ctype best) (type fixnum c) (type ,rtype))
                    (with-float-safe
                      (dotimes (i n)
                        (let ((z (aref d (the fixnum (+ off i)))))
                          (unless (%cx-nan-p z)
                            (incf c)
                            (when (or first
                                      (if maxp (%clt best z) (%clt z best)))
                              (setf best z first nil))))))
                    (values best c)))))
    (let ((et (array-element-type (vt-data flat))))
      (cond ((equal et '(complex double-float)) (pull (complex double-float) double-float))
            ((equal et '(complex single-float)) (pull (complex single-float) single-float))
            (t (error "%cx-nan-extremum-count: 非复数缓冲区 ~a" et))))))

;; SBCL 2.6.9 起把 `(* x x)` 的返回类型推断成「非负浮点」（`(FLOAT (0.0))`），
;; 而 NaN 不属于该类型（NaN 与 0 的序关系全为假）。本函数恰恰**必须**能返回
;; NaN：`mean` 为 NaN（输入 ±Inf 混合）或 `0 × +Inf` 时累加出 NaN。若放任推断，
;; 调用点会按推断出的函数类型插一条返回值检查，NaN 当场 TYPE-ERROR。
;; 显式 ftype 把记录在 fdefn 上的类型钉回 `float`（NaN 是 float），检查即消失，
;; 且不影响循环内的数值推导（不产生运行时开销）。
(declaim (ftype (function (t t t) (values float)) %cx-nan-sumsq))

(defun %cx-nan-sumsq (flat mr mi)
  "Σ|z-mean|²，跳过 NaN（单趟）。NaN 位的贡献按原流水线的 `(* sq not-nan)`
   复刻为 `0 × |0-mean|²` —— MEAN 为 ±Inf 时该位给 NaN，与旧实现逐位相同。"
  (macrolet ((pull (ctype rtype)
               `(let ((d (the (simple-array ,ctype (*)) (vt-data flat)))
                      (off (the fixnum (vt-offset flat)))
                      (n (the fixnum (vt-size flat)))
                      (m0 (coerce mr ',rtype))
                      (m1 (coerce mi ',rtype)))
                  (declare (type (simple-array ,ctype (*)) d)
                           (type fixnum off n)
                           (type ,rtype m0 m1)
                           (optimize (speed 3) (safety 1)))
                  (let ((a (coerce 0 ',rtype)))
                    (declare (type ,rtype a))
                    (with-float-safe
                      (dotimes (i n)
                        (let ((z (aref d (the fixnum (+ off i)))))
                          (if (%cx-nan-p z)
                              (incf a (* (coerce 0 ',rtype) (+ (* m0 m0) (* m1 m1))))
                              (let ((dr (- (%creal z) m0)) (dm (- (%cimag z) m1)))
                                (declare (type ,rtype dr dm))
                                (incf a (+ (* dr dr) (* dm dm))))))))
                    a))))
    (let ((et (array-element-type (vt-data flat))))
      (cond ((equal et '(complex double-float)) (pull (complex double-float) double-float))
            ((equal et '(complex single-float)) (pull (complex single-float) single-float))
            (t (error "%cx-nan-sumsq: 非复数缓冲区 ~a" et))))))

(defun %cx-nan-scalar-result (name shape dt val out)
  "把标量 VAL（已是对应 Lisp 类型）包成 SHAPE 形状的结果张量（含 :out 契约）。"
  (let ((res (if out (vt-check-out out shape dt :op-name name)
                 (make-vt shape 0 :dtype dt))))
    (setf (aref (vt-data res) (vt-offset res)) val)
    res))

(defun %cx-nan-reduce-global (name in dt keepdims out op)
  "全归约（axis=nil）的单趟实现。OP ∈ :sum :mean :max :min。
   DT 由调用方保证是复数 dtype。"
  (let* ((flat (%cx-flat in))
         (off (the fixnum (vt-offset flat)))
         (n (the fixnum (vt-size flat)))
         (rt (vt-dtype->lisp-type (vt-complex-real-dtype dt)))
         (shape (if keepdims (make-list (length (vt-shape in)) :initial-element 1) nil))
         (nan (%cx-complex-nan-of dt)))
    (declare (ignore off n))
    (ecase op
      (:sum
       (multiple-value-bind (s c) (%cx-nan-sum-count flat)
         (declare (ignore c))
         (%cx-nan-scalar-result name shape dt s out)))
      (:mean
       (multiple-value-bind (s c) (%cx-nan-sum-count flat)
         (%cx-nan-scalar-result name shape dt
                                (if (zerop c) nan (/ s (coerce c rt)))
                                out)))
      ((:max :min)
       (multiple-value-bind (v c) (%cx-nan-extremum-count flat (eq op :max))
         (%cx-nan-scalar-result name shape dt (if (zerop c) nan v) out))))))

(defun %cx-nan-reduce (name tensor axis keepdims dtype out op)
  "nan 族归约的公共外壳。OP ∈ :sum :prod :max :min :mean。

   全归约（axis=nil）且 OP ≠ :prod 时走**单趟**扫描（%CX-NAN-REDUCE-GLOBAL）；
   其余情形仍是「掩码 + 单位元替换 + 既有归约内核」的流水线。
   :prod 不并入单趟 —— 它的单位元是 1 而不是 0，且实测不是热点。"
  (let* ((in0 (ensure-vt tensor))
         (dt0 (or dtype (vt-dtype in0))))
    (unless (vt-complex-dtype-p dt0)
      (error "~a: 复数输入的 :dtype 必须是复数类型，收到 ~a" name dt0))
    (when (and (%cx-nan-fast-p axis) (not (eq op :prod)))
      (return-from %cx-nan-reduce
        (%cx-nan-reduce-global name in0 dt0 keepdims out op))))
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
        ;; max / min：全 NaN 切片用 count==0 检测 → **NaN+NaNj**（numpy 同）
        ((member op '(:max :min))
         (let* ((cnt (vt-sum (vt-astype (vt-logical-not mask) rdt)
                             :axis axis :keepdims keepdims :dtype rdt))
                (empty (vt-= cnt (vt-cast 0 rdt)))
                (nanv (%cx-complex-nan-of dt)))
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
(defun %cx-op-nanmean (tensor &key axis keepdims dtype out)
  "忽略 NaN 的算术平均（对标 numpy.nanmean）。

   复数（§16.18）：结果为**复数**（complex64 → complex64）；
   全 NaN 切片 → NaN+NaNj（与 numpy 一致）。"
(%cx-nan-reduce "VT-NANMEAN" tensor axis keepdims dtype out :mean))


;;; ---- nanvar / nanstd（恒实数）--------------------------------------
(defun %cx-nan-var-global (name in rdt ddof out std-p keepdims)
  "全归约（axis=nil）的两趟实现：sum+count → Σ|z-mean|²。
   语义与流水线版逐条对齐：count=0 → NaN；count ≤ ddof → NaN。"
  (let* ((flat (%cx-flat in))
         (n (the fixnum (vt-size flat)))
         (rt (vt-dtype->lisp-type rdt))
         (shape (if keepdims (make-list (length (vt-shape in)) :initial-element 1) nil))
         (nan (coerce +vt-dfloat-nan+ rt)))
    (declare (ignore n))
    (multiple-value-bind (s c) (%cx-nan-sum-count flat)
      (let* ((cnt (coerce c rt))
             (dd (coerce ddof rt))
             (div (- cnt dd))
             (var (if (or (zerop c) (<= div (coerce 0 rt)))
                      nan
                      (let ((mean (/ s cnt)))
                        (/ (%cx-nan-sumsq flat (%creal mean) (%cimag mean)) div)))))
        (%cx-nan-scalar-result name shape rdt (if std-p (sqrt var) var) out)))))

(defun %cx-nan-var (name tensor axis keepdims ddof dtype out std-p)
  "复数 nanvar/nanstd：mean(|x-mean|²) 跳过 NaN，结果恒实数（§16.7 同理）。

   全归约（axis=nil）走两趟单趟扫描；沿轴版本仍是流水线。"
  (let* ((in0 (ensure-vt tensor))
         (rdt0 (vt-complex-real-dtype (vt-dtype in0))))
    (when (%cx-nan-fast-p axis)
      (return-from %cx-nan-var
        (%cx-nan-var-global name in0 rdt0 ddof out std-p keepdims))))
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

(defun %cx-op-nanvar (tensor &key axis keepdims (ddof 0) dtype out)
  "忽略 NaN 的方差（对标 numpy.nanvar）。复数结果为**实数**（|x-mean|² 的均值）。"
(%cx-nan-var "VT-NANVAR" tensor axis keepdims ddof dtype out nil))


(defun %cx-op-nanstd (tensor &key axis keepdims (ddof 0) dtype out)
  "忽略 NaN 的标准差（对标 numpy.nanstd）。复数结果为**实数**。"
(%cx-nan-var "VT-NANSTD" tensor axis keepdims ddof dtype out t))


;;; ---- nanargmax / nanargmin ----------------------------------------
(defun %cx-nan-arg-run (data shape ax op)
  "跳过 NaN 的 argmax/argmin 内核。返回 fixnum 向量。
   全 NaN 切片 → 报错（对标 numpy 的 ValueError: All-NaN slice encountered）。"
  (%cx-dispatch data d zero
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
(defun %cx-nan-median-fiber (data base stride n dt &optional tmp)
  "一条 fiber 的中位数（跳过 NaN，字典序）。全 NaN / 空 fiber → NaN+NaNj
   （numpy 的 nanmedian 在全 NaN 上也给 nan+nanj）。

   TMP 是可选的复用缓冲区（长度 ≥ N）；不给就新分配。调用方在逐 fiber 循环里
   应当传入同一个 TMP —— 否则每条 fiber 都要分配/清零一次。

   v0.4.18：收数改走 %CX-COMPACT-FIBER（类型特化），取中位改走 %CX-SELECT-IN
   （O(N) 而非 O(N log N)）。1M complex128 实测 216 → 34 ms。"
  (let ((tmp (or tmp (make-array (max n 1) :element-type (array-element-type data))))
        (cnt 0))
    (declare (type fixnum cnt))
    (setf cnt (%cx-compact-fiber tmp data base stride n))
    (cond
      ;; 全 NaN / 空 fiber → NaN+NaNj（第 26 轮修正）。此前返回 **NaN+0j**，
      ;; 而 numpy 2.4.6 的 nanmedian / nanmax / nanmin 在全 NaN 上都是
      ;; `nan+nanj` —— 虚部那个 0 是「先取 nan，再补 0」的副产品，不是语义。
      ((zerop cnt) (%cx-complex-nan-of dt))
      (t
       ;; 第 26 轮：从「整段排序再取中位」改成 quickselect（O(cnt) vs O(cnt log cnt)）。
       ;; 同一缓冲区连调两次是安全的：%CX-SELECT-IN 只重排，元素集合不变，
       ;; 第 K 小的**值**与排列无关。1M complex128 216 ms → 见 §16.14.6。
       (if (oddp cnt)
           (%cx-select-in tmp cnt (floor cnt 2))
           (let ((a (%cx-select-in tmp cnt (1- (floor cnt 2))))
                 (b (%cx-select-in tmp cnt (floor cnt 2))))
             (* 0.5s0 (+ a b))))))))

(defun %cx-nanmedian-multi (in axes keepdims out dtype)
  "复数：多轴 nanmedian —— 折叠成 (H, D) 后复用单轴内核，再重塑回目标形状。"
  (with-float-safe
    (multiple-value-bind (two target) (%fold-reduce-axes in axes keepdims)
      (let* ((dt (or dtype (vt-dtype in)))
             (res (vt-reshape (%cx-op-nanmedian two :axis 1 :dtype dt) target)))
        (cond ((null out) res)
              ((eq res out) out)
              (t (vt-check-out out target dt :op-name "VT-NANMEDIAN")
                 (vt-copy-into out res)
                 out))))))

(defun %cx-op-nanmedian (tensor &key axis keepdims out dtype)
  "忽略 NaN 的中位数（复数）：按字典序；偶数个取中间两值均值。
   全 NaN / 空 fiber → NaN+NaNj（numpy 的 nanmedian 同样给 nan+nanj）。
   本函数只做复数分支；公开入口在 reduce-stats.lisp 的单一 DEFUN 里分流（§16.32）。
   多轴（如 '(0 1)）由入口折叠成 2-D 后复用本函数的单轴分支（§16.35.2）。"
  (let ((in (ensure-vt tensor)))
    (when (and axis (listp axis) (cdr axis))
      (return-from %cx-op-nanmedian
        (%cx-nanmedian-multi in (vt-normalize-axes axis (length (vt-shape in)))
                             keepdims out dtype)))
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
                   ;; 逐 fiber 复用一个缓冲区：每条 fiber 都新分配/清零一次
                   ;; 会让「多条短 fiber」的用例被分配器主导（§16.14.6）。
                   (tmp (make-array (max dim 1) :element-type (array-element-type data)))
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
                                                 ax-stride dim dt tmp))
                     (incf k))))
               (%cx-write-flat res vals dt)
               res)))
          ;; 多轴已在函数入口分流（见上），这里只剩单轴/展平两种
          (t (error "vt-nanmedian: 内部错误 —— 多轴未在入口分流")))))))

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

(defun %cx-correlate (a v mode-kw &key (conj-p t) (same-anchor :v))
  "复数 1D 互相关（三 mode）。CONJ-P 见 %CX-CORRELATE-FULL。

   SAME-ANCHOR：`same` 的裁切起点第二项用 n/2（`:v`，correlate）还是
   (n-1)/2（`:a`，convolve）。numpy 两个入口的偏移公式不同，见
   `VT-CORRELATE` 的文档与 CONVENTIONS §17.10.15。"
  (let* ((av (ensure-vt a))
         (bv (ensure-vt v))
         (dt (vt-promote-type (vt-dtype av) (vt-dtype bv)))
         (dt (if (vt-complex-dtype-p dt) dt :complex128))
         (af (%cx-flat (vt-copy (vt-astype av dt))))
         (vf (%cx-flat (vt-copy (vt-astype bv dt))))
         (n (vt-size af))
         (m (vt-size vf))
         (full (%cx-correlate-full af vf conj-p))
         (full-len (length full)))
    (declare (type fixnum n m full-len))
    (flet ((slice (len start)
             (%make-vt :data (subseq full start (min full-len (+ start len)))
                       :shape (list (max 0 len)) :strides '(1)
                       :offset 0 :dtype dt)))
      (ecase mode-kw
        (:full (%make-vt :data full :shape (list full-len) :strides '(1)
                         :offset 0 :dtype dt))
        ;; VALID：长 |n-m|+1、起点 min(n,m)-1（原为 (max 0 (1+ (- n m))) +
        ;; offset，n<m 时给空张量、n>=m 时起点恰好等同）。
        (:valid (slice (1+ (abs (- n m))) (1- (min n m))))
        ;; SAME：起点 min((m-1)/2, n/2) 或 min((m-1)/2, (n-1)/2)，
        ;; 原为 (floor (- full-len (max n m)) 2)，|n-m| 为奇数时方向相反。
        (:same (slice (max n m)
                      (min (floor (1- m) 2)
                           (floor (if (eq same-anchor :v) n (1- n)) 2))))))))

(defun %cx-op-correlate (a v &key (mode :full) (same-anchor :v))
  "1D 互相关（对标 numpy.correlate）。复数走专用内核（累加器为复数）。

   复数语义（与 numpy 一致）：c[k] = Σ a[n+k]·conj(v[n])，即**第二个操作数取共轭**。
   注意默认值是 :full（与 numpy 的 :valid 不同，见原实现文档）。
   SAME-ANCHOR 见 `%CX-CORRELATE`（内部参数，convolve 传 :a）。"
(let ((mk (if (null mode) :full (%normalize-mode-keyword mode))))
  (%cx-correlate a v mk :conj-p t :same-anchor same-anchor)))


(defun %cx-op-convolve (a v &key (mode :full))
  "1D 卷积（对标 numpy.convolve）。复数：c[k] = Σ a[n]·v[k-n]，**不取共轭**。

   实数路径转发原实现；复数复用互相关内核（对 flip(v) 不共轭地相关），
   与 numpy 的 convolve/correlate 分工一致。SAME 的起点要按 convolve 的
   (n-1)/2 算（`same-anchor :a`），否则偶数 n + 长核时差一个位置。"
(let ((mk (if (null mode) :full (%normalize-mode-keyword mode))))
  (%cx-correlate a (vt-flip (ensure-vt v)) mk :conj-p nil :same-anchor :a)))


;;; ==================================================================
;;; 9. 张量缩并（tensordot / inner）
;;; ==================================================================
;;; 实数路径靠 vt-einsum 实现，而 einsum 的 dtype 白名单只含 4 种实数
;;; 类型（§16.11 的算子清单）。复数这里绕开 einsum：
;;;   transpose（把收缩轴挪到 a 末尾 / b 开头）→ reshape 成 2D →
;;;   **复数专用 matmul 内核** → reshape 回自由轴形状。
;;; 数学上与 einsum 等价，且完全复用已优化的矩阵乘。

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

(defun %cx-op-tensordot (a b &key (axes 2))
  "张量缩并（对标 numpy.tensordot）。复数走 transpose+reshape+matmul 专用路径。"
(%cx-tensordot a b axes nil nil))


(defun %cx-op-inner (a b &key dtype out)
  "内积（对标 numpy.inner）。复数**不取共轭**（numpy 亦如此；共轭点积见 vt-vdot）。"
(let* ((av (ensure-vt a))
       (bv (ensure-vt b))
       (ar (length (vt-shape av)))
       (br (length (vt-shape bv))))
  (cond
    ;; 0 维：numpy.inner 退化为广播乘法
    ((or (zerop ar) (zerop br))
     (%cx-to-out (vt-* av bv :dtype dtype) out "VT-INNER"))
    (t
     (%cx-tensordot av bv (list (list (1- ar)) (list (1- br))) dtype out)))))


;;; ==================================================================
;;; 10. 缺失值累积 与 共轭点积
;;; ==================================================================

(defun %cx-nan-fill (in fill)
  "把 NaN 元素替换成 FILL（复数标量），其余保持。用于 nancumsum/nancumprod。"
  (let ((dt (vt-dtype in)))
    (with-float-safe
      (vt-where (vt-isnan in) (vt-cast fill dt) in :dtype dt))))

(defun %cx-op-nancumsum (tensor &key axis dtype out)
  "沿 AXIS 累积和，NaN 视为 0（对标 numpy.nancumsum）。

   复数：原实现走 `(vt-astype input :float64)`（取实部），会**静默丢掉虚部**。
   这里改为「NaN→0+0i 后调用复数 cumsum」，语义与 numpy 完全相同，
   且复用已优化的扫描内核。"
(vt-cumsum (%cx-nan-fill tensor 0) :axis axis :dtype dtype :out out))


(defun %cx-op-nancumprod (tensor &key axis dtype out)
  "沿 AXIS 累积积，NaN 视为 1（对标 numpy.nancumprod）。复数同 vt-nancumsum。"
(vt-cumprod (%cx-nan-fill tensor 1) :axis axis :dtype dtype :out out))


(defun %cx-op-vdot (a b &key dtype out)
  "向量点积（对标 numpy.vdot）。复数：**第一个操作数取共轭**
   （vdot(a,b) = Σ conj(a)·b，与 numpy 一致；vt-dot 不取共轭）。"
(vt-dot (vt-conj (ensure-vt a)) (ensure-vt b) :dtype dtype :out out))


;;; ------------------------------------------------------------------
;;; 11. 明确拒绝（与 numpy 一致，但给出可读错误）
;;; ------------------------------------------------------------------

;;; vt-digitize 的复数拒绝在归属 DEFUN 的守卫里（§16.32），本文件不再有对应符号。


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

;;; ---- 12.1 值级内核（**已前移**）------------------------------------
;;; %COP-EXPM1 / %COP-LOG1P / %COP-SINC / %COP-SIGN 四个内核原先定义在这里，
;;; 但调用点（extensions3 的 vt-expm1/vt-log1p/vt-sign、elementwise2 的
;;; vt-sinc）在加载顺序上**早于**本文件 —— 编译调用点时符号还不存在，
;;; `(function %cop-xxx)` 只能编成真实函数调用，内核本体也被独立编译成
;;; 「泛型 NUMBER」版本（1M complex128：vt-sign 104 ms，内联后 17 ms）。
;;; 第 22 轮按 §16.14.2 第 2 条把它们移到 src/complex-kernels.lisp（纯标量
;;; 内核，无状态、无张量依赖），本文件不再保留任何定义。
;;; 为什么不是"在本文件里 declaim inline"：inline 声明只在**编译调用点之前**
;;; 可见才生效，而调用点已经编译过了 —— 声明永远迟到。

;;; ---- 12.2 clip / clamp：字典序夹取（numpy 语义）-------------------
;;; numpy 的 clip 实现是 `np.minimum(np.maximum(a, lo), hi)`，
;;; 而 maximum/minimum 对复数走字典序（§16.8）—— 所以复数的 clip
;;; **不是**按分量裁剪，而是整元素按字典序夹取。实测：
;;;   v=[1+2j, 3-4j, 5+0j], clip(0, 2) → [1+2j, 2+0j, 2+0j]
;;; 第一项虚部 2 超出 2 却没被裁，正是因为比较的是 (1,2) 整体而不是 2。

(defun %cx-op-clip (v min-val max-val min-p max-p out dtype)
  "复数 vt-clip 的实现（V 已 ENSURE-VT）。按**字典序**整元素夹取，不按分量。

   `(vt-clip a min)` 只给下限 → 报错（numpy 语义：clip() missing a_max）。
   本函数只做复数分支；公开入口在 elementwise.lisp 的单一 DEFUN 里分流（§16.32）。
   若改成事后重定义，重载 elementwise.lisp 后复数会被**逐分量**裁剪 —— 静默错答案。"
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
    (%cx-to-out r out "VT-CLIP")))

;;; ---- 12.3 standardize：按复数均值居中、按实标准差缩放 --------------
;;; 复数的 std 恒为实数（§16.7），所以 (x-mean)/std 是「复数 / 实数」，
;;; 结果仍是复数。原实现把 final-dtype 定成 float64，导致 vt-mean 收到
;;; 复数输入 + 实数 dtype 而报错（我们第一版就撞上了这个）。

(defun %cx-op-standardize (vt &key axis (ddof 0) dtype out)
  "Z-score 标准化 (x - mean) / std（对标 sklearn.preprocessing.scale）。
   复数：均值是复数、标准差是实数（|x-mean|² 的均值开方），结果保复数。"
  (let ((v (ensure-vt vt)))
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
    (%cx-to-out r out "VT-STANDARDIZE"))))


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

(defun %cx-op-cov (m &key (y nil y-p) (rowvar t) (ddof nil ddof-p) (bias nil))
  "协方差矩阵（对标 numpy.cov）。复数得到**厄米特**矩阵（X·X^H）。"
  (declare (ignore y-p))
  (let ((ddof (if ddof-p ddof 1)))
    (%cx-cov m y rowvar ddof bias)))


(defun %cx-op-corrcoef (m &key (y nil y-p) (rowvar t))
  "相关系数矩阵（对标 numpy.corrcoef）。复数结果仍是厄米特矩阵，
   对角为 1.0+0.0j（实数 1）。"
  (declare (ignore y-p))
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
        (vt-where zero-d ident r0 :dtype :complex128)))))


(defun %cx-op-positive (tensor &key out dtype)
  "逐元素取正（一元 +，对标 numpy.positive）。复数恒等映射，保复数 dtype。"
  (let ((v (ensure-vt tensor)))
    (vt-fast-map #'identity v :out out
    :dtype (%cx-result-complex-dtype (vt-dtype v) dtype "VT-POSITIVE"))))


;;; ==================================================================
;;; 13. 明确拒绝：numpy 同样不做，但当前实现会**静默丢虚部**
;;; ==================================================================
;;; 这一节的函数在 numpy 2.4.6 上全部抛 TypeError（逐项实测见 §16.19）。
;;; 我们此前要么把复数当实数算（静默错答案），要么漏出裸 TYPE-ERROR
;;; （错误信息指向内部实现，用户看不懂）。统一改为确定性错误。
;;;
;;; 判据：只要**任一参数**含复数就拒绝 —— 与 %CX-DEF-UNSUPPORTED 一致。

;;; ---- 12.5 any / all：复数按「非零」判定真值 ----------------------
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

