;;;; setops.lisp — 集合操作

(in-package :clvt)

;;; ==================================================================
;;; 会员判定（in1d / isin / intersect1d / setdiff1d / setxor1d 共用）
;;;
;;; v0.4.2 性能修复：原实现把备查集合装成 **list**，再对每个待判定元素做
;;; `(member x set :test #'vt-float-nan-inf-=)` —— 即 O(n×m) 次泛型浮点
;;; 比较。1M 元素 × 1k 集合 = 1e9 次调用，实测 **87 秒**
;;; （numpy 同类操作约 20 ms，差 4000 倍）。
;;; 改为「一次建哈希表 + O(1) 查表」：O(n+m)。
;;;
;;; 键的规范化保证语义不变（仍是 vt-float-nan-inf-= 的等价关系）：
;;;   · NaN  → :%clvt-nan（NaN 与自身相等，故统一到哨兵键）
;;;   · ±0.0 → 0.0d0（= 认为 -0.0 与 0.0 相等）
;;;   · 浮点 → double-float（1.0s0 与 1.0d0 数值相等）
;;;   · 整数 → 两数组皆整数时保留整数（避免 >2^53 精度丢失）；
;;;            任一侧为浮点时统一转 double-float（与 numpy 把混合类型
;;;            提升到 float64 的行为一致）
;;; ==================================================================

(defun %membership-float-mode-p (a b)
  "两个备查数组中任一侧为浮点 → 用「全 double-float」键模式。"
  (flet ((f (v) (subtypep (array-element-type (vt-data v)) 'float)))
    (or (f a) (f b))))

(defun %membership-key (x float-mode)
  "会员判定的哈希键。**调用方必须已屏蔽浮点陷阱**（`(not (= x x))` 在
   NaN 上会触发 FLOATING-POINT-INVALID-OPERATION）。
   v0.4.3：去掉内层 with-float-safe —— 它等价于每个元素各改一次 FP 控制字，
   在 1M 元素上本身就是可观的固定开销；所有调用点（vt-map 外壳 /
   %membership-table / 本文件的循环）都已在屏蔽区内。"
  (cond ((floatp x)
         (cond ((not (= x x)) :%clvt-nan)
               ((zerop x) 0.0d0)
               (t (coerce x 'double-float))))
        (float-mode (coerce x 'double-float))
        (t x)))

(defun %membership-table (test float-mode)
  "把 TEST 的元素装进 EQL 哈希表，供 O(1) 会员判定。"
  (let ((h (make-hash-table :test #'eql))
        (flat (vt-contiguous (vt-flatten test)))
        (n (vt-size test)))
    (with-float-safe
      (dotimes (i n)
        (setf (gethash (%membership-key (aref (vt-data flat) i) float-mode) h) t)))
    h))

(defun %membership-map (elem table float-mode invert)
  "对 ELEM 逐元素查表，返回 **:bool** 张量（INVERT 时取反）。
   与 numpy 一致：`in1d` / `isin` 返回 bool_ 数组，不是整数掩码（§17）。"
  (vt-map (lambda (x)
            (let ((hit (if (gethash (%membership-key x float-mode) table) 1 0)))
              (if invert (- 1 hit) hit)))
          elem :dtype :bool))

;;; ==================================================================
;;; 唯一值（vt-unique）
;;;
;;; v0.4.3 性能修复：原实现对「下标向量」做 stable-sort，比较器是闭包，
;;; 里面两次泛型 aref + 两次 %nan-p。1M 元素 ≈ 2e7 次闭包调用，
;;; 实测 497ms（numpy.unique 20ms，差 24 倍），并连带把 intersect1d /
;;; setdiff1d / setxor1d / union1d 全部拖到 500–700ms。
;;;
;;; 改法：直接对**元素值本身**排序（拷贝到专用数组后交给 reduce-stats 的
;;; %sort-fibers：稳定归并/插入，NaN 稳定压到末尾），再做一次 run-length
;;; 得到唯一值 + 计数。只有 return-index / return-inverse 需要「原始位置」
;;; 时才额外走一遍哈希（O(n)），默认调用完全不用哈希。
;;;
;;; 稳定性保证与旧实现一致：稳定归并使同值元素保持原相对顺序，
;;; 故每组首个元素的原下标 = 该值的首次出现位置（numpy 语义）。
;;; 元素类型不在 %sort-fibers 支持集内（int8/int16/uint8/uint16）时
;;; 回退到旧实现，语义完全相同。
;;; ==================================================================

;;; ------------------------------------------------------------------
;;; 按轴去重（:axis，numpy 2.x 语义）
;;;
;;; numpy 的算法（lib/_arraysetops_impl.py，unique(axis=...)）：
;;;   1. moveaxis(axis, 0) 后 reshape 成 (n, m) —— 沿 AXIS 的每个切片是一行，
;;;      行内元素按「其余轴的 C 序」展开；
;;;   2. 用**稳定**排序（return_index 时 numpy 用 mergesort）按逐元素字典序
;;;      排行，NaN 排末尾；
;;;   3. 相邻行做 aux[i] != aux[i-1] 的全元素比较，全等才算重复。
;;;      → NaN != NaN，所以**含 NaN 的行永不合并**（每个 NaN 行独立保留）。
;;;        这与展平模式（NaN 折叠为一个）是 numpy 内部的另一条分支，
;;;        本实现如实分层：axis 模式不折叠 NaN（§16.23）；
;;;   4. return_inverse 是长度 n 的一维数组（axis 模式下不 reshape），
;;;      return_index 是沿 AXIS 的原始下标，return_counts 长度 = 唯一行数。
;;;
;;; 实现要点：先把 AXIS 搬到首轴并落地为 C 连续副本，这样「行」在内存里
;;; 连续（行 i = [i*m, (i+1)*m)），比较器无需偏移表；输出再用逆置换转回。
;;; 与展平路径完全独立 —— 展平路径的 NaN 折叠语义不能复用。
;;; ------------------------------------------------------------------

(declaim (inline %axis-elem-lessp %axis-elem-eq))

(defun %axis-elem-lessp (x y)
  "行内元素的字典序比较：NaN 视为最大，且两 NaN 之间返回 NIL（即「不更小」），
   稳定排序下保持输入顺序 —— 与 numpy 对 NaN 的处理一致。
   复数按 (实部, 虚部) 字典序（numpy 2.x 复数排序约定，§16.8）。"
  (cond ((%cx-nan-p x) nil)
        ((%cx-nan-p y) t)
        ((complexp x)
         (let ((xr (realpart x)) (yr (realpart y)))
           (cond ((< xr yr) t)
                 ((> xr yr) nil)
                 (t (< (imagpart x) (imagpart y))))))
        (t (< x y))))

(defun %axis-elem-eq (x y)
  "行内元素的**严格**数值相等：NaN 与自身不等 → 含 NaN 的行不会被合并。
   调用方必须已屏蔽浮点陷阱（NaN 参与 `=` 会抛 FLOATING-POINT-INVALID-OPERATION）。"
  (if (complexp x)
      (and (= (realpart x) (realpart y))
           (= (imagpart x) (imagpart y)))
      (= x y)))

(defun %axis-row-lessp (data i j m)
  "行 I 是否字典序小于行 J（两行各 M 个元素，行在 DATA 中连续）。"
  (declare (type (simple-array * (*)) data) (type fixnum i j m))
  (let ((bi (the fixnum (* i m)))
        (bj (the fixnum (* j m))))
    (declare (type fixnum bi bj))
    (dotimes (k m nil)
      (let ((x (aref data (the fixnum (+ bi k))))
            (y (aref data (the fixnum (+ bj k)))))
        (cond ((%axis-elem-eq x y))            ; 该位相同 → 比下一位
              ((%axis-elem-lessp x y) (return t))
              (t (return nil)))))))

(defun %axis-row-eq-p (data i j m)
  "两行是否逐元素严格相等（全部相等才算重复行）。"
  (declare (type (simple-array * (*)) data) (type fixnum i j m))
  (let ((bi (the fixnum (* i m)))
        (bj (the fixnum (* j m))))
    (declare (type fixnum bi bj))
    (dotimes (k m t)
      (unless (%axis-elem-eq (aref data (the fixnum (+ bi k)))
                             (aref data (the fixnum (+ bj k))))
        (return nil)))))

(defun %unique-axis (tensor ax return-index return-inverse return-counts)
  "沿 AX AXIS 拆切片去重（numpy 2.x 语义）。实数/整数/复数通用。
   TENSOR 已由调用方 ENSURE-VT；AX 已归一化为正轴且 RANK ≥ 2。"
  (with-float-safe
    (let* ((shape (vt-shape tensor))
           (rank (length shape))
           ;; 轴置换：AX 搬到首轴，其余保持原相对顺序
           (perm (cons ax (loop for i below rank unless (= i ax) collect i)))
           (moved (vt-contiguous (vt-transpose tensor perm)))
           (data (vt-data moved))
           (n (first (vt-shape moved)))
           (m (if (zerop n) 0 (the fixnum (floor (vt-size moved) n))))
           (dt (vt-dtype tensor))
           (order (make-array n :element-type 'fixnum
                                :initial-contents (loop for i below n collect i)))
           ;; STABLE-SORT：同值行保持输入顺序 → 组首 = 首次出现（numpy 语义）
           (sorted (stable-sort order
                                (lambda (i j) (%axis-row-lessp data i j m))))
           (uniq-rows (make-array (max n 1) :element-type 'fixnum))
           (cnts (make-array (max n 1) :element-type '(signed-byte 64)))
           (inv (when return-inverse
                  (make-array n :element-type '(signed-byte 64))))
           (u 0) (pos 0))
      (declare (type fixnum u pos))
      ;; ---- run-length：等价行相邻，故与组首比较即可 ----
      (loop while (< pos n) do
        (let* ((start pos)
               (row (aref sorted pos)))
          (incf pos)
          (loop while (and (< pos n) (%axis-row-eq-p data row (aref sorted pos) m))
                do (incf pos))
          (setf (aref uniq-rows u) row)
          (setf (aref cnts u) (the fixnum (- pos start)))
          (when inv
            (loop for q of-type fixnum from start below pos
                  do (setf (aref inv (aref sorted q)) u)))
          (incf u)))
      ;; ---- 输出：把唯一行按「首轴 = 唯一值」组装，再转回原轴序 ----
      (let* ((rest-shape (rest (vt-shape moved)))
             (res (make-vt (cons u rest-shape) 0 :dtype dt))
             (rd (vt-data res))
             (inv-perm (let ((v (make-array rank :element-type 'fixnum)))
                         (loop for p in perm for i from 0
                               do (setf (aref v p) i))
                         (coerce v 'list))))
        (dotimes (i u)
          (declare (type fixnum i))
          (let ((src (the fixnum (* (aref uniq-rows i) m)))
                (dst (the fixnum (* i m))))
            (declare (type fixnum src dst))
            (dotimes (k m)
              (setf (aref rd (the fixnum (+ dst k)))
                    (aref data (the fixnum (+ src k)))))))
        (let* ((out (if (= ax 0)
                        res
                        ;; 轴序还原后再落地成 C 连续：numpy 的 unique(axis=)
                        ;; 返回的就是新数组（flags 连续），而本库的展平路径
                        ;; 也返回连续结果 —— 两条路径保持一致，避免用户拿到
                        ;; 「形状对、vt-contiguous-p 却为 NIL」的视图。
                        ;; 代价最多一次结果大小的拷贝（唯一行 ≤ 输入行）。
                        (vt-contiguous (vt-transpose res inv-perm))))
               (idx-vt (when return-index
                         (%make-vt :data (if (= u n) (copy-seq uniq-rows)
                                             (subseq uniq-rows 0 u))
                                   :shape (list u) :strides '(1)
                                   :offset 0 :dtype :int64)))
               (cnt-vt (%make-vt :data (if (= u n) (copy-seq cnts)
                                           (subseq cnts 0 u))
                                 :shape (list u) :strides '(1)
                                 :offset 0 :dtype :int64))
               (inv-vt (when inv
                         (%make-vt :data inv :shape (list n) :strides '(1)
                                   :offset 0 :dtype :int64))))
          (if (or return-index return-inverse return-counts)
              (values out
                      (when return-index idx-vt)
                      (when return-inverse inv-vt)
                      (when return-counts cnt-vt))
              out))))))

(defun %unique-slow (tensor)
  "旧实现（list 化 + stable-sort 下标 + 闭包比较器）。作为非特化元素类型的兜底。"
  (with-float-safe
    (let* ((flat (vt-flatten tensor))
           (n (vt-size flat))
           (src-data (vt-data flat))
           (dtype (vt-dtype flat))
           (elem-type (vt-element-type flat))
           (sorted-idx (make-array n :element-type '(signed-byte 64)
                                     :initial-contents
                                     (loop for i below n collect i))))
      (setf sorted-idx (stable-sort sorted-idx
                             (lambda (a b)
                               (let ((va (aref src-data a))
                                     (vb (aref src-data b)))
                                 (cond ((%nan-p va) nil)
                                       ((%nan-p vb) t)
                                       (t (< va vb)))))))
      (let ((unique-vals
              (make-array 0 :element-type elem-type :adjustable t
                            :fill-pointer t))
            (first-idx
              (make-array 0 :element-type '(signed-byte 64)
                            :adjustable t :fill-pointer t))
            (cnts (make-array 0 :element-type '(signed-byte 64)
                                :adjustable t :fill-pointer t))
            (inverse (make-array n :element-type '(signed-byte 64)
                                   :initial-element 0)))
        (loop with pos = 0
              while (< pos n)
              for uniq-num from 0
              for idx0 = (aref sorted-idx pos)
              for val = (aref src-data idx0)
              for start = pos
              do (vector-push-extend val unique-vals)
                 (vector-push-extend idx0 first-idx)
                 (if (%nan-p val)
                     (loop while (and (< pos n)
                                      (%nan-p (aref src-data (aref sorted-idx pos))))
                           for orig = (aref sorted-idx pos)
                           do (setf (aref inverse orig) uniq-num)
                              (incf pos))
                     (loop while (and (< pos n)
                                      (= val (aref src-data (aref sorted-idx pos))))
                           for orig = (aref sorted-idx pos)
                           do (setf (aref inverse orig) uniq-num) (incf pos)))
                 (vector-push-extend (- pos start) cnts))
        (values (vt-from-sequence unique-vals :dtype dtype)
                (vt-from-sequence first-idx :dtype :int64)
                (%make-vt :data inverse :shape (list n) :strides '(1)
                          :offset 0 :dtype :int64)
                (vt-from-sequence cnts :dtype :int64))))))

(defun %unique-flat (tensor return-index return-inverse return-counts)
  "展平去重（vt-unique 的 axis=NIL 路径；nan 折叠为一个）。"
  (with-float-safe
    (let* ((flat (vt-contiguous (vt-flatten tensor)))
           (n (vt-size flat))
           (dtype (vt-dtype tensor))
           (src (vt-data flat))
           (soff (the fixnum (vt-offset flat)))
           (et (array-element-type src)))
      (if (or (zerop n) (not (member et (list 'double-float 'single-float
                                              '(signed-byte 64) '(signed-byte 32))
                               :test #'equal)))
          ;; 空数组或非特化元素类型 → 旧路径（语义相同，仅速度差异）
          (multiple-value-bind (u i inv c) (%unique-slow tensor)
            (if (or return-index return-inverse return-counts)
                (values u
                        (when return-index i)
                        (when return-inverse inv)
                        (when return-counts c))
                u))
          (let* ((sorted (make-array n :element-type et))
                 (uniq-vals (make-array n :element-type et))
                 (cnts (make-array n :element-type '(signed-byte 64)))
                 (u 0) (pos 0))
            (declare (type fixnum u pos))
            (replace sorted src :start2 soff :end2 (the fixnum (+ soff n)))
            (%sort-fibers sorted n 1)
            ;; ---- run-length：升序 + NaN 稳定压尾，故同值必相邻 ----
            (loop while (< pos n) do
              (let* ((val (aref sorted pos))
                     (nan (and (floatp val) (%nan-p val)))
                     (start pos))
                (declare (type fixnum start))
                (if nan
                    (loop while (and (< pos n)
                                     (let ((w (aref sorted pos)))
                                       (and (floatp w) (%nan-p w))))
                          do (incf pos))
                    (loop while (and (< pos n) (= val (aref sorted pos)))
                          do (incf pos)))
                (setf (aref uniq-vals u) val)
                (setf (aref cnts u) (the fixnum (- pos start)))
                (incf u)))
            (let* ((udata (if (= u n) uniq-vals (subseq uniq-vals 0 u)))
                   (uniq-vt (%make-vt :data udata :shape (list u)
                                      :strides '(1) :offset 0 :dtype dtype))
                   (cdata (if (= u n) cnts (subseq cnts 0 u)))
                   (cnt-vt (%make-vt :data cdata :shape (list u)
                                     :strides '(1) :offset 0 :dtype :int64))
                   (idx-vt nil) (inv-vt nil))
              ;; ---- 需要「原始位置」时才建哈希 ----
              (when (or return-index return-inverse)
                (let ((fm (subtypep et 'float))
                      (h (make-hash-table :test #'eql))
                      (first-idx (make-array u :element-type '(signed-byte 64)))
                      (inv (when return-inverse
                             (make-array n :element-type '(signed-byte 64)))))
                  (dotimes (i u)
                    (setf (gethash (%membership-key (aref uniq-vals i) fm) h) i))
                  (dotimes (i u)
                    (setf (aref first-idx i) -1))
                  (dotimes (i n)
                    (let* ((v (aref src (the fixnum (+ soff i))))
                           (uidx (gethash (%membership-key v fm) h)))
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
                  uniq-vt)))))))

(defun vt-unique (tensor &key return-index return-inverse return-counts axis)
  "唯一元素（对标 numpy.unique）。

   AXIS = NIL（默认）：展平去重，结果升序、NaN 折叠为一个。
   AXIS = 整数：沿该轴把切片当元素去重（numpy 2.x 语义）——
     · 结果是「去重后的切片」，该轴长度替换为唯一行数，其余轴顺序不变；
     · 行按逐元素字典序排序（numpy 对 structured dtype 的排序约定），
       NaN 排末尾；含 NaN 的行**不合并**（NaN != NaN，与展平模式不同）；
     · return-index 是沿 AXIS 的原始下标；return-inverse 是长度 = 该轴长度
       的一维数组（可用 (vt-take u inv :axis axis) 重建输入）；
       return-counts 是每个唯一行的出现次数。
   秩为 1 时 AXIS 被忽略（numpy 同样走展平分支）。

   复数（§16.8）：展平路径按 (实部, 虚部) 字典序，NaN 折叠为一个；
   AXIS 路径无需第二份实现 —— %AXIS-ELEM-LESSP / %AXIS-ELEM-EQ 本身
   与元素类型无关，故 AXIS 路径无需复数分支。展平路径的复数比较器在
   complex-ops.lisp；本函数只做分流（§16.32）。"
  (let* ((in (ensure-vt tensor))
         (rank (length (vt-shape in))))
    (if (and axis (> rank 1))
        (%unique-axis in (vt-normalize-axis axis rank)
                      return-index return-inverse return-counts)
        (if (%cx-complex-dtype-p* (vt-dtype in))
            (%cx-op-unique in return-index return-inverse return-counts axis)
            (%unique-flat tensor return-index return-inverse return-counts)))))

(defun %set-filter (u keep-table invert)
  "按 KEEP-TABLE 过滤已升序的 U（保持顺序），返回与 U 同 dtype 的新张量。
   INVERT 为真时保留「不在表中」的元素。"
  (let* ((n (vt-size u))
         (dt (vt-dtype u))
         (src (vt-data u))
         (et (array-element-type src))
         (fm (subtypep et 'float))
         (buf (make-array n :element-type et))
         (k 0))
    (declare (type fixnum k))
    (with-float-safe
      (dotimes (i n)
        (let* ((v (aref src i))
               (hit (not (null (gethash (%membership-key v fm) keep-table)))))
          (when (if invert (not hit) hit)
            (setf (aref buf k) v)
            (incf k)))))
    (%make-vt :data (if (= k n) buf (subseq buf 0 k))
              :shape (list k) :strides '(1) :offset 0 :dtype dt)))

(defun vt-intersect1d (t1 t2)
  "求两个 1D 张量的交集，结果升序去重（对标 numpy.intersect1d）。"
  (when (%cx-any-p t1 t2)
    (return-from vt-intersect1d (%cx-op-intersect1d t1 t2)))
  (let* ((v1 (ensure-vt t1))
         (v2 (ensure-vt t2))
         (u2 (vt-unique v2))
         (fm (%membership-float-mode-p v1 v2)))
    ;; u1 本身升序（NaN 在末尾），过滤后天然仍是升序 → 无需再排序
    (%set-filter (vt-unique v1) (%membership-table u2 fm) nil)))

(defun vt-union1d (t1 t2)
  "求两个 1D 张量的并集，结果升序去重（对标 numpy.union1d）。"
  (when (%cx-any-p t1 t2)
    (return-from vt-union1d (%cx-op-union1d t1 t2)))
  (vt-unique (vt-concatenate 0 (vt-unique (ensure-vt t1)) (vt-unique (ensure-vt t2)))))

(defun vt-setdiff1d (t1 t2)
  "求差集 t1 \ t2（在 t1 中但不在 t2），结果升序去重（对标 numpy.setdiff1d）。"
  (when (%cx-any-p t1 t2)
    (return-from vt-setdiff1d (%cx-op-setdiff1d t1 t2)))
  (let* ((v1 (ensure-vt t1))
         (v2 (ensure-vt t2))
         (fm (%membership-float-mode-p v1 v2)))
    (%set-filter (vt-unique v1) (%membership-table (vt-unique v2) fm) t)))

(defun vt-setxor1d (t1 t2)
  "求对称差（仅出现在其中一个集合中），结果升序去重（对标 numpy.setxor1d）。"
  (when (%cx-any-p t1 t2)
    (return-from vt-setxor1d (%cx-op-setxor1d t1 t2)))
  (let* ((v1 (ensure-vt t1))
         (v2 (ensure-vt t2))
         (dt (vt-dtype v1))
         (fm (%membership-float-mode-p v1 v2))
         (u1 (vt-unique v1))
         (u2 (vt-unique v2))
         (only1 (%set-filter u1 (%membership-table u2 fm) t))
         (only2 (%set-filter u2 (%membership-table u1 fm) t))
         ;; 两段各自升序，但拼起来未必有序；交给 vt-numpy-sort 合并。
         ;; 过滤后规模 ≤ |u1|+|u2|，比原实现少了 O(n×m) 的 member 扫描。
         (merged (append (coerce (vt-data only1) 'list)
                         (coerce (vt-data only2) 'list))))
    (vt-from-sequence (vt-numpy-sort merged #'<) :dtype dt)))

(defun vt-in1d (t1 t2)
  "逐元素判定是否属于 TEST（对标 numpy.in1d）。返回与元素形状相同的
   **:bool** 布尔张量（§17）。
   v0.4.1 修复（差分测试 R7）：此前返回 :float64。
   v0.4.2 性能修复：哈希会员判定，O(n+m)（原 O(n×m)，1M×1k 需 87s）。"
  (when (%cx-any-p t1 t2)
    (return-from vt-in1d (%cx-op-in1d t1 t2)))
  (let* ((elem (ensure-vt t1))
         (test (ensure-vt t2))
         (fm (%membership-float-mode-p elem test)))
    (%membership-map elem (%membership-table (vt-unique test) fm) fm nil)))
