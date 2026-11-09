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

(declaim (inline %membership-bits))
(defun %membership-bits (d)
  "`DOUBLE-FLOAT` 的 IEEE-754 位模式（高低 32 位交换后），用作哈希表的**整数**键。

   v0.6.x：`EQL` 哈希表装不下「规整浮点」—— 而规整浮点恰恰是真实数据的常态
   （计数、金额、坐标、0.5 步长的采样点）。SBCL 的 `SXHASH` 分桶只看低位，
   而 0.5 步长的 double 位模式低 21 位以上恒为 0：两万个键全部落进**同一个桶**，
   建表 **678 ms**、二十万次查询 **11.5 s**（每元素约 58 µs 的线性扫描）。
   这不是 `COERCE` 的锅 —— 把数组元素原样当键也一样慢（11558 ms）；同一张表
   换随机尾数的 double 只要 **4 ms / 14 ms**。所以代价完全由「键的形状」决定，
   与元素是 f32 还是 f64 **无关**：同为 0.5 步长时 f64 与 f32 的 `vt-in1d`
   都是 3.07 s（见 §7.4.15）。

   换成位模式整数键后退化**仍然存在**：位模式的低位同样是一串 0（f32 升上来的
   低 29 位恒 0，原生 double 的规整值也至少有 21 位是 0）—— 建表 807 ms、
   查询 14.0 s。**高低 32 位交换**（仍是单射）把有信息的位挪到低位：
   3 ms / 9 ms。

   `+0.0d0` 的位模式就是 0，所以 `(ZEROP X)` 归一后与整数 0 同键 —— 与
   「-0.0 与 0.0 相等」的既有语义一致。"
  (let ((b (sb-kernel:double-float-bits d)))
    (logior (ash (logand b #xFFFFFFFF) 32) (ash b -32))))

(defun %membership-key (x float-mode)
  "会员判定的哈希键。**调用方必须已屏蔽浮点陷阱**（`(not (= x x))` 在
   NaN 上会触发 FLOATING-POINT-INVALID-OPERATION）。
   v0.4.3：去掉内层 with-float-safe —— 它等价于每个元素各改一次 FP 控制字，
   在 1M 元素上本身就是可观的固定开销；所有调用点（vt-map 外壳 /
   %membership-table / 本文件的循环）都已在屏蔽区内。
   v0.6.x：浮点键改为位模式整数（见 `%MEMBERSHIP-BITS`）。"
  (cond ((floatp x)
         (cond ((not (= x x)) :%clvt-nan)
               ((zerop x) 0)
               (t (%membership-bits (float x 1.0d0)))))
        (float-mode (%membership-bits (coerce x 'double-float)))
        (t x)))

(defun %membership-table (test float-mode)
  "把 TEST 的元素装进 EQL 哈希表，供 O(1) 会员判定。

   v0.6.x：按元素类型走三条**键就绪**的循环，不再逐元素调 `%MEMBERSHIP-KEY`。
   后者对每个元素要做 `FLOATP` 分派 + `(FLOAT x 1.0d0)` —— 而 `FLOAT` 是 CL 的
   泛型转换，参数类型未知时走 typecase，实测 100k 元素就要 **200 ms**
   （同规模位模式内联只要 3 ms）。"
  (let ((h (make-hash-table :test #'eql))
        (flat (vt-contiguous (vt-flatten test)))
        (n (vt-size test))
        (et (array-element-type (vt-data test))))
    (with-float-safe
      (cond
        ((and float-mode (equal et 'double-float))
         (let ((d (the (simple-array double-float (*)) (vt-data flat))))
           (dotimes (i n)
             (let ((x (aref d i)))
               (setf (gethash (cond ((/= x x) :%clvt-nan)
                                    ((zerop x) 0)
                                    (t (%membership-bits x)))
                              h)
                     t)))))
        ((and (not float-mode) (equal et '(signed-byte 64)))
         (let ((d (the (simple-array (signed-byte 64) (*)) (vt-data flat))))
           (dotimes (i n) (setf (gethash (aref d i) h) t))))
        (t
         (let ((d (vt-data flat)))
           (dotimes (i n)
             (setf (gethash (%membership-key (aref d i) float-mode) h) t))))))
    h))

;;; ------------------------------------------------------------------
;;; 查表内核：表与 INVERT 用特殊变量传，算子必须是**具名符号**
;;;
;;; v0.6.x 执行层（§7.4.15）：原实现是 `(vt-map (lambda (x) …) elem)` ——
;;; 匿名闭包让 `vt-fast-map` 退回 `(apply #'vt-map …)`，每元素一次 funcall
;;; + 泛型 aref + 装箱。1M 元素实测 **142 ms**，其中查表本身只占约 35 ms；
;;; 同样的输入交给 `vt-fast-map` 的恒等算子只要 7.3 ms。
;;;
;;; `vt-fast-map` 只对 `(function <符号>)` 字面形式内联，所以「表」不能是
;;; 闭包捕获的变量，只能是特殊变量（每元素一次 special 读约 2 ns）。
;;; ------------------------------------------------------------------

(defvar *membership-lookup* nil
  "会员判定的目标哈希表。仅由 `%MEMBERSHIP-MAP` 动态绑定。")
(defvar *membership-floating* nil
  "键模式：真 = 全 double-float（任一侧浮点），假 = 保留整数。")
(defvar *membership-invert* nil
  "`vt-in1d` / `vt-isin` 的 INVERT 参数（真 = 取反）。")

(defun %op-in-set (x)
  "会员判定内核（通用键）。调用方须已屏蔽浮点陷阱（`VT-FAST-MAP` 内层有）。"
  (let ((hit (if (gethash (%membership-key x *membership-floating*)
                          *membership-lookup*)
                 1 0)))
    (if *membership-invert* (- 1 hit) hit)))

(defun %op-in-set-f64 (x)
  "`DOUBLE-FLOAT` 输入的键就绪版：跳过 `%MEMBERSHIP-KEY` 的 FLOATP/COERCE 分派。
   ±0 归一、NaN 归哨兵、位模式整数键三条语义与 `%MEMBERSHIP-KEY` 逐字相同。

   这里**不能**写 `(declare (type double-float x))`：`VT-FAST-MAP` 会把同一份
   算子体展开进 10 路按元素类型的掩码内核，整数那几路的 AREF 类型对不上，
   SBCL 会给 10 条警告。类型信息在这种「一体多用」的算子上只能靠调用点保证。"
  (let ((hit (if (gethash (cond ((/= x x) :%clvt-nan)
                               ((zerop x) 0)
                               (t (%membership-bits x)))
                          *membership-lookup*)
                 1 0)))
    (if *membership-invert* (- 1 hit) hit)))

(defun %op-in-set-i64 (x)
  "`(SIGNED-BYTE 64)` 输入的键就绪版：键就是值本身（整数域不需要规范化）。
   这条路径原先走通用算子，每元素多做两次 `FLOATP` 与一次 `%MEMBERSHIP-BITS`
   分派 —— 1M 元素 27 ms，键就绪后 5 ms。"
  (let ((hit (if (gethash x *membership-lookup*) 1 0)))
    (if *membership-invert* (- 1 hit) hit)))

(defun %membership-prepare (elem test)
  "把两侧规整到**同一个键尺度**，返回 (values ELEM' TEST' FLOAT-MODE)。

   任一侧浮点 → 两侧都 `:float64`（键 = IEEE-754 位模式）；否则两侧都 `:int64`
   （键 = 整数值）。这与 `%MEMBERSHIP-KEY` 的等价关系逐条一致 —— 浮点侧本来就
   把所有键 `COERCE` 到 double，整数侧本来就保留整数。

   为什么要**在数组层面**转换而不是逐元素：`(FLOAT x 1.0d0)` 是 CL 的泛型
   转换，参数类型未知时走 typecase，100k 元素实测 **200 ms**；而 `VT-ASTYPE`
   整段转换 1M 元素只要 3 ms。转换只做一次，之后的循环里没有类型分派。"
  (let* ((e (ensure-vt elem))
         (t2 (ensure-vt test))
         (ef (subtypep (array-element-type (vt-data e)) 'float))
         (tf (subtypep (array-element-type (vt-data t2)) 'float)))
    (if (or ef tf)
        (values (if (eq (vt-dtype e) :float64) e (vt-astype e :float64))
                (if (eq (vt-dtype t2) :float64) t2 (vt-astype t2 :float64))
                t)
        (values (if (eq (vt-dtype e) :int64) e (vt-astype e :int64))
                (if (eq (vt-dtype t2) :int64) t2 (vt-astype t2 :int64))
                nil))))

(defun %membership-map (elem table float-mode invert)
  "对 ELEM 逐元素查表，返回 **:bool** 张量（INVERT 时取反）。
   与 numpy 一致：`in1d` / `isin` 返回 bool_ 数组，不是整数掩码（§17）。"
  (with-float-safe
    (let ((*membership-lookup* table)
          (*membership-floating* float-mode)
          (*membership-invert* invert))
      (cond ((and float-mode (eq (vt-dtype elem) :float64))
             (vt-fast-map #'%op-in-set-f64 elem :dtype :bool))
            ((and (not float-mode) (eq (vt-dtype elem) :int64))
             (vt-fast-map #'%op-in-set-i64 elem :dtype :bool))
            (t (vt-fast-map #'%op-in-set elem :dtype :bool))))))

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

;;; ------------------------------------------------------------------
;;; 集合运算的归并内核（intersect / setdiff / union / setxor 共用）
;;;
;;; v0.6.x 执行层（§7.4.15）：这四个函数原先都走「哈希表 + 逐元素 gethash」，
;;; 而两侧的 `vt-unique` 输出**本来就是升序**（NaN 在末尾）—— 有序数组求交并差
;;; 是一趟双指针，不需要哈希：
;;;
;;;   · 1M×100k 实测（键已是位模式整数）：intersect 233 → 188 ms、setdiff 245 → 202、
;;;     union 324 → 205、setxor **493 → 195**（原来还要 COERCE 到 list 再排一次）；
;;;   · 每元素省掉「键规范化 + 一次 gethash」（交换后的位模式键是 bignum，约 120 ns）；
;;;   · union 收益最大：它原来没有键可用，只能把两侧拼起来重排一次（1.58×）。
;;;
;;; 注意这只是本轮的**第二**收益。第一收益在 `%MEMBERSHIP-BITS`：改前的键让
;;; 这四个函数在规整数据上退化成 O(n×m)（1M×100k 约 5 分钟），先修键才有上面
;;; 这些毫秒数可谈。
;;;
;;; 比较语义必须与 `%MEMBERSHIP-KEY` 的等价关系逐条一致：
;;;   · ±0.0 相等      → `<` 与 `>` 都为假，落进「相等」分支；
;;;   · NaN 相等       → 排序已把 NaN 稳定压到尾，三路比较里显式判 NaN（NaN 最大）；
;;;   · f32 → f64 精确 → `%SET-ALIGN` 在数组层面统一到 `:float64`；
;;;   · 整数不丢精度   → 两侧皆整数时统一到 `:int64`（不是 double）。
;;; ------------------------------------------------------------------

(defun %set-align (v1 v2)
  "把两个**已升序**的集合输入对齐到同一比较域，返回 (values A B OK)。
   OK 为 NIL 表示元素类型不在归并内核的支持集内，调用方回退哈希路径。"
  (let ((et1 (array-element-type (vt-data v1)))
        (et2 (array-element-type (vt-data v2))))
    (cond
      ((equal et1 et2) (values v1 v2))
      ;; 任一侧浮点 → 统一 double（f32 → f64 精确；整数 → f64 会丢 >2^53 的精度，
      ;; 但那正是 %MEMBERSHIP-KEY 的既有语义，不能在这里悄悄改）
      ((or (subtypep et1 'float) (subtypep et2 'float))
       (values (if (equal et1 'double-float) v1 (vt-astype v1 :float64))
               (if (equal et2 'double-float) v2 (vt-astype v2 :float64))))
      ;; 两侧皆整数 → 统一 int64（无损；走 double 反而会丢精度）
      (t (values (if (equal et1 '(signed-byte 64)) v1 (vt-astype v1 :int64))
                 (if (equal et2 '(signed-byte 64)) v2 (vt-astype v2 :int64)))))))

(defmacro %set-merge% (lt mode d1 n1 d2 n2 out k)
  "按 MODE 归并两个升序数组。D1/D2/OUT 的元素类型为 LT，K 为输出游标。
   MODE 的动作表：输出「相等」「仅左」「仅右」三格各自开关。

   循环一律写成 `(LOOP (WHEN 终止 (RETURN)) BODY)` 而不是 `(LOOP WHILE C DO BODY)`：
   后者的 body 没有结束标记，宏拼接时只要多一个括号，后面的「剩余拷贝」循环就会
   变成主循环 body 的一部分（每轮都跑，把游标直接推到末尾）—— 这个坑本轮踩过一次，
   症状是 diff 输出整个左集合而不是差集。"
  (let* ((emit-eq   (and (member mode '(:intersect :union)) t))
         (emit-left  (and (member mode '(:diff :union :symdiff)) t))
         (emit-right (and (member mode '(:union :symdiff)) t))
         (i (gensym "I")) (j (gensym "J")) (kk (gensym "K"))
         (x (gensym "X")) (y (gensym "Y")))
    (flet ((emit (src) (if src `((setf (aref ,out ,kk) ,src) (incf ,kk)) nil)))
      `(let ((,i 0) (,j 0) (,kk ,k))
         (declare (type fixnum ,i ,j ,kk))
         ;; ---- 主循环：两侧都没走完 ----
         (loop
           (when (or (>= ,i ,n1) (>= ,j ,n2)) (return))
           (let ((,x (aref ,d1 ,i)) (,y (aref ,d2 ,j)))
             (cond
               ((< ,x ,y) ,@(emit (and emit-left x)) (incf ,i))
               ((> ,x ,y) ,@(emit (and emit-right y)) (incf ,j))
               ,@(if (subtypep lt 'float)
                     `((t (cond ((and (%nan-p ,x) (%nan-p ,y))
                                 ;; 排序已把 NaN 压到尾；两个 NaN 按「相等」处理
                                 ;; （`%MEMBERSHIP-KEY` 的哨兵键语义）
                                 ,@(emit (and emit-eq x)) (incf ,i) (incf ,j))
                                ;; X 是 NaN、Y 不是 → Y 在前（NaN 排末尾）
                                ((%nan-p ,x) ,@(emit (and emit-right y)) (incf ,j))
                                ((%nan-p ,y) ,@(emit (and emit-left x)) (incf ,i))
                                ;; 两个都不是 NaN 且比较相等 → 真等
                                (t ,@(emit (and emit-eq x)) (incf ,i) (incf ,j)))))
                     `((t ,@(emit (and emit-eq x)) (incf ,i) (incf ,j)))))))
         ;; ---- 剩余：一侧已耗尽 ----
         ,@(when emit-left
             `((loop
                 (when (>= ,i ,n1) (return))
                 (setf (aref ,out ,kk) (aref ,d1 ,i))
                 (incf ,i) (incf ,kk))))
         ,@(when emit-right
             `((loop
                 (when (>= ,j ,n2) (return))
                 (setf (aref ,out ,kk) (aref ,d2 ,j))
                 (incf ,j) (incf ,kk))))
         (setf ,k ,kk)))))

(defmacro %set-merge (u1 u2 mode out-dtype)
  "归并两个已升序（NaN 末尾、同元素类型）的集合，返回新 VT；不可归并时 NIL。
   MODE ∈ :intersect / :diff / :union / :symdiff。"
  (let ((v1 (gensym "V1")) (v2 (gensym "V2")) (d1 (gensym "D1")) (d2 (gensym "D2"))
        (n1 (gensym "N1")) (n2 (gensym "N2")) (out (gensym "OUT")) (k (gensym "K"))
        (et (gensym "ET")))
    `(block nil
       (let* ((,v1 ,u1) (,v2 ,u2)
              (,d1 (vt-data ,v1)) (,d2 (vt-data ,v2))
              (,n1 (vt-size ,v1)) (,n2 (vt-size ,v2))
              (,et (array-element-type ,d1)))
         (when (equal ,et (array-element-type ,d2))
           (unless (member ,et '(double-float single-float
                                 (signed-byte 64) (signed-byte 32))
                           :test #'equal)
             (return nil))
           (let ((,out (make-array (the fixnum (+ ,n1 ,n2)) :element-type ,et))
                 (,k 0))
             (declare (type fixnum ,k))
             (cond ((equal ,et 'double-float)
                    (%set-merge% double-float ,mode ,d1 ,n1 ,d2 ,n2 ,out ,k))
                   ((equal ,et 'single-float)
                    (%set-merge% single-float ,mode ,d1 ,n1 ,d2 ,n2 ,out ,k))
                   ((equal ,et '(signed-byte 64))
                    (%set-merge% (signed-byte 64) ,mode ,d1 ,n1 ,d2 ,n2 ,out ,k))
                   (t
                    (%set-merge% (signed-byte 32) ,mode ,d1 ,n1 ,d2 ,n2 ,out ,k)))
             (return
               (%make-vt :data (if (= ,k (length ,out)) ,out (subseq ,out 0 ,k))
                         :shape (list ,k) :strides '(1) :offset 0
                         :dtype ,out-dtype))))))))

(defun %set-op (t1 t2 mode)
  "四个集合运算的公共外壳：unique × 2 → 对齐比较域 → 一趟归并。
   元素类型不在内核支持集（int8/int16/uint8/uint16/bool/复数已在上层分流）
   时回退旧的哈希路径，语义完全相同。"
  (with-float-safe                   ; 归并里的 < / > 会撞上 NaN（§7.4.15）
    (let* ((u1 (vt-unique (ensure-vt t1)))
           (u2 (vt-unique (ensure-vt t2)))
           ;; union 的结果类型是两侧的提升（numpy.union1d 同）；其余三个沿用左值。
           (out-dtype (if (eq mode :union)
                          (vt-promote-type (vt-dtype u1) (vt-dtype u2))
                          (vt-dtype u1))))
      (multiple-value-bind (a b) (%set-align u1 u2)
        ;; MODE 必须是**字面量**：`%SET-MERGE` 在宏展开期按它决定「输出相等 /
        ;; 仅左 / 仅右」三格的动作，传一个变量进去会得到「三格全关」的空结果。
        (let ((r (case mode
                   (:intersect (%set-merge a b :intersect out-dtype))
                   (:diff (%set-merge a b :diff out-dtype))
                   (:union (%set-merge a b :union out-dtype))
                   (t (%set-merge a b :symdiff out-dtype)))))
          (when r
            (return-from %set-op (if (eq (vt-dtype r) out-dtype)
                                     r
                                     ;; 归并域是 double/int64，结果要落回契约 dtype
                                     (vt-astype r out-dtype))))))
      ;; ---- 回退：元素类型不在内核支持集 ----
      (let ((fm (%membership-float-mode-p u1 u2)))
        (case mode
          (:intersect (%set-filter u1 (%membership-table u2 fm) nil))
          (:diff (%set-filter u1 (%membership-table u2 fm) t))
          (:union (vt-unique (vt-concatenate 0 u1 u2)))
          (t (let ((only1 (%set-filter u1 (%membership-table u2 fm) t))
                   (only2 (%set-filter u2 (%membership-table u1 fm) t)))
               (vt-unique (vt-concatenate 0 only1 only2)))))))))

(defun vt-intersect1d (t1 t2)
  "求两个 1D 张量的交集，结果升序去重（对标 numpy.intersect1d）。"
  (when (%cx-any-p t1 t2)
    (return-from vt-intersect1d (%cx-op-intersect1d t1 t2)))
  (%set-op t1 t2 :intersect))

(defun vt-union1d (t1 t2)
  "求两个 1D 张量的并集，结果升序去重（对标 numpy.union1d）。"
  (when (%cx-any-p t1 t2)
    (return-from vt-union1d (%cx-op-union1d t1 t2)))
  (%set-op t1 t2 :union))

(defun vt-setdiff1d (t1 t2)
  "求差集 t1 \ t2（在 t1 中但不在 t2），结果升序去重（对标 numpy.setdiff1d）。"
  (when (%cx-any-p t1 t2)
    (return-from vt-setdiff1d (%cx-op-setdiff1d t1 t2)))
  (%set-op t1 t2 :diff))

(defun vt-setxor1d (t1 t2)
  "求对称差（仅出现在其中一个集合中），结果升序去重（对标 numpy.setxor1d）。"
  (when (%cx-any-p t1 t2)
    (return-from vt-setxor1d (%cx-op-setxor1d t1 t2)))
  (%set-op t1 t2 :symdiff))

(defun vt-in1d (t1 t2)
  "逐元素判定是否属于 TEST（对标 numpy.in1d）。返回与元素形状相同的
   **:bool** 布尔张量（§17）。
   v0.4.1 修复（差分测试 R7）：此前返回 :float64。
   v0.4.2 性能修复：哈希会员判定，O(n+m)（原 O(n×m)，1M×1k 需 87s）。"
  (when (%cx-any-p t1 t2)
    (return-from vt-in1d (%cx-op-in1d t1 t2)))
  (multiple-value-bind (elem test fm) (%membership-prepare t1 t2)
    (%membership-map elem (%membership-table (vt-unique test) fm) fm nil)))
