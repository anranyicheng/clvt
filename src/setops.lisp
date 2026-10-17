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
  "对 ELEM 逐元素查表，返回 :int8 的 0/1（INVERT 时取反）。"
  (vt-map (lambda (x)
            (let ((hit (if (gethash (%membership-key x float-mode) table) 1 0)))
              (if invert (- 1 hit) hit)))
          elem :dtype :int8))

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

(defun vt-unique (tensor &key return-index return-inverse return-counts)
  "返回展平后唯一元素（升序，nan 视为同一值，只保留一个）。
   可选返回 return-index（首次出现下标）/ return-inverse（原→唯一映射）/
   return-counts（出现次数）。语义对标 numpy.unique。"
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
  (let* ((v1 (ensure-vt t1))
         (v2 (ensure-vt t2))
         (u2 (vt-unique v2))
         (fm (%membership-float-mode-p v1 v2)))
    ;; u1 本身升序（NaN 在末尾），过滤后天然仍是升序 → 无需再排序
    (%set-filter (vt-unique v1) (%membership-table u2 fm) nil)))

(defun vt-union1d (t1 t2)
  "求两个 1D 张量的并集，结果升序去重（对标 numpy.union1d）。"
  (vt-unique (vt-concatenate 0 (vt-unique (ensure-vt t1)) (vt-unique (ensure-vt t2)))))

(defun vt-setdiff1d (t1 t2)
  "求差集 t1 \ t2（在 t1 中但不在 t2），结果升序去重（对标 numpy.setdiff1d）。"
  (let* ((v1 (ensure-vt t1))
         (v2 (ensure-vt t2))
         (fm (%membership-float-mode-p v1 v2)))
    (%set-filter (vt-unique v1) (%membership-table (vt-unique v2) fm) t)))

(defun vt-setxor1d (t1 t2)
  "求对称差（仅出现在其中一个集合中），结果升序去重（对标 numpy.setxor1d）。"
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
  "逐元素判定是否属于 TEST（对标 numpy.in1d）。返回与元素形状相同的布尔张量。
   dtype 为 :int8（F4：比较/逻辑结果一律 :int8）。
   v0.4.1 修复（差分测试 R7）：此前返回 :float64。
   v0.4.2 性能修复：哈希会员判定，O(n+m)（原 O(n×m)，1M×1k 需 87s）。"
  (let* ((elem (ensure-vt t1))
         (test (ensure-vt t2))
         (fm (%membership-float-mode-p elem test)))
    (%membership-map elem (%membership-table (vt-unique test) fm) fm nil)))
