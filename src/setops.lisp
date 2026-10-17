;;;; setops.lisp — 集合操作

(in-package :clvt)

(defun vt-unique (tensor &key return-index return-inverse return-counts)
  "返回展平后唯一元素（升序，nan 视为同一值）。"
  (with-float-safe
    (let* ((flat (vt-flatten tensor))
           (n (vt-size flat))
           (src-data (vt-data flat))
           (dtype (vt-dtype flat))
           (elem-type (vt-element-type flat))
           (sorted-idx (make-array n :element-type '(signed-byte 64)
                                     :initial-contents
                                     (loop for i below n collect i))))
      ;; v0.4.0 修复：sort 不稳定，相同值的多处出现顺序不定，
      ;; 导致 return-index 取到的可能不是「首次出现」下标。
      ;; NumPy 语义：return-index 必须返回每个唯一值的**首次出现**位置，
      ;; 故改用 stable-sort 保持原相对顺序（首个出现的下标先被访问）。
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
        (let ((uniq-vt (vt-from-sequence unique-vals :dtype dtype))
              (idx-vt (when return-index (vt-from-sequence first-idx :dtype :int64)))
              (inv-vt (when return-inverse
                        (let ((v (make-array n :element-type '(signed-byte 64))))
                          (dotimes (i n)
                            (setf (aref v i) (aref inverse i)))
                          (%make-vt :data v :shape (list n) :strides '(1)
                                    :offset 0 :dtype :int64))))
              (cnt-vt (when return-counts (vt-from-sequence cnts :dtype :int64))))
          (if (or return-index return-inverse return-counts)
              (values uniq-vt
                      (when return-index idx-vt)
                      (when return-inverse inv-vt)
                      (when return-counts cnt-vt))
              uniq-vt))))))

(defun vt-intersect1d (t1 t2)
  "求两个 1D 张量的交集，结果升序去重（对标 numpy.intersect1d）。"
  (let* ((u1 (vt-unique t1))
         (u2 (vt-unique t2))
         (u2-set (coerce (vt-data u2) 'list))
         (result '()))
    (vt-do-each (ptr val u1)
      (declare (ignore ptr))
      (when (member val u2-set :test #'vt-float-nan-inf-=)
        (push val result)))
    (vt-from-sequence (vt-numpy-sort result #'<) :dtype (vt-dtype t1))))

(defun vt-union1d (t1 t2)
  "求两个 1D 张量的并集，结果升序去重（对标 numpy.union1d）。"
  (vt-unique (vt-concatenate 0 (vt-unique t1) (vt-unique t2))))

(defun vt-setdiff1d (t1 t2)
  "求差集 t1 \ t2（在 t1 中但不在 t2），结果升序去重（对标 numpy.setdiff1d）。"
  (let* ((u1 (vt-unique t1))
         (u2 (vt-unique t2))
         (u2-set (coerce (vt-data u2) 'list))
         (result '()))
    (vt-do-each (ptr val u1)
      (declare (ignore ptr))
      (unless (member val u2-set :test #'vt-float-nan-inf-=)
        (push val result)))
    (vt-from-sequence (vt-numpy-sort result #'<) :dtype (vt-dtype t1))))

(defun vt-setxor1d (t1 t2)
  "求对称差（仅出现在其中一个集合中），结果升序去重（对标 numpy.setxor1d）。"
  (let* ((u1 (vt-unique t1))
         (u2 (vt-unique t2))
         (u1-set (coerce (vt-data u1) 'list))
         (u2-set (coerce (vt-data u2) 'list))
         (result '()))
    (vt-do-each (ptr val u1)
      (declare (ignore ptr))
      (unless (member val u2-set :test #'vt-float-nan-inf-=)
        (push val result)))
    (vt-do-each (ptr val u2)
      (declare (ignore ptr))
      (unless (member val u1-set :test #'vt-float-nan-inf-=)
        (push val result)))
    (vt-from-sequence (vt-numpy-sort result #'<) :dtype (vt-dtype t1))))

;;; ------------------------------------------------------------------
;;; 会员判定（in1d / isin 共用）
;;;
;;; v0.4.2 性能修复：原实现把 TEST 装成 **list**，再对每个待判定元素做
;;; `(member x set :test #'vt-float-nan-inf-=)` —— 即 O(n×m) 次泛型浮点
;;; 比较。1M 元素 × 1k 测试集 = 1e9 次调用，实测 **87 秒**
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
;;; ------------------------------------------------------------------

(defun %membership-float-mode-p (a b)
  "两个备查数组中任一侧为浮点 → 用「全 double-float」键模式。"
  (flet ((f (v) (subtypep (array-element-type (vt-data v)) 'float)))
    (or (f a) (f b))))

(defun %membership-key (x float-mode)
  (cond ((floatp x)
         (with-float-safe
           (cond ((not (= x x)) :%clvt-nan)
                 ((zerop x) 0.0d0)
                 (t (coerce x 'double-float)))))
        (float-mode (coerce x 'double-float))
        (t x)))

(defun %membership-table (test float-mode)
  "把 TEST 的元素装进 EQL 哈希表，供 O(1) 会员判定。"
  (let ((h (make-hash-table :test #'eql))
        (flat (vt-contiguous (vt-flatten test)))
        (n (vt-size test)))
    (dotimes (i n)
      (setf (gethash (%membership-key (vt-ref flat i) float-mode) h) t))
    h))

(defun %membership-map (elem table float-mode invert)
  "对 ELEM 逐元素查表，返回 :int8 的 0/1（INVERT 时取反）。"
  (vt-map (lambda (x)
            (let ((hit (if (gethash (%membership-key x float-mode) table) 1 0)))
              (if invert (- 1 hit) hit)))
          elem :dtype :int8))

(defun vt-in1d (t1 t2)
  "逐元素判定是否属于 TEST（对标 numpy.in1d）。返回与元素形状相同的布尔张量。
   dtype 为 :int8（F4：比较/逻辑结果一律 :int8）。
   v0.4.1 修复（差分测试 R7）：此前返回 :float64。
   v0.4.2 性能修复：哈希会员判定，O(n+m)（原 O(n×m)，1M×1k 需 87s）。"
  (let* ((elem (ensure-vt t1))
         (test (ensure-vt t2))
         (fm (%membership-float-mode-p elem test)))
    (%membership-map elem (%membership-table (vt-unique test) fm) fm nil)))
