;;;; entry-equal-apply-test.lisp — 数据入口 / 相等判定 / 沿轴应用 / 外积的回归测试
;;;;
;;;; 四个函数在同一类病根上：**每元素一次通用件**。都不是算法问题。
;;;;
;;;;   1. src/creation.lisp       %FROM-ARRAY-FILL（VT-FROM-ARRAY 的搬运层）
;;;;      旧实现逐元素 `(VT-CAST (row-major-aref arr i) dtype)`。VT-CAST 每次
;;;;      进出都要读写 FPU 控制字（实测 216 ns/次），1M 元素 = 250 ms 的
;;;;      「控制字搬运」。现在同元素类型整段 REPLACE（恒等），跨类型逐元素
;;;;      `%VT-CAST-RAW`、屏蔽提到循环外一次。1M f64：266 ms → 7 ms。
;;;;
;;;;   2. src/extensions3.lisp    VT-ARRAY-EQUAL
;;;;      旧实现把两侧都 `VT-ASTYPE … :float64` 再逐元素 `%VT-REF-RAW`。
;;;;      性能 1M 83 ms；语义上还有两条静默错值：虚部被 astype 丢掉
;;;;      （#C(1 2) 与 #C(1 3) 判等）、int64 超过 2^53 的相邻值被 double
;;;;      抹成同一个数。现在同 dtype 直接比，异 dtype 走 VT-PROMOTE-TYPE。
;;;;
;;;;   3. src/extensions2.lisp    VT-APPLY-ALONG-AXIS
;;;;      每个 fiber 三笔开销都按元素收钱：泛型 AREF 取数、VT-FROM-ARRAY
;;;;      里逐元素 VT-CAST、写回再逐元素 VT-CAST。1M 元素 700 ms。
;;;;
;;;;   4. src/linalg.lisp         VT-OUTER
;;;;      旧实现走 VT-EINSUM "i,j->ij"：无收缩，却仍按元素做 strides
;;;;      里程计。1M 元素 14.3 ms；改成两个广播视图的逐元素乘（numpy 的
;;;;      实现同构），1.6 ms。
;;;;
;;;; 断言分两层：
;;;;   A 段 —— 逐位/逐值等价。四个函数各自与**逐字复刻的旧实现**对拍，
;;;;       覆盖 dtype、物理形态（转置/切片/翻转/列视图）、退化形状、
;;;;       值域（NaN / ±Inf / ±0.0 / 2^53 边界 / 复数虚部 / 布尔）。
;;;;       旧实现本身错的地方（复数虚部、int64 大值）另外钉死期望值，
;;;;       期望值取自 numpy 2.4.6 实测。
;;;;   B 段 —— 机器无关的倍率护栏。锚点取「成本构成相近、又不在保护范围内」
;;;;       的算子（§16.16 第 7 条）：array-equal ÷ isclose、
;;;;       from-array ÷ 旧实现、outer ÷ einsum、apply ÷ 手写同访问模式地板。

(require :asdf)
#+quicklisp (ql:quickload :clvt :silent t)
(asdf:load-system :clvt)

(defpackage :clvt-entry-test
  (:use :cl :clvt)
  (:import-from :clvt
                #:ensure-vt #:vt-data #:vt-shape #:vt-dtype #:vt-size
                #:vt-offset #:vt-strides #:vt-cast #:%vt-cast-raw
                #:vt-dtype->lisp-type #:%item-number
                #:%apply-along-axis-result-dtype #:with-float-safe)
  (:export #:run))

(in-package :clvt-entry-test)

;;; ==================================================================
;;; 0. 断言与工具
;;; ==================================================================
(declaim (special *n* *pass* *fail* *failures*))
(defparameter *n* 0)
(defparameter *pass* 0)
(defparameter *fail* 0)
(defparameter *failures* nil)

(defun check (label ok &optional detail)
  (incf *n*)
  (if ok
      (progn (incf *pass*) (format t "  [ OK ] ~a~%" label))
      (progn (incf *fail*) (push label *failures*)
             (format t "  [FAIL] ~a~@[ —— ~a~]~%" label detail))))

(defun check-at-most (label ratio limit what)
  (incf *n*)
  (if (and (numberp ratio) (<= ratio limit))
      (progn (incf *pass*)
             (format t "  [ OK ] ~a（~a = ~,2f× ≤ ~,1f×）~%" label what (float ratio) limit))
      (progn (incf *fail*) (push label *failures*)
             (format t "  [FAIL] ~a（~a = ~,2f× > ~,1f×）~%" label what (float ratio) limit))))

(defparameter *nan* (sb-kernel:make-double-float #x7ff80000 0))
(defparameter *pinf* sb-ext:double-float-positive-infinity)
(defparameter *ninf* (- sb-ext:double-float-positive-infinity))

(defun best-ms-per (thunk &key (reps 3) (target-ms 20.0))
  "THUNK **单次调用**的平均耗时（毫秒）。取 REPS 批里最快的一批。

   为什么不能直接给单次调用计时：`GET-INTERNAL-REAL-TIME` 的刻度是 1 ms，
   1.5 ms 的操作单次只能读成 1 或 2（±30%），再叠上分配与 GC 的抖动
   （同一操作能读到 1、1、6），倍率护栏就成了随机数。这里先粗测一次，
   把迭代次数定成「一批 ≈ TARGET-MS」，再取整批的均值。"
  (funcall thunk)
  (let* ((p0 (get-internal-real-time))
         (probe (progn
                  (funcall thunk)
                  (/ (* 1000.0 (- (get-internal-real-time) p0))
                     internal-time-units-per-second)))
         (iters (max 1 (round target-ms (max probe 0.5))))
         (best most-positive-double-float))
    (dotimes (i reps best)
      (declare (ignore i))
      (let* ((t0 (get-internal-real-time))
             (per (progn
                    (dotimes (k iters) (declare (ignore k)) (funcall thunk))
                    (/ (* 1000.0 (- (get-internal-real-time) t0))
                       internal-time-units-per-second iters))))
        (when (< per best) (setf best per))))))

(defun same-bits (x y)
  "两个 vt 的 shape / dtype / 元素位模式全等。
   浮点用 EQL（SBCL 对浮点的 EQL 是位比较：0.0 与 -0.0 不相等，
   NaN 与同载荷 NaN 相等），整数/复数同理。"
  (and (equal (vt-shape x) (vt-shape y))
       (eq (vt-dtype x) (vt-dtype y))
       (let ((xd (vt-data x)) (yd (vt-data y))
             (n (vt-size x)))
         (dotimes (i n t)
           (unless (eql (aref xd i) (aref yd i))
             (return nil))))))

(defun val-list (dtype n)
  "按 dtype 造 N 个有代表性的值：符号零、NaN/Inf（浮点域）、
   int64 的 2^53 边界、复数虚部非零、布尔真值。"
  (let ((pool (ecase dtype
                (:float64 (list 0d0 -0d0 1.5d0 -2.25d0 *nan* *pinf* *ninf* 1d300))
                (:float32 (list 0.0 -0.0 1.5 -2.25 1e30))
                (:complex128 (list (complex 0d0 0d0) (complex 1d0 -2d0)
                                   (complex 0d0 1d0) (complex 1d300 2d0)))
                (:complex64 (list (complex 0.0 0.0) (complex 1.0 -2.0)
                                  (complex 0.0 1.0)))
                (:int64 (list 0 -1 7 9007199254740993 4611686018427387903))
                (:int32 (list 0 -1 7 2147483647 -2147483648))
                (:int16 (list 0 -1 7 32767 -32768))
                (:int8 (list 0 -1 7 127 -128))
                (:uint8 (list 0 1 7 255))
                (:uint16 (list 0 1 7 65535))
                (:bool (list t nil t t)))))
    (loop for i below n collect (nth (mod i (length pool)) pool))))

(defun mk (vals dtype shape)
  "按 SHAPE 把 VALUES 建成张量（VT-FROM-SEQUENCE 只吃嵌套结构，不吃 :SHAPE）。"
  (vt-reshape (vt-from-sequence vals :dtype dtype) (if (listp shape) shape (list shape))))

;;; ==================================================================
;;; 1. 旧实现复刻（逐字复制改动前的代码，只改函数名）
;;; ==================================================================
(defun %array-contains-complex-p-legacy (arr size)
  (dotimes (i size nil)
    (when (complexp (row-major-aref arr i))
      (return t))))

(defun %from-array-legacy (arr &key (dtype nil) (fast nil))
  "改动前的 VT-FROM-ARRAY（推断 + 逐元素 VT-CAST）。"
  (let* ((shape (array-dimensions arr))
         (size (vt-shape-to-size shape))
         (cl-etype (array-element-type arr))
         (infer (cond
                  ((subtypep cl-etype '(complex double-float)) :complex128)
                  ((subtypep cl-etype '(complex single-float)) :complex64)
                  ((subtypep cl-etype 'double-float) :float64)
                  ((subtypep cl-etype 'single-float) :float32)
                  ((subtypep cl-etype '(signed-byte 8))   :int8)
                  ((subtypep cl-etype '(unsigned-byte 8))  :uint8)
                  ((subtypep cl-etype '(signed-byte 16))  :int16)
                  ((subtypep cl-etype '(unsigned-byte 16)) :uint16)
                  ((subtypep cl-etype '(signed-byte 32)) :int32)
                  ((subtypep cl-etype '(signed-byte 64)) :int64)
                  ((subtypep cl-etype 'fixnum) :int64)
                  (t (if (%array-contains-complex-p-legacy arr size)
                         :complex128 :float64))))
         (final (or dtype infer))
         (lisp-type (vt-dtype->lisp-type final))
         (data (make-array size :element-type lisp-type)))
    (if fast
        (dotimes (i size)
          (setf (aref data i) (coerce (row-major-aref arr i) lisp-type)))
        (dotimes (i size)
          (setf (aref data i) (vt-cast (row-major-aref arr i) final))))
    (clvt::%make-vt :data data :shape shape :strides (vt-compute-strides shape)
                    :offset 0 :dtype final)))

(defun %array-equal-legacy (a b)
  "改动前的 VT-ARRAY-EQUAL（两侧都 astype :float64，再逐元素 %VT-REF-RAW）。"
  (let ((av (ensure-vt a)) (bv (ensure-vt b)))
    (and (equal (vt-shape av) (vt-shape bv))
         (let ((af (vt-ravel (vt-astype av :float64)))
               (bf (vt-ravel (vt-astype bv :float64)))
               (n (vt-size av)))
           (with-float-safe
             (dotimes (i n t)
               (unless (= (clvt::%vt-ref-raw af i) (clvt::%vt-ref-raw bf i))
                 (return nil))))))))

(defun %apply-along-axis-legacy (func axis vt)
  "改动前的 VT-APPLY-ALONG-AXIS（泛型 AREF + 逐元素 VT-CAST）。"
  (with-float-safe
    (let* ((vt-c (vt-contiguous vt))
           (shape (vt-shape vt-c))
           (rank (length shape))
           (ax (vt-normalize-axis axis rank))
           (ax-dim (nth ax shape))
           (ltype (vt-dtype->lisp-type (vt-dtype vt-c)))
           (before (subseq shape 0 ax))
           (after (subseq shape (1+ ax)))
           (outer (reduce #'* before :initial-value 1))
           (inner (reduce #'* after :initial-value 1))
           (data (vt-data vt-c))
           (data-off (vt-offset vt-c)))
      (declare (fixnum ax-dim outer inner))
      (let* ((probe (make-array ax-dim :element-type ltype
                                         :initial-element (coerce 0 ltype)))
             (probe-vt (vt-from-array probe :dtype (vt-dtype vt-c)))
             (sample (funcall func probe-vt))
             (out-dt (%apply-along-axis-result-dtype sample (vt-dtype vt-c))))
        (cond
          ((or (numberp sample) (and (vt-p sample) (null (vt-shape sample))))
           (let* ((out-shape (append before after))
                  (result (vt-zeros out-shape :dtype out-dt))
                  (rdata (vt-data result)))
             (dotimes (bi outer)
               (dotimes (ii inner)
                 (let ((arr (make-array ax-dim :element-type ltype)))
                   (dotimes (k ax-dim)
                     (setf (aref arr k)
                           (aref data (+ data-off
                                         (* bi ax-dim inner)
                                         (* k inner) ii))))
                   (let* ((sl (vt-from-array arr :dtype (vt-dtype vt-c)))
                          (rv (funcall func sl))
                          (raw (%item-number rv)))
                     (setf (aref rdata (+ (* bi inner) ii))
                           (vt-cast raw out-dt))))))
             result))
          ((and (vt-p sample) (= (length (vt-shape sample)) 1))
           (let* ((new-dim (car (vt-shape sample)))
                  (out-shape (append before (list new-dim) after))
                  (result (vt-zeros out-shape :dtype out-dt))
                  (rdata (vt-data result)))
             (declare (fixnum new-dim))
             (dotimes (bi outer)
               (dotimes (ii inner)
                 (let ((arr (make-array ax-dim :element-type ltype)))
                   (dotimes (k ax-dim)
                     (setf (aref arr k)
                           (aref data (+ data-off
                                         (* bi ax-dim inner)
                                         (* k inner) ii))))
                   (let* ((sl (vt-from-array arr :dtype (vt-dtype vt-c)))
                          (ov (funcall func sl)))
                     (dotimes (k new-dim)
                       (setf (aref rdata (+ (* bi new-dim inner)
                                            (* k inner) ii))
                             (vt-cast (clvt::%vt-ref-raw ov k) out-dt)))))))
             result))
          (t (error "vt-apply-along-axis: func must return scalar or 1D VT")))))))

;;; ==================================================================
;;; 2. VT-FROM-ARRAY：与旧实现逐位对拍
;;; ==================================================================
(defparameter *all-dtypes*
  '(:float64 :float32 :int64 :int32 :int16 :int8 :uint8 :uint16
    :complex128 :complex64 :bool))

(defun cl-array-of (etype values)
  "把 VALUES 装进元素类型为 ETYPE 的一维 CL 数组。"
  (let ((a (make-array (length values) :element-type etype)))
    (loop for v in values for i from 0 do (setf (aref a i) v))
    a))

(defparameter *source-etables*
  ;; (源数组元素类型 . 值池)
  (list (cons 'double-float (list 0d0 -0d0 1.5d0 *nan* *pinf* *ninf* 1d300))
        (cons 'single-float (list 0.0 -0.0 1.5 1e30))
        (cons '(signed-byte 64) (list 0 -1 7 9007199254740993))
        (cons '(signed-byte 32) (list 0 -1 7 2147483647))
        (cons '(signed-byte 16) (list 0 -1 7 32767))
        (cons '(signed-byte 8) (list 0 -1 7 127))
        (cons '(unsigned-byte 8) (list 0 1 7 255))
        (cons '(unsigned-byte 16) (list 0 1 7 65535))
        (cons '(complex double-float) (list (complex 0d0 0d0) (complex 1d0 -2d0)))
        (cons '(complex single-float) (list (complex 0.0 0.0) (complex 1.0 -2.0)))
        (cons t (list 0d0 1.5d0 (complex 1d0 2d0)))))

(defun run-from-array-tests ()
  (format t "~&~%[A1] vt-from-array 与旧实现逐位对拍（11 源类型 × 11 目标 dtype × 2 模式）~%")
  (let ((bad 0) (cases 0))
    (dolist (src *source-etables*)
      (destructuring-bind (etype . pool) src
        (let ((arr (cl-array-of etype pool)))
          (dolist (dt (cons nil *all-dtypes*))
            (dolist (fast '(nil t))
              ;; FAST 模式不做安全转换：源值必须能直接 COERCE，否则两边
              ;; 抛的不是同一种条件。只在「源类型 == 目标类型」时比 FAST。
              (when (or (not fast)
                        (and dt (equal etype (vt-dtype->lisp-type dt))))
                (incf cases)
                (let ((new (if dt (vt-from-array arr :dtype dt :fast fast)
                               (vt-from-array arr :fast fast)))
                      (old (if dt (%from-array-legacy arr :dtype dt :fast fast)
                               (%from-array-legacy arr :fast fast))))
                  (unless (same-bits new old)
                    (incf bad)
                    (when (<= bad 5)
                      (format t "      不一致：源 ~a → dtype ~a fast ~a~%"
                              etype (or dt '推断) fast))))))))))
    (check (format nil "from-array 全部用例逐位一致（~d 例）" cases) (zerop bad)
           (format nil "~d 例不一致" bad)))
  ;; 形状：0 长、0 维、2 维、非 0 offset 无关（入口总是 offset 0）
  (format t "~%[A2] vt-from-array 形状与 dtype 推断~%")
  (dolist (case (list (list '(0) (make-array 0 :element-type 'double-float))
                      (list nil (make-array nil :element-type 'double-float
                                                :initial-element 2.5d0))
                      (list '(2 3) (make-array '(2 3) :element-type 'single-float
                                                      :initial-element 1.0))))
    (destructuring-bind (shape arr) case
      (let ((v (vt-from-array arr)))
        (check (format nil "形状 ~a 保持（推断 dtype ~a）" shape (vt-dtype v))
               (and (equal (vt-shape v) shape)
                    (eq (vt-dtype v) (if (equal (array-element-type arr) 'single-float)
                                         :float32 :float64)))
               (format nil "得到 ~a / ~a" (vt-shape v) (vt-dtype v))))))
  (check "元素类型 T 且含复数 → :complex128（不丢虚部）"
         (eq (vt-dtype (vt-from-array (vector (complex 1d0 2d0)))) :complex128))
  (check "元素类型 T 且全为实数 → :float64（既有默认）"
         (eq (vt-dtype (vt-from-array (vector 1d0 2d0))) :float64))
  ;; 布尔目标必须真值化（REPLACE 的字节拷贝会把 2 原样留下来）
  (check "u8 → :bool 真值化（2 → 1，不是字节拷贝）"
         (equal (vt-to-list (vt-from-array (cl-array-of '(unsigned-byte 8) '(2 0 5))
                                           :dtype :bool))
                '(1 0 1)))
  (check "u8 数组 → :bool 后物理字节只有 0/1（§17.1 不变量）"
         (let ((d (vt-data (vt-from-array (cl-array-of '(unsigned-byte 8) '(2 9 255))
                                          :dtype :bool))))
           (and (= (aref d 0) 1) (= (aref d 1) 1) (= (aref d 2) 1))))
  ;; 显式 dtype 与推断 dtype 一致性
  (check "显式 :dtype 覆盖推断"
         (eq (vt-dtype (vt-from-array (cl-array-of 'double-float '(1d0)) :dtype :int32))
             :int32)))

;;; ==================================================================
;;; 3. VT-ARRAY-EQUAL：差分 + 错误回归
;;; ==================================================================
(defun view-forms (m)
  "同一**逻辑张量** M 的 5 种物理形态（内容相同，布局不同）：
   连续 / 双转置 / 双翻转 / 带偏移子块 / 步长 2 切片。
   注意：转置只做一次就换了逻辑内容，所以这里成对施加（转两次、翻两次）。"
  (let* ((sh (vt-shape m))
         (r (first sh))
         (c (second sh))
         (dt (vt-dtype m)))
    (let* ((padded (vt-zeros (list (+ r 2) (+ c 2)) :dtype dt))
           (pview (vt-slice padded (list 1 (1+ r)) (list 1 (1+ c))))
           (strided (vt-zeros (list r (* 2 c)) :dtype dt))
           (sview (vt-slice strided '(:all) (list 0 (* 2 c) 2))))
      (vt-copy-into m pview)
      (vt-copy-into m sview)
      (list (cons "连续" (vt-contiguous m))
            (cons "双转置" (vt-transpose (vt-transpose m)))
            (cons "双翻转" (vt-flip (vt-flip m :axis 0) :axis 0))
            (cons "偏移子块" pview)
            (cons "步长切片" sview)))))

(defun run-array-equal-tests ()
  (format t "~&~%[B1] vt-array-equal 与旧实现逐值对拍（实数族，旧实现正确的那部分）~%")
  (let ((bad 0) (cases 0))
    (dolist (pair '(("f64/f64" :float64 :float64)
                    ("f32/f32" :float32 :float32)
                    ("i64/i64" :int64 :int64)
                    ("i32/i64" :int32 :int64)
                    ("i8/u8" :int8 :uint8)
                    ("u16/i16" :uint16 :int16)
                    ("bool/bool" :bool :bool)
                    ("bool/i64" :bool :int64)
                    ("f32/f64" :float32 :float64)
                    ("i64/f64" :int64 :float64)))
      (destructuring-bind (label dt1 dt2) pair
        (let* ((x (mk (val-list dt1 12) dt1 '(3 4)))
               (y (mk (val-list dt2 12) dt2 '(3 4)))
               (z (vt-zeros '(3 4) :dtype dt2)))
          ;; 相同值（NaN / ±0.0 / 大整数都在池里）
          (incf cases)
          (unless (eq (vt-array-equal x y) (%array-equal-legacy x y))
            (incf bad)
            (format t "      不一致：~a（相同值）~%" label))
          ;; 不同值（0 填充）
          (incf cases)
          (unless (eq (vt-array-equal x z) (%array-equal-legacy x z))
            (incf bad)
            (format t "      不一致：~a（与全零不同值）~%" label))
          ;; 非连续形态两侧都比一遍（新实现的 RAVEL 必须保留 offset）
          (incf cases)
          (let ((xt (vt-transpose (vt-reshape x '(4 3))))
                (yt (vt-transpose (vt-reshape y '(4 3)))))
            (unless (eq (vt-array-equal xt yt) (%array-equal-legacy xt yt))
              (incf bad)
              (format t "      不一致：~a（转置视图）~%" label))))))
    (check (format nil "array-equal 实数族 ~d 例与旧实现一致" cases) (zerop bad)
           (format nil "~d 例不一致" bad)))

  (format t "~%[B2] 旧实现的两处静默错值（期望值取自 numpy 2.4.6）~%")
  (check "复数虚部不同 → NIL（numpy False；旧实现给 T）"
         (not (vt-array-equal (vt-from-sequence (vector (complex 1d0 2d0)) :dtype :complex128)
                              (vt-from-sequence (vector (complex 1d0 3d0)) :dtype :complex128))))
  (check "复数完全相同 → T"
         (vt-array-equal (vt-from-sequence (vector (complex 1d0 2d0)) :dtype :complex128)
                         (vt-from-sequence (vector (complex 1d0 2d0)) :dtype :complex128)))
  (check "int64 2^54+1 vs 2^54+2 → NIL（numpy False；旧实现给 T）"
         (not (vt-array-equal (vt-from-sequence (vector 18014398509481985) :dtype :int64)
                              (vt-from-sequence (vector 18014398509481986) :dtype :int64))))
  (check "int64 2^54+1 vs 2^54+3 → NIL"
         (not (vt-array-equal (vt-from-sequence (vector 18014398509481985) :dtype :int64)
                              (vt-from-sequence (vector 18014398509481987) :dtype :int64))))
  (check "int64 2^63-1 vs 自身 → T（大整数不经过 double）"
         (let ((v (vt-from-sequence (vector 9223372036854775807) :dtype :int64)))
           (vt-array-equal v v)))

  (format t "~%[B3] numpy 语义（array_equal）逐条钉死~%")
  (check "±0.0 数值相等 → T（numpy True；不能用 EQL）"
         (vt-array-equal (vt-from-sequence (vector 0d0)) (vt-from-sequence (vector -0d0))))
  (check "float32 的 -0.0 与 float64 的 0.0 → T"
         (vt-array-equal (vt-from-sequence (vector -0.0) :dtype :float32)
                         (vt-from-sequence (vector 0d0) :dtype :float64)))
  (check "NaN 恒不等 → NIL"
         (not (vt-array-equal (vt-from-sequence (vector *nan*))
                              (vt-from-sequence (vector *nan*)))))
  (check "空张量 → T（shape 相同、无元素）"
         (vt-array-equal (vt-zeros '(0)) (vt-zeros '(0))))
  (check "0 维 → T"
         (vt-array-equal (vt-zeros nil) (vt-zeros nil)))
  (check "0 维 1.0 与 1 维 (1) 形状不同 → NIL"
         (not (vt-array-equal (vt-from-sequence (vector 1d0) :dtype :float64)
                              (vt-zeros nil :dtype :float64))))
  (check "形状不同 → NIL"
         (not (vt-array-equal (vt-zeros '(2 3)) (vt-zeros '(3 2)))))
  (check "bool 与 int64 1 → T（numpy True）"
         (vt-array-equal (vt-from-sequence (vector t) :dtype :bool)
                         (vt-from-sequence (vector 1) :dtype :int64)))
  (check "bool T 与 NIL → NIL"
         (not (vt-array-equal (vt-from-sequence (vector t) :dtype :bool)
                              (vt-from-sequence (vector nil) :dtype :bool))))
  (check "complex64 带虚部 vs float32 → NIL（numpy False）"
         (not (vt-array-equal (vt-from-sequence (vector (complex 1.0 2.0)) :dtype :complex64)
                              (vt-from-sequence (vector 1.0) :dtype :float32))))
  (check "complex64 虚部为 0 vs float32 1.0 → T"
         (vt-array-equal (vt-from-sequence (vector (complex 1.0 0.0)) :dtype :complex64)
                         (vt-from-sequence (vector 1.0) :dtype :float32)))
  (check "int8 -1 vs uint8 255 → NIL（提升到 int16 后 255 ≠ -1）"
         (not (vt-array-equal (vt-from-sequence (vector -1) :dtype :int8)
                              (vt-from-sequence (vector 255) :dtype :uint8))))
  (check "uint16 65535 vs int32 65535 → T"
         (vt-array-equal (vt-from-sequence (vector 65535) :dtype :uint16)
                         (vt-from-sequence (vector 65535) :dtype :int32)))

  (format t "~%[B4] 五种物理形态（同一逻辑张量）~%")
  (dolist (pair '(("实数" . :float64) ("复数" . :complex128)))
    (destructuring-bind (label . dt) pair
      (let* ((m (mk (val-list dt 16) dt '(4 4)))
             (forms (view-forms m)))
        ;; 自洽：每种形态必须与连续形态判等
        (dolist (f forms)
          (check (format nil "~a：~a vs 连续" label (car f))
                 (vt-array-equal (cdr f) (cdr (first forms)))))
        ;; 两两互相判等（转置/翻转成对施加，内容不变）
        (dolist (a forms)
          (dolist (b forms)
            (unless (eq a b)
              (check (format nil "~a：~a vs ~a" label (car a) (car b))
                     (vt-array-equal (cdr a) (cdr b)))))))))
  (let* ((m (mk (val-list :float64 16) :float64 '(4 4))))
    (check "内容不同 → NIL"
           (not (vt-array-equal m (vt-zeros '(4 4) :dtype :float64))))))

;;; ==================================================================
;;; 4. VT-APPLY-ALONG-AXIS：差分
;;; ==================================================================
(defun run-apply-along-axis-tests ()
  (format t "~&~%[C1] vt-apply-along-axis 与旧实现逐值对拍~%")
  (dolist (dt '(:float64 :float32 :int32 :int64 :uint8 :complex128))
    (dolist (shape '(("(12 9) 轴0" (12 9) 0) ("(12 9) 轴1" (12 9) 1)
                     ("(4 5 6) 轴1" (4 5 6) 1) ("(4 5 6) 轴2" (4 5 6) 2)
                     ("(7 1) 轴0" (7 1) 0)))
      (destructuring-bind (label sh ax) shape
        (let* ((n (reduce #'* sh))
               (v (mk (val-list dt n) dt sh)))
          ;; 标量返回
          (check (format nil "~a ~a sum（标量返回）" dt label)
                 (same-bits (vt-apply-along-axis #'vt-sum ax v)
                            (%apply-along-axis-legacy #'vt-sum ax v)))
          ;; 1D 返回
          (check (format nil "~a ~a copy（1D 返回）" dt label)
                 (same-bits (vt-apply-along-axis #'vt-copy ax v)
                            (%apply-along-axis-legacy #'vt-copy ax v)))))))
  (format t "~%[C2] 非连续输入（转置 / 列视图 / 翻转）~%")
  (let* ((m (mk (val-list :float64 60) :float64 '(6 10)))
         (views (list (cons "转置" (vt-transpose m))
                      (cons "列视图" (vt-slice m '(:all) '(nil nil 2)))
                      (cons "翻转" (vt-flip m :axis 0)))))
    (dolist (v views)
      (check (format nil "~a 输入：copy 与旧实现一致" (car v))
             (same-bits (vt-apply-along-axis #'vt-copy 0 (cdr v))
                        (%apply-along-axis-legacy #'vt-copy 0 (cdr v))))
      (check (format nil "~a 输入：sum 与旧实现一致" (car v))
             (same-bits (vt-apply-along-axis #'vt-sum 1 (cdr v))
                        (%apply-along-axis-legacy #'vt-sum 1 (cdr v))))))
  (format t "~%[C3] 0 长度轴与「函数返回视图」~%")
  (let ((z (vt-zeros '(0 5))))
    (check "0 长度轴 → 空结果（不崩）"
           (equal (vt-shape (vt-apply-along-axis #'vt-copy 0 z)) '(0 5)))
    (check "0 长度轴 sum → 形状 (5)"
           (equal (vt-shape (vt-apply-along-axis #'vt-sum 0 z)) '(5))))
  (let* ((m (mk (val-list :float64 24) :float64 '(4 6)))
         (rev (lambda (row) (vt-flip row :axis 0))))
    (check "func 返回非连续视图（flip）与旧实现一致"
           (same-bits (vt-apply-along-axis rev 0 m)
                      (%apply-along-axis-legacy rev 0 m))))
  (let* ((m (mk (val-list :float64 24) :float64 '(4 6))))
    (check "布尔输入：sum 结果按 FUNC 返回 dtype（§17.10.13）"
           (let ((b (vt-astype m :bool)))
             (eq (vt-dtype (vt-apply-along-axis #'vt-sum 1 b)) :int64)))))

;;; ==================================================================
;;; 5. VT-OUTER：与旧实现（einsum）对拍
;;; ==================================================================
(defun outer-pool (dtype n)
  "外积对拍用的值池：**不含 ±0.0**，也不含会溢出的极值。
   这两类不是「等价性」能覆盖的：einsum 的内核是「累加器 + 乘积」（`0 + x*y`），
   符号零在累加时会被加成正零；整数溢出则因为写回不做回绕而直接抛类型错误。
   两条都由 [D2] 单独按 numpy 的答案钉死（见 §7.4.13）。"
  (let ((pool (ecase dtype
                (:float64 (list 1d0 2d0 -3d0 0.5d0 1d200))
                (:float32 (list 1.0 2.0 -3.0 0.5))
                (:int64 (list 1 2 -3 7))
                (:int32 (list 1 2 -3 7))
                (:int8 (list 1 2 -3 7))
                (:complex128 (list (complex 1d0 2d0) (complex -1d0 3d0)))
                (:complex64 (list (complex 1.0 2.0) (complex -1.0 3.0)))
                (:bool (list t nil t t)))))
    (loop for i below n collect (nth (mod i (length pool)) pool))))

(defun run-outer-tests ()
  (format t "~&~%[D1] vt-outer 与 einsum 旧路径逐位对拍（无 ±0 / 无溢出的值域）~%")
  (dolist (dt '(:float64 :float32 :int32 :int64 :complex128 :complex64 :bool))
    (let* ((n 7) (m 5)
           (a (mk (outer-pool dt n) dt (list n)))
           (b (mk (outer-pool dt m) dt (list m))))
      (check (format nil "~a 1D×1D" dt)
             (same-bits (vt-outer a b) (vt-einsum "i,j->ij" a b)))
      ;; 复数走 %CX-OP-OUTER（复数语义层），不经过这里的分支
      (unless (member dt '(:complex128 :complex64))
        (check (format nil "~a 显式 :dtype 仍走 EINSUM（计算 dtype 语义）" dt)
               (same-bits (vt-outer a b :dtype :float64)
                          (vt-einsum "i,j->ij" a b :dtype :float64))))
      (check (format nil "~a :out 复用" dt)
             (let ((o (vt-zeros (list n m) :dtype dt))
                   (o2 (vt-zeros (list n m) :dtype dt)))
               (vt-outer a b :out o)
               (vt-einsum "i,j->ij" a b :out o2)
               (same-bits o o2)))))
  (format t "~%[D2] 广播乘法的两条「比 einsum 更贴 numpy」的差异（期望取自 numpy 2.4.6）~%")
  (check "0.0 × -0.0 → -0.0（numpy array([[-0.]]);einsum 给 +0.0）"
         (let* ((x (vt-outer (vt-from-sequence (vector 0d0))
                             (vt-from-sequence (vector -0d0))))
                (v (aref (vt-data x) 0)))
           (and (zerop v) (minusp (float-sign v)))))
  (check "int32 溢出按 int32 回绕：max×max → 1（numpy 1；einsum 抛类型错误）"
         (let* ((mx (vt-from-sequence (vector 2147483647) :dtype :int32))
                (r (vt-outer mx mx)))
           (and (eq (vt-dtype r) :int32) (= (aref (vt-data r) 0) 1))))
  (check "int8 外积保持 int8 并回绕（numpy int8；einsum 不支持窄整型 out）"
         (let* ((a (mk (list 100) :int8 '(1)))
                (b (mk (list 100) :int8 '(1)))
                (r (vt-outer a b)))
           (and (eq (vt-dtype r) :int8) (= (aref (vt-data r) 0) 16))))
  (check "布尔外积是 AND（numpy True/False 与 :bool dtype）"
         (let* ((a (vt-from-sequence (vector t nil) :dtype :bool))
                (b (vt-from-sequence (vector t t) :dtype :bool))
                (r (vt-outer a b)))
           (and (eq (vt-dtype r) :bool)
                (= (aref (vt-data r) 0) 1) (= (aref (vt-data r) 1) 1)
                (= (aref (vt-data r) 2) 0) (= (aref (vt-data r) 3) 0))))
  ;; 形状族
  (let* ((a (mk (outer-pool :float64 6) :float64 '(2 3)))
         (b (mk (outer-pool :float64 4) :float64 '(4)))
         (c (vt-from-sequence (vector 3d0) :dtype :float64)))
    (check "2D × 1D：按 flatten=真 展平"
           (same-bits (vt-outer a b)
                      (vt-einsum "i,j->ij" (vt-ravel a) b)))
    (check "0 维 × 1D → (1 m)"
           (equal (vt-shape (vt-outer c b)) '(1 4)))
    (check "0 长度 → (0 m) 且不报错"
           (equal (vt-shape (vt-outer (vt-zeros '(0)) b)) '(0 4)))
    (check "flatten nil（高维外积）路径未改动"
           (equal (vt-shape (vt-outer a b :flatten nil)) '(2 3 4)))))

;;; ==================================================================
;;; 6. 倍率护栏（机器无关）
;;; ==================================================================
(defun run-ratio-guards ()
  (format t "~&~%[E1] 相等判定 ÷ isclose（同类逐元素判定的锚点）~%")
  (let* ((v (vt-random-uniform 1000000))
         (t-eq (best-ms-per (lambda () (vt-array-equal v v))))
         (t-ic (best-ms-per (lambda () (vt-isclose v v)))))
    ;; 改前：83 ms ÷ 5 ms ≈ 16×；现在约 0.4×。护栏 2.0 留 5 倍余量。
    (check-at-most "array-equal(1M) ÷ isclose(1M)" (/ t-eq (max t-ic 1e-9)) 2.0 "耗时比"))

  (format t "~%[E2] from-array ÷ 旧实现（同一源数组，同 dtype）~%")
  (let* ((arr (make-array 300000 :element-type 'double-float :initial-element 0.5d0))
         (t-new (best-ms-per (lambda () (vt-from-array arr))))
         (t-old (best-ms-per (lambda () (%from-array-legacy arr)))))
    ;; 改前两者是同一个函数；现在约 0.02×。护栏 0.25 表示「至少快 4 倍」。
    (check-at-most "from-array(300k, 同类型) ÷ 旧实现" (/ t-new (max t-old 1e-9)) 0.25 "耗时比"))

  (format t "~%[E3] outer ÷ einsum（旧路径）~%")
  (let* ((a (vt-random-uniform 1000))
         (b (vt-random-uniform 1000))
         (t-new (best-ms-per (lambda () (vt-outer a b))))
         (t-old (best-ms-per (lambda () (vt-einsum "i,j->ij" a b)))))
    ;; 改前两者是同一个函数；现在约 0.11×。
    (check-at-most "outer(1000×1000) ÷ einsum" (/ t-new (max t-old 1e-9)) 0.5 "耗时比"))

  (format t "~%[E4] apply-along-axis ÷ 同访问模式的手写地板~%")
  (let* ((n 1000)
         (m (vt-random-uniform (list n n)))
         (data (vt-data m))
         (floor-ms
           (best-ms-per
            (lambda ()
              (let ((acc 0))
                (dotimes (i n)
                  (let ((buf (make-array n :element-type 'double-float)))
                    (declare (type (simple-array double-float (*)) buf))
                    (let ((p i))
                      (declare (type fixnum p))
                      (dotimes (k n)
                        (declare (type fixnum k))
                        (setf (aref buf k) (aref (the (simple-array double-float (*)) data) p))
                        (incf p n)))
                    (incf acc (vt-size (vt-copy (vt-from-array buf :dtype :float64 :fast t))))))
                acc))))
         (t-apply (best-ms-per (lambda () (vt-apply-along-axis #'vt-copy 0 m)))))
    ;; 地板 = 同样的列访问 + 同样的 1000 次 fiber 构造/拷贝。
    ;; 改前 apply 是 644 ms（护栏会读到 ~50×）；现在约 0.4×。护栏 3.0。
    (check-at-most "apply-along-axis(1000×1000) ÷ 手写地板"
                   (/ t-apply (max floor-ms 1e-9)) 3.0 "耗时比")))

;;; ==================================================================
;;; 7. 入口
;;; ==================================================================
(defun run ()
  (format t "~&===== entry-equal-apply-test =====~%")
  (run-from-array-tests)
  (run-array-equal-tests)
  (run-apply-along-axis-tests)
  (run-outer-tests)
  (run-ratio-guards)
  (format t "~&~%通过 ~d / ~d~%" *pass* *n*)
  (when (plusp *fail*)
    (format t "失败 ~d 条：~%~{  - ~a~%~}" *fail* (reverse *failures*))
    (error "entry-equal-apply-test 失败 ~d 条" *fail*))
  *pass*)
