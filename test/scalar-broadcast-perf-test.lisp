;;;; scalar-broadcast-perf-test.lisp — 广播标量在比较 / 判定 / 裁剪三处的执行层回归
;;;;
;;;; 三处的病根是同一句话：**广播标量被当成「类型不匹配」或「形状不匹配」**。
;;;;
;;;;   1. src/map-reduce.lisp    %MASK-ALIGN（比较族）
;;;;      比较算子的结果 dtype 是 :bool，而 %WEAK-CAST1 的判据是「种类秩
;;;;      源 ≤ 目标」；:bool 的秩是 -1，于是 f32 标量对 f64 数组永远对不齐。
;;;;      掩码内核按 (in-lt, in-lt) 成对派发，类型不齐就掉进 vt-map 的
;;;;      funcall 兜底：`(vt-< f64数组 0.5)`（0.5 是单精度字面量）69 ms，
;;;;      同一个式子写 0.5d0 是 4 ms。
;;;;
;;;;   2. src/reduce-stats.lisp VT-ISCLOSE
;;;;      特化内核的前置条件写的是 `(equal (vt-shape b) out-shape)`，0 维
;;;;      标量天然不满足 → 684 ms（numpy.isclose 4 ms）。内核本来就支持
;;;;      SIZE=1 的 stride 0 读，前置条件收紧过头了。
;;;;
;;;;   3. src/elementwise/extensions  VT-CLIP / VT-CLIP-TENSOR
;;;;      闭包交给 vt-map：每元素一次 funcall + 泛型 aref，41 ms / 14 ms。
;;;;      改具名 inline 算子走三元内联循环。
;;;;
;;;; 语义不变由两块守：A 段与**逐字复刻的改动前实现**逐位对拍（位模式比较，
;;;; 不用 =，因为 ±0 与 NaN 载荷都要看住），B 段钉 numpy 2.4.6 的答案
;;;; （NaN / ±Inf / min>max 的夹取顺序），C 段是机器无关的倍率护栏。

(require :asdf)
#+quicklisp (ql:quickload :clvt :silent t)
(asdf:load-system :clvt)

(defpackage :clvt-scalar-test
  (:use :cl :clvt)
  (:import-from :clvt
                #:ensure-vt #:vt-data #:vt-shape #:vt-dtype #:vt-size
                #:vt-offset #:vt-strides #:%op-eq #:%op-ne #:%op-lt
                #:%op-le #:%op-gt #:%op-ge #:vt-map)
  (:export #:run))

(in-package :clvt-scalar-test)

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
      (progn (incf *fail*)
             (push (format nil "~a~@[ — ~a~]" label detail) *failures*)
             (format t "  [FAIL] ~a~@[ — ~a~]~%" label detail))))

(defun check-at-most (label ratio limit what)
  (check (format nil "~a（~a = ~,2f× ≤ ~,2f×）" label what ratio limit)
         (<= ratio limit)
         (format nil "实际 ~,3f×，上限 ~,2f×" ratio limit)))

(defparameter *pinf* sb-ext:double-float-positive-infinity)
(defparameter *ninf* (- sb-ext:double-float-positive-infinity))
(defparameter *nan* (sb-int:with-float-traps-masked (:invalid) (/ 0d0 0d0)))

(defun best-ms-per (thunk &key (reps 3) (target-ms 20.0))
  "THUNK 单次调用的平均耗时（毫秒）。取 REPS 批里最快的一批。

   时钟刻度 1 ms：给 4 ms 的操作单次计时只有 ±25%，再叠上分配与 GC 的抖动，
   倍率护栏就成了随机数。这里按「一批 ≈ TARGET-MS」自适应定迭代次数。"
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
   浮点用 EQL（SBCL 对浮点是位比较：0.0 与 -0.0 不等、同载荷 NaN 相等）。"
  (and (equal (vt-shape x) (vt-shape y))
       (eq (vt-dtype x) (vt-dtype y))
       (let* ((n (vt-size x))
              (xd (vt-data x)) (yd (vt-data y))
              (xo (vt-offset x)) (yo (vt-offset y)))
         (loop for i below n
               always (eql (aref xd (+ xo i)) (aref yd (+ yo i)))))))

(defun mk (values shape &optional (dtype :float64))
  "把扁平值列表按 SHAPE 摆成张量（vt-from-sequence 没有 :shape 参数）。"
  (vt-reshape (vt-from-sequence values :dtype dtype) shape))

(defun nan-p* (x) (and (floatp x) (not (= x x))))

;;; ==================================================================
;;; 1. 改动前的逐字复刻（对拍基准）
;;; ==================================================================
(defun legacy-compare (op a b)
  "改动前的比较族兜底路径：`(vt-map (function OP) a b :dtype :bool)`。
   比较算子的结果 dtype 是 :bool，而掩码内核按 (in-lt, in-lt) 派发，
   两个输入的元素类型不同时走的就是这一条。"
  (vt-map (symbol-function op) (ensure-vt a) (ensure-vt b) :dtype :bool))

(defun legacy-isclose (a b &key (rtol 1e-5) (atol 1e-8))
  "VT-ISCLOSE 改动前的兜底闭包（逐字复刻，含 NaN / ±Inf 两条短路）。"
  (vt-map (lambda (x y)
            (let ((fa (clvt::vt-cast x :float64)) (fb (clvt::vt-cast y :float64)))
              (cond ((or (nan-p* fa) (nan-p* fb)) 0)
                    ((or (= fa *pinf*) (= fa *ninf*) (= fb *pinf*) (= fb *ninf*))
                     (if (= fa fb) 1 0))
                    (t (if (<= (abs (- fa fb))
                               (+ atol (* rtol (abs fb))))
                           1 0)))))
          (ensure-vt a) (ensure-vt b) :dtype :bool))

(defun legacy-clip (tensor lo hi)
  "VT-CLIP 双侧裁剪的语义（= `%OP-CLIP`）：**先换下限、再比上限**，
   即 `(let ((x (if (< x l) l x))) (if (> x h) h x))`。
   min>max 时以 max 收尾（numpy 同），与 VT-CLIP-TENSOR 的
   「先比上限」顺序相反 —— 两者在 min>max 时结果不同，所以不能共用复刻。
   `vt-map` 的 dtype 交给提升规则（f32 输入 → f32 输出），与 vt-fast-map 同。"
  (vt-map (lambda (x l h)
            (let ((x (if (< x l) l x)))
              (if (> x h) h x)))
          (ensure-vt tensor) lo hi))

(defun legacy-clip3 (tensor lo hi)
  "VT-CLIP-TENSOR 改动前的实现（匿名 lambda + vt-map）。"
  (vt-map (lambda (x l h)
            (if (> x h) h (if (< x l) l x)))
          (ensure-vt tensor) (ensure-vt lo) (ensure-vt hi)))

;;; ==================================================================
;;; 2. A 段：比较族 + 标量，与改动前逐位对拍
;;; ==================================================================
(defun run-compare-tests ()
  (format t "~&~%[A] 比较族 × 广播标量：与改动前逐位对拍~%")
  (let* ((vals (list -1.5d0 -1d0 -0d0 0d0 1d0 1.5d0 2d0 *pinf* *ninf* *nan* 1d-300))
         (f64 (vt-from-sequence vals))
         (f32 (vt-from-sequence vals :dtype :float32))
         (i64 (vt-from-sequence '(-2 -1 0 1 2 3) :dtype :int64))
         (scalars (list 0.5 0.5d0 1 1d0 -1 2.5 -0.0)))
    ;; A1：f64 数组 × 各种标量（含单精度字面量 —— 这就是被对齐的那条）
    (dolist (op '(clvt::%op-eq clvt::%op-ne clvt::%op-lt clvt::%op-le clvt::%op-gt clvt::%op-ge))
      (dolist (s scalars)
        (let* ((name (string-downcase (symbol-name op)))
               (new (funcall (ecase op
                               (clvt::%op-eq (lambda (a b) (vt-= a b)))
                               (clvt::%op-ne (lambda (a b) (vt-/= a b)))
                               (clvt::%op-lt (lambda (a b) (vt-< a b)))
                               (clvt::%op-le (lambda (a b) (vt-<= a b)))
                               (clvt::%op-gt (lambda (a b) (vt-> a b)))
                               (clvt::%op-ge (lambda (a b) (vt->= a b))))
                             f64 s))
               (old (legacy-compare op f64 s)))
          (check (format nil "~a(f64, ~s) ≡ 改动前" name s) (same-bits new old)))))
    ;; A2：f32 数组 × 标量（另一侧对齐方向）
    (dolist (s '(0.5 0.5d0 1d0))
      (check (format nil "vt-<(f32, ~s) ≡ 改动前" s)
             (same-bits (vt-< f32 s) (legacy-compare 'clvt::%op-lt f32 s))))
    ;; A3：整数数组 × 整数/浮点标量
    (dolist (s '(1 1d0 0.5 -1))
      (check (format nil "vt-<(i64, ~s) ≡ 改动前" s)
             (same-bits (vt-< i64 s) (legacy-compare 'clvt::%op-lt i64 s))))
    ;; A4：物理形态（转置 / 翻转 / 步进切片）× 标量
    (let* ((m (mk (loop for i below 12 collect (coerce i 'double-float)) '(3 4)))
           (views (list m (vt-transpose m) (vt-flip m) (vt-narrow m 1 1 3))))
      (dolist (v views)
        (check (format nil "vt-<(~a 视图, 0.5) ≡ 改动前" (vt-shape v))
               (same-bits (vt-< v 0.5) (legacy-compare 'clvt::%op-lt v 0.5)))))
    ;; A5：显式 :int8 掩码（旧的整数掩码口径也要跟着对齐）
    (check "vt-=(f64, 0.5, :dtype :int8) ≡ 改动前"
           (same-bits (vt-= f64 0.5 :dtype :int8)
                      (vt-map #'%op-eq f64 0.5 :dtype :int8)))
    ;; A6：0 维 × 标量、标量 × 标量（不能因为 SIZE=1 判据而改变）
    (let ((z (vt-from-sequence '(2.5))))
      (check "0 维 × 标量"
             (equal (vt-to-list (vt-< z 3)) (list 1)))
      (check "标量 × 标量（两侧都 SIZE=1）"
             (eq t (vt-item (vt-< 1 2)))
             (format nil "实际 ~s" (vt-item (vt-< 1 2)))))
    ;; A7：:out 复用
    (let ((o (vt-zeros '(11) :dtype :bool)))
      (check "vt-<(f64, 0.5, :out) 写入指定缓冲"
             (same-bits (vt-< f64 0.5 :out o) (legacy-compare 'clvt::%op-lt f64 0.5))))))

;;; ==================================================================
;;; 3. B 段：numpy 2.4.6 钉死的语义点
;;; ==================================================================
(defun run-numpy-pins ()
  (format t "~&~%[B] numpy 2.4.6 钉死的语义点~%")
  ;; 比较：NaN 全假、±0 相等
  (let ((v (vt-from-sequence (list *nan* 0d0 -0d0 1d0))))
    (check "NaN == NaN → False（numpy）"
           (equal (vt-to-list (vt-= v v)) '(0 1 1 1)))
    (check "0.0 == -0.0 → True（numpy；EQL 会是 False）"
           (equal (vt-to-list (vt-= (vt-from-sequence (list 0d0))
                                    (vt-from-sequence (list -0d0))))
                  '(1)))
    (check "NaN < 1 → False；-0.0 < 1 → True"
           (equal (vt-to-list (vt-< v 1d0)) '(0 1 1 0))
           (format nil "实际 ~s" (vt-to-list (vt-< v 1d0)))))
  ;; isclose：numpy.isclose(1M…) 的四个边界
  (let* ((a (vt-from-sequence (list 1d0 1d0 *nan* *pinf* *ninf* 0d0)))
         (r (vt-isclose a (vt-from-sequence (list 1.000001d0 1.1d0 1d0 *pinf* *pinf* 0d0)))))
    ;; numpy: [True True False True False True]
    (check "isclose 标量广播：容差内 / 容差外 / NaN / +Inf / -Inf vs +Inf / 0"
           (equal (vt-to-list r) '(1 0 0 1 0 1))
           (format nil "实际 ~s（numpy 给 [True False False True False True]）"
                   (vt-to-list r))))
  (check "allclose(a, a+1e-9)"
         (vt-allclose (vt-from-sequence (list 1d0 2d0))
                      (vt-from-sequence (list 1.0000000001d0 2.0000000001d0))))
  (check "allclose 含 NaN → NIL（numpy equal_nan=False）"
         (not (vt-allclose (vt-from-sequence (list *nan*))
                           (vt-from-sequence (list *nan*)))))
  (check "isclose 0 维 × 0 维"
         (eq t (vt-item (vt-isclose 1d0 1d0))))
  ;; clip：min>max 时 numpy 以 max 收尾；NaN 原样穿过
  (let ((v (vt-from-sequence (list 1d0 3d0 5d0))))
    (check "clip(v, 2, 4) → 2 3 4"
           (equal (vt-to-list (vt-clip v 2d0 4d0)) '(2.0d0 3.0d0 4.0d0)))
    (check "clip(v, 4, 2) → 2 2 2（numpy 以 max 收尾）"
           (equal (vt-to-list (vt-clip v 4d0 2d0)) '(2.0d0 2.0d0 2.0d0)))
    (check "clip 仅上限 / 仅下限"
           (and (equal (vt-to-list (vt-clip v nil 2d0)) '(1.0d0 2.0d0 2.0d0))
                (equal (vt-to-list (vt-clip v 2d0 nil)) '(2.0d0 3.0d0 5.0d0))))
    (check "clip(NaN, 2, 4) → NaN（numpy）"
           (let ((r (vt-to-list (vt-clip (vt-from-sequence (list *nan*)) 2d0 4d0))))
             (nan-p* (first r))))
    (check "clip 保 ±0.0 位模式（-0.0 原样留下）"
           (eql (first (vt-to-list (vt-clip (vt-from-sequence (list -0d0)) 0d0 1d0)))
                -0d0))))

;;; ==================================================================
;;; 4. C 段：isclose / clip 与改动前逐位对拍
;;; ==================================================================
(defun run-legacy-equiv ()
  (format t "~&~%[C] isclose / clip：与改动前逐位对拍~%")
  (let* ((vals (list -1.5d0 -0d0 0d0 1d0 1.000001d0 *pinf* *ninf* *nan*))
         (v (vt-from-sequence vals)))
    (dolist (s (list 0.5 0.5d0 1d0 2.5 -1))
      (check (format nil "isclose(v, ~s) ≡ 改动前" s)
             (same-bits (vt-isclose v s) (legacy-isclose v s))))
    (dolist (s (list 0.5d0 1d0))
      (check (format nil "isclose(~s, v) ≡ 改动前（标量在左）" s)
             (same-bits (vt-isclose s v) (legacy-isclose s v))))
    (check "isclose(v, v) ≡ 改动前（数组 × 数组仍走内核）"
           (same-bits (vt-isclose v v) (legacy-isclose v v)))
    (check "isclose 自定义 rtol/atol ≡ 改动前"
           (same-bits (vt-isclose v 1d0 :rtol 0.5d0 :atol 0.25d0)
                      (legacy-isclose v 1d0 :rtol 0.5d0 :atol 0.25d0)))
    (check "isclose 0 维 × 0 维 ≡ 改动前"
           (same-bits (vt-isclose 1d0 1d0) (legacy-isclose 1d0 1d0))))
  ;; clip / clip-tensor 与改动前对拍（含 min>max、±0、NaN、单精度）
  (let* ((v (vt-from-sequence (list -1d0 -0d0 0d0 0.5d0 1d0 2d0 *nan*)))
         (v32 (vt-from-sequence (list -1d0 0d0 1d0 2d0) :dtype :float32)))
    (dolist (pair '((0.2d0 0.8d0) (2d0 1d0) (-0.0d0 0.0d0) (0.5 0.5)))
      (destructuring-bind (lo hi) pair
        (check (format nil "vt-clip(v, ~s, ~s) ≡ 改动前" lo hi)
               (same-bits (vt-clip v lo hi) (legacy-clip v lo hi)))
        (check (format nil "vt-clip-tensor(v, ~s, ~s) ≡ 改动前" lo hi)
               (same-bits (vt-clip-tensor v lo hi) (legacy-clip3 v lo hi)))))
    (check "vt-clip(f32, 0.2, 0.8) ≡ 改动前"
           (same-bits (vt-clip v32 0.2 0.8) (legacy-clip v32 0.2 0.8)))
    (check "vt-clip 仅上限 ≡ 改动前"
           (same-bits (vt-clip v nil 0.8d0)
                      (vt-map (lambda (x h) (min x h)) v 0.8d0 :dtype :float64)))
    (check "vt-clip 仅下限 ≡ 改动前"
           (same-bits (vt-clip v 0.2d0 nil)
                      (vt-map (lambda (x l) (max x l)) v 0.2d0 :dtype :float64)))
    ;; 张量边界：三输入内联路径
    (let ((lo (vt-from-sequence (list 0d0 0d0 0d0 0d0 0d0 0d0 0d0)))
          (hi (vt-from-sequence (list 0.5d0 0.5d0 0.5d0 0.5d0 0.5d0 0.5d0 0.5d0))))
      (check "vt-clip-tensor 张量边界 ≡ 改动前"
             (same-bits (vt-clip-tensor v lo hi) (legacy-clip3 v lo hi))))
    ;; 视图
    (let* ((m (mk (loop for i below 12 collect (coerce i 'double-float)) '(3 4)))
           (tm (vt-transpose m)))
      (check "vt-clip(转置视图) ≡ 改动前"
             (same-bits (vt-clip tm 2d0 8d0) (legacy-clip tm 2d0 8d0)))
      (check "vt-clip-tensor(翻转视图) ≡ 改动前"
             (same-bits (vt-clip-tensor (vt-flip m) 2d0 8d0)
                        (legacy-clip3 (vt-flip m) 2d0 8d0))))))

;;; ==================================================================
;;; 5. D 段：倍率护栏（机器无关）
;;; ==================================================================
(defun run-ratio-guards ()
  (format t "~&~%[D] 倍率护栏（机器无关）~%")
  (let* ((a (vt-random-uniform 1000000))
         (b (vt-random-uniform 1000000)))
    ;; D1：标量档与数组档同量级。改前 69 ÷ 3.8 ≈ 18×，现在 ≈ 1.2×。
    (let ((t-s (best-ms-per (lambda () (vt-< a 0.5))))
          (t-v (best-ms-per (lambda () (vt-< a b)))))
      (check-at-most "vt-<(1M, 单精度标量) ÷ vt-<(1M, 数组)" (/ t-s (max t-v 1e-9)) 2.0 "耗时比"))
    ;; D2：isclose 的标量档 ÷ 数组档。改前 684 ÷ 5 ≈ 137×，现在 ≈ 1.0×。
    (let ((t-s (best-ms-per (lambda () (vt-isclose a 0.5d0))))
          (t-v (best-ms-per (lambda () (vt-isclose a b)))))
      (check-at-most "vt-isclose(1M, 标量) ÷ vt-isclose(1M, 数组)" (/ t-s (max t-v 1e-9)) 2.0 "耗时比"))
    ;; D3：clip 相对**改动前实现**至少快 2 倍（2 次比较 + 数据相关分支，
    ;;     随机数据上分支不可预测是固有成本，不能拿算术算子当锚点）。
    (let* ((v (vt-random-uniform (list 1000 1000)))
           (t-new (best-ms-per (lambda () (vt-clip v 0.2d0 0.8d0))))
           (t-old (best-ms-per (lambda () (legacy-clip3 v 0.2d0 0.8d0)))))
      (check-at-most "vt-clip ÷ 改动前实现" (/ t-new (max t-old 1e-9)) 0.5 "耗时比"))))

;;; ==================================================================
;;; 6. 入口
;;; ==================================================================
(defun run ()
  (format t "~&===== scalar-broadcast-perf-test =====~%")
  (run-compare-tests)
  (run-numpy-pins)
  (run-legacy-equiv)
  (run-ratio-guards)
  (format t "~&~%通过 ~d / ~d~%" *pass* *n*)
  (when (plusp *fail*)
    (format t "失败 ~d 条：~%~{  - ~a~%~}" *fail* (reverse *failures*))
    (error "scalar-broadcast-perf-test 失败 ~d 条" *fail*))
  *pass*)
