;;;; strided-ew-simd-test.lisp — 非连续逐元素路径的遍历形态 + run 级 SIMD
;;;;
;;;; 为什么需要这套：本轮把 `%inline1-strided` / `%inline2-strided` 的遍历
;;;; 从「整块 odometer 逐元素」改成「外层 odometer × 内层连续 run」，并在
;;;; 内层接了一个 run 级 SIMD 钩子。三处都可能「看起来对、其实错」：
;;;;
;;;;   1. 内层 run 的边界。原来每个元素都走一次 `(aref idx d)` 推进坐标，
;;;;      现在末轴被提出来单独成环 —— 末轴长度、末轴 stride（含**负** stride，
;;;;      即 vt-flip 的视图）、以及「末轴 dim=1 而其他输入在该轴广播」的
;;;;      组合，任何一处算错都只会错**一部分**元素，而且往往是错列/错行。
;;;;      所以全部与 `vt-map`（闭包参考，语义基准）逐位对拍。
;;;;   2. run 级 SIMD 的接入条件。它只在「三方内层 stride 都恰好是 1」时接管，
;;;;      并且只认 + - * / 与 `%sq-diff-*`、只认 f64/f32。判据多写一个等号
;;;;      就会把 stride=2 的视图也送进「连续」内核，读错列。
;;;;   3. 位级语义。向量加法与标量加法是同一台 IEEE 运算，但 `±0` 的符号、
;;;;      NaN 的载荷、Inf 的传播都必须逐位一致 —— 用 `=` 比较会漏报
;;;;      （NaN ≠ NaN，+0.0 = −0.0），所以这里一律比位模式。
;;;;
;;;; 用法：sbcl --dynamic-space-size 4096 --non-interactive \
;;;;             --load test/strided-ew-simd-test.lisp

(require :asdf)
(defparameter cl-user::*clvt-test-dir*
  (make-pathname :name nil :type nil :defaults *load-truename*))
(pushnew (merge-pathnames "../" cl-user::*clvt-test-dir*) asdf:*central-registry* :test #'equal)
(handler-bind ((warning #'muffle-warning)) (asdf:load-system :clvt))

(in-package :clvt)

(defvar *pass* 0)
(defvar *fail* 0)
(defvar *failures* '())

(defun ok (label)
  (declare (ignore label))
  (incf *pass*))

(defun ng (label fmt &rest args)
  (incf *fail*)
  (let ((msg (apply #'format nil fmt args)))
    (push (format nil "~a — ~a" label msg) *failures*)
    (format t "~&  [FAIL] ~a~%         ~a~%" label msg))
  nil)

(defun check (label test)
  (if test (ok label) (ng label "断言为假")))

;;; ------------------------------------------------------------------
;;; 逐位比较辅助
;;; ------------------------------------------------------------------

(defun df-bits (x) (sb-kernel:double-float-bits (coerce x 'double-float)))
(defun sf-bits (x) (sb-kernel:single-float-bits (coerce x 'single-float)))

(defun elt-bits= (dtype a b)
  (case dtype
    (:float64 (= (df-bits a) (df-bits b)))
    (:float32 (= (sf-bits a) (sf-bits b)))
    (t (eql a b))))

(defun flat (x)
  "把 VT-TO-LIST 的嵌套列表摊平。叶子是元素本身（0 维/单元素时 VT-TO-LIST
   直接返回标量，不是列表），所以非 cons 要包成单元素表。"
  (cond ((not (consp x)) (list x))
        ((consp (car x)) (mapcan #'flat x))
        (t x)))

(defun all-bits= (dtype xs ys)
  "逐元素逐位比较两个（可能是嵌套的）列表（NaN 也按位比）。"
  (let ((xs (flat xs)) (ys (flat ys)))
    (and (= (length xs) (length ys))
         (every (lambda (x y) (elt-bits= dtype x y)) xs ys))))

;;; ------------------------------------------------------------------
;;; 参考实现：vt-map（闭包）—— CONVENTIONS 规定的语义基准
;;; ------------------------------------------------------------------

(defun ref-binop (op a b dtype)
  "用 vt-map 的闭包路径算同一个算子，强制同一结果 dtype。"
  (vt-map (case op
            (+ (lambda (x y) (+ x y)))
            (- (lambda (x y) (- x y)))
            (* (lambda (x y) (* x y)))
            (/ (lambda (x y) (/ x y))))
          a b :dtype dtype))

;;; ------------------------------------------------------------------
;;; A. 遍历形态 × 算子 × dtype：与闭包参考逐位对拍
;;; ------------------------------------------------------------------

(defun view-cases ()
  "返回 (名字 . 视图) 的列表。每个视图都必须与非连续路径打得上照面。"
  (let* ((m (vt-reshape (vt-from-sequence
                         (loop for i below 800 collect (coerce (- (* i 0.25) 90.0) 'double-float))
                         :dtype :float64)
                        '(8 100)))
         (t3 (vt-reshape (vt-copy m) '(2 4 100))))
    (list
     (cons "contiguous (8 100)" m)
     (cons "row-slice (4 100) stride (100 1)" (vt-slice m '(0 4) '(nil nil)))
     (cons "col-slice (8 50) stride (100 2)" (vt-slice m '(nil nil) '(nil nil 2)))
     (cons "transpose (100 8) stride (1 100)" (vt-transpose m))
     (cons "flip rows (8 100) stride (-100 1)" (vt-flip m :axis 0))
     (cons "flip cols (8 100) stride (100 -1)" (vt-flip m :axis 1))
     (cons "3d slice (2 4 50) stride (400 100 2)" (vt-slice t3 '(nil nil) '(nil nil) '(nil nil 2)))
     (cons "3d transpose (2 100 4)" (vt-transpose t3 '(0 2 1)))
     (cons "1d stride 3 (200)" (vt-slice (vt-reshape (vt-copy m) '(800)) '(nil nil 3)))
     (cons "single element (1 1)" (vt-slice m '(3 4) '(7 8)))
     (cons "rank0 ()" (vt-reshape (vt-slice m '(3 4) '(7 8)) '())))))

(defun vop (op a b)
  (case op
    (+ (vt-add a b))
    (- (vt-sub a b))
    (* (vt-mul a b))
    ;; 注意是 `vt-/`（true_divide）而不是 `vt-div`（floor_divide）：
    ;; 后者对 Inf/Inf 有 floor 特判，语义与闭包参考不同源。
    (/ (vt-/ a b))))

(defvar *cases* (view-cases))

(dolist (dt '(:float64 :float32 :int64))
  ;; 整数没有「逐元素真除」：/ 会变成连除或比例，与闭包参考不同源，故跳过。
  (let ((ops (if (eq dt :int64) '(+ - *) '(+ - * /))))
    (dolist (op ops)
      (dolist (case *cases*)
        (let* ((label (format nil "A ~a ~a ~a" dt op (car case)))
               (v (vt-astype (cdr case) dt))
               (w (vt-copy v))
               (fast (vop op v w))
               (slow (ref-binop op v w dt)))
          (check label
                 (and (equal (vt-shape fast) (vt-shape slow))
                      (all-bits= dt (vt-to-list fast) (vt-to-list slow)))))))))

;;; ------------------------------------------------------------------
;;; B. 广播轴（dim=1 的输入沿末轴广播）
;;; ------------------------------------------------------------------

(let* ((row (vt-reshape (vt-from-sequence
                         (loop for i below 100 collect (coerce i 'double-float))
                         :dtype :float64)
                        '(1 100)))
       (m (vt-reshape (vt-copy (vt-astype (vt-arange 800) :float64)) '(8 100)))
       (view (vt-slice m '(1 8) '(nil nil)))  ; (7 100) strides (100 1)，非连续
       (fast (vt-add view row))
       (slow (ref-binop '+ view row :float64)))
  (check "B1 行广播 + 行切片视图（内层 run 归约到广播）"
         (all-bits= :float64 (vt-to-list fast) (vt-to-list slow)))
  ;; 反向：标量在前
  (check "B2 广播在前（标量 + 视图）"
         (all-bits= :float64 (vt-to-list (vt-add row view))
                    (vt-to-list (ref-binop '+ row view :float64))))
  ;; 单元素张量（size=1，stride 全 0）
  (let ((s (vt-slice m '(2 3) '(5 6))))
    (check "B3 size=1 张量 + 切片视图"
           (all-bits= :float64 (vt-to-list (vt-mul view s))
                      (vt-to-list (ref-binop '* view s :float64))))))

;;; ------------------------------------------------------------------
;;; C. 位级语义：±0、NaN 载荷、Inf
;;; ------------------------------------------------------------------

(let* ((xs (list 0.0d0 -0.0d0 1.0d0 -1.0d0
                 sb-ext:double-float-positive-infinity
                 sb-ext:double-float-negative-infinity
                 (sb-kernel:make-double-float #x7ff80000 0)   ; 规范 NaN
                 (sb-kernel:make-double-float #x7ff80000 12345) ; 带载荷的 NaN
                 3.5d0 -2.25d0 1.0d-300 1.0d300))
       (a (vt-from-sequence xs :dtype :float64))
       ;; 让输入变成非连续：转置 (12,) → 不行，用 sv 视图 (3 4) 转置
       (m (vt-reshape (vt-copy a) '(3 4)))
       (v (vt-transpose m)))                            ; (4 3) strides (1 4)
  (dolist (op '(+ - * /))
    (let* ((fast (case op (+ (vt-add v v)) (- (vt-sub v v))
                      (* (vt-mul v v)) (/ (vt-/ v v))))
           (slow (case op (+ (ref-binop '+ v v :float64))
                      (- (ref-binop '- v v :float64))
                      (* (ref-binop '* v v :float64))
                      (/ (ref-binop '/ v v :float64)))))
      (check (format nil "C1 转置视图 ~a：±0/NaN载荷/Inf 逐位一致" op)
             (all-bits= :float64 (vt-to-list fast) (vt-to-list slow)))))
  ;; 明确钉住几个已知点（不只依赖「与参考一致」）
  (let ((z (vt-sub (vt-from-sequence (list 0.0d0 0.0d0) :dtype :float64)
                   (vt-from-sequence (list 0.0d0 -0.0d0) :dtype :float64))))
    (check "C2 (+0.0) − (+0.0) = +0.0"
           (eql (df-bits (first (vt-to-list z))) (df-bits 0.0d0)))
    (check "C3 (+0.0) − (−0.0) = +0.0"
           (eql (df-bits (second (vt-to-list z))) (df-bits 0.0d0))))
  (let ((z (vt-mul (vt-from-sequence (list -1.0d0 -1.0d0) :dtype :float64)
                   (vt-from-sequence (list 0.0d0 -0.0d0) :dtype :float64))))
    (check "C4 −1·(+0.0) = −0.0" (eql (df-bits (first (vt-to-list z))) (df-bits -0.0d0)))
    (check "C5 −1·(−0.0) = +0.0" (eql (df-bits (second (vt-to-list z))) (df-bits 0.0d0)))))

;;; ------------------------------------------------------------------
;;; D. run 级 SIMD 钩子的接入判据
;;; ------------------------------------------------------------------

(let* ((nc (vt-astype (vt-reshape (vt-astype (vt-arange 2000000) :float64)
                                  '(1000 2000))
                      :float64))
       (rows (vt-slice nc '(nil nil 2) '(nil nil)))   ; (500 2000) stride (4000 1)
       (cols (vt-slice nc '(nil nil) '(nil nil 2)))   ; (1000 1000) stride (2000 2)
       (orig *simd-ew2-run-fn*))
  (flet ((calls (v)
           (let ((n 0))
             (let ((*simd-ew2-run-fn*
                     (lambda (&rest args)
                       (incf n)
                       (apply orig args))))
               (vt-add v v))
             n)))
    (check "D1 内层 stride=1 的切片视图 → 钩子被调用" (plusp (calls rows)))
    (check "D2 内层 stride=2 的切片视图 → 钩子不被调用" (zerop (calls cols)))
    (check "D3 连续张量 → 钩子不被调用（走扁平 SIMD）"
           (zerop (calls (vt-astype (vt-arange 100000) :float64))))
    ;; 短于阈值（+simd-run-min+ 的 32）时不接管：4 元素行
    (check "D4 内层 run 短于阈值 → 钩子不被调用"
           (zerop (calls (vt-slice nc '(nil nil) '(nil 4))))))
  ;; 关掉钩子后结果必须逐位不变
  (dolist (op '(+ - * /))
    (let* ((on (case op (+ (vt-add rows rows)) (- (vt-sub rows rows))
                    (* (vt-mul rows rows)) (/ (vt-div rows rows))))
           (off (let ((*simd-ew2-run-fn* nil))
                  (case op (+ (vt-add rows rows)) (- (vt-sub rows rows))
                        (* (vt-mul rows rows)) (/ (vt-div rows rows))))))
      (check (format nil "D5 钩子开/关 ~a 结果逐位相同" op)
             (all-bits= :float64 (vt-to-list on) (vt-to-list off))))))

;;; ------------------------------------------------------------------
;;; E. 与写入侧契约的交互：out=视图 / out=输入本身
;;; ------------------------------------------------------------------

(let* ((base (vt-astype (vt-reshape (vt-astype (vt-arange 800) :float64) '(8 100)) :float64))
       (view (vt-slice base '(0 4) '(nil nil)))        ; 非连续 out
       (b (vt-astype (vt-reshape (vt-astype (vt-arange 800) :float64) '(8 100)) :float64))
       (bv (vt-slice b '(0 4) '(nil nil)))
       ;; 参考必须在**原地写之前**取快照：:OUT 指向 VIEW 时下面那一步会
       ;; 就地改写 VIEW 的底层内存。
       (want (ref-binop '+ (vt-copy view) (vt-copy bv) :float64))
       (got (vt-add view bv :out view)))
  (check "E1 out 是非连续视图时结果正确"
         (all-bits= :float64 (vt-to-list got) (vt-to-list want))))

;;; ------------------------------------------------------------------
;;; F. 性能护栏（机器无关倍率 + 绝对上界的宽松版）
;;; ------------------------------------------------------------------

(defun now-us ()
  (multiple-value-bind (s us) (sb-ext:get-time-of-day) (+ (* s 1000000) us)))

(defun best-ms (thunk &optional (reps 15))
  (dotimes (i 3) (funcall thunk))
  (let ((best most-positive-fixnum))
    (dotimes (i reps)
      (let* ((t0 (now-us)) (v (progn (funcall thunk) (now-us))))
        (declare (ignore v))
        (let ((dt (- (now-us) t0)))
          (when (< dt best) (setf best dt)))))
    (/ best 1000.0)))

(let* ((nc (vt-astype (vt-reshape (vt-astype (vt-arange 2000000) :float64)
                                  '(1000 2000))
                      :float64))
       (rows (vt-slice nc '(nil nil 2) '(nil nil)))
       (on  (best-ms (lambda () (vt-add rows rows))))
       (off (best-ms (lambda () (let ((*simd-ew2-run-fn* nil))
                                  (vt-add rows rows))))))
  (check "F1 run 级 SIMD 至少快 1.3×（实测约 1.7×）"
         (<= on (* 0.77 off)))
  (check "F2 1M 元素行切片 add ≤ 2.0ms（实测约 1.3ms）"
         (<= on 2.0))
  (format t "~&  [info] rows add: SIMD ~,3f ms / 标量 ~,3f ms（~,2f×）~%"
          on off (/ off on)))

;;; ------------------------------------------------------------------

(format t "~&~%==== strided-ew-simd: ~d 通过, ~d 失败 ====~%" *pass* *fail*)
(when *failures*
  (format t "~&失败项：~%")
  (dolist (f (reverse *failures*)) (format t "  - ~a~%" f)))
(sb-ext:exit :code (if (zerop *fail*) 0 1))
