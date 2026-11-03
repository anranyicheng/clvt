;;;; perf-simd-ew-test.lisp — 逐元素 SIMD 快路径 / 布尔掩码内核 / 融合内核
;;;;                          的正确性 + 选路结构 + 机器无关倍率护栏
;;;;
;;;; 为什么需要这套：本轮把三条「只影响速度」的路径接进了热路径，而它们
;;;; 一旦写错，结果仍然「看起来对」——错的只是末位或某个特殊值：
;;;;
;;;;   1. `src/simd-ew.lisp` 的 f64 4 路 / f32 8 路向量循环。它把「一条指令
;;;;      算 4/8 个元素」替换「一条指令算 1 个元素」，所以**必须**与标量循环
;;;;      逐位相同；尾块还必须走同一个标量表达式（不能另写一种近似）。
;;;;      取负/绝对值用符号位运算而不是 0−x / 条件判断：0.0 −(+0.0) 给 +0.0，
;;;;      而 (− +0.0) 必须是 −0.0，靠数值测试抓不到单点，这里逐位比。
;;;;   2. 比较/谓词类结果 dtype 是 `:bool`（物理 ub8），此前一律回落到
;;;;      `vt-map` 的闭包路径（1M 实测 40ms）。现在进掩码内核，但写回必须
;;;;      走**真值化**（`%bool-flag`）而不是 `%cast-to` 截断 —— 后者会把
;;;;      `(vt-abs x :dtype :bool)` 的 0.84 写成 0。两条路必须逐位相同。
;;;;   3. 若干把「两趟两数组」合成「一趟一数组」的算子（方差、softmax 的
;;;;      减最大值+exp、layer-norm 的归一化、GELU/Softplus/LeakyReLU 由闭包
;;;;      改具名算子）。融合只允许省掉中间数组，不允许改任何一次舍入 ——
;;;;      所以这里全部与「分步参考实现」逐位对拍。
;;;;
;;;; 用法：sbcl --dynamic-space-size 4096 --non-interactive \
;;;;             --load test/perf-simd-ew-test.lisp

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
  (incf *pass*)
  t)

(defun ng (label fmt &rest args)
  (incf *fail*)
  (let ((msg (apply #'format nil fmt args)))
    (push (format nil "~a — ~a" label msg) *failures*)
    (format t "~&  [FAIL] ~a~%         ~a~%" label msg))
  nil)

(defun check (label test)
  (if test (ok label) (ng label "断言为假")))

(defun %report-ratio (label ratio limit what)
  (if (and (numberp ratio) (<= ratio limit))
      (progn (ok label)
             (format t "  [ OK ] ~a（~a = ~,2f× ≤ ~,1f×）~%" label what ratio limit))
      (ng label "~a = ~,2f× > ~,1f×" what (if (numberp ratio) (float ratio) -1.0) limit)))

(defmacro check-ratio (label form limit what)
  "倍率护栏。FORM 求值得到「快路径 / 参考实现」的倍率。

   超过 LIMIT 时**重测**，最多三次取最小：倍率护栏抓的是「快路径没接管」
   这类**结构性**回归（那种情形倍率通常在 3× 以上），而本机的噪声源是
   邻居进程 / 降频 / 大页抖动 —— 一次抖动不该把整轮判红。D4 的融合倍率
   实测 0.85×、限 0.9×，余量只有 6%，全套跑的时候正好被噪声顶过线一次。
   重测只在**已经超标**时发生，通过时开销与之前完全相同。"
  `(let* ((limit ,limit)
          (r1 (funcall (lambda () ,form)))
          (ratio (if (<= r1 limit)
                     r1
                     (min r1 (funcall (lambda () ,form)) (funcall (lambda () ,form))))))
     (%report-ratio ,label ratio limit ,what)))

;;; ------------------------------------------------------------------
;;; 逐位比较工具
;;;
;;; 用 EQL 逐元素比，**不**用 isclose：这套测试要抓的正是「差 1 ulp」和
;;; 「-0.0 vs +0.0」这类数值比较看不见的偏差。EQL 对浮点的语义是「同类型
;;; 同值」，-0.0 与 +0.0 用 EQL 判为不同（SBCL 下 (eql -0.0d0 0.0d0) → NIL）。
;;; NaN 需要单独处理：两个 NaN 用 EQL 恒假，改为比位模式。
;;; ------------------------------------------------------------------
(defun %bits (x)
  (cond ((typep x 'double-float) (sb-kernel:double-float-bits x))
        ((typep x 'single-float) (sb-kernel:single-float-bits x))
        (t x)))

(defun %elem-eq (a b)
  (if (and (floatp a) (floatp b)
           (not (= a a)) (not (= b b)))
      (eql (%bits a) (%bits b))
      (eql a b)))

(defun same-bits-p (x y)
  "形状 / dtype / 每个元素逐位相同（含 -0.0 与 NaN 载荷）。"
  (and (equal (vt-shape x) (vt-shape y))
       (equal (vt-dtype x) (vt-dtype y))
       (let ((dx (vt-data x)) (dy (vt-data y))
             (ox (vt-offset x)) (oy (vt-offset y))
             (n (vt-size x)))
         (dotimes (i n t)
           (unless (%elem-eq (aref dx (+ ox i)) (aref dy (+ oy i)))
             (return nil))))))

;;; ------------------------------------------------------------------
;;; 计时（倍率护栏用）：先校准批量次数让单次测量跨 ≥ 20ms，再取中位数。
;;; 只用于**同进程内两个实现的比值**，不写绝对毫秒。
;;; ------------------------------------------------------------------
(defun %once-us (thunk)
  (let ((t0 (get-internal-real-time)))
    (funcall thunk)
    (* (/ 1000000 internal-time-units-per-second)
       (- (get-internal-real-time) t0))))

(defun %calibrate (thunk)
  (let ((inner 1))
    (loop
      (when (>= (* inner (%once-us thunk)) 20000) (return inner))
      (setf inner (* inner 4))
      (when (> inner 100000) (return inner)))))

(defun %median-us (thunk)
  (let* ((inner (%calibrate thunk))
         (samples
           (loop repeat 7
                 collect (/ (let ((t0 (get-internal-real-time)))
                              (dotimes (i inner) (funcall thunk))
                              (* 1000000 (- (get-internal-real-time) t0)))
                            (* inner internal-time-units-per-second)))))
    (nth 3 (sort samples #'<))))

;;; ==================================================================
;;; A. 二元 SIMD：与标量内联循环逐位相同
;;; ==================================================================
(format t "~%=== A. 二元 SIMD ≡ 标量内联（逐位）===~%")

(defun %mk (shape dt &key (start 0.5d0) (step 0.75d0))
  (vt-astype (vt-reshape (vt-arange (reduce #'* shape) :dtype :float64
                                     :start start :step step)
                         shape)
             dt))

(let ((shapes '((1) (3) (7) (8) (63) (1000) (2 3) (8 8) (17 13) (4 1) (5 3 2))))
  (dolist (dt '(:float64 :float32))
    (dolist (shape shapes)
      (let ((a (%mk shape dt))
            (b (%mk shape dt :start 1.25d0 :step 0.5d0)))
        (flet ((cmp (name thunk)
                 (let ((fast (funcall thunk))
                       (slow (let ((*simd-ew1-fn* nil) (*simd-ew2-fn* nil))
                               (funcall thunk))))
                   (check (format nil "A ~a ~a ~a" name dt shape)
                          (same-bits-p fast slow)))))
          (cmp "+" (lambda () (vt-+ a b)))
          (cmp "-" (lambda () (vt-- a b)))
          (cmp "*" (lambda () (vt-* a b)))
          (cmp "/" (lambda () (vt-/ a b)))
          (cmp "neg" (lambda () (vt-- a)))
          (cmp "abs" (lambda () (vt-abs a)))
          (cmp "sqrt" (lambda () (vt-sqrt (vt-abs a)))))))))

;;; 标量广播（长度 1 的另一侧）
(let ((a (%mk '(1000) :float64)))
  (flet ((cmp (name thunk)
           (check (format nil "A 标量广播 ~a" name)
                  (same-bits-p (funcall thunk)
                               (let ((*simd-ew1-fn* nil) (*simd-ew2-fn* nil))
                                 (funcall thunk))))))
    (cmp "+2.5" (lambda () (vt-+ a 2.5d0)))
    (cmp "*3" (lambda () (vt-* a 3)))
    (cmp "-0.25" (lambda () (vt-- a 0.25d0)))))

;;; 行广播（bias 相加 / 按行缩放）：SIMD 的 :row 形态
(let* ((m (%mk '(8 15) :float64))
       (r (%mk '(15) :float64 :start 1.0 :step 0.25))
       (c (%mk '(8 1) :float64 :start 2.0)))
  (flet ((cmp (name thunk)
           (check (format nil "A 行广播 ~a" name)
                  (same-bits-p (funcall thunk)
                               (let ((*simd-ew2-fn* nil)) (funcall thunk))))))
    (cmp "A+行" (lambda () (vt-+ m r)))
    (cmp "A*行" (lambda () (vt-* m r)))
    (cmp "行-A" (lambda () (vt-- r m)))
    (cmp "A/行" (lambda () (vt-/ m r)))
    (cmp "A+列" (lambda () (vt-+ m c)))))

;;; 列广播（每行一个标量：(N,K) ∘ (N,1)、3-D 的 (N,M,K) ∘ (N,M,1)）：
;;; 归一化层与偏置的形态，SIMD 的 :col 分支。此前它不满足「末轴对齐」，
;;; 一律落到 odometer 通用路径；结果是「看着一样、慢 8 倍」，只有逐位对拍
;;; 与选路断言才能同时锁住两头。
(let* ((m (%mk '(8 15) :float64))
       (c (%mk '(8 1) :float64 :start 2.0))
       (r (%mk '(1 15) :float64 :start 0.5))
       (m3 (%mk '(4 3 7) :float64))
       (c3 (%mk '(4 3 1) :float64 :start 1.5)))
  (flet ((cmp (name thunk)
           (check (format nil "A 列广播 ~a" name)
                  (same-bits-p (funcall thunk)
                               (let ((*simd-ew2-fn* nil)) (funcall thunk))))))
    (cmp "A+列" (lambda () (vt-+ m c)))
    (cmp "列+A" (lambda () (vt-+ c m)))
    (cmp "A-列" (lambda () (vt-- m c)))
    (cmp "列-A" (lambda () (vt-- c m)))
    (cmp "A*列" (lambda () (vt-* m c)))
    (cmp "A/列" (lambda () (vt-/ m c)))
    (cmp "A+行(1,K)" (lambda () (vt-+ m r)))
    (cmp "3D A+列" (lambda () (vt-+ m3 c3)))
    (cmp "3D A*列" (lambda () (vt-* m3 c3)))
    (cmp "f32 A-列" (lambda () (vt-- (vt-astype m :float32)
                                     (vt-astype c :float32))))))

;;; 列广播的选路断言：钩子必须真的接管（否则上述对拍两边都是慢路径，
;;; 「一致」一文不值）。
(let* ((m (%mk '(8 15) :float64))
       (c (%mk '(8 1) :float64 :start 2.0))
       (res (clvt::%alloc-vt '(8 15) :float64)))
  (check "A 列广播由 %sew2 接管（:col 选路）"
         (and (clvt::%sew2-try '- m c res) t))
  (check "A 列广播结果写满且正确"
         (same-bits-p res (let ((*simd-ew2-fn* nil)) (vt-- m c)))))

;;; 特殊值：±0 / ±Inf / NaN / 次正规 / 极值
(let* ((vals (vector 0d0 -0.0d0 1d0 -1d0 0.5d0 -0.5d0
                     most-positive-double-float (- most-positive-double-float)
                     1d-320 1d300
                     sb-ext:double-float-positive-infinity
                     sb-ext:double-float-negative-infinity
                     (vt-get-nan :float64)))
       (a (vt-asarray (coerce vals 'list) :dtype :float64))
       (b (vt-asarray (coerce (reverse vals) 'list) :dtype :float64)))
  (flet ((cmp (name thunk)
           (check (format nil "A 特殊值 ~a" name)
                  (same-bits-p (funcall thunk)
                               (let ((*simd-ew1-fn* nil) (*simd-ew2-fn* nil))
                                 (funcall thunk))))))
    (cmp "+" (lambda () (vt-+ a b)))
    (cmp "*" (lambda () (vt-* a b)))
    (cmp "/" (lambda () (vt-/ a b)))
    (cmp "neg" (lambda () (vt-- a)))
    (cmp "abs" (lambda () (vt-abs a)))
    ;; sqrt 的负数分支最容易被「硬件 sqrtsd」写错：`(sqrt −1.0d0)` 在 CL 里
    ;; 是复数 #C(0.0d0 1.0d0)，标量内核换成规范 NaN（+vt-dfloat-nan+），而
    ;; `sqrtsd` 给的是符号位为 1 的 indefinite NaN。两个都「是 NaN」，
    ;; 逐位对拍才看得见。
    (cmp "sqrt(含负数)" (lambda () (vt-sqrt a)))
    (cmp "sqrt(-inf/次正规)" (lambda () (vt-sqrt (vt-- a))))))

;;; 非连续输入：SIMD 不接管，结果必须与「一直走标量」完全一致
(let* ((m (%mk '(8 15) :float64))
       (t-view (vt-transpose m)))
  (check "A 非连续输入逐位一致"
         (same-bits-p (vt-+ t-view t-view)
                      (let ((*simd-ew2-fn* nil)) (vt-+ t-view t-view))))
  (check "A 非连续输入仍正确（与连续副本比）"
         (same-bits-p (vt-+ t-view t-view)
                      (vt-+ (vt-contiguous t-view) (vt-contiguous t-view)))))

;;; ==================================================================
;;; B. 布尔掩码内核：真值化，不是截断
;;; ==================================================================
(format t "~%=== B. :bool 结果逐位一致（真值 vs 截断）===~%")

(let* ((a (vt-asarray (list 0d0 0.5d0 1d0 2d0 -1d0 -0.0d0
                             most-positive-double-float
                             sb-ext:double-float-positive-infinity
                             (vt-get-nan :float64))))
       (b (vt-asarray (list 1d0 0.5d0 2d0 1d0 1d0 0d0 0d0 0d0 0d0))))
  (flet ((cmp (name thunk)
           (check (format nil "B ~a" name)
                  (same-bits-p (funcall thunk)
                               (let ((clvt::*simd-ew1-fn* nil)
                                     (clvt::*simd-ew2-fn* nil))
                                 (funcall thunk))))))
    (cmp "<" (lambda () (vt-< a b)))
    (cmp ">" (lambda () (vt-> a b)))
    (cmp "<=" (lambda () (vt-<= a b)))
    (cmp ">=" (lambda () (vt->= a b)))
    (cmp "=" (lambda () (vt-= a b)))
    (cmp "/=" (lambda () (vt-/= a b))))
  (check "B 比较结果 dtype 是 :bool" (eq (vt-dtype (vt-< a b)) :bool))
  (check "B 比较结果只含 0/1"
         (let ((d (vt-data (vt-< a b))) (n (vt-size (vt-< a b))))
           (dotimes (i n t) (unless (member (aref d i) '(0 1)) (return nil)))))
  ;; 关键的真值化用例：算子结果不是 0/1（abs 0.5 → 真值 1，截断 0）
  ;; 关键：0.5 与 2.0 的真值都是 1，截断则分别是 0 与 2。参考取 vt-map
  ;; （布尔结果在 vt-map 里经 vt-cast-fun → %bool-flag 写回，是语义基准）。
  (let ((probe (vt-asarray '(0.5d0 0d0 2d0 -0.0d0) :dtype :float64)))
    (check "B (vt-abs :dtype :bool) 真值化而非截断"
           (same-bits-p (vt-abs probe :dtype :bool)
                        (vt-map #'abs probe :dtype :bool)))
    (check "B 真值化结果为 0/1（0.5→1、2→1）"
           (equal (vt-to-list (vt-abs probe :dtype :bool)) '(1 0 1 0))))
  (check "B (vt-isnan) 只对 NaN 为真"
         (equal (mapcar (lambda (v) (= v 1)) (vt-to-list (vt-isnan a)))
                (list nil nil nil nil nil nil nil nil t))))

;;; 一元的 bool 结果（isinf/isfinite 走掩码路径，且是 4 种实数输入之一）
(let ((v (vt-asarray (list 1d0 sb-ext:double-float-positive-infinity
                           (vt-get-nan :float64) 0d0))))
  (check "B isinf 正确"
         (equal (mapcar (lambda (b) (= b 1)) (vt-to-list (vt-isinf v)))
                '(nil t nil nil)))
  (check "B isfinite 正确"
         (equal (mapcar (lambda (b) (= b 1)) (vt-to-list (vt-isfinite v)))
                '(t nil nil t))))

;;; 非连续输入（步长切片）的掩码路径：结果与「一直走 vt-map」逐位一致
(let* ((a (vt-slice (vt-arange 128 :dtype :float64 :start 0.5) '(nil nil 2)))
       (b (vt-arange 64 :dtype :float64 :start 1.5)))
  (check "B 非连续输入掩码与闭包路径一致"
         (same-bits-p (vt-< a b)
                      (let ((*simd-ew1-fn* nil) (*simd-ew2-fn* nil))
                        (vt-map #'%op-lt a b :dtype :bool)))))

;;; ==================================================================
;;; C. 融合内核 ≡ 分步参考实现（逐位）
;;; ==================================================================
(format t "~%=== C. 融合算子 ≡ 分步实现（逐位）===~%")

(defparameter *ln-a* (vt-reshape (vt-arange 40 :dtype :float64 :start 1.0 :step 0.37)
                                 '(5 8)))
(defparameter *ln-g* (vt-arange 8 :dtype :float64 :start 0.5 :step 0.11))
(defparameter *ln-b* (vt-arange 8 :dtype :float64 :start -0.2 :step 0.03))

;; C1 方差：(x−mean)² 合成一趟 vs 两趟分开
(let* ((a *ln-a*)
       (mean (vt-mean a :axis 1 :keepdims t :dtype :float64))
       (fused (vt-var a :axis 1 :dtype :float64))
       (two-pass (vt-/ (vt-sum (vt-square (vt-- a mean :dtype :float64)
                                          :dtype :float64)
                               :axis 1 :keepdims nil :dtype :float64)
                        (coerce 8 'double-float)
                        :dtype :float64)))
  (check "C1 vt-var 沿轴 ≡「mean→sub→square→sum→div」" (same-bits-p fused two-pass)))
(let* ((a *ln-a*)
       (n (coerce (vt-size a) 'double-float))
       (mean (vt-/ (vt-item (vt-sum a :dtype :float64)) n))
       (fused (vt-var a :dtype :float64))
       (two-pass (vt-/ (vt-item (vt-sum (vt-square (vt-- a mean :dtype :float64)
                                                   :dtype :float64)
                                        :dtype :float64))
                       n :dtype :float64)))
  (check "C1 vt-var 全归约 ≡ 两趟参考"
         (same-bits-p fused two-pass)))
(check "C1 vt-std = sqrt(vt-var)"
       (same-bits-p (vt-std *ln-a* :axis 1)
                    (vt-sqrt (vt-var *ln-a* :axis 1) :dtype :float64)))

;; C2 softmax：减最大值再 exp 合成一趟 vs 两趟
(let* ((a *ln-a*)
       (max-val (vt-amax a :axis 1 :keepdims t :dtype :float64))
       (fused (vt-softmax a :axis 1))
       (two-pass (let* ((e (vt-exp (vt-- a max-val :dtype :float64) :dtype :float64))
                        (s (vt-sum e :axis 1 :keepdims t :dtype :float64)))
                   (vt-/ e s :dtype :float64))))
  (check "C2 vt-softmax ≡「amax→sub→exp→sum→div」" (same-bits-p fused two-pass)))
(check "C2 softmax 每行和为 1"
       (let ((s (vt-to-list (vt-sum (vt-softmax *ln-a* :axis 1) :axis 1))))
         (every (lambda (v) (< (abs (- v 1.0d0)) 1d-12)) s)))
(check "C2 softmax float32 保持 float32"
       (eq (vt-dtype (vt-softmax (vt-astype *ln-a* :float32) :axis 1)) :float32))
;; C3 log-softmax
(let* ((a *ln-a*)
       (max-val (vt-amax a :axis 1 :keepdims t :dtype :float64))
       (fused (vt-log-softmax a :axis 1))
       (two-pass (let* ((sh (vt-- a max-val :dtype :float64))
                        (lse (vt-log (vt-sum (vt-exp sh :dtype :float64)
                                             :axis 1 :keepdims t :dtype :float64)
                                     :dtype :float64)))
                   (vt-- sh lse :dtype :float64))))
  (check "C3 vt-log-softmax ≡「shift→exp→sum→log→sub」" (same-bits-p fused two-pass)))

;; C4 layer-norm：(x−mean)/std 合成一趟 vs 两趟
(let* ((a *ln-a*)
       (mean (vt-mean a :axis 1 :keepdims t :dtype :float64))
       (var (vt-var a :axis 1 :keepdims t :dtype :float64))
       (std (vt-sqrt (vt-+ var 1d-5 :dtype :float64) :dtype :float64))
       (fused (vt-layer-norm a '(8) :gamma *ln-g* :beta *ln-b*))
       (two-pass (let* ((normed (vt-/ (vt-- a mean :dtype :float64) std :dtype :float64))
                        (scaled (vt-* normed *ln-g* :dtype :float64)))
                   (vt-+ scaled *ln-b* :dtype :float64))))
  (check "C4 vt-layer-norm ≡「sub→div→mul→add」" (same-bits-p fused two-pass)))
(check "C4 layer-norm 不含 gamma/beta 时 ≡ 归一化本身"
       (same-bits-p (vt-layer-norm *ln-a* '(8))
                    (let* ((a *ln-a*)
                           (mean (vt-mean a :axis 1 :keepdims t :dtype :float64))
                           (var (vt-var a :axis 1 :keepdims t :dtype :float64))
                           (std (vt-sqrt (vt-+ var 1d-5 :dtype :float64) :dtype :float64)))
                      (vt-/ (vt-- a mean :dtype :float64) std :dtype :float64))))

;; C5 激活函数：具名算子 ≡ 闭包参考（闭包参考直接照抄原 lambda 体）
(defun %ref-gelu (x)
  (let* ((c (sqrt (/ 2.0d0 (coerce pi 'double-float))))
         (x3 (* x x x))
         (inner (+ x (* 0.044715d0 x3)))
         (tanh-val (tanh (* c inner))))
    (* 0.5d0 x (+ 1.0d0 tanh-val))))
(defun %ref-softplus (x)
  (if (> x 20.0d0) x (log (+ 1.0d0 (exp x)))))

(dolist (dt '(:float64 :float32))
  (let* ((v (vt-astype (vt-arange 64 :dtype :float64 :start -4.0 :step 0.13) dt))
         (ref-gelu (vt-map #'%ref-gelu v :dtype dt))
         (ref-sp (vt-map #'%ref-softplus v :dtype dt)))
    (check (format nil "C5 vt-gelu ~a ≡ 参考 lambda" dt)
           (same-bits-p (vt-gelu v) ref-gelu))
    (check (format nil "C5 vt-softplus ~a ≡ 参考 lambda" dt)
           (same-bits-p (vt-softplus v) ref-sp))
    (check (format nil "C5 vt-leaky-relu ~a ≡ 参考 lambda" dt)
           (same-bits-p (vt-leaky-relu v)
                        (vt-map (lambda (x) (if (> x 0.0d0) x (* 0.01d0 x)))
                                v :dtype dt)))))
(check "C5 gelu 整数输入 → float64 且值正确"
       (let ((g (vt-gelu (vt-asarray '(1) :dtype :int64))))
         (and (eq (vt-dtype g) :float64)
              (< (abs (- (vt-ref g 0) 0.8411919906082768d0)) 1d-15))))
(check "C5 mish ≡ x*tanh(softplus(x))"
       (same-bits-p (vt-mish *ln-a*)
                    (let ((sp (vt-softplus *ln-a*)))
                      (vt-* *ln-a* (vt-tanh sp) :dtype :float64))))

;;; ==================================================================
;;; D. 选路结构 + 倍率护栏（不写绝对毫秒）
;;; ==================================================================
(format t "~%=== D. 选路结构与倍率护栏 ===~%")

(defparameter *big-a* (vt-arange 1000000 :dtype :float64 :start 1.0))
(defparameter *big-b* (vt-arange 1000000 :dtype :float64 :start 2.0))
(defparameter *big-m* (vt-reshape *big-a* '(1000 1000)))
(defparameter *big-r* (vt-arange 1000 :dtype :float64 :start 1.0))
(defparameter *big-c* (vt-reshape (vt-arange 1000 :dtype :float64 :start 0.5) '(1000 1)))

(check "D1 SIMD 钩子已注册（二元）" (functionp clvt::*simd-ew2-fn*))
(check "D1 SIMD 钩子已注册（一元）" (functionp clvt::*simd-ew1-fn*))
(check "D1 钩子对连续 f64 加法返回 T"
       (let ((res (clvt::%alloc-vt '(1000000) :float64)))
         (and (clvt::%sew2-try '+ *big-a* *big-b* res) t)))
(check "D1 钩子对 int64 返回 NIL（不接管）"
       (not (clvt::%sew2-try '+ (vt-astype *big-a* :int64) *big-b*
                             (clvt::%alloc-vt '(1000000) :float64))))
(check "D1 钩子对行广播返回 T"
       (let ((res (clvt::%alloc-vt '(1000 1000) :float64)))
         (and (clvt::%sew2-try '+ *big-m* *big-r* res) t)))

;;; 三条倍率都**复用预分配的 `:out`**。这里要量的是循环本身，不是分配器：
;;; 每调用一次 `vt-+` 就新分配 8 MB，样本内 20 次调用会撞上 GC，而 GC 的
;;; 时刻与快慢路径无关 —— 于是同一个 1 ms 的运算会在 1× 到 1.5× 之间随机
;;; 漂移（实测出现过 simd/scalar = 1.46× 的假红灯）。写进同一块 out 之后
;;; 两侧都零分配，倍率稳定在 0.3× 附近。
(let ((out1 (%alloc-vt '(1000000) :float64)))
  (check-ratio "D2 SIMD 加法 ≤ 0.75× 标量内联"
               (/ (%median-us (lambda () (vt-+ *big-a* *big-b* :out out1)))
                  (%median-us (lambda () (let ((clvt::*simd-ew2-fn* nil))
                                           (vt-+ *big-a* *big-b* :out out1)))))
               0.75 "simd/scalar"))

(let ((out1 (%alloc-vt '(1000000) :float64)))
  (check-ratio "D2 SIMD 乘法 ≤ 0.75× 标量内联"
               (/ (%median-us (lambda () (vt-* *big-a* *big-b* :out out1)))
                  (%median-us (lambda () (let ((clvt::*simd-ew2-fn* nil))
                                           (vt-* *big-a* *big-b* :out out1)))))
               0.75 "simd/scalar"))

;;; 基线必须**两个钩子都关**：`*simd-ew2-fn*` 关掉之后，行广播这个形态
;;; 会掉进 `%inline2-strided`，而那条路的内层 run 现在也接到了
;;; `*simd-ew2-run-fn*`（§7.4.4）—— 于是「关掉 SIMD」实际关的是扁平向量
;;; 循环，比的成了两套向量实现的快慢，护栏就失去意义了（实测 1.48× 假红灯）。
(let ((out2 (%alloc-vt '(1000 1000) :float64)))
  (check-ratio "D2 SIMD 行广播加法 ≤ 0.5× 标量 odometer"
               (/ (%median-us (lambda () (vt-+ *big-m* *big-r* :out out2)))
                  (%median-us (lambda () (let ((clvt::*simd-ew2-fn* nil)
                                               (clvt::*simd-ew2-run-fn* nil))
                                           (vt-+ *big-m* *big-r* :out out2)))))
               0.5 "simd/strided"))

;;; 列广播同形（归一化层的实际形态）
(let ((out3 (%alloc-vt '(1000 1000) :float64)))
  (check-ratio "D2 SIMD 列广播减法 ≤ 0.5× 标量 odometer"
               (/ (%median-us (lambda () (vt-- *big-m* *big-c* :out out3)))
                  (%median-us (lambda () (let ((clvt::*simd-ew2-fn* nil))
                                           (vt-- *big-m* *big-c* :out out3)))))
               0.5 "simd/col"))

;;; 比较类：从 vt-map 闭包路径降到掩码内联循环，量级差距最大
(check-ratio "D3 float64 比较 ≤ 0.3× 闭包路径"
             (/ (%median-us (lambda () (vt-< *big-a* *big-b*)))
                (%median-us (lambda () (let ((clvt::*simd-ew2-fn* nil))
                                         (vt-map #'%op-lt *big-a* *big-b*
                                                 :dtype :bool)))))
             0.3 "mask/closure")

;;; 融合类：方差与 softmax 的倍率（对照各自的分步参考实现）
;;;
;;; 0.8 → 0.9：分步参考里的 `(vt-- m mean)` 现在也走 :col 向量循环，于是
;;; 「省一趟」在总耗时里的占比被摊薄（分子分母同时变快）。方差剩下的
;;; 时间主要是 mean / sum-axis 两趟归约与两次分配 —— 融合只能省掉
;;; sub 那一趟和一块中间数组，省不出第三趟。护栏的作用是「不许比
;;; 分步更慢」，实测 0.85~0.88×（随机器状态浮动），取 0.95 留出噪声余量：
;;; 护栏要抓的是「融合没接管」（那种情形会掉到 1.5× 以上），不是 3% 抖动。
(check-ratio "D4 vt-var 沿轴 ≤ 0.95× 分步实现"
             (/ (%median-us (lambda () (vt-var *big-m* :axis 1)))
                (%median-us (lambda () (let ((mean (vt-mean *big-m* :axis 1 :keepdims t)))
                                         (vt-sum (vt-square (vt-- *big-m* mean)
                                                            :dtype :float64)
                                                 :axis 1)))))
             0.95 "fused/stepwise")

;;; GELU 在 x≈−0.75 处有极小值，**不是**单调函数（−0.17 附近的下凹是
;;; tanh 近似式的固有形状），所以这里只钉三条可判定的事实：下界、
;;; 零点、大正数处的恒等性。
(check "D5 GELU 值域/零点/大值恒等（数值回归网）"
       (let* ((v (vt-to-list (vt-gelu (vt-arange 41 :dtype :float64 :start -10.0))))
              (minv (reduce #'min v)))
         (and (every #'numberp v)
              (> minv -0.171d0)
              (< (abs (vt-ref (vt-gelu (vt-asarray '(0.0d0))) 0)) 1d-15)
              (< (abs (- (vt-ref (vt-gelu (vt-asarray '(8.0d0))) 0) 8.0d0)) 1d-6))))

;;; ==================================================================
;;; E. 原地更新（out 与输入是同一视图）
;;;
;;; `(vt-+ a b :out a)` 是 NN 里的常态写法。它既是别名的一种，又是唯一
;;; 「逐元素写不破坏未读数据」的别名形态：写 out[i] 时读的正是同一位置的
;;; a[i]。此前的别名保护对它一律整块 vt-copy 再走通用循环（1M f64 实测
;;; 55 ms），现在留在快路径上并跳过快照。这里同时钉住两头：
;;;   · 结果与「非原地」逐位相同；
;;;   · 错位别名（翻转/错位切片）仍然按 numpy 的「先读全量再写」语义走快照。
;;; ==================================================================
(format t "~%=== E. 原地更新与别名额 ===~%")

(let* ((base (%mk '(64 32) :float64 :start 0.5d0))
       (b (%mk '(64 32) :float64 :start 1.25d0 :step 0.5d0)))
  (flet ((inplace (op)
           (let ((a (vt-copy base)))
             (funcall op a b a))))
    (check "E1 (vt-+ a b :out a) ≡ 非原地"
           (same-bits-p (inplace (lambda (x y out) (vt-+ x y :out out)))
                        (vt-+ base b)))
    (check "E1 (vt-* a b :out a) ≡ 非原地"
           (same-bits-p (inplace (lambda (x y out) (vt-* x y :out out)))
                        (vt-* base b)))
    (check "E1 (vt-- a b :out a) ≡ 非原地"
           (same-bits-p (inplace (lambda (x y out) (vt-- x y :out out)))
                        (vt-- base b)))
    (check "E1 (vt-/ a b :out a) ≡ 非原地"
           (same-bits-p (inplace (lambda (x y out) (vt-/ x y :out out)))
                        (vt-/ base b)))
    ;; 列广播 + 原地：归一化层的实际形态
    (check "E1 原地 + 列广播 ≡ 非原地"
           (let* ((c (%mk '(64 1) :float64 :start 2.0))
                  (a (vt-copy base)))
             (vt-- a c :out a)
             (same-bits-p a (vt-- base c))))))

;;; 别名判定的结构断言：同一视图不拷贝，错位视图必须拷贝
(let* ((z (vt-copy (%mk '(8) :float64)))
       (rev (vt-flip z)))
  (check "E2 同一视图不做快照" (eq (first (vt-out-snapshot z (list z))) z))
  (check "E2 错位（翻转）视图必须快照"
         (not (eq (first (vt-out-snapshot rev (list z))) z))))

;;; 错位别名的语义：numpy 的「先读全量、再写」，不能被原地快路optimize 掉
(let* ((z (vt-asarray (list 1d0 2d0 3d0 4d0)))
       (rev (vt-flip z))
       (want (let ((ref (vt-asarray (list 1d0 2d0 3d0 4d0))))
               (vt-+ (vt-copy ref) (vt-copy ref)))))
  (vt-+ z z :out rev)
  ;; numpy 实测该调用后底层缓冲为 [8,6,4,2]：out 是 z 的翻转视图，
  ;; 先按 out 的逻辑序算出 [2,4,6,8]，再逆序写回 z。
  ;; same-bits-p 按 (data, offset) 线性比较，只对连续视图成立；
  ;; 这里两侧都是翻转视图，先物化再比。
  (check "E3 (vt-+ z z :out (flip z)) 按「先读全量」语义"
         (same-bits-p (vt-contiguous z) (vt-contiguous (vt-flip want)))))

;;; ==================================================================
;;; 汇总
;;; ==================================================================
(format t "~&~%===== 汇总: ~a 通过 / ~a 失败 =====~%" *pass* *fail*)
(format t "Total: ~a | Pass: ~a | Fail: ~a~%" (+ *pass* *fail*) *pass* *fail*)
(when *failures*
  (format t "失败清单:~%")
  (dolist (f (nreverse *failures*)) (format t "  - ~a~%" f)))
(finish-output)
(sb-ext:exit :code (if (zerop *fail*) 0 1))
