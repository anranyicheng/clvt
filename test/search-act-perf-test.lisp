;;;; search-act-perf-test.lisp — 搜箱 / 取整 / 激活三组「执行层」改写的回归测试
;;;;
;;;; 三处改动，病根各不相同，但都是「执行层」的改写，逻辑层一字未动：
;;;;
;;;;   1. src/indexing.lisp    vt-digitize
;;;;      每元素一次泛型二分（BIN-DATA 是 `(simple-array t)`，每次比较都要
;;;;      查 widetag + 拆箱 + 泛型算术）→ 按「比较域」特化两条内核
;;;;      （有浮点走 double，全整型走 (signed-byte 64)），X / BINS 各转一次。
;;;;      1M / 1000 bins：138 ms → 16.0 ms。
;;;;   2. src/elementwise1.lisp %floor1 / %ceiling1
;;;;      走 `ffloor` / `fceiling` 而不是「取整数结果再 (float res x) 转回」——
;;;;      后者对 double 要走 `%UNARY-TRUNCATE-DOUBLE-FLOAT-TO-BIGNUM` 的分派。
;;;;      1M：3.9 → 2.8（floor）、4.0 → 2.7（ceiling）。
;;;;      ROUND / TRUNCATE **不换**（它们的整数版有 cvtsd2si 直通，浮点版
;;;;      反而慢 6 倍），这条也在本套件里用「不许变慢」的反向断言钉住。
;;;;   3. src/nn.lisp          vt-swish / vt-leaky-relu / vt-hard-sigmoid
;;;;      swish 的第二步是 `(vt-map #'* …)`（函数版，算子无法内联）；
;;;;      leaky-relu 把 ALPHA 绑在特殊变量上（循环里每元素一次 SYMBOL-VALUE）；
;;;;      hard-sigmoid 走 scale→add→clip 三趟，每趟一次 1M 元素的读写往返。
;;;;      1M f64：22.6→11.2、16.2→6.4、8.7→8.0。
;;;;
;;;; 本套件的核心断言是**逐位等价**：每个函数都与一份逐字复刻的改动前实现
;;;; 逐元素比对（浮点比 IEEE 位模式，−0.0 与 +0.0 视为不同），覆盖 NaN /
;;;; ±Inf / ±0.0 / denormal / 2^53±1 这些「改了就不对」的值。
;;;; 不写手抄期望值 —— 抄期望值只能证明「现在是这样」，挡不住偏移。
;;;;
;;;; 第二层是**机器无关的倍率护栏**（§16.16 第 7 条：不写绝对毫秒）：
;;;; 改回旧写法结果照样正确、A 段全绿，只是慢 2~10 倍。所以每组再配一条
;;;; 「相对某个同进程基准的比值上界」，两边同机同进程跑。

(require :asdf)
#+quicklisp (ql:quickload :clvt :silent t)
(asdf:load-system :clvt)

(defpackage :clvt-search-act-test
  (:use :cl :clvt)
  (:import-from :clvt
                #:ensure-vt #:vt-fast-map #:vt-flatten #:vt-data #:vt-size
                #:vt-do-each #:%make-vt #:vt-compute-strides #:vt-ravel
                #:%bins-monotonicity #:%floor-family-body #:%floor-family-body1
                #:with-float-safe
                #:*nn-act-param*)
  (:export #:run))

(in-package :clvt-search-act-test)

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
  "倍率护栏：RATIO 必须 ≤ LIMIT。"
  (incf *n*)
  (if (and (numberp ratio) (<= ratio limit))
      (progn (incf *pass*)
             (format t "  [ OK ] ~a（~a = ~,2f× ≤ ~,1f×）~%" label what (float ratio) limit))
      (progn (incf *fail*) (push label *failures*)
             (format t "  [FAIL] ~a（~a = ~,2f× > ~,1f×）~%" label what (float ratio) limit))))

(defvar *pinf* sb-ext:double-float-positive-infinity)
(defvar *ninf* (- sb-ext:double-float-positive-infinity))
(defparameter *nan* (sb-kernel:make-double-float #x7ff80000 0))

(defun nan-p (x)
  (and (floatp x) (sb-int:with-float-traps-masked (:invalid) (not (= x x)))))

(defun eqv (a b)
  "逐位比较：浮点看 IEEE 位模式（−0.0 ≠ +0.0），NaN 互认，整数 EQL。"
  (cond ((and (floatp a) (floatp b))
         (cond ((and (nan-p a) (nan-p b)) t)
               ((or (nan-p a) (nan-p b)) nil)
               (t (eql (if (typep a 'double-float)
                           (sb-kernel:double-float-bits a)
                           (sb-kernel:single-float-bits a))
                       (if (typep b 'double-float)
                           (sb-kernel:double-float-bits b)
                           (sb-kernel:single-float-bits b))))))
        ((and (null a) (null b)) t)
        ((or (null a) (null b)) nil)
        (t (eql a b))))

(defun flat (v)
  "按 C 序读出逻辑值（不经 vt-to-list，避免它对 :bool 的折叠）。"
  (let ((f (vt-ravel v)) (n (vt-size v)))
    (loop for i below n collect (vt-ref f i))))

(defun same-tensor-p (a b)
  (and (equal (vt-shape a) (vt-shape b))
       (eq (vt-dtype a) (vt-dtype b))
       (let ((la (flat a)) (lb (flat b)))
         (and (= (length la) (length lb))
              (loop for x in la for y in lb always (eqv x y))))))

(defun check-same (label got want)
  (check label (same-tensor-p got want)
         (format nil "got=~a want=~a"
                 (handler-case (subseq (flat got) 0 (min 8 (length (flat got))))
                   (error (e) (format nil "<~a>" e)))
                 (handler-case (subseq (flat want) 0 (min 8 (length (flat want))))
                   (error (e) (format nil "<~a>" e))))))

(defun check-error (label thunk)
  (check label (handler-case (progn (funcall thunk) nil) (error () t))))

(defvar *sink* nil
  "基准的落地点：把被测表达式的返回值存下来，避免编译器把它当死代码删掉。")

(defun %elapsed-ms (t0)
  (* 1000.0 (/ (- (get-internal-real-time) t0) internal-time-units-per-second)))

(defun best-ms-fn (thunk reps)
  "把 THUNK 跑 REPS 轮（每轮自动放大到 ≥100 ms），返回**单次调用**的毫秒数。

   三个要点，任何一个漏了基准就是错的：
     · 被测量必须是 `funcall` 进来的**函数**。内联在调用点的表达式会让
       编译器把两次读时钟当纯调用做公共子表达式消除，差值恒为 0。
     · 必须**放大**。本机 `GET-INTERNAL-REAL-TIME` 的名义单位是 1 微秒，
       但底层时钟只按 1 ms 跳（实测 tick 全是 1000 的整数倍），0.7 ms 的
       操作会被测成 0 或 1 ms。所以先校准出「单轮 ≥ 100 ms」的循环次数，
       再换算回单次耗时。
     · 结果写 `*SINK*`，否则整段计算可能被当死代码删掉。
   与 test/bench-access-view.lisp 的 `%MEASURE` / `%CALIBRATE` 是同一套做法。"
  (setf *sink* (funcall thunk))         ; 预热（含首次分配 / 页错误）
  (let ((inner 1))
    ;; 校准
    (loop
      (let* ((t0 (get-internal-real-time)))
        (dotimes (i inner) (setf *sink* (funcall thunk)))
        (let ((ms (%elapsed-ms t0)))
          (when (>= ms 100.0) (return))
          (setf inner (max (1+ inner)
                           (min (* inner 8)
                                (ceiling (* inner (/ 150.0 (max ms 1.0))))))))))
    ;; 正式测量
    (let ((best most-positive-double-float))
      (dotimes (r reps (/ best inner))
        (declare (ignore r))
        (let* ((t0 (get-internal-real-time)))
          (dotimes (i inner) (setf *sink* (funcall thunk)))
          (setf best (min best (%elapsed-ms t0))))))))

(defmacro best-ms (reps &body body)
  "`(best-ms REPS form)` —— 把 FORM 包成 thunk 交给 `BEST-MS-FN`。"
  `(best-ms-fn (lambda () ,@body) ,reps))

;;; ==================================================================
;;; 1. DIGITIZE：与「逐字复刻的改动前实现」逐位对拍
;;; ==================================================================
;;; 改动前的实现：泛型二分 + vt-do-each（BIN-DATA 不带元素类型）。
(defun old-bisect (data n reverse-p side-right-p val)
  (let ((lo 0) (hi n))
    (loop while (< lo hi) do
      (let* ((mid (ash (+ lo hi) -1))
             (mv (aref data (if reverse-p (- n 1 mid) mid))))
        (if (if side-right-p (> mv val) (>= mv val))
            (setf hi mid)
            (setf lo (1+ mid)))))
    lo))

(defun old-digitize (x bins &key right)
  ;; 复刻的实现没有屏蔽浮点陷阱；NaN 参与 `>' 比较在 x86 上会置 invalid。
  (with-float-safe
   (let* ((flat-x (vt-flatten (ensure-vt x)))
         (flat-bins (vt-flatten (ensure-vt bins)))
         (bin-data (vt-data flat-bins))
         (nbins (vt-size flat-bins))
         (mono (clvt::%bins-monotonicity bin-data nbins))
         (side-right-p (not right))
         (reverse-p (= mono -1))
         (result (make-array (vt-size flat-x) :element-type '(signed-byte 64))))
    (when (zerop mono)
      (error "vt-digitize: bins 必须单调递增或递减（numpy 语义）"))
    (vt-do-each (ptr val flat-x)
      (let ((ss (old-bisect bin-data nbins reverse-p side-right-p val)))
        (setf (aref result ptr) (if reverse-p (- nbins ss) ss))))
      (%make-vt :data result :shape (vt-shape (ensure-vt x))
                :strides (vt-compute-strides (vt-shape (ensure-vt x)))
                :offset 0 :dtype :int64))))

(defun digitize-suite ()
  (format t "~%=== [1] VT-DIGITIZE：比较域特化 ===~%")
  (let* ((inc (vt-from-sequence '(-4 -2 -1 0 1 2 4) :dtype :float64))
         (dec (vt-flip inc))
         (inc-i (vt-from-sequence '(-4 -2 -1 0 1 2 4) :dtype :int64))
         (inc-32 (vt-astype inc :float32))
         ;; 覆盖 bins 两侧、边界值本身、以及 NaN/±Inf
         (xs (vt-from-sequence
              (list -1d9 -5d0 -4d0 -3.5d0 -2d0 -1.5d0 -1d0 -0.5d0 -0.0d0 0.0d0
                    0.5d0 1d0 2d0 2.5d0 4d0 7d0 1d9 *nan* *pinf* *ninf*)
              :dtype :float64)))
    ;; A1：递增 / 递减 × side 左右 × bins dtype × x dtype
    (dolist (side '(:left :right))
      (let ((r (eq side :right)))
        (check-same (format nil "digitize f64/inc bins ~a" side)
                    (vt-digitize xs inc :right r)
                    (old-digitize xs inc :right r))
        (check-same (format nil "digitize f64/dec bins ~a" side)
                    (vt-digitize xs dec :right r)
                    (old-digitize xs dec :right r))
        (check-same (format nil "digitize f64/i64 bins ~a" side)
                    (vt-digitize xs inc-i :right r)
                    (old-digitize xs inc-i :right r))
        (check-same (format nil "digitize i64 bins ~a" side)
                    (vt-digitize (vt-astype xs :int64) inc-i :right r)
                    (old-digitize (vt-astype xs :int64) inc-i :right r))
        (check-same (format nil "digitize f32 x ~a" side)
                    (vt-digitize (vt-astype xs :float32) inc-32 :right r)
                    (old-digitize (vt-astype xs :float32) inc-32 :right r))
        (check-same (format nil "digitize uint8 x ~a" side)
                    (vt-digitize (vt-astype (vt-abs (vt-astype xs :int64)) :uint8) inc :right r)
                    (old-digitize (vt-astype (vt-abs (vt-astype xs :int64)) :uint8) inc :right r))))

    ;; A2：物理形态无关性 —— 同一逻辑数据，6 种视图结果必须逐位相同
    (let* ((base (vt-astype (vt-arange 60 :start 0 :step 1 :dtype :float64) :float64))
           (ref (vt-digitize base inc)))
      (dolist (pair (list (cons "step 2" (vt-slice base (list 0 nil 2)))
                          (cons "flip" (vt-flip base))
                          (cons "reshape+transpose"
                                (vt-transpose (vt-reshape base (list 6 10))))
                          (cons "2d" (vt-reshape base (list 6 10)))
                          (cons "3d permute"
                                (vt-transpose (vt-reshape base (list 3 4 5)) '(2 0 1)))))
        (let ((view (car pair)))
          (when (string= view "2d")
            (check (format nil "digitize ~a 形状与输入一致" view)
                   (equal (vt-shape (vt-digitize (cdr pair) inc)) (list 6 10))))
          (when (string= view "3d permute")
            (check (format nil "digitize ~a 形状与输入一致" view)
                   (equal (vt-shape (vt-digitize (cdr pair) inc)) (list 5 3 4))))
          (check-same (format nil "digitize ~a ≡ 连续副本" view)
                      (vt-digitize (cdr pair) inc)
                      (old-digitize (cdr pair) inc))))
      (check "digitize 连续输入结果与自身形状一致"
             (equal (vt-shape ref) (vt-shape base))))

    ;; A3：退化输入
    (check "digitize 空 x 返回空"
           (equal (vt-shape (vt-digitize (vt-from-sequence '() :dtype :float64) inc)) '(0)))
    (check "digitize 单 bin"
           (equal (flat (vt-digitize (vt-from-sequence '(0.0d0 2.0d0) :dtype :float64)
                                     (vt-from-sequence '(1.0d0) :dtype :float64)))
                  '(0 1)))
    (let* ((scalar (vt-reshape (vt-from-sequence '(1.5d0) :dtype :float64) '()))
           (got (vt-digitize scalar inc)))
      (check "digitize 0 维输入：结果仍是 0 维且值为 5"
             (and (null (vt-shape got)) (equal (flat got) '(5)))
             (format nil "shape=~a flat=~a" (vt-shape got) (flat got))))
    (check-error "digitize 非单调 bins 报错"
                 (lambda () (vt-digitize xs (vt-from-sequence '(0 2 1) :dtype :float64))))
    (check-error "digitize 复数 x 报错"
                 (lambda () (vt-digitize (vt-astype xs :complex128) inc)))
    (check "digitize bins 传 Lisp list"
           (same-tensor-p (vt-digitize xs '(-4 -2 -1 0 1 2 4))
                          (vt-digitize xs inc)))))

;;; ==================================================================
;;; 2. FLOOR / CEILING：浮点取整原语
;;; ==================================================================
(defun old-floor1 (x) (clvt::%floor-family-body1 x floor))
(defun old-ceiling1 (x) (clvt::%floor-family-body1 x ceiling))

(defun floor-suite ()
  (format t "~%=== [2] VT-FLOOR / VT-CEILING：ffloor / fceiling ===~%")
  ;; 边界值：±0.0、denormal、2^53±1、极大/极小、NaN、±Inf、±1 的邻域
  (let ((edge (vt-from-sequence
               (list -0.0d0 0.0d0 -0.5d0 0.5d0 -1.5d0 1.5d0 -2.5d0 2.5d0
                     -1d-310 1d-310
                     4503599627370495.0d0 4503599627370496.0d0 4503599627370497.0d0
                     -4503599627370496.0d0 -4503599627370497.0d0
                     1d300 -1d300 2.2250738585072014d-308 -2.2250738585072014d-308
                     *nan* *pinf* *ninf*)
               :dtype :float64)))
    (check-same "floor   边界值 ≡ 改前展开"
                (vt-floor edge) (vt-fast-map #'old-floor1 edge))
    (check-same "ceiling 边界值 ≡ 改前展开"
                (vt-ceiling edge) (vt-fast-map #'old-ceiling1 edge))
    ;; 一对一：每个边界值的符号零语义（§7.4.8：与 numpy 对齐）
    (let ((f (flat (vt-floor edge)))
          (c (flat (vt-ceiling edge))))
      (check "floor(-0.0) = -0.0（numpy 语义）"
             (eql (sb-kernel:double-float-bits (nth 0 f))
                  (sb-kernel:double-float-bits -0.0d0)))
      (check "floor(+0.0) = +0.0"
             (eql (sb-kernel:double-float-bits (nth 1 f))
                  (sb-kernel:double-float-bits 0.0d0)))
      (check "ceiling(-0.5) = -0.0（numpy 语义）"
             (eql (sb-kernel:double-float-bits (nth 2 c))
                  (sb-kernel:double-float-bits -0.0d0)))
      (check "floor(-0.5) = -1.0" (eql (nth 2 f) -1.0d0))
      (check "ceiling(0.5) = 1.0" (eql (nth 3 c) 1.0d0))
      ;; 2^53 以上：整数中转会经过 bignum，浮点取整不会
      (check "floor(2^53+1) 保持原值" (eql (nth 12 f) 4503599627370497.0d0))
      (check "ceiling(2^53+1) 保持原值" (eql (nth 12 c) 4503599627370497.0d0))
      (check "floor(NaN) = NaN" (nan-p (nth 19 f)))
      (check "floor(+Inf) = +Inf" (eql (nth 20 f) *pinf*))
      (check "ceiling(-Inf) = -Inf" (eql (nth 21 c) *ninf*)))
    ;; 随机语料 + f32 + 整数 + 各种形态
    (let* ((rnd (vt-random-uniform (list 777) :low -50.0d0 :high 50.0d0 :dtype :float64))
           (rnd-int (vt-astype (vt-scale rnd 100.0d0) :int64)))
      (check-same "floor   随机 ≡ 改前" (vt-floor rnd) (vt-fast-map #'old-floor1 rnd))
      (check-same "ceiling 随机 ≡ 改前" (vt-ceiling rnd) (vt-fast-map #'old-ceiling1 rnd))
      (check-same "floor   f32 ≡ 改前"
                  (vt-floor (vt-astype rnd :float32))
                  (vt-fast-map #'old-floor1 (vt-astype rnd :float32)))
      (check-same "floor   整数输入 ≡ 改前"
                  (vt-floor rnd-int) (vt-fast-map #'old-floor1 rnd-int))
      (check-same "floor   转置视图 ≡ 连续副本"
                  (vt-floor (vt-transpose (vt-reshape rnd (list 7 111))))
                  (vt-floor (vt-copy (vt-transpose (vt-reshape rnd (list 7 111))))))
      (check-same "floor   翻转视图 ≡ 连续副本"
                  (vt-floor (vt-flip rnd))
                  (vt-floor (vt-copy (vt-flip rnd))))
      (check-same "floor   :out 契约"
                  (vt-floor rnd :out (vt-zeros (vt-shape rnd)))
                  (vt-fast-map #'old-floor1 rnd))
      ;; DIVISOR≠1 是库扩展，仍走原闭包路径（`%FLOOR-FAMILY-BODY` 的展开
      ;; 引用调用点词法作用域里的 DIVISOR 变量），这条钉住它未被改动。
      (check-same "floor divisor=3 ≡ 逐元素参考"
                  (vt-floor rnd :divisor 3)
                  (vt-map (lambda (x) (nth-value 0 (floor x 3))) rnd)))
    ;; 反向断言：ROUND / TRUNCATE 不许因为「统一换 f 前缀」而变慢
    ;; （fround 实测比 round 慢 6 倍，ftruncate 慢 2 倍 —— 这条钉住不要改）
    (check "round(2.5) = 2（最近偶数，未被 floor 的改动波及）"
           (eql (vt-ref (vt-round (vt-from-sequence '(2.5d0) :dtype :float64)) 0) 2.0d0))
    (check "truncate(-2.7) = -2.0（向零，未被波及）"
           (eql (vt-ref (vt-truncate (vt-from-sequence '(-2.7d0) :dtype :float64)) 0) -2.0d0))))

;;; ==================================================================
;;; 3. NN 三兄弟：SWISH / LEAKY-RELU / HARD-SIGMOID
;;; ==================================================================
(defvar *alpha* 0.01d0)

(defun old-leaky-form (x a)
  (if (> x 0.0d0) x (* a x)))

(declaim (inline old-leaky-df))
(defun old-leaky-df (x)
  "改动前的算子形状：ALPHA 来自特殊变量（每元素一次 SYMBOL-VALUE）。"
  (let ((x (coerce x 'double-float)))
    (old-leaky-form x clvt::*nn-act-param*)))

(defun old-swish (vt) (let ((sig (vt-sigmoid vt))) (vt-map #'* vt sig)))

(defun old-hard-sigmoid (vt)
  (let ((scaled (vt-+ (vt-scale vt 0.2d0) 0.5d0 :dtype :float64)))
    (vt-clip scaled 0.0d0 1.0d0 :dtype :float64)))

(defun nn-suite ()
  (format t "~%=== [3] SWISH / LEAKY-RELU / HARD-SIGMOID ===~%")
  (let ((xs (vt-from-sequence
             (list -1d3 -20d0 -6d0 -3d0 -1d0 -0.5d0 -0.0d0 0.0d0 0.5d0 1d0
                   3d0 6d0 20d0 700d0 1d3 *nan* *pinf* *ninf*)
             :dtype :float64))
        (rnd (vt-random-uniform (list 1000) :low -6.0d0 :high 6.0d0 :dtype :float64)))
    ;; A1 边界值
    (check-same "swish 边界值 ≡ 改前" (vt-swish xs) (old-swish xs))
    (check-same "hard-sigmoid 边界值 ≡ 改前"
                (vt-hard-sigmoid xs) (old-hard-sigmoid xs))
    (let ((clvt::*nn-act-param* 0.01d0))
      (check-same "leaky-relu 边界值 ≡ 改前"
                  (vt-leaky-relu xs)
                  (vt-fast-map #'old-leaky-df xs :dtype :float64)))
    ;; A2 随机语料 × dtype × 形态
    (check-same "swish 随机 ≡ 改前" (vt-swish rnd) (old-swish rnd))
    (check-same "hard-sigmoid 随机 ≡ 改前"
                (vt-hard-sigmoid rnd) (old-hard-sigmoid rnd))
    (let ((clvt::*nn-act-param* 0.01d0))
      (check-same "leaky-relu 随机 ≡ 改前"
                  (vt-leaky-relu rnd)
                  (vt-fast-map #'old-leaky-df rnd :dtype :float64)))
    ;; f32：swish / leaky-relu 走 single 内核；hard-sigmoid 保留三趟
    (let* ((r32 (vt-astype rnd :float32))
           (old-hs32 (lambda (v)
                       (vt-clip (vt-+ (vt-scale v 0.2d0) 0.5d0 :dtype :float32)
                                0.0d0 1.0d0 :dtype :float32))))
      (check "swish f32 保持 float32 dtype"
             (eq (vt-dtype (vt-swish r32)) :float32))
      (check "leaky-relu f32 保持 float32 dtype"
             (eq (vt-dtype (vt-leaky-relu r32)) :float32))
      (check-same "hard-sigmoid f32 ≡ 改前三趟"
                  (vt-hard-sigmoid r32) (funcall old-hs32 r32)))
    ;; 整数输入必须提升到 float64（否则 alpha*x 会被截断）
    (check "leaky-relu 整数输入提升为 float64"
           (eq (vt-dtype (vt-leaky-relu (vt-from-sequence '(-2 3 -4) :dtype :int64)))
               :float64))
    (check "leaky-relu (-2 3 -4) alpha=0.01 → (-0.02 3.0 -0.04)"
           (equal (flat (vt-leaky-relu (vt-from-sequence '(-2 3 -4) :dtype :int64)))
                  '(-0.02d0 3.0d0 -0.04d0)))
    ;; 其它 ALPHA 值
    (dolist (a '(0.0d0 0.1d0 1.0d0 -0.5d0))
      (let ((clvt::*nn-act-param* a))
        (check-same (format nil "leaky-relu alpha=~a ≡ 改前" a)
                    (vt-leaky-relu rnd :alpha a)
                    (vt-fast-map #'old-leaky-df rnd :dtype :float64))))
    ;; 形态：转置 / 翻转 / :out
    (let ((m (vt-reshape rnd (list 25 40))))
      (check-same "swish 转置视图 ≡ 连续副本"
                  (vt-swish (vt-transpose m)) (vt-swish (vt-copy (vt-transpose m))))
      (check-same "hard-sigmoid 翻转视图 ≡ 连续副本"
                  (vt-hard-sigmoid (vt-flip m))
                  (vt-hard-sigmoid (vt-copy (vt-flip m)))))
    (check-same "swish :out 契约"
                (vt-swish rnd :out (vt-zeros (vt-shape rnd))) (old-swish rnd))
    (check-same "hard-sigmoid :out 契约"
                (vt-hard-sigmoid rnd :out (vt-zeros (vt-shape rnd)))
                (old-hard-sigmoid rnd))
    ;; 0 维 / 空
    (check-same "swish 0 维" (vt-swish (vt-full '() 1.5d0)) (old-swish (vt-full '() 1.5d0)))
    (check "swish 空输入" (equal (vt-shape (vt-swish (vt-zeros '(0)))) '(0)))))

;;; ==================================================================
;;; 4. 机器无关倍率护栏
;;; ==================================================================
;;; 目标不是「快多少倍」，而是「一旦有人把改动退回旧写法，这里会红」。
;;; 每个基准都取同一个 REP 数、同一进程、同一份 1M 数据。
(defun perf-suite ()
  (format t "~%=== [4] 倍率护栏（同进程、机器无关）===~%")
  (let* ((n 1000000)
         (v (vt-random-uniform (list n) :low -5.0d0 :high 5.0d0 :dtype :float64))
         (v32 (vt-astype v :float32))
         (xi (vt-astype (vt-scale v 100.0d0) :int64))
         (bins (vt-astype (vt-arange 1001 :start 0 :step 1 :dtype :float64) :float64))
         (bins-i (vt-astype (vt-arange 1001 :start 0 :step 1 :dtype :int64) :int64))
         (reps 7))
    (setf (vt-ref bins 0) -1000.0d0)
    ;; 基准：一次 1M 的 vt-abs（0.77 ms 量级），所有比值都相对它
    (let ((t-abs (best-ms reps (vt-abs v))))
      (format t "  [基准] vt-abs 1M = ~,3f ms~%" t-abs)
      (let ((t-dig (best-ms reps (vt-digitize v bins))))
        (check-at-most "digitize(1M, 1000 bins) ≤ 45× abs" (/ t-dig t-abs) 45.0
                       "digitize ÷ abs"))
      (let ((t-dig-i (best-ms reps (vt-digitize xi bins-i))))
        (check-at-most "digitize(1M i64, 1000 bins) ≤ 70× abs" (/ t-dig-i t-abs) 70.0
                       "digitize-i64 ÷ abs"))
      (let ((t-fl (best-ms reps (vt-floor v))))
        (check-at-most "floor 1M ≤ 6× abs" (/ t-fl t-abs) 6.0 "floor ÷ abs"))
      (let ((t-ce (best-ms reps (vt-ceiling v))))
        (check-at-most "ceiling 1M ≤ 6× abs" (/ t-ce t-abs) 6.0 "ceiling ÷ abs"))
      (let ((t-sw (best-ms reps (vt-swish v))))
        (check-at-most "swish 1M ≤ 25× abs" (/ t-sw t-abs) 25.0 "swish ÷ abs"))
      (let ((t-lr (best-ms reps (vt-leaky-relu v))))
        (check-at-most "leaky-relu 1M ≤ 18× abs" (/ t-lr t-abs) 18.0 "leaky-relu ÷ abs"))
      (let ((t-hs (best-ms reps (vt-hard-sigmoid v))))
        (check-at-most "hard-sigmoid 1M ≤ 20× abs" (/ t-hs t-abs) 20.0 "hard-sigmoid ÷ abs"))
      ;; floor/ceiling 不许慢于同族 round：这是「统一换 f 前缀」的护栏 ——
      ;; fround 比 round 慢 6 倍，若有人把四个函数一起改成 f 前缀，这里会红。
      (let ((t-round (best-ms reps (vt-round v))))
        (check-at-most "floor ≤ 2.0× round" (/ (best-ms reps (vt-floor v)) t-round) 2.0
                       "floor ÷ round")
        (check-at-most "ceiling ≤ 2.0× round" (/ (best-ms reps (vt-ceiling v)) t-round) 2.0
                       "ceiling ÷ round"))
      ;; f32 路径
      (let ((t-sw32 (best-ms reps (vt-swish v32))))
        (check-at-most "swish(f32) ≤ 20× abs(f64)" (/ t-sw32 t-abs) 20.0 "swish-f32 ÷ abs")))))

;;; ==================================================================
;;; 入口
;;; ==================================================================
(defun run ()
  (setf *n* 0 *pass* 0 *fail* 0 *failures* nil)
  (format t "~%########## search-act-perf-test ##########~%")
  (digitize-suite)
  (floor-suite)
  (nn-suite)
  (perf-suite)
  (format t "~%------------------------------------------------------------~%")
  (format t "  Total: ~a | Pass: ~a | Fail: ~a | Skip: 0~%" *n* *pass* *fail*)
  (format t "------------------------------------------------------------~%")
  (when *failures*
    (format t "~%Failed:~{~%  - ~a~}~%" (reverse *failures*)))
  (zerop *fail*))

(sb-ext:exit :code (if (handler-case (run)
                         (error (e) (format t "~&[FAIL] ~a~%" e) nil))
                       0 1))
