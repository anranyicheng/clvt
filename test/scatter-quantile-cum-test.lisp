;;;; scatter-quantile-cum-test.lisp — 分散写 / 分位 / NaN 累积三组「执行层」改写的回归测试
;;;;
;;;; 三处改动，逻辑层一字未动，病根都是「某个通用的东西留在每元素路径上」：
;;;;
;;;;   1. src/indexing.lisp   vt-put（+ 新增 %put-flat-contiguous）
;;;;      通用路径把下标逐个 TRUNCATE 收进 LIST、值也收进 LIST，再对每个元素
;;;;      做「多维下标分解」（每元素 `(reverse shape)` / `(reverse strides)`
;;;;      各分配一个 list）+ `vt-cast`。1M 索引实测 238 ms（numpy 1.2 ms）。
;;;;      连续目标上「逻辑线性下标 ≡ 物理偏移」，三样东西都能提到循环外：
;;;;      1M 实测 2.3 ms。
;;;;
;;;;   2. src/reduce-stats.lisp %percent-double-buffer / src/extensions3.lisp
;;;;      %e3-quantile-buffer
;;;;      `:linear` 分支对**同一个缓冲区**连做两次 `%select-double`：第一次
;;;;      把 K1 就位后，第二次仍从 lo=0/hi=n-1 起选，三数取中的中位数样本
;;;;      恰好退化成「已就位的 K1」，每趟只削掉一个元素 → 累计访问量冲破
;;;;      4N 护栏 → 整段重排。1M 取 25% 分位实测 145 ms（单次 select 只要
;;;;      17 ms，numpy 10.5 ms）。改为 `%select-two-double`：K2 = K1+1 时
;;;;      「尾部最小值」就是答案（选完 K1 后 {buf[0..k1]} 已是最小的 K1+1
;;;;      个元素），一趟线性扫描。1M 实测 16.5 ms。
;;;;
;;;;   3. src/extensions3.lisp %e3-store-double-vector
;;;;      把 double 缓冲写回结果时**不论连续与否**都走 `vt-do-each`（每元素
;;;;      一次多维坐标 + `%cast-to`），且 `tmp` 没有元素类型声明（通用 AREF）。
;;;;      1M 实测 11 ms，而裸拷贝 0.5 ms。`vt-nancumsum` 因此是 13.9 ms
;;;;      （同结构的 `vt-cumsum` 只要 1.1 ms）。补上连续直写与类型声明后
;;;;      1M 实测 3.6 ms。
;;;;
;;;; 断言分两层：
;;;;   A 段 —— **逐位等价**。`vt-put` 与一份逐字复刻的改动前实现逐点比对
;;;;      （覆盖 9 种目标 dtype × 6 种下标 dtype × 3 种 mode × 连续/转置/
;;;;      翻转/带偏移切片 × 单值/循环值/标量/布尔 × 空/0 维下标）；
;;;;      分位与「两次 select 的旧内核」逐位比对（五种 interpolation ×
;;;;      q=0..1 含端点与 NaN）；NaN 累积与按定义手写的参考实现比对。
;;;;   B 段 —— **机器无关的倍率护栏**（§16.16 第 7 条：不写绝对毫秒）。
;;;;      改回旧写法结果照样正确、A 段全绿，只是慢 5~100 倍，所以每条都配
;;;;      一个「相对同进程基准的比值上界」。
;;;;
;;;; 不写手抄期望值 —— 抄期望值只能证明「现在是这样」，挡不住偏移。

(require :asdf)
#+quicklisp (ql:quickload :clvt :silent t)
(asdf:load-system :clvt)

(defpackage :clvt-sqc-test
  (:use :cl :clvt)
  (:import-from :clvt
                #:ensure-vt #:vt-flatten #:vt-data #:vt-size #:vt-offset
                #:vt-strides #:vt-shape #:vt-dtype #:vt-p #:vt-out-writable-p
                #:vt-cast #:%index->int64 #:vt-flatten-sequence
                #:%nan-free-double-buffer #:%select-double
                #:with-float-safe #:%cast-to #:vt-do-each)
  (:export #:run))

(in-package :clvt-sqc-test)

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

(defun same-list-p (a b)
  (and (= (length a) (length b))
       (loop for x in a for y in b always (eqv x y))))

(defun same-tensor-p (a b)
  (and (equal (vt-shape a) (vt-shape b))
       (eq (vt-dtype a) (vt-dtype b))
       (same-list-p (flat a) (flat b))))

(defun short (l)
  (handler-case (subseq l 0 (min 8 (length l)))
    (error () l)))

(defun check-same (label got want)
  (check label (same-tensor-p got want)
         (format nil "got=~a want=~a" (short (flat got)) (short (flat want)))))

(defun check-base-same (label base-a base-b)
  "比较两个「基数组」（VT-PUT 会就地改底层数据，形状/步长可能相同而底层不同）。
   BASE-A / BASE-B 是张量，比值要看它们**底层**的数组。"
  (let ((la (coerce (vt-data base-a) 'list))
        (lb (coerce (vt-data base-b) 'list)))
    (check label (same-list-p la lb)
           (format nil "got=~a want=~a" (short la) (short lb)))))

(defvar *sink* nil)

(defun %elapsed-ms (t0)
  (* 1000.0 (/ (- (get-internal-real-time) t0) internal-time-units-per-second)))

(defun best-ms-fn (thunk reps)
  "把 THUNK 跑 REPS 轮（每轮自动放大到 ≥100 ms），返回**单次调用**的毫秒数。
   三个要点：被测表达式必须包成函数（否则 SBCL 对两次读时钟做 CSE，差值恒 0）；
   必须放大（本机时钟只按 1 ms 跳）；结果写 `*SINK*`（否则可能被当死代码删）。
   与 test/bench-access-view.lisp 的 `%MEASURE` / `%CALIBRATE` 同构。"
  (setf *sink* (funcall thunk))
  (let ((inner 1))
    (loop
      (let ((t0 (get-internal-real-time)))
        (dotimes (i inner) (setf *sink* (funcall thunk)))
        (let ((ms (%elapsed-ms t0)))
          (when (>= ms 100.0) (return))
          (setf inner (max (1+ inner)
                           (min (* inner 8)
                                (ceiling (* inner (/ 150.0 (max ms 1.0))))))))))
    (let ((best most-positive-double-float))
      (dotimes (r reps (/ best inner))
        (let ((t0 (get-internal-real-time)))
          (dotimes (i inner) (setf *sink* (funcall thunk)))
          (setf best (min best (%elapsed-ms t0))))))))

(defmacro best-ms (reps &body body)
  `(best-ms-fn (lambda () ,@body) ,reps))

(defun has-error-p (thunk)
  (handler-case (progn (funcall thunk) nil) (error () t)))

;;; ==================================================================
;;; 1. VT-PUT：与「逐字复刻的改动前实现」逐位对拍
;;; ==================================================================
;;; 改动前的实现：下标与值都收进 LIST，逐元素多维分解 + vt-cast。
(defun old-vt-put (tensor indices values &key (mode :raise))
  (let* ((total (vt-size tensor))
         (idx-list (if (numberp indices) (list (round indices))
                       (let ((flat (vt-flatten (ensure-vt indices))))
                         (loop for i below (vt-size flat)
                               collect (truncate (aref (vt-data flat) i))))))
         (val-list (cond
                     ((or (eq values t) (null values)) (list (if values 1 0)))
                     ((numberp values) (list values))
                     ((or (listp values) (arrayp values))
                      (vt-flatten-sequence values :bool (eq (vt-dtype tensor) :bool)))
                     ((vt-p values)
                      (let ((flat (vt-flatten (ensure-vt values))))
                        (loop for i below (vt-size flat)
                              collect (aref (vt-data flat) i))))))
         (n-values (length val-list)) (shape (vt-shape tensor))
         (strides (vt-strides tensor)) (offset (vt-offset tensor))
         (data (vt-data tensor)))
    (unless (vt-out-writable-p tensor)
      (error "目标只读"))
    (when (zerop n-values) (return-from old-vt-put tensor))
    (loop for raw in idx-list
          for vi from 0
          for val = (nth (mod vi n-values) val-list)
          do (let ((idx raw))
               (cond ((and (>= idx 0) (< idx total)) nil)
                     ((eq mode :clip) (setq idx (max 0 (min idx (1- total)))))
                     ((eq mode :wrap) (setq idx (mod idx total)))
                     ((and (minusp idx) (>= idx (- total))) (incf idx total))
                     (t (error "索引 ~d 超出范围 [0, ~d)" raw total)))
               (let ((rem idx) (phys offset))
                 (loop for dim in (reverse shape)
                       for stride in (reverse strides)
                       do (multiple-value-bind (q r) (floor rem dim)
                            (incf phys (* r stride)) (setf rem q)))
                 (setf (aref data phys) (vt-cast val (vt-dtype tensor))))))
    tensor))

(defun put-dtypes () '(:float64 :float32 :int64 :int32 :int16 :int8 :uint16 :uint8 :bool))

;; 不含 :uint8 —— 它装不下负下标（-1 会变成 255），而本组用例要测负索引折叠。
(defun put-idx-dtypes () '(:int64 :int32 :int16 :int8))

(defun put-suite ()
  (format t "~%=== [1] VT-PUT：连续目标 + 单值的线性下标内核 ===~%")
  ;; ---- 1.1 各种目标 dtype × 各种下标 dtype：单值、越界安全的索引 ----
  (dolist (dt (put-dtypes))
    (dolist (idt (put-idx-dtypes))
      (let* ((n 24)
             (mk (lambda ()
                   (let ((a (vt-zeros (list n) :dtype dt))
                         ;; 整数 / 布尔目标的 VT-REF 写入不做 cast，直接给整数
                         (float-p (member dt '(:float64 :float32))))
                     (loop for i below n
                           do (setf (vt-ref a i)
                                    (if float-p (* 1.0d0 i)
                                        (if (eq dt :bool) (if (evenp i) 0 1) i))))
                     a)))
             (idx (vt-astype (vt-from-sequence '(0 3 5 23 -1 -24 7) :dtype :int64) idt))
             (a1 (funcall mk)) (a2 (funcall mk)))
        (vt-put a1 idx 9.0d0)
        (old-vt-put a2 idx 9.0d0)
        (check (format nil "put 目标=~a 下标=~a ≡ 改前" dt idt)
               (same-tensor-p a1 a2)
               (format nil "got=~a want=~a" (short (flat a1)) (short (flat a2)))))))
  ;; ---- 1.2 多维目标（2 维 / 3 维）与「线性下标 ≡ 物理偏移」 ----
  (dolist (shape '((2 3) (2 3 4) (1 6) (6 1)))
    (let* ((n (reduce #'* shape))
           (idx (vt-from-sequence (list 0 (1- n) (floor n 2)) :dtype :int64))
           (a1 (vt-zeros shape :dtype :float64))
           (a2 (vt-zeros shape :dtype :float64)))
      (vt-put a1 idx 5.0d0) (old-vt-put a2 idx 5.0d0)
      (check (format nil "put 形状 ~a ≡ 改前" shape) (same-tensor-p a1 a2))))
  ;; ---- 1.3 三种 mode（含负索引折叠 / 裁剪 / 取模） ----
  (dolist (mode '(:raise :clip :wrap))
    (dolist (idxs (list '(0 1 2) '(-1 -2) '(-7 9) '(5 -5 100 -100)))
      (let ((a1 (vt-zeros '(6) :dtype :float64))
            (a2 (vt-zeros '(6) :dtype :float64))
            (idx (vt-from-sequence idxs :dtype :int64)))
        (let ((e1 (has-error-p (lambda () (vt-put a1 idx 3.0d0 :mode mode))))
              (e2 (has-error-p (lambda () (old-vt-put a2 idx 3.0d0 :mode mode)))))
          (check (format nil "put mode=~a idx=~a 报错行为一致" mode idxs) (eq e1 e2))
          (unless e1
            (check (format nil "put mode=~a idx=~a ≡ 改前" mode idxs)
                   (same-tensor-p a1 a2)))))))
  ;; ---- 1.4 值的四种形态：标量 / T / 列表循环 / 张量循环 ----
  (dolist (vals (list 2.5d0 t nil
                      (list 1.0d0 2.0d0 3.0d0)
                      (vt-from-sequence '(7.0d0 8.0d0) :dtype :float64)))
    (let ((a1 (vt-zeros '(7) :dtype :float64))
          (a2 (vt-zeros '(7) :dtype :float64))
          (idx (vt-from-sequence '(0 1 2 3 4 5 6) :dtype :int64)))
      (vt-put a1 idx vals) (old-vt-put a2 idx vals)
      (check (format nil "put 值形态 ~a ≡ 改前"
                     (cond ((eq vals t) "T") ((null vals) "NIL")
                           ((vt-p vals) "张量") ((listp vals) "列表") (t "标量")))
             (same-tensor-p a1 a2))))
  ;; ---- 1.5 非连续目标：转置 / 翻转 / 带偏移切片 ----
  (flet ((pair (make)
           (let ((b1 (vt-zeros '(4 5) :dtype :float64))
                 (b2 (vt-zeros '(4 5) :dtype :float64)))
             (loop for i below 20
                   do (setf (vt-ref b1 (floor i 5) (mod i 5)) (* 1.0d0 i)))
             (loop for i below 20
                   do (setf (vt-ref b2 (floor i 5) (mod i 5)) (* 1.0d0 i)))
             (values (funcall make b1) (funcall make b2) b1 b2))))
    (dolist (spec (list (cons "转置" (lambda (b) (vt-transpose b)))
                        (cons "翻转" (lambda (b) (vt-flip b :axis 1)))
                        (cons "切片偏移" (lambda (b) (vt-narrow b 0 1 3)))))
      (multiple-value-bind (t1 t2 b1 b2) (pair (cdr spec))
        (let ((idx (vt-from-sequence '(0 3 9) :dtype :int64)))
          (vt-put t1 idx 4.0d0) (old-vt-put t2 idx 4.0d0)
          (check-base-same (format nil "put ~a目标 ≡ 改前（含底层数据）" (car spec)) b1 b2)))))
  ;; ---- 1.6 空下标 / 0 维下标 / 标量下标 ----
  (let ((a1 (vt-zeros '(4) :dtype :float64)) (a2 (vt-zeros '(4) :dtype :float64)))
    (vt-put a1 (vt-zeros '(0) :dtype :int64) 1.0d0)
    (old-vt-put a2 (vt-zeros '(0) :dtype :int64) 1.0d0)
    (check "put 空下标是空操作" (same-tensor-p a1 a2)))
  (let ((a1 (vt-zeros '(4) :dtype :float64)) (a2 (vt-zeros '(4) :dtype :float64))
        (i0 (vt-astype (vt-full '() 2.0d0) :int64)))
    (vt-put a1 i0 6.0d0) (old-vt-put a2 i0 6.0d0)
    (check "put 0 维下标张量 ≡ 改前" (same-tensor-p a1 a2)))
  (let ((a1 (vt-zeros '(4) :dtype :float64)) (a2 (vt-zeros '(4) :dtype :float64)))
    (vt-put a1 3 6.0d0) (old-vt-put a2 3 6.0d0)
    (check "put 标量下标 ≡ 改前" (same-tensor-p a1 a2)))
  ;; ---- 1.7 越界报错仍然报（快路径不许把检查省掉） ----
  (let ((a (vt-zeros '(4) :dtype :float64)))
    (check "put 下标 = total 报错"
           (has-error-p (lambda () (vt-put a (vt-from-sequence '(4) :dtype :int64) 1.0d0))))
    (check "put 下标 = -total-1 报错"
           (has-error-p (lambda () (vt-put a (vt-from-sequence '(-5) :dtype :int64) 1.0d0))))
    (check "put 浮点下标报错"
           (has-error-p (lambda () (vt-put a (vt-from-sequence '(1.0d0) :dtype :float64) 1.0d0)))))
  ;; ---- 1.8 dtype 转换与改前一致（cast 提到循环外不许改数值） ----
  (dolist (v (list 2.9d0 -2.9d0 0.5d0 -0.5d0 1d10 -1d10))
    (dolist (dt '(:int8 :int16 :uint8 :bool :float32))
      (let ((a1 (vt-zeros '(2) :dtype dt)) (a2 (vt-zeros '(2) :dtype dt))
            (idx (vt-from-sequence '(0 1) :dtype :int64)))
        (vt-put a1 idx v) (old-vt-put a2 idx v)
        (check (format nil "put cast ~a → ~a ≡ 改前" v dt) (same-tensor-p a1 a2))))))

;;; ==================================================================
;;; 2. 分位：与「两次独立 select」的旧内核逐位对拍
;;; ==================================================================
(defun old-percent-double-buffer (buf m q interpolation)
  "改动前的 `%percent-double-buffer`：`:linear` 与 `:midpoint` 各做两次独立 select。"
  (declare (type (simple-array double-float (*)) buf) (type fixnum m)
           (type double-float q))
  (let* ((idx (* q (1- m)))
         (lower (floor idx))
         (upper (min (ceiling idx) (1- m)))
         (frac (- idx lower)))
    (declare (type fixnum lower upper) (type double-float frac))
    (case interpolation
      (:linear (if (= lower upper)
                   (%select-double buf m lower)
                   (+ (* (- 1.0d0 frac) (%select-double buf m lower))
                      (* frac (%select-double buf m upper)))))
      (:lower (%select-double buf m lower))
      (:higher (%select-double buf m upper))
      (:midpoint (/ (+ (%select-double buf m lower)
                       (%select-double buf m upper))
                    2.0d0))
      (:nearest (%select-double buf m (if (<= frac 0.5d0) lower upper))))))

(defun old-e3-quantile-buffer (buf m q interpolation)
  "改动前的 `%e3-quantile-buffer`：`:linear` 分支对 lo / hi 各选一次，
   而 `(select lo)` 在同一个缓冲区上被算了**两遍**（共三次 select）。"
  (declare (type (simple-array double-float (*)) buf)
           (type fixnum m) (type double-float q))
  (let* ((pos (* q (1- m)))
         (lo (floor pos))
         (hi (min (1+ lo) (1- m)))
         (frac (- pos lo)))
    (declare (type fixnum lo hi) (type double-float frac))
    (case interpolation
      (:lower (%select-double buf m lo))
      (:higher (%select-double buf m hi))
      (:nearest (%select-double buf m (round pos)))
      (:midpoint (/ (+ (%select-double buf m lo) (%select-double buf m hi)) 2.0d0))
      (t (+ (%select-double buf m lo)
            (* frac (- (%select-double buf m hi) (%select-double buf m lo))))))))

(defun old-percentile-global (tensor q interpolation)
  "改动前的全局分位（缓冲区准备没变，只有 select 次数变了）。"
  (multiple-value-bind (buf m has-nan) (%nan-free-double-buffer tensor)
    (if (or has-nan (zerop m))
        *nan*
        (old-percent-double-buffer buf m (coerce q 'double-float) interpolation))))

(defun q-roundtrip (q)
  "复现 `vt-quantile` → `vt-percentile` 的那次换算：`(/ (* q 100) 100.0d0)`。
   1/3 这类值经这一次往返会变最后一位，参考实现必须走同一条路。"
  (coerce (/ (* q 100) 100.0d0) 'double-float))

(defun item0 (x)
  "取出标量结果：0 维张量用无下标的 VT-REF，1 维用下标 0。"
  (if (null (vt-shape x)) (vt-ref x) (vt-ref x 0)))

(defparameter *pct-qs* (list 0 0.01d0 0.1d0 0.25d0 (/ 1.0d0 3.0d0) 0.5d0 0.75d0 0.9d0 0.99d0 1))
(defparameter *pct-interps* '(:linear :lower :higher :midpoint :nearest))

(defun pct-suite ()
  (format t "~%=== [2] 分位：一次 select + 尾部最小值 ===~%")
  (let* ((n 997)                       ; 质数长度：让 q*(n-1) 落在各种位置
         (vals (append (list *nan* 3.0d0 -3.0d0 0.0d0 -0.0d0 1d300 -1d300)
                       (loop for i below n collect (- (random 100.0d0) 50.0d0))))
         (clean (remove-if #'nan-p vals))
         (v (vt-from-sequence clean :dtype :float64))
         (vnan (vt-from-sequence vals :dtype :float64)))
    ;; ---- 2.1 全局：五种 interpolation × 十个 q ----
    (dolist (interp *pct-interps*)
      (let ((bad 0))
        (dolist (q *pct-qs*)
          (let ((got (item0 (vt-quantile v q :interpolation interp)))
                (want (old-percentile-global v (q-roundtrip q) interp)))
            (unless (eqv got want) (incf bad))))
        (check (format nil "quantile 全局 ~a：全部 q 逐位 ≡ 改前" interp)
               (zerop bad) (format nil "~a 个 q 不一致" bad))))
    ;; ---- 2.2 含 NaN：整段视为 NaN ----
    (dolist (interp *pct-interps*)
      (let ((got (item0 (vt-quantile vnan 0.25d0 :interpolation interp))))
        (check (format nil "quantile 含 NaN → NaN（~a）" interp) (nan-p got))))
    ;; ---- 2.3 各 dtype 输入（float32 / int64 / int8 / bool）----
    (dolist (dt '(:float32 :int64 :int32 :int8 :uint8))
      (let* ((src (vt-astype (vt-scale v 3.0d0) dt))
             (bad 0))
        (dolist (q '(0.1d0 0.25d0 0.5d0 0.9d0))
          (unless (eqv (item0 (vt-quantile src q))
                       (old-percentile-global src (q-roundtrip q) :linear))
            (incf bad)))
        (check (format nil "quantile ~a 输入逐位 ≡ 改前" dt) (zerop bad))))
    ;; ---- 2.4 沿轴：与「逐行 / 逐列取全局分位」对拍 ----
    (let* ((sq (vt-narrow v 0 0 992))       ; 31 × 32 = 992
           (m (vt-reshape sq '(31 32)))
           (bad1 0) (bad2 0))
      (dolist (interp *pct-interps*)
        (let ((got1 (vt-percentile m 25 :axis 1 :interpolation interp))
              (got0 (vt-percentile m 25 :axis 0 :interpolation interp)))
          (dotimes (i 31)
            (unless (eqv (vt-ref got1 i)
                         (old-percentile-global (vt-narrow m 0 i (1+ i)) (q-roundtrip 0.25d0) interp))
              (incf bad1)))
          (dotimes (j 32)
            (unless (eqv (vt-ref got0 j)
                         (old-percentile-global (vt-narrow m 1 j (1+ j)) (q-roundtrip 0.25d0) interp))
              (incf bad2)))))
      (check "percentile axis=1 逐行 ≡ 改前" (zerop bad1))
      (check "percentile axis=0 逐列 ≡ 改前" (zerop bad2)))
    ;; ---- 2.5 keepdims 形状 ----
    (check "percentile keepdims 形状"
           (equal (vt-shape (vt-percentile (vt-reshape (vt-narrow v 0 0 992) '(31 32))
                                           25 :axis 0 :keepdims t))
                  '(1 32)))
    ;; ---- 2.6 分位序列：逐个算再堆叠，与逐个单值调用一致 ----
    (let ((seq (vt-percentile v (list 10 50 90)))
          (bad 0))
      (dolist (q '(10 50 90))
        (unless (eqv (vt-ref seq (position q '(10 50 90)))
                     (item0 (vt-percentile v q)))
          (incf bad)))
      (check "percentile 序列 ≡ 逐个单值调用" (zerop bad)))
    ;; ---- 2.7 长数组（1M）：与旧内核逐位对拍一次（含 :midpoint，两个 select） ----
    (let* ((big (vt-random-uniform (list 100000) :low -1.0d0 :high 1.0d0 :dtype :float64))
           (bad 0))
      (dolist (q '(0.25d0 0.5d0 0.75d0 0.9d0))
        (dolist (interp '(:linear :midpoint))
          (unless (eqv (item0 (vt-quantile big q :interpolation interp))
                       (old-percentile-global big (q-roundtrip q) interp))
            (incf bad))))
      (check "quantile 100k 随机数据逐位 ≡ 改前" (zerop bad)))
    ;; ---- 2.8 %e3-quantile-buffer（nan 分位的内核）：与「三次 select」的旧写法对拍 ----
    (let ((bad 0))
      (dotimes (trial 20)
        (let* ((m (+ 2 (random 400)))
               (buf (make-array m :element-type 'double-float)))
          (dotimes (i m) (setf (aref buf i) (- (random 1000.0d0) 500.0d0)))
          (dolist (q '(0.0d0 0.1d0 0.25d0 0.5d0 0.73d0 0.9d0 1.0d0))
            (dolist (interp *pct-interps*)
              (let ((c1 (copy-seq buf)) (c2 (copy-seq buf)))
                (unless (eqv (clvt::%e3-quantile-buffer c1 m q interp)
                             (old-e3-quantile-buffer c2 m q interp))
                  (incf bad)))))))
      (check "%e3-quantile-buffer ≡ 三次 select 的旧写法" (zerop bad)))
    ;; ---- 2.9 nanpercentile：插不插 NaN 结果不变（NaN 只被跳过） ----
    (let* ((vals2 (append (list 3.0d0 -3.0d0 5.0d0 1d300 -1d300)
                          (loop for i below 500 collect (- (random 100.0d0) 50.0d0))))
           (with-nan (append (list *nan*) vals2 (list *nan*)))
           (a (vt-from-sequence with-nan :dtype :float64))
           (b (vt-from-sequence vals2 :dtype :float64))
           (bad 0))
      (dolist (q '(25.0d0 50.0d0 90.0d0))
        (dolist (interp *pct-interps*)
          (unless (eqv (item0 (vt-nanpercentile a q :interpolation interp))
                       (item0 (vt-nanpercentile b q :interpolation interp)))
            (incf bad))))
      (check "nanpercentile 插不插 NaN 结果相同" (zerop bad)))))

;;; ==================================================================
;;; 3. NaN 累积：与「按定义手写的参考实现」逐位对拍
;;; ==================================================================
(defun ref-nancum (values op &key (init (if (eq op :sum) 0.0d0 1.0d0)))
  "按 numpy 语义手写：首元素掩码成单位元，其后遇 NaN 跳过、否则累积。"
  (let ((acc (coerce init 'double-float))
        (out '()))
    (loop for i from 0
          for x in values
          do (if (zerop i)
                 (setf acc (if (nan-p x) (coerce init 'double-float)
                               (coerce x 'double-float)))
                 (unless (nan-p x)
                   (setf acc (if (eq op :sum)
                                 (+ acc (coerce x 'double-float))
                                 (* acc (coerce x 'double-float))))))
             (push acc out))
    (nreverse out)))

(defun ref-nancum-axis (rows op)
  "沿轴 1 的参考实现：逐行调用 REF-NANCUM。"
  (apply #'append (mapcar (lambda (r) (ref-nancum r op)) rows)))

(defun rows-of (m)
  (let* ((shape (vt-shape m)) (r (first shape)) (c (second shape)))
    (loop for i below r
          collect (loop for j below c collect (vt-ref m i j)))))

(defun cum-suite ()
  (format t "~%=== [3] NaN 累积：连续直写 + tmp 类型声明 ===~%")
  (let* ((vals (list 1.0d0 *nan* 2.0d0 3.0d0 *nan* -1.0d0 0.0d0 -0.0d0 1d300))
         (v (vt-from-sequence vals :dtype :float64)))
    ;; ---- 3.1 全局：与手写参考逐位相同 ----
    (check "nancumsum ≡ 定义（f64）"
           (same-list-p (flat (vt-nancumsum v)) (ref-nancum vals :sum)))
    (check "nancumprod ≡ 定义（f64）"
           (same-list-p (flat (vt-nancumprod v)) (ref-nancum vals :prod)))
    ;; 各输入 dtype：库里的结果是「double 累积后逐步写回该 dtype」，
    ;; 参考实现因此也要按**结果 dtype** 收敛再比（f32 的每一步都要舍入）。
    (dolist (dt '(:float32 :float64 :int64 :int32 :int16 :uint8))
      (let* ((src (vt-astype (vt-from-sequence '(3 -1 4 1 5 9 2 6) :dtype :int64) dt))
             (xs (flat src))
             (out-dt (vt-dtype (vt-nancumsum src))))
        (check (format nil "nancumsum ~a ≡ 定义（按 ~a 收敛）" dt out-dt)
               (same-list-p (flat (vt-nancumsum src))
                            (mapcar (lambda (x) (vt-cast x out-dt)) (ref-nancum xs :sum))))
        (check (format nil "nancumprod ~a ≡ 定义（按 ~a 收敛）" dt out-dt)
               (same-list-p (flat (vt-nancumprod src))
                            (mapcar (lambda (x) (vt-cast x out-dt)) (ref-nancum xs :prod))))))
    ;; 首元素是 NaN / -0.0 的边界
    (dolist (head (list *nan* -0.0d0 0.0d0))
      (let ((xs (list head 1.0d0 2.0d0)))
        (check (format nil "nancumsum 首元素=~a ≡ 定义" head)
               (same-list-p (flat (vt-nancumsum (vt-from-sequence xs :dtype :float64)))
                            (ref-nancum xs :sum)))))
    ;; ---- 3.3 axis=nil 对 2 维输入必须展平（形状） ----
    (let ((m (vt-reshape (vt-from-sequence '(1.0d0 2.0d0 3.0d0 4.0d0) :dtype :float64) '(2 2))))
      (check "nancumsum 2 维无 axis → 形状 (4)"
             (equal (vt-shape (vt-nancumsum m)) '(4)))
      ;; ---- 3.4 沿轴：与逐行参考相同 ----
      (let* ((mm (vt-reshape (vt-from-sequence
                              (list 1.0d0 *nan* 2.0d0 3.0d0
                                    4.0d0 5.0d0 *nan* 6.0d0) :dtype :float64)
                             '(2 4)))
             (rows (rows-of mm)))
        (check "nancumsum axis=1 ≡ 逐行定义"
               (same-list-p (flat (vt-nancumsum mm :axis 1)) (ref-nancum-axis rows :sum)))
        (check "nancumprod axis=1 ≡ 逐行定义"
               (same-list-p (flat (vt-nancumprod mm :axis 1)) (ref-nancum-axis rows :prod)))))
    ;; ---- 3.5 :dtype 目标（整数 / 布尔 / 单精度）----
    (dolist (dt '(:float32 :int64 :int32 :int8 :uint8 :bool))
      (let ((got (vt-nancumsum v :dtype dt)))
        (check (format nil "nancumsum :dtype ~a 结果 dtype" dt) (eq (vt-dtype got) dt))
        (check (format nil "nancumsum :dtype ~a 值 ≡ 定义（按 ~a 收敛）" dt dt)
               (same-list-p (mapcar (lambda (x) (if (eq dt :bool) (if x 1 0) x)) (flat got))
                            (mapcar (lambda (x)
                                      ;; :bool 写回是**真值化**（2.5 → 1），
                                      ;; 其余 dtype 是截断 —— 与库内一致。
                                      (if (eq dt :bool) (if (zerop x) 0 1)
                                          (vt-cast x dt)))
                                    (ref-nancum vals :sum))))))
    ;; ---- 3.6 :out 契约（连续 / 非连续目标）----
    (let ((o (vt-zeros (list (vt-size v)) :dtype :float64)))
      (vt-nancumsum v :out o)
      (check "nancumsum :out 连续 ≡ 定义"
             (same-list-p (flat o) (ref-nancum vals :sum))))
    (let ((o (vt-flip (vt-zeros (list (vt-size v)) :dtype :float64))))
      (vt-nancumsum v :out o)
      (check "nancumsum :out 非连续（列视图）≡ 定义"
             (same-list-p (flat o) (ref-nancum vals :sum))))
    ;; ---- 3.7 空输入 ----
    (check "nancumsum 空输入 → 形状 (0)"
           (equal (vt-shape (vt-nancumsum (vt-zeros '(0)))) '(0)))))

;;; ==================================================================
;;; 4. 机器无关倍率护栏
;;; ==================================================================
(defun perf-suite ()
  (format t "~%=== [4] 倍率护栏（同进程、机器无关）===~%")
  (let* ((n 1000000)
         (v (vt-random-uniform (list n) :low -5.0d0 :high 5.0d0 :dtype :float64))
         (idx (vt-astype (vt-scale v 150000.0d0) :int64))
         (reps 7))
    ;; 基准：一次 1M 的 vt-abs
    (let ((t-abs (best-ms reps (vt-abs v))))
      (format t "  [基准] vt-abs 1M = ~,3f ms~%" t-abs)
      ;; put 不许慢过同规模的 gather（旧写法是它的 80 倍）
      (let ((t-take (best-ms reps (vt-take v idx))))
        (let ((t-put (best-ms reps (vt-put (vt-copy v) idx 0.0d0))))
          (check-at-most "put(1M) ≤ 3× take(1M)" (/ t-put t-take) 3.0 "put ÷ take")))
      ;; 分位不许慢过整段排序（旧写法 25% 分位比排序还慢）
      (let ((t-sort (best-ms 3 (vt-sort v))))
        (let ((t-p25 (best-ms reps (vt-percentile v 25))))
          (check-at-most "percentile(25) ≤ 0.5× sort" (/ t-p25 t-sort) 0.5
                         "percentile25 ÷ sort"))
        (let ((t-p90 (best-ms reps (vt-percentile v 90))))
          (check-at-most "percentile(90) ≤ 0.5× sort" (/ t-p90 t-sort) 0.5
                         "percentile90 ÷ sort")))
      ;; NaN 累积不许比无 NaN 版本慢太多（旧写法 12 倍）
      (let ((t-cum (best-ms reps (vt-cumsum v))))
        (let ((t-ncum (best-ms reps (vt-nancumsum v))))
          (check-at-most "nancumsum ≤ 6× cumsum" (/ t-ncum t-cum) 6.0
                         "nancumsum ÷ cumsum"))
        (let ((t-nprod (best-ms reps (vt-nancumprod v))))
          (check-at-most "nancumprod ≤ 6× cumsum" (/ t-nprod t-cum) 6.0
                         "nancumprod ÷ cumsum"))))))

;;; ==================================================================
;;; 入口
;;; ==================================================================
(defun run ()
  (setf *n* 0 *pass* 0 *fail* 0 *failures* nil)
  (format t "~%########## scatter-quantile-cum-test ##########~%")
  (put-suite)
  (pct-suite)
  (cum-suite)
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
