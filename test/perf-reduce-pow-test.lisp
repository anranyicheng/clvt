;;;; perf-reduce-pow-test.lisp — 标量幂专用循环 + 归约内核优化的
;;;;                           语义保持 / 选路结构 / 倍率护栏
;;;;
;;;; 为什么需要这套：本轮动的三处都是「只影响速度」的路径，写错的形态
;;;; 全都是「看起来对、其实某个输入上错」：
;;;;
;;;;   1. `%POW-SCALAR-RUN`（`vt-pow` 的标量指数专用循环）。它绕开了
;;;;      `vt-fast-map` 的内联机制自己分派类型，所以必须证明：
;;;;        · 结果与 `(expt x power)` **逐位相同**（不做任何 coerce ——
;;;;          整数指数仍走 CL 的整数幂，不会被悄悄转成浮点指数；
;;;;          也没有退化成 `x*x`，那是 numpy `fast_scalar_power` 的取舍，
;;;;          实测 `pow(x,2)` 与 `x*x` 有 0.05% 的输入位差、`pow(x,3)` 8.7%）；
;;;;        · 指数类型与元素类型不匹配时**不接管**（f32 张量 × double 指数
;;;;          在 `(expt single double)` 与 `(expt single single)` 之间
;;;;          有中间精度差，属数值变更而非提速）。
;;;;   2. 归约内核的 `(locally (declare (optimize (speed 3) (safety 0))))`
;;;;      与指针递增。它们不改变累加顺序，但关掉 safety 后任何一处下标
;;;;      算错都会静默读错内存，所以这里全部与**朴素顺序参考实现**逐位对拍。
;;;;   3. `%OP-STEP` 的 max/min 判定重排。跳过 NaN（nanmax）的累加器
;;;;      初值**本身就是 NaN**（全 NaN 时结果即 NaN），重排时最容易漏掉
;;;;      「acc 是 NaN 就替换」这一步 —— 漏了就是 nanmax 永远返回 NaN。
;;;;      core-test 的 `vt-nanmax` 一条就抓到过，这里把它扩成系统测试。
;;;;
;;;; 用法：sbcl --dynamic-space-size 4096 --non-interactive \
;;;;             --load test/perf-reduce-pow-test.lisp

(setf *read-default-float-format* 'single-float)
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

;;; ------------------------------------------------------------------
;;; 逐位比较辅助
;;; ------------------------------------------------------------------

(defun df-bits (x)
  (sb-kernel:double-float-bits (coerce x 'double-float)))

(defun sf-bits (x)
  (sb-kernel:single-float-bits (coerce x 'single-float)))

(defun df-bits= (a b)
  "逐位相等（NaN 也按位比 —— NaN ≠ NaN，用 = 会漏报）。"
  (= (df-bits a) (df-bits b)))

(defun sf-bits= (a b)
  (= (sf-bits a) (sf-bits b)))

;;; ------------------------------------------------------------------
;;; 朴素顺序参考实现（与库的累加顺序一致：沿归约轴 r = 0,1,2,… 递增）
;;; ------------------------------------------------------------------

(defun ref-sum (xs)
  (let ((a 0.0d0)) (declare (type double-float a))
    (dolist (v xs a) (setf a (+ a (coerce v 'double-float))))))

(defun ref-prod (xs)
  (let ((a 1.0d0)) (declare (type double-float a))
    (dolist (v xs a) (setf a (* a (coerce v 'double-float))))))

(defun ref-max (xs)
  (let ((a (coerce (first xs) 'double-float))) (declare (type double-float a))
    (dolist (v (rest xs) a)
      (let ((v (coerce v 'double-float))) (when (> v a) (setf a v))))))

(defun ref-min (xs)
  (let ((a (coerce (first xs) 'double-float))) (declare (type double-float a))
    (dolist (v (rest xs) a)
      (let ((v (coerce v 'double-float))) (when (< v a) (setf a v))))))

(defun rows-of (vt)
  "2 维张量 → list of lists。"
  (let ((lst (vt-to-list vt)))
    (if (and (consp lst) (consp (car lst))) lst (mapcar #'list lst))))

(defun cols-of (vt)
  "2 维张量 → 转置后的 list of lists（每行是一列，顺序即行序）。"
  (let* ((rows (rows-of vt)) (n (length (first rows))))
    (loop for j below n collect (mapcar (lambda (r) (nth j r)) rows))))

(defun flat-of (vt)
  (let ((lst (vt-to-list vt)))
    (if (and (consp lst) (consp (car lst))) (apply #'append lst) lst)))

;;; ==================================================================
;;; A. `vt-pow` 标量指数：与逐元素 `expt` 逐位相同
;;; ==================================================================

(defparameter +pow-specials+
  (list -3.0d0 -2.0d0 -1.0d0 -0.5d0 -0.0d0 0.0d0 0.5d0 1.0d0 2.0d0 3.0d0
        1.0d300 1.0d-300 1.0000001d0 0.9999999d0
        sb-ext:double-float-positive-infinity
        sb-ext:double-float-negative-infinity
        +vt-dfloat-nan+))

(defun expected-pow-df (x p)
  "`vt-pow` 的语义参考：`(expt x p)`，非实数结果 → 规范 NaN。"
  (let ((r (expt x p)))
    (if (realp r) (coerce r 'double-float) +vt-dfloat-nan+)))

(defun test-pow-df (label power)
  (let* ((v (vt-from-sequence +pow-specials+ :dtype :float64))
         (got (vt-to-list (vt-pow v power)))
         (bad 0))
    (loop for x in +pow-specials+ for g in got
          unless (df-bits= g (expected-pow-df x power)) do (incf bad))
    (check label (zerop bad))))

(sb-int:with-float-traps-masked (:invalid :divide-by-zero :overflow)
  (test-pow-df "A1 f64 pow 2.0d0 ≡ expt（特殊值逐位）" 2.0d0)
  (test-pow-df "A2 f64 pow 0.5d0 ≡ expt（特殊值逐位）" 0.5d0)
  (test-pow-df "A3 f64 pow -1.0d0 ≡ expt（特殊值逐位）" -1.0d0)
  (test-pow-df "A4 f64 pow 3.0d0 ≡ expt（特殊值逐位）" 3.0d0)
  (test-pow-df "A5 f64 pow 2（整数指数）≡ expt" 2)
  (test-pow-df "A6 f64 pow 0（整数指数）≡ expt" 0)
  (test-pow-df "A7 f64 pow -3（整数指数）≡ expt" -3)

  ;; 长数组（走专用循环主路径，而不是特殊值集合）
  (let* ((n 5000)
         (xs (loop for i below n
                   collect (* 1.0d-3 (+ 1d0 (* i 7))) 
                   collect (- (* 1.0d-3 (+ 1d0 (* i 7))))))
         (v (vt-from-sequence xs :dtype :float64))
         (got (vt-to-list (vt-pow v 2.0d0)))
         (bad 0))
    (loop for x in xs for g in got
          unless (df-bits= g (expected-pow-df x 2.0d0)) do (incf bad))
    (check "A8 f64 pow 2.0d0 ≡ expt（5000 元素含正负）" (zerop bad)))

  ;; f32：只在与指数类型匹配时才走专用循环
  (let* ((xs (list -3.0d0 -1.0d0 -0.0d0 0.0d0 0.5d0 1.0d0 2.0d0 3.0d0))
         (v (vt-astype (vt-from-sequence xs :dtype :float64) :float32))
         (got (vt-to-list (vt-pow v 2.0 :dtype :float32)))
         (bad 0))
    (loop for x in xs for g in got
          for e = (expt (coerce x 'single-float) (coerce 2.0 'single-float))
          unless (sf-bits= g (if (realp e) e +vt-sfloat-nan+)) do (incf bad))
    (check "A9 f32 pow 2.0（single 指数）≡ expt" (zerop bad))
    (check "A10 f32 结果 dtype 保持 :float32"
           (eq (vt-dtype (vt-pow v 2.0 :dtype :float32)) :float32))))

;;; ==================================================================
;;; B. `%POW-SCALAR-RUN` 的接管判据（必须保守：不匹配就回退）
;;; ==================================================================

(let* ((z (vt-from-sequence (list 1.0d0 2.0d0 3.0d0 4.0d0) :dtype :float64))
       (out (vt-zeros '(4) :dtype :float64))
       (f32 (vt-astype z :float32))
       (out32 (vt-zeros '(4) :dtype :float32))
       (rev (vt-slice z '(3 nil -1)))
       (i64 (vt-astype z :int64))
       (outi (vt-zeros '(4) :dtype :int64)))
  (check "B1 连续 f64 + double 指数 → 接管"
         (eq t (%pow-scalar-run z out 2.0d0)))
  (check "B2 连续 f64 + 整数指数 → 接管"
         (eq t (%pow-scalar-run z out 2)))
  (check "B3 连续 f64 + single 指数 → 回退（类型不匹配）"
         (null (%pow-scalar-run z out 2.0)))
  (check "B4 连续 f32 + single 指数 → 接管"
         (eq t (%pow-scalar-run f32 out32 (coerce 2.0 'single-float))))
  (check "B5 连续 f32 + double 指数 → 回退（中间精度不同）"
         (null (%pow-scalar-run f32 out32 2.0d0)))
  (check "B6 非连续输入 → 回退"
         (null (%pow-scalar-run rev out 2.0d0)))
  (check "B7 整数 dtype → 回退（本身由整数幂分支处理）"
         (null (%pow-scalar-run i64 outi 2)))
  (check "B8 输出 dtype 与输入不一致 → 回退"
         (null (%pow-scalar-run z out32 2.0d0)))
  ;; 原地更新（同一视图）也要能接管，且结果正确
  (let ((a (vt-from-sequence (list 2.0d0 3.0d0 4.0d0) :dtype :float64)))
    (vt-pow a 2.0d0 :out a)
    (check "B9 原地 (vt-pow a 2.0d0 :out a) 接管并正确"
           (every #'df-bits= (vt-to-list a) (list 4.0d0 9.0d0 16.0d0)))))

;;; ==================================================================
;;; C. 归约：与朴素顺序参考实现逐位相同
;;; ==================================================================

(defun make-test-matrix ()
  "7×5 的非平凡数据：正负、大小混合，保证求和/求积有舍入。"
  (let ((rows '()))
    (dotimes (i 7)
      (push (loop for j below 5
                  collect (* 1.0d0 (if (evenp (+ i j)) 1 -1)
                             (+ 1 (* 3 i)) (/ (+ 1 j) 7.0d0)))
            rows))
    (vt-from-sequence (nreverse rows) :dtype :float64)))

(let* ((m (make-test-matrix))
       (rows (rows-of m))
       (cols (cols-of m))
       (flat (flat-of m)))
  ;; 沿 axis=1（每行）
  (check "C1 sum axis=1 逐位 ≡ 顺序参考"
         (every #'df-bits=
                (vt-to-list (vt-sum m :axis 1))
                (mapcar #'ref-sum rows)))
  (check "C2 prod axis=1 逐位 ≡ 顺序参考"
         (every #'df-bits=
                (vt-to-list (vt-prod m :axis 1))
                (mapcar #'ref-prod rows)))
  (check "C3 amax axis=1 逐位 ≡ 顺序参考"
         (every #'df-bits=
                (vt-to-list (vt-amax m :axis 1))
                (mapcar #'ref-max rows)))
  (check "C4 amin axis=1 逐位 ≡ 顺序参考"
         (every #'df-bits=
                (vt-to-list (vt-amin m :axis 1))
                (mapcar #'ref-min rows)))
  ;; 沿 axis=0（每列）
  (check "C5 sum axis=0 逐位 ≡ 顺序参考"
         (every #'df-bits=
                (vt-to-list (vt-sum m :axis 0))
                (mapcar #'ref-sum cols)))
  (check "C6 amax axis=0 逐位 ≡ 顺序参考"
         (every #'df-bits=
                (vt-to-list (vt-amax m :axis 0))
                (mapcar #'ref-max cols)))
  ;; axis=-1 与 axis=1 等价
  (check "C7 axis=-1 ≡ axis=1"
         (equal (vt-to-list (vt-sum m :axis -1))
                (vt-to-list (vt-sum m :axis 1))))
  ;; 全归约（row-major 顺序）
  (check "C8 sum 全归约 逐位 ≡ 顺序参考"
         (df-bits= (vt-item (vt-sum m)) (ref-sum flat)))
  (check "C9 amax 全归约 逐位 ≡ 顺序参考"
         (df-bits= (vt-item (vt-amax m)) (ref-max flat)))
  ;; mean = sum / n（同一累加顺序 → 逐位一致）
  (check "C10 mean axis=1 ≡ sum/n"
         (every #'df-bits=
                (vt-to-list (vt-mean m :axis 1))
                (mapcar (lambda (r) (/ (ref-sum r) 5.0d0)) rows)))
  ;; keepdims
  (check "C11 sum axis=1 :keepdims 形状"
         (equal (vt-shape (vt-sum m :axis 1 :keepdims t)) '(7 1))))

;;; 非连续输入：转置后沿 axis=0 归约 ≡ 原张量沿 axis=1 归约
(let* ((m (make-test-matrix))
       (mt (vt-transpose m)))
  (check "C12 转置视图 sum axis=0 ≡ 原张量 axis=1"
         (equal (vt-to-list (vt-sum mt :axis 0))
                (vt-to-list (vt-sum m :axis 1))))
  (check "C13 转置视图 amax axis=0 ≡ 原张量 axis=1"
         (equal (vt-to-list (vt-amax mt :axis 0))
                (vt-to-list (vt-amax m :axis 1)))))

;;; 多轴归约（3 维）
(let* ((v (vt-from-sequence
           (loop for i below 24 collect (* 1.0d0 (+ 1 i) 0.5d0))
           :dtype :float64))
       (t3 (vt-reshape v '(2 3 4))))
  (check "C14 3 维 sum axis=(1 2) ≡ 逐块顺序参考"
         (let ((got (vt-to-list (vt-sum t3 :axis '(1 2)))))
           (every #'df-bits= got
                  (list (ref-sum (loop for i below 12 collect (* 1.0d0 (+ 1 i) 0.5d0)))
                        (ref-sum (loop for i from 12 below 24
                                       collect (* 1.0d0 (+ 1 i) 0.5d0))))))))

;;; ==================================================================
;;; D. NaN 语义（nanmax 的累加器初值就是 NaN，最容易在重排时漏掉）
;;; ==================================================================

(sb-int:with-float-traps-masked (:invalid :divide-by-zero)
  (let ((nan +vt-dfloat-nan+))
    (flet ((mk (xs) (vt-from-sequence xs :dtype :float64)))
      ;; nanmax：跳过 NaN
      (check "D1 nanmax 跳过 NaN"
             (df-bits= (vt-item (vt-nanmax (mk (list 1.0d0 nan 5.0d0 3.0d0)))) 5.0d0))
      (check "D2 nanmax 全部 NaN → NaN"
             (let ((r (vt-item (vt-nanmax (mk (list nan nan)))))) (/= r r)))
      (check "D3 nanmax axis=1 逐行跳过 NaN"
             (let ((r (vt-to-list (vt-nanmax (mk (list (list 1.0d0 nan)
                                                      (list nan 2.0d0)))
                                             :axis 1))))
               (and (df-bits= (first r) 1.0d0) (df-bits= (second r) 2.0d0))))
      (check "D4 nanmin 跳过 NaN"
             (df-bits= (vt-item (vt-nanmin (mk (list 1.0d0 nan 5.0d0)))) 1.0d0))
      (check "D5 nanmin 全部 NaN → NaN"
             (let ((r (vt-item (vt-nanmin (mk (list nan nan)))))) (/= r r)))
      ;; amax：NaN 传播
      (check "D6 amax 传播 NaN（首个位置）"
             (let ((r (vt-item (vt-amax (mk (list nan 1.0d0 5.0d0)))))) (/= r r)))
      (check "D7 amax 传播 NaN（末位）"
             (let ((r (vt-item (vt-amax (mk (list 1.0d0 5.0d0 nan)))))) (/= r r)))
      (check "D8 amin 传播 NaN"
             (let ((r (vt-item (vt-amin (mk (list 1.0d0 nan)))))) (/= r r)))
      ;; nansum / nanprod
      (check "D9 nansum 跳过 NaN"
             (df-bits= (vt-item (vt-nansum (mk (list 1.0d0 nan 2.0d0)))) 3.0d0))
      (check "D10 nansum 全部 NaN → 0"
             (df-bits= (vt-item (vt-nansum (mk (list nan nan)))) 0.0d0))
      ;; 含 ±0 的 max/min（符号位定号）
      (check "D11 amax(-0.0 +0.0) 保留首个"
             (df-bits= (vt-item (vt-amax (mk (list -0.0d0 0.0d0)))) -0.0d0))
      (check "D12 amax(+0.0 -0.0) 保留首个"
             (df-bits= (vt-item (vt-amax (mk (list 0.0d0 -0.0d0)))) 0.0d0)))))

;;; ==================================================================
;;; E. 窄整型 / 布尔归约（走 %SMALL-INT-AXIS-FORM，同样受本轮改动影响）
;;; ==================================================================

(check "E1 int8 sum → int64 且值正确"
       (let ((r (vt-sum (vt-from-sequence (list 100 100 100) :dtype :int8))))
         (and (eq (vt-dtype r) :int64) (= (vt-item r) 300))))
(check "E2 int16 sum 沿 axis=1"
       (let ((r (vt-sum (vt-from-sequence (list (list 300 300) (list 1 2))
                                          :dtype :int16)
                        :axis 1)))
         (and (eq (vt-dtype r) :int64)
              (equal (vt-to-list r) '(600 3)))))
(check "E3 uint8 amax 保持 uint8"
       (let ((r (vt-amax (vt-from-sequence (list 3 250 7) :dtype :uint8))))
         (and (eq (vt-dtype r) :uint8) (= (vt-item r) 250))))
(check "E4 int8 amax 沿 axis=0"
       (let ((r (vt-amax (vt-from-sequence (list (list 1 9) (list 5 2))
                                           :dtype :int8)
                         :axis 0)))
         (equal (vt-to-list r) '(5 9))))
(check "E5 :bool sum → int64（真值化后计数）"
       (let* ((b (vt-astype (vt-from-sequence (list 1.0d0 0.0d0 2.0d0)
                                              :dtype :float64)
                            :bool))
              (r (vt-sum b)))
         (and (eq (vt-dtype r) :int64) (= (vt-item r) 2))))

;;; ==================================================================
;;; F. 倍率护栏（机器无关：只抓「快路径没接管」这类结构性回归）
;;; ==================================================================

(defun %best-of (label fn n reps)
  "跑 REPS 轮取最小耗时（毫秒），减少单次抖动误判。"
  (declare (ignore label))
  (funcall fn)                                  ; 预热
  (let ((best most-positive-fixnum))
    (dotimes (_ reps best)
      (let ((t0 (get-internal-real-time)))
        (dotimes (_ n) (funcall fn))
        (let ((ms (/ (* 1000.0 (- (get-internal-real-time) t0))
                     internal-time-units-per-second n)))
          (when (< ms best) (setf best ms)))))))

(defmacro check-ratio (label form limit what)
  `(let* ((ratio (let ((r 1d9))
                   (dotimes (_ 3 r)
                     (let ((v (coerce ,form 'double-float)))
                       (when (< v r) (setf r v))))))
          (r (coerce ratio 'double-float)))
     (if (<= r (coerce ,limit 'double-float))
         (progn (ok ,label)
                (format t "  [ OK ] ~a（~a = ~,2f× ≤ ~,1f×）~%"
                        ,label ,what r (coerce ,limit 'double-float)))
         (ng ,label "~a = ~,2f× > ~,1f×" ,what r (coerce ,limit 'double-float)))))

(let* ((n 1000000)
       (a (vt-astype (vt-linspace 1.0d0 10.0d0 n) :float64))
       (o (vt-zeros (list n) :dtype :float64))
       (m (vt-reshape (vt-astype (vt-linspace 0.0d0 1.0d0 n) :float64)
                      (list 1000 1000))))
  ;; F1 pow：专用循环 vs vt-map 闭包参考（同一表达式，只差包装层）
  (let ((fast (%best-of "pow-fast" (lambda () (vt-pow a 2.0d0 :out o)) 10 3))
        (slow (%best-of "pow-slow"
                        (lambda () (vt-map (lambda (x) (expt x 2.0d0)) a
                                           :out o :dtype :float64))
                        10 3)))
    ;; 阈值 0.8：实测 0.6~0.7（专用循环 18 ms vs 闭包 28 ms）。护栏要抓的是
    ;; 「专用循环没接管」—— 那种情形会回到 1.0× 附近，不是 20% 抖动。
    (check-ratio "F1 pow 专用循环 ≤ 0.8× 闭包参考" (/ fast slow) 0.8 "fast/slow"))
  ;; F2 sum 沿最内轴：绝对值护栏（本轮 2.0 → 1.0 ms 的改善如果回退会超过 1.4）
  (let ((ms (%best-of "sum-axis1" (lambda () (vt-sum m :axis 1)) 10 3)))
    (check-ratio "F2 sum axis=1 ≤ 1.4 ms（1M f64）" (/ ms 1.0) 1.4 "ms"))
  ;; F3 amax 全归约：NaN 判定重排后的绝对值护栏
  (let ((ms (%best-of "amax-all" (lambda () (vt-amax a)) 20 3)))
    (check-ratio "F3 amax 全归约 ≤ 1.0 ms（1M f64）" (/ ms 1.0) 1.0 "ms"))
  ;; F4 amax 沿最内轴
  (let ((ms (%best-of "amax-axis1" (lambda () (vt-amax m :axis 1)) 10 3)))
    (check-ratio "F4 amax axis=1 ≤ 1.6 ms（1M f64）" (/ ms 1.0) 1.6 "ms")))

;;; ==================================================================
;;; 汇总
;;; ==================================================================

(format t "~&~%========================================~%")
(format t "perf-reduce-pow-test: PASS=~a FAIL=~a TOTAL=~a~%"
        *pass* *fail* (+ *pass* *fail*))
(when *failures*
  (format t "~%失败项:~%")
  (dolist (f (reverse *failures*)) (format t "  - ~a~%" f)))
(format t "========================================~%")

(sb-ext:exit :code (if (zerop *fail*) 0 1))
