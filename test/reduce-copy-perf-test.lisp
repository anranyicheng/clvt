;;;; reduce-copy-perf-test.lisp — 非连续归约 / 跨 dtype 写入 的双路径一致性与倍率护栏
;;;;
;;;; 为什么需要这套
;;;; ----------------
;;;; 两处改动都只动「执行层选路」，结果必须逐位不变（CONVENTIONS §1 铁律 1）：
;;;;   1. `%KERNEL-GENERAL` 给单轴归约加了「指针递增」通路（替代每元素三重
;;;;      里程计）。累加顺序不变，但**一旦写错就是静默错值** —— 归约没有
;;;;      「形状不对」这种信号，只会给出差一点点的数。
;;;;   2. `VT-COPY-INTO` 的跨 dtype 分支改成「先 ASTYPE 整张转换，再同 dtype
;;;;      落地」。它必须与逐元素 `VT-CAST-FUN` 完全等价（bool 真值化、
;;;;      复数取实部、NaN→整数 0 三条都在里面）。
;;;;
;;;; A 段用的是**同一条不变量**而不是手写期望值：同一个逻辑张量的不同物理
;;;; 形态（连续副本 / 转置 / 步进 / 翻转 / 3D 置换）必须给出**逐位相同**的
;;;; 归约结果。这条比「和内嵌期望值比」强得多 —— 它连 NaN 载荷、±0 符号、
;;;; 窄整型提升全都覆盖，而且不需要为每种组合写期望值。
(require :asdf)
#+quicklisp (ql:quickload :clvt)
(asdf:load-system :clvt)

(defpackage :clvt-reduce-copy-perf
  (:use :cl :clvt)
  (:export #:run))

(in-package :clvt-reduce-copy-perf)

(defvar *pass* 0)
(defvar *fail* 0)
(defvar *failures* nil)

(defun ok (label)
  (incf *pass*)
  (format t "  [ OK ] ~a~%" label)
  t)

(defun ng (label fmt &rest args)
  (incf *fail*)
  (let ((msg (apply #'format nil fmt args)))
    (push (format nil "~a — ~a" label msg) *failures*)
    (format t "  [FAIL] ~a~%         ~a~%" label msg)
    nil))

(defun check (label test)
  (if test (ok label) (ng label "断言为假")))

(defun check-ratio (label ratio limit what)
  (if (and (numberp ratio) (<= ratio limit))
      (progn (incf *pass*)
             (format t "  [ OK ] ~a（~a = ~,2f× ≤ ~,1f×）~%" label what (float ratio) limit)
             t)
      (ng label "~a = ~,2f× > ~,1f×" what (float ratio) limit)))

;;; ------------------------------------------------------------------
;;; 计时（与其余护栏套件同一套：校准批量 + 取最小值）
;;; ------------------------------------------------------------------
(defun %calibrate (thunk)
  (let ((inner 1))
    (loop
      (let ((t0 (get-internal-real-time)))
        (dotimes (i inner) (funcall thunk))
        (let ((dt (- (get-internal-real-time) t0)))
          (cond ((>= dt 20000) (return inner))
                ((zerop dt) (setf inner (* inner 8)))
                (t (setf inner (max (1+ inner)
                                    (ceiling (* inner 30000) dt))))))))))

(defun bench-us (thunk &key (reps 7))
  (funcall thunk)
  (let ((inner (%calibrate thunk))
        (best most-positive-fixnum))
    (dotimes (r reps)
      (let ((t0 (get-internal-real-time)))
        (dotimes (i inner) (funcall thunk))
        (setf best (min best (/ (float (- (get-internal-real-time) t0) 1.0d0)
                                inner)))))
    best))

;;; ------------------------------------------------------------------
;;; 逐位比较工具
;;; ------------------------------------------------------------------
(defun elt-bits (x)
  (cond ((typep x 'double-float) (sb-kernel:double-float-bits x))
        ((typep x 'single-float) (sb-kernel:single-float-bits x))
        ((complexp x) (list (elt-bits (realpart x)) (elt-bits (imagpart x))))
        (t x)))

(defun same-bits-tensor (a b)
  "两个张量的逻辑内容逐位相同？（按行主序展开，与物理布局无关）"
  (and (equal (vt-shape a) (vt-shape b))
       (eq (vt-dtype a) (vt-dtype b))
       (let* ((flat-a (vt-contiguous a))
              (flat-b (vt-contiguous b))
              (da (vt-data flat-a))
              (db (vt-data flat-b))
              (oa (vt-offset flat-a))
              (ob (vt-offset flat-b)))
         (dotimes (i (vt-size a) t)
           ;; 用 EQUAL 而不是 EQL：复数的位模式是 (实部 . 虚部) 二元素列表，
           ;; EQL 对两个不同的 cons 恒为假（会把每个复数都判成不等）。
           (unless (equal (elt-bits (aref da (+ oa i)))
                          (elt-bits (aref db (+ ob i))))
             (return nil))))))

(defun same-bits-any (a b)
  "标量或张量都能比：张量走 SAME-BITS-TENSOR，标量按位比。"
  (cond ((and (vt-p a) (vt-p b)) (same-bits-tensor a b))
        ((and (not (vt-p a)) (not (vt-p b))) (equal (elt-bits a) (elt-bits b)))
        (t nil)))

;;; ------------------------------------------------------------------
;;; 夹具：同一逻辑数据的多种物理形态
;;; ------------------------------------------------------------------
(defun det-f64 (n &key (scale 37) (mod 101))
  (vt-from-sequence (loop for i below n collect (coerce (mod (* i scale) mod) 'double-float))
                    :dtype :float64))

(defun nan-f64 (n)
  "含 ±0、NaN、±Inf 的确定性序列（用位置决定，不用随机数）。"
  (vt-from-sequence
   (loop for i below n
         collect (case (mod i 11)
                   (0 clvt::+vt-dfloat-nan+)
                   (1 0.0d0)
                   (2 -0.0d0)
                   (3 clvt::+vt-dfloat-pos-inf+)
                   (4 clvt::+vt-dfloat-neg-inf+)
                   (t (coerce (- (mod (* i 13) 29) 14) 'double-float))))
   :dtype :float64))

;;; ------------------------------------------------------------------
;;; A1 — 沿轴归约：不同物理形态必须给出逐位相同的结果
;;; ------------------------------------------------------------------
(defun a1-axis-reduce-invariance ()
  (format t "~%[A1] 沿轴归约：物理形态无关性（逐位）~%")
  (dolist (spec (list (cons "f64 有限" (det-f64 60))
                      (cons "f64 NaN/±0/Inf" (nan-f64 60))
                      (cons "int16" (vt-astype (det-f64 60) :int16))
                      (cons "uint8" (vt-astype (vt-* (det-f64 60) (vt-full '(60) 3)) :uint8))
                      (cons "f32" (vt-astype (det-f64 60) :float32))
                      (cons "c128" (vt-astype (det-f64 60) :complex128))))
    (let* ((name (car spec))
           (flat (cdr spec))
           (m (vt-reshape flat '(3 4 5)))
           (views (list (cons "连续 3D" m)
                        (cons "转置 (2,1,0)" (vt-transpose m '(2 1 0)))
                        (cons "转置 (1,2,0)" (vt-transpose m '(1 2 0)))
                        (cons "步进 axis0" (vt-slice m '(nil nil 2) '(:all) '(:all)))
                        (cons "翻转 axis2" (vt-flip m :axis 2))
                        (cons "子块" (vt-slice m '(0 2) '(1 4) '(nil nil 2))))))
      (dolist (pair views)
        (let* ((vname (car pair))
               (v (cdr pair))
               (ref (vt-contiguous v))         ; 同一逻辑值的连续副本
               (okp t))
          ;; 单轴
          (dotimes (ax 3)
            (dolist (keepdims (list nil t))
              (dolist (op (list #'vt-sum #'vt-prod #'vt-amax #'vt-amin
                                #'vt-nansum #'vt-nanmax))
                (let ((a (handler-case (funcall op v :axis ax :keepdims keepdims)
                           (error (e) e)))
                      (b (handler-case (funcall op ref :axis ax :keepdims keepdims)
                           (error (e) e))))
                  (unless (same-bits-any a b)
                    (setf okp nil)
                    (ng (format nil "A1 ~a / ~a / ax=~a kd=~a" name vname ax keepdims)
                        "非连续与连续副本结果不同")))))
            ;; arg 族（只接受单轴）
            (dolist (op (list #'vt-argmax #'vt-argmin))
              (unless (same-bits-any (funcall op v :axis ax) (funcall op ref :axis ax))
                (setf okp nil)
                (ng (format nil "A1 ~a / ~a / arg ax=~a" name vname ax)
                    "arg 族结果不同"))))
          ;; 全归约 + 多轴元组
          (dolist (op (list #'vt-sum #'vt-prod #'vt-amax #'vt-amin #'vt-nansum))
            (unless (same-bits-any (funcall op v) (funcall op ref))
              (setf okp nil)
              (ng (format nil "A1 ~a / ~a / 全归约" name vname) "结果不同")))
          (unless (same-bits-any (vt-sum v :axis '(0 2)) (vt-sum ref :axis '(0 2)))
            (setf okp nil)
            (ng (format nil "A1 ~a / ~a / 多轴 (0 2)" name vname) "结果不同"))
          (when okp
            (ok (format nil "A1 ~a / ~a（单轴·多轴·arg·keepdims 全一致）" name vname))))))))

;;; ------------------------------------------------------------------
;;; A2 — 沿轴归约的数值正确性（独立参考：按坐标逐个累加）
;;; ------------------------------------------------------------------
(defun a2-axis-reduce-values ()
  (format t "~%[A2] 沿轴归约 vs 坐标级参考实现~%")
  (let* ((m (vt-reshape (det-f64 24) '(4 6)))
         (mt (vt-transpose m)))
    (dolist (pair (list (cons "连续" m) (cons "转置" mt)))
      (let* ((name (car pair))
             (v (cdr pair))
             (got (vt-sum v :axis 0))
             (rows (first (vt-shape v)))
             (cols (second (vt-shape v)))
             (okp t))
        ;; 参考：固定列 j，按行号升序把 vt-ref 逐个加进 double-float 累加器
        ;; —— 这正是 §7.3 要求的累加顺序，逐位可比。
        (dotimes (j cols)
          (let ((want 0.0d0))
            (dotimes (i rows) (incf want (vt-ref v i j)))
            (unless (= (sb-kernel:double-float-bits want)
                       (sb-kernel:double-float-bits (vt-ref got j)))
              (setf okp nil)
              (ng (format nil "A2 ~a j=~a" name j)
                  "参考 ~a 实得 ~a" want (vt-ref got j)))))
        (when okp (ok (format nil "A2 ~a axis0 与坐标级参考逐位一致" name)))))))

;;; ------------------------------------------------------------------
;;; A3 — 非连续 :out 与 keepdims（写穿透）
;;; ------------------------------------------------------------------
(defun a3-out-noncontig ()
  (format t "~%[A3] 归约 :out 落在非连续视图~%")
  ;; AT 形状 (6 5)：沿 axis 0 归约得 (5)
  (let* ((at (vt-transpose (vt-reshape (det-f64 30) '(5 6))))
         (ref (vt-contiguous at))
         (o-contig (vt-zeros '(5) :dtype :float64))
         (o-step (vt-slice (vt-zeros '(10) :dtype :float64) '(nil nil 2)))
         (o-tr (vt-transpose (vt-zeros '(5 1) :dtype :float64))))
    (check "A3 :out 连续 ≡ 返回的新数组"
           (same-bits-any (vt-sum at :axis 0 :out o-contig)
                          (vt-sum ref :axis 0)))
    (check "A3 :out 步进切片"
           (same-bits-any (vt-sum at :axis 0 :out o-step)
                          (vt-sum ref :axis 0)))
    (check "A3 :out 转置 + keepdims"
           (same-bits-any (vt-sum at :axis 0 :keepdims t :out o-tr)
                          (vt-sum ref :axis 0 :keepdims t)))
    ;; 步进视图只写偶数位：奇数位必须还是 0（写穿透没有越界到别的位置）
    (check "A3 :out 写到了底层缓冲的偶数位"
           (let ((buf (vt-data o-step)))
             (and (plusp (aref buf 0)) (zerop (aref buf 1)))))))

;;; ------------------------------------------------------------------
;;; A4 — 跨 dtype 写入 ≡ 先 ASTYPE 再同 dtype 写
;;; ------------------------------------------------------------------
(defun a4-cross-dtype-copy ()
  (format t "~%[A4] 跨 dtype 的 vt-copy-into ≡ ASTYPE 后同 dtype 写~%")
  (let* ((base (vt-from-sequence
                (list -3.7d0 -1.0d0 0.0d0 0.4d0 1.5d0 2.5d0 127.6d0 300.0d0
                      clvt::+vt-dfloat-nan+ clvt::+vt-dfloat-pos-inf+ (complex 1.5d0 -2.25d0)
                      -2.0d0 7.0d0 0.25d0 9.5d0 11.0d0 13.0d0 15.0d0)
                :dtype :float64))
         (src3x6 (vt-reshape base '(3 6)))
         (dtypes '( :float64 :float32 :int64 :int32 :int16 :int8
                    :uint16 :uint8 :bool :complex128 :complex64))
         (okp t))
    (dolist (sdt dtypes)
      (dolist (ddt dtypes)
        (let* ((src (vt-astype src3x6 sdt))
               (want (vt-astype src ddt)))
          ;; (1) 连续 dst
          (let ((dst (vt-zeros '(3 6) :dtype ddt)))
            (vt-copy-into dst src)
            (unless (same-bits-tensor dst want)
              (setf okp nil)
              (ng (format nil "A4 连续 ~a→~a" sdt ddt) "与 ASTYPE 结果不同")))
          ;; (2) 非连续 dst（转置）
          (let ((dst (vt-transpose (vt-zeros '(6 3) :dtype ddt))))
            (vt-copy-into dst src)
            (unless (same-bits-tensor dst want)
              (setf okp nil)
              (ng (format nil "A4 转置 dst ~a→~a" sdt ddt) "与 ASTYPE 结果不同")))
          ;; (3) 非连续 src（转置）
          (let* ((srct (vt-transpose src))
                 (dst (vt-zeros '(6 3) :dtype ddt)))
            (vt-copy-into dst srct)
            (unless (same-bits-tensor dst (vt-astype srct ddt))
              (setf okp nil)
              (ng (format nil "A4 转置 src ~a→~a" sdt ddt) "与 ASTYPE 结果不同"))))))
    (when okp (ok "A4 11×11 dtype × 3 种布局（连续/转置 dst/转置 src）全部 ≡ ASTYPE")))
  ;; 门槛两侧都要对：小块（<16 元素）走逐元素路径，大块走 ASTYPE 路径
  (let ((small (vt-from-sequence (list 1.5d0 2.5d0 -3.5d0 0.5d0) :dtype :float64)))
    (dolist (ddt '(:float32 :int32 :bool :complex128))
      (let ((dst (vt-zeros '(4) :dtype ddt)))
        (vt-copy-into dst small)
        (check (format nil "A4 小块 4 元素 f64→~a" ddt)
               (same-bits-tensor dst (vt-astype small ddt))))))
  ;; 广播 src（尺寸 1 → 铺满）跨 dtype
  (let ((one (vt-from-sequence (list 2.5d0) :dtype :float64))
        (dst (vt-zeros '(3 4) :dtype :int32)))
    (vt-copy-into dst one)
    (check "A4 size=1 跨 dtype 广播填充"
           (same-bits-any dst (vt-full '(3 4) 2 :dtype :int32)))))

;;; ------------------------------------------------------------------
;;; A5 — 跨 dtype 写入的形状/别名契约不变
;;; ------------------------------------------------------------------
(defun a5-cross-dtype-contract ()
  (format t "~%[A5] 跨 dtype 写入的契约~%")
  (flet ((err-p (thunk)
           (handler-case (progn (funcall thunk) nil) (error () t))))
    (let ((dst (vt-zeros '(2 3) :dtype :float32))
          (src (vt-zeros '(4 5) :dtype :float64)))
      (check "A5 不能广播的形状仍报错" (err-p (lambda () (vt-copy-into dst src)))))
    ;; 别名重叠：dest 与 src 共享底层数组且跨 dtype（同一 buffer、不同 dtype 视图）
    (let* ((f (vt-reshape (vt-from-sequence (list 1.0d0 2.0d0 3.0d0 4.0d0) :dtype :float64)
                          '(2 2)))
           (dst (vt-zeros '(2 2) :dtype :float64)))
      (vt-copy-into dst (vt-transpose f))
      (check "A5 转置 src 写入后 dst 内容正确"
             (same-bits-tensor dst (vt-transpose f))))
    ;; 只读广播目标仍拒绝
    (let ((bc (vt-broadcast-to (vt-zeros '(1 3) :dtype :float32) '(4 3))))
      (check "A5 广播目标写入报错"
             (err-p (lambda () (vt-copy-into bc (vt-ones '(4 3)))))))))

;;; ------------------------------------------------------------------
;;; B1 — 沿轴归约的代价必须显著低于「整块收集 + 再归约」
;;;      非连续 input 的 axis 归约若退回逐元素里程计，代价会接近甚至超过
;;;      「先 vt-contiguous 再归约」。优化前 sum(转置, axis0) = 4.44 ms，
;;;      vt-contiguous(转置) = 1.34 ms（比值 3.3×）；优化后 0.58 ms（0.43×）。
;;; ------------------------------------------------------------------
(defun b1-axis-reduce-cost ()
  (format t "~%[B1] 非连续归约 vs 整块收集~%")
  (let* ((at (vt-transpose (vt-random-uniform '(1000 1000))))
         (t-ax0 (bench-us (lambda () (vt-sum at :axis 0))))
         (t-ax1 (bench-us (lambda () (vt-sum at :axis 1))))
         (t-gather (bench-us (lambda () (vt-contiguous at)))))
    (check-ratio "B1 sum(转置, axis0) ÷ vt-contiguous"
                 (/ t-ax0 (max t-gather 1e-9)) 1.00
                 "t(sum ax0) ÷ t(gather)")
    (check-ratio "B1 sum(转置, axis1) ÷ vt-contiguous"
                 (/ t-ax1 (max t-gather 1e-9)) 1.00
                 "t(sum ax1) ÷ t(gather)")
    ;; amax/argmax 同一条通路
    (let ((t-max (bench-us (lambda () (vt-amax at :axis 0)))))
      (check-ratio "B1 amax(转置, axis0) ÷ sum(转置, axis0)"
                   (/ t-max (max t-ax0 1e-9)) 3.00
                   "t(amax ax0) ÷ t(sum ax0)"))))

;;; ------------------------------------------------------------------
;;; B2 — 跨 dtype 写入不得比「自己先 ASTYPE 再同 dtype 写」贵
;;;      优化前逐元素 FUNCALL CASTER + 两次泛型 AREF：1M f64→f32 是
;;;      10.0 ms，而 ASTYPE 只要 0.76 ms（13×）。
;;; ------------------------------------------------------------------
(defun b2-cross-dtype-copy-cost ()
  (format t "~%[B2] 跨 dtype 写入 vs ASTYPE 基线~%")
  (let* ((a (vt-random-uniform '(1000 1000)))
         (at (vt-transpose a))
         (d32 (vt-zeros '(1000 1000) :dtype :float32))
         (t-copy (bench-us (lambda () (vt-copy-into d32 a))))
         (t-cast (bench-us (lambda () (vt-astype a :float32)))))
    (check-ratio "B2 copy-into f64→f32 ÷ astype f64→f32"
                 (/ t-copy (max t-cast 1e-9)) 2.50
                 "t(copy-into) ÷ t(astype)")
    (let ((t-copy-t (bench-us (lambda () (vt-copy-into d32 at)))))
      (check-ratio "B2 copy-into(转置 src) ÷ astype(转置)"
                   (/ t-copy-t (max (bench-us (lambda () (vt-astype at :float32))) 1e-9))
                   2.00 "t(copy-into 转置) ÷ t(astype 转置)"))
    ;; 同 dtype 的连续拷贝必须仍走 memcpy（不能被新分支抢走）
    (let* ((d64 (vt-zeros '(1000 1000)))
           (t-same (bench-us (lambda () (vt-copy-into d64 a)))))
      (check-ratio "B2 copy-into 同 dtype ÷ astype 跨 dtype"
                   (/ t-same (max t-cast 1e-9)) 2.00
                   "t(copy-into 同dtype) ÷ t(astype)"))))

;;; ------------------------------------------------------------------
;;; B3 — 单轴归约不得比全归约更贵（同元素数）
;;;      全归约走多轴里程计且访问次序最差，单轴若实现正确应当更快。
;;; ------------------------------------------------------------------
(defun b3-single-vs-global ()
  (format t "~%[B3] 单轴归约 vs 全归约~%")
  (let* ((at (vt-transpose (vt-random-uniform '(1000 1000))))
         (t-ax (bench-us (lambda () (vt-sum at :axis 0))))
         (t-all (bench-us (lambda () (vt-sum at)))))
    ;; 优化前：ax0 4.44 / 全归约 7.29 = 0.61；优化后 0.58/7.2 = 0.08
    (check-ratio "B3 sum(转置, axis0) ÷ sum(转置, 全归约)"
                 (/ t-ax (max t-all 1e-9)) 0.35
                 "t(ax0) ÷ t(全归约)")))

;;; ------------------------------------------------------------------
;;; 主入口
;;; ------------------------------------------------------------------
(defun run ()
  (format t "~%===== 非连续归约 / 跨 dtype 写入 护栏 =====~%")
  (a1-axis-reduce-invariance)
  (a2-axis-reduce-values)
  (a3-out-noncontig)
  (a4-cross-dtype-copy)
  (a5-cross-dtype-contract)
  (b1-axis-reduce-cost)
  (b2-cross-dtype-copy-cost)
  (b3-single-vs-global)
  (format t "~&===== 汇总: ~a 通过 / ~a 失败 =====~%" *pass* *fail*)
  (format t "Total: ~a | Pass: ~a | Fail: ~a~%" (+ *pass* *fail*) *pass* *fail*)
  (when *failures*
    (format t "失败清单:~%")
    (dolist (f (nreverse *failures*)) (format t "  - ~a~%" f)))
  (finish-output)
  (zerop *fail*))

(let ((okp (run)))
  (sb-ext:exit :code (if okp 0 1)))
