;;;; perf-fastpath-test.lisp — 非复数快路径护栏（选路结构 + 机器无关倍率）
;;;;
;;;; 为什么需要这套：matmul / 归约 / astype / 连续化 的快路径都是「选路」
;;;; 性质的 —— 一旦选路条件写错（比如内层维门槛失效、线程数改回核数-1、
;;;; 归约退回 odometer），结果**仍然正确**，只是慢几倍到几十倍。
;;;; 纯正确性测试全部绿，性能悄悄退化。
;;;;
;;;; 护栏分两类，都不写绝对毫秒（那不可移植，见 CONVENTIONS §16.16 第 7 条）：
;;;;   · 结构类：直接断言选路条件（快路径必须被选中）
;;;;   · 倍率类：同规模下两个操作的**耗时比**，两边同机同进程，
;;;;     机器快慢在比值里基本抵消
(require :asdf)
#+quicklisp (ql:quickload :clvt)
(asdf:load-system :clvt)
(defpackage :clvt-perfguard
  (:use :cl :clvt)
  (:export #:run))

(in-package :clvt-perfguard)

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
  "断言 RATIO ≤ LIMIT，打印实测值（失败时可定位是哪一侧变了）。"
  (if (and (numberp ratio) (<= ratio limit))
      (progn (incf *pass*)
             (format t "  [ OK ] ~a（~a = ~,2f× ≤ ~,1f×）~%" label what ratio limit)
             t)
      (ng label "~a = ~,2f× > ~,1f×" what (float ratio) limit)))

;;; ------------------------------------------------------------------
;;; 计时：系统时钟只有 1ms 刻度，单次调用测不到 —— 必须先校准批量次数，
;;; 让每次测量跨 ≥ 20 个刻度，再取中位数。
;;; ------------------------------------------------------------------
(defun %calibrate (thunk)
  "返回 INNER，使一次测量跨 ≥ 20 ms。

   单位陷阱：`get-internal-real-time` 的刻度是 1 µs（units/s = 1e6），
   但本机实际分辨率是 1 ms —— 短于 1 ms 的操作只会返回 0 或 1000。
   所以「撑够」的目标必须写成 20000（= 20 ms），不能写成 20。"
  (let ((inner 1))
    (loop
      (let* ((t0 (get-internal-real-time)))
        (dotimes (i inner) (declare (ignore i)) (funcall thunk))
        (let ((dt (- (get-internal-real-time) t0)))
          (cond ((>= dt 20000) (return inner))
                ((zerop dt) (setf inner (* inner 8)))
                (t (setf inner (max (1+ inner)
                                    (ceiling (* inner 30000) dt))))))))))

(defun median-us (thunk &key (reps 7))
  "取**最小值**（微秒/次），不是中位数。

   为什么用最小值：这台机器（共享 VM）的耗时分布是双峰的 —— 同样的调用
   时而 12ms、时而 60ms，差 5 倍，是外部抢占而不是被测代码。中位数会被
   这些干扰拉走，最小值则是「最少干扰的那一次」，比值护栏因此稳定得多。
   THUNK 单次可能远小于 1ms 时钟刻度，故先校准批量次数。"
  (funcall thunk)
  (let* ((inner (%calibrate thunk))
         (ts '()))
    (dotimes (r reps)
      (let ((t0 (get-internal-real-time)))
        (dotimes (i inner) (declare (ignore i)) (funcall thunk))
        (push (/ (float (- (get-internal-real-time) t0)) inner) ts)))
    (reduce #'min ts)))

;;; ------------------------------------------------------------------
;;; G1 — GEMM 分块：大尺寸的「每 flop 时间」不得比小尺寸差太多
;;;      分块失效时，1000³ 会退化成内存带宽受限（每 flop 时间暴涨），
;;;      而 200³ 仍在缓存里，比值立刻拉开。
;;; ------------------------------------------------------------------
(defun g1-gemm-blocking ()
  (let* ((a200 (vt-random-uniform '(200 200)))
         (b200 (vt-random-uniform '(200 200)))
         (a1k (vt-random-uniform '(1000 1000)))
         (b1k (vt-random-uniform '(1000 1000)))
         (t200 (median-us (lambda () (vt-matmul a200 b200)) :reps 5))
         (t1k  (median-us (lambda () (vt-matmul a1k b1k)) :reps 3))
         ;; 每 flop 时间比：体积比 125×，理想值 1；分块失效时 1000³ 会慢 4 倍以上
         (ratio (/ (/ t1k 1e9) (/ t200 8e6))))
    (check-ratio "G1 GEMM 1000³ 每 flop 时间 / 200³" ratio 4.0
                 "t(1000³)/1e9 ÷ t(200³)/8e6")))

;;; ------------------------------------------------------------------
;;; G2 — 归约行累加：axis=0 求和不得慢于「整数组拷贝」
;;;      两者读的字节数相同，求和只写 1/n 的输出。
;;;      走列扫（旧路径）时 sum 会比一次拷贝慢 2.5 倍以上。
;;; ------------------------------------------------------------------
(defun g2-reduce-row-acc ()
  (let* ((a (vt-random-uniform '(1000 1000)))
         (t-sum (median-us (lambda () (vt-sum a :axis 0)) :reps 5))
         (t-copy (median-us (lambda () (vt-copy a)) :reps 5)))
    (check-ratio "G2 sum:axis0 / vt-copy（同数组）" (/ t-sum t-copy) 2.0
                 "t(sum axis0) ÷ t(copy)")
    ;; 结构断言：内层维 ≥ 门槛时必须走行累加（门槛常量本身也要在合理区间）
    (check "G2b 行累加门槛 +row-acc-min-inner+ 合理（8~64）"
           (<= 8 clvt::+row-acc-min-inner+ 64))))

;;; ------------------------------------------------------------------
;;; G3 — 归约：窄整型不得退回 odometer
;;;      同一形态下 int16 求和与 int64 求和的耗时比应接近 1
;;;      （都是「读窄数组 + 累加到 int64」）；退回路径 3 时差 6 倍以上。
;;; ------------------------------------------------------------------
(defun g3-small-int-reduce ()
  (let* ((base (vt-random-uniform '(1000 1000)))
         (i16 (vt-astype base :int16))
         (i64 (vt-astype base :int64))
         (t16 (median-us (lambda () (vt-sum i16 :axis 0)) :reps 5))
         (t64 (median-us (lambda () (vt-sum i64 :axis 0)) :reps 5)))
    ;; 阈值放到 10：这条只拦「退回 odometer」这种量级（旧路径是 14 倍），
    ;; 不拦 2~3 倍的小波动（本机耗时分布受外部抢占影响是双峰的）。
    (check-ratio "G3 sum:axis0 int16 / int64" (/ t16 (max t64 1e-9)) 10.0
                 "t(int16) ÷ t(int64)")))

;;; ------------------------------------------------------------------
;;; G4 — astype：窄整型源不得退回 vt-cast-fun 逐元素
;;;      int8→f64 与 f32→f64 同为「读窄写宽」，比值应接近 1；
;;;      退回通用 fallback 时差 5 倍以上。
;;; ------------------------------------------------------------------
(defun g4-astype-narrow ()
  (let* ((base (vt-random-uniform '(1000 1000)))
         (i8 (vt-astype base :int8))
         (f32 (vt-random-uniform '(1000 1000) :dtype :float32))
         (t-i8 (median-us (lambda () (vt-astype i8 :float64)) :reps 5))
         (t-f32 (median-us (lambda () (vt-astype f32 :float64)) :reps 5)))
    (check-ratio "G4 astype int8→f64 / f32→f64" (/ t-i8 (max t-f32 1e-9)) 3.0
                 "t(int8→f64) ÷ t(f32→f64)")))

;;; ------------------------------------------------------------------
;;; G5 — 连续化：转置视图的 collect 不得比连续拷贝慢太多
;;;      两者搬的字节数相同（都是 8MB），转置多的是寻址。
;;;      走通用 walker 时比值 2.5× 以上。
;;; ------------------------------------------------------------------
(defun g5-contiguous-gather ()
  (let* ((a (vt-random-uniform '(1000 1000)))
         (at (vt-transpose a))
         (t-t (median-us (lambda () (vt-contiguous at)) :reps 5))
         (t-c (median-us (lambda () (vt-copy a)) :reps 5)))
    ;; 理想 2.4×（转置多的是寻址，不是字节）；退回通用 walker 是 6.8×。
    ;; 阈值 5.0 卡在中间，且取最小值以避开 GC/抢占干扰。
    (check-ratio "G5 vt-contiguous(转置) / vt-copy(连续)" (/ t-t (max t-c 1e-9)) 5.0
                 "t(collect 转置) ÷ t(copy 连续)")
    ;; 零拷贝语义：连续输入必须原样返回（同一对象）
    (check "G5b vt-contiguous 对连续输入零拷贝（eq）" (eq (vt-contiguous a) a))
    (check "G5c vt-contiguous 对转置输入返回新对象"
           (not (eq (vt-contiguous at) at)))))

;;; ------------------------------------------------------------------
;;; G6 — 线程数契约：GEMM 不得把线程数开到核数-1（L2 会被打穿）
;;; ------------------------------------------------------------------
(defun g6-thread-count ()
  (check "G6 GEMM 线程数上限 ≤ 16"
         (<= clvt::*simd-matmul-thread-count* 16))
  (check "G6b GEMM 线程数 ≥ 1"
         (>= clvt::*simd-matmul-thread-count* 1)))

;;; ------------------------------------------------------------------
;;; G7 — 结果不变性：快路径与回退路径必须逐位相同
;;;      （铁律 1 的可执行版本；关闭 SIMD 后 matmul 应与开启时逐位相等）
;;; ------------------------------------------------------------------
(defun g7-same-result ()
  "vt-matmul 与 vt-einsum 必须**逐位相同**：两者走同一个 GEMM 内核
   （`*simd-gemm-raw-fn*`），若哪天有人给其中一条另写一套内核，
   这条会立刻变红。注意不能拿「关掉 SIMD」当对照 —— 回退路径是
   mul+add（两次舍入），与 FMA（一次舍入）本来就不逐位相同。"
  (let* ((a (vt-random-uniform '(120 80)))
         (b (vt-random-uniform '(80 90)))
         (m1 (vt-data (vt-matmul a b)))
         (m2 (vt-data (vt-einsum "ij,jk->ik" a b))))
    (check "G7 vt-matmul ≡ vt-einsum（同内核 ⇒ 逐位相同）"
           (dotimes (i (length m1) t)
             (unless (eql (aref m1 i) (aref m2 i)) (return nil))))))

;;; ------------------------------------------------------------------
;;; G8 — astype 的切片/跨步视图：必须走「逐 run 单趟」
;;;      切片视图 a[::2,:] 的内层 run = 1000（≥ 门槛），与「同规模的连续
;;;      半张表」相比读的字节数相同，只多一次跨步寻址。
;;;      退回「先收集成连续副本 → 再整数组转换」时是 4×（两趟全量流量
;;;      + 一块全尺寸临时分配）。
;;; ------------------------------------------------------------------
(defun g8-astype-strided-fused ()
  (let* ((a (vt-random-uniform '(1000 1000)))
         (rows (vt-slice a '(nil nil 2) '(:all)))   ; 500×1000，内层 run=1000
         (half (vt-slice a '(0 500) '(:all)))       ; 同规模、连续
         (t-rows (median-us (lambda () (vt-astype rows :float32)) :reps 5))
         (t-half (median-us (lambda () (vt-astype half :float32)) :reps 5)))
    (check-ratio "G8 astype(切片) / astype(同规模连续)"
                 (/ t-rows (max t-half 1e-9)) 1.8
                 "t(astype a[::2,:]→f32) ÷ t(astype 连续半张→f32)")
    ;; 结构断言：融合门槛本身要在合理区间（太大 → 切片视图退回双趟；
    ;; 太小 → 转置视图也逐 run，调用开销吞掉收益）
    (check "G8b 融合门槛 +astype-fuse-min-run+ 合理（8~64）"
           (<= 8 clvt::+astype-fuse-min-run+ 64))))

;;; ------------------------------------------------------------------
;;; 主入口
;;; ------------------------------------------------------------------
(defun run ()
  (format t "~%===== 非复数快路径护栏 =====~%")
  (g1-gemm-blocking)
  (g2-reduce-row-acc)
  (g3-small-int-reduce)
  (g4-astype-narrow)
  (g5-contiguous-gather)
  (g6-thread-count)
  (g7-same-result)
  (g8-astype-strided-fused)
  (format t "~&===== 汇总: ~a 通过 / ~a 失败 =====~%" *pass* *fail*)
  (format t "Total: ~a | Pass: ~a | Fail: ~a~%" (+ *pass* *fail*) *pass* *fail*)
  (when *failures*
    (format t "失败清单:~%")
    (dolist (f (nreverse *failures*)) (format t "  - ~a~%" f)))
  (finish-output)
  (zerop *fail*))

(let ((ok (run)))
  (sb-ext:exit :code (if ok 0 1)))
