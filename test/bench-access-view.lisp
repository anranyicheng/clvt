;;;; bench-access-view.lisp — 访问 / 切片 / 视图 / 连续化 / 拼接 的独立基准
;;;;
;;;; 运行方式（与 test/run-tests.sh --all 的基准套件同一入口）：
;;;;   bash test/run-tests.sh --suite bench-access-view
;;;; 或单独：
;;;;   sbcl --dynamic-space-size 4096 --non-interactive \
;;;;        --eval '(asdf:load-system :clvt)' --eval '(load "test/bench-access-view.lisp")'
;;;;
;;;; 计时口径（本机是共享 KVM，时钟刻度 1ms、耗时分布双峰，必须按下面做）：
;;;;   1. 先校准「一次测量撑到 ≥ 200ms」的重复次数（噪声只会让测量变慢）；
;;;;   2. 重复 5 批，取**最小值**；
;;;;   3. 每个用例单次调用远小于 1ms 很正常（视图构造只有几十纳秒），
;;;;      所以输出里同时给 ms / 单次调用 与 ns/元素 两种口径。
;;;;
;;;; 用法建议：改性能前跑一遍存盘，改完再跑一遍，比同一用例的两行数字。
;;;; 绝对毫秒不可跨机器比较；跨机器请看 CONVENTIONS §7.4.5 的倍率护栏
;;;; （test/access-view-perf-test.lisp 的 B 段）。
(require :asdf)
#+quicklisp (ql:quickload :clvt)
(asdf:load-system :clvt)

(defpackage :clvt-bench-access-view
  (:use :cl :clvt)
  (:export #:run))

(in-package :clvt-bench-access-view)

(defvar *sink* nil)

(defparameter *target-ms* 200)          ; 单批目标时长
(defparameter *batches* 5)              ; 批次数（取最小值）

(defun %ms (ticks)
  (* 1000.0d0 (/ ticks (float internal-time-units-per-second 1.0d0))))

(defun %measure (thunk reps)
  (let ((t0 (get-internal-real-time)))
    (dotimes (i reps) (setf *sink* (funcall thunk)))
    (%ms (- (get-internal-real-time) t0))))

(defun %calibrate (thunk)
  "返回使单批耗时 ≥ *TARGET-MS* 的重复次数（最多试探 8 轮）。"
  (let ((reps 1))
    (dotimes (try 8 reps)
      (declare (ignore try))
      (let* ((dt (%measure thunk reps))
             (per (max dt 0.5d0)))
        (when (>= dt *target-ms*) (return reps))
        (setf reps (max (1+ reps)
                        (min (* reps 8)
                             (ceiling (* reps (/ *target-ms* per))))))))))

(defun bench (name thunk &key (elems nil))
  (setf *sink* (funcall thunk))         ; 预热（含首次分配 / 页错误）
  (let ((reps (%calibrate thunk))
        (best most-positive-double-float))
    (dotimes (b *batches*)
      (declare (ignore b))
      (let ((dt (/ (%measure thunk reps) reps)))
        (when (< dt best) (setf best dt))))
    (if elems
        (format t "  ~44a ~11,3f ms  (~7,3f ns/elt)~%"
                name best (/ (* best 1000000.0d0) elems))
        (format t "  ~44a ~11,3f ms  (~7,1f ns/call)~%" name best (/ (* best 1000000.0d0) 1)))
    (finish-output)
    best))

(defun header (s) (format t "~%~a~%" s) (finish-output))

(defun run ()
  (let* ((a (vt-random-uniform '(1000 1000)))
         (b (vt-random-uniform '(1000 1000)))
         (at (vt-transpose a))
         (av (vt-slice a '(:all) '(nil nil 2)))
         (a1 (vt-reshape a '(1000000)))
         (c8t (vt-transpose (vt-astype a :complex128)))
         (rows (loop repeat 64 collect (vt-slice a '(0) '(:all))))
         (rowsv (loop repeat 64 collect (vt-slice a '(0) '(nil nil 2))))
         (big (loop repeat 8 collect (vt-random-uniform '(1000 1000))))
         (small (loop repeat 200 collect (vt-random-uniform '(10 10))))
         (nc (loop repeat 8 collect (vt-transpose b))))

    (header "[1] 标量访问（1M 次调用）")
    (bench "vt-ref 1D 连续" (lambda () (dotimes (i 1000000) (setf *sink* (vt-ref a1 i)))) :elems 1000000)
    (bench "vt-ref 2D 连续" (lambda () (let ((s 0.0d0))
                                        (dotimes (i 1000000)
                                          (incf s (vt-ref a (floor i 1000) (mod i 1000))))
                                        s))
           :elems 1000000)
    (bench "vt-ref 2D 转置视图" (lambda () (let ((s 0.0d0))
                                             (dotimes (i 1000000)
                                               (incf s (vt-ref at (floor i 1000) (mod i 1000))))
                                             s))
           :elems 1000000)
    (bench "setf vt-ref 2D" (lambda () (dotimes (i 1000000)
                                         (setf (vt-ref a (floor i 1000) (mod i 1000)) 1.5d0)))
           :elems 1000000)
    (bench "%vt-ref-raw 1D（内核下界）"
           (lambda () (dotimes (i 1000000) (setf *sink* (clvt::%vt-ref-raw a1 i))))
           :elems 1000000)

    (header "[2] 切片视图创建（1M 次）")
    (bench "vt-slice 行 a[i,:]" (lambda () (dotimes (i 1000000)
                                            (setf *sink* (vt-slice a (list (mod i 1000)) '(:all)))))
           :elems 1000000)
    (bench "vt-slice 块 a[i:i+10,5:200:3]"
           (lambda () (dotimes (i 1000000)
                        (setf *sink* (vt-slice a (list (mod i 900) (+ (mod i 900) 10)) '(5 200 3)))))
           :elems 1000000)
    (bench "vt-slice 全选" (lambda () (dotimes (i 1000000)
                                       (setf *sink* (vt-slice a '(:all) '(:all)))))
           :elems 1000000)
    (bench "vt-slice 3D :newa" (lambda () (dotimes (i 1000000)
                                           (setf *sink* (vt-slice a '(:newa) '(:all) '(:all)))))
           :elems 1000000)

    (header "[3] 视图操作（1M 次）")
    (bench "vt-transpose 2D" (lambda () (dotimes (i 1000000) (setf *sink* (vt-transpose a))))
           :elems 1000000)
    (bench "vt-reshape 2D→1D" (lambda () (dotimes (i 1000000)
                                          (setf *sink* (vt-reshape a '(1000000)))))
           :elems 1000000)
    (bench "vt-flip axis0" (lambda () (dotimes (i 1000000) (setf *sink* (vt-flip a :axis 0))))
           :elems 1000000)
    (bench "vt-contiguous-p 连续" (lambda () (dotimes (i 1000000)
                                              (setf *sink* (vt-contiguous-p a))))
           :elems 1000000)
    (bench "vt-contiguous-p 转置" (lambda () (dotimes (i 1000000)
                                              (setf *sink* (vt-contiguous-p at))))
           :elems 1000000)
    (bench "vt-out-writable-p 转置" (lambda () (dotimes (i 1000000)
                                                (setf *sink* (clvt::vt-out-writable-p at))))
           :elems 1000000)

    (header "[4] 连续化（1M 元素）")
    (bench "vt-contiguous 转置 f64" (lambda () (vt-contiguous at)) :elems 1000000)
    (bench "vt-contiguous 步进2 f64 (500k)" (lambda () (vt-contiguous av)) :elems 500000)
    (bench "vt-contiguous 转置 c128" (lambda () (vt-contiguous c8t)) :elems 1000000)
    (bench "vt-copy 转置 f64" (lambda () (vt-copy at)) :elems 1000000)
    (bench "vt-astype 转置 f64→f32" (lambda () (vt-astype at :float32)) :elems 1000000)

    (header "[5] 非连续逐元素 / 归约（1M，对照用）")
    (bench "vt-+ 连续" (lambda () (vt-+ a a)) :elems 1000000)
    (bench "vt-+ 转置" (lambda () (vt-+ at at)) :elems 1000000)
    (bench "vt-sum 连续" (lambda () (vt-sum a)) :elems 1000000)
    (bench "vt-sum 转置" (lambda () (vt-sum at)) :elems 1000000)

    (header "[6] 拼接")
    (bench "concatenate 8×(1000×1000) ax0 连续"
           (lambda () (apply #'vt-concatenate 0 big)) :elems 8000000)
    (bench "concatenate 8×(1000×1000) ax1 连续"
           (lambda () (apply #'vt-concatenate 1 big)) :elems 8000000)
    (bench "concatenate 64 行视图 ax0"
           (lambda () (apply #'vt-concatenate 0 rows)) :elems 64000)
    (bench "concatenate 64 步进行视图 ax0"
           (lambda () (apply #'vt-concatenate 0 rowsv)) :elems 32000)
    (bench "concatenate 8 转置视图 ax0"
           (lambda () (apply #'vt-concatenate 0 nc)) :elems 8000000)
    (bench "concatenate 200×(10×10) ax0"
           (lambda () (apply #'vt-concatenate 0 small)) :elems 20000)
    (bench "vstack 64 行视图" (lambda () (apply #'vt-vstack rows)) :elems 64000)
    (bench "hstack 64 行视图" (lambda () (apply #'vt-hstack rows)) :elems 64000)

    (format t "~%完成。绝对值仅供同机前后对比；跨机请看~%")
    (format t "test/access-view-perf-test.lisp 的 B 段倍率护栏。~%")
    (finish-output)))

(run)
