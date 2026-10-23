;;;; astype-cast-test.lisp — vt-astype 转换矩阵 + 复数快路径回归
;;;;
;;;; 覆盖四件事（CONVENTIONS §16.4 转换契约 / §16.14 执行层声明策略）：
;;;;   1) 矩阵对照：10 种逻辑 dtype × 10 种目标 dtype = 100 格，期望值由
;;;;      test/gen_astype_cast_expected.py 从本机 numpy 实测生成
;;;;      （test/astype-cast-expected.sexp）。
;;;;   2) 契约断言：返回连续副本、形状保持、dtype 精确、复数→实数取实部、
;;;;      实数→复数虚部补 0、NaN/Inf→整数返回 0（§7.2）、符号零保号（§16.34.3）。
;;;;   3) 非连续等价：转置 / 翻转 / 切片视图的转换结果必须与「先物化连续副本
;;;;      再转换」逐位相同（铁律 1：连续性只影响选路，不影响结果）。
;;;;   4) 性能护栏：复合型块拷贝、复数↔实数转换相对实数线的倍率上限。
;;;;      优化前复数转换走「泛型 aref + funcall」单趟游走，实测 17~41 ms/1M
;;;;      （实数线 1~3 ms）；特化后 1~4 ms。护栏按**倍率**而非绝对毫秒，
;;;;      避免与机器速度耦合。
;;;;
;;;; 用法：sbcl --dynamic-space-size 4096 --non-interactive --load test/astype-cast-test.lisp

(require :asdf)
(defparameter cl-user::*clvt-test-dir*
  (make-pathname :name nil :type nil :defaults *load-truename*))
(pushnew (merge-pathnames "../" cl-user::*clvt-test-dir*) asdf:*central-registry* :test #'equal)
(handler-bind ((warning #'muffle-warning)) (asdf:load-system :clvt))

(in-package :clvt)

(defparameter *clvt-test-dir* cl-user::*clvt-test-dir*)

;;; ------------------------------------------------------------------
;;; 骨架
;;; ------------------------------------------------------------------

(defvar *pass* 0)
(defvar *fail* 0)
(defvar *failures* '())

(defun expect-true (name form)
  (if form
      (incf *pass*)
      (progn (incf *fail*)
             (push name *failures*)
             (format t "~&FAIL ~a~%" name))))

(defun expect-equal (name got want)
  (if (equal got want)
      (incf *pass*)
      (progn (incf *fail*)
             (push (format nil "~a: 期望 ~s 实得 ~s" name want got) *failures*)
             (format t "~&FAIL ~a: 期望 ~s 实得 ~s~%" name want got))))

(defun num= (a b tol)
  "数值比较：复数按分量递归；NaN 只与 NaN 相等；其余按容差。
   测试进程不屏蔽浮点陷阱（§16.34.3），故 NaN 必须走 float-nan-p，
   不能用 (= a a) —— 那会直接抛 FLOATING-POINT-INVALID-OPERATION。"
  (cond
    ((and (complexp a) (complexp b))
     (and (num= (realpart a) (realpart b) tol)
          (num= (imagpart a) (imagpart b) tol)))
    ((or (complexp a) (complexp b)) nil)
    ((and (floatp a) (sb-ext:float-nan-p a))
     (and (floatp b) (sb-ext:float-nan-p b)))
    ((and (floatp b) (sb-ext:float-nan-p b)) nil)
    ((and (floatp a) (floatp b) (or (> (abs a) 1d300) (> (abs b) 1d300)))
     (eql a b))
    (t (<= (abs (- a b)) tol))))

(defun expect-close (name got want tol)
  (if (num= got want tol)
      (incf *pass*)
      (progn (incf *fail*)
             (push (format nil "~a: 期望 ~s 实得 ~s (tol ~a)" name want got tol) *failures*)
             (format t "~&FAIL ~a: 期望 ~s 实得 ~s~%" name want got))))

(defun dst-tol (dst)
  "按目标 dtype 分级的容差（§10.4）：float64/complex128 1e-12、
   float32/complex64 1e-6，整数目标要求逐位相等（0 容差）。"
  (case dst
    ((:float64 :complex128) 1d-12)
    ((:float32 :complex64) 1d-6)
    (t 0)))

(defun neg-zero-p (x)
  "判断 x 是否为 -0.0（= 判不出符号，必须查 float-sign）。"
  (and (floatp x) (zerop x) (minusp (float-sign x))))

;;; ------------------------------------------------------------------
;;; 1. 矩阵对照（10 dtype × 10 dtype = 100 格）
;;; ------------------------------------------------------------------

(defun load-cases ()
  (with-open-file (s (merge-pathnames "astype-cast-expected.sexp" *clvt-test-dir*))
    (loop for form = (read s nil :eof)
          until (eq form :eof)
          collect form)))

(defun run-matrix ()
  (format t "~&--- 1. 转换矩阵（10 dtype × 10 dtype）---~%")
  (dolist (case (load-cases))
    (destructuring-bind (key src dst mode vals) case
      (declare (ignore mode))
      (let* ((in (vt-from-sequence vals :dtype src))
             (out (vt-astype in dst))
             (got (vt-to-list out))
             (tol (dst-tol dst)))
        (expect-equal (format nil "~a dtype" key) (vt-dtype out) dst)
        (expect-equal (format nil "~a shape" key) (vt-shape out) (list (length vals)))
        (expect-true (format nil "~a 连续" key) (vt-contiguous-p out))
        (expect-true (format nil "~a 独立缓冲" key) (not (eq (vt-data out) (vt-data in))))
        (expect-equal (format nil "~a 元素个数" key) (length got) (length vals))
        (if (zerop tol)
            (expect-equal (format nil "~a 值" key) got vals)
            (expect-true (format nil "~a 值" key)
                         (every (lambda (g w) (num= g w tol)) got vals)))))))

;;; ------------------------------------------------------------------
;;; 2. 契约断言
;;; ------------------------------------------------------------------

(defun run-contracts ()
  (format t "~&--- 2. 契约断言 ---~%")
  (let ((c128 (vt-from-sequence (list #C(1.5d0 2.5d0) #C(-1.5d0 -2.5d0)) :dtype :complex128))
        (c64  (vt-from-sequence (list #C(1.5s0 2.5s0)) :dtype :complex64))
        (f64  (vt-from-sequence (list 1.5d0) :dtype :float64)))

    ;; ---- 复数 → 实数/整数：取实部，虚部丢弃（§16.4，对标 numpy ComplexWarning）
    (expect-equal "c128->f64 取实部"
                  (vt-to-list (vt-astype c128 :float64)) (list 1.5d0 -1.5d0))
    (expect-equal "c128->i64 取实部后截断"
                  (vt-to-list (vt-astype (vt-from-sequence (list #C(2.75d0 -9.0d0)) :dtype :complex128)
                                         :int64))
                  (list 2))
    (expect-equal "c128->i64 负数向零截断"
                  (vt-to-list (vt-astype (vt-from-sequence (list #C(-2.75d0 9.0d0)) :dtype :complex128)
                                         :int64))
                  (list -2))
    (expect-equal "c64->f32 取实部"
                  (vt-to-list (vt-astype c64 :float32)) (list 1.5s0))
    (expect-equal "c128->int8 回绕"
                  (vt-to-list (vt-astype (vt-from-sequence (list #C(200.0d0 0.0d0)) :dtype :complex128)
                                         :int8))
                  (list -56))

    ;; ---- 实数/整数 → 复数：虚部补 +0.0
    (let ((r (vt-astype f64 :complex128)))
      (expect-equal "f64->c128 dtype" (vt-dtype r) :complex128)
      (expect-equal "f64->c128 值" (vt-to-list r) (list #C(1.5d0 0.0d0)))
      (expect-true "f64->c128 虚部是 +0.0（不是 -0.0）"
                   (not (neg-zero-p (imagpart (first (vt-to-list r)))))))
    (expect-equal "i64->c64 值"
                  (vt-to-list (vt-astype (vt-from-sequence (list 7) :dtype :int64) :complex64))
                  (list #C(7.0s0 0.0s0)))

    ;; ---- 复数 → 复数：分量各自降/升精度，虚部保留
    (expect-equal "c128->c64 保留虚部"
                  (vt-to-list (vt-astype c128 :complex64))
                  (list #C(1.5s0 2.5s0) #C(-1.5s0 -2.5s0)))
    (expect-equal "c64->c128 保留虚部"
                  (vt-to-list (vt-astype c64 :complex128)) (list #C(1.5d0 2.5d0)))

    ;; ---- 符号零：转换不得把 -0.0 洗成 +0.0（§16.34.3）
    (let* ((z (vt-from-sequence (list #C(-0.0d0 -0.0d0)) :dtype :complex128))
           (rf (first (vt-to-list (vt-astype z :float64))))
           (rc (first (vt-to-list (vt-astype z :complex64))))
           (zc (first (vt-to-list (vt-astype z :complex128)))))
      (expect-true "c128->f64 实部保 -0.0 符号" (neg-zero-p rf))
      (expect-true "c128->c64 实部保 -0.0 符号" (neg-zero-p (realpart rc)))
      (expect-true "c128->c64 虚部保 -0.0 符号" (neg-zero-p (imagpart rc)))
      (expect-true "c128->c128 实部保 -0.0 符号" (neg-zero-p (realpart zc))))

    ;; ---- NaN / Inf：浮点目标逐位传播；整数目标返回 0（§7.2 本库约定）
    (let* ((nan +vt-dfloat-nan+)
           (src (vt-zeros (list 4) :dtype :complex128)))
      (setf (aref (vt-data src) 0) (complex nan 0d0)
            (aref (vt-data src) 1) (complex 0d0 nan)
            (aref (vt-data src) 2) (complex 1d300 0d0)
            (aref (vt-data src) 3) (complex 0d0 -1d300))
      (let ((to-f64 (vt-to-list (vt-astype src :float64)))
            (to-i64 (vt-to-list (vt-astype src :int64)))
            (to-c64 (vt-to-list (vt-astype src :complex64))))
        (expect-true "c128->f64 NaN 传播（实部）" (sb-ext:float-nan-p (first to-f64)))
        (expect-true "c128->f64 虚部 NaN 不影响实部" (zerop (second to-f64)))
        (expect-true "c128->c64 NaN 传播（实部）"
                     (sb-ext:float-nan-p (realpart (first to-c64))))
        (expect-true "c128->c64 NaN 传播（虚部）"
                     (sb-ext:float-nan-p (imagpart (second to-c64))))
        (expect-equal "c128->i64 NaN/Inf 一律 0（§7.2）" to-i64 (list 0 0 0 0))))

    ;; ---- 尺寸与边界
    (let ((empty (vt-astype (vt-zeros (list 0 3) :dtype :complex128) :float64)))
      (expect-equal "空张量 shape" (vt-shape empty) (list 0 3))
      (expect-equal "空张量 dtype" (vt-dtype empty) :float64))
    (let ((one (vt-astype (vt-zeros (list 1) :dtype :complex64) :complex128)))
      (expect-equal "长度 1" (vt-to-list one) (list #C(0.0d0 0.0d0))))
    (expect-equal "rank-0 标量 c128->f64"
                  (vt-to-list (vt-astype (vt-reshape c64 (list)) :float64))
                  1.5d0)

    ;; ---- 广播视图（stride=0 轴）必须被物化，不能共享
    (let* ((base (vt-from-sequence (list #C(1.0d0 1.0d0) #C(2.0d0 2.0d0)) :dtype :complex128))
           (bc (vt-broadcast-to base (list 3 2)))
           (r (vt-astype bc :complex64)))
      (expect-equal "广播视图 shape" (vt-shape r) (list 3 2))
      (expect-equal "广播视图 值"
                    (vt-to-list r)
                    (list (list #C(1.0s0 1.0s0) #C(2.0s0 2.0s0))
                          (list #C(1.0s0 1.0s0) #C(2.0s0 2.0s0))
                          (list #C(1.0s0 1.0s0) #C(2.0s0 2.0s0)))))))

;;; ------------------------------------------------------------------
;;; 3. 非连续视图等价（铁律 1/2）
;;; ------------------------------------------------------------------

(defun same-values-p (a b tol)
  (labels ((walk (x y)
             (cond ((and (listp x) (listp y))
                    (and (= (length x) (length y))
                         (every #'walk x y)))
                   ((and (listp x) (not (listp y))) nil)
                   ((and (listp y) (not (listp x))) nil)
                   (t (num= x y tol)))))
    (walk a b)))

(defun run-strided ()
  (format t "~&--- 3. 非连续视图等价 ---~%")
  (let* ((rows (list (mapcar (lambda (v) (complex v (* 2 v))) '(1.0d0 2.0d0 -3.0d0))
                     (mapcar (lambda (v) (complex v (- (* 2 v)))) '(4.0d0 -5.0d0 0.5d0))
                     (mapcar (lambda (v) (complex v v)) '(7.0d0 0.0d0 -1.0d0))))
         (base (vt-from-sequence rows :dtype :complex128))
         (views (list (cons "转置" (vt-transpose base))
                      (cons "翻转" (vt-flip base :axis 0))
                      (cons "切片" (vt-slice base (list 0 3 2) (list 0 3 1)))
                      (cons "广播" (vt-broadcast-to
                                    (vt-slice base (list 0 1) (list 0 3))
                                    (list 4 3))))))
    (dolist (v views)
      (dolist (dst '(:complex128 :complex64 :float64 :float32 :int64 :int32 :int8 :uint8))
        (let* ((view (cdr v))
               (got (vt-to-list (vt-astype view dst)))
               ;; 参照：先把视图物化成连续副本，再转换（两次都是已特化的路径）
               (want (vt-to-list (vt-astype (vt-copy view) dst))))
          (expect-true (format nil "~a视图 -> ~a 与连续副本一致" (car v) dst)
                       (same-values-p got want (dst-tol dst))))))
    ;; 负步长 + 非零 offset：起点也必须参与寻址
    (let* ((sl (vt-slice base (list 2 0 -1) (list 1 3 1)))
           (got (vt-to-list (vt-astype sl :complex64)))
           (want (vt-to-list (vt-astype (vt-copy sl) :complex64))))
      (expect-true "负步长 + offset 视图一致" (same-values-p got want 1d-6)))))

;;; ------------------------------------------------------------------
;;; 4. 性能护栏（倍率，不看绝对毫秒）
;;; ------------------------------------------------------------------

(defparameter *perf-n* 1000000)

(defun bench-ms (fn &key (runs 3))
  "跑 RUNS+1 次取最好（GC 前置），返回 double 毫秒 —— 不做整数截断，
   否则 1ms 级的实数线会被算成 0，倍率失去意义。"
  (let ((best most-positive-fixnum))
    (dotimes (_ (1+ runs))
      (sb-ext:gc :full t)
      (let ((t0 (get-internal-real-time)))
        (funcall fn)
        (setf best (min best (- (get-internal-real-time) t0)))))
    (/ (* 1000.0d0 best) internal-time-units-per-second)))

(defun run-perf ()
  (format t "~&--- 4. 性能护栏（n=~d，倍率上限）---~%" *perf-n*)
  (let* ((n *perf-n*)
         (z128 (vt-+ (vt-zeros (list n) :dtype :complex128) #C(1.0d0 0.5d0)))
         (d64  (vt-+ (vt-zeros (list n)) 1.0d0))
         (f64-f64 (bench-ms (lambda () (vt-astype d64 :float64))))
         (f64-f32 (bench-ms (lambda () (vt-astype d64 :float32))))
         (c128-c128 (bench-ms (lambda () (vt-astype z128 :complex128))))
         (c128-c64 (bench-ms (lambda () (vt-astype z128 :complex64))))
         (c128-f64 (bench-ms (lambda () (vt-astype z128 :float64))))
         (tz (vt-transpose (vt-reshape z128 (list 1000 1000))))
         (td (vt-transpose (vt-reshape d64 (list 1000 1000))))
         (t-c128 (bench-ms (lambda () (vt-astype tz :complex128))))
         (t-d64  (bench-ms (lambda () (vt-astype td :float64)))))
    (format t "~&    f64->f64 ~,2f ms | c128->c128 ~,2f ms | c128->c64 ~,2f ms | c128->f64 ~,2f ms~%"
            f64-f64 c128-c128 c128-c64 c128-f64)
    (format t "~&    转置视图: f64T->f64 ~,2f ms | c128T->c128 ~,2f ms~%" t-d64 t-c128)
    (expect-true (format nil "c128->c128 ≤ 8× f64->f64（实测 ~,1fx）"
                         (/ c128-c128 (max f64-f64 0.001d0)))
                 (<= c128-c128 (* 8.0d0 (max f64-f64 0.001d0))))
    (expect-true (format nil "c128->c64 ≤ 10× f64->f32（实测 ~,1fx）"
                         (/ c128-c64 (max f64-f32 0.001d0)))
                 (<= c128-c64 (* 10.0d0 (max f64-f32 0.001d0))))
    (expect-true (format nil "c128->f64 ≤ 8× f64->f64（实测 ~,1fx）"
                         (/ c128-f64 (max f64-f64 0.001d0)))
                 (<= c128-f64 (* 8.0d0 (max f64-f64 0.001d0))))
    (expect-true (format nil "c128T->c128 ≤ 8× f64T->f64（实测 ~,1fx）"
                         (/ t-c128 (max t-d64 0.001d0)))
                 (<= t-c128 (* 8.0d0 (max t-d64 0.001d0))))))

;;; ------------------------------------------------------------------
;;; 入口
;;; ------------------------------------------------------------------

(format t "~&=====================================~%")
(format t "vt-astype 转换矩阵 + 复数快路径 (CONVENTIONS §16.4 / §16.14)~%")
(format t "=====================================~%")
(run-matrix)
(run-contracts)
(run-strided)
(run-perf)
(format t "~&=====================================~%")
(format t "Total: ~d / Pass: ~d / Fail: ~d~%" (+ *pass* *fail*) *pass* *fail*)
(format t "PASS=~d FAIL=~d~%" *pass* *fail*)
(when *failures*
  (format t "~&--- 失败清单 ---~%")
  (dolist (f (reverse *failures*)) (format t "  · ~a~%" f)))
(when (plusp *fail*) (sb-ext:exit :code 1))
