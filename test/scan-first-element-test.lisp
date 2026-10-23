;;;; scan-first-element-test.lisp — 累积扫描的「首元素」语义回归（第十四轮）
;;;;
;;;; 背景（CONVENTIONS §16.30.3）：
;;;;   numpy 的 **accumulate**（cumsum/cumprod）与 **reduce**（sum/prod）在首元素上
;;;;   方向相反 —— reduce 从单位元起算，accumulate 的**首个输出是输入本身**。
;;;;   numpy 实测：
;;;;       np.prod([inf])          => inf          (reduce 单位元参与)
;;;;       np.cumsum([-0.0])       => [-0.]        (accumulate 直拷，符号位保留)
;;;;       np.cumsum([-0.0, 0.0])  => [-0.  0.]    (0 + -0.0 = +0.0)
;;;;
;;;;   复数线（%CX-SCAN-RUN）在第十轮已改为「首元素直拷」；**实数线一直是
;;;;   从单位元起算**，于是 `(vt-cumsum -0.0)` 静默返回 +0.0 —— 同一份代码
;;;;   在实数与复数上给了不同的位模式。本套件把实数线也钉住。
;;;;
;;;; 为什么这个 bug 能活这么久：-0.0 与 +0.0 在 `=` / `eql` 下相等，
;;;; `(equal 0.0d0 -0.0d0)` 也只是 NIL（因为位模式不同才会 NIL，但
;;;; 大多数断言写的是数值相等），所以只有显式查 `float-sign` 才看得见。
;;;; 唯一可观测的后果是 `1/x`（→ ∓Inf）与 `log(x)`（→ ∓Inf）、分支切割。
;;;;
;;;; 期望值全部来自本机 numpy 2.4.6 实测（CONVENTIONS §10.4 禁手写）。
;;;;
;;;; 用法：sbcl --non-interactive --load test/scan-first-element-test.lisp

(require :asdf)
(defparameter *clvt-test-dir* (make-pathname :name nil :type nil :defaults *load-truename*))
(pushnew (merge-pathnames "../" *clvt-test-dir*) asdf:*central-registry* :test #'equal)
(handler-bind ((warning #'muffle-warning)) (asdf:load-system :clvt))

(in-package :clvt)

;;; ------------------------------------------------------------------
;;; 骨架
;;; ------------------------------------------------------------------

(defvar *pass* 0)
(defvar *fail* 0)
(defvar *failures* '())

(defun report (name ok &optional detail)
  (if ok
      (incf *pass*)
      (progn (incf *fail*) (push name *failures*)
             (format t "~&FAIL ~a ~@[— ~a~]" name detail))))

;;; 逐位比较：数值部分要求 `=` 成立，**符号零单独判**。
(defun neg-zero-p (x)
  "X 是负零（符号位为 1 的零）吗？"
  (and (realp x) (zerop x) (minusp (float-sign (coerce x 'double-float)))))

(defun sign-of (x)
  (cond ((and (floatp x) (sb-ext:float-nan-p x)) :nan)
        ((neg-zero-p x) :-0+)
        ((zerop x) :+0+)
        ((plusp x) :+)
        (t :-)))

(defun signs (v)
  (let* ((c (vt-contiguous (vt-ravel v)))
         (d (vt-data c)) (o (vt-offset c)) (n (vt-size c)))
    (loop for i from 0 below n collect (sign-of (aref d (+ o i))))))

(defun vals (v)
  (mapcar (lambda (x) (if (floatp x) (coerce x 'double-float) x))
          (vt-to-list (vt-ravel v))))

;;; NaN 一律用位级判定（sb-ext:float-nan-p）。
;;; 测试进程**故意不屏蔽浮点陷阱**（SBCL 默认全开），而 `(= x x)` / `(/= x x)`
;;; 对 NaN 会直接抛 FLOATING-POINT-INVALID-OPERATION —— CONVENTIONS §16.28.5
;;; 记的就是这个坑。
(defun nan-p (x) (and (floatp x) (sb-ext:float-nan-p x)))

(defun close-enough (a b)
  (cond ((and (floatp a) (floatp b))
         (cond ((and (nan-p a) (nan-p b)) t)
               ((or (nan-p a) (nan-p b)) nil)
               ((= a b) t)
               (t (<= (abs (- a b)) (* 1d-12 (max 1d0 (abs a) (abs b)))))))
        (t (equal a b))))

(defun same-values-p (got want)
  (and (= (length got) (length want))
       (every #'close-enough got want)))

;;; ------------------------------------------------------------------
;;; 1. 实数 cumsum：首元素直拷（-0.0 符号位）
;;; ------------------------------------------------------------------

(defun run-real-cumsum ()
  (format t "~&--- 1. 实数 cumsum 首元素 ---~%")
  (let ((a (vt-from-sequence (list -0.0d0) :dtype :float64)))
    (report "cumsum([-0.0]) 值为 -0.0"
            (equal (signs (vt-cumsum a)) '(:-0+)))
    (report "cumsum([-0.0]) 数值为 0"
            (same-values-p (vals (vt-cumsum a)) '(0d0)))
    (report "cumsum 结果与原数组不是同一缓冲"
            (not (eq (vt-data (vt-cumsum a)) (vt-data a)))))

  (let ((a (vt-from-sequence (list -0.0d0 0.0d0) :dtype :float64)))
    (report "cumsum([-0.0,0.0]) 符号 = (-0,+0)"
            (equal (signs (vt-cumsum a)) '(:-0+ :+0+)))
    (report "cumsum([-0.0,0.0]) 值 = (0,0)"
            (same-values-p (vals (vt-cumsum a)) '(0d0 0d0))))

  ;; 二维 axis=0：**每一列的首行**都是直拷（numpy signbit 实测
  ;; [[True True False] [False False False]]）
  (let ((a (vt-from-sequence '((-0.0d0 -0.0d0 1.0d0)
                              (2.0d0 3.0d0 4.0d0)) :dtype :float64)))
    (report "cumsum(2x3, axis=0) 符号表"
            (equal (signs (vt-cumsum a :axis 0))
                   '(:-0+ :-0+ :+ :+ :+ :+))
            (format nil "~a" (signs (vt-cumsum a :axis 0))))
    (report "cumsum(2x3, axis=0) 数值"
            (same-values-p (vals (vt-cumsum a :axis 0))
                           '(-0d0 -0d0 1d0 2d0 3d0 5d0)))
    (report "cumsum(2x3, axis=1) 符号表（每行首元素直拷）"
            (equal (signs (vt-cumsum a :axis 1))
                   '(:-0+ :-0+ :+ :+ :+ :+)))
    (report "cumsum(2x3, axis=1) 数值"
            (same-values-p (vals (vt-cumsum a :axis 1))
                           '(-0d0 -0d0 1d0 2d0 5d0 9d0)))
    (report "cumsum(2x3, axis=nil) 展平后符号表"
            (equal (signs (vt-cumsum a)) '(:-0+ :-0+ :+ :+ :+ :+)))
    (report "cumsum(2x3, axis=nil) 展平后数值"
            (same-values-p (vals (vt-cumsum a))
                           '(-0d0 -0d0 1d0 3d0 6d0 10d0)))
    (report "cumsum(axis=nil) 形状为 (6)"
            (equal (vt-shape (vt-cumsum a)) '(6))))

  ;; float32：符号位同样必须保留
  (let ((a (vt-from-sequence (list -0.0) :dtype :float32)))
    (report "cumsum(float32 [-0.0]) 保留符号"
            (eq (first (signs (vt-cumsum a))) :-0+))
    (report "cumsum(float32) dtype 仍为 :float32"
            (eq (vt-dtype (vt-cumsum a)) :float32)))

  ;; 轴长 1：首元素即唯一元素，退化成直拷
  (let ((a (vt-from-sequence '((-0.0d0) (2.0d0)) :dtype :float64)))
    (report "cumsum(轴长 1, axis=0) 等于输入"
            (equal (signs (vt-cumsum a :axis 0)) '(:-0+ :+))))

  ;; 负步长视图：通用路径（非连续）也必须直拷首元素
  (let ((a (vt-slice (vt-from-sequence '(1.0d0 -0.0d0) :dtype :float64) '(1 nil -1))))
    (report "cumsum(负步长视图) 首元素保留 -0.0"
            (eq (first (signs (vt-cumsum a :axis 0))) :-0+)))

  ;; 空输入
  (let ((e (vt-zeros '(0) :dtype :float64)))
    (report "cumsum(空) 返回空且不报错"
            (and (equal (vt-shape (vt-cumsum e)) '(0))
                 (null (vals (vt-cumsum e)))))))

;;; ------------------------------------------------------------------
;;; 2. 实数 cumprod：首元素直拷 + inf 的传播
;;; ------------------------------------------------------------------

(defun run-real-cumprod ()
  (format t "~&--- 2. 实数 cumprod 首元素 ---~%")
  (let ((a (vt-from-sequence (list -0.0d0) :dtype :float64)))
    (report "cumprod([-0.0]) 值为 -0.0" (eq (first (signs (vt-cumprod a))) :-0+)))
  ;; numpy: np.cumprod([inf, 0]) => [inf nan]（inf*0 = nan）
  (let ((a (vt-from-sequence (list +vt-dfloat-pos-inf+ 0.0d0) :dtype :float64)))
    (let ((r (vals (vt-cumprod a))))
      (report "cumprod([inf,0]) 第一项 = inf" (eql (first r) +vt-dfloat-pos-inf+))
      (report "cumprod([inf,0]) 第二项 = NaN（inf*0，不抛浮点陷阱）" (nan-p (second r)))))
  ;; 二维 axis=0：numpy cumprod([[-0,-0,1],[2,3,4]], axis=0) = [[-0,-0,1],[-0,-0,4]]
  (let ((a (vt-from-sequence '((-0.0d0 -0.0d0 1.0d0)
                              (2.0d0 3.0d0 4.0d0)) :dtype :float64)))
    (report "cumprod(2x3, axis=0) 首行符号表"
            (equal (signs (vt-cumprod a :axis 0))
                   '(:-0+ :-0+ :+ :-0+ :-0+ :+)))
    (report "cumprod(2x3, axis=0) 数值"
            (same-values-p (vals (vt-cumprod a :axis 0))
                           '(-0d0 -0d0 1d0 -0d0 -0d0 4d0)))))

;;; ------------------------------------------------------------------
;;; 3. nan 族：NaN 用单位元掩码，但**非 NaN 的首元素仍直拷**
;;; ------------------------------------------------------------------

(defun run-nan-family ()
  (format t "~&--- 3. nancumsum / nancumprod ---~%")
  (let ((a (vt-from-sequence (list -0.0d0) :dtype :float64)))
    (report "nancumsum([-0.0]) 保留符号"
            (eq (first (signs (vt-nancumsum a))) :-0+))
    (report "nancumprod([-0.0]) 保留符号"
            (eq (first (signs (vt-nancumprod a))) :-0+)))
  ;; numpy: np.nancumsum([nan, -0.0]) => [0. 0.]；首元素 NaN 用单位元 0（+0.0）
  (let ((a (vt-from-sequence (list +vt-dfloat-nan+ -0.0d0) :dtype :float64)))
    (report "nancumsum([nan,-0.0]) 符号 = (+0,+0)"
            (equal (signs (vt-nancumsum a)) '(:+0+ :+0+))
            (format nil "~a" (signs (vt-nancumsum a))))
    (report "nancumsum([nan,-0.0]) 值 = (0,0)"
            (same-values-p (vals (vt-nancumsum a)) '(0d0 0d0))))
  ;; numpy: np.nancumprod([nan]) => [1.]
  (let ((a (vt-from-sequence (list +vt-dfloat-nan+) :dtype :float64)))
    (report "nancumprod([nan]) = 1（单位元）"
            (same-values-p (vals (vt-nancumprod a)) '(1d0))))
  ;; numpy: np.nancumsum([[-0,-0],[3,4]], axis=0) = [[-0,-0],[3,4]]
  (let ((a (vt-from-sequence '((-0.0d0 -0.0d0) (3.0d0 4.0d0)) :dtype :float64)))
    (report "nancumsum(2x2, axis=0) 首行符号表"
            (equal (signs (vt-nancumsum a :axis 0)) '(:-0+ :-0+ :+ :+)))
    (report "nancumsum(2x2, axis=0) 数值"
            (same-values-p (vals (vt-nancumsum a :axis 0))
                           '(-0d0 -0d0 3d0 4d0)))))

;;; ------------------------------------------------------------------
;;; 4. 类型契约不回归（首元素直拷不能绕过 dtype 提升）
;;; ------------------------------------------------------------------

(defun run-dtype-contract ()
  (format t "~&--- 4. dtype 契约 ---~%")
  ;; numpy: np.cumsum(int8[100,100]) => [100 200] dtype int64
  (let* ((a (vt-astype (vt-from-sequence '(100 100) :dtype :int8) :int8))
         (r (vt-cumsum a)))
    (report "cumsum(int8) 提升为 :int64"
            (eq (vt-dtype r) :int64) (format nil "~a" (vt-dtype r)))
    (report "cumsum(int8) 不溢出（100+100=200）"
            (same-values-p (vals r) '(100 200))))
  ;; 显式 :dtype 仍然生效
  (let ((r (vt-cumsum (vt-from-sequence '(-0.0d0 1.0d0) :dtype :float64) :dtype :float32)))
    (report "cumsum 显式 :dtype :float32 生效" (eq (vt-dtype r) :float32))
    (report "cumsum 显式 :dtype 后首元素符号仍保留"
            (eq (first (signs r)) :-0+)))
  ;; 复数路径回归（第十轮已修，不能被本轮改动带坏）
  (let ((z (vt-from-sequence (list (complex -0.0d0 0.0d0) 1.0d0) :dtype :complex128)))
    (let ((r (vt-cumsum z)))
      (report "cumsum(complex) -0.0 实部符号保留"
              (neg-zero-p (realpart (vt-ref r 0))))
      (report "cumsum(complex) dtype 保持 :complex128"
              (eq (vt-dtype r) :complex128)))))

;;; ------------------------------------------------------------------
;;; 5. :out 契约（非连续目标上的首元素落点）
;;; ------------------------------------------------------------------

(defun run-out-contract ()
  (format t "~&--- 5. :out 非连续目标 ---~%")
  (let* ((base (vt-zeros '(2 6) :dtype :float64))
         (v (vt-slice base '(:all) '(0 6 2)))          ; 列视图 (2 3)，stride 2
         (a (vt-from-sequence '((-0.0d0 1.0d0 2.0d0)
                               (3.0d0 4.0d0 5.0d0)) :dtype :float64)))
    (vt-cumsum a :axis 1 :out v)
    (report "cumsum :out 列视图落点正确"
            (same-values-p (vals v) '(-0d0 1d0 3d0 3d0 7d0 12d0)))
    (report "cumsum :out 列视图首元素符号保留"
            (eq (first (signs v)) :-0+))
    (report "cumsum :out 其余位置不动"
            (same-values-p (list (vt-ref base 0 1) (vt-ref base 0 3)
                                 (vt-ref base 0 5) (vt-ref base 1 1))
                           '(0d0 0d0 0d0 0d0)))))

;;; ------------------------------------------------------------------
;;; 入口
;;; ------------------------------------------------------------------

(format t "~&=====================================")
(format t "~&累积扫描首元素语义（第十四轮）")
(format t "~&=====================================~%")
(run-real-cumsum)
(run-real-cumprod)
(run-nan-family)
(run-dtype-contract)
(run-out-contract)

(format t "~&=====================================~%")
(format t "Total: ~d / Pass: ~d / Fail: ~d~%" (+ *pass* *fail*) *pass* *fail*)
(format t "PASS=~d FAIL=~d~%" *pass* *fail*)
(when *failures*
  (format t "~&--- 失败清单 ---~%")
  (dolist (f (reverse *failures*)) (format t "  · ~a~%" f)))
(when (plusp *fail*) (sb-ext:exit :code 1))
