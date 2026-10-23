;;;; complex-scan-axis-test.lisp — 复数「沿轴扫描」回归网（§16.29）
;;;;
;;;; 背景：vt-cumsum / vt-cumprod / vt-nancumsum / vt-nancumprod 的连续内核
;;;; %CX-SCAN-RUN 的 axis 分支漏了一层 fiber 遍历（没有遍历 AX 轴之后的
;;;; AX-STRIDE 个位置），导致 **axis < rank-1 时只算了第一条 fiber，其余位置
;;;; 静默为 0**。此前所有测试只测了 1-D（axis 缺省），所以一直没暴露。
;;;;
;;;; 本套件用 numpy 2.3.5 实测值做硬编码期望（数据是二进制可精确表示的小数，
;;;; 累加也是精确的，故要求**逐位相等**而不是近似）：
;;;;   第 k 个元素：re = (k mod 7)*0.5 - 1.5, im = (k mod 5)*0.25 - 0.5
;;;;
;;;; 用法：sbcl --non-interactive --load test/complex-scan-axis-test.lisp

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

;;; ------------------------------------------------------------------
;;; 数据（与 numpy 侧完全一致）
;;; ------------------------------------------------------------------

(defun zval (k)
  (complex (- (* (mod k 7) 0.5d0) 1.5d0)
           (- (* (mod k 5) 0.25d0) 0.5d0)))

(defun rval (k) (- (* (mod k 7) 0.5d0) 1.5d0))

(defun mk (dims &key (offset 0) (dtype :complex128))
  (let* ((n (reduce #'* dims :initial-value 1))
         (v (make-vt dims 0 :dtype dtype))
         (d (vt-data v)))
    (dotimes (i n)
      (setf (aref d i)
            (if (eq dtype :complex128) (zval (+ i offset))
                (if (eq dtype :complex64)
                    (complex (coerce (realpart (zval (+ i offset))) 'single-float)
                             (coerce (imagpart (zval (+ i offset))) 'single-float))
                    (rval (+ i offset))))))
    v))

(defun flat (v)
  (let* ((c (vt-contiguous (vt-ravel v)))
         (d (vt-data c)) (o (the fixnum (vt-offset c))) (n (the fixnum (vt-size c))))
    (loop for i from o below (+ o n) collect (aref d i))))

(defun bits= (v expected &optional (tol 0d0))
  "逐元素比较。TOL=0 时要求逐位相等（数据是精确二进制小数）。"
  (let ((got (flat v)))
    (and (= (length got) (length expected))
         (every (lambda (g e)
                  (if (complexp e)
                      (and (<= (abs (- (realpart g) (realpart e))) tol)
                           (<= (abs (- (imagpart g) (imagpart e))) tol))
                      (<= (abs (- g e)) tol)))
                got expected))))

;;; ------------------------------------------------------------------
;;; 1. cumsum / cumprod：2-D 两个轴（numpy 2.3.5 实测）
;;; ------------------------------------------------------------------

(defparameter *cumsum-ax0*
  '(#C(-1.5d0 -0.5d0) #C(-1.0d0 -0.25d0) #C(-0.5d0 0.0d0) #C(0.0d0 0.25d0)
    #C(-1.0d0 0.0d0) #C(0.0d0 -0.75d0) #C(1.0d0 -0.25d0) #C(-1.5d0 0.25d0)
    #C(-2.0d0 0.25d0) #C(-0.5d0 -0.25d0) #C(1.0d0 -0.75d0) #C(-1.0d0 0.0d0)))

(defparameter *cumsum-ax1*
  '(#C(-1.5d0 -0.5d0) #C(-2.5d0 -0.75d0) #C(-3.0d0 -0.75d0) #C(-3.0d0 -0.5d0)
    #C(0.5d0 0.5d0) #C(1.5d0 0.0d0) #C(3.0d0 -0.25d0) #C(1.5d0 -0.25d0)
    #C(-1.0d0 0.25d0) #C(-1.5d0 0.75d0) #C(-1.5d0 0.25d0) #C(-1.0d0 0.0d0)))

(defparameter *cumprod-ax0*
  '(#C(-1.5d0 -0.5d0) #C(-1.0d0 -0.25d0) #C(-0.5d0 0.0d0) #C(0.0d0 0.25d0)
    #C(-0.5d0 -1.0d0) #C(-1.125d0 0.25d0) #C(-0.75d0 0.125d0) #C(0.0d0 -0.375d0)
    #C(0.75d0 0.875d0) #C(0.4375d0 -0.6875d0) #C(0.0625d0 0.375d0) #C(-0.09375d0 -0.1875d0)))

(defparameter *cumprod-ax1*
  '(#C(-1.5d0 -0.5d0) #C(1.375d0 0.875d0) #C(-0.6875d0 -0.4375d0) #C(0.109375d0 -0.171875d0)
    #C(0.5d0 0.5d0) #C(0.75d0 0.25d0) #C(1.1875d0 0.1875d0) #C(-1.78125d0 -0.28125d0)
    #C(-1.0d0 0.25d0) #C(0.375d0 -0.625d0) #C(-0.3125d0 -0.1875d0) #C(-0.203125d0 -0.015625d0)))

;;; 3-D（2,3,4）：三个轴
(defparameter *cumsum-3d-ax0*
  '(#C(-1.5d0 -0.5d0) #C(-1.0d0 -0.25d0) #C(-0.5d0 0.0d0) #C(0.0d0 0.25d0)
    #C(0.5d0 0.5d0) #C(1.0d0 -0.5d0) #C(1.5d0 -0.25d0) #C(-1.5d0 0.0d0)
    #C(-1.0d0 0.25d0) #C(-0.5d0 0.5d0) #C(0.0d0 -0.5d0) #C(0.5d0 -0.25d0)
    #C(-0.5d0 -0.5d0) #C(0.5d0 0.0d0) #C(-2.0d0 0.5d0) #C(-1.0d0 -0.25d0)
    #C(0.0d0 0.25d0) #C(1.0d0 -0.5d0) #C(2.0d0 0.0d0) #C(-0.5d0 0.5d0)
    #C(0.5d0 -0.25d0) #C(-2.0d0 0.25d0) #C(-1.0d0 -0.5d0) #C(0.0d0 0.0d0)))

(defparameter *cumsum-3d-ax1*
  '(#C(-1.5d0 -0.5d0) #C(-1.0d0 -0.25d0) #C(-0.5d0 0.0d0) #C(0.0d0 0.25d0)
    #C(-1.0d0 0.0d0) #C(0.0d0 -0.75d0) #C(1.0d0 -0.25d0) #C(-1.5d0 0.25d0)
    #C(-2.0d0 0.25d0) #C(-0.5d0 -0.25d0) #C(1.0d0 -0.75d0) #C(-1.0d0 0.0d0)
    #C(1.0d0 0.0d0) #C(1.5d0 0.25d0) #C(-1.5d0 0.5d0) #C(-1.0d0 -0.5d0)
    #C(0.5d0 -0.25d0) #C(1.5d0 0.25d0) #C(-1.0d0 0.75d0) #C(0.0d0 0.0d0)
    #C(2.0d0 -0.75d0) #C(0.0d0 0.0d0) #C(-2.0d0 0.75d0) #C(-0.5d0 0.25d0)))

(defparameter *cumsum-3d-ax2*
  '(#C(-1.5d0 -0.5d0) #C(-2.5d0 -0.75d0) #C(-3.0d0 -0.75d0) #C(-3.0d0 -0.5d0)
    #C(0.5d0 0.5d0) #C(1.5d0 0.0d0) #C(3.0d0 -0.25d0) #C(1.5d0 -0.25d0)
    #C(-1.0d0 0.25d0) #C(-1.5d0 0.75d0) #C(-1.5d0 0.25d0) #C(-1.0d0 0.0d0)
    #C(1.0d0 0.0d0) #C(2.5d0 0.25d0) #C(1.0d0 0.75d0) #C(0.0d0 0.25d0)
    #C(-0.5d0 -0.25d0) #C(-0.5d0 -0.25d0) #C(0.0d0 0.0d0) #C(1.0d0 0.5d0)
    #C(1.5d0 -0.5d0) #C(0.0d0 -0.75d0) #C(-1.0d0 -0.75d0) #C(-1.5d0 -0.5d0)))

;;; 转置视图（4,3）沿 axis=0
(defparameter *cumsum-T-ax0*
  '(#C(-1.5d0 -0.5d0) #C(0.5d0 0.5d0) #C(-1.0d0 0.25d0)
    #C(-2.5d0 -0.75d0) #C(1.5d0 0.0d0) #C(-1.5d0 0.75d0)
    #C(-3.0d0 -0.75d0) #C(3.0d0 -0.25d0) #C(-1.5d0 0.25d0)
    #C(-3.0d0 -0.5d0) #C(1.5d0 -0.25d0) #C(-1.0d0 0.0d0)))

(defun run-scan-cases ()
  (format t "~&--- 1. cumsum / cumprod 沿轴（2-D）---~%")
  (let ((a (mk '(3 4))))
    (report "cumsum axis=0" (bits= (vt-cumsum a :axis 0) *cumsum-ax0*))
    (report "cumsum axis=1" (bits= (vt-cumsum a :axis 1) *cumsum-ax1*))
    (report "cumsum axis=-1（等价 axis=1）" (bits= (vt-cumsum a :axis -1) *cumsum-ax1*))
    (report "cumsum axis=-2（等价 axis=0）" (bits= (vt-cumsum a :axis -2) *cumsum-ax0*))
    (report "cumprod axis=0" (bits= (vt-cumprod a :axis 0) *cumprod-ax0*))
    (report "cumprod axis=1" (bits= (vt-cumprod a :axis 1) *cumprod-ax1*))
    (report "cumsum 形状不变 (3,4)" (equal (vt-shape (vt-cumsum a :axis 0)) '(3 4)))
    (report "cumsum dtype 保持 complex128" (eq (vt-dtype (vt-cumsum a :axis 0)) :complex128)))

  (format t "~&--- 2. 3-D 三个轴（最容易暴露「只算第一条 fiber」）---~%")
  (let ((c3 (mk '(2 3 4))))
    (report "cumsum 3-D axis=0" (bits= (vt-cumsum c3 :axis 0) *cumsum-3d-ax0*))
    (report "cumsum 3-D axis=1" (bits= (vt-cumsum c3 :axis 1) *cumsum-3d-ax1*))
    (report "cumsum 3-D axis=2" (bits= (vt-cumsum c3 :axis 2) *cumsum-3d-ax2*)))

  (format t "~&--- 3. 非连续输入（转置视图）---~%")
  (let ((a (mk '(3 4))))
    (report "cumsum(transpose) axis=0" (bits= (vt-cumsum (vt-transpose a) :axis 0) *cumsum-T-ax0*))
    (report "cumsum(transpose) 形状 (4,3)"
            (equal (vt-shape (vt-cumsum (vt-transpose a) :axis 0)) '(4 3))))

  (format t "~&--- 4. nan 族（无 NaN 时应与普通扫描逐位相同）---~%")
  (let ((a (mk '(3 4))))
    (report "nancumsum axis=0 == cumsum axis=0"
            (bits= (vt-nancumsum a :axis 0) *cumsum-ax0*))
    (report "nancumsum axis=1 == cumsum axis=1"
            (bits= (vt-nancumsum a :axis 1) *cumsum-ax1*))
    (report "nancumprod axis=0 == cumprod axis=0"
            (bits= (vt-nancumprod a :axis 0) *cumprod-ax0*)))

  (format t "~&--- 5. complex64 与 complex128 数值一致 ---~%")
  (let ((a (mk '(3 4)))
        (a64 (mk '(3 4) :dtype :complex64)))
    (report "c64 cumsum axis=0 ≈ c128" (bits= (vt-cumsum a64 :axis 0) *cumsum-ax0* 1d-6))
    (report "c64 cumsum axis=1 ≈ c128" (bits= (vt-cumsum a64 :axis 1) *cumsum-ax1* 1d-6))
    (report "c64 结果 dtype 仍是 complex64"
            (eq (vt-dtype (vt-cumsum a64 :axis 0)) :complex64)))

  (format t "~&--- 6. 退化与 :out ---~%")
  (report "axis 长度 1 的轴（(1,3) axis=0）"
          (bits= (vt-cumsum (mk '(1 3)) :axis 0) (flat (mk '(1 3)))))
  (report "0 长度轴（(0,3) axis=0）结果为空"
          (zerop (vt-size (vt-cumsum (mk '(0 3)) :axis 0))))
  (report "axis=nil 展平扫描（1-D，形状 (12)）"
          (let ((r (vt-cumsum (mk '(3 4)))))
            (and (equal (vt-shape r) '(12))
                 (= (vt-size r) 12))))
  (report ":out 契约（连续 out）"
          (let* ((out (make-vt '(3 4) 0 :dtype :complex128))
                 (r (vt-cumsum (mk '(3 4)) :axis 0 :out out)))
            (and (eq r out) (bits= r *cumsum-ax0*))))
  (report ":out 非连续（列切片）也能写对"
          (let* ((big (make-vt '(3 8) 0 :dtype :complex128))
                 (sub (vt-slice big '(:all) '(0 4)))
                 (r (vt-cumsum (mk '(3 4)) :axis 0 :out sub)))
            (and (eq r sub) (bits= r *cumsum-ax0*)))))

;;; ------------------------------------------------------------------
;;; 7. 实数回归（同一个内核家族，别被改坏）
;;; ------------------------------------------------------------------

(defun run-real-regression ()
  (format t "~&--- 7. 实数路径回归 ---~%")
  (let* ((a (mk '(3 4) :dtype :float64))
         (want '(#C(-1.5 0) #C(-1.0 0) #C(-0.5 0) #C(0.0 0)
                 #C(-1.0 0) #C(0.0 0) #C(1.0 0) #C(-1.5 0)
                 #C(-2.0 0) #C(-0.5 0) #C(1.0 0) #C(-1.0 0))))
    (report "实数 cumsum axis=0（回归）"
            (bits= (vt-cumsum a :axis 0)
                   (loop for z in want collect (realpart z))))
    (report "实数 cumsum axis=1 首行末元素 = -1.5-1.0-0.5+0.0（回归）"
            (< (abs (- (vt-item (vt-ref (vt-cumsum a :axis 1) 0 3)) -3.0d0)) 1d-12))))

;;; ------------------------------------------------------------------
;;; 入口
;;; ------------------------------------------------------------------

(format t "~&=====================================")
(format t "~&复数沿轴扫描回归（§16.29）")
(format t "~&=====================================~%")
(run-scan-cases)
(run-real-regression)

(format t "~&=====================================~%")
(format t "Total: ~d / Pass: ~d / Fail: ~d~%" (+ *pass* *fail*) *pass* *fail*)
(format t "PASS=~d FAIL=~d~%" *pass* *fail*)
(when *failures*
  (format t "~&--- 失败清单 ---~%")
  (dolist (f (reverse *failures*)) (format t "  · ~a~%" f)))
(when (plusp *fail*) (sb-ext:exit :code 1))
