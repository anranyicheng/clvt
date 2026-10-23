;;;; complex-gap-test.lisp — 复数入口/构造/幂/归一化缺口回归网（§16.28）
;;;;
;;;; 覆盖本轮修掉的 6 个缺口（CONVENTIONS §16.28）：
;;;;   1. vt-from-array   复数 CL 数组 → 旧实现推断成 :float64，虚部静默丢失
;;;;   2. vt-from-sequence 复数嵌套序列 → 同上（缺省 :dtype 写死 :float64）
;;;;   3. vt-full        复数填充值 → 同上
;;;;   4. vt-float-power 复数 → 旧实现 coerce 到 double-float 后算实数幂
;;;;   5. vt-cbrt        复数 → numpy 拒绝（ufunc not supported）；旧实现静默取实部
;;;;   6. vt-layer-norm  复数 → 旧实现在 (:dtype :float64) 处撞 REAL 断言
;;;;
;;;; 断言分三类：
;;;;   · dtype 契约（入口推断、复数保持、显式 :dtype 优先级）
;;;;   · 数值契约（与 numpy 实测值/闭式解对照；实数路径回归）
;;;;   · 报错契约（cbrt 的确定性拒绝、显式非复数 :dtype 的 safe-casting 拒绝）
;;;;
;;;; 用法：sbcl --non-interactive --load test/complex-gap-test.lisp

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

(defun flat (v)
  "按逻辑顺序展平成 list（尊重 offset）。"
  (let* ((c (vt-contiguous (vt-ravel (if (typep v 'vt) v (ensure-vt v)))))
         (d (vt-data c)) (o (the fixnum (vt-offset c))) (n (the fixnum (vt-size c))))
    (loop for i from o below (+ o n) collect (aref d i))))

(defun nan-p* (x)
  "NaN 判定。⚠ 不能用 (= x x) —— 测试进程里 FP 陷阱是**开着**的，
   NaN 比较会直接抛 FLOATING-POINT-INVALID-OPERATION。
   用 SBCL 的位级判定 sb-ext:float-nan-p（不算术，不触发陷阱）。"
  (and (floatp x) (sb-ext:float-nan-p x)))

(defun approx (a b &optional (tol 1d-12))
  (cond
    ;; NaN 对 NaN 视为相等（用于「应为 NaN」的断言）
    ((and (complexp a) (complexp b))
     (and (if (and (nan-p* (realpart a)) (nan-p* (realpart b)))
              t
              (<= (abs (- (realpart a) (realpart b))) tol))
          (if (and (nan-p* (imagpart a)) (nan-p* (imagpart b)))
              t
              (<= (abs (- (imagpart a) (imagpart b))) tol))))
    ((complexp a) (approx a (complex b 0) tol))
    ((complexp b) (approx (complex a 0) b tol))
    ((or (nan-p* a) (nan-p* b)) (and (nan-p* a) (nan-p* b)))
    (t (<= (abs (- a b)) tol))))

(defun all-approx (v expected &optional (tol 1d-12))
  "EXPECTED 是标量（对所有元素）或 list。"
  (let* ((got (flat v))
         (exp (if (listp expected) expected (make-list (length got) :initial-element expected))))
    (and (= (length got) (length exp))
         (every (lambda (g e) (approx g e tol)) got exp))))

;;; ------------------------------------------------------------------
;;; 1. vt-from-array / vt-from-sequence / vt-full 的 dtype 推断
;;; ------------------------------------------------------------------

(defun run-entry-dtype ()
  (format t "~&--- 1. 入口 dtype 推断（from-array / from-sequence / full）---~%")

  (let ((a (make-array 3 :element-type '(complex double-float)
                         :initial-contents (list #C(1.0d0 2.0d0) #C(-3.0d0 4.0d0) #C(0.5d0 -0.5d0)))))
    (report "from-array 复数数组 → complex128"
            (eq (vt-dtype (vt-from-array a)) :complex128)
            (format nil "实得 ~a" (vt-dtype (vt-from-array a))))
    (report "from-array 复数数组保住虚部"
            (all-approx (vt-from-array a) (list #C(1.0d0 2.0d0) #C(-3.0d0 4.0d0) #C(0.5d0 -0.5d0))))
    (report "from-array :fast t 也保住虚部"
            (all-approx (vt-from-array a :fast t)
                        (list #C(1.0d0 2.0d0) #C(-3.0d0 4.0d0) #C(0.5d0 -0.5d0)))))

  (let ((a (make-array 2 :element-type '(complex single-float)
                         :initial-contents (list #C(1.5s0 2.5s0) #C(-1.5s0 0.5s0)))))
    (report "from-array complex-single 数组 → complex64"
            (eq (vt-dtype (vt-from-array a)) :complex64)))

  (let ((a (make-array '(2 2) :element-type '(complex double-float)
                         :initial-contents (list (list #C(1.0d0 1.0d0) #C(2.0d0 2.0d0))
                                                 (list #C(3.0d0 3.0d0) #C(4.0d0 4.0d0))))))
    (let ((v (vt-from-array a)))
      (report "from-array 2D 复数数组形状" (equal (vt-shape v) '(2 2)))
      (report "from-array 2D 复数数组值"
              (all-approx v (list #C(1.0d0 1.0d0) #C(2.0d0 2.0d0)
                                  #C(3.0d0 3.0d0) #C(4.0d0 4.0d0))))))

  ;; 显式 :dtype 优先（契约 §16.5：安全转换取实部），不受推断影响
  (let* ((a (make-array 2 :element-type '(complex double-float)
                          :initial-contents (list #C(1.0d0 9.0d0) #C(2.0d0 -9.0d0))))
         (v (vt-from-array a :dtype :float64)))
    (report "from-array 显式 :dtype :float64 → 取实部（契约不变）"
            (and (eq (vt-dtype v) :float64) (all-approx v (list 1.0d0 2.0d0)))))

  ;; from-sequence
  (let ((v (vt-from-sequence (list #C(1.0d0 2.0d0) #C(3.0d0 4.0d0)))))
    (report "from-sequence 复数 list → complex128"
            (eq (vt-dtype v) :complex128)
            (format nil "实得 ~a" (vt-dtype v)))
    (report "from-sequence 复数 list 值"
            (all-approx v (list #C(1.0d0 2.0d0) #C(3.0d0 4.0d0)))))
  (let ((v (vt-from-sequence (list (list #C(1.0d0 0.0d0) #C(0.0d0 1.0d0))
                                   (list #C(2.0d0 0.0d0) #C(0.0d0 -2.0d0))))))
    (report "from-sequence 嵌套复数 list 形状" (equal (vt-shape v) '(2 2)))
    (report "from-sequence 嵌套复数 list dtype" (eq (vt-dtype v) :complex128)))
  (report "from-sequence 纯实数 list 仍是 float64（不回归）"
          (eq (vt-dtype (vt-from-sequence (list 1.0d0 2.0d0))) :float64))
  (report "from-sequence 整数 list 仍是 float64（不回归）"
          (eq (vt-dtype (vt-from-sequence (list 1 2 3))) :float64))
  (report "from-sequence 空序列仍是 (0) float64（不回归）"
          (and (equal (vt-shape (vt-from-sequence nil)) '(0))
               (eq (vt-dtype (vt-from-sequence nil)) :float64)))
  (report "from-sequence 显式 :dtype :complex64"
          (eq (vt-dtype (vt-from-sequence (list #C(1.0d0 2.0d0)) :dtype :complex64)) :complex64))

  ;; vt-full
  (report "vt-full 复数填充 → complex128"
          (eq (vt-dtype (vt-full '(2 2) #C(3.0d0 1.0d0))) :complex128))
  (report "vt-full 复数填充保住虚部"
          (all-approx (vt-full '(2) #C(3.0d0 1.0d0)) #C(3.0d0 1.0d0)))
  (report "vt-full 实数填充仍是 float64（不回归）"
          (eq (vt-dtype (vt-full '(2) 1.5d0)) :float64))
  (report "vt-full 整数填充仍是 float64（不回归，未变成 int64）"
          (eq (vt-dtype (vt-full '(2) 3)) :float64))
  (report "vt-full 显式 :dtype :complex64"
          (eq (vt-dtype (vt-full '(2) #C(1.0d0 1.0d0) :dtype :complex64)) :complex64))
  (report "vt-full-like 继承复数输入 dtype"
          (eq (vt-dtype (vt-full-like (vt-full '(2) #C(1.0d0 1.0d0)) #C(2.0d0 2.0d0)))
              :complex128)))

;;; ------------------------------------------------------------------
;;; 2. vt-float-power 的复数分支
;;; ------------------------------------------------------------------

(defun run-float-power ()
  (format t "~&--- 2. vt-float-power ---~%")
  ;; numpy: np.float_power(2+1j, 2+0j) = 3+4j
  (report "float_power (2+1j)^(2+0j) = 3+4j（numpy 实测）"
          (all-approx (vt-float-power (list #C(2.0d0 1.0d0)) (list #C(2.0d0 0.0d0)))
                      #C(3.0d0 4.0d0)))
  ;; numpy: np.float_power(-1+0j, 0.5+0j) = 6.123234e-17+1j
  (report "float_power (-1+0j)^0.5 走主值分支（numpy 实测 6.12e-17+1j）"
          (all-approx (vt-float-power (list #C(-1.0d0 0.0d0)) (list #C(0.5d0 0.0d0)))
                      (complex 6.123233995736766d-17 1.0d0) 1d-15))
  ;; numpy: nan^0 = 1、0^0 = 1
  (report "float_power (nan+0j)^(0+0j) = 1（numpy 边界）"
          (all-approx (vt-float-power (list (complex (vt-get-nan :float64) 0.0d0))
                                      (list #C(0.0d0 0.0d0)))
                      1.0d0))
  (report "float_power (0+0j)^(0+0j) = 1（numpy 边界）"
          (all-approx (vt-float-power (list #C(0.0d0 0.0d0)) (list #C(0.0d0 0.0d0))) 1.0d0))
  (report "float_power (0+0j)^(-1+0j) = NaN（numpy 边界）"
          (let ((z (first (flat (vt-float-power (list #C(0.0d0 0.0d0)) (list #C(-1.0d0 0.0d0)))))))
            (and (nan-p* (realpart z)) (nan-p* (imagpart z)))))
  (report "float_power 复数结果 dtype = 提升后的复数 dtype"
          (eq (vt-dtype (vt-float-power (vt-full '(2) #C(1.0d0 1.0d0) :dtype :complex64)
                                        (vt-full '(2) #C(2.0d0 0.0d0))))
              :complex128))
  (report "float_power 显式非复数 :dtype 被拒绝（safe-casting）"
          (handler-case (progn (vt-float-power (vt-full '(2) #C(1.0d0 1.0d0)) 2.0d0
                                               :dtype :float64)
                               nil)
            (error () t)))
  (report "float_power 实数路径不回归（2^3=8, float64）"
          (and (eq (vt-dtype (vt-float-power (vt-full '(2) 2.0d0) (vt-full '(2) 3.0d0))) :float64)
               (all-approx (vt-float-power (vt-full '(2) 2.0d0) (vt-full '(2) 3.0d0)) 8.0d0)))
  (report "float_power 负底数 + 分数指数 → NaN（实数路径契约）"
          (nan-p* (first (flat (vt-float-power (vt-full '(1) -2.0d0) (vt-full '(1) 0.5d0))))))
  (report "float_power :out 契约"
          (let* ((out (vt-full '(2) #C(0.0d0 0.0d0) :dtype :complex128))
                 (r (vt-float-power (vt-full '(2) #C(2.0d0 1.0d0))
                                    (vt-full '(2) #C(2.0d0 0.0d0)) :out out)))
            (and (eq r out) (all-approx r #C(3.0d0 4.0d0))))))

;;; ------------------------------------------------------------------
;;; 3. vt-cbrt：与 numpy 一致的确定性拒绝
;;; ------------------------------------------------------------------

(defun run-cbrt ()
  (format t "~&--- 3. vt-cbrt（numpy 拒绝复数）---~%")
  (report "vt-cbrt 复数入参 → 确定性报错（不再是静默取实部）"
          (handler-case (progn (vt-cbrt (list #C(8.0d0 0.0d0))) nil)
            (error (e) (search "不支持复数" (format nil "~a" e)))))
  (report "vt-cbrt 复数标量也拒绝"
          (handler-case (progn (vt-cbrt #C(8.0d0 1.0d0)) nil)
            (error () t)))
  (report "vt-cbrt 实数路径不回归（cbrt(8)=2, cbrt(-8)=-2）"
          (and (all-approx (vt-cbrt (vt-full '(1) 8.0d0)) 2.0d0)
               (all-approx (vt-cbrt (vt-full '(1) -8.0d0)) -2.0d0))))

;;; ------------------------------------------------------------------
;;; 4. vt-layer-norm 的复数分支
;;; ------------------------------------------------------------------

(defun run-layer-norm ()
  (format t "~&--- 4. vt-layer-norm ---~%")
  (let* ((a (vt-from-sequence
             (list (list #C(1.0d0 1.0d0) #C(2.0d0 -1.0d0) #C(3.0d0 0.5d0) #C(4.0d0 -0.5d0))
                   (list #C(0.0d0 2.0d0) #C(-1.0d0 0.0d0) #C(1.0d0 -2.0d0) #C(2.0d0 1.0d0)))))
         (r (vt-layer-norm a '(4))))
    (report "layer-norm 复数 → complex128，形状不变"
            (and (eq (vt-dtype r) :complex128) (equal (vt-shape r) '(2 4)))
            (format nil "~a ~a" (vt-dtype r) (vt-shape r)))
    (report "layer-norm 归一化后每个 fiber 均值为 0（复数）"
            (let ((m (vt-mean r :axis '(1))))
              (every (lambda (z) (< (abs z) 1d-10)) (flat m))))
    ;; 注意 eps=1e-5：归一化后 var = var/(var+eps) ≈ 1 - 1e-5，故容差取 1e-4
    (report "layer-norm 归一化后每个 fiber var 为 1（实数，容差含 eps 项）"
            (let ((v (vt-var r :axis '(1))))
              (every (lambda (x) (< (abs (- x 1.0d0)) 1d-4)) (flat v))))
    (report "layer-norm 虚部为 0 的复数输入 ≈ 实数路径"
            (let* ((rr (vt-real (vt-from-sequence
                                 (list (list #C(1.0d0 0.0d0) #C(2.0d0 0.0d0)
                                             #C(3.0d0 0.0d0) #C(4.0d0 0.0d0))
                                       (list #C(0.0d0 0.0d0) #C(-1.0d0 0.0d0)
                                             #C(1.0d0 0.0d0) #C(2.0d0 0.0d0))))))
                   (c (vt-layer-norm (vt-from-sequence
                                      (list (list #C(1.0d0 0.0d0) #C(2.0d0 0.0d0)
                                                  #C(3.0d0 0.0d0) #C(4.0d0 0.0d0))
                                            (list #C(0.0d0 0.0d0) #C(-1.0d0 0.0d0)
                                                  #C(1.0d0 0.0d0) #C(2.0d0 0.0d0))))
                                     '(4)))
                   (r-ref (vt-layer-norm rr '(4))))
              (all-approx c (flat r-ref) 1d-12)))
    (report "layer-norm 复数 gamma/beta"
            (let* ((g (vt-full '(4) #C(2.0d0 1.0d0) :dtype :complex128))
                   (b (vt-full '(4) #C(0.5d0 -0.5d0) :dtype :complex128))
                   (rg (vt-layer-norm a '(4) :gamma g :beta b)))
              (and (eq (vt-dtype rg) :complex128)
                   ;; 均值为 beta（因为 normed 均值 0）
                   (every (lambda (z) (< (abs z) 1d-10))
                          (flat (vt-- (vt-mean rg :axis '(1)) (vt-mean b :axis nil)))))))
    (report "layer-norm 复数 :out 契约"
            (let ((out (vt-full '(2 4) #C(0.0d0 0.0d0) :dtype :complex128)))
              (eq (vt-layer-norm a '(4) :out out) out)))))

;;; ------------------------------------------------------------------
;;; 5. 回归网：已支持的复数函数抽查（确认本轮改动没把它们弄坏）
;;; ------------------------------------------------------------------

(defun run-regression ()
  (format t "~&--- 5. 已支持函数抽查 ---~%")
  (let ((z (vt-from-sequence (list #C(1.0d0 2.0d0) #C(3.0d0 -1.0d0) #C(0.0d0 0.5d0)))))
    (report "vt-conj 复数" (eq (vt-dtype (vt-conj z)) :complex128))
    (report "vt-abs 复数 → float64" (eq (vt-dtype (vt-abs z)) :float64))
    (report "vt-pow 复数（与 float_power 同一内核）"
            (all-approx (vt-pow (ensure-vt (list #C(2.0d0 1.0d0))) 2) #C(3.0d0 4.0d0)))
    (report "vt-sqrt 复数" (all-approx (vt-sqrt (ensure-vt (list #C(-1.0d0 0.0d0)))) #C(0.0d0 1.0d0)))
    (report "vt-sum 复数 → 复数标量"
            (all-approx (vt-sum z) #C(4.0d0 1.5d0)))
    (report "vt-norm 复数 → 实数" (eq (vt-dtype (vt-norm z)) :float64))
    (report "vt-matmul 复数 dtype" (eq (vt-dtype (vt-matmul (vt-identity 3 :dtype :complex128)
                                                            (vt-identity 3 :dtype :complex128)))
                                       :complex128))
    (report "vt-take-along-axis 复数仍可用"
            (eq (vt-dtype (vt-take-along-axis
                           (vt-reshape (vt-full '(12) #C(1.0d0 1.0d0)) '(3 4))
                           (vt-from-array #2A((0 2)(1 3)(0 0))) 1))
                :complex128))
    (report "vt-compress 复数仍可用"
            (eq (vt-dtype (vt-compress '(1 0 1) z)) :complex128))
    (report "vt-kron 复数仍可用"
            (eq (vt-dtype (vt-kron (vt-reshape (vt-full '(4) #C(1.0d0 1.0d0)) '(2 2))
                                   (vt-reshape (vt-full '(4) #C(1.0d0 1.0d0)) '(2 2))))
                :complex128))))

;;; ------------------------------------------------------------------
;;; 入口
;;; ------------------------------------------------------------------

(format t "~&=====================================")
(format t "~&复数缺口回归（§16.28）")
(format t "~&=====================================~%")
(run-entry-dtype)
(run-float-power)
(run-cbrt)
(run-layer-norm)
(run-regression)

(format t "~&=====================================~%")
(format t "Total: ~d / Pass: ~d / Fail: ~d~%" (+ *pass* *fail*) *pass* *fail*)
(format t "PASS=~d FAIL=~d~%" *pass* *fail*)
(when *failures*
  (format t "~&--- 失败清单 ---~%")
  (dolist (f (reverse *failures*)) (format t "  · ~a~%" f)))
(when (plusp *fail*) (sb-ext:exit :code 1))
