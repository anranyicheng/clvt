;;;; dtype-weak-scalar-test.lisp — 第十一轮回归：弱标量提升（numpy NEP 50）
;;;;
;;;; 本轮用「complex64 的 dtype 对拍」扫出来的问题：**Lisp 标量被当成强类型**
;;;; 参与提升，于是窄 dtype 的每个「+标量」都静默升宽：
;;;;
;;;;   表达式              修复前      修复后(=numpy)   代价
;;;;   f32 数组 + 1.5      float64     float32          内存 ×2、带宽 ×2
;;;;   f32 数组 + 1        float64     float32
;;;;   i32 数组 + 1        int64       int32
;;;;   i8  数组 + 1        int64       int8
;;;;   u8  数组 + 1        int64       uint8
;;;;   c64 数组 + 1+2j     complex128  complex64
;;;;   c64 数组 + 1.5      complex128  complex64
;;;;   f32 数组 + 1+2j     complex128  complex64
;;;;   f32 数组 / 2        float64     float32
;;;;   c64 ** 2            complex128  complex64
;;;;   maximum(c64, 标量)  complex128  complex64
;;;;
;;;; 根因：逐元素运算先把标量 ensure-vt 成 float64/complex128 再 promote，
;;;; 「谁是标量」这个信息在 ensure 之后已经丢失。numpy 2.x（NEP 50）里
;;;; Python 标量是**弱**类型：只贡献种类、不贡献宽度。
;;;; 依 CONVENTIONS §0「与 NumPy 冲突以 NumPy 为准」修正。
;;;;
;;;; 本文件**只**收录上面这批「修复前确实错」的用例 —— 张量×张量的 promote
;;;; 100 格、c64 的其余 dtype 契约等，既有套件已覆盖，不在此重复。
;;;;
;;;; 用法：sbcl --non-interactive --load test/dtype-weak-scalar-test.lisp

(require :asdf)
(defparameter *clvt-test-dir* (make-pathname :name nil :type nil :defaults *load-truename*))
(pushnew (merge-pathnames "../" *clvt-test-dir*) asdf:*central-registry* :test #'equal)
(handler-bind ((warning #'muffle-warning)) (asdf:load-system :clvt))

(in-package :clvt)

(defvar *pass* 0)
(defvar *fail* 0)
(defvar *failures* '())

(defun report (name ok &optional detail)
  (if ok
      (incf *pass*)
      (progn (incf *fail*) (push name *failures*)
             (format t "~&FAIL ~a ~@[— ~a~]" name detail))))

(defun dt (x) (if (typep x 'vt) (vt-dtype x) :lisp-scalar))

(defmacro expect-dtype (name expr want)
  "断言 EXPR 的结果 dtype = WANT（并回报实际值，便于定位）。"
  `(let ((r ,expr))
     (report ,name (eq (dt r) ,want)
             (format nil "got ~a want ~a" (dt r) ,want))))

(defun vals (v)
  (let* ((c (vt-contiguous (vt-ravel v)))
         (d (vt-data c)) (o (the fixnum (vt-offset c))) (n (the fixnum (vt-size c))))
    (loop for i from o below (+ o n) collect (aref d i))))

(defun approx-p (v expected &optional (tol 1d-6))
  (let ((g (vals v)))
    (and (= (length g) (length expected))
         (every (lambda (a b) (<= (abs (- a b)) tol)) g expected))))

;;; ------------------------------------------------------------------

(format t "~&=====================================")
(format t "~&弱标量提升回归（第十一轮，§16.31）")
(format t "~&=====================================~%")

(let ((f32 (vt-zeros '(3) :dtype :float32))
      (i32 (vt-zeros '(3) :dtype :int32))
      (i8  (vt-zeros '(3) :dtype :int8))
      (u8  (vt-zeros '(3) :dtype :uint8))
      (c64 (vt-zeros '(3) :dtype :complex64)))

  ;; ---- 实数：float 标量 ----
  (expect-dtype "f32 数组 + 1.5d0 保持 float32（修复前 float64）"
                (vt-+ f32 1.5d0) :float32)
  (expect-dtype "f32 数组 + 1 保持 float32（修复前 float64）"
                (vt-+ f32 1) :float32)

  ;; ---- 实数：整数标量 ----
  (expect-dtype "i32 数组 + 1 保持 int32（修复前 int64）"
                (vt-+ i32 1) :int32)
  (expect-dtype "i8 数组 + 1 保持 int8（修复前 int64）"
                (vt-+ i8 1) :int8)
  (expect-dtype "u8 数组 + 1 保持 uint8（修复前 int64）"
                (vt-+ u8 1) :uint8)

  ;; ---- 复数：复数标量 ----
  (expect-dtype "c64 数组 + #C(1 2) 保持 complex64（修复前 complex128）"
                (vt-+ c64 (complex 1.0d0 2.0d0)) :complex64)
  (expect-dtype "f32 数组 + #C(1 2) → complex64（修复前 complex128）"
                (vt-+ f32 (complex 1.0d0 2.0d0)) :complex64)
  (expect-dtype "c64 数组 + 1.5 → complex64（修复前 complex128）"
                (vt-+ c64 1.5d0) :complex64)

  ;; ---- 除法（true_divide 的浮点提升也不能被标量带偏）----
  (expect-dtype "f32 数组 / 2 保持 float32（修复前 float64）"
                (vt-/ f32 2) :float32)

  ;; ---- 幂 / 极值：各自单独算 dtype，曾各自漏掉 ----
  (expect-dtype "c64 ** 2 保持 complex64（修复前 complex128）"
                (vt-pow c64 2) :complex64)
  (expect-dtype "maximum(c64, 标量) 保持 complex64（修复前 complex128）"
                (vt-maximum c64 (complex 0.0d0 0.0d0)) :complex64)

  ;; ---- 值不能因为「不再升宽」而算错（f32 内计算，1.5 精确可表示）----
  (report "f32 + 1.5d0 的值仍正确（在 float32 内计算）"
          (approx-p (vt-+ f32 1.5d0) (list 1.5 1.5 1.5)))
  (report "c64 + #C(1 2) 的值仍正确"
          (approx-p (vt-+ c64 (complex 1.0d0 2.0d0))
                    (list (complex 1.0d0 2.0d0) (complex 1.0d0 2.0d0) (complex 1.0d0 2.0d0))))
  (report "i8 + 1 的值仍正确且不溢出"
          (equal (vals (vt-+ (vt-zeros '(3) :dtype :int8) 1)) '(1 1 1))))

(format t "~&=====================================~%")
(format t "Total: ~d / Pass: ~d / Fail: ~d~%" (+ *pass* *fail*) *pass* *fail*)
(format t "PASS=~d FAIL=~d~%" *pass* *fail*)
(when *failures*
  (format t "~&--- 失败清单 ---~%")
  (dolist (f (reverse *failures*)) (format t "  · ~a~%" f)))
(when (plusp *fail*) (sb-ext:exit :code 1))
