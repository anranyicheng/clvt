;;;; dtype-weak-scalar-test.lisp — 弱标量提升回归（numpy NEP 50，§16.31）
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

(defmacro expect-sym (name e1 e2)
  "断言两个表达式的 dtype 相等。用于「同一算子交换操作数顺序」。

   二元算子的提升与顺序无关（`a ⊔ b == b ⊔ a`），只交换顺序就换 dtype
   说明标量身份在传参途中丢了 —— 这个判据不依赖对照 numpy。"
  `(let ((a ,e1) (b ,e2))
     (report ,name (eq (dt a) (dt b))
             (format nil "张量在前=~a 标量在前=~a" (dt a) (dt b)))))

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
(format t "~&弱标量提升回归（§16.31）")
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

;;; ==================================================================
;;; §16.31.7：自己先算 dtype 再把已 ensure 的值往下传的那批
;;; ==================================================================
;;; §16.31.4 只把弱标量接进了 vt-map / vt-fast-map / vt-/ / vt-maximum。
;;; 下面这些函数各自算 dtype、再交给下游，于是「谁是标量」在传参途中丢失。
;;; 每条都先断言「交换顺序不改 dtype」（expect-sym），再断言绝对值
;;; （expect-dtype）—— 只有对称性只能说明「一致地错」，必须两条一起。

(format t "~&-- §16.31.7：交换顺序不许换 dtype --~%")

(let* ((f32 (vt-astype (vt-from-sequence '(1.0 2.0 3.0)) :float32))
       (i8  (vt-astype (vt-from-sequence '(1 2 3)) :int8))
       (i32 (vt-astype (vt-from-sequence '(1 2 3)) :int32))
       (c64 (vt-astype (vt-from-sequence (list #C(1 2) #C(3 4))) :complex64)))

  ;; ---- 对称性：vt-- ----
  (expect-sym "vt-- f32：交换操作数顺序不改 dtype"
              (vt-- f32 1.0d0) (vt-- 1.0d0 f32))
  (expect-sym "vt-- i8：交换操作数顺序不改 dtype"
              (vt-- i8 1) (vt-- 1 i8))
  (expect-sym "vt-- i32：交换操作数顺序不改 dtype"
              (vt-- i32 1) (vt-- 1 i32))
  (expect-sym "vt-- c64：交换操作数顺序不改 dtype"
              (vt-- c64 (complex 1.0d0 2.0d0)) (vt-- (complex 1.0d0 2.0d0) c64))

  ;; ---- 对称性：vt-pow ----
  (expect-sym "vt-pow f32（张量指数）：交换顺序不改 dtype"
              (vt-pow f32 i32) (vt-pow i32 f32))
  (expect-sym "vt-pow i8：整数标量底数不改宽度"
              (vt-pow i8 2) (vt-pow 2 i8))

  ;; ---- 对称性：一元数学的二元形态 ----
  (expect-sym "vt-atan2 f32：交换顺序不改 dtype"
              (vt-atan2 f32 1.5d0) (vt-atan2 1.5d0 f32))
  (expect-sym "vt-hypot f32：交换顺序不改 dtype"
              (vt-hypot f32 1.5d0) (vt-hypot 1.5d0 f32))

  ;; ---- 对称性：整数族 ----
  (expect-sym "vt-gcd i8：整数标量不改宽度"
              (vt-gcd i8 2) (vt-gcd 2 i8))
  (expect-sym "vt-lcm i8：整数标量不改宽度"
              (vt-lcm i8 2) (vt-lcm 2 i8))

  ;; ---- 对称性：「以 double 计算」那一族 ----
  (expect-sym "vt-logaddexp f32：交换顺序不改 dtype"
              (vt-logaddexp f32 1.5d0) (vt-logaddexp 1.5d0 f32))
  (expect-sym "vt-copysign f32：交换顺序不改 dtype"
              (vt-copysign f32 1.5d0) (vt-copysign 1.5d0 f32))
  (expect-sym "vt-nextafter f32：交换顺序不改 dtype"
              (vt-nextafter f32 1.5d0) (vt-nextafter 1.5d0 f32))
  (expect-sym "vt-float-power f32：交换顺序不改 dtype"
              (vt-float-power f32 2) (vt-float-power 2 f32))

  ;; ---- 三元选择：两个分支各自可能是标量 ----
  (let ((mask (vt-from-sequence '(1 0 1) :dtype :bool)))
    (expect-sym "vt-where (cond, f32, 1.5d0)：交换分支不改 dtype"
                (vt-where mask f32 1.5d0) (vt-where mask 1.5d0 f32))
    (expect-dtype "vt-where (cond, f32, 1.5d0) → float32（修复前 float64）"
                  (vt-where mask f32 1.5d0) :float32)
    (report "vt-where (cond, f32, 1.5d0) 的值正确"
            (approx-p (vt-where mask f32 1.5d0) '(1.0d0 1.5d0 3.0d0))))

  ;; ---- 绝对值（对称性只能排除「换个顺序就变」，还要钉住「变对了」）----
  (expect-dtype "vt-- 1.0d0 - f32 数组 → float32（修复前 float64）"
                (vt-- 1.0d0 f32) :float32)
  (expect-dtype "vt-- 1 - i8 数组 → int8（修复前 int64）"
                (vt-- 1 i8) :int8)
  (expect-dtype "vt-- #C(1 2) - c64 数组 → complex64（修复前 complex128）"
                (vt-- (complex 1.0d0 2.0d0) c64) :complex64)
  (expect-dtype "vt-pow 2.0d0 ** f32 数组 → float32（修复前 float64）"
                (vt-pow 2.0d0 f32) :float32)
  (expect-dtype "vt-pow 2 ** i8 数组 → int8（修复前 int64）"
                (vt-pow 2 i8) :int8)
  (expect-dtype "vt-atan2 f32 遇 double 标量 → float32（修复前 float64）"
                (vt-atan2 f32 1.5d0) :float32)
  (expect-dtype "vt-hypot f32 遇 double 标量 → float32（修复前 float64）"
                (vt-hypot f32 1.5d0) :float32)
  (expect-dtype "vt-gcd 2 与 i8 数组 → int8（修复前 int64）"
                (vt-gcd 2 i8) :int8)
  (expect-dtype "vt-lcm 2 与 i8 数组 → int8（修复前 int64）"
                (vt-lcm 2 i8) :int8)
  (expect-dtype "vt-logaddexp 两侧全 f32 → float32（修复前 float64）"
                (vt-logaddexp f32 f32) :float32)
  (expect-dtype "vt-copysign 两侧全 f32 → float32（修复前 float64）"
                (vt-copysign f32 f32) :float32)
  (expect-dtype "vt-nextafter 两侧全 f32 → float32（修复前 float64）"
                (vt-nextafter f32 f32) :float32)
  (expect-dtype "vt-float-power 两侧全 f32 → float32（修复前 float64）"
                (vt-float-power f32 f32) :float32)

  ;; ---- 值：改的必须是 dtype，不能是数值 ----
  ;; 参照物用「显式在 float64 下算完再降 f32」，f32 精度内应一致。
  (report "vt-- 1.0d0 - f32 的值正确"
          (approx-p (vt-- 1.0d0 f32)
                    (mapcar (lambda (x) (- 1.0d0 x)) '(1.0d0 2.0d0 3.0d0))))
  (report "vt-pow 2.0d0 ** f32 的值正确"
          (approx-p (vt-pow 2.0d0 f32)
                    (mapcar (lambda (x) (expt 2.0d0 x)) '(1.0d0 2.0d0 3.0d0))))
  (report "vt-atan2 f32 的值正确（0.785398 ≈ π/4）"
          (approx-p (vt-atan2 f32 f32)
                    (list (atan 1.0d0 1.0d0) (atan 2.0d0 2.0d0) (atan 3.0d0 3.0d0))
                    1d-6))
  (report "vt-hypot f32 的值正确（sqrt(2)·(1,2,3)）"
          (approx-p (vt-hypot f32 f32)
                    (list (sqrt 2.0d0) (* 2 (sqrt 2.0d0)) (* 3 (sqrt 2.0d0)))
                    1d-6))
  (report "vt-logaddexp f32 的值正确（log(e^x + e^1.5)）"
          (approx-p (vt-logaddexp f32 1.5d0)
                    (mapcar (lambda (x) (log (+ (exp x) (exp 1.5d0))))
                            '(1.0d0 2.0d0 3.0d0))
                    1d-6))
  (report "vt-gcd 2 与 i8 的值正确"
          (equal (vals (vt-gcd (vt-from-sequence '(12 18 20) :dtype :int8) 2))
                 '(2 2 2)))
  (report "vt-lcm 2 与 i8 的值正确"
          (equal (vals (vt-lcm (vt-from-sequence '(4 6 8) :dtype :int8) 2))
                 '(4 6 8)))
  (report "vt-copysign f32 遇 -2.5 的符号位正确（全 -0.0）"
          (every (lambda (v) (minusp (float-sign v))) (vals (vt-copysign f32 -2.5d0)))))

(format t "~&=====================================~%")
(format t "Total: ~d / Pass: ~d / Fail: ~d~%" (+ *pass* *fail*) *pass* *fail*)
(format t "PASS=~d FAIL=~d~%" *pass* *fail*)
(when *failures*
  (format t "~&--- 失败清单 ---~%")
  (dolist (f (reverse *failures*)) (format t "  · ~a~%" f)))
(when (plusp *fail*) (sb-ext:exit :code 1))
