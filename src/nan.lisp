;;;; nan.lisp — NaN / Inf 的精确定义与判定（可移植，不依赖实现内部符号）

(in-package :clvt)

(eval-when (:compile-toplevel :load-toplevel :execute)

  (defun vt-float-nan-p (x)
    "判断 x 是否为 NaN（单双精度均可）。IEEE 754：NaN != NaN。"
    (and (floatp x) (with-float-safe (not (= x x)))))

  (defun vt-float-inf-p (x)
    "判断 x 是否为无穷大（单双精度均可）。"
    (and (floatp x)
         (with-float-safe
           (let* ((one (coerce 1.0 (type-of x)))
                  (zero (coerce 0.0 (type-of x))))
             (or (= x (/ one zero)) (= x (/ (- one) zero)))))))

  (defun vt-float-pos-inf-p (x)
    (and (floatp x)
         (with-float-safe (= x (/ (coerce 1.0 (type-of x))
                                  (coerce 0.0 (type-of x)))))))

  (defun vt-float-neg-inf-p (x)
    (and (floatp x)
         (with-float-safe (= x (/ (coerce -1.0 (type-of x))
                                  (coerce 0.0 (type-of x)))))))

  (defun vt-float-nan-= (a b)
    "两个 NaN 视为相等。"
    (and (vt-float-nan-p a) (vt-float-nan-p b)))

  (defun vt-float-inf-= (a b)
    "两个同号 Inf 相等；NaN 不等于任何值。"
    (and (not (vt-float-nan-p a)) (not (vt-float-nan-p b))
         (vt-float-inf-p a) (vt-float-inf-p b)
         (with-float-safe (= a b))))

  (defun vt-float-nan-inf-= (a b)
    "统一比较：NaN 与 NaN 相等，Inf 与 Inf 相等，其余数值正常比较。"
    (cond ((and (vt-float-nan-p a) (vt-float-nan-p b)) t)
          ((or (vt-float-nan-p a) (vt-float-nan-p b)) nil)
          (t (with-float-safe (= a b))))))

;;; ------------------------------------------------------------------
;;; 常量（加载期生成，零运行时开销）
;;; ------------------------------------------------------------------

(eval-when (:compile-toplevel :load-toplevel :execute)
  (defun %make-nan (float-type)
    (with-float-safe
      (ecase float-type
        (single-float (locally (declare (notinline /)) (/ 0.0s0 0.0s0)))
        (double-float (locally (declare (notinline /)) (/ 0.0d0 0.0d0))))))

  (defun %make-pos-inf (float-type)
    (with-float-safe
      (ecase float-type
        (single-float (locally (declare (notinline /)) (/ 1.0s0 0.0s0)))
        (double-float (locally (declare (notinline /)) (/ 1.0d0 0.0d0))))))

  (defun %make-neg-inf (float-type)
    (with-float-safe
      (ecase float-type
        (single-float (locally (declare (notinline /)) (/ -1.0s0 0.0s0)))
        (double-float (locally (declare (notinline /)) (/ -1.0d0 0.0d0)))))))

(defconstant +vt-dfloat-nan+ (load-time-value (%make-nan 'double-float)))
(defconstant +vt-sfloat-nan+ (load-time-value (%make-nan 'single-float)))
(defconstant +vt-dfloat-pos-inf+ (load-time-value (%make-pos-inf 'double-float)))
(defconstant +vt-sfloat-pos-inf+ (load-time-value (%make-pos-inf 'single-float)))
(defconstant +vt-dfloat-neg-inf+ (load-time-value (%make-neg-inf 'double-float)))
(defconstant +vt-sfloat-neg-inf+ (load-time-value (%make-neg-inf 'single-float)))

;; 向后兼容的别名
(defconstant +vt-float-nan+ +vt-dfloat-nan+)
(defconstant +vt-float-pos-inf+ +vt-dfloat-pos-inf+)
(defconstant +vt-float-neg-inf+ +vt-dfloat-neg-inf+)

;;; ------------------------------------------------------------------
;;; 按 dtype 取常量的统一入口
;;; ------------------------------------------------------------------

(declaim (inline vt-get-nan vt-get-pos-inf vt-get-neg-inf))

(defun vt-get-nan (dtype)
  (if (eq dtype :float32) +vt-sfloat-nan+ +vt-dfloat-nan+))

(defun vt-get-pos-inf (dtype)
  (if (eq dtype :float32) +vt-sfloat-pos-inf+ +vt-dfloat-pos-inf+))

(defun vt-get-neg-inf (dtype)
  (if (eq dtype :float32) +vt-sfloat-neg-inf+ +vt-dfloat-neg-inf+))

;;; 公开的 getter 函数（对标 README 中 vt-float-nan / vt-float-pos-inf / vt-float-neg-inf）
(defun vt-float-nan (&optional (dtype :float64))
  (vt-get-nan dtype))

(defun vt-float-pos-inf (&optional (dtype :float64))
  (vt-get-pos-inf dtype))

(defun vt-float-neg-inf (&optional (dtype :float64))
  (vt-get-neg-inf dtype))
