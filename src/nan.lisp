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
    "判定浮点数 X 是否为正无穷（非浮点数返回 NIL）。"
    (and (floatp x)
         (with-float-safe (= x (/ (coerce 1.0 (type-of x))
                                  (coerce 0.0 (type-of x)))))))

  (defun vt-float-neg-inf-p (x)
    "判定浮点数 X 是否为负无穷（非浮点数返回 NIL）。"
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

;;; ------------------------------------------------------------------
;;; 按 dtype 取常量的统一入口
;;; ------------------------------------------------------------------

(declaim (inline vt-get-nan vt-get-pos-inf vt-get-neg-inf))

(defun vt-get-nan (dtype)
  "返回指定浮点 dtype 的 NaN 常量。:float32 → single-float NaN，其余（含 :float64）→ double-float NaN。"
  (if (eq dtype :float32) +vt-sfloat-nan+ +vt-dfloat-nan+))

(defun vt-get-pos-inf (dtype)
  "返回指定浮点 dtype 的正无穷常量（:float32 → single-float，其余 → double-float）。"
  (if (eq dtype :float32) +vt-sfloat-pos-inf+ +vt-dfloat-pos-inf+))

(defun vt-get-neg-inf (dtype)
  "返回指定浮点 dtype 的负无穷常量（:float32 → single-float，其余 → double-float）。"
  (if (eq dtype :float32) +vt-sfloat-neg-inf+ +vt-dfloat-neg-inf+))

(defun vt-float-nan (&optional (dtype :float64))
  "按 dtype 取 NaN 的公开入口（DTYPE 缺省 :float64）。等价 (vt-get-nan dtype)。"
  (vt-get-nan dtype))

(defun vt-float-pos-inf (&optional (dtype :float64))
  "按 dtype 取正无穷的公开入口（DTYPE 缺省 :float64）。等价 (vt-get-pos-inf dtype)。"
  (vt-get-pos-inf dtype))

(defun vt-float-neg-inf (&optional (dtype :float64))
  "按 dtype 取负无穷的公开入口（DTYPE 缺省 :float64）。等价 (vt-get-neg-inf dtype)。"
  (vt-get-neg-inf dtype))

;;; ------------------------------------------------------------------
;;; 快速 NaN / Inf 判定（内联，调用方需确保浮点陷阱已屏蔽）
;;; 用于 vt-map / vt-reduce / vt-numpy-sort / vt-unique 等已 with-float-safe 的热路径。
;;; ------------------------------------------------------------------

(declaim (inline %inf-p %pos-inf-p %neg-inf-p %nan-or-inf-p))
;; %NAN-P 不列入：它在更早加载的文件里已有调用点（in-package 之前的工具层），
;; 再 DECLAIM INLINE 只会得到「2 calls were previously compiled」告警，
;; 且对那两处无效；后续调用点本来就已编译在 with-float-safe 的快判里。

;;; 类型声明（性能）：四个谓词分别只在 double-float / single-float 上被调用，
;;; 返回恒为 boolean。声明后调用点不再需要把结果当 T 装箱，也可被内联进
;;; 比较/裁剪/极值热循环的布尔分支中。
(declaim (ftype (function (t) boolean) %inf-p)
         (ftype (function (t) boolean) %pos-inf-p)
         (ftype (function (t) boolean) %neg-inf-p))

(defun %inf-p (x)
  "快速 Inf 判定：|x| > most-positive-double-float（对 single/double 均成立）。"
  (and (floatp x) (> (abs x) most-positive-double-float)))

(defun %pos-inf-p (x)
  (and (floatp x) (> x most-positive-double-float)))

(defun %neg-inf-p (x)
  (and (floatp x) (< x most-negative-double-float)))

(declaim (inline %safe-truncate))
;;; 类型声明（性能）：输入恒为实数，输出恒为整数。这是所有窄化转换
;;; （%wrap-* → %coerce-* → 写入 specialized-array）的第一跳，缺了它
;;; 整条链路都退化为 GENERIC-TRUNCATE + 装箱。
(declaim (ftype (function (real) integer) %safe-truncate))
(defun %safe-truncate (v)
  "把数值截断为整数；浮点 NaN/±Inf 返回 0。调用方需屏蔽浮点陷阱。"
  (declare (type real v))
  (if (floatp v)
      (if (%nan-or-inf-p v)
	  0
	  (truncate v))
      (truncate v)))
