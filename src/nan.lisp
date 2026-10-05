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

;;; 注（v0.3.6）：原 +vt-float-nan+ / +vt-float-pos-inf+ / +vt-float-neg-inf+
;;; 三个"默认 double"的别名常量已**删除**。原因：
;;;   1) 名字里的 float 未指明精度，在混合精度代码里极易误用
;;;      —— 把 double NaN 写进 float32 张量会静默变宽再截断；
;;;   2) 与 numpy 的 dtype 显式化原则冲突（numpy 没有"默认 float 常量"）。
;;; 迁移办法：
;;;   常量              → 函数形式（按 dtype 取正确精度的值）
;;;   +vt-float-nan+    → (vt-get-nan dtype)     或 (vt-float-nan dtype)
;;;   +vt-float-pos-inf+→ (vt-get-pos-inf dtype) 或 (vt-float-pos-inf dtype)
;;;   +vt-float-neg-inf+→ (vt-get-neg-inf dtype) 或 (vt-float-neg-inf dtype)
;;; 需要 double 常量时直接用 +vt-dfloat-nan+ / +vt-dfloat-pos-inf+ /
;;; +vt-dfloat-neg-inf+（名字已含精度，不会误用）。

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

;;; 公开的 getter 函数（对标 README 中 vt-float-nan / vt-float-pos-inf / vt-float-neg-inf）
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

(declaim (inline %nan-p %inf-p %pos-inf-p %neg-inf-p))

(defun %nan-p (x)
  "快速 NaN 判定：IEEE 754 中 NaN != NaN。调用方需屏蔽浮点陷阱。"
  (and (floatp x) (not (= x x))))

(defun %inf-p (x)
  "快速 Inf 判定：|x| > most-positive-double-float（对 single/double 均成立）。"
  (and (floatp x) (> (abs x) most-positive-double-float)))

(defun %pos-inf-p (x)
  (and (floatp x) (> x most-positive-double-float)))

(defun %neg-inf-p (x)
  (and (floatp x) (< x most-negative-double-float)))

(defun %nan-or-inf-p (x)
  "浮点 NaN 或 ±Inf 判定。调用方需屏蔽浮点陷阱。"
  (and (floatp x)
       (or (not (= x x))
	   (> (abs x) most-positive-double-float))))

(declaim (inline %safe-truncate))
(defun %safe-truncate (v)
  "把数值截断为整数；浮点 NaN/±Inf 返回 0。调用方需屏蔽浮点陷阱。"
  (if (floatp v)
      (if (%nan-or-inf-p v)
	  0
	  (truncate v))
      (truncate v)))
