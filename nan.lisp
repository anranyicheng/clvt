(in-package :clvt)

;;;; 关于 nan 和 inf 的精确定义 (区分单双精度)

(eval-when (:compile-toplevel :load-toplevel :execute)

  ;; === 安全生成函数 (仅在加载期执行一次) ===
  (defun vt-make-float-nan (float-type)
    "安全生成指定类型的 NaN."
    (with-float-safe
      (ecase float-type
        (single-float (/ 0.0s0 0.0s0))
        (double-float (/ 0.0d0 0.0d0)))))

  (defun vt-make-float-pos-inf (float-type)
    "安全生成指定类型的正无穷大."
    (with-float-safe
      (ecase float-type
        (single-float (/ 1.0s0 0.0s0))
        (double-float (/ 1.0d0 0.0d0)))))

  (defun vt-make-float-neg-inf (float-type)
    "安全生成指定类型的负无穷大."
    (with-float-safe
      (ecase float-type
        (single-float (/ -1.0s0 0.0s0))
        (double-float (/ -1.0d0 0.0d0)))))

  ;; === 谓词函数 (兼容 SBCL 特性，并提供 ANSI 回退) ===
  (defun vt-float-nan-p (x)
    "判断是否为 NaN (单双精度均可)."
    (and (floatp x)
         #+sbcl (sb-kernel::float-nan-p x)
         #-sbcl (with-float-safe (not (= x x))))) ; IEEE 标准: NaN != NaN

  (defun vt-float-inf-p (x)
    "判断是否为无穷大 (单双精度均可)."
    (and (floatp x)
         #+sbcl (sb-kernel::float-infinity-p x)
         #-sbcl (with-float-safe (or (= x (/ 1.0d0 0.0d0)) 
                                     (= x (/ -1.0d0 0.0d0))))))

  ;; === 比较函数 ===
  (defun vt-float-nan-= (a b)
    "两个 NaN 视为相等."
    (and (vt-float-nan-p a) (vt-float-nan-p b)))

  (defun vt-float-inf-= (a b)
    "两个 Inf 相等."
    (with-float-safe (= a b)))
  
  (defun vt-float-nan-inf-= (a b)
    "统一比较: NaN 与 NaN 相等，Inf 与 Inf 相等，其余数值正常比较."
    (cond
      ((and (vt-float-nan-p a) (vt-float-nan-p b)) t)
      ((or (vt-float-nan-p a) (vt-float-nan-p b)) nil)
      (t (with-float-safe (= a b)))))
  )
;; === 全局常量 (加载期初始化，零运行时开销，零编译警告) ===
(defconstant +vt-dfloat-nan+
  (load-time-value (vt-make-float-nan 'double-float))
  "双精度 NaN.")

(defconstant +vt-sfloat-nan+
  (load-time-value (vt-make-float-nan 'single-float))
  "单精度 NaN.")

(defconstant +vt-dfloat-pos-inf+
  (load-time-value (vt-make-float-pos-inf 'double-float))
  "双精度正无穷.")

(defconstant +vt-sfloat-pos-inf+
  (load-time-value (vt-make-float-pos-inf 'single-float))
  "单精度正无穷.")

(defconstant +vt-dfloat-neg-inf+
  (load-time-value (vt-make-float-neg-inf 'double-float))
  "双精度负无穷.")

(defconstant +vt-sfloat-neg-inf+
  (load-time-value (vt-make-float-neg-inf 'single-float))
  "单精度负无穷.")

;; === 向后兼容的旧常量别名 (如果你的代码其他地方用到了旧名) ===
(defconstant +vt-float-nan+ +vt-dfloat-nan+)
(defconstant +vt-float-pos-inf+ +vt-dfloat-pos-inf+)
(defconstant +vt-float-neg-inf+ +vt-dfloat-neg-inf+)

;; === 上层统一访问接口 (消除反复的 if 判断) ===
(declaim (inline vt-get-nan vt-get-pos-inf vt-get-neg-inf))
(defun vt-get-nan (dtype)
  "根据 dtype 获取对应类型的 NaN 常量."
  (if (eq dtype :float32) +vt-sfloat-nan+ +vt-dfloat-nan+))

(defun vt-get-pos-inf (dtype)
  "根据 dtype 获取对应类型的正无穷常量."
  (if (eq dtype :float32) +vt-sfloat-pos-inf+ +vt-dfloat-pos-inf+))

(defun vt-get-neg-inf (dtype)
  "根据 dtype 获取对应类型的负无穷常量."
  (if (eq dtype :float32) +vt-sfloat-neg-inf+ +vt-dfloat-neg-inf+))
