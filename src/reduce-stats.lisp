;;;; reduce-stats.lisp — 归约、统计、排序、nan 感知统计、信号类函数

(in-package :clvt)

(defun vt-unravel-index (offset shape strides)
  "将物理偏移还原为逻辑坐标列表。"
  (loop with rem = offset
        for dim in shape
        for stride in strides
        collect (multiple-value-bind (idx r)
                    (floor rem stride)
                  (setf rem r) idx)))

;;; ============================================================
;;; 编译期算子描述
;;; ============================================================

(eval-when (:compile-toplevel :load-toplevel :execute)

  (defparameter +kernel-lts+
    '(double-float single-float (signed-byte 64) (signed-byte 32))
    "类型特化内核支持的 Lisp 元素类型（即 4 种存储级物理 dtype，
     与 *vt-storage-dtypes* 一一对应：执行层只对存储级类型生成特化内核。")

  (defparameter +small-int-lts+
    '((signed-byte 16) (signed-byte 8) (unsigned-byte 8) (unsigned-byte 16))
    "小整型逻辑 dtype 对应的元素类型（非存储级）。

     三层分离约定：逻辑层声明的 8 种 dtype 都必须可用（单一事实来源），
     但执行层只为 4 种存储级类型生成 in-lt × res-lt 特化内核——
     小整型输入统一路由到 %kernel-small-general 通用内核，
     保证正确性（连续性/特化只影响性能）。")

  (defparameter +all-lts+ (append +kernel-lts+ +small-int-lts+)
    "全部 8 种元素类型（= 8 种逻辑 dtype）。用于空归约单位元等
     需要覆盖全部逻辑 dtype 的编译期展开。")

  (defparameter +small-int-ets+
    '((signed-byte 16) (signed-byte 8) (unsigned-byte 8) (unsigned-byte 16))
    "小整型的运行时 array-element-type 形态（= +small-int-lts+，别名以示用途）。")

  (defun %op-arg-p (op)
    "是否为 arg 类归约（输出索引）。"
    (member op '(:argmax :argmin :nanargmax :nanargmin)))

  (defun %op-base (op)
    (ecase op
      ((:max :nanmax :argmax :nanargmax) :max)
      ((:min :nanmin :argmin :nanargmin) :min)
      ((:sum :nansum) :sum)
      ((:prod :nanprod) :prod)
      ((:all) :all)
      ((:any) :any)))

  (defun %op-nan-skip-p (op)
    (member op '(:nanmax :nanmin :nansum :nanprod :nanargmax :nanargmin)))

  (defun %op-nan-prop-p (op)
    (member op '(:max :min :argmax :argmin)))

  (defun %dtype->lt (dtype)
    (case dtype
      (:float64 'double-float)
      (:float32 'single-float)
      (:int64   '(signed-byte 64))
      (:int32   '(signed-byte 32))
      (:int16   '(signed-byte 16))
      (:int8    '(signed-byte 8))
      (:uint8   '(unsigned-byte 8))
      (:uint16  '(unsigned-byte 16))
      (t nil)))

  (defun %et->lt (in-et)
    (cond ((equal in-et 'double-float)     'double-float)
          ((equal in-et 'single-float)     'single-float)
          ((equal in-et '(signed-byte 64)) '(signed-byte 64))
          ((equal in-et '(signed-byte 32)) '(signed-byte 32))
          ((equal in-et '(signed-byte 16)) '(signed-byte 16))
          ((equal in-et '(signed-byte 8))  '(signed-byte 8))
          ((equal in-et '(unsigned-byte 8))  '(unsigned-byte 8))
          ((equal in-et '(unsigned-byte 16)) '(unsigned-byte 16))
          (t nil)))

  (defun %lt-rank (lt)
    (cond ((eq lt 'double-float) 5)
          ((eq lt 'single-float) 4)
          ((equal lt '(signed-byte 64)) 3)
          ((equal lt '(signed-byte 32)) 2)
          ((equal lt '(signed-byte 16)) 1)
          ((equal lt '(unsigned-byte 16)) 1)
          ((equal lt '(signed-byte 8)) 0)
          ((equal lt '(unsigned-byte 8)) 0)
          (t -1)))

  (defun %small-int-lt-p (lt)
    (member lt '((signed-byte 16) (signed-byte 8)
                 (unsigned-byte 8) (unsigned-byte 16))))

  (defun %op-acc-lt (op lt res-lt)
    "累加器 Lisp 类型：sum/prod 提升到 lt/res-lt 中较宽者（小整型按
     numpy 语义提升到 int64 累加，避免中途回绕）；
     max/min/arg 用 lt；all/any 用 int64。"
    (cond
      ((member (%op-base op) '(:all :any)) '(signed-byte 64))
      ((%op-arg-p op) lt)
      ((and (member (%op-base op) '(:sum :prod))
            (or (equal lt '(signed-byte 32))
                (%small-int-lt-p lt)))
       '(signed-byte 64))
      (t (if (>= (%lt-rank lt) (%lt-rank res-lt)) lt res-lt))))

  (defun %small-int-extreme (dtype which)
    "小整型 dtype 的可表示极值（which=:min 取最小值，:max 取最大值）。
     运行时辅助，供 %kernel-small-general 的 max/min 初始化使用。"
    (ecase dtype
      (:int8   (if (eq which :min) -128  127))
      (:int16  (if (eq which :min) -32768 32767))
      (:uint8  (if (eq which :min) 0     255))
      (:uint16 (if (eq which :min) 0     65535))))

  (defun %cast-form (lt form)
    (cond ((eq lt 'double-float) `(coerce ,form 'double-float))
          ((eq lt 'single-float) `(coerce ,form 'single-float))
          ((equal lt '(signed-byte 64)) `(%coerce-int64 ,form))
          ((equal lt '(signed-byte 32)) `(%coerce-int32 ,form))
          ((equal lt '(signed-byte 16)) `(%coerce-int16 ,form))
          ((equal lt '(signed-byte 8))  `(%coerce-int8 ,form))
          ((equal lt '(unsigned-byte 8))  `(%coerce-uint8 ,form))
          ((equal lt '(unsigned-byte 16)) `(%wrap-uint16 ,form))
          (t form)))

  (defun %op-init (op lt res-lt)
    "归约初始值（编译期常量，类型与 acc-lt 匹配）。"
    (let* ((acc-lt (%op-acc-lt op lt res-lt))
           (skip   (%op-nan-skip-p op)))
      (ecase (%op-base op)
        (:sum  (cond ((eq acc-lt 'double-float) 0.0d0)
                     ((eq acc-lt 'single-float) 0.0s0)
                     (t 0)))
        (:prod (cond ((eq acc-lt 'double-float) 1.0d0)
                     ((eq acc-lt 'single-float) 1.0s0)
                     (t 1)))
        (:all 1)
        (:any 0)
        (:max  (cond ((and skip (eq acc-lt 'double-float)) '(vt-get-nan :float64))
                     ((and skip (eq acc-lt 'single-float)) '(vt-get-nan :float32))
                     ((eq acc-lt 'double-float) '+vt-dfloat-neg-inf+)
                     ((eq acc-lt 'single-float) '+vt-sfloat-neg-inf+)
                     ((equal acc-lt '(signed-byte 64)) -9223372036854775808)
                     ((equal acc-lt '(signed-byte 32)) -2147483648)
                     ((equal acc-lt '(signed-byte 16)) -32768)
                     ((equal acc-lt '(signed-byte 8)) -128)
                     ((equal acc-lt '(unsigned-byte 8)) 0)
                     ((equal acc-lt '(unsigned-byte 16)) 0)
                     (t 0)))
        (:min  (cond ((and skip (eq acc-lt 'double-float)) '(vt-get-nan :float64))
                     ((and skip (eq acc-lt 'single-float)) '(vt-get-nan :float32))
                     ((eq acc-lt 'double-float) '+vt-dfloat-pos-inf+)
                     ((eq acc-lt 'single-float) '+vt-sfloat-pos-inf+)
                     ((equal acc-lt '(signed-byte 64)) 9223372036854775807)
                     ((equal acc-lt '(signed-byte 32)) 2147483647)
                     ((equal acc-lt '(signed-byte 16)) 32767)
                     ((equal acc-lt '(signed-byte 8)) 127)
                     ((equal acc-lt '(unsigned-byte 8)) 255)
                     ((equal acc-lt '(unsigned-byte 16)) 65535)
                     (t 0))))))
  
  (defun %op-out-dtype (op in-dtype)
    (case op
      ((:all :any) :int64)
      ((:argmax :argmin :nanargmax :nanargmin) :int32)
      ((:sum :prod :nansum :nanprod)
       (case in-dtype
         ((:int8 :uint8 :int16 :uint16 :int32) :int64)
         (t in-dtype)))
      (t in-dtype)))

  ) ; end eval-when

;;; ============================================================
;;; 归约步宏
;;; ============================================================

(defmacro %op-step (op acc val lt acc-lt)
  "生成一次 acc ← f(acc, val) 更新。val 类型 lt，acc 类型 acc-lt。"
  (let* ((base (%op-base op))
         (skip (%op-nan-skip-p op))
         (prop (%op-nan-prop-p op))
         (float (subtypep lt 'float))
         (casted (if (equal lt acc-lt) val (%cast-form acc-lt val))))
    (ecase base
      (:sum  (if (and skip float)
                 `(when (not (%nan-p ,val)) (incf ,acc ,casted))
                 `(incf ,acc ,casted)))
      (:prod (if (and skip float)
                 `(when (not (%nan-p ,val)) (setf ,acc (* ,acc ,casted)))
                 `(setf ,acc (* ,acc ,casted))))
      (:all  `(when (zerop ,val) (setf ,acc 0)))
      (:any  `(when (/= ,val 0) (setf ,acc 1)))
      (:max  (cond ((and skip float)
                    `(cond ((%nan-p ,val) nil)
                           ((%nan-p ,acc) (setf ,acc ,casted))
                           ((> ,val ,acc) (setf ,acc ,casted))))
                   ((and prop float)
                    `(cond ((%nan-p ,acc) nil)
                           ((%nan-p ,val) (setf ,acc ,casted))
                           ((> ,val ,acc) (setf ,acc ,casted))))
                   (t `(when (> ,val ,acc) (setf ,acc ,casted)))))
      (:min  (cond ((and skip float)
                    `(cond ((%nan-p ,val) nil)
                           ((%nan-p ,acc) (setf ,acc ,casted))
                           ((< ,val ,acc) (setf ,acc ,casted))))
                   ((and prop float)
                    `(cond ((%nan-p ,acc) nil)
                           ((%nan-p ,val) (setf ,acc ,casted))
                           ((< ,val ,acc) (setf ,acc ,casted))))
                   (t `(when (< ,val ,acc) (setf ,acc ,casted))))))))

(defmacro %op-arg-step (op acc best-i r val lt acc-lt)
  "生成一次 arg 更新：acc ← val，best-i ← r。acc-lt 对 arg 恒等于 lt。"
  (declare (ignore acc-lt))
  (let* ((base (%op-base op))
         (skip (%op-nan-skip-p op))
         (prop (%op-nan-prop-p op))
         (float (subtypep lt 'float)))
    (ecase base
      (:max (cond ((and skip float)
                   `(cond ((%nan-p ,val) nil)
                          ((%nan-p ,acc) (setf ,acc ,val ,best-i ,r))
                          ((> ,val ,acc) (setf ,acc ,val ,best-i ,r))))
                  ((and prop float)
                   `(cond ((%nan-p ,val)
                           (when (not (%nan-p ,acc))
                             (setf ,acc ,val ,best-i ,r)))
                          ((%nan-p ,acc) nil)
                          ((> ,val ,acc) (setf ,acc ,val ,best-i ,r))))
                  (t `(when (> ,val ,acc) (setf ,acc ,val ,best-i ,r)))))
      (:min (cond ((and skip float)
                   `(cond ((%nan-p ,val) nil)
                          ((%nan-p ,acc) (setf ,acc ,val ,best-i ,r))
                          ((< ,val ,acc) (setf ,acc ,val ,best-i ,r))))
                  ((and prop float)
                   `(cond ((%nan-p ,val)
                           (when (not (%nan-p ,acc))
                             (setf ,acc ,val ,best-i ,r)))
                          ((%nan-p ,acc) nil)
                          ((< ,val ,acc) (setf ,acc ,val ,best-i ,r))))
                  (t `(when (< ,val ,acc) (setf ,acc ,val ,best-i ,r))))))))

;;; ============================================================
;;; 三条内核路径宏
;;; ============================================================

(defmacro %kernel-global (op lt res-lt)
  "路径 1：连续 + 全局。"
  (let* ((acc-lt (%op-acc-lt op lt res-lt))
         (init   (%op-init   op lt res-lt))
         (arg-p  (%op-arg-p  op))
         (nan-skip-arg (and arg-p (%op-nan-skip-p op) (subtypep lt 'float))))
    `(let ((d (the (simple-array ,lt (*)) in-data)))
       (let ((acc (the ,acc-lt ,init))
             ,@(when arg-p `((best-i 0)))
             (p   (the fixnum in-off))
             (end (the fixnum (+ in-off in-size))))
         (declare (type fixnum p end)
                  ,@(when arg-p '((type fixnum best-i))))
         (loop while (< p end) do
           (let ((v (the ,lt (aref d p))))
             ,(if arg-p
                  `(%op-arg-step ,op acc best-i (- p in-off) v ,lt ,acc-lt)
                  `(%op-step ,op acc v ,lt ,acc-lt)))
           (incf p))
         ,@(when nan-skip-arg
             `((when (%nan-p acc)
                 (error "~a: All-NaN slice encountered" ',op))))
         (setf (aref (the (simple-array ,res-lt (*)) res-data) res-off)
               ,(%cast-form res-lt (if arg-p 'best-i 'acc)))))))

(defmacro %kernel-single-axis (op lt res-lt)
  "路径 2：连续 + 单轴。"
  (let* ((acc-lt (%op-acc-lt op lt res-lt))
         (init   (%op-init   op lt res-lt))
         (arg-p  (%op-arg-p  op))
         (nan-skip-arg (and arg-p (%op-nan-skip-p op) (subtypep lt 'float))))
    `(let ((d  (the (simple-array ,lt (*)) in-data))
           (od (the (simple-array ,res-lt (*)) res-data))
           (init-v (the ,acc-lt ,init)))
       (declare (type fixnum outer red inner))
       (dotimes (o outer)
         (let ((in-base (the fixnum (+ in-off (* o red inner)))))
           (declare (type fixnum in-base))
           (dotimes (i inner)
             (let ((acc init-v)
                   ,@(when arg-p `((best-i 0))))
               (declare ,@(when arg-p '((type fixnum best-i))))
               (dotimes (r red)
                 (let ((v (the ,lt (aref d (the fixnum (+ in-base (* r inner) i))))))
                   ,(if arg-p
                        `(%op-arg-step ,op acc best-i r v ,lt ,acc-lt)
                        `(%op-step ,op acc v ,lt ,acc-lt))))
               ,@(when nan-skip-arg
                   `((when (%nan-p acc)
                       (error "~a: All-NaN slice encountered" ',op))))
               (setf (aref od (the fixnum (+ res-off (* o inner) i)))
                     ,(%cast-form res-lt (if arg-p 'best-i 'acc))))))))))

(defmacro %kernel-general (op lt res-lt)
  "路径 3：通用回退（支持非连续 :out）。"
  (let* ((acc-lt (%op-acc-lt op lt res-lt))
         (init   (%op-init   op lt res-lt))
         (arg-p  (%op-arg-p  op))
         (nan-skip-arg (and arg-p (%op-nan-skip-p op) (subtypep lt 'float))))
    `(let ((d  (the (simple-array ,lt (*)) in-data))
           (od (the (simple-array ,res-lt (*)) res-data)))
       (declare (type fixnum n-red n-non red-size non-size))
       (let ((non-idx (make-array (max n-non 1) :element-type 'fixnum :initial-element 0))
             (red-idx (make-array (max n-red 1) :element-type 'fixnum :initial-element 0))
             (init-v  (the ,acc-lt ,init))
             (out-ptr res-off))
         (declare (type (simple-array fixnum (*)) non-idx red-idx)
                  (type fixnum out-ptr))
         (dotimes (_ non-size)
           (let ((base in-off))
             (declare (type fixnum base))
             (dotimes (dd n-non)
               (incf base (* (aref non-idx dd) (svref non-strides dd))))
             (fill red-idx 0)
             (let ((acc init-v)
                   ,@(when arg-p `((best-i 0))))
               (declare (type ,acc-lt acc)
                        ,@(when arg-p '((type fixnum best-i))))
               (dotimes (r red-size)
                 (let ((offset 0) (linear 0) (mult 1))
                   (declare (type fixnum offset linear mult))
                   (loop for dd fixnum from (1- n-red) downto 0 do
                     (incf offset (* (aref red-idx dd) (svref red-strides-v dd)))
                     (incf linear (* (aref red-idx dd) mult))
                     (setf mult (* mult (svref red-sizes dd))))
                   (let ((v (the ,lt (aref d (the fixnum (+ base offset))))))
                     ,(if arg-p
                          `(%op-arg-step ,op acc best-i linear v ,lt ,acc-lt)
                          `(%op-step ,op acc v ,lt ,acc-lt))))
                 (when (plusp n-red)
                   (let ((dd (1- n-red)))
                     (declare (type fixnum dd))
                     (loop
                       (incf (aref red-idx dd))
                       (when (< (aref red-idx dd) (svref red-sizes dd)) (return))
                       (setf (aref red-idx dd) 0)
                       (decf dd)
                       (when (< dd 0) (return))))))
               ,@(when nan-skip-arg
                   `((when (%nan-p acc)
                       (error "~a: All-NaN slice encountered" ',op))))
               (setf (aref od out-ptr)
                     ,(%cast-form res-lt (if arg-p 'best-i 'acc))))
             (when (plusp n-non)
               (let ((dd (1- n-non)))
                 (declare (type fixnum dd))
                 (loop
                   (incf (aref non-idx dd))
                   (incf out-ptr (svref out-non-strides dd))
                   (when (< (aref non-idx dd) (svref non-sizes dd)) (return))
                   (setf (aref non-idx dd) 0)
                   (decf out-ptr (* (svref non-sizes dd)
                                    (svref out-non-strides dd)))
                   (decf dd)
                   (when (< dd 0) (return)))))))))))

(defmacro %kernel-small-general (op)
  "路径 3 内的小整型通用内核（不按 in-lt × res-lt 特化，每算子仅展开一份）。

   三层分离约定：类型特化内核只服务 4 种存储级 dtype；int16/int8/uint8/uint16
   属逻辑级 dtype，读取经通用 aref（无 the 特化）、累加器按算子语义
   （sum/prod 提升至 int64，max/min 用输入 dtype 极值初始化），
   写出经 vt-cast 按结果 dtype 转换。正确性由本内核兜底，
   与特化内核遵循同一算子语义；连续性/特化只影响性能。"
  (let* ((arg-p (%op-arg-p op))
         (base  (%op-base op))
         (init-v
           (ecase base
             (:sum  0)
             (:prod 1)
             (:all  1)
             (:any  0)
             (:max  `(%small-int-extreme (vt-dtype tensor) :min))
             (:min  `(%small-int-extreme (vt-dtype tensor) :max)))))
    `(let ((d  in-data)
           (od res-data))
       (declare (type fixnum n-red n-non red-size non-size))
       (let ((non-idx (make-array (max n-non 1) :element-type 'fixnum :initial-element 0))
             (red-idx (make-array (max n-red 1) :element-type 'fixnum :initial-element 0))
             (init-v  ,init-v)
             (out-ptr res-off))
         (declare (type (simple-array fixnum (*)) non-idx red-idx)
                  (type fixnum out-ptr))
         (dotimes (_ non-size)
           (let ((base-idx in-off))
             (declare (type fixnum base-idx))
             (dotimes (dd n-non)
               (incf base-idx (* (aref non-idx dd) (svref non-strides dd))))
             (fill red-idx 0)
             (let ((acc init-v)
                   ,@(when arg-p `((best-i 0))))
               (declare ,@(when arg-p '((type fixnum best-i))))
               (dotimes (r red-size)
                 (let ((offset 0) (linear 0) (mult 1))
                   (declare (type fixnum offset linear mult))
                   (loop for dd fixnum from (1- n-red) downto 0 do
                     (incf offset (* (aref red-idx dd) (svref red-strides-v dd)))
                     (incf linear (* (aref red-idx dd) mult))
                     (setf mult (* mult (svref red-sizes dd))))
                   (let ((v (aref d (the fixnum (+ base-idx offset)))))
                     ,(ecase base
                        (:sum  `(incf acc v))
                        (:prod `(setf acc (* acc v)))
                        (:all  `(when (zerop v) (setf acc 0)))
                        (:any  `(when (/= v 0) (setf acc 1)))
                        (:max  (if arg-p
                                   `(when (> v acc) (setf acc v best-i linear))
                                   `(when (> v acc) (setf acc v))))
                        (:min  (if arg-p
                                   `(when (< v acc) (setf acc v best-i linear))
                                   `(when (< v acc) (setf acc v)))))))
                 ;; red-idx 进位（dotimes(r) 每轮执行）
                 (when (plusp n-red)
                   (let ((dd (1- n-red)))
                     (declare (type fixnum dd))
                     (loop
                       (incf (aref red-idx dd))
                       (when (< (aref red-idx dd) (svref red-sizes dd)) (return))
                       (setf (aref red-idx dd) 0)
                       (decf dd)
                       (when (< dd 0) (return))))))
               ;; 本输出元素写回（在 acc 作用域内）
               (setf (aref od out-ptr)
                     ,(if arg-p
                          `(vt-cast best-i final-out-dtype)
                          `(vt-cast acc final-out-dtype))))
             ;; non-idx 进位
             (when (plusp n-non)
               (let ((dd (1- n-non)))
                 (declare (type fixnum dd))
                 (loop
                   (incf (aref non-idx dd))
                   (incf out-ptr (svref out-non-strides dd))
                   (when (< (aref non-idx dd) (svref non-sizes dd)) (return))
                   (setf (aref non-idx dd) 0)
                   (decf out-ptr (* (svref non-sizes dd)
                                    (svref out-non-strides dd)))
                   (decf dd)
                   (when (< dd 0) (return)))))))))))
;;; ============================================================
;;; 统一生成宏
;;; ============================================================
(defmacro def-vt-reduce (name op)
  "为算子 op 生成函数 vt-<name>。
   执行层选路：存储级 dtype（4 种）→ in-et × res-lt 类型特化内核；
   小整型逻辑 dtype（4 种）→ %kernel-small-general 通用内核（路径 3 结构）。
   路径 1/2/3 的输出对同一语义必须一致；连续性只影响性能。"
  (let ((fn-name (intern (format nil "VT-~a" name)))
        (arg-p   (%op-arg-p op)))
    (flet ((dispatch (kernel-macro &key small-int-p)
             `(cond
                ,@(when small-int-p
                    `(((member in-et +small-int-ets+ :test #'equal)
                       (%kernel-small-general ,op))))
                ,@(loop for lt in +kernel-lts+
                        collect
                        `((equal in-et ',lt)
                          (cond
                            ,@(loop for rlt in +kernel-lts+
                                    collect
                                    `((equal res-lt ',rlt)
                                      (,kernel-macro ,op ,lt ,rlt)))
                            (t (error "unsupported output dtype ~a" res-lt)))))
                (t (error "unsupported input dtype ~a" in-et)))))
      `(defun ,fn-name (tensor &key axis keepdims dtype out)
         ,(format nil "沿 AXIS 归约（~a 算子）；AXIS 为 NIL 时归约全部元素。
  dtype 语义（§4.2 H3「严格相等」）：结果 dtype = 显式 :dtype，否则按输入提升
  （含 init-val 类型影响）；**不由 out 决定**，out 必须精确匹配结果 shape/dtype。
  空归约区返回该算子的单位元（sum→0 / prod→1 / all→t / any→nil …）。
  实现在存储级 dtype 上做类型特化，连续性只影响性能、不影响语义。"
                   (case op (:sum "求和") (:prod "求积") (:max "最大值")
                         (:min "最小值") (:all "逻辑与") (:any "逻辑或")
                         (:nansum "忽略 NaN 求和") (:nanprod "忽略 NaN 求积")
                         (:nanmax "忽略 NaN 最大值") (:nanmin "忽略 NaN 最小值")
                         (:argmax "最大值的下标") (:argmin "最小值的下标")
                         (:nanargmax "忽略 NaN 的最大值下标")
                         (:nanargmin "忽略 NaN 的最小值下标")
                         (t (string-downcase op))))
         (declare (type vt tensor)
                  (type (or null fixnum list) axis)
                  (type (or null vt) out))
         (with-float-safe
           (let* ((in-shape (vt-shape tensor))
                  (rank (length in-shape))
                  (axes (vt-normalize-axes axis rank))
                  (global (null axes))
                  (single-axis-p (and (= (length axes) 1) (not global)))
                  (out-shape
                    (cond ((and global (not keepdims)) nil)
                          (global (make-list rank :initial-element 1))
                          ((not keepdims)
                           (loop for d in in-shape for i fixnum from 0
                                 unless (member i axes) collect d))
                          (t (loop for d in in-shape for i fixnum from 0
                                   collect (if (member i axes) 1 d))))))
             ;; ---- :out 硬契约（统一原语，§4.2）----
             ;; H5（:dtype 与 :out 冲突）→ H1–H4（类型/形状/可写）。
             ;; 与 vt-map / vt-einsum 共用同一实现，杜绝多份检查漂移。
             (vt-check-out-dtype-consistency
              (format nil "vt-~a" ',name) dtype out)
             (let* ((axis-size (if axes
                                   (reduce #'* (mapcar (lambda (a) (nth a in-shape)) axes)
                                           :initial-value 1)
                                   (reduce #'* in-shape :initial-value 1)))
                    (in-data (vt-data tensor))
                    (in-off  (vt-offset tensor))
                    (in-et   (array-element-type in-data))
                    (in-size (vt-size tensor))
                    ;; ---- 结果 dtype（§4.2 H3「严格相等」）----
                    ;; compute-dtype 是「按输入提升后的自然结果 dtype」，**不由 out 决定**；
                    ;; final-out-dtype = 显式 :dtype 优先，否则取 compute-dtype。
                    ;; out 的 dtype 必须精确等于 final-out-dtype，否则 vt-check-out 报错
                    ;; —— 「算到临时缓冲再 cast 写入 out」的解耦路径已被该裁决废弃。
                    (compute-dtype (%op-out-dtype ,op (vt-dtype tensor)))
                    (final-out-dtype (or dtype compute-dtype))
                    (res-lt (%dtype->lt final-out-dtype))
                    ;; out 给出时统一入口校验（含 dtype 严格相等与可写性）；
                    ;; 未给出时新建。
                    (res (if out
                             (vt-check-out out out-shape final-out-dtype
                                           :op-name (format nil "vt-~a" ',name))
                             (make-vt out-shape 0 :dtype final-out-dtype)))
                    (res-data (vt-data res))
                    (res-off  (vt-offset res)))
               (declare (fixnum rank axis-size in-size))
               (unless res-lt
                 (error "vt-~a: unsupported output dtype ~a" ',name final-out-dtype))
               ;; ---- 空归约语义（numpy 对齐，v0.3.6 起）----
               ;; numpy 2.1.3 实测（修正旧 §5 表中 mean/amax 行的误记）：
               ;;   * 输出为空（size=0，如 (3,0) axis=0）→ 所有算子直接返回空结果；
               ;;   * 输出非空且归约区为空：
               ;;     - max/min/nanmax/nanmin → ValueError（zero-size array to
               ;;       reduction operation ... which has no identity），
               ;;       取代 v0.3.5 的 NaN 填充约定；
               ;;     - argmax/argmin/nanargmax/nanargmin → ValueError（attempt
               ;;       to get argmax of an empty sequence）——仅在输出非空时报；
               ;;     - sum/prod/all/any（含 nan 变体）有单位元 → 填充单位元
               ;;       （sum→0、prod→1、all→1、any→0）。
               (when (or (zerop axis-size) (zerop in-size))
                 (cond
                   ;; 1) 输出为空：直接返回空结果
                   ((zerop (vt-shape-to-size out-shape))
                    (return-from ,fn-name res))
                   ;; 2) arg 族：输出非空 → 报错（numpy 语义）
                   ,@(when arg-p
                       `((t (error "vt-~a: attempt to get arg~a of an empty sequence（numpy 语义：输出非空且归约区为空）"
                                   ',name ,(if (member op '(:argmin :nanargmin))
                                               "min" "max")))))
                   ;; 3) max/min 族：输出非空 → 报错（numpy 语义，取代旧 NaN 填充约定）
                   ,@(when (member (%op-base op) '(:max :min))
                       `((t (error "vt-~a: zero-size array to reduction operation ~a which has no identity（numpy 语义：输出非空且归约区为空）"
                                   ',name ,(if (eq (%op-base op) :max)
                                               "maximum" "minimum")))))
                   ;; 4) 单位元族：填充单位元（%op-init 提供编译期常量）
                   (t
                    (let ((lt (%et->lt in-et)))
                      (vt-fill res
                               (cond
                                 ,@(loop for rlt in +kernel-lts+
                                         collect
                                         `((equal res-lt ',rlt)
                                           (cond
                                             ,@(loop for l in +all-lts+
                                                     collect
                                                     `((equal lt ',l)
                                                       ,(%op-init op l rlt)))
                                             (t 0))))
                                 (t 0)))
                      (return-from ,fn-name res)))))
               ;; ---- 主分派 ----
               ;; 小整型（逻辑级 dtype）不进入类型特化内核：路径 1/2 以
               ;; wide-et 守卫排除，统一由路径 3 的 %kernel-small-general 兜底。
               (cond
                 ;; 路径 1：连续 + 全局 + 存储级 dtype
                 ((and global (vt-contiguous-p tensor)
                       (not (member in-et +small-int-ets+ :test #'equal)))
                  ,(dispatch '%kernel-global))
                 ;; 路径 2：连续 + 单轴 + 存储级 dtype
                 ((and single-axis-p
                       (vt-contiguous-p tensor)
                       (vt-contiguous-p res)
                       (not (member in-et +small-int-ets+ :test #'equal)))
                  (let* ((ax (first axes))
                         (outer (reduce #'* in-shape :end ax :initial-value 1))
                         (red   (nth ax in-shape))
                         (inner (reduce #'* in-shape :start (1+ ax) :initial-value 1)))
                    (declare (fixnum ax outer red inner))
                    ,(dispatch '%kernel-single-axis)))

                 ;; 路径 3：通用回退（含小整型 %kernel-small-general）
                 (t
                  (let* ((global-red  (null axes))
                         (eff-red-axes (if global-red
                                           (loop for i below rank collect i)
                                           axes))
                         (eff-non-axes (if global-red
                                           nil
                                           (loop for i below rank
                                                 unless (member i axes) collect i)))
                         (in-strides  (vt-strides tensor))
                         (out-strides (vt-strides res))         ; ★ 新增
                         (in-shape-vec   (coerce in-shape 'simple-vector))
                         (in-strides-vec (coerce in-strides 'simple-vector))
                         (red-axes (coerce eff-red-axes 'simple-vector))
                         (non-axes (coerce eff-non-axes 'simple-vector))
                         (n-red (length eff-red-axes))
                         (n-non (- rank n-red))
                         (red-sizes (map 'vector (lambda (a) (svref in-shape-vec a))
                                         red-axes))
                         (red-strides-v (map 'vector (lambda (a) (svref in-strides-vec a))
                                             red-axes))
                         (non-sizes (map 'vector (lambda (a) (svref in-shape-vec a))
                                         non-axes))
                         (non-strides (map 'vector (lambda (a) (svref in-strides-vec a))
                                           non-axes))
                         (out-non-strides
                           (coerce
                            (unless global-red
                              (if keepdims
                                  (loop for a in eff-non-axes
                                        collect (nth a out-strides))
                                  (loop for a in eff-non-axes
                                        for out-i = (loop for i below a
                                                          count (not (member i axes)))
                                        collect (nth out-i out-strides))))
                            'simple-vector))
                         (red-size (reduce #'* red-sizes :initial-value 1))
                         (non-size (reduce #'* non-sizes :initial-value 1)))
                    (declare (fixnum n-red n-non red-size non-size))
                    ,(dispatch '%kernel-general :small-int-p t))))
               ;; 主分派结束：res 即最终结果（out 存在时已就地写入并返回同一对象）
               res)))))))

;;; ============================================================
;;; 归约族定义
;;; ============================================================

(def-vt-reduce sum  :sum)
(def-vt-reduce prod :prod)
(def-vt-reduce amax :max)
(def-vt-reduce amin :min)
(def-vt-reduce all  :all)
(def-vt-reduce any  :any)

(def-vt-reduce nansum  :nansum)
(def-vt-reduce nanprod :nanprod)
(def-vt-reduce nanmax  :nanmax)
(def-vt-reduce nanmin  :nanmin)

(def-vt-reduce argmax    :argmax)
(def-vt-reduce argmin    :argmin)
(def-vt-reduce nanargmax :nanargmax)
(def-vt-reduce nanargmin :nanargmin)


(defun vt-isclose (t1 t2 &key (rtol 1e-5) (atol 1e-8) out)
  "逐元素判断 |t1 - t2| <= atol + rtol*|t2|（对标 numpy.isclose），返回布尔张量。RTOL/ATOL 缺省 1e-5 / 1e-8。"
  (vt-map (lambda (a b)
            (cond ((or (%nan-p a) (%nan-p b)) 0.0d0)
                  ((or (%inf-p a) (%inf-p b)) (if (= a b) 1.0d0 0.0d0))
                  (t (if (<= (abs (- a b))
                             (+ atol (* rtol (max (abs a) (abs b)))))
                         1.0d0 0.0d0))))
          t1 t2 :dtype :float64 :out out))

(defun vt-allclose (t1 t2 &key (rtol 1e-5) (atol 1e-8))
  "判断两个张量在 RTOL/ATOL 容差下是否全部相等（对标 numpy.allclose），返回布尔。"
  (= (vt-item (vt-all (vt-isclose t1 t2 :rtol rtol :atol atol))) 1.0d0))

(defun vt-isfinite (vt &key out)
  "逐元素判断是否有限（既非 NaN 也非 ±Inf），返回 int8 布尔张量（1/0）。
      dtype 固定 :int8（承载布尔语义，CONVENTIONS §3.1）。"
  (vt-map (lambda (x) (if (and (not (%nan-p x)) (not (%inf-p x))) 1 0))
          vt :dtype :int8 :out out))

(defun vt-isinf (vt &key out)
  "逐元素判断是否为 ±Inf，返回 int8 布尔张量（1/0）。dtype 固定 :int8。"
  (vt-map (lambda (x) (if (%inf-p x) 1 0)) vt :dtype :int8 :out out))

(defun vt-isnan (vt &key out)
  "逐元素判断是否为 NaN，返回 int8 布尔张量（1/0）。dtype 固定 :int8。"
  (vt-map (lambda (x) (if (%nan-p x) 1 0)) vt :dtype :int8 :out out))

;;; ------------------------------------------------------------------
;;; 均值 / 方差 / 标准差
;;; ------------------------------------------------------------------

(defun %get-axes-count (axis rank shape)
  (let* ((axes (vt-normalize-axes axis rank))
         (count (if axes (reduce #'* (mapcar (lambda (a)
                                               (nth a shape))
                                             axes)
                                 :initial-value 1)
                    (reduce #'* shape :initial-value 1))))
    (values axes count)))

(defun vt-average (tensor weights &key axis keepdims dtype out)
  "加权平均：sum(tensor * weights) / sum(weights)。对标 numpy.average。

  dtype 语义（§4.2 H3「严格相等」）：
    结果 dtype 由「输入提升 + 显式 :dtype」决定，**不由 out 决定**。
    out 的 dtype 必须精确等于结果 dtype，否则报错。
    numpy 的 average 对整型输入会提升到 float64（本库一致）。"
  (with-float-safe
    (let ((a-shape (vt-shape tensor))
          (w-shape (vt-shape weights))
          (eff weights))
      (cond (axis (let* ((rank (length a-shape))
                         (ax (vt-normalize-axis axis rank))
                         (ax-size (nth ax a-shape)))
                    (unless (and (= (length w-shape) 1)
                                 (= (first w-shape) ax-size))
                      (error "1D weights expected when axis is specified"))
                    (setf eff (vt-reshape weights (loop for i below rank
                                                        collect (if (= i ax) ax-size 1))))))
            (t (unless (equal w-shape a-shape)
                 (error "weights must have same shape as a when axis is nil"))))
      (let* ((prod (vt-map #'* tensor eff))
             (weighted-sum (vt-sum prod :axis axis :keepdims keepdims))
             (sum-weights (vt-item (vt-sum weights)))
             (in-dtype (vt-dtype weighted-sum))
             (need-promote (member in-dtype '(:int32 :int64 :int16 :int8 :uint8 :uint16)))
             ;; ---- 结果 dtype：输入提升 + 显式 :dtype，不由 out 决定 ----
             (final-dtype (cond (dtype dtype)
                                (need-promote :float64)
                                (t in-dtype)))
             ;; out 硬契约：在计算开始前一次性校验（形状/dtype 严格相等/可写），
             ;; 之后 out 仅作写入目标，绝不下传给内部中间结果（避免 H5 误报）。
             (out-shape (vt-shape weighted-sum))
             (nan-val (vt-get-nan final-dtype))
             (scalar-divisor (coerce sum-weights (if (eq final-dtype :float32)
                                                     'single-float 'double-float))))
        (when out
          (vt-check-out out out-shape final-dtype :op-name "vt-average"))
        (cond ((%nan-p sum-weights)
               (cond (out (vt-fill out nan-val) out)
                     (t (vt-full out-shape nan-val :dtype final-dtype))))
              ((zerop sum-weights) (error "Weights sum to zero"))
              (t (vt-map (lambda (s) (/ s scalar-divisor))
                         weighted-sum :dtype final-dtype :out out)))))))

(defun vt-mean (tensor &key axis keepdims dtype out)
  "算术平均（对标 numpy.mean）。
  dtype 语义（§4.2 H3「严格相等」）：结果 dtype = 显式 :dtype，否则按输入提升
  （:float32 → :float32，其余 → :float64，均值恒为浮点）。**不由 out 决定**；
  out 必须精确匹配结果 shape/dtype。空归约区结果为 NaN。"
  (let* ((shape (vt-shape tensor))
         (rank (length shape)))
    (multiple-value-bind (axes count) (%get-axes-count axis rank shape)
      (let* ((compute-dtype (if (eq (vt-dtype tensor) :float32) :float32 :float64))
             ;; ---- 结果 dtype：输入提升 + 显式 :dtype，**不由 out 决定** ----
             ;; 用户裁决（§4.2 H3「严格相等」）：out 的 dtype 必须精确等于
             ;; 结果 dtype。旧有的「按输入提升计算、最后 cast 写入 out」解耦
             ;; 路径已弃用（exec-dtype 恒等于 final-dtype，write-out 恒为假）。
             (final-dtype (or dtype compute-dtype))
             (exec-dtype final-dtype))
        ;; 用户 out 的硬契约在此处校验一次（H1–H4：形状/可写/dtype 严格相等），
        ;; 之后 out 只作为「写入目标」使用，绝不再作为内部中间结果的容器下传，
        ;; 否则 vt-sum 的 H5（:dtype 与 :out 一致性）会先于本函数的 H3 触发，
        ;; 报出令人困惑的 "vt-SUM: ..." 错误信息。
        (when out
          (vt-check-out out
                        (if keepdims
                            (loop for d in shape for i below rank
                                  collect (if (or (null axes) (member i axes)) 1 d))
                            (loop for d in shape for i below rank
                                  unless (or (null axes) (member i axes)) collect d))
                        final-dtype :op-name "vt-mean"))
        (when (= count 0)
          (let ((nan (vt-get-nan exec-dtype))
                (out-shape (if keepdims
                               (loop for d in shape
                                     for i below rank
                                     collect (if (or (null axes) (member i axes)) 1 d))
                               (loop for d in shape
                                     for i below rank
                                     unless (or (null axes) (member i axes))
                                       collect d))))
            (return-from vt-mean
              (cond (out (vt-fill out nan) out)
                    (t (vt-full out-shape nan :dtype exec-dtype))))))
        (let* ((sum-result (vt-sum tensor :axis axes :keepdims keepdims
                                   :dtype exec-dtype))
               (div (coerce count (if (eq exec-dtype :float32)
                                      'single-float 'double-float))))
          (vt-map (lambda (s) (/ s div))
                  sum-result :dtype exec-dtype :out out))))))

(defun vt-var (tensor &key axis keepdims (ddof 0) dtype out)
  "方差（对标 numpy.var）。
  dtype 语义同 vt-mean（§4.2 H3「严格相等」）。DDOF 为自由度增量（缺省 0，
  即总体方差）。有效样本数 ≤ ddof 时结果为 NaN。"
  (let* ((shape (vt-shape tensor))
         (rank (length shape)))
    (multiple-value-bind (axes n) (%get-axes-count axis rank shape)
      (let* ((compute-dtype (if (eq (vt-dtype tensor) :float32) :float32 :float64))
             ;; 结果 dtype 由「输入提升 + 显式 :dtype」决定，不由 out 决定
             ;; （§4.2 H3「严格相等」）。
             (final-dtype (or dtype compute-dtype))
             (exec-dtype final-dtype)
             (divisor (- n ddof))
             (out-shape (if keepdims
                            (loop for d in shape for i below rank
                                  collect (if (or (null axes) (member i axes)) 1 d))
                            (loop for d in shape for i below rank
                                  unless (or (null axes) (member i axes)) collect d)))
             (emit (lambda (result-vt)
                     ;; 统一出口：out 已在外层经 vt-check-out 校验过形状/dtype，
                     ;; 这里只做写入（dtype 必然一致，直接用 vt-copy-into）
                     (if out
                         (progn (unless (eq result-vt out)
                                  (vt-copy-into out result-vt))
                                out)
                         result-vt))))
        ;; out 硬契约：在计算开始前校验（H3 dtype 严格相等）
        (when out
          (vt-check-out out out-shape final-dtype :op-name "vt-var"))
        (cond
          ((<= divisor 0)
           ;; ddof ≥ 归约区元素数：方差无定义 → NaN（numpy 语义）
           (funcall emit (vt-full out-shape (vt-get-nan exec-dtype)
                                  :dtype exec-dtype)))
          (t
           (let* ((mean-val (vt-mean tensor :axis axes :keepdims t
                                     :dtype exec-dtype))
                  (sq-diff (vt-square (vt-- tensor mean-val :dtype exec-dtype)
                                      :dtype exec-dtype))
                  (sum-sq (vt-sum sq-diff :axis axes :keepdims keepdims
                                  :dtype exec-dtype))
                  (res (vt-/ sum-sq divisor :dtype exec-dtype)))
             (funcall emit res))))))))

(defun vt-std (tensor &key axis keepdims (ddof 0) dtype out)
  "标准差 sqrt(vt-var)（对标 numpy.std）。dtype 语义同 vt-var。"
  (let* ((compute-dtype (if (eq (vt-dtype tensor) :float32) :float32 :float64))
         ;; 结果 dtype 同 vt-mean/vt-var：输入提升 + 显式 :dtype，不由 out 决定。
         (final-dtype (or dtype compute-dtype))
         (exec-dtype final-dtype))
    ;; 先做一次 out 硬契约校验（形状/dtype 严格相等/可写），让错误信息以
    ;; "vt-std" 开头，而不是透出内部 vt-var 的报错。
    (when out
      (let* ((shape (vt-shape tensor))
             (rank (length shape)))
        (multiple-value-bind (axes count) (%get-axes-count axis rank shape)
          (declare (ignore count))
          (vt-check-out out
                        (if keepdims
                            (loop for d in shape for i below rank
                                  collect (if (or (null axes) (member i axes)) 1 d))
                            (loop for d in shape for i below rank
                                  unless (or (null axes) (member i axes)) collect d))
                        final-dtype :op-name "vt-std"))))
    (let ((variance (vt-var tensor :axis axis :keepdims keepdims :ddof ddof
                            :dtype exec-dtype)))
      (setf variance (vt-sqrt variance :dtype exec-dtype))
      (cond ((null out) variance)
            ((eq variance out) out)
            (t (vt-copy-into out variance) out)))))

;;; ------------------------------------------------------------------
;;; 累积
;;; ------------------------------------------------------------------

(defun vt-cumulative (tensor op init-val &key axis dtype out)
  "累积归约核心原语：沿 AXIS 依次累积 OP（#\+ / #\*），以 INIT-VAL 起始。
  dtype 语义（§4.2 H3）：结果 dtype = 显式 :dtype，否则为输入 dtype；out 必须精确匹配。"
  (let* ((shape (vt-shape tensor))
         (rank (length shape))
         ;; 结果 dtype 由「输入 dtype + 显式 :dtype」决定，不由 out 决定。
         (final-dtype (or dtype (vt-dtype tensor)))
         (lisp-type (vt-dtype->lisp-type final-dtype))
         ;; out 给出时统一入口校验（形状/dtype 严格相等/可写性）
         (result (if out
                     (vt-check-out out shape final-dtype :op-name "vt-cumulative")
                     (vt-zeros shape :dtype final-dtype)))
         (in-data (vt-data tensor)) (out-data (vt-data result))
         (in-strides (vt-strides tensor)) (in-offset (vt-offset tensor))
         (out-strides (vt-strides result)) (out-offset (vt-offset result)))
    (if axis
        (let* ((ax (vt-normalize-axis axis rank))
               (ax-dim (nth ax shape))
               (indices (make-array rank :element-type '(signed-byte 64)
                                         :initial-element 0)))
          (labels ((advance ()
                     (loop for d from (1- rank) downto 0
                           when (/= d ax)
                             do (incf (aref indices d))
                                (if (< (aref indices d) (nth d shape))
                                    (return-from advance t)
                                    (setf (aref indices d) 0)))))
            (loop
              (let ((in-ptr in-offset) (out-ptr out-offset))
                (loop for d from 0 below rank do
                  (incf in-ptr (* (aref indices d)
                                  (nth d in-strides)))
                  (incf out-ptr (* (aref indices d)
                                   (nth d out-strides))))
                (let ((in-stride (nth ax in-strides))
                      (out-stride (nth ax out-strides))
                      (cum (coerce init-val lisp-type)))
                  (loop for i from 0 below ax-dim do
                    (setf cum (funcall op cum (aref in-data (+ in-ptr (* i in-stride)))))
                    (setf (aref out-data (+ out-ptr (* i out-stride)))
                          (vt-cast cum final-dtype)))))
              (unless (advance) (return)))))
        (let* ((flat (vt-ravel tensor))
               (flat-in (vt-data flat))
               (flat-offset (vt-offset flat))
               (cum (coerce init-val lisp-type))
               (out-shape-vec (coerce (vt-shape result) 'simple-vector))
               (out-strs-vec (coerce out-strides 'simple-vector))
               (out-rank (length out-shape-vec)))
          (labels ((recurse (depth out-ptr flat-idx)
                     (if (= depth out-rank)
                         (progn (setf cum (funcall op cum (aref flat-in (+ flat-offset flat-idx))))
                                (setf (aref out-data out-ptr)
                                      (vt-cast cum final-dtype))
                                (1+ flat-idx))
                         (let ((dim (svref out-shape-vec depth))
                               (stride (svref out-strs-vec depth)))
                           (loop for i from 0 below dim
                                 for cur = out-ptr then (+ cur stride)
                                 do (setf flat-idx (recurse (1+ depth) cur flat-idx)))))))
            (recurse 0 out-offset 0))))
    result))

(defun vt-cumsum (tensor &key axis dtype out)
  "沿 AXIS 的累积和（axis 缺省全展平）。等价 (vt-cumulative tensor #\+ 0 :axis axis)。"
  (vt-cumulative tensor #'+ 0 :axis axis :dtype dtype :out out))

(defun vt-cumprod (tensor &key axis dtype out)
  "沿 AXIS 的累积积（axis 缺省全展平）。等价 (vt-cumulative tensor #\* 1 :axis axis)。"
  (vt-cumulative tensor #'* 1 :axis axis :dtype dtype :out out))

;;; ------------------------------------------------------------------
;;; 中位数 / 百分位 / 直方图
;;; ------------------------------------------------------------------
(defun vt-median (tensor &key axis keepdims)
  "中位数（对标 numpy.median）。偶数个元素取中间两数平均。"
  (with-float-safe
    (if axis
        ;; ---- 轴归约 ----
        (let* ((shape (vt-shape tensor))
               (rank (length shape))
               (ax (vt-normalize-axis axis rank))
               (out-shape (if keepdims
                              (loop for d in shape for i from 0
                                    collect (if (= i ax) 1 d))
                              (loop for d in shape for i from 0
                                    unless (= i ax) collect d)))
               (result (vt-zeros out-shape :dtype :float64))
               (out-strides (vt-strides result)))
          (vt-do-each (ptr val result)
            (declare (ignore val))
            (let* ((out-idx (vt-unravel-index ptr out-shape out-strides))
                   (specs
                     (loop for i from 0 below rank
                           for out-i = (cond ((= i ax) nil)
                                             (keepdims i)
                                             ((< i ax) i)
                                             (t (1- i)))
                           collect (if (null out-i)
                                       '(:all)
                                       (list (nth out-i out-idx)))))
                   (fiber (apply #'vt-slice tensor specs))
                   (fs (vt-size fiber)))
              (setf (aref (vt-data result) ptr)
                    (if (zerop fs)
                        (vt-get-nan :float64)
                        (let ((vals (loop for i below fs collect (vt-ref fiber i))))
                          (if (some #'%nan-p vals)
                              (vt-get-nan :float64)
                              (let ((sv (vt-numpy-sort vals #'<)))
                                (if (oddp fs)
                                    (vt-cast (nth (floor fs 2) sv) :float64)
                                    (/ (+ (vt-cast (nth (1- (floor fs 2)) sv) :float64)
                                          (vt-cast (nth (floor fs 2) sv) :float64))
                                       2.0d0)))))))))
          result)
        ;; ---- 全局归约 ----
        (let* ((in-shape (vt-shape tensor))
               (rank (length in-shape))
               (flat (vt-flatten tensor))
               (size (vt-size flat))
               (out-shape (if keepdims
                              (make-list rank :initial-element 1)
                              nil)))
          (if (zerop size)
              (make-vt out-shape (vt-get-nan :float64) :dtype :float64)
              (let ((vals (loop for i below size collect (aref (vt-data flat) i))))
                (if (some #'%nan-p vals)
                    (make-vt out-shape (vt-get-nan :float64) :dtype :float64)
                    (let ((sv (vt-numpy-sort vals #'<)))
                      (if (oddp size)
                          (make-vt out-shape
                                   (vt-cast (nth (floor size 2) sv) :float64)
                                   :dtype :float64)
                          (make-vt out-shape
                                   (/ (+ (vt-cast (nth (1- (floor size 2)) sv) :float64)
                                         (vt-cast (nth (floor size 2) sv) :float64))
                                      2.0d0)
                                   :dtype :float64))))))))))

(defun %percent-from-sorted (sorted q interpolation)
  (let* ((n (length sorted))
         (idx (* q (1- n)))
         (lower (floor idx))
         (upper (min (ceiling idx) (1- n)))
         (frac (- idx lower)))
    (case interpolation
      (:linear (if (= lower upper) (vt-cast (nth lower sorted) :float64)
                   (+ (* (- 1 frac) (vt-cast (nth lower sorted) :float64))
                      (* frac (vt-cast (nth upper sorted) :float64)))))
      (:lower (vt-cast (nth lower sorted) :float64))
      (:higher (vt-cast (nth upper sorted) :float64))
      (:midpoint (/ (+ (vt-cast (nth lower sorted) :float64)
                       (vt-cast (nth upper sorted) :float64))
                    2.0d0))
      (:nearest (vt-cast (nth (if (<= frac 0.5d0) lower upper) sorted) :float64)))))

(defun vt-percentile (tensor percentile &key axis keepdims (interpolation :linear))
  "计算百分位数（对标 numpy.percentile）。
   PERCENTILE 必须在 [0, 100] 内：
     - 0   → 最小值
     - 50  → 中位数
     - 100 → 最大值
     越界（<0 或 >100）报错；NaN 亦报错（NaN 不满足 <= 比较）。
   INTERPOLATION 取值：:linear / :lower / :higher / :midpoint / :nearest，
     对标 numpy.percentile 的 interpolation 参数。默认 :linear。
   KEEPDIMS = t 时，轴归约保留归约轴为 1，全局归约返回 (1 1 ... 1)，
     与 numpy 的 keepdims=True 一致。
   返回：0 维张量（当 axis=nil 且 keepdims=nil）、形状 (1 1 ... 1)
         的张量（当 axis=nil 且 keepdims=t）、或降维后的张量。"
  (with-float-safe
    (unless (realp percentile)
      (error "vt-percentile: percentile 必须为实数，得到 ~a (type ~a)"
             percentile (type-of percentile)))
    (unless (<= 0 percentile 100)
      (error "vt-percentile: percentile 必须在 [0, 100] 内，得到 ~a"
             percentile))
    (unless (member interpolation '(:linear :lower :higher :midpoint :nearest))
      (error "vt-percentile: interpolation 必须是 :linear/:lower/:higher/:midpoint/:nearest 之一，得到 ~a"
             interpolation))
    (let ((q (/ percentile 100.0d0)))
      (if axis
          ;; ---- 轴归约 ----
          (let* ((shape (vt-shape tensor))
                 (rank (length shape))
                 (ax (vt-normalize-axis axis rank))
                 (out-shape (if keepdims
                                (loop for d in shape for i from 0
                                      collect (if (= i ax) 1 d))
                                (loop for d in shape for i from 0
                                      unless (= i ax) collect d)))
                 (result (vt-zeros out-shape :dtype :float64))
                 (out-strides (vt-strides result)))
            (vt-do-each (ptr val result)
              (declare (ignore val))
              (let* ((out-idx (vt-unravel-index ptr out-shape out-strides))
                     (specs
                       (loop for i from 0 below rank
                             for out-i = (cond ((= i ax) nil)
                                               (keepdims i)
                                               ((< i ax) i)
                                               (t (1- i)))
                             collect (if (null out-i)
                                         '(:all)
                                         (list (nth out-i out-idx)))))
                     (fiber (apply #'vt-slice tensor specs))
                     (fs (vt-size fiber)))
                (setf (aref (vt-data result) ptr)
                      (if (zerop fs)
                          (vt-get-nan :float64)
                          (let ((raw (loop for i below fs collect (vt-ref fiber i))))
                            (if (some #'%nan-p raw)
                                (vt-get-nan :float64)
                                (%percent-from-sorted
                                 (vt-numpy-sort raw #'<) q interpolation)))))))
            result)
          ;; ---- 全局归约 ----
          (let* ((in-shape (vt-shape tensor))
                 (rank (length in-shape))
                 (flat (vt-flatten tensor))
                 (size (vt-size flat))
                 (out-shape (if keepdims
                                (make-list rank :initial-element 1)
                                nil)))
            (if (zerop size)
                (make-vt out-shape (vt-get-nan :float64) :dtype :float64)
                (let ((raw (loop for i below size collect (aref (vt-data flat) i))))
                  (if (some #'%nan-p raw)
                      (make-vt out-shape (vt-get-nan :float64) :dtype :float64)
                      (make-vt out-shape
                               (%percent-from-sorted
                                (vt-numpy-sort raw #'<) q interpolation)
                               :dtype :float64)))))))))

(defun vt-quantile (tensor q &key axis keepdims (interpolation :linear))
  "计算分位数（对标 numpy.quantile）。
   Q 必须在 [0, 1] 内：
     - 0.0  → 最小值
     - 0.5  → 中位数
     - 1.0  → 最大值
     越界（<0 或 >1）报错；NaN 亦报错（NaN 不满足 <= 比较）。
   等价于 (vt-percentile tensor (* q 100) ...)。
   INTERPOLATION 含义见 vt-percentile；KEEPDIMS 语义见 vt-percentile。"
  (unless (realp q)
    (error "vt-quantile: q 必须为实数，得到 ~a (type ~a)"
           q (type-of q)))
  (unless (<= 0 q 1)
    (error "vt-quantile: q 必须在 [0, 1] 内，得到 ~a" q))
  (vt-percentile tensor (* q 100)
                 :axis axis
                 :keepdims keepdims
                 :interpolation interpolation))

(defun vt-ptp (tensor &key axis)
  "峰谷差 max - min（对标 numpy.ptp）。"
  (if axis (vt-- (vt-amax tensor :axis axis) (vt-amin tensor :axis axis))
      (- (vt-item (vt-amax tensor)) (vt-item (vt-amin tensor)))))

(defun vt-histogram (tensor &key bins range density)
  "直方图统计（对标 numpy.histogram）。返回 (values counts edges)。"
  (with-float-safe
    (let* ((flat (vt-flatten tensor))
           (data (vt-data flat))
           (size (vt-size flat))
           (bins (or bins 10))
           (data-min nil)
           (data-max nil))
      (unless (and (integerp bins) (plusp bins))
        (error "vt-histogram: bins (~a) 必须为正整数" bins))

      (if range
          (progn
            (unless (and (listp range) (= (length range) 2))
              (error "vt-histogram: range 必须是 (min max) 二元列表，得到 ~a"
                     range))
            (setf data-min (first range)
                  data-max (second range))
            (unless (< data-min data-max)
              (error "vt-histogram: range 中 max (~a) 必须严格大于 min (~a)"
                     data-max data-min)))
          (progn
            (unless (= (vt-item (vt-all (vt-isfinite tensor))) 1.0d0)
              (error "自动确定 bin 范围需要有限输入"))
            (setf data-min (vt-item (vt-amin tensor))
                  data-max (vt-item (vt-amax tensor)))
            (when (= data-min data-max)
              (setf data-min (- data-min 0.5)
                    data-max (+ data-max 0.5)))))
      (let* ((bin-width (/ (- data-max data-min) bins))
             (hist (make-array bins :initial-element 0))
             (edges (make-array (1+ bins) :element-type t)))
        (loop for i from 0 to bins do
          (setf (aref edges i) (+ data-min (* i bin-width))))
        (loop for i from 0 below size
              for val = (aref data i)
              when (and (>= val data-min) (<= val data-max))
                do (let ((bi (if (= val data-max)
                                 (1- bins)
                                 (floor (- val data-min) bin-width))))
                     (incf (aref hist bi))))
        (when density
          (let ((total (reduce #'+ hist)))
            (if (zerop total)
                (loop for i from 0 below bins do
                  (setf (aref hist i) 0.0d0))
                (loop for i from 0 below bins do
                  (setf (aref hist i)
                        (/ (aref hist i)
                           (* total bin-width)))))))
        (values (vt-from-sequence hist :dtype :float64)
                (vt-from-sequence edges :dtype :float64))))))

;;; ------------------------------------------------------------------
;;; 排序
;;; ------------------------------------------------------------------

(defun vt-sort (tensor &key (axis -1))
  "升序排序（对标 numpy.sort）。AXIS 缺省 -1；NaN 排在末尾。"
  (if axis
      (let* ((shape (vt-shape tensor))
             (rank (length shape))
             (ax (vt-normalize-axis axis rank))
             (ax-dim (nth ax shape))
             (in-strides (vt-strides tensor))
             (in-offset (vt-offset tensor))
             (in-data (vt-data tensor))
             (result (vt-copy tensor))
             (out-strides (vt-strides result))
             (out-data (vt-data result)))
        (labels ((recurse (depth in-ptr out-ptr)
                   (cond ((= depth ax)
                          (let* ((in-stride (nth ax in-strides))
                                 (out-stride (nth ax out-strides))
                                 (vals (loop for i from 0 below ax-dim
                                             for off = (+ in-ptr (* i in-stride))
                                             collect (aref in-data off)))
                                 (sv (vt-numpy-sort vals #'<)))
                            (loop for val in sv
                                  for off = out-ptr then (+ off out-stride)
                                  do (setf (aref out-data off) val))))
                         ((< depth rank)
                          (let ((dim (nth depth shape))
                                (in-stride (nth depth in-strides))
                                (out-stride (nth depth out-strides)))
                            (loop for i from 0 below dim do
                              (recurse (1+ depth)
                                       (+ in-ptr (* i in-stride))
                                       (+ out-ptr (* i out-stride))))))
                         (t nil))))
          (recurse 0 in-offset 0))
        result)
      (let* ((flat (vt-flatten tensor))
             (data (coerce (vt-data flat) 'list)))
        (vt-from-sequence (vt-numpy-sort data #'<) :dtype (vt-dtype tensor)))))

(defun vt-argsort (tensor &key (axis -1))
  "返回升序排序后的下标（对标 numpy.argsort）。AXIS 缺省 -1。"
  (with-float-safe
    (if (null axis)
        (let* ((flat (vt-ravel tensor))
               (n (vt-size flat))
               (in-data (vt-data flat))
               (pairs (loop for i from 0 below n
                            collect (cons (aref in-data i) i)))
               (non-nans '())
               (nans '()))
          (dolist (p pairs) (if (%nan-p (car p)) (push p nans) (push p non-nans)))
          (setf non-nans (stable-sort (nreverse non-nans) #'< :key #'car)
                nans (nreverse nans))
          (%make-vt :data (make-array n :element-type '(signed-byte 64)
                                        :initial-contents (mapcar #'cdr (append non-nans nans)))
                    :shape (list n) :strides '(1) :offset 0 :dtype :int64))
        (let* ((shape (vt-shape tensor))
               (rank (length shape))
               (ax (vt-normalize-axis axis rank))
               (ax-dim (nth ax shape))
               (in-strides (vt-strides tensor))
               (in-offset (vt-offset tensor))
               (in-data (vt-data tensor))
               (result (vt-zeros shape :dtype :int64))
               (out-strides (vt-strides result))
               (out-data (vt-data result)))
          (labels
              ((recurse (depth in-ptr out-ptr)
                 (cond ((< depth ax)
                        (let ((dim (nth depth shape))
                              (in-stride (nth depth in-strides))
                              (out-stride (nth depth out-strides)))
                          (dotimes (i dim)
                            (recurse (1+ depth)
                                     (+ in-ptr (* i in-stride))
                                     (+ out-ptr (* i out-stride))))))
                       ((= depth ax)
                        (let* ((in-stride (nth ax in-strides))
                               (out-stride (nth ax out-strides))
                               (tail-dims (subseq shape (1+ ax)))
                               (tail-size (reduce #'* tail-dims :initial-value 1))
                               (tail-in-strides (subseq in-strides (1+ ax)))
                               (tail-out-strides (subseq out-strides (1+ ax))))
                          (dotimes (tail-i tail-size)
                            (let ((extra-in 0)
                                  (extra-out 0)
                                  (rem tail-i))
                              (loop for idx from (1- (length tail-dims)) downto 0
                                    for dim = (nth idx tail-dims)
                                    for is = (nth idx tail-in-strides)
                                    for os = (nth idx tail-out-strides)
                                    do (multiple-value-bind (q r) (floor rem dim)
                                         (incf extra-in (* r is))
                                         (incf extra-out (* r os)) (setf rem q)))
                              (let ((pairs (loop for pos from 0 below ax-dim
                                                 for off = (+ in-ptr (* pos in-stride) extra-in)
                                                 collect (cons (aref in-data off) pos)))
                                    (non-nans '()) (nans '()))
                                (dolist (p pairs)
                                  (if (%nan-p (car p))
                                      (push p nans)
                                      (push p non-nans)))
                                (setf non-nans (stable-sort (nreverse non-nans) #'< :key #'car)
                                      nans (nreverse nans))
                                (loop for (val . pos) in (append non-nans nans)
                                      for off = out-ptr then (+ off out-stride)
                                      do (setf (aref out-data (+ off extra-out)) pos)))))))
                       (t nil))))
            (recurse 0 in-offset 0))
          result))))

;;; ------------------------------------------------------------------
;;; nan 感知统计
;;; ------------------------------------------------------------------

(defun vt-nanmean (tensor &key axis keepdims dtype out)
  "忽略 NaN 的算术平均。全为 NaN 的归约区结果为 NaN。

  dtype 语义（§4.2 H3「严格相等」）：结果 dtype = 输入浮点提升 + 显式 :dtype，
  **不由 out 决定**；out 的 dtype 必须精确等于结果 dtype。"
  (let* ((mask (vt-isnan tensor))
         (not-nan (vt-logical-not mask))
         (compute-dtype (if (eq (vt-dtype tensor) :float32) :float32 :float64))
         ;; 结果 dtype：输入提升 + 显式 :dtype（nan 族恒为浮点）
         (final-dtype (or dtype compute-dtype))
         (zero (if (eq final-dtype :float32) 0.0s0 0.0d0))
         (nan (vt-get-nan final-dtype))
         (clean (vt-where mask zero tensor :dtype final-dtype))
         (count (vt-sum not-nan :axis axis :keepdims keepdims :dtype final-dtype))
         (sum (vt-sum clean :axis axis :keepdims keepdims :dtype final-dtype)))
    ;; out 硬契约校验前置：错误信息以 "vt-nanmean" 开头
    (when out
      (vt-check-out out (vt-shape sum) final-dtype :op-name "vt-nanmean"))
    (vt-map (lambda (s c) (if (<= c zero) nan (/ s c)))
            sum count :dtype final-dtype :out out)))

(defun vt-nanvar (tensor &key axis keepdims (ddof 0) dtype out)
  "忽略 NaN 的方差。有效样本数 ≤ ddof 时结果为 NaN。

  dtype 语义同 vt-nanmean（§4.2 H3「严格相等」）。"
  (let* ((mask (vt-isnan tensor))
         (not-nan (vt-logical-not mask))
         (compute-dtype (if (eq (vt-dtype tensor) :float32) :float32 :float64))
         (final-dtype (or dtype compute-dtype))
         (nan (vt-get-nan final-dtype))
         (zero (if (eq final-dtype :float32) 0.0s0 0.0d0))
         (clean (vt-where mask zero tensor :dtype final-dtype))
         (count (vt-sum not-nan :axis axis :keepdims keepdims :dtype final-dtype))
         (mean (vt-nanmean tensor :axis axis :keepdims t :dtype final-dtype))
         (sq-diff (vt-* (vt-map (lambda (c m) (* (- c m) (- c m)))
                                clean mean :dtype final-dtype)
                        not-nan :dtype final-dtype))
         (sum2 (vt-sum sq-diff :axis axis :keepdims keepdims :dtype final-dtype))
         (ddof-f (coerce ddof (vt-dtype->lisp-type final-dtype)))
         (divisor (vt-map (lambda (c) (if (< c ddof-f) zero (- c ddof-f)))
                          count :dtype final-dtype)))
    (when out
      (vt-check-out out (vt-shape sum2) final-dtype :op-name "vt-nanvar"))
    (vt-map (lambda (s d) (if (<= d zero) nan (/ s d)))
            sum2 divisor :dtype final-dtype :out out)))

(defun vt-nanstd (tensor &key axis keepdims (ddof 0) dtype out)
  "忽略 NaN 的标准差（sqrt(vt-nanvar)）。dtype 语义同 vt-nanvar。"
  (let* ((compute-dtype (if (eq (vt-dtype tensor) :float32) :float32 :float64))
         (final-dtype (or dtype compute-dtype)))
    (let ((var (vt-nanvar tensor :axis axis :keepdims keepdims :ddof ddof
                          :dtype final-dtype)))
      (setf var (vt-sqrt var :dtype final-dtype))
      (cond ((null out) var)
            ((eq var out) out)
            (t (vt-check-out out (vt-shape var) final-dtype :op-name "vt-nanstd")
               (vt-copy-into out var)
               out)))))

(defun vt-nanmedian (tensor &key axis keepdims out)
  "忽略 NaN 的中位数（对标 numpy.nanmedian）。全为 NaN 的归约区结果为 NaN。
  结果 dtype 恒为 :float64；out 必须形状/dtype 精确匹配。"
  (with-float-safe
    (let* ((nan (vt-get-nan :float64))
           (in-data (vt-data tensor))
           (in-strides (vt-strides tensor))
           (in-offset (vt-offset tensor))
           (in-shape (vt-shape tensor))
           (rank (length in-shape)))
      (if (null axis)
          (let ((vals '()))
            (vt-do-each (ptr val tensor)
              (declare (ignore ptr))
              (unless (%nan-p val) (push val vals)))
            (setf vals (sort vals #'<))
            (let ((result (cond ((null vals) nan)
                                ((oddp (length vals))
                                 (coerce (nth (floor (length vals) 2) vals) 'double-float))
                                (t (/ (+ (nth (1- (/ (length vals) 2)) vals)
                                         (nth (/ (length vals) 2) vals))
                                      2.0d0)))))
              (if out
                  (progn
                    ;; 全局归约（axis=nil）：结果形状为 NIL（标量）。
                    ;; keepdims 时为全 1 形状。历史实现还容许 '(1)，保留兼容。
                    (let ((want (if keepdims
                                    (make-list (length (vt-shape tensor))
                                               :initial-element 1)
                                    nil)))
                      (unless (or (equal (vt-shape out) want)
                                  (equal (vt-shape out) '(1)))
                        (%vt-out-error "vt-nanmedian"
                                       ":out 形状 ~a 与全局归约结果 ~a 不兼容"
                                       (vt-shape out) want))
                      (unless (eq (vt-dtype out) :float64)
                        (%vt-out-error "vt-nanmedian"
                                       ":out dtype ~a 与结果 dtype FLOAT64 不匹配"
                                       (vt-dtype out)))
                      (unless (vt-out-writable-p out)
                        (%vt-out-error "vt-nanmedian"
                                       ":out 是只读的广播视图（存在 dim>1 且 stride=0 的轴）")))
                    (vt-fill out result)
                    out)
                  (make-vt nil result :dtype :float64))))
          (let* ((ax (vt-normalize-axis axis rank))
                 (ax-size (nth ax in-shape))
                 (ax-stride (nth ax in-strides))
                 (out-shape (if keepdims
                                (loop for d in in-shape for i from 0
                                      collect (if (= i ax) 1 d))
                                (loop for d in in-shape for i from 0
                                      unless (= i ax) collect d)))
                 (res (vt-zeros out-shape :dtype :float64))
                 (res-data (vt-data res)) (res-offset (vt-offset res))
                 (out-rank (length out-shape))
                 (out-dims (coerce out-shape 'simple-vector))
                 (out-strs (coerce (vt-strides res) 'simple-vector))
                 (in-map (if keepdims
                             (let ((m (make-array rank :element-type 'fixnum)))
                               (dotimes (i rank)
                                 (setf (aref m i) i))
                               m)
                             (let ((m (make-array (1- rank)
                                                  :element-type 'fixnum))
                                   (k 0))
                               (loop for i from 0 below rank unless (= i ax)
                                     do (setf (aref m k) i) (incf k))
                               m)))
                 (in-strs (coerce (loop for i below out-rank
                                        collect (nth (aref in-map i) in-strides))
                                  'simple-vector)))
            (labels ((compute (depth in-ptr out-ptr)
                       (if (= depth out-rank)
                           (let ((vals '()))
                             (loop for i from 0 below ax-size
                                   for ptr = in-ptr then (+ ptr ax-stride)
                                   for v = (aref in-data ptr)
                                   unless (%nan-p v)
                                     do (push (coerce v 'double-float) vals))
                             (setf vals (sort vals #'<))
                             (setf (aref res-data out-ptr)
                                   (cond ((null vals) nan)
                                         ((oddp (length vals))
                                          (nth (floor (length vals) 2) vals))
                                         (t (/ (+ (nth (1- (/ (length vals) 2)) vals)
                                                  (nth (/ (length vals) 2) vals))
                                               2.0d0)))))
                           (let ((dim (svref out-dims depth))
                                 (out-str (svref out-strs depth))
                                 (in-str (svref in-strs depth)))
                             (loop for i from 0 below dim do
                               (compute (1+ depth) in-ptr out-ptr)
                               (incf in-ptr in-str) (incf out-ptr out-str))))))
              (compute 0 in-offset res-offset))
            (if out
                (progn
                  ;; 统一硬契约（§4.2 H2/H3）：形状与 dtype 必须精确匹配
                  (vt-check-out out (vt-shape res) :float64 :op-name "vt-nanmedian")
                  (vt-copy-into out res)
                  out)
                res))))))

;;; ------------------------------------------------------------------
;;; 差分 / 积分 / 相关 / 卷积 / 插值 / 梯度
;;; ------------------------------------------------------------------

(defun vt-diff (vt &key (axis -1) (n 1))
  "沿 AXIS 计算 N 阶离散差分（对标 numpy.diff）。"
  (let ((result vt))
    (loop repeat n do
      (let* ((sh (vt-shape result))
             (ax (vt-normalize-axis axis (length sh)))
             (len (nth ax sh)))
        (when (< len 2)
          (return-from vt-diff
            (vt-zeros (append (subseq sh 0 ax) '(0) (subseq sh (1+ ax)))
                      :dtype (vt-dtype result))))
        (setf result (vt-- (vt-narrow result ax 1 len) (vt-narrow result ax 0 (1- len)))))
          finally (return result))))

(defun vt-trapz (y &key (x nil) (dx 1.0d0) (axis -1))
  "梯形法则数值积分（对标 numpy.trapz）。X 给定则用非均匀间距，否则用 DX。"
  (let* ((sh (vt-shape y))
         (ax (vt-normalize-axis axis (length sh)))
         (n (nth ax sh)))
    (when (< n 2)
      (return-from vt-trapz
        (vt-zeros (append (subseq sh 0 ax) (subseq sh (1+ ax))) :dtype (vt-dtype y))))
    (let ((h (if x
                 (vt-diff (ensure-vt x))
                 (make-vt (list (1- n)) dx :dtype (vt-dtype y)))))
      (setf h (vt-reshape h (append (make-list ax :initial-element 1) (list (1- n))
                                    (make-list (- (length sh) ax 1) :initial-element 1))))
      (let* ((left (vt-narrow y ax 0 (1- n)))
             (right (vt-narrow y ax 1 n))
             (integrand (vt-map (lambda (l r hh)
                                  (* 0.5d0 (+ l r) hh))
                                left right h)))
        (vt-sum integrand :axis ax)))))

(defun %normalize-mode-keyword (mode)
  "将 \"full\", :full 归一化为关键字 :full，方便 numpy 用户习惯。"
  (etypecase mode
    (keyword mode)
    (string (intern (string-upcase mode) "KEYWORD"))))

(defun vt-correlate (a v &key (mode :full))
  "1D 互相关，对标 np.correlate。mode 支持关键字(:full/:valid/:same)或字符串。"
  (let* ((mode-kw (%normalize-mode-keyword mode))
         (a-flat (vt-contiguous (vt-flatten a)))
         (v-flat (vt-contiguous (vt-flatten v)))
         (n (vt-size a-flat))
         (m (vt-size v-flat))
         (a-data (vt-data a-flat))
         (v-data (vt-data v-flat)))
    (flet ((compute (k)
             (let ((sum 0.0d0))
               (loop for j from (max 0 (- k)) below (min m (- n k))
                     do (incf sum (* (aref a-data (+ j k)) (aref v-data j))))
               sum)))
      (let* ((full-len (+ n m -1))
             (offset (1- m))
             (full (make-array full-len :element-type 'double-float)))
        (loop for k from (- offset) below n for i from 0
              do (setf (aref full i) (compute k)))
        (ecase mode-kw
          (:full (%make-vt :data full :shape (list full-len) :strides '(1)
                           :offset 0 :dtype :float64))
          (:valid (let* ((len (max 0 (1+ (- n m)))) (start offset)
                                                    (data (make-array len :element-type 'double-float)))
                    (loop for i from 0 below len do
                      (setf (aref data i) (aref full (+ start i))))
                    (%make-vt :data data :shape (list len) :strides '(1)
                              :offset 0 :dtype :float64)))
          (:same (let* ((out-len (max n m))
                        (start (floor (- full-len out-len) 2))
                        (data (make-array out-len :element-type 'double-float)))
                   (loop for i from 0 below out-len do
                     (setf (aref data i) (aref full (+ start i))))
                   (%make-vt :data data :shape (list out-len) :strides '(1)
                             :offset 0 :dtype :float64))))))))

(defun vt-convolve (a v &key (mode :full))
  "1D 卷积，对标 np.convolve。mode 支持关键字(:full/:valid/:same)或字符串。"
  (vt-correlate (vt-contiguous a)
                (vt-contiguous (vt-flip v))
                :mode (%normalize-mode-keyword mode)))

(defun vt-interp (x xp fp &key (left nil) (right nil))
  "一维线性插值（对标 numpy.interp）。
   在样本点 (xp[i], fp[i]) 上对查询点 x 做线性插值。
   参数：
     x       查询点，任意形状（输出恒展平为 1D）。
     xp      样本点横坐标，1D，**必须严格递增**。
     fp      样本点纵坐标，1D，长度必须等于 xp。
     :left   x < xp[0] 时的返回值，默认 fp[0]。
     :right  x > xp[last] 时的返回值，默认 fp[last]。
   返回：1D float64 张量，长度 = |x|。
   与 numpy 差异：
     · 输出总是 1D（numpy 保留 x 的形状）。
     · 无 period 参数。
     · xp 含 NaN 时行为未定义。
   示例：
     (vt-interp 2.5d0 #(1d0 3d0) #(10d0 30d0))  => 25.0d0
     (vt-interp #(0d0 5d0) #(1d0 3d0) #(10d0 30d0) :left -1d0 :right -2d0)
       => #(-1.0d0 -2.0d0)"
  (with-float-safe
    (let* ((xp-vt (vt-contiguous
                   (if (eq (vt-dtype (ensure-vt xp)) :float64)
                       (ensure-vt xp)
                       (vt-astype (ensure-vt xp) :float64))))
           (fp-vt (vt-contiguous
                   (if (eq (vt-dtype (ensure-vt fp)) :float64)
                       (ensure-vt fp)
                       (vt-astype (ensure-vt fp) :float64))))
           (n (vt-size xp-vt)))
      ;; ---- 前置校验 ----
      (when (zerop n)
        (error "vt-interp: xp 不能为空"))
      (unless (= n (vt-size fp-vt))
        (error "vt-interp: xp 长度 ~a ≠ fp 长度 ~a" n (vt-size fp-vt)))
      (loop for i from 1 below n
            when (< (vt-ref xp-vt i) (vt-ref xp-vt (1- i)))
              do (error "vt-interp: xp 必须在第 ~a 个位置递增（~a -> ~a 下降）"
                        i (vt-ref xp-vt (1- i)) (vt-ref xp-vt i)))
      (let* ((x-vt (vt-contiguous
                    (vt-flatten (ensure-vt x :dtype :float64))))
             (xp-data (vt-data xp-vt))
             (fp-data (vt-data fp-vt))
             (x-data (vt-data x-vt))
             (x-size (vt-size x-vt))
             (out (vt-zeros (list x-size) :dtype :float64))
             (out-data (vt-data out))
             (xp0 (aref xp-data 0))
             (fp0 (aref fp-data 0))
             (xp-end (aref xp-data (1- n)))
             (fp-end (aref fp-data (1- n)))
             (left-val (if left (vt-cast left :float64) fp0))
             (right-val (if right (vt-cast right :float64) fp-end)))
        (loop for i from 0 below x-size
              for xi = (aref x-data i) do
                (setf (aref out-data i)
                      (cond ((<= xi xp0) left-val)
                            ((>= xi xp-end) right-val)
                            (t (let ((lo 0) (hi (- n 2)))
                                 (loop while (< lo hi) do
                                   (let ((mid (ash (+ lo hi 1) -1)))
                                     (if (<= (aref xp-data mid) xi)
                                         (setf lo mid)
                                         (setf hi (1- mid)))))
                                 (let* ((xl (aref xp-data lo))
                                        (xr (aref xp-data (1+ lo)))
                                        (yl (aref fp-data lo))
                                        (yr (aref fp-data (1+ lo)))
                                        (denom (- xr xl)))
                                   (if (zerop denom)
                                       yl
                                       (+ yl (* (- yr yl)
                                                (/ (- xi xl) denom))))))))))
        out))))

(defun vt-gradient (tensor &key (spacing 1.0d0) axis)
  "数值梯度（对标 numpy.gradient）。SPACING 为标量或每轴的间距；AXIS 限定求梯度的轴。"
  (let* ((shape (vt-shape tensor))
         (rank (length shape))
         (axes (cond ((null axis)
                      (loop for i below rank collect i))
                     ((integerp axis) (list (vt-normalize-axis axis rank)))
                     ((listp axis) (mapcar (lambda (a) (vt-normalize-axis a rank)) axis))
                     (t (error "axis 必须是 nil、整数或整数列表"))))
         (spacings (cond ((numberp spacing) (make-list (length axes) :initial-element spacing))
                         ((listp spacing) spacing)
                         ((vt-p spacing) (list spacing))
                         (t (error "spacing 必须是数字、列表或 1d 张量")))))
    (labels ((slice-specs (ax s e)
               (loop for d from 0 below rank
                     collect (if (= d ax) (list s e) '(:all))))
             (grad-along (ax sp)
               (let ((n (nth ax shape)))
                 (when (< n 2) (error "轴 ~a 长度 ~a 太小" ax n))
                 (if (numberp sp)
                     (if (= n 2)
                         (let ((edge (vt-/ (vt-- (apply #'vt-slice tensor (slice-specs ax 1 2))
                                                 (apply #'vt-slice tensor (slice-specs ax 0 1)))
                                           sp)))
                           (vt-concatenate ax edge edge))
                         (let ((left (vt-/ (vt-- (apply #'vt-slice tensor (slice-specs ax 1 2))
                                                 (apply #'vt-slice tensor (slice-specs ax 0 1)))
                                           sp))
                               (inner (vt-/ (vt-- (apply #'vt-slice tensor (slice-specs ax 2 n))
                                                  (apply #'vt-slice tensor (slice-specs ax 0 (- n 2))))
                                            (* 2.0d0 sp)))
                               (right (vt-/ (vt-- (apply #'vt-slice tensor (slice-specs ax (1- n) n))
                                                  (apply #'vt-slice tensor (slice-specs ax (- n 2) (1- n))))
                                            sp)))
                           (vt-concatenate ax left inner right)))
                     (let* ((h (ensure-vt sp)) (nh (vt-size h)))
                       (assert (= n nh) (sp) "spacing 数组长度必须与轴一致")
                       (if (= n 2)
                           (let* ((hd (vt-- (vt-slice h '(1 2)) (vt-slice h '(0 1))))
                                  (df (vt-- (apply #'vt-slice tensor (slice-specs ax 1 2))
                                            (apply #'vt-slice tensor (slice-specs ax 0 1))))
                                  (g (vt-/ df hd)))
                             (vt-concatenate ax g g))
                           (let ((hl (vt-- (vt-slice h '(1 2)) (vt-slice h '(0 1))))
                                 (hr (vt-- (vt-slice h (list (1- n) n)) (vt-slice h (list (- n 2) (1- n)))))
                                 (hi (vt-- (vt-slice h (list 2 n)) (vt-slice h (list 0 (- n 2)))))
                                 (dl (vt-- (apply #'vt-slice tensor (slice-specs ax 1 2))
                                           (apply #'vt-slice tensor (slice-specs ax 0 1))))
                                 (dr (vt-- (apply #'vt-slice tensor (slice-specs ax (1- n) n))
                                           (apply #'vt-slice tensor (slice-specs ax (- n 2) (1- n)))))
                                 (di (vt-- (apply #'vt-slice tensor (slice-specs ax 2 n))
                                           (apply #'vt-slice tensor (slice-specs ax 0 (- n 2))))))
                             (vt-concatenate ax (vt-/ dl hl) (vt-/ di hi) (vt-/ dr hr)))))))))
      (let ((results (loop for ax in axes for sp in spacings
                           collect (grad-along ax sp))))
        (if (and (or (null axis) (integerp axis))
                 (null (cdr results)))
            (car results) results)))))

