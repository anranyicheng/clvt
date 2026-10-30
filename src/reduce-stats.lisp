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

  (defparameter +all-lts+ (append +kernel-lts+ +small-int-lts+)
    "全部 8 种元素类型（= 8 种逻辑 dtype）。用于空归约单位元等
     需要覆盖全部逻辑 dtype 的编译期展开。")

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
      ;; :bool 与 :uint8 共用物理存储（§17.1）；归约内核只按元素类型特化，
      ;; 因此布尔输入天然复用 (unsigned-byte 8) 的那套内核。
      (:bool    '(unsigned-byte 8))
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
     运行时辅助，供 %kernel-small-general 的 max/min 初始化使用。
     `:bool` 的极值是 0 / 1：布尔归约的 max 单位元为 0（False）、
     min 单位元为 1（True），与 numpy 的 `np.max(bool_array)` 一致。"
    (ecase dtype
      (:int8   (if (eq which :min) -128  127))
      (:int16  (if (eq which :min) -32768 32767))
      (:uint8  (if (eq which :min) 0     255))
      (:uint16 (if (eq which :min) 0     65535))
      (:bool   (if (eq which :min) 0     1))))

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
      ;; numpy 的逻辑归约返回 bool_（§17）：全归约给 0 维 :bool 张量，
      ;; 沿轴归约给 :bool 数组。此前承载为 :int8，导致 `vt-all` 的结果
      ;; 能被当成整数继续算术（True+True=2），语义立刻跑偏。
      ((:all :any) :bool)
      ;; numpy 的 argmax/argmin 返回 intp（64 位平台为 int64）；
      ;; 修正（审查 P2#1）：原实现返回 :int32，与 numpy 不符。
      ((:argmax :argmin :nanargmax :nanargmin) :int64)
      ((:sum :prod :nansum :nanprod)
       (case in-dtype
         ;; numpy：整数与布尔的 sum/prod 一律提升到 int64（bool 也是「整数类」，
         ;; `np.sum(np.array([True,True]))` 是 2，dtype int64）。
         ((:bool :int8 :uint8 :int16 :uint16 :int32) :int64)
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
  "路径 2 的列扫内核：连续 + 归约区可塌成一层（单轴或多轴连续区间）。"
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

(defconstant +row-acc-min-inner+ 16
  "行累加内核（%KERNEL-ROW-ACC）的启用门槛：内层维 ≥ 16 个元素。
   内层维太窄时「累加器常驻 L1」没有意义（累加器本身只有几十字节，
   而每行只有 1~2 条 cache line），反而多一次临时数组分配与写回，
   此时列扫内核（%KERNEL-SINGLE-AXIS）本来就是顺序访问。")

(eval-when (:compile-toplevel :load-toplevel :execute)

  (defun %small-int-axis-form (op)
    "窄整型（4 种逻辑级 dtype）的沿轴归约分派：in-lt × res-lt 全部实例化。

     与存储级 dtype 用同一套内核模板，只是 in-lt 换成窄整型：
     内层维够宽走行累加（%KERNEL-ROW-ACC），否则走列扫（%KERNEL-SINGLE-AXIS）。
     这两个内核的输入侧都只有 `(the (simple-array ,lt (*)))` 与
     `%op-step`，对窄整型成立（累加器提升到 int64 由 %OP-ACC-LT 决定）。

     单独成函数、而不是写进 DEF-VT-REDUCE 的模板里：模板已经是三层嵌套的
     反引号，再往里塞「按类型展开的 cond」既读不懂也容易括号错位。"
    `(cond
       ,@(loop for lt in '((signed-byte 16) (signed-byte 8)
                           (unsigned-byte 16) (unsigned-byte 8))
               collect
               `((equal in-et ',lt)
                 (cond
                   ;; res-lt 用 +ALL-LTS+：窄整型输入的 max/min 结果**保持**
                   ;; 输入 dtype（int8 进 int8 出），sum/prod 才提升到 int64。
                   ;; 只列 +KERNEL-LTS+ 会让 `(vt-amax int16-array)` 报
                   ;; "unsupported output dtype (SIGNED-BYTE 16)"。
                   ,@(loop for rlt in +all-lts+
                           collect
                           `((equal res-lt ',rlt)
                             (if (>= inner +row-acc-min-inner+)
                                 (%kernel-row-acc ,op ,lt ,rlt)
                                 (%kernel-single-axis ,op ,lt ,rlt))))
                   (t (error "vt-~a: unsupported output dtype ~a" ',op res-lt)))))
       (t (error "vt-~a: unsupported input dtype ~a" ',op in-et))))

  (defun %contig-axis-test ()
    "路径 2 的几何条件：输入/输出都连续，且归约轴在 shape 里连成一段。

     连成一段是「多层轴塌成一层」的充要条件：单轴天然满足；
     (0 1)、(1 2) 满足；(0 2) 不满足（中间夹着一层不归约的轴）。"
    '((consp axes)                          ; 非全归约（全归约走路径 1）
      (vt-contiguous-p tensor)
      (vt-contiguous-p res)
      (loop for x in axes
            for y in (cdr axes)
            always (= y (1+ x)))))

  (defun %contig-axis-body (op arg-p kernel)
    "路径 2 的 clause body（**forms 列表**，供 CONS 拼到测试条件后面）。

     三种内容共用同一段几何量绑定（A/B/OUTER/RED/INNER），只在内层分叉：
       · arg 族：只有列扫实例（argmax/argmin 只接受单轴，无需门槛判断）；
       · 窄整型：行累加或列扫（由 %SMALL-INT-AXIS-FORM 决定）；
       · 存储级：内层维够宽走行累加，否则列扫。
     代码本体是一层反引号模板，读起来就是真代码；KERNEL 是
     (lambda (macro-name) form)，把 DEF-VT-REDUCE 里的局部 DISPATCH 包一层。"
    (let ((inner-form
            (cond (arg-p (funcall kernel '%kernel-single-axis))
                  (t `(if (member in-et +small-int-ets+ :test #'equal)
                          ,(%small-int-axis-form op)
                          (if (>= inner +row-acc-min-inner+)
                              ,(funcall kernel '%kernel-row-acc)
                              ,(funcall kernel '%kernel-single-axis)))))))
      `((let* ((a (the fixnum (first axes)))
               (b (the fixnum (1+ (the fixnum (car (last axes))))))
               (outer (let ((r 1))
                        (dotimes (i a r)
                          (setf r (* r (the fixnum (nth i in-shape)))))))
               (red (let ((r 1))
                      (loop for i from a below b
                            do (setf r (* r (the fixnum (nth i in-shape)))))
                      r))
               (inner (let ((r 1))
                        (loop for i from b below (length in-shape)
                              do (setf r (* r (the fixnum (nth i in-shape)))))
                        r)))
          (declare (fixnum a b outer red inner))
          ,inner-form))))

  (defun %contig-axis-clause (op arg-p dispatch)
    "生成「归约轴连成一段」的那条 COND clause（恒为 1 条，包成列表便于 ,@）。

     一条 clause 覆盖四种情形，几何条件只判一次：
       单轴 / 多轴  ×  存储级 dtype / 窄整型
     arg 族只服务单轴 + 存储级：numpy 的 argmax/argmin 不接受轴元组，
     窄整型也没有 arg 实例（留在路径 3 的 %KERNEL-SMALL-GENERAL）。

     结构与写法：测试条件由 %CONTIG-AXIS-TEST 拼、代码由 %CONTIG-AXIS-BODY
     用反引号写，这里只做 `(cons 测试 body)` 的装配 —— 不再用 '(quote ...)
     手搭整棵 clause（那样既读不懂，也最容易数错括号）。"
    (let* ((kernel (lambda (macro-name) (funcall dispatch macro-name)))
           (extra (if arg-p
                      '((= (length axes) 1)
                        (not (member in-et +small-int-ets+ :test #'equal)))
                      '((not (eq (vt-data res) in-data)))))
           (test (cons 'and (append (%contig-axis-test) extra))))
      (list (cons test (%contig-axis-body op arg-p kernel))))))


(defmacro %kernel-row-acc (op lt res-lt)
  "行累加内核：连续 + 归约区可塌成一层 + 内层维够宽（INNER ≥ %ROW-ACC-MIN-INNER）。

   列扫内核（%KERNEL-SINGLE-AXIS）的内层循环是「固定列 i，沿归约轴跨 stride 走」，
   步长 = INNER。
   axis=0 时 INNER = n（1000×1000 即 8KB）：每个 (r,i) 都要换一条 cache line，
   整列扫完再扫下一列时那些行早已被挤出 L2 —— 实测 sum@axis0 1M 需要 3ms，
   而同一份数据 numpy 只要 0.15ms。amax@axis0 更差（20ms）。

   本路径把内层维整体当成累加器：acc[0..INNER) 常驻 L1，按**行**顺序推进，
   一次读取正好是一条连续区间，两个流（d 与 acc）都是顺序的。

   铁律 1（连续性只影响选路）：同一输出元素看到的累加次序仍是
   r = 0,1,2,… 与列扫内核完全一致，故结果**逐位相同**；
   只是把「不同输出元素之间」的循环翻到外层，不改变任何单个元素的求和顺序。

   arg 族不进本路径（需要同步维护下标数组，收益/复杂度不划算）。"
  (let* ((acc-lt (%op-acc-lt op lt res-lt))
         (init   (%op-init   op lt res-lt)))
    `(let ((d  (the (simple-array ,lt (*)) in-data))
           (od (the (simple-array ,res-lt (*)) res-data))
           (init-v (the ,acc-lt ,init)))
       (declare (type fixnum outer red inner))
       (let ((acc (make-array inner :element-type ',acc-lt)))
         (declare (type (simple-array ,acc-lt (*)) acc))
         (dotimes (o outer)
           (let ((in-base (the fixnum (+ in-off (* o red inner))))
                 (out-base (the fixnum (+ res-off (* o inner)))))
             (declare (type fixnum in-base out-base))
             (dotimes (i inner) (setf (aref acc i) init-v))
             (dotimes (r red)
               (let ((row (the fixnum (+ in-base (* r inner)))))
                 (declare (type fixnum row))
                 (dotimes (i inner)
                   (%op-step ,op (aref acc i) (aref d (the fixnum (+ row i)))
                             ,lt ,acc-lt))))
             (dotimes (i inner)
               (setf (aref od (the fixnum (+ out-base i)))
                     ,(%cast-form res-lt '(aref acc i))))))))))

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
  (let (;; 公开名由 DEF-VT-REDUCE-FACADE 生成，这里不需要 FN-NAME
        ;; 宏生成的函数体改名为内部实现，公开名由 DEF-VT-REDUCE-FACADE
        ;; 生成（唯一的一个 DEFUN），复数分派就在那个门面里。这样
        ;;   · 公开符号只有一个定义，重载任何单个文件都不会丢复数支持；
        ;;   · 复数分支需要「把变换后的实参再喂给实数实现」（all/any 走
        ;;     掩码）时，可以直接调 %REDUCE-<NAME>，不需要 %CX-ORIG 快照。
        (impl-name (intern (format nil "%%REDUCE-~a" name)))
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
                            ;; all/any 的累加器恒为 0/1，窄整型不会溢出；内核宏
                            ;; 对 res-lt 全参数化，按**字面量** res-lt 生成实例
                            ;; （:bool 的物理类型是 (unsigned-byte 8)，与 :uint8
                            ;; 同格；写死 (signed-byte 8) 会让内核里的
                            ;; `(the (simple-array …))` 断言与真实缓冲区不符）。
                            ;; sum/prod 等需要大累加域的算子不开放此路径。
                            ,@(loop for rlt in +small-int-lts+
                                    collect
                                    `((and (member ',op '(:all :any))
                                           (equal res-lt ',rlt))
                                      (,kernel-macro ,op ,lt ,rlt)))
                            (t (error "unsupported output dtype ~a" res-lt)))))
                (t (error "unsupported input dtype ~a" in-et)))))
      `(defun ,impl-name (tensor &key axis keepdims dtype out)
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
                    (return-from ,impl-name res))
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
                                 ;; all/any 的窄整型输出（:bool 的物理类型
                                 ;; 也在这一格里）——单位元与 int64 相同
                                 ;; （all→1 / any→0）
                                 ((and (member ',op '(:all :any))
                                       (member res-lt +small-int-lts+ :test #'equal))
                                  (cond
                                    ,@(loop for l in +all-lts+
                                            collect
                                            `((equal lt ',l)
                                              ,(%op-init op l '(signed-byte 64))))
                                    (t 0)))
                                 (t 0)))
                      (return-from ,impl-name res)))))
               ;; ---- 主分派 ----
               ;; 路径 1：全归约（只服务存储级 dtype；窄整型由路径 3 兜底）。
               ;; 路径 2：沿轴归约，条件是「归约轴在 shape 里连成一段」——
               ;;         单轴与多轴、存储级与窄整型都收在这一条 clause 里，
               ;;         具体内核由 %CONTIG-AXIS-CLAUSE 决定。
               ;; 路径 3：通用回退（任意 strides / 不连续轴 / arg 族窄整型）。
               (cond
                 ;; 路径 1：连续 + 全局 + 存储级 dtype
                 ((and global (vt-contiguous-p tensor)
                       (not (member in-et +small-int-ets+ :test #'equal)))
                  ,(dispatch '%kernel-global))
                 ;; 路径 2：连续 + 归约轴连成一段（单轴或多轴，含窄整型）。
                 ;; 选路细节（几何条件、内核、门槛）全在 %CONTIG-AXIS-CLAUSE
                 ;; 里，模板只留一行 —— 这里的嵌套反引号已经有四层，
                 ;; 再往模板里塞代码生成内容只会制造括号事故。
                 ,@(%contig-axis-clause op arg-p #'dispatch)

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

(defmacro def-vt-reduce-facade (name op &key fast)
  "为 DEF-VT-REDUCE 生成的 %REDUCE-<NAME> 补一个**唯一**的公开 DEFUN。
   复数入参在此分派；实数路径原样转给实现函数。

   FAST（可选）是一个 5 参函数 (TENSOR AXIS KEEPDIMS DTYPE OUT) → 结果或 NIL，
   在门面最前面尝试。这样「全归约快路径」不必覆盖公开符号（§16.32 禁止
   靠加载顺序覆盖），也不需要在宏里塞算子专属的 dtype 规则。"
  (let ((fn-name (intern (format nil "VT-~a" name)))
        (impl-name (intern (format nil "%%REDUCE-~a" name))))
    `(defun ,fn-name (tensor &key axis keepdims dtype out)
       ,(format nil "沿 AXIS 归约；复数输入的分派入口（实现在 %REDUCE-~a）。"
                name)
       (declare (type vt tensor)
                (type (or null fixnum list) axis)
                (type (or null vt) out))
       ,@(when fast
           `((let ((fast-result (funcall ,fast tensor axis keepdims dtype out)))
               (when fast-result (return-from ,fn-name fast-result)))))
       (when (%cx-any-p tensor)
         (return-from ,fn-name
           (%cx-reduce-dispatch ,op ,(string fn-name) (function ,impl-name)
                                tensor axis keepdims dtype out)))
       (,impl-name tensor :axis axis :keepdims keepdims
                   :dtype dtype :out out))))

(def-vt-reduce sum  :sum)
(def-vt-reduce-facade sum :sum)
(def-vt-reduce prod :prod)
(def-vt-reduce-facade prod :prod)
(def-vt-reduce amax :max)
(def-vt-reduce-facade amax :max)
(def-vt-reduce amin :min)
(def-vt-reduce-facade amin :min)
(def-vt-reduce all  :all)
(def-vt-reduce-facade all :all)
(def-vt-reduce any  :any)
(def-vt-reduce-facade any :any)

(def-vt-reduce nansum  :nansum)

(defun %nansum-fast (tensor axis keepdims dtype out)
  "`VT-NANSUM` 的全归约快路径：单趟 `%NAN-SUM-COUNT`。
   不适用时返回 NIL，由门面继续走通用实现。

   为什么需要它：生成版只能走通用 `%OP-STEP` 循环 —— 每元素一次 `%NAN-P`
   调用 + odometer 逐维取址。1M double 实测 **8.9 ms**，而同族
   `VT-NANMEAN` 早就有单趟快路径（0.7 ms）：同一个「跳过 NaN 求和」，
   两条路差 12 倍。"
  (let* ((v (ensure-vt tensor))
         (in-dt (vt-dtype v))
         (rdt (or dtype in-dt)))
    (when (and (%nan-fast-p axis keepdims rdt)
               (eq rdt in-dt)
               (member in-dt '(:float32 :float64)))
      (multiple-value-bind (sum c) (%nan-sum-count (%nan-prepare v rdt) rdt)
        (declare (ignore c))
        (%nan-scalar-result out
                            (if keepdims
                                (make-list (length (vt-shape v)) :initial-element 1)
                                nil)
                            rdt sum "VT-NANSUM")))))

(def-vt-reduce-facade nansum :nansum :fast #'%nansum-fast)
(def-vt-reduce nanprod :nanprod)
(def-vt-reduce-facade nanprod :nanprod)
(def-vt-reduce nanmax  :nanmax)
(def-vt-reduce-facade nanmax :nanmax)
(def-vt-reduce nanmin  :nanmin)
(def-vt-reduce-facade nanmin :nanmin)

(def-vt-reduce argmax    :argmax)
(def-vt-reduce-facade argmax :argmax)
(def-vt-reduce argmin    :argmin)
(def-vt-reduce-facade argmin :argmin)
(def-vt-reduce nanargmax :nanargmax)
(def-vt-reduce-facade nanargmax :nanargmax)
(def-vt-reduce nanargmin :nanargmin)
(def-vt-reduce-facade nanargmin :nanargmin)


;;; ------------------------------------------------------------------
;;; 判定算子（具名 + 内联）
;;;
;;; 类型声明（性能）：这三个函数原本以 lambda 形式交给 vt-map，
;;; 循环体里没有任何类型信息（泛型 aref + 装箱 funcall + 泛型 cast），
;;; 而结果 dtype 固定 :bool（物理 (unsigned-byte 8)），正落在
;;; %vt-mask1-fast 的「窄整型输出」矩阵
;;; （double/single/int64/int32 进 → (unsigned-byte 8) 出）内。
;;; 改成具名 inline 算子后即可命中该特化循环。
;;; ------------------------------------------------------------------
(declaim (inline %op-isfinite %op-isinf %op-isnan))

(defun %op-isfinite (x) (if (and (not (%nan-p x)) (not (%inf-p x))) 1 0))

(defun %op-isinf (x) (if (%inf-p x) 1 0))

(defun %op-isnan (x) (if (%nan-p x) 1 0))

(defun vt-isclose (t1 t2 &key (rtol 1e-5) (atol 1e-8) out)
  "逐元素判断 |a - b| <= atol + rtol*|b|，返回 **:bool** 布尔张量（§17）。
   v0.4.x 修复两点：
     1) 公式原用 max(|a|,|b|)，与 numpy.isclose 的 |b| 不一致，已对齐；
     2) 结果 dtype 原误为 :float64，违反「比较返回布尔」，已修正。
   NaN 恒不接近（numpy 默认 equal_nan=False）；±Inf 仅与相等的 ±Inf 接近。
   RTOL/ATOL 缺省 1e-5 / 1e-8（float64 计算精度，对标 numpy 内部实现）。

   性能：两输入连续、同形、同元素类型（double/single/int64/int32）且
   out 不与输入重叠时走特化内核（RTOL/ATOL 提为循环外局部变量）；
   其余情形回退 vt-map 通用路径，结果逐位相同。"
  (with-float-safe
    (let* ((a (ensure-vt t1)) (b (ensure-vt t2))
           (out-shape (vt-broadcast-shapes (vt-shape a) (vt-shape b)))
           (res (if out
                    (vt-check-out out out-shape :bool :op-name "vt-isclose")
                    (make-vt out-shape 0 :dtype :bool))))
      (macrolet ((close-kernel (lt)
                   `(let ((od (the (simple-array (unsigned-byte 8) (*)) (vt-data res)))
                          (d0 (the (simple-array ,lt (*)) (vt-data a)))
                          (d1 (the (simple-array ,lt (*)) (vt-data b)))
                          (rt (coerce rtol 'double-float))
                          (at (coerce atol 'double-float))
                          (n (the fixnum (vt-size res)))
                          (p0 (the fixnum (vt-offset a)))
                          (p1 (the fixnum (vt-offset b))))
                      (declare (type (simple-array (unsigned-byte 8) (*)) od)
                               (type (simple-array ,lt (*)) d0 d1)
                               (type double-float rt at)
                               (type fixnum n p0 p1))
                      (dotimes (i n)
                        (declare (type fixnum i))
                        (let* ((x (coerce (aref d0 (+ p0 i)) 'double-float))
                               (y (coerce (aref d1 (+ p1 i)) 'double-float)))
                          (setf (aref od i)
                                (cond ((or (%nan-p x) (%nan-p y)) 0)
                                      ((or (%inf-p x) (%inf-p y)) (if (= x y) 1 0))
                                      (t (if (<= (abs (- x y))
                                                 (+ at (* rt (abs y))))
                                             1 0)))))))))
        (let ((et (array-element-type (vt-data a))))
          (when (and (eq (vt-dtype res) :bool)
                     (equal (array-element-type (vt-data res)) '(unsigned-byte 8))
                     (eq (vt-offset res) 0)
                     (equal (vt-strides res) (vt-compute-strides out-shape))
                     (vt-contiguous-p a) (vt-contiguous-p b)
                     (equal (vt-shape a) out-shape)
                     (equal (vt-shape b) out-shape)
                     (equal et (array-element-type (vt-data b)))
                     (member et (list 'double-float 'single-float
                                      '(signed-byte 64) '(signed-byte 32))
                             :test #'equal)
                     ;; 别名（out 与输入重叠）交给 vt-map 的快照机制
                     (not (and out (or (%vt-views-overlap-p out a)
                                       (%vt-views-overlap-p out b)))))
            (cond ((equal et 'double-float)     (close-kernel double-float))
                  ((equal et 'single-float)     (close-kernel single-float))
                  ((equal et '(signed-byte 64)) (close-kernel (signed-byte 64)))
                  ((equal et '(signed-byte 32)) (close-kernel (signed-byte 32))))
            (return-from vt-isclose res)))))
    (vt-map (lambda (a b)
              (let ((fa (vt-cast a :float64)) (fb (vt-cast b :float64)))
                (cond ((or (%nan-p fa) (%nan-p fb)) 0)
                      ((or (%inf-p fa) (%inf-p fb)) (if (= fa fb) 1 0))
                      (t (if (<= (abs (- fa fb))
                                 (+ atol (* rtol (abs fb))))
                             1 0)))))
            t1 t2 :dtype :bool :out out)))

(defun vt-allclose (t1 t2 &key (rtol 1e-5) (atol 1e-8))
  "判断两个张量在 RTOL/ATOL 容差下是否全部相等（对标 numpy.allclose），
   返回 **Lisp 布尔值**（numpy 返回 Python bool，不是 0-d 数组）。
   判据直接取 `VT-ITEM` 对 `:bool` 张量的解包结果（T/NIL，§17.8），
   不是和 1.0d0 比大小 —— 后者对任何非 1 的数值都会静默给 NIL，
   把「0/1 承载」的假设固化进调用方。"
  (vt-item (vt-all (vt-isclose t1 t2 :rtol rtol :atol atol))))

(defun vt-isfinite (vt &key out)
  "逐元素判断是否有限（既非 NaN 也非 ±Inf），返回 **:bool** 布尔张量。
      dtype 固定 :bool（对标 numpy.isfinite，CONVENTIONS §17）。"
  (when (%cx-any-p vt)
    (return-from vt-isfinite
      (vt-fast-map (function %cx-finite-flag) (ensure-vt vt)
                   :out out :dtype :bool)))
  (vt-fast-map #'%op-isfinite vt :dtype :bool :out out))

(defun vt-isinf (vt &key out)
  "逐元素判断是否为 ±Inf，返回 **:bool** 布尔张量。dtype 固定 :bool。"
  (when (%cx-any-p vt)
    (return-from vt-isinf
      (vt-fast-map (function %cx-inf-flag) (ensure-vt vt)
                   :out out :dtype :bool)))
  (vt-fast-map #'%op-isinf vt :dtype :bool :out out))

(defun vt-isnan (vt &key out)
  "逐元素判断是否为 NaN，返回 **:bool** 布尔张量。dtype 固定 :bool。
   NaN 判定与 dtype 无关：整数/布尔输入恒为 False（numpy 同样如此，
   整数数组不可能含 NaN，只是本库不为此报错）。"
  (when (%cx-any-p vt)
    (return-from vt-isnan
      (vt-fast-map (function %cx-nan-flag) (ensure-vt vt)
                   :out out :dtype :bool)))
  (vt-fast-map #'%op-isnan vt :dtype :bool :out out))

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

(defun vt-average (tensor &rest args)
  "加权平均：sum(tensor * weights) / sum(weights)。对标 numpy.average。

  WEIGHTS 缺省（不传或传 NIL）→ 等权平均，等价 numpy 的 `weights=None`，
  即 `(vt-average a)` ≡ `(vt-mean a)`。此时直接走均值内核，不额外构造
  全 1 权重张量（v0.4.2 对齐：原实现把 weights 定为必填，`(vt-average a)`
  会报 invalid number of arguments，与 numpy 的 `weights=None` 不符）。

  为什么用 &REST 而不是 &OPTIONAL：`&optional` 紧跟 `&key` 时，CL 规定
  关键字实参可能被可选参数吞掉 —— `(vt-average a :axis 0)` 会把 `:axis`
  当成 WEIGHTS、再把 `0` 当成游离的关键字而报
  「odd number of &KEY arguments」。用 &REST 先摘掉位置 weights 再转发，
  两种调用形态都成立：
     (vt-average a)                   等权
     (vt-average a w)                 位置 weights（历史 API）
     (vt-average a :axis 0)           等权 + 轴
     (vt-average a w :axis 0)         位置 weights + 轴
  注：vt-params-audit 会把本函数标为「用 &REST 吞参数」，这是上述兼容的
  有意代价，不是 §6.2 的疏漏。

  dtype 语义（§4.2 H3「严格相等」）：
    结果 dtype 由「输入提升 + 显式 :dtype」决定，**不由 out 决定**。
    out 的 dtype 必须精确等于结果 dtype，否则报错。
    numpy 的 average 对整型输入会提升到 float64（本库一致）。"
  (let ((weights nil))
    (when (and args (not (keywordp (first args))))
      (setf weights (pop args)))
    ;; 复数分派：原先 complex.lisp 用同名 defun 覆盖本函数并转发
    ;; 回来（%cx-orig），重载本文件就会把复数静默送进实数内核。
    ;;
    ;; 语义保持既有契约（CONVENTIONS §16.11）：无参数的复数 average 就是
    ;; vt-mean（vt-mean 本身支持复数，走 %average-impl 的「等权」分支）；
    ;; 带 weights/axis 的复数路径尚未验收，显式报错而不是猜。
    (when (and (%cx-any-p tensor) args)
      (error "vt-average: 复数路径不支持参数 ~a（CONVENTIONS §16.11）" args))
    (apply #'%average-impl tensor weights args)))

(defun %average-impl (tensor weights &key axis keepdims dtype out)
  "vt-average 的实现体（weights 已由外层归一：NIL 表示等权）。"
  ;; weights 缺省 → 等权（numpy: weights=None）。走 vt-mean 既省一次全 1 张量
  ;; 与一次逐元素乘法，又天然继承 axis/keepdims/dtype/out 的同一套契约。
  (when (null weights)
    (return-from %average-impl
      (vt-mean tensor :axis axis :keepdims keepdims :dtype dtype :out out)))
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
             (need-promote (member in-dtype '(:bool :int32 :int64 :int16 :int8
                                             :uint8 :uint16)))
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
  (when (%cx-any-p tensor)
    (return-from vt-mean (%cx-op-mean tensor :axis axis :keepdims keepdims :dtype dtype :out out)))
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
  (when (%cx-any-p tensor)
    (return-from vt-var (%cx-op-var tensor :axis axis :keepdims keepdims :ddof ddof :dtype dtype :out out)))
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
          ((null axes)
           ;; 全归约：两趟单趟扫描（sum → mean → Σ(x-mean)²），而不是
           ;; mean + sub + square + sum + div 五趟四临时。1M double 实测
           ;; 6.45 → 1.8 ms；复数侧 %CX-VAR-GLOBAL 是同一做法。
           (let* ((flat (%nan-prepare tensor exec-dtype))
                  ;; WITH-FLOAT-SAFE 必须包住：`(%NAN-P X)` 是 `(not (= x x))`，
                  ;; 而 SBCL 的 UCOMISD 遇 NaN 会置 invalid 并**触发陷阱**。
                  (total (with-float-safe (vt-item (vt-sum flat :dtype exec-dtype)))))
             (funcall emit
                      (if (with-float-safe (%nan-p total))
                          (vt-full out-shape (vt-get-nan exec-dtype) :dtype exec-dtype)
                          (let* ((nf (coerce n (if (eq exec-dtype :float32)
                                                   'single-float 'double-float)))
                                 (mean (/ total nf))
                                 (ss (%sumsq-all flat exec-dtype mean)))
                            (vt-/ (vt-full out-shape ss :dtype exec-dtype)
                                  divisor :dtype exec-dtype))))))
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
  (when (%cx-any-p tensor)
    (return-from vt-std (%cx-op-std tensor :axis axis :keepdims keepdims :ddof ddof :dtype dtype :out out)))
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

(defun %cumulative-result-dtype (tensor dtype)
  "累积类算子的结果 dtype：显式 DTYPE 优先，否则按提升表推导。
   累积是**会溢出**的算子（第 i 个结果是前 i 个元素的和），所以不能沿用输入
   的窄类型：

   | 输入 dtype | 结果 dtype |
   |---|---|
   | `:bool` `:int8` `:uint8` `:int16` `:uint16` `:int32` | `:int64` |
   | 其余（`:int64` / 浮点 / 复数） | 同输入 |

   `:bool` 归进窄整型一档的原因：`cumsum(bool)` 在 numpy 里是 `int64` 的
   **计数**（`[T F T T]` → `[1 1 2 3]`），不是逻辑或。若沿用 `:bool`，
   `2`/`3` 会写进 `(unsigned-byte 8)` 的布尔数组 —— 既破 0/1 不变量，
   又让 `(vt-sum (vt-cumsum b))` 变成 7 而不是 4。

   §17.10.12：这张表原先只写在 `%VT-CUMULATIVE` 里，`nancumsum`/`nancumprod`
   走了另一条内核（`%E3-CUMULATIVE-SKIP-NAN`）自带一份
   `(or dtype (vt-dtype input))`，于是同一族里 `cumsum` 提升、
   `nancumsum` 不提升 —— `(nancumsum int8 #(100 100))` 静默回绕成 -56。"
  (or dtype
      (case (vt-dtype tensor)
        ((:bool :int8 :uint8 :int16 :uint16 :int32) :int64)
        (t (vt-dtype tensor)))))

(defun %vt-cumulative (tensor op init-val &key axis dtype out)
  "VT-CUMULATIVE 的内核（不含浮点陷阱屏蔽）；语义与契约见下方的公开入口。"
  ;; 别名保护（§4.3 H6）：累积是**流式**读写 —— 读到第 i 个就写第 i 个位置。
  ;; 当 out 与输入在物理区间上重叠（如 out = base[1:7]、in = base[0:6]）时，
  ;; 先写的输出会污染后面还没读到的输入，实测 `(vt-cumsum in :out out)`
  ;; 得到 1,2,4,8,16,32 这种「越滚越大」的错值。这里统一走 vt-out-snapshot，
  ;; 与 vt-map / SIMD matmul 的口径一致（§16.35.3）。
  (let* ((tensor (if out (first (vt-out-snapshot out (list tensor))) tensor))
         (shape (vt-shape tensor))
         (rank (length shape))
         ;; 结果 dtype 由「输入 dtype + 显式 :dtype」决定，不由 out 决定。
         ;; v0.4.x 修复：numpy cumsum/cumprod 对精度低于平台整型的整型
         ;; 输入提升为 int64（np.cumsum(int8) → int64），原实现保持 int8
         ;; 导致 100+100 溢出回绕为 -56。提升表见 %CUMULATIVE-RESULT-DTYPE
         ;; （与 %op-out-dtype 的 sum/prod 一致）。
         (final-dtype (%cumulative-result-dtype tensor dtype))
         (lisp-type (vt-dtype->lisp-type final-dtype))
         ;; 结果形状（逻辑契约）：numpy 在 axis=None 时先展平再累积。
         ;; 原实现沿用输入形状，且 axis=nil 的递归遍历在 rank>=2 上直接
         ;; 信号 "The value NIL is not of type NUMBER"（实测原版
         ;; (vt-cumsum 2x2) 必崩）。按 CONVENTIONS §0「与 NumPy 冲突以
         ;; NumPy 为准」改为展平形状，并去掉该递归。
         (res-shape (if axis shape (list (vt-size tensor))))
         ;; 归一化后的轴（nil 表示展平）。提前算：快路径与通用路径共用，
         ;; 也保证「轴越界/重复」在每个路径上都是同一个错误时机。
         (ax (when axis (vt-normalize-axis axis rank)))
         ;; out 给出时统一入口校验（形状/dtype 严格相等/可写性）
         (result (if out
                     (vt-check-out out res-shape final-dtype :op-name "vt-cumulative")
                     (vt-zeros res-shape :dtype final-dtype)))
         (in-data (vt-data tensor)) (out-data (vt-data result))
         (in-strides (vt-strides tensor)) (in-offset (vt-offset tensor))
         (out-strides (vt-strides result)) (out-offset (vt-offset result)))
    ;; ================================================================
    ;; 快路径（类型声明驱动）：op 是内建 +/* 且输入/结果的**元素类型相同**
    ;; 且该类型属于四种物理存储类型。
    ;;
    ;; 通用路径是「逐元素 funcall op + 逐元素 vt-cast」，实测 1M 元素沿轴
    ;; 需 196ms（约 196ns/元素，比 numpy 慢 59 倍），瓶颈全部来自没有类型
    ;; 信息：
    ;;   · aref 对未声明元素类型的数组走泛型分派（慢 ~20x）
    ;;   · (funcall op ...) 每次调用都要装箱
    ;;   · (vt-cast cum final-dtype) 走 ecase + 泛型算术
    ;; 这里把「数组元素类型」与「累加器类型」同时声明，并把 op 提到循环外
    ;; 做一次 eq 分派，让 +/* 直接编译成机器指令。
    ;;
    ;; 关键：**平面与沿轴共用同一个 RUN**（区别只在步长 1 与 stride）。
    ;; 只给 axis=nil 特化、不管沿轴，会让同一个函数在
    ;; `(vt-cumsum a)` 与 `(vt-cumsum a :axis 0)` 上差两个数量级（实测 196×）。
    ;; 结果与通用路径逐位相同（同为顺序累加，铁律 1/2）。
    ;; ================================================================
    (macrolet ((cum-typed (lt)
                 ;; 存储表达式按元素类型在**编译期**选定：整数用 dtype.lisp 中
                 ;; 已声明返回类型的 %coerce-* 做回绕收窄，浮点直接存。
                 (let ((store (cond ((equal lt '(signed-byte 64)) `(%coerce-int64 acc))
                                    ((equal lt '(signed-byte 32)) `(%coerce-int32 acc))
                                    (t 'acc))))
                   `(labels ((run (sp dp cnt istep ostep)
                             ;; 从 SP 起沿 ISTEP 走 CNT 个，写到 DP 起沿 OSTEP。
                             ;; 首元素**直接拷贝**，从第二个起才做 OP：
                             ;; numpy 的 accumulate 就是这个语义（§16.30.3）——
                             ;;   np.cumsum([-0.0]) => [-0.]  而 0.0 + (-0.0) = +0.0
                             ;; 「从单位元起算」会静默洗掉 -0.0 的符号位，
                             ;; cumprod 还会在 inf 首元素上凭空造出 0*inf。
                             (declare (type fixnum sp dp cnt istep ostep)
                                      (optimize (speed 3) (safety 1)))
                             (let ((src (the (simple-array ,lt (*)) in-data))
                                   (dst (the (simple-array ,lt (*)) out-data)))
                               (declare (type (simple-array ,lt (*)) src dst))
                               (when (plusp cnt)
                                 (let ((acc (the ,lt (aref src sp))))
                                   (declare (type ,lt acc))
                                   (setf (aref dst dp) acc)
                                   (if (eq op #'+)
                                       (loop for si of-type fixnum = (+ sp istep) then (+ si istep)
                                             for di of-type fixnum = (+ dp ostep) then (+ di ostep)
                                             for i of-type fixnum from 1 below cnt
                                             do (setf acc (+ acc (aref src si)))
                                                (setf (aref dst di) ,store))
                                       (loop for si of-type fixnum = (+ sp istep) then (+ si istep)
                                             for di of-type fixnum = (+ dp ostep) then (+ di ostep)
                                             for i of-type fixnum from 1 below cnt
                                             do (setf acc (* acc (aref src si)))
                                                (setf (aref dst di) ,store)))))))
                           (run-flat ()
                             ;; 展平：整块一次顺序扫描（已保证输入/结果连续）
                             (run in-offset out-offset (vt-size result) 1 1))
                           (run-axis ()
                             ;; 沿轴：除 AX 外各维做里程计，每条 fiber 只算一次
                             ;; 起点；内层整段交给 RUN（步长 = 该轴 stride，可为负）。
                             (let ((dim (the fixnum (nth ax shape)))
                                   (istep (the fixnum (nth ax in-strides)))
                                   (ostep (the fixnum (nth ax out-strides)))
                                   (idx (make-array (max 1 rank)
                                                    :element-type 'fixnum
                                                    :initial-element 0)))
                               (declare (type (simple-array fixnum (*)) idx))
                               (loop
                                 (let ((sp in-offset) (dp out-offset))
                                   (dotimes (d rank)
                                     (unless (= d ax)
                                       (incf sp (* (aref idx d) (nth d in-strides)))
                                       (incf dp (* (aref idx d) (nth d out-strides)))))
                                   (run sp dp dim istep ostep))
                                 ;; 里程计：跳过 AX 维，逐位向左进位
                                 (let ((carry t))
                                   (loop for d of-type fixnum from (1- rank) downto 0
                                         while carry
                                         do (unless (= d ax)
                                              (incf (aref idx d))
                                              (if (< (aref idx d) (nth d shape))
                                                  (setf carry nil)
                                                  (setf (aref idx d) 0))))
                                   (when carry (return)))))))
                     (if ax (run-axis) (run-flat))))))
      (when (and (member op (list #'+ #'*))
                 (equal (array-element-type in-data)
                        (array-element-type out-data))
                 (member (array-element-type in-data)
                         (list 'double-float 'single-float
                               '(signed-byte 64) '(signed-byte 32))
                         :test #'equal)
                 ;; axis=nil 时要求全连续（展平后才是行主序）；沿轴时
                 ;; 走 strides，任意布局都成立，无需连续。
                 (or ax
                     (and (vt-contiguous-p tensor)
                          (vt-contiguous-p result))))
        (let ((et (array-element-type in-data)))
          (cond ((equal et 'double-float)     (cum-typed double-float))
                ((equal et 'single-float)     (cum-typed single-float))
                ((equal et '(signed-byte 64)) (cum-typed (signed-byte 64)))
                ((equal et '(signed-byte 32)) (cum-typed (signed-byte 32)))
                (t nil)))
        (return-from %vt-cumulative result)))
    (if ax
        ;; ---------------- 沿指定轴累积（形状不变）----------------
        ;; 通用回退路径：dtype 不在四种存储类型内、或输入/结果元素类型不同
        ;; （如 int8→int64 提升、显式 :dtype 改变宽度）。语义正确但慢
        ;; （逐元素 funcall + vt-cast），正常情况都被上面的特化路径接走。
        (let* ((ax-dim (nth ax shape))
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
                  ;; 首元素直拷（numpy accumulate 语义，§16.30.3）：
                  ;; 每条 fiber 的第一个输出 = 输入本身，不是 OP(单位元, 输入)。
                  (when (plusp ax-dim)
                    (let ((v (aref in-data in-ptr)))
                      (setf cum v)
                      (setf (aref out-data out-ptr) (vt-cast v final-dtype))))
                  (loop for i from 1 below ax-dim do
                    (setf cum (funcall op cum (aref in-data (+ in-ptr (* i in-stride)))))
                    (setf (aref out-data (+ out-ptr (* i out-stride)))
                          (vt-cast cum final-dtype)))))
              (unless (advance) (return)))))
        ;; ---------------- 全局累积（axis=nil）：展平后线性累加 ----------------
        ;; 原实现是「递归遍历 + funcall op + 逐元素 vt-cast」：rank>=2 时
        ;; 直接崩，且 1M 元素耗时 227ms。此处改为对展平视图的一次线性扫描，
        ;; 语义与 numpy 的 flat 累积一致；非常规 dtype 仍走通用 funcall/cast
        ;; 兜底（结果正确，仅慢）。
        (let* ((flat (vt-ravel tensor))
               (src (vt-data flat))
               (soff (the fixnum (vt-offset flat)))
               (dst (vt-data result))
               (n (the fixnum (vt-size result)))
               (doff (the fixnum (vt-offset result)))
               (conv (vt-cast-fun final-dtype))
               (cum (coerce init-val lisp-type)))
          (declare (type fixnum soff n doff))
          ;; 首元素直拷：cumsum([-0.0]) 必须是 -0.0（numpy 实测），
          ;; 而 0.0 + (-0.0) = +0.0 会把符号位洗掉。§16.30.3
          (when (plusp n)
            (let ((v (aref src soff)))
              (setf cum v)
              (setf (aref dst doff) (funcall conv v))))
          (loop for i of-type fixnum from 1 below n
                do (setf cum (funcall op cum (aref src (+ soff i))))
                   (setf (aref dst (+ doff i)) (funcall conv cum)))))
    result))

(defun vt-cumulative (tensor op init-val &key axis dtype out)
  "累积归约核心原语：沿 AXIS 依次累积 OP（#\+ / #\*），以 INIT-VAL 起始。
  dtype 语义（§4.2 H3）：结果 dtype = 显式 :dtype，否则为输入 dtype；out 必须精确匹配。

  结果形状（v0.4.1 逻辑修复，对标 numpy）：
    axis=NIL → **先展平再累积**，返回一维（np.cumsum([[1,2],[3,4]]) -> (4,)）；
    axis 给定 → 形状与输入相同，仅该轴被累积。

  首元素直拷（§16.30.3）：numpy 的 accumulate 不是「从单位元起算」，
  首个输出等于输入本身 —— `np.cumsum([-0.0])` 给 `-0.0`，而 `0.0 + (-0.0)` 给
  `+0.0`。符号零是语义信息（`1/x` → ∓Inf、`log` 的分支切割），不能静默洗掉。

  NaN/Inf（§5.5 / §7）：外壳把内核包在 WITH-FLOAT-SAFE 里。SBCL 默认四类浮点
  陷阱全开，`inf + (-inf)`、`inf * 0` 会直接抛 FLOATING-POINT-INVALID-OPERATION，
  而 numpy 只是给出 nan 并继续（`np.cumprod([inf, 0])` → `[inf, nan]`）。
  用「外壳 + 内核」两层而不是把 150 行循环体整体缩进，是为了让 diff 只反映语义变更。"
  (with-float-safe
   (%vt-cumulative tensor op init-val :axis axis :dtype dtype :out out)))

(defun vt-cumsum (tensor &key axis dtype out)
  "沿 AXIS 的累积和（axis 缺省全展平）。等价 (vt-cumulative tensor #\+ 0 :axis axis)。"
  (when (%cx-any-p tensor)
    (return-from vt-cumsum (%cx-op-cumsum tensor :axis axis :dtype dtype :out out)))
  (vt-cumulative tensor #'+ 0 :axis axis :dtype dtype :out out))

(defun vt-cumprod (tensor &key axis dtype out)
  "沿 AXIS 的累积积（axis 缺省全展平）。等价 (vt-cumulative tensor #\* 1 :axis axis)。"
  (when (%cx-any-p tensor)
    (return-from vt-cumprod (%cx-op-cumprod tensor :axis axis :dtype dtype :out out)))
  (vt-cumulative tensor #'* 1 :axis axis :dtype dtype :out out))

;;; ------------------------------------------------------------------
;;; 中位数 / 百分位 / 直方图
;;; ------------------------------------------------------------------
;;; ------------------------------------------------------------------
;;; 连续 fiber 的原地升序排序内核（vt-sort / vt-median / vt-percentile 共用）
;;; 定义位置提前到 vt-median 之前，避免调用方前向引用产生编译告警。
;;; ------------------------------------------------------------------
;;; ------------------------------------------------------------------
;;; 连续 fiber 的原地升序排序（vt-sort 快路径的引擎）
;;;
;;; 为什么单独写引擎：通用路径把每个 fiber 摊成 **list** 再 stable-sort，
;;; 1M 元素要 cons 1M 个 cell、再做指针追逐式归并 —— 实测 520ms
;;; （numpy.sort 5.5ms）。而 SBCL 通用 SORT 作用在 specialized-array 上
;;; 反而更慢（1057ms，因为它走泛型序列协议）。所以这里按元素类型各生成
;;; 一份「数组 + 下标」的内核：
;;;   · 长度 ≤ 32 → 插入排序（小 fiber 免去 scratch 开销，典型场景是
;;;     vt-median / vt-sort 沿轴排序长度几十的 fiber）
;;;   · 更长 → 自底向上稳定归并（顺序访存、无 cons）
;;; 语义与通用路径逐位一致：升序；NaN 统一排在各 fiber 末尾，且保持
;;; 原有相对顺序（稳定压缩 + 尾部回填）。铁律 1/2 不变。
;;; ------------------------------------------------------------------
(macrolet ((def-sorters (sorter msort lt floatp)
             `(progn
                (defun ,msort (v base aux n)
                  "对 V 的 [BASE, BASE+N) 段做自底向上稳定归并排序，AUX 为等长 scratch。"
                  (declare (type (simple-array ,lt (*)) v aux)
                           (type fixnum base n))
                  (let ((w 1))
                    (declare (type fixnum w))
                    (loop while (< w n) do
                      (let ((lo 0))
                        (declare (type fixnum lo))
                        (loop while (< lo n) do
                          (let* ((mid (min (+ lo w) n))
                                 (hi (min (+ lo w w) n))
                                 (i lo) (j mid) (k lo))
                            (declare (type fixnum mid hi i j k))
                            (loop while (and (< i mid) (< j hi)) do
                              (if (< (aref v (+ base j)) (aref v (+ base i)))
                                  (progn (setf (aref aux k) (aref v (+ base j))) (incf j))
                                  (progn (setf (aref aux k) (aref v (+ base i))) (incf i)))
                              (incf k))
                            (loop while (< i mid) do
                              (setf (aref aux k) (aref v (+ base i))) (incf i) (incf k))
                            (loop while (< j hi) do
                              (setf (aref aux k) (aref v (+ base j))) (incf j) (incf k))
                            (replace v aux :start1 (+ base lo) :end1 (+ base hi)
                                       :start2 lo :end2 hi))
                          (incf lo (* 2 w))))
                      (setf w (* 2 w)))))
                (defun ,sorter (data n count)
                  "把 DATA 中 COUNT 段、每段长度 N 的连续 fiber 各自原地升序排序。"
                  (declare (type (simple-array ,lt (*)) data)
                           (type fixnum n count))
                  (with-float-safe
                    (let ((aux (make-array (max n 1) :element-type ',lt))
                          ,@(when floatp
                              `((nanv (make-array (max n 1) :element-type ',lt)))))
                      (declare (type (simple-array ,lt (*)) aux))
                      (dotimes (c count)
                        (let* ((base (* c n))
                               (k n)
                               ,@(when floatp `((m 0))))
                          (declare (type fixnum base k))
                          ,@(when floatp
                              `((declare (type fixnum m))
                                ;; 稳定压缩：非 NaN 依次前移，NaN 收集备用
                                (let ((w 0))
                                  (declare (type fixnum w))
                                  (dotimes (i n)
                                    (let ((x (aref data (+ base i))))
                                      (if (not (= x x))
                                          (progn (setf (aref nanv m) x) (incf m))
                                          (progn (setf (aref data (+ base w)) x)
                                                 (incf w)))))
                                  (setf k w))))
                          (when (> k 1)
                            (if (<= k 32)
                                ;; ---- 插入排序（小 fiber）----
                                (dotimes (ii (1- k))
                                  (let* ((i (1+ ii))
                                         (x (aref data (+ base i)))
                                         (j (1- i)))
                                    (declare (type fixnum i j))
                                    (loop while (and (>= j 0)
                                                     (< x (aref data (+ base j)))) do
                                      (setf (aref data (+ base (1+ j))) (aref data (+ base j)))
                                      (decf j))
                                    (setf (aref data (+ base (1+ j))) x)))
                                ;; ---- 归并排序（大 fiber）----
                                (progn (,msort data base aux k)
                                       (replace data aux
                                                :start1 base :end1 (+ base k)
                                                :start2 0 :end2 k))))
                          ,@(when floatp
                              `((dotimes (i m)
                                  (setf (aref data (+ base k i)) (aref nanv i)))))))))))))
  (def-sorters %sort-fibers-df %msort-df double-float t)
  (def-sorters %sort-fibers-sf %msort-sf single-float t)
  (def-sorters %sort-fibers-i64 %msort-i64 (signed-byte 64) nil)
  (def-sorters %sort-fibers-i32 %msort-i32 (signed-byte 32) nil))

(defun %sort-fibers (data n count)
  "按 DATA 的实际元素类型分派到对应的 fiber 排序内核；其余类型返回 NIL
   （调用方据此回退通用 list 路径，铁律 2）。"
  (declare (type fixnum n count))
  (let ((et (array-element-type data)))
    (cond ((equal et 'double-float)     (%sort-fibers-df data n count) t)
          ((equal et 'single-float)     (%sort-fibers-sf data n count) t)
          ((equal et '(signed-byte 64)) (%sort-fibers-i64 data n count) t)
          ((equal et '(signed-byte 32)) (%sort-fibers-i32 data n count) t)
          (t nil))))


;;; ------------------------------------------------------------------
;;; 顺序统计量的公共取数层（median / percentile / quantile / nanmedian 共用）
;;;
;;; v0.4.3 性能修复：这四个函数原来统一
;;;     (loop for i below n collect (aref data i))   → list
;;;     (vt-numpy-sort list #'<)
;;; 而 vt-numpy-sort 内部又 (coerce sequence 'list)，然后按 NaN 拆两个 list、
;;; append、stable-sort —— 1M 元素要 cons 两遍、再做指针追逐式归并。
;;; 实测 median 773ms / percentile 709ms / nanmedian 644ms
;;; （numpy 分别 10.0ms / 12.3ms / 10.2ms，差 60–77 倍）。
;;;
;;; 改用「专用数组 + %sort-fibers」：一次 replace + 原地稳定归并，零 cons，
;;; 之后按下标直接取顺序统计量。元素类型不在支持集内时仍回退 list 路径
;;; （语义完全相同）。
;;;
;;; 顺序保证：升序，NaN 稳定压到末尾 —— 因此「末尾元素是 NaN」⇔「存在
;;; NaN」，检查一个元素即可判定，无需再 O(n) 扫一遍。
;;; ------------------------------------------------------------------

(defun %sortable-element-type-p (et)
  (member et '(double-float single-float (signed-byte 64) (signed-byte 32))
          :test #'equal))

;;; ==================================================================
;;; 顺序统计（quickselect）——median / percentile / nanpercentile 共用
;;;
;;; 原实现为了取一个分位点，把整份数据**全排序**（1M double 约 133ms，
;;; numpy 用 introselect 只要 12–19ms）。这里补一个 O(N) 的选择内核：
;;;   1. %NAN-FREE-DOUBLE-BUFFER：一趟扫描把非 NaN 元素收进 double 数组，
;;;      同时记 NaN 个数（8 路元素类型特化）；
;;;   2. %SELECT-DOUBLE：Lomuto 分区 + 三数取中，迭代式 quickselect；
;;;   3. %PERCENT-DOUBLE-BUFFER：在未排序缓冲上按 numpy 的位置公式取分位。
;;; 语义与「排序后按下标取」等价（同多重集上的第 K 小唯一），
;;; 差别只在浮点插值的计算顺序：linear 情形与 %PERCENT-FROM-SORTED
;;; 完全相同的算式与实参类型，故结果逐位一致。
;;; ==================================================================

(defun %nan-free-double-buffer (tensor)
  "把 TENSOR 的非 NaN 元素收进 double 数组（返回 (values BUF M HAS-NAN-P)）。
   BUF 长度 = 张量元素总数，只有前 M 个有效。"
  (let* ((c (vt-contiguous (vt-flatten tensor)))
         (size (vt-size c))
         (buf (make-array size :element-type 'double-float))
         (m 0)
         (nans 0))
    (declare (type fixnum size m nans)
             (type (simple-array double-float (*)) buf))
    (macrolet ((scan (lt)
                 `(let ((d (the (simple-array ,lt (*)) (vt-data c)))
                        (off (vt-offset c)))
                    (declare (type (simple-array ,lt (*)) d)
                             (type fixnum off))
                    (dotimes (i size)
                      (declare (type fixnum i))
                      (let ((x (coerce (aref d (the fixnum (+ off i))) 'double-float)))
                        (if (/= x x)
                            (incf nans)
                            (progn (setf (aref buf m) x) (incf m))))))))
      (with-float-safe
        (let ((et (array-element-type (vt-data c))))
          (cond ((equal et 'double-float) (scan double-float))
                ((equal et 'single-float) (scan single-float))
                ((equal et '(signed-byte 64)) (scan (signed-byte 64)))
                ((equal et '(signed-byte 32)) (scan (signed-byte 32)))
                ((equal et '(signed-byte 16)) (scan (signed-byte 16)))
                ((equal et '(signed-byte 8)) (scan (signed-byte 8)))
                ((equal et '(unsigned-byte 16)) (scan (unsigned-byte 16)))
                ((equal et '(unsigned-byte 8)) (scan (unsigned-byte 8)))
                (t (dotimes (i size)
                     (declare (type fixnum i))
                     (let ((x (coerce (aref (vt-data c) (the fixnum (+ (vt-offset c) i)))
                                      'double-float)))
                       (if (/= x x)
                           (incf nans)
                           (progn (setf (aref buf m) x) (incf m))))))))))
    (values buf m (plusp nans))))

(defun %select-double (v n k)
  "返回 V[0,N) 中第 K 小（0 基）的值；V 会被就地重排。
   三分区（<, =, >）+ 三数取中，迭代实现（无递归、无额外分配）。

   两条护栏，各自对应一类**实测过**的真实输入：

   1. **必须三分区**（不能用 Lomuto 两分区）：两分区在「大量重复值」上退化
      —— 每趟只剥掉 pivot 那一个元素。400k 全等：两分区 20.3 s、三分区
      0 ms；7 个值循环：364 ms → 2 ms。三分区一趟就把等于 pivot 的整段
      [LT, GT] 剥掉，命中即返回，全等输入 O(n) 结束。

   2. **必须设访问量上限**：三分区在三数取中的坏排列上仍会退化 —— 400k
      **整体降序**且 K = N/2-1 时，每趟只削掉极窄一段：451 趟、30.2M 次
      访问、142 ms（同一数组 K = N/2 只要 1 趟 2 ms）。累计访问超过 4N 就
      改用库内归并内核 %SORT-FIBERS 排完整段再取第 K 个，把最坏情况钉在
      O(n log n)：实测该用例 142 ms → 15 ms，而常见路径只多一次 INCF/元素。"
  (declare (type (simple-array double-float (*)) v)
           (type fixnum n k))
  (labels ((swp (a b)
             (declare (type fixnum a b))
             (let ((tmp (aref v a))) (setf (aref v a) (aref v b) (aref v b) tmp))))
    (let ((lo 0)
          (hi (the fixnum (1- n)))
          (visits 0)
          (limit (the fixnum (* 4 n))))
      (declare (type fixnum lo hi visits limit))
      (loop
        (when (>= lo hi) (return (aref v lo)))
        (let ((mid (the fixnum (floor (+ lo hi) 2))))
          (declare (type fixnum mid))
          (when (> (aref v mid) (aref v hi)) (swp mid hi))
          (when (> (aref v lo) (aref v hi)) (swp lo hi))
          (when (> (aref v mid) (aref v lo)) (swp mid lo)))
        (let ((pivot (aref v lo))
              (lt lo)                 ; [lo, lt)  < pivot
              (gt hi)                 ; (gt, hi]  > pivot
              (i lo))                 ; 游标
          (declare (type fixnum lt gt i) (type double-float pivot))
          (loop
            (when (> i gt) (return))
            (incf visits)
            (cond ((< (aref v i) pivot) (swp lt i) (incf lt) (incf i))
                  ((> (aref v i) pivot) (swp i gt) (decf gt))
                  (t (incf i))))
          ;; 退化护栏：整段排序后直接取第 K 个（缓冲区仍含全部 N 个元素）
          (when (> visits limit)
            (%sort-fibers v n 1)
            (return (aref v k)))
          (cond ((< k lt) (setf hi (the fixnum (1- lt))))
                ((> k gt) (setf lo (the fixnum (1+ gt))))
                (t (return pivot))))))))

(defun %percent-double-buffer (buf m q interpolation)
  "在未排序缓冲 BUF[0,M) 上按 numpy 位置公式取 Q 分位点（就地重排 BUF）。"
  (declare (type (simple-array double-float (*)) buf) (type fixnum m)
           (type double-float q))
  (let* ((idx (* q (1- m)))
         (lower (floor idx))
         (upper (min (ceiling idx) (1- m)))
         (frac (- idx lower)))
    (declare (type fixnum lower upper) (type double-float frac))
    (case interpolation
      (:linear (if (= lower upper)
                   (%select-double buf m lower)
                   (+ (* (- 1.0d0 frac) (%select-double buf m lower))
                      (* frac (%select-double buf m upper)))))
      (:lower (%select-double buf m lower))
      (:higher (%select-double buf m upper))
      (:midpoint (/ (+ (%select-double buf m lower)
                       (%select-double buf m upper))
                    2.0d0))
      (:nearest (%select-double buf m (if (<= frac 0.5d0) lower upper))))))

(defun %sorted-values (tensor)
  "把 TENSOR 的全部元素升序排序（NaN 在末尾）并返回 (values SEQ N)。
   SEQ 为专用数组（快路径）或 list（兜底），二者都支持 ELT / LENGTH。"
  (let* ((c (vt-contiguous (vt-flatten tensor)))
         (n (vt-size c))
         (et (array-element-type (vt-data c))))
    (if (or (zerop n) (not (%sortable-element-type-p et)))
        (with-float-safe
          (values (if (zerop n)
                      '()
                      (vt-numpy-sort (coerce (vt-data c) 'list) #'<))
                  n))
        (let ((buf (make-array n :element-type et)))
          (replace buf (vt-data c))
          (with-float-safe (%sort-fibers buf n 1))
          (values buf n)))))

;;; ------------------------------------------------------------------
;;; 间接排序（vt-argsort 的引擎）
;;;
;;; v0.4.4 性能修复：vt-argsort 原来把每个 fiber 摊成 (cons 值 . 下标) 的
;;; **list**，再 (stable-sort list #'< :key #'car)。1M 元素 = 1M 个 cons +
;;; 指针追逐式归并，实测 146ms（numpy.argsort 19.7ms，差 7.4 倍）。
;;;
;;; 改为「值拷贝到专用数组 + 下标数组间接归并」：
;;;   · 比较走 (aref vals (aref idx j))，值数组有元素类型声明 → 标量比较；
;;;   · NaN 先用一趟 O(n) **稳定分区**挪到末尾，于是比较器里只剩一条 <
;;;     （否则每次比较都要两次 FLOATP + 两次 (not (= x x))，成本翻倍）；
;;;   · 长度 ≤ 32 用插入排序，免去 scratch 分配（沿轴 argsort 的 fiber
;;;     经常只有几十个元素）。
;;; 语义与旧实现逐位一致：升序、稳定、NaN 压下且保持原相对顺序。
;;; ------------------------------------------------------------------
(macrolet ((def-argsorter (name lt)
             `(progn
                (defun ,name (vals idx n)
                  "对 IDX[0,N) 按 VALS 升序稳定排序（间接归并，就地改 IDX）。"
                  (declare (type (simple-array ,lt (*)) vals)
                           (type (simple-array (signed-byte 64) (*)) idx)
                           (type fixnum n))
                  (cond
                    ((<= n 1) nil)
                    ((<= n 32)
                     ;; ---- 插入排序（小 fiber）----
                     (dotimes (ii (1- n))
                       (let* ((i (1+ ii))
                              (x (aref idx i))
                              (xv (aref vals x))
                              (j (1- i)))
                         (declare (type fixnum i j) (type ,lt xv))
                         (loop while (and (>= j 0)
                                          (< xv (aref vals (aref idx j))))
                               do (setf (aref idx (1+ j)) (aref idx j))
                                  (decf j))
                         (setf (aref idx (1+ j)) x))))
                    (t
                     ;; ---- 自底向上稳定归并 ----
                     (let ((aux (make-array n :element-type '(signed-byte 64)))
                           (w 1))
                       (declare (type (simple-array (signed-byte 64) (*)) aux)
                                (type fixnum w))
                       (loop while (< w n) do
                         (let ((lo 0))
                           (declare (type fixnum lo))
                           (loop while (< lo n) do
                             (let* ((mid (min (+ lo w) n))
                                    (hi (min (+ lo w w) n))
                                    (i lo) (j mid) (k lo))
                               (declare (type fixnum mid hi i j k))
                               (loop while (and (< i mid) (< j hi)) do
                                 (if (< (aref vals (aref idx j))
                                        (aref vals (aref idx i)))
                                     (progn (setf (aref aux k) (aref idx j))
                                            (incf j))
                                     (progn (setf (aref aux k) (aref idx i))
                                            (incf i)))
                                 (incf k))
                               (loop while (< i mid) do
                                 (setf (aref aux k) (aref idx i))
                                 (incf i) (incf k))
                               (loop while (< j hi) do
                                 (setf (aref aux k) (aref idx j))
                                 (incf j) (incf k))
                               (replace idx aux :start1 lo :end1 hi
                                            :start2 lo :end2 hi))
                             (incf lo (* 2 w))))
                         (setf w (* 2 w))))))))))
  (def-argsorter %argsort-idx-df double-float)
  (def-argsorter %argsort-idx-sf single-float)
  (def-argsorter %argsort-idx-i64 (signed-byte 64))
  (def-argsorter %argsort-idx-i32 (signed-byte 32)))

(defun %argsort-dispatch (vals idx n)
  "按 VALS 的元素类型分派间接排序；不支持的类型返回 NIL。"
  (declare (type fixnum n))
  (let ((et (array-element-type vals)))
    (cond ((equal et 'double-float)     (%argsort-idx-df vals idx n) t)
          ((equal et 'single-float)     (%argsort-idx-sf vals idx n) t)
          ((equal et '(signed-byte 64)) (%argsort-idx-i64 vals idx n) t)
          ((equal et '(signed-byte 32)) (%argsort-idx-i32 vals idx n) t)
          (t nil))))

(defun %argsort-fiber (data base stride n)
  "对 DATA 中从 BASE 起、步长 STRIDE、长度 N 的 fiber 求升序下标数组
   （int64，**相对 fiber 起点**；NaN 稳定压末尾）。
   元素类型不在支持集内时返回 NIL，调用方据此回退通用 list 路径。"
  (declare (type fixnum base stride n))
  (let ((et (array-element-type data)))
    (unless (%sortable-element-type-p et)
      (return-from %argsort-fiber nil))
    (let ((vals (make-array n :element-type et))
          (idx  (make-array n :element-type '(signed-byte 64)))
          (nans (make-array n :element-type '(signed-byte 64)))
          (m 0) (tc 0))
      (declare (type fixnum m tc))
      (with-float-safe
        (if (= stride 1)
            (replace vals data :start2 base :end2 (the fixnum (+ base n)))
            (dotimes (i n)
              (declare (type fixnum i))
              (setf (aref vals i) (aref data (the fixnum (+ base (* i stride)))))))
        ;; 稳定分区：非 NaN 保序进 IDX 前缀，NaN 下标顺序收集到 NANS
        (dotimes (i n)
          (declare (type fixnum i))
          (let ((v (aref vals i)))
            (if (and (floatp v) (%nan-p v))
                (progn (setf (aref nans tc) i) (incf tc))
                (progn (setf (aref idx m) i) (incf m)))))
        (when (> m 1)
          (%argsort-dispatch vals idx m))
        (when (plusp tc)
          (replace idx nans :start1 m :start2 0 :end2 tc)))
      idx)))

(defun %sorted-has-nan-p (seq n)
  "升序序列的末尾是 NaN ⇔ 序列中存在 NaN。"
  (and (plusp n)
       (let ((w (elt seq (1- n))))
         (and (floatp w) (%nan-p w)))))

(defun vt-median (tensor &key axis keepdims)
  "中位数（对标 numpy.median）。偶数个元素取中间两数平均。"
  (when (%cx-any-p tensor)
    (return-from vt-median (%cx-op-median tensor :axis axis :keepdims keepdims)))
  (with-float-safe
    (let ((nanv (vt-get-nan :float64)))
      (if axis
          ;; ---------------- 轴归约 ----------------
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
                          nanv
                          ;; v0.4.10：fiber 不再整根排序，改 quickselect。
                          ;; 1000×1000 沿轴归约 76ms → ~25ms。
                          (multiple-value-bind (buf m has-nan)
                              (%nan-free-double-buffer fiber)
                            (if (or has-nan (zerop m))
                                nanv
                                (let ((half (the fixnum (floor m 2))))
                                  (declare (type fixnum half))
                                  (if (oddp m)
                                      (%select-double buf m half)
                                      (/ (+ (%select-double buf m (1- half))
                                            (%select-double buf m half))
                                         2.0d0)))))))))
            result)
          ;; ---------------- 全局归约 ----------------
          (let* ((in-shape (vt-shape tensor))
                 (rank (length in-shape))
                 (out-shape (if keepdims
                                (make-list rank :initial-element 1)
                                nil)))
            (if (zerop (vt-size tensor))
                (make-vt out-shape nanv :dtype :float64)
                ;; v0.4.10：改用 quickselect 只取第 k 小，不再全排序 ——
                ;; 1M double 从 134ms 降到 ~20ms（numpy.median 12ms）。
                (multiple-value-bind (buf m has-nan) (%nan-free-double-buffer tensor)
                  (if has-nan
                      (make-vt out-shape nanv :dtype :float64)
                      (let ((half (the fixnum (floor m 2))))
                        (declare (type fixnum half))
                        (if (oddp m)
                            (make-vt out-shape (%select-double buf m half) :dtype :float64)
                            (make-vt out-shape
                                     (/ (+ (%select-double buf m (1- half))
                                           (%select-double buf m half))
                                        2.0d0)
                                     :dtype :float64)))))))))))

(defun %percent-from-sorted (sorted q interpolation)
  "从**已升序**的序列取 Q 分位点。SORTED 可以是专用数组（ELT = O(1)）或
   list（ELT 退化为线性，仅兜底路径）。v0.4.3：原来用 NTH，只能吃 list，
   迫使调用方把数据摊成 list —— 现统一用 ELT，可直接吃排序后的专用数组。"
  (let* ((n (length sorted))
         (idx (* q (1- n)))
         (lower (floor idx))
         (upper (min (ceiling idx) (1- n)))
         (frac (- idx lower)))
    (case interpolation
      (:linear (if (= lower upper) (vt-cast (elt sorted lower) :float64)
                   (+ (* (- 1 frac) (vt-cast (elt sorted lower) :float64))
                      (* frac (vt-cast (elt sorted upper) :float64)))))
      (:lower (vt-cast (elt sorted lower) :float64))
      (:higher (vt-cast (elt sorted upper) :float64))
      (:midpoint (/ (+ (vt-cast (elt sorted lower) :float64)
                       (vt-cast (elt sorted upper) :float64))
                    2.0d0))
      (:nearest (vt-cast (elt sorted (if (<= frac 0.5d0) lower upper)) :float64)))))

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
      (when (%cx-reject tensor percentile)
    (error "vt-percentile: 不支持复数输入（numpy 同样拒绝）。见 CONVENTIONS §16.11"))
(when (%cx-reject tensor percentile)
    (error "vt-percentile: 不支持复数输入（numpy 同样拒绝）。见 CONVENTIONS §16.11"))
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
                          ;; v0.4.x：归约区为空 → numpy ValueError 语义
                          (error "vt-percentile: zero-size array to reduction operation percentile which has no identity（numpy 语义）")
                          ;; v0.4.10：fiber 不再整根排序，改 quickselect。
                          (multiple-value-bind (buf m has-nan)
                              (%nan-free-double-buffer fiber)
                            (if (or has-nan (zerop m))
                                (vt-get-nan :float64)
                                (%percent-double-buffer buf m q interpolation)))))))
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
                ;; v0.4.x：numpy 对空数组 percentile 抛 ValueError
                ;; （zero-size array to reduction operation percentile
                ;;   which has no identity），不再返回 NaN。
                (error "vt-percentile: zero-size array to reduction operation percentile which has no identity（numpy 语义）")
                ;; v0.4.10：quickselect 取分位（不再全排序）。
                (multiple-value-bind (buf m has-nan) (%nan-free-double-buffer tensor)
                  (if has-nan
                      (make-vt out-shape (vt-get-nan :float64) :dtype :float64)
                      (make-vt out-shape
                               (%percent-double-buffer buf m q interpolation)
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
      (when (%cx-reject tensor q)
    (error "vt-quantile: 不支持复数输入（numpy 同样拒绝）。见 CONVENTIONS §16.11"))
(when (%cx-reject tensor q)
    (error "vt-quantile: 不支持复数输入（numpy 同样拒绝）。见 CONVENTIONS §16.11"))
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
  "峰谷差 max - min（对标 numpy.ptp）。

   布尔输入报错（CONVENTIONS §17.3、§17.10.13）：峰谷差本身就是一个减法，
   numpy 明确拒绝（`TypeError: numpy boolean subtract, the - operator, is
   not supported`）。原实现的 `:axis` 分支走 `vt--`（报的是正确的布尔
   减法消息），非 axis 分支写成 `(- (vt-item …) (vt-item …))`，而
   `(vt-item bool标量)` 自 §17.8 起返回 T/NIL —— 于是同一个函数在两个
   分支上报两种错：一个是「布尔减法未定义」，另一个是
   「The value T is not of type NUMBER」。后者是内部类型泄漏，用户既
   看不出是布尔的问题，也不知道该换成什么。"
  (let ((tv (ensure-vt tensor)))
    (when (eq (vt-dtype tv) :bool)
      (error "vt-ptp: 布尔张量不支持峰谷差（numpy 同样拒绝：布尔减法未定义）。 ~
              要计数请用 vt-count-nonzero，要看相邻差异请用 vt-diff（相邻异或）； ~
              见 CONVENTIONS §17.3"))
    (if axis
        (vt-- (vt-amax tv :axis axis) (vt-amin tv :axis axis))
        (- (vt-item (vt-amax tv)) (vt-item (vt-amin tv))))))

(defun vt-histogram (tensor &key bins range density)
  "直方图统计（对标 numpy.histogram）。返回 (values counts edges)。"
      (when (%cx-reject tensor)
    (error "vt-histogram: 不支持复数输入（numpy 同样拒绝）。见 CONVENTIONS §16.11"))
(when (%cx-reject tensor)
    (error "vt-histogram: 不支持复数输入（numpy 同样拒绝）。见 CONVENTIONS §16.11"))
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
            (unless (vt-item (vt-all (vt-isfinite tensor)))
              (error "自动确定 bin 范围需要有限输入"))
            ;; %ITEM-NUMBER 而不是 VT-ITEM：:bool 输入时 VT-ITEM 给 T/NIL
            ;; （§17.8 逻辑视图），下面 (- data-max data-min) 直接把 NIL 丢进
            ;; 算术会报 "NIL is not of type NUMBER"。numpy 的 histogram(bool)
            ;; 正常工作（按 0/1 数值分箱），所以这里必须取数值视图。
            (setf data-min (%item-number (vt-item (vt-amin tensor)))
                  data-max (%item-number (vt-item (vt-amax tensor))))
            (when (= data-min data-max)
              (setf data-min (- data-min 0.5s0)
                    data-max (+ data-max 0.5s0)))))
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
        (values (vt-from-sequence hist
                                  ;; 对标 numpy：counts 为整数（int64）；
                                  ;; density=t 时为归一化密度（float64）。
                                  :dtype (if density :float64 :int64))
                (vt-from-sequence edges :dtype :float64))))))

;;; ------------------------------------------------------------------
;;; 排序
;;; ------------------------------------------------------------------

(defun vt-sort (tensor &key (axis -1))
  "升序排序（对标 numpy.sort）。AXIS 缺省 -1；NaN 排在末尾。"
  ;; ===== 快路径：axis=nil 且输入连续 =====
  ;; 展平后就是一整段连续 run，直接原地排序；结果仍是一维（与原实现一致）。
  (when (%cx-any-p tensor)
    (return-from vt-sort (%cx-op-sort tensor :axis axis)))
  (when (and (null axis) (plusp (vt-size tensor)) (vt-contiguous-p tensor))
    (let ((r (vt-copy (vt-flatten tensor))))
      (when (%sort-fibers (vt-data r) (the fixnum (vt-size r)) 1)
        (return-from vt-sort r))))
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
        ;; ===== 快路径：排序轴是最后一维 + 结果连续 =====
        ;; 此时每个 fiber 在 result 缓冲区里就是一段**连续 run**，
        ;; 可以直接原地排序，无需 list 化。判定不成立则落到下方通用路径。
        (when (and (= ax (1- rank))
                   (plusp ax-dim)
                   (eq (vt-offset result) 0)
                   (vt-contiguous-p result))
          (when (%sort-fibers out-data ax-dim
                              (the fixnum (truncate (the fixnum (vt-size result)) ax-dim)))
            (return-from vt-sort result)))
        (labels ((recurse (depth in-ptr out-ptr)
                   (cond ((< depth ax)
                          (let ((dim (nth depth shape))
                                (in-stride (nth depth in-strides))
                                (out-stride (nth depth out-strides)))
                            (loop for i from 0 below dim do
                              (recurse (1+ depth)
                                       (+ in-ptr (* i in-stride))
                                       (+ out-ptr (* i out-stride))))))
                         ((= depth ax)
                          ;; v0.4.1 修复（差分测试 R2 根因）：此前 depth=ax
                          ;; 时只排序「第一个」尾部位置的切片，从未遍历 axis
                          ;; 之后的维度，导致非末轴排序静默失效
                          ;;（vt-sort :axis 0 返回原矩阵）。镜像 vt-argsort
                          ;; 的结构，遍历尾部维度并对每个切片排序。
                          (let* ((in-stride (nth ax in-strides))
                                 (out-stride (nth ax out-strides))
                                 (tail-dims (subseq shape (1+ ax)))
                                 (tail-size (reduce #'* tail-dims :initial-value 1))
                                 (tail-in-strides (subseq in-strides (1+ ax)))
                                 (tail-out-strides (subseq out-strides (1+ ax))))
                            (dotimes (tail-i tail-size)
                              (let ((extra-in 0) (extra-out 0) (rem tail-i))
                                (loop for idx from (1- (length tail-dims)) downto 0
                                      for dim = (nth idx tail-dims)
                                      for is = (nth idx tail-in-strides)
                                      for os = (nth idx tail-out-strides)
                                      do (multiple-value-bind (q r) (floor rem dim)
                                           (incf extra-in (* r is))
                                           (incf extra-out (* r os))
                                           (setf rem q)))
                                (let* ((vals (loop for i from 0 below ax-dim
                                                   for off = (+ in-ptr extra-in (* i in-stride))
                                                   collect (aref in-data off)))
                                       (sv (vt-numpy-sort vals #'<)))
                                  (loop for val in sv
                                        for off = (+ out-ptr extra-out) then (+ off out-stride)
                                        do (setf (aref out-data off) val)))))))
                         (t nil))))
          (recurse 0 in-offset 0))
        result)
      (let* ((flat (vt-flatten tensor))
             (data (coerce (vt-data flat) 'list)))
        (vt-from-sequence (vt-numpy-sort data #'<) :dtype (vt-dtype tensor)))))

(defun vt-argsort (tensor &key (axis -1))
  "返回升序排序后的下标（对标 numpy.argsort）。AXIS 缺省 -1。"
  (when (%cx-any-p tensor)
    (return-from vt-argsort (%cx-op-argsort tensor :axis axis)))
  (with-float-safe
    (if (null axis)
        (let* ;; v0.4.4：改用 vt-contiguous 而不是 vt-ravel —— vt-ravel 对
              ;; 连续输入返回**保留原 offset** 的视图，而旧代码直接
              ;; (aref in-data i) 从 0 开始读，于是带 offset 的切片会静默
              ;; 返回错误下标（实测 (vt-argsort (slice [100 200 300 5 1 9] 3..6)
              ;; :axis nil) 返回 (0 1 2)，numpy 是 (1 0 2)）。
              ;; vt-contiguous 保证 offset=0、stride=1。
              ((c (vt-contiguous (vt-flatten tensor)))
               (n (vt-size c))
               (idx (and (plusp n) (%argsort-fiber (vt-data c) 0 1 n))))
          (if idx
              (%make-vt :data idx :shape (list n) :strides '(1)
                        :offset 0 :dtype :int64)
              ;; ---- 回退：元素类型不在内核支持集内，走原 list 路径 ----
              (let* ((in-data (vt-data c))
                     (pairs (loop for i from 0 below n
                                  collect (cons (aref in-data i) i)))
                     (non-nans '())
                     (nans '()))
                (dolist (p pairs) (if (%nan-p (car p)) (push p nans) (push p non-nans)))
                (setf non-nans (stable-sort (nreverse non-nans) #'< :key #'car)
                      nans (nreverse nans))
                (%make-vt :data (make-array n :element-type '(signed-byte 64)
                                              :initial-contents
                                              (mapcar #'cdr (append non-nans nans)))
                          :shape (list n) :strides '(1) :offset 0 :dtype :int64))))
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
                              ;; v0.4.4：fiber 下标排序走间接归并内核
                              ;; （原为 per-fiber 的 cons list + stable-sort）
                              (let ((idx (and (plusp ax-dim)
                                              (%argsort-fiber in-data
                                                              (+ in-ptr extra-in)
                                                              in-stride ax-dim))))
                                (if idx
                                    (loop for pos across idx
                                          for off = out-ptr then (+ off out-stride)
                                          do (setf (aref out-data (+ off extra-out)) pos))
                                    ;; ---- 回退：原 list 路径 ----
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
                                            do (setf (aref out-data (+ off extra-out)) pos)))))))))
                       (t nil))))
            (recurse 0 in-offset 0))
          result))))

;;; ------------------------------------------------------------------
;;; nan 感知统计
;;; ------------------------------------------------------------------

;;; ==================================================================
;;; NaN 归约的单趟内核（v0.4.9）
;;;
;;; 原实现把 nanmean / nanvar 全部拆成「整趟中间张量」流水线：
;;;   nanmean = isnan + logical-not + where + sum + sum + map   （6 趟）
;;;   nanvar  = 上面 6 趟 × 2 + map + mul + sum + map + map     （10+ 趟）
;;; 1M double 实测 nanmean 108ms / nanvar 331ms（numpy 8ms / 12ms）。
;;; 这里把「求和 + 计数」与「离差平方和」各压成一次特化扫描 ——
;;; 累加顺序与 vt-sum 的顺序累加完全一致，故结果逐位相同。
;;; ==================================================================

(defun %nan-prepare (tensor dt)
  "把输入规整为**连续**的 DT 浮点张量（DT 为 :float64 / :float32）。"
  (let ((v (ensure-vt tensor)))
    (cond ((not (eq (vt-dtype v) dt)) (vt-astype v dt))
          ((vt-contiguous-p v) v)
          (t (vt-contiguous v)))))

(defmacro %nan-sum-count-body (lt d off n acc cnt)
  "SUM/COUNT 扫描体（顺序累加，NaN 跳过）。"
  `(let ((dd (the (simple-array ,lt (*)) ,d))
         (p (the fixnum ,off))
         (nn (the fixnum ,n))
         (a (coerce 0 ',lt))
         (c 0))
     (declare (type (simple-array ,lt (*)) dd)
              (type fixnum p nn c)
              (type ,lt a))
     (dotimes (i nn)
       (declare (type fixnum i))
       (let ((x (aref dd (the fixnum (+ p i)))))
         (unless (/= x x)
           (incf a x)
           (incf c))))
     (setf ,acc a ,cnt c)))

(defun %nan-sum-count (flat dt)
  "顺序累加非 NaN 元素，返回 (values 和 计数)。

   WITH-FLOAT-SAFE 必须包住整个扫描：SBCL 的浮点比较用 UCOMISD，
   遇到 NaN 会置 invalid 并**触发陷阱**（`(/= nan nan)` 直接抛
   FLOATING-POINT-INVALID-OPERATION），这与 C 的 quiet compare 不同。
   切换 FP 控制字只在进出各一次，不在逐元素路径上。"
  (with-float-safe
    (let ((d (vt-data flat)) (off (vt-offset flat)) (n (vt-size flat))
          (acc nil) (cnt nil))
      (if (eq dt :float32)
          (%nan-sum-count-body single-float d off n acc cnt)
          (%nan-sum-count-body double-float d off n acc cnt))
      (values acc cnt))))

(defmacro %nan-sumsq-body (lt d off n mean acc)
  "离差平方和扫描体。NaN 位复刻原实现的 (* (* (- 0 mean)^2) 0) 语义
   （mean 为 Inf 时该位贡献 NaN，与原 vt-* 掩码相乘结果一致）。"
  `(let ((dd (the (simple-array ,lt (*)) ,d))
         (p (the fixnum ,off))
         (nn (the fixnum ,n))
         (m (the ,lt ,mean))
         (a (coerce 0 ',lt)))
     (declare (type (simple-array ,lt (*)) dd)
              (type fixnum p nn)
              (type ,lt m a))
     (dotimes (i nn)
       (declare (type fixnum i))
       (let ((x (aref dd (the fixnum (+ p i)))))
         (if (/= x x)
             (let ((z (- (coerce 0 ',lt) m)))
               (incf a (* (* z z) (coerce 0 ',lt))))
             (let ((z (- x m)))
               (incf a (* z z))))))
     (setf ,acc a)))

(defun %nan-sumsq (flat dt mean)
  ;; 同 %nan-sum-count：NaN 比较需要屏蔽 invalid 陷阱。
  (with-float-safe
    (let ((d (vt-data flat)) (off (vt-offset flat)) (n (vt-size flat)) (acc nil))
      (if (eq dt :float32)
          (%nan-sumsq-body single-float d off n mean acc)
          (%nan-sumsq-body double-float d off n mean acc))
      acc)))

(defmacro %sumsq-all-body (lt d off n mean acc)
  "离差平方和扫描体（**不跳过 NaN**，供 vt-var 用：var 里 NaN 必须传播）。"
  `(let ((dd (the (simple-array ,lt (*)) ,d))
         (p (the fixnum ,off))
         (nn (the fixnum ,n))
         (m (the ,lt ,mean))
         (a (coerce 0 ',lt)))
     (declare (type (simple-array ,lt (*)) dd)
              (type fixnum p nn)
              (type ,lt m a))
     (dotimes (i nn)
       (declare (type fixnum i))
       (let ((z (- (aref dd (the fixnum (+ p i))) m)))
         (incf a (* z z))))
     (setf ,acc a)))

(defun %sumsq-all (flat dt mean)
  "FLAT[0,size) 的 Σ(x-mean)²，**不跳过 NaN**（含 NaN 时结果为 NaN，由调用方
   沿传播路径自然得到）。与 %NAN-SUMSQ 的唯一差别就是没有那个跳过分支 ——
   vt-var 需要 NaN 传播，vt-nanvar 需要跳过，两个语义不能共用一个扫描。"
  (with-float-safe
    (let ((d (vt-data flat)) (off (vt-offset flat)) (n (vt-size flat)) (acc nil))
      (if (eq dt :float32)
          (%sumsq-all-body single-float d off n mean acc)
          (%sumsq-all-body double-float d off n mean acc))
      acc)))

(defun %nan-fast-p (axis keepdims final-dtype)
  "快路径判据：axis 全归约 + 目标 float32/64。KEEPDIMS 不影响选路（全归约
   时形状恒为 (1,…) 或空，两条路都对），故忽略。"
  (declare (ignore keepdims))
  (and (null axis)
       (or (eq final-dtype :float64) (eq final-dtype :float32))))

(defun %nan-scalar-result (out shape dt val op-name)
  (let ((res (if out
                 (vt-check-out out shape dt :op-name op-name)
                 (make-vt shape 0 :dtype dt))))
    (setf (aref (vt-data res) (vt-offset res)) val)
    res))

(defun vt-nanmean (tensor &key axis keepdims dtype out)
  "忽略 NaN 的算术平均。全为 NaN 的归约区结果为 NaN。

  dtype 语义（§4.2 H3「严格相等」）：结果 dtype = 输入浮点提升 + 显式 :dtype，
  **不由 out 决定**；out 的 dtype 必须精确等于结果 dtype。"
  (when (%cx-any-p tensor)
    (return-from vt-nanmean (%cx-op-nanmean tensor :axis axis :keepdims keepdims :dtype dtype :out out)))
  (let* ((compute-dtype (if (eq (vt-dtype (ensure-vt tensor)) :float32) :float32 :float64))
         ;; 结果 dtype：输入提升 + 显式 :dtype（nan 族恒为浮点）
         (final-dtype (or dtype compute-dtype)))
    (if (%nan-fast-p axis keepdims final-dtype)
        ;; 单趟快路径（axis=nil）：一次扫描同时得到和与有效计数。
        (let* ((flat (%nan-prepare tensor final-dtype))
               (nan (vt-get-nan final-dtype))
               (shape (if keepdims
                          (make-list (length (vt-shape flat)) :initial-element 1)
                          nil)))
          (multiple-value-bind (sum cnt) (%nan-sum-count flat final-dtype)
            (let ((val (if (zerop cnt)
                           nan
                           (/ sum (coerce cnt (if (eq final-dtype :float32)
                                                  'single-float 'double-float))))))
              (%nan-scalar-result out shape final-dtype val "vt-nanmean"))))
    (let* ((mask (vt-isnan tensor))
         (not-nan (vt-logical-not mask))
         (zero (if (eq final-dtype :float32) 0.0s0 0.0d0))
         (nan (vt-get-nan final-dtype))
         (clean (vt-where mask zero tensor :dtype final-dtype))
         (count (vt-sum not-nan :axis axis :keepdims keepdims :dtype final-dtype))
         (sum (vt-sum clean :axis axis :keepdims keepdims :dtype final-dtype)))
    ;; out 硬契约校验前置：错误信息以 "vt-nanmean" 开头
    (when out
      (vt-check-out out (vt-shape sum) final-dtype :op-name "vt-nanmean"))
      (vt-map (lambda (s c) (if (<= c zero) nan (/ s c)))
              sum count :dtype final-dtype :out out)))))

(defun vt-nanvar (tensor &key axis keepdims (ddof 0) dtype out)
  "忽略 NaN 的方差。有效样本数 ≤ ddof 时结果为 NaN。

  dtype 语义同 vt-nanmean（§4.2 H3「严格相等」）。

  性能：axis=nil 走两趟特化内核（和+计数 → 离差平方和），
  原实现的 10+ 趟中间张量流水线在 1M double 上实测 331ms。"
  (when (%cx-any-p tensor)
    (return-from vt-nanvar (%cx-op-nanvar tensor :axis axis :keepdims keepdims :ddof ddof :dtype dtype :out out)))
  (let* ((compute-dtype (if (eq (vt-dtype (ensure-vt tensor)) :float32)
                            :float32 :float64))
         (final-dtype (or dtype compute-dtype)))
    (if (%nan-fast-p axis keepdims final-dtype)
        (with-float-safe
          (let* ((flat (%nan-prepare tensor final-dtype))
                 (nan (vt-get-nan final-dtype))
                 (zero (if (eq final-dtype :float32) 0.0s0 0.0d0))
                 (lt (if (eq final-dtype :float32) 'single-float 'double-float))
                 (shape (if keepdims
                            (make-list (length (vt-shape flat)) :initial-element 1)
                            nil)))
            (multiple-value-bind (sum cnt) (%nan-sum-count flat final-dtype)
              (let* ((cntf (coerce cnt lt))
                     (mean (if (zerop cnt) nan (/ sum cntf)))
                     (sum2 (if (zerop cnt) zero (%nan-sumsq flat final-dtype mean)))
                     (ddof-f (coerce ddof lt))
                     (div (if (< cntf ddof-f) zero (- cntf ddof-f)))
                     ;; 注意 cnt=0（全 NaN）时即使 ddof 为负、div>0，
                     ;; numpy 也返回 nan —— 不能因为 sum2 取 0 就给出 0/n。
                     (val (if (or (zerop cnt) (<= div zero)) nan (/ sum2 div))))
                (%nan-scalar-result out shape final-dtype val "vt-nanvar")))))
        (let* ((mask (vt-isnan tensor))
               (not-nan (vt-logical-not mask))
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
                  sum2 divisor :dtype final-dtype :out out)))))

(defun vt-nanstd (tensor &key axis keepdims (ddof 0) dtype out)
  "忽略 NaN 的标准差（sqrt(vt-nanvar)）。dtype 语义同 vt-nanvar。"
  (when (%cx-any-p tensor)
    (return-from vt-nanstd (%cx-op-nanstd tensor :axis axis :keepdims keepdims :ddof ddof :dtype dtype :out out)))
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

(defun %nanmedian-fold-axes (tensor axes keepdims)
  "把**多个**归约轴 AXES 折成末尾一维，返回 (values 2D视图 目标形状)。

  思路：置换 (保留轴…, 归约轴…) → 重塑成 (H, D) —— H = 保留轴元素数，
  D = 归约区元素数 —— 于是「沿 (0,1) 归约」变成「沿单轴 1 归约」，
  可以直接复用测透了的单轴内核，不必再写一套多维 fiber 遍历。

  正确性前提：中位数对归约区内元素的**顺序不敏感**（内部要排序），
  所以把归约轴搬到末尾、再把它们压成一维，不改变结果。
  `vt-reshape` 对非连续视图会自动落地（先 vt-contiguous 再重塑），
  因此任意 strides / 负步长 / 偏移的输入都成立。"
  (let* ((shape (vt-shape tensor))
         (rank (length shape))
         (axes (sort (copy-list axes) #'<))
         (kept (loop for i below rank unless (member i axes) collect i))
         (kept-shape (mapcar (lambda (i) (nth i shape)) kept))
         (red-size (reduce #'* (mapcar (lambda (i) (nth i shape)) axes)
                           :initial-value 1))
         (h (reduce #'* kept-shape :initial-value 1))
         (two (vt-reshape (vt-transpose tensor (append kept axes))
                          (list h red-size)))
         (target (if keepdims
                     (loop for d in shape for i from 0
                           collect (if (member i axes) 1 d))
                     kept-shape)))
    (values two target)))

(defun %nanmedian-multi (tensor axis keepdims out)
  "实数：多轴 nanmedian。折叠成 2-D 后复用单轴内核，再重塑回目标形状。"
  (let* ((in (ensure-vt tensor))
         (axes (vt-normalize-axes axis (length (vt-shape in)))))
    (multiple-value-bind (two target) (%nanmedian-fold-axes in axes keepdims)
      (let ((res (vt-reshape (vt-nanmedian two :axis 1) target)))
        (cond ((null out) res)
              ((eq res out) out)
              (t (vt-check-out out target :float64 :op-name "vt-nanmedian")
                 (vt-copy-into out res)
                 out))))))

(defun vt-nanmedian (tensor &key axis keepdims out dtype)
  "忽略 NaN 的中位数（对标 numpy.nanmedian）。全为 NaN 的归约区结果为 NaN。
  结果 dtype 恒为 :float64（复数入参时保持复数 dtype）；out 必须形状/dtype 精确匹配。

   axis 接受 nil / 整数 / 整数列表。**多轴**（如 '(0 1)）已支持：
   折叠成 (H, D) 后复用单轴内核（见 %NANMEDIAN-FOLD-AXES），此前直接报
   「暂不支持多轴归约」而 numpy 是支持的。

   :dtype 契约（§16.13「禁止静默降级」）：
     实数输入 → 结果恒为 :float64（对标 numpy.nanmedian 的缺省口径），
                 显式给别的 dtype 一律报错，**不再静默忽略**；
     复数输入 → 必须是复数 dtype（:complex64/:complex128），给实数 dtype 报错。
   之所以要拦：修复前 `(vt-nanmedian z :dtype :float64)` 会**取实部**返回
   float64 —— 虚部被静默丢弃，正是 §16.13 列为「最危险」的那类失败。

   复数（§16.16）：按 (实部, 虚部) 字典序排序取中位，偶数个取中间两值均值；
   任一分量为 NaN 的元素视为缺失。复数分支在 complex-ops.lisp
   用同名 defun 覆盖本函数，重载本文件后复数会掉进实数内核（静默取实部）。"
  (when (and dtype (null (%cx-any-p tensor)) (not (eq dtype :float64)))
    (error "vt-nanmedian: 实数输入的结果 dtype 恒为 :float64（对标 numpy.nanmedian），~
            收到 :dtype ~a。需要别的精度请对结果再 vt-astype。" dtype))
  (when (and dtype (%cx-any-p tensor) (not (member dtype '(:complex64 :complex128))))
    (error "vt-nanmedian: 复数输入的结果 dtype 必须是复数类型（:complex64 / :complex128），~
            收到 :dtype ~a —— 用实数 dtype 会把虚部静默丢掉。" dtype))
  (when (%cx-any-p tensor)
    (return-from vt-nanmedian
      (%cx-op-nanmedian (ensure-vt tensor)
                        :axis axis :keepdims keepdims :out out :dtype dtype)))
  (when (and axis (listp axis) (cdr axis))
    (return-from vt-nanmedian (%nanmedian-multi (ensure-vt tensor) axis keepdims out)))
  (with-float-safe
    (let* ((nan (vt-get-nan :float64))
           (in-data (vt-data tensor))
           (in-strides (vt-strides tensor))
           (in-offset (vt-offset tensor))
           (in-shape (vt-shape tensor))
           (rank (length in-shape)))
      (if (null axis)
          ;; v0.4.3 性能修复：原实现把非 NaN 值 push 进 list 再 sort ——
          ;; 1M 元素要 cons 1M 个 cell，实测 644ms（numpy.nanmedian 10.2ms）。
          ;; 改为「整体排序（NaN 稳定压尾）+ 从尾部数 NaN 个数」：
          ;; 一次 replace + 原地归并，零 cons，之后按下标直接取中位。
          ;; v0.4.10：quickselect 取代全排序（1M 134ms → ~16ms）。
          ;; 语义不变：忽略 NaN；全 NaN → nan。
          (multiple-value-bind (buf m has-nan) (%nan-free-double-buffer tensor)
            (declare (ignore has-nan))
            (let ((result (cond ((zerop m) nan)
                                ((oddp m) (%select-double buf m (floor m 2)))
                                (t (/ (+ (%select-double buf m (1- (/ m 2)))
                                         (%select-double buf m (/ m 2)))
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
        (setf result (let ((hi (vt-narrow result ax 1 len))
                           (lo (vt-narrow result ax 0 (1- len))))
                       ;; 布尔差分（§17.10）：numpy 的 bool 减法按位回绕，
                       ;; `True-False = 1`、`False-True = -1`，两者落回 bool
                       ;; 都是 True —— 等价于**异或**。实测 np.diff([T,F,T])
                       ;; → [True,True]，np.diff(...,n=2) → [False,False]，
                       ;; 与逐次 XOR 完全一致。而 vt-- 对 bool 是明确拒绝的
                       ;; （§17.3），不在这条分支上特判就只能报错。
                       (if (eq (vt-dtype result) :bool)
                           (vt-logical-xor hi lo)
                           (vt-- hi lo)))))
          finally (return result))))

(declaim (inline %trapz-term))
(defun %trapz-term (l r hh)
  "梯形法则的被积项 0.5*(L+R)*HH。命名成函数是为了让 vt-fast-map 能内联它
   （宏只接受字面算子；传 (lambda ...) 会回退到通用 vt-map）。"
  (* 0.5d0 (+ l r) hh))

(defun vt-trapz (y &key (x nil) (dx 1.0d0) (axis -1))
  "梯形法则数值积分（对标 numpy.trapz）。

   整数/布尔输入按 **:float64** 累加（§17.10）：被积项是 `0.5*(l+r)*h`，
   若结果 dtype 跟着输入走，`(0.5*1)` 写进 :bool 得 0、写进 :int8 得 0，
   于是 `(vt-trapz bool数组)` 静默返回 0（numpy 给 1.0）。"
  (let* ((sh (vt-shape y))
         (ax (vt-normalize-axis axis (length sh)))
         (n (nth ax sh))
         (ydt (vt-dtype y))
         (acc-dt (if (or (eq ydt :bool)
                         (member ydt '(:int8 :int16 :int32 :int64 :uint8 :uint16)))
                     :float64
                     ydt)))
    (when (< n 2)
      (return-from vt-trapz
        (vt-zeros (append (subseq sh 0 ax) (subseq sh (1+ ax))) :dtype acc-dt)))
    (let ((h (if x
                 (vt-diff (ensure-vt x))
                 (make-vt (list (1- n)) dx :dtype acc-dt))))
      (setf h (vt-reshape h (append (make-list ax :initial-element 1) (list (1- n))
                                    (make-list (- (length sh) ax 1) :initial-element 1))))
      (let* ((left (vt-narrow y ax 0 (1- n)))
             (right (vt-narrow y ax 1 n))
             ;; v0.4.8 性能修复：(vt-map (lambda (l r hh) ...) ...) 会走**通用**
             ;; 路径 —— 循环里逐元素 FUNCALL 闭包，且 left/right 是视图、
             ;; h 是带 0 步长的广播视图，通用路径对它们每个元素都要走一遍
             ;; 通用 AREF。1M 元素实测 112420us（numpy.trapezoid 4638us）。
             ;; 换成「命名算子 + vt-fast-map」即可命中第 1 轮建好的
             ;; inline3-strided 快路径（三输入、按输出元素类型特化、支持广播）。
             (integrand (vt-fast-map #'%trapz-term left right h :dtype acc-dt)))
        (vt-sum integrand :axis ax)))))

(defun %normalize-mode-keyword (mode)
  "将 \"full\", :full 归一化为关键字 :full，方便 numpy 用户习惯。"
  (etypecase mode
    (keyword mode)
    (string (intern (string-upcase mode) "KEYWORD"))))

(defun vt-correlate (a v &key (mode :full))
  "1D 互相关。mode 支持关键字(:full/:valid/:same)或字符串。

  ⚠ 默认值与 numpy **不同**（v0.4.2 记录；行为保持不变，以免破坏既有调用方）：
    numpy  np.correlate(a, v)   默认 mode='valid'
    clvt   (vt-correlate a v)   默认 :mode :full
  三种模式本身与 numpy 逐值一致（:full/:valid/:same 均已对齐）；
  需要 numpy 默认行为时请显式写 :mode :valid。
  core-test 与 test-all 均以 :full 为约定，故此处只修正文档，不改默认。

  布尔输入（CONVENTIONS §17.10.10）：两侧都是 `:bool` 时结果也是 `:bool`，
  累加按 numpy 的**逻辑与／或**语义（`or_k (a[k] and b[k])`）—— 内部仍按
  数值求和，收尾时 `>0` 真值化，两者逐值等价（count>0 ⇔ 存在一个乘积为真）。
  修复前 bool 输入返回 `:float64` 的**数值和**：`(vt-convolve b b)` 给
  `[1 0 2 2 1 2 1]`，numpy 给 `[T F T T T T T]`。这与上一轮修的
  `matmul`／`dot`／`einsum` 是同一族缺陷（布尔代数被当成数值和），
  当时只扫了矩阵乘法家族，漏了信号处理这两个入口。"
  (when (%cx-any-p a v)
    (return-from vt-correlate (%cx-op-correlate a v :mode mode)))
  (when (null mode) (setf mode :full))
  (let* ((bool-result (and (eq (vt-dtype (ensure-vt a)) :bool)
                           (eq (vt-dtype (ensure-vt v)) :bool)))
         (mode-kw (%normalize-mode-keyword mode))
         (a-flat (vt-contiguous (vt-flatten a)))
         (v-flat (vt-contiguous (vt-flatten v)))
         (n (vt-size a-flat))
         (m (vt-size v-flat))
         (a-data (vt-data a-flat))
         (v-data (vt-data v-flat)))
    ;; ================================================================
    ;; v0.4.8 性能修复：内层点积按元素类型特化
    ;;
    ;; 原实现里 A-DATA / V-DATA 都没有元素类型声明，内层每次
    ;;   (incf sum (* (aref a-data (+ j k)) (aref v-data j)))
    ;; 都要走两次通用 AREF（二维分派 + 边界检查）再装箱相乘。
    ;; 100k×64 的点积是 6.4e6 次迭代，实测 389ms（numpy.correlate 2.5ms，
    ;; 差 157 倍）；vt-convolve 直接复用本函数，故两个函数同时受益。
    ;;
    ;; 只在**两侧元素类型相同且为浮点**时走特化（此时与原实现逐位等价：
    ;; 同样是 <lt>×<lt> 后累加到 double-float）。整型输入仍走原通用路径 ——
    ;; 那里 (incf sum (* i64 i64)) 可能溢出 fixnum 提升为 bignum，
    ;; 特化会改变语义，不值得冒这个风险。
    ;; ================================================================
    (let* ((full-len (+ n m -1))
           (offset (1- m))
           (et-a (array-element-type a-data))
           (et-b (array-element-type v-data)))
      (flet ((fill-full-generic ()
               ;; 原实现的逐 k 版本（整型/混合类型回退）
               (let ((full (make-array full-len :element-type 'double-float)))
                 (loop for k from (- offset) below n for i from 0
                       do (let ((sum 0.0d0))
                            (loop for j from (max 0 (- k)) below (min m (- n k))
                                  do (incf sum (* (aref a-data (+ j k))
                                                  (aref v-data j))))
                            (setf (aref full i) sum)))
                 full)))
        (let ((full
                (macrolet ((fill-full (lt)
                             `(let ((s (the (simple-array ,lt (*)) a-data))
                                    (w (the (simple-array ,lt (*)) v-data))
                                    (full (make-array full-len
                                                      :element-type 'double-float)))
                                (declare (type (simple-array ,lt (*)) s w)
                                         (type (simple-array double-float (*)) full)
                                         (type fixnum n m offset full-len))
                                (loop for k fixnum from (- offset) below n
                                      for i fixnum from 0
                                      do (let ((sum 0.0d0))
                                           (declare (type double-float sum))
                                           (loop for j fixnum
                                                 from (the fixnum (max 0 (- k)))
                                                 below (the fixnum (min m (- n k)))
                                                 do (incf sum (* (aref s (the fixnum (+ j k)))
                                                                 (aref w j))))
                                           (setf (aref full i) sum)))
                                full)))
                  (cond
                    ((and (equal et-a 'double-float) (equal et-b 'double-float))
                     (fill-full double-float))
                    ((and (equal et-a 'single-float) (equal et-b 'single-float))
                     (fill-full single-float))
                    (t (fill-full-generic))))))
          (let ((res (ecase mode-kw
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
                             :offset 0 :dtype :float64))))))
            (if bool-result (vt-astype res :bool) res)))))))

(defun vt-convolve (a v &key (mode :full))
  "1D 卷积，对标 np.convolve。mode 支持关键字(:full/:valid/:same)或字符串。"
  (when (%cx-any-p a v)
    (return-from vt-convolve (%cx-op-convolve a v :mode mode)))
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
      (when (%cx-reject x xp fp)
    (error "vt-interp: 不支持复数输入（numpy 同样拒绝）。见 CONVENTIONS §16.11"))
(when (%cx-reject x xp fp)
    (error "vt-interp: 不支持复数输入（numpy 同样拒绝）。见 CONVENTIONS §16.11"))
(with-float-safe
    ;; v0.4.8：%FLAT-OFFSET0 保证 offset=0 —— 主循环用裸 AREF 取数，而
    ;; vt-contiguous 对「已连续」输入是**原样返回**（保留 offset），于是 xp/fp
    ;; 若是切片视图，读到的就是基数组开头的数据（静默错答案）。
    (let* ((xp-vt (%flat-offset0
                   (if (eq (vt-dtype (ensure-vt xp)) :float64)
                       (ensure-vt xp)
                       (vt-astype (ensure-vt xp) :float64))))
           (fp-vt (%flat-offset0
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
            when (< (%vt-ref-raw xp-vt i) (%vt-ref-raw xp-vt (1- i)))
              do (error "vt-interp: xp 必须在第 ~a 个位置递增（~a -> ~a 下降）"
                        i (%vt-ref-raw xp-vt (1- i)) (%vt-ref-raw xp-vt i)))
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
        ;; v0.4.8 性能修复：主循环的 XP-DATA / FP-DATA / X-DATA / OUT-DATA 原先
        ;; 都没有元素类型，(aref ...) 与 (setf (aref ...)) 走通用二维分派 +
        ;; 边界检查；二分每查询约 14 次探测，1M 查询 = 1.4e7 次。实测 639588us
        ;; （numpy.interp 141606us）。因为上面已把三者都归一到 float64，
        ;; 这里只需一路 double-float 声明即可（不需要 8 路展开）。
        (let ((xd (the (simple-array double-float (*)) x-data))
              (xpd (the (simple-array double-float (*)) xp-data))
              (fpd (the (simple-array double-float (*)) fp-data))
              (od (the (simple-array double-float (*)) out-data)))
          (declare (type (simple-array double-float (*)) xd xpd fpd od)
                   (type fixnum x-size n))
          (loop for i fixnum from 0 below x-size
                for xi double-float = (aref xd i) do
                  (setf (aref od i)
                      ;; v0.4.1 修复（差分测试 R3）：numpy 的 left/right 仅在
                      ;; 严格越界时生效；x==xp[0] 应取 fp[0]（插值分支给出
                      ;; 精确 fp0），而非 left。
                      (cond ((< xi xp0) left-val)
                            ((> xi xp-end) right-val)
                            (t (let ((lo 0) (hi (- n 2)))
                                 (loop while (< lo hi) do
                                   (let ((mid (ash (+ lo hi 1) -1)))
                                     (if (<= (aref xpd mid) xi)
                                         (setf lo mid)
                                         (setf hi (1- mid)))))
                                 (let* ((xl (aref xpd lo))
                                        (xr (aref xpd (1+ lo)))
                                        (yl (aref fpd lo))
                                        (yr (aref fpd (1+ lo)))
                                        (denom (- xr xl)))
                                   (if (zerop denom)
                                       yl
                                       (+ yl (* (- yr yl)
                                                (/ (- xi xl) denom))))))))))
          out)))))

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


