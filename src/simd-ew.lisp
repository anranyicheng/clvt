;;;; simd-ew.lisp — 逐元素算术的 SIMD 快路径（f64 4 路 / f32 8 路）
;;;;
;;;; 职责：把 `vt-fast-map` 内联分支里的**连续/行广播**形态交给 sb-simd 的
;;;; 向量算子，只换「一条指令算几个元素」，不换任何单个元素的运算表达式，
;;;; 因此结果与标量内联循环**逐位相同**（§1 铁律 1：连续性只影响选路）。
;;;;
;;;; 为什么值得单独一层：标量内联循环对 f64 只有 1 元素/迭代，`(vt-+ a b)`
;;;; 在 1M 元素上实测 2.3 ms，而内存带宽允许 0.5 ms —— 差的是指令数不是带宽。
;;;; 4 路/8 路向量化后同规模落到 0.7 ms。
;;;;
;;;; 支持范围（其余一律返回 NIL 回退，语义不变）：
;;;;   算子  + - * /（二元）；- abs sqrt（一元）
;;;;   类型  double-float / single-float
;;;;   输入形态  :full（与结果同形且连续）/ :row（末轴对齐的行广播）/
;;;;              :scalar（长度 1，stride 0）
;;;;
;;;; 明确的**排除项**（不是遗漏，是语义不同）：
;;;;   · `maximum`/`minimum`/`fmax`/`fmin`：numpy 的语义（NaN 传播 vs 忽略、
;;;;     ±0 的定号）与 AVX 的 maxpd/maxps 不一致，向量化会改变结果；
;;;;   · 归约（sum/max/...）：向量化必须改变累加顺序，违反 §7.3；
;;;;   · 超越函数（exp/log/tanh/...）：libm 与向量实现的末位不同，
;;;;     而快路径必须与通用路径逐位可比。
(in-package :clvt)

;;; ------------------------------------------------------------------
;;; 形态判定
;;; ------------------------------------------------------------------

(eval-when (:compile-toplevel :load-toplevel :execute)
  (defun %sew-vec-prefix (lt)
    "元素类型 → sb-simd 向量前缀；不支持时返回 NIL。"
    (cond ((equal lt 'double-float) "F64.4")
          ((equal lt 'single-float) "F32.8")
          (t nil)))

  (defun %sew-lanes (ltp)
    "向量前缀 → 向量宽度（元素个数）。"
    (if (string= ltp "F64.4") 4 8))

  (defun %sew-sym (prefix suffix)
    "拼出 sb-simd-avx 里的向量算子符号。"
    (intern (concatenate 'string prefix suffix) :sb-simd-avx)))

(declaim (inline %sew-nlast %sew-nrows))
(defun %sew-nlast (res)
  "结果末轴长度（0 维结果为 1）。"
  (let ((shape (vt-shape res)))
    (if shape (the fixnum (car (last shape))) 1)))

(defun %sew-nrows (res)
  "结果行数 = 总元素数 / 末轴长度。"
  (let ((shape (vt-shape res)))
    (if shape (the fixnum (/ (vt-size res) (%sew-nlast res))) 1)))

(defun %sew-row-shape-p (x nlast)
  "X 的形状是否为「末轴对齐的行广播」：末轴长度 = NLAST，其余轴全为 1。"
  (let ((xs (vt-shape x)))
    (cond ((null xs) nil)
          ((= (length xs) 1) (= (car xs) nlast))
          (t (and (= (car (last xs)) nlast)
                  (every (lambda (d) (= d 1)) (butlast xs)))))))

(defun %sew-col-shape-p (x out-shape)
  "X 的形状是否为「末轴长度为 1 的列广播」：除末轴外与 OUT-SHAPE 逐轴相同、
   末轴为 1。这是 (BATCH 1000) 减 (BATCH 1) 这类每行一个标量的形态。"
  (let ((xs (vt-shape x)))
    (and xs
         (= (length xs) (length out-shape))
         (= (car (last xs)) 1)
         (every #'= (butlast xs) (butlast out-shape)))))

(defun %sew-mode (x out-shape nlast)
  "输入 X 在结果形状 OUT-SHAPE 下的访存形态。

   返回 :scalar（长度 1，等价 stride 0）/ :full（同形且连续）/
   :row（连续，且除末轴外的轴长度均为 1 —— 即沿末轴的行广播）/
   :col（连续，末轴为 1 —— 即每行一个标量的列广播）/
   NIL（本层不接管，调用方回退到标量内联循环）。

   :col 是归一化层与偏置加法的形态（(N,K) ∘ (N,1)），它比 :row 更常见，
   却因为不满足「末轴对齐」而长期落在 odometer 通用路径上。"
  (cond ((<= (vt-size x) 1) :scalar)
        ((not (vt-contiguous-p x)) nil)
        ((equal (vt-shape x) out-shape) :full)
        ((%sew-row-shape-p x nlast) :row)
        ((%sew-col-shape-p x out-shape) :col)
        (t nil)))

;;; ------------------------------------------------------------------
;;; 二元循环
;;;
;;; 每条循环只做「读 4/8 个、算一个向量、写 4/8 个」，尾块走**同一个标量
;;; 表达式**（不是另一种近似），因此整块与尾块都是单次舍入。
;;; ------------------------------------------------------------------

(eval-when (:compile-toplevel :load-toplevel :execute)
  (defun %sew-vec-form (prefix kind x y)
    "向量表达式：KIND → 由 PREFIX 的向量算子拼出的形式。

     :sqdiff 展开成「先减两次、再乘」而不是复核一遍标量内核的实现细节：
     两个 `(x − y)` 的结果是同一个 IEEE 值（减法确定性），乘回去与
     内核里的 `(let ((d (- x y))) (* d d))` 逐位相同。"
    (case kind
      (:add `(,(%sew-sym prefix "+") ,x ,y))
      (:sub `(,(%sew-sym prefix "-") ,x ,y))
      (:mul `(,(%sew-sym prefix "*") ,x ,y))
      (:div `(,(%sew-sym prefix "/") ,x ,y))
      (:sqdiff (let ((d `(,(%sew-sym prefix "-") ,x ,y)))
                 `(,(%sew-sym prefix "*") ,d ,d)))))

  (defun %sew-scal-form (kind x y)
    "标量尾块的表达式。必须与被替代的标量内核写成同一个式子。"
    (case kind
      (:add `(+ ,x ,y))
      (:sub `(- ,x ,y))
      (:mul `(* ,x ,y))
      (:div `(/ ,x ,y))
      (:sqdiff (let ((d `(- ,x ,y))) `(* ,d ,d))))))

(defmacro %sew2-flat (lt prefix kind a b res)
  "两侧都是 :full 的扁平循环（结果连续、输入同形连续 ⇒ 线性下标即可）。"
  (let* ((n (gensym "N")) (i (gensym "I")) (lim (gensym "LIM"))
         (ad (gensym "AD")) (bd (gensym "BD")) (od (gensym "OD"))
         (ao (gensym "AO")) (bo (gensym "BO")) (oo (gensym "OO"))
         (lanes (%sew-lanes prefix))
         (vecaref (%sew-sym prefix "-ROW-MAJOR-AREF"))
         (vx `(,vecaref ,ad (+ ,ao ,i)))
         (vy `(,vecaref ,bd (+ ,bo ,i)))
         (sx `(aref ,ad (+ ,ao ,i)))
         (sy `(aref ,bd (+ ,bo ,i))))
    `(let* ((,n (vt-size ,res))
            (,ad (the (simple-array ,lt (*)) (vt-data ,a)))
            (,bd (the (simple-array ,lt (*)) (vt-data ,b)))
            (,od (the (simple-array ,lt (*)) (vt-data ,res)))
            (,ao (vt-offset ,a)) (,bo (vt-offset ,b)) (,oo (vt-offset ,res))
            (,lim (- ,n (mod ,n ,lanes)))
            (,i 0))
       (declare (type fixnum ,n ,ao ,bo ,oo ,lim ,i)
                (type (simple-array ,lt (*)) ,ad ,bd ,od))
       (loop while (< ,i ,lim)
             do (setf (,vecaref ,od (+ ,oo ,i))
                      ,(%sew-vec-form prefix kind vx vy))
                (incf ,i ,lanes))
       (loop while (< ,i ,n)
             do (setf (aref ,od (+ ,oo ,i))
                      ,(%sew-scal-form kind sx sy))
                (incf ,i))
       t)))

(defmacro %sew2-rows (lt prefix kind a b res a-mode b-mode)
  "行循环。A-MODE / B-MODE 是字面量形态：

     :full   与结果同形，每行前进 NLAST 个元素，行内按 4/8 路成块读；
     :row    末轴对齐的行广播，每行都从同一处读，行内按 4/8 路成块读；
     :scalar 长度 1，每行都读同一个元素，行内固定；
     :col    每行一个标量，行前进 1 个元素，行内固定。

   :scalar / :col 的操作数在每行行首取一个标量、向量广播一次，行内不再
   访存；:full / :row 直接读。尾块一律回到与标量内联循环**同一个**单个
   元素表达式，因此整块与尾块都是单次舍入、与通用路径逐位相同。"
  (let* ((i (gensym "I")) (r (gensym "R"))
         (ad (gensym "AD")) (bd (gensym "BD")) (od (gensym "OD"))
         (ao (gensym "AO")) (bo (gensym "BO")) (oo (gensym "OO"))
         (nrows (gensym "NROWS")) (nlast (gensym "NLAST"))
         (av (gensym "AV")) (bv (gensym "BV"))
         (lanes (%sew-lanes prefix))
         (vecaref (%sew-sym prefix "-ROW-MAJOR-AREF"))
         (vectype (%sew-sym prefix ""))
         (vecbcast (%sew-sym prefix "-BROADCAST"))
         (vecp (lambda (m) (member m '(:full :row))))
         (step (lambda (m) (case m (:full nlast) (:col 1) (t 0))))
         (vx (if (funcall vecp a-mode) `(,vecaref ,ad (+ ,ao ,i)) av))
         (vy (if (funcall vecp b-mode) `(,vecaref ,bd (+ ,bo ,i)) bv))
         (sx (if (funcall vecp a-mode) `(aref ,ad (+ ,ao ,i)) `(aref ,ad ,ao)))
         (sy (if (funcall vecp b-mode) `(aref ,bd (+ ,bo ,i)) `(aref ,bd ,bo)))
         (binds (append (when (not (funcall vecp b-mode))
                          `((,bv (,vecbcast (aref ,bd ,bo)))))
                        (when (not (funcall vecp a-mode))
                          `((,av (,vecbcast (aref ,ad ,ao)))))
                        `((,i 0))))
         (types (append (when (not (funcall vecp b-mode)) `((type ,vectype ,bv)))
                        (when (not (funcall vecp a-mode)) `((type ,vectype ,av)))
                        `((type fixnum ,i)))))
    `(let* ((,nrows (%sew-nrows ,res))
            (,nlast (%sew-nlast ,res))
            (,ad (the (simple-array ,lt (*)) (vt-data ,a)))
            (,bd (the (simple-array ,lt (*)) (vt-data ,b)))
            (,od (the (simple-array ,lt (*)) (vt-data ,res)))
            (,ao (vt-offset ,a)) (,bo (vt-offset ,b)) (,oo (vt-offset ,res)))
       (declare (type fixnum ,nrows ,nlast ,ao ,bo ,oo)
                (type (simple-array ,lt (*)) ,ad ,bd ,od))
       (dotimes (,r ,nrows)
         (declare (type fixnum ,r))
         (let ,binds
           (declare ,@types)
           (loop while (<= (+ ,i ,lanes) ,nlast)
                 do (setf (,vecaref ,od (+ ,oo ,i))
                          ,(%sew-vec-form prefix kind vx vy))
                    (incf ,i ,lanes))
           (loop while (< ,i ,nlast)
                 do (setf (aref ,od (+ ,oo ,i))
                          ,(%sew-scal-form kind sx sy))
                    (incf ,i))
           (incf ,oo ,nlast)
           ,@(let ((s (funcall step a-mode)))
               (if (eq s 0) nil `((incf ,ao ,s))))
           ,@(let ((s (funcall step b-mode)))
               (if (eq s 0) nil `((incf ,bo ,s))))))
       t)))

(defmacro %sew2-dispatch (lt op a b res)
  "按形态组合派发二元 SIMD 循环；不支持时返回 NIL（调用方回退）。

   KIND 表把「算子符号」映射到「向量表达式种类」：算术四则一一对应，
   `%sq-diff-*`（方差的融合内核 `(x−y)²`）借用同一套循环，只是表达式不同。

   OP 只以**符号**形式拼进生成的代码（`(eq op '+)` 里的 op 是调用方
   `%sew2` 的形参），本宏体内并不求值它，故显式 IGNORE，免得零 WARNING 门禁
   把这条 STYLE-WARNING 当成新问题。"
  (declare (ignore op))
  (let* ((prefix (%sew-vec-prefix lt))
         (clauses
           (loop for (o kind) in '((+ :add) (- :sub) (* :mul) (/ :div)
                                   (%sq-diff-df :sqdiff) (%sq-diff-sf :sqdiff))
                 append
                 (cons `((and (eq op ',o) (eq am :full) (eq bm :full))
                         (%sew2-flat ,lt ,prefix ,kind ,a ,b ,res))
                       (loop for (x y) in '((:full :row) (:row :full)
                                            (:full :scalar) (:scalar :full)
                                            (:full :col) (:col :full)
                                            (:row :col) (:col :row))
                             collect
                             `((and (eq op ',o) (eq am ,x) (eq bm ,y))
                               (%sew2-rows ,lt ,prefix ,kind
                                           ,a ,b ,res ,x ,y)))))))
    `(let ((out-shape (vt-shape ,res)))
       (if (or (null out-shape) (zerop (vt-size ,res)))
           nil
           (let* ((nlast (%sew-nlast ,res))
                  (am (%sew-mode ,a out-shape nlast))
                  (bm (%sew-mode ,b out-shape nlast)))
             (cond
               ,@clauses
               (t nil)))))))

(defun %sew2 (op a b res)
  "二元逐元素 SIMD 入口。返回 T 表示已写满 RES；NIL 表示不接管。

   前置条件由调用方（vt-fast-map）保证：RES 连续、三个张量的元素类型相同、
   结果 dtype 不是 :bool。这里只再确认「元素类型是 f64/f32」与「形态可向量化」。"
  (let* ((lt (array-element-type (vt-data res)))
         (ok (and (equal lt (array-element-type (vt-data a)))
                  (equal lt (array-element-type (vt-data b))))))
    ;; 三类元素类型必须一致：下面两个宏用 `(the (simple-array LT (*)) ...)`
    ;; 硬声明缓冲区，类型不符时的行为属于未定义。这里显式判一次，
    ;; 不一致就交回调用方（铁律 2：快路径失败必须回退）。
    (cond ((not ok) nil)
          ((equal lt 'double-float) (%sew2-dispatch double-float op a b res))
          ((equal lt 'single-float) (%sew2-dispatch single-float op a b res))
          (t nil))))

;;; ------------------------------------------------------------------
;;; 一元循环（取负 / 绝对值 / 开方）
;;;
;;; 取负与绝对值用**符号位**运算而不是 0−x / 条件判断：
;;;   0.0 − (+0.0) 给 +0.0，而 (− +0.0) 必须是 −0.0；
;;;   比较式 abs 在 NaN 上会走分支，符号位清位则与 (abs x) 逐位相同。
;;;
;;; 开方不能直接用 `sqrtsd`：标量内核 `%op-sqrt-df` 对负数返回**规范 NaN**
;;; （CL 的 `(sqrt −1.0d0)` 给的是复数 `#C(0.0d0 1.0d0)`，库内一律换成
;;; `+vt-dfloat-nan+`），而硬件 `sqrtsd` 给的是 indefinite NaN
;;; （0xFFF8…，符号位为 1）。两者都「是 NaN」，位模式却不同，而快路径与
;;; 通用路径必须逐位相同。所以在向量侧补一次显式选择：
;;;     r = if (x < 0) then +vt-dfloat-nan+ else sqrt(x)
;;; 比较对 NaN 恒假（无序），因此 NaN 输入仍走 sqrt(NaN)（nan 载荷保留），
;;; −0.0 不小于 0，仍走 sqrt(−0.0) = −0.0 —— 与标量内核 `(minusp x)` 的
;;; 分支完全同构。尾块直接用标量内核的写法。
;;; ------------------------------------------------------------------

(defmacro %sew1-flat (lt op a res)
  "一元扁平循环。OP 为字面量（- / abs / sqrt），按元素类型展开成向量算子。"
  (let* ((n (gensym "N")) (i (gensym "I")) (lim (gensym "LIM"))
         (ad (gensym "AD")) (od (gensym "OD"))
         (ao (gensym "AO")) (oo (gensym "OO"))
         (sign (gensym "SIGN")) (zvec (gensym "ZV")) (nvec (gensym "NV"))
         (prefix (%sew-vec-prefix lt))
         (lanes (%sew-lanes prefix))
         (sign-bit (if (equal lt 'double-float) '-0.0d0 '-0.0s0))
         (pos-zero (if (equal lt 'double-float) '0.0d0 '0.0s0))
         (nan (if (equal lt 'double-float) '+vt-dfloat-nan+ '+vt-sfloat-nan+))
         (vecaref (%sew-sym prefix "-ROW-MAJOR-AREF"))
         (vecbcast (%sew-sym prefix "-BROADCAST"))
         (vx `(,vecaref ,ad (+ ,ao ,i)))
         (body (case op
                 (- `(,(%sew-sym prefix "-XOR") ,vx ,sign))
                 (abs `(,(%sew-sym prefix "-ANDC1") ,sign ,vx))
                 (sqrt `(,(%sew-sym prefix "-IF")
                         (,(%sew-sym prefix "<") ,vx ,zvec)
                         ,nvec
                         (,(%sew-sym prefix "-SQRT") ,vx)))))
         (tail (case op
                 (- `(- (aref ,ad (+ ,ao ,i))))
                 (abs `(abs (aref ,ad (+ ,ao ,i))))
                 (sqrt `(if (minusp (aref ,ad (+ ,ao ,i))) ,nan
                            (sqrt (aref ,ad (+ ,ao ,i))))))))
    `(let* ((,n (vt-size ,res))
            (,ad (the (simple-array ,lt (*)) (vt-data ,a)))
            (,od (the (simple-array ,lt (*)) (vt-data ,res)))
            (,ao (vt-offset ,a)) (,oo (vt-offset ,res))
            (,lim (- ,n (mod ,n ,lanes)))
            (,i 0)
            ,@(when (member op '(- abs))
                `((,sign (,vecbcast ,sign-bit))))
            ,@(when (eq op 'sqrt)
                `((,zvec (,vecbcast ,pos-zero))
                  (,nvec (,vecbcast ,nan)))))
       (declare (type fixnum ,n ,ao ,oo ,lim ,i)
                (type (simple-array ,lt (*)) ,ad ,od))
       (loop while (< ,i ,lim)
             do (setf (,vecaref ,od (+ ,oo ,i)) ,body)
                (incf ,i ,lanes))
       (loop while (< ,i ,n)
             do (setf (aref ,od (+ ,oo ,i)) ,tail)
                (incf ,i))
       t)))

(defun %sew1 (op a res)
  "一元逐元素 SIMD 入口（取负 / abs / sqrt）。返回 T 表示已写满 RES。"
  (let* ((lt (array-element-type (vt-data res)))
         (ok (equal lt (array-element-type (vt-data a)))))
    (cond ((not ok) nil)
          ((and (equal lt 'double-float) (member op '(- abs %op-sqrt-df)))
           (case op
             (- (%sew1-flat double-float - a res))
             (abs (%sew1-flat double-float abs a res))
             (%op-sqrt-df (%sew1-flat double-float sqrt a res))))
          ((and (equal lt 'single-float) (member op '(- abs %op-sqrt-sf)))
           (case op
             (- (%sew1-flat single-float - a res))
             (abs (%sew1-flat single-float abs a res))
             (%op-sqrt-sf (%sew1-flat single-float sqrt a res))))
          (t nil))))

;;; ------------------------------------------------------------------
;;; 选路外壳：形态不满足时返回 NIL（调用方回退，铁律 2）
;;; ------------------------------------------------------------------

(defun %sew2-try (op a b res)
  "二元 SIMD 的选路外壳。"
  (if (and (vt-contiguous-p res) (plusp (vt-size res)))
      (%sew2 op a b res)
      nil))

(defun %sew1-try (op a res)
  "一元 SIMD 的选路外壳。"
  (if (and (vt-contiguous-p res)
           (vt-contiguous-p a)
           (equal (vt-shape a) (vt-shape res))
           (plusp (vt-size res)))
      (%sew1 op a res)
      nil))

;;; 注册到 map-reduce 的钩子（§8.3）。写在文件末尾：本文件加载完成后钩子
;;; 才指向真实函数，调用点在运行期解引用，因此加载顺序不构成前向依赖。
(setf *simd-ew2-fn* #'%sew2-try
      *simd-ew1-fn* #'%sew1-try)
