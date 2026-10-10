;;;; io.lisp — 序列/数组与张量互转、打印

(in-package :clvt)

;;; ------------------------------------------------------------------
;;; 展平与嵌套
;;; ------------------------------------------------------------------

(defun vt-flatten-sequence (seq)
  "深度优先遍历 seq 及其嵌套序列，返回所有原子元素的列表（行主序）。"
  (with-float-safe
    (labels ((sequence-p (obj)
               (or (listp obj) (arrayp obj))))
      (if (not (sequence-p seq))
          (list seq)
          (let ((result '())
                (stack (list (cons seq (if (listp seq) seq 0)))))
            (loop
              (unless stack (return))
              (let* ((frame (pop stack))
                     (s (car frame))
                     (state (cdr frame)))
                (cond
                  ((listp s)
                   (when state
                     (let ((elt (car state)) (new-state (cdr state)))
                       (push (cons s new-state) stack)
                       (if (sequence-p elt)
                           (push (cons elt (if (listp elt) elt 0)) stack)
                           (push elt result)))))
                  ((arrayp s)
                   (let ((len (array-total-size s)))
                     (when (< state len)
                       (let ((elt (row-major-aref s state)))
                         (push (cons s (1+ state)) stack)
                         (if (sequence-p elt)
                             (push (cons elt (if (listp elt) elt 0)) stack)
                             (push elt result))))))
                  (t (error "~s 不是序列" s)))))
            (nreverse result))))))

(defun vt-from-sequence (contents &key (dtype :float64) (fast nil))
  "从嵌套序列创建张量（行主序）。支持任意维度规则嵌套；空序列 -> 形状 (0)。
   fast t 意味着直接使用 coerce 转换，可能报错
        nil 则用 vt-cast 安全转换 (默认)"
  (with-float-safe
    (labels
        ((infer-shape (seq)
           (typecase seq
             (list
              (if (null seq)
                  (list 0)
                  (let* ((first (car seq))
                         (rest-shape (typecase first
                                       (list (infer-shape first))
                                       (vector (infer-shape first))
                                       (t nil))))
                    (if rest-shape
                        (cons (length seq)
                              (loop for sub in (cdr seq)
                                    unless (equal (infer-shape sub) rest-shape)
                                      do (error "不规则嵌套")
                                    finally (return rest-shape)))
                        (progn
                          (loop for sub in (cdr seq)
                                when (or (listp sub) (typep sub 'vector))
                                  do (error "不规则嵌套"))
                          (list (length seq)))))))
             (vector
              (let ((len (length seq)))
                (if (zerop len) (list 0)
                    (let* ((first (aref seq 0))
                           (rest-shape (typecase first
                                         (list (infer-shape first))
                                         (vector (infer-shape first))
                                         (t nil))))
                      (if rest-shape
                          (cons len
                                (loop for i from 1 below len
                                      for sub = (aref seq i)
                                      unless (equal (infer-shape sub) rest-shape)
                                        do (error "不规则嵌套")
                                      finally (return rest-shape)))
                          (progn
                            (loop for i from 1 below len
                                  for sub = (aref seq i)
                                  when (or (listp sub) (typep sub 'vector))
                                    do (error "不规则嵌套"))
                            (list len)))))))
             (t (error "无法从 ~s 创建张量" seq))))
         (fill-tensor (data seq shape strides flat-idx)
           (if (null shape)
               (setf (aref data flat-idx)
                     (if fast
                         (coerce seq (vt-dtype->lisp-type dtype))
                         (vt-cast seq dtype)))
               (let ((stride (first strides)) (current flat-idx))
                 (typecase seq
                   (list
                    (dolist (elem seq)
                      (fill-tensor data elem (rest shape) (rest strides) current)
                      (incf current stride)))
                   (vector
                    (loop for elem across seq do
                      (fill-tensor data elem (rest shape) (rest strides) current)
                      (incf current stride)))
                   (t (error "fill-tensor: 不支持的序列类型")))))))
      (let* ((shape (infer-shape contents))
             (size (vt-shape-to-size shape))
             (lisp-type (vt-dtype->lisp-type dtype))
             (data (make-array size :element-type lisp-type
                                    :initial-element (coerce 0 lisp-type)))
             (strides (vt-compute-strides shape)))
        (fill-tensor data contents shape strides 0)
        (%make-vt :data data :shape shape :strides strides :offset 0 :dtype dtype)))))

(defun vt-flatten-to-nested (dims data)
  "将行主序一维 data 转换为符合 dims 的嵌套列表。"
  (let ((idx 0))
    (labels
        ((recurse (dims)
           (if (null dims)
               (prog1 (aref data idx) (incf idx))
               (let ((n (first dims)) (result nil))
                 (dotimes (i n)
                   (declare (fixnum i))
                   (push (recurse (rest dims)) result))
                 (nreverse result)))))
      (recurse dims))))

(defun vt-to-list (vt)
  "将张量转换为嵌套列表，正确处理任意 strides/offset 的视图。"
  (labels
      ((build (shape strides offset data)
         (if (null shape)
             (aref data offset)
             (let ((dim (first shape))
                   (stride (first strides))
                   (result nil))
               (loop for i fixnum from (1- dim) downto 0
                     for sub = (+ offset (* i stride))
                     do (push (build (rest shape) (rest strides) sub data)
                              result))
               result))))
    (let ((shape (vt-shape vt))
          (strides (vt-strides vt))
          (offset (vt-offset vt)) (data (vt-data vt)))
      (if shape
          (build shape strides offset data)
          (aref data offset)))))

(defun vt-to-array (vt &key dtype (fast nil))
  "将张量转换为原生多维数组。使用 vt-do-each 遍历，同时维护逻辑坐标。
   fast t 意味着直接使用 coerce 转换，可能报错
        nil 则用 vt-cast 安全转换 (默认)"
  (unless dtype (setf dtype (vt-dtype vt)))
  (let ((shape (vt-shape vt))
        (lisp-type (vt-dtype->lisp-type dtype)))
    (if (null shape)
        ;; 标量：0 维数组
        (make-array nil
                    :initial-element
                    (vt-cast (aref (vt-data vt) (vt-offset vt)) dtype)
                    :element-type lisp-type)
        ;; 非标量
        (let* ((rank (length shape))
               (dims (coerce shape 'simple-vector))          ; 各维度大小
               (arr (make-array shape :element-type lisp-type))
               (coords (make-list rank :initial-element 0))) ; 初始坐标全 0
          (vt-do-each (ptr val vt)
            (declare (ignore val))
            ;; 使用当前坐标设置目标数组
            (setf (apply #'aref arr coords)
                  (if fast
                      (coerce (aref (vt-data vt) ptr) lisp-type)
                      (vt-cast (aref (vt-data vt) ptr) dtype)))
            ;; 更新坐标到下一个逻辑位置（C 顺序）
            (let ((i (1- rank)))
              (loop
                (incf (nth i coords))                ; 当前位 +1
                (when (< (nth i coords) (svref dims i))
                  (return))                           ; 未溢出，更新完成
                (setf (nth i coords) 0)              ; 溢出归零，进位
                (decf i)
                (when (< i 0) (return)))))           ; 所有位都溢出，遍历结束
          arr))))

;;; ------------------------------------------------------------------
;;; 打印
;;; ------------------------------------------------------------------

(defvar *vt-print-threshold* 3 "超过该数量后打印开始省略")
(defvar *vt-print-precision* 6 "浮点打印精度")
(defvar *vt-indent-step* 1 "缩进步长")

(defun %type-category (type)
  (cond ((or (eq type 'fixnum)
             (eq type 'integer)
             (eq type 'bit)
             (and (listp type)
                  (member (first type) '(signed-byte unsigned-byte))))
         :integer)
        ((member type '(single-float double-float short-float long-float float))
         :float)
        (t :other)))

(defun %format-number (val type)
  (case (%type-category type)
    (:integer (format nil "~d" val))
    (:float
     (let* ((str (format nil "~,vf" *vt-print-precision* val))
            (trimmed (string-right-trim "0" str)))
       (when (and (> (length trimmed) 0)
                  (char= (char trimmed (1- (length trimmed))) #\.))
         (setf trimmed (concatenate 'string trimmed "0")))
       trimmed))
    (otherwise (format nil "~a" val))))

(defun %phys-idx (vt indices)
  (loop with strides = (vt-strides vt)
        with offset = (vt-offset vt)
        for idx in indices
        for stride in strides
        sum (* idx stride) into res
        finally (return (+ res offset))))

(defun print-vt-recursive
    (vt axis current-indices base-indent col-width element-type stream)
  (let* ((shape (vt-shape vt))
         (rank (length shape))
         (dim-size (nth axis shape))
         (is-last-axis (= axis (1- rank)))
         (truncated-p (> dim-size (* 2 *vt-print-threshold*)))
         (edge *vt-print-threshold*)
         (current-level-indent (+ base-indent (* (1+ axis) *vt-indent-step*))))
    (write-char #\[ stream)
    (flet ((print-item (idx)
             (if is-last-axis
                 (let* ((phys-idx (%phys-idx vt (append current-indices (list idx))))
                        (val (aref (vt-data vt) phys-idx))
                        (str (%format-number val element-type)))
                   (format stream "~v@a" col-width str))
                 (print-vt-recursive vt (1+ axis) (append current-indices (list idx))
                                     base-indent col-width element-type stream))))
      (cond
        ((not truncated-p)
         (loop for i from 0 below dim-size
               when (> i 0)
                 do (if is-last-axis
                        (write-string ", " stream)
                        (format stream ",~%~v@a" current-level-indent ""))
               do (print-item i)))
        (t
         (loop for i from 0 below edge
               when (> i 0)
                 do (if is-last-axis
                        (write-string ", " stream)
                        (format stream ",~%~v@a" current-level-indent ""))
               do (print-item i))
         (if is-last-axis
             (format stream ", ...")
             (format stream ",~%~v@a..." current-level-indent ""))
         (loop for i from (- dim-size edge) below dim-size
               do
                  (progn (if is-last-axis
                             (write-string ", " stream)
                             (format stream ",~%~v@a" current-level-indent ""))
                         (print-item i))))))
    (write-char #\] stream)))

(defmethod print-object ((obj vt) stream)
  (print-unreadable-object (obj stream :type t :identity nil)
    (let ((shape (vt-shape obj)) (element-type (vt-element-type obj)))
      (format stream "shape:~a dtype:~a " shape element-type)
      (cond
        ((and shape (zerop (reduce #'* shape :initial-value 1)))
         (format stream "[] (empty)"))
        ((null shape)
         (format stream "~a"
                 (%format-number (aref (vt-data obj) (vt-offset obj)) element-type)))
        (t
         (let ((max-width 0))
           (labels ((visible-indices (dim)
                      "返回 print-vt-recursive 在给定 dim 下真正会打印的轴索引列表。
                       与 print-vt-recursive 的截断规则严格一致：
                       截断时 = 头 threshold 个 ∪ 尾 threshold 个；否则 = 全部。"
                      (let ((edge *vt-print-threshold*))
                        (if (> dim (* 2 edge))
                            (append (loop for i from 0 below edge collect i)
                                    (loop for i from (- dim edge) below dim collect i))
                            (loop for i from 0 below dim collect i))))
                    (scan-axis (current-idxs axis)
                      "递归遍历，把实际会被打印的每个元素都扫一遍，累计 max-width。"
                      (let* ((dim (nth axis shape))
                             (is-last (= axis (1- (length shape)))))
                        (dolist (i (visible-indices dim))
                          (if is-last
                              (let* ((phys (%phys-idx obj (append current-idxs (list i))))
                                     (w (length (%format-number
                                                 (aref (vt-data obj) phys)
                                                 element-type))))
                                (setf max-width (max max-width w)))
                              (scan-axis (append current-idxs (list i))
                                         (1+ axis)))))))
             (scan-axis nil 0))
           (incf max-width 1)
           (fresh-line stream)
           (format stream "  ")
           (print-vt-recursive obj 0 nil 2 max-width element-type stream)))))))

(defun vt-set-print-options (&key threshold precision indent-step)
  "设置张量打印选项（对标 numpy.set_printoptions + torch.set_printoptions）。"
  (when threshold (setf *vt-print-threshold* threshold))
  (when precision (setf *vt-print-precision* precision))
  (when indent-step (setf *vt-indent-step* indent-step))
  (values))

(defun vt-get-print-options ()
  "返回当前打印选项的副本（便于临时保存/恢复）。"
  (list *vt-print-threshold* *vt-print-precision* *vt-indent-step*))

;;; ==================================================================
;;; 持久化与互操作：vt-save / vt-load（对标 numpy.save / numpy.load）
;;; ==================================================================
;;;
;;; 两种格式，由 AS 关键字选择：
;;;   :lisp — clvt 原生 s-expression 文本格式（Lisp reader 严格往返）：
;;;           (:CLVT-TENSOR :SHAPE (2 3) :DTYPE :FLOAT64 :DATA #(1.0d0 ...))
;;;           DATA 按行主序（C-order）展平；非有限浮点编码为关键字
;;;           :NAN / :INF / :NEG-INF（有限浮点原样可读，-0.0 符号位保留）。
;;;   :npy  — NumPy .npy 二进制格式（v1.0，头超长自动升 v2.0），
;;;           C-order、小端，与 np.save / np.load 双向互操作。
;;;
;;; 实现说明：
;;;   - 全部字节 I/O 走 (unsigned-byte 8) 缓冲 + 流；浮点用 IEEE-754 手工
;;;     编解码（SBCL 不支持 float element-type 的文件流），显式小端，
;;;     不依赖实现内部符号，与 .npy 规范一致。
;;;   - 保存时按行主序逻辑顺序（机制 A：stride 驱动遍历）物化任意视图
;;;     （转置/切片/广播 stride=0）；加载结果恒为 C 连续。
;;;   - 读取 lisp 格式时关闭 *read-eval*（文件内不执行任何代码，
;;;     对标 numpy.load 默认 allow_pickle=False 的安全立场）。
;;;   - npy 格式 NaN 逐位保留（符号位与 payload，SAP/位模式重解释，
;;;     对标 numpy.save 的位级语义）；:lisp 文本格式不携带 payload，
;;;     NaN 映射为库内 NaN 常量（与对方 np.save 的文本格式行为一致）。
;;;   - ±Inf 与 -0.0 位模式精确保留。
;;;   - 本实现按 .npy 规范输出小端（'<' descr）；大端文件确定性报错。

;;; ------------------------------------------------------------------
;;; IEEE-754 编解码（binary64 / binary32，可移植实现）
;;; ------------------------------------------------------------------

(declaim (inline %f64-bits %bits->f64 %f32-bits %bits->f32))

(defun %f64-bits (x)
  "double-float → IEEE-754 binary64 位模式（非负整数，按小端写出）。
   NaN 逐位保留（符号位与 payload，直接取物理位模式；对标 numpy 语义）。
   调用方需处于 with-float-safe 上下文（NaN/Inf 判定要求屏蔽陷阱）。"
  (declare (type double-float x))
  (cond ((%nan-p x)
         (logand (sb-kernel:double-float-bits x) #xFFFFFFFFFFFFFFFF))
        ((%pos-inf-p x) #x7FF0000000000000)
        ((%neg-inf-p x) #xFFF0000000000000)
        (t
         (multiple-value-bind (sig exp sign) (integer-decode-float x)
           (let ((sign-bit (if (minusp sign) #x8000000000000000 0)))
             (cond ((zerop sig) sign-bit)
                   ((>= sig (ash 1 52))
                    (logior sign-bit
                            (ash (+ exp 1075) 52)
                            (logand (- sig (ash 1 52)) #xFFFFFFFFFFFFF)))
                   (t
                    (logior sign-bit sig))))))))

(defun %bits->nan-f64 (bits)
  "NaN 位模式 → double-float：符号位与 payload 逐位还原（SAP 重解释，
   与文件字节往返完全等价；不涉及任何浮点比较，无需屏蔽陷阱）。"
  (declare (type (unsigned-byte 64) bits))
  (let ((buf (make-array 8 :element-type '(unsigned-byte 8))))
    (declare (type (simple-array (unsigned-byte 8) (8)) buf))
    (dotimes (i 8)
      (setf (aref buf i) (ldb (byte 8 (* 8 i)) bits)))
    (sb-sys:with-pinned-objects (buf)
      (sb-sys:sap-ref-double (sb-sys:vector-sap buf) 0))))

(defun %bits->f64 (bits)
  "IEEE-754 binary64 位模式 → double-float（精确，无舍入）。
   NaN 按位模式逐位还原（含符号位与 payload）；±Inf / 有限值精确重建。"
  (declare (type (unsigned-byte 64) bits))
  (let ((sign (if (logbitp 63 bits) -1 1))
        (e (ldb (byte 11 52) bits))
        (frac (ldb (byte 52 0) bits)))
    (cond ((= e #x7FF)
           (if (zerop frac)
               (if (minusp sign) +vt-dfloat-neg-inf+ +vt-dfloat-pos-inf+)
               (%bits->nan-f64 bits)))
          ((zerop e)
           (* sign (scale-float (float frac 1.0d0) -1074)))
          (t
           (* sign (scale-float (float (+ frac (ash 1 52)) 1.0d0)
                                (- e 1075)))))))

(defun %f32-bits (x)
  "single-float → IEEE-754 binary32 位模式。
   NaN 逐位保留（符号位与 payload，直接取物理位模式；对标 numpy 语义）。
   调用方需处于 with-float-safe 上下文。"
  (declare (type single-float x))
  (cond ((%nan-p x)
         (logand (sb-kernel:single-float-bits x) #xFFFFFFFF))
        ((%pos-inf-p x) #x7F800000)
        ((%neg-inf-p x) #xFF800000)
        (t
         (multiple-value-bind (sig exp sign) (integer-decode-float x)
           (let ((sign-bit (if (minusp sign) #x80000000 0)))
             (cond ((zerop sig) sign-bit)
                   ((>= sig (ash 1 23))
                    (logior sign-bit
                            (ash (+ exp 150) 23)
                            (logand (- sig (ash 1 23)) #x7FFFFF)))
                   (t
                    (logior sign-bit sig))))))))

(defun %bits->nan-f32 (bits)
  "NaN 位模式 → single-float：符号位与 payload 逐位还原（SAP 重解释）。"
  (declare (type (unsigned-byte 32) bits))
  (let ((buf (make-array 4 :element-type '(unsigned-byte 8))))
    (declare (type (simple-array (unsigned-byte 8) (4)) buf))
    (dotimes (i 4)
      (setf (aref buf i) (ldb (byte 8 (* 8 i)) bits)))
    (sb-sys:with-pinned-objects (buf)
      (sb-sys:sap-ref-single (sb-sys:vector-sap buf) 0))))

(defun %bits->f32 (bits)
  "IEEE-754 binary32 位模式 → single-float（精确，无舍入）。
   NaN 按位模式逐位还原（含符号位与 payload）；±Inf / 有限值精确重建。"
  (declare (type (unsigned-byte 32) bits))
  (let ((sign (if (logbitp 31 bits) -1 1))
        (e (ldb (byte 8 23) bits))
        (frac (ldb (byte 23 0) bits)))
    (cond ((= e #xFF)
           (if (zerop frac)
               (if (minusp sign) +vt-sfloat-neg-inf+ +vt-sfloat-pos-inf+)
               (%bits->nan-f32 bits)))
          ((zerop e)
           (* sign (scale-float (float frac 1.0s0) -149)))
          (t
           (* sign (scale-float (float (+ frac (ash 1 23)) 1.0s0)
                                (- e 150)))))))

;;; ------------------------------------------------------------------
;;; 小端字节装配
;;; ------------------------------------------------------------------

(declaim (inline %set-uint-le! %uint-le-at))

(defun %set-uint-le! (buf off value nbytes)
  "把非负整数 VALUE 以小端序写入 BUF 的 OFF 起 NBYTES 个字节。"
  (declare (type (simple-array (unsigned-byte 8) (*)) buf)
           (type fixnum off nbytes))
  (dotimes (i nbytes)
    (setf (aref buf (+ off i)) (ldb (byte 8 (* 8 i)) value))))

(defun %uint-le-at (buf off nbytes)
  "从 BUF 的 OFF 起按小端读取 NBYTES 字节，组装为非负整数。"
  (declare (type (simple-array (unsigned-byte 8) (*)) buf)
           (type fixnum off nbytes))
  (loop with u of-type (unsigned-byte 64) = 0
        for i of-type fixnum from 0 below nbytes
        do (setf u (logior u (ash (aref buf (+ off i)) (* 8 i))))
        finally (return u)))

;;; ------------------------------------------------------------------
;;; npy 元素编码 / 解码（位模式 ↔ 张量元素）
;;; ------------------------------------------------------------------

(defun %npy-encode-bits (dtype x)
  "张量元素 → npy 位模式（非负整数）。整型即补码位模式。
   调用方需处于 with-float-safe 上下文。"
  (ecase dtype
    (:float64 (%f64-bits x))
    (:float32 (%f32-bits x))
    (:int64   (logand x #xFFFFFFFFFFFFFFFF))
    (:int32   (logand x #xFFFFFFFF))
    (:int16   (logand x #xFFFF))
    (:int8    (logand x #xFF))
    ((:uint8 :uint16) x)))

(defun %npy-decode-bits (dtype u)
  "npy 位模式 → 张量元素（类型与 dtype 的物理元素类型一致）。"
  (ecase dtype
    (:float64 (%bits->f64 u))
    (:float32 (%bits->f32 u))
    (:int64   (if (logbitp 63 u) (- u (ash 1 64)) u))
    (:int32   (if (logbitp 31 u) (- u (ash 1 32)) u))
    (:int16   (if (logbitp 15 u) (- u (ash 1 16)) u))
    (:int8    (if (logbitp 7 u) (- u 256) u))
    ((:uint8 :uint16) u)))

;;; ------------------------------------------------------------------
;;; npy dtype ↔ descr
;;; ------------------------------------------------------------------

(defun %npy-dtype->descr (dtype)
  "clvt dtype → npy descr 字符串。1 字节类型用 '|' 前缀（与 numpy 写法一致）。"
  (ecase dtype
    (:float64 "<f8")
    (:float32 "<f4")
    (:int64 "<i8")
    (:int32 "<i4")
    (:int16 "<i2")
    (:int8 "|i1")
    (:uint8 "|u1")
    (:uint16 "<u2")))

(defun %npy-descr->dtype (descr)
  "npy descr 字符串 → clvt dtype；不支持/非法时确定性报错（L2 契约）。"
  (labels ((fail (why)
             (error "vt-load: npy dtype '~a' 不受支持（~a）；支持 f8/f4/i8/i4/i2/i1/u2/u1"
                    descr why)))
    (when (or (not (stringp descr)) (zerop (length descr)))
      (fail "descr 为空"))
    (let ((c0 (char descr 0))
          (pos 0))
      (when (find c0 "<=|>" :test #'char=)
        (when (char= c0 #\>)
          (fail "大端字节序不在支持范围（本实现按 .npy 规范输出小端）"))
        (incf pos))
      (when (>= pos (length descr))
        (fail "缺少类型码"))
      (let ((code (char descr pos))
            (tail (subseq descr (1+ pos))))
        (let ((size (or (ignore-errors (parse-integer tail))
                        (fail "itemsize 非数字"))))
          (or (cond ((char= code #\f) (case size (8 :float64) (4 :float32)))
                    ((char= code #\i) (case size (8 :int64) (4 :int32)
                                        (2 :int16) (1 :int8)))
                    ((char= code #\u) (case size (2 :uint16) (1 :uint8))))
              (fail "类型码/宽度不在支持矩阵内")))))))

;;; ------------------------------------------------------------------
;;; npy header 构造
;;; ------------------------------------------------------------------

(defun %npy-shape-text (shape)
  "clvt shape → numpy header shape 元组文本：
   NIL → \"()\"；(5) → \"(5,)\"；(2 3) → \"(2, 3)\"。"
  (if (null shape)
      "()"
      (format nil "(~{~a~^, ~}~a)" shape (if (= (length shape) 1) "," ""))))

(defun %npy-write-header (stream descr shape)
  "写入 .npy magic、版本与 64 字节对齐的 header dict（与 numpy 输出逐字节一致）。
   返回数据区起点字节偏移。头总长放不下 2 字节长度域时自动升级 v2.0。"
  (let* ((dict (format nil "{'descr': '~a', 'fortran_order': False, 'shape': ~a, }"
                       descr (%npy-shape-text shape)))
         (major (if (> (+ 10 (length dict)) 65535) 2 1))
         (hlen-size (if (= major 1) 2 4))
         (pad (mod (- 64 (mod (+ 9 hlen-size (length dict)) 64)) 64))
         (hlen (+ (length dict) pad 1)))
    (when (> hlen 65535)
      (error "vt-save: npy header 过长（~a 字节），张量秩异常" hlen))
    (dotimes (i 6)
      (write-byte (aref #(147 78 85 77 80 89) i) stream))
    (write-byte major stream)
    (write-byte 0 stream)
    (dotimes (i hlen-size)
      (write-byte (ldb (byte 8 (* 8 i)) hlen) stream))
    (dotimes (i (length dict))
      (write-byte (char-code (char dict i)) stream))
    (dotimes (i pad)
      (write-byte 32 stream))
    (write-byte 10 stream)
    (+ 8 hlen-size hlen)))

;;; ------------------------------------------------------------------
;;; npy header 解析
;;; ------------------------------------------------------------------

(defun %npy-magic-p (buf)
  "判定 BUF 前 6 字节是否为 \\x93NUMPY magic。"
  (and (>= (length buf) 6)
       (= (aref buf 0) 147) (= (aref buf 1) 78) (= (aref buf 2) 85)
       (= (aref buf 3) 77) (= (aref buf 4) 80) (= (aref buf 5) 89)))

(defun %npy-dict-value (dict key)
  "在 numpy header dict 文本中查找 KEY 的值文本（原始子串），找不到返回 NIL。
   键与字符串值接受单引号或双引号（对非 numpy 的第三方写出器更宽容）；
   值形态：'字符串' / \"字符串\" / (元组) / True / False。与 numpy 写出的
   header 完全兼容。"
  (let ((kstart (or (search (format nil "'~a'" key) dict)
                    (search (format nil "\"~a\"" key) dict))))
    (when kstart
      (let ((colon (position #\: dict :start (+ kstart (length key) 2))))
        (when colon
          (let ((i (position-if (lambda (c) (char/= c #\space)) dict
                                :start (1+ colon))))
            (when i
              (let ((c (char dict i)))
                (cond ((or (char= c #\') (char= c #\"))
                       (let ((end (position c dict :start (1+ i))))
                         (and end (subseq dict (1+ i) end))))
                      ((char= c #\()
                       (let ((end (position #\) dict :start i)))
                         (and end (subseq dict i (1+ end)))))
                      (t
                       (let ((end (or (position #\, dict :start i)
                                      (position #\} dict :start i)
                                      (length dict))))
                         (string-trim " " (subseq dict i end)))))))))))))

(defun %npy-parse-shape (text)
  "numpy shape 元组文本 → clvt shape：
   \"()\" → NIL，\"(5,)\" → (5)，\"(2, 3)\" → (2 3)。"
  (unless (and (> (length text) 1)
               (char= (char text 0) #\()
               (char= (char text (1- (length text))) #\)))
    (error "vt-load: npy header shape 非法：~a" text))
  (let ((inner (string-trim " " (subseq text 1 (1- (length text))))))
    (if (zerop (length inner))
        nil
        (mapcar #'%npy-parse-dim (%npy-split-dims inner)))))

(defun %npy-split-dims (inner)
  "把 shape 元组内部文本按逗号切分（丢弃空 token，容忍尾逗号）。"
  (let ((tokens '())
        (start 0))
    (dotimes (i (1+ (length inner)))
      (when (or (= i (length inner)) (char= (char inner i) #\,))
        (let ((tok (string-trim " " (subseq inner start i))))
          (unless (zerop (length tok))
            (push tok tokens)))
        (setf start (1+ i))))
    (nreverse tokens)))

(defun %npy-parse-dim (tok)
  "单个 shape 维度文本 → 非负整数。"
  (let ((n (or (ignore-errors (parse-integer tok))
               (error "vt-load: npy header shape 含非整数维度：~a" tok))))
    (when (minusp n)
      (error "vt-load: npy header shape 维度必须 >= 0，收到 ~a" n))
    n))

(defun %npy-parse-fortran-order (text)
  "fortran_order 字段文本 → 布尔值。"
  (cond ((string= text "True") t)
        ((string= text "False") nil)
        (t (error "vt-load: npy header fortran_order 非法：~a" text))))

;;; ------------------------------------------------------------------
;;; npy 数据区编码 / 解码
;;; ------------------------------------------------------------------

(defun %npy-encode-data-buffer (vt)
  "把张量按行主序逻辑顺序编码为 npy 数据区字节向量（小端、紧凑）。"
  (let* ((dtype (vt-dtype vt))
         (itemsize (vt-dtype-itemsize dtype))
         (n (vt-size vt))
         (buf (make-array (* n itemsize)
                          :element-type '(unsigned-byte 8)
                          :initial-element 0)))
    (with-float-safe
      (let ((k 0))
        (declare (fixnum k))
        (vt-do-each (ptr val vt)
          (declare (ignore ptr))
          (%set-uint-le! buf (* k itemsize) (%npy-encode-bits dtype val) itemsize)
          (incf k))))
    buf))

(defun %npy-f-strides (shape)
  "Fortran（列主序）步长：f-strides[0]=1，f-strides[i]=f-strides[i-1]×shape[i-1]。"
  (let ((acc 1)
        (result '()))
    (dolist (d shape (nreverse result))
      (push acc result)
      (setf acc (* acc d)))))

(defun %make-vt-from-raw (raw shape dtype fortran-order)
  "把按文件顺序解码的元素向量 RAW 组装为 C 连续 vt。
   fortran-order=T 时 RAW 为列主序，按列主序步长散布到行主序缓冲。"
  (let* ((count (length raw))
         (buf (make-array count
                          :element-type (vt-dtype->lisp-type dtype)
                          :initial-element (vt-dtype-default-value dtype))))
    (if fortran-order
        (%scatter-forder raw buf shape count)
        (dotimes (k count)
          (setf (aref buf k) (aref raw k))))
    (%make-vt :data buf
              :shape shape
              :strides (vt-compute-strides shape)
              :offset 0
              :dtype dtype)))

(defun %scatter-forder (raw buf shape count)
  "把列主序的 RAW 散布到行主序缓冲 BUF（npy fortran_order=True 加载路径）。"
  (let* ((rank (length shape))
         (fstr (coerce (%npy-f-strides shape) 'simple-vector))
         (dims (coerce shape 'simple-vector))
         (idx (make-array rank :element-type 'fixnum :initial-element 0)))
    (dotimes (k count)
      (setf (aref buf k) (aref raw (%forder-pos idx fstr rank)))
      (when (< k (1- count))
        (%advance-idx idx dims rank)))))

(defun %forder-pos (idx fstr rank)
  "当前逻辑 multi-index IDX 的列主序线性位置。"
  (let ((fpos 0))
    (declare (fixnum fpos))
    (dotimes (d rank)
      (incf fpos (* (aref idx d) (aref fstr d))))
    fpos))

(defun %advance-idx (idx dims rank)
  "IDX 前进到下一个行主序位置。"
  (let ((d (1- rank)))
    (loop
      (incf (aref idx d))
      (if (< (aref idx d) (aref dims d))
          (return)
          (progn
            (setf (aref idx d) 0)
            (decf d))))))

(defun %npy-decode-data (oct dtype count)
  "npy 数据区字节缓冲 → 已解码元素向量（按文件顺序）。"
  (let ((raw (make-array count))
        (itemsize (vt-dtype-itemsize dtype)))
    (with-float-safe
      (dotimes (k count)
        (setf (aref raw k)
              (%npy-decode-bits dtype
                                (%uint-le-at oct (* k itemsize) itemsize)))))
    raw))

;;; ------------------------------------------------------------------
;;; npy 文件读取（分步，避免深嵌套）
;;; ------------------------------------------------------------------

(defun %read-n-octets (s n what)
  "从二进制流读入恰好 N 字节；不足则确定性报错。"
  (let ((buf (make-array n :element-type '(unsigned-byte 8) :initial-element 0)))
    (when (< (read-sequence buf s) n)
      (error "vt-load: npy ~a 被截断" what))
    buf))

(defun %npy-read-header (s path)
  "读取并校验 npy magic/版本/header dict。
   返回 (values dict-string major hlen)；流定位在数据区起点。"
  (let ((magic (%read-n-octets s 6 "magic")))
    (unless (%npy-magic-p magic)
      (error "vt-load: ~a 不是有效的 npy 文件（\\x93NUMPY magic 不匹配）" path))
    (let ((major (read-byte s))
          (minor (read-byte s)))
      (unless (member major '(1 2 3))
        (error "vt-load: 不支持的 npy 版本 ~a.~a（支持 1.0/2.0/3.0）" major minor))
      (let* ((hlen (if (= major 1)
                       (logior (read-byte s) (ash (read-byte s) 8))
                       (logior (read-byte s) (ash (read-byte s) 8)
                               (ash (read-byte s) 16) (ash (read-byte s) 24))))
             (hdr-buf (%read-n-octets s hlen "header")))
        (values (map 'string #'code-char hdr-buf) major hlen)))))

(defun %npy-check-data-length (total data-start needed descr shape)
  "校验数据区长度：不足则确定性报错。"
  (let ((available (- total data-start)))
    (when (< available needed)
      (error "vt-load: npy 数据区不完整：shape ~a / ~a 需要 ~a 字节，文件只有 ~a 字节"
             (or shape :0-d) descr needed (max 0 available)))))

(defun %load-npy-format (path)
  "从 NumPy .npy 文件加载张量（v1.0/v2.0/v3.0 头；fortran_order=True
   自动按列主序重排；结果恒为 C 连续 vt）。"
  (with-open-file (s path :direction :input :element-type '(unsigned-byte 8))
    (multiple-value-bind (dict major hlen) (%npy-read-header s path)
      (let* ((total (file-length s))
             (data-start (+ 8 (if (= major 1) 2 4) hlen))
             (descr (or (%npy-dict-value dict "descr")
                        (error "vt-load: npy header 缺少 'descr' 字段")))
             (fo-text (or (%npy-dict-value dict "fortran_order")
                          (error "vt-load: npy header 缺少 'fortran_order' 字段")))
             (shape-text (or (%npy-dict-value dict "shape")
                             (error "vt-load: npy header 缺少 'shape' 字段")))
             (dtype (%npy-descr->dtype descr))
             (fortran-order (%npy-parse-fortran-order fo-text))
             (shape (%npy-parse-shape shape-text))
             (count (reduce #'* shape))
             (needed (* count (vt-dtype-itemsize dtype))))
        (%npy-check-data-length total data-start needed descr shape)
        (let ((oct (%read-n-octets s needed "数据区")))
          (%make-vt-from-raw (%npy-decode-data oct dtype count)
                             shape dtype fortran-order))))))

;;; ------------------------------------------------------------------
;;; lisp 格式（s-expression）
;;; ------------------------------------------------------------------

(defun %lisp-data-vector (vt)
  "行主序逻辑数据向量；非有限浮点编码为 :NAN / :INF / :NEG-INF 关键字
   （Lisp reader 严格往返；有限浮点以可读数字原样存储）。"
  (let ((dtype (vt-dtype vt))
        (vec (make-array (vt-size vt) :initial-element 0)))
    (with-float-safe
      (let ((k 0))
        (declare (fixnum k))
        (vt-do-each (ptr val vt)
          (declare (ignore ptr))
          (setf (aref vec k) (%encode-lisp-element dtype val))
          (incf k))))
    vec))

(defun %encode-lisp-element (dtype x)
  "张量元素 → lisp 格式元素：非有限浮点编码为关键字，其余原样。"
  (if (member dtype '(:float64 :float32))
      (with-float-safe
        (cond ((%nan-p x) :nan)
              ((%pos-inf-p x) :inf)
              ((%neg-inf-p x) :neg-inf)
              (t x)))
      x))

(defun %save-lisp-format (vt path)
  "把张量写为 clvt lisp 格式文本文件（单个可读 s-expression）。"
  (with-open-file (s path :direction :output :if-exists :supersede
                          :if-does-not-exist :create)
    (let ((*print-readably* t)
          (*print-pretty* nil)
          (*print-length* nil)
          (*print-level* nil))
      (prin1 (list :clvt-tensor
                   :version 1
                   :shape (vt-shape vt)
                   :dtype (vt-dtype vt)
                   :data (%lisp-data-vector vt))
             s)
      (terpri s))))

(defun %decode-lisp-element (dtype v)
  "lisp 格式元素 → dtype 物理元素值：
   浮点 dtype 接受 :nan/:inf/:neg-inf 标记与数字；整型 dtype 仅接受整数。"
  (cond ((member dtype '(:float64 :float32))
         (cond ((eq v :nan) (vt-get-nan dtype))
               ((eq v :inf) (vt-get-pos-inf dtype))
               ((eq v :neg-inf) (vt-get-neg-inf dtype))
               ((numberp v) (funcall (vt-cast-fun dtype) v))
               (t (error "vt-load: :data 元素必须是数字或非有限标记（:nan/:inf/:neg-inf），收到 ~a" v))))
        ((integerp v)
         (funcall (vt-cast-fun dtype) v))
        (t
         (error "vt-load: 整型 dtype 的 :data 元素必须是整数，收到 ~a" v))))

(defun %validate-lisp-form (form path)
  "校验 lisp 格式顶层结构，返回 (values shape dtype data)。
   兼容无 :VERSION 的旧文件；带 :VERSION 时仅支持版本 1。"
  (unless (and (consp form) (eq (car form) :clvt-tensor))
    (error "vt-load: ~a 不是有效的 clvt lisp 格式（缺少 :CLVT-TENSOR 标签）" path))
  (let ((version (getf (cdr form) :version 1)))
    (unless (eql version 1)
      (error "vt-load: lisp 格式版本不支持：~s（本实现支持版本 1）" version)))
  (let ((shape (getf (cdr form) :shape :%missing%))
        (dtype (getf (cdr form) :dtype :%missing%))
        (data (getf (cdr form) :data :%missing%)))
    (when (eq shape :%missing%)
      (error "vt-load: lisp 格式缺少 :shape 字段"))
    (when (eq dtype :%missing%)
      (error "vt-load: lisp 格式缺少 :dtype 字段"))
    (when (eq data :%missing%)
      (error "vt-load: lisp 格式缺少 :data 字段"))
    (unless (and (listp shape)
                 (every (lambda (d) (and (integerp d) (>= d 0))) shape))
      (error "vt-load: :shape 必须是非负整数列表，收到 ~a" shape))
    (unless (member dtype *vt-dtypes*)
      (error "vt-load: :dtype 必须是 ~s 之一，收到 ~a" *vt-dtypes* dtype))
    (unless (or (vectorp data) (listp data))
      (error "vt-load: :data 必须是向量或列表，收到 ~a" data))
    (values shape dtype data)))

(defun %load-lisp-format (path)
  "从 clvt lisp 格式文件加载张量（*read-eval* 关闭，文件内容不执行）。
   注意 *package* 必须保持默认：绑到 :keyword 会把文件中的 NIL 读成 :NIL，
   破坏 0-d 张量（shape 为 NIL）的往返。"
  (let ((form (let ((*read-eval* nil))
                (with-open-file (s path :direction :input)
                  (read s nil :%eof%)))))
    (when (eq form :%eof%)
      (error "vt-load: ~a 不是有效的 clvt lisp 格式（空文件）" path))
    (multiple-value-bind (shape dtype data) (%validate-lisp-form form path)
      (let* ((count (reduce #'* shape))
             (elems (coerce data 'list)))
        (unless (= (length elems) count)
          (error "vt-load: :data 长度 ~a 与 shape ~a 期望的 ~a 不符"
                 (length elems) (or shape :0-d) count))
        (%make-vt-from-list elems shape dtype)))))

(defun %make-vt-from-list (elems shape dtype)
  "按行主序元素列表构造 C 连续 vt。"
  (let ((buf (make-array (length elems)
                         :element-type (vt-dtype->lisp-type dtype)
                         :initial-element (vt-dtype-default-value dtype))))
    (loop for v in elems
          for k of-type fixnum from 0
          do (setf (aref buf k) (%decode-lisp-element dtype v)))
    (%make-vt :data buf
              :shape shape
              :strides (vt-compute-strides shape)
              :offset 0
              :dtype dtype)))

;;; ------------------------------------------------------------------
;;; 保存入口
;;; ------------------------------------------------------------------

(defun %save-npy-format (vt path)
  "把张量写为 NumPy .npy 文件（C-order、小端，与 np.save 输出逐字节一致）。"
  (with-open-file (s path :direction :output :if-exists :supersede
                          :if-does-not-exist :create
                          :element-type '(unsigned-byte 8))
    (%npy-write-header s (%npy-dtype->descr (vt-dtype vt)) (vt-shape vt))
    (write-sequence (%npy-encode-data-buffer vt) s)))

(defun vt-save (tensor path &key (as :lisp))
  "把张量 TENSOR 持久化到 PATH，返回 TENSOR 本身（便于链式调用）。

  AS 关键字选择格式：
    :lisp（缺省）— clvt 原生 s-expression 文本格式：
        (:CLVT-TENSOR :VERSION 1 :SHAPE (2 3) :DTYPE :FLOAT64 :DATA #(1.0d0 ...))
        DATA 按行主序（C-order）展平；任意 strides/offset 的视图（转置、
        切片、广播 stride=0）都按逻辑内容物化；非有限浮点编码为关键字
        :NAN / :INF / :NEG-INF（Lisp reader 严格往返），-0.0 符号位保留。
        加载端兼容无 :VERSION 的旧文件，版本仅支持 1。
    :npy — NumPy .npy 格式（v1.0，头超长自动升 v2.0），C-order、小端，
        与 np.save / np.load 双向互操作：写出的文件 numpy 可直接读。
        8 种 dtype 映射 <f8 <f4 <i8 <i4 <i2 |i1 |u1 <u2（1 字节类型用
        '|' 前缀，与 numpy 写法一致，输出与 np.save 逐字节相同）。

  NaN/Inf 行为：npy 格式逐位往返（符号位与 payload 全保留，对标
  numpy.save）；lisp 文本格式经关键字标记往返（:NAN/:INF/:NEG-INF，
  文本格式不携带 payload，与 numpy 的文本序列化行为一致）。

  错误契约（L2）：tensor 非 vt、path 非路径名、as 非 :lisp/:npy 时
  确定性报错，前缀 vt-save:。

  对标 numpy.save：不支持 file 对象追加模式；格式/对齐/字节序与
  numpy.save 输出逐字节一致。

  示例：
    (vt-save (vt-arange 6) \"/tmp/a.npy\" :as :npy)  ; numpy 可 np.load
    (vt-save (vt-ones '(2 3)) \"/tmp/a.sexp\")       ; 缺省 :lisp 格式"
  (unless (vt-p tensor)
    (error "vt-save: tensor 必须是 vt 张量，收到 ~a" tensor))
  (unless (or (stringp path) (pathnamep path))
    (error "vt-save: path 必须是路径名字符串或 pathname，收到 ~a" path))
  (unless (member as '(:lisp :npy))
    (error "vt-save: as 只支持 :lisp 或 :npy，收到 ~a" as))
  (if (eq as :lisp)
      (%save-lisp-format tensor path)
      (%save-npy-format tensor path))
  tensor)

;;; ------------------------------------------------------------------
;;; 加载入口
;;; ------------------------------------------------------------------

(defun %npy-file-p (path)
  "探测 PATH 是否以 \\x93NUMPY magic 开头（用于 as=nil 自动检测）。"
  (with-open-file (s path :direction :input
                          :element-type '(unsigned-byte 8)
                          :if-does-not-exist nil)
    (when s
      (let ((magic (make-array 6 :element-type '(unsigned-byte 8)
                                 :initial-element 0)))
        (and (>= (read-sequence magic s :end 6) 6)
             (%npy-magic-p magic))))))

(defun vt-load (path &key as)
  "从 PATH 加载张量，返回新建的 C 连续 vt（张量函数一律返回 vt）。

  AS 关键字选择格式：
    :lisp — 读取 vt-save :as :lisp 的 s-expression 文本（读取时关闭
        *read-eval*，文件内容不执行任何代码）；
    :npy  — 读取 NumPy .npy 文件（v1.0/v2.0/v3.0 头；C-order 与
        fortran_order=True 均支持，numpy 写出的文件可直接加载）；
    nil（缺省）— 自动检测：文件以 \\x93NUMPY magic 开头按 :npy，
        否则按 :lisp。

  npy dtype 支持矩阵与 vt-save 一致（f8 f4 i8 i4 i2 i1 u2 u1）；
  float16/bool/complex/字符串等 descr 确定性报错（前缀 vt-load:）。
  NaN 按位模式逐位还原（符号位与 payload 全保留，npy 格式），±Inf 与
  整数补码位模式精确保留；:lisp 格式 NaN 映射为库内 NaN 常量。

  对标 numpy.load：mmap_mode / allow_pickle 不适用（无 pickle 依赖，
  天然安全）；缺少文件、magic 不匹配、dtype 不支持、数据截断均报错。

  示例：
    (vt-load \"/tmp/a.npy\" :as :npy)  ; → #<VT (2 3) :FLOAT64>
    (vt-load \"/tmp/a.sexp\")          ; 自动检测 → lisp 格式"
  (unless (or (stringp path) (pathnamep path))
    (error "vt-load: path 必须是路径名字符串或 pathname，收到 ~a" path))
  (unless (member as '(nil :lisp :npy))
    (error "vt-load: as 只支持 :lisp、:npy 或 nil（自动检测），收到 ~a" as))
  (unless (probe-file path)
    (error "vt-load: 文件不存在，收到 ~a" path))
  (let ((fmt (or as (if (%npy-file-p path) :npy :lisp))))
    (if (eq fmt :lisp)
        (%load-lisp-format path)
        (%load-npy-format path))))
