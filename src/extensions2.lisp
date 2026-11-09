;;;; extensions2.lisp — 补充 NumPy/PyTorch 重要缺失函数
(in-package :clvt)

;;; ------------------------------------------------------------------
;;; 1. 方便别名：fliplr / flipud
;;; ------------------------------------------------------------------
(defun vt-fliplr (vt)
  "左右翻转（沿最后一个轴），对标 np.fliplr。"
  (let ((rank (length (vt-shape vt))))
    (when (< rank 2) (error "vt-fliplr requires at least 2D tensor"))
    (vt-flip vt :axis (1- rank))))

(defun vt-flipud (vt)
  "上下翻转（沿第0轴），对标 np.flipud。"
  (let ((rank (length (vt-shape vt))))
    (when (< rank 1) (error "vt-flipud requires at least 1D tensor"))
    (vt-flip vt :axis 0)))

;;; ------------------------------------------------------------------
;;; 2. ediff1d — 相邻元素差（对标 np.ediff1d）
;;; ------------------------------------------------------------------
(defun %as-1d (x dtype)
  "将任意输入（标量、列表、任意秩张量）折叠为 1 维张量。
   标量 → shape (1)；其余 → 展平为一维。"
  (let ((tv (ensure-vt x :dtype dtype)))
    (if (null (vt-shape tv))
        (vt-reshape tv '(1))
        (vt-flatten tv))))

(defun vt-ediff1d (vt &key to-end to-beginning)
  "1D张量相邻元素差，对标 np.ediff1d。
   to-beginning / to-end 可为标量、列表或张量，统一按 1 维拼接。"
  (with-float-safe
    (let* ((flat (vt-flatten vt))
           (n (vt-size flat))
           (diff (if (<= n 1)
                     (vt-zeros (list 0) :dtype (vt-dtype flat))
                     (let ((src (vt-slice flat `(0 ,(1- n))))
                           (dst (vt-slice flat `(1 ,n))))
                       (vt-- dst src))))
           (parts (list diff)))
      (when to-beginning
        (push (%as-1d to-beginning (vt-dtype diff)) parts))
      (when to-end
        (setf parts (append parts
                            (list (%as-1d to-end (vt-dtype diff))))))
      (if (rest parts) (apply #'vt-concatenate 0 parts) diff))))

;;; ------------------------------------------------------------------
;;; 3. geomspace — 等比数列（对标 np.geomspace）
;;; ------------------------------------------------------------------
(defun vt-geomspace (start stop num &key (dtype :float64) (endpoint t))
  "等比数列（对数尺度等间距），对标 np.geomspace。
   前置条件（否则报错）：
   - num >= 1
   - start 和 stop 均非零
   - start 和 stop 同号
   符号处理：结果为全正或全负，符号取自 start（= stop 的符号）。
   DTYPE 语义：
   - 计算始终在 double-float 精度下进行；
   - 计算完成后按 DTYPE 转换：
       :float64 / :float32 → 浮点返回
       整型（:int64/:int32/:int16/:int8/:uint8/:uint16）
                          → 走 vt-astype，NumPy 截断/回绕语义
       :bool              → 非零为真（vt-astype 真值化；numpy 的
                            np.geomspace(…, dtype=bool) 同样是 != 0）。
                            §17.10.14 之前这里**唯一**拒绝 bool 的
                            「按区间采样」构造函数，与 vt-linspace /
                            vt-logspace 不一致（那两个一直支持）。

   示例：
     (vt-geomspace 1 1000 4)               => [1.0, 10.0, 100.0, 1000.0] (float64)
     (vt-geomspace 1 1000 4 :dtype :int64) => [1, 10, 100, 1000]        (int64)
     (vt-geomspace 1 1000 4 :dtype :int32) => [1, 10, 100, 1000]        (int32)
     (vt-geomspace -1 -1000 4)             => [-1.0, -10.0, -100.0, -1000.0]
     (vt-geomspace 1 -100 5)               => error（异号）
     (vt-geomspace 0 100 5)                => error（含零）"
  (declare (fixnum num))
  ;; NUM = 0 返回空张量（对标 np.geomspace：采样点为空是合法请求）。
  ;; 注意顺序 —— numpy 先做 start/stop 的零与同号校验，所以
  ;; np.geomspace(0, 10, 0) 仍然报错，只有合法的端点才允许空序列。
  (when (minusp num)
    (error "vt-geomspace: num 必须 >= 0，当前为 ~a" num))
  ;; dtype 合法性校验：整型或浮点，其他非法
  (unless (member dtype '(:float64 :float32
                          :int64 :int32 :int16 :int8
                          :uint8 :uint16 :bool))
    (error "vt-geomspace: 不支持的 dtype ~a" dtype))
  (let ((s (coerce start 'double-float))
        (e (coerce stop  'double-float)))
    ;; ---- 前置校验：与 NumPy 一致 ----
    (when (or (zerop s) (zerop e))
      (error "vt-geomspace: start (~a) 和 stop (~a) 均不能为零" start stop))
    (when (not (eql (minusp s) (minusp e)))
      (error "vt-geomspace: start (~a) 和 stop (~a) 必须同号" start stop))
    (when (zerop num)
      (return-from vt-geomspace (vt-zeros (list 0) :dtype dtype)))
    (when (= num 1)
      (return-from vt-geomspace
        (vt-full (list 1) s :dtype dtype)))
        (let* ((log-s (log (abs s)))
           (log-e (log (abs e)))
           (sign  (if (minusp s) -1d0 1d0))
           (div   (if endpoint (1- num) num))
           (result (vt-zeros (list num) :dtype :float64))
           (rdata  (vt-data result)))
      (declare (type fixnum div)
               (type double-float log-s log-e sign))
      (dotimes (i num)
        (let* ((frac (/ (coerce i 'double-float)
                        (coerce div 'double-float)))
               (lv (+ log-s (* frac (- log-e log-s))))
               (v  (* sign (exp lv))))
          (setf (aref rdata i) v)))
      (setf (aref rdata 0) s)
      (when endpoint
        (setf (aref rdata (1- num)) e))
      (if (member dtype '(:int8 :int16 :int32 :int64 :uint8 :uint16))
          (vt-astype (vt-round result) dtype)
          (vt-astype result dtype)))))


;;; ------------------------------------------------------------------
;;; 4. ravel-multi-index
;;; ------------------------------------------------------------------
(defun vt-ravel-multi-index (multi-index shape)
  "多维索引→扁平索引，对标 np.ravel_multi_index。支持标量和批量模式。

   批量模式的入参形态是「每个维度一个序列」，原实现按
   `(nth i mi)` 逐维度取值 —— 对 LIST 而言 NTH 是 O(i)，于是整个批量
   成了 O(n²·k)：n = 1e5 时光还原坐标就要十几秒。这里先把每个序列
   COERCE 成 SIMPLE-VECTOR（一次 O(n)），循环内只做 `SVREF`。

   索引的取值语义不变：整数原样使用，浮点按 TRUNCATE 截断（与
   numpy 的整数下标要求不同，本库沿用既有行为）。非序列元素
   （既不是 LIST 也不是 VECTOR）仍走原来的 NTH 兜底路径。"
  (labels ((strides-of (dims)
             (let ((s 1) (acc '()))
               (dolist (d (reverse dims) acc)
                 (push s acc)
                 (setf s (* s d))))))
    (let* ((strides (strides-of shape)))
      (if (every #'numberp multi-index)
          (reduce #'+ (mapcar #'* multi-index strides))
          (let* ((axes (mapcar (lambda (mi)
                                 (if (typep mi '(or list vector))
                                     (coerce mi 'simple-vector)
                                     mi))
                               multi-index))
                 (n (length (first axes)))
                 (result (vt-zeros (list n) :dtype :int64))
                 (rdata (vt-data result)))
            (declare (type fixnum n)
                     (type (simple-array (signed-byte 64) (*)) rdata))
            (if (every (lambda (a) (typep a 'simple-vector)) axes)
                (dotimes (i n result)
                  (declare (type fixnum i))
                  (let ((idx 0))
                    (declare (type (signed-byte 64) idx))
                    (loop for ax in axes
                          for st fixnum in strides
                          do (let ((v (svref ax i)))
                               (incf idx (* (if (integerp v) v (truncate v))
                                            (the fixnum st)))))
                    (setf (aref rdata i) idx)))
                ;; 兜底：元素不是序列（与旧实现逐字同路径）
                (progn
                  (loop for i fixnum below n do
                    (let ((idx 0))
                      (declare (type (signed-byte 64) idx))
                      (loop for mi in multi-index
                            for str in strides
                            do (incf idx (* (truncate (nth i mi)) (the fixnum str))))
                      (setf (aref rdata i) idx)))
                  result)))))))

;;; ------------------------------------------------------------------
;;; 5. triu_indices / tril_indices
;;;
;;; 两个函数只差「取哪一侧」，共用 %TRIANGLE-INDICES。原实现用 PUSH 收集
;;; 两条 LIST 再 NREVERSE、最后 `:initial-contents` 整表复制一遍 —— 对
;;; 三角矩阵这种「元素数 = O(n²)」的输出，列表节点与数组两份内存都白付。
;;; 现在先按行闭区间数出元素个数（O(n)），再按同一区间直接写进
;;; (signed-byte 64) 数组：零 cons、零二次拷贝。
;;; ------------------------------------------------------------------
(defun %triangle-indices (n cols k lower)
  "返回 (values ROWS COLS)：N × COLS 矩阵里满足三角条件的位置，行主序。
   LOWER 为真取 j ≤ i + k（下三角），否则取 j ≥ i + k（上三角）。"
  (declare (type fixnum n cols k))
  (flet ((row-range (i)
           "该行被接受的列区间 [LO, HI)。"
           (if lower
               (values 0 (min cols (max 0 (+ i k 1))))
               (values (max 0 (min cols (+ i k))) cols))))
    (let ((cnt 0))
      (declare (type fixnum cnt))
      (dotimes (i n)
        (multiple-value-bind (lo hi) (row-range i)
          (declare (type fixnum lo hi))
          (incf cnt (- hi lo))))
      (let ((rows (make-array cnt :element-type '(signed-byte 64)))
            (cl (make-array cnt :element-type '(signed-byte 64)))
            (p 0))
        (declare (type (simple-array (signed-byte 64) (*)) rows cl)
                 (type fixnum p))
        (dotimes (i n)
          (multiple-value-bind (lo hi) (row-range i)
            (declare (type fixnum lo hi))
            (loop for j fixnum from lo below hi
                  do (setf (aref rows p) i)
                     (setf (aref cl p) j)
                     (incf p))))
        ;; 直接把已经写好的 (signed-byte 64) 数组包成 1 维 :int64 张量。
        ;; 走 VT-FROM-ARRAY 会再逐元素 `(vt-cast (row-major-aref …) :int64)`
        ;; 复制一遍 —— 对 O(n²) 个下标来说，这一趟复制比填表本身还贵一个量级。
        (flet ((as-vt (arr)
                 (%make-vt :data arr :shape (list cnt) :strides (list 1)
                           :offset 0 :dtype :int64)))
          (values (as-vt rows) (as-vt cl)))))))

(defun vt-tril-indices (n &key (k 0) m)
  "返回下三角索引 (rows, cols)，对标 np.tril_indices。"
  (declare (fixnum n k))
  (when (minusp n)
    (error "vt-tril-indices: n (~a) 不能为负" n))
  (let ((cols (or m n)))
    (declare (fixnum cols))
    (when (minusp cols)
      (error "vt-tril-indices: m (~a) 不能为负" m))
    (%triangle-indices n cols k t)))

(defun vt-triu-indices (n &key (k 0) m)
  "返回上三角索引 (rows, cols)，对标 np.triu_indices。"
  (declare (fixnum n k))
  (when (minusp n)
    (error "vt-triu-indices: n (~a) 不能为负" n))
  (let ((cols (or m n)))
    (declare (fixnum cols))
    (when (minusp cols)
      (error "vt-triu-indices: m (~a) 不能为负" m))
    (%triangle-indices n cols k nil)))

;;; ------------------------------------------------------------------
;;; 6. vander
;;; ------------------------------------------------------------------
(defun vt-vander (x &key (n nil) (increasing nil))
  "范德蒙德矩阵，对标 np.vander。
   - n 缺省（NIL）时取 x 的元素个数（np.vander 缺省 N = len(x)）
   - increasing = t : 第 j 列为 x^j
   - increasing = nil : 第 j 列为 x^(ncols-1-j)
   实现用一行内 O(ncols) 递推，避免每个元素重复做幂运算。
   ★ 对整数类型，溢出时按 NumPy 语义回绕（wrap），不再因 bignum 写入报错。
   ★ 布尔输入的结果 dtype 取 **:int64**（numpy：`np.vander(bool)` → int64）。
     保持 `:bool` 会让「第 3 列 = x²」与 `(vt-pow b 2)`（§17.10.11 已定为
     int64）在同一个表达式上给出两种 dtype；值虽然都是 0/1，但下游一旦
     就地写入（`(setf (vt-slice …) 3)`）就会撞上 bool 的 0/1 值域。

   递推的每一步都必须「按结果 dtype 回绕」。原实现把这一步写成
   `(vt-cast (* base xi) dtype)` —— 每元素一次 ECASE 分派 + 通用算术 + COERCE，
   而它要的只是算术里的一次截断。现在按元素类型各展开一份循环：窄整型用
   `(ldb (byte w 0) …)` 取低位后按有/无符号还原，int64 用
   `(the (signed-byte 64) …)`（SBCL 在该声明下发模乘，溢出即回绕，不产生
   bignum），浮点用同精度的原生乘法 —— 与 `(vt-cast … dtype)` 逐位相同。

   `X-OFF` 参与寻址：`vt-contiguous` 对「已连续但 offset 非零」的视图原样
   返回，漏掉它就会从底层数组的第 0 个元素开始取 x。"
  (let* ((flat (vt-flatten x))
         ;; :bool 的结果 dtype 是 :int64（见 docstring），先把输入按结果 dtype
         ;; 落地：否则下面按结果元素类型声明的 XS/RS 会撞上「输入是 0/1 字节
         ;; 数组」而抛 TYPE-ERROR。
         (dtype (let ((d (vt-dtype flat))) (if (eq d :bool) :int64 d)))
         (xv (vt-contiguous (if (eq (vt-dtype flat) dtype) flat (vt-astype flat dtype))))
         (len (vt-size xv))
         (ncols (or n len))
         (result (vt-zeros (list len ncols) :dtype dtype))
         (xdata (vt-data xv))
         (x-off (vt-offset xv))
         (rdata (vt-data result)))
    (declare (type fixnum len ncols x-off))
    (macrolet ((rows (lt one next)
                 `(let ((xs (the (simple-array ,lt (*)) xdata))
                        (rs (the (simple-array ,lt (*)) rdata)))
                    (declare (type (simple-array ,lt (*)) xs rs)
                             (type fixnum len ncols x-off))
                    (dotimes (i len)
                      (declare (type fixnum i))
                      (let ((xi (aref xs (the fixnum (+ x-off i))))
                            (row-start (the fixnum (* i ncols))))
                        (declare (type ,lt xi) (type fixnum row-start))
                        (let ((base (the ,lt ,one)))
                          (declare (type ,lt base))
                          (if increasing
                              (dotimes (j ncols)
                                (declare (type fixnum j))
                                (setf (aref rs (the fixnum (+ row-start j))) base)
                                (setf base ,next))
                              ;; 递减：j=0 写入最高次幂 xi^(ncols-1)，j=ncols-1 写入 xi^0
                              (dotimes (j ncols)
                                (declare (type fixnum j))
                                (setf (aref rs (the fixnum (+ row-start (- ncols 1 j))))
                                      base)
                                (setf base ,next))))))))
               (wrap-signed (bits)
                 "有符号窄整型的回绕：取低 BITS 位后按二进制补码还原。"
                 `(let ((u (ldb (byte ,bits 0) (* base xi))))
                    (if (> u ,(1- (ash 1 (1- bits))))
                        (- u ,(ash 1 bits))
                        u)))
               (wrap-unsigned (bits)
                 "无符号窄整型的回绕：取低 BITS 位。"
                 `(ldb (byte ,bits 0) (* base xi))))
      (let ((et (array-element-type rdata)))
        (cond ((equal et 'double-float)            (rows double-float 1.0d0 (* base xi)))
              ((equal et 'single-float)            (rows single-float 1.0s0 (* base xi)))
              ;; int64 的回绕只能靠「声明驱动的模乘」实现（SBCL 在该声明下
              ;; 发 IMUL + 溢出转 bignum 再截断，结果恰是补码回绕）。默认
              ;; safety 1 会为返回值插类型检查、超界即抛 TYPE-ERROR，所以
              ;; 这段纯下标/纯标量算术按 §7.4.3 的口径放进 safety 0：
              ;; 循环边界与数组类型都由外层构造保证，不需要运行期检查。
              ((equal et '(signed-byte 64))
               (locally (declare (optimize (speed 3) (safety 0)))
                 (rows (signed-byte 64) 1 (the (signed-byte 64) (* base xi)))))
              ((equal et '(signed-byte 32))        (rows (signed-byte 32) 1 (wrap-signed 32)))
              ((equal et '(signed-byte 16))        (rows (signed-byte 16) 1 (wrap-signed 16)))
              ((equal et '(signed-byte 8))         (rows (signed-byte 8) 1 (wrap-signed 8)))
              ((equal et '(unsigned-byte 16))      (rows (unsigned-byte 16) 1 (wrap-unsigned 16)))
              ((equal et '(unsigned-byte 8))       (rows (unsigned-byte 8) 1 (wrap-unsigned 8)))
              ;; 未知元素类型：逐元素沿用 VT-CAST（语义一致，只是慢）
              (t (dotimes (i len result)
                   (declare (type fixnum i))
                   (let ((xi (aref xdata (the fixnum (+ x-off i))))
                         (row-start (the fixnum (* i ncols)))
                         (base (vt-cast 1 dtype)))
                     (declare (type fixnum row-start))
                     (if increasing
                         (dotimes (j ncols)
                           (setf (aref rdata (the fixnum (+ row-start j))) base)
                           (setf base (vt-cast (* base xi) dtype)))
                         (dotimes (j ncols)
                           (setf (aref rdata (the fixnum (+ row-start (- ncols 1 j)))) base)
                           (setf base (vt-cast (* base xi) dtype)))))))))
      result)))

;;; ------------------------------------------------------------------
;;; 7. one-hot
;;; ------------------------------------------------------------------
(defun vt-one-hot (x num-classes &key (dtype :float64))
  "one-hot 编码，对标 torch.nn.functional.one_hot。
   x 可为标量、列表或任意形状张量；非连续视图会先落地为连续副本，
   索引超出 [0, num-classes) 时按 PyTorch 语义报错。"
  (declare (fixnum num-classes))
  (when (< num-classes 1)
    (error "vt-one-hot: num-classes (~a) 必须为正整数" num-classes))
  (let* ((x-contig (vt-contiguous (vt-astype (ensure-vt x) :int64)))
         (in-shape (vt-shape x-contig))
         (out-shape (append in-shape (list num-classes)))
         (result (vt-zeros out-shape :dtype dtype))
         (xdata (vt-data x-contig))
         (x-off (vt-offset x-contig))
         (rdata (vt-data result))
         (total (reduce #'* in-shape :initial-value 1))
         (one-val (coerce 1 (vt-dtype->lisp-type dtype))))
    (declare (type fixnum total))
    ;; v0.4.8 性能修复：XDATA / RDATA 都没有元素类型声明，scatter 时
    ;; `(setf (aref rdata ...) one-val)` 与 `(aref xdata ...)` 都走通用二维
    ;; 分派（每元素一次 widetag 查询 + 边界检查）。1M×100 实测 555250us
    ;; （numpy 同工作量 142740us）。这里按输出 dtype 展开 8 路特化：
    ;; 只有 1M 次 scatter，但输出缓冲是 100M 元素（800MB），
    ;; 分配与 memset 仍占大头，特化能省的是「写偏差点」这部分。
    (macrolet ((scatter (lt)
                 `(let ((rd (the (simple-array ,lt (*)) rdata))
                        (xs (the (simple-array (signed-byte 64) (*)) xdata))
                        (one (the ,lt one-val)))
                    (declare (type (simple-array ,lt (*)) rd)
                             (type (simple-array (signed-byte 64) (*)) xs)
                             (type ,lt one)
                             (type fixnum total num-classes x-off))
                    (dotimes (i total)
                      (declare (type fixnum i))
                      (let ((cls (aref xs (the fixnum (+ x-off i)))))
                        (declare (type fixnum cls))
                        (unless (and (>= cls 0) (< cls num-classes))
                          (error "vt-one-hot: 类别 ~a 越界 [0, ~a)" cls num-classes))
                        (setf (aref rd (the fixnum (+ (* i num-classes) cls)))
                              one))))))
      (let ((et (array-element-type rdata)))
        (cond
          ((equal et 'double-float)       (scatter double-float))
          ((equal et 'single-float)       (scatter single-float))
          ((equal et '(signed-byte 64))   (scatter (signed-byte 64)))
          ((equal et '(signed-byte 32))   (scatter (signed-byte 32)))
          ((equal et '(signed-byte 16))   (scatter (signed-byte 16)))
          ((equal et '(signed-byte 8))    (scatter (signed-byte 8)))
          ((equal et '(unsigned-byte 16)) (scatter (unsigned-byte 16)))
          ((equal et '(unsigned-byte 8))  (scatter (unsigned-byte 8)))
          ;; 防御：其他元素类型 → 通用路径
          (t (dotimes (i total)
               (declare (type fixnum i))
               (let ((cls (aref xdata (+ x-off i))))
                 (unless (and (>= cls 0) (< cls num-classes))
                   (error "vt-one-hot: 类别 ~a 越界 [0, ~a)" cls num-classes))
                 (setf (aref rdata (+ (* i num-classes) cls)) one-val)))))))
    result))

;;; ------------------------------------------------------------------
;;; 8. standardize (z-score)
;;; ------------------------------------------------------------------
(defun vt-standardize (vt &key axis (ddof 0) dtype out)
  "Z-score 标准化：(x - mean) / std，对标 sklearn.preprocessing.scale。
   结果 dtype 永远是浮点：显式传入整数 dtype 会报错，
   未指定时按输入 dtype 提升到 :float32 / :float64。"
  (when (%cx-any-p vt)
    (return-from vt-standardize (%cx-op-standardize vt :axis axis :ddof ddof :dtype dtype :out out)))
  (when (and dtype (member dtype '(:int8 :int16 :int32 :int64
                                   :uint8 :uint16 :bool)))
    ;; :bool 必须在这里拦住。漏掉它会一路走到内部的 `(vt-- vt m :dtype :bool)`，
    ;; 于是用户传 `:dtype :bool` 收到的是「vt--: 布尔张量不支持减法/取负」——
    ;; 报错归因到了用户从没调用的函数上（实测于 §17.10.15）。
    (error "vt-standardize: 标准化结果必须为浮点 dtype，收到 ~a~@[（:bool 也是非浮点）~]"
           dtype (eq dtype :bool)))
  (let* ((final-dtype (or dtype
                          (if (eq (vt-dtype vt) :float32) :float32 :float64)))
         (m (vt-mean vt :axis axis :keepdims t :dtype final-dtype))
         (s (vt-std vt :axis axis :keepdims t :ddof ddof :dtype final-dtype))
         (centered (vt-- vt m :dtype final-dtype))
         (zero (if (eq final-dtype :float32) 0.0s0 0.0d0)))
    (vt-map (lambda (x sd)
              (if (< (abs (float sd 1d0)) 1d-12)
                  zero
                  (/ x sd)))
            centered s :dtype final-dtype :out out)))

;;; ------------------------------------------------------------------
;;; 9. Layer Normalization
;;; ------------------------------------------------------------------
(defun vt-layer-norm (vt normalized-shape &key (eps 1d-5) gamma beta dtype out)
  "Layer Normalization，对标 torch.nn.functional.layer_norm。
   normalized-shape 是输入张量的尾部形状，要求：
   - 其秩 <= 输入张量的秩；
   - 与输入张量的尾部维度逐一相等。
   例如 vt-shape=(2,3,4) 时 normalized-shape 可为 (4)、(3,4)、(2,3,4)。
   gamma / beta 的形状必须严格等于 normalized-shape。"
  (let* ((in-shape (vt-shape vt))
         (rank (length in-shape))
         (ndim (length normalized-shape)))

    ;; 两层前置校验，避免误导性的 \"轴重复\" 报错
    (when (> ndim rank)
      (error "vt-layer-norm: normalized-shape 秩 (~a) 不能大于输入张量秩 (~a)"
             ndim rank))
    (let ((tail (subseq in-shape (- rank ndim))))
      (unless (equal tail normalized-shape)
        (error "vt-layer-norm: normalized-shape ~a 与输入尾部维度 ~a 不匹配"
               normalized-shape tail)))

    ;; gamma / beta 形状校验
    (when gamma
      (unless (equal (vt-shape gamma) normalized-shape)
        (error "vt-layer-norm: gamma 形状 ~a 与 normalized-shape ~a 不匹配"
               (vt-shape gamma) normalized-shape)))
    (when beta
      (unless (equal (vt-shape beta) normalized-shape)
        (error "vt-layer-norm: beta 形状 ~a 与 normalized-shape ~a 不匹配"
               (vt-shape beta) normalized-shape)))

    ;; ── 复数分支（v0.5.x 新增）──────────────────────────────────
    ;;
    ;; 语义：沿 NORM-AXES 把每个 fiber 归一化到「均值 0、L2 离散度 1」。
    ;;   mean = vt-mean（复数，逐分量平均）
    ;;   var  = vt-var （**实数**：numpy 对复数取 mean(|x-mean|²)，§16.10）
    ;;   std  = sqrt(var + eps)（实数）
    ;;   normed = (x - mean) / std（复数 / 实数 → 复数）
    ;; 因此只要不让 :dtype 把中间量强制成实数，整条链路就是复数安全的。
    ;; 旧实现把 FINAL-DTYPE 写死成「float32 或 float64」，于是
    ;; (vt-layer-norm 复数) 在 (vt-+ var eps :dtype :float64) 处撞上
    ;; 「#C(0 0) is not of type REAL」。
    (when (vt-complex-dtype-p (vt-dtype vt))
      (let* ((norm-axes (loop for i from (- rank ndim) below rank collect i))
             (cx-dtype (or dtype (vt-dtype vt)))
             (real-dtype (vt-complex-real-dtype (vt-dtype vt)))
             (mean (vt-mean vt :axis norm-axes :keepdims t :dtype cx-dtype))
             (var (vt-var vt :axis norm-axes :keepdims t :dtype real-dtype))
             (eps-typed (coerce eps (if (eq real-dtype :float32)
                                        'single-float
                                        'double-float)))
             (std (vt-sqrt (vt-+ var eps-typed :dtype real-dtype)
                           :dtype real-dtype))
             (normed (vt-/ (vt-- vt mean :dtype cx-dtype) std :dtype cx-dtype)))
        (let ((result (if gamma
                          (vt-* normed gamma :dtype cx-dtype :out out)
                          (if out
                              (progn (vt-copy-into normed out) out)
                              normed))))
          (return-from vt-layer-norm
            (if beta
                (vt-+ result beta :dtype cx-dtype :out result)
                result)))))

    (let* ((norm-axes (loop for i from (- rank ndim) below rank collect i))
           (mean (vt-mean vt :axis norm-axes :keepdims t :dtype dtype))
           (var (vt-var vt :axis norm-axes :keepdims t :dtype dtype))
           (final-dtype (or dtype
                            (if (eq (vt-dtype vt) :float32) :float32 :float64)))
           (eps-typed (coerce eps (if (eq final-dtype :float32)
                                      'single-float
                                      'double-float)))
           (std (vt-sqrt (vt-+ var eps-typed :dtype final-dtype)
                         :dtype final-dtype))
           (normed (vt-/ (vt-- vt mean :dtype final-dtype)
                         std :dtype final-dtype)))
      (let ((result (if gamma
                        (vt-* normed gamma :dtype final-dtype :out out)
                        (if out
                            (progn (vt-copy-into normed out) out)
                            normed))))
        (if beta
            (vt-+ result beta :dtype final-dtype :out result)
            result)))))

;;; ------------------------------------------------------------------
;;; 10. apply-along-axis
;;; ------------------------------------------------------------------
(defun %apply-along-axis-result-dtype (sample in-dtype)
  "VT-APPLY-ALONG-AXIS 的结果 dtype。

   非布尔输入保持旧契约（结果跟随输入 dtype）。布尔输入是特例：结果 dtype
   由 FUNC 的返回值决定 —— 把 `(vt-sum row)` 的 3 塞进 `:bool` 数组，要么被
   `%bool-flag` 压成 1（静默错值），要么在 1D→1D 分支里直接把 5、6 这种越界
   字节写进去（破坏 §17.1 的 0/1 不变量）。numpy 同样按 FUNC 的返回类型定
   结果 dtype。"
  (if (not (eq in-dtype :bool))
      in-dtype
      (cond ((vt-p sample) (vt-dtype sample))
            ((integerp sample) :int64)
            ((floatp sample) :float64)
            ((complexp sample) :complex128)
            (t :bool))))

(defun %aax-gather (sd soff ax-dim inner ii arr)
  "把 SD 中从 SOFF+II 起、步长 INNER 的 AX-DIM 个元素按序写进 ARR。

   按**存储元素类型**展开成 10 份特化循环（§7.4.13）：旧写法在
   `(aref data (+ data-off …))` 上每元素走一次通用 AREF（1M 元素 12 ms）。
   ARR 的分配留在调用方（每个 fiber 一份），语义与「每次新建缓冲」一致 ——
   若被调函数把入参原样返回，各 fiber 的结果仍是各自独立的数组。"
  (declare (type fixnum soff ax-dim inner ii))
  (macrolet
      ((gather (lt)
         `(let ((s (the (simple-array ,lt (*)) sd))
                (d (the (simple-array ,lt (*)) arr)))
            (declare (type (simple-array ,lt (*)) s d)
                     (type fixnum soff ax-dim inner ii))
            (let ((p (the fixnum (+ soff ii))))
              (declare (type fixnum p))
              (dotimes (k ax-dim)
                (declare (type fixnum k))
                (setf (aref d k) (aref s p))
                (incf p inner))))))
    (let ((et (array-element-type sd)))
      (cond
        ((equal et 'double-float)           (gather double-float))
        ((equal et 'single-float)           (gather single-float))
        ((equal et '(signed-byte 64))       (gather (signed-byte 64)))
        ((equal et '(signed-byte 32))       (gather (signed-byte 32)))
        ((equal et '(signed-byte 16))       (gather (signed-byte 16)))
        ((equal et '(signed-byte 8))        (gather (signed-byte 8)))
        ((equal et '(unsigned-byte 16))     (gather (unsigned-byte 16)))
        ((equal et '(unsigned-byte 8))      (gather (unsigned-byte 8)))
        ((equal et '(complex double-float)) (gather (complex double-float)))
        ((equal et '(complex single-float)) (gather (complex single-float)))
        (t
         (let ((p (the fixnum (+ soff ii))))
           (dotimes (k ax-dim)
             (setf (aref arr k) (aref sd p))
             (incf p inner))))))))

(defun %aax-scatter (rd base ov new-dim inner ii out-dt)
  "把 OV 的 NEW-DIM 个元素写进 RD，落点 = BASE + K×INNER + II。

   两条路：OV 与原结果**同 dtype 且连续**时逐元素直写（恒等转换）；
   否则 `%VT-CAST-RAW` + `%VT-REF-RAW`。旧写法无条件 `VT-CAST`，
   那层浮点陷阱屏蔽每次进出的固定开销是 216 ns —— 1M 输出即 249 ms。"
  (declare (type fixnum base new-dim inner ii))
  (if (and (eq (vt-dtype ov) out-dt) (vt-contiguous-p ov))
      (let ((od (vt-data ov))
            (oo (the fixnum (vt-offset ov)))
            (et (array-element-type rd)))
        (macrolet
            ((scatter (lt)
               `(let ((d (the (simple-array ,lt (*)) rd))
                      (s (the (simple-array ,lt (*)) od)))
                  (declare (type (simple-array ,lt (*)) d s)
                           (type fixnum oo base new-dim inner ii))
                  (let ((p base))
                    (declare (type fixnum p))
                    (dotimes (k new-dim)
                      (declare (type fixnum k))
                      (setf (aref d (the fixnum (+ p ii)))
                            (aref s (the fixnum (+ oo k))))
                      (incf p inner))))))
          (cond
            ((equal et 'double-float)           (scatter double-float))
            ((equal et 'single-float)           (scatter single-float))
            ((equal et '(signed-byte 64))       (scatter (signed-byte 64)))
            ((equal et '(signed-byte 32))       (scatter (signed-byte 32)))
            ((equal et '(signed-byte 16))       (scatter (signed-byte 16)))
            ((equal et '(signed-byte 8))        (scatter (signed-byte 8)))
            ((equal et '(unsigned-byte 16))     (scatter (unsigned-byte 16)))
            ((equal et '(unsigned-byte 8))      (scatter (unsigned-byte 8)))
            ((equal et '(complex double-float)) (scatter (complex double-float)))
            ((equal et '(complex single-float)) (scatter (complex single-float)))
            (t
             (let ((p base))
               (dotimes (k new-dim)
                 (setf (aref rd (the fixnum (+ p ii)))
                       (%vt-cast-raw (aref od (the fixnum (+ oo k))) out-dt))
                 (incf p inner)))))))
      (let ((p base))
        (declare (type fixnum p))
        (dotimes (k new-dim)
          (declare (type fixnum k))
          (setf (aref rd (the fixnum (+ p ii)))
                (%vt-cast-raw (%vt-ref-raw ov k) out-dt))
          (incf p inner)))))

(defun vt-apply-along-axis (func axis vt)
  "沿指定轴对 1D 切片应用函数，对标 np.apply_along_axis。
   支持 func 返回标量（1D→scalar）或 1D 张量（1D→1D）。
   注意：func 会被额外调用一次用于探测输出形状，因此不应有副作用。
   布尔输入的结果 dtype 取 FUNC 返回值的 dtype（见
   `%APPLY-ALONG-AXIS-RESULT-DTYPE`，CONVENTIONS §17.10.13）。

   执行层（§7.4.13）：每个 fiber 的三笔固定开销都在改前路径上
   —— 泛型 AREF 逐元素取数、`VT-FROM-ARRAY` 逐元素 `VT-CAST`、
   写回再逐元素 `VT-CAST`。前两笔各按元素收一次钱（1M 元素 = 250ms + 250ms），
   现在都换成按元素类型特化的循环，写回的转换只在 dtype 不同时才发生。"
  (with-float-safe
    (let* ((vt-c (vt-contiguous vt))
           (shape (vt-shape vt-c))
           (rank (length shape))
           (ax (vt-normalize-axis axis rank))
           (ax-dim (nth ax shape))
           (ltype (vt-dtype->lisp-type (vt-dtype vt-c)))
           (before (subseq shape 0 ax))
           (after (subseq shape (1+ ax)))
           (outer (reduce #'* before :initial-value 1))
           (inner (reduce #'* after :initial-value 1))
           (data (vt-data vt-c))
           (data-off (vt-offset vt-c)))
      (declare (fixnum ax-dim outer inner))
      ;; 探测输出形状
      (let* ((probe (make-array ax-dim :element-type ltype
                                         :initial-element (coerce 0 ltype)))
             (probe-vt (vt-from-array probe :dtype (vt-dtype vt-c)))
             (sample (funcall func probe-vt))
             (out-dt (%apply-along-axis-result-dtype sample (vt-dtype vt-c))))
        (cond
          ;; 1D -> scalar
          ((or (numberp sample) (and (vt-p sample) (null (vt-shape sample))))
           (let* ((out-shape (append before after))
                  (result (vt-zeros out-shape :dtype out-dt))
                  (rdata (vt-data result)))
             (dotimes (bi outer)
               (dotimes (ii inner)
                 (let ((arr (make-array ax-dim :element-type ltype)))
                   (%aax-gather data (+ data-off (* bi ax-dim inner))
                                ax-dim inner ii arr)
                   (let* ((sl (vt-from-array arr :dtype (vt-dtype vt-c)))
                          (rv (funcall func sl))
                          (raw (%item-number rv)))
                     (setf (aref rdata (+ (* bi inner) ii))
                           (%vt-cast-raw raw out-dt))))))
             result))
          ;; 1D -> 1D
          ((and (vt-p sample) (= (length (vt-shape sample)) 1))
           (let* ((new-dim (car (vt-shape sample)))
                  (out-shape (append before (list new-dim) after))
                  (result (vt-zeros out-shape :dtype out-dt))
                  (rdata (vt-data result)))
             (declare (fixnum new-dim))
             (dotimes (bi outer)
               (dotimes (ii inner)
                 (let ((arr (make-array ax-dim :element-type ltype)))
                   (%aax-gather data (+ data-off (* bi ax-dim inner))
                                ax-dim inner ii arr)
                   (let* ((sl (vt-from-array arr :dtype (vt-dtype vt-c)))
                          (ov (funcall func sl)))
                     ;; VT-REF 兼容非连续/视图返回值；连续且同 dtype 走特化直写
                     (%aax-scatter rdata (* bi new-dim inner) ov new-dim inner ii out-dt)))))
             result))
          (t (error "vt-apply-along-axis: func must return scalar or 1D VT")))))))

;;; ==================================================================
;;; 加载期自检：参数契约基础设施
;;; ==================================================================
;;; parcontract.lisp 的 %parcontract-self-check 依赖 vt-slice / vt-ref /
;;; vt-from-sequence / vt-copy 等，故放在所有组件加载完毕后执行。
;;; 失败将中断加载（这是刻意的：基础设施失效比功能失效更危险）。
(eval-when (:load-toplevel :execute)
  (%parcontract-self-check))
