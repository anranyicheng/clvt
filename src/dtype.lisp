;;;; dtype.lisp — 元素数据类型系统（单一事实来源）

(in-package :clvt)

(defparameter *vt-dtypes*
  '(:complex128 :complex64
    :float64 :float32 :int64 :int32 :int16 :int8 :uint8 :uint16 :bool)
  "库内识别到的 dtype 全集（前 6 者为物理存储类型）。
   顺序约定：宽 → 窄、类别由高到低（复数 > 浮点 > 整数 > 布尔），与
   `+vt-elem-types+` 的「命中概率降序」排序保持同一风格。
   complex 见 CONVENTIONS §16，bool 见 §17。")

(defparameter *vt-storage-dtypes*
  '(:complex128 :complex64 :float64 :float32 :int64 :int32)
  "实际可分配底层数组的元素类型（含复数：interleaved 的
   (complex double-float) / (complex single-float) 特化数组）。")

(defparameter *vt-complex-dtypes* '(:complex128 :complex64)
  "复数 dtype 全集（CONVENTIONS §16.1）。")

(declaim
 (inline vt-dtype-p vt-float-dtype-p vt-int-dtype-p vt-storage-dtype-p
         vt-complex-dtype-p vt-bool-dtype-p vt-complex-real-dtype))

(defun vt-dtype->lisp-type (dtype)
  "将内部 dtype 符号映射为 Common Lisp 数组元素类型。
   `:bool` 与 `:uint8` 共用 `(unsigned-byte 8)` 物理存储（见 §17）：
   布尔值以 0/1 承载，1 字节，与 numpy 的 `bool_` 尺寸一致。"
  (declare (optimize (speed 3) (safety 0)))
  (ecase dtype
    (:complex128 '(complex double-float))
    (:complex64  '(complex single-float))
    (:float64 'double-float)
    (:float32 'single-float)
    (:int64   '(signed-byte 64))
    (:int32   '(signed-byte 32))
    (:int16   '(signed-byte 16))
    (:int8    '(signed-byte 8))
    (:uint8   '(unsigned-byte 8))
    (:uint16  '(unsigned-byte 16))
    (:bool    '(unsigned-byte 8))))

(defun vt-dtype-itemsize (dtype)
  "返回每个元素的字节大小（复数 = 2 × 分量宽度，interleaved 存储）。"
  (declare (optimize (speed 3) (safety 0)))
  (ecase dtype
    (:complex128 16)
    (:complex64  8)
    (:float64 8)
    (:float32 4)
    (:int64   8)
    (:int32   4)
    (:int16   2)
    (:int8    1)
    (:uint8   1)
    (:uint16  2)
    (:bool    1)))

(defun vt-dtype-p (x)
  "判定 X 是否为合法逻辑 dtype（见 *vt-dtypes*，共 11 种：9 种非复数 + 2 种复数）。"
  (declare (optimize (speed 3) (safety 0)))
  (not (null (member x *vt-dtypes*))))

(defun vt-storage-dtype-p (x)
  "判定 X 是否为物理存储 dtype（见 *vt-storage-dtypes*，共 6 种：complex128/complex64/float64/float32/int64/int32）。"
  (declare (optimize (speed 3) (safety 0)))
  (not (null (member x *vt-storage-dtypes*))))

(defun vt-float-dtype-p (dtype)
  "判定 DTYPE 是否为浮点类型（:float32 / :float64）。"
  (declare (optimize (speed 3) (safety 0)))
  (member dtype '(:float32 :float64)))

(defun vt-complex-dtype-p (dtype)
  "判定 DTYPE 是否为复数类型（:complex64 / :complex128）。CONVENTIONS §16.1。"
  (declare (optimize (speed 3) (safety 0)))
  (member dtype '(:complex64 :complex128)))

(defun vt-bool-dtype-p (dtype)
  "判定 DTYPE 是否为布尔类型（:bool）。CONVENTIONS §17。

   `:bool` 是**逻辑 dtype**，物理存储与 `:uint8` 相同（(unsigned-byte 8)，
   1 字节，取值只是 0/1）。它存在的理由是语义：比较、逻辑、谓词类函数
   在 numpy 里返回 `bool_`，而在算术语义里布尔**不能**被当成整数参与
   算术（`True + True` 在 numpy 里仍按逻辑或给 True，`sum(bool)` 则升到
   int64）。把返回值标成 `:int8` 时这两种语义都表达不出来。"
  (declare (optimize (speed 3) (safety 0)))
  (eq dtype :bool))

(defun vt-complex-real-dtype (dtype)
  "复数 dtype → 其分量的实数 dtype（:complex128→:float64，:complex64→:float32）。
   非复数 dtype 原样返回——这样「把复数当作实部参与的普通 dtype」的
   提升规则（§16.3）可以用同一个入口表达。"
  (declare (optimize (speed 3) (safety 0)))
  (case dtype
    (:complex128 :float64)
    (:complex64  :float32)
    (t dtype)))

(defun vt-real-dtype-of (dtype)
  "取「实部语义」dtype：复数返回分量浮点类型，其余原样返回。
   用于 abs / real / imag / angle / var / std / norm 这类**必然实数输出**
   的函数推导结果 dtype（§16.6）。"
  (vt-complex-real-dtype dtype))

;;; ------------------------------------------------------------------
;;; 弱标量提升（numpy NEP 50）
;;;
;;; numpy 2.x 起，**Lisp 侧的字面数/标量**（对应 Python 的 int/float/complex）
;;; 是「弱」类型：它们只贡献「种类」，不贡献宽度。于是
;;;     f32 数组 + 1.5   → float32      （不是 float64）
;;;     c64 数组 + 1+2j  → complex64    （不是 complex128）
;;;     int32 数组 + 1   → int32        （不是 int64）
;;;     int32 数组 + 1.5 → float64      （整型装不下浮点种类 → 落到默认浮点）
;;; 而 **张量与张量** 之间仍是完整的 promote 表（§3.2 / §16.3）。
;;;
;;; 本库此前把标量先 ensure-vt 成 float64/complex128 再 promote，于是
;;; f32/c64 的每个「+标量」都静默升宽（内存翻倍、速度减半），且与 numpy 不符。
;;; CONVENTIONS §0「与 NumPy 冲突以 NumPy 为准」，故在此按 NEP 50 修正。
;;;
;;; 唯一有意保留的差异：弱整数越界（int8 + 300）本库按回绕处理，
;;; numpy 抛 OverflowError；回绕是既有语义，不在本次改动范围。
;;; ------------------------------------------------------------------

(declaim (inline %scalar-kind))
(defun %scalar-kind (x)
  "操作数分类：张量 → NIL；Lisp 数 → :INT / :FLOAT / :COMPLEX（弱标量）。"
  (cond ((vt-p x) nil)
        ((complexp x) :complex)
        ((integerp x) :int)
        ((floatp x) :float)
        (t nil)))

(declaim (inline %dtype-kind))
(defun %dtype-kind (dtype)
  "dtype 的大类：:COMPLEX / :FLOAT / :INT / :BOOL。"
  (cond ((member dtype '(:complex64 :complex128)) :complex)
        ((member dtype '(:float32 :float64)) :float)
        ((eq dtype :bool) :bool)
        (t :int)))

(declaim (inline %dtype-kind-rank))
(defun %dtype-kind-rank (dtype)
  "种类的「宽度秩」：bool=-1 < int=0 < float=1 < complex=2。
   单元素预转只允许 **源 ≤ 目标**（int→float→complex 是精确/无损方向）；
   反向（float→int）会截断，把 `nan < 1` 这类比较直接算错。
   bool 的秩为 -1：任何数值 dtype 都能无损承载 0/1，故 bool→任意数值
   方向永远放行；反过来 float→bool 是「真值化」（0.5 → 1）而不是截断，
   不能当成普通预转处理。"
  (case (%dtype-kind dtype) (:bool -1) (:int 0) (:float 1) (t 2)))

(defun %int-family-dtype-p (dtype)
  "整数族（含无符号）。注意与 VT-INT-DTYPE-P 不同，后者只有有符号。"
  (declare (optimize (speed 3) (safety 0)))
  (member dtype '(:int8 :int16 :int32 :int64 :uint8 :uint16)))

(defun %op-final-dtype (kinds dtypes)
  "逐元素运算的结果 dtype。KINDS 每项来自 %SCALAR-KIND（NIL=张量），
   DTYPES 是各操作数（张量）的 dtype。张量之间走 VT-PROMOTE-TYPE，
   弱标量按 NEP 50 只贡献种类。

   `:bool` 的两条额外规则（numpy 实测，§17）：
     · 布尔张量 ⊔ 布尔张量 → `:bool`（不进整数族，`True+True` 仍是 bool）；
     · 布尔张量 ⊔ 弱整数标量 → `:int64`（`np.array([True]) + 1` 是 int64）。
       弱整数标量原先对任何 base 都「不改变 dtype」，那会让 bool 结果
       留住 2 这种越界值，破坏「布尔数组只含 0/1」的不变量。"
  (let ((base nil)
        (tensor-dtypes '()))
    ;; 张量之间**一次提升全部**，不是逐对折叠。
    ;; 折叠与一次提升在二元时等价，三元起会分叉：
    ;;   (vt-+ int16 uint16 float32)
    ;;     折叠：int16 ⊔ uint16 = int32，再 ⊔ float32 → int32 超 24 位 → float64
    ;;     一次：最大有符号 16 位、最大无符号 16 位、有 float32 → float32
    ;; numpy 的 np.result_type 是「一次看全部」（按最大位宽判定），给 float32；
    ;; 库自己的 VT-PROMOTE-TYPE、VT-CONCATENATE（apply 全表）、
    ;; VT-STACK 也都是 float32 —— 只有这里折叠成了 float64。
    ;; 同一套提升规则在库内出现了两种结果，属于「同语义两条实现」。
    (loop for k in kinds for d in dtypes
          do (unless k (push d tensor-dtypes)))
    (when tensor-dtypes
      (setf base (apply #'vt-promote-type (nreverse tensor-dtypes))))
    (loop for k in kinds
          when k do
            (setf base
                  (cond ((null base)
                         (ecase k (:int :int64) (:float :float64) (:complex :complex128)))
                        ((eq k :int)
                         ;; 整数标量：不改变数值 base；bool 例外（升至默认整数）
                         (if (eq base :bool) :int64 base))
                        ((eq k :float)
                         ;; 整型/布尔 base 装不下浮点种类 → 默认浮点；否则保持宽度
                         (if (or (eq base :bool) (%int-family-dtype-p base))
                             :float64 base))
                        ((eq k :complex)
                         (cond ((or (eq base :bool) (%int-family-dtype-p base))
                                :complex128)
                               ((eq base :float32) :complex64)
                               ((eq base :float64) :complex128)
                               (t base))))))
    (or base :float64)))

(defun vt-int-dtype-p (dtype)
  "判定 DTYPE 是否为有符号整数类型（:int8/:int16/:int32/:int64）。"
  (declare (optimize (speed 3) (safety 0)))
  (member dtype '(:int8 :int16 :int32 :int64)))

(defun vt-signed-dtype-p (dtype)
  "判定 DTYPE 是否为有符号类型（当前与 vt-int-dtype-p 一致）。"
  (declare (optimize (speed 3) (safety 0)))
  (member dtype '(:int8 :int16 :int32 :int64)))

(defun vt-unsigned-dtype-p (dtype)
  "判定 DTYPE 是否为无符号类型（:uint8 / :uint16）。"
  (declare (optimize (speed 3) (safety 0)))
  (member dtype '(:uint8 :uint16))) ; 以后扩展 uint32/uint64 时在这里加

(defun vt-int-bits (dtype)
  "返回整数 dtype 的位宽（:int8 → 8 … :int64 → 64）。非整数 dtype 报错。"
  (ecase dtype
    ((:int8 :uint8)   8)
    ((:int16 :uint16) 16)
    (:int32           32)
    (:int64           64)))

(defun vt-bits->signed-dtype (bits)
  "把位宽映射为对应的有符号 dtype（8→:int8, 16→:int16, 32→:int32, 64→:int64）。"
  (declare (optimize (speed 3) (safety 0))
	   (fixnum bits))
  (ecase bits
    (8  :int8)
    (16 :int16)
    (32 :int32)
    (64 :int64)))

(defun vt-bits->unsigned-dtype (bits)
  "返回 bits 位宽对应的无符号 dtype。
   注意：:uint32/:uint64 为保留符号——当前尚不属于 *vt-storage-dtypes*，
   仅在未来扩展 uint32/uint64 存储类型时可达（与 vt-unsigned-dtype-p 成员表一致）。"
  (declare (optimize (speed 3) (safety 0))
	   (fixnum bits))
  (ecase bits
    (8  :uint8)
    (16 :uint16)
    (32 :uint32)    
    (64 :uint64))) 

(defun vt-next-signed-bits (unsigned-bits)
  "返回能容纳 unsigned-bits 无符号整数范围的有符号位宽；若没有则返回 nil。"
  (declare (optimize (speed 3) (safety 0))
	   (fixnum unsigned-bits))
  (ecase unsigned-bits
    (8  16)
    (16 32)
    (32 64)
    (64 nil)))

(defun %promote-real-dtype (dtypes)
  "实数 dtype 的提升内核（复数的分量提升复用同一实现）。

   `:bool` 的位置（numpy 实测，§17.2）：布尔**不贡献位宽**，只在
   「全部操作数都是布尔」时才把结果留在 `:bool`。
     bool ⊔ bool    → bool
     bool ⊔ int8    → int8
     bool ⊔ uint8   → uint8
     bool ⊔ float32 → float32

   实现注记：入参刻意用**一个 list** 而不是 `&rest`。SBCL 2.6.8 在
   「&rest + (optimize (speed 3) (safety 0)) + 由 call-site 用 apply 调用」
   三条件同时成立时会踩到代码生成缺陷（实测：Memory fault at (nil)，
   `(apply #'%promote-real-dtype '(:float64 :int8))` 直接段错误）。
   改成 list 入参后同一调用路径实测正常；本函数因此不与 &rest 组合。"
  (declare (optimize (speed 3) (safety 0)))
  (let ((has-f64 nil)
        (has-f32 nil)
        (has-bool nil)
        (max-signed-bits 0)
        (max-unsigned-bits 0))
    (declare (type fixnum max-signed-bits max-unsigned-bits))
    (dolist (d dtypes)
      (cond
        ((eq d :float64)
         (setf has-f64 t))
        ((eq d :float32)
         (setf has-f32 t))
        ((eq d :bool)
         (setf has-bool t))
        ((vt-signed-dtype-p d)
         (setf max-signed-bits
               (max max-signed-bits (the fixnum (vt-int-bits d)))))
        ((vt-unsigned-dtype-p d)
         (setf max-unsigned-bits
               (max max-unsigned-bits (the fixnum (vt-int-bits d)))))))

    (cond
      ;; float64 优先
      (has-f64 :float64)

      ;; float32 与整数混合：超过 24 位有效位则提升到 float64
      (has-f32
       (if (> (max max-signed-bits max-unsigned-bits) 24)
           :float64
           :float32))

      ;; 有符号 + 无符号混合
      ((and (> max-signed-bits 0) (> max-unsigned-bits 0))
       (if (> max-signed-bits max-unsigned-bits)
           (vt-bits->signed-dtype max-signed-bits)
           (let ((next-bits (vt-next-signed-bits max-unsigned-bits)))
             (if next-bits
                 (vt-bits->signed-dtype next-bits)
                 :float64))))

      ;; 只有有符号整数
      ((> max-signed-bits 0)
       (vt-bits->signed-dtype max-signed-bits))

      ;; 只有无符号整数
      ((> max-unsigned-bits 0)
       (vt-bits->unsigned-dtype max-unsigned-bits))

      ;; 只有布尔（bool ⊔ bool）
      (has-bool :bool)

      ;; 空输入或未知类型兜底
      (t :float64))))

(defun vt-promote-type (&rest dtypes)
  "按 numpy 类型提升规则对若干 dtype 求公共结果 dtype。

   实数部分：64 格已与 numpy 2.3/2.4 实测一致（见 CONVENTIONS §3.2）。
   复数部分（CONVENTIONS §16.3，numpy 2.4.6 逐格实测）：
     complex 的**实部 dtype 参与普通提升**，提升得到的实数结果再映射为
     复数：:float32 → :complex64、:float64 → :complex128。
   该单一规则复现了 numpy 的整张 100 格提升表，例如
     complex64 + int16  → complex64   （f32 ⊔ int16 = f32）
     complex64 + int32  → complex128  （f32 ⊔ int32 = f64，因 24 位尾数规则）
     complex64 + float64→ complex128
     complex64 + complex128 → complex128"
  (declare (optimize (speed 3) (safety 0)))
  (let ((has-complex nil)
        (real-dtypes '()))
    (dolist (d dtypes)
      (cond ((eq d :complex128)
             (setf has-complex t)
             (push :float64 real-dtypes))
            ((eq d :complex64)
             (setf has-complex t)
             (push :float32 real-dtypes))
            (t
             (push d real-dtypes))))
    (let ((real (%promote-real-dtype (nreverse real-dtypes))))
      (if (null has-complex)
          real
          (if (eq real :float64) :complex128 :complex64)))))

(declaim (inline %wrap-int8 %wrap-int16 %wrap-uint8 %wrap-uint16
                 %wrap-int32 %wrap-int64
                 %coerce-int8 %coerce-int16 %coerce-uint8 %coerce-uint16
                 %coerce-int32 %coerce-int64))

;;; 类型声明（性能）：这 12 个函数是**全部**窄化转换的唯一出口
;;; （vt-cast / vt-astype / %cast-to / %vt-map-run 的写回都经过它们）。
;;; 没有返回类型声明时，SBCL 只能把它们的结果当作 INTEGER / REAL，
;;; 于是在热循环写回 specialized-array 时插入 range check 并对结果装箱，
;;; 这是 int8/int16/uint8/uint16 结果张量（比较运算、布尔掩码）慢 ~20x 的根因。
;;; 这里声明的返回类型与各函数体内的 mod/回绕实现严格一致（可证明），
;;; 只是把「已经成立的事实」告诉编译器，不改变任何语义。
(declaim (ftype (function (real) (signed-byte 8))   %wrap-int8)
         (ftype (function (real) (signed-byte 16))  %wrap-int16)
         (ftype (function (real) (signed-byte 32))  %wrap-int32)
         (ftype (function (real) (signed-byte 64))  %wrap-int64)
         (ftype (function (real) (unsigned-byte 8)) %wrap-uint8)
         (ftype (function (real) (unsigned-byte 16)) %wrap-uint16)
         (ftype (function (real) (signed-byte 8))   %coerce-int8)
         (ftype (function (real) (signed-byte 16))  %coerce-int16)
         (ftype (function (real) (signed-byte 32))  %coerce-int32)
         (ftype (function (real) (signed-byte 64))  %coerce-int64)
         (ftype (function (real) (unsigned-byte 8)) %coerce-uint8)
         (ftype (function (real) (unsigned-byte 16)) %coerce-uint16))

(defun %wrap-int8 (v)
  "int8 范围 [-128,127] 回绕（NumPy 语义），NaN/Inf→0。"
  (let ((m (mod (%safe-truncate v) 256)))
    (if (>= m 128) (- m 256) m)))

(defun %wrap-int16 (v)
  "int16 范围 [-32768,32767] 回绕（NumPy 语义），NaN/Inf→0。"
  (let ((m (mod (%safe-truncate v) 65536)))
    (if (>= m 32768) (- m 65536) m)))

(defun %wrap-uint8 (v)
  "uint8 范围 [0,255] 回绕（NumPy 语义），NaN/Inf→0。"
  (mod (%safe-truncate v) 256))

(defun %wrap-uint16 (v)
  "uint16 范围 [0,65535] 回绕（NumPy 语义），NaN/Inf→0。"
  (mod (%safe-truncate v) 65536))

(defun %wrap-int32 (v)
  (let ((m (mod (%safe-truncate v) #.(expt 2 32))))
    (if (>= m #.(expt 2 31))
 	(- m #.(expt 2 32)) m)))

(defun %wrap-int64 (v)
  (let ((m (mod (%safe-truncate v) #.(expt 2 64))))
    (if (>= m #.(expt 2 63))
 	(- m #.(expt 2 64)) m)))

;;; 范围检查 + 回退到 wrap 的快速 coerce（热路径用，避免不必要 mod）
(defun %coerce-int64 (v)
  (let ((n (if (and (floatp v) (%nan-or-inf-p v)) 0 (truncate v))))
    (if (typep n '(signed-byte 64)) n (%wrap-int64 n))))

(defun %coerce-int32 (v)
  (let ((n (if (and (floatp v) (%nan-or-inf-p v)) 0 (truncate v))))
    (if (typep n '(signed-byte 32)) n (%wrap-int32 n))))

(defun %coerce-int16 (v)
  (let ((n (if (and (floatp v) (%nan-or-inf-p v)) 0 (truncate v))))
    (if (typep n '(signed-byte 16)) n (%wrap-int16 n))))

(defun %coerce-int8 (v)
  (let ((n (if (and (floatp v) (%nan-or-inf-p v)) 0 (truncate v))))
    (if (typep n '(signed-byte 8)) n (%wrap-int8 n))))

(defun %coerce-uint8 (v)
  (let ((n (if (and (floatp v) (%nan-or-inf-p v)) 0 (truncate v))))
    (if (typep n '(unsigned-byte 8)) n (%wrap-uint8 n))))

(defun %coerce-uint16 (v)
  (let ((n (if (and (floatp v) (%nan-or-inf-p v)) 0 (truncate v))))
    (if (typep n '(unsigned-byte 16)) n (%wrap-uint16 n))))

;;; ------------------------------------------------------------------
;;; 布尔化原语（CONVENTIONS §17）
;;;
;;; 「真值」是布尔 dtype 的唯一语义入口：把任意元素收敛成 0/1。
;;; 所有产生布尔张量的函数最终都写 0/1，因此这里必须覆盖
;;; 「Lisp 布尔 / 整数 / 浮点（含 NaN、±Inf、-0.0）/ 复数」四种输入形态。
;;; ------------------------------------------------------------------

(declaim (inline %bool-flag))
(defun %bool-flag (v)
  "任意元素 → 布尔 0/1（numpy 的真值语义，§17.1）。

   · `t` / `nil`：直接映射 1 / 0（Lisp 布尔写进布尔张量的自然入口）；
   · 复数：**任一分量非零**即为真（`bool(0+0j)`→0、`bool(nan+0j)`→1）；
   · 浮点 NaN：真（`np.bool_(np.nan)` 为 True，因为 `nan != 0`）；
   · ±Inf：真；`-0.0`：假（`np.bool_(-0.0)` 为 False）；
   · 其它数值：非零即真。

   前置条件：调用方在 with-float-safe 内（`zerop`/`/=` 遇 NaN 需屏蔽陷阱）。"
  (cond ((eq v t) 1)
        ((null v) 0)
        ((complexp v) (if (or (/= (realpart v) 0) (/= (imagpart v) 0)) 1 0))
        ((numberp v) (if (zerop v) 0 1))
        (t (error "%bool-flag: ~s 不是布尔/数值元素" v))))

(declaim (inline %truthy-p))
(defun %truthy-p (v)
  "任意元素 → Lisp 布尔（T / NIL）。与 `%bool-flag` 互为反向：那个收敛到
   物理表示 0/1，这个收敛到 Lisp 语义 T/NIL。

   存在理由是 `VT-ITEM` 的返回类型变了（§17.8）：`(vt-item bool张量)` 现在
   给 T/NIL，而 `VT-COMPRESS` 这类「消费一个未知 dtype 的单元素张量」的
   调用方仍要按「0 为假」判断。**不能**直接把 `(vt-item x)` 丢进 `if` ——
   CL 里 0 是真值，`(if 0 'a 'b)` 给 A。

   与 `%bool-flag` 不同，这里**自带** with-float-safe：NaN 的 ZEROP 在陷阱
   开启时会抛 FLOATING-POINT-INVALID-OPERATION，而本函数的调用点分散在
   索引/归约各处，指望每处都记得包一层不现实。"
  (cond ((eq v t) t)
        ((null v) nil)
        ((numberp v) (with-float-safe (not (zerop v))))
        (t t)))

(defun vt-cast (val dtype)
  "安全类型转换。浮点转整数时执行截断，超出目标整数范围时按 NumPy 语义回绕（mod）。
   NaN/Inf 转整数返回 0（确定性哨兵）。
   复数语义（CONVENTIONS §16.5）：
     - 目标为复数：实部/虚部分别 coerce 到分量浮点类型（实数入参虚部补 0）；
     - 目标为实数/整数且入参为复数：取**实部**后按原规则转换
       （对标 numpy astype 的 ComplexWarning 语义：静默丢弃虚部，
        本库不产生 warning，行为在文档中显式声明）。
   布尔语义（§17.1）：目标为 :bool 时按**真值**收敛（非零→1、零→0、
   NaN→1），而不是像整数那样截断 —— `(vt-astype #(0.5) :bool)` 必须是
   True，`(truncate 0.5)` 会给 0。
   **复数 → :bool 不走「取实部」那条**：真值优先按整体判定（任一分量非零），
   与 `%BOOL-FLAG` 的契约、`VT-ASTYPE` 的数组路径、`(SETF VT-REF)` 一致。
   §17.10.12：把 `:bool` 分支放在剥虚部之前是刻意的 —— 放在之后就得到
   `(vt-cast #C(0 1) :bool)` = 0，而 `(vt-astype … #C(0 1) … :bool)` = 1，
   同一个值在两条入口上给出相反答案。"

  (with-float-safe
    ;; 先判 :bool —— 见 docstring 末段。%bool-flag 自己处理复数（任一分量非零），
    ;; 若先剥虚部再判，虚部就永久丢失了。
    (when (eq dtype :bool)
      (return-from vt-cast (%bool-flag val)))
    ;; Lisp 布尔 → 1/0。九个数值 dtype 都要认，不只是 :bool 目标。
    ;; 此前 `(vt-cast t :float64)` 直接
    ;; TYPE-ERROR: The value T is not of type REAL，而同一件事的数组入口
    ;; `(vt-astype (vt-from-sequence '(t) :dtype :bool) :float64)` 给 1.0 ——
    ;; 又是「同语义两条入口分叉」。受影响的调用点不止一个：
    ;;   (vt-fill float数组 t)、(vt-full '(3) t :dtype :float64)、
    ;;   (setf (vt-ref float数组 0) t)、(vt-* float数组 t) 全部崩在这。
    ;; numpy 一律接受（np.zeros(3)[0] = True → 1.0），故按 1/0 收敛。
    (when (or (eq val t) (null val))
      (setf val (if (eq val t) 1 0)))
    (when (and (complexp val) (not (vt-complex-dtype-p dtype)))
      (setf val (realpart val)))
    (ecase dtype
      (:complex128 (coerce val '(complex double-float)))
      (:complex64  (coerce val '(complex single-float)))
      (:float64 (coerce val 'double-float))
      (:float32 (coerce val 'single-float))
      (:int64   (%wrap-int64 val))
      (:int32   (%wrap-int32 val))
      (:int16   (%wrap-int16 val))
      (:int8    (%wrap-int8 val))
      (:uint8   (%wrap-uint8 val))
      (:uint16  (%wrap-uint16 val)))))

(defun vt-cast-fun (dtype)
  "返回将数值转换为 dtype 的转换函数(与 vt-cast 语义一致：回绕 + NaN 安全)。"
  (ecase dtype
    (:complex128 (lambda (val)
                   (coerce (if (complexp val) val (complex val 0))
                           '(complex double-float))))
    (:complex64  (lambda (val)
                   (coerce (if (complexp val) val (complex val 0))
                           '(complex single-float))))
    (:float64 (lambda (val)
                (coerce (if (complexp val) (realpart val) val) 'double-float)))
    (:float32 (lambda (val)
                (coerce (if (complexp val) (realpart val) val) 'single-float)))
    ;; 布尔目标：真值收敛（§17.1），不是截断。
    (:bool    (function %bool-flag))
    ;; 整数目标：复数入参先取实部（§16.5 astype 语义），再走范围检查 + 回绕。
    (:int64   (lambda (val) (%coerce-int64 (if (complexp val) (realpart val) val))))
    (:int32   (lambda (val) (%coerce-int32 (if (complexp val) (realpart val) val))))
    (:int16   (lambda (val) (%coerce-int16 (if (complexp val) (realpart val) val))))
    ;; 所有整型统一走 %coerce-*（范围检查 + 回绕，语义一致）。
    ;; v0.3.5 修复：:uint16 误用 %wrap-uint16，与 v0.3.2 的统一政策不一致。
    (:int8    (lambda (val) (%coerce-int8 (if (complexp val) (realpart val) val))))
    (:uint8   (lambda (val) (%coerce-uint8 (if (complexp val) (realpart val) val))))
    (:uint16  (lambda (val) (%coerce-uint16 (if (complexp val) (realpart val) val))))))

(defun vt-dtype-default-value (dtype)
  "返回 dtype 对应的零值。"
  (declare (optimize (speed 3) (safety 0)))
  (ecase dtype
    (:complex128 (complex 0.0d0 0.0d0))
    (:complex64  (complex 0.0s0 0.0s0))
    (:float64 0.0d0)
    (:float32 0.0f0)
    ((:int64 :int32 :int16 :int8 :uint8 :uint16 :bool) 0)))

;;; ------------------------------------------------------------------
;;; 复数值级原语（CONVENTIONS §16.6）
;;;
;;; 这一组是「复数语义的唯一出口」：所有复数相关的判定/比较/取模
;;; 必须经由此处，禁止在各函数内散落 `(realpart …)` 的临时写法。
;;; 全部是 (declaim inline) 的单指令级函数，热路径零开销。
;;; ------------------------------------------------------------------

;;; 性能：这一组必须 INLINE。排序/比较/极值内核每元素要调用它们
;;; 1e6~2e7 次，函数调用开销（~5ns/次）会直接主导总耗时：
;;; 实测复数 sort 1M 元素 1370ms 中约 1000ms 是比较器的 funcall。
(declaim (inline %c2 %creal %cimag %cabs %carg %cconj
                 %cisnan %cisinf %cisfinite %cis-zero
                 %c-lex-compare %clt %cle %cgt %cge %cneq))

(defun %c2 (re im)
  "由实部/虚部构造复数（RE 已保证为实数）。"
  (complex re im))

(defun %creal (x)
  "取实部：复数返回 realpart，实数原样返回（对标 numpy.real 的标量语义）。"
  (if (complexp x) (realpart x) x))

(defun %cimag (x)
  "取虚部：复数返回 imagpart，实数返回**同类 +0.0**（对标 numpy.imag）。

   实数分支必须是 +0.0，不能用 `(* 0 x)`：后者对负数给出 **-0.0**
   （实测 (-2.0) → -0.0），而 numpy 的 `np.imag(实数数组)` 恒为 +0.0
   （signbit 全 0）。差分探针 p0392 的 :exact 位级比较会因 -0.0 判不等，
   这也是 §16.6「符号零是语义信息」的另一面：**不该出现 -0.0 的地方不能有**。"
  (cond ((complexp x) (imagpart x))
        ((floatp x) (float 0 x))        ; +0.0，保持单/双精度
        (t 0)))

(defun %cabs (x)
  "复数的模（欧几里得范数）；实数即绝对值。
   numpy 的 abs 对 complex64 返回 float32、complex128 返回 float64，
   CL `abs` 对复数已返回模，二者的数值语义相同（§16.6）。"
  (abs x))

(defun %carg (x)
  "复数幅角 atan2(im, re) ∈ (-π, π]；实数按符号返回 0 / π，NaN 传播。

   numpy 实测（2.4.6）的实数语义：
     angle(0.0)  = 0.0      angle(-0.0) = π（**看符号位**，不看数值比较）
     angle(1.0)  = 0.0      angle(-1.0) = π      angle(nan) = nan
   因此负零必须用 `float-sign` 判定（`minusp` 对 -0.0 返回 NIL，会漏掉），
   非负分支必须返回 +0.0（不能 `(* 0 x)`，那会给 -0.0）。

   复数的 (0,0) 点更麻烦：atan2 在 (±0, ±0) 上按符号位给出 0/±π 四种答案，
   全是 numpy 认可的结果——
     angle( 0.0+0.0i)= 0.0    angle(-0.0+0.0i)= π
     angle( 0.0-0.0i)=-0.0    angle(-0.0-0.0i)=-π
   原先这里对「两分量都为零」统一返回 `(float 0 re)`，把负实轴上的点
   （-0.0+0.0i）判成了 0 而不是 π —— 相差 180°（§17.10.23）。"
  (if (complexp x)
      ;; 复数直接用 atan(y, x)：四象限 atan2 就是 numpy.angle 的定义，
      ;; 而 CL 的 (atan y x) 在 (±0, ±0) 上按 IEEE 754 给出四种答案
      ;; （0 / π / -0.0 / -π），与 numpy 逐位相同。原先为了「零就是零」
      ;; 加了一层 (and (zerop re) (zerop im)) 的短路，反而把 −0.0 的
      ;; 符号位吃掉（§17.10.23）。
      (atan (imagpart x) (realpart x))
      (cond ((and (floatp x) (vt-float-nan-p x)) x)          ; NaN 传播
            ((or (minusp x)
                 (and (floatp x) (minusp (float-sign x))))    ; 负数 或 -0.0 → π
             (coerce pi (if (and (floatp x) (typep x 'single-float))
                            'single-float 'double-float)))
            (t (if (floatp x) (float 0 x) 0)))))

(defun %cconj (x)
  "共轭：复数返回 conj，实数原样返回（对标 numpy.conj）。"
  (if (complexp x) (conjugate x) x))

;;; ------------------------------------------------------------------
;;; 快判原语（**前置条件：调用方已在 with-float-safe 内**）
;;;
;;; 为什么需要这一组：vt-float-nan-p / vt-float-inf-p 内部各包一层
;;; `with-float-traps-masked`，单次约 200ns；而比较/极值/排序内核每元素
;;; 要判 2~4 次分量 —— 1M 元素光「屏蔽/恢复陷阱」就吃掉 ~400-800ms
;;; （实测 %cisnan(复数) 399ms/1M，而真正的 4 次浮点比较只要 34ms）。
;;; 内核全部运行在 vt-map / vt-fast-map / %cx-* 的 with-float-safe 之下，
;;; 陷阱已屏蔽，这里直接 `(not (= x x))` 即可，**不再嵌套屏蔽**。
;;;
;;; 边界函数（用户入口、非屏蔽上下文）仍用带屏蔽的 %cisnan/%cisinf，
;;; 分工见 CONVENTIONS §16.14。
;;; ------------------------------------------------------------------

(declaim (inline %cx-nan-p %cx-inf-p))

(defun %cx-nan-p (x)
  "已屏蔽陷阱上下文中的 NaN 判定（实数返回自身未定义域，复数任一分量）。
   前置条件：调用方在 with-float-safe 内。"
  (if (complexp x)
      (or (not (= (realpart x) (realpart x)))
          (not (= (imagpart x) (imagpart x))))
      (and (floatp x) (not (= x x)))))

(defun %cx-inf-p (x)
  "已屏蔽陷阱上下文中的 ±Inf 判定（实数/复数；复数任一分量）。"
  (flet ((infp (v)
           (and (floatp v) (> (abs v) most-positive-double-float))))
    (if (complexp x)
        (or (infp (realpart x)) (infp (imagpart x)))
        (infp x))))

(defun %cisnan (x)
  "NaN 判定：复数「任一分量为 NaN」即 NaN（对标 numpy.isnan，§16.7）。

   这是**带陷阱屏蔽的边界版本**（单次 ~200ns）：只用于用户入口与任何
   未屏蔽上下文；内核热路径请用 %CX-NAN-P。"
  (if (complexp x)
      (or (vt-float-nan-p (realpart x)) (vt-float-nan-p (imagpart x)))
      (vt-float-nan-p x)))

(defun %cisinf (x)
  "Inf 判定：复数「任一分量为 ±Inf」即 Inf（对标 numpy.isinf，§16.7）。"
  (if (complexp x)
      (or (vt-float-inf-p (realpart x)) (vt-float-inf-p (imagpart x)))
      (vt-float-inf-p x)))

(defun %cisfinite (x)
  "有限性判定：复数的两个分量都有限才为有限（对标 numpy.isfinite，§16.7）。
   边界版本（带屏蔽）；内核里请用 %CX-FINITE-P。"
  (if (complexp x)
      (and (not (vt-float-nan-p (realpart x))) (not (vt-float-inf-p (realpart x)))
           (not (vt-float-nan-p (imagpart x))) (not (vt-float-inf-p (imagpart x))))
      (and (not (vt-float-nan-p x)) (not (vt-float-inf-p x)))))

(declaim (inline %cx-finite-p))
(defun %cx-finite-p (x)
  "已屏蔽陷阱上下文中的有限性判定（非 NaN 且非 Inf）。"
  (not (or (%cx-nan-p x) (%cx-inf-p x))))

(defun %cis-zero (x)
  "零判定：复数要求实部与虚部同时为 +0.0/-0.0（对标 numpy `z == 0`）。"
  (if (complexp x)
      (and (zerop (realpart x)) (zerop (imagpart x)))
      (zerop x)))

;;; 字典序比较（CONVENTIONS §16.8）：
;;;   numpy 2.x 对复数定义了**字典序**（先比实部，实部相等再比虚部），
;;;   `<`/`<=`/`>`/`>=`/sort/argsort/min/max/argmin/argmax 全部遵此语义。
;;;   任一分量为 NaN 时所有有序比较恒假（与实数 NaN 规则一致）。
(declaim (inline %c-lex-compare))
(defun %c-lex-compare (a b)
  "返回 -1 / 0 / 1（字典序）；任一分量含 NaN 时返回 NIL（不可比）。"
  (let ((ar (%creal a)) (ai (%cimag a))
        (br (%creal b)) (bi (%cimag b)))
    ;; 前置条件：调用方在 with-float-safe 内（排序/极值/比较内核皆然）。
    ;; 用 %CX-NAN-P 而非 VT-FLOAT-NAN-P：后者每次都要屏蔽/恢复陷阱，
    ;; 每元素 4 次 → 1M 元素多付约 800ms（实测）。
    (cond ((or (%cx-nan-p ar) (%cx-nan-p br)
               (%cx-nan-p ai) (%cx-nan-p bi))
           nil)
          ((< ar br) -1)
          ((> ar br) 1)
          ((< ai bi) -1)
          ((> ai bi) 1)
          (t 0))))

(defun %clt (a b)
  "字典序小于。NaN 参与时恒假。"
  (let ((c (%c-lex-compare a b))) (and c (minusp c))))

(defun %cle (a b)
  "字典序小于等于（相等判定走精确 `=`，与 numpy 一致：值为 ±0.0 时相等）。"
  (let ((c (%c-lex-compare a b))) (and c (<= c 0))))

(defun %cgt (a b)
  "字典序大于。NaN 参与时恒假。"
  (let ((c (%c-lex-compare a b))) (and c (plusp c))))

(defun %cge (a b)
  "字典序大于等于。NaN 参与时恒假。"
  (let ((c (%c-lex-compare a b))) (and c (>= c 0))))

(defun %cneq (a b)
  "不等判定：任一分量为 NaN → 不等（真），其余用 CL `=`。
   对标 numpy（NaN+1j != NaN+1j）。"
  (if (or (%cisnan a) (%cisnan b))
      t
      (not (= a b))))


;;; ------------------------------------------------------------------
;;; 元素类型分组（编译期常量）
;;; ------------------------------------------------------------------
;;; 放在 dtype.lisp 而不是 reduce-stats.lisp：core.lisp（vt-astype 的窄整型
;;; 特化表）与 reduce-stats.lisp 都要用，而 core.lisp 先编译。
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

  (defparameter +small-int-ets+
    '((signed-byte 16) (signed-byte 8) (unsigned-byte 8) (unsigned-byte 16))
    "小整型的运行时 array-element-type 形态（= +small-int-lts+，别名以示用途）。"))
