;;;; dtype.lisp — 元素数据类型系统（单一事实来源）

(in-package :clvt)

(defparameter *vt-dtypes*
  '(:complex128 :complex64
    :float64 :float32 :int64 :int32 :int16 :int8 :uint8 :uint16)
  "库内识别到的 dtype 全集（前 6 者为物理存储类型）。
   顺序约定：宽 → 窄、类别由高到低（复数 > 浮点 > 整数），与
   `+vt-elem-types+` 的「命中概率降序」排序保持同一风格。
   complex 见 CONVENTIONS §16。")

(defparameter *vt-storage-dtypes*
  '(:complex128 :complex64 :float64 :float32 :int64 :int32)
  "实际可分配底层数组的元素类型（含复数：interleaved 的
   (complex double-float) / (complex single-float) 特化数组）。")

(defparameter *vt-complex-dtypes* '(:complex128 :complex64)
  "复数 dtype 全集（CONVENTIONS §16.1）。")

(declaim
 (inline vt-dtype-p vt-float-dtype-p vt-int-dtype-p vt-storage-dtype-p
         vt-complex-dtype-p vt-complex-real-dtype))

(defun vt-dtype->lisp-type (dtype)
  "将内部 dtype 符号映射为 Common Lisp 数组元素类型。"
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
    (:uint16  '(unsigned-byte 16))))

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
    (:uint16  2)))

(defun vt-dtype-p (x)
  "判定 X 是否为合法逻辑 dtype（见 *vt-dtypes*，共 8 种）。"
  (declare (optimize (speed 3) (safety 0)))
  (not (null (member x *vt-dtypes*))))

(defun vt-storage-dtype-p (x)
  "判定 X 是否为物理存储 dtype（见 *vt-storage-dtypes*，共 4 种：float64/float32/int64/int32）。"
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

   实现注记：入参刻意用**一个 list** 而不是 `&rest`。SBCL 2.6.8 在
   「&rest + (optimize (speed 3) (safety 0)) + 由 call-site 用 apply 调用」
   三条件同时成立时会踩到代码生成缺陷（实测：Memory fault at (nil)，
   `(apply #'%promote-real-dtype '(:float64 :int8))` 直接段错误）。
   改成 list 入参后同一调用路径实测正常；本函数因此不与 &rest 组合。"
  (declare (optimize (speed 3) (safety 0)))
  (let ((has-f64 nil)
        (has-f32 nil)
        (max-signed-bits 0)
        (max-unsigned-bits 0))
    (declare (type fixnum max-signed-bits max-unsigned-bits))
    (dolist (d dtypes)
      (cond
        ((eq d :float64)
         (setf has-f64 t))
        ((eq d :float32)
         (setf has-f32 t))
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

(defun vt-cast (val dtype)
  "安全类型转换。浮点转整数时执行截断，超出目标整数范围时按 NumPy 语义回绕（mod）。
   NaN/Inf 转整数返回 0（确定性哨兵）。
   复数语义（CONVENTIONS §16.5）：
     - 目标为复数：实部/虚部分别 coerce 到分量浮点类型（实数入参虚部补 0）；
     - 目标为实数/整数且入参为复数：取**实部**后按原规则转换
       （对标 numpy astype 的 ComplexWarning 语义：静默丢弃虚部，
        本库不产生 warning，行为在文档中显式声明）。"
  (with-float-safe
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
    (:complex64  (complex 0.0 0.0))
    (:float64 0.0d0)
    (:float32 0.0f0)
    ((:int64 :int32 :int16 :int8 :uint8 :uint16) 0)))

;;; ------------------------------------------------------------------
;;; 复数值级原语（CONVENTIONS §16.6）
;;;
;;; 这一组是「复数语义的唯一出口」：所有复数相关的判定/比较/取模
;;; 必须经由此处，禁止在各函数内散落 `(realpart …)` 的临时写法。
;;; 全部是 (declaim inline) 的单指令级函数，热路径零开销。
;;; ------------------------------------------------------------------

(declaim (inline %c2 %creal %cimag %cabs %carg %cconj
                 %cisnan %cisinf %cisfinite %cis-zero
                 %clt %cle %cgt %cge %cneq))

(defun %c2 (re im)
  "由实部/虚部构造复数（RE 已保证为实数）。"
  (complex re im))

(defun %creal (x)
  "取实部：复数返回 realpart，实数原样返回（对标 numpy.real 的标量语义）。"
  (if (complexp x) (realpart x) x))

(defun %cimag (x)
  "取虚部：复数返回 imagpart，实数返回同类零（对标 numpy.imag）。"
  (if (complexp x) (imagpart x) (* 0 x)))

(defun %cabs (x)
  "复数的模（欧几里得范数）；实数即绝对值。
   numpy 的 abs 对 complex64 返回 float32、complex128 返回 float64，
   CL `abs` 对复数已返回模，二者的数值语义相同（§16.6）。"
  (abs x))

(defun %carg (x)
  "复数幅角 atan2(im, re) ∈ (-π, π]；实数按符号返回 0 / π / -π（对标 numpy.angle）。
   注意 numpy：angle(-1.0) = π、angle(-1.0-0.0j) = -π（尊重 -0.0 虚部符号）。"
  (if (complexp x)
      (let ((re (realpart x)) (im (imagpart x)))
        (cond ((and (zerop re) (zerop im)) (* 0 re))
              (t (atan im re))))
      (if (minusp x) (let ((p (coerce pi (type-of (float x))))) p) (* 0 x))))

(defun %cconj (x)
  "共轭：复数返回 conj，实数原样返回（对标 numpy.conj）。"
  (if (complexp x) (conjugate x) x))

(defun %cisnan (x)
  "NaN 判定：复数「任一分量为 NaN」即 NaN（对标 numpy.isnan，§16.7）。"
  (if (complexp x)
      (or (vt-float-nan-p (realpart x)) (vt-float-nan-p (imagpart x)))
      (vt-float-nan-p x)))

(defun %cisinf (x)
  "Inf 判定：复数「任一分量为 ±Inf」即 Inf（对标 numpy.isinf，§16.7）。"
  (if (complexp x)
      (or (vt-float-inf-p (realpart x)) (vt-float-inf-p (imagpart x)))
      (vt-float-inf-p x)))

(defun %cisfinite (x)
  "有限性判定：复数的两个分量都有限才为有限（对标 numpy.isfinite，§16.7）。"
  (if (complexp x)
      (and (not (vt-float-nan-p (realpart x))) (not (vt-float-inf-p (realpart x)))
           (not (vt-float-nan-p (imagpart x))) (not (vt-float-inf-p (imagpart x))))
      (and (not (vt-float-nan-p x)) (not (vt-float-inf-p x)))))

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
    (cond ((or (vt-float-nan-p ar) (vt-float-nan-p br)
               (vt-float-nan-p ai) (vt-float-nan-p bi))
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
