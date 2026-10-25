(in-package :clvt)

(defun %pow-odd-int-p (y)
  "y 是否为奇整数值（支持浮点表示的整数，如 -3.0d0）。"
  (cond ((integerp y) (oddp y))
        ((and (floatp y) (= y (ftruncate y))) (oddp (truncate y)))
        (t nil)))

(defun %pow-int-valued-p (y)
  "y 是否为整数值（整数或整值浮点）。"
  (or (integerp y)
      (and (floatp y) (= y (ftruncate y)))))

(defun %pow-op (x y nan)
  "逐元素幂的 C99 pow 特例表（对标 numpy.power 实数语义）。
   覆盖：pow(x,±0)=1、pow(±1,y)=1、pow(±0,y)=±0/±Inf（奇偶）、
   pow(±Inf,y)=±Inf/±0/NaN（整值奇偶）、pow(负底数,非整指数)=NaN。
   x/y 可为整数或浮点（vt-map 原始元素）；结果由写入层按 dtype 转换。"
  (labels ((pos-inf () sb-ext:double-float-positive-infinity)
           (neg-inf () sb-ext:double-float-negative-infinity))
    (cond
      ;; 1) pow(x, ±0) = 1（C99，含 NaN 底数与 ±Inf 底数）
      ((and (numberp y) (zerop y)) 1)
      ;; 2) NaN 底数
      ((and (floatp x) (%nan-p x)) x)
      ;; 3) pow(1, y) = 1（C99：含 y=NaN）
      ((and (numberp x) (= x 1)) 1)
      ;; 4) NaN 指数
      ((and (floatp y) (%nan-p y)) y)
      ;; 5) ±Inf 底数（y 非零非 NaN）
      ((and (floatp x) (%inf-p x))
       (cond ((not (%pow-int-valued-p y))
              (if (plusp x) (pos-inf) nan))       ; -Inf**非整数 → NaN
             ((minusp y)
              (if (and (minusp x) (%pow-odd-int-p y)) -0.0d0 0.0d0))
             (t
              (if (and (minusp x) (%pow-odd-int-p y)) (neg-inf) (pos-inf)))))
      ;; 6) ±Inf 指数（x 有限；x=±1 已被 3) 拦截）
      ((and (floatp y) (%inf-p y))
       (if (< (abs x) 1)
           (if (plusp y) 0.0d0 (pos-inf))
           (if (plusp y) (pos-inf) 0.0d0)))
      ;; 7) x = ±0（y 非零非 NaN）：正指数 → ±0，负指数 → ±Inf（奇偶定号）
      ((and (numberp x) (zerop x))
       (if (plusp y)
           (if (and (minusp x) (%pow-odd-int-p y)) -0.0d0 0.0d0)
           (if (and (minusp x) (%pow-odd-int-p y)) (neg-inf) (pos-inf))))
      ;; 8) 负底数 + 非整值指数 → NaN
      ((and (minusp x) (not (%pow-int-valued-p y))) nan)
      ;; 9) 常规幂（整数指数 → 精确；负整数指数的整数对由调用方处理）
      (t (expt x y)))))

(defun vt-pow (vt power &key out dtype)
  "逐元素幂 VT**POWER。POWER 为标量指数或张量（张量时逐元素广播幂，
      对标 numpy.power，v0.4.x 起支持）。
      整数 + 非负整数指数 → 精确整数幂，保持整数 dtype；负整数指数按
      numpy 整型语义返回 0。其余走浮点幂并按 C99 pow 特例表处理
      非有限值（pow(-8,1/3)=NaN、pow(0,-1)=+Inf、pow(1,NaN)=1 等）。
      :out 契约同 vt-+。"
  (when (%cx-any-p vt power)
    (return-from vt-pow (%cx-op-pow vt power :out out :dtype dtype)))
  (let ((vt (ensure-vt vt)))
    (if (vt-p power)
        ;; ---- 张量指数：逐元素广播幂（自然提升，与 numpy.power 一致）----
        (let* ((promoted (if dtype
                             dtype
                             (vt-promote-type (vt-dtype vt) (vt-dtype power))))
               (int-result (member promoted
                                   '(:int8 :int16 :int32 :int64 :uint8 :uint16)))
               (nan (vt-get-nan :float64)))
          (if int-result
              ;; numpy 整型幂：负整数指数 → 0（C99 pow 后截断为整型）
              (vt-map (lambda (x y) (if (minusp y) 0 (expt x y)))
                      vt power :out out :dtype dtype)
              (vt-map (lambda (x y) (%pow-op x y nan))
                      vt power :out out :dtype dtype)))
        ;; ---- 标量指数：原路径 ----
        (let* ((in-dt (vt-dtype vt))
               (int-pow-p (and (integerp power) (plusp power)))
               ;; 整数基 + 正整数指数 → 保持整数 dtype（对标 numpy）；
               ;; 其余一律浮点（float32 进 float32 出，整数→float64）。
               (dt (cond (dtype dtype)
                         ((and int-pow-p (member in-dt '(:int8 :int16 :int32 :int64
                                                         :uint8 :uint16)))
                          in-dt)
                         ((eq in-dt :float32) :float32)
                         (t :float64)))
               (nan (vt-get-nan (if (member dt '(:float64 :float32)) dt :float64))))
          (cond
            (int-pow-p
             (vt-map (lambda (x) (expt x power)) vt :out out :dtype dt))
            (t
             (let ((vf (if (member dt '(:float64 :float32)) vt (vt-astype vt dt))))
               (vt-map (lambda (x)
                         (let ((result (handler-case (expt x power)
                                         (error () nan))))
                           (if (realp result) result nan)))
                       vf :out out :dtype dt))))))))

(defun vt-expt (vt power &key out dtype)
  "vt-pow 的别名（CL 习惯命名）。语义与参数契约见 vt-pow。"
  (vt-pow vt power :out out :dtype dtype))

(defun vt-square (vt &key out dtype)
  "逐元素平方。特化为 vt-fast-map #'* 以避免 vt-pow 的 handler-case/expt/realp 开销。"
  (when (%cx-any-p vt)
    (return-from vt-square
      (let* ((in (ensure-vt vt))
             (dt (%cx-result-complex-dtype (vt-dtype in) dtype "VT-SQUARE")))
        (vt-fast-map (function %cop-square) in :out out :dtype dt))))
  (vt-fast-map #'* vt vt :out out :dtype dtype))
  "逐元素平方。特化为 vt-fast-map #'* 以避免 vt-pow 的
      handler-case/expt/realp 开销。:out 契约同 vt-+。"

(defun vt-sqrt (vt &key out dtype)
  "逐元素平方根。负数输入返回 NaN（对标 numpy，不抛条件）。
      整数输入输出 float64，float32 进 float32 出。:out 契约同 vt-+。"
  (when (%cx-any-p vt)
    (return-from vt-sqrt
      (let* ((in (ensure-vt vt))
             (dt (%cx-result-complex-dtype (vt-dtype in) dtype "VT-SQRT")))
        (vt-fast-map (function %cop-sqrt) in :out out :dtype dt))))
(let ((dt (%infer-float-dtype vt dtype)))
    (%float-map-pair %op-sqrt-df %op-sqrt-sf vt out dt)))

(defun vt-log (vt &key base out dtype)
  "逐元素自然对数；:base 给定时为换底对数。
      语义对标 numpy.log：x>0 正常；x=0 → -Inf（或 log(base)<0 时 +Inf）；
      x<0 → NaN；非法 base（<=0 或 =1）→ 全 NaN。
      整数输入输出 float64，float32 进 float32 出。:out 契约同 vt-+。

  性能：base 缺省 / base=2.0d0 / base=10.0d0 三种最常见情形走
  `%float-map-pair` 的编译期内联循环（原实现对每种情形都用 lambda，
  一律回退 vt-map 的 funcall 路径，1M 元素实测 log 57ms / log2 103ms）；
  任意其它 base 仍是 lambda 路径（base 是运行期变量，无法内联）。"
  ;; 判据用 %CX-ANY-P 而非 %CX-TENSOR-P：后者对 list 返回 NIL，
  ;; (vt-log (list #C(0 1))) 会掉进实数路径（log(-1+0i) 从 πi 变成 NaN）。
  (when (%cx-any-p vt)
    (return-from vt-log
      (let* ((in (ensure-vt vt))
             (dt (%cx-result-complex-dtype (vt-dtype in) dtype "VT-LOG")))
        (if base
            (vt-map (lambda (z) (%cop-log-base z base)) in :out out :dtype dt)
            ;; 无 base 是常见情形：走 vt-fast-map 的内联循环（原先误用 vt-map，
            ;; 每元素一次 funcall + 泛型 aref —— %cop-log 在白名单里也白搭）。
            (vt-fast-map (function %cop-log) in :out out :dtype dt)))))
  (let ((dt (%infer-float-dtype vt dtype)))
    (cond
      ((null base)
       (%float-map-pair %op-log-df %op-log-sf vt out dt))
      ((eql base 2.0d0)
       (%float-map-pair %op-log2-df %op-log2-sf vt out dt))
      ((eql base 10.0d0)
       (%float-map-pair %op-log10-df %op-log10-sf vt out dt))
      ((or (<= base 0) (= base 1))
       (let ((nan (vt-get-nan dt)))
         (vt-map (lambda (x) (declare (ignore x)) nan)
                 (%coerce-float-input (ensure-vt vt) dt) :out out :dtype dt)))
      (t
       (let* ((nan (vt-get-nan dt))
              (neginf (vt-get-neg-inf dt))
              (posinf (vt-get-pos-inf dt))
              (zero-result (if (plusp (log base)) neginf posinf)))
         (vt-map (lambda (x)
                   (if (> x 0)
                       (log x base)
                       (if (zerop x) zero-result nan)))
                 (%coerce-float-input (ensure-vt vt) dt) :out out :dtype dt))))))

(defun vt-log10 (vt &key out dtype)
  "逐元素常用对数（底 10）。语义等价 (vt-log vt :base 10)。
      :out 契约同 vt-+。"
  (when (%cx-any-p vt)
    (return-from vt-log10
      (let* ((in (ensure-vt vt))
             (dt (%cx-result-complex-dtype (vt-dtype in) dtype "VT-LOG10")))
        (vt-fast-map (function %cop-log10) in :out out :dtype dt))))
  (%float-map-pair %op-log10-df %op-log10-sf vt out
                   (%infer-float-dtype vt dtype)))

(defun vt-log2 (vt &key out dtype)
  "逐元素二进制对数（底 2）。语义等价 (vt-log vt :base 2)。
      :out 契约同 vt-+。"
  (when (%cx-any-p vt)
    (return-from vt-log2
      (let* ((in (ensure-vt vt))
             (dt (%cx-result-complex-dtype (vt-dtype in) dtype "VT-LOG2")))
        (vt-fast-map (function %cop-log2) in :out out :dtype dt))))
  (%float-map-pair %op-log2-df %op-log2-sf vt out
                   (%infer-float-dtype vt dtype)))

(defun vt-abs (vt &key out dtype)
  "逐元素绝对值。整数保持整数 dtype（对标 numpy.abs）。
      :out 契约同 vt-+。"
  (when (%cx-any-p vt)
    (return-from vt-abs (%cx-op-abs vt :out out :dtype dtype)))
  (vt-fast-map #'abs vt :out out :dtype dtype))

(declaim (inline %sign1))
(defun %sign1 (x)
  "符号函数：-1/0/+1，NaN 传播，浮点零一律返回 +0.0（对标 numpy.sign）。
   浮点分支用 (float 1 x) 而非字面 1.0d0 —— 后者写进 single-float 数组
   会触发 specialized-array 类型错误。"
  (cond ((%nan-p x) x)
        ((zerop x)    (if (floatp x) (float 0.0 x) 0))
        ((plusp x)    (if (floatp x) (float 1 x) 1))
        (t            (if (floatp x) (float -1 x) -1))))

(defun vt-signum (vt &key out dtype)
  "逐元素符号函数（-1 / 0 / +1）。整数保持整数 dtype，
      NaN 原样传播。:out 契约同 vt-+。
      v0.4.x 修复：signum(-0.0) 原 CL 语义返回 -0.0，现对浮点零
      统一返回 +0.0（对标 numpy.sign：sign(-0.0)=0.0）。
      v0.4.3：改用内联字面算子 %sign1（原匿名 lambda 让 vt-map 退回
      funcall + 泛型 aref，1M 元素 69ms）。"
  (when (%cx-any-p vt)
    (return-from vt-signum
      (let* ((in (ensure-vt vt))
             (dt (%cx-result-complex-dtype (vt-dtype in) dtype "VT-SIGNUM")))
        (vt-fast-map (function %cop-signum) in :out out :dtype dt))))
  (vt-fast-map #'%sign1 vt :out out :dtype dtype))

(defun vt-positive-p (vt &key out (dtype :int8))
  "逐元素正数判定，返回值 1/0，dtype 默认 :int8（承载布尔语义）。
      对标 numpy 的符号过滤语义（>0）。NaN 比较恒假 → 0。:out 契约同 vt-=。"
  (vt-map (lambda (v)
            (if (> v 0.0d0) 1 0))
          vt :out out :dtype dtype))

(defun vt-negative-p (vt &key out (dtype :int8))
  "逐元素负数判定，返回值 1/0，dtype 默认 :int8（承载布尔语义）。
      NaN 比较恒假 → 0。:out 契约同 vt-=。"
  (vt-map (lambda (v)
            (if (< v 0.0d0) 1 0))
          vt :out out :dtype dtype))

(defun vt-zero-p (vt &key out (dtype :int8))
  "逐元素零判定，返回值 1/0，dtype 默认 :int8（承载布尔语义）。
      :out 契约同 vt-=。"
  (vt-map (lambda (v)
            (if (zerop v) 1 0))
          vt :out out :dtype dtype))

(defun vt-nonzero-p (vt &key out (dtype :int8))
  "逐元素非零判定，返回值 1/0，dtype 默认 :int8（承载布尔语义）。
      :out 契约同 vt-=。"
  (vt-map (lambda (v)
            (if (zerop v) 0 1))
          vt :out out :dtype dtype))

(defun vt-even-p (vt &key out (dtype :int8))
  "逐元素偶数判定，返回值 1/0，dtype 默认 :int8（承载布尔语义）。
      NaN/±Inf 视为非偶数 → 0；非整数值视为非偶数 → 0
      （对标模拟基准 v%2==0）。:out 契约同 vt-=。"
    (when (%cx-reject vt)
    (error "vt-even-p: 不支持复数输入（numpy 同样拒绝）。见 CONVENTIONS §16.11"))
(vt-map (lambda (v)
            (cond ((or (%nan-p v) (%inf-p v)) 0)
                  ((and (floatp v) (/= v (floor v))) 0)
                  (t (if (evenp (floor v)) 1 0))))
          vt :out out :dtype dtype))

(defun vt-odd-p (vt &key out (dtype :int8))
  "逐元素奇数判定，返回值 1/0，dtype 默认 :int8（承载布尔语义）。
      NaN/±Inf 视为非奇数 → 0；非整数值视为非奇数 → 0
      （对标模拟基准 v%2==1）。:out 契约同 vt-=。"
    (when (%cx-reject vt)
    (error "vt-odd-p: 不支持复数输入（numpy 同样拒绝）。见 CONVENTIONS §16.11"))
(vt-map (lambda (v)
            (cond ((or (%nan-p v) (%inf-p v)) 0)
                  ((and (floatp v) (/= v (floor v))) 0)
                  (t (if (oddp (floor v)) 1 0))))
          vt :out out :dtype dtype))

(defun %mod-nan-or-inf-p (x)
  "取模语境下的非有限值判定（需已屏蔽浮点陷阱）。"
  (and (floatp x) (%nan-or-inf-p x)))

(declaim (inline %mod-float-nan))
(defun %mod-float-nan (x)
  "取模遇 NaN 时的返回值。写入按输出 dtype 转换，恒返 double NaN 即可。"
  (declare (ignore x))
  +vt-dfloat-nan+)

(defun %float-exact-rational (x)
  "浮点的精确有理数值（integer-decode-float）。
   注意不能用 rationalize：它取区间内最简分数而非精确值。"
  (multiple-value-bind (sig exp sign) (integer-decode-float x)
    (* sig (expt 2 exp) sign)))

(defun %float-fmod-exact (x y)
  "精确 fmod（IEEE 754：fmod 结果必然可被输入格式精确表示，coerce 无舍入）。
   返回与 x 同号。仅用于 float32 语境（任一操作数为 single-float）；
   float64 语境的 (mod double double) 误差 ≤ 1ulp，走容差路径。"
  (let ((rx (%float-exact-rational x)) (ry (%float-exact-rational y)))
    (multiple-value-bind (q r) (truncate rx ry)
      (declare (ignore q))
      (coerce r (if (or (typep x 'double-float) (typep y 'double-float))
                    'double-float 'single-float)))))

(defun %exact-float-mod (x y)
  "浮点 mod（结果与除数同号，对标 numpy.remainder）。
   v0.4.x 修复：实测 SBCL 的 (mod single single) 存在数十 ulp 舍入误差，
   且 numpy 的 float32 remainder 并非精确 fmod——其算法为：
     r = fmod(a, b)（精确）
     若 r ≠ 0 且 sign(r) ≠ sign(b)：r = r + b（float32 算术，含一次舍入）
     修正后若符号仍与 b 不符（加法进位所致）→ 0
   本函数逐算法复刻该语义；float64 语境沿用 (mod double double)
   （误差 ≤ 1ulp，通过容差比对）。混合整型操作数沿用 CL 语义。"
  (if (and (floatp x) (floatp y)
           (or (typep x 'single-float) (typep y 'single-float)))
      (let ((r (%float-fmod-exact x y)))
        (cond ((= r 0.0f0) r)
              ((or (and (plusp r) (minusp y))
                   (and (minusp r) (plusp y)))
               (let ((r2 (+ r y)))
                 (if (or (and (plusp r2) (minusp y))
                         (and (minusp r2) (plusp y)))
                     0.0f0
                     r2)))
              (t r)))
      (mod x y)))

(defun %exact-float-rem (x y)
  "浮点 rem（结果与被除数同号，对标 numpy.fmod = C fmodf，精确）。"
  (if (and (floatp x) (floatp y)
           (or (typep x 'single-float) (typep y 'single-float)))
      (%float-fmod-exact x y)
      (rem x y)))

(defun vt-mod (vt divisor &key out dtype)
  "逐元素取模。DIVISOR 可为标量或张量（自动广播）。
   语义对标 numpy.remainder：结果与除数同号；
   除数为 0 时按 numpy dtype 语义（v0.3.6 起）：
   浮点语境（任一操作数为浮点）→ NaN；整型语境 → 0；
   NaN 任一侧出现 → NaN；被除数为 ±Inf → NaN；除数为 ±Inf → 被除数本身。"
    (when (%cx-reject vt divisor)
    (error "vt-mod: 不支持复数输入（numpy 同样拒绝）。见 CONVENTIONS §16.11"))
(flet ((zero-div (x)
           ;; numpy 语义：浮点语境零除 → NaN，整型语境 → 0
           (if (floatp x) (%mod-float-nan x) 0)))
    (if (numberp divisor)
  (vt-map (lambda (x)
                (cond ((%mod-nan-or-inf-p x) (%mod-float-nan x))
                      ((%mod-nan-or-inf-p divisor) (%mod-float-nan x))
                      ((and (numberp divisor) (zerop divisor)) (zero-div x))
                      ((and (floatp divisor) (%inf-p divisor)) x)
                      (t (%exact-float-mod x divisor))))
              vt :out out :dtype dtype)
      (vt-map (lambda (x y)
                (cond ((%mod-nan-or-inf-p x) (%mod-float-nan x))
                      ((%mod-nan-or-inf-p y) (%mod-float-nan x))
                      ((zerop y) (zero-div x))
                      ((and (floatp y) (%inf-p y)) x)
                      (t (%exact-float-mod x y))))
              vt divisor :out out :dtype dtype))))

(defun vt-rem (vt divisor &key out dtype)
  "逐元素余数。DIVISOR 可为标量或张量（自动广播）。
   CL:rem 为截断除法余数（与被除数同号），对标 numpy.fmod
   （docstring 曾误标 numpy.remainder，现已更正）。
   除数为 0 时按 numpy dtype 语义（v0.3.6 起）：
   浮点语境（任一操作数为浮点）→ NaN；整型语境 → 0；
   NaN 任一侧 → NaN；被除数为 ±Inf → NaN；除数为 ±Inf → 被除数本身。"
    (when (%cx-reject vt divisor)
    (error "vt-rem: 不支持复数输入（numpy 同样拒绝）。见 CONVENTIONS §16.11"))
(flet ((zero-div (x)
           ;; numpy 语义：浮点语境零除 → NaN，整型语境 → 0
           (if (floatp x) (%mod-float-nan x) 0)))
    (if (numberp divisor)
  (vt-map (lambda (x)
                (cond ((%mod-nan-or-inf-p x) (%mod-float-nan x))
                      ((%mod-nan-or-inf-p divisor) (%mod-float-nan x))
                      ((zerop divisor) (zero-div x))
                      ((and (floatp divisor) (%inf-p divisor)) x)
                      (t (%exact-float-rem x divisor))))
              vt :out out :dtype dtype)
      (vt-map (lambda (x y)
                (cond ((%mod-nan-or-inf-p x) (%mod-float-nan x))
                      ((%mod-nan-or-inf-p y) (%mod-float-nan x))
                      ((zerop y) (zero-div x))
                      ((and (floatp y) (%inf-p y)) x)
                      (t (%exact-float-rem x y))))
              vt divisor :out out :dtype dtype))))

(defun vt-atan2 (vty vtx &key out dtype)
  "逐元素双参数反正切 atan2(y, x)。参数顺序 VTY, VTX。
      整数输入输出 float64，float32 进 float32 出。:out 契约同 vt-+。"
    (when (%cx-reject vty vtx)
    (error "vt-atan2: 不支持复数输入（numpy 同样拒绝）。见 CONVENTIONS §16.11"))
(vt-fast-map #'atan vty vtx :out out :dtype dtype))

;; floor 族对 NaN/±Inf 统一传播本身（对标 numpy：floor/ceil/trunc/round(nan)=nan，
;; floor(±inf)=±inf）。SBCL 对非有限值直接调用 floor/round 会 signal
;; FLOATING-POINT-INVALID-OPERATION（trap 屏蔽也无法避免），必须显式拦截。
(defmacro %floor-family-body (x op)
  ;; 对标 numpy：结果为 ±0 时符号跟随输入（floor/trunc(-0.5)=-0.0，
  ;; ceil(-0.5)=-0.0）；CL floor 族对 (-0.5,0) 区间返回泛化整数 0（恒 +0），
  ;; 需显式恢复负零符号。
  `(if (and (floatp ,x) (%nan-or-inf-p ,x))
       ,x
       (let ((res (nth-value 0 (,op ,x divisor))))
         (if (floatp ,x)
             (if (and (zerop res) (minusp (float-sign ,x)))
                 (- (float 0.0 ,x))
                 (float res ,x))
             res))))

;;; ------------------------------------------------------------------
;;; 取整族的 DIVISOR=1 特化（v0.4.3 性能修复）
;;;
;;; 原实现四个函数统一写成
;;;     (vt-map (lambda (x) (%floor-family-body x op)) vt ...)
;;; 闭包会让 vt-fast-map 的算子内联**彻底失效** —— 该宏只认 #'symbol
;;; 形式字面算子，遇到 (lambda ...) 直接 `(apply #'vt-map ...)` 回退，
;;; 于是循环里既没有数组元素类型声明、又是逐元素 funcall，还要为每个
;;; 元素求值一次 (%nan-or-inf-p ...)。实测 1M 元素 69ms，而同类的一元
;;; 逐元素 vt-abs（用 #'abs 走内联）只要 1.7ms —— 差 40 倍；
;;; numpy.floor 是 0.19ms，差 370 倍。
;;;
;;; 改法：把 DIVISOR=1（默认值，也是 numpy 唯一存在的形态）固化成
;;; 单参**命名**函数，交给 vt-fast-map 内联进元素类型循环；DIVISOR≠1
;;; 是本库扩展（numpy 无对应物），仍走原闭包路径，语义逐位不变。
;;; ------------------------------------------------------------------
(defmacro %floor-family-body1 (x op)
  "DIVISOR 恒为 1 时 %floor-family-body 的等价展开（供内联快路径使用）。"
  `(if (and (floatp ,x) (%nan-or-inf-p ,x))
       ,x
       (let ((res (nth-value 0 (,op ,x))))
         (if (floatp ,x)
             (if (and (zerop res) (minusp (float-sign ,x)))
                 (- (float 0.0 ,x))
                 (float res ,x))
             res))))

(declaim (inline %floor1 %ceiling1 %round1 %truncate1))
(defun %floor1 (x)    (%floor-family-body1 x floor))
(defun %ceiling1 (x)  (%floor-family-body1 x ceiling))
(defun %round1 (x)    (%floor-family-body1 x round))
(defun %truncate1 (x) (%floor-family-body1 x truncate))

(defun vt-floor (vt &key (divisor 1) out dtype)
  "逐元素向下取整（可除 DIVISOR）。
      NaN/±Inf 原样传播（对标 numpy.floor）；整数输入保持整数 dtype。
      :out 契约同 vt-+。"
    (when (%cx-reject vt)
    (error "vt-floor: 不支持复数输入（numpy 同样拒绝）。见 CONVENTIONS §16.11"))
(if (eql divisor 1)
      (vt-fast-map #'%floor1 vt :out out :dtype dtype)
      (vt-map (lambda (x) (%floor-family-body x floor))
              vt :out out :dtype dtype)))

(defun vt-ceiling (vt &key (divisor 1) out dtype)
  "逐元素向上取整（可除 DIVISOR）。
      NaN/±Inf 原样传播（对标 numpy.ceil）；整数输入保持整数 dtype。
      :out 契约同 vt-+。"
    (when (%cx-reject vt)
    (error "vt-ceiling: 不支持复数输入（numpy 同样拒绝）。见 CONVENTIONS §16.11"))
(if (eql divisor 1)
      (vt-fast-map #'%ceiling1 vt :out out :dtype dtype)
      (vt-map (lambda (x) (%floor-family-body x ceiling))
              vt :out out :dtype dtype)))

(defun vt-round (vt &key (divisor 1) out dtype)
  "逐元素四舍五入取整（可除 DIVISOR）。
      NaN/±Inf 原样传播（对标 numpy.round）；整数输入保持整数 dtype。
      注意：舍入策略为「最近偶数」，与 CL round 一致。"
  (when (%cx-any-p vt)
    (return-from vt-round (%cx-op-round vt :divisor divisor :out out :dtype dtype)))
  (if (eql divisor 1)
      (vt-fast-map #'%round1 vt :out out :dtype dtype)
      (vt-map (lambda (x) (%floor-family-body x round))
              vt :out out :dtype dtype)))

(defun vt-truncate (vt &key (divisor 1) out dtype)
  "逐元素向零取整（可除 DIVISOR）。
      NaN/±Inf 原样传播（对标 numpy.trunc）；整数输入保持整数 dtype。
      :out 契约同 vt-+。"
    (when (%cx-reject vt)
    (error "vt-truncate: 不支持复数输入（numpy 同样拒绝）。见 CONVENTIONS §16.11"))
(if (eql divisor 1)
      (vt-fast-map #'%truncate1 vt :out out :dtype dtype)
      (vt-map (lambda (x) (%floor-family-body x truncate))
              vt :out out :dtype dtype)))

(defun vt-rint (vt &key out dtype)
  "逐元素取整到最近偶数（对标 numpy.rint）。
      NaN/±Inf 原样传播；整数输入保持整数 dtype。:out 契约同 vt-+。"
    (when (%cx-reject vt)
    (error "vt-rint: 不支持复数输入（numpy 同样拒绝）。见 CONVENTIONS §16.11"))
(vt-map (lambda (x)
            (if (and (floatp x) (%nan-or-inf-p x))
                x
                (let ((res (nth-value 0 (round x))))
                  (if (floatp x)
                      (if (and (zerop res) (minusp (float-sign x)))
                          (- (float 0.0 x))
                          (float res x))
                      res))))
          vt :out out :dtype dtype))

#+sbcl (sb-ext::gc :full t) ;; 在编译期间垃圾回收内存，防止爆炸
