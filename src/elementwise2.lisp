(in-package :clvt)

(declaim (inline %op-eq %op-ne %op-lt %op-le %op-gt %op-ge))
(defun %op-eq (a b) (if (=  a b) 1.0d0 0.0d0))

(defun %op-ne (a b) (if (/= a b) 1.0d0 0.0d0))

(defun %op-lt (a b) (if (<  a b) 1.0d0 0.0d0))

(defun %op-le (a b) (if (<= a b) 1.0d0 0.0d0))

(defun %op-gt (a b) (if (>  a b) 1.0d0 0.0d0))

(defun %op-ge (a b) (if (>= a b) 1.0d0 0.0d0))

(defun vt-= (t1 t2 &key (dtype :bool) out)
  "逐元素相等判定，返回 **:bool** 布尔张量（CONVENTIONS §17）。
      NaN 与任何值比较恒假 → False（含 NaN == NaN）。
      :out dtype 必须精确等于结果 dtype（§4.2 H3，默认 :bool）；
      显式传 :dtype :int8 仍可用，得到旧的 0/1 整型掩码。"
  (when (%cx-any-p t1 t2)
    (return-from vt-=
      (vt-fast-map (function %cx-eq-flag) (ensure-vt t1) (ensure-vt t2)
                   :out out :dtype dtype)))
  (vt-fast-map #'%op-eq (ensure-vt t1) (ensure-vt t2)
               :dtype dtype :out out))

(defun vt-/= (t1 t2 &key (dtype :bool) out)
  "逐元素不等判定，返回 **:bool** 布尔张量。
      NaN 与任何值比较恒真 → True（含 NaN != NaN）。:out 契约同 vt-=。"
  (when (%cx-any-p t1 t2)
    (return-from vt-/=
      (vt-fast-map (function %cx-ne-flag) (ensure-vt t1) (ensure-vt t2)
                   :out out :dtype dtype)))
  (vt-fast-map #'%op-ne (ensure-vt t1) (ensure-vt t2)
               :dtype dtype :out out))

(defun vt-< (t1 t2 &key (dtype :bool) out)
  "逐元素小于判定，返回 **:bool** 布尔张量。
      NaN 参与的比较恒假 → False。:out 契约同 vt-=。"
  (when (%cx-any-p t1 t2)
    (return-from vt-<
      (vt-fast-map (function %cx-lt-flag) (ensure-vt t1) (ensure-vt t2)
                   :out out :dtype dtype)))
  (vt-fast-map #'%op-lt (ensure-vt t1) (ensure-vt t2)
               :dtype dtype :out out))

(defun vt-<= (t1 t2 &key (dtype :bool) out)
  "逐元素小于等于判定，返回 **:bool** 布尔张量。
      NaN 参与的比较恒假 → False。:out 契约同 vt-=。"
  (when (%cx-any-p t1 t2)
    (return-from vt-<=
      (vt-fast-map (function %cx-le-flag) (ensure-vt t1) (ensure-vt t2)
                   :out out :dtype dtype)))
  (vt-fast-map #'%op-le (ensure-vt t1) (ensure-vt t2)
               :dtype dtype :out out))

(defun vt-> (t1 t2 &key (dtype :bool) out)
  "逐元素大于判定，返回 **:bool** 布尔张量。
      NaN 参与的比较恒假 → False。:out 契约同 vt-=。"
  (when (%cx-any-p t1 t2)
    (return-from vt->
      (vt-fast-map (function %cx-gt-flag) (ensure-vt t1) (ensure-vt t2)
                   :out out :dtype dtype)))
  (vt-fast-map #'%op-gt (ensure-vt t1) (ensure-vt t2)
               :dtype dtype :out out))

(defun vt->= (t1 t2 &key (dtype :bool) out)
  "逐元素大于等于判定，返回 **:bool** 布尔张量。
      NaN 参与的比较恒假 → False。:out 契约同 vt-=。"
  (when (%cx-any-p t1 t2)
    (return-from vt->=
      (vt-fast-map (function %cx-ge-flag) (ensure-vt t1) (ensure-vt t2)
                   :out out :dtype dtype)))
  (vt-fast-map #'%op-ge (ensure-vt t1) (ensure-vt t2)
               :dtype dtype :out out))

(defun vt-rad2deg (vt &key out dtype)
  "弧度转角度（×180/π）。整数输入输出 float64，
      float32 进 float32 出。:out 契约同 vt-+。"
    (when (%cx-reject vt)
    (error "vt-rad2deg: 不支持复数输入（numpy 同样拒绝）。见 CONVENTIONS §16.11"))
(let* ((dt (%infer-float-dtype vt dtype))
         (factor (if (eq dt :float32) (/ 180.0s0 (coerce pi 'single-float))
                     (/ 180.0d0 pi))))
    (vt-map (lambda (x) (* x factor)) vt :out out :dtype dt)))

(defun vt-deg2rad (vt &key out dtype)
  "角度转弧度（×π/180）。整数输入输出 float64，
      float32 进 float32 出。:out 契约同 vt-+。"
    (when (%cx-reject vt)
    (error "vt-deg2rad: 不支持复数输入（numpy 同样拒绝）。见 CONVENTIONS §16.11"))
(let* ((dt (%infer-float-dtype vt dtype))
         (factor (if (eq dt :float32) (/ (coerce pi 'single-float) 180.0s0)
                     (/ pi 180.0d0))))
    (vt-map (lambda (x) (* x factor)) vt :out out :dtype dt)))

;;; ------------------------------------------------------------------
;;; 极值 / 逻辑算子（具名 + 内联）
;;;
;;; 类型声明（性能）：这些算子此前全部以 `(vt-map (lambda (a b) ...))` 形式
;;; 传入，`vt-map` 只能在运行时经 funcall 调用闭包，循环体拿不到任何类型
;;; 信息 —— aref 走泛型分派、结果装箱、写回再做一次泛型 cast。
;;; 实测 1M float64 的 vt-maximum 需 22ms（numpy 0.24ms，慢 93 倍）。
;;; 改成「具名函数 + inline 声明」后，`vt-fast-map` 能在编译期把算子
;;; 内联进 %inline{1,2,3}-loop 的特化循环，读写两端都是特化 aref。
;;; 语义与原来的 lambda 逐字相同（同样的 NaN 分支顺序）。
;;; ------------------------------------------------------------------
(declaim (inline %op-maximum %op-minimum %op-fmax %op-fmin
                 %op-logical-and %op-logical-or %op-logical-not
                 %op-logical-xor))

(defun %op-maximum (a b)
  "NaN 传播取大（对标 numpy.maximum）：任一为 NaN 即取该侧（左优先）。"
  (cond ((%nan-p a) a)
        ((%nan-p b) b)
        (t (max a b))))

(defun %op-minimum (a b)
  "NaN 传播取小（对标 numpy.minimum）。"
  (cond ((%nan-p a) a)
        ((%nan-p b) b)
        (t (min a b))))

(defun %op-fmax (a b)
  "忽略 NaN 取大（对标 numpy.fmax）：仅两侧皆 NaN 时返回 NaN。"
  (cond ((and (%nan-p a) (%nan-p b)) a)
        ((%nan-p a) b)
        ((%nan-p b) a)
        (t (max a b))))

(defun %op-fmin (a b)
  "忽略 NaN 取小（对标 numpy.fmin）。"
  (cond ((and (%nan-p a) (%nan-p b)) a)
        ((%nan-p a) b)
        ((%nan-p b) a)
        (t (min a b))))

(defun %op-logical-and (a b) (if (and (not (zerop a)) (not (zerop b))) 1 0))

(defun %op-logical-or (a b) (if (or (not (zerop a)) (not (zerop b))) 1 0))

(defun %op-logical-not (v) (if (zerop v) 1 0))

(defun %op-logical-xor (a b) (if (not (eq (not (zerop a)) (not (zerop b)))) 1 0))

(defun vt-maximum (t1 t2 &key out dtype)
  "逐元素取较大值，**NaN 传播**（任一为 NaN → NaN，且优先取左）
      对标 numpy.maximum。:out 契约同 vt-+。"
  (when (%cx-any-p t1 t2)
    (return-from vt-maximum
      (let* ((a (ensure-vt t1)) (b (ensure-vt t2))
             (p (%op-final-dtype (list (%scalar-kind t1) (%scalar-kind t2))
                                 (list (vt-dtype a) (vt-dtype b))))
             (dt (or dtype p)))
        (unless (%cx-complex-dtype-p* dt)
          (error "vt-maximum: 复数输入的结果 dtype 必须是复数类型，收到 ~a"
                 dt))
        (vt-fast-map (function %cop-max) a b :out out :dtype dt))))
  (vt-fast-map #'%op-maximum t1 t2 :out out :dtype dtype))

(defun vt-minimum (t1 t2 &key out dtype)
  "逐元素取较小值，**NaN 传播**（任一为 NaN → NaN，且优先取左）
      对标 numpy.minimum。:out 契约同 vt-+。"
  (when (%cx-any-p t1 t2)
    (return-from vt-minimum
      (let* ((a (ensure-vt t1)) (b (ensure-vt t2))
             (p (%op-final-dtype (list (%scalar-kind t1) (%scalar-kind t2))
                                 (list (vt-dtype a) (vt-dtype b))))
             (dt (or dtype p)))
        (unless (%cx-complex-dtype-p* dt)
          (error "vt-minimum: 复数输入的结果 dtype 必须是复数类型，收到 ~a"
                 dt))
        (vt-fast-map (function %cop-min) a b :out out :dtype dt))))
  (vt-fast-map #'%op-minimum t1 t2 :out out :dtype dtype))

(defun vt-fmax (t1 t2 &key out dtype)
  "逐元素取较大值，**忽略 NaN**（仅两侧皆 NaN 时返回 NaN）
      对标 numpy.fmax。:out 契约同 vt-+。"
  (when (%cx-any-p t1 t2)
    (return-from vt-fmax
      (let* ((a (ensure-vt t1)) (b (ensure-vt t2))
             (p (%op-final-dtype (list (%scalar-kind t1) (%scalar-kind t2))
                                 (list (vt-dtype a) (vt-dtype b))))
             (dt (or dtype p)))
        (unless (%cx-complex-dtype-p* dt)
          (error "vt-fmax: 复数输入的结果 dtype 必须是复数类型，收到 ~a"
                 dt))
        (vt-fast-map (function %cop-fmax) a b :out out :dtype dt))))
  (vt-fast-map #'%op-fmax t1 t2 :out out :dtype dtype))

(defun vt-fmin (t1 t2 &key out dtype)
  "逐元素取较小值，**忽略 NaN**（仅两侧皆 NaN 时返回 NaN）
      对标 numpy.fmin。:out 契约同 vt-+。"
  (when (%cx-any-p t1 t2)
    (return-from vt-fmin
      (let* ((a (ensure-vt t1)) (b (ensure-vt t2))
             (p (%op-final-dtype (list (%scalar-kind t1) (%scalar-kind t2))
                                 (list (vt-dtype a) (vt-dtype b))))
             (dt (or dtype p)))
        (unless (%cx-complex-dtype-p* dt)
          (error "vt-fmin: 复数输入的结果 dtype 必须是复数类型，收到 ~a"
                 dt))
        (vt-fast-map (function %cop-fmin) a b :out out :dtype dt))))
  (vt-fast-map #'%op-fmin t1 t2 :out out :dtype dtype))

(defun vt-logical-and (t1 t2 &key out (dtype :bool))
  "逐元素逻辑与（非零视为真），返回 **:bool** 布尔张量（对标 numpy.logical_and）。
   dtype 默认 :bool（CONVENTIONS §17）；显式 :dtype :int8 等仍可退化为 0/1 整数掩码。"
  (vt-fast-map #'%op-logical-and t1 t2 :dtype dtype :out out))

(defun vt-logical-or (t1 t2 &key out (dtype :bool))
  "逐元素逻辑或（非零视为真），返回 **:bool** 布尔张量。dtype 默认 :bool（§17）。"
  (vt-fast-map #'%op-logical-or t1 t2 :dtype dtype :out out))

(defun vt-logical-not (vt &key out (dtype :bool))
  "逐元素逻辑非，返回 **:bool** 布尔张量。dtype 默认 :bool（§17）。"
  (vt-fast-map #'%op-logical-not vt :dtype dtype :out out))

(defun vt-logical-xor (t1 t2 &key out (dtype :bool))
  "逐元素逻辑异或，返回 **:bool** 布尔张量。dtype 默认 :bool（§17）。"
  (vt-fast-map #'%op-logical-xor t1 t2 :dtype dtype :out out))

(defun vt-bit-and (t1 t2 &key out dtype)
  "逐元素按位与 logand。输入应为整数 dtype；:out 契约同 vt-+。"
    (when (%cx-reject t1 t2)
    (error "vt-bit-and: 不支持复数输入（numpy 同样拒绝）。见 CONVENTIONS §16.11"))
(vt-map #'logand (ensure-vt t1) (ensure-vt t2) :dtype dtype :out out))

(defun vt-bit-ior (t1 t2 &key out dtype)
  "逐元素按位或 logior。输入应为整数 dtype；:out 契约同 vt-+。"
    (when (%cx-reject t1 t2)
    (error "vt-bit-ior: 不支持复数输入（numpy 同样拒绝）。见 CONVENTIONS §16.11"))
(vt-map #'logior (ensure-vt t1) (ensure-vt t2) :dtype dtype :out out))

(defun vt-bit-xor (t1 t2 &key out dtype)
  "逐元素按位异或 logxor。输入应为整数 dtype；:out 契约同 vt-+。"
    (when (%cx-reject t1 t2)
    (error "vt-bit-xor: 不支持复数输入（numpy 同样拒绝）。见 CONVENTIONS §16.11"))
(vt-map #'logxor (ensure-vt t1) (ensure-vt t2) :dtype dtype :out out))

(defun vt-bit-not (vt &key out dtype)
  "逐元素按位取反 lognot。输入应为整数 dtype；:out 契约同 vt-+。"
    (when (%cx-reject vt)
    (error "vt-bit-not: 不支持复数输入（numpy 同样拒绝）。见 CONVENTIONS §16.11"))
(vt-map #'lognot vt :dtype dtype :out out))

(defun vt-left-shift (vt shift &key out dtype)
  "逐元素左移 SHIFT 位（ash x shift）。SHIFT 为整数标量。
      输入应为整数 dtype；:out 契约同 vt-+。"
    (when (%cx-reject vt shift)
    (error "vt-left-shift: 不支持复数输入（numpy 同样拒绝）。见 CONVENTIONS §16.11"))
(vt-map (lambda (x) (ash x shift))
          vt :dtype dtype :out out))

(defun vt-right-shift (vt shift &key out dtype)
  "逐元素右移 SHIFT 位（ash x -shift，算术右移）。SHIFT 为整数标量。
      输入应为整数 dtype；:out 契约同 vt-+。"
    (when (%cx-reject vt shift)
    (error "vt-right-shift: 不支持复数输入（numpy 同样拒绝）。见 CONVENTIONS §16.11"))
(vt-map (lambda (x) (ash x (- shift)))
          vt :dtype dtype :out out))

(declaim (inline %op-clip))
(defun %op-clip (x minv maxv)
  "三元 clip：先下限后上限，即 (min maxv (max minv x))。
   v0.4.0 修复：原实现先上限后下限，当 minv > maxv 时结果为 minv，
   而 numpy.clip(min>max) 的结果是 maxv（clamp 顺序以 max 收尾）。"
  (let ((x (if (< x minv) minv x)))
    (if (> x maxv) maxv x)))

(defun vt-clip (vt &optional (min-val nil min-p) (max-val nil max-p)
                   &key out dtype)
  "逐元素裁剪，完全对标 numpy.clip 的签名与边界语义。

  签名对齐 numpy 2.3.5 的 `clip(a, a_min=None, a_max=None)`：
    - `(vt-clip a)`              两个边界均省略 → 返回 a 的副本（不裁剪）；
    - `(vt-clip a nil max)`      仅给上限（MIN-VAL 传 nil）→ min(x, max)；
    - `(vt-clip a min nil)`      仅给下限（MAX-VAL 传 nil）→ max(x, min)；
    - `(vt-clip a min max)`      双侧裁剪 → min(max, max(min, x))；
    - `(vt-clip a min)`          仅给 MIN-VAL 而没有 MAX-VAL → **报错**
      （对齐 numpy 的 `TypeError: clip() missing 1 required
      positional argument: 'a_max'`；numpy 不允许只给下限）。

  MIN-VAL / MAX-VAL 是标量，不是张量。:dtype / :out 契契约同 vt-map
  （结果 dtype 由输入提升或显式 :dtype 决定，out 必须精确匹配）。

  复数（§16.8）：按**字典序**整元素夹取，不按分量 —— numpy 的 clip 定义为
  `minimum(maximum(a, lo), hi)`，而 maximum/minimum 对复数是字典序。
  实测 v=[1+2j, 3-4j, 5+0j], clip(0, 2) → [1+2j, 2+0j, 2+0j]：第一项虚部 2
  超出 2 却没被裁，因为比较的是 (1,2) 整体而不是分量 2。

  复数分支在 complex-ops.lisp 的 %CX-OP-CLIP；本函数只做分流（§16.32）。
  若改成事后重定义，重载本文件后复数会被逐分量裁剪（静默错答案）。"
  (when (%cx-any-p vt)
    (return-from vt-clip
      (%cx-op-clip (ensure-vt vt) min-val max-val min-p max-p out dtype)))
  (when (and min-p (not max-p))
    (error "vt-clip: 只提供 MIN-VAL 而未提供 MAX-VAL（numpy 语义：clip() \
missing a_max；若只要下限请显式传 MAX-VAL 为 nil）"))
  (cond
    ;; 不裁剪（两边界均省略）：直接返回副本，保持 dtype/形状不变
    ((and (null min-val) (null max-val))
     (let ((r (if out
                  (vt-check-out out (vt-shape vt) (or dtype (vt-dtype vt))
                                :op-name "vt-clip")
                  (vt-zeros (vt-shape vt) :dtype (or dtype (vt-dtype vt))))))
       (vt-copy-into r vt)
       r))
    ;; 仅上限：min(x, max)
    ((null min-val)
     (vt-map (lambda (x) (min x max-val)) vt :dtype dtype :out out))
    ;; 仅下限：max(x, min)
    ((null max-val)
     (vt-map (lambda (x) (max x min-val)) vt :dtype dtype :out out))
    ;; 双侧：min(max, max(min, x))，与 %op-clip 同义
    (t (vt-fast-map #'%op-clip vt min-val max-val :dtype dtype :out out))))

(defun vt-lerp (start end weight &key out dtype)
  "逐元素线性插值 start + (end - start) * weight，
      对标 numpy 的 a + (b-a)*t 形式（三者均支持广播）。
      :dtype 决定结果 dtype（缺省按提升规则），:out 契约同 vt-+。"
  (vt-map (lambda (s e w)
            (+ s (* (- e s) w)))
          (ensure-vt start) (ensure-vt end) (ensure-vt weight)
          :dtype dtype :out out))

(defun vt-cbrt (vt &key out dtype)
  "逐元素立方根。定义为 signum(x)*|x|^(1/3)，因此负数返回实数值
      （对标 numpy.cbrt，不返回 NaN）。整数输入输出 float64。
      v0.4.x 修复：cbrt(NaN) 原经 (expt NaN 1/3) 泄漏复数 NaN 导致
      类型错误，现显式拦截：NaN → NaN、±Inf → ±Inf（numpy 对齐）。
  性能：改用符号算子 %OP-CBRT-Df/SF，编译期内联（原 lambda 走 vt-map
  的 funcall 路径，1M 元素实测 143ms）。"
    (when (%cx-reject vt)
    (error "vt-cbrt: 不支持复数输入（numpy 同样拒绝）。见 CONVENTIONS §16.11"))
(let ((dt (%infer-float-dtype vt dtype)))
    (%float-map-pair %op-cbrt-df %op-cbrt-sf vt out dt)))

;;; v0.4.3 性能修复：vt-hypot / vt-lerp / vt-cbrt 等原本一律写成
;;;     (vt-map (lambda (a b) ...) ...)
;;; 闭包让 vt-fast-map 的算子内联失效（该宏只认 #'symbol 形式），循环里
;;; 既无数组元素类型声明、又是逐元素 funcall —— hypot 1M 元素实测 100ms
;;; （numpy 5.9ms，差 17 倍）。改成命名算子 + 按 dtype 分派即可内联。
(declaim (inline %hypot-df %hypot-sf))
(defun %hypot-df (a b)
  "hypot 的 double-float 内核。"
  (let ((abs-a (abs a)) (abs-b (abs b)))
    (cond ((or (%inf-p abs-a) (%inf-p abs-b))
           ;; NumPy hypot：任一参数 ±Inf → +Inf（即便另一参数为 NaN）
           (vt-get-pos-inf :float64))
          ((%nan-p abs-a) abs-a)
          ((%nan-p abs-b) abs-b)
          ((zerop abs-a) abs-b)
          ((zerop abs-b) abs-a)
          (t (let* ((mx (max abs-a abs-b))
                    (mn (min abs-a abs-b))
                    (r  (/ mn mx)))
               (* mx (sqrt (+ 1.0d0 (* r r)))))))))

(defun %hypot-sf (a b)
  "hypot 的 single-float 内核（常量与结果都保持 single，避免写入
   specialized-array 时的类型错误）。"
  (let ((abs-a (abs a)) (abs-b (abs b)))
    (cond ((or (%inf-p abs-a) (%inf-p abs-b))
           (vt-get-pos-inf :float32))
          ((%nan-p abs-a) abs-a)
          ((%nan-p abs-b) abs-b)
          ((zerop abs-a) abs-b)
          ((zerop abs-b) abs-a)
          (t (let* ((mx (max abs-a abs-b))
                    (mn (min abs-a abs-b))
                    (r  (/ mn mx)))
               (* mx (sqrt (+ 1.0s0 (* r r)))))))))

(defun vt-hypot (t1 t2 &key out dtype)
  "逐元素直角三角斜边 sqrt(x² + y²)（对标 numpy.hypot）。任一输入为 ±Inf → 结果 +Inf；NaN 传播。"
    (when (%cx-reject t1 t2)
    (error "vt-hypot: 不支持复数输入（numpy 同样拒绝）。见 CONVENTIONS §16.11"))
(let* ((v1 (ensure-vt t1))
         (v2 (ensure-vt t2))
         (dt (or dtype (if (or (eq (vt-dtype v1) :float32)
                               (eq (vt-dtype v2) :float32))
                           :float32 :float64))))
    (if (eq dt :float32)
        (vt-fast-map #'%hypot-sf v1 v2 :out out :dtype dt)
        (vt-fast-map #'%hypot-df v1 v2 :out out :dtype dt))))

(defun vt-reciprocal (vt &key out dtype)
  "逐元素倒数 (1/x)。

   dtype 语义（对标 numpy.reciprocal，v0.3.6 起）：
     整数输入 → **保持整数 dtype**（numpy 1//x 整数除法，1/2 → 0）；
     浮点输入 → 保持原浮点 dtype（float32 进 float32 出）；
     显式 :dtype 优先。
   零除：浮点 → ±Inf（IEEE 754）；整数 → 确定性报错，与 numpy
     整数零除行为一致。"
  (when (%cx-any-p vt)
    (return-from vt-reciprocal
      (let* ((in (ensure-vt vt))
             (dt (%cx-result-complex-dtype (vt-dtype in) dtype "VT-RECIPROCAL")))
        (vt-fast-map (function %cop-recip) in :out out :dtype dt))))
  (let* ((in-dt (vt-dtype (ensure-vt vt)))
         (dt (or dtype in-dt)))
    (if (vt-int-dtype-p dt)
        ;; 整数倒数：保持整数语义（1//x），零除交给 CL 整数除零错误
        (vt-map (lambda (v) (if (zerop v) (error "vt-reciprocal: 整数零除") (truncate (/ 1 v))))
                vt :out out :dtype dt)
        (let ((one (if (eq dt :float32) 1.0s0 1.0d0)))
          (vt-map (lambda (v) (/ one v)) vt :out out :dtype dt)))))

(defun vt-negative (vt &key out dtype)
  "逐元素取负（-x）。这是一元 vt-- 的具名入口，
      NaN/±Inf 原样传播（对标 numpy.negative）。:out 契约同 vt-+。"
  (when (%cx-any-p vt)
    (return-from vt-negative
      (let* ((in (ensure-vt vt))
             (dt (%cx-result-complex-dtype (vt-dtype in) dtype "VT-NEGATIVE")))
        (vt-fast-map (function %cop-neg) in :out out :dtype dt))))
  (vt-fast-map #'- vt :dtype dtype :out out))

(defun vt-sinc (tensor &key out dtype)
  "逐元素归一化 sinc 函数 sin(πx)/(πx)，且 sinc(0)=1（对标 numpy.sinc）。
      整数输入输出 float64，float32 进 float32 出。:out 契约同 vt-+。"
  (when (%cx-any-p tensor)
    (return-from vt-sinc
      (let* ((in (ensure-vt tensor))
             (dt (%cx-result-complex-dtype (vt-dtype in) dtype "VT-SINC")))
        (vt-fast-map (function %cop-sinc) in :out out :dtype dt))))
  (let* ((dt (%infer-float-dtype tensor dtype))
         (x-pi (vt-scale tensor pi :dtype dt)))
    (vt-map (lambda (x)
              (if (zerop x) 1.0d0 (/ (sin x) x)))
            x-pi :out out :dtype dt)))

;; 在编译期间垃圾回收内存，防止爆炸
#+sbcl (sb-ext::gc :full t)
