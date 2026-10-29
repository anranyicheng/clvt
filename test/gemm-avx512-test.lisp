;;;; gemm-avx512-test.lisp — AVX-512 float64 GEMM 机器码内核的回归网
;;;;
;;;; 为什么单独一套
;;;; ==============
;;;; src/gemm-avx512.lisp 是**运行时发射机器码**的：EVEX 前缀、ModRM/SIB、
;;;; disp8 的元组单位、掩码寄存器，任何一处算错都不会被编译器和类型检查
;;;; 拦住 —— 轻则结果算错，重则越界写坏堆。所以这套测试的断言重点不是
;;;; 「数值近似相等」，而是：
;;;;   1. 与 sb-simd 路径**逐位相同**（同一舍入链，见 CONVENTIONS §7.3）；
;;;;   2. 每一种尾块形态（n % 16、m % 8、k % KC）都踩到；
;;;;   3. 内核只写它该写的那些元素（缓冲区前后放哨兵，用非对齐地址）；
;;;;   4. 关掉开关能立刻退回 sb-simd 路径（选路可控，见 §7.4）。
;;;;
;;;; 逐位比较用 SAP 读 64 位整数，而不是 EQL：EQL 对 ±0.0 的判定是实现
;;;; 相关的，而 -0.0 与 +0.0 的差别正是「舍入链走岔了」的第一现场。

(require :asdf)
#+quicklisp (ql:quickload :clvt)
(asdf:load-system :clvt)
(in-package :clvt)

;;; ============================================================
;;; 小工具
;;; ============================================================
(defvar *pass* 0)
(defvar *fail* 0)
(defvar *failures* nil)

(defun ok (label)
  (incf *pass*)
  (format t "  [ OK ] ~a~%" label)
  t)

(defun ng (label fmt &rest args)
  (incf *fail*)
  (let ((msg (apply #'format nil fmt args)))
    (push (format nil "~a — ~a" label msg) *failures*)
    (format t "  [FAIL] ~a~%         ~a~%" label msg)
    nil))

(defun check (label test &optional (fmt "断言为假") &rest args)
  (if test (ok label) (apply #'ng label fmt args)))

(defun %bit-diff (d1 d2)
  "逐位比较两个 double-float 数组。返回 (values 不同元素数 首个不同下标)。
   只按 min 长度比 —— 长度不等由调用方先断言。"
  (let ((n (min (length d1) (length d2))) (cnt 0) (first nil))
    (sb-sys:with-pinned-objects (d1 d2)
      (let ((p1 (sb-sys:vector-sap d1))
            (p2 (sb-sys:vector-sap d2)))
        (dotimes (i n)
          (unless (= (sb-sys:sap-ref-64 p1 (* i 8))
                     (sb-sys:sap-ref-64 p2 (* i 8)))
            (incf cnt)
            (unless first (setf first i))))))
    (values cnt first)))

(defun %nan? (x)
  "不触发 FP 陷阱的 NaN 判定：只读位型（指数全 1、尾数非 0）。
   库里的 %NAN-P 是 `(not (= x x))`，测试运行时 FP 陷阱开着会直接报错。"
  (let ((b (sb-kernel:double-float-bits x)))
    (and (= (ldb (byte 11 52) b) #x7ff)
         (/= (ldb (byte 52 0) b) 0))))

(defun %same-value-p (x y)
  "位相同，或两者都是同号 NaN（NaN 的 payload 不在契约内）。"
  (or (= (sb-kernel:double-float-bits x) (sb-kernel:double-float-bits y))
      (and (%nan? x) (%nan? y)
           (eql (logbitp 63 (sb-kernel:double-float-bits x))
                (logbitp 63 (sb-kernel:double-float-bits y))))))

(defun %tail-start (n)
  "「列尾标量区」的起始列号。

     sb-simd 的 f64 驱动（`BUILD-DRIVER`）把列分成三段：8 列微块、按 4 列
     对齐的整宽向量、以及最后 <4 列走**标量**端口 —— 而标量是
     `(incf c (* a b))`，两次舍入；AVX-512 内核对这些列用 FMA，一次舍入。
     所以 n % 4 != 0 时最后几列本来就会差 1~2 ulp（512 侧更精确）。

     这不是缺陷而是**已知且可解释**的差异，测试因此把「逐位相同」放宽成
     「差异只能出现在这些尾部列，且相对误差 ≤ 1e-15」：
       · n % 4 == 0 → 尾部集合为空 → 自动退化为严格逐位相同；
       · 差异跑到别的列 → 立刻报错（那才是真 bug）。"
  (- n (mod n 4)))

(defun matmul-data (a b)
  "单线程跑一次 vt-matmul，返回 (values C 的 data)。"
  (let ((*simd-matmul-thread-count* 1))
    (let ((c (vt-matmul a b)))
      (values (vt-data c) (vt-shape c)))))

(defun gemm-both (a b)
  "分别用 AVX-512 内核与 sb-simd 内核算一次，返回两个 data。"
  (let* ((*simd-matmul-thread-count* 1)
         (*avx512-gemm-enabled* t)
         (d512 (let ((c (vt-matmul a b))) (vt-data c)))
         (*avx512-gemm-enabled* nil)
         (dsimd (let ((c (vt-matmul a b))) (vt-data c))))
    (values d512 dsimd)))

(defun check-bits (label a b &key (shape nil))
  "断言 A·B 在两条内核下「逐位相同，或只在尾部标量列差 ≤ 1e-15」，并核对形状。"
  (multiple-value-bind (d512 dsimd) (gemm-both a b)
    (cond
      ((not (= (length d512) (length dsimd)))
       (ng label "长度不等: 512=~a sb-simd=~a" (length d512) (length dsimd)))
      (t
       (let* ((n (second (vt-shape b)))
              (tstart (%tail-start n))
              (cnt 0) (outside 0) (maxrel 0.0d0) (first nil))
         (dotimes (idx (length d512))
           (let ((x (aref d512 idx)) (y (aref dsimd idx)))
             (unless (%same-value-p x y)
               (incf cnt)
               (let* ((j (mod idx n))
                      (rel (/ (abs (- x y))
                              (max (abs x) (abs y) 1.0d-300))))
                 (unless (>= j tstart) (incf outside))
                 (when (> rel maxrel) (setf maxrel rel))
                 (unless first (setf first (list (floor idx n) j x y)))))))
         (cond
           ((zerop cnt)
            (ok (format nil "~a（~a 元素逐位相同）" label (length d512))))
           ((and (zerop outside) (<= maxrel 1.0d-15))
            (ok (format nil "~a（~a/~a 元素差 ≤ ~,1e，全在尾部 ~a 列 j≥~a）"
                        label cnt (length d512) maxrel (- n tstart) tstart)))
           (t
            (ng label "~a 个元素不同（~a 个在尾部列之外），最大相对差 ~e，首个 ~s"
                cnt outside maxrel first))))
       ;; 形状核对（若给了 :shape）
       (when shape
         (let ((got (nth-value 1 (matmul-data a b))))
           (unless (equal got shape)
             (ng (concatenate 'string label " [形状]")
                 "期望 ~a 实际 ~a" shape got))))))))

;;; ============================================================
;;; T1 — 可用性与选路开关
;;; ============================================================
(defun t1-availability ()
  (format t "~&[1/6] 可用性与选路开关~%")
  (let* ((*avx512-gemm-enabled* t)
         (cpu (and (%cpu-avx512f-p) t))
         (kern (nth-value 0 (avx512-gemm-kernels)))
         (kernm (nth-value 1 (avx512-gemm-kernels))))
    (when cpu
      (check "T1a CPU/OS 报 AVX-512F 时两个内核都已发射"
             (and kern kernm)))
    (unless cpu
      (format t "  [SKIP] T1 本机 /proc/cpuinfo 无 avx512f，后续逐位对照退化为自比~%"))
    ;; 开关必须**立刻**生效：置 NIL 后分派函数回到 sb-simd 版本
    (let ((*avx512-gemm-enabled* nil))
      (check "T1b enabled=NIL ⇒ 分派到 %SIMD-MATMUL-F64"
             (eq (%f64-gemm-kernel) #'%simd-matmul-f64)))
    (let ((*avx512-gemm-enabled* t))
      (check "T1c enabled=T ⇒ 分派到 %SIMD-MATMUL-F64-512（本机支持时）"
             (if (and cpu kern)
                 (eq (%f64-gemm-kernel) #'%simd-matmul-f64-512)
                 (eq (%f64-gemm-kernel) #'%simd-matmul-f64))))
    ;; 内核字节长度：两个版本都不该是空的，且掩码版更长（多了 {k}{z} 前缀）
    (when (and cpu kern)
      (let ((b0 (%avx512-gemm-kernel-bytes nil))
            (b1 (%avx512-gemm-kernel-bytes t)))
        (check "T1d 内核字节长度合理（无掩码 < 掩码，且都 ≥ 200 字节）"
               (and (> (length b0) 200) (> (length b1) (length b0)))
               "无掩码=~a 掩码=~a" (length b0) (length b1))
        ;; 两次发射必须完全一致（纯函数，不留状态）
        (check "T1e 重复发射字节一致（发射器无状态）"
               (equalp b0 (%avx512-gemm-kernel-bytes nil)))))))

;;; ============================================================
;;; T2 — 与 sb-simd 路径逐位一致：尾块矩阵
;;; ============================================================
;;; 分块常量（MC=64 / KC=256 / NC=256）决定了尾块形态：
;;;   · n 方向：nf = n/16 个整块 + 一个 n%16 的掩码块；
;;;   · m 方向：每 8 行为一个微块，m%8 的余数交回 sb-simd；
;;;   · k 方向：k%256 的余数走最后一轮 kc 循环。
;;; 下面每个形状至少踩中其中一条。
(defun t2-tail-shapes ()
  (format t "~&[2/6] 尾块形态逐位对照~%")
  (dolist (spec '(;; (m k n) 说明
                  (64 256 5)    ; n<8：掩码只用低 5 位
                  (64 256 7)    ; n<8 的边界（7 = 低 8 位全用满前一步）
                  (64 256 8)    ; n=8：nf=0 但 rm=8（低半掩码满）
                  (64 256 15)   ; n=15：两半掩码都用、都不满
                  (64 256 16)   ; n=16：恰好一个整块，无掩码
                  (64 256 17)   ; n=17：1 整块 + 1 列
                  (64 256 31)   ; n=31：1 整块 + 15 列
                  (64 300 300)  ; k=300 非 KC 倍数 + n 跨 NC 块（256+44）
                  (9 300 300)   ; m=9：1 整块 + 1 行尾（交回 sb-simd）
                  (13 200 260)  ; m/n/k 全非倍数
                  (100 130 100); m=100：12 整块 + 4 行尾
                  (1 4096 1)    ; m=n=1：整块退化为 1 行 1 列
                  (8 1 8)       ; k=1：单次 kc 循环
                  (200 200 200) ; 常规方阵
                  (300 300 300) ; 跨 MC/KC/NC 三个方向
                  (7 60000 3))) ; k 极大、m/n 极小（kc 循环 235 轮）
    (destructuring-bind (m k n) spec
      (let ((a (vt-random-uniform (list m k)))
            (b (vt-random-uniform (list k n))))
        (check-bits (format nil "T2 m=~a k=~a n=~a" m k n) a b
                    :shape (list m n))))))

;;; ============================================================
;;; T3 — 视图与地址对齐
;;; ============================================================
;;; 内核用的是 vmovupd（不要求 64 字节对齐），而 SBCL 的数组数据只保证
;;; 16 字节对齐（实测 1000×1000 float64 的 SAP 低 6 位 = 16）。这里故意
;;; 用「偏移 1/3/5 个元素的连续切片」把基址推到各种非对齐位置。
(defun t3-layouts ()
  (format t "~&[3/6] 视图/布局/对齐~%")
  ;; T3a 转置输入（vt-matmul 内部要先连续化，走的是 AT·B 的同一内核）
  (let ((a (vt-random-uniform '(300 200)))
        (b (vt-random-uniform '(300 100))))
    (check-bits "T3a A 转置输入" (vt-transpose a) b :shape '(200 100)))
  (let ((a (vt-random-uniform '(200 300)))
        (b (vt-random-uniform '(100 300))))
    (check-bits "T3b B 转置输入" a (vt-transpose b) :shape '(200 100)))
  ;; T3c 偏移切片：数据指针不再是分配起点
  (dolist (off '(1 3 5))
    (let* ((big (vt-random-uniform (list (+ off 200) 200)))
           (a (vt-slice big (list off (+ off 200)) '(:all)))
           (b (vt-random-uniform '(200 200))))
      (check-bits (format nil "T3c A = big[~a:~a, :]（偏移 ~a 元素）"
                          off (+ off 200) off)
                  a b :shape '(200 200))))
  ;; T3d 跨步切片：不是连续内存，vt-matmul 必须先收集
  (let* ((big (vt-random-uniform '(600 400)))
         (a (vt-slice big '(nil nil 3) '(:all)))
         (b (vt-random-uniform '(400 128))))
    (check-bits "T3d A = big[::3, :]（跨步视图）" a b :shape '(200 128)))
  ;; T3e 批量 3D：走 %SIMD-BATCHED-MATMUL-F64，同样用 512 内核
  (let ((a (vt-random-uniform '(4 64 128)))
        (b (vt-random-uniform '(4 128 40))))
    (let* ((*simd-matmul-thread-count* 1)
           (*avx512-gemm-enabled* t)
           (c512 (vt-data (vt-matmul a b)))
           (*avx512-gemm-enabled* nil)
           (csimd (vt-data (vt-matmul a b))))
      (multiple-value-bind (cnt first) (%bit-diff c512 csimd)
        (if (zerop cnt)
            (ok (format nil "T3e 3D 批量（4×64×128 @ 4×128×40）逐位相同（~a 元素）"
                        (length c512)))
            (ng "T3e 3D 批量" "~a 个元素不同，首个 ~a" cnt first))))))

;;; ============================================================
;;; T4 — 写边界：缓冲区前后放哨兵，且 C 指针非对齐
;;; ============================================================
;;; 直接调 %SIMD-MATMUL-F64-512，把 C 放在大缓冲的中间（偏移 33 个元素 =
;;; 264 字节，模 64 = 8，故意不对齐）。算完检查：
;;;   · C 区域之外一个元素都没被改；
;;;   · C 区域之内的值与 sb-simd 路径逐位相同。
(defun %sap-kernel-test (m k n i0 i1 j0 j1 label)
  (let* ((pad-before 33)
         (pad-after 8)
         (total (+ pad-before (* m n) pad-after))
         (a (vt-random-uniform (list m k)))
         (b (vt-random-uniform (list k n)))
         (ref (make-array (* m n) :element-type 'double-float
                                  :initial-element 0.0d0))
         (buf (make-array total :element-type 'double-float
                                 :initial-element 999.0d0))
         (ad (vt-data a))
         (bd (vt-data b)))
    ;; C 区先清 0：内核语义是 C += A·B（与 einsum 的 incf 一致）
    (dotimes (k2 (* m n)) (setf (aref buf (+ pad-before k2)) 0.0d0))
    ;; 参考值：sb-simd 内核算到 ref（长度恰好 m*n，越界即崩，顺带当第二道哨兵）
    (sb-sys:with-pinned-objects (ad bd ref)
      (%simd-matmul-f64 (sb-sys:vector-sap ad) (sb-sys:vector-sap bd)
                        (sb-sys:vector-sap ref) k n 0 m 0 n))
    ;; 被测：512 内核写进 buf 的中间区域
    (sb-sys:with-pinned-objects (ad bd buf)
      (%simd-matmul-f64-512 (sb-sys:vector-sap ad)
                            (sb-sys:vector-sap bd)
                            (sb-sys:sap+ (sb-sys:vector-sap buf) (* pad-before 8))
                            k n i0 i1 j0 j1))
    ;; 1. 哨兵
    (let ((dirty nil))
      (dotimes (i pad-before)
        (unless (= (aref buf i) 999.0d0) (push (list :before i) dirty)))
      (loop for i from (+ pad-before (* m n)) below total
            unless (= (aref buf i) 999.0d0) do (push (list :after i) dirty))
      (if dirty
          (ng (concatenate 'string label " [哨兵]")
              "越界写了 ~a 个元素（前 ~a / 后 ~a）"
              (length dirty)
              (count :before dirty :key #'car)
              (count :after dirty :key #'car))
          (ok (concatenate 'string label " [哨兵] 缓冲区前后未被触碰"))))
    ;; 2. 与 sb-simd 逐位一致（尾部标量列允许 ≤1e-15，见 %TAIL-START 的说明）
    (let ((cnt 0) (outside 0) (maxrel 0.0d0) (first nil)
          (tstart (%tail-start n)))
      (loop for i from i0 below i1 do
        (loop for j from j0 below j1 do
          (let* ((idx (+ pad-before (* i n) j))
                 (ref-idx (+ (* i n) j))
                 (x (aref buf idx))
                 (y (aref ref ref-idx)))
            (unless (%same-value-p x y)
              (incf cnt)
              (unless (>= j tstart) (incf outside))
              (let ((rel (/ (abs (- x y)) (max (abs x) (abs y) 1.0d-300))))
                (when (> rel maxrel) (setf maxrel rel)))
              (unless first (setf first (list i j x y)))))))
      (cond
        ((zerop cnt)
         (ok (format nil "~a [数值] 子块 ~a×~a 逐位相同"
                     label (- i1 i0) (- j1 j0))))
        ((and (zerop outside) (<= maxrel 1.0d-15))
         (ok (format nil "~a [数值] 子块 ~a×~a：~a 个元素差 ≤ ~,1e（尾部标量列）"
                     label (- i1 i0) (- j1 j0) cnt maxrel)))
        (t
         (ng (concatenate 'string label " [数值]")
             "~a 个元素不同（~a 个在尾部列外），最大相对差 ~e，首个 ~s"
             cnt outside maxrel first))))
    ;; 3. 子块之外必须原样（C 区填的是 0.0d0，哨兵区是 999）
    (let ((untouched t) (bad nil))
      (loop for i from 0 below m do
        (loop for j from 0 below n do
          (unless (and (>= i i0) (< i i1) (>= j j0) (< j j1))
            (unless (= (aref buf (+ pad-before (* i n) j)) 0.0d0)
              (setf untouched nil) (push (list i j) bad)))))
      (check (concatenate 'string label " [子块外] 区域外元素未被写")
             untouched "被改写的元素: ~a" (subseq bad 0 (min 5 (length bad)))))))

(defun t4-write-bounds ()
  (format t "~&[4/6] 写边界（哨兵 + 非对齐 C 指针）~%")
  (when (%cpu-avx512f-p)
    ;; 整块：m=8 行、n=16 列（无掩码内核 + 满 nf）
    (%sap-kernel-test 8 32 16 0 8 0 16 "T4a 8×16 整块")
    ;; 掩码块：n=5（只有低位掩码）
    (%sap-kernel-test 8 32 5 0 8 0 5 "T4b 8×5 低位掩码")
    ;; 掩码块：n=13（低位满 8 + 高位 5）
    (%sap-kernel-test 8 32 13 0 8 0 13 "T4c 8×13 双掩码")
    ;; 子范围调用：只算右下角 8×16，左上角必须保持哨兵
    (%sap-kernel-test 16 40 32 8 16 16 32 "T4d 16×32 只算子块 [8:16, 16:32]")
    ;; 行尾不足 8 行也交回 sb-simd（iend=0 时会整块交回）
    (%sap-kernel-test 5 40 16 0 5 0 16 "T4e 5 行（全部交回 sb-simd）")))

;;; ============================================================
;;; T5 — 数值极端量级（不改变舍入链，只是确认没有溢出/下溢以外的意外）
;;; ============================================================
(defun t5-magnitudes ()
  (format t "~&[5/6] 量级与特殊值~%")
  ;; 大值：|a| ≤ 1e150、|b| ≤ 1e150、k=64 ⇒ C 到 1e301 量级，仍在 double 范围内
  (let ((a (vt-random-uniform '(64 64) :low -1.0d150 :high 1.0d150))
        (b (vt-random-uniform '(64 64) :low -1.0d150 :high 1.0d150)))
    (check-bits "T5a 量级 1e150 × 1e150（结果 ~1e301）" a b))
  ;; 小值：1e-160 × 1e-160 × 64 ⇒ 1e-319，进入次正规数区间 —— 逐位仍须相同
  (let ((a (vt-random-uniform '(64 64) :low -1.0d-160 :high 1.0d-160))
        (b (vt-random-uniform '(64 64) :low -1.0d-160 :high 1.0d-160)))
    (check-bits "T5b 量级 1e-160 × 1e-160（次正规数）" a b))
  ;; 符号零：a 全零 ⇒ C 全是 ±0.0，舍入链仍有符号差异，必须逐位相同
  (let ((a (vt-zeros '(64 32) :dtype :float64))
        (b (vt-random-uniform '(32 32) :low -1.0d0 :high 1.0d0)))
    (check-bits "T5c 全零 A（结果为 ±0.0）" a b))
  ;; NaN 传播：A[0,0] 换成 NaN ⇒ 第 0 行全 NaN；两条路径的 NaN 位型也要一致。
  ;; 不能用 (/ 0.0d0 0.0d0) 造 NaN —— 库的测试运行时 FP 陷阱是开着的，
  ;; 直接报 FLOATING-POINT-INVALID-OPERATION。直接构造位型最省事。
  (let* ((a (vt-copy (vt-random-uniform '(64 32))))
         (b (vt-random-uniform '(32 32) :low -1.0d0 :high 1.0d0)))
    (setf (aref (vt-data a) 0) (sb-kernel:make-double-float #x7ff80000 1))
    (check-bits "T5d A 含 NaN（NaN 位置与符号一致）" a b)
    ;; 再确认第 0 行确实全 NaN（不是「两条路径一起算错」）
    (multiple-value-bind (d512 dsimd) (gemm-both a b)
      (declare (ignore dsimd))
      (let ((all-nan t))
        (dotimes (j 32)
          (unless (%nan? (aref d512 j)) (setf all-nan nil)))
        (check "T5e NaN 沿整行传播（第 0 行 32 个元素全 NaN）" all-nan)))))

;;; ============================================================
;;; T6 — 与 einsum 通道同源（两条入口必须落到同一内核）
;;; ============================================================
(defun t6-same-kernel ()
  (format t "~&[6/6] vt-matmul ≡ vt-einsum（同内核 ⇒ 逐位相同）~%")
  (let* ((a (vt-random-uniform '(150 90)))
         (b (vt-random-uniform '(90 70)))
         (*simd-matmul-thread-count* 1)
         (m1 (vt-data (vt-matmul a b)))
         (m2 (vt-data (vt-einsum "ij,jk->ik" a b))))
    (multiple-value-bind (cnt first) (%bit-diff m1 m2)
      (if (zerop cnt)
          (ok "T6 vt-matmul ≡ vt-einsum（150×90 @ 90×70）")
          (ng "T6 vt-matmul vs vt-einsum" "~a 个元素不同，首个 ~a" cnt first)))))

;;; ============================================================
;;; 主入口
;;; ============================================================
(defun run ()
  (format t "~%===== AVX-512 float64 GEMM 内核测试 =====~%")
  (t1-availability)
  (t2-tail-shapes)
  (t3-layouts)
  (t4-write-bounds)
  (t5-magnitudes)
  (t6-same-kernel)
  (format t "~&===== 汇总: ~a 通过 / ~a 失败 =====~%" *pass* *fail*)
  (format t "Total: ~a | Pass: ~a | Fail: ~a~%" (+ *pass* *fail*) *pass* *fail*)
  (when *failures*
    (format t "失败清单:~%")
    (dolist (f (nreverse *failures*)) (format t "  - ~a~%" f)))
  (finish-output)
  (zerop *fail*))

(let ((good (handler-case (run)
              (error (e)
                (format t "~&❌ 测试主体异常: ~a~%" e)
                nil))))
  (sb-ext:exit :code (if good 0 1)))
