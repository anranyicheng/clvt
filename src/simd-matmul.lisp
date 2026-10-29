;;;; simd-matmul.lisp — sb-simd 快速路径（f64 / f32 / i32 / i64）
(in-package :clvt)

;;; ============================================================
;;; 配置
;;; ============================================================

(defvar *simd-matmul-enabled* t)
(defvar *simd-matmul-threshold* 50000)
(defvar *simd-matmul-thread-threshold* 5000000)
;; GEMM 的线程数**不是**越多越好：分块内核本身已把单线程推到算力上限，
;; 再加线程只是抢同一个内存带宽。实测 1000³ float64 在 8~16 线程都是
;; 11~16ms，而 31 线程（= 核数-1）出现过 87ms 的崩溃式退化 —— 30 个线程
;; 各自的工作集互相踢出 L2，且线程创建/汇合的开销无法被 8ms 的活摊薄。
;; 因此默认上限取 16：既能吃满多核，又不会把 L2 打穿。
(defvar *simd-matmul-thread-count* (max 1 (min 16 (- *processor-number* 1)))
  "GEMM 并行线程数上限（默认 min(16, 核数-1)）。")

;;; ============================================================
;;; 分块 GEMM 内核
;;; ============================================================
;;; 旧的「1 行 × VW 列」gaxpy 内核有两个致命访存问题：
;;;   1. 每个 k 步都要把 C 的一整行读一遍再写一遍（C 流量 = 2·m·k·n·8 字节）；
;;;   2. B 的每一行被 m 次重读（m=1000 时 B 要从内存过 8 GB）。
;;; 实测 1000³ float64 单线程 235ms ≈ 纯内存带宽上限，与算力无关。
;;;
;;; 这里换成标准的三层分块：
;;;   · 寄存器微块 MR×NR：C 的一个 MR×(NR·VW) 块在整段 k 里驻留寄存器，
;;;     C 只在每个 kc 块边界读写一次；B 的每个向量被 MR 行复用；
;;;   · 缓存分块 MC/KC/NC：A/B/C 三个瓦片一起塞进 L2（默认 64/256/256 →
;;;     A 128KB + B 512KB + C 128KB），DRAM 流量降到 O(m·k·n/MC)；
;;;   · 内层用指针步进（每个 k 步只有 MR 次 A 指针自增 + 1 次 B 指针自增），
;;;     没有乘法、没有地址重算。
;;;
;;; 数值语义（铁律 1：连续性只影响选路，不影响结果）：k 方向的累加次序
;;; 仍是 l = 0,1,2,…（kc 分块按序、块内递增），FMA 仍是单次舍入，
;;; 因此与旧内核**逐位相同**。kc 边界处把寄存器写回 C 再读回是无损的
;;; （f64/f32/i32/i64 的存储都是精确的）。
;;;
;;; 实测（本机 AVX2、单线程、1000³）：f64 235ms → 82ms，f32 → 39ms。
;;; 两者都已在 AVX2 峰值附近（f64 4 宽、f32 8 宽，只差 2 倍），
;;; 单线程已无余量，后续提速只能靠多线程。

(defvar *gemm-mc* 64
  "GEMM 行分块高度（C/A 瓦片）。与 *GEMM-KC*/*GEMM-NC* 一起决定 L2 工作集。")
(defvar *gemm-kc* 256 "GEMM 深度分块（k 方向）。")
(defvar *gemm-nc* 256 "GEMM 列分块宽度（B/C 瓦片）。")

;; 代码生成器必须在编译期可用：宏展开发生在编译期，而它们在本文件里
;; 以 DEFUN 形式出现，故统一包进 EVAL-WHEN（与 core.lisp 的
;; %ASTYPE-CAST-FORM 同一手法）。
(eval-when (:compile-toplevel :load-toplevel :execute)

(defun step-form (fmadd add mul a b c)
  (if fmadd (list fmadd a b c) (list add c (list mul a b))))

;;; ---------------------------------------------------------------
;;; 标量尾块
;;; ---------------------------------------------------------------
(defun build-scalar (name elem-ref es)
  `(defun ,name (pa pb pc k n i0 i1 j0 j1)
     (declare (type sb-sys:system-area-pointer pa pb pc)
              (type fixnum k n i0 i1 j0 j1)
              (optimize (speed 3) (safety 0) (debug 0)))
     (loop for i of-type fixnum from i0 below i1 do
       (loop for j of-type fixnum from j0 below j1 do
         (let ((acc (,elem-ref pc (the fixnum (* (the fixnum (+ (the fixnum (* i n)) j)) ,es)))))
           (loop for l of-type fixnum from 0 below k do
             (setf acc (+ acc
                          (* (,elem-ref pa (the fixnum (* (the fixnum (+ (the fixnum (* i k)) l)) ,es)))
                             (,elem-ref pb (the fixnum (* (the fixnum (+ (the fixnum (* l n)) j)) ,es)))))))
           (setf (,elem-ref pc (the fixnum (* (the fixnum (+ (the fixnum (* i n)) j)) ,es))) acc))))))

;;; ---------------------------------------------------------------
;;; 向量行尾块（1 行 × VW 列）
;;; ---------------------------------------------------------------
(defun build-vecrow (name vw es elem-ref vec-ctor vec-ref add mul fmadd)
  `(defun ,name (pa pb pc k n i0 i1 j0 j1)
     (declare (type sb-sys:system-area-pointer pa pb pc)
              (type fixnum k n i0 i1 j0 j1)
              (optimize (speed 3) (safety 0) (debug 0)))
     (loop for i of-type fixnum from i0 below i1 do
       (let ((arow (the fixnum (* i k)))
             (crow (the fixnum (* i n))))
         (declare (type fixnum arow crow))
         (loop for j of-type fixnum from j0 below j1 by ,vw do
           (let ((acc (,vec-ref pc (the fixnum (+ crow j)))))
             (loop for l of-type fixnum from 0 below k do
               (setf acc ,(step-form fmadd add mul
                                     `(,vec-ctor (,elem-ref pa (the fixnum
                                                                   (* (the fixnum (+ arow l)) ,es))))
                                     `(,vec-ref pb (the fixnum (+ (the fixnum (* l n)) j)))
                                     'acc)))
             (setf (,vec-ref pc (the fixnum (+ crow j))) acc)))))))

;;; ---------------------------------------------------------------
;;; 微块：MR 行 × (NR·VW) 列，C 驻留寄存器，指针步进
;;; ---------------------------------------------------------------
(defun build-micro (name vw es mr nr elem-ref vec-ctor vec-ref add mul fmadd)
  (let* ((pa-r (loop for r below mr collect (gensym "PAR")))
         (pc-r (loop for r below mr collect (gensym "PCR")))
         (kes (gensym "KES"))
         (pbp (gensym "PBP"))
         (nb (gensym "NB"))
         (accs (loop for r below mr append
                     (loop for q below nr collect (gensym "ACC"))))
         (bvs (loop for q below nr collect (gensym "BV")))
         (avs (loop for r below mr collect (gensym "AV")))
         ;; C 列地址：pc-r + q*VW（常数偏移）
         (cix (lambda (r q) `(the fixnum (+ ,(nth r pc-r) ,(* q vw)))))
         (bix (lambda (q) `(the fixnum (+ ,pbp ,(* q vw)))))
         (acc (lambda (r q) (nth (+ (* r nr) q) accs)))
         ;; 头部绑定
         (binds
          (append
           (list (list kes `(the fixnum (* k ,es))))
           (loop for r below mr collect
                 (list (nth r pa-r)
                       `(the fixnum (+ (the fixnum (* (the fixnum (+ (the fixnum (* i k)) kc0)) ,es))
                                       ,@(when (plusp r) `((the fixnum (* ,r ,kes))))))))
           (list (list pbp `(the fixnum (+ (the fixnum (* kc0 n)) j))))
           (loop for r below mr collect
                 (list (nth r pc-r)
                       `(the fixnum (+ (the fixnum (* (the fixnum (+ i ,r)) n)) j))))
           (list (list nb `(the fixnum (- kc-end kc0))))
           (loop for r below mr append
                 (loop for q below nr collect
                       (list (funcall acc r q) (list vec-ref 'pc (funcall cix r q)))))))
         ;; 每轮 k 的绑定：A 的 MR 次广播 + B 的 NR 个向量
         (loop-binds
          (append (loop for r below mr collect
                        (list (nth r avs) (list vec-ctor (list elem-ref 'pa (nth r pa-r)))))
                  (loop for q below nr collect
                        (list (nth q bvs) (list vec-ref 'pb (funcall bix q))))))
         ;; 累加
         (updates
          (loop for r below mr append
                (loop for q below nr append
                      (list (funcall acc r q)
                            (step-form fmadd add mul (nth r avs) (nth q bvs) (funcall acc r q))))))
         ;; 指针步进
         (incfs
          (append (loop for r below mr collect `(incf ,(nth r pa-r) ,es))
                  (list `(incf ,pbp n))))
         ;; 写回
         (stores
          (loop for r below mr append
                (loop for q below nr append
                      (list (list vec-ref 'pc (funcall cix r q)) (funcall acc r q))))))
    `(defun ,name (pa pb pc k n i j kc0 kc-end)
       (declare (type sb-sys:system-area-pointer pa pb pc)
                (type fixnum k n i j kc0 kc-end)
                (optimize (speed 3) (safety 0) (debug 0)))
       (let* ,binds
         (declare (type fixnum ,kes ,@pa-r ,@pc-r ,pbp ,nb))
         (loop repeat ,nb do
           (let* ,loop-binds
             (setf ,@updates))
           ,@incfs)
         (setf ,@stores)
         nil))))

;;; ---------------------------------------------------------------
;;; 驱动
;;; ---------------------------------------------------------------
(defun build-driver (name micro vrow scal mr cols vw)
  `(defun ,name (pa pb pc k n i0 i1 j0 j1)
     (declare (type sb-sys:system-area-pointer pa pb pc)
              (type fixnum k n i0 i1 j0 j1)
              (optimize (speed 3) (safety 0) (debug 0)))
     (let* ((mc (the fixnum *gemm-mc*))
            (kc (the fixnum *gemm-kc*))
            (nc (the fixnum *gemm-nc*))
            (i-end (the fixnum (- i1 (the fixnum (mod (the fixnum (- i1 i0)) ,mr)))))
            (j-end (the fixnum (- j1 (the fixnum (mod (the fixnum (- j1 j0)) ,cols)))))
            (j-vw (the fixnum (- j1 (the fixnum (mod (the fixnum (- j1 j0)) ,vw))))))
       (declare (type fixnum mc kc nc i-end j-end j-vw))
       (loop for ic of-type fixnum from i0 below i-end by mc do
         (loop for jc of-type fixnum from j0 below j-end by nc do
           (loop for kc0 of-type fixnum from 0 below k by kc do
             (let ((kc-end (the fixnum (min k (the fixnum (+ kc0 kc)))))
                   (ic-end (the fixnum (min i-end (the fixnum (+ ic mc)))))
                   (jc-end (the fixnum (min j-end (the fixnum (+ jc nc))))))
               (declare (type fixnum kc-end ic-end jc-end))
               (loop for i of-type fixnum from ic below ic-end by ,mr do
                 (loop for j of-type fixnum from jc below jc-end by ,cols do
                   (,micro pa pb pc k n i j kc0 kc-end)))))))
       ;; 行尾（< MR 行）：整宽向量扫
       (,vrow pa pb pc k n i-end i1 j0 j-vw)
       ;; 列尾（COLS 的余数，按 VW 对齐部分）
       (,vrow pa pb pc k n i0 i-end j-end j-vw)
       ;; 残余列（< VW）：标量
       (,scal pa pb pc k n i0 i1 j-vw j1)
       nil)))

) ; end eval-when

(defmacro define-blocked-gemm (name vw es mr nr elem-ref vec-ctor vec-ref add mul fmadd)
  (let* ((cols (* nr vw))
         (micro (intern (format nil "~a-M" (string name))))
         (vrow (intern (format nil "~a-V" (string name))))
         (scal (intern (format nil "~a-S" (string name)))))
    `(progn
       ,(build-scalar scal elem-ref es)
       ,(build-vecrow vrow vw es elem-ref vec-ctor vec-ref add mul fmadd)
       ,(build-micro micro vw es mr nr elem-ref vec-ctor vec-ref add mul fmadd)
       ,(build-driver name micro vrow scal mr cols vw))))


;;; ============================================================
;;; 四个类型的内核实例化
;;; ============================================================
;; 微块尺寸取舍：f64 用 4×2（8 个 ymm 累加器 = 4 行 × 8 列，每 k 步
;; 2 次 B 载入 + 4 次广播 + 8 次 FMA，正好喂满 2 个 FMA 端口）；
;; f32 用 4×2（4 行 × 16 列）；整型 4×1（没有 FMA，寄存器压力小些）。

(define-blocked-gemm %simd-matmul-f64 4 8 4 2
  sb-sys:sap-ref-double
  sb-simd-avx:f64.4 sb-simd-avx:f64.4-sap-ref
  sb-simd-avx:f64.4+ sb-simd-avx:f64.4* sb-simd-fma:f64.4-fmadd)

(define-blocked-gemm %simd-matmul-f32 8 4 4 2
  sb-sys:sap-ref-single
  sb-simd-avx:f32.8 sb-simd-avx:f32.8-sap-ref
  sb-simd-avx:f32.8+ sb-simd-avx:f32.8* sb-simd-fma:f32.8-fmadd)

(define-blocked-gemm %simd-matmul-i32 8 4 4 1
  sb-sys:signed-sap-ref-32
  sb-simd-avx:s32.8 sb-simd-avx:s32.8-sap-ref
  sb-simd-fma:s32.8+ sb-simd-fma:s32.8-mullo nil)

(define-blocked-gemm %simd-matmul-i64 4 8 4 1
  sb-sys:signed-sap-ref-64
  sb-simd-avx:s64.4 sb-simd-avx:s64.4-sap-ref
  sb-simd-fma:s64.4+ sb-simd-fma:s64.4-mul nil)

;;; ============================================================
;;; f64 的 AVX-512 接管（gemm-avx512.lisp 提供机器码内核）
;;; ============================================================
;;; 与上面的 sb-simd 版本相比只换「寄存器微块」这一层：8 行 × 16 列（两个
;;; 512 位向量）驻留寄存器，每个 k 步 16 条 vfmadd231pd 的广播直接取自
;;; A 的内存操作数（{1to8}），不像 AVX2 那样额外占一条 vbroadcastsd。
;;; 分块顺序、k 递增次序、累加器初值（从 C 读入）都与 sb-simd 版本一致，
;;; 因此结果逐位相同（§7.3）。
;;;
;;; 尾块处理：
;;;   · 列方向不足 16 列 → 掩码内核（k1/k2 分别管低/高 8 列），不越界写；
;;;   · 行方向不足 8 行 → 交回 sb-simd 的 %SIMD-MATMUL-F64（≤7 行，
;;;     占整块算力的 1% 以内，不值得再发射一份 MR=4 的内核）。
(defun %simd-matmul-f64-512 (pa pb pc k n i0 i1 j0 j1)
  (declare (type sb-sys:system-area-pointer pa pb pc)
           (type fixnum k n i0 i1 j0 j1)
           (optimize (speed 3) (safety 0) (debug 0)))
  (multiple-value-bind (kern kernm) (avx512-gemm-kernels)
    (unless (and kern kernm)
      (return-from %simd-matmul-f64-512 (%simd-matmul-f64 pa pb pc k n i0 i1 j0 j1)))
    (let* ((pa0 (sb-sys:sap-int pa))
           (pb0 (sb-sys:sap-int pb))
           (pc0 (sb-sys:sap-int pc))
           (ast (the fixnum (* k 8)))
           (bst (the fixnum (* n 8)))
           (iend (the fixnum (- i1 (the fixnum (mod (the fixnum (- i1 i0)) 8))))))
      (declare (type fixnum pa0 pb0 pc0 ast bst iend))
      (loop for jc of-type fixnum from j0 below j1 by *gemm-nc* do
        (let* ((nc (the fixnum (min *gemm-nc* (the fixnum (- j1 jc)))))
               (nf (the fixnum (floor nc 16)))
               (rm (the fixnum (mod nc 16)))
               (mk (if (plusp rm) (the fixnum (1- (ash 1 rm))) 0)))
          (declare (type fixnum nc nf rm mk))
          (loop for kc0 of-type fixnum from 0 below k by *gemm-kc* do
            (let ((kc (the fixnum (min *gemm-kc* (the fixnum (- k kc0))))))
              (declare (type fixnum kc))
              (loop for ic of-type fixnum from i0 below iend by *gemm-mc* do
                (loop for i of-type fixnum from ic
                        below (the fixnum (min iend (the fixnum (+ ic *gemm-mc*))))
                        by 8 do
                  (let ((pa-i (the fixnum (+ pa0 (the fixnum (* i ast)) (the fixnum (* kc0 8)))))
                        (pb-j (the fixnum (+ pb0 (the fixnum (* kc0 bst)) (the fixnum (* jc 8)))))
                        (pc-j (the fixnum (+ pc0 (the fixnum (* i bst)) (the fixnum (* jc 8))))))
                    (declare (type fixnum pa-i pb-j pc-j))
                    (when (plusp nf)
                      (call-avx512-gemm kern pa-i pb-j pc-j kc ast bst nf 0))
                    (when (plusp rm)
                      (call-avx512-gemm kernm pa-i
                                        (the fixnum (+ pb-j (the fixnum (* nf 128))))
                                        (the fixnum (+ pc-j (the fixnum (* nf 128))))
                                        kc ast bst 1 mk)))))))))
      ;; 行尾（< 8 行）交回 sb-simd 微块
      (when (< iend i1)
        (%simd-matmul-f64 pa pb pc k n iend i1 j0 j1)))
    nil))

(defun %f64-gemm-kernel ()
  "当前可用的 f64 GEMM 内核：有 AVX-512 机器码就用它，否则 sb-simd 版本。"
  (multiple-value-bind (kern kernm) (avx512-gemm-kernels)
    (declare (ignore kernm))
    (if kern #'%simd-matmul-f64-512 #'%simd-matmul-f64)))

;;; ============================================================
;;; 并行调度（C 的二维矩形划分）
;;; ============================================================
;;; 只按行切分时，每个线程都要把整个 B 过一遍：线程数翻倍，B 的 DRAM
;;; 流量也翻倍。实测 31 线程只切行比 8 线程慢 3 倍。改成 (i, j) 二维
;;; 网格后，总流量 ≈ (m/MC)·k·n + (n/NC)·m·k，与线程数基本无关。

(defun %gemm-grid (nthr m n)
  "把 NTHR 个线程排成 MI×NJ 的网格（尽量让两边均衡，减少带宽放大）。"
  (declare (type fixnum nthr m n))
  (let* ((side (max 1 (isqrt (max 1 nthr))))
         (mi (max 1 (min side m)))
         (nj (max 1 (min (floor nthr mi) n))))
    (values mi nj)))

(defun %simd-matmul-parallel (kern pa pb pc m k n)
  "在 M×N 上开线程跑 KERN；KERN 的签名是 (pa pb pc k n i0 i1 j0 j1)。"
  (declare (type function kern)
           (type sb-sys:system-area-pointer pa pb pc)
           (type fixnum m k n))
  (let ((nthr (if (or (<= *simd-matmul-thread-count* 1)
                      (< (* m k n) *simd-matmul-thread-threshold*))
                  1
                  (min *simd-matmul-thread-count* (the fixnum (* m n))))))
    (declare (type fixnum nthr))
    (if (<= nthr 1)
        (funcall kern pa pb pc k n 0 m 0 n)
        (multiple-value-bind (mi nj) (%gemm-grid nthr m n)
          (declare (type fixnum mi nj))
          (let* ((ri (the fixnum (ceiling m mi)))
                 (rj (the fixnum (ceiling n nj)))
                 (threads nil))
            (declare (type fixnum ri rj))
            (dotimes (ii mi)
              (dotimes (jj nj)
                (let* ((i0 (the fixnum (* ii ri)))
                       (j0 (the fixnum (* jj rj)))
                       (i1 (the fixnum (min m (the fixnum (+ i0 ri)))))
                       (j1 (the fixnum (min n (the fixnum (+ j0 rj))))))
                  (declare (type fixnum i0 j0 i1 j1))
                  (when (and (< i0 i1) (< j0 j1))
                    (push (sb-thread:make-thread
                           (lambda () (funcall kern pa pb pc k n i0 i1 j0 j1)))
                          threads)))))
            (dolist (th threads) (sb-thread:join-thread th))))))
  nil)

;;; ============================================================
;;; 通用 dispatch
;;; ============================================================

(defun %simd-matmul-2d-generic (a b dtype out expected-dtype kern elem-size)
  "各类型的公共 dispatch 逻辑。
   elem-size: 每个元素的字节数（f64/i64=8，f32/i32=4）。"
  (declare (type vt a b)
           (type (or null symbol) dtype)
           (type (or null vt) out)
           (type symbol expected-dtype)
           (type function kern)
           (type fixnum elem-size))
  (unless *simd-matmul-enabled*
    (return-from %simd-matmul-2d-generic nil))

  (let ((ashape (vt-shape a))
        (bshape (vt-shape b)))
    (unless (and (= (length ashape) 2) (= (length bshape) 2))
      (return-from %simd-matmul-2d-generic nil))

    (let* ((m  (the fixnum (first ashape)))
           (k  (the fixnum (second ashape)))
           (k2 (the fixnum (first bshape)))
           (n  (the fixnum (second bshape))))
      (declare (type fixnum m k k2 n))
      (unless (= k k2) (return-from %simd-matmul-2d-generic nil))
      (unless (>= (* m k n) *simd-matmul-threshold*)
        (return-from %simd-matmul-2d-generic nil))

      ;; 结果 dtype 由「输入提升 + 显式 :dtype」决定，**不由 out 决定**（§4.2 H3）。
      ;; 先算出自然结果 dtype，与调用方期望比较；out 的 dtype 若与之不符，
      ;; 直接拒绝快路径（返回 nil 交回 vt-einsum，由它给出带 "vt-einsum" 前缀
      ;; 的清晰报错），绝不能把 out 的 dtype 当作结果 dtype 接受。
      (let ((final (or dtype (vt-promote-type (vt-dtype a) (vt-dtype b)))))
        (unless (eq final expected-dtype)
          (return-from %simd-matmul-2d-generic nil))
        (when (and out (not (eq (vt-dtype out) final)))
          (return-from %simd-matmul-2d-generic nil)))

      (when out
        (unless (and (equal (vt-shape out) (list m n))
                     (vt-contiguous-p out)
                     (vt-out-writable-p out))
          (return-from %simd-matmul-2d-generic nil)))

      (let* ((a-c1 (if (and (eq (vt-dtype a) expected-dtype)
                           (vt-contiguous-p a))
                      a (vt-contiguous (vt-astype a expected-dtype))))
             (b-c1 (if (and (eq (vt-dtype b) expected-dtype)
                           (vt-contiguous-p b))
                      b (vt-contiguous (vt-astype b expected-dtype))))
             ;; 别名保护：物理区间重叠判定（不能只比 vt-data 指针，
             ;; 否则漏掉「同 data 异 offset」的重叠视图）。
             (a-c (if (and out (%vt-views-overlap-p out a-c1))
                      (vt-copy a-c1) a-c1))
             (b-c (if (and out (%vt-views-overlap-p out b-c1))
                      (vt-copy b-c1) b-c1))
             (output (or out (vt-zeros (list m n) :dtype expected-dtype)))
             (a-data (vt-data a-c))
             (b-data (vt-data b-c))
             (out-data (vt-data output)))
        (when out (vt-fill output 0))

        (let ((a-off (the fixnum (* (vt-offset a-c) elem-size)))
              (b-off (the fixnum (* (vt-offset b-c) elem-size)))
              (out-off (the fixnum (* (vt-offset output) elem-size))))
          (declare (type fixnum a-off b-off out-off))
          (sb-sys:with-pinned-objects (a-data b-data out-data)
            (let ((pa (sb-sys:sap+ (sb-sys:vector-sap a-data) a-off))
                  (pb (sb-sys:sap+ (sb-sys:vector-sap b-data) b-off))
                  (pc (sb-sys:sap+ (sb-sys:vector-sap out-data) out-off)))
              (declare (type sb-sys:system-area-pointer pa pb pc))
              (%simd-matmul-parallel kern pa pb pc m k n))))
        output))))

(defun %simd-matmul-2d (a b dtype out)
  "按 dtype 分发。返回 VT 成功，NIL 让 vt-matmul 回退。"
  (with-float-safe
    (or (%simd-matmul-2d-generic a b dtype out :float64
				 (%f64-gemm-kernel) 8)
	(%simd-matmul-2d-generic a b dtype out :float32
				 #'%simd-matmul-f32 4)
	(%simd-matmul-2d-generic a b dtype out :int32
				 #'%simd-matmul-i32 4)
	(%simd-matmul-2d-generic a b dtype out :int64
				 #'%simd-matmul-i64 8)
	nil)))

;; *simd-matmul-2d-fn* 在 linalg.lisp 文件中定义，这里注册
(setf *simd-matmul-2d-fn* #'%simd-matmul-2d)

;;; ============================================================
;;; 裸数据 2D GEMM（供 linalg.lisp 的 einsum 批量 GEMM 通道调用）
;;; ============================================================
;;; 背景（§16.26）：einsum 的批量 GEMM 快通道原先只调 linalg.lisp 的标量
;;; 内核（%MATMUL-DF-FAST-KERNEL 家族：4 路手写展开、无 SIMD、无分块），
;;; 而 VT-MATMUL 走本文件的 AVX/FMA + 64×256×256 分块内核。两条路径互不
;;; 相通，于是同一台机器上
;;;     (vt-einsum "ij,jk->ik" a b)   比   (vt-matmul a b)   慢 5.7~6.4×
;;; （1024² float64，单线程；多线程下内存带宽掩盖一部分，实测 1.74×）。
;;;
;;; 这里提供一个「裸数据」入口：只做 SAP 换算 + 阈值判定，其余全部复用
;;; 已有内核与 %SIMD-MATMUL-PARALLEL 的线程调度，因此两条路径的数值行为
;;; 逐位一致（同一内核、同一分块顺序）。
;;;
;;; 约定：A/B/C 均为行主序连续、互不重叠；偏移是**元素**下标（非字节）；
;;; 语义是累加 C += A·B（与 einsum 的 incf 语义一致，调用方负责清零）。
(defun %simd-gemm-raw (a-data b-data c-data m k n
                       a-off b-off c-off elem-kind)
  (declare (type (simple-array * (*)) a-data b-data c-data)
           (type fixnum m k n a-off b-off c-off)
           (type symbol elem-kind)
           (optimize (speed 3) (safety 0) (debug 0)))
  (when (or (not *simd-matmul-enabled*)
            (< (* m k n) *simd-matmul-threshold*))
    (return-from %simd-gemm-raw nil))
  (multiple-value-bind (kern elem-size)
      (case elem-kind
        (:float64 (values (%f64-gemm-kernel) 8))
        (:float32 (values #'%simd-matmul-f32 4))
        (:int64   (values #'%simd-matmul-i64 8))
        (:int32   (values #'%simd-matmul-i32 4))
        (t        (values nil 0)))
    (unless kern
      (return-from %simd-gemm-raw nil))
    (let ((kern% (the function kern))
          (esize (the fixnum elem-size)))
      (declare (type function kern%) (type fixnum esize))
      (sb-sys:with-pinned-objects (a-data b-data c-data)
        (let ((pa (sb-sys:sap+ (sb-sys:vector-sap a-data) (the fixnum (* a-off esize))))
              (pb (sb-sys:sap+ (sb-sys:vector-sap b-data) (the fixnum (* b-off esize))))
              (pc (sb-sys:sap+ (sb-sys:vector-sap c-data) (the fixnum (* c-off esize)))))
          (declare (type sb-sys:system-area-pointer pa pb pc))
          (%simd-matmul-parallel kern% pa pb pc m k n))))
    t))

;; *simd-gemm-raw-fn* 在 linalg.lisp 中定义（它先于本文件编译），这里注册。
(setf *simd-gemm-raw-fn* #'%simd-gemm-raw)

;;;; 批量矩阵乘法的 SIMD 快速路径
;;;;
;;;; 支持：
;;;;   A: (..., M, K) @ B: (..., K, N) -> (..., M, N)     [批对齐]
;;;;   A: (..., M, K) @ B: (K, N)     -> (..., M, N)     [B 广播]
;;;;
;;;; 只支持 f64 / f32（i32/i64 的批量场景少，且 s64.4-mul 是软件模拟）。
;;;; 复用已定义的 %simd-matmul-{f64,f32}-rows 内核。
;;; ============================================================
;;; 核心：批量 SIMD matmul
;;; ============================================================

(defun %simd-batched-matmul-core
    (a b dtype out expected-dtype kern elem-size)
  "批量矩阵乘法核心。
   A: (..., M, K)，B: (..., K, N)。任一侧可缺批维度（numpy 广播）。
   OUT: (..., M, N)。ELEM-SIZE: f64/i64=8，f32/i32=4。"
  (declare (type vt a b)
           (type (or null symbol) dtype)
           (type (or null vt) out)
           (type symbol expected-dtype)
           (type function kern)
           (type fixnum elem-size))
  (block core
    (unless *simd-matmul-enabled*
      (return-from core nil))
    (let* ((ashape (vt-shape a))
           (bshape (vt-shape b))
           (arank  (length ashape))
           (brank  (length bshape)))
      (unless (and (>= arank 2) (>= brank 2))
        (return-from core nil))
      (unless (or (>= arank 3) (>= brank 3))
        (return-from core nil))
      (let* ((m  (the fixnum (nth (- arank 2) ashape)))
             (k  (the fixnum (nth (1- arank) ashape)))
             (k2 (the fixnum (nth (- brank 2) bshape)))
             (n  (the fixnum (nth (1- brank) bshape))))
        (declare (type fixnum m k k2 n))
        (unless (= k k2)
          (return-from core nil))
        (let* ((a-batch     (subseq ashape 0 (- arank 2)))
               (b-batch     (subseq bshape 0 (- brank 2)))
               (a-broadcast (null a-batch))
               (b-broadcast (null b-batch)))
          (unless (or a-broadcast b-broadcast (equal a-batch b-batch))
            (return-from core nil))
          (let* ((batch-shape (cond (a-broadcast b-batch)
                                    (b-broadcast a-batch)
                                    (t           a-batch)))
                 (batch-size  (reduce #'* batch-shape))
                 (total-work  (* batch-size m k n))
                 (out-shape   (append batch-shape (list m n))))
            (declare (type fixnum batch-size total-work))
            (assert (consp batch-shape) ()
                    "SIMD 批量 matmul: batch-shape 异常 = ~a (a-batch=~a b-batch=~a)"
                    batch-shape a-batch b-batch)
            (assert (= (length out-shape) (+ (length batch-shape) 2)) ()
                    "SIMD 批量 matmul: out-shape 长度错 = ~a" out-shape)
            (unless (>= total-work *simd-matmul-threshold*)
              (return-from core nil))
            ;; 类型检查
            ;; 结果 dtype 由「输入提升 + 显式 :dtype」决定，**不由 out 决定**
            ;; （§4.2 H3 严格相等契约）。out 的 dtype 必须精确匹配，
            ;; 否则返回 nil 让上层走通用路径（那里会给出规范错误）。
            (let ((final (or dtype
                             (vt-promote-type (vt-dtype a) (vt-dtype b)))))
              (unless (eq final expected-dtype)
                (return-from core nil))
              (when (and out (not (eq (vt-dtype out) final)))
                (return-from core nil)))
            ;; OUT 形状检查：形状必须精确等于批量 matmul 结果形状。
            ;; 非连续 out 不在此拦截 —— 本 SIMD 内核要求连续存储，
            ;; 非连续 out 返回 nil 交给 einsum 通用路径（同样正确，只是慢）。
            (when out
              (unless (and (equal (vt-shape out) out-shape)
                           (vt-out-writable-p out)
                           (vt-contiguous-p out))
                (return-from core nil)))
            ;; 连续性 + 类型强制 + 创建 output
            (let* ((a-c1 (if (and (eq (vt-dtype a) expected-dtype)
                                 (vt-contiguous-p a))
                            a
                            (vt-contiguous (vt-astype a expected-dtype))))
                   (b-c1 (if (and (eq (vt-dtype b) expected-dtype)
                                 (vt-contiguous-p b))
                            b
                            (vt-contiguous (vt-astype b expected-dtype))))
                   ;; 别名保护：out 与 a/b 共享底层存储且**物理区间重叠**时，
                   ;; 必须先快照。原实现只比较 (vt-data ...) 指针，
                   ;; 会漏掉「同一底层数组但 offset 不同」的重叠
                   ;; （如 out = base[2:], a = base[:4]），
                   ;; 导致结果被就地覆盖而静默出错。改用统一原语
                   ;; %vt-views-overlap-p，它比较实际访问区间。
                   (a-c (if (and out (%vt-views-overlap-p out a-c1))
                            (vt-copy a-c1) a-c1))
                   (b-c (if (and out (%vt-views-overlap-p out b-c1))
                            (vt-copy b-c1) b-c1))
                   (output (or out
                               (vt-zeros out-shape :dtype expected-dtype))))
              (when out (vt-fill output 0))
              (assert (equal (vt-shape output) out-shape) ()
                      "SIMD 批量 matmul: output 形状 ~a ≠ 期望 ~a"
                      (vt-shape output) out-shape)
              (let ((a-data (vt-data a-c))
                    (b-data (vt-data b-c))
                    (out-data (vt-data output))
                    (a-batch-stride   (if a-broadcast 0 (* m k)))
                    (b-batch-stride   (if b-broadcast 0 (* k n)))
                    (out-batch-stride (* m n))
                    (a-off   (* (vt-offset a-c)     elem-size))
                    (b-off   (* (vt-offset b-c)     elem-size))
                    (out-off (* (vt-offset output)  elem-size)))
                (declare (type fixnum a-batch-stride b-batch-stride
                               out-batch-stride a-off b-off out-off))
                (sb-sys:with-pinned-objects (a-data b-data out-data)
                  (let ((a-sap   (sb-sys:sap+ (sb-sys:vector-sap a-data)  a-off))
                        (b-sap   (sb-sys:sap+ (sb-sys:vector-sap b-data)  b-off))
                        (out-sap (sb-sys:sap+ (sb-sys:vector-sap out-data) out-off)))
                    (declare (type sb-sys:system-area-pointer a-sap b-sap out-sap))
                    (flet ((run-one-batch (bi)
                             "跑一个 batch 的 M×N 输出。A/B 广播时 stride=0，指针不变。"
                             (funcall kern
                                      (sb-sys:sap+
                                       a-sap (* bi a-batch-stride elem-size))
                                      (sb-sys:sap+
                                       b-sap (* bi b-batch-stride elem-size))
                                      (sb-sys:sap+
                                       out-sap (* bi out-batch-stride elem-size))
                                      k n 0 m 0 n)))
                      (declare (inline run-one-batch))
                      (cond
                        ;; 总工作量小 → 串行
                        ((or (< batch-size 2)
                             (< total-work *simd-matmul-thread-threshold*))
                         (dotimes (bi batch-size)
                           (run-one-batch bi)))
                        ;; 总工作量大 → 批间并行
                        (t
                         (let* ((nthr (min *simd-matmul-thread-count*
                                           batch-size))
                                (batches-per (ceiling batch-size nthr))
                                (threads '()))
                           (declare (type fixnum nthr batches-per))
                           (loop for tid of-type fixnum below nthr
                                 do (let* ((b0 (* tid batches-per))
                                           (b1 (min batch-size
                                                    (+ b0 batches-per))))
                                      (declare (type fixnum b0 b1))
                                      (when (< b0 b1)
                                        (push (sb-thread:make-thread
                                               (lambda ()
                                                 (loop for bi from b0 below b1
                                                       do (run-one-batch bi))))
                                              threads))))
                           (dolist (th threads)
                             (sb-thread:join-thread th)))))))))
              output)))))))

;;; ============================================================
;;; 分派：f64 → f32，其余回退
;;; ============================================================

(defun %simd-batched-matmul-f64 (a b dtype out)
  (with-float-safe
    (%simd-batched-matmul-core a b dtype out :float64
                               (%f64-gemm-kernel) 8)))

(defun %simd-batched-matmul-f32 (a b dtype out)
  (with-float-safe
    (%simd-batched-matmul-core a b dtype out :float32
                               #'%simd-matmul-f32 4)))

(defun %simd-batched-matmul-i32 (a b dtype out)
  (with-float-safe
    (%simd-batched-matmul-core a b dtype out :int32
                                #'%simd-matmul-i32 4)))

(defun %simd-batched-matmul-i64 (a b dtype out)
  (with-float-safe
    (%simd-batched-matmul-core a b dtype out :int64
                                #'%simd-matmul-i64 8)))

(defun %simd-batched-matmul-dispatch (a b dtype out)
  "顶层分派。失败返回 NIL 让 vt-matmul 回退 einsum。"
  (or (%simd-batched-matmul-f64 a b dtype out)
      (%simd-batched-matmul-f32 a b dtype out)
      (%simd-batched-matmul-i32 a b dtype out)
      (%simd-batched-matmul-i64 a b dtype out)
      nil))

;; 注册到 vt-matmul
(setf *simd-batched-matmul-fn* #'%simd-batched-matmul-dispatch)
