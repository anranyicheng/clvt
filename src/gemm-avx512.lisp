;;;; gemm-avx512.lisp — 运行时发射的 AVX-512 float64 GEMM 微内核
;;;;
;;;; 为什么需要这一层
;;;; ================
;;;; sb-simd 只覆盖到 AVX2（256 位），每个 k 步的「广播 + FMA」要拆成两条指令，
;;;; 实测内层每 8 次 FMA 要花掉 47 个 uop（前端受限），1000³ float64 单线程
;;;; 停在算力峰值的 ~25%。AVX-512 把广播融进 FMA 的内存操作数（{1to8}），
;;;; 16 次 FMA 只剩 18 个 uop，理论上限翻倍。
;;;;
;;;; 但 sb-simd / SBCL 2.6.8 都还没有 zmm 寄存器与 EVEX 编码（没有
;;;; SIMD-PACK-512 的 primitive type，也没有 ZMM TN），写 VOP 无从下手。
;;;; 因此这里直接**发射机器码**：一段 8 行 × 16 列的微内核，用
;;;; SB-SYS:ALLOCATE-SYSTEM-MEMORY + mprotect 变成可执行页，再用
;;;; SB-ALIEN:ALIEN-FUNCALL 调用。
;;;;
;;;; 数值语义（§7.3）
;;;; ================
;;;; 累加器**从 C 的值开始**（先 vmovupd 载入、末尾原样写回），k 仍按
;;;; 0,1,2,… 递增且单次舍入（FMA），因此与 sb-simd 路径的
;;;; ((C + a₀b₀) + a₁b₁)… 完全同一条舍入链 —— 逐位相同。
;;;;
;;;; 选路（§7.4）
;;;; ============
;;;; 只有「x86-64 + Linux + CPU/OS 支持 AVX-512F」且 `*avx512-gemm-enabled*`
;;;; 为真时才接管，否则 `AVX512-GEMM-KERNELS` 返回 NIL，调用方原样走
;;;; sb-simd 路径。字节发射本身与平台无关（它在任何平台上都能跑通、只是
;;;; 结果没人用），因此这一层不会让库在别的架构上加载失败。

(in-package :clvt)

;;; ============================================================
;;; 1. 字节发射器（只覆盖本文件用到的指令）
;;; ============================================================
;;; 寄存器编号：rax=0 rcx=1 rdx=2 rbx=3 rsp=4 rbp=5 rsi=6 rdi=7 r8…r15=8…15，
;;; zmm 为 0…31。EVEX 的字段语义是按 gas 的字节逐个核对出来的（不是照抄
;;; 手册）：R̄/R'/B̄/X̄/v̄/V' 全部取反位，唯独 v 字段那 4 位是「寄存器号左移
;;; 一位再取反」，这条不写下来下次一定踩。

(defconstant +reg-rax+ 0)
(defconstant +reg-rcx+ 1)
(defconstant +reg-rdx+ 2)
(defconstant +reg-rbx+ 3)
(defconstant +reg-rbp+ 5)
(defconstant +reg-rsi+ 6)
(defconstant +reg-rdi+ 7)
(defconstant +reg-r8+ 8)
(defconstant +reg-r9+ 9)
(defconstant +reg-r12+ 12)
(defconstant +reg-r13+ 13)
(defconstant +reg-r14+ 14)
(defconstant +reg-r15+ 15)

(defun %eb (buf x) (vector-push-extend (logand x #xff) buf))
(defun %ed (buf x) (%eb buf x) (%eb buf (ash x -8)) (%eb buf (ash x -16)) (%eb buf (ash x -24)))
(defun %epos (buf) (fill-pointer buf))

(defun %patch-rel32 (buf at target)
  "把位置 AT 处的 rel32 填成「跳到 TARGET」（位移相对 AT+4）。"
  (let ((v (- target (+ at 4))))
    (setf (aref buf at) (logand v #xff)
          (aref buf (+ at 1)) (logand (ash v -8) #xff)
          (aref buf (+ at 2)) (logand (ash v -16) #xff)
          (aref buf (+ at 3)) (logand (ash v -24) #xff))))

(defun %e-rex (buf w r x &optional b)
  (let ((v (logior #x40 (if w 8 0) (if r 4 0) (if x 2 0) (if b 1 0))))
    (unless (= v #x40) (%eb buf v))))

(defun %e-push (buf r)
  (if (< r 8) (%eb buf (+ #x50 r)) (progn (%eb buf #x41) (%eb buf (+ #x50 (- r 8))))))
(defun %e-pop (buf r)
  (if (< r 8) (%eb buf (+ #x58 r)) (progn (%eb buf #x41) (%eb buf (+ #x58 (- r 8))))))

(defun %e-mov-rr (buf dst src)          ; MOV dst, src   (89 /r，dst=rm)
  (%e-rex buf t (>= src 8) nil (>= dst 8))
  (%eb buf #x89)
  (%eb buf (logior #xc0 (ash (logand src 7) 3) (logand dst 7))))

(defun %e-mov-from-rsp (buf dst disp)   ; MOV dst, [rsp+disp]   (8b /r + SIB)
  (%e-rex buf t (>= dst 8) nil nil)
  (%eb buf #x8b)
  (cond ((zerop disp) (%eb buf (logior #x04 (ash (logand dst 7) 3))) (%eb buf #x24))
        ((<= -128 disp 127)
         (%eb buf (logior #x44 (ash (logand dst 7) 3))) (%eb buf #x24) (%eb buf disp))
        (t (%eb buf (logior #x84 (ash (logand dst 7) 3))) (%eb buf #x24) (%ed buf disp))))

(defun %e-lea (buf dst base index scale) ; LEA dst, [base + index*scale]
  (%e-rex buf t (>= dst 8) (and index (>= index 8)) (>= base 8))
  (%eb buf #x8d)
  (let ((sc (ecase scale (1 0) (2 1) (4 2) (8 3))))
    ;; base 的低 3 位是 101（rbp/r13）时 mod=00 会被读成 RIP 相对，必须补 disp8=0。
    (let ((mod (if (= (logand base 7) 5) 1 0)))
      (%eb buf (logior (ash mod 6) (ash (logand dst 7) 3) 4))
      (%eb buf (logior (ash sc 6) (ash (logand index 7) 3) (logand base 7)))
      (when (= mod 1) (%eb buf 0)))))

(defun %e-add-rr (buf dst src)          ; ADD dst, src   (01 /r，dst=rm)
  (%e-rex buf t (>= src 8) nil (>= dst 8))
  (%eb buf #x01)
  (%eb buf (logior #xc0 (ash (logand src 7) 3) (logand dst 7))))

(defun %e-add-imm (buf dst imm)         ; ADD dst, imm（能塞进 imm8 就用 83 /0）
  (%e-rex buf t nil nil (>= dst 8))
  (if (<= -128 imm 127)
      (progn (%eb buf #x83) (%eb buf (logior #xc0 (logand dst 7))) (%eb buf imm))
      (progn (%eb buf #x81) (%eb buf (logior #xc0 (logand dst 7))) (%ed buf imm))))

(defun %e-dec (buf r)                   ; DEC r   (ff /1)
  (%e-rex buf t nil nil (>= r 8))
  (%eb buf #xff) (%eb buf (logior #xc8 (logand r 7))))

(defun %e-shr32 (buf r32 imm)           ; SHR r32, imm8   (c1 /5)
  (%eb buf #xc1) (%eb buf (logior #xe8 (logand r32 7))) (%eb buf imm))

(defun %e-kmov-to-k (buf k r32)         ; KMOVW k, r32   (VEX.128.0F.W0 92 /r)
  (%eb buf #xc5) (%eb buf #xf8) (%eb buf #x92)
  (%eb buf (logior #xc0 (ash (logand k 7) 3) (logand r32 7))))

(defun %e-jnz (buf)                     ; JNZ rel32，返回 rel32 字段位置供回填
  (%eb buf #x0f) (%eb buf #x85) (let ((p (%epos buf))) (%ed buf 0) p))

(defun %e-vzeroupper (buf) (%eb buf #xc5) (%eb buf #xf8) (%eb buf #x77))
(defun %e-ret (buf) (%eb buf #xc3))

;;; --- EVEX 三/四操作数 ---
(defun %e-evex (buf mm pp w opcode reg vvvv base index scale disp tuple
                &key (bcst 0) (z 0) (aaa 0))
  "EVEX 前缀 + opcode + ModRM( + SIB )( + disp )。TUPLE 是 disp8 的缩放单位
   （整向量 64 字节、广播 8 字节）。VVVV 为 NIL 表示该指令不用 vvvv。"
  (let* ((unused (null vvvv))
         (vv (or vvvv 0))
         (field (if unused #xf (logand (lognot (ash vv 1)) #xf)))
         (vbar (if unused 1 (logand (lognot (ash vv -3)) 1)))
         (vprime (if unused 1 (logand (lognot (ash vv -4)) 1))))
    (%eb buf #x62)
    (%eb buf (logior (ash (logand (lognot (ash reg -3)) 1) 7)
                     (ash (if index (logand (lognot (ash index -3)) 1) 1) 6)
                     (ash (logand (lognot (ash base -3)) 1) 5)
                     (ash (logand (lognot (ash reg -4)) 1) 4)
                     mm))
    (%eb buf (logior (ash w 7) (ash vbar 6) (ash field 2) pp))
    (%eb buf (logior (ash z 7) (ash 2 5) (ash bcst 4) (ash vprime 3) aaa))
    (%eb buf opcode)
    (let* ((sc (ecase scale (1 0) (2 1) (4 2) (8 3)))
           (sib (or index (= (logand base 7) 4)))
           (rm (if sib 4 (logand base 7)))
           (mod (cond ((and (zerop disp) (/= (logand base 7) 5)) 0)
                      ((and (zerop (mod disp tuple)) (<= (abs (/ disp tuple)) 127)) 1)
                      (t 2))))
      (%eb buf (logior (ash mod 6) (ash (logand reg 7) 3) rm))
      (when sib
        (%eb buf (logior (ash sc 6) (ash (if index (logand index 7) 4) 3) (logand base 7))))
      (case mod
        (1 (%eb buf (/ disp tuple)))
        (2 (%ed buf disp))))))

(defun %e-vmovupd-load (buf zreg base disp &key aaa z)
  (%e-evex buf 1 1 1 #x10 zreg nil base nil 1 disp 64 :aaa (or aaa 0) :z (if z 1 0)))

(defun %e-vmovupd-store (buf zreg base disp &key aaa)
  (%e-evex buf 1 1 1 #x11 zreg nil base nil 1 disp 64 :aaa (or aaa 0)))

(defun %e-vfmadd231pd (buf zreg bsrc base index disp)
  "dst = bsrc * m64{1to8} + dst —— 广播直接取自内存，不占独立指令。"
  (%e-evex buf 2 1 1 #xb8 zreg bsrc base index 1 disp 8 :bcst 1))

;;; ============================================================
;;; 2. 微内核：8 行 × 16 列
;;; ============================================================
;;; 参数（SysV，全部按整数传）：rcx=kc rdi=A rsi=B rdx=C r8=A 行距（字节）
;;; r9=B/C 行距（字节）[rsp+8]=16 列块个数 [rsp+16]=列掩码（仅掩码版）
;;;
;;; 寄存器分配：
;;;   zmm0/1     B 的两个 8 宽向量
;;;   zmm2…17    16 个累加器（c[r][j] = 2 + 2r + j）
;;;   rax rbx rbp r12  行 0/2/4/6 的基址（奇行用 + r8 索引，省掉一半指针）
;;;   r8=A 行距 r9=B/C 行距 r13=B 指针 r14=k 计数 r15=C 行指针 r10=块计数
(defconstant +gemm-mr+ 8)
(defconstant +gemm-nr+ 16)
(defconstant +zmm-acc0+ 2)
(defconstant +zmm-accn+ 16)

(defun %avx512-gemm-kernel-bytes (masked)
  "生成内核字节。MASKED 为真时用 k1/k2 掩码处理「不足 16 列」的尾块。"
  (let ((buf (make-array 0 :adjustable t :fill-pointer 0 :element-type '(unsigned-byte 8)))
        (fixups '()))
    (dolist (r (list +reg-rbx+ +reg-rbp+ +reg-r12+ +reg-r13+ +reg-r14+ +reg-r15+))
      (%e-push buf r))
    (%e-mov-from-rsp buf 10 56)                  ; r10 = 列块个数
    (when masked
      (%e-mov-from-rsp buf +reg-rax+ 64)         ; rax = 掩码
      (%e-kmov-to-k buf 1 +reg-rax+)             ; k1 = 低 8 列
      (%e-shr32 buf +reg-rax+ 8)
      (%e-kmov-to-k buf 2 +reg-rax+))            ; k2 = 高 8 列
    (%e-mov-rr buf 11 +reg-rdi+)                 ; r11 = A 起点（每轮复位）
    (let ((top (%epos buf)))
      ;; --- 累加器从 C 载入（不是清零：保证与顺序累加同一条舍入链）---
      (%e-mov-rr buf +reg-r15+ +reg-rdx+)
      (dotimes (r +gemm-mr+)
        (%e-vmovupd-load buf (+ +zmm-acc0+ (* 2 r)) +reg-r15+ 0
                         :aaa (if masked 1 0) :z masked)
        (%e-vmovupd-load buf (+ +zmm-acc0+ (* 2 r) 1) +reg-r15+ 64
                         :aaa (if masked 2 0) :z masked)
        (%e-add-rr buf +reg-r15+ +reg-r9+))
      ;; --- 每轮的行基址 / B 指针 / k 计数 ---
      (%e-mov-rr buf +reg-rax+ 11)
      (%e-lea buf +reg-rbx+ 11 +reg-r8+ 2)
      (%e-lea buf +reg-rbp+ +reg-rbx+ +reg-r8+ 2)
      (%e-lea buf +reg-r12+ +reg-rbp+ +reg-r8+ 2)
      (%e-mov-rr buf +reg-r13+ +reg-rsi+)
      (%e-mov-rr buf +reg-r14+ +reg-rcx+)
      (let ((kloop (%epos buf)))
        (if masked
            (progn (%e-vmovupd-load buf 0 +reg-r13+ 0 :aaa 1 :z t)
                   (%e-vmovupd-load buf 1 +reg-r13+ 64 :aaa 2 :z t))
            (progn (%e-vmovupd-load buf 0 +reg-r13+ 0)
                   (%e-vmovupd-load buf 1 +reg-r13+ 64)))
        (dotimes (r +gemm-mr+)
          (let ((q (ecase (ash r -1)
                     (0 +reg-rax+) (1 +reg-rbx+) (2 +reg-rbp+) (3 +reg-r12+)))
                (idx (if (evenp r) nil +reg-r8+)))
            (dotimes (j 2)
              (%e-vfmadd231pd buf (+ +zmm-acc0+ (* 2 r) j) j q idx 0))))
        (dolist (r (list +reg-rax+ +reg-rbx+ +reg-rbp+ +reg-r12+))
          (%e-add-imm buf r 8))
        (%e-add-rr buf +reg-r13+ +reg-r9+)
        (%e-dec buf +reg-r14+)
        (push (cons (%e-jnz buf) kloop) fixups))
      ;; --- 原样写回 ---
      (%e-mov-rr buf +reg-r15+ +reg-rdx+)
      (dotimes (r +gemm-mr+)
        (%e-vmovupd-store buf (+ +zmm-acc0+ (* 2 r)) +reg-r15+ 0 :aaa (if masked 1 0))
        (%e-vmovupd-store buf (+ +zmm-acc0+ (* 2 r) 1) +reg-r15+ 64 :aaa (if masked 2 0))
        (%e-add-rr buf +reg-r15+ +reg-r9+))
      ;; --- 下一个 16 列块 ---
      (%e-add-imm buf +reg-rsi+ 128)
      (%e-add-imm buf +reg-rdx+ 128)
      (%e-dec buf 10)
      (push (cons (%e-jnz buf) top) fixups))
    (%e-vzeroupper buf)
    (dolist (r (list +reg-r15+ +reg-r14+ +reg-r13+ +reg-r12+ +reg-rbp+ +reg-rbx+))
      (%e-pop buf r))
    (%e-ret buf)
    (dolist (f fixups) (%patch-rel32 buf (car f) (cdr f)))
    (coerce buf '(simple-array (unsigned-byte 8) (*)))))

;;; ============================================================
;;; 3. 可执行内存 + 平台探测
;;; ============================================================
(defvar *avx512-gemm-enabled* t
  "总开关。探测失败或用户显式置 NIL 时，GEMM 一律走 sb-simd 路径。")

(defvar *avx512-kernels* :uninitialized
  "NIL 表示不可用；否则是 (无掩码函数 . 掩码函数)。")

(defun %cpu-avx512f-p ()
  "CPU 与操作系统是否都启用了 AVX-512F。

   只认 /proc/cpuinfo 的 flags：Linux 在填充这份表之前会先用 XGETBV 检查
   XCR0 里的 opmask/ZMM 状态位，操作系统没开 AVX-512 时它会把该标志摘掉。
   因此这里不必自己再跑一遍 CPUID —— 相比之下「先发一段探测代码再执行」
   的代价是真出错时进程 SIGILL 死掉，而不是返回 NIL。"
  #+x86-64
  (let ((path #p"/proc/cpuinfo"))
    (and (probe-file path)
         (with-open-file (s path :if-does-not-exist nil)
           (and s
                (loop for line = (read-line s nil nil)
                      while line
                      thereis (and (> (length line) 9)
                                   (string= line "flags" :end1 5 :end2 5)
                                   (search " avx512f "
                                           (concatenate 'string " " line " "))))))))
  #-x86-64
  nil)

#+sbcl
(progn
  (sb-alien:define-alien-routine ("mprotect" %avx512-mprotect) sb-alien:int
    (addr sb-sys:system-area-pointer) (len sb-alien:unsigned-long) (prot sb-alien:int))

  (defun %avx512-install (bytes)
    "把 BYTES 拷进一段可执行内存，返回可直接 alien-funcall 的函数对象。"
    (let* ((n (length bytes))
           (sap (sb-sys:allocate-system-memory n)))
      (dotimes (i n) (setf (sb-sys:sap-ref-8 sap i) (aref bytes i)))
      (unless (zerop (%avx512-mprotect sap n 7))    ; PROT_READ|WRITE|EXEC
        (error "clvt: 无法把 AVX-512 内核页设为可执行"))
      (sb-alien:sap-alien
       sap
       (sb-alien:function sb-alien:void
                          sb-alien:unsigned-long sb-alien:unsigned-long
                          sb-alien:unsigned-long sb-alien:unsigned-long
                          sb-alien:unsigned-long sb-alien:unsigned-long
                          sb-alien:unsigned-long sb-alien:unsigned-long))))

  (defun call-avx512-gemm (kern pa pb pc kc a-stride b-stride jblocks mask)
    "调用内核：PA/PB/PC 是 SAP 的整数地址，其余分别是 k 长度、A 行距（字节）、
       B/C 行距（字节）、16 列块个数、列掩码（掩码版才用）。"
    (sb-alien:alien-funcall kern pa pb pc kc a-stride b-stride jblocks mask))

  (defun %avx512-gemm-install ()
    (handler-case
        (cons (%avx512-install (%avx512-gemm-kernel-bytes nil))
              (%avx512-install (%avx512-gemm-kernel-bytes t)))
      (error () nil))))

(defun avx512-gemm-kernels ()
  "返回 (values 无掩码内核 掩码内核)；不可用时两者都是 NIL。

   机器码只在首次调用时发射（一次 mmap + 两次 mprotect），之后走缓存。
   `*AVX512-GEMM-ENABLED*` 在**每次**调用时都会检查：把它置 NIL 会立即让
   调用方退回 sb-simd 路径（已发射的页保留着，置回 T 又立刻生效），因此
   这个开关可以直接当 A/B 对比的开关用。"
  (when (eq *avx512-kernels* :uninitialized)
    (setf *avx512-kernels*
          (when (and *avx512-gemm-enabled* (%cpu-avx512f-p))
            #+sbcl (%avx512-gemm-install)
            #-sbcl nil)))
  (when (and *avx512-gemm-enabled* (consp *avx512-kernels*))
    (values (car *avx512-kernels*) (cdr *avx512-kernels*))))
