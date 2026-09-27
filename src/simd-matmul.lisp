;;;; simd-matmul.lisp — sb-simd 快速路径（f64 / f32 / i32 / i64）
(in-package :clvt)

;;; ============================================================
;;; 配置
;;; ============================================================

(defvar *simd-matmul-enabled* t)
(defvar *simd-matmul-threshold* 50000)
(defvar *simd-matmul-thread-threshold* 5000000)
(defvar *simd-matmul-thread-count* (- *processor-number* 1))

;;; ============================================================
;;; SIMD 内核宏
;;; ============================================================
;;; 生成分块（64×256×256）+ SIMD 的行范围内核。
;;; 参数：
;;;   vwidth   : SIMD 宽度（4 或 8）
;;;   elem-size: 元素字节大小
;;;   elem-ref : 标量 SAP 读函数（带字节偏移）
;;;   vec-ctor : SIMD 广播构造函数
;;;   vec-ref  : SIMD SAP 读函数（带元素索引，非字节！）
;;;   vec-add  : SIMD 向量加
;;;   vec-mul  : SIMD 向量乘
;;;   vec-fmadd: SIMD FMA（可选，nil 时走 mul+add）

(defmacro define-simd-matmul-rows
    (fn-name vwidth elem-size elem-ref vec-ctor vec-ref vec-add vec-mul
     &optional vec-fmadd)
  `(progn
     (declaim (inline ,fn-name))
     (defun ,fn-name (pa pb pc k n i-start i-end)
       (declare (type sb-sys:system-area-pointer pa pb pc)
                (type fixnum k n i-start i-end)
                (optimize (speed 3) (safety 0) (debug 0)))
       (let ((ii 64) (jj 256) (kk 256))
         (declare (type fixnum ii jj kk))
         (loop for ii0 of-type fixnum from i-start below i-end by ii do
           (let ((ii-end (the fixnum (min i-end (the fixnum (+ ii0 ii))))))
             (declare (type fixnum ii-end))
             (loop for jj0 of-type fixnum from 0 below n by jj do
               (let* ((jj-end (the fixnum (min n (the fixnum (+ jj0 jj)))))
                      (len    (the fixnum (- jj-end jj0)))
                      (len-main (the fixnum (logand len (- ,vwidth)))))
                 (declare (type fixnum jj-end len len-main))
                 (loop for kk0 of-type fixnum from 0 below k by kk do
                   (let ((kk-end (the fixnum (min k (the fixnum (+ kk0 kk))))))
                     (declare (type fixnum kk-end))
                     (loop for i of-type fixnum from ii0 below ii-end do
                       (let ((a-row (the fixnum (* i k)))
                             (c-row (the fixnum (+ (the fixnum (* i n)) jj0))))
                         (declare (type fixnum a-row c-row))
                         (loop for l of-type fixnum from kk0 below kk-end do
                           (let ((a-val (,elem-ref pa (* (+ a-row l) ,elem-size)))
                                 (b-base (the fixnum (+ (the fixnum (* l n)) jj0))))
                             (declare (type fixnum b-base))
                             (let ((a-vec (,vec-ctor a-val))
                                   (b-ptr b-base)
                                   (c-ptr c-row))
                               (declare (type fixnum b-ptr c-ptr))
                               ;; 主循环：vwidth 宽 SIMD
                               (loop for j of-type fixnum from 0 below len-main by ,vwidth do
                                 (let ((c-vec (,vec-ref pc c-ptr))
                                       (b-vec (,vec-ref pb b-ptr)))
                                   (setf (,vec-ref pc c-ptr)
                                         ,(if vec-fmadd
                                              `(,vec-fmadd a-vec b-vec c-vec)
                                              `(,vec-add c-vec
                                                         (,vec-mul a-vec b-vec)))))
                                 (incf c-ptr ,vwidth)
                                 (incf b-ptr ,vwidth))
                               ;; 尾部标量
                               (loop for j of-type fixnum from len-main below len do
                                 (setf (,elem-ref pc (* c-ptr ,elem-size))
                                       (+ (,elem-ref pc (* c-ptr ,elem-size))
                                          (* a-val
                                             (,elem-ref pb (* b-ptr ,elem-size)))))
                                 (incf c-ptr)
                                 (incf b-ptr)))))))))))))))))

;;; ============================================================
;;; 四个类型的内核
;;; ============================================================

;; float64: 4 宽 + FMA
(define-simd-matmul-rows %simd-matmul-f64-rows
  4 8
  sb-sys:sap-ref-double
  sb-simd-avx:f64.4
  sb-simd-avx:f64.4-sap-ref
  sb-simd-avx:f64.4+
  sb-simd-avx:f64.4*
  sb-simd-fma:f64.4-fmadd)

;; float32: 8 宽 + FMA
(define-simd-matmul-rows %simd-matmul-f32-rows
  8 4
  sb-sys:sap-ref-single
  sb-simd-avx:f32.8
  sb-simd-avx:f32.8-sap-ref
  sb-simd-avx:f32.8+
  sb-simd-avx:f32.8*
  sb-simd-fma:f32.8-fmadd)

;; int32: 8 宽，用 mullo（AVX2 的 vpmulld，低 32 位乘法）
(define-simd-matmul-rows %simd-matmul-i32-rows
  8 4
  sb-sys:signed-sap-ref-32
  sb-simd-avx:s32.8
  sb-simd-avx:s32.8-sap-ref
  sb-simd-fma:s32.8+
  sb-simd-fma:s32.8-mullo)

;; int64: 4 宽，用 s64.4-mul（可能是模拟，需实测）
(define-simd-matmul-rows %simd-matmul-i64-rows
  4 8
  sb-sys:signed-sap-ref-64
  sb-simd-avx:s64.4
  sb-simd-avx:s64.4-sap-ref
  sb-simd-fma:s64.4+
  sb-simd-fma:s64.4-mul)

;;; ============================================================
;;; 并行调度
;;; ============================================================

(defun %simd-matmul-parallel (rows-fn pa pb pc m k n)
  (declare (type function rows-fn)
           (type sb-sys:system-area-pointer pa pb pc)
           (type fixnum m k n))
  (if (or (<= *simd-matmul-thread-count* 1)
          (< (* m k n) *simd-matmul-thread-threshold*))
      (funcall rows-fn pa pb pc k n 0 m)
      (let* ((nthr (min *simd-matmul-thread-count* m))
             (rows-per (ceiling m nthr))
             (threads nil))
        (declare (type fixnum nthr rows-per))
        (loop for tid of-type fixnum from 0 below nthr
              do (let* ((i0 (the fixnum (* tid rows-per)))
                        (i1 (the fixnum (min m (+ i0 rows-per)))))
                   (declare (type fixnum i0 i1))
                   (when (< i0 i1)
                     (push (sb-thread:make-thread
                            (lambda () (funcall rows-fn pa pb pc k n i0 i1)))
                           threads))))
        (dolist (th threads) (sb-thread:join-thread th))))
  nil)

;;; ============================================================
;;; 通用 dispatch
;;; ============================================================

(defun %simd-matmul-2d-generic (a b dtype out expected-dtype rows-fn elem-size)
  "各类型的公共 dispatch 逻辑。
   elem-size: 每个元素的字节数（f64/i64=8，f32/i32=4）。"
  (declare (type vt a b)
           (type (or null symbol) dtype)
           (type (or null vt) out)
           (type symbol expected-dtype)
           (type function rows-fn)
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

      (let ((final (cond (out (vt-dtype out))
                         (dtype dtype)
                         (t (vt-promote-type (vt-dtype a) (vt-dtype b))))))
        (unless (eq final expected-dtype)
          (return-from %simd-matmul-2d-generic nil)))

      (when out
        (unless (and (equal (vt-shape out) (list m n))
                     (vt-contiguous-p out))
          (return-from %simd-matmul-2d-generic nil)))

      (let* ((a-c (if (and (eq (vt-dtype a) expected-dtype)
                           (vt-contiguous-p a))
                      a (vt-contiguous (vt-astype a expected-dtype))))
             (b-c (if (and (eq (vt-dtype b) expected-dtype)
                           (vt-contiguous-p b))
                      b (vt-contiguous (vt-astype b expected-dtype))))
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
              (%simd-matmul-parallel rows-fn pa pb pc m k n))))
        output))))

(defun %simd-matmul-2d (a b dtype out)
  "按 dtype 分发。返回 VT 成功，NIL 让 vt-matmul 回退。"
  (with-float-safe
    (or (%simd-matmul-2d-generic a b dtype out :float64
				 #'%simd-matmul-f64-rows 8)
	(%simd-matmul-2d-generic a b dtype out :float32
				 #'%simd-matmul-f32-rows 4)
	(%simd-matmul-2d-generic a b dtype out :int32
				 #'%simd-matmul-i32-rows 4)
	(%simd-matmul-2d-generic a b dtype out :int64
				 #'%simd-matmul-i64-rows 8)
	nil)))

;; *simd-matmul-2d-fn* 在 linalg.lisp 文件中定义，这里注册
(setf *simd-matmul-2d-fn* #'%simd-matmul-2d)

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
    (a b dtype out expected-dtype rows-fn elem-size)
  "批量矩阵乘法核心。
   A: (..., M, K)，B: (..., K, N)。任一侧可缺批维度（numpy 广播）。
   OUT: (..., M, N)。ELEM-SIZE: f64/i64=8，f32/i32=4。"
  (declare (type vt a b)
           (type (or null symbol) dtype)
           (type (or null vt) out)
           (type symbol expected-dtype)
           (type function rows-fn)
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
            (let ((final (cond (out (vt-dtype out))
                               (dtype dtype)
                               (t (vt-promote-type (vt-dtype a)
                                                   (vt-dtype b))))))
              (unless (eq final expected-dtype)
                (return-from core nil)))
            ;; OUT 形状检查
            (when out
              (unless (and (equal (vt-shape out) out-shape)
                           (vt-contiguous-p out))
                (return-from core nil)))
            ;; 连续性 + 类型强制 + 创建 output
            (let* ((a-c (if (and (eq (vt-dtype a) expected-dtype)
                                 (vt-contiguous-p a))
                            a
                            (vt-contiguous (vt-astype a expected-dtype))))
                   (b-c (if (and (eq (vt-dtype b) expected-dtype)
                                 (vt-contiguous-p b))
                            b
                            (vt-contiguous (vt-astype b expected-dtype))))
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
                             (funcall rows-fn
                                      (sb-sys:sap+
                                       a-sap (* bi a-batch-stride elem-size))
                                      (sb-sys:sap+
                                       b-sap (* bi b-batch-stride elem-size))
                                      (sb-sys:sap+
                                       out-sap (* bi out-batch-stride elem-size))
                                      k n 0 m)))
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
                               #'%simd-matmul-f64-rows 8)))

(defun %simd-batched-matmul-f32 (a b dtype out)
  (with-float-safe
    (%simd-batched-matmul-core a b dtype out :float32
                               #'%simd-matmul-f32-rows 4)))

(defun %simd-batched-matmul-i32 (a b dtype out)
  (with-float-safe
    (%simd-batched-matmul-core a b dtype out :int32
                                #'%simd-matmul-i32-rows 4)))

(defun %simd-batched-matmul-i64 (a b dtype out)
  (with-float-safe
    (%simd-batched-matmul-core a b dtype out :int64
                                #'%simd-matmul-i64-rows 8)))

(defun %simd-batched-matmul-dispatch (a b dtype out)
  "顶层分派。失败返回 NIL 让 vt-matmul 回退 einsum。"
  (or (%simd-batched-matmul-f64 a b dtype out)
      (%simd-batched-matmul-f32 a b dtype out)
      (%simd-batched-matmul-i32 a b dtype out)
      (%simd-batched-matmul-i64 a b dtype out)
      nil))

;; 注册到 vt-matmul
(setf *simd-batched-matmul-fn* #'%simd-batched-matmul-dispatch)
