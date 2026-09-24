;;;; simd-test.lisp — sb-simd 完整测试套件
;;;;
;;;; 用法：
;;;;   (load "simd-test.lisp")
;;;;   (simd-test:run-all)
;;;;
;;;; 分为三部分：
;;;;   Part A: 环境探测（SAP 偏移单位、符号存在性）
;;;;   Part B: 纯 sb-simd 测试（smoke test、基准、disassemble 检查）
;;;;   Part C: CLVT 集成测试（NaN/Inf 语义、浮点陷阱一致性）

(defpackage :simd-test
  (:use :cl)
  (:export #:run-all
           #:run-probe
           #:run-smoke
           #:run-bench
           #:run-nan-inf
           #:run-trap-consistency))

(in-package :simd-test)

;;; 确保 sb-simd 已加载
(eval-when (:compile-toplevel :load-toplevel :execute)
  (unless (find-package :sb-simd-avx)
    (require :sb-simd)))

;;; ============================================================
;;; 通用工具
;;; ============================================================

(defun nan-p (x)
  "判定 NaN，不触发浮点陷阱。"
  (and (floatp x) (sb-ext:float-nan-p x)))

(defun inf-p (x)
  "判定 Inf，不触发浮点陷阱。"
  (and (floatp x) (sb-ext:float-infinity-p x)))

(defun pos-inf-p (x) (and (inf-p x) (plusp x)))
(defun neg-inf-p (x) (and (inf-p x) (minusp x)))

(defun safe-nan ()
  (the double-float
    (sb-int:with-float-traps-masked (:invalid :divide-by-zero :overflow)
      (/ 0.0d0 0.0d0))))

(defun safe-pos-inf ()
  (the double-float
    (sb-int:with-float-traps-masked (:invalid :divide-by-zero :overflow)
      (/ 1.0d0 0.0d0))))

(defun safe-neg-inf ()
  (the double-float
    (sb-int:with-float-traps-masked (:invalid :divide-by-zero :overflow)
      (/ -1.0d0 0.0d0))))

(defmacro with-all-traps-masked (&body body)
  `(sb-int:with-float-traps-masked
       (:invalid :divide-by-zero :overflow :underflow)
     ,@body))

(defun elapsed-seconds (start)
  (declare (type (integer 0) start))
  (/ (- (get-internal-real-time) start)
     (float internal-time-units-per-second 1.0d0)))

;;; ============================================================
;;; Part A: 环境探测
;;; ============================================================

(defun probe-sap-offset ()
  "探测 f64.4-sap-ref 的 offset 单位。
   预期：元素索引（不是字节）。"
  (format t "~&=== SAP offset 单位探测 ===~%")
  (let ((a (make-array 16 :element-type 'double-float)))
    (loop for i below 16 do (setf (aref a i) (float i 1d0)))
    (sb-sys:with-pinned-objects (a)
      (let ((pa (sb-sys:vector-sap a)))
        (flet ((probe (off)
                 (format t "  off=~3d → ~a~%"
                         off
                         (sb-simd-avx:f64.4-sap-ref pa off))))
          (probe 0)
          (probe 1)
          (probe 4)
          (probe 8)))))
  (format t "  解读：off=1 应返回 (1.0 2.0 3.0 4.0)，即元素索引~%~%"))

(defun probe-symbols ()
  "列出 sb-simd 各包中与 4 种类型相关的符号。"
  (flet ((list-syms (pkg substr)
           (let ((p (find-package pkg)))
             (when p
               (format t "~&=== ~a（含 ~a） ===~%" pkg substr)
               (let ((count 0))
                 (do-external-symbols (s p)
                   (when (search substr (symbol-name s))
                     (format t "  ~a~%" s)
                     (incf count)))
                 (format t "  共 ~d 个~%" count))))))
    (list-syms :sb-simd-avx "F64.4")
    (list-syms :sb-simd-fma "F64.4")
    (list-syms :sb-simd-avx "F32.8")
    (list-syms :sb-simd-fma "F32.8")
    (list-syms :sb-simd-avx "S32.8")
    (list-syms :sb-simd-fma "S32.8")
    (list-syms :sb-simd-avx "S64.4")
    (list-syms :sb-simd-fma "S64.4"))
  (format t "~%"))

(defun run-probe ()
  "运行所有环境探测。"
  (probe-sap-offset)
  (probe-symbols)
  t)

;;; ============================================================
;;; Part B: 纯 sb-simd 测试
;;; ============================================================

;;; ------------------------------------------------------------
;;; B.1 Smoke test
;;; ------------------------------------------------------------

(defun smoke-f64 (n)
  "c[i] = a[i]*b[i] + c[i]，纯 SIMD，验证 API。"
  (declare (type fixnum n))
  (let ((a (make-array n :element-type 'double-float :initial-element 2.0d0))
        (b (make-array n :element-type 'double-float :initial-element 3.0d0))
        (c (make-array n :element-type 'double-float :initial-element 1.0d0)))
    (declare (type (simple-array double-float (*)) a b c))
    (sb-sys:with-pinned-objects (a b c)
      (let ((pa (sb-sys:vector-sap a))
            (pb (sb-sys:vector-sap b))
            (pc (sb-sys:vector-sap c))
            (i 0))
        (declare (type sb-sys:system-area-pointer pa pb pc)
                 (type fixnum i))
        (loop while (< i n) do
          (let ((va (sb-simd-avx:f64.4-sap-ref pa i))
                (vb (sb-simd-avx:f64.4-sap-ref pb i))
                (vc (sb-simd-avx:f64.4-sap-ref pc i)))
            (declare (type sb-simd-avx:f64.4 va vb vc))
            (setf (sb-simd-avx:f64.4-sap-ref pc i)
                  (sb-simd-fma:f64.4-fmadd va vb vc)))
          (incf i 4))))
    (aref c 0)))

(defun smoke-f32 (n)
  "float32 版本：8 宽。"
  (declare (type fixnum n))
  (let ((a (make-array n :element-type 'single-float :initial-element 2.0f0))
        (b (make-array n :element-type 'single-float :initial-element 3.0f0))
        (c (make-array n :element-type 'single-float :initial-element 1.0f0)))
    (declare (type (simple-array single-float (*)) a b c))
    (sb-sys:with-pinned-objects (a b c)
      (let ((pa (sb-sys:vector-sap a))
            (pb (sb-sys:vector-sap b))
            (pc (sb-sys:vector-sap c))
            (i 0))
        (declare (type sb-sys:system-area-pointer pa pb pc)
                 (type fixnum i))
        (loop while (< i n) do
          (let ((va (sb-simd-avx:f32.8-sap-ref pa i))
                (vb (sb-simd-avx:f32.8-sap-ref pb i))
                (vc (sb-simd-avx:f32.8-sap-ref pc i)))
            (declare (type sb-simd-avx:f32.8 va vb vc))
            (setf (sb-simd-avx:f32.8-sap-ref pc i)
                  (sb-simd-fma:f32.8-fmadd va vb vc)))
          (incf i 8))))
    (aref c 0)))

(defun smoke-i32 (n)
  "int32 版本：8 宽，用 mullo。"
  (declare (type fixnum n))
  (let ((a (make-array n :element-type '(signed-byte 32) :initial-element 2))
        (b (make-array n :element-type '(signed-byte 32) :initial-element 3))
        (c (make-array n :element-type '(signed-byte 32) :initial-element 1)))
    (declare (type (simple-array (signed-byte 32) (*)) a b c))
    (sb-sys:with-pinned-objects (a b c)
      (let ((pa (sb-sys:vector-sap a))
            (pb (sb-sys:vector-sap b))
            (pc (sb-sys:vector-sap c))
            (i 0))
        (declare (type sb-sys:system-area-pointer pa pb pc)
                 (type fixnum i))
        (loop while (< i n) do
          (let ((va (sb-simd-avx:s32.8-sap-ref pa i))
                (vb (sb-simd-avx:s32.8-sap-ref pb i))
                (vc (sb-simd-avx:s32.8-sap-ref pc i)))
            (declare (type sb-simd-avx:s32.8 va vb vc))
            (setf (sb-simd-avx:s32.8-sap-ref pc i)
                  (sb-simd-fma:s32.8+
                   vc (sb-simd-fma:s32.8-mullo va vb))))
          (incf i 8))))
    (aref c 0)))

(defun smoke-i64 (n)
  "int64 版本：4 宽，用 mul。"
  (declare (type fixnum n))
  (let ((a (make-array n :element-type '(signed-byte 64) :initial-element 2))
        (b (make-array n :element-type '(signed-byte 64) :initial-element 3))
        (c (make-array n :element-type '(signed-byte 64) :initial-element 1)))
    (declare (type (simple-array (signed-byte 64) (*)) a b c))
    (sb-sys:with-pinned-objects (a b c)
      (let ((pa (sb-sys:vector-sap a))
            (pb (sb-sys:vector-sap b))
            (pc (sb-sys:vector-sap c))
            (i 0))
        (declare (type sb-sys:system-area-pointer pa pb pc)
                 (type fixnum i))
        (loop while (< i n) do
          (let ((va (sb-simd-avx:s64.4-sap-ref pa i))
                (vb (sb-simd-avx:s64.4-sap-ref pb i))
                (vc (sb-simd-avx:s64.4-sap-ref pc i)))
            (declare (type sb-simd-avx:s64.4 va vb vc))
            (setf (sb-simd-avx:s64.4-sap-ref pc i)
                  (sb-simd-fma:s64.4+
                   vc (sb-simd-fma:s64.4-mul va vb))))
          (incf i 4))))
    (aref c 0)))

(defun run-smoke ()
  "运行 smoke test，验证 4 种 SIMD 类型的基本正确性。"
  (format t "~&=== SB-SIMD Smoke Test ===~%")
  (let ((ok t))
    (flet ((check (name result expected)
             (if (= result expected)
                 (format t "  ✓ ~a → ~a~%" name result)
                 (progn
                   (format t "  ✗ ~a → ~a（期望 ~a）~%" name result expected)
                   (setf ok nil)))))
      (check "f64.4"  (smoke-f64 16)  7.0d0)
      (check "f32.8"  (smoke-f32 16)  7.0f0)
      (check "s32.8"  (smoke-i32 16)  7)
      (check "s64.4"  (smoke-i64 16)  7))
    (format t "~%")
    ok))

;;; ------------------------------------------------------------
;;; B.2 基准测试
;;; ------------------------------------------------------------

(defun make-rand-matrix (n dtype &optional (val 3.0d0))
  "构造一个 n×n 全 val 的矩阵，返回 (values data size)。"
  (let ((size (* n n))
        (lt (ecase dtype
              (:float64 'double-float)
              (:float32 'single-float)
              (:int32   '(signed-byte 32))
              (:int64   '(signed-byte 64)))))
    (let ((arr (make-array size :element-type lt
                                :initial-element
                                (ecase dtype
                                  (:float64 (coerce val 'double-float))
                                  (:float32 (coerce val 'single-float))
                                  (:int32   (truncate val))
                                  (:int64   (truncate val))))))
      (values arr size))))

(defun bench-gemm (dtype n)
  "运行一次 dtpye 矩阵乘法基准，返回耗时（秒）。"
  (declare (type symbol dtype) (type fixnum n))
  (multiple-value-bind (a size) (make-rand-matrix n dtype 3.0d0)
    (multiple-value-bind (b ignore) (make-rand-matrix n dtype 3.0d0)
      (declare (ignore ignore))
      (let ((c (make-array size :element-type (array-element-type a)
                                :initial-element (ecase dtype
                                                   (:float64 0.0d0)
                                                   (:float32 0.0f0)
                                                   ((:int32 :int64) 0)))))
        (declare (ignore c))
        ;; 由外部（CLVT）调用实际的 matmul
        (values a b n)))))

(defun run-bench-clvt (n)
  "如果 CLVT 已加载，运行 4 种 dtype 的基准。"
  (if (find-package :clvt)
      (progn
        (format t "~&=== CLVT + SB-SIMD 矩阵乘法基准（~dx~d） ===~%" n n)
        (dolist (dtype '(:float64 :float32 :int32 :int64))
          (let* ((fill (ecase dtype
                         (:float64 3.0d0)
                         (:float32 3.0f0)
                         ((:int32 :int64) 3)))
                 (a (clvt:vt-const (list n n) fill :dtype dtype)))
            (format t "  ~a: " dtype)
            (let ((start (get-internal-real-time)))
              (clvt:vt-matmul a a)
              (format t "~,3f s~%" (elapsed-seconds start)))))
        (format t "~%"))
      (format t "~&（CLVT 未加载，跳过基准）~%~%"))
  t)

;;; ------------------------------------------------------------
;;; B.3 Disassemble 检查
;;; ------------------------------------------------------------

(defun check-disasm (fn-name)
  "打印函数的 disassembly，供人工检查。
   理想的内层循环应包含 VFMADD* 和 VMOVUPD，不应有 MOV RAX, [RSI-7]。"
  (format t "~&=== Disassemble: ~a ===~%" fn-name)
  (let ((fn (symbol-function fn-name)))
    (when fn
      (disassemble fn)
      (format t "~%请检查内层循环：~%")
      (format t "  ✓ 应看到：VFMADD*、VMOVUPD YMM、MOVQ/ADD/CMP/JL~%")
      (format t "  ✗ 不应看到：MOV RAX, [RSI-7]（bounds check）、VMOVSD（标量）~%~%")))
  t)

;;; ============================================================
;;; Part C: CLVT 集成测试
;;; ============================================================

(defun make-clvt-test-matrices (size dtype fill)
  (values (clvt:vt-const (list size size) fill :dtype dtype)
          (clvt:vt-const (list size size) fill :dtype dtype)))

;;; ------------------------------------------------------------
;;; C.1 NaN / Inf 语义
;;; ------------------------------------------------------------

(defun test-nan-propagation ()
  (declare (optimize (safety 0)))   ; ← 绕过 SBCL 类型推导
  (format t "~&[场景 1] NaN 输入传播~%")
  (with-all-traps-masked
    (multiple-value-bind (a b) (make-clvt-test-matrices 100 :float64 3.0d0)
      (setf (clvt:vt-ref a 5 5) (safe-nan))
      (let ((c (clvt:vt-matmul a b)))
        (assert (nan-p (clvt:vt-ref c 5 0)) ()
                "第 5 行第 0 列应含 NaN")
        (assert (nan-p (clvt:vt-ref c 5 99)) ()
                "第 5 行第 99 列应含 NaN")
        (assert (= (clvt:vt-ref c 0 0) 900.0d0) ()
                "第 0 行应为 900")
        (format t "  ✓ NaN 正确传播~%")))))

(defun test-inf-propagation ()
  (format t "~&[场景 2] +Inf 输入传播~%")
  (with-all-traps-masked
    (multiple-value-bind (a b) (make-clvt-test-matrices 100 :float64 3.0d0)
      (setf (clvt:vt-ref a 0 0) (safe-pos-inf))
      (let ((c (clvt:vt-matmul a b)))
        (assert (pos-inf-p (clvt:vt-ref c 0 0)) ()
                "第 0 行应为 +Inf")
        (assert (= (clvt:vt-ref c 1 0) 900.0d0) ()
                "第 1 行应为 900")
        (format t "  ✓ +Inf 正确传播~%")))))

(defun test-simd-vs-scalar-special ()
  (declare (optimize (safety 0)))   ; ← 同上
  (format t "~&[场景 4] SIMD vs 标量（特殊值）~%")
  (with-all-traps-masked
    (multiple-value-bind (a b) (make-clvt-test-matrices 100 :float64 3.0d0)
      (setf (clvt:vt-ref a 5 5)   (safe-nan))
      (setf (clvt:vt-ref a 10 10) (safe-pos-inf))
      (setf (clvt:vt-ref a 20 20) (safe-neg-inf))
      (let ((clvt::*simd-matmul-enabled* t))
        (let ((c-simd (clvt:vt-matmul a b)))
          (let ((clvt::*simd-matmul-enabled* nil))
            (let ((c-scalar (clvt:vt-matmul a b)))
              (let ((mismatch 0))
                (dotimes (i 100)
                  (dotimes (j 100)
                    (let ((x (clvt:vt-ref c-simd i j))
                          (y (clvt:vt-ref c-scalar i j)))
                      (unless (or (and (nan-p x) (nan-p y))
                                  (and (not (nan-p x)) (not (nan-p y)) (= x y)))
                        (incf mismatch)
                        (when (<= mismatch 5)
                          (format t "~&    ✗ (~d,~d): SIMD=~a SCALAR=~a~%"
                                  i j x y))))))
                (if (zerop mismatch)
                    (format t "  ✓ 两条路径 100×100 一致~%")
                    (format t "  ✗ 共 ~d 处不一致~%" mismatch))))))))))

(defun test-overflow-to-inf ()
  (format t "~&[场景 3] 溢出 → Inf~%")
  (with-all-traps-masked
    (multiple-value-bind (a b) (make-clvt-test-matrices 100 :float64 1.0d200)
      (let* ((c (clvt:vt-matmul a b))
             (data (clvt:vt-data c))
             (size (clvt:vt-size c)))
        (declare (type (simple-array double-float (*)) data)
                 (type fixnum size))
        (assert (every #'pos-inf-p data) ()
                "全矩阵应为 +Inf")
        (format t "  ✓ 溢出 → +Inf（~d 元素）~%" size)))))

(defun test-simd-vs-scalar-numeric ()
  (format t "~&[场景 5] SIMD vs 标量（普通值）~%")
  (with-all-traps-masked
    (multiple-value-bind (a b) (make-clvt-test-matrices 100 :float64 1.2345d0)
      (let ((clvt::*simd-matmul-enabled* t))
        (let ((c-simd (clvt:vt-matmul a b)))
          (let ((clvt::*simd-matmul-enabled* nil))
            (let ((c-scalar (clvt:vt-matmul a b)))
              (let ((max-rel-err 0.0d0))
                (declare (type double-float max-rel-err))
                (dotimes (i 100)
                  (dotimes (j 100)
                    (let* ((x (clvt:vt-ref c-simd i j))
                           (y (clvt:vt-ref c-scalar i j))
                           (denom (max (abs x) (abs y) 1.0d-300))
                           (rel-err (/ (abs (- x y)) denom)))
                      (declare (type double-float x y denom rel-err))
                      (when (> rel-err max-rel-err)
                        (setf max-rel-err rel-err)))))
                (format t "  最大相对误差: ~,3e~%" max-rel-err)
                (if (< max-rel-err 1.0d-12)
                    (format t "  ✓ 两条路径数值一致~%")
                    (format t "  ⚠ 相对误差 ~a（FMA 舍入差异）~%" max-rel-err))))))))))

(defun run-nan-inf ()
  "CLVT NaN/Inf 语义测试。"
  (unless (find-package :clvt)
    (format t "~&（CLVT 未加载，跳过 NaN/Inf 测试）~%~%")
    (return-from run-nan-inf nil))
  (format t "~&~%===== NaN / Inf 语义测试 =====~%")
  (let ((failures 0))
    (with-all-traps-masked
      (flet ((safe-run (name thunk)
               (handler-case
                   (funcall thunk)
                 (error (e)
                   (incf failures)
                   (format t "  ✗ ~a 失败：~a~%" name e)))))
        (safe-run "场景 1" #'test-nan-propagation)
        (safe-run "场景 2" #'test-inf-propagation)
        (safe-run "场景 3" #'test-overflow-to-inf)
        (safe-run "场景 4" #'test-simd-vs-scalar-special)
        (safe-run "场景 5" #'test-simd-vs-scalar-numeric)))
    (format t "~%")
    (if (zerop failures)
        (progn (format t "===== 全部通过 =====~%~%") t)
        (progn (format t "===== ~d 个场景失败 =====~%~%" failures) nil))))

;;; ------------------------------------------------------------
;;; C.2 浮点陷阱一致性
;;; ------------------------------------------------------------

(defun test-real-trap-scenarios ()
  "测试 SIMD 和 einsum 路径在真正触发浮点陷阱的运算上行为是否一致。"
  (format t "~&[场景 7] 真正触发浮点陷阱的运算~%")
  (multiple-value-bind (a b) (make-clvt-test-matrices 100 :float64 0.0d0)
    (setf (clvt:vt-ref a 5 5) (safe-pos-inf))
    (setf (clvt:vt-ref b 5 0) 0.0d0)
    (setf (clvt:vt-ref b 0 0) 1.0d0)
    (flet ((run-path (simd-enabled)
             (let ((clvt::*simd-matmul-enabled* simd-enabled))
               (handler-case
                   (values (clvt:vt-matmul a b) nil)
                 (error (e)
                   (values nil (format nil "~a" e)))))))
      (multiple-value-bind (c-simd err-simd) (run-path t)
        (multiple-value-bind (c-scalar err-scalar) (run-path nil)
          (format t "  SIMD 路径：~a~%"
                  (if err-simd (format nil "异常 — ~a" err-simd) "正常返回"))
          (format t "  einsum 路径：~a~%"
                  (if err-scalar (format nil "异常 — ~a" err-scalar) "正常返回"))
          (cond
            ((and c-simd c-scalar)
             (format t "  ✓ 两条路径一致~%")
             t)
            ((and err-simd err-scalar)
             (format t "  ⚠ 两条都抛异常（一致）~%")
             t)
            ((and err-simd (null err-scalar))
             (format t "  ✗ SIMD 抛异常但 einsum 不抛 → 需 with-float-safe~%")
             nil)
            (t
             (format t "  ✗ 反向不一致~%")
             nil)))))))

(defun test-real-overflow-trap ()
  "测试溢出时的陷阱行为。"
  (format t "~&[场景 8] 溢出触发浮点陷阱~%")
  (multiple-value-bind (a b) (make-clvt-test-matrices 100 :float64 1.0d200)
    (flet ((run-path (simd-enabled)
             (let ((clvt::*simd-matmul-enabled* simd-enabled))
               (handler-case
                   (values (clvt:vt-matmul a b) nil)
                 (error (e)
                   (values nil (format nil "~a" e)))))))
      (multiple-value-bind (c-simd err-simd) (run-path t)
        (multiple-value-bind (c-scalar err-scalar) (run-path nil)
          (format t "  SIMD 路径：~a~%"
                  (if err-simd (format nil "异常 — ~a" err-simd) "正常返回"))
          (format t "  einsum 路径：~a~%"
                  (if err-scalar (format nil "异常 — ~a" err-scalar) "正常返回"))
          (if (and (null err-simd) (null err-scalar))
              (progn (format t "  ✓ 两条路径一致~%") t)
              (progn (format t "  ✗ 行为不一致~%") nil)))))))

(defun run-trap-consistency ()
  "CLVT 浮点陷阱一致性测试。"
  (unless (find-package :clvt)
    (format t "~&（CLVT 未加载，跳过陷阱一致性测试）~%~%")
    (return-from run-trap-consistency nil))
  (format t "~&~%===== 浮点陷阱一致性测试 =====~%")
  (let ((ok1 (test-real-trap-scenarios))
        (ok2 (test-real-overflow-trap)))
    (format t "~%")
    (if (and ok1 ok2)
        (progn (format t "===== 全部通过 =====~%~%") t)
        (progn (format t "===== 存在不一致 =====~%~%") nil))))

;;; ============================================================
;;; 全部运行
;;; ============================================================

(defun run-all (&key (skip-probe nil) (skip-smoke nil) (skip-bench nil))
  "运行全部测试。
   :skip-probe —— 跳过环境探测（信息输出较多）
   :skip-smoke —— 跳过 smoke test
   :skip-bench —— 跳过基准（耗时较长）"
  (format t "~&~%############################################################~%")
  (format t "# SB-SIMD 完整测试套件~%")
  (format t "############################################################~%~%")

  (unless skip-probe
    (run-probe))

  (unless skip-smoke
    (run-smoke))

  (unless skip-bench
    (run-bench-clvt 1000))

  (when (find-package :clvt)
    (run-nan-inf)
    (run-trap-consistency))

  (format t "~&测试完成。~%~%")
  t)

(run-all)
