;;;; SIMD 批量矩阵乘法的完整测试
(require :asdf)
#+quicklisp (ql:quickload :clvt)
(asdf:load-system :clvt)
(in-package :clvt)

(defun test-simd-batched-matmul ()
  "SIMD 批量矩阵乘法的完整测试。全部使用 ASSERT 断言。"
  (format t "~%=== test-simd-batched-matmul ===~%")

  (macrolet
      ((assert-shape (tensor expected &optional label)
         "断言 TENSOR 形状等于 EXPECTED（字面量列表或变量列表均可）。"
         `(let ((got (vt-shape ,tensor))
                (exp (list ,@expected)))     ; ← 关键修复
            (assert (equal got exp)
                    (got exp)
                    "~@[~a: ~]形状错误，期望 ~a，实际 ~a"
                    ,label exp got)))
       (assert-close (a b tol &optional label)
         "断言 A 与 B 逐元素相近（max|Δ| < TOL），返回 max|Δ|。"
         `(let ((max-d (coerce
                        (vt-item (vt-amax (vt-abs (vt-- ,a ,b))))
                        'double-float)))
            (assert (< max-d ,tol)
                    (max-d)
                    "~@[~a: ~]数值不一致，max|Δ| = ~e（阈值 ~e）"
                    ,label max-d ,tol)
            max-d))
       (with-simd-disabled (&body body)
         "在动态绑定中关闭批量 SIMD 快速路径。"
         `(let ((clvt::*simd-batched-matmul-fn* nil))
            ,@body)))

    ;; ============================================================
    ;; [1/5] 基础正确性
    ;; ============================================================
    (format t "~&  [1/5] 基础正确性...~%")
    (let* ((a1 '((1.0d0 2.0d0 3.0d0)
                 (4.0d0 5.0d0 6.0d0)))
           (a2 '((0.5d0 1.5d0 2.5d0)
                 (3.5d0 4.5d0 5.5d0)))
           (b1 '((1.0d0 0.0d0)
                 (0.0d0 1.0d0)
                 (1.0d0 1.0d0)))
           (b2 '((2.0d0 1.0d0)
                 (1.0d0 2.0d0)
                 (1.0d0 1.0d0)))
           (a-3d (vt-concatenate
                  0
                  (vt-reshape (vt-from-sequence a1 :dtype :float64) '(1 2 3))
                  (vt-reshape (vt-from-sequence a2 :dtype :float64) '(1 2 3))))
           (b-3d (vt-concatenate
                  0
                  (vt-reshape (vt-from-sequence b1 :dtype :float64) '(1 3 2))
                  (vt-reshape (vt-from-sequence b2 :dtype :float64) '(1 3 2)))))
      (let ((c (vt-matmul a-3d b-3d)))
        (assert-shape c (2 2 2) "基础批量 matmul 输出")
        (let ((c0 (vt-to-list (vt-slice c '(0))))
              (c1 (vt-to-list (vt-slice c '(1)))))
          (assert (equal c0 '((4.0d0 5.0d0) (10.0d0 11.0d0)))
                  (c0)
                  "batch 0 错误：得到 ~a，期望 ((4 5) (10 11))" c0)
          (assert (equal c1 '((5.0d0 6.0d0) (17.0d0 18.0d0)))
                  (c1)
                  "batch 1 错误：得到 ~a，期望 ((5 6) (17 18))" c1))))

    ;; ============================================================
    ;; [2/5] einsum 对拍
    ;; ============================================================
    (format t "~&  [2/5] einsum 对拍...~%")
    (dolist (spec '((2 4 3 5)
                    (4 8 6 10)
                    (8 16 12 10)
                    (16 32 24 20)))
      (destructuring-bind (bs m k n) spec
        (let* ((a (vt-random-normal (list bs m k)))
               (b (vt-random-normal (list bs k n)))
               (c-simd   (vt-matmul a b))
               (c-einsum (with-simd-disabled (vt-matmul a b))))
          (assert-shape c-simd (bs m n)
                        (format nil "对拍 ~a" spec))
          (assert-close c-simd c-einsum 1.0d-10
                        (format nil "对拍 ~a" spec)))))

    ;; ============================================================
    ;; [3/5] 广播语义
    ;; ============================================================
    (format t "~&  [3/5] 广播语义...~%")
    ;; 3a. A=(B,M,K), B=(K,N) → C=(B,M,N)
    (let* ((a (vt-random-normal '(8 64 32)))
           (b (vt-random-normal '(32 16)))
           (c-simd (vt-matmul a b))
           (c-manual (vt-zeros '(8 64 16))))
      (assert-shape c-simd (8 64 16) "A 批 B 广播")
      (dotimes (i 8)
        (setf (vt-slice c-manual (list i))
              (vt-matmul (vt-contiguous (vt-slice a (list i))) b)))
      (assert-close c-simd c-manual 1.0d-10 "广播 vs 逐 batch 手动"))
    ;; 3b. A=(M,K), B=(B,K,N) → C=(B,M,N)
    (let ((c (vt-matmul (vt-random-normal '(4 5))
                        (vt-random-normal '(2 5 6)))))
      (assert-shape c (2 4 6) "B 批 A 广播"))

    ;; ============================================================
    ;; [4/5] 边界情况
    ;; ============================================================
    (format t "~&  [4/5] 边界情况...~%")
    (assert-shape (vt-matmul (vt-random-normal '(1 4 3))
                             (vt-random-normal '(1 3 2)))
                  (1 4 2) "batch=1")
    (assert-shape (vt-matmul (vt-random-normal '(2 3 4 5))
                             (vt-random-normal '(2 3 5 6)))
                  (2 3 4 6) "4D batch")
    (assert-shape (vt-matmul (vt-random-normal '(4 5))
                             (vt-random-normal '(5 6)))
                  (4 6) "2D @ 2D")
    (let* ((a (vt-random-normal '(2 3 4 5)))
           (b (vt-random-normal '(2 3 5 6)))
           (c-simd   (vt-matmul a b))
           (c-einsum (with-simd-disabled (vt-matmul a b))))
      (assert-close c-simd c-einsum 1.0d-10 "4D 对拍"))

    ;; ============================================================
    ;; [5/5] 禁用 SIMD 回退
    ;; ============================================================
    (format t "~&  [5/5] 禁用 SIMD 回退...~%")
    (with-simd-disabled
	(let* ((a (vt-random-normal '(4 8 6)))
               (b (vt-random-normal '(4 6 10)))
               (c (vt-matmul a b)))
          (assert-shape c (4 8 10) "禁用 SIMD 时输出")
          (assert (every (lambda (x)
                           (and (numberp x) (= x x)))
			 (vt-to-list (vt-flatten c)))
                  ()
                  "禁用 SIMD 时输出含 NaN 或非数值元素")))

    (format t "~&✅ test-simd-batched-matmul 全部通过~%")
    t))

(test-simd-batched-matmul)
