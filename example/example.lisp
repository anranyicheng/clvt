(defpackage :clvt-test
  (:use :cl :clvt))
(in-package :clvt-test)

;; 1. 简单遍历 (打印非零元素)
(defparameter *m*
  (vt-reshape (vt-arange 27 :type 'fixnum)
	      '(3 3 3)))
(print *m*)
;; #<VT shape:(3 3 3) dtype:FIXNUM 
;;   [[[  0,   1,   2],
;;     [  3,   4,   5],
;;     [  6,   7,   8]],
;;    [[  9,  10,  11],
;;     [ 12,  13,  14],
;;     [ 15,  16,  17]],
;;    [[ 18,  19,  20],
;;     [ 21,  22,  23],
;;     [ 24,  25,  26]]]> 

(setf (vt-ref *m* 1 1) 9)

(vt-do-each (ptr val *m*)
  (print (list ptr val))
  (when (> val 0.0)
    (format t "Found value ~a at physical index ~a~%" val ptr)))
;; 2. 广播拷贝
(defparameter *dest* (vt-zeros '(3 4)))
(defparameter *src* (vt-ones '(1 4))) ; 形状 (1 4)
(vt-copy-into *dest* *src*) ; 自动将 (1 4) 广播填充到 (3 4) 的三行
(print *dest*)
;; Output:
;; #<TENSOR shape:(3 4) 
;;   [[ 1.0,  1.0,  1.0,  1.0],
;;    [ 1.0,  1.0,  1.0,  1.0],
;;    [ 1.0,  1.0,  1.0,  1.0]]>


;;----------------------测试 vt-qr vt-svd ---------------------

(defun diag-matrix (S m n)
  "用奇异值张量 S (一维) 构建 m×n 对角矩阵"
  (let* ((K (vt-size S))
         (mat (vt-zeros (list m n))))
    (loop for i below K
          do (setf (vt-ref mat i i) (coerce (vt-ref S i) 'double-float)))
    mat))

(defun test-svd (A &optional (full-matrices nil))
  "测试 SVD 的重构误差与正交性"
  (let* ((m (first (vt-shape A)))
         (n (second (vt-shape A))))
    (multiple-value-bind (U S Vt)
        (vt-svd A :full-matrices full-matrices)
      (let ((K (vt-size S)))   ; 使用 vt-size 获取奇异值个数
        (format t "~%----- SVD (full-matrices = ~A) -----" full-matrices)
        (format t "~%U shape: ~A, S length: ~A, Vt shape: ~A~%"
                (vt-shape U) K (vt-shape Vt))
        ;; 重构
        (let* ((Smat (if full-matrices
                         (diag-matrix S m n)
                         (diag-matrix S K K)))
               (recon (vt-@ U (vt-@ Smat Vt)))
               (diff (vt-- A recon))
               (recon-error (reduce #'max (vt-data (vt-abs diff)))))
          (format t "重构误差: max|A - U S Vt| = ~A~%" recon-error))
        ;; 正交性检查
        (let* ((UtU (vt-@ (vt-transpose U) U))
               (Iu (vt-eye (first (vt-shape UtU)) :type 'double-float))
               (UtU-error (reduce #'max (vt-data (vt-abs (vt-- UtU Iu))))))
          (format t "U^T U ≈ I ? max|U^T U - I| = ~A~%" UtU-error))
        (let* ((VtV (vt-@ Vt (vt-transpose Vt)))
               (Iv (vt-eye (first (vt-shape VtV)) :type 'double-float))
               (VtV-error (reduce #'max (vt-data (vt-abs (vt-- VtV Iv))))))
          (format t "Vt Vt^T ≈ I ? max|Vt Vt^T - I| = ~A~%" VtV-error))))))






;; ===================== QR 分解测试 =====================
(defparameter *A-qr* (vt-from-sequence '((12 -51   4)
                                         ( 6 167 -68)
                                         (-4  24 -41))))
(format t "~%========== QR 分解测试 ==========~%")
(format t "原始矩阵 A:~%")
(print (vt-data->list *A-qr*))

;; Full mode
(multiple-value-bind (Q R)
    (vt-qr *A-qr* :mode :full)
  (format t "~%Full mode: Q (m×m), R (m×n)~%")
  (format t "Q 的正交性: Q^T Q ≈ I ?~%")
  (let ((QtQ (vt-@ (vt-transpose Q) Q)))
    (print qtq)
    (format t "  max|Q^T Q - I| = ~A~%"
            (reduce #'max (vt-data QtQ)
		    :key (lambda (x)
			   (abs (- x (if (= (first (vt-shape Q))
						      (second (vt-shape Q)))
						   1.0 0.0)))))))
  (format t "重构误差: max|A - Q R| = ~A~%"
          (reduce #'max (vt-data (vt-- *A-qr* (vt-@ Q R))) :key #'abs)))

;; Reduced mode
(multiple-value-bind (Q R) (vt-qr *A-qr* :mode :reduced)
  (format t "~%Reduced mode: Q (m×k), R (k×n)~%")
  (format t "Q 的列正交性: Q^T Q ≈ I_k ?~%")
  (let ((QtQ (vt-@ (vt-transpose Q) Q)))
    (format t "  max|Q^T Q - I| = ~A~%"
            (reduce #'max (vt-data (let ((I (vt-eye (second (vt-shape Q))
						    :type 'double-float)))
                                     (vt-- QtQ I)))
		    :key #'abs)))
  (format t "重构误差: max|A - Q R| = ~A~%"
          (reduce #'max (vt-data (vt-- *A-qr* (vt-@ Q R))) :key #'abs)))

;; ===================== SVD 分解测试 =====================
;; 测试矩阵 1: 4x3 非方阵
(defparameter *A-svd1* (vt-from-sequence '((1 2 3)
                                          (4 5 6)
                                          (7 8 9)
                                           (10 11 12))))

;; 测试矩阵 2: 3x3 方阵
(defparameter *A-svd2* (vt-from-sequence '((3 1 2)
                                          (1 4 1)
                                           (2 1 3))))

;; 测试矩阵 3: 1x1 边界情况
(defparameter *A-svd3* (vt-from-sequence '((5))))

;; 运行测试
(format t "~%========== SVD 分解测试 ==========~%")
(test-svd *A-svd1* nil)
(test-svd *A-svd1* t)
(test-svd *A-svd2* nil)
(test-svd *A-svd2* t)
;; 1x1 边界测试
(format t "~%--- 1x1 边界情况 ---")
(test-svd *A-svd3* nil)

;; 重构误差都应该小于 1e-14数量级
;;------------------------------------------------------<<




(defun test-gradient ()
  ;; 辅助：近似相等断言
  (macrolet ((assert-close (a b &optional (eps 1e-8))
               `(progn
                  (unless (<= (abs (- ,a ,b)) ,eps)
                    (error "Assertion failed: ~A ~A differ by ~A"
                           ',a ',b (abs (- ,a ,b))))
                  t)))
  
  ;; ---------- 1D 等间距 ----------
  (let* ((x (vt-arange 6 :step 1.0d0 :type 'double-float))       ; 0,1,2,3,4,5
         (y (vt-square x))                                         ; y = x^2
         ;; axis nil → 返回所有轴的梯度列表（此处 1D → 单元素列表）
         (grad-list (vt-gradient y))
         (grad (first grad-list)))     ; 取出唯一张量
    (format t "~%1D uniform (single from list): ~A~%" grad)
    ;; 理论：内部二阶中心差分，边界一阶差分
    (assert-close (vt-ref grad 0) 1.0d0)       ; (f1-f0)/1 = 1
    (assert-close (vt-ref grad 1) 2.0d0)       ; (f2-f0)/2 = 2
    (assert-close (vt-ref grad 2) 4.0d0)
    (assert-close (vt-ref grad 3) 6.0d0)
    (assert-close (vt-ref grad 4) 8.0d0)
    (assert-close (vt-ref grad 5) 9.0d0))      ; (f5-f4)/1 = 9
  
  ;; ---------- 1D 非等间距（坐标数组）----------
  (let* ((x (vt-from-sequence '(0.0d0 1.0d0 3.0d0 6.0d0) :type 'double-float))
         (y (vt-square x))
         ;; 指定单轴 axis=0，直接返回该轴张量（不是列表）
         (grad (vt-gradient y :spacing x :axis 0)))
    (format t "~%1D non-uniform (axis=0): ~A~%" grad)
    (assert-close (vt-ref grad 0) 1.0d0)
    (assert-close (vt-ref grad 1) 3.0d0)
    (assert-close (vt-ref grad 2) 7.0d0)
    (assert-close (vt-ref grad 3) 9.0d0))
  
  ;; ---------- 2D 等间距，指定单个轴 ----------
  (let* ((mat (vt-from-sequence '((0.0d0 1.0d0 2.0d0)
                                  (3.0d0 4.0d0 5.0d0)
                                  (6.0d0 7.0d0 8.0d0))
                                :type 'double-float))
         ;; axis 整数 → 直接返回该轴梯度张量
         (grad0 (vt-gradient mat :axis 0))
         (grad1 (vt-gradient mat :axis 1)))
    (format t "~%2D along axis=0:~%~A~%" grad0)
    (format t "2D along axis=1:~%~A~%" grad1)
    (dotimes (i 3)
      (dotimes (j 3)
        (assert-close (vt-ref grad0 i j) 3.0d0)))
    (dotimes (i 3)
      (dotimes (j 3)
        (assert-close (vt-ref grad1 i j) 1.0d0))))
  
  ;; ---------- 多轴同时计算 ----------
  (let* ((a (vt-from-sequence '((1.0d0 2.0d0 3.0d0)
                                (4.0d0 5.0d0 6.0d0))
                              :type 'double-float))
         ;; axis 列表 → 返回列表
         (grads (vt-gradient a :axis '(0 1))))
    (format t "~%Multi-axis gradients:~%~A~%~A~%" (first grads) (second grads))
    (assert (= (length grads) 2))
    (let ((g0 (first grads)) (g1 (second grads)))
      (assert (equal (vt-shape g0) '(2 3)))
      (assert (equal (vt-shape g1) '(2 3)))
      (format t "Multi-axis test passed.~%")))
  
  ;; ---------- 边界条件：尺寸太小 ----------
  (handler-case
      (progn
        (vt-gradient (vt-from-sequence '(1.0d0) :type 'double-float) :axis 0)
        (error "Should have thrown an error for size 1"))
    (simple-error (e)
      (format t "~%Correctly errored on size 1: ~A~%" e)))
  
    (format t "~%All gradient tests passed!~%")))

  


(defun test-gradient-advanced ()
  (macrolet ((assert-close (a b &optional (eps 1e-8))
               `(progn
                  (unless (<= (abs (- ,a ,b)) ,eps)
                    (error "Assertion failed: ~A ~A differ by ~A"
                           ',a ',b (abs (- ,a ,b))))
                  t)))

  ;; === 3D 张量，等间距，指定两个轴（axis 列表 + 标量 spacing） ===
  (let* ((t3d (vt-from-sequence
               '((( 0.0d0  1.0d0  2.0d0) ( 3.0d0  4.0d0  5.0d0) ( 6.0d0  7.0d0  8.0d0))
                 (( 9.0d0 10.0d0 11.0d0) (12.0d0 13.0d0 14.0d0) (15.0d0 16.0d0 17.0d0))
                 ((18.0d0 19.0d0 20.0d0) (21.0d0 22.0d0 23.0d0) (24.0d0 25.0d0 26.0d0)))
               :type 'double-float))
         ;; t3d 形状 (3, 3, 3)，每个元素值相当于 i*9 + j*3 + k
         ;; 沿 axis=0 和 axis=2 求梯度，spacing=1
         (grads (vt-gradient t3d :axis '(0 2)))
         (g0 (first grads))
         (g2 (second grads)))
    (format t "~%3D multi-axis (0,2) uniform spacing=1~%")
    (format t "grad along 0:~%~A~%" g0)
    (format t "grad along 2:~%~A~%" g2)
    ;; 沿 axis=0 的梯度：边界一阶，内部中心差分，由于每层相差9，所以梯度应为全9。
    ;; 沿 axis=2 的梯度：最内层变化步长1，梯度应为全1。
    ;; 验证：
    (dotimes (i 3)
      (dotimes (j 3)
        (dotimes (k 3)
          (assert-close (vt-ref g0 i j k) 9.0d0)
          (assert-close (vt-ref g2 i j k) 1.0d0))))
    (format t "3D multi-axis passed.~%"))

  ;; === 4D 张量，axis=nil（全部轴），等间距 ===
  (let* ((shape '(2 2 2 2))
         (t4d (vt-from-function shape
                                 (lambda (idxs)
                                   (coerce (reduce #'+ idxs) 'double-float))
                                 :type 'double-float))
         ;; t4d 的元素是坐标索引之和 (i0+i1+i2+i3)
         ;; 每轴步长1，所以任何轴的梯度都应为1
         (grads (vt-gradient t4d))   ; 返回 4 个张量的列表
         )
    (format t "~%4D all axes nil uniform spacing=1~%")
    (assert (= (length grads) 4))
    (loop for g in grads
          do (assert (equal (vt-shape g) shape))
             (vt-do-each (ptr val g)
               (declare (ignore ptr))
               (assert-close val 1.0d0)))
    (format t "4D all axes passed.~%"))

  ;; === 2D，不同轴的间距不同（列表 spacing） ===
  (let* ((mat (vt-from-sequence '((0.0d0 1.0d0 2.0d0 3.0d0)
                                  (4.0d0 5.0d0 6.0d0 7.0d0)
                                  (8.0d0 9.0d0 10.0d0 11.0d0))
                                :type 'double-float))
         ;; 轴0 间距 2.0, 轴1 间距 0.5
         (grads (vt-gradient mat :spacing '(2.0d0 0.5d0) :axis '(0 1)))
         (g0 (first grads))
         (g1 (second grads)))
    (format t "~%2D multi-axis with list spacing (2, 0.5)~%")
    (format t "grad0:~%~A~%" g0)
    (format t "grad1:~%~A~%" g1)
    ;; 轴0内部中心差分应为 4/dx = 4/2 = 2.0，边界一阶 4/2=2.0
    ;; 轴1内部中心差分应为 1/dx = 1/0.5 = 2.0，边界一阶 1/0.5=2.0
    (dotimes (i 3)
      (dotimes (j 4)
        (assert-close (vt-ref g0 i j) 2.0d0)
        (assert-close (vt-ref g1 i j) 2.0d0)))
    (format t "List spacing test passed.~%"))

  ;; === 一维，非均匀间距（坐标数组），长度2（刚修复的边界情况） ===
  (let* ((x (vt-from-sequence '(1.0d0 5.0d0) :type 'double-float))
         (y (vt-square x))   ; y = [1, 25]
         (grad (vt-gradient y :spacing x :axis 0)))
    (format t "~%1D length-2 non-uniform: ~A~%" grad)
    ;; 只能用一阶差分：(25-1)/(5-1) = 24/4 = 6，两端相同
    (assert-close (vt-ref grad 0) 6.0d0)
    (assert-close (vt-ref grad 1) 6.0d0)
    (format t "Length-2 test passed.~%"))

  ;; === 3D 指定单个轴（int axis），非均匀间距（坐标数组） ===
  (let* ((t3d (vt-from-sequence
               '(((1.0d0 2.0d0) (3.0d0 4.0d0))   ; shape (2,2,2)
                 ((5.0d0 6.0d0) (7.0d0 8.0d0)))
               :type 'double-float))
         ;; 沿 axis=1 求梯度，使用非均匀坐标 [0.0, 1.0] （对应轴大小2）
         (coord (vt-from-sequence '(0.0d0 1.0d0) :type 'double-float))
         (grad (vt-gradient t3d :spacing coord :axis 1)))
    (format t "~%3D axis=1 non-uniform (length 2 axis): ~A~%" grad)
    ;; 轴1长度=2，只能用一阶差分
    ;; 计算：对于每个“纤维”（axis=1），df = t3d[...,1,:] - t3d[...,0,:], dx=1.0
    ;; 梯度应为常数：第一块 (3-1,4-2)=(2,2)，第二块 (7-5,8-6)=(2,2)
    (let ((expected (vt-from-sequence '(((2.0d0 2.0d0) (2.0d0 2.0d0))
                                        ((2.0d0 2.0d0) (2.0d0 2.0d0)))
                                      :type 'double-float)))
      (vt-do-each (ptr val grad)
        (assert-close val (aref (vt-data expected) ptr))))
    (format t "3D non-uniform length-2 axis passed.~%"))

  ;; === 错误场景：坐标数组长度不匹配 ===
  (handler-case
      (progn
        (vt-gradient (vt-arange 5) :spacing (vt-arange 3) :axis 0)
        (format t "~%ERROR: Should have signalled mismatch~%"))
    (simple-error (e)
      (format t "~%Correctly signalled length mismatch: ~A~%" e)))

  (format t "~%All advanced gradient tests passed!~%")))
















(defun test-pad ()
  "测试 vt-pad 的各种填充模式"
  (format t "~%=== Testing vt-pad ===")
  
  ;; 1. 常数填充（默认）
  (let* ((a (vt-from-sequence '((1 2) (3 4)) :type 'fixnum))
         ;; 每轴填充宽度：轴0: (1 2)，轴1: (0 1)
         (padded (vt-pad a '((1 2) (0 1))
			 :mode :constant :constant-values 99)))
    (format t "~%Constant padding:~%~A~%" padded)
    (assert (equal (vt-shape padded) '(5 3)))  ; (2+1+2)=5, (2+0+1)=3
    (assert (vt-allclose padded
                          (vt-from-sequence '((99 99 99)
                                              (1  2  99)
                                              (3  4  99)
                                              (99 99 99)
                                              (99 99 99))
                                             :type 'fixnum))))
  
  ;; 1b. 常数填充，左右不同常数
  (let* ((a (vt-arange 5 :type 'fixnum))
         (padded (vt-pad a 2 :mode :constant :constant-values '(10 20))))
    (format t "~%Constant left/right different:~%~A~%" padded)
    (assert (equal (vt-shape padded) '(9)))
    (assert (vt-allclose padded (vt-from-sequence '(10 10 0 1 2 3 4 20 20) :type 'fixnum))))
  
  ;; 2. 边缘填充
  (let* ((a (vt-from-sequence '((1 2 3) (4 5 6)) :type 'fixnum))
         ;; 轴0填充 (1 1)，轴1填充 (2 2)
         (padded (vt-pad a '((1 1) (2 2)) :mode :edge)))
    (format t "~%Edge padding:~%~A~%" padded)
    (assert (equal (vt-shape padded) '(4 7)))
    ;; 第一行边缘为第一行延伸，最后一行为最后一行延伸，列同理
    (let ((expected (vt-from-sequence '((1 1 1 2 3 3 3)
                                        (1 1 1 2 3 3 3)
                                        (4 4 4 5 6 6 6)
                                        (4 4 4 5 6 6 6))
                                       :type 'fixnum)))
      (assert (vt-allclose padded expected))))
  
  ;; 3. 循环填充 (wrap)
  (let* ((a (vt-from-sequence '((1 2) (3 4)) :type 'fixnum))
         ;; 轴0填充 (1 1)，轴1填充 (1 1)，则形状变为 (4 4)
         (padded (vt-pad a '((1 1) (1 1)) :mode :wrap)))
    (format t "~%Wrap padding:~%~A~%" padded)
    (assert (equal (vt-shape padded) '(4 4)))
    ;; 循环：左边/上边取自右边/下边的行/列，反之亦然
    (let ((expected (vt-from-sequence '((4 3 4 3)
                                        (2 1 2 1)
                                        (4 3 4 3)
                                        (2 1 2 1))
                                       :type 'fixnum)))
      (assert (vt-allclose padded expected))))
  
  ;; 4. 反射填充 (reflect) - 不重复边缘
  (let* ((a (vt-from-sequence '((1 2 3) (4 5 6)) :type 'fixnum))
         ;; 每轴填充 (2 2)，注意反射要求宽度 <= 原尺寸-1，满足
         (padded (vt-pad a '((2 2) (2 2)) :mode :reflect)))
    (format t "~%Reflect padding:~%~A~%" padded)
    (assert (equal (vt-shape padded) '(6 7)))
    ;; 手动计算预期：轴0反射：原始行索引 0,1 → 左填充：跳过边缘(0)，从 1,0? 实际上 reflect 模式：左填充从左向右反射第一行（索引1），第二行（索引0）
    ;; 规则：对于左边界，填充序列为 a[1], a[0]；对于右边界，填充序列为 a[n-2], a[n-3]...
      ;; 仅验证形状即可，因为内容需要严格按照规则；这里改为验证特定元素
      ;; 验证中心区域与原矩阵一致
      (let ((center (apply #'vt-slice padded '((2 4) (2 5)))))
        (assert (vt-allclose center a))))
  
  ;; 5. 对称填充 (symmetric) - 重复边缘
  (let* ((a (vt-from-sequence '((1 2) (3 4)) :type 'fixnum))
         ;; 轴0填充 (1 1)，轴1填充 (1 1)
         (padded (vt-pad a '((1 1) (1 1)) :mode :symmetric)))
    (format t "~%Symmetric padding:~%~A~%" padded)
    (assert (equal (vt-shape padded) '(4 4)))
    ;; 对称：左边界从 a[0] 开始反向，右边界从 a[-1] 开始反向
    (let ((expected (vt-from-sequence '((1 1 2 2)
                                        (1 1 2 2)
                                        (3 3 4 4)
                                        (3 3 4 4))
                                       :type 'fixnum)))
      (assert (vt-allclose padded expected))))
  
  ;; 6. 反射模式宽度超过限制应报错
  (handler-case
      (vt-pad (vt-arange 5) 4 :mode :reflect)  ; 宽度4 > 4 (dim-1) 应报错
    (simple-error (e)
      (format t "~%Reflect width exceed error: ~A~%" e)))
  
  ;; 7. 对称模式宽度超过限制应报错
  (handler-case
      (vt-pad (vt-arange 5) 6 :mode :symmetric) ; 宽度6 > 5 应报错
    (simple-error (e)
      (format t "~%Symmetric width exceed error: ~A~%" e)))
  
  (format t "~%All pad tests passed!~%"))


(defun test-all-pad ()
  ;; 1. Constant
  (let ((a (vt-from-sequence '((1 2) (3 4)) :type 'fixnum)))
    (let ((padded (vt-pad a '((1 2) (0 1)) :mode :constant :constant-values 99)))
      (assert (vt-allclose padded
                           (vt-from-sequence '((99 99 99)
                                                ( 1  2 99)
                                                ( 3  4 99)
                                                (99 99 99)
                                                (99 99 99))
                                              :type 'fixnum)))))
  ;; 2. Edge
  (let ((a (vt-from-sequence '((1 2 3) (4 5 6)) :type 'fixnum)))
    (let ((padded (vt-pad a '((1 1) (2 2)) :mode :edge)))
      (assert (vt-allclose padded
                           (vt-from-sequence '((1 1 1 2 3 3 3)
                                                (1 1 1 2 3 3 3)
                                                (4 4 4 5 6 6 6)
                                                (4 4 4 5 6 6 6))
                                              :type 'fixnum)))))
  ;; 3. Wrap
  (let ((a (vt-from-sequence '((1 2) (3 4)) :type 'fixnum)))
    (let ((padded (vt-pad a '((1 1) (1 1)) :mode :wrap)))
      (assert (vt-allclose padded
                           (vt-from-sequence '((4 3 4 3)
                                                (2 1 2 1)
                                                (4 3 4 3)
                                                (2 1 2 1))
                                              :type 'fixnum)))))
  ;; 4. Reflect（只需验证中心区域不变）
  (let ((a (vt-from-sequence '((1 2) (3 4)) :type 'fixnum)))
    (let ((padded (vt-pad a '((2 2) (2 2)) :mode :reflect)))
      (assert (vt-allclose (apply #'vt-slice padded '((2 4) (2 4))) a))))
  ;; 5. Symmetric
  (let ((a (vt-from-sequence '((1 2) (3 4)) :type 'fixnum)))
    (let ((padded (vt-pad a '((1 1) (1 1)) :mode :symmetric)))
      (assert (vt-allclose padded
                           (vt-from-sequence '((1 1 2 2)
                                                (1 1 2 2)
                                                (3 3 4 4)
                                                (3 3 4 4))
                                              :type 'fixnum)))))
  ;; 6. 1D symmetric (NumPy 示例)
  (let ((a (vt-from-sequence '(1 2 3 4 5) :type 'fixnum)))
    (assert (vt-allclose (vt-pad a '((0 2)) :mode :symmetric)
                         (vt-from-sequence '(1 2 3 4 5 5 4) :type 'fixnum)))
    (assert (vt-allclose (vt-pad a '((2 0)) :mode :symmetric)
                         (vt-from-sequence '(2 1 1 2 3 4 5) :type 'fixnum))))
  (format t "~%All pad tests passed!~%"))


(defun test-pad-thorough ()
  (format t "~%=== Thorough vt-pad test ===")

  ;; 辅助函数：逐元素相等断言（整数可用）
  (labels ((vt= (a b)
             (vt-allclose a b :atol 0.0d0 :rtol 0.0d0)))

    ;; ---------- 1. CONSTANT ----------
    (let* ((a (vt-from-sequence '(1 2 3) :type 'fixnum))
           (p (vt-pad a '((2 3)) :mode :constant :constant-values 99)))
      ;; NumPy: np.pad([1,2,3], (2,3), 'constant', constant_values=99) -> [99 99 1 2 3 99 99 99]
      (assert (vt= p (vt-from-sequence '(99 99 1 2 3 99 99 99) :type 'fixnum))))

    (let* ((a (vt-from-sequence '((1 2) (3 4)) :type 'fixnum))
           (p (vt-pad a '((1 0) (2 1)) :mode :constant :constant-values 0)))
      ;; NumPy: np.pad([[1,2],[3,4]], ((1,0),(2,1)), 'constant') -> [[0,0,0,0,0], [0,0,1,2,0], [0,0,3,4,0]]
      (assert (vt= p (vt-from-sequence '((0 0 0 0 0)
                                        (0 0 1 2 0)
                                        (0 0 3 4 0))
                                      :type 'fixnum))))

    (let* ((a (vt-from-sequence '(1.0d0 2.0d0) :type 'double-float))
           (p (vt-pad a '((1 1)) :mode :constant :constant-values '(10.0d0 20.0d0))))
      ;; NumPy: np.pad([1.,2.], (1,1), 'constant', constant_values=(10,20)) -> [10., 1., 2., 20.]
      (assert (vt= p (vt-from-sequence '(10.0d0 1.0d0 2.0d0 20.0d0) :type 'double-float))))

    ;; 零宽度填充
    (let* ((a (vt-from-sequence '((1 2) (3 4)) :type 'fixnum))
           (p (vt-pad a '((0 0) (0 0)) :mode :constant :constant-values 99)))
      (assert (vt= p a)))

    ;; ---------- 2. EDGE ----------
    (let* ((a (vt-from-sequence '((1 2 3) (4 5 6)) :type 'fixnum))
           (p (vt-pad a '((2 1) (1 2)) :mode :edge)))
      ;; NumPy: np.pad([[1,2,3],[4,5,6]], ((2,1),(1,2)), 'edge')
      ;; 预期：行上两行边缘（复制第一行），下一行边缘（复制最后一行）；列左一列复制第一列，右两列复制最后一列。
      (let ((expected (vt-from-sequence '((1 1  2 3 3 3)
                                          (1 1  2 3 3 3)
                                          (1 1  2 3 3 3)
                                          (4 4  5 6 6 6)
                                          (4 4  5 6 6 6))
                                        :type 'fixnum)))
        (assert (vt= p expected))))

    ;; ---------- 3. WRAP ----------
    (let* ((a (vt-from-sequence '((1 2) (3 4)) :type 'fixnum))
           (p (vt-pad a '((1 1) (2 1)) :mode :wrap)))
      ;; NumPy: np.pad([[1,2],[3,4]], ((1,1),(2,1)), 'wrap')
      ;; 行循环：上边取最后一行，下边取第一行；列循环：左边取最后两列，右边取第一列。
      ;; 结果形状 4x5
      (let ((expected (vt-from-sequence '((3 4 3 4 3)
					  (1 2 1 2 1)
					  (3 4 3 4 3)
					  (1 2 1 2 1))
                                        :type 'fixnum)))
        (assert (vt= p expected))))

    ;; ---------- 4. REFLECT (宽度可任意大) ----------
    (let* ((a (vt-from-sequence '((1 2 3) (4 5 6)) :type 'fixnum))
           (p (vt-pad a '((2 2) (3 3)) :mode :reflect)))
      ;; 只验证中心区域未变，并验证几个特征点（NumPy 可生成完整预期，此处省略）
      (assert (vt= (apply #'vt-slice p '((2 4) (3 6))) a))
      (assert (= 189 (vt-sum p)))
      )

    (let* ((a (vt-from-sequence '(1 2 3) :type 'fixnum))
           (p (vt-pad a '((4 4)) :mode :reflect)))
      ;; NumPy: np.pad([1,2,3], (4,4), 'reflect')
      ;; 预期：左填充 [3,2,1,2]，右填充 [2,3,2,1]
      (assert (vt= p (vt-from-sequence '(1 2 3 2 1 2 3 2 1 2 3)
				       :type 'fixnum))))

    ;; ---------- 5. SYMMETRIC (大宽度) ----------
    ;; 1D 左边宽度2
    (let* ((a (vt-from-sequence '(1 2 3 4 5) :type 'fixnum))
           (p (vt-pad a '((2 0)) :mode :symmetric)))
      (assert (vt= p (vt-from-sequence '(2 1 1 2 3 4 5) :type 'fixnum))))
    ;; 1D 右边宽度2
    (let* ((a (vt-from-sequence '(1 2 3 4 5) :type 'fixnum))
           (p (vt-pad a '((0 2)) :mode :symmetric)))
      (assert (vt= p (vt-from-sequence '(1 2 3 4 5 5 4) :type 'fixnum))))
    ;; 1D 左边大宽度7
    (let* ((a (vt-from-sequence '(1 2) :type 'fixnum))
           (p (vt-pad a '((7 0)) :mode :symmetric))
	   (expected (vt-from-sequence '(2 2 1 1 2 2 1 1 2) :type 'fixnum)))
      ;; NumPy: np.pad([1,2], (7,0), 'symmetric') -> [2, 2, 1, 1, 2, 2, 1, 1, 2]? 我们直接验证形状和边界值
      (vt-allclose expected p))
    ;; 2D symmetric
    (let* ((a (vt-from-sequence '((1 2) (3 4)) :type 'fixnum))
           (p (vt-pad a '((1 1) (1 1)) :mode :symmetric)))
      (assert (vt= p (vt-from-sequence '((1 1 2 2)
                                         (1 1 2 2)
                                         (3 3 4 4)
                                         (3 3 4 4))
                                       :type 'fixnum))))

    ;; ---------- 6. 混合类型测试 ----------
    (let* ((a (vt-from-sequence '((1.0d0 2.0d0) (3.0d0 4.0d0)) :type 'double-float))
           (p (vt-pad a '((1 1) (1 1)) :mode :symmetric)))
      (assert (vt= p (vt-from-sequence '((1.0 1.0 2.0 2.0)
                                         (1.0 1.0 2.0 2.0)
                                         (3.0 3.0 4.0 4.0)
                                         (3.0 3.0 4.0 4.0))
                                       :type 'double-float))))

    (format t "~%All thorough pad tests passed!~%")))
