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
