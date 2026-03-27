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
