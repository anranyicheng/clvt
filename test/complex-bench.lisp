;;;; complex-bench.lisp — 复数层性能基线（1M 元素）
;;;;
;;;; 用法：sbcl --dynamic-space-size 4096 --non-interactive --load test/complex-bench.lisp
;;;; 输出：每个算子的 clvt 耗时 + 同规模实数耗时（对照）+ 比值。

(require :asdf)
(push #p"./" asdf:*central-registry*)
(push #p"/workspace/clvt/" asdf:*central-registry*)
(handler-bind ((warning #'muffle-warning)) (asdf:load-system :clvt))
(in-package :clvt)

(defparameter *n* 1000000)

(defmacro bench (label form &key (runs 3))
  "跑 RUNS 次取最好（ms）。GC 前置，减少噪声。"
  `(let ((best most-positive-fixnum))
     (dotimes (_ (1+ ,runs))
       (sb-ext:gc :full t)
       (let ((t0 (get-internal-real-time)))
         ,form
         (let ((ms (round (* 1000.0 (- (get-internal-real-time) t0))
                          internal-time-units-per-second)))
           (setf best (min best ms)))))
     (format t "~&~36a ~8d ms~%" ,label best)
     best))

(defmacro bench-pair (label real-form complex-form)
  "同一算子：实数 vs 复数，打印倍率。"
  `(let ((r (bench ,(concatenate 'string label " [f64]") ,real-form))
         (c (bench ,(concatenate 'string label " [c128]") ,complex-form)))
     (format t "    → 复数/实数 = ~,1f x~%" (if (zerop r) 0.0 (/ (float c) r)))
     (values r c)))

(defun main ()
  (format t "~&===== 复数层基线（n=~d）=====~%" *n*)
  (let* ((z1 (vt-+ (vt-zeros (list *n*) :dtype :complex128) 1.0d0))
         (z2 (vt-+ (vt-zeros (list *n*) :dtype :complex128) #C(2.0d0 -1.0d0)))
         (d1 (vt-+ (vt-zeros (list *n*)) 1.0d0))
         (d2 (vt-+ (vt-zeros (list *n*)) 2.0d0))
         (zi (vt-+ (vt-zeros (list *n*) :dtype :complex64) 1.0d0)))
    (bench-pair "add" (vt-+ d1 d2) (vt-+ z1 z2))
    (bench-pair "mul" (vt-* d1 d2) (vt-* z1 z2))
    (bench-pair "div" (vt-/ d1 d2) (vt-/ z1 z2))
    (bench-pair "sqrt" (vt-sqrt d1) (vt-sqrt z1))
    (bench-pair "exp" (vt-exp d1) (vt-exp z1))
    (bench-pair "abs" (vt-abs d1) (vt-abs z1))
    (bench-pair "sum" (vt-sum d1) (vt-sum z1))
    (bench-pair "sum axis=0 (1000x1000)"
                (vt-sum (vt-reshape d1 (list 1000 1000)) :axis 0)
                (vt-sum (vt-reshape z1 (list 1000 1000)) :axis 0))
    (bench-pair "cumsum" (vt-cumsum d1) (vt-cumsum z1))
    (bench-pair "sort" (vt-sort d1) (vt-sort z1))
    (bench-pair "compare <" (vt-< d1 d2) (vt-< z1 z2))
    (bench-pair "isnan" (vt-isnan d1) (vt-isnan z1))
    (format t "~&--- 复数内部：c64 vs c128 ---~%")
    (bench "add c64 (vs c128 上面一行)" (vt-+ zi zi))
    (format t "~&--- 线代 256x256 ---~%")
    (let ((zr (vt-reshape (vt-+ (vt-zeros (list 65536) :dtype :complex128) #C(1.0d0 0.5d0))
                          (list 256 256)))
          (dr (vt-reshape (vt-+ (vt-zeros (list 65536)) 1.5d0) (list 256 256))))
      (bench-pair "matmul 256x256" (vt-matmul dr dr) (vt-matmul zr zr)))
    (format t "~&--- 归约 argmax/median ---~%")
    (bench-pair "argmax" (vt-argmax d1) (vt-argmax z1))
    (bench-pair "median" (vt-median d1) (vt-median z1))))

(main)
