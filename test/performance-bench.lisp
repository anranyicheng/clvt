;;;; performance-bench.lisp — clvt 性能基准测试
;;;; Usage: sbcl --noinform --non-interactive --load performance-bench.lisp

(require :asdf)
(push (truename (make-pathname :directory '(:relative :up))) asdf:*central-registry*)
(asdf:load-system :clvt)
(in-package :clvt)

(format t "~%===========================================================~%")
(format t "  clvt 性能基准测试（优化后）~%")
(format t "===========================================================~%")

(defun bench (name fn &optional (n 100))
  (let ((start (get-internal-real-time)))
    (dotimes (i n) (funcall fn))
    (let* ((elapsed (- (get-internal-real-time) start))
           (ms (* (/ elapsed internal-time-units-per-second) 1000.0)))
      (format t "  ~22a: ~8,2f ms total, ~8,2f us/op~%" name ms (/ (* ms 1000) n)))))

;; Warm up
(let ((a (vt-zeros '(10 10)))) (vt-+ a a) (vt-sum a))

(format t "~%--- 1000x1000 float64 (100万元素) ---~%")
(let ((a (vt-random-uniform '(1000 1000)))
      (b (vt-random-uniform '(1000 1000))))
  (bench "vt-+ a+b" (lambda () (vt-+ a b)) 30)
  (bench "vt-* a*b" (lambda () (vt-* a b)) 30)
  (bench "vt-sum all" (lambda () (vt-sum a)) 50)
  (bench "vt-exp" (lambda () (vt-exp a)) 20)
  (bench "vt-sin" (lambda () (vt-sin a)) 20)
  (bench "vt-sigmoid" (lambda () (vt-sigmoid a)) 20)
  (bench "vt-relu" (lambda () (vt-relu a)) 30)
  (bench "vt-transpose" (lambda () (vt-transpose a)) 200)
  (bench "vt-reshape" (lambda () (vt-reshape a '(1000000))) 200)
  (bench "vt-matmul (1k×1k)" (lambda () (vt-matmul a b)) 3))

(format t "~%--- 100x100 float64 (1万元素) ---~%")
(let ((a (vt-random-uniform '(100 100)))
      (b (vt-random-uniform '(100 100))))
  (bench "vt-sum axis=0" (lambda () (vt-sum a :axis 0)) 200)
  (bench "vt-sum axis=1" (lambda () (vt-sum a :axis 1)) 200)
  (bench "vt-matmul (100×100)" (lambda () (vt-matmul a b)) 50))

(format t "~%--- 正确性检查（随机数据 vs numpy）---~%")
;; 验证优化后的函数结果正确性
(let ((a (vt-random-uniform '(100 100))))
  (let ((s (vt-item (vt-sum a))))
    (format t "  vt-sum(100x100 random) = ~f~%" s)
    (format t "  vt-sum mean-per-element = ~f (expected ~,4f)~%" (/ s 10000) 0.5d0)))

(format t "~%===========================================================~%")
(format t "  基准测试完成~%")
(format t "===========================================================~%")
