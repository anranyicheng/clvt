(ql:quickload :clvt)
(in-package :clvt)

(defparameter *failures* 0)
(defparameter *checks* 0)

(defmacro check (desc expected actual &key (test #'equalp))
  `(progn
     (incf *checks*)
     (let ((desc ,desc) (exp ,expected) (act ,actual))
       (if (funcall ,test exp act)
           (format t "PASS: ~a~%" desc)
           (progn
             (incf *failures*)
             (format t "FAIL: ~a~%  expected: ~s~%  actual:   ~s~%"
                     desc exp act))))))

(defun nan-p (x)
  (and (floatp x)
       (sb-int:with-float-traps-masked
           (:invalid :divide-by-zero :overflow :underflow)
         (not (= x x)))))


(format t "===== 1. vt-reduce 未初始化累加器 =====~%")
(defvar *junk* nil)
(defun make-junk ()
  (setf *junk*
        (loop for k below 30
              collect (let ((a (make-array 100000
                                           :element-type 'double-float)))
                        (fill a (coerce (+ 1d17 k) 'double-float))
                        a))))
(defun drop-junk () (setf *junk* nil) (sb-ext:gc :full t))
(defun test-uninit-reduce ()
  (make-junk) (drop-junk)
  (let* ((m (vt-reshape (vt-arange 100 :dtype :float64) '(10 10)))
         (r (vt-reduce m '(0) 0.0d0 #'+)))
    (vt-to-list r)))

(check "vt-reduce axis=0 sum after GC junk = 每列 450..540"
       '(450.0d0 460.0d0 470.0d0 480.0d0 490.0d0
         500.0d0 510.0d0 520.0d0 530.0d0 540.0d0)
       (test-uninit-reduce)
       :test (lambda (e a)
               (and (listp a) (= (length a) 10)
                    (every (lambda (x y) (and (floatp x) (= x y))) e a))))

(format t "~%===== 2. vt-reduce 全局 min 的 NaN 语义 =====~%")
(let* ((a (vt-from-sequence (list 1.0d0 (clvt::vt-get-nan :float64) 3.0d0)))
       (rmin (vt-reduce a nil most-positive-double-float #'min))
       (rmax (vt-reduce a nil most-negative-double-float #'max)))
  (check "vt-reduce #'max 含 NaN → NaN" t (nan-p (vt-item rmax)))
  (check "vt-reduce #'min 含 NaN → NaN" t (nan-p (vt-item rmin)))
  (check "vt-amin 含 NaN → NaN (对照)" t (nan-p (vt-item (vt-amin a))))
  (check "vt-amax 含 NaN → NaN (对照)" t (nan-p (vt-item (vt-amax a)))))

(let* ((a (vt-from-sequence (list 1.0d0 (clvt::vt-get-nan :float64) 3.0d0)))
       (rmin-inf (vt-reduce a nil clvt::+vt-dfloat-pos-inf+ #'min))
       (rmax-inf (vt-reduce a nil clvt::+vt-dfloat-neg-inf+ #'max)))
  (check "快路径1: init=+inf 的 #'min 含 NaN → NaN" t (nan-p (vt-item rmin-inf)))
  (check "快路径1: init=-inf 的 #'max 含 NaN → NaN" t (nan-p (vt-item rmax-inf))))


(format t "~%===== 3. vt-map :out 与输入内存重叠 =====~%")
(let* ((base (vt-arange 10 :dtype :float64))
       (a (vt-slice base '(0 8)))
       (o (vt-slice base '(2 10)))
       (expected (vt-to-list (vt-+ a 1.0d0)))
       (_r (vt-+ a 1.0d0 :out o))
       (actual (vt-to-list o)))
  (declare (ignore _r))
  (check "vt-+ :out=重叠视图 (numpy 会先快照)" expected actual))

(format t "~%===== 4. matmul/einsum :out 别名输入 =====~%")
(let* ((a (vt-reshape (vt-arange 16 :dtype :float64) '(4 4)))
       (b (vt-ones '(4 4) :dtype :float64))
       (r (vt-matmul a b :out a)))
  (check "vt-matmul a b :out a → 行和 = (24 88 152 216) (NumPy 对照)"
         '(24.0d0 88.0d0 152.0d0 216.0d0)
         (map 'list (lambda (row) (apply #'+ row)) (vt-to-list r))))

(format t "~%===== 5. reduce 家族对 int8/uint8 支持 =====~%")
(handler-case
    (let* ((v (vt-from-sequence '(1 2 3) :dtype :int8))
           (r (vt-sum v)))
      (check "vt-sum int8 张量可用 (numpy 支持)"
             6 (vt-item r) :test #'=))
  (error (e)
    (incf *checks*) (incf *failures*)
    (format t "FAIL: vt-sum int8 抛错: ~a~%" e)))

(handler-case
    (let* ((v (vt-from-sequence '(1 2 3) :dtype :uint8))
           (r (vt-amax v)))
      (check "vt-amax uint8 张量可用"
             3 (vt-item r) :test #'=))
  (error (e)
    (incf *checks*) (incf *failures*)
    (format t "FAIL: vt-amax uint8 抛错: ~a~%" e)))

(format t "~%===== 6. int32 求和溢出提升到 int64 =====~%")
(let ((v (vt-from-sequence '(2000000000 2000000000) :dtype :int32)))
  (check "vt-sum int32 溢出应得 4000000000 (numpy: int64)"
         4000000000
         (vt-item (vt-sum v))
         :test #'=))

(format t "~%===== 7. vt-take 标量视图索引 offset =====~%")
(let* ((idx-base (vt-from-sequence '(5 7 9) :dtype :int64))
       (idx-scalar-view (vt-slice idx-base '(1)))    ; offset=1, 值=7
       (src (vt-from-sequence '(10 20 30 40 50 60 70 80 90 100)))
       (r (vt-take src idx-scalar-view)))
  (check "vt-take src (标量视图 offset=1) → src[7]=80 (NumPy 对照)"
         80 (vt-item r) :test #'=))

(format t "~%===== 8. vt-relu NaN 语义 =====~%")
(let* ((v6 (vt-from-sequence (list (clvt::vt-get-nan :float64)
                                   -1.0d0 2.0d0 4.0d0 5.0d0 6.0d0)))
       (vv (vt-reshape v6 '(2 3)))
       (vw (vt-transpose vv))
       (r1 (vt-to-list (vt-relu vv)))
       (r2 (vt-to-list (vt-relu vw))))
  (check "vt-relu 连续 NaN→NaN" t (nan-p (first (first r1))))
  (check "vt-relu 非连续(回退) NaN→NaN" t (nan-p (first (first r2)))))


(format t "~%===== 9. vt-clip vs vt-clip-tensor =====~%")
(let* ((v (vt-from-sequence (list (clvt::vt-get-nan :float64) 5.0d0)))
       (rc (vt-clip v 0.0d0 1.0d0))
       (rct (vt-clip-tensor v 0.0d0 1.0d0)))
  (check "vt-clip NaN → NaN" t (nan-p (first (vt-to-list rc))))
  (check "vt-clip-tensor NaN → NaN" t (nan-p (first (vt-to-list rct)))))

(let* ((v (vt-from-sequence '(1.0d0 2.0d0 3.0d0)))
       (rc (vt-to-list (vt-clip v 2.0d0 1.0d0)))
       (rct (vt-to-list (vt-clip-tensor v 2.0d0 1.0d0))))
  (format t "  vt-clip(min>max)         → ~s~%" rc)
  (format t "  vt-clip-tensor(min>max)  → ~s~%" rct))


(format t "~%===== 10. ensure-vt 列表 dtype 推断 =====~%")
(handler-case
    (let ((r (vt-to-list (vt-bit-and '(3 5 6) 1))))
      (check "vt-bit-and '(3 5 6) 1 → (1 1 0)" '(1 1 0) r))
  (error (e)
    (incf *checks*) (incf *failures*)
    (format t "FAIL: vt-bit-and 列表参数抛错: ~a~%" e)))


(format t "~%===== 11. vt-geomspace dtype =====~%")
(let ((r (vt-geomspace 1 1000 4 :dtype :int64)))
  (check "vt-geomspace :int64 → dtype 保持 :int64"
         :int64 (vt-dtype r))
  (check "vt-geomspace :int64 → 值 (1 10 100 1000)"
         '(1 10 100 1000) (vt-to-list r)))


(format t "~%===== 12. vt-tile reps 含 0 =====~%")
(handler-case
    (let* ((a (vt-from-sequence '(1 2)))
           (r (vt-tile a '(0 2))))
      (check "vt-tile '(0 2) → shape (0 4) (NumPy 对照)"
             '(0 4) (vt-shape r)))
  (error (e)
    (incf *checks*) (incf *failures*)
    (format t "FAIL: vt-tile reps=0 抛错: ~a~%" e)))

(format t "~%===== 13. vt-extract 标量条件 =====~%")
(handler-case
    (let* ((a (vt-from-sequence '(1 2 3 4)))
           (r (vt-extract 1 a)))
      (check "vt-extract 标量条件 → 全部元素" '(1 2 3 4) (vt-to-list r)))
  (error (e)
    (incf *checks*) (incf *failures*)
    (format t "FAIL: vt-extract 标量条件抛错: ~a~%" e)))

(format t "~%===== 14. vt-put 空 values =====~%")
(handler-case
    (progn
      (vt-put (vt-from-sequence '(1 2 3)) '(0) '())
      (incf *checks*)
      (format t "PASS: 空 values 不崩溃~%"))
  (division-by-zero (e)
    (incf *checks*) (incf *failures*)
    (format t "FAIL: vt-put 空 values 除零: ~a~%" e)))

(format t "~%===== 15. quantile q 越界校验 =====~%")
(handler-case
    (progn
      (vt-percentile (vt-from-sequence '(1 2 3 4 5)) 200)
      (incf *checks*) (incf *failures*)
      (format t "FAIL: percentile 200 未抛错~%"))
  (error (e)
    (incf *checks*)
    (format t "PASS: percentile 200 抛错: ~a~%" e)))

(handler-case
    (progn
      (vt-quantile (vt-from-sequence '(1 2 3 4 5)) 1.5)
      (incf *checks*) (incf *failures*)
      (format t "FAIL: quantile 1.5 未抛错~%"))
  (error (e)
    (incf *checks*)
    (format t "PASS: quantile 1.5 抛错: ~a~%" e)))

(format t "~%===== 16. vt-roll shift=0 别名 =====~%")
(let* ((a (vt-from-sequence '(1 2 3)))
       (r (vt-roll a 0 :axis 0)))
  (incf *checks*)
  (if (eq (vt-data r) (vt-data a))
      (format t "  NOTE(非致命): vt-roll 0 返回原对象别名~%")
      (format t "PASS: 返回拷贝~%")))

(format t "~%===== 17. vt-random int dtype =====~%")
(let ((r (vt-random '(5) :dtype :int64)))
  (format t "  vt-random dtype :int64 → ~s (静默全 0 陷阱)~%"
          (vt-to-list r))
  (incf *checks*))

(format t "~%===== 18. :out 为广播视图 =====~%")
(handler-case
    (let* ((base (vt-from-sequence '(1 2 3)))
           (exp (vt-expand-dims base 0))
           (r (vt-+ exp 1.0d0 :out exp)))
      (declare (ignore r))
      (incf *checks*)
      (format t "  NOTE: :out 广播视图未报错 → base 被改为 ~s~%"
              (vt-to-list base)))
  (error (e)
    (incf *checks*)
    (format t "  :out 广播视图抛错(可接受): ~a~%" e)))

(format t "~%~%========= 总结: ~a 项检查, ~a 项失败 =========~%"
        *checks* *failures*)

(and (= *checks* 28)
     (= *failures* 2))
