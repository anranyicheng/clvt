;;;; robustness-test.lisp — 底层张量库鲁棒性测试
;;;; 覆盖: 边界条件、数值稳定性、内存安全、类型系统、广播、归约、线性代数
;;;; 目标: 让这个库尽可能稳健

(require :asdf)
(push (truename (make-pathname :directory '(:relative :up))) asdf:*central-registry*)
(asdf:load-system :clvt)
(in-package :clvt)

;;; 测试框架
(defvar *N* 0) (defvar *P* 0) (defvar *F* 0) (defvar *F-list* nil)

(defun approx (e a &optional (tol 1e-6))
  (cond ((and (numberp e) (numberp a))
         (cond ((and (floatp e) (floatp a))
                (or (< (abs (- e a)) (+ tol (* 0.001 (abs e))))
                    (and (vt-float-nan-p e) (vt-float-nan-p a))))
               ((and (integerp e) (integerp a)) (eql e a))
               (t (< (abs (- (float e 1.0d0) (float a 1.0d0))) tol))))
        ((and (listp e) (listp a))
         (and (= (length e) (length a))
              (every (lambda (x y) (approx x y tol)) e a)))
        (t (equal e a))))

(defun T! (name expected actual &optional (tol 1e-6))
  (incf *N*)
  (if (approx expected actual tol)
      (incf *P*)
      (progn (incf *F*) (push name *F-list*)
             (format t "  ❌ ~a~%     exp: ~a~%     got: ~a~%" name
                     (if (listp expected) (subseq expected 0 (min 5 (length expected))) expected)
                     (if (listp actual) (subseq actual 0 (min 5 (length actual))) actual)))))

(defun summary ()
  (format t "~%========================================~%")
  (format t "  Total: ~a | Pass: ~a | Fail: ~a~%" *N* *P* *F*)
  (format t "========================================~%")
  (when *F-list*
    (format t "Failed:~{~%  - ~a~}~%" (reverse *F-list*)))
  (zerop *F*))

;;; ============================================================
;;; 1. 标量张量 (0维)
;;; ============================================================
(defun test-scalar ()
  (format t "~%--- 1. 标量张量 (0维) ---~%")

  ;; 创建标量
  (let ((s (make-vt nil 42.0d0 :dtype :float64)))
    (T! "scalar: shape" nil (vt-shape s))
    (T! "scalar: size" 1 (vt-size s))
    (T! "scalar: order" 0 (vt-order s))
    (T! "scalar: ref" 42.0d0 (vt-ref s))
    (T! "scalar: item" 42.0d0 (vt-item s))
    (T! "scalar: to-list" 42.0d0 (vt-to-list s)))

  ;; 标量运算
  (let ((a (make-vt nil 3.0d0 :dtype :float64))
        (b (make-vt nil 4.0d0 :dtype :float64)))
    (T! "scalar add" 7.0d0 (vt-item (vt-+ a b)))
    (T! "scalar mul" 12.0d0 (vt-item (vt-* a b)))
    (T! "scalar neg" -3.0d0 (vt-item (vt-- a)))
    (T! "scalar abs" 3.0d0 (vt-item (vt-abs (vt-- a)))))

  ;; 标量与向量运算
  (let ((s (make-vt nil 10.0d0 :dtype :float64))
        (v (vt-from-sequence '(1.0 2.0 3.0))))
    (T! "scalar+vec" '(11.0d0 12.0d0 13.0d0) (vt-to-list (vt-+ s v)))
    (T! "scalar*vec" '(10.0d0 20.0d0 30.0d0) (vt-to-list (vt-* s v)))))

;;; ============================================================
;;; 2. 空张量 (0元素)
;;; ============================================================
(defun test-empty ()
  (format t "~%--- 2. 空张量 (0元素) ---~%")

  ;; 1D 空
  (let ((e (vt-zeros '(0) :dtype :float64)))
    (T! "empty 1d: shape" '(0) (vt-shape e))
    (T! "empty 1d: size" 0 (vt-size e))
    (T! "empty 1d: to-list" '() (vt-to-list e)))

  ;; 2D 空 (0行)
  (let ((e (vt-zeros '(0 5) :dtype :float64)))
    (T! "empty 2d(0,5): shape" '(0 5) (vt-shape e))
    (T! "empty 2d(0,5): size" 0 (vt-size e)))

  ;; 2D 空 (0列)
  (let ((e (vt-zeros '(3 0) :dtype :float64)))
    (T! "empty 2d(3,0): shape" '(3 0) (vt-shape e))
    (T! "empty 2d(3,0): size" 0 (vt-size e)))

  ;; 空张量归约
  (let ((e (vt-zeros '(0) :dtype :float64)))
    (T! "empty sum" 0.0d0 (vt-item (vt-sum e)))
    (T! "empty prod" 1.0d0 (vt-item (vt-prod e))))

  ;; 空 concatenate
  (let ((a (vt-from-sequence '(1.0 2.0 3.0)))
        (e (vt-zeros '(0) :dtype :float64)))
    (T! "concat with empty" '(1.0d0 2.0d0 3.0d0) (vt-to-list (vt-concatenate 0 a e)))
    (T! "concat empty+empty" '() (vt-to-list (vt-concatenate 0 e e)))))

;;; ============================================================
;;; 3. 单元素张量
;;; ============================================================
(defun test-single-element ()
  (format t "~%--- 3. 单元素张量 ---~%")

  (let ((s (vt-from-sequence '(42.0))))
    (T! "single: shape" '(1) (vt-shape s))
    (T! "single: ref" 42.0d0 (vt-ref s 0))
    (T! "single: sum" 42.0d0 (vt-item (vt-sum s)))
    (T! "single: mean" 42.0d0 (vt-item (vt-mean s)))
    (T! "single: std" 0.0d0 (vt-item (vt-std s)))))

;;; ============================================================
;;; 4. 所有 dtype 覆盖
;;; ============================================================
(defun test-dtypes ()
  (format t "~%--- 4. 所有 dtype 覆盖 ---~%")

  ;; int32
  (let ((a (vt-from-sequence '(1 2 3) :dtype :int32)))
    (T! "int32: dtype" :int32 (vt-dtype a))
    (T! "int32: val" '(1 2 3) (vt-to-list a))
    (T! "int32: sum" 6 (vt-item (vt-sum a))))

  ;; int64
  (let ((a (vt-from-sequence '(1 2 3) :dtype :int64)))
    (T! "int64: dtype" :int64 (vt-dtype a))
    (T! "int64: val" '(1 2 3) (vt-to-list a)))

  ;; float32
  (let ((a (vt-from-sequence '(1.0 2.0 3.0) :dtype :float32)))
    (T! "float32: dtype" :float32 (vt-dtype a))
    (T! "float32: sum" 6.0d0 (vt-item (vt-sum a))))

  ;; float64
  (let ((a (vt-from-sequence '(1.0 2.0 3.0) :dtype :float64)))
    (T! "float64: dtype" :float64 (vt-dtype a))
    (T! "float64: sum" 6.0d0 (vt-item (vt-sum a))))

  ;; astype 转换
  (let* ((i (vt-from-sequence '(1 2 3) :dtype :int32))
         (f (vt-astype i :float64)))
    (T! "int32->float64" :float64 (vt-dtype f))
    (T! "int32->float64 val" '(1.0d0 2.0d0 3.0d0) (vt-to-list f)))

  (let* ((f (vt-from-sequence '(1.7 2.3 3.9) :dtype :float64))
         (i (vt-astype f :int32)))
    (T! "float64->int32(trunc)" '(1 2 3) (vt-to-list i)))

  ;; 类型提升: int32 + float64 -> float64
  (let* ((a (vt-from-sequence '(1 2 3) :dtype :int32))
         (b (vt-from-sequence '(0.5 0.5 0.5) :dtype :float64))
         (c (vt-+ a b)))
    (T! "int32+float64->float64" :float64 (vt-dtype c))
    (T! "int32+float64 val" '(1.5d0 2.5d0 3.5d0) (vt-to-list c)))

  ;; 类型提升: int64 + float32 -> float64
  (let* ((a (vt-from-sequence '(1 2 3) :dtype :int64))
         (b (vt-from-sequence '(0.5 0.5 0.5) :dtype :float32))
         (c (vt-+ a b)))
    (T! "int64+float32->float64" :float64 (vt-dtype c)))

  ;; 类型提升: int32 * int32 -> int32
  (let* ((a (vt-from-sequence '(2 3 4) :dtype :int32))
         (b (vt-from-sequence '(5 6 7) :dtype :int32))
         (c (vt-* a b)))
    (T! "int32*int32->int32" :int32 (vt-dtype c))
    (T! "int32*int32 val" '(10 18 28) (vt-to-list c))))

;;; ============================================================
;;; 5. 广播规则 (全面覆盖)
;;; ============================================================
(defun test-broadcasting ()
  (format t "~%--- 5. 广播规则 ---~%")

  ;; 标量 + 任意形状
  (let ((s (make-vt nil 10.0d0 :dtype :float64))
        (v (vt-from-sequence '(1.0 2.0 3.0)))
        (m (vt-from-sequence '((1.0 2.0) (3.0 4.0))))
        (t3 (vt-reshape (vt-arange 8 :dtype :float64) '(2 2 2))))
    (T! "scalar+1d" '(11.0d0 12.0d0 13.0d0) (vt-to-list (vt-+ s v)))
    (T! "scalar+2d" '((11.0d0 12.0d0) (13.0d0 14.0d0)) (vt-to-list (vt-+ s m)))
    (T! "scalar+3d" '(10.0d0 11.0d0 12.0d0 13.0d0 14.0d0 15.0d0 16.0d0 17.0d0)
        (vt-to-list (vt-flatten (vt-+ s t3)))))

  ;; 1d 广播到 2d
  (let ((v (vt-from-sequence '(10.0 20.0 30.0)))
        (m (vt-reshape (vt-arange 6 :dtype :float64) '(2 3))))
    (T! "1d+2d(row)" '((10.0d0 21.0d0 32.0d0) (13.0d0 24.0d0 35.0d0))
        (vt-to-list (vt-+ m v))))

  ;; 列向量广播
  (let ((col (vt-from-sequence '((10.0) (20.0))))
        (m (vt-from-sequence '((1.0 2.0 3.0) (4.0 5.0 6.0)))))
    (T! "col+2d" '((11.0d0 12.0d0 13.0d0) (24.0d0 25.0d0 26.0d0))
        (vt-to-list (vt-+ m col))))

  ;; 2d 广播到 3d (正确对齐)
  (let ((m (vt-from-sequence '((1.0 2.0) (3.0 4.0))))
        (t3 (vt-reshape (vt-arange 8 :dtype :float64) '(2 2 2))))
    (T! "2d+3d shape" '(2 2 2) (vt-shape (vt-+ t3 m))))

  ;; 不可广播应报错
  (let ((a (vt-from-sequence '(1.0 2.0 3.0)))
        (b (vt-from-sequence '(1.0 2.0))))
    (handler-case (progn (vt-+ a b) (T! "broadcast error" t nil))
      (error () (T! "broadcast error" t t)))))

;;; ============================================================
;;; 6. 视图与内存安全
;;; ============================================================
(defun test-views ()
  (format t "~%--- 6. 视图与内存安全 ---~%")

  ;; 切片视图是零拷贝
  (let* ((a (vt-reshape (vt-arange 12 :dtype :int64) '(3 4)))
         (row (vt-slice a '(1) '(:all))))
    (T! "slice: shape" '(4) (vt-shape row))
    (T! "slice: data shared" t (eq (vt-data a) (vt-data row)))
    (T! "slice: val" '(4 5 6 7) (vt-to-list row)))

  ;; 转置视图是零拷贝
  (let* ((a (vt-reshape (vt-arange 6 :dtype :int64) '(2 3)))
         (at (vt-transpose a)))
    (T! "transpose: data shared" t (eq (vt-data a) (vt-data at)))
    (T! "transpose: shape" '(3 2) (vt-shape at)))

  ;; 修改切片影响原始张量
  (let ((a (vt-reshape (vt-arange 9 :dtype :int64) '(3 3))))
    (setf (vt-slice a '(0) '(:all)) 99)
    (T! "setf slice affects original" '((99 99 99) (3 4 5) (6 7 8)) (vt-to-list a)))

  ;; 修改转置视图影响原始张量
  (let ((a (vt-from-sequence '((1.0 2.0) (3.0 4.0)))))
    (setf (vt-ref (vt-transpose a) 0 1) 99.0)
    (T! "setf transpose affects original" '((1.0d0 2.0d0) (99.0d0 4.0d0)) (vt-to-list a)))

  ;; 连续性检查
  (let* ((a (vt-reshape (vt-arange 6 :dtype :int64) '(2 3)))
         (at (vt-transpose a)))
    (T! "reshape contiguous" t (vt-contiguous-p a))
    (T! "transpose not contiguous" nil (vt-contiguous-p at))
    (T! "copy makes contiguous" t (vt-contiguous-p (vt-copy at))))

  ;; 深拷贝是独立的
  (let* ((a (vt-from-sequence '(1.0 2.0 3.0)))
         (b (vt-copy a)))
    (setf (vt-ref b 0) 99.0)
    (T! "copy is independent" 1.0d0 (vt-ref a 0))))

;;; ============================================================
;;; 7. 索引边界与负索引
;;; ============================================================
(defun test-indexing ()
  (format t "~%--- 7. 索引边界与负索引 ---~%")

  (let ((a (vt-from-sequence '(10.0 20.0 30.0 40.0 50.0))))
    ;; 正索引
    (T! "ref[0]" 10.0d0 (vt-ref a 0))
    (T! "ref[4]" 50.0d0 (vt-ref a 4))

    ;; 负索引
    (T! "ref[-1]" 50.0d0 (vt-ref a -1))
    (T! "ref[-5]" 10.0d0 (vt-ref a -5))

    ;; 越界应报错
    (handler-case (progn (vt-ref a 5) (T! "out of bounds" t nil))
      (error () (T! "out of bounds" t t)))
    (handler-case (progn (vt-ref a -6) (T! "neg out of bounds" t nil))
      (error () (T! "neg out of bounds" t t))))

  ;; 2D 负索引
  (let ((m (vt-from-sequence '((1.0 2.0) (3.0 4.0)))))
    (T! "2d ref[-1,-1]" 4.0d0 (vt-ref m -1 -1))
    (T! "2d ref[-2,0]" 1.0d0 (vt-ref m -2 0)))

  ;; setf 负索引
  (let ((a (vt-from-sequence '(1.0 2.0 3.0))))
    (setf (vt-ref a -1) 99.0)
    (T! "setf ref[-1]" 99.0d0 (vt-ref a 2))))

;;; ============================================================
;;; 8. 归约: 所有轴组合
;;; ============================================================
(defun test-reduction ()
  (format t "~%--- 8. 归约: 所有轴组合 ---~%")

  ;; 2D 归约
  (let ((m (vt-astype (vt-reshape (vt-arange 12 :dtype :int64) '(3 4)) :float64)))
    (T! "2d sum all" 66.0d0 (vt-item (vt-sum m)))
    (T! "2d sum ax=0" '(12.0d0 15.0d0 18.0d0 21.0d0) (vt-to-list (vt-sum m :axis 0)))
    (T! "2d sum ax=1" '(6.0d0 22.0d0 38.0d0) (vt-to-list (vt-sum m :axis 1)))
    (T! "2d sum ax=-1" '(6.0d0 22.0d0 38.0d0) (vt-to-list (vt-sum m :axis -1)))
    (T! "2d sum ax=0 kd" '((12.0d0 15.0d0 18.0d0 21.0d0))
        (vt-to-list (vt-sum m :axis 0 :keepdims t)))
    (T! "2d mean all" 5.5d0 (vt-item (vt-mean m)))
    (T! "2d max all" 11.0d0 (vt-item (vt-amax m)))
    (T! "2d min ax=0" '(0.0d0 1.0d0 2.0d0 3.0d0) (vt-to-list (vt-amin m :axis 0)))
    (T! "2d argmax ax=1" '(3 3 3) (vt-to-list (vt-argmax m :axis 1)))
    (T! "2d argmin ax=0" '(0 0 0 0) (vt-to-list (vt-argmin m :axis 0)))
    (T! "2d prod all" 0.0d0 (vt-item (vt-prod m)))
    (T! "2d std all" 3.452052529534663d0 (vt-item (vt-std m)) 1e-6)
    (T! "2d var all" 11.916666666666666d0 (vt-item (vt-var m)) 1e-6))

  ;; 3D 归约
  (let ((t3 (vt-astype (vt-reshape (vt-arange 24 :dtype :int64) '(2 3 4)) :float64)))
    (T! "3d sum all" 276.0d0 (vt-item (vt-sum t3)))
    (T! "3d sum ax=0" '(12.0d0 14.0d0 16.0d0 18.0d0 20.0d0 22.0d0 24.0d0 26.0d0 28.0d0 30.0d0 32.0d0 34.0d0)
        (vt-to-list (vt-flatten (vt-sum t3 :axis 0))))
    (T! "3d sum ax=1" '(12.0d0 15.0d0 18.0d0 21.0d0 48.0d0 51.0d0 54.0d0 57.0d0)
        (vt-to-list (vt-flatten (vt-sum t3 :axis 1))))
    (T! "3d sum ax=2" '(6.0d0 22.0d0 38.0d0 54.0d0 70.0d0 86.0d0)
        (vt-to-list (vt-flatten (vt-sum t3 :axis 2))))
    (T! "3d sum ax=-1" '(6.0d0 22.0d0 38.0d0 54.0d0 70.0d0 86.0d0)
        (vt-to-list (vt-flatten (vt-sum t3 :axis -1))))
    (T! "3d sum ax=(0,1)" '(60.0d0 66.0d0 72.0d0 78.0d0)
        (vt-to-list (vt-sum t3 :axis '(0 1))))
    (T! "3d max ax=2" '(3.0d0 7.0d0 11.0d0 15.0d0 19.0d0 23.0d0)
        (vt-to-list (vt-flatten (vt-amax t3 :axis 2)))))

  ;; cumsum / cumprod
  (let ((v (vt-from-sequence '(1 2 3 4) :dtype :int64)))
    (T! "cumsum" '(1 3 6 10) (vt-to-list (vt-cumsum v)))
    (T! "cumprod" '(1 2 6 24) (vt-to-list (vt-cumprod v))))

  ;; median
  (T! "median odd" 3.0d0 (vt-item (vt-median (vt-from-sequence '(3.0 1.0 4.0 1.0 5.0)))))
  (T! "median even" 2.5d0 (vt-item (vt-median (vt-from-sequence '(1.0 2.0 3.0 4.0)))))

  ;; percentile
  (let ((v (vt-from-sequence '(1.0 2.0 3.0 4.0 5.0))))
    (T! "pct0" 1.0d0 (vt-item (vt-percentile v 0)))
    (T! "pct50" 3.0d0 (vt-item (vt-percentile v 50)))
    (T! "pct100" 5.0d0 (vt-item (vt-percentile v 100))))

  ;; ptp
  (T! "ptp" 8.0d0 (vt-item (vt-ptp (vt-from-sequence '(3.0 1.0 4.0 1.0 9.0)))))

  ;; sort
  (let ((v (vt-from-sequence '(3.0 1.0 4.0 1.0 5.0 9.0 2.0 6.0))))
    (T! "sort" '(1.0d0 1.0d0 2.0d0 3.0d0 4.0d0 5.0d0 6.0d0 9.0d0) (vt-to-list (vt-sort v)))))

;;; ============================================================
;;; 9. NaN/Inf 传播
;;; ============================================================
(defun test-nan-inf ()
  (format t "~%--- 9. NaN/Inf 传播 ---~%")

  ;; NaN 传播
  (let ((a (vt-from-sequence (list 1.0d0 +vt-float-nan+ 3.0d0) :dtype :float64)))
    (T! "isnan" '(0.0d0 1.0d0 0.0d0) (vt-to-list (vt-isnan a)))
    (T! "isfinite" '(1.0d0 0.0d0 1.0d0) (vt-to-list (vt-isfinite a))))

  ;; Inf
  (let ((a (vt-from-sequence (list 1.0d0 +vt-float-pos-inf+ +vt-float-neg-inf+) :dtype :float64)))
    (T! "isinf" '(0.0d0 1.0d0 1.0d0) (vt-to-list (vt-isinf a)))
    (T! "isfinite(inf)" '(1.0d0 0.0d0 0.0d0) (vt-to-list (vt-isfinite a))))

  ;; nan-aware 统计
  (let ((a (vt-from-sequence (list 1.0d0 +vt-float-nan+ 3.0d0 4.0d0) :dtype :float64)))
    (T! "nanmean" 2.6666666666666665d0 (vt-item (vt-nanmean a)) 1e-6)
    (T! "nansum" 8.0d0 (vt-item (vt-nansum a)))
    (T! "nanmax" 4.0d0 (vt-item (vt-nanmax a)))
    (T! "nanmin" 1.0d0 (vt-item (vt-nanmin a))))

  ;; NaN 在比较中
  (let ((a (make-vt nil +vt-float-nan+ :dtype :float64))
        (b (make-vt nil 1.0d0 :dtype :float64)))
    (T! "nan == nan" 0.0d0 (vt-item (vt-= a (make-vt nil +vt-float-nan+ :dtype :float64))))
    (T! "nan < 1" 0.0d0 (vt-item (vt-< a b)))))

;;; ============================================================
;;; 10. 数值稳定性
;;; ============================================================
(defun test-numerical-stability ()
  (format t "~%--- 10. 数值稳定性 ---~%")

  ;; 大数相加 (int64 溢出)
  (let* ((big (vt-from-sequence '(1000000000000 1000000000000) :dtype :int64))
         (sum (vt-sum big)))
    (T! "int64 big sum" 2000000000000 (vt-item sum)))

  ;; softmax 数值稳定性 (减去 max)
  (let* ((logits (vt-from-sequence '(1000.0 1001.0 1002.0)))
         (probs (vt-softmax logits)))
    (T! "softmax large values: sum" 1.0d0 (vt-item (vt-sum probs)) 1e-5)
    (T! "softmax large values: monotone" t
        (> (vt-ref probs 2) (vt-ref probs 1))
        (> (vt-ref probs 1) (vt-ref probs 0))))

  ;; log-softmax 数值稳定性
  (let* ((logits (vt-from-sequence '(1000.0 1001.0 1002.0)))
         (lsm (vt-log-softmax logits)))
    (T! "log-softmax: all negative" t
        (every (lambda (x) (< x 0)) (vt-to-list lsm))))

  ;; sigmoid 大值
  (let* ((x (vt-from-sequence '(-100.0 0.0 100.0)))
         (s (vt-sigmoid x)))
    (T! "sigmoid(-100)≈0" 0.0d0 (vt-ref s 0) 1e-10)
    (T! "sigmoid(0)=0.5" 0.5d0 (vt-ref s 1) 1e-10)
    (T! "sigmoid(100)≈1" 1.0d0 (vt-ref s 2) 1e-10))

  ;; relu 负值
  (let* ((x (vt-from-sequence '(-100.0 -1.0 0.0 1.0 100.0)))
         (r (vt-relu x)))
    (T! "relu" '(0.0d0 0.0d0 0.0d0 1.0d0 100.0d0) (vt-to-list r)))

  ;; log(0) = -inf
  (let ((a (vt-from-sequence '(0.0 1.0) :dtype :float64)))
    (T! "log(0) is -inf" t (vt-float-inf-p (vt-ref (vt-log a) 0)))))

;;; ============================================================
;;; 11. 线性代数: 边界情况
;;; ============================================================
(defun test-linalg-edges ()
  (format t "~%--- 11. 线性代数: 边界情况 ---~%")

  ;; 单位矩阵
  (let ((I (vt-eye 3 :dtype :float64)))
    (T! "eye: det=1" 1.0d0 (vt-item (vt-det I)))
    (T! "eye: inv=I" '((1.0d0 0.0d0 0.0d0) (0.0d0 1.0d0 0.0d0) (0.0d0 0.0d0 1.0d0))
        (vt-to-list (vt-inv I)))
    (T! "eye: rank=3" 3 (vt-matrix-rank I)))

  ;; 对角矩阵
  (let ((D (vt-from-sequence '((2.0 0.0) (0.0 3.0)))))
    (T! "diag det" 6.0d0 (vt-item (vt-det D)))
    (T! "diag inv" '((0.5d0 0.0d0) (0.0d0 0.3333333333333333d0))
        (vt-to-list (vt-inv D)) 1e-6))

  ;; 奇异矩阵应报错
  (let ((S (vt-from-sequence '((1.0 2.0) (2.0 4.0)))))
    (handler-case (progn (vt-inv S) (T! "singular inv error" t nil))
      (error () (T! "singular inv error" t t))))

  ;; 矩阵秩
  (T! "rank full" 3 (vt-matrix-rank (vt-eye 3 :dtype :float64)))
  (T! "rank deficient" 1 (vt-matrix-rank (vt-from-sequence '((1.0 2.0) (2.0 4.0)))))
  (T! "rank zero" 0 (vt-matrix-rank (vt-zeros '(3 3))))

  ;; Cholesky
  (let ((A (vt-from-sequence '((4.0 2.0) (2.0 3.0)))))
    (let ((L (vt-cholesky A)))
      (T! "chol: L@L^T=A" 0.0d0
          (vt-item (vt-amax (vt-abs (vt-- A (vt-@ L (vt-transpose L)))))) 1e-10)))

  ;; 非正定应报错
  (let ((S (vt-from-sequence '((1.0 2.0) (2.0 1.0)))))
    (handler-case (progn (vt-cholesky S) (T! "non-posdef error" t nil))
      (error () (T! "non-posdef error" t t))))

  ;; 特征值: 对角矩阵
  (let ((D (vt-from-sequence '((5.0 0.0) (0.0 2.0)))))
    (multiple-value-bind (vals vecs) (vt-eig D)
      (T! "eig diag" '(5.0d0 2.0d0) (vt-to-list vals) 1e-6)))

  ;; 特征值: 对称矩阵
  (let ((A (vt-from-sequence '((2.0 1.0) (1.0 2.0)))))
    (multiple-value-bind (vals vecs) (vt-eig A)
      (T! "eig symmetric" '(3.0d0 1.0d0) (vt-to-list vals) 1e-6)
      (T! "eig recon err" 0.0d0
          (vt-item (vt-amax (vt-abs (vt-- A (vt-@ vecs (vt-@ (vt-diag vals) (vt-transpose vecs)))))))
          1e-6)))

  ;; SVD: 各种形状
  (let ((A (vt-from-sequence '((1.0 2.0) (3.0 4.0) (5.0 6.0)))))
    (multiple-value-bind (U S Vt) (vt-svd A)
      (T! "svd 3x2: recon err" 0.0d0
          (vt-item (vt-amax (vt-abs (vt-- A (vt-@ U (vt-@ (vt-diag S) Vt)))))) 1e-10)))

  ;; norm
  (T! "norm(3,4)" 5.0d0 (vt-item (vt-norm (vt-from-sequence '(3.0 4.0)))))
  (T! "l1-norm" 6.0d0 (vt-item (vt-l1-norm (vt-from-sequence '(-1.0 2.0 -3.0))))))

;;; ============================================================
;;; 12. 随机数: 统计属性
;;; ============================================================
(defun test-random ()
  (format t "~%--- 12. 随机数: 统计属性 ---~%")

  ;; 均匀分布 [0,1)
  (vt-random-seed 42)
  (let ((r (vt-random '(10000))))
    (T! "uniform: all >=0" t (every (lambda (x) (>= x 0)) (vt-to-list r)))
    (T! "uniform: all <1" t (every (lambda (x) (< x 1)) (vt-to-list r)))
    (T! "uniform: mean≈0.5" 0.5d0 (vt-item (vt-mean r)) 0.02)
    (T! "uniform: std≈0.289" 0.289d0 (vt-item (vt-std r)) 0.02))

  ;; 正态分布
  (vt-random-seed 42)
  (let ((r (vt-random-normal '(10000) :mean 0.0 :std 1.0)))
    (T! "normal: mean≈0" 0.0d0 (vt-item (vt-mean r)) 0.05)
    (T! "normal: std≈1" 1.0d0 (vt-item (vt-std r)) 0.05))

  ;; 种子可复现
  (vt-random-seed 123)
  (let ((r1 (vt-random '(5))))
    (vt-random-seed 123)
    (let ((r2 (vt-random '(5))))
      (T! "seed reproducible" t (approx (vt-to-list r1) (vt-to-list r2))))))

;;; ============================================================
;;; 13. einsum: 各种模式
;;; ============================================================
(defun test-einsum-patterns ()
  (format t "~%--- 13. einsum 模式 ---~%")

  ;; 内积
  (T! "einsum dot" 32.0d0
      (vt-item (vt-einsum "i,i->"
                          (vt-from-sequence '(1 2 3) :dtype :int64)
                          (vt-from-sequence '(4 5 6) :dtype :int64))))

  ;; 矩阵乘
  (let ((A (vt-reshape (vt-arange 6 :dtype :int64) '(2 3)))
        (B (vt-reshape (vt-arange 6 :dtype :int64) '(3 2))))
    (T! "einsum matmul" '((10 13) (28 40))
        (vt-to-list (vt-einsum "ij,jk->ik" A B))))

  ;; 转置
  (let ((A (vt-from-sequence '((1 2) (3 4)) :dtype :int64)))
    (T! "einsum transpose" '((1 3) (2 4)) (vt-to-list (vt-einsum "ij->ji" A))))

  ;; 对角线
  (let ((A (vt-reshape (vt-arange 9 :dtype :int64) '(3 3))))
    (T! "einsum diag" '(0 4 8) (vt-to-list (vt-einsum "ii->i" A)))
    (T! "einsum trace" 12.0d0 (vt-item (vt-einsum "ii->" A))))

  ;; 外积
  (T! "einsum outer" '((4 5) (8 10) (12 15))
      (vt-to-list (vt-einsum "i,j->ij"
                             (vt-from-sequence '(1 2 3) :dtype :int64)
                             (vt-from-sequence '(4 5) :dtype :int64))))

  ;; 3D reduce
  (let ((t3 (vt-astype (vt-reshape (vt-arange 24 :dtype :int64) '(2 3 4)) :float64)))
    (T! "einsum 3d->1d" '(66.0d0 210.0d0)
        (vt-to-list (vt-einsum "ijk->i" t3)))
    (T! "einsum 3d->2d" '(6.0d0 22.0d0 38.0d0 54.0d0 70.0d0 86.0d0)
        (vt-to-list (vt-flatten (vt-einsum "ijk->ij" t3))))))

;;; ============================================================
;;; 14. 复合赋值与原地操作
;;; ============================================================
(defun test-inplace ()
  (format t "~%--- 14. 复合赋值与原地操作 ---~%")

  ;; vt-fill
  (let ((a (vt-zeros '(3 3))))
    (vt-fill a 7.0)
    (T! "fill" '((7.0d0 7.0d0 7.0d0) (7.0d0 7.0d0 7.0d0) (7.0d0 7.0d0 7.0d0))
        (vt-to-list a)))

  ;; setf vt-ref
  (let ((a (vt-from-sequence '(1.0 2.0 3.0))))
    (setf (vt-ref a 1) 99.0)
    (T! "setf ref" '(1.0d0 99.0d0 3.0d0) (vt-to-list a)))

  ;; setf vt-slice (标量广播)
  (let ((a (vt-arange 5 :dtype :int64)))
    (setf (vt-slice a '(1 4)) 0)
    (T! "setf slice scalar" '(0 0 0 0 4) (vt-to-list a)))

  ;; setf vt-slice (张量赋值)
  (let ((a (vt-arange 6 :dtype :int64))
        (b (vt-from-sequence '(99 98 97) :dtype :int64)))
    (setf (vt-slice a '(0 3)) b)
    (T! "setf slice tensor" '(99 98 97 3 4 5) (vt-to-list a))))

;;; ============================================================
;;; 15. 高维操作
;;; ============================================================
(defun test-high-dim ()
  (format t "~%--- 15. 高维操作 ---~%")

  ;; 4D 张量
  (let ((t4 (vt-reshape (vt-arange 120 :dtype :float64) '(2 3 4 5))))
    (T! "4d: shape" '(2 3 4 5) (vt-shape t4))
    (T! "4d: size" 120 (vt-size t4))
    (T! "4d: sum all" 7140.0d0 (vt-item (vt-sum t4)))
    (T! "4d: sum ax=2" '(30.0d0 34.0d0 38.0d0 42.0d0 46.0d0 110.0d0 114.0d0 118.0d0 122.0d0 126.0d0 190.0d0 194.0d0 198.0d0 202.0d0 206.0d0 270.0d0 274.0d0 278.0d0 282.0d0 286.0d0 350.0d0 354.0d0 358.0d0 362.0d0 366.0d0 430.0d0 434.0d0 438.0d0 442.0d0 446.0d0)
        (vt-to-list (vt-flatten (vt-sum t4 :axis 2)))))

  ;; 5D 张量
  (let ((t5 (vt-reshape (vt-arange 32 :dtype :float64) '(2 2 2 2 2))))
    (T! "5d: shape" '(2 2 2 2 2) (vt-shape t5))
    (T! "5d: sum all" 496.0d0 (vt-item (vt-sum t5))))

  ;; 4D 矩阵乘法
  (let ((A (vt-reshape (vt-arange 16 :dtype :float64) '(2 2 2 2)))
        (B (vt-reshape (vt-arange 16 :dtype :float64) '(2 2 2 2))))
    (let ((C (vt-@ A B)))
      (T! "4d matmul: shape" '(2 2 2 2) (vt-shape C)))))

;;; ============================================================
;;; 16. einsum 批量矩阵乘法
;;; ============================================================
(defun test-einsum-batch ()
  (format t "~%--- 16. einsum 批量矩阵乘法 ---~%")

  ;; 2x2 批量
  (let* ((A (vt-reshape (vt-arange 8 :dtype :int64) '(2 2 2)))
         (B (vt-reshape (vt-arange 8 :dtype :int64) '(2 2 2)))
         (C (vt-einsum "...ij,...jk->...ik" A B)))
    (T! "bmm 2x2: batch0" '((2 3) (6 11))
        (vt-to-list (vt-slice C '(0) '(:all) '(:all))))
    (T! "bmm 2x2: batch1" '((46 55) (66 79))
        (vt-to-list (vt-slice C '(1) '(:all) '(:all)))))

  ;; 批量 matmul 函数
  (let* ((A (vt-reshape (vt-arange 8 :dtype :float64) '(2 2 2)))
         (B (vt-reshape (vt-arange 8 :dtype :float64) '(2 2 2)))
         (C (vt-matmul A B)))
    (T! "matmul batch: shape" '(2 2 2) (vt-shape C))))

;;; ============================================================
;;; 17. 形状推断 (-1)
;;; ============================================================
(defun test-shape-inference ()
  (format t "~%--- 17. 形状推断 (-1) ---~%")

  (let* ((a (vt-arange 12 :dtype :int64))
         (b (vt-reshape a '(3 -1))))
    (T! "reshape (3,-1)" '(3 4) (vt-shape b)))

  (let* ((a (vt-arange 12 :dtype :int64))
         (b (vt-reshape a '(-1 4))))
    (T! "reshape (-1,4)" '(3 4) (vt-shape b)))

  (let* ((a (vt-arange 24 :dtype :int64))
         (b (vt-reshape a '(2 -1 4))))
    (T! "reshape (2,-1,4)" '(2 3 4) (vt-shape b)))

  ;; 不能整除应报错
  (handler-case (progn (vt-reshape (vt-arange 10 :dtype :int64) '(3 -1))
                       (T! "reshape error" t nil))
    (error () (T! "reshape error" t t))))

;;; ============================================================
;;; 18. 连续性与内存布局
;;; ============================================================
(defun test-contiguity ()
  (format t "~%--- 18. 连续性与内存布局 ---~%")

  ;; 行主序连续
  (let ((a (vt-reshape (vt-arange 6 :dtype :int64) '(2 3))))
    (T! "reshape contiguous" t (vt-contiguous-p a))
    (T! "reshape strides" '(3 1) (vt-strides a)))

  ;; 转置不连续
  (let ((at (vt-transpose (vt-reshape (vt-arange 6 :dtype :int64) '(2 3)))))
    (T! "transpose strides" '(1 3) (vt-strides at))
    (T! "transpose not contiguous" nil (vt-contiguous-p at))
    ;; 强制连续化
    (let ((c (vt-contiguous at)))
      (T! "contiguous copy" t (vt-contiguous-p c))
      (T! "contiguous copy data" '((0 3) (1 4) (2 5)) (vt-to-list c))))

  ;; 切片可能不连续
  (let* ((a (vt-reshape (vt-arange 12 :dtype :int64) '(3 4)))
         (col (vt-slice a '(:all) '(1))))
    (T! "column slice strides" '(4) (vt-strides col))
    (T! "column slice not contiguous" nil (vt-contiguous-p col))))

;;; ============================================================
;;; 运行所有测试
;;; ============================================================
(defun run-robustness-tests ()
  (format t "~%========================================~%")
  (format t "  clvt ROBUSTNESS TESTS~%")
  (format t "  底层张量库鲁棒性测试~%")
  (format t "========================================~%")

  (test-scalar)
  (test-empty)
  (test-single-element)
  (test-dtypes)
  (test-broadcasting)
  (test-views)
  (test-indexing)
  (test-reduction)
  (test-nan-inf)
  (test-numerical-stability)
  (test-linalg-edges)
  (test-random)
  (test-einsum-patterns)
  (test-inplace)
  (test-high-dim)
  (test-einsum-batch)
  (test-shape-inference)
  (test-contiguity)

  (summary))

(run-robustness-tests)
