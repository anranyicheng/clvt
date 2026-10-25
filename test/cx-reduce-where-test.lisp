;;;; cx-reduce-where-test.lisp — 极值归约（amax/amin）与掩码三元选择（vt-where）
;;;;                              的正确性 + 倍率护栏
;;;;
;;;; 背景（CONVENTIONS §16.8.1 / §16.14.2）：
;;;;   1. 极值归约把累加器写成 `(let ((acc (aref d 0))) … (vector acc))` 时，
;;;;      SBCL 因为 (vector acc) 接受任意类型、且 %COP-MAX 的派生返回类型是 T，
;;;;      会把 ACC 放宽成 T —— 每元素装箱/解包，1M complex128 实测 41ms；
;;;;      补上 (type (complex double-float) acc) 后 3ms。
;;;;   2. `vt-where` 是三元选择，三个输入与结果的元素类型永不相同（掩码恒为
;;;;      :int8），进不去内联快路径，只能走 vt-map 的闭包：1M complex128 32ms。
;;;;      %SELECT-FAST 按元素类型特化后 5ms；nansum/nanmean/nanvar/nanstd/
;;;;      nanmax/nancumsum 全部建在它上面，收益直接传导。
;;;;
;;;; 本套件：
;;;;   A. vt-where：与逐元素参考一致（8 实数 dtype + 2 复数 dtype × 4 种掩码）；
;;;;   B. vt-amax / vt-amin：NaN 传播、单轴、c64/c128、空归约报错；
;;;;   C. 排序/argsort 在「双缓冲乒乓」之后仍正确（跨过插入排序阈值）；
;;;;   D. 护栏：where 与极值归约的复数/实数倍率。
;;;;
;;;; 用法：sbcl --dynamic-space-size 4096 --non-interactive --load test/cx-reduce-where-test.lisp

(require :asdf)
(defparameter cl-user::*clvt-test-dir*
  (make-pathname :name nil :type nil :defaults *load-truename*))
(pushnew (merge-pathnames "../" cl-user::*clvt-test-dir*) asdf:*central-registry* :test #'equal)
(handler-bind ((warning #'muffle-warning)) (asdf:load-system :clvt))

(in-package :clvt)

(defvar *pass* 0)
(defvar *fail* 0)
(defvar *failures* '())

(defun expect-true (name form)
  (if form
      (incf *pass*)
      (progn (incf *fail*) (push name *failures*) (format t "~&FAIL ~a~%" name))))

(defun expect-equal (name got want)
  (if (equal got want)
      (incf *pass*)
      (progn (incf *fail*)
             (push (format nil "~a: 期望 ~s 实得 ~s" name want got) *failures*)
             (format t "~&FAIL ~a: 期望 ~s 实得 ~s~%" name want got))))

(defun feq (x y)
  "NaN 只与 NaN 相等（先判 NaN 再走 `=`，避免开着浮点陷阱时抛异常）。"
  (cond ((and (floatp x) (sb-ext:float-nan-p x))
         (and (floatp y) (sb-ext:float-nan-p y)))
        ((and (floatp y) (sb-ext:float-nan-p y)) nil)
        (t (= x y))))

(defun val= (a b)
  (cond ((and (complexp a) (complexp b))
         (and (feq (realpart a) (realpart b)) (feq (imagpart a) (imagpart b))))
        ((or (complexp a) (complexp b)) nil)
        (t (feq a b))))

(defun flat (tv)
  "展平后的普通列表（VT-TO-LIST 对多维返回嵌套列表）。"
  (if (zerop (vt-order tv))
      (list (vt-item tv))
      (vt-to-list (vt-flatten tv))))

(defun cxn (z)
  (or (sb-ext:float-nan-p (realpart z)) (sb-ext:float-nan-p (imagpart z))))

;;; ==================================================================
;;; A. vt-where
;;; ==================================================================

(defparameter *mask-specs*
  '((:zero . "全 0") (:one . "全 1") (:alt . "交替") (:neg . "负值算真")))

(defun mask-value (kind i)
  (case kind
    (:zero 0)
    (:one 1)
    (:alt (if (evenp i) 1 0))
    (:neg (if (zerop (mod i 3)) -5 0))))

(defun ref-select (xs ys ms)
  (loop for x in xs for y in ys for m in ms collect (if (/= m 0) x y)))

(defun test-where-real ()
  (let* ((n 12)
         (xs (loop for i below n collect (coerce (+ 1 (* i 0.5)) 'double-float)))
         (ys (loop for i below n collect (coerce (- 10 (* i 0.25)) 'double-float))))
    (dolist (dt '(:float64 :float32 :int64 :int32 :int16 :int8 :uint16 :uint8))
      (let ((x (vt-from-sequence xs :dtype dt))
            (y (vt-from-sequence ys :dtype dt)))
        (dolist (spec *mask-specs*)
          (let* ((kind (car spec))
                 (ml (loop for i below n collect (mask-value kind i)))
                 (m (vt-from-sequence ml :dtype :int8))
                 (got (flat (vt-where m x y)))
                 (want (ref-select (flat x) (flat y) ml)))
            (expect-true (format nil "where ~a ~a 取值" dt (cdr spec))
                         (every #'val= got want))
            (expect-equal (format nil "where ~a ~a dtype" dt (cdr spec))
                          (vt-dtype (vt-where m x y)) dt)))))))

(defun test-where-complex ()
  (let* ((n 12)
         (cx (loop for i below n collect (complex (coerce i 'double-float) 0.5d0)))
         (cy (loop for i below n collect (complex 0d0 (coerce (- i) 'double-float))))
         (ml (loop for i below n collect (mod i 2))))
    (dolist (dt (list :complex128 :complex64))
      (let* ((x (vt-from-sequence cx :dtype dt))
             (y (vt-from-sequence cy :dtype dt))
             (m (vt-from-sequence ml :dtype :int8))
             (got (flat (vt-where m x y)))
             (want (ref-select (flat x) (flat y) ml)))
        (expect-true (format nil "where 复数 ~a 取值" dt) (every #'val= got want))
        (expect-equal (format nil "where 复数 ~a dtype" dt)
                      (vt-dtype (vt-where m x y)) dt)))))

(defun test-where-layouts ()
  (let* ((n 9)
         (base (vt-from-sequence (loop for i below n collect (coerce i 'double-float))))
         (ml (loop for i below n collect (mod i 3)))
         (m (vt-from-sequence ml :dtype :int8))
         (mt (vt-transpose (vt-reshape base (list 3 3))))
         (m2 (vt-reshape (vt-from-sequence (loop for i below 9 collect (mod i 2))
                                          :dtype :int8)
                         (list 3 3))))
    (expect-true "where 标量 x 广播"
                 (every #'val= (flat (vt-where m (vt-cast 100 :float64) base))
                        (loop for i below n collect
                              (if (/= (mod i 3) 0) 100d0 (coerce i 'double-float)))))
    (expect-true "where 转置视图 == 物化副本"
                 (every #'val= (flat (vt-where m2 mt mt))
                        (flat (vt-where m2 (vt-copy mt) (vt-copy mt)))))
    (let ((out (vt-zeros (list n) :dtype :float64)))
      (vt-where m base base :out out)
      (expect-equal "where :out 写满" (flat out) (flat base)))
    (expect-equal "where 单参数返回坐标个数" (length (vt-where m)) 1)
    (expect-true "where 只给 x 报错"
                 (handler-case (progn (vt-where m base) nil) (error () t)))
    (expect-true "where :out dtype 不一致报错"
                 (handler-case
                     (progn (vt-where m base base :out (vt-zeros (list n) :dtype :float32)) nil)
                   (error () t)))))

;;; ==================================================================
;;; B. vt-amax / vt-amin
;;; ==================================================================

(defvar *z0* 0d0)

(defun mk-nan () (with-float-safe (/ *z0* *z0*)))

(defun test-extremum-reduce ()
  (let* ((nan (mk-nan))
         (z (vt-from-sequence (list (complex 1d0 2d0) (complex 0d0 9d0)
                                    (complex nan 0d0) (complex 2d0 0d0))
                              :dtype :complex128)))
    (expect-true "amax 复数 NaN 传播"
                 (sb-ext:float-nan-p (realpart (vt-item (vt-amax z)))))
    (expect-true "amin 复数 NaN 传播"
                 (sb-ext:float-nan-p (realpart (vt-item (vt-amin z))))))
  (let ((z2 (vt-from-sequence (list (complex 1d0 2d0) (complex 0d0 9d0)
                                    (complex 2d0 0d0))
                              :dtype :complex128)))
    (expect-true "amax 无 NaN 取字典序最大"
                 (val= (vt-item (vt-amax z2)) (complex 2d0 0d0)))
    ;; 字典序：(0+9i) < (1+2i) < (2+0i) —— 实部优先，所以最小是 0+9i
    (expect-true "amin 无 NaN 取字典序最小"
                 (val= (vt-item (vt-amin z2)) (complex 0d0 9d0))))
  (let* ((m (vt-reshape (vt-from-sequence
                         (list (complex 3d0 0d0) (complex 1d0 5d0)
                               (complex 2d0 0d0) (complex 4d0 -1d0))
                         :dtype :complex128)
                        (list 2 2)))
         (a1 (vt-amax m :axis 1))
         (a0 (vt-amax m :axis 0)))
    (expect-equal "amax axis=1 形状" (vt-shape a1) (list 2))
    (expect-true "amax axis=1 行 0" (val= (vt-item (vt-ref a1 0)) (complex 3d0 0d0)))
    (expect-true "amax axis=1 行 1" (val= (vt-item (vt-ref a1 1)) (complex 4d0 -1d0)))
    (expect-true "amax axis=0 列 0" (val= (vt-item (vt-ref a0 0)) (complex 3d0 0d0)))
    ;; 第 1 列 = {1+5i, 4-1i}，字典序比实部 → 最大 4-1i
    (expect-true "amax axis=0 列 1" (val= (vt-item (vt-ref a0 1)) (complex 4d0 -1d0))))
  (let ((z64 (vt-from-sequence (list (complex 1s0 2s0) (complex 0s0 9s0)
                                     (complex 2s0 0s0))
                               :dtype :complex64)))
    (expect-true "amax c64 取值" (val= (vt-item (vt-amax z64)) (complex 2s0 0s0)))
    (expect-equal "amax c64 保 dtype" (vt-dtype (vt-amax z64)) :complex64))
  (expect-true "amax 空数组报错"
               (handler-case (progn (vt-amax (vt-zeros (list 0) :dtype :complex128)) nil)
                 (error () t))))

;;; ==================================================================
;;; C. 排序双缓冲乒乓
;;; ==================================================================

(defun cx-le-safe (a b)
  (cond ((cxn a) t)
        ((cxn b) nil)
        (t (or (< (realpart a) (realpart b))
               (and (= (realpart a) (realpart b))
                    (<= (imagpart a) (imagpart b)))))))

(defun nondecreasing-p (lst)
  (loop for (a b) on lst while b always (cx-le-safe a b)))

(defun test-sort-pingpong ()
  (dolist (n '(1 2 3 16 17 18 33 64 65 1000))
    (let* ((z (vt-random-uniform (list n) :low 0d0 :high 1d0 :dtype :complex128))
           (sv (vt-to-list (vt-sort z)))
           (si (vt-to-list (vt-argsort z)))
           (orig (vt-to-list z)))
      (expect-true (format nil "n=~a 排序非降序" n) (nondecreasing-p sv))
      (expect-equal (format nil "n=~a argsort 是排列" n)
                    (sort (copy-list si) #'<)
                    (loop for i below n collect i))
      (expect-true (format nil "n=~a 按下标取值 == vt-sort" n)
                   (every #'val= (mapcar (lambda (i) (nth i orig)) si) sv)))))

;;; ==================================================================
;;; D. 护栏
;;; ==================================================================

(defun us-now ()
  "微秒分辨率时钟。**不要**用 GET-INTERNAL-REAL-TIME：它在 SBCL 上只有
   1ms 的 tick（INTERNAL-TIME-UNITS-PER-SECOND = 1000），比 tick 短的操作
   一律量成 0 —— `vt-amax` float64 1M 约 0.8ms，会被量成 0，于是
   「c128 / f64 倍率」的分母落到下限 0.05，护栏偶发报 60x 假失败。"
  (multiple-value-bind (s u) (sb-ext:get-time-of-day)
    (+ (* s 1000000) u)))

(defun ms-of (thunk &key (runs 2))
  (let ((best most-positive-fixnum))
    (dotimes (_ (1+ runs))
      (sb-ext:gc :full t)
      (let ((t0 (us-now)))
        (funcall thunk)
        (setf best (min best (- (us-now) t0)))))
    (/ best 1000.0d0)))

(defun test-guardrails ()
  (let* ((n 1000000)
         (zc (vt-random-uniform (list n) :low 0d0 :high 1d0 :dtype :complex128))
         (rc (vt-random-uniform (list n) :low 0d0 :high 1d0))
         (mk (vt-from-sequence (loop for i below n collect (mod i 2)) :dtype :int8))
         (t-where-c (ms-of (lambda () (vt-where mk zc zc))))
         (t-where-r (ms-of (lambda () (vt-where mk rc rc))))
         (t-amax-c  (ms-of (lambda () (vt-amax zc))))
         (t-amax-r  (ms-of (lambda () (vt-amax rc)))))
    (let ((ratio (/ t-where-c (max t-where-r 0.05d0))))
      (expect-true (format nil "where c128/f64 倍率 ~,1fx ≤ 4x" ratio)
                   (<= ratio 4.0d0)))
    (let ((ratio (/ t-amax-c (max t-amax-r 0.05d0))))
      (expect-true (format nil "amax c128/f64 倍率 ~,1fx ≤ 8x" ratio)
                   (<= ratio 8.0d0)))
    (expect-true (format nil "amax c128 绝对 < 40ms（实测 ~,1f）" t-amax-c)
                 (< t-amax-c 40.0d0))))

(defun main ()
  (format t "~&=== 复数归约 / 掩码选择 测试 ===")
  (test-where-real)
  (test-where-complex)
  (test-where-layouts)
  (test-extremum-reduce)
  (test-sort-pingpong)
  (test-guardrails)
  (format t "~&~%通过: ~a  失败: ~a~%" *pass* *fail*)
  (when *failures*
    (format t "~&失败清单:~%")
    (dolist (f (reverse *failures*)) (format t "  - ~a~%" f)))
  (when (plusp *fail*)
    (sb-ext:exit :code 1)))

(main)
