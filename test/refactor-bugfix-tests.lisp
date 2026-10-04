;;;; refactor-bugfix-tests.lisp — v0.3.5 三层分离重构回归测试
;;;;
;;;; 覆盖本轮修复的 8 类 bug（详见 CHANGELOG v0.3.5）：
;;;;   1. floor 族 / mod / rem 非有限值传播（原直接崩溃）
;;;;   2. vt-mod / vt-rem 张量除数广播
;;;;   3. vt-insert 重复索引顺序 + 负索引解析（numpy 语义）
;;;;   4. vt-solve b 形状校验（原可越界读写）
;;;;   5. vt-inner 0 维输入
;;;;   6. vt-tensordot 负轴归一化
;;;;   7. vt-from-array dtype 推断覆盖完整全集
;;;;   8. vt-cast-fun :uint16 与 %coerce-* 统一
(require :asdf)
#+quicklisp (ql:quickload :clvt)
(asdf:load-system :clvt)
(in-package :clvt)

(defparameter *pass* 0)
(defparameter *fail* 0)

(defparameter +nan+ (vt-get-nan :float64))
(defparameter +inf+ (vt-get-pos-inf :float64))
(defparameter -inf+ (vt-get-neg-inf :float64))

(defun tb-nan-p (x)
  (and (floatp x)
       (with-float-safe (not (= x x)))))

(defmacro check (desc expected actual &key (test #'equalp))
  `(progn
     (format t "CASE: ~a~%" ,desc)
     (let ((exp ,expected) (act ,actual))
       (if (funcall ,test exp act)
           (progn (incf *pass*) (format t "PASS: ~a~%" ,desc))
           (progn (incf *fail*)
                  (format t "FAIL: ~a~%  expected: ~s~%  actual:   ~s~%"
                          ,desc exp act))))))

(defmacro check-error (desc form)
  `(progn
     (format t "CASE: ~a~%" ,desc)
     (handler-case (progn ,form (incf *fail*)
                         (format t "FAIL: ~a（未报错）~%" ,desc))
       (error () (incf *pass*) (format t "PASS: ~a~%" ,desc)))))

(defun =* (eps)
  "返回近似相等比较闭包（NaN 感知，容差 eps）。eps 词法捕获。"
  (lambda (e a) (and (= (length e) (length a))
                     (every (lambda (x y)
                              (cond ((tb-nan-p x) (tb-nan-p y))
                                    ((floatp x) (and (floatp y)
                                                     (< (abs (- x y)) eps)))
                                    (t (= x y))))
                            e a))))

(defun v1= (val)
  "断言 0 维张量取值等于 val（NaN 感知）。签名 (expected actual) 以适配 check 的 :test。"
  (lambda (e a)
    (declare (ignore e))
    (if (tb-nan-p val)
        (tb-nan-p a)
        (and (numberp a) (= a val)))))

;;; ================================================================
;;; 1. floor 族：NaN/±Inf 传播（对标 numpy.floor/ceil/trunc/round）
;;; ================================================================
(format t "~%===== 1. floor 族非有限值传播 =====~%")
(check "floor(nan) = nan" t (vt-item (vt-floor (vt-const '() +nan+))) :test (v1= +nan+))
(check "floor(+inf) = +inf" t (vt-item (vt-floor (vt-const '() +inf+)))
       :test (v1= +inf+))
(check "floor(-inf) = -inf" t (vt-item (vt-floor (vt-const '() -inf+)))
       :test (v1= -inf+))
(check "ceiling(nan) = nan" t (vt-item (vt-ceiling (vt-const '() +nan+))) :test (v1= +nan+))
(check "truncate(nan) = nan" t (vt-item (vt-truncate (vt-const '() +nan+))) :test (v1= +nan+))
(check "round(nan) = nan" t (vt-item (vt-round (vt-const '() +nan+))) :test (v1= +nan+))
(check "rint(nan) = nan" t (vt-item (vt-rint (vt-const '() +nan+))) :test (v1= +nan+))
(check "floor(-2.5) = -3" t (vt-item (vt-floor (vt-const '() -2.5d0))) :test (v1= -3.0d0))
(check "floor(2.9) = 2（有限值路径不变）" t (vt-item (vt-floor (vt-const '() 2.9d0)))
       :test (v1= 2.0d0))
;; float32 输入的 NaN 传播
(check "floor float32 nan = nan" t
       (vt-item (vt-floor (make-vt nil (vt-get-nan :float32) :dtype :float32)))
       :test (v1= (vt-get-nan :float32)))

;;; ================================================================
;;; 2. vt-mod / vt-rem：张量除数 + 非有限值
;;; ================================================================
(format t "~%===== 2. vt-mod / vt-rem =====~%")
(check "mod([5,6],[3,4]) = [2,2]" '(2.0d0 2.0d0)
       (coerce (vt-data (vt-mod (vt-from-sequence '(5 6))
                                (vt-from-sequence '(3 4)))) 'list)
       :test (=* 1e-12))
(check "mod(-5,3) = 1（与除数同号，numpy.remainder）" '(1.0d0)
       (coerce (vt-data (vt-mod (vt-from-sequence '(-5)) 3)) 'list)
       :test (=* 1e-12))
(check "rem(-5,3) = -2（与被除数同号，numpy.fmod）" '(-2.0d0)
       (coerce (vt-data (vt-rem (vt-from-sequence '(-5)) 3)) 'list)
       :test (=* 1e-12))
(check "mod([5,7],0) = [nan,nan]（v0.3.6：浮点被除数语境 IEEE 零除）"
       (list +nan+ +nan+)
       (coerce (vt-data (vt-mod (vt-from-sequence '(5 7)) 0)) 'list)
       :test (=* 1e-12))
(check "mod(5,nan) = nan" t (vt-item (vt-mod (vt-const '() 5.0d0) +nan+)) :test (v1= +nan+))
(check "mod(nan,3) = nan" t (vt-item (vt-mod (vt-const '() +nan+) 3)) :test (v1= +nan+))
(check "rem(5,nan) = nan" t (vt-item (vt-rem (vt-const '() 5.0d0) +nan+)) :test (v1= +nan+))
(check "mod(5,tensor[0]) 逐元素零除 = [nan,nan]（v0.3.6：浮点除数语境）"
       (list +nan+ +nan+)
       (coerce (vt-data (vt-mod (vt-from-sequence '(5 7))
                                (vt-from-sequence '(0 0)))) 'list)
       :test (=* 1e-12))
;; 标量除数路径回归（原有行为不变）
(check "mod([5,6],3) = [2,0]（标量路径回归）" '(2.0d0 0.0d0)
       (coerce (vt-data (vt-mod (vt-from-sequence '(5 6)) 3)) 'list)
       :test (=* 1e-12))
(check "mod 广播: [5,6] mod [[3],[4]] → (2 2) [[2,0],[1,2]]" t
       (let ((r (vt-mod (vt-from-sequence '(5 6))
                        (vt-from-sequence '((3) (4))))))
         (and (equalp (list 2 2) (vt-shape r))
              (< (abs (- (vt-ref r 0 0) 2.0d0)) 1e-12)
              (< (abs (- (vt-ref r 0 1) 0.0d0)) 1e-12)
              (< (abs (- (vt-ref r 1 0) 1.0d0)) 1e-12)
              (< (abs (- (vt-ref r 1 1) 2.0d0)) 1e-12))))

;;; ================================================================
;;; 3. vt-insert：numpy 语义（重复索引顺序 / 负索引）
;;; ================================================================
(format t "~%===== 3. vt-insert numpy 语义 =====~%")
(check "insert dup idx '(1 1) [10,20] = [0,10,20,1,2]"
       '(0.0d0 10.0d0 20.0d0 1.0d0 2.0d0)
       (coerce (vt-data (vt-insert (vt-from-sequence '(0 1 2))
                                   '(1 1) (vt-from-sequence '(10 20)))) 'list)
       :test (=* 1e-12))
(check "insert dup idx '(1 1) [20,10] = [0,20,10,1,2]（保 values 顺序）"
       '(0.0d0 20.0d0 10.0d0 1.0d0 2.0d0)
       (coerce (vt-data (vt-insert (vt-from-sequence '(0 1 2))
                                   '(1 1) (vt-from-sequence '(20 10)))) 'list)
       :test (=* 1e-12))
(check "insert 负+正混合 [-1,0] [10,20] = [20,0,1,10,2]"
       '(20.0d0 0.0d0 1.0d0 10.0d0 2.0d0)
       (coerce (vt-data (vt-insert (vt-from-sequence '(0 1 2))
                                   '(-1 0) (vt-from-sequence '(10 20)))) 'list)
       :test (=* 1e-12))
(check "insert 单索引（原有路径回归）" '(0.0d0 10.0d0 20.0d0 1.0d0 2.0d0)
       (coerce (vt-data (vt-insert (vt-from-sequence '(0 1 2))
                                   1 (vt-from-sequence '(10 20)))) 'list)
       :test (=* 1e-12))
(check "insert 尾部插入（pos=size）" '(0.0d0 1.0d0 2.0d0 99.0d0)
       (coerce (vt-data (vt-insert (vt-from-sequence '(0 1 2))
                                   3 (vt-from-sequence '(99)))) 'list)
       :test (=* 1e-12))
;; axis 模式：重复位置块顺序
(check "axis insert rows [1 1] 保块顺序"
       (list 1.0d0 2.0d0 5.0d0 6.0d0 7.0d0 8.0d0 3.0d0 4.0d0)
       (coerce (vt-data (vt-insert (vt-from-array
                                    (make-array '(2 2) :initial-contents
                                                '((1.0d0 2.0d0) (3.0d0 4.0d0))))
                                   '(1 1)
                                   (vt-from-array
                                    (make-array '(2 2) :initial-contents
                                                '((5.0d0 6.0d0) (7.0d0 8.0d0))))
                                   :axis 0))
               'list)
       :test (=* 1e-12))
(check "axis insert 负索引 -1（插入到倒数第 1 行前，numpy 语义）"
       (list 1.0d0 2.0d0 9.0d0 9.0d0 3.0d0 4.0d0)
       (coerce (vt-data (vt-insert (vt-from-array
                                    (make-array '(2 2) :initial-contents
                                                '((1.0d0 2.0d0) (3.0d0 4.0d0))))
                                   -1 (vt-from-sequence '((9.0d0 9.0d0)))
                                   :axis 0))
               'list)
       :test (=* 1e-12))
(check-error "insert 越界索引报错"
             (vt-insert (vt-from-sequence '(0 1 2)) 4 (vt-from-sequence '(9))))

;;; ================================================================
;;; 4. vt-solve：b 形状校验 + 正确性回归
;;; ================================================================
(format t "~%===== 4. vt-solve 形状校验 =====~%")
(let ((a (vt-from-array (make-array '(2 2) :initial-contents
                                    '((2.0d0 1.0d0) (1.0d0 3.0d0))))))
  (check-error "solve b 行数不足 (1,2) → 确定性报错"
               (vt-solve a (vt-from-array (make-array '(1 2)
                                                      :initial-contents '((5.0d0 6.0d0))))))
  (check-error "solve b 行数超出 (3,1) → 确定性报错"
               (vt-solve a (vt-from-array (make-array '(3 1)
                                                      :initial-contents '((1.0d0) (2.0d0) (3.0d0))))))
  (check "solve 正确性回归: 2x2 唯一解" '(2.0d0 1.0d0)
         (coerce (vt-data (vt-solve a (vt-from-sequence '((5.0d0) (5.0d0))))) 'list)
         :test (lambda (e act)
                 (and (= (length e) (length act))
                      (every (lambda (x y) (< (abs (- x y)) 1e-9)) e act))))
  (check "solve 多右端项 (2,2)" t
         (let ((r (vt-solve a (vt-from-array
                               (make-array '(2 2) :initial-contents
                                           '((5.0d0 10.0d0) (5.0d0 10.0d0)))))))
           (and (equalp (vt-shape r) (list 2 2))
                (< (abs (- (vt-ref r 0 0) 2.0d0)) 1e-9)
                (< (abs (- (vt-ref r 1 0) 1.0d0)) 1e-9)))))

;;; ================================================================
;;; 5. vt-inner：0 维输入
;;; ================================================================
(format t "~%===== 5. vt-inner 0 维 =====~%")
(check "inner(3,4) = 12" t (vt-item (vt-inner 3 4)) :test (v1= 12.0d0))
(check "inner(scalar, [1,2]) = [3,6]" '(3.0d0 6.0d0)
       (coerce (vt-data (vt-inner 3 (vt-from-sequence '(1 2)))) 'list)
       :test (=* 1e-12))
(check "inner([1,2],3) = [3,6]" '(3.0d0 6.0d0)
       (coerce (vt-data (vt-inner (vt-from-sequence '(1 2)) 3)) 'list)
       :test (=* 1e-12))
(check "inner([1,2],[3,4]) = 11（原有路径回归）" t
       (vt-item (vt-inner (vt-from-sequence '(1 2)) (vt-from-sequence '(3 4))))
       :test (v1= 11.0d0))
(check "inner 2D×1D = [5,11]（最后一维收缩）" '(5.0d0 11.0d0)
       (coerce (vt-data (vt-inner (vt-from-array
                                   (make-array '(2 2) :initial-contents
                                               '((1.0d0 2.0d0) (3.0d0 4.0d0))))
                                  (vt-from-sequence '(1 2)))) 'list)
       :test (=* 1e-12))

;;; ================================================================
;;; 6. vt-tensordot：负轴归一化
;;; ================================================================
(format t "~%===== 6. vt-tensordot 负轴 =====~%")
(let ((a (vt-from-array (make-array '(2 3) :initial-contents
                                    '((1.0d0 2.0d0 3.0d0) (4.0d0 5.0d0 6.0d0)))))
      (b (vt-from-array (make-array '(3 2) :initial-contents
                                    '((1.0d0 2.0d0) (3.0d0 4.0d0) (5.0d0 6.0d0))))))
  (check "tensordot 负轴 [[-1],[0]] ≡ [[1],[0]]" t
         (let ((r (vt-tensordot a b :axes '((-1) (0)))))
           (and (equalp (vt-shape r) (list 2 2))
                (< (abs (- (vt-ref r 0 0) 22.0d0)) 1e-9)
                (< (abs (- (vt-ref r 1 1) 64.0d0)) 1e-9))))
  (check "tensordot 整数 axes=1 回归（= matmul, r[1][0]=49）" t
         (let ((r (vt-tensordot a b :axes 1)))
           (and (equalp (vt-shape r) (list 2 2))
                (< (abs (- (vt-ref r 1 0) 49.0d0)) 1e-9))))
  (check-error "tensordot 越界轴报错"
               (vt-tensordot a b :axes '((3) (0))))
  (check-error "tensordot 负轴越界报错"
               (vt-tensordot a b :axes '((-4) (0)))))

;;; ================================================================
;;; 7. vt-from-array：dtype 推断全集
;;; ================================================================
(format t "~%===== 7. vt-from-array dtype 推断 =====~%")
(check "(signed-byte 8) → :int8" :int8
       (vt-dtype (vt-from-array (make-array '(2) :element-type '(signed-byte 8)
                                            :initial-contents '(1 -2)))))
(check "(unsigned-byte 8) → :uint8" :uint8
       (vt-dtype (vt-from-array (make-array '(2) :element-type '(unsigned-byte 8)
                                            :initial-contents '(1 2)))))
(check "(signed-byte 16) → :int16" :int16
       (vt-dtype (vt-from-array (make-array '(2) :element-type '(signed-byte 16)
                                            :initial-contents '(1 -300)))))
(check "(unsigned-byte 16) → :uint16" :uint16
       (vt-dtype (vt-from-array (make-array '(2) :element-type '(unsigned-byte 16)
                                            :initial-contents '(1 65500)))))
(check "(signed-byte 32) → :int32" :int32
       (vt-dtype (vt-from-array (make-array '(2) :element-type '(signed-byte 32)
                                            :initial-contents '(1 70000)))))
(check "(signed-byte 64) → :int64" :int64
       (vt-dtype (vt-from-array (make-array '(2) :element-type '(signed-byte 64)
                                            :initial-contents '(1 5000000000)))))
(check "single-float → :float32" :float32
       (vt-dtype (vt-from-array (make-array '(2) :element-type 'single-float
                                            :initial-contents '(1.0s0 2.0s0)))))
(check "double-float → :float64" :float64
       (vt-dtype (vt-from-array (make-array '(2) :element-type 'double-float
                                            :initial-contents '(1.0d0 2.0d0)))))
(check "uint8 值保持不变" '(7 200)
       (coerce (vt-data (vt-from-array (make-array '(2) :element-type '(unsigned-byte 8)
                                                   :initial-contents '(7 200)))) 'list)
       :test #'equalp)
(check "显式 :dtype 仍可覆盖推断" :float64
       (vt-dtype (vt-from-array (make-array '(2) :element-type '(unsigned-byte 8)
                                            :initial-contents '(7 200))
                                :dtype :float64)))

;;; ================================================================
;;; 8. vt-cast-fun 一致性
;;; ================================================================
(format t "~%===== 8. vt-cast-fun 一致性 =====~%")
(check "cast-fun :uint16 走 %coerce-uint16" t
       (let ((f (vt-cast-fun :uint16)))
         ;; %coerce-uint16 对范围内整数恒等，越界回绕；验证大整数回绕行为
         (= (funcall f 70000) 4464)))
(check "vt-cast uint16 回绕回归" 4464 (vt-cast 70000 :uint16))

;;; ================================================================
;;; 汇总
;;; ================================================================
(format t "~%~%Total: ~a~%Pass: ~a~%Fail: ~a~%" (+ *pass* *fail*) *pass* *fail*)
(finish-output)
(sb-ext:exit :code (if (zerop *fail*) 0 1))
