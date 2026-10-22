;;;; complex-dispatch-test.lisp — 第十二轮：分派结构回归（16 断言）
;;;;
;;;; 这个套件断言的不是数值，而是**结构契约**：
;;;;
;;;;   1. 已迁移的公开符号，其 SYMBOL-FUNCTION 必须是**具名 DEFUN**，
;;;;      不能是 `(setf (symbol-function ...))` 装进去的匿名 LAMBDA。
;;;;      这一条是「重载单个源文件不会丢复数支持」的充分必要条件 ——
;;;;      匿名包装一旦被重载的实数 DEFUN 覆盖就永久消失，而具名 DEFUN
;;;;      自己就带分派，重载只会把它重建成同样的东西。
;;;;
;;;;   2. 分派判据必须认得 list 入参（%CX-ANY-P 走 %CX-DEEP-P）。
;;;;      曾经 (vt-sqrt (list #C(-1 0))) 会掉进实数路径返回 NaN float64。
;;;;
;;;;   3. 实数路径不得被复数分派改变语义（抽样对照）。

(require :asdf)
(push #p"/workspace/src/clvt-r12/" asdf:*central-registry*)
(handler-bind ((warning #'muffle-warning)) (asdf:load-system :clvt))
(in-package :clvt)

(defvar *pass* 0) (defvar *fail* 0)

(defun report (label ok)
  (if ok (incf *pass*) (incf *fail*))
  (format t "~&[~a] ~a~%" (if ok "PASS" "FAIL") label))

(defun defined-name (name)
  "符号当前函数对象的「定义名」：DEFUN 给符号，匿名 LAMBDA 给 NIL。"
  (let ((fn (and (fboundp name) (symbol-function name))))
    (and fn (nth-value 2 (function-lambda-expression fn)))))

;;; ------------------------------------------------------------------
;;; 1. 已迁移的 92 个符号必须是具名 DEFUN
;;; ------------------------------------------------------------------
(defparameter *migrated*
  '(;; S1 拒绝型（41）：守卫在归属文件里
    vt-atan2 vt-bincount vt-bit-and vt-bit-ior vt-bit-not vt-bit-xor
    vt-cbrt vt-ceiling vt-copysign vt-deg2rad vt-div vt-even-p
    vt-floor vt-gelu vt-hard-sigmoid vt-hard-tanh vt-histogram vt-hypot
    vt-interp vt-leaky-relu vt-left-shift vt-logaddexp vt-mish vt-mod
    vt-nanpercentile vt-nanquantile vt-nextafter vt-odd-p vt-percentile
    vt-quantile vt-rad2deg vt-relu vt-rem vt-right-shift vt-rint
    vt-sigmoid vt-signbit vt-softplus vt-spacing vt-swish vt-truncate
    ;; S2 一元超越（24）
    vt-acos vt-acosh vt-asin vt-asinh vt-atan vt-atanh vt-cos vt-cosh
    vt-exp vt-expm1 vt-log10 vt-log1p vt-log2 vt-negative vt-reciprocal
    vt-sign vt-signum vt-sin vt-sinc vt-sinh vt-sqrt vt-square vt-tan vt-tanh
    ;; S3 比较 / 谓词 / 极值（13）
    vt-= vt-/= vt-< vt-<= vt-> vt->= vt-isnan vt-isinf vt-isfinite
    vt-maximum vt-minimum vt-fmax vt-fmin
    ;; S4 归约族（14，含宏生成的门面 10 个）
    vt-sum vt-prod vt-amax vt-amin
    vt-nansum vt-nanprod vt-nanmax vt-nanmin
    vt-any vt-all vt-argmax vt-argmin vt-nanargmax vt-nanargmin
    ;; S5-A 顶层单 IF 形态（29）
    vt-convolve vt-corrcoef vt-correlate vt-digitize vt-in1d vt-inner
    vt-intersect1d vt-isin vt-nancumprod vt-nancumsum vt-nanmean vt-nanstd
    vt-nanvar vt-repeat vt-setdiff1d vt-setxor1d vt-tensordot vt-union1d
    vt-vdot vt-argsort vt-cumprod vt-cumsum vt-dot vt-matmul vt-median
    vt-outer vt-round vt-sort vt-trace
    ;; S5-B1 LET 包裹形态（6）
    vt-abs vt-pow vt-cov vt-positive vt-standardize vt-searchsorted
    ;; S5-B2 LET + UNLESS 早退形态（5）
    vt-mean vt-norm vt-std vt-var vt-nan-to-num
    ;; S5-C 线代（11）
    vt-cholesky vt-det vt-eig vt-inv vt-lstsq vt-lu vt-matrix-power
    vt-pinv vt-qr vt-svd vt-solve))

(report "S1–S5C 共 143 个符号已登记" (= 143 (length *migrated*)))

(let (bad)
  (dolist (n *migrated*)
    (unless (eq (defined-name n) n)
      (push (list n (defined-name n)) bad)))
  (report (format nil "143 个符号全部是具名 DEFUN（非匿名包装）；异常: ~a"
                  (if bad (length bad) 0))
          (null bad))
  (when bad
    (format t "~&  异常清单（前 10）: ~a~%" (subseq bad 0 (min 10 (length bad))))))

;;; ------------------------------------------------------------------
;;; 2. 分派判据的 list 入参
;;; ------------------------------------------------------------------
(report "list 入参走复数路径：sqrt(list #C(-1 0)) → complex128 的 i"
        (let ((r (vt-sqrt (list #C(-1.0d0 0.0d0)))))
          (and (eq (vt-dtype r) :complex128)
               (let ((z (aref (vt-data (vt-contiguous (vt-ravel r)))
                              (vt-offset (vt-contiguous (vt-ravel r))))))
                 (and (< (abs (realpart z)) 1d-12)
                      (< (abs (- (imagpart z) 1.0d0)) 1d-12))))))

(report "list 入参的拒绝型守卫：cbrt(list #C(8 0)) 报错"
        (handler-case (progn (vt-cbrt (list #C(8.0d0 0.0d0))) nil)
          (error (e) (search "不支持复数" (format nil "~a" e)))))

;;; ------------------------------------------------------------------
;;; 3. 实数路径未被分派改变（抽样）
;;; ------------------------------------------------------------------
(report "实数 sqrt 不回归：sqrt(4)=2、dtype float64"
        (let ((r (vt-sqrt (vt-full '(1) 4.0d0))))
          (and (eq (vt-dtype r) :float64)
               (< (abs (- (vt-item r) 2.0d0)) 1d-12))))

(report "实数 sum 不回归：(1 2 3) → 6.0"
        (< (abs (- (vt-item (vt-sum (vt-from-sequence (list 1.0d0 2.0d0 3.0d0))))
                   6.0d0))
           1d-12))

(report "实数比较仍返回 int8"
        (eq (vt-dtype (vt-< (vt-from-sequence (list 1.0d0)) (vt-from-sequence (list 2.0d0))))
            :int8))

(report "实数 maximum 仍走实数语义（非字典序）"
        (< (abs (- (vt-item (vt-maximum (vt-from-sequence (list -1.0d0))
                                        (vt-from-sequence (list 2.0d0))))
                   2.0d0))
           1d-12))

;;; ------------------------------------------------------------------
;;; 4. 复数语义抽样（与整改前一致）
;;; ------------------------------------------------------------------
(report "复数 maximum 是字典序：1+2i vs 0+9i → 1+2i"
        (let ((r (vt-maximum (vt-from-sequence (list #C(1.0d0 2.0d0)))
                             (vt-from-sequence (list #C(0.0d0 9.0d0))))))
          (and (< (abs (- (aref (vt-data (vt-contiguous (vt-ravel r)))
                                (vt-offset (vt-contiguous (vt-ravel r))))
                           #C(1.0d0 2.0d0)))
                  1d-12))))

(report "复数 any：0+0i → 0、(nan+0i) → 1"
        (let ((z0 (vt-any (vt-from-sequence (list #C(0.0d0 0.0d0)))))
              (zn (vt-any (vt-from-sequence
                           (list (complex (vt-get-nan :float64) 0.0d0))))))
          (and (= 0 (vt-item z0)) (= 1 (vt-item zn)))))

(report "复数 argmax 返回 int64"
        (eq (vt-dtype (vt-argmax (vt-from-sequence (list #C(1.0d0 0.0d0)
                                                         #C(2.0d0 0.0d0)))))
            :int64))

(report "复数 sum 保持 complex64（不被提升）"
        (eq (vt-dtype (vt-sum (vt-astype (vt-ones (list 3)) :complex64))) :complex64))

(report "拒绝型：实数路径照常工作（cbrt(8)=2、relu(-1)=0）"
        (and (< (abs (- (vt-item (vt-cbrt (vt-full '(1) 8.0d0))) 2.0d0)) 1d-12)
             (< (abs (vt-item (vt-relu (vt-full '(1) -1.0d0)))) 1d-12)))

(report "复数归约的 :out 契约仍生效"
        (let ((o (vt-zeros (list 4) :dtype :complex128)))
          (vt-sum (vt-astype (vt-ones (list 3 4)) :complex128) :axis 0 :out o)
          (every (lambda (x) (< (abs (- x (complex 3.0d0 0.0d0))) 1d-12))
                 (coerce (vt-data o) 'list))))

(report "argmax 全 NaN 不崩、返回首个 NaN 下标"
        (let ((r (vt-argmax (vt-from-sequence
                             (list (complex (vt-get-nan :float64) 0.0d0)
                                   #C(9.0d0 0.0d0))))))
          (= 0 (vt-item r))))

(format t "~%~%PASS=~a FAIL=~a~%" *pass* *fail*)
(sb-ext:exit :code (if (zerop *fail*) 0 1))
