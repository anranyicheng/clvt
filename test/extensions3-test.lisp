;;;; extensions3-test.lisp — 测试 extensions3.lisp 新增的 NumPy 缺失函数
;;;; （输出与 run-tests.sh 兼容的标准格式）
(require :asdf)
(push (truename (make-pathname :directory '(:relative :up))) asdf:*central-registry*)
(handler-bind ((warning #'muffle-warning)) (asdf:load-system :clvt))
(in-package :clvt)

(defvar *N* 0) (defvar *P* 0) (defvar *F* 0) (defvar *F-list* nil)

(defun nan-p (x)
  (and (floatp x) (sb-int:with-float-traps-masked (:invalid)
                     (not (= x x)))))

(defun ->list (x)
  (cond ((vt-p x) (->list (vt-to-list x)))
        ((nan-p x) (if (typep x 'double-float) x (sb-int:with-float-traps-masked (:invalid) (float x 1d0))))
        ((numberp x) (if (floatp x) (sb-int:with-float-traps-masked (:invalid) (float x 1d0)) (float x 1d0)))
        ((consp x) (mapcar #'->list x))
        (t x)))

(defun approx (e a &optional (tol 1d-9))
  (let ((el (->list e)) (al (->list a)))
    (cond ((and (numberp el) (numberp al))
           (cond ((nan-p el) (nan-p al))
                 ((nan-p al) nil)
                 ((and (floatp el) (> (abs el) 1d300)) (= (signum el) (signum al)))
                 (t (< (abs (- el al)) tol))))
          ((and (consp el) (consp al))
           (and (= (length el) (length al))
                (every (lambda (x y) (approx x y tol)) el al)))
          (t (equalp el al)))))

(defun check (name expected actual &optional (tol 1d-9))
  (incf *N*)
  (if (approx expected actual tol) (incf *P*)
      (progn (incf *F*) (push name *F-list*)
             (format t "  ❌ ~a~%     exp: ~a~%     got: ~a~%~%" name
                     (let ((e (->list expected)))
                       (if (consp e) (subseq e 0 (min 8 (length e))) e))
                     (let ((g (->list actual)))
                       (if (consp g) (subseq g 0 (min 8 (length g))) g))))))

(defun check-true (name actual)
  (incf *N*)
  (if actual (incf *P*)
      (progn (incf *F*) (push name *F-list*) (format t "  ❌ ~a (期望真，实得假)~%" name))))

(defun check-false (name actual)
  (incf *N*)
  (if (not actual) (incf *P*)
      (progn (incf *F*) (push name *F-list*) (format t "  ❌ ~a (期望假，实得真)~%" name))))

(defun check-error (name thunk)
  (incf *N*)
  (handler-case (progn (funcall thunk) (incf *F*) (push name *F-list*)
                       (format t "  ❌ ~a (期望报错，未报)~%" name))
    (error () (incf *P*))))

(defun mk (data &key (dtype :float64))
  (labels ((f (x) (cond ((numberp x) (float x 1d0)) ((consp x) (mapcar #'f x)) (t x)))
           (sh (x) (if (consp x) (cons (length x) (sh (car x))) nil)))
    (let* ((fd (f data)) (s (sh fd)))
      (vt-from-array (make-array s :element-type 'double-float :initial-contents fd) :dtype dtype))))

(defun mki (data)
  (labels ((sh (x) (if (consp x) (cons (length x) (sh (car x))) nil)))
    (let ((s (sh data)))
      (vt-from-array (make-array s :element-type '(signed-byte 64) :initial-contents data) :dtype :int64))))

(defun summary ()
  (format t "~%============================================================~%")
  (format t "  Total: ~a | Pass: ~a | Fail: ~a | Skip: 0~%" *N* *P* *F*)
  (format t "============================================================~%")
  (when *F-list*
    (format t "~%Failed:~{~%  - ~a~}~%" (reverse *F-list*)))
  (zerop *F*))

(format t "~%=== Extensions3 补充 NumPy 缺失函数测试 ===~%~%")

(defvar *nan* (vt-get-nan :float64))
(defvar *pinf* (vt-get-pos-inf :float64))
(defvar *ninf* (vt-get-neg-inf :float64))

;;; ============================================================
;;; 1. 创建类
;;; ============================================================
(format t "--- 创建类 ---~%")

(let ((a (vt-asarray '(1 2 3))))
  (check "asarray 值" '(1 2 3) a)
  (check "asarray dtype" :int64 (vt-dtype a))
  (check "asarray scalar shape" nil (vt-shape (vt-asarray 5)))
  (check "asarray dtype 显式" :float32 (vt-dtype (vt-asarray '(1 2 3) :dtype :float32)))
  (check "asarray 透传 vt" :float64 (vt-dtype (vt-asarray (mk '(1d0 2d0))))))

(check "fromiter" '(1d0 2d0 3d0) (vt-fromiter '(1 2 3) :dtype :float64))
(check "fromiter 空" 0 (vt-size (vt-fromiter '() :dtype :float64)))
(check "fromiter count" '(1 2 3) (vt-fromiter '(1 2 3) :dtype :int64 :count 3))

(check "tri 3" '((1d0 0d0 0d0) (1d0 1d0 0d0) (1d0 1d0 1d0)) (vt-tri 3))
(check "tri 3 k=1" '((1d0 1d0 0d0) (1d0 1d0 1d0) (1d0 1d0 1d0)) (vt-tri 3 :k 1))
(check "tri 3,4,-1" '((0d0 0d0 0d0 0d0) (1d0 0d0 0d0 0d0) (1d0 1d0 0d0 0d0)) (vt-tri 3 :m 4 :k -1))
(check "tri dtype" :int64 (vt-dtype (vt-tri 3 :dtype :int64)))
(check "tri 0" '(0 0) (vt-shape (vt-tri 0)))

(check "diagflat 1d" '((1d0 0d0 0d0) (0d0 2d0 0d0) (0d0 0d0 3d0)) (vt-diagflat '(1 2 3)))
(check "diagflat 2d" '((1d0 0d0 0d0 0d0) (0d0 2d0 0d0 0d0) (0d0 0d0 3d0 0d0) (0d0 0d0 0d0 4d0))
       (vt-diagflat '((1 2) (3 4))))
(check "diagflat k=1" '((0d0 1d0 0d0) (0d0 0d0 2d0) (0d0 0d0 0d0)) (vt-diagflat '(1 2) :k 1))

(check "trim-zeros" '(1 2 0 3) (vt-trim-zeros (mki '(0 0 1 2 0 3 0 0))))
(check "trim-zeros f" '(1 2 0 3 0 0) (vt-trim-zeros (mki '(0 0 1 2 0 3 0 0)) :trim :f))
(check "trim-zeros b" '(0 0 1 2 0 3) (vt-trim-zeros (mki '(0 0 1 2 0 3 0 0)) :trim :b))
(check "trim-zeros 全零" 0 (vt-size (vt-trim-zeros (mki '(0 0 0)))))
(check "trim-zeros dtype" :int64 (vt-dtype (vt-trim-zeros (mki '(0 1 0)))))

;;; ============================================================
;;; 2. 形状操作类
;;; ============================================================
(format t "--- 形状操作类 ---~%")

(check "rollaxis a,2" '(5 3 4) (vt-shape (vt-rollaxis (vt-ones '(3 4 5)) 2)))
(check "rollaxis a,2,0" '(5 3 4) (vt-shape (vt-rollaxis (vt-ones '(3 4 5)) 2 0)))
(check "rollaxis a,1" '(4 3 5) (vt-shape (vt-rollaxis (vt-ones '(3 4 5)) 1)))
(check "rollaxis a,0,3" '(4 5 3) (vt-shape (vt-rollaxis (vt-ones '(3 4 5)) 0 3)))
(check-error "rollaxis start 越界" (lambda () (vt-rollaxis (vt-ones '(3 4 5)) 1 9)))

(check "column-stack" '((1d0 4d0) (2d0 5d0) (3d0 6d0))
       (vt-column-stack (list (mk '(1d0 2d0 3d0)) (mk '(4d0 5d0 6d0)))))
(check "column-stack 单" '((1) (2) (3))
       (vt-column-stack (list (mki '(1 2 3)))))

(check "block 2x2" '((1d0 1d0 0d0) (1d0 1d0 0d0) (0d0 0d0 1d0))
       (vt-block (list (list (vt-ones '(2 2)) (vt-zeros '(2 1)))
                       (list (vt-zeros '(1 2)) (vt-ones '(1 1))))))
(check "block 1d" '(1 2 3) (vt-block (list (mki '(1)) (mki '(2)) (mki '(3)))))

(let ((ba (vt-broadcast-arrays (list (vt-ones '(3 1)) (vt-ones '(1 4))))))
  (check "broadcast-arrays len" 2 (length ba))
  (check "broadcast-arrays shape0" '(3 4) (vt-shape (first ba)))
  (check "broadcast-arrays shape1" '(3 4) (vt-shape (second ba))))

(check "resize 2x3" '((1d0 2d0 3d0) (1d0 2d0 3d0)) (vt-resize (mk '(1d0 2d0 3d0)) '(2 3)))
(check "resize 5" '(1d0 2d0 3d0 1d0 2d0) (vt-resize (mk '(1d0 2d0 3d0)) '(5)))
(check "resize 2" '(1d0 2d0) (vt-resize (mk '(1d0 2d0 3d0)) '(2)))
(check "resize 空源" 0 (vt-size (vt-resize (vt-fromiter '() :dtype :float64) '(0))))
(check "resize dtype" :int64 (vt-dtype (vt-resize (mki '(1 2 3)) '(5))))

;;; ============================================================
;;; 3. 索引类
;;; ============================================================
(format t "--- 索引类 ---~%")

(check "take-along-axis" '((1d0) (4d0))
       (vt-take-along-axis (mk '((1d0 2d0) (3d0 4d0))) (mki '((0) (1))) 1))
(check "take-along-axis 轴0" '((1d0 2d0) (3d0 4d0))
       (vt-take-along-axis (mk '((1d0 2d0) (3d0 4d0))) (mki '((0) (1))) 0))
(check-error "take-along-axis 秩不符"
             (lambda () (vt-take-along-axis (mk '(1d0 2d0 3d0)) (mk '((1d0 2d0))) 0)))

(check "put-along-axis" '((9d0 2d0) (3d0 9d0))
       (vt-put-along-axis (mk '((1d0 2d0) (3d0 4d0))) (mki '((0) (1))) (mk '((9d0) (9d0))) 1))
(let ((src (mk '((1d0 2d0) (3d0 4d0)))))
  (vt-put-along-axis src (mki '((0) (0))) (mk '((9d0) (9d0))) 1)
  (check "put-along-axis 不改输入" '((1d0 2d0) (3d0 4d0)) src))

(check "compress 展平" '(0d0 2d0 4d0) (vt-compress '(1 0 1 0 1) (mk '(0d0 1d0 2d0 3d0 4d0))))
(check "compress axis0" '((1d0 2d0)) (vt-compress '(1 0) (mk '((1d0 2d0) (3d0 4d0))) :axis 0))
(check "compress axis1" '((1d0) (3d0)) (vt-compress '(1 0) (mk '((1d0 2d0) (3d0 4d0))) :axis 1))
(check "compress 短布尔" '(0d0 2d0) (vt-compress '(1 0 1) (mk '(0d0 1d0 2d0 3d0 4d0))))

(check "indices 形状" '(2 2 3) (vt-shape (vt-indices '(2 3))))
(check "indices 值" '(((0 0 0) (1 1 1)) ((0 1 2) (0 1 2))) (vt-indices '(2 3)))
(let ((sp (vt-indices '(2 3) :sparse t)))
  (check "indices sparse len" 2 (length sp))
  (check "indices sparse 形状0" '(2 1) (vt-shape (first sp)))
  (check "indices sparse 形状1" '(1 3) (vt-shape (second sp)))
  (check "indices sparse 值" '((0) (1)) (first sp)))
(check-error "indices 空" (lambda () (vt-indices '())))

(check "fill-diagonal 3x3" '((9d0 0d0 0d0) (0d0 9d0 0d0) (0d0 0d0 9d0))
       (vt-fill-diagonal (vt-zeros '(3 3)) 9))
(check "fill-diagonal 矩形" '((5d0 0d0 0d0 0d0) (0d0 5d0 0d0 0d0) (0d0 0d0 5d0 0d0))
       (vt-fill-diagonal (vt-zeros '(3 4)) 5))
(check "fill-diagonal 序列" '((1d0 0d0 0d0) (0d0 2d0 0d0) (0d0 0d0 3d0))
       (vt-fill-diagonal (vt-zeros '(3 3)) (mk '(1d0 2d0 3d0))))
(check "fill-diagonal 返回原对象" '((7d0 0d0) (0d0 7d0))
       (vt-fill-diagonal (vt-zeros '(2 2)) 7))

;;; ============================================================
;;; 4. 数学类
;;; ============================================================
(format t "--- 数学类 ---~%")

(check "absolute 整型" '(3 2 1) (vt-absolute (mki '(-3 -2 1))))
(check "absolute dtype" :int64 (vt-dtype (vt-absolute (mki '(-3 -2)))))
(check "absolute 浮点" '(3d0 2d0 1d0) (vt-absolute (mk '(-3d0 -2d0 1d0))))
(check "absolute NaN" (list *nan* 2d0) (vt-absolute (list *nan* -2d0)))

(check "sign 整型" '(-1 0 1) (vt-sign (mki '(-3 0 5))))
(check "sign dtype" :int64 (vt-dtype (vt-sign (mki '(-3 0 5)))))
(check "sign 浮点 NaN" (list -1d0 *nan* 1d0) (vt-sign (list -3d0 *nan* 5d0)))
(check "sign -0.0" '(0d0) (vt-sign (list -0.0d0)))

(check "positive" '(1d0 -2d0 3d0) (vt-positive (mk '(1d0 -2d0 3d0))))
(check "positive 整型" '(1 -2 3) (vt-positive (mki '(1 -2 3))))

(check "expm1 0" '(0d0) (vt-expm1 '(0.0d0)))
(check "expm1 小量" '(1.00000000005d-10) (vt-expm1 '(1.0d-10)))
(check "expm1 1" '(1.718281828459045d0) (vt-expm1 '(1.0d0)))
(check "log1p 0" '(0d0) (vt-log1p '(0.0d0)))
(check "log1p 小量" '(9.999999999500001d-11) (vt-log1p '(1.0d-10)))
(check "log1p -1" (list *ninf*) (vt-log1p '(-1.0d0)))
(check "log1p 1" '(0.6931471805599453d0) (vt-log1p '(1.0d0)))

(check "logaddexp 0,0" '(0.6931471805599453d0) (vt-logaddexp '(0d0) '(0d0)))
(check "logaddexp 大数" '(1d10) (vt-logaddexp '(1d10) '(1d-10)))
(check "logaddexp inf" (list *pinf*) (vt-logaddexp (list *pinf*) '(1d0)))

(check "float-power" '(4d0 9d0) (vt-float-power '(2 3) 2))
(check "float-power dtype" :float64 (vt-dtype (vt-float-power '(2 3) 2)))
(check "float-power 负底" (list *nan*) (vt-float-power '(-1.0d0) '(0.5d0)))

(check "copysign" '(-1d0 2d0 -3d0) (vt-copysign '(1 -2 3) '(-1 1 -1)))

(check "signbit" '(1 0 0) (vt-signbit '(-1 0 1)))
(check "signbit dtype" :int8 (vt-dtype (vt-signbit '(-1 0 1))))
(check "signbit -0.0" '(1) (vt-signbit (list -0.0d0)))

(check "nextafter 1->2" '(1.0000000000000002d0) (vt-nextafter '(1.0d0) '(2.0d0)))
(check "nextafter 0->1" '(4.9406564584124654d-324) (vt-nextafter '(0.0d0) '(1.0d0)))
(check "nextafter 1->0" '(0.9999999999999999d0) (vt-nextafter '(1.0d0) '(0.0d0)))

(check "spacing 1" '(2.220446049250313d-16) (vt-spacing '(1.0d0)))
(check "spacing 0" '(4.9406564584124654d-324) (vt-spacing '(0.0d0)))

(check "gcd" '(4 6) (vt-gcd '(12 18) '(8 24)))
(check "gcd 零" '(4 5) (vt-gcd '(0 5) '(4 0)))
(check "gcd dtype" :int64 (vt-dtype (vt-gcd (mki '(12)) (mki '(8)))))
(check-error "gcd 浮点报错" (lambda () (vt-gcd '(2.5d0) '(1))))

(check "lcm" '(12 24) (vt-lcm '(4 6) '(6 8)))
(check "lcm 零" '(0) (vt-lcm '(0) '(5)))

(check "divmod 商" '(3) (nth 0 (multiple-value-list (vt-divmod '(7) '(2)))))
(check "divmod 余" '(1) (nth 1 (multiple-value-list (vt-divmod '(7) '(2)))))
(check "divmod 负" '(-3) (nth 0 (multiple-value-list (vt-divmod '(-7) '(3)))))
(check "divmod 负余" '(2) (nth 1 (multiple-value-list (vt-divmod '(-7) '(3)))))

(check "nan-to-num NaN" '(0d0) (vt-nan-to-num (list *nan*)))
(check "nan-to-num 常规" '(1d0) (vt-nan-to-num '(1.0d0)))
(check "nan-to-num posinf 自定义" '(100d0) (vt-nan-to-num (list *pinf*) :posinf 100.0d0))
(check "nan-to-num 默认大数"
       (list most-positive-double-float)
       (vt-nan-to-num (list *pinf*)))

(check "real" '(1d0 2d0) (vt-real '(1.0d0 2.0d0)))
(check "imag" '(0d0 0d0) (vt-imag '(1.0d0 2.0d0)))
(check "conj" '(1d0 2d0) (vt-conj '(1.0d0 2.0d0)))

(check "angle 正负零" (list 0d0 (coerce pi 'double-float) 0d0) (vt-angle '(1.0d0 -1.0d0 0.0d0)))
(check "angle deg" '(0d0 180d0) (vt-angle '(1.0d0 -1.0d0) :deg t))
(check "angle NaN" (list *nan*) (vt-angle (list *nan*)))

;;; ============================================================
;;; 5. 统计类
;;; ============================================================
(format t "--- 统计类 ---~%")

(check "nancumsum 全局" '(1d0 1d0 4d0) (vt-nancumsum (list 1.0d0 *nan* 3.0d0)))
(check "nancumprod 全局" '(1d0 1d0 3d0) (vt-nancumprod (list 1.0d0 *nan* 3.0d0)))
(let ((m (vt-zeros '(2 2))))
  (setf (vt-ref m 0 0) 1d0) (setf (vt-ref m 0 1) *nan*)
  (setf (vt-ref m 1 0) 2d0) (setf (vt-ref m 1 1) 4d0)
  (check "nancumsum axis0" '((1d0 0d0) (3d0 4d0)) (vt-nancumsum m :axis 0))
  (check "nancumprod axis0" '((1d0 1d0) (2d0 4d0)) (vt-nancumprod m :axis 0)))

(check "nanpercentile 50" 3d0 (vt-nanpercentile (list 1.0d0 *nan* 3.0d0 4.0d0) 50))
(check "nanpercentile 0" 1d0 (vt-nanpercentile (list 1.0d0 *nan* 3.0d0 4.0d0) 0))
(check "nanpercentile 100" 4d0 (vt-nanpercentile (list 1.0d0 *nan* 3.0d0 4.0d0) 100))
(check "nanpercentile 全 NaN" *nan*
       (vt-nanpercentile (list *nan* *nan*) 50))
(check "nanpercentile axis" '(1.5d0 2d0)
       (vt-nanpercentile (mk '((1d0 0d0) (2d0 4d0))) 50 :axis 0))
(check "nanquantile 0.5" 3d0 (vt-nanquantile (list 1.0d0 *nan* 3.0d0 4.0d0) 0.5))
(check-error "nanquantile q 越界" (lambda () (vt-nanquantile '(1d0) 1.5)))

(check "cov" '((1d0 1d0) (1d0 1d0)) (vt-cov (mk '((1d0 2d0 3d0) (4d0 5d0 6d0)))))
(check "cov 1d" 1d0 (vt-cov (mk '(1d0 2d0 3d0))))
(check "cov rowvar nil" '((1d0 2d0) (2d0 4d0))
       (vt-cov (mk '((1d0 2d0) (2d0 4d0) (3d0 6d0))) :rowvar nil))
(check "corrcoef" '((1d0 1d0) (1d0 1d0)) (vt-corrcoef (mk '((1d0 2d0 3d0) (4d0 5d0 6d0)))))

(check "cross 3d" '(-3 6 -3) (vt-cross (mki '(1 2 3)) (mki '(4 5 6))))
(check "cross 2d" -3 (vt-cross (mki '(1 2)) (mki '(4 5))))
(check "cross 批量" '((0 0 0) (0 0 0))
       (vt-cross (mki '((1 1 1) (1 1 1))) (mki '((1 1 1) (1 1 1)))))
(check-error "cross 分量错" (lambda () (vt-cross (mki '(1 2 3 4)) (mki '(1 2 3 4)))))

;;; ============================================================
;;; 6. 线性代数类
;;; ============================================================
(format t "--- 线性代数类 ---~%")

(check "vdot 1d" 32 (vt-vdot '(1 2 3) '(4 5 6)))
(check "vdot 2d 展平" 70 (vt-vdot (mki '((1 2) (3 4))) (mki '((5 6) (7 8)))))

(check "eigvals 对角" '(3d0 2d0) (vt-eigvals (mk '((2d0 0d0) (0d0 3d0)))))
;; v0.4.0：eigvalsh 已按 numpy.linalg.eigvalsh 约定升序排列
(check "eigvalsh 对称" '(1d0 3d0) (vt-eigvalsh (mk '((2d0 1d0) (1d0 2d0)))))

(check "matrix-power 2" '((7 10) (15 22)) (vt-matrix-power (mki '((1 2) (3 4))) 2))
(check "matrix-power 0" '((1 0) (0 1)) (vt-matrix-power (mki '((1 2) (3 4))) 0))
(check "matrix-power 1" '((1 2) (3 4)) (vt-matrix-power (mki '((1 2) (3 4))) 1))

(check "cond 单位阵" 1d0 (vt-cond (vt-eye 3)))
(check "cond 1-范数" 1.5d0 (vt-cond (mk '((1d0 2d0) (3d0 4d0))) :p 1))
(check "cond inf 范数" 2.3333333333333335d0 (vt-cond (mk '((1d0 2d0) (3d0 4d0))) :p :inf))

(check "multi-dot 形状" '(2 2)
       (vt-shape (vt-multi-dot (list (vt-ones '(2 3)) (vt-ones '(3 4)) (vt-ones '(4 2))))))
(check "multi-dot 2矩阵" '((19 22) (43 50))
       (vt-multi-dot (list (mki '((1 2) (3 4))) (mki '((5 6) (7 8))))))
(check-error "multi-dot 不足" (lambda () (vt-multi-dot (list (mki '(1 2))))))
(check-error "multi-dot 维度错"
             (lambda () (vt-multi-dot (list (vt-ones '(2 3)) (vt-ones '(4 5))))))

;;; ============================================================
;;; 7. 逻辑类
;;; ============================================================
(format t "--- 逻辑类 ---~%")

(check-true "array-equal" (vt-array-equal '(1 2) '(1 2)))
(check-false "array-equal 形状异" (vt-array-equal '(1 2) '((1 2))))
(check-false "array-equal NaN" (vt-array-equal (list 1 *nan*) (list 1 *nan*)))
(check-true "array-equal 2d" (vt-array-equal (mki '((1 2) (3 4))) (mki '((1 2) (3 4)))))

(check-true "array-equiv 广播" (vt-array-equiv '(1 2) '((1 2) (1 2))))
(check-true "array-equiv 单例" (vt-array-equiv '(1 2) '(1 2)))
(check-false "array-equiv 不等" (vt-array-equiv '(1 2) '(1 3)))

(check "isposinf" '(1 0 0) (vt-isposinf (list *pinf* 1.0d0 *ninf*)))
(check "isposinf dtype" :int8 (vt-dtype (vt-isposinf '(1d0))))
(check "isneginf" '(0 0 1) (vt-isneginf (list *pinf* 1.0d0 *ninf*)))
(check "isposinf NaN" '(0) (vt-isposinf (list *nan*)))

;;; ============================================================
;;; 8. 集合类
;;; ============================================================
(format t "--- 集合类 ---~%")

(check "isin" '(0 1 0 1) (vt-isin (mki '(1 2 3 4)) (mki '(2 4))))
(check "isin invert" '(1 0 1 0) (vt-isin (mki '(1 2 3 4)) (mki '(2 4)) :invert t))
(check "isin 2d" '((0 1) (0 1)) (vt-isin (mki '((1 2) (3 4))) (mki '(2 4))))
(check "isin dtype" :int8 (vt-dtype (vt-isin (mki '(1 2)) (mki '(2)))))
(check "isin 空" '(0 0 0) (vt-isin (mki '(1 2 3)) (vt-fromiter '() :dtype :int64)))

;;; ============================================================
;;; 9. :out 契约测试（CONVENTIONS §4）
;;; ============================================================
(format t "--- :out 契约 ---~%")

(let ((o (vt-zeros '(3) :dtype :int64)))
  (check "out absolute" '(3 2 1) (vt-absolute (mki '(-3 -2 1)) :out o)))
(let ((o (vt-zeros '(3) :dtype :float64)))
  (check "out expm1" (list 0d0 1.718281828459045d0 *nan*)
         (vt-expm1 (list 0.0d0 1.0d0 (vt-get-nan :float64)) :out o)))
(check-error "out dtype 不匹配"
             (lambda () (vt-absolute (mki '(-3 -2 1)) :out (vt-zeros '(3) :dtype :float64))))
(check-error "out 形状不匹配"
             (lambda () (vt-sign (mki '(-3 -2 1)) :out (vt-zeros '(2) :dtype :int64))))
(let ((o (vt-zeros '(5) :dtype :float64)))
  (check "out nancumsum" '(1d0 1d0 4d0 8d0 13d0)
         (vt-nancumsum (list 1.0d0 *nan* 3.0d0 4.0d0 5.0d0) :out o)))
(check-error "out nancumsum 形状错"
             (lambda () (vt-nancumsum (list 1.0d0 *nan* 3.0d0) :out (vt-zeros '(2) :dtype :float64))))

;;; ============================================================
;;; 10. 非连续输入测试（C7）
;;; ============================================================
(format t "--- 非连续输入 ---~%")

(let ((base (mki '((1 2 3) (4 5 6)))))
  ;; base 沿 axis0 取第 1 行 -> 连续；转置后非连续
  (check "absolute 转置" '((1 4) (2 5) (3 6))
         (vt-absolute (vt-transpose base)))
  (check "sign 转置" '((1 1) (1 1) (1 1))
         (vt-sign (vt-transpose base)))
  (check "expm1 非连续" '((0d0 0d0) (0d0 0d0))
         (vt-expm1 (vt-zeros '(2 2))))
  (check "isin 转置" '((0 0) (1 1) (0 0))
         (vt-isin (vt-transpose base) (mki '(2 5)))))

;;; ============================================================
;;; 11. dtype 提升测试（C2）
;;; ============================================================
(format t "--- dtype 提升 ---~%")

(check "gcd dtype int32" :int32 (vt-dtype (vt-gcd (vt-fromiter '(12) :dtype :int32)
                                                   (vt-fromiter '(8) :dtype :int32))))
(check "absolute int32" :int32 (vt-dtype (vt-absolute (vt-fromiter '(-3) :dtype :int32))))
(check "sign int8" :int8 (vt-dtype (vt-sign (vt-fromiter '(-3) :dtype :int8))))
(check "float-power 恒 float64" :float64 (vt-dtype (vt-float-power (vt-fromiter '(2) :dtype :int8) 2)))
(check "logaddexp 恒 float64" :float64 (vt-dtype (vt-logaddexp (vt-fromiter '(1) :dtype :int8) 1)))

;;; ============================================================
;;; 12. NaN/Inf 语义（C5）
;;; ============================================================
(format t "--- NaN/Inf 语义 ---~%")

(check "nancumsum 全 NaN" '(0d0 0d0) (vt-nancumsum (list *nan* *nan*)))
(check "any-nan sign" (list *nan*) (vt-sign (list *nan*)))
(check "copysign NaN 符号" '(1d0) (vt-copysign (list 1.0d0) (list *nan*)))
(check "isposinf nan" '(0) (vt-isposinf (list *nan*)))
(check "spacing nan" (list *nan*) (vt-spacing (list *nan*)))

;;; ============================================================
;;; 13. 错误路径（L2 底线）
;;; ============================================================
(format t "--- 错误路径 ---~%")

(check-error "tri 负数" (lambda () (vt-tri -1)))
(check-error "trim-zeros 非法 trim" (lambda () (vt-trim-zeros '(0d0) :trim :x)))
(check-error "nanpercentile 越界" (lambda () (vt-nanpercentile '(1d0) 150)))
(check-error "cond p 非法" (lambda () (vt-cond (vt-eye 2) :p 3)))
(check-error "fill-diagonal 1d" (lambda () (vt-fill-diagonal (vt-zeros '(3)) 1)))

;;; ============================================================
;;; 审查修复回归测试（clvt-review-report.md）
;;; ============================================================
(format t "~%--- 审查修复回归 ---~%")

;; [P0-1] vt-relu 整数输入曾触发段错误（%relu-double 类型声明被违背）。
;;       现要求整数输入正常提升为 float64，且数值与 numpy 一致。
(check "P0-1 relu int64"  '(1d0 2d0 3d0 4d0)
       (vt-relu (vt-asarray '(1 2 3 4) :dtype :int64)))
(check "P0-1 relu int64 负数截断" '(0d0 0d0 3d0)
       (vt-relu (vt-asarray '(-1 -2 3) :dtype :int64)))
(check-true "P0-1 relu int64 dtype"
            (eq :float64 (vt-dtype (vt-relu (vt-asarray '(1 2 3) :dtype :int64)))))
(check "P0-1 relu int32 2d" '((1d0 0d0) (0d0 4d0))
       (vt-relu (vt-from-array #2A((1 -2)(-3 4)) :dtype :int32)))
(check "P0-1 relu float32 保持" '(1.0 2.0 0.0)
       (vt-relu (vt-asarray '(1.0 2.0 -3.0) :dtype :float32)))
(check-true "P0-1 relu float32 dtype"
            (eq :float32 (vt-dtype (vt-relu (vt-asarray '(1.0 -1.0) :dtype :float32)))))

;; [P1-1] 整数输入的数学函数曾按 float32 精度计算（SBCL (sin 1) 只给 single-float）。
;;        现要求 float64 输入走完整 double 精度（与 numpy 逐位一致）。
(check "P1-1 sin int64 精度" '(0.8414709848078965d0 0.9092974268256817d0 0.1411200080598672d0)
       (vt-sin (vt-asarray '(1 2 3) :dtype :int64)))
(check "P1-1 cos int64 精度" '(0.5403023058681398d0 -0.4161468365471424d0 -0.9899924966004454d0)
       (vt-cos (vt-asarray '(1 2 3) :dtype :int64)))
(check "P1-1 exp int64 精度" '(2.718281828459045d0 7.38905609893065d0 20.085536923187668d0)
       (vt-exp (vt-asarray '(1 2 3) :dtype :int64)))
(check "P1-1 log int64 精度" '(0.0d0 0.6931471805599453d0 1.0986122886681098d0)
       (vt-log (vt-asarray '(1 2 3) :dtype :int64)))
(check "P1-1 sqrt int64" '(1d0 2d0 3d0)
       (vt-sqrt (vt-asarray '(1 4 9) :dtype :int64)))
(check-true "P1-1 sin int64 dtype float64"
            (eq :float64 (vt-dtype (vt-sin (vt-asarray '(1 2 3) :dtype :int64)))))
(check-true "P1-1 sin float32 保持 float32"
            (eq :float32 (vt-dtype (vt-sin (vt-asarray '(1.0 2.0) :dtype :float32)))))
(check "P1-1 sin float32 值" '(0.8414709568023682 0.9092974066734314)
       (vt-sin (vt-asarray '(1.0 2.0) :dtype :float32)))

;; [P1-2] vt-expm1 / vt-log1p 对整数输入曾不提升（返回被截断的 int64）。
(check "P1-2 expm1 int64" '(1.718281828459045d0 6.38905609893065d0 19.085536923187668d0)
       (vt-expm1 (vt-asarray '(1 2 3) :dtype :int64)))
(check "P1-2 log1p int64" '(0.6931471805599453d0 1.0986122886681098d0 1.3862943611198906d0)
       (vt-log1p (vt-asarray '(1 2 3) :dtype :int64)))
(check-true "P1-2 expm1 int64 dtype float64"
            (eq :float64 (vt-dtype (vt-expm1 (vt-asarray '(1 2 3) :dtype :int64)))))
(check-true "P1-2 log1p int64 dtype float64"
            (eq :float64 (vt-dtype (vt-log1p (vt-asarray '(1 2 3) :dtype :int64)))))

;; [P2#1] argmax/argmin/nanargmax/nanargmin 曾返回 int32；numpy 返回 intp(int64)。
(check-true "P2-1 argmax dtype int64"
            (eq :int64 (vt-dtype (vt-argmax (vt-asarray '(1 3 2) :dtype :int64)))))
(check-true "P2-1 argmin dtype int64"
            (eq :int64 (vt-dtype (vt-argmin (vt-asarray '(1 3 2) :dtype :int64)))))
(check-true "P2-1 nanargmax dtype int64"
            (eq :int64 (vt-dtype (vt-nanargmax (vt-asarray '(1.0d0 3.0d0 2.0d0) :dtype :float64)))))
(check-true "P2-1 nanargmin dtype int64"
            (eq :int64 (vt-dtype (vt-nanargmin (vt-asarray '(1.0d0 3.0d0 2.0d0) :dtype :float64)))))
(check "P2-1 argmax 值" 1 (vt-argmax (vt-asarray '(1 3 2) :dtype :int64)))

;; [P2#3] vt-cov 归一化分母须与 numpy 一致：缺省 ddof 等价 numpy 缺省（N-1），
;;        显式 ddof=0 → N，ddof=1 → N-1，bias=True → N。
(check "P2-3 cov 缺省=N-1" '((1d0 1d0) (1d0 1d0))
       (vt-cov (mk '((1d0 2d0 3d0) (4d0 5d0 6d0)))))
(check "P2-3 cov ddof=0" '((0.6666666666666666d0 0.6666666666666666d0)
                           (0.6666666666666666d0 0.6666666666666666d0))
       (vt-cov (mk '((1d0 2d0 3d0) (4d0 5d0 6d0))) :ddof 0))
(check "P2-3 cov ddof=1" '((1d0 1d0) (1d0 1d0))
       (vt-cov (mk '((1d0 2d0 3d0) (4d0 5d0 6d0))) :ddof 1))
(check "P2-3 cov bias" '((0.6666666666666666d0 0.6666666666666666d0)
                         (0.6666666666666666d0 0.6666666666666666d0))
       (vt-cov (mk '((1d0 2d0 3d0) (4d0 5d0 6d0))) :bias t))
(check "P2-3 cov 1d 缺省" 1d0 (vt-cov (mk '(1d0 2d0 3d0))))

;; [P2#6] vt-pow：整数基 + 正整数指数应保持整数 dtype（对标 numpy）；
;;        非整数指数 → float64。v0.4.x 契约变更：张量指数不再报错，
;;        改为对标 numpy.power 的逐元素语义（C99 pow 特例表，根治了
;;        曾静默返回 NaN 的问题）；形状不可广播时仍报错。
(check "P2-6 pow int**2" '(4 9) (vt-pow (vt-asarray '(2 3) :dtype :int64) 2))
(check-true "P2-6 pow int**2 dtype int64"
            (eq :int64 (vt-dtype (vt-pow (vt-asarray '(2 3) :dtype :int64) 2))))
(check "P2-6 pow int**0.5" '(2d0 3d0) (vt-pow (vt-asarray '(4 9) :dtype :int64) 0.5d0))
(check-true "P2-6 pow int**0.5 dtype float64"
            (eq :float64 (vt-dtype (vt-pow (vt-asarray '(4 9) :dtype :int64) 0.5d0))))
(check "P2-6 pow 张量指数逐元素" '(8 27)
       (vt-pow (vt-asarray '(2 3) :dtype :int64) (vt-asarray '(3 3) :dtype :int64)))
(check-true "P2-6 pow 张量指数 dtype int64"
            (eq :int64 (vt-dtype (vt-pow (vt-asarray '(2 3) :dtype :int64)
                                         (vt-asarray '(3 3) :dtype :int64)))))
;; vt-asarray 首参是数据：((1 2 3)(2 2 2))→(2,3) 与指数 (2,3) 同形逐元素幂
(check "P2-6 pow 张量指数同形" '((1 8 27) (16 16 16))
       (vt-pow (vt-asarray '((1 2 3) (2 2 2)) :dtype :int64)
               (vt-asarray '((3 3 3) (4 4 4)) :dtype :int64)))
(check-error "P2-6 pow 形状不可广播报错"
             (lambda () (vt-pow (vt-asarray '((1 2 3) (4 5 6)) :dtype :int64)
                                (vt-asarray '((1 2 3) (4 5 6) (7 8 9)) :dtype :int64))))

;; [P2#2] vt-logical-* 曾默认返回 :float64，违反 §3.1「比较/逻辑返回 :int8」。
(check-true "P2-2 logical-and dtype int8"
            (eq :int8 (vt-dtype (vt-logical-and (vt-asarray '(1 0) :dtype :int64)
                                                (vt-asarray '(1 1) :dtype :int64)))))
(check-true "P2-2 logical-or dtype int8"
            (eq :int8 (vt-dtype (vt-logical-or (vt-asarray '(1 0) :dtype :int64)
                                               (vt-asarray '(0 0) :dtype :int64)))))
(check-true "P2-2 logical-not dtype int8"
            (eq :int8 (vt-dtype (vt-logical-not (vt-asarray '(1 0) :dtype :int64)))))
(check-true "P2-2 logical-xor dtype int8"
            (eq :int8 (vt-dtype (vt-logical-xor (vt-asarray '(1 0) :dtype :int64)
                                                (vt-asarray '(1 1) :dtype :int64)))))
(check "P2-2 logical-and 值" '(1 0)
       (vt-logical-and (vt-asarray '(1 0) :dtype :int64) (vt-asarray '(1 1) :dtype :int64)))

;;; ============================================================
;;; [R-3] 整数输入激活函数提升回归
;;;   vt-softplus / vt-gelu 缺省 dtype 时，lambda 交给 vt-map 会走
;;;   vt-promote-type → 整数输入得到整数结果 dtype，浮点值被静默截断。
;;;   例：(vt-softplus (vt-asarray '(0))) → 0（应 0.693…）；
;;;       (vt-gelu (vt-asarray '(1 2 3))) → (0 1 2)（应 0.841/1.955/2.996）。
;;;   对标 numpy：整型进 float64 出、float32 进 float32 出。
;;; ============================================================
;; softplus：dtype 提升
(check-true "R-3 softplus int64 dtype → float64"
            (eq :float64 (vt-dtype (vt-softplus (vt-asarray '(1 2 3) :dtype :int64)))))
(check-true "R-3 softplus int8 dtype → float64"
            (eq :float64 (vt-dtype (vt-softplus (vt-asarray '(1 2 3) :dtype :int8)))))
(check-true "R-3 softplus float32 保持 float32"
            (eq :float32 (vt-dtype (vt-softplus (vt-asarray '(1.0f0 2.0f0) :dtype :float32)))))
;; softplus：数值不再被截断（numpy 参考值）
(check "R-3 softplus([0])=ln2" '(0.6931471805599453d0)
       (vt-softplus (vt-asarray '(0) :dtype :int64)))
(check "R-3 softplus([-1])" '(0.31326168751822286d0)
       (vt-softplus (vt-asarray '(-1) :dtype :int64)))
(check "R-3 softplus([3])" '(3.048587351573742d0)
       (vt-softplus (vt-asarray '(3) :dtype :int64)))
(check "R-3 softplus([1,2,3])"
       '(1.3132616875182228d0 2.1269280110429727d0 3.048587351573742d0)
       (vt-softplus (vt-asarray '(1 2 3) :dtype :int64)))
(check "R-3 softplus([-5,0,5,25])"
       '(0.006715348489118068d0 0.6931471805599453d0 5.006715348489118d0 25.0d0)
       (vt-softplus (vt-asarray '(-5 0 5 25) :dtype :int64)))
;; gelu：dtype 提升 + 数值
(check-true "R-3 gelu int64 dtype → float64"
            (eq :float64 (vt-dtype (vt-gelu (vt-asarray '(1 2 3) :dtype :int64)))))
(check-true "R-3 gelu int8 dtype → float64"
            (eq :float64 (vt-dtype (vt-gelu (vt-asarray '(1 2 3) :dtype :int8)))))
(check-true "R-3 gelu float32 保持 float32"
            (eq :float32 (vt-dtype (vt-gelu (vt-asarray '(1.0f0 2.0f0) :dtype :float32)))))
(check "R-3 gelu([1,2,3])"
       '(0.8411919906082768d0 1.954597694087775d0 2.996362607918227d0)
       (vt-gelu (vt-asarray '(1 2 3) :dtype :int64)) 1d-12)
;; 整数输入 + :out float64 现在可用（修复前因结果 dtype 为 int64 而报错）
(check "R-3 softplus int64 + :out float64"
       '(1.3132616875182228d0 2.1269280110429727d0 3.048587351573742d0)
       (let ((o (vt-zeros '(3) :dtype :float64)))
         (vt-softplus (vt-asarray '(1 2 3) :dtype :int64) :out o)
         o))
;; 其余激活函数本已正确（守护，防止将来退化）
(check-true "R-3 sigmoid int64 dtype float64"
            (eq :float64 (vt-dtype (vt-sigmoid (vt-asarray '(1 2 3) :dtype :int64)))))
(check-true "R-3 swish int64 dtype float64"
            (eq :float64 (vt-dtype (vt-swish (vt-asarray '(1 2 3) :dtype :int64)))))
(check-true "R-3 mish int64 dtype float64"
            (eq :float64 (vt-dtype (vt-mish (vt-asarray '(1 2 3) :dtype :int64)))))
(check-true "R-3 hard-tanh int64 dtype float64"
            (eq :float64 (vt-dtype (vt-hard-tanh (vt-asarray '(1 2 3) :dtype :int64)))))
(check-true "R-3 hard-sigmoid int64 dtype float64"
            (eq :float64 (vt-dtype (vt-hard-sigmoid (vt-asarray '(1 2 3) :dtype :int64)))))
(check-true "R-3 leaky-relu int64 dtype float64"
            (eq :float64 (vt-dtype (vt-leaky-relu (vt-asarray '(1 2 3) :dtype :int64)))))
;; leaky-relu 负值本会被 int64 存储截断成 0（(-2 3 -4) → (0 3 0)）；
;; 提升后应按 alpha=0.01 保留小数（对标 torch F.leaky_relu(0.01)）。
(check "R-3 leaky-relu int64 负值不截断"
       '(-0.02d0 3.0d0 -0.04d0)
       (vt-leaky-relu (vt-asarray '(-2 3 -4) :dtype :int64)) 1d-12)
(check-true "R-3 leaky-relu float32 保持 float32"
            (eq :float32 (vt-dtype (vt-leaky-relu (vt-asarray '(1.0f0 -2.0f0)
                                                             :dtype :float32)))))

;;; ============================================================
;;; [CE-LOGITS] vt-cross-entropy-logits：raw logits + 整数类别索引
;;;   对标 torch.nn.CrossEntropyLoss 的标准用法。
;;;   原 vt-cross-entropy 文档写「对标 torch.nn.CrossEntropyLoss」，
;;;   但实现是概率/one-hot 输入，二者不一致（见 nn.lisp 文档修正）。
;;; ============================================================
(defparameter *ce-z* (vt-asarray '((1.0d0 2.0d0 3.0d0) (2.0d0 0.0d0 1.0d0))))
(defparameter *ce-lab* (vt-asarray '(0 1) :dtype :int64))
;; 基准值：torch.nn.CrossEntropyLoss(logits, labels) = 2.4076059644443806
(check "CE-LOGITS 用户原文调用 (2,3)+(2,)int64" 2.4076059644443806d0
       (vt-item (vt-cross-entropy-logits *ce-z* *ce-lab*)) 1d-12)
(check "CE-LOGITS reduction :sum" 4.815211928888761d0
       (vt-item (vt-cross-entropy-logits *ce-z* *ce-lab* :reduction :sum)) 1d-12)
(check "CE-LOGITS reduction :none"
       '(2.4076059644443806d0 2.4076059644443806d0)
       (vt-to-list (vt-cross-entropy-logits *ce-z* *ce-lab* :reduction :none)) 1d-12)
;; 无批次 (C,) + 长度 1 labels
(check "CE-LOGITS 无批次 (3,)+(1,)" 2.4076059644443806d0
       (vt-item (vt-cross-entropy-logits (vt-asarray '(1.0d0 2.0d0 3.0d0))
                                         (vt-asarray '(0) :dtype :int64))) 1d-12)
;; 任意秩 (2,2,3) + (2,2)
(check "CE-LOGITS 3D (2,2,3)+(2,2)" 2.4076059644443806d0
       (vt-item (vt-cross-entropy-logits
                 (vt-asarray '(((1.0d0 2.0d0 3.0d0) (1.0d0 2.0d0 3.0d0))
                               ((2.0d0 0.0d0 1.0d0) (2.0d0 0.0d0 1.0d0))))
                 (vt-asarray '((0 0) (1 1)) :dtype :int64))) 1d-12)
;; axis 参数
(check "CE-LOGITS axis=0" 2.4076059644443806d0
       (vt-item (vt-cross-entropy-logits
                 (vt-asarray '((1.0d0 2.0d0) (2.0d0 0.0d0) (3.0d0 1.0d0)))
                 (vt-asarray '(0 1) :dtype :int64) :axis 0)) 1d-12)
;; 数值稳定：大 logits（log-softmax 不减最大值会溢出）
(check "CE-LOGITS 大 logits 数值稳定" 0.0d0
       (vt-item (vt-cross-entropy-logits (vt-asarray '((1.0d4 0.0d0 0.0d0)))
                                         (vt-asarray '(0) :dtype :int64))))
(check "CE-LOGITS 完美预测=0" 0.0d0
       (vt-item (vt-cross-entropy-logits (vt-asarray '((100.0d0 -100.0d0)))
                                         (vt-asarray '(0) :dtype :int64))))
;; 与 softmax+log 路径一致（logits=[1,2,3] 取类别 2 → -log_softmax[2] = 0.4076…）
(check "CE-LOGITS 与 softmax+log 一致" 0.4076059644443804d0
       (vt-item (vt-cross-entropy-logits (vt-asarray '((1.0d0 2.0d0 3.0d0)))
                                         (vt-asarray '(2) :dtype :int64))) 1d-12)
;; dtype
(check-true "CE-LOGITS float32 保持 float32"
            (eq :float32 (vt-dtype (vt-cross-entropy-logits
                                    (vt-asarray '((1.0f0 2.0f0 3.0f0)) :dtype :float32)
                                    (vt-asarray '(0) :dtype :int64)))))
(check-true "CE-LOGITS float64 logits → float64"
            (eq :float64 (vt-dtype (vt-cross-entropy-logits
                                    (vt-asarray '((1.0d0 2.0d0 3.0d0)))
                                    (vt-asarray '(0) :dtype :int64)))))
;; :out
(check "CE-LOGITS :out 标量" 2.4076059644443806d0
       (let ((o (vt-zeros nil :dtype :float64)))
         (vt-cross-entropy-logits *ce-z* *ce-lab* :out o)
         (vt-item o)) 1d-12)
;; 错误路径
(check-error "CE-LOGITS labels 形状不符报错"
             (lambda () (vt-cross-entropy-logits *ce-z*
                                                 (vt-asarray '(0 1 2) :dtype :int64))))
(check-error "CE-LOGITS 索引越界报错"
             (lambda () (vt-cross-entropy-logits (vt-asarray '((1.0d0 2.0d0 3.0d0)))
                                                 (vt-asarray '(5) :dtype :int64))))
(check-error "CE-LOGITS 负数索引报错"
             (lambda () (vt-cross-entropy-logits (vt-asarray '((1.0d0 2.0d0 3.0d0)))
                                                 (vt-asarray '(-1) :dtype :int64))))
(check-error "CE-LOGITS 浮点 labels 报错"
             (lambda () (vt-cross-entropy-logits (vt-asarray '((1.0d0 2.0d0 3.0d0)))
                                                 (vt-asarray '(0.0d0)))))
(check-error "CE-LOGITS 未知 reduction 报错"
             (lambda () (vt-cross-entropy-logits (vt-asarray '((1.0d0 2.0d0 3.0d0)))
                                                 (vt-asarray '(0) :dtype :int64)
                                                 :reduction :bogus)))
;; 旧接口语义未受影响（仍为概率输入）
(check "CE 旧接口 概率输入仍工作" (- (log 0.9d0))
       (vt-item (vt-cross-entropy (vt-from-sequence '(1.0d0 0.0d0))
                                  (vt-from-sequence '(0.9d0 0.1d0)))) 1d-6)

;;; ============================================================
(summary)
