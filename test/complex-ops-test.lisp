;;;; complex-ops-test.lisp — 复数「数据搬运层」测试套件（CONVENTIONS §16.18）
;;;;
;;;; 组成：
;;;;   1) 差分对照（96 例）：test/complex-ops-expected.sexp 由
;;;;      gen_complex_ops_expected.py 从**本机 numpy 实测**生成。
;;;;   2) 契约断言：dtype / shape 保持、:out 硬契约、dtype 提升、
;;;;      未支持组合的确定性报错。
;;;;   3) 实数回归网：本层覆盖的都是「双路分派」函数，
;;;;      必须验证实数入参仍走原实现且结果不变。
;;;;
;;;; 用法：sbcl --non-interactive --load test/complex-ops-test.lisp

(require :asdf)
;; 系统根目录由本文件位置推出（不依赖硬编码路径）
(defparameter *clvt-test-dir* (make-pathname :name nil :type nil :defaults *load-truename*))
(pushnew (merge-pathnames "../" *clvt-test-dir*) asdf:*central-registry* :test #'equal)
(handler-bind ((warning #'muffle-warning)) (asdf:load-system :clvt))

(in-package :clvt)

;;; ------------------------------------------------------------------
;;; 骨架
;;; ------------------------------------------------------------------

(defvar *pass* 0)
(defvar *fail* 0)
(defvar *failures* '())

(defun approx= (a b &optional (tol 1d-9))
  "复数/实数近似相等；NaN 只与 NaN 相等；±Inf 同号相等。
   NaN 比较会触发 invalid-operation 陷阱，故整段在 with-float-safe 下求值。"
  (with-float-safe
    (cond
      ((and (complexp a) (complexp b))
       (and (approx= (realpart a) (realpart b) tol)
            (approx= (imagpart a) (imagpart b) tol)))
      ((or (complexp a) (complexp b)) nil)
      ((and (floatp a) (not (= a a))) (and (floatp b) (not (= b b))))
      ((and (floatp b) (not (= b b))) nil)
      ((and (floatp a) (or (> a 1d300) (< a -1d300)))
       (and (floatp b) (= a b)))
      ((and (floatp b) (or (> b 1d300) (< b -1d300)))
       (and (floatp a) (= a b)))
      (t (<= (abs (- a b)) tol)))))

(defun expect-true (name form)
  (if form
      (incf *pass*)
      (progn (incf *fail*)
             (push name *failures*)
             (format t "~&FAIL ~a~%" name))))

(defun expect-equal (name got want)
  (if (equal got want)
      (incf *pass*)
      (progn (incf *fail*)
             (push (format nil "~a: 期望 ~s 实得 ~s" name want got) *failures*)
             (format t "~&FAIL ~a: 期望 ~s 实得 ~s~%" name want got))))

(defun expect-error (name thunk)
  "断言 THUNK 抛出错误（未支持组合必须是确定性报错，不能静默出错值）。"
  (handler-case (progn (funcall thunk)
                       (incf *fail*)
                       (push (format nil "~a: 期望报错但正常返回" name) *failures*)
                       (format t "~&FAIL ~a: 期望报错但正常返回~%" name))
    (error () (incf *pass*))))

;;; ------------------------------------------------------------------
;;; 输入数据（与 gen_complex_ops_expected.py 一一对应）
;;; ------------------------------------------------------------------

(defun z (x) (coerce x '(complex double-float)))
(defparameter +nan+ +vt-dfloat-nan+)

(defparameter cx-a (vt-from-sequence (list (z #C(1 2)) (z #C(3 -4))) :dtype :complex128))
(defparameter cx-b (vt-from-sequence (list (z #C(2 -1)) (z #C(1 1))) :dtype :complex128))
(defparameter cx-c (vt-from-sequence (list (z #C(1 2)) (z #C(3 -4)) (z #C(5 0)))
                                     :dtype :complex128))
(defparameter cx-z3 (vt-zeros (list 3) :dtype :complex128))
(defparameter cx-na (vt-from-sequence (list (list (z #C(1 2))
                                                  (complex +nan+ 0.0d0)
                                                  (z #C(3 -4)))
                                            (list (z #C(5 0))
                                                  (z #C(6 1))
                                                  (complex 0.0d0 +nan+)))
                                      :dtype :complex128))
(defparameter cx-m2 (vt-from-sequence (list (list (z #C(1 1)) (z #C(2 0)))
                                            (list (z #C(3 0)) (z #C(0 4))))
                                      :dtype :complex128))
(defparameter cx-dup (vt-from-sequence (list (z #C(3 1)) (z #C(1 5)) (z #C(1 2)) (z #C(3 1)))
                                       :dtype :complex128))
;; 结构算子用例的输入（与生成器一一对应，§16.19.8）
(defparameter cx-nanc (vt-from-sequence (list (complex +nan+ 0.0d0)) :dtype :complex128))
(defparameter cx-emptyc (vt-zeros (list 0) :dtype :complex128))
(defparameter cx-idxcol (vt-from-sequence (list (list 1) (list 0)) :dtype :int64))
(defparameter cx-idxrow (vt-from-sequence (list (list 1 0)) :dtype :int64))

(defparameter flat4 (vt-from-sequence (list 1.0d0 2.0d0 3.0d0 4.0d0) :dtype :float64))

;;; ------------------------------------------------------------------
;;; 期望值读取
;;; ------------------------------------------------------------------

(defun normalize-real (x)
  (case x
    (:nan +vt-dfloat-nan+)
    (:inf +vt-dfloat-pos-inf+)
    (:neg-inf +vt-dfloat-neg-inf+)
    (t x)))

(defun normalize-scalar (x)
  (cond ((consp x) (if (eq (first x) :cx)
                       (complex (normalize-real (second x))
                                (normalize-real (third x)))
                       (error "无法归一化 ~a" x)))
        (t (normalize-real x))))

(defun flatten-list (v)
  "把 vt-to-list 的结果（可能嵌套）压平；标量 → 单元素列表。"
  (cond ((null v) nil)
        ((listp v) (mapcan #'flatten-list v))
        (t (list v))))

(defvar *cases* nil)
(defvar *generator* nil)

(defparameter *expected-file*
  (merge-pathnames "complex-ops-expected.sexp" cl-user::*clvt-test-dir*)
  "由 gen_complex_ops_expected.py 生成的 numpy 实测期望值。")

(with-open-file (s *expected-file*)
  (let ((data (read s)))
    (setf *generator* (getf data :generator)
          *cases* (getf data :cases))))

;;; ------------------------------------------------------------------
;;; 1. 差分对照
;;; ------------------------------------------------------------------

(defun run-differential ()
  (format t "~&--- 差分对照（numpy ~a，~d 例）~%" *generator* (length *cases*))
  (dolist (c *cases*)
    (let ((name (getf c :name))
          (form (getf c :form))
          (value (getf c :value))
          (dtype (getf c :dtype))
          (shape (getf c :shape))
          (kind (getf c :kind)))
      (handler-case
          (let ((got (eval (read-from-string form))))
            (if (eq kind :symbol)
                (expect-equal name got value)
                (let* ((flat (flatten-list (if (vt-p got) (vt-to-list got) got)))
                       (want (mapcar #'normalize-scalar value))
                       ;; 生成器直接写 :DTYPE :COMPLEX128 / :SHAPE (2 4)，
                       ;; sexp 读入后就是关键字与整数表，无需再解析字符串
                       (want-dtype dtype)
                       (want-shape shape))
                  (cond
                    ((not (= (length flat) (length want)))
                     (incf *fail*)
                     (push (format nil "~a: 元素数 ~a ≠ 期望 ~a" name (length flat) (length want))
                           *failures*)
                     (format t "~&FAIL ~a: 元素数 ~a ≠ ~a~%" name (length flat) (length want)))
                    ((not (vt-p got))
                     (if (every #'approx= flat want)
                         (incf *pass*)
                         (progn (incf *fail*) (push name *failures*)
                                (format t "~&FAIL ~a: 值不符 ~a~%" name flat))))
                    (t
                     (let ((bad nil))
                       (unless (eq (vt-dtype got) want-dtype)
                         (setf bad (format nil "dtype ~a ≠ ~a" (vt-dtype got) want-dtype)))
                       (unless (equal (vt-shape got) want-shape)
                         (setf bad (format nil "shape ~a ≠ ~a" (vt-shape got) want-shape)))
                       (unless (every #'approx= flat want)
                         (setf bad (format nil "值不符 ~a (期望 ~a)" flat want)))
                       (if bad
                           (progn (incf *fail*) (push (format nil "~a: ~a" name bad) *failures*)
                                  (format t "~&FAIL ~a: ~a~%" name bad))
                           (incf *pass*))))))))
        (error (e)
          (incf *fail*)
          (push (format nil "~a: 求值异常 ~a" name e) *failures*)
          (format t "~&FAIL ~a: 求值异常 ~a~%" name e)))))
  (format t "~&差分对照完成：PASS=~d FAIL=~d~%" *pass* *fail*))

;;; ------------------------------------------------------------------
;;; 2. 契约断言
;;; ------------------------------------------------------------------

(defun run-contracts ()
  (format t "~%~%--- 契约断言 ---~%")

  ;; ---- dtype 提升：拼接/栈 按 promote-type ----
  (expect-equal "concat c128+c64 → c128"
                (vt-dtype (vt-concatenate 0 cx-a (vt-astype cx-b :complex64)))
                :complex128)
  (expect-equal "concat c128+f64 → c128"
                (vt-dtype (vt-concatenate 0 cx-a (vt-from-sequence (list 1.5d0) :dtype :float64)))
                :complex128)
  (expect-equal "concat c64+c64 → c64"
                (vt-dtype (vt-concatenate 0 (vt-astype cx-a :complex64)
                                            (vt-astype cx-b :complex64)))
                :complex64)
  (expect-equal "stack 值" (vt-to-list (vt-stack 0 cx-a cx-b))
                (list (list #C(1.0d0 2.0d0) #C(3.0d0 -4.0d0))
                      (list #C(2.0d0 -1.0d0) #C(1.0d0 1.0d0))))

  ;; ---- 剪切不复制：切片是视图（写回基数组） ----
  (let ((base (vt-copy cx-c)))
    (let ((sl (vt-slice base (list 0 3 2))))
      (setf (aref (vt-data sl) (vt-offset sl)) (z #C(9 9)))
      (expect-equal "slice 是视图（别名）" (vt-ref base 0) (z #C(9 9)))))

  ;; ---- :out 硬契约（nan 族） ----
  (let ((o (vt-zeros (list 3) :dtype :complex128)))
    (expect-equal "nansum :out 写入"
                  (vt-to-list (vt-nansum cx-na :axis 0 :out o))
                  (list #C(6.0d0 2.0d0) #C(6.0d0 1.0d0) #C(3.0d0 -4.0d0))))
  (expect-error "nansum :out dtype 不符"
                (lambda () (vt-nansum cx-na :axis 0
                                      :out (vt-zeros (list 3) :dtype :float64))))
  (expect-error "nanvar :out dtype 必须实数"
                (lambda () (vt-nanvar cx-na :axis 1
                                      :out (vt-zeros (list 2) :dtype :complex128))))

  ;; ---- 未支持的组合必须确定性报错 ----
  (expect-error "view 非法重解释对"
                (lambda () (vt-view cx-a nil :dtype :int64)))
  (expect-error "view 形状不匹配"
                (lambda () (vt-view cx-a (list 3) :dtype :float64)))
  ;; 多轴 nanmedian 已接通（§16.35.4）；此前这里断言的是「必须报错」。
  ;; numpy 2.4.6: nanmedian([[1+2j, 3-4j, 5+0j]], axis=(0,1)) => 3-4j
  (expect-true "nanmedian 多轴 (0 1) 已支持"
               (let ((r (vt-ref (vt-nanmedian (vt-reshape cx-c (list 1 3))
                                              :axis (list 0 1)))))
                 (and (complexp r)
                      (= (realpart r) 3d0)
                      (= (imagpart r) -4d0))))
  (expect-error "nanargmax 全 NaN"
                (lambda () (vt-nanargmax (vt-from-sequence
                                          (list (complex +nan+ 0.0d0)) :dtype :complex128))))
  (expect-error "nanmean :dtype 必须复数"
                (lambda () (vt-nanmean cx-a :dtype :float64)))

  ;; ---- NaN 语义 ----
  (expect-equal "unique 把 NaN 折叠为一个"
                (vt-to-list (vt-unique (vt-from-sequence
                                        (list (z #C(1 2)) (complex +nan+ 0.0d0)
                                              (complex +nan+ 0.0d0) (z #C(1 2)))
                                        :dtype :complex128)))
                (list (z #C(1 2)) (complex +nan+ 0.0d0)))
  (expect-equal "in1d 视 NaN 与 NaN 相等"
                (vt-to-list (vt-in1d (vt-from-sequence (list (complex +nan+ 0.0d0))
                                                       :dtype :complex128)
                                     (vt-from-sequence (list (complex +nan+ 0.0d0))
                                                       :dtype :complex128)))
                (list 1))
  (expect-equal "nansum 全 NaN = 0"
                (vt-to-list (vt-nansum (vt-from-sequence
                                        (list (complex +nan+ 0.0d0) (complex +nan+ 0.0d0))
                                        :dtype :complex128)))
                (z #C(0 0)))

  ;; ---- 读取格式自检（防止 *read-default-float-format* 回归）----
  ;; 若源码里再次出现裸 `0.0`，在 `(setf *read-default-float-format*
  ;; 'double-float)` 的环境下编译后，complex64 的零值会变成 double 复数，
  ;; 往 c64 数组里 fill 时 SBCL 会抛
  ;;   "The value #C(0.0 0.0) is not of type (COMPLEX SINGLE-FLOAT)
  ;;    when binding SB-C::ITEM"
  ;; 这三条断言在「读取环境被改成 double」时必然失败，起到哨兵作用。
  (expect-equal "c64 默认零值精度（读取格式哨兵）"
                (type-of (vt-dtype-default-value :complex64))
                '(complex single-float))
  (expect-equal "c64 vt-cast 精度（读取格式哨兵）"
                (type-of (vt-cast 0 :complex64))
                '(complex single-float))
  (expect-equal "c128 默认零值精度"
                (type-of (vt-dtype-default-value :complex128))
                '(complex double-float))

  ;; ---- complex64 往返：nan 族保精度 ----
  (expect-equal "c64 nanmean dtype"
                (vt-dtype (vt-nanmean (vt-astype cx-na :complex64) :axis 1))
                :complex64)
  (expect-equal "c64 nansum 值"
                (vt-to-list (vt-nansum (vt-astype cx-na :complex64) :axis 1))
                (list (complex 4.0s0 -2.0s0) (complex 11.0s0 1.0s0)))

  ;; ---- 持久化往返（lisp 与 npy 两种格式，含 NaN） ----
  (let* ((v (vt-from-sequence (list (z #C(1 2)) (complex +nan+ 0.0d0)) :dtype :complex128))
         (p1 (merge-pathnames "complex-ops-roundtrip.sexp" cl-user::*clvt-test-dir*))
         (p2 (merge-pathnames "complex-ops-roundtrip.npy" cl-user::*clvt-test-dir*)))
    (unwind-protect
         (progn
           (vt-save v p1)
           (let ((r (vt-load p1)))
             (expect-equal "lisp 格式往返 dtype" (vt-dtype r) :complex128)
             (expect-true "lisp 格式往返值"
                          (and (approx= (vt-ref r 0) (z #C(1 2)))
                               (vt-float-nan-p (realpart (vt-ref r 1))))))
           (vt-save v p2 :as :npy)
           (let ((r (vt-load p2)))
             (expect-equal "npy 格式往返 dtype" (vt-dtype r) :complex128)
             (expect-true "npy 格式往返值"
                          (and (approx= (vt-ref r 0) (z #C(1 2)))
                               (vt-float-nan-p (realpart (vt-ref r 1))))))
           (vt-save (vt-astype v :complex64) p2 :as :npy)
           (expect-equal "npy c64 往返 dtype" (vt-dtype (vt-load p2)) :complex64))
      (ignore-errors (delete-file p1))
      (ignore-errors (delete-file p2))))

  ;; ---- 实数路径零改动（双路分派函数的原实现转发） ----
  (expect-equal "real repeat" (vt-to-list (vt-repeat (vt-from-sequence (list 1.0d0 2.0d0) :dtype :float64) 2))
                (list 1.0d0 1.0d0 2.0d0 2.0d0))
  (expect-equal "real unique" (vt-to-list (vt-unique (vt-from-sequence (list 3.0d0 1.0d0 3.0d0) :dtype :float64)))
                (list 1.0d0 3.0d0))
  (expect-equal "real searchsorted"
                (vt-to-list (vt-searchsorted (vt-from-sequence (list 1.0d0 3.0d0) :dtype :float64) 2.0d0))
                1)
  (expect-equal "real nanmean"
                (vt-to-list (vt-nanmean (vt-from-sequence (list 1.0d0 +nan+) :dtype :float64)))
                1.0d0)
  (expect-equal "real nanvar"
                (vt-to-list (vt-nanvar (vt-from-sequence (list 1.0d0 3.0d0) :dtype :float64)))
                1.0d0)
  (expect-equal "real tensordot"
                (vt-to-list (vt-tensordot (vt-from-sequence (list (list 1.0d0 2.0d0)) :dtype :float64)
                                          (vt-from-sequence (list (list 1.0d0) (list 1.0d0)) :dtype :float64)
                                          :axes 1))
                (list (list 3.0d0)))
  (expect-equal "real correlate"
                (vt-to-list (vt-correlate (vt-from-sequence (list 1.0d0 2.0d0) :dtype :float64)
                                          (vt-from-sequence (list 3.0d0 1.0d0) :dtype :float64)
                                          :mode :full))
                (list 1.0d0 5.0d0 6.0d0))
  (expect-equal "real view" (vt-to-list (vt-view (vt-from-sequence (list 1.0d0 2.0d0) :dtype :float64)
                                                 (list 2 1)))
                (list (list 1.0d0) (list 2.0d0))))

;;; ------------------------------------------------------------------
;;; 3. 入口
;;; ------------------------------------------------------------------

(format t "~&=====================================~%")
(format t "复数数据搬运层测试 (CONVENTIONS §16.18)~%")
(format t "=====================================~%")
(run-differential)
(run-contracts)
(format t "~&=====================================~%")
(format t "Total: ~d / Pass: ~d / Fail: ~d~%" (+ *pass* *fail*) *pass* *fail*)
(format t "PASS=~d FAIL=~d~%" *pass* *fail*)
(when *failures*
  (format t "~&--- 失败清单 ---~%")
  (dolist (f (reverse *failures*)) (format t "  · ~a~%" f)))
(when (plusp *fail*) (sb-ext:exit :code 1))
