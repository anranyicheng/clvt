;;;; setops-axis-trace-test.lisp — 按轴去重 / 迹（§16.23 / §16.24）
;;;;
;;;; 覆盖两个「numpy 有、clvt 之前缺」的通用特性（不是复数专属）：
;;;;   1) vt-unique :axis —— numpy 2.x 的按轴去重
;;;;      含 return-index / return-inverse / return-counts，退化形状、
;;;;      NaN 行不折叠、非连续视图输入；
;;;;   2) vt-trace :offset + :axis1/:axis2 —— numpy.trace 的完整签名。
;;;;
;;;; 组成：
;;;;   1) 差分对照（37 例）：test/setops-axis-expected.sexp 由
;;;;      gen_setops_axis_expected.py 从**本机 numpy 实测**生成
;;;;      （CONVENTIONS §10.4：禁手写期望值）。
;;;;   2) 契约断言：dtype / shape 保持、axis 越界报错、秩不足报错、
;;;;      非连续 :out 写入。
;;;;   3) 实数回归网：默认（无 axis / 无 offset）路径必须与原实现逐位一致。
;;;;
;;;; 用法：sbcl --non-interactive --load test/setops-axis-trace-test.lisp

(require :asdf)
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
  (with-float-safe
    (cond
      ((and (complexp a) (complexp b))
       (and (approx= (realpart a) (realpart b) tol)
            (approx= (imagpart a) (imagpart b) tol)))
      ((or (complexp a) (complexp b)) nil)
      ((and (floatp a) (not (= a a))) (and (floatp b) (not (= b b))))
      ((and (floatp b) (not (= b b))) nil)
      (t (<= (abs (- a b)) tol)))))

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

(defun expect-error (name thunk)
  (handler-case (progn (funcall thunk)
                       (incf *fail*)
                       (push (format nil "~a: 期望报错但正常返回" name) *failures*)
                       (format t "~&FAIL ~a: 期望报错但正常返回~%" name))
    (error () (incf *pass*))))

;;; ------------------------------------------------------------------
;;; 输入数据（与 gen_setops_axis_expected.py 一一对应）
;;; ------------------------------------------------------------------

(defparameter +nan+ +vt-dfloat-nan+)
(defun z (x) (coerce x '(complex double-float)))

(defparameter ua (vt-from-sequence '((2 0 1 5) (1 5 2 0) (2 0 1 5)) :dtype :int64))
(defparameter ub (vt-from-sequence '((3 1 3) (1 2 2) (3 1 3) (5 5 5)) :dtype :int64))
(defparameter uc (vt-from-sequence (list (list +nan+ 1.0d0)
                                        (list +nan+ 1.0d0)
                                        (list 2.0d0 0.0d0))
                                  :dtype :float64))
(defparameter ud (vt-from-sequence (list (list (list 0.0d0 1.0d0) (list 0.0d0 1.0d0) (list 5.0d0 5.0d0))
                                        (list (list 9.0d0 9.0d0) (list 0.0d0 1.0d0) (list 5.0d0 5.0d0)))
                                  :dtype :float64))
(defparameter ue (vt-from-sequence (list (list (z #C(1 2)) (z #C(0 0)))
                                        (list (z #C(1 2)) (z #C(0 0)))
                                        (list (z #C(0 1)) (z #C(3 0))))
                                  :dtype :complex128))
(defparameter ue2 (vt-from-sequence (list (list (z #C(1 2)) (z #C(0 0)) (z #C(5 -1)))
                                         (list (z #C(0 1)) (z #C(3 0)) (z #C(5 -1)))
                                         (list (z #C(1 2)) (z #C(0 0)) (z #C(5 -1))))
                                   :dtype :complex128))
(defparameter uempty (vt-zeros '(0 3) :dtype :float64))
(defparameter uzero (vt-zeros '(3 0) :dtype :float64))

(defparameter tm (vt-from-sequence '((1.0d0 2.0d0 3.0d0 4.0d0)
                                     (5.0d0 6.0d0 7.0d0 8.0d0)
                                     (9.0d0 10.0d0 11.0d0 12.0d0)
                                     (13.0d0 14.0d0 15.0d0 16.0d0))
                               :dtype :float64))
(defparameter tr (vt-from-sequence '((1.0d0 2.0d0 3.0d0 4.0d0 5.0d0)
                                     (6.0d0 7.0d0 8.0d0 9.0d0 10.0d0)
                                     (11.0d0 12.0d0 13.0d0 14.0d0 15.0d0))
                               :dtype :float64))
(defparameter tc (vt-from-sequence (list (list (z #C(1 2)) (z #C(3 -1)) (z #C(0 0)))
                                        (list (z #C(2 0)) (z #C(4 4)) (z #C(1 -1)))
                                        (list (z #C(0 3)) (z #C(5 0)) (z #C(6 2))))
                                  :dtype :complex128))
(defparameter td (vt-reshape (vt-arange 18 :dtype :float64) '(2 3 3)))

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
  (cond ((null v) nil)
        ((listp v) (mapcan #'flatten-list v))
        (t (list v))))

(defvar *cases* nil)
(defvar *generator* nil)

(defparameter *expected-file*
  (merge-pathnames "setops-axis-expected.sexp" cl-user::*clvt-test-dir*))

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
          (shape (getf c :shape)))
      (handler-case
          (let ((got (eval (read-from-string form))))
            (let* ((flat (flatten-list (if (vt-p got) (vt-to-list got) got)))
                   (want (mapcar #'normalize-scalar value)))
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
                   (unless (eq (vt-dtype got) dtype)
                     (setf bad (format nil "dtype ~a ≠ ~a" (vt-dtype got) dtype)))
                   (unless (equal (vt-shape got) shape)
                     (setf bad (format nil "shape ~a ≠ ~a" (vt-shape got) shape)))
                   (unless (every #'approx= flat want)
                     (setf bad (format nil "值不符 ~a (期望 ~a)" flat want)))
                   (if bad
                       (progn (incf *fail*) (push (format nil "~a: ~a" name bad) *failures*)
                              (format t "~&FAIL ~a: ~a~%" name bad))
                       (incf *pass*)))))))
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

  ;; ---- unique :axis 的形状/dtype 保持 ----
  (expect-equal "unique axis=0 dtype 保持"
                (vt-dtype (vt-unique ua :axis 0)) :int64)
  (expect-equal "unique axis=1 形状（其余轴不变）"
                (vt-shape (vt-unique ub :axis 1)) '(4 3))
  (expect-equal "unique axis=1 三列去重后仍 3 行轴"
                (second (vt-shape (vt-unique ub :axis 1))) 3)
  (expect-equal "unique 复数 axis=0 dtype"
                (vt-dtype (vt-unique ue :axis 0)) :complex128)

  ;; ---- unique :axis 的 NaN 行不折叠（与展平模式相反）----
  (expect-equal "unique axis=0 NaN 行不合并（3 行）"
                (first (vt-shape (vt-unique uc :axis 0))) 3)
  (expect-equal "unique 展平 NaN 折叠为一个"
                (length (vt-to-list (vt-unique (vt-reshape uc '(6)))))
                4)  ; nan, 0, 1, 2

  ;; ---- return-inverse 可重建输入（numpy 语义：vt-take + axis）----
  (multiple-value-bind (u i inv c)
      (vt-unique ua :axis 0 :return-index t :return-inverse t :return-counts t)
    (declare (ignore i c))
    (expect-equal "axis=0 的 inverse 能重建输入"
                  (vt-to-list (vt-take u inv :axis 0))
                  (vt-to-list ua))
    (expect-equal "inverse 长度 = 该轴长度"
                  (vt-shape inv) '(3)))

  ;; ---- 越界 axis 必须报错 ----
  (expect-error "unique axis=5 越界报错"
                (lambda () (vt-unique ua :axis 5)))
  (expect-error "trace 秩 1 报错"
                (lambda () (vt-trace (vt-from-sequence (list 1.0d0 2.0d0) :dtype :float64))))

  ;; ---- trace :offset 契约 ----
  (expect-equal "trace :offset 越界返回 0"
                (vt-to-list (vt-trace tm :offset 9)) 0.0d0)
  (expect-equal "trace :offset 负越界返回 0"
                (vt-to-list (vt-trace tm :offset -9)) 0.0d0)
  (expect-equal "trace 复数 dtype 保持"
                (vt-dtype (vt-trace tc :offset 1)) :complex128)
  (expect-equal "trace :axis1 单给报错"
                (handler-case (progn (vt-trace td :axis1 1) :no-error)
                  (error () :ok))
                :ok)

  ;; ---- trace :out（非连续写入口）----
  (let* ((base (vt-zeros '(4) :dtype :float64))
         (out (vt-slice base '(1))))  ; 0 维视图（带非零 offset）
    (expect-equal "trace :out（非连续 0 维视图）"
                  (vt-to-list (vt-trace tm :offset 1 :out out))
                  (reduce #'+ (vt-to-list (vt-diagonal tm :offset 1))))))

;;; ------------------------------------------------------------------
;;; 3. 实数回归网（默认路径不得改变）
;;; ------------------------------------------------------------------

(defun run-regressions ()
  (format t "~%~%--- 回归 ---~%")
  ;; 无 axis 的 vt-unique 与旧语义逐位一致
  (expect-equal "vt-unique 默认（展平）"
                (vt-to-list (vt-unique (vt-from-sequence (list 3.0d0 1.0d0 3.0d0) :dtype :float64)))
                (list 1.0d0 3.0d0))
  (expect-equal "vt-unique 默认含 NaN 折叠"
                (length (vt-to-list (vt-unique uc)))
                4)
  ;; 无 offset 的 vt-trace 与 vt-diagonal 之和一致
  (expect-equal "vt-trace 默认 = diag 之和"
                (vt-to-list (vt-trace tm))
                (reduce #'+ (vt-to-list (vt-diagonal tm))))
  (expect-equal "vt-trace 复数默认"
                (vt-to-list (vt-trace tc))
                (reduce #'+ (vt-to-list (vt-diagonal tc)))))

;;; ------------------------------------------------------------------
;;; 入口
;;; ------------------------------------------------------------------

(format t "~&=====================================~%")
(format t "按轴去重 / 迹 测试（通用特性补齐）~%")
(format t "=====================================~%")
(run-differential)
(run-contracts)
(run-regressions)
(format t "~&=====================================~%")
(format t "Total: ~d / Pass: ~d / Fail: ~d~%" (+ *pass* *fail*) *pass* *fail*)
(format t "PASS=~d FAIL=~d~%" *pass* *fail*)
(when *failures*
  (format t "~&--- 失败清单 ---~%")
  (dolist (f (reverse *failures*)) (format t "  · ~a~%" f)))
(when (plusp *fail*) (sb-ext:exit :code 1))
