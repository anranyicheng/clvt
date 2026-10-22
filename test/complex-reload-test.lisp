;;;; complex-reload-test.lisp — S7 验收：重载单个源文件后复数语义不变
;;;;
;;;; 为什么这条是整改的**验收核心**
;;;; ------------------------------
;;;; 覆盖式架构最致命的不是「代码脏」，而是它**静默降级**：
;;;;   只重载 src/elementwise.lisp（改实数实现时最常见的动作），
;;;;   (vt-sqrt #C(-1 0)) 从 complex128 的 i 变成 float64 的 NaN，不报错。
;;;;
;;;; 单一 DEFUN 分派把这件事从「必然发生」变成「结构上不可能」：符号指向的
;;;; 具名函数自己就带分派，重载只是把它重建成同样的东西。但「结构上不可能」
;;;; 需要一条可执行的断言来守护，否则下次有人加个 `(setf (symbol-function …))`
;;;; 又会悄悄退回去。
;;;;
;;;; 做法：同一 image 里逐个 `load` 归属文件（源码，重新编译），每轮跑一遍
;;;; 16 个探针。探针覆盖第十三轮迁移的全部 16 个符号 + 第十二轮的三个代表。
;;;;
;;;; 注意：本套件会重定义全局函数，**必须排在最后**（run-tests.sh 的
;;;; DEFAULT_ORDER 已如此安排）。

;; 独立运行（不经 run-tests.sh）时：用 quicklisp 的 ASDF（`(require :asdf)`
;; 会拉起 SBCL 自带的 2.x 再走 UPGRADE-ASDF 编译，反而更慢更脆）。
(unless (find-package :clvt)
  (load (merge-pathnames "quicklisp/setup.lisp" (user-homedir-pathname)))
  (push (uiop:pathname-parent-directory-pathname
           (uiop:pathname-directory-pathname *load-truename*))
        asdf:*central-registry*)
  (handler-bind ((warning #'muffle-warning)) (asdf:load-system :clvt)))
(in-package :clvt)

(defvar *pass* 0)
(defvar *fail* 0)

(defun report (label ok)
  (if ok (incf *pass*) (incf *fail*))
  (format t "~&[~a] ~a~%" (if ok "PASS" "FAIL") label))

(defvar *root* (uiop:pathname-parent-directory-pathname
                 (uiop:pathname-directory-pathname *load-truename*)))

(defun near (a b &optional (eps 1d-9))
  (< (abs (- a b)) eps))

(defun czero (x)
  (and (numberp x) (near (realpart x) 0d0) (near (imagpart x) 0d0)))

(defun cval (x)
  "取单元素张量的标量值（0 维或 1 元素）。"
  (vt-item x))

(defun vlist (x)
  "1 维张量 → simple-vector（便于与字面量逐元素比）。"
  (coerce (vt-to-list x) 'vector))

;;; ------------------------------------------------------------------
;;; 探针：每个返回 T/NIL，不抛异常（异常由调用方 ignore-errors 兜住）
;;; ------------------------------------------------------------------
(defparameter *probes*
  (list
   ;; —— 第十二轮代表（回归保护）——
   (cons "sqrt(-1+0i) → complex128 的 i"
         (lambda ()
           (let ((r (vt-sqrt (vt-complex -1.0d0 0.0d0))))
             (and (eq (vt-dtype r) :complex128)
                  (near (realpart (cval r)) 0d0)
                  (near (imagpart (cval r)) 1d0)))))
   (cons "sum 复数保持 complex128"
         (lambda ()
           (and (eq (vt-dtype (vt-sum (vt-from-sequence (list #C(1d0 2d0)))))
                    :complex128)
                (near (cval (vt-sum (vt-from-sequence (list #C(1d0 2d0)))))
                      #C(1d0 2d0)))))
   (cons "maximum 复数字典序"
         (lambda ()
           (near (cval (vt-maximum (vt-from-sequence (list #C(1d0 2d0)))
                                   (vt-from-sequence (list #C(0d0 9d0)))))
                 #C(1d0 2d0))))

   ;; —— 第十三轮 16 个符号 ——
   (cons "vt-@ 复数"
         (lambda ()
           (near (cval (vt-@ (vt-from-sequence '((#C(1d0 2d0))) :dtype :complex128)
                             (vt-from-sequence '((#C(3d0 0d0))) :dtype :complex128)))
                 #C(3d0 6d0))))
   (cons "vt-expt 复数"
         (lambda ()
           (let ((z (cval (vt-expt (vt-complex 2.0d0 1.0d0) 2))))
             (and (near (realpart z) 3d0) (near (imagpart z) 4d0)))))
   (cons "vt-real / vt-imag 复数"
         (lambda ()
           (let ((z (vt-from-sequence (list #C(1d0 2d0)))))
             (and (eq (vt-dtype (vt-real z)) :float64)
                  (near (cval (vt-real z)) 1d0)
                  (near (cval (vt-imag z)) 2d0)))))
   (cons "vt-conj 复数"
         (lambda () (near (cval (vt-conj (vt-from-sequence (list #C(1d0 2d0)))))
                          #C(1d0 -2d0))))
   (cons "vt-angle 复数"
         (lambda ()
           (near (cval (vt-angle (vt-from-sequence (list #C(0d0 1d0)))))
                 (/ pi 2))))
   (cons "vt-log 复数（无 base）"
         (lambda ()
           (let ((z (cval (vt-log (vt-complex -1.0d0 0.0d0)))))
             (and (eq (vt-dtype (vt-log (vt-complex -1.0d0 0.0d0))) :complex128)
                  (near (realpart z) 0d0) (near (imagpart z) pi)))))
   (cons "vt-log 复数 :base（第十三轮修掉的除零）"
         (lambda () (near (realpart (cval (vt-log (vt-complex 8.0d0 0.0d0)
                                                  :base 2.0d0)))
                          3d0)))
   (cons "vt-float-power 复数"
         (lambda ()
           (let ((z (cval (vt-float-power (vt-complex 2.0d0 1.0d0)
                                          (vt-complex 2.0d0 0.0d0)))))
             (and (near (realpart z) 3d0) (near (imagpart z) 4d0)))))
   (cons "vt-view :dtype 重解释"
         (lambda ()
           (let ((z (vt-view (vt-from-sequence '(1d0 2d0) :dtype :float64)
                             :dtype :complex128)))
             (and (eq (vt-dtype z) :complex128)
                  (near (cval z) #C(1d0 2d0))))))
   (cons "vt-average 复数（等权）"
         (lambda ()
           (near (cval (vt-average (vt-from-sequence (list #C(2d0 2d0)))))
                 #C(2d0 2d0))))
   (cons "vt-clip 复数字典序"
         (lambda ()
           (equalp (vlist (vt-clip (vt-from-sequence
                                    (list #C(1d0 2d0) #C(3d0 -4d0) #C(5d0 0d0))
                                    :dtype :complex128)
                                  0 2))
                   (vector #C(1d0 2d0) #C(2d0 0d0) #C(2d0 0d0)))))
   (cons "vt-clamp 复数（无边界 = 副本）"
         (lambda ()
           (near (cval (vt-clamp (vt-from-sequence (list #C(1d0 2d0))
                                                  :dtype :complex128)))
                 #C(1d0 2d0))))
   (cons "vt-fromiter 复数推断 dtype"
         (lambda ()
           (and (eq (vt-dtype (vt-fromiter (list #C(1d0 2d0)))) :complex128)
                (near (cval (vt-fromiter (list #C(1d0 2d0)))) #C(1d0 2d0)))))
   (cons "vt-unique 复数去重"
         (lambda ()
           (equalp (vlist (vt-unique (vt-from-sequence
                                      (list #C(3d0 0d0) #C(1d0 2d0) #C(3d0 0d0))
                                      :dtype :complex128)))
                   (vector #C(1d0 2d0) #C(3d0 0d0)))))
   (cons "vt-nanmedian 复数"
         (lambda ()
           (near (cval (vt-nanmedian
                        (vt-from-sequence
                         (list #C(1d0 0d0) (complex (vt-get-nan :float64) 0d0)
                               #C(3d0 0d0))
                         :dtype :complex128)))
                 #C(2d0 0d0))))
   (cons "vt-matrix-rank 复数"
         (lambda ()
           (= 2 (vt-matrix-rank
                 (vt-from-sequence '((#C(1d0 0d0) #C(0d0 1d0))
                                     (#C(1d0 0d0) #C(0d0 -1d0)))
                                   :dtype :complex128)))))))

(defun run-probes ()
  "返回失败的探针标签列表。"
  (let (bad)
    (dolist (pr *probes*)
      (unless (ignore-errors (funcall (cdr pr)))
        (push (car pr) bad)))
    (nreverse bad)))

(defun report-round (label)
  (let ((bad (run-probes)))
    (report (format nil "~a：~d 个探针全过~@[；失败 ~d 个: ~a~]"
                    label (length *probes*) (length bad) bad)
            (null bad))))

;;; ------------------------------------------------------------------
;;; 1. 基线
;;; ------------------------------------------------------------------
(report-round "基线（完整加载后）")

;;; ------------------------------------------------------------------
;;; 2. 逐文件重载
;;; ------------------------------------------------------------------
;;; 覆盖全部「曾经写过公开符号」的文件 + 内核文件。任何一个出现降级
;;; 都说明分派逻辑又回到了符号替换上。
(defparameter *reload-targets*
  '("src/dispatch.lisp"
    "src/complex-kernels.lisp"
    "src/manip.lisp"
    "src/elementwise.lisp"
    "src/reduce-stats.lisp"
    "src/setops.lisp"
    "src/linalg.lisp"
    "src/extensions.lisp"
    "src/extensions3.lisp"
    "src/complex.lisp"
    "src/complex-ops.lisp"
    "src/complex-linalg.lisp"))

(dolist (rel *reload-targets*)
  (let ((path (merge-pathnames rel *root*)))
    (if (not (probe-file path))
        (report (format nil "重载 ~a：文件不存在" rel) nil)
        (progn
          (handler-bind ((warning #'muffle-warning))
            (load path))
          (report-round (format nil "重载 ~a 后" rel))))))

;;; ------------------------------------------------------------------
;;; 3. 结构判据（与 complex-dispatch-test 互补：这里断言「没有任何符号被
;;;    替换成 setf/lambda」，那边断言「159 个符号是具名 DEFUN」）
;;; ------------------------------------------------------------------
(report "全库无 (setf (symbol-function …)) 形式的运行期替换"
        (let ((n 0))
          (dolist (rel (directory (merge-pathnames "src/*.lisp" *root*)))
            (with-open-file (in rel)
              (loop for line = (read-line in nil)
                    while line
                    do (when (search "(setf (symbol-function" line)
                         ;; 只算代码行，注释行（以 ; 开头的行首注释）不算
                         (let ((trimmed (string-left-trim " " line)))
                           (unless (char= (char trimmed 0) #\;)
                             (incf n)))))))
          (zerop n)))

(format t "~&~%PASS=~d FAIL=~d~%" *pass* *fail*)
(sb-ext:exit :code (if (zerop *fail*) 0 1))
