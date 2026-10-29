;;;; example-smoke-test.lisp
;;;; =============================================================================
;;;; 把 example/ 下的两个示例脚本当作回归门禁跑一遍。
;;;;
;;;; 为什么需要这个套件
;;;; ----
;;;; example/ 里两个文件此前只是「手动跑跑看」的脚本，不进任何自动化：
;;;;
;;;;   1. example.lisp —— 106 个 test-* 函数，6898 行，覆盖大量公开 API 的
;;;;      真实用法。它内部用 ASSERT，任一断言失败即抛错中断。
;;;;   2. clvt-perf-test-examples.lisp —— 96 条正确性断言 + 约 60 项性能基准。
;;;;
;;;; 结果就是：库改了、示例坏了，没人知道。引入 :bool 之后跑这两个文件，
;;;; 立刻翻出三处示例自身的问题（详见 CONVENTIONS §17.7）：
;;;;
;;;;   · 9 处 `(assert (vt-all X))` 恒真 —— ASSERT 只判条件是否为 NIL，而
;;;;     `vt-all` 返回的是 VT 对象，任何 VT 对象都不为 NIL。这条断言在
;;;;     `:int8` 时代就已经什么都没检查，换成 `:bool` 之后更隐蔽；
;;;;   · 其中 2 条改成真断言后立刻失败：作者对「全 NaN 时 nanmax/nanmin
;;;;     返回什么」的猜测是 ±Inf，而 numpy（以及 clvt）返回 NaN；
;;;;   · 基准里 `(vt-extract a mask)` 参数顺序反了（对标 numpy.extract
;;;;     (condition, arr)），测出来的不是 a[mask]，还慢了 10 倍。
;;;;
;;;; 本套件只跑断言，不跑性能基准（基准的绝对值随机器漂移，不适合做门禁；
;;;; 需要时手动 `(clvt-examples:run)`）。
;;;;
;;;; 失败即抛错 → 运行器按 exit code 判 FAIL；成功则打印 PASS=/FAIL= 供解析。
;;;; =============================================================================

(defpackage :clvt-example-smoke
  (:use :cl)
  (:export #:main))

(in-package :clvt-example-smoke)

(defparameter *pass* 0)
(defparameter *fail* 0)

;; 两个示例文件都会 (IN-PACKAGE :CLVT) / (IN-PACKAGE :CLVT-EXAMPLES)，LOAD 之后
;; 当前包就被换掉了；因此每次 LOAD 完都要显式切回来，否则下面的 RECORD 调用
;; 会按 CLVT 包的符号去找，报 undefined function。
(defparameter *root*
  (make-pathname :name nil :type nil :defaults *load-truename*)
  "本文件所在目录（test/）的 pathname 目录形态。")

(defun example-dir ()
  (merge-pathnames "../example/" *root*))

(defun record (ok label &optional detail)
  (if ok
      (progn (incf *pass*)
             (format t "~&  ok   ~a~%" label))
      (progn (incf *fail*)
             (format t "~&  FAIL ~a~@[~%       ~a~]~%" label detail))))

;;; ---------------------------------------------------------------------------
;;; 1. example/example.lisp
;;; ---------------------------------------------------------------------------

(format t "~&~%=== example.lisp（106 个 test-* 函数，内部用 ASSERT）===~%")
(let ((ok nil) (detail nil))
  (handler-case
      (progn
        (load (merge-pathnames "example.lisp" (example-dir)))
        (in-package :clvt-example-smoke)
        (funcall (intern "RUN-ALL-TESTS" :clvt))
        ;; 跑到 "all test passed" 说明 106 个测试函数全部执行完毕且无断言失败。
        (setf ok t))
    (error (e)
      (in-package :clvt-example-smoke)
      (setf ok nil detail (princ-to-string e))))
  (record ok "example.lisp: 全部 test-* 通过（含 9 条新生效的 vt-all 断言）" detail))

;;; ---------------------------------------------------------------------------
;;; 2. example/clvt-perf-test-examples.lisp（只跑断言，不跑基准）
;;; ---------------------------------------------------------------------------

(format t "~&~%=== clvt-perf-test-examples.lisp（正确性断言）===~%")
(let ((ok nil) (detail nil) (n 0))
  (handler-case
      (progn
        (load (merge-pathnames "clvt-perf-test-examples.lisp" (example-dir)))
        (in-package :clvt-example-smoke)
        (setf ok (funcall (intern "RUN" :clvt-examples) :bench nil))
        ;; RUN 的返回值是 T/NIL，但它自己的计数更细，直接搬运。
        (setf n (symbol-value (intern "*PASS*" :clvt-examples)))
        (incf *pass* n)
        (incf *fail* (symbol-value (intern "*FAIL*" :clvt-examples))))
    (error (e)
      (in-package :clvt-example-smoke)
      (setf ok nil detail (princ-to-string e))))
  (record ok (format nil "perf-examples: ~d 条断言（含 bool 掩码/提取一组）" n) detail))

;;; ---------------------------------------------------------------------------

(in-package :clvt-example-smoke)
;; 两种汇总格式都印：Total/Pass/Fail 供 test/run-tests.sh 解析计数，
;; PASS=/FAIL= 供人眼与其它工具读取。
(format t "~&~%Total: ~d  Pass: ~d  Fail: ~d~%" (+ *pass* *fail*) *pass* *fail*)
(format t "~&PASS=~d FAIL=~d~%" *pass* *fail*)
(when (plusp *fail*)
  (format t "~&example smoke test 失败~%")
  (sb-ext:exit :code 1))
(format t "~&example smoke test 全部通过~%")
