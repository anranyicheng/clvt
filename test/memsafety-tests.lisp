;;;; vt-sigmoid / vt-relu :out 快路径内存安全验证
(require :asdf)
#+quicklisp (ql:quickload :clvt)
(asdf:load-system :clvt)
(defpackage :clvt-memtest
  (:use :cl :clvt)
  (:export #:run))

(in-package :clvt-memtest)

;;; ------------------------------------------------------------------
;;; 计数器与失败清单
;;; ------------------------------------------------------------------

(defvar *pass* 0)
(defvar *fail* 0)
(defvar *failures* nil
  "每个失败项的简短描述，供末尾汇总。")

;;; ------------------------------------------------------------------
;;; 断言辅助
;;; ------------------------------------------------------------------

(defun expect-error (label thunk)
  "成功条件：THUNK 抛出错误。
   —— 用于断言「非法 :out 必须被拒绝」。"
  (let ((signaled-p nil)
        (signaled-cond nil)
        (returned-value :unset))
    (handler-case
        (setf returned-value (funcall thunk))
      (error (e) (setf signaled-p t signaled-cond e)))
    (cond
      (signaled-p
       (incf *pass*)
       (format t "  [ OK ] ~a~%         抛错: ~a~%" label (type-of signaled-cond))
       t)
      (t
       (incf *fail*)
       (push (format nil "~a — 期望抛错但静默返回 ~s" label returned-value)
             *failures*)
       (format t "  [FAIL] ~a~%         期望抛错，实际返回 ~s~%" label returned-value)
       nil))))

(defun expect-values (label thunk expected)
  "成功条件：THUNK 返回的值与 EXPECTED 相等（equal）。"
  (let ((actual :unset)
        (errored nil)
        (cond-object nil))
    (handler-case
        (setf actual (funcall thunk))
      (error (e) (setf errored t cond-object e)))
    (cond
      (errored
       (incf *fail*)
       (push (format nil "~a — 意外抛错: ~a" label cond-object) *failures*)
       (format t "  [FAIL] ~a~%         意外抛错: ~a~%" label cond-object)
       nil)
      ((equal actual expected)
       (incf *pass*)
       (format t "  [ OK ] ~a~%         值: ~s~%" label actual)
       t)
      (t
       (incf *fail*)
       (push (format nil "~a — got ~s, want ~s" label actual expected)
             *failures*)
       (format t "  [FAIL] ~a~%         got ~s, want ~s~%" label actual expected)
       nil))))

;;; ------------------------------------------------------------------
;;; 测试入口
;;; ------------------------------------------------------------------

(defun run ()
  "执行全部内存安全测试，返回 T（全过）/ NIL（有失败）。"
  (setf *pass* 0
        *fail* 0
        *failures* nil)
  (format t "~&===== :out 快路径内存安全测试（计数器版） =====~%")
  (finish-output)

  ;; ---------- T1：:out 尺寸小于输入 ----------
  (expect-error
   "T1 relu :out 尺寸过小 (4 -> 2)"
   (lambda ()
     (let ((x (vt-ones '(4) :dtype :float64))
           (y (vt-zeros '(2) :dtype :float64)))
       (vt-relu x :out y))))

  ;; ---------- T2：:out dtype 与输入不匹配 ----------
  (expect-error
   "T2 relu :out dtype=int32 而 a=float64"
   (lambda ()
     (let ((x (vt-from-sequence '(1d0 2d0 3d0 4d0)))
           (y (vt-zeros '(4) :dtype :int32)))
       (vt-relu x :out y))))

  ;; ---------- T3：sigmoid :out dtype 不匹配 ----------
  (expect-error
   "T3 sigmoid a=float64, :dtype=:float64, :out=float32"
   (lambda ()
     (let ((x (vt-from-sequence '(1d0 2d0 3d0 4d0)))
           (y (vt-zeros '(4) :dtype :float32)))
       (vt-sigmoid x :dtype :float64 :out y))))

  ;; ---------- T4：:out 形状不同但元素数相同 ----------
  (expect-error
   "T4 relu a.shape=(3 1), :out.shape=(3)"
   (lambda ()
     (let ((x (vt-ones '(3 1) :dtype :float64))
           (y (vt-zeros '(3) :dtype :float64)))
       (vt-relu x :out y))))

  ;; ---------- T5：合法用法（relu）----------
  (expect-values
   "T5 合法 relu :out (同形同型)"
   (lambda ()
     (let ((x (vt-from-sequence '(-1d0 2d0)))
           (y (vt-zeros '(2) :dtype :float64)))
       (vt-relu x :out y)
       (coerce (vt-data y) 'list)))
   '(0d0 2d0))

  ;; ---------- T6：合法用法（sigmoid）----------
  (expect-values
   "T6 合法 sigmoid :out (同形同型)"
   (lambda ()
     (let ((x (vt-from-sequence '(0d0 0d0)))
           (y (vt-zeros '(2) :dtype :float64)))
       (vt-sigmoid x :out y)
       (coerce (vt-data y) 'list)))
   '(0.5d0 0.5d0))

  ;; ---------- 汇总 ----------
  (format t "~&===== 汇总: ~a 通过 / ~a 失败 =====~%"
          *pass* *fail*)
  (when *failures*
    (format t "失败清单:~%")
    (dolist (f (nreverse *failures*))
      (format t "  - ~a~%" f)))
  (finish-output)
  (zerop *fail*))

(run)
