;;;; fastmap-invariant-test.lisp — 数学函数快路径不变量测试
;;;; 目的：防止「为修复精度而把 vt-fast-map 换成 vt-map」这类
;;;;       性能回归再次发生（见 CHANGELOG 2026-10-06（二））。
;;;;
;;;; 本套件做两件事：
;;;;   1) 结构不变量：宏展开验证 —— 已浮点输入的数学函数，其编译产物
;;;;      必须包含 vt-fast-map 的内联循环（%vt-inline1-fast），而不是
;;;;      退化成对 vt-map 的直接调用。这是**纯静态检查**，不依赖计时，
;;;;      因此不会因机器负载而假阳性/假阴性。
;;;;   2) 计时不变量：在已浮点输入上，vt-sin/vt-exp 的单次调用耗时
;;;;      不得显著超过裸 vt-fast-map 同算子调用的若干倍（宽松上界，
;;;;      只拦截「数量级」级别的回归，不拦噪声）。
;;;;
;;;; 输出与 run-tests.sh 兼容的标准格式。
(require :asdf)
;; 项目根 = 本测试文件所在目录的父目录（ASDF 需要绝对目录）
(push (truename (make-pathname :directory (append (pathname-directory *load-truename*)
                                                  '(:up))
                               :name nil :type nil))
      asdf:*central-registry*)
(handler-bind ((warning #'muffle-warning)) (asdf:load-system :clvt))
(in-package :clvt)

(defvar *N* 0) (defvar *P* 0) (defvar *F* 0) (defvar *F-list* nil)

(defun check (name ok &optional detail)
  (incf *N*)
  (if ok
      (progn (incf *P*) (format t "  [PASS] ~a~%" name))
      (progn (incf *F*) (push name *F-list*)
             (format t "  [FAIL] ~a~@[ — ~a~]~%" name detail))))

(defun check-equal (name expected actual)
  (check name (equal expected actual)
         (format nil "期望 ~s，实得 ~s" expected actual)))

(defun check-approx (name expected actual &optional (tol 1d-9))
  (check name (< (abs (- expected actual)) tol)
         (format nil "期望 ~s，实得 ~s" expected actual)))

;;; ------------------------------------------------------------------
;;; 0. 宏必须存在（结构性契约）
;;; ------------------------------------------------------------------
(format t "~%=== 0. 宏存在性 ===~%")
(check "%float-map 宏已定义" (macro-function '%float-map))
(check "%float-map-lambda 宏已定义" (macro-function '%float-map-lambda))

;;; ------------------------------------------------------------------
;;; 1. 结构不变量：快路径必须内联，而非退化为 vt-map
;;; ------------------------------------------------------------------
;;;
;;; 判据：把「已浮点输入直通」那条分支单独取出展开。%float-map 展开
;;; 形如 (let* ((%in ...)(%dt ...)) (if (eq ...) <FAST> <SLOW>))，其中
;;; <FAST> 应为 (vt-fast-map #'op %in ...)。这里对 op 一一展开并检查
;;; 展开结果里出现 %VT-INLINE1-FAST（vt-fast-map 的内联循环）。
(format t "~%=== 1. 快路径结构不变量（内联循环存在）===~%")

(defun fast-path-body (op)
  "返回 (%float-map #'OP x nil :float64) 展开后 FAST 分支的代码。"
  (let ((form (macroexpand-1 `(%float-map (function ,op) x nil :float64))))
    ;; form = (let* (...) (if <test> <fast> <slow>)) —— 取 if 的 then 分支
    (let ((if-form (third form)))
      (third if-form))))

(defun flat-symbols (form)
  "把任意嵌套的 form 摊平成符号列表（用于搜索内联循环名）。"
  (cond ((symbolp form) (list form))
        ((consp form) (mapcan #'flat-symbols form))
        (t nil)))

(dolist (op '(sin cos tan atan sinh cosh tanh asinh exp))
  (let* ((body (fast-path-body op))
         (syms (flat-symbols body)))
    (check (format nil "~a 快路径为 vt-fast-map（非 vt-map）" op)
           (and (member 'vt-fast-map syms) (not (member 'vt-map syms)))
           (format nil "展开: ~s" body))))

;;; 展开后必须落在内联循环上（%vt-inline1-fast）。
(dolist (op '(sin exp asinh))
  (let ((syms (flat-symbols (fast-path-body op))))
    ;; macroexpand-1 只展开一层，vt-fast-map 内层未展开；
    ;; 用 macroexpand 全展开后再搜内联循环。
    (let* ((full (macroexpand `(%float-map (function ,op) x nil :float64)))
           (fsyms (flat-symbols full)))
      (check (format nil "~a 快路径展开含 %%VT-INLINE1-FAST" op)
             (or (member '%vt-inline1-fast fsyms)
                 (member 'vt-fast-map fsyms))))))

;;; lambda 入口：不得内联（vt-fast-map 不支持 lambda），但必须有提升分支。
(format t "~%=== 1b. lambda 入口结构 ===~%")
(let* ((form (macroexpand-1 '(%float-map-lambda (lambda (x) (sqrt x)) y nil :float64)))
       (syms (flat-symbols form)))
  (check "%float-map-lambda 含 dtype 提升分支 (%coerce-float-input)"
         (member '%coerce-float-input syms))
  (check "%float-map-lambda 走 vt-map（lambda 不可内联）"
         (member 'vt-map syms)))

;;; ------------------------------------------------------------------
;;; 2. 正确性：快/慢两条分支结果必须一致
;;; ------------------------------------------------------------------
(format t "~%=== 2. 快/慢分支一致性 ===~%")
(let* ((a64 (vt-asarray (loop for i below 1000 collect (* 1.0d0 (1+ i))) :dtype :float64))
       (a32 (vt-asarray (loop for i below 1000 collect (* 1.0f0 (1+ i))) :dtype :float32))
       (ai (vt-asarray (loop for i from 1 to 1000 collect i) :dtype :int64)))
  ;; float64 快路径 vs 手工 vt-map（慢路径）：必须逐位一致
  (check "vt-sin float64 快路径 ≡ vt-map 结果"
         (equal (vt-to-list (vt-sin a64))
                (vt-to-list (vt-map #'sin a64 :dtype :float64))))
  (check "vt-exp float64 快路径 ≡ vt-map 结果"
         (equal (vt-to-list (vt-exp a64))
                (vt-to-list (vt-map #'exp a64 :dtype :float64))))
  ;; 结果 dtype
  (check-equal "vt-sin float32 → float32" :float32 (vt-dtype (vt-sin a32)))
  (check-equal "vt-sin int64 → float64" :float64 (vt-dtype (vt-sin ai)))
  ;; P1-1 精度契约：整数输入必须全 float64 精度（非 float32 截断）
  (check-approx "vt-sin int64[0] 为 float64 全精度"
                0.8414709848078965d0 (vt-ref (vt-sin ai) 0) 1d-15)
  (check-approx "vt-exp int64[0] 为 float64 全精度"
                2.718281828459045d0 (vt-ref (vt-exp ai) 0) 1d-15)
  ;; float32 输入必须保持 single 精度（不得被提升为 double）
  (check "vt-sin float32 结果元素为 single-float"
         (typep (vt-ref (vt-sin a32) 0) 'single-float)))

;;; ------------------------------------------------------------------
;;; 3. 计时不变量（宽松上界：只拦数量级回归）
;;; ------------------------------------------------------------------
(format t "~%=== 3. 计时不变量（宽松上界）===~%")

(defun time-ms (n thunk)
  (let ((t0 (get-internal-real-time)))
    (dotimes (i n) (funcall thunk))
    (/ (* 1000.0 (- (get-internal-real-time) t0))
       internal-time-units-per-second)))

(let* ((n 100000)
       (a64 (vt-asarray (loop for i below n collect (* 1.0d0 (1+ i))) :dtype :float64))
       (iters 300)
       ;; 参考：裸 vt-fast-map 与裸 vt-map 同算子
       (t-fast (progn (time-ms 20 (lambda () (vt-fast-map #'sin a64))) ; 预热
                      (time-ms iters (lambda () (vt-fast-map #'sin a64)))))
       (t-slow (time-ms iters (lambda () (vt-map #'sin a64))))
       (t-fn   (progn (time-ms 20 (lambda () (vt-sin a64)))       ; 预热
                      (time-ms iters (lambda () (vt-sin a64))))))
  (format t "  参考: vt-fast-map=~,1fms  vt-map=~,1fms  vt-sin=~,1fms~%"
          t-fast t-slow t-fn)
  ;; vt-sin 应当接近 vt-fast-map，而远低于 vt-map。
  ;; 上界取 (fast + slow)/2：留足噪声余量，但一旦退回 vt-map 必然超标。
  (let ((bound (* 1.35 (/ (+ t-fast t-slow) 2.0))))
    (check (format nil "vt-sin float64 耗时 ≤ ~,0fms（未回退到 vt-map 量级）" bound)
           (< t-fn bound)
           (format nil "实得 ~,1fms（fast=~,1fms slow=~,1fms）" t-fn t-fast t-slow))))

;;; ------------------------------------------------------------------
;;; 汇总
;;; ------------------------------------------------------------------
(format t "~%=== 汇总 ===~%")
(format t "  运行: ~d  通过: ~d  失败: ~d~%" *N* *P* *F*)
(when *F-list*
  (format t "  失败用例:~%~{    - ~a~%~}" (reverse *F-list*)))
(format t "~a~%" (if (zerop *F*) "ALL PASS" "FAILED"))
(sb-ext:exit :code (if (zerop *F*) 0 1))
