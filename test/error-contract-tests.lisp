;;;; error-contract-tests.lisp — 错误路径契约套件（C7 类目）
;;;;
;;;; 契约底线 L2（TEST-PLAN.md §1）：非法输入 → 确定性报错，绝不静默错值。
;;;; 每个契约错误都必须有对应断言；这里的期望全部来自 2026-10-03 探针实测。
;;;;
;;;; 注：`sum :out dtype 静默降精度` 是已知契约缺口（TEST-PLAN.md §6 #2），
;;;; 不在本套件中断言为错误——套件只固化已成立的契约。
;;;;
;;;; 运行：
;;;;   tmp/sbcl-2.6.8-install/bin/sbcl --noinform --non-interactive \
;;;;     --load clvt/test/error-contract-tests.lisp
;;;; 退出码 0 = 全部通过。

(asdf:load-system :clvt)
(in-package :clvt)

(defparameter *pass* 0)
(defparameter *fail* 0)

(defun check-error (label thunk)
  (declare (optimize (speed 0)))
  (handler-case
      (progn (funcall thunk)
             (incf *fail*)
             (format t "  FAIL ~a（未报错——违反 L2：非法输入静默通过）~%" label))
    (error ()
      (incf *pass*)
      (format t "  ok  ~a~%" label)))
  (finish-output))

(defun run ()
  (setf *pass* 0 *fail* 0)
  (format t "--- 错误路径契约（L2 底线）---~%")

  ;; —— 广播/形状 ——
  (check-error "+ (2 3)+(2 4) 形状不兼容"
               (lambda () (vt-+ (vt-zeros '(2 3)) (vt-zeros '(2 4)))))
  (check-error "+ (4 3)+(3 4) 广播不兼容"
               (lambda () (vt-+ (vt-zeros '(4 3)) (vt-zeros '(3 4)))))
  (check-error "reshape (6)->(2 2) 元素数不一致"
               (lambda () (vt-reshape (vt-zeros '(6)) '(2 2))))
  (check-error "reshape (0)->(1) 空→非空"
               (lambda () (vt-reshape (vt-zeros '(0)) '(1))))
  (check-error "transpose perm 长度 ≠ 秩"
               (lambda () (vt-transpose (vt-zeros '(2 3)) '(0 1 2))))
  (check-error "concatenate 秩不匹配"
               (lambda () (vt-concatenate 0 (vt-zeros '(2)) (vt-zeros '(2 2)))))
  (check-error "stack 输入形状不匹配"
               (lambda () (vt-stack 0 (vt-zeros '(2 2)) (vt-zeros '(3 2)))))

  ;; —— 线性代数 ——
  (check-error "matmul 内维冲突 (2 3)@(2 2)"
               (lambda () (vt-matmul (vt-zeros '(2 3)) (vt-zeros '(2 2)))))
  (check-error "einsum 标签维度冲突 ij,jk->ik"
               (lambda () (vt-einsum "ij,jk->ik" (vt-zeros '(2 3)) (vt-zeros '(4 2)))))

  ;; —— 索引 ——
  (check-error "ref 索引越界"
               (lambda () (vt-ref (vt-zeros '(2 3)) 5 0)))
  (check-error "take 索引越界"
               (lambda () (vt-take (vt-const '(3) 1d0) (vt-const '(2) 9d0 :dtype :int64))))
  (check-error "slice 步长 0"
               (lambda () (vt-slice (vt-zeros '(4)) '(nil nil 0))))
  (check-error "写入只读广播视图（dim>1 且 stride=0）"
               (lambda () (vt-copy-into (vt-broadcast-to (vt-zeros '(1 1)) '(2 2))
                                        (vt-ones '(2 2)))))

  ;; —— 数值域 ——
  (check-error "one-hot 类别越界"
               (lambda () (vt-one-hot (vt-const '(3) 5d0 :dtype :int64) 3)))
  (check-error "percentile 超出 [0,100]"
               (lambda () (vt-item (vt-percentile (vt-const '(4) 1d0) 150))))

  ;; —— 随机数参数校验（文档化契约）——
  (check-error "uniform low > high"
               (lambda () (vt-random-uniform '(3) :low 2d0 :high 1d0)))
  (check-error "uniform NaN low（契约达成，错误类型欠佳：FP 异常，见 TEST-PLAN §6 #4）"
               (lambda () (vt-random-uniform '(3) :low +vt-float-nan+)))

  ;; —— 已文档化的空归约约定（与 numpy 的分歧见 TEST-PLAN §5）——
  (check-error "argmax 空输入（约定：arg 归约无定义）"
               (lambda () (vt-argmax (vt-zeros '(0)))))

  ;; —— 算术 ——
  (check-error "整数除以 0（比 numpy 更严格：报错而非回 0）"
               (lambda () (vt-item (vt-/ (vt-const '(1) 3 :dtype :int64)
                                         (vt-const '(1) 0 :dtype :int64)))))

  (format t "~%通过 ~a / 失败 ~a~%" *pass* *fail*)
  (finish-output)
  (zerop *fail*))

(sb-ext:exit :code (if (run) 0 1))
