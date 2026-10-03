;;;; shape-degenerate-tests.lisp — 空形状/退化形状表驱动套件（C1/C2 类目）
;;;;
;;;; 规划依据：test/TEST-PLAN.md §2 类别矩阵；期望值全部来自
;;;; 2026-10-03 探针实测（tmp/probe-edge.lisp）并与 numpy 语义对表。
;;;; 与 numpy 的分歧（mean/amax 空归约 NaN、argmax 空输入报错）
;;;; 为文档化设计约定，见 TEST-PLAN.md §5。
;;;;
;;;; bug 回归：bug-stack-empty-ax1 —— stack axis=1 堆叠空 1D 曾泄漏
;;;; CL 底层序列错误（vt-copy-into 零尺寸写入未早退），本轮修复。
;;;;
;;;; 运行：
;;;;   tmp/sbcl-2.6.8-install/bin/sbcl --noinform --non-interactive \
;;;;     --load clvt/test/shape-degenerate-tests.lisp
;;;; 退出码 0 = 全部通过。


(asdf:load-system :clvt)
(in-package :clvt)

;;; ------------------------------------------------------------------
;;; 断言辅助（沿用 out-contig-tests 风格）
;;; ------------------------------------------------------------------

(defparameter *pass* 0)
(defparameter *fail* 0)

(defun check (label cond)
  (if cond
      (progn (incf *pass*) (format t "  ok  ~a~%" label))
      (progn (incf *fail*) (format t "  FAIL ~a~%" label)))
  (finish-output))

(defun check-error (label thunk)
  (check label (handler-case (progn (funcall thunk) nil)
                 (error () t))))

(defun %flatten (x)
  (if (atom x) (list x) (mapcan #'%flatten x)))

(defun check-list (label got expected &optional (tol 1e-12))
  "嵌套 list 数值近似比较（flatten 后逐元素 |a-b|<=tol）。"
  (let ((g (%flatten got)) (e (%flatten expected)))
    (check label
           (and (= (length g) (length e))
                (every (lambda (a b)
                         (and (numberp a) (numberp b) (<= (abs (- a b)) tol)))
                       g e)))))

(defun check-shape (label tensor expected-shape)
  (check label (equal (vt-shape tensor) expected-shape)))

(defun %test-nan-p (x) (and (numberp x) (with-float-safe (not (= x x)))))

(defun check-shape-and-nan (label tensor expected-shape)
  "断言形状正确且每个元素都是 NaN（空归约 NaN 约定）。"
  (check label
         (and (equal (vt-shape tensor) expected-shape)
              (every #'%test-nan-p (%flatten (vt-to-list tensor))))))

;;; ------------------------------------------------------------------
;;; 表驱动执行器：期望 = (:ok 值检查) 或 (:error)
;;; ------------------------------------------------------------------
(defun run-case (label expectation thunk)
  (ecase (car expectation)
    (:error (check-error label thunk))
    (:ok
     (handler-case
         (let ((r (funcall thunk)))
           (case (cadr expectation)
             (:shape  (check-shape (format nil "~a [shape]" label) r (caddr expectation)))
             (:value  (check-list (format nil "~a [value]" label)
                                  (vt-to-list r) (caddr expectation)))
             (:scalar (if (eq (caddr expectation) 'nan)
                          (check (format nil "~a [nan约定]" label) (%test-nan-p r))
                          (check-list (format nil "~a [scalar]" label) (list r)
                                      (list (caddr expectation)))))
             (:nan    (check-shape-and-nan (format nil "~a [nan约定]" label) r (caddr expectation))))
           (when (and (vt-p r) (eq (cadr expectation) :shape))
             ;; 形状检查同时隐含"没崩"= L1 底线达成
             t))
       (error (e)
         (incf *fail*)
         (format t "  FAIL ~a（意外报错: ~a）~%" label e)
         (finish-output))))))

(defun run ()
  (format t "==== shape-degenerate-tests（C1 空形状 / C2 退化形状）====~%")

  ;; ================================================================
  ;; 1. 创建：空形状
  ;; ================================================================
  (run-case "zeros (0)"      '(:ok :shape (0))    (lambda () (vt-zeros '(0))))
  (run-case "zeros (0 3)"    '(:ok :shape (0 3))  (lambda () (vt-zeros '(0 3))))
  (run-case "zeros (3 0)"    '(:ok :shape (3 0))  (lambda () (vt-zeros '(3 0))))
  (run-case "const (0 2)"    '(:ok :shape (0 2))  (lambda () (vt-const '(0 2) 5d0)))
  (run-case "arange 0"       '(:ok :shape (0))    (lambda () (vt-arange 0)))

  ;; ================================================================
  ;; 2. 逐元素：空形状（连续 + strided）
  ;; ================================================================
  (run-case "+ (0 3)+(0 3)"  '(:ok :shape (0 3))  (lambda () (vt-+ (vt-zeros '(0 3)) (vt-zeros '(0 3)))))
  (run-case "abs (3 0)"      '(:ok :shape (3 0))  (lambda () (vt-abs (vt-zeros '(3 0)))))
  (run-case "abs T(3 0)->(0 3)" '(:ok :shape (0 3))
            (lambda () (vt-abs (vt-transpose (vt-zeros '(3 0))))))
  ;; bug-stack-empty-ax1 的同族写入路径回归：(setf vt-slice) 空切片
  (let ((dst (vt-zeros '(0 2)))
        (src (vt-const '(0 1) 1d0)))
    (check "2.5 bug-stack-empty-ax1: (setf vt-slice) 零尺寸写入不崩"
           (handler-case (progn (setf (vt-slice dst '(:all) '(1 2)) src) t)
             (error (e) (format t "    泄漏错误: ~a~%" e) nil))))

  ;; ================================================================
  ;; 3. 归约：单位元 / NaN 约定 / arg 报错约定
  ;; ================================================================
  (run-case "sum (0)"            '(:ok :scalar 0.0d0)            (lambda () (vt-item (vt-sum (vt-zeros '(0))))))
  (run-case "sum (3 0)"          '(:ok :scalar 0.0d0)            (lambda () (vt-item (vt-sum (vt-zeros '(3 0))))))
  (run-case "sum (3 0) ax0"      '(:ok :shape (0))               (lambda () (vt-sum (vt-zeros '(3 0)) :axis 0)))
  (run-case "sum (3 0) ax1=0"    '(:ok :value (0.0d0 0.0d0 0.0d0)) (lambda () (vt-sum (vt-zeros '(3 0)) :axis 1)))
  (run-case "sum (0 4) ax0=0"    '(:ok :value (0.0d0 0.0d0 0.0d0 0.0d0)) (lambda () (vt-sum (vt-zeros '(0 4)) :axis 0)))
  (run-case "prod (0)=1"         '(:ok :scalar 1.0d0)            (lambda () (vt-item (vt-prod (vt-zeros '(0))))))
  (run-case "prod (3 0) ax1=1"   '(:ok :value (1.0d0 1.0d0 1.0d0)) (lambda () (vt-prod (vt-zeros '(3 0)) :axis 1)))
  (run-case "nansum (0)=0"       '(:ok :scalar 0.0d0)            (lambda () (vt-item (vt-nansum (vt-zeros '(0))))))
  ;; 设计约定（TEST-PLAN §5）：max/mean 族空归约 → NaN 信号
  (run-case "mean (0)=NaN约定"   '(:ok :scalar nan)             (lambda () (vt-item (vt-mean (vt-zeros '(0))))))
  (run-case "amax (0)=NaN约定"   '(:ok :scalar nan)             (lambda () (vt-item (vt-amax (vt-zeros '(0))))))
  (run-case "amax (3 0) ax1 NaN约定" '(:ok :nan (3))             (lambda () (vt-amax (vt-zeros '(3 0)) :axis 1)))
  (run-case "amax (3 0) ax0"     '(:ok :shape (0))               (lambda () (vt-amax (vt-zeros '(3 0)) :axis 0)))
  ;; 设计约定（TEST-PLAN §5）：arg 归约空输入一律报错
  (run-case "argmax (0) 报错约定"    '(:error) (lambda () (vt-argmax (vt-zeros '(0)))))
  (run-case "argmax (3 0) ax1 报错约定" '(:error) (lambda () (vt-argmax (vt-zeros '(3 0)) :axis 1)))
  (run-case "argmax (3 0) ax0 报错约定" '(:error) (lambda () (vt-argmax (vt-zeros '(3 0)) :axis 0)))
  (run-case "all (0)=T"   '(:ok :scalar 1) (lambda () (vt-item (vt-all (vt-zeros '(0))))))
  (run-case "any (0)=NIL" '(:ok :scalar 0) (lambda () (vt-item (vt-any (vt-zeros '(0))))))
  (run-case "cumsum (0)"  '(:ok :shape (0)) (lambda () (vt-cumsum (vt-zeros '(0)))))

  ;; ================================================================
  ;; 4. 线性代数：空 matmul 含零填充值检查
  ;; ================================================================
  (run-case "matmul (0 3)@(3 2)" '(:ok :shape (0 2))
            (lambda () (vt-matmul (vt-zeros '(0 3)) (vt-zeros '(3 2)))))
  (run-case "matmul (2 0)@(0 3)=零" '(:ok :value ((0.0d0 0.0d0 0.0d0) (0.0d0 0.0d0 0.0d0)))
            (lambda () (vt-matmul (vt-zeros '(2 0)) (vt-zeros '(0 3)))))
  (run-case "dot (0)x(0)=0"      '(:ok :scalar 0.0d0)
            (lambda () (vt-item (vt-dot (vt-zeros '(0)) (vt-zeros '(0))))))
  (run-case "outer (0)x(3)"      '(:ok :shape (0 3))
            (lambda () (vt-outer (vt-zeros '(0)) (vt-zeros '(3)))))
  (run-case "trace (0 0)=0"      '(:ok :scalar 0.0d0)
            (lambda () (vt-item (vt-trace (vt-zeros '(0 0))))))

  ;; ================================================================
  ;; 5. 形变：reshape / transpose / flatten / squeeze
  ;; ================================================================
  (run-case "reshape (0 3)->(3 0)" '(:ok :shape (3 0)) (lambda () (vt-reshape (vt-zeros '(0 3)) '(3 0))))
  (run-case "reshape (0)->(0 3)"   '(:ok :shape (0 3)) (lambda () (vt-reshape (vt-zeros '(0)) '(0 3))))
  (run-case "T (0 3)"              '(:ok :shape (3 0)) (lambda () (vt-transpose (vt-zeros '(0 3)))))
  (run-case "T (1 0 2)"            '(:ok :shape (2 0 1)) (lambda () (vt-transpose (vt-zeros '(1 0 2)))))
  (run-case "flatten (3 0)"        '(:ok :shape (0))   (lambda () (vt-flatten (vt-zeros '(3 0)))))
  (run-case "squeeze (1 0)"        '(:ok :shape (0))   (lambda () (vt-squeeze (vt-zeros '(1 0)))))
  (run-case "squeeze (1 0 2)"      '(:ok :shape (0 2)) (lambda () (vt-squeeze (vt-zeros '(1 0 2)))))
  (run-case "expand (0) ax0"       '(:ok :shape (1 0)) (lambda () (vt-expand-dims (vt-zeros '(0)) 0)))

  ;; ================================================================
  ;; 6. 拼接/堆叠：空与非空混合（值检查）+ bug-stack-empty-ax1 回归
  ;; ================================================================
  (run-case "concat ax0 (0 2)+(2 2)" '(:ok :value ((7.0d0 7.0d0) (7.0d0 7.0d0)))
            (lambda () (vt-concatenate 0 (vt-zeros '(0 2)) (vt-const '(2 2) 7d0))))
  (run-case "concat ax1 (2 0)+(2 3)" '(:ok :value ((7.0d0 7.0d0 7.0d0) (7.0d0 7.0d0 7.0d0)))
            (lambda () (vt-concatenate 1 (vt-zeros '(2 0)) (vt-const '(2 3) 7d0))))
  (run-case "stack ax0 two (0)"  '(:ok :shape (2 0)) (lambda () (vt-stack 0 (vt-zeros '(0)) (vt-zeros '(0)))))
  ;; bug-stack-empty-ax1：修复前泄漏 "bounding indices ... bad for a sequence of length 0"
  (run-case "bug-stack-empty-ax1: stack ax1 two (0)" '(:ok :shape (0 2))
            (lambda () (vt-stack 1 (vt-zeros '(0)) (vt-zeros '(0)))))
  (run-case "stack ax0 two (0 2)" '(:ok :shape (2 0 2)) (lambda () (vt-stack 0 (vt-zeros '(0 2)) (vt-zeros '(0 2)))))

  ;; ================================================================
  ;; 7. 其余形变/索引/排序：空张量 no-op
  ;; ================================================================
  (run-case "flip (0)"            '(:ok :shape (0))   (lambda () (vt-flip (vt-zeros '(0)))))
  (run-case "flip (0 2) ax0"      '(:ok :shape (0 2)) (lambda () (vt-flip (vt-zeros '(0 2)) :axis 0)))
  (run-case "tile (0) 2"          '(:ok :shape (0))   (lambda () (vt-tile (vt-zeros '(0)) 2)))
  (run-case "repeat (0) 3"        '(:ok :shape (0))   (lambda () (vt-repeat (vt-zeros '(0)) 3)))
  (run-case "roll (0) 2"          '(:ok :shape (0))   (lambda () (vt-roll (vt-zeros '(0)) 2)))
  (run-case "sort (0)"            '(:ok :shape (0))   (lambda () (vt-sort (vt-zeros '(0)))))
  (run-case "slice (0) [::2]"     '(:ok :shape (0))   (lambda () (vt-slice (vt-zeros '(0)) '(nil nil 2))))
  (run-case "slice (0) [::-1]"    '(:ok :shape (0))   (lambda () (vt-slice (vt-zeros '(0)) '(nil nil -1))))
  (run-case "slice (3 0) all all" '(:ok :shape (3 0)) (lambda () (vt-slice (vt-zeros '(3 0)) '(:all) '(:all))))
  (run-case "where (0)x3"         '(:ok :shape (0))
            (lambda () (vt-where (vt-zeros '(0) :dtype :int64) (vt-zeros '(0)) (vt-zeros '(0)))))
  ;; nonzero (0)：对标 numpy 返回单元素 tuple，首个张量 shape (0)
  (let ((r (vt-nonzero (vt-zeros '(0)))))
    (check "nonzero (0) → 1 个 (0) 张量"
           (and (= (length r) 1) (equal (vt-shape (car r)) '(0)))))

  ;; ================================================================
  ;; 8. NN / setops
  ;; ================================================================
  (run-case "relu (0 2)"          '(:ok :shape (0 2)) (lambda () (vt-relu (vt-zeros '(0 2)))))
  (run-case "sigmoid (0)"         '(:ok :shape (0))   (lambda () (vt-sigmoid (vt-zeros '(0)))))
  (run-case "softmax (0 2) ax1"   '(:ok :shape (0 2)) (lambda () (vt-softmax (vt-zeros '(0 2)) :axis 1)))
  (run-case "softmax (0)"         '(:ok :shape (0))   (lambda () (vt-softmax (vt-zeros '(0)))))
  (run-case "unique (0)"          '(:ok :shape (0))   (lambda () (vt-unique (vt-zeros '(0)))))
  (run-case "union1d (0)+(2,2,2)" '(:ok :value (2.0d0))
            (lambda () (vt-union1d (vt-zeros '(0)) (vt-const '(3) 2d0))))
  (run-case "setdiff1d (5,5,5)+(0)" '(:ok :value (5.0d0))
            (lambda () (vt-setdiff1d (vt-const '(3) 5d0) (vt-zeros '(0)))))
  (run-case "intersect1d (0)+(1,1)" '(:ok :shape (0))
            (lambda () (vt-intersect1d (vt-zeros '(0)) (vt-const '(2) 1d0))))

  ;; ================================================================
  ;; 9. C2：size-1 / 广播 / 内部空维
  ;; ================================================================
  (run-case "(1 3)+(3 1) 广播" '(:ok :value ((2.0d0 3.0d0 4.0d0) (2.0d0 3.0d0 4.0d0) (2.0d0 3.0d0 4.0d0)))
            (lambda () (vt-+ (vt-from-sequence '((1d0 2d0 3d0)))
                             (vt-from-sequence '((1d0) (1d0) (1d0))))))
  (let ((bc (vt-broadcast-to (vt-const '(1) 2d0) '(3 4))))
    (run-case "广播零stride读视图参与 +"
              '(:ok :value ((3.0d0 3.0d0 3.0d0 3.0d0) (3.0d0 3.0d0 3.0d0 3.0d0) (3.0d0 3.0d0 3.0d0 3.0d0)))
              (lambda () (vt-+ bc (vt-ones '(3 4))))))
  (run-case "sum (1 3) ax1 keepdims" '(:ok :value ((6.0d0)))
            (lambda () (vt-sum (vt-from-sequence '((1d0 2d0 3d0))) :axis 1 :keepdims t)))
  (run-case "内部空维 (1 0 2) sum"   '(:ok :scalar 0.0d0) (lambda () (vt-item (vt-sum (vt-zeros '(1 0 2))))))
  (run-case "内部空维 reshape (1 0 2)->(0 2)" '(:ok :shape (0 2)) (lambda () (vt-reshape (vt-zeros '(1 0 2)) '(0 2))))
  (run-case "内部空维 concat ax0"    '(:ok :shape (2 0 2)) (lambda () (vt-concatenate 0 (vt-zeros '(1 0 2)) (vt-zeros '(1 0 2)))))

  ;; ================================================================
  ;; 汇总
  ;; ================================================================
  (format t "~%通过 ~a / 失败 ~a~%" *pass* *fail*)
  (finish-output)
  (zerop *fail*))

(sb-ext:exit :code (if (run) 0 1))
