(require :asdf)
#+quicklisp (ql:quickload :clvt)
(asdf:load-system :clvt)
(in-package :clvt)

;;;; test-ai-edge-cases.lisp — AI 主流函数语义回归测试
;;;; 覆盖 4 个已修复 bug 的最小 repro + 相关语义边界（对标 numpy/torch）：
;;;;   BUG-1 vt-matmul (N-D)@(1-D) / (1-D)@(N-D) 形状与数值
;;;;   BUG-2 广播 (0) op (1) 与空张量 softmax
;;;;   BUG-3 vt-binary-cross-entropy NaN 传播
;;;;   BUG-4 vt-vstack 1-D 提升 / vt-dstack 0-d 提升

(defparameter *failures* 0)
(defparameter *checks* 0)

(defmacro check (desc expected actual &key (test #'equalp))
  `(progn
     (incf *checks*)
     (let ((desc ,desc) (exp ,expected) (act ,actual))
       (if (funcall ,test exp act)
           (format t "PASS: ~a~%" desc)
           (progn
             (incf *failures*)
             (format t "FAIL: ~a~%  expected: ~s~%  actual:   ~s~%"
                     desc exp act))))))

(defmacro check-approx (desc expected actual &optional (tol 1d-9))
  `(progn
     (incf *checks*)
     (let* ((desc ,desc) (exp ,expected) (act ,actual))
       (if (and (= (length exp) (length act))
                (every (lambda (a b) (<= (abs (- (coerce a 'double-float)
                                                 (coerce b 'double-float)))
                                        ,tol))
                       exp act))
           (format t "PASS: ~a~%" desc)
           (progn
             (incf *failures*)
             (format t "FAIL: ~a~%  expected: ~s~%  actual:   ~s~%"
                     desc exp act))))))

(defun nan-p (x)
  (and (floatp x)
       (sb-int:with-float-traps-masked
           (:invalid :divide-by-zero :overflow :underflow)
         (not (= x x)))))

(defun flat (x)
  (if (atom x) (list x) (mapcan #'flat x)))

(defun approx-p (exp act tol)
  (let ((fe (flat exp)) (fa (flat act)))
    (and (= (length fe) (length fa))
         (every (lambda (a b) (<= (abs (- (coerce a 'double-float)
                                          (coerce b 'double-float)))
                                  tol))
                fe fa))))

(defmacro check-flat-approx (desc expected actual &optional (tol 1d-9))
  `(progn
     (incf *checks*)
     (let ((desc ,desc) (exp ,expected) (act ,actual))
       (if (approx-p exp act ,tol)
           (format t "PASS: ~a~%" desc)
           (progn
             (incf *failures*)
             (format t "FAIL: ~a~%  expected: ~s~%  actual:   ~s~%"
                     desc exp act))))))

(format t "===== BUG-1: vt-matmul (N-D)@(1-D) / (1-D)@(N-D) =====~%")
;; (2,3,4) @ (4,) → (2,3)，数值与手动 einsum 一致
(let* ((a (vt-from-sequence '((1 2 3 4) (5 6 7 8) (9 10 11 12)
                              (13 14 15 16) (17 18 19 20) (21 22 23 24))
                            :dtype :float64))
       (a3 (vt-reshape a '(2 3 4)))        ; batch=2, 每个 (3,4)
       (b (vt-from-sequence '(1 1 1 1) :dtype :float64)))
  (check "BUG-1a (2,3,4)@(4,) → shape (2,3)" '(2 3)
         (vt-shape (vt-matmul a3 b)))
  ;; 每个 batch: (3,4)@(4,) = 行和 → (2,3) = [[10,26,42],[58,74,90]]
  (check-flat-approx "BUG-1b (2,3,4)@(4,) 数值" '((10 26 42) (58 74 90))
                     (vt-to-list (vt-matmul a3 b)) 1d-9))
;; (3,) @ (3,3,2) → (3,2)：行向量对每个 batch 左乘
(let* ((v (vt-from-sequence '(1 0 0) :dtype :float64))
       (b (vt-from-sequence '(((1 2) (3 4) (5 6))
                              ((7 8) (9 10) (11 12))
                              ((13 14) (15 16) (17 18))) :dtype :float64)))
  (check "BUG-1c (3,)@(3,3,2) → shape (3,2)" '(3 2) (vt-shape (vt-matmul v b)))
  ;; 取每个 batch 的第一行
  (check-flat-approx "BUG-1d (3,)@(3,3,2) 数值" '((1 2) (7 8) (13 14))
                     (vt-to-list (vt-matmul v b)) 1d-9))
;; (3,) @ (3,4,5)：j=3 ≠ 4，numpy 同样报错 → 保持报错
(let ((err-expected
        (handler-case (progn (vt-matmul (vt-from-sequence '(1 2 3) :dtype :float64)
                                        (vt-reshape (vt-from-sequence '(1 2 3 4 5 6 7 8 9 10 11 12 13 14 15) :dtype :float64) '(3 4 5)))
                             nil)
          (error () t))))
  (check "BUG-1e (3,)@(3,4,5) 维度不匹配报错（与 numpy 一致）" t err-expected))
;; 既有 2d@1d 与 1d@2d 语义保持
(let ((a (vt-from-sequence '((1 2) (3 4)) :dtype :float64))
      (v (vt-from-sequence '(1 1) :dtype :float64)))
  (check-flat-approx "BUG-1f 2d@1d 语义保持" '(3 7) (vt-to-list (vt-matmul a v)) 1d-9)
  (check-flat-approx "BUG-1g 1d@2d 语义保持" '(4 6) (vt-to-list (vt-matmul v a)) 1d-9))

(format t "===== BUG-2: 广播 (0) op (1) 与空张量 softmax =====~%")
(let ((r (handler-case
             (vt-- (vt-from-sequence '() :dtype :float64)
                   (vt-from-sequence '(1.0d0)))
           (error (e) (format t "  crash: ~a~%" e) :crash))))
  (check "BUG-2a (0)-(1) 广播 → shape (0)" '(0)
         (if (eq r :crash) :crash (vt-shape r))))
(let ((s (handler-case
             (progn (vt-softmax (vt-from-sequence '() :dtype :float64)) :ok)
           (error (e) (format t "  crash: ~a~%" e) :crash))))
  (check "BUG-2b softmax 空向量 → ValueError（v0.3.6：scipy 对齐，内部 amax 空归约）"
         t (eq s :crash)))
;; (2,0) op (2,1) 同类广播（strided 路径）
(let ((r (handler-case
             (vt-- (vt-reshape (vt-from-sequence '() :dtype :float64) '(2 0))
                   (vt-from-sequence '((1.0d0) (2.0d0))))
           (error (e) (format t "  crash: ~a~%" e) :crash))))
  (check "BUG-2c (2,0)-(2,1) 广播 → shape (2,0)" '(2 0)
         (if (eq r :crash) :crash (vt-shape r))))
;; 正常广播不受空结果保护影响
(check-flat-approx "BUG-2d (3)-(1) 正常广播" '(0 1 2)
                   (vt-to-list (vt-- (vt-from-sequence '(1 2 3))
                                     (vt-from-sequence '(1.0d0)))) 1d-12)

(format t "===== BUG-3: vt-binary-cross-entropy NaN 传播 =====~%")
;; y=1, p=0 → -log(eps)
(check-flat-approx "BUG-3a BCE y=1,p=0 → -log(1e-7)"
                   (list (- (log 1d-7)))
                   (flat (vt-to-list
                          (vt-binary-cross-entropy (vt-from-sequence '(1.0d0))
                                                   (vt-from-sequence '(0.0d0)))))
                   1d-6)
;; p=NaN → NaN（torch.BCELoss 语义）
(let* ((nan (vt-get-nan :float64))
       (l (vt-binary-cross-entropy (vt-from-sequence '(1.0d0))
                                   (vt-from-sequence (list nan)))))
  (check "BUG-3b BCE p=NaN → NaN" t (nan-p (vt-to-list l))))
;; 正常值不变: y=[1,0], p=[0.9,0.1] → BCE = -(log 0.9 + log 0.1)/2
(check-flat-approx "BUG-3c BCE 正常值"
                   (list (- (log 0.9d0)))  ; y=1,p=0.9→-log0.9; y=0,p=0.1→-log(0.9)
                   (flat (vt-to-list
                          (vt-binary-cross-entropy (vt-from-sequence '(1.0d0 0.0d0))
                                                   (vt-from-sequence '(0.9d0 0.1d0)))))
                   1d-9)
;; y=NaN → NaN 传播
(let* ((nan (vt-get-nan :float64))
       (l (vt-binary-cross-entropy (vt-from-sequence (list nan))
                                   (vt-from-sequence '(0.9d0)))))
  (check "BUG-3d BCE y=NaN → NaN" t (nan-p (vt-to-list l))))

(format t "===== BUG-4: vt-vstack / vt-dstack 提升语义 =====~%")
;; vstack 1-D → (2,2)（torch.vstack/np.vstack）
(let ((r (vt-vstack (vt-from-sequence '(1 2)) (vt-from-sequence '(3 4)))))
  (check "BUG-4a vstack 1-D → shape (2,2)" '(2 2) (vt-shape r))
  (check-flat-approx "BUG-4b vstack 1-D 数值" '((1 2) (3 4))
                     (vt-to-list r) 1d-12))
;; vstack 2-D 语义保持 → (4,2)
(let ((r (vt-vstack (vt-from-sequence '((1 2) (3 4))) (vt-from-sequence '((5 6) (7 8))))))
  (check "BUG-4c vstack 2-D → shape (4,2)" '(4 2) (vt-shape r)))
;; dstack 0-d → (1,1,2)（numpy 语义）
(let ((r (vt-dstack (vt-from-sequence '(1) :dtype :float64) (vt-from-sequence '(2) :dtype :float64))))
  (check "BUG-4d dstack 0-d → shape (1,1,2)" '(1 1 2) (vt-shape r))
  (check-flat-approx "BUG-4e dstack 0-d 数值" '((1 2)) (vt-to-list r) 1d-12))
;; hstack 1-D 语义保持 → (4,)
(let ((r (vt-hstack (vt-from-sequence '(1 2)) (vt-from-sequence '(3 4)))))
  (check "BUG-4f hstack 1-D → shape (4)" '(4) (vt-shape r)))

(format t "===== 相关语义回归（审查确认正确的行为保持） =====~%")
;; softmax 稳定性与 axis 语义
(let* ((x (vt-from-sequence '(1000.0d0 1001.0d0)))
       (s (vt-softmax x)))
  (check-flat-approx "softmax [1000,1001] 稳定" '(0.2689414213699951d0 0.7310585786300049d0)
                     (vt-to-list s) 1d-12))
(let* ((x (vt-from-sequence '((1 2) (3 4))))
       (s (vt-softmax x :axis 0)))
  (check-flat-approx "softmax axis=0 列归一"
                     (list (/ 1.0d0 (+ 1.0d0 (exp 2.0d0))) (/ 1.0d0 (+ 1.0d0 (exp 2.0d0)))
                           (/ (exp 2.0d0) (+ 1.0d0 (exp 2.0d0))) (/ (exp 2.0d0) (+ 1.0d0 (exp 2.0d0))))
                     (flat (vt-to-list s)) 1d-12))
;; layer-norm: 有偏方差 + eps 在 sqrt 内（torch 一致）
(let ((n (vt-layer-norm (vt-from-sequence '((5.0d0 5.0d0 5.0d0) (1.0d0 2.0d0 3.0d0))) '(3) :eps 1d-12)))
  (check-flat-approx "layer-norm 常数行→0 / 有偏方差"
                     '((0 0 0) (-1.2247448713906706d0 0.0d0 1.2247448713906706d0))
                     (vt-to-list n) 1d-9))
;; argmax: NaN 位置优先（numpy 语义）+ 平局取首
(let ((a (vt-argmax (vt-from-sequence (list 1.0d0 (vt-get-nan :float64) 2.0d0)))))
  (check "argmax [1,NaN,2] → NaN 位置 1" 1 (vt-to-list a)))
(check "argmax 平局取首" 0 (vt-to-list (vt-argmax (vt-from-sequence '(3 3 1)))))
;; sigmoid 大值稳定
(check-flat-approx "sigmoid(±1000)" '(1.0d0 0.0d0)
                   (vt-to-list (vt-sigmoid (vt-from-sequence '(1000.0d0 -1000.0d0)))) 1d-12)
;; gelu tanh 近似值（文档语义）
(check-flat-approx "gelu(1) tanh 近似" '(0.8411919906082768d0)
                   (vt-to-list (vt-gelu (vt-from-sequence '(1.0d0)))) 1d-12)
;; swish/mish 就地 :out
(let ((x (vt-from-sequence '(-1.0d0 0.0d0 2.0d0))))
  (vt-swish x :out x)
  (check-flat-approx "swish :out 就地"
                     (list (* -1.0d0 (/ 1.0d0 (+ 1.0d0 (exp 1.0d0))))
                           0.0d0
                           (* 2.0d0 (/ 1.0d0 (+ 1.0d0 (exp -2.0d0)))))
                     (vt-to-list x) 1d-9))
(let ((x (vt-from-sequence '(-1.0d0 0.0d0 2.0d0))))
  (vt-mish x :out x)
  (check "mish :out 就地 dtype 保持" :float64 (vt-dtype x)))
;; cumsum 非连续视图 + 就地
(let* ((x (vt-from-sequence '((1 2) (3 4))))
       (xt (vt-transpose x)))
  (check-flat-approx "cumsum 转置视图 axis=0" '((1.0d0 3.0d0) (3.0d0 7.0d0)) (vt-to-list (vt-cumsum xt :axis 0)) 1d-12)
  (let ((y (vt-from-sequence '(1.0d0 2.0d0 3.0d0))))
    (vt-cumsum y :out y)
    (check-flat-approx "cumsum :out 就地" '(1 3 6) (vt-to-list y) 1d-12)))
;; one-hot / take / var / concat 提升
(check-flat-approx "one-hot [1,0,2] k=3" '((0 1 0) (1 0 0) (0 0 1))
                   (vt-to-list (vt-one-hot (vt-from-sequence '(1 0 2) :dtype :int64) 3)) 1d-12)
(check-flat-approx "take 负索引" '(3.0d0 2.0d0)
                   (vt-to-list (vt-take (vt-from-sequence '(1 2 3))
                                        (vt-from-sequence '(-1 -2) :dtype :int64))) 1d-12)
(check-flat-approx "var ddof=0 → 1.25" '(1.25d0)
                   (flat (vt-to-list (vt-var (vt-from-sequence '(1 2 3 4))))) 1d-12)
(check "concat int32+float32 → :float64" :float64
       (vt-dtype (vt-concatenate 0 (vt-from-sequence '(1 2) :dtype :int32)
                                 (vt-from-sequence '(1.0 2.0) :dtype :float32))))
;; cross-entropy 基本值与 NaN 传播
(check-flat-approx "cross-entropy y=[1,0],p=[0.9,0.1] → -log(0.9)"
                   '(0.10536051565782628d0)
                   (flat (vt-to-list (vt-cross-entropy (vt-from-sequence '(1.0d0 0.0d0))
                                                       (vt-from-sequence '(0.9d0 0.1d0))))) 1d-9)
(let* ((nan (vt-get-nan :float64))
       (l (vt-cross-entropy (vt-from-sequence '(1.0d0 0.0d0))
                            (vt-from-sequence (list nan 0.1d0)))))
  (check "cross-entropy p 含 NaN → NaN" t (nan-p (vt-to-list l))))
;; mse 广播
(check-flat-approx "mse (2,2) → 0.25" '(0.25d0)
                   (flat (vt-to-list (vt-mean-squared-error
                                      (vt-from-sequence '((0 0) (1 1)))
                                      (vt-from-sequence '((0 0) (0 1)))))) 1d-12)
;; log-softmax ≡ log(softmax)
(let* ((x (vt-from-sequence '(-1.0d0 2.0d0 0.5d0))))
  (check-flat-approx "log-softmax ≡ log(softmax)"
                     (vt-to-list (vt-log (vt-softmax x)))
                     (vt-to-list (vt-log-softmax x)) 1d-12))

;; 供 run-tests.sh 解析的机器可读汇总（格式 1）
(format t "Total: ~a | Pass: ~a | Fail: ~a~%"
        *checks* (- *checks* *failures*) *failures*)
(format t "~%===== 结果: ~a 检查 / ~a 失败 =====~%" *checks* *failures*)
(if (zerop *failures*)
    (progn (format t "ALL PASS~%") (sb-ext:exit :code 0))
    (progn (format t "FAILURES~%") (sb-ext:exit :code 1)))
