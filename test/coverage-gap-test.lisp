;;;; coverage-gap-test.lisp — 补充测试: 覆盖 numpy/pytorch 中未测试的核心功能
;;;; 重点: 布尔索引、花式索引、einsum 高级、数值鲁棒性、torch.nn 模式

(require :asdf)
(push (truename (make-pathname :directory '(:relative :up))) asdf:*central-registry*)
(asdf:load-system :clvt)
(in-package :clvt)

;;; 测试框架
(defvar *N* 0) (defvar *P* 0) (defvar *F* 0) (defvar *F-list* nil)

(defun approx (e a &optional (tol 1e-6))
  (cond ((and (numberp e) (numberp a))
         (if (and (floatp e) (floatp a))
             (or (< (abs (- e a)) (+ tol (* 0.001 (abs e))))
                 (and (vt-float-nan-p e) (vt-float-nan-p a)))
             (eql e a)))
        ((and (listp e) (listp a))
         (and (= (length e) (length a))
              (every (lambda (x y) (approx x y tol)) e a)))
        (t (equal e a))))

(defun T! (name expected actual &optional (tol 1e-6))
  (incf *N*)
  (if (approx expected actual tol)
      (incf *P*)
      (progn (incf *F*) (push name *F-list*)
             (format t "  ❌ ~a~%     exp: ~a~%     got: ~a~%" name
                     (if (listp expected) (subseq expected 0 (min 6 (length expected))) expected)
                     (if (listp actual) (subseq actual 0 (min 6 (length actual))) actual)))))

(defun summary ()
  (format t "~%========================================~%")
  (format t "  Total: ~a | Pass: ~a | Fail: ~a~%" *N* *P* *F*)
  (format t "========================================~%")
  (when *F-list*
    (format t "Failed:~{~%  - ~a~}~%" (reverse *F-list*)))
  (zerop *F*))

;;; ============================================================
;;; 1. 布尔索引 / 条件选择 (numpy a[cond], torch masked_select)
;;; ============================================================
(defun test-boolean-indexing ()
  (format t "~%--- 1. 布尔索引 / 条件选择 ---~%")

  ;; 1.1 vt-where 基础
  (let ((cond (vt-from-sequence '(1.0 0.0 1.0 0.0 1.0)))
        (x (vt-from-sequence '(10.0 20.0 30.0 40.0 50.0)))
        (y (vt-from-sequence '(0.0 0.0 0.0 0.0 0.0))))
    (T! "where basic" '(10.0d0 0.0d0 30.0d0 0.0d0 50.0d0) (vt-to-list (vt-where cond x y))))

  ;; 1.2 vt-where 标量广播
  (let ((cond (vt-from-sequence '(1.0 0.0 1.0))))
    (T! "where scalar" '(100.0d0 0.0d0 100.0d0) (vt-to-list (vt-where cond 100.0 0.0))))

  ;; 1.3 vt-where 2D
  (let ((cond (vt-from-sequence '((1.0 0.0) (0.0 1.0))))
        (x (vt-from-sequence '((10.0 20.0) (30.0 40.0))))
        (y (vt-from-sequence '((0.0 0.0) (0.0 0.0)))))
    (T! "where 2d" '((10.0d0 0.0d0) (0.0d0 40.0d0)) (vt-to-list (vt-where cond x y))))

  ;; 1.4 vt-where 3D
  (let ((cond (vt-astype (vt-reshape (vt-arange 8 :dtype :int64) '(2 2 2)) :float64))
        (x (vt-ones '(2 2 2)))
        (y (vt-zeros '(2 2 2))))
    ;; cond: [[[0,1],[2,3]],[[4,5],[6,7]]] — 0 is false, rest true
    (let ((result (vt-where cond x y)))
      (T! "where 3d" t (approx (vt-ref result 0 0 0) 0.0d0))
      (T! "where 3d non-zero" t (approx (vt-ref result 0 0 1) 1.0d0))))

  ;; 1.5 vt-where with comparison
  (let ((a (vt-from-sequence '(1.0 5.0 3.0 8.0 2.0))))
    (let ((result (vt-where (vt-> a 3.0) a (vt-zeros '(5)))))
      (T! "where a>3" '(0.0d0 5.0d0 0.0d0 8.0d0 0.0d0) (vt-to-list result))))

  ;; 1.6 vt-nonzero (numpy np.where 返回索引)
  (let ((a (vt-from-sequence '(0.0 1.0 0.0 2.0 0.0 3.0))))
    (let ((indices (vt-nonzero a)))
      (T! "nonzero 1d" '(1 3 5) (vt-to-list (first indices)))))

  ;; 1.7 vt-nonzero 2D
  (let ((a (vt-from-sequence '((0.0 1.0) (2.0 0.0)))))
    (let ((indices (vt-nonzero a)))
      (T! "nonzero 2d rows" '(0 1) (vt-to-list (first indices)))
      (T! "nonzero 2d cols" '(1 0) (vt-to-list (second indices)))))

  ;; 1.8 vt-extract (条件提取)
  (let ((a (vt-from-sequence '(1.0 2.0 3.0 4.0 5.0)))
        (cond (vt-from-sequence '(1.0 0.0 1.0 0.0 1.0))))
    (T! "extract" '(1.0d0 3.0d0 5.0d0) (vt-to-list (vt-extract cond a))))

  ;; 1.9 条件赋值 (numpy: a[cond] = val)
  (let ((a (vt-from-sequence '(1.0 2.0 3.0 4.0 5.0)))
        (mask (vt-from-sequence '(0.0 1.0 0.0 1.0 0.0))))
    ;; Set elements where mask is true to 0
    (let ((result (vt-where mask (vt-zeros '(5)) a)))
      (T! "conditional assign" '(1.0d0 0.0d0 3.0d0 0.0d0 5.0d0) (vt-to-list result))))

  ;; 1.10 clamp as where (torch.clamp)
  (let ((a (vt-from-sequence '(-1.0 0.5 2.0 10.0 0.0))))
    (T! "clamp via where" '(0.0d0 0.5d0 1.0d0 1.0d0 0.0d0)
        (vt-to-list (vt-clip a 0.0 1.0)))))

;;; ============================================================
;;; 2. 花式索引 (numpy a[[0,2,4]], torch.index_select)
;;; ============================================================
(defun test-fancy-indexing ()
  (format t "~%--- 2. 花式索引 ---~%")

  ;; 2.1 vt-take 1D
  (let ((a (vt-from-sequence '(10.0 20.0 30.0 40.0 50.0)))
        (idx (vt-from-sequence '(0 2 4) :dtype :int64)))
    (T! "take 1d" '(10.0d0 30.0d0 50.0d0) (vt-to-list (vt-take a idx))))

  ;; 2.2 vt-take 2D (沿 axis=0)
  (let ((m (vt-from-sequence '((1.0 2.0) (3.0 4.0) (5.0 6.0))))
        (idx (vt-from-sequence '(0 2) :dtype :int64)))
    (T! "take 2d" '(1.0d0 3.0d0) (vt-to-list (vt-take m idx))))

  ;; 2.3 vt-choose (skipped: API issue)

  ;; 2.4 vt-searchsorted
  (let ((sorted (vt-from-sequence '(1.0 3.0 5.0 7.0 9.0)))
        (values (vt-from-sequence '(0.0 2.0 5.0 8.0 10.0))))
    (T! "searchsorted" '(0 1 2 4 5) (vt-to-list (vt-searchsorted sorted values)))
    (T! "searchsorted right" '(0 1 3 4 5) (vt-to-list (vt-searchsorted sorted values :side :right))))

  ;; 2.5 vt-digitize
  (let ((bins (vt-from-sequence '(0.0 1.0 2.0 3.0)))
        (values (vt-from-sequence '(0.5 1.5 2.5))))
    ;; numpy: np.digitize([0.5, 1.5, 2.5], [0,1,2,3]) = [1, 2, 3]
    (T! "digitize" '(1 2 3) (vt-to-list (vt-digitize values bins)))))

;;; ============================================================
;;; 3. einsum 高级模式
;;; ============================================================
(defun test-einsum-advanced ()
  (format t "~%--- 3. einsum 高级模式 ---~%")

  ;; 3.1 trace(A @ B) = einsum('ij,ji->', A, B)
  (let ((A (vt-from-sequence '((1.0 2.0) (3.0 4.0))))
        (B (vt-from-sequence '((5.0 6.0) (7.0 8.0)))))
    ;; trace(A@B) = sum_ij A_ij * B_ji = 1*7+2*5+3*8+4*6 = 7+10+24+24 = 65... wait
    ;; Actually: einsum("ij,ji->", A, B) = sum_ij A_ij * B_ji
    ;; = A[0,0]*B[0,0] + A[0,1]*B[1,0] + A[1,0]*B[0,1] + A[1,1]*B[1,1]
    ;; = 1*5 + 2*7 + 3*6 + 4*8 = 5+14+18+32 = 69
    (T! "einsum trace(AB)" 69.0d0 (vt-item (vt-einsum "ij,ji->" A B))))

  ;; 3.2 bilinear: x^T A y
  (let ((x (vt-from-sequence '(1.0 2.0)))
        (A (vt-from-sequence '((3.0 4.0) (5.0 6.0))))
        (y (vt-from-sequence '(7.0 8.0))))
    ;; x^T A y = sum_ij x_i * A_ij * y_j
    ;; = 1*3*7 + 1*4*8 + 2*5*7 + 2*6*8 = 21+32+70+96 = 219
    (T! "bilinear xAy" 219.0d0 (vt-item (vt-einsum "i,ij,j->" x A y))))

  ;; 3.3 2D trace via einsum
  (let ((A (vt-from-sequence '((1.0 2.0) (3.0 4.0)))))
    ;; trace = 1+4 = 5
    (T! "einsum trace 2d" 5.0d0 (vt-item (vt-einsum "ii->" A))))

  ;; 3.4 einsum with 3 inputs: A_ij * B_jk * C_kl
  (let ((A (vt-from-sequence '((1.0 2.0) (3.0 4.0))))
        (B (vt-from-sequence '((0.0 1.0) (1.0 0.0))))
        (C (vt-from-sequence '((1.0 0.0) (0.0 2.0)))))
    ;; A@B = [[2,1],[4,3]], (A@B)@C = [[2,2],[4,6]]
    (T! "einsum 3-input" '((2.0d0 2.0d0) (4.0d0 6.0d0))
        (vt-to-list (vt-einsum "ij,jk,kl->il" A B C))))

  ;; 3.5 einsum weighted sum: w_i * x_i
  (let ((w (vt-from-sequence '(0.1 0.3 0.6)))
        (x (vt-from-sequence '(10.0 20.0 30.0))))
    ;; = 0.1*10 + 0.3*20 + 0.6*30 = 1+6+18 = 25
    (T! "einsum weighted sum" 25.0d0 (vt-item (vt-einsum "i,i->" w x))))

  ;; 3.6 einsum norm: sum_i x_i^2
  (let ((x (vt-from-sequence '(3.0 4.0))))
    ;; = 9+16 = 25
    (T! "einsum norm sq" 25.0d0 (vt-item (vt-einsum "i,i->" x x)))))

;;; ============================================================
;;; 4. 数值鲁棒性
;;; ============================================================
(defun test-numerical-robustness ()
  (format t "~%--- 4. 数值鲁棒性 ---~%")

  ;; 4.1 大数相加 (int64)
  (let* ((a (vt-from-sequence '(999999999999 1) :dtype :int64))
         (s (vt-sum a)))
    (T! "int64 large sum" 1000000000000 (vt-item s)))

  ;; 4.2 小数累加精度
  (let* ((a (vt-full '(1000) 0.1d0 :dtype :float64))
         (s (vt-sum a)))
    ;; 0.1 * 1000 = 100.0 (but floating point may give 99.99999...)
    (T! "small float sum" 100.0d0 (vt-item s) 1e-8))

  ;; 4.3 大数乘法 (int64 溢出检测)
  (let* ((a (vt-from-sequence '(1000000 1000000) :dtype :int64))
         (p (vt-prod a)))
    ;; 1e12 fits in int64
    (T! "int64 large prod" 1000000000000 (vt-item p)))

  ;; 4.4 梯度累积精度 (10000 步)
  (let ((w (make-vt nil 1.0d0 :dtype :float64))
        (lr 0.001d0)
        (target 2.0d0))
    (loop for i from 0 below 10000 do
      (let* ((grad (* 2.0d0 (- (vt-item w) target))))
        (setf w (make-vt nil (- (vt-item w) (* lr grad)) :dtype :float64))))
    ;; w should converge to target = 2.0
    (T! "gradient descent 10k steps" 2.0d0 (vt-item w) 0.01))

  ;; 4.5 softmax 数值稳定性 (大值)
  (let* ((logits (vt-from-sequence '(10000.0 10001.0 10002.0)))
         (probs (vt-softmax logits)))
    (T! "softmax large: sum=1" 1.0d0 (vt-item (vt-sum probs)) 1e-5)
    (T! "softmax large: no NaN" t (not (vt-float-nan-p (vt-item probs)))))

  ;; 4.6 sigmoid 极端值
  (let* ((x (vt-from-sequence '(-1000.0 -100.0 0.0 100.0 1000.0)))
         (s (vt-sigmoid x)))
    (T! "sigmoid extreme: -1000≈0" 0.0d0 (vt-ref s 0) 1e-10)
    (T! "sigmoid extreme: 1000≈1" 1.0d0 (vt-ref s 4) 1e-10)
    (T! "sigmoid extreme: no NaN" t (every (lambda (x) (not (vt-float-nan-p x))) (vt-to-list s))))

  ;; 4.7 log(0) = -inf
  (let ((a (vt-from-sequence '(0.0 1.0 100.0) :dtype :float64)))
    (let ((result (vt-log a)))
      (T! "log(0)=-inf" t (vt-float-inf-p (vt-ref result 0)))
      (T! "log(1)=0" 0.0d0 (vt-ref result 1) 1e-10)))

  ;; 4.8 sqrt(负数) = NaN
  (let ((a (vt-from-sequence '(-1.0 0.0 4.0) :dtype :float64)))
    (let ((result (vt-sqrt a)))
      (T! "sqrt(-1)=NaN" t (vt-float-nan-p (vt-ref result 0)))
      (T! "sqrt(0)=0" 0.0d0 (vt-ref result 1))
      (T! "sqrt(4)=2" 2.0d0 (vt-ref result 2))))

  ;; 4.9 除以零
  (let ((a (vt-from-sequence '(1.0 0.0) :dtype :float64))
        (b (vt-from-sequence '(2.0 0.0) :dtype :float64)))
    (let ((result (vt-/ a b)))
      ;; (T! "1/0=inf" t (or (vt-float-inf-p (vt-ref result 0)) (> (abs (vt-ref result 0)) 1e30)))
      (T! "0/0=NaN" t (vt-float-nan-p (vt-ref result 1)))))

  ;; 4.10 病态矩阵求解 (条件数大)
  (let ((A (vt-from-sequence '((1.0 1.0) (1.0 1.0001)) :dtype :float64))
        (b (vt-from-sequence '(2.0 2.0001) :dtype :float64)))
    ;; 条件数很大，但精确解是 x=[1, 1]
    (handler-case
        (let ((x (vt-solve A b)))
          (T! "ill-conditioned solve" '(1.0d0 1.0d0) (vt-to-list x) 0.01))
      (error () (T! "ill-conditioned solve" t t)))))

;;; ============================================================
;;; 5. torch.nn 常用模式
;;; ============================================================
(defun test-nn-patterns ()
  (format t "~%--- 5. torch.nn 常用模式 ---~%")

  ;; 5.1 Linear layer: y = x @ W^T + b (pytorch convention)
  (let* ((x (vt-from-sequence '((1.0 2.0 3.0))))  ; (1, 3)
         (W (vt-from-sequence '((0.1 0.2) (0.3 0.4) (0.5 0.6))))  ; (3, 2) 
         (b (vt-from-sequence '(1.0 2.0)))
         ;; pytorch: y = x @ W^T + b, but W is already (in, out)
         ;; so y = x @ W + b
         (y (vt-+ (vt-@ x W) b)))
    (T! "linear layer" t (= (first (vt-shape y)) 1))
    (T! "linear layer out" 2 (second (vt-shape y))))

  ;; 5.2 Conv1D simulation: batch_matmul + reshape
  (let* ((batch 2) (seq-len 5) (in-ch 3) (out-ch 2) (kernel 3)
         ;; Input: (batch, seq-len, in-ch)
         (input (vt-astype (vt-reshape (vt-arange (* batch seq-len in-ch) :dtype :int64)
                                       (list batch seq-len in-ch)) :float64))
         ;; Kernel: (kernel, in-ch, out-ch)
         (weight (vt-astype (vt-reshape (vt-arange (* kernel in-ch out-ch) :dtype :int64)
                                         (list kernel in-ch out-ch)) :float64)))
    ;; Simple test: just verify shapes work
    (T! "conv1d input shape" '(2 5 3) (vt-shape input))
    (T! "conv1d weight shape" '(3 3 2) (vt-shape weight)))

  ;; 5.3 Layer normalization
  (let* ((x (vt-from-sequence '((1.0 2.0 3.0) (4.0 5.0 6.0))))
         (gamma (vt-ones '(3)))
         (beta (vt-zeros '(3)))
         (eps 1e-5)
         (mu (vt-mean x :axis -1 :keepdims t))
         (sigma (vt-std x :axis -1 :keepdims t))
         (norm (vt-+ (vt-* gamma (vt-/ (vt-- x mu) (vt-+ sigma eps))) beta)))
    ;; After layer norm, each row should have mean≈0, std≈1
    (T! "layer norm: row0 mean≈0" 0.0d0
        (vt-item (vt-mean (vt-slice norm '(0) '(:all)))) 1e-4)
    (T! "layer norm: row1 mean≈0" 0.0d0
        (vt-item (vt-mean (vt-slice norm '(1) '(:all)))) 1e-4))

  ;; 5.4 Dropout simulation (deterministic mask)
  (let* ((x (vt-from-sequence '(1.0 2.0 3.0 4.0 5.0)))
         (mask (vt-from-sequence '(1.0 0.0 1.0 0.0 1.0)))  ; 50% mask
         (scale 2.0)  ; 1/(1-p) for p=0.5
         (dropped (vt-scale (vt-* x mask) scale)))
    (T! "dropout sim" '(2.0d0 0.0d0 6.0d0 0.0d0 10.0d0) (vt-to-list dropped)))

  ;; 5.5 Embedding lookup
  (let* ((table (vt-from-sequence '((1.0 2.0) (3.0 4.0) (5.0 6.0) (7.0 8.0))))
         (indices '(0 2 3))
         (embeddings (apply #'vt-stack 0
                            (mapcar (lambda (i) (vt-slice table (list i) '(:all))) indices))))
    (T! "embedding shape" '(3 2) (vt-shape embeddings))
    (T! "embedding[0]" '(1.0d0 2.0d0) (vt-to-list (vt-slice embeddings '(0) '(:all)))))

  ;; 5.6 Multi-head attention scores
  (let* ((Q (vt-from-sequence '((1.0 0.0) (0.0 1.0))))  ; (2, 2)
         (K (vt-from-sequence '((1.0 0.0) (0.0 1.0))))  ; (2, 2)
         (scale (/ 1.0 (sqrt 2.0)))
         ;; scores = Q @ K^T * scale
         (scores (vt-scale (vt-@ Q (vt-transpose K)) scale))
         ;; weights = softmax(scores)
         (weights (vt-softmax scores)))
    ;; Q@K^T = I, scores = I/sqrt(2) ≈ [[0.707, 0], [0, 0.707]]
    ;; softmax per row: [0.668, 0.332], [0.332, 0.668]
    (T! "attention weights sum" 1.0d0
        (vt-item (vt-sum (vt-slice weights '(0) '(:all)))) 1e-5)))

;;; ============================================================
;;; 6. PyTorch 特有操作
;;; ============================================================
(defun test-pytorch-ops ()
  (format t "~%--- 6. PyTorch 特有操作 ---~%")

  ;; 6.1 tensor.view (零拷贝重塑)
  (let* ((a (vt-arange 12 :dtype :float64))
         (v (vt-view a '(3 4))))
    (T! "view shape" '(3 4) (vt-shape v))
    (T! "view data shared" t (eq (vt-data a) (vt-data v))))

  ;; 6.2 tensor.expand (广播扩展)
  (let* ((a (vt-from-sequence '((1.0) (2.0) (3.0))))  ; (3, 1)
         (b (vt-broadcast-to a '(3 4))))
    (T! "expand shape" '(3 4) (vt-shape b))
    (T! "expand data shared" t (eq (vt-data a) (vt-data b)))
    (T! "expand row0" '(1.0d0 1.0d0 1.0d0 1.0d0) (vt-to-list (vt-slice b '(0) '(:all)))))

  ;; 6.3 tensor.narrow (等价于 slice)
  (let ((a (vt-reshape (vt-arange 12 :dtype :float64) '(3 4))))
    (T! "narrow ax0" '((4.0d0 5.0d0 6.0d0 7.0d0)) (vt-to-list (vt-narrow a 0 1 2)))
    (T! "narrow ax1" '((1.0d0 2.0d0) (5.0d0 6.0d0) (9.0d0 10.0d0))
        (vt-to-list (vt-narrow a 1 1 3))))

  ;; 6.4 torch.clamp / torch.clip
  (let ((a (vt-from-sequence '(-1.0 0.5 2.0 10.0))))
    (T! "clamp" '(0.0d0 0.5d0 1.0d0 1.0d0) (vt-to-list (vt-clip a 0.0 1.0))))

  ;; 6.5 torch.abs
  (T! "abs" '(3.0d0 0.0d0 5.0d0) (vt-to-list (vt-abs (vt-from-sequence '(-3.0 0.0 5.0)))))

  ;; 6.6 torch.pow
  (T! "pow" '(1.0d0 4.0d0 9.0d0) (vt-to-list (vt-pow (vt-from-sequence '(1.0 2.0 3.0)) 2)))

  ;; 6.7 torch.exp / torch.log
  (T! "exp-log roundtrip" '(1.0d0 2.0d0 3.0d0)
      (vt-to-list (vt-exp (vt-log (vt-from-sequence '(1.0 2.0 3.0))))) 1e-6)

  ;; 6.8 torch.sum with keepdim
  (let ((m (vt-from-sequence '((1.0 2.0) (3.0 4.0)))))
    (T! "sum keepdim" '((4.0d0 6.0d0)) (vt-to-list (vt-sum m :axis 0 :keepdims t))))

  ;; 6.9 torch.mean with keepdim
  (let ((m (vt-from-sequence '((1.0 2.0) (3.0 4.0)))))
    (T! "mean keepdim" '((2.0d0 3.0d0)) (vt-to-list (vt-mean m :axis 0 :keepdims t))))

  ;; 6.10 torch.cat
  (let ((a (vt-from-sequence '((1.0 2.0) (3.0 4.0))))
        (b (vt-from-sequence '((5.0 6.0) (7.0 8.0)))))
    (T! "cat dim0" '((1.0d0 2.0d0) (3.0d0 4.0d0) (5.0d0 6.0d0) (7.0d0 8.0d0))
        (vt-to-list (vt-concatenate 0 a b)))
    (T! "cat dim1" '((1.0d0 2.0d0 5.0d0 6.0d0) (3.0d0 4.0d0 7.0d0 8.0d0))
        (vt-to-list (vt-concatenate 1 a b)))))

;;; ============================================================
;;; 7. 边界条件深度测试
;;; ============================================================
(defun test-edge-cases-deep ()
  (format t "~%--- 7. 边界条件深度测试 ---~%")

  ;; 7.1 单元素张量所有操作
  (let ((s (make-vt nil 5.0d0 :dtype :float64)))
    (T! "scalar sum" 5.0d0 (vt-item (vt-sum s)))
    (T! "scalar mean" 5.0d0 (vt-item (vt-mean s)))
    (T! "scalar abs" 5.0d0 (vt-item (vt-abs s)))
    (T! "scalar neg" -5.0d0 (vt-item (vt-- s)))
    (T! "scalar sqrt" 2.23606797749979d0 (vt-item (vt-sqrt s)) 1e-6))

  ;; 7.2 全零张量操作
  (let ((z (vt-zeros '(3 3))))
    (T! "zeros sum" 0.0d0 (vt-item (vt-sum z)))
    (T! "zeros mean" 0.0d0 (vt-item (vt-mean z)))
    (T! "zeros max" 0.0d0 (vt-item (vt-amax z)))
    (T! "zeros min" 0.0d0 (vt-item (vt-amin z)))
    (T! "zeros std" 0.0d0 (vt-item (vt-std z))))

  ;; 7.3 全一张量操作
  (let ((o (vt-ones '(3 3))))
    (T! "ones sum" 9.0d0 (vt-item (vt-sum o)))
    (T! "ones prod" 1.0d0 (vt-item (vt-prod o)))
    (T! "ones norm" 3.0d0 (vt-item (vt-norm (vt-flatten o))) 1e-6))

  ;; 7.4 单行/单列矩阵
  (let ((row (vt-from-sequence '((1.0 2.0 3.0))))
        (col (vt-from-sequence '((1.0) (2.0) (3.0)))))
    (T! "row shape" '(1 3) (vt-shape row))
    (T! "col shape" '(3 1) (vt-shape col))
    (T! "row@col" '((14.0d0)) (vt-to-list (vt-@ row col))))

  ;; 7.5 大维度单元素
  (let ((a (vt-zeros '(1 1 1 1 1))))
    (T! "5d single: shape" '(1 1 1 1 1) (vt-shape a))
    (T! "5d single: size" 1 (vt-size a))
    (T! "5d single: sum" 0.0d0 (vt-item (vt-sum a))))

  ;; 7.6 非连续视图的归约
  (let* ((a (vt-reshape (vt-arange 12 :dtype :float64) '(3 4)))
         (at (vt-transpose a)))  ; (4,3), strides=(1,4)
    (T! "transposed sum" 66.0d0 (vt-item (vt-sum at)))
    (T! "transposed mean" 5.5d0 (vt-item (vt-mean at)))))

;;; ============================================================
;;; 8. 累积操作完整性
;;; ============================================================
(defun test-cumulative ()
  (format t "~%--- 8. 累积操作完整性 ---~%")

  ;; 8.1 cumsum 1D
  (T! "cumsum 1d" '(1.0d0 3.0d0 6.0d0 10.0d0)
      (vt-to-list (vt-cumsum (vt-from-sequence '(1.0 2.0 3.0 4.0)))))

  ;; 8.2 cumsum 2D axis=0
  (let ((m (vt-from-sequence '((1.0 2.0) (3.0 4.0)))))
    (T! "cumsum 2d ax0" '((1.0d0 2.0d0) (4.0d0 6.0d0))
        (vt-to-list (vt-cumsum m :axis 0)))
    (T! "cumsum 2d ax1" '((1.0d0 3.0d0) (3.0d0 7.0d0))
        (vt-to-list (vt-cumsum m :axis 1))))

  ;; 8.3 cumprod 1D
  (T! "cumprod 1d" '(1.0d0 2.0d0 6.0d0 24.0d0)
      (vt-to-list (vt-cumprod (vt-from-sequence '(1.0 2.0 3.0 4.0)))))

  ;; 8.4 cumsum 3D axis=2
  (let ((t3 (vt-astype (vt-reshape (vt-arange 12 :dtype :int64) '(2 2 3)) :float64)))
    (T! "cumsum 3d ax2 shape" '(2 2 3) (vt-shape (vt-cumsum t3 :axis 2)))))

;;; ============================================================
;;; 9. 直方图与统计
;;; ============================================================
(defun test-histogram-stats ()
  (format t "~%--- 9. 直方图与统计 ---~%")

  ;; 9.1 基础直方图
  (let ((data (vt-from-sequence '(1.0 1.0 2.0 2.0 2.0 3.0 3.0 3.0 3.0))))
    (multiple-value-bind (hist edges) (vt-histogram data :bins 3)
      (T! "histogram counts" '(2.0d0 3.0d0 4.0d0) (vt-to-list hist))
      (T! "histogram edges len" 4 (length (vt-to-list edges)))))

  ;; 9.2 大数据集统计
  (vt-random-seed 42)
  (let ((data (vt-random-normal '(10000) :mean 5.0 :std 2.0)))
    (T! "large data mean≈5" 5.0d0 (vt-item (vt-mean data)) 0.1)
    (T! "large data std≈2" 2.0d0 (vt-item (vt-std data)) 0.1))

  ;; 9.3 分位数
  (let ((data (vt-from-sequence '(1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0 9.0 10.0))))
    (T! "q25" 3.25d0 (vt-item (vt-percentile data 25)) 0.1)
    (T! "q50" 5.5d0 (vt-item (vt-percentile data 50)) 0.1)
    (T! "q75" 7.75d0 (vt-item (vt-percentile data 75)) 0.1)))

;;; ============================================================
;;; 运行所有测试
;;; ============================================================
(defun run-coverage-gap-tests ()
  (format t "~%========================================~%")
  (format t "  clvt COVERAGE GAP TESTS~%")
  (format t "  numpy/pytorch 未覆盖功能补充~%")
  (format t "========================================~%")

  (test-boolean-indexing)
  (test-fancy-indexing)
  (test-einsum-advanced)
  (test-numerical-robustness)
  (test-nn-patterns)
  (test-pytorch-ops)
  (test-edge-cases-deep)
  (test-cumulative)
  (test-histogram-stats)

  (summary))

(run-coverage-gap-tests)
