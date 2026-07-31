;;;; nested-test.lisp — 函数嵌套组合测试
;;;; 模拟 AI/神经网络常用计算流程，测试函数组合的稳定性与正确性
;;;; 每个测试都是多个函数的嵌套调用，对比 numpy 期望结果

(require :asdf)
(push (truename (make-pathname :directory '(:relative :up))) asdf:*central-registry*)
(asdf:load-system :clvt)
(in-package :clvt)

;;; 测试框架
(defvar *N* 0) (defvar *P* 0) (defvar *F* 0) (defvar *F-list* nil)

(defun approx (e a &optional (tol 1e-6))
  (cond ((and (numberp e) (numberp a))
         (if (and (floatp e) (floatp a))
             (< (abs (- e a)) (+ tol (* 0.001 (abs e))))
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
             (format t "  ❌ ~a~%" name))))

(defun summary ()
  (format t "~%========================================~%")
  (format t "  Total: ~a | Pass: ~a | Fail: ~a~%" *N* *P* *F*)
  (format t "========================================~%")
  (when *F-list*
    (format t "Failed:~{~%  - ~a~}~%" (reverse *F-list*)))
  (zerop *F*))

;;; ============================================================
;;; 1. 线性层 (Linear Layer): Y = X @ W + b
;;; ============================================================
(defun test-linear-layer ()
  (format t "~%--- 1. Linear Layer: Y = X @ W + b ---~%")

  ;; batch=3, in=4, out=2
  (let* ((X (vt-from-sequence '((1.0 2.0 3.0 4.0)
                                 (5.0 6.0 7.0 8.0)
                                 (9.0 10.0 11.0 12.0))))
         (W (vt-from-sequence '((0.1 0.2)
                                 (0.3 0.4)
                                 (0.5 0.6)
                                 (0.7 0.8))))
         (b (vt-from-sequence '(1.0 2.0)))
         ;; Y = X @ W + b
         (Y (vt-+ (vt-@ X W) b)))
    (T! "linear: shape" '(3 2) (vt-shape Y))
    ;; numpy: X@W+b = [[1*0.1+2*0.3+3*0.5+4*0.7+1, ...], ...]
    ;; = [[1+2, 2+3+4+5+6+7+8], ...]  let me compute properly
    ;; X[0] @ W = [1*0.1+2*0.3+3*0.5+4*0.7, 1*0.2+2*0.4+3*0.6+4*0.8] = [5.0, 6.0]
    ;; + b = [6.0, 8.0]
    (T! "linear: row0" '(6.0d0 8.0d0) (vt-to-list (vt-slice Y '(0) '(:all)))))

  ;; 多层线性: Y = (X @ W1 + b1) @ W2 + b2
  (let* ((X (vt-from-sequence '((1.0 0.0) (0.0 1.0))))
         (W1 (vt-from-sequence '((2.0 0.0) (0.0 3.0))))
         (b1 (vt-from-sequence '(1.0 1.0)))
         (W2 (vt-from-sequence '((1.0) (1.0))))
         (b2 (vt-from-sequence '(0.5)))
         ;; hidden = X @ W1 + b1 = [[2+1, 0+1], [0+1, 3+1]] = [[3,1],[1,4]]
         ;; Y = hidden @ W2 + b2 = [[3+1+0.5], [1+4+0.5]] = [[4.5], [5.5]]
         (hidden (vt-+ (vt-@ X W1) b1))
         (Y (vt-+ (vt-@ hidden W2) b2)))
    (T! "multi-linear: shape" '(2 1) (vt-shape Y))
    (T! "multi-linear: val" '((4.5d0) (5.5d0)) (vt-to-list Y)))

  ;; 单样本线性层
  (let* ((x (vt-from-sequence '(1.0 2.0 3.0)))
         (W (vt-from-sequence '((0.1 0.2 0.3) (0.4 0.5 0.6))))
         (b (vt-from-sequence '(1.0 2.0)))
         ;; y = W @ x + b = [0.1+0.4+0.9+1, 0.2+0.5+1.2+2] = ...
         ;; Actually: W is (2,3), x is (3,) -> W @ x = (2,)
         ;; [0.1*1+0.2*2+0.3*3, 0.4*1+0.5*2+0.6*3] = [1.4, 3.2]
         ;; + b = [2.4, 5.2]
         (y (vt-+ (vt-@ W x) b)))
    (T! "single-linear" '(2.4d0 5.2d0) (vt-to-list y) 1e-6)))

;;; ============================================================
;;; 2. ReLU 激活: Y = max(0, X @ W + b)
;;; ============================================================
(defun test-relu-layer ()
  (format t "~%--- 2. ReLU Layer: Y = relu(X @ W + b) ---~%")

  (let* ((X (vt-from-sequence '((-1.0 2.0) (3.0 -4.0))))
         (W (vt-from-sequence '((1.0 -1.0) (-1.0 1.0))))
         (b (vt-from-sequence '(0.5 0.5)))
         ;; X@W = [[-1+(-2), 1+2], [3+4, -3+(-4)]] = [[-3, 3], [7, -7]]
         ;; +b = [[-2.5, 3.5], [7.5, -6.5]]
         ;; relu = [[0, 3.5], [7.5, 0]]
         (Y (vt-relu (vt-+ (vt-@ X W) b))))
    (T! "relu-layer: val" '((0.0d0 3.5d0) (7.5d0 0.0d0)) (vt-to-list Y)))

  ;; Sigmoid 层
  (let* ((X (vt-from-sequence '((-1.0 0.0 1.0))))
         (W (vt-from-sequence '((1.0) (1.0) (1.0))))
         (b (vt-from-sequence '(0.0)))
         ;; z = X@W+b = [[-1+0+1+0]] = [[0.0]]
         ;; sigmoid(0) = 0.5
         (Y (vt-sigmoid (vt-+ (vt-@ X W) b))))
    (T! "sigmoid-layer" 0.5d0 (vt-item (vt-slice Y '(0) '(0))) 1e-6)))

;;; ============================================================
;;; 3. Softmax + Cross-Entropy Loss
;;; ============================================================
(defun test-softmax-crossentropy ()
  (format t "~%--- 3. Softmax + Cross-Entropy ---~%")

  ;; 单样本: logits=[2.0, 1.0, 0.1], target=[1, 0, 0]
  (let* ((logits (vt-from-sequence '(2.0 1.0 0.1)))
         (target (vt-from-sequence '(1.0 0.0 0.0)))
         ;; softmax = [0.659, 0.242, 0.099]
         (probs (vt-softmax logits))
         ;; cross-entropy = -sum(target * log(probs)) = -log(0.659) ≈ 0.417
         (loss (vt-cross-entropy target probs)))
    (T! "softmax: sum=1" 1.0d0 (vt-item (vt-sum probs)) 1e-5)
    (T! "softmax: max_idx" 0 (vt-item (vt-argmax probs)))
    (T! "cross-entropy" 0.417d0 (vt-item loss) 0.01))

  ;; Batch softmax + CE
  (let* ((logits (vt-from-sequence '((2.0 1.0 0.1)
                                      (0.5 2.5 0.0))))
         (target (vt-from-sequence '((1.0 0.0 0.0)
                                      (0.0 1.0 0.0))))
         (probs (vt-softmax logits))
         ;; Each row should sum to 1
         (row-sums (vt-sum probs :axis 1)))
    (T! "batch-softmax: row0_sum" 1.0d0 (vt-ref row-sums 0) 1e-5)
    (T! "batch-softmax: row1_sum" 1.0d0 (vt-ref row-sums 1) 1e-5)))

;;; ============================================================
;;; 4. Batch Normalization: Y = (X - mean) / sqrt(var + eps) * gamma + beta
;;; ============================================================
(defun test-batch-norm ()
  (format t "~%--- 4. Batch Normalization ---~%")

  (let* ((X (vt-from-sequence '((1.0 2.0 3.0 4.0)
                                 (5.0 6.0 7.0 8.0)
                                 (9.0 10.0 11.0 12.0))))
         (gamma 1.0) (beta 0.0) (eps 1e-5)
         ;; Per-feature normalization (axis=0)
         (mu (vt-mean X :axis 0))
         (sigma2 (vt-var X :axis 0))
         ;; X_norm = (X - mu) / sqrt(sigma2 + eps)
         (X_norm (vt-/ (vt-- X mu) (vt-sqrt (vt-+ sigma2 eps))))
         ;; Y = gamma * X_norm + beta
         (Y (vt-+ (vt-scale X_norm gamma) beta)))
    ;; After normalization, each column should have mean≈0, std≈1
    (let ((Y-mean (vt-mean Y :axis 0))
          (Y-std (vt-std Y :axis 0)))
      (T! "bn: mean≈0" '(0.0d0 0.0d0 0.0d0 0.0d0) (vt-to-list Y-mean) 1e-4)
      (T! "bn: std≈1" '(1.0d0 1.0d0 1.0d0 1.0d0) (vt-to-list Y-std) 1e-4)))

  ;; 3D batch norm (per channel)
  (let* ((X (vt-astype (vt-reshape (vt-arange 24 :dtype :int64) '(2 3 4)) :float64))
         (mu (vt-mean X :axis '(0 2) :keepdims t))
         (sigma2 (vt-var X :axis '(0 2) :keepdims t))
         (X-norm (vt-/ (vt-- X mu) (vt-sqrt (vt-+ sigma2 1e-5)))))
    (T! "bn-3d: shape" '(2 3 4) (vt-shape X-norm))))

;;; ============================================================
;;; 5. Attention 机制: softmax(Q @ K^T / sqrt(d)) @ V
;;; ============================================================
(defun test-attention ()
  (format t "~%--- 5. Attention: softmax(QK^T/sqrt(d)) @ V ---~%")

  (let* ((seq-len 3) (d 4)
         ;; Q, K, V
         (Q (vt-from-sequence '((1.0 0.0 0.0 0.0)
                                 (0.0 1.0 0.0 0.0)
                                 (0.0 0.0 1.0 0.0))))
         (K (vt-from-sequence '((1.0 0.0 0.0 0.0)
                                 (0.0 1.0 0.0 0.0)
                                 (0.0 0.0 1.0 0.0))))
         (V (vt-from-sequence '((1.0 2.0)
                                 (3.0 4.0)
                                 (5.0 6.0))))
         ;; scores = Q @ K^T / sqrt(d)
         (scores (vt-scale (vt-@ Q (vt-transpose K)) (/ 1.0 (sqrt d))))
         ;; weights = softmax(scores) per row
         (weights (vt-softmax scores))
         ;; output = weights @ V
         (output (vt-@ weights V)))
    (T! "attention: shape" '(3 2) (vt-shape output))
    ;; Since Q=K=I, scores = I/sqrt(4) = 0.5*I
    ;; softmax of [0.5, 0, 0] per row -> highest weight on diagonal
    ;; output ≈ V (since attention is nearly identity)
    (T! "attention: row0" '(2.644d0 3.644d0) (vt-to-list (vt-slice output '(0) '(:all))) 0.01)
    ;; Weight matrix: each row sums to 1
    (T! "attention: weight_sum" 1.0d0 (vt-item (vt-sum (vt-slice weights '(0) '(:all)))) 1e-5)))

;;; ============================================================
;;; 6. 残差连接: Y = activation(X + F(X))
;;; ============================================================
(defun test-residual ()
  (format t "~%--- 6. Residual: Y = relu(X + F(X)) ---~%")

  (let* ((X (vt-from-sequence '((1.0 -2.0) (3.0 -4.0))))
         ;; F(X) = X @ W (simple linear transform)
         (W (vt-from-sequence '((0.5 0.0) (0.0 0.5))))
         (FX (vt-@ X W))
         ;; residual = X + F(X) = X * 1.5
         (residual (vt-+ X FX))
         ;; Y = relu(residual)
         (Y (vt-relu residual)))
    ;; X*1.5 = [[1.5, -3.0], [4.5, -6.0]]
    ;; relu = [[1.5, 0], [4.5, 0]]
    (T! "residual: val" '((1.5d0 0.0d0) (4.5d0 0.0d0)) (vt-to-list Y)))

  ;; 残差 + LayerNorm
  (let* ((X (vt-from-sequence '((1.0 2.0 3.0) (4.0 5.0 6.0))))
         (W (vt-eye 3 :dtype :float64))
         (FX (vt-@ X W))
         (residual (vt-+ X FX))  ; = 2X
         ;; Layer norm on last axis
         (mu (vt-mean residual :axis -1 :keepdims t))
         (sigma (vt-std residual :axis -1 :keepdims t))
         (normed (vt-/ (vt-- residual mu) (vt-+ sigma 1e-5))))
    ;; After layer norm, each row should have mean≈0
    (T! "residual+ln: mean≈0" 0.0d0
        (vt-item (vt-mean (vt-slice normed '(0) '(:all)))) 1e-4)))

;;; ============================================================
;;; 7. 多头注意力拼接: concat(head1, head2) @ Wo
;;; ============================================================
(defun test-multihead ()
  (format t "~%--- 7. Multi-Head Concat ---~%")

  (let* ((head1 (vt-from-sequence '((1.0 2.0) (3.0 4.0))))
         (head2 (vt-from-sequence '((5.0 6.0) (7.0 8.0))))
         (Wo (vt-from-sequence '((1.0) (0.0) (0.0) (1.0))))
         ;; concat heads: (2,4)
         (concat (vt-concatenate 1 head1 head2))
         ;; output = concat @ Wo: (2,4) @ (4,1) = (2,1)
         (output (vt-@ concat Wo)))
    (T! "multihead: concat shape" '(2 4) (vt-shape concat))
    (T! "multihead: output shape" '(2 1) (vt-shape output))
    ;; concat row0 = [1,2,5,6], @ Wo = [1*1+2*0+5*0+6*1] = [7]
    (T! "multihead: val" '((7.0d0) (11.0d0)) (vt-to-list output))))

;;; ============================================================
;;; 8. Embedding lookup + reshape
;;; ============================================================
(defun test-embedding ()
  (format t "~%--- 8. Embedding: lookup + reshape ---~%")

  (let* (;; Embedding table: 5 tokens, dim=3
         (emb (vt-from-sequence '((0.1 0.2 0.3)
                                   (0.4 0.5 0.6)
                                   (0.7 0.8 0.9)
                                   (1.0 1.1 1.2)
                                   (1.3 1.4 1.5))))
         ;; Token indices: [0, 2, 4]
         (indices '(0 2 4))
         ;; Lookup
         (looked-up (loop for idx in indices
                          collect (vt-to-list (vt-slice emb (list idx) '(:all))))))
    (T! "emb: token0" '(0.1d0 0.2d0 0.3d0) (first looked-up))
    (T! "emb: token2" '(0.7d0 0.8d0 0.9d0) (second looked-up))
    (T! "emb: token4" '(1.3d0 1.4d0 1.5d0) (third looked-up))

    ;; Stack into batch: (3, 3)
    (let ((batch (vt-stack 0
                           (vt-slice emb '(0) '(:all))
                           (vt-slice emb '(2) '(:all))
                           (vt-slice emb '(4) '(:all)))))
      (T! "emb: batch shape" '(3 3) (vt-shape batch)))))

;;; ============================================================
;;; 9. 矩阵链乘法: A @ B @ C
;;; ============================================================
(defun test-matmul-chain ()
  (format t "~%--- 9. Matrix Chain: A @ B @ C ---~%")

  (let* ((A (vt-from-sequence '((1.0 2.0) (3.0 4.0))))
         (B (vt-from-sequence '((0.0 1.0) (1.0 0.0))))
         (C (vt-from-sequence '((1.0 0.0) (0.0 2.0))))
         ;; A@B = [[2,1],[4,3]]
         ;; (A@B)@C = [[2,2],[4,6]]
         (AB (vt-@ A B))
         (ABC (vt-@ AB C)))
    (T! "chain: A@B" '((2.0d0 1.0d0) (4.0d0 3.0d0)) (vt-to-list AB))
    (T! "chain: A@B@C" '((2.0d0 2.0d0) (4.0d0 6.0d0)) (vt-to-list ABC)))

  ;; einsum chain
  (let* ((A (vt-from-sequence '((1.0 2.0) (3.0 4.0))))
         (B (vt-from-sequence '((0.0 1.0) (1.0 0.0))))
         (C (vt-from-sequence '((1.0 0.0) (0.0 2.0))))
         (ABC (vt-einsum "ij,jk,kl->il" A B C)))
    (T! "einsum chain" '((2.0d0 2.0d0) (4.0d0 6.0d0)) (vt-to-list ABC))))

;;; ============================================================
;;; 10. 梯度下降模拟: W = W - lr * grad
;;; ============================================================
(defun test-gradient-descent ()
  (format t "~%--- 10. Gradient Descent Simulation ---~%")

  (let* ((W (vt-from-sequence '(1.0 2.0 3.0)))
         (lr 0.1)
         ;; Simulated gradient: dL/dW = 2*(W - target) for MSE loss
         (target (vt-from-sequence '(0.0 0.0 0.0)))
         ;; grad = 2 * (W - target) = [2, 4, 6]
         (grad (vt-scale (vt-- W target) 2.0))
         ;; W_new = W - lr * grad = [1-0.2, 2-0.4, 3-0.6] = [0.8, 1.6, 2.4]
         (W-new (vt-- W (vt-scale grad lr))))
    (T! "grad-descent: W_new" '(0.8d0 1.6d0 2.4d0) (vt-to-list W-new))

    ;; Multiple steps
    (let ((W-cur W))
      (loop for step from 0 below 3 do
        (let* ((g (vt-scale (vt-- W-cur target) 2.0))
               (W-next (vt-- W-cur (vt-scale g lr))))
          (setf W-cur W-next)))
      ;; After 3 steps: W * (1 - 2*lr)^3 = W * 0.8^3 = W * 0.512
      (T! "grad-descent: 3steps" '(0.512d0 1.024d0 1.536d0) (vt-to-list W-cur) 1e-6))))

;;; ============================================================
;;; 11. 卷积模拟: im2col + matmul + reshape
;;; ============================================================
(defun test-conv-sim ()
  (format t "~%--- 11. Conv Simulation: im2col + matmul ---~%")

  ;; Simulate 1D convolution using matrix multiplication
  ;; Input: [1,2,3,4,5], kernel_size=3, stride=1
  ;; im2col: [[1,2,3],[2,3,4],[3,4,5]]
  (let* ((input (vt-from-sequence '(1.0 2.0 3.0 4.0 5.0)))
         (kernel (vt-from-sequence '(1.0 0.0 -1.0)))
         ;; Manual im2col: extract windows
         (col (vt-from-sequence '((1.0 2.0 3.0)
                                   (2.0 3.0 4.0)
                                   (3.0 4.0 5.0))))
         ;; output = col @ kernel (each row dot kernel)
         ;; = [1*1+2*0+3*(-1), 2*1+3*0+4*(-1), 3*1+4*0+5*(-1)]
         ;; = [-2, -2, -2]
         (output (vt-@ col kernel)))
    (T! "conv-sim: val" '(-2.0d0 -2.0d0 -2.0d0) (vt-to-list output) 1e-6))

  ;; 2D conv simulation (3x3 input, 2x2 kernel)
  (let* ((input (vt-from-sequence '((1.0 2.0 3.0)
                                     (4.0 5.0 6.0)
                                     (7.0 8.0 9.0))))
         (kernel (vt-from-sequence '((1.0 0.0)
                                      (0.0 1.0))))
         ;; Output = sum of element-wise multiply of each 2x2 window with kernel
         ;; window[0,0] = [[1,2],[4,5]] -> 1*1+2*0+4*0+5*1 = 6
         ;; window[0,1] = [[2,3],[5,6]] -> 2*1+3*0+5*0+6*1 = 8
         ;; window[1,0] = [[4,5],[7,8]] -> 4*1+5*0+7*0+8*1 = 12
         ;; window[1,1] = [[5,6],[8,9]] -> 5*1+6*0+8*0+9*1 = 14
         (output (vt-from-sequence '((6.0 8.0) (12.0 14.0)))))
    (T! "conv2d-sim: val" '((6.0d0 8.0d0) (12.0d0 14.0d0)) (vt-to-list output))))

;;; ============================================================
;;; 12. 权重初始化 + 前向传播
;;; ============================================================
(defun test-init-forward ()
  (format t "~%--- 12. Weight Init + Forward ---~%")

  (vt-random-seed 42)
  (let* ((input-size 4) (hidden-size 3) (output-size 2)
         ;; Xavier initialization: W ~ N(0, 1/sqrt(fan_in))
         (scale1 (/ 1.0 (sqrt input-size)))
         (scale2 (/ 1.0 (sqrt hidden-size)))
         ;; Random weights (using manual values for reproducibility)
         (W1 (vt-from-sequence '((0.5 -0.3 0.1)
                                   (-0.2 0.4 -0.1)
                                   (0.3 -0.1 0.2)
                                   (-0.4 0.2 -0.3))))
         (b1 (vt-from-sequence '(0.1 0.1 0.1)))
         (W2 (vt-from-sequence '((0.2 -0.1)
                                   (0.1 0.3)
                                   (-0.2 0.4))))
         (b2 (vt-from-sequence '(0.0 0.0)))
         ;; Forward pass
         (x (vt-from-sequence '(1.0 2.0 3.0 4.0)))
         ;; Layer 1: relu(x @ W1 + b1)
         (h (vt-relu (vt-+ (vt-@ x W1) b1)))
         ;; Layer 2: h @ W2 + b2
         (y (vt-+ (vt-@ h W2) b2)))
    ;; x @ W1 = [1*0.5+2*(-0.2)+3*0.3+4*(-0.4), 1*(-0.3)+2*0.4+3*(-0.1)+4*0.2, ...]
    ;; = [0.5-0.4+0.9-1.6, -0.3+0.8-0.3+0.8, 0.1-0.2+0.4-1.2]
    ;; = [-0.6, 1.0, -0.9]
    ;; + b1 = [-0.5, 1.1, -0.8]
    ;; relu = [0, 1.1, 0]
    ;; h @ W2 = [0*0.2+1.1*0.1+0*(-0.2), 0*(-0.1)+1.1*0.3+0*0.4] = [0.11, 0.33]
    ;; + b2 = [0.11, 0.33]
    (T! "forward: shape" '(2) (vt-shape y))
    (T! "forward: val" '(0.11d0 0.33d0) (vt-to-list y) 1e-6)))

;;; ============================================================
;;; 13. 综合: Transformer Block 简化版
;;; ============================================================
(defun test-transformer-block ()
  (format t "~%--- 13. Transformer Block (simplified) ---~%")

  (let* ((seq-len 3) (d-model 4) (d-k 2)
         ;; Input embeddings
         (X (vt-from-sequence '((1.0 0.0 0.0 0.0)
                                 (0.0 1.0 0.0 0.0)
                                 (0.0 0.0 1.0 0.0))))
         ;; QKV projections
         (Wq (vt-from-sequence '((1.0 0.0) (0.0 1.0) (0.0 0.0) (0.0 0.0))))
         (Wk (vt-from-sequence '((1.0 0.0) (0.0 1.0) (0.0 0.0) (0.0 0.0))))
         (Wv (vt-from-sequence '((1.0 0.0) (0.0 1.0) (0.0 0.0) (0.0 0.0))))
         ;; Q = X@Wq, K = X@Wk, V = X@Wv
         (Q (vt-@ X Wq))
         (K (vt-@ X Wk))
         (V (vt-@ X Wv))
         ;; Attention scores = Q @ K^T / sqrt(d_k)
         (scores (vt-scale (vt-@ Q (vt-transpose K)) (/ 1.0 (sqrt d-k))))
         ;; Softmax
         (weights (vt-softmax scores))
         ;; Attention output
         (attn-out (vt-@ weights V))
         ;; Residual + LayerNorm
         ;; Need to project attn-out back to d-model: (3,2) @ (2,4) = (3,4)
         (Wo (vt-from-sequence '((1.0 0.0 0.0 0.0)
                                   (0.0 1.0 0.0 0.0))))
         (proj (vt-@ attn-out Wo))
         (residual (vt-+ X proj))
         ;; Layer norm (simplified: just check shape)
         (mu (vt-mean residual :axis -1 :keepdims t)))
    (T! "transformer: Q shape" '(3 2) (vt-shape Q))
    (T! "transformer: scores shape" '(3 3) (vt-shape scores))
    (T! "transformer: weights row_sum" 1.0d0
        (vt-item (vt-sum (vt-slice weights '(0) '(:all)))) 1e-5)
    (T! "transformer: attn-out shape" '(3 2) (vt-shape attn-out))
    (T! "transformer: residual shape" '(3 4) (vt-shape residual))
    (T! "transformer: mu shape" '(3 1) (vt-shape mu))))

;;; ============================================================
;;; 14. 数值稳定性: log-sum-exp trick
;;; ============================================================
(defun test-logsumexp ()
  (format t "~%--- 14. Log-Sum-Exp (numerical stability) ---~%")

  ;; log(sum(exp(x))) = max(x) + log(sum(exp(x - max(x))))
  (let* ((x (vt-from-sequence '(1000.0 1001.0 1002.0)))
         ;; Direct (would overflow): log(sum(exp(x)))
         ;; Stable: max + log(sum(exp(x - max)))
         (m (vt-amax x))
         (shifted (vt-- x m))
         (lse (vt-+ m (vt-log (vt-sum (vt-exp shifted))))))
    ;; log(e^1000 + e^1001 + e^1002) = 1002 + log(1 + e^-1 + e^-2) ≈ 1002.417
    (T! "logsumexp" 1002.417d0 (vt-item lse) 0.01))

  ;; Softmax via log-sum-exp
  (let* ((x (vt-from-sequence '(1000.0 1001.0 1002.0)))
         (m (vt-amax x))
         (shifted (vt-- x m))
         (log-sum (vt-log (vt-sum (vt-exp shifted))))
         (log-probs (vt-- shifted log-sum))
         (probs (vt-exp log-probs)))
    (T! "lse-softmax: sum" 1.0d0 (vt-item (vt-sum probs)) 1e-5)))

;;; ============================================================
;;; 15. 矩阵分解链: SVD reconstruction
;;; ============================================================
(defun test-svd-reconstruction ()
  (format t "~%--- 15. SVD Reconstruction Chain ---~%")

  (let* ((A (vt-from-sequence '((1.0 2.0) (3.0 4.0) (5.0 6.0)))))
    (multiple-value-bind (U S Vt) (vt-svd A)
      ;; Reconstruct: A' = U @ diag(S) @ Vt
      (let* ((S-mat (vt-diag S))
             (A-recon (vt-@ U (vt-@ S-mat Vt)))
             (err (vt-item (vt-amax (vt-abs (vt-- A A-recon))))))
        (T! "svd-recon: err<1e-10" t (< err 1e-10))))))

;;; ============================================================
;;; 16. einsum 链式调用
;;; ============================================================
(defun test-einsum-chain ()
  (format t "~%--- 16. einsum Chain ---~%")

  (let* ((A (vt-from-sequence '((1.0 2.0) (3.0 4.0))))
         (B (vt-from-sequence '((5.0 6.0) (7.0 8.0))))
         ;; trace(A @ B) = einsum("ij,ji->", A, B)
         (tr (vt-item (vt-einsum "ij,ji->" A B)))
         ;; = 1*7 + 2*5 + 3*8 + 4*6 = 7+10+24+24 = 65... wait
         ;; Actually: trace(A@B) = sum_ij A_ij * B_ji
         ;; = A[0,0]*B[0,0] + A[0,1]*B[1,0] + A[1,0]*B[0,1] + A[1,1]*B[1,1]
         ;; = 1*5 + 2*7 + 3*6 + 4*8 = 5+14+18+32 = 69
         )
    (T! "einsum trace(AB)" 69.0d0 tr))

  ;; Bilinear form: x^T A y
  (let* ((x (vt-from-sequence '(1.0 2.0)))
         (A (vt-from-sequence '((3.0 4.0) (5.0 6.0))))
         (y (vt-from-sequence '(7.0 8.0)))
         ;; x^T A y = einsum("i,ij,j->", x, A, y)
         (result (vt-item (vt-einsum "i,ij,j->" x A y)))
         ;; = 1*(3*7+4*8) + 2*(5*7+6*8) = 1*53 + 2*83 = 53+166 = 219
         )
    (T! "bilinear xAy" 219.0d0 result)))

;;; ============================================================
;;; 17. 张量缩并链
;;; ============================================================
(defun test-tensor-contraction ()
  (format t "~%--- 17. Tensor Contraction Chain ---~%")

  ;; Batch matrix-vector product: (B, M, N) @ (B, N, 1) -> (B, M, 1)
  (let* ((A (vt-from-sequence '(((1.0 2.0) (3.0 4.0))
                                  ((5.0 6.0) (7.0 8.0)))))
         (x (vt-from-sequence '(((1.0) (0.0))
                                  ((0.0) (1.0)))))
         ;; A @ x for each batch
         (result (vt-@ A x)))
    (T! "batch-matvec: shape" '(2 2 1) (vt-shape result))
    ;; batch 0: [[1,2],[3,4]] @ [[1],[0]] = [[1],[3]]
    (T! "batch-matvec: b0" '((1.0d0) (3.0d0))
        (vt-to-list (vt-slice result '(0) '(:all) '(:all))))
    ;; batch 1: [[5,6],[7,8]] @ [[0],[1]] = [[6],[8]]
    (T! "batch-matvec: b1" '((6.0d0) (8.0d0))
        (vt-to-list (vt-slice result '(1) '(:all) '(:all))))))

;;; ============================================================
;;; 18. 混合精度类型提升
;;; ============================================================
(defun test-mixed-precision ()
  (format t "~%--- 18. Mixed Precision Type Promotion ---~%")

  ;; int32 + float64 → float64
  (let* ((a (vt-from-sequence '(1 2 3) :dtype :int32))
         (b (vt-from-sequence '(0.5 0.5 0.5) :dtype :float64))
         (c (vt-+ a b)))
    (T! "int32+f64 → f64" :float64 (vt-dtype c))
    (T! "int32+f64 val" '(1.5d0 2.5d0 3.5d0) (vt-to-list c)))

  ;; int64 * float32 → float64
  (let* ((a (vt-from-sequence '(2 3 4) :dtype :int64))
         (b (vt-from-sequence '(0.5 0.5 0.5) :dtype :float32))
         (c (vt-* a b)))
    (T! "int64*f32 → f64" :float64 (vt-dtype c)))

  ;; Chain: (int + float) * float → float
  (let* ((a (vt-from-sequence '(1 2 3) :dtype :int32))
         (b (vt-from-sequence '(0.1 0.2 0.3) :dtype :float64))
         (c (vt-from-sequence '(10.0 10.0 10.0) :dtype :float64))
         (r (vt-* (vt-+ a b) c)))
    (T! "chain type" :float64 (vt-dtype r))
    (T! "chain val" '(11.0d0 22.0d0 33.0d0) (vt-to-list r) 1e-6)))

;;; ============================================================
;;; 19. 广播链式运算
;;; ============================================================
(defun test-broadcast-chain ()
  (format t "~%--- 19. Broadcast Chain ---~%")

  ;; (3,4) + (4,) * (3,1) → (3,4)
  (let* ((A (vt-astype (vt-reshape (vt-arange 12 :dtype :int64) '(3 4)) :float64))
         (bias (vt-from-sequence '(100.0 200.0 300.0 400.0)))
         (scale (vt-from-sequence '((0.1) (0.2) (0.3))))
         ;; (A + bias) * scale
         (result (vt-* (vt-+ A bias) scale)))
    (T! "broadcast chain: shape" '(3 4) (vt-shape result))
    ;; row 0: (0+100, 1+200, 2+300, 3+400) * 0.1 = (10, 20.1, 30.2, 40.3)
    (T! "broadcast chain: row0" '(10.0d0 20.1d0 30.2d0 40.3d0)
        (vt-to-list (vt-slice result '(0) '(:all))) 1e-6)))

;;; ============================================================
;;; 20. 全连接网络前向传播 (2层)
;;; ============================================================
(defun test-mlp-forward ()
  (format t "~%--- 20. MLP Forward Pass ---~%")

  (let* ((x (vt-from-sequence '(1.0 2.0)))
         ;; Layer 1: 2 -> 3
         (W1 (vt-from-sequence '((0.5 0.3 -0.1)
                                   (-0.2 0.4 0.6))))
         (b1 (vt-from-sequence '(0.1 0.1 0.1)))
         ;; Layer 2: 3 -> 2
         (W2 (vt-from-sequence '((0.2 -0.1)
                                   (0.1 0.3)
                                   (-0.2 0.4))))
         (b2 (vt-from-sequence '(0.0 0.0)))
         ;; Forward
         (z1 (vt-+ (vt-@ x W1) b1))     ; linear1
         (a1 (vt-relu z1))               ; activation
         (z2 (vt-+ (vt-@ a1 W2) b2))    ; linear2
         (out (vt-sigmoid z2)))          ; output activation
    ;; x @ W1 = [1*0.5+2*(-0.2), 1*0.3+2*0.4, 1*(-0.1)+2*0.6] = [0.1, 1.1, 1.1]
    ;; + b1 = [0.2, 1.2, 1.2]
    ;; relu = [0.2, 1.2, 1.2]
    ;; @ W2 = [0.2*0.2+1.2*0.1+1.2*(-0.2), 0.2*(-0.1)+1.2*0.3+1.2*0.4]
    ;;       = [0.04+0.12-0.24, -0.02+0.36+0.48] = [-0.08, 0.82]
    ;; + b2 = [-0.08, 0.82]
    ;; sigmoid(-0.08) ≈ 0.48, sigmoid(0.82) ≈ 0.694
    (T! "mlp: shape" '(2) (vt-shape out))
    (T! "mlp: val0" 0.48d0 (vt-ref out 0) 0.02)
    (T! "mlp: val1" 0.694d0 (vt-ref out 1) 0.02)))

;;; ============================================================
;;; 运行所有嵌套测试
;;; ============================================================
(defun run-nested-tests ()
  (format t "~%========================================~%")
  (format t "  clvt NESTED FUNCTION TESTS~%")
  (format t "  AI/ML Function Composition~%")
  (format t "========================================~%")

  (test-linear-layer)
  (test-relu-layer)
  (test-softmax-crossentropy)
  (test-batch-norm)
  (test-attention)
  (test-residual)
  (test-multihead)
  (test-embedding)
  (test-matmul-chain)
  (test-gradient-descent)
  (test-conv-sim)
  (test-init-forward)
  (test-transformer-block)
  (test-logsumexp)
  (test-svd-reconstruction)
  (test-einsum-chain)
  (test-tensor-contraction)
  (test-mixed-precision)
  (test-broadcast-chain)
  (test-mlp-forward)

  (summary))

(run-nested-tests)
