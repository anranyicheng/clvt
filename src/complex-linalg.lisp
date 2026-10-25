;;;; complex-linalg.lisp — 复数线性代数：LU / Cholesky / QR / SVD /
;;;;                       厄米特特征值 / det / inv / solve / pinv / lstsq / rank
;;;;
;;;; 设计约束：
;;;;
;;;;  1) **实数路径零改动**：每个入口先捕获原实现（%CX-ORIG），
;;;;     只有「含复数」的调用进本文件的内核。
;;;;
;;;;  2) **算法直接写在复数域上**，不做 2n×2n 实数嵌入。理由：
;;;;     嵌入后 SVD/特征值的特征向量要再拆回复数，误差与代码量都翻倍；
;;;;     而复数 LU/Cholesky/QR/Jacobi 的公式只是比实数版本多几次
;;;;     CONJ，直接实现更短也更准。
;;;;
;;;;  3) **所有「转置」都是共轭转置**（X^H）。这是复数线代与实数线代
;;;;     唯一真正的语义分叉：A^T A 在复数里必须写成 A^H A，否则得到的
;;;;     不是厄米特矩阵，Cholesky / 特征值 / SVD 全部失效。
;;;;
;;;;  4) **声明分级沿用 §16.14.1**：内核用 (speed 3) (safety 1)，
;;;;     入口层不写 optimize（契约检查与错误信息都在那一层）。

(in-package :clvt)

;;; ==================================================================
;;; 1. 表示与工具
;;; ==================================================================
;;; 内部统一用「行主序、C 连续、complex128」的裸缓冲：
;;;   元素 (i,j) 的线性下标 = i*LD + j（LD = leading dimension）。
;;; 对外仍然是 vt 张量；缓冲只在算法内部流转，不泄漏步长概念。

(declaim (inline %cl-i %cl-c0 %cl-zero))

(defun %cl-i (ld i j)
  "行主序线性下标。"
  (declare (type fixnum ld i j))
  (the fixnum (+ (the fixnum (* i ld)) j)))

(defun %cl-c0 ()
  "complex128 的 0+0i。"
  (complex 0.0d0 0.0d0))

(defun %cl-zero (x)
  "判定 X 是否**精确**为零（复数两个分量都为 0）。"
  (and (zerop (realpart x)) (zerop (imagpart x))))

(defun %cl-abs (x)
  "复数模（避免 SBCL 的 (abs complex) 在极端量级下溢出）。"
  (%cx-hypot (realpart x) (imagpart x)))

(defun %cl-cplx-2d (x what)
  "把 X 落地成 2 维 C 连续 complex128 **副本**，返回 (values data m n)。
   副本保证内核可以原地改写而不污染调用方。"
  (let* ((v (ensure-vt x))
         (shape (vt-shape v)))
    (unless (= 2 (length shape))
      (error "~a: 需要 2 维矩阵，收到形状 ~a" what shape))
    (let* ((c (vt-contiguous (vt-astype (vt-copy v) :complex128)))
           (m (first shape))
           (n (second shape))
           (data (vt-data c)))
      (declare (type (simple-array (complex double-float) (*)) data))
      (values data m n))))

(defun %cl-make-buf (size &key (element-type '(complex double-float)))
  "分配长度为 SIZE 的裸缓冲。"
  (make-array (max size 1) :element-type element-type
                          :initial-element (if (equal element-type '(complex double-float))
                                               (complex 0.0d0 0.0d0)
                                               0.0d0)))

(defun %cl-vt (data shape dtype)
  "把裸缓冲包装成 vt 张量（不复制）。"
  (%make-vt :data data :shape (copy-list shape)
            :strides (vt-compute-strides shape) :offset 0 :dtype dtype))

(defun %cl-eye-buf (n)
  "n×n complex128 单位阵缓冲。"
  (let ((buf (%cl-make-buf (* n n))))
    (dotimes (i n buf)
      (setf (aref buf (%cl-i n i i)) (complex 1.0d0 0.0d0)))))

(defun %cl-conj-t-of (data m n)
  "把 m×n 拷贝成 n×m 的共轭转置缓冲。"
  (let ((out (%cl-make-buf (* m n))))
    (declare (type (simple-array (complex double-float) (*)) data out))
    (dotimes (i m out)
      (dotimes (j n)
        (setf (aref out (%cl-i m j i)) (%cconj (aref data (%cl-i n i j))))))))

;;; ==================================================================
;;; 2. LU 分解（部分选主元，按 |a| 选）
;;; ==================================================================
;;; 与实数版本的唯一差别是选主元的判据：实数用 |a|，复数同样用 |a|
;;; （模），因为复数没有全序。消元公式本身一模一样 —— 除法/乘法在
;;; 复数域上封闭，不需要任何额外共轭。

(defun %clu-factor! (a ld m n piv)
  "原地 LU 分解。A 为 m×n 行主序（行距 LD），PIV 为长度 m 的 fixnum
   向量（调用方初始化为 0..m-1，返回时记录行置换）。
   返回 (values sign singular-p)：SIGN 为 ±1（行交换次数奇偶），
   SINGULAR-P 为真表示某一列主元全零。"
  (declare (type (simple-array (complex double-float) (*)) a)
           (type (simple-array fixnum (*)) piv)
           (type fixnum ld m n)
           (optimize (speed 3) (safety 1)))
  (let ((kmax (min m n))
        (sign 1)
        (singular nil))
    (declare (type fixnum kmax sign))
    (dotimes (k kmax)
      (declare (type fixnum k))
      ;; ---- 选主元：当前列 k 中行 k..m-1 里 |a| 最大者 ----
      (let ((p k)
            (best 0.0d0))
        (declare (type fixnum p) (type double-float best))
        (loop for i of-type fixnum from k below m do
          (let ((v (%cl-abs (aref a (%cl-i ld i k)))))
            (declare (type double-float v))
            (when (> v best) (setf best v p i))))
        (cond
          ((zerop best)
           ;; 该列以下全零：矩阵奇异。记录下来但仍继续（vt-lu 需要完整
           ;; 的 LU 形态；det 会据此返回 0）。
           (setf singular t))
          (t
           (unless (= p k)
             (setf sign (- sign))
             (rotatef (aref piv p) (aref piv k))
             (loop for j of-type fixnum from 0 below n
                   do (rotatef (aref a (%cl-i ld p j))
                               (aref a (%cl-i ld k j)))))
           (let ((piv-k (aref a (%cl-i ld k k))))
             (declare (type (complex double-float) piv-k))
             (loop for i of-type fixnum from (1+ k) below m do
               (let ((lik (/ (aref a (%cl-i ld i k)) piv-k)))
                 (declare (type (complex double-float) lik))
                 (setf (aref a (%cl-i ld i k)) lik)
                 (loop for j of-type fixnum from (1+ k) below n
                       do (decf (aref a (%cl-i ld i j))
                                (* lik (aref a (%cl-i ld k j))))))))))))
    (values sign singular)))

(defun %clu-permute-rows! (b ldb n nrhs piv)
  "按 PIV 重排 B 的行（P·b：新第 i 行 = 原第 PIV[i] 行）。"
  (declare (type (simple-array (complex double-float) (*)) b)
           (type (simple-array fixnum (*)) piv)
           (type fixnum ldb n nrhs)
           (optimize (speed 3) (safety 1)))
  (let ((tmp (%cl-make-buf (* n ldb))))
    (declare (type (simple-array (complex double-float) (*)) tmp))
    (dotimes (i n)
      (declare (type fixnum i))
      (let ((src (* (aref piv i) ldb))
            (dst (* i ldb)))
        (declare (type fixnum src dst))
        (dotimes (c nrhs)
          (declare (type fixnum c))
          (setf (aref tmp (+ dst c)) (aref b (+ src c))))))
    (dotimes (k (* n ldb))
      (declare (type fixnum k))
      (setf (aref b k) (aref tmp k)))
    b))

(defun %clu-subst! (lu ld n piv b ldb nrhs)
  "用 LU 结果解 A X = B（B 原地变成解）。奇异主元 → 报错（对标
   numpy.linalg.LinAlgError: Singular matrix）。"
  (declare (type (simple-array (complex double-float) (*)) lu b)
           (type (simple-array fixnum (*)) piv)
           (type fixnum ld n ldb nrhs)
           (optimize (speed 3) (safety 1)))
  (%clu-permute-rows! b ldb n nrhs piv)
  ;; 前代：L 是单位下三角（对角 1，下三角存乘数）
  (dotimes (i n)
    (declare (type fixnum i))
    (loop for j of-type fixnum from 0 below i do
      (let ((lij (aref lu (%cl-i ld i j))))
        (declare (type (complex double-float) lij))
        (loop for c of-type fixnum from 0 below nrhs
              do (decf (aref b (+ (* i ldb) c))
                       (* lij (aref b (+ (* j ldb) c))))))))
  ;; 回代：U 上三角
  (loop for i of-type fixnum from (1- n) downto 0 do
    (let ((d (aref lu (%cl-i ld i i))))
      (declare (type (complex double-float) d))
      (when (%cl-zero d)
        (error "vt-solve: 矩阵奇异（第 ~a 个主元为零），无法求解" i))
      (loop for c of-type fixnum from 0 below nrhs
            do (setf (aref b (+ (* i ldb) c))
                     (/ (aref b (+ (* i ldb) c)) d)))
      (loop for j of-type fixnum from 0 below i do
        (let ((u (aref lu (%cl-i ld j i))))
          (declare (type (complex double-float) u))
          (loop for c of-type fixnum from 0 below nrhs
                do (decf (aref b (+ (* j ldb) c))
                         (* u (aref b (+ (* i ldb) c)))))))))
  b)

;;; ==================================================================
;;; 3. 对外：vt-lu / vt-det / vt-solve / vt-inv
;;; ==================================================================

(defun %cl-plain-1d->2d (v n what)
  "把 b（1 维或 2 维）落地成 n×nrhs 的 complex128 副本，返回
   (values buf nrhs orig-1d-p)。"
  (let* ((shape (vt-shape v))
         (one-d (or (null shape) (= 1 (length shape)))))
    (unless (or (null shape) (<= (length shape) 2))
      (error "~a: b 至多 2 维，收到形状 ~a" what shape))
    (let* ((rows (cond ((null shape) 1) (t (first shape))))
           (nrhs (cond ((or (null shape) (= 1 (length shape))) 1)
                       (t (second shape)))))
      (unless (= rows n)
        (error "~a: b 的行数 ~a 与系数矩阵阶数 ~a 不匹配（b 形状 ~a）"
               what rows n shape))
      (let* ((c (vt-contiguous (vt-astype (vt-copy (ensure-vt v)) :complex128))))
        (values (vt-data c) nrhs one-d)))))

(defun %cx-op-lu (matrix)
  "部分选主元 LU 分解（复数版）。返回 (values a piv sign)：
   a    : 原地修改后的合并 LU 矩阵（complex128，非方阵亦可）
   piv  : fixnum 向量（行置换），满足 P·A = L·U，P[i, piv[i]] = 1
   sign : 行交换符号（+1 / -1）

   与实数版的唯一差别是选主元用**模**（复数无全序）。"
  (let ((v (ensure-vt matrix)))
    (with-float-safe
    (multiple-value-bind (a m n) (%cl-cplx-2d v "vt-lu")
      (let ((piv (make-array m :element-type 'fixnum)))
        (dotimes (i m) (setf (aref piv i) i))
        ;; 行主序 m×n 的行距是列数 n（不是行数 m）—— 非方阵时写反会静默错位
          (multiple-value-bind (sign singular) (%clu-factor! a n m n piv)
          (declare (ignore singular))
          (values (%cl-vt a (list m n) :complex128) piv sign)))))))


(defun %cx-op-det (matrix &key out)
  "基于 LU 分解的行列式（复数版，返回 0 维 complex128 张量）。
   奇异矩阵 → 0+0i（不报错，与 numpy.linalg.det 一致）。"
  (let ((v (ensure-vt matrix)))
    (with-float-safe
    (let ((shape (vt-shape v)))
      (unless (and (= 2 (length shape)) (= (first shape) (second shape)))
        (error "vt-det: 行列式仅支持方阵，收到形状 ~a" shape))
      (when out (vt-check-out out nil :complex128 :op-name "vt-det"))
      (multiple-value-bind (a n n2) (%cl-cplx-2d v "vt-det")
        (declare (ignore n2))
        (let ((piv (make-array n :element-type 'fixnum)))
          (dotimes (i n) (setf (aref piv i) i))
          (multiple-value-bind (sign singular) (%clu-factor! a n n n piv)
            (let ((det (if singular
                           (%cl-c0)
                           (let ((acc (complex (coerce sign 'double-float) 0.0d0)))
                             (declare (type (complex double-float) acc))
                             (dotimes (i n acc)
                               (setf acc (* acc (aref a (%cl-i n i i)))))))))
              (if out
                  (progn (vt-fill out det) out)
                  (make-vt nil det :dtype :complex128))))))))))


(defun %cx-op-solve (a b &key out)
  "求解线性方程组 A X = B（复数版，支持多右端项）。
   B 为 1 维时返回 1 维解；为 2 维时返回 n×nrhs。
   奇异矩阵 → 报错（对标 numpy.linalg.LinAlgError）。"
  (let ((av (ensure-vt a))
        (bv (ensure-vt b)))
    (with-float-safe
    (let ((a-shape (vt-shape av)))
      (unless (and (= 2 (length a-shape)) (= (first a-shape) (second a-shape)))
        (error "vt-solve: 线性方程组求解要求系数矩阵为方阵，收到形状 ~a" a-shape))
      (multiple-value-bind (lu n n2) (%cl-cplx-2d av "vt-solve")
        (declare (ignore n2))
        (multiple-value-bind (bdata nrhs one-d) (%cl-plain-1d->2d bv n "vt-solve")
          (let ((piv (make-array n :element-type 'fixnum)))
            (dotimes (i n) (setf (aref piv i) i))
            (multiple-value-bind (sign singular) (%clu-factor! lu n n n piv)
              (declare (ignore sign))
              (when singular
                (error "vt-solve: 系数矩阵奇异（LU 主元为零），无法求解"))
              (%clu-subst! lu n n piv bdata nrhs nrhs)
              (let ((res (if one-d
                             (%cl-vt bdata (list n) :complex128)
                             (%cl-vt bdata (list n nrhs) :complex128))))
                (if out
                    (progn (vt-copy-into (vt-check-out
                                          out (vt-shape res) :complex128
                                          :op-name "vt-solve")
                                         res)
                           out)
                    res))))))))))


(defun %cx-op-inv (matrix)
  "矩阵求逆（复数版）。奇异矩阵 → 报错。"
  (let ((v (ensure-vt matrix)))
    (with-float-safe
    (let ((shape (vt-shape v)))
      (unless (and (= 2 (length shape)) (= (first shape) (second shape)))
        (error "vt-inv: 求逆要求方阵，收到形状 ~a" shape))
      (multiple-value-bind (lu n n2) (%cl-cplx-2d v "vt-inv")
        (declare (ignore n2))
        (let ((piv (make-array n :element-type 'fixnum))
              (eye (%cl-eye-buf n)))
          (dotimes (i n) (setf (aref piv i) i))
          (multiple-value-bind (sign singular) (%clu-factor! lu n n n piv)
            (declare (ignore sign))
            (when singular
              (error "vt-inv: 矩阵奇异，无法求逆"))
            (%clu-subst! lu n n piv eye n n)
            (%cl-vt eye (list n n) :complex128))))))))


(defun %cx-op-matrix-power (matrix n)
  "整数幂方阵（复数版转由 vt-matmul / vt-inv 组合；实数路径不变）。"
  (let ((v (ensure-vt matrix)))
    (with-float-safe
    (let ((shape (vt-shape v)))
      (unless (and (= 2 (length shape)) (= (first shape) (second shape)))
        (error "vt-matrix-power: 要求方阵，收到形状 ~a" shape))
      ;; 快速幂：acc = A^k（n<0 时先求逆）。负幂下 base 就是 A^-1，
      ;; 后续照样走平方-乘，与 numpy.linalg.matrix_power 的语义一致。
      (let ((k (abs n))
            (acc (vt-eye (first shape) :dtype :complex128))
            (b (if (minusp n) (vt-inv v) (vt-astype v :complex128))))
        (loop while (plusp k) do
          (when (oddp k) (setf acc (vt-matmul acc b)))
          (setf k (ash k -1))
          (when (plusp k) (setf b (vt-matmul b b))))
        acc)))))


;;; ==================================================================
;;; 4. Cholesky（厄米特正定）
;;; ==================================================================
;;; A = L·L^H（下三角）。与实数版的区别：
;;;   · 内积是 <l_ik, l_jk> = l_ik · conj(l_jk)；
;;;   · 对角元 a_jj - Σ|l_jk|² 必须是**实数且为正**，否则不是正定矩阵。
;;; 这两点是复数 Cholesky 的全部内容 —— 少了 conj 就会得到错的 L
;;; （而且不会报错，只是结果不满足 L·L^H = A）。

(defun %cchol-factor! (a l n)
  "原地算 L（n×n 行主序）。非正定 → 报错。"
  (declare (type (simple-array (complex double-float) (*)) a l)
           (type fixnum n)
           (optimize (speed 3) (safety 1)))
  (dotimes (j n)
    (declare (type fixnum j))
    ;; 对角：a_jj - Σ_{k<j} |l_jk|²  （实数）
    (let ((s (realpart (aref a (%cl-i n j j)))))
      (declare (type double-float s))
      (loop for k of-type fixnum from 0 below j do
        (let ((v (aref l (%cl-i n j k))))
          (declare (type (complex double-float) v))
          (decf s (%cx-abs2 v))))
      (when (or (not (floatp s)) (<= s 0.0d0))
        (error "vt-cholesky: 矩阵不是厄米特正定（第 ~a 个对角元 ~a ≤ 0）" j s))
      (setf (aref l (%cl-i n j j)) (complex (sqrt s) 0.0d0)))
    ;; 下三角：l_ij = (a_ij - Σ_{k<j} l_ik·conj(l_jk)) / l_jj
    (let ((ljj (aref l (%cl-i n j j))))
      (declare (type (complex double-float) ljj))
      (loop for i of-type fixnum from (1+ j) below n do
        (let ((s (aref a (%cl-i n i j))))
          (declare (type (complex double-float) s))
          (loop for k of-type fixnum from 0 below j do
            (decf s (* (aref l (%cl-i n i k))
                       (%cconj (aref l (%cl-i n j k))))))
          (setf (aref l (%cl-i n i j)) (/ s ljj))))))
  l)

(defun %cx-op-cholesky (matrix &key (upper nil))
  "厄米特正定矩阵的 Cholesky 分解（复数版）。返回 L（下三角）或 L^H。
   非正定 → 报错（对标 numpy.linalg.LinAlgError）。"
  (let ((v (ensure-vt matrix)))
    (with-float-safe
    (let ((shape (vt-shape v)))
      (unless (and (= 2 (length shape)) (= (first shape) (second shape)))
        (error "vt-cholesky: 要求方阵，收到形状 ~a" shape))
      (multiple-value-bind (a n n2) (%cl-cplx-2d v "vt-cholesky")
        (declare (ignore n2))
        (let ((l (%cl-make-buf (* n n))))
          (%cchol-factor! a l n)
          (let ((lt (%cl-vt l (list n n) :complex128)))
            ;; A = L·L^H，所以「上三角」是 L^H = conj-transpose(L)。
            ;; 实数下 L^T = L^H，写 transpose 看不出来；复数下写成
            ;; transpose 会返回 L^T —— 它满足的是 A = L·(L^T)^T，
            ;; 而不是厄米特分解，属于静默错答案。
            (if upper (vt-transpose (vt-conj lt)) lt))))))))


;;; ==================================================================
;;; 5. QR（复 Householder）
;;; ==================================================================
;;; 复数 Householder 反射器 H = I - β·v·v^H。取 v = x - α·e1，
;;; 其中 α = -phase(x₀)·‖x‖（phase 让 α 与 x₀ 反相，避免相消）。
;;; 与实数版的差别全在「左乘用 conj(v) 做内积、右乘外侧才用 v」——
;;; 写反了仍然会得到一个正交矩阵，但**不是**能零化目标列的反射。

(defun %cqr-householder! (r ld m n i betas vbuf voffs)
  "对第 I 列构造反射器并就地作用到 R[i:m, i:n] 与（由调用方）Q 上。
   VBUF/VOFFS 预分配反射器缓冲，返回该反射器的范数平方（0 表示跳过）。"
  (declare (type (simple-array (complex double-float) (*)) r vbuf)
           (type (simple-array double-float (*)) betas)
           (type (simple-array fixnum (*)) voffs)
           (type fixnum ld m n i)
           (optimize (speed 3) (safety 1)))
  (let* ((len (- m i))
         (vo (aref voffs i))
         (v vbuf)
         (norm2 0.0d0))
    (declare (type fixnum len vo) (type double-float norm2))
    ;; 1) 收集 x = R[i:m, i] 到反射器缓冲
    (loop for k of-type fixnum from 0 below len do
      (let ((x (aref r (%cl-i ld (+ i k) i))))
        (setf (aref v (+ vo k)) x)
        (incf norm2 (%cx-abs2 x))))
    (when (<= norm2 0.0d0)
      (setf (aref betas i) 0.0d0)
      (return-from %cqr-householder! 0.0d0))
    (let* ((sigma (sqrt norm2))
           (x0 (aref v vo))
           (ax0 (%cl-abs x0))
           ;; phase：x0 不是零就取它的相位，否则取 1
           (phase (if (zerop ax0) (complex 1.0d0 0.0d0) (/ x0 ax0)))
           (alpha (* (- phase) sigma)))
      (declare (type double-float sigma) (type (complex double-float) phase alpha))
      ;; v = x - alpha·e1
      (decf (aref v vo) alpha)
      ;; beta = 2 / ‖v‖²
      (let ((vn2 0.0d0))
        (declare (type double-float vn2))
        (loop for k of-type fixnum from 0 below len do
          (incf vn2 (%cx-abs2 (aref v (+ vo k)))))
        (when (<= vn2 0.0d0)
          (setf (aref betas i) 0.0d0)
          (return-from %cqr-householder! 0.0d0))
        (let ((beta (/ 2.0d0 vn2)))
          (declare (type double-float beta))
          (setf (aref betas i) beta)
          ;; 2) R[i:m, i:n] ← (I - β v v^H) R
          ;;    w_j = Σ_l conj(v_l)·R[i+l, j]；R[i+k,j] -= β·v_k·w_j
          (loop for j of-type fixnum from i below n do
            (let ((w (%cl-c0)))
              (declare (type (complex double-float) w))
              (loop for k of-type fixnum from 0 below len do
                (incf w (* (%cconj (aref v (+ vo k))) (aref r (%cl-i ld (+ i k) j)))))
              (setf w (* w beta))
              (loop for k of-type fixnum from 0 below len do
                (decf (aref r (%cl-i ld (+ i k) j))
                      (* w (aref v (+ vo k)))))))
          beta)))))

(defun %cqr-apply-left! (q qld rows ncols i beta vbuf voffs)
  "Q ← H_i·Q（**左乘**，H_i = I - β v v^H，v 支撑在行 i..m-1）。

   为什么必须是左乘 + 逆序遍历：
   约化 QR 的 Q 只有 k = min(m,n) 列，而反射器 v 的支撑长度是 m-i，
   通常伸到第 k 列**之外**。若沿用「Q ← Q·H_i 顺次右乘」的写法，
   内积 Σ_k Q[r,k]·v_k 会去读 Q 不存在的列（截断后信息已丢失），
   结果在 m > n（tall）时是错的 —— 方阵与宽矩阵因为 Q 恰好是完整
   的 m×m，碰巧不受影响。
   正确做法是把 Q 写成 H_0·H_1···H_{k-1}·I_{m×k}，从 **i = k-1 递减**
   到 0 依次左乘：每次只改行 i..m-1，这些行在截断的 Q 里都存在。"
  ;; 注意：这里原先还声明了 (type (simple-array double-float (*)) v)，但
  ;; 形参表里**没有** V（反射器存在 VBUF 里，复数值），该声明是一个
  ;; undefined-variable WARNING 的来源；SBCL 因此把本文件编译成
  ;; 「1 个 warning 条件」的文件，进而影响 ASDF 的 failure-p 判定。
  (declare (type (simple-array (complex double-float) (*)) q vbuf)
           (type (simple-array fixnum (*)) voffs)
           (type fixnum qld rows ncols i)
           (type double-float beta)
           (optimize (speed 3) (safety 1)))
  (let* ((len (- rows i))
         (vo (aref voffs i)))
    (declare (type fixnum len vo))
    (when (and (plusp beta) (plusp len))
      (dotimes (j ncols)
        (declare (type fixnum j))
        (let ((w (%cl-c0)))
          (declare (type (complex double-float) w))
          (loop for l of-type fixnum from 0 below len do
            (incf w (* (%cconj (aref vbuf (+ vo l)))
                       (aref q (%cl-i qld (+ i l) j)))))
          (setf w (* w beta))
          (loop for l of-type fixnum from 0 below len do
            (decf (aref q (%cl-i qld (+ i l) j))
                  (* w (aref vbuf (+ vo l)))))))))
  q)

(defun %cx-op-qr (matrix &key (mode :reduced))
  "矩阵 QR 分解（复数版）。返回 (values q r)：
   mode = :reduced → q(m×k)、r(k×n)，k = min(m,n)；
   mode = :full    → q(m×m)、r(m×n)。
   Q 的列是复正交基（Q^H·Q = I）。"
  (let ((v (ensure-vt matrix)))
    (with-float-safe
    (multiple-value-bind (r m n) (%cl-cplx-2d v "vt-qr")
      (let* ((k (min m n))
             (full (eq mode :full))
             (nq (if full m k))
             (q (%cl-make-buf (* m nq)))
             (vtotal (loop for i from 0 below k sum (- m i)))
             (vbuf (%cl-make-buf vtotal))
             (betas (make-array (max k 1) :element-type 'double-float
                                         :initial-element 0.0d0))
             (voffs (make-array (max k 1) :element-type 'fixnum
                                         :initial-element 0)))
        (declare (type (simple-array double-float (*)) betas)
                 (type (simple-array fixnum (*)) voffs))
        (let ((acc 0))
          (dotimes (i k)
            (setf (aref voffs i) acc)
            (incf acc (- m i))))
        ;; Q 初始为单位阵的前 nq 列（m×nq，行距 nq）
        (dotimes (i (min m nq))
          (setf (aref q (%cl-i nq i i)) (complex 1.0d0 0.0d0)))
        ;; 第一遍：逐列构造反射器，就地作用到 R（H_{k-1}···H_0·A = R）
        (dotimes (i k)
          (declare (type fixnum i))
          (%cqr-householder! r n m n i betas vbuf voffs))
        ;; 第二遍：Q = H_0···H_{k-1}·I_{m×nq}，**逆序左乘**
        (dotimes (ii k)
          (declare (type fixnum ii))
          (let* ((i (- k 1 ii))
                 (beta (aref betas i)))
            (when (plusp beta)
              (%cqr-apply-left! q nq m nq i beta vbuf voffs))))
        ;; Q 实际形状 m×nq
        (let ((qvt (%cl-vt q (list m nq) :complex128))
              (rvt (%cl-vt r (if full (list m n) (list k n)) :complex128)))
          (values qvt rvt)))))))


;;; ==================================================================
;;; 6. SVD（单边 Jacobi，复旋转）
;;; ==================================================================
;;; 单边 Jacobi 直接对 A 的列做酉旋转使它们两两正交，不形成 A^H A，
;;; 因此条件数的平方问题不存在。复数旋转 (c 实、s 复) 的推导：
;;;
;;;   记 α = ‖a_p‖²、β = ‖a_q‖²（实数），γ = a_p^H a_q（复数），
;;;   旋转 [a_p a_q]·J，J = [[c, s],[-conj(s), c]]，要求新内积为 0：
;;;       t(α-β) + γ - t²·conj(γ) = 0,  t = s/c
;;;   令 γ = |γ|e^{iφ}、u = (α-β)/|γ|，代换 t = e^{iφ}·w 后方程退化成
;;;   实系数二次式 w² - u·w - 1 = 0，取模 ≤ 1 的小根：
;;;       w = -2 / (u + sign(u)·sqrt(u²+4))
;;;       t = e^{iφ}·w,  c = 1/sqrt(1+w²),  s = t·c
;;;
;;; 这一步是复数 SVD 唯一容易写错的地方：直接把实数的
;;; t = sign(θ)/(|θ|+sqrt(1+θ²)) 套上来会在 γ 非实时不收敛。

(defun %csvd-jacobi! (a m n v max-sweeps tol)
  "单边 Jacobi。A 为 m×n（m ≥ n，行主序、LD = n），V 为 n×n 单位阵，
   结束时 A 的列两两正交（列范数即奇异值）、V 的列是对应右奇异向量。
   返回实际扫描次数。"
  (declare (type (simple-array (complex double-float) (*)) a v)
           (type fixnum m n max-sweeps)
           (type double-float tol)
           (optimize (speed 3) (safety 1)))
  (let ((sweeps 0))
    (declare (type fixnum sweeps))
    (dotimes (sweep max-sweeps)
      (let ((rotated nil))
        (dotimes (p (1- n))
          (declare (type fixnum p))
          (loop for q of-type fixnum from (1+ p) below n do
            (let ((alpha 0.0d0)
                  (beta 0.0d0)
                  (gamma (%cl-c0)))
              (declare (type double-float alpha beta)
                       (type (complex double-float) gamma))
              (dotimes (i m)
                (declare (type fixnum i))
                (let ((ap (aref a (%cl-i n i p)))
                      (aq (aref a (%cl-i n i q))))
                  (declare (type (complex double-float) ap aq))
                  (incf alpha (%cx-abs2 ap))
                  (incf beta (%cx-abs2 aq))
                  (incf gamma (* (%cconj ap) aq))))
              (let ((mag (%cl-abs gamma)))
                (declare (type double-float mag))
                (when (and (plusp mag)
                           (plusp alpha) (plusp beta)
                           (> mag (* tol (sqrt (* alpha beta)))))
                  (setf rotated t)
                  (let* ((u (/ (- alpha beta) mag))
                         (sg (if (>= u 0.0d0) 1.0d0 -1.0d0))
                         (w (/ -2.0d0 (+ u (* sg (sqrt (+ (* u u) 4.0d0))))))
                         (c (/ 1.0d0 (sqrt (+ 1.0d0 (* w w)))))
                         (phase (/ gamma mag))
                         (s (* phase w c)))
                    (declare (type double-float u sg w c)
                             (type (complex double-float) phase s))
                    ;; 旋转 A 的 p/q 列
                    (dotimes (i m)
                      (declare (type fixnum i))
                      (let ((ap (aref a (%cl-i n i p)))
                            (aq (aref a (%cl-i n i q))))
                        (declare (type (complex double-float) ap aq))
                        (setf (aref a (%cl-i n i p)) (- (* c ap) (* (%cconj s) aq)))
                        (setf (aref a (%cl-i n i q)) (+ (* s ap) (* c aq)))))
                    ;; 同样旋转 V 的 p/q 列（累积右奇异向量）
                    (dotimes (i n)
                      (declare (type fixnum i))
                      (let ((vp (aref v (%cl-i n i p)))
                            (vq (aref v (%cl-i n i q))))
                        (declare (type (complex double-float) vp vq))
                        (setf (aref v (%cl-i n i p)) (- (* c vp) (* (%cconj s) vq)))
                        (setf (aref v (%cl-i n i q)) (+ (* s vp) (* c vq)))))))))))
        (incf sweeps)
        (unless rotated (return))))
    sweeps))

(defun coerce-map-double (src idx)
  "按 IDX 重排 SRC（double-float 向量），返回新的 simple-array。
   用于「奇异值/特征值排序后按同一置换重排」。"
  (let* ((n (length idx))
         (out (make-array (max n 1) :element-type 'double-float :initial-element 0.0d0)))
    (declare (type (simple-array double-float (*)) out))
    (dotimes (j n out)
      (setf (aref out j) (aref src (aref idx j))))))

(defun %cl-sort-perm (keys)
  "返回按 KEYS 降序的下标向量（稳定）。"
  (let* ((n (length keys))
         (idx (make-array n :element-type 'fixnum)))
    (dotimes (i n) (setf (aref idx i) i))
    (stable-sort idx #'> :key (lambda (i) (aref keys i)))))

(defun %cl-orthonormal-complete (u-buf m k)
  "把 m×k 的正交列补成 m×m 正交方阵（Gram–Schmidt 从标准基里取）。
   用于 :full-matrices 的 U 补全。"
  (declare (type (simple-array (complex double-float) (*)) u-buf)
           (type fixnum m k))
  (let ((out (%cl-make-buf (* m m)))
        (filled 0))
    (declare (type fixnum filled))
    ;; 先拷入已有的 k 列
    (dotimes (j k)
      (dotimes (i m)
        ;; u-buf 是 m×k（行距 k），out 是 m×m（行距 m）
        (setf (aref out (%cl-i m i j)) (aref u-buf (%cl-i k i j)))))
    (setf filled k)
    ;; 逐个标准基向量做正交化，直到补满
    (dotimes (e m)
      (when (< filled m)
        (let ((cand (%cl-make-buf m)))
          (setf (aref cand e) (complex 1.0d0 0.0d0))
          ;; 减去在已有列上的投影
          (dotimes (j filled)
            (let ((proj (%cl-c0)))
              (dotimes (i m)
                (incf proj (* (%cconj (aref out (%cl-i m i j)))
                              (aref cand i))))
              (dotimes (i m)
                (decf (aref cand i) (* proj (aref out (%cl-i m i j)))))))
          (let ((nrm 0.0d0))
            (dotimes (i m) (incf nrm (%cx-abs2 (aref cand i))))
            (when (> nrm 1.0d-24)
              (let ((inv (/ 1.0d0 (sqrt nrm))))
                (dotimes (i m)
                  (setf (aref out (%cl-i m i filled)) (* (aref cand i) inv))))
              (incf filled))))))
    out))

(defun %csvd-core (a m n &key (max-sweeps 60) (tol 1e-15))
  "对 m×n 复矩阵 A（会被销毁）做 SVD，返回
   (values u-buffer(m×k) s(double k) vh-buffer(k×n) k)，奇异值降序。"
  (declare (type (simple-array (complex double-float) (*)) a)
           (type fixnum m n))
  (let ((k (min m n)))
    (declare (type fixnum k))
    (if (>= m n)
        ;; ---- 高矩阵：直接做单边 Jacobi ----
        (let* ((v (%cl-eye-buf n))
               (colnorms (make-array n :element-type 'double-float :initial-element 0.0d0)))
          (declare (type (simple-array double-float (*)) colnorms))
          (%csvd-jacobi! a m n v max-sweeps (coerce tol 'double-float))
          ;; 列范数 = 奇异值
          (dotimes (j n)
            (let ((acc 0.0d0))
              (dotimes (i m) (incf acc (%cx-abs2 (aref a (%cl-i n i j)))))
              (setf (aref colnorms j) (sqrt acc))))
          (let* ((order (%cl-sort-perm colnorms))
                 (s (coerce-map-double colnorms order))
                 (u (%cl-make-buf (* m k)))
                 (vh (%cl-make-buf (* k n))))
            (declare (type (simple-array (complex double-float) (*)) u vh))
            ;; U 列 = A 的列 / σ；V^H 行 = V 的列取共轭
            (dotimes (j k)
              (let ((sj (aref s j))
                    (oj (aref order j)))
                (dotimes (i m)
                  ;; u 是 m×k，行主序行距 = k（写成 m 在方阵上碰巧对，
                  ;; 非方阵直接越界 —— 这是本轮第二次踩同一个坑）
                  (setf (aref u (%cl-i k i j))
                        (if (> sj 0.0d0)
                            (* (aref a (%cl-i n i oj)) (/ 1.0d0 sj))
                            (%cl-c0))))
                (dotimes (i n)
                  (setf (aref vh (%cl-i n j i)) (%cconj (aref v (%cl-i n i oj)))))))
            (values u s vh k)))
        ;; ---- 宽矩阵：对 A^H 做 SVD，再换回 U/V 的角色 ----
        (let ((ah (%cl-conj-t-of a m n)))       ; n×m
          (multiple-value-bind (uc sc vhc k2)
              (%csvd-core ah n m :max-sweeps max-sweeps :tol tol)
            (declare (ignore k2))
            ;; A = (A^H)^H = (Uc S Vc^H)^H = Vc S Uc^H
            ;;   U = Vc = (Vh_c)^H → u(m×k) 的第 j 列 = conj(vhc 第 j 行)
            ;;   Vh = Uc^H → vh(k×n) 的第 j 行 = conj(uc 第 j 列)
            (let ((u (%cl-make-buf (* m k)))
                  (vh (%cl-make-buf (* k n))))
              (declare (type (simple-array (complex double-float) (*)) u vh))
              (dotimes (j k)
                (dotimes (i m)
                  (setf (aref u (%cl-i m i j)) (%cconj (aref vhc (%cl-i m j i)))))
                (dotimes (i n)
                  ;; uc 是 n×k，行距是 k（不是 n）
                  (setf (aref vh (%cl-i n j i)) (%cconj (aref uc (%cl-i k i j))))))
              (values u sc vh k)))))))

(defun %cx-op-svd (matrix &key (full-matrices nil) (max-sweeps 60) (tol 1e-15))
  "奇异值分解 A = U·S·V^H（复数版）。返回 (values u s vh)：
   full-matrices = nil → u(m×k)、s(k)、vh(k×n)，k = min(m,n)
   full-matrices = t   → u(m×m)、s(k)、vh(n×n)
   奇异值**降序**，s 是实数（float64）。
   实现为单边 Jacobi（§ 本节注释里的复旋转推导）。"
  (let ((v (ensure-vt matrix)))
    (with-float-safe
    (multiple-value-bind (a m n) (%cl-cplx-2d v "vt-svd")
      (multiple-value-bind (u s vh k)
          (%csvd-core a m n :max-sweeps max-sweeps :tol tol)
        (let* ((sv (%cl-vt s (list k) :float64))
               (uvt (%cl-vt u (list m k) :complex128))
               (vvt (%cl-vt vh (list k n) :complex128)))
          (if full-matrices
              (let ((uf (%cl-orthonormal-complete u m k))
                    (vhf (%cl-make-buf (* n n))))
                ;; V^H 补全 = (补全 V)^H：先补 V 的列（= VH 的行取共轭）
                (let ((vfull (%cl-make-buf (* n n))))
                  ;; V 的列 j（j<k）= conj(vh 第 j 行)
                  (dotimes (j k)
                    (dotimes (i n)
                      (setf (aref vfull (%cl-i n i j))
                            (%cconj (aref vh (%cl-i n j i))))))
                  (let ((vcomp (%cl-orthonormal-complete vfull n k)))
                    (dotimes (j n)
                      (dotimes (i n)
                        (setf (aref vhf (%cl-i n j i))
                              (%cconj (aref vcomp (%cl-i n i j))))))))
                (values (%cl-vt uf (list m m) :complex128)
                        sv
                        (%cl-vt vhf (list n n) :complex128)))
              (values uvt sv vvt))))))))


;;; ==================================================================
;;; 7. 厄米特特征值（Jacobi，复数）
;;; ==================================================================
;;; 复厄米特 2×2 块的零化与 SVD 的旋转同源（同样解 t(α-β)+γ-t²conj(γ)=0），
;;; 只是这里 α、β 是**实对角元**、γ 是 a_pq。转置语义上：
;;;   A ← U^H A U，U = [[c, s],[-conj(s), c]]
;;; 列变换用 (c, conj(s))，行变换用 (c, s) 且第二行取 conj(s) ——
;;; 这两处不对称正是 U^H 与 U 的区别。

(defun %ceig-jacobi! (a v n max-iter tol)
  "原地厄米特 Jacobi。A 为 n×n（会被销毁，对角留下特征值），
   V 为 n×n 单位阵（列 = 特征向量）。返回迭代次数。"
  (declare (type (simple-array (complex double-float) (*)) a v)
           (type fixnum n max-iter)
           (type double-float tol)
           (optimize (speed 3) (safety 1)))
  (let ((iter 0))
    (declare (type fixnum iter))
    (dotimes (sweep max-iter)
      (let ((off 0.0d0)
            (rotated nil))
        (declare (type double-float off))
        (dotimes (p (1- n))
          (declare (type fixnum p))
          (loop for q of-type fixnum from (1+ p) below n do
            (let* ((apq (aref a (%cl-i n p q)))
                   (mag (%cl-abs apq)))
              (declare (type (complex double-float) apq) (type double-float mag))
              (incf off (* mag mag))
              (when (> mag 1.0d-300)
                (setf rotated t)
                (let* ((app (realpart (aref a (%cl-i n p p))))
                       (aqq (realpart (aref a (%cl-i n q q))))
                       (u (/ (- app aqq) mag))
                       (sg (if (>= u 0.0d0) 1.0d0 -1.0d0))
                       (w (/ -2.0d0 (+ u (* sg (sqrt (+ (* u u) 4.0d0))))))
                       (c (/ 1.0d0 (sqrt (+ 1.0d0 (* w w)))))
                       (phase (/ apq mag))
                       (s (* phase w c)))
                  (declare (type double-float app aqq u sg w c)
                           (type (complex double-float) phase s))
                  ;; 1) 列变换 A ← A·U：new_p = c·col_p - conj(s)·col_q
                  (dotimes (i n)
                    (declare (type fixnum i))
                    (let ((cp (aref a (%cl-i n i p)))
                          (cq (aref a (%cl-i n i q))))
                      (declare (type (complex double-float) cp cq))
                      (setf (aref a (%cl-i n i p)) (- (* c cp) (* (%cconj s) cq)))
                      (setf (aref a (%cl-i n i q)) (+ (* s cp) (* c cq)))))
                  ;; 2) 行变换 A ← U^H·A：new_p = c·row_p - s·row_q
                  (dotimes (j n)
                    (declare (type fixnum j))
                    (let ((rp (aref a (%cl-i n p j)))
                          (rq (aref a (%cl-i n q j))))
                      (declare (type (complex double-float) rp rq))
                      (setf (aref a (%cl-i n p j)) (- (* c rp) (* s rq)))
                      (setf (aref a (%cl-i n q j)) (+ (* (%cconj s) rp) (* c rq)))))
                  ;; 3) 特征向量累积 V ← V·U（与列变换同形）
                  (dotimes (i n)
                    (declare (type fixnum i))
                    (let ((vp (aref v (%cl-i n i p)))
                          (vq (aref v (%cl-i n i q))))
                      (declare (type (complex double-float) vp vq))
                      (setf (aref v (%cl-i n i p)) (- (* c vp) (* (%cconj s) vq)))
                      (setf (aref v (%cl-i n i q)) (+ (* s vp) (* c vq))))))))))
        (incf iter)
        (when (or (not rotated) (<= (sqrt off) tol))
          (return))))
    iter))

(defun %ceig-core (a n &key (max-iter 200) (tol 1e-14))
  "对 n×n 复厄米特矩阵（会被销毁）做 Jacobi，返回
   (values vals-float64-vector vecs-complex-buffer)。特征值**降序**。"
  (declare (type (simple-array (complex double-float) (*)) a)
           (type fixnum n))
  (let ((v (%cl-eye-buf n)))
    (%ceig-jacobi! a v n max-iter (coerce tol 'double-float))
    ;; 对角为特征值（虚部只有浮点噪声），按降序重排
    (let* ((vals (make-array n :element-type 'double-float :initial-element 0.0d0))
           (order nil))
      (dotimes (i n)
        (setf (aref vals i) (realpart (aref a (%cl-i n i i)))))
      (setf order (%cl-sort-perm vals))
      (let ((sorted (coerce-map-double vals order))
            (vecs (%cl-make-buf (* n n))))
        (dotimes (j n)
          (let ((oj (aref order j)))
            (dotimes (i n)
              (setf (aref vecs (%cl-i n i j)) (aref v (%cl-i n i oj))))))
        (values sorted vecs)))))

(defun %cl-hermitian-mirror! (a n uplo)
  "把指定三角镜像到另一三角（复数版：上三角 = conj(下三角)），并把
   对角元的虚部清零。

   为什么要清对角虚部：厄米特矩阵的对角元必为实数，因此「只取一个三角」
   的语义里对角元的虚部本来就是无意义的浮点噪声（LAPACK 的 zheevd 也是
   直接按实部使用）。输入若真的带复对角元（非厄米输入，GIGO），
   不清零会让 Jacobi 拿 (realpart diag) 当特征值候选、却让复对角元
   参与旋转，最终收敛到**错误**的特征值 —— 实测 [[2+1j,1],[1-1j,3+2j]]
   会给出 (3.925, 1.075) 而不是 numpy 的 (4, 1)。"
  (declare (type (simple-array (complex double-float) (*)) a)
           (type fixnum n))
  (ecase uplo
    (:lower
     (dotimes (i n)
       (loop for j of-type fixnum from (1+ i) below n do
         (setf (aref a (%cl-i n i j)) (%cconj (aref a (%cl-i n j i)))))))
    (:upper
     (dotimes (i n)
       (loop for j of-type fixnum from (1+ i) below n do
         (setf (aref a (%cl-i n j i)) (%cconj (aref a (%cl-i n i j))))))))
  ;; 对角元实化（真厄米输入恒等，非厄米输入与 numpy 的取实部一致）
  (dotimes (i n)
    (setf (aref a (%cl-i n i i))
          (complex (realpart (aref a (%cl-i n i i))) 0.0d0)))
  a)

(defun %cx-op-eig (matrix &key (max-iter 200) (tol 1e-10) (uplo :lower))
  "厄米特矩阵特征值分解（复数版，Jacobi 旋转）。
   与实数 vt-eig 的契约一致（numpy.linalg.eigh 的 UPLO 语义）：不校验
   厄米特性，只使用 UPLO 指定的三角，并**用共轭**镜像到另一三角。
   返回 (values vals vecs)：vals 为 float64（厄米特矩阵的特征值必为实数），
   降序；vecs 为 complex128 n×n，列是特征向量。"
  (let ((v (ensure-vt matrix)))
    (with-float-safe
    (let ((shape (vt-shape v)))
      (unless (and (= 2 (length shape)) (= (first shape) (second shape)))
        (error "vt-eig: 特征值分解要求方阵，收到形状 ~a" shape))
      (multiple-value-bind (a n n2) (%cl-cplx-2d v "vt-eig")
        (declare (ignore n2))
        (%cl-hermitian-mirror! a n uplo)
        (multiple-value-bind (vals vecs)
            (%ceig-core a n :max-iter max-iter :tol tol)
          (values (vt-from-sequence (coerce vals 'list) :dtype :float64)
                  (%cl-vt vecs (list n n) :complex128))))))))


;;; ==================================================================
;;; 8. pinv / lstsq / matrix-rank
;;; ==================================================================
;;; 三者都以 SVD 为底座。复数语义的关键：伪逆是
;;;   A⁺ = V·Σ⁺·U^H
;;; 而不是实数版的 V·Σ⁺·U^T —— 少一个 conj 就得到 A⁺ 的转置共轭，
;;; 数值上「看起来像」，但 A·A⁺ 不再是厄米特投影算子。

(defun %csvd-of (matrix what)
  "取 SVD 的三件套（复数），用于 pinv/lstsq/rank。
   WHAT 只用于调用点自述（错误信息里由调用方自己拼），此处不用。"
  (declare (ignore what))
  (multiple-value-bind (u s vh) (vt-svd matrix)
    (values u s vh)))

(defun %cx-op-pinv (matrix &key (rcond 1e-15))
  "Moore-Penrose 伪逆（复数版）：A⁺ = V·Σ⁺·U^H。"
  (let ((v (ensure-vt matrix)))
    (with-float-safe
    (multiple-value-bind (u s vh) (%csvd-of v "vt-pinv")
      (let* ((k (vt-size s))
             (sd (vt-data s))
             (smax 0.0d0)
             (cutoff 0.0d0))
        (declare (type double-float smax cutoff))
        (dotimes (i k) (setf smax (max smax (aref sd i))))
        (setf cutoff (* rcond smax))
        ;; Σ⁺：k×k 对角
        (let ((sp (%cl-make-buf (* k k))))
          (dotimes (i k)
            (when (> (aref sd i) cutoff)
              (setf (aref sp (%cl-i k i i))
                    (complex (/ 1.0d0 (aref sd i)) 0.0d0))))
          ;; A⁺ = V·Σ⁺·U^H = (vh)^H · sp · u^H
          (let* ((vv (vt-transpose (vt-conj vh)))       ; n×k
                 (spv (%cl-vt sp (list k k) :complex128))
                 (uh (vt-transpose (vt-conj u))))       ; k×m
            (vt-matmul (vt-matmul vv spv) uh))))))))


(defun %cx-op-lstsq (a b &key (rcond 1e-15))
  "最小二乘 min‖Ax-b‖₂（复数版）。返回 (values x residual rank s)。
   residual：仅当 m>n 且满列秩时给出；否则 0.0d0（与实数版一致）。"
  (let ((av (ensure-vt a))
        (bv (ensure-vt b)))
    (with-float-safe
    (let* ((shape (vt-shape av)))
      (unless (= 2 (length shape))
        (error "vt-lstsq: 系数矩阵必须是 2 维，收到形状 ~a" shape))
      (let* ((m (first shape))
             (n (second shape))
             (b-shape (vt-shape bv))
             (nrhs (if (or (null b-shape) (= 1 (length b-shape)))
                       1
                       (second b-shape)))
             (bmat (if (or (null b-shape) (= 1 (length b-shape)))
                       (vt-reshape (vt-astype (vt-copy bv) :complex128) (list m 1))
                       (vt-astype (vt-copy bv) :complex128))))
        (multiple-value-bind (u s vh) (%csvd-of av "vt-lstsq")
          (let* ((k (vt-size s))
                 (sd (vt-data s))
                 (smax 0.0d0)
                 (cutoff 0.0d0)
                 (rank 0))
            (declare (type fixnum rank) (type double-float smax cutoff))
            (dotimes (i k) (setf smax (max smax (aref sd i))))
            (setf cutoff (* rcond smax))
            (dotimes (i k) (when (> (aref sd i) cutoff) (incf rank)))
            ;; x = V·Σ⁺·U^H·b
            (let* ((sp (%cl-make-buf (* k k)))
                   (uhb (vt-matmul (vt-transpose (vt-conj u)) bmat))  ; k×nrhs
                   (scaled (%cl-make-buf (* k nrhs))))
              (declare (type (simple-array (complex double-float) (*)) scaled))
              (dotimes (i k)
                (when (> (aref sd i) cutoff)
                  (setf (aref sp (%cl-i k i i))
                        (complex (/ 1.0d0 (aref sd i)) 0.0d0))))
              (dotimes (i k)
                (dotimes (j nrhs)
                  (setf (aref scaled (%cl-i nrhs i j))
                        (* (aref (vt-data uhb) (%cl-i nrhs i j))
                           (aref sp (%cl-i k i i))))))
              (let* ((scaled-vt (%cl-vt scaled (list k nrhs) :complex128))
                     (vv (vt-transpose (vt-conj vh)))
                     (x (vt-matmul vv scaled-vt)))
                ;; residual 与实数版契约一致
                (let ((res (if (and (> m n) (= rank n))
                               (float (vt-item (vt-norm
                                                (vt-flatten
                                                 (vt-- bmat (vt-matmul av x)))))
                                      1.0d0)
                               0.0d0))
                      (xr (if (= nrhs 1)
                              (vt-flatten (vt-slice x '(:all) '(0)))
                              x)))
                  (values xr res rank s)))))))))))


(defun %cx-op-matrix-rank (v tol)
  "复数矩阵秩（V 已 ENSURE-VT）：基于奇异值。
   阈值 TOL 显式给出时用之，否则 s_max·max(m,n)·eps（numpy 的缺省口径）。

   本函数只做复数分支；公开入口在 linalg.lisp 的单一 DEFUN 里分流（§16.32）。"
  (with-float-safe
    (let ((shape (vt-shape v)))
      (unless (= 2 (length shape))
        (error "vt-matrix-rank: 要求 2 维矩阵，收到形状 ~a" shape))
      (let* ((m (first shape))
             (n (second shape)))
        (multiple-value-bind (u s vh) (%csvd-of v "vt-matrix-rank")
          (declare (ignore u vh))
          (let* ((k (vt-size s))
                 (sd (vt-data s))
                 (smax 0.0d0)
                 (thr 0.0d0)
                 (rank 0))
            (declare (type fixnum rank) (type double-float smax thr))
            (dotimes (i k) (setf smax (max smax (aref sd i))))
            (setf thr (if tol
                          (coerce tol 'double-float)
                          (* smax (max m n) double-float-epsilon)))
            (dotimes (i k) (when (> (aref sd i) thr) (incf rank)))
            rank))))))
