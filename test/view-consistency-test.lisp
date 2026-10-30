;;;; view-consistency-test.lisp — 物理形态无关性 + 分解的纯函数性 + lstsq 契约
;;;;
;;;; 前面那些套件的尺子都是「随机输入 + 跨实现比对」，前提是**函数只被调用
;;;; 一次**。这个前提漏掉两件事：
;;;;
;;;;   1. 同一个逻辑张量**换一种物理形态**（转置、负步长、offset、广播、
;;;;      别名）之后，还是不是同一个张量？
;;;;   2. 同一个输入**调用两次**，给不给同一个答案？会不会顺手推走全局随机流？
;;;;
;;;; 本套件把这两件事变成断言，并顺带钉住两条与 numpy 对齐的契约：
;;;; 复数 SVD 的 U 必须酉、VT-LSTSQ 的形状与残差（CONVENTIONS §17.10.22）。
;;;;
;;;;   A. 视图一致性：f(view) 与 f(contiguous(view)) 必须**逐位**相等
;;;;      （不是 allclose —— 差一个 ULP 就说明某条路径读了未初始化内存，
;;;;      或走了不同的求和次序：SIMD 分部分和 vs 标量累加）。
;;;;      10 类视图 × 函数族 × 3 种 dtype；另外断言 f 不得改动入参。
;;;;
;;;;   B. `:out` 指向视图：写回必须按 OUT 自己的 stride 走，不能假定连续；
;;;;      OUT 与输入共享 buffer 时（读写别名）必须给正确结果，不能自我覆盖。
;;;;
;;;;   C. 分解的纯函数性：同输入两次 SVD 逐位相等（实数、:FULL-MATRICES、
;;;;      复数三条线），且调用不改变 VT-RANDOM 的后续序列。修复前用
;;;;      VT-RANDOM 补零奇异值列，于是秩亏 2×2 的 U 第二列两次方向相反。
;;;;
;;;;   D. 复数 SVD 的 U 必须酉：零奇异值列原来直接填 (0+0i)，UᴴU 的对角
;;;;      出现 0。零向量不是单位向量，必须做正交补全。
;;;;
;;;;   E. VT-LSTSQ 契约：x 形状跟 b（(m,1) → (n,1)，不压平）；residual 是
;;;;      **平方和**（numpy 口径，不是 2-范数）；无残差时是形状 (0,) 的空
;;;;      数组而不是标量 0。复数版同一套。
;;;;
;;;;   F. EIG / CHOLESKY 只读一个三角：这不是缺陷，是 LAPACK 的语义
;;;;      （numpy eigh 用 UPLO='L'，zheev 还忽略对角虚部）。这里把它钉成
;;;;      断言，防止有人「顺手修好」成读全矩阵。
;;;;
;;;; 期望值一律由独立路径重算（重构误差、UᵀU−I、‖Ax−b‖² 直接算），
;;;; 不手写魔数（CONVENTIONS §10.4）。
;;;; 用法：sbcl --non-interactive --load test/view-consistency-test.lisp

(require :asdf)
(defparameter *clvt-test-dir* (make-pathname :name nil :type nil :defaults *load-truename*))
(pushnew (merge-pathnames "../" *clvt-test-dir*) asdf:*central-registry* :test #'equal)
(handler-bind ((warning #'muffle-warning)) (asdf:load-system :clvt))

(in-package :clvt)

;;; ------------------------------------------------------------------
;;; 骨架
;;; ------------------------------------------------------------------

(defvar *pass* 0)
(defvar *fail* 0)
(defvar *failures* '())

(defun report (name ok &optional detail)
  (if ok
      (incf *pass*)
      (progn (incf *fail*) (push name *failures*)
             (format t "~&FAIL ~a ~@[— ~a~]" name detail))))

(defun section (title) (format t "~&~%--- ~a ---~%" title))

;;; ------------------------------------------------------------------
;;; 逐位比较：把结果拍成「dtype + shape + 每个元素的位模式」的令牌
;;; ------------------------------------------------------------------
;;; 为什么不用 VT-ARRAY-EQUIV / 容差比较：视图不变性的判据是「同一条计算
;;; 路径」，逐位相等才说明物理形态真的没有泄漏进数值。容差会把「求和次序
;;; 变了」这种真实的选路差异放过去。

(defun bits-of (x)
  (typecase x
    (double-float (list :d (sb-kernel:double-float-bits x)))
    (single-float (list :s (sb-kernel:single-float-bits x)))
    ((complex double-float)
     (list :cd (sb-kernel:double-float-bits (realpart x))
              (sb-kernel:double-float-bits (imagpart x))))
    ((complex single-float)
     (list :cs (sb-kernel:single-float-bits (realpart x))
              (sb-kernel:single-float-bits (imagpart x))))
    (integer x)
    (boolean (if x :t :f))
    (t x)))

(defun token (x)
  "把任意返回值拍成可 EQUAL 的令牌（张量 / 列表 / 标量递归处理）。"
  (cond ((vt-p x)
         (list :vt (vt-dtype x) (copy-list (vt-shape x))
               (map 'list #'bits-of (vt-to-list (vt-ravel x)))))
        ((consp x) (cons :list (map 'list #'token x)))
        ((vectorp x) (cons :vector (map 'list #'token x)))
        (t (list :scalar (type-of x) (bits-of x)))))

(defun call-token (thunk)
  "调用 THUNK；成功给 (:ok 令牌)，报错给 (:err 条件类型)。"
  (handler-case (list :ok (token (funcall thunk)))
    (error (e) (list :err (type-of e)))))

;;; ------------------------------------------------------------------
;;; A. 视图一致性
;;; ------------------------------------------------------------------

(defvar *n-view* 0)

(defun base-tensor (dtype)
  "4×5（复数 3×4）的取值确定的基张量：值互不相同且都不为 0，
   这样「读错位置」和「读成 0」都能被逐位比较抓到。"
  (if (eq dtype :complex128)
      (let ((a (vt-zeros '(3 4) :dtype :complex128)))
        (dotimes (i 3 a)
          (dotimes (j 4)
            (setf (vt-ref a i j)
                  (complex (coerce (+ 1 j) 'double-float)
                           (coerce (- i) 'double-float))))))
      (let ((a (vt-zeros '(4 5) :dtype dtype)))
        (dotimes (i 4 a)
          (dotimes (j 5)
            (setf (vt-ref a i j)
                  (if (eq dtype :int64)
                      (+ 1 (* i 5) j)
                      (coerce (+ 1 (* i 5) j) 'double-float))))))))

(defun views-of (a)
  "10 类物理形态：连续 / 转置 / 翻转 / 两个方向的翻转 / 子块 / 负步长 /
   广播 / 转置后切片 / 单列 / 单行反向 / 子块再转置。"
  (list (cons "contig" a)
        (cons "transpose" (vt-transpose a))
        (cons "flip0" (vt-flip a :axis 0))
        (cons "flip-all" (vt-flip a))
        (cons "sub-block" (vt-slice a '(1 3) '(1 4)))
        (cons "neg-step" (vt-slice a '(nil nil -1) '(nil nil -1)))
        (cons "broadcast" (vt-broadcast-to (vt-slice a '(0 1)) (vt-shape a)))
        (cons "transposed-slice" (vt-slice (vt-transpose a) '(1 3)))
        (cons "single-col" (vt-slice a '(:all) '(2)))
        (cons "row-reversed" (vt-slice a '(0) '(nil nil -1)))
        (cons "sub-then-transpose" (vt-transpose (vt-slice a '(1 3) '(0 4))))))

(defmacro fn1 (name form) `(cons ,name (lambda (v) ,form)))

(defun fns-common ()
  "对任何 dtype、任何秩都成立的函数（含归约、变换、集合、IO 之外的算子）。"
  (list
   (fn1 "abs" (vt-abs v))
   (fn1 "square" (vt-square v))
   (fn1 "negative" (vt-negative v))
   (fn1 "add-1" (vt-add v 1))
   (fn1 "sub-0.5" (vt-sub v 0.5))
   (fn1 "mul-2" (vt-mul v 2))
   (fn1 "div-2" (vt-div v 2))
   (fn1 "gt-1" (vt-> v 1))
   (fn1 "eq-1" (vt-= v 1))
   (fn1 "sum" (vt-sum v))
   (fn1 "sum-axis0" (vt-sum v :axis 0))
   (fn1 "sum-axis-last" (vt-sum v :axis -1))
   (fn1 "sum-keepdims" (vt-sum v :axis 0 :keepdims t))
   (fn1 "prod" (vt-prod v))
   (fn1 "amax" (vt-amax v))
   (fn1 "amin-axis0" (vt-amin v :axis 0))
   (fn1 "mean" (vt-mean v))
   (fn1 "mean-axis0" (vt-mean v :axis 0))
   (fn1 "var" (vt-var v))
   (fn1 "std-axis0" (vt-std v :axis 0))
   (fn1 "all" (vt-all v))
   (fn1 "any" (vt-any v))
   (fn1 "argmax" (vt-argmax v))
   (fn1 "argmin-axis0" (vt-argmin v :axis 0))
   (fn1 "cumsum" (vt-cumsum v))
   (fn1 "cumprod" (vt-cumprod v))
   (fn1 "sort" (vt-sort v))
   (fn1 "argsort" (vt-argsort v))
   (fn1 "unique" (vt-unique v))
   (fn1 "count-nonzero" (vt-count-nonzero v))
   (fn1 "flip" (vt-flip v))
   (fn1 "ravel" (vt-ravel v))
   (fn1 "flatten" (vt-flatten v))
   (fn1 "repeat" (vt-repeat v 2))
   (fn1 "tile" (vt-tile v '(2 1)))
   (fn1 "pad" (vt-pad v 1))
   (fn1 "roll" (vt-roll v 1))
   (fn1 "concat0" (vt-concat 0 v v))
   (fn1 "stack0" (vt-stack 0 v v))
   (fn1 "unique-counts" (vt-unique v :return-counts t))
   (fn1 "sort-axis0" (vt-sort v :axis 0))
   (fn1 "argsort-axis0" (vt-argsort v :axis 0))
   (fn1 "transpose" (vt-transpose v))
   (fn1 "reshape" (vt-reshape v (list (vt-size v))))
   (fn1 "concat-last" (vt-concat (if (= 2 (vt-order v)) 1 0) v v))
   (fn1 "stack-last" (vt-stack (if (= 2 (vt-order v)) 1 0) v v))))

(defun fns-2d ()
  (list
   (fn1 "sum-axis1" (vt-sum v :axis 1))
   (fn1 "cumsum-axis1" (vt-cumsum v :axis 1))
   (fn1 "cumprod-axis0" (vt-cumprod v :axis 0))
   (fn1 "matmul-with-transpose" (vt-matmul v (vt-transpose v)))
   (fn1 "diagonal" (vt-diagonal v))
   (fn1 "cov" (vt-cov v))
   (fn1 "corrcoef" (vt-corrcoef v))))

(defun fns-1d ()
  (list
   (fn1 "cumsum-axis0" (vt-cumsum v :axis 0))
   (fn1 "outer" (vt-outer v v))
   (fn1 "inner-1d" (vt-inner v v))
   (fn1 "dot-1d" (vt-dot v v))))

(defun check-view (vname view fname fn)
  "f(连续副本) 与 f(视图) 逐位相等，且 f 不改动视图。"
  (incf *n-view*)
  (let* ((copy (vt-contiguous view))
         (before (token view))
         (a (call-token (lambda () (funcall fn copy))))
         (b (call-token (lambda () (funcall fn view))))
         (after (token view)))
    (report (format nil "view/~a/~a" vname fname) (equal a b)
            (format nil "copy=~s view=~s"
                    (let ((s (format nil "~s" a))) (subseq s 0 (min 90 (length s))))
                    (let ((s (format nil "~s" b))) (subseq s 0 (min 90 (length s))))))
    (report (format nil "view-immutable/~a/~a" vname fname) (equal before after)
            "视图被算子改写")))

(defun run-view-consistency ()
  (section "A 视图一致性（f(view) 逐位 == f(contiguous(view))）")
  (dolist (dtype '(:float64 :int64 :complex128))
    (let ((a (base-tensor dtype)))
      (dolist (vw (views-of a))
        (let ((view (cdr vw)))
          (dolist (f (append (fns-common)
                             (when (= 2 (vt-order view)) (fns-2d))
                             (when (= 1 (vt-order view)) (fns-1d))))
            (check-view (car vw) view (car f) (cdr f)))))))
  ;; 3-D 的全排列：置换只改 strides，六个排列必须给出同一组归约结果
  (let* ((a (vt-zeros '(2 3 4) :dtype :float64))
         (perms '((0 1 2) (0 2 1) (1 0 2) (1 2 0) (2 0 1) (2 1 0))))
    (dotimes (i 24) (setf (vt-ref a (floor i 12) (mod (floor i 4) 3) (mod i 4))
                         (coerce (1+ i) 'double-float)))
    (let* ((contig (vt-contiguous a))
           (want-sum (vt-item (vt-sum contig)))
           (want-ax2 (vt-sum contig :axis 2)))
      (dolist (p perms)
        (let* ((v (vt-transpose a p))
               (expect-sum (vt-item (vt-sum (vt-transpose contig p))))
               ;; 轴 2 在置换后是第 (position 2 p) 轴；归约后剩下的轴按 P 中
               ;; 的先后排列 —— 所以参照值要按 (remove 2 p) 再转置一次才能比。
               (ax (position 2 p))
               (rem (remove 2 p)))
          (incf *n-view*)
          (report (format nil "permute/~s/sum-bitexact" p)
                  (eql want-sum expect-sum)
                  (format nil "~s vs ~s" want-sum expect-sum))
          (incf *n-view*)
          (report (format nil "permute/~s/sum-axis2" p)
                  (equal (token (vt-sum v :axis ax))
                         (token (vt-transpose want-ax2 rem)))
                  (format nil "得到 ~s，期望 ~s"
                          (vt-shape (vt-sum v :axis ax))
                          (vt-shape (vt-transpose want-ax2 rem)))))))))

;;; ------------------------------------------------------------------
;;; B. `:out` 指向视图 / 别名
;;; ------------------------------------------------------------------

(defun run-out-view ()
  (section "B :out 指向视图（写穿透 strides）与读写别名")
  (let* ((src (base-tensor :float64))
         (want (vt-add src 1))
         (buf (vt-zeros '(5 4) :dtype :float64))
         (out (vt-transpose buf)))          ; strides (1 5)，非连续
    (vt-add src 1 :out out)
    (report "out/transposed-view" (equal (token out) (token want))
            "写回没有按 out 的 stride 走"))
  ;; 负步长的 out：写回必须穿过 stride。判据分两层 ——
  ;; ① out 自身按逻辑顺序等于期望值；② 底层 buf 上确实呈现镜像布局
  ;; （若实现偷偷把结果写成连续布局再 memcpy，② 会露馅）。
  (let* ((src (base-tensor :float64))
         (want (vt-mul src 2))
         (buf (vt-zeros '(4 5) :dtype :float64))
         (out (vt-slice buf '(nil nil -1))))     ; 只反转行
    (vt-mul src 2 :out out)
    (report "out/negative-stride-view" (equal (token out) (token want))
            "负步长写回顺序错了")
    (report "out/negative-stride-layout"
            (equal (token buf) (token (vt-flip want :axis 0)))
            "底层 buffer 没有呈现镜像布局 → 写回没穿过 stride"))
  (let* ((src (base-tensor :float64))
         (want (vt-mul src 3))
         (buf (vt-zeros '(4 5) :dtype :float64))
         (out (vt-slice buf '(:all) '(nil nil -1))))  ; 只反转列
    (vt-mul src 3 :out out)
    (report "out/column-reversed-view" (equal (token out) (token want))
            "列反转写回顺序错了")
    (report "out/column-reversed-layout"
            (equal (token buf) (token (vt-slice want '(:all) '(nil nil -1))))
            "底层 buffer 的列没有镜像"))
  ;; 读写别名：out 与唯一输入是同一个缓冲区（含转置形态）
  (let* ((a (base-tensor :float64))
         (want (vt-add (base-tensor :float64) (base-tensor :float64))))
    (vt-add a a :out a)
    (report "alias/self-add" (equal (token a) (token want))
            "别名自加被按元素覆盖"))
  (let* ((base (base-tensor :float64))
         (want (vt-add (vt-transpose (base-tensor :float64))
                       (vt-transpose (base-tensor :float64))))
         (v (vt-transpose base)))
    (vt-add v v :out v)
    (report "alias/transposed-self-add" (equal (token v) (token want))
            "转置视图上的别名自加被覆盖")))

;;; ------------------------------------------------------------------
;;; C. 分解的纯函数性
;;; ------------------------------------------------------------------

(defun svd-triple (a) (multiple-value-list (vt-svd a)))

(defun run-svd-determinism ()
  (section "C 分解是输入的纯函数（同输入两次同值 + 不消费全局随机流）")
  (flet ((same-twice (name a &key full)
           (let ((t1 (multiple-value-list (vt-svd a :full-matrices full)))
                 (t2 (multiple-value-list (vt-svd a :full-matrices full))))
             (report (format nil "purity/~a" name)
                     (and (equal (token (first t1)) (token (first t2)))
                          (equal (token (second t1)) (token (second t2)))
                          (equal (token (third t1)) (token (third t2))))
                     "两次 SVD 结果不逐位相等"))))
    (same-twice "diag3" (vt-zeros '(3 3) :dtype :float64))
    (same-twice "rankdef2x2"
                (vt-from-sequence '((1.0d0 2.0d0) (2.0d0 4.0d0)) :dtype :float64))
    (same-twice "rankdef3"
                (vt-from-sequence '((1.0d0 2.0d0 3.0d0)
                                    (2.0d0 4.0d0 6.0d0)
                                    (3.0d0 6.0d0 9.0d0)) :dtype :float64))
    (same-twice "zero2x3" (vt-zeros '(2 3) :dtype :float64))
    (same-twice "tall4x2-full"
                (vt-from-sequence '((1.0d0 2.0d0) (3.0d0 4.0d0)
                                    (5.0d0 6.0d0) (7.0d0 8.0d0))
                                  :dtype :float64)
                :full t)
    (same-twice "complex-rank1"
                (vt-from-sequence (list (list (complex 1.0d0 0.0d0) (complex 2.0d0 0.0d0))
                                        (list (complex 2.0d0 0.0d0) (complex 4.0d0 0.0d0)))
                                  :dtype :complex128))
    (same-twice "complex-zero2x2" (vt-zeros '(2 2) :dtype :complex128)))
  ;; 不消费全局随机流：同一个种子，中间夹一次 SVD，后续序列必须不变
  (flet ((draw () (progn (vt-random-seed 987654321) (vt-to-list (vt-random '(5))))))
    (let ((before (draw)))
      (vt-random-seed 987654321)
      (vt-svd (vt-from-sequence '((1.0d0 2.0d0 3.0d0)
                                 (2.0d0 4.0d0 6.0d0)
                                 (3.0d0 6.0d0 9.0d0)) :dtype :float64))
      (vt-svd (vt-from-sequence '((1.0d0 2.0d0) (3.0d0 4.0d0)
                                  (5.0d0 6.0d0) (7.0d0 8.0d0))
                                :dtype :float64)
              :full-matrices t)
      (report "purity/svd-does-not-consume-rng"
              (equal before (vt-to-list (vt-random '(5))))
              "vt-svd 推走了全局随机流"))))

;;; ------------------------------------------------------------------
;;; D. 复数 SVD 的 U 必须酉
;;; ------------------------------------------------------------------

(defun ctranspose (x)
  "共轭转置 Xᴴ。本库没有单独的 adjoint 入口：就是 conjugate ∘ transpose。"
  (vt-conjugate (vt-transpose x)))

(defun unite-err (u &key (conjugate nil))
  "max|MᴴM − I|；CONJUGATE 为 NIL 时按实数口径 max|MᵀM − I|。"
  (let* ((uh (if conjugate (ctranspose u) (vt-transpose u)))
         (uhu (vt-@ uh u))
         (k (second (vt-shape u)))
         (d (vt-abs (vt-- uhu (vt-eye k :dtype :float64)))))
    (if (zerop (vt-size d)) 0.0d0 (vt-item (vt-amax d)))))

(defun vorth-err (m)
  "max|M·Mᴴ − I|（右酉性；对 V 用）。"
  (let* ((mh (ctranspose m))
         (mmh (vt-@ m mh))
         (k (first (vt-shape m)))
         (d (vt-abs (vt-- mmh (vt-eye k :dtype :float64)))))
    (if (zerop (vt-size d)) 0.0d0 (vt-item (vt-amax d)))))

(defun run-complex-unitary ()
  (section "D 复数 SVD 的 U 必须酉（零奇异值列要正交补全）")
  (flet ((ok-unitary (name a &key full tol)
           (multiple-value-bind (u s vt) (vt-svd a :full-matrices full)
             ;; :FULL-MATRICES 时 U 是 m×m、Vᴴ 是 n×n，而 Σ 只有 min(m,n) 阶 ——
             ;; 重构必须先裁到前 K 列/行（numpy 的 A = U[:, :k] @ diag(s) @ Vh[:k]）。
             (let* ((k (vt-size s))
                    (uk (if (= (second (vt-shape u)) k) u
                            (vt-slice u '(:all) (list 0 k))))
                    (vtk (if (= (first (vt-shape vt)) k) vt
                             (vt-slice vt (list 0 k))))
                    (eu (unite-err u :conjugate t))
                    (ev (vorth-err vt))
                    (rec (vt-@ uk (vt-@ (vt-diag s) vtk)))
                    (er (vt-item (vt-amax (vt-abs (vt-- rec a))))))
               (report (format nil "cx-svd/~a/UH-U=I" name) (< eu (or tol 1d-13))
                       (format nil "max|UᴴU−I| = ~,3e" eu))
               (report (format nil "cx-svd/~a/V-VH=I" name) (< ev (or tol 1d-13))
                       (format nil "max|VVᴴ−I| = ~,3e" ev))
               (report (format nil "cx-svd/~a/recon" name) (< er (or tol 1d-12))
                       (format nil "max|UΣVᴴ−A| = ~,3e" er))))))
    ;; 秩 1：σ = (5, 0)，第二列必须补成单位向量
    (ok-unitary "rank1-2x2"
                (vt-from-sequence (list (list (complex 1.0d0 0.0d0) (complex 2.0d0 0.0d0))
                                        (list (complex 2.0d0 0.0d0) (complex 4.0d0 0.0d0)))
                                  :dtype :complex128))
    ;; 零矩阵：两列都要补
    (ok-unitary "zero2x2" (vt-zeros '(2 2) :dtype :complex128))
    (ok-unitary "zero3x3" (vt-zeros '(3 3) :dtype :complex128))
    ;; 带虚部的秩 1
    (ok-unitary "rank1-3x3-imag"
                (vt-from-sequence (list (list (complex 1.0d0 1.0d0) (complex 2.0d0 0.0d0) (complex 0.0d0 2.0d0))
                                        (list (complex 2.0d0 2.0d0) (complex 4.0d0 0.0d0) (complex 0.0d0 4.0d0))
                                        (list (complex 0.0d0 0.0d0) (complex 0.0d0 0.0d0) (complex 0.0d0 0.0d0)))
                                  :dtype :complex128))
    ;; 满秩复数矩阵：补全不该被触发，结果必须仍与旧行为一致（酉 + 重构）
    (ok-unitary "full3x3"
                (vt-from-sequence (list (list (complex 1.0d0 1.0d0) (complex 2.0d0 0.0d0) (complex 0.0d0 1.0d0))
                                        (list (complex 0.0d0 1.0d0) (complex 3.0d0 0.0d0) (complex 1.0d0 0.0d0))
                                        (list (complex 1.0d0 0.0d0) (complex 0.0d0 2.0d0) (complex 2.0d0 0.0d0)))
                                  :dtype :complex128))
    ;; 高矩阵 :FULL-MATRICES：U 补到 m×m，仍须酉
    (ok-unitary "tall4x2-full"
                (vt-from-sequence (list (list (complex 1.0d0 0.0d0) (complex 2.0d0 1.0d0))
                                        (list (complex 3.0d0 0.0d0) (complex 4.0d0 0.0d0))
                                        (list (complex 5.0d0 1.0d0) (complex 6.0d0 0.0d0))
                                        (list (complex 7.0d0 0.0d0) (complex 8.0d0 0.0d0)))
                                  :dtype :complex128)
                :full t)))

;;; ------------------------------------------------------------------
;;; E. VT-LSTSQ 契约
;;; ------------------------------------------------------------------

(defun residual-sum-of-squares (a x b-mat)
  "独立重算 ‖Ax − b‖²（逐列），不用 lstsq 内部任何中间量。"
  (let* ((r (vt-- (vt-@ a x) b-mat))
         (sq (vt-square (vt-abs r))))
    (vt-sum sq :axis 0 :dtype :float64)))

(defun run-lstsq-contract ()
  (section "E VT-LSTSQ 的形状与残差契约")
  (let* ((a (vt-from-sequence '((1.0d0 0.0d0 0.0d0)
                                (0.0d0 1.0d0 0.0d0)
                                (0.0d0 0.0d0 1.0d0)
                                (1.0d0 1.0d0 1.0d0))
                              :dtype :float64))
         (b2 (vt-from-sequence '((1.0d0) (0.0d0) (2.0d0) (1.0d0)) :dtype :float64))
         (b1 (vt-from-sequence '(1.0d0 0.0d0 2.0d0 1.0d0) :dtype :float64)))
    ;; b 给 (4,1)：x 必须是 (3,1)
    (multiple-value-bind (x res rank s) (vt-lstsq a b2)
      (declare (ignore s))
      (report "lstsq/b-2d-x-shape" (equal (vt-shape x) '(3 1))
              (format nil "x 形状 ~s，期望 (3 1)" (vt-shape x)))
      (report "lstsq/b-2d-rank" (= rank 3) (format nil "rank=~a" rank))
      (report "lstsq/b-2d-residual-shape" (equal (vt-shape res) '(1))
              (format nil "residual 形状 ~s，期望 (1)" (vt-shape res)))
      ;; residual 是平方和：与独立重算的 ‖Ax−b‖² 比对
      (let* ((want (residual-sum-of-squares a x b2))
             (got (vt-item (vt-ravel res)))
             (w (vt-item want)))
        (report "lstsq/b-2d-residual-is-sum-of-squares"
                (< (abs (- got w)) (* 1d-12 (max 1d0 (abs w))))
                (format nil "residual=~,12e，‖Ax−b‖²=~,12e" got w)))
      ;; 再判一次「不是 2-范数」：上面那个 b 的 ‖Ax−b‖ 恰好 ≈ 1.0，平方后区分
      ;; 不出开方与否；这里用一个离群点大的右端让两个口径差 5 个数量级。
      (let ((b-far (vt-from-sequence '((1.0d0) (0.0d0) (2.0d0) (1000.0d0))
                                     :dtype :float64)))
        (multiple-value-bind (x res) (vt-lstsq a b-far)
          (let* ((w (vt-item (residual-sum-of-squares a x b-far)))
                 (got (vt-item (vt-ravel res))))
            (report "lstsq/b-2d-residual-not-2-norm"
                    (and (< (abs (- got w)) (* 1d-12 (max 1d0 (abs w))))
                         (> (abs (- got (sqrt w))) (* 1d-3 (max 1d0 (abs w)))))
                    (format nil "residual=~,12e，‖Ax−b‖²=~,12e，2-范数=~,12e"
                            got w (sqrt w)))))))
    ;; b 给 (4,)：x 必须是 (3,)
    (multiple-value-bind (x res rank s) (vt-lstsq a b1)
      (declare (ignore s))
      (report "lstsq/b-1d-x-shape" (equal (vt-shape x) '(3))
              (format nil "x 形状 ~s，期望 (3)" (vt-shape x)))
      (report "lstsq/b-1d-rank" (= rank 3))
      (report "lstsq/b-1d-residual-shape" (equal (vt-shape res) '(1))))
    ;; 多右端 (4,2)：x 是 (3,2)，residual 是 (2,)
    (let ((b-multi (vt-from-sequence '((1.0d0 0.0d0) (0.0d0 1.0d0)
                                       (2.0d0 1.0d0) (1.0d0 3.0d0))
                                     :dtype :float64)))
      (multiple-value-bind (x res rank s) (vt-lstsq a b-multi)
        (declare (ignore s))
        (report "lstsq/multi-x-shape" (equal (vt-shape x) '(3 2))
                (format nil "x 形状 ~s，期望 (3 2)" (vt-shape x)))
        (report "lstsq/multi-residual-shape" (equal (vt-shape res) '(2))
                (format nil "residual 形状 ~s，期望 (2)" (vt-shape res)))
        (report "lstsq/multi-rank" (= rank 3))
        (let ((want (residual-sum-of-squares a x b-multi)))
          (dotimes (j 2)
            (report (format nil "lstsq/multi-residual-col~a" j)
                    (< (abs (- (vt-item (vt-ref res j))
                               (vt-item (vt-ref want j))))
                       (* 1d-12 (max 1d0 (abs (vt-item (vt-ref want j))))))))))))
  ;; m < n（欠定）：无残差，x 是 (n,)，residual 是 (0,)
  (let* ((a (vt-from-sequence '((1.0d0 2.0d0 3.0d0 4.0d0 5.0d0)
                                (0.0d0 1.0d0 0.0d0 1.0d0 0.0d0)
                                (1.0d0 0.0d0 1.0d0 0.0d0 1.0d0))
                              :dtype :float64))
         (b (vt-from-sequence '(1.0d0 2.0d0 3.0d0) :dtype :float64)))
    (multiple-value-bind (x res rank s) (vt-lstsq a b)
      (declare (ignore s))
      (report "lstsq/underdetermined-x-shape" (equal (vt-shape x) '(5)))
      (report "lstsq/underdetermined-residual-empty"
              (equal (vt-shape res) '(0))
              (format nil "residual 形状 ~s，期望 (0)" (vt-shape res)))
      (report "lstsq/underdetermined-rank" (= rank 3))))
  ;; 秩亏方阵：rank < n，numpy 同样给空残差
  (let* ((a (vt-from-sequence '((1.0d0 2.0d0 3.0d0)
                                (2.0d0 4.0d0 6.0d0)
                                (0.0d0 0.0d0 1.0d0))
                              :dtype :float64))
         (b (vt-from-sequence '(1.0d0 2.0d0 3.0d0) :dtype :float64)))
    (multiple-value-bind (x res rank s) (vt-lstsq a b)
      (declare (ignore x s))
      (report "lstsq/rank-deficient-rank" (= rank 2) (format nil "rank=~a" rank))
      (report "lstsq/rank-deficient-residual-empty" (equal (vt-shape res) '(0)))))
  ;; 复数版同一套
  (let* ((a (vt-from-sequence (list (list (complex 1.0d0 0.0d0) (complex 0.0d0 0.0d0))
                                    (list (complex 0.0d0 0.0d0) (complex 1.0d0 0.0d0))
                                    (list (complex 1.0d0 1.0d0) (complex 1.0d0 0.0d0))
                                    (list (complex 2.0d0 0.0d0) (complex 0.0d0 1.0d0)))
                              :dtype :complex128))
         (b2 (vt-from-sequence (list (list (complex 1.0d0 0.0d0))
                                     (list (complex 0.0d0 0.0d0))
                                     (list (complex 2.0d0 1.0d0))
                                     (list (complex 1.0d0 0.0d0)))
                               :dtype :complex128)))
    (multiple-value-bind (x res rank s) (vt-lstsq a b2)
      (declare (ignore s))
      (report "cx-lstsq/b-2d-x-shape" (equal (vt-shape x) '(2 1))
              (format nil "x 形状 ~s，期望 (2 1)" (vt-shape x)))
      (report "cx-lstsq/b-2d-residual-shape" (equal (vt-shape res) '(1)))
      (report "cx-lstsq/b-2d-rank" (= rank 2))
      (let* ((want (vt-item (residual-sum-of-squares a x b2)))
             (got (vt-item (vt-ravel res))))
        (report "cx-lstsq/b-2d-residual-is-sum-of-squares"
                (< (abs (- got want)) (* 1d-12 (max 1d0 (abs want))))
                (format nil "residual=~,12e，‖Ax−b‖²=~,12e" got want))))
    (multiple-value-bind (x res rank s)
        (vt-lstsq a (vt-flatten b2))
      (declare (ignore s))
      (report "cx-lstsq/b-1d-x-shape" (equal (vt-shape x) '(2))
              (format nil "x 形状 ~s，期望 (2)" (vt-shape x)))
      (report "cx-lstsq/b-1d-rank" (= rank 2))
      (report "cx-lstsq/b-1d-residual-shape" (equal (vt-shape res) '(1))))))

;;; ------------------------------------------------------------------
;;; F. EIG / CHOLESKY 只读一个三角（不是缺陷，钉住它）
;;; ------------------------------------------------------------------

(defun run-triangle-semantics ()
  (section "F EIG / CHOLESKY 的「只读一个三角」（LAPACK 语义）")
  (let* ((base (vt-from-sequence '((4.0d0 1.0d0 2.0d0)
                                   (1.0d0 3.0d0 0.0d0)
                                   (2.0d0 0.0d0 5.0d0))
                                 :dtype :float64))
         (lower (vt-from-sequence '((4.0d0 0.0d0 0.0d0)
                                    (1.0d0 3.0d0 0.0d0)
                                    (2.0d0 0.0d0 5.0d0))
                                  :dtype :float64))
         (upper (vt-from-sequence '((4.0d0 1.0d0 2.0d0)
                                    (0.0d0 3.0d0 0.0d0)
                                    (0.0d0 0.0d0 5.0d0))
                                  :dtype :float64)))
    (multiple-value-bind (wl vl) (vt-eig base)
      (declare (ignore vl))
      (multiple-value-bind (wl2 vl2) (vt-eig lower)
        (declare (ignore vl2))
        (report "eig/reads-lower-triangle-only"
                (equal (token wl) (token wl2))
                "打零上三角后特征值变了 → 说明读了上三角"))
      (multiple-value-bind (wu vu) (vt-eig upper)
        (declare (ignore vu))
        (report "eig/ignores-upper-triangle"
                (not (equal (token wl) (token wu)))
                "只给上三角却给出同一组特征值 → UPLO 不对")))
    ;; 复数：对角虚部必须被忽略（zheev 对厄米特输入只取实对角）
    (let* ((c1 (vt-from-sequence (list (list (complex 3.0d0 0.0d0) (complex 1.0d0 1.0d0))
                                       (list (complex 1.0d0 -1.0d0) (complex 2.0d0 0.0d0)))
                                 :dtype :complex128))
           (c2 (vt-from-sequence (list (list (complex 3.0d0 7.0d0) (complex 1.0d0 1.0d0))
                                       (list (complex 1.0d0 -1.0d0) (complex 2.0d0 -4.0d0)))
                                 :dtype :complex128)))
      (multiple-value-bind (w1 v1) (vt-eig c1)
        (multiple-value-bind (w2 v2) (vt-eig c2)
          (report "cx-eig/ignores-diagonal-imag"
                  (and (equal (token w1) (token w2))
                       (equal (token v1) (token v2)))
                  "对角虚部被计入 → 与 zheev 语义不符"))))
    ;; cholesky 只读下三角，且返回下三角
    (let ((l (vt-cholesky base))
          (l2 (vt-cholesky (vt-from-sequence '((4.0d0 99.0d0 99.0d0)
                                               (1.0d0 3.0d0 99.0d0)
                                               (2.0d0 0.0d0 5.0d0))
                                             :dtype :float64))))
      (report "chol/reads-lower-triangle-only" (equal (token l) (token l2))
              "上三角的垃圾值影响了结果")
      (report "chol/returns-lower-triangular"
              (equal (token (vt-triu l :k 1)) (token (vt-zeros '(3 3) :dtype :float64)))
              "返回的不是下三角")
      (report "chol/reconstructs-lower-product"
              (< (vt-item (vt-amax (vt-abs (vt-- (vt-@ l (vt-transpose l)) base))))
                 1d-13)))))

;;; ------------------------------------------------------------------
;;; G. MATMUL 的收缩轴校验：实数与复数必须同时拒绝
;;; ------------------------------------------------------------------

(defun ok-matmul-shape-error (name a b)
  "断言 A@B 报错且消息里带「收缩轴」。"
  (handler-case
      (let ((r (vt-matmul a b)))
        (report name nil (format nil "没有报错，返回形状 ~s" (vt-shape r))))
    (error (e)
      (let ((msg (princ-to-string e)))
        (report name (and (search "收缩轴" msg) t)
                (format nil "消息：~a" msg))))))

(defun run-matmul-shape-check ()
  (section "G MATMUL 收缩轴校验（复数路径原先静默给出垃圾）")
  (let ((r (vt-zeros '(2 3) :dtype :float64))
        (c (vt-zeros '(2 3) :dtype :complex128)))
    (ok-matmul-shape-error "matmul/real-(2 3)x(2 2)" r (vt-zeros '(2 2) :dtype :float64))
    (ok-matmul-shape-error "matmul/cx-(2 3)x(2 2)" c (vt-zeros '(2 2) :dtype :complex128))
    ;; 这条是修复前的原始现场：复数路径不校验 → 按 k=4 越界读 B，
    ;; 返回一个 (4 2) 的矩阵，元素量级 1e77，且**没有任何信号**。
    (ok-matmul-shape-error "matmul/cx-(4 4)x(2 2)"
                           (vt-zeros '(4 4) :dtype :complex128)
                           (vt-zeros '(2 2) :dtype :complex128))
    (ok-matmul-shape-error "matmul/cx-batched-(2 4 4)x(2 2 2)"
                           (vt-zeros '(2 4 4) :dtype :complex128)
                           (vt-zeros '(2 2 2) :dtype :complex128))
    (ok-matmul-shape-error "matmul/real-(3 2)x(4 2)"
                           (vt-zeros '(3 2) :dtype :float64)
                           (vt-zeros '(4 2) :dtype :float64)))
  ;; 合法输入不得被这条校验误伤：1-D 提升的四种形态 + 批量 + 复数
  (let ((u (vt-from-sequence '(1.0d0 2.0d0 3.0d0 4.0d0) :dtype :float64))
        (m (vt-from-sequence '((1.0d0 2.0d0 3.0d0 4.0d0)
                               (0.0d0 1.0d0 0.0d0 1.0d0)) :dtype :float64)))
    (report "matmul/1d-x-1d"
            (and (equal (vt-shape (vt-matmul u u)) '())
                 (< (abs (- (vt-item (vt-matmul u u)) 30.0d0)) 1d-12)))
    (report "matmul/2d-x-1d" (equal (vt-shape (vt-matmul m u)) '(2)))
    (report "matmul/1d-x-2d" (equal (vt-shape (vt-matmul u (vt-transpose m))) '(2)))
    (report "matmul/batched-complex"
            (equal (vt-shape (vt-matmul (vt-zeros '(2 3 4) :dtype :complex128)
                                        (vt-zeros '(2 4 5) :dtype :complex128)))
                   '(2 3 5)))
    (report "matmul/broadcast-batch" (equal (vt-shape (vt-matmul (vt-zeros '(1 3 4) :dtype :complex128)
                                                                (vt-zeros '(6 4 5) :dtype :complex128)))
                                           '(6 3 5)))))

;;; ------------------------------------------------------------------

(run-view-consistency)
(run-out-view)
(run-svd-determinism)
(run-complex-unitary)
(run-lstsq-contract)
(run-triangle-semantics)
(run-matmul-shape-check)

(format t "~&=====================================~%")
(format t "视图扫描项数: ~d~%" *n-view*)
(format t "Total: ~d / Pass: ~d / Fail: ~d~%" (+ *pass* *fail*) *pass* *fail*)
(format t "PASS=~d FAIL=~d~%" *pass* *fail*)
(when *failures*
  (format t "~&--- 失败清单 ---~%")
  (dolist (f (reverse *failures*)) (format t "  · ~a~%" f)))
(when (plusp *fail*) (sb-ext:exit :code 1))
