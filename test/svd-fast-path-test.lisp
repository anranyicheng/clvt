;;;; svd-fast-path-test.lisp — vt-svd 双路径测试（SIMD 快速内核 / 通用内核）
;;;;
;;;; 覆盖类目（对应 TEST-PLAN.md）：
;;;;   D1 差分参考   ：与优化前【原实现】的内嵌期望值逐项判定
;;;;                   （s 向量 / 重建误差上界 / 正交误差上界 / 首列符号向量）
;;;;   C8 选路不变量 ：*vt-svd-simd-enabled* = t(AVX) ≡ nil(通用) 结果一致
;;;;   C2/C3 数学性质：重建 a=u·s·vt、奇异值降序非负、U/Vt 正交（econ + full）
;;;;   G6 numpy 对比 ：奇异值 vs numpy.linalg.svd 实时对比（python3 不可用时跳过）
;;;;   C1 退化形状   ：1×1 / 1×n / n×1 / 零矩阵 / 空形状(0×3, 3×0)
;;;;   其他         ：秩亏随机正交补（固定种子）/ 非连续视图输入 / dtype 提升 /
;;;;                  vt-pinv 回归（依赖 vt-svd）
;;;;
;;;; 期望值来源：优化前原实现 (commit 4b0443d 的 vt-svd) 与 numpy 2.5.3，
;;;; 由 scripts/gen-svd-reference.lisp 生成；零奇异值补全用例固定随机种子。
;;;; 全部使用小矩阵（≤5×5 + 基准另有上限约定），运行时间亚秒级。

(require :asdf)
;; 依据本文件位置定位 clvt 根目录（不依赖 cwd，兼容 run-tests.sh 与独立运行）
(push (truename (make-pathname :directory (append (pathname-directory (truename *load-truename*))
                                                  '(:up))))
      asdf:*central-registry*)
(asdf:load-system :clvt)
(in-package :clvt)

;;; ============================================================
;;; 测试框架（与 property-test.lisp 同款约定）
;;; ============================================================
(defvar *svd-N* 0)
(defvar *svd-P* 0)
(defvar *svd-F* 0)
(defvar *svd-F-list* nil)

(defun svd-ok (name)
  (declare (ignore name))
  (incf *svd-N*) (incf *svd-P*))

(defun svd-fail (name &optional detail)
  (incf *svd-N*) (incf *svd-F*) (push name *svd-F-list*)
  (format t "  FAIL ~a~@[  ~a~]~%" name detail))

(defun svd-t! (name ok &optional detail)
  (if ok (svd-ok name) (svd-fail name detail)))

(defun svd-close-p (a b &optional (tol 1d-9))
  "混合容差：相对差 ≤ tol 或极小绝对差（处理 0 值）。"
  (let ((diff (abs (- a b)))
        (scale (max (abs a) (abs b) 1.0d-30)))
    (or (<= diff (* tol scale))
        (<= diff 1.0d-28))))

(defun svd-vec-close-p (xs ys &optional (tol 1d-9))
  (and (= (length xs) (length ys))
       (loop for x in xs for y in ys
             always (svd-close-p x y tol))))

(defun svd-tensor-close-p (t1 t2 &optional (tol 1d-9))
  "同形状且逐元素接近；空张量仅比形状。"
  (and (equal (vt-shape t1) (vt-shape t2))
       (or (zerop (vt-size t1))
           (<= (vt-ref (vt-amax (vt-abs (vt-- t1 t2)))) tol))))

(defun a= (row-list) (vt-from-sequence row-list :dtype :float64))

(defun svd-props (a &key full)
  "返回 (values u s vt recon-err orth-err s-list)。
recon-err = max|u·diag(s)·vt − a|；orth-err = max|uᵀu − I|。"
  (multiple-value-bind (u s vt) (vt-svd a :full-matrices full)
    (let* ((recon (vt-@ u (vt-@ (vt-diag s) vt)))
           (err (if (zerop (vt-size recon)) 0.0d0
                    (vt-ref (vt-amax (vt-abs (vt-- recon a))))))
           (kk (second (vt-shape u)))
           (utu (vt-@ (vt-transpose u) u))
           (u-err (if (zerop (vt-size utu)) 0.0d0
                      (vt-ref (vt-amax (vt-abs (vt-- utu (vt-eye kk :dtype :float64))))))))
      (values u s vt err u-err (and (> (vt-size s) 0) (vt-to-list s))))))

;;; ============================================================
;;; 固定测试矩阵
;;; ============================================================
(defparameter *m-diag3*    (a= (list (list 5.0d0 0.0d0 0.0d0)
                                     (list 0.0d0 3.0d0 0.0d0)
                                     (list 0.0d0 0.0d0 1.0d0))))
(defparameter *m-general3* (a= (list (list 4.0d0 11.0d0 14.0d0)
                                     (list 8.0d0 7.0d0 -2.0d0)
                                     (list 1.0d0 5.0d0 9.0d0))))
(defparameter *m-rankdef3* (a= (list (list 1.0d0 2.0d0 3.0d0)
                                     (list 2.0d0 4.0d0 6.0d0)
                                     (list 3.0d0 6.0d0 9.0d0))))
(defparameter *m-tall4x2*  (a= (list (list 1.0d0 2.0d0)
                                     (list 3.0d0 4.0d0)
                                     (list 5.0d0 6.0d0)
                                     (list 7.0d0 8.0d0))))
(defparameter *m-wide2x4*  (a= (list (list 1.0d0 2.0d0 3.0d0 4.0d0)
                                     (list 5.0d0 6.0d0 7.0d0 8.0d0))))
(defparameter *m-one1x1*   (a= (list (list 5.0d0))))
(defparameter *m-zero2x3*  (a= (list (list 0.0d0 0.0d0 0.0d0)
                                     (list 0.0d0 0.0d0 0.0d0))))
(defparameter *m-view4x4*  (a= (list (list 1.0d0 2.0d0 3.0d0 4.0d0)
                                     (list 5.0d0 6.0d0 7.0d0 8.0d0)
                                     (list 9.0d0 10.0d0 11.0d0 12.0d0)
                                     (list 13.0d0 14.0d0 15.0d0 16.0d0))))
(defparameter *m-fx44*     (a= (list (list 4.0d0 1.0d0 -2.0d0 3.0d0)
                                     (list 0.0d0 5.0d0 2.0d0 -1.0d0)
                                     (list 1.0d0 1.0d0 6.0d0 0.0d0)
                                     (list 2.0d0 -3.0d0 1.0d0 7.0d0))))

;;; ============================================================
;;; D1 差分参考：与优化前原实现内嵌期望值对比
;;; ============================================================
(defun test-differential-reference ()
  (format t "~%--- D1 差分参考（原实现期望值）---~%")
  ;; diag3: 精确对角
  (multiple-value-bind (u s vt err uerr s-list) (svd-props *m-diag3*)
    (declare (ignore vt s))
    (svd-t! "ref/diag3/s"        (svd-vec-close-p s-list '(5.0d0 3.0d0 1.0d0) 1d-12))
    (svd-t! "ref/diag3/recon=0"  (< err 1d-12))
    (svd-t! "ref/diag3/orth"     (< uerr 1d-12))
    (svd-t! "ref/diag3/u00"      (svd-close-p (vt-ref u 0 0) 1.0d0 1d-12)))
  ;; general3: 奇异值 = (√457, √999/10, √15/√190)，与 numpy 一致
  (multiple-value-bind (u s vt err uerr s-list) (svd-props *m-general3*)
    (declare (ignore vt s))
    (svd-t! "ref/general3/s"      (svd-vec-close-p
                                   s-list '(21.37807435879723d0 9.994953042650415d0 0.2808031070870468d0)
                                   1d-9))
    (svd-t! "ref/general3/recon"  (< err 1d-13))
    (svd-t! "ref/general3/orth"   (< uerr 1d-13))
    (svd-t! "ref/general3/u00"    (svd-close-p (vt-ref u 0 0) 0.8517613106069895d0 1d-9)))
  ;; rankdef3: 秩 1，σ=(14,0,0)
  (multiple-value-bind (u s vt err uerr s-list) (svd-props *m-rankdef3*)
    (declare (ignore vt s))
    (svd-t! "ref/rankdef3/s"      (svd-vec-close-p s-list '(14.000000000000002d0 0.0d0 0.0d0) 1d-9))
    (svd-t! "ref/rankdef3/recon"  (< err 1d-13))
    (svd-t! "ref/rankdef3/orth"   (< uerr 1d-13))
    (svd-t! "ref/rankdef3/u00"    (svd-close-p (vt-ref u 0 0) 0.2672612419124244d0 1d-9)))
  ;; tall4x2 / wide2x4
  (multiple-value-bind (u s vt err uerr s-list) (svd-props *m-tall4x2*)
    (declare (ignore vt s))
    (svd-t! "ref/tall4x2/s"       (svd-vec-close-p s-list '(14.269095499261482d0 0.6268282324175408d0) 1d-9))
    (svd-t! "ref/tall4x2/recon"   (< err 1d-13))
    (svd-t! "ref/tall4x2/orth"    (< uerr 1d-12))
    (svd-t! "ref/tall4x2/u00"     (svd-close-p (vt-ref u 0 0) 0.1524832333102009d0 1d-9)))
  (multiple-value-bind (u s vt err uerr s-list) (svd-props *m-wide2x4*)
    (declare (ignore vt s))
    (svd-t! "ref/wide2x4/s"       (svd-vec-close-p s-list '(14.227407412633744d0 1.2573298353791111d0) 1d-9))
    (svd-t! "ref/wide2x4/recon"   (< err 1d-13))
    (svd-t! "ref/wide2x4/orth"    (< uerr 1d-12))
    ;; 注：u00 与原始实现（stash 复测 2026-10-11）及数学特征向量一致为 0.37617…
    ;; 初次生成脚本该值 0.98613 为录制异常（数学上 σ₁=14.2274 唯一确定 |u00|=0.37617）
    (svd-t! "ref/wide2x4/u00"     (svd-close-p (vt-ref u 0 0) 0.3761682344281407d0 1d-9)))
  ;; 1×1
  (multiple-value-bind (u s vt err uerr s-list) (svd-props *m-one1x1*)
    (declare (ignore vt s uerr))
    (svd-t! "ref/one1x1/s"        (svd-vec-close-p s-list '(5.0d0) 1d-12))
    (svd-t! "ref/one1x1/recon"    (< err 1d-12))
    (svd-t! "ref/one1x1/u00"      (svd-close-p (vt-ref u 0 0) 1.0d0 1d-12)))
  ;; 零矩阵：奇异值全 0，U 列为随机正交补（固定种子 777 可复现）
  (vt-random-seed 777)
  (multiple-value-bind (u s vt err uerr s-list) (svd-props *m-zero2x3*)
    (declare (ignore vt s))
    (svd-t! "ref/zero2x3/s"       (svd-vec-close-p s-list '(0.0d0 0.0d0) 1d-12))
    (svd-t! "ref/zero2x3/recon"   (< err 1d-12))
    (svd-t! "ref/zero2x3/orth"    (< uerr 1d-12))
    (svd-t! "ref/zero2x3/u00"     (svd-close-p (vt-ref u 0 0) 0.4507195595519486d0 1d-9))
    (svd-t! "ref/zero2x3/u10"     (svd-close-p (vt-ref u 1 0) 0.8926656029204314d0 1d-9)))
  ;; 秩亏随机补全列（固定种子 777）：第 3 列
  (vt-random-seed 777)
  (multiple-value-bind (u s vt err uerr s-list) (svd-props *m-rankdef3*)
    (declare (ignore vt err uerr s))
    (svd-t! "ref/rankdef3-seed/s"  (svd-vec-close-p s-list '(14.000000000000002d0 0.0d0 0.0d0) 1d-9))
    (svd-t! "ref/rankdef3-seed/u02" (svd-close-p (vt-ref u 0 2) 0.8923807245936775d0 1d-9)))
  ;; 随机 5×5（固定种子 2026）
  (vt-random-seed 2026)
  (multiple-value-bind (u s vt err uerr s-list)
      (svd-props (vt-random '(5 5)))
    (declare (ignore vt s))
    (svd-t! "ref/rand5/s"      (svd-vec-close-p
                                s-list '(2.8718370770831827d0 0.8450056527944279d0 0.6933809783769873d0
                                         0.40985465123415155d0 0.09792607664425912d0)
                                1d-9))
    (svd-t! "ref/rand5/recon"  (< err 1d-12))
    (svd-t! "ref/rand5/orth"   (< uerr 1d-12))
    (svd-t! "ref/rand5/u00"    (svd-close-p (vt-ref u 0 0) 0.5473222216712316d0 1d-9)))
  ;; full-matrices 形状契约
  (multiple-value-bind (u s vt) (vt-svd *m-tall4x2* :full-matrices t)
    (svd-t! "ref/tall4x2-full/shapes"
            (and (equal (vt-shape u) '(4 4)) (equal (vt-shape s) '(2))
                 (equal (vt-shape vt) '(2 2)))))
  (multiple-value-bind (u s vt) (vt-svd *m-wide2x4* :full-matrices t)
    (svd-t! "ref/wide2x4-full/shapes"
            (and (equal (vt-shape u) '(2 2)) (equal (vt-shape s) '(2))
                 (equal (vt-shape vt) '(4 4)))))
  ;; 空形状（0×3 / 3×0）
  (multiple-value-bind (u s vt) (vt-svd (vt-zeros '(0 3) :dtype :float64))
    (svd-t! "ref/empty0x3/shapes"
            (and (equal (vt-shape u) '(0 0)) (= (vt-size s) 0)
                 (equal (vt-shape vt) '(0 3)))))
  (multiple-value-bind (u s vt) (vt-svd (vt-zeros '(3 0) :dtype :float64))
    (svd-t! "ref/empty3x0/shapes"
            (and (equal (vt-shape u) '(3 0)) (= (vt-size s) 0)
                 (equal (vt-shape vt) '(0 0)))))
  ;; 非连续视图输入（子矩阵切片）
  (multiple-value-bind (u s vt err uerr s-list)
      (svd-props (vt-slice *m-view4x4* '(1 3) '(1 3)))
    (declare (ignore u vt s))
    (svd-t! "ref/view2x2/s"      (svd-vec-close-p s-list '(17.491360824150707d0 0.22868432251864088d0) 1d-9))
    (svd-t! "ref/view2x2/recon"  (< err 1d-12))
    (svd-t! "ref/view2x2/orth"   (< uerr 1d-12)))
  ;; vt-pinv 回归（内部依赖 vt-svd）
  (let ((p (vt-pinv *m-general3*)))
    (svd-t! "ref/pinv-general3/p00" (svd-close-p (vt-ref p 0 0) -1.2166666666666637d0 1d-9))
    (svd-t! "ref/pinv-general3/p12" (svd-close-p (vt-ref p 1 2) -2.0000000000000004d0 1d-9))
    (svd-t! "ref/pinv-general3/shape" (equal (vt-shape p) '(3 3)))))

;;; ============================================================
;;; C8 选路不变量：SIMD 快速内核 ≡ 通用内核
;;; ============================================================
(defun test-route-invariance ()
  (format t "~%--- C8 选路不变量（AVX ≡ 通用内核）---~%")
  (flet ((route-props (a)
           (multiple-value-list (svd-props a))))
    (dolist (c (list (cons "general3" *m-general3*)
                     (cons "tall4x2"  *m-tall4x2*)
                     (cons "wide2x4"  *m-wide2x4*)
                     (cons "fx44"     *m-fx44*)
                     (cons "rankdef3" *m-rankdef3*)))
      (destructuring-bind (name . a) c
        (vt-random-seed 4242)
        (let ((p1 (route-props a)))
          (setf *vt-svd-simd-enabled* nil)
          (vt-random-seed 4242)
          (let ((p2 (route-props a)))
            (setf *vt-svd-simd-enabled* t)
            (destructuring-bind (u1 s1 vt1 e1 o1 l1) p1
	       (declare (ignore s1))
              (destructuring-bind (u2 s2 vt2 e2 o2 l2) p2
                (declare (ignore s2))
                (svd-t! (format nil "route/~a/s" name)
                        (svd-vec-close-p l1 l2 1d-12))
                (svd-t! (format nil "route/~a/u" name)
                        (svd-tensor-close-p u1 u2 1d-9))
                (svd-t! (format nil "route/~a/vt" name)
                        (svd-tensor-close-p vt1 vt2 1d-9))
                (svd-t! (format nil "route/~a/recon" name)
                        (and (< e1 1d-10) (< e2 1d-10)))
                (svd-t! (format nil "route/~a/orth" name)
                        (and (< o1 1d-10) (< o2 1d-10)))))))))))

;;; ============================================================
;;; C2/C3 数学性质（econ + full，随机矩阵族）
;;; ============================================================
(defun svd-props-full (a)
  "full 模式属性：返回 (values u s vt recon-err s-list)。
full 模式下 Σ 为 m×n 长方形对角阵（numpy 同款语义），不能直接用 vt-diag。"
  (multiple-value-bind (u s vt) (vt-svd a :full-matrices t)
    (let* ((m (first (vt-shape a)))
           (n (second (vt-shape a)))
           (k (vt-size s))
           (sig (vt-zeros (list m n) :dtype :float64)))
      (dotimes (i k)
        (setf (vt-ref sig i i) (vt-ref s i)))
      (let* ((recon (vt-@ u (vt-@ sig vt)))
             (err (if (zerop (vt-size recon)) 0.0d0
                      (vt-ref (vt-amax (vt-abs (vt-- recon a)))))))
        (values u s vt err (and (> (vt-size s) 0) (vt-to-list s)))))))

(defun test-math-properties ()
  (format t "~%--- C2/C3 数学性质（随机矩阵族）---~%")
  (dolist (shape '((2 2) (3 3) (4 6) (6 4) (5 5) (1 4) (4 1)))
    (vt-random-seed (apply #'+ shape))
    (let* ((a (vt-random shape))
           (m (first shape))
           (n (second shape))
           (k (min m n)))
      (multiple-value-bind (u s vt err uerr s-list) (svd-props a)
        (declare (ignore u vt s))
        (svd-t! (format nil "prop/~dx~d/recon" m n)    (< err 1d-10))
        (svd-t! (format nil "prop/~dx~d/orth" m n)     (< uerr 1d-10))
        (svd-t! (format nil "prop/~dx~d/len" m n)      (= (length s-list) k))
        (svd-t! (format nil "prop/~dx~d/desc" m n)
                (loop for i from 0 below (1- k)
                      always (>= (elt s-list i) (- (elt s-list (1+ i)) 1d-12))))
        (svd-t! (format nil "prop/~dx~d/nonneg" m n)
                (every (lambda (x) (>= x -1d-12)) s-list)))
      ;; vt 正交性（econ: vt·vtᵀ = I_k）
      (multiple-value-bind (u s vt) (vt-svd a)
        (declare (ignore u s))
        (let ((vvt (vt-@ vt (vt-transpose vt)))
              (eye (vt-eye k :dtype :float64)))
          (svd-t! (format nil "prop/~dx~d/vt-orth" m n)
                  (<= (vt-ref (vt-amax (vt-abs (vt-- vvt eye)))) 1d-10))))
      ;; full 模式重建与形状
      (multiple-value-bind (u s vt err s-list) (svd-props-full a)
        (declare (ignore s-list))
        (svd-t! (format nil "prop/~dx~d/full-recon" m n) (< err 1d-10))
        (svd-t! (format nil "prop/~dx~d/full-shape" m n)
                (and (= (first (vt-shape u)) m) (= (second (vt-shape u)) m)
                     (= (first (vt-shape vt)) n) (= (second (vt-shape vt)) n)
                     (= (vt-size s) k)))))))

;;; ============================================================
;;; G6 numpy 实时对比（python3+numpy 不可用时优雅跳过）
;;; ============================================================
(defun numpy-svd-s (matrix-literal)
  "调用 python3+numpy 计算奇异值，返回 double-float 精确列表或 nil（不可用）。
python 端以整数分数（Fraction）输出，Lisp 端读取后转 double —— 完全无损，
避免 repr 无 d 后缀被 Lisp 读成 single-float 的精度陷阱。"
  (handler-case
      (let* ((py (format nil
                         "import sys,numpy as np~%~
                          from fractions import Fraction~%~
                          a=np.array(eval(sys.argv[1]),dtype=np.float64)~%~
                          s=np.linalg.svd(a,compute_uv=False)~%~
                          print('('+ ' '.join(str(Fraction(float(x)).numerator)+'/'+str(Fraction(float(x)).denominator) for x in s) +')')~%"))
             (out (uiop:run-program (list "python3" "-c" py matrix-literal)
                                    :output :string :error-output :string)))
        (when (and out (plusp (length (string-trim '(#\Space #\Newline #\Return) out))))
          (mapcar (lambda (r) (float r 1.0d0))
                  (read-from-string out))))
    (error () nil)))

(defun matrix-literal (a)
  "把 2D 张量转为 python 可 eval 的嵌套元组字面量（定点 17 位，避免 1.0d0 语法）。"
  (let ((m (first (vt-shape a)))
        (n (second (vt-shape a))))
    (format nil "(~{~A~^,~})"
            (loop for i of-type fixnum from 0 below m
                  collect (format nil "(~{~A~^,~})"
                                  (loop for j of-type fixnum from 0 below n
                                        collect (format nil "~,17F" (vt-ref a i j))))))))

(defun test-numpy-live ()
  (format t "~%--- G6 numpy 实时对比 ---~%")
  (let* ((py-probe (ignore-errors (numpy-svd-s "((1.0,2.0),(3.0,4.0))"))))
    (if (null py-probe)
        (format t "  SKIP: python3/numpy 不可用，跳过实时对比（差分参考已覆盖 numpy 语义）~%")
        (progn
          (svd-t! "numpy/probe" (svd-vec-close-p py-probe '(5.4649857042190426d0 0.36596619062625751d0) 1d-9))
          (dolist (c (list (cons "general3" *m-general3*)
                           (cons "fx44" *m-fx44*)
                           (cons "tall4x2" *m-tall4x2*)))
            (destructuring-bind (name . a) c
              (let* ((np-s (numpy-svd-s (matrix-literal a)))
                     (clvt-s (multiple-value-list (vt-svd a))))
                (svd-t! (format nil "numpy/~a/s" name)
                        (svd-vec-close-p (vt-to-list (second clvt-s)) np-s 1d-8)))))))))

;;; ============================================================
;;; dtype 提升与契约
;;; ============================================================
(defun test-dtype-contract ()
  (format t "~%--- dtype 契约 ---~%")
  ;; float32 输入 → float64 输出
  (multiple-value-bind (u s vt) (vt-svd (vt-from-sequence '((1.0 2.0) (3.0 4.0)) :dtype :float32))
    (svd-t! "dtype/float32-in"
            (and (eq (vt-dtype u) :float64) (eq (vt-dtype s) :float64)
                 (eq (vt-dtype vt) :float64))
            (format nil "~a ~a ~a" (vt-dtype u) (vt-dtype s) (vt-dtype vt)))
    (svd-t! "dtype/float32-values"
            (svd-vec-close-p (vt-to-list s) '(5.4649857042190426d0 0.36596619062625751d0) 1d-9)))
  ;; int64 输入 → float64 输出
  (multiple-value-bind (u s vt) (vt-svd (vt-from-sequence '((1 2) (3 4) (5 6)) :dtype :int64))
    (svd-t! "dtype/int64-in"
            (and (eq (vt-dtype u) :float64) (eq (vt-dtype s) :float64)
                 (eq (vt-dtype vt) :float64)))
    (svd-t! "dtype/int64-len" (= (vt-size s) 2)))
  ;; 输入不被修改（转置布局只读源数据）
  (let ((a (a= (list (list 1.0d0 2.0d0) (list 3.0d0 4.0d0)))))
    (vt-svd a)
    (svd-t! "dtype/input-unmodified"
            (and (svd-close-p (vt-ref a 0 0) 1.0d0 1d-30)
                 (svd-close-p (vt-ref a 0 1) 2.0d0 1d-30)
                 (svd-close-p (vt-ref a 1 0) 3.0d0 1d-30)
                 (svd-close-p (vt-ref a 1 1) 4.0d0 1d-30)))))

;;; ============================================================
;;; 运行
;;; ============================================================
(defun run-svd-fast-path-tests ()
  (format t "~%========================================~%")
  (format t "  clvt SVD FAST-PATH TESTS~%")
  (format t "  SIMD(AVX) ≡ generic route + differential ref~%")
  (format t "========================================~%")
  (test-differential-reference)
  (test-route-invariance)
  (test-math-properties)
  (test-numpy-live)
  (test-dtype-contract)
  (format t "~%========================================~%")
  (format t "  SVD Fast-Path Tests: ~a total, ~a pass, ~a fail~%"
          *svd-N* *svd-P* *svd-F*)
  ;; 供 run-tests.sh 解析的机器可读汇总（格式 1）
  (format t "  Total: ~a | Pass: ~a | Fail: ~a~%" *svd-N* *svd-P* *svd-F*)
  (format t "========================================~%")
  (when *svd-F-list*
    (format t "Failed:~{~%  - ~a~}~%" (reverse *svd-F-list*)))
  (zerop *svd-F*))

;; 运行并以退出码兜底：0 = 全部通过, 1 = 存在失败
(let ((ok (run-svd-fast-path-tests)))
  (sb-ext:exit :code (if ok 0 1)))
