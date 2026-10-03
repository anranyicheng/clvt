;;;; property-strided-tests.lisp — 选路不变量与代数恒等式套件（C8 类目）
;;;;
;;;; 核心不变量（TEST-PLAN.md L3 底线的机制化）：
;;;;   任意非连续视图上的运算结果 ≡ 同一视图 contiguize 后的结果。
;;;;   优化路径只允许影响性能，绝不影响数值。
;;;;
;;;; 数据确定性：vt-random-seed 固定种子，形状表固定——每次运行完全一致。
;;;;
;;;; 运行：
;;;;   tmp/sbcl-2.6.8-install/bin/sbcl --noinform --non-interactive \
;;;;     --load clvt/test/property-strided-tests.lisp
;;;; 退出码 0 = 全部通过。

(asdf:load-system :clvt)
(in-package :clvt)

(defparameter *pass* 0)
(defparameter *fail* 0)

(defun check (label cond)
  (declare (optimize (speed 0)))
  (if cond
      (progn (incf *pass*) (format t "  ok  ~a~%" label))
      (progn (incf *fail*) (format t "  FAIL ~a~%" label)))
  (finish-output))

(defun %flatten (x)
  (if (atom x) (list x) (mapcan #'%flatten x)))

(defun check-allclose (label got expected &optional (rtol 1e-13) (atol 1e-14))
  "数值容差比较（约简求和顺序可能产生 1ulp 级差异，元素级运算实际逐位一致）。"
  (let ((g (%flatten got)) (e (%flatten expected)))
    (check label
           (and (= (length g) (length e))
                (every (lambda (a b)
                         (with-float-safe
                           (<= (abs (- a b))
                               (+ atol (* rtol (abs b))))))
                       g e)))))

;;; ------------------------------------------------------------------
;;; A. 选路不变量：strided ≡ contiguous
;;; ------------------------------------------------------------------

(defparameter *prop-shapes*
  '((5) (6) (2 3) (3 2) (2 5) (1 7) (4 3 2) (2 2 2 2))
  "性质测试形状表：rank 1-4，含奇数维、size-1 轴、非方阵。")

(defun %strided-view (base)
  "在第一个 dim>1 的轴上做 step-2 切片，保证非连续；其余轴 :all。
返回 view 与其连续副本 cview。"
  (let* ((shape (vt-shape base))
         (used nil)
         (specs (loop for d in shape
                      collect (if (and (not used) (> d 1))
                                  (progn (setf used t) '(nil nil 2))
                                  '(:all)))))
    (let ((view (apply #'vt-slice base specs)))
      (values view (vt-copy view)))))

(defun run-props-for-shape (shape)
  (vt-random-seed (+ 1000 (reduce (lambda (a b) (+ (* a 10) b)) shape)))
  (let* ((base (vt-random-uniform shape :low 0.05d0 :high 0.95d0)))
    (multiple-value-bind (view cview) (%strided-view base)
      (unless (vt-contiguous-p view)
        (let ((vshape (vt-shape view)))
          ;; 元素级：逐位一致
          (check-allclose (format nil "~a abs" vshape)
                          (vt-to-list (vt-abs view)) (vt-to-list (vt-abs cview)) 0 0)
          (check-allclose (format nil "~a sqrt(abs)" vshape)
                          (vt-to-list (vt-sqrt (vt-abs view)))
                          (vt-to-list (vt-sqrt (vt-abs cview))) 0 0)
          (check-allclose (format nil "~a exp" vshape)
                          (vt-to-list (vt-exp view)) (vt-to-list (vt-exp cview)) 1e-15 0)
          (check-allclose (format nil "~a x+x" vshape)
                          (vt-to-list (vt-+ view view)) (vt-to-list (vt-+ cview cview)) 0 0)
          (check-allclose (format nil "~a x*x" vshape)
                          (vt-to-list (vt-* view view)) (vt-to-list (vt-* cview cview)) 0 0)
          (check-allclose (format nil "~a x+2.5" vshape)
                          (vt-to-list (vt-+ view 2.5d0)) (vt-to-list (vt-+ cview 2.5d0)) 0 0)
          ;; 归约：容差 1e-13（求和顺序允许 1ulp 差异）
          (check-allclose (format nil "~a sum()" vshape)
                          (vt-to-list (vt-sum view)) (vt-to-list (vt-sum cview)))
          (check-allclose (format nil "~a amax()" vshape)
                          (vt-to-list (vt-amax view)) (vt-to-list (vt-amax cview)) 0 0)
          (check-allclose (format nil "~a amin()" vshape)
                          (vt-to-list (vt-amin view)) (vt-to-list (vt-amin cview)) 0 0)
          (when (>= (length vshape) 2)
            (check-allclose (format nil "~a sum(ax0)" vshape)
                            (vt-to-list (vt-sum view :axis 0))
                            (vt-to-list (vt-sum cview :axis 0)))
            (check-allclose (format nil "~a transpose" vshape)
                            (vt-to-list (vt-transpose view))
                            (vt-to-list (vt-transpose cview)) 0 0))
          (check (format nil "~a argmax()" vshape)
                 (equal (vt-to-list (vt-argmax view))
                        (vt-to-list (vt-argmax cview))))
          ;; matmul：仅 2D 且两维都 >= 2
          (when (and (= (length vshape) 2) (every (lambda (d) (> d 1)) vshape))
            (let ((b (vt-random-uniform (list (second vshape) 3) :low -0.5d0 :high 0.5d0)))
              (check-allclose (format nil "~a @ (d1,3)" vshape)
                              (vt-to-list (vt-matmul view b))
                              (vt-to-list (vt-matmul cview b)) 1e-12 0))))))))

(defun run-strided-props ()
  (format t "--- A. strided ≡ contiguous 选路不变量 ---~%")
  (dolist (s *prop-shapes*)
    (run-props-for-shape s)))

;;; ------------------------------------------------------------------
;;; B. 代数恒等式（性质优于实例）
;;; ------------------------------------------------------------------

(defun run-identities ()
  (format t "--- B. 代数恒等式 ---~%")
  (vt-random-seed 20260)
  ;; flip∘flip = id
  (let ((a (vt-random-uniform '(7))))
    (check "flip∘flip = id (1d)"
           (equal (vt-to-list (vt-flip (vt-flip a))) (vt-to-list a))))
  (let ((a (vt-random-uniform '(3 4))))
    (check "flip∘flip = id (2d)"
           (equal (vt-to-list (vt-flip (vt-flip a :axis 0) :axis 0)) (vt-to-list a)))
    ;; transpose∘transpose = id
    (check "transpose∘transpose = id"
           (equal (vt-to-list (vt-transpose (vt-transpose a))) (vt-to-list a)))
    ;; reshape/flatten 往返
    (check "flatten∘reshape = id"
           (equal (vt-to-list (vt-flatten (vt-reshape a '(2 2 3))))
                  (vt-to-list (vt-flatten a))))
    ;; concat 轴 0 后可切回原两块
    (let* ((b (vt-random-uniform '(2 4)))
           (m (vt-concatenate 0 a b)))
      (check-allclose "concat ax0 可逆（上半）"
                      (vt-to-list (vt-slice m '(0 3) '(:all))) (vt-to-list a) 0 0)
      (check-allclose "concat ax0 可逆（下半）"
                      (vt-to-list (vt-slice m '(3 5) '(:all))) (vt-to-list b) 0 0)))
  ;; where(c,a,a) = a（含 NaN 行不选中分支）
  (let* ((a (vt-random-uniform '(2 3)))
         (c (vt-const '(2 3) 1d0 :dtype :int64)))
    (check-allclose "where(c,a,a) = a"
                    (vt-to-list (vt-where c a a)) (vt-to-list a) 0 0))
  ;; 多轴 sum 分解：sum(sum(x,0),0) = sum(x)（3D）
  (let ((x (vt-random-uniform '(3 4 5))))
    (check-allclose "sum(ax0)∘sum(ax0) = sum()"
                    (vt-to-list (vt-sum (vt-sum (vt-sum x :axis 0) :axis 0)))
                    (vt-to-list (vt-sum x)) 1e-13 0))
  ;; 整数值 dtype 往返：float64→int64→float64 = id
  (let ((i (vt-from-sequence '(1d0 2d0 3d0 -4d0))))
    (check-allclose "astype 往返（整值）"
                    (vt-to-list (vt-astype (vt-astype i :int64) :float64))
                    (vt-to-list i) 0 0))
  ;; sort ≡ take(argsort)
  (let ((x (vt-random-uniform '(9) :low -2d0 :high 2d0)))
    (check-allclose "sort ≡ take(argsort)"
                    (vt-to-list (vt-sort x))
                    (vt-to-list (vt-take x (vt-argsort x))) 0 0)))

;;; ------------------------------------------------------------------

(defun run ()
  (setf *pass* 0 *fail* 0)
  (run-strided-props)
  (run-identities)
  (format t "~%通过 ~a / 失败 ~a~%" *pass* *fail*)
  (finish-output)
  (zerop *fail*))

(sb-ext:exit :code (if (run) 0 1))
