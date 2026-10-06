;;;; out-contig-tests.lisp — 非连续 :out 回归测试
;;;;
;;;; 前置条件：
;;;;   1. def-vt-reduce 已应用"方案 A"（非连续 :out 支持）
;;;;   2. def-vt-reduce 已加"广播 :out 拒绝"校验
;;;;   没有 1，测试 2/3/4/5/6 会报 "must be contiguous"；
;;;;   没有 2，测试 7 会失败（静默写同一物理位置）。
;;;;
;;;; 运行：
;;;;   scripts/run-sbcl.sh \
;;;;     --eval '(ql:quickload :clvt :verbose nil)' \
;;;;     --load scripts/out-contig-tests.lisp \
;;;;     --eval '(sb-ext:exit :code (if (clvt-outtest:run) 0 1))'

(require :asdf)
#+quicklisp (ql:quickload :clvt)
(asdf:load-system :clvt)
(in-package :clvt)
;;; ------------------------------------------------------------------
;;; 断言辅助
;;; ------------------------------------------------------------------

(defparameter *pass* 0)
(defparameter *fail* 0)

(defun check (label cond)
  "断言 COND 为真。"
  (if cond
      (progn (incf *pass*) (format t "  [ OK ] ~a~%" label))
      (progn (incf *fail*) (format t "  [FAIL] ~a~%" label))))

(defun check-error (label thunk)
  "断言 (funcall THUNK) 抛错。"
  (let ((signaled nil) (msg nil))
    (handler-case (funcall thunk)
      (error (e) (setf signaled t msg (format nil "~a" e))))
    (if signaled
        (progn (incf *pass*) (format t "  [ OK ] ~a~%         报错: ~a~%" label msg))
        (progn (incf *fail*) (format t "  [FAIL] ~a: 应报错却成功~%" label)))))

(defun check-ref-equal (label tensor expected-coords-values)
  "断言 TENSOR 在每个 (coords . value) 位置读取正确。
   expected-coords-values 形如 (((0 0) 1d0) ((1 0) 2d0))。
   避免在期望值里手数嵌套括号层数。"
  (let ((ok t))
    (dolist (pair expected-coords-values)
      (unless (= (apply #'vt-ref tensor (first pair)) (second pair))
        (setf ok nil)
        (format t "        位置 ~s: got ~s, want ~s~%"
                (first pair)
                (apply #'vt-ref tensor (first pair))
                (second pair))))
    (check label ok)))

;;; ------------------------------------------------------------------
;;; 测试入口
;;; ------------------------------------------------------------------

(defun run ()
  (setf *pass* 0 *fail* 0)
  (format t "~&===== 非连续 :out 回归测试 =====~%")
  (finish-output)

  ;; ================================================================
  ;; 1. 连续 out（回归基线）
  ;; ================================================================
  ;; x = ((1 2) (3 4)), axis=0 → (1+3, 2+4) = (4, 6)
  (let* ((x (vt-from-sequence '((1d0 2d0) (3d0 4d0))))
         (o (vt-zeros '(2) :dtype :float64)))
    (vt-sum x :axis 0 :out o)
    (check-ref-equal "1  sum 连续 out" o
                     '(((0) 4d0) ((1) 6d0))))

  ;; ================================================================
  ;; 2. 非连续 out（1D 视图 base[::2]）
  ;; ================================================================
  ;; base : (4) 连续；view = base[::2] → shape (2), strides (2)
  ;; 写入 view 只应影响 base 的偶数位。
  (let* ((x (vt-from-sequence '((1d0 2d0) (3d0 4d0))))
         (base (vt-zeros '(4) :dtype :float64))
         (view (vt-slice base '(nil nil 2))))          ; 单个 spec 参数
    (check "2a view.strides = (2)，非连续"
           (equal (vt-strides view) '(2)))
    (vt-sum x :axis 0 :out view)
    (check-ref-equal "2b base 偶数位被写入" base
                     '(((0) 4d0) ((2) 6d0)))
    (check-ref-equal "2c base 奇数位保持 0" base
                     '(((1) 0d0) ((3) 0d0)))
    (check-ref-equal "2d view 逻辑内容" view
                     '(((0) 4d0) ((1) 6d0))))

  ;; ================================================================
  ;; 3. keepdims + 非连续 out
  ;; ================================================================
  ;; base : (1 4)；view = base[:, ::2] → shape (1 2), strides (4 2)
  ;; ★ vt-slice 的 &rest：两个 spec = 两个独立参数
  (let* ((x (vt-from-sequence '((1d0 2d0) (3d0 4d0))))
         (base (vt-zeros '(1 4) :dtype :float64))
         (view (vt-slice base '(:all) '(nil nil 2))))
    (check "3a view.shape = (1 2)"
           (equal (vt-shape view) '(1 2)))
    (check "3b view.strides = (4 2)，非连续"
           (equal (vt-strides view) '(4 2)))
    (vt-sum x :axis 0 :keepdims t :out view)
    (check-ref-equal "3c 写入位置正确" base
                     '(((0 0) 4d0) ((0 2) 6d0)))
    (check-ref-equal "3d 未写位置保持 0" base
                     '(((0 1) 0d0) ((0 3) 0d0))))

  ;; ================================================================
  ;; 4. argmax 走同一路径（int64 out，对标 numpy intp）
  ;; ================================================================
  ;; x = ((1 5) (3 4)), axis=0
  ;;   第 0 列：1 vs 3 → max 在 row 1 → idx=1
  ;;   第 1 列：5 vs 4 → max 在 row 0 → idx=0
  (let* ((x (vt-from-sequence '((1d0 5d0) (3d0 4d0))))
         (base (vt-zeros '(4) :dtype :int64))
         (view (vt-slice base '(nil nil 2))))
    (vt-argmax x :axis 0 :out view)
    (check-ref-equal "4  argmax 非连续 out" base
                     '(((0) 1) ((2) 0)))
    (check-ref-equal "4b 未写位置保持 0" base
                     '(((1) 0) ((3) 0))))

  ;; ================================================================
  ;; 5. nanmax 走同一路径
  ;; ================================================================
  ;; x = ((1 nan) (3 4)), axis=0
  ;;   第 0 列：1 vs 3 → max=3（跳过 NaN）
  ;;   第 1 列：nan vs 4 → max=4（跳过 NaN）
  (let* ((nan (vt-float-nan))
         (x (vt-from-sequence (list (list 1d0 nan) (list 3d0 4d0))))
         (base (vt-zeros '(4) :dtype :float64))
         (view (vt-slice base '(nil nil 2))))
    (vt-nanmax x :axis 0 :out view)
    (check-ref-equal "5  nanmax 非连续 out" base
                     '(((0) 3d0) ((2) 4d0)))
    (check-ref-equal "5b 未写位置保持 0" base
                     '(((1) 0d0) ((3) 0d0))))

  ;; ================================================================
  ;; 6. 多轴归约 + 非连续 out（keepdims=nil）
  ;; ================================================================
  ;; x = arange(24).reshape(2,3,4)
  ;; sum(axis=(0,2)) → (60, 92, 124)
  ;; 独立参考：对每个 i 显式求和 x[:, i, :]
  (let* ((x (vt-reshape (vt-arange 24 :dtype :float64) '(2 3 4)))
         (base (vt-zeros '(6) :dtype :float64))
         (view (vt-slice base '(nil nil 2))))          ; shape (3), strides (2)
    ;; 独立参考值（显式三重循环，避免心算错误）
    (let ((ref (loop for i below 3
                     collect (loop for a below 2
                                   sum (loop for c below 4
                                             sum (vt-ref x a i c))))))
      (check "6a 参考值 = (60 92 124)"
             (equal ref '(60d0 92d0 124d0))))

    (check "6b view.strides = (2)，非连续"
           (equal (vt-strides view) '(2)))

    (vt-sum x :axis '(0 2) :out view)

    (check-ref-equal "6c 写入偶数位正确" base
                     '(((0) 60d0) ((2) 92d0) ((4) 124d0)))
    (check-ref-equal "6d 奇数位保持 0" base
                     '(((1) 0d0) ((3) 0d0) ((5) 0d0))))

  ;; ================================================================
  ;; 6b. 多轴归约 + keepdims=t + 非连续 out
  ;; ================================================================
  ;; x = arange(24).reshape(2,3,4), axis=(0,2), keepdims=t
  ;; → out.shape = (1 3 1)
  ;; base : (1 3 2)；view = base[:, :, ::2] → shape (1 3 1), strides (6 2 2)
  ;; ★ 三个 spec = 三个独立参数
  (let* ((x (vt-reshape (vt-arange 24 :dtype :float64) '(2 3 4)))
         (base (vt-zeros '(1 3 2) :dtype :float64))
         (view (vt-slice base '(:all) '(:all) '(nil nil 2))))
    (check "6b-1 view.shape = (1 3 1)"
           (equal (vt-shape view) '(1 3 1)))
    (check "6b-2 view.strides = (6 2 2)，非连续"
           (equal (vt-strides view) '(6 2 2)))

    (vt-sum x :axis '(0 2) :keepdims t :out view)

    (check-ref-equal "6b-3 写入位置正确" base
                     '(((0 0 0) 60d0)
                       ((0 1 0) 92d0)
                       ((0 2 0) 124d0)))
    (check-ref-equal "6b-4 未写位置保持 0" base
                     '(((0 0 1) 0d0)
                       ((0 1 1) 0d0)
                       ((0 2 1) 0d0))))
  ;; ================================================================
  ;; 7. 广播 out 必须报错
  ;; ================================================================
  ;; 形状 (1 2) 与期望匹配，但存在 dim>1 且 stride=0 的轴 —— 广播视图。
  ;; 写入会重复写同一物理位置，语义不成立。
  (let* ((x (vt-from-sequence '((1d0 2d0) (3d0 4d0))))
	 (bcast (vt-broadcast-to (vt-zeros '(1 1)) '(1 2))))
    (check "7a bcast 是广播视图（存在 dim>1 且 stride=0 的轴）"
           (loop for d in (vt-shape bcast)
		 for s in (vt-strides bcast)
		   thereis (and (> d 1) (zerop s))))
    (check-error "7b 广播 out 必须报错"
		 (lambda () (vt-sum x :axis 0 :keepdims t :out bcast))))

  ;; ================================================================
  ;; 汇总
  ;; ================================================================
  (format t "~%通过 ~a / 失败 ~a~%" *pass* *fail*)
  (finish-output)
  (zerop *fail*))
(run)
