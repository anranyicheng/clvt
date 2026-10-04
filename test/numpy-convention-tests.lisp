;;;; numpy-convention-tests.lisp — numpy 2.1.3 对齐语义套件（v0.3.6）
;;;;
;;;; 覆盖 v0.3.6 修复的五大 numpy 对齐项（TEST-PLAN §5 重校准 / §6 缺口收敛）：
;;;;   1. 空归约语义：输出空 → 空结果；max/min/arg 族输出非空且归约区空 →
;;;;      ValueError（取代 v0.3.5 的 NaN 填充约定）；单位元族填单位元。
;;;;   2. :out 精度解耦：按输入提升计算、最后 cast 写入 out，与无 :out
;;;;      的结果逐位一致（sum/prod/mean/var/std/nan* 全族）。
;;;;   3. mod/rem 零除 dtype 语义：浮点语境 → NaN，整型语境 → 0。
;;;;   4. vt-/ true_divide 语义：整数输入提升 float64，零除按 IEEE
;;;;      得 ±Inf/NaN（不再报错）。
;;;;   5. random NaN/Inf 参数校验前移：干净参数错误（屏蔽 FP 陷阱）。
;;;;
;;;; 运行：
;;;;   bash test/run-tests.sh --suite numpy-convention-tests
;;;; 或直接：
;;;;   sbcl --noinform --non-interactive --load clvt/test/numpy-convention-tests.lisp

(asdf:load-system :clvt)
(in-package :clvt)

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

(defun %nan (x) (and (numberp x) (with-float-safe (not (= x x)))))

(defun run ()
  (setf *pass* 0 *fail* 0)
  (format t "--- numpy 2.1.3 对齐语义（v0.3.6）---~%")

  ;; ================================================================
  ;; 1. 空归约语义
  ;; ================================================================
  (format t "~%[1] 空归约：输出空 → 空结果~%")
  (check "amax (3 0) ax0 → 空结果 shape (0)"
         (equal (vt-shape (vt-amax (vt-zeros '(3 0)) :axis 0)) '(0)))
  (check "argmax (3 0) ax0 → 空结果 shape (0)"
         (equal (vt-shape (vt-argmax (vt-zeros '(3 0)) :axis 0)) '(0)))
  (check "sum (3 0) ax0 → 空结果 shape (0)"
         (equal (vt-shape (vt-sum (vt-zeros '(3 0)) :axis 0)) '(0)))

  (format t "~%[1] 空归约：max/min 族输出非空且归约区空 → ValueError~%")
  (check-error "amax (0)（zero-size → no identity）"
               (lambda () (vt-amax (vt-zeros '(0)))))
  (check-error "amax (3 0) ax1"
               (lambda () (vt-amax (vt-zeros '(3 0)) :axis 1)))
  (check-error "amin (3 0) ax1"
               (lambda () (vt-amin (vt-zeros '(3 0)) :axis 1)))
  (check-error "nanmax (3 0) ax1"
               (lambda () (vt-nanmax (vt-zeros '(3 0)) :axis 1)))
  (check-error "amax (0) :out 合法空 out 时仍报错（归约区空）"
               (lambda () (let ((o (vt-zeros '(0) :dtype :float64)))
                            (declare (ignore o))
                            (vt-amax (vt-zeros '(0)) :out (vt-zeros '(0) :dtype :float64)))))

  (format t "~%[1] 空归约：arg 族输出非空且归约区空 → ValueError~%")
  (check-error "argmax (0)"
               (lambda () (vt-argmax (vt-zeros '(0)))))
  (check-error "argmin (0)"
               (lambda () (vt-argmin (vt-zeros '(0)))))
  (check-error "argmax (3 0) ax1"
               (lambda () (vt-argmax (vt-zeros '(3 0)) :axis 1)))

  (format t "~%[1] 空归约：单位元族填单位元~%")
  (check "sum (0) = 0" (= (vt-item (vt-sum (vt-zeros '(0)))) 0.0d0))
  (check "prod (0) = 1" (= (vt-item (vt-prod (vt-zeros '(0) :dtype :int64))) 1))
  (check "all (0) = 1" (= (vt-item (vt-all (vt-zeros '(0)))) 1))
  (check "any (0) = 0" (= (vt-item (vt-any (vt-zeros '(0)))) 0))
  (check "nansum (3 0) ax1 = (0 0 0)"
         (equalp (vt-to-list (vt-nansum (vt-zeros '(3 0)) :axis 1)) '(0.0d0 0.0d0 0.0d0)))

  (format t "~%[1] 空归约：mean 空归约 → NaN（numpy 同款）~%")
  (check "mean (0) = NaN" (%nan (vt-item (vt-mean (vt-zeros '(0))))))

  ;; ================================================================
  ;; 2. :out 精度解耦
  ;; ================================================================
  (format t "~%[2] :out 精度解耦（按输入提升计算，最后 cast 写入）~%")
  (let ((o (make-vt '() 0 :dtype :int32)))
    (vt-sum (vt-from-sequence '(0.9d0 0.9d0)) :out o)
    (check "sum (0.9 0.9) :out int32 = 1" (= (vt-item o) 1)))
  (let ((o (make-vt '() 0 :dtype :float32)))
    (vt-sum (vt-from-sequence '(16777217 1) :dtype :int64) :out o)
    (check "sum (2^24+1 1) :out f32 = 16777218（int64 提升后计算）"
           (= (vt-item o) 1.6777218e7)))
  (let ((o (make-vt '() 0 :dtype :int32)))
    (vt-mean (vt-from-sequence '(1 2)) :out o)
    (check "mean (1 2) :out int32 = 1" (= (vt-item o) 1)))
  (let ((o (make-vt '() 0 :dtype :float32)))
    (vt-mean (vt-from-sequence '(1 2)) :out o)
    (check "mean (1 2) :out f32 = 1.5" (= (vt-item o) 1.5)))
  (let ((o (make-vt '() 0 :dtype :float64)))
    (vt-var (vt-from-sequence '(1 2 3)) :out o)
    (check "var (1 2 3) :out f64 = 2/3"
           (< (abs (- (vt-item o) #.(coerce 2/3 'double-float))) 1e-12)))
  (let ((o (make-vt '() 0 :dtype :float64)))
    (vt-nanmean (vt-from-sequence (list 1.0d0 +vt-float-nan+ 3.0d0)) :out o)
    (check "nanmean out f64 = 2.0" (= (vt-item o) 2.0d0)))
  (let ((o (make-vt '() 0 :dtype :int8)))
    (vt-sum (vt-arange 5 :dtype :int64) :out o)
    (check "sum arange(5) :out int8 = 10（int64 累加后 cast）" (= (vt-item o) 10)))
  ;; 与无 :out 结果逐位一致
  (let ((direct (vt-item (vt-mean (vt-from-sequence '(1 2 3 4)))))
        (into (make-vt '() 0 :dtype :float64)))
    (vt-mean (vt-from-sequence '(1 2 3 4)) :out into)
    (check "mean :out f64 与无 :out 逐位一致" (= direct (vt-item into))))

  ;; ================================================================
  ;; 3. mod/rem 零除 dtype 语义
  ;; ================================================================
  (format t "~%[3] mod/rem 零除：浮点 → NaN，整型 → 0~%")
  (check "mod 5.0d0 0.0d0 → NaN"
         (%nan (vt-item (vt-mod 5.0d0 0.0d0))))
  (check "rem 5.0d0 0.0d0 → NaN"
         (%nan (vt-item (vt-rem 5.0d0 0.0d0))))
  (check "mod 5 0 → 0（整型语境）"
         (= (vt-item (vt-mod 5 0)) 0))
  (check "rem 5 0 → 0（整型语境）"
         (= (vt-item (vt-rem 5 0)) 0))
  (check "mod 5.0d0 0（混合：浮点语义 NaN）"
         (%nan (vt-item (vt-mod 5.0d0 0))))
  (check "mod (1.0 2.0)/0.0 → 全 NaN"
         (every #'%nan (vt-to-list (vt-mod (vt-from-sequence '(1.0d0 2.0d0)) 0.0d0))))

  ;; ================================================================
  ;; 4. vt-/ true_divide（IEEE 零除 + 整数提升）
  ;; ================================================================
  (format t "~%[4] vt-/ true_divide 语义~%")
  (check "5/0 → +Inf" (= (vt-item (vt-/ 5 0)) (vt-float-pos-inf)))
  (check "-5/0 → -Inf" (= (vt-item (vt-/ -5 0)) (vt-float-neg-inf)))
  (check "0/0 → NaN" (%nan (vt-item (vt-/ 0 0))))
  (check "5/2 = 2.5d0（float64 提升）"
         (and (= (vt-item (vt-/ 5 2)) 2.5d0)
              (eq (vt-dtype (vt-/ 5 2)) :float64)))
  (check "整型输入零除不报错（IEEE）"
         (handler-case (progn (vt-/ (vt-const '(2) 3 :dtype :int64)
                                    (vt-const '(2) 0 :dtype :int64))
                              t)
           (error () nil)))
  (check "float32 标量零除 → +Inf"
         (= (vt-item (vt-/ 1.0s0 0.0s0))
            (vt-float-pos-inf)))

  ;; ================================================================
  ;; 5. random NaN/Inf 参数校验前移
  ;; ================================================================
  (format t "~%[5] random 参数校验（干净参数错误）~%")
  (check-error "uniform NaN low"  (lambda () (vt-random-uniform '(2) :low +vt-float-nan+)))
  (check-error "uniform NaN high" (lambda () (vt-random-uniform '(2) :high +vt-float-nan+)))
  (check-error "uniform inf low"  (lambda () (vt-random-uniform '(2) :low (vt-float-pos-inf))))
  (check-error "normal NaN mean"  (lambda () (vt-random-normal '(2) :mean +vt-float-nan+)))
  (check-error "normal inf std"   (lambda () (vt-random-normal '(2) :std (vt-float-pos-inf))))
  (check-error "normal -inf std"  (lambda () (vt-random-normal '(2) :std (vt-float-neg-inf))))
  (check-error "normal NaN std"   (lambda () (vt-random-normal '(2) :std +vt-float-nan+)))
  (check "normal std=0 → mean 填充"
         (= (vt-item (vt-random-normal nil :mean 5.0d0 :std 0.0d0)) 5.0d0))
  (check "uniform low=high → 常量填充"
         (= (vt-item (vt-random-uniform nil :low 2.5d0 :high 2.5d0)) 2.5d0))

  (format t "~%通过 ~a / 失败 ~a~%" *pass* *fail*)
  (finish-output)
  (zerop *fail*))

(sb-ext:exit :code (if (run) 0 1))
