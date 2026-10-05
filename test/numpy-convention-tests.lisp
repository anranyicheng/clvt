;;;; numpy-convention-tests.lisp — numpy 2.1.3 对齐语义套件（v0.3.6）
;;;;
;;;; 覆盖 v0.3.6 修复的五大 numpy 对齐项（TEST-PLAN §5 重校准 / §6 缺口收敛）：
;;;;   1. 空归约语义：输出空 → 空结果；max/min/arg 族输出非空且归约区空 →
;;;;      ValueError（取代 v0.3.5 的 NaN 填充约定）；单位元族填单位元。
;;;;   2. :out dtype 契约（v0.3.6 起）：结果 dtype 由「输入提升 + 显式
;;;;      :dtype」决定，**不由 out 决定**；out 的 dtype 必须严格相等，
;;;;      否则报错（用户裁决，取代 numpy 原生 out= 的静默 cast 语义）。
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
  ;; 2. :out dtype 契约（v0.3.6 起改为「严格相等」）
  ;; ================================================================
  ;; 旧行为（已弃用）：numpy 原生 out= 语义——按输入提升计算，最后一步
  ;;   cast 写入 out，因此 `:out int32` 配 float64 输入会静默截断。
  ;; 新行为（用户裁决）：结果 dtype 由「输入提升 + 显式 :dtype」决定，
  ;;   **不由 out 决定**；out 的 dtype 必须精确等于结果 dtype，否则报错。
  ;;   需要低精度输出时，显式传 :dtype（并且 out 也用同一 dtype）。
  (format t "~%[2] :out dtype 严格相等契约（v0.3.6）~%")
  ;; 2a) 违反契约 → 必须报错（下面四个 case 覆盖 4 类典型误用）
  (flet ((must-error (label thunk)
           (check label
                  (handler-case (progn (funcall thunk) nil)
                    (error () t)))))
    (must-error "sum f64 → :out int32 报错（禁止静默截断）"
                (lambda () (vt-sum (vt-from-sequence '(0.9d0 0.9d0))
                                   :out (make-vt '() 0 :dtype :int32))))
    (must-error "sum int64 → :out float32 报错（结果 dtype 为 int64）"
                (lambda () (vt-sum (vt-from-sequence '(16777217 1) :dtype :int64)
                                   :out (make-vt '() 0 :dtype :float32))))
    (must-error "sum int64 → :out int8 报错（结果 dtype 为 int64）"
                (lambda () (vt-sum (vt-arange 5 :dtype :int64)
                                   :out (make-vt '() 0 :dtype :int8))))
    (must-error "mean int64 → :out int32 报错（mean 恒为浮点）"
                (lambda () (vt-mean (vt-from-sequence '(1 2))
                                    :out (make-vt '() 0 :dtype :int32)))))
  ;; 2b) 契约满足 → 正确写入（含非连续 out）
  (let ((o (make-vt '() 0 :dtype :float64)))
    (vt-sum (vt-from-sequence '(0.9d0 0.9d0)) :out o)
    (check "sum (0.9 0.9) :out f64 = 1.8" (< (abs (- (vt-item o) 1.8d0)) 1d-12)))
  (let ((o (make-vt '() 0 :dtype :float64)))
    (vt-mean (vt-from-sequence '(1 2)) :out o)
    (check "mean (1 2) :out f64 = 1.5" (= (vt-item o) 1.5d0)))
  (let ((o (make-vt '() 0 :dtype :float32)))
    ;; 想得到 float32 结果，必须显式传 :dtype :float32（此时 out 也必须是 float32）。
    ;; 只传 f32 的 out 而不给 :dtype 会因「严格相等」契约报错——这正是 2a) 的语义。
    (check "mean f64 输入 + 显式 :dtype :float32 → out f32 = 1.5"
           (= (vt-item (vt-mean (vt-from-sequence '(1.0d0 2.0d0))
                                :dtype :float32 :out o))
              1.5)))
  (let ((o (make-vt '() 0 :dtype :int64)))
    (vt-sum (vt-arange 5 :dtype :int64) :out o)
    (check "sum arange(5) :out int64 = 10" (= (vt-item o) 10)))
  (let ((o (make-vt '() 0 :dtype :float64)))
    (vt-var (vt-from-sequence '(1 2 3)) :out o)
    (check "var (1 2 3) :out f64 = 2/3"
           (< (abs (- (vt-item o) #.(coerce 2/3 'double-float))) 1e-12)))
  (let ((o (make-vt '() 0 :dtype :float64)))
    (vt-nanmean (vt-from-sequence (list 1.0d0 +vt-dfloat-nan+ 3.0d0)) :out o)
    (check "nanmean out f64 = 2.0" (= (vt-item o) 2.0d0)))
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
  (check-error "uniform NaN low"  (lambda () (vt-random-uniform '(2) :low +vt-dfloat-nan+)))
  (check-error "uniform NaN high" (lambda () (vt-random-uniform '(2) :high +vt-dfloat-nan+)))
  (check-error "uniform inf low"  (lambda () (vt-random-uniform '(2) :low (vt-float-pos-inf))))
  (check-error "normal NaN mean"  (lambda () (vt-random-normal '(2) :mean +vt-dfloat-nan+)))
  (check-error "normal inf std"   (lambda () (vt-random-normal '(2) :std (vt-float-pos-inf))))
  (check-error "normal -inf std"  (lambda () (vt-random-normal '(2) :std (vt-float-neg-inf))))
  (check-error "normal NaN std"   (lambda () (vt-random-normal '(2) :std +vt-dfloat-nan+)))
  (check "normal std=0 → mean 填充"
         (= (vt-item (vt-random-normal nil :mean 5.0d0 :std 0.0d0)) 5.0d0))
  (check "uniform low=high → 常量填充"
         (= (vt-item (vt-random-uniform nil :low 2.5d0 :high 2.5d0)) 2.5d0))

  ;; ================================================================
  ;; 6. clip / where 对齐 numpy 2.x 签名（本轮新增）
  ;; ================================================================
  (format t "~%[6] clip / where numpy 2.x 签名对齐~%")
  (let ((a (vt-from-sequence '(1.0d0 2.0d0 3.0d0))))
    ;; clip 无边界 → 返回原值（numpy: np.clip(a) == a）
    (check "clip 无边 → 原值"
           (equal (vt-to-list (vt-clip a)) '(1.0d0 2.0d0 3.0d0)))
    (check "clip(nil,nil) → 原值"
           (equal (vt-to-list (vt-clip a nil nil)) '(1.0d0 2.0d0 3.0d0)))
    ;; clip 只给 max（min=nil）→ numpy: np.clip(a,None,2) == [1,2,2]
    (check "clip(nil,2) 只上限"
           (equal (vt-to-list (vt-clip a nil 2.0d0)) '(1.0d0 2.0d0 2.0d0)))
    ;; clip 只给 min（max=nil）→ numpy: np.clip(a,2,None) == [2,2,3]
    (check "clip(2,nil) 只下限"
           (equal (vt-to-list (vt-clip a 2.0d0 nil)) '(2.0d0 2.0d0 3.0d0)))
    ;; clip 双侧
    (check "clip(1.5,2.5) 双侧"
           (equal (vt-to-list (vt-clip a 1.5d0 2.5d0)) '(1.5d0 2.0d0 2.5d0)))
    ;; clip min>max → 全为 max（numpy: np.clip(a,2,2) == [2,2,2]）
    (check "clip(2,2) min>max → 全 max"
           (equal (vt-to-list (vt-clip a 2.0d0 2.0d0)) '(2.0d0 2.0d0 2.0d0)))
    ;; clip 只给 min 不给 max → 报错（numpy TypeError）
    (check-error "clip(2) 只给 min → 报错"
                 (lambda () (vt-clip a 2.0d0)))
    ;; clip 整型 dtype 保持
    (let ((ai (vt-from-sequence '(1 2 3) :dtype :int64)))
      (check "clip int 无边 dtype 保持"
             (eq :int64 (vt-dtype (vt-clip ai))))
      (check "clip int(nil,2) 保 dtype 且截断"
             (and (eq :int64 (vt-dtype (vt-clip ai nil 2)))
                  (equal (vt-to-list (vt-clip ai nil 2)) '(1 2 2))))))

  ;; where 单参 → 返回 (n,rank) 坐标（对齐 np.where(cond)）
  (check "where 单参 1d → (n,1) 坐标"
         (equal (vt-to-list (vt-where (vt-from-sequence '(1 0 1)))) '((0) (2))))
  (check "where 单参 2d → (n,2) 坐标"
         (equal (vt-to-list (vt-where (vt-from-sequence '((1 0) (0 1)))))
                '((0 0) (1 1))))
  (check "where 单参与 argwhere 一致"
         (equal (vt-to-list (vt-where (vt-from-sequence '(1 0 1))))
                (vt-to-list (vt-argwhere (vt-from-sequence '(1 0 1))))))
  ;; where 三元形式回归
  (check "where 三元正常"
         (equal (vt-to-list (vt-where (vt-from-sequence '(1 0 1)) 10 20)) '(10 20 10)))
  ;; where 只给 x → 报错
  (check-error "where 只给 x → 报错"
               (lambda () (vt-where (vt-from-sequence '(1 0 1)) 5)))

  ;; vt-div 整数零除 → 0（对标 np.floor_divide）
  (check "div int 7/0 → 0（不报错）"
         (equal (vt-to-list (vt-div (vt-from-sequence '(7) :dtype :int64) 0)) '(0)))
  (check "div int 7/0 dtype 保持 INT64"
         (eq :int64 (vt-dtype (vt-div (vt-from-sequence '(7) :dtype :int64) 0))))
  (check "div int 7/2 → 3（截断）"
         (equal (vt-to-list (vt-div (vt-from-sequence '(7) :dtype :int64) 2)) '(3)))
  (check "div int -7/3 → -2（截断非 floor）"
         (equal (vt-to-list (vt-div (vt-from-sequence '(-7) :dtype :int64) 3)) '(-2)))

  (format t "~%通过 ~a / 失败 ~a~%" *pass* *fail*)
  (finish-output)
  (zerop *fail*))

(sb-ext:exit :code (if (run) 0 1))
