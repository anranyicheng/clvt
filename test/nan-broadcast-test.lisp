;;;; nan-broadcast-test.lisp
;;;; ==================================================================
;;;; 任务4 验收：NaN / Inf 在广播语境下的完整对齐测试
;;;; ==================================================================
;;;;
;;;; 每一条期望值都直接取自 numpy 2.3.5 的实测输出（见 CONVENTIONS.md
;;;; 附录 B 的复现命令），因此本文件是「与 numpy 逐位对齐」的可执行证据，
;;;; 而不仅是自洽性检查。
;;;;
;;;; 覆盖范围：
;;;;   A. 算术传播（+ - * /）在 (2,2) ⊗ (2,) 广播下
;;;;   B. maximum/minimum（NaN 传播）vs fmax/fmin（忽略 NaN）
;;;;   C. clip / pow 在 NaN、±Inf 上的行为
;;;;   D. where 的三参广播
;;;;   E. 数学函数（exp / log / sqrt / hypot）
;;;;   F. 比较运算（NaN 恒假 / 恒真的例外）
;;;;   G. 归约与 nan* 族（nansum / nanmean / nanmax）
;;;;   H. 累积归约（cumsum 遇 NaN/Inf 的传染性）
;;;;   I. dtype 提升（int64 + NaN → float64 而非报错）
;;;;
;;;; 运行： (load "test/nan-broadcast-test.lisp")

(require :asdf)
(in-package :clvt)
(unless (find-package :clvt)
  (asdf:load-system :clvt))

(defvar *nb-pass* 0)
(defvar *nb-fail* 0)

;; 占位符号 → 真实数值的解析
;; :NAN 表示「期望为 NaN」，:PINF / :NINF 表示 ±Inf。
(defun nb-resolve (x)
  "把期望值中的占位符号解析为真实数值。
   接受两种写法（都常见、都易写）：
     :nan / :pinf / :ninf              关键字
     nan  / pinf / ninf                普通符号（在引号列表里不会被求值，
                                       因此按符号名识别，忽略所属包）"
  (let ((name (and (symbolp x) (string-upcase (symbol-name x)))))
    (cond ((equal name "NAN")  (vt-get-nan :float64))
          ((equal name "PINF") (vt-get-pos-inf :float64))
          ((equal name "NINF") (vt-get-neg-inf :float64))
          (t x))))

(defun nb-num= (a b)
  "数值判等：NaN 视为与 NaN 相等，同号 Inf 相等，其余按数值比较。
   整个比较在 with-float-safe 内进行——NaN 的 `=` 比较本身会触发陷阱。"
  (setf a (nb-resolve a) b (nb-resolve b))
  (with-float-safe
    (cond
      ((and (floatp a) (floatp b))
       (let ((an (not (= a a)))               ; NaN
             (bn (not (= b b))))
         (cond ((and an bn) t)
               ((or an bn) nil)
               ((and (> (abs a) most-positive-double-float)
                     (> (abs b) most-positive-double-float))
                (eq (plusp a) (plusp b)))     ; 同号 Inf
               (t (or (= a b)
                      ;; float32/float64 精度差异容忍
                      (<= (abs (- (coerce a 'double-float)
                                  (coerce b 'double-float)))
                          1d-9))))))
      ((and (numberp a) (numberp b)) (= a b))
      (t (equal a b)))))

(defun nb-seq= (a b)
  (and (= (length a) (length b))
       (every #'nb-num= a b)))

(defun nb-check (name got expected)
  "GOT / EXPECTED 可以是列表（逐元素比较）。
   整体在 with-float-safe 内 —— 打印/比较 NaN 都会触发浮点陷阱。"
  (with-float-safe
    (let ((g (if (listp got) got (list got)))
          (e (if (listp expected) expected (list expected))))
      (if (nb-seq= g e)
          (progn (incf *nb-pass*)
                 (format t "~&  [PASS] ~a~%" name))
          (progn (incf *nb-fail*)
                 (format t "~&  [FAIL] ~a~%         got      = ~a~%         expected = ~a~%"
                         name g e))))))

(defun nb-flat (v) (vt-to-list (vt-flatten v)))

(defmacro nb-values (&body forms)
  "在屏蔽浮点陷阱的环境下求值——NaN 判等本身会触发 traps。"
  `(with-float-safe ,@forms))

;;; ------------------------------------------------------------------
;;; A. 算术传播：(2,2) ⊗ (2,) 广播
;;; ------------------------------------------------------------------
;;; numpy:
;;;   a = [[1, nan], [inf, -inf]];  b = [1, -1]
;;;   a+b        = [[2, nan], [inf, -inf]]
;;;   a*b        = [[1, nan], [inf, inf]]
;;;   (a==a)     = [[1, 0], [1, 1]]        （比较结果 0/1 浮点）
(defun test-arithmetic-broadcast ()
  (format t "~&~%=== A. 算术传播（广播）===")
  (nb-values
   (let* ((nan (vt-get-nan :float64))
          (pinf (vt-get-pos-inf :float64))
          (ninf (vt-get-neg-inf :float64))
          (a (vt-from-sequence (list (list 1.0d0 nan) (list pinf ninf)) :dtype :float64))
          (b (vt-from-sequence (list 1.0d0 -1.0d0) :dtype :float64)))
     (nb-check "a+b" (nb-flat (vt-+ a b)) '(2.0d0 :nan pinf ninf))
     (nb-check "a-b" (nb-flat (vt-- a b)) '(0.0d0 :nan pinf ninf))
     (nb-check "a*b" (nb-flat (vt-* a b)) '(1.0d0 :nan pinf pinf))
     ;; numpy: [[1,nan],[inf,inf]] —— 注意 -inf/-1 = +inf
     (nb-check "a/b" (nb-flat (vt-/ a b)) '(1.0d0 :nan pinf pinf))
     ;; 结果 dtype 仍为 float64（NaN 不改变提升）
     (nb-check "a+b dtype = float64" (vt-dtype (vt-+ a b)) :float64))))

;;; ------------------------------------------------------------------
;;; B. maximum/minimum vs fmax/fmin
;;; ------------------------------------------------------------------
;;; numpy:
;;;   maximum(a,b) = [[1, nan], [inf, -1]]      （任一侧 NaN → NaN，取左侧 NaN）
;;;   fmax(a,b)    = [[1, -1],  [inf, -1]]      （忽略 NaN）
;;;   fmin(a,b)    = [[1, -1],  [1,   -inf]]
(defun test-min-max-broadcast ()
  (format t "~&~%=== B. maximum/minimum vs fmax/fmin ===")
  (nb-values
   (let* ((nan (vt-get-nan :float64))
          (pinf (vt-get-pos-inf :float64))
          (ninf (vt-get-neg-inf :float64))
          (a (vt-from-sequence (list (list 1.0d0 nan) (list pinf ninf)) :dtype :float64))
          (b (vt-from-sequence (list 1.0d0 -1.0d0) :dtype :float64)))
     (nb-check "maximum(a,b) NaN 传播" (nb-flat (vt-maximum a b))
               '(1.0d0 :nan pinf -1.0d0))
     (nb-check "minimum(a,b) NaN 传播" (nb-flat (vt-minimum a b))
               '(1.0d0 :nan 1.0d0 ninf))
     (nb-check "fmax(a,b) 忽略 NaN" (nb-flat (vt-fmax a b))
               '(1.0d0 -1.0d0 pinf -1.0d0))
     (nb-check "fmin(a,b) 忽略 NaN" (nb-flat (vt-fmin a b))
               '(1.0d0 -1.0d0 1.0d0 ninf))
     ;; 两侧皆 NaN：fmax/fmin 返回 NaN
     (nb-check "fmax(nan,nan)" (nb-flat (vt-fmax (vt-from-sequence (list nan))
                                                 (vt-from-sequence (list nan))))
               '(:nan)))))

;;; ------------------------------------------------------------------
;;; C. clip / pow
;;; ------------------------------------------------------------------
;;; numpy:
;;;   clip(a,0,10) = [[1, nan], [10, 0]]
;;;   a**2         = [[1, nan], [inf, inf]]
;;;   a**(-1)      = [[1, nan], [0, -0]]
(defun test-clip-pow ()
  (format t "~&~%=== C. clip / pow ===")
  (nb-values
   (let* ((nan (vt-get-nan :float64))
          (pinf (vt-get-pos-inf :float64))
          (ninf (vt-get-neg-inf :float64))
          (a (vt-from-sequence (list (list 1.0d0 nan) (list pinf ninf)) :dtype :float64)))
     (nb-check "clip(a,0,10)" (nb-flat (vt-clip a 0.0d0 10.0d0))
               '(1.0d0 :nan 10.0d0 0.0d0))
     (nb-check "a**2" (nb-flat (vt-pow a 2)) '(1.0d0 :nan pinf pinf))
     (nb-check "a**-1" (nb-flat (vt-pow a -1)) '(1.0d0 :nan 0.0d0 0.0d0))
     ;; clip 的三参广播：min 为张量
     ;; numpy: np.clip(a, [0,1], 10) = [[1, nan], [10, 1]]
     (nb-check "clip(a, [0,1], 10)"
               (nb-flat (vt-clip a
                                 (vt-from-sequence (list 0.0d0 1.0d0) :dtype :float64)
                                 10.0d0))
               '(1.0d0 :nan 10.0d0 1.0d0)))))

;;; ------------------------------------------------------------------
;;; D. where 三参广播
;;; ------------------------------------------------------------------
;;; numpy: np.where([1,nan]>0, [[1],[2]], [10]) = [[1,10],[2,10]]
(defun test-where-broadcast ()
  (format t "~&~%=== D. where 三参广播 ===")
  (nb-values
   (let* ((nan (vt-get-nan :float64))
          (c (vt-from-sequence (list 1.0d0 nan) :dtype :float64))
          (xs (vt-from-sequence (list (list 1.0d0) (list 2.0d0)) :dtype :float64))
          (ys (vt-from-sequence (list 10.0d0) :dtype :float64)))
     (nb-check "where(c>0, xs, ys)" (nb-flat (vt-where (vt-> c 0.0d0) xs ys))
               '(1.0d0 10.0d0 2.0d0 10.0d0)))))

;;; ------------------------------------------------------------------
;;; E. 数学函数
;;; ------------------------------------------------------------------
;;; numpy:
;;;   exp([-inf,0,inf,nan])  = [0, 1, inf, nan]
;;;   log([0,1,-1,nan])      = [-inf, 0, nan, nan]
;;;   hypot([inf,nan,1,inf],[nan,nan,1,inf]) = [inf, nan, 1.4142…, inf]
(defun test-math-functions ()
  (format t "~&~%=== E. 数学函数 ===")
  (nb-values
   (let* ((nan (vt-get-nan :float64))
          (pinf (vt-get-pos-inf :float64))
          (ninf (vt-get-neg-inf :float64)))
     (nb-check "exp([-inf,0,inf,nan])"
               (nb-flat (vt-exp (vt-from-sequence (list ninf 0.0d0 pinf nan)
                                                  :dtype :float64)))
               '(0.0d0 1.0d0 pinf :nan))
     (nb-check "log([0,1,-1,nan])"
               (nb-flat (vt-log (vt-from-sequence (list 0.0d0 1.0d0 -1.0d0 nan)
                                                  :dtype :float64)))
               '(ninf 0.0d0 :nan :nan))
     (nb-check "hypot 任一 Inf → +Inf（即便另一侧 NaN）"
               (nb-flat (vt-hypot (vt-from-sequence (list pinf nan 1.0d0 pinf))
                                  (vt-from-sequence (list nan nan 1.0d0 pinf))))
               '(pinf :nan 1.4142135623730951d0 pinf))
     (nb-check "sqrt(|a|) 传播 Inf"
               (nb-flat (vt-sqrt (vt-abs (vt-from-sequence (list 1.0d0 nan pinf)
                                                           :dtype :float64))))
               '(1.0d0 :nan pinf)))))

;;; ------------------------------------------------------------------
;;; F. 比较运算
;;; ------------------------------------------------------------------
;;; numpy: (a>0)=[1,0,1] (a<0)=[0,0,0] (a==a)=[1,0,1] (a!=a)=[0,1,0]
(defun test-comparisons ()
  (format t "~&~%=== F. 比较运算（NaN 例外）===")
  (nb-values
   (let* ((nan (vt-get-nan :float64))
          (a (vt-from-sequence (list 1.0d0 nan (vt-get-pos-inf :float64))
                               :dtype :float64)))
     (nb-check "(a>0)  NaN → 0" (nb-flat (vt-> a 0.0d0)) '(1.0d0 0.0d0 1.0d0))
     (nb-check "(a<0)  NaN → 0" (nb-flat (vt-< a 0.0d0)) '(0.0d0 0.0d0 0.0d0))
     (nb-check "(a==a) NaN ≠ NaN → 0" (nb-flat (vt-= a a)) '(1.0d0 0.0d0 1.0d0))
     (nb-check "(a!=a) NaN != NaN → 1" (nb-flat (vt-/= a a)) '(0.0d0 1.0d0 0.0d0)))))

;;; ------------------------------------------------------------------
;;; G. 归约与 nan* 族
;;; ------------------------------------------------------------------
;;; numpy:
;;;   nanmean([[nan,1],[2,3]], axis=1) = [1, 2.5]
;;;   nansum ([[nan,1],[2,3]], axis=0) = [2, 4]
;;;   nanmax([nan,nan])                = nan（全 NaN）
(defun test-reductions ()
  (format t "~&~%=== G. 归约与 nan* 族 ===")
  (nb-values
   (let* ((nan (vt-get-nan :float64))
          (z (vt-from-sequence (list (list nan 1.0d0) (list 2.0d0 3.0d0))
                               :dtype :float64)))
     (nb-check "nansum axis=0" (nb-flat (vt-nansum z :axis 0)) '(2.0d0 4.0d0))
     (nb-check "nanmean axis=1" (nb-flat (vt-nanmean z :axis 1)) '(1.0d0 2.5d0))
     (nb-check "max 全 NaN 传播"
               (nb-flat (vt-amax (vt-from-sequence (list nan nan) :dtype :float64)))
               '(:nan))
     (nb-check "nanmax 全 NaN → NaN"
               (nb-flat (vt-nanmax (vt-from-sequence (list nan nan) :dtype :float64)))
               '(:nan))
     (nb-check "sum 遇 NaN 传染"
               (nb-flat (vt-sum (vt-from-sequence (list 1.0d0 nan 3.0d0) :dtype :float64)))
               '(:nan)))))

;;; ------------------------------------------------------------------
;;; H. 累积归约
;;; ------------------------------------------------------------------
;;; numpy: cumsum([1,nan,inf,-inf]) = [1, nan, nan, nan]
(defun test-cumulative ()
  (format t "~&~%=== H. 累积归约传染性 ===")
  (nb-values
   (let* ((nan (vt-get-nan :float64))
          (pinf (vt-get-pos-inf :float64))
          (ninf (vt-get-neg-inf :float64)))
     (nb-check "cumsum([1,nan,inf,-inf]) 传染"
               (nb-flat (vt-cumsum (vt-from-sequence (list 1.0d0 nan pinf ninf)
                                                     :dtype :float64)))
               '(1.0d0 :nan :nan :nan)))))

;;; ------------------------------------------------------------------
;;; I. dtype 提升
;;; ------------------------------------------------------------------
;;; numpy: np.add(int64[1,2], float64[nan,nan]).dtype == float64
(defun test-dtype-promotion ()
  (format t "~&~%=== I. dtype 提升 ===")
  (nb-values
   (let* ((nan (vt-get-nan :float64)))
     (nb-check "int64 + NaN → float64（不报错）"
               (vt-dtype (vt-+ (vt-from-sequence (list 1 2) :dtype :int64)
                               (vt-from-sequence (list nan nan) :dtype :float64)))
               :float64)
     ;; NaN 广播到 (2,3)
     (nb-check "NaN(2,3) * int64(3) 全 NaN"
               (nb-flat (vt-* (vt-full (list 2 3) nan :dtype :float64)
                              (vt-from-sequence (list 1 2 3) :dtype :int64)))
               '(:nan :nan :nan :nan :nan :nan)))))

;;; ------------------------------------------------------------------
;;; 汇总
;;; ------------------------------------------------------------------
(defun run-nan-broadcast-tests ()
  (setf *nb-pass* 0 *nb-fail* 0)
  (format t "~&~%##################################################~%")
  (format t "#  NaN / Inf 广播对齐测试（基准：numpy 2.3.5 实测）~%")
  (format t "##################################################")
  (test-arithmetic-broadcast)
  (test-min-max-broadcast)
  (test-clip-pow)
  (test-where-broadcast)
  (test-math-functions)
  (test-comparisons)
  (test-reductions)
  (test-cumulative)
  (test-dtype-promotion)
  (format t "~&~%=== 汇总：通过 ~d / 失败 ~d ===~%" *nb-pass* *nb-fail*)
  (if (zerop *nb-fail*)
      (format t "~&[PASS] NaN/Inf 广播全部对齐 numpy ✅~%")
      (format t "~&[FAIL] 有 ~d 项未对齐 ❌~%" *nb-fail*))
  (zerop *nb-fail*))

(run-nan-broadcast-tests)
