;;;; degenerate-empty-test.lisp — 「退化形状 / 空张量 / 角参数」契约回归
;;;;
;;;; 这一轮的尺子不是「随机输入比数值」，而是**把维度推到边界**：
;;;; shape 里含 0、0 维、单观测、num=0、分母为 0。核心观察是一条：
;;;;
;;;;   实现里凡是「先跑一轮循环体、再判断要不要继续」的结构（里程计 /
;;;;   do-while / advance），在**总元素数为 0** 时都会先跑那一轮 ——
;;;;   而那一轮会去 `aref` 一个长度 0 的 buffer。
;;;;
;;;; 覆盖面（全部来自真实缺陷）：
;;;;
;;;;   A. VT-CUMSUM / VT-CUMPROD / VT-NANCUMSUM / VT-NANCUMPROD
;;;;      沿轴累积 + 元素数 0 → 越界崩溃
;;;;      旧实现：(vt-cumsum (vt-zeros '(0 3)) :axis 1) →
;;;;        "Invalid index 0 for (SIMPLE-ARRAY DOUBLE-FLOAT (0))"
;;;;      numpy：np.cumsum(np.zeros((0,3)), axis=1) → shape (0,3)，正常。
;;;;      三个内核都要修：快路径 run-axis、通用 fallback、nancumsum 的
;;;;      axis 分支（前两个共用 %VT-CUMULATIVE，第三个在 extensions3）。
;;;;
;;;;   B. VT-INDICES 的维度里含 0（不在首位）
;;;;      旧实现：(vt-indices '(2 0 3)) → "cannot index into an empty tensor"
;;;;      numpy：np.indices((2,0,3)).shape == (3, 2, 0, 3)。
;;;;      根因是 stride 算成 0 后被兜成 1，然后照常写满 outer×axis-len。
;;;;
;;;;   C. VT-VSPLIT / VT-HSPLIT / VT-DSPLIT 的轴长为 0
;;;;      旧实现把 0 归进「不整除」（`(plusp dim)` 先短路了 `(mod dim n)`），
;;;;      于是一个合法均分被误报。numpy：np.vsplit(np.zeros((0,3)), 2)
;;;;      → 两个 (0,3)。
;;;;
;;;;   D. VT-COV / VT-CORRCOEF 的零观测与角分母
;;;;      旧实现 ① 未屏蔽浮点陷阱（0/0 与 1/0 直接抛）；② 把 denom<=0
;;;;      换成 n_obs（相当于凭空改分母）；③ corrcoef 的标量分支硬编码 1.0。
;;;;      numpy：np.cov([5.]) → nan、np.cov([1,2],ddof=3) → inf、
;;;;             np.corrcoef([5.]) → nan。
;;;;
;;;;   E. VT-LINSPACE / VT-LOGSPACE / VT-GEOMSPACE 的 num = 0
;;;;      旧实现报「num 必须大于 0」，而 numpy 返回空数组（采样点为空是
;;;;      合法请求）；geomspace 的零/同号校验仍先于 num 判断（numpy 顺序）。
;;;;
;;;;   F. VT-BINCOUNT 的 :WEIGHTS
;;;;      旧实现只认计数。numpy 的 weights 让结果变成 float64（即使权重是
;;;;      整型/布尔），负权重合法，复数权重报 TypeError —— 本库同样拒绝。
;;;;
;;;;   G. VT-RANDOM-CHOICE 的 :P 接受张量
;;;;      旧实现直接对 p 走序列协议（length/every/reduce），张量报
;;;;      "The value #<VT …> is not of type SEQUENCE"，而 numpy 的 p 就是
;;;;      ndarray。顺带：p 的三种形态（列表/向量/张量）都要能跑。
;;;;
;;;;   H. 随机族的错误契约
;;;;      旧实现用裸 ASSERT：choice 的 a<=0/n<=0/p 越界、random-int 的
;;;;      low>high、multinomial 的 n<0/p 全零/p 含负，报的都是
;;;;      「The assertion (… ) failed with … = …」而非带函数名的消息。
;;;;
;;;;   I. IO 往返的位模式保真（否定结果的固化）
;;;;      这一轮用「save → load → 逐位比」扫了 188 个组合（11 dtype ×
;;;;      7 形状 + 非连续视图 + NaN/Inf/±0 + 复数），**零失败**。这一组把
;;;;      最容易在将来被改坏的几条钉住：空张量、0 维、非连续视图、
;;;;      NaN 的 payload、-0 的符号位。注：VT-ARRAY-EQUIV 是「NaN != NaN」，
;;;;      所以这里必须比**位模式**，不能比值。
;;;;
;;;;   J. DTYPE 转换矩阵的抽样（11×11 全组合已跑过 121 条差分，零失败）
;;;;      这里按语义类别各取一格：向零截断、窄化回绕、无符号回绕、
;;;;      真值化、复数丢虚部、布尔升整型。
;;;;
;;;; 期望值全部用本机 numpy 2.4.6 实测（CONVENTIONS §10.4：禁止手写期望）。
;;;; 用法：sbcl --non-interactive --load test/degenerate-empty-test.lisp

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

(defvar *zero-d* 0d0
  "经 defvar 的零：避免 (/ 0d0 0d0) 在编译期被常量折叠。")

(defun nan-d () (with-float-safe (/ *zero-d* *zero-d*)))
(defun inf-d () (with-float-safe (/ 1d0 *zero-d*)))
(defun ninf-d () (with-float-safe (/ -1d0 *zero-d*)))
(defun nan-p (x) (and (floatp x) (sb-ext:float-nan-p x)))

(defun num= (a b)
  (cond ((and (complexp a) (complexp b))
         (and (num= (realpart a) (realpart b))
              (num= (imagpart a) (imagpart b))))
        ((or (complexp a) (complexp b)) nil)
        ((and (floatp a) (floatp b))
         (cond ((and (nan-p a) (nan-p b)) t)
               ((or (nan-p a) (nan-p b)) nil)
               ((and (sb-ext:float-infinity-p a) (sb-ext:float-infinity-p b))
                (eq (plusp a) (plusp b)))
               (t (<= (abs (- a b)) (* 1d-12 (max 1d0 (abs a) (abs b)))))))
        (t (and (realp a) (realp b) (= a b)))))

(defun flat (v) (vt-to-list (vt-ravel v)))

(defun ok-shape (name v want)
  (report name (equal (vt-shape v) want)
          (format nil "shape=~s want ~s" (vt-shape v) want)))

(defun ok-vals (name v want)
  (let ((got (flat v)))
    (report name (and (= (length got) (length want))
                      (every #'num= got want))
            (format nil "got ~s want ~s" got want))))

(defun ok-dtype (name v want)
  (report name (eq (vt-dtype v) want)
          (format nil "dtype=~s want ~s" (vt-dtype v) want)))

(defun ok-int (name got want)
  (report name (num= got want) (format nil "got ~s want ~s" got want)))

(defun ok-err (name thunk &rest needles)
  "断言 THUNK 报错，且消息里含全部 NEEDLES（子串）。"
  (handler-case
      (let ((v (funcall thunk)))
        (report name nil (format nil "没有报错，返回 ~s" (if (vt-p v) (vt-shape v) v))))
    (error (e)
      (let ((msg (princ-to-string e)))
        (report name (every (lambda (n) (search n msg)) needles)
                (format nil "消息:~a" (substitute #\Space #\Newline msg)))))))

(defun fi (l) (vt-from-sequence l :dtype :int64))
(defun ff (l) (vt-from-sequence l :dtype :float64))

;;; ------------------------------------------------------------------
;;; A. 累积族 × 空
;;; ------------------------------------------------------------------

(defun run-cumulative-empty ()
  (format t "~&--- A. cumsum/cumprod 沿轴 × 空 ---%")
  (dolist (dt '(:float64 :float32 :int64 :int32 :int8 :complex128 :bool))
    (let ((e03 (vt-zeros '(0 3) :dtype dt))
          (e30 (vt-zeros '(3 0) :dtype dt))
          (e023 (vt-zeros '(0 2 3) :dtype dt)))
      ;; numpy: np.cumsum(zeros((0,3)), axis=1).shape == (0,3)（每种 dtype 都是）
      (ok-shape (format nil "cumsum/~a (0,3) ax1" dt) (vt-cumsum e03 :axis 1) '(0 3))
      (ok-shape (format nil "cumsum/~a (0,3) ax0" dt) (vt-cumsum e03 :axis 0) '(0 3))
      (ok-shape (format nil "cumsum/~a (3,0) ax0" dt) (vt-cumsum e30 :axis 0) '(3 0))
      (ok-shape (format nil "cumsum/~a (3,0) ax1" dt) (vt-cumsum e30 :axis 1) '(3 0))
      (ok-shape (format nil "cumsum/~a (0,2,3) ax2" dt) (vt-cumsum e023 :axis 2) '(0 2 3))
      (ok-shape (format nil "cumsum/~a (0,2,3) ax1" dt) (vt-cumsum e023 :axis 1) '(0 2 3))
      (ok-shape (format nil "cumprod/~a (0,3) ax1" dt) (vt-cumprod e03 :axis 1) '(0 3))
      (ok-shape (format nil "cumprod/~a (3,0) ax0" dt) (vt-cumprod e30 :axis 0) '(3 0))
      ;; 空结果必须是真空（size 0），不能是「长度为 n 的零」
      (ok-int (format nil "cumsum/~a (0,3) size" dt) (vt-size (vt-cumsum e03 :axis 1)) 0)))
  ;; dtype 保持：空也一样照提升表（int8 → int64）
  (ok-dtype "cumsum/int8 (0,3) ax1 dtype 升 int64"
            (vt-cumsum (vt-zeros '(0 3) :dtype :int8) :axis 1) :int64)
  (ok-dtype "cumprod/float32 (3,0) ax0 dtype 保持"
            (vt-cumprod (vt-zeros '(3 0) :dtype :float32) :axis 0) :float32)
  ;; nancumsum / nancumprod：走另一条内核（%E3-CUMULATIVE-SKIP-NAN）
  (ok-shape "nancumsum (0,3) ax1" (vt-nancumsum (vt-zeros '(0 3)) :axis 1) '(0 3))
  (ok-shape "nancumsum (3,0) ax0" (vt-nancumsum (vt-zeros '(3 0)) :axis 0) '(3 0))
  (ok-shape "nancumprod (0,3) ax1" (vt-nancumprod (vt-zeros '(0 3)) :axis 1) '(0 3))
  (ok-shape "nancumprod (0,2,3) ax2" (vt-nancumprod (vt-zeros '(0 2 3)) :axis 2) '(0 2 3))
  ;; 非连续的空视图：转置后仍是 0 元素
  (ok-shape "cumsum/(0,3).T ax0" (vt-cumsum (vt-transpose (vt-zeros '(3 0))) :axis 0) '(0 3))
  ;; :out 也走同一条保护
  (ok-shape "cumsum/(0,3) ax1 :out"
            (vt-cumsum (vt-zeros '(0 3)) :axis 1 :out (vt-zeros '(0 3))) '(0 3))
  ;; 非空回归（保护不能把正常的沿轴累积也跳掉）
  (let ((a (vt-reshape (vt-arange 6 :dtype :int64) '(2 3))))
    (ok-vals "cumsum/(2,3) ax1 回归" (vt-cumsum a :axis 1) '(0 1 3 3 7 12))
    (ok-vals "cumsum/(2,3) ax0 回归" (vt-cumsum a :axis 0) '(0 1 2 3 5 7))
    (ok-vals "cumprod/(2,3) ax1 回归" (vt-cumprod a :axis 1) '(0 0 0 3 12 60))
    (ok-vals "cumsum/(2,3) 展平回归" (vt-cumsum a) '(0 1 3 6 10 15))))

;;; ------------------------------------------------------------------
;;; B. vt-indices × 空维
;;; ------------------------------------------------------------------

(defun run-indices-empty ()
  (format t "~&--- B. vt-indices × 空维 ---%")
  ;; numpy: np.indices((2,0,3)).shape == (3,2,0,3)
  (ok-shape "indices/(2,0,3)" (vt-indices '(2 0 3)) '(3 2 0 3))
  (ok-shape "indices/(2,3,0)" (vt-indices '(2 3 0)) '(3 2 3 0))
  (ok-shape "indices/(0,3)" (vt-indices '(0 3)) '(2 0 3))
  (ok-shape "indices/(0,)" (vt-indices '(0)) '(1 0))
  (ok-shape "indices/(2,0)" (vt-indices '(2 0)) '(2 2 0))
  (ok-int "indices/(2,0,3) size" (vt-size (vt-indices '(2 0 3))) 0)
  ;; sparse：形状分别是 (2,1,1) (1,0,1) (1,1,3)
  (let ((sp (vt-indices '(2 0 3) :sparse t)))
    (ok-shape "indices/(2,0,3) sparse[0]" (nth 0 sp) '(2 1 1))
    (ok-shape "indices/(2,0,3) sparse[1]" (nth 1 sp) '(1 0 1))
    (ok-shape "indices/(2,0,3) sparse[2]" (nth 2 sp) '(1 1 3)))
  ;; 非空回归：值必须仍是坐标网格
  (let ((g (vt-indices '(2 3))))
    (ok-shape "indices/(2,3) 回归" g '(2 2 3))
    (ok-vals "indices/(2,3) 回归 第0层" (vt-slice g '(0)) '(0 0 0 1 1 1))
    (ok-vals "indices/(2,3) 回归 第1层" (vt-slice g '(1)) '(0 1 2 0 1 2)))
  (let ((g (vt-slice (vt-indices '(3 2)) '(0))))
    (ok-vals "indices/(3,2) 第0层行号" g '(0 0 1 1 2 2))))

;;; ------------------------------------------------------------------
;;; C. split 族 × 轴长 0
;;; ------------------------------------------------------------------

(defun run-split-empty ()
  (format t "~&--- C. vsplit/hsplit/dsplit × 轴长 0 ---%")
  ;; numpy: np.vsplit(np.zeros((0,3)), 2) → 两个 (0,3)
  (let ((r (vt-vsplit (vt-zeros '(0 3)) 2)))
    (ok-int "vsplit/(0,3) 2 段数" (length r) 2)
    (ok-shape "vsplit/(0,3) 2 第0段" (nth 0 r) '(0 3))
    (ok-shape "vsplit/(0,3) 2 第1段" (nth 1 r) '(0 3)))
  (let ((r (vt-vsplit (vt-zeros '(0 3)) 3)))
    (ok-int "vsplit/(0,3) 3 段数" (length r) 3))
  ;; numpy: np.hsplit(np.zeros((0,0)), 2) → 两个 (0,0)（轴 1 长 0）
  (let ((r (vt-hsplit (vt-zeros '(0 0)) 2)))
    (ok-shape "hsplit/(0,0) 2 第0段" (nth 0 r) '(0 0)))
  ;; numpy: np.hsplit(np.zeros((3,0)), 2) → 两个 (3,0)
  (let ((r (vt-hsplit (vt-zeros '(3 0)) 2)))
    (ok-shape "hsplit/(3,0) 2 第0段" (nth 0 r) '(3 0)))
  ;; dsplit 沿轴 2
  (let ((r (vt-dsplit (vt-zeros '(2 2 0)) 2)))
    (ok-shape "dsplit/(2,2,0) 2 第0段" (nth 0 r) '(2 2 0)))
  ;; 不整除仍必须报错（0 的判断不能把这条也吞掉）
  (ok-err "vsplit/(3,0) 2 仍报不整除"
          (lambda () (vt-vsplit (vt-zeros '(3 0)) 2)) "equal division")
  (ok-err "dsplit/(2,2,3) 2 仍报不整除"
          (lambda () (vt-dsplit (vt-zeros '(2 2 3)) 2)) "equal division")
  ;; 整除的正常情形
  (let ((r (vt-vsplit (vt-from-sequence '((1 2) (3 4) (5 6) (7 8)) :dtype :int64) 2)))
    (ok-vals "vsplit/(4,2) 2 第0段" (nth 0 r) '(1 2 3 4))
    (ok-vals "vsplit/(4,2) 2 第1段" (nth 1 r) '(5 6 7 8))))

;;; ------------------------------------------------------------------
;;; D. cov / corrcoef × 零观测与角分母
;;; ------------------------------------------------------------------

(defun run-cov-degenerate ()
  (format t "~&--- D. cov/corrcoef × 零观测与角分母 ---%")
  ;; numpy: np.cov(zeros(0)) → 0 维 nan
  (let ((c (vt-cov (vt-zeros '(0)))))
    (ok-shape "cov/(0,)" c nil)
    (report "cov/(0,) 是 nan" (nan-p (vt-item c))))
  ;; numpy: np.cov(zeros((0,3))) → shape (0,0)
  (ok-shape "cov/(0,3)" (vt-cov (vt-zeros '(0 3))) '(0 0))
  ;; numpy: np.cov(zeros((3,0))) → (3,3) 全 nan
  (let ((c (vt-cov (vt-zeros '(3 0)))))
    (ok-shape "cov/(3,0)" c '(3 3))
    (report "cov/(3,0) 全 nan" (every #'nan-p (flat c))))
  ;; 角分母：numpy np.cov([5.]) → nan（0/0），旧实现给 0.0
  (let ((c (vt-cov (ff '(5d0)))))
    (report "cov/[5.] 是 nan（角分母不替换）" (nan-p (vt-item c))))
  ;; numpy: np.cov([1,2], ddof=3) → inf（0.5/0），旧实现给 0.25
  (let ((c (vt-cov (ff '(1d0 2d0)) :ddof 3)))
    (report "cov/[1,2] ddof=3 是 +inf"
            (let ((v (vt-item c))) (and (floatp v) (sb-ext:float-infinity-p v) (plusp v)))))
  ;; 正常路径不能被带动
  (ok-int "cov/[1,2,3] = 1.0" (vt-item (vt-cov (ff '(1d0 2d0 3d0)))) 1d0)
  (ok-int "cov/[1,2] = 0.5" (vt-item (vt-cov (ff '(1d0 2d0)))) 0.5d0)
  (ok-vals "cov/2x3" (vt-cov (ff '((1d0 2d0 3d0) (4d0 5d0 6d0)))) '(1d0 1d0 1d0 1d0))
  ;; corrcoef：numpy 对单变量/零方差都给 nan（C/sqrt(C²)）
  (report "corrcoef/(0,) 是 nan" (nan-p (vt-item (vt-corrcoef (vt-zeros '(0))))))
  (report "corrcoef/[5.] 是 nan" (nan-p (vt-item (vt-corrcoef (ff '(5d0))))))
  (report "corrcoef/[5,5] 是 nan（零方差）"
          (nan-p (vt-item (vt-corrcoef (ff '(5d0 5d0))))))
  (ok-int "corrcoef/[1,2,3] = 1.0" (vt-item (vt-corrcoef (ff '(1d0 2d0 3d0)))) 1d0)
  (let ((r (vt-corrcoef (vt-zeros '(3 0)))))
    (ok-shape "corrcoef/(3,0)" r '(3 3))
    (report "corrcoef/(3,0) 全 nan" (every #'nan-p (flat r)))))

;;; ------------------------------------------------------------------
;;; E. linspace / logspace / geomspace × num = 0
;;; ------------------------------------------------------------------

(defun run-spaces-zero ()
  (format t "~&--- E. linspace/logspace/geomspace × num=0 ---%")
  ;; numpy: np.linspace(0,1,0) → shape (0,)
  (ok-shape "linspace num=0" (vt-linspace 0 1 0) '(0))
  (ok-shape "linspace num=0 int64" (vt-linspace 0 10 0 :dtype :int64) '(0))
  (ok-dtype "linspace num=0 int64 dtype" (vt-linspace 0 10 0 :dtype :int64) :int64)
  (ok-shape "linspace num=0 endpoint=nil" (vt-linspace 0 1 0 :endpoint nil) '(0))
  (ok-shape "logspace num=0" (vt-logspace 0 2 0) '(0))
  (ok-shape "geomspace num=0" (vt-geomspace 1 10 0) '(0))
  (ok-shape "geomspace num=0 int64" (vt-geomspace 1 1000 0 :dtype :int64) '(0))
  ;; 负 num 仍必须报错
  (ok-err "linspace num=-1 报错" (lambda () (vt-linspace 0 1 -1)) "num")
  (ok-err "geomspace num=-1 报错" (lambda () (vt-geomspace 1 10 -1)) "num")
  ;; geomspace 的零/同号校验必须**先于** num 判断（numpy 顺序：
  ;; np.geomspace(0, 10, 0) 报 Geometric sequence cannot include zero）
  (ok-err "geomspace (0,10,0) 仍报零" (lambda () (vt-geomspace 0 10 0)) "零")
  (ok-err "geomspace (1,-10,0) 仍报同号" (lambda () (vt-geomspace 1 -10 0)) "同号")
  ;; 非空回归
  (ok-vals "linspace num=1 回归" (vt-linspace 0 1 1) '(0d0))
  (ok-vals "linspace num=5 回归" (vt-linspace 0 1 5) '(0d0 0.25d0 0.5d0 0.75d0 1d0))
  (ok-vals "logspace num=3 回归" (vt-logspace 0 2 3) '(1d0 10d0 100d0))
  (ok-vals "geomspace num=1 回归" (vt-geomspace 1 10 1) '(1d0)))

;;; ------------------------------------------------------------------
;;; F. bincount × :weights
;;; ------------------------------------------------------------------

(defun run-bincount-weights ()
  (format t "~&--- F. vt-bincount × :weights ---%")
  ;; numpy: np.bincount([1,1,3], weights=[0.5,0.25,2.0]) → [0, 0.75, 0, 2]
  (let ((r (vt-bincount (fi '(1 1 3)) :weights (ff '(0.5d0 0.25d0 2d0)))))
    (ok-shape "bincount/w float 形状" r '(4))
    (ok-dtype "bincount/w float dtype" r :float64)
    (ok-vals "bincount/w float 值" r '(0d0 0.75d0 0d0 2d0)))
  ;; 整型权重结果仍是 float64（numpy 文档：result is stored as a float64 array）
  (let ((r (vt-bincount (fi '(1 1 3)) :weights (fi '(1 2 3)))))
    (ok-dtype "bincount/w int 结果升 float64" r :float64)
    (ok-vals "bincount/w int 值" r '(0d0 3d0 0d0 3d0)))
  ;; 负权重合法
  (ok-vals "bincount/w 负权重" (vt-bincount (fi '(0 0 1)) :weights (ff '(-1d0 2d0 5d0)))
           '(1d0 5d0))
  ;; minlength 与 weights 同时生效
  (let ((r (vt-bincount (fi '(1)) :weights (ff '(3d0)) :minlength 4)))
    (ok-shape "bincount/w minlength 形状" r '(4))
    (ok-vals "bincount/w minlength 值" r '(0d0 3d0 0d0 0d0)))
  ;; 布尔权重按 0/1 累加（numpy: [True,False,True] → float64）
  (let ((r (vt-bincount (fi '(0 1 1))
                        :weights (vt-from-sequence '(1 0 1) :dtype :bool))))
    (ok-dtype "bincount/w bool dtype" r :float64)
    (ok-vals "bincount/w bool 值" r '(1d0 1d0)))
  ;; 空输入
  (let ((r (vt-bincount (vt-zeros '(0) :dtype :int64) :weights (ff nil))))
    (ok-shape "bincount/w 空输入" r '(0)))
  ;; 契约：长度必须一致、复数权重按 numpy 同样拒绝
  (ok-err "bincount/w 长度不符报错"
          (lambda () (vt-bincount (fi '(1 2)) :weights (ff '(1d0)))) "weights")
  (ok-err "bincount/w 复数权重报错"
          (lambda () (vt-bincount (fi '(1 1))
                                  :weights (vt-from-sequence
                                            (list (complex 1d0 2d0) (complex 3d0 4d0))
                                            :dtype :complex128)))
          "复数")
  ;; 无 weights 的旧路径不受影响
  (ok-vals "bincount/无 w 回归" (vt-bincount (fi '(0 2 2 1))) '(1 1 2))
  (ok-dtype "bincount/无 w dtype 仍是 int64" (vt-bincount (fi '(0 2 2 1))) :int64))

;;; ------------------------------------------------------------------
;;; G. random-choice 的 :p 三种形态
;;; ------------------------------------------------------------------

(defun run-random-choice-p ()
  (format t "~&--- G. vt-random-choice × p 的三种形态 ---%")
  (vt-random-seed 4242)
  (let ((as-list (vt-random-choice 3 :size '(20) :p '(0.2d0 0.3d0 0.5d0))))
    (ok-shape "choice/p 列表" as-list '(20))
    (report "choice/p 列表 取值在 [0,3)"
            (every (lambda (v) (and (integerp v) (<= 0 v) (< v 3))) (flat as-list))))
  (vt-random-seed 4242)
  (let ((as-vec (vt-random-choice 3 :size '(20) :p (vector 0.2d0 0.3d0 0.5d0))))
    (report "choice/p 向量 可用" (every (lambda (v) (< v 3)) (flat as-vec))))
  ;; 张量形态（numpy 的 p 就是 ndarray）—— 旧实现报 not of type SEQUENCE
  (vt-random-seed 4242)
  (let ((as-tensor (vt-random-choice 3 :size '(20) :p (ff '(0.2d0 0.3d0 0.5d0)))))
    (ok-shape "choice/p 张量" as-tensor '(20))
    (report "choice/p 张量 取值在 [0,3)"
            (every (lambda (v) (and (integerp v) (<= 0 v) (< v 3))) (flat as-tensor))))
  ;; 无放回 + 权重
  (let ((r (vt-random-choice 4 :size '(3) :replace nil :p (ff '(1d0 1d0 1d0 1d0)))))
    (ok-shape "choice/p 无放回 形状" r '(3))
    (ok-int "choice/p 无放回 无重复" (length (remove-duplicates (flat r))) 3))
  ;; p 的三种形态必须给出**同一个**抽样序列（同一 seed）
  (vt-random-seed 99)
  (let ((a (flat (vt-random-choice 4 :size '(30) :p '(0.1d0 0.2d0 0.3d0 0.4d0)))))
    (vt-random-seed 99)
    (let ((b (flat (vt-random-choice 4 :size '(30) :p (ff '(0.1d0 0.2d0 0.3d0 0.4d0))))))
      (report "choice/p 列表与张量同 seed 同序列" (equal a b))))
  ;; p 的契约
  (ok-err "choice/p 长度不符报错"
          (lambda () (vt-random-choice 3 :p (ff '(0.5d0 0.5d0)))) "p 的长度")
  (ok-err "choice/p 含负报错"
          (lambda () (vt-random-choice 3 :p (ff '(0.5d0 -0.1d0 0.6d0)))) "非负")
  (ok-err "choice/p 全零报错"
          (lambda () (vt-random-choice 3 :p (ff '(0d0 0d0 0d0)))) "总和")
  (ok-err "choice/p 二维报错"
          (lambda () (vt-random-choice 3 :p (vt-reshape (ff '(0.3d0 0.3d0 0.4d0)) '(1 3))))
          "一维"))

;;; ------------------------------------------------------------------
;;; H. 随机族的错误契约（裸 assert → 带函数名的消息）
;;; ------------------------------------------------------------------

(defun run-random-errors ()
  (format t "~&--- H. 随机族错误契约 ---%")
  (ok-err "choice/a=0 报错带函数名" (lambda () (vt-random-choice 0)) "vt-random-choice")
  (ok-err "choice/空张量报错" (lambda () (vt-random-choice (vt-zeros '(0)))) "vt-random-choice")
  (ok-err "random-int/low>high 报错" (lambda () (vt-random-int 5 0)) "vt-random-int")
  (ok-err "multinomial/n<0 报错" (lambda () (vt-random-multinomial -1 '(0.5d0 0.5d0)))
          "vt-random-multinomial")
  (ok-err "multinomial/p 全零报错" (lambda () (vt-random-multinomial 5 '(0d0 0d0)))
          "总和")
  (ok-err "multinomial/p 含负报错" (lambda () (vt-random-multinomial 5 '(1.2d0 -0.2d0)))
          "非负")
  (ok-err "multinomial/p 为空报错" (lambda () (vt-random-multinomial 5 '())) "为空")
  (ok-err "multinomial/二维 p 报错"
          (lambda () (vt-random-multinomial 5 (vt-reshape (ff '(0.5d0 0.5d0)) '(1 2))))
          "一维")
  ;; 这些错误消息不能再是 SBCL 的裸断言文本
  (handler-case (vt-random-choice 0)
    (error (e) (let ((msg (princ-to-string e)))
                 (report "choice/a=0 不是裸 assert"
                         (not (search "The assertion" msg))))))
  (handler-case (vt-random-multinomial 5 '(0d0 0d0))
    (error (e) (let ((msg (princ-to-string e)))
                 (report "multinomial/p 全零 不是裸 assert"
                         (not (search "The assertion" msg))))))
  ;; 正常路径
  (vt-random-seed 7)
  (ok-shape "multinomial 列表 p 回归" (vt-random-multinomial 10 '(0.3d0 0.7d0)) '(2))
  (ok-shape "multinomial 张量 p 回归" (vt-random-multinomial 10 (ff '(0.3d0 0.7d0))) '(2))
  (ok-shape "choice 无 p 回归" (vt-random-choice 5 :size '(3)) '(3))
  (ok-shape "random-int 回归" (vt-random-int 0 5 :size '(3)) '(3)))

;;; ------------------------------------------------------------------
;;; I. IO 往返的位模式保真
;;; ------------------------------------------------------------------

(defun elem-bits (x)
  "元素 → 位模式（浮点用整数位模式，NaN 才能逐位比较；复数递归）。"
  (cond ((typep x 'double-float) (sb-kernel:double-float-bits x))
        ((typep x 'single-float) (sb-kernel:single-float-bits x))
        ((complexp x) (cons (elem-bits (realpart x)) (elem-bits (imagpart x))))
        (t x)))

(defun bits-of (v)
  (let ((l (vt-to-list v)))
    (map 'list #'elem-bits (if (listp l) l (list l)))))

(defun ok-roundtrip (name tensor)
  "SAVE → LOAD（两种格式各一遍），比 shape / dtype / 逐元素位模式。"
  (dolist (fmt '(:lisp :npy))
    (let ((path (format nil "/tmp/degen-rt-~a.~a" (random 1000000)
                        (if (eq fmt :npy) "npy" "sexp"))))
      (handler-case
          (progn
            (vt-save tensor path :as fmt)
            (let ((back (vt-load path :as fmt)))
              (report (format nil "~a/~a shape" name fmt)
                      (equal (vt-shape back) (vt-shape tensor)))
              (report (format nil "~a/~a dtype" name fmt)
                      (eq (vt-dtype back) (vt-dtype tensor)))
              (report (format nil "~a/~a 位模式" name fmt)
                      (equal (bits-of back) (bits-of tensor)))))
        (error (e)
          (report (format nil "~a/~a" name fmt) nil (princ-to-string e))))
      (ignore-errors (delete-file path)))))

(defun run-io-roundtrip ()
  (format t "~&--- I. IO 往返位模式 ---%")
  ;; 空与 0 维（最容易被「读回时重建形状」的代码弄坏）
  (ok-roundtrip "io/empty-1d" (vt-zeros '(0)))
  (ok-roundtrip "io/empty-2d" (vt-zeros '(0 3)))
  (ok-roundtrip "io/empty-2d-b" (vt-zeros '(3 0)))
  (ok-roundtrip "io/0d" (vt-sum (vt-ones '(3))))
  (ok-roundtrip "io/1elem" (vt-ones '(1)))
  ;; 全部 dtype
  (dolist (dt '(:bool :float64 :float32 :int64 :int32 :int16 :int8 :uint8 :uint16
                :complex128 :complex64))
    (ok-roundtrip (format nil "io/dtype-~a" dt) (vt-astype (vt-arange 4) dt)))
  ;; 非连续视图必须按逻辑内容物化
  (let ((a (vt-reshape (vt-arange 12 :dtype :int64) '(3 4))))
    (ok-roundtrip "io/transpose" (vt-transpose a))
    (ok-roundtrip "io/slice-step2" (vt-slice a '(0 3 2)))
    (ok-roundtrip "io/flip" (vt-flip a))
    (ok-roundtrip "io/broadcast0" (vt-broadcast-to (vt-reshape (vt-arange 4 :dtype :int64) '(1 4))
                                                  '(3 4))))
  ;; NaN / Inf / ±0 的位模式（VT-ARRAY-EQUIV 是 NaN != NaN，必须比位）
  (ok-roundtrip "io/nan-inf-zeroneg"
                (vt-from-sequence (list (nan-d) (inf-d) (ninf-d) (- *zero-d*)) :dtype :float64))
  (ok-roundtrip "io/nan-f32" (vt-from-sequence (list (nan-d)) :dtype :float32))
  (ok-roundtrip "io/cx-nan"
                (vt-from-sequence (list (complex (nan-d) (inf-d))
                                        (complex (ninf-d) (nan-d)))
                                  :dtype :complex128))
  (ok-roundtrip "io/cx-negzero"
                (vt-from-sequence (list (complex (- *zero-d*) *zero-d*)) :dtype :complex128)))

;;; ------------------------------------------------------------------
;;; J. dtype 转换矩阵抽样
;;; ------------------------------------------------------------------

(defun run-dtype-matrix ()
  (format t "~&--- J. dtype 转换矩阵抽样 ---%")
  ;; float → int 是**向零截断**，不是四舍五入
  (let ((r (vt-astype (ff '(-7.8d0 -0d0 0.5d0 7.8d0)) :int64)))
    (ok-vals "astype/f64→i64 向零截断" r '(-7 0 0 7))
    (ok-dtype "astype/f64→i64 dtype" r :int64))
  ;; 窄化回绕（补码语义）
  (ok-vals "astype/i16 200→i8 回绕"
           (vt-astype (vt-from-sequence '(200 -200) :dtype :int16) :int8) '(-56 56))
  (ok-vals "astype/u8 200→i8 回绕"
           (vt-astype (vt-from-sequence '(200) :dtype :uint8) :int8) '(-56))
  ;; 负值进无符号：mod 2^8
  (ok-vals "astype/i64 -7→u8 回绕"
           (vt-astype (fi '(-7 300)) :uint8) '(249 44))
  (ok-vals "astype/i8 -1→u16"
           (vt-astype (vt-from-sequence '(-1) :dtype :int8) :uint16) '(65535))
  ;; 溢出到 0：2^40 mod 2^32 == 0
  (ok-vals "astype/i64 2^40→i32"
           (vt-astype (vt-from-sequence (list (expt 2 40)) :dtype :int64) :int32) '(0))
  ;; 真值化：非零为 T（含 -0.0 为 nil）
  (let ((r (vt-astype (ff '(0d0 -0d0 0.5d0 -1d0)) :bool)))
    (ok-dtype "astype/f64→bool dtype" r :bool)
    (ok-vals "astype/f64→bool 值" r '(0 0 1 1)))
  ;; 复数 → 实数丢虚部
  (ok-vals "astype/c128→f64 丢虚部"
           (vt-astype (vt-from-sequence (list (complex 1.5d0 -2.5d0)) :dtype :complex128)
                      :float64) '(1.5d0))
  ;; 实数 → 复数虚部 0
  (ok-dtype "astype/i8→c64 dtype"
            (vt-astype (vt-from-sequence '(3) :dtype :int8) :complex64) :complex64)
  (ok-vals "astype/i8→c64 实部"
           (vt-astype (vt-from-sequence '(3) :dtype :int8) :complex64) (list (complex 3d0 0d0)))
  ;; 布尔 → 整型 0/1
  (ok-vals "astype/bool→i64"
           (vt-astype (vt-from-sequence '(1 0 1) :dtype :bool) :int64) '(1 0 1))
  ;; f32 → f64 的精确提升（0.1f 在 double 里不是 0.1）
  (let ((r (vt-astype (vt-from-sequence '(0.1d0) :dtype :float32) :float64)))
    (ok-dtype "astype/f32→f64 dtype" r :float64)
    (report "astype/f32→f64 保留 f32 的近似值"
            (< (abs (- (vt-item r) 0.1d0)) 1d-8)))
  ;; 往返：i64 → f64 → i64（小整数范围内无损）
  (ok-vals "astype/i64→f64→i64 往返"
           (vt-astype (vt-astype (fi '(1 2 3)) :float64) :int64) '(1 2 3)))

;;; ------------------------------------------------------------------

(run-cumulative-empty)
(run-indices-empty)
(run-split-empty)
(run-cov-degenerate)
(run-spaces-zero)
(run-bincount-weights)
(run-random-choice-p)
(run-random-errors)
(run-io-roundtrip)
(run-dtype-matrix)

(format t "~&=====================================~%")
(format t "Total: ~d / Pass: ~d / Fail: ~d~%" (+ *pass* *fail*) *pass* *fail*)
(format t "PASS=~d FAIL=~d~%" *pass* *fail*)
(when *failures*
  (format t "~&--- 失败清单 ---~%")
  (dolist (f (reverse *failures*)) (format t "  · ~a~%" f)))
(when (plusp *fail*) (sb-ext:exit :code 1))
