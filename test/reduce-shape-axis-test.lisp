;;;; reduce-shape-axis-test.lisp — 归约/求解的「形状 × 轴 × 序列参数」契约回归
;;;;
;;;; 覆盖面（全部来自差分模糊测试发现的真实缺陷）：
;;;;
;;;;   A. VT-SOLVE / VT-INV 的结果秩只由**输入秩**决定（CONVENTIONS §6.5）
;;;;      旧实现用 `(= nrhs 1)` 判定返回秩，于是 b 的形状 (n,1) 被压成 (n)、
;;;;      1×1 方程组的解退化成 shape (1) 而不是 (1 1)。
;;;;      numpy 对照：np.linalg.solve(A, [[1],[2],[3]]).shape == (3, 1)。
;;;;
;;;;   B. VT-PTP 的全归约返回**0 维张量**而非 Lisp 标量，且支持 :KEEPDIMS
;;;;      旧实现 `(- (vt-item (vt-amax …)) (vt-item (vt-amin …)))` 直接返回
;;;;      Lisp 数 —— 同族的 vt-amax/vt-amin 都返回 vt，只有它漏了。
;;;;
;;;;   C. VT-NANMEDIAN 的 :KEEPDIMS 在全局归约时返回全 1 形状
;;;;      旧实现照搬 vt-median 的 keepdims 分支，返回 0 维（vt-median 给 (1 1)）。
;;;;
;;;;   D. VT-MEDIAN 的多轴 / 单元素列表 / 负轴 / 顺序无关
;;;;      旧实现 `(vt-normalize-axis '(0 1) rank)` 直接抛
;;;;      「The value (0 1) is not of type …」，而 numpy 的 axis 元组是支持的。
;;;;
;;;;   E. VT-PERCENTILE / VT-QUANTILE 的多轴 + 分位点序列 + :KEEPDIMS 组合
;;;;      两类参数此前都是「单独能跑、一起就炸」：分位点序列走逐个标量调用，
;;;;      标量调用又把轴元组塞进单轴内核，报的是内部类型泄漏
;;;;      「The value (0 1) is not of type REAL」。
;;;;
;;;;   F. VT-LEFT-SHIFT / VT-RIGHT-SHIFT 的第二操作数可以是张量（逐元素位移）
;;;;      旧实现 `(ash x shift)` 里 shift 是标量形参，传张量报
;;;;      「The value … is not of type INTEGER」；numpy 支持广播。
;;;;
;;;;   G. VT-HISTOGRAM 的 :BINS 接受**边界序列**
;;;;      旧实现只认正整数，边界序列报「必须为正整数」；numpy 的两种语义
;;;;      都支持（箱边界 vs 箱数）。
;;;;
;;;;   H. 复数守卫去重后的行为不变（VT-SIGMOID/RELU/… 仍按契约拒绝复数）
;;;;      源码里同一段 `%cx-reject` 守卫被粘贴了两份（第一份是死代码），
;;;;      删除时若误删第二份，这些函数会静默接受复数。本组钉住「拒绝」。
;;;;
;;;; 期望值全部用本机 numpy 2.4.6 实测（CONVENTIONS §10.4：禁止手写期望）。
;;;; 用法：sbcl --non-interactive --load test/reduce-shape-axis-test.lisp

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
  "经 defvar 的零：避免 (/ 0d0 0d0) 在编译期被常量折叠（折叠期会抛
   FLOATING-POINT-INVALID-OPERATION，与陷阱屏蔽无关）。")

(defun nan-d () (with-float-safe (/ *zero-d* *zero-d*)))

(defun nan-p (x) (and (floatp x) (sb-ext:float-nan-p x)))

(defun num= (a b)
  (cond ((and (complexp a) (complexp b))
         (and (num= (realpart a) (realpart b))
              (num= (imagpart a) (imagpart b))))
        ((or (complexp a) (complexp b)) nil)
        ((and (floatp a) (floatp b))
         (cond ((and (nan-p a) (nan-p b)) t)
               ((or (nan-p a) (nan-p b)) nil)
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

(defun ok-true (name form)
  (report name (and form t)))

(defun ok-err (name thunk needle)
  "断言 THUNK 报错，且消息里含 NEEDLE（子串）。"
  (handler-case
      (let ((v (funcall thunk)))
        (report name nil (format nil "没有报错，返回 ~s" v)))
    (error (e)
      (let ((msg (princ-to-string e)))
        (report name (and (search needle msg) t)
                (format nil "消息不含「~a」：~a" needle
                        (substitute #\Space #\Newline msg)))))))

;;; ------------------------------------------------------------------
;;; A. vt-solve / vt-inv 的结果秩由输入秩决定
;;; ------------------------------------------------------------------

(defun run-solve-rank ()
  (format t "~&~%--- A. vt-solve / vt-inv 结果秩（§6.5）---~%")
  (let* ((a2 (vt-from-sequence '((2d0 0d0) (0d0 4d0)) :dtype :float64))
         (b-col (vt-from-sequence '((2d0) (8d0)) :dtype :float64))
         (b-vec (vt-from-sequence '(2d0 8d0) :dtype :float64))
         (b-2col (vt-from-sequence '((2d0 4d0) (8d0 12d0)) :dtype :float64)))
    ;; numpy: solve([[2,0],[0,4]], [[2],[8]]) -> [[1],[2]]
    (ok-shape "solve/b(n,1) 保持二维" (vt-solve a2 b-col) '(2 1))
    (ok-vals  "solve/b(n,1) 值" (vt-solve a2 b-col) '(1d0 2d0))
    ;; numpy: solve(..., [2,8]) -> [1,2]
    (ok-shape "solve/b(n,) 保持一维" (vt-solve a2 b-vec) '(2))
    (ok-vals  "solve/b(n,) 值" (vt-solve a2 b-vec) '(1d0 2d0))
    ;; numpy: solve(..., [[2,4],[8,12]]) -> [[1,2],[2,3]]
    (ok-shape "solve/b(n,2) 二维" (vt-solve a2 b-2col) '(2 2))
    (ok-vals  "solve/b(n,2) 值" (vt-solve a2 b-2col) '(1d0 2d0 2d0 3d0)))
  ;; 退化情形：唯一的一列曾被 nrhs==1 判定压平
  (let ((a1 (vt-from-sequence '((3d0)) :dtype :float64)))
    (ok-shape "solve/1x1 b(1,1) 不塌成一维" (vt-solve a1 (vt-from-sequence '((6d0)) :dtype :float64)) '(1 1))
    (ok-vals  "solve/1x1 b(1,1) 值" (vt-solve a1 (vt-from-sequence '((6d0)) :dtype :float64)) '(2d0))
    (ok-shape "solve/1x1 b(1,)" (vt-solve a1 (vt-from-sequence '(6d0) :dtype :float64)) '(1))
    (ok-shape "inv/1x1 保持二维" (vt-inv (vt-from-sequence '((2d0)) :dtype :float64)) '(1 1))
    (ok-vals  "inv/1x1 值" (vt-inv (vt-from-sequence '((2d0)) :dtype :float64)) '(0.5d0)))
  ;; 复数同样按输入秩：numpy 给 (2,1)
  (let ((ac (vt-from-sequence (list (list #C(2d0 0d0) #C(0d0 0d0))
                                    (list #C(0d0 0d0) #C(4d0 0d0)))
                              :dtype :complex128))
        (bc (vt-from-sequence (list (list #C(2d0 0d0)) (list #C(8d0 0d0))) :dtype :complex128)))
    (ok-shape "solve/复数 b(n,1)" (vt-solve ac bc) '(2 1))
    (ok-vals  "solve/复数 b(n,1) 值" (vt-solve ac bc) (list #C(1d0 0d0) #C(2d0 0d0))))
  ;; 形状非法仍然报错（回归网：改动不能把检查冲掉）
  (ok-err "solve/b 行数不匹配报错"
          (lambda () (vt-solve (vt-from-sequence '((1d0 0d0) (0d0 1d0)) :dtype :float64)
                               (vt-from-sequence '(1d0 2d0 3d0) :dtype :float64)))
          "行数"))

;;; ------------------------------------------------------------------
;;; B. vt-ptp：返回 0 维 vt + :keepdims
;;; ------------------------------------------------------------------

(defun run-ptp ()
  (format t "~&~%--- B. vt-ptp 返回值形状与 :keepdims ---~%")
  (let* ((a (vt-from-sequence '((1d0 2d0) (4d0 8d0)) :dtype :float64))
         (r (vt-ptp a)))
    (ok-true  "ptp/全归约返回 vt 而非 Lisp 标量" (typep r 'vt))
    (ok-shape "ptp/全归约是 0 维" r nil)
    (ok-vals  "ptp/全归约值" r '(7d0)))
  (let ((a (vt-from-sequence '((1d0 2d0) (4d0 8d0)) :dtype :float64)))
    ;; numpy: ptp(a, axis=0) -> [3, 6]
    (ok-shape "ptp/axis 0" (vt-ptp a :axis 0) '(2))
    (ok-vals  "ptp/axis 0 值" (vt-ptp a :axis 0) '(3d0 6d0))
    ;; numpy: ptp(a, axis=0, keepdims=True) -> [[3, 6]]
    (ok-shape "ptp/axis 0 keepdims" (vt-ptp a :axis 0 :keepdims t) '(1 2))
    (ok-vals  "ptp/axis 0 keepdims 值" (vt-ptp a :axis 0 :keepdims t) '(3d0 6d0))
    ;; numpy: ptp(a, keepdims=True) -> [[7]]
    (ok-shape "ptp/全归约 keepdims" (vt-ptp a :keepdims t) '(1 1))
    (ok-vals  "ptp/全归约 keepdims 值" (vt-ptp a :keepdims t) '(7d0)))
  ;; numpy: ptp([[1,5],[2,9]], axis=-1, keepdims=True) -> [[4],[7]]
  (let ((a (vt-from-sequence '((1d0 5d0) (2d0 9d0)) :dtype :float64)))
    (ok-shape "ptp/负轴 keepdims" (vt-ptp a :axis -1 :keepdims t) '(2 1))
    (ok-vals  "ptp/负轴 keepdims 值" (vt-ptp a :axis -1 :keepdims t) '(4d0 7d0)))
  ;; 整型保持 dtype（numpy: ptp(int64) -> int64）
  (let ((a (vt-from-sequence '((1 9) (3 5)) :dtype :int64)))
    (ok-dtype "ptp/int64 保持 dtype" (vt-ptp a) :int64)
    (ok-vals  "ptp/int64 值" (vt-ptp a) '(8)))
  ;; NaN 传播（numpy: ptp 用 max-min，NaN 参与即 NaN）
  (let ((a (vt-from-sequence (list 1d0 5d0 (nan-d)) :dtype :float64)))
    (ok-true "ptp/NaN 传播" (nan-p (first (flat (vt-ptp a))))))
  ;; 空归约区（numpy: ValueError zero-size array to reduction operation）
  (ok-err "ptp/空张量报错"
          (lambda () (vt-ptp (vt-zeros '(0) :dtype :float64)))
          "zero-size")
  ;; 布尔减法未定义（§17.3）：两个分支报同一条消息
  (let ((b (vt-from-sequence (list t nil t) :dtype :bool)))
    (ok-err "ptp/bool 全归约报错" (lambda () (vt-ptp b)) "布尔")
    (ok-err "ptp/bool axis 报错" (lambda () (vt-ptp b :axis 0)) "布尔")))

;;; ------------------------------------------------------------------
;;; C. vt-nanmedian 的 :keepdims
;;; ------------------------------------------------------------------

(defun run-nanmedian-keepdims ()
  (format t "~&~%--- C. vt-nanmedian :keepdims ---~%")
  (let ((a (vt-from-sequence '((1d0 2d0) (3d0 4d0)) :dtype :float64)))
    ;; numpy: nanmedian(a) -> 2.5（0 维），nanmedian(a, keepdims=True) -> [[2.5]]
    (ok-shape "nanmedian/全归约 0 维" (vt-nanmedian a) nil)
    (ok-shape "nanmedian/全归约 keepdims 形状" (vt-nanmedian a :keepdims t) '(1 1))
    (ok-vals  "nanmedian/全归约 keepdims 值" (vt-nanmedian a :keepdims t) '(2.5d0))
    ;; numpy: nanmedian(a, axis=0, keepdims=True) -> [[2, 3]]
    (ok-shape "nanmedian/axis 0 keepdims" (vt-nanmedian a :axis 0 :keepdims t) '(1 2))
    (ok-vals  "nanmedian/axis 0 keepdims 值" (vt-nanmedian a :axis 0 :keepdims t) '(2d0 3d0))
    ;; 与 vt-median 对齐：两者的 keepdims 语义必须一致
    (ok-shape "median/全归约 keepdims 同形" (vt-median a :keepdims t) '(1 1)))
  ;; 含 NaN 时 nanmedian 跳过、median 传播
  (let ((a (vt-from-sequence (list 1d0 3d0 (nan-d) 8d0) :dtype :float64)))
    (ok-vals "nanmedian/跳过 NaN（keepdims）" (vt-nanmedian a :keepdims t) '(3d0))
    (ok-true "median/传播 NaN（keepdims）" (nan-p (first (flat (vt-median a :keepdims t)))))))

;;; ------------------------------------------------------------------
;;; D. vt-median / vt-nanmedian 的多轴
;;; ------------------------------------------------------------------

(defun run-median-multi-axis ()
  (format t "~&~%--- D. vt-median 多轴归约 ---~%")
  (let ((a (vt-reshape (vt-arange 24 :dtype :float64) '(2 3 4))))
    ;; numpy: median(a, axis=(0,1)) -> [10, 11, 12, 13]
    (ok-shape "median/axis(0 1)" (vt-median a :axis '(0 1)) '(4))
    (ok-vals  "median/axis(0 1) 值" (vt-median a :axis '(0 1)) '(10d0 11d0 12d0 13d0))
    ;; numpy: median(a, axis=(0,1), keepdims=True) -> [[[10, 11, 12, 13]]]
    (ok-shape "median/axis(0 1) keepdims" (vt-median a :axis '(0 1) :keepdims t) '(1 1 4))
    (ok-vals  "median/axis(0 1) keepdims 值" (vt-median a :axis '(0 1) :keepdims t)
              '(10d0 11d0 12d0 13d0))
    ;; numpy: median(a, axis=(1,2)) -> [5.5, 17.5]
    (ok-shape "median/axis(1 2)" (vt-median a :axis '(1 2)) '(2))
    (ok-vals  "median/axis(1 2) 值" (vt-median a :axis '(1 2)) '(5.5d0 17.5d0))
    (ok-shape "median/axis(1 2) keepdims" (vt-median a :axis '(1 2) :keepdims t) '(2 1 1))
    (ok-vals  "median/axis(1 2) keepdims 值" (vt-median a :axis '(1 2) :keepdims t) '(5.5d0 17.5d0))
    ;; 轴元组顺序无关、负轴等价（numpy 的 axistuple 归一化语义）
    (ok-vals "median/轴序无关" (vt-median a :axis '(2 1)) '(5.5d0 17.5d0))
    (ok-shape "median/轴序无关 形状" (vt-median a :axis '(2 1)) '(2))
    (ok-vals "median/负轴等价" (vt-median a :axis '(-1 -2)) '(5.5d0 17.5d0))
    ;; 单元素列表退化成整数轴（不能走折叠路径）
    (ok-shape "median/单元素列表 -> 整数轴" (vt-median a :axis '(1)) '(2 4))
    ;; numpy: median(a, axis=1).shape -> (2, 4)，值 [4,5,6,7] / [16,17,18,19]
    (ok-vals  "median/单元素列表 值" (vt-median a :axis '(1))
              '(4d0 5d0 6d0 7d0 16d0 17d0 18d0 19d0))
    ;; 单个整数轴仍是老路径
    (ok-shape "median/整数轴不回归" (vt-median a :axis 0) '(3 4))
    ;; nanmedian 走同一条折叠路径
    (ok-vals "nanmedian/axis(0 1)" (vt-nanmedian a :axis '(0 1)) '(10d0 11d0 12d0 13d0))
    (ok-shape "nanmedian/axis(0 1) keepdims" (vt-nanmedian a :axis '(0 1) :keepdims t) '(1 1 4)))
  ;; 复数多轴：字典序中位数（§16）
  (let ((c (vt-from-sequence (list (list #C(1d0 1d0) #C(3d0 3d0))
                                   (list #C(2d0 2d0) #C(5d0 5d0)))
                             :dtype :complex128)))
    ;; 排序后 1+1i, 2+2i, 3+3i, 5+5i -> 中位 = 2.5+2.5i
    (ok-shape "median/复数 axis(0 1)" (vt-median c :axis '(0 1)) nil)
    (ok-vals  "median/复数 axis(0 1) 值" (vt-median c :axis '(0 1)) (list #C(2.5d0 2.5d0))))
  ;; 非连续（转置）视图上的多轴折叠：折叠只用 transpose/reshape
  (let* ((a (vt-reshape (vt-arange 24 :dtype :float64) '(2 3 4)))
         (t-view (vt-transpose a '(1 0 2))))
    ;; 转置后形状 (3 2 4)；median over (0 1) == 原来的 median over (1 0)
    (ok-vals "median/非连续视图 axis(0 1)" (vt-median t-view :axis '(0 1)) '(10d0 11d0 12d0 13d0)))
  ;; 错误契约
  (let ((a (vt-reshape (vt-arange 24 :dtype :float64) '(2 3 4))))
    (ok-err "median/重复轴报错" (lambda () (vt-median a :axis '(1 1))) "重复")
    (ok-err "median/越界轴报错" (lambda () (vt-median a :axis '(0 9))) "out of bounds")))

;;; ------------------------------------------------------------------
;;; E. vt-percentile / vt-quantile：多轴 × 分位点序列
;;; ------------------------------------------------------------------

(defun run-percentile ()
  (format t "~&~%--- E. vt-percentile / vt-quantile 多轴与序列 ---~%")
  (let ((a (vt-reshape (vt-arange 24 :dtype :float64) '(2 3 4))))
    ;; ---- 分位点序列（原本就支持，防止回归）----
    ;; numpy: percentile(a, [25,50,75]) -> [5.75, 11.5, 17.25]
    (ok-shape "pct/序列 全归约" (vt-percentile a '(25 50 75)) '(3))
    (ok-vals  "pct/序列 全归约 值" (vt-percentile a '(25 50 75)) '(5.75d0 11.5d0 17.25d0))
    (ok-shape "pct/单元素序列" (vt-percentile a '(50)) '(1))
    (ok-vals  "pct/单元素序列 值" (vt-percentile a '(50)) '(11.5d0))
    ;; numpy: percentile(a, [25,50,75], axis=0).shape -> (3, 3, 4)
    (ok-shape "pct/序列 axis 0" (vt-percentile a '(25 50 75) :axis 0) '(3 3 4))
    (ok-shape "pct/序列 axis 0 keepdims" (vt-percentile a '(25 50 75) :axis 0 :keepdims t) '(3 1 3 4))
    ;; numpy: percentile(a, [25,75], axis=2)[0][0] -> [0.75, 4.75, 8.75]
    (ok-shape "pct/序列 axis 2" (vt-percentile a '(25 75) :axis 2) '(2 2 3))
    (ok-vals  "pct/序列 axis 2 值" (vt-percentile a '(25 75) :axis 2)
              '(0.75d0 4.75d0 8.75d0 12.75d0 16.75d0 20.75d0
                2.25d0 6.25d0 10.25d0 14.25d0 18.25d0 22.25d0))
    ;; numpy: percentile(a, [25,75], keepdims=True).shape -> (2, 1, 1, 1)
    (ok-shape "pct/序列 全归约 keepdims" (vt-percentile a '(25 75) :keepdims t) '(2 1 1 1))
    (ok-vals  "pct/序列 全归约 keepdims 值" (vt-percentile a '(25 75) :keepdims t) '(5.75d0 17.25d0))

    ;; ---- 多轴（轴元组）：此前抛「The value (0 1) is not of type REAL」----
    ;; numpy: percentile(a, [10,90], axis=(0,1)) -> [[2,3,4,5],[18,19,20,21]]
    (ok-shape "pct/多轴(0 1)" (vt-percentile a '(10 90) :axis '(0 1)) '(2 4))
    (ok-vals  "pct/多轴(0 1) 值" (vt-percentile a '(10 90) :axis '(0 1))
              '(2d0 3d0 4d0 5d0 18d0 19d0 20d0 21d0))
    ;; numpy: keepdims -> (2, 1, 1, 4)
    (ok-shape "pct/多轴(0 1) keepdims" (vt-percentile a '(10 90) :axis '(0 1) :keepdims t) '(2 1 1 4))
    ;; numpy: percentile(a, 50, axis=(0,1)) -> [10,11,12,13]
    (ok-shape "pct/多轴 标量分位点" (vt-percentile a 50 :axis '(0 1)) '(4))
    (ok-vals  "pct/多轴 标量分位点 值" (vt-percentile a 50 :axis '(0 1)) '(10d0 11d0 12d0 13d0))
    (ok-shape "pct/多轴 标量分位点 keepdims" (vt-percentile a 50 :axis '(0 1) :keepdims t) '(1 1 4))
    ;; numpy: percentile(a, [10,90], axis=(1,2), method='lower') -> [[0,1,2,3],[16,17,18,19]]
    (ok-vals  "pct/多轴 :lower 插值" (vt-percentile a '(10 90) :axis '(0 1) :interpolation :lower)
              '(0d0 1d0 2d0 3d0 16d0 17d0 18d0 19d0))
    ;; numpy: quantile(a, [0.25,0.75], axis=(0,2)) -> [[1.75,5.75,9.75],[13.25,17.25,21.25]]
    (ok-shape "quant/多轴(0 2)" (vt-quantile a '(0.25 0.75) :axis '(0 2)) '(2 3))
    (ok-vals  "quant/多轴(0 2) 值" (vt-quantile a '(0.25 0.75) :axis '(0 2))
              '(1.75d0 5.75d0 9.75d0 13.25d0 17.25d0 21.25d0))
    ;; 与 vt-median 保持一致：median == percentile 50（多轴下也必须成立）
    (ok-vals "pct/多轴与 median 一致" (vt-percentile a 50 :axis '(1 2))
              (flat (vt-median a :axis '(1 2))))
    (ok-shape "pct/单元素轴列表" (vt-percentile a '(25 75) :axis '(1)) '(2 2 4))
    ;; 非连续视图（转置）上的多轴
    (ok-vals "pct/多轴 非连续视图" (vt-percentile (vt-transpose a '(1 0 2)) '(10 90) :axis '(0 1))
              '(2d0 3d0 4d0 5d0 18d0 19d0 20d0 21d0)))
  ;; 错误契约
  (let ((a (vt-reshape (vt-arange 24 :dtype :float64) '(2 3 4))))
    (ok-err "pct/空序列报错" (lambda () (vt-percentile a '())) "不能为空")
    (ok-err "pct/越界分位点报错" (lambda () (vt-percentile a '(101))) "0, 100")
    (ok-err "quant/越界分位数报错" (lambda () (vt-quantile a 1.5)) "0, 1")
    (ok-err "pct/重复轴报错" (lambda () (vt-percentile a 50 :axis '(1 1))) "重复")
    (ok-err "pct/越界轴报错" (lambda () (vt-percentile a 50 :axis '(0 9))) "out of bounds")
    (ok-err "pct/非法插值报错" (lambda () (vt-percentile a 50 :interpolation :cubic)) "interpolation")))

;;; ------------------------------------------------------------------
;;; F. vt-left-shift / vt-right-shift 的张量位移
;;; ------------------------------------------------------------------

(defun run-shift ()
  (format t "~&~%--- F. 位移的第二操作数可以是张量 ---~%")
  (let ((a (vt-from-sequence '(1 2 4 8) :dtype :int64))
        (s (vt-from-sequence '(0 1 2 3) :dtype :int64)))
    ;; numpy: left_shift([1,2,4,8], 1) -> [2,4,8,16]
    (ok-vals "lshift/标量位移不变" (vt-left-shift a 1) '(2 4 8 16))
    ;; numpy: left_shift([1,2,4,8], [0,1,2,3]) -> [1,4,16,64]
    (ok-shape "lshift/张量位移形状" (vt-left-shift a s) '(4))
    (ok-vals  "lshift/张量位移值" (vt-left-shift a s) '(1 4 16 64))
    ;; numpy: right_shift([1,2,4,8], [0,1,2,3]) -> [1,1,1,1]
    (ok-vals "rshift/张量位移值" (vt-right-shift a s) '(1 1 1 1))
    (ok-dtype "lshift/张量位移 dtype" (vt-left-shift a s) :int64)
    ;; 张量位移与标量位移在「同一个 shift」上必须逐位一致
    (ok-vals "lshift/张量≡标量（s=2）"
             (vt-left-shift a (vt-from-sequence '(2 2 2 2) :dtype :int64))
             (flat (vt-left-shift a 2))))
  ;; 广播：numpy: left_shift([[1,2],[4,8]], [1,2]) -> [[2,8],[8,32]]
  (let ((a (vt-from-sequence '((1 2) (4 8)) :dtype :int64))
        (s (vt-from-sequence '(1 2) :dtype :int64)))
    (ok-shape "lshift/广播 形状" (vt-left-shift a s) '(2 2))
    (ok-vals  "lshift/广播 值" (vt-left-shift a s) '(2 8 8 32)))
  ;; uint8 的右移：dtype 遵循 NEP 50 的「张量不算弱标量」规则 ——
  ;; numpy 实测 right_shift(uint8, int64张量) -> int64，而 right_shift(uint8, 1) -> uint8。
  (let ((a (vt-from-sequence '(200 100) :dtype :uint8))
        (s (vt-from-sequence '(1 2) :dtype :int64)))
    (ok-vals "rshift/uint8 张量位移" (vt-right-shift a s) '(100 25))
    (ok-dtype "rshift/uint8 张量位移 dtype（张量参与 → int64）" (vt-right-shift a s) :int64)
    (ok-dtype "rshift/uint8 标量位移 dtype（弱标量 → uint8）" (vt-right-shift a 1) :uint8)
    (ok-vals "rshift/uint8 标量位移 值" (vt-right-shift a 1) '(100 50)))
  ;; :out 落点（非连续目标）
  (let* ((base (vt-zeros '(2 4) :dtype :int64))
         (col (vt-slice base '(0 2) '(1 3))))
    (vt-left-shift (vt-from-sequence '((1 2) (3 4)) :dtype :int64)
                   (vt-from-sequence '((1 1) (1 1)) :dtype :int64)
                   :out col)
    (ok-vals "lshift/:out 非连续落点" (vt-copy col) '(2 4 6 8))
    (ok-vals "lshift/:out 不越界" (vt-ravel base) '(0 2 4 0 0 6 8 0)))
  ;; 错误契约：浮点位移无定义（ash 要求整数）
  (ok-err "lshift/浮点张量位移报错"
          (lambda () (vt-left-shift (vt-from-sequence '(1.5d0) :dtype :float64)
                                    (vt-from-sequence '(1) :dtype :int64)))
          "INTEGER")
  (ok-err "lshift/形状不匹配报错"
          (lambda () (vt-left-shift (vt-from-sequence '(1 2 3) :dtype :int64)
                                    (vt-from-sequence '(1 2) :dtype :int64)))
          "无法广播"))

;;; ------------------------------------------------------------------
;;; G. vt-histogram 的边界序列
;;; ------------------------------------------------------------------

(defun run-histogram ()
  (format t "~&~%--- G. vt-histogram :bins 边界序列 ---~%")
  (let ((a (vt-from-sequence '(-1d0 0d0 1d0 2d0 3d0 5d0) :dtype :float64)))
    ;; 正整数分支不回归：numpy histogram(a, bins=3)[0] -> [2,2,2]
    (ok-vals "hist/bins 正整数" (nth-value 0 (vt-histogram a :bins 3)) '(2 2 2))
    ;; numpy: histogram(a, bins=[-2,0,2,6]) -> counts [1,2,3], edges 同输入
    (multiple-value-bind (c e) (vt-histogram a :bins '(-2d0 0d0 2d0 6d0))
      (ok-vals "hist/边界序列 counts" c '(1 2 3))
      (ok-vals "hist/边界序列 edges 原样返回" e '(-2d0 0d0 2d0 6d0))
      (ok-dtype "hist/边界序列 edges dtype" e :float64)
      (ok-dtype "hist/边界序列 counts dtype" c :int64))
    ;; numpy: histogram(a, bins=[0,2,4]) -> [2,2]（区间外的 -1、5 丢弃）
    (ok-vals "hist/边界序列 丢弃区间外" (nth-value 0 (vt-histogram a :bins '(0d0 2d0 4d0))) '(2 2))
    ;; numpy: histogram(a, bins=[0,1,2,3,4]) -> [1,1,1,1]
    ;; （恰好等于内部边界的值归右侧箱，正好等于末边界的归最后一箱）
    (ok-vals "hist/边界序列 左闭右开" (nth-value 0 (vt-histogram a :bins '(0d0 1d0 2d0 3d0 4d0))) '(1 1 1 1))
    ;; numpy: histogram(arange(4), bins=2, density=True) -> [1/3, 1/3]
    (let ((d (nth-value 0 (vt-histogram (vt-arange 4 :dtype :float64) :bins 2 :density t))))
      (ok-dtype "hist/density dtype" d :float64)
      (ok-vals  "hist/density 值" d '(0.3333333333333333d0 0.3333333333333333d0)))
    ;; 边界可以是 vt（1-D 张量），也可以是整数列表
    (ok-vals "hist/边界为 vt" (nth-value 0 (vt-histogram a :bins (vt-from-sequence '(0d0 2d0 4d0) :dtype :float64)))
             '(2 2))
    ;; NaN 不属于任何箱（会从两侧比较里漏过去，一路落进末箱）
    ;; numpy: histogram([0, nan, 1.5], bins=[0,1,2]) -> [1,1]
    (ok-vals "hist/边界序列 NaN 不计入"
             (nth-value 0 (vt-histogram (vt-from-sequence (list 0d0 (nan-d) 1.5d0) :dtype :float64)
                                        :bins '(0d0 1d0 2d0)))
             '(1 1))
    ;; numpy: histogram([0, inf, -inf, 1.5], bins=[0,1,2]) -> [1,1]
    (ok-vals "hist/边界序列 ±Inf 不计入"
             (nth-value 0 (vt-histogram (vt-from-sequence
                                         (list 0d0 (vt-float-pos-inf) (vt-float-neg-inf) 1.5d0)
                                         :dtype :float64)
                                        :bins '(0d0 1d0 2d0)))
             '(1 1))
    ;; 等宽分支同样丢弃区间外的 NaN（numpy: bins=2, range=(0,2) -> [1,1]）
    (ok-vals "hist/等宽箱区间外 NaN 丢弃"
             (nth-value 0 (vt-histogram (vt-from-sequence (list 0d0 (nan-d) 1.5d0) :dtype :float64)
                                        :bins 2 :range '(0d0 2d0)))
             '(1 1))
    ;; numpy: histogram([1,2,3,4], bins=[1,3,5]) -> [2,2]，整型输入也走同一路径
    (ok-vals "hist/整型数据 + 整型边界"
             (nth-value 0 (vt-histogram (vt-from-sequence '(1 2 3 4) :dtype :int64) :bins '(1 3 5)))
             '(2 2)))
  ;; 错误契约
  (let ((a (vt-arange 4 :dtype :float64)))
    (ok-err "hist/单个边界报错" (lambda () (vt-histogram a :bins '(1d0))) "至少需要 2")
    (ok-err "hist/边界非递增报错" (lambda () (vt-histogram a :bins '(0d0 2d0 2d0))) "严格递增")
    (ok-err "hist/边界递减报错" (lambda () (vt-histogram a :bins '(3d0 2d0 1d0))) "严格递增")
    (ok-err "hist/复数输入报错"
            (lambda () (vt-histogram (vt-from-sequence (list #C(1d0 0d0)) :dtype :complex128)))
            "复数")))

;;; ------------------------------------------------------------------
;;; H. 复数守卫去重后的拒绝契约（源码清理的回归网）
;;; ------------------------------------------------------------------

(defun run-cx-guards ()
  (format t "~&~%--- H. 复数守卫去重后仍拒绝（nn / 统计）---~%")
  (let ((z (vt-from-sequence (list #C(1d0 2d0) #C(3d0 0d0)) :dtype :complex128))
        (r (vt-from-sequence '(-1d0 0.5d0 2d0) :dtype :float64)))
    (dolist (pair (list (cons "sigmoid" #'vt-sigmoid)
                        (cons "relu" #'vt-relu)
                        (cons "leaky-relu" #'vt-leaky-relu)
                        (cons "swish" #'vt-swish)
                        (cons "softplus" #'vt-softplus)
                        (cons "gelu" #'vt-gelu)
                        (cons "mish" #'vt-mish)))
      (ok-err (format nil "cx-guard/~a 拒绝复数" (car pair))
              (lambda () (funcall (cdr pair) z))
              "复数")
      ;; 实数路径未被清理动过
      (ok-true (format nil "cx-guard/~a 实数路径仍可用" (car pair))
               (typep (funcall (cdr pair) (vt-copy r)) 'vt)))
    ;; 统计侧：histogram / percentile 拒绝复数，median 支持复数
    (ok-err "cx-guard/histogram 拒绝复数" (lambda () (vt-histogram z)) "复数")
    (ok-err "cx-guard/percentile 拒绝复数" (lambda () (vt-percentile z 50)) "复数")
    (ok-true "cx-guard/median 支持复数" (typep (vt-median z) 'vt))
    ;; 复数 median 仍按字典序（清理守卫不能改变分派）
    (ok-vals "cx-guard/median 复数字典序"
             (vt-median (vt-from-sequence (list #C(1d0 9d0) #C(2d0 0d0) #C(2d0 1d0)) :dtype :complex128))
             (list #C(2d0 0d0)))))

;;; ------------------------------------------------------------------
;;; I. 交叉一致性：多轴归约族的形状契约
;;; ------------------------------------------------------------------

(defun run-cross-consistency ()
  (format t "~&~%--- I. 多轴归约族的形状一致性 ---~%")
  (let* ((a (vt-reshape (vt-arange 24 :dtype :float64) '(2 3 4)))
         (ax '(0 1)))
    ;; sum/mean/amax/amin/median/percentile 在同一轴元组上必须给出同一形状
    (let ((s1 (vt-shape (vt-sum a :axis ax)))
          (s2 (vt-shape (vt-mean a :axis ax)))
          (s3 (vt-shape (vt-amax a :axis ax)))
          (s4 (vt-shape (vt-amin a :axis ax)))
          (s5 (vt-shape (vt-median a :axis ax)))
          (s6 (vt-shape (vt-percentile a 50 :axis ax)))
          (s7 (vt-shape (vt-nanmean a :axis ax))))
      (report "cross/多轴形状一致（7 个函数）"
              (every (lambda (s) (equal s '(4))) (list s1 s2 s3 s4 s5 s6 s7))
              (format nil "~s" (list s1 s2 s3 s4 s5 s6 s7))))
    ;; 同一个结论在 keepdims 下也必须一致
    (let ((s1 (vt-shape (vt-sum a :axis ax :keepdims t)))
          (s2 (vt-shape (vt-median a :axis ax :keepdims t)))
          (s3 (vt-shape (vt-percentile a 50 :axis ax :keepdims t))))
      (report "cross/多轴 keepdims 形状一致"
              (and (equal s1 '(1 1 4)) (equal s2 s1) (equal s3 s1))
              (format nil "~s" (list s1 s2 s3))))
    ;; 数值交叉：median == percentile 50 == quantile 0.5
    (ok-true "cross/median ≡ percentile 50"
             (every #'num= (flat (vt-median a :axis ax))
                    (flat (vt-percentile a 50 :axis ax))))
    (ok-true "cross/percentile 50 ≡ quantile 0.5"
             (every #'num= (flat (vt-percentile a 50 :axis ax))
                    (flat (vt-quantile a 0.5 :axis ax))))
    ;; 形状守卫：多轴归约结果再归约必须回到 0 维
    (ok-shape "cross/多轴结果再归约" (vt-sum (vt-median a :axis ax)) nil)))

;;; ------------------------------------------------------------------
;;; 入口
;;; ------------------------------------------------------------------

(format t "~&=====================================")
(format t "~&归约/求解的形状 × 轴 × 序列契约")
(format t "~&=====================================~%")
(run-solve-rank)
(run-ptp)
(run-nanmedian-keepdims)
(run-median-multi-axis)
(run-percentile)
(run-shift)
(run-histogram)
(run-cx-guards)
(run-cross-consistency)

(format t "~&=====================================~%")
(format t "Total: ~d / Pass: ~d / Fail: ~d~%" (+ *pass* *fail*) *pass* *fail*)
(format t "PASS=~d FAIL=~d~%" *pass* *fail*)
(when *failures*
  (format t "~&--- 失败清单 ---~%")
  (dolist (f (reverse *failures*)) (format t "  · ~a~%" f)))
(when (plusp *fail*) (sb-ext:exit :code 1))
