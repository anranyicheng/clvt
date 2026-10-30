;;;; empty-axis-binning-test.lisp — 「长度为 0 的轴 / −0.0 / 先算摘要再判断」契约回归
;;;;
;;;; 上一轮（degenerate-empty-test）的判据是「零元素不是零值」；这一轮追的是
;;;; 同一族缺陷里**最靠底层**的三处：
;;;;
;;;;   ① (1- dim) 在 dim = 0 时是 -1 —— 用它算 offset，offset 就变成负数，
;;;;      而 offset 的类型是 (unsigned-byte 44)。于是「翻转一个空轴」这种
;;;;      完全合法的请求，会在**下一个**碰 offset 的操作上炸：
;;;;        (vt-flatten (vt-flip (vt-zeros '(0))))     → The value -1 is not of
;;;;                                                     type (UNSIGNED-BYTE 44)
;;;;        (vt-flatten (vt-flip (vt-zeros '(0 3))))    → subseq 越界
;;;;      numpy：np.zeros((0,))[::-1] 正常，data 指针不动。
;;;;
;;;;   ② 「先算个摘要再判断」的第二步：空集合没有 min/max/全等值可算，
;;;;      直接把内部归约的错误抛给用户。
;;;;        (vt-histogram (vt-zeros '(0)) :bins 3)  → vt-AMIN: zero-size array…
;;;;        (vt-softmax  (vt-zeros '(0)))          → vt-AMAX: zero-size array…
;;;;      numpy：np.histogram(np.zeros(0), bins=3) → ([0,0,0], [0,1/3,2/3,1])；
;;;;      softmax 在空轴上没有任何元素要算，输出就是空的。
;;;;
;;;;   ③ 第一步就把类型/符号丢掉：`0.5s0`（单精度字面量）与 `zerop`。
;;;;        histogram 全等值时用 (±0.5s0) 造宽度：1e8 起单精度表示不出 ±0.5，
;;;;        边界退化成三个同一个数；1e18 时样本一个都落不进箱。
;;;;        angle 用 (zerop re)(zerop im) 判「零」：−0.0+0.0i 是负实轴上的点，
;;;;        相位应为 π，实部符号被 zerop 吃掉后返回 0（差 180°）。
;;;;
;;;; 覆盖面（每一条都对应一处真实缺陷）：
;;;;
;;;;   A. 空轴视图的 offset 不变 —— flip / slice(负步长) / rot90 / fliplr /
;;;;      flipud / ravel / flatten / reshape，1D/2D/3D × 每一条轴。
;;;;   B. 空输入的「摘要族」不许泄漏内部错误 —— softmax / log-softmax /
;;;;      histogram。
;;;;   C. histogram 全等值的量级扫描（1 / 1e8 / 1e10 / 1e15 / 1e18）。
;;;;   D. histogram 的整型输入与浮点输入必须给出**同一组边界**（逐位）。
;;;;   E. histogram 的 density：总计 0 → NaN；不等宽边界按**逐箱**宽度归一。
;;;;   F. histogram 的 :weights（整型/浮点 dtype、逐元素配对、形状校验）。
;;;;   G. 相位 −0.0：angle 的四种 (±0, ±0) 组合与实数 −0.0。
;;;;   H. 空序列的 convolve / correlate 必须带 vt- 前缀报错。
;;;;   I. 报错前缀：gradient 的过短轴、pad 的空轴。
;;;;   J. bins 参数的错类型（浮点 bins 不许落进「边界序列」分支）。
;;;;
;;;; 期望值全部用本机 numpy 2.4.6 实测（CONVENTIONS §10.4：禁止手写期望）。
;;;; 用法：sbcl --non-interactive --load test/empty-axis-binning-test.lisp

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
(defun nan-p (x) (and (floatp x) (sb-ext:float-nan-p x)))

(defun num= (a b)
  (cond ((and (floatp a) (floatp b))
         (cond ((and (nan-p a) (nan-p b)) t)
               ((or (nan-p a) (nan-p b)) nil)
               ((and (sb-ext:float-infinity-p a) (sb-ext:float-infinity-p b))
                (eq (plusp a) (plusp b)))
               (t (<= (abs (- a b)) (* 1d-12 (max 1d0 (abs a) (abs b)))))))
        (t (and (realp a) (realp b) (= a b)))))

(defun bits (x) (%f64-bits (coerce x 'double-float)))

(defun bit= (a b)
  "两个浮点数逐位相同（−0.0 与 0.0 算不同）。"
  (and (floatp a) (floatp b) (= (bits a) (bits b))))

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

(defun ok-err (name thunk &rest needles)
  "断言 THUNK 报错，且消息里含全部 NEEDLES（子串）。"
  (handler-case
      (let ((v (funcall thunk)))
        (report name nil (format nil "没有报错，返回 ~s"
                                 (if (vt-p v) (list (vt-dtype v) (vt-shape v)) v))))
    (error (e)
      (let ((msg (princ-to-string e)))
        (report name (every (lambda (n) (search n msg)) needles)
                (format nil "消息:~a" (substitute #\Space #\Newline msg)))))))

(defun ff (l) (vt-from-sequence l :dtype :float64))
(defun fi (l) (vt-from-sequence l :dtype :int64))

;;; ------------------------------------------------------------------
;;; A. 空轴视图的 offset
;;;
;;; 判据有两条，缺一不可：
;;;   (a) 视图的 offset 必须留在 [0, 缓冲区长度] —— 空轴没有元素可翻，
;;;       任何位移都只能是 0；
;;;   (b) 由这个视图再派生（flatten / ravel / reshape / view）必须能跑通，
;;;       因为用户拿到的就是这个视图。
;;; ------------------------------------------------------------------

(defun run-empty-axis-views ()
  (format t "~&--- A. 空轴视图：offset 不许变负，派生操作不许炸 ---%")
  (dolist (shape '((0) (0 3) (3 0) (0 3 2) (2 0 3) (2 3 0)))
    (let ((z (vt-zeros shape :dtype :float64))
          (label (format nil "~{~a~^-~}" shape)))
      ;; 全轴翻转
      (let ((f (vt-flip z)))
        (report (format nil "empty-view/~a/flip-offset" label)
                (zerop (vt-offset f))
                (format nil "offset=~a" (vt-offset f)))
        (report (format nil "empty-view/~a/flip-flatten" label)
                (equal (vt-shape (vt-flatten f)) (list 0)))
        (report (format nil "empty-view/~a/flip-ravel" label)
                (equal (vt-shape (vt-ravel f)) (list 0)))
        (report (format nil "empty-view/~a/flip-reshape" label)
                (equal (vt-shape (vt-reshape f (list 0))) (list 0)))
        (report (format nil "empty-view/~a/flip-copy-shape" label)
                (equal (vt-shape (vt-copy f)) shape)))
      ;; 逐轴翻转
      (dotimes (ax (length shape))
        (let ((f (vt-flip z :axis ax)))
          (report (format nil "empty-view/~a/flip~a-offset" label ax)
                  (zerop (vt-offset f))
                  (format nil "offset=~a" (vt-offset f)))
          (report (format nil "empty-view/~a/flip~a-flatten" label ax)
                  (equal (vt-shape (vt-flatten f)) (list 0)))))
      ;; 负步长切片：逐轴、以及全轴
      (dotimes (ax (length shape))
        (let ((specs (loop for d below (length shape)
                           collect (if (= d ax) '(nil nil -1) '(:all)))))
          (let ((s (apply #'vt-slice z specs)))
            (report (format nil "empty-view/~a/slice~a-offset" label ax)
                    (zerop (vt-offset s))
                    (format nil "offset=~a" (vt-offset s)))
            (report (format nil "empty-view/~a/slice~a-flatten" label ax)
                    (equal (vt-shape (vt-flatten s)) (list 0))))))
      ;; 空轴上取整数索引：越界必须报错（numpy 同样），但不许泄漏类型错误
      (report (format nil "empty-view/~a/index-empty-axis" label)
              (handler-case
                  (progn (apply #'vt-slice z
                                (loop for d in shape
                                      collect (if (zerop d) '(0) '(:all))))
                         nil)
                (error (e) (search "out of bounds" (princ-to-string e)))))))
  ;; 非空翻转的值仍然正确（防止修 offset 时把正路改坏）
  (let* ((x (vt-reshape (ff '(1d0 2d0 3d0 4d0 5d0 6d0)) '(2 3)))
         (f (vt-flip x :axis 1)))
    (ok-vals "empty-view/nonempty-flip-axis1" f '(3d0 2d0 1d0 6d0 5d0 4d0))
    (ok-shape "empty-view/nonempty-flip-axis1-shape" f '(2 3)))
  ;; 1D 负步长/整轴翻转的值（与 numpy x[::-1] 一致）
  (ok-vals "empty-view/nonempty-flip-all"
           (vt-flip (ff '(1d0 2d0 3d0 4d0))) '(4d0 3d0 2d0 1d0))
  ;; 空轴的 rot90 / fliplr / flipud：这两个入口都建在 flip 上
  (ok-shape "empty-view/rot90-(0 3)" (vt-rot90 (vt-zeros '(0 3) :dtype :float64)) '(3 0))
  (ok-shape "empty-view/fliplr-(3 0)" (vt-fliplr (vt-zeros '(3 0) :dtype :float64)) '(3 0))
  (ok-shape "empty-view/flipud-(0 3)" (vt-flipud (vt-zeros '(0 3) :dtype :float64)) '(0 3))
  (report "empty-view/fliplr-(3 0)-flatten"
          (equal (vt-shape (vt-flatten (vt-fliplr (vt-zeros '(3 0) :dtype :float64)))) '(0)))
  ;; 翻转视图再写回：padded 的 [::-1] 原地写必须落到正确位置（非空）
  (let ((a (ff '(1d0 2d0 3d0 4d0))))
    (let ((r (vt-flip a)))
      (setf (vt-ref r 0) 9d0))
    (ok-vals "empty-view/flip-write-through" a '(1d0 2d0 3d0 9d0))))

;;; ------------------------------------------------------------------
;;; B. 空输入的「摘要族」不许泄漏内部错误
;;; ------------------------------------------------------------------

(defun run-empty-summary ()
  (format t "~&--- B. 空输入的摘要族 ---%")
  ;; softmax / log-softmax：空轴上没有元素要算 → 输出同形状空张量
  (dolist (shape '((0) (0 3) (3 0)))
    (let ((label (format nil "~{~a~^-~}" shape)))
      (let ((s (vt-softmax (vt-zeros shape :dtype :float64))))
        (ok-shape (format nil "empty-summary/softmax~a-shape" label) s shape)
        (ok-dtype (format nil "empty-summary/softmax~a-dtype" label) s :float64))
      (let ((s (vt-log-softmax (vt-zeros shape :dtype :float64))))
        (ok-shape (format nil "empty-summary/logsoftmax~a-shape" label) s shape))))
  ;; float32 输入保持 float32（与 pack 的 dtype 规则一致）
  (ok-dtype "empty-summary/softmax-f32-dtype"
            (vt-softmax (vt-zeros '(0) :dtype :float32)) :float32)
  ;; 整型输入 → float64
  (ok-dtype "empty-summary/softmax-int64-dtype"
            (vt-softmax (vt-zeros '(0) :dtype :int64)) :float64)
  ;; :out 契约在空输入上同样成立
  (let ((o (vt-zeros '(0) :dtype :float64)))
    (report "empty-summary/softmax-out"
            (eq o (vt-softmax (vt-zeros '(0) :dtype :float64) :out o))))
  ;; histogram：numpy 给 3 个 0 箱 + 默认区间 [0, 1)
  (multiple-value-bind (c e) (vt-histogram (vt-zeros '(0) :dtype :float64) :bins 3)
    (ok-vals "empty-summary/hist-counts" c '(0 0 0))
    (ok-dtype "empty-summary/hist-dtype" c :int64)
    (report "empty-summary/hist-edges"
            (and (= 4 (vt-size e))
                 (bit= (vt-item (vt-slice e '(0 1))) 0d0)
                 (bit= (vt-item (vt-slice e '(3 4))) 1d0))
            (format nil "edges=~s" (vt-to-list e)))
    (report "empty-summary/hist-edges-middle"
            (and (num= (vt-item (vt-slice e '(1 2))) 0.3333333333333333d0)
                 (num= (vt-item (vt-slice e '(2 3))) 0.6666666666666666d0))
            (format nil "edges=~s" (vt-to-list e))))
  ;; 空输入 + 显式 range：区间照用
  (multiple-value-bind (c e) (vt-histogram (vt-zeros '(0) :dtype :float64)
                                          :bins 2 :range '(2d0 4d0))
    (ok-vals "empty-summary/hist-range-counts" c '(0 0))
    (ok-vals "empty-summary/hist-range-edges" e '(2d0 3d0 4d0)))
  ;; 空输入 + density：numpy 是 0/0 = NaN（不是 0）
  (let ((c (nth-value 0 (vt-histogram (vt-zeros '(0) :dtype :float64)
                                      :bins 3 :density t))))
    (report "empty-summary/hist-density-nan"
            (every (lambda (x) (nan-p x)) (flat c))
            (format nil "got ~s" (flat c)))))

;;; ------------------------------------------------------------------
;;; C. histogram 全等值的量级扫描
;;;
;;; 全等值时边界由 (±0.5) 造出来。这句 0.5 一旦写成单精度字面量，
;;; 1e8 以上就表示不出 ±0.5：边界退化成三个同一个数；1e18 时 (±0.5)
;;; 在 double 上也表示不出，numpy 在那时报「range 太小」而不是给 garbage。
;;; ------------------------------------------------------------------

(defun run-hist-single-value ()
  (format t "~&--- C. histogram 全等值 × 量级 ---%")
  (labels ((chk (val want-edges)
             (multiple-value-bind (c e)
                 (vt-histogram (vt-full '(5) val :dtype :int64) :bins 2)
               (ok-vals (format nil "hist-single/~a/counts" val) c '(0 5))
               (report (format nil "hist-single/~a/edges" val)
                       (and (= 3 (vt-size e))
                            (every #'bit= (flat e) want-edges))
                       (format nil "got ~s want ~s" (flat e) want-edges)))))
    (chk 5 '(4.5d0 5d0 5.5d0))
    (chk 7 '(6.5d0 7d0 7.5d0))
    ;; 1e8 / 1e10：单精度的 ±0.5 在这里完全丢失（ulp(1e8) in f32 = 8）
    (chk 100000000 '(99999999.5d0 1d8 100000000.5d0))
    (chk 10000000000 '(9999999999.5d0 1d10 10000000000.5d0))
    (chk 1000000000000000 '(999999999999999.5d0 1d15 1000000000000000.5d0)))
  ;; 1e18：double 的 ulp 已经是 128，±0.5 表示不出 → 箱宽为 0。
  ;; numpy 报「Too many bins for data range」，本库同样拒绝而不是给 garbage。
  (ok-err "hist-single/1e18-rejected"
          (lambda () (vt-histogram (vt-full '(5) 1000000000000000000 :dtype :int64)
                                   :bins 2))
          "vt-histogram:" "太小")
  ;; 全等值的浮点输入同样走这条路（0.1 的边界要与 numpy 逐位一致）
  (multiple-value-bind (c e)
      (vt-histogram (vt-full '(5) 0.1d0 :dtype :float64) :bins 2)
    (ok-vals "hist-single/0.1-counts" c '(0 5))
    (report "hist-single/0.1-edges"
            (every #'bit= (flat e) '(-0.4d0 0.09999999999999998d0 0.6d0))
            (format nil "got ~s" (flat e)))))

;;; ------------------------------------------------------------------
;;; D. 整型输入与浮点输入必须给出同一组边界
;;;
;;; 整数运算在 CL 里是精确有理数：(- max min) 是整数、(/ … bins) 是分数，
;;; 于是「整数数据」的边界按精确有理数算、再转 double，与 numpy 先转
;;; double 再算的结果差 1 ULP。用户看到的是「同一组数，dtype 一变边界就变」
;;; （CONVENTIONS §17.10.22 的纯度判据在 dtype 维度的延伸）。
;;; ------------------------------------------------------------------

(defun run-hist-int-vs-float ()
  (format t "~&--- D. histogram 整型/浮点边界一致 ---%")
  (dolist (n '(3 4 5 7 9 11))
    (dolist (bins '(1 2 3 4 5 7))
      (let* ((data (loop for i below n collect i))
             (ei (nth-value 1 (vt-histogram (fi data) :bins bins)))
             (ef (nth-value 1 (vt-histogram (ff (mapcar (lambda (x) (coerce x 'double-float)) data))
                                            :bins bins)))
             (bi (flat ei)) (bf (flat ef))
             (label (format nil "n=~a,bins=~a" n bins)))
        (report (format nil "hist-dtype/~a/edges-bit-identical" label)
                (and (= (length bi) (length bf))
                     (every #'bit= bi bf))
                (format nil "int=~s float=~s" bi bf))
        ;; 计数也必须一致
        (report (format nil "hist-dtype/~a/counts-eq" label)
                (equal (flat (nth-value 0 (vt-histogram (fi data) :bins bins)))
                       (flat (nth-value 0 (vt-histogram (ff (mapcar (lambda (x) (coerce x 'double-float)) data))
                                                        :bins bins))))))))
  ;; 已知精确值（numpy: np.histogram(np.arange(1,6), bins=3)）
  (report "hist-dtype/arange15-bins3-edges"
          (every #'bit= (flat (nth-value 1 (vt-histogram (fi '(1 2 3 4 5)) :bins 3)))
                 '(1d0 2.333333333333333d0 3.6666666666666665d0 5d0))
          (format nil "got ~s" (flat (nth-value 1 (vt-histogram (fi '(1 2 3 4 5)) :bins 3))))))

;;; ------------------------------------------------------------------
;;; E. density：总计 0 → NaN；不等宽边界按逐箱宽度归一
;;; ------------------------------------------------------------------

(defun run-hist-density ()
  (format t "~&--- E. histogram density ---%")
  ;; 总计为 0（空采样）：numpy 是 0/0 = NaN
  (let ((c (nth-value 0 (vt-histogram (vt-zeros '(0) :dtype :float64)
                                      :bins 2 :range '(0d0 4d0) :density t))))
    (report "hist-density/zero-total-nan" (every (lambda (x) (nan-p x)) (flat c))
            (format nil "got ~s" (flat c))))
  ;; 权重全零同样「没有数据」
  (let ((c (nth-value 0 (vt-histogram (ff '(1d0 2d0 3d0)) :bins 2 :density t
                                      :weights (fi '(0 0 0))))))
    (report "hist-density/zero-weights-nan" (every (lambda (x) (nan-p x)) (flat c))
            (format nil "got ~s" (flat c))))
  ;; 不等宽边界：密度必须按**逐箱**宽度归一，否则总和 ≠ 1
  ;; numpy: np.histogram([1,2,3], bins=[0,1,2,3,10], density=True)[0]
  ;;        -> [0, 1/3, 1/3, 1/21]
  (let ((c (nth-value 0 (vt-histogram (ff '(1d0 2d0 3d0))
                                      :bins '(0d0 1d0 2d0 3d0 10d0) :density t))))
    (ok-dtype "hist-density/uneven-dtype" c :float64)
    (ok-vals "hist-density/uneven" c '(0d0 0.3333333333333333d0
                                        0.3333333333333333d0
                                        0.047619047619047616d0))))
  ;; 等宽边界：density 的总和必须等于 1/binwidth
  ;; numpy: histogram(arange(4), bins=2, density=True) -> [1/3, 1/3]
  (let ((c (nth-value 0 (vt-histogram (vt-arange 4 :dtype :float64)
                                      :bins 2 :density t))))
    (ok-dtype "hist-density/even-dtype" c :float64)
    (ok-vals "hist-density/even" c '(0.3333333333333333d0 0.3333333333333333d0)))

;;; ------------------------------------------------------------------
;;; F. histogram 的 :weights
;;; ------------------------------------------------------------------

(defun run-hist-weights ()
  (format t "~&--- F. histogram :weights ---%")
  ;; numpy: np.histogram(np.arange(1,6.), bins=2, weights=[1,1,1,1,3])
  ;;        -> ([2, 5], int64)
  (multiple-value-bind (c e) (vt-histogram (ff '(1d0 2d0 3d0 4d0 5d0)) :bins 2
                                          :weights (fi '(1 1 1 1 3)))
    (ok-vals "hist-w/int-counts" c '(2 5))
    (ok-dtype "hist-w/int-dtype" c :int64)
    (ok-vals "hist-w/int-edges" e '(1d0 3d0 5d0)))
  ;; 浮点权重 → counts 变 float64（numpy 的 result_type 规则）
  ;; numpy: weights=[0.5,0.5,0.5,0.5,3.] -> ([1., 4.], float64)
  (multiple-value-bind (c e) (vt-histogram (ff '(1d0 2d0 3d0 4d0 5d0)) :bins 2
                                          :weights (ff '(0.5d0 0.5d0 0.5d0 0.5d0 3d0)))
    (ok-vals "hist-w/float-counts" c '(1d0 4d0))
    (ok-dtype "hist-w/float-dtype" c :float64)
    (ok-vals "hist-w/float-edges" e '(1d0 3d0 5d0)))
  ;; 显式边界 + 权重
  ;; numpy: histogram(np.arange(1,6.), bins=[1,2,4,6], weights=[1,1,1,1,3])
  ;;        -> [1, 2, 4]（1 → bin0；2,3 → bin1；4,5 → bin2）
  (ok-vals "hist-w/explicit-edges"
           (nth-value 0 (vt-histogram (ff '(1d0 2d0 3d0 4d0 5d0))
                                      :bins '(1d0 2d0 4d0 6d0)
                                      :weights (fi '(1 1 1 1 3))))
           '(1 2 4))
  ;; 权重 + density（总量 = 权重之和 7，箱宽 2）
  ;; numpy: ([2,5]/7/2, int64→float64) -> [0.142857…, 0.357142…]
  (let ((c (nth-value 0 (vt-histogram (ff '(1d0 2d0 3d0 4d0 5d0)) :bins 2
                                      :density t :weights (fi '(1 1 1 1 3))))))
    (ok-vals "hist-w/density" c '(0.14285714285714285d0 0.35714285714285715d0)))
  ;; 区间外的样本不计入（权重也不计）：0 与 1 都落进 [0,2)
  ;; numpy: histogram([0,1,5,9], bins=2, range=(0,4), weights=[100,1,2,3])
  ;;        -> [101, 0]
  (ok-vals "hist-w/out-of-range"
           (nth-value 0 (vt-histogram (ff '(0d0 1d0 5d0 9d0)) :bins 2
                                      :range '(0d0 4d0) :weights (fi '(100 1 2 3))))
           '(101 0))
  ;; 形状必须与输入完全一致（numpy: "weights should have the same shape as a"）
  (ok-err "hist-w/shape-mismatch"
          (lambda () (vt-histogram (ff '(1d0 2d0 3d0)) :bins 2 :weights (fi '(1 2))))
          "vt-histogram:" "weights" "不一致")
  (ok-err "hist-w/shape-mismatch-2d"
          (lambda () (vt-histogram (vt-reshape (ff '(1d0 2d0 3d0 4d0)) '(2 2))
                                   :bins 2 :weights (fi '(1 2 3 4))))
          "vt-histogram:" "weights")
  ;; 复数权重明确拒绝（numpy 同样）
  (ok-err "hist-w/complex-rejected"
          (lambda () (vt-histogram (ff '(1d0 2d0)) :bins 2
                                   :weights (vt-from-sequence
                                             (list (complex 1d0 0d0) (complex 1d0 0d0))
                                             :dtype :complex128)))
          "vt-histogram:" "weights")
  ;; 布尔权重：按 0/1 计数（numpy 的 b1 → int64）
  (ok-vals "hist-w/bool-weights"
           (nth-value 0 (vt-histogram (ff '(1d0 2d0 3d0)) :bins 2
                                      :range '(1d0 3d0)
                                      :weights (vt-from-sequence '(t nil t) :dtype :bool)))
           '(1 1)))

;;; ------------------------------------------------------------------
;;; G. 相位 −0.0
;;;
;;; (−0.0+0.0i) 是负实轴上的点，相位 π。用 (zerop re) 判「零」会把实部
;;; 的符号位吃掉，于是返回 0 —— 差 180°。
;;; ------------------------------------------------------------------

(defun run-angle-signed-zero ()
  (format t "~&--- G. angle 的 −0.0 ---%")
  (let* ((a (vt-from-sequence
             (list (complex 0d0 0d0) (complex -0d0 0d0)
                   (complex 0d0 -0d0) (complex -0d0 -0d0))
             :dtype :complex128))
         (r (vt-angle a))
         (got (flat r)))
    (ok-shape "angle/zero-combos-shape" r '(4))
    (ok-dtype "angle/zero-combos-dtype" r :float64)
    ;; numpy: [0.0, π, -0.0, -π]
    (report "angle/-0+0j-is-pi" (num= (nth 1 got) (coerce pi 'double-float))
            (format nil "got ~s" (nth 1 got)))
    (report "angle/+0-0j-is-neg-zero" (bit= (nth 2 got) -0d0)
            (format nil "got ~s" (nth 2 got)))
    (report "angle/-0-0j-is-neg-pi" (num= (nth 3 got) (- (coerce pi 'double-float)))
            (format nil "got ~s" (nth 3 got)))
    (report "angle/+0+0j-is-zero" (bit= (nth 0 got) 0d0)
            (format nil "got ~s" (nth 0 got))))
  ;; 负实轴 + 非零虚部仍然走 atan2（不能被零值特例带偏）
  (ok-vals "angle/neg-real-axis"
           (vt-angle (vt-from-sequence
                      (list (complex -1d0 0d0) (complex -1d0 -0d0) (complex 1d0 -0d0))
                      :dtype :complex128))
           (list (coerce pi 'double-float) (- (coerce pi 'double-float)) -0d0))
  ;; 非连续视图（走 vt-map 通用路径而不是特化内核）必须与连续路径一致
  (let* ((contig (vt-from-sequence
                  (list (complex -0d0 0d0) (complex 0d0 -0d0)) :dtype :complex128))
         (rev (vt-flip contig)))
    (report "angle/noncontig-matches-contig"
            (equal (mapcar #'bits (flat (vt-angle contig)))
                   (nreverse (mapcar #'bits (flat (vt-angle rev)))))
            (format nil "contig=~s rev=~s" (flat (vt-angle contig)) (flat (vt-angle rev)))))
  ;; 实数输入：angle(-0.0) = π（原本就对的路径，防止回归）
  (ok-vals "angle/real-neg-zero"
           (vt-angle (ff '(-0d0 0d0 -1d0)))
           (list (coerce pi 'double-float) 0d0 (coerce pi 'double-float)))
  ;; 度数制走同一内核的乘法路径
  (report "angle/deg-neg-zero"
          (num= (vt-item (vt-angle (vt-from-sequence (list (complex -0d0 0d0))
                                                     :dtype :complex128)
                                   :deg t))
                180d0)
          (format nil "got ~s"
                  (vt-item (vt-angle (vt-from-sequence (list (complex -0d0 0d0))
                                                       :dtype :complex128) :deg t)))))

;;; ------------------------------------------------------------------
;;; H. 空序列的 convolve / correlate
;;;
;;; numpy 对空序列报 ValueError（"a cannot be empty" / "v cannot be empty"）。
;;; 本库原来在 m = 0 时把 offset 算成 -1 去下标，抛的是
;;; 「The value -1 is not of type (UNSIGNED-BYTE 44)」。
;;; ------------------------------------------------------------------

(defun run-empty-convolve ()
  (format t "~&--- H. 空序列的 convolve / correlate ---%")
  (ok-err "convolve/empty-a"
          (lambda () (vt-convolve (vt-zeros '(0) :dtype :float64) (ff '(1d0))))
          "vt-convolve:" "不能为空")
  (ok-err "convolve/empty-v"
          (lambda () (vt-convolve (ff '(1d0)) (vt-zeros '(0) :dtype :float64)))
          "vt-convolve:" "不能为空")
  (ok-err "correlate/empty-a"
          (lambda () (vt-correlate (vt-zeros '(0) :dtype :float64) (ff '(1d0))))
          "vt-correlate:" "不能为空")
  (ok-err "correlate/empty-v"
          (lambda () (vt-correlate (ff '(1d0)) (vt-zeros '(0) :dtype :float64)))
          "vt-correlate:" "不能为空")
  ;; 复数路径不得给出不同的回答（两条 dtype 路径必须都报错）
  (ok-err "convolve/empty-complex"
          (lambda () (vt-convolve (vt-zeros '(0) :dtype :complex128)
                                  (vt-ones '(1) :dtype :complex128)))
          "不能为空")
  ;; 单元素边界仍然正常（1×1 卷积等于乘积）
  (ok-vals "convolve/single" (vt-convolve (ff '(3d0)) (ff '(4d0))) '(12d0)))

;;; ------------------------------------------------------------------
;;; I. 报错前缀
;;; ------------------------------------------------------------------

(defun run-error-prefix ()
  (format t "~&--- I. 报错前缀 ---%")
  (ok-err "prefix/gradient-too-short"
          (lambda () (vt-gradient (ff '(3d0))))
          "vt-gradient:" "太小")
  (ok-err "prefix/gradient-empty"
          (lambda () (vt-gradient (vt-zeros '(0) :dtype :float64)))
          "vt-gradient:")
  (ok-err "prefix/pad-empty-edge"
          (lambda () (vt-pad (vt-zeros '(0) :dtype :float64) '(2 2) :mode :edge))
          "vt-pad:" "空轴")
  (ok-err "prefix/pad-empty-reflect"
          (lambda () (vt-pad (vt-zeros '(0) :dtype :float64) '(2 2) :mode :reflect))
          "vt-pad:")
  (ok-err "prefix/pad-reflect-len1"
          (lambda () (vt-pad (ff '(1d0)) '(2 2) :mode :reflect))
          "vt-pad:" "reflect"))

;;; ------------------------------------------------------------------
;;; J. bins 参数的错类型
;;; ------------------------------------------------------------------

(defun run-bins-type ()
  (format t "~&--- J. bins 参数类型 ---%")
  ;; numpy: TypeError: `bins` must be an integer, a string, or an array
  (ok-err "bins/float-rejected"
          (lambda () (vt-histogram (ff '(1d0 2d0 3d0)) :bins 2.0d0))
          "vt-histogram:" "正整数" "边界序列")
  (ok-err "bins/zero-rejected"
          (lambda () (vt-histogram (ff '(1d0 2d0 3d0)) :bins 0))
          "vt-histogram:" "正整数")
  (ok-err "bins/negative-rejected"
          (lambda () (vt-histogram (ff '(1d0 2d0 3d0)) :bins -3))
          "vt-histogram:" "正整数")
  (ok-err "bins/single-edge-rejected"
          (lambda () (vt-histogram (ff '(1d0 2d0 3d0)) :bins '(1d0)))
          "vt-histogram:" "至少需要 2 个元素")
  (ok-err "bins/unsorted-edges-rejected"
          (lambda () (vt-histogram (ff '(1d0 2d0 3d0)) :bins '(3d0 2d0 1d0)))
          "vt-histogram:" "严格递增")
  (ok-err "bins/duplicate-edges-rejected"
          (lambda () (vt-histogram (ff '(1d0 2d0 3d0)) :bins '(1d0 1d0 2d0)))
          "vt-histogram:" "严格递增")
  ;; 合法形态全部走通：正整数 / 列表 / 向量 / 张量
  (ok-vals "bins/int" (nth-value 0 (vt-histogram (ff '(1d0 2d0 3d0)) :bins 2 :range '(1d0 3d0)))
           '(1 2))
  ;; numpy: histogram([1,2,3], bins=[0,2,4]) -> [1, 2]
  (ok-vals "bins/list" (nth-value 0 (vt-histogram (ff '(1d0 2d0 3d0)) :bins '(0d0 2d0 4d0)))
           '(1 2))
  (ok-vals "bins/vector" (nth-value 0 (vt-histogram (ff '(1d0 2d0 3d0))
                                                    :bins (vector 0d0 2d0 4d0)))
           '(1 2))
  (ok-vals "bins/tensor" (nth-value 0 (vt-histogram (ff '(1d0 2d0 3d0))
                                                    :bins (ff '(0d0 2d0 4d0))))
           '(1 2)))

;;; ------------------------------------------------------------------

(defun main ()
  (run-empty-axis-views)
  (run-empty-summary)
  (run-hist-single-value)
  (run-hist-int-vs-float)
  (run-hist-density)
  (run-hist-weights)
  (run-angle-signed-zero)
  (run-empty-convolve)
  (run-error-prefix)
  (run-bins-type)
  (format t "~&~%PASS=~d FAIL=~d~%" *pass* *fail*)
  (when (plusp *fail*)
    (format t "失败项：~%~{  ~a~%~}" (reverse *failures*))
    (sb-ext:exit :code 1)))

(main)
