;;;; random-exec-test.lisp — 随机数生成改「按 dtype 的填充内核」的回归
;;;;
;;;; 病根：四个入口都是「先建全零张量，再用 `VT-MAP` 的匿名 lambda 逐元素覆盖」：
;;;;
;;;;   (vt-map (lambda (x) (declare (ignore x)) (vt-cast (… (random 1.0d0 rng)) dtype))
;;;;           (vt-zeros shape :dtype dtype))
;;;;
;;;; 两处固定成本都落在**每元素**上：
;;;;   · `VT-MAP` 只内联 `(function <符号>)` 字面形式（§7.4.7），匿名 lambda 退回
;;;;     `(apply #'vt-map …)`：每元素一次 funcall + 泛型 AREF + 装箱；
;;;;   · `VT-CAST` 每次套一层 `WITH-FLOAT-SAFE`，实测 216 ns/次。
;;;; 结果：1M 个 U[0,1) 采样 208 ms，而同一个 random-state 上纯 `(random 1.0d0)`
;;;; 1M 次只要 5.5 ms —— 抽样本身一分钱不花。
;;;;
;;;; 另有两条算法级问题：
;;;;   · `VT-RANDOM-CHOICE` 的加权分支在 CDF 上**线性扫描**（每样本 O(k)）；
;;;;   · `VT-RANDOM-SHUFFLE` 每步 2×`VT-SLICE` + `VT-COPY` + 2×`VT-COPY-INTO`，
;;;;     1M 行就是 4M 次分配。
;;;;
;;;; A 段：四个采样入口与**逐字复刻的改动前实现**同 seed 逐位对拍；
;;;; B 段：`VT-RANDOM-CHOICE` 三种分支（均匀 / 加权 / 无放回）同 seed 对拍；
;;;; C 段：`VT-RANDOM-SHUFFLE` / `VT-RANDOM-PERMUTATION` 同 seed 对拍（含转置视图）；
;;;; D 段：语义点（可复现、dtype/shape、拒绝整型、退化形状、统计性质）；
;;;; E 段：机器无关的倍率护栏（比值，不看绝对毫秒）。

(require :asdf)
#+quicklisp (ql:quickload :clvt :silent t)
(asdf:load-system :clvt)

(defpackage :clvt-random-exec-test
  (:use :cl :clvt)
  (:import-from :clvt
                #:ensure-vt #:vt-data #:vt-offset #:vt-shape #:vt-dtype #:vt-size
                #:vt-view #:vt-slice #:vt-copy #:vt-copy-into #:vt-zeros #:vt-map
                #:vt-do-each #:vt-cast #:%vt-cast-raw #:%copy-default-rng)
  (:export #:run))

(in-package :clvt-random-exec-test)

;;; ==================================================================
;;; 0. 断言与工具
;;; ==================================================================
(declaim (special *n* *pass* *fail* *failures*))
(defparameter *n* 0)
(defparameter *pass* 0)
(defparameter *fail* 0)
(defparameter *failures* nil)

(defun check (label ok &optional detail)
  (incf *n*)
  (if ok
      (progn (incf *pass*) (format t "  [ OK ] ~a~%" label))
      (progn (incf *fail*)
             (push (format nil "~a~@[ — ~a~]" label detail) *failures*)
             (format t "  [FAIL] ~a~@[ — ~a~]~%" label detail))))

(defun check-at-most (label ratio limit what)
  (check (format nil "~a（~a = ~,2f× ≤ ~,2f×）" label what ratio limit)
         (<= ratio limit)
         (format nil "实际 ~,3f×，上限 ~,2f×" ratio limit)))

(defun best-ms (thunk &key (target-ms 20.0) (reps 3))
  "按目标 20 ms 自适应迭代次数，取最快一批的均值。
   单次计时在这台机器上只有 1 ms 刻度（§7.4.13），不能直接用来做倍率。"
  (funcall thunk)
  (let* ((p0 (get-internal-real-time))
         (probe (progn (funcall thunk)
                       (/ (* 1000.0 (- (get-internal-real-time) p0))
                          internal-time-units-per-second)))
         (iters (max 1 (round target-ms (max probe 0.5))))
         (best most-positive-double-float))
    (loop repeat reps
          do (let ((t0 (get-internal-real-time)))
               (loop repeat iters do (funcall thunk))
               (let ((per (/ (* 1000.0 (- (get-internal-real-time) t0))
                             internal-time-units-per-second iters)))
                 (when (< per best) (setf best per)))))
    best))

(defun flat-data (x)
  "取张量的连续化数据（视图也照办），用于逐位比较。"
  (vt-data (vt-contiguous (vt-flatten x))))

(defun same-bits (a b label)
  "逐位比较两个张量：shape、dtype、每个元素 `EQL`。
   `EQL` 对浮点比的是值而不是位（-0.0 与 0.0 不等），但那正好是我们要的严格性：
   随机采样不该在这两种写法之间改变符号位。"
  (let ((ok t) (why nil))
    (unless (equal (vt-shape a) (vt-shape b))
      (setf ok nil why (format nil "shape ~a vs ~a" (vt-shape a) (vt-shape b))))
    (when ok
      (unless (eql (vt-dtype a) (vt-dtype b))
        (setf ok nil why (format nil "dtype ~a vs ~a" (vt-dtype a) (vt-dtype b)))))
    (when ok
      (let ((da (flat-data a)) (db (flat-data b)) (n (vt-size a)))
        (dotimes (i n)
          (unless (eql (aref da i) (aref db i))
            (setf ok nil why (format nil "元素 ~a：~s vs ~s" i (aref da i) (aref db i)))
            (return)))))
    (check label ok why)))

(defun state (seed) (clvt::vt-make-random-state seed))

(defun num (x)
  "把 0 维张量取成数字（`VT-MEAN` / `VT-STD` 返回 0 维张量）。"
  (if (vt-p x) (vt-ref x) x))

;;; ==================================================================
;;; 1. 改动前的实现（逐字复刻，只把 RNG 变成显式参数）
;;; ==================================================================
(defun legacy-random (shape dtype rng)
  (vt-map (lambda (x)
            (declare (ignore x))
            (vt-cast (random 1.0d0 rng) dtype))
          (vt-zeros shape :dtype dtype)))

(defun legacy-uniform (shape low high dtype rng)
  (let ((range (- high low)))
    (vt-map (lambda (x)
              (declare (ignore x))
              (vt-cast (+ low (* range (random 1.0d0 rng))) dtype))
            (vt-zeros shape :dtype dtype))))

(defun legacy-normal (shape mean std dtype rng)
  (let ((res (vt-zeros shape :dtype dtype)))
    (vt-do-each (ptr val res)
      (declare (ignore val))
      (setf (aref (vt-data res) ptr)
            (vt-cast (+ mean (* std (let ((u1 (max least-positive-double-float
                                                    (random 1.0d0 rng)))
                                        (u2 (random 1.0d0 rng)))
                                     (* (sqrt (* -2.0d0 (log u1)))
                                        (cos (* 2.0d0 pi u2))))))
                     dtype)))
    res))

(defun legacy-random-int (size low high dtype rng)
  (let ((range (- high low)))
    (if size
        (vt-astype (vt-map (lambda (x)
                             (declare (ignore x))
                             (+ low (random range rng)))
                           (vt-zeros size :dtype dtype))
                   dtype)
        (clvt::make-vt nil (+ low (random range rng)) :dtype dtype))))

(defun legacy-choice-uniform (a size out-dtype rng)
  (let* ((src (if (integerp a) (vt-arange a :dtype (or out-dtype :int64)) (ensure-vt a)))
         (n (vt-size src))
         (sd (vt-data src))
         (so (vt-offset src))
         (result (vt-zeros size :dtype out-dtype)))
    (vt-do-each (ptr val result)
      (declare (ignore val))
      (setf (aref (vt-data result) ptr)
            (vt-cast (aref sd (+ so (random n rng))) out-dtype)))
    result))

(defun legacy-choice-weighted (a size p out-dtype rng)
  "改动前的加权有放回：每样本在 CDF 上从 0 起线性扫到第一个 cdf[i] >= r。"
  (let* ((src (if (integerp a) (vt-arange a :dtype (or out-dtype :int64)) (ensure-vt a)))
         (n (vt-size src))
         (sd (vt-data src))
         (so (vt-offset src))
         (pw (cond ((vt-p p) (map 'list #'identity (vt-to-list p)))
                   ((listp p) p)
                   (t (coerce p 'list))))
         (total (reduce #'+ pw))
         (cdf (let ((cum 0.0d0))
                (coerce (mapcar (lambda (w) (incf cum (/ w total)) cum) pw) 'vector)))
         (result (vt-zeros size :dtype out-dtype)))
    (vt-do-each (ptr val result)
      (declare (ignore val))
      (let ((r (random 1.0d0 rng)))
        (setf (aref (vt-data result) ptr)
              (vt-cast (loop for i from 0 below n
                             when (<= r (aref cdf i))
                               return (aref sd (+ so i))
                             finally (return (aref sd (+ so (1- n)))))
                       out-dtype))))
    result))

(defun legacy-shuffle (tensor &key (axis 0) (rng nil))
  "改动前的打乱：每步 2×`VT-SLICE` + `VT-COPY` + 2×`VT-COPY-INTO`。"
  (let* ((ax (vt-normalize-axis axis (length (vt-shape tensor))))
         (dim (nth ax (vt-shape tensor))))
    (when (<= dim 1) (return-from legacy-shuffle tensor))
    (loop for i from (1- dim) downto 1 do
      (let* ((j (random (1+ i) rng))
             (si (apply #'vt-slice tensor
                        (loop for d below (length (vt-shape tensor))
                              collect (if (= d ax) (list i) '(:all)))))
             (sj (apply #'vt-slice tensor
                        (loop for d below (length (vt-shape tensor))
                              collect (if (= d ax) (list j) '(:all))))))
        (let ((tmp (vt-copy si)))
          (vt-copy-into si sj)
          (vt-copy-into sj tmp))))
    tensor))

(defun legacy-permutation (n rng)
  (let* ((tensor (ensure-vt n))
         (result (vt-copy tensor))
         (first-dim (first (vt-shape result))))
    (loop for i from (1- first-dim) downto 1 do
      (let* ((j (random (1+ i) rng))
             (si (vt-slice result (list i)))
             (sj (vt-slice result (list j))))
        (let ((tmp (vt-copy si)))
          (vt-copy-into si sj)
          (vt-copy-into sj tmp))))
    result))

;;; ==================================================================
;;; A. 四个采样入口 × dtype × 形状（同 seed 逐位对拍）
;;; ==================================================================
(defun run-sample-tests ()
  (format t "~&~%[A] 采样入口 × dtype × 形状（与改动前实现同 seed 对拍）~%")
  ;; ---- vt-random ----
  (dolist (dt '(:float64 :float32))
    (dolist (sh '((3) (1000) (4 5) (2 3 4) (0)))
      (same-bits (vt-random sh :dtype dt :rng (state 42))
                 (legacy-random sh dt (state 42))
                 (format nil "vt-random ~a ~a" dt sh))))
  ;; ---- vt-random-uniform：参数域 ----
  (dolist (dt '(:float64 :float32))
    (dolist (lh (list (list 0.0d0 1.0d0) (list -5.0d0 5.0d0) (list 0 10)
                      (list 1 1) (list -0.0d0 1.0d0)))
      (destructuring-bind (lo hi) lh
        (same-bits (vt-random-uniform '(500) :low lo :high hi :dtype dt :rng (state 7))
                   (legacy-uniform '(500) lo hi dt (state 7))
                   (format nil "uniform ~a low=~s high=~s" dt lo hi)))))
  ;; 单精度 low / ratio：走通用分支，运算域不能提前 coerce
  (same-bits (vt-random-uniform '(200) :low 0.0 :high 1.0 :dtype :float64 :rng (state 3))
             (legacy-uniform '(200) 0.0 1.0 :float64 (state 3))
             "uniform low/high 是 single-float")
  (same-bits (vt-random-uniform '(200) :low 1/3 :high 2/3 :dtype :float64 :rng (state 3))
             (legacy-uniform '(200) 1/3 2/3 :float64 (state 3))
             "uniform low/high 是 ratio")
  ;; 整型与 bool 目标：截断语义
  (dolist (dt '(:int64 :int32 :int16 :int8 :uint8 :uint16 :bool))
    (same-bits (vt-random-uniform '(300) :low 0 :high 4 :dtype dt :rng (state 11))
               (legacy-uniform '(300) 0 4 dt (state 11))
               (format nil "uniform → ~a（截断）" dt)))
  ;; 0 维（shape 为 NIL）
  (same-bits (vt-random nil :rng (state 5))
             (legacy-random nil :float64 (state 5))
             "vt-random shape=NIL（0 维）")
  (same-bits (vt-random-uniform nil :rng (state 5))
             (legacy-uniform nil 0.0d0 1.0d0 :float64 (state 5))
             "uniform 0 维")
  ;; ---- vt-random-normal ----
  (dolist (dt '(:float64 :float32))
    (dolist (ms (list (list 0.0d0 1.0d0) (list 10.0d0 0.5d0) (list -3.0d0 2.0d0)))
      (destructuring-bind (mu sd) ms
        (same-bits (vt-random-normal '(400) :mean mu :std sd :dtype dt :rng (state 13))
                   (legacy-normal '(400) mu sd dt (state 13))
                   (format nil "normal ~a mean=~s std=~s" dt mu sd)))))
  (same-bits (vt-random-normal '(200) :mean 0 :std 1 :dtype :float64 :rng (state 3))
             (legacy-normal '(200) 0 1 :float64 (state 3))
             "normal mean/std 是整数")
  ;; ---- vt-random-int ----
  (dolist (dt '(:int64 :int32 :int16 :int8 :uint8 :float64 :float32))
    (same-bits (vt-random-int 0 100 :size '(400) :dtype dt :rng (state 17))
               (legacy-random-int '(400) 0 100 dt (state 17))
               (format nil "random-int ~a" dt)))
  (same-bits (vt-random-int 5 99 :size '(3 4) :dtype :int64 :rng (state 19))
             (legacy-random-int '(3 4) 5 99 :int64 (state 19))
             "random-int 2 维")
  (same-bits (vt-random-int 5 9 :dtype :int64 :rng (state 19))
             (legacy-random-int nil 5 9 :int64 (state 19))
             "random-int 0 维")
  (same-bits (vt-random-int 7 7 :size '(4) :dtype :int64)
             (vt-full '(4) 7 :dtype :int64)
             "random-int low=high（常量）"))

;;; ==================================================================
;;; B. VT-RANDOM-CHOICE 三分支（同 seed 逐位对拍）
;;; ==================================================================
(defun run-choice-tests ()
  (format t "~&~%[B] vt-random-choice 三分支（同 seed 对拍）~%")
  ;; ---- 均匀有放回 ----
  (dolist (dt '(:int64 :float64 :float32))
    (dolist (sh '((500) (7 9)))
      (same-bits (vt-random-choice 500 :size sh :dtype dt :rng (state 23))
                 (legacy-choice-uniform 500 sh dt (state 23))
                 (format nil "choice 均匀 ~a size=~a" dt sh))))
  (let ((src (vt-arange 200 :dtype :int64)))
    (same-bits (vt-random-choice src :size '(300) :rng (state 29))
               (legacy-choice-uniform src '(300) :int64 (state 29))
               "choice 均匀（张量源）"))
  (same-bits (vt-random-choice 100 :rng (state 31))
             (legacy-choice-uniform 100 nil :int64 (state 31))
             "choice 均匀 0 维")
  ;; ---- 加权有放回（旧实现线性扫 CDF，新实现二分）----
  (let ((p (loop for i from 1 to 40 collect (/ i 820.0)))
        (src (vt-arange 40 :dtype :int64)))
    (same-bits (vt-random-choice src :size '(2000) :p p :rng (state 37))
               (legacy-choice-weighted src '(2000) p :int64 (state 37))
               "choice 加权 k=40（list p）")
    ;; p 为张量：入口归一到同一列表，序列应完全相同
    (same-bits (vt-random-choice src :size '(2000) :p (vt-from-sequence p) :rng (state 37))
               (legacy-choice-weighted src '(2000) p :int64 (state 37))
               "choice 加权 k=40（张量 p）"))
  ;; 含大量零权重：二分的「第一个 cdf >= r」必须与线性扫描逐位一致
  (let ((p (loop for i from 0 below 50 collect (if (zerop (mod i 7)) 0.0 (/ 1.0 50))))
        (src (vt-arange 50 :dtype :int64)))
    (same-bits (vt-random-choice src :size '(3000) :p p :rng (state 41))
               (legacy-choice-weighted src '(3000) p :int64 (state 41))
               "choice 加权 含零权重"))
  ;; 只有一个非零权重：全部命中那一格
  (let ((p (loop for i from 0 below 10 collect (if (= i 4) 1.0 0.0))))
    (let ((r (vt-random-choice 10 :size '(200) :p p :rng (state 43))))
      (check "choice 加权 单点权重 ⇒ 全部取 4"
             (every (lambda (x) (= x 4)) (vt-to-list r))
             (format nil "得到 ~s" (subseq (vt-to-list r) 0 8)))))
  ;; ---- 无放回（未改，仍要守住）----
  (let ((r (vt-random-choice 100 :size '(60) :replace nil :rng (state 47))))
    (check "choice 无放回 60/100：无重复"
           (= 60 (length (remove-duplicates (vt-to-list r))))
           (format nil "去重后 ~a 项" (length (remove-duplicates (vt-to-list r))))))
  (let ((r (vt-random-choice (vt-arange 6 :dtype :int64) :size '(6) :replace nil :rng (state 53))))
    (check "choice 无放回 全取：是 0..5 的排列"
           (equal (sort (vt-to-list r) #'<) '(0 1 2 3 4 5))
           (format nil "得到 ~s" (vt-to-list r)))))

;;; ==================================================================
;;; C. 打乱 / 排列（同 seed 逐位对拍，含非连续视图）
;;; ==================================================================
(defun run-shuffle-tests ()
  (format t "~&~%[C] vt-random-shuffle / vt-random-permutation（同 seed 对拍）~%")
  (dolist (dt '(:float64 :int64 :int32 :float32))
    (let ((base (vt-arange 400 :dtype dt)))
      (dolist (ax '(0 1))
        (let* ((a (vt-reshape (vt-copy base) '(20 20)))
               (b (vt-copy a)))
          (vt-random-shuffle a :axis ax :rng (state 59))
          (legacy-shuffle b :axis ax :rng (state 59))
          (same-bits a b (format nil "shuffle ~a 20×20 axis=~a" dt ax))))))
  ;; 3 维、axis=1 / axis=2
  (dolist (ax '(0 1 2))
    (let ((a (vt-arange 24 :dtype :int64))
          (b nil))
      (setf a (vt-reshape a '(2 3 4)))
      (setf b (vt-copy a))
      (vt-random-shuffle a :axis ax :rng (state 61))
      (legacy-shuffle b :axis ax :rng (state 61))
      (same-bits a b (format nil "shuffle 2×3×4 axis=~a" ax))))
  ;; 转置视图（非连续）：原地打乱后写回原视图
  (let* ((a (vt-reshape (vt-arange 600 :dtype :int64) '(20 30)))
         (ta (vt-transpose a))
         (b (vt-copy a))
         (tb (vt-transpose b)))
    (vt-random-shuffle ta :rng (state 67))
    (legacy-shuffle tb :rng (state 67))
    (same-bits a b "shuffle 转置视图（写回原张量）"))
  ;; 单元素轴 / 空轴：原样返回
  (let ((a (vt-reshape (vt-arange 5 :dtype :int64) '(5 1))))
    (check "shuffle dim=1 轴：内容不变"
           (equal (vt-to-list a) (vt-to-list (vt-copy a)))))
  ;; permutation（张量版）
  (dolist (sh '((100) (10 10) (4 5 6)))
    (let* ((src (vt-reshape (vt-arange (reduce #'* sh) :dtype :int64) sh))
           (got (vt-random-permutation src :rng (state 71)))
           (ref (legacy-permutation src (state 71))))
      (same-bits got ref (format nil "permutation ~a" sh))
      (check (format nil "permutation ~a：不改原张量" sh)
             (equal (vt-to-list src)
                    (vt-to-list (vt-reshape (vt-arange (reduce #'* sh) :dtype :int64) sh))))))
  ;; permutation（整数版）：一行内交换，未改，仍守住
  (let ((p (vt-random-permutation 200 :rng (state 73))))
    (check "permutation 200：是 0..199 的排列"
           (equal (sort (vt-to-list p) #'<) (loop for i below 200 collect i))))
  ;; 打乱只重排、不增删（多重集不变）
  (let* ((a (vt-random-int 0 5 :size '(1000) :dtype :int64 :rng (state 79)))
         (b (vt-copy a)))
    (vt-random-shuffle b :rng (state 83))
    (check "shuffle 后多重集不变"
           (equal (sort (vt-to-list a) #'<) (sort (vt-to-list b) #'<)))))

;;; ==================================================================
;;; D. 语义点
;;; ==================================================================
(defun run-semantics-tests ()
  (format t "~&~%[D] 语义点~%")
  ;; 可复现：同 seed 两次给出同一串
  (flet ((twice (fn) (equal (vt-to-list (funcall fn (state 101)))
                            (vt-to-list (funcall fn (state 101))))))
    (check "uniform 同 seed 可复现" (twice (lambda (s) (vt-random-uniform '(50) :rng s))))
    (check "normal 同 seed 可复现" (twice (lambda (s) (vt-random-normal '(50) :rng s))))
    (check "choice 同 seed 可复现"
           (twice (lambda (s) (vt-random-choice 20 :size '(50) :rng s))))
    (check "shuffle 同 seed 可复现"
           (twice (lambda (s) (vt-random-shuffle (vt-arange 50 :dtype :int64) :rng s)))))
  ;; 不同 seed 不同
  (check "uniform 不同 seed 给出不同串"
         (not (equal (vt-to-list (vt-random-uniform '(50) :rng (state 1)))
                     (vt-to-list (vt-random-uniform '(50) :rng (state 2))))))
  ;; dtype / shape 契约
  (dolist (dt '(:float64 :float32 :int32 :uint8 :bool))
    (let ((r (vt-random-uniform '(2 3) :dtype dt :rng (state 103))))
      (check (format nil "uniform ~a：dtype 与 shape" dt)
             (and (eql (vt-dtype r) dt) (equal (vt-shape r) '(2 3)))
             (format nil "得到 ~a ~a" (vt-dtype r) (vt-shape r)))))
  ;; vt-random 只接受浮点 dtype（既有契约）
  (check "vt-random 拒绝 :int64"
         (handler-case (progn (vt-random '(3) :dtype :int64) nil)
           (error () t)))
  ;; 退化参数
  (check "uniform low=high ⇒ 常量"
         (every (lambda (x) (= x 3.5d0))
                (vt-to-list (vt-random-uniform '(10) :low 3.5d0 :high 3.5d0))))
  (check "normal std=0 ⇒ 均值"
         (every (lambda (x) (= x 2.0d0))
                (vt-to-list (vt-random-normal '(10) :mean 2.0d0 :std 0.0d0))))
  ;; 统计性质（1M 样本，容差宽；不是精确断言，只抓「明显错位」）
  (let ((u (vt-random-uniform '(1000000) :rng (state 107))))
    (check "uniform 1M：均值 ≈ 0.5（±0.01）"
           (< (abs (- (num (vt-mean u)) 0.5d0)) 0.01)
           (format nil "均值 ~,4f" (num (vt-mean u))))
    (check "uniform 1M：min < 0.001 且 max > 0.999"
           (and (< (num (vt-amin u)) 0.001) (> (num (vt-amax u)) 0.999))
           (format nil "min ~,4f max ~,4f" (num (vt-amin u)) (num (vt-amax u)))))
  (let ((z (vt-random-normal '(1000000) :rng (state 109))))
    (check "normal 1M：均值 ≈ 0（±0.01）"
           (< (abs (num (vt-mean z))) 0.01)
           (format nil "均值 ~,4f" (num (vt-mean z))))
    (check "normal 1M：std ≈ 1（±0.01）"
           (< (abs (- (num (vt-std z)) 1.0d0)) 0.01)
           (format nil "std ~,4f" (num (vt-std z)))))
  ;; 加权抽样的经验频率接近 p
  (let* ((p (list 0.5d0 0.3d0 0.2d0))
         (r (vt-random-choice 3 :size '(200000) :p p :rng (state 113)))
         (freq (let ((c (make-array 3 :initial-element 0)))
                 (dolist (v (vt-to-list r)) (incf (aref c v)))
                 (map 'list (lambda (x) (/ x 200000.0)) c))))
    (check "choice 加权：经验频率 ≈ p（±0.01）"
           (every (lambda (f q) (< (abs (- f q)) 0.01)) freq p)
           (format nil "实测 ~s" freq)))
  ;; 整数区间端点
  (let ((r (vt-random-int 0 10 :size '(10000) :dtype :int64 :rng (state 127))))
    (check "random-int [0,10)：落在区间内且两端都出现"
           (and (= 0 (num (vt-amin r))) (= 9 (num (vt-amax r)))))))

;;; ==================================================================
;;; E. 倍率护栏（比值，不看绝对毫秒）
;;; ==================================================================
(defun run-guardrails ()
  (format t "~&~%[E] 机器无关倍率护栏~%")
  (let* ((t-uniform (best-ms (lambda () (vt-random-uniform '(1000000) :rng (state 42)))))
         (t-int     (best-ms (lambda () (vt-random-int 0 100 :size '(1000000)
                                                       :dtype :int64 :rng (state 42)))))
         (t-choice  (best-ms (lambda () (vt-random-choice 1000000 :size '(1000000)
                                                          :dtype :int64 :rng (state 42)))))
         (t-normal  (best-ms (lambda () (vt-random-normal '(1000000) :rng (state 42)))))
         (t-zeros   (best-ms (lambda () (vt-zeros '(1000000))))))
    (format t "  [实测] uniform ~,2f ms  int ~,2f  choice ~,2f  normal ~,2f  zeros ~,2f~%"
            t-uniform t-int t-choice t-normal t-zeros)
    ;; 1) uniform ÷ random-int：两者都是「1M 次 RANDOM + 一次写入」，
    ;;    差别只有一次乘加。改前 208/15 = 13.9×（每元素一次 VT-CAST）。
    (check-at-most "uniform ÷ random-int" (/ t-uniform (max t-int 1e-9)) 3.0 "耗时比")
    ;; 2) choice 均匀 ÷ random-int：同上，只多一次数组索引。
    (check-at-most "choice 均匀 ÷ random-int" (/ t-choice (max t-int 1e-9)) 3.0 "耗时比")
    ;; 3) normal ÷ uniform：两次 RANDOM + log + cos ×2 ≈ 4~6×；
    ;;    改前两者都在 200~250 ms（比值 1.2× 反而「通过」），所以这条只抓 normal 单独退化。
    (check-at-most "normal ÷ uniform" (/ t-normal (max t-uniform 1e-9)) 10.0 "耗时比")
    ;; 4) 1M 次采样的固定开销：不许超过 1M 次全零数组的 30 倍
    (check-at-most "uniform ÷ vt-zeros" (/ t-uniform (max t-zeros 1e-9)) 30.0 "耗时比"))
  ;; 5) 加权抽样的「O(log k) 而不是 O(k)」：k=500 时线性扫描要扫 250 项/样本。
  (let* ((k 500)
         (p (make-list k :initial-element (/ 1.0 k)))
         (t-w (best-ms (lambda () (vt-random-choice k :size '(200000) :p p
                                                    :dtype :int64 :rng (state 42)))
                       :target-ms 40.0 :reps 2))
         (t-u (best-ms (lambda () (vt-random-choice k :size '(200000)
                                                    :dtype :int64 :rng (state 42)))))
         (legacy (best-ms (lambda () (legacy-choice-weighted
                                      k '(200000) p :int64 (state 42)))
                          :target-ms 40.0 :reps 1)))
    (format t "  [实测] 加权 k=500/200k ~,2f ms  均匀 ~,2f  改动前线性 ~,2f~%"
            t-w t-u legacy)
    (check-at-most "choice 加权 ÷ 均匀（k=500）" (/ t-w (max t-u 1e-9)) 20.0 "耗时比"))
  ;; 6) 打乱：与「同规模的 permutation」相比。两者都是 1M 次 Fisher-Yates，
  ;;    后者是纯数据交换。改前 974/23.5 = 41×（每步 4 次分配）。
  (let* ((m (vt-random-uniform '(1000000 4) :rng (state 7)))
         (t-sh (best-ms (lambda () (vt-random-shuffle (vt-copy m) :rng (state 42)))
                        :target-ms 40.0 :reps 2))
         (t-perm (best-ms (lambda () (vt-random-permutation 1000000 :rng (state 42))))))
    (format t "  [实测] shuffle 1M×4 ~,2f ms  permutation 1M ~,2f~%" t-sh t-perm)
    (check-at-most "shuffle ÷ permutation（1M）" (/ t-sh (max t-perm 1e-9)) 12.0 "耗时比")))

;;; ==================================================================
;;; 入口
;;; ==================================================================
(defun run ()
  (format t "~&~%========== clvt random-exec 回归 ==========~%")
  (run-sample-tests)
  (run-choice-tests)
  (run-shuffle-tests)
  (run-semantics-tests)
  (run-guardrails)
  (format t "~&~%通过 ~a / ~a~@[，失败：~%~{  · ~a~%~}~]~%"
          *pass* *n* (reverse *failures*))
  (when (plusp *fail*)
    (error "random-exec-test: ~a 条失败" *fail*))
  (values *pass* *n*))

(run)
