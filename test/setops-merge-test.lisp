;;;; setops-merge-test.lisp — 集合运算改「有序归并」+ 会员判定键就绪的回归
;;;;
;;;; 两处病根：
;;;;
;;;;   1. src/setops.lisp  %MEMBERSHIP-KEY / %MEMBERSHIP-TABLE / %MEMBERSHIP-MAP
;;;;      三处「每元素一次通用件」，最要命的是**键的形状**：
;;;;        · 浮点键用 DOUBLE-FLOAT 对象本身（或任何以浮点为值的东西）→
;;;;          SXHASH 分桶只看低位，而规整值（0.5 步长、整数化的 double、
;;;;          f32 升上来的值）位模式低位全是 0 → 全撞一个桶。
;;;;          20k 键建表 678 ms、200k 次查询 11.5 s（随机尾数的 double 只要
;;;;          4 ms / 14 ms）。后果：INTERSECT1D 100k×10k 3.09 s、200k×20k
;;;;          12.3 s（4 倍时间 / 2 倍数据 = 平方），与元素是 f64 还是 f32 无关。
;;;;        · 位模式整数还不够：低位同样全零，未交换时 807 ms / 14.0 s；
;;;;          高低 32 位交换后 3 ms / 9 ms。
;;;;        · 查表用匿名闭包交给 VT-MAP → 每元素一次 funcall + 泛型 aref。
;;;;      分别改为「位模式整数键 + 高低 32 位交换」「数组层面规整到同一比较域」
;;;;      「具名算子 + VT-FAST-MAP」。
;;;;
;;;;   2. src/setops.lisp  INTERSECT1D / SETDIFF1D / UNION1D / SETXOR1D
;;;;      四个函数都走「哈希表 + 逐元素 GETHASH」，而两侧的 VT-UNIQUE 输出
;;;;      **本来就是升序**（NaN 在末尾）。键修好之后，剩下的成本是
;;;;      「键规范化 + 一次 GETHASH」（位模式键是 bignum，一次约 120 ns）：
;;;;      intersect 233 → 188 ms、setdiff 245 → 202、union 324 → 205、
;;;;      setxor 493 → 195（原来还要 COERCE 到 list 再用 list 比较器排一次）。
;;;;
;;;; A 段：四个集合运算与**哈希参考实现**（O(n+m)）对拍，覆盖 12 组边界；
;;;; B 段：IN1D / ISIN 的键就绪路径与改动前的闭包实现逐位对拍；
;;;; C 段：钉 numpy 2.4.6 的若干语义点（±0、NaN、空集、dtype 沿用左值）；
;;;; D 段：四条机器无关的倍率护栏（比值，不看绝对毫秒）。

(require :asdf)
#+quicklisp (ql:quickload :clvt :silent t)
(asdf:load-system :clvt)

(defpackage :clvt-setops-test
  (:use :cl :clvt)
  (:import-from :clvt
                #:ensure-vt #:vt-data #:vt-shape #:vt-dtype #:vt-size
                #:%membership-key #:%membership-table #:%membership-map
                #:%membership-prepare #:vt-map)
  (:export #:run))

(in-package :clvt-setops-test)

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

(defvar *zero* 0.0d0)
(defvar *nan* (sb-int:with-float-traps-masked (:invalid) (/ *zero* *zero*)))
(defvar *pinf* sb-ext:double-float-positive-infinity)
(defvar *ninf* (- sb-ext:double-float-positive-infinity))

(defun nanp (x) (and (floatp x) (sb-ext:float-nan-p x)))

(defun sem-eq (x y)
  "会员判定的等价关系：NaN 与自己相等、±0 相等、整数与同值浮点相等（numpy 同）。"
  (if (and (numberp x) (numberp y))
      (cond ((nanp x) (nanp y))
            ((nanp y) nil)
            (t (= x y)))
      (equal x y)))

(defun norm-key (x)
  "参考实现的哈希键：把 `%MEMBERSHIP-KEY` 的等价关系编码成 EQL/EQUAL 键。"
  (cond ((nanp x) :nan)
        ((and (numberp x) (zerop x)) 0)
        ((floatp x) (coerce x 'double-float))
        (t x)))

(defun ref-set (a b mode)
  "哈希参考实现（O(n+m)）。A/B 为 list；返回升序去重（NaN 末尾）的 list。"
  (let* ((ua (remove-duplicates a :test #'sem-eq))
         (ub (remove-duplicates b :test #'sem-eq))
         (hb (make-hash-table :test #'equal :size (max 16 (length ub))))
         (ha (make-hash-table :test #'equal :size (max 16 (length ua))))
         (r nil))
    (dolist (x ub) (setf (gethash (norm-key x) hb) t))
    (dolist (x ua) (setf (gethash (norm-key x) ha) t))
    (ecase mode
      (:intersect (dolist (x ua) (when (gethash (norm-key x) hb) (push x r))))
      (:diff (dolist (x ua) (unless (gethash (norm-key x) hb) (push x r))))
      (:symdiff (dolist (x ua) (unless (gethash (norm-key x) hb) (push x r)))
                (dolist (x ub) (unless (gethash (norm-key x) ha) (push x r))))
      (:union (dolist (x ua) (push x r)) (dolist (x ub) (push x r))))
    (let ((r (if (eq mode :union) (remove-duplicates r :test #'sem-eq) r)))
      (sort r (lambda (x y) (cond ((nanp x) nil) ((nanp y) t) (t (< x y))))))))

(defun same-list (r ref label)
  "比较 VT 与参考 list：长度相同且逐项 SEM-EQ（`EQUAL` 对 -0.0 / 0.0 给 NIL）。"
  (let ((got (vt-to-list r)))
    (check label
           (and (= (length got) (length ref)) (every #'sem-eq got ref))
           (format nil "got ~a 项 ~s，ref ~a 项 ~s"
                   (length got) (subseq got 0 (min 6 (length got)))
                   (length ref) (subseq ref 0 (min 6 (length ref)))))))

(defun all-four (a b tag)
  (same-list (vt-intersect1d a b) (ref-set a b :intersect) (format nil "~a：intersect" tag))
  (same-list (vt-setdiff1d a b) (ref-set a b :diff) (format nil "~a：setdiff" tag))
  (same-list (vt-union1d a b) (ref-set a b :union) (format nil "~a：union" tag))
  (same-list (vt-setxor1d a b) (ref-set a b :symdiff) (format nil "~a：setxor" tag)))

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
    ;; 用 LOOP 而不是 (DOTIMES (I …))：后者若写 (declare (ignore i)) 会招
    ;; 「I is being set even though it was declared to be ignored」。
    (loop repeat reps
          do (let ((t0 (get-internal-real-time)))
               (loop repeat iters do (funcall thunk))
               (let ((per (/ (* 1000.0 (- (get-internal-real-time) t0))
                             internal-time-units-per-second iters)))
                 (when (< per best) (setf best per)))))
    best))

;;; ==================================================================
;;; A. 四个集合运算 × 边界（与哈希参考实现对拍）
;;; ==================================================================
(defun run-set-op-tests ()
  (format t "~&~%[A] 集合运算 × 边界（参考实现 = 哈希 O(n+m)）~%")
  (all-four '(3 1 2 2 5 -0.0 0.0) '(2 4 5 5) "f64 含 ±0")
  (all-four (list 1.0 *nan* 2.0 *nan* 3.0) (list *nan* 2.0 9.0) "f64 含 NaN")
  (all-four (list *pinf* 1.0 *ninf*) (list *pinf* *ninf* 7.0) "f64 含 ±Inf")
  (all-four '(5 3 1) '(4 3 2) "i64")
  (all-four '(7 7 7) '(8) "i64 左全重复")
  (all-four '(1) '(1) "单元素相同")
  (all-four '(1) '(2) "单元素不同")
  (all-four nil '(1 2) "左空")
  (all-four '(1 2) nil "右空")
  (all-four nil nil "两侧空")
  (all-four (loop for i from 0 below 2000 collect (mod i 37))
            (loop for i from 0 below 500 collect (mod (* i 3) 101))
            "2000×500 整数（多重复）")
  (let ((a (make-list 2000 :initial-element 0.5d0))
        (b (make-list 2000 :initial-element 0.5d0)))
    (all-four a b "全同值 2000×2000"))
  ;; 大规模随机（与参考实现一致）
  (let* ((a (vt-random-uniform 20000))
         (b (vt-random-uniform 2000))
         (la (coerce (vt-to-list a) 'list))
         (lb (coerce (vt-to-list b) 'list)))
    (dolist (m '(:intersect :diff :union :symdiff))
      (let ((r (ecase m
                 (:intersect (vt-intersect1d a b))
                 (:diff (vt-setdiff1d a b))
                 (:union (vt-union1d a b))
                 (:symdiff (vt-setxor1d a b)))))
        (same-list r (ref-set la lb m) (format nil "20k 随机：~a" m)))))
  ;; f32 / 回退 dtype / 混合 dtype
  (let ((a (vt-astype (vt-from-sequence '(1.5 2.5 3.5 3.5)) :float32))
        (b (vt-astype (vt-from-sequence '(2.5 9.5)) :float32)))
    (same-list (vt-intersect1d a b) '(2.5) "f32 intersect")
    (same-list (vt-union1d a b) '(1.5 2.5 3.5 9.5) "f32 union")
    (check "f32 结果 dtype 保持 :float32"
           (eq (vt-dtype (vt-intersect1d a b)) :float32)))
  (let ((a (vt-from-sequence '(1 2 3 3) :dtype :int8))
        (b (vt-from-sequence '(3 4) :dtype :int8)))
    (same-list (vt-intersect1d a b) '(3) "int8 intersect（回退路径）")
    (same-list (vt-union1d a b) '(1 2 3 4) "int8 union（回退路径）")
    (same-list (vt-setdiff1d a b) '(1 2) "int8 setdiff（回退路径）")
    (same-list (vt-setxor1d a b) '(1 2 4) "int8 setxor（回退路径）"))
  (let* ((a (vt-from-sequence '(1 2 3) :dtype :int64))
         (b (vt-from-sequence '(2.0d0 4.0d0)))
         (r (vt-intersect1d a b)))
    (same-list r '(2.0d0) "i64 × f64 intersect（比较域 = double）")
    ;; 契约：intersect/setdiff/setxor 沿用**左值** dtype（改前如此，本轮不改）
    (check "i64 × f64 的结果 dtype 沿用左值 :int64"
           (eq (vt-dtype r) :int64)
           (format nil "实际 ~a" (vt-dtype r))))
  (let* ((a (vt-from-sequence '(1 2 3) :dtype :int32))
         (b (vt-from-sequence '(2 3 4) :dtype :int32)))
    (same-list (vt-intersect1d a b) '(2 3) "int32 intersect（归并内核）")
    (check "int32 结果 dtype 保持 :int32"
           (eq (vt-dtype (vt-intersect1d a b)) :int32)))
  ;; 视图输入（非连续）
  (let* ((m (vt-reshape (vt-random-uniform 12) '(3 4)))
         (v (vt-transpose m))
         (vl (coerce (vt-to-list (vt-flatten v)) 'list)))
    (same-list (vt-intersect1d v v) (ref-set vl vl :intersect)
               "转置视图（2 维）intersect 自身 = 展平去重升序")
    (check "集合运算的输出恒为 1 维"
           (equal (vt-shape (vt-intersect1d v v))
                  (list (length (ref-set vl vl :intersect)))))))

;;; ==================================================================
;;; B. IN1D / ISIN：键就绪路径 vs 改动前的闭包实现
;;; ==================================================================
(defun legacy-in1d (elem test &key invert)
  "改动前的实现：`%MEMBERSHIP-KEY` 逐元素 + 匿名闭包 + VT-MAP。"
  (let* ((e (ensure-vt elem))
         (t2 (ensure-vt test))
         (fm (or (subtypep (array-element-type (vt-data e)) 'float)
                 (subtypep (array-element-type (vt-data t2)) 'float)))
         (tab (%membership-table (vt-unique t2) fm)))
    (with-float-safe
      (vt-map (lambda (x)
                (let ((hit (if (gethash (%membership-key x fm) tab) 1 0)))
                  (if invert (- 1 hit) hit)))
              e :dtype :bool))))

(defun legacy-map (elem tab fm)
  "改动前的映射部分：匿名闭包交给 `VT-MAP`（每元素一次 funcall + 泛型 AREF）。
   只复刻「遍历 + 查表」，表由调用方建好 —— 用于隔离映射层的成本。"
  (with-float-safe
    (vt-map (lambda (x) (if (gethash (%membership-key x fm) tab) 1 0))
            elem :dtype :bool)))

(defun same-bool (r ref label)
  (let ((a (vt-to-list r)) (b (vt-to-list ref)))
    (check label (equal a b) (format nil "got ~s，ref ~s" a b))))

(defun run-membership-tests ()
  (format t "~&~%[B] IN1D / ISIN：键就绪路径与改动前的闭包实现逐位对拍~%")
  (let* ((elems (list 1.0 *nan* -0.0 0.0 3.5 *pinf* *ninf* 7.0))
         (tests (list 1.0 0.0 *nan* 3.5 *pinf* 99.0))
         (e (vt-from-sequence elems))
         (t2 (vt-from-sequence tests)))
    (same-bool (vt-in1d e t2) (legacy-in1d e t2) "f64 in1d（NaN / ±0 / ±Inf）")
    (same-bool (vt-isin e t2) (legacy-in1d e t2) "f64 isin 同结果")
    (same-bool (vt-isin e t2 :invert t) (legacy-in1d e t2 :invert t) "f64 isin :invert"))
  ;; f32（键必须与 f64 表对齐）
  (let* ((e (vt-astype (vt-from-sequence '(1.5 2.5 3.5 -0.0)) :float32))
         (t2 (vt-astype (vt-from-sequence '(2.5 0.0)) :float32)))
    (same-bool (vt-in1d e t2) (legacy-in1d e t2) "f32 in1d（键尺度对齐）"))
  ;; i64 / int32 / int8 / bool
  (dolist (dt '(:int64 :int32 :int8))
    (let ((e (vt-astype (vt-from-sequence '(1 2 3 4)) dt))
          (t2 (vt-astype (vt-from-sequence '(2 4 9)) dt)))
      (same-bool (vt-in1d e t2) (legacy-in1d e t2) (format nil "~a in1d" dt))))
  (let ((e (vt-from-sequence '(1 0 1))) (t2 (vt-from-sequence '(1))))
    (same-bool (vt-in1d e t2) (legacy-in1d e t2) "bool in1d"))
  ;; 混合 dtype（int 数组 vs 浮点集合 → 全 double 域）
  (let ((e (vt-from-sequence '(1 2 3) :dtype :int64))
        (t2 (vt-from-sequence '(2.0d0 3.0d0))))
    (same-bool (vt-in1d e t2) (legacy-in1d e t2) "i64 × f64 in1d"))
  ;; 形状保持 + 空
  (let ((e (vt-reshape (vt-random-uniform 12) '(3 4)))
        (t2 (vt-random-uniform 5)))
    (check "in1d 保持输入形状"
           (equal (vt-shape (vt-in1d e t2)) '(3 4)))
    (check "in1d 结果 dtype 恒为 :bool"
           (eq (vt-dtype (vt-in1d e t2)) :bool))
    (check "空 elem → 空 :bool"
           (equal (vt-to-list (vt-in1d (vt-from-sequence nil) t2)) nil))))

;;; ==================================================================
;;; C. numpy 2.4.6 钉死
;;; ==================================================================
(defun run-numpy-pins ()
  (format t "~&~%[C] numpy 2.4.6 钉死~%")
  ;; np.intersect1d([3,1,2,2,5,-0.0,0.0],[2,4,5,5]) → [2., 5.]
  (same-list (vt-intersect1d '(3 1 2 2 5 -0.0 0.0) '(2 4 5 5)) '(2 5) "pin: intersect")
  ;; np.setdiff1d(...) → [-0., 1., 3.]（±0 折叠，保留先出现的 -0.0）
  (same-list (vt-setdiff1d '(3 1 2 2 5 -0.0 0.0) '(2 4 5 5)) '(-0.0 1 3) "pin: setdiff")
  ;; np.union1d(...) → [-0., 1., 2., 3., 4., 5.]
  (same-list (vt-union1d '(3 1 2 2 5 -0.0 0.0) '(2 4 5 5)) '(-0.0 1 2 3 4 5) "pin: union")
  ;; np.setxor1d(...) → [-0., 1., 3., 4.]
  (same-list (vt-setxor1d '(3 1 2 2 5 -0.0 0.0) '(2 4 5 5)) '(-0.0 1 3 4) "pin: setxor")
  ;; NaN：unique 把多个 NaN 折叠为一个；相交时 NaN 与 NaN 视为相等
  (let ((r (vt-intersect1d (list *nan* 1.0) (list *nan* 2.0))))
    (check "pin: intersect 里 NaN 与 NaN 相等（折叠成一个）"
           (and (= (vt-size r) 1) (nanp (first (vt-to-list r))))
           (format nil "实际 ~s" (vt-to-list r))))
  ;; np.isin([nan, 0.0, -0.0, 1.0], [nan, 0.0]) → [True, True, True, False]
  (same-bool (vt-in1d (list *nan* 0.0 -0.0 1.0) (list *nan* 0.0))
             (vt-from-sequence '(1 1 1 0) :dtype :bool)
             "pin: isin 的 NaN / ±0")
  ;; 空集：np.intersect1d([], [1,2]) → []
  (check "pin: 空交集长度为 0"
         (zerop (vt-size (vt-intersect1d nil '(1 2)))))
  (same-list (vt-union1d nil '(2 1)) '(1 2) "pin: 空并集 = 非空侧（升序）"))

;;; ==================================================================
;;; D. 倍率护栏（机器无关）
;;; ==================================================================
(defun run-ratio-guards ()
  (format t "~&~%[D] 倍率护栏~%")
  (let* ((a (vt-random-uniform 1000000))
         (b (vt-random-uniform 100000))
         (t-u1 (best-ms (lambda () (vt-unique a))))
         (t-u2 (best-ms (lambda () (vt-unique b))))
         (t-int (best-ms (lambda () (vt-intersect1d a b))))
         (t-xor (best-ms (lambda () (vt-setxor1d a b)))))
    ;; 归并本身应当只占「两次 sort + run-length」的一小部分：
    ;; 改后约 1.1×（unique 1M 142 + unique 100k 17.5 → intersect 188）。
    (check-at-most "intersect(1M,100k) ÷ (unique(1M)+unique(100k))"
                   (/ t-int (max (+ t-u1 t-u2) 1e-9)) 1.4 "耗时比")
    ;; 对称差不再走「coerce 到 list + list 比较器排序」：改前是 intersect 的 2.2×
    (check-at-most "setxor ÷ intersect" (/ t-xor (max t-int 1e-9)) 1.2 "耗时比"))
  ;; f32 的键必须是 O(1)：改前（位模式低位全 0）是平方复杂度，
  ;; 100k 就要 3.0 s；现在与 f64 同数量级。
  (let* ((a (vt-random-uniform 200000))
         (b (vt-random-uniform 20000))
         (af (vt-astype a :float32))
         (bf (vt-astype b :float32))
         (t-f64 (best-ms (lambda () (vt-in1d a b))))
         (t-f32 (best-ms (lambda () (vt-in1d af bf)))))
    (check-at-most "in1d(f32) ÷ in1d(f64)" (/ t-f32 (max t-f64 1e-9)) 3.0 "耗时比"))
  ;; 会员判定的**映射**部分不再用匿名闭包。表由外面建好，只比「遍历 + 查表」：
  ;; 闭包路径每元素一次 funcall + 泛型 AREF + 装箱（1M 元素 142 ms），
  ;; 具名算子走 VT-FAST-MAP 的内联内核（80 ms）。用 VT-IN1D 整体比会因为
  ;; 建表成本占比不小而在小规模上落到 1.00×（1 ms 时钟分辨不出差异）。
  (let* ((a (vt-random-uniform 1000000))
         (b (vt-random-uniform 100000))
         (tab (%membership-table (vt-unique b) t)))
    (check-at-most "in1d 映射（具名算子 ÷ 改动前闭包）"
                   (/ (best-ms (lambda () (%membership-map a tab t nil)))
                      (max (best-ms (lambda () (legacy-map a tab t))) 1e-9))
                   0.90 "耗时比"))
  ;; 键的散布不能依赖「值的形状」：随机尾数的 double 在新旧两版键上都是线性的，
  ;; 规整值（0.5 步长 —— 计数/金额/坐标/采样点的常态）才会暴露退化。
  ;; 这条护栏直接编码「规整数据也必须 O(1)」：两条路径的耗时比应接近 1。
  (let* ((ar (vt-arange 200000 :step 0.5 :dtype :float64))
         (br (vt-arange 20000 :start 100 :step 5 :dtype :float64))
         (an (vt-random-uniform 200000))
         (bn (vt-random-uniform 20000)))
    (check-at-most "in1d(规整值) ÷ in1d(随机值)"
                   (/ (best-ms (lambda () (vt-in1d ar br)))
                      (max (best-ms (lambda () (vt-in1d an bn))) 1e-9))
                   3.0 "耗时比")))

;;; ==================================================================
;;; 入口
;;; ==================================================================
(defun run ()
  (setf *n* 0 *pass* 0 *fail* 0 *failures* nil)
  (run-set-op-tests)
  (run-membership-tests)
  (run-numpy-pins)
  (run-ratio-guards)
  (format t "~&~%==================== setops-merge-test ====================~%")
  (format t "通过 ~a / ~a~%" *pass* *n*)
  (when *failures*
    (format t "失败 ~a 条：~%~{  · ~a~%~}" *fail* (reverse *failures*)))
  (format t "===========================================================~%")
  (when (plusp *fail*) (error "setops-merge-test: ~a 条失败" *fail*))
  (values *pass* *n*))

;;; ------------------------------------------------------------------
;;; 自运行入口：run-tests.sh 只 (load) 本文件、不调用 RUN。
;;; 失败时 RUN 用 ERROR 退出 → sbcl 退出码非 0 → 运行器判 FAIL。
;;; ------------------------------------------------------------------
(run)
