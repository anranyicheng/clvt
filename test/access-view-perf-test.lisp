;;;; access-view-perf-test.lisp — 访问 / 切片 / 视图 / 连续性 的性能与逐位一致护栏
;;;;
;;;; 为什么需要这套
;;;; ----------------
;;;; 本轮优化的全部对象都是**只影响性能、不影响语义**的代码（CONVENTIONS §1
;;;; 铁律 1）：形状乘积、C 连续判定、单元素访问、视图构造、只读广播检查。
;;;; 这类改动与 §7.4 的快路径同源 —— 改错了（例如连续判定漏掉 dim=1 的跳过
;;;; 规则、单元素访问的 dtype 分派与物理存储对不上）结果会悄悄变错，而
;;;; **所有纯功能测试仍然绿**；改慢了（例如把 `vt-contiguous-p` 退回
;;;; `(nth i shape)` 的 O(rank²) 写法）同样无人发现。
;;;;
;;;; 因此本套件分两半：
;;;;   A 段（逐位一致）—— 每个被改的内核都与「朴素参考实现」对拍：
;;;;      尺寸、连续判定、取数、写数、只读判定、视图变换后的落地内容。
;;;;   B 段（性能护栏）—— 只写**倍率**，不写绝对毫秒（不可移植，见
;;;;      §16.16 第 7 条）。比值两边同机同进程，机器快慢基本抵消。
(require :asdf)
#+quicklisp (ql:quickload :clvt)
(asdf:load-system :clvt)

(defpackage :clvt-access-view-perf
  (:use :cl :clvt)
  (:export #:run))

(in-package :clvt-access-view-perf)

(defvar *pass* 0)
(defvar *fail* 0)
(defvar *failures* nil)

(defun ok (label)
  (incf *pass*)
  (format t "  [ OK ] ~a~%" label)
  t)

(defun ng (label fmt &rest args)
  (incf *fail*)
  (let ((msg (apply #'format nil fmt args)))
    (push (format nil "~a — ~a" label msg) *failures*)
    (format t "  [FAIL] ~a~%         ~a~%" label msg)
    nil))

(defun check (label test)
  (if test (ok label) (ng label "断言为假")))

(defun check-ratio (label ratio limit what)
  (if (and (numberp ratio) (<= ratio limit))
      (progn (incf *pass*)
             (format t "  [ OK ] ~a（~a = ~,2f× ≤ ~,1f×）~%" label what (float ratio) limit)
             t)
      (ng label "~a = ~,2f× > ~,1f×" what (float ratio) limit)))

;;; ------------------------------------------------------------------
;;; 计时（与 perf-fastpath-test.lisp 同一套：校准批量 + 取最小值）
;;; ------------------------------------------------------------------
(defun %calibrate (thunk)
  (let ((inner 1))
    (loop
      (let ((t0 (get-internal-real-time)))
        (dotimes (i inner) (funcall thunk))
        (let ((dt (- (get-internal-real-time) t0)))
          (cond ((>= dt 20000) (return inner))     ; 本机 1ms 刻度 ⇒ 撑到 20ms
                ((zerop dt) (setf inner (* inner 8)))
                (t (setf inner (max (1+ inner)
                                    (ceiling (* inner 30000) dt))))))))))

(defun bench-us (thunk &key (reps 7))
  "返回单次调用的**最小**耗时（微秒）。最小值而非中位数：本机是共享 VM，
   耗时分布双峰（外部抢占），最小值最接近「无干扰」的那一次。"
  (funcall thunk)
  (let ((inner (%calibrate thunk))
        (best most-positive-fixnum))
    (dotimes (r reps)
      (let ((t0 (get-internal-real-time)))
        (dotimes (i inner) (funcall thunk))
        (setf best (min best (/ (float (- (get-internal-real-time) t0) 1.0d0)
                                inner)))))
    best))

;;; ------------------------------------------------------------------
;;; 朴素参考实现（不碰 strides，只按逻辑坐标算）
;;; ------------------------------------------------------------------
(defun ref-size (shape)
  (reduce #'* shape :initial-value 1))

(defun ref-contiguous-p (shape strides)
  "CONVENTIONS §2.2 的 C 连续定义，逐字照抄成「末轴起严格递减」的参考版。"
  (if (or (null shape) (some #'zerop shape))
      t
      (let ((expected 1) (contig t))
        (loop for d in (reverse shape)
              for s in (reverse strides)
              do (cond ((= d 1))
                       ((= s expected) (setf expected (* expected d)))
                       (t (setf contig nil))))
        contig)))

(defun ref-offset (shape strides offset idxs)
  "把逻辑坐标换算成物理下标（负索引按 numpy 规则回绕）。"
  (+ offset
     (loop for d in shape
           for s in strides
           for i in idxs
           sum (* (mod i d) s))))

(defun ref-view-content (vt)
  "视图的**逻辑内容**，按行主序展开成一维列表（与物理布局无关）。"
  (let* ((shape (vt-shape vt))
         (rank (length shape))
         (size (ref-size shape))
         (strides (vt-strides vt))
         (data (vt-data vt))
         (out '()))
    (when (plusp size)
      (let ((idx (make-list rank :initial-element 0)))
        (dotimes (k size)
          (push (aref data (ref-offset shape strides (vt-offset vt) idx)) out)
          ;; 里程计：末轴先动
          (loop for d from (1- rank) downto 0
                do (let ((v (1+ (nth d idx))))
                     (if (< v (nth d shape))
                         (progn (setf (nth d idx) v) (return))
                         (setf (nth d idx) 0)))))))
    (nreverse out)))

(defun same-bits-list (a b)
  (and (= (length a) (length b))
       (every (lambda (x y)
                (cond ((and (typep x 'double-float) (typep y 'double-float))
                       (= (sb-kernel:double-float-bits x) (sb-kernel:double-float-bits y)))
                      (t (eql x y))))
              a b)))

(defun dfs (xs)
  "把整数列表变成 double-float 列表（EQL 比较需要同类型）。"
  (mapcar (lambda (x) (coerce x 'double-float)) xs))

;;; ------------------------------------------------------------------
;;; A1 — 形状乘积
;;; ------------------------------------------------------------------
(defun a1-shape-product ()
  (format t "~%[A1] %shape-product / vt-size ≡ (reduce #'* shape)~%")
  (let ((all-ok t))
    (dolist (shape '(nil (0) (1) (7) (3 4) (1 1) (2 3 4) (5 0 3) (2 3 4 5)
                     (1 1 1 1) (100 100) (2 1 3 1)))
      (let ((got (clvt::vt-shape-to-size shape))
            (want (ref-size shape)))
        (unless (= got want)
          (setf all-ok nil)
          (ng (format nil "A1 shape ~a" shape) "得到 ~a，参考 ~a" got want)))
      ;; 与张量上的 VT-SIZE 必须一致
      (let* ((vt (clvt::make-vt (or shape nil) 0.0d0)))
        (unless (= (vt-size vt) (ref-size shape))
          (setf all-ok nil)
          (ng (format nil "A1 vt-size shape ~a" shape)
              "得到 ~a，参考 ~a" (vt-size vt) (ref-size shape)))))
    (when all-ok (ok "A1 形状乘积与 reduce 一致（12 种形状，含 0 与单例轴）"))))

;;; ------------------------------------------------------------------
;;; A2 — C 连续判定：覆盖全部视图形态
;;; ------------------------------------------------------------------
(defun a2-contiguous-p ()
  (format t "~%[A2] vt-contiguous-p ≡ 参考实现~%")
  (let* ((a (clvt::make-vt '(4 6) 0.0d0))
         (c3 (clvt::make-vt '(3 4 5) 0.0d0))
         (v5 (clvt::make-vt '(7) 0.0d0))
         (scalar (vt-reshape (clvt::make-vt '(1) 0.0d0) '()))
         (views (list (cons "基础 2D" a)
                      (cons "行切片 a[1:3,:]" (vt-slice a '(1 3) '(:all)))
                      (cons "列切片 a[:,2:5]" (vt-slice a '(:all) '(2 5)))
                      (cons "步进 a[::2,::2]" (vt-slice a '(nil nil 2) '(nil nil 2)))
                      (cons "转置" (vt-transpose a))
                      (cons "转置后切片" (vt-slice (vt-transpose a) '(1 4) '(:all)))
                      (cons "翻转 axis0" (vt-flip a :axis 0))
                      (cons "翻转 axis1" (vt-flip a :axis 1))
                      (cons "全翻转" (vt-flip a))
                      (cons "单列 a[:,0]（降维）" (vt-slice a '(:all) '(0)))
                      (cons "单行 a[0,:]（降维）" (vt-slice a '(0) '(:all)))
                      (cons "单列再展开（dim=1 轴）"
                            (vt-slice a '(:all) '(0 1)))
                      (cons "广播 (4,6)←(1,6)" (vt-broadcast-to (vt-slice a '(0 1) '(:all)) '(4 6)))
                      (cons "广播 (4,6)←(4,1)" (vt-broadcast-to (vt-slice a '(:all) '(0 1)) '(4 6)))
                      (cons "新轴 a[:,newa,:]" (vt-slice a '(:all) '(:newa) '(:all)))
                      (cons "1D 连续" v5)
                      (cons "1D 步进" (vt-slice v5 '(nil nil 2)))
                      (cons "1D 翻转" (vt-flip v5 :axis 0))
                      (cons "3D 连续" c3)
                      (cons "3D 置换" (vt-transpose c3 '(2 0 1)))
                      (cons "3D 子块" (vt-slice c3 '(0 2) '(1 3) '(2 5)))
                      (cons "空 (0,3)" (clvt::make-vt '(0 3) 0.0d0))
                      (cons "空 (2,0,3)" (clvt::make-vt '(2 0 3) 0.0d0))
                      (cons "零维" scalar))))
    (dolist (pair views)
      (let* ((name (car pair))
             (v (cdr pair))
             (got (vt-contiguous-p v))
             (want (ref-contiguous-p (vt-shape v) (vt-strides v))))
        (if (eq got want)
            (ok (format nil "A2 ~a" name))
            (ng (format nil "A2 ~a" name)
                "得到 ~a，参考 ~a（shape=~a strides=~a）"
                got want (vt-shape v) (vt-strides v)))))))

;;; ------------------------------------------------------------------
;;; A3 — 单元素读：与「坐标 → 物理下标」参考换算逐值一致
;;; ------------------------------------------------------------------
(defun a3-vt-ref ()
  (format t "~%[A3] vt-ref / %vt-ref-raw ≡ 坐标换算~%")
  (let ((a (clvt::make-vt '(5 7) 0.0d0)))
    ;; 填成 i*100+j，便于一眼看出取错轴
    (dotimes (i 5) (dotimes (j 7) (setf (vt-ref a i j) (coerce (+ (* i 100) j) 'double-float))))
    (setf (vt-ref a 0 0) 0.0d0)
    (let ((views (list (cons "连续" a)
                       (cons "转置" (vt-transpose a))
                       (cons "行切片" (vt-slice a '(1 5) '(:all)))
                       (cons "步进" (vt-slice a '(nil nil 2) '(nil nil 2)))
                       (cons "列翻转" (vt-flip a :axis 1))
                       (cons "3D 视图" (vt-reshape a '(5 7 1))))))
      (dolist (pair views)
        (let* ((name (car pair))
               (v (cdr pair))
               (shape (vt-shape v))
               (strides (vt-strides v))
               (okp t))
          (dotimes (k (min 12 (ref-size shape)))
            (let* ((idxs (loop for d in shape collect (mod (+ k 1) d)))
                   (want (aref (vt-data v) (ref-offset shape strides (vt-offset v) idxs)))
                   (got (apply #'vt-ref v idxs)))
              (unless (eql got want)
                (setf okp nil)
                (ng (format nil "A3 ~a idx=~a" name idxs)
                    "vt-ref=~a 参考=~a" got want))))
          (when okp (ok (format nil "A3 ~a（逐坐标一致）" name)))))
      ;; 负索引与边界
      (check "A3 负索引 a[-1,-1] ≡ a[4,6]" (eql (vt-ref a -1 -1) (vt-ref a 4 6)))
      (check "A3 转置负索引" (eql (vt-ref (vt-transpose a) -1 -2)
                                  (vt-ref (vt-transpose a) 6 3)))
      ;; :bool 的逻辑视图（a[1,0] = 100 → 真）
      (let ((m (vt-astype (vt-slice a '(:all) '(0)) :bool)))
        (check "A3 :bool 逻辑视图返回 T" (eq (vt-ref m 1) t))
        (check "A3 :bool 逻辑视图返回 NIL" (null (vt-ref m 0)))
        (check "A3 :bool 物理视图仍是 0/1" (eql (clvt::%vt-ref-raw m 1) 1)))
      ;; 错误契约
      (flet ((err-p (thunk)
               (handler-case (progn (funcall thunk) nil)
                 (error () t))))
        (check "A3 越界报错" (err-p (lambda () (vt-ref a 5 0))))
        (check "A3 越界报错（负向）" (err-p (lambda () (vt-ref a 0 -8))))
        (check "A3 索引数过多报错" (err-p (lambda () (vt-ref a 0 0 0))))
        (check "A3 索引数过少报错" (err-p (lambda () (vt-ref a 0))))
        (check "A3 空张量取数报错"
               (err-p (lambda () (vt-ref (clvt::make-vt '(0) 0.0d0) 0))))
        (check "A3 非整数索引报错（布尔掩码）"
               (err-p (lambda () (vt-ref a (vt-astype a :bool)))))))))

;;; ------------------------------------------------------------------
;;; A4 — 单元素写：11 种 dtype × 值类型（含 T/NIL/复数）
;;; ------------------------------------------------------------------
(defun a4-setf-vt-ref ()
  (format t "~%[A4] (setf vt-ref) 全 dtype 写读往返~%")
  (dolist (dtype '(:float64 :float32 :int64 :int32 :int16 :int8
                   :uint16 :uint8 :bool :complex128 :complex64))
    (let* ((tensor (clvt::make-vt '(3 4) 0 :dtype dtype))
           (int-p (member dtype '(:int64 :int32 :int16 :int8 :uint16 :uint8)))
           ;; 整型目标只接受整数值：`COERCE` 到 (SIGNED-BYTE n) 对 1.5d0
           ;; 直接报错（CL 的 coerce 要求「数学值不变」，浮点→整数不满足）。
           ;; 这是本库既有契约，本轮未改。
           (vals (cond ((member dtype '(:uint16 :uint8)) (list 1 2 7))
                       (int-p (list 1 -2 7))
                       (t (list 1.5d0 -2.25d0 7))))
           (okp t))
      (dolist (val vals)
        (setf (vt-ref tensor 1 2) val)
        (let ((want (if (eq dtype :bool)
                        (if (zerop (clvt::%bool-flag val)) nil t)
                        (clvt::vt-cast val dtype)))
              (got (vt-ref tensor 1 2)))
          (unless (eql got want)
            (setf okp nil)
            (ng (format nil "A4 ~a ← ~a" dtype val) "得到 ~a，期望 ~a" got want))))
      (when okp (ok (format nil "A4 ~a 数值写入与 vt-cast 一致" dtype)))
      ;; 整型目标写浮点：必须确定性报错（不是静默截断，也不是写坏位模式）
      (when int-p
        (if (handler-case (progn (setf (vt-ref tensor 0 0) 1.5d0) nil)
              (error () t))
            (ok (format nil "A4 ~a ← 浮点 报错（契约不变）" dtype))
            (ng (format nil "A4 ~a ← 浮点" dtype) "应当报错却静默写入了")))))
  ;; 布尔目标：T / NIL / 0 / 非零
  (let ((b (clvt::make-vt '(2 2) 0 :dtype :bool)))
    (setf (vt-ref b 0 0) t)
    (setf (vt-ref b 0 1) nil)
    (setf (vt-ref b 1 0) 0.0d0)
    (setf (vt-ref b 1 1) 2.5d0)
    (check "A4 :bool ← T/NIL/0/2.5" (and (eq (vt-ref b 0 0) t)
                                          (null (vt-ref b 0 1))
                                          (null (vt-ref b 1 0))
                                          (eq (vt-ref b 1 1) t))))
  ;; 复数目标：复数写入保虚部；实数写入虚部为 0
  (let ((c (clvt::make-vt '(2) 0 :dtype :complex128)))
    (setf (vt-ref c 0) #c(1.0d0 2.0d0))
    (setf (vt-ref c 1) 3.0d0)
    (check "A4 complex128 ← 复数" (eql (vt-ref c 0) #c(1.0d0 2.0d0)))
    (check "A4 complex128 ← 实数" (eql (vt-ref c 1) #c(3.0d0 0.0d0))))
  ;; 实数目标 ← 复数：取实部（§16.5）
  (let ((f (clvt::make-vt '(1) 0.0d0)))
    (setf (vt-ref f 0) #c(1.5d0 9.0d0))
    (check "A4 float64 ← 复数取实部" (eql (vt-ref f 0) 1.5d0)))
  ;; 只读广播视图必须拒绝写入（H4）
  (let ((bc (vt-broadcast-to (vt-slice (clvt::make-vt '(1 3) 0.0d0) '(0) '(:all)) '(4 3))))
    (check "A4 广播视图写入报错"
           (handler-case (progn (setf (vt-ref bc 1 1) 1.0d0) nil)
             (error () t))))
  ;; 转置视图写入落在同一个底层单元上
  (let* ((a (clvt::make-vt '(3 3) 0.0d0))
         (at (vt-transpose a)))
    (setf (vt-ref at 1 2) 42.0d0)
    (check "A4 转置视图写入穿透底层" (eql (vt-ref a 2 1) 42.0d0))))

;;; ------------------------------------------------------------------
;;; A5 — 视图变换后的落地内容（与参考置换逐位一致）
;;; ------------------------------------------------------------------
(defun a5-view-content ()
  (format t "~%[A5] 视图 → 连续化 的内容 ≡ 参考置换~%")
  (let ((a (clvt::make-vt '(3 4) 0.0d0)))
    (dotimes (i 3) (dotimes (j 4) (setf (vt-ref a i j) (coerce (+ (* i 10) j) 'double-float))))
    (check "A5 转置内容 = 原列主序"
           (same-bits-list (ref-view-content (vt-transpose a))
                           (dfs (loop for j below 4 append
                                      (loop for i below 3 collect (+ (* i 10) j))))))
    (check "A5 翻转 axis0 内容"
           (same-bits-list (ref-view-content (vt-flip a :axis 0))
                           (dfs (loop for i from 2 downto 0 append
                                      (loop for j below 4 collect (+ (* i 10) j))))))
    (check "A5 翻转 axis1 内容"
           (same-bits-list (ref-view-content (vt-flip a :axis 1))
                           (dfs (loop for i below 3 append
                                      (loop for j from 3 downto 0 collect (+ (* i 10) j))))))
    (check "A5 步进 a[::2,::2] 内容"
           (same-bits-list (ref-view-content (vt-slice a '(nil nil 2) '(nil nil 2)))
                           (dfs (list 0 2 20 22))))
    (check "A5 重塑 1D 内容"
           (same-bits-list (ref-view-content (vt-reshape a '(12)))
                           (dfs (loop for i below 3 append
                                      (loop for j below 4 collect (+ (* i 10) j))))))
    ;; 零拷贝语义：视图与原张量共享 data
    (check "A5 转置零拷贝（共享 data）" (eq (vt-data (vt-transpose a)) (vt-data a)))
    (check "A5 翻转零拷贝（共享 data）" (eq (vt-data (vt-flip a :axis 1)) (vt-data a)))
    (check "A5 切片零拷贝（共享 data）" (eq (vt-data (vt-slice a '(1 2))) (vt-data a)))
    ;; 空轴翻转 / 切片不移动 offset（§17.10.23）
    (let ((e (clvt::make-vt '(0 3) 0.0d0)))
      (check "A5 空轴翻转 offset 不变"
             (= (vt-offset (vt-flip e :axis 0)) (vt-offset e)))
      (check "A5 空轴切片 offset 不变"
             (= (vt-offset (vt-slice e '(:all) '(1 2))) (vt-offset e))))))

;;; ------------------------------------------------------------------
;;; A6 — 只读判定（vt-out-writable-p）与参考实现一致
;;; ------------------------------------------------------------------
(defun a6-out-writable ()
  (format t "~%[A6] vt-out-writable-p ≡ 参考实现~%")
  (let* ((a (clvt::make-vt '(4 6) 0.0d0))
         (cases (list (cons "连续" a)
                      (cons "转置（可写）" (vt-transpose a))
                      (cons "行切片（可写）" (vt-slice a '(1 3) '(:all)))
                      (cons "广播 (1,6)→(4,6)（只读）"
                            (vt-broadcast-to (vt-slice a '(0 1) '(:all)) '(4 6)))
                      (cons "广播 (4,1)→(4,6)（只读）"
                            (vt-broadcast-to (vt-slice a '(:all) '(0 1)) '(4 6)))
                      (cons "空 (0,6)（可写）" (clvt::make-vt '(0 6) 0.0d0))
                      (cons "空 (0,6) 广播（可写）"
                            (vt-broadcast-to (clvt::make-vt '(1 6) 0.0d0) '(0 6))))))
    (dolist (pair cases)
      (let* ((name (car pair))
             (v (cdr pair))
             (want (if (zerop (ref-size (vt-shape v)))
                       t
                       (not (some (lambda (d s) (and (> d 1) (zerop s)))
                                  (vt-shape v) (vt-strides v)))))
             (got (clvt::vt-out-writable-p v)))
        (if (eq got want)
            (ok (format nil "A6 ~a" name))
            (ng (format nil "A6 ~a" name) "得到 ~a，参考 ~a" got want))))))

;;; ------------------------------------------------------------------
;;; A7 — 拼接：非连续输入路径 ≡ 先连续化再拼接（铁律 1 的可执行版本）
;;; ------------------------------------------------------------------
(defun a7-concatenate ()
  (format t "~%[A7] vt-concatenate 的非连续路径 ≡ 连续路径~%")
  (let* ((a (clvt::make-vt '(4 6) 0.0d0))
         (b (clvt::make-vt '(2 6) 0.0d0)))
    (dotimes (i 4) (dotimes (j 6) (setf (vt-ref a i j) (coerce (+ (* i 10) j) 'double-float))))
    (dotimes (i 2) (dotimes (j 6) (setf (vt-ref b i j) (coerce (+ 100 (* i 10) j) 'double-float))))
    ;; 连续输入（块拷贝快路径）作为基准
    (check "A7 连续 axis0 形状" (equal (vt-shape (vt-concatenate 0 a b)) '(6 6)))
    (check "A7 连续 axis0 内容"
           (same-bits-list (ref-view-content (vt-concatenate 0 a b))
                           (append (ref-view-content a) (ref-view-content b))))
    ;; 步进行视图 vs 先 vt-copy 再拼接
    (let* ((sa (vt-slice a '(nil nil 2) '(:all)))
           (sb (vt-slice b '(nil nil 2) '(:all)))
           (via-view (vt-concatenate 0 sa sb))
           (via-copy (vt-concatenate 0 (vt-copy sa) (vt-copy sb))))
      (check "A7 步进行视图拼接 ≡ 先连续化再拼接"
             (same-bits-list (ref-view-content via-view) (ref-view-content via-copy)))
      (check "A7 步进行视图拼接 形状" (equal (vt-shape via-view) '(3 6))))
    ;; 转置视图沿 axis 1
    (let* ((ta (vt-transpose a)) (tb (vt-transpose b))
           (via-view (vt-concatenate 1 ta tb))
           (via-copy (vt-concatenate 1 (vt-copy ta) (vt-copy tb))))
      (check "A7 转置视图拼接 ≡ 先连续化再拼接"
             (same-bits-list (ref-view-content via-view) (ref-view-content via-copy)))
      (check "A7 转置视图拼接 形状" (equal (vt-shape via-view) '(6 6))))
    ;; 单行视图（run 长度 = 1 的极端：走逐元素分支）
    (let* ((r0 (vt-slice a '(0) '(:all)))
           (r2 (vt-slice a '(2) '(:all)))
           (got (vt-concatenate 0 r0 r2)))
      (check "A7 单行视图拼接内容"
             (same-bits-list (ref-view-content got)
                             (append (ref-view-content r0) (ref-view-content r2)))))
    ;; 栈/堆叠
    (check "A7 vstack 两行 = concatenate axis0"
           (same-bits-list (ref-view-content (vt-vstack (vt-slice a '(0) '(:all))
                                                        (vt-slice a '(1) '(:all))))
                           (ref-view-content (vt-concatenate 0 (vt-slice a '(0) '(:all))
                                                             (vt-slice a '(1) '(:all))))))))

;;; ------------------------------------------------------------------
;;; B1 — 单元素访问：逻辑层开销必须远小于「再做一遍类型校验」
;;;      vt-ref / %vt-ref-raw 的比值退化说明又回到了「先整表扫一遍
;;;      find-if-not #'integerp」或「四次 array-element-type 比较」的老路。
;;; ------------------------------------------------------------------
(defun b1-scalar-access ()
  (format t "~%[B1] 单元素访问的路径开销~%")
  (let* ((v (clvt::make-vt '(1000) 0.0d0))
         (raw (bench-us (lambda () (dotimes (i 1000) (clvt::%vt-ref-raw v i)))))
         (pub (bench-us (lambda () (dotimes (i 1000) (vt-ref v i)))))
         (wr  (bench-us (lambda () (dotimes (i 1000) (setf (vt-ref v i) 1.0d0))))))
    ;; 优化前：pub/raw = 2.13×（多一趟 find-if-not），wr/raw = 4.02×
    (check-ratio "B1 vt-ref ÷ %vt-ref-raw" (/ pub (max raw 1e-9)) 1.60
                 "t(vt-ref) ÷ t(%vt-ref-raw)")
    (check-ratio "B1 (setf vt-ref) ÷ %vt-ref-raw" (/ wr (max raw 1e-9)) 2.60
                 "t(setf vt-ref) ÷ t(%vt-ref-raw)")
    ;; 2D 与 1D 的单元素代价应当是同一量级（索引数不改复杂度）
    (let* ((a (clvt::make-vt '(1000 1000) 0.0d0))
           (t2 (bench-us (lambda () (dotimes (i 1000) (vt-ref a (mod i 1000) (mod (* i 7) 1000)))))))
      (check-ratio "B1 vt-ref 2D ÷ 1D" (/ t2 (max pub 1e-9)) 2.20
                   "t(2D) ÷ t(1D)"))))

;;; ------------------------------------------------------------------
;;; B2 — 视图构造：转置 / 重塑 / 翻转 / 切片 的相对代价
;;;      四者都是「分配一个视图对象 + 算 strides + 判一次连续」，量级应当
;;;      接近。这里只拦「量级退化」（某个操作慢 2 倍以上），阈值按实测
;;;      1.0~1.4 留一倍余量 —— 本机噪声大，比值护栏宁可松一点也不误红。
;;; ------------------------------------------------------------------
(defun b2-view-construction ()
  (format t "~%[B2] 视图构造的相对代价~%")
  (let* ((a (clvt::make-vt '(1000 1000) 0.0d0))
         (t-tr (bench-us (lambda () (dotimes (i 1000) (vt-transpose a)))))
         (t-rs (bench-us (lambda () (dotimes (i 1000) (vt-reshape a '(1000000))))))
         (t-fl (bench-us (lambda () (dotimes (i 1000) (vt-flip a :axis 0)))))
         (t-sl (bench-us (lambda () (dotimes (i 1000) (vt-slice a '(0) '(:all))))))
         (t-cp (bench-us (lambda () (dotimes (i 1000) (vt-contiguous-p a))))))
    (check-ratio "B2 vt-reshape ÷ vt-transpose" (/ t-rs (max t-tr 1e-9)) 2.00
                 "t(reshape) ÷ t(transpose)")
    (check-ratio "B2 vt-flip ÷ vt-transpose" (/ t-fl (max t-tr 1e-9)) 2.00
                 "t(flip) ÷ t(transpose)")
    (check-ratio "B2 vt-slice ÷ vt-transpose" (/ t-sl (max t-tr 1e-9)) 3.00
                 "t(slice 行) ÷ t(transpose)")
    ;; 连续判定必须**显著**便宜于任何一次视图构造（它只读两个链表、
    ;; 不分配）。优化前 0.55×（O(rank²) 的 NTH 循环），优化后 0.15~0.23×。
    (check-ratio "B2 vt-contiguous-p ÷ vt-transpose" (/ t-cp (max t-tr 1e-9)) 0.50
                 "t(contiguous-p) ÷ t(transpose)")))

;;; ------------------------------------------------------------------
;;; B3 — 高秩的连续判定仍然便宜
;;;      优化前 rank 6 的 `vt-contiguous-p` 是 O(rank²) 的两次 NTH/轴，
;;;      代价与「秩 6 的转置（要构造 perm + 校验 + 建表）」相当（0.61×）；
;;;      优化后每轴只走一次 CDR，降到 0.22×。用「同秩转置」当标尺，
;;;      机器快慢在比值里抵消。
;;; ------------------------------------------------------------------
(defun b3-contiguous-high-rank ()
  (format t "~%[B3] 高秩连续判定的代价~%")
  (let* ((v6 (clvt::make-vt '(2 2 2 2 2 8) 0.0d0))
         (t-cp (bench-us (lambda () (dotimes (i 1000) (vt-contiguous-p v6)))))
         (t-tr (bench-us (lambda () (dotimes (i 1000) (vt-transpose v6))))))
    (check-ratio "B3 vt-contiguous-p(rank 6) ÷ vt-transpose(rank 6)"
                 (/ t-cp (max t-tr 1e-9)) 0.50
                 "t(contiguous-p r6) ÷ t(transpose r6)")))

;;; ------------------------------------------------------------------
;;; B4 — 拼接：非连续输入的每元素代价不得比连续输入高一个数量级
;;; ------------------------------------------------------------------
(defun b4-concat-forms ()
  (format t "~%[B4] 拼接的物理形态代价~%")
  (let* ((a (clvt::make-vt '(64 128) 0.0d0))
         (rows (loop repeat 64 collect (vt-slice a '(0) '(:all))))
         (t-rows (bench-us (lambda () (apply #'vt-concatenate 0 rows))))
         (t-contig (bench-us (lambda () (vt-concatenate 0 a a))))
         ;; 元素数：64×128 vs 128×128 —— 每元素代价之比才是可移植的量
         (per-rows (/ t-rows (* 64.0 128)))
         (per-contig (/ t-contig (* 128.0 128))))
    ;; 64 次调用、每次 128 个元素：调用开销主导，但不得高到 10×/元素
    (check-ratio "B4 拼接 64 行视图 ÷ 拼接连续块（每元素）"
                 (/ per-rows (max per-contig 1e-9)) 10.0
                 "ns/元素(行视图) ÷ ns/元素(连续)")))

;;; ------------------------------------------------------------------
;;; 主入口
;;; ------------------------------------------------------------------
(defun run ()
  (format t "~%===== 访问 / 切片 / 视图 / 连续性 护栏 =====~%")
  (a1-shape-product)
  (a2-contiguous-p)
  (a3-vt-ref)
  (a4-setf-vt-ref)
  (a5-view-content)
  (a6-out-writable)
  (a7-concatenate)
  (b1-scalar-access)
  (b2-view-construction)
  (b3-contiguous-high-rank)
  (b4-concat-forms)
  (format t "~&===== 汇总: ~a 通过 / ~a 失败 =====~%" *pass* *fail*)
  (format t "Total: ~a | Pass: ~a | Fail: ~a~%" (+ *pass* *fail*) *pass* *fail*)
  (when *failures*
    (format t "失败清单:~%")
    (dolist (f (nreverse *failures*)) (format t "  - ~a~%" f)))
  (finish-output)
  (zerop *fail*))

(let ((okp (run)))
  (sb-ext:exit :code (if okp 0 1)))
