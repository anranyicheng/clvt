;;;; cumulative-axis-test.lisp — 累积的「沿轴」路径与 nanmedian 多轴（§16.35）
;;;;
;;;; 两个主题，都是 §16.34.7 记下的遗留项，已在 §16.35 收口：
;;;;
;;;;  A. `vt-cumsum/cumprod` 的 **axis 路径**一直是通用回退（逐元素 FUNCALL +
;;;;     逐元素 VT-CAST），1000×1000 实测 196 ms 而 numpy 3.3 ms（59×）。
;;;;     修法是把「平面」与「沿轴」统一到同一个按元素类型特化的 RUN 上。
;;;;     这里用**独立的朴素参照**（纯 list + 坐标换算，不碰 strides/aref）
;;;;     逐位对照，防止特化改错；并加一条性能护栏（沿轴 vs 展平的比值）。
;;;;     为什么不用 numpy 硬编码：沿轴累积的期望值有 1M 个数字，硬编码不现实；
;;;;     「快路径 == 朴素定义」才是铁律 1/2 要保的东西（numpy 一致性由
;;;;     scan-first-element-test 与 differential-probes 覆盖）。
;;;;
;;;;  B. `vt-nanmedian` 的**多轴**归约（axis 为列表）此前直接报
;;;;     「暂不支持多轴归约」，而 numpy 支持。修法是用「折叠成 (H, D) +
;;;;     复用单轴内核」实现。这里用 numpy 2.4.6 实测值做硬编码期望
;;;;     （数据是二进制可精确表示的小整数，故要求逐位相等）。
;;;;
;;;; 用法：sbcl --non-interactive --load test/cumulative-axis-test.lisp

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

(defun close-enough (a b)
  (cond ((and (floatp a) (floatp b))
         (cond ((and (sb-ext:float-nan-p a) (sb-ext:float-nan-p b)) t)
               ((or (sb-ext:float-nan-p a) (sb-ext:float-nan-p b)) nil)
               (t (= a b))))
        (t (equal a b))))

;;; ------------------------------------------------------------------
;;; A. 累积沿轴：独立朴素参照
;;; ------------------------------------------------------------------

(defun idx->coords (p dims)
  (let ((c (make-list (length dims) :initial-element 0)))
    (loop for d from (1- (length dims)) downto 0
          do (multiple-value-bind (q r) (floor p (nth d dims))
               (setf (nth d c) r p q)))
    c))

(defun coords->idx (c dims)
  (let ((p 0))
    (loop for d in dims for x in c do (setf p (+ (* p d) x)))
    p))

(defun ref-cum (tensor op axis)
  "朴素定义：行主序 flat 上按 accumulate 语义（首元素直拷）。不碰 strides。"
  (let* ((dims (vt-shape tensor))
         (flat (vt-to-list (vt-ravel tensor)))
         (n (length flat))
         (v (coerce flat 'vector))
         (out (make-array n)))
    (if (null axis)
        (loop with acc = nil
              for i from 0 below n
              do (let ((x (aref v i)))
                   (setf acc (if (zerop i) x (funcall op acc x)))
                   (setf (aref out i) acc)))
        (let ((ax (if (minusp axis) (+ axis (length dims)) axis)))
          (dotimes (p n)
            (let* ((c (idx->coords p dims))
                   (acc nil))
              (dotimes (i (1+ (nth ax c)))
                (let* ((c2 (copy-list c))
                       (x (aref v (coords->idx (progn (setf (nth ax c2) i) c2) dims))))
                  (setf acc (if (zerop i) x (funcall op acc x)))))
              (setf (aref out p) acc)))))
    (coerce out 'list)))

(defun mk (dims fn &key (dtype :float64))
  (let ((total (reduce #'* dims :initial-value 1)))
    (vt-reshape (vt-from-sequence
                 (loop for p below total
                       collect (funcall fn (idx->coords p dims)))
                 :dtype dtype)
                dims)))

(defun check-cum (tag tensor result op axis)
  (let* ((dims (vt-shape tensor))
         (want (ref-cum tensor op axis))
         (got (vt-to-list (vt-ravel result)))
         (want-shape (if axis dims (list (vt-size tensor)))))
    (report tag
            (and (equal want-shape (vt-shape result))
                 (= (length got) (length want))
                 (every #'close-enough got want))
            (unless (and (= (length got) (length want))
                         (every #'close-enough got want))
              (format nil "got ~a / want ~a" got want)))))

(defun run-accumulate-axis ()
  (format t "~&--- A. 累积沿轴（朴素参照对照）---~%")
  (let ((base (mk '(4 5) (lambda (c) (* 1.0d0 (+ (* 10 (first c)) (second c)))))))
    (dolist (ax (list nil 0 1 -1 -2))
      (check-cum (format nil "2D axis=~a cumsum" ax) base (vt-cumsum base :axis ax) #'+ ax)
      (check-cum (format nil "2D axis=~a cumprod" ax) base (vt-cumprod base :axis ax) #'* ax))

    ;; 非连续：步长切片 / 转置 / 负步长（三条都走 strides 分支）
    (let ((s (vt-slice base '(:all) '(0 5 2))))
      (dolist (ax (list 0 1))
        (check-cum (format nil "步长切片 axis=~a" ax) s (vt-cumsum s :axis ax) #'+ ax)))
    (let ((tr (vt-transpose base)))
      (dolist (ax (list nil 0 1))
        (check-cum (format nil "转置 axis=~a" ax) tr (vt-cumsum tr :axis ax) #'+ ax)))
    (let ((r (vt-slice base '(3 nil -1) '(4 nil -1))))
      (dolist (ax (list 0 1))
        (check-cum (format nil "负步长 axis=~a" ax) r (vt-cumsum r :axis ax) #'+ ax)))

    ;; dtype 变体：float32/int32/int64 走特化；int8→int64 走通用回退
    (dolist (dt (list :float32 :int32 :int64))
      (let ((a (vt-astype base dt)))
        (dolist (ax (list 0 1))
          (check-cum (format nil "~a axis=~a" dt ax) a (vt-cumsum a :axis ax) #'+ ax))))
    (let ((a (vt-astype (mk '(3 4) (lambda (c) (mod (* 7 (second c)) 60)) :dtype :int8) :int8)))
      (dolist (ax (list 0 1))
        (check-cum (format nil "int8(->int64) axis=~a" ax) a (vt-cumsum a :axis ax) #'+ ax)))

    ;; 3D 四向
    (let ((c (mk '(3 4 5) (lambda (k) (* 1.0d0 (+ (* 100 (first k)) (* 10 (second k)) (third k)))))))
      (dolist (ax (list 0 1 2 -1))
        (check-cum (format nil "3D axis=~a cumsum" ax) c (vt-cumsum c :axis ax) #'+ ax)
        (check-cum (format nil "3D axis=~a cumprod" ax) c (vt-cumprod c :axis ax) #'* ax)))

    ;; 空轴 / 轴长 1 / 0 维
    (check-cum "空 (0 3) axis=0" (vt-zeros '(0 3)) (vt-cumsum (vt-zeros '(0 3)) :axis 0) #'+ 0)
    (check-cum "空 (3 0) axis=1" (vt-zeros '(3 0)) (vt-cumsum (vt-zeros '(3 0)) :axis 1) #'+ 1)
    (let ((o (mk '(5 1) (lambda (c) (* 1.0d0 (first c))))))
      (check-cum "轴长 1" o (vt-cumsum o :axis 1) #'+ 1))

    ;; :out 非连续（列视图），同时核对未触及位置没被写
    (let* ((res (vt-zeros '(4 5)))
           (view (vt-slice res '(:all) '(0 5 2)))
           (sub (vt-slice base '(:all) '(0 5 2))))
      (vt-cumsum sub :axis 0 :out view)
      (check-cum ":out 列视图 axis=0" sub view #'+ 0)
      (report ":out 未触及位置仍为 0"
              (every #'zerop (list (vt-ref res 0 1) (vt-ref res 0 3)
                                   (vt-ref res 3 1) (vt-ref res 3 3)))))
    ;; :out 与输入完全别名（快照语义）
    (let ((a (mk '(4 1) (lambda (c) (* 1.0d0 (first c))))))
      (vt-cumsum a :axis 0 :out a)
      (report ":out 完全别名 axis=0（快照）"
              (close-enough (vt-to-list (vt-ravel a)) '(0d0 1d0 3d0 6d0))
              (format nil "~a" (vt-to-list (vt-ravel a)))))))

;;; ------------------------------------------------------------------
;;; A2. 性能护栏：沿轴不得再掉回通用路径
;;;
;;; 通用路径 1000×1000 实测 196ms，特化后 4ms，展平路径 1ms。取
;;; 「沿轴 ≤ 40× 展平」：留足 CI 抖动余量，同时 196× 一定被抓到。
;;; ------------------------------------------------------------------

(defun best-ms (fn &optional (k 3))
  (let ((best most-positive-fixnum))
    (dotimes (i k)
      (let ((t0 (get-internal-real-time)))
        (funcall fn)
        (setf best (min best (- (get-internal-real-time) t0)))))
    (* 1000.0 (/ best internal-time-units-per-second))))

(defun run-perf-guard ()
  (format t "~&--- A2. 性能护栏 ---~%")
  (let* ((m (mk '(1000 1000) (lambda (c) (* 1.0d0 (mod (+ (first c) (second c)) 5)))))
         (flat (best-ms (lambda () (vt-cumsum m))))
         (ax0 (best-ms (lambda () (vt-cumsum m :axis 0))))
         (ax1 (best-ms (lambda () (vt-cumsum m :axis 1))))
         (ratio (max (/ ax0 flat) (/ ax1 flat))))
    (format t "~&    展平 ~,2f ms / axis0 ~,2f ms / axis1 ~,2f ms（比值 ~,1f×）~%"
            flat ax0 ax1 ratio)
    (report "沿轴累积 ≤ 40× 展平累积（性能护栏）"
            (< ratio 40.0)
            (format nil "比值 ~,1f×（展平 ~,2f ms, axis0 ~,2f ms, axis1 ~,2f ms）"
                    ratio flat ax0 ax1))))

;;; ------------------------------------------------------------------
;;; B. nanmedian 多轴（期望值 = numpy 2.4.6 实测，逐位相等）
;;; ------------------------------------------------------------------

(defun f64 (x) (coerce x 'double-float))

(defun run-nanmedian-multi ()
  (format t "~&--- B. nanmedian 多轴 ---~%")
  (let* ((i (vt-from-sequence '(0 1) :dtype :int64))
         (j (vt-from-sequence '(0 1 2) :dtype :int64))
         (k (vt-from-sequence '(0 1 2 3) :dtype :int64))
         ;; a[i,j,k] = 100i + 10j + k，shape (2 3 4)
         (a (mk '(2 3 4) (lambda (c) (f64 (+ (* 100 (first c)) (* 10 (second c)) (third c)))))))
    (declare (ignore i j k))

    ;; numpy: nanmedian(a, axis=(0,1)) = [60 61 62 63]
    (let ((r (vt-nanmedian a :axis '(0 1))))
      (report "axis=(0,1) 值" (equal (mapcar #'f64 (vt-to-list (vt-ravel r)))
                                     (mapcar #'f64 '(60 61 62 63))))
      (report "axis=(0,1) 形状" (equal (vt-shape r) '(4))))
    ;; numpy: axis=(1,2) = [11.5 111.5]
    (let ((r (vt-nanmedian a :axis '(1 2))))
      (report "axis=(1,2) 值" (equal (mapcar #'f64 (vt-to-list (vt-ravel r)))
                                     (mapcar #'f64 '(11.5 111.5))))
      (report "axis=(1,2) 形状" (equal (vt-shape r) '(2))))
    ;; numpy: axis=(0,2) = [51.5 61.5 71.5]
    (let ((r (vt-nanmedian a :axis '(0 2))))
      (report "axis=(0,2) 值" (equal (mapcar #'f64 (vt-to-list (vt-ravel r)))
                                     (mapcar #'f64 '(51.5 61.5 71.5))))
      (report "axis=(0,2) 形状" (equal (vt-shape r) '(3))))
    ;; numpy: axis=(0,1,2) = 61.5（0 维）
    (let ((r (vt-nanmedian a :axis '(0 1 2))))
      (report "axis=(0,1,2) 值" (close-enough (vt-ref r) 61.5d0))
      (report "axis=(0,1,2) 形状为 0 维" (null (vt-shape r))))
    ;; keepdims
    (let ((r (vt-nanmedian a :axis '(0 1) :keepdims t)))
      (report "keepdims 形状 (1 1 4)" (equal (vt-shape r) '(1 1 4)))
      (report "keepdims 值" (equal (mapcar #'f64 (vt-to-list (vt-ravel r)))
                                   (mapcar #'f64 '(60 61 62 63)))))
    (let ((r (vt-nanmedian a :axis '(1 2) :keepdims t)))
      (report "keepdims(1,2) 形状 (2 1 1)" (equal (vt-shape r) '(2 1 1)))
      (report "keepdims(1,2) 值" (equal (mapcar #'f64 (vt-to-list (vt-ravel r)))
                                        (mapcar #'f64 '(11.5 111.5)))))

    ;; 负轴 / 轴顺序无关
    (report "负轴 (-2 -1) = axis=(1,2)"
            (equal (mapcar #'f64 (vt-to-list (vt-ravel (vt-nanmedian a :axis '(-2 -1)))))
                   (mapcar #'f64 '(11.5 111.5))))
    (report "轴顺序无关 (2 1) = (1 2)"
            (equal (mapcar #'f64 (vt-to-list (vt-ravel (vt-nanmedian a :axis '(2 1)))))
                   (mapcar #'f64 '(11.5 111.5))))

    ;; NaN 参与：b[0,0,0] = NaN
    ;; numpy: nanmedian(b, axis=(0,1)) = [100 61 62 63]
    (let ((b (vt-astype a :float64)))
      (setf (vt-ref b 0 0 0) +vt-dfloat-nan+)
      (report "含 NaN axis=(0,1)"
              (equal (mapcar #'f64 (vt-to-list (vt-ravel (vt-nanmedian b :axis '(0 1)))))
                     (mapcar #'f64 '(100 61 62 63))))
      ;; numpy: nanmedian(b, axis=(1,2), keepdims) = [[[12]], [[111.5]]]
      (let ((r (vt-nanmedian b :axis '(1 2) :keepdims t)))
        (report "含 NaN axis=(1,2) keepdims"
                (and (equal (vt-shape r) '(2 1 1))
                     (equal (mapcar #'f64 (vt-to-list (vt-ravel r)))
                            (mapcar #'f64 '(12 111.5)))))))

    ;; 复数（字典序）：z = a + i(1i + 2j + 3k)
    ;; numpy: nanmedian(z, axis=(0,2)) = [51.5+5j 61.5+7j 71.5+9j]
    (let ((z (vt-from-sequence
              (loop for p below 24
                    for c = (idx->coords p '(2 3 4))
                    collect (complex (f64 (+ (* 100 (first c)) (* 10 (second c)) (third c)))
                                     (f64 (+ (first c) (* 2 (second c)) (* 3 (third c))))))
              :dtype :complex128)))
      (setf z (vt-reshape z '(2 3 4)))
      (let ((r (vt-nanmedian z :axis '(0 2))))
        (report "复数 axis=(0,2) 值"
                (equal (vt-to-list (vt-ravel r))
                       (list (complex 51.5d0 5d0) (complex 61.5d0 7d0) (complex 71.5d0 9d0))))
        (report "复数 axis=(0,2) dtype" (eq (vt-dtype r) :complex128)))
      ;; numpy: nanmedian(z, axis=(0,1), keepdims) = [[[60+2.5j 61+5.5j 62+8.5j 63+11.5j]]]
      (let ((r (vt-nanmedian z :axis '(0 1) :keepdims t)))
        (report "复数 axis=(0,1) keepdims 形状" (equal (vt-shape r) '(1 1 4)))
        (report "复数 axis=(0,1) keepdims 值"
                (equal (vt-to-list (vt-ravel r))
                       (list (complex 60d0 2.5d0) (complex 61d0 5.5d0)
                             (complex 62d0 8.5d0) (complex 63d0 11.5d0))))))

    ;; :out（连续）必须返回同一对象；形状不符必须报错
    (let* ((o (vt-zeros '(4) :dtype :float64))
           (r (vt-nanmedian a :axis '(0 1) :out o)))
      (report ":out 连续返回同一对象" (eq r o))
      (report ":out 连续值" (equal (mapcar #'f64 (vt-to-list (vt-ravel r)))
                                   (mapcar #'f64 '(60 61 62 63)))))
    (report ":out 形状不符报错"
            (handler-case
                (progn (vt-nanmedian a :axis '(0 1) :out (vt-zeros '(3) :dtype :float64)) nil)
              (error () t)))
    ;; 非连续 :out
    (let* ((base (vt-zeros '(1 1 8) :dtype :float64))
           (view (vt-slice base '(:all) '(:all) '(0 8 2)))
           (r (vt-nanmedian a :axis '(0 1) :keepdims t :out view)))
      (report ":out 非连续形状" (equal (vt-shape r) '(1 1 4)))
      (report ":out 非连续值" (equal (mapcar #'f64 (vt-to-list (vt-ravel r)))
                                     (mapcar #'f64 '(60 61 62 63))))
      (report ":out 非连续未越界"
              (every #'zerop (loop for j from 1 below 8 by 2 collect (vt-ref base 0 0 j)))))

    ;; 单轴 / 全归约不回归（多轴实现不得影响既有路径）
    (report "单轴 axis=1 形状" (equal (vt-shape (vt-nanmedian a :axis 1)) '(2 4)))
    (report "全归约 0 维 + 61.5" (and (null (vt-shape (vt-nanmedian a)))
                                      (close-enough (vt-ref (vt-nanmedian a)) 61.5d0)))
    ;; 重复轴必须报错（与 vt-normalize-axes 一致）
    (report "重复轴 (0 0) 报错"
            (handler-case (progn (vt-nanmedian a :axis '(0 0)) nil) (error () t)))
    (report "轴越界 (0 9) 报错"
            (handler-case (progn (vt-nanmedian a :axis '(0 9)) nil) (error () t)))))

;;; ------------------------------------------------------------------
;;; C. 契约：:out 重叠保护 + nanmedian 的 dtype 降级拦截
;;;
;;; 两组都是 §16.36 补的洞：
;;;   C1 累积是**流式**读写（读到第 i 个就写第 i 个），当 out 与输入在物理
;;;      区间上部分重叠（out = base[1:7]、in = base[0:6]）时，先写的结果会
;;;      污染还没读到的输入 —— 修前 `(vt-cumsum in :out out)` 给出
;;;      1,2,4,8,16,32 这种「越滚越大」的错值。§4.3 要求先快照。
;;;   C2 `(vt-nanmedian z :dtype :float64)` 曾**取实部**返回 float64，
;;;      虚部被静默丢弃（§16.13 列为最危险的失败模式）。
;;; ------------------------------------------------------------------

(defun run-contract ()
  (format t "~&--- C. 契约（重叠保护 / dtype 降级）---~%")

  ;; C1: 偏移 1 的重叠。base 初始 (1 2 3 4 5 0 0 0)，
  ;; a = base[0:6] = (1 2 3 4 5 0)，out = base[1:7]。
  ;; 正确（先快照）：cumsum(a) = (1 3 6 10 15 15) → base = (1 1 3 6 10 15 15 0)
  (flet ((shifted (fn)
           (let ((base (vt-zeros '(8) :dtype :float64)))
             (dotimes (i 5) (setf (vt-ref base i) (coerce (1+ i) 'double-float)))
             (funcall fn (vt-slice base '(0 6)) :out (vt-slice base '(1 7)))
             (mapcar (lambda (x) (coerce x 'double-float))
                     (vt-to-list (vt-ravel base))))))
    (report "cumsum :out 偏移重叠（先快照）"
            (equal (shifted #'vt-cumsum)
                   (list 1d0 1d0 3d0 6d0 10d0 15d0 15d0 0d0))
            (format nil "~a" (shifted #'vt-cumsum)))
    (report "cumprod :out 偏移重叠（先快照）"
            ;; cumprod((1 2 3 4 5 0)) = (1 2 6 24 120 0) → base[1:7] 落位
            (equal (shifted #'vt-cumprod)
                   (list 1d0 1d0 2d0 6d0 24d0 120d0 0d0 0d0))
            (format nil "~a" (shifted #'vt-cumprod)))
    ;; 完全别名（cumsum 天然安全，但必须保证快照逻辑不把它改坏）
    (let ((a (vt-from-sequence '(1 2 3 4 5) :dtype :float64)))
      (vt-cumsum a :out a)
      (report "cumsum :out 完全别名"
              (equal (mapcar (lambda (x) (coerce x 'double-float))
                             (vt-to-list (vt-ravel a)))
                     (list 1d0 3d0 6d0 10d0 15d0)))))

  ;; C2: dtype 降级必须报错（而不是静默取实部/静默忽略）
  (let ((z (vt-from-sequence (list (list #C(1 2) #C(3 -4) #C(5 0)))
                             :dtype :complex128))
        (r (vt-from-sequence '((1.0d0 2.0d0 3.0d0)) :dtype :float64)))
    (report "复数 nanmedian + :dtype :float64 报错"
            (handler-case
                (progn (sb-int:with-float-traps-masked (:invalid :divide-by-zero :overflow)
                         (vt-nanmedian z :axis 1 :dtype :float64))
                       nil)
              (error () t)))
    (report "实数 nanmedian + :dtype :float32 报错"
            (handler-case (progn (vt-nanmedian r :axis 1 :dtype :float32) nil)
              (error () t)))
    (report "复数 nanmedian + :dtype :complex64 正常"
            (sb-int:with-float-traps-masked (:invalid :divide-by-zero :overflow)
              (eq (vt-dtype (vt-nanmedian z :axis 1 :dtype :complex64)) :complex64)))
    (report "实数 nanmedian 缺省仍为 :float64"
            (eq (vt-dtype (vt-nanmedian r :axis 1)) :float64)))

  ;; C3: 退化（归约区长度 0）多轴 → NaN，与 numpy 一致
  (let ((d (vt-nanmedian (vt-zeros '(2 0 3)) :axis '(1 2))))
    (report "多轴归约区为空 → (2) 全 NaN"
            (and (equal (vt-shape d) '(2))
                 (every (lambda (x) (sb-ext:float-nan-p x))
                        (mapcar (lambda (x) (coerce x 'double-float))
                                (vt-to-list (vt-ravel d))))))))

;;; ------------------------------------------------------------------
;;; 入口
;;; ------------------------------------------------------------------

(format t "~&=====================================")
(format t "~&累积沿轴 + nanmedian 多轴（§16.35）")
(format t "~&=====================================~%")
(run-accumulate-axis)
(run-perf-guard)
(run-nanmedian-multi)
(run-contract)

(format t "~&=====================================~%")
(format t "Total: ~d / Pass: ~d / Fail: ~d~%" (+ *pass* *fail*) *pass* *fail*)
(format t "PASS=~d FAIL=~d~%" *pass* *fail*)
(when *failures*
  (format t "~&--- 失败清单 ---~%")
  (dolist (f (reverse *failures*)) (format t "  · ~a~%" f)))
(when (plusp *fail*) (sb-ext:exit :code 1))
