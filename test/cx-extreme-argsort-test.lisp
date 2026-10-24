;;;; cx-extreme-argsort-test.lisp — 复数极值族（maximum/minimum/fmax/fmin/clip）
;;;;                                与 argsort/topk 的正确性 + 倍率护栏
;;;;
;;;; 背景（CONVENTIONS §16.8 / §16.14）：
;;;;   1. 极值内核原先用边界版 %CISNAN 判 NaN —— 它内部套一层
;;;;      with-float-traps-masked（单次约 200ns），每元素判两次，
;;;;      1M 元素光屏蔽/恢复陷阱就吃掉 800ms。内核里必须用 %CX-NAN-P。
;;;;   2. argsort 原先 `stable-sort` + lambda 比较器（每比较一次 funcall），
;;;;      改成下标归并内核后零 funcall、零 cons。
;;;;   3. topk 原先同时 vt-sort 和 vt-argsort（两趟 O(n log n)），
;;;;      改成「只算一次序 + 按下标 gather」。
;;;;
;;;; 本套件：
;;;;   A. 极值族语义：与「逐对按字典序规则折叠」的独立参照逐点一致
;;;;      （覆盖 NaN 任一分量 / Inf / ±0.0 / 相等实部 / 广播 / 非连续输入）；
;;;;   B. clip 复数 = min(max(x, lo), hi)；
;;;;   C. argsort：双射、非降序、与 vt-sort 一致、稳定性、多轴；
;;;;   D. topk：与「排序后切片」一致、索引回指、边界 k；
;;;;   E. 护栏：复数/实数倍率（不看绝对毫秒，避免与机器速度耦合）。
;;;;
;;;; 用法：sbcl --dynamic-space-size 4096 --non-interactive --load test/cx-extreme-argsort-test.lisp

(require :asdf)
(defparameter cl-user::*clvt-test-dir*
  (make-pathname :name nil :type nil :defaults *load-truename*))
(pushnew (merge-pathnames "../" cl-user::*clvt-test-dir*) asdf:*central-registry* :test #'equal)
(handler-bind ((warning #'muffle-warning)) (asdf:load-system :clvt))

(in-package :clvt)

;;; ------------------------------------------------------------------
;;; 骨架
;;; ------------------------------------------------------------------

(defvar *pass* 0)
(defvar *fail* 0)
(defvar *failures* '())

(defun expect-true (name form)
  (if form
      (incf *pass*)
      (progn (incf *fail*)
             (push name *failures*)
             (format t "~&FAIL ~a~%" name))))

(defun expect-equal (name got want)
  (if (equal got want)
      (incf *pass*)
      (progn (incf *fail*)
             (push (format nil "~a: 期望 ~s 实得 ~s" name want got) *failures*)
             (format t "~&FAIL ~a: 期望 ~s 实得 ~s~%" name want got))))

(defun report (name ok)
  (if ok
      (incf *pass*)
      (progn (incf *fail*)
             (push name *failures*)
             (format t "~&FAIL ~a~%" name))))

;;; ------------------------------------------------------------------
;;; 字典序参照实现（**独立于库内内核**：只用 realpart/imagpart 与
;;; SB-EXT:FLOAT-NAN-P，不碰 %clt / %cx-nan-p / %cop-*）。
;;; 注意：测试进程不屏蔽浮点陷阱，NaN 一旦进 `=` / `<` 就会抛
;;; FLOATING-POINT-INVALID-OPERATION —— 所以所有比较前先判 NaN。
;;; ------------------------------------------------------------------

(defun cxn (z)
  "任一分量为 NaN → 真（CONVENTIONS §16.9）。"
  (or (sb-ext:float-nan-p (realpart z))
      (sb-ext:float-nan-p (imagpart z))))

(defun cx-lt (a b)
  "字典序小于；任一为 NaN → NIL（§16.8）。"
  (and (not (or (cxn a) (cxn b)))
       (or (< (realpart a) (realpart b))
           (and (= (realpart a) (realpart b))
                (< (imagpart a) (imagpart b))))))

(defun cx-le (a b)
  "排序用的非降序判定：NaN 一律视作最大（排在末尾）。"
  (cond ((cxn a) t)
        ((cxn b) nil)
        (t (or (< (realpart a) (realpart b))
               (and (= (realpart a) (realpart b))
                    (<= (imagpart a) (imagpart b)))))))

(defun ref-max (a b) (cond ((cxn a) a) ((cxn b) b) ((cx-lt a b) b) (t a)))
(defun ref-min (a b) (cond ((cxn a) a) ((cxn b) b) ((cx-lt a b) a) (t b)))
(defun ref-fmax (a b) (cond ((cxn a) b) ((cxn b) a) ((cx-lt a b) b) (t a)))
(defun ref-fmin (a b) (cond ((cxn a) b) ((cxn b) a) ((cx-lt a b) a) (t b)))

(defun feq (x y)
  "分量比较：NaN 只与 NaN 相等（不触发陷阱 —— 先判 NaN 再走 `=`）。"
  (cond ((and (floatp x) (sb-ext:float-nan-p x))
         (and (floatp y) (sb-ext:float-nan-p y)))
        ((and (floatp y) (sb-ext:float-nan-p y)) nil)
        (t (= x y))))

(defun cx= (a b)
  "复数逐分量相等（NaN 视为相等的同一值，用于「同一内核不同路径」对照）。"
  (and (feq (realpart a) (realpart b))
       (feq (imagpart a) (imagpart b))))

(defun cx-lists= (la lb)
  (and (= (length la) (length lb))
       (every #'cx= la lb)))

;;; ------------------------------------------------------------------
;;; 构造
;;; ------------------------------------------------------------------

(defvar *zero-d* 0d0
  "经 defvar 的零：编译期不参与常量折叠，否则 (/ 0d0 0d0) 会在折叠期就抛
   FLOATING-POINT-INVALID-OPERATION（与陷阱屏蔽无关），并多出一条 STYLE-WARNING。")
(defvar *zero-s* 0s0)

(defun mk-nan-d ()
  "NaN 只能「算」出来：0/0 在 with-float-safe 下返回 NaN。
   必须经由**变量**相除 —— 直接写 (/ 0d0 0d0) 会被编译器常量折叠，
   在折叠期就抛 FLOATING-POINT-INVALID-OPERATION（实测多一条 STYLE-WARNING）。"
  (with-float-safe (/ *zero-d* *zero-d*)))

(defun mk-nan-s ()
  (with-float-safe (/ *zero-s* *zero-s*)))

(defvar *nan-d* nil)
(defvar *nan-s* nil)

(defun cx-vals ()
  "覆盖：普通值、实部相等（比虚部）、NaN 在实部、NaN 在虚部、双 NaN、
   ±Inf、±0.0、重复值。"
  (list (complex 1d0 2d0) (complex 0d0 9d0)
        (complex 1d0 2d0) (complex 1d0 1d0)
        (complex *nan-d* 0d0) (complex 0d0 *nan-d*)
        (complex *nan-d* *nan-d*)
        (complex 1d0 0d0) (complex -1d0 0d0)
        (complex 0d0 0d0)
        (complex -0d0 0d0) (complex 0d0 -0d0)
        (complex 1d100 0d0) (complex 0d0 -1d100)
        (complex -1d100 5d0) (complex 3d0 -2d0)))

(defun cx-pair-vals ()
  "两侧配对，覆盖矩阵：有序 / 相等实部 / 一侧 NaN / 两侧 NaN / Inf / 零"
  (let ((v (cx-vals)))
    (loop for a in v append (loop for b in v collect (cons a b)))))

(defun cx-vec (vals dtype)
  (let ((t1 (vt-from-sequence vals :dtype :complex128)))
    (if (eq dtype :complex64) (vt-astype t1 :complex64) t1)))

;;; ------------------------------------------------------------------
;;; A. 极值族语义（独立参照）
;;; ------------------------------------------------------------------

(defun test-extremum-semantics ()
  (let ((pairs (cx-pair-vals))
        (ops (list (list "maximum" (function vt-maximum) (function ref-max))
                   (list "minimum" (function vt-minimum) (function ref-min))
                   (list "fmax"    (function vt-fmax)    (function ref-fmax))
                   (list "fmin"    (function vt-fmin)    (function ref-fmin)))))
    (dolist (dt (list :complex128 :complex64))
      (let* ((a (cx-vec (mapcar #'car pairs) dt))
             (b (cx-vec (mapcar #'cdr pairs) dt))
             ;; 参照必须从**张量里的实际值**算起：c64 那次是 astype 过来的，
             ;; 1d100 会变成单精度 Inf，拿原始 double 列表当参照就错位了。
             (la (vt-to-list a))
             (lb (vt-to-list b)))
        (dolist (op ops)
          (destructuring-bind (nm fn ref) op
            (let ((got (vt-to-list (funcall fn a b)))
                  (want (mapcar ref la lb)))
              (expect-true (format nil "~a ~a：~a 对取值与字典序参照一致"
                                   dt nm (length pairs))
                           (cx-lists= got want)))))))))

(defun test-extremum-layouts ()
  "同一语义在下列布局下必须一致：连续 / 广播标量 / 转置视图 / 步进视图。
   内联快路径与 vt-map 兜底的选择取决于连续性，落点语义不能跟着变。"
  (let* ((base (cx-vec (list (complex 1d0 2d0) (complex 0d0 3d0) (complex 2d0 0d0)
                             (complex 1d0 1d0) (complex -1d0 4d0) (complex 3d0 -1d0))
                       :complex128))
         (lo (cx-vec (list (complex 0d0 1d0) (complex 1d0 0d0) (complex 0d0 0d0)
                           (complex 2d0 0d0) (complex 0d0 0d0) (complex 1d0 2d0))
                     :complex128))
         (m2 (vt-reshape base (list 2 3)))
         (mt (vt-transpose m2))
         (lo2 (vt-reshape lo (list 3 2)))
         (st (vt-slice base (list 0 6 2)))
         (lo3 (vt-slice lo (list 0 6 2))))
    (flet ((chk (nm a b)
             (let* ((got (vt-to-list (vt-flatten (vt-maximum a b))))
                    ;; VT-TO-LIST 对多维返回嵌套列表 —— 这里一律先展平，
                    ;; 落点都是行主序，展平后与 GOT 逐位对应。
                    (la (vt-to-list (vt-flatten (vt-copy a))))
                    (lb (vt-to-list (vt-flatten (vt-copy b))))
                    (want (if (= (length lb) 1)
                              (mapcar (lambda (x) (ref-max x (first lb))) la)
                              (mapcar (function ref-max) la lb))))
               (expect-true (format nil "~a：极值与参照一致" nm)
                            (cx-lists= got want)))))
      (chk "连续×连续" base lo)
      (chk "连续×广播标量" base (cx-vec (list (complex 1d0 1d0)) :complex128))
      (chk "转置视图×连续" mt lo2)
      (chk "步进视图×步进视图" st lo3))
    ;; 形状与 dtype 不因布局而变
    (expect-equal "转置视图极值的形状" (vt-shape (vt-maximum mt lo2)) (list 3 2))
    (expect-equal "转置视图极值的 dtype" (vt-dtype (vt-maximum mt lo2)) :complex128)
    (expect-equal "c64 极值保 dtype"
                  (vt-dtype (vt-maximum (vt-astype base :complex64)
                                        (vt-astype lo :complex64)))
                  :complex64)))

(defun test-clip ()
  (let* ((x (cx-vec (list (complex 1d0 5d0) (complex 0d0 0d0) (complex 9d0 -9d0)
                          (complex *nan-d* 0d0) (complex 2d0 2d0))
                    :complex128))
         (lo #C(1d0 1d0))
         (hi #C(4d0 4d0))
         (got (vt-to-list (vt-clip x lo hi)))
         (want (mapcar (function ref-min)
                       (list hi hi hi hi hi)
                       (mapcar (lambda (v) (ref-max v (complex 1d0 1d0)))
                               (vt-to-list x)))))
    (expect-true "clip = min(max(x, lo), hi)" (cx-lists= got want))
    (expect-equal "clip 只给上限" (vt-to-list (vt-clip x nil hi))
                  (vt-to-list (vt-minimum x hi)))
    (expect-equal "clip 边界全省 = 副本" (vt-to-list (vt-clip x)) (vt-to-list x))
    (expect-true "clip 返回新缓冲" (not (eq (vt-data (vt-clip x)) (vt-data x))))))

;;; ------------------------------------------------------------------
;;; C. argsort
;;; ------------------------------------------------------------------

(defun pick (lst idx) (mapcar (lambda (i) (nth i lst)) idx))

(defparameter *sort-vals*
  (list (complex 1d0 2d0) (complex 0d0 9d0) (complex 1d0 2d0)
        (complex 1d0 1d0) (complex 0d0 0d0) (complex -1d0 3d0)
        (complex 5d0 -5d0) (complex 5d0 -5d0) (complex 2d0 0d0))
  "含重复值（1+2i、5-5i 各两次）与相等实部（1+2i vs 1+1i），
   同时检查排序正确性与稳定性。")

(defun stable-order-p (sv idx)
  "等值元素的原始下标必须递增（argsort 的稳定性）。SV 是排好序的值，
   IDX 是对应下标。"
  (loop for rest on sv
        for i from 0
        while (cdr rest)
        always (let ((x (first rest)) (y (second rest))
                     (ix (nth i idx)) (iy (nth (1+ i) idx)))
                 (or (not (cx= x y)) (<= ix iy)))))

(defun test-argsort-basic ()
  (dolist (dt (list :complex128 :complex64))
    (let* ((a (cx-vec *sort-vals* dt))
           (la (vt-to-list a))
           (n (length la))
           (idx (vt-to-list (vt-argsort a)))
           (sv (vt-to-list (vt-sort a))))
      (expect-equal (format nil "~a argsort 是 0..~a 的排列" dt (1- n))
                    (sort (copy-list idx) #'<)
                    (loop for i below n collect i))
      (expect-true (format nil "~a 按下标取值 == vt-sort" dt)
                   (cx-lists= (pick la idx) sv))
      (expect-true (format nil "~a argsort 结果非降序" dt)
                   (loop for (x y) on sv while y always (cx-le x y)))
      (expect-true (format nil "~a argsort 稳定（等值下标递增）" dt)
                   (stable-order-p sv idx)))))

(defun test-argsort-axes ()
  (let* ((m (vt-reshape (cx-vec (list (complex 3d0 0d0) (complex 1d0 5d0)
                                      (complex 2d0 0d0) (complex 1d0 0d0)
                                      (complex 0d0 9d0) (complex 4d0 -1d0))
                            :complex128)
                        (list 2 3)))
         (i0 (vt-argsort m :axis 0))
         (i1 (vt-argsort m :axis 1)))
    (expect-equal "argsort axis=0 形状" (vt-shape i0) (list 2 3))
    (expect-equal "argsort axis=1 形状" (vt-shape i1) (list 2 3))
    ;; axis=0：每列升序
    (dotimes (c 3)
      (let* ((col (loop for r below 2 collect (vt-ref m r c)))
             (ord (loop for r below 2 collect (vt-ref i0 r c)))
             (picked (mapcar (lambda (i) (nth i col)) ord)))
        (expect-true (format nil "axis=0 第 ~a 列升序" c)
                     (loop for (x y) on picked while y always (cx-le x y)))))
    ;; axis=1：每行升序
    (dotimes (r 2)
      (let* ((row (loop for c below 3 collect (vt-ref m r c)))
             (ord (loop for c below 3 collect (vt-ref i1 r c)))
             (picked (mapcar (lambda (i) (nth i row)) ord)))
        (expect-true (format nil "axis=1 第 ~a 行升序" r)
                     (loop for (x y) on picked while y always (cx-le x y)))))
    ;; axis=nil 展平
    (expect-equal "argsort axis=nil 形状" (vt-shape (vt-argsort m :axis nil)) (list 6))
    ;; 非连续输入（转置视图）与连续副本同结果
    (expect-equal "转置视图 argsort == 先物化再 argsort"
                  (vt-to-list (vt-argsort (vt-transpose m)))
                  (vt-to-list (vt-argsort (vt-copy (vt-transpose m)))))))

(defun test-argsort-nan-tail ()
  (let* ((a (cx-vec (list (complex 2d0 0d0) (complex *nan-d* 0d0)
                          (complex 1d0 0d0) (complex 0d0 *nan-d*)
                          (complex 0d0 1d0))
                    :complex128))
         (sv (vt-to-list (vt-sort a))))
    (expect-true "NaN 排在末尾（两处）"
                 (and (cxn (nth 3 sv)) (cxn (nth 4 sv))))
    (expect-true "非 NaN 前缀仍按字典序"
                 (and (= (realpart (nth 0 sv)) 0d0) (= (imagpart (nth 0 sv)) 1d0)
                      (= (realpart (nth 1 sv)) 1d0)
                      (= (realpart (nth 2 sv)) 2d0)))))

;;; ------------------------------------------------------------------
;;; D. topk
;;; ------------------------------------------------------------------

(defun test-topk ()
  (let* ((la (list (complex 3d0 0d0) (complex 1d0 5d0) (complex 2d0 0d0)
                   (complex 9d0 -1d0) (complex 1d0 0d0) (complex 0d0 9d0)
                   (complex 5d0 5d0) (complex -1d0 0d0)))
         (a (cx-vec la :complex128))
         (n (length la))
         (sv (vt-to-list (vt-sort a)))
         (si (vt-to-list (vt-argsort a))))
    (dolist (k (list 0 1 3 n))
      ;; largest：最后 k 个，降序
      (multiple-value-bind (v i) (vt-topk a k)
        (let ((want (reverse (last sv k)))
              (wi (reverse (last si k))))
          (expect-true (format nil "topk k=~a 值 == 排序尾部逆序" k)
                       (cx-lists= (vt-to-list v) want))
          (expect-equal (format nil "topk k=~a 下标 == 排序尾部逆序" k)
                        (vt-to-list i) wi)
          ;; 索引回指：按下标取值必须等于返回的值
          (expect-true (format nil "topk k=~a 下标回指正确" k)
                       (cx-lists= (pick la (vt-to-list i)) (vt-to-list v)))))
      ;; largest=nil：开头 k 个，升序
      (multiple-value-bind (v i) (vt-topk a k :largest nil)
        (let ((want (subseq sv 0 k))
              (wi (subseq si 0 k)))
          (expect-true (format nil "topk k=~a largest=nil 值" k)
                       (cx-lists= (vt-to-list v) want))
          (expect-equal (format nil "topk k=~a largest=nil 下标" k)
                        (vt-to-list i) wi)))
      ;; sorted=nil：按原位置升序输出
      (multiple-value-bind (v i) (vt-topk a k :sorted nil)
        (let ((ii (vt-to-list i)))
          (expect-true (format nil "topk k=~a sorted=nil 下标升序" k)
                       (loop for (x y) on ii while y always (<= x y)))
          (expect-true (format nil "topk k=~a sorted=nil 值仍与下标对应" k)
                       (cx-lists= (pick la ii) (vt-to-list v))))))
    ;; 二维沿轴
    (let* ((m (vt-reshape a (list 2 4)))
           (i1 (vt-argsort m :axis 1)))
      (multiple-value-bind (v i) (vt-topk m 2)
        (expect-equal "二维 topk 形状" (vt-shape v) (list 2 2))
        (expect-equal "二维 topk 下标 dtype" (vt-dtype i) :int64)
        (dotimes (r 2)
          (let* ((row (loop for c below 4 collect (vt-ref m r c)))
                 (ord (loop for c below 2 collect (vt-ref i1 r (- 3 c))))
                 (picked (mapcar (lambda (j) (nth j row)) ord)))
            (expect-true (format nil "二维 topk 第 ~a 行取到最大两个" r)
                         (cx-lists= picked
                                    (loop for c below 2 collect (vt-ref v r c))))))))))

;;; ------------------------------------------------------------------
;;; E. 护栏（倍率，不看绝对毫秒）
;;; ------------------------------------------------------------------

(defun ms-of (thunk &key (runs 2))
  (let ((best most-positive-fixnum))
    (dotimes (_ (1+ runs))
      (sb-ext:gc :full t)
      (let ((t0 (get-internal-real-time)))
        (funcall thunk)
        (setf best (min best (- (get-internal-real-time) t0)))))
    (/ (* 1000.0d0 best) internal-time-units-per-second)))

(defun test-guardrails ()
  ;; 规模取 1M：200k 时实数线只要 0.2~0.6ms，落到计时器噪声里
  ;; （实测 get-internal-real-time 的有效分辨率约 1ms），倍率会算成 1000x+。
  (let* ((n 1000000)
         (z1 (vt-+ (vt-zeros (list n) :dtype :complex128) #C(1.5d0 0.5d0)))
         (z2 (vt-+ (vt-zeros (list n) :dtype :complex128) #C(0.5d0 1.5d0)))
         (r1 (vt-+ (vt-zeros (list n)) 1.5d0))
         (r2 (vt-+ (vt-zeros (list n)) 0.5d0))
         (zx (vt-random-uniform (list n) :low 0d0 :high 1d0 :dtype :complex128))
         (rx (vt-random-uniform (list n) :low 0d0 :high 1d0))
         ;; 基准与被测项同口径：数据先构造好（随机数生成不算进计时）
         (t-max-c (ms-of (lambda () (vt-maximum z1 z2))))
         (t-max-r (ms-of (lambda () (vt-maximum r1 r2))))
         (t-arg-c (ms-of (lambda () (vt-argsort zx))))
         (t-arg-r (ms-of (lambda () (vt-argsort rx))))
         (t-topk  (ms-of (lambda () (vt-topk zx 10))))
         (t-sort  (ms-of (lambda () (vt-sort zx)))))
    ;; 复数极值：字典序要连虚部一起比，天然比 max(实数) 多做几倍工作，
    ;; 但绝不该出现「每元素一次函数分派」的数量级（历史上是 400 倍）。
    (report (format nil "极值族 c128/f64 倍率 ~,1fx ≤ 8x" (/ t-max-c (max t-max-r 1d-3)))
            (<= t-max-c (* 8.0d0 (max t-max-r 1d-3))))
    (report (format nil "argsort c128/f64 倍率 ~,1fx ≤ 3x" (/ t-arg-c (max t-arg-r 1d-3)))
            (<= t-arg-c (* 3.0d0 (max t-arg-r 1d-3))))
    ;; topk 只该比 argsort 多一次 gather：再退化成「又排一次序」会到 1.7x 以上
    (report (format nil "topk ≤ 1.4x argsort（~,1fx）" (/ t-topk (max t-arg-c 1d-3)))
            (<= t-topk (* 1.4d0 (max t-arg-c 1d-3))))
    ;; argsort 与 sort 同量级（同一归并内核，只多一个下标数组的搬运）
    (report (format nil "argsort ≤ 2.5x sort（~,1fx）" (/ t-arg-c (max t-sort 1d-3)))
            (<= t-arg-c (* 2.5d0 (max t-sort 1d-3))))))

;;; ------------------------------------------------------------------
;;; main
;;; ------------------------------------------------------------------

(defun main ()
  (setf *nan-d* (mk-nan-d) *nan-s* (mk-nan-s))
  (format t "~&=== 复数极值族 / argsort / topk 测试 ===")
  (test-extremum-semantics)
  (test-extremum-layouts)
  (test-clip)
  (test-argsort-basic)
  (test-argsort-axes)
  (test-argsort-nan-tail)
  (test-topk)
  (test-guardrails)
  (format t "~&~%通过: ~a  失败: ~a~%" *pass* *fail*)
  (when *failures*
    (format t "~&失败清单:~%")
    (dolist (f (reverse *failures*)) (format t "  - ~a~%" f)))
  (when (plusp *fail*)
    (sb-ext:exit :code 1)))

(main)
