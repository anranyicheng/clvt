;;;; put-sort-scale-test.lisp — 下标批量写回的线性化 + 归并乒乓的回归测试
;;;;
;;;; 两处改动，逻辑层未动，都是「通用做法在每元素/每轮上重复一次」：
;;;;
;;;;   1. src/indexing.lisp  vt-put 的通用路径 + 新增 %put-flat-multi-1to1
;;;;      旧通用路径按 `(nth (mod vi n-values) val-list)` 逐个取值 ——
;;;;      VAL-LIST 是长度 N 的 LIST，NTH 本身 O(N)，整条循环因此退化成
;;;;      O(N²)。1M 索引实测：1k→1ms、10k→57ms、50k→1360ms、100k→5437ms，
;;;;      严格平方律外推约 9 分钟。另外每个元素还要 `(reverse shape)`
;;;;      做一次多维分解。连续目标上「逻辑线性下标 ≡ 物理偏移」，于是
;;;;      值改存 VECTOR（O(1) 取用）+ 下标一次收进 int64 数组 + 免分解，
;;;;      1M 实测 32 ms。
;;;;
;;;;   2. src/reduce-stats.lisp  %msort-*（sort 的归并内核）
;;;;      旧实现每轮「写 AUX + REPLACE 回 V」，一趟归并要碰两遍内存。
;;;;      改成 A/B 交替作源与目标（乒乓），每轮只读写一遍；轮数为奇数时
;;;;      才补一次回拷。1M double 实测 137 → 121 ms。
;;;;
;;;; 断言分两层：
;;;;   A 段 —— 逐位等价。`%VT-CAST-RAW` 与 `VT-CAST` 全值域对拍（拆函数
;;;;      的唯一前提就是「同一段 body」）；`vt-put` 与逐字复刻的旧通用路径
;;;;      对拍（9 目标 dtype × 3 mode × 连续/偏移切片/转置 × 1:1/循环/多值
;;;;      × 负索引）；`vt-sort` 与 STABLE-SORT 参考实现逐位相同（含 NaN 压尾）。
;;;;   B 段 —— 机器无关的倍率护栏。写回是**线性**的：1M 与 200k 的耗时比
;;;;      必须接近元素比（5×），平方实现会给出 25×。
;;;;
;;;; 不写绝对毫秒（§16.16 第 7 条），只写比值。

(require :asdf)
#+quicklisp (ql:quickload :clvt :silent t)
(asdf:load-system :clvt)

(defpackage :clvt-pscale-test
  (:use :cl :clvt)
  (:import-from :clvt
                #:ensure-vt #:vt-flatten #:vt-data #:vt-size #:vt-offset
                #:vt-strides #:vt-shape #:vt-dtype #:vt-p #:vt-out-writable-p
                #:vt-cast #:%vt-cast-raw #:vt-flatten-sequence #:with-float-safe
                #:vt-contiguous-p)
  (:export #:run))

(in-package :clvt-pscale-test)

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
      (progn (incf *fail*) (push label *failures*)
             (format t "  [FAIL] ~a~@[ —— ~a~]~%" label detail))))

(defun check-at-most (label ratio limit what)
  (incf *n*)
  (if (and (numberp ratio) (<= ratio limit))
      (progn (incf *pass*)
             (format t "  [ OK ] ~a（~a = ~,2f× ≤ ~,1f×）~%" label what (float ratio) limit))
      (progn (incf *fail*) (push label *failures*)
             (format t "  [FAIL] ~a（~a = ~,2f× > ~,1f×）~%" label what (float ratio) limit))))

(defparameter *nan* (sb-kernel:make-double-float #x7ff80000 0))
(defparameter *pinf* sb-ext:double-float-positive-infinity)
(defparameter *ninf* (- sb-ext:double-float-positive-infinity))

(defun nan-p (x)
  (and (floatp x) (sb-int:with-float-traps-masked (:invalid) (not (= x x)))))

(defun eqv (a b)
  "逐位比较：浮点看 IEEE 位模式（−0.0 ≠ +0.0），NaN 互认，其余 EQL。"
  (cond ((and (floatp a) (floatp b))
         (cond ((and (nan-p a) (nan-p b)) t)
               ((or (nan-p a) (nan-p b)) nil)
               (t (eql (if (typep a 'double-float)
                           (sb-kernel:double-float-bits a)
                           (sb-kernel:single-float-bits a))
                       (if (typep b 'double-float)
                           (sb-kernel:double-float-bits b)
                           (sb-kernel:single-float-bits b))))))
        ((and (null a) (null b)) t)
        ((or (null a) (null b)) nil)
        (t (eql a b))))

(defun base-list (tensor)
  (coerce (vt-data tensor) 'list))

(defun same-base-p (a b)
  (let ((la (base-list a)) (lb (base-list b)))
    (and (= (length la) (length lb))
         (loop for x in la for y in lb always (eqv x y)))))

(defun check-same-base (label got want)
  (check label (same-base-p got want)
         (format nil "got=~a want=~a"
                 (subseq (base-list got) 0 (min 6 (length (base-list got))))
                 (subseq (base-list want) 0 (min 6 (length (base-list want)))))))

;;; ==================================================================
;;; 1. %VT-CAST-RAW ≡ VT-CAST
;;;    拆函数的前提是「同一段 body」。批量循环只是把外层那一次
;;;    WITH-FLOAT-TRAPS-MASKED 提到循环外，转换结果必须与单值入口逐位相同。
;;; ==================================================================
(defparameter *cast-values*
  (list 0 1 -1 0 1.0d0 -1.0d0 0.5d0 -0.5d0 1.7d0 -1.7d0 0.49999999999999994d0
        127 128 -128 -129 255 256 65535 65536 2147483647 2147483648
        (expt 2 53) (1+ (expt 2 53)) -1.0d300 1.0d300 *pinf* *ninf* *nan*
        0.0d0 -0.0d0 0.0 -0.0 t nil
        (complex 3.0d0 4.0d0) (complex 0.0d0 1.0d0)))

(defparameter *cast-dtypes*
  '(:float64 :float32 :int64 :int32 :int16 :int8 :uint8 :uint16 :bool
    :complex128 :complex64))

(defun run-cast-raw-tests ()
  (format t "~&~%[A1] %VT-CAST-RAW 与 VT-CAST 逐位一致~%")
  (dolist (dt *cast-dtypes*)
    (let ((bad 0))
      (dolist (v *cast-values*)
        (let ((a (handler-case (with-float-safe (%vt-cast-raw v dt))
                   (error (e) (list :err (princ-to-string e)))))
              (b (handler-case (vt-cast v dt)
                   (error (e) (list :err (princ-to-string e))))))
          (unless (if (and (consp a) (eq (car a) :err))
                      (and (consp b) (eq (car b) :err))
                      (eqv a b))
            (incf bad)
            (format t "    ~a ~a: raw=~a vt-cast=~a~%" dt v a b))))
      (check (format nil "cast-raw ≡ vt-cast（~a，~d 个值）" dt (length *cast-values*))
             (zerop bad)))))

;;; ==================================================================
;;; 2. VT-PUT ≡ 逐字复刻的旧通用路径
;;; ==================================================================
(defun ref-put (tensor indices values mode)
  "改动前的通用路径原样复刻（LIST + NTH + 每元素多维分解）。仅用于对拍。"
  (let* ((total (vt-size tensor))
         (val-list (cond
                     ((or (eq values t) (null values)) (list (if values 1 0)))
                     ((numberp values) (list values))
                     ((or (listp values) (arrayp values))
                      (vt-flatten-sequence values :bool (eq (vt-dtype tensor) :bool)))
                     ((vt-p values)
                      (let ((flat (vt-flatten (ensure-vt values))))
                        (loop for i below (vt-size flat)
                              collect (aref (vt-data flat) i))))))
         (n-values (length val-list))
         (shape (vt-shape tensor))
         (strides (vt-strides tensor))
         (offset (vt-offset tensor))
         (data (vt-data tensor)))
    (unless (vt-out-writable-p tensor) (error "H4"))
    (when (zerop n-values) (return-from ref-put tensor))
    (let ((idx-list (if (numberp indices) (list (round indices))
                        (let ((flat (vt-flatten (ensure-vt indices))))
                          (loop for i below (vt-size flat)
                                collect (truncate (aref (vt-data flat) i)))))))
      (loop for raw in idx-list
            for vi from 0
            for val = (nth (mod vi n-values) val-list)
            do (let ((idx raw))
                 (cond ((and (>= idx 0) (< idx total)) nil)
                       ((eq mode :clip) (setq idx (max 0 (min idx (1- total)))))
                       ((eq mode :wrap) (setq idx (mod idx total)))
                       ((and (minusp idx) (>= idx (- total))) (incf idx total))
                       (t (error "索引 ~d 超出范围 [0, ~d)" raw total)))
                 (let ((rem idx) (phys offset))
                   (loop for dim in (reverse shape)
                         for stride in (reverse strides)
                         do (multiple-value-bind (q r) (floor rem dim)
                              (incf phys (* r stride)) (setf rem q)))
                   (setf (aref data phys) (vt-cast val (vt-dtype tensor))))))
      tensor)))

(defun make-base (dt n)
  (cond ((eq dt :bool) (vt-zeros (list n) :dtype :bool))
        ((eq dt :float32) (vt-zeros (list n) :dtype :float32))
        (t (vt-zeros (list n) :dtype dt))))

(defun fill-base (dt n)
  "用可预测的初值填满基数组（便于发现问题落在哪个下标）。"
  (let ((a (make-base dt n)))
    (dotimes (i n a)
      (setf (vt-ref a i)
            (case dt
              ((:bool :int8 :uint8 :int16 :uint16 :int32 :int64) (mod i 7))
              (:float32 (coerce (+ 1.0 (/ i 2.0)) 'single-float))
              (t (+ 1.0d0 (* i 0.5d0))))))))

(defun fresh-copies (dt n make-view)
  "两份独立的 (基数组 . 视图)：VT-PUT 与 REF-PUT 各改一份，互不干扰。"
  (let ((b1 (fill-base dt n))
        (b2 (fill-base dt n)))
    (values (funcall make-view b1) (funcall make-view b2))))

(defun run-put-diff-tests ()
  (format t "~&~%[A2] VT-PUT 与旧通用路径逐位对拍~%")
  (let* ((n 24)
         (dtypes '(:float64 :float32 :int64 :int32 :int16 :int8 :uint8 :uint16 :bool))
         (modes '(:raise :clip :wrap))
         (layouts
           (list (cons "contiguous" (lambda (b) b))
                 (cons "offset-slice" (lambda (b) (vt-narrow b 0 3 15)))
                 (cons "transposed" (lambda (b) (vt-transpose (vt-reshape b '(4 6))))))))
    (dolist (layout layouts)
      (dolist (dt dtypes)
        (dolist (mode modes)
          (dolist (case '(:one-to-one :cycle :longer))
            (multiple-value-bind (a b)
                (fresh-copies dt n (cdr layout))
              (let* ((total (vt-size (funcall (cdr layout) (fill-base dt n))))
                     (idx (cond ((eq case :one-to-one)
                                 (vt-from-sequence
                                  (loop for i below total
                                        collect (mod (* i 7) total))
                                  :dtype :int64))
                                ((eq case :cycle)
                                 (vt-from-sequence
                                  (loop for i below (max 1 (floor total 2))
                                        collect (mod (* i 5) total))
                                  :dtype :int64))
                                (t (vt-from-sequence
                                    (loop for i below (max 1 (floor total 3))
                                          collect (mod (* i 3) total))
                                    :dtype :int64))))
                     (vals (cond ((eq case :longer)
                                  (vt-from-sequence
                                   (loop for i below total
                                         collect (+ 1 (mod i 13)))
                                   :dtype :float64))
                                 (t (vt-from-sequence
                                     (loop for i below (vt-size idx)
                                           collect (+ 1 (mod i 11)))
                                     :dtype :float64)))))
                (handler-case
                    (progn
                      (vt-put a idx vals :mode mode)
                      (ref-put b idx vals mode)
                      (check-same-base
                       (format nil "put ~a ~a ~a ~a" (car layout) dt case mode)
                       a b))
                  (error (e)
                    ;; 两边都应报错；只有单边报错才算失败
                    (let ((other (handler-case
                                     (progn (ref-put b idx vals mode) :ok)
                                   (error () :err))))
                      (check (format nil "put ~a ~a ~a ~a（两侧一致报错）"
                                     (car layout) dt case mode)
                             (eq other :err)
                             (princ-to-string e))))))))))))
  ;; 负索引：:raise 折算、:clip 裁到 0、:wrap 折回
  (dolist (mode '(:raise :clip :wrap))
    (let ((vals (vt-from-sequence '(10.0d0 20.0d0 30.0d0) :dtype :float64))
          (idx (vt-from-sequence '(-1 -3 -5) :dtype :int64)))
      (multiple-value-bind (a b) (fresh-copies :float64 12 #'identity)
        (cond ((eq mode :raise)
               (vt-put a idx vals :mode mode)
               (ref-put b idx vals mode)
               (check-same-base (format nil "put 负索引 ~a" mode) a b))
              (t
               (vt-put a idx vals :mode mode)
               (ref-put b idx vals mode)
               (check-same-base (format nil "put 负索引 ~a" mode) a b)))))
    ;; 越界：raise 应报错，clip/wrap 不报
    (let ((vals (vt-from-sequence '(1.0d0 2.0d0) :dtype :float64))
          (idx (vt-from-sequence '(100 -100) :dtype :int64)))
      (multiple-value-bind (a b) (fresh-copies :float64 12 #'identity)
        (if (eq mode :raise)
            (check (format nil "put 越界 ~a 报错" mode)
                   (and (handler-case (progn (vt-put a idx vals :mode mode) nil)
                          (error () t))
                        (handler-case (progn (ref-put b idx vals mode) nil)
                          (error () t))))
            (progn (vt-put a idx vals :mode mode)
                   (ref-put b idx vals mode)
                   (check-same-base (format nil "put 越界 ~a" mode) a b)))))))

;;; ==================================================================
;;; 3. VT-SORT ≡ STABLE-SORT（含 NaN 压尾）
;;; ==================================================================
(defun ref-sorted (l)
  "参考实现：非 NaN 稳定升序，NaN 保持原相对顺序压到末尾。"
  (let (nums nans)
    (dolist (x l)
      (if (nan-p x) (push x nans) (push x nums)))
    (append (stable-sort (nreverse nums) #'<) (nreverse nans))))

(defun run-sort-tests ()
  (format t "~&~%[A3] VT-SORT 与 STABLE-SORT 逐位一致~%")
  (let ((seed 424242))
    (labels ((rnd () (setf seed (mod (+ (* seed 1103515245) 12345) 2147483648))))
      (dolist (n '(0 1 2 3 31 32 33 64 100 1000 5000))
        (let* ((l (loop for i below n
                        collect (let ((x (rnd)))
                                  (case (mod i 17)
                                    (0 *nan*)
                                    (1 (if (evenp x) *pinf* *ninf*))
                                    (t (- (/ x 2147483648.0d0) 0.5d0))))))
               (v (vt-from-sequence l :dtype :float64))
               (got (coerce (vt-data (vt-sort v)) 'list))
               (want (ref-sorted (copy-list l))))
          (check (format nil "sort f64 n=~d" n)
                 (and (= (length got) (length want))
                      (loop for x in got for y in want always (eqv x y)))
                 (format nil "got=~a want=~a" (subseq got 0 (min 5 (length got)))
                         (subseq want 0 (min 5 (length want)))))))
      ;; 轴排序：2D 每行独立
      (let* ((m (vt-reshape (vt-from-sequence
                             (loop for i below 600 collect
                                   (let ((x (rnd))) (- (/ x 2147483648.0d0) 0.5d0)))
                             :dtype :float64) '(100 6)))
             (s (vt-sort m :axis 1))
             (want (loop for r below 100
                         collect (ref-sorted
                                  (loop for c below 6
                                        collect (vt-ref m r c))))))
        (check "sort 2D axis=1 每行独立"
               (loop for r below 100
                     always (loop for c below 6
                                  always (eqv (vt-ref s r c)
                                              (nth c (nth r want)))))))
      ;; 单精度 / 整数
      (let* ((l (loop for i below 1000 collect (- (mod (* i 37) 1000) 500)))
             (vi (vt-from-sequence l :dtype :int64))
             (got (coerce (vt-data (vt-sort vi)) 'list))
             (want (sort (copy-list l) #'<)))
        (check "sort int64" (equal got want)))
      (let* ((l (loop for i below 1000 collect (- (mod (* i 37) 1000) 500)))
             (vf (vt-astype (vt-from-sequence l :dtype :int64) :float32))
             (got (coerce (vt-data (vt-sort vf)) 'list))
             (want (mapcar (lambda (x) (coerce x 'single-float)) (sort (copy-list l) #'<))))
        (check "sort float32" (every #'eqv got want))))))

;;; ==================================================================
;;; 4. 倍率护栏：写回必须是线性的
;;; ==================================================================
(defvar *sink* nil)

(defun %elapsed-ms (t0)
  (* 1000.0 (/ (- (get-internal-real-time) t0) internal-time-units-per-second)))

(defun best-ms (thunk &key (reps 3) (min-ms 100.0))
  (sb-ext:gc :full t)
  (setf *sink* (funcall thunk))
  (let ((inner 1))
    (loop
      (let ((t0 (get-internal-real-time)))
        (dotimes (i inner) (setf *sink* (funcall thunk)))
        (let ((ms (%elapsed-ms t0)))
          (when (>= ms min-ms) (return))
          (setf inner (max (1+ inner)
                           (min (* inner 8)
                                (ceiling (* inner (/ (* 2.0 min-ms) (max ms 1.0))))))))))
    (let ((best most-positive-double-float))
      (dotimes (r reps (/ best inner))
        (let ((t0 (get-internal-real-time)))
          (dotimes (i inner) (setf *sink* (funcall thunk)))
          (setf best (min best (%elapsed-ms t0))))))))

(defun make-put-case (n)
  (let ((tgt (vt-zeros (list n) :dtype :float64))
        (idx (vt-from-sequence (loop for i below n collect (mod (* i 7919) n))
                               :dtype :int64))
        (vals (vt-from-sequence (loop for i below n collect (mod i 101))
                                :dtype :float64)))
    (list tgt idx vals)))

(defun run-scale-guards ()
  (format t "~&~%[B1] 写回复杂度护栏（线性 vs 平方）~%")
  (let* ((small (make-put-case 200000))
         (large (make-put-case 1000000))
         (t-small (best-ms (lambda ()
                             (apply #'vt-put (first small) (rest small)))))
         (t-large (best-ms (lambda ()
                             (apply #'vt-put (first large) (rest large))))))
    (check-at-most "put 1M ÷ put 200k（元素 5×，平方实现给 25×）"
                   (/ t-large (max t-small 1e-9)) 12.0 "耗时比"))
  (format t "~&~%[B2] 排序复杂度护栏~%")
  (let* ((case-s (vt-from-sequence (loop for i below 200000
                                         collect (mod (* i 611953) 1000))
                                   :dtype :float64))
         (case-l (vt-from-sequence (loop for i below 1000000
                                         collect (mod (* i 611953) 1000))
                                   :dtype :float64))
         (t-s (best-ms (lambda () (vt-sort case-s)) :reps 3))
         (t-l (best-ms (lambda () (vt-sort case-l)) :reps 3)))
    (check-at-most "sort 1M ÷ sort 200k（N log N 给 6×，平方给 25×）"
                   (/ t-l (max t-s 1e-9)) 9.0 "耗时比")))

;;; ==================================================================
;;; 5. 入口
;;; ==================================================================
(defun run ()
  (format t "~&===== put-sort-scale-test =====~%")
  (run-cast-raw-tests)
  (run-put-diff-tests)
  (run-sort-tests)
  (run-scale-guards)
  (format t "~&~%通过 ~d / ~d~%" *pass* *n*)
  (when (plusp *fail*)
    (format t "失败 ~d 条：~%~{  - ~a~%~}" *fail* (reverse *failures*))
    (error "put-sort-scale-test 失败 ~d 条" *fail*))
  *pass*)
