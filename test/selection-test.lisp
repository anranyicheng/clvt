;;;; selection-test.lisp — 顺序统计量的**算法**（§16.14.6）
;;;;
;;;; 背景：median / nanmedian / topk 此前都是「整份排序再取要的那几个」。
;;;;   取中位数只要 O(N)、取前 k 个只要 O(N log k)，所以这是**算法选错**。
;;;;   1M complex128 实测 median 225 ms（同形 float64 走 quickselect 只要 28），
;;;;   topk k=10 174 ms（改为有界堆后 2.2）。
;;;;
;;;; 本套件：
;;;;   A. median：与 numpy 2.4.6 逐例对照（期望值见 selection-expected.sexp，
;;;;      输入由「整数取模 + 一次 double 除法」的公式生成，两边逐位相同）；
;;;;      覆盖 1/2/3 维、axis/keepdims、重复值极高（三分区）、整体降序
;;;;      （访问量护栏）、NaN、单分量 NaN、空输入。
;;;;   B. nanmedian：同上，外加全部 NaN / 沿轴全 NaN / NaN 密集。
;;;;   C. topk：与「argsort 整份排序再切」的参照**逐位一致**（值 + 下标），
;;;;      覆盖 6 种 dtype × 并列/NaN × 多轴 × sorted=nil —— 这条不能与 numpy 比，
;;;;      因为 numpy 测不出并列序的漂移。
;;;;   D. 语义：全 NaN 的复数结果必须是 NaN+NaNj（此前是 NaN+0j）；
;;;;      median 的单分量 NaN 一律 NaN+NaNj（§16.15 已知差异）。
;;;;   E. 护栏：median/nanmedian 的 c128/f64 倍率 ≤ 2.5（算法对齐的直接证据）、
;;;;      topk k=10 绝对 < 40ms 且 ≤ argsort 路径的 1/5。
;;;;
;;;; 用法：sbcl --dynamic-space-size 4096 --non-interactive --load test/selection-test.lisp

(require :asdf)
(defparameter cl-user::*clvt-test-dir*
  (make-pathname :name nil :type nil :defaults *load-truename*))
(pushnew (merge-pathnames "../" cl-user::*clvt-test-dir*) asdf:*central-registry* :test #'equal)
(handler-bind ((warning #'muffle-warning)) (asdf:load-system :clvt))

(in-package :clvt)

(defvar *pass* 0)
(defvar *fail* 0)
(defvar *failures* '())

(defun expect-true (name form)
  (if form
      (incf *pass*)
      (progn (incf *fail*) (push name *failures*) (format t "~&FAIL ~a~%" name))))

(defun feat (x y tol)
  "浮点近似相等（NaN 只与 NaN 相等；Inf 要求同号）。TOL 为相对容差。"
  (cond ((and (floatp x) (sb-ext:float-nan-p x))
         (and (floatp y) (sb-ext:float-nan-p y)))
        ((and (floatp y) (sb-ext:float-nan-p y)) nil)
        ((= x y) t)
        (t (<= (abs (- x y)) (* tol (max (abs x) (abs y)))))))

(defun ceq (a b tol)
  (and (feat (realpart a) (realpart b) tol)
       (feat (imagpart a) (imagpart b) tol)))

;;; ==================================================================
;;; 0. 输入生成器 —— 必须与 test/gen_selection_expected.py 的 gen_* 逐位一致
;;;    （整数取模 + 一次 double 除法；两边都是 IEEE 最近舍入）
;;; ==================================================================

(defun g-ramp (n)
  (loop for i from 0 below n
        collect (complex (/ (float (mod (* i 37) 1009) 1d0) 1009d0)
                         (/ (float (mod (* i 61) 1013) 1d0) 1013d0))))

(defparameter +grid-cols+ (vector -0.5d0 0d0 0.25d0 1d0))

(defun g-grid (n)
  (loop for i from 0 below n
        collect (complex (aref +grid-cols+ (mod i 4))
                         (aref +grid-cols+ (mod (floor i 4) 4)))))

(defun g-alleq (n) (loop repeat n collect (complex 0.5d0 -0.5d0)))

(defun g-desc (n)
  (loop for i from 0 below n
        for v = (/ (float (- n i) 1d0) (float n 1d0))
        collect (complex v (- v))))

(defun splice-every (seq step fn)
  "把 SEQ 中下标为 0, STEP, 2·STEP, … 的元素替换成 (FUNCALL FN)。"
  (loop for x in seq for i from 0
        collect (if (zerop (mod i step)) (funcall fn) x)))

(defun g-nan (n)
  (splice-every (g-ramp n) 7 (lambda () (complex +vt-dfloat-nan+ +vt-dfloat-nan+))))

(defun g-nanre (n)
  (splice-every (g-ramp n) 5 (lambda () (complex +vt-dfloat-nan+ 0.25d0))))

(defun g-allnan (n)
  (loop repeat n collect (complex +vt-dfloat-nan+ +vt-dfloat-nan+)))

(defun g-dense (shape)
  "先按 RAMP 公式铺满，再改：每 4 个一个 NaN+NaNj、下标 1 mod 11 一个 NaN+0.75j。"
  (let* ((n (reduce #'* shape))
         (seq (g-ramp n))
         (v (coerce seq 'vector)))
    (loop for i from 0 below n
          do (cond ((zerop (mod i 4))
                    (setf (aref v i) (complex +vt-dfloat-nan+ +vt-dfloat-nan+)))
                   ((= 1 (mod i 11))
                    (setf (aref v i) (complex +vt-dfloat-nan+ 0.75d0)))))
    (coerce v 'list)))

(defun normalize-input (seq shape)
  (let ((tv (vt-from-sequence seq :dtype :complex128)))
    (if shape (vt-reshape tv shape) tv)))

(defun build-input (kind shape)
  (let* ((n (if shape (reduce #'* shape) 0))
         (seq (cond ((string= kind "ramp") (if (zerop n) '() (g-ramp n)))
                    ((string= kind "grid") (g-grid n))
                    ((string= kind "alleq") (g-alleq n))
                    ((string= kind "desc") (g-desc n))
                    ((string= kind "nan") (g-nan n))
                    ((string= kind "nanre") (g-nanre n))
                    ((string= kind "allnan") (g-allnan n))
                    ((string= kind "dense") (g-dense shape))
                    (t (error "未知的输入类别 ~a" kind)))))
    (normalize-input seq shape)))

;;; ==================================================================
;;; A / B. median 与 nanmedian 对照
;;; ==================================================================

(defvar *expected* nil)

(defun read-expected ()
  (or *expected*
      (setf *expected*
            (with-open-file (s (merge-pathnames "selection-expected.sexp"
                                                cl-user::*clvt-test-dir*))
              (let ((*read-default-float-format* 'double-float))
                (first (read s)))))))

(defun eval-sym (x)
  "把期望值文件里的 nan / posinf / neginf 记号换成真正的浮点常量。"
  (case x
    (nan +vt-dfloat-nan+)
    (posinf +vt-dfloat-pos-inf+)
    (neginf +vt-dfloat-neg-inf+)
    (t x)))

(defun v->list (tv)
  (let ((f (vt-ravel tv)) (acc '()))
    (dotimes (i (vt-size f) (nreverse acc))
      (push (vt-ref f i) acc))))

(defun run-expect-cases (fn-name fn prefix tol &key skip)
  "只跑名字以 PREFIX 开头的用例（同一个期望值文件里并存 median 与 nanmedian）。
   SKIP 里的名字不比数值 —— 它们是 §16.15 的已知差异，由 D 节单独钉死。"
  (dolist (c (read-expected))
    (destructuring-bind (name kind shape axis keepdims vals) c
      (when (and (eql 0 (search prefix name))
                 (not (member name skip :test #'string=)))
      (let* ((in (build-input kind shape))
             (res (if axis
                      (funcall fn in :axis axis :keepdims (and keepdims t))
                      (funcall fn in :keepdims (and keepdims t))))
             (got (v->list res))
             (want (mapcar (lambda (p) (complex (eval-sym (first p))
                                                (eval-sym (second p))))
                           vals)))
          (expect-true (format nil "~a ~a" fn-name name)
                       (and (= (length got) (length want))
                            (every (lambda (a b) (ceq a b tol)) got want))))))))

(defun test-median-numpy ()
  (format t "~&--- A. median vs numpy 2.4.6 ---")
  (run-expect-cases "median" #'vt-median "median-" 1d-12
                    :skip '("median-flat-nanre200")))

(defun test-nanmedian-numpy ()
  (format t "~&--- B. nanmedian vs numpy 2.4.6 ---")
  ;; dense-axis{0,1}：那条轴上有**整根 fiber 全 NaN**（每 4 个平铺下标一个 NaN，
  ;; 步长 4 的 fiber 于是全命中），numpy 给 nan+0j、本库给 nan+nanj，见 §16.15。
  (run-expect-cases "nanmedian" #'vt-nanmedian "nanmedian-" 1d-12
                    :skip '("nanmedian-3d-dense-axis0" "nanmedian-3d-dense-axis1")))

;;; ==================================================================
;;; C. topk 与 argsort 路径逐位一致
;;; ==================================================================

(defun ref-topk (tensor k &key (axis -1) (largest t) (sorted t))
  "参照实现 = 旧路径：argsort 整份排序后切 k 个。"
  (let* ((shape (vt-shape tensor))
         (rank (length shape))
         (ax (vt-normalize-axis axis rank))
         (ax-dim (nth ax shape))
         (si (vt-argsort tensor :axis ax))
         (idxs (if largest
                   (vt-flip (vt-narrow si ax (- ax-dim k) ax-dim) :axis ax)
                   (vt-narrow si ax 0 k)))
         (vals (%gather-axis-k tensor idxs ax)))
    (if sorted
        (values vals idxs)
        (let ((perm (vt-argsort idxs :axis ax)))
          (values (%gather-along-axis vals perm ax)
                  (%gather-along-axis idxs perm ax))))))

(defun same-elem (x y)
  (cond ((and (floatp x) (sb-ext:float-nan-p x))
         (and (floatp y) (sb-ext:float-nan-p y)))
        ((complexp x)
         (and (complexp y)
              (same-elem (realpart x) (realpart y))
              (same-elem (imagpart x) (imagpart y))))
        ((complexp y) nil)
        (t (equal x y))))

(defun topk-case (label tensor k &key (axis -1) (largest t) (sorted t))
  (let (rv ri gv gi)
    (multiple-value-setq (rv ri) (ref-topk tensor k :axis axis
                                           :largest largest :sorted sorted))
    (multiple-value-setq (gv gi) (vt-topk tensor k :axis axis
                                          :largest largest :sorted sorted))
    (let ((va (v->list rv)) (vb (v->list gv))
          (ia (v->list ri)) (ib (v->list gi)))
      (expect-true (format nil "topk ~a（值）~a" label (if largest "max" "min"))
                   (and (= (length va) (length vb))
                        (every #'same-elem va vb)))
      ;; 下标必须逐位相同 —— 并列序一变就会在这里红
      (expect-true (format nil "topk ~a（下标）~a" label (if largest "max" "min"))
                   (equal ia ib))
      (expect-true (format nil "topk ~a（形状）~a" label (if largest "max" "min"))
                   (equal (vt-shape gv) (vt-shape rv))))))

(defun mk-dtype (seq dtype shape)
  (vt-reshape (vt-from-sequence seq :dtype dtype) shape))

(defun test-topk-match ()
  (format t "~&--- C. topk 与 argsort 路径逐位一致 ---")
  ;; n = 600：k=1 走堆（1·8 ≤ 600）、k=75 恰好卡在阈值（8·75=600）、
  ;; k=76 起走 argsort —— 三条支路都被覆盖。
  (dolist (pair '((:float64 . "f64") (:float32 . "f32")
                  (:int64 . "i64") (:int32 . "i32")
                  (:complex128 . "c128") (:complex64 . "c64")))
    (let* ((dtype (car pair)) (tag (cdr pair))
           (base (g-ramp 600)))
      (dolist (prof '(:distinct :ties :nan))
        (let* ((seq (case prof
                      (:distinct (mapcar (lambda (z) (realpart z)) base))
                      (:ties (mapcar (lambda (z) (mod (round (* 1000 (realpart z))) 7)) base))
                      (:nan (mapcar (lambda (z)
                                      (if (zerop (mod (round (* 1000 (realpart z))) 23))
                                          +vt-dfloat-nan+
                                          (realpart z)))
                                    base))))
               (conv (case dtype
                       ((:float64) seq)
                       ((:float32) (mapcar (lambda (v) (coerce v 'single-float)) seq))
                       ((:int64 :int32) (mapcar (lambda (v)
                                                  (if (and (floatp v)
                                                           (sb-ext:float-nan-p v))
                                                      0
                                                      (round v)))
                                                seq))
                       ((:complex128) (mapcar (lambda (v) (complex v 0.5d0)) seq))
                       ((:complex64) (mapcar (lambda (v)
                                               (complex (coerce v 'single-float) 0.5s0))
                                             seq))))
               (t1 (mk-dtype conv dtype (list 600))))
          (dolist (k '(1 3 10 75 76 200))
            (topk-case (format nil "1D ~a/~a k=~a" tag prof k) t1 k)
            (topk-case (format nil "1D ~a/~a k=~a" tag prof k) t1 k :largest nil)
            (topk-case (format nil "1D ~a/~a k=~a unsorted" tag prof k)
                       t1 k :sorted nil))
          (let ((t2 (vt-reshape t1 (list 24 25))))
            (dolist (ax '(0 1 -1))
              (dolist (k '(1 3 10))
                (topk-case (format nil "2D ax~a ~a/~a k=~a" ax tag prof k) t2 k :axis ax)
                (topk-case (format nil "2D ax~a ~a/~a k=~a" ax tag prof k)
                           t2 k :axis ax :largest nil))))
          (let ((t3 (vt-reshape t1 (list 5 8 15))))
            (dolist (ax '(0 1 2))
              (topk-case (format nil "3D ax~a ~a/~a" ax tag prof) t3 2 :axis ax))))))))

(defun test-topk-layouts ()
  (format t "~&--- C2. topk 非连续 / 边界 ---")
  (let* ((a (mk-dtype (g-ramp 120) :float64 (list 10 12)))
         (tr (vt-transpose a)))
    (topk-case "transposed k=3 ax0" tr 3 :axis 0)
    (topk-case "transposed k=3 ax1" tr 3 :axis 1)
    (topk-case "narrow k=2" (vt-narrow a 1 2 8) 2 :axis 1))
  (let ((v (mk-dtype '(3d0 1d0 2d0) :float64 '(3))))
    (topk-case "k=0" v 0)
    (topk-case "k=n" v 3)
    (topk-case "单元素" (mk-dtype '(7d0) :float64 '(1)) 1)))

;;; ==================================================================
;;; D. 语义断言
;;; ==================================================================

(defun all-nan-p (z)
  (and (sb-ext:float-nan-p (realpart z))
       (sb-ext:float-nan-p (imagpart z))))

(defun test-semantics ()
  (format t "~&--- D. 全 NaN / 单分量 NaN 的取值 ---")
  ;; 全 NaN：numpy 2.4.6 的 nanmedian / nanmax / nanmin 都给 nan+nanj。
  ;; 本库此前给 **nan+0j**（虚部那个 0 是「先取 NaN 再补 0」的副产品）。
  (let ((z (normalize-input (g-allnan 20) '(20))))
    (expect-true "全 NaN nanmedian → NaN+NaNj"
                 (all-nan-p (vt-item (vt-nanmedian z))))
    (expect-true "全 NaN nanmax → NaN+NaNj"
                 (all-nan-p (vt-item (vt-nanmax z))))
    (expect-true "全 NaN nanmin → NaN+NaNj"
                 (all-nan-p (vt-item (vt-nanmin z)))))
  ;; 空输入
  (expect-true "空数组 nanmedian → NaN+NaNj"
               (all-nan-p (vt-item (vt-nanmedian (normalize-input '() '(0))))))
  (expect-true "空数组 median → NaN+NaNj"
               (all-nan-p (vt-item (vt-median (normalize-input '() '(0))))))
  ;; 沿轴全 NaN 的 fiber：本库统一 NaN+NaNj（numpy 自己不一致，见 §16.15）
  (let* ((seq (g-ramp 12))
         (v (coerce seq 'vector)))
    (loop for i from 3 below 6 do (setf (aref v i) (complex +vt-dfloat-nan+ +vt-dfloat-nan+)))
    (let* ((t2 (normalize-input (coerce v 'list) '(4 3)))
           (r (vt-nanmedian t2 :axis 1)))
      (expect-true "沿轴全 NaN 的 fiber → NaN+NaNj（该行）"
                   (all-nan-p (vt-ref r 1)))
      (expect-true "同一次归约里未受影响的行照常出值"
                   (not (all-nan-p (vt-ref r 0))))))
  ;; median 遇单分量 NaN：本库一律 NaN+NaNj（numpy 返回那个元素本身）
  (let ((z (normalize-input (g-nanre 200) '(200))))
    (expect-true "median 遇单分量 NaN → NaN+NaNj（§16.15 已知差异）"
                 (all-nan-p (vt-item (vt-median z)))))
  ;; 重复值极多（三分区）与整体降序（访问量护栏）都要给出正确值
  (expect-true "全等 500 个的中位 = 0.5-0.5j"
               (ceq (vt-item (vt-median (normalize-input (g-alleq 500) '(500))))
                    (complex 0.5d0 -0.5d0) 0d0))
  (let* ((n 4000)
         (z (normalize-input (g-desc n) (list n)))
         (half (/ n 2))
         (want (* 0.5d0 (+ (/ (float (- n (1- half)) 1d0) (float n 1d0))
                           (/ (float (- n half) 1d0) (float n 1d0))))))
    (expect-true "整体降序 4000 个的中位（访问量护栏不改变结果）"
                 (feat (realpart (vt-item (vt-median z))) want 1d-12))))

;;; ==================================================================
;;; E. 护栏
;;; ==================================================================

(defun us-now ()
  (multiple-value-bind (s u) (sb-ext:get-time-of-day) (+ (* s 1000000) u)))

(defun ms-of (thunk &key (runs 2))
  (let ((best most-positive-fixnum))
    (dotimes (_ (1+ runs))
      (sb-ext:gc :full t)
      (let ((t0 (us-now)))
        (funcall thunk)
        (setf best (min best (- (us-now) t0)))))
    (/ best 1000.0d0)))

(defun test-guardrails ()
  (format t "~&--- E. 倍率 / 绝对护栏 ---")
  (let* ((n 1000000)
         (zr (vt-random-uniform (list n) :low 0d0 :high 1d0))
         (zc (vt-astype zr :complex128)))
    ;; median：算法对齐后复数只该比实数贵「元素宽度 + 字典序比较」那一点。
    ;; 修复前是 8.3 倍（整份排序 vs quickselect）。
    (let* ((tr (ms-of (lambda () (vt-median zr))))
           (tc (ms-of (lambda () (vt-median zc))))
           (ratio (/ tc (max tr 0.05d0))))
      (expect-true (format nil "median c128/f64 倍率 ~,1fx ≤ 2.5x" ratio)
                   (<= ratio 2.5d0)))
    (let* ((tr (ms-of (lambda () (vt-nanmedian zr))))
           (tc (ms-of (lambda () (vt-nanmedian zc))))
           (ratio (/ tc (max tr 0.05d0))))
      (expect-true (format nil "nanmedian c128/f64 倍率 ~,1fx ≤ 2.5x" ratio)
                   (<= ratio 2.5d0)))
    ;; topk：k=10 时堆应当远快于 argsort（旧路径 = ref-topk）
    (let* ((th (ms-of (lambda () (vt-topk zr 10))))
           (ta (ms-of (lambda () (ref-topk zr 10)))))
      (expect-true (format nil "topk f64 k=10 绝对 ~,1fms < 40ms（修复前 174）" th)
                   (< th 40.0d0))
      (expect-true (format nil "topk f64 k=10 比 argsort 路径快 ~,1fx ≥ 5x"
                           (/ ta (max th 0.05d0)))
                   (<= (* 5.0d0 th) ta)))
    (let ((tc (ms-of (lambda () (vt-topk zc 10)))))
      (expect-true (format nil "topk c128 k=10 绝对 ~,1fms < 60ms（修复前 297）" tc)
                   (< tc 60.0d0)))))

;;; ==================================================================

(defun main ()
  (format t "~&=== 顺序统计量（median / nanmedian / topk）测试 ===")
  (test-median-numpy)
  (test-nanmedian-numpy)
  (test-topk-match)
  (test-topk-layouts)
  (test-semantics)
  (test-guardrails)
  (format t "~&~%通过: ~a  失败: ~a~%" *pass* *fail*)
  (when *failures*
    (format t "~&失败清单:~%")
    (dolist (f (reverse *failures*)) (format t "  - ~a~%" f)))
  (when (plusp *fail*)
    (sb-ext:exit :code 1)))

(main)
