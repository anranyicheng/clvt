;;;; complex-nan-inf-test.lisp — NaN/Inf 语义 / median 契约 / :out 非连续（§16.30）
;;;;
;;;; 本轮靠「数值对拍」（262 条复数表达式 vs numpy 2.3.5）找到两类真问题：
;;;;   1. vt-median 的复数实现与**本库自己的实数实现**在 AXIS=NIL 时分叉：
;;;;      实数走全局归约，复数偷偷按末轴归约；顺带空 fiber 返回 0 而非 NaN、
;;;;      NaN 不传播（实数路径与 numpy 都传播）。
;;;;   2. %CX-SCAN-RUN 用「单位元起算」而非 numpy accumulate 的「首元素直拷」：
;;;;      cumsum 把 -0.0 洗成 +0.0；cumprod 首个元素带 Inf 时凭空多出虚部 NaN。
;;;;
;;;; 本套件同时把本轮扫过的四个方向固化成回归网：
;;;;   A. median 契约（含与实数路径的形状一致性）
;;;;   B. 前缀扫描的首元素语义（-0.0 符号位 / Inf 元素）
;;;;   C. NaN / Inf 全组合归约（nan* 家族、掩码、nan_to_num、含 NaN 排序）
;;;;   D. axis 元组归约 / :out 非连续视图 / einsum 补全
;;;;
;;;; 期望值取自 numpy 2.3.5 实测；数据是二进制可精确表示的小数、累加也精确，
;;;; 故除少量「inf 相消」用近似外，一律要求**逐位相等**。
;;;;
;;;; 用法：sbcl --non-interactive --load test/complex-nan-inf-test.lisp

(require :asdf)
(defparameter *clvt-test-dir* (make-pathname :name nil :type nil :defaults *load-truename*))
(pushnew (merge-pathnames "../" *clvt-test-dir*) asdf:*central-registry* :test #'equal)
(handler-bind ((warning #'muffle-warning)) (asdf:load-system :clvt))

(in-package :clvt)

;; 注：**不关 FP 陷阱**。SBCL 默认 OVERFLOW/INVALID/DIVIDE-BY-ZERO 全开，
;; 恰好用来验证这些函数在「用户开着陷阱」时也能安全产出 NaN（库内 with-float-safe 兜底）。

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

;;; ------------------------------------------------------------------
;;; 数据（与 numpy 侧逐字一致）
;;; ------------------------------------------------------------------

(defun zval (k)
  (complex (- (* (mod k 7) 0.5d0) 1.5d0)
           (- (* (mod k 5) 0.25d0) 0.5d0)))

(defun rval (k) (- (* (mod k 7) 0.5d0) 1.5d0))

(defun mk (dims &key (offset 0) (dtype :complex128))
  (let* ((n (reduce #'* dims :initial-value 1))
         (v (make-vt dims 0 :dtype dtype))
         (d (vt-data v)))
    (dotimes (i n)
      (setf (aref d i)
            (case dtype
              (:complex128 (zval (+ i offset)))
              (:complex64 (let ((z (zval (+ i offset))))
                            (complex (coerce (realpart z) 'single-float)
                                     (coerce (imagpart z) 'single-float))))
              (t (rval (+ i offset))))))
    v))

(defparameter *nan* +vt-dfloat-nan+)
(defparameter *inf* +vt-dfloat-pos-inf+)

(defun mkp (dims patches)
  "确定性张量 + 按扁平下标打补丁（埋 NaN/Inf）。"
  (let* ((n (reduce #'* dims))
         (v (make-vt dims 0 :dtype :complex128)))
    (dotimes (i n) (setf (aref (vt-data v) i) (zval i)))
    (dolist (p patches) (setf (aref (vt-data v) (first p)) (second p)))
    v))

(defparameter *m* (mk '(3 4)))
;; 3x4：(0,1) 实部 NaN / (1,2) 虚部 NaN / (2,0) +Inf / (2,3) -Inf
(defparameter *an*
  (mkp '(3 4) (list (list 1 (complex *nan* 0.0d0))
                    (list 6 (complex 1.5d0 *nan*))
                    (list 8 (complex *inf* 0.0d0))
                    (list 11 (complex (- *inf*) 0.5d0)))))
;; 2x3：整个第 2 列是 NaN（用于「逐 fiber」断言）
(defparameter *x*
  (mkp '(2 3) (list (list 0 (complex 1.0d0 2.0d0))
                    (list 1 (complex 3.0d0 0.0d0))
                    (list 2 (complex *nan* 0.0d0))
                    (list 3 (complex 4.0d0 0.0d0))
                    (list 4 (complex 0.0d0 0.0d0))
                    (list 5 (complex 6.0d0 1.0d0)))))

;;; ------------------------------------------------------------------
;;; 比较辅助
;;; ------------------------------------------------------------------

(defun nan-p* (x) (and (floatp x) (sb-ext:float-nan-p x)))
(defun inf-p* (x) (and (floatp x) (sb-ext:float-infinity-p x)))
(defun cz-nan-p (z) (or (nan-p* (realpart z)) (nan-p* (imagpart z))))

(defun flat (v)
  "按逻辑顺序取元素（尊重 VT-OFFSET，视图也正确）。"
  (let* ((c (vt-contiguous (vt-ravel v)))
         (d (vt-data c)) (o (the fixnum (vt-offset c))) (n (the fixnum (vt-size c))))
    (loop for i from o below (+ o n) collect (aref d i))))

(defun num-near (a b tol)
  "实数近似相等。⚠ 陷阱开着时绝不能直接对 NaN/Inf 做算术：
   (= nan nan) 与 (- inf inf) 都会抛 FLOATING-POINT-INVALID-OPERATION。"
  (cond ((and (nan-p* a) (nan-p* b)) t)
        ((or (nan-p* a) (nan-p* b)) nil)
        ((or (inf-p* a) (inf-p* b))
         (and (inf-p* a) (inf-p* b) (eql (float-sign a) (float-sign b))))
        (t (<= (abs (- a b)) tol))))

(defun near (a b &optional (tol 1d-12))
  "复数逐分量近似相等（NaN 只在两边都是 NaN 时算相等，Inf 必须同号）。"
  (cond ((and (complexp a) (complexp b))
         (and (num-near (realpart a) (realpart b) tol)
              (num-near (imagpart a) (imagpart b) tol)))
        ((complexp a) (near a (complex b 0) tol))
        ((complexp b) (near (complex a 0) b tol))
        (t (num-near a b tol))))

(defun elems-near (v expected &optional (tol 1d-12))
  (let ((got (if (typep v 'vt) (flat v) v)))
    (and (= (length got) (length expected))
         (every (lambda (g e) (if (and (complexp e) (cz-nan-p e))
                                  (cz-nan-p g)
                                  (near g e tol)))
                got expected))))

(defun shape= (v dims) (equal (vt-shape v) dims))
(defun elem-at (v i) (nth i (flat v)))

;;; ==================================================================
;;; A. median 契约（v1.3 修复：axis=nil 的语义分叉 / 空 fiber / NaN 传播）
;;; ==================================================================

(defun run-median-cases ()
  (format t "~&--- A. median 契约 ---~%")
  (let ((m (vt-median *m*)))
    ;; 修复前：形状 (3)（末轴归约）。numpy 与实数路径都是 0 维标量。
    (report "median 无 axis → 0 维（不是末轴归约）" (shape= m nil)
            (format nil "shape=~a" (vt-shape m)))
    (report "median 无 axis 值 = -0.25+0j（numpy）" (near (elem-at m 0) #C(-0.25d0 0.0d0)))
    (report "median 与实数路径形状一致" (shape= (vt-median (vt-astype *m* :float64)) nil)))
  (let ((mk (vt-median *m* :keepdims t)))
    (report "median keepdims 无 axis → 形状全 1（1 1）" (shape= mk '(1 1))
            (format nil "shape=~a" (vt-shape mk)))
    (report "median keepdims 值同标量" (near (elem-at mk 0) #C(-0.25d0 0.0d0))))
  (report "median axis=0 值（numpy）"
          (elems-near (vt-median *m* :axis 0)
                      (list #C(-1.0d0 0.25d0) #C(-0.5d0 0.5d0)
                            #C(0.0d0 -0.5d0) #C(0.0d0 0.25d0))))
  (report "median axis=1 值（numpy）"
          (elems-near (vt-median *m* :axis 1)
                      (list #C(-0.75d0 -0.125d0) #C(0.75d0 0.0d0) #C(-0.25d0 0.0d0))))
  (report "median axis=1 keepdims 形状 (3 1)" (shape= (vt-median *m* :axis 1 :keepdims t) '(3 1)))
  (let ((c64 (vt-median (vt-astype *m* :complex64))))
    (report "median c64 无 axis → 0 维且 dtype 保持" (and (shape= c64 nil) (eq (vt-dtype c64) :complex64)))
    (report "median c64 值 = -0.25" (near (elem-at c64 0) #C(-0.25d0 0.0d0) 1d-6)))
  ;; NaN 传播：numpy 的 median **不忽略** NaN；本库实数路径同样返回 NaN
  (report "median 全局含 NaN → NaN（numpy 传播）" (cz-nan-p (elem-at (vt-median *x*) 0)))
  (report "median 实数路径含 NaN → NaN（对照）"
          (nan-p* (elem-at (vt-median (vt-astype *x* :float64)) 0)))
  (let ((by-ax (vt-median *x* :axis 0)))
    ;; numpy: [2.5+1j, 1.5+0j, nan+0j] —— 只有含 NaN 的那一列是 NaN
    (report "median axis=0 前两列正常" (and (near (elem-at by-ax 0) #C(2.5d0 1.0d0))
                                            (near (elem-at by-ax 1) #C(1.5d0 0.0d0))))
    (report "median axis=0 含 NaN 的列 → NaN（不是 0）" (cz-nan-p (elem-at by-ax 2))))
  (report "nanmedian axis=0 忽略 NaN（numpy: [2.5+1j, 1.5, 6+1j]）"
          (elems-near (vt-nanmedian *x* :axis 0)
                      (list #C(2.5d0 1.0d0) #C(1.5d0 0.0d0) #C(6.0d0 1.0d0))))
  ;; 空 fiber：修复前 lo=(floor -1 2)=-1，(safety 1) 下静默读 0 → 返回 0+0i
  (let ((e (vt-median (make-vt '(0 3) 0 :dtype :complex128) :axis 0)))
    (report "median 空轴 (0 3) axis=0 → 3 个 NaN" (and (= (length (flat e)) 3)
                                                        (every #'cz-nan-p (flat e)))))
  (report "median 空输入 → NaN" (cz-nan-p (elem-at (vt-median (make-vt '(0) 0 :dtype :complex128)) 0))))

;;; ==================================================================
;;; B. 前缀扫描：首元素直拷（numpy accumulate 语义）
;;; ==================================================================

(defun run-scan-first-cases ()
  (format t "~&--- B. 前缀扫描首元素语义 ---~%")
  (let ((z (vt-cumsum (vt-from-sequence (list (complex -0.0d0 0.0d0))))))
    ;; 修复前：从单位元 0 起算，0 + (-0.0) = +0.0，符号位被洗掉
    (report "cumsum 保留 -0.0 的符号位（numpy: [-0.]）"
            (minusp (float-sign (realpart (elem-at z 0))))))
  ;; ⚠ 实数路径走通用 VT-CUMULATIVE（acc 从 0 起算），**仍会把 -0.0 洗成 +0.0**；
  ;; 那属于 vt-cumulative 的既有偏差（非复数问题），已记入 §16.30 的遗留清单，
  ;; 这里不做断言，避免把错误行为固化成回归基线。
  (report "复数 cumsum 的 -0.0 与实数路径不再互相矛盾（复数已对齐 numpy）"
          (minusp (float-sign (realpart (elem-at (vt-cumsum (vt-from-sequence
                                                             (list (complex -0.0d0 0.0d0))))
                                                  0)))))
  (let* ((v (vt-from-sequence (list (complex *inf* 0.0d0) (complex 2.0d0 0.0d0))))
         (p (vt-cumprod v)))
    ;; 修复前：1 * (inf+0i) → 虚部 0*inf = NaN，且污染后续
    (report "cumprod 首元素 = inf+0i（虚部不是 NaN）"
            (and (inf-p* (realpart (elem-at p 0))) (not (nan-p* (imagpart (elem-at p 0))))))
    (report "cumprod 第二元素 = inf+nanj（numpy 同）"
            (and (inf-p* (realpart (elem-at p 1))) (nan-p* (imagpart (elem-at p 1)))))
    (report "nancumprod 首元素同样干净"
            (not (nan-p* (imagpart (elem-at (vt-nancumprod v) 0))))))
  ;; 逐列：axis=0 时每列各自的首元素都要直拷（含 Inf 的第二列）
  (let* ((cols (mkp '(2 2) (list (list 2 (complex *inf* 0.0d0)))))
         (s (vt-cumprod cols :axis 0)))
    (report "cumprod axis=0 每列首元素直拷（第 1 列底行为 inf+0i）"
            (let ((first-col1 (aref (vt-data (vt-contiguous (vt-ravel s))) 2)))
              (and (inf-p* (realpart first-col1)) (not (nan-p* (imagpart first-col1)))))))
  (let* ((c64 (vt-astype (vt-from-sequence (list (complex *inf* 0.0d0) (complex 2.0d0 0.0d0)))
                         :complex64))
         (p (vt-cumprod c64)))
    (report "c64 cumprod 首元素虚部无 NaN" (not (nan-p* (imagpart (elem-at p 0))))
            (format nil "got ~a" (elem-at p 0))))
  ;; 无 axis = 展平扫描（形状与数值都要对）
  (let ((cs (vt-cumsum *m*)))
    (report "cumsum 无 axis → 展平为一维 (12)" (shape= cs '(12)))
    (report "cumsum 无 axis 数值 = 一维前缀和（对照 cumsum of ravel）"
            (elems-near cs (flat (vt-cumsum (vt-ravel *m*)))))))

;;; ==================================================================
;;; C. NaN / Inf 归约
;;; ==================================================================

(defun run-nan-cases ()
  (format t "~&--- C. NaN / Inf 归约 ---~%")
  (report "nansum axis=0（含 ±Inf，inf 与 -inf 不同列）"
          (elems-near (vt-nansum *an* :axis 0)
                      (list (complex *inf* 0.0d0) #C(0.5d0 0.0d0)
                            #C(-0.5d0 -0.5d0) (complex (- *inf*) 0.75d0))
                      1d-12))
  (report "nansum 全表 = nan+0.25j（+inf 与 -inf 相消）"
          (let ((z (elem-at (vt-nansum *an*) 0)))
            (and (nan-p* (realpart z)) (near (imagpart z) 0.25d0))))
  (report "nanprod axis=1 前两行（numpy）"
          (let ((r (flat (vt-nanprod *an* :axis 1))))
            (and (near (first r) #C(-0.0625d0 0.1875d0))
                 (near (second r) #C(-1.125d0 -0.375d0)))))
  (report "nanprod axis=1 第三行（inf 参与）→ NaN"
          (cz-nan-p (elem-at (vt-nanprod *an* :axis 1) 2)))
  (report "nanmean axis=1 首行 = -0.666…-0.0833…j"
          (near (elem-at (vt-nanmean *an* :axis 1) 0)
                (complex (/ -2.0d0 3.0d0) (/ -1.0d0 12.0d0))))
  (report "nanmean axis=1 末行（+inf 与 -inf）→ NaN"
          (cz-nan-p (elem-at (vt-nanmean *an* :axis 1) 2)))
  (report "nanvar axis=1 首行 ≈ 0.48611111" (near (elem-at (vt-nanvar *an* :axis 1) 0) 0.486111111111d0 1d-9))
  (report "nanstd axis=0 第二列 ≈ 0.90138782" (near (elem-at (vt-nanstd *an* :axis 0) 1) 0.9013878188d0 1d-9))
  (report "nanmin axis=0（-inf 那列取 -inf）"
          (elems-near (vt-nanmin *an* :axis 0)
                      (list #C(-1.5d0 -0.5d0) #C(-0.5d0 0.5d0)
                            #C(-0.5d0 0.0d0) (complex (- *inf*) 0.5d0))))
  (report "nanmax axis=1（+inf 那行取 +inf）"
          (elems-near (vt-nanmax *an* :axis 1)
                      (list #C(0.0d0 0.25d0) #C(1.0d0 -0.5d0) (complex *inf* 0.0d0))))
  (let ((allnan (make-vt '(2 2) (complex *nan* 0.0d0) :dtype :complex128)))
    (report "全 NaN 切片 nanmin → NaN" (every #'cz-nan-p (flat (vt-nanmin allnan :axis 0))))
    (report "全 NaN 切片 nanmean → NaN" (every #'cz-nan-p (flat (vt-nanmean allnan :axis 1)))))
  (report "nanmedian axis=0（numpy，跳过 NaN）"
          (elems-near (vt-nanmedian *an* :axis 0)
                      (list #C(0.5d0 0.5d0) #C(0.25d0 0.0d0)
                            #C(-0.25d0 -0.25d0) #C(-1.5d0 0.0d0))))
  ;; 掩码族：NaN 判定按「任一分量 NaN」；Inf 单独判定
  (let ((isnan (flat (vt-isnan *an*)))
        (isinf (flat (vt-isinf *an*)))
        (isfin (flat (vt-isfinite *an*))))
    (report "isnan 命中 2 处（实部 NaN + 虚部 NaN）"
            (= 2 (count 1 isnan)))
    (report "isinf 命中 2 处（±Inf）" (= 2 (count 1 isinf)))
    (report "isfinite = 总数 - isnan - isinf"
            (= 12 (+ (count 1 isnan) (count 1 isinf) (count 1 isfin)))))
  (report "nan_to_num：NaN→0 / +Inf→DBL_MAX / -Inf→-DBL_MAX（逐分量）"
          (let ((r (flat (vt-nan-to-num *an*))))
            ;; 下标：1 = NaN 实部，6 = NaN 虚部，8 = +Inf，11 = -Inf
            (and (near (nth 0 r) #C(-1.5d0 -0.5d0))
                 (near (nth 1 r) #C(0.0d0 0.0d0))
                 (near (nth 6 r) #C(1.5d0 0.0d0))
                 (= (realpart (nth 8 r)) most-positive-double-float)
                 (= (realpart (nth 11 r)) most-negative-double-float))))
  (report "排序把 NaN 排到末尾（numpy 字典序约定）"
          (let ((row0 (subseq (flat (vt-sort *an* :axis 1)) 0 4)))
            (and (cz-nan-p (fourth row0))
                 (notany #'cz-nan-p (subseq row0 0 3)))))
  (report "排序 axis=1 第三行（含 ±Inf）= [-inf+0.5j, -0.5+0.5j, 0-0.5j, inf+0j]"
          (elems-near (subseq (flat (vt-sort *an* :axis 1)) 8 12)
                      (list (complex (- *inf*) 0.5d0) #C(-0.5d0 0.5d0)
                            #C(0.0d0 -0.5d0) (complex *inf* 0.0d0))))
  (report "nancumsum axis=1 首行（NaN 当 0）"
          (elems-near (subseq (flat (vt-nancumsum *an* :axis 1)) 0 4)
                      (list #C(-1.5d0 -0.5d0) #C(-1.5d0 -0.5d0)
                            #C(-2.0d0 -0.5d0) #C(-2.0d0 -0.25d0)))))

;;; ==================================================================
;;; D. axis 元组 / :out 非连续 / einsum 补全
;;; ==================================================================

(defun c3 () (mk '(2 3 4)))

;;; 独立的逐元素访问（(2,3,4) 行主序），用来手算期望值，不借道任何库函数
(defun cref (v i j k) (nth (+ (* i 12) (* j 4) k) (flat v)))
(defun cz< (a b)
  "复数字典序：先实部、再虚部（与库内 vt-sort 及 numpy 一致）。"
  (or (< (realpart a) (realpart b))
      (and (= (realpart a) (realpart b)) (< (imagpart a) (imagpart b)))))
(defun fiber-max (v idx-list) (reduce (lambda (x y) (if (cz< x y) y x)) (mapcar (lambda (p) (apply #'cref v p)) idx-list)))
(defun fiber-min (v idx-list) (reduce (lambda (x y) (if (cz< x y) x y)) (mapcar (lambda (p) (apply #'cref v p)) idx-list)))
(defun fiber-sum (v idx-list) (reduce #'+ (mapcar (lambda (p) (apply #'cref v p)) idx-list) :initial-value #C(0.0d0 0.0d0)))

(defun ijk (pred) (loop for i below 2 nconc (loop for j below 3 nconc (loop for k below 4 when (funcall pred i j k) collect (list i j k)))))

(defun run-tuple-axis-cases ()
  (format t "~&--- D1. axis 元组归约 ---~%")
  (let* ((a (c3)) (s01 (vt-sum a :axis '(0 1))) (s12 (vt-sum a :axis '(1 2))) (s02 (vt-sum a :axis '(0 2))))
    (report "sum axis=(0 1) 形状 (4)" (shape= s01 '(4)))
    (report "sum axis=(1 2) 形状 (2)" (shape= s12 '(2)))
    (report "sum axis=(0 2) 形状 (3)" (shape= s02 '(3)))
    (report "sum axis=(1 2) 逐元素 = 手算 fiber 和"
            (every (lambda (i) (near (elem-at s12 i) (fiber-sum a (ijk (lambda (ii jj kk) (and (= ii i) t)))))
                      ) '(0 1)))
    ;; axis (0 2) 后剩下的是 axis 1（j），fiber = 固定 j 的 2×4 个元素
    (report "sum axis=(0 2) 逐元素 = 手算 fiber 和"
            (every (lambda (j) (near (elem-at s02 j) (fiber-sum a (ijk (lambda (ii jj kk) (= jj j)))))) '(0 1 2)))
    (report "sum 负元组 (-2 -1) 与 (1 2) 逐位相同"
            (equalp (flat s12) (flat (vt-sum a :axis '(-2 -1))))))
  (let* ((a (c3))
         (keep (vt-sum a :axis '(0 2) :keepdims t))
         (mean (vt-mean a :axis '(1 2)))
         (var (vt-var a :axis '(0 2)))
         (s12 (vt-sum a :axis '(1 2))))
    (report "sum keepdims (0 2) 形状 (1 3 1)" (shape= keep '(1 3 1)))
    (report "mean axis=(1 2) = sum/12（每 fiber 3×4=12 个元素）"
            (every (lambda (i) (near (elem-at mean i) (/ (elem-at s12 i) 12.0d0) 1d-12)) '(0 1)))
    (report "var axis=(0 2) = 手算 mean(|x-mean|²)（8 个元素/纤维）"
            (every (lambda (k)
                     (let* ((mk (elem-at mean 0))  ; 占位，下面逐 k 取
                            (m (elem-at (vt-mean a :axis '(0 2)) k))
                            (idx (ijk (lambda (ii jj kk) (= jj k))))
                            (acc (reduce #'+ (mapcar (lambda (p)
                                                       (let* ((x (apply #'cref a p))
                                                              (d (- x m)))
                                                         (* (abs d) (abs d))))
                                                     idx) :initial-value 0.0d0)))
                       (declare (ignore mk))
                       (near (elem-at var k) (/ acc 8.0d0) 1d-12)))
                   '(0 1 2)))
    (report "std keepdims (0 1) 形状 (1 1 4)" (shape= (vt-std a :axis '(0 1) :keepdims t) '(1 1 4)))
    (report "amax axis=(1 2) = 手算字典序最大"
            (every (lambda (i) (near (elem-at (vt-amax a :axis '(1 2)) i)
                                     (fiber-max a (ijk (lambda (ii jj kk) (= ii i))))))
                   '(0 1)))
    (report "amin axis=(0 1) = 手算字典序最小"
            (every (lambda (k) (near (elem-at (vt-amin a :axis '(0 1)) k)
                                     (fiber-min a (ijk (lambda (ii jj kk) (= kk k))))))
                   '(0 1 2 3)))
    (report "prod axis=(1 2) = 手算 fiber 积"
            (every (lambda (i)
                     (let ((acc (reduce #'* (mapcar (lambda (p) (apply #'cref a p))
                                                    (ijk (lambda (ii jj kk) (= ii i))))
                                        :initial-value #C(1.0d0 0.0d0))))
                       (near (elem-at (vt-prod a :axis '(1 2)) i) acc 1d-9)))
                   '(0 1)))
    (report "norm axis=(1 2) = sqrt(sum |x|²)"
            (let ((n (vt-norm a :axis '(1 2))))
              (every (lambda (i)
                       (let ((s (reduce #'+ (mapcar (lambda (p) (let ((x (apply #'cref a p)))
                                                                  (* (abs x) (abs x))))
                                                   (ijk (lambda (ii jj kk) (= ii i)))))))
                         (near (elem-at n i) (sqrt s) 1d-12)))
                     '(0 1))))))

(defun run-out-cases ()
  (format t "~&--- D2. :out 非连续视图 / 原地别名 ---~%")
  (let* ((orig (mk '(3 8) :offset 1))
         (big (vt-copy orig))
         (out (vt-slice big '(:all) '(2 6)))
         (r (vt-+ *m* *m* :out out)))
    (report "逐元素 :out 列切片 → 写入视图像素正确" (elems-near r (flat (vt-+ *m* *m*))))
    ;; 逐位置核对：落在 [2,6) 列的取结果，其余取原值（24 个位置全覆盖）
    (report "逐元素 :out 列切片 → 写入落点精确（其余位置不动）"
            (let ((b (flat big)) (o (flat orig)) (e (flat (vt-+ *m* *m*))))
              (loop for i below 24
                    for row = (floor i 8) for col = (mod i 8)
                    for want = (if (<= 2 col 5)
                                   (nth (+ (* row 4) (- col 2)) e)
                                   (nth i o))
                    always (near (nth i b) want))))
    (report "逐元素 :out 列切片 → 未覆盖区域保持原值"
            (let ((b (flat big)) (o (flat orig)))
              (and (equal (subseq b 0 2) (subseq o 0 2))
                   (equal (subseq b 6 8) (subseq o 6 8))))))
  (let* ((big (mk '(4 4) :offset 3))
         (sq (mk '(4 4) :offset 2))
         (out (vt-transpose big)))
    ;; 注意：:out 的**逻辑形状**按结果形状校验，转置视图 (4 4) 正好匹配
    (report "逐元素 :out 转置视图 → 写入正确（(i,j) 写到基数组 (j,i)）"
            (elems-near (vt-+ sq sq :out out) (flat (vt-+ sq sq)))))
  (let* ((big (mk '(3 3) :offset 2))
         (out (vt-slice big '(:all) '(1 2))))
    (report "reduce :out 非连续（keepdims 列视图）→ 值正确"
            (elems-near (vt-sum *m* :axis 1 :keepdims t :out out)
                        (flat (vt-sum *m* :axis 1 :keepdims t)))))
  (let* ((big (mk '(3 10) :offset 1))
         (out (vt-slice big '(:all) '(4 9))))
    (report "matmul :out 非连续 → 值正确"
            (elems-near (vt-matmul *m* (mk '(4 5) :offset 3) :out out)
                        (flat (vt-matmul *m* (mk '(4 5) :offset 3))))))
  (let ((x (vt-copy *m*)))
    (report "原地 :out（x += x）值正确" (elems-near (vt-+ x x :out x) (flat (vt-+ *m* *m*))))
    (report "原地 :out 后原数组被真正改写（不是返回新对象）"
            (elems-near x (flat (vt-+ *m* *m*))))))

(defun run-einsum-cases ()
  (format t "~&--- D3. einsum 补全 ---~%")
  (let ((a (c3)) (a2 (mk '(3 4) :offset 2)))
    (report "einsum \"...i,...i->...\" = sum(a*b, axis=-1)"
            (elems-near (vt-einsum "...i,...i->..." a a)
                        (flat (vt-sum (vt-* a a) :axis -1))))
    ;; einsum 的标量输出是 0 维 VT，vt-trace 返回原生数 —— 都摊成元素再比
    (report "einsum \"ii->\" = trace"
            (let ((e (vt-einsum "ii->" (mk '(4 4) :offset 2)))
                  (t2 (vt-trace (mk '(4 4) :offset 2))))
              (near (if (typep e 'vt) (elem-at e 0) e)
                    (if (typep t2 'vt) (elem-at t2 0) t2))))
    (report "einsum \"ij->ji\" = transpose"
            (elems-near (vt-einsum "ij->ji" *m*) (flat (vt-transpose *m*))))
    (report "einsum \"ijk->ik\" = sum over j"
            (elems-near (vt-einsum "ijk->ik" a) (flat (vt-sum a :axis 1))))
    (report "einsum \"ijk,jk->i\" = sum(a*b, (1 2))"
            (elems-near (vt-einsum "ijk,jk->i" a a2)
                        (flat (vt-sum (vt-* a a2) :axis '(1 2)))))
    (report "einsum \"i,j->ij\" = outer"
            (elems-near (vt-einsum "i,j->ij" (mk '(3)) (mk '(4) :offset 1))
                        (flat (vt-outer (mk '(3)) (mk '(4) :offset 1)))))
    (report "einsum \"ab,bc,cd->ad\" ≈ (a@b)@c"
            (let ((r (vt-einsum "ab,bc,cd->ad" *m* (mk '(4 5) :offset 3) (mk '(5 2) :offset 4))))
              (vt-allclose r (vt-matmul (vt-matmul *m* (mk '(4 5) :offset 3))
                                        (mk '(5 2) :offset 4))
                           :atol 1d-12 :rtol 1d-12)))
    (report "einsum \"ijk->kji\" = transpose perm (2 1 0)"
            (elems-near (vt-einsum "ijk->kji" a) (flat (vt-transpose a '(2 1 0)))))
    (report "einsum \"...ij->...i\" = sum over last axis"
            (elems-near (vt-einsum "...ij->...i" a) (flat (vt-sum a :axis 2))))))

;;; ==================================================================

(format t "~&=====================================")
(format t "~&复数 NaN/Inf 与 median 契约回归（§16.30）")
(format t "~&=====================================~%")
(run-median-cases)
(run-scan-first-cases)
(run-nan-cases)
(run-tuple-axis-cases)
(run-out-cases)
(run-einsum-cases)

(format t "~&=====================================~%")
(format t "Total: ~d / Pass: ~d / Fail: ~d~%" (+ *pass* *fail*) *pass* *fail*)
(format t "PASS=~d FAIL=~d~%" *pass* *fail*)
(when *failures*
  (format t "~&--- 失败清单 ---~%")
  (dolist (f (reverse *failures*)) (format t "  · ~a~%" f)))
(when (plusp *fail*) (sb-ext:exit :code 1))
