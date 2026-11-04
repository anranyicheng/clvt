;;;; extensions-opt-test.lisp — extensions/extensions2/extensions3 六个函数优化的回归测试
;;;;
;;;; 背景：本轮把下列函数从「逐元素 VT-REF/VT-CAST + 通用 AREF」改成
;;;; 「按元素类型展开的线性下标循环」：
;;;;     extensions.lisp   vt-flatnonzero
;;;;     extensions2.lisp  vt-ravel-multi-index / vt-tril-indices / vt-triu-indices / vt-vander
;;;;     extensions3.lisp  vt-resize / vt-indices
;;;; 这些改动全部只碰**执行层**，逻辑层语义一字未动（CONVENTIONS 三层分离）。
;;;; 因此本套件的核心断言是**逐位等价**：
;;;;   同一输入，优化实现的结果 ≡ 一份独立重写的朴素参照（shape / dtype / 每个元素）
;;;; 而不是「抄一份期望值」—— 抄期望值只能证明「现在是这样」，证明不了
;;;; 「和参考语义一致」，也挡不住「改对了 C 序、改错了 dtype」这类偏移。
;;;;
;;;; 第二层是**机器无关的倍率护栏**（CONVENTIONS §16.16 第 7 条：不写绝对毫秒）：
;;;; 优化一旦退回逐元素 VT-REF/VT-CAST，结果照样正确、测试照样全绿，只是慢几十倍。
;;;; 所以每个函数再配一条「优化版 ÷ 旧算法」的耗时比下界，两边同进程同机跑，
;;;; 机器快慢在比值里基本抵消。
;;;;
;;;; 三处容易踩的语义点（本套件专门覆盖）：
;;;;   1. `vt-flatnonzero` 的 NaN 必须**计入**非零（numpy: np.flatnonzero([nan]) → [0]）
;;;;   2. `vt-vander` 的窄整型溢出按补码**回绕**（numpy 语义），不是抛 TYPE-ERROR；
;;;;      `:bool` 输入的结果 dtype 是 `:int64`
;;;;   3. `vt-resize` 的空源 → 全零、源比目标长 → 截断（都是 numpy.resize 语义）
(require :asdf)
#+quicklisp (ql:quickload :clvt :silent t)
(asdf:load-system :clvt)

(defpackage :clvt-ext-opt-test
  (:use :cl :clvt)
  (:export #:run))

(in-package :clvt-ext-opt-test)

;;; ==================================================================
;;; 0. 断言与工具
;;; ==================================================================
(defvar *n* 0)
(defvar *pass* 0)
(defvar *fail* 0)
(defvar *failures* nil)

(defun nan-p (x)
  (and (floatp x)
       (sb-int:with-float-traps-masked (:invalid) (not (= x x)))))

(defun eqv (a b)
  "逐位比较：整数 EQL；浮点 EQL（因此 −0.0 ≠ 0.0 会被抓出来）+ NaN 互认；复数递归。"
  (cond ((and (complexp a) (complexp b))
         (and (eqv (realpart a) (realpart b))
              (eqv (imagpart a) (imagpart b))))
        ((or (complexp a) (complexp b)) nil)
        ((and (floatp a) (floatp b))
         (cond ((and (nan-p a) (nan-p b)) t)
               ((or (nan-p a) (nan-p b)) nil)
               ;; EQL 下 0.0/-0.0 本已不等，这里显式写出来是为了表明「故意的」
               ((and (zerop a) (zerop b)) (eql (float-sign a) (float-sign b)))
               (t (eql a b))))
        ((and (null a) (null b)) t)
        ((or (null a) (null b)) nil)
        (t (eql a b))))

(defun preview (lst &optional (k 10))
  "截断过长的期望/实得列表，只打印前 K 个（末尾用 :MORE 标记）。"
  (if (<= (length lst) k) lst (append (subseq lst 0 k) (list :more))))

(defun flat-read (v)
  "按 C 序读出展平后的**逻辑值**列表（bool 得到 T/NIL，不碰物理字节）。"
  (let* ((f (vt-ravel v))
         (n (vt-size f)))
    (loop for i below n collect (vt-ref f i))))

(defun expect (label shape dtype values thunk)
  "断言 THUNK 返回的张量与期望的 shape / dtype / 元素（C 序）三者全等。"
  (incf *n*)
  (handler-case
      (let* ((act (funcall thunk))
             (got (flat-read act))
             (gs (vt-shape act))
             (gd (vt-dtype act)))
        (if (and (equal gs shape)
                 (eq gd dtype)
                 (= (length got) (length values))
                 (every #'eqv values got))
            (incf *pass*)
            (progn
              (incf *fail*) (push label *failures*)
              (format t "  ❌ ~a~%     期望: shape=~a dtype=~a ~a~%     实得: shape=~a dtype=~a ~a~%"
                      label shape dtype (preview values) gs gd (preview got)))))
    (error (e)
      (incf *fail*) (push label *failures*)
      (format t "  ❌ ~a（意外报错 ~a: ~a）~%" label (type-of e) e))))

(defun expect-val (label expected thunk)
  "断言 THUNK 返回的标量（或列表）与 EXPECTED 逐位相等。"
  (incf *n*)
  (handler-case
      (let ((got (funcall thunk)))
        (if (eqv expected got)
            (incf *pass*)
            (progn (incf *fail*) (push label *failures*)
                   (format t "  ❌ ~a~%     期望: ~a~%     实得: ~a~%" label expected got))))
    (error (e)
      (incf *fail*) (push label *failures*)
      (format t "  ❌ ~a（意外报错 ~a: ~a）~%" label (type-of e) e))))

(defun expect-error (label thunk)
  (incf *n*)
  (handler-case
      (progn (funcall thunk)
             (incf *fail*) (push label *failures*)
             (format t "  ❌ ~a（期望报错，未报）~%" label))
    (error () (incf *pass*))))

(defun expect-true (label form-value)
  (incf *n*)
  (if form-value
      (incf *pass*)
      (progn (incf *fail*) (push label *failures*)
             (format t "  ❌ ~a（期望真，实得假）~%" label))))

;;; ==================================================================
;;; 1. 参考实现（独立重写，不共享被测实现的任何结构）
;;;    —— 全部按「定义」朴素地算：坐标、幂、取模，不碰 strides 特化循环
;;; ==================================================================
(defun ref-strides (shape)
  "C 序步长。"
  (let ((acc '()) (s 1))
    (dolist (d (reverse shape) acc)
      (push s acc)
      (setf s (* s d)))))

(defun zero-val (dtype)
  "DTYPE 的零值（物理零，逻辑层读回来的样子）。"
  (case dtype
    (:bool nil)
    (:float64 0d0)
    (:float32 0.0)
    (:complex128 (complex 0d0 0d0))
    (:complex64 (complex 0.0 0.0))
    (t 0)))

(defun cast-val (v dtype)
  "把整数 V 按 DTYPE 落地（只处理下标用得到的非负小整数）。"
  (case dtype
    (:float64 (coerce v 'double-float))
    (:float32 (coerce v 'single-float))
    (:complex128 (complex (coerce v 'double-float) 0d0))
    (:complex64 (complex (coerce v 'single-float) 0.0))
    (:bool (not (zerop v)))
    (t v)))

(defun ref-flatnonzero (tensor)
  "逐元素 VT-REF 的朴素参照：非零位置（逻辑展平序）。NaN 计入非零。"
  (let* ((f (vt-ravel tensor))
         (n (vt-size f))
         (bool (eq (vt-dtype f) :bool))
         (acc '()))
    (dotimes (i n (nreverse acc))
      (let ((v (vt-ref f i)))
        (unless (if bool (null v) (zerop v))
          (push i acc))))))

(defun ref-ravel-multi-index (multi-index shape)
  "标量模式 → 整数；批量模式 → 扁平列表。浮点按 TRUNCATE 截断（本库既有语义）。"
  (flet ((iv (x) (if (integerp x) x (truncate x))))
    (let ((st (ref-strides shape)))
      (if (every #'numberp multi-index)
          (loop for i in multi-index for s in st sum (* (iv i) s))
          (let ((n (length (first multi-index))))
            (loop for k below n collect
                  (loop for mi in multi-index for s in st
                        sum (* (iv (elt mi k)) s))))))))

(defun ref-triangle (n cols k lower)
  "按定义枚举：行主序，LOWER 取 j ≤ i+k，否则 j ≥ i+k。"
  (let ((rs '()) (cs '()))
    (dotimes (i n)
      (dotimes (j cols)
        (when (if lower (<= j (+ i k)) (>= j (+ i k)))
          (push i rs) (push j cs))))
    (values (nreverse rs) (nreverse cs))))

(defun %wrap (v bits signed)
  "取低 BITS 位；SIGNED 时按二进制补码还原。"
  (let ((u (ldb (byte bits 0) v)))
    (if (and signed (logbitp (1- bits) u)) (- u (ash 1 bits)) u)))

(defun wrap-int (v dtype)
  (case dtype
    (:int8 (%wrap v 8 t))
    (:int16 (%wrap v 16 t))
    (:int32 (%wrap v 32 t))
    (:int64 (%wrap v 64 t))
    (:uint8 (%wrap v 8 nil))
    (:uint16 (%wrap v 16 nil))
    (t v)))

(defun ref-pow (x p dtype)
  "X^P 按 DTYPE 落地。
   整数：用 bignum 算真实幂再**一次性**回绕 —— 回绕对 × 是同态，故
   「末端回绕 ≡ 逐步回绕」，但这样写与实现的递推结构完全无关。
   浮点：在目标精度里逐步乘（与 np.vander 的幂语义一致，避免 EXPT 走 exp/log）。
   复数：同样逐步乘（REALPART 的精度由 dtype 决定）。"
  (cond ((complexp x)
         (let ((acc (if (eq dtype :complex64)
                        (complex (coerce 1 'single-float) (coerce 0 'single-float))
                        (complex (coerce 1 'double-float) (coerce 0 'double-float)))))
           (dotimes (k p acc) (setf acc (* acc x)))))
        ((floatp x)
         (let ((acc (coerce 1 (if (eq dtype :float32) 'single-float 'double-float))))
           (dotimes (k p acc) (setf acc (* acc x)))))
        (t (wrap-int (expt x p) dtype))))

(defun ref-vander (vals ncols increasing dtype)
  "VALS 为定长列表，返回行主序扁平列表。"
  (let ((acc '()))
    (dolist (xi vals (nreverse acc))
      (let ((x (cond ((eq dtype :float64) (coerce xi 'double-float))
                     ((eq dtype :float32) (coerce xi 'single-float))
                     ((eq dtype :complex128) (complex (coerce xi 'double-float) 0d0))
                     ((eq dtype :complex64) (complex (coerce xi 'single-float) 0.0))
                     (t xi))))
        (dotimes (j ncols)
          (push (ref-pow x (if increasing j (- ncols 1 j)) dtype) acc))))))

(defun ref-resize (tensor shape)
  "按行主序循环取用源元素填满 SHAPE（源空 → 全零）。"
  (let* ((f (vt-ravel tensor))
         (n (vt-size f))
         (total (reduce #'* shape :initial-value 1))
         (dt (vt-dtype tensor)))
    (cond ((zerop total) '())
          ((zerop n) (loop repeat total collect (zero-val dt)))
          (t (loop for i below total collect (vt-ref f (mod i n)))))))

(defun ref-indices-layer (sh d dtype)
  "形状 SH 的一层：第 K 个位置的值是 K 在 SH 下的第 D 个坐标。
   SH 对 sparse 模式是本层自己的形状（只有 D 轴保留长度），故两模式共用。"
  (let* ((total (reduce #'* sh :initial-value 1))
         (st (ref-strides sh))
         (acc '()))
    (dotimes (k total (nreverse acc))
      (push (cast-val (mod (floor k (nth d st)) (nth d sh)) dtype) acc))))

;;; ==================================================================
;;; 2. 旧算法（性能护栏的基线）—— 逐元素 VT-REF / VT-CAST + 通用 AREF
;;; ==================================================================
(defun old-flatnonzero (tensor &key (dtype :int64))
  "旧实现逐字复刻：`(AREF DATA PTR)` 走无元素类型声明的通用分派、
   `(+ OFFSET (* I STRIDE))` 是无声明（装箱）算术、结果用 PUSH/NREVERSE
   攒成 list 再 VT-FROM-SEQUENCE 建张量。"
  (let* ((flat (vt-ravel tensor))
         (data (vt-data flat))
         (size (vt-size flat))
         (offset (vt-offset flat))
         (stride (first (vt-strides flat)))
         (result '()))
    (loop for i from 0 below size
          for ptr = (+ offset (* i stride))
          when (/= (aref data ptr) 0)
            do (push i result))
    (setf result (nreverse result))
    (if result
        (vt-from-sequence result :dtype dtype)
        (vt-zeros '(0) :dtype dtype))))

(defun old-ravel-multi-index (multi-index shape)
  "旧实现：逐元素 `(NTH i mi)` 取坐标后直接写 rdata。"
  (let* ((st (ref-strides shape))
         (n (length (first multi-index)))
         (res (vt-zeros (list n) :dtype :int64))
         (d (vt-data res)))
    (declare (type (simple-array (signed-byte 64) (*)) d))
    (dotimes (i n res)
      (setf (aref d i)
            (loop for s of-type fixnum in st
                  for mi in multi-index
                  sum (* (let ((v (nth i mi))) (if (integerp v) v (truncate v))) s))))))

(defun old-tril-indices (n &key (k 0) (m nil) (lower t))
  "旧实现：双层循环 + PUSH/NREVERSE + :INITIAL-CONTENTS 整表复制。"
  (let* ((cols (or m n))
         (rs '()) (cs '()))
    (dotimes (i n)
      (dotimes (j cols)
        (when (if lower (<= j (+ i k)) (>= j (+ i k)))
          (push i rs) (push j cs))))
    (values (vt-from-array (coerce (nreverse rs) 'vector) :dtype :int64)
            (vt-from-array (coerce (nreverse cs) 'vector) :dtype :int64))))

(defun old-vander (x &key (n nil) (increasing nil))
  "旧实现：每元素一次 `(VT-CAST (* base xi) DTYPE)`（ECASE 分派 + 通用算术）。"
  (let* ((flat (vt-flatten x))
         (dt (let ((d (vt-dtype flat))) (if (eq d :bool) :int64 d)))
         (xv (vt-contiguous (if (eq (vt-dtype flat) dt) flat (vt-astype flat dt))))
         (len (vt-size xv))
         (ncols (or n len))
         (res (vt-zeros (list len ncols) :dtype dt)))
    (dotimes (i len res)
      (let ((xi (vt-ref xv i))
            (base (clvt::vt-cast 1 dt)))
        (dotimes (j ncols)
          (if increasing
              (progn (setf (vt-ref res i j) base)
                     (setf base (clvt::vt-cast (* base xi) dt)))
              (progn (setf (vt-ref res i (- ncols 1 j)) base)
                     (setf base (clvt::vt-cast (* base xi) dt)))))))))

(defun old-resize (tensor new-shape)
  "旧实现：逐元素 `(MOD i n)` + `(SETF (VT-REF …))`。"
  (let* ((flat (vt-ravel tensor))
         (n (vt-size flat))
         (res (vt-zeros new-shape :dtype (vt-dtype tensor)))
         (total (vt-size res)))
    (when (or (zerop total) (zerop n)) (return-from old-resize res))
    (dotimes (i total res)
      (setf (vt-ref res i) (vt-ref flat (mod i n))))))

(defun old-indices (dims dtype)
  "旧实现：每元素一次 VT-CAST + 通用 (SETF (VT-REF …))，无预转缓冲。
   落点是层自己的 C 序线性下标（层由 VT-ZEROS 建成、恒为连续），
   这比真正的旧代码（多维 VT-REF 坐标折算）**更快**，所以 B8 的倍率下界
   是偏保守的。"
  (let* ((r (length dims))
         (total (reduce #'* dims :initial-value 1))
         (st (ref-strides dims))
         (layers (loop for d below r collect (vt-zeros dims :dtype dtype)))
         (datas (mapcar #'vt-data layers)))
    (dotimes (k total)
      (let ((cs (loop for s of-type fixnum in st
                      for d in dims
                      collect (mod (floor k s) d))))
        (dotimes (d r)
          (setf (aref (nth d datas) k) (clvt::vt-cast (nth d cs) dtype)))))
    layers))

;;; ==================================================================
;;; 3. 计时（机器无关倍率护栏）
;;; ==================================================================
(defun %calibrate (thunk)
  "返回 INNER，使一次测量跨 ≥ 20 ms。
   刻度陷阱：GET-INTERNAL-REAL-TIME 声称 1 µs，本机实际分辨率 1 ms，
   所以「够长」的阈值写成 20000 而不是 20。"
  (let ((inner 1))
    (loop
      (let ((t0 (get-internal-real-time)))
        (loop repeat inner do (funcall thunk))
        (let ((dt (- (get-internal-real-time) t0)))
          (cond ((>= dt 20000) (return inner))
                ((zerop dt) (setf inner (* inner 8)))
                (t (setf inner (max (1+ inner)
                                    (ceiling (* inner 30000) dt))))))))))

(defun min-us (thunk &key (reps 5))
  "单次调用耗时（µs），取**最小值**：共享虚拟机上中位数会被外部抢占拉走。"
  (funcall thunk)
  (let* ((inner (%calibrate thunk))
         (best most-positive-single-float))
    (loop repeat reps
          do (let ((t0 (get-internal-real-time)))
               (loop repeat inner do (funcall thunk))
               (let ((us (/ (float (- (get-internal-real-time) t0)) inner)))
                 (when (< us best) (setf best us)))))
    best))

(defun check-at-least (label ratio limit what)
  "断言 RATIO ≥ LIMIT（优化版相对旧算法的加速倍数下界）。"
  (incf *n*)
  (if (and (numberp ratio) (>= ratio limit))
      (progn (incf *pass*)
             (format t "  [ OK ] ~a（~a = ~,2f× ≥ ~,1f×）~%" label what ratio limit))
      (progn (incf *fail*) (push label *failures*)
             (format t "  ❌ ~a（~a = ~,2f× < ~,1f×）~%" label what (float ratio) limit))))

;;; ==================================================================
;;; A1 — vt-flatnonzero
;;; ==================================================================
(defun t-flatnonzero ()
  (format t "~%--- A1 vt-flatnonzero ---~%")
  (expect "A1.1 int64 基本 (0 1 0 2 3 0 5)" '(4) :int64 '(1 3 4 6)
          (lambda () (vt-flatnonzero (vt-from-sequence '(0 1 0 2 3 0 5)))))
  (expect "A1.2 负整数也计入非零" '(2) :int64 '(0 2)
          (lambda () (vt-flatnonzero (vt-from-sequence '(-1 0 -2) :dtype :int8))))
  (expect "A1.3 全零" '(0) :int64 '()
          (lambda () (vt-flatnonzero (vt-zeros '(5)))))
  (expect "A1.4 全非零" '(3) :int64 '(0 1 2)
          (lambda () (vt-flatnonzero (vt-from-sequence '(7 8 9)))))
  (expect "A1.5 空张量" '(0) :int64 '()
          (lambda () (vt-flatnonzero (vt-zeros '(0)))))
  (expect "A1.6 0 维零" '(0) :int64 '()
          (lambda () (vt-flatnonzero (vt-zeros '()))))
  (expect "A1.7 2D #2A((0 1)(2 0))" '(2) :int64 '(1 2)
          (lambda () (vt-flatnonzero (vt-from-array #2A((0 1) (2 0))))))
  (expect "A1.8 偏移视图 (2 6)" '(2) :int64 '(0 2)
          (lambda () (vt-flatnonzero
                      (vt-slice (vt-from-sequence '(9 0 1 0 2 0)) '(2 6)))))
  (expect "A1.9 步长视图 (0 9 3)" '(1) :int64 '(1)
          (lambda () (vt-flatnonzero
                      (vt-slice (vt-from-sequence '(0 1 0 2 0 3 0 4 0)) '(0 9 3)))))
  ;; NaN 必须算非零（numpy: np.flatnonzero([nan, 1., 0.]) → [0 1]）
  (expect "A1.10 NaN 计入非零" '(2) :int64 '(0 1)
          (lambda () (vt-flatnonzero
                      (vt-from-sequence (list (vt-get-nan :float64) 1d0 0d0)))))
  (expect "A1.11 复数：虚部非零即非零" '(2) :int64 '(1 2)
          (lambda () (vt-flatnonzero
                      (vt-from-sequence (list #c(0 0) #c(1 0) #c(0 2))
                                        :dtype :complex128))))
  (expect "A1.12 输出 dtype :int32" '(4) :int32 '(1 3 4 6)
          (lambda () (vt-flatnonzero (vt-from-sequence '(0 1 0 2 3 0 5)) :dtype :int32)))
  (expect "A1.13 输出 dtype :float64" '(4) :float64 '(1d0 3d0 4d0 6d0)
          (lambda () (vt-flatnonzero (vt-from-sequence '(0 1 0 2 3 0 5)) :dtype :float64)))
  ;; 全 dtype 与独立参照对拍（bool 单独用 T/NIL 建张量）
  (dolist (dt '(:int64 :int32 :int16 :int8 :uint16 :uint8
                :float64 :float32 :complex128 :complex64))
    (let ((v (vt-from-sequence '(0 1 0 2 3 0 5) :dtype dt)))
      (expect (format nil "A1.14 dtype ~a ≡ 参照" dt)
              (list (length (ref-flatnonzero v))) :int64 (ref-flatnonzero v)
              (lambda () (vt-flatnonzero v)))))
  (let ((b (vt-from-sequence '(nil t nil t t nil t) :dtype :bool)))
    (expect "A1.15 :bool ≡ 参照" '(4) :int64 '(1 3 4 6)
            (lambda () (vt-flatnonzero b))))
  ;; 大输入（跨页、非零稀疏）与参照一致
  (let* ((n 50000)
         (v (vt-from-sequence (loop for i below n collect (if (zerop (mod i 7)) 0 i))
                              :dtype :int64))
         (ref (ref-flatnonzero v)))
    (expect "A1.16 5 万元素 ≡ 参照" (list (length ref)) :int64 ref
            (lambda () (vt-flatnonzero v)))))

;;; ==================================================================
;;; A2 — vt-ravel-multi-index
;;; ==================================================================
(defun t-ravel-multi-index ()
  (format t "~%--- A2 vt-ravel-multi-index ---~%")
  (expect-val "A2.1 标量模式 (1 2 3) in (4 5 6) = 45" 45
              (lambda () (vt-ravel-multi-index '(1 2 3) '(4 5 6))))
  (expect-val "A2.2 标量模式 (0 0 0) = 0" 0
              (lambda () (vt-ravel-multi-index '(0 0 0) '(4 5 6))))
  (expect-val "A2.3 标量模式末元素 (3 4 5) = 119" 119
              (lambda () (vt-ravel-multi-index '(3 4 5) '(4 5 6))))
  (expect "A2.4 批量：LIST 形态" '(2) :int64 '(1 2)
          (lambda () (vt-ravel-multi-index '((0 1) (1 0)) '(2 2))))
  (expect "A2.5 批量：VECTOR 形态" '(2) :int64 '(1 2)
          (lambda () (vt-ravel-multi-index (list (vector 0 1) (vector 1 0)) '(2 2))))
  (expect "A2.6 批量：LIST/VECTOR 混用" '(2) :int64 '(1 2)
          (lambda () (vt-ravel-multi-index (list '(0 1) (vector 1 0)) '(2 2))))
  (expect "A2.7 浮点按截断处理" '(2) :int64 '(2 1)
          (lambda () (vt-ravel-multi-index (list '(1.9 0.2) '(0.7 1.5)) '(2 2))))
  (expect "A2.8 空批量" '(0) :int64 '()
          (lambda () (vt-ravel-multi-index (list '() '()) '(2 2))))
  (expect "A2.9 单元素批量" '(1) :int64 '(1)
          (lambda () (vt-ravel-multi-index (list '(0) '(1)) '(3 4))))
  ;; 步长 (20 5 1)：k=0 (0,0,0)=0；k=1 (1,0,1)=21；k=2 (2,0,1)=41
  (expect "A2.10 3D 批量" '(3) :int64 '(0 21 41)
          (lambda () (vt-ravel-multi-index '((0 1 2) (0 0 0) (0 1 1)) '(3 4 5))))
  ;; 与参照大规模对拍
  (let* ((shape '(7 11 13))
         (n 500)
         (axes (loop for s in shape
                     collect (loop for k below n collect (mod (* (1+ k) 3) s)))))
    (expect "A2.11 500×3D ≡ 参照" (list n) :int64
            (ref-ravel-multi-index axes shape)
            (lambda () (vt-ravel-multi-index axes shape)))))

;;; ==================================================================
;;; A3 — vt-tril-indices / vt-triu-indices
;;; ==================================================================
(defun check-triangle-case (n cols k &key (use-default-m nil))
  "USE-DEFAULT-M 为真时不传 :M（覆盖 (OR M N) 分支），否则显式传 COLS。"
  (multiple-value-bind (er ec) (ref-triangle n cols k t)
    (multiple-value-bind (ar ac) (vt-tril-indices n :k k :m (if use-default-m nil cols))
      (let ((tag (format nil "n=~a m=~a k=~a" n cols k)))
        (expect (format nil "A3 tril ~a rows" tag) (list (length er)) :int64 er
                (lambda () ar))
        (expect (format nil "A3 tril ~a cols" tag) (list (length ec)) :int64 ec
                (lambda () ac)))))
  (multiple-value-bind (er ec) (ref-triangle n cols k nil)
    (multiple-value-bind (ar ac) (vt-triu-indices n :k k :m (if use-default-m nil cols))
      (let ((tag (format nil "n=~a m=~a k=~a" n cols k)))
        (expect (format nil "A3 triu ~a rows" tag) (list (length er)) :int64 er
                (lambda () ar))
        (expect (format nil "A3 triu ~a cols" tag) (list (length ec)) :int64 ec
                (lambda () ac))))))

(defun t-triangle-indices ()
  (format t "~%--- A3 vt-tril-indices / vt-triu-indices ---~%")
  ;; 硬编码基准（把参照实现钉在 numpy 的已知答案上）
  (multiple-value-bind (r c) (vt-tril-indices 3)
    (expect "A3.1 tril(3) rows" '(6) :int64 '(0 1 1 2 2 2) (lambda () r))
    (expect "A3.2 tril(3) cols" '(6) :int64 '(0 0 1 0 1 2) (lambda () c)))
  (multiple-value-bind (r c) (vt-triu-indices 3)
    (expect "A3.3 triu(3) rows" '(6) :int64 '(0 0 0 1 1 2) (lambda () r))
    (expect "A3.4 triu(3) cols" '(6) :int64 '(0 1 2 1 2 2) (lambda () c)))
  (multiple-value-bind (r c) (vt-tril-indices 3 :k 2)
    (expect "A3.5 tril(3,k=2) 全选（j ≤ i+2 恒成立）" '(9) :int64
            '(0 0 0 1 1 1 2 2 2) (lambda () r))
    (expect "A3.6 tril(3,k=2) cols" '(9) :int64 '(0 1 2 0 1 2 0 1 2) (lambda () c)))
  (multiple-value-bind (r c) (vt-tril-indices 3 :k -2)
    (expect "A3.7 tril(3,k=-2) 只剩 1 个" '(1) :int64 '(2) (lambda () r))
    (expect "A3.8 tril(3,k=-2) cols" '(1) :int64 '(0) (lambda () c)))
  ;; n = 0 / 1
  (multiple-value-bind (r c) (vt-tril-indices 0)
    (expect "A3.9 tril(0) rows 空" '(0) :int64 '() (lambda () r))
    (expect "A3.10 tril(0) cols 空" '(0) :int64 '() (lambda () c)))
  (multiple-value-bind (r c) (vt-triu-indices 1)
    (expect "A3.11 triu(1) rows" '(1) :int64 '(0) (lambda () r))
    (expect "A3.12 triu(1) cols" '(1) :int64 '(0) (lambda () c)))
  ;; 参数化 × 参照
  (dolist (n '(0 1 3 5 8))
    (dolist (k '(-9 -2 0 2 9))
      (check-triangle-case n n k)))
  ;; m ≠ n（矩形）
  (dolist (pair '((2 . 4) (4 . 2) (3 . 7) (7 . 3)))
    (check-triangle-case (car pair) (cdr pair) 0)
    (check-triangle-case (car pair) (cdr pair) 1))
  ;; 默认 m（= n）单独走一遍，覆盖 (or m n) 分支
  (check-triangle-case 5 5 1 :use-default-m t)
  ;; 错误路径
  (expect-error "A3.13 tril n 为负报错" (lambda () (vt-tril-indices -1)))
  (expect-error "A3.14 triu n 为负报错" (lambda () (vt-triu-indices -1)))
  (expect-error "A3.15 tril m 为负报错" (lambda () (vt-tril-indices 2 :m -1))))

;;; ==================================================================
;;; A4 — vt-vander
;;; ==================================================================
(defun vander-vals (dt)
  "各 dtype 能表示的一组底数（无符号类型不能带负数，否则 FROM-SEQUENCE 直接报错）。"
  (case dt
    (:uint8 '(0 1 3 5 200))
    (:uint16 '(0 1 3 5 65535))
    (t '(-3 0 2 5 100))))

(defun check-vander-case (label vals &key (n nil) (increasing nil) (dtype :int64)
                                     (expect-dtype nil))
  (let* ((dt (or expect-dtype dtype))
         (ncols (or n (length vals)))
         (expected (ref-vander vals ncols increasing dt)))
    (expect label (list (length vals) ncols) dt expected
            (lambda () (let ((x (vt-from-sequence vals :dtype dtype)))
                         (vt-vander x :n n :increasing increasing))))))

(defun t-vander ()
  (format t "~%--- A4 vt-vander ---~%")
  ;; 硬编码：numpy.linalg.vander([1,2,3]) = [[1 1 1][4 2 1][9 3 1]]
  (expect "A4.1 f64 递减（默认）" '(3 3) :float64
          '(1d0 1d0 1d0 4d0 2d0 1d0 9d0 3d0 1d0)
          (lambda () (vt-vander (vt-from-sequence '(1 2 3) :dtype :float64))))
  (expect "A4.2 f64 递增" '(3 3) :float64
          '(1d0 1d0 1d0 1d0 2d0 4d0 1d0 3d0 9d0)
          (lambda () (vt-vander (vt-from-sequence '(1 2 3) :dtype :float64)
                                :increasing t)))
  (expect "A4.3 n=nil 取 len" '(3 3) :int64
          '(1 1 1 4 2 1 9 3 1)
          (lambda () (vt-vander (vt-from-sequence '(1 2 3) :dtype :int64))))
  (expect "A4.4 n=0 空列" '(3 0) :int64 '()
          (lambda () (vt-vander (vt-from-sequence '(1 2 3) :dtype :int64) :n 0)))
  (expect "A4.5 n=1 只留 x^0" '(3 1) :int64 '(1 1 1)
          (lambda () (vt-vander (vt-from-sequence '(1 2 3) :dtype :int64) :n 1)))
  (expect "A4.6 空输入" '(0 0) :int64 '()
          (lambda () (vt-vander (vt-zeros '(0) :dtype :int64))))
  (expect "A4.7 空输入 + n=4" '(0 4) :int64 '()
          (lambda () (vt-vander (vt-zeros '(0) :dtype :int64) :n 4)))
  ;; 窄整型溢出按补码回绕：100^2 = 10000 → int8 回绕成 16
  ;; 递减模式第 0 列是最高次幂 x^(n-1)：10000 mod 256 = 16
  (expect-val "A4.8 int8 回绕（100^2 → 16）" 16
              (lambda () (vt-ref (vt-vander (vt-from-sequence '(100 12) :dtype :int8)
                                            :n 3)
                                 0 0)))
  (expect-val "A4.9 uint8 回绕（200^2 → 64）" 64
              (lambda () (vt-ref (vt-vander (vt-from-sequence '(200 12) :dtype :uint8)
                                            :n 3)
                                 0 0)))
  ;; :bool 输入 → 结果 dtype :int64（numpy 语义；保持 :bool 会与 vt-pow 的
  ;; int64 约定打架，见实现 docstring）
  (expect "A4.10 :bool 输入 → int64" '(3 3) :int64
          '(1 1 1 1 1 1 1 1 1)
          (lambda () (vt-vander (vt-from-sequence '(t t t) :dtype :bool))))
  (expect "A4.11 :bool 混合 → int64" '(2 2) :int64 '(0 1 1 1)
          (lambda () (vt-vander (vt-from-sequence '(nil t) :dtype :bool))))
  ;; 偏移视图：底层数组前 1 个元素是 9，不能被算进去
  (expect "A4.12 偏移视图" '(3 2) :int64 '(2 1 3 1 4 1)
          (lambda () (vt-vander (vt-slice (vt-from-sequence '(9 2 3 4) :dtype :int64)
                                          '(1 4))
                                :n 2)))
  ;; 全 dtype × 参照（含负底数、溢出、浮点精度）
  (dolist (dt '(:int64 :int32 :int16 :int8 :uint16 :uint8 :float64 :float32
                :complex128 :complex64))
    (check-vander-case (format nil "A4.13 dtype ~a 递减 ≡ 参照" dt)
                       (vander-vals dt) :n 4 :dtype dt)
    (check-vander-case (format nil "A4.14 dtype ~a 递增 ≡ 参照" dt)
                       (vander-vals dt) :n 4 :increasing t :dtype dt))
  ;; 更长的行（强制多步递推，溢出链更长）
  (check-vander-case "A4.15 int8 长行回绕链" '(3 7 -11 100) :n 9 :dtype :int8)
  (check-vander-case "A4.16 int16 长行回绕链" '(300 -7 1234) :n 8 :dtype :int16)
  (check-vander-case "A4.17 uint16 长行回绕链" '(300 65535 7) :n 6 :dtype :uint16)
  (check-vander-case "A4.18 int32 溢出（46341^2 越界）" '(46341 -46341) :n 3
                     :dtype :int32)
  (check-vander-case "A4.19 int64 极大值回绕" '(9223372036854775807 3) :n 3
                     :dtype :int64)
  (check-vander-case "A4.20 f32 精度" '(0.1 1.7 -2.3) :n 5 :dtype :float32)
  ;; 注：numpy.vander 只接受 1 维输入，本库对 2 维输入走 VT-FLATTEN 静默展平。
  ;; 这是本轮优化之前就存在的语义偏差（优化没有改变它），修复它会改动公开
  ;; 契约，不属于「性能优化」范围，故这里只记录、不断言。
  )

;;; ==================================================================
;;; A5 — vt-resize
;;; ==================================================================
(defun check-resize-case (label tensor shape &key (zero-fill nil))
  (let* ((dt (vt-dtype tensor))
         (expected (ref-resize tensor shape)))
    (expect label shape dt
            (if zero-fill
                (loop repeat (reduce #'* shape :initial-value 1)
                      collect (zero-val dt))
                expected)
            (lambda () (vt-resize tensor shape)))))

(defun t-resize ()
  (format t "~%--- A5 vt-resize ---~%")
  (expect "A5.1 放大 3 → 5" '(5) :int64 '(1 2 3 1 2)
          (lambda () (vt-resize (vt-from-sequence '(1 2 3) :dtype :int64) '(5))))
  (expect "A5.2 缩小 6 → 4（截断）" '(4) :int64 '(1 2 3 4)
          (lambda () (vt-resize (vt-from-sequence '(1 2 3 4 5 6) :dtype :int64) '(4))))
  (expect "A5.3 2D 重排 (2 3) → (3 3)" '(3 3) :int64 '(1 2 3 4 5 6 1 2 3)
          (lambda () (vt-resize (vt-from-sequence '(1 2 3 4 5 6) :dtype :int64) '(3 3))))
  (expect "A5.4 等长（无拷贝语义变化）" '(4) :int64 '(1 2 3 4)
          (lambda () (vt-resize (vt-from-sequence '(1 2 3 4) :dtype :int64) '(4))))
  (expect "A5.5 目标元素数为 0" '(0) :int64 '()
          (lambda () (vt-resize (vt-from-sequence '(1 2 3) :dtype :int64) '(0))))
  (expect "A5.6 空源 → 全零" '(4) :int64 '(0 0 0 0)
          (lambda () (vt-resize (vt-zeros '(0) :dtype :int64) '(4))))
  (expect "A5.7 空源 + 2D 目标 → 全零" '(2 2) :int64 '(0 0 0 0)
          (lambda () (vt-resize (vt-zeros '(0) :dtype :int64) '(2 2))))
  (expect "A5.8 0 维源 → 重复标量" '(3) :int64 '(7 7 7)
          (lambda () (vt-resize (vt-reshape (vt-from-sequence '(7) :dtype :int64) '()) '(3))))
  (expect "A5.9 偏移视图源" '(6) :int64 '(2 3 2 3 2 3)
          (lambda () (vt-resize (vt-slice (vt-from-sequence '(9 2 3 9) :dtype :int64) '(1 3))
                                '(6))))
  (expect "A5.10 步长视图源" '(5) :int64 '(1 3 5 1 3)
          (lambda () (vt-resize (vt-slice (vt-from-sequence '(1 9 3 9 5) :dtype :int64) '(0 5 2))
                                '(5))))
  (expect "A5.11 :bool 展开" '(5) :bool '(t nil t nil t)
          (lambda () (vt-resize (vt-from-sequence '(t nil) :dtype :bool) '(5))))
  ;; 全 dtype × 参照
  (dolist (dt '(:int64 :int32 :int8 :uint8 :float64 :float32 :bool))
    (let ((src (cond ((eq dt :bool) (vt-from-sequence '(t nil t) :dtype dt))
                     ((eq dt :uint8) (vt-from-sequence '(1 0 2 3) :dtype dt))
                     (t (vt-from-sequence '(1 0 -2 3) :dtype dt)))))
      (check-resize-case (format nil "A5.12 dtype ~a ×2.5 ≡ 参照" dt) src '(10))
      (check-resize-case (format nil "A5.13 dtype ~a 缩小 ≡ 参照" dt) src '(2))))
  ;; 大目标（触发块倍增的多轮翻倍）
  (let* ((src (vt-from-sequence (loop for i below 1000 collect (mod i 97)) :dtype :int64))
         (dst '(100000)))
    (check-resize-case "A5.14 1000 → 100000 ≡ 参照" src dst)))

;;; ==================================================================
;;; A6 — vt-indices
;;; ==================================================================
(defun check-indices-case (label dims &key (dtype :int64) (sparse nil))
  (let* ((r (length dims))
         (shs (loop for d below r
                    collect (if sparse
                                (loop for e below r collect (if (= e d) (nth e dims) 1))
                                dims)))
         (expected (loop for d below r
                         collect (ref-indices-layer (nth d shs) d dtype))))
    (if sparse
        (let ((got (vt-indices dims :dtype dtype :sparse t)))
          (loop for d below r do
            (expect (format nil "~a [层 ~a]" label d) (nth d shs) dtype (nth d expected)
                    (lambda () (nth d got)))))
        (expect label (cons r dims) dtype
                (apply #'append expected)
                (lambda () (vt-indices dims :dtype dtype))))))

(defun t-indices ()
  (format t "~%--- A6 vt-indices ---~%")
  (check-indices-case "A6.1 (3)" '(3))
  (check-indices-case "A6.2 (2 3)" '(2 3))
  (check-indices-case "A6.3 (2 1 2)" '(2 1 2))
  (check-indices-case "A6.4 (1 1)" '(1 1))
  (check-indices-case "A6.5 (0)" '(0))
  (check-indices-case "A6.6 (2 0 3)（深层含 0 长维）" '(2 0 3))
  (check-indices-case "A6.7 (4 5 6) 3D" '(4 5 6))
  (check-indices-case "A6.8 sparse=t" '(3 4) :sparse t)
  (check-indices-case "A6.9 sparse=t 3D" '(2 3 4) :sparse t)
  ;; dtype 变体
  (dolist (dt '(:int64 :int32 :int16 :int8 :uint16 :uint8 :float64 :float32))
    (check-indices-case (format nil "A6.10 dtype ~a (3 4)" dt) '(3 4) :dtype dt))
  (check-indices-case "A6.11 dtype :complex128" '(2 3) :dtype :complex128)
  ;; :bool 的坐标按 0/1 落地（物理字节），逻辑层读回 T/NIL
  (check-indices-case "A6.12 dtype :bool" '(2 3) :dtype :bool)
  (check-indices-case "A6.13 dtype :bool 3D" '(2 2 3) :dtype :bool)
  ;; 负维度报错
  (expect-error "A6.14 负维度报错" (lambda () (vt-indices '(-1 3)))))

;;; ==================================================================
;;; B — 机器无关倍率护栏（优化版 ÷ 旧算法）
;;; ==================================================================
(defun t-ratios ()
  (format t "~%--- B 倍率护栏（优化版 / 旧算法）---~%")
  ;; B1 flatnonzero：20 万元素
  (let* ((n 200000)
         (v (vt-from-sequence (loop for i below n collect (if (zerop (mod i 3)) 0 i))
                              :dtype :int64))
         (fast (min-us (lambda () (vt-flatnonzero v)) :reps 5))
         (slow (min-us (lambda () (old-flatnonzero v)) :reps 3)))
    (check-at-least "B1 flatnonzero 200k" (/ slow fast) 3.0 "旧算法/优化版"))
  ;; B2 ravel-multi-index：8000 个 3D 索引（旧算法 NTH 是 O(n)，整批 O(n²)）
  (let* ((shape '(10 20 30))
         (n 8000)
         (axes (loop for s in shape collect (loop for k below n collect (mod k s))))
         (fast (min-us (lambda () (vt-ravel-multi-index axes shape)) :reps 5))
         (slow (min-us (lambda () (old-ravel-multi-index axes shape)) :reps 3)))
    (check-at-least "B2 ravel-multi-index 8000×3D" (/ slow fast) 10.0 "旧算法/优化版"))
  ;; B3 tril-indices：400×400（≈8 万个下标）
  (let* ((n 400)
         (fast (min-us (lambda () (vt-tril-indices n)) :reps 5))
         (slow (min-us (lambda () (old-tril-indices n)) :reps 3)))
    (check-at-least "B3 tril-indices 400" (/ slow fast) 3.0 "旧算法/优化版"))
  (let* ((n 400)
         (fast (min-us (lambda () (vt-triu-indices n)) :reps 5))
         (slow (min-us (lambda () (old-tril-indices n :lower nil)) :reps 3)))
    (check-at-least "B4 triu-indices 400" (/ slow fast) 3.0 "旧算法/优化版"))
  ;; B5 vander：20000 × 8（f64，走到显式浮点分支）
  (let* ((x (vt-from-sequence (loop for i below 20000 collect (coerce (mod i 17) 'double-float))
                              :dtype :float64))
         (fast (min-us (lambda () (vt-vander x :n 8)) :reps 5))
         (slow (min-us (lambda () (old-vander x :n 8)) :reps 3)))
    (check-at-least "B5 vander 20000×8 f64" (/ slow fast) 2.0 "旧算法/优化版"))
  ;; B6 vander：int32（走到窄整型回绕分支）
  (let* ((x (vt-from-sequence (loop for i below 20000 collect (mod i 97)) :dtype :int32))
         (fast (min-us (lambda () (vt-vander x :n 8)) :reps 5))
         (slow (min-us (lambda () (old-vander x :n 8)) :reps 3)))
    (check-at-least "B6 vander 20000×8 int32" (/ slow fast) 2.0 "旧算法/优化版"))
  ;; B7 resize：1000 → 100 万
  (let* ((src (vt-from-sequence (loop for i below 1000 collect (mod i 97)) :dtype :int64))
         (fast (min-us (lambda () (vt-resize src '(1000000))) :reps 5))
         (slow (min-us (lambda () (old-resize src '(1000000))) :reps 3)))
    (check-at-least "B7 resize 1000→1e6" (/ slow fast) 3.0 "旧算法/优化版"))
  ;; B8 indices：(60 60 60) = 21.6 万元素 × 3 层
  (let* ((dims '(60 60 60))
         (fast (min-us (lambda () (vt-indices dims)) :reps 5))
         (slow (min-us (lambda () (old-indices dims :int64)) :reps 3)))
    (check-at-least "B8 indices (60 60 60)" (/ slow fast) 2.0 "旧算法/优化版")))

;;; ==================================================================
;;; 主流程
;;; ==================================================================
(defun run ()
  (format t "~%============================================================~%")
  (format t "  extensions 优化回归测试（extensions / extensions2 / extensions3）~%")
  (format t "============================================================~%")
  (t-flatnonzero)
  (t-ravel-multi-index)
  (t-triangle-indices)
  (t-vander)
  (t-resize)
  (t-indices)
  (t-ratios)
  (format t "~%============================================================~%")
  (format t "  Total: ~a | Pass: ~a | Fail: ~a | Skip: 0~%" *n* *pass* *fail*)
  (format t "============================================================~%")
  (when *failures*
    (format t "~%Failed:~{~%  - ~a~}~%" (reverse *failures*)))
  (zerop *fail*))

(sb-ext:exit :code (if (run) 0 1))
