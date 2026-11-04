;;;; predicate-mask-perf-test.lisp — 谓词族 / 掩码 / 索引收集 一轮优化的回归测试
;;;;
;;;; 背景（v0.4.21）：本轮把下列函数从「逐元素闭包 + 通用 AREF + 逐元素 VT-CAST」
;;;; 改成「具名内联算子 / 按元素类型展开的线性下标循环」，只碰**执行层**，
;;;; 逻辑层语义一字未动（CONVENTIONS §1 三层分离）：
;;;;   elementwise1.lisp  vt-positive-p / vt-negative-p / vt-zero-p /
;;;;                      vt-nonzero-p / vt-even-p / vt-odd-p
;;;;   extensions3.lisp   vt-isposinf / vt-isneginf / vt-signbit /
;;;;                      vt-trim-zeros / vt-corrcoef
;;;;   core.lisp          %to-bool（= vt-astype … :bool）
;;;;   indexing.lisp      vt-choose / vt-nonzero / vt-argwhere / vt-extract
;;;;                      + 新共享内核 %flat-nonzero-indices / %coords-from-flat
;;;;   extensions.lisp    vt-flatnonzero（改为委托共享内核）
;;;;   map-reduce.lisp    +vt-mask-in-types+ 增加 (unsigned-byte 8) 一格
;;;;
;;;; 因此核心断言是**逐位等价**：同一输入，优化实现 ≡ 一份独立重写的朴素参照
;;;; （按定义枚举坐标 / 逐元素按真值语义判定），比对 shape / dtype / 每个元素。
;;;; 不抄期望值 —— 抄期望值只能证明「现在是这样」，挡不住「改对了算术、
;;;; 改错了 NaN 或 −0.0」这类偏移，而本轮的改动恰好全部踩在掩码上。
;;;;
;;;; 第二层是**机器无关的倍率护栏**（CONVENTIONS §16.16 第 7 条：不写绝对毫秒）：
;;;; 优化一旦退回闭包路径，结果照样正确、测试照样全绿，只是慢十几倍。
;;;; 所以每个函数再配一条「旧算法 ÷ 优化版」的耗时比下界，两边同进程同机跑。
;;;;
;;;; 四类容易踩的语义点（本套件专门覆盖）：
;;;;   1. `vt-nonzero-p` 的 NaN 必须是**真**（numpy: bool(nan) is True），
;;;;      −0.0 必须是**假**；复数是「任一分量非零」。
;;;;   2. `vt-astype … :bool` 是**真值化**而不是整数截断（0.5 → T，不是 NIL），
;;;;      且 `:uint8` 里合法存放的 2 也必须收敛成 1。
;;;;   3. `vt-argwhere` / `vt-nonzero` 的坐标分解必须与「按定义枚举」一致，
;;;;      包括转置 / 步长 / 偏移视图（里程计的实现只在升序下标上成立）。
;;;;   4. `vt-choose` 的混合 dtype 分支与 `:wrap` / `:clip` 归一化。
(require :asdf)
#+quicklisp (ql:quickload :clvt :silent t)
(asdf:load-system :clvt)

(defpackage :clvt-pred-test
  (:use :cl :clvt)
  (:export #:run))

(in-package :clvt-pred-test)

;;; ==================================================================
;;; 0. 断言与工具
;;; ==================================================================
(defvar *n* 0)
(defvar *pass* 0)
(defvar *fail* 0)
(defvar *failures* nil)

(defun nan-p (x)
  (and (floatp x) (sb-int:with-float-traps-masked (:invalid) (not (= x x)))))

(defun eqv (a b)
  "逐位比较：整数 EQL；浮点 EQL（−0.0 ≠ 0.0 会被抓出来）+ NaN 互认；复数递归；
   布尔按 T/NIL 比。复数与时，任一侧非复数即判不等。"
  (cond ((and (complexp a) (complexp b))
         (and (eqv (realpart a) (realpart b))
              (eqv (imagpart a) (imagpart b))))
        ((or (complexp a) (complexp b)) nil)
        ((and (floatp a) (floatp b))
         (cond ((and (nan-p a) (nan-p b)) t)
               ((or (nan-p a) (nan-p b)) nil)
               ((and (zerop a) (zerop b)) (eql (float-sign a) (float-sign b)))
               (t (eql a b))))
        ((and (null a) (null b)) t)
        ((or (null a) (null b)) nil)
        (t (eql a b))))

(defun preview (lst &optional (k 10))
  (if (<= (length lst) k) lst (append (subseq lst 0 k) (list :more))))

(defun flat-read (v)
  "按 C 序读出展平后的**逻辑值**列表。"
  (let* ((f (vt-ravel v))
         (n (vt-size f)))
    (loop for i below n collect (vt-ref f i))))

(defun expect (label shape dtype values thunk)
  (incf *n*)
  (handler-case
      (let* ((act (funcall thunk))
             (got (flat-read act))
             (gs (vt-shape act))
             (gd (vt-dtype act)))
        (if (and (equal gs shape) (eq gd dtype)
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

;;; ==================================================================
;;; 1. 参考实现（按定义朴素重写，不共享被测实现的任何结构）
;;; ==================================================================

;;; --- 1.0 浮点陷阱 ---
(defmacro with-ieee (&body body)
  "参照实现用普通 CL 算术（`>` / `floor` / `abs`），一旦遇上 NaN 或 ±Inf
   就会撞上 SBCL 默认开启的浮点陷阱。被测实现内部有 `with-float-safe`，
   参照实现没有 —— 这里显式屏蔽，目的是让参照**算得出结果**去和实现比对，
   而不是「参照自己崩了，于是没人发现实现错了」。"
  `(sb-int:with-float-traps-masked (:invalid) ,@body))

;;; --- 1.1 真值语义（CONVENTIONS §17.1）---
(defun ref-truthy (x)
  "任意元素 → T/NIL：非零为真，NaN 为真，−0.0 为假，复数任一分量非零为真。
   与 `%bool-flag` 的定义逐条对应，但独立写成 Lisp 谓词。"
  (with-ieee
    (cond ((eq x t) t)
          ((null x) nil)
          ((complexp x) (or (not (zerop (realpart x))) (not (zerop (imagpart x)))))
          ((floatp x) (if (nan-p x) t (not (zerop x))))
          (t (not (zerop x))))))

;;; --- 1.2 谓词族 ---
(defun ref-positive-p (x) (with-ieee (if (and (not (complexp x)) (> x 0.0d0)) t nil)))
(defun ref-negative-p (x) (with-ieee (if (and (not (complexp x)) (< x 0.0d0)) t nil)))
(defun ref-zero-p     (x) (with-ieee (if (zerop x) t nil)))
(defun ref-nonzero-p  (x) (with-ieee (if (zerop x) nil t)))

(defun ref-even-p (x)
  "NaN/±Inf → 假；非整数 → 假；否则 FLOOR 后判偶。"
  (with-ieee
    (cond ((nan-p x) nil)
          ((and (floatp x) (> (abs x) most-positive-double-float)) nil)
          ((and (floatp x) (/= x (floor x))) nil)
          (t (evenp (floor x))))))

(defun ref-odd-p (x)
  (with-ieee
    (cond ((nan-p x) nil)
          ((and (floatp x) (> (abs x) most-positive-double-float)) nil)
          ((and (floatp x) (/= x (floor x))) nil)
          (t (oddp (floor x))))))

(defun ref-posinf (x)
  (with-ieee (and (floatp x) (not (nan-p x)) (> x most-positive-double-float))))
(defun ref-neginf (x)
  (with-ieee (and (floatp x) (not (nan-p x)) (< x (- most-positive-double-float)))))

(defun ref-signbit (x)
  "IEEE 符号位：位模式最高位。"
  (logbitp 63 (ldb (byte 64 0)
                   (sb-kernel:double-float-bits (coerce x 'double-float)))))

;;; --- 1.3 坐标枚举（按定义递归）---
(defun ref-coords (v)
  "按定义枚举 C 序的每个元素，收集非零者的坐标；返回坐标列表（每条长度 = rank）。
   rank 0 时返回 :scalar 标记（由调用方单独处理）。"
  (let ((shape (vt-shape v))
        (bool (eq (vt-dtype v) :bool))
        (acc '())
        (coord nil))
    (setf coord (loop repeat (length shape) collect 0))
    (labels ((truthy-at ()
               (let ((x (apply #'vt-ref v coord)))
                 (if bool (not (null x)) (not (zerop x)))))
             (walk (d)
               (if (= d (length shape))
                   (when (truthy-at) (push (copy-list coord) acc))
                   (dotimes (i (nth d shape))
                     (setf (nth d coord) i)
                     (walk (1+ d))))))
      (if (null shape)
          (progn (push '() acc) (nreverse acc))
          (progn (walk 0) (nreverse acc))))))

(defun ref-strides (shape)
  "C 序（行优先）步长：最后一维 1，逐维乘以后一维长度。"
  (let ((s 1) (out '()))
    (dolist (d (reverse shape) out)
      (push s out)
      (setf s (* s d)))))

(defun coords->argwhere-values (coords rank)
  "坐标列表 → argwhere 那串 (n×rank) 的展平值。RANK 用不上（每条坐标已定长），
   留在参数位上只是为了让调用处与 `coords->nonzero-layer` 对齐可读。"
  (declare (ignore rank))
  (loop for c in coords append c))
(defun coords->nonzero-layer (coords d)
  (loop for c in coords collect (nth d c)))

;;; ==================================================================
;;; 2. 输入样本
;;; ==================================================================
(defun %mk (vals dtype)
  (vt-from-array (coerce vals 'vector) :dtype dtype))

(defun %nan-df (negative)
  "构造一个 quiet double-float NaN（位模式 7FF80000_00000000 / FFF80000_00000000）；
   NEGATIVE 为真时符号位为 1。库内没有负 NaN 的公开构造函数，
   而 `vt-signbit` 的契约恰好要求「符号位」对 NaN 也成立，所以这里直接按位拼。"
  (sb-kernel:make-double-float (if negative #x-80000 #x7FF80000) 0))

(defun sample-vectors ()
  "返回 (名字 . 张量) 列表：覆盖 9 种逻辑 dtype + 特殊值。
   每个样本都是 (n) 形状，n 固定为 12。"
  (let ((f (list -2.5d0 -1.0d0 -0.0d0 0.0d0 0.5d0 1.0d0 2.0d0 3.0d0
                 clvt::+vt-dfloat-nan+ clvt::+vt-dfloat-pos-inf+
                 clvt::+vt-dfloat-neg-inf+ 4.5d0)))
    (list (cons "f64"  (%mk f :float64))
          (cons "f32"  (vt-astype (%mk f :float64) :float32))
          (cons "i64"  (%mk '(0 -1 2 -3 4 0 -5 6 0 -7 8 -9) :int64))
          (cons "i32"  (%mk '(0 -1 2 -3 4 0 -5 6 0 -7 8 -9) :int32))
          (cons "i16"  (%mk '(0 -1 2 -3 4 0 -5 6 0 -7 8 -9) :int16))
          (cons "i8"   (%mk '(0 -1 2 -3 4 0 -5 6 0 -7 8 -9) :int8))
          (cons "u8"   (%mk '(0 1 2 3 4 0 5 6 0 7 8 9) :uint8))
          (cons "u16"  (%mk '(0 1 2 3 4 0 5 6 0 7 8 9) :uint16))
          (cons "bool" (%mk (loop for x in '(0 1 2 0 1 0 3 0 1 0 1 0) collect (not (zerop x)))
                            :bool))
          (cons "c128" (%mk (list #C(0.0d0 0.0d0) #C(0.0d0 -0.0d0) #C(0.0d0 1.0d0)
                                  #C(1.0d0 0.0d0) #C(-1.0d0 0.0d0) #C(0.0d0 0.0d0)
                                  #C(2.0d0 -3.0d0) #C(0.0d0 0.0d0) #C(-0.0d0 -0.0d0)
                                  #C(0.0d0 0.0d0) #C(1.5d0 1.5d0) #C(0.0d0 0.0d0))
                            :complex128))
          (cons "c64"  (vt-astype (%mk (list #C(0.0d0 0.0d0) #C(0.0d0 -0.0d0) #C(0.0d0 1.0d0)
                                             #C(1.0d0 0.0d0) #C(-1.0d0 0.0d0) #C(0.0d0 0.0d0)
                                             #C(2.0d0 -3.0d0) #C(0.0d0 0.0d0) #C(-0.0d0 -0.0d0)
                                             #C(0.0d0 0.0d0) #C(1.5d0 1.5d0) #C(0.0d0 0.0d0))
                                   :complex128)
                            :complex64)))))

;;; ==================================================================
;;; A1 — 谓词族逐位等价
;;; ==================================================================
(defun %as-number (x)
  "`:bool` 张量的逻辑值是 T/NIL；谓词族在物理层看到的是 ub8 的 1/0。
   参照实现也先落到 1/0 再判，才是在跟同一层比。"
  (cond ((eq x t) 1) ((null x) 0) (t x)))

(defun ref-flag-list (v ref)
  "把 REF 逐元素作用在 V 上，收敛成 T/NIL 列表（与 `:bool` 张量的逻辑值对齐）。"
  (mapcar (lambda (x) (let ((r (funcall ref (%as-number x)))) (if r t nil)))
          (flat-read v)))

(defun t-predicates ()
  (format t "~%--- A1 谓词族：vt-positive-p / negative-p / zero-p / nonzero-p ---~%")
  (dolist (pair (sample-vectors))
    (let ((name (car pair)) (v (cdr pair)))
      ;; 复数：`vt-positive-p` / `vt-negative-p` 没有专门的复数内核
      ;; （掩码表 +vt-mask-in-types+ 不含复数），匿名/具名算子都会落到
      ;; `(> x 0)` 上，由 `>` 抛 TYPE-ERROR —— 这是改动前就有的行为，
      ;; 本轮只换执行层，错误条件也必须一模一样。
      (if (member (vt-dtype v) '(:complex128 :complex64))
          (progn
            (expect-error (format nil "A1.1 ~a positive-p 拒绝复数" name)
                          (lambda () (vt-positive-p v)))
            (expect-error (format nil "A1.2 ~a negative-p 拒绝复数" name)
                          (lambda () (vt-negative-p v))))
          (progn
            (expect (format nil "A1.1 ~a positive-p" name) '(12) :bool
                    (ref-flag-list v #'ref-positive-p)
                    (lambda () (vt-positive-p v)))
            (expect (format nil "A1.2 ~a negative-p" name) '(12) :bool
                    (ref-flag-list v #'ref-negative-p)
                    (lambda () (vt-negative-p v)))))
      (expect (format nil "A1.3 ~a zero-p" name) '(12) :bool
              (ref-flag-list v #'ref-zero-p)
              (lambda () (vt-zero-p v)))
      (expect (format nil "A1.4 ~a nonzero-p" name) '(12) :bool
              (ref-flag-list v #'ref-nonzero-p)
              (lambda () (vt-nonzero-p v)))))
  ;; 显式 dtype（非 :bool 时要按 0/1 数值落地）
  (let ((v (%mk (list -1.0d0 0.0d0 2.0d0) :float64)))
    (expect "A1.5 positive-p :dtype :float64 落 0/1" '(3) :float64 '(0.0d0 0.0d0 1.0d0)
            (lambda () (vt-positive-p v :dtype :float64)))
    (expect "A1.6 nonzero-p :dtype :int32 落 0/1" '(3) :int32 '(1 0 1)
            (lambda () (vt-nonzero-p v :dtype :int32))))
  ;; 视图：偏移 / 步长 / 转置
  (let ((base (%mk '(9 0 -1 0 3 0 -5 0 7 0 -9 0) :float64)))
    (expect "A1.7 偏移视图 positive-p" '(6) :bool '(nil nil t nil nil nil)
            (lambda () (vt-positive-p (vt-slice base '(2 8)))))
    ;; 步长视图：base[0,3,6,9] = 9,0,-5,0 —— 让「真」和「假」交错，
    ;; 否则一条全零的视图会让任何实现都通过
    (expect "A1.8 步长视图 nonzero-p" '(4) :bool '(t nil t nil)
            (lambda () (vt-nonzero-p (vt-slice base '(0 12 3)))))
    (let ((m (vt-reshape (%mk '(0 1 -2 3 0 -4 5 0 6 -7 0 8) :float64) '(3 4))))
      (expect "A1.9 转置视图 zero-p" '(4 3) :bool
              '(t t nil  nil nil nil  nil nil t  nil t nil)
              (lambda () (vt-zero-p (vt-transpose m))))))
  ;; :out 契约
  (let ((o (vt-zeros '(3) :dtype :bool)))
    (vt-nonzero-p (%mk '(0 1 2) :int64) :out o)
    (expect "A1.10 :out 就地写入" '(3) :bool '(nil t t) (lambda () o)))

  (format t "~%--- A2 谓词族：vt-even-p / vt-odd-p ---~%")
  (dolist (pair (sample-vectors))
    (let ((name (car pair)) (v (cdr pair)))
      (if (member (vt-dtype v) '(:complex128 :complex64))
          (progn
            (expect-error (format nil "A2.1 ~a even-p 拒绝复数" name)
                          (lambda () (vt-even-p v)))
            (expect-error (format nil "A2.2 ~a odd-p 拒绝复数" name)
                          (lambda () (vt-odd-p v))))
          (progn
            (expect (format nil "A2.3 ~a even-p" name) '(12) :bool
                    (ref-flag-list v #'ref-even-p)
                    (lambda () (vt-even-p v)))
            (expect (format nil "A2.4 ~a odd-p" name) '(12) :bool
                    (ref-flag-list v #'ref-odd-p)
                    (lambda () (vt-odd-p v)))))))
  (let ((v (%mk '(0.0d0 1.0d0 2.0d0 -2.0d0 -3.0d0 2.5d0 -2.5d0) :float64)))
    (expect "A2.5 浮点整值/半值" '(7) :bool
            '(t nil t t nil nil nil) (lambda () (vt-even-p v)))
    (expect "A2.6 浮点奇偶互补" '(7) :bool
            '(nil t nil nil t nil nil) (lambda () (vt-odd-p v))))

  (format t "~%--- A3 谓词族：vt-isposinf / vt-isneginf / vt-signbit ---~%")
  (dolist (pair (sample-vectors))
    (let ((name (car pair)) (v (cdr pair)))
      (when (member (vt-dtype v) '(:float64 :float32 :int64 :int32 :uint8))
        (expect (format nil "A3.1 ~a isposinf" name) '(12) :bool
                (ref-flag-list (vt-astype v :float64) #'ref-posinf)
                (lambda () (vt-isposinf v)))
        (expect (format nil "A3.2 ~a isneginf" name) '(12) :bool
                (ref-flag-list (vt-astype v :float64) #'ref-neginf)
                (lambda () (vt-isneginf v)))
        (expect (format nil "A3.3 ~a signbit" name) '(12) :bool
                (ref-flag-list (vt-astype v :float64) #'ref-signbit)
                (lambda () (vt-signbit v))))))
  ;; 负 NaN 的符号位必须是 1，正 NaN 是 0
  (let ((pz (%mk (list (%nan-df nil)) :float64))
        (nz (%mk (list (%nan-df t)) :float64)))
    (expect "A3.4 负 NaN 符号位 → 真" '(1) :bool '(t) (lambda () (vt-signbit nz)))
    (expect "A3.5 正 NaN 符号位 → 假" '(1) :bool '(nil) (lambda () (vt-signbit pz)))))

;;; ==================================================================
;;; A4 — vt-astype … :bool（%to-bool 真值化）
;;; ==================================================================
(defun t-astype-bool ()
  (format t "~%--- A4 vt-astype … :bool（真值化）---~%")
  (dolist (pair (sample-vectors))
    (let ((name (car pair)) (v (cdr pair)))
      (expect (format nil "A4.1 ~a → :bool" name) '(12) :bool
              (mapcar #'ref-truthy (flat-read v))
              (lambda () (vt-astype v :bool)))))
  ;; 非整数浮点：0.5 必须是 T（不是整数截断的 0）
  (expect "A4.2 0.5 → T（不是截断）" '(4) :bool '(nil t t nil)
          (lambda () (vt-astype (%mk '(0.0d0 0.5d0 -0.5d0 0.0d0) :float64) :bool)))
  ;; :uint8 里合法的 2 必须收敛成 1（不能原样搬字节）
  (expect "A4.3 :uint8 的 2 → 1" '(4) :bool '(nil t t t)
          (lambda () (vt-astype (%mk '(0 1 2 255) :uint8) :bool)))
  ;; 复数：任一分量非零（第 4 个的虚部是 −0.0，两分量都为零 → 假）
  (expect "A4.4 复数任一分量非零" '(4) :bool '(nil t t nil)
          (lambda () (vt-astype (%mk (list #C(0.0d0 0.0d0) #C(0.0d0 1.0d0)
                                           #C(1.0d0 0.0d0) #C(0.0d0 -0.0d0))
                                     :complex128)
                                :bool)))
  ;; 2D + 视图（转置后 [[0 2][1 0][0 3]] → 展平 0,2,1,0,0,3）
  (let ((m (vt-reshape (%mk '(0 1 0 2 0 3) :int64) '(2 3))))
    (expect "A4.5 2D 形状保持" '(2 3) :bool '(nil t nil t nil t)
            (lambda () (vt-astype m :bool)))
    (expect "A4.6 转置视图" '(3 2) :bool '(nil t t nil nil t)
            (lambda () (vt-astype (vt-transpose m) :bool))))
  (expect "A4.7 空张量" '(0) :bool '()
          (lambda () (vt-astype (vt-zeros '(0) :dtype :int64) :bool)))
  (expect "A4.8 0 维" '() :bool '(t)
          (lambda () (vt-astype (vt-reshape (%mk '(3) :int64) '()) :bool)))
  ;; 幂等：:bool → :bool
  (let ((b (%mk '(1 0 1 0) :bool)))
    (expect "A4.9 :bool → :bool 幂等" '(4) :bool '(t nil t nil)
            (lambda () (vt-astype b :bool)))))

;;; ==================================================================
;;; A5 — vt-choose
;;; ==================================================================
(defun ref-choose (choices idx mode)
  "按定义：result[i] = choices[normalize(idx[i])][i]。CHOICES 与 IDX 都是
   已展平的 Lisp 值列表（同一长度）。"
  (let* ((n (length choices))
         (out '()))
    (dolist (raw idx)
      (let ((k (ecase mode
                 (:raise raw)
                 (:wrap  (mod raw n))
                 (:clip  (max 0 (min (1- n) raw))))))
        (when (or (< k 0) (>= k n))
          (error "ref-choose: 越界 ~a" raw))
        (push k out)))
    (mapcar (lambda (k ci) (nth ci (nth k choices)))
            (nreverse out)
            (loop for i below (length idx) collect i))))

(defun t-choose ()
  (format t "~%A5 vt-choose ---~%")
  (let* ((c0 (%mk '(1 2 3 4 5 6) :int64))
         (c1 (%mk '(10 20 30 40 50 60) :int64))
         (c2 (%mk '(100 200 300 400 500 600) :int64))
         (i0 (%mk '(0 1 2 0 1 2) :int64))
         (iw (%mk '(0 -1 -4 3 1 -2) :int64))
         (ic (%mk '(0 -5 2 9 1 -1) :int64))
         (f0 (%mk '(1.5d0 2.5d0 3.5d0 4.5d0 5.5d0 6.5d0) :float64)))
    (expect "A5.1 三选一 :raise" '(6) :int64
            (ref-choose (list '(1 2 3 4 5 6) '(10 20 30 40 50 60) '(100 200 300 400 500 600))
                        '(0 1 2 0 1 2) :raise)
            (lambda () (vt-choose (list c0 c1 c2) i0)))
    (expect "A5.2 :wrap 负数取模" '(6) :int64
            (ref-choose (list '(1 2 3 4 5 6) '(10 20 30 40 50 60) '(100 200 300 400 500 600))
                        '(0 -1 -4 3 1 -2) :wrap)
            (lambda () (vt-choose (list c0 c1 c2) iw :mode :wrap)))
    (expect "A5.3 :clip 裁剪" '(6) :int64
            (ref-choose (list '(1 2 3 4 5 6) '(10 20 30 40 50 60) '(100 200 300 400 500 600))
                        '(0 -5 2 9 1 -1) :clip)
            (lambda () (vt-choose (list c0 c1 c2) ic :mode :clip)))
    (expect-error "A5.4 :raise 越界报错"
                  (lambda () (vt-choose (list c0 c1 c2) ic)))
    ;; 广播索引 (2,1) × 候选 (3) → (2,3)。语义是「逐元素配对」：
    ;; result[r][c] = choices[idx[r][c]][r][c]，索引广播后每行整体选一个候选。
    (let ((ib (vt-reshape (%mk '(0 1) :int64) '(2 1))))
      (expect "A5.5 索引广播" '(2 3) :int64 '(1 2 3 10 20 30)
              (lambda () (vt-choose (list (vt-slice c0 '(0 3)) (vt-slice c1 '(0 3))) ib))))
    ;; 候选广播（标量候选）
    (expect "A5.6 标量候选" '(3) :int64 '(7 5 7)
            (lambda () (vt-choose (list (%mk '(5) :int64) (%mk '(7) :int64))
                                  (%mk '(1 0 1) :int64))))
    ;; float64 / 混合 dtype 提升
    (expect "A5.7 float64" '(6) :float64 '(1.5d0 2.5d0 3.5d0 4.5d0 5.5d0 6.5d0)
            (lambda () (vt-choose (list f0 f0) (vt-zeros '(6) :dtype :int64))))
    (let ((f32 (vt-astype f0 :float32)))
      (expect "A5.8 f64×f32 → f64" '(6) :float64
              '(1.5d0 2.5d0 3.5d0 4.5d0 5.5d0 6.5d0)
              (lambda () (vt-choose (list f0 f32) (vt-zeros '(6) :dtype :int64))))
      (expect "A5.9 i64×f32 → f64" '(6) :float64
              '(0.0d0 0.0d0 0.0d0 0.0d0 0.0d0 0.0d0)
              (lambda () (vt-choose (list (vt-zeros '(6) :dtype :int64) f32)
                                    (vt-zeros '(6) :dtype :int64)))))
    ;; 布尔候选（结果 dtype 仍是 :bool）
    (let ((b0 (%mk '(1 0 1 0 1 0) :bool)))
      (expect "A5.10 布尔候选" '(6) :bool '(t nil t nil t nil)
              (lambda () (vt-choose (list b0 b0) (vt-zeros '(6) :dtype :int64)))))
    ;; 复数候选（三个候选，i0 里的 2 才不会越界；三个都指向同一张量，
    ;; 结果必须原样等于它 —— 这一条只验「复数候选能走通路且不丢虚部」）
    (let ((cx (%mk (list #C(1.0d0 2.0d0) #C(3.0d0 4.0d0) #C(5.0d0 6.0d0)
                         #C(0.0d0 0.0d0) #C(0.0d0 0.0d0) #C(0.0d0 0.0d0))
                     :complex128)))
      (expect "A5.11 复数候选" '(6) :complex128
              '(#C(1.0d0 2.0d0) #C(3.0d0 4.0d0) #C(5.0d0 6.0d0)
                #C(0.0d0 0.0d0) #C(0.0d0 0.0d0) #C(0.0d0 0.0d0))
              (lambda () (vt-choose (list cx cx cx) i0))))
    ;; 空索引：候选必须也能广播到 (0)，否则是形状错误而不是「空结果」
    (expect "A5.12 空索引" '(0) :int64 '()
            (lambda () (vt-choose (list (vt-zeros '(0) :dtype :int64)
                                        (vt-zeros '(0) :dtype :int64))
                                  (vt-zeros '(0) :dtype :int64))))
    ;; 浮点索引必须报错
    (expect-error "A5.13 浮点索引报错"
                  (lambda () (vt-choose (list c0 c1) (%mk '(0.5 1.5) :float64))))
    ;; 2D + 非连续候选（偏移视图）：off=(1 2 9 3)、off2=(2 9 3 4)，
    ;; idx=(0 1 0 1) 逐元素配对 → 1, 9, 9, 4
    (let* ((src (vt-from-sequence '(9 1 2 9 3 4 9) :dtype :int64))
           (off (vt-slice src '(1 5)))
           (off2 (vt-slice src '(2 6))))
      (expect "A5.14 偏移候选" '(4) :int64 '(1 9 9 4)
              (lambda () (vt-choose (list off off2) (%mk '(0 1 0 1) :int64)))))))

;;; ==================================================================
;;; A6 — vt-corrcoef
;;; ==================================================================
(defun ref-corrcoef (m rowvar)
  "由协方差矩阵按定义归一化：R[i][j] = C[i][j] / sqrt(C[i][i]*C[j][j])，
   对角线遇到零方差时给 1（非对角给 0）。"
  (let* ((a (if rowvar m (vt-transpose m)))
         (c (vt-cov a))
         (n (first (vt-shape c)))
         (out '()))
    (dotimes (i n (nreverse out))
      (let ((row '()))
        (dotimes (j n)
          (let ((cii (vt-ref c i i)) (cjj (vt-ref c j j)))
            (push (if (or (zerop cii) (zerop cjj))
                      (if (= i j) 1.0d0 0.0d0)
                      (/ (vt-ref c i j) (sqrt (* cii cjj))))
                  row)))
        (push (nreverse row) out)))))

(defun t-corrcoef ()
  (format t "~%--- A6 vt-corrcoef ---~%")
  (let ((m (vt-reshape (%mk (loop for i below 24 collect (coerce (mod (* i 5) 7) 'double-float))
                             :float64)
                       '(3 8))))
    (let ((want (loop for row in (ref-corrcoef m t) append row)))
      (expect "A6.1 3×8 逐位" '(3 3) :float64 want (lambda () (vt-corrcoef m))))
    (let ((want (loop for row in (ref-corrcoef m nil) append row)))
      (expect "A6.3 rowvar=nil" '(8 8) :float64 want
              (lambda () (vt-corrcoef m :rowvar nil)))))
  ;; 对角线恒 1（方差非零时）
  (let* ((m (vt-reshape (%mk '(1.0d0 2.0d0 3.0d0 4.0d0 2.0d0 0.0d0) :float64) '(2 3)))
         (r (vt-corrcoef m)))
    (expect-val "A6.4 对角为 1" 1.0d0 (lambda () (vt-ref r 0 0)))
    (expect-val "A6.5 对角为 1（第二行）" 1.0d0 (lambda () (vt-ref r 1 1))))
  ;; 零方差 → 对角线 1、非对角 0
  (let* ((m (vt-reshape (%mk '(2.0d0 2.0d0 2.0d0 5.0d0 5.0d0 5.0d0) :float64) '(2 3)))
         (r (vt-corrcoef m)))
    (expect "A6.6 零方差" '(2 2) :float64 '(1.0d0 0.0d0 0.0d0 1.0d0)
            (lambda () r)))
  ;; 单变量 → 标量 nan（0/0）
  (let ((v (vt-from-array (vector 5.0d0) :dtype :float64)))
    (expect-val "A6.7 单变量 → NaN" t
                (lambda () (nan-p (vt-item (vt-corrcoef v)))))))

;;; ==================================================================
;;; A7 — vt-nonzero / vt-argwhere / vt-flatnonzero / vt-extract
;;; ==================================================================
(defun t-nonzero-family ()
  (format t "~%--- A7 vt-nonzero / vt-argwhere / vt-flatnonzero / vt-extract ---~%")
  (dolist (pair (sample-vectors))
    (let ((name (car pair)) (v (cdr pair)))
      (let* ((coords (ref-coords v))
             (rank (length (vt-shape v)))
             (cnt (length coords)))
        ;; argwhere：(cnt, rank)
        (expect (format nil "A7.1 ~a argwhere" name)
                (list cnt rank) :int64
                (coords->argwhere-values coords rank)
                (lambda () (vt-argwhere v)))
        ;; nonzero：rank 个 (cnt) 张量
        (let ((got (vt-nonzero v)))
          (expect (format nil "A7.2 ~a nonzero 层数" name) '(1) :int64
                  (list rank) (lambda () (vt-from-sequence (list (length got)) :dtype :int64)))
          (dotimes (d rank)
            (expect (format nil "A7.3 ~a nonzero 层 ~a" name d) (list cnt) :int64
                    (coords->nonzero-layer coords d)
                    (lambda () (nth d got)))))
        ;; flatnonzero：逻辑扁平的线性下标
        (expect (format nil "A7.4 ~a flatnonzero" name) (list cnt) :int64
                (mapcar (lambda (c)
                          (loop for x in c for s in (ref-strides (vt-shape v))
                                sum (* x s)))
                        coords)
                (lambda () (vt-flatnonzero v)))
        ;; extract：按 C 序取出非零位置的值
        (expect (format nil "A7.5 ~a extract" name) (list cnt) (vt-dtype v)
                (mapcar (lambda (c) (apply #'vt-ref v c)) coords)
                (lambda () (vt-extract (vt-nonzero-p v) v))))))
  ;; 多维形状
  (dolist (shp '((2 3) (3 2) (2 2 2) (1 5) (5 1) (2 3 4) (1 1) (2 1 3)))
    (let* ((n (reduce #'* shp))
           (v (vt-reshape (%mk (loop for i below n
                                     collect (coerce (- (mod i 4) 1) 'double-float))
                                :float64)
                          shp))
           (coords (ref-coords v))
           (rank (length shp))
           (cnt (length coords)))
      (expect (format nil "A7.6 ~a argwhere" shp) (list cnt rank) :int64
              (coords->argwhere-values coords rank)
              (lambda () (vt-argwhere v)))
      (let ((got (vt-nonzero v)))
        (dotimes (d rank)
          (expect (format nil "A7.7 ~a nonzero 层 ~a" shp d) (list cnt) :int64
                  (coords->nonzero-layer coords d)
                  (lambda () (nth d got)))))
      (expect (format nil "A7.8 ~a flatnonzero" shp) (list cnt) :int64
              (mapcar (lambda (c) (loop for x in c for s in (ref-strides shp)
                                        sum (* x s)))
                      coords)
              (lambda () (vt-flatnonzero v)))))
  ;; 视图（转置 / 步长 / 偏移）—— 里程计实现只在下标升序上成立，必须验证
  (let* ((a (vt-reshape (%mk (loop for i below 24 collect (coerce (mod i 3) 'double-float))
                                 :float64)
                        '(2 3 4))))
    (dolist (spec (list (cons "转置(2,0,1)" (vt-transpose a '(2 0 1)))
                        (cons "步长(0 2)(0 3 2)(0 4)" (vt-slice a '(0 2) '(0 3 2) '(0 4)))
                        (cons "偏移(1 2)(1 3)(1 4)" (vt-slice a '(1 2) '(1 3) '(1 4)))
                        (cons "翻转" (vt-flip a :axis 2))))
      (let* ((label (car spec)) (v (cdr spec))
             (coords (ref-coords v))
             (rank (length (vt-shape v)))
             (cnt (length coords)))
        (expect (format nil "A7.9 ~a argwhere" label) (list cnt rank) :int64
                (coords->argwhere-values coords rank)
                (lambda () (vt-argwhere v)))
        (let ((got (vt-nonzero v)))
          (dotimes (d rank)
            (expect (format nil "A7.10 ~a nonzero 层 ~a" label d) (list cnt) :int64
                    (coords->nonzero-layer coords d)
                    (lambda () (nth d got)))))
        (expect (format nil "A7.11 ~a extract" label) (list cnt) :float64
                (mapcar (lambda (c) (apply #'vt-ref v c)) coords)
                (lambda () (vt-extract (vt-nonzero-p v) v))))))
  ;; 边界：全零 / 空 / 单元素
  (expect "A7.12 全零 argwhere" '(0 2) :int64 '()
          (lambda () (vt-argwhere (vt-zeros '(2 3) :dtype :int64))))
  (let ((nz (vt-nonzero (vt-zeros '(2 3) :dtype :int64))))
    (expect "A7.13 全零 nonzero 层数" '(1) :int64 '(2)
            (lambda () (vt-from-sequence (list (length nz)) :dtype :int64)))
    (expect "A7.14 全零 nonzero 层 0" '(0) :int64 '() (lambda () (nth 0 nz)))
    (expect "A7.15 全零 nonzero 层 1" '(0) :int64 '() (lambda () (nth 1 nz))))
  (expect "A7.16 空张量 argwhere" '(0 1) :int64 '()
          (lambda () (vt-argwhere (vt-zeros '(0) :dtype :int64))))
  (expect "A7.17 空张量 flatnonzero" '(0) :int64 '()
          (lambda () (vt-flatnonzero (vt-zeros '(0) :dtype :int64))))
  (expect "A7.18 空张量 extract" '(0) :int64 '()
          (lambda () (vt-extract (vt-zeros '(0) :dtype :int64)
                                 (vt-zeros '(0) :dtype :int64))))
  ;; argwhere 的 :dtype
  (expect "A7.19 argwhere :int32" '(2 1) :int32 '(1 3)
          (lambda () (vt-argwhere (%mk '(0 5 0 6) :int64) :dtype :int32)))
  ;; extract 的广播 / 裁剪分支
  (expect "A7.20 extract 条件比数据长 → 裁剪到数据长度" '(2) :int64 '(5 6)
          (lambda () (vt-extract (%mk '(1 0 1) :int64) (%mk '(5 6) :int64))))
  (expect "A7.21 extract 条件比数据短" '(2) :int64 '(5 6)
          (lambda () (vt-extract (%mk '(1 1) :int64) (%mk '(5 6 7) :int64))))
  (expect-error "A7.22 extract 空数据含真值报错"
                (lambda () (vt-extract (%mk '(1) :int64) (vt-zeros '(0) :dtype :int64))))
  ;; count-nonzero
  (dolist (pair (sample-vectors))
    (let ((name (car pair)) (v (cdr pair)))
      (expect-val (format nil "A7.23 ~a count-nonzero" name)
                  (length (ref-coords v))
                  (lambda () (vt-item (vt-count-nonzero v)))))))

;;; ==================================================================
;;; A8 — vt-trim-zeros
;;; ==================================================================
(defun ref-trim (vals trim)
  "按定义：从两端按「等于 0」收缩（浮点上 -0.0 也是 0；复数按实部与虚部都为 0）。"
  (let* ((z (lambda (x) (zerop x)))
         (n (length vals))
         (start 0) (end n))
    (when (member trim '(:fb :f))
      (loop while (and (< start end) (funcall z (nth start vals))) do (incf start)))
    (when (member trim '(:fb :b))
      (loop while (and (< start end) (funcall z (nth (1- end) vals))) do (decf end)))
    (subseq vals start end)))

(defun t-trim-zeros ()
  (format t "~%--- A8 vt-trim-zeros ---~%")
  (dolist (case (list (list '(0 0 1 2 0 3 0 0) :fb)
                      (list '(0 0 1 2 0 3 0 0) :f)
                      (list '(0 0 1 2 0 3 0 0) :b)
                      (list '(0 0 0 0) :fb)
                      (list '(0 0 0 0) :f)
                      (list '(0 0 0 0) :b)
                      (list '(1 2 3) :fb)
                      (list '() :fb)
                      (list '(0 -0.0 0.0 0) :fb)))
    (destructuring-bind (vals trim) case
      (let ((v (%mk vals :float64)))
        (expect (format nil "A8.1 ~a ~a" vals trim)
                (list (length (ref-trim vals trim))) :float64
                ;; 张量是 :float64，期望值也必须落成 double —— `EQV` 用 EQL
                ;; 比浮点，整数 1 与 1.0d0 不算相等（这正是我们要的严格性）
                (mapcar (lambda (x) (coerce x 'double-float)) (ref-trim vals trim))
                (lambda () (vt-trim-zeros v :trim trim))))))
  ;; 非 float64 dtype 保持
  (expect "A8.2 int64 保持 dtype" '(2) :int64 '(7 8)
          (lambda () (vt-trim-zeros (vt-from-sequence '(0 0 7 8 0) :dtype :int64))))
  (expect "A8.3 int8" '(1) :int8 '(5)
          (lambda () (vt-trim-zeros (vt-from-sequence '(0 5 0) :dtype :int8))))
  ;; 复数按**实部**判零（与改动前的实现一致：原实现先 ASTYPE 到 float64 再判，
  ;; 复数落到 float64 只剩实部）。所以 #C(0 1) 算是零，会被整条丢掉。
  (expect "A8.4 复数按实部判零" '(0) :complex128 '()
          (lambda () (vt-trim-zeros
                      (%mk (list #C(0.0d0 1.0d0) #C(0.0d0 0.0d0)) :complex128))))
  ;; 偏移视图：slice(1 4) = (0 1 0) → 首尾各去一步 → (1)
  (expect "A8.5 偏移视图" '(1) :int64 '(1)
          (lambda () (vt-trim-zeros
                      (vt-slice (vt-from-sequence '(9 0 1 0 9) :dtype :int64) '(1 4)))))
  ;; 0 维
  (expect "A8.6 0 维零" '(0) :int64 '()
          (lambda () (vt-trim-zeros (vt-reshape (%mk '(0) :int64) '()))))
  (expect "A8.7 非零 0 维" '(1) :int64 '(3)
          (lambda () (vt-trim-zeros (vt-reshape (%mk '(3) :int64) '()))))
  (expect-error "A8.8 非法 trim 报错"
                (lambda () (vt-trim-zeros (%mk '(1 0) :float64) :trim :x))))

;;; ==================================================================
;;; B — 机器无关倍率护栏（旧算法 ÷ 优化版）
;;; ==================================================================
(defun %calibrate (thunk)
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
  (incf *n*)
  (if (and (numberp ratio) (>= ratio limit))
      (progn (incf *pass*)
             (format t "  [ OK ] ~a（~a = ~,2f× ≥ ~,1f×）~%" label what (float ratio) limit))
      (progn (incf *fail*) (push label *failures*)
             (format t "  ❌ ~a（~a = ~,2f× < ~,1f×）~%" label what (float ratio) limit))))

;;; --- 旧算法（逐字复刻改动前的实现） ---
(defun old-positive-p (v)
  "改动前：匿名 lambda → vt-map 的闭包路径（每元素 funcall + 两次泛型 AREF）。"
  (vt-map (lambda (x) (if (> x 0.0d0) 1 0)) v :dtype :bool))

(defun old-nonzero-p (v)
  (vt-map (lambda (x) (if (zerop x) 0 1)) v :dtype :bool))

(defun old-signbit (v)
  (vt-map (lambda (x) (if (ref-signbit x) 1 0))
          (clvt::ensure-vt v :dtype :float64) :dtype :bool))

(defun old-to-bool (tensor)
  "改动前：VT-DO-EACH 递归里程计 + 无类型声明的 AREF + 逐元素 %BOOL-FLAG。"
  (let* ((in (clvt::ensure-vt tensor))
         (res (clvt::%alloc-vt (vt-shape in) :bool))
         (od (vt-data res))
         (i 0))
    (declare (type (simple-array (unsigned-byte 8) (*)) od)
             (type fixnum i))
    (vt-do-each (ptr val in)
      (declare (ignore ptr))
      (setf (aref od i) (the (unsigned-byte 8) (clvt::%bool-flag val)))
      (incf i))
    res))

(defun old-argwhere (condition)
  "改动前：递归里程计 + VECTOR-PUSH-EXTEND 逐分量追加。"
  (let* ((shape (vt-shape condition))
         (rank (length shape))
         (data (vt-data condition))
         (offset (vt-offset condition))
         (shape-vec (coerce shape 'simple-vector))
         (strides-vec (coerce (vt-strides condition) 'simple-vector))
         (result (make-array 64 :element-type '(signed-byte 64)
                                :fill-pointer 0 :adjustable t))
         (coord (make-array rank :element-type '(signed-byte 64))))
    (labels ((recurse (depth ptr)
               (if (= depth rank)
                   (when (/= (aref data ptr) 0)
                     (if (zerop rank)
                         (vector-push-extend 0 result)
                         (loop for c across coord do (vector-push-extend c result))))
                   (let ((dim (svref shape-vec depth))
                         (stride (svref strides-vec depth)))
                     (loop for i from 0 below dim do
                       (setf (aref coord depth) i)
                       (recurse (1+ depth) (+ ptr (* i stride))))))))
      (recurse 0 offset))
    (let* ((total (length result))
           (count (if (zerop rank) total (floor total rank))))
      (if (zerop count)
          (vt-zeros (list 0 rank) :dtype :int64)
          (let ((fd (make-array total :element-type '(signed-byte 64))))
            (loop for i from 0 below total for idx across result
                  do (setf (aref fd i) idx))
            (clvt::%make-vt :data fd :shape (list count rank)
                            :strides (list rank 1) :offset 0 :dtype :int64))))))

(defun old-nonzero (condition)
  "改动前：PUSH/NREVERSE 收集扁平下标 + (NTH D ARRAYS) 逐分量回填。"
  (let* ((shape (vt-shape condition))
         (rank (length shape))
         (data (vt-data condition))
         (offset (vt-offset condition))
         (in-strides (vt-strides condition))
         (logic-strides (vt-compute-strides shape))
         (nz '()))
    (labels ((recurse (depth logic-idx ptr)
               (if (= depth rank)
                   (unless (zerop (aref data ptr)) (push logic-idx nz))
                   (let ((dim (nth depth shape))
                         (in-stride (nth depth in-strides))
                         (logic-stride (nth depth logic-strides)))
                     (loop for i from 0 below dim do
                       (recurse (1+ depth)
                                (+ logic-idx (* i logic-stride))
                                (+ ptr (* i in-stride))))))))
      (recurse 0 0 offset))
    (setf nz (nreverse nz))
    (let ((num (length nz)))
      (cond
        ((zerop rank) (list (vt-zeros '(0) :dtype :int64)))
        ((zerop num) (loop repeat rank collect (vt-zeros '(0) :dtype :int64)))
        (t
         (let ((arrays (loop repeat rank
                             collect (make-array num :element-type '(signed-byte 64)))))
           (loop for flat in nz for i from 0 do
             (let ((rem flat))
               (loop for d from 0 below rank
                     for stride in logic-strides do
                 (let ((coord (floor rem stride)))
                   (setf (aref (nth d arrays) i) coord)
                   (decf rem (* coord stride))))))
           (loop for arr in arrays
                 collect (clvt::%make-vt :data arr :shape (list num)
                                         :strides '(1) :offset 0 :dtype :int64))))))))

(defun old-trim-zeros (tensor trim)
  "改动前：整条 COERCE 成 LIST + NTH/SUBSEQ + VT-FROM-SEQUENCE。"
  (let* ((input (clvt::ensure-vt tensor))
         (one-d (if (null (vt-shape input))
                    (vt-reshape input '(1))
                    (vt-ravel input)))
         (n (vt-size one-d)))
    (if (zerop n)
        one-d
        (let ((vals (coerce (vt-data (vt-astype one-d :float64)) 'list)))
          (let ((start 0) (end n))
            (when (member trim '(:fb :f))
              (loop while (and (< start end) (zerop (nth start vals))) do (incf start)))
            (when (member trim '(:fb :b))
              (loop while (and (< start end) (zerop (nth (1- end) vals))) do (decf end)))
            (vt-from-sequence (subseq vals start end) :dtype (vt-dtype one-d)))))))

(defun old-choose (choices indices mode)
  "改动前：每元素 (NTH IDX CHOICES) + VT-DATA/VT-OFFSET 访问器 + VT-CAST。"
  (let* ((n (length choices))
         (idx-vt (clvt::ensure-vt indices))
         (choice-vts (mapcar #'clvt::ensure-vt choices))
         (final-shape (reduce #'vt-broadcast-shapes
                              (cons (vt-shape idx-vt) (mapcar #'vt-shape choice-vts))))
         (idx-b (vt-contiguous (vt-broadcast-to idx-vt final-shape)))
         (flat-choices (mapcar (lambda (c) (vt-contiguous (vt-broadcast-to c final-shape)))
                               choice-vts))
         (final-size (vt-shape-to-size final-shape))
         (idx-data (vt-data idx-b))
         (idx-off (vt-offset idx-b))
         (result-type (apply #'clvt::vt-promote-type (mapcar #'vt-dtype flat-choices)))
         (result-data (make-array final-size
                                  :element-type (clvt::vt-dtype->lisp-type result-type))))
    (loop for i from 0 below final-size
          for raw = (truncate (aref idx-data (+ idx-off i)))
          for idx = (ecase mode
                      (:raise raw)
                      (:wrap  (mod raw n))
                      (:clip  (max 0 (min (1- n) raw))))
          do (when (or (< idx 0) (>= idx n))
               (error "old-choose: 越界"))
             (let* ((cv (nth idx flat-choices))
                    (cd (vt-data cv))
                    (co (vt-offset cv)))
               (setf (aref result-data i)
                     (clvt::vt-cast (aref cd (+ co i)) result-type))))
    (clvt::%make-vt :data result-data :shape final-shape
                    :strides (vt-compute-strides final-shape) :offset 0
                    :dtype result-type)))

;;; corrcoef 的「旧算法」：n×n 逐元素 (%VT-REF-RAW c i i) + (SETF (VT-REF r i j))
(defun old-corrcoef (m)
  (with-float-safe
    (let* ((c (vt-cov m))
           (n (first (vt-shape c))))
      (if (null (vt-shape c))
          (let ((v (vt-item c)))
            (vt-full nil (/ v (sqrt (* v v))) :dtype :float64))
          (let ((r (vt-zeros (list n n) :dtype :float64)))
            (dotimes (i n r)
              (dotimes (j n)
                (let ((cii (clvt::%vt-ref-raw c i i)) (cjj (clvt::%vt-ref-raw c j j)))
                  (setf (vt-ref r i j)
                        (if (or (zerop cii) (zerop cjj))
                            (if (= i j) 1.0d0 0.0d0)
                            (/ (clvt::%vt-ref-raw c i j) (sqrt (* cii cjj)))))))))))))

(defun t-ratios ()
  (format t "~%--- B 机器无关倍率护栏（旧算法 ÷ 优化版）---~%")
  (let* ((n 1000000)
         (f (vt-from-array (make-array n :element-type 'double-float
                                         :initial-contents
                                         (loop for i below n collect (coerce (mod i 97) 'double-float)))
                           :dtype :float64))
         (mask (vt-from-array (make-array n :element-type '(unsigned-byte 8)
                                            :initial-contents
                                            (loop for i below n collect (if (evenp i) 1 0)))
                              :dtype :bool))
         (g2 (vt-reshape (vt-from-array
                          (make-array 1000000 :element-type 'double-float
                                              :initial-contents
                                              (loop for i below 1000000
                                                    collect (coerce (mod i 61) 'double-float)))
                          :dtype :float64)
                         '(1000 1000))))
    ;; B1 positive-p：闭包 vs 具名内联
    (let ((fast (min-us (lambda () (vt-positive-p f))))
          (slow (min-us (lambda () (old-positive-p f)) :reps 3)))
      (check-at-least "B1 positive-p 1M" (/ slow fast) 2.0 "旧/新"))
    ;; B2 nonzero-p：闭包 vs 具名内联
    (let ((fast (min-us (lambda () (vt-nonzero-p f))))
          (slow (min-us (lambda () (old-nonzero-p f)) :reps 3)))
      (check-at-least "B2 nonzero-p 1M" (/ slow fast) 2.0 "旧/新"))
    ;; B3 nonzero-p(:bool 输入)：掩码表 ub8 一格
    (let ((fast (min-us (lambda () (vt-nonzero-p mask))))
          (slow (min-us (lambda () (old-nonzero-p mask)) :reps 3)))
      (check-at-least "B3 nonzero-p bool 1M" (/ slow fast) 1.5 "旧/新"))
    ;; B4 signbit
    (let ((fast (min-us (lambda () (vt-signbit f))))
          (slow (min-us (lambda () (old-signbit f)) :reps 3)))
      (check-at-least "B4 signbit 1M" (/ slow fast) 2.0 "旧/新"))
    ;; B5 astype → :bool
    (let ((fast (min-us (lambda () (vt-astype f :bool))))
          (slow (min-us (lambda () (old-to-bool f)) :reps 3)))
      (check-at-least "B5 astype->bool 1M f64" (/ slow fast) 2.0 "旧/新"))
    (let ((fast (min-us (lambda () (vt-astype mask :bool))))
          (slow (min-us (lambda () (old-to-bool mask)) :reps 3)))
      (check-at-least "B6 astype->bool 1M bool" (/ slow fast) 2.0 "旧/新"))
    ;; B7 argwhere
    (let ((fast (min-us (lambda () (vt-argwhere mask))))
          (slow (min-us (lambda () (old-argwhere mask)) :reps 3)))
      (check-at-least "B7 argwhere 1M" (/ slow fast) 2.0 "旧/新"))
    ;; B8 nonzero
    (let ((fast (min-us (lambda () (vt-nonzero f))))
          (slow (min-us (lambda () (old-nonzero f)) :reps 3)))
      (check-at-least "B8 nonzero 1M" (/ slow fast) 2.0 "旧/新"))
    ;; B9 nonzero 2D
    (let ((fast (min-us (lambda () (vt-nonzero g2))))
          (slow (min-us (lambda () (old-nonzero g2)) :reps 3)))
      (check-at-least "B9 nonzero 1000x1000" (/ slow fast) 1.5 "旧/新"))
    ;; B10 trim-zeros（两端各走几步，用带 100 个前导零的输入保证两种实现都真在扫）
    ;; 注意：新实现在连续输入上是 O(两端步数) 且 `VT-CONTIGUOUS` 对已是连续的
    ;; 切片零拷贝，所以 FAST 会落到亚微秒级 —— 读数可能被时钟粒度量化到 0。
    ;; 分母夹一个 1µs 下限，避免除零，也让这条断言不依赖时钟精度。
    (let* ((pad (vt-pad f '((100 100))))
           (fast (max (min-us (lambda () (vt-trim-zeros pad)) :reps 5) 1.0d-3))
           (slow (min-us (lambda () (old-trim-zeros pad :fb)) :reps 2)))
      (check-at-least "B10 trim-zeros 1M（旧实现摊平成 list）" (/ slow fast) 3.0 "旧/新"))
    ;; B11 choose（索引必须落在 [0,3) 内 —— 候选只有三个）
    (let* ((c1 (vt-from-array (make-array n :element-type 'double-float
                                             :initial-contents
                                             (loop for i below n collect (coerce (mod (* i 3) 89) 'double-float)))
                              :dtype :float64))
           (c2 (vt-zeros (list n) :dtype :float64))
           (idx (vt-from-array (make-array n :element-type '(signed-byte 64)
                                             :initial-contents
                                             (loop for i below n collect (mod i 3)))
                               :dtype :int64))
           (fast (min-us (lambda () (vt-choose (list f c1 c2) idx))))
           (slow (min-us (lambda () (old-choose (list f c1 c2) idx :raise)) :reps 3)))
      (check-at-least "B11 choose 1M" (/ slow fast) 3.0 "旧/新"))
    ;; B12 corrcoef
    (let ((fast (min-us (lambda () (vt-corrcoef g2))))
          (slow (min-us (lambda () (old-corrcoef g2)) :reps 2)))
      (check-at-least "B12 corrcoef 1000x1000" (/ slow fast) 1.5 "旧/新"))))

;;; ==================================================================
;;; 主流程
;;; ==================================================================
(defun run ()
  (format t "~%============================================================~%")
  (format t "  谓词族 / 掩码 / 索引收集 优化回归测试（v0.4.21）~%")
  (format t "============================================================~%")
  (t-predicates)
  (t-astype-bool)
  (t-choose)
  (t-corrcoef)
  (t-nonzero-family)
  (t-trim-zeros)
  (t-ratios)
  (format t "~%============================================================~%")
  (format t "  Total: ~a | Pass: ~a | Fail: ~a | Skip: 0~%" *n* *pass* *fail*)
  (format t "============================================================~%")
  (when *failures*
    (format t "~%Failed:~{~%  - ~a~}~%" (reverse *failures*)))
  (zerop *fail*))

(sb-ext:exit :code (if (run) 0 1))
