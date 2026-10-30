;;;; bool-dtype-test.lisp — 布尔 dtype（CONVENTIONS §17）的对照测试
;;;;
;;;; 背景：本库此前**没有**布尔 dtype，所有「真值」结果（比较、逻辑、
;;;;   谓词、any/all、isclose、isin …）都用 :int8 的 0/1 承载。
;;;;   这带来两类隐患：
;;;;     · 语义隐患：`(vt-dtype (vt-> a b))` 是 :int8，用户会把它当整数
;;;;       继续算术（True+True→2、sum 得 int8 溢出），而 numpy 的 bool_
;;;;       在算术里按逻辑运算走、sum 升 int64；
;;;;     · 契约隐患：:int8 掩码与真正的 int8 数据在类型上无法区分，
;;;;       `vt-where` / `vt-compress` 这类吃掩码的函数无法校验入参。
;;;;   本次把 `:bool` 立为第九种逻辑 dtype（物理存储 (unsigned-byte 8)），
;;;;   并把所有「numpy 返回布尔」的函数改为返回它。
;;;;
;;;; 本套件：
;;;;   A. 与 numpy 2.3.5 逐例对照（期望值见 bool-expected.sexp，52 例）：
;;;;      内容测试 / 比较 / 逻辑 / any-all（含 axis、keepdims、空、NaN）/
;;;;      isclose / array_equal / allclose / isin / astype / sum。
;;;;   B. dtype 契约：上述函数的默认结果 dtype 必须是 :bool；
;;;;      显式 :dtype :int8 仍可退回旧的 0/1 整型掩码（向后兼容）。
;;;;   C. 不变量：布尔张量的底层字节只含 0/1（打印、:out、astype 之后亦然）。
;;;;   D. 布尔与数值的互操作：sum/prod 升 int64、算术提升、where/compress 吃掩码。
;;;;   E. 打印：布尔张量印 T/NIL 且头部 dtype 为 BOOL（不与 uint8 混淆）。
;;;;   F. npy 往返：|b1 描述符保存/读回后仍是 :bool。
;;;;   G. 布尔算术：+/* 按逻辑或/与、- 报错、/ 升 float64、// 与 % 落 int8 ——
;;;;      numpy 的 bool_ 不按整数算，否则 True+True=2 会破掉 0/1 不变量。
;;;;   H. 逻辑视图 vs 物理视图（§17.8）：vt-ref / vt-item / 打印给 T/NIL；
;;;;      %vt-ref-raw / vt-data / vt-to-list 默认给 0/1 字节；:bool t 才布尔化。
;;;;      含四个内部解包点（allclose / histogram / extract / bincount）的回归。
;;;;   I. 布尔字面量构造与 dtype 重解释：vt-from-sequence :dtype :bool 认
;;;;      T/NIL 叶子（NIL 不再当空子序列）、0 维标量、vt-view 的
;;;;      bool→uint8/int8、vt-flatten-sequence :bool、vt-put 布尔标量。
;;;;
;;;; 用法：sbcl --dynamic-space-size 4096 --non-interactive \
;;;;            --load test/bool-dtype-test.lisp

(require :asdf)
(defparameter cl-user::*clvt-test-dir*
  (make-pathname :name nil :type nil :defaults *load-truename*))
(pushnew (merge-pathnames "../" cl-user::*clvt-test-dir*) asdf:*central-registry*
         :test #'equal)
(handler-bind ((warning #'muffle-warning)) (asdf:load-system :clvt))

(in-package :clvt)

(defvar *fail* 0)
(defvar *pass* 0)

(defmacro check (name form expected &key test)
  "断言 FORM 与 EXPECTED 相等（TEST 为比较函数，缺省 #'EQUAL）。"
  `(let ((got ,form) (want ,expected))
     (if (funcall (or ,test #'equal) got want)
         (incf *pass*)
         (progn (incf *fail*)
                (format t "~&FAIL ~a~%  got:  ~s~%  want: ~s~%" ,name got want)))))

(defmacro check-true (name form)
  `(if ,form (incf *pass*)
       (progn (incf *fail*) (format t "~&FAIL ~a~%" ,name))))

;;; ==================================================================
;;; 输入：与 gen_bool_expected.py 的 BX / BY 逐位相同
;;; ==================================================================

(defun bool-tensor (x)
  "把 Lisp 布尔包成 0 维 :bool 张量（对标 numpy 的 np.bool_ 标量）。"
  (vt-astype (vt-from-sequence (vector (if x 1 0))) :bool))

(defparameter *bx*
  (vt-from-sequence (list -1.0d0 0.0d0 0.5d0 1.0d0
                          (vt-float-nan) (vt-float-pos-inf) (vt-float-neg-inf)
                          -0.0d0 3.0d0)))
(defparameter *bm* (vt-reshape *bx* '(3 3)))
(defparameter *by*
  (vt-from-sequence (list 2.0d0 -1.0d0 0.0d0 1.0d0 0.0d0
                          -3.0d0 0.0d0 0.5d0 -0.0d0)))
(defparameter *m1* (vt-isfinite *bx*))
(defparameter *m2* (vt-signbit *bx*))

;;; ==================================================================
;;; 对照用例表（名字 → 结果张量 thunk），与期望文件一一对应
;;; ==================================================================

(defparameter *cases*
  (list
   ;; 🔍 数组内容测试
   (cons "isnan-1d"     (lambda () (vt-isnan *bx*)))
   (cons "isinf-1d"     (lambda () (vt-isinf *bx*)))
   (cons "isfinite-1d"  (lambda () (vt-isfinite *bx*)))
   (cons "isposinf-1d"  (lambda () (vt-isposinf *bx*)))
   (cons "isneginf-1d"  (lambda () (vt-isneginf *bx*)))
   (cons "signbit-1d"   (lambda () (vt-signbit *bx*)))
   (cons "isnan-2d"     (lambda () (vt-isnan *bm*)))
   (cons "isfinite-2d"  (lambda () (vt-isfinite *bm*)))
   ;; ⚖️ 比较运算
   (cons "eq-self-1d"   (lambda () (vt-= *bx* *bx*)))
   (cons "ne-self-1d"   (lambda () (vt-/= *bx* *bx*)))
   (cons "gt-zero-1d"   (lambda () (vt-> *bx* 0.0d0)))
   (cons "ge-zero-1d"   (lambda () (vt->= *bx* 0.0d0)))
   (cons "lt-zero-1d"   (lambda () (vt-< *bx* 0.0d0)))
   (cons "le-zero-1d"   (lambda () (vt-<= *bx* 0.0d0)))
   (cons "eq-xy-1d"     (lambda () (vt-= *bx* *by*)))
   (cons "lt-xy-1d"     (lambda () (vt-< *bx* *by*)))
   (cons "gt-xy-1d"     (lambda () (vt-> *bx* *by*)))
   ;; 🔗 逻辑运算
   (cons "logical-and"  (lambda () (vt-logical-and *m1* *m2*)))
   (cons "logical-or"   (lambda () (vt-logical-or *m1* *m2*)))
   (cons "logical-not"  (lambda () (vt-logical-not *m1*)))
   (cons "logical-xor"  (lambda () (vt-logical-xor *m1* *m2*)))
   (cons "logical-and-num" (lambda () (vt-logical-and *bx* *by*)))
   ;; 🧭 any / all
   (cons "any-1d"       (lambda () (vt-any *bx*)))
   (cons "all-1d"       (lambda () (vt-all *bx*)))
   (cons "any-2d"       (lambda () (vt-any *bm*)))
   (cons "all-2d"       (lambda () (vt-all *bm*)))
   (cons "any-axis0"    (lambda () (vt-any *bm* :axis 0)))
   (cons "all-axis0"    (lambda () (vt-all *bm* :axis 0)))
   (cons "any-axis1"    (lambda () (vt-any *bm* :axis 1)))
   (cons "all-axis1"    (lambda () (vt-all *bm* :axis 1)))
   (cons "any-axis0-keep" (lambda () (vt-any *bm* :axis 0 :keepdims t)))
   (cons "all-axis1-keep" (lambda () (vt-all *bm* :axis 1 :keepdims t)))
   (cons "any-empty"    (lambda () (vt-any (vt-from-sequence '(0.0d0 0.0d0 0.0d0)))))
   (cons "all-empty"    (lambda () (vt-all (vt-from-sequence '(0.0d0 0.0d0 0.0d0)))))
   (cons "any-nan"      (lambda () (vt-any (vt-from-sequence (list (vt-float-nan))))))
   (cons "all-nan"      (lambda () (vt-all (vt-from-sequence (list (vt-float-nan))))))
   ;; 🤝 数组关系
   (cons "isclose-same" (lambda () (vt-isclose *bx* *bx*)))
   (cons "isclose-near" (lambda () (vt-isclose (vt-from-sequence '(1.0d0 1.000001d0))
                                               (vt-from-sequence '(1.0d0 1.0d0)))))
   (cons "isclose-far"  (lambda () (vt-isclose (vt-from-sequence '(1.0d0 2.0d0))
                                               (vt-from-sequence '(1.0d0 1.0d0)))))
   (cons "array-equal"      (lambda () (bool-tensor (vt-array-equal *bx* (vt-copy *bx*)))))
   (cons "array-equal-nan"  (lambda () (bool-tensor (vt-array-equal *bx* *bx*))))
   (cons "allclose-true"    (lambda () (bool-tensor (vt-allclose '(1.0d0 2.0d0) '(1.0d0 2.0d0)))))
   (cons "allclose-false"   (lambda () (bool-tensor (vt-allclose '(1.0d0 2.0d0) '(1.0d0 2.5d0)))))
   ;; 📚 集合 / 转型 / 归约
   (cons "isin"          (lambda () (vt-isin *bx* '(0.0d0 1.0d0))))
   (cons "isin-invert"   (lambda () (vt-isin *bx* '(0.0d0 1.0d0) :invert t)))
   (cons "in1d"          (lambda () (vt-in1d *bx* '(2.0d0 3.0d0))))
   (cons "astype-bool"   (lambda () (vt-astype *bx* :bool)))
   (cons "astype-bool-2d" (lambda () (vt-astype *bm* :bool)))
   (cons "sum-bool"      (lambda () (vt-sum (vt-astype *bx* :bool))))
   (cons "sum-bool-axis0" (lambda () (vt-sum (vt-astype *bm* :bool) :axis 0)))
   (cons "count-nonzero" (lambda () (vt-count-nonzero *bx*)))
   ;; 🧪 布尔掩码参与选择
   (cons "any-of-compare" (lambda () (vt-any (vt-> *bx* 0.0d0))))))

;;; ==================================================================
;;; A. 与 numpy 期望值对照
;;; ==================================================================

(defun flat-ints (tensor)
  "把张量展平成整数列表（布尔按 0/1）。"
  (let* ((flat (vt-ravel tensor))
         (n (vt-size flat))
         (dt (vt-dtype flat)))
    (unless (or (eq dt :bool) (member dt '(:int8 :int16 :int32 :int64
                                           :uint8 :uint16)))
      (error "flat-ints: 期望布尔/整数张量，收到 ~a" dt))
    (loop for i below n collect (truncate (%vt-ref-raw flat i)))))

(let* ((path (merge-pathnames "bool-expected.sexp" cl-user::*clvt-test-dir*))
       (expected (with-open-file (s path :direction :input) (read s))))
  (format t "~&== A. 与 numpy 对照（~d 例） ==~%" (length expected))
  (dolist (row expected)
    (destructuring-bind (name shape dtype vals) row
      (let ((thunk (cdr (assoc name *cases* :test #'string=))))
        (if (null thunk)
            (progn (incf *fail*) (format t "~&FAIL ~a: 用例未实现~%" name))
            (let ((got (funcall thunk)))
              (check (format nil "~a/shape" name) (vt-shape got) shape)
              (check (format nil "~a/dtype" name) (vt-dtype got) dtype)
              (check (format nil "~a/values" name) (flat-ints got) vals)))))))

;;; ==================================================================
;;; B. dtype 契约：布尔类函数必须返回 :bool；显式 :int8 仍兼容
;;; ==================================================================

(format t "~&== B. dtype 契约 ==~%")

(check "cmp-dtype/gt"  (vt-dtype (vt-> *bx* 0.0d0)) :bool)
(check "cmp-dtype/eq"  (vt-dtype (vt-= *bx* *bx*)) :bool)
(check "cmp-dtype/le"  (vt-dtype (vt-<= *bx* *bx*)) :bool)
(check "logic-dtype/and" (vt-dtype (vt-logical-and *m1* *m2*)) :bool)
(check "logic-dtype/not" (vt-dtype (vt-logical-not *m1*)) :bool)
(check "pred-dtype/isnan" (vt-dtype (vt-isnan *bx*)) :bool)
(check "pred-dtype/isinf" (vt-dtype (vt-isinf *bx*)) :bool)
(check "pred-dtype/isfinite" (vt-dtype (vt-isfinite *bx*)) :bool)
(check "pred-dtype/signbit" (vt-dtype (vt-signbit *bx*)) :bool)
(check "pred-dtype/isposinf" (vt-dtype (vt-isposinf *bx*)) :bool)
(check "pred-dtype/isneginf" (vt-dtype (vt-isneginf *bx*)) :bool)
(check "pred-dtype/positive-p" (vt-dtype (vt-positive-p *bx*)) :bool)
(check "pred-dtype/nonzero-p" (vt-dtype (vt-nonzero-p *bx*)) :bool)
(check "red-dtype/any" (vt-dtype (vt-any *bx*)) :bool)
(check "red-dtype/all" (vt-dtype (vt-all *bx*)) :bool)
(check "red-dtype/any-axis" (vt-dtype (vt-any *bm* :axis 0)) :bool)
(check "isclose-dtype" (vt-dtype (vt-isclose *bx* *bx*)) :bool)
(check "isin-dtype"    (vt-dtype (vt-isin *bx* '(0.0d0))) :bool)
(check "in1d-dtype"    (vt-dtype (vt-in1d *bx* '(0.0d0))) :bool)

;; 向后兼容：显式 :dtype :int8 仍得到旧的 0/1 整型掩码
(check "compat/int8-gt-dtype" (vt-dtype (vt-> *bx* 0.0d0 :dtype :int8)) :int8)
(check "compat/int8-gt-values" (flat-ints (vt-> *bx* 0.0d0 :dtype :int8))
       '(0 0 1 1 0 1 0 0 1))
(check "compat/int8-logical-dtype"
       (vt-dtype (vt-logical-and *m1* *m2* :dtype :int8)) :int8)

;; 布尔与数值的 dtype 提升（numpy 语义）
(check "promote/bool-bool" (vt-promote-type :bool :bool) :bool)
(check "promote/bool-int8" (vt-promote-type :bool :int8) :int8)
(check "promote/bool-uint8" (vt-promote-type :bool :uint8) :uint8)
(check "promote/bool-float32" (vt-promote-type :bool :float32) :float32)
(check "promote/bool-int64" (vt-promote-type :bool :int64) :int64)

;;; ==================================================================
;;; C. 不变量：布尔张量的底层字节只含 0/1
;;; ==================================================================

(format t "~&== C. 0/1 不变量 ==~%")

(defun bool-bytes-ok-p (tensor)
  "张量表层的每个元素都必须是 0 或 1。"
  (let ((flat (vt-ravel tensor))
        (n (vt-size tensor)))
    (loop for i below n always (member (%vt-ref-raw flat i) '(0 1)))))

(check-true "invariant/isnan"    (bool-bytes-ok-p (vt-isnan *bx*)))
(check-true "invariant/gt"      (bool-bytes-ok-p (vt-> *bx* 0.0d0)))
(check-true "invariant/astype"  (bool-bytes-ok-p (vt-astype *bx* :bool)))
(check-true "invariant/any"     (bool-bytes-ok-p (vt-any *bm* :axis 0)))
(check-true "invariant/logical" (bool-bytes-ok-p (vt-logical-xor *m1* *m2*)))
;; astype 的真值化：0.5 → True（不是截断成 0）、NaN → True、-0.0 → False
(check "astype/0.5"   (%vt-ref-raw (vt-astype (vt-from-sequence '(0.5d0)) :bool) 0) 1)
(check "astype/nan"   (%vt-ref-raw (vt-astype (vt-from-sequence (list (vt-float-nan))) :bool) 0) 1)
(check "astype/negzero" (%vt-ref-raw (vt-astype (vt-from-sequence '(-0.0d0)) :bool) 0) 0)
(check "astype/zero"  (%vt-ref-raw (vt-astype (vt-from-sequence '(0.0d0)) :bool) 0) 0)

;;; ==================================================================
;;; D. 布尔与数值的互操作
;;; ==================================================================

(format t "~&== D. 互操作 ==~%")

(let ((b (vt-astype *bx* :bool)))
  ;; sum / prod 升 int64（numpy 语义），不留在 :bool
  (check "interop/sum-dtype" (vt-dtype (vt-sum b)) :int64)
  (check "interop/sum-value" (vt-item (vt-sum b)) 7)
  (check "interop/prod-dtype" (vt-dtype (vt-prod b)) :int64)
  ;; 布尔 ⊔ 弱整数标量 → int64（否则 True+1 会写进 :bool 缓冲产生 2）
  (check "interop/plus-int-dtype" (vt-dtype (vt-+ b 1)) :int64)
  ;; 布尔 when/where：掩码必须是 :bool 也能走选择
  (let ((w (vt-where (vt-> *bx* 0.0d0) *bx* (vt-zeros (vt-shape *bx*)))))
    (check "interop/where-shape" (vt-shape w) (vt-shape *bx*))
    (check "interop/where-v0" (vt-ref w 0) 0.0d0)
    (check "interop/where-v1" (vt-ref w 2) 0.5d0))
  ;; 布尔 → 数值：astype 回整数
  (check "interop/astype-int8" (flat-ints (vt-astype (vt-> *bx* 0.0d0) :int8))
         '(0 0 1 1 0 1 0 0 1))
  (check "interop/astype-f64" (vt-ref (vt-astype (vt-> *bx* 0.0d0) :float64) 2) 1.0d0)
  ;; 空布尔张量与尺寸
  (let ((e (vt-zeros '(0) :dtype :bool)))
    (check "interop/empty-itemsize" (vt-itemsize e) 1)
    (check "interop/empty-nbytes" (vt-nbytes e) 0))
  (check "interop/itemsize" (vt-itemsize b) 1)
  (check "interop/nbytes" (vt-nbytes b) 9))

;;; ==================================================================
;;; E. 打印：True/False + 头部 dtype 为 BOOL
;;; ==================================================================

(format t "~&== E. 打印 ==~%")

(let* ((s (with-output-to-string (out) (prin1 (vt-> (vt-from-sequence '(1.0d0 -1.0d0)) 0.0d0) out))))
  ;; 布尔元素印 `T` / `NIL`（Lisp 惯例，§17.8），不再是 numpy 的 True/False
  (check-true "print/T"   (search "T," s))
  (check-true "print/NIL" (search "NIL" s))
  (check-true "print/dtype-tag" (search "BOOL" s))
  (check-true "print/no-True"  (null (search "True" s)))
  (check-true "print/no-False" (null (search "False" s)))
  ;; 不能出现 0/1 这样的整型渲染
  (check-true "print/no-ints" (null (search " 0," s))))

;;; ==================================================================
;;; F. npy 往返：|b1
;;; ==================================================================

(format t "~&== F. npy 往返 ==~%")

(let* ((b (vt-> *bx* 0.0d0))
       (tmp (merge-pathnames "bool-roundtrip.npy" cl-user::*clvt-test-dir*)))
  (vt-save b tmp :as :npy)
  (let ((r (vt-load tmp :as :npy)))
    (check "npy/dtype" (vt-dtype r) :bool)
    (check "npy/shape" (vt-shape r) (vt-shape b))
    (check "npy/values" (flat-ints r) (flat-ints b)))
  (ignore-errors (delete-file tmp)))

;;; ==================================================================
;;; G. 布尔参与算术（CONVENTIONS §17.3）
;;;
;;; numpy 的 bool_ 在算术里不按整数算：True+True 仍为 True（逻辑或），
;;; True-True 直接 TypeError。若照整数算，(vt-+ mask mask) 会把 2 写进
;;; 布尔数组 —— 不变量 B1 破掉，且 (safety 0) 的写回路径不报错。
;;; ==================================================================

(format t "~&== G. 布尔算术 ==~%")

(defmacro check-errors (name form)
  "断言 FORM 必定报错（对标 numpy 在这些位置抛 TypeError）。"
  `(if (handler-case (progn ,form nil) (error () t))
       (incf *pass*)
       (progn (incf *fail*) (format t "~&FAIL ~a（应当报错但没有）~%" ,name))))

(let ((b (vt-astype (vt-from-sequence #(1 1 0) :dtype :int8) :bool))
      (c (vt-astype (vt-from-sequence #(1 1 0) :dtype :int8) :bool)))

  ;; b + b：逻辑或，仍是 :bool（numpy: [True True False]）
  (let ((r (vt-+ b c)))
    (check "arith/add-dtype" (vt-dtype r) :bool)
    (check "arith/add-values" (flat-ints r) '(1 1 0))
    (check-true "arith/add-invariant" (bool-bytes-ok-p r)))
  ;; 三元与一元
  (check "arith/add-3" (flat-ints (vt-+ b c b)) '(1 1 0))
  (check "arith/add-1" (flat-ints (vt-+ b)) '(1 1 0))
  ;; b * b：逻辑与（逐位与整数相乘同值，但必须仍落在 0/1）
  (let ((r (vt-* b c)))
    (check "arith/mul-dtype" (vt-dtype r) :bool)
    (check "arith/mul-values" (flat-ints r) '(1 1 0))
    (check-true "arith/mul-invariant" (bool-bytes-ok-p r)))
  ;; 二元特化入口必须与通用入口一致
  (check "arith/add-entry" (flat-ints (vt-add b c)) '(1 1 0))
  (check "arith/mul-entry" (flat-ints (vt-mul b c)) '(1 1 0))
  ;; :out 路径（折叠只在最后一步写入 out）
  (let ((dst (vt-zeros '(3) :dtype :bool)))
    (vt-+ b c :out dst)
    (check "arith/out" (flat-ints dst) '(1 1 0)))

  ;; 减法/取负：报错（numpy: TypeError）
  (check-errors "arith/sub-errors" (vt-- b c))
  (check-errors "arith/neg-errors" (vt-- b))
  (check-errors "arith/sub-entry-errors" (vt-sub b c))

  ;; 弱标量是逃生门：升到默认数值 dtype（numpy: b+1 → int64 值 2）
  (let ((r (vt-+ b 1)))
    (check "arith/weak-int-dtype" (vt-dtype r) :int64)
    (check "arith/weak-int-values" (flat-ints r) '(2 2 1)))
  (check "arith/weak-float-dtype" (vt-dtype (vt-+ b 1.0d0)) :float64)
  ;; 真除：恒浮点（numpy: [1. nan 0.]，0/0 走 IEEE 而非 CL 的除零错误）
  (let ((r (vt-/ b c)))
    (check "arith/div-dtype" (vt-dtype r) :float64)
    ;; vt-item 对 :bool 现在给 Lisp 布尔（§17.8），不再是 1
    (check "arith/div-0-0-nan" (vt-item (vt-isnan (vt-ref r 2))) t))
  ;; floor_divide / remainder：商与余数不是布尔量 → int8（numpy 同）
  (check "arith/floordiv-dtype" (vt-dtype (vt-div b c)) :int8)
  (check "arith/floordiv-values" (flat-ints (vt-div b c)) '(1 1 0))
  (check "arith/rem-dtype" (vt-dtype (vt-mod b c)) :int8)
  (check "arith/rem-values" (flat-ints (vt-mod b c)) '(0 0 0))
  ;; abs / maximum：保持 :bool（numpy: abs(bool) → bool）
  (check "arith/abs-dtype" (vt-dtype (vt-abs b)) :bool)
  (check "arith/max-dtype" (vt-dtype (vt-maximum b c)) :bool)

  ;; 显式数值 dtype 是向后兼容的逃生门：旧的 0/1 整数掩码行为不变
  (let ((r (vt-+ b c :dtype :int8)))
    (check "arith/explicit-int8-dtype" (vt-dtype r) :int8)
    (check "arith/explicit-int8-values" (flat-ints r) '(2 2 0))))

;;; ==================================================================
;;; H. 逻辑视图 vs 物理视图（§17.8）
;;;    逻辑视图（vt-ref / vt-item / 打印）= Lisp 布尔 T / NIL
;;;    物理视图（%vt-ref-raw / vt-data / vt-to-list 默认）= 0/1 字节
;;; ==================================================================

(format t "~&== H. 逻辑视图 / 物理视图 ==~%")

(let* ((b  (vt-astype (vt-from-sequence #(1 0 1) :dtype :int8) :bool))
       (b4 (vt-astype (vt-from-sequence #(1 1 0 1) :dtype :int8) :bool))
       (m  (vt-reshape b4 '(2 2)))
       (f  (vt-from-sequence '(1.0d0 2.0d0 3.0d0)))
       (zero-t (vt-full nil t :dtype :bool))
       (zero-f (vt-full nil nil :dtype :bool)))
  ;; --- 逻辑视图：vt-item 给 T / NIL ---
  (check "view/item-true"  (vt-item (vt-slice b '(0 1))) t)
  (check "view/item-false" (vt-item (vt-slice b '(1 2))) nil)
  (check "view/item-type"  (type-of (vt-item (vt-slice b '(0 1)))) 'boolean)
  (check "view/item-0d-true"  (vt-item zero-t) t)
  (check "view/item-0d-false" (vt-item zero-f) nil)
  (check "view/item-2d-single"
         (vt-item (vt-reshape (vt-astype (vt-from-sequence #(1) :dtype :int8) :bool) '(1 1)))
         t)
  (check "view/item-multi-throws"
         (handler-case (progn (vt-item b) nil) (error () t)) t)
  ;; 可以直接当 Lisp 布尔用，不必再写 (= 1 ...)
  (check "view/if-item-all" (if (vt-item (vt-all m)) :yes :no) :no)
  (check "view/if-item-any" (if (vt-item (vt-any m)) :yes :no) :yes)
  (check "view/allclose-true"
         (vt-allclose (vt-from-sequence '(1.0d0 2.0d0)) (vt-from-sequence '(1.0d0 2.0d0))) t)
  (check "view/allclose-false"
         (vt-allclose (vt-from-sequence '(1.0d0 2.0d0)) (vt-from-sequence '(1.0d0 9.0d0))) nil)

  ;; --- 逻辑视图：VT-REF 与 VT-ITEM 同款，给 T / NIL ---
  (check "view/ref-true"  (vt-ref b 0) t)
  (check "view/ref-false" (vt-ref b 1) nil)
  (check "view/ref-type"  (type-of (vt-ref b 0)) 'boolean)
  (check "view/ref-2d" (vt-ref m 1 0) nil)
  (check "view/ref-view-slice" (vt-ref (vt-slice m '(:all) '(1)) 0) t)
  (check "view/ref-neg-index" (vt-ref b -1) t)
  ;; 可以直接当条件用
  (check "view/if-ref" (if (vt-ref b 0) :yes :no) :yes)
  ;; setter 接受 T/NIL/数值（%bool-flag 真值化）
  (let ((c (vt-copy b)))
    (setf (vt-ref c 0) nil)
    (check "view/setf-nil" (vt-to-list c) '(0 0 1)))
  (let ((c (vt-copy b)))
    (setf (vt-ref c 1) t)
    (check "view/setf-t" (vt-to-list c) '(1 1 1)))
  (let ((c (vt-copy b)))
    (setf (vt-ref c 0) 0.5d0)
    (check "view/setf-truthy" (vt-to-list c) '(1 0 1)))
  (let ((c (vt-copy b)))
    (setf (vt-ref c 0) -0.0d0)
    (check "view/setf-neg-zero" (vt-to-list c) '(0 0 1)))
  (let ((c (vt-copy b)))
    (setf (vt-ref c 0) (vt-ref c 1))          ; 逻辑值直接对拷
    (check "view/setf-copy-ref" (vt-to-list c) '(0 0 1)))

  ;; --- 物理视图：%VT-REF-RAW / VT-DATA 仍是 0/1 字节 ---
  (check "view/raw-ref-stays-01" (%vt-ref-raw b 0) 1)
  (check "view/raw-ref-numeric-p" (integerp (%vt-ref-raw b 1)) t)
  (check "view/data-bytes" (coerce (vt-data b) 'list) '(1 0 1))

  ;; --- vt-to-list：默认数值，:bool t 才布尔化 ---
  (check "view/tolist-default" (vt-to-list b) '(1 0 1))
  (check "view/tolist-bool"    (vt-to-list b :bool t) '(t nil t))
  (check "view/tolist-2d-bool" (vt-to-list m :bool t) '((t t) (nil t)))
  ;; 非布尔张量加 :bool t 无副作用
  (check "view/tolist-nonbool" (vt-to-list f :bool t) '(1.0d0 2.0d0 3.0d0))
  ;; 往返不变式：默认的 0/1 能重建；T/NIL 视图也能重建（I 组给出完整对照）
  (check "view/roundtrip-default"
         (flat-ints (vt-from-sequence (vt-to-list b) :dtype :bool)) '(1 0 1))

  ;; --- 打印：T / NIL，不再出现 True / False ---
  (let ((s1 (format nil "~a" b))
        (s2 (format nil "~a" m)))
    (check "view/print-has-NIL" (and (search "NIL" s1) t) t)
    (check "view/print-elem-T"  (and (search "T," s1) t) t)
    (check "view/print-no-True" (null (search "True" s1)) t)
    (check "view/print-no-False" (null (search "False" s1)) t)
    (check "view/print-2d-NIL" (and (search "NIL" s2) t) t))
  ;; 0 维：shape 本身就印 NIL，所以只能按「dtype:BOOL <值>」判
  (check "view/print-0d-true"  (and (search "BOOL T" (format nil "~a" zero-t)) t) t)
  (check "view/print-0d-false" (and (search "BOOL NIL" (format nil "~a" zero-f)) t) t)

  ;; --- 内部解包点（改 vt-item 时最先崩的地方）---
  (check "view/compress-bool-mask" (vt-to-list (vt-compress b f)) '(1.0d0 3.0d0))
  (check "view/compress-int8-mask"
         (vt-to-list (vt-compress (vt-from-sequence #(1 0 1) :dtype :int8) f)) '(1.0d0 3.0d0))
  ;; vt-extract 的 0 维 condition 分支：bool 与 int8 掩码必须走同一判据
  ;; （改 vt-item 之前这里写的是 (/= (vt-item condition) 0)，bool 会直接崩）。
  ;; 注意与 numpy 的既有偏差：np.extract(np.array(True), a) 给 [1.]（0 维条件
  ;; 被当成长度 1 的数组），clvt 给整个 a —— 这是 vt-extract 本来的解释，
  ;; 本轮只保证「布尔条件不再崩溃且与 int8 条件一致」。
  (check "view/extract-0d-true"  (vt-to-list (vt-extract zero-t f)) '(1.0d0 2.0d0 3.0d0))
  (check "view/extract-0d-false" (null (vt-to-list (vt-extract zero-f f))) t)
  (check "view/extract-int8-0d-true"
         (vt-to-list (vt-extract (vt-reshape (vt-from-sequence #(1) :dtype :int8) nil) f))
         '(1.0d0 2.0d0 3.0d0))
  (check "view/bincount-bool" (vt-to-list (vt-bincount b)) '(1 2))
  (check "view/histogram-nan-throws"
         (handler-case (progn (vt-histogram (vt-from-sequence (list 1.0d0 (vt-float-nan)))) nil)
           (error () t))
         t))

;; %truthy-p / %item-number：两个「视图转换」原语
(check "view/truthy-0" (%truthy-p 0) nil)
(check "view/truthy-1" (%truthy-p 1) t)
(check "view/truthy-neg-zero" (%truthy-p -0.0d0) nil)
(check "view/truthy-nan" (%truthy-p (vt-float-nan)) t)
(check "view/truthy-nil" (%truthy-p nil) nil)
(check "view/truthy-t" (%truthy-p t) t)
(check "view/item-number-true" (%item-number (vt-full nil t :dtype :bool)) 1)
(check "view/item-number-false" (%item-number (vt-full nil nil :dtype :bool)) 0)

;;; ==================================================================
;;; I. 布尔字面量构造与 dtype 重解释
;;;    这三处原先都停在「bool 就是 0/1 数字」的旧契约上：
;;;      · VT-FROM-SEQUENCE :dtype :bool 把 NIL 当空子序列 → 报「不规则嵌套」
;;;      · VT-VIEW 的重解释白名单只写了复数两对 → bool→uint8 直接拒
;;;      · VT-FLATTEN-SEQUENCE 丢 NIL、VT-PUT 收到 T/NIL 静默空操作
;;; ==================================================================

(format t "~&== I. 布尔构造 / dtype 重解释 ==~%")

;; --- VT-FROM-SEQUENCE :dtype :bool：T/NIL 是叶子 ---
(check "ctor/list-t-nil-t"
       (vt-to-list (vt-from-sequence (list t nil t) :dtype :bool) :bool t) '(t nil t))
(check "ctor/shape-t-nil-t"
       (vt-shape (vt-from-sequence (list t nil t) :dtype :bool)) '(3))
;; NIL 不再被当成空子序列：形状是 (1) 而非 (1 0)
(check "ctor/lone-nil-shape"
       (vt-shape (vt-from-sequence (list nil) :dtype :bool)) '(1))
(check "ctor/lone-nil-values"
       (vt-to-list (vt-from-sequence (list nil) :dtype :bool)) '(0))
;; 显式记录 Lisp 独有的歧义：'(NIL NIL) 是长度 2 的 [NIL NIL]，
;; 不是 numpy 的 (2 0) 空数组 —— 要空数组写 (vt-zeros '(2 0) :dtype :bool)
(check "ctor/two-nil-is-two-false"
       (vt-shape (vt-from-sequence (list nil nil) :dtype :bool)) '(2))
;; 顶层 NIL 仍是空数组（对标 np.array([], dtype=bool)）
(check "ctor/top-nil-shape" (vt-shape (vt-from-sequence nil :dtype :bool)) '(0))
(check "ctor/top-nil-size" (vt-size (vt-from-sequence nil :dtype :bool)) 0)
;; 0 维标量叶子（对标 np.array(True) 形状 ())
(check "ctor/scalar-t-shape" (vt-shape (vt-from-sequence t :dtype :bool)) nil)
(check "ctor/scalar-t" (vt-item (vt-from-sequence t :dtype :bool)) t)
;; 2D 与向量
(check "ctor/2d"
       (vt-to-list (vt-from-sequence (list (list t nil) (list nil t)) :dtype :bool) :bool t)
       '((t nil) (nil t)))
(check "ctor/2d-shape"
       (vt-shape (vt-from-sequence (list (list t nil) (list nil t)) :dtype :bool)) '(2 2))
(check "ctor/vector"
       (vt-to-list (vt-from-sequence (vector t nil) :dtype :bool) :bool t) '(t nil))
;; 数值走真值化（与 VT-CAST 同规则）
(check "ctor/truthy-numbers"
       (vt-to-list (vt-from-sequence (list 2 -1 0 0.5d0) :dtype :bool)) '(1 1 0 1))
;; :fast t 也必须真值化（COERCE 不认 T/NIL）
(check "ctor/fast-true"
       (vt-to-list (vt-from-sequence (list t nil t) :dtype :bool :fast t) :bool t)
       '(t nil t))
;; ensure-vt 复用同一条路径
(check "ctor/ensure-vt"
       (vt-to-list (ensure-vt (list t nil t) :dtype :bool) :bool t) '(t nil t))
;; 不规则嵌套仍然报错
(check "ctor/ragged-throws"
       (handler-case (progn (vt-from-sequence (list (list t nil) (list t)) :dtype :bool) nil)
         (error () t))
       t)
;; 非布尔模式行为不变：NIL 依旧被当成空子序列
(check "ctor/nonbool-nil-still-ragged"
       (handler-case (progn (vt-from-sequence (list 1 nil 3)) nil) (error () t)) t)
;; (vt-to-list b :bool t) → (vt-from-sequence … :dtype :bool) 往返成立
(check "ctor/roundtrip-bool-view"
       (vt-to-list (vt-from-sequence (vt-to-list *m1* :bool t) :dtype :bool) :bool t)
       (vt-to-list *m1* :bool t))

;; --- VT-VIEW：bool → uint8/int8 ---
(let ((b (vt-from-sequence (list 1 0 1) :dtype :bool)))
  (check "view/dtype-uint8" (vt-dtype (vt-view b :dtype :uint8)) :uint8)
  (check "view/dtype-uint8-values" (vt-to-list (vt-view b :dtype :uint8)) '(1 0 1))
  ;; 省略 shape 保持原形状（旧实现当成 0 维，报「元素总数不一致 旧 3 新 1」）
  (check "view/omit-shape-keeps" (vt-shape (vt-view b :dtype :uint8)) '(3))
  (check "view/omit-shape-same-dtype" (vt-shape (vt-view b :dtype :bool)) '(3))
  (check "view/dtype-int8-values" (vt-to-list (vt-view b :dtype :int8)) '(1 0 1))
  (check "view/dtype-with-shape"
         (vt-shape (vt-view b '(3 1) :dtype :uint8)) '(3 1))
  ;; 反向不放行：uint8 → bool 会带进 2、255 这类非法布尔，破坏 0/1 不变量
  (check "view/uint8-to-bool-rejected"
         (handler-case
             (progn (vt-view (vt-from-sequence (list 1 0) :dtype :uint8) :dtype :bool) nil)
           (error () t))
         t)
  (check "view/float-to-bool-rejected"
         (handler-case
             (progn (vt-view (vt-from-sequence (list 1.0d0)) :dtype :bool) nil)
           (error () t))
         t)
  ;; 需要反向时用 VT-ASTYPE（真值化，语义明确）
  (check "view/astype-is-the-way"
         (vt-to-list (vt-astype (vt-from-sequence (list 1 2) :dtype :uint8) :bool)) '(1 1)))

;; --- VT-FLATTEN-SEQUENCE / VT-PUT：布尔标量与值列表 ---
(check "flat/default-drops-nil" (vt-flatten-sequence (list t nil t)) '(t t))
(check "flat/bool-keeps-nil" (vt-flatten-sequence (list t nil t) :bool t) '(t nil t))
(check "flat/bool-nested"
       (vt-flatten-sequence (list (list t nil) (list nil t)) :bool t) '(t nil nil t))
(let ((c (vt-from-sequence (list 1 1 1 1) :dtype :bool)))
  (vt-put c '(0 1 2) (list t nil t))
  (check "put/bool-list" (vt-to-list c) '(1 0 1 1)))
(let ((c (vt-from-sequence (list 1 1 1) :dtype :bool)))
  (vt-put c 1 nil)
  (check "put/scalar-nil" (vt-to-list c) '(1 0 1)))
(let ((c (vt-from-sequence (list 0 0 0) :dtype :bool)))
  (vt-put c 1 t)
  (check "put/scalar-t" (vt-to-list c) '(0 1 0)))

;;; ==================================================================
;;; J. 审计修复回归（§17.10）：用户函数写 bool、按位取反、直方图、
;;;    布尔差分/积分、布尔缩并（matmul/dot/outer/inner/tensordot/einsum）
;;; ==================================================================

(format t "~&== J. 审计修复回归 (§17.10) ==~%")

(defparameter *jb* (vt-from-sequence '(1 1 0 0) :dtype :bool)
  "J 组的布尔样本：两个真、两个假。")

;; --- J1. VT-MAP 写 :bool：必须写回之前真值化，且不能截断 ---
(check "map/int-2"   (flat-ints (vt-map (lambda (x) 2) *jb*)) '(1 1 1 1))
(check "map/float-half" (flat-ints (vt-map (lambda (x) 0.5d0) *jb*)) '(1 1 1 1))
(check "map/float-zero" (flat-ints (vt-map (lambda (x) 0.0d0) *jb*)) '(0 0 0 0))
(check "map/nil"     (flat-ints (vt-map (lambda (x) nil) *jb*)) '(0 0 0 0))
(check "map/t"       (flat-ints (vt-map (lambda (x) t) *jb*)) '(1 1 1 1))
(check "map/dtype"   (vt-dtype (vt-map (lambda (x) 7) *jb*)) :bool)
(check-true "map/invariant" (bool-bytes-ok-p (vt-map (lambda (x) 7) *jb*)))
(check "map/explicit-bool" (flat-ints (vt-map (lambda (x) 5) *jb* :dtype :bool)) '(1 1 1 1))
;; 写入非连续 out（转置视图）时也只碰逻辑位置
(let ((d (vt-zeros '(2 2) :dtype :bool)))
  (vt-map (lambda (x) 1) (vt-reshape *jb* '(2 2)) :out (vt-transpose d))
  (check "map/out-view" (vt-to-list d) '((1 1) (1 1))))
;; 数值 dtype 不受影响（不能把 uint8 的 200 也归一成 1）
(check "map/uint8-200" (flat-ints (vt-map (lambda (x) 200) (vt-from-sequence '(1 0) :dtype :uint8)))
       '(200 200))

;; --- J2. VT-BIT-NOT：布尔走逻辑非，不是字节取反 254/255 ---
(check "bitnot/bool" (flat-ints (vt-bit-not *jb*)) '(0 0 1 1))
(check "bitnot/bool-dtype" (vt-dtype (vt-bit-not *jb*)) :bool)
(check-true "bitnot/invariant" (bool-bytes-ok-p (vt-bit-not *jb*)))
(check "bitnot/bool-vs-logical"
       (flat-ints (vt-bit-not *jb*)) (flat-ints (vt-logical-not *jb*)))
;; 数值逃生门保持字节语义
(check "bitnot/int8" (flat-ints (vt-bit-not (vt-from-sequence '(1 0) :dtype :int8))) '(-2 -1))
(check "bitnot/bool->int8" (flat-ints (vt-bit-not *jb* :dtype :int8)) '(-2 -2 -1 -1))

;; --- J3. VT-HISTOGRAM：布尔不再因 T/NIL 进算术而报错 ---
(let ((h (vt-histogram *jb*)))
  (check "hist/counts" (flat-ints h) '(2 0 0 0 0 0 0 0 0 2))
  (check "hist/dtype" (vt-dtype h) :int64))
(let ((h (vt-histogram *jb* :bins 2 :range '(0 2))))
  (check "hist/range" (flat-ints h) '(2 2)))

;; --- J4. VT-DIFF：布尔差分 ≡ 逐次异或（numpy 的 bool 减法回绕） ---
(check "diff/bool" (flat-ints (vt-diff *jb*)) '(0 1 0))
(check "diff/bool-n2" (flat-ints (vt-diff *jb* :n 2)) '(1 1))
(check "diff/bool-dtype" (vt-dtype (vt-diff *jb*)) :bool)
(check "diff/float-unchanged"
       (vt-to-list (vt-diff (vt-from-sequence '(1 0 3)))) '(-1.0d0 3.0d0))

;; --- J5. VT-TRAPZ：整数/布尔按 float64 累加，不再截断成 0 ---
(check "trapz/bool" (vt-item (vt-trapz *jb*)) 1.5d0 :test #'=)
(check "trapz/bool-dtype" (vt-dtype (vt-trapz *jb*)) :float64)
(check "trapz/int8" (vt-item (vt-trapz (vt-from-sequence '(1 0 1) :dtype :int8))) 1.0d0 :test #'=)
(check "trapz/float-unchanged" (vt-item (vt-trapz (vt-from-sequence '(1.0 0.0 1.0)))) 1.0d0 :test #'=)

;; --- J6. 布尔缩并：numpy 按「与/或」缩并，结果仍是 :bool ---
(defparameter *bm4* (vt-reshape (vt-from-sequence '(1 0 1 1) :dtype :bool) '(2 2)))
(check "mm/dtype" (vt-dtype (vt-matmul *bm4* *bm4*)) :bool)
(check "mm/values" (flat-ints (vt-matmul *bm4* *bm4*)) '(1 0 1 1))
(check-true "mm/invariant" (bool-bytes-ok-p (vt-matmul *bm4* *bm4*)))
(check "mm/dot-same" (flat-ints (vt-dot *bm4* *bm4*)) '(1 0 1 1))
(check "mm/outer" (flat-ints (vt-outer (vt-from-sequence '(1 0) :dtype :bool)
                                       (vt-from-sequence '(1 1) :dtype :bool)))
       '(1 1 0 0))
(check "mm/inner" (flat-ints (vt-inner *bm4* *bm4*)) '(1 1 1 1))
(check "mm/tensordot" (vt-item (vt-tensordot *jb* (vt-from-sequence '(1 0 1 1) :dtype :bool)
                                             :axes 1))
       t)
(check "mm/einsum"
       (flat-ints (vt-einsum "ij,jk->ik" *bm4* *bm4*)) '(1 0 1 1))
;; 逻辑缩并而不是数值和：[[T,F],[T,T]] @ 自身 不会出现 2
(check-true "mm/no-numeric-sum"
            (notany (lambda (v) (> v 1)) (flat-ints (vt-matmul *bm4* *bm4*))))
;; 200 个 1 的内积：int32 计数累加 → 真值，不溢出不误报
(check "mm/long-dot" (vt-item (vt-matmul (vt-ones '(200) :dtype :bool)
                                         (vt-ones '(200) :dtype :bool)))
       t)
(check "mm/zero-dot" (vt-item (vt-matmul (vt-zeros '(3) :dtype :bool)
                                         (vt-ones '(3) :dtype :bool)))
       nil)
;; 逃生门：显式 :dtype 仍走数值路径
(check "mm/explicit-f64" (vt-dtype (vt-einsum "i,i->" *jb* *jb* :dtype :float64)) :float64)
(check "mm/mixed-promotes"
       (vt-dtype (vt-dot *jb* (vt-from-sequence '(1.0 0.0 1.0 1.0)))) :float64)
;; :out 为 :bool 时按真值写回
(let ((o (vt-zeros '(2 2) :dtype :bool)))
  (vt-matmul *bm4* *bm4* :out o)
  (check "mm/out-bool" (vt-to-list o) '((1 0) (1 1))))
;; trace 仍走数值（numpy 的 np.trace(bool) 也是 int64）
(check "mm/trace-int64" (vt-dtype (vt-trace *bm4*)) :int64)
(check "mm/trace-value" (vt-item (vt-trace *bm4*)) 2)

;; --- J7. 「值位置」上的 T/NIL：传得进去，且不误伤空表语义 (§17.10.8) ---
;; ensure-vt：T 是值 1（缺省 :bool），NIL 只有显式 :bool 才是 0
(check "ensure/t"      (list (vt-dtype (ensure-vt t)) (vt-item (ensure-vt t))) '(:bool t))
(check "ensure/t-i64"  (list (vt-dtype (ensure-vt t :dtype :int64))
                             (vt-item (ensure-vt t :dtype :int64))) '(:int64 1))
(check "ensure/nil-empty" (vt-shape (ensure-vt nil)) '(0))
(check "ensure/nil-bool"  (vt-item (ensure-vt nil :dtype :bool)) nil)
;; vt-insert：values 为 T/NIL 时按布尔标量插入
(check "insert/t"   (vt-to-list (vt-insert *jb* 0 t)) '(1 1 1 0 0))
(check "insert/nil" (vt-to-list (vt-insert *jb* 0 nil)) '(0 1 1 0 0))
(check "insert/axis" (vt-to-list (vt-insert *bm4* 0 t :axis 1))
       '((1 1 0) (1 1 1)))
;; vt-pad：constant-values 的 T/NIL（标量 / 一对 / 逐维）
(check "pad/t"   (vt-to-list (vt-pad *jb* '(1 1) :constant-values t))
       '(1 1 1 0 0 1))
(check "pad/nil" (vt-to-list (vt-pad *jb* '(1 1) :constant-values nil))
       '(0 1 1 0 0 0))
(check "pad/pair" (vt-to-list (vt-pad *jb* '(1 1) :constant-values '(nil t)))
       '(0 1 1 0 0 1))
;; vt-where：三元选择的两个分支都是值位置，(vt-where b t nil) ≡ np.where(b,T,F)
(check "where/tnil" (list (vt-dtype (vt-where *jb* t nil))
                          (vt-to-list (vt-where *jb* t nil))) '(:bool (1 1 0 0)))
(check "where/nilt" (vt-to-list (vt-where *jb* nil t)) '(0 0 1 1))
(check "where/t-int" (vt-dtype (vt-where *jb* t 0)) :int64)
;; vt-append / vt-fill-diagonal / vt-fill：T/NIL 当元素值
(check "append/t" (vt-to-list (vt-append *jb* t)) '(1 1 0 0 1))
(let ((m (vt-zeros '(2 2) :dtype :bool)))
  (vt-fill-diagonal m t)
  (check "fill-diagonal/t" (vt-to-list m) '((1 0) (0 1))))
(check "fill/nil" (let ((c (vt-copy *jb*))) (vt-fill c nil) (vt-to-list c))
       '(0 0 0 0))
;; vt-map 的 6/7 输入走 &REST 分支（ECASE 的 T 子句不是默认子句）
(check "map/six-inputs" (flat-ints (vt-map (lambda (a b c d e f) 2)
                                           *jb* *jb* *jb* *jb* *jb* *jb*))
       '(1 1 1 1))
(check "map/seven-inputs" (flat-ints (vt-map (lambda (a b c d e f g) 3)
                                             *jb* *jb* *jb* *jb* *jb* *jb* *jb*))
       '(1 1 1 1))

;; --- K. 结果 dtype 的提升表 & 布尔缩并的漏网（§17.10.10 / §17.10.11） ---
;; 「值算对了但 dtype 错」和「dtype 对了但值是脏字节」是两类不同的失败，
;; 本组两条都盯：dtype 断言 + 物理字节 / 值断言。
(defmacro check-error (name form)
  "断言 FORM 抛错（用于「numpy 同样拒绝」的契约）。"
  `(handler-case (progn ,form
                    (incf *fail*)
                    (format t "~&FAIL ~a~%  got:  正常返回（期望报错）~%" ,name))
     (error () (incf *pass*))))

;; K1. vt-negative：numpy 拒绝 bool 取负；原实现写回 255（脏字节）
(check-error "negative/bool" (vt-negative *jb*))
(check "negative/bool-i8" (flat-ints (vt-negative *jb* :dtype :int8)) '(-1 -1 0 0))
(check "negative/i64" (flat-ints (vt-negative (vt-from-sequence '(1 2) :dtype :int64))) '(-1 -2))

;; K2. vt-atan2：docstring 一直写「整数输入输出 float64」，实现曾把结果留在
;;     整数/布尔 dtype 上，0.785 被截断成 0
(check "atan2/bool-dtype" (vt-dtype (vt-atan2 *jb* *jb*)) :float64)
(check-true "atan2/bool-vals" (< (abs (- (vt-ref (vt-atan2 *jb* *jb*) 0)
                                         0.7853981633974483d0)) 1d-12))
(check "atan2/i64-dtype" (vt-dtype (vt-atan2 (vt-from-sequence '(1 2) :dtype :int64)
                                             (vt-from-sequence '(1 2) :dtype :int64)))
       :float64)
(check-true "atan2/i64-vals" (< (abs (- (vt-ref (vt-atan2 (vt-from-sequence '(1 2) :dtype :int64)
                                                         (vt-from-sequence '(1 2) :dtype :int64)) 1)
                                         0.7853981633974483d0)) 1d-12))
(check "atan2/f32-dtype" (vt-dtype (vt-atan2 (vt-astype *jb* :float32)
                                             (vt-astype *jb* :float32))) :float32)

;; K3. vt-pow：bool 基 + 正整数指数 → :int64（§17.10.1 的同一条提升规则）
(check "pow/bool-2-dtype" (vt-dtype (vt-pow *jb* 2)) :int64)
(check "pow/bool-2-vals"  (flat-ints (vt-pow *jb* 2)) '(1 1 0 0))

;; K4. vt-rem：与 vt-mod 同为「布尔 → :int8」
(check "rem/bool-dtype" (vt-dtype (vt-rem *jb* *jb*)) :int8)
(check "rem/bool-vals"  (flat-ints (vt-rem *jb* *jb*)) '(0 0 0 0))
(check "mod/bool-dtype" (vt-dtype (vt-mod *jb* *jb*)) :int8)

;; K5. vt-convolve / vt-correlate：bool 输入走逻辑与/或，结果仍 :bool
;;     （修复前返回 :float64 的数值和，[1 1 1 0 0 0 0] 的 1 会变成计数）
(check "convolve/bool-dtype"   (vt-dtype (vt-convolve *jb* *jb*)) :bool)
(check "convolve/bool-full"    (flat-ints (vt-convolve *jb* *jb*)) '(1 1 1 0 0 0 0))
(check "convolve/bool-valid"   (flat-ints (vt-convolve *jb* *jb* :mode :valid)) '(0))
(check "correlate/bool-dtype"  (vt-dtype (vt-correlate *jb* *jb*)) :bool)
(check "correlate/bool-full"   (flat-ints (vt-correlate *jb* *jb*)) '(0 0 1 1 1 0 0))
(check "correlate/bool-valid"  (flat-ints (vt-correlate *jb* *jb* :mode :valid)) '(1))
;; 浮点路径不受影响
(check "convolve/f64" (vt-dtype (vt-convolve (vt-astype *jb* :float64)
                                             (vt-astype *jb* :float64))) :float64)

;; K6. vt-cross：numpy 抛 TypeError（叉积含减法）；原实现静默给 :bool (1 1 1)
(check-error "cross/bool" (vt-cross (vt-from-sequence '(t nil t) :dtype :bool)
                                    (vt-from-sequence '(nil t t) :dtype :bool)))
(check "cross/bool-i64" (flat-ints (vt-cross (vt-from-sequence '(t nil t) :dtype :bool)
                                             (vt-from-sequence '(nil t t) :dtype :bool)
                                             :dtype :int64))
       '(-1 -1 1))

;; --- L. 累积提升表与「复数 → 布尔」的两条入口（§17.10.12） ---
;; L 组盯的是**同一族里只改了一个**这类缺陷：`cumsum` 与 `nancumsum`
;; 走两条内核，前者有提升表、后者没有；`vt-cast` 与 `vt-astype` 是两个
;; 入口，前者先剥虚部再判 :bool、后者按整体判。两组都是「各自单看都自洽，
;; 放在一起才矛盾」。
(defun bool-bad-bytes (tensor)
  "统计 :bool 张量里物理字节不在 {0,1} 的个数（0/1 不变量扫描）。"
  (let ((d (vt-data tensor)) (n 0))
    (when (eq (vt-dtype tensor) :bool)
      (dotimes (i (length d)) (unless (<= 0 (aref d i) 1) (incf n))))
    n))

;; L1. nancumsum / nancumprod：结果必须是 int64 的**计数**，不是 :bool
(check "nancumsum/bool-dtype" (vt-dtype (vt-nancumsum *jb*)) :int64)
(check "nancumsum/bool-vals"  (flat-ints (vt-nancumsum *jb*)) '(1 2 2 2))
(check-true "nancumsum/bool-bytes" (zerop (bool-bad-bytes (vt-nancumsum *jb*))))
(check "nancumprod/bool-dtype" (vt-dtype (vt-nancumprod *jb*)) :int64)
(check "nancumprod/bool-vals"  (flat-ints (vt-nancumprod *jb*)) '(1 1 0 0))
;; 与同族的 cumsum 必须给出同一个 dtype
(check "nancumsum/dtype-same-as-cumsum"
       (vt-dtype (vt-nancumsum *jb*)) (vt-dtype (vt-cumsum *jb*)))

;; L2. 同一个提升表也覆盖窄整型（原实现静默回绕）
(check "nancumsum/int8-narrow"  (vt-dtype (vt-nancumsum (vt-from-sequence '(100 100) :dtype :int8))) :int64)
(check "nancumsum/int8-vals"    (flat-ints (vt-nancumsum (vt-from-sequence '(100 100) :dtype :int8))) '(100 200))
(check "nancumsum/uint8-vals"   (flat-ints (vt-nancumsum (vt-from-sequence '(200 200) :dtype :uint8))) '(200 400))
(check "nancumsum/int16-vals"   (flat-ints (vt-nancumsum (vt-from-sequence '(30000 30000) :dtype :int16))) '(30000 60000))
(check "nancumprod/int8-vals"   (flat-ints (vt-nancumprod (vt-from-sequence '(100 100) :dtype :int8))) '(100 10000))
;; 大 dtype 不该被无谓提升
(check "nancumsum/int64-dtype"  (vt-dtype (vt-nancumsum (vt-from-sequence '(1 2) :dtype :int64))) :int64)
(check "nancumsum/float32-dtype" (vt-dtype (vt-nancumsum (vt-from-sequence '(1.0 2.0) :dtype :float32))) :float32)

;; L3. 复数 → :bool：任一分量非零即为真（%bool-flag 的契约）
(check "cast/complex-0+1j" (vt-cast (complex 0 1) :bool) 1)
(check "cast/complex-0+0j" (vt-cast (complex 0 0) :bool) 0)
(check "cast/complex-1+0j" (vt-cast (complex 1 0) :bool) 1)
;; 与数组路径一致
(check "astype/complex-0+1j"
       (flat-ints (vt-astype (vt-from-sequence (list (complex 0.0d0 1.0d0))) :bool)) '(1))
;; 与 (setf vt-ref) 一致 —— 四个写入入口必须认同一个真值判定
(check "write/complex-0+1j" (flat-ints (let ((x (vt-zeros '(2) :dtype :bool)))
                                         (vt-write x '(0) (complex 0 1)) x)) '(1 0))
(check "put/complex-0+1j"   (flat-ints (let ((x (vt-zeros '(2) :dtype :bool)))
                                         (vt-put x '(0) (list (complex 0 1))) x)) '(1 0))
(check "fill/complex-0+1j"  (flat-ints (let ((x (vt-zeros '(2) :dtype :bool)))
                                         (vt-fill x (complex 0 1)) x)) '(1 1))
(check "setf-ref/complex-0+1j" (flat-ints (let ((x (vt-zeros '(2) :dtype :bool)))
                                            (setf (vt-ref x 0) (complex 0 1)) x)) '(1 0))
(check "pad/complex-0+1j"   (flat-ints (vt-pad (vt-from-sequence '(t) :dtype :bool) '(1 1)
                                               :constant-values (complex 0 1))) '(1 1 1))

;; --- M. 「消费布尔」的入口（§17.10.13）---
;; 前四遍审计都是「布尔当数据算」；这一遍换的是「布尔被谁读走」：倒数/整除
;; 把 0/1 当数字、峰谷差把 vt-item 的 T 当数、apply-along-axis 把函数返回值
;; 塞回 bool 数组、形状工具收到张量直接段错误。共同点是**入口不在算术族里**，
;; 所以按算子名扫不到。
(defparameter *mb* (vt-from-sequence '((t nil t t) (t t nil t)) :dtype :bool))

;; M1. vt-reciprocal：numpy 的 bool 循环落 int8（1//x）。原实现因为
;;     `(vt-int-dtype-p :bool)` 为假而落到浮点路径，1/0 在 with-float-safe
;;     下是 +Inf，写回 :bool 又被 %bool-flag 收成 1 —— (T NIL T T) 静默变
;;     成 (T T T T)，假值位置变真。
(check "reciprocal/all-true-dtype"
       (vt-dtype (vt-reciprocal (vt-from-sequence '(t t) :dtype :bool))) :int8)
(check "reciprocal/all-true-vals"
       (flat-ints (vt-reciprocal (vt-from-sequence '(t t) :dtype :bool))) '(1 1))
(check-error "reciprocal/with-false" (vt-reciprocal *jb*))  ; 整数零除（本库契约）
(check "reciprocal/int8-unchanged"
       (flat-ints (vt-reciprocal (vt-from-sequence '(2 4) :dtype :int8))) '(0 0))
(check "reciprocal/float32-unchanged"
       (vt-dtype (vt-reciprocal (vt-from-sequence '(2.0 4.0) :dtype :float32))) :float32)

;; M2. vt-divmod：商必须与 mod/rem/div 同族（:int8）。原实现让 floor(0/0)
;;     的 NaN 落进 :bool 得到真，商数组整个变成 (T T ...)。
(check "divmod/bool-dtype" (vt-dtype (nth-value 0 (vt-divmod *jb* *jb*))) :int8)
(check "divmod/bool-quot" (flat-ints (nth-value 0 (vt-divmod *jb* *jb*))) '(1 1 0 0))
(check "divmod/bool-rem"  (flat-ints (nth-value 1 (vt-divmod *jb* *jb*))) '(0 0 0 0))
(check "divmod/dtype-same-as-mod"
       (vt-dtype (nth-value 0 (vt-divmod *jb* *jb*))) (vt-dtype (vt-mod *jb* *jb*)))
(check "divmod/mixed-dtype"
       (vt-dtype (nth-value 0 (vt-divmod *jb* (vt-from-sequence '(2 3 4 5) :dtype :int64))))
       :int64)

;; M3. vt-ptp：布尔峰谷差就是减法，numpy 拒绝。原实现的非 axis 分支把
;;     `(vt-item bool标量)` 的 T 直接丢进 `-`，报的是「T is not of type
;;     NUMBER」；同一函数的 :axis 分支走 vt--，报的却是正确的布尔消息。
(check-error "ptp/bool" (vt-ptp *jb*))
(check-error "ptp/bool-axis" (vt-ptp *mb* :axis 0))
(check-true "ptp/bool-error-mentions-bool"
            (search "布尔" (handler-case (progn (vt-ptp *jb*) "")
                             (error (e) (princ-to-string e)))))
(check "ptp/int-unchanged" (vt-ptp (vt-from-sequence '(3 9 1))) 8.0d0)

;; M4. 形状工具收到张量：原实现在 (safety 0) + `(declare (list …))` 下取
;;     car 链，实测 `Unhandled memory fault at #x0`。误用应当拿到错误消息。
(check-error "broadcast-shapes/rejects-vt" (vt-broadcast-shapes *jb* '(2)))
(check-error "broadcast-strides/rejects-vt" (vt-broadcast-strides *jb* '(2) '(1)))
(check "broadcast-shapes/ok" (vt-broadcast-shapes '(3 1) '(1 4)) '(3 4))
(check "broadcast-strides/ok" (vt-broadcast-strides '(3 1) '(3 4) '(1 1)) '(1 0))

;; M5. vt-apply-along-axis：结果 dtype 原来固定取输入 dtype。布尔输入时
;;     1D→scalar 分支把 (vt-sum row) 的 3 压成 1；1D→1D 分支直接 AREF
;;     写入 5、6 这种越界字节（破 0/1 不变量）。现在布尔输入按 FUNC 的
;;     返回类型定 dtype，非布尔输入维持旧契约。
(check "apply-along/sum-dtype" (vt-dtype (vt-apply-along-axis #'vt-sum 1 *mb*)) :int64)
(check "apply-along/sum-vals"  (flat-ints (vt-apply-along-axis #'vt-sum 1 *mb*)) '(3 3))
(check "apply-along/1d1d-dtype"
       (vt-dtype (vt-apply-along-axis (lambda (r) (vt-from-sequence '(5 6) :dtype :int64)) 1 *mb*))
       :int64)
(check "apply-along/1d1d-vals"
       (flat-ints (vt-apply-along-axis (lambda (r) (vt-from-sequence '(5 6) :dtype :int64)) 1 *mb*))
       '(5 6 5 6))
(check-true "apply-along/1d1d-no-dirty-bytes"
            (zerop (bool-bad-bytes
                    (vt-apply-along-axis (lambda (r) (vt-from-sequence '(5 6) :dtype :int64))
                                         1 *mb*))))
(check "apply-along/bool-stays-bool"
       (vt-dtype (vt-apply-along-axis #'vt-logical-not 1 *mb*)) :bool)
(check "apply-along/int8-keeps"
       (vt-dtype (vt-apply-along-axis #'vt-sum 1 (vt-from-sequence '((1 2) (3 4)) :dtype :int8)))
       :int8)

;; M6. vt-vander：numpy 给 int64。保持 :bool 会让「第 3 列 = x²」与
;;     (vt-pow b 2)（§17.10.11 定为 int64）在同一个表达式上给出两种 dtype。
(check "vander/bool-dtype" (vt-dtype (vt-vander *jb*)) :int64)
(check "vander/bool-vals"
       (flat-ints (vt-vander (vt-from-sequence '(t nil t) :dtype :bool)))
       '(1 1 1 0 0 1 1 1 1))
(check "vander/int-unchanged"
       (vt-dtype (vt-vander (vt-from-sequence '(1 2) :dtype :int64))) :int64)
(check "vander/float-unchanged"
       (vt-dtype (vt-vander (vt-from-sequence '(1.0 2.0)))) :float64)

;;; ==================================================================

(format t "~&~%==== bool dtype: ~d passed, ~d failed ====~%" *pass* *fail*)
(when (plusp *fail*) (sb-ext:exit :code 1))
