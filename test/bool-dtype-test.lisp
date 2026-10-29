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
;;;;   E. 打印：布尔张量印 True/False 且头部 dtype 为 BOOL（不与 uint8 混淆）。
;;;;   F. npy 往返：|b1 描述符保存/读回后仍是 :bool。
;;;;   G. 布尔算术：+/* 按逻辑或/与、- 报错、/ 升 float64、// 与 % 落 int8 ——
;;;;      numpy 的 bool_ 不按整数算，否则 True+True=2 会破掉 0/1 不变量。
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
    (loop for i below n collect (truncate (vt-ref flat i)))))

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
    (loop for i below n always (member (vt-ref flat i) '(0 1)))))

(check-true "invariant/isnan"    (bool-bytes-ok-p (vt-isnan *bx*)))
(check-true "invariant/gt"      (bool-bytes-ok-p (vt-> *bx* 0.0d0)))
(check-true "invariant/astype"  (bool-bytes-ok-p (vt-astype *bx* :bool)))
(check-true "invariant/any"     (bool-bytes-ok-p (vt-any *bm* :axis 0)))
(check-true "invariant/logical" (bool-bytes-ok-p (vt-logical-xor *m1* *m2*)))
;; astype 的真值化：0.5 → True（不是截断成 0）、NaN → True、-0.0 → False
(check "astype/0.5"   (vt-ref (vt-astype (vt-from-sequence '(0.5d0)) :bool) 0) 1)
(check "astype/nan"   (vt-ref (vt-astype (vt-from-sequence (list (vt-float-nan))) :bool) 0) 1)
(check "astype/negzero" (vt-ref (vt-astype (vt-from-sequence '(-0.0d0)) :bool) 0) 0)
(check "astype/zero"  (vt-ref (vt-astype (vt-from-sequence '(0.0d0)) :bool) 0) 0)

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
  (check-true "print/True"  (search "True" s))
  (check-true "print/False" (search "False" s))
  (check-true "print/dtype-tag" (search "BOOL" s))
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
    (check "arith/div-0-0-nan" (vt-item (vt-isnan (vt-ref r 2))) 1))
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

(format t "~&~%==== bool dtype: ~d passed, ~d failed ====~%" *pass* *fail*)
(when (plusp *fail*) (sb-ext:exit :code 1))
