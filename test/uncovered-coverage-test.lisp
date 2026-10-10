;;;; uncovered-coverage-test.lisp — 覆盖缺口测试套件
;;;;
;;;; 目的：CONVENTIONS.md §12 V1 要求 25 个测试套件全绿，但审计发现
;;;; 一批**已导出**的公开函数从未被任何测试或 example 调用。本套件
;;;; 为这些函数补齐「至少一条」语义断言，期望值全部来自 NumPy 2.4.6
;;;; 实测（见 test/gen_uncovered_expected.py）或 CONVENTIONS.md 契约。
;;;;
;;;; 覆盖清单（来源：comm 对比 src/package.lisp 导出 与 test/*.lisp + example）：
;;;;   vt-add vt-sub vt-mul vt-<= vt->= vt-atan vt-asinh vt-acosh vt-atanh
;;;;   vt-expt vt-lerp vt-negative vt-positive-p vt-negative-p vt-zero-p
;;;;   vt-nonzero-p vt-even-p vt-odd-p vt-logical-and vt-logical-or
;;;;   vt-logical-not vt-logical-xor vt-bit-ior vt-bit-xor vt-bit-not
;;;;   vt-left-shift vt-right-shift vt-identity vt-empty vt-empty-like
;;;;   vt-ones-like vt-zeros-like vt-full-like vt-concat vt-unsqueeze
;;;;   vt-vsplit vt-dsplit vt-clamp vt-float-nan-inf-= vt-select
;;;;   vt-nonzero-p vt-lstsq vt-copy-to! vt-vander vt-flatten-sequence
;;;;   vt-compute-strides vt-compute-logical-strides vt-normalize-axis
;;;;   vt-element-type vt-out-contig-p vt-out-writable-p vt-write
;;;;   vt-count-nonzero vt-flatnonzero vt-inner vt-topk vt-moveaxis
;;;;   vt-fliplr vt-flipud vt-ediff1d vt-ravel-multi-index
;;;;   vt-tril-indices vt-triu-indices vt-standardize vt-seed-sequence
;;;;   vt-generator vt-spawn-generators
;;;;
;;;; 运行：sbcl --noinform --non-interactive --load test/uncovered-coverage-test.lisp

(require :asdf)
;; 由 run-tests.sh 预设 asdf:*central-registry*；直接 --load 时补齐项目根。
(unless (find-if (lambda (p) (probe-file (merge-pathnames "clvt.asd" p)))
                 asdf:*central-registry*)
  (push (truename (make-pathname :directory '(:relative :up))) asdf:*central-registry*))
#+quicklisp (handler-bind ((warning #'muffle-warning)) (ql:quickload :clvt :silent t))
(handler-bind ((warning #'muffle-warning)) (asdf:load-system :clvt))
(in-package :clvt)

;;; ============================================================
;;; 测试框架（与 refactor-bugfix-tests 同款：浮点近似 + 结构严格）
;;; ============================================================
(defvar *n* 0)
(defvar *p* 0)
(defvar *f* 0)
(defvar *fails* nil)

(defun vclose (a b &optional (tol 1d-9))
  "浮点/结构混合比较：数值走容差，序列递归，其余 equal。"
  (cond
    ((and (numberp a) (numberp b))
     (let ((aa (float a 1d0)) (bb (float b 1d0)))
       (or (and (sb-ext:float-nan-p aa) (sb-ext:float-nan-p bb))
           (< (abs (- aa bb)) (+ tol (* 1d-6 (max (abs aa) (abs bb) 1d0)))))))
    ((and (listp a) (listp b))
     (and (= (length a) (length b)) (every #'vclose a b)))
    ((and (vectorp a) (vectorp b))
     (and (= (length a) (length b)) (every #'vclose (coerce a 'list) (coerce b 'list))))
    (t (equal a b))))

(defun scalar (x)
  "把结果归一为 Lisp 标量：vt →（必要时再 vt-item），非 vt 原样返回。"
  (if (vt-p x) (vt-item x) x))

(defun firstval (vt)
  "取 1 维 vt 的第 0 个元素为 Lisp 标量（用 vt-ref 展平 0 维 vt 的问题）。"
  (scalar (vt-ref vt 0)))

(defun nlist (vt)
  "取 vt 的嵌套列表（0 维返回标量）。"
  (when (vt-p vt)
    (if (null (vt-shape vt)) (scalar vt) (vt-to-list vt))))

(defun check (name expected actual &optional (tol 1d-9))
  (incf *n*)
  (if (vclose expected actual tol)
      (incf *p*)
      (progn (incf *f*) (push name *fails*)
             (format t "  ❌ ~a~%     exp: ~a~%     got: ~a~%" name expected actual))))

(defun check-error (name thunk)
  "断言 thunk 抛出错误（L2 底线）。"
  (incf *n*)
  (handler-case (progn (funcall thunk)
                       (incf *f*) (push name *fails*)
                       (format t "  ❌ ~a (应报错但成功)~%" name))
    (error () (incf *p*))))

(defun check-shape-dtype (name vt exp-shape exp-dtype)
  (check (format nil "~a.shape" name) exp-shape (vt-shape vt))
  (check (format nil "~a.dtype" name) exp-dtype (vt-dtype vt)))

;;; ============================================================
;;; 1. 二元算术别名 & 比较（numpy 实测值）
;;; ============================================================
(defun test-binary-aliases ()
  (format t "~%[1] 二元算术别名 / 比较~%")
  (check "vt-add" '(11.0d0 22.0d0 33.0d0)
         (nlist (vt-add (vt-from-sequence '(1 2 3)) (vt-from-sequence '(10 20 30)))))
  (check "vt-sub" '(9.0d0 18.0d0 27.0d0)
         (nlist (vt-sub (vt-from-sequence '(10 20 30)) (vt-from-sequence '(1 2 3)))))
  (check "vt-mul scalar" '(8.0d0 12.0d0)
         (nlist (vt-mul (vt-from-sequence '(2 3)) 4)))
  ;; vt-<= / vt->= 返回 int8 布尔语义（CONVENTIONS §3.1）
  (let ((r (vt-<= (vt-from-sequence '(1 2 3)) 2)))
    (check-shape-dtype "vt-<=" r '(3) :int8)
    (check "vt-<= 值" '(1 1 0) (nlist r)))
  (let ((r (vt->= (vt-from-sequence '(1 2 3)) 2)))
    (check-shape-dtype "vt->=" r '(3) :int8)
    (check "vt->= 值" '(0 1 1) (nlist r)))
  ;; 比较运算 dtype 契约（CONVENTIONS §3.1：统一 :int8 承载布尔）
  (check "vt-= dtype" :int8 (vt-dtype (vt-= (vt-from-sequence '(1 2)) (vt-from-sequence '(1 3)))))
  (check "vt-< dtype" :int8 (vt-dtype (vt-< (vt-from-sequence '(1 2)) 2)))
  (check "vt-> dtype" :int8 (vt-dtype (vt-> (vt-from-sequence '(1 2)) 2)))
  (check "vt-/= dtype" :int8 (vt-dtype (vt-/= (vt-from-sequence '(1 2)) 2)))
  (check "vt-isnan dtype" :int8 (vt-dtype (vt-isnan (vt-from-sequence (list 1.0d0 +vt-dfloat-nan+)))))
  (check "vt-isfinite dtype" :int8 (vt-dtype (vt-isfinite (vt-from-sequence '(1.0 2.0)))))
  ;; 逐元素乘 + out（out 必须与结果 dtype 一致）
  (let ((out (vt-zeros '(2))))
    (vt-mul (vt-from-sequence '(2 3)) 4 :out out)
    (check "vt-mul :out" '(8.0d0 12.0d0) (nlist out)))
  ;; 比较的 out 必须为 int8（H3）
  (let ((out (vt-zeros '(3) :dtype :int8)))
    (vt-<= (vt-from-sequence '(1 2 3)) 2 :out out)
    (check "vt-<= :out int8" '(1 1 0) (nlist out))))

;;; ============================================================
;;; 2. 三角函数 / 双曲（numpy 实测）
;;; ============================================================
(defun test-trig-hyperbolic ()
  (format t "~%[2] 三角 / 双曲未覆盖函数~%")
  (check "vt-atan" '(0.0d0 0.7853981633974483d0)
         (nlist (vt-atan (vt-from-sequence '(0.0 1.0)))))
  (check "vt-asinh" '(0.0d0 0.881373587019543d0)
         (nlist (vt-asinh (vt-from-sequence '(0.0 1.0)))))
  (check "vt-acosh" '(0.0d0 1.3169578969248168d0)
         (nlist (vt-acosh (vt-from-sequence '(1.0 2.0)))))
  (check "vt-atanh" '(0.0d0 0.5493061443340549d0)
         (nlist (vt-atanh (vt-from-sequence '(0.0 0.5)))))
  ;; acosh 定义域外 → NaN
  (check "vt-acosh 定义域外 NaN" t
         (sb-ext:float-nan-p (scalar (vt-ref (vt-acosh (vt-from-sequence '(0.5))) 0))))
  ;; atanh ±1 → ±Inf
  (check "vt-atanh(+1)=+Inf" t
         (vt-float-pos-inf-p (scalar (vt-ref (vt-atanh (vt-from-sequence '(1.0))) 0))))
  ;; dtype 保持
  (check-shape-dtype "vt-atan float32" (vt-atan (vt-ones '(3) :dtype :float32)) '(3) :float32))

;;; ============================================================
;;; 3. 幂 / 插值 / 取负
;;; ============================================================
(defun test-expt-lerp-negative ()
  (format t "~%[3] vt-expt / vt-lerp / vt-negative~%")
  (check "vt-expt" '(4.0d0 9.0d0) (nlist (vt-expt (vt-from-sequence '(2 3)) 2)))
  (check "vt-expt 0.5" '(2.0d0 3.0d0) (nlist (vt-expt (vt-from-sequence '(4 9)) 0.5)))
  (check "vt-lerp w=0.5" '(5.0d0 10.5d0)
         (nlist (vt-lerp (vt-from-sequence '(0.0 1.0)) (vt-from-sequence '(10.0 20.0)) 0.5)))
  (check "vt-lerp w=0.25" '(2.5d0)
         (nlist (vt-lerp (vt-from-sequence '(0.0)) (vt-from-sequence '(10.0)) 0.25)))
  ;; CL 参数顺序 (start end w)：w=0 取 start，w=1 取 end
  (check "vt-lerp w=0" '(0.0d0) (nlist (vt-lerp (vt-from-sequence '(0.0)) (vt-from-sequence '(10.0)) 0)))
  (check "vt-lerp w=1" '(10.0d0) (nlist (vt-lerp (vt-from-sequence '(0.0)) (vt-from-sequence '(10.0)) 1)))
  (check "vt-negative" '(-1.0d0 2.0d0 -3.0d0)
         (nlist (vt-negative (vt-from-sequence '(1 -2 3)))))
  ;; 整数输入 dtype 保持
  (check-shape-dtype "vt-negative int32" (vt-negative (vt-ones '(3) :dtype :int32)) '(3) :int32))

;;; ============================================================
;;; 4. 谓词族（返回 int8 布尔语义）
;;; ============================================================
(defun test-predicates ()
  (format t "~%[4] 谓词族~%")
  (check "vt-positive-p" '(1 0 1) (nlist (vt-positive-p (vt-from-sequence '(1 -2 3)))))
  (check "vt-negative-p" '(0 1 0) (nlist (vt-negative-p (vt-from-sequence '(1 -2 3)))))
  (check "vt-zero-p" '(1 0 1) (nlist (vt-zero-p (vt-from-sequence '(0 1 0)))))
  (check "vt-nonzero-p" '(0 1 0) (nlist (vt-nonzero-p (vt-from-sequence '(0 1 0)))))
  (check "vt-even-p" '(1 0 1) (nlist (vt-even-p (vt-from-sequence '(2 3 4)))))
  (check "vt-odd-p" '(0 1 0) (nlist (vt-odd-p (vt-from-sequence '(2 3 4)))))
  ;; CONVENTIONS §5.5：非有限值一律判定不成立（返回 0），不报错
  (check "vt-even-p NaN → 0" '(0)
         (nlist (vt-even-p (vt-from-sequence (list +vt-dfloat-nan+)))))
  (check "vt-odd-p Inf → 0" '(0)
         (nlist (vt-odd-p (vt-from-sequence (list +vt-dfloat-pos-inf+)))))
  (check "vt-positive-p NaN → 0" '(0)
         (nlist (vt-positive-p (vt-from-sequence (list +vt-dfloat-nan+)))))
  ;; 谓词 dtype 契约
  (check "vt-even-p dtype" :int8 (vt-dtype (vt-even-p (vt-from-sequence '(2 3)))))
  (check "vt-positive-p dtype" :int8 (vt-dtype (vt-positive-p (vt-from-sequence '(1 -1))))))

;;; ============================================================
;;; 5. 逻辑 / 位运算（numpy 实测）
;;; ============================================================
(defun test-logic-bitwise ()
  (format t "~%[5] 逻辑 / 位运算~%")
  (check "vt-logical-and" '(1 0) (nlist (vt-logical-and (vt-from-sequence '(1 0)) (vt-from-sequence '(1 1)))))
  (check "vt-logical-or" '(1 0) (nlist (vt-logical-or (vt-from-sequence '(1 0)) (vt-from-sequence '(0 0)))))
  (check "vt-logical-not" '(1 0) (nlist (vt-logical-not (vt-from-sequence '(0 1)))))
  (check "vt-logical-xor" '(0 1) (nlist (vt-logical-xor (vt-from-sequence '(1 0)) (vt-from-sequence '(1 1)))))
  (let ((a (vt-from-sequence '(1 2) :dtype :int32)) (b (vt-from-sequence '(3 4) :dtype :int32)))
    (check "vt-bit-ior" '(3 6) (nlist (vt-bit-ior a b)))
    (check "vt-bit-xor" '(2 6) (nlist (vt-bit-xor a b))))
  (check "vt-bit-not" '(-1 -2) (nlist (vt-bit-not (vt-from-sequence '(0 1) :dtype :int32))))
  (check "vt-left-shift" '(2 4) (nlist (vt-left-shift (vt-from-sequence '(1 2) :dtype :int32) 1)))
  (check "vt-right-shift" '(2 4) (nlist (vt-right-shift (vt-from-sequence '(4 8) :dtype :int32) 1))))

;;; ============================================================
;;; 6. 创建族：identity / empty / *-like（★ 回归 vt-identity bug）
;;; ============================================================
(defun test-creation ()
  (format t "~%[6] 创建族（含 vt-identity 回归）~%")
  ;; ★ bug 回归：vt-identity 缺省 dtype 曾触发 `NIL fell through ECASE'
  (let ((m (vt-identity 3)))
    (check-shape-dtype "vt-identity 缺省" m '(3 3) :float64)
    (check "vt-identity 值" '((1.0d0 0.0d0 0.0d0) (0.0d0 1.0d0 0.0d0) (0.0d0 0.0d0 1.0d0)) (nlist m)))
  (let ((m (vt-identity 3 :dtype :int64)))
    (check-shape-dtype "vt-identity int64" m '(3 3) :int64)
    (check "vt-identity int64 值" '((1 0 0) (0 1 0) (0 0 1)) (nlist m)))
  (let ((e (vt-empty '(2 3))))
    (check-shape-dtype "vt-empty" e '(2 3) :float64))
  (let ((e (vt-empty-like (vt-zeros '(2 3) :dtype :int32))))
    (check-shape-dtype "vt-empty-like 继承 dtype" e '(2 3) :int32))
  (check-shape-dtype "vt-zeros-like 继承 dtype" (vt-zeros-like (vt-ones '(2 3) :dtype :float32)) '(2 3) :float32)
  (check-shape-dtype "vt-ones-like 继承 dtype" (vt-ones-like (vt-zeros '(2 3) :dtype :int16)) '(2 3) :int16)
  (check "vt-ones-like 值" '((1.0 1.0) (1.0 1.0)) (nlist (vt-ones-like (vt-zeros '(2 2) :dtype :float32))))
  (check "vt-full-like 值" '((7.0d0 7.0d0) (7.0d0 7.0d0)) (nlist (vt-full-like (vt-zeros '(2 2)) 7)))
  (check-shape-dtype "vt-full-like 继承" (vt-full-like (vt-ones '(3) :dtype :int32) 5) '(3) :int32))

;;; ============================================================
;;; 7. 拼接 / 维度
;;; ============================================================
(defun test-concat-dims ()
  (format t "~%[7] 拼接 / 维度~%")
  ;; vt-concat 的轴参数在前（区别于 vt-concatenate）
  (check "vt-concat 1d" '(1.0d0 2.0d0 3.0d0 4.0d0)
         (nlist (vt-concat 0 (vt-from-sequence '(1 2)) (vt-from-sequence '(3 4)))))
  (check "vt-concat 2d axis=1 shape" '(2 4)
         (vt-shape (vt-concat 1 (vt-zeros '(2 2)) (vt-zeros '(2 2)))))
  (check-shape-dtype "vt-unsqueeze axis0" (vt-unsqueeze (vt-from-sequence '(1 2 3)) 0) '(1 3) :float64)
  (check-shape-dtype "vt-unsqueeze axis1" (vt-unsqueeze (vt-from-sequence '(1 2 3)) 1) '(3 1) :float64)
  ;; vsplit / dsplit
  (let ((parts (vt-vsplit (vt-zeros '(4 2)) 2)))
    (check "vt-vsplit 段数" 2 (length parts))
    (check "vt-vsplit 段形状" '(2 2) (vt-shape (first parts))))
  (let ((parts (vt-dsplit (vt-zeros '(2 2 6)) 3)))
    (check "vt-dsplit 段数" 3 (length parts))
    (check "vt-dsplit 段形状" '(2 2 2) (vt-shape (first parts))))
  ;; dsplit 对 <3D 输入必须报错（与 numpy 一致）
  (check-error "vt-dsplit 2D 报错" (lambda () (vt-dsplit (vt-zeros '(2 6)) 6))))

;;; ============================================================
;;; 8. vt-clamp / vt-select / vt-copy-to! / vt-float-nan-inf-=
;;; ============================================================
(defun test-clamp-select ()
  (format t "~%[8] vt-clamp / vt-select / vt-copy-to!~%")
  (check "vt-clamp" '(2.0d0 5.0d0 8.0d0)
         (nlist (vt-clamp (vt-from-sequence '(1.0 5.0 10.0)) 2.0 8.0)))
  (check "vt-clamp 全低于下限" '(2.0d0 2.0d0) (nlist (vt-clamp (vt-from-sequence '(0.0 1.0)) 2.0 8.0)))
  (check "vt-clamp 全高于上限" '(8.0d0 8.0d0) (nlist (vt-clamp (vt-from-sequence '(9.0 10.0)) 2.0 8.0)))
  ;; vt-select：condlist/choicelist 为列表
  (check "vt-select" '(10.0d0 200.0d0 30.0d0)
         (nlist (vt-select (list (vt-from-sequence '(1 0 1)) (vt-from-sequence '(0 1 0)))
                           (list (vt-from-sequence '(10 20 30)) (vt-from-sequence '(100 200 300))))))
  (check "vt-select default" '(9.0d0 9.0d0 9.0d0)
         (nlist (vt-select (list (vt-from-sequence '(0 0 0))) (list (vt-from-sequence '(10 20 30))) :default 9)))
  (check-error "vt-select 长度不等报错"
               (lambda () (vt-select (list (vt-from-sequence '(1))) (list (vt-from-sequence '(1)) (vt-from-sequence '(2))))))
  ;; copy-to!
  (let ((dst (vt-zeros '(2 2))) (src (vt-ones '(2 2))))
    (vt-copy-to! dst src)
    (check "vt-copy-to! 值" '((1.0d0 1.0d0) (1.0d0 1.0d0)) (nlist dst)))
  ;; nan-inf 等值判定
  (check "vt-float-nan-inf-= nan" t (vt-float-nan-inf-= +vt-dfloat-nan+ +vt-dfloat-nan+))
  (check "vt-float-nan-inf-= number" t (vt-float-nan-inf-= 1.0d0 1.0d0))
  (check "vt-float-nan-inf-= nan vs 1" nil (vt-float-nan-inf-= +vt-dfloat-nan+ 1.0d0)))

;;; ============================================================
;;; 9. vt-vander（★ 回归：缺省 n 曾报 odd number of &KEY arguments）
;;; ============================================================
(defun test-vander ()
  (format t "~%[9] vt-vander（缺省 n 回归）~%")
  ;; ★ bug 回归：缺省 n 时必须取 len(x)
  (check "vt-vander 缺省 n shape" '(3 3) (vt-shape (vt-vander (vt-from-sequence '(1 2 3)))))
  (check "vt-vander 缺省 n 值" '((1.0d0 1.0d0 1.0d0) (4.0d0 2.0d0 1.0d0) (9.0d0 3.0d0 1.0d0))
         (nlist (vt-vander (vt-from-sequence '(1 2 3)))))
  (check "vt-vander n=2" '((1.0d0 1.0d0) (2.0d0 1.0d0) (3.0d0 1.0d0))
         (nlist (vt-vander (vt-from-sequence '(1 2 3)) :n 2)))
  (check "vt-vander increasing" '((1.0d0 1.0d0 1.0d0) (1.0d0 2.0d0 4.0d0) (1.0d0 3.0d0 9.0d0))
         (nlist (vt-vander (vt-from-sequence '(1 2 3)) :increasing t))))

;;; ============================================================
;;; 10. 结构原语（core / parcontract）
;;; ============================================================
(defun test-primitives ()
  (format t "~%[10] 结构原语~%")
  (check "vt-flatten-sequence" '(1 2 3 4) (vt-flatten-sequence '((1 2) (3 4))))
  (check "vt-compute-strides" '(3 1) (vt-compute-strides '(2 3)))
  (check "vt-compute-strides 3d" '(12 4 1) (vt-compute-strides '(2 3 4)))
  (check "vt-compute-logical-strides" '(12 4 1) (vt-compute-logical-strides '(2 3 4)))
  (check "vt-normalize-axis -1 rank3" 2 (vt-normalize-axis -1 3))
  (check "vt-normalize-axis 0 rank3" 0 (vt-normalize-axis 0 3))
  (check "vt-element-type int32" '(signed-byte 32) (vt-element-type (vt-zeros '(2) :dtype :int32)))
  (check "vt-element-type float64" 'double-float (vt-element-type (vt-zeros '(2))))
  ;; out 结构判定
  (check "vt-out-contig-p 连续" t (vt-out-contig-p (vt-zeros '(2 2))))
  (check "vt-out-writable-p 普通" t (vt-out-writable-p (vt-zeros '(2 2))))
  ;; 广播 stride=0 视图不可写（CONVENTIONS §4.2 H4）
  (check "vt-out-writable-p 广播" nil (vt-out-writable-p (vt-broadcast-to (vt-ones '(1 3)) '(4 3))))
  ;; vt-write：strides 驱动的逻辑索引写入
  (let ((o (vt-zeros '(3))))
    (vt-write o (list 1) 42.0d0)
    (check "vt-write 1d" '(0.0d0 42.0d0 0.0d0) (nlist o)))
  (let ((o (vt-zeros '(2 3))))
    (vt-write o (list 1 2) 9.0d0)
    (check "vt-write 2d" '((0.0d0 0.0d0 0.0d0) (0.0d0 0.0d0 9.0d0)) (nlist o)))
  ;; 非连续视图写入：写入位置 = offset + Σ idx*stride
  (let* ((base (vt-zeros '(2 6)))
         (v (vt-slice base '(:all) '(0 6 2))))   ; 列步长 2 的非连续视图
    (vt-write v (list 1 1) 5.0d0)              ; 逻辑 (1,1) → 物理 (1,2)
    (check "vt-write 非连续寻址" 5.0d0 (scalar (vt-ref base 1 2)))))

;;; ============================================================
;;; 11. extensions：count-nonzero / flatnonzero / inner / topk / moveaxis
;;; ============================================================
(defun test-extensions-basic ()
  (format t "~%[11] 扩展函数~%")
  (check "vt-count-nonzero" 2 (scalar (vt-count-nonzero (vt-from-sequence '(0 1 0 2)))))
  (check "vt-flatnonzero" '(1 3) (nlist (vt-flatnonzero (vt-from-sequence '(0 1 0 2)))))
  (check "vt-inner" 32.0d0 (scalar (vt-inner (vt-from-sequence '(1 2 3)) (vt-from-sequence '(4 5 6)))))
  (check "vt-topk 降序" '(4.0d0 3.0d0) (nlist (vt-topk (vt-from-sequence '(3.0 1.0 4.0 1.5)) 2)))
  ;; vt-moveaxis：期望值全部来自 numpy 2.4.6 实测
  (check "vt-moveaxis 0->2" '(3 4 2) (vt-shape (vt-moveaxis (vt-zeros '(2 3 4)) 0 2)))
  (check "vt-moveaxis 2->0" '(4 2 3) (vt-shape (vt-moveaxis (vt-zeros '(2 3 4)) 2 0)))
  (check "vt-moveaxis -1->0" '(4 2 3) (vt-shape (vt-moveaxis (vt-zeros '(2 3 4)) -1 0)))
  (check "vt-moveaxis -1->1" '(2 4 3) (vt-shape (vt-moveaxis (vt-zeros '(2 3 4)) -1 1)))
  (check "vt-moveaxis (0,1)->(2,3)" '(4 5 2 3)
         (vt-shape (vt-moveaxis (vt-zeros '(2 3 4 5)) '(0 1) '(2 3)))))

;;; ============================================================
;;; 12. extensions2：fliplr / flipud / ediff1d / indices / standardize
;;; ============================================================
(defun test-extensions2-basic ()
  (format t "~%[12] extensions2~%")
  (check "vt-fliplr" '((2.0d0 1.0d0) (4.0d0 3.0d0)) (nlist (vt-fliplr (vt-from-sequence '((1 2) (3 4))))))
  (check "vt-flipud" '((3.0d0 4.0d0) (1.0d0 2.0d0)) (nlist (vt-flipud (vt-from-sequence '((1 2) (3 4))))))
  (check-error "vt-fliplr 1D 报错" (lambda () (vt-fliplr (vt-from-sequence '(1 2 3)))))
  (check "vt-ediff1d" '(1.0d0 2.0d0 3.0d0) (nlist (vt-ediff1d (vt-from-sequence '(1 2 4 7)))))
  (check "vt-ravel-multi-index" 6 (vt-ravel-multi-index '(1 2) '(3 4)))
  (let ((ti (vt-tril-indices 3)))
    ;; 返回 (2, n) 或 2 个张量？按实现返回两个值/列表，此处断言首值
    (check "vt-tril-indices 行" '(0 1 1 2 2 2) (nlist (if (listp ti) (first ti) ti))))
  (let ((ti (vt-triu-indices 3)))
    (check "vt-triu-indices 行" '(0 0 0 1 1 2) (nlist (if (listp ti) (first ti) ti))))
  (check "vt-standardize" '(-1.224744871391589d0 0.0d0 1.224744871391589d0)
         (nlist (vt-standardize (vt-from-sequence '(1.0 2.0 3.0))))))

;;; ============================================================
;;; 13. 随机：SeedSequence / Generator 生命周期
;;; ============================================================
(defun test-random-lifecycle ()
  (format t "~%[13] 随机数底层对象~%")
  (let ((ss (make-seed-sequence 42)))
    (check "vt-seed-sequence-entropy" 42 (vt-seed-sequence-entropy ss))
    (check "seed-sequence-spawn 数量" 3 (length (seed-sequence-spawn ss 3))))
  (check "make-generator 类型" t (vt-generator-p (make-generator)))
  (check "vt-generator-state 存在" t (not (null (vt-generator-state (make-generator)))))
  ;; 同一 seed 派生的 Generator 可复现（经 with-generator 作用域宏）
  (let ((va (with-generator ((generator-from-seed-sequence (make-seed-sequence 7)))
              (vt-random-uniform '(5))))
        (vb (with-generator ((generator-from-seed-sequence (make-seed-sequence 7)))
              (vt-random-uniform '(5)))))
    (check "同 seed Generator 可复现" (nlist va) (nlist vb)))
  (check "spawn-generators 数量" 3 (length (spawn-generators 42 3))))

;;; ============================================================
;;; 14. 最小二乘（覆盖缺口）
;;; ============================================================
(defun test-lstsq ()
  (format t "~%[14] vt-lstsq~%")
  ;; 过定方程 y = 2x 精确拟合
  (let ((s (vt-lstsq (vt-from-sequence '((1.0) (2.0) (3.0))) (vt-from-sequence '(2.0 4.0 6.0)))))
    (check "vt-lstsq 解" '(2.0d0) (nlist s)))
  ;; 多列
  (let ((s (vt-lstsq (vt-from-sequence '((1.0 0.0) (0.0 1.0) (1.0 1.0)))
                     (vt-from-sequence '(3.0 5.0 8.0)))))
    (check "vt-lstsq 多列" '(3.0d0 5.0d0) (nlist s))))

;;; ============================================================
;;; 入口
;;; ============================================================
(defun run-uncovered-coverage-tests ()
  (setf *n* 0 *p* 0 *f* 0 *fails* nil)
  (format t "~&============================================================~%")
  (format t "  clvt 覆盖缺口测试套件 (uncovered-coverage-test)~%")
  (format t "============================================================~%")
  (test-binary-aliases)
  (test-trig-hyperbolic)
  (test-expt-lerp-negative)
  (test-predicates)
  (test-logic-bitwise)
  (test-creation)
  (test-concat-dims)
  (test-clamp-select)
  (test-vander)
  (test-primitives)
  (test-extensions-basic)
  (test-extensions2-basic)
  (test-random-lifecycle)
  (test-lstsq)
  (format t "~%============================================================~%")
  (format t "  Total: ~a | Pass: ~a | Fail: ~a~%" *n* *p* *f*)
  (format t "============================================================~%")
  (when *fails*
    (format t "~%Failed:~{~%  - ~a~}~%" (reverse *fails*)))
  (zerop *f*))

(run-uncovered-coverage-tests)
(sb-ext:exit :code (if (zerop *f*) 0 1))
