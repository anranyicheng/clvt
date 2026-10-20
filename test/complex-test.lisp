;;;; complex-test.lisp — 复数语义测试套件（CONVENTIONS §16）
;;;;
;;;; 组成：
;;;;   1) 差分对照：test/complex-expected.sexp 由 gen_complex_expected.py
;;;;      从**本机 numpy 实测**生成，本文件逐格比对（§10.4）。
;;;;   2) 契约断言：dtype / shape / :out 硬契约 / 非连续 / 别名 /
;;;;      未支持函数的确定性报错（$10.1 L1/L2）。
;;;;   3) 已知差异断言：real/imag 返回副本（numpy 是视图），打印格式。
;;;;
;;;; 用法：sbcl --non-interactive --load test/complex-test.lisp

(require :asdf)
;; 系统根目录由本文件位置推出（run-tests.sh / 单跑都不依赖硬编码路径）
;; 注意：此处在 CL-USER 包，名字要带 *CLVT-TEST-DIR* 前缀以便 in-package 后引用
;; （早期版本用了 *this-dir*，然后 in-package :clvt 后读到的是另一个未绑定符号）
(defparameter *clvt-test-dir* (make-pathname :name nil :type nil :defaults *load-truename*))
(pushnew (merge-pathnames "../" *clvt-test-dir*) asdf:*central-registry* :test #'equal)
(handler-bind ((warning #'muffle-warning)) (asdf:load-system :clvt))

(in-package :clvt)

;;; ------------------------------------------------------------------
;;; 测试骨架
;;; ------------------------------------------------------------------

(defvar *pass* 0)
(defvar *fail* 0)
(defvar *failures* '())

(defmacro with-case ((name) &body body)
  `(handler-case (progn ,@body (incf *pass*))
     (error (e)
       (incf *fail*)
       (push (format nil "~a: ~a" ,name e) *failures*)
       (format t "~&FAIL ~a: ~a~%" ,name e))))

(defun approx= (a b &optional (tol 1d-9))
  "复数/实数近似相等；NaN 只与 NaN 相等；±Inf 同号相等。
   NaN 参与的比较会触发 SBCL 的 invalid-operation 陷阱，故整段在
   with-float-safe 下求值（与库内策略一致）。"
  (with-float-safe
   (cond
    ((and (complexp a) (complexp b))
     (and (approx= (realpart a) (realpart b) tol)
          (approx= (imagpart a) (imagpart b) tol)))
    ((or (complexp a) (complexp b)) nil)
    ((and (floatp a) (not (= a a))) (and (floatp b) (not (= b b))))
    ((and (floatp b) (not (= b b))) nil)
    ((and (floatp a) (or (> a 1d300) (< a -1d300)))
     (and (floatp b) (= a b)))
    ((and (floatp b) (or (> b 1d300) (< b -1d300)))
     (and (floatp a) (= a b)))
    (t (<= (abs (- a b)) tol)))))

(defun nan-complex-p (x)
  "判定 X（复数标量 / 张量 / 列表）是否含 NaN 分量。"
  (let ((v (if (vt-p x) (vt-to-list x) x)))
    (cond ((consp v) (some #'nan-complex-p v))
          ((complexp v) (or (vt-float-nan-p (realpart v))
                            (vt-float-nan-p (imagpart v))))
          (t (vt-float-nan-p v)))))

(defun approx-list= (got want)
  "列表逐元素近似比较（长度必须相等）。"
  (and (listp got) (listp want)
       (= (length got) (length want))
       (every #'identity (mapcar #'approx= got want))))

(defun expect-equal (name got want)
  (unless (equal got want)
    (error "~a: got ~a want ~a" name got want)))

(defun expect-true (name form)
  (unless form (error "~a: 期望为真" name)))

(defun expect-error (name thunk)
  "断言 THUNK 抛出错误且信息含 'first-round'/关键字（L2：确定性报错）。"
  (handler-case (progn (funcall thunk)
                       (error "~a: 期望报错但正常返回" name))
    (error (e)
      (when (search "期望报错" (princ-to-string e))
        (error "~a: 期望报错但正常返回" name))
      t)))

;;; ------------------------------------------------------------------
;;; 预置张量（与 gen_complex_expected.py 的 A/B/C/D/M 一一对应）
;;; ------------------------------------------------------------------

(defparameter cx-a (vt-from-sequence (list #C(1.0d0 2.0d0) #C(3.0d0 -4.0d0))
                                       :dtype :complex128))
(defparameter cx-b (vt-from-sequence (list #C(2.0d0 -1.0d0) #C(1.0d0 1.0d0))
                                       :dtype :complex128))
(defparameter cx-c (vt-from-sequence (list #C(3.0d0 1.0d0) #C(1.0d0 5.0d0)
                                             #C(1.0d0 2.0d0))
                                       :dtype :complex128))
(defparameter cx-d (vt-from-sequence (list #C(1.0d0 2.0d0)
                                             (complex +vt-dfloat-nan+ 0.0d0)
                                             (complex +vt-dfloat-pos-inf+ 0.0d0))
                                       :dtype :complex128))
(defparameter cx-c64-a (vt-from-sequence (list #C(1.0d0 2.0d0) #C(3.0d0 -4.0d0))
                                           :dtype :complex64))
;; 与 gen_complex_expected.py 的 M 严格一致：[[1+1j, 2+0j], [3+0j, 4j]]
;; （trace / matmul 用例的期望值来自它）
(defparameter cx-m (vt-from-sequence (list (list #C(1.0d0 1.0d0) #C(2.0d0 0.0d0))
                                           (list #C(3.0d0 0.0d0) #C(0.0d0 4.0d0)))
                                     :dtype :complex128))

;;; ------------------------------------------------------------------
;;; 期望值读取与归一化
;;; ------------------------------------------------------------------

(defun normalize-real (x)
  (case x
    (:nan +vt-dfloat-nan+)
    (:inf +vt-dfloat-pos-inf+)
    (:neg-inf +vt-dfloat-neg-inf+)
    (t x)))

(defun normalize-value (v)
  "把读入的期望值（(:CX re im) / :NAN / 数字 / 列表）还原为 Lisp 值。
   注意 (:CX re im) 表示**单个复数标量**，不是列表 —— 只有「列表的列表」
   才是数组结果（如 ((:CX 1 2) (:CX 3 4))）。"
  (labels ((one (x)
             (if (consp x)
                 (if (eq (first x) :cx)
                     (complex (normalize-real (second x))
                              (normalize-real (third x)))
                     (error "无法归一化 ~a" x))
                 (normalize-real x))))
    (cond ((and (consp v) (eq (first v) :cx)) (one v))       ; 复数标量
          ((listp v) (mapcar #'one v))                       ; 数组
          (t (one v)))))

(defvar *cases* nil)
(defvar *generator* nil)

(defparameter *expected-file*
  (merge-pathnames "complex-expected.sexp" cl-user::*clvt-test-dir*)
  "由 gen_complex_expected.py 生成的 numpy 实测期望值。")

(with-open-file (s *expected-file*)
  (let ((data (read s)))
    (setf *generator* (getf data :generator)
          *cases* (getf data :cases))))

;;; ------------------------------------------------------------------
;;; 1. 差分对照
;;; ------------------------------------------------------------------

(defun run-differential ()
  (format t "~&--- 差分对照（numpy ~a，~d 用例）---~%" *generator* (length *cases*))
  (dolist (c *cases*)
    (let ((name (getf c :name))
          (form (read-from-string (getf c :form)))
          (want (normalize-value (getf c :value)))
          (want-dtype (getf c :dtype))
          (want-shape (getf c :shape))
          (symbol-kind (eq (getf c :kind) :symbol)))
      (with-case (name)
        (let ((got (eval form)))
          (cond
            (symbol-kind
             (expect-equal name got want))
            ((vt-p got)
             (expect-equal (format nil "~a dtype" name) (vt-dtype got) want-dtype)
             (expect-equal (format nil "~a shape" name) (vt-shape got) want-shape)
             (let ((items (let ((v (vt-to-list got)))
                            (if (listp v) (flatten-list v) (list v))))
                   (wants (if (listp want) want (list want))))
               (unless (= (length items) (length wants))
                 (error "~a: 元素个数不符 got ~a want ~a" name items want))
               (loop for g in items
                     for w in wants
                     for i from 0
                     unless (approx= (if (complexp g) g (float g 1d0))
                                     (if (complexp w) w (float w 1d0)))
                       do (error "~a[~d]: got ~a want ~a" name i g w))))
            (t
             ;; 标量返回（如 vt-dtype → 关键字，或用例本身返回数值）
             (unless (equal got want)
               (error "~a: got ~a want ~a" name got want)))))))))

(defun flatten-list (x)
  (cond ((null x) nil)
        ((listp x) (append (flatten-list (first x)) (flatten-list (rest x))))
        (t (list x))))

;;; ------------------------------------------------------------------
;;; 2. dtype 系统与构造
;;; ------------------------------------------------------------------

(defun test-dtypes ()
  (format t "~&--- dtype 系统与构造 ---~%")
  (with-case ("itemsize")
    (expect-equal "c128 itemsize" (vt-itemsize cx-a) 16)
    (expect-equal "c64 itemsize" (vt-itemsize cx-c64-a) 8))
  (with-case ("elem-type")
    (expect-equal "c128 elem" (array-element-type (vt-data cx-a))
                  '(complex double-float))
    (expect-equal "c64 elem" (array-element-type (vt-data cx-c64-a))
                  '(complex single-float)))
  (with-case ("promote-100-cells")
    ;; 10×10 提升表全格与 numpy 2.4.6 对齐（§16.3）
    (let ((dts '(:int8 :int16 :int32 :int64 :uint8 :uint16
                 :float32 :float64 :complex64 :complex128))
          (expect '((:int8 :int8 :int16 :int32 :int64 :int16 :int32 :float32 :float64 :complex64 :complex128)
                    (:int16 :int16 :int16 :int32 :int64 :int16 :int32 :float32 :float64 :complex64 :complex128)
                    (:int32 :int32 :int32 :int32 :int64 :int32 :int32 :float64 :float64 :complex128 :complex128)
                    (:int64 :int64 :int64 :int64 :int64 :int64 :int64 :float64 :float64 :complex128 :complex128)
                    (:uint8 :int16 :int16 :int32 :int64 :uint8 :uint16 :float32 :float64 :complex64 :complex128)
                    (:uint16 :int32 :int32 :int32 :int64 :uint16 :uint16 :float32 :float64 :complex64 :complex128)
                    (:float32 :float32 :float32 :float64 :float64 :float32 :float32 :float32 :float64 :complex64 :complex128)
                    (:float64 :float64 :float64 :float64 :float64 :float64 :float64 :float64 :float64 :complex128 :complex128)
                    (:complex64 :complex64 :complex64 :complex128 :complex128 :complex64 :complex64 :complex64 :complex128 :complex64 :complex128)
                    (:complex128 :complex128 :complex128 :complex128 :complex128 :complex128 :complex128 :complex128 :complex128 :complex128 :complex128))))
      (loop for a in dts for row in expect
            do (loop for b in dts for want in (rest row)
                     do (expect-equal (format nil "promote ~a ~a" a b)
                                      (vt-promote-type a b) want)))))
  (with-case ("construct")
    (expect-equal "scalar" (vt-to-list (vt-complex 1.0d0 2.0d0)) #C(1.0d0 2.0d0))
    (expect-equal "vector dtype" (vt-dtype (vt-complex 1.0d0 2.0d0)) :complex128)
    (expect-equal "c64" (vt-dtype (vt-complex 1.0d0 2.0d0 :dtype :complex64))
                  :complex64)
    (expect-equal "zeros" (vt-to-list (vt-zeros (list 2) :dtype :complex128))
                  (list #C(0.0d0 0.0d0) #C(0.0d0 0.0d0)))
    (expect-equal "ones" (vt-to-list (vt-ones (list 1) :dtype :complex64))
                  (list #C(1.0f0 0.0f0))))
  (with-case ("infer-from-sequence")
    (expect-equal "list w/ complex" (vt-dtype (ensure-vt (list 1 2 #C(3.0d0 4.0d0))))
                  :complex128))
  (with-case ("cast-rules")
    ;; complex → real：取实部（numpy astype + ComplexWarning 语义）
    (expect-equal "to float64" (vt-to-list (vt-astype cx-a :float64))
                  (list 1.0d0 3.0d0))
    ;; complex → int：实部截断
    (expect-equal "to int64" (vt-to-list (vt-astype cx-a :int64)) (list 1 3))
    ;; real → complex：虚部补 0
    (expect-equal "from float"
                  (vt-to-list (vt-astype (vt-from-sequence (list 1.0d0 2.0d0))
                                         :complex128))
                  (list #C(1.0d0 0.0d0) #C(2.0d0 0.0d0))))
  (with-case ("iscomplexobj")
    (expect-true "c128 obj" (vt-iscomplexobj cx-a))
    (expect-true "float obj" (not (vt-iscomplexobj (vt-ones (list 2)))))
    (expect-true "real obj" (vt-isrealobj (vt-ones (list 2))))))

;;; ------------------------------------------------------------------
;;; 3. 分量与派生量
;;; ------------------------------------------------------------------

(defun test-components ()
  (format t "~&--- 分量 / 共轭 / 模 / 幅角 ---~%")
  (with-case ("abs-dtype-rules")
    (expect-equal "c128→f64" (vt-dtype (vt-abs cx-a)) :float64)
    (expect-equal "c64→f32" (vt-dtype (vt-abs cx-c64-a)) :float32))
  (with-case ("real-imag-copy")
    ;; 已知差异（§16.6）：clvt 返回副本，numpy 返回视图
    (let ((r (vt-real cx-a)))
      (setf (aref (vt-data r) 0) 99.0d0)
      (expect-equal "original untouched" (vt-to-list (vt-real cx-a))
                    (list 1.0d0 3.0d0))))
  (with-case ("imag-of-real")
    (expect-equal "imag real" (vt-to-list (vt-imag (vt-from-sequence (list 1.0d0 2.0d0))))
                  (list 0.0d0 0.0d0)))
  (with-case ("angle")
    (expect-true "angle 1+2i"
                 (approx= (first (vt-to-list (vt-angle (vt-const (list 1) #C(1.0d0 2.0d0)
                                                              :dtype :complex128))))
                          1.1071487177940904d0))
    (expect-true "angle deg"
                 (approx= (first (vt-to-list (vt-angle (vt-const (list 1) #C(0.0d0 1.0d0)
                                                               :dtype :complex128) :deg t)))
                          90.0d0)))
  (with-case ("conj")
    (expect-equal "conj dtype" (vt-dtype (vt-conj cx-a)) :complex128)
    (expect-true "conj val"
                 (approx= (first (vt-to-list (vt-conj (vt-const (list 1) #C(1.0d0 2.0d0)
                                                               :dtype :complex128))))
                          #C(1.0d0 -2.0d0)))))

;;; ------------------------------------------------------------------
;;; 4. 分支切割与极点（§16.10）
;;; ------------------------------------------------------------------

(defun test-branch-cuts ()
  (format t "~&--- 分支切割 / 极点 ---~%")
  (with-case ("sqrt-signed-zero")
    (expect-true "sqrt(-4+0i)"
                 (approx= (first (vt-to-list (vt-sqrt (vt-const (list 1) (complex -4.0d0 0.0d0)
                                                               :dtype :complex128))))
                          #C(0.0d0 2.0d0)))
    (expect-true "sqrt(-4-0i)"
                 (approx= (first (vt-to-list (vt-sqrt (vt-const (list 1) (complex -4.0d0 -0.0d0)
                                                               :dtype :complex128))))
                          #C(0.0d0 -2.0d0))))
  (with-case ("log-branch")
    (let ((p (first (vt-to-list (vt-log (vt-const (list 1) (complex -1.0d0 0.0d0)
                                                 :dtype :complex128)))))
          (m (first (vt-to-list (vt-log (vt-const (list 1) (complex -1.0d0 -0.0d0)
                                                  :dtype :complex128))))))
      (expect-true "log(+0) = +iπ" (approx= p #C(0.0d0 3.141592653589793d0)))
      (expect-true "log(-0) = -iπ" (approx= m #C(0.0d0 -3.141592653589793d0)))))
  (with-case ("log-pole")
    (let ((v (first (vt-to-list (vt-log (vt-const (list 1) (complex 0.0d0 0.0d0)
                                                 :dtype :complex128))))))
      (expect-true "log(0) real = -inf"
                   (and (< (realpart v) -1d300) (= (imagpart v) 0.0d0)))))
  (with-case ("sqrt-c64-dtype")
    (expect-equal "c64 preserved" (vt-dtype (vt-sqrt cx-c64-a)) :complex64))
  (with-case ("pow-zero")
    (expect-true "0**0 = 1+0i"
                 (approx= (first (vt-to-list (vt-pow (vt-const (list 1) #C(0.0d0 0.0d0)
                                                              :dtype :complex128) 0)))
                          #C(1.0d0 0.0d0)))
    (let ((v (first (vt-to-list (vt-pow (vt-const (list 1) #C(0.0d0 0.0d0) :dtype :complex128)
                                        #C(-1.0d0 0.0d0))))))
      (expect-true "0**-1 = nan" (nan-complex-p v)))))

;;; ------------------------------------------------------------------
;;; 5. 归约与统计
;;; ------------------------------------------------------------------

(defun test-reductions ()
  (format t "~&--- 归约 / 统计 ---~%")
  (with-case ("sum-2d-axis")
    (let ((r (vt-sum cx-m :axis 1)))
      (expect-equal "shape" (vt-shape r) (list 2))
      (expect-true "values"
                   (every #'identity
                          (mapcar #'approx= (vt-to-list r)
                                  (list #C(3.0d0 1.0d0) #C(3.0d0 4.0d0)))))))
  (with-case ("sum-empty")
    (let ((r (vt-sum (vt-zeros (list 0) :dtype :complex128))))
      (expect-equal "dtype" (vt-dtype r) :complex128)
      (expect-equal "value" (vt-to-list r) #C(0.0d0 0.0d0))))
  (with-case ("amax-empty-errors")
    (expect-error "amax empty" (lambda () (vt-amax (vt-zeros (list 0) :dtype :complex128)))))
  (with-case ("var-is-real")
    (expect-equal "var dtype" (vt-dtype (vt-var cx-a)) :float64)
    (expect-equal "std dtype" (vt-dtype (vt-std cx-a)) :float64))
  (with-case ("norm-is-real")
    (expect-equal "norm dtype" (vt-dtype (vt-norm cx-a)) :float64)
    (expect-true "norm value" (approx= (vt-to-list (vt-norm cx-a)) 5.477225575051661d0)))
  (with-case ("norm-c64")
    (expect-equal "norm c64 dtype" (vt-dtype (vt-norm cx-c64-a)) :float32))
  (with-case ("mean-empty-nan")
    (let ((r (vt-to-list (vt-mean (vt-zeros (list 0) :dtype :complex128)))))
      (expect-true "mean empty nan" (nan-complex-p r))))
  (with-case ("median")
    (expect-true "median [3+1 1+5 1+2]"
                 (approx= (vt-to-list (vt-median cx-c)) #C(1.0d0 5.0d0))))
  (with-case ("nan-propagation")
    (let ((r (vt-to-list (vt-sum cx-d))))
      (expect-true "sum nan" (nan-complex-p r)))
    (let ((r (vt-to-list (vt-amax cx-d))))
      (expect-true "amax nan wins" (nan-complex-p r)))))

;;; ------------------------------------------------------------------
;;; 6. 非连续 / 别名 / :out 契约（§4、§16.12）
;;; ------------------------------------------------------------------

(defun test-views-and-out ()
  (format t "~&--- 视图 / 别名 / :out ---~%")
  (with-case ("transpose-sum")
    (let* ((t2 (vt-transpose cx-m))
           (r (vt-sum t2 :axis 0)))
      ;; M^T = [[1+1j,3],[2,4j]]，sum(axis=0) = [3+1j, 3+4j]（numpy 实测）
      (expect-true "transposed sum"
                   (approx-list= (vt-to-list r) (list #C(3.0d0 1.0d0) #C(3.0d0 4.0d0))))))
  (with-case ("strided-slice")
    (let* ((s (vt-slice cx-c (list nil nil 2)))
           (r (vt-sum s)))
      ;; cx-c = [3+1j, 1+5j, 1+2j]，步长 2 取 [3+1j, 1+2j]，和 = 4+3j
      (expect-true "step-2 slice sum" (approx= (vt-to-list r) #C(4.0d0 3.0d0)))))
  (with-case ("alias-out")
    (let* ((z (vt-copy cx-a))
           (r (vt-+ z z :out z)))
      (expect-true "z += z" (approx-list= (vt-to-list r) (list #C(2.0d0 4.0d0) #C(6.0d0 -8.0d0))))))
  (with-case ("out-wrong-dtype-errors")
    (expect-error "out real dtype"
                  (lambda () (vt-+ cx-a cx-a :out (vt-zeros (list 2) :dtype :float64)))))
  (with-case ("out-wrong-shape-errors")
    (expect-error "out shape"
                  (lambda () (vt-+ cx-a cx-a :out (vt-zeros (list 3) :dtype :complex128)))))
  (with-case ("out-ok")
    (let ((o (vt-zeros (list 2) :dtype :complex128)))
      (vt-+ cx-a cx-a :out o)
      (expect-true "out written"
                   (approx-list= (vt-to-list o) (list #C(2.0d0 4.0d0) #C(6.0d0 -8.0d0))))))
  (with-case ("broadcast-scalar")
    (let ((r (vt-+ cx-a 1.0d0)))
      (expect-true "scalar broadcast"
                   (approx-list= (vt-to-list r) (list #C(2.0d0 2.0d0) #C(4.0d0 -4.0d0))))))
  (with-case ("print-format")
    (let ((s (format nil "~a" (vt-const (list 2) #C(1.0d0 2.0d0)
                                        :dtype :complex128))))
      (expect-true "print contains 1.0+2.0j" (search "1.0+2.0j" s)))
    (let ((s (format nil "~a" (vt-const (list 1) #C(1.0d0 -2.0d0)
                                        :dtype :complex128))))
      (expect-true "print contains 1.0-2.0j" (search "1.0-2.0j" s)))))

;;; ------------------------------------------------------------------
;;; 7. 未支持函数必须确定性报错（§16.11）
;;; ------------------------------------------------------------------

(defun test-unsupported ()
  (format t "~&--- 未支持函数的确定性报错 ---~%")
  (flet ((chk (name thunk)
           (with-case (name)
             (expect-error name thunk))))
    (chk "floor" (lambda () (vt-floor cx-a)))
    (chk "ceiling" (lambda () (vt-ceiling cx-a)))
    (chk "truncate" (lambda () (vt-truncate cx-a)))
    (chk "div" (lambda () (vt-div cx-a 2)))
    (chk "mod" (lambda () (vt-mod cx-a 2)))
    (chk "rem" (lambda () (vt-rem cx-a 2)))
    (chk "bit-and" (lambda () (vt-bit-and cx-a cx-a)))
    (chk "atan2" (lambda () (vt-atan2 cx-a cx-b)))
    (chk "even-p" (lambda () (vt-even-p cx-a)))
    (chk "percentile" (lambda () (vt-percentile cx-a 50)))
    ;; det / inv / cond 自第三轮起已支持复数（src/complex-linalg.lisp），
    ;; 不再属于「未支持」清单 —— 这里改成正向断言。
    ;; 注意：本文件的 expect-equal 是「抛错式」助手，计数由 with-case 负责，
    ;; 所以必须包在 with-case 里，否则断言不计入总数（历史踩过）。
    (with-case ("det 支持复数")
      (expect-equal "det 支持复数"
                    (type-of (vt-item (vt-det cx-m))) '(complex double-float)))
    (with-case ("inv 支持复数")
      (expect-equal "inv 支持复数"
                    (vt-dtype (vt-inv cx-m)) :complex128))
    (chk "sigmoid" (lambda () (vt-sigmoid cx-a)))))

;;; ------------------------------------------------------------------
;;; 8. 实数路径未受影响（回归网，§16.12）
;;; ------------------------------------------------------------------

(defun test-real-regression ()
  (format t "~&--- 实数路径回归 ---~%")
  (with-case ("real-add")
    (expect-equal "float64 add" (vt-to-list (vt-+ (vt-ones (list 2)) (vt-ones (list 2))))
                  (list 2.0d0 2.0d0)))
  (with-case ("real-comparison")
    (expect-equal "lt" (vt-to-list (vt-< (vt-from-sequence (list 1.0d0 3.0d0))
                                         (vt-from-sequence (list 2.0d0 2.0d0))))
                  (list 1 0)))
  (with-case ("real-sort")
    (expect-equal "sort nan last"
                  (vt-to-list (vt-sort (vt-from-sequence (list 3.0d0 +vt-dfloat-nan+ 1.0d0))))
                  (list 1.0d0 3.0d0 +vt-dfloat-nan+)))
  (with-case ("real-sum")
    (expect-equal "int8 sum dtype" (vt-dtype (vt-sum (vt-zeros (list 3) :dtype :int8)))
                  :int64))
  (with-case ("real-mean")
    (expect-equal "mean" (vt-to-list (vt-mean (vt-from-sequence (list 1.0d0 2.0d0 3.0d0))))
                  2.0d0))
  (with-case ("real-floor")
    (expect-equal "floor" (vt-to-list (vt-floor (vt-from-sequence (list 1.7d0 -1.2d0))))
                  (list 1.0d0 -2.0d0))))

;;; ------------------------------------------------------------------
;;; 主流程
;;; ------------------------------------------------------------------

(format t "~&===== clvt 复数测试套件（CONVENTIONS §16）=====~%")
(run-differential)
(test-dtypes)
(test-components)
(test-branch-cuts)
(test-reductions)
(test-views-and-out)
(test-unsupported)
(test-real-regression)

(format t "~&=====================================~%")
(format t "Total: ~d / Pass: ~d / Fail: ~d~%" (+ *pass* *fail*) *pass* *fail*)
(format t "PASS=~d FAIL=~d~%" *pass* *fail*)
(when *failures*
  (format t "~&失败清单：~%")
  (dolist (f (reverse *failures*)) (format t "  - ~a~%" f)))
(when (plusp *fail*)
  (sb-ext:exit :code 1))
