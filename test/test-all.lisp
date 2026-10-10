;;;; test-all.lisp — clvt 全量导出函数测试套件
;;;;
;;;; 契约基准：CONVENTIONS.md（凡与 NumPy 冲突者一律以 NumPy 为准；
;;;; 唯一例外 vt-arange，见 CONVENTIONS.md §9.1）。
;;;; 覆盖 :clvt 包全部导出函数（378 个符号），按模块分组逐一验证：
;;;;   1. 物理层访问器与视图   2. 创建函数      3. 形状操作
;;;;   4. 索引与定位           5. 逐元素算术    6. 比较/逻辑/位运算
;;;;   7. 归约与统计           8. 线性代数      9. 神经网络层
;;;;  10. 集合操作            11. 随机数       12. 参数契约(:out)
;;;;  13. NaN/Inf 基础设施    14. 映射/迭代原语 15. 图像旋转
;;;;
;;;; 运行方式（clvt 已加载后）：
;;;;   (load "test/test-all.lisp")
;;;;   (run-all-tests)          ; 或 (run-tests 'test-core-accessors)
;;;;
;;;; 返回值：全部通过 → T；存在失败 → NIL（并在 stdout 打印明细）。

(require :asdf)
#+quicklisp (ql:quickload :clvt :silent t)
(asdf:load-system :clvt)
(in-package :clvt)

;;; ==================================================================
;;; 测试基础设施
;;; ==================================================================

(defparameter *assert-pass* 0 "已通过的断言数。")
(defparameter *assert-fail* 0 "已失败的断言数。")
(defparameter *current-failures* nil "当前测试的失败明细（倒序）。")
(defparameter *registered-tests* nil "按注册顺序的测试函数名。")

(defmacro check (desc form)
  "断言 FORM 求值为非 NIL。失败时记录 DESC 与错误信息，继续执行后续断言。"
  (let ((e (gensym "E")))
    `(handler-case
         (progn
           (assert ,form ()
                   "断言失败: ~a~%      表达式: ~s 求值为 NIL"
                   ,desc ',form)
           (incf *assert-pass*))
       (error (,e)
         (incf *assert-fail*)
         (push (format nil "    [FAIL] ~a~%           ↳ ~a" ,desc ,e)
               *current-failures*)))))

(defmacro check-error (desc form)
  "期望 FORM 求值时发出 ERROR（契约规定的报错行为）。"
  (let ((ok (gensym "OK")))
    `(progn
       (let ((,ok nil))
         (handler-case (progn ,form (setf ,ok t))
           (error () nil))
         (if ,ok
             (progn
               (incf *assert-fail*)
               (push (format nil "    [FAIL] ~a（预期报错，实际成功返回）" ,desc)
                     *current-failures*))
             (incf *assert-pass*))))))

(defmacro check-not-error (desc form)
  "期望 FORM 求值不发出错误（用于覆盖崩溃类回归）。"
  (let ((ok (gensym "OK")))
    `(progn
       (let ((,ok nil))
         (handler-case (progn ,form (setf ,ok t))
           (error (e) (push (format nil "    [FAIL] ~a（预期不报错）↳ ~a" ,desc e)
                            *current-failures*))))
       (if ,ok (incf *assert-pass*) (incf *assert-fail*)))))

;;; ---------- 数值比较（NaN 感知 + 容差） ----------

(defun %feq (a b &optional (rtol 1d-9) (atol 1d-9))
  "NaN 感知、容差比较：NaN==NaN、+Inf==+Inf、-Inf==-Inf；其余按 rtol/atol。"
  (cond ((and (numberp a) (numberp b))
         (let ((da (coerce a 'double-float))
               (db (coerce b 'double-float)))
           (cond ((and (vt-float-nan-p da) (vt-float-nan-p db)) t)
                 ((or (vt-float-nan-p da) (vt-float-nan-p db)) nil)
                 ((and (vt-float-pos-inf-p da) (vt-float-pos-inf-p db)) t)
                 ((and (vt-float-neg-inf-p da) (vt-float-neg-inf-p db)) t)
                 ((or (vt-float-inf-p da) (vt-float-inf-p db)) nil)
                 (t (<= (abs (- da db))
                        (+ atol (* rtol (max (abs da) (abs db)))))))))
        ((and (numberp a) (null b)) nil)
        (t (eql a b))))

(defun %flat (obj)
  "把 vt / 列表 / 向量 / 标量展平为数值列表（行主序）。"
  (cond ((vt-p obj) (vt-flatten-sequence (vt-to-list obj)))
        ((or (listp obj) (vectorp obj)) (vt-flatten-sequence obj))
        (t (list obj))))

(defun allclose-lists (got expected &optional (rtol 1d-9) (atol 1d-9))
  "逐元素比较（NaN 感知），长度也必须相等。"
  (let ((g (%flat got)) (e (%flat expected)))
    (and (= (length g) (length e))
         (loop for x in g for y in e always (%feq x y rtol atol)))))

(defmacro check-eq (desc got &rest expected)
  "逐元素数值比较；EXPECTED 为一个列表/张量，或多个标量（自动收集为列表）。"
  (let ((exp (if (= (length expected) 1) (first expected) `(list ,@expected))))
    `(check ,desc (allclose-lists ,got ,exp 1d-9 1d-9))))

(defmacro check-shape (desc vt expected)
  `(check ,desc (equal (vt-shape ,vt) ,expected)))

(defmacro check-dtype (desc vt expected)
  `(check ,desc (eq (vt-dtype ,vt) ,expected)))

(defmacro check-item (desc vt expected)
  `(check ,desc (let ((n (vt-item ,vt)))
                  (if (numberp n)
                      (%feq n ,expected)
                      (eql n ,expected)))))

;;; ---------- 测试注册与运行 ----------

(defmacro deftest (name docstring &body body)
  `(progn
     (defun ,name ()
       ,docstring
       (setf *current-failures* nil)
       ,@body
       (nreverse *current-failures*))
     (pushnew ',name *registered-tests*)))

(defun run-tests (&rest names)
  "运行指定测试（缺省为全部），打印明细，返回 (values 通过数 失败数)。"
  (let ((todo (or names (reverse *registered-tests*)))
        (failed-tests 0)
        (passed-tests 0))
    (format t "~&==================== clvt test-all ====================~%")
    (dolist (n todo)
      (let ((fn (find-symbol (symbol-name n) :clvt)))
        (unless (and fn (fboundp fn))
          (error "未找到测试: ~a" n))
        (let* ((fails (handler-case (funcall fn)
                        (error (e)
                          (list (format nil "    ✗ 测试函数本身抛出未处理错误: ~a" e)))))
               (npass *assert-pass*))
          (if fails
              (progn
                (incf failed-tests)
                (format t "~%[FAIL] ~a~%" n)
                (dolist (f fails) (format t "~a~%" f)))
              (progn
                (incf passed-tests)
                (format t "[PASS] ~a (~d 断言)~%" n npass))))))
    (format t "~%---------------- 汇总 ----------------~%")
    (format t "测试: ~d 通过 / ~d 失败   断言: ~d 通过 / ~d 失败~%"
            passed-tests failed-tests *assert-pass* *assert-fail*)
    (format t "结果: ~a~%"
            (if (zerop failed-tests) "====> 全部通过 <====" "====> 存在失败 <===="))
    (format t "~&Total: ~d~%Pass: ~d~%Fail: ~d~%"
        (+ passed-tests failed-tests) passed-tests failed-tests) 
    (values passed-tests failed-tests)))

;;; ==================================================================
;;; 1. 物理层访问器与视图（core.lisp）
;;; ==================================================================

(deftest test-core-accessors
    "core: vt-p/shape/strides/offset/data/dtype/element-type/order/size/itemsize/nbytes/contiguous-p"
  (let ((a (vt-from-sequence '((1.0d0 2.0d0 3.0d0) (4.0d0 5.0d0 6.0d0)))))
    (check "vt-p 对 vt 张量返回 T" (vt-p a))
    (check "vt-p 对标量返回 NIL" (not (vt-p 3)))
    (check-shape "vt-shape 返回 (2 3)" a '(2 3))
    (check "vt-strides C 连续为 (3 1)" (equal (vt-strides a) '(3 1)))
    (check "vt-offset 起点为 0" (= (vt-offset a) 0))
    (check "vt-dtype 为 :float64" (eq (vt-dtype a) :float64))
    (check "vt-element-type 为 double-float" (eq (vt-element-type a) 'double-float))
    (check "vt-order 秩为 2" (= (vt-order a) 2))
    (check "vt-size 为 6" (= (vt-size a) 6))
    (check "vt-itemsize float64 为 8 字节" (= (vt-itemsize a) 8))
    (check "vt-nbytes 为 48 字节" (= (vt-nbytes a) 48))
    (check "vt-contiguous-p 连续张量为 T" (vt-contiguous-p a))
    (let ((s (vt-slice a '(0 nil) '(1 nil))))
      (check "切片视图 vt-p 仍为 T" (vt-p s))
      (check-shape "切片视图形状 (2 2)" s '(2 2))
      (check-eq "切片视图数据正确" (vt-to-list s) '((2.0d0 3.0d0) (5.0d0 6.0d0)))
      (check "切片视图 offset 为 1" (= (vt-offset s) 1))
      (check "切片视图非 C 连续" (not (vt-contiguous-p s))))
    (let ((i (vt-from-sequence '(1 2 3 4) :dtype :int32)))
      (check-dtype "from-sequence :int32 生效" i :int32)
      (check "int32 itemsize 为 4" (= (vt-itemsize i) 4))
      (check "int32 nbytes 为 16" (= (vt-nbytes i) 16))
      (check "int32 element-type 为 (signed-byte 32)"
             (equal (vt-element-type i) '(signed-byte 32))))))

(deftest test-core-shape-utils
    "core: vt-shape-to-size / vt-compute-strides / vt-compute-logical-strides / vt-normalize-axis"
  (check "vt-shape-to-size (2 3 4)=24" (= (vt-shape-to-size '(2 3 4)) 24))
  (check "vt-shape-to-size 空形状=1" (= (vt-shape-to-size nil) 1))
  (check "vt-compute-strides (2 3 4)=(12 4 1)"
         (equal (vt-compute-strides '(2 3 4)) '(12 4 1)))
  (check "vt-compute-logical-strides (2 3)=(3 1)"
         (equal (vt-compute-logical-strides '(2 3)) '(3 1)))
  (check "vt-normalize-axis -1→1 (rank 2)" (= (vt-normalize-axis -1 2) 1))
  (check "vt-normalize-axis 1→1" (= (vt-normalize-axis 1 2) 1))
  (check-error "vt-normalize-axis 越界报错" (vt-normalize-axis 2 2))
  (check-error "vt-normalize-axis -3 越界报错" (vt-normalize-axis -3 2)))

(deftest test-core-broadcast
    "core: vt-broadcast-shapes / vt-broadcast-strides（广播约定 §2.1）"
  (check "广播 (3 1 5)+(2 3 1 1)→(2 3 1 5)"
         (equal (vt-broadcast-shapes '(3 1 5) '(2 3 1 1)) '(2 3 1 5)))
  (check "广播 秩不等 (3 1)+(4,)→(3 4)"
         (equal (vt-broadcast-shapes '(3 1) '(4)) '(3 4)))
  (check "广播 (2 3)+(2 3)→(2 3)"
         (equal (vt-broadcast-shapes '(2 3) '(2 3)) '(2 3)))
  (check "广播 0 轴 vs 1 轴 → 该轴为 0（NumPy 对齐）"
         (equal (vt-broadcast-shapes '(0 3) '(1 3)) '(0 3)))
  (check-error "广播 0 轴 vs 非0非1 轴报错（NumPy 对齐）"
               (vt-broadcast-shapes '(3 0) '(1 4)))
  (check-error "双方均非 1 且不等报错" (vt-broadcast-shapes '(2 3) '(4 3)))
  (check "vt-broadcast-strides (2 3)→(4 2 3) 得 (0 3 1)"
         (equal (vt-broadcast-strides '(2 3) '(4 2 3) (vt-compute-strides '(2 3)))
                '(0 3 1))))

(deftest test-core-copy-fill
    "core: vt-copy / vt-copy-into! / vt-fill / vt-fill 非连续视图"
  (let* ((a (vt-from-sequence '(1.0d0 2.0d0 3.0d0)))
         (c (vt-copy a)))
    (check "vt-copy 数据一致" (allclose-lists (vt-to-list c) '(1.0d0 2.0d0 3.0d0)))
    (check "vt-copy 与原张量不共享存储"
           (not (eq (vt-data a) (vt-data c))))
    (vt-fill c 9.0d0)
    (check "vt-copy 修改不影响原张量" (= (vt-item (vt-ref a 0)) 1.0d0))
    (check-eq "vt-fill 填充连续张量" (vt-to-list c) '(9.0d0 9.0d0 9.0d0)))
  (let* ((base (vt-zeros '(2 4)))
         (v (vt-slice base '(:all) '(0 nil 2))))
    (vt-fill v 7.0d0)
    (check-eq "vt-fill 支持非连续视图（写入底层数据）"
              (vt-to-list base) '((7.0d0 0.0d0 7.0d0 0.0d0)
                                  (7.0d0 0.0d0 7.0d0 0.0d0))))
  (let ((a (vt-ones '(3))) (dst (vt-zeros '(3))))
    (vt-copy-to! dst a)
    (check-eq "vt-copy-to! 复制数据" (vt-to-list dst) '(1.0d0 1.0d0 1.0d0))
    (check "vt-copy-to! 返回 dst" (eq (vt-copy-to! dst a) dst))))

;;; ==================================================================
;;; 2. 创建函数（creation.lisp / io.lisp）
;;; ==================================================================

(deftest test-creation-basic
    "creation: vt-zeros / ones / full / empty / const / *-like"
  (check-shape "vt-zeros (2 3)" (vt-zeros '(2 3)) '(2 3))
  (check-eq "vt-zeros 全 0" (vt-to-list (vt-zeros '(2 2))) '((0.0d0 0.0d0) (0.0d0 0.0d0)))
  (check-dtype "vt-zeros 默认 float64" (vt-zeros '(2)) :float64)
  (check-dtype "vt-zeros :dtype :int32" (vt-zeros '(2) :dtype :int32) :int32)
  (check-eq "vt-ones 全 1" (vt-to-list (vt-ones '(3))) '(1.0d0 1.0d0 1.0d0))
  (check-eq "vt-full 填充 2.5" (vt-to-list (vt-full '(2) 2.5d0)) '(2.5d0 2.5d0))
  (check-dtype "vt-full :dtype :uint8" (vt-full '(2) 1 :dtype :uint8) :uint8)
  (let ((e (vt-empty '(4))))
    (check-shape "vt-empty 形状正确" e '(4))
    (check "vt-empty 尺寸正确" (= (vt-size e) 4)))
  (check-eq "vt-const 填充" (vt-to-list (vt-const '(2 2) 3 :dtype :int64))
            '((3 3) (3 3)))
  (check "vt-empty-like 形状继承" (equal (vt-shape (vt-empty-like (vt-ones '(2 3)))) '(2 3)))
  (check-eq "vt-zeros-like 全 0" (vt-to-list (vt-zeros-like (vt-ones '(3))))
            '(0.0d0 0.0d0 0.0d0))
  (check-eq "vt-ones-like 全 1" (vt-to-list (vt-ones-like (vt-ones '(2))))
            '(1.0d0 1.0d0))
  (check-eq "vt-full-like 填充 7" (vt-to-list (vt-full-like (vt-ones '(2)) 7.0d0))
            '(7.0d0 7.0d0))
  (check-dtype "vt-zeros-like 继承 dtype"
               (vt-zeros-like (vt-ones '(2) :dtype :int8)) :int8))

(deftest test-creation-arange-linspace
    "creation: vt-arange（§9.1 唯一例外）/ vt-linspace / vt-logspace / vt-geomspace"
  (check-eq "vt-arange(5) = 0..4" (vt-to-list (vt-arange 5)) '(0.0d0 1.0d0 2.0d0 3.0d0 4.0d0))
  (check-eq "vt-arange(3,start 1,step 2) = 1,3,5（count 语义，NumPy 唯一例外）"
            (vt-to-list (vt-arange 3 :start 1 :step 2)) '(1.0d0 3.0d0 5.0d0))
  (check-dtype "vt-arange :dtype :int64" (vt-arange 3 :dtype :int64) :int64)
  ;; 回绕语义（§4.4 表格 & §9.1）：int8 溢出回绕
  (let ((a (vt-arange 300 :dtype :int8)))
    (check "vt-arange int8 尺寸 300" (= (vt-size a) 300))
    (check-item "vt-arange int8 回绕: 第 300-1 项 = 43" (vt-ref a 299) 43))
  (check-eq "vt-linspace(0,1,5) 端点含 1"
            (vt-to-list (vt-linspace 0.0d0 1.0d0 5)) '(0.0d0 0.25d0 0.5d0 0.75d0 1.0d0))
  (check-eq "vt-linspace endpoint=nil"
            (vt-to-list (vt-linspace 0.0d0 1.0d0 4 :endpoint nil))
            '(0.0d0 0.25d0 0.5d0 0.75d0))
  (check-dtype "vt-linspace :dtype :float32" (vt-linspace 0 1 3 :dtype :float32) :float32)
  (check "vt-linspace float32 以 double 计算后舍入存储（§9.3）"
         (let ((f (vt-linspace 0.0d0 1.0d0 3 :dtype :float32)))
           (allclose-lists (vt-to-list f) '(0.0 0.5 1.0))))
  (check-eq "vt-logspace(0,2,3) = 1,10,100"
            (vt-to-list (vt-logspace 0.0d0 2.0d0 3)) '(1.0d0 10.0d0 100.0d0))
  (check-eq "vt-geomspace(1,100,3) = 1,10,100"
            (vt-to-list (vt-geomspace 1.0d0 100.0d0 3)) '(1.0d0 10.0d0 100.0d0)))

(deftest test-creation-eye-diag
    "creation: vt-eye / vt-identity / vt-diag / vt-tri / vt-diagflat / vt-vander"
  (check-eq "vt-eye(2) 单位阵"
            (vt-to-list (vt-eye 2)) '((1.0d0 0.0d0) (0.0d0 1.0d0)))
  (check-eq "vt-eye(2,cols 3,k 1) 对标 np.eye(2,3,k=1)"
            (vt-to-list (vt-eye 2 :cols 3 :k 1))
            '((0.0d0 1.0d0 0.0d0) (0.0d0 0.0d0 1.0d0)))
  (check-dtype "vt-identity :dtype :int64" (vt-identity 3 :dtype :int64) :int64)
  (check-shape "vt-identity(3) 形状 (3 3)" (vt-identity 3) '(3 3))
  (check-eq "vt-diag([1,2,3],k=1) 对标 np.diag"
            (vt-to-list (vt-diag (vt-from-sequence '(1 2 3) :dtype :int64) :k 1))
            '((0 1 0 0) (0 0 2 0) (0 0 0 3) (0 0 0 0)))
  (check-eq "vt-tri(2) 下三角"
            (vt-to-list (vt-tri 2))
            '((1.0d0 0.0d0) (1.0d0 1.0d0)))
  (check-eq "vt-diagflat([1,2],k=1)"
            (vt-to-list (vt-diagflat (vt-from-sequence '(1 2) :dtype :int64) :k 1))
            '((0 1 0) (0 0 2) (0 0 0)))
  (check-eq "vt-vander([1,2,3],3) 降幂（NumPy 默认）"
            (vt-to-list (vt-vander (vt-from-sequence '(1 2 3) :dtype :int64) :n 3))
            '((1 1 1) (4 2 1) (9 3 1))))

(deftest test-creation-from
    "creation: vt-from-sequence / vt-from-array / vt-to-list / vt-to-array / vt-flatten-sequence / vt-from-function / vt-fromiter / vt-asarray / vt-kron / vt-meshgrid / vt-indices"
  (check-eq "vt-from-sequence 嵌套列表"
            (vt-to-list (vt-from-sequence '((1.0d0 2.0d0) (3.0d0 4.0d0))))
            '((1.0d0 2.0d0) (3.0d0 4.0d0)))
  (check-dtype "vt-from-sequence :dtype :float32" 
               (vt-from-sequence '(1 2) :dtype :float32) :float32)
  (check-error "vt-from-sequence 锯齿列表报错"
               (vt-from-sequence '((1 2) (3))))
  (let ((arr (make-array '(2 2) :initial-contents '((1.0d0 2.0d0) (3.0d0 4.0d0)))))
    (check-eq "vt-from-array 接受 2d CL 数组"
              (vt-to-list (vt-from-array arr)) '((1.0d0 2.0d0) (3.0d0 4.0d0)))
    (check-dtype "vt-from-array :dtype :int32" 
                 (vt-from-array arr :dtype :int32) :int32))
  (check-eq "vt-flatten-sequence 深度优先行主序"
            (vt-flatten-sequence '(1 (2 3) (4 (5 6)))) '(1 2 3 4 5 6))
  (check-eq "vt-from-function (2 3) 依索引列表生成（fn 接收索引列表）"
            (vt-to-list (vt-from-function '(2 3) (lambda (idx) (apply #'+ idx))))
            '(0.0d0 1.0d0 2.0d0 1.0d0 2.0d0 3.0d0))
  (check-dtype "vt-from-function :dtype :int64"
               (vt-from-function '(2) (lambda (idx)
					(declare (ignore idx)) 1)
				 :dtype :int64) :int64)
  (check-eq "vt-fromiter 从生成器取 count 个"
            (vt-to-list (vt-fromiter (loop for i below 4 collect (* i i))
                                     :dtype :int64 :count 4))
            '(0 1 4 9))
  (check-eq "vt-asarray 标量→0 维张量" (vt-to-list (vt-asarray 5.0d0)) '(5.0d0))
  (check "vt-asarray 对已是 vt 的输入原样返回"
         (let ((a (vt-ones '(2)))) (eq (vt-asarray a) a)))
  (check-eq "vt-kron 对标 np.kron"
            (vt-to-list (vt-kron (vt-from-sequence '((1 2) (3 4)) :dtype :int64)
                                 (vt-from-sequence '((0 1) (1 0)) :dtype :int64)))
            '((0 1 0 2) (1 0 2 0) (0 3 0 4) (3 0 4 0)))
  ;; vt-meshgrid / vt-indices 返回张量列表（而非多值）
  (destructuring-bind (xs ys)
      (vt-meshgrid (list (vt-from-sequence '(1 2)) (vt-from-sequence '(3 4 5))))
    (check-shape "vt-meshgrid :xy 第一输出 (3 2)" xs '(3 2))
    (check-eq "vt-meshgrid :xy 行向量广播" (vt-to-list xs)
              '((1.0d0 2.0d0) (1.0d0 2.0d0) (1.0d0 2.0d0)))
    (check-eq "vt-meshgrid :xy 列向量广播" (vt-to-list ys)
              '((3.0d0 3.0d0) (4.0d0 4.0d0) (5.0d0 5.0d0))))
  (destructuring-bind (xs ys)
      (vt-meshgrid (list (vt-from-sequence '(1 2)) (vt-from-sequence '(3 4 5)))
                   :indexing :ij)
    (declare (ignore ys))
    (check-shape "vt-meshgrid :ij 第一输出 (2 3)" xs '(2 3))
    (check-eq "vt-meshgrid :ij 第一行" (vt-to-list xs)
              '((1.0d0 1.0d0 1.0d0) (2.0d0 2.0d0 2.0d0))))
  (destructuring-bind (xs ys)
      (vt-meshgrid (list (vt-from-sequence '(1 2)) (vt-from-sequence '(3 4 5)))
                   :sparse t)
    (check-shape "vt-meshgrid sparse x 形状 (1 2)" xs '(1 2))
    (check-shape "vt-meshgrid sparse y 形状 (3 1)" ys '(3 1)))
  ;; vt-indices 返回 (2 2 3) 堆叠张量（对标 np.indices 的 tuple 形式）
  (let ((ij (vt-indices '(2 3))))
    (check-shape "vt-indices 堆叠形状 (2 2 3)" ij '(2 2 3))
    (check-eq "vt-indices 行索引" (vt-to-list (vt-slice ij '(0) '(:all) '(:all)))
              '((0 0 0) (1 1 1)))
    (check-eq "vt-indices 列索引" (vt-to-list (vt-slice ij '(1) '(:all) '(:all)))
              '((0 1 2) (0 1 2))))
  ;; vt-to-array：与 numpy 数组互转
  (let* ((a (vt-from-sequence '((1.0d0 2.0d0) (3.0d0 4.0d0))))
         (arr (vt-to-array a)))
    (check "vt-to-array 返回 CL 2d 数组" (= (array-rank arr) 2))
    (check "vt-to-array 元素正确" (= (aref arr 1 0) 3.0d0))))

;;; ==================================================================
;;; 3. 形状操作（manip.lisp）
;;; ==================================================================

(deftest test-manip-reshape-family
    "manip: vt-astype / vt-view / vt-reshape / vt-flatten / vt-ravel / vt-contiguous"
  (let ((a (vt-from-sequence '((1.0d0 2.0d0 3.0d0) (4.0d0 5.0d0 6.0d0)))))
    (check-dtype "vt-astype :int32 转换" (vt-astype a :int32) :int32)
    (check-eq "vt-astype 数值截断正确" (vt-to-list (vt-astype a :int32)) '(1 2 3 4 5 6))
    (check-shape "vt-reshape (3 2)" (vt-reshape a '(3 2)) '(3 2))
    (check-eq "vt-reshape 行主序" (vt-to-list (vt-reshape a '(3 2)))
              '((1.0d0 2.0d0) (3.0d0 4.0d0) (5.0d0 6.0d0)))
    (check "vt-reshape -1 推断维度" (equal (vt-shape (vt-reshape a '(3 -1))) '(3 2)))
    (check-error "vt-reshape 两个 -1 报错" (vt-reshape a '(-1 -1)))
    (check-error "vt-reshape 尺寸不符报错" (vt-reshape a '(4 2)))
    (check "vt-view 零拷贝要求连续（共享存储）"
           (let ((v (vt-view a '(3 2))))
             (and (vt-p v) (eq (vt-data v) (vt-data a)))))
    (check "vt-view 非连续输入报错"
           (handler-case (progn (vt-view (vt-slice a '(:all) '(0 nil 2)) '(3 2)) nil)
             (error () t)))
    (check-shape "vt-flatten → (6)" (vt-flatten a) '(6))
    (check-eq "vt-flatten 行主序" (vt-to-list (vt-flatten a)) '(1.0d0 2.0d0 3.0d0 4.0d0 5.0d0 6.0d0))
    (check-shape "vt-ravel → (6)" (vt-ravel a) '(6))
    (let ((c (vt-contiguous a)))
      (check "vt-contiguous 对连续输入可能共享存储" (vt-p c))
      (check-eq "vt-contiguous 数据不变" (vt-to-list c) '(1.0d0 2.0d0 3.0d0 4.0d0 5.0d0 6.0d0)))
    (let ((s (vt-contiguous (vt-slice a '(:all) '(0 nil 2)))))
      (check "vt-contiguous 对非连续视图生成连续副本" (vt-contiguous-p s))
      (check-eq "vt-contiguous 副本数据正确" (vt-to-list s) '(1.0d0 3.0d0 4.0d0 6.0d0)))))

(deftest test-manip-axes
    "manip: vt-transpose / vt-squeeze / vt-unsqueeze / vt-expand-dims / vt-swapaxes / vt-moveaxis / vt-rollaxis"
  (let ((a (vt-from-sequence '((1.0d0 2.0d0 3.0d0) (4.0d0 5.0d0 6.0d0)))))
    (check-shape "vt-transpose 默认反转全部轴" (vt-transpose a) '(3 2))
    (check-eq "vt-transpose 元素正确" (vt-to-list (vt-transpose a))
              '((1.0d0 4.0d0) (2.0d0 5.0d0) (3.0d0 6.0d0)))
    (check-shape "vt-transpose 显式 perm (1 0)" (vt-transpose a '(1 0)) '(3 2)))
  (check "vt-squeeze 默认移除全部长度 1 轴（§9.3: (1 3 1)→(3)）"
         (equal (vt-shape (vt-squeeze (vt-ones '(1 3 1)))) '(3)))
  (check "vt-squeeze :axis 0 只移除指定轴"
         (equal (vt-shape (vt-squeeze (vt-ones '(1 3 1)) :axis 0)) '(3 1)))
  (check-error "vt-squeeze 非长度 1 轴报错"
               (vt-squeeze (vt-ones '(2 3)) :axis 0))
  (check "vt-unsqueeze 0 插入最前" (equal (vt-shape (vt-unsqueeze (vt-ones '(3)) 0)) '(1 3)))
  (check "vt-unsqueeze 1 插入中间" (equal (vt-shape (vt-unsqueeze (vt-ones '(2 3)) 1)) '(2 1 3)))
  (check "vt-expand-dims 等价 unsqueeze" (equal (vt-shape (vt-expand-dims (vt-ones '(3)) 1)) '(3 1)))
  (let ((a (vt-arange 24 :dtype :int64)))
    (let ((t3 (vt-reshape a '(2 3 4))))
      (check-shape "vt-swapaxes(0,2) → (4 3 2)" (vt-swapaxes t3 0 2) '(4 3 2))
      (check-item "vt-swapaxes 元素正确" (vt-ref (vt-swapaxes t3 0 2) 0 0 0) 0)
      (check-item "vt-swapaxes 元素正确(1,1,1)" (vt-ref (vt-swapaxes t3 0 2) 1 1 1) 17)
      (check-shape "vt-moveaxis(0,-1) → (3 4 2)" (vt-moveaxis t3 0 -1) '(3 4 2))
      (check-shape "vt-rollaxis(2) → (4 2 3)（对标 np.rollaxis）"
                   (vt-rollaxis t3 2) '(4 2 3))
      (check-item "vt-rollaxis 元素正确" (vt-ref (vt-rollaxis t3 2) 0 1 2) 20))))

(deftest test-manip-split-join
    "manip: vt-split / vsplit / hsplit / dsplit / stack / vstack / hstack / dstack / concatenate / column_stack / block"
  (let ((a (vt-arange 6 :dtype :int64)))
    ;; vt-split 返回张量列表
    (destructuring-bind (p1 p2) (vt-split a 2 :axis 0)
      (check-eq "vt-split 等分前半" (vt-to-list p1) '(0 1 2))
      (check-eq "vt-split 等分后半" (vt-to-list p2) '(3 4 5)))
    (destructuring-bind (p1 p2) (vt-split a '(2) :axis 0)
      (check-eq "vt-split 索引切分 p1" (vt-to-list p1) '(0 1))
      (check-eq "vt-split 索引切分 p2" (vt-to-list p2) '(2 3 4 5))))
  (let ((m (vt-reshape (vt-arange 6 :dtype :int64) '(2 3))))
    (destructuring-bind (h1 h2 h3) (vt-hsplit m 3)
      (declare (ignore h3))
      (check-shape "vt-hsplit 列切分" h1 '(2 1))
      (check-item "vt-hsplit 元素" (vt-ref h2 0 0) 1))
    (destructuring-bind (v1 v2) (vt-vsplit m 2)
      (check-shape "vt-vsplit 行切分" v1 '(1 3))
      (check-item "vt-vsplit 元素" (vt-ref v2 0 0) 3)))
  (let* ((a (vt-ones '(2 2))) (b (vt-full '(2 2) 2.0d0)))
    (check-shape "vt-stack 默认 axis 0 → (2 2 2)" (vt-stack 0 a b) '(2 2 2))
    (check-shape "vt-stack axis 1 → (2 2 2)" (vt-stack 1 a b) '(2 2 2))
    (check-item "vt-stack axis 0 元素[1,0,1]=b[0,1]" (vt-ref (vt-stack 0 a b) 1 0 1) 2.0d0)
    (check-item "vt-stack axis 1 元素[0,1,0]=b[0,0]" (vt-ref (vt-stack 1 a b) 0 1 0) 2.0d0)
    (check-shape "vt-vstack → (4 2)" (vt-vstack a b) '(4 2))
    (check-shape "vt-hstack → (2 4)" (vt-hstack a b) '(2 4))
    (check-shape "vt-dstack → (2 2 2)" (vt-dstack a b) '(2 2 2)))
  (let ((a (vt-ones '(2 2))) (b (vt-full '(3 2) 2.0d0)))
    (check-shape "vt-concatenate axis 0 → (5 2)" (vt-concatenate 0 a b) '(5 2))
    (check-item "vt-concatenate 元素" (vt-ref (vt-concatenate 0 a b) 3 1) 2.0d0)
    (check-dtype "vt-concatenate dtype 提升 int8+float64→float64"
                 (vt-concatenate 0 (vt-ones '(2) :dtype :int8) (vt-ones '(2)))
                 :float64))
  (let ((a (vt-ones '(2 2) :dtype :int32)) (b (vt-full '(2 2) 5 :dtype :int64)))
    (check-dtype "vt-concatenate int32+int64→int64" (vt-concatenate 0 a b) :int64))
  (let ((cs (vt-column-stack (list (vt-from-sequence '(1 2)) (vt-from-sequence '(3 4))))))
    (check-shape "vt-column_stack 单一张量 (2 2)" cs '(2 2))
    (check-eq "vt-column_stack 数据（对标 np.column_stack）" (vt-to-list cs)
              '((1.0d0 3.0d0) (2.0d0 4.0d0))))
  (let* ((a (vt-from-sequence '((1.0d0 2.0d0) (3.0d0 4.0d0))))
         (b (vt-from-sequence '((5.0d0 6.0d0) (7.0d0 8.0d0))))
         (z (vt-zeros '(2 2)))
         (blk (vt-block (list (list a b) (list z z)))))
    (check-shape "vt-block 2x2 块 → (4 4)" blk '(4 4))
    (check-item "vt-block 元素[0,2]" (vt-ref blk 0 2) 5.0d0)
    (check-item "vt-block 元素[3,1]" (vt-ref blk 3 1) 0.0d0)))

(deftest test-manip-repeat-tile-pad
    "manip: vt-repeat / vt-tile / vt-pad（对标 np.pad 各模式）/ vt-rot90 / vt-flip / vt-roll"
  (check-eq "vt-repeat 标量重复（axis nil 展平）"
            (vt-to-list (vt-repeat (vt-from-sequence '(1 2 3) :dtype :int64) 2))
            '(1 1 2 2 3 3))
  (check-eq "vt-repeat 列表重复"
            (vt-to-list (vt-repeat (vt-from-sequence '(1 2) :dtype :int64) '(3 2)))
            '(1 1 1 2 2))
  (check-eq "vt-repeat axis 0"
            (vt-to-list (vt-repeat (vt-from-sequence '((1 2) (3 4)) :dtype :int64) 2 :axis 0))
            '((1 2) (1 2) (3 4) (3 4)))
  (check-eq "vt-tile (2) x (2 2) → (2 4)"
            (vt-to-list (vt-tile (vt-from-sequence '((1 2)) :dtype :int64) '(2 2)))
            '((1 2 1 2) (1 2 1 2)))
  (let ((m (vt-from-sequence '((1.0d0 2.0d0) (3.0d0 4.0d0)))))
    (check-eq "vt-pad 常量 1 圈默认 0（对标 np.pad）"
              (vt-to-list (vt-pad m 1))
              '((0.0d0 0.0d0 0.0d0 0.0d0) (0.0d0 1.0d0 2.0d0 0.0d0)
                (0.0d0 3.0d0 4.0d0 0.0d0) (0.0d0 0.0d0 0.0d0 0.0d0)))
    (check-eq "vt-pad :constant 显式值"
              (vt-to-list (vt-pad (vt-from-sequence '(1 2 3)) 1 :mode :constant :constant-values 9.0d0))
              '(9.0d0 1.0d0 2.0d0 3.0d0 9.0d0))
    (check-eq "vt-pad :edge" (vt-to-list (vt-pad (vt-from-sequence '(1 2 3)) '(2 1) :mode :edge))
              '(1.0d0 1.0d0 1.0d0 2.0d0 3.0d0 3.0d0))
    (check-eq "vt-pad :wrap" (vt-to-list (vt-pad (vt-from-sequence '(1 2 3)) '(2 1) :mode :wrap))
              '(2.0d0 3.0d0 1.0d0 2.0d0 3.0d0 1.0d0))
    (check-eq "vt-pad :reflect" (vt-to-list (vt-pad (vt-from-sequence '(1 2 3)) '(2 1) :mode :reflect))
              '(3.0d0 2.0d0 1.0d0 2.0d0 3.0d0 2.0d0))
    (check-eq "vt-pad :symmetric" (vt-to-list (vt-pad (vt-from-sequence '(1 2 3)) '(2 1) :mode :symmetric))
              '(2.0d0 1.0d0 1.0d0 2.0d0 3.0d0 3.0d0)))
  (let ((m (vt-reshape (vt-arange 6 :dtype :int64) '(2 3))))
    (check-eq "vt-rot90 逆时针 90°（对标 np.rot90）"
              (vt-to-list (vt-rot90 m)) '((2 5) (1 4) (0 3)))
    (check-eq "vt-rot90 k=2" (vt-to-list (vt-rot90 m :k 2)) '((5 4 3) (2 1 0)))
    (check-eq "vt-rot90 k=0 原样" (vt-to-list (vt-rot90 m :k 0)) '((0 1 2) (3 4 5)))
    (check-eq "vt-flip 默认反转全部轴（np.flip axis=None）"
              (vt-to-list (vt-flip m)) '((5 4 3) (2 1 0)))
    (check-eq "vt-flip :axis 0"
              (vt-to-list (vt-flip m :axis 0)) '((3 4 5) (0 1 2)))
    (check-eq "vt-fliplr 左右翻转" (vt-to-list (vt-fliplr m)) '((2 1 0) (5 4 3)))
    (check-eq "vt-flipud 上下翻转" (vt-to-list (vt-flipud m)) '((3 4 5) (0 1 2)))
    (check-eq "vt-roll 正向滚动（axis nil 展平）"
              (vt-to-list (vt-roll (vt-arange 4 :dtype :int64) 1)) '(3 0 1 2))
    (check-eq "vt-roll 负向滚动"
              (vt-to-list (vt-roll (vt-arange 4 :dtype :int64) -1)) '(1 2 3 0))
    (check-eq "vt-roll :axis 0 二维滚动"
              (vt-to-list (vt-roll m 1 :axis 0)) '((3 4 5) (0 1 2)))))

(deftest test-manip-triangle
    "manip: vt-triu / vt-tril / vt-diagonal / vt-tril-indices / vt-triu-indices / vt-fill-diagonal / vt-ravel-multi-index / vt-unravel-index"
  (let ((m (vt-from-sequence '((1.0d0 2.0d0 3.0d0) (4.0d0 5.0d0 6.0d0) (7.0d0 8.0d0 9.0d0)))))
    (check-eq "vt-triu k=0"
              (vt-to-list (vt-triu m))
              '((1.0d0 2.0d0 3.0d0) (0.0d0 5.0d0 6.0d0) (0.0d0 0.0d0 9.0d0)))
    (check-eq "vt-tril k=0"
              (vt-to-list (vt-tril m))
              '((1.0d0 0.0d0 0.0d0) (4.0d0 5.0d0 0.0d0) (7.0d0 8.0d0 9.0d0)))
    (check-eq "vt-triu k=1" (vt-to-list (vt-triu m :k 1))
              '((0.0d0 2.0d0 3.0d0) (0.0d0 0.0d0 6.0d0) (0.0d0 0.0d0 0.0d0)))
    (check-eq "vt-diagonal 主对角线" (vt-to-list (vt-diagonal m)) '(1.0d0 5.0d0 9.0d0))
    (check-eq "vt-diagonal :offset 1" (vt-to-list (vt-diagonal m :offset 1)) '(2.0d0 6.0d0))
    (multiple-value-bind (r c) (vt-tril-indices 2)
      (check-eq "vt-tril-indices 行" (vt-to-list r) '(0 1 1))
      (check-eq "vt-tril-indices 列" (vt-to-list c) '(0 0 1)))
    (multiple-value-bind (r c) (vt-triu-indices 2 :k 1)
      (check-eq "vt-triu-indices k=1 行（2x2 严格上三角仅 1 处）" (vt-to-list r) '(0))
      (check-eq "vt-triu-indices k=1 列" (vt-to-list c) '(1)))
    (let ((sq (vt-zeros '(3 3))))
      (vt-fill-diagonal sq 7.0d0)
      (check-eq "vt-fill-diagonal 写主对角线"
                (vt-to-list (vt-diagonal sq)) '(7.0d0 7.0d0 7.0d0)))
    (check "vt-ravel-multi-index (1,2) in (2 3) → 5"
           (= (vt-item (vt-ravel-multi-index (list 1 2) '(2 3))) 5))
    (let ((ij (vt-unravel-index 5 '(2 3) '(3 1))))
      (check "vt-unravel-index 5 → (1 2)"
             (and (= (first ij) 1) (= (second ij) 2))))))

(deftest test-manip-append-insert-delete
    "manip: vt-append / vt-insert / vt-delete / vt-trim-zeros / vt-resize / vt-broadcast-to / vt-broadcast-arrays / vt-flatten-to-nested"
  (check-eq "vt-append axis nil 展平拼接"
            (vt-to-list (vt-append (vt-from-sequence '(1 2) :dtype :int64)
                                   (vt-from-sequence '(3) :dtype :int64)))
            '(1 2 3))
  (check-eq "vt-insert 位置 1 插标量"
            (vt-to-list (vt-insert (vt-from-sequence '(1 2 3) :dtype :int64) 1 9))
            '(1 9 2 3))
  (check-eq "vt-insert 位置 1 插列表"
            (vt-to-list (vt-insert (vt-from-sequence '(1 2 3) :dtype :int64) 1 '(9 8)))
            '(1 9 8 2 3))
  (check-eq "vt-delete 位置 1"
            (vt-to-list (vt-delete (vt-from-sequence '(1 2 3) :dtype :int64) 1))
            '(1 3))
  (check-eq "vt-delete 列表索引"
            (vt-to-list (vt-delete (vt-from-sequence '(1 2 3 4) :dtype :int64) '(0 2)))
            '(2 4))
  (check-eq "vt-trim-zeros 去除首尾 0（:fb 默认）"
            (vt-to-list (vt-trim-zeros (vt-from-sequence '(0.0d0 1.0d0 2.0d0 0.0d0 0.0d0))))
            '(1.0d0 2.0d0))
  (check-eq "vt-trim-zeros :f 只去首部"
            (vt-to-list (vt-trim-zeros (vt-from-sequence '(0.0d0 1.0d0 0.0d0)) :trim :f))
            '(1.0d0 0.0d0))
  (check-eq "vt-resize 循环填充（对标 np.resize）"
            (vt-to-list (vt-resize (vt-from-sequence '(1 2 3) :dtype :int64) '(2 3)))
            '(1 2 3 1 2 3))
  (check-eq "vt-resize 截断"
            (vt-to-list (vt-resize (vt-from-sequence '(1 2 3) :dtype :int64) '(2)))
            '(1 2))
  (let ((b (vt-broadcast-to (vt-from-sequence '(1 2)) '(2 2))))
    (check-shape "vt-broadcast-to (2)→(2 2)" b '(2 2))
    (check-eq "vt-broadcast-to 数据（stride-0 虚拟重复）" (vt-to-list b) '((1.0d0 2.0d0) (1.0d0 2.0d0))))
  (check-error "vt-broadcast-to 形状不兼容报错"
               (vt-broadcast-to (vt-from-sequence '(1 2)) '(3 3)))
  (let ((br (vt-broadcast-arrays (list (vt-from-sequence '(1 2)) (vt-from-sequence '((1) (2)))))))
    (let ((ba (first br)) (bb (second br)))
      (check-shape "vt-broadcast-arrays 统一为 (2 2)" ba '(2 2))
      (check-shape "vt-broadcast-arrays 第二输出" bb '(2 2))))
  (check-eq "vt-flatten-to-nested 由扁平数据还原嵌套（data 须为向量）"
            (vt-flatten-to-nested '(2 2) (make-array 4 :initial-contents '(1 2 3 4)))
            '((1 2) (3 4))))

;;; ==================================================================
;;; 4. 索引与定位（indexing.lisp）
;;; ==================================================================

(deftest test-indexing-ref-slice
    "indexing: vt-ref / vt-item / vt-slice（D1 语法：无 :range）"
  (let ((m (vt-reshape (vt-arange 6 :dtype :int64) '(2 3))))
    (check "vt-ref 全整数 → 标量" (= (vt-ref m 1 2) 5))
    (check "vt-ref 负索引" (= (vt-ref m -1 -1) 5))
    (check-error "vt-ref 越界报错" (vt-ref m 2 0))
    (check-error "vt-ref 负越界报错" (vt-ref m -3 0))
    ;; 视图切片用 vt-slice（vt-ref 仅接受整数下标）
    (let ((v (vt-slice m '(:all) '(1 2))))
      (check-shape "vt-slice 取列 1 → (2 1)" v '(2 1))
      (check-eq "vt-slice 列切片数据" (vt-to-list v) '((1) (4))))
    ;; 单元素 spec = 索引选择（移除该轴），对标 numpy m[0]
    (let ((v (vt-slice m '(0) '(:all))))
      (check-shape "vt-slice 索引 0 → (3)" v '(3))
      (check-eq "vt-slice 行数据" (vt-to-list v) '(0 1 2)))
    (check "vt-item 提取 0 维标量" (= (vt-item (vt-asarray 42)) 42))
    (let ((s (vt-slice m '(0 2) '(1 3))))
      (check-shape "vt-slice (0 2)(1 3) → (2 2)" s '(2 2))
      (check-eq "vt-slice 数据" (vt-to-list s) '((1 2) (4 5))))
    (let ((s (vt-slice m '(:all) '(0 nil 2))))
      (check-shape "vt-slice :all + 步长 2 → (2 2)" s '(2 2))
      (check-eq "vt-slice 步长数据" (vt-to-list s) '((0 2) (3 5))))
    (let ((s (vt-slice m '(2 nil -1) '(:all))))
      (check-shape "vt-slice 负步长反转行 → (2 3)" s '(2 3))
      (check-eq "vt-slice 负步长数据" (vt-to-list s) '((3 4 5) (0 1 2))))
    (let ((s (vt-slice m '(1) '(:newa))))
      (check-shape "vt-slice :newa 保留轴 → (1 3)" s '(1 3)))))

(deftest test-indexing-where-family
    "indexing: vt-where / vt-argwhere / vt-nonzero / vt-flatnonzero / vt-extract / vt-select"
  (let ((c (vt-from-sequence '(1 0 2 0) :dtype :int64)))
    (let ((w (vt-where c)))
      (check-shape "vt-where 单参 → (n rank) 坐标" w '(2 1))
      (check-eq "vt-where 坐标 [0 2]" (vt-to-list w) '(0 2)))
    (check-eq "vt-argwhere 坐标" (vt-to-list (vt-argwhere c)) '(0 2))
    (let ((nz (vt-nonzero c)))
      (check "vt-nonzero 返回 rank 个索引张量" (= (length nz) 1))
      (check-eq "vt-nonzero 索引" (vt-to-list (first nz)) '(0 2)))
    (check-eq "vt-flatnonzero" (vt-to-list (vt-flatnonzero c)) '(0 2))
    (check-eq "vt-extract 抽取真值元素"
              (vt-to-list (vt-extract c (vt-from-sequence '(10 20 30 40) :dtype :int64)))
              '(10 30))
    (let* ((cond1 (vt-from-sequence '(1 0 1) :dtype :int64))
           (cond2 (vt-from-sequence '(0 1 0) :dtype :int64))
           (sel (vt-select (list cond1 cond2)
                           (list (vt-full '(3) 1.0d0) (vt-full '(3) 2.0d0))
                           :default 7.0d0)))
      (check-eq "vt-select 条件优先级 + default" (vt-to-list sel) '(1.0d0 2.0d0 1.0d0))))
  (let* ((x (vt-from-sequence '(1.0d0 2.0d0 3.0d0 4.0d0)))
         (y (vt-from-sequence '(10.0d0 20.0d0 30.0d0 40.0d0)))
         (c (vt-from-sequence '(1 0 1 0) :dtype :int8)))
    (check-eq "vt-where 三参广播选择" (vt-to-list (vt-where c x y)) '(1.0d0 20.0d0 3.0d0 40.0d0))))

(deftest test-indexing-take-put
    "indexing: vt-take / vt-put（mode raise 默认）/ vt-compress / take-along-axis / put-along-axis"
  (let ((a (vt-from-sequence '(10 20 30 40) :dtype :int64)))
    (check-eq "vt-take 展平索引"
              (vt-to-list (vt-take a (vt-from-sequence '(0 2) :dtype :int64))) '(10 30))
    (check-eq "vt-take :axis 0"
              (vt-to-list (vt-take (vt-reshape a '(2 2)) (vt-from-sequence '(1) :dtype :int64) :axis 0))
              '((30 40)))
    (check "vt-take 越界 mode :raise 报错"
           (handler-case (progn (vt-take a (vt-from-sequence '(4) :dtype :int64)) nil)
             (error () t)))
    (let ((p (vt-copy a)))
      (vt-put p (vt-from-sequence '(0 2) :dtype :int64) (vt-from-sequence '(9 8) :dtype :int64))
      (check-eq "vt-put 默认 :raise 就地写入" (vt-to-list p) '(9 20 8 40)))
    (let ((p (vt-copy a)))
      (vt-put p (vt-from-sequence '(0 6) :dtype :int64) (vt-from-sequence '(9 8) :dtype :int64) :mode :clip)
      (check-eq "vt-put :mode clip（对标 np.put clip）" (vt-to-list p) '(9 20 30 8)))
    (check-error "vt-put :raise 越界报错"
                 (vt-put (vt-copy a) (vt-from-sequence '(6) :dtype :int64)
                         (vt-from-sequence '(1) :dtype :int64) :mode :raise)))
  (let ((m (vt-reshape (vt-arange 6 :dtype :int64) '(2 3))))
    (let ((r (vt-compress (vt-from-sequence '(1 0) :dtype :int8) m :axis 0)))
      (check-shape "vt-compress axis 0 → (1 3)" r '(1 3))
      (check-eq "vt-compress 数据（条件 1 0 保留首行）" (vt-to-list r) '((0 1 2))))
    (let ((r (vt-take-along-axis m (vt-from-sequence '((0 2) (1 0)) :dtype :int64) 1)))
      (check-shape "vt-take-along-axis 形状同索引" r '(2 2))
      (check-eq "vt-take-along-axis 数据" (vt-to-list r) '((0 2) (4 3)))
    ;; 契约：返回新张量、不修改输入（对标 numpy.put_along_axis）；
    ;; values 形状须与 indices 形状精确相等（见 vt-put-along-axis docstring）
    (let ((r (vt-put-along-axis (vt-copy m) (vt-from-sequence '((0 1) (2 0)) :dtype :int64)
                                (vt-from-sequence '((99 88) (77 66)) :dtype :int64) 1)))
      (check-eq "vt-put-along-axis 逐位写入（返回新张量）" (vt-to-list r)
                '((99 88 2) (66 4 77)))
      (check-eq "vt-put-along-axis 输入保持不变" (vt-to-list m) '((0 1 2) (3 4 5)))))))

(deftest test-indexing-search
    "indexing: vt-searchsorted / vt-digitize / vt-bincount / vt-choose / vt-count / vt-count-nonzero / vt-clip-tensor / vt-clamp / vt-fill-diagonal"
  (let ((s (vt-from-sequence '(1 3 5) :dtype :int64)))
    (check-eq "vt-searchsorted side :left"
              (vt-to-list (vt-searchsorted s (vt-from-sequence '(2 5 6) :dtype :int64)))
              '(1 2 3))
    (check-eq "vt-searchsorted side :right"
              (vt-to-list (vt-searchsorted s (vt-from-sequence '(2 5 6) :dtype :int64) :side :right))
              '(1 3 3)))
  (let ((x (vt-from-sequence '(0.5d0 1.0d0 1.5d0 3.0d0 3.5d0))))
    (check-eq "vt-digitize right=nil"
              (vt-to-list (vt-digitize x (vt-from-sequence '(1.0d0 3.0d0)))) '(0 1 1 2 2))
    (check-eq "vt-digitize right=t"
              (vt-to-list (vt-digitize x (vt-from-sequence '(1.0d0 3.0d0)) :right t)) '(0 0 1 1 2)))
  (check-eq "vt-bincount minlength=5"
            (vt-to-list (vt-bincount (vt-from-sequence '(0 1 1 3) :dtype :int64) :minlength 5))
            '(1 2 0 1 0))
  (let ((r (vt-choose (list (vt-full '(4) 10.0d0) (vt-full '(4) 11.0d0) (vt-full '(4) 12.0d0))
                      (vt-from-sequence '(0 1 2 1) :dtype :int64))))
    (check-eq "vt-choose 按索引选取" (vt-to-list r) '(10.0d0 11.0d0 12.0d0 11.0d0))
    (check "vt-choose 越界 mode :raise 报错"
           (handler-case
               (progn (vt-choose (list (vt-full '(2) 1.0d0) (vt-full '(2) 2.0d0))
                                 (vt-from-sequence '(0 2) :dtype :int64))
                      nil)
             (error () t))))
  (let ((a (vt-from-sequence '(1 2 3 4) :dtype :int64)))
    (check "vt-count 计 2 出现次数" (= (vt-item (vt-count a 2)) 1))
    (check "vt-count-nonzero 计非零" (= (vt-item (vt-count-nonzero a)) 4))
    (check-dtype "vt-count 默认 :int64" (vt-count a 1) :int64))
  (let ((a (vt-from-sequence '(1.0d0 5.0d0 9.0d0))))
    (check-eq "vt-clip 上下限" (vt-to-list (vt-clip a 2.0d0 8.0d0)) '(2.0d0 5.0d0 8.0d0))
    (check-eq "vt-clip 仅上限（min 缺省）" (vt-to-list (vt-clip a nil 8.0d0)) '(1.0d0 5.0d0 8.0d0))
    (check-eq "vt-clamp 等价 clip" (vt-to-list (vt-clamp a 2.0d0 8.0d0)) '(2.0d0 5.0d0 8.0d0))
    (check-eq "vt-clip-tensor 等价 clip" (vt-to-list (vt-clip-tensor a 2.0d0 8.0d0)) '(2.0d0 5.0d0 8.0d0))
    (check "vt-clip 无界参数 → 原样复制" (allclose-lists (vt-to-list (vt-clip a)) '(1.0d0 5.0d0 9.0d0)))
    (check-error "vt-clip 两个有位置参数都给时报错" (vt-clip a 2.0d0 8.0d0 1.0d0))
    (check "vt-clip min>max → 全部取 max（NumPy 对齐）"
           (allclose-lists (vt-to-list (vt-clip a 5.0d0 2.0d0)) '(2.0d0 2.0d0 2.0d0)))))

;;; ==================================================================
;;; 5. 逐元素算术（elementwise.lisp）
;;; ==================================================================

(deftest test-arith-basic
    "elementwise: vt-+ * - / 与 vt-add/sub/mul/div/scale（§4.4 类型提升）"
  (let ((a (vt-from-sequence '(1.0d0 2.0d0 3.0d0)))
        (b (vt-from-sequence '(10.0d0 20.0d0 30.0d0))))
    (check-eq "vt-+ 两参" (vt-to-list (vt-+ a b)) '(11.0d0 22.0d0 33.0d0))
    (check-eq "vt-+ 广播 (3)+(1 3)" (vt-to-list (vt-+ a (vt-ones '(1 3))))
              '((2.0d0 3.0d0 4.0d0)))
    (check-eq "vt-* 两参" (vt-to-list (vt-* a b)) '(10.0d0 40.0d0 90.0d0))
    (check-eq "vt-- 单参取负" (vt-to-list (vt-- a)) '(-1.0d0 -2.0d0 -3.0d0))
    (check-eq "vt-- 两参相减" (vt-to-list (vt-- b a)) '(9.0d0 18.0d0 27.0d0))
    (check-eq "vt-/ 两参相除" (vt-to-list (vt-/ b a)) '(10.0d0 10.0d0 10.0d0))
    (check-eq "vt-+ 标量混合" (vt-to-list (vt-+ a 1)) '(2.0d0 3.0d0 4.0d0))
    (check-eq "vt-add 等价 +" (vt-to-list (vt-add a b)) '(11.0d0 22.0d0 33.0d0))
    (check-eq "vt-sub" (vt-to-list (vt-sub b a)) '(9.0d0 18.0d0 27.0d0))
    (check-eq "vt-mul" (vt-to-list (vt-mul a b)) '(10.0d0 40.0d0 90.0d0))
    (check-eq "vt-div" (vt-to-list (vt-div b a)) '(10.0d0 10.0d0 10.0d0))
    (check-eq "vt-scale" (vt-to-list (vt-scale a 2.0d0)) '(2.0d0 4.0d0 6.0d0))
    (check-dtype "vt-add int8+int8 → int8（同型保持）"
                 (vt-add (vt-ones '(2) :dtype :int8) (vt-ones '(2) :dtype :int8)) :int8)
    (check-dtype "vt-add int8+uint8 → int16（64 格提升表）"
                 (vt-add (vt-ones '(2) :dtype :int8) (vt-ones '(2) :dtype :uint8)) :int16)
    (check-dtype "vt-add int16+uint16 → int32"
                 (vt-add (vt-ones '(2) :dtype :int16) (vt-ones '(2) :dtype :uint16)) :int32)
    ;; 注：:uint32 不在库支持 dtype 集（*vt-dtypes*）内，不测 int32+uint32
    (check-dtype "vt-add float32+float64 → float64"
                 (vt-add (vt-ones '(2) :dtype :float32) (vt-ones '(2) :dtype :float64)) :float64)
    (check-dtype "vt-add int8+float32 → float32"
                 (vt-add (vt-ones '(2) :dtype :int8) (vt-ones '(2) :dtype :float32)) :float32)
    (check-dtype "vt-add int64+float32 → float64"
                 (vt-add (vt-ones '(2) :dtype :int64) (vt-ones '(2) :dtype :float32)) :float64))
  ;; vt-/ = true_divide：整数输入 → float64；零除按 IEEE 得 ±Inf（D16 与之区分）
  (let ((a (vt-from-sequence '(1 2) :dtype :int64)))
    (check-dtype "vt-/ 整数 → true divide float64" (vt-/ a 2) :float64)
    (check-eq "vt-/ 整数值" (vt-to-list (vt-/ a 2)) '(0.5d0 1.0d0))
    (check-eq "vt-/ 整数零除 → +Inf（IEEE）"
              (vt-to-list (vt-/ a 0))
              (list (vt-float-pos-inf :float64) (vt-float-pos-inf :float64))))
  (let ((a (vt-from-sequence '(7 -7) :dtype :int64)))
    ;; D16：vt-div 整数零除返回 0，dtype 保持整型
    (check-dtype "vt-div 整数零除 dtype 保持整型" (vt-div a 0) :int64)
    (check-eq "vt-div 整数零除 → 0" (vt-to-list (vt-div a 0)) '(0 0))))

(deftest test-arith-div-floor-mod
    "elementwise: vt-div floor 除法 / vt-mod（remainder）/ vt-rem（fmod）/ vt-divmod"
  ;; vt-div 对标 np.floor_divide：整数零除 → 0，dtype 保持整型（CONVENTIONS D16）
  (let ((a (vt-from-sequence '(7 -7) :dtype :int64)))
    (check-eq "vt-div floor 除法 7//3=2" (vt-to-list (vt-div a 3)) '(2 -3))
    (let ((z (handler-case (vt-to-list (vt-div a 0))
               (error (e) (list e)))))
      (check "vt-div 整数零除不抛错（返回 0）" (equal z '(0 0))))
    (check-eq "vt-mod = remainder（符号随除数）" (vt-to-list (vt-mod a 3)) '(1 2))
    (check-eq "vt-rem = fmod（符号随被除数）" (vt-to-list (vt-rem a 3)) '(1 -1))
    (multiple-value-bind (q r) (vt-divmod a 3)
      (check-eq "vt-divmod 商" (vt-to-list q) '(2 -3))
      (check-eq "vt-divmod 余" (vt-to-list r) '(1 2))))
  (let ((f (vt-from-sequence '(3.5d0 -3.5d0))))
    (check-eq "vt-mod 浮点零除 → NaN" (vt-to-list (vt-mod f 0.0d0))
              (vt-float-nan :float64) (vt-float-nan :float64))))

(deftest test-arith-floor-family
    "elementwise: vt-round / vt-rint / vt-floor / vt-ceiling / vt-truncate（半偶数舍入）"
  (let ((a (vt-from-sequence '(0.5d0 1.5d0 2.5d0 -0.5d0 -1.5d0))))
    ;; numpy.round / rint 均为「最近偶数」：0.5→0, 1.5→2, 2.5→2, -0.5→-0(0), -1.5→-2
    (check-eq "vt-round 最近偶数" (vt-to-list (vt-round a)) '(0.0d0 2.0d0 2.0d0 -0.0d0 -2.0d0))
    (check-eq "vt-rint 最近偶数" (vt-to-list (vt-rint a)) '(0.0d0 2.0d0 2.0d0 -0.0d0 -2.0d0)))
  (let ((a (vt-from-sequence '(1.2d0 1.7d0 -1.2d0 -1.7d0))))
    (check-eq "vt-floor" (vt-to-list (vt-floor a)) '(1.0d0 1.0d0 -2.0d0 -2.0d0))
    (check-eq "vt-ceiling" (vt-to-list (vt-ceiling a)) '(2.0d0 2.0d0 -1.0d0 -1.0d0))
    (check-eq "vt-truncate 向零" (vt-to-list (vt-truncate a)) '(1.0d0 1.0d0 -1.0d0 -1.0d0))
    (check-eq "vt-round 可带 divisor" (vt-to-list (vt-round (vt-from-sequence '(1234.0d0)) :divisor 100.0d0))
              '(12.0d0)))
  (let ((a (vt-from-sequence (list (vt-float-nan :float64) (vt-float-pos-inf :float64)
                                   (vt-float-neg-inf :float64)))))
    (check-eq "vt-floor NaN/Inf 传播" (vt-to-list (vt-floor a))
              (list (vt-float-nan :float64) (vt-float-pos-inf :float64) (vt-float-neg-inf :float64)))
    (check-eq "vt-round NaN/Inf 传播" (vt-to-list (vt-round a))
              (list (vt-float-nan :float64) (vt-float-pos-inf :float64) (vt-float-neg-inf :float64)))))

(deftest test-arith-powers-logs
    "elementwise: vt-square / pow / expt / sqrt / exp / log 族 / cbrt / reciprocal / expm1 / log1p / logaddexp / float-power"
  (let ((a (vt-from-sequence '(2.0d0 4.0d0 9.0d0))))
    (check-eq "vt-square" (vt-to-list (vt-square a)) '(4.0d0 16.0d0 81.0d0))
    (check-eq "vt-pow" (vt-to-list (vt-pow a 2.0d0)) '(4.0d0 16.0d0 81.0d0))
    (check-eq "vt-expt 等价 pow" (vt-to-list (vt-expt a 0.5d0)) '(1.4142135623730951d0 2.0d0 3.0d0))
    (check-eq "vt-sqrt" (vt-to-list (vt-sqrt a)) '(1.4142135623730951d0 2.0d0 3.0d0))
    (check-eq "vt-log 自然对数" (vt-to-list (vt-log a))
              '(0.6931471805599453d0 1.3862943611198906d0 2.1972245773362196d0))
    (check-eq "vt-log2" (vt-to-list (vt-log2 a)) '(1.0d0 2.0d0 3.1699250014423126d0))
    (check-eq "vt-log10" (vt-to-list (vt-log10 a)) '(0.3010299956639812d0 0.6020599913279624d0 0.9542425094393488d0))
    (check-eq "vt-cbrt" (vt-to-list (vt-cbrt (vt-from-sequence '(8.0d0 27.0d0)))) '(2.0d0 3.0d0))
    (check-eq "vt-reciprocal 浮点" (vt-to-list (vt-reciprocal (vt-from-sequence '(2.0d0 4.0d0))))
              '(0.5d0 0.25d0))
    (check "vt-reciprocal 整数 → 0（对标 np.reciprocal 整型）"
           (equal (vt-to-list (vt-reciprocal (vt-from-sequence '(2) :dtype :int32))) '(0)))
    (check-eq "vt-exp" (vt-to-list (vt-exp (vt-from-sequence '(0.0d0 1.0d0)))) '(1.0d0 2.718281828459045d0))
    (check-eq "vt-expm1" (vt-to-list (vt-expm1 (vt-from-sequence '(0.0d0 1.0d0)))) '(0.0d0 1.718281828459045d0))
    (check-eq "vt-log1p" (vt-to-list (vt-log1p (vt-from-sequence '(0.0d0 1.0d0)))) '(0.0d0 0.6931471805599453d0))
    (check-eq "vt-logaddexp(0,0)=ln2" (vt-to-list (vt-logaddexp (vt-from-sequence '(0.0d0)) (vt-from-sequence '(0.0d0))))
              '(0.6931471805599453d0))
    (check-eq "vt-float-power 恒浮点" (vt-to-list (vt-float-power (vt-from-sequence '(2) :dtype :int64) 3))
              '(8.0d0))
    (check-eq "vt-float-power 负底数非整数指数 → NaN"
              (vt-to-list (vt-float-power (vt-from-sequence '(-2.0d0)) 0.5d0))
              (vt-float-nan :float64))))

(deftest test-arith-trig
    "elementwise: 三角/反三角（asin 命名，§9.3）/ 双曲 / atan2 / hypot(inf,nan)=inf / sinc / deg2rad"
  (check-eq "vt-sin(pi/2)=1" (vt-to-list (vt-sin (vt-from-sequence (list (/ pi 2.0d0)))))
            '(1.0d0))
  (check-eq "vt-cos(0)=1" (vt-to-list (vt-cos (vt-zeros '(1)))) '(1.0d0))
  (check-eq "vt-tan(0)=0" (vt-to-list (vt-tan (vt-zeros '(1)))) '(0.0d0))
  (check-eq "vt-asin 命名（非 arcsin）" (vt-to-list (vt-asin (vt-from-sequence '(1.0d0))))
            (list (/ pi 2.0d0)))
  (check-eq "vt-acos(1)=0" (vt-to-list (vt-acos (vt-from-sequence '(1.0d0)))) '(0.0d0))
  (check-eq "vt-atan(0)=0" (vt-to-list (vt-atan (vt-zeros '(1)))) '(0.0d0))
  (check-eq "vt-atan2(1,1)=pi/4" (vt-to-list (vt-atan2 (vt-ones '(1)) (vt-ones '(1))))
            (list (/ pi 4.0d0)))
  (check-eq "vt-asinh(0)=0" (vt-to-list (vt-asinh (vt-zeros '(1)))) '(0.0d0))
  (check-eq "vt-acosh(1)=0" (vt-to-list (vt-acosh (vt-ones '(1)))) '(0.0d0))
  (check-eq "vt-atanh(0)=0" (vt-to-list (vt-atanh (vt-zeros '(1)))) '(0.0d0))
  (check-eq "vt-sinh(0)=0" (vt-to-list (vt-sinh (vt-zeros '(1)))) '(0.0d0))
  (check-eq "vt-cosh(0)=1" (vt-to-list (vt-cosh (vt-zeros '(1)))) '(1.0d0))
  (check-eq "vt-tanh(0)=0" (vt-to-list (vt-tanh (vt-zeros '(1)))) '(0.0d0))
  (check-eq "vt-sinc(0)=1, sinc(x)=sin(pi x)/(pi x)"
            (vt-to-list (vt-sinc (vt-from-sequence '(0.0d0 0.5d0))))
            '(1.0d0 0.6366197723675814d0))
  (check-eq "vt-deg2rad 180→pi" (vt-to-list (vt-deg2rad (vt-from-sequence '(180.0d0))))
            (list pi))
  (check-eq "vt-rad2deg pi→180" (vt-to-list (vt-rad2deg (vt-from-sequence (list pi))))
            '(180.0d0))
  ;; §9.2: hypot(∞,nan)=inf（IEEE posinf > nan 判定）
  (check-eq "vt-hypot(inf,nan)=inf（NumPy 对齐）"
            (vt-to-list (vt-hypot (vt-from-sequence (list (vt-float-pos-inf :float64)))
                                  (vt-from-sequence (list (vt-float-nan :float64)))))
            (list (vt-float-pos-inf :float64)))
  (check-eq "vt-hypot(3,4)=5" (vt-to-list (vt-hypot (vt-from-sequence '(3.0d0)) (vt-from-sequence '(4.0d0))))
            '(5.0d0)))

(deftest test-arith-sign-clip-compare
    "elementwise: vt-abs/absolute/sign/positive/negative/clip/min-max 族/fmax-fmin/maximum-NaN 语义"
  (let ((a (vt-from-sequence '(-3.0d0 -0.0d0 2.5d0))))
    (check-eq "vt-abs" (vt-to-list (vt-abs a)) '(3.0d0 0.0d0 2.5d0))
    (check-eq "vt-absolute 等价 abs" (vt-to-list (vt-absolute a)) '(3.0d0 0.0d0 2.5d0))
    (check-eq "vt-sign (-3,-0,2.5)→(-1,0,1)" (vt-to-list (vt-sign a)) '(-1.0d0 0.0d0 1.0d0))
    ;; vt-positive/vt-negative 对标 np.positive/np.negative（恒等/取负，非 relu）
    (check-eq "vt-positive = +x（np.positive 恒等）"
              (vt-to-list (vt-positive a)) '(-3.0d0 -0.0d0 2.5d0))
    (check-eq "vt-negative = -x（np.negative 取负）"
              (vt-to-list (vt-negative a)) '(3.0d0 0.0d0 -2.5d0))
    (check-eq "vt-signum 等价 sign" (vt-to-list (vt-signum a)) '(-1.0d0 0.0d0 1.0d0)))
  (let ((a (vt-from-sequence '(1.0d0 5.0d0 9.0d0))))
    (check-eq "vt-clip 上下限" (vt-to-list (vt-clip a 2.0d0 8.0d0)) '(2.0d0 5.0d0 8.0d0)))
  (let ((x (vt-from-sequence (list (vt-float-nan :float64) 1.0d0)))
        (y (vt-from-sequence '(2.0d0 3.0d0))))
    ;; §9.2: np.maximum([nan,1],[2,3]) → [nan,3]；np.fmax → [2,3]
    (check-eq "vt-maximum NaN 传播" (vt-to-list (vt-maximum x y)) (vt-float-nan :float64) 3.0d0)
    (check-eq "vt-fmax NaN 忽略" (vt-to-list (vt-fmax x y)) '(2.0d0 3.0d0))
    (check-eq "vt-minimum NaN 传播" (vt-to-list (vt-minimum x y)) (vt-float-nan :float64) 1.0d0)
    (check-eq "vt-fmin NaN 忽略" (vt-to-list (vt-fmin x y)) '(2.0d0 1.0d0))
    ;; maximum 广播遇 NaN 传播（对标 np.maximum([2,3],[nan,1]) = [nan,3]）
    (check-eq "vt-maximum 广播+NaN 传播" (vt-to-list (vt-maximum y x))
              (vt-float-nan :float64) 3.0d0)))

(deftest test-arith-predicates-logic
    "elementwise: *-p 谓词 / logical 族 / 位运算 / 移位"
  (let ((a (vt-from-sequence '(1 -2 0) :dtype :int32)))
    (check-eq "vt-positive-p" (vt-to-list (vt-positive-p a)) '(1 0 0))
    (check-eq "vt-negative-p" (vt-to-list (vt-negative-p a)) '(0 1 0))
    (check-eq "vt-zero-p" (vt-to-list (vt-zero-p a)) '(0 0 1))
    (check-eq "vt-nonzero-p" (vt-to-list (vt-nonzero-p a)) '(1 1 0))
    (check-eq "vt-even-p" (vt-to-list (vt-even-p a)) '(0 1 1))
    (check-eq "vt-odd-p" (vt-to-list (vt-odd-p a)) '(1 0 0))
    (check-dtype "谓词默认 :int8 承载布尔（§9.3）" (vt-positive-p a) :int8))
  (let ((x (vt-from-sequence '(1 0 2) :dtype :int64))
        (y (vt-from-sequence '(0 3 2) :dtype :int64)))
    (check-eq "vt-logical-and" (vt-to-list (vt-logical-and x y)) '(0.0d0 0.0d0 1.0d0))
    (check-eq "vt-logical-or" (vt-to-list (vt-logical-or x y)) '(1.0d0 1.0d0 1.0d0))
    (check-eq "vt-logical-not" (vt-to-list (vt-logical-not x)) '(0.0d0 1.0d0 0.0d0))
    (check-eq "vt-logical-xor" (vt-to-list (vt-logical-xor x y)) '(1.0d0 1.0d0 0.0d0)))
  (let ((a (vt-from-sequence '(12 10) :dtype :int32))
        (b (vt-from-sequence '(10 6) :dtype :int32)))
    (check-eq "vt-bit-and" (vt-to-list (vt-bit-and a b)) '(8 2))
    (check-eq "vt-bit-ior" (vt-to-list (vt-bit-ior a b)) '(14 14))
    (check-eq "vt-bit-xor" (vt-to-list (vt-bit-xor a b)) '(6 12))
    (check-eq "vt-bit-not" (vt-to-list (vt-bit-not (vt-from-sequence '(0) :dtype :int32))) '(-1))
    (check-eq "vt-left-shift" (vt-to-list (vt-left-shift (vt-from-sequence '(1) :dtype :int32) 3)) '(8))
    (check-eq "vt-right-shift" (vt-to-list (vt-right-shift (vt-from-sequence '(8) :dtype :int32) 3)) '(1))))

(deftest test-arith-misc-math
    "elementwise: vt-copysign / signbit / nextafter / spacing / gcd / lcm / nan-to-num / real / imag / conj / angle / lerp"
  (check-eq "vt-copysign(-2,1)=2" (vt-to-list (vt-copysign (vt-from-sequence '(-2.0d0)) (vt-from-sequence '(1.0d0))))
            '(2.0d0))
  (check-eq "vt-signbit 负数" (vt-to-list (vt-signbit (vt-from-sequence '(-1.0d0 1.0d0)))) '(1.0d0 0.0d0))
  (check-eq "vt-nextafter(1,2) 是 1 的上邻"
            (vt-to-list (vt-nextafter (vt-from-sequence '(1.0d0)) (vt-from-sequence '(2.0d0))))
            (list (+ 1.0d0 double-float-epsilon)))
  (check-eq "vt-spacing(1) = double-float-epsilon"
            (vt-to-list (vt-spacing (vt-from-sequence '(1.0d0))))
            (list double-float-epsilon))
  (check-eq "vt-gcd" (vt-to-list (vt-gcd (vt-from-sequence '(12 18) :dtype :int64)
                                         (vt-from-sequence '(8) :dtype :int64)))
            '(4 2))
  (check-eq "vt-lcm" (vt-to-list (vt-lcm (vt-from-sequence '(4 6) :dtype :int64)
                                         (vt-from-sequence '(6 8) :dtype :int64)))
            '(12 24))
  (let ((a (vt-from-sequence (list (vt-float-nan :float64) (vt-float-pos-inf :float64)
                                   (vt-float-neg-inf :float64) 3.0d0))))
    (check-eq "vt-nan-to-num 默认 NaN→0、±Inf→±大数、有限值不变"
              (vt-to-list (vt-nan-to-num a))
              0.0d0 most-positive-double-float most-negative-double-float 3.0d0)
    (check-eq "vt-nan-to-num 显式替换值"
              (vt-to-list (vt-nan-to-num a :nan -1.0d0 :posinf 100.0d0 :neginf -100.0d0))
              '(-1.0d0 100.0d0 -100.0d0 3.0d0)))
  (let ((a (vt-from-sequence '(-1.0d0 2.0d0))))
    (check-eq "vt-real 实部=自身" (vt-to-list (vt-real a)) '(-1.0d0 2.0d0))
    (check-eq "vt-imag 实数张量虚部=0" (vt-to-list (vt-imag a)) '(0.0d0 0.0d0))
    (check-eq "vt-conj 实数张量=自身" (vt-to-list (vt-conj a)) '(-1.0d0 2.0d0))
    (check-eq "vt-angle z<0→π，z>0→0" (vt-to-list (vt-angle a)) (list pi 0.0d0))
    (check-eq "vt-angle :deg t → 180" (vt-to-list (vt-angle a :deg t)) '(180.0d0 0.0d0)))
  (check-eq "vt-lerp 中点" (vt-to-list (vt-lerp (vt-from-sequence '(1.0d0 2.0d0))
                                                (vt-from-sequence '(3.0d0 4.0d0)) 0.5d0))
            '(2.0d0 3.0d0)))

(deftest test-arith-comparisons
    "elementwise: vt-= /= < <= > >= 全部返回 :int8 0/1（§9.3）"
  (let ((a (vt-from-sequence '(1.0d0 2.0d0 3.0d0)))
        (b (vt-from-sequence '(1.0d0 0.0d0 3.0d0))))
    (check-dtype "vt-= 默认 :int8" (vt-= a b) :int8)
    (check-eq "vt-=" (vt-to-list (vt-= a b)) '(1 0 1))
    (check-eq "vt-/=（不等，CL 风格命名）" (vt-to-list (vt-/= a b)) '(0 1 0))
    (check-eq "vt-<" (vt-to-list (vt-< a b)) '(0 0 0))
    (check-eq "vt-<=" (vt-to-list (vt-<= a b)) '(1 0 1))
    (check-eq "vt->" (vt-to-list (vt-> a b)) '(0 1 0))
    (check-eq "vt->=" (vt-to-list (vt->= a b)) '(1 1 1)))
  (let ((x (vt-from-sequence (list (vt-float-nan :float64) 1.0d0))))
    (check-eq "vt-= NaN 不等（IEEE）" (vt-to-list (vt-= x x)) '(0 1))
    (check "vt-float-nan-=：NaN==NaN 判真"
           (eq (vt-float-nan-= (vt-ref x 0) (vt-ref x 0)) t))))

;;; ==================================================================
;;; 7. 归约与统计（reduce-stats.lisp）
;;; ==================================================================

(deftest test-reduce-sum-prod
    "reduce: vt-sum / vt-prod（§4.4 dtype 提升与空归约三分类 D11）"
  (let ((a (vt-reshape (vt-arange 6 :dtype :int64) '(2 3))))
    (check "vt-sum 全归约=15" (= (vt-item (vt-sum a)) 15))
    (check-eq "vt-sum axis 0" (vt-to-list (vt-sum a :axis 0)) '(3 5 7))
    (check-eq "vt-sum axis 1" (vt-to-list (vt-sum a :axis 1)) '(3 12))
    (check-eq "vt-sum axis -1 = axis 1" (vt-to-list (vt-sum a :axis -1)) '(3 12))
    (check-shape "vt-sum keepdims 保留 (2 1)" (vt-sum a :axis 1 :keepdims t) '(2 1))
    (check "vt-prod 全归约=720（1..6 连乘）"
           (= (vt-item (vt-prod (vt-arange 6 :start 1 :dtype :int64))) 720))
    (check-eq "vt-prod axis 0" (vt-to-list (vt-prod a :axis 0)) '(0 4 10)))
  (check-dtype "vt-sum int8 → int64（§4.4）"
               (vt-sum (vt-ones '(3) :dtype :int8)) :int64)
  (check-dtype "vt-sum float32 → float32（不升 float64）"
               (vt-sum (vt-ones '(3) :dtype :float32)) :float32)
  (check-dtype "vt-prod int32 → int64" (vt-prod (vt-ones '(3) :dtype :int32)) :int64)
  ;; 空归约三分类（D11）：单位元族填充单位元
  (check "vt-sum 空张量 → 0" (= (vt-item (vt-sum (vt-zeros '(0)))) 0.0d0))
  (check "vt-prod 空张量 → 1" (= (vt-item (vt-prod (vt-zeros '(0)))) 1.0d0))
  (check "vt-sum (0,3) axis 0 → (3) 零填充"
         (allclose-lists (vt-to-list (vt-sum (vt-zeros '(0 3)) :axis 0)) '(0.0d0 0.0d0 0.0d0)))
  (check-error "vt-amax (0,3) axis 0 输出非空 → 报错"
               (vt-amax (vt-zeros '(0 3)) :axis 0))
  (check "vt-sum (3,0) axis 1 → (3) 零填充"
         (allclose-lists (vt-to-list (vt-sum (vt-zeros '(3 0)) :axis 1)) '(0.0d0 0.0d0 0.0d0))))

(deftest test-reduce-max-min-arg
    "reduce: vt-amax / vt-amin / vt-argmax / vt-argmin（首个极值；NaN 下标行为 D14）"
  (let ((a (vt-from-sequence '(3.0d0 1.0d0 4.0d0 1.0d0 5.0d0))))
    (check "vt-amax=5" (= (vt-item (vt-amax a)) 5.0d0))
    (check "vt-amin=1" (= (vt-item (vt-amin a)) 1.0d0))
    (check "vt-argmax=4（首个）" (= (vt-item (vt-argmax a)) 4))
    (check "vt-argmin=1（首个）" (= (vt-item (vt-argmin a)) 1)))
  (let* ((m (vt-reshape (vt-from-sequence '(1.0d0 7.0d0 3.0d0 9.0d0 2.0d0 8.0d0)) '(2 3))))
    (check-eq "vt-amax axis 0" (vt-to-list (vt-amax m :axis 0)) '(9.0d0 7.0d0 8.0d0))
    (check-eq "vt-argmax axis 1（首个极值）" (vt-to-list (vt-argmax m :axis 1)) '(1 0))
    (check-shape "vt-amin keepdims" (vt-amin m :axis 0 :keepdims t) '(1 3)))
  (let ((x (vt-from-sequence (list 1.0d0 (vt-float-nan :float64) 5.0d0))))
    ;; numpy: argmax 遇 NaN 返回首个 NaN 下标；amax 返回 NaN
    (check "vt-argmax NaN 优先（首个 NaN 下标=1）" (= (vt-item (vt-argmax x)) 1))
    (check "vt-amax 含 NaN → NaN" (vt-float-nan-p (vt-item (vt-amax x))))
    (check "vt-argmin NaN 优先" (= (vt-item (vt-argmin x)) 1)))
  (check-error "vt-argmax 空归约区报错" (vt-argmax (vt-zeros '(0))))
  (check-error "vt-amax 空张量报错" (vt-amax (vt-zeros '(0)))))

(deftest test-reduce-all-any-boolean
    "reduce: vt-all / vt-any / vt-isfinite / isinf / isnan / isposinf / isneginf"
  (check "vt-all 全真" (= (vt-item (vt-all (vt-ones '(3)))) 1.0d0))
  (check "vt-all 含假 → 0" (= (vt-item (vt-all (vt-from-sequence '(1 0 1)))) 0.0d0))
  (check "vt-any 有真 → 1" (= (vt-item (vt-any (vt-from-sequence '(0 1 0)))) 1.0d0))
  (check "vt-any 全假 → 0" (= (vt-item (vt-any (vt-zeros '(3)))) 0.0d0))
  (check "vt-all 空张量 → 1（单位元）" (= (vt-item (vt-all (vt-zeros '(0)))) 1.0d0))
  (check "vt-any 空张量 → 0" (= (vt-item (vt-any (vt-zeros '(0)))) 0.0d0))
  (let ((a (vt-from-sequence (list 1.0d0 (vt-float-nan :float64)
                                   (vt-float-pos-inf :float64) (vt-float-neg-inf :float64)))))
    (check-eq "vt-isfinite" (vt-to-list (vt-isfinite a)) '(1 0 0 0))
    (check-eq "vt-isinf" (vt-to-list (vt-isinf a)) '(0 0 1 1))
    (check-eq "vt-isnan" (vt-to-list (vt-isnan a)) '(0 1 0 0))
    (check-eq "vt-isposinf" (vt-to-list (vt-isposinf a)) '(0 0 1 0))
    (check-eq "vt-isneginf" (vt-to-list (vt-isneginf a)) '(0 0 0 1))))

(deftest test-reduce-mean-var
    "reduce: vt-mean / vt-average / vt-var / vt-std（整数→float64；ddof）"
  (let ((a (vt-from-sequence '(1.0d0 2.0d0 3.0d0 4.0d0))))
    (check "vt-mean=2.5" (= (vt-item (vt-mean a)) 2.5d0))
    (check "vt-var ddof0=1.25" (allclose-lists (list (vt-item (vt-var a))) '(1.25d0)))
    (check "vt-var ddof1=1.6667" (allclose-lists (list (vt-item (vt-var a :ddof 1))) (list (/ 5.0d0 3.0d0))))
    (check "vt-std=sqrt(var)" (allclose-lists (list (vt-item (vt-std a))) '(1.118033988749895d0)))
    (check-dtype "vt-mean 整数输入 → float64" (vt-mean (vt-from-sequence '(1 2) :dtype :int64)) :float64)
    (check-dtype "vt-var 整数输入 → float64" (vt-var (vt-from-sequence '(1 2) :dtype :int64)) :float64)
    (check-dtype "vt-mean float32 → float32" (vt-mean (vt-ones '(2) :dtype :float32)) :float32)
    (check-eq "vt-mean axis 0" (vt-to-list (vt-mean (vt-reshape a '(2 2)) :axis 0))
              '(2.0d0 3.0d0))
    (check-eq "vt-average 加权" (vt-to-list (vt-average a (vt-from-sequence '(1 1 1 1) :dtype :float64)))
              '(2.5d0))
    (check-eq "vt-average 加权 [1 2 3 4] w=[4 3 2 1] → 2.0"
              (vt-to-list (vt-average a (vt-from-sequence '(4 3 2 1) :dtype :float64))) '(2.0d0))))

(deftest test-reduce-cum-median-percentile
    "reduce: vt-cumsum / vt-cumprod / vt-median / vt-percentile / vt-quantile / vt-ptp"
  (let ((a (vt-from-sequence '(1 2 3) :dtype :int64)))
    (check-eq "vt-cumsum" (vt-to-list (vt-cumsum a)) '(1 3 6))
    (check-eq "vt-cumprod" (vt-to-list (vt-cumprod a)) '(1 2 6))
    (check-dtype "vt-cumsum 整数保持 :int64（numpy cumsum 不升级）"
                 (vt-cumsum a) :int64))
  (check-eq "vt-cumsum 2d axis 1"
            (vt-to-list (vt-cumsum (vt-reshape (vt-arange 6 :dtype :int64) '(2 3)) :axis 1))
            '((0.0d0 1.0d0 3.0d0) (3.0d0 7.0d0 12.0d0)))
  (check "vt-median 偶数个 → 平均" (= (vt-item (vt-median (vt-from-sequence '(1.0d0 2.0d0 3.0d0 4.0d0)))) 2.5d0))
  (check "vt-median 奇数个 → 中位" (= (vt-item (vt-median (vt-from-sequence '(1.0d0 2.0d0 3.0d0)))) 2.0d0))
  (let ((a (vt-from-sequence '(1.0d0 2.0d0 3.0d0 4.0d0))))
    (check "vt-percentile 25 → 1.75（linear 插值）"
           (= (vt-item (vt-percentile a 25)) 1.75d0))
    (check "vt-percentile 50 → 2.5" (= (vt-item (vt-percentile a 50)) 2.5d0))
    (check "vt-percentile 75 → 3.25" (= (vt-item (vt-percentile a 75)) 3.25d0))
    ;; 注：vt-percentile 的 q 仅支持实数标量（不支持序列/张量 q），为已知 API 限制
    (check "vt-quantile 0.5 = percentile 50" (= (vt-item (vt-quantile a 0.5d0)) 2.5d0))
    (check "vt-ptp=3" (= (vt-item (vt-ptp a)) 3.0d0))))

(deftest test-reduce-nan-family
    "reduce: vt-nansum/nanprod/nanmax/nanmin/nanargmax/nanargmin/nanmean/nanvar/nanstd/nanmedian/nancumsum/nancumprod/nanpercentile"
  (let* ((a (vt-from-sequence (list 1.0d0 (vt-float-nan :float64) 3.0d0)))
         (m (vt-reshape (vt-from-sequence (list 1.0d0 (vt-float-nan :float64) 3.0d0 4.0d0)) '(2 2))))
    (check "vt-nansum=4" (= (vt-item (vt-nansum a)) 4.0d0))
    (check "vt-nanprod=3" (= (vt-item (vt-nanprod a)) 3.0d0))
    (check "vt-nanmax=3" (= (vt-item (vt-nanmax a)) 3.0d0))
    (check "vt-nanmin=1" (= (vt-item (vt-nanmin a)) 1.0d0))
    (check "vt-nanargmax=2" (= (vt-item (vt-nanargmax a)) 2))
    (check "vt-nanargmin=0" (= (vt-item (vt-nanargmin a)) 0))
    (check "vt-nanmean=2" (= (vt-item (vt-nanmean a)) 2.0d0))
    (check "vt-nanvar=1" (allclose-lists (list (vt-item (vt-nanvar a))) '(1.0d0)))
    (check "vt-nanstd=1" (allclose-lists (list (vt-item (vt-nanstd a))) '(1.0d0)))
    (check "vt-nanmedian=2" (= (vt-item (vt-nanmedian a)) 2.0d0))
    (check-eq "vt-nancumsum 跳过 NaN" (vt-to-list (vt-nancumsum a)) '(1.0d0 1.0d0 4.0d0))
    (check-eq "vt-nancumprod 跳过 NaN" (vt-to-list (vt-nancumprod a)) '(1.0d0 1.0d0 3.0d0))
    (check "vt-nanpercentile 50=2" (= (vt-item (vt-nanpercentile a 50)) 2.0d0))
    (check "vt-nanquantile 0.5=2" (= (vt-item (vt-nanquantile a 0.5d0)) 2.0d0))
    (check-eq "vt-nanmean axis 0 忽略 NaN 列"
              (vt-to-list (vt-nanmean m :axis 0)) '(2.0d0 4.0d0))))

(deftest test-reduce-sort-argsort
    "reduce: vt-sort / vt-argsort（NaN 末尾；降序=升序逆序，D13）"
  (let ((a (vt-from-sequence '(3.0d0 1.0d0 2.0d0))))
    (check-eq "vt-sort 升序" (vt-to-list (vt-sort a)) '(1.0d0 2.0d0 3.0d0))
    (check-eq "vt-argsort" (vt-to-list (vt-argsort a)) '(1 2 0)))
  (let ((a (vt-from-sequence (list 3.0d0 (vt-float-nan :float64) 1.0d0))))
    (check-eq "vt-sort NaN 末尾" (vt-to-list (vt-sort a))
              (list 1.0d0 3.0d0 (vt-float-nan :float64)))
    (check-eq "vt-argsort NaN 末尾" (vt-to-list (vt-argsort a)) '(2 0 1)))
  (let ((m (vt-reshape (vt-from-sequence '(3.0d0 1.0d0 2.0d0 2.0d0 5.0d0 6.0d0)) '(2 3))))
    (check-eq "vt-sort axis -1 每行排序" (vt-to-list (vt-sort m))
              '((1.0d0 2.0d0 3.0d0) (2.0d0 5.0d0 6.0d0)))
    (check-eq "vt-sort axis 0 每列排序" (vt-to-list (vt-sort m :axis 0))
              '((2.0d0 1.0d0 2.0d0) (3.0d0 5.0d0 6.0d0)))))

(deftest test-reduce-signal
    "reduce: vt-diff / vt-ediff1d / vt-gradient / vt-trapz / vt-convolve / vt-correlate / vt-interp / vt-histogram"
  (let ((a (vt-from-sequence '(1 4 9 16) :dtype :int64)))
    (check-eq "vt-diff n=1" (vt-to-list (vt-diff a)) '(3 5 7))
    (check-eq "vt-diff n=2" (vt-to-list (vt-diff a :n 2)) '(2 2))
    (check-eq "vt-ediff1d" (vt-to-list (vt-ediff1d a)) '(3 5 7)))
  (check "vt-gradient 中心差分+单侧边界"
         (allclose-lists (vt-to-list (vt-gradient (vt-from-sequence '(1.0d0 2.0d0 4.0d0 7.0d0))))
                         '(1.0d0 1.5d0 2.5d0 3.0d0)))
  (check "vt-gradient spacing 2.0"
         (allclose-lists (vt-to-list (vt-gradient (vt-from-sequence '(1.0d0 2.0d0 4.0d0 7.0d0)) :spacing 2.0d0))
                         '(0.5d0 0.75d0 1.25d0 1.5d0)))
  (check "vt-trapz 默认 dx=1" (= (vt-item (vt-trapz (vt-from-sequence '(1.0d0 2.0d0 3.0d0)))) 4.0d0))
  (check "vt-trapz 给 x" (= (vt-item (vt-trapz (vt-from-sequence '(1.0d0 2.0d0 3.0d0))
                                               :x (vt-from-sequence '(0.0d0 2.0d0 4.0d0))))
                            8.0d0))
  (check-eq "vt-convolve full"
            (vt-to-list (vt-convolve (vt-from-sequence '(1 2 3) :dtype :int64)
                                     (vt-from-sequence '(0 1) :dtype :int64)))
            '(0 1 2 3))
  (check-eq "vt-convolve same"
            (vt-to-list (vt-convolve (vt-from-sequence '(1 2 3) :dtype :int64)
                                     (vt-from-sequence '(0 1) :dtype :int64) :mode :same))
            '(0 1 2))
  (check-eq "vt-convolve valid"
            (vt-to-list (vt-convolve (vt-from-sequence '(1 2 3) :dtype :int64)
                                     (vt-from-sequence '(0 1) :dtype :int64) :mode :valid))
            '(1 2))
  (check-eq "vt-correlate full"
            (vt-to-list (vt-correlate (vt-from-sequence '(1 2 3) :dtype :int64)
                                      (vt-from-sequence '(0 1) :dtype :int64)))
            '(1 2 3 0))
  (check-eq "vt-correlate valid"
            (vt-to-list (vt-correlate (vt-from-sequence '(1 2 3) :dtype :int64)
                                      (vt-from-sequence '(0 1) :dtype :int64) :mode :valid))
            '(2 3))
  (check "vt-interp 中点与端点（left/right 缺省用端点值，§9.3）"
         (allclose-lists
          (vt-to-list (vt-interp (vt-from-sequence '(0.5d0 1.0d0 2.5d0 -1.0d0))
                                 (vt-from-sequence '(0.0d0 1.0d0 2.0d0))
                                 (vt-from-sequence '(0.0d0 10.0d0 20.0d0))))
          '(5.0d0 10.0d0 20.0d0 0.0d0)))
  (multiple-value-bind (h e)
      (vt-histogram (vt-from-sequence '(1.0d0 2.0d0 3.0d0 4.0d0)) :bins 2)
    (check-eq "vt-histogram 计数" (vt-to-list h) '(2 2))
    (check-eq "vt-histogram 边界" (vt-to-list e) '(1.0d0 2.5d0 4.0d0)))
  (multiple-value-bind (h e)
      (vt-histogram (vt-from-sequence '(1.0d0 2.0d0 3.0d0 4.0d0)) :bins 2 :density t)
    (check "vt-histogram density 归一化" (allclose-lists (vt-to-list h) (list (/ 1.0d0 3.0d0) (/ 1.0d0 3.0d0))))
    (check-eq "vt-histogram density 边界不变" (vt-to-list e) '(1.0d0 2.5d0 4.0d0))))

(deftest test-reduce-isclose
    "reduce: vt-isclose / vt-allclose（rtol 1e-5 atol 1e-8 默认；inf==inf）"
  (let ((a (vt-from-sequence '(1.0d0 1.00001d0 100.0d0)))
        (b (vt-from-sequence '(1.0d0 1.0d0 100.001d0))))
    (check-eq "vt-isclose 默认容差" (vt-to-list (vt-isclose a b)) '(1 1 1)))
  (let ((a (vt-from-sequence (list (vt-float-nan :float64) (vt-float-pos-inf :float64) 1.0d0)))
        (b (vt-from-sequence (list (vt-float-nan :float64) (vt-float-pos-inf :float64) 1.1d0))))
    (check-eq "vt-isclose NaN 不判等，inf==inf 判等"
              (vt-to-list (vt-isclose a b)) '(0 1 0)))
  (check "vt-allclose 真" (eq (vt-allclose (vt-ones '(3)) (vt-ones '(3))) t))
  (check "vt-allclose 假" (not (vt-allclose (vt-ones '(3)) (vt-full '(3) 2.0d0)))))

(deftest test-reduce-stat-ext
    "reduce: vt-cov / vt-corrcoef / vt-cross / vt-isin / vt-in1d（extensions2/3/setops）"
  (multiple-value-bind (c)
      (vt-cov (vt-reshape (vt-from-sequence '(1.0d0 2.0d0 3.0d0 4.0d0 5.0d0 6.0d0)) '(2 3)))
    (check-eq "vt-cov 行变量同调 → 全 1" (vt-to-list c) '((1.0d0 1.0d0) (1.0d0 1.0d0))))
  (multiple-value-bind (c)
      (vt-cov (vt-from-sequence '(1.0d0 2.0d0 3.0d0)) :y (vt-from-sequence '(5.0d0 7.0d0 9.0d0)))
    (check-eq "vt-cov y 参数 → [[1 2][2 4]]" (vt-to-list c) '((1.0d0 2.0d0) (2.0d0 4.0d0))))
  (multiple-value-bind (c)
      (vt-corrcoef (vt-from-sequence '(1.0d0 2.0d0 3.0d0)) :y (vt-from-sequence '(5.0d0 7.0d0 9.0d0)))
    (check-eq "vt-corrcoef 完全正相关 → 全 1" (vt-to-list c) '((1.0d0 1.0d0) (1.0d0 1.0d0))))
  (check-eq "vt-cross 叉积"
            (vt-to-list (vt-cross (vt-from-sequence '(1.0d0 2.0d0 3.0d0))
                                  (vt-from-sequence '(4.0d0 5.0d0 6.0d0))))
            '(-3.0d0 6.0d0 -3.0d0))
  (check-eq "vt-isin 成员判定"
            (vt-to-list (vt-isin (vt-from-sequence '(1 2 3 4) :dtype :int64)
                                 (vt-from-sequence '(2 4) :dtype :int64)))
            '(0 1 0 1))
  (check-eq "vt-isin :invert 取反"
            (vt-to-list (vt-isin (vt-from-sequence '(1 2 3 4) :dtype :int64)
                                 (vt-from-sequence '(2 4) :dtype :int64) :invert t))
            '(1 0 1 0)))

;;; ==================================================================
;;; 8. 线性代数（linalg.lisp + simd-matmul.lisp）
;;; ==================================================================

(deftest test-linalg-matmul-dot
    "linalg: vt-matmul / vt-@ / vt-dot / vt-vdot / vt-inner / vt-outer / vt-tensordot / vt-trace / 批量 matmul"
  (let ((a (vt-from-sequence '((1.0d0 2.0d0) (3.0d0 4.0d0))))
        (b (vt-from-sequence '((5.0d0 6.0d0) (7.0d0 8.0d0)))))
    (check-eq "vt-matmul 2d×2d" (vt-to-list (vt-matmul a b)) '((19.0d0 22.0d0) (43.0d0 50.0d0)))
    (check-eq "vt-@ 等价 matmul" (vt-to-list (vt-@ a b)) '((19.0d0 22.0d0) (43.0d0 50.0d0)))
    (check-eq "vt-dot 2d 同 matmul" (vt-to-list (vt-dot a b)) '((19.0d0 22.0d0) (43.0d0 50.0d0)))
    (check-eq "vt-einsum ij,jk->ik" (vt-to-list (vt-einsum "ij,jk->ik" a b))
              '((19.0d0 22.0d0) (43.0d0 50.0d0))))
  (let ((u (vt-from-sequence '(1.0d0 2.0d0 3.0d0)))
        (v (vt-from-sequence '(4.0d0 5.0d0 6.0d0))))
    (check "vt-matmul 1d×1d → 标量 32" (= (vt-item (vt-matmul u v)) 32.0d0))
    (check "vt-dot 1d×1d → 32" (= (vt-item (vt-dot u v)) 32.0d0))
    (check "vt-vdot 展平内积=32（忽略形状）" (= (vt-item (vt-vdot u v)) 32.0d0))
    (check "vt-inner 内积=11" (= (vt-item (vt-inner (vt-from-sequence '(1.0d0 2.0d0))
                                                    (vt-from-sequence '(3.0d0 4.0d0))))
                               11.0d0))
    (check-eq "vt-outer (2)×(3) → (2 3)"
              (vt-to-list (vt-outer (vt-from-sequence '(1.0d0 2.0d0))
                                    (vt-from-sequence '(3.0d0 4.0d0 5.0d0))))
              '((3.0d0 4.0d0 5.0d0) (6.0d0 8.0d0 10.0d0))))
  (let ((u (vt-from-sequence '(1.0d0 2.0d0 3.0d0)))
        (a23 (vt-from-sequence '((1.0d0 0.0d0 1.0d0) (0.0d0 1.0d0 1.0d0)))))
    (check-eq "vt-matmul 1d×2d 行提升（u@a23）" (vt-to-list (vt-matmul (vt-from-sequence '(1.0d0 2.0d0)) a23))
              '(1.0d0 2.0d0 3.0d0))
    (check-eq "vt-matmul 2d×1d 列提升（a23@u）" (vt-to-list (vt-matmul a23 u))
              '(4.0d0 5.0d0)))
  (let ((e (vt-einsum "ij->ji" (vt-from-sequence '((1.0d0 2.0d0) (3.0d0 4.0d0))))))
    (check-eq "vt-einsum ij->ji 转置" (vt-to-list e) '((1.0d0 3.0d0) (2.0d0 4.0d0))))
  (let ((v (vt-einsum "i->" (vt-from-sequence '(1.0d0 2.0d0 3.0d0)))))
    (check "vt-einsum i-> 求和=6" (= (vt-item v) 6.0d0)))
  (check-eq "vt-einsum ii->i 对角线"
            (vt-to-list (vt-einsum "ii->i" (vt-from-sequence '((1.0d0 2.0d0) (3.0d0 4.0d0)))))
            '(1.0d0 4.0d0))
  (check-eq "vt-einsum ij,j->i"
            (vt-to-list (vt-einsum "ij,j->i" (vt-from-sequence '((1.0d0 2.0d0) (3.0d0 4.0d0)))
                                    (vt-from-sequence '(5.0d0 6.0d0))))
            '(17.0d0 39.0d0))
  (let* ((b1 (vt-from-sequence '((1.0d0 2.0d0) (3.0d0 4.0d0))))
         (b2 (vt-from-sequence '((5.0d0 6.0d0) (7.0d0 8.0d0))))
         (batch (vt-stack 0 b1 b2)))
    (check-eq "批量 matmul (2 2 2)（SIMD 路径，各 batch 独立相乘）"
              (vt-to-list (vt-matmul batch batch))
              '(((7.0d0 10.0d0) (15.0d0 22.0d0)) ((67.0d0 78.0d0) (91.0d0 106.0d0)))))
  (check "vt-trace = 5" (= (vt-item (vt-trace (vt-from-sequence '((1.0d0 2.0d0) (3.0d0 4.0d0))))) 5.0d0))
  (check "vt-norm (3 4) = 5" (= (vt-item (vt-norm (vt-from-sequence '(3.0d0 4.0d0)))) 5.0d0))
  (check "vt-l1-norm = 7" (= (vt-item (vt-l1-norm (vt-from-sequence '(3.0d0 4.0d0)))) 7.0d0))
  (check "vt-frobenius-norm = sqrt(30)"
         (allclose-lists (list (vt-item (vt-frobenius-norm (vt-from-sequence '((1.0d0 2.0d0) (3.0d0 4.0d0))))))
                         (list (sqrt 30.0d0))))
  (check-eq "vt-tensordot axes=1 等价 matmul"
            (vt-to-list (vt-tensordot (vt-from-sequence '((1.0d0 2.0d0) (3.0d0 4.0d0)))
                                      (vt-from-sequence '((5.0d0 6.0d0) (7.0d0 8.0d0))) :axes 1))
            '((19.0d0 22.0d0) (43.0d0 50.0d0)))
  (check "vt-tensordot axes=2 双轴收缩 → 标量 70"
         (= (vt-item (vt-tensordot (vt-from-sequence '((1.0d0 2.0d0) (3.0d0 4.0d0)))
                                   (vt-from-sequence '((5.0d0 6.0d0) (7.0d0 8.0d0))) :axes 2))
            70.0d0)))

(deftest test-linalg-solvers
    "linalg: vt-solve / vt-inv / vt-det / vt-lu / vt-qr / vt-svd / vt-matrix-rank / vt-cholesky / vt-pinv / vt-lstsq / vt-matrix-power / vt-cond / vt-multi-dot"
  (let ((a (vt-from-sequence '((2.0d0 0.0d0) (0.0d0 4.0d0)))))
    (check-eq "vt-solve 对角方程" (vt-to-list (vt-solve a (vt-from-sequence '(2.0d0 8.0d0))))
              '(1.0d0 2.0d0))
    (check-eq "vt-inv 对角逆" (vt-to-list (vt-inv a)) '((0.5d0 0.0d0) (0.0d0 0.25d0)))
    (check-eq "vt-matrix-power 2" (vt-to-list (vt-matrix-power a 2))
              '((4.0d0 0.0d0) (0.0d0 16.0d0)))
    (check-eq "vt-matrix-power 0 → 单位阵" (vt-to-list (vt-matrix-power a 0))
              '((1.0d0 0.0d0) (0.0d0 1.0d0)))
    (check-eq "vt-matrix-power 1 → 原矩阵" (vt-to-list (vt-matrix-power a 1))
              '((2.0d0 0.0d0) (0.0d0 4.0d0))))
  (check "vt-det = -2" (= (vt-item (vt-det (vt-from-sequence '((1.0d0 2.0d0) (3.0d0 4.0d0))))) -2.0d0))
  ;; LU：返回 (values packed piv nswaps)——packed 下三角存 L 的乘数
  ;; （单位对角），上三角（含对角）存 U；piv 为行交换向量
  (let ((a (vt-from-sequence '((4.0d0 2.0d0) (2.0d0 3.0d0)))))
    (multiple-value-bind (packed piv nswaps) (vt-lu a)
      (check "vt-lu packed 为 2x2 张量"
             (and (vt-p packed) (equal (vt-shape packed) '(2 2))))
      (check "vt-lu piv 为行交换向量" (and (vectorp piv) (= (length piv) 2)))
      (check "vt-lu nswaps 为整数" (integerp nswaps))
      ;; 由 packed 还原 L（单位下三角）与 U（上三角），验证 L@U = a
      (let* ((l (vt-from-sequence (list (list 1.0d0 0.0d0)
                                        (list (vt-ref packed 1 0) 1.0d0))))
             (u (vt-from-sequence (list (list (vt-ref packed 0 0) (vt-ref packed 0 1))
                                        (list 0.0d0 (vt-ref packed 1 1))))))
        (check "vt-lu L@U 还原 a（主对角占优无需换行）"
               (allclose-lists (vt-to-list (vt-matmul l u))
                               '((4.0d0 2.0d0) (2.0d0 3.0d0)))))))
  (let ((a (vt-from-sequence '((1.0d0 2.0d0) (3.0d0 4.0d0)))))
    (multiple-value-bind (q r) (vt-qr a)
      (check "vt-qr q@r = a"
             (allclose-lists (vt-to-list (vt-matmul q r)) '((1.0d0 2.0d0) (3.0d0 4.0d0))))
      (check "vt-qr q 正交（q^t q = I）"
             (allclose-lists (vt-to-list (vt-matmul (vt-transpose q) q))
                             '((1.0d0 0.0d0) (0.0d0 1.0d0))))))
  (multiple-value-bind (u s v) (vt-svd (vt-from-sequence '((3.0d0 0.0d0) (0.0d0 4.0d0))))
    (declare (ignore u))
    (check "vt-svd 奇异值 {4,3}" (allclose-lists (vt-to-list s) '(4.0d0 3.0d0)))
    (check "vt-svd vt 为正交阵"
           (allclose-lists (vt-to-list (vt-matmul v (vt-transpose v)))
                           '((1.0d0 0.0d0) (0.0d0 1.0d0)))))
  (check "vt-matrix-rank 满秩=2"
         (= (vt-item (vt-matrix-rank (vt-from-sequence '((1.0d0 0.0d0) (0.0d0 1.0d0))))) 2))
  (check "vt-matrix-rank 奇异=1"
         (= (vt-item (vt-matrix-rank (vt-from-sequence '((1.0d0 2.0d0) (2.0d0 4.0d0))))) 1))
  (multiple-value-bind (l) (vt-cholesky (vt-from-sequence '((4.0d0 2.0d0) (2.0d0 3.0d0))))
    (check "vt-cholesky l@l^t = a"
           (allclose-lists (vt-to-list (vt-matmul l (vt-transpose l)))
                           '((4.0d0 2.0d0) (2.0d0 3.0d0)))))
  (check-eq "vt-pinv 伪逆"
            (vt-to-list (vt-pinv (vt-from-sequence '((1.0d0 0.0d0) (0.0d0 0.0d0)))))
            '((1.0d0 0.0d0) (0.0d0 0.0d0)))
  (multiple-value-bind (x) (vt-lstsq (vt-from-sequence '((1.0d0 0.0d0) (1.0d0 1.0d0)))
                                     (vt-from-sequence '(2.0d0 3.0d0)))
    (check "vt-lstsq 最小二乘解 [2 1]" (allclose-lists (vt-to-list x) '(2.0d0 1.0d0))))
  (check "vt-cond diag(1,2) 条件数=2"
         (allclose-lists (list (vt-item (vt-cond (vt-from-sequence '((1.0d0 0.0d0) (0.0d0 2.0d0))))))
                         '(2.0d0)))
  (check-eq "vt-multi-dot 链式乘法"
            (vt-to-list (vt-multi-dot (list (vt-from-sequence '((1.0d0 2.0d0)))
                                            (vt-from-sequence '((3.0d0 4.0d0) (5.0d0 6.0d0)))
                                            (vt-from-sequence '((7.0d0) (8.0d0)))))
            )
            '((219.0d0))))

(deftest test-linalg-eigen
    "linalg: vt-eig / vt-eigvals / vt-eigvalsh（对称矩阵 Jacobi）"
  (let ((a (vt-from-sequence '((2.0d0 1.0d0) (1.0d0 2.0d0)))))
    (multiple-value-bind (w v) (vt-eig a)
      (check "vt-eig 特征值 {3,1}" (allclose-lists (vt-to-list (vt-sort w)) '(1.0d0 3.0d0)))
      ;; v 的列与未排序 w 一一对应：a@v = v@diag(w)
      (check "vt-eig a@v ≈ v@diag(w)"
             (allclose-lists
              (vt-to-list (vt-matmul a v))
              (vt-to-list (vt-matmul v (vt-diag w))))))
    (check "vt-eigvals {3,1}" (allclose-lists (vt-to-list (vt-sort (vt-eigvals a))) '(1.0d0 3.0d0)))
    (check "vt-eigvalsh 升序 {1,3}" (allclose-lists (vt-to-list (vt-eigvalsh a)) '(1.0d0 3.0d0)))))

;;; ==================================================================
;;; 9. 神经网络层（nn.lisp + extensions2.lisp）
;;; ==================================================================

(deftest test-nn-activations
    "nn: sigmoid / relu / leaky-relu / swish / softplus / gelu / mish / hard-tanh / hard-sigmoid"
  (check-eq "vt-sigmoid(0)=0.5" (vt-to-list (vt-sigmoid (vt-zeros '(1)))) '(0.5d0))
  (check "vt-sigmoid 大值饱和" (allclose-lists (vt-to-list (vt-sigmoid (vt-full '(1) 1000.0d0))) '(1.0d0)))
  (check-eq "vt-relu" (vt-to-list (vt-relu (vt-from-sequence '(-2.0d0 3.0d0)))) '(0.0d0 3.0d0))
  (check-eq "vt-leaky-relu alpha=0.01" (vt-to-list (vt-leaky-relu (vt-from-sequence '(-2.0d0 3.0d0))))
            '(-0.02d0 3.0d0))
  (check "vt-swish(1)=x·sigmoid(x)≈0.7311"
         (allclose-lists (vt-to-list (vt-swish (vt-ones '(1)))) '(0.7310585786300049d0)))
  (check "vt-softplus(0)=ln2" (allclose-lists (vt-to-list (vt-softplus (vt-zeros '(1))))
                                              '(0.6931471805599453d0)))
  (check "vt-gelu(0)=0" (allclose-lists (vt-to-list (vt-gelu (vt-zeros '(1)))) '(0.0d0)))
  (check "vt-gelu(1)≈0.8412（tanh 近似）"
         (allclose-lists (vt-to-list (vt-gelu (vt-ones '(1)))) '(0.8411919906082768d0) 1.0d-4))
  (check "vt-mish(0)=0" (allclose-lists (vt-to-list (vt-mish (vt-zeros '(1)))) '(0.0d0)))
  (check "vt-mish(1)≈0.8651"
         (allclose-lists (vt-to-list (vt-mish (vt-ones '(1)))) '(0.8650983524613677d0) 1.0d-5))
  (check-eq "vt-hard-tanh 截断 [-1,1]"
            (vt-to-list (vt-hard-tanh (vt-from-sequence '(-2.0d0 0.5d0 2.0d0))))
            '(-1.0d0 0.5d0 1.0d0))
  (check-eq "vt-hard-sigmoid clip(x/5+0.5,0,1)"
            (vt-to-list (vt-hard-sigmoid (vt-from-sequence '(0.0d0 2.5d0 -2.5d0 10.0d0))))
            '(0.5d0 1.0d0 0.0d0 1.0d0)))

(deftest test-nn-losses-norm
    "nn: softmax / log-softmax / mse / bce / cross-entropy / one-hot / standardize / layer-norm / apply-along-axis / topk"
  (check "vt-softmax 概率和为 1"
         (allclose-lists (list (vt-item (vt-sum (vt-softmax (vt-from-sequence '(1.0d0 2.0d0 3.0d0))))))
                         '(1.0d0)))
  (check "vt-softmax 数值稳定（大值不溢出）"
         (allclose-lists (vt-to-list (vt-softmax (vt-full '(3) 1000.0d0)))
                         (list (/ 1.0d0 3.0d0) (/ 1.0d0 3.0d0) (/ 1.0d0 3.0d0))))
  (check "vt-log-softmax = log(softmax)"
         (allclose-lists (vt-to-list (vt-log-softmax (vt-from-sequence '(1.0d0 2.0d0 3.0d0))))
                         '(-2.40760596444438d0 -1.40760596444438d0 -0.40760596444438d0) 1.0d-6))
  (check "vt-mean-squared-error=0.5"
         (allclose-lists (list (vt-item (vt-mean-squared-error
                                         (vt-from-sequence '(1.0d0 2.0d0))
                                         (vt-from-sequence '(1.0d0 3.0d0)))))
                         '(0.5d0)))
  (check "vt-binary-cross-entropy=-ln(0.9)"
         (allclose-lists (list (vt-item (vt-binary-cross-entropy
                                         (vt-from-sequence '(1.0d0 0.0d0))
                                         (vt-from-sequence '(0.9d0 0.1d0)))))
                         (list (- (log 0.9d0))) 1.0d-6))
  (check "vt-cross-entropy 同 bce（概率输入）"
         (allclose-lists (list (vt-item (vt-cross-entropy
                                         (vt-from-sequence '(1.0d0 0.0d0))
                                         (vt-from-sequence '(0.9d0 0.1d0)))))
                         (list (- (log 0.9d0))) 1.0d-6))
  (check-eq "vt-one-hot (0 2) 3 类"
            (vt-to-list (vt-one-hot (vt-from-sequence '(0 2) :dtype :int64) 3))
            '((1.0d0 0.0d0 0.0d0) (0.0d0 0.0d0 1.0d0)))
  (check "vt-standardize 标准化 → 均值 0 方差 1"
         (allclose-lists (vt-to-list (vt-standardize (vt-from-sequence '(1.0d0 2.0d0 3.0d0))))
                         '(-1.2247448713915892d0 0.0d0 1.2247448713915892d0)))
  (check "vt-layer-norm 末轴归一化同 standardize"
         (allclose-lists (vt-to-list (vt-layer-norm (vt-from-sequence '(1.0d0 2.0d0 3.0d0)) '(3)))
                         '(-1.2247409607003905d0 0.0d0 1.2247409607003905d0) 1.0d-5))
  (check-eq "vt-apply-along-axis 沿轴求和"
            (vt-to-list (vt-apply-along-axis
                         (lambda (x) (vt-item (vt-sum x))) 1
                         (vt-reshape (vt-arange 6 :dtype :int64) '(2 3))))
            '(3.0d0 12.0d0))
  (multiple-value-bind (vals idxs) (vt-topk (vt-from-sequence '(3.0d0 1.0d0 4.0d0 1.0d0 5.0d0 9.0d0 2.0d0 6.0d0)) 3)
    (check-eq "vt-topk values" (vt-to-list vals) '(9.0d0 6.0d0 5.0d0))
    (check-eq "vt-topk indices" (vt-to-list idxs) '(5 7 4))))

;;; ==================================================================
;;; 10. 集合操作（setops.lisp）
;;; ==================================================================

(deftest test-setops
    "setops: vt-unique(+flags) / intersect1d / union1d / setdiff1d / setxor1d / in1d / array-equal / array-equiv"
  (let ((a (vt-from-sequence '(3 1 3 2 1) :dtype :int64)))
    (check-eq "vt-unique 升序去重" (vt-to-list (vt-unique a)) '(1 2 3))
    (multiple-value-bind (u idx inv cnt) (vt-unique a :return-index t :return-inverse t :return-counts t)
      (declare (ignore u))
      (check-eq "vt-unique return-index 首次出现位置" (vt-to-list idx) '(1 3 0))
      (check-eq "vt-unique return-inverse 映射" (vt-to-list inv) '(2 0 2 1 0))
      (check-eq "vt-unique return-counts 计数" (vt-to-list cnt) '(2 1 2))))
  (check-eq "vt-intersect1d 交集"
            (vt-to-list (vt-intersect1d (vt-from-sequence '(3 1 2) :dtype :int64)
                                        (vt-from-sequence '(2 3 4) :dtype :int64)))
            '(2 3))
  (check-eq "vt-union1d 并集"
            (vt-to-list (vt-union1d (vt-from-sequence '(3 1 2) :dtype :int64)
                                    (vt-from-sequence '(2 3 4) :dtype :int64)))
            '(1 2 3 4))
  (check-eq "vt-setdiff1d 差集"
            (vt-to-list (vt-setdiff1d (vt-from-sequence '(1 2 3) :dtype :int64)
                                      (vt-from-sequence '(2) :dtype :int64)))
            '(1 3))
  (check-eq "vt-setxor1d 对称差"
            (vt-to-list (vt-setxor1d (vt-from-sequence '(1 2 3) :dtype :int64)
                                     (vt-from-sequence '(2 4) :dtype :int64)))
            '(1 3 4))
  (check-eq "vt-in1d 成员测试"
            (vt-to-list (vt-in1d (vt-from-sequence '(1 2) :dtype :int64)
                                 (vt-from-sequence '(2 3) :dtype :int64)))
            '(0 1))
  (check "vt-array-equal 同形状同值 → T"
         (eq (vt-array-equal (vt-from-sequence '(1 2) :dtype :int64)
                             (vt-from-sequence '(1 2) :dtype :int64)) t))
  (check "vt-array-equal 值不同 → NIL"
         (not (vt-array-equal (vt-from-sequence '(1 2) :dtype :int64)
                              (vt-from-sequence '(1 3) :dtype :int64))))
  (check "vt-array-equal 形状不同 → NIL"
         (not (vt-array-equal (vt-from-sequence '(1 2) :dtype :int64)
                              (vt-from-sequence '((1 2)) :dtype :int64))))
  (check "vt-array-equiv 广播等价 ((1 2)) vs (1 2) → T"
         (eq (vt-array-equiv (vt-from-sequence '((1 2)) :dtype :int64)
                             (vt-from-sequence '(1 2) :dtype :int64)) t))
  (check "vt-array-equiv 不可广播 → NIL"
         (not (vt-array-equiv (vt-from-sequence '(1 2 3) :dtype :int64)
                              (vt-from-sequence '(1 2) :dtype :int64)))))

;;; ==================================================================
;;; 11. 随机数（random.lisp）
;;; ==================================================================

(deftest test-random-core
    "random: with-seed 确定性 / uniform 区间 / normal / random-int / random（缺省）"
  (check "with-seed 可复现：同种子同序列"
         (equal (with-seed (42) (vt-to-list (vt-random-uniform '(4))))
                (with-seed (42) (vt-to-list (vt-random-uniform '(4))))))
  (let ((u (with-seed (7) (vt-random-uniform '(100) :low 2.0d0 :high 5.0d0))))
    (check "vt-random-uniform 落在 [low, high)"
           (and (>= (reduce #'min (vt-to-list u)) 2.0d0)
                (< (reduce #'max (vt-to-list u)) 5.0d0))))
  (check "vt-random-normal std=0 → 常量 mean 填充（§9.3 约定）"
         (allclose-lists (vt-to-list (vt-random-normal '(4) :mean 3.0d0 :std 0.0d0))
                         '(3.0d0 3.0d0 3.0d0 3.0d0)))
  (let ((n (with-seed (9) (vt-random-normal '(1000) :mean 5.0d0 :std 2.0d0))))
    (check "vt-random-normal 样本均值接近 mean"
           (< (abs (- (vt-item (vt-mean n)) 5.0d0)) 0.3d0)))
  (let ((ints (with-seed (11) (vt-random-int 0 10 :size 100))))
    (check "vt-random-int 落在 [low, high) 且为整数"
           (and (>= (reduce #'min (vt-to-list ints)) 0)
                (< (reduce #'max (vt-to-list ints)) 10)
                (every #'integerp (vt-to-list ints)))))
  (check "vt-random-integers 与 vt-random-int 同签名可用"
         (vt-p (with-seed (3) (vt-random-integers 0 5 :size 4))))
  (check "vt-random 缺省 [0,1) uniform" 
         (let ((r (with-seed (5) (vt-random '(50)))))
           (and (>= (reduce #'min (vt-to-list r)) 0.0d0)
                (< (reduce #'max (vt-to-list r)) 1.0d0))))
  (check "vt-random :dtype :float32 可用"
         (eq (vt-dtype (with-seed (5) (vt-random '(4) :dtype :float32))) :float32)))

(deftest test-random-choice-perm
    "random: choice（replace/p）/ permutation / shuffle（就地）/ multinomial / 生成器基础设施"
  (let ((a (vt-from-sequence '(1 2 3 4 5) :dtype :int64)))
    (check "vt-random-choice 默认有放回"
           (let ((c (with-seed (13) (vt-random-choice a :size 10))))
             (and (= (vt-size c) 10)
                  (every (lambda (x) (member x '(1 2 3 4 5))) (vt-to-list c)))))
    (check "vt-random-choice 无放回 → 元素互异"
           (let ((c (with-seed (13) (vt-random-choice a :size 3 :replace nil))))
             (= (vt-size (vt-unique c)) 3)))
    (check "vt-random-choice p 权重（p 全集中在下标 0）"
           (let ((c (with-seed (13) (vt-random-choice a :size 4
                                                   :p '(1.0d0 0.0d0 0.0d0 0.0d0 0.0d0)))))
             (every (lambda (x) (= x 1)) (vt-to-list c))))
    (check "vt-random-permutation(n) 是 0..n-1 的排列"
           (let ((p (with-seed (17) (vt-random-permutation 5))))
             (equal (vt-to-list (vt-sort p :axis -1)) '(0 1 2 3 4))))
    (let* ((orig (vt-copy a))
           (shuf (with-seed (19) (vt-random-shuffle (vt-copy a)))))
      (check "vt-random-shuffle 就地打乱但保持多重集"
             (and (allclose-lists (vt-to-list (vt-sort shuf :axis -1)) '(1 2 3 4 5))
                  (allclose-lists (vt-to-list orig) '(1 2 3 4 5)))))
    (let ((m (with-seed (23) (vt-random-multinomial 10 (vt-from-sequence '(0.2d0 0.3d0 0.5d0))))))
      (check "vt-random-multinomial 计数和 = n" (= (vt-item (vt-sum m)) 10)))
    (let ((g1 (make-generator 100)) (g2 (make-generator 100)))
      (check "make-generator 同种子序列一致"
             (equal (vt-to-list (vt-random '(4) :rng g1))
                    (vt-to-list (vt-random '(4) :rng g2)))))
    (let ((gens (spawn-generators 7 3)))
      (check "spawn-generators 返回 n 个生成器" (= (length gens) 3)))
    (check "make-seed-sequence 可创建且含熵属性"
           (let ((ss (make-seed-sequence 123)))
             (and ss (not (null (vt-seed-sequence-entropy ss))))))
    (check "seed-sequence-spawn 派生多个子序列"
           (= (length (seed-sequence-spawn (make-seed-sequence 5) 2)) 2))
    (check "generator-from-seed-sequence 可用"
           (vt-p (vt-random '(2) :rng (generator-from-seed-sequence (make-seed-sequence 9)))))
    (check "vt-make-random-state 可创建" (not (null (vt-make-random-state 42))))
    (progn (vt-random-seed 1) (check "vt-random-seed 设置后可生成" (vt-p (vt-random '(3)))))))

;;; ==================================================================
;;; 12. 参数契约（parcontract.lisp，:out 硬契约 H1–H5）
;;; ==================================================================

(deftest test-parcontract-out
    "contract: vt-check-out H1-H4 / vt-out-writable-p / 非连续 out 生效 / stride-0 out 报错 / H5 dtype 冲突"
  (let ((out (vt-ones '(2 3))))
    (check "vt-check-out 合法 out 返回原对象" (eq (vt-check-out out '(2 3) :float64) out)))
  (check-error "H2 形状不匹配报错" (vt-check-out (vt-ones '(2 3)) '(3 2) :float64))
  (check-error "H3 dtype 不匹配报错" (vt-check-out (vt-ones '(2 3)) '(2 3) :float32))
  (check-error "H1 out 非 vt 报错" (vt-check-out 3 '(1) :float64))
  (check "vt-out-writable-p 普通视图可写"
         (eq (vt-out-writable-p (vt-slice (vt-ones '(2 3)) '(:all) '(1 nil))) t))
  (check "vt-out-writable-p 广播视图（stride-0）不可写"
         (not (vt-out-writable-p (vt-broadcast-to (vt-ones '(1 3)) '(2 3)))))
  ;; 端到端：非连续 out 必须生效（CONVENTIONS §附录B out 契约样例）
  (let* ((base (vt-zeros '(2 6) :dtype :int64))
         (v (vt-slice base '(:all) '(0 nil 2)))
         (a (vt-reshape (vt-arange 6 :dtype :int64) '(2 3))))
    (vt-add a a :out v)
    (check-eq "非连续 out（步长 2 切片）写入生效"
              (vt-to-list (vt-slice base '(:all) '(0 nil 2)))
              '((0 2 4) (6 8 10)))
    (check-eq "非连续 out 写入不污染相邻元素"
              (vt-to-list (vt-slice base '(:all) '(1 2)))
              '((0) (0))))
  (check-error "stride-0 广播视图作为 out → 报错（§2.1 只读约定）"
               (let ((a (vt-ones '(3))))
                 (vt-add a a :out (vt-broadcast-to (vt-ones '(1 3)) '(2 3)))))
  (check-error "H5 :dtype 与 :out dtype 冲突报错"
               (let ((a (vt-ones '(3))))
                 (vt-add a a :dtype :float32 :out (vt-ones '(3))))))
  (let ((z (vt-from-sequence '(1.0d0 2.0d0 3.0d0 4.0d0)))
        (indep (vt-ones '(4))))
    (let ((rev (vt-slice z '(3 nil -1))))
      (check "vt-out-snapshot 重叠输入被快照（返回新列表）"
             (not (eq (first (vt-out-snapshot (vt-zeros '(4)) (list rev))) z)))
      (check "vt-out-snapshot 非重叠输入原样返回"
             (eq (first (vt-out-snapshot (vt-zeros '(4)) (list indep))) indep))))

(deftest test-nan-infrastructure
    "nan: vt-float-nan-p / pos-inf-p / neg-inf-p / nan-= / inf-= / nan-inf-= / vt-get-nan / 常量函数"
  (check "vt-float-nan-p(NaN)=T" (eq (vt-float-nan-p (vt-float-nan :float64)) t))
  (check "vt-float-nan-p(1.0)=NIL" (not (vt-float-nan-p 1.0d0)))
  (check "vt-float-nan-p(非数)=NIL" (not (vt-float-nan-p "x")))
  (check "vt-float-pos-inf-p(+Inf)=T" (eq (vt-float-pos-inf-p (vt-float-pos-inf :float64)) t))
  (check "vt-float-neg-inf-p(-Inf)=T" (eq (vt-float-neg-inf-p (vt-float-neg-inf :float64)) t))
  (check "vt-float-pos-inf-p(-Inf)=NIL" (not (vt-float-pos-inf-p (vt-float-neg-inf :float64))))
  (check "vt-float-nan-= 两个 NaN 判等" (eq (vt-float-nan-= (vt-float-nan :float64) (vt-float-nan :float64)) t))
  (check "vt-float-nan-= NaN vs 1 → NIL" (not (vt-float-nan-= (vt-float-nan :float64) 1.0d0)))
  (check "vt-float-inf-= 同号 Inf 判等"
         (eq (vt-float-inf-= (vt-float-pos-inf :float64) (vt-float-pos-inf :float64)) t))
  (check "vt-float-inf-= 异号 Inf → NIL"
         (not (vt-float-inf-= (vt-float-pos-inf :float64) (vt-float-neg-inf :float64))))
  (check "vt-float-nan-inf-= NaN==NaN → T"
         (eq (vt-float-nan-inf-= (vt-float-nan :float64) (vt-float-nan :float64)) t))
  (check "vt-float-nan-inf-= 数值正常比较"
         (and (eq (vt-float-nan-inf-= 1.0d0 1.0d0) t)
              (not (vt-float-nan-inf-= 1.0d0 2.0d0))))
  (check "vt-get-nan :float32 → single-float" (typep (vt-get-nan :float32) 'single-float))
  (check "vt-get-nan :float64 → double-float" (typep (vt-get-nan :float64) 'double-float))
  (check "vt-float-nan-p(vt-get-nan)=T" (vt-float-nan-p (vt-get-nan :float32))))

;;; ==================================================================
;;; 13. 映射/迭代原语与杂项（map-reduce / iterator / io / package）
;;; ==================================================================

(deftest test-primitives-map-reduce
    "primitives: vt-map / vt-reduce / vt-do-each / vt-reduce-dtypes / vt-params-audit"
  (check-eq "vt-map 单张量映射"
            (vt-to-list (vt-map (lambda (x) (* 2.0d0 x)) (vt-from-sequence '(1.0d0 2.0d0))))
            '(2.0d0 4.0d0))
  (check-eq "vt-map 双张量+标量混合广播"
            (vt-to-list (vt-map (lambda (x y) (+ x y))
                                (vt-from-sequence '(1.0d0 2.0d0)) 10.0d0))
            '(11.0d0 12.0d0))
  (check "vt-reduce 全轴求和=6"
         (= (vt-item (vt-reduce (vt-from-sequence '(1.0d0 2.0d0 3.0d0)) 0 0.0d0 #'+)) 6.0d0))
  (check "vt-reduce axis 求和（2d）"
         (allclose-lists
          (vt-to-list (vt-reduce (vt-reshape (vt-arange 6 :dtype :int64) '(2 3)) 0 0.0d0 #'+))
          '(3.0d0 5.0d0 7.0d0)))
  (check "vt-reduce 求最大值（自定义 reducer）"
         (allclose-lists (list (vt-item (vt-reduce (vt-from-sequence '(3.0d0 1.0d0 4.0d0)) 0 0.0d0 #'max)))
                         '(4.0d0)))
  (check "vt-do-each 遍历求和"
         (let ((s 0.0d0))
           (vt-do-each (ptr v (vt-from-sequence '(1.0d0 2.0d0 3.0d0))) (declare (ignore ptr)) (incf s v))
           (= s 6.0d0)))
  (check "vt-do-each 支持非连续视图"
         (let ((s 0.0d0))
           (vt-do-each (ptr v (vt-slice (vt-arange 6 :dtype :float64) '(0 nil 2)))
             (declare (ignore ptr))
             (incf s v))
           (= s 6.0d0)))  ; 0+2+4
  ;; vt-reduce-dtypes 返回 (values compute-dtype exec-dtype)
  (multiple-value-bind (compute exec) (vt-reduce-dtypes (list (vt-ones '(2) :dtype :float32)) nil)
    (check "vt-reduce-dtypes float32 求和不升级（§4.4）" (eq compute :float32))
    (check "vt-reduce-dtypes exec-dtype 与 compute 一致" (eq exec :float32)))
  (multiple-value-bind (compute) (vt-reduce-dtypes (list (vt-ones '(2) :dtype :int8) (vt-ones '(2) :dtype :int8)) nil)
    (check "vt-reduce-dtypes int8 求和 → int64（§4.4）" (eq compute :int64)))
  (multiple-value-bind (compute) (vt-reduce-dtypes (list (vt-ones '(2) :dtype :int8) (vt-ones '(2) :dtype :float64)) nil)
    (check "vt-reduce-dtypes 混合 → float64" (eq compute :float64)))
  (check "vt-params-audit 返回 alist 且包含 vt-sum"
         (and (listp (vt-params-audit))
              (find 'vt-sum (vt-params-audit) :key #'car))))

(deftest test-print-misc
    "io: vt-set-print-options / vt-get-print-options 往返 / print-vt-recursive / *vt-fun-list*"
  (let ((saved (vt-get-print-options)))
    (vt-set-print-options :precision 3 :threshold 50)
    (let ((opts (vt-get-print-options)))
      (check "vt-set-print-options 往返生效"
             (and (= (second opts) 3) (= (first opts) 50))))
    (apply #'vt-set-print-options
           (append (list :threshold (first saved) :precision (second saved))
                   (when (third saved) (list :indent-step (third saved))))))
  (check "print-vt-recursive 输出非空字符串"
         (> (length (with-output-to-string (s) (print-vt-recursive (vt-ones '(2 2)) 0 nil 2 4 'double-float s))) 0))
  (check "print-vt-recursive 截断阈值生效"
         (> (length (with-output-to-string (s) (print-vt-recursive (vt-arange 100 :dtype :float64) 0 nil 2 8 'double-float s))) 0))
  (progn (refresh-vt-fun-list)
         (check "refresh-vt-fun-list 收集 >300 个导出符号"
                (> (length *vt-fun-list*) 300)))
  (check "导出特殊变量已绑定"
         (and (boundp '*vt-print-precision*) (boundp '*vt-print-threshold*)
              (boundp '*vt-indent-step*) (boundp '*vt-einsum-parse-cache*))))

(deftest test-rotate-smoke
    "rotate: vt-rotate（内部函数，度数制逆时针）/ vt-rotate-origin 冒烟"
  (let ((m (vt-from-sequence '((1.0d0 2.0d0 3.0d0) (4.0d0 5.0d0 6.0d0)))))
    (let ((r0 (vt-rotate m 0.0d0 :reshape t)))
      (check-shape "vt-rotate 0° reshape 形状不变" r0 '(2 3))
      (check-eq "vt-rotate 0° 数据不变" (vt-to-list r0) '((1.0d0 2.0d0 3.0d0) (4.0d0 5.0d0 6.0d0))))
    (let ((r90 (vt-rotate m 90.0d0 :reshape t)))
      (check-shape "vt-rotate 90° reshape → (3 2)" r90 '(3 2))
      (check-eq "vt-rotate 90° 逆时针（scipy 对齐）" (vt-to-list r90)
                '((3.0d0 6.0d0) (2.0d0 5.0d0) (1.0d0 4.0d0))))
    (check "vt-rotate-origin 0° 恒等"
           (allclose-lists (vt-to-list (vt-rotate-origin m 0.0d0))
                           '((1.0d0 2.0d0 3.0d0) (4.0d0 5.0d0 6.0d0))))))

;;; ==================================================================
;;; 运行入口
;;; ==================================================================

(defun run-all-tests ()
  "运行全部注册测试，返回 T 当且仅当全部通过。"
  (setf *assert-pass* 0 *assert-fail* 0)
  (multiple-value-bind (p f) (run-tests)
    (declare (ignore p))
    (zerop f)))

(unless (run-all-tests)
  #+sbcl (sb-ext:exit :code 1)
  #-sbcl (error "test-all failed"))
