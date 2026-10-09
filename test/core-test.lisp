;;;; test.lisp —— clvt 前 200 个导出符号（1 个结构类型 vt + 199 个函数）逐一测试
;;;;
;;;; 期望值依据 CONVENTIONS.md（对标 NumPy 2.3.5；vt-arange 为唯一例外）编写。
;;;; 覆盖约定：§3 类型提升 / §4 :out 硬契约 / §5 NaN·Inf·零除语义 /
;;;;           §6 返回值形状（张量函数返回 vt，vt-item/vt-ref 返回 Lisp 标量）/
;;;;           D5 clip / D6 where / D12 归约 dtype / D14 amax NaN / D16 div 零除。
;;;; 运行： sbcl --script test.lisp
;;;; 输出： 逐项 PASS/FAIL，末尾汇总与失败清单，退出码 0/1。

(require :asdf)

(defparameter *clvt-root*
  (make-pathname :directory (butlast (pathname-directory *load-truename*))))
(pushnew *clvt-root* asdf:*central-registry*)

(handler-bind ((warning #'muffle-warning))
  (asdf:load-system :clvt))

(defpackage :clvt-test
  (:use :cl :clvt))
(in-package :clvt-test)

;;; ================================================================
;;; 测试框架
;;; ================================================================

(defvar *pass* 0)
(defvar *fail* 0)
(defvar *failed-names* nil)
(defvar *section* "")

(defmacro section (name)
  `(setf *section* ,name))

(defparameter +inf+ sb-ext:double-float-positive-infinity)
(defparameter +ninf+ sb-ext:double-float-negative-infinity)
(defparameter +nan+
  (sb-int:with-float-traps-masked (:invalid :divide-by-zero) (/ 0.0d0 0.0d0)))

;; 数值比较：NaN==NaN 视为相等（测试中显式期望 NaN/Inf），浮点用相对容差
(defun %num= (a b)
  (let ((da (coerce a 'double-float))
        (db (coerce b 'double-float)))
    (cond ((and (sb-ext:float-nan-p da) (sb-ext:float-nan-p db)) t)
          ((and (sb-ext:float-infinity-p da) (sb-ext:float-infinity-p db))
           (= da db))
          ((or (sb-ext:float-nan-p da) (sb-ext:float-nan-p db)) nil)
          ((or (sb-ext:float-infinity-p da) (sb-ext:float-infinity-p db)) nil)
          (t (<= (abs (- da db)) (* 1.0d-9 (max 1.0d0 (abs da) (abs db))))))))

;; 通用比较：vt 与期望嵌套列表、vt 与 vt、列表、标量
(defun %val= (a b)
  (cond ((vt-p a) (%val= (vt-to-list a) b))
        ((and (numberp a) (numberp b)) (%num= a b))
        ((and (consp a) (consp b))
         (and (%val= (car a) (car b)) (%val= (cdr a) (cdr b))))
        ((and (null a) (null b)) t)
        ((and (symbolp a) (symbolp b)) (eq a b))
        ((and (stringp a) (stringp b)) (string= a b))
        ((and (vectorp a) (vectorp b))
         (and (= (length a) (length b))
              (loop for x across a
                    for y across b
                    always (%val= x y))))
        (t (equal a b))))

(defun safe-call (thunk)
  (handler-case (funcall thunk)
    (error (e) (list :error (format nil "~a" e)))))

(defun %report (name ok got)
  (if ok
      (progn
        (incf *pass*)
        (format t "✓ PASS ~a :: ~a~%" *section* name))
      (progn
        (incf *fail*)
        (push name *failed-names*)
        (format t "✗ FAIL ~a :: ~a~%        got: ~a~%" *section* name got))))

;; check：结果为 vt → 比较 to-list（可加 :dtype :shape）；结果为标量 → 数值比较
(defun %check-result (name v expected &key dtype shape)
  (if (and (listp v) (eq (first v) :error))
      (%report name nil (format nil "意外报错: ~a" (second v)))
      ;; 0 维 vt 用 item 取 Lisp 标量比较（§6.5：归约返回 0 维 vt）
      (let ((ok (cond
                  ((vt-p v)
                   (and (if shape (equal (vt-shape v) shape) t)
                        (if dtype (eq (vt-dtype v) dtype) t)
                        (%val= (if (null (vt-shape v)) (vt-item v) v)
                               expected)))
                  (t (%val= v expected)))))
        (%report name ok
                 (if (vt-p v)
                     (format nil "to-list=~s shape=~s dtype=~s"
                             (vt-to-list v) (vt-shape v) (vt-dtype v))
                     (format nil "~s" v))))))

(defmacro check (name form expected &rest keys &key dtype shape)
  (declare (ignore dtype shape))
  `(%check-result ,name (safe-call (lambda () ,form)) ,expected ,@keys))

(defmacro check-true (name form)
  (let ((v (gensym)))
    `(let ((,v (safe-call (lambda () ,form))))
       (%report ,name (and (not (and (listp ,v) (eq (first ,v) :error))) ,v)
                (format nil "~s" ,v)))))

(defmacro check-false (name form)
  (let ((v (gensym)))
    `(let ((,v (safe-call (lambda () ,form))))
       (%report ,name (and (not (and (listp ,v) (eq (first ,v) :error)))
                           (not ,v))
                (format nil "~s" ,v)))))

;; check-err：表达式必须确定性报错
(defmacro check-err (name form)
  (let ((v (gensym)))
    `(let ((,v (safe-call (lambda () ,form))))
       (if (and (listp ,v) (eq (first ,v) :error))
           (%report ,name t "")
           (%report ,name nil (format nil "未报错，返回 ~s" ,v))))))

;; 多值返回（如 vt-histogram）逐个检查
(defmacro check-mv (name form &body clauses)
  (let ((res (gensym)))
    `(let ((,res (multiple-value-list (safe-call (lambda () ,form)))))
       (if (and (first ,res) (listp (first ,res)) (eq (first (first ,res)) :error))
           (%report ,name nil (format nil "意外报错: ~a" (second (first ,res))))
           (progn
             ,@(loop for clause in clauses
                     for i from 0
                     collect `(let ((got (nth ,i ,res)))
                                (%check-result (format nil "~a [第~d返回值]" ,name ,(1+ i))
                                               (safe-call (lambda () got))
                                               ,(second clause)
                                               ,@(cddr clause)))))))))

;;; ================================================================
;;; 第 1 部分：vt 结构与物理层原语（导出符号 1–19）
;;; ================================================================

(section "① 结构访问器与物理层原语 (1-19)")

;; 1. vt 结构类型：视图 = data/shape/strides/offset/dtype
(check-true "vt(1): 结构类型存在且 (vt-zeros) 是其实例"
  (typep (vt-zeros '(2 2)) 'vt))
(check "vt(1): 默认成员 (dtype :float64, offset 0, strides C 序)"
  (list (vt-dtype (vt-zeros '(2 2)))
        (vt-offset (vt-zeros '(2 2)))
        (vt-strides (vt-zeros '(2 2))))
  (list :float64 0 (list 2 1)))

;; 2. vt-shape
(check "vt-shape(2): 形状读取" (vt-shape (vt-zeros '(2 3))) '(2 3))
;; 3. vt-strides（元素步长，C 连续 (2 3) → (3 1)）
(check "vt-strides(3): C 连续步长" (vt-strides (vt-zeros '(2 3))) '(3 1))
;; 4. vt-offset（切片视图的起点）
(check "vt-offset(4): 新建为 0" (vt-offset (vt-ones '(2))) 0)
(check "vt-offset(4): 切片视图 offset=2"
  (vt-offset (vt-slice (vt-reshape (vt-arange 4) '(2 2)) '(1) '(:all))) 2)
;; 5. vt-data（底层一维缓冲区）
(check-true "vt-data(5): 底层缓冲区长度=元素总数"
  (= (length (vt-data (vt-zeros '(2 3)))) 6))
;; 6. vt-element-type
(check "vt-element-type(6): float64→double-float"
  (vt-element-type (vt-zeros '(2))) 'double-float)
(check "vt-element-type(6): int32→(signed-byte 32)"
  (vt-element-type (vt-zeros '(2) :dtype :int32)) '(signed-byte 32))
;; 7. vt-dtype
(check "vt-dtype(7): 默认 :float64" (vt-dtype (vt-ones '(1))) :float64)
;; 8. vt-order（秩）
(check "vt-order(8): (2 3) 秩=2" (vt-order (vt-zeros '(2 3))) 2)
(check "vt-order(8): 标量 0 维 秩=0" (vt-order (vt-const nil 1)) 0)
;; 9. vt-p
(check-true "vt-p(9): vt 实例为真" (vt-p (vt-zeros '(1))))
(check-false "vt-p(9): 非 vt 为假" (vt-p '(1 2 3)))
;; 10. vt-itemsize（字节）
(check "vt-itemsize(10): float64=8" (vt-itemsize (vt-zeros '(1))) 8)
(check "vt-itemsize(10): int8=1" (vt-itemsize (vt-zeros '(1) :dtype :int8)) 1)
;; 11. vt-nbytes
(check "vt-nbytes(11): (2 3) float64 = 48" (vt-nbytes (vt-zeros '(2 3))) 48)
;; 12. vt-contiguous-p
(check-true "vt-contiguous-p(12): C 连续为真"
  (vt-contiguous-p (vt-zeros '(2 3))))
(check-false "vt-contiguous-p(12): 转置视图非连续"
  (vt-contiguous-p (vt-transpose (vt-zeros '(2 3)))))
;; 13. vt-shape-to-size
(check "vt-shape-to-size(14): (2 3)→6" (vt-shape-to-size '(2 3)) 6)
(check "vt-shape-to-size(14): nil→1（标量）" (vt-shape-to-size nil) 1)
;; 14. vt-compute-strides
(check "vt-compute-strides(15): (2 3)→(3 1)" (vt-compute-strides '(2 3)) '(3 1))
(check "vt-compute-strides(15): nil→nil" (vt-compute-strides nil) nil)
;; 15. vt-compute-logical-strides
(check "vt-compute-logical-strides(16): (2 3)→(3 1)"
  (vt-compute-logical-strides '(2 3)) '(3 1))
;; 16. vt-broadcast-shapes（§2.1 尾轴对齐）
(check "vt-broadcast-shapes(17): (2 3)+(3)→(2 3)"
  (vt-broadcast-shapes '(2 3) '(3)) '(2 3))
(check "vt-broadcast-shapes(17): (2 1)+(1 3)→(2 3)"
  (vt-broadcast-shapes '(2 1) '(1 3)) '(2 3))
(check-err "vt-broadcast-shapes(17): (2 3)+(4) 报错"
  (vt-broadcast-shapes '(2 3) '(4)))
;; 17. vt-broadcast-strides（广播维度步长=0）
(check "vt-broadcast-strides(18): (3)→(2 3) 步长 (0 1)"
  (vt-broadcast-strides '(3) '(2 3) '(1)) '(0 1))
;; 18. vt-normalize-axis（负轴规范化）
(check "vt-normalize-axis(19): -1 with rank2 → 1" (vt-normalize-axis -1 2) 1)
(check "vt-normalize-axis(19): 0 → 0" (vt-normalize-axis 0 2) 0)
(check-err "vt-normalize-axis(19): 轴越界报错" (vt-normalize-axis 2 2))
;; 19. 标量 0 维 vt 支持（§6.5）
(check "0 维张量（通用属性）: (vt-const nil 5) 秩 0" (vt-order (vt-const nil 5.0d0)) 0)

;;; ================================================================
;;; 第 2 部分：创建函数（导出符号 20–41）
;;; ================================================================

(section "② 创建函数 (20-41)")

(check "vt-zeros(20): (2 3) 全 0 float64" (vt-zeros '(2 3)) '((0 0 0) (0 0 0)))
(check "vt-zeros(20): 显式 dtype" (vt-zeros '(2) :dtype :int32) '(0 0) :dtype :int32)
(check "vt-ones(21): (2 3) 全 1" (vt-ones '(2 3)) '((1 1 1) (1 1 1)))
(check "vt-full(22): 全 7.5" (vt-full '(2 2) 7.5d0) '((7.5 7.5) (7.5 7.5)))
;; vt-empty：仅保证形状/大小，值未定义
(check "vt-empty(23): 形状与大小正确（值未定义）"
  (list (vt-shape (vt-empty '(2 3))) (vt-size (vt-empty '(2 3))))
  (list '(2 3) 6))
(check "vt-zeros-like(24): 继承形状与 dtype"
  (vt-zeros-like (vt-ones '(2 2) :dtype :int16)) '((0 0) (0 0)) :dtype :int16)
(check "vt-ones-like(25): dtype 覆盖"
  (vt-ones-like (vt-ones '(2)) :dtype :float32) '(1 1) :dtype :float32)
(check "vt-full-like(26): 继承形状填充指定值"
  (vt-full-like (vt-ones '(3)) 9.0d0) '(9 9 9))
(check "vt-empty-like(27): 形状正确"
  (vt-shape (vt-empty-like (vt-ones '(3 4)))) '(3 4))
(check "vt-const(28): 常量填充" (vt-const '(2 2) 3.0d0) '((3 3) (3 3)))
(check "vt-const(28): 0 维标量" (vt-const nil 5) 5 :shape nil)
;; vt-arange：§9.1 唯一例外——首个参数是元素个数 TOTAL-NUM 而非 stop
(check "vt-arange(29): (vt-arange 5) → 5 个元素 [0,5)"
  (vt-arange 5) '(0 1 2 3 4) :dtype :float64)
(check "vt-arange(29): start/step 语义 (10 12 14)"
  (vt-arange 3 :start 10 :step 2) '(10 12 14))
(check "vt-arange(29): 浮点步长"
  (vt-arange 3 :start 0.0d0 :step 0.5d0) '(0.0 0.5 1.0))
(check "vt-linspace(30): 含端点 5 点均分"
  (vt-linspace 0.0d0 1.0d0 5) '(0 0.25 0.5 0.75 1.0))
(check "vt-linspace(30): endpoint=nil 不含端点"
  (vt-linspace 0.0d0 1.0d0 5 :endpoint nil) '(0.0d0 0.2d0 0.4d0 0.6d0 0.8d0))
(check "vt-logspace(31): 10^0 10^1 10^2"
  (vt-logspace 0.0d0 2.0d0 3) '(1 10 100))
(check "vt-eye(32): 2x2 单位阵" (vt-eye 2) '((1 0) (0 1)))
(check "vt-eye(32): k=1 对角线偏移" (vt-eye 2 :k 1) '((0 1) (0 0)))
;; vt-diag：1 维 → 构造对角阵；2 维 → 提取对角线
(check "vt-diag(33): 1维→对角阵"
  (vt-diag (vt-from-sequence '(1 2 3))) '((1 0 0) (0 2 0) (0 0 3)))
(check "vt-diag(33): 2维→提取对角线"
  (vt-diag (vt-from-sequence '((1 2) (3 4)))) '(1 4))
(check "vt-identity(34): 3x3 单位阵"
  (vt-identity 3) '((1 0 0) (0 1 0) (0 0 1)))
(check "vt-from-sequence(35): 嵌套列表→张量"
  (vt-from-sequence '((1 2) (3 4))) '((1 2) (3 4)))
(check "vt-from-array(36): Lisp 数组→张量"
  (vt-to-list (vt-from-array (make-array '(2 2) :initial-contents '((1 2) (3 4)))))
  '((1 2) (3 4)))
(check "vt-from-function(37): fn 接收索引列表"
  (vt-from-function '(2 2) (lambda (idx) (+ (first idx) (* 10 (second idx)))))
  '((0 10) (1 11)))
(check "vt-flatten-sequence(38): 任意嵌套展平"
  (vt-flatten-sequence '(1 (2 (3 4)) 5)) '(1 2 3 4 5))
(check "vt-to-list(39): 张量→嵌套列表"
  (vt-to-list (vt-ones '(2 2))) '((1.0 1.0) (1.0 1.0)))
(check-true "vt-to-array(40): Lisp 多维数组维度正确"
  (let ((a (vt-to-array (vt-ones '(2 3)))))
    (and (typep a '(array * (2 3))) (= (array-total-size a) 6))))
(check "vt-astype(41): float→int 截断"
  (vt-astype (vt-from-sequence '(1.7 2.3)) :int32) '(1 2) :dtype :int32)
(check "vt-astype(41): NaN→整数 cast 得 0（§7.2，不触发陷阱）"
  (vt-astype (vt-const '(1) +nan+) :int32) '(0) :dtype :int32)

;;; ================================================================
;;; 第 3 部分：形状操作与视图（导出符号 42–76）
;;; ================================================================

(section "③ 形状操作 / 视图 / 拼接 (42-76)")

(check "vt-view(42): 零拷贝重塑（连续输入）"
  (vt-view (vt-arange 6) '(2 3)) '((0 1 2) (3 4 5)))
(check-true "vt-view(42): 视图与原张量共享 data"
  (let ((a (vt-arange 6)))
    (eq (vt-data (vt-view a '(2 3))) (vt-data a))))
(check "vt-reshape(43): (2 3)→(3 2)"
  (vt-reshape (vt-arange 6) '(3 2)) '((0 1) (2 3) (4 5)))
(check "vt-reshape(43): 支持 -1 推导一个维度"
  (vt-reshape (vt-arange 6) '(-1 3)) '((0 1 2) (3 4 5)))
(check-err "vt-reshape(43): 两个 -1 报错" (vt-reshape (vt-arange 6) '(-1 -1)))
(check "vt-transpose(44): 缺省逆转全部轴"
  (vt-to-list (vt-transpose (vt-reshape (vt-arange 6 :dtype :int64) '(2 3))))
  '((0 3) (1 4) (2 5)))
(check "vt-transpose(44): 指定 perm=(0 1) 等价原样"
  (vt-to-list (vt-transpose (vt-reshape (vt-arange 6 :dtype :int64) '(2 3)) '(0 1)))
  '((0 1 2) (3 4 5)))
(check "vt-squeeze(45): 移除全部长度 1 轴"
  (vt-shape (vt-squeeze (vt-zeros '(1 3 1)))) '(3))
(check "vt-squeeze(45): 指定 axis"
  (vt-shape (vt-squeeze (vt-zeros '(1 3 1)) :axis 0)) '(3 1))
(check "vt-unsqueeze(46): axis=0 插入新轴"
  (vt-to-list (vt-unsqueeze (vt-arange 3 :dtype :int64) 0)) '((0 1 2)))
(check "vt-expand-dims(47): axis=-1 末尾插轴"
  (vt-to-list (vt-expand-dims (vt-arange 3 :dtype :int64) -1)) '((0) (1) (2)))
(check "vt-flatten(48): (2 3)→(6)"
  (vt-to-list (vt-flatten (vt-reshape (vt-arange 6 :dtype :int64) '(2 3))))
  '(0 1 2 3 4 5))
(check "vt-ravel(49): 同 flatten"
  (vt-shape (vt-ravel (vt-ones '(2 3)))) '(6))
(check "vt-swapaxes(50): 交换轴 0/1"
  (vt-to-list (vt-swapaxes (vt-reshape (vt-arange 6 :dtype :int64) '(2 3)) 0 1))
  '((0 3) (1 4) (2 5)))
;; vt-rot90：逆时针旋转 90°，对标 np.rot90
(check "vt-rot90(51): k=1 逆时针旋转"
  (vt-to-list (vt-rot90 (vt-from-sequence '((1 2) (3 4))) :k 1)) '((2 4) (1 3)))
(check "vt-rot90(51): k=2 转 180°"
  (vt-to-list (vt-rot90 (vt-from-sequence '((1 2) (3 4))) :k 2)) '((4 3) (2 1)))
(check "vt-narrow(52): 沿轴取 [start,end) 视图"
  (vt-to-list (vt-narrow (vt-from-sequence '((1 2 3) (4 5 6))) 0 0 1)) '((1 2 3)))
(check "vt-split(53): 按段数均分返回列表"
  (mapcar #'vt-shape (vt-split (vt-zeros '(4 2)) 2)) '((2 2) (2 2)))
(check "vt-vsplit(54): 按行拆分"
  (mapcar #'vt-to-list (vt-vsplit (vt-from-sequence '((1 2) (3 4) (5 6) (7 8))) 2))
  '(((1 2) (3 4)) ((5 6) (7 8))))
(check "vt-hsplit(55): 按列拆分"
  (mapcar #'vt-to-list (vt-hsplit (vt-from-sequence '((1 2 3 4))) 2))
  '(((1 2)) ((3 4))))
(check "vt-dsplit(56): 沿第 3 轴拆分"
  (mapcar #'vt-shape (vt-dsplit (vt-zeros '(2 2 4)) 2)) '((2 2 2) (2 2 2)))
;; vt-stack/vt-concatenate：axis 是第一个位置参数
(check "vt-stack(57): axis=0 堆叠 1 维→(2 2)"
  (vt-to-list (vt-stack 0 (vt-from-sequence '(1 2)) (vt-from-sequence '(3 4))))
  '((1 2) (3 4)))
(check "vt-stack(57): axis=1 堆叠 1 维→(2 2)"
  (vt-to-list (vt-stack 1 (vt-from-sequence '(1 2)) (vt-from-sequence '(3 4))))
  '((1 3) (2 4)))
(check "vt-vstack(58): 行堆叠"
  (vt-to-list (vt-vstack (vt-from-sequence '(1 1)) (vt-from-sequence '(2 2))))
  '((1 1) (2 2)))
(check "vt-hstack(59): 列拼接"
  (vt-to-list (vt-hstack (vt-from-sequence '(1 1)) (vt-from-sequence '(2 2))))
  '(1 1 2 2))
(check "vt-dstack(60): 深度堆叠 (2)+(2)→(1 2 2)"
  (vt-to-list (vt-dstack (vt-from-sequence '(1 1)) (vt-from-sequence '(2 2))))
  '(((1 2) (1 2))))
(check "vt-concatenate(61): axis=0 沿轴连接"
  (vt-to-list (vt-concatenate 0 (vt-ones '(1 2)) (vt-zeros '(1 2))))
  '((1 1) (0 0)))
(check "vt-concat(62): concatenate 别名"
  (vt-to-list (vt-concat 0 (vt-ones '(2)) (vt-zeros '(2)))) '(1 1 0 0))
(check "vt-repeat(63): 展平重复"
  (vt-to-list (vt-repeat (vt-from-sequence '(1 2)) 2)) '(1 1 2 2))
(check "vt-repeat(63): axis=0 整体重复"
  (vt-to-list (vt-repeat (vt-from-sequence '((1 2))) 2 :axis 0)) '((1 2) (1 2)))
(check "vt-tile(64): 标量 reps 平铺"
  (vt-to-list (vt-tile (vt-from-sequence '(1 2)) 2)) '(1 2 1 2))
(check "vt-tile(64): 列表 reps (2 1) → 堆两行"
  (vt-to-list (vt-tile (vt-from-sequence '(1 2)) '(2 1))) '((1 2) (1 2)))
(check "vt-pad(65): 常量填充两侧各 1"
  (vt-to-list (vt-pad (vt-from-sequence '(1 2 3)) 1)) '(0 1 2 3 0))
(check "vt-pad(65): 指定填充值 9"
  (vt-to-list (vt-pad (vt-from-sequence '(1 2 3)) 1 :constant-values 9)) '(9 1 2 3 9))
(check "vt-broadcast-to(66): (3)→(2 3) stride-0 广播视图"
  (vt-to-list (vt-broadcast-to (vt-from-sequence '(1 2 3)) '(2 3)))
  '((1 2 3) (1 2 3)))
(check-true "vt-contiguous(67): 已连续时零拷贝返回同一对象"
  (let ((a (vt-zeros '(2 3))))
    (eq (vt-contiguous a) a)))
(check "vt-flip(68): axis=0 翻转行"
  (vt-to-list (vt-flip (vt-from-sequence '((1 2) (3 4))) :axis 0)) '((3 4) (1 2)))
(check "vt-flip(68): axis=nil 全轴翻转"
  (vt-to-list (vt-flip (vt-from-sequence '((1 2) (3 4))) :axis nil)) '((4 3) (2 1)))
(check "vt-roll(69): 展平滚动 +2"
  (vt-to-list (vt-roll (vt-arange 5 :dtype :int64) 2)) '(3 4 0 1 2))
(check "vt-roll(69): 负向滚动 -1"
  (vt-to-list (vt-roll (vt-arange 5 :dtype :int64) -1)) '(1 2 3 4 0))
(check "vt-triu(70): 上三角"
  (vt-to-list (vt-triu (vt-from-sequence '((1 2) (3 4))))) '((1 2) (0 4)))
(check "vt-tril(71): 下三角"
  (vt-to-list (vt-tril (vt-from-sequence '((1 2) (3 4))))) '((1 0) (3 4)))
(check "vt-diagonal(72): 提取对角线"
  (vt-to-list (vt-diagonal (vt-from-sequence '((1 2) (3 4))))) '(1 4))
(check "vt-diagonal(72): offset=1 次对角线"
  (vt-to-list (vt-diagonal (vt-from-sequence '((1 2) (3 4))) :offset 1)) '(2))
(check "vt-flatten-to-nested(73): 一维数据→嵌套列表"
  (vt-flatten-to-nested '(2 2) (make-array 4 :initial-contents '(1 2 3 4)))
  '((1 2) (3 4)))
(check "vt-append(74): axis=nil 展平追加"
  (vt-to-list (vt-append (vt-from-sequence '(1 1)) (vt-from-sequence '(2 2))))
  '(1 1 2 2))
(check "vt-insert(75): 按位置插入"
  (vt-to-list (vt-insert (vt-from-sequence '(1 1 1)) 1 99)) '(1 99 1 1))
(check "vt-delete(76): 删除索引 1"
  (vt-to-list (vt-delete (vt-from-sequence '(1 2 3)) 1)) '(1 3))

;;; ================================================================
;;; 第 4 部分：索引与查询（导出符号 77–90）
;;; ================================================================

(section "④ 索引与查询 (77-90)")

;; vt-ref：全整数索引返回 Lisp 标量（§6.5）
(check "vt-ref(77): 全整数索引 → Lisp 标量"
  (vt-ref (vt-from-sequence '((1 2) (3 4))) 0 1) 2)
(check "vt-ref(77): 负索引"
  (vt-ref (vt-from-sequence '(1 2 3)) -1) 3)
;; vt-slice：spec 形式 (:all)/(idx)/(start end[ step])
(check "vt-slice(78): (idx)(:all) 取行"
  (vt-to-list (vt-slice (vt-from-sequence '((1 2 3) (4 5 6))) '(1) '(:all))) '(4 5 6))
(check "vt-slice(78): 范围切片 [0,2)"
  (vt-to-list (vt-slice (vt-from-sequence '((0 1 2) (3 4 5))) '(0) '(0 2))) '(0 1))
(check-true "vt-slice(78): 返回视图共享 data"
  (let ((a (vt-reshape (vt-arange 6) '(2 3))))
    (eq (vt-data (vt-slice a '(0) '(:all))) (vt-data a))))
;; vt-item：0 维 → Lisp 标量
(check "vt-item(79): 0 维取值" (vt-item (vt-const nil 5)) 5)
(check "vt-take(80): axis=nil 展平索引"
  (vt-to-list (vt-take (vt-from-sequence '(10 20 30)) (vt-from-sequence '(0 2))))
  '(10 30))
(check "vt-take(80): 重复索引"
  (vt-to-list (vt-take (vt-from-sequence '(10 20 30)) (vt-from-sequence '(1 1))))
  '(20 20))
(check-true "vt-put(81): 就地写入返回张量本身"
  (let ((a (vt-zeros '(3) :dtype :int64)))
    (and (eq (vt-put a 1 99) a) (= (aref (vt-data a) 1) 99)))
  )
(check "vt-put(81): 列表索引批量写入"
  (vt-to-list (vt-put (vt-zeros '(3) :dtype :int64) (vt-from-sequence '(0 2)) 7 :mode :raise))
  '(7 0 7))
(check "vt-where(82): 三参 where(cond, x, y)"
  (vt-to-list (vt-where (vt-from-sequence '(1 0) :dtype :int8)
                        (vt-const '(2) 1.0d0) (vt-const '(2) 2.0d0)))
  '(1 2))
;; D6：单参 where 返回 (n, rank) 坐标
(check "vt-where(82): 单参 → (n,rank) 坐标 (D6)"
  (vt-to-list (vt-where (vt-from-sequence '((1 0) (0 1)) :dtype :int8)))
  '((0 0) (1 1)) :shape '(2 2))
(check "vt-argwhere(83): 非零坐标 (n,rank)"
  (vt-to-list (vt-argwhere (vt-from-sequence '((1 0) (0 1)) :dtype :int8)))
  '((0 0) (1 1)) :shape '(2 2))
(check-true "vt-nonzero(84): 返回每轴索引张量列表"
  (let ((r (vt-nonzero (vt-from-sequence '((1 0) (0 1)) :dtype :int8))))
    (and (listp r) (= (length r) 2)
         (equal (vt-to-list (first r)) '(0 1))
         (equal (vt-to-list (second r)) '(0 1)))))
(check "vt-choose(85): 按索引从候选中选择"
  (vt-to-list (vt-choose (list (vt-from-sequence '((1 2) (3 4)))
                               (vt-from-sequence '((10 20) (30 40))))
                         (vt-from-sequence '((1 0) (1 1)) :dtype :int64)))
  '((10 2) (30 40)))
(check "vt-select(86): condlist/choicelist 选择，default=0"
  (vt-to-list (vt-select (list (vt-from-sequence '(1 0) :dtype :int8)
                               (vt-from-sequence '(0 1) :dtype :int8))
                         (list (vt-from-sequence '(10 20))
                               (vt-from-sequence '(30 40)))
                         :default 0))
  '(10 40))
(check "vt-extract(87): 取条件为真的元素"
  (vt-to-list (vt-extract (vt-from-sequence '(1 0 1) :dtype :int8)
                          (vt-from-sequence '(10 20 30))))
  '(10 30))
(check "vt-searchsorted(88): 有序插入位置"
  (vt-to-list (vt-searchsorted (vt-from-sequence '(10 20 30))
                               (vt-from-sequence '(5 15 25))))
  '(0 1 2))
(check "vt-digitize(89): 分箱索引（right=nil 左闭右开）"
  (vt-to-list (vt-digitize (vt-from-sequence '(0.2d0 6.4d0 3.0d0 1.6d0))
                           (vt-from-sequence '(0.0d0 1.0d0 2.5d0 4.0d0 10.0d0))))
  '(1 4 3 2))
(check "vt-bincount(90): 计数（含中间空箱）"
  (vt-to-list (vt-bincount (vt-from-sequence '(0 1 1 3) :dtype :int64)))
  '(1 2 0 1))
(check "vt-bincount(90): minlength 补零"
  (vt-to-list (vt-bincount (vt-from-sequence '(0 0) :dtype :int64) :minlength 4))
  '(2 0 0 0))

;;; ================================================================
;;; 第 5 部分：算术与逐元素运算（导出符号 91–148）
;;; ================================================================

(section "⑤ 算术与逐元素 (91-148)")

(check "vt-+(91): N 元加法广播"
  (vt-+ (vt-ones '(2 2)) (vt-ones '(2)) (vt-const '(2 2) 5.0d0))
  '((7 7) (7 7)))
(check "vt-+(91): int8+int8→int8（64 格提升表）"
  (vt-+ (vt-const '(2) 1 :dtype :int8) (vt-const '(2) 1 :dtype :int8))
  '(2 2) :dtype :int8)
(check "vt--(92): 二元减法" (vt-- (vt-const '(2) 5.0d0) (vt-const '(2) 2.0d0)) '(3 3))
(check "vt--(92): 一元取负" (vt-- (vt-const '(2) 3.0d0)) '(-3 -3))
(check "vt-*(93): 逐元素乘" (vt-* (vt-const '(2) 3.0d0) (vt-const '(2) 4.0d0)) '(12 12))
(check "vt-/ (94): true_divide 整型→float64"
  (vt-/ (vt-const '(2) 1 :dtype :int32) (vt-const '(2) 4 :dtype :int32))
  '(0.25 0.25) :dtype :float64)
(check "vt-/ (94): 浮点零除→+Inf（IEEE 754）"
  (vt-/ (vt-const '(1) 1.0d0) (vt-const '(1) 0.0d0)) (list +inf+))
(check "vt-/ (94): 0/0→NaN"
  (vt-/ (vt-const '(1) 0.0d0) (vt-const '(1) 0.0d0)) (list +nan+))
(check "vt-add(95): add 别名" (vt-add (vt-ones '(2)) (vt-ones '(2))) '(2 2))
(check "vt-sub(96): sub" (vt-sub (vt-const '(2) 5.0d0) (vt-ones '(2))) '(4 4))
(check "vt-mul(97): mul" (vt-mul (vt-const '(2) 3.0d0) (vt-ones '(2))) '(3 3))
(check "vt-div(98): floor_divide 整数 7/2→3，-7/2→-4（D16）"
  (vt-div (vt-from-sequence '(7 -7) :dtype :int64) (vt-const '(2) 2 :dtype :int64))
  '(3 -4) :dtype :int64)
(check "vt-div(98): 整数零除→0，dtype 保持整型（D16）"
  (vt-div (vt-const '(1) 3 :dtype :int64) (vt-const '(1) 0 :dtype :int64))
  '(0) :dtype :int64)
(check "vt-scale(99): 张量×标量" (vt-scale (vt-ones '(2)) 5.0d0) '(5 5))
(check "vt-square(100): 平方" (vt-square (vt-const '(2) 3.0d0)) '(9 9))
(check "vt-expt(101): 乘方" (vt-expt (vt-const '(2) 2.0d0) 3.0d0) '(8 8))
(check "vt-pow(102): expt 别名" (vt-pow (vt-const '(2) 2.0d0) 3.0d0) '(8 8))
(check "vt-sqrt(103): sqrt(4)=2" (vt-sqrt (vt-const '(1) 4.0d0)) '(2))
(check "vt-sqrt(103): 负数→NaN" (vt-sqrt (vt-const '(1) -1.0d0)) (list +nan+))
(check "vt-abs(104): 绝对值" (vt-abs (vt-from-sequence '(-3.0d0 3.0d0))) '(3 3))
(check "vt-signum(105): 符号函数" (vt-signum (vt-from-sequence '(-5.0d0 0.0d0 5.0d0))) '(-1 0 1))
(check "vt-signum(105): NaN→NaN（§9.2）" (vt-signum (vt-const '(1) +nan+)) (list +nan+))
(check "vt-mod(106): 余数跟随除数 mod(-7,3)=2"
  (vt-mod (vt-const '(1) -7 :dtype :int64) (vt-const '(1) 3 :dtype :int64)) '(2) :dtype :int64)
(check "vt-mod(106): 整数零除→0（D16）"
  (vt-mod (vt-const '(1) 3 :dtype :int64) (vt-const '(1) 0 :dtype :int64)) '(0) :dtype :int64)
(check "vt-rem(107): fmod 余数跟随被除数 rem(-7,3)=-1"
  (vt-rem (vt-const '(1) -7 :dtype :int64) (vt-const '(1) 3 :dtype :int64)) '(-1) :dtype :int64)
(check "vt-round(108): 银行家舍入 round(2.5)=2"
  (vt-round (vt-from-sequence '(2.5d0 3.5d0))) '(2 4))
(check "vt-floor(109): 向下取整" (vt-floor (vt-const '(1) 1.7d0)) '(1))
(check "vt-floor(109): NaN→NaN（SBCL 陷阱已拦截 §7.1）"
  (vt-floor (vt-const '(1) +nan+)) (list +nan+))
(check "vt-ceiling(110): 向上取整" (vt-ceiling (vt-const '(1) 1.3d0)) '(2))
(check "vt-truncate(111): 向零截断" (vt-truncate (vt-const '(1) -1.7d0)) '(-1))
(check "vt-truncate(111): Inf→Inf（§7.1）" (vt-truncate (vt-const '(1) +inf+)) (list +inf+))
(check "vt-rint(112): 半到偶舍入 (0.5 1.5)→(0 2)"
  (vt-rint (vt-from-sequence '(0.5d0 1.5d0))) '(0 2))
(check "vt-log(113): 自然对数" (vt-log (vt-exp (vt-const '(1) 1.0d0))) '(1))
(check "vt-log(113): :base 2 log2(8)=3" (vt-log (vt-const '(1) 8.0d0) :base 2.0d0) '(3))
(check "vt-log2(114): log2(8)=3" (vt-log2 (vt-const '(1) 8.0d0)) '(3))
(check "vt-log10(115): log10(1000)=3" (vt-log10 (vt-const '(1) 1000.0d0)) '(3))
(check "vt-exp(116): exp(0)=1" (vt-exp (vt-const '(1) 0.0d0)) '(1))
(check "vt-clip(117): 双侧裁剪（D5）"
  (vt-clip (vt-from-sequence '(1.0d0 2.5d0 4.0d0)) 2.0d0 3.0d0) '(2 2.5 3))
(check "vt-clip(117): 仅上限 (clip a nil max)（D5）"
  (vt-clip (vt-from-sequence '(1.0d0 2.5d0 4.0d0)) nil 3.0d0) '(1 2.5 3))
(check-err "vt-clip(117): 仅下限报错（D5 对齐 numpy TypeError）"
  (vt-clip (vt-from-sequence '(1.0d0)) 2.0d0))
(check "vt-sin(118): sin(0)=0" (vt-sin (vt-const '(1) 0.0d0)) '(0))
(check "vt-cos(119): cos(0)=1" (vt-cos (vt-const '(1) 0.0d0)) '(1))
(check "vt-tan(120): tan(0)=0" (vt-tan (vt-const '(1) 0.0d0)) '(0))
(check "vt-asin(121): asin(1)=π/2" (vt-asin (vt-const '(1) 1.0d0)) (list (/ pi 2)))
(check "vt-asin(121): 定义域外→NaN" (vt-asin (vt-const '(1) 2.0d0)) (list +nan+))
(check "vt-acos(122): acos(1)=0" (vt-acos (vt-const '(1) 1.0d0)) '(0))
(check "vt-atan(123): atan(0)=0" (vt-atan (vt-const '(1) 0.0d0)) '(0))
(check "vt-atan2(124): atan2(1,1)=π/4" (vt-atan2 (vt-const '(1) 1.0d0) (vt-const '(1) 1.0d0)) (list (/ pi 4)))
(check "vt-sinh(125): sinh(0)=0" (vt-sinh (vt-const '(1) 0.0d0)) '(0))
(check "vt-cosh(126): cosh(0)=1" (vt-cosh (vt-const '(1) 0.0d0)) '(1))
(check "vt-tanh(127): tanh(0)=0" (vt-tanh (vt-const '(1) 0.0d0)) '(0))
(check "vt-hypot(128): hypot(3,4)=5" (vt-hypot (vt-const '(1) 3.0d0) (vt-const '(1) 4.0d0)) '(5))
(check "vt-hypot(128): hypot(Inf,NaN)=Inf（D4）"
  (vt-hypot (vt-const '(1) +inf+) (vt-const '(1) +nan+)) (list +inf+))
(check "vt-sinc(129): sinc(0)=1" (vt-sinc (vt-const '(1) 0.0d0)) '(1))
(check "vt-sinc(129): sinc(0.5)=2/π≈0.63662"
  (vt-sinc (vt-const '(1) 0.5d0)) (list (/ 2.0d0 pi)))
(check "vt-deg2rad(130): 180°=π" (vt-deg2rad (vt-const '(1) 180.0d0)) (list pi))
(check "vt-rad2deg(131): π=180°" (vt-rad2deg (vt-const '(1) pi)) '(180))
(check "vt-asinh(132): asinh(0)=0" (vt-asinh (vt-const '(1) 0.0d0)) '(0))
(check "vt-acosh(133): acosh(1)=0" (vt-acosh (vt-const '(1) 1.0d0)) '(0))
(check "vt-atanh(134): atanh(±1)→±Inf（F3）"
  (vt-atanh (vt-from-sequence '(1.0d0 -1.0d0))) (list +inf+ +ninf+))
(check "vt-atanh(134): |x|>1→NaN" (vt-atanh (vt-const '(1) 2.0d0)) (list +nan+))
(check "vt-cbrt(135): 立方根(8)=2" (vt-cbrt (vt-const '(1) 8.0d0)) '(2))
(check "vt-cbrt(135): 负数立方根(−8)=−2" (vt-cbrt (vt-const '(1) -8.0d0)) '(-2))
(check "vt-reciprocal(136): 整数→截断 0（D9 numpy 语义）"
  (vt-reciprocal (vt-const '(1) 2 :dtype :int32)) '(0) :dtype :int32)
(check "vt-reciprocal(136): 浮点 1/4"
  (vt-reciprocal (vt-const '(1) 4.0d0)) '(0.25))
(check "vt-negative(137): 取负" (vt-negative (vt-const '(2) 3.0d0)) '(-3 -3))
(check "vt-lerp(138): 线性插值 lerp(0,10,0.25)=2.5"
  (vt-lerp (vt-const '(1) 0.0d0) (vt-const '(1) 10.0d0) 0.25d0) '(2.5))
(check "vt-bit-and(139): 6&3=2"
  (vt-bit-and (vt-const '(1) 6 :dtype :int32) (vt-const '(1) 3 :dtype :int32)) '(2) :dtype :int32)
(check "vt-bit-ior(140): 6|3=7"
  (vt-bit-ior (vt-const '(1) 6 :dtype :int32) (vt-const '(1) 3 :dtype :int32)) '(7) :dtype :int32)
(check "vt-bit-xor(141): 6^3=5"
  (vt-bit-xor (vt-const '(1) 6 :dtype :int32) (vt-const '(1) 3 :dtype :int32)) '(5) :dtype :int32)
(check "vt-bit-not(142): ~0=-1"
  (vt-bit-not (vt-const '(1) 0 :dtype :int32)) '(-1) :dtype :int32)
(check "vt-left-shift(143): 1<<3=8"
  (vt-left-shift (vt-const '(1) 1 :dtype :int32) 3) '(8) :dtype :int32)
(check "vt-right-shift(144): 8>>3=1"
  (vt-right-shift (vt-const '(1) 8 :dtype :int32) 3) '(1) :dtype :int32)
(check "vt-fmax(145): 忽略 NaN fmax(NaN,1)=1（D10）"
  (vt-fmax (vt-const '(2) +nan+) (vt-const '(2) 1.0d0)) '(1 1))
(check "vt-fmin(146): 忽略 NaN fmin(NaN,3)=3"
  (vt-fmin (vt-const '(2) +nan+) (vt-const '(2) 3.0d0)) '(3 3))
(check "vt-maximum(147): NaN 传播 maximum(NaN,1)=NaN（D10）"
  (vt-maximum (vt-const '(2) +nan+) (vt-const '(2) 1.0d0)) (list +nan+ +nan+))
(check "vt-minimum(148): NaN 传播"
  (vt-minimum (vt-const '(2) +nan+) (vt-const '(2) 3.0d0)) (list +nan+ +nan+))

;;; ================================================================
;;; 第 6 部分：比较与逻辑（导出符号 149–171）
;;; ================================================================

(section "⑥ 比较与逻辑 (149-171)")

;; §3.1 / F4：比较运算返回 :int8 承载布尔（0/1）
(check "vt-=(149): 相等→int8 0/1（F4）"
  (vt-= (vt-from-sequence '(1 2) :dtype :int32) (vt-from-sequence '(1 3) :dtype :int32))
  '(1 0) :dtype :int8)
(check "vt-/=(150): 不等" (vt-/= (vt-ones '(2)) (vt-ones '(2))) '(0 0) :dtype :int8)
(check "vt-<(151): 小于" (vt-< (vt-const '(2) 1.0d0) (vt-const '(2) 2.0d0)) '(1 1) :dtype :int8)
(check "vt-<=(152): 小于等于" (vt-<= (vt-ones '(2)) (vt-ones '(2))) '(1 1) :dtype :int8)
(check "vt->(153): 大于" (vt-> (vt-const '(2) 2.0d0) (vt-const '(2) 1.0d0)) '(1 1) :dtype :int8)
(check "vt->=(154): 大于等于" (vt->= (vt-ones '(2)) (vt-ones '(2))) '(1 1) :dtype :int8)
(check "vt-positive-p(155): >0→1（int8）"
  (vt-positive-p (vt-from-sequence '(2.0d0 -2.0d0))) '(1 0) :dtype :int8)
(check "vt-negative-p(156): <0→1"
  (vt-negative-p (vt-from-sequence '(2.0d0 -2.0d0))) '(0 1) :dtype :int8)
(check "vt-zero-p(157): =0→1"
  (vt-zero-p (vt-from-sequence '(0.0d0 1.0d0))) '(1 0) :dtype :int8)
(check "vt-nonzero-p(158): ≠0→1"
  (vt-nonzero-p (vt-from-sequence '(0.0d0 1.0d0))) '(0 1) :dtype :int8)
(check "vt-even-p(159): 偶数→1"
  (vt-even-p (vt-from-sequence '(2 3) :dtype :int64)) '(1 0) :dtype :int8)
(check "vt-odd-p(160): 奇数→1"
  (vt-odd-p (vt-from-sequence '(2 3) :dtype :int64)) '(0 1) :dtype :int8)
(check "vt-logical-and(161): 逻辑与"
  (vt-logical-and (vt-from-sequence '(1 0 1) :dtype :int8)
                  (vt-from-sequence '(1 1 0) :dtype :int8))
  '(1 0 0) :dtype :int8)
(check "vt-logical-or(162): 逻辑或"
  (vt-logical-or (vt-from-sequence '(1 0 0) :dtype :int8)
                 (vt-from-sequence '(0 1 0) :dtype :int8))
  '(1 1 0) :dtype :int8)
(check "vt-logical-not(163): 逻辑非"
  (vt-logical-not (vt-from-sequence '(1 0) :dtype :int8)) '(0 1) :dtype :int8)
(check "vt-logical-xor(164): 逻辑异或"
  (vt-logical-xor (vt-from-sequence '(1 1 0) :dtype :int8)
                  (vt-from-sequence '(1 0 0) :dtype :int8))
  '(0 1 0) :dtype :int8)
(check "vt-all(165): 全真→1" (vt-all (vt-ones '(2 2))) 1)
(check "vt-all(165): axis 归约" (vt-to-list (vt-all (vt-from-sequence '((1 0)) :dtype :int8) :axis 0)) '(1 0))
(check "vt-any(166): 存在真→1" (vt-any (vt-from-sequence '(0 0 1) :dtype :int8)) 1)
(check "vt-any(166): 全假→0" (vt-any (vt-from-sequence '(0 0) :dtype :int8)) 0)
(check "vt-isclose(167): 默认 rtol=1e-5 atol=1e-8"
  (vt-isclose (vt-const '(1) 1.0d0) (vt-const '(1) (+ 1.0d0 1.0d-7))) '(1))
(check "vt-isclose(167): NaN 互不接近"
  (vt-isclose (vt-const '(1) +nan+) (vt-const '(1) +nan+)) '(0))
(check-true "vt-allclose(168): 元素全部接近→真"
  (= (vt-item (vt-all (vt-isclose (vt-ones '(3)) (vt-ones '(3))))) 1))
(check-false "vt-allclose(168): 有元素不接近→假"
  (= (vt-item (vt-all (vt-isclose (vt-ones '(3)) (vt-const '(3) 2.0d0)))) 1))
(check "vt-isfinite(169): 有限→1，Inf/NaN→0"
  (vt-isfinite (vt-from-sequence (list 1.0d0 +inf+ +nan+))) '(1 0 0) :dtype :int8)
(check "vt-isinf(170): Inf→1（含 ±Inf）"
  (vt-isinf (vt-from-sequence (list +inf+ +ninf+ 0.0d0))) '(1 1 0) :dtype :int8)
(check "vt-isnan(171): NaN→1"
  (vt-isnan (vt-from-sequence (list +nan+ 1.0d0))) '(1 0) :dtype :int8)

;;; ================================================================
;;; 第 7 部分：归约与统计（导出符号 172–200）
;;; ================================================================

(section "⑦ 归约与统计 (172-200)")

(check "vt-sum(172): 全归约→0 维 vt（§6.5）"
  (vt-sum (vt-ones '(2 3))) 6 :shape nil)
(check "vt-sum(172): int8→int64（D12）"
  (vt-sum (vt-ones '(3) :dtype :int8)) 3 :dtype :int64)
(check "vt-sum(172): float32→float32 保持（D12）"
  (vt-sum (vt-ones '(3) :dtype :float32)) 3 :dtype :float32)
(check "vt-sum(172): axis=0 归约"
  (vt-to-list (vt-sum (vt-ones '(2 3)) :axis 0)) '(2 2 2))
(check "vt-sum(172): 空归约→0（单位元族 §4.5）" (vt-sum (vt-zeros '(0))) 0)
(check "vt-mean(173): int32→float64（D12）"
  (vt-mean (vt-from-sequence '(1 2 3) :dtype :int32)) 2.0 :dtype :float64)
(check "vt-mean(173): axis 归约" (vt-to-list (vt-mean (vt-ones '(2 3)) :axis 0)) '(1 1 1))
(check "vt-average(174): 加权平均"
  (vt-average (vt-from-sequence '(1 2 3)) (vt-ones '(3))) 2.0)
(check "vt-std(175): 标准差 (1 2 3) ddof=0 → √(2/3)"
  (vt-std (vt-from-sequence '(1.0d0 2.0d0 3.0d0))) (sqrt (/ 2.0d0 3.0d0)))
(check "vt-std(175): ddof=1 → 1.0"
  (vt-std (vt-from-sequence '(1.0d0 2.0d0 3.0d0) ) :ddof 1) 1.0 :shape nil)
(check "vt-var(176): 方差 (1 2 3) = 2/3"
  (vt-var (vt-from-sequence '(1.0d0 2.0d0 3.0d0))) (/ 2.0d0 3.0d0))
(check "vt-var(176): ddof=1 → 1.0"
  (vt-var (vt-from-sequence '(1.0d0 2.0d0 3.0d0)) :ddof 1) 1.0 :shape nil)
(check "vt-amax(177): 最大值" (vt-amax (vt-from-sequence '(1.0d0 5.0d0 3.0d0))) 5)
(check "vt-amax(177): 首个 NaN 胜出（D14）"
  (vt-amax (vt-from-sequence (list +nan+ 1.0d0 -5.0d0))) +nan+)
(check "vt-amin(178): 最小值" (vt-amin (vt-from-sequence '(1.0d0 5.0d0 3.0d0))) 1)
(check "vt-amin(178): NaN 传播" (vt-amin (vt-from-sequence (list +nan+ 1.0d0))) +nan+)
(check "vt-argmax(179): 最大值索引" (vt-argmax (vt-from-sequence '(1.0d0 5.0d0 3.0d0))) 1)
(check "vt-argmax(179): 首个 NaN 下标（D14）"
  (vt-argmax (vt-from-sequence (list 1.0d0 +nan+ 5.0d0))) 1)
(check "vt-argmin(180): 最小值索引" (vt-argmin (vt-from-sequence '(3.0d0 1.0d0 2.0d0))) 1)
(check "vt-prod(181): 连乘" (vt-prod (vt-from-sequence '(1 2 3 4))) 24)
(check "vt-cumsum(182): 前缀和" (vt-to-list (vt-cumsum (vt-from-sequence '(1 2 3)))) '(1 3 6))
(check "vt-cumprod(183): 前缀积" (vt-to-list (vt-cumprod (vt-from-sequence '(1 2 3)))) '(1 2 6))
(check "vt-median(184): 偶数个取均值" (vt-median (vt-from-sequence '(1.0d0 2.0d0 3.0d0 4.0d0))) 2.5 :shape nil)
(check "vt-median(184): 奇数个取中位" (vt-median (vt-from-sequence '(3.0d0 1.0d0 2.0d0))) 2.0)
(check "vt-percentile(185): 50→中位数"
  (vt-percentile (vt-from-sequence '(1.0d0 2.0d0 3.0d0 4.0d0)) 50) 2.5 :shape nil)
(check "vt-percentile(185): 0→最小值" (vt-percentile (vt-from-sequence '(1.0d0 2.0d0)) 0) 1.0 :shape nil)
(check-err "vt-percentile(185): 越界 >100 报错"
  (vt-percentile (vt-from-sequence '(1.0d0 2.0d0)) 101))
(check "vt-quantile(186): 0.5→中位数"
  (vt-quantile (vt-from-sequence '(1.0d0 2.0d0 3.0d0 4.0d0)) 0.5d0) 2.5 :shape nil)
(check "vt-ptp(187): 极差 max−min"
  (vt-ptp (vt-from-sequence '(1.0d0 2.0d0 3.0d0 4.0d0))) 3.0 :shape nil)
;; vt-histogram：多值返回 (counts edges)
(check-mv "vt-histogram(188): 2 箱直方图 counts"
  (vt-histogram (vt-from-sequence '(1.0d0 2.0d0 3.0d0)) :bins 2 :range (list 1.0d0 3.0d0))
  (:first '(1 2) :shape '(2) :dtype :int64)
  (:second '(1.0 2.0 3.0) :shape '(3)))
(check "vt-trapz(189): 梯形积分 dx=1 → 4"
  (vt-trapz (vt-from-sequence '(1.0d0 2.0d0 3.0d0))) 4.0 :shape nil)
(check "vt-trapz(189): 非均匀 x → 6.5"
  (vt-trapz (vt-from-sequence '(1.0d0 2.0d0 3.0d0)) :x (vt-from-sequence '(0.0d0 1.0d0 3.0d0)))
  6.5 :shape nil)
(check "vt-gradient(190): 中心差分 (1 2 4 8)→(1 1.5 3 4)"
  (vt-to-list (vt-gradient (vt-from-sequence '(1.0d0 2.0d0 4.0d0 8.0d0)))) '(1 1.5 3 4))
(check "vt-diff(191): 一阶差分" (vt-to-list (vt-diff (vt-from-sequence '(1 4 9)))) '(3 5))
(check "vt-diff(191): n=2 二阶差分" (vt-to-list (vt-diff (vt-from-sequence '(1 4 9)) :n 2)) '(2))
(check "vt-correlate(192): 一维相关（对标 np.correlate full）"
  (vt-to-list (vt-correlate (vt-from-sequence '(1.0d0 2.0d0 3.0d0))
                            (vt-from-sequence '(0.0d0 1.0d0 0.5d0))))
  '(0.5 2.0 3.5 3.0 0.0))
(check "vt-convolve(193): 一维卷积（对标 np.convolve full）"
  (vt-to-list (vt-convolve (vt-from-sequence '(1.0d0 2.0d0 3.0d0))
                           (vt-from-sequence '(0.0d0 1.0d0 0.5d0))))
  '(0.0 1.0 2.5 4.0 1.5))
(check "vt-sort(194): 升序，NaN 在末尾（§5.6）"
  (vt-to-list (vt-sort (vt-from-sequence (list 3.0d0 +nan+ 1.0d0))))
  (list 1.0d0 3.0d0 +nan+))
(check "vt-argsort(195): NaN 在末尾的索引"
  (vt-to-list (vt-argsort (vt-from-sequence (list 3.0d0 +nan+ 1.0d0)))) '(2 0 1))
(check "vt-sort(194): 2 维沿 axis=0"
  (vt-to-list (vt-sort (vt-from-sequence '((3 1) (2 4))) :axis 0)) '((2 1) (3 4)))
(check "vt-argsort(195): axis=0"
  (vt-to-list (vt-argsort (vt-from-sequence '((3 1) (2 4))) :axis 0)) '((1 0) (0 1)))
(check "vt-nansum(196): 跳过 NaN" (vt-nansum (vt-from-sequence (list +nan+ 1.0d0 3.0d0))) 4.0 :shape nil)
(check "vt-nanmean(197): 跳过 NaN 均值" (vt-nanmean (vt-from-sequence (list +nan+ 1.0d0 3.0d0))) 2.0 :shape nil)
(check "vt-nanstd(198): 跳过 NaN 标准差" (vt-nanstd (vt-from-sequence (list +nan+ 1.0d0 3.0d0))) 1.0 :shape nil)
(check "vt-nanvar(199): 跳过 NaN 方差" (vt-nanvar (vt-from-sequence (list +nan+ 1.0d0 3.0d0))) 1.0 :shape nil)
(check "vt-nanmax(200): 跳过 NaN 最大" (vt-nanmax (vt-from-sequence (list +nan+ 1.0d0 3.0d0))) 3.0 :shape nil)

;;; ================================================================
;;; 汇总
;;; ================================================================

(format t "~%================================================================~%")
(format t "测试汇总: PASS=~d  FAIL=~d  TOTAL=~d~%" *pass* *fail* (+ *pass* *fail*))
(if (null *failed-names*)
    (format t "全部通过 ✔~%")
    (progn
      (format t "失败清单:~%")
      (dolist (n (nreverse *failed-names*))
        (format t "  - ~a~%" n))))
(format t "================================================================~%")
(sb-ext:exit :code (if (zerop *fail*) 0 1))
