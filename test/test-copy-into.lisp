;;;; test-copy-into.lisp — vt-copy-into 完整正确性测试
;;;;
;;;; 用法：
;;;;   (in-package :clvt)
;;;;   (load "test-copy-into.lisp")
;;;;   (run-copy-into-tests)
(require :asdf)
#+quicklisp (ql:quickload :clvt)
(asdf:load-system :clvt)
(in-package :clvt)

;;; ==================================================================
;;; 轻量测试框架
;;; ==================================================================

(defparameter *copy-test-passed* 0)
(defparameter *copy-test-failed* 0)

(defmacro check (form &optional (label ""))
  "断言 FORM 为真；失败时打印标签与表达式。"
  `(if ,form
       (incf *copy-test-passed*)
       (progn
         (incf *copy-test-failed*)
         (format t "~&  [FAIL] ~a — ~s~%" ,label ',form))))

(defmacro check-equal (expected actual &optional (label ""))
  "断言两个值 equal。"
  `(let ((e ,expected) (a ,actual))
     (if (equal e a)
         (incf *copy-test-passed*)
         (progn
           (incf *copy-test-failed*)
           (format t "~&  [FAIL] ~a~%    expected: ~s~%    actual:   ~s~%"
                   ,label e a)))))

(defmacro check-error (form &optional (label ""))
  "断言 FORM 会报错。"
  `(handler-case (progn ,form
                        (incf *copy-test-failed*)
                        (format t "~&  [FAIL] ~a — 期望报错但未报错~%" ,label))
     (error () (incf *copy-test-passed*))))

;;; ==================================================================
;;; 1. 连续 + 同形 + 同 dtype
;;; ==================================================================

(defun test-copy-contig-same-dtype ()
  ;; 基础 2D
  (let ((dest (vt-zeros '(2 3) :dtype :float64))
        (src  (vt-from-sequence '((1 2 3) (4 5 6)) :dtype :float64)))
    (vt-copy-into dest src)
    (check-equal '((1.0d0 2.0d0 3.0d0) (4.0d0 5.0d0 6.0d0))
                 (vt-to-list dest)
                 "连续同形同型 2D"))

  ;; 返回值必须是 dest 本身
  (let ((dest (vt-zeros '(3) :dtype :float64))
        (src  (vt-ones  '(3) :dtype :float64)))
    (check (eq (vt-copy-into dest src) dest) "返回 dest 本身"))

  ;; 1D 大数组（压测快路径）
  (let ((dest (vt-zeros '(100) :dtype :float64))
        (src  (vt-arange 100 :dtype :float64)))
    (vt-copy-into dest src)
    (check-equal (vt-to-list src) (vt-to-list dest) "连续同形同型 1D=100")))

;;; ==================================================================
;;; 2. 连续 + 同形 + 类型转换
;;; ==================================================================

(defun test-copy-contig-type-convert ()
  ;; int32 -> float64
  (let ((dest (vt-zeros '(2 3) :dtype :float64))
        (src  (vt-from-sequence '((1 2 3) (4 5 6)) :dtype :int32)))
    (vt-copy-into dest src)
    (check-equal '((1.0d0 2.0d0 3.0d0) (4.0d0 5.0d0 6.0d0))
                 (vt-to-list dest)
                 "int32 -> float64"))

  ;; float64 -> float32
  (let ((dest (vt-zeros '(3) :dtype :float32))
        (src  (vt-from-sequence '(1.5 2.5 3.5) :dtype :float64)))
    (vt-copy-into dest src)
    (check-equal '(1.5f0 2.5f0 3.5f0) (vt-to-list dest) "float64 -> float32"))

  ;; float64 -> int32（截断）
  (let ((dest (vt-zeros '(4) :dtype :int32))
        (src  (vt-from-sequence '(1.9 -2.9 3.1 0.0) :dtype :float64)))
    (vt-copy-into dest src)
    (check-equal '(1 -2 3 0) (vt-to-list dest) "float64 -> int32 截断")))

;;; ==================================================================
;;; 3. 广播
;;; ==================================================================

(defun test-copy-broadcast ()
  ;; 行向量 (3,) -> (2,3)
  (let ((dest (vt-zeros '(2 3) :dtype :float64))
        (src  (vt-from-sequence '(1 2 3) :dtype :float64)))
    (vt-copy-into dest src)
    (check-equal '((1.0d0 2.0d0 3.0d0) (1.0d0 2.0d0 3.0d0))
                 (vt-to-list dest)
                 "广播 (3,) -> (2,3)"))

  ;; 行向量 (1,3) -> (2,3)
  (let ((dest (vt-zeros '(2 3) :dtype :float64))
        (src  (vt-from-sequence '((1 2 3)) :dtype :float64)))
    (vt-copy-into dest src)
    (check-equal '((1.0d0 2.0d0 3.0d0) (1.0d0 2.0d0 3.0d0))
                 (vt-to-list dest)
                 "广播 (1,3) -> (2,3)"))

  ;; 列向量 (2,1) -> (2,3)
  (let ((dest (vt-zeros '(2 3) :dtype :float64))
        (src  (vt-from-sequence '((1) (2)) :dtype :float64)))
    (vt-copy-into dest src)
    (check-equal '((1.0d0 1.0d0 1.0d0) (2.0d0 2.0d0 2.0d0))
                 (vt-to-list dest)
                 "广播 (2,1) -> (2,3)"))

  ;; 标量 nil -> (2,3)
  (let ((dest (vt-zeros '(2 3) :dtype :float64))
        (src  (make-vt nil 7.0d0 :dtype :float64)))
    (vt-copy-into dest src)
    (check-equal '((7.0d0 7.0d0 7.0d0) (7.0d0 7.0d0 7.0d0))
                 (vt-to-list dest)
                 "广播 () -> (2,3)"))

  ;; size=1 视图广播
  (let* ((a    (vt-from-sequence '(42 1 2 3) :dtype :float64))
         (dest (vt-zeros '(4) :dtype :float64))
         (src  (vt-slice a '(0 1))))            ; shape (1,)，size=1
    (vt-copy-into dest src)
    (check-equal '(42.0d0 42.0d0 42.0d0 42.0d0)
                 (vt-to-list dest)
                 "size=1 视图广播"))

  ;; 广播 + 类型转换：(3,) int32 -> (2,3) float64
  (let ((dest (vt-zeros '(2 3) :dtype :float64))
        (src  (vt-from-sequence '(1 2 3) :dtype :int32)))
    (vt-copy-into dest src)
    (check-equal '((1.0d0 2.0d0 3.0d0) (1.0d0 2.0d0 3.0d0))
                 (vt-to-list dest)
                 "广播 + 类型转换")))

;;; ==================================================================
;;; 4. 非连续 src
;;; ==================================================================

(defun test-copy-noncontig-src ()
  ;; 转置视图
  (let* ((base (vt-from-sequence '((1 2 3) (4 5 6)) :dtype :float64))
         (src  (vt-transpose base))                     ; (3,2) 非连续
         (dest (vt-zeros '(3 2) :dtype :float64)))
    (vt-copy-into dest src)
    (check-equal '((1.0d0 4.0d0) (2.0d0 5.0d0) (3.0d0 6.0d0))
                 (vt-to-list dest)
                 "非连续 src（转置）"))

  ;; 负 stride 翻转
  (let* ((base (vt-from-sequence '(1 2 3 4 5) :dtype :float64))
         (src  (vt-flip base))
         (dest (vt-zeros '(5) :dtype :float64)))
    (vt-copy-into dest src)
    (check-equal '(5.0d0 4.0d0 3.0d0 2.0d0 1.0d0)
                 (vt-to-list dest)
                 "非连续 src（翻转）"))

  ;; 步长切片
  (let* ((base (vt-arange 10 :dtype :float64))
         (src  (vt-slice base '(1 10 2)))               ; (1 3 5 7 9)
         (dest (vt-zeros '(5) :dtype :float64)))
    (vt-copy-into dest src)
    (check-equal '(1.0d0 3.0d0 5.0d0 7.0d0 9.0d0)
                 (vt-to-list dest)
                 "非连续 src（步长 2 切片）"))

  ;; 非连续 + 类型转换
  (let* ((base (vt-from-sequence '((1 2 3) (4 5 6)) :dtype :int32))
         (src  (vt-transpose base))
         (dest (vt-zeros '(3 2) :dtype :float64)))
    (vt-copy-into dest src)
    (check-equal '((1.0d0 4.0d0) (2.0d0 5.0d0) (3.0d0 6.0d0))
                 (vt-to-list dest)
                 "非连续 src + 类型转换")))

;;; ==================================================================
;;; 5. 非连续 dest
;;; ==================================================================

(defun test-copy-noncontig-dest ()
  ;; dest 为转置视图；写入后应反映到其底层数组
  (let* ((dest-base (vt-zeros '(3 2) :dtype :float64))
         (dest      (vt-transpose dest-base))           ; (2,3) 非连续
         (src       (vt-from-sequence '((1 2 3) (4 5 6)) :dtype :float64)))
    (vt-copy-into dest src)
    (check-equal '((1.0d0 4.0d0) (2.0d0 5.0d0) (3.0d0 6.0d0))
                 (vt-to-list dest-base)
                 "非连续 dest（转置视图）写入底层数组"))

  ;; 非连续 dest + 类型转换
  (let* ((dest-base (vt-zeros '(3 2) :dtype :float64))
         (dest      (vt-transpose dest-base))
         (src       (vt-from-sequence '((1 2 3) (4 5 6)) :dtype :int32)))
    (vt-copy-into dest src)
    (check-equal '((1.0d0 4.0d0) (2.0d0 5.0d0) (3.0d0 6.0d0))
                 (vt-to-list dest-base)
                 "非连续 dest + 类型转换")))

;;; ==================================================================
;;; 6. 非连续 src + 非连续 dest
;;; ==================================================================

(defun test-copy-noncontig-both ()
  (let* ((src-base  (vt-from-sequence '((1 2 3) (4 5 6)) :dtype :float64))
         (src       (vt-transpose src-base))            ; (3,2)
         (dest-base (vt-zeros '(2 3) :dtype :float64))
         (dest      (vt-transpose dest-base)))          ; (3,2)
    (vt-copy-into dest src)
    ;; dest 是从 dest-base 转置得到；写入后 dest-base 应等于 src-base
    (check-equal '((1.0d0 2.0d0 3.0d0) (4.0d0 5.0d0 6.0d0))
                 (vt-to-list dest-base)
                 "非连续 src → 非连续 dest")))

;;; ==================================================================
;;; 7. 重叠拷贝（memmove 语义）
;;; ==================================================================

(defun test-copy-overlap ()
  ;; 7a. 前向重叠：a[1:5] = a[0:4]，期望 a = (0 0 1 2 3)
  (let* ((a    (vt-from-sequence '(0 1 2 3 4) :dtype :float64))
         (dest (vt-slice a '(1 5)))
         (src  (vt-slice a '(0 4))))
    (vt-copy-into dest src)
    (check-equal '(0.0d0 0.0d0 1.0d0 2.0d0 3.0d0)
                 (vt-to-list a)
                 "memmove 前向重叠 a[1:5]=a[0:4]"))

  ;; 7b. 后向重叠：a[0:4] = a[1:5]，期望 a = (1 2 3 4 4)
  (let* ((a    (vt-from-sequence '(0 1 2 3 4) :dtype :float64))
         (dest (vt-slice a '(0 4)))
         (src  (vt-slice a '(1 5))))
    (vt-copy-into dest src)
    (check-equal '(1.0d0 2.0d0 3.0d0 4.0d0 4.0d0)
                 (vt-to-list a)
                 "memmove 后向重叠 a[0:4]=a[1:5]"))

  ;; 7c. 自赋值
  (let ((a (vt-from-sequence '(1 2 3 4 5) :dtype :float64)))
    (vt-copy-into a a)
    (check-equal '(1.0d0 2.0d0 3.0d0 4.0d0 5.0d0)
                 (vt-to-list a)
                 "自赋值"))

  ;; 7d. dest 是反向视图（负 stride）；逐元素写入顺序与物理顺序相反
  (let* ((a    (vt-from-sequence '(0 1 2 3 4) :dtype :float64))
         (dest (vt-flip a))                            ; dest[i] = a[4-i]
         (src  (vt-copy a)))                           ; 独立拷贝 (0..4)
    (vt-copy-into dest src)
    ;; dest[i] = src[i]  =>  a[4-i] = i  =>  a = (4 3 2 1 0)
    (check-equal '(4.0d0 3.0d0 2.0d0 1.0d0 0.0d0)
                 (vt-to-list a)
                 "反向视图作为 dest"))

  ;; 7e. 共享底层数组，但物理区间不重叠
  (let* ((a    (vt-from-sequence '(0 1 2 3 4 5) :dtype :float64))
         (dest (vt-slice a '(0 3)))                    ; 物理 [0,2]
         (src  (vt-slice a '(3 6))))                   ; 物理 [3,5]
    (vt-copy-into dest src)
    (check-equal '(3.0d0 4.0d0 5.0d0 3.0d0 4.0d0 5.0d0)
                 (vt-to-list a)
                 "同数组不重叠区间")))

;;; ==================================================================
;;; 8. rank 0 / 1 / 3 / 4+ （覆盖全部专门化 + 高维回退）
;;; ==================================================================

(defun test-copy-rank-cases ()
  ;; rank 0
  (let ((dest (make-vt nil 0.0d0 :dtype :float64))
        (src  (make-vt nil 42.0d0 :dtype :float64)))
    (vt-copy-into dest src)
    (check-equal 42.0d0 (vt-item dest) "rank 0"))

  ;; rank 1（非平凡长度）
  (let ((dest (vt-zeros '(5) :dtype :float64))
        (src  (vt-arange 5 :dtype :float64)))
    (vt-copy-into dest src)
    (check-equal '(0.0d0 1.0d0 2.0d0 3.0d0 4.0d0)
                 (vt-to-list dest)
                 "rank 1"))

  ;; rank 3
  (let* ((dest (vt-zeros '(2 2 2) :dtype :float64))
         (src  (vt-reshape (vt-arange 8 :dtype :float64) '(2 2 2))))
    (vt-copy-into dest src)
    (check-equal (vt-to-list src) (vt-to-list dest) "rank 3 连续"))

  ;; rank 4 连续（走里程计回退）
  (let* ((dest (vt-zeros '(2 2 2 2) :dtype :float64))
         (src  (vt-reshape (vt-arange 16 :dtype :float64) '(2 2 2 2))))
    (vt-copy-into dest src)
    (check-equal (vt-to-list src) (vt-to-list dest) "rank 4 连续"))

  ;; rank 4 非连续（转置最后两轴）
  (let* ((src-base (vt-reshape (vt-arange 16 :dtype :float64) '(2 2 2 2)))
         (src      (vt-transpose src-base '(0 1 3 2)))  ; 非连续
         (dest     (vt-zeros '(2 2 2 2) :dtype :float64)))
    (vt-copy-into dest src)
    (check-equal (vt-to-list src) (vt-to-list dest) "rank 4 非连续")))

;;; ==================================================================
;;; 9. 空张量
;;; ==================================================================

(defun test-copy-empty ()
  (let ((dest (vt-zeros '(0 3) :dtype :float64))
        (src  (vt-zeros '(0 3) :dtype :float64)))
    (vt-copy-into dest src)
    (check t "空 (0,3) 拷贝"))

  (let ((dest (vt-zeros '(0) :dtype :float64))
        (src  (vt-zeros '(0) :dtype :float64)))
    (vt-copy-into dest src)
    (check t "空 (0,) 拷贝"))

  (let ((dest (vt-zeros '(3 0) :dtype :float64))
        (src  (vt-zeros '(3 0) :dtype :float64)))
    (vt-copy-into dest src)
    (check t "空 (3,0) 拷贝")))

;;; ==================================================================
;;; 10. 错误路径
;;; ==================================================================

(defun test-copy-errors ()
  ;; 只读广播视图作为 dest（stride=0）
  (let* ((base  (vt-ones '(3) :dtype :float64))
         (bcast (vt-broadcast-to base '(2 3))))       ; stride[0] = 0
    (check-error (vt-copy-into bcast (vt-zeros '(2 3) :dtype :float64))
                 "只读广播视图写入报错"))

  ;; 形状不匹配（dest 无法容纳 src）
  (check-error (vt-copy-into (vt-zeros '(2 3) :dtype :float64)
                             (vt-zeros '(3 4) :dtype :float64))
               "形状不匹配报错")

  ;; dest 比 src 广播后形状小
  (check-error (vt-copy-into (vt-zeros '(2 3) :dtype :float64)
                             (vt-zeros '(2 4) :dtype :float64))
               "dest 形状不足报错"))

;;; ==================================================================
;;; 运行入口
;;; ==================================================================

(defun run-copy-into-tests ()
  "运行所有 vt-copy-into 正确性测试。返回 (values passed failed)。"
  (setf *copy-test-passed* 0)
  (setf *copy-test-failed* 0)
  (format t "~&=== vt-copy-into 正确性测试 ===~%")
  (test-copy-contig-same-dtype)
  (test-copy-contig-type-convert)
  (test-copy-broadcast)
  (test-copy-noncontig-src)
  (test-copy-noncontig-dest)
  (test-copy-noncontig-both)
  (test-copy-overlap)
  (test-copy-rank-cases)
  (test-copy-empty)
  (test-copy-errors)
  (format t "~&=== 通过 ~d, 失败 ~d ===~%"
          *copy-test-passed* *copy-test-failed*)
  (values *copy-test-passed* *copy-test-failed*))

;; 加载时自动跑一遍
(run-copy-into-tests)
