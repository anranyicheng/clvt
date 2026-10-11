;;;; contract-regression-test.lisp — numpy/pytorch 契约对齐回归套件
;;;;
;;;; =============================================================================
;;;; 固化「与 numpy/pytorch 契约不一致」的行为修正结果。
;;;; 每条断言在修正前的 clvt 上都会失败（或直接报错），修正后通过。
;;;; 期望值全部由 NumPy 2.4.6 实测生成（见每节注释中的等价 numpy 表达式），
;;;; 不是「照抄 clvt 输出」——即真正的差分回归。
;;;; =============================================================================
;;;;
;;;; 覆盖（按行为归类）：
;;;;   写入契约   (setf vt-ref)/(setf vt-slice) 拒绝 stride-0 广播视图写入（H4）
;;;;   别名语义   vt-roll 位移取模为 0 时返回拷贝（不与输入共享底层内存）
;;;;   轴参数     vt-diagonal 支持 :axis1/:axis2（默认保持 pytorch 末两轴）
;;;;   均分契约   vt-hsplit / vt-dsplit 整数段数需整除；dsplit 另需 rank>=3
;;;;   返回形态   vt-where 单参 = np.nonzero（按轴分组元组）
;;;;   广播输入   vt-select 接受标量 choice
;;;;   索引类型   vt-choose 浮点索引报错（对齐 numpy TypeError）
;;;;   形状保持   vt-digitize 多维输入保持形状；bins 接受 list/vector；递减 bins
;;;;   索引类型   vt-take / vt-put 浮点索引报错（对齐 numpy IndexError）
;;;;   axis 语义  vt-insert axis 路径施加 moveaxis(values, 0, axis)
;;;;
;;;; 运行：
;;;;   SBCL=/usr/local/bin/sbcl bash test/run-tests.sh --suite contract-regression-test
;;;; 退出码 0 = 全部通过。
;;;; =============================================================================

(asdf:load-system :clvt)
(in-package :clvt)

(defparameter *pass* 0)
(defparameter *fail* 0)

(defun report-pass (label)
  (incf *pass*) (format t "  ok   ~a~%" label) (finish-output))

(defun report-fail (label detail)
  (incf *fail*)
  (format t "  FAIL ~a~%" label)
  (when detail (format t "       ~a~%" detail))
  (finish-output))

;; 断言辅助一律**异常安全**：实现损坏时抛出的错误被计为一条失败，
;; 而不是中断整个套件——这样在未修复的源码上能一次看到全部失败项。
(defmacro check (label form)
  `(handler-case (if ,form (report-pass ,label) (report-fail ,label nil))
     (error (e) (report-fail ,label (format nil "意外报错: ~a" e)))))

(defmacro check-eq (label expected form)
  `(handler-case
       (let ((got ,form) (exp ,expected))
         (if (equal got exp)
             (report-pass ,label)
             (report-fail ,label (format nil "期望 ~S / 实得 ~S" exp got))))
     (error (e) (report-fail ,label (format nil "意外报错: ~a" e)))))

(defmacro check-shape (label expected form)
  `(check-eq ,label ,expected (vt-shape ,form)))

(defmacro section (name &body body)
  "执行一节断言；该节若因硬错误中断，只记 1 条失败，后续节继续执行。"
  `(progn
     (format t "~%--- ~a ---~%" ,name)
     (handler-case (progn ,@body)
       (error (e) (report-fail (format nil "[~a] 本节因意外错误中断" ,name)
                               (format nil "~a" e))))))

(defun check-error (label thunk)
  (handler-case
      (progn (funcall thunk)
             (report-fail (format nil "~a（应报错但静默通过——静默错值是最坏结果）" label)
                          nil))
    (error () (report-pass (format nil "~a（正确报错）" label)))))

;;; ---------------------------------------------------------------------------
;;; 共享夹具（32 位小尺寸便于逐值比对）
;;; ---------------------------------------------------------------------------
(defun a32 () (vt-reshape (vt-arange 6 :dtype :int64) '(3 2)))    ; [[0 1][2 3][4 5]]
(defun b234 () (vt-reshape (vt-arange 24 :dtype :int64) '(2 3 4)))
(defun b2345 () (vt-reshape (vt-arange 120 :dtype :int64) '(2 3 4 5)))
(defun sq33 () (vt-reshape (vt-arange 9 :dtype :int64) '(3 3)))
(defun bcast () (vt-broadcast-to (vt-narrow (a32) 0 0 1) '(3 2))) ; 列 stride=0 的只读视图

(defun run ()
  (setf *pass* 0 *fail* 0)

  ;; =========================================================================
  (section "写入契约：广播视图写穿拦截（H4）"

  ;; numpy: np.broadcast_to(x, (3,2)) 只读，赋值抛
  ;;        ValueError: assignment destination is read-only
  (let* ((base (a32))
         (b (vt-broadcast-to (vt-narrow base 0 0 1) '(3 2)))
         (snapshot (vt-to-list base)))
    (check-eq "广播视图形状 (3 2)" '(3 2) (vt-shape b))
    (check-eq "广播视图 strides=(0 1)" '(0 1) (vt-strides b))
    (check-error "(setf (vt-ref b 1 1) 99) 被拒（numpy 只读）"
                 (lambda () (setf (vt-ref b 1 1) 99)))
    (check-error "(setf (vt-slice b '(1) '(1)) 99) 标量路径被拒"
                 (lambda () (setf (vt-slice b '(1) '(1)) 99)))
    (check-error "(setf (vt-slice b '(0 2) '(0 1)) 张量) 被拒"
                 (lambda () (setf (vt-slice b '(0 2) '(0 1))
                                  (vt-full '(2 1) 5 :dtype :int64))))
    (check-error "(setf (vt-slice b '(2) '(:all)) 张量) 被拒（广播轴被索引掉也要拦）"
                 (lambda () (setf (vt-slice b '(2) '(:all))
                                  (vt-full '(2) 5 :dtype :int64))))
    (check-eq "三次失败写入后底层数据未污染" snapshot (vt-to-list base)))

  (let ((c (a32)))
    (setf (vt-ref c 1 1) 99)
    (check-eq "普通连续张量 (setf vt-ref) 仍可写" 99 (vt-ref c 1 1)))
  (let ((c (a32)))
    (setf (vt-slice c '(0 2) '(0 1)) (vt-full '(2 1) 7 :dtype :int64))
    (check-eq "普通视图 (setf vt-slice) 仍可写（rows0:2,col0）"
              '((7 1) (7 3) (4 5)) (vt-to-list c)))
  ;; 对照：numpy.a[np.newaxis].put(0, v) 合法（size=1 的 stride-0 轴非广播）
  (let ((c (a32)))
    (check "(:newa) size=1/stride=0 视图 vt-put 不再误拒（numpy 允许）"
           (progn (vt-put (vt-slice c '(:newa) '(:all) '(:all)) 0 42)
                  (= (vt-ref c 0 0) 42))))

  ;; =========================================================================
)

  (section "vt-roll：位移取模为 0 必须返回拷贝"

  ;; numpy: np.roll(a, 0) 总是返回新数组；b = np.roll(a,0); b[0]=9 → a 不变
  (let* ((m (b234))
         (r (vt-roll m 0 :axis 0)))
    (check-eq "roll(0,:axis 0) 值不变" (vt-to-list m) (vt-to-list r))
    (check "roll(0,:axis 0) 不共享底层内存"
           (not (eq (vt-data r) (vt-data m))))
    (setf (vt-ref r 0 0 0) -1)
    (check-eq "写结果不回写输入" 0 (vt-ref m 0 0 0)))
  (let* ((m (b234)) (r (vt-roll m 3 :axis 0)))   ; 3 行 → mod 3 = 0
    (check "shift 为轴长整数倍时同样拷贝" (not (eq (vt-data r) (vt-data m)))))
  (let* ((m (b234)) (r (vt-roll m 0)))
    (check "roll(0) 无 axis 分支也不共享内存" (not (eq (vt-data r) (vt-data m)))))
  (let* ((m (b234)) (r (vt-roll m '(0 0) :axis '(0 1))))
    (check "多轴全 0 位移也不共享内存" (not (eq (vt-data r) (vt-data m)))))
  (check-eq "roll(arange5,0) 数值正确" '(0 1 2 3 4)
            (vt-to-list (vt-roll (vt-arange 5 :dtype :int64) 0)))
  (check-shape "roll(zeros(0),2) 退化形状不报错" '(0)
               (vt-roll (vt-zeros '(0)) 2))

  ;; =========================================================================
)

  (section "vt-diagonal：:axis1 / :axis2 轴参数"

  ;; numpy 实测（arange(24).reshape(2,3,4)）
  (check-eq "默认 = pytorch 末两轴（同 np.diagonal(a,-2,-1)）"
            '((0 5 10) (12 17 22))
            (vt-to-list (vt-diagonal (b234))))
  (check-eq ":axis1 0 :axis2 1 = np.diagonal(a, 0, 1)"
            '((0 16) (1 17) (2 18) (3 19))
            (vt-to-list (vt-diagonal (b234) :axis1 0 :axis2 1)))
  (check-eq ":axis1 0 :axis2 2 = np.diagonal(a, 0, 2)"
            '((0 13) (4 17) (8 21))
            (vt-to-list (vt-diagonal (b234) :axis1 0 :axis2 2)))
  (check-eq ":axis1 1 :axis2 2 = 末两轴"
            '((0 5 10) (12 17 22))
            (vt-to-list (vt-diagonal (b234) :axis1 1 :axis2 2)))
  (check-eq "负轴 :axis1 -2 :axis2 -1 = 末两轴"
            '((0 5 10) (12 17 22))
            (vt-to-list (vt-diagonal (b234) :axis1 -2 :axis2 -1)))
  (check-eq ":axis1 0 :axis2 2 :offset 1 = np.diagonal(a, 1, 0, 2)"
            '((1 14) (5 18) (9 22))
            (vt-to-list (vt-diagonal (b234) :offset 1 :axis1 0 :axis2 2)))
  (check-shape "4D :axis1 1 :axis2 3 形状 (2 4 3)"
               '(2 4 3) (vt-diagonal (b2345) :axis1 1 :axis2 3))
  (check-eq "4D :axis1 1 :axis2 3 值"
            '(((0 21 42) (5 26 47) (10 31 52) (15 36 57))
              ((60 81 102) (65 86 107) (70 91 112) (75 96 117)))
            (vt-to-list (vt-diagonal (b2345) :axis1 1 :axis2 3)))
  (check-shape "4D :axis1 0 :axis2 3 形状 (3 4 2)"
               '(3 4 2) (vt-diagonal (b2345) :axis1 0 :axis2 3))
  (check-eq "4D :axis1 0 :axis2 3 值（对角线轴置于末尾）"
            '(((0 61) (5 66) (10 71) (15 76))
              ((20 81) (25 86) (30 91) (35 96))
              ((40 101) (45 106) (50 111) (55 116)))
            (vt-to-list (vt-diagonal (b2345) :axis1 0 :axis2 3)))
  (check-eq "2D 默认仍为 np.diagonal(a)" '(0 4 8) (vt-to-list (vt-diagonal (sq33))))
  (check-eq "2D offset=1 仍为 np.diagonal(a,1)" '(1 5)
            (vt-to-list (vt-diagonal (sq33) :offset 1)))
  (check-error "只给 :axis1 报错（无法确定对角平面）"
               (lambda () (vt-diagonal (b234) :axis1 0)))
  (check-error "只给 :axis2 报错"
               (lambda () (vt-diagonal (b234) :axis2 1)))
  (check-error ":axis1 = :axis2 报错"
               (lambda () (vt-diagonal (b234) :axis1 1 :axis2 1)))
  (check-error ":axis1 越界（rank 3 传 3）报错"
               (lambda () (vt-diagonal (b234) :axis1 3 :axis2 1)))

  ;; =========================================================================
)

  (section "vt-hsplit / vt-dsplit：整数段数需整除"

  ;; numpy: np.hsplit(np.zeros((2,5)), 2) → ValueError: array split does not
  ;;        result in an equal division
  (check-error "hsplit (2,5) 2 不整除 → 报错"
               (lambda () (vt-hsplit (vt-zeros '(2 5)) 2)))
  (check-shape "hsplit (2,6) 2 整除的第一段 (2 3)"
               '(2 3) (first (vt-hsplit (vt-zeros '(2 6)) 2)))
  (check-error "hsplit 1D 长度 10 分 3 段不整除 → 报错"
               (lambda () (vt-hsplit (vt-arange 10 :dtype :int64) 3)))
  (check-eq "hsplit 1D 长度 10 分 5 段" '((2) (2) (2) (2) (2))
            (mapcar #'vt-shape (vt-hsplit (vt-arange 10 :dtype :int64) 5)))
  (check-eq "hsplit 索引列表无需整除（np.hsplit(a,[1,3])）"
            '((2 1) (2 2) (2 2))
            (mapcar #'vt-shape (vt-hsplit (vt-zeros '(2 5)) '(1 3))))
  (check-error "dsplit (2,3,4) 3 不整除 → 报错"
               (lambda () (vt-dsplit (vt-zeros '(2 3 4)) 3)))
  (check-eq "dsplit (2,3,4) 2 整除" '((2 3 2) (2 3 2))
            (mapcar #'vt-shape (vt-dsplit (vt-zeros '(2 3 4)) 2)))
  (check-eq "dsplit 索引列表无需整除" '((2 3 1) (2 3 2) (2 3 1))
            (mapcar #'vt-shape (vt-dsplit (vt-zeros '(2 3 4)) '(1 3))))
  (check-error "dsplit 2D → 报错（numpy: 只支持 >=3 维）"
               (lambda () (vt-dsplit (vt-zeros '(2 6)) 2)))
  (check-error "dsplit 1D → 报错"
               (lambda () (vt-dsplit (vt-arange 6 :dtype :int64) 2)))
  (check-error "vsplit 不整除仍报错（回归）"
               (lambda () (vt-vsplit (vt-zeros '(3 4)) 2)))
  (check-eq "vsplit (4,2) 2 整除" '((2 2) (2 2))
            (mapcar #'vt-shape (vt-vsplit (vt-zeros '(4 2)) 2)))
  ;; vt-split 本身保持 array_split 语义（允许不整除）
  (check-eq "vt-split (2,5) 2 仍为 array_split 语义 (2 3)+(2 2)"
            '((2 3) (2 2))
            (mapcar #'vt-shape (vt-split (vt-zeros '(2 5)) 2 :axis 1)))

  ;; =========================================================================
)

  (section "vt-where 单参 = np.nonzero（按轴分组元组）"

  ;; numpy: np.where(c) 文档明确等价 np.asarray(c).nonzero()
  (let* ((c (vt-from-sequence '(1 0 2 0) :dtype :int64))
         (w (vt-where c)))
    (check "1D 返回列表（元组），长度 = rank" (= (length w) 1))
    (check-eq "1D 索引 = [0 2]（np.nonzero）" '(0 2)
              (vt-to-list (first w)))
    (check-eq "与 vt-nonzero 完全一致"
              (mapcar #'vt-to-list (vt-nonzero c))
              (mapcar #'vt-to-list w)))
  (let* ((c (vt-from-sequence '((1 0) (0 1)) :dtype :int64))
         (w (vt-where c)))
    (check "2D 返回长度 2 的列表" (= (length w) 2))
    (check-eq "2D 轴0 索引 = [0 1]" '(0 1) (vt-to-list (first w)))
    (check-eq "2D 轴1 索引 = [0 1]" '(0 1) (vt-to-list (second w)))
    (check-eq "2D 每项形状 (n,)" '((2) (2)) (mapcar #'vt-shape w))
    (check-eq "2D 与 np.nonzero 一致（按轴分组，非 (n,rank)）"
              (mapcar #'vt-to-list (vt-nonzero c))
              (mapcar #'vt-to-list w)))
  (let* ((c (vt-= (b234) 5))
         (w (vt-where c)))
    (check "3D 返回长度 3 的列表" (= (length w) 3))
    (check-eq "3D 索引逐轴 = ([0],[1],[1])（np.nonzero）"
              '((0) (1) (1)) (mapcar #'vt-to-list w)))
  (let* ((z (vt-zeros '(4) :dtype :int64))
         (w (vt-where z)))
    (check "全零 → 长度 1 的列表" (= (length w) 1))
    (check-eq "全零 → 形状 (0,)" '(0) (vt-shape (first w))))
  (let ((c (vt-from-sequence '(1 0 2 0) :dtype :int64)))
    (check-shape "vt-argwhere 仍为 (n,rank)" '(2 1) (vt-argwhere c))
    (check-eq "vt-argwhere 值不受影响" '((0) (2)) (vt-to-list (vt-argwhere c))))
  (check-eq "三元 where 不受影响"
            '(10 20 10)
            (vt-to-list (vt-where (vt-from-sequence '(1 0 1)) 10 20)))

  ;; =========================================================================
)

  (section "vt-select：支持标量 choice"

  ;; numpy: np.select([c!=0],[99],default=0) → [99 0 99 0]
  (let* ((c (vt-from-sequence '(1 0 2 0) :dtype :int64))
         (r (vt-select (list (vt-/= c 0)) (list 99) :default 0)))
    (check-eq "标量 choice [99]" '(99 0 99 0) (vt-to-list r))
    (check "dtype 为 int64（与 numpy 一致）" (eq :int64 (vt-dtype r))))
  (let* ((c (vt-from-sequence '(1 0 2 0) :dtype :int64))
         (r (vt-select (list (vt-= c 0) (vt-> c 1))
                       (list 99 -7) :default 0)))
    (check-eq "两个标量 choice + 标量 default" '(0 99 -7 99) (vt-to-list r)))
  (let* ((c (vt-from-sequence '(1 0 2 0) :dtype :int64))
         (r (vt-select (list (vt-/= c 0))
                       (list (vt-from-sequence '(99 100 101 102) :dtype :int64)))))
    (check-eq "张量 choice 仍正常" '(99 0 101 0) (vt-to-list r)))
  (let* ((c (vt-from-sequence '(1 0 2 0) :dtype :int64))
         (r (vt-select (list (vt-/= c 0) (vt-< c 1))
                       (list (vt-from-sequence '(1 2 3 4) :dtype :int64) -5)
                       :default 0)))
    ;; numpy: np.select([c!=0, c<1], [array([1,2,3,4]), -5], default=0)
    ;;        → [1 -5 3 -5]（依次 cond 覆盖，后者优先）
    (check-eq "混合 张量/标量 choice" '(1 -5 3 -5) (vt-to-list r)))

  ;; =========================================================================
)

  (section "vt-choose：浮点索引报错"

  ;; numpy: np.choose(np.array([0.5]), [...]) → TypeError:
  ;;        'float' object cannot be interpreted as an integer
  (check-error "float64 索引 → 报错"
               (lambda () (vt-choose (list (a32) (a32))
                                     (vt-full '(3 2) 0.5d0 :dtype :float64))))
  (check-error "float32 索引 → 报错"
               (lambda () (vt-choose (list (a32) (a32))
                                     (vt-full '(3 2) 0.5 :dtype :float32))))
  (check-error "0-d 浮点索引 → 报错"
               (lambda () (vt-choose (list (a32) (a32)) 0.5d0)))
  (let* ((c0 (vt-from-sequence '((1 2) (3 4)) :dtype :int64))
         (c1 (vt-from-sequence '((10 20) (30 40)) :dtype :int64))
         (idx (vt-from-sequence '((1 0) (1 1)) :dtype :int64))
         (r (vt-choose (list c0 c1) idx)))
    ;; numpy: np.choose([[1,0],[1,1]], [[[1,2],[3,4]],[[10,20],[30,40]]])
    ;;        → [[10 2] [30 40]]
    (check-eq "int64 索引结果 = np.choose" '((10 2) (30 40)) (vt-to-list r)))
  (check-error "vt-take 浮点索引 → 报错（numpy IndexError）"
               (lambda () (vt-take (vt-arange 5) (vt-from-sequence '(0.5) :dtype :float64))))
  (check-error "vt-put 浮点张量索引 → 报错"
               (lambda () (vt-put (vt-zeros '(4) :dtype :int64)
                                  (vt-from-sequence '(0.5) :dtype :float64) 7)))
  (check-error "vt-put 浮点标量索引 → 报错"
               (lambda () (vt-put (vt-zeros '(4) :dtype :int64) 0.5 7)))

  ;; =========================================================================
)

  (section "vt-digitize：保持 X 形状 / bins 接受 list"

  ;; numpy: np.digitize(np.arange(12).reshape(3,4), [4,8])
  (let* ((x (vt-reshape (vt-arange 12 :dtype :int64) '(3 4)))
         (bins (vt-from-sequence '(4 8) :dtype :int64))
         (r (vt-digitize x bins)))
    (check-eq "多维 X 形状保持 (3 4)（原实现返回 (12)）" '(3 4) (vt-shape r))
    (check-eq "多维 X 值 = np.digitize"
              '((0 0 0 0) (1 1 1 1) (2 2 2 2)) (vt-to-list r)))
  (let* ((x (vt-reshape (vt-arange 12 :dtype :int64) '(3 4)))
         (r (vt-digitize x (vt-from-sequence '(4 8) :dtype :int64) :right t)))
    (check-eq "right=t 值 = np.digitize(..., right=True)"
              '((0 0 0 0) (0 1 1 1) (1 2 2 2)) (vt-to-list r)))
  (let* ((x (vt-reshape (vt-arange 12 :dtype :int64) '(3 4)))
         (r (vt-digitize x '(4 8))))            ; bins 传 Lisp list
    (check-eq "bins 传 Lisp list 可用（原实现报 not-of-type VT）"
              '((0 0 0 0) (1 1 1 1) (2 2 2 2)) (vt-to-list r)))
  (check-eq "bins 传 vector 可用"
            '(0 1 2)
            (vt-to-list (vt-digitize (vt-from-sequence '(0 5 9)) #(4 8))))
  ;; numpy 允许单调递减 bins（内部走 len(bins) - searchsorted(bins[::-1], x, side)）
  (check-eq "单调递减 bins 合法 = np.digitize([0,4,5,8,9],[8,4])"
            '(2 1 1 0 0)
            (vt-to-list (vt-digitize (vt-from-sequence '(0 4 5 8 9)) '(8 4))))
  (check-eq "递减 bins + right=t = np.digitize(...,right=True)"
            '(2 2 2 1 1 0 0)
            (vt-to-list (vt-digitize (vt-from-sequence '(-1 0 4 5 8 9 100)) '(8 4)
                                     :right t)))
  (check-eq "递增 bins + right=t 边界值 = [0 0 1 1 2]"
            '(0 0 1 1 2)
            (vt-to-list (vt-digitize (vt-from-sequence '(-1 2 3 5 6)) '(2 5) :right t)))
  (check-eq "全等 bins 合法（numpy 不报错）= [0 3 3 3]"
            '(0 3 3 3)
            (vt-to-list (vt-digitize (vt-from-sequence '(0 1 3 9)) '(1 1 1))))
  (check-error "非单调 bins → 报错（numpy ValueError）"
               (lambda () (vt-digitize (vt-from-sequence '(1)) '(3 1 2))))

  ;; =========================================================================
)

  (section "vt-insert：axis 路径 moveaxis(values, 0, axis)"

  ;; numpy 实测 a = np.arange(6).reshape(3,2)
  (check-eq "insert(a,1,[7,8,9],axis=1)（标量 obj → moveaxis）"
            '((0 7 1) (2 8 3) (4 9 5))
            (vt-to-list (vt-insert (a32) 1 (vt-from-sequence '(7 8 9) :dtype :int64)
                                   :axis 1)))
  (check-eq "insert(a,[1],[7,8,9],axis=1)（列表 obj → 不 moveaxis，广播）"
            '((0 7 8 9 1) (2 7 8 9 3) (4 7 8 9 5))
            (vt-to-list (vt-insert (a32) '(1) (vt-from-sequence '(7 8 9) :dtype :int64)
                                   :axis 1)))
  (check-eq "insert(a,1,[[7],[8],[9]],axis=1)（标量 obj + 列向量）"
            '((0 7 8 9 1) (2 7 8 9 3) (4 7 8 9 5))
            (vt-to-list (vt-insert (a32) 1 (vt-from-sequence '((7) (8) (9)) :dtype :int64)
                                   :axis 1)))
  (check-eq "insert(a,1,6,axis=1)（标量 values 广播）"
            '((0 6 1) (2 6 3) (4 6 5))
            (vt-to-list (vt-insert (a32) 1 6 :axis 1)))
  (check-eq "insert(a,[0,1],7,axis=1)（多索引 + 标量 values 广播）"
            '((7 0 7 1) (7 2 7 3) (7 4 7 5))
            (vt-to-list (vt-insert (a32) '(0 1) 7 :axis 1)))
  (check-eq "insert(a,[0,2],7,axis=1)"
            '((7 0 1 7) (7 2 3 7) (7 4 5 7))
            (vt-to-list (vt-insert (a32) '(0 2) 7 :axis 1)))
  (check-eq "insert(a,[1],[7,8],axis=1)（列表 obj → 不 moveaxis，行广播）"
            '((0 7 8 1) (2 7 8 3) (4 7 8 5))
            (vt-to-list (vt-insert (a32) '(1) (vt-from-sequence '(7 8) :dtype :int64)
                                   :axis 1)))
  (check-error "insert(a,1,[7,8],axis=1) 不可广播 → 报错（numpy 同样 ValueError）"
               (lambda () (vt-insert (a32) 1 (vt-from-sequence '(7 8) :dtype :int64)
                                     :axis 1)))
  (check-eq "1D insert(arange5,1,[7,8,9],axis=0)"
            '(0 7 8 9 1 2 3 4)
            (vt-to-list (vt-insert (vt-arange 5 :dtype :int64) 1
                                   (vt-from-sequence '(7 8 9) :dtype :int64) :axis 0)))
  ;; 3D
  (check-eq "3D insert(b,1,[[9,9,9,9],[9,9,9,9]],axis=1)"
            '(((0 1 2 3) (9 9 9 9) (4 5 6 7) (8 9 10 11))
              ((12 13 14 15) (9 9 9 9) (16 17 18 19) (20 21 22 23)))
            (vt-to-list (vt-insert (b234) 1
                                   (vt-from-sequence '((9 9 9 9) (9 9 9 9)) :dtype :int64)
                                   :axis 1)))
  (check-eq "3D insert(b,1,9,axis=2)"
            '(((0 9 1 2 3) (4 9 5 6 7) (8 9 9 10 11))
              ((12 9 13 14 15) (16 9 17 18 19) (20 9 21 22 23)))
            (vt-to-list (vt-insert (b234) 1 9 :axis 2)))
  (check-eq "3D insert(b,[1],[[9],[9]],axis=1)（列表 obj 不 moveaxis → 2 行）"
            '(((0 1 2 3) (9 9 9 9) (9 9 9 9) (4 5 6 7) (8 9 10 11))
              ((12 13 14 15) (9 9 9 9) (9 9 9 9) (16 17 18 19) (20 21 22 23)))
            (vt-to-list (vt-insert (b234) '(1)
                                   (vt-from-sequence '((9) (9)) :dtype :int64) :axis 1)))
  (check-error "insert(a,7,M[:,0:2] 形状) 不可广播 → 报错（numpy 同样报错）"
               (lambda () (vt-insert (a32) 7 (vt-narrow (a32) 1 0 2) :axis 1)))
  ;; 原「静默错值」场景：1000x1000，values=(2,1000)，obj=7（标量），axis=1
  (let* ((m (vt-reshape (vt-arange 1000000) '(1000 1000)))
         (r (vt-insert m 7 (vt-narrow m 0 0 2) :axis 1)))
    (check-shape "1000x1000 insert 形状 (1000 1002)" '(1000 1002) r)
    ;; numpy: 插入列 7 = i，列 8 = 1000+i（moveaxis 后的 values）
    (check "列7 = i（i=0,1,5,999）"
           (and (= (vt-ref r 0 7) 0.0d0) (= (vt-ref r 1 7) 1.0d0)
                (= (vt-ref r 5 7) 5.0d0) (= (vt-ref r 999 7) 999.0d0)))
    (check "列8 = 1000+i（原实现误填 2i/2i+1）"
           (and (= (vt-ref r 0 8) 1000.0d0) (= (vt-ref r 1 8) 1001.0d0)
                (= (vt-ref r 5 8) 1005.0d0) (= (vt-ref r 999 8) 1999.0d0)))
    (check "插入点前原列保持（列6 = i*1000+6）"
           (and (= (vt-ref r 0 6) 6.0d0) (= (vt-ref r 999 6) 999006.0d0)))
    (check "插入点后原列右移（列9 = i*1000+7）"
           (and (= (vt-ref r 0 9) 7.0d0) (= (vt-ref r 999 9) 999007.0d0))))
  ;; numpy: np.insert(M,7,M[:,0:1],axis=1) → (1000,2000)，插入列 k 为 1000k
  (let* ((m (vt-reshape (vt-arange 1000000) '(1000 1000)))
         (r (vt-insert m 7 (vt-narrow m 1 0 1) :axis 1)))
    (check-shape "values=M[:,0:1] → 形状 (1000 2000)" '(1000 2000) r)
    (check "插入的 1000 列 = 1000k（列7=0, 列8=1000, 列9=2000，与 i 无关）"
           (and (= (vt-ref r 0 7) 0.0d0) (= (vt-ref r 0 8) 1000.0d0)
                (= (vt-ref r 0 9) 2000.0d0) (= (vt-ref r 7 7) 0.0d0)
                (= (vt-ref r 999 8) 1000.0d0)))
    ;; 插入块占位 7..1006，原列 7 右移到位置 1007（numpy 实测 999007）
    (check "原列 7 右移到位置 1007" (= (vt-ref r 999 1007) 999007.0d0))
    (check "插入块首尾 = 1000k（列7=0, 列1006=999000）"
           (and (= (vt-ref r 999 7) 0.0d0) (= (vt-ref r 999 1006) 999000.0d0))))
  ;; 浮点 positions
  (check-error "insert 浮点 positions 张量 → 报错"
               (lambda () (vt-insert (a32) (vt-from-sequence '(1.0) :dtype :float64) 9 :axis 1)))
  (check-error "insert 浮点 positions 列表 → 报错"
               (lambda () (vt-insert (a32) '(1.0) 9 :axis 1)))

  ;; =========================================================================
)

  (section "回归护栏：既有正确行为不得被本次修改破坏 "

  (check-eq "vt-roll 非零位移数值不变"
            '((3 0 1 2) (7 4 5 6) (11 8 9 10))
            (vt-to-list (vt-roll (vt-reshape (vt-arange 12 :dtype :int64) '(3 4))
                                 1 :axis 1)))
  (check-eq "vt-nonzero 语义不变（每轴一个索引张量）" '((0 2))
            (mapcar #'vt-to-list (vt-nonzero (vt-from-sequence '(1 0 2 0)))))
  (check-eq "vt-argwhere 语义不变" '((0) (2))
            (vt-to-list (vt-argwhere (vt-from-sequence '(1 0 2 0)))))
  (check-eq "vt-diagonal 2D 默认不变" '(0 4 8) (vt-to-list (vt-diagonal (sq33))))
  (check-eq "vt-hsplit 整除情形不变" '((2 3) (2 3))
            (mapcar #'vt-shape (vt-hsplit (vt-zeros '(2 6)) 2)))
  (check-eq "vt-select 全张量形式不变"
            '(1 2 1)
            (vt-to-list (vt-select (list (vt-from-sequence '(1 0 1) :dtype :int64)
                                         (vt-from-sequence '(0 1 0) :dtype :int64))
                                   (list (vt-full '(3) 1 :dtype :int64)
                                         (vt-full '(3) 2 :dtype :int64))
                                   :default 7)))
  (check-eq "vt-pt: 普通连续张量写保护不误伤"
            '(42 1 2 3)
            (let ((v (vt-from-sequence '(0 1 2 3) :dtype :int64)))
              (vt-put v 0 42) (vt-to-list v))))

(format t "~%通过 ~a / 失败 ~a~%" *pass* *fail*)
  (finish-output)
  (zerop *fail*))

(sb-ext:exit :code (if (run) 0 1))
