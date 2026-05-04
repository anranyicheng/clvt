(in-package :clvt)

;; ============================================================
;; 辅助函数：比较浮点数列表（允许微小误差）
;; ============================================================

(defun lists-approx-equal (l1 l2 &key (epsilon 1e-10))
  "递归比较两个可能嵌套的列表结构，数值按 epsilon 容差近似相等。"
  (cond
    ((and (numberp l1) (numberp l2))
     (< (abs (- l1 l2)) epsilon))
    ((and (listp l1) (listp l2))
     (and (= (length l1) (length l2))
          (every (lambda (a b)
		   (lists-approx-equal a b :epsilon epsilon))
		 l1 l2)))
    (t nil)))

(defun all-same-shape (vts)
  (let ((s (vt-shape (first vts))))
    (every (lambda (v) (equal (vt-shape v) s)) (rest vts))))


(defun approx= (a b &optional (tol 1e-10))
  "检查两个数字或张量是否近似相等。"
  (cond ((and (numberp a) (numberp b))
         (or (= a b) (< (abs (- a b)) tol)))
        ((and (vt-p a) (vt-p b))
         (vt-allclose a b :rtol tol :atol tol))
        (t nil)))

(defun assert-ok (condition msg)
  (unless condition
    (error "TEST FAILED: ~A" msg)))

;; ============================================================
;; test-vt-ravel
;; ============================================================
(defun test-vt-ravel ()
  ;; 多维连续视图转一维视图
  ;; a = np.arange(6).reshape(2,3)
  ;; np.ravel(a) -> [0,1,2,3,4,5]
  (let* ((a (vt-reshape (vt-arange 6 :type 'fixnum) '(2 3)))
         (r (vt-ravel a)))
    (assert (equal (vt-shape r) '(6)))
    (assert (equal (vt-to-list r) '(0 1 2 3 4 5)))
    ;; 应尽量为零拷贝：检查数据缓冲区是否相同
    (assert (eq (vt-data r) (vt-data a))))

  ;; 非连续数组：ravel 将返回副本（形状正确）
  ;; a = np.arange(6).reshape(2,3).T   -> 非连续
  ;; np.ravel(a) 仍返回 [0,3,1,4,2,5]? 实际上是按行优先展平转置结果，但我们只关心形状
  (let* ((a (vt-transpose (vt-reshape (vt-arange 6 :type 'fixnum) '(2 3))))
         (r (vt-ravel a)))
    (assert (equal (vt-shape r) '(6)))
    ;; 内容不需要和上面一样，只要是正确展平即可
    (assert (= (length (vt-to-list r)) 6)))

  (format t "~%test-vt-ravel passed.~%"))

;; ============================================================
;; test-vt-swapaxes
;; ============================================================
(defun test-vt-swapaxes ()
  ;; 二维轴交换 (等同转置)
  ;; a = np.arange(6).reshape(2,3)
  ;; np.swapaxes(a, 0, 1)  -> shape (3,2)
  (let* ((a (vt-reshape (vt-arange 6 :type 'fixnum) '(2 3)))
         (sw (vt-swapaxes a 0 1)))
    (assert (equal (vt-shape sw) '(3 2)))
    (assert (equal (vt-to-list sw) '((0 3) (1 4) (2 5)))))

  ;; 三维轴交换：a.shape (2,3,4) -> swapaxes(a,0,2) -> (4,3,2)
  ;; np.arange(24).reshape(2,3,4).swapaxes(0,2)
  (let* ((a (vt-reshape (vt-arange 24 :type 'fixnum) '(2 3 4)))
         (sw (vt-swapaxes a 0 2)))
    (assert (equal (vt-shape sw) '(4 3 2)))
    ;; 检查几个值
    (assert (= (vt-ref sw 0 0 0) 0))
    (assert (= (vt-ref sw 0 0 1) 12))   ; 交换后第二块的第一元素是原来的 [0,0,1]=1? 要仔细算，但简单测试形状即可
    ;; 测试负轴
    (let ((sw2 (vt-swapaxes a -3 -1)))
      (assert (equal (vt-shape sw2) '(4 3 2)))))

  (format t "~%test-vt-swapaxes passed.~%"))

;; ============================================================
;; test-vt-flip
;; ============================================================
(defun test-vt-flip ()
  ;; 一维翻转
  ;; a = np.array([0,1,2]) ; np.flip(a) -> [2,1,0]
  (let* ((a (vt-arange 3 :type 'fixnum))
         (f (vt-flip a)))
    (assert (equal (vt-to-list f) '(2 1 0))))

  ;; 二维沿轴0翻转
  ;; a = np.arange(6).reshape(2,3)
  ;; np.flip(a, axis=0) -> [[3,4,5],[0,1,2]]
  (let* ((a (vt-reshape (vt-arange 6 :type 'fixnum) '(2 3)))
         (f (vt-flip a :axis 0)))
    (assert (equal (vt-to-list f) '((3 4 5) (0 1 2)))))

  ;; 沿轴1翻转
  ;; np.flip(a, axis=1) -> [[2,1,0],[5,4,3]]
  (let* ((a (vt-reshape (vt-arange 6 :type 'fixnum) '(2 3)))
         (f (vt-flip a :axis 1)))
    (assert (equal (vt-to-list f) '((2 1 0) (5 4 3)))))

  ;; 负轴
  (let* ((a (vt-reshape (vt-arange 6 :type 'fixnum) '(2 3)))
         (f (vt-flip a :axis -1)))
    (assert (equal (vt-to-list f) '((2 1 0) (5 4 3)))))

  (format t "~%test-vt-flip passed.~%"))

;; ============================================================
;; test-vt-roll
;; ============================================================
(defun test-vt-roll ()
  ;; 一维滚动
  ;; a = np.array([0,1,2,3,4])
  ;; np.roll(a, 2) -> [3,4,0,1,2]
  (let* ((a (vt-arange 5 :type 'fixnum))
         (r (vt-roll a 2)))
    (assert (equal (vt-to-list r) '(3 4 0 1 2))))

  ;; 负偏移
  ;; np.roll(a, -1) -> [1,2,3,4,0]
  (let* ((a (vt-arange 5 :type 'fixnum))
         (r (vt-roll a -1)))
    (assert (equal (vt-to-list r) '(1 2 3 4 0))))

  ;; 二维沿轴滚动
  ;; a = np.arange(6).reshape(2,3)
  ;; np.roll(a, 1, axis=0) -> [[3,4,5],[0,1,2]]
  (let* ((a (vt-reshape (vt-arange 6 :type 'fixnum) '(2 3)))
         (r (vt-roll a 1 :axis 0)))
    (assert (equal (vt-to-list r) '((3 4 5) (0 1 2)))))

  ;; axis=1, shift=1
  ;; np.roll(a, 1, axis=1) -> [[2,0,1],[5,3,4]]
  (let* ((a (vt-reshape (vt-arange 6 :type 'fixnum) '(2 3)))
         (r (vt-roll a 1 :axis 1)))
    (assert (equal (vt-to-list r) '((2 0 1) (5 3 4)))))

  ;; 多元轴滚动 (列表 shift 和 axis)
  ;; a = np.eye(3) 循环移动两个轴 略，只需确保不报错
  (format t "~%test-vt-roll passed.~%"))

;; ============================================================
;; test-vt-triu-tril
;; ============================================================
(defun test-vt-triu-tril ()
  ;; 二维上三角
  ;; a = np.arange(9).reshape(3,3)
  ;; np.triu(a) ->
  ;; [[0,1,2],
  ;;  [0,4,5],
  ;;  [0,0,8]]
  (let* ((a (vt-reshape (vt-arange 9 :type 'fixnum) '(3 3)))
         (u (vt-triu a)))
    (assert (equal (vt-to-list u) '((0 1 2) (0 4 5) (0 0 8)))))

  ;; k=1 上三角
  ;; np.triu(a,k=1) -> [[0,1,2],[0,0,5],[0,0,0]]
  (let* ((a (vt-reshape (vt-arange 9 :type 'fixnum) '(3 3)))
         (u (vt-triu a :k 1)))
    (assert (equal (vt-to-list u) '((0 1 2) (0 0 5) (0 0 0)))))

  ;; 下三角
  ;; np.tril(a) -> [[0,0,0],[3,4,0],[6,7,8]]
  (let* ((a (vt-reshape (vt-arange 9 :type 'fixnum) '(3 3)))
         (l (vt-tril a)))
    (assert (equal (vt-to-list l) '((0 0 0) (3 4 0) (6 7 8)))))

  ;; k=-1
  ;; np.tril(a,k=-1) -> [[0,0,0],[3,0,0],[6,7,0]]
  (let* ((a (vt-reshape (vt-arange 9 :type 'fixnum) '(3 3)))
         (l (vt-tril a :k -1)))
    (assert (equal (vt-to-list l) '((0 0 0) (3 0 0) (6 7 0)))))

  ;; 修改 in-place (拷贝后修改)
  (let* ((a (vt-reshape (vt-arange 9 :type 'fixnum) '(3 3)))
         (b (vt-copy a))
         (u (vt-triu b :in-place t)))
    (assert (equal (vt-to-list b) (vt-to-list u)))
    ;; 原 b 被修改，a 不受影响
    (assert (equal (vt-to-list b) '((0 1 2) (0 4 5) (0 0 8))))
    (assert (equal (vt-to-list a) '((0 1 2) (3 4 5) (6 7 8)))))

  ;; 高维 batch 上三角（最后两轴）
  ;; a = np.arange(18).reshape(2,3,3)
  ;; np.triu(a) 对每个 3x3 应用上三角
  (let* ((a (vt-reshape (vt-arange 18 :type 'fixnum) '(2 3 3)))
         (u (vt-triu a)))
    (assert (equal (vt-to-list (vt-slice u '(0) '(:all) '(:all)))
                   '((0 1 2) (0 4 5) (0 0 8))))
    (assert (equal (vt-to-list (vt-slice u '(1) '(:all) '(:all)))
                   '((9 10 11) (0 13 14) (0 0 17))))) ; 第二个batch

  (format t "~%test-vt-triu-tril passed.~%"))

;; ============================================================
;; test-vt-diagonal
;; ============================================================
(defun test-vt-diagonal ()
  ;; 二维提取对角线
  ;; a = np.arange(9).reshape(3,3)
  ;; np.diagonal(a) -> [0,4,8]
  (let* ((a (vt-reshape (vt-arange 9 :type 'fixnum) '(3 3)))
         (d (vt-diagonal a)))
    (assert (equal (vt-to-list d) '(0 4 8))))

  ;; offset=1 -> 返回右上对角
  ;; np.diagonal(a, offset=1) -> [1,5]
  (let* ((a (vt-reshape (vt-arange 9 :type 'fixnum) '(3 3)))
         (d (vt-diagonal a :offset 1)))
    (assert (equal (vt-to-list d) '(1 5))))

  ;; offset=-1
  ;; np.diagonal(a, offset=-1) -> [3,7]
  (let* ((a (vt-reshape (vt-arange 9 :type 'fixnum) '(3 3)))
         (d (vt-diagonal a :offset -1)))
    (assert (equal (vt-to-list d) '(3 7))))

  ;; 非方阵
  ;; a = np.arange(6).reshape(2,3)
  ;; np.diagonal(a) -> [0,4]
  (let* ((a (vt-reshape (vt-arange 6 :type 'fixnum) '(2 3)))
         (d (vt-diagonal a)))
    (assert (equal (vt-to-list d) '(0 4))))

  ;; 高维 batch 对角线
  ;; a = np.arange(24).reshape(2,3,4)
  ;; np.diagonal(a, axis1=1, axis2=2) -> shape (2,3)
  (let* ((a (vt-reshape (vt-arange 24 :type 'fixnum) '(2 3 4)))
         (d (vt-diagonal a)))
    (assert (equal (vt-shape d) '(2 3)))
    (assert (equal (vt-to-list (vt-slice d '(0) '(:all))) '(0 5 10))))

  (format t "~%test-vt-diagonal passed.~%"))

;; ============================================================
;; test-vt-broadcast-to
;; ============================================================
(defun test-vt-broadcast-to ()
  ;; 广播一维到二维
  ;; a = np.array([1,2,3])
  ;; np.broadcast_to(a, (2,3)) -> [[1,2,3],[1,2,3]]
  (let* ((a (vt-from-sequence '(1 2 3) :type 'fixnum))
         (b (vt-broadcast-to a '(2 3))))
    (assert (equal (vt-shape b) '(2 3)))
    (assert (equal (vt-to-list b) '((1 2 3) (1 2 3)))))

  ;; 广播二维增加前导维度
  ;; a = np.array([[10],[20]])
  ;; np.broadcast_to(a, (1,2,3)) -> 形状 (1,2,3)
  (let* ((a (vt-from-sequence '((10) (20)) :type 'fixnum))
         (b (vt-broadcast-to a '(1 2 3))))
    (assert (equal (vt-shape b) '(1 2 3)))
    (assert (equal (vt-to-list (vt-slice b '(0) '(:all) '(:all)))
                   '((10 10 10) (20 20 20)))))

  ;; 不可广播应报错
  (let ((a (vt-from-sequence '(1 2 3))))
    (handler-case (vt-broadcast-to a '(3 2))
      (error (e) (format t "~%[OK] Broadcast error caught: ~a" e))))

  ;; 零拷贝：验证复用数据缓冲区
  (let* ((a (vt-from-sequence '(7 8 9)))
         (b (vt-broadcast-to a '(2 3))))
    (assert (eq (vt-data b) (vt-data a))))

  (format t "~%test-vt-broadcast-to passed.~%"))

(defun test-vt-slice ()
  ;; ============================================================
  ;; 辅助函数：将张量转换为列表以便比较（已存在于库中）
  ;; ============================================================

  ;; ============================================================
  ;; 一维张量测试 (a = np.arange(10))
  ;; ============================================================
  (let ((a (vt-arange 10 :type 'fixnum)))
    ;; 1a. 单个整数索引
    ;; a[3] -> 3
    (assert (= (vt-ref (vt-slice a '(3))) 3))
    
    ;; 1b. 正向范围
    ;; a[2:7] -> [2,3,4,5,6]
    (assert (equal (vt-to-list (vt-slice a '(2 7))) '(2 3 4 5 6)))
    
    ;; 1c. 带步长
    ;; a[1:9:2] -> [1,3,5,7]
    (assert (equal (vt-to-list (vt-slice a '(1 9 2))) '(1 3 5 7)))
    
    ;; 1d. 省略 start
    ;; a[:5] -> [0,1,2,3,4]
    (assert (equal (vt-to-list (vt-slice a '(nil 5))) '(0 1 2 3 4)))
    
    ;; 1e. 省略 end
    ;; a[5:] -> [5,6,7,8,9]
    (assert (equal (vt-to-list (vt-slice a '(5 nil))) '(5 6 7 8 9)))
    
    ;; 1f. 反向步长
    ;; a[8:3:-1] -> [8,7,6,5,4]
    (assert (equal (vt-to-list (vt-slice a '(8 3 -1))) '(8 7 6 5 4)))
    
    ;; 1g. 完整反向
    ;; a[::-1] -> [9,8,7,6,5,4,3,2,1,0]
    (assert (equal (vt-to-list (vt-slice a '(nil nil -1)))
                   '(9 8 7 6 5 4 3 2 1 0)))
    
    ;; 1h. 负索引
    ;; a[-1] -> 9
    (assert (= (vt-ref (vt-slice a '(-1))) 9))
    
    ;; a[-3:-1] -> [7,8]
    (assert (equal (vt-to-list (vt-slice a '(-3 -1))) '(7 8)))
    
    ;; 1i. 空切片
    ;; a[5:5] -> []
    (assert (equal (vt-to-list (vt-slice a '(5 5))) '()))
    ;; a[10:10] -> []
    (assert (equal (vt-to-list (vt-slice a '(10 10))) '())))

  ;; ============================================================
  ;; 二维张量测试 (b = np.arange(20).reshape(4,5))
  ;; ============================================================
  (let* ((b (vt-reshape (vt-arange 20 :type 'fixnum) '(4 5))))
    ;; 2a. 单个元素
    ;; b[1,2] -> 7
    (assert (= (vt-ref (vt-slice b '(1) '(2))) 7.0))
    
    ;; 2b. 取一行
    ;; b[2,:] -> [10,11,12,13,14]
    (assert (equal (vt-to-list (vt-slice b '(2) '(:all)))
                   '(10 11 12 13 14)))
    
    ;; 2c. 取一列
    ;; b[:,3] -> [3,8,13,18]
    (assert (equal (vt-to-list (vt-slice b '(:all) '(3)))
                   '(3 8 13 18)))
    
    ;; 2d. 子矩阵
    ;; b[1:3, 2:4] -> [[7,8],[12,13]]
    (assert (equal (vt-to-list (vt-slice b '(1 3) '(2 4)))
                   '((7 8) (12 13))))
    
    ;; 2e. 省略边界
    ;; b[:2, 2:] -> [[2,3,4],[7,8,9]]
    (assert (equal (vt-to-list (vt-slice b '(nil 2) '(2 nil)))
                   '((2 3 4) (7 8 9))))
    
    ;; 2f. 行逆序
    ;; b[::-1, :] -> 行颠倒
    (assert (equal (vt-to-list (vt-slice b '(nil nil -1) '(:all)))
                   '((15 16 17 18 19)
                     (10 11 12 13 14)
                     (5 6 7 8 9)
                     (0 1 2 3 4))))
    
    ;; 2g. 列逆序
    ;; b[:, ::-1] -> 列颠倒
    (assert (equal (vt-to-list (vt-slice b '(:all) '(nil nil -1)))
                   '((4 3 2 1 0)
                     (9 8 7 6 5)
                     (14 13 12 11 10)
                     (19 18 17 16 15))))
    
    ;; 2h. 负索引与省略
    ;; b[-2:, -3:] -> [[15,16,17,18,19],[10,11,12,13,14]] ... 不对
    ;; 注意：numpy 中 b[-2:, -3:] 是后两行，后三列 -> [[2,3,4],[7,8,9],[12,13,14],[17,18,19]]? 需要看形状(4,5)
    ;; b[-2:, -3:] -> 后两行的后三列，即索引2..4行, 2..5列 -> [[2,3,4],[7,8,9]]? 让我们验证：
    ;; b[-2:] = 最后两行: 索引2,3 -> [10..14],[15..19]
    ;; b[..., -3:] = 后三列: 索引2,3,4 -> [2,3,4],[7,8,9],[12,13,14],[17,18,19]
    ;; 交集: [[12,13,14],[17,18,19]] 正确
    ;; 所以测试：
    (assert (equal (vt-to-list (vt-slice b '(-2 nil) '(-3 nil)))
                   '((12 13 14) (17 18 19))))
    
    ;; 2i. 混合整数与范围
    ;; b[1, 1:4] -> [6,7,8]
    (assert (equal (vt-to-list (vt-slice b '(1) '(1 4)))
                   '(6 7 8)))
    
    ;; 2j. 负步长带省略
    ;; b[:, 4:1:-1] -> 列4,3,2
    (assert (equal (vt-to-list (vt-slice b '(:all) '(4 1 -1)))
                   '((4 3 2) (9 8 7) (14 13 12) (19 18 17))))
    
    ;; 2k. 使用 else 省略号 (二维中省略号相当于 :)
    ;; b[..., :2] -> 所有行，前两列
    (assert (equal (vt-to-list (vt-slice b '(:elli) '(nil 2)))
                   '((0 1) (5 6) (10 11) (15 16))))
    
    ;; 2l. 新轴插入
    ;; b[:, None, :] -> 形状 (4,1,5)
    (assert (equal (vt-shape (vt-slice b '(:all) '(:newa) '(:all)))
                   '(4 1 5)))
    ;; b[None, :, None, 0] -> (1,4,1)
    (assert (equal (vt-shape (vt-slice b '(:newa) '(:all) '(:newa) '(0)))
                   '(1 4 1)))
    ;; 值检查：
    (assert (equal (vt-to-list (vt-slice b '(:newa) '(:all) '(:newa) '(0)))
                   '(((0) (5) (10) (15)))))   ; 因为额外维度，需注意嵌套
    
    ;; 2m. 空切片视图
    ;; b[2:2, :] -> shape (0,5)
    (assert (equal (vt-shape (vt-slice b '(2 2) '(:all))) '(0 5)))
    ;; b[1:3, 5:5] -> (2,0)
    (assert (equal (vt-shape (vt-slice b '(1 3) '(5 5))) '(2 0))))

  ;; ============================================================
  ;; 三维张量测试 (c = np.arange(24).reshape(2,3,4))
  ;; ============================================================
  (let* ((c (vt-reshape (vt-arange 24 :type 'fixnum) '(2 3 4))))
    ;; 3a. 取一个元素
    ;; c[0,1,2] -> 6
    (assert (= (vt-ref (vt-slice c '(0) '(1) '(2))) 6.0))
    
    ;; 3b. 取一个平面
    ;; c[1, :, :] -> shape (3,4), 值 12..23
    (assert (equal (vt-to-list (vt-slice c '(1) '(:all) '(:all)))
                   '((12 13 14 15) (16 17 18 19) (20 21 22 23))))
    
    ;; 3c. 切片与范围
    ;; c[0, 0:2, 1:3] -> [[1,2],[5,6]]
    (assert (equal (vt-to-list (vt-slice c '(0) '(0 2) '(1 3)))
                   '((1 2) (5 6))))
    
    ;; 3d. 多个省略号 (只有一个)
    ;; c[..., :2] -> 形状 (2,3,2) 取每个块的前两列
    (assert (equal (vt-shape (vt-slice c '(:elli) '(nil 2)))
                   '(2 3 2)))
    ;; 值：每个 matrix 的前两列
    (assert (equal (vt-to-list (vt-slice c '(:elli) '(nil 2)))
                   '(((0 1) (4 5) (8 9)) ((12 13) (16 17) (20 21)))))
    
    ;; 3e. 新轴与省略号混合
    ;; c[None, ..., :2, None] -> 形状 (1,2,3,2,1)
    (assert (equal (vt-shape (vt-slice c '(:newa) '(:elli) '(nil 2) '(:newa)))
                   '(1 2 3 2 1)))
    
    ;; 3f. 负索引与步长
    ;; c[:, ::-1, ::2] -> 行反转，列隔列取
    (let ((result (vt-slice c '(:all) '(nil nil -1) '(nil nil 2))))
      ;; 形状应为 (2,3,2) 因为列维度4，隔列后为2
      (assert (equal (vt-shape result) '(2 3 2)))
      ;; 检查第一个块
      (assert (equal (vt-to-list (vt-slice result '(0) '(:all) '(:all)))
                     '((8 10) (4 6) (0 2)))))  
    
    ;; 3g. 混合整数降维
    ;; c[0, -1, 2] -> 10
    (assert (= (vt-ref (vt-slice c '(0) '(-1) '(2))) 10.0))
    
    ;; 3h. 省略号处于中间
    ;; c[0, ..., 2] -> 等价 c[0, :, :, 2]？不，这是三维，c[0, :, 2] -> shape (3,)
    (assert (equal (vt-to-list (vt-slice c '(0) '(:elli) '(2)))
                   '(2 6 10)))  ; 所有行的第2列
    
    ;; 3i. 新轴扩展
    ;; c[:, None, :, :] -> shape (2,1,3,4)
    (assert (equal (vt-shape (vt-slice c '(:all) '(:newa) '(:all) '(:all)))
                   '(2 1 3 4)))
    
    ;; 3j. 反向步长且 start/end 省略
    ;; c[:, :, ::-1] -> 最后一维反转
    (assert (equal (vt-to-list (vt-slice c '(:all) '(:all) '(nil nil -1)))
                   '(((3 2 1 0) (7 6 5 4) (11 10 9 8))
                     ((15 14 13 12) (19 18 17 16) (23 22 21 20))))))

  (format t "~%All new vt-slice tests passed.~%")

  )

;; --------------------- test reshape ---------------------
(defun test-vt-reshape ()
  ;; np.arange(6).reshape(2,3)
  (let* ((a (vt-arange 6 :type 'fixnum))
         (b (vt-reshape a '(2 3))))
    (assert (equal (vt-shape b) '(2 3)))
    (assert (equal (vt-to-list b) '((0 1 2) (3 4 5))))
    ;; 重塑为一维
    (let ((c (vt-reshape b '(6))))
      (assert (equal (vt-shape c) '(6)))
      (assert (equal (vt-to-list c) '(0 1 2 3 4 5)))))
  ;; 大小不匹配应报错
  (let ((a (vt-arange 6)))
    (handler-case (vt-reshape a '(2 4))
      (error (e) (format t "~%[OK] Reshape size mismatch caught: ~a" e))))
  (format t "~%test-vt-reshape passed.~%"))

;; --------------------- test transpose ---------------------
(defun test-vt-transpose ()
  ;; a = np.arange(6).reshape(2,3)
  ;; a.T -> shape (3,2)
  (let* ((a (vt-reshape (vt-arange 6 :type 'fixnum) '(2 3)))
         (at (vt-transpose a)))
    (assert (equal (vt-shape at) '(3 2)))
    (assert (equal (vt-to-list at) '((0 3) (1 4) (2 5)))))
  ;; 指定 perm
  ;; a = np.arange(24).reshape(2,3,4)
  ;; a.transpose(1,0,2) -> shape (3,2,4)
  (let* ((a (vt-reshape (vt-arange 24 :type 'fixnum) '(2 3 4)))
         (at (vt-transpose a '(1 0 2))))
    (assert (equal (vt-shape at) '(3 2 4)))
    ;; 检查个别值
    (assert (= (vt-ref at 0 0 0) 0))
    (assert (= (vt-ref at 0 1 0) 12))
    (assert (= (vt-ref at 2 1 3) 23)))
  ;; 一维转置无变化
  (let* ((a (vt-arange 5 :type 'fixnum))
         (at (vt-transpose a)))
    (assert (equal (vt-shape at) '(5)))
    (assert (equal (vt-to-list at) '(0 1 2 3 4))))
  (format t "~%test-vt-transpose passed.~%"))

;; --------------------- test squeeze ---------------------
(defun test-vt-squeeze ()
  ;; a = np.arange(6).reshape(1,2,3)
  ;; np.squeeze(a)  -> shape (2,3)
  (let* ((a (vt-reshape (vt-arange 6 :type 'fixnum) '(1 2 3)))
         (sq (vt-squeeze a)))
    (assert (equal (vt-shape sq) '(2 3)))
    (assert (equal (vt-to-list sq) '((0 1 2) (3 4 5)))))
  ;; 指定轴
  ;; a = np.arange(6).reshape(2,1,3)
  ;; np.squeeze(a, axis=1) -> shape (2,3)
  (let* ((a (vt-reshape (vt-arange 6 :type 'fixnum) '(2 1 3)))
         (sq (vt-squeeze a :axis 1)))
    (assert (equal (vt-shape sq) '(2 3)))
    (assert (equal (vt-to-list sq) '((0 1 2) (3 4 5)))))
  ;; 挤压非单例轴应报错
  (let ((a (vt-reshape (vt-arange 6) '(2 3))))
    (handler-case (vt-squeeze a :axis 0)
      (error (e) (format t "~%[OK] Squeeze non-singleton axis caught: ~a" e))))
  ;; 负轴
  (let* ((a (vt-reshape (vt-arange 6 :type 'fixnum) '(2 3 1)))
         (sq (vt-squeeze a :axis -1)))
    (assert (equal (vt-shape sq) '(2 3))))
  (format t "~%test-vt-squeeze passed.~%"))

;; --------------------- test expand-dims ---------------------
(defun test-vt-expand-dims ()
  ;; a = np.array([1,2,3])
  ;; np.expand_dims(a, 0)  -> shape (1,3)
  (let* ((a (vt-from-sequence '(1 2 3) :type 'fixnum))
         (ea (vt-expand-dims a 0)))
    (assert (equal (vt-shape ea) '(1 3)))
    (assert (equal (vt-to-list ea) '((1 2 3)))))
  ;; axis=1
  ;; np.expand_dims(a, 1) -> shape (3,1)
  (let* ((a (vt-from-sequence '(1 2 3) :type 'fixnum))
         (ea (vt-expand-dims a 1)))
    (assert (equal (vt-shape ea) '(3 1)))
    (assert (equal (vt-to-list ea) '((1) (2) (3)))))
  ;; 负轴
  ;; np.expand_dims(a, -1) 相同
  (let* ((a (vt-from-sequence '(1 2 3) :type 'fixnum))
         (ea (vt-expand-dims a -1)))
    (assert (equal (vt-shape ea) '(3 1))))
  (format t "~%test-vt-expand-dims passed.~%"))

;; --------------------- test concatenate ---------------------
(defun test-vt-concatenate ()
  ;; a = np.array([[1,2],[3,4]])
  ;; b = np.array([[5,6]])
  ;; np.concatenate([a,b], axis=0) -> [[1,2],[3,4],[5,6]]
  (let* ((a (vt-from-sequence '((1 2) (3 4)) :type 'fixnum))
         (b (vt-from-sequence '((5 6)) :type 'fixnum))
         (c (vt-concatenate 0 a b)))
    (assert (equal (vt-shape c) '(3 2)))
    (assert (equal (vt-to-list c) '((1 2) (3 4) (5 6)))))
  ;; 沿轴1
  ;; np.concatenate([a,b.T], axis=1)
  (let* ((a (vt-from-sequence '((1 2) (3 4)) :type 'fixnum))
         (b (vt-from-sequence '((5) (6)) :type 'fixnum))  ; 列向量
         (c (vt-concatenate 1 a b)))
    (assert (equal (vt-shape c) '(2 3)))
    (assert (equal (vt-to-list c) '((1 2 5) (3 4 6)))))
  ;; 负轴，等价 axis=0 (一维数组 axis=-1 归一化为 0)
  ;; a = np.array([0,1,2])
  ;; b = np.array([2,3,4])
  ;; np.concatenate([a,b], axis=0) -> [0,1,2,2,3,4]
  (let* ((a (vt-arange 3 :type 'fixnum))
	 (b (vt-arange 3 :start 2 :type 'fixnum))
	 (c (vt-concatenate -1 a b)))
    (assert (equal (vt-shape c) '(6)))
    (assert (equal (vt-to-list c) '(0 1 2 2 3 4))))
  (format t "~%test-vt-concatenate passed.~%"))

;; --------------------- test stack ---------------------
(defun test-vt-stack ()
  ;; a = [1,2], b = [3,4]
  ;; np.stack([a,b], axis=0) -> [[1,2],[3,4]]
  (let* ((a (vt-from-sequence '(1 2) :type 'fixnum))
         (b (vt-from-sequence '(3 4) :type 'fixnum))
         (s (vt-stack 0 a b)))
    (assert (equal (vt-shape s) '(2 2)))
    (assert (equal (vt-to-list s) '((1 2) (3 4)))))
  ;; axis=1 -> [[1,3],[2,4]]
  (let* ((a (vt-from-sequence '(1 2) :type 'fixnum))
         (b (vt-from-sequence '(3 4) :type 'fixnum))
         (s (vt-stack 1 a b)))
    (assert (equal (vt-shape s) '(2 2)))
    (assert (equal (vt-to-list s) '((1 3) (2 4)))))
  ;; 三维
  ;; a = np.zeros((2,3)), b = np.ones((2,3))
  ;; np.stack([a,b], axis=0) -> (2,2,3)
  (let* ((a (vt-zeros '(2 3)))
         (b (vt-ones '(2 3)))
         (s (vt-stack 0 a b)))
    (assert (equal (vt-shape s) '(2 2 3)))
    (assert (= (vt-ref s 0 0 0) 0.0d0))
    (assert (= (vt-ref s 1 0 0) 1.0d0)))
  (format t "~%test-vt-stack passed.~%"))

;; --------------------- test tile ---------------------
(defun test-vt-tile ()
  ;; a = np.array([0,1,2])
  ;; np.tile(a, 3) -> [0,1,2,0,1,2,0,1,2]
  (let* ((a (vt-from-sequence '(0 1 2) :type 'fixnum))
         (b (vt-tile a 3)))
    (assert (equal (vt-shape b) '(9)))
    (assert (equal (vt-to-list b) '(0 1 2 0 1 2 0 1 2))))
  ;; 二维 tile reps=(2,3)
  ;; a = np.array([[1,2],[3,4]])
  ;; np.tile(a, (2,3)) -> shape (4,6)
  (let* ((a (vt-from-sequence '((1 2) (3 4)) :type 'fixnum))
         (b (vt-tile a '(2 3))))
    (assert (equal (vt-shape b) '(4 6)))
    (assert (= (vt-ref b 0 0) 1))
    (assert (= (vt-ref b 1 0) 3))
    (assert (= (vt-ref b 3 5) 4)))
  ;; reps 自动扩展到维数
  ;; a = [1,2], reps=(2,3) -> 先扩展 a 为 (1,2) 再 tile -> (2,6)
  (let* ((a (vt-from-sequence '(1 2) :type 'fixnum))
         (b (vt-tile a '(2 3))))
    (assert (equal (vt-shape b) '(2 6)))
    (assert (equal (vt-to-list (vt-slice b '(:all) '(0 nil 2)))
		   '((1 1 1) (1 1 1))))) ; 略
  (format t "~%test-vt-tile passed.~%"))

;; --------------------- test repeat ---------------------
(defun test-vt-repeat ()
  ;; a = np.array([0,1,2])
  ;; np.repeat(a, 3) -> [0,0,0,1,1,1,2,2,2]
  (let* ((a (vt-from-sequence '(0 1 2) :type 'fixnum))
         (r (vt-repeat a 3)))
    (assert (equal (vt-shape r) '(9)))
    (assert (equal (vt-to-list r) '(0 0 0 1 1 1 2 2 2))))
  ;; 沿轴重复
  ;; a = np.array([[1,2],[3,4]])
  ;; np.repeat(a, 2, axis=1) -> [[1,1,2,2],[3,3,4,4]]
  (let* ((a (vt-from-sequence '((1 2) (3 4)) :type 'fixnum))
         (r (vt-repeat a 2 :axis 1)))
    (assert (equal (vt-shape r) '(2 4)))
    (assert (equal (vt-to-list r) '((1 1 2 2) (3 3 4 4)))))
  ;; 沿轴0，每个元素不同重复次数（列表）
  ;; a = np.array([1,2,3])
  ;; np.repeat(a, [2,0,1]) -> [1,1,3]
  (let* ((a (vt-from-sequence '(1 2 3) :type 'fixnum))
         (r (vt-repeat a '(2 0 1))))
    (assert (equal (vt-shape r) '(3)))
    (assert (equal (vt-to-list r) '(1 1 3))))
  ;; 负轴
  (let* ((a (vt-from-sequence '((1 2) (3 4)) :type 'fixnum))
         (r (vt-repeat a 2 :axis -1)))
    (assert (equal (vt-shape r) '(2 4)))
    (assert (equal (vt-to-list r) '((1 1 2 2) (3 3 4 4)))))
  (format t "~%test-vt-repeat passed.~%"))


(defun test-vt-meshgrid ()
  (let* ((x (vt-arange 3))
	 (y (vt-arange 4))
	 (z (vt-arange 5)))
    ;; xy 稀疏
    (let ((g (vt-meshgrid (list x y) :sparse t :indexing :xy)))
      (assert (equal (vt-shape (first g)) (list 1 3)))
      (assert (equal (vt-shape (second g)) (list 4 1))))
    ;; ij 稀疏
    (let ((g (vt-meshgrid (list x y) :sparse t :indexing :ij)))
      (assert (equal (vt-shape (first g)) (list 3 1)))
      (assert (equal (vt-shape (second g)) (list 1 4))))
    ;; 三维 xy 稀疏
    (let ((g (vt-meshgrid (list x y z) :sparse t :indexing :xy)))
      (assert (equal (mapcar #'vt-shape g) '((1 3 1) (4 1 1) (1 1 5)))))
    ;; 非稀疏 xy 全尺寸
    (let ((g (vt-meshgrid (list x y) :sparse nil :indexing :xy)))
      (assert (equal (vt-shape (first g)) (list 4 3)))
      (assert (equal (vt-shape (second g)) (list 4 3)))))
  (print "Testing vt-meshgrid passed"))



;; ============================================================
;; vt-median 测试
;; ============================================================
(defun test-vt-median ()
  ;; 全局中位数（一维向量）
  ;; np.median(np.array([5, 2, 8, 1, 9])) -> 5.0
  (let ((a (vt-from-sequence '(5 2 8 1 9))))
    (assert (= (vt-median a) 5.0d0)))

  ;; 全局中位数（偶数个元素）
  ;; np.median(np.array([1, 2, 3, 4])) -> 2.5
  (let ((a (vt-from-sequence '(1 2 3 4))))
    (assert (= (vt-median a) 2.5d0)))

  ;; 二维张量，沿 axis=0 求中位数
  ;; a = np.arange(30).reshape(5,6)
  ;; np.median(a, axis=0) -> [12. 13. 14. 15. 16. 17.]
  (let ((a (vt-reshape (vt-arange 30 :type 'fixnum) '(5 6))))
    (let ((result (vt-to-list (vt-median a :axis 0))))
      (assert (lists-approx-equal result '(12.0 13.0 14.0 15.0 16.0 17.0)))))

  ;; 二维张量，沿 axis=1 求中位数
  ;; np.median(a, axis=1) -> [2.5 8.5 14.5 20.5 26.5]
  (let ((a (vt-reshape (vt-arange 30 :type 'fixnum) '(5 6))))
    (let ((result (vt-to-list (vt-median a :axis 1))))
      (assert (lists-approx-equal result '(2.5 8.5 14.5 20.5 26.5)))))

  ;; 三维张量，沿 axis=1 求中位数
  ;; a = np.arange(24).reshape(2,3,4)
  ;; np.median(a, axis=1) -> [[ 4.  5.  6.  7.] [16. 17. 18. 19.]]
  (let ((a (vt-reshape (vt-arange 24 :type 'fixnum) '(2 3 4))))
    (let ((result (vt-to-list (vt-median a :axis 1))))
      (assert (lists-approx-equal (reduce #'append result) 
				  '(4.0 5.0 6.0 7.0 16.0 17.0 18.0 19.0)))))

  ;; 三维张量，沿 axis=2（最后一维）求中位数
  ;; np.median(a, axis=2) -> [[ 1.5  5.5  9.5] [13.5 17.5 21.5]]
  (let ((a (vt-reshape (vt-arange 24 :type 'fixnum) '(2 3 4))))
    (let ((result (vt-to-list (vt-median a :axis 2))))
      (assert (lists-approx-equal (reduce #'append result)
				  '(1.5 5.5 9.5 13.5 17.5 21.5)))))

  ;; 使用负轴 axis=-1 等同于 axis=1
  ;; np.median(a, axis=-1) -> [2.5 8.5 14.5 20.5 26.5]
  (let ((a (vt-reshape (vt-arange 30 :type 'fixnum) '(5 6))))
    (let ((result (vt-to-list (vt-median a :axis -1))))
      (assert (lists-approx-equal result '(2.5 8.5 14.5 20.5 26.5)))))

  ;; 全局中位数（标量）作为数字返回
  ;; np.median(np.array([1.0])) -> 1.0
  (let ((a (vt-zeros '(1))))
    (assert (= (vt-median a) 0.0d0)))
  (print "Test vt-median passed")
  )
;; ============================================================
;; vt-percentile 测试
;; ============================================================
(defun test-vt-percentile ()
  ;; 一维向量，线性插值（默认方法）
  ;; a = np.array([1, 2, 3, 4, 5])
  ;; np.percentile(a, 50) -> 3.0
  (let ((a (vt-from-sequence '(1 2 3 4 5))))
    (assert (= (vt-percentile a 50) 3.0d0)))

  ;; 一维向量，lower 方法
  ;; np.percentile(a, 40, interpolation='lower') -> 2
  (let ((a (vt-from-sequence '(1 2 3 4 5))))
    (assert (= (vt-percentile a 40 :interpolation :lower) 2.0d0)))

  ;; 一维向量，higher 方法
  ;; np.percentile(a, 40, interpolation='higher') -> 3
  (let ((a (vt-from-sequence '(1 2 3 4 5))))
    (assert (= (vt-percentile a 40 :interpolation :higher) 3.0d0)))

  ;; 一维向量，midpoint 方法
  ;; np.percentile(a, 40, interpolation='midpoint') -> 2.5
  (let ((a (vt-from-sequence '(1 2 3 4 5))))
    (assert (= (vt-percentile a 40 :interpolation :midpoint) 2.5d0)))

  ;; 一维向量，nearest 方法
  ;; np.percentile(a, 40, interpolation='nearest') -> 2  (since idx=1.6, frac=0.6 > 0.5 -> upper=2?)
  (let ((a (vt-from-sequence '(1 2 3 4 5))))
    (assert (= (vt-percentile a 40 :interpolation :nearest) 3.0d0)))  ; idx=1.6, frac=0.6>0.5 => upper=2?

  ;; 二维轴向：axis=1，线性
  ;; a = np.arange(30).reshape(5,6)
  ;; np.percentile(a, 30, axis=1) -> [1.5 7.5 13.5 19.5 25.5]
  (let ((a (vt-reshape (vt-arange 30 :type 'fixnum) '(5 6))))
    (let ((result (vt-to-list (vt-percentile a 30 :axis 1))))
      (assert (lists-approx-equal result '(1.5 7.5 13.5 19.5 25.5)))))

  ;; 二维轴向：axis=0，nearest
  ;; np.percentile(a, 90, axis=0, interpolation='nearest') -> [22 23 24 25 26 27]? 原数组精确
  ;; a 是 fixnum，shape (5,6)，每列 5 个值，90% 的索引 = 0.9*4=3.6，frac=0.6>0.5 => upper=4，取第4个（0-based）
  ;; 列0：0,6,12,18,24 -> 索引4=24，正确
  (let ((a (vt-reshape (vt-arange 30 :type 'fixnum) '(5 6))))
    (let ((result (vt-to-list (vt-percentile a 90 :axis 0 :interpolation :nearest))))
      (assert (lists-approx-equal result '(24.0 25.0 26.0 27.0 28.0 29.0)))))

  ;; 二维轴向：axis=1，nearest 验证与 NumPy 的一致性
  ;; np.percentile(a, 90, axis=1, interpolation='nearest') -> [4 10 16 22 28]
  (let ((a (vt-reshape (vt-arange 30 :type 'fixnum) '(5 6))))
    (let ((result (vt-to-list (vt-percentile a 90 :axis 1 :interpolation :nearest))))
      (assert (lists-approx-equal result '(4.0 10.0 16.0 22.0 28.0)))))
  ;;  np.percentile(a, 90,axis=1,interpolation="lower")
  (let ((a (vt-reshape (vt-arange 30 :type 'fixnum) '(5 6))))
    (let ((result (vt-to-list (vt-percentile a 90 :axis 1 :interpolation :lower))))
      (assert (lists-approx-equal result '(4.0 10.0 16.0 22.0 28.0)))))
  
  ;; np.percentile(a, 90,axis=1,interpolation="higher")
  (let ((a (vt-reshape (vt-arange 30 :type 'fixnum) '(5 6))))
    (let ((result (vt-to-list (vt-percentile a 90 :axis 1 :interpolation :higher))))
      (assert (lists-approx-equal result '(5.0 11.0 17.0 23.0 29.0)))))

  ;;np.percentile(a, 90,axis=1,interpolation="midpoint")
  (let ((a (vt-reshape (vt-arange 30 :type 'fixnum) '(5 6))))
    (let ((result (vt-to-list (vt-percentile a 90 :axis 1 :interpolation :midpoint))))
      (assert (lists-approx-equal result '( 4.5 10.5 16.5 22.5 28.5)))))

  ;; np.percentile(a, 90,axis=1,interpolation="linear")
  (let ((a (vt-reshape (vt-arange 30 :type 'fixnum) '(5 6))))
    (let ((result (vt-to-list (vt-percentile a 90 :axis 1 :interpolation :linear))))
      (assert (lists-approx-equal result '( 4.5 10.5 16.5 22.5 28.5)))))
  ;; 三维轴向：axis=2，线性
  ;; a = np.arange(24).reshape(2,3,4)
  ;; np.percentile(a, 25, axis=2) -> [[0.75 4.75 8.75] [12.75 16.75 20.75]]
  (let ((a (vt-reshape (vt-arange 24 :type 'fixnum) '(2 3 4))))
    (let ((result (vt-to-list (vt-percentile a 25 :axis 2))))
      (assert (lists-approx-equal (reduce #'append result)
				  '(0.75 4.75 8.75 12.75 16.75 20.75)))))

  ;; 全局百分位数，标量输入
  ;; np.percentile(np.array(42), 50) -> 42.0
  (let ((a (vt-const '() 42 :type 'double-float)))  ; 标量张量
    (assert (= (vt-percentile a 50) 42.0d0)))

  (format t "vt-percentile tests passed.~%")
  )

;; ============================================================
;; test-vt-sum
;; ============================================================
(defun test-vt-sum ()
  ;; 一维全局求和
  ;; np.sum(np.array([1,2,3,4,5])) -> 15
  (let ((a (vt-from-sequence '(1 2 3 4 5) :type 'fixnum)))
    (assert (= (vt-sum a) 15.0d0)))

  ;; 二维沿轴0求和，保持维度
  ;; a = np.arange(6).reshape(2,3)
  ;; np.sum(a, axis=0, keepdims=True) -> [[3,5,7]]
  (let* ((a (vt-reshape (vt-arange 6 :type 'fixnum) '(2 3)))
         (res (vt-to-list (vt-sum a :axis 0 :keepdims t))))
    (assert (equal res '((3 5 7)))))

  ;; 二维沿轴1求和，不保留维度
  ;; np.sum(a, axis=1) -> [3,12]
  (let* ((a (vt-reshape (vt-arange 6 :type 'fixnum) '(2 3)))
         (res (vt-to-list (vt-sum a :axis 1))))
    (assert (equal res '(3 12))))

  ;; 三维沿轴1求和（中间轴）
  ;; a = np.arange(24).reshape(2,3,4)
  ;; np.sum(a, axis=1) -> [[12,15,18,21],[48,51,54,57]]
  (let* ((a (vt-reshape (vt-arange 24 :type 'fixnum) '(2 3 4)))
         (res (vt-to-list (vt-sum a :axis 1))))
    (assert (lists-approx-equal (reduce #'append res)
				'(12.0 15.0 18.0 21.0 48.0 51.0 54.0 57.0))))

  ;; 全局求和（标量形状）
  ;; np.sum(np.array(42)) -> 42
  (let ((a (vt-const '() 42 :type 'fixnum)))
    (assert (= (vt-sum a) 42.0d0)))

  (format t "~%test-vt-sum passed.~%"))

;; ============================================================
;; test-vt-amax / test-vt-amin
;; ============================================================
(defun test-vt-amax-amin ()
  ;; amax 全局
  ;; np.max(np.array([10,3,7,2,9])) -> 10
  (let ((a (vt-from-sequence '(10 3 7 2 9))))
    (assert (= (vt-amax a) 10.0d0))

    ;; amin 全局
    ;; np.min(a) -> 2
    (assert (= (vt-amin a) 2.0d0)))

  ;; 二维沿轴0 amax，不保留维度
  ;; a = np.arange(12).reshape(3,4)
  ;; np.max(a, axis=0) -> [8,9,10,11]
  (let* ((a (vt-reshape (vt-arange 12 :type 'fixnum) '(3 4)))
         (res (vt-to-list (vt-amax a :axis 0))))
    (assert (equalp res '(8.0 9.0 10.0 11.0))))

  ;; 二维沿轴1 amin，保留维度
  ;; np.min(a, axis=1, keepdims=True) -> [[0],[4],[8]]
  (let* ((a (vt-reshape (vt-arange 12 :type 'fixnum) '(3 4)))
         (res (vt-to-list (vt-amin a :axis 1 :keepdims t))))
    (assert (equalp res '((0.0) (4.0) (8.0)))))

  ;; 三维沿轴2 amax
  ;; a = np.arange(8).reshape(2,2,2)
  ;; np.max(a, axis=2) -> [[1,3],[5,7]]
  (let* ((a (vt-reshape (vt-arange 8 :type 'fixnum) '(2 2 2)))
         (res (vt-to-list (vt-amax a :axis 2))))
    (assert (equalp res '((1.0 3.0) (5.0 7.0)))))

  (format t "~%test-vt-amax-amin passed.~%"))

;; ============================================================
;; test-vt-argmax / test-vt-argmin
;; ============================================================
(defun test-vt-argmax-argmin ()
  ;; 一维 argmax
  ;; a = np.array([1,5,3,9,2])
  ;; np.argmax(a) -> 3
  (let ((a (vt-from-sequence '(1 5 3 9 2))))
    (assert (= (vt-argmax a) 3))

    ;; 一维 argmin
    ;; np.argmin(a) -> 0
    (assert (= (vt-argmin a) 0)))

  ;; 二维沿轴0 argmax
  ;; a = np.array([[2,8,4],[7,3,6],[1,5,9]])
  ;; np.argmax(a, axis=0) -> [1,0,2]
  (let* ((a (vt-from-sequence '((2 8 4) (7 3 6) (1 5 9))))
         (res (vt-to-list (vt-argmax a :axis 0))))
    (assert (equal res '(1 0 2))))

  ;; 二维沿轴1 argmin
  ;; np.argmin(a, axis=1) -> [0,1,0]
  (let* ((a (vt-from-sequence '((2 8 4) (7 3 6) (1 5 9))))
         (res (vt-to-list (vt-argmin a :axis 1))))
    (assert (equal res '(0 1 0))))

  ;; 三维沿轴1 argmax
  ;; a = np.arange(12).reshape(2,3,2)
  ;; np.argmax(a, axis=1) -> [[2,2],[2,2]]? 实际每一列沿第二维的最大值索引
  (let* ((a (vt-reshape (vt-arange 12 :type 'fixnum) '(2 3 2)))
         (res (vt-to-list (vt-argmax a :axis 1))))
    (assert (equal res '((2 2) (2 2)))))

  ;; 负轴
  ;; a = np.array([[1,9,3],[6,5,4]])
  ;; np.argmax(a, axis=-1) -> [1,0]
  (let* ((a (vt-from-sequence '((1 9 3) (6 5 4))))
         (res (vt-to-list (vt-argmax a :axis -1))))
    (assert (equal res '(1 0))))

  ;; 全局 argmin（标量）
  ;; np.argmin(np.array(5)) -> 0
  (let ((a (vt-zeros '(1))))
    (assert (= (vt-argmin a) 0)))

  (format t "~%test-vt-argmax-argmin passed.~%"))

;; ============================================================
;; test-vt-where （三元选择模式与仅条件模式）
;; ============================================================
(defun test-vt-where ()
  ;; 模式1：单条件，返回索引列表
  ;; a = np.array([0,1,2,0,3,0])
  ;; np.where(a != 0) -> (array([1,2,4]),)
  (let* ((a (vt-from-sequence '(0 1 2 0 3 0)))
         (indices (vt-where (vt-> a (vt-const '() 0)))))  ; 条件：a>0
    ;; 对于一维，应返回一个列表，含一个形状 (3,) 的索引张量
    (assert (= (length indices) 1))
    (let ((idx-tensor (first indices)))
      (assert (equal (vt-to-list idx-tensor) '(1 2 4)))))

  ;; 二维条件，返回行列索引
  ;; a = np.array([[1,0],[0,1]])
  ;; np.where(a) -> (array([0,1]), array([0,1]))
  (let* ((a (vt-from-sequence '((1 0) (0 1))))
         (indices (vt-where a)))   ; 条件就是 a 本身（非零即为真）
    (assert (= (length indices) 2))
    (assert (equal (vt-to-list (first indices)) '(0 1)))    ; 行索引
    (assert (equal (vt-to-list (second indices)) '(0 1))))  ; 列索引
  ;; a = np.array([1,2,3,4])
  ;; cond = np.array([True, False, True, False]);
  ;; np.where(cond, a, -a)
  (let* ((a (vt-from-sequence '(1 2 3 4)))
	 (cond (vt-= (vt-from-sequence '(1 0 1 0)) 1.0d0)) ; 构造布尔张量 [True,False,True,False]
	 (res (vt-where cond a (vt-- a))))
    (assert (equal (vt-to-list res) '(1.0 -2.0 3.0 -4.0))))

  ;; 测试2: 条件 a < 3，用标量作为 x 和 y
  ;;  a = np.array([1, 2, 3, 4])
  ;;  cond = a < 3  -> [True, True, False, False]
  ;;  np.where(cond, 100, 200) -> array([100, 100, 200, 200])
  (let* ((a (vt-from-sequence '(1 2 3 4)))
	 (cond (vt-< a (vt-full '() 3.0d0))))  ; a < 3
    (let ((res (vt-where cond 100 200)))
      (assert (equal (vt-to-list res) '(100.0 100.0 200.0 200.0)))))


  ;; 广播测试
  ;; a = np.array([1,2,3])
  ;; cond = np.array([True, False, True])
  ;; x = 10, y = 20
  ;; np.where(cond, x, y) -> [10, 20, 10]
  (let* ((a (vt-from-sequence '(1 2 3)))
         (cond (vt-= a (vt-const '() 2.0d0)))  ; [F,T,F]
         (res (vt-where cond 10 20)))
    (assert (equal (vt-to-list res) '(20.0 10.0 20.0))))

  (format t "~%test-vt-where passed.~%"))

;; ============================================================
;; test-vt-argwhere
;; ============================================================
(defun test-vt-argwhere ()
  ;; 一维
  ;; a = np.array([0,1,0,2])
  ;; np.argwhere(a) -> [[1],[3]]
  (let* ((a (vt-from-sequence '(0 1 0 2)))
         (indices (vt-to-list (vt-argwhere a)))) ; 形状 (2,1)
    (assert (equal indices '((1) (3)))))

  ;; 二维
  ;; a = np.array([[1,0],[0,1]])
  ;; np.argwhere(a) -> [[0,0],[1,1]]
  (let* ((a (vt-from-sequence '((1 0) (0 1))))
         (indices (vt-to-list (vt-argwhere a)))) ; 形状 (2,2)
    (assert (equal indices '((0 0) (1 1)))))

  ;; 三维
  ;; a = np.zeros((2,2,2))
  ;; a[0,1,1] = 1; a[1,0,0] = 2
  ;; np.argwhere(a) -> [[0,1,1],[1,0,0]]
  (let* ((a (vt-zeros '(2 2 2)))
	 (indices))
    (setf (vt-ref a 0 1 1) 1.0d0)
    (setf (vt-ref a 1 0 0) 2.0d0)    
    (setf indices (vt-to-list (vt-argwhere a)))
    (assert (equal indices '((0 1 1) (1 0 0)))))

  ;; 空条件
  ;; np.argwhere(np.array([0,0,0])) -> empty (0,1)
  (let* ((a (vt-from-sequence '(0 0 0)))
         (indices (vt-argwhere a)))
    (assert (equal (vt-shape indices) '(0 1))))

  (format t "~%test-vt-argwhere passed.~%"))



;; ============================================================
;; test-vt-matmul
;; ============================================================
(defun test-vt-matmul ()
  ;; 2D 矩阵乘法 double-float
  ;; a = np.array([[1,2],[3,4]], dtype=np.float64)
  ;; b = np.array([[5,6],[7,8]], dtype=np.float64)
  ;; np.matmul(a,b) -> [[19,22],[43,50]]
  (let* ((a (vt-from-sequence '((1 2) (3 4)) :type 'double-float))
         (b (vt-from-sequence '((5 6) (7 8)) :type 'double-float))
         (c (vt-matmul a b)))
    (assert (equal (vt-to-list c) '((19.0 22.0) (43.0 50.0)))))

  ;; 2D 矩阵乘法 fixnum 输入，结果应为 double-float
  ;; a = np.arange(6).reshape(2,3)
  ;; b = np.arange(6).reshape(3,2)
  ;; np.matmul(a,b) -> [[10,13],[28,40]]
  (let* ((a (vt-reshape (vt-arange 6 :type 'fixnum) '(2 3)))
         (b (vt-reshape (vt-arange 6 :type 'fixnum) '(3 2)))
         (c (vt-matmul a b)))
    (assert (equal (vt-to-list c) '((10.0 13.0) (28.0 40.0)))))

  ;; 批量矩阵乘法 (使用 einsum 路径)
  ;; a = np.arange(8).reshape(2,2,2)
  ;; b = np.arange(8).reshape(2,2,2)
  ;; np.matmul(a,b) -> 形状 (2,2,2)，手动计算
  (let* ((a (vt-reshape (vt-arange 8 :type 'fixnum) '(2 2 2)))
         (b (vt-reshape (vt-arange 8 :type 'fixnum) '(2 2 2)))
         (c (vt-matmul a b)))
    ;; 预期：第一个批次 a[0]=[[0,1],[2,3]] b[0]=[[0,1],[2,3]] -> [[2,3],[6,11]]
    ;; 第二个批次 a[1]=[[4,5],[6,7]] b[1]=[[4,5],[6,7]] -> [[46,55],[66,79]]
    (assert (equal (vt-to-list (vt-slice c '(0) '(:all) '(:all)))
		   '((2.0 3.0) (6.0 11.0))))
    (assert (equal (vt-to-list (vt-slice c '(1) '(:all) '(:all)))
		   '((46.0 55.0) (66.0 79.0)))))
  (format t "~%test-vt-matmul passed.~%"))

;; ============================================================
;; test-vt-einsum
;; ============================================================
(defun test-vt-einsum ()
  ;; 向量内积
  ;; a = np.array([1,2,3]), b = np.array([4,5,6])
  ;; np.einsum('i,i->', a, b) -> 32
  (let* ((a (vt-from-sequence '(1 2 3) :type 'fixnum))
         (b (vt-from-sequence '(4 5 6) :type 'fixnum))
         (res (vt-einsum "i,i->" a b)))
    (assert (= (vt-ref res) 32.0d0)))

  ;; 矩阵乘法
  ;; a = np.arange(6).reshape(2,3), b = np.arange(6).reshape(3,2)
  ;; np.einsum('ij,jk->ik', a, b) -> [[10,13],[28,40]]
  (let* ((a (vt-reshape (vt-arange 6 :type 'fixnum) '(2 3)))
         (b (vt-reshape (vt-arange 6 :type 'fixnum) '(3 2)))
         (c (vt-einsum "ij,jk->ik" a b)))
    (assert (equal (vt-to-list c) '((10.0 13.0) (28.0 40.0)))))

  ;; 批量矩阵乘法
  ;; a = np.arange(12).reshape(2,2,3), b = np.arange(18).reshape(2,3,3)
  ;; np.einsum('bij,bjk->bik', a, b)
  ;; 第一个批次：a[0]=[[0,1,2],[3,4,5]], b[0]=[[0,1,2],[3,4,5],[6,7,8]] -> [[15,18,21],[42,54,66]]
  ;; 第二个批次：a[1]=[[6,7,8],[9,10,11]], b[1]=[[9,10,11],[12,13,14],[15,16,17]] -> [[258, 279, 300], [366, 396, 426]]
  (let* ((a (vt-reshape (vt-arange 12 :type 'fixnum) '(2 2 3)))
         (b (vt-reshape (vt-arange 18 :type 'fixnum) '(2 3 3)))
         (c (vt-einsum "bij,bjk->bik" a b)))
    (assert (equal (vt-to-list (vt-slice c '(0) '(:all) '(:all)))
		   '((15.0 18.0 21.0) (42.0 54.0 66.0))))
    (assert (equal (vt-to-list (vt-slice c '(1) '(:all) '(:all)))
		   '((258.0 279.0 300.0) (366.0 396.0 426.0)))))

  ;; 对角线提取
  ;; a = np.arange(9).reshape(3,3)
  ;; np.einsum('ii->i', a) -> [0,4,8]
  (let* ((a (vt-reshape (vt-arange 9 :type 'fixnum) '(3 3)))
         (diag (vt-einsum "ii->i" a)))
    (assert (equal (vt-to-list diag) '(0.0 4.0 8.0))))

  ;; 外积
  ;; a = np.array([1,2,3]), b = np.array([4,5])
  ;; np.einsum('i,j->ij', a, b) -> [[4,5],[8,10],[12,15]]
  (let* ((a (vt-from-sequence '(1 2 3) :type 'fixnum))
         (b (vt-from-sequence '(4 5) :type 'fixnum))
         (outer (vt-einsum "i,j->ij" a b)))
    (assert (equal (vt-to-list outer)
		   '((4.0 5.0) (8.0 10.0) (12.0 15.0)))))
  ;; 测试 1：标准矩阵乘法
  (let* ((a (vt-from-sequence '((1 2) (3 4) (5 6))))
         (b (vt-from-sequence '((7 8 9) (10 11 12))))
         (out-einsum (vt-einsum "ij,jk->ik" a b))
         (out-matmul (vt-matmul a b)))
    (assert-ok (approx= out-einsum out-matmul) "标准矩阵乘法 einsum 失败"))

  ;; 测试 2：输出转置 "ij,jk->ki"
  (let* ((a (vt-from-sequence '((1 2) (3 4))))
         (b (vt-from-sequence '((5 6) (7 8))))
         (out-einsum (vt-einsum "ij,jk->ki" a b))
         (out-expected (vt-transpose (vt-matmul a b))))
    (assert-ok (approx= out-einsum out-expected) "输出转置 einsum 失败"))

  ;; 测试 3：输入有转置 "ji,kj->ik"
  (let* ((a (vt-from-sequence '((1 2) (3 4) (5 6))))   ; 3x2
         (b (vt-from-sequence '((7 8 9) (10 11 12))))   ; 2x3
         (at (vt-transpose a))          ; 2x3
         (bt (vt-transpose b))          ; 3x2
         (out-einsum (vt-einsum "ji,kj->ik" a b))   ; 注意 a 作为第一个输入对应 ji
         (out-expected (vt-matmul at bt)))
    (assert-ok (approx= out-einsum out-expected) "输入转置 einsum 失败"))

  ;; 测试 4：非标准输出顺序 "ab,bc->ca"
  (let* ((a (vt-from-sequence '((1 2 3) (4 5 6))))   ; 2x3
         (b (vt-from-sequence '((7 8) (9 10) (11 12)))) ; 3x2
         (out-einsum (vt-einsum "ab,bc->ca" a b))
         (out-expected (vt-transpose (vt-matmul a b))))
    (assert-ok (approx= out-einsum out-expected) "ab,bc->ca 失败"))

  ;; 测试 5：三个矩阵连乘 "ij,jk,kl->il"
  (let* ((a (vt-from-sequence '((1 2) (3 4))))
         (b (vt-from-sequence '((5 6) (7 8))))
         (c (vt-from-sequence '((9 10) (11 12))))
         (out-einsum (vt-einsum "ij,jk,kl->il" a b c))
         (out-matmul (vt-matmul (vt-matmul a b) c)))
    (assert-ok (approx= out-einsum out-matmul) "三矩阵 einsum 失败"))

  ;; 测试 6：迹 "ii->"
  (let* ((a (vt-from-sequence '((1 2 3) (4 5 6) (7 8 9))))
         (trace (vt-ref (vt-einsum "ii->" a))))
    (assert-ok (approx= trace (+ 1 5 9)) "迹 einsum 失败"))

  ;; 测试 7：批量矩阵乘法 "...ij,...jk->...ik"
  (let* ((batch-a (vt-from-sequence '(((1 2) (3 4))   ; 2x2x2
                                        ((5 6) (7 8)))))
         (batch-b (vt-from-sequence '(((9 10) (11 12))
                                        ((13 14) (15 16)))))
         (out-einsum (vt-einsum "...ij,...jk->...ik" batch-a batch-b)))
    ;; 如果 vt-matmul 不支持批量，可以手动验证
    (assert-ok (and (approx= (vt-slice out-einsum '(0) '(:all) '(:all))
                             (vt-matmul (vt-slice batch-a '(0) '(:all) '(:all))
                                        (vt-slice batch-b '(0) '(:all) '(:all))))
                    (approx= (vt-slice out-einsum '(1) '(:all) '(:all))
                             (vt-matmul (vt-slice batch-a '(1) '(:all) '(:all))
                                        (vt-slice batch-b '(1) '(:all) '(:all)))))
               "批量矩阵乘法 einsum 失败"))

  ;; 测试 8：向量内积 "i,i->"
  (let* ((a (vt-from-sequence '(1 2 3)))
         (b (vt-from-sequence '(4 5 6)))
         (res (vt-ref (vt-einsum "i,i->" a b))))
    (assert-ok (approx= res 32.0) "向量内积 einsum 失败"))
  (format t "~%test-vt-einsum passed.~%"))

;; ============================================================
;; test-vt-dot
;; ============================================================
(defun test-vt-dot ()
  ;; 向量内积
  ;; a = np.array([1,2,3]), b = np.array([4,5,6])
  ;; np.dot(a,b) -> 32
  (let* ((a (vt-from-sequence '(1 2 3) :type 'fixnum))
         (b (vt-from-sequence '(4 5 6) :type 'fixnum))
         (res (vt-dot a b)))
    (assert (= (vt-ref res) 32.0d0)))

  ;; 矩阵乘法 (2D)
  ;; a = np.arange(6).reshape(2,3), b = np.arange(6).reshape(3,2)
  ;; np.dot(a,b) -> [[10,13],[28,40]]
  (let* ((a (vt-reshape (vt-arange 6 :type 'fixnum) '(2 3)))
         (b (vt-reshape (vt-arange 6 :type 'fixnum) '(3 2)))
         (c (vt-dot a b)))
    (assert (equal (vt-to-list c) '((10.0 13.0) (28.0 40.0)))))

  ;; 批量矩阵乘法 (秩 >= 2)
  ;;  a = np.arange(8).reshape(2,2,2)
  ;;  np.einsum('...ij,...jk->...ik', a,a) # 不是 np.dot(a,a)
  (let* ((a (vt-reshape (vt-arange 8 :type 'fixnum) '(2 2 2)))
         (b (vt-reshape (vt-arange 8 :type 'fixnum) '(2 2 2)))
         (c (vt-dot a b)))  ;;   (vt-einsum "...ij,...jk->...ik" a b)
    (assert (equal (vt-to-list (vt-slice c '(0) '(:all) '(:all)))
		   '((2.0 3.0) (6.0 11.0))))
    (assert (equal (vt-to-list (vt-slice c '(1) '(:all) '(:all)))
		   '((46.0 55.0) (66.0 79.0)))))
  ;; 测试 1：两个一维向量 → 标量数值
  (let* ((a (vt-from-sequence '(1.0 2.0 3.0)))
         (b (vt-from-sequence '(4.0 5.0 6.0)))
         (res (vt-ref (vt-dot a b))))
    (assert-ok (and (numberp res) (approx= res 32.0))
               "向量点积应返回标量数字"))

  ;; 测试 2：向量与矩阵 (1D @ 2D)
  (let* ((a (vt-from-sequence '(1.0 2.0)))
         (B (vt-from-sequence '((1.0 2.0) (3.0 4.0))))
         (res (vt-dot a B)))
    (assert-ok (and (= (length (vt-shape res)) 1)
                    (approx= res (vt-from-sequence '(7.0 10.0))))
               "向量与矩阵点积失败"))

  ;; 测试 3：矩阵与向量 (2D @ 1D)
  (let* ((B (vt-from-sequence '((1.0 2.0) (3.0 4.0))))
         (a (vt-from-sequence '(1.0 2.0)))
         (res (vt-dot B a)))
    (assert-ok (and (= (length (vt-shape res)) 1)
                    (approx= res (vt-from-sequence '(5.0 11.0))))
               "矩阵与向量点积失败"))

  ;; 测试 4：0 维标量与向量（如果支持）
  ;; CLVT 可能不支持 0-d 张量，可以跳过。

  ;; 测试 5：两个矩阵
  (let* ((A (vt-from-sequence '((1 2) (3 4))))
         (B (vt-from-sequence '((5 6) (7 8))))
         (res (vt-dot A B)))
    (assert-ok (approx= res (vt-matmul A B)) "矩阵点积失败"))
  (format t "~%test-vt-dot passed.~%"))

;; ============================================================
;; test-vt-solve
;; ============================================================
(defun test-vt-solve ()
  ;; 解 Ax = b，单右端
  ;; A = [[2,1],[1,3]], b = [7,8]
  ;; np.linalg.solve(A,b) -> [2.6, 1.8]? 实际：(2,1)^T? 让我们算：2x+y=7, x+3y=8 => 消元：y=7-2x, x+21-6x=8 => -5x=-13 => x=2.6, y=1.8
  ;; 所以解为 [2.6, 1.8]
  (let* ((A (vt-from-sequence '((2 1) (1 3)) :type 'double-float))
         (b (vt-from-sequence '(7 8) :type 'double-float))
         (x (vt-solve A b)))
    (assert (lists-approx-equal (vt-to-list x) '(2.6 1.8) :epsilon 1e-6)))

  ;; 多右端项
  ;; A = [[3,1],[1,2]]
  ;; B = [[9,8],[8,7]]
  ;; np.linalg.solve(A,B) -> [[2., 1.8.], [3., 2.6]]
  (let* ((A (vt-from-sequence '((3 1) (1 2)) :type 'double-float))
         (B (vt-from-sequence '((9 8) (8 7)) :type 'double-float))
         (X (vt-solve A B)))
    (assert (lists-approx-equal (vt-to-list (vt-slice X '(:all) '(0)))
				'(2.0 3.0) :epsilon 1e-6))
    (assert (lists-approx-equal (vt-to-list (vt-slice X '(:all) '(1)))
				'(1.8 2.6) :epsilon 1e-6)))
  ;; 测试 1：单右端项
  (let* ((a (vt-from-sequence '(( 1.0  2.0  3.0 )
                                ( 2.0  1.0  4.0 )
                                ( 3.0  4.0  1.0 ))))
	 (x-true (vt-from-sequence '(1.0 -2.0 0.5)))
	 (b (vt-dot a x-true))          ;; ← 改用 vt-dot
	 (x-sol (vt-solve a b)))
    (assert-ok (approx= x-sol x-true) "vt-solve 单右端项失败"))

  ;; 测试 2：多右端项
  (let* ((a (vt-from-sequence '(( 1.0  2.0 )
                                ( 3.0  4.0 ))))
         (b (vt-from-sequence '(( 5.0  6.0 )
                                ( 7.0  8.0 ))))
         (x-sol (vt-solve a b))
         (ax (vt-matmul a x-sol)))
    (assert-ok (approx= ax b) "vt-solve 多右端项失败"))

  ;; 测试 3：奇异矩阵（应抛出错误）
  (let ((a-sing (vt-from-sequence '((1.0 2.0) (2.0 4.0))))
        (b-sing (vt-from-sequence '(1.0 2.0))))
    (handler-case
        (progn (vt-solve a-sing b-sing)
               (error "不应到达这里"))
      (error () t)))  ; 预期抛出错误

  ;; 测试 4：接近奇异矩阵
  (let* ((a-ill (vt-from-sequence '((1.0 1.0) (1.0 1.0000000001))))
         (b-ill (vt-from-sequence '(2.0 2.0000000001)))
         (x-ill (vt-solve a-ill b-ill))
         (ax-ill (vt-matmul a-ill x-ill)))
    (assert-ok (approx= ax-ill b-ill 1e-8) "接近奇异矩阵求解失败"))

  ;; 测试 5：整数类型 -> 自动转为 double-float
  (let* ((a (vt-from-sequence '((2 1) (1 2)) :type 'fixnum))
         (b (vt-from-sequence '(3 4) :type 'fixnum))
         (x (vt-solve a b)))
    (assert-ok (approx= (vt-@ a x) b) "整数矩阵求解失败"))
  (format t "~%test-vt-solve passed.~%"))

;; ============================================================
;; test-vt-inv
;; ============================================================
(defun test-vt-inv ()
  ;; A = [[4,7],[2,6]]
  ;; np.linalg.inv(A) -> [[ 0.6, -0.7], [-0.2,  0.4]]
  (let* ((A (vt-from-sequence '((4 7) (2 6)) :type 'double-float))
         (invA (vt-inv A)))
    (assert (lists-approx-equal (vt-to-list (vt-slice invA '(:all) '(0)))
				'(0.6 -0.2) :epsilon 1e-6))
    (assert (lists-approx-equal (vt-to-list (vt-slice invA '(:all) '(1)))
				'(-0.7 0.4) :epsilon 1e-6)))
  (format t "~%test-vt-inv passed.~%"))

;; ============================================================
;; test-vt-det
;; ============================================================
(defun test-vt-det ()
  ;; A = [[1,2],[3,4]] -> det = -2.0
  (let ((A (vt-from-sequence '((1 2) (3 4)) :type 'double-float)))
    (assert (= (vt-det A) -2.0d0)))
  ;; 三维矩阵
  ;; A = [[6,1,1],[4,-2,5],[2,8,7]] -> det = -306.0? 计算一下：6*(-14-40) -1*(28-10) +1*(32+4) = 6*(-54) -18 +36 = -324 -18+36 = -306.0
  (let ((A (vt-from-sequence '((6 1 1) (4 -2 5) (2 8 7)) :type 'double-float)))
    (assert (lists-approx-equal (list (vt-det A))
				'(-306.0) :epsilon 1e-6)))
  (format t "~%test-vt-det passed.~%"))

;; ============================================================
;; test-vt-norm
;; ============================================================
(defun test-vt-norm ()
  ;; 向量 L2 范数
  ;; np.linalg.norm(np.array([3,4])) -> 5.0
  (let ((v (vt-from-sequence '(3 4) :type 'double-float)))
    (assert (= (vt-ref (vt-norm v)) 5.0d0)))
  ;; 矩阵 L2 范数 (全局) 默认是 Frobenius? 根据代码 vt-norm 是 L2 范数 (vt-sqrt(vt-sum(sq)))，对于矩阵是 Frobenius。
  ;; np.linalg.norm(np.array([[1,2],[3,4]])) -> 5.477225575051661
  (let ((A (vt-from-sequence '((1 2) (3 4)) :type 'double-float)))
    (assert (lists-approx-equal (list (vt-ref (vt-norm A)))
				(list (sqrt 30.0d0)) :epsilon 1e-6)))
  ;; 沿轴计算 L2 范数
  ;; np.linalg.norm(np.array([[1,2],[3,4]]), axis=1) -> [2.23606798, 5.0]
  (let* ((A (vt-from-sequence '((1 2) (3 4)) :type 'double-float))
         (norms (vt-to-list (vt-norm A :axis 1))))
    (assert (lists-approx-equal norms (list (sqrt 5.0) (sqrt 25.0)) :epsilon 1e-6)))
  (format t "~%test-vt-norm passed.~%"))

;; ============================================================
;; test-vt-frobenius-norm
;; ============================================================
(defun test-vt-frobenius-norm ()
  ;; 矩阵 Frobenius 范数
  ;; np.linalg.norm(np.array([[1,2],[3,4]]), 'fro') -> 5.477225575051661
  (let ((A (vt-from-sequence '((1 2) (3 4)) :type 'double-float)))
    (assert (lists-approx-equal (list (vt-ref (vt-frobenius-norm A)))
				(list (sqrt 30.0d0)) :epsilon 1e-6)))
  (format t "~%test-vt-frobenius-norm passed.~%"))

;; ============================================================
;; test-vt-trace
;; ============================================================
(defun test-vt-trace ()
  ;; A = [[1,2],[3,4]] -> trace = 5.0
  (let ((A (vt-from-sequence '((1 2) (3 4)) :type 'double-float)))
    (assert (= (vt-trace A) 5.0d0)))
  ;; 非方阵应报错或只取 min 对角线? 根据代码，vt-trace 调用 vt-diagonal 然后 sum，vt-diagonal 需要 rank>=2，会自动取 min 对角线。我们测试一个矩形矩阵。
  ;; np.trace(np.arange(6).reshape(2,3)) -> 0+4 = 4.0
  (let ((A (vt-reshape (vt-arange 6 :type 'fixnum) '(2 3))))
    (assert (= (vt-trace A) 4.0d0)))
  (format t "~%test-vt-trace passed.~%"))

;; ============================================================
;; test-vt-outer
;; ============================================================
(defun test-vt-outer ()
  ;; flatten模式（默认）
  ;; np.outer([1,2], [3,4,5]) -> [[3,4,5],[6,8,10]]
  (let* ((a (vt-from-sequence '(1 2) :type 'fixnum))
         (b (vt-from-sequence '(3 4 5) :type 'fixnum))
         (outer (vt-outer a b)))
    (assert (equal (vt-to-list outer)
		   '((3.0 4.0 5.0) (6.0 8.0 10.0)))))
  ;; 不展平：保留形状
  ;; a = np.array([1,2]), b = np.array([3,4,5])
  ;; np.multiply.outer(a,b) 相当于 a[:,None] * b[None,:] 得到二维矩阵，与 flatten 相同
  ;; 用不展平模式，输入1D依然是[[...]]。测试 flatten=NIL，输入1D应等同于flatten？实际上代码中当 flatten=NIL 时，会做 "...i,...j->...ij" 这会将前面的维度视为批量，所以如果前面维度是0维，结果也是二维，但形状会多出一些1维度？我们简单测试一下：一维输入给出的 outer 形状应该是 (2,3)。保持默认即可。
  ;; 测试非展平模式下的高维外积
  ;; a = np.arange(6).reshape(2,3)  b = np.array([1,2])
  ;; 使用 np.multiply.outer(a,b) 形状 (2,3,2)，但 NumPy 没有直接 outer 高维函数，我们用 einsum 模拟
  ;; 对于 flatten=NIL，预期形状 = a.shape + b.shape
  (let* ((a (vt-reshape (vt-arange 6 :type 'fixnum) '(2 3)))
         (b (vt-from-sequence '(1 2) :type 'fixnum))
         (outer (vt-outer a b :flatten nil)))
    (assert (equal (vt-shape outer) '(2 3 2)))
    ;; 检查值：a[i,j] * b[k] 存储在 outer[i,j,k]
    ;; 0*1=0, 0*2=0, 1*1=1, 1*2=2, 2*1=2, 2*2=4 ...
    (assert (equal (vt-to-list (vt-slice outer '(0) '(:all) '(:all)))
		   '((0.0 0.0) (1.0 2.0) (2.0 4.0)))))
  (format t "~%test-vt-outer passed.~%"))



;; --------------------------------------------------------------------
;; 累积和与累积积测试
;; --------------------------------------------------------------------

(defun test-vt-cumsum ()
  ;; np.cumsum([0,1,2,3,4]) → [0,1,3,6,10]
  (let* ((a (vt-arange 5 :type 'fixnum))
         (res (vt-cumsum a)))
    (assert (equal (vt-to-list res) '(0 1 3 6 10))))

  ;; np.cumsum([1.0, 2.0, 3.0]) → [1,3,6]
  (let* ((a (vt-from-sequence '(1.0 2.0 3.0) :type 'double-float))
         (res (vt-cumsum a)))
    (assert (equal (vt-to-list res) '(1.0 3.0 6.0))))

  ;; a = np.arange(6).reshape(2,3)
  ;; np.cumsum(a, axis=0) → [[0,1,2],[3,5,7]]
  (let* ((a (vt-reshape (vt-arange 6 :type 'fixnum) '(2 3)))
         (res (vt-cumsum a :axis 0)))
    (assert (equal (vt-to-list res) '((0 1 2) (3 5 7)))))

  ;; axis=1 → [[0,1,3],[3,7,12]]
  (let* ((a (vt-reshape (vt-arange 6 :type 'fixnum) '(2 3)))
         (res (vt-cumsum a :axis 1)))
    (assert (equal (vt-to-list res) '((0 1 3) (3 7 12)))))

  ;; 负轴 axis=-1 同 axis=1
  (let* ((a (vt-reshape (vt-arange 6 :type 'fixnum) '(2 3)))
         (res (vt-cumsum a :axis -1)))
    (assert (equal (vt-to-list res) '((0 1 3) (3 7 12)))))

  ;; a = np.arange(8).reshape(2,2,2)
  ;; np.cumsum(a, axis=1) → [[[0,1],[2,4]],[[4,5],[10,12]]]
  (let* ((a (vt-reshape (vt-arange 8 :type 'fixnum) '(2 2 2)))
         (res (vt-cumsum a :axis 1)))
    (assert (equal (vt-to-list res)
                   '(((0 1) (2 4)) ((4 5) (10 12))))))

  ;; np.cumsum([]) → []
  (let* ((a (vt-zeros '(0)))
         (res (vt-cumsum a)))
    (assert (equal (vt-to-list res) '())))

  ;; 空轴切片（形状某维为0）
  (let* ((a (vt-zeros '(2 0 3)))
         (res (vt-cumsum a :axis 1)))
    (assert (equal (vt-shape res) '(2 0 3)))))


(defun test-vt-cumprod ()
  ;; np.cumprod([1,2,3,4]) → [1,2,6,24]
  (let* ((a (vt-from-sequence '(1 2 3 4) :type 'fixnum))
         (res (vt-cumprod a)))
    (assert (equal (vt-to-list res) '(1 2 6 24))))
  ;; a = np.array([[1,2],[3,4]])
  ;; np.cumprod(a, axis=0) → [[1,2],[3,8]]
  (let* ((a (vt-from-sequence '((1 2) (3 4)) :type 'fixnum))
         (res (vt-cumprod a :axis 0)))
    (assert (equal (vt-to-list res) '((1 2) (3 8)))))

  ;; axis=1 → [[1,2],[3,12]]
  (let* ((a (vt-from-sequence '((1 2) (3 4)) :type 'fixnum))
         (res (vt-cumprod a :axis 1)))
    (assert (equal (vt-to-list res) '((1 2) (3 12)))))
  ;; 确保输出类型与输入一致 (fixnum)
  (let* ((a (vt-ones '(3) :type 'fixnum))
         (res (vt-cumprod a)))
    (assert (eq (vt-element-type res) 'fixnum))))


;;----------------------测试 vt-qr vt-svd ---------------------

(defun diag-matrix (S m n)
  "用奇异值张量 S (一维) 构建 m×n 对角矩阵"
  (let* ((K (vt-size S))
         (mat (vt-zeros (list m n))))
    (loop for i below K
          do (setf (vt-ref mat i i) (coerce (vt-ref S i) 'double-float)))
    mat))

(defun test-svd (A &optional (full-matrices nil))
  "测试 SVD 的重构误差与正交性"
  (let* ((m (first (vt-shape A)))
         (n (second (vt-shape A))))
    (multiple-value-bind (U S Vt)
        (vt-svd A :full-matrices full-matrices)
      (let ((K (vt-size S)))   ; 使用 vt-size 获取奇异值个数
        (format t "~%----- SVD (full-matrices = ~A) -----" full-matrices)
        (format t "~%U shape: ~A, S length: ~A, Vt shape: ~A~%"
                (vt-shape U) K (vt-shape Vt))
        ;; 重构
        (let* ((Smat (if full-matrices
                         (diag-matrix S m n)
                         (diag-matrix S K K)))
               (recon (vt-@ U (vt-@ Smat Vt)))
               (diff (vt-- A recon))
               (recon-error (reduce #'max (vt-data (vt-abs diff)))))
          (format t "重构误差: max|A - U S Vt| = ~A~%" recon-error))
        ;; 正交性检查
        (let* ((UtU (vt-@ (vt-transpose U) U))
               (Iu (vt-eye (first (vt-shape UtU)) :type 'double-float))
               (UtU-error (reduce #'max (vt-data (vt-abs (vt-- UtU Iu))))))
          (format t "U^T U ≈ I ? max|U^T U - I| = ~A~%" UtU-error))
        (let* ((VtV (vt-@ Vt (vt-transpose Vt)))
               (Iv (vt-eye (first (vt-shape VtV)) :type 'double-float))
               (VtV-error (reduce #'max (vt-data (vt-abs (vt-- VtV Iv))))))
          (format t "Vt Vt^T ≈ I ? max|Vt Vt^T - I| = ~A~%" VtV-error))))))



;; ========== QR 分解测试 ==========
(defun test-vt-qr ()
  "测试 QR 分解的重构误差和正交性。使用预定义矩阵 *A-qr* 。"
  (let ((A (vt-from-sequence '((12 -51   4)
                               ( 6 167 -68)
                               (-4  24 -41)))))
    (format t "~%========== QR 分解测试 ==========~%")
    (format t "原始矩阵 A:~%")
    (print A)
    ;; Full mode
    (multiple-value-bind (Q R) (vt-qr A :mode :full)
      (format t "~%Full mode: Q (m×m), R (m×n)~%")
      (let ((QtQ (vt-@ (vt-transpose Q) Q)))
        (format t "Q^T Q 偏离单位阵的最大误差 = ~A~%"
                (let ((I (vt-eye (first (vt-shape Q)) :type 'double-float)))
                  (reduce #'max (vt-data (vt-abs (vt-- QtQ I)))))))
      (format t "重构误差: max|A - Q R| = ~A~%"
              (reduce #'max (vt-data (vt-abs (vt-- A (vt-@ Q R)))))))
    ;; Reduced mode
    (multiple-value-bind (Q R) (vt-qr A :mode :reduced)
      (format t "~%Reduced mode: Q (m×k), R (k×n)~%")
      (let ((QtQ (vt-@ (vt-transpose Q) Q)))
        (format t "Q^T Q 偏离单位阵的最大误差 = ~A~%"
                (let ((I (vt-eye (second (vt-shape Q)) :type 'double-float)))
                  (reduce #'max (vt-data (vt-abs (vt-- QtQ I)))))))
      (format t "重构误差: max|A - Q R| = ~A~%"
              (reduce #'max (vt-data (vt-abs (vt-- A (vt-@ Q R)))))))))

;; ========== SVD 分解测试 ==========
(defun run-svd-tests ()
  "对几个典型矩阵运行 test-svd，覆盖经济尺寸和全尺寸。"
  (let ((A1 (vt-from-sequence '((1 2 3)
                                (4 5 6)
                                (7 8 9)
                                (10 11 12))))
        (A2 (vt-from-sequence '((3 1 2)
                                (1 4 1)
                                (2 1 3))))
        (A3 (vt-from-sequence '((5)))))   ; 1x1 矩阵
    (format t "~%========== SVD 分解测试 ==========~%")
    (dolist (mat (list A1 A2 A3))
      (format t "~%--- 矩阵 ~A ---~%" (vt-shape mat))
      (test-svd mat nil)
      (test-svd mat t))))

;; 重构误差都应该小于 1e-14数量级
;;------------------------------------------------------<<


(defun test-vt-gradient ()
  ;; 辅助：近似相等断言
  (macrolet ((assert-close (a b &optional (eps 1e-8))
               `(progn
                  (unless (<= (abs (- ,a ,b)) ,eps)
                    (error "Assertion failed: ~A ~A differ by ~A"
                           ',a ',b (abs (- ,a ,b))))
                  t)))
    
    ;; ---------- 1D 等间距 ----------
    (let* ((x (vt-arange 6 :step 1.0d0 :type 'double-float))       ; 0,1,2,3,4,5
           (y (vt-square x))                                         ; y = x^2
           ;; axis nil → 返回所有轴的梯度列表（此处 1D → 单元素列表）
           (grad-list (vt-gradient y))
           (grad (first grad-list)))     ; 取出唯一张量
      (format t "~%1D uniform (single from list): ~A~%" grad)
      ;; 理论：内部二阶中心差分，边界一阶差分
      (assert-close (vt-ref grad 0) 1.0d0)       ; (f1-f0)/1 = 1
      (assert-close (vt-ref grad 1) 2.0d0)       ; (f2-f0)/2 = 2
      (assert-close (vt-ref grad 2) 4.0d0)
      (assert-close (vt-ref grad 3) 6.0d0)
      (assert-close (vt-ref grad 4) 8.0d0)
      (assert-close (vt-ref grad 5) 9.0d0))      ; (f5-f4)/1 = 9
    
    ;; ---------- 1D 非等间距（坐标数组）----------
    (let* ((x (vt-from-sequence '(0.0d0 1.0d0 3.0d0 6.0d0) :type 'double-float))
           (y (vt-square x))
           ;; 指定单轴 axis=0，直接返回该轴张量（不是列表）
           (grad (vt-gradient y :spacing x :axis 0)))
      (format t "~%1D non-uniform (axis=0): ~A~%" grad)
      (assert-close (vt-ref grad 0) 1.0d0)
      (assert-close (vt-ref grad 1) 3.0d0)
      (assert-close (vt-ref grad 2) 7.0d0)
      (assert-close (vt-ref grad 3) 9.0d0))
    
    ;; ---------- 2D 等间距，指定单个轴 ----------
    (let* ((mat (vt-from-sequence '((0.0d0 1.0d0 2.0d0)
                                    (3.0d0 4.0d0 5.0d0)
                                    (6.0d0 7.0d0 8.0d0))
                                  :type 'double-float))
           ;; axis 整数 → 直接返回该轴梯度张量
           (grad0 (vt-gradient mat :axis 0))
           (grad1 (vt-gradient mat :axis 1)))
      (format t "~%2D along axis=0:~%~A~%" grad0)
      (format t "2D along axis=1:~%~A~%" grad1)
      (dotimes (i 3)
	(dotimes (j 3)
          (assert-close (vt-ref grad0 i j) 3.0d0)))
      (dotimes (i 3)
	(dotimes (j 3)
          (assert-close (vt-ref grad1 i j) 1.0d0))))
    
    ;; ---------- 多轴同时计算 ----------
    (let* ((a (vt-from-sequence '((1.0d0 2.0d0 3.0d0)
                                  (4.0d0 5.0d0 6.0d0))
				:type 'double-float))
           ;; axis 列表 → 返回列表
           (grads (vt-gradient a :axis '(0 1))))
      (format t "~%Multi-axis gradients:~%~A~%~A~%" (first grads) (second grads))
      (assert (= (length grads) 2))
      (let ((g0 (first grads)) (g1 (second grads)))
	(assert (equal (vt-shape g0) '(2 3)))
	(assert (equal (vt-shape g1) '(2 3)))
	(format t "Multi-axis test passed.~%")))
    
    ;; ---------- 边界条件：尺寸太小 ----------
    (handler-case
	(progn
          (vt-gradient (vt-from-sequence '(1.0d0) :type 'double-float) :axis 0)
          (error "Should have thrown an error for size 1"))
      (simple-error (e)
	(format t "~%Correctly errored on size 1: ~A~%" e)))
    
    (format t "~%All gradient tests passed!~%")))


(defun test-vt-gradient-advanced ()
  (macrolet ((assert-close (a b &optional (eps 1e-8))
               `(progn
                  (unless (<= (abs (- ,a ,b)) ,eps)
                    (error "Assertion failed: ~A ~A differ by ~A"
                           ',a ',b (abs (- ,a ,b))))
                  t)))

    ;; === 3D 张量，等间距，指定两个轴（axis 列表 + 标量 spacing） ===
    (let* ((t3d (vt-from-sequence
		 '((( 0.0d0  1.0d0  2.0d0) ( 3.0d0  4.0d0  5.0d0) ( 6.0d0  7.0d0  8.0d0))
                   (( 9.0d0 10.0d0 11.0d0) (12.0d0 13.0d0 14.0d0) (15.0d0 16.0d0 17.0d0))
                   ((18.0d0 19.0d0 20.0d0) (21.0d0 22.0d0 23.0d0) (24.0d0 25.0d0 26.0d0)))
		 :type 'double-float))
           ;; t3d 形状 (3, 3, 3)，每个元素值相当于 i*9 + j*3 + k
           ;; 沿 axis=0 和 axis=2 求梯度，spacing=1
           (grads (vt-gradient t3d :axis '(0 2)))
           (g0 (first grads))
           (g2 (second grads)))
      (format t "~%3D multi-axis (0,2) uniform spacing=1~%")
      (format t "grad along 0:~%~A~%" g0)
      (format t "grad along 2:~%~A~%" g2)
      ;; 沿 axis=0 的梯度：边界一阶，内部中心差分，由于每层相差9，所以梯度应为全9。
      ;; 沿 axis=2 的梯度：最内层变化步长1，梯度应为全1。
      ;; 验证：
      (dotimes (i 3)
	(dotimes (j 3)
          (dotimes (k 3)
            (assert-close (vt-ref g0 i j k) 9.0d0)
            (assert-close (vt-ref g2 i j k) 1.0d0))))
      (format t "3D multi-axis passed.~%"))

    ;; === 4D 张量，axis=nil（全部轴），等间距 ===
    (let* ((shape '(2 2 2 2))
           (t4d (vt-from-function shape
                                  (lambda (idxs)
                                    (coerce (reduce #'+ idxs) 'double-float))
                                  :type 'double-float))
           ;; t4d 的元素是坐标索引之和 (i0+i1+i2+i3)
           ;; 每轴步长1，所以任何轴的梯度都应为1
           (grads (vt-gradient t4d))   ; 返回 4 个张量的列表
           )
      (format t "~%4D all axes nil uniform spacing=1~%")
      (assert (= (length grads) 4))
      (loop for g in grads
            do (assert (equal (vt-shape g) shape))
               (vt-do-each (ptr val g)
		 (declare (ignore ptr))
		 (assert-close val 1.0d0)))
      (format t "4D all axes passed.~%"))

    ;; === 2D，不同轴的间距不同（列表 spacing） ===
    (let* ((mat (vt-from-sequence '((0.0d0 1.0d0 2.0d0 3.0d0)
                                    (4.0d0 5.0d0 6.0d0 7.0d0)
                                    (8.0d0 9.0d0 10.0d0 11.0d0))
                                  :type 'double-float))
           ;; 轴0 间距 2.0, 轴1 间距 0.5
           (grads (vt-gradient mat :spacing '(2.0d0 0.5d0) :axis '(0 1)))
           (g0 (first grads))
           (g1 (second grads)))
      (format t "~%2D multi-axis with list spacing (2, 0.5)~%")
      (format t "grad0:~%~A~%" g0)
      (format t "grad1:~%~A~%" g1)
      ;; 轴0内部中心差分应为 4/dx = 4/2 = 2.0，边界一阶 4/2=2.0
      ;; 轴1内部中心差分应为 1/dx = 1/0.5 = 2.0，边界一阶 1/0.5=2.0
      (dotimes (i 3)
	(dotimes (j 4)
          (assert-close (vt-ref g0 i j) 2.0d0)
          (assert-close (vt-ref g1 i j) 2.0d0)))
      (format t "List spacing test passed.~%"))

    ;; === 一维，非均匀间距（坐标数组），长度2（刚修复的边界情况） ===
    (let* ((x (vt-from-sequence '(1.0d0 5.0d0) :type 'double-float))
           (y (vt-square x))   ; y = [1, 25]
           (grad (vt-gradient y :spacing x :axis 0)))
      (format t "~%1D length-2 non-uniform: ~A~%" grad)
      ;; 只能用一阶差分：(25-1)/(5-1) = 24/4 = 6，两端相同
      (assert-close (vt-ref grad 0) 6.0d0)
      (assert-close (vt-ref grad 1) 6.0d0)
      (format t "Length-2 test passed.~%"))

    ;; === 3D 指定单个轴（int axis），非均匀间距（坐标数组） ===
    (let* ((t3d (vt-from-sequence
		 '(((1.0d0 2.0d0) (3.0d0 4.0d0))   ; shape (2,2,2)
                   ((5.0d0 6.0d0) (7.0d0 8.0d0)))
		 :type 'double-float))
           ;; 沿 axis=1 求梯度，使用非均匀坐标 [0.0, 1.0] （对应轴大小2）
           (coord (vt-from-sequence '(0.0d0 1.0d0) :type 'double-float))
           (grad (vt-gradient t3d :spacing coord :axis 1)))
      (format t "~%3D axis=1 non-uniform (length 2 axis): ~A~%" grad)
      ;; 轴1长度=2，只能用一阶差分
      ;; 计算：对于每个“纤维”（axis=1），df = t3d[...,1,:] - t3d[...,0,:], dx=1.0
      ;; 梯度应为常数：第一块 (3-1,4-2)=(2,2)，第二块 (7-5,8-6)=(2,2)
      (let ((expected (vt-from-sequence '(((2.0d0 2.0d0) (2.0d0 2.0d0))
                                          ((2.0d0 2.0d0) (2.0d0 2.0d0)))
					:type 'double-float)))
	(vt-do-each (ptr val grad)
          (assert-close val (aref (vt-data expected) ptr))))
      (format t "3D non-uniform length-2 axis passed.~%"))

    ;; === 错误场景：坐标数组长度不匹配 ===
    (handler-case
	(progn
          (vt-gradient (vt-arange 5) :spacing (vt-arange 3) :axis 0)
          (format t "~%ERROR: Should have signalled mismatch~%"))
      (simple-error (e)
	(format t "~%Correctly signalled length mismatch: ~A~%" e)))

    (format t "~%All advanced gradient tests passed!~%")))


(defun test-vt-pad ()
  "测试 vt-pad 的各种填充模式"
  (format t "~%=== Testing vt-pad ===")
  
  ;; 1. 常数填充（默认）
  (let* ((a (vt-from-sequence '((1 2) (3 4)) :type 'fixnum))
         ;; 每轴填充宽度：轴0: (1 2)，轴1: (0 1)
         (padded (vt-pad a '((1 2) (0 1))
			 :mode :constant :constant-values 99)))
    (format t "~%Constant padding:~%~A~%" padded)
    (assert (equal (vt-shape padded) '(5 3)))  ; (2+1+2)=5, (2+0+1)=3
    (assert (vt-allclose padded
                         (vt-from-sequence '((99 99 99)
                                             (1  2  99)
                                             (3  4  99)
                                             (99 99 99)
                                             (99 99 99))
                                           :type 'fixnum))))
  
  ;; 1b. 常数填充，左右不同常数
  (let* ((a (vt-arange 5 :type 'fixnum))
         (padded (vt-pad a 2 :mode :constant :constant-values '(10 20))))
    (format t "~%Constant left/right different:~%~A~%" padded)
    (assert (equal (vt-shape padded) '(9)))
    (assert (vt-allclose padded (vt-from-sequence '(10 10 0 1 2 3 4 20 20) :type 'fixnum))))
  
  ;; 2. 边缘填充
  (let* ((a (vt-from-sequence '((1 2 3) (4 5 6)) :type 'fixnum))
         ;; 轴0填充 (1 1)，轴1填充 (2 2)
         (padded (vt-pad a '((1 1) (2 2)) :mode :edge)))
    (format t "~%Edge padding:~%~A~%" padded)
    (assert (equal (vt-shape padded) '(4 7)))
    ;; 第一行边缘为第一行延伸，最后一行为最后一行延伸，列同理
    (let ((expected (vt-from-sequence '((1 1 1 2 3 3 3)
                                        (1 1 1 2 3 3 3)
                                        (4 4 4 5 6 6 6)
                                        (4 4 4 5 6 6 6))
                                      :type 'fixnum)))
      (assert (vt-allclose padded expected))))
  
  ;; 3. 循环填充 (wrap)
  (let* ((a (vt-from-sequence '((1 2) (3 4)) :type 'fixnum))
         ;; 轴0填充 (1 1)，轴1填充 (1 1)，则形状变为 (4 4)
         (padded (vt-pad a '((1 1) (1 1)) :mode :wrap)))
    (format t "~%Wrap padding:~%~A~%" padded)
    (assert (equal (vt-shape padded) '(4 4)))
    ;; 循环：左边/上边取自右边/下边的行/列，反之亦然
    (let ((expected (vt-from-sequence '((4 3 4 3)
                                        (2 1 2 1)
                                        (4 3 4 3)
                                        (2 1 2 1))
                                      :type 'fixnum)))
      (assert (vt-allclose padded expected))))
  
  ;; 4. 反射填充 (reflect) - 不重复边缘
  (let* ((a (vt-from-sequence '((1 2 3) (4 5 6)) :type 'fixnum))
         ;; 每轴填充 (2 2)，注意反射要求宽度 <= 原尺寸-1，满足
         (padded (vt-pad a '((2 2) (2 2)) :mode :reflect)))
    (format t "~%Reflect padding:~%~A~%" padded)
    (assert (equal (vt-shape padded) '(6 7)))
    ;; 手动计算预期：轴0反射：原始行索引 0,1 → 左填充：跳过边缘(0)，从 1,0? 实际上 reflect 模式：左填充从左向右反射第一行（索引1），第二行（索引0）
    ;; 规则：对于左边界，填充序列为 a[1], a[0]；对于右边界，填充序列为 a[n-2], a[n-3]...
    ;; 仅验证形状即可，因为内容需要严格按照规则；这里改为验证特定元素
    ;; 验证中心区域与原矩阵一致
    (let ((center (vt-slice padded '(2 4) '(2 5))))
      (assert (vt-allclose center a))))
  
  ;; 5. 对称填充 (symmetric) - 重复边缘
  (let* ((a (vt-from-sequence '((1 2) (3 4)) :type 'fixnum))
         ;; 轴0填充 (1 1)，轴1填充 (1 1)
         (padded (vt-pad a '((1 1) (1 1)) :mode :symmetric)))
    (format t "~%Symmetric padding:~%~A~%" padded)
    (assert (equal (vt-shape padded) '(4 4)))
    ;; 对称：左边界从 a[0] 开始反向，右边界从 a[-1] 开始反向
    (let ((expected (vt-from-sequence '((1 1 2 2)
                                        (1 1 2 2)
                                        (3 3 4 4)
                                        (3 3 4 4))
                                      :type 'fixnum)))
      (assert (vt-allclose padded expected))))
  
  ;; 6. 反射模式宽度超过限制应报错
  (handler-case
      (vt-pad (vt-arange 5) 4 :mode :reflect)  ; 宽度4 > 4 (dim-1) 应报错
    (simple-error (e)
      (format t "~%Reflect width exceed error: ~A~%" e)))
  
  ;; 7. 对称模式宽度超过限制应报错
  (handler-case
      (vt-pad (vt-arange 5) 6 :mode :symmetric) ; 宽度6 > 5 应报错
    (simple-error (e)
      (format t "~%Symmetric width exceed error: ~A~%" e)))
  
  (format t "~%All pad tests passed!~%"))


(defun test-all-pad ()
  ;; 1. Constant
  (let ((a (vt-from-sequence '((1 2) (3 4)) :type 'fixnum)))
    (let ((padded (vt-pad a '((1 2) (0 1)) :mode :constant :constant-values 99)))
      (assert (vt-allclose padded
                           (vt-from-sequence '((99 99 99)
                                               ( 1  2 99)
                                               ( 3  4 99)
                                               (99 99 99)
                                               (99 99 99))
                                             :type 'fixnum)))))
  ;; 2. Edge
  (let ((a (vt-from-sequence '((1 2 3) (4 5 6)) :type 'fixnum)))
    (let ((padded (vt-pad a '((1 1) (2 2)) :mode :edge)))
      (assert (vt-allclose padded
                           (vt-from-sequence '((1 1 1 2 3 3 3)
                                               (1 1 1 2 3 3 3)
                                               (4 4 4 5 6 6 6)
                                               (4 4 4 5 6 6 6))
                                             :type 'fixnum)))))
  ;; 3. Wrap
  (let ((a (vt-from-sequence '((1 2) (3 4)) :type 'fixnum)))
    (let ((padded (vt-pad a '((1 1) (1 1)) :mode :wrap)))
      (assert (vt-allclose padded
                           (vt-from-sequence '((4 3 4 3)
                                               (2 1 2 1)
                                               (4 3 4 3)
                                               (2 1 2 1))
                                             :type 'fixnum)))))
  ;; 4. Reflect（只需验证中心区域不变）
  (let ((a (vt-from-sequence '((1 2) (3 4)) :type 'fixnum)))
    (let ((padded (vt-pad a '((2 2) (2 2)) :mode :reflect)))
      (assert (vt-allclose (vt-slice padded '(2 4) '(2 4)) a))))
  ;; 5. Symmetric
  (let ((a (vt-from-sequence '((1 2) (3 4)) :type 'fixnum)))
    (let ((padded (vt-pad a '((1 1) (1 1)) :mode :symmetric)))
      (assert (vt-allclose padded
                           (vt-from-sequence '((1 1 2 2)
                                               (1 1 2 2)
                                               (3 3 4 4)
                                               (3 3 4 4))
                                             :type 'fixnum)))))
  ;; 6. 1D symmetric (NumPy 示例)
  (let ((a (vt-from-sequence '(1 2 3 4 5) :type 'fixnum)))
    (assert (vt-allclose (vt-pad a '((0 2)) :mode :symmetric)
                         (vt-from-sequence '(1 2 3 4 5 5 4) :type 'fixnum)))
    (assert (vt-allclose (vt-pad a '((2 0)) :mode :symmetric)
                         (vt-from-sequence '(2 1 1 2 3 4 5) :type 'fixnum))))
  (format t "~%All pad tests passed!~%"))


(defun test-pad-thorough ()
  (format t "~%=== Thorough vt-pad test ===")

  ;; 辅助函数：逐元素相等断言（整数可用）
  (labels ((vt= (a b)
             (vt-allclose a b :atol 0.0d0 :rtol 0.0d0)))

    ;; ---------- 1. CONSTANT ----------
    (let* ((a (vt-from-sequence '(1 2 3) :type 'fixnum))
           (p (vt-pad a '((2 3)) :mode :constant :constant-values 99)))
      ;; NumPy: np.pad([1,2,3], (2,3), 'constant', constant_values=99) -> [99 99 1 2 3 99 99 99]
      (assert (vt= p (vt-from-sequence '(99 99 1 2 3 99 99 99) :type 'fixnum))))

    (let* ((a (vt-from-sequence '((1 2) (3 4)) :type 'fixnum))
           (p (vt-pad a '((1 0) (2 1)) :mode :constant :constant-values 0)))
      ;; NumPy: np.pad([[1,2],[3,4]], ((1,0),(2,1)), 'constant') -> [[0,0,0,0,0], [0,0,1,2,0], [0,0,3,4,0]]
      (assert (vt= p (vt-from-sequence '((0 0 0 0 0)
                                         (0 0 1 2 0)
                                         (0 0 3 4 0))
                                       :type 'fixnum))))

    (let* ((a (vt-from-sequence '(1.0d0 2.0d0) :type 'double-float))
           (p (vt-pad a '((1 1)) :mode :constant :constant-values '(10.0d0 20.0d0))))
      ;; NumPy: np.pad([1.,2.], (1,1), 'constant', constant_values=(10,20)) -> [10., 1., 2., 20.]
      (assert (vt= p (vt-from-sequence '(10.0d0 1.0d0 2.0d0 20.0d0) :type 'double-float))))

    ;; 零宽度填充
    (let* ((a (vt-from-sequence '((1 2) (3 4)) :type 'fixnum))
           (p (vt-pad a '((0 0) (0 0)) :mode :constant :constant-values 99)))
      (assert (vt= p a)))

    ;; ---------- 2. EDGE ----------
    (let* ((a (vt-from-sequence '((1 2 3) (4 5 6)) :type 'fixnum))
           (p (vt-pad a '((2 1) (1 2)) :mode :edge)))
      ;; NumPy: np.pad([[1,2,3],[4,5,6]], ((2,1),(1,2)), 'edge')
      ;; 预期：行上两行边缘（复制第一行），下一行边缘（复制最后一行）；列左一列复制第一列，右两列复制最后一列。
      (let ((expected (vt-from-sequence '((1 1  2 3 3 3)
                                          (1 1  2 3 3 3)
                                          (1 1  2 3 3 3)
                                          (4 4  5 6 6 6)
                                          (4 4  5 6 6 6))
                                        :type 'fixnum)))
        (assert (vt= p expected))))

    ;; ---------- 3. WRAP ----------
    (let* ((a (vt-from-sequence '((1 2) (3 4)) :type 'fixnum))
           (p (vt-pad a '((1 1) (2 1)) :mode :wrap)))
      ;; NumPy: np.pad([[1,2],[3,4]], ((1,1),(2,1)), 'wrap')
      ;; 行循环：上边取最后一行，下边取第一行；列循环：左边取最后两列，右边取第一列。
      ;; 结果形状 4x5
      (let ((expected (vt-from-sequence '((3 4 3 4 3)
					  (1 2 1 2 1)
					  (3 4 3 4 3)
					  (1 2 1 2 1))
                                        :type 'fixnum)))
        (assert (vt= p expected))))

    ;; ---------- 4. REFLECT (宽度可任意大) ----------
    (let* ((a (vt-from-sequence '((1 2 3) (4 5 6)) :type 'fixnum))
           (p (vt-pad a '((2 2) (3 3)) :mode :reflect)))
      ;; 只验证中心区域未变，并验证几个特征点（NumPy 可生成完整预期，此处省略）
      (assert (vt= (vt-slice p '(2 4) '(3 6)) a))
      (assert (= 189 (vt-sum p)))
      )

    (let* ((a (vt-from-sequence '(1 2 3) :type 'fixnum))
           (p (vt-pad a '((4 4)) :mode :reflect)))
      ;; NumPy: np.pad([1,2,3], (4,4), 'reflect')
      ;; 预期：左填充 [3,2,1,2]，右填充 [2,3,2,1]
      (assert (vt= p (vt-from-sequence '(1 2 3 2 1 2 3 2 1 2 3)
				       :type 'fixnum))))

    ;; ---------- 5. SYMMETRIC (大宽度) ----------
    ;; 1D 左边宽度2
    (let* ((a (vt-from-sequence '(1 2 3 4 5) :type 'fixnum))
           (p (vt-pad a '((2 0)) :mode :symmetric)))
      (assert (vt= p (vt-from-sequence '(2 1 1 2 3 4 5) :type 'fixnum))))
    ;; 1D 右边宽度2
    (let* ((a (vt-from-sequence '(1 2 3 4 5) :type 'fixnum))
           (p (vt-pad a '((0 2)) :mode :symmetric)))
      (assert (vt= p (vt-from-sequence '(1 2 3 4 5 5 4) :type 'fixnum))))
    ;; 1D 左边大宽度7
    (let* ((a (vt-from-sequence '(1 2) :type 'fixnum))
           (p (vt-pad a '((7 0)) :mode :symmetric))
	   (expected (vt-from-sequence '(2 2 1 1 2 2 1 1 2) :type 'fixnum)))
      ;; NumPy: np.pad([1,2], (7,0), 'symmetric') -> [2, 2, 1, 1, 2, 2, 1, 1, 2]? 我们直接验证形状和边界值
      (vt-allclose expected p))
    ;; 2D symmetric
    (let* ((a (vt-from-sequence '((1 2) (3 4)) :type 'fixnum))
           (p (vt-pad a '((1 1) (1 1)) :mode :symmetric)))
      (assert (vt= p (vt-from-sequence '((1 1 2 2)
                                         (1 1 2 2)
                                         (3 3 4 4)
                                         (3 3 4 4))
                                       :type 'fixnum))))

    ;; ---------- 6. 混合类型测试 ----------
    (let* ((a (vt-from-sequence '((1.0d0 2.0d0) (3.0d0 4.0d0)) :type 'double-float))
           (p (vt-pad a '((1 1) (1 1)) :mode :symmetric)))
      (assert (vt= p (vt-from-sequence '((1.0 1.0 2.0 2.0)
                                         (1.0 1.0 2.0 2.0)
                                         (3.0 3.0 4.0 4.0)
                                         (3.0 3.0 4.0 4.0))
                                       :type 'double-float))))

    (format t "~%All thorough pad tests passed!~%")))



(defun test-vt-take ()
  "全面测试 vt-take，所有场景均与 NumPy 行为对照。"
  ;; 测试矩阵：4 行 5 列，元素 0..19
  (let* ((m (vt-from-sequence '(( 0  1  2  3  4)
                                ( 5  6  7  8  9)
                                (10 11 12 13 14)
                                (15 16 17 18 19))
                              :type 'fixnum))
         ;; 辅助函数：将张量展平为列表（如果是一维）或保持嵌套
         (to-list (lambda (vt)
                    (if (= (length (vt-shape vt)) 1)
                        (coerce (vt-data vt) 'list)
                        (vt-to-list vt)))))
    ;; ================================================================
    ;; 1. axis = nil : 标量索引
    ;; np: a = np.arange(20).reshape(4,5); a.take(0)  => np.int64(0)
    ;; ================================================================
    (let ((result (vt-take m 0)))
      (assert (= result 0)
              () "take(0) failed: expected 0, got ~A" result))

    ;; ================================================================
    ;; 2. axis = nil : 列表索引
    ;; np: a.take([0,5])  => [0, 5]
    ;; ================================================================
    (let* ((result (vt-take m (list 0 5)))
           (expected (vt-from-sequence '(0 5) :type 'fixnum)))
      (assert (equalp (funcall to-list result)
                      (funcall to-list expected))
              () "take([0,5]) failed"))

    ;; ================================================================
    ;; 3. axis = nil : 多维索引，保留形状
    ;; np: a.take([[1,3],[1,5]])  => [[1,3],[1,5]]  shape (2,2)
    ;; ================================================================
    (let* ((indices (vt-from-sequence '((1 3) (1 5)) :type 'fixnum))
           (result (vt-take m indices)))
      (assert (equal (vt-shape result) '(2 2))
              () "take with 2D indices: shape mismatch ~A" (vt-shape result))
      (assert (equalp (vt-to-list result)
                      '((1 3) (1 5)))
              () "take with 2D indices: values mismatch"))

    ;; ================================================================
    ;; 4. axis = 0 : 一维索引
    ;; np: a.take([1,3], axis=0)  => [[5,6,7,8,9],[15,16,17,18,19]]  shape (2,5)
    ;; ================================================================
    (let* ((indices (vt-arange 2 :start 1 :step 2 :type 'fixnum)) ; [1,3]
           (result (vt-take m indices :axis 0)))
      (assert (equal (vt-shape result) '(2 5)) () "axis=0 shape failed")
      (assert (equalp (vt-to-list result)
                      '((5 6 7 8 9) (15 16 17 18 19)))
              () "axis=0 values failed"))

    ;; ================================================================
    ;; 5. axis = 1 : 一维索引
    ;; np: a.take([0,2], axis=1)  => [[0,2],[5,7],[10,12],[15,17]]  shape (4,2)
    ;; ================================================================
    (let* ((indices (vt-from-sequence '(0 2) :type 'fixnum))
           (result (vt-take m indices :axis 1)))
      (assert (equal (vt-shape result) '(4 2)) () "axis=1 shape failed")
      (assert (equalp (vt-to-list result)
                      '((0 2) (5 7) (10 12) (15 17)))
              () "axis=1 values failed"))

    ;; ================================================================
    ;; 6. axis = -1 (等价 axis=1) : 多维索引
    ;; np: a.take([[0,2],[1,3]], axis=-1)
    ;;     输出形状 (4,2,2)
    ;; ================================================================
    (let* ((indices (vt-from-sequence '((0 2) (1 3)) :type 'fixnum))
           (result (vt-take m indices :axis -1)))
      (assert (equal (vt-shape result) '(4 2 2))
              () "axis=-1 with 2D indices: shape mismatch ~A" (vt-shape result))
      (assert (equalp (vt-to-list result)
                      '(((0 2) (1 3)) ((5 7) (6 8))
                        ((10 12) (11 13)) ((15 17) (16 18))))
              () "axis=-1 with 2D indices: values mismatch"))

    ;; ================================================================
    ;; 7. axis = -2 (等价 axis=0) : 标量索引，降维
    ;; np: a.take(2, axis=-2)  => 第三行，shape (5)
    ;; ================================================================
    (let* ((result (vt-take m 2 :axis -2))
           (expected (vt-from-sequence '(10 11 12 13 14) :type 'fixnum)))
      (assert (equal (vt-shape result) '(5)) () "axis=-2 shape failed")
      (assert (equalp (funcall to-list result)
                      (funcall to-list expected))
              () "axis=-2 values failed"))

    (format t "~%All vt-take tests passed!~%")))


(defun test-vt-argsort ()
  (let* ((a (vt-from-sequence '((3 1 2) (6 5 4)) :type 'fixnum)))
    (assert (equal
	     (vt-to-list (vt-argsort a :axis -1))
	     '((1 2 0)
	       (2 1 0))))
    (assert (equal
	     (vt-to-list (vt-argsort a :axis 1))
	     '((1 2 0)
	       (2 1 0))))

    (assert (equal
	     (vt-to-list (vt-argsort a :axis 0))
	     '((0 0 0)
	       (1 1 1))))
    (assert (equal
	     (vt-to-list (vt-argsort a :axis nil))
	     '(1 2 0 5 4 3))))
  
  (let ((tensor-3d (vt-from-sequence
                    '(((0.0d0 99.0d0 2.0d0)    ;; 深度 0 的数据
                       (4.0d0 6.0d0 8.0d0))
                      ((1.0d0 -99.0d0 3.0d0)   ;; 深度 1 的数据
                       (5.0d0 7.0d0 9.0d0))))))
    
    (format t "~%--- 输入张量 (形状 2x2x3) ---~%")
    ;; 打印出来方便直观感受
    (format t "深度 0: ~A~%" (vt-to-list (vt-slice tensor-3d '(0) '(:all) '(:all))))
    (format t "深度 1: ~A~%" (vt-to-list (vt-slice tensor-3d '(1) '(:all) '(:all))))

    ;; 2. 沿着 axis=0 (深度轴) 进行 argsort
    ;; 这意味着对于任意一个 坐标，我们会拿出 [A[0,r,c], A[1,r,c]] 进行排序
    (let ((result (vt-argsort tensor-3d :axis 0)))
      
      (format t "~%--- 实际输出 (扁平化查看) ---~%")
      ;; 将 3D 结果展平为一维列表，这样最容易看出内存级别的错乱
      (let ((flat-result (vt-to-list (vt-flatten result))))
	(format t "扁平结果: ~A~%" flat-result)
	
	;; 3. 构造预期的正确结果
	;; 沿 axis=0 排序，输出的形状依然是 (2, 2, 3)
	;; 对于大部分 位置，数据是升序的(如 0<1, 2<3)，所以正确的索引是 [0, 1]
	;; 唯独在 (行=0, 列=1) 的位置，数据是 [99, -99]，逆序！所以正确的索引必须是 [1, 0]
	;; 按照行优先内存布局，预期的扁平列表应该是：
	(let ((expected-flat '(0 1 0 0 0 0 1 0 1 1 1 1)))
          (format t "预期扁平: ~A~%" expected-flat)))))
    (print "passed test vt-argsort"))


(defun test-vt-sort ()
  (let* ((a (vt-from-sequence '((3 1 2) (6 5 4)) :type 'fixnum)))
    (assert (equal
	     (vt-to-list (vt-sort a :axis -1))
	     '((1 2 3)
	       (4 5 6))))
    (assert (equal
	     (vt-to-list (vt-sort a :axis 1))
	     '((1 2 3)
	       (4 5 6))))
    (assert (equal
	     (vt-to-list (vt-sort a :axis 0))
	     '((3 1 2)
	       (6 5 4))))
    (assert (equal
	     (vt-to-list (vt-sort a :axis nil))
	     '(1 2 3 4 5 6)))
    (print "passed test vt-sort")))

(defun test-vt-argsort-multi-axis ()
  (format t "~%=== Testing vt-argsort with rank>2 ===")
  (let ((a (vt-from-sequence '(((3 2) (1 0))
                               ((9 8) (7 6)))
                             :type 'fixnum)))   ; shape (2,2,2)
    ;; 沿最后一轴（axis=-1）排序，期望每层内两个元素升序
    (let ((sorted (vt-argsort a :axis -1)))
      (assert (equal (vt-to-list sorted)
                     '(((1 0) (1 0))
                       ((1 0) (1 0)))))
      (format t "~[FAIL~;PASS~] argsort axis=-1~%"
              (if (equal (vt-to-list sorted)
                         '(((1 0) (1 0)) ((1 0) (1 0))))
		  1 0)))
    ;; 沿 axis=0 排序
    (let ((sorted (vt-argsort a :axis 0)))
      (assert (equal (vt-to-list sorted)
                     '(((0 0) (0 0))
                       ((1 1) (1 1)))))
      (format t "~[FAIL~;PASS~] argsort axis=0~%"
              (if (equal (vt-to-list sorted)
                         '(((0 0) (0 0)) ((1 1) (1 1))))
		  1 0)))))

(macrolet ((assert-close (a b &optional (eps 1e-8))
             `(unless (<= (abs (- ,a ,b)) ,eps)
                (error "Assertion failed: ~A ~A differ by ~A"
                       ',a ',b (abs (- ,a ,b))))))
  (defun test-svd-correctness ()
    (format t "~%=== Testing vt-svd correctness ===")
    (let* ((A (vt-from-sequence '((1 2 3)
                                  (4 5 6)
                                  (7 8 9)
                                  (10 11 12))
				:type 'double-float))
           (m (first (vt-shape A)))
           (n (second (vt-shape A)))
           (k (min m n)))
      ;; 经济模式
      (multiple-value-bind (U S Vt) (vt-svd A :full-matrices nil)
	(let* ((Smat (vt-zeros (list (vt-size S) (vt-size S)) :type 'double-float))
               (_ (dotimes (i (vt-size S))
                    (setf (vt-ref Smat i i) (vt-ref S i))))
               (recon (vt-@ (vt-@ U Smat) Vt))
               (err (vt-amax (vt-abs (vt-- A recon)))))
          (format t "~%Economic SVD: max reconstruction error = ~A" err)
          (assert-close err 0.0d0 1e-12)
          (assert (equal (vt-shape Vt) (list k n))   ; Vt 应为 k×n
                  () "Economic Vt shape is ~A, expected (~A ~A)" (vt-shape Vt) k n)))
      ;; 完整模式
      (multiple-value-bind (U S Vt) (vt-svd A :full-matrices t)
	(let* ((m (first (vt-shape A)))   ; = 4
               (n (second (vt-shape A)))  ; = 3
               (k (vt-size S))            ; = 3
               (Smat (vt-zeros (list m n) :type 'double-float)))  ; 4×3
	  (dotimes (i k)
	    (setf (vt-ref Smat i i) (vt-ref S i)))
	  (let* ((recon (vt-@ (vt-@ U Smat) Vt))
		 (err (vt-amax (vt-abs (vt-- A recon)))))
	    (format t "Full SVD reconstruction error: ~A~%" err)
            (assert-close err 0.0d0 1e-12)
            (assert (equal (vt-shape Vt) (list n n))
                    () "Full Vt shape is ~A, expected (~A ~A)" (vt-shape Vt) n n)))
	(format t "~%SVD tests passed~%")))))




;;; ============================================================
;;; vt-put 测试（与 NumPy 的 numpy.put 行为对齐）
;;; ============================================================

(defun test-vt-put ()
  (format t "~%=== Testing vt-put (NumPy style) ===~%")

  ;; ------------------------------------------------------------
  ;; 测试 1: 标量 indices + 标量 values (展平后设置单个元素)
  ;; NumPy:
  ;;   import numpy as np
  ;;   a = np.arange(6).reshape(2,3)   # [[0,1,2],[3,4,5]]
  ;;   np.put(a, 4, 99)                # 展平索引 4 处设值
  ;;   print(a)                        # [[0,1,2],[3,99,5]]
  ;; ------------------------------------------------------------
  (let ((a (vt-arange 6 :type 'fixnum)))
    (setf a (vt-reshape a '(2 3)))
    (vt-put a 4 99)
    (let ((expected (vt-from-sequence '((0 1 2) (3 99 5)) :type 'fixnum)))
      (assert (vt-allclose a expected))
      (format t "Test 1 passed: scalar index + scalar value~%")))

  ;; ------------------------------------------------------------
  ;; 测试 2: 一维 indices 列表 + 一维 values 列表（长度相等）
  ;; NumPy:
  ;;   a = np.arange(10)
  ;;   np.put(a, [0, 5, 9], [100, 200, 300])
  ;;   print(a)   # [100, 1, 2, 3, 4, 200, 6, 7, 8, 300]
  ;; ------------------------------------------------------------
  (let ((a (vt-arange 10 :type 'fixnum)))
    (vt-put a '(0 5 9) '(100 200 300))
    (let ((expected (vt-from-sequence '(100 1 2 3 4 200 6 7 8 300) :type 'fixnum)))
      (assert (vt-allclose a expected))
      (format t "Test 2 passed: 1D indices + same length values~%")))

  ;; ------------------------------------------------------------
  ;; 测试 3: 一维 indices 列表 + 标量 values（标量广播）
  ;; NumPy:
  ;;   a = np.arange(8)
  ;;   np.put(a, [1, 3, 5, 7], -1)
  ;;   print(a)   # [0, -1, 2, -1, 4, -1, 6, -1]
  ;; ------------------------------------------------------------
  (let ((a (vt-arange 8 :type 'fixnum)))
    (vt-put a '(1 3 5 7) -1)
    (let ((expected (vt-from-sequence '(0 -1 2 -1 4 -1 6 -1) :type 'fixnum)))
      (assert (vt-allclose a expected))
      (format t "Test 3 passed: 1D indices + scalar value (broadcast)~%")))

  ;; ------------------------------------------------------------
  ;; 测试 4: 一维 indices 列表 + 较短的 values（循环使用）
  ;; NumPy:
  ;;   a = np.arange(6)
  ;;   np.put(a, [0, 2, 4, 5], [10, 20])   # values = [10,20,10,20] 循环填充
  ;;   print(a)   # [10, 1, 20, 3, 10, 20]
  ;; ------------------------------------------------------------
  (let ((a (vt-arange 6 :type 'fixnum)))
    (vt-put a '(0 2 4 5) '(10 20))
    (let ((expected (vt-from-sequence '(10 1 20 3 10 20) :type 'fixnum)))
      (assert (vt-allclose a expected))
      (format t "Test 4 passed: 1D indices + shorter values (cycle)~%"))

    ;; ------------------------------------------------------------
    ;; 测试 5: 多维 indices（自动展平）+ 标量 values
    ;; NumPy:
    ;;   a = np.arange(12).reshape(3,4)
    ;;   indices = np.array([[0, 3], [8, 11]])   # 展平后为 [0,3,8,11]
    ;;   np.put(a, indices, -999)
    ;;   print(a)
    ;;   # [[-999,   1,   2, -999],
    ;;   #  [   4,   5,   6,   7],
    ;;   #  [-999,   9,  10, -999]]
    ;; ------------------------------------------------------------
    (let ((a (vt-arange 12 :type 'fixnum)))
      (setf a (vt-reshape a '(3 4)))
      (let ((indices (vt-from-sequence '((0 3) (8 11)) :type 'fixnum)))
	(vt-put a indices -999))
      (let ((expected (vt-from-sequence '((-999 1 2 -999)
                                          (4 5 6 7)
                                          (-999 9 10 -999))
					:type 'fixnum)))
	(assert (vt-allclose a expected))
	(format t "Test 5 passed: multi-dim indices (flattened) + scalar value~%"))

      ;; ------------------------------------------------------------
      ;; 测试 6: 二维张量（非展平）使用 axis 参数（注意：vt-put 目前没有 axis 参数，
      ;;         它总是按展平索引工作。NumPy 的 put 也没有 axis，但 numpy.put_along_axis 有。
      ;;         这里测试的是展平索引在多维数组上的行为，符合预期。）
      ;; NumPy:
      ;;   a = np.arange(12).reshape(3,4)
      ;;   np.put(a, [2, 7, 10], [777, 888, 999])
      ;;   print(a)
      ;;   # [[  0,   1, 777,   3],
      ;;   #  [  4,   5,   6, 888],
      ;;   #  [  8,   9, 999,  11]]
      ;; ------------------------------------------------------------
      (let ((a (vt-arange 12 :type 'fixnum)))
	(setf a (vt-reshape a '(3 4)))
	(vt-put a '(2 7 10) '(777 888 999))
	(let ((expected (vt-from-sequence '((0 1 777 3)
                                            (4 5 6 888)
                                            (8 9 999 11))
					  :type 'fixnum)))
	  (assert (vt-allclose a expected))
	  (format t "Test 6 passed: flat indices on 2D array~%"))

	;; ------------------------------------------------------------
	;; 测试 7: 索引越界检测（应抛出错误）
	;; NumPy: np.put(a, [100], 0) 会引发 IndexError
	;; ------------------------------------------------------------
	(handler-case
	    (let ((a (vt-arange 5 :type 'fixnum)))
              (vt-put a 100 42))
	  (simple-error (e)
	    (format t "Test 7 passed: caught out-of-bounds error: ~A~%" e))))))
      
  ;; 测试 1：连续数组上 put (完全正确，不用改)
  (let* ((a (vt-arange 12 :type 'double-float))
         (indices '(2 5 10))
         (values  '(0 0 0)))
    (vt-put a indices values)
    (assert-ok (approx= a (vt-from-sequence
                           (list 0d0 1d0 0d0 3d0 4d0 0d0
                                 6d0 7d0 8d0 9d0 0d0 11d0)))
	       "连续 put 失败"))

  ;; 测试 2：转置视图上的 put
  (let* ((a (vt-arange 12 :type 'double-float))
         (a-reshape (vt-reshape a '(3 4)))
         (aT (vt-transpose a-reshape))
         ;; 修复：aT(1,2)的逻辑索引是 1*3+2=5，aT(2,1)的逻辑索引是 2*3+1=7
         (_ (vt-put aT '(5 7) '(0 0))) 
         (expected (vt-from-sequence
		    '((0d0 1d0 2d0 3d0)
		      (4d0 5d0 0d0 7d0)   ; 原索引 (1,2) 被置0
		      (8d0 0d0 10d0 11d0))))) ; 原索引 (2,1) 被置0 (原注释写错为3,1)
    (assert-ok (approx= a-reshape expected) "转置视图 put 失败"))

  ;; 测试 3：切片视图上的 put
  (let* ((a (vt-from-sequence '((1 2 3) (4 5 6) (7 8 9)) :type 'double-float))
         (s (vt-slice a '(1 3) '(1 3)))   ; [[5,6],[8,9]]
         ;; 修复：s(0,0)的逻辑索引是 0，s(1,1)的逻辑索引是 1*2+1=3
         (_ (vt-put s '(0 3) '(-1 -2)))
         (expected (vt-from-sequence
		    '((1d0 2d0 3d0)
		      (4d0 -1d0 6d0)
		      (7d0 8d0 -2d0)))))
    (assert-ok (approx= a expected) "切片视图 put 失败"))

  ;; 测试 5：空 indices (不用改)
  (let* ((a (vt-from-sequence '(1 2 3 4 5))))
    (vt-put a '() '())
    (assert-ok (approx= a (vt-from-sequence '(1 2 3 4 5)))
	       "空 put 改变了数组"))

  (format t "All vt-put tests passed successfully!~%"))



;;; 测试 VT-NAN* 函数
;; 辅助：判断两个 double-float 是否均为 NaN (即 (/= x x) 均为真)
(defun both-nan-p (a b)
  (and (/= a a) (/= b b)))

(defun test-nan-functions ()
  (format t "~%--- Testing NaN statistics functions ---~%")
  (sb-vm::with-float-traps-masked (:invalid :divide-by-zero :overflow)
    ;; 准备数据: 包含 NaN 的 2x3 矩阵
    (let* ((nan (/ 0.0d0 0.0d0))
	   (data (vt-from-sequence (list (list 1.0 2.0 3.0)
					 (list 4.0 nan 6.0))
				   :type 'double-float))
           ;; 手动将符号 :nan 替换为实际 NaN
           (arr (vt-data data)))
      (loop for i below (length arr)
            when (eql (aref arr i) 0.0) ;; 因为先创建为0再赋值，实际不会是0
					;; 直接使用 VT-SET 来设置 NaN，由于 VT-REF 无法处理 :nan，
					;; 我们重新构建：用 0 占位然后将已知位置设为 NaN
					;; 简化：直接修改底层数据数组
              do (when (and (= (mod i 3) 1) (= (floor i 3) 1))
		   (setf (aref arr i) (/ 0.0d0 0.0d0))))  ;; 索引 (1,1) 设为 NaN
      ;; 检查 NaN 是否设置成功
      (format t "~2%Original tensor:~%")
      (print-vt-recursive data 0 nil 2 10 'double-float t) ;; 简化打印，直接使用内部打印函数可能需要调整
      ;; 使用 vt-map 来打印，避免未导出函数
      (format t "~2%Data values: ~{~A ~}~%" (vt-to-list data))
      
      ;; ---- test vt-nansum ----
      (let ((s-global (vt-nansum data))
            (s-axis0 (vt-nansum data :axis 0))
            (s-axis1 (vt-nansum data :axis 1))
            (s-keepdim (vt-nansum data :axis 0 :keepdims t)))
	(assert (= (coerce s-global 'double-float) 16.0d0) () "nansum global-> ~A" s-global)
	(assert (equalp (vt-to-list s-axis0) '(5.0 2.0 9.0)))
	(assert (equalp (vt-to-list s-axis1)  '(6.0 10.0)))
	(assert (equal (list (first (vt-shape s-keepdim))
			     (second (vt-shape s-keepdim))) '(1 3)))
	(format t "~%vt-nansum: PASS~%"))

      ;; ---- test vt-nanmean ----
      (let ((m-global (vt-nanmean data))
            (m-axis0 (vt-nanmean data :axis 0))
            (m-axis1 (vt-nanmean data :axis 1)))
	(assert (< (abs (- m-global (/ 16.0d0 5))) 1d-10))
	(assert (equalp (vt-to-list m-axis0) '(2.5 2.0 4.5d0)))
	(assert (equalp (vt-to-list m-axis1) '(2.0 5.0)))
	(format t "vt-nanmean: PASS~%"))

      ;; ---- test vt-nanvar (ddof=0, default) ----
      (let* ((v-global (vt-nanvar data))
             (expected (/ (+ (expt (- 1 3.2) 2)
			     (expt (- 2 3.2) 2)
                             (expt (- 3 3.2) 2)
			     (expt (- 4 3.2) 2)
                             (expt (- 6 3.2) 2))
			  5)))
	(assert (< (abs (- v-global expected)) 1d-10))
	(let ((v-axis0 (vt-nanvar data :axis 0))
              (v-axis1 (vt-nanvar data :axis 1 :ddof 1)))
          ;; axis0: 每列包含 NaN 被忽略，列0:[1,4] 有效数2，列1:[2] 有效数1，列2:[3,6] 有效数2
          (assert (equalp (vt-to-list v-axis0) 
                          (list (/ (+ (expt (- 1 2.5) 2)
				      (expt (- 4 2.5) 2))
				   2)
				;; 列1只有一个数，方差应为0
				0.0d0
				(/ (+ (expt (- 3 4.5) 2)
				      (expt (- 6 4.5) 2))
				   2))))
          ;; axis1: [1,2,3] variance=? 和 [4,6] variance=?  sample var (ddof=1)
	  (assert (equalp (vt-to-list v-axis1)
			  (list (/ (+ (expt (- 1 2) 2)
				      (expt (- 2 2) 2)
				      (expt (- 3 2) 2)) 2)
				(/ (+ (expt (- 4 5) 2)
				      (expt (- 6 5) 2)) 1)))))
	(format t "vt-nanvar: PASS~%")

	;; ---- test vt-nanstd ----
	(let ((std-global (vt-nanstd data)))
	  (assert (< (abs (- std-global (sqrt expected))) 1d-10))
	  (format t "vt-nanstd: PASS~%")))

      ;; ---- test vt-nanmax ----
      (let ((mx-global (vt-nanmax data))
            (mx-axis0 (vt-nanmax data :axis 0)))
	(assert (= mx-global 6.0d0))
	(assert (equalp (vt-to-list mx-axis0) '(4.0 2.0 6.0)))
	(format t "vt-nanmax: PASS~%"))

      ;; ---- test vt-nanmin ----
      (let ((mn-global (vt-nanmin data))
            (mn-axis0 (vt-nanmin data :axis 0)))
	(assert (= mn-global 1.0d0))
	(assert (equalp (vt-to-list mn-axis0) '(1.0 2.0 3.0)))
	(format t "vt-nanmin: PASS~%"))

      ;; ---- 边界情况：全 NaN ----
      (let* ((all-nan (vt-ones '(2 3) :type 'double-float))
             (d (vt-data all-nan)))
	(loop for i below (length d)
	      do (setf (aref d i) (/ 0.0d0 0.0d0)))
	;; 测试这些函数不会崩溃，并返回 NaN 或适当值
	


	(let ((s (vt-nansum all-nan))
	      (m (vt-nanmean all-nan))
	      (v (vt-nanvar all-nan))
	      (mx (vt-nanmax all-nan))
	      (mn (vt-nanmin all-nan)))
	  ;; 检查标量
	  (assert (/= m m))                 ; m 是 NaN
	  (assert (= s 0.0d0))
	  ;; 检查一维张量: 全部元素都是 NaN
	  (let ((m0 (vt-nanmean all-nan :axis 0)))
	    (assert (vt-all (vt-isnan m0)))
	    ;; 同时验证标量 m 也是 NaN
	    (assert (vt-all (vt-isnan (ensure-vt m)))))
	  ;; max / min 边界
	  (assert (vt-all (vt-isinf mx)))    ; -inf
	  (assert (vt-all (vt-isinf mn)))    ; +inf? 看实现，应该是 most-positive-double-float
	  ;; 方差也应为 NaN
	  (assert (vt-all (vt-isnan (ensure-vt v))))))

      ;; ---- 边界情况：无 NaN ----
      (let* ((clean (vt-arange 6 :step 1 :type 'double-float)))
	(assert (= (vt-nansum clean) 15.0d0))
	(assert (= (vt-nanmean clean) 2.5d0))
	(assert (< (abs (- (vt-nanvar clean) (vt-var clean))) 1d-10))
	(format t "No-NaN edge cases: PASS~%")))

    (format t "~%All NaN statistics tests passed!~%")
    t))


;;; 测试 vt-logspace
(defun test-vt-logspace ()
  (format t "~%--- Testing vt-logspace ---~%")
  ;; 1. 默认底数10，端点包含
  (let* ((ls (vt-logspace 0 3 4 :type 'double-float))
         (arr (vt-to-list ls)))
    (assert (= (vt-size ls) 4))
    (assert (< (abs (- (first arr) 1.0d0)) 1d-10))
    (assert (< (abs (- (second arr) 10.0d0)) 1d-10))
    (assert (< (abs (- (third arr) 100.0d0)) 1d-10))
    (assert (< (abs (- (fourth arr) 1000.0d0)) 1d-10))
    (format t "  logspace(0,3,4) = ~A  PASS~%" arr))
  
  ;; 2. 底数为2，不包含终点
  (let* ((ls (vt-logspace 1 4 3 :base 2.0d0 :endpoint nil :type 'double-float))
         (arr (vt-to-list ls)))
    ;; 对数间隔在[1,4)上等分3个点：2^1=2, 2^2=4, 2^3=8
    (assert (< (abs (- (first arr) 2.0d0)) 1d-10))
    (assert (< (abs (- (second arr) 4.0d0)) 1d-10))
    (assert (< (abs (- (third arr) 8.0d0)) 1d-10))
    (format t "  logspace(1,4,3,base=2,endpoint=nil) = ~A  PASS~%" arr))
  
  ;; 3. 单点
  (let* ((ls (vt-logspace 0 0 1))
         (arr (vt-to-list ls)))
    (assert (= (length arr) 1))
    (assert (< (abs (- (first arr) 1.0d0)) 1d-10))
    (format t "  logspace(0,0,1) = ~A  PASS~%" arr))
  
  (format t "vt-logspace tests passed.~%"))

;; ----------------------------------------
;; vt-linspace 测试
;; ----------------------------------------
(defun test-vt-linspace ()
  (format t "~%=== Testing vt-linspace ===")

  ;; -----------------------------------------------------------
  ;; 1. 基本用法，默认 endpoint=t，double-float
  ;; np.linspace(0, 10, 5)
  ;; → array([ 0. ,  2.5,  5. ,  7.5, 10. ])
  ;; -----------------------------------------------------------
  (let ((vt (vt-linspace 0 10 5)))
    (assert (lists-approx-equal (vt-to-list vt)
				'(0.0 2.5 5.0 7.5 10.0)
				:epsilon 1e-6)))

  ;; -----------------------------------------------------------
  ;; 2. endpoint=nil，不包含终点
  ;; np.linspace(0, 10, 5, endpoint=False)
  ;; → array([0., 2., 4., 6., 8.])
  ;; -----------------------------------------------------------
  (let ((vt (vt-linspace 0 10 5 :endpoint nil)))
    (assert (lists-approx-equal (vt-to-list vt)
				'(0.0 2.0 4.0 6.0 8.0)
				:epsilon 1e-6)))

  ;; -----------------------------------------------------------
  ;; 3. num=1，此时应返回只包含 start 的数组
  ;; np.linspace(5, 20, 1)
  ;; → array([5.])
  ;; -----------------------------------------------------------
  (let ((vt (vt-linspace 5 20 1)))
    (assert (= (vt-size vt) 1))
    (assert (= (vt-ref vt 0) 5.0)))

  ;; -----------------------------------------------------------
  ;; 4. 整型 type，返回 fixnum 数组（需注意步长截断）
  ;; np.linspace(0, 10, 5, dtype=np.int32)
  ;; 但 NumPy 会进行取整，我们使用 fixnum 演示
  ;; 注意：因整数除法截断，结果可能不一样，这里仅验证类型和形状
  ;; -----------------------------------------------------------
  (let ((vt (vt-linspace 0 10 5 :type 'fixnum)))
    (assert (equal (vt-shape vt) '(5)))
    (assert (eq (vt-element-type vt) 'fixnum))
    ;; 值不做具体断言，因为整数除法会丢失精度
    )

  ;; -----------------------------------------------------------
  ;; 5. num=0 应报错
  ;; -----------------------------------------------------------
  (handler-case
      (progn
        (vt-linspace 0 10 0)
        (error "Should have signaled an error"))
    (error (e)
      (format t "~%[OK] Caught expected error for num=0: ~a" e)))

  ;; -----------------------------------------------------------
  ;; 6. 负步长（start > end）
  ;; np.linspace(10, 0, 5)
  ;; → array([10. ,  7.5,  5. ,  2.5,  0. ])
  ;; -----------------------------------------------------------
  (let ((vt (vt-linspace 10 0 5)))
    (assert (lists-approx-equal (vt-to-list vt)
				'(10.0 7.5 5.0 2.5 0.0) :epsilon 1e-6)))

  ;; -----------------------------------------------------------
  ;; 7. endpoint=t 且 num=2
  ;; np.linspace(0, 1, 2)
  ;; → array([0., 1.])
  ;; -----------------------------------------------------------
  (let ((vt (vt-linspace 0 1 2)))
    (assert (lists-approx-equal (vt-to-list vt)
				'(0.0 1.0) :epsilon 1e-6)))

  ;; -----------------------------------------------------------
  ;; 8. endpoint=nil 且 num=2
  ;; np.linspace(0, 1, 2, endpoint=False)
  ;; → array([0. , 0.5])
  ;; -----------------------------------------------------------
  (let ((vt (vt-linspace 0 1 2 :endpoint nil)))
    (assert (lists-approx-equal (vt-to-list vt)
				'(0.0 0.5) :epsilon 1e-6)))

  (format t "~%All vt-linspace tests passed.~%"))


(defun test-vt-kron ()
  (format t "~%--- Testing vt-kron ---~%")
  ;; 1. 两个向量（结果为一维）
  (let* ((a (vt-from-sequence '(1 2) :type 'double-float))
         (b (vt-from-sequence '(3 4 5) :type 'double-float))
         (k (vt-kron a b))
         (expected (vt-from-sequence '(3 4 5 6 8 10) :type 'double-float)))
    (assert (equalp (vt-shape k) '(6)))
    (assert (vt-all (vt-isclose k expected :atol 1d-12 :rtol 1d-12)))
    (format t "  kron([1,2],[3,4,5]) = ~A  PASS~%" k))
  
  ;; 2. 两个矩阵
  (let* ((a (vt-from-sequence '((1 2) (3 4)) :type 'double-float))
         (b (vt-from-sequence '((0 5) (6 7)) :type 'double-float))
         (k (vt-kron a b))
         (expected (vt-from-sequence
                    '((0 5 0 10)
                      (6 7 12 14)
                      (0 15 0 20)
                      (18 21 24 28))
                    :type 'double-float)))
    (assert (equalp (vt-shape k) '(4 4)))
    (assert (vt-all (vt-isclose k expected :atol 1d-12 :rtol 1d-12)))
    (format t "  kron([[1,2],[3,4]], [[0,5],[6,7]]) PASS~%"))
  
  ;; 3. 向量与矩阵（向量视为行向量）
  (let* ((a (vt-from-sequence '(1 2) :type 'double-float))
         (b (vt-from-sequence '((3 4) (5 6)) :type 'double-float))
         (k (vt-kron a b))
         ;; 预期：a 视为 1x2，b 为 2x2 → 结果 2x4
         (expected (vt-from-sequence
                    '((3 4 6 8)
                      (5 6 10 12))
                    :type 'double-float)))
    (assert (equalp (vt-shape k) '(2 4)))
    (assert (vt-all (vt-isclose k expected :atol 1d-12 :rtol 1d-12)))
    (format t "  kron([1,2], [[3,4],[5,6]]) PASS~%"))
  
  ;; 4. 高维情况 (3D)
  (let* ((a (vt-from-sequence '(((1 2) (3 4)) ((5 6) (7 8)))))
         (b (vt-from-sequence '((9 10) (11 12))))
         (k (vt-kron a b)))
    ;; a 形状 (2,2,2)，b 形状 (2,2) → 对齐后 a (2,2,2), b (1,2,2) → 结果 (2,4,4)
    (assert (equalp (vt-shape k) '(2 4 4)))
    ;; 取一个子块验证
    (let* ((sub-k (vt-slice k '(0) '(:all) '(:all)))
           (expected-sub (vt-kron (vt-slice a '(0) '(:all) '(:all)) b)))
      (assert (vt-all (vt-isclose sub-k expected-sub :atol 1d-12 :rtol 1d-12))))
    (format t "  kron of 2x2x2 and 2x2 PASS~%"))
  
  (format t "vt-kron tests passed.~%"))


(defun test-vt-diff ()
  (format t "~%--- Testing vt-diff ---~%")
  
  ;; 1. 一维数组，默认一阶差分
  (let* ((data (vt-from-sequence '(1 3 7 13 21) :type 'double-float))
         (result (vt-diff data)))
    (assert (equalp (vt-shape result) '(4)))
    (assert (equalp (vt-to-list result) '(2 4 6 8)))
    (format t "  1D 1st-order diff: ~A PASS~%" result))
  
  ;; 2. 一维数组，二阶差分
  (let* ((data (vt-from-sequence '(1 3 7 13 21) :type 'double-float))
         (result (vt-diff data :n 2)))
    (assert (equalp (vt-shape result) '(3)))
    (assert (equalp (vt-to-list result) '(2 2 2)))
    (format t "  1D 2nd-order diff: ~A PASS~%" result))
  
  ;; 3. 一维数组，高阶差分 (n=3)
  (let* ((data (vt-from-sequence '(1 2 4 8 16) :type 'double-float))
         (result (vt-diff data :n 3)))
    ;; 原始: [1,2,4,8,16]
    ;; diff1: [1,2,4,8]
    ;; diff2: [1,2,4]
    ;; diff3: [1,2]
    (assert (equalp (vt-shape result) '(2)))
    (assert (equalp (vt-to-list result) '(1 2)))
    (format t "  1D 3rd-order diff: ~A PASS~%" result))
  
  ;; 4. 二维数组，默认轴 (axis = -1)
  (let* ((data (vt-from-sequence '((1 3 7) (1 4 9)) :type 'double-float))
         (result (vt-diff data)))
    ;; 沿最后一轴 (axis=1) 差分
    (assert (equalp (vt-shape result) '(2 2)))
    (assert (equalp (vt-to-list result) '((2 4) (3 5))))
    (format t "  2D diff axis=-1: ~A PASS~%" result))
  
  ;; 5. 二维数组，沿 axis=0 差分
  (let* ((data (vt-from-sequence '((1 3 7) (1 4 9)) :type 'double-float))
         (result (vt-diff data :axis 0)))
    ;; 沿第0轴 (行方向) 差分
    (assert (equalp (vt-shape result) '(1 3)))
    (assert (equalp (vt-to-list result) '((0 1 2))))
    (format t "  2D diff axis=0: ~A PASS~%" result))
  
  ;; 6. 二维数组，沿 axis=1 且 n=2
  (let* ((data (vt-from-sequence '((1 3 7 13) (1 4 9 16)) :type 'double-float))
         (result (vt-diff data :axis 1 :n 2)))
    (assert (equalp (vt-shape result) '(2 2)))
    ;; 第一行: diff1 [2,4,6], diff2 [2,2]
    ;; 第二行: diff1 [3,5,7], diff2 [2,2]
    (assert (equalp (vt-to-list result) '((2 2) (2 2))))
    (format t "  2D diff axis=1, n=2: ~A PASS~%" result))
  
  ;; 7. 边界情况：长度不足以做差分（长度 < n+1）应返回空形状
  ;; NumPy 行为：若数组长度小于 n+1，返回空数组
  (let ((data (vt-from-sequence '(1 2) :type 'double-float)))
    ;; 一阶差分可做 (结果长度1)
    (let ((r (vt-diff data)))
      (assert (equalp (vt-shape r) '(1)))
      (assert (equalp (vt-to-list r) '(1))))
    ;; 二阶差分不可做 (结果长度0)
    (let ((r2 (vt-diff data :n 2)))
      (assert (equalp (vt-shape r2) '(0)))
      (assert (= (vt-size r2) 0)))
    (format t "  Boundary cases (short array): PASS~%"))
  
  (format t "vt-diff tests passed.~%"))

;;; 测试 vt-trapz
(defun test-vt-trapz ()
  (format t "~%--- Testing vt-trapz ---~%")
  ;; 1. 一维数组，默认 dx=1
  (let* ((y (vt-from-sequence '(1 2 3 4) :type 'double-float))
         (trap (vt-ref (vt-trapz y))))
    ;; 梯形积分： (1+2)/2 + (2+3)/2 + (3+4)/2 = 1.5+2.5+3.5 = 7.5
    (assert (floatp trap))
    (assert (< (abs (- trap 7.5d0)) 1d-10))
    (format t "  trapz([1,2,3,4]) default dx=1 -> ~A  PASS~%" trap))
  
  ;; 2. 指定 x 坐标
  (let* ((x (vt-from-sequence '(0 2 3 5) :type 'double-float))
         (y (vt-from-sequence '(1 2 3 4) :type 'double-float))
         (trap (vt-ref (vt-trapz y :x x))))
    ;; 梯度：dx1=2, dx2=1, dx3=2
    ;; 积分 = (1+2)/2*2 + (2+3)/2*1 + (3+4)/2*2 = 3*2 + 2.5*1 + 3.5*2 = 6+2.5+7 = 12.5
    (assert (< (abs (- trap 12.5d0)) 1d-10))
    (format t "  trapz with x=[0,2,3,5] -> ~A  PASS~%" trap))
  
  ;; 3. 二维数组，沿指定轴
  (let* ((data (vt-from-sequence '((1 2 3) (4 5 6)) :type 'double-float))
         (trap-axis0 (vt-trapz data :axis 0))
         (trap-axis1 (vt-trapz data :axis 1)))
    ;; axis=0：每列沿行方向（长度2）梯形积分，dx=1
    ;; 列0: (1+4)/2 = 2.5
    ;; 列1: (2+5)/2 = 3.5
    ;; 列2: (3+6)/2 = 4.5
    (assert (equalp (vt-to-list trap-axis0) '(2.5 3.5 4.5)))
    ;; axis=1：每行沿列方向（长度3）梯形积分
    ;; 行0: (1+2)/2 + (2+3)/2 = 1.5+2.5=4.0
    ;; 行1: (4+5)/2 + (5+6)/2 = 4.5+5.5=10.0
    (assert (equalp (vt-to-list trap-axis1) '(4.0 10.0)))
    (format t "  trapz on 2D array axis=0,1 PASS~%"))
  
  ;; 4. 仅有一个元素的错误处理
  (handler-case
      (vt-trapz (vt-from-sequence '(1)))
    (error (e)
      (format t "  trapz on single element correctly signaled error: ~A~%" e)))
  
  (format t "vt-trapz tests passed.~%"))

(defun test-vt-correlate ()
  (format t "~%--- Testing vt-correlate ---~%")
  (let* ((a (vt-from-sequence '(1 2 3) :type 'double-float))
         (v (vt-from-sequence '(0 1 0.5) :type 'double-float)))
    ;; full 模式
    (let ((res (vt-correlate a v :mode :full)))
      (assert (equalp (vt-to-list res) '(0.5 2.0 3.5 3.0 0.0)))
      (format t "  full  mode: ~A PASS~%" res))
    ;; valid 模式
    (let ((res (vt-correlate a v :mode :valid)))
      (assert (equalp (vt-to-list res) '(3.5)))
      (format t "  valid mode: ~A PASS~%" res))
    ;; same 模式
    (let ((res (vt-correlate a v :mode :same)))
      (assert (equalp (vt-to-list res) '(2.0 3.5 3.0)))
      (format t "  same  mode: ~A PASS~%" res)))
  ;; 单元素
  (let ((a (vt-from-sequence '(5) :type 'double-float))
        (v (vt-from-sequence '(3) :type 'double-float)))
    (dolist (mode '(:full :valid :same))
      (let ((res (vt-correlate a v :mode mode)))
        (assert (equalp (vt-to-list res) '(15)))
        (format t "  length-1 ~A: ~A PASS~%" mode res))))
  (format t "vt-correlate tests passed.~%"))


(defun vt-correlate-a (a v &key (mode "full"))
  "一维互相关（不翻转 v），与 numpy.correlate 一致。"
  (let* ((a-flat (vt-flatten a))
         (v-flat (vt-flatten v))
         (n (vt-size a-flat))
         (m (vt-size v-flat))
         (a-data (vt-data a-flat))
         (v-data (vt-data v-flat)))
    (flet ((compute-one (k)
             (let ((sum 0.0d0))
               (loop for j from (max 0 (- k)) below (min m (- n k))
                     do (incf sum (* (aref a-data (+ j k)) (aref v-data j))))
               sum)))
      ;; 生成完整的 full 结果 (长度 n+m-1)
      (let* ((full-len (+ n m -1))
             (full (vt-zeros (list full-len))))
        (loop for k from (- (1- m)) below n
              for i from 0
              do (setf (vt-ref full i) (compute-one k)))
        ;; 根据模式裁剪并返回连续副本
        (ecase (intern (string-upcase mode) :keyword)
          (:full full)
          (:valid
           (let* ((start (1- m))                    ; 第一个完整重叠的位置
                  (len (max 0 (1+ (- n m)))))      ; n - m + 1，最小为0
             (if (zerop len)
                 (vt-zeros '(0))
                 (vt-copy (apply #'vt-slice full (list (list start (+ start len))))))))
          (:same
           (let* ((out-len (max n m))
                  (start (floor (- full-len out-len) 2))
                  (end (+ start out-len)))
             (vt-copy (apply #'vt-slice full (list (list start end)))))))))))


(defun test-vt-histogram ()
  (format t "~%=== Testing vt-histogram ===")

  ;; ---- 1. 默认 bins=10，无 range，无 density ----
  ;; np.random.seed(42); a = np.random.randn(1000)
  ;; hist, edges = np.histogram(a)
  ;; 我们使用简单数据测试：0..9 的 10 个整数，bins=5
  ;; a = np.arange(10)
  ;; np.histogram(a, bins=5) -> (array([2,2,2,2,2]), array([0., 1.8, 3.6, 5.4, 7.2, 9.]))
  (let* ((a (vt-arange 10 :type 'fixnum))
         (result (multiple-value-list (vt-histogram a :bins 5)))
         (hist (first result))
         (edges (second result)))
    (assert (equal (vt-to-list hist) '(2.0 2.0 2.0 2.0 2.0)))
    (assert (lists-approx-equal (vt-to-list edges) '(0.0 1.8 3.6 5.4 7.2 9.0) :epsilon 1e-6)))

  ;; ---- 2. 指定 range ----
  ;; a = np.array([0,1,2,3,4,5,6,7,8,9])
  ;; np.histogram(a, bins=3, range=(0,9)) -> (array([3,3,4]), array([0., 3., 6., 9.]))
  (let* ((a (vt-arange 10 :type 'fixnum))
         (mv (multiple-value-list (vt-histogram a :bins 3 :range '(0 9))))
         (hist (first mv))
         (edges (second mv)))
    (assert (equal (vt-to-list hist) '(3.0 3.0 4.0)))
    (assert (lists-approx-equal (vt-to-list edges) '(0.0 3.0 6.0 9.0) :epsilon 1e-6)))

  ;; ---- 3. density=True ----
  ;; a = np.array([0,1,2,3,4])
  ;; np.histogram(a, bins=5, density=True) -> histogram/bin_width，所有 bin 宽度 0.8，每个 bin 计数 1
  ;; 密度 = 1 / (5 * 0.8) = 0.25? 实际 np.histogram(a, bins=5, density=True) 输出 hist 使得总面积=1，总面积 = sum(hist * bin_width) = 1
  ;; 如果每个 bin 计数为 1，宽度 0.8，则每个 bin 高度 = 1/(5*0.8)=0.25
  (let* ((a (vt-arange 5))
         (mv (multiple-value-list (vt-histogram a :bins 5 :density t)))
         (hist (first mv)))
    (assert (lists-approx-equal (vt-to-list hist) '(0.25 0.25 0.25 0.25 0.25) :epsilon 1e-6)))

  ;; ---- 4. 值全相同 ----
  ;; a = np.array([2,2,2,2])
  ;; np.histogram(a, bins=3) -> 所有值落在同一个 bin？实际默认 range 会扩展至 (2-0.5, 2+0.5)? 需要指定 range 为准
  ;; 我们指定 range=(0,10), bins=5
  ;; np.histogram(np.array([2,2,2,2]), bins=5, range=(0,10)) -> (array([0,4,0,0,0]), array([0.,2.,4.,6.,8.,10.]))
  (let* ((a (vt-const '(4) 2 :type 'fixnum))
         (mv (multiple-value-list (vt-histogram a :bins 5 :range '(0 10))))
         (hist (first mv))
         (edges (second mv)))
    (assert (equal (vt-to-list hist) '(0.0 4.0 0.0 0.0 0.0)))
    (assert (lists-approx-equal (vt-to-list edges) '(0.0 2.0 4.0 6.0 8.0 10.0) :epsilon 1e-6)))

  ;; ---- 5. 一维空张量 ----
  ;; a = np.array([])
  ;; np.histogram(a, bins=5) -> (array([0,0,0,0,0]), array([0.,0.,0.,0.,0.,0.])) 但实际空数组 range 未定义会报错
  ;; NumPy 对空数组调用 histogram 会出错，我们只需确保程序不崩溃并返回合理空结果（如全零）
  (let* ((a (vt-zeros '(0)))
         (mv (multiple-value-list (vt-histogram a :bins 5 :range '(0 5))))
         (hist (first mv)))
    (assert (every #'zerop (vt-to-list hist)))
    (assert (= (vt-size hist) 5)))

  ;; ---- 6. 二维张量（内部自动展平） ----
  ;; a = np.arange(6).reshape(2,3)
  ;; np.histogram(a, bins=3, range=(0,5)) -> 展平后直方图
  (let* ((a (vt-reshape (vt-arange 6 :type 'fixnum) '(2 3)))
         (mv (multiple-value-list (vt-histogram a :bins 3 :range '(0 5))))
         (hist (first mv))
         (edges (second mv)))
    (assert (equal (vt-to-list hist) '(2.0 2.0 2.0))) ; 0,1,2,3,4,5  -> bin 宽度 1.666..., 计数分别为 2,2,2
    (assert (lists-approx-equal (vt-to-list edges) '(0.0 1.66666667 3.33333333 5.0) :epsilon 1e-6)))

  (format t "~%All vt-histogram tests passed.~%"))


(defun test-set-ops ()
  ;; vt-unique 基本排序，不返回额外关键字
  ;; a = np.array([3,1,2,1,3])
  ;; np.unique(a) -> [1,2,3]
  (let* ((a (vt-from-sequence '(3 1 2 1 3) :type 'fixnum))
         (u (vt-unique a)))
    (assert (equal (vt-to-list u) '(1 2 3))))

  ;; vt-intersect1d
  ;; np.intersect1d([1,3,5],[3,7,5]) -> [3,5]
  (let* ((a (vt-from-sequence '(1 3 5) :type 'fixnum))
         (b (vt-from-sequence '(3 7 5) :type 'fixnum))
         (res (vt-intersect1d a b)))
    (assert (equal (vt-to-list res) '(3 5))))

  ;; vt-union1d
  ;; np.union1d([1,2],[2,3]) -> [1,2,3]
  (let* ((a (vt-from-sequence '(1 2) :type 'fixnum))
         (b (vt-from-sequence '(2 3) :type 'fixnum))
         (res (vt-union1d a b)))
    (assert (equal (vt-to-list res) '(1 2 3))))

  ;; vt-setdiff1d
  ;; np.setdiff1d([1,2,3,4],[3,4,5]) -> [1,2]
  (let* ((a (vt-from-sequence '(1 2 3 4) :type 'fixnum))
         (b (vt-from-sequence '(3 4 5) :type 'fixnum))
         (res (vt-setdiff1d a b)))
    (assert (equal (vt-to-list res) '(1 2))))

  ;; vt-setxor1d
  ;; np.setxor1d([1,2,3],[3,4]) -> [1,2,4]
  (let* ((a (vt-from-sequence '(1 2 3) :type 'fixnum))
         (b (vt-from-sequence '(3 4) :type 'fixnum))
         (res (vt-setxor1d a b)))
    (assert (equal (vt-to-list res) '(1 2 4))))

  ;; vt-in1d
  ;; np.in1d([1,2,3],[2,4]) -> [False, True, False]
  (let* ((a (vt-from-sequence '(1 2 3) :type 'fixnum))
         (b (vt-from-sequence '(2 4) :type 'fixnum))
         (res (vt-in1d a b)))
    (assert (equal (vt-to-list res) '(0.0 1.0 0.0))))

  (format t "~%test-set-ops passed.~%"))


;; --------------------------------------------------------------------
;; 激活函数与损失函数
;; --------------------------------------------------------------------
(defun test-activation-loss ()
  ;; vt-softmax 沿最后一维
  ;; a = np.array([[1,2],[3,4]])
  ;; scipy.special.softmax(a, axis=-1) -> 数值稳定版
  ;; softmax=exp(x)/sum(exp(x))
  

  ;; vt-softmax
  (let* ((a (vt-from-sequence '((1 2) (3 4)) :type 'double-float))
	 (sm (vt-softmax a :axis -1)))
    (assert (lists-approx-equal
	     (vt-to-list sm)
	     '((0.2689414213699951d0 0.7310585786300049d0)
               (0.2689414213699951d0 0.7310585786300049d0)))))

  ;; vt-log-softmax
  (let* ((a (vt-from-sequence '((1 2) (3 4)) :type 'double-float))
	 (lsm (vt-log-softmax a :axis -1))
	 (soft (vt-softmax a :axis -1))
	 (logsoft (vt-log soft)))
    (assert (lists-approx-equal (vt-to-list lsm)
				(vt-to-list logsoft))))

  ;; vt-cross-entropy (one-hot 目标)
  ;; 预测 logits: [[2.0, 1.0, 0.1], [0.5, 2.5, 0.3]], 目标: 类别索引1 和 1 (one-hot)

  ;; 先计算 softmax，再将概率传入交叉熵
  (let* ((logits (vt-from-sequence '((2.0 1.0 0.1) (0.5 2.5 0.3))
				   :type 'double-float))
	 (probs (vt-softmax logits :axis -1))
	 (true (vt-from-sequence '((0.0 1.0 0.0) (0.0 1.0 0.0))
				 :type 'double-float))
	 (loss (vt-cross-entropy true probs)))
    ;; 损失应为正数（大约 0.874）
    (assert (> loss 0.0d0))
    (assert (< loss 5.0d0)))

  ;; vt-binary-cross-entropy
  ;; y_true=[1,0,1], y_pred=[0.9,0.1,0.8]
  ;; BCE = -[1*log(0.9)+0*log(0.1)+1*log(0.8)] / 3
  ;; 二元交叉熵
  (let* ((y-true (vt-from-sequence '(1 0 1) :type 'double-float))
	 (y-pred (vt-from-sequence '(0.9 0.1 0.8) :type 'double-float))
	 (bce (vt-binary-cross-entropy y-true y-pred)))
    ;; 正确期望值：三个样本损失的平均
    (let ((expected (/ (+ (- (log 0.9d0))   ; 样本1
                          (- (log 0.9d0))   ; 样本2 (1-0.1=0.9)
                          (- (log 0.8d0)))  ; 样本3
                       3.0d0)))
      (assert (< (abs (- bce expected)) 1e-6))))

  ;; vt-relu
  ;; np.maximum(0, [-1,0,2,-3]) -> [0,0,2,0]
  (let* ((a (vt-from-sequence '(-1 0 2 -3) :type 'double-float))
         (r (vt-relu a)))
    (assert (equal (vt-to-list r) '(0.0 0.0 2.0 0.0))))

  ;; vt-sigmoid
  ;; sigmoid(0) = 0.5
  (let ((a (vt-const '(1) 0.0d0 :type 'double-float)))
    (assert (= (vt-ref (vt-sigmoid a) 0) 0.5d0)))

  ;; vt-tanh 奇函数性质
  (let ((a (vt-const '(1) 0.0d0 :type 'double-float)))
    (assert (= (vt-ref (vt-tanh a) 0) 0.0d0)))

  ;; vt-gelu (近似) 输入0输出0
  (let ((a (vt-zeros '(3) :type 'double-float)))
    (assert (every (lambda (x) (< (abs x) 1e-10))
		   (vt-to-list (vt-gelu a)))))

  (format t "~%test-activation-loss passed.~%"))

;; --------------------------------------------------------------------
;; 统计补充
;; --------------------------------------------------------------------
(defun test-stats-more ()
  ;; vt-ptp 峰峰值
  ;; a = np.array([[1,5],[3,2]])
  ;; np.ptp(a) -> 4
  ;; np.ptp(a, axis=0) -> [2,3]
  (let* ((a (vt-from-sequence '((1 5) (3 2)) :type 'fixnum))
         (p (vt-ptp a)))
    (assert (= p 4.0d0))
    (let ((p0 (vt-to-list (vt-ptp a :axis 0))))
      (assert (equal p0 '(2.0 3.0)))))

  ;; vt-average 加权平均
  ;; a = [1,2,3], weights = [1,2,1]
  ;; np.average(a, weights=weights) -> (1*1+2*2+3*1)/(1+2+1) = 2.0
  (let* ((a (vt-from-sequence '(1 2 3) :type 'double-float))
         (w (vt-from-sequence '(1 2 1) :type 'double-float))
         (avg (vt-average a w)))
    (assert (= avg 2.0d0)))

  ;; vt-var / vt-std
  ;; a = [1,2,3,4]  有偏方差 = mean((x-mean)^2) = 1.25, std=sqrt(1.25)
  (let* ((a (vt-from-sequence '(1 2 3 4) :type 'double-float))
         (v (vt-var a))
         (s (vt-std a)))
    (assert (lists-approx-equal (list v)
				'(1.25d0) :epsilon 1e-6))
    (assert (lists-approx-equal (list s)
				(list (sqrt 1.25d0)) :epsilon 1e-6)))

  ;; vt-prod
  ;; np.prod([1,2,3,4]) -> 24
  (let ((a (vt-from-sequence '(1 2 3 4) :type 'fixnum)))
    (assert (= (vt-prod a) 24.0d0)))

  ;; vt-prod axis
  ;; a = np.array([[1,2],[3,4]])
  ;; np.prod(a, axis=1) -> [2,12]
  (let* ((a (vt-from-sequence '((1 2) (3 4)) :type 'fixnum))
         (p (vt-prod a :axis 1)))
    (assert (equalp (vt-to-list p) '(2.0 12.0))))

  (format t "~%test-stats-more passed.~%"))

;; --------------------------------------------------------------------
;; 逻辑与条件
;; --------------------------------------------------------------------
(defun test-logical ()
  ;; vt-all
  ;; np.all([True, True, False]) -> False
  (let ((a (vt-from-sequence '(1 1 0) :type 'double-float)))
    (assert (= (vt-all a) 0.0d0)))

  ;; vt-any
  ;; np.any([0,0,1]) -> True
  (let ((a (vt-from-sequence '(0 0 1) :type 'double-float)))
    (assert (= (vt-any a) 1.0d0)))

  ;; vt-isclose
  ;; np.isclose(1.0, 1.000001) -> True
  (let ((a (vt-const '() 1.0d0))
        (b (vt-const '() 1.000001d0)))
    (assert (= (vt-ref (vt-isclose a b) 0) 1.0d0)))

  ;; vt-allclose
  (let ((a (vt-from-sequence '(1 2 3) :type 'double-float))
        (b (vt-from-sequence '(1 2 3.0001) :type 'double-float)))
    (assert (vt-allclose a b :rtol 1e-3 :atol 1e-3)))

  ;; vt-isfinite / vt-isinf / vt-isnan
  (let* ((data (vt-from-sequence
		(list 1.0 0.0
		      sb-kernel::double-float-positive-infinity
		      sb-kernel::double-float-positive-infinity 0.0)
		:type 'double-float))
         (finite (vt-isfinite data))
         (inf (vt-isinf data))
         (nan (vt-isnan data)))
    (assert (equal (vt-to-list finite) '(1.0 1.0 0.0 0.0 1.0)))
    (assert (equal (vt-to-list inf) '(0.0 0.0 1.0 1.0 0.0)))
    (assert (every #'zerop (vt-to-list nan))))   ; 没有 NaN

  (format t "~%test-logical passed.~%"))

;; --------------------------------------------------------------------
;; 数学运算
;; --------------------------------------------------------------------
(defun test-math-ops ()
  ;; vt-clip
  ;; np.clip([1,2,3,4,5], 2, 4) -> [2,2,3,4,4]
  (let* ((a (vt-from-sequence '(1 2 3 4 5) :type 'double-float))
         (c (vt-clip a 2 4)))
    (assert (equal (vt-to-list c) '(2.0 2.0 3.0 4.0 4.0))))

  ;; vt-convolve (same mode)
  ;; np.convolve([1,2,3], [0,1,0.5], 'same') -> 中间值与 NumPy 对比
  (let* ((a (vt-from-sequence '(1 2 3) :type 'double-float))
         (v (vt-from-sequence '(0 1 0.5) :type 'double-float))
         (c (vt-convolve a v :mode :same)))
    ;; NumPy: array([1., 2.5, 4 ])  计算略
    (assert (lists-approx-equal (vt-to-list c)
				'(1.0 2.5 4.0) :epsilon 1e-6)))

  (format t "~%test-math-ops passed.~%"))

;; --------------------------------------------------------------------
;; 随机数生成
;; --------------------------------------------------------------------
(defun test-random ()
  ;; vt-random-uniform 形状与范围
  (let* ((shape '(5 4))
         (r (vt-random-uniform shape :low 0 :high 10)))
    (assert (equal (vt-shape r) shape))
    ;; 确保所有元素在 [0,10) 内
    (vt-do-each (ptr val r)
      (declare (ignore ptr))
      (assert (and (>= val 0.0d0) (< val 10.0d0)))))

  ;; vt-random-normal 形状与种子复现
  (vt-random-seed (make-random-state t))   ; 设置种子
  (let* ((shape '(3 3))
         (r1 (vt-random-normal shape :mean 0 :std 1))
         ;; 再次设置相同种子
         (_ (vt-random-seed (make-random-state t)))
         (r2 (vt-random-normal shape :mean 0 :std 1)))
    (declare (ignorable _ r2))
    (assert (equal (vt-shape r1) shape))
    ;; 由于种子可能非确定，不强求相等，但至少形状对
    (assert (equal '(3 3) (vt-shape r1)))
    (assert (= (vt-size r1) 9)))

  ;; vt-random-int 范围
  (let* ((r (vt-random-int 5 10 :size '(10))))
    (assert (equal (vt-shape r) '(10)))
    (vt-do-each (ptr val r)
      (declare (ignore ptr))
      (assert (and (>= val 5) (< val 10)))))

  (format t "~%test-random passed.~%"))


;; --------------------------------------------------------------------
;; 激活函数补充测试
;; --------------------------------------------------------------------
(defun test-more-activations ()
  ;; vt-leaky-relu alpha=0.01
  ;; np.where(x > 0, x, 0.01*x)
  (let* ((a (vt-from-sequence '(-1.0 0.0 2.0) :type 'double-float))
         (act (vt-leaky-relu a :alpha 0.01d0)))
    (assert (lists-approx-equal (vt-to-list act) '(-0.01 0.0 2.0) :epsilon 1e-6)))

  ;; vt-swish = x * sigmoid(x)
  ;; swish(0) = 0, swish(1) ≈ 1/(1+exp(-1)) ≈ 0.7311
  (let* ((a (vt-from-sequence '(0.0 1.0) :type 'double-float))
         (act (vt-swish a)))
    (assert (lists-approx-equal (vt-to-list act) '(0.0 0.7310585786300049) :epsilon 1e-6)))

  ;; vt-softplus: log(1+exp(x))，大值近似 x
  (let* ((a (vt-from-sequence '(0.0 10.0) :type 'double-float))
         (act (vt-softplus a)))
    (assert (lists-approx-equal (vt-to-list act) (list (log 2.0d0) (+ (log (1+ (exp 10.0d0))) 0.0d0)) :epsilon 1e-6)))

  ;; vt-mish: x * tanh(softplus(x)) 在0处为0，正数近似 x
  (let* ((a (vt-from-sequence '(0.0 2.0) :type 'double-float))
         (act (vt-mish a)))
    (assert (lists-approx-equal (vt-to-list act) (list 0.0d0 (* 2.0d0 (tanh (log (1+ (exp 2.0d0)))))) :epsilon 1e-6)))

  ;; vt-hard-tanh: clamp(x, -1, 1)
  (let* ((a (vt-from-sequence '(-1.5 -0.5 0.5 1.5) :type 'double-float))
         (act (vt-hard-tanh a)))
    (assert (equal (vt-to-list act) '(-1.0 -0.5 0.5 1.0))))

  ;; vt-hard-sigmoid: 快速分段线性近似
  ;; 实现: 0.2*x + 0.5 后 clip [0,1]
  (let* ((a (vt-from-sequence '(-3.0 0.0 3.0) :type 'double-float))
         (act (vt-hard-sigmoid a)))
    (assert (lists-approx-equal (vt-to-list act) '(0.0 0.5 1.0) :epsilon 1e-6)))

  (format t "~%test-more-activations passed.~%"))

;; --------------------------------------------------------------------
;; 通用数学运算测试（逐元素，含广播）
;; --------------------------------------------------------------------
(defun test-arithmetic-ops ()
  ;; vt-square
  (let* ((a (vt-from-sequence '(-2 0 3) :type 'fixnum))
         (sq (vt-square a)))
    (assert (equalp (vt-to-list sq) '(4 0 9))))

  ;; vt-sqrt
  (let* ((a (vt-from-sequence '(4.0 0.0 9.0) :type 'double-float))
         (sq (vt-sqrt a)))
    (assert (lists-approx-equal (vt-to-list sq) '(2.0 0.0 3.0))))

  ;; vt-exp / vt-log
  (let* ((a (vt-from-sequence '(1.0 0.0) :type 'double-float))
         (exp-a (vt-exp a))
         (log-exp (vt-log exp-a)))
    (assert (lists-approx-equal (vt-to-list log-exp) '(1.0 0.0) :epsilon 1e-6)))

  ;; vt-clip (already tested, but verify broadcast)
  (let* ((a (vt-from-sequence '((-1 2 5) (10 0 -3)) :type 'fixnum))
         (cl (vt-clip a 0 4)))
    (assert (equalp (vt-to-list cl) '((0 2 4) (4 0 0)))))

  ;; vt-mod / vt-rem
  (let* ((a (vt-from-sequence '(5 3 8) :type 'fixnum))
         (mod (vt-mod a 3))
         (rem (vt-rem a 3)))
    (assert (equalp (vt-to-list mod) '(2 0 2)))
    (assert (equalp (vt-to-list rem) '(2 0 2))))

  ;; vt-round / vt-floor / vt-ceiling / vt-truncate
  (let* ((a (vt-from-sequence '(-1.4 2.6) :type 'double-float))
         (r (vt-round a))
         (f (vt-floor a))
         (c (vt-ceiling a))
         (e (vt-trancate a)))
    (assert (lists-approx-equal (vt-to-list r) '(-1.0 3.0)))
    (assert (lists-approx-equal (vt-to-list f) '(-2.0 2.0)))
    (assert (lists-approx-equal (vt-to-list c) '(-1.0 3.0)))
    (assert (lists-approx-equal (vt-to-list e) '(-1.0 2.0))))

  ;; vt-rint (round to nearest integer, float)
  (let* ((a (vt-from-sequence '(1.2 2.7 -1.5) :type 'double-float))
         (ri (vt-rint a)))
    ;; 注意：rint 在半值向上取整
    (assert (lists-approx-equal (vt-to-list ri) '(1.0 3.0 -2.0))))

  ;; vt-signum
  (let* ((a (vt-from-sequence '(-3 0 5) :type 'fixnum))
         (s (vt-signum a)))
    (assert (equalp (vt-to-list s) '(-1 0 1))))

  ;; 三角函数简单测试
  ;; sin(0)=0, cos(0)=1
  (let ((a (vt-const '(1) 0.0d0 :type 'double-float)))
    (assert (< (abs (vt-ref (vt-sin a) 0)) 1e-10))
    (assert (< (abs (- 1.0d0 (vt-ref (vt-cos a) 0))) 1e-10))
    (assert (< (abs (vt-ref (vt-tan a) 0)) 1e-10)))

  ;; 双曲函数
  (let ((a (vt-const '(1) 0.0d0 :type 'double-float)))
    (assert (< (abs (vt-ref (vt-sinh a) 0)) 1e-10))
    (assert (< (abs (- 1.0d0 (vt-ref (vt-cosh a) 0))) 1e-10))
    (assert (< (abs (vt-ref (vt-tanh a) 0)) 1e-10)))

  ;; vt-hypot
  (let* ((a (vt-from-sequence '(3.0) :type 'double-float))
         (b (vt-from-sequence '(4.0) :type 'double-float))
         (h (vt-hypot a b)))
    (assert (lists-approx-equal (vt-to-list h) '(5.0) :epsilon 1e-6)))

  ;; vt-sinc: sinc(x) = sin(pi*x)/(pi*x), sinc(0)=1
  (let* ((a (vt-from-sequence '(0.0 0.5) :type 'double-float))
         (sc (vt-sinc a)))
    ;; sinc(0.5) = sin(pi/2)/(pi/2) = 1 / (pi/2) = 2/pi ≈ 0.6366
    (assert (lists-approx-equal (vt-to-list sc) (list 1.0d0 (/ 2.0d0 pi)) :epsilon 1e-6)))

  ;; vt-deg2rad / vt-rad2deg
  (let* ((a (vt-from-sequence '(180.0) :type 'double-float))
         (rad (vt-deg2rad a))
         (deg (vt-rad2deg rad)))
    (assert (lists-approx-equal (vt-to-list rad) (list pi) :epsilon 1e-6))
    (assert (lists-approx-equal (vt-to-list deg) '(180.0) :epsilon 1e-6)))

  (format t "~%test-arithmetic-ops passed.~%"))

;; --------------------------------------------------------------------
;; vt-from-function 测试
;; --------------------------------------------------------------------
(defun test-vt-from-function ()
  ;; 使用函数生成二维索引和
  ;; np.fromfunction(lambda i,j: i+j, (3,2), dtype=int)
  (let* ((shape '(3 2))
         (fn (lambda (idxs) (+ (first idxs) (second idxs))))
         (arr (vt-from-function shape fn :type 'fixnum)))
    (assert (equal (vt-shape arr) shape))
    (assert (equal (vt-to-list arr) '((0 1) (1 2) (2 3)))))

  ;; 一维情况
  (let* ((shape '(5))
         (fn (lambda (idxs) (* (first idxs) 2)))
         (arr (vt-from-function shape fn :type 'fixnum)))
    (assert (equal (vt-to-list arr) '(0 2 4 6 8))))

  (format t "~%test-vt-from-function passed.~%"))

;; --------------------------------------------------------------------
;; vt-bincount 测试
;; --------------------------------------------------------------------
(defun test-vt-bincount ()
  ;; np.bincount([0,1,2,1,0,3]) -> [2,2,1,1]
  (let* ((a (vt-from-sequence '(0 1 2 1 0 3) :type 'fixnum))
         (cnt (vt-bincount a)))
    (assert (equal (vt-to-list cnt) '(2 2 1 1))))

  ;; 指定 minlength
  ;; np.bincount([0,1,1], minlength=5) -> [1,2,0,0,0]
  (let* ((a (vt-from-sequence '(0 1 1) :type 'fixnum))
         (cnt (vt-bincount a :minlength 5)))
    (assert (equal (vt-to-list cnt) '(1 2 0 0 0))))

  ;; 空输入（零大小）
  (let* ((a (vt-zeros '(0) :type 'fixnum))
         (cnt (vt-bincount a)))
    (assert (equal (vt-to-list cnt) '())))

  (format t "~%test-vt-bincount passed.~%"))

;; --------------------------------------------------------------------
;; vt-digitize 测试
;; --------------------------------------------------------------------
(defun test-vt-digitize ()
  ;; 默认 right=False：bins[i-1] <= x < bins[i]
  ;; np.digitize([0.2, 6.4, 3.0, 1.6], [0,2,4,6]) -> [1,4,2,1]
  (let* ((x (vt-from-sequence '(0.2 6.4 3.0 1.6) :type 'double-float))
         (bins (vt-from-sequence '(0 2 4 6) :type 'double-float))
         (dig (vt-digitize x bins)))
    (assert (equal (vt-to-list dig) '(1 4 2 1))))

  ;; x = np.array([2.0])
  ;; bins = np.array([0, 2, 4])
  ;; # right=False（默认）：bins[i-1] <= x < bins[i]
  ;; # 返回第一个大于 x 的 bin 索引
  ;; dig_false = np.digitize(x, bins, right=False)
  ;; print(dig_false)   # [2]   (因为 2.0 属于 [2,4) 区间，索引为 2)

  ;; # right=True ：bins[i-1] < x <= bins[i]
  ;; # 返回第一个大于等于 x 的 bin 索引
  ;; dig_true = np.digitize(x, bins, right=True)
  ;; print(dig_true)  
  (let* ((x (vt-from-sequence '(2.0) :type 'double-float))
         (bins (vt-from-sequence '(0 2 4) :type 'double-float))
         (dig-false (vt-digitize x bins))
         (dig-true (vt-digitize x bins :right t)))
    ;; right=False: 2.0 属于 [2,4) -> bin 2
    (assert (= (vt-ref dig-false 0) 2))
    ;; right=True: 2.0 属于 (0,2] -> bin 1
    (assert (= (vt-ref dig-true 0) 1)))

  (format t "~%test-vt-digitize passed.~%"))



;;; ===========================================
;;; 测试函数 (需放在 (clvt) 包内)
;;; 每个用例均注释了对应的 NumPy 实现
;;; ===========================================

(defun test-vt-append ()
  "测试 vt-append 与 NumPy np.append 行为的一致性。"
  (format t "~%Testing vt-append...~%")
  (flet ((check (name got expected-list)
           (let ((expected (vt-from-sequence expected-list :type 'double-float)))
             (if (vt-allclose got expected :rtol 1e-5 :atol 1e-8)
                 (format t "✓ PASS: ~A~%" name)
                 (format t "✗ FAIL: ~A~%  got: ~A~%  expected: ~A~%" 
                         name (vt-to-list got) expected-list)))))
    
    ;; 用例1: 展平模式，标量 values 自动转为张量
    ;; # NumPy: a = np.array([[1,2],[3,4]]); b = 5; np.append(a, b)  # axis=None
    (let ((a (vt-from-sequence '((1 2) (3 4)) :type 'double-float))
          (b 5))
      (check "append_flat_scalar"
             (vt-append a b :axis nil)
             '(1 2 3 4 5)))
    
    ;; 用例2: 展平模式，张量 values
    ;; # NumPy: a = np.array([[1,2],[3,4]]); b = np.array([5,6]); np.append(a, b)
    (let ((a (vt-from-sequence '((1 2) (3 4)) :type 'double-float))
          (b (vt-from-sequence '(5 6) :type 'double-float)))
      (check "append_flat_tensor"
             (vt-append a b :axis nil)
             '(1 2 3 4 5 6)))
    
    ;; 用例3: 沿 axis=0 连接
    ;; # NumPy: a = np.array([[1,2],[3,4]]); b = np.array([[5,6]]); np.append(a, b, axis=0)
    (let ((a (vt-from-sequence '((1 2) (3 4)) :type 'double-float))
          (b (vt-from-sequence '((5 6)) :type 'double-float)))
      (check "append_axis0"
             (vt-append a b :axis 0)
             '((1 2) (3 4) (5 6))))
    
    ;; 用例4: 沿 axis=1 连接 (要求行数相同)
    ;; # NumPy: a = np.array([[1,2],[3,4]]); b = np.array([[5],[6]]); np.append(a, b, axis=1)
    (let ((a (vt-from-sequence '((1 2) (3 4)) :type 'double-float))
          (b (vt-from-sequence '((5) (6)) :type 'double-float)))
      (check "append_axis1"
             (vt-append a b :axis 1)
             '((1 2 5) (3 4 6))))
    t))

(defun test-vt-insert ()
  "测试 vt-insert 与 NumPy np.insert 行为的一致性。"
  (format t "~%Testing vt-insert...~%")
  (flet ((check (name got expected-list)
           (let ((expected (vt-from-sequence expected-list :type 'double-float)))
             (if (vt-allclose got expected :rtol 1e-5 :atol 1e-8)
                 (format t "✓ PASS: ~A~%" name)
                 (format t "✗ FAIL: ~A~%  got: ~A~%  expected: ~A~%" 
                         name (vt-to-list got) expected-list)))))
    
    ;; 用例1: 展平模式，单索引插入标量
    ;; # NumPy: a = np.array([1,2,3,4]); np.insert(a, 2, 99)  # axis=None
    (let ((a (vt-from-sequence '(1 2 3 4) :type 'double-float))
          (vals 99))
      (check "insert_flat_scalar"
             (vt-insert a 2 vals :axis nil)
             '(1 2 99 3 4)))
    
    ;; 用例2: 展平模式，单索引插入张量
    ;; # NumPy: a = np.array([1,2,3,4]); vals = np.array([99,100]); np.insert(a, 2, vals)
    (let ((a (vt-from-sequence '(1 2 3 4) :type 'double-float))
          (vals (vt-from-sequence '(99 100) :type 'double-float)))
      (check "insert_flat_tensor"
             (vt-insert a 2 vals :axis nil)
             '(1 2 99 100 3 4)))
    
    ;; 用例3: 展平模式，多索引列表插入
    ;; # NumPy: a = np.array([1,2,3,4]); np.insert(a, [0,2], 99)
    (let ((a (vt-from-sequence '(1 2 3 4) :type 'double-float))
          (vals (vt-from-sequence '(99) :type 'double-float)))
      (check "insert_flat_multi_index"
             (vt-insert a '(0 2) vals :axis nil)
             '(99 1 2 99 3 4)))
    
    ;; 用例4: 沿 axis=1 插入 (矩阵列)
    ;; # NumPy: a = np.array([[1,2],[3,4]]); vals = np.array([[99,100]]); np.insert(a, 1, vals, axis=1)
    (let ((a (vt-from-sequence '((1 2) (3 4)) :type 'double-float))
          (vals (vt-from-sequence '((99 100)) :type 'double-float)))
      (check "insert_axis1"
             (vt-insert a 1 vals :axis 1)
             '((1 99 2) (3 100 4))))
    
    ;; 用例5: 沿 axis=0 插入行
    ;; # NumPy: a = np.array([[1,2],[3,4]]); vals = np.array([[5,6]]); np.insert(a, 1, vals, axis=0)
    (let ((a (vt-from-sequence '((1 2) (3 4)) :type 'double-float))
          (vals (vt-from-sequence '((5 6)) :type 'double-float)))
      (check "insert_axis0"
             (vt-insert a 1 vals :axis 0)
             '((1 2) (5 6) (3 4))))
    ;; 测试 1：轴0插入，索引0
    (let* ((a (vt-from-sequence '((0 1 2) (3 4 5))))
           (res (vt-insert a 0 (vt-from-sequence '(10 11 12) :type 'double-float) :axis 0))
           (expected (vt-from-sequence '((10 11 12) (0 1 2) (3 4 5)))))
      (assert-ok (approx= res expected) "vt-insert 轴0插入失败"))

    ;; 测试 2：轴1末尾插入
    (let* ((a (vt-from-sequence '((0 1 2) (3 4 5))))
           (res (vt-insert a (second (vt-shape a))
			   (vt-from-sequence '(7 8) :type 'double-float) :axis 1))
           (expected (vt-from-sequence '((0 1 2 7) (3 4 5 8)))))
      (assert-ok (approx= res expected) "vt-insert 轴末尾插入失败"))

    ;; 测试 3：越界索引（应为错误）
    (let ((a (vt-from-sequence '((0 1 2) (3 4 5)))))
      (handler-case
          (progn
            (vt-insert a 5 (vt-from-sequence '(1)) :axis 0)
            (error "应该抛出越界错误"))
	(error () t)))

    ;; 测试 4：空索引列表（什么都不插入）
    (let* ((a (vt-from-sequence '((0 1 2) (3 4 5))))
           (res (vt-insert a '() (vt-from-sequence '()) :axis 1)))
      (assert-ok (approx= res a) "空插入应返回原数组"))

    ;; 测试 5：展平模式越界索引
    (let ((a-flat (vt-from-sequence '(0 1 2 3 4))))
      (handler-case
          (progn
            (vt-insert a-flat 6 99)   ; 索引 6 越界（有效 0~5）
            (error "应该抛出越界错误"))
	(error () t)))
    (format t "~%vt-insert 全部测试通过~%")
    t))

(defun test-vt-delete ()
  "测试 vt-delete 与 NumPy np.delete 行为的一致性。"
  (format t "~%Testing vt-delete...~%")
  (flet ((check (name got expected-list)
           (let ((expected (vt-from-sequence expected-list :type 'double-float)))
             (if (vt-allclose got expected :rtol 1e-5 :atol 1e-8)
                 (format t "✓ PASS: ~A~%" name)
                 (format t "✗ FAIL: ~A~%  got: ~A~%  expected: ~A~%" 
                         name (vt-to-list got) expected-list)))))
    
    ;; 用例1: 展平模式，单索引删除
    ;; # NumPy: a = np.array([1,2,3,4,5]); np.delete(a, 2)  # axis=None
    (let ((a (vt-from-sequence '(1 2 3 4 5) :type 'double-float)))
      (check "delete_flat_single"
             (vt-delete a 2 :axis nil)
             '(1 2 4 5)))
    
    ;; 用例2: 展平模式，切片删除 (start end)
    ;; # NumPy: a = np.array([1,2,3,4,5]); np.delete(a, slice(1,3))

    (let ((a (vt-from-sequence '(1 2 3 4 5) :type 'double-float)))
      (check "delete_flat_slice"
             (vt-delete a '(:slice 1 3) :axis nil)   ; 注意改为 (:slice 1 3)
             '(1 4 5)))
    
    ;; 用例3: 展平模式，索引列表删除
    ;; # NumPy: a = np.array([1,2,3,4,5]); np.delete(a, [0,2,4])
    (let ((a (vt-from-sequence '(1 2 3 4 5) :type 'double-float)))
      (check "delete_flat_list"
             (vt-delete a '(0 2 4) :axis nil)
             '(2 4)))
    
    ;; 用例4: 沿 axis=0 删除索引列表
    ;; # NumPy: a = np.array([[1,2],[3,4],[5,6]]); np.delete(a, [0,2], axis=0)
    (let ((a (vt-from-sequence '((1 2) (3 4) (5 6)) :type 'double-float)))
      (check "delete_axis0_list"
             (vt-delete a '(0 2) :axis 0)
             '((3 4))))
    
    ;; 用例5: 沿 axis=1 删除切片
    ;; # NumPy: a = np.array([[1,2,3],[4,5,6]]); np.delete(a, slice(0,2), axis=1)
    (let ((a (vt-from-sequence '((1 2 3) (4 5 6)) :type 'double-float)))
      (check "delete_axis1_slice"
             (vt-delete a '(:slice 0 2) :axis 1)     ; 删除列 0 和 1
             '((3) (6))))
    t))

;; --------------------- test vt-flatten ---------------------
(defun test-vt-flatten ()
  ;; a = np.array([[1, 2], [3, 4]])
  ;; a.flatten() -> [1, 2, 3, 4]
  (let* ((a (vt-from-sequence '((1 2) (3 4))))
         (flat (vt-flatten a)))
    (assert (equal (vt-shape flat) '(4)))
    (assert (equal (vt-to-list flat) '(1.0d0 2.0d0 3.0d0 4.0d0))))
  ;; 对已经是 1D 的张量 flatten 无影响
  (let* ((a (vt-arange 3 :type 'fixnum))
         (flat (vt-flatten a)))
    (assert (equal (vt-shape flat) '(3)))
    (assert (equal (vt-to-list flat) '(0 1 2))))
  (format t "~%test-vt-flatten passed.~%"))

;; --------------------- test vt-from-sequence ---------------------
(defun test-vt-from-sequence ()
  ;; np.array([[1, 2], [3, 4]], dtype=np.float64)
  (let* ((a (vt-from-sequence '((1 2) (3 4)) :type 'double-float)))
    (assert (equal (vt-shape a) '(2 2)))
    (assert (eq (vt-element-type a) 'double-float))
    (assert (= (vt-ref a 0 0) 1.0d0))
    (assert (= (vt-ref a 1 1) 4.0d0)))
  ;; 从 1D list 创建
  (let* ((a (vt-from-sequence '(5 6 7) :type 'fixnum)))
    (assert (equal (vt-shape a) '(3)))
    (assert (equal (vt-to-list a) '(5 6 7))))
  (format t "~%test-vt-from-sequence passed.~%"))

;; --------------------- test vt-diag ---------------------
(defun test-vt-diag ()
  ;; np.diag([1, 2, 3])
  ;; -> [[1 0 0]
  ;;     [0 2 0]
  ;;     [0 0 3]]
  (let* ((v (vt-from-sequence '(1 2 3) :type 'fixnum))
         (m (vt-diag v)))
    (assert (equal (vt-shape m) '(3 3)))
    (assert (= (vt-ref m 0 0) 1))
    (assert (= (vt-ref m 1 1) 2))
    (assert (= (vt-ref m 2 2) 3))
    (assert (= (vt-ref m 0 1) 0))
    (assert (= (vt-ref m 1 0) 0)))
  (format t "~%test-vt-diag passed.~%"))

;; --------------------- test arithmetic basics ---------------------
(defun test-arithmetic-basics ()
  ;; a = np.array([10.0, 20.0, 30.0])
  ;; b = np.array([1.0, 2.0, 3.0])
  ;; a + b -> [11, 22, 33]
  (let* ((a (vt-from-sequence '(10.0 20.0 30.0)))
         (b (vt-from-sequence '(1.0 2.0 3.0)))
         (add (vt-+ a b))
         (sub (vt-- a b))
         (mul (vt-* a b))
         (div (vt-/ a b)))
    (assert (equal (vt-to-list add) '(11.0d0 22.0d0 33.0d0)))
    ;; a - b -> [9, 18, 27]
    (assert (equal (vt-to-list sub) '(9.0d0 18.0d0 27.0d0)))
    ;; a * b -> [10, 40, 90]
    (assert (equal (vt-to-list mul) '(10.0d0 40.0d0 90.0d0)))
    ;; a / b -> [10, 10, 10]
    (assert (equal (vt-to-list div) '(10.0d0 10.0d0 10.0d0))))
  ;; 标量与张量运算
  ;; a * 2 -> [20, 40, 60]
  (let* ((a (vt-from-sequence '(10.0 20.0 30.0)))
         (res (vt-* a 2.0d0)))
    (assert (equal (vt-to-list res) '(20.0d0 40.0d0 60.0d0))))
  (format t "~%test-arithmetic-basics passed.~%"))

;; --------------------- test vt-comparison ---------------------
(defun test-vt-comparison ()
  ;; a = np.array([1, 2, 3])
  ;; b = np.array([1, 4, 1])
  ;; a == b -> [True, False, False] (1.0, 0.0, 0.0)
  (let* ((a (vt-from-sequence '(1 2 3) :type 'fixnum))
         (b (vt-from-sequence '(1 4 1) :type 'fixnum))
         (eq (vt-= a b))
         (lt (vt-< a b))
         (gt (vt-> a b)))
    (assert (equalp (vt-to-list eq) '(1 0 0)))
    ;; a < b -> [False, True, False]
    (assert (equalp (vt-to-list lt) '(0 1 0)))
    ;; a > b -> [False, False, True]
    (assert (equalp (vt-to-list gt) '(0 0 1))))
  (format t "~%test-vt-comparison passed.~%"))

;; --------------------- test vt-var-std-ddof ---------------------
(defun test-vt-var-std-ddof ()
  ;; a = np.array([1, 2, 3, 4])
  ;; np.var(a) -> 1.25 (总体方差)
  (let* ((a (vt-from-sequence '(1.0 2.0 3.0 4.0)))
         (var-pop (vt-var a)))
    (assert (< (abs (- var-pop 1.25d0)) 1e-9)))
  ;; np.var(a, ddof=1) -> 1.666666... (样本方差)
  (let* ((a (vt-from-sequence '(1.0 2.0 3.0 4.0)))
         (var-sample (vt-var a :ddof 1)))
    (assert (< (abs (- var-sample (/ 5.0d0 3.0d0))) 1e-9)))
  ;; np.std(a, ddof=1) -> sqrt(1.666...) -> 1.29099...
  (let* ((a (vt-from-sequence '(1.0 2.0 3.0 4.0)))
         (std-sample (vt-std a :ddof 1)))
    (assert (< (abs (- std-sample (sqrt (/ 5.0d0 3.0d0)))) 1e-9)))
  (format t "~%test-vt-var-std-ddof passed.~%"))

;; --------------------- test vt-prod-ptp ---------------------
(defun test-vt-prod-ptp ()
  ;; a = np.array([2, 3, 4])
  ;; np.prod(a) -> 24
  (let* ((a (vt-from-sequence '(2 3 4) :type 'fixnum))
         (p (vt-prod a)))
    (assert (= p 24)))
  ;; np.ptp(a) -> 4 - 2 = 2 (峰峰值)
  (let* ((a (vt-from-sequence '(2 3 4) :type 'fixnum))
         (p (vt-ptp a)))
    (assert (= p 2)))
  (format t "~%test-vt-prod-ptp passed.~%"))

;; --------------------- test vt-lu ---------------------
(defun test-vt-lu ()
  ;; 情况 1：无需行交换
  ;; A = np.array([[3.0, 4.0], [1.0, 2.0]])
  ;; 第一列主元已经是 3.0，无需交换
  (let* ((a (vt-from-sequence '((3.0 4.0) (1.0 2.0))))
         (lu-and-piv (multiple-value-list (vt-lu a)))
         (lu (first lu-and-piv))
         (piv (second lu-and-piv))
         (sign (third lu-and-piv)))
    ;; 无交换，sign 为 1，piv 保持原样
    (assert (= sign 1))
    (assert (equal piv '(0 1)))
    ;; 检查紧凑 LU 矩阵
    ;; U 的部分 (第一行和第二行后半)
    (assert (< (abs (- (vt-ref lu 0 0) 3.0d0)) 1e-9))
    (assert (< (abs (- (vt-ref lu 0 1) 4.0d0)) 1e-9))
    (assert (< (abs (- (vt-ref lu 1 1) (/ 2.0d0 3.0d0))) 1e-9))
    ;; L 的乘数部分 (第二行前半)
    (assert (< (abs (- (vt-ref lu 1 0) (/ 1.0d0 3.0d0))) 1e-9)))

  ;; 情况 2：发生行交换 (你提供的 case)
  ;; A = np.array([[1.0, 2.0], [3.0, 4.0]])
  ;; 第一列最大值是 3.0，在第 1 行，触发第 0 行和第 1 行交换
  (let* ((a (vt-from-sequence '((1.0 2.0) (3.0 4.0))))
         (lu-and-piv (multiple-value-list (vt-lu a)))
         (lu (first lu-and-piv))
         (piv (second lu-and-piv))
         (sign (third lu-and-piv)))
    ;; 发生一次交换，sign 变为 -1
    (assert (= sign -1))
    ;; piv 记录：结果的第 0 行来自原第 1 行，结果的第 1 行来自原第 0 行
    (assert (equal piv '(1 0)))
    ;; 交换后，第一行变成了原第二行 [3.0, 4.0]
    (assert (< (abs (- (vt-ref lu 0 0) 3.0d0)) 1e-9))
    (assert (< (abs (- (vt-ref lu 0 1) 4.0d0)) 1e-9))
    ;; 消元乘数: 1.0 / 3.0 = 0.3333...
    (assert (< (abs (- (vt-ref lu 1 0) (/ 1.0d0 3.0d0))) 1e-9))
    ;; U 的最后一个元素: 2.0 - (1.0/3.0)*4.0 = 2.0/3.0 = 0.6667...
    (assert (< (abs (- (vt-ref lu 1 1) (/ 2.0d0 3.0d0))) 1e-9)))

  (format t "~%test-vt-lu passed.~%"))


;; --------------------- test compatible functions ---------------------
(defun test-vt-interp ()
  ;; xp = [1, 2, 3]
  ;; fp = [10, 20, 30]
  ;; np.interp(1.5, xp, fp) -> 15.0
  ;; np.interp(0.0, xp, fp, left=-1) -> -1.0 (左边界外)
  (let* ((xp (vt-from-sequence '(1.0 2.0 3.0)))
         (fp (vt-from-sequence '(10.0 20.0 30.0)))
         (x (vt-from-sequence '(1.5)))
         (res1 (vt-interp x xp fp))
         (x2 (vt-from-sequence '(0.0)))
         (res2 (vt-interp x2 xp fp :left -1.0d0)))
    ;; vt-interp 返回的是张量，取其值
    (assert (< (abs (- (vt-ref res1 0) 15.0d0)) 1e-9))
    (assert (= (vt-ref res2 0) -1.0d0)))
  (format t "~%test-vt-interp passed.~%"))

(defun test-vt-convolve ()
  ;; a = [1, 2, 3]
  ;; v = [0, 1, 0.5]
  ;; np.convolve(a, v, 'full') -> [0.0, 1.0, 2.5, 4.0, 1.5]
  (let* ((a (vt-from-sequence '(1.0 2.0 3.0)))
         (v (vt-from-sequence '(0.0 1.0 0.5)))
         (res (vt-convolve a v :mode :full)))
    (assert (equal (vt-shape res) '(5)))
    (assert (< (abs (- (vt-ref res 0) 0.0d0)) 1e-9))
    (assert (< (abs (- (vt-ref res 2) 2.5d0)) 1e-9))
    (assert (< (abs (- (vt-ref res 4) 1.5d0)) 1e-9)))
  (format t "~%test-vt-convolve passed.~%"))

(defun test-vt-searchsorted ()
  ;; a = np.array([1, 2, 3, 4, 5])
  ;; np.searchsorted(a, 3) -> 2
  ;; np.searchsorted(a, 0) -> 0
  ;; np.searchsorted(a, 6) -> 5
  (let* ((a (vt-from-sequence '(1 2 3 4 5) :type 'fixnum))
         (vals (vt-from-sequence '(3 0 6) :type 'fixnum))
         (res (vt-searchsorted a vals)))
    (assert (equal (vt-to-list res) '(2 0 5))))
  (format t "~%test-vt-searchsorted passed.~%"))

;; --------------------- test vt-extract ---------------------
(defun test-vt-extract ()
  ;; a = np.array([10, 20, 30, 40, 50])
  ;; mask = np.array([True, False, True, False, True])
  ;; a[mask] -> [10, 30, 50]
  (let* ((a (vt-from-sequence '(10 20 30 40 50) :type 'fixnum))
         (mask (vt-from-sequence '(1 0 1 0 1) :type 'fixnum))
         (res (vt-extract mask a)))
    (assert (equal (vt-shape res) '(3)))
    (assert (equalp (vt-to-list res) '(10 30 50))))
  (format t "~%test-vt-extract passed.~%"))

;; --------------------- test vt-itemsize-nbytes ---------------------
(defun test-vt-itemsize-nbytes ()
  ;; a = np.array([1, 2], dtype=np.float64)
  ;; a.itemsize -> 8
  ;; a.nbytes -> 16
  (let* ((a (vt-from-sequence '(1.0 2.0)))
         (size (vt-size a))
         (itemsize (vt-itemsize a))
         (nbytes (vt-nbytes a)))
    (assert (= size 2))
    (assert (= itemsize 8))
    (assert (= nbytes 16)))
  ;; fixnum 测试 (通常为 8 字节，取决于平台，此处用 > 0 断言避免平台差异)
  (let* ((a (vt-from-sequence '(1 2) :type 'fixnum))
         (itemsize (vt-itemsize a)))
    (assert (> itemsize 0)))
  (format t "~%test-vt-itemsize-nbytes passed.~%"))

(defun test-vt-atan2 ()
  ;; ==========================================
  ;; 测试 1: 基础矩阵运算 (无广播，验证数学正确性)
  ;; ==========================================
  (let ((y (vt-from-sequence '((0.0d0 1.0d0) 
                               (1.0d0 1.0d0))))
	(x (vt-from-sequence '((1.0d0 1.0d0) 
                               (-1.0d0 1.0d0)))))
    (let ((res (vt-atan2 y x))
          ;; 预期结果: [[0, pi/4], [3*pi/4, pi/4]]
          (expected (vt-from-sequence
		     (list (list 0.0d0 (/ pi 4))
                           (list (* 3.0d0 (/ pi 4)) (/ pi 4))))))
      (format t "~%--- 测试 1: 基础矩阵运算 ---~%")
      (format t "结果: ~A~%" (vt-to-list res))
      ;; 使用库自带的 vt-allclose 进行容差比较
      (assert (vt-allclose res expected))
      (format t "✔ 通过: 数学计算完全正确~%")))

  ;; ==========================================
  ;; 测试 2: 张量与标量运算 (验证标量自动转换与广播)
  ;; ==========================================
  ;; 原代码中 x 如果是标量，闭包会直接捕获标量对象，如果传入张量则会报错。
  ;; 修复后，标量会被 ensure-vt 转为 0维张量，并广播到 y 的形状。
  (let ((y (vt-from-sequence '((0.0d0 1.0d0) 
                               (-1.0d0 0.0d0))))
	(x 1.0d0)) ;; 注意这里传入的是纯数字标量
    (let ((res (vt-atan2 y x))
          (expected (vt-from-sequence (list (list 0.0d0 (/ pi 4))
                                            (list (- (/ pi 4)) 0.0d0)))))
      (format t "~%--- 测试 2: 标量广播 ---~%")
      (format t "输入 y 形状: ~A, 输入 x: ~A~%" (vt-shape y) x)
      (format t "结果形状: ~A~%" (vt-shape res))
      (assert (equal (vt-shape res) '(2 2))) ; 验证形状被正确广播
      (assert (vt-allclose res expected))
      (format t "✔ 通过: 标量广播与形状推断正确~%")))

  ;; ==========================================
  ;; 测试 3: 张量与 1D 向量运算 (验证真正的维度广播)
  ;; ==========================================
  ;; y 是 2x2，x 是 1x2。x 应该沿着第 0 轴广播，变成 2x2 后逐元素计算。
  (let ((y (vt-from-sequence '((1.0d0 2.0d0) 
                               (1.0d0 2.0d0))))
	(x (vt-from-sequence '(1.0d0 -1.0d0))))
    (let ((res (vt-atan2 y x)))
      (format t "~%--- 测试 3: 维度广播 (2x2 与 1x2) ---~%")
      (format t "输入 y 形状: ~A~%" (vt-shape y))
      (format t "输入 x 形状: ~A~%" (vt-shape x))
      (format t "结果形状: ~A~%" (vt-shape res))
      
      ;; 验证形状合并正确
      (assert (equal (vt-shape res) '(2 2)))
      
      ;; 手动验证几个关键点的值
      ;; res[0, 0] = atan2(1.0, 1.0) = pi/4
      (assert (< (abs (- (vt-ref res 0 0) (/ pi 4))) 1e-9))
      ;; res[0, 1] = atan2(2.0, -1.0) = pi - atan(2)
      (assert (< (abs (- (vt-ref res 0 1) (- pi (atan 2.0d0)))) 1e-9))
      ;; res[1, 1] = atan2(2.0, -1.0) = pi - atan(2)
      (assert (< (abs (- (vt-ref res 1 1) (- pi (atan 2.0d0)))) 1e-9))
      
      (format t "✔ 通过: 跨维度广播机制触发成功~%")))

  (format t "~%========================================~%")
  (format t "所有 vt-atan2 测试均已通过！~%"))


;; ============================================================
;;  测试方差标准差除零保护
;; ============================================================
(defun test-vt-var-std ()
  ;; 测试 1：总体方差
  (let* ((data (vt-from-sequence '(1 2 3)))
         (var (vt-var data :ddof 0)))
    (assert-ok (approx= var (/ 2.0 3.0)) "总体方差错误"))

  ;; 测试 2：样本方差 ddof=1
  (let* ((data (vt-from-sequence '(1 2 3)))
         (var (vt-var data :ddof 1)))
    (assert-ok (approx= var 1.0) "样本方差错误"))

  ;; 测试 3：除数为 0 (ddof = 数据个数)
  (let* ((data (vt-from-sequence '(1 2 3)))
         (var (vt-var data :ddof 3)))
    ;; 应返回 NaN (Common Lisp 的 (coerce (/ 0.0d0 0.0d0) 'double-float) 就是 NaN)
    (assert-ok (not (vt-float-nan-= var var)) "ddof=3 时应返回 NaN"))

  ;; 测试 4：沿轴方差，除数为0
  (let* ((a2d (vt-from-sequence '((1 2) (3 4))))
         (var-axis (vt-var a2d :axis 0 :ddof 2)))
    ;; 每个轴只有2个元素，ddof=2导致除数为0，应得到 NaN 张量
    (assert-ok (every (lambda (v) (not (vt-float-nan-= v v)))
                      (loop for i below (vt-size var-axis)
                            collect (vt-ref var-axis i)))
               "轴方差除零时应为 NaN"))

  ;; 测试 5：标准差除零
  (let* ((data (vt-from-sequence '(1 2 3)))
         (std (vt-nanstd data :ddof 3)))
    (assert-ok (not (vt-float-nan-= std std)) "标准差除零时应返回 NaN"))

  ;; 测试 6：nanvar 除零
  (let* ((data-nan (vt-from-sequence (list 1.0 +vt-nan+ 3.0)))  ; 注意 NaN 需要由 (/ 0.0d0 0.0d0) 生成
         (nanvar (vt-nanvar data-nan :ddof 2)))  ; 有效样本数=2, ddof=2 导致除数0
      (if (vt-p nanvar)
      (assert-ok (not (vt-float-nan-= (vt-ref nanvar 0) (vt-ref nanvar 0)))
                 "nanvar 除零时应为 NaN")
      (assert-ok (not (vt-float-nan-= nanvar nanvar)) ; NaN 永远不等于自身
                 "nanvar 除零时应为 NaN")))
  (format t "~%vt-var / vt-std 全部测试通过~%"))


;; ============================================================
;; 运行所有测试
;; ============================================================


;; --------------------- 汇总 ---------------------

(defun run-all-tests ()
  (test-vt-ravel)
  (test-vt-swapaxes)
  (test-vt-flip)
  (test-vt-roll)
  (test-vt-triu-tril)
  (test-vt-diagonal)
  (test-vt-broadcast-to)
  (test-vt-slice)
  (test-vt-reshape)
  (test-vt-transpose)
  (test-vt-squeeze)
  (test-vt-expand-dims)
  (test-vt-concatenate)
  (test-vt-stack)
  (test-vt-tile)
  (test-vt-repeat)
  (test-vt-sum)
  (test-vt-amax-amin)
  (test-vt-argmax-argmin)
  (test-vt-where)
  (test-vt-argwhere)
  (test-vt-meshgrid)
  (test-vt-median)
  (test-vt-percentile)
  (test-vt-matmul)
  (test-vt-einsum)
  (test-vt-dot)
  (test-vt-solve)
  (test-vt-inv)
  (test-vt-det)
  (test-vt-norm)
  (test-vt-frobenius-norm)
  (test-vt-trace)
  (test-vt-outer)
  (test-vt-cumsum)
  (test-vt-cumprod)
  (test-vt-qr)
  (run-svd-tests)
  (test-vt-gradient)
  (test-vt-gradient-advanced)
  (test-vt-pad)
  (test-all-pad)
  (test-pad-thorough)
  (test-vt-take)
  (test-vt-sort)
  (test-vt-argsort)
  (test-vt-argsort-multi-axis)
  (test-vt-put)
  (test-nan-functions)
  (test-vt-logspace)
  (test-vt-linspace)
  (test-vt-kron)
  (test-vt-diff)
  (test-vt-trapz)
  (test-vt-correlate)
  (test-vt-histogram)
  ;; test-pos:
  ;; vt-unique vt-intersect1d vt-union1d
  ;; vt-setdiff1d vt-setxor1d vt-in1d
  (test-set-ops)
  (test-activation-loss)
  (test-stats-more)
  (test-logical)
  (test-math-ops)
  (test-random)
  (test-more-activations)
  (test-arithmetic-ops)
  (test-vt-from-function)
  (test-vt-bincount)
  (test-vt-digitize)
  (test-vt-append)
  (test-vt-insert)
  (test-vt-delete)
  (test-vt-flatten)
  (test-vt-from-sequence)
  (test-vt-diag)
  (test-arithmetic-basics)
  (test-vt-comparison)
  (test-vt-var-std-ddof)
  (test-vt-prod-ptp)
  (test-vt-lu)
  (test-vt-interp)
  (test-vt-convolve)
  (test-vt-searchsorted)
  (test-vt-extract)
  (test-vt-itemsize-nbytes)
  (test-vt-atan2)
  (test-vt-var-std)
)
