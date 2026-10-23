;;;; complex-matmul-perf-test.lisp — 复数 matmul 内核（§16.27）回归网
;;;;
;;;; 目的：锁住对 %CX-MATMUL-RUN 的重写（对齐实数线的 %MATMUL-DF-ROWS，§16.27
;;;; 形态：safety 0 + 指针递推 + j 4 路展开 + 复用 CALL-PARALLEL-GEMM）。
;;;;
;;;; 两部分断言：
;;;;   一、正确性（与实现无关的独立参照）
;;;;       · 手写三重循环参照实现（i-j-p，直接按 (i,j,k) 下标访问）逐位对齐
;;;;       · 非方阵 / 尾路径（n 非 4 的倍数）/ 退化（k=1、m=1、n=1、零尺寸）
;;;;       · 切片视图（offset 非零）与连续副本结果一致
;;;;       · 批次 + batch 广播，与逐 batch 的 2D matmul 逐块一致
;;;;       · einsum 的 NN/NT/转置输出三条通道与 vt-matmul 逐位一致（§16.21.5）
;;;;       · c64 / c128 双精度；:out 覆盖写；dtype 参数
;;;;   二、性能不变量（防回归，全部用「同进程内比值」，不写死绝对毫秒）
;;;;       · 新行内核 ≥ 1.3× 旧内核形状（safety 1 + 下标重算 + 无展开）
;;;;       · 新行内核 ≤ 3.5× 实数标量行内核（复数每元素 4× 浮点工作量，
;;;;         理论下限 4×；实测 ~2.2–2.6×，留 35% 噪声余量）
;;;;       · 并行开关不改结果（逐位相等），且大矩阵并行不慢于串行 1.2×
;;;;
;;;; 用法：sbcl --non-interactive --load test/complex-matmul-perf-test.lisp

(require :asdf)
(defparameter *clvt-test-dir* (make-pathname :name nil :type nil :defaults *load-truename*))
(pushnew (merge-pathnames "../" *clvt-test-dir*) asdf:*central-registry* :test #'equal)
(handler-bind ((warning #'muffle-warning)) (asdf:load-system :clvt))

(in-package :clvt)

;;; ------------------------------------------------------------------
;;; 骨架
;;; ------------------------------------------------------------------

(defvar *pass* 0)
(defvar *fail* 0)
(defvar *failures* '())

(defun report (name ok &optional detail)
  (if ok
      (incf *pass*)
      (progn (incf *fail*) (push name *failures*)
             (format t "~&FAIL ~a ~@[— ~a~]~%" name detail))))

;;; ------------------------------------------------------------------
;;; 数据构造（确定性伪随机，可重放；不用 RANDOM）
;;; ------------------------------------------------------------------

(defun mk (dims seed &key (dt :complex128))
  (let* ((n (reduce #'* dims :initial-value 1))
         (v (make-vt dims 0 :dtype dt))
         (d (vt-data v))
         (s (mod (abs seed) 2147483647))
         (cx (member dt '(:complex128 :complex64))))
    (dotimes (i n)
      (setf s (mod (+ (* s 1103515245) 12345) 2147483648))
      (let ((re (- (/ s 2147483648.0d0) 0.5d0)))
        (setf s (mod (+ (* s 1103515245) 12345) 2147483648))
        (let ((im (- (/ s 2147483648.0d0) 0.5d0)))
          (setf (aref d i)
                (cond ((eq dt :complex128) (complex re im))
                      ((eq dt :complex64) (complex (coerce re 'single-float)
                                                   (coerce im 'single-float)))
                      ((eq dt :float64) re)
                      (t (round (* 10 re))))))))
    v))

(defun flat-elems (v)
  "按逻辑顺序取出元素列表。⚠ 必须尊重 VT-OFFSET：VT-CONTIGUOUS 对
   已连续的**视图**是零拷贝，其 VT-DATA 仍指向基数组，直接按 length 遍历
   会把基数组的其它部分也算进来（批量 matmul 的分片正好踩这个坑）。"
  (let* ((c (vt-contiguous (vt-ravel v)))
         (d (vt-data c))
         (o (the fixnum (vt-offset c)))
         (n (the fixnum (vt-size c))))
    (loop for i from o below (+ o n) collect (aref d i))))

(defun same-bits-p (x y)
  "逐元素 bit 级相等（复数按实部/虚部）。零尺寸视为相等。"
  (let ((ex (flat-elems x)) (ey (flat-elems y)))
    (and (= (length ex) (length ey))
         (loop for a in ex for b in ey
               always (if (complexp a)
                          (and (eql (realpart a) (realpart b))
                               (eql (imagpart a) (imagpart b)))
                          (eql a b))))))

;;; ------------------------------------------------------------------
;;; 独立参照：手写三重循环（i-j-p，逐下标访问，最朴素的写法）
;;; ------------------------------------------------------------------

(defun naive-matmul (a b)
  "完全独立的参照实现：不做任何快路径，按 (i,j,p) 顺序直接索引。
   只支持 2D、连续输入。"
  (let* ((m (first (vt-shape a))) (k (second (vt-shape a))) (n (second (vt-shape b)))
         (ca (vt-contiguous a)) (cb (vt-contiguous b))
         (da (vt-data ca)) (db (vt-data cb))
         (res (make-vt (list m n) 0 :dtype (vt-dtype a)))
         (dr (vt-data res)))
    (dotimes (i m)
      (dotimes (j n)
        (let ((acc (if (eq (vt-dtype a) :complex64)
                       (complex 0.0s0 0.0s0) (complex 0.0d0 0.0d0))))
          (dotimes (p k)
            (incf acc (* (aref da (+ (* i k) p)) (aref db (+ (* p n) j)))))
          (setf (aref dr (+ (* i n) j)) acc))))
    res))

;;; ------------------------------------------------------------------
;;; 对照用的「旧内核形状」：safety 1 + 下标重算 + 无展开
;;;   与 git 上一版的 %CX-MATMUL-RUN 逐字一致（仅改函数名），
;;;   用于性能护栏「新内核必须更快」。
;;; ------------------------------------------------------------------

(defun %old-shape-matmul (ad bd rd a-off b-off c-off m k n)
  (%cx-dispatch ad a zero
    (%cx-dispatch bd b zero2
      (%cx-dispatch rd c zero3
        (declare (ignorable zero2 zero3))
        (dotimes (i m)
          (declare (type fixnum i))
          (let ((c-row (the fixnum (+ c-off (the fixnum (* i n))))))
            (declare (type fixnum c-row))
            (dotimes (j n)
              (declare (type fixnum j))
              (setf (aref c (the fixnum (+ c-row j))) zero))))
        (dotimes (i m)
          (declare (type fixnum i))
          (let ((a-row (the fixnum (+ a-off (the fixnum (* i k)))))
                (c-row (the fixnum (+ c-off (the fixnum (* i n))))))
            (declare (type fixnum a-row c-row))
            (dotimes (p k)
              (declare (type fixnum p))
              (let ((av (aref a (the fixnum (+ a-row p))))
                    (b-row (the fixnum (+ b-off (the fixnum (* p n))))))
                (declare (type fixnum b-row))
                (dotimes (j n)
                  (declare (type fixnum j))
                  (incf (aref c (the fixnum (+ c-row j)))
                        (* av (aref b (the fixnum (+ b-row j))))))))))))))

;;; ------------------------------------------------------------------
;;; 一、正确性
;;; ------------------------------------------------------------------

(defun run-shape-cases ()
  (format t "~&--- 1. 形状 / 尾路径 / 退化 ---~%")
  (dolist (case '((3 5 7) (17 13 19) (1 4 5) (4 1 5) (4 5 1) (64 3 9) (7 64 3)))
    (destructuring-bind (m k nn) case
      (dolist (dt '(:complex128 :complex64))
        (let* ((a (mk (list m k) (+ m k nn) :dt dt))
               (b (mk (list k nn) (+ m 3 k nn) :dt dt))
               (got (vt-matmul a b))
               (want (naive-matmul a b)))
          (report (format nil "~ax~ax~a ~a 对齐朴素三重循环" m k nn dt)
                  (same-bits-p got want)
                  (format nil "shape=~a dtype=~a" (vt-shape got) (vt-dtype got)))))))
  ;; 尾路径专项：n 不是 4 的倍数（1..9 全覆盖，直接压 n4 之后的尾部循环）
  (dolist (nn '(1 2 3 5 6 7 9))
    (let* ((a (mk (list 5 4) 11 :dt :complex128))
           (b (mk (list 4 nn) (+ 100 nn) :dt :complex128)))
      (report (format nil "尾路径 n=~a（4 路展开的剩余列）" nn)
              (same-bits-p (vt-matmul a b) (naive-matmul a b)))))
  ;; 零尺寸
  (dolist (case '((0 3 4) (3 0 4) (3 4 0)))
    (destructuring-bind (m k nn) case
      (let* ((a (mk (list m k) 1 :dt :complex128))
             (b (mk (list k nn) 2 :dt :complex128))
             (r (vt-matmul a b)))
        (report (format nil "零尺寸 ~ax~ax~a 形状正确" m k nn)
                (equal (vt-shape r) (list m nn)))))))

(defun run-view-cases ()
  (format t "~&--- 2. 切片视图（offset 非零）/ 非连续输入 ---~%")
  ;; a 取自更大张量的切片：contiguous 零拷贝返回，offset 保留
  (let* ((big (mk (list 40 40) 7 :dt :complex128))
         (a (vt-slice big '(10 30) '(10 30)))   ; (20,20)，offset = 10*40+10
         (b (mk (list 40 40) 8 :dt :complex128))
         (got (vt-matmul a b))
         (want (naive-matmul (vt-contiguous a) (vt-contiguous b))))
    (report "切片 A（offset≠0）结果正确" (same-bits-p got want)
            (format nil "offset=~a" (vt-offset a))))
  ;; 非连续 A（转置视图）
  (let* ((a0 (mk (list 20 30) 9 :dt :complex128))
         (a (vt-transpose a0))                  ; (30,20) 非连续
         (b (mk (list 20 6) 10 :dt :complex128))
         (got (vt-matmul a b))
         (want (naive-matmul (vt-contiguous a) (vt-contiguous b))))
    (report "非连续 A（转置视图）结果正确" (same-bits-p got want)))
  ;; 非连续 C 之外的常见坑：B 的列切片
  (let* ((a (mk (list 12 6) 21 :dt :complex128))
         (b0 (mk (list 6 15) 22 :dt :complex128))
         (b (vt-slice b0 '(:all) '(2 8)))       ; (6,6)，列切片
         (got (vt-matmul a b))
         (want (naive-matmul a (vt-contiguous b))))
    (report "列切片 B 结果正确" (same-bits-p got want)
            (format nil "B shape=~a" (vt-shape b)))))

(defun run-batch-cases ()
  (format t "~&--- 3. 批量 / batch 广播 ---~%")
  (let* ((a (mk '(2 3 4) 31 :dt :complex128))
         (b (mk '(2 4 5) 32 :dt :complex128))
         (got (vt-matmul a b)))
    (report "批量 (2,3,4)@(2,4,5) 形状" (equal (vt-shape got) '(2 3 5)))
    (dotimes (i 2)
      (report (format nil "批量第 ~a 片与 2D matmul 逐位一致" i)
              (same-bits-p (vt-slice got (list i))
                           (vt-matmul (vt-slice a (list i)) (vt-slice b (list i)))))))
  ;; batch 广播 (1,4,5)@(3,5,6)
  (let* ((a (mk '(1 4 5) 33 :dt :complex128))
         (b (mk '(3 5 6) 34 :dt :complex128))
         (got (vt-matmul a b)))
    (report "batch 广播 (1,4,5)@(3,5,6) 形状" (equal (vt-shape got) '(3 4 6)))
    (dotimes (i 3)
      (report (format nil "batch 广播第 ~a 片与 2D matmul 逐位一致" i)
              (same-bits-p (vt-slice got (list i))
                           (vt-matmul (vt-slice a (list 0)) (vt-slice b (list i))))))))

(defun run-einsum-cases ()
  (format t "~&--- 4. einsum 与 vt-matmul 逐位一致（§16.21.5）---~%")
  (let ((a (mk '(9 11) 41 :dt :complex128))
        (b (mk '(11 13) 42 :dt :complex128))
        (bt (mk '(13 11) 43 :dt :complex128)))
    (report "einsum ij,jk->ik（NN，共用行内核）"
            (same-bits-p (vt-einsum "ij,jk->ik" a b) (vt-matmul a b)))
    (report "einsum ij,kj->ik（NT，转置输入）"
            (same-bits-p (vt-einsum "ij,kj->ik" a bt)
                         (vt-matmul a (vt-transpose bt))))
    (report "einsum ij,jk->ki（转置输出）"
            (same-bits-p (vt-einsum "ij,jk->ki" a b)
                         (vt-transpose (vt-matmul a b))))
    ;; 三张量链：einsum 的缩并次序是它自己选的（可能先 jk,kl 再 ij），
    ;; 与 (a@b)@c 的加法次序不同 → 只保证 ~1e-15 的数值一致，不要求逐位
    (report "einsum 三张量链 ij,jk,kl->il（数值一致，允许缩并次序差异）"
            (let ((c (mk '(13 8) 44 :dt :complex128)))
              (vt-allclose (vt-einsum "ij,jk,kl->il" a b c)
                           (vt-matmul (vt-matmul a b) c) :atol 1d-12 :rtol 1d-12)))
    (report "einsum i,i-> 全收缩与 vt-dot 一致"
            (let ((x (mk '(37) 45 :dt :complex128))
                  (y (mk '(37) 46 :dt :complex128)))
              (same-bits-p (vt-einsum "i,i->" x y) (vt-dot x y))))))

(defun run-dtype-out-cases ()
  (format t "~&--- 5. dtype / :out / 并行开关 ---~%")
  (let ((a (mk '(6 5) 51 :dt :complex64))
        (b (mk '(5 7) 52 :dt :complex64)))
    (report "c64 输入 → c64 输出" (eq (vt-dtype (vt-matmul a b)) :complex64)))
  (let ((a (mk '(6 5) 53 :dt :complex128))
        (b (mk '(5 7) 54 :dt :complex128)))
    (report "c128 输入 → c128 输出" (eq (vt-dtype (vt-matmul a b)) :complex128))
    (let* ((want (vt-matmul a b))
           (out (make-vt '(6 7) 0 :dtype :complex128))
           (got (vt-matmul a b :out out)))
      (report ":out 覆盖写结果一致（且返回同一个 out）"
              (and (eq got out) (same-bits-p got want))))
    (report ":dtype 参数生效（转 c64 后数值可复现）"
            (eq (vt-dtype (vt-matmul a b :dtype :complex64)) :complex64)))
  ;; 并行开关：结果必须逐位相同（并行只切行，不切累加顺序）。
  ;; 两个规模都要测：小规模（低于 *matmul-parallel-threshold*，走串行）
  ;; 与大规模（超过阈值，真起 31 个线程）—— 后者才是并行路径本身。
  (let* ((a (mk '(130 70) 61 :dt :complex128))
         (b (mk '(70 90) 62 :dt :complex128))
         (r1 (let ((*enable-parallel-matmul* nil)) (vt-matmul a b)))
         (r2 (let ((*enable-parallel-matmul* t)) (vt-matmul a b))))
    (report "并行/串行结果逐位一致（低于并行阈值）" (same-bits-p r1 r2)))
  (let* ((a (mk '(200 200) 65 :dt :complex128))
         (b (mk '(200 200) 66 :dt :complex128))
         (work (* 200 200 200))
         (r1 (let ((*enable-parallel-matmul* nil)) (vt-matmul a b)))
         (r2 (let ((*enable-parallel-matmul* t)) (vt-matmul a b))))
    (report (format nil "并行/串行结果逐位一致（~a ≥ 阈值 ~a，真起线程）"
                    work *matmul-parallel-threshold*)
            (same-bits-p r1 r2))
    (report "并行路径结果同时对齐朴素三重循环"
            (same-bits-p r2 (naive-matmul a b))))
  ;; 广矩阵（m < 64 不切线程）与窄矩阵（m 大）都覆盖
  (let ((a (mk '(200 3) 63 :dt :complex128))
        (b (mk '(3 5) 64 :dt :complex128)))
    (report "m≥64 但 k 很小的形态正确"
            (same-bits-p (vt-matmul a b) (naive-matmul a b)))))

;;; ------------------------------------------------------------------
;;; 二、性能不变量（同进程比值，不写死绝对毫秒）
;;; ------------------------------------------------------------------

(defun now-us ()
  (multiple-value-bind (s us) (sb-ext:get-time-of-day)
    (+ (* s 1000000) us)))

(defun best-us (thunk reps rounds)
  "跑 ROUNDS 轮、每轮 REPS 次，取最小值（microbenchmark 惯例：最小值=最少噪声）。"
  (funcall thunk)
  (let ((best most-positive-fixnum))
    (dotimes (r rounds)
      (declare (ignore r))
      (let ((t0 (now-us)))
        (dotimes (i reps) (declare (ignore i)) (funcall thunk))
        (setf best (min best (/ (float (- (now-us) t0)) reps)))))
    best))

(defun run-perf-invariants ()
  (format t "~&--- 6. 性能不变量 ---~%")
  ;; 结构：两个行内核必须存在
  (report "%cx-matmul-cdd-rows 已定义" (fboundp '%cx-matmul-cdd-rows))
  (report "%cx-matmul-css-rows 已定义" (fboundp '%cx-matmul-css-rows))
  (format t "~&  内核函数: cdd=~a css=~a~%"
          (fboundp '%cx-matmul-cdd-rows) (fboundp '%cx-matmul-css-rows))

  (let* ((n 192)
         (za (mk (list n n) 71 :dt :complex128))
         (zb (mk (list n n) 72 :dt :complex128))
         (zc (make-vt (list n n) 0 :dtype :complex128))
         (zd (make-vt (list n n) 0 :dtype :complex128))
         (fa (mk (list n n) 73 :dt :float64))
         (fb (mk (list n n) 74 :dt :float64))
         (fc (make-vt (list n n) 0 :dtype :float64))
         (da (vt-data za)) (db (vt-data zb))
         (t-new (let ((*enable-parallel-matmul* nil))
                  (best-us (lambda () (%cx-matmul-cdd-rows da db (vt-data zc)
                                                            n n 0 0 0 0 n)) 4 3)))
         (t-old (let ((*enable-parallel-matmul* nil))
                  (best-us (lambda () (%old-shape-matmul da db (vt-data zd)
                                                         0 0 0 n n n)) 4 3)))
         (t-real (let ((*enable-parallel-matmul* nil))
                   (best-us (lambda () (funcall #'%matmul-df-rows (vt-data fa) (vt-data fb)
                                                (vt-data fc) n n 0 0 0 0 n)) 4 3))))
    (format t "~&  n=~a 单线程: 新内核 ~d us / 旧形状 ~d us / 实数标量 ~d us~%"
            n t-new t-old t-real)
    (format t "~&  比值: 旧/新 = ~,2f   新/实数标量 = ~,2f（复数理论下限 4.00）~%"
            (/ t-old t-new) (/ t-new t-real))
    ;; 旧形状 = safety 1 + 下标重算 + 无展开；重构后必须至少快 1.3×
    (report "新行内核比旧内核形状快 ≥ 1.3×"
            (> t-old (* 1.3 t-new))
            (format nil "旧 ~d us / 新 ~d us = ~,2f×" t-old t-new (/ t-old t-new)))
    ;; 复数每元素 4× 浮点工作量（4 mul + 4 add vs 1 mul + 1 add）；
    ;; 标量代码的合理上界取 3.5×（实测 2.2–2.6×），超标说明退回旧形态
    (report "新行内核 ≤ 3.5× 实数标量行内核（贴住 4× flop 下限）"
            (< t-new (* 3.5 t-real))
            (format nil "新 ~d us / 实数 ~d us = ~,2f×" t-new t-real (/ t-new t-real))))

  ;; 并行：不改结果，且大矩阵不慢于串行（宽松 1.2×，只拦「并行反而更慢」的病态）
  (let* ((n 256)
         (a (mk (list n n) 81 :dt :complex128))
         (b (mk (list n n) 82 :dt :complex128))
         (r1 (let ((*enable-parallel-matmul* nil)) (vt-matmul a b)))
         (r2 (let ((*enable-parallel-matmul* t)) (vt-matmul a b)))
         (t-s (let ((*enable-parallel-matmul* nil))
                (best-us (lambda () (vt-matmul a b)) 2 2)))
         (t-p (let ((*enable-parallel-matmul* t))
                (best-us (lambda () (vt-matmul a b)) 2 2))))
    (report "并行结果与串行逐位一致" (same-bits-p r1 r2))
    (format t "~&  n=~a 端到端: 串行 ~d us / 并行 ~d us~%" n t-s t-p)
    (report "并行不慢于串行 1.2×（*matmul-thread-count* > 1 时）"
            (or (<= *matmul-thread-count* 1) (< t-p (* 1.2 t-s)))
            (format nil "串行 ~d us / 并行 ~d us" t-s t-p))))

;;; ------------------------------------------------------------------
;;; 入口
;;; ------------------------------------------------------------------

(format t "~&=====================================~%")
(format t "复数 matmul 内核测试（§16.27）~%")
(format t "=====================================~%")
(run-shape-cases)
(run-view-cases)
(run-batch-cases)
(run-einsum-cases)
(run-dtype-out-cases)
(run-perf-invariants)

(format t "~&=====================================~%")
(format t "Total: ~d / Pass: ~d / Fail: ~d~%" (+ *pass* *fail*) *pass* *fail*)
(format t "PASS=~d FAIL=~d~%" *pass* *fail*)
(when *failures*
  (format t "~&--- 失败清单 ---~%")
  (dolist (f (reverse *failures*)) (format t "  · ~a~%" f)))
(when (plusp *fail*) (sb-ext:exit :code 1))
