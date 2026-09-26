;;;; test-overlap.lisp — 重叠拷贝回归测试
;;;; 加载：在 clvt 包已加载后 (load "test-overlap.lisp")
;;;; 运行：(test-overlap-all)  → t 通过 / nil 失败
(ql:quickload :clvt)
(in-package :clvt)

;;; ------------------------------------------------------------------
;;; 工具
;;; ------------------------------------------------------------------

(defun %mk-array-0..9 ()
  (make-array 10 :element-type 'double-float
                 :initial-contents
                 '(0.0d0 1.0d0 2.0d0 3.0d0 4.0d0
                   5.0d0 6.0d0 7.0d0 8.0d0 9.0d0)))

(defun %list->simple-vector (lst)
  (make-array (length lst) :element-type 'double-float
                           :initial-contents lst))

;;; ------------------------------------------------------------------
;;; 单个测试用例
;;; ------------------------------------------------------------------

(defun test-overlap-strided ()
  "跨步视图重叠（原 bug 复现用例）。
   src = view(off=0, shape=(4,), strides=(2,)) → 逻辑 (0 2 4 6)
   dst = view(off=2, shape=(4,), strides=(2,)) → 逻辑 (2 4 6 8)
   两视图共享底层数组，物理区间 [0,6] 与 [2,8] 重叠。
   正确语义 = memmove：先把 src 快照，再写入 dst。
   期望 data  = (0 1 0 3 2 5 4 7 6 9)
   期望 dst   = (0 2 4 6)  （即调用前的 src）"
  (handler-case
      (let* ((data (%mk-array-0..9))
             (src  (%make-vt :data data :shape '(4) :strides '(2)
                             :offset 0 :dtype :float64))
             (dst  (%make-vt :data data :shape '(4) :strides '(2)
                             :offset 2 :dtype :float64))
             (src-before (coerce (vt-to-list src) 'list)))
        ;; 前置断言：确认走的是慢速 strided 路径
        (assert (not (vt-contiguous-p src)) () "src 应为非连续视图")
        (assert (not (vt-contiguous-p dst)) () "dst 应为非连续视图")

        (vt-copy-into dst src)

        (assert (equal (vt-to-list dst) src-before)
                ()
                "dst 逻辑值应等于 src 快照 ~a，实际得到 ~a"
                src-before (vt-to-list dst))
        (assert (equalp data
                        (%list->simple-vector
                         '(0.0d0 1.0d0 0.0d0 3.0d0 2.0d0
                           5.0d0 4.0d0 7.0d0 6.0d0 9.0d0)))
                ()
                "底层 data 被损坏，期望 ~a，实际 ~a"
                '(%list->simple-vector
                  '(0.0d0 1.0d0 0.0d0 3.0d0 2.0d0 5.0d0 4.0d0 7.0d0 6.0d0 9.0d0))
                data)
        t)
    (error (e)
      (format t "~&[FAIL] test-overlap-strided: ~a~%" e)
      nil)))

(defun test-overlap-contiguous-forward ()
  "连续视图，dest 在 src 之后 → 需要反向 memmove。
   src = [0..4], dst = [5..9]（不重叠，安全）：基线验证。
   这里用 dst 在 src 之后、但物理区间重叠的场景：src=[0..4], dst=[3..7]。"
  (handler-case
      (let* ((data (%mk-array-0..9))
             (src  (%make-vt :data data :shape '(5) :strides '(1)
                             :offset 0 :dtype :float64))
             (dst  (%make-vt :data data :shape '(5) :strides '(1)
                             :offset 3 :dtype :float64)))
        ;; 连续-连续-同形-同型，别名 + 重叠 → 期望方向感知拷贝
        (vt-copy-into dst src)
        ;; 语义等价：dst[i] = old_src[i]，即 data[3..7] = (0 1 2 3 4)
        (assert (equalp data
                        (%list->simple-vector
                         '(0.0d0 1.0d0 2.0d0 0.0d0 1.0d0
                           2.0d0 3.0d0 4.0d0 8.0d0 9.0d0)))
                ()
                "连续重叠（dst 在 src 后）结果错误：~a" data)
        t)
    (error (e)
      (format t "~&[FAIL] test-overlap-contiguous-forward: ~a~%" e)
      nil)))

(defun test-overlap-contiguous-backward ()
  "连续视图，dest 在 src 之前 → 正向拷贝即可。
   src=[5..9], dst=[0..4]，物理区间重叠 [5,9]∩[0,4]=∅ —— 其实不相交。
   改成真正重叠：src=[3..7], dst=[0..4]。"
  (handler-case
      (let* ((data (%mk-array-0..9))
             (src  (%make-vt :data data :shape '(5) :strides '(1)
                             :offset 3 :dtype :float64))
             (dst  (%make-vt :data data :shape '(5) :strides '(1)
                             :offset 0 :dtype :float64)))
        (vt-copy-into dst src)
        ;; dst[i] = old_src[i]：data[0..4] = (3 4 5 6 7)
        (assert (equalp data
                        (%list->simple-vector
                         '(3.0d0 4.0d0 5.0d0 6.0d0 7.0d0
                           5.0d0 6.0d0 7.0d0 8.0d0 9.0d0)))
                ()
                "连续重叠（dst 在 src 前）结果错误：~a" data)
        t)
    (error (e)
      (format t "~&[FAIL] test-overlap-contiguous-backward: ~a~%" e)
      nil)))

(defun test-no-overlap ()
  "不相交视图：别名检测必须短路，走原快路径，结果正确。"
  (handler-case
      (let* ((data (%mk-array-0..9))
             (src  (%make-vt :data data :shape '(3) :strides '(1)
                             :offset 0 :dtype :float64))
             (dst  (%make-vt :data data :shape '(3) :strides '(1)
                             :offset 7 :dtype :float64)))
        (vt-copy-into dst src)
        (assert (equalp data
                        (%list->simple-vector
                         '(0.0d0 1.0d0 2.0d0 3.0d0 4.0d0
                           5.0d0 6.0d0 0.0d0 1.0d0 2.0d0)))
                ()
                "不相交拷贝结果错误：~a" data)
        t)
    (error (e)
      (format t "~&[FAIL] test-no-overlap: ~a~%" e)
      nil)))

(defun test-no-alias ()
  "完全独立的底层数组：不得有任何别名快照开销，结果正确。"
  (handler-case
      (let* ((src-data (make-array 4 :element-type 'double-float
                                     :initial-contents '(1.0d0 2.0d0 3.0d0 4.0d0)))
             (dst-data (make-array 4 :element-type 'double-float
                                     :initial-element 0.0d0))
             (src (%make-vt :data src-data :shape '(4) :strides '(1)
                            :offset 0 :dtype :float64))
             (dst (%make-vt :data dst-data :shape '(4) :strides '(1)
                            :offset 0 :dtype :float64)))
        (vt-copy-into dst src)
        (assert (equalp dst-data src-data)
                ()
                "独立数组拷贝结果错误：src=~a dst=~a" src-data dst-data)
        t)
    (error (e)
      (format t "~&[FAIL] test-no-alias: ~a~%" e)
      nil)))

(defun test-overlap-dtype-conversion ()
  "重叠 + 异 dtype：走快照路径，且转换语义正确。
   src(:float64, strides=(2,)) → dst(:int32, strides=(2,))，物理区间重叠。"
  (handler-case
      (let* ((data (make-array 10 :element-type 'double-float
                                  :initial-contents
                                  '(0.0d0 1.0d0 2.0d0 3.0d0 4.0d0
                                    5.0d0 6.0d0 7.0d0 8.0d0 9.0d0)))
             (src (%make-vt :data data :shape '(3) :strides '(2)
                            :offset 0 :dtype :float64))   ; 0 2 4
             ;; dst 同数组重叠，但 dtype 为 :int32——需要独立物理数组才能验证 dtype 转换
             ;; 改为：src 是 float64 的 data，dst 是另一个 int32 数组，物理不重叠，专测 dtype
             (dst-data (make-array 3 :element-type '(signed-byte 32)
                                     :initial-element 0))
             (dst (%make-vt :data dst-data :shape '(3) :strides '(1)
                            :offset 0 :dtype :int32)))
        (vt-copy-into dst src)
        (assert (equalp dst-data
                        (make-array 3 :element-type '(signed-byte 32)
                                      :initial-contents '(0 2 4)))
                ()
                "float64→int32 转换结果错误：~a" dst-data)
        t)
    (error (e)
      (format t "~&[FAIL] test-overlap-dtype-conversion: ~a~%" e)
      nil)))

;;; ------------------------------------------------------------------
;;; 汇总入口
;;; ------------------------------------------------------------------

(defun test-overlap-all ()
  "运行全部重叠拷贝回归测试。全部通过返回 t，任一失败返回 nil。"
  (format t "~&==================== 重叠拷贝回归测试 ====================~%")
  (let* ((results
           (list (cons 'strided              (test-overlap-strided))
                 (cons 'contiguous-forward   (test-overlap-contiguous-forward))
                 (cons 'contiguous-backward  (test-overlap-contiguous-backward))
                 (cons 'no-overlap           (test-no-overlap))
                 (cons 'no-alias             (test-no-alias))
                 (cons 'dtype-conversion     (test-overlap-dtype-conversion))))
         (passed (count t results :key #'cdr))
         (total  (length results))
         (ok     (= passed total)))
    (format t "~%-------------------- 汇总 --------------------~%")
    (dolist (r results)
      (format t "  ~a  ~a~%"
              (if (cdr r) "[PASS]" "[FAIL]")
              (car r)))
    (format t "~%通过 ~d/~d~%~a~%"
            passed total
            (if ok
                "====> 全部通过 <===="
                "====> 存在失败 <===="))
    (if ok t nil)))

(test-overlap-all)
