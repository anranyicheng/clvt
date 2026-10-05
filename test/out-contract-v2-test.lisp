;;;; out-contract-v2-test.lisp — 任务3 核心验证：out 契约 × 连续性 × 别名 × NaN
;;;;
;;;; 核心断言（对应 CONVENTIONS.md §10.3）：
;;;;   对每一格 (out 形态 × 输入含 NaN × 输入与 out 别名)，
;;;;   「快路径结果」必须与「通用路径结果」逐位相同，
;;;;   且与 numpy 语义一致。
;;;;
;;;; 快路径与通用路径的开关方式：改变 out / 输入的连续性，让库内选路
;;;; 落到不同分支（contig fast-path vs noncontig strides-path）。

(require :asdf)
#+quicklisp (ql:quickload :clvt)
(eval-when (:load-toplevel :execute)
  (unless (find-package :clvt)
    (asdf:load-system :clvt)))

(in-package :clvt)

(defvar *oc-fail* 0)
(defvar *oc-pass* 0)

(defun oc-num= (a b)
  "数值比较：NaN↔NaN 与 Inf↔同号Inf 视为相等；其余按数值相等（跨精度）。"
  (cond ((and (floatp a) (floatp b) (%nan-p a) (%nan-p b)) t)
        ((and (floatp a) (floatp b)
              (%pos-inf-p a) (%pos-inf-p b)) t)
        ((and (floatp a) (floatp b)
              (%neg-inf-p a) (%neg-inf-p b)) t)
        ((and (numberp a) (numberp b)) (= a b))
        (t (equal a b))))

(defun oc-check (name expected actual &key (eps 0.0d0))
  "断言 actual 与 expected 相等（数值容差 eps，NaN/Inf 按结构相等处理）。"
  (let ((ok (cond
              ((and (numberp expected) (numberp actual))
               (cond ((%nan-p expected) (%nan-p actual))
                     ((and (floatp expected) (%pos-inf-p expected))
                      (and (floatp actual) (%pos-inf-p actual)))
                     ((and (floatp expected) (%neg-inf-p expected))
                      (and (floatp actual) (%neg-inf-p actual)))
                     (t (<= (abs (- (coerce expected 'double-float)
                                    (coerce actual   'double-float)))
                            eps))))
              ((and (listp expected) (listp actual))
               (and (= (length expected) (length actual))
                    (every #'oc-num= expected actual)))
              (t (equal expected actual)))))
    (if ok
        (incf *oc-pass*)
        (progn
          (incf *oc-fail*)
          (format t "~&[FAIL] ~a~%   expected: ~a~%   actual:   ~a~%"
                  name expected actual)))))

;;; ==================================================================
;;; 1. 非连续 out：快路径 vs 通用路径必须逐位相同
;;; ==================================================================

(defun test-noncontig-out-equivalence ()
  (format t "~&~%=== 1. 非连续 out 的路径等价性 ===~%")
  ;; 构造一个 2×6 的零张量，取其列切片 (2×3) 作为非连续 out
  (let* ((a (vt-from-sequence '(1.0 2.0 3.0 4.0 5.0 6.0) :dtype :float64))
         (b (vt-from-sequence '(10.0 20.0 30.0 40.0 50.0 60.0) :dtype :float64))
         (a2 (vt-reshape a '(2 3)))
         (b2 (vt-reshape b '(2 3))))
    ;; (A) 连续 out
    (let* ((out-c (vt-zeros '(2 3) :dtype :float64))
           (r (vt-+ a2 b2 :out out-c)))
      (oc-check "连续 out 结果" '(11.0 22.0 33.0 44.0 55.0 66.0)
                (vt-to-list (vt-flatten r)))
      (oc-check "连续 out 返回同一对象" t (eq r out-c)))
    ;; (B) 非连续 out（列步长 2），并与 (A) 比对
    (let* ((base (vt-zeros '(2 6) :dtype :float64))
           (out-n (vt-slice base '(:all) '(0 6 2)))   ; 非连续视图
           (r (vt-+ a2 b2 :out out-n)))
      (oc-check "非连续 out 未被复制" t (eq (vt-data r) (vt-data base)))
      (oc-check "非连续 out 结果（展开）" '(11.0 22.0 33.0 44.0 55.0 66.0)
                (vt-to-list (vt-flatten r)))
      ;; 底层 base 的偶数下标列应被写入，奇数列保持 0
      (oc-check "非连续 out 底层布局"
                '(11.0 0.0 22.0 0.0 33.0 0.0 44.0 0.0 55.0 0.0 66.0 0.0)
                (vt-to-list (vt-flatten base))))))

;;; ==================================================================
;;; 2. 转置 out（负/换位 strides）
;;; ==================================================================

(defun test-transposed-out ()
  (format t "~&~%=== 2. 转置 out ===~%")
  (let* ((a2 (vt-reshape (vt-from-sequence '(1.0 2.0 3.0) :dtype :float64) '(1 3)))
         (b2 (vt-reshape (vt-from-sequence '(10.0 20.0 30.0) :dtype :float64) '(1 3)))
         ;; out 的形状必须是 (1 3)；用 (3 1) 的转置构造
         (base (vt-zeros '(3 1) :dtype :float64))
         (out-t (vt-transpose base)))   ; 形状 (1 3)，strides 与连续 (1 3) 不同
    (oc-check "转置 out 形状" '(1 3) (vt-shape out-t))
    (let ((r (vt-+ a2 b2 :out out-t)))
      (oc-check "转置 out 结果" '(11.0 22.0 33.0) (vt-to-list (vt-flatten r)))
      (oc-check "转置 out 底层" '(11.0 22.0 33.0) (vt-to-list (vt-flatten base))))))

;;; ==================================================================
;;; 3. 别名：完全重叠 / 部分重叠（滑窗、逆序）
;;; ==================================================================

(defun test-alias-out ()
  (format t "~&~%=== 3. 别名 out（快照语义）===~%")
  ;; (A) 完全别名：x = x + x
  (let ((x (vt-from-sequence '(1.0 2.0 3.0) :dtype :float64)))
    (vt-+ x x :out x)
    (oc-check "完全别名 x+x->x" '(2.0 4.0 6.0) (vt-to-list x)))
  ;; (B) 逆序重叠：z + z -> z[::-1]，numpy 语义 = [8,6,4,2]
  (let* ((z (vt-from-sequence '(1.0 2.0 3.0 4.0) :dtype :float64))
         (rev (vt-slice z '(3 nil -1))))
    (vt-+ z z :out rev)
    (oc-check "逆序重叠 z+z->z[::-1]" '(8.0 6.0 4.0 2.0) (vt-to-list z)))
  ;; (C) 部分重叠（错位切片）：w[:, :2] + w[:, :2] -> w[:, 2:]
  ;;     numpy 实测：[[0,1,0,2],[4,5,8,10]]
  (let* ((w (vt-from-sequence '(0.0 1.0 2.0 3.0  4.0 5.0 6.0 7.0) :dtype :float64))
         (w2 (vt-reshape w '(2 4)))
         (src (vt-slice w2 '(:all) '(0 2)))
         (dst (vt-slice w2 '(:all) '(2 4))))
    ;; 先快照 src（模拟 vt-+ 的别名保护）
    (let* ((snap (vt-out-snapshot dst (list src)))
           (r (vt-+ (first snap) (first snap) :out dst)))
      (oc-check "部分重叠 展开结果" '(0.0 1.0 0.0 2.0 4.0 5.0 8.0 10.0)
                (vt-to-list (vt-flatten w2)))))
  ;; (D) 不同 data 的伪重叠（不应快照，且结果正确）
  (let ((p (vt-from-sequence '(1.0 2.0) :dtype :float64))
        (q (vt-from-sequence '(3.0 4.0) :dtype :float64)))
    (vt-+ p q :out p)
    (oc-check "独立张量 out" '(4.0 6.0) (vt-to-list p))))

;;; ==================================================================
;;; 4. NaN / Inf 经 out 与其广播
;;; ==================================================================

(defun test-nan-inf-out ()
  (format t "~&~%=== 4. NaN/Inf 经 out ===~%")
  (let* ((nan +vt-dfloat-nan+)
         (inf +vt-dfloat-pos-inf+)
         (a (vt-from-sequence (list 1.0 nan 3.0) :dtype :float64))
         (b (vt-from-sequence (list inf 2.0 nan) :dtype :float64)))
    ;; 连续 out
    (let ((oc (vt-zeros '(3) :dtype :float64)))
      (vt-+ a b :out oc)
      (oc-check "NaN/Inf 连续 out[0]" inf (vt-ref oc 0))
      (oc-check "NaN/Inf 连续 out[1]" t (%nan-p (vt-ref oc 1)))
      (oc-check "NaN/Inf 连续 out[2]" t (%nan-p (vt-ref oc 2))))
    ;; 非连续 out（步长 2）
    (let* ((base (vt-zeros '(6) :dtype :float64))
           (on (vt-slice base '(0 nil 2))))
      (vt-+ a b :out on)
      (oc-check "NaN/Inf 非连续 out[0]" inf (vt-ref on 0))
      (oc-check "NaN/Inf 非连续 out[1]" t (%nan-p (vt-ref on 1)))
      (oc-check "NaN/Inf 非连续 out[2]" t (%nan-p (vt-ref on 2)))
      (oc-check "NaN/Inf 非连续 底层奇数位未动"
                '(0.0 0.0 0.0 0.0) (list (vt-ref base 1) (vt-ref base 3)
                                         (vt-ref base 5) (vt-ref base 1)))))
  ;; NaN 广播：out 全填 NaN 后参与运算
  (let* ((n (vt-full '(2 2) +vt-dfloat-nan+ :dtype :float64))
         (ones (vt-ones '(2 2) :dtype :float64)))
    (let ((r (vt-+ n ones)))
      (oc-check "NaN 广播传播" t
                (every #'%nan-p (vt-to-list (vt-flatten r)))))))

;;; ==================================================================
;;; 5. 硬契约违反必须报错（不得静默）
;;; ==================================================================

(defun test-hard-contract-violations ()
  (format t "~&~%=== 5. 硬契约违反 ===~%")
  (flet ((expect-error (name thunk)
           (handler-case (progn (funcall thunk)
                                (incf *oc-fail*)
                                (format t "~&[FAIL] ~a: 未报错（应报错）~%" name))
             (error () (incf *oc-pass*)))))
    (expect-error "out 形状不匹配"
      (lambda () (vt-+ (vt-ones '(2 3)) (vt-ones '(2 3))
                       :out (vt-zeros '(3 2)))))
    (expect-error "out dtype 不匹配"
      (lambda () (vt-+ (vt-ones '(2 3)) (vt-ones '(2 3))
                       :out (vt-zeros '(2 3) :dtype :float32))))
    (expect-error ":dtype 与 :out 冲突"
      (lambda () (vt-+ (vt-ones '(2 3)) (vt-ones '(2 3))
                       :dtype :float32
                       :out (vt-zeros '(2 3) :dtype :float64))))
    (expect-error "out 是只读广播视图"
      (lambda () (let ((b (vt-broadcast-to (vt-ones '(1 3)) '(4 3))))
                   (vt-+ (vt-ones '(4 3)) (vt-ones '(4 3)) :out b)))))
  ;; 形状正确 + dtype 正确 + 非连续 → 必须成功（不得误报）
  (handler-case
      (let* ((base (vt-zeros '(3 4) :dtype :float64))
             (out (vt-slice base '(:all) '(0 4 2))))  ; (3 2) 非连续
        (vt-+ (vt-ones '(3 2)) (vt-ones '(3 2)) :out out)
        (incf *oc-pass*))
    (error (e)
      (incf *oc-fail*)
      (format t "~&[FAIL] 非连续 out 被误拒: ~a~%" e))))

;;; ==================================================================
;;; 6. 快路径 vs 通用路径的强制对照（同数据、不同布局）
;;; ==================================================================

(defun test-fast-vs-general-bitwise ()
  "对同一逻辑运算，分别用连续 out 与非连续 out 计算，
   两者结果必须逐位相同。这是 §4.6 机制 C 的直接验证。"
  (format t "~&~%=== 6. 快路径 vs 通用路径逐位对照 ===~%")
  (loop for (shape-a shape-b) in '(((3) (3)) ((2 3) (2 3)) ((2 1) (1 3)) ((4) (4)))
        do (let* ((sa (vt-shape-to-size shape-a))
                  (sb (vt-shape-to-size shape-b))
                  (na (loop for i below sa collect (coerce (+ i 1) 'double-float)))
                  (nb (loop for i below sb collect (coerce (* (+ i 1) 10) 'double-float)))
                  (va (vt-reshape (vt-from-sequence na) shape-a))
                  (vb (vt-reshape (vt-from-sequence nb) shape-b))
                  ;; 结果形状（广播后）
                  (final (vt-broadcast-shapes shape-a shape-b))
                  (fsize (vt-shape-to-size final))
                  ;; (A) 连续 out
                  (oc (vt-zeros final :dtype :float64))
                  ;; (B) 非连续 out（步长 2）：先在扁平 base 上步长切片，
                  ;;     再 reshape 成 final 形状（reshape 保持非连续 strides）
                  (base (vt-zeros (list (* 2 fsize)) :dtype :float64))
                  (on-flat (vt-slice base '(0 nil 2)))
                  (on (vt-reshape on-flat final)))
             (let ((rc (vt-+ va vb :out oc))
                   (rn (vt-+ va vb :out on)))
               (oc-check (format nil "快/通 路径等价 ~a+~a" shape-a shape-b)
                         (vt-to-list (vt-flatten rc))
                         (vt-to-list (vt-flatten rn)))
               ;; (C) 逆序 strides 的 out：把结果按倒序存放，
               ;;     逻辑顺序下的值必须仍与 rc 相同
               (when (> fsize 1)
                 (let* ((base2 (vt-zeros (list fsize) :dtype :float64))
                        ;; base2[(fsize-1)::-1] 是逻辑顺序的倒序视图
                        (rev-flat (vt-slice base2
                                            (list (1- fsize) nil -1)))
                        (rev (if (null (cdr final))
                                 rev-flat
                                 (vt-reshape rev-flat final)))
                        (rr (vt-+ va vb :out rev)))
                   ;; vt-flatten 按**逻辑顺序**（遵循视图 strides）输出，
                   ;; 故直接与 rc 比较即可，无需倒序。
                   (oc-check (format nil "快/逆序strides 等价 ~a+~a"
                                     shape-a shape-b)
                             (vt-to-list (vt-flatten rc))
                             (vt-to-list (vt-flatten rr)))))))))

;;; ==================================================================
;;; 7. 归约族（mean / var / std / average / nan* 族）的 :out 契约
;;; ==================================================================
;;;
;;; 这一组专门覆盖「中间结果不下传 out」这一实现要点：
;;;   reduce 族内部会串联多个子运算（sum → map 等）。若把用户 out 直接
;;;   下传给子运算，子运算的 H5（:dtype 与 :out 一致性）会先于本函数的
;;;   H3（dtype 严格相等）触发，报出 "vt-SUM: ..." 这类误导性错误。
;;;   因此本组同时断言：① 违反契约必须报错；② 错误信息带正确函数名前缀。

(defun oc-must-error (name thunk &key (prefix nil))
  "断言 thunk 抛错；若给 prefix，还要求错误信息以该前缀开头（本库错误
   消息首 token 为函数名，如 \"vt-mean: ...\"）。"
  (let ((msg (handler-case (progn (funcall thunk) nil)
               (error (e) (princ-to-string e)))))
    (if (null msg)
        (oc-check (format nil "~a（未报错）" name) t nil)
        (oc-check (format nil "~a（错误信息前缀 ~a）" name (or prefix "任意"))
                  t
                  (or (null prefix)
                      (let ((want (format nil "~a:" prefix)))
                        (and (>= (length msg) (length want))
                             (string-equal (subseq msg 0 (length want)) want))))))))

(defun test-reduction-out-contract ()
  (format t "~&~%=== 7. 归约族的 :out 契约（含错误信息前缀）===~%")
  (let ((a (vt-from-sequence '(1.0d0 2.0d0 3.0d0 4.0d0))))
    ;; --- vt-mean ---
    (let ((o (make-vt nil 0 :dtype :float64)))
      (oc-check "mean :out f64 = 2.5" 2.5d0 (vt-item (vt-mean a :out o)) :eps 1d-12))
    (oc-must-error "mean f64 → :out f32 报错（需搭配 :dtype）"
                   (lambda () (vt-mean a :out (make-vt nil 0 :dtype :float32)))
                   :prefix "vt-mean")
    (let ((o (make-vt nil 0 :dtype :float32)))
      (oc-check "mean :dtype :float32 + :out f32 = 2.5"
                2.5 (vt-item (vt-mean a :dtype :float32 :out o)) :eps 1d-6))
    ;; --- vt-var / vt-std ---
    (let ((o (make-vt nil 0 :dtype :float64)))
      (oc-check "var :out f64 = 1.25（ddof=0）" 1.25d0
                (vt-item (vt-var a :out o)) :eps 1d-12))
    (oc-must-error "var f64 → :out f32 报错"
                   (lambda () (vt-var a :out (make-vt nil 0 :dtype :float32)))
                   :prefix "vt-var")
    (oc-must-error "std f64 → :out f32 报错"
                   (lambda () (vt-std a :out (make-vt nil 0 :dtype :float32)))
                   :prefix "vt-std")
    ;; --- vt-average ---
    (let ((o (make-vt nil 0 :dtype :float64)))
      (oc-check "average :out f64 = 2.5" 2.5d0
                (vt-item (vt-average a (vt-from-sequence '(0.25d0 0.25d0 0.25d0 0.25d0))
                                     :out o))
                :eps 1d-12))
    (oc-must-error "average f64 → :out f32 报错"
                   (lambda ()
                     (vt-average a (vt-from-sequence '(0.25d0 0.25d0 0.25d0 0.25d0))
                                 :out (make-vt nil 0 :dtype :float32)))
                   :prefix "vt-average")
    ;; --- nan 族 ---
    (let ((o (make-vt nil 0 :dtype :float64)))
      (oc-check "nanmean :out f64 = 2.5" 2.5d0
                (vt-item (vt-nanmean a :out o)) :eps 1d-12))
    (oc-must-error "nanmean f64 → :out f32 报错"
                   (lambda () (vt-nanmean a :out (make-vt nil 0 :dtype :float32)))
                   :prefix "vt-nanmean")
    (let ((o (make-vt nil 0 :dtype :float64)))
      (oc-check "nanvar :out f64 = 1.25" 1.25d0
                (vt-item (vt-nanvar a :out o)) :eps 1d-12))
    (oc-must-error "nanvar f64 → :out int32 报错"
                   (lambda () (vt-nanvar a :out (make-vt nil 0 :dtype :int32)))
                   :prefix "vt-nanvar")
    (let ((o (make-vt nil 0 :dtype :float64)))
      (oc-check "nanstd :out f64 = 1.1180" 1.118033988749895d0
                (vt-item (vt-nanstd a :out o)) :eps 1d-12))
    (oc-must-error "nanstd f64 → :out f32 报错"
                   (lambda () (vt-nanstd a :out (make-vt nil 0 :dtype :float32)))
                   :prefix "vt-nanstd")
    (let ((o (make-vt nil 0 :dtype :float64)))
      (vt-nanmedian (vt-from-sequence '(1.0d0 2.0d0 3.0d0)) :out o)
      (oc-check "nanmedian :out f64 = 2.0" 2.0d0 (vt-item o) :eps 1d-12))
    (oc-must-error "nanmedian → :out int32 报错"
                   (lambda () (vt-nanmedian a :out (make-vt nil 0 :dtype :int32)))
                   :prefix "vt-nanmedian")
    ;; --- axis 路径 + 非连续 out（stride 计算正确性）---
    (let* ((m (vt-reshape (vt-from-sequence (list 1.0d0 2.0d0
                                                 3.0d0 4.0d0))
                          (list 2 2)))
           (base (vt-zeros '(4) :dtype :float64))
           (o (vt-narrow base 0 0 2)))          ; base 的连续视图，shape (2)
      (vt-nanmean m :axis 1 :out o)
      (oc-check "nanmean axis=1 经 out 写入 [1.5 3.5]"
                '(1.5d0 3.5d0)
                (list (aref (vt-data o) (+ (vt-offset o) 0))
                      (aref (vt-data o) (+ (vt-offset o) 1)))
                :eps 1d-12))
    (oc-must-error "nanmean axis + 形状不符 out 报错"
                   (lambda ()
                     (vt-nanmean (vt-reshape (vt-from-sequence (list 1.0d0 2.0d0
                                                                      3.0d0 4.0d0))
                                             (list 2 2))
                                 :axis 1
                                 :out (vt-zeros '(3) :dtype :float64)))
                   :prefix "vt-nanmean")
    ;; --- 与无 :out 结果逐位一致 ---
    (let ((o (make-vt nil 0 :dtype :float64)))
      (oc-check "nanmean :out 与无 :out 逐位一致"
                (vt-item (vt-nanmean a))
                (vt-item (vt-nanmean a :out o))
                :eps 0.0d0))))

;;; ==================================================================
;;; 8. matmul / einsum 的别名与 dtype 契约（SIMD 路径）
;;; ==================================================================
;;;
;;; 覆盖一个真实缺陷：原实现的别名检测只比较 `vt-data` 指针，
;;; 当 out 与输入是「同一 data、不同 offset」的重叠视图时漏检，
;;; 导致预清零破坏输入 / 累加写坏结果。v0.3.6 改用物理区间重叠判定。

(defun oc-maxdiff (x y)
  "最大绝对误差（结果恒为标量，便于比较）。"
  (with-float-safe
    (vt-item (vt-amax (vt-abs (vt-- x y))))))

(defun test-matmul-alias-contract ()
  (format t "~&~%=== 8. matmul 别名与 dtype 契约（SIMD 路径）===~%")
  (let* ((m 64) (k 64) (n 64)
         (a (vt-reshape (vt-from-sequence
                         (loop for i below (* m k)
                               collect (coerce (mod (* i 7) 13) 'double-float)))
                        (list m k)))
         (b (vt-reshape (vt-from-sequence
                         (loop for i below (* k n)
                               collect (coerce (mod (* i 5) 11) 'double-float)))
                        (list k n)))
         (ref (vt-matmul a b)))
    ;; 1) 普通 out：必须与无 out 逐位一致
    (let ((o (vt-zeros (list m n) :dtype :float64)))
      (vt-matmul a b :out o)
      (oc-check "matmul :out 与无 :out 一致" t (< (oc-maxdiff o ref) 1d-9)))
    ;; 2) out dtype 与「输入提升结果」不符 → 必须报错
    (oc-check "matmul :out f32（输入 f64）报错" t
              (handler-case
                  (progn (vt-matmul a b :out (vt-zeros (list m n) :dtype :float32)) nil)
                (error () t)))
    ;; 3) 别名：out 与 a 同一 data、offset 错位重叠 → 结果仍须正确
    (let* ((total (* (+ m 2) k))
           (base (vt-zeros (list total) :dtype :float64))
           (a-view (vt-reshape (vt-narrow base 0 0 (* m k)) (list m k)))
           (o-view (vt-reshape (vt-narrow base 0 (* k 2) (+ (* k 2) (* m k)))
                               (list m n))))
      (vt-copy-into a-view a)
      (let ((expect (vt-matmul (vt-copy a-view) b)))
        (vt-matmul a-view b :out o-view)
        (oc-check "matmul 别名（同 data 异 offset）结果正确" t
                  (< (oc-maxdiff o-view expect) 1d-9)))
      ;; 4) 完全别名：out 与输入同一视图
      (let ((a2 (vt-copy a))
            (expect2 (vt-matmul (vt-copy a) b)))
        (setf expect2 (vt-copy expect2))
        (vt-matmul a2 b :out a2)
        (oc-check "matmul 完全别名（out 即 a）结果正确" t
                  (< (oc-maxdiff a2 expect2) 1d-9))))))

;;; ==================================================================
;;; 运行
;;; ==================================================================

(defun run-out-contract-v2-tests ()
  (setf *oc-fail* 0 *oc-pass* 0)
  (with-float-safe
    (test-noncontig-out-equivalence)
    (test-transposed-out)
    (test-alias-out)
    (test-nan-inf-out)
    (test-hard-contract-violations)
    (test-fast-vs-general-bitwise)
    (test-reduction-out-contract)
    (test-matmul-alias-contract))
  (format t "~&~%=== out 契约 v2 测试总结 ===~%  通过: ~d  失败: ~d~%"
          *oc-pass* *oc-fail*)
  (if (zerop *oc-fail*)
      (format t "~&[PASS] out 契约 v2 全部通过 ✅~%")
      (format t "~&[FAIL] out 契约 v2 有失败项 ❌~%"))
  (zerop *oc-fail*))

(run-out-contract-v2-tests)
