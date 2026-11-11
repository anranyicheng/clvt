;;;; io-exec-test.lisp — vt-save / vt-load 执行层的回归
;;;;
;;;; 病根：npy 数据区一直是「逐元素编码」——
;;;;   · 写：`vt-do-each` 每个元素一次泛型 AREF + 装箱，再走
;;;;     `%NPY-ENCODE-BITS`（用 `integer-decode-float` 重建位模式）和
;;;;     `%SET-UINT-LE!`（逐字节拆分）；
;;;;   · 读：每个元素一次 `%UINT-LE-AT`（逐字节组装）+ `%BITS->F64`
;;;;     （用 `scale-float` 重建浮点），产物先落进一个 T 元素的普通向量，
;;;;     再由 `%MAKE-VT-FROM-RAW` 逐元素复制进 specialized array。
;;;; 而 npy 数据区恒为「小端 + 紧凑 + 无 padding」，SBCL 在 x86-64 上的
;;;; specialized array 布局与它逐字节相同 —— 整个数据区本来就已经是
;;;; 编码好的字节，逐元素做一遍只是把同样的位模式拆开再拼回去。
;;;; 1M 元素（8 MB）实测：写 143 → 9 ms，读 200 → 11 ms。
;;;;
;;;; lisp 文本格式另有两笔固定成本：
;;;;   · 写：先建 1M 元素的中间向量，`prin1` 整棵树时再读一遍；且
;;;;     `%ENCODE-LISP-ELEMENT` 在**每个元素**上套一次 `WITH-FLOAT-SAFE`
;;;;     （216 ns/次）；
;;;;   · 读：`(coerce data 'list)` 为 1M 元素 cons 一遍，随后用 `LOOP FOR … IN`
;;;;     走链，每元素再 `(funcall (vt-cast-fun dtype) v)`。
;;;;
;;;; A 段：整块拷贝路径与逐元素路径**逐字节**相同（11 dtype × 形状矩阵）；
;;;; B 段：解码路径同样逐位相同（含 NaN payload / -0.0 / 复数分量 / bool）；
;;;; C 段：`vt-save :as :lisp` 的文本与旧实现逐字符相同（格式未变）；
;;;; D 段：往返语义（0 维 / 空 / 非连续视图 / 自动检测 / 可读性）；
;;;; E 段：机器无关的倍率护栏（整块 ÷ 逐元素，只看比值不看绝对毫秒）。

(require :asdf)
#+quicklisp (ql:quickload :clvt :silent t)
(asdf:load-system :clvt)

(defpackage :clvt-io-exec-test
  (:use :cl :clvt)
  (:import-from :clvt
                #:vt-dtype-itemsize #:vt-complex-dtype-p #:vt-do-each
                #:with-float-safe #:%set-uint-le! #:%uint-le-at
                #:%npy-encode-bits #:%npy-decode-bits #:%f64-bits #:%f32-bits
                #:%bits->f64 #:%bits->f32 #:%npy-encode-data-buffer
                #:%npy-decode-direct
                #:*native-little-endian-p* #:*npy-fast-copy* #:%save-lisp-format
                #:%encode-lisp-element)
  (:export #:run))

(in-package :clvt-io-exec-test)

;;; ==================================================================
;;; 0. 断言与工具
;;; ==================================================================
(declaim (special *n* *pass* *fail* *failures*))
(defparameter *n* 0)
(defparameter *pass* 0)
(defparameter *fail* 0)
(defparameter *failures* nil)

(defun check (label ok &optional detail)
  (incf *n*)
  (if ok
      (progn (incf *pass*) (format t "  [ OK ] ~a~%" label))
      (progn (incf *fail*)
             (push (format nil "~a~@[ — ~a~]" label detail) *failures*)
             (format t "  [FAIL] ~a~@[ — ~a~]~%" label detail))))

(defun check-at-least (label ratio limit what)
  (check (format nil "~a（~a = ~,2f× ≥ ~,2f×）" label what ratio limit)
         (>= ratio limit)
         (format nil "实际 ~,3f×，下限 ~,2f×" ratio limit)))

(defun check-at-most (label ratio limit what)
  (check (format nil "~a（~a = ~,2f× ≤ ~,2f×）" label what ratio limit)
         (<= ratio limit)
         (format nil "实际 ~,3f×，上限 ~,2f×" ratio limit)))

(defun best-ms (fn &key (iters 3))
  "取最快一批的均值（GET-INTERNAL-REAL-TIME 刻度 1 ms，单次量不准）。"
  (funcall fn)
  (let ((best most-positive-fixnum))
    (dotimes (k 3)
      (let ((s (get-internal-real-time)))
        (dotimes (i iters) (funcall fn))
        (setf best (min best (- (get-internal-real-time) s)))))
    (/ (* 1000.0 best) iters internal-time-units-per-second)))

(defparameter *tmp-dir*
  (merge-pathnames (make-pathname :directory '(:relative "clvt-io-exec-test"))
                   (pathname (or (ignore-errors (sb-unix::posix-getenv "TMPDIR"))
                                 "/tmp/"))))
(ensure-directories-exist *tmp-dir*)

(defun tmp-path (name)
  (merge-pathnames name *tmp-dir*))

;;; ==================================================================
;;; 1. 逐字复刻的**改动前**实现（对拍基准）
;;; ==================================================================

(defun legacy-encode-buffer (vt)
  "改动前 `%NPY-ENCODE-DATA-BUFFER` 的逐元素版本（逐字复刻）。"
  (let* ((dtype (vt-dtype vt))
         (itemsize (vt-dtype-itemsize dtype))
         (n (vt-size vt))
         (buf (make-array (* n itemsize)
                          :element-type '(unsigned-byte 8)
                          :initial-element 0)))
    (with-float-safe
      (let ((k 0))
        (declare (fixnum k))
        (vt-do-each (ptr val vt)
          (declare (ignore ptr))
          (if (vt-complex-dtype-p dtype)
              (let* ((half (floor itemsize 2))
                     (base (* k itemsize))
                     (bits-fn (if (= half 8) #'%f64-bits #'%f32-bits)))
                (%set-uint-le! buf base (funcall bits-fn (realpart val)) half)
                (%set-uint-le! buf (+ base half) (funcall bits-fn (imagpart val)) half))
              (%set-uint-le! buf (* k itemsize) (%npy-encode-bits dtype val) itemsize))
          (incf k))))
    buf))

(defun legacy-decode-data (oct dtype count)
  "改动前 `%NPY-DECODE-DATA` 的逐元素版本（逐字复刻），返回 T 元素向量。"
  (let ((raw (make-array count))
        (itemsize (vt-dtype-itemsize dtype)))
    (with-float-safe
      (if (vt-complex-dtype-p dtype)
          (let ((half (floor itemsize 2))
                (bits-fn (if (= (floor itemsize 2) 8) #'%bits->f64 #'%bits->f32)))
            (dotimes (k count)
              (let ((base (* k itemsize)))
                (setf (aref raw k)
                      (complex (funcall bits-fn (%uint-le-at oct base half))
                               (funcall bits-fn (%uint-le-at oct (+ base half) half)))))))
          (dotimes (k count)
            (setf (aref raw k)
                  (%npy-decode-bits dtype (%uint-le-at oct (* k itemsize) itemsize))))))
    raw))

;;; --- 逐位比较（NaN 也要「相等」，不能用 = 或 equalp）---

(defun elem-bits (dtype x)
  "元素 → 位模式（整数或整数列表），用于逐位比较。"
  (cond
    ((eq dtype :float64) (%f64-bits x))
    ((eq dtype :float32) (%f32-bits x))
    ((vt-complex-dtype-p dtype)
     (let ((rdt (if (eq dtype :complex128) :float64 :float32)))
       (list (elem-bits rdt (realpart x)) (elem-bits rdt (imagpart x)))))
    ((eq dtype :bool) (if (zerop x) 0 1))
    (t x)))

(defun same-bits-p (dtype a b)
  (equal (elem-bits dtype a) (elem-bits dtype b)))

(defun same-data-p (dtype a b)
  "两个同 dtype 向量是否逐位相同（长度也必须相同）。"
  (and (= (length a) (length b))
       (loop for x across a
             for y across b
             always (same-bits-p dtype x y))))

;;; ==================================================================
;;; 2. 用例素材
;;; ==================================================================

(defun special-f64 ()
  "NaN / +Inf / -Inf / -0.0 / 0.0 / 1.5 六个特殊位模式。"
  (vt-from-sequence (list (vt-get-nan :float64) (vt-get-pos-inf :float64)
                          (vt-get-neg-inf :float64) -0.0d0 0.0d0 1.5d0)
                    :dtype :float64))

(defparameter *all-dtypes*
  '(:float64 :float32 :int64 :int32 :int16 :int8 :uint16 :uint8 :bool
    :complex128 :complex64))

(defun tensor-of (dtype n &optional (start 0))
  "第 i 个元素 = START+I，按 DTYPE 强制。复数取 (i, -i)，便于区分两个分量。"
  (let ((v (vt-astype (vt-arange (+ start n)) :int64)))
    (case dtype
      (:bool (vt-astype (vt-mod v 2) :bool))
      ((:complex128 :complex64)
       (vt-astype (vt-astype v :float64) dtype))
      (t (vt-astype v dtype)))))

(defun shape-cases ()
  "（形状, 构造 thunk）——1 维 / 2 维 / 3 维 / 0 维 / 空。"
  (list (cons '(7) (lambda (dt) (tensor-of dt 7)))
        (cons '(3 4) (lambda (dt) (vt-reshape (tensor-of dt 12) '(3 4))))
        (cons '(2 3 2) (lambda (dt) (vt-reshape (tensor-of dt 12) '(2 3 2))))
        (cons nil (lambda (dt) (vt-reshape (tensor-of dt 1) nil)))
        (cons '(2 0 3) (lambda (dt) (vt-reshape (tensor-of dt 0) '(2 0 3))))))

;;; ==================================================================
;;; 3. A 段：编码路径逐字节相同
;;; ==================================================================
(defun run-encode-equivalence ()
  (format t "~%A. npy 编码：整块拷贝 vs 逐元素（逐字节）~%")
  (check "本机小端（整块路径的前提）" *native-little-endian-p*)
  (dolist (dt *all-dtypes*)
    (dolist (case (shape-cases))
      (let* ((label (format nil "~a ~a" dt (or (car case) :0d)))
             (vt (funcall (cdr case) dt))
             (fast (%npy-encode-data-buffer vt))
             (slow (legacy-encode-buffer vt)))
        (check (format nil "~a 字节等价" label) (equalp fast slow)))))
  ;; 非连续视图：转置 / 翻转 —— 必须按逻辑顺序编码，且与逐元素路径同字节
  (let* ((m (vt-reshape (tensor-of :float64 12) '(3 4)))
         (views (list (cons "转置" (vt-transpose m))
                      (cons "翻转" (vt-flip m :axis 1))
                      (cons "转置+翻转" (vt-flip (vt-transpose m) :axis 0)))))
    (dolist (v views)
      (let ((vt (cdr v)))
        (check (format nil "视图（~a）非连续判定" (car v))
               (not (vt-contiguous-p vt)))
        (check (format nil "视图（~a）字节等价" (car v))
               (equalp (%npy-encode-data-buffer vt) (legacy-encode-buffer vt)))))))

;;; ==================================================================
;;; 4. B 段：解码路径逐位相同
;;; ==================================================================
(defun run-decode-equivalence ()
  (format t "~%B. npy 解码：整块重解释 vs 逐元素（逐位）~%")
  (dolist (dt *all-dtypes*)
    (dolist (case (shape-cases))
      (let* ((label (format nil "~a ~a" dt (or (car case) :0d)))
             (vt (funcall (cdr case) dt))
             (oct (%npy-encode-data-buffer vt))
             (count (vt-size vt))
             (direct (%npy-decode-direct oct dt count))
             (slow (legacy-decode-data oct dt count)))
        (if (zerop count)
            (check (format nil "~a 空张量走通用路径" label) (null direct))
            (check (format nil "~a 逐位等价" label)
                   (and direct (same-data-p dt direct slow)))))))
  ;; 特殊位模式：NaN payload / ±Inf / -0.0
  (let* ((f (special-f64))
         (oct (%npy-encode-data-buffer f))
         (direct (%npy-decode-direct oct :float64 6))
         (path (tmp-path "bits.npy"))
         (back (progn (vt-save f path :as :npy) (vt-load path :as :npy))))
    (check "NaN/±Inf/-0.0 整块解码与逐元素逐位相同"
           (same-data-p :float64 direct (legacy-decode-data oct :float64 6)))
    (check "-0.0 符号位经文件往返保留"
           (logbitp 63 (%f64-bits (aref (vt-data back) 3))))
    (check "NaN 经文件往返仍是 NaN"
           (sb-ext:float-nan-p (aref (vt-data back) 0))))
  ;; bool 的非 0/1 字节必须真值化（numpy 只写 0/1，别的工具可能不）
  (let* ((oct (make-array 4 :element-type '(unsigned-byte 8) :initial-element 0)))
    (setf (aref oct 0) 0 (aref oct 1) 1 (aref oct 2) 2 (aref oct 3) 255)
    (check "|b1 的非 0/1 字节被真值化成 0/1"
           (equalp (%npy-decode-direct oct :bool 4) #(0 1 1 1))))
  ;; 选路开关：关掉之后必须真的回退（成果与性能护栏都依赖这一点）
  (let* ((f (tensor-of :float64 8))
         (oct (%npy-encode-data-buffer f)))
    (check "开关关闭时整块解码让路"
           (null (let ((*npy-fast-copy* nil)) (%npy-decode-direct oct :float64 8))))
    (check "开关关闭时整块编码让路"
           (equalp (let ((*npy-fast-copy* nil)) (%npy-encode-data-buffer f))
                   (legacy-encode-buffer f)))))

;;; ==================================================================
;;; 5. C 段：lisp 文本格式逐字符未变
;;; ==================================================================
(defun legacy-lisp-data-vector (vt)
  "改动前 `%LISP-DATA-VECTOR`：每元素一次 `WITH-FLOAT-SAFE`（216 ns/次）。"
  (let ((dtype (vt-dtype vt))
        (vec (make-array (vt-size vt) :initial-element 0)))
    (with-float-safe
      (let ((k 0))
        (declare (fixnum k))
        (vt-do-each (ptr val vt)
          (declare (ignore ptr))
          (setf (aref vec k) (%encode-lisp-element dtype val))
          (incf k))))
    vec))

(defun legacy-lisp-text (vt)
  "改动前 `%SAVE-LISP-FORMAT` 的正文（`prin1` 整棵树）。"
  (let ((s (make-string-output-stream)))
    (let ((*print-readably* t)
          (*print-pretty* nil)
          (*print-length* nil)
          (*print-level* nil))
      (prin1 (list :clvt-tensor
                   :version 1
                   :shape (vt-shape vt)
                   :dtype (vt-dtype vt)
                   :data (legacy-lisp-data-vector vt))
             s)
      (terpri s))
    (get-output-stream-string s)))

(defun new-lisp-text (vt)
  "当前 `%SAVE-LISP-FORMAT` 写出的文本（走真实文件，取其全部字符）。"
  (let ((path (tmp-path "text.sexp")))
    (%save-lisp-format vt path)
    (with-open-file (s path :direction :input)
      (let* ((n (file-length s))
             (buf (make-string n)))
        (read-sequence buf s)
        buf))))

(defun run-lisp-format-stability ()
  (format t "~%C. lisp 文本格式：流式写 vs 旧 prin1（逐字符）~%")
  (dolist (dt '(:float64 :float32 :int64 :int8 :bool :complex128 :complex64))
    (dolist (case (list (cons '(6) (lambda (d) (tensor-of d 6)))
                        (cons '(2 3) (lambda (d) (vt-reshape (tensor-of d 6) '(2 3))))
                        (cons nil (lambda (d) (vt-reshape (tensor-of d 1) nil)))))
      (let ((vt (funcall (cdr case) dt)))
        (check (format nil "~a ~a 文本逐字符相同" dt (or (car case) :0d))
               (string= (new-lisp-text vt) (legacy-lisp-text vt))))))
  ;; 非有限值也必须走同一条编码（:NAN/:INF/:NEG-INF 与 (:CX …) 形式）
  (let ((vt (vt-view (special-f64) '(6))))
    (check "NaN/±Inf 文本逐字符相同"
           (string= (new-lisp-text vt) (legacy-lisp-text vt))))
  (let ((vt (vt-from-sequence (list #C(1.0d0 2.0d0) #C(0.0d0 0.0d0)) :dtype :complex128)))
    (check "复数文本逐字符相同"
           (string= (new-lisp-text vt) (legacy-lisp-text vt)))))

;;; ==================================================================
;;; 6. D 段：往返语义
;;; ==================================================================
(defun run-roundtrip ()
  (format t "~%D. 往返语义~%")
  (dolist (dt *all-dtypes*)
    (dolist (case (shape-cases))
      (let* ((vt (funcall (cdr case) dt))
             (label (format nil "~a ~a" dt (or (car case) :0d))))
        (dolist (fmt '(:npy :lisp))
          (let* ((path (tmp-path "rt.tmp"))
                 (back (progn (vt-save vt path :as fmt) (vt-load path :as fmt))))
            (check (format nil "~a ~a 往返形状" label fmt)
                   (equal (vt-shape back) (vt-shape vt)))
            (check (format nil "~a ~a 往返 dtype" label fmt)
                   (eq (vt-dtype back) (vt-dtype vt)))
            (check (format nil "~a ~a 往返逐位" label fmt)
                   (same-data-p dt (vt-data back) (vt-data vt))))))))
  ;; 非连续视图：内容按逻辑顺序物化
  (let* ((m (vt-reshape (tensor-of :float64 12) '(3 4)))
         (tp (vt-transpose m))
         (path (tmp-path "view.npy"))
         (back (progn (vt-save tp path :as :npy) (vt-load path :as :npy))))
    (check "转置视图 npy 往返形状" (equal (vt-shape back) '(4 3)))
    ;; 与「逻辑内容物化后的连续副本」比较：视图自身的 VT-DATA 是底层原数组
    (check "转置视图 npy 往返逐位"
           (same-data-p :float64 (vt-data back) (vt-data (vt-copy tp)))))
  ;; 自动检测
  (let* ((f (tensor-of :float64 5))
         (np (tmp-path "auto.npy"))
         (lp (tmp-path "auto.sexp")))
    (vt-save f np :as :npy)
    (vt-save f lp :as :lisp)
    (check "as=nil 自动识别 npy"
           (same-data-p :float64 (vt-data (vt-load np)) (vt-data f)))
    (check "as=nil 自动识别 lisp"
           (same-data-p :float64 (vt-data (vt-load lp)) (vt-data f))))
  ;; 新写出的 lisp 文本必须能被旧 reader（read）解析成同一结构
  (let* ((f (vt-from-sequence (list 1.0d0 (vt-get-nan :float64) 3.0d0) :dtype :float64))
         (text (new-lisp-text f))
         (form (let ((*read-eval* nil)) (read-from-string text))))
    (check "流式输出是单个可读 s-expression"
           (and (consp form) (eq (car form) :clvt-tensor)))
    (check "流式输出的 :data 长度正确"
           (= 3 (length (getf (cdr form) :data))))))

;;; ==================================================================
;;; 7. E 段：倍率护栏（只看比值）
;;; ==================================================================
(defun run-perf-guards ()
  (format t "~%E. 倍率护栏~%")
  (when *native-little-endian-p*
    (let* ((n 1000000)
           (f (vt-astype (vt-arange n) :float64))
           (oct (%npy-encode-data-buffer f))
           (fast-enc (lambda () (let ((*npy-fast-copy* t)) (%npy-encode-data-buffer f))))
           (slow-enc (lambda () (let ((*npy-fast-copy* nil)) (%npy-encode-data-buffer f))))
           (fast-dec (lambda () (let ((*npy-fast-copy* t)) (%npy-decode-direct oct :float64 n))))
           (slow-dec (lambda () (legacy-decode-data oct :float64 n)))
           (copy (lambda () (vt-copy f))))
      ;; 整块拷贝的上界就是「一次全量拷贝」：逐元素至少比它贵 8 倍，
      ;; 否则说明逐元素编码并不贵、这轮改动不值。
      (check-at-least "npy 编码 逐元素 ÷ 整块"
                      (/ (best-ms slow-enc :iters 1) (best-ms fast-enc))
                      8.0 "slow/fast")
      (check-at-least "npy 解码 逐元素 ÷ 整块"
                      (/ (best-ms slow-dec) (best-ms fast-dec))
                      4.0 "slow/fast")
      ;; 整块解码相对「纯拷贝」不能贵太多 —— 它做的是同一件事
      (check-at-most "整块解码 ÷ 纯拷贝"
                     (/ (best-ms fast-dec) (max (best-ms copy) 0.001))
                     3.0 "fast/copy")
      ;; lisp 流式写 vs 旧整棵树 prin1（两边都落盘，只差中间向量与重复屏蔽）
      (let ((new (lambda () (%save-lisp-format f (tmp-path "perf-new.sexp"))))
            (old (lambda () (let ((*print-readably* t) (*print-pretty* nil)
                                  (*print-length* nil) (*print-level* nil))
                              (with-open-file (s (tmp-path "perf-old.sexp")
                                                 :direction :output :if-exists :supersede)
                                (prin1 (list :clvt-tensor :version 1 :shape (vt-shape f)
                                             :dtype (vt-dtype f)
                                             :data (legacy-lisp-data-vector f))
                                       s)
                                (terpri s))))))
        (check-at-least "lisp 写 旧 ÷ 新"
                        (/ (best-ms old :iters 1) (best-ms new :iters 1))
                        1.2 "old/new")))))

;;; ==================================================================
;;; 8. 入口
;;; ==================================================================
(defun run ()
  (setf *n* 0 *pass* 0 *fail* 0 *failures* nil)
  (run-encode-equivalence)
  (run-decode-equivalence)
  (run-lisp-format-stability)
  (run-roundtrip)
  (run-perf-guards)
  (format t "~%==================================================~%")
  (format t "io-exec-test: ~a / ~a 通过~%" *pass* *n*)
  (when *failures*
    (format t "~%失败项：~%")
    (dolist (f (reverse *failures*)) (format t "  - ~a~%" f)))
  (format t "==================================================~%")
  (when (plusp *fail*)
    (sb-ext:exit :code 1))
  (values *pass* *n*))

(run)
