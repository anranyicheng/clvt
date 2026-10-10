;;;; io-interop-test.lisp — vt-load / vt-save 持久化与 numpy .npy 互操作测试套件
;;;;
;;;; 覆盖矩阵：
;;;;   1. :lisp / :npy 两种格式 × 8 种 dtype 的保存-加载往返（L1 内容一致）
;;;;   2. 形状矩阵：0-d / 1-d / 空张量 (2 0 3) / 2-d / 3-d / 非连续视图（转置）
;;;;   3. 非有限值：NaN / ±Inf / -0.0 往返
;;;;   4. 跨语言互操作：加载 numpy 2.5.3 真实生成的 .npy 字节固件
;;;;      （C-order / F-order / 0-d / 空张量），且我方 .npy 输出与
;;;;      np.save 逐字节一致（字节级锁定，无 python 依赖）
;;;;   5. python3+numpy 可用时：由 numpy 读回我方写出的 .npy（不可用则 Skip）
;;;;   6. as=nil 自动检测
;;;;   7. L2 错误契约：非法参数 / 坏 magic / 不支持 dtype / 数据截断 / 坏 lisp 格式
;;;;
;;;; 运行：bash test/run-tests.sh --suite io-interop-test
;;;; 退出码 0 = 全部通过。

(asdf:load-system :clvt)
(in-package :clvt)

(defparameter *pass* 0)
(defparameter *fail* 0)
(defparameter *skip* 0)

(defparameter *tmp-dir*
  (merge-pathnames (make-pathname :directory '(:relative "clvt-io-interop-test"))
                   (pathname (or (ignore-errors (sb-unix::posix-getenv "TMPDIR"))
                                 "/tmp/"))))
(ensure-directories-exist *tmp-dir*)

;;;; fixtures — numpy 2.5.3 实际生成的 .npy 字节（np.save 输出）
;;;; 生成方式：np.save(BytesIO(), arr) 的原始字节，用于跨语言互操作测试
(defparameter *npy-fixture-f64-c*
  ;; np.arange(6,dtype=float64).reshape(2,3) C-order → shape=[2, 3] dtype=float64 nbytes=176
  #(147 78 85 77 80 89 1 0 118 0 123 39 100 101 115 99 114 39 58 32 39 60 102 56 39 44 32 39 102 111 114 116 114 97 110 95 111 114 100 101 114 39 58 32 70 97 108 115 101 44 32 39 115 104 97 112 101 39 58 32 40 50 44 32 51 41 44 32 125 32 32 32 32 32 32 32 32 32 32 32 32 32 32 32 32 32 32 32 32 32 32 32 32 32 32 32 32 32 32 32 32 32 32 32 32 32 32 32 32 32 32 32 32 32 32 32 32 32 32 32 32 32 32 32 32 32 32 10 0 0 0 0 0 0 0 0 0 0 0 0 0 0 240 63 0 0 0 0 0 0 0 64 0 0 0 0 0 0 8 64 0 0 0 0 0 0 16 64 0 0 0 0 0 0 20 64))
(defparameter *npy-fixture-i32-c*
  ;; np.arange(12,dtype=int32).reshape(3,4)-42 C-order → shape=[3, 4] dtype=int32 nbytes=176
  #(147 78 85 77 80 89 1 0 118 0 123 39 100 101 115 99 114 39 58 32 39 60 105 52 39 44 32 39 102 111 114 116 114 97 110 95 111 114 100 101 114 39 58 32 70 97 108 115 101 44 32 39 115 104 97 112 101 39 58 32 40 51 44 32 52 41 44 32 125 32 32 32 32 32 32 32 32 32 32 32 32 32 32 32 32 32 32 32 32 32 32 32 32 32 32 32 32 32 32 32 32 32 32 32 32 32 32 32 32 32 32 32 32 32 32 32 32 32 32 32 32 32 32 32 32 32 32 10 214 255 255 255 215 255 255 255 216 255 255 255 217 255 255 255 218 255 255 255 219 255 255 255 220 255 255 255 221 255 255 255 222 255 255 255 223 255 255 255 224 255 255 255 225 255 255 255))
(defparameter *npy-fixture-u8-c*
  ;; np.array([[1,2],[3,4],[250,0]],uint8) C-order → shape=[3, 2] dtype=uint8 nbytes=134
  #(147 78 85 77 80 89 1 0 118 0 123 39 100 101 115 99 114 39 58 32 39 124 117 49 39 44 32 39 102 111 114 116 114 97 110 95 111 114 100 101 114 39 58 32 70 97 108 115 101 44 32 39 115 104 97 112 101 39 58 32 40 51 44 32 50 41 44 32 125 32 32 32 32 32 32 32 32 32 32 32 32 32 32 32 32 32 32 32 32 32 32 32 32 32 32 32 32 32 32 32 32 32 32 32 32 32 32 32 32 32 32 32 32 32 32 32 32 32 32 32 32 32 32 32 32 32 32 10 1 2 3 4 250 0))
(defparameter *npy-fixture-f64-f*
  ;; np.asfortranarray(arange(6).reshape(2,3)) F-order → shape=[2, 3] dtype=float64 nbytes=176
  #(147 78 85 77 80 89 1 0 118 0 123 39 100 101 115 99 114 39 58 32 39 60 102 56 39 44 32 39 102 111 114 116 114 97 110 95 111 114 100 101 114 39 58 32 84 114 117 101 44 32 39 115 104 97 112 101 39 58 32 40 50 44 32 51 41 44 32 125 32 32 32 32 32 32 32 32 32 32 32 32 32 32 32 32 32 32 32 32 32 32 32 32 32 32 32 32 32 32 32 32 32 32 32 32 32 32 32 32 32 32 32 32 32 32 32 32 32 32 32 32 32 32 32 32 32 32 32 10 0 0 0 0 0 0 0 0 0 0 0 0 0 0 8 64 0 0 0 0 0 0 240 63 0 0 0 0 0 0 16 64 0 0 0 0 0 0 0 64 0 0 0 0 0 0 20 64))
(defparameter *npy-fixture-f64-0d*
  ;; np.array(3.5) 0-d → shape=nil dtype=float64 nbytes=136
  #(147 78 85 77 80 89 1 0 118 0 123 39 100 101 115 99 114 39 58 32 39 60 102 56 39 44 32 39 102 111 114 116 114 97 110 95 111 114 100 101 114 39 58 32 70 97 108 115 101 44 32 39 115 104 97 112 101 39 58 32 40 41 44 32 125 32 32 32 32 32 32 32 32 32 32 32 32 32 32 32 32 32 32 32 32 32 32 32 32 32 32 32 32 32 32 32 32 32 32 32 32 32 32 32 32 32 32 32 32 32 32 32 32 32 32 32 32 32 32 32 32 32 32 32 32 32 32 10 0 0 0 0 0 0 12 64))
(defparameter *npy-fixture-i16-empty*
  ;; np.zeros((2,0,3),int16) 空 → shape=[2, 0, 3] dtype=int16 nbytes=128
  #(147 78 85 77 80 89 1 0 118 0 123 39 100 101 115 99 114 39 58 32 39 60 105 50 39 44 32 39 102 111 114 116 114 97 110 95 111 114 100 101 114 39 58 32 70 97 108 115 101 44 32 39 115 104 97 112 101 39 58 32 40 50 44 32 48 44 32 51 41 44 32 125 32 32 32 32 32 32 32 32 32 32 32 32 32 32 32 32 32 32 32 32 32 32 32 32 32 32 32 32 32 32 32 32 32 32 32 32 32 32 32 32 32 32 32 32 32 32 32 32 32 32 32 32 32 32 32 10))

;;; ---- 辅助 ----

(defun check (label cond)
  (if cond
      (progn (incf *pass*) (format t "  ok   ~a~%" label))
      (progn (incf *fail*) (format t "  FAIL ~a~%" label)))
  (finish-output))

(defun check-error (label thunk)
  (declare (optimize (speed 0)))
  (handler-case
      (progn (funcall thunk)
             (incf *fail*)
             (format t "  FAIL ~a（未报错）~%" label))
    (error ()
      (incf *pass*)
      (format t "  ok   ~a（确定性报错）~%" label)))
  (finish-output))

(defun check-skip (label)
  (incf *skip*)
  (format t "  SKIP ~a~%" label)
  (finish-output))

;; 行主序逻辑元素收集（支持任意 strides/offset 视图）
(defun %elements (vt)
  (let ((acc (make-array (vt-size vt)))
        (k 0))
    (vt-do-each (ptr v vt)
      (declare (ignore ptr))
      (setf (aref acc k) v)
      (incf k))
    acc))

;; NaN/Inf 感知的元素比较（非浮点直接 =）
(defun %elem-equal (a b)
  (if (and (floatp a) (floatp b))
      (with-float-safe (vt-float-nan-inf-= a b))
      (= a b)))

(defun %same-tensor-p (a b)
  (and (equal (vt-shape a) (vt-shape b))
       (eq (vt-dtype a) (vt-dtype b))
       (let ((ea (%elements a)) (eb (%elements b)))
         (and (= (length ea) (length eb))
              (every #'%elem-equal ea eb)))))

(defun %read-file-bytes (path)
  (with-open-file (s path :element-type '(unsigned-byte 8))
    (let ((buf (make-array (file-length s) :element-type '(unsigned-byte 8))))
      (read-sequence buf s)
      buf)))

(defun %write-bytes (path bytes)
  (with-open-file (s path :direction :output :if-exists :supersede
                          :element-type '(unsigned-byte 8))
    (write-sequence bytes s)))

;; -0.0 符号位检测：1/(-0.0) = -Inf
(defun %neg-zero-p (v)
  (and (floatp v) (zerop v)
       ;; 用 integer-decode-float 的符号位判断（无浮点除法，
       ;; 解释执行下同样可靠；-0.0 → (0 0 -1)，+0.0 → (0 0 1)）
       (minusp (nth-value 2 (integer-decode-float v)))))

;; 非有限值归类（有限值统一转 double-float 便于跨精度比较）
(defun %classify-float (v)
  (with-float-safe
    (cond ((not (floatp v)) v)
          ((vt-float-nan-p v) :nan)
          ((vt-float-pos-inf-p v) :inf)
          ((vt-float-neg-inf-p v) :neg-inf)
          (t (coerce v 'double-float)))))

(defun %path (name)
  (merge-pathnames name *tmp-dir*))

(defun %save+load (vt path fmt)
  "保存后按显式格式加载。vt-save 返回张量本身，加载必须传 path。"
  (vt-save vt path :as fmt)
  (vt-load path :as fmt))

(defun %save+load-auto (vt path fmt)
  "以 FMT 保存后，用 as=nil 自动检测加载。"
  (vt-save vt path :as fmt)
  (vt-load path))

(defun %roundtrip (vt fmt label)
  (let ((path (%path (format nil "rt-~a.~a" label (string-downcase (string fmt))))))
    (%same-tensor-p vt (%save+load vt path fmt))))

;; 构造含 NaN/±Inf 的浮点张量（vt-from-sequence 不接受关键字标记）
(defun %make-nonfinite-vt (dtype)
  (let ((vt (make-vt '(2 2) 0 :dtype dtype)))
    (setf (aref (vt-data vt) 0) (coerce 1.5 (vt-dtype->lisp-type dtype)))
    (setf (aref (vt-data vt) 1) (vt-get-nan dtype))
    (setf (aref (vt-data vt) 2) (vt-get-pos-inf dtype))
    (setf (aref (vt-data vt) 3) (vt-get-neg-inf dtype))
    vt))

(defun %save-fixture (bytes name)
  (let ((path (%path name)))
    (%write-bytes path bytes)
    path))

;;; ==================================================================
;;; 1/2. 往返矩阵：8 dtype × 形状矩阵 × 两种格式
;;; ==================================================================

(defun run-roundtrip-matrix ()
  (format t "~%--- 往返矩阵（8 dtype × 两种格式）---~%")
  (dolist (dtype '(:float64 :float32 :int64 :int32 :int16 :int8 :uint8 :uint16))
    (let ((vt (make-vt '(2 3) 0 :dtype dtype)))
      (vt-do-each (ptr v vt)
        (declare (ignore v))
        (setf (aref (vt-data vt) ptr) (funcall (vt-cast-fun dtype) (+ 17 ptr))))
      (check (format nil "lisp 往返 ~a (2 3)" dtype)
             (%roundtrip vt :lisp (symbol-name dtype)))
      (check (format nil "npy  往返 ~a (2 3)" dtype)
             (%roundtrip vt :npy (symbol-name dtype))))))

(defun run-shape-matrix ()
  (format t "~%--- 形状矩阵 ---~%")
  ;; 0-d
  (let ((s (make-vt nil 7 :dtype :int32)))
    (check "lisp 往返 0-d" (%roundtrip s :lisp "0d-i32"))
    (check "npy  往返 0-d" (%roundtrip s :npy "0d-i32"))
    (check "0-d 加载后 item 正确"
           (= (vt-item (%save+load s (%path "0d-v.npy") :npy)) 7)))
  ;; 1-d
  (let ((v (vt-arange 5 :dtype :float64)))
    (check "lisp 往返 (5)" (%roundtrip v :lisp "1d"))
    (check "npy  往返 (5)" (%roundtrip v :npy "1d")))
  ;; 空张量
  (let ((e (make-vt '(2 0 3) 1 :dtype :int16)))
    (check "lisp 往返 (2 0 3)" (%roundtrip e :lisp "empty"))
    (check "npy  往返 (2 0 3)" (%roundtrip e :npy "empty"))
    (check "空张量 size 保持 0"
           (zerop (vt-size (%save+load e (%path "empty.npy") :npy)))))
  ;; 3-d
  (let ((c (make-vt '(2 2 2) 0 :dtype :uint8)))
    (vt-do-each (ptr v c)
      (declare (ignore v))
      (setf (aref (vt-data c) ptr) ptr))
    (check "lisp 往返 (2 2 2)" (%roundtrip c :lisp "3d"))
    (check "npy  往返 (2 2 2)" (%roundtrip c :npy "3d"))))

(defun run-view-tests ()
  (format t "~%--- 非连续视图 ---~%")
  (let* ((base (vt-reshape (vt-arange 6 :dtype :float64) '(2 3)))
         (tp (vt-transpose base)))
    (check "转置视图 npy 往返" (%roundtrip tp :npy "transpose"))
    (check "转置视图内容与逻辑一致"
           (let ((back (%save+load tp (%path "tp.npy") :npy)))
             (equal (vt-to-list back) (vt-to-list tp))))))

;;; ==================================================================
;;; 3. 非有限值：NaN / ±Inf / -0.0
;;; ==================================================================

(defun run-nonfinite-tests ()
  (format t "~%--- NaN/Inf/-0.0 ---~%")
  (dolist (dtype '(:float64 :float32))
    (let ((vt (%make-nonfinite-vt dtype)))
      (check (format nil "~a npy 往返 NaN/±Inf" dtype)
             (let ((back (%save+load vt (%path (format nil "nf-~a.npy" dtype)) :npy)))
               (equal (mapcar #'%classify-float (coerce (%elements back) 'list))
                      (list 1.5d0 :nan :inf :neg-inf))))
      (check (format nil "~a lisp 往返 NaN/±Inf" dtype)
             (let ((back (%save+load vt (%path (format nil "nf-~a.sexp" dtype)) :lisp)))
               (equal (mapcar #'%classify-float (coerce (%elements back) 'list))
                      (list 1.5d0 :nan :inf :neg-inf))))))
  ;; -0.0 符号位
  (let ((z (make-vt '(2) 0.0d0 :dtype :float64)))
    (setf (aref (vt-data z) 0) -0.0d0)
    (setf (aref (vt-data z) 1) 0.0d0)
    (check "npy  -0.0 符号位保留"
           (let ((back (%save+load z (%path "negz.npy") :npy)))
             (and (%neg-zero-p (aref (vt-data back) 0))
                  (not (%neg-zero-p (aref (vt-data back) 1))))))
    (check "lisp -0.0 符号位保留"
           (let ((back (%save+load z (%path "negz.sexp") :lisp)))
             (and (%neg-zero-p (aref (vt-data back) 0))
                  (not (%neg-zero-p (aref (vt-data back) 1))))))))

;;; ==================================================================
;;; 4. numpy 固件加载 + 字节级锁定
;;; ==================================================================

(defun run-fixture-tests ()
  (format t "~%--- numpy 2.5.3 固件加载 ---~%")
  (let ((back (vt-load (%save-fixture *npy-fixture-f64-c* "fx-a.npy") :as :npy)))
    (check "固件 A：float64 (2 3) C-order"
           (and (equal (vt-shape back) '(2 3))
                (eq (vt-dtype back) :float64)
                (equal (coerce (%elements back) 'list)
                       '(0.0d0 1.0d0 2.0d0 3.0d0 4.0d0 5.0d0)))))
  (let ((back (vt-load (%save-fixture *npy-fixture-i32-c* "fx-b.npy") :as :npy)))
    (check "固件 B：int32 (3 4) C-order 负数补码"
           (equal (coerce (%elements back) 'list)
                  '(-42 -41 -40 -39 -38 -37 -36 -35 -34 -33 -32 -31))))
  (let ((back (vt-load (%save-fixture *npy-fixture-u8-c* "fx-c.npy") :as :npy)))
    (check "固件 C：uint8 (3 2) |u1 descr"
           (equal (coerce (%elements back) 'list) '(1 2 3 4 250 0))))
  (let ((back (vt-load (%save-fixture *npy-fixture-f64-f* "fx-d.npy") :as :npy)))
    (check "固件 D：fortran_order=True 自动重排"
           (equal (coerce (%elements back) 'list)
                  '(0.0d0 1.0d0 2.0d0 3.0d0 4.0d0 5.0d0))))
  (let ((back (vt-load (%save-fixture *npy-fixture-f64-0d* "fx-e.npy") :as :npy)))
    (check "固件 E：0-d shape=nil"
           (and (null (vt-shape back)) (= (vt-item back) 3.5d0))))
  (let ((back (vt-load (%save-fixture *npy-fixture-i16-empty* "fx-f.npy") :as :npy)))
    (check "固件 F：(2 0 3) 空张量"
           (and (equal (vt-shape back) '(2 0 3)) (zerop (vt-size back)))))
  ;; 字节级锁定：我方 npy 输出 == np.save 输出
  (let ((a (vt-reshape (vt-arange 6 :dtype :float64) '(2 3)))
        (path (%path "byte-lock.npy")))
    (vt-save a path :as :npy)
    (check "我方 npy 输出与 np.save 逐字节一致"
           (equalp (%read-file-bytes path) *npy-fixture-f64-c*))))

;;; ==================================================================
;;; 5. python3 + numpy 读回我方文件（可用性检测失败则 Skip）
;;; ==================================================================

(defun %python3-available-p ()
  (ignore-errors
    (zerop (sb-ext:process-exit-code
            (sb-ext:run-program "python3" '("-c" "import numpy")
                                :search t :wait t :output nil :error nil)))))

(defun run-python-interop ()
  (format t "~%--- numpy 读回验证 ---~%")
  (if (%python3-available-p)
      (let* ((script-path (%path "verify-npy.py"))
             (a (vt-reshape (vt-arange 6 :dtype :float64) '(2 3)))
             (i (make-vt '(3) 0 :dtype :int32)))
        (vt-do-each (ptr v i)
          (declare (ignore v))
          (setf (aref (vt-data i) ptr) (- (* ptr 10) 20)))
        (vt-save a (%path "py-a.npy") :as :npy)
        (vt-save i (%path "py-i.npy") :as :npy)
        (with-open-file (s script-path :direction :output :if-exists :supersede)
          (format s "import numpy as np~%")
          (format s "a = np.load(~s)~%" (namestring (%path "py-a.npy")))
          (format s "assert a.shape == (2, 3), a.shape~%")
          (format s "assert a.dtype == np.float64, a.dtype~%")
          (format s "assert (a == np.arange(6, dtype=np.float64).reshape(2, 3)).all(), a~%")
          (format s "i = np.load(~s)~%" (namestring (%path "py-i.npy")))
          (format s "assert i.dtype == np.int32, i.dtype~%")
          (format s "assert (i == np.array([-20, -10, 0], dtype=np.int32)).all(), i~%")
          (format s "print('PY-OK')~%"))
        (let ((code (sb-ext:process-exit-code
                     (sb-ext:run-program "python3" (list (namestring script-path))
                                         :search t :wait t :output nil :error nil))))
          (check "numpy 读回我方 float64/int32 npy" (zerop code))))
      (check-skip "python3/numpy 不可用，跳过 numpy 读回验证")))

;;; ==================================================================
;;; 6. 自动检测（as=nil）
;;; ==================================================================

(defun run-autodetect-tests ()
  (format t "~%--- 自动检测 ---~%")
  (let ((vt (vt-reshape (vt-arange 4 :dtype :int64) '(2 2))))
    (check "自动检测：npy 文件"
           (let ((back (%save+load-auto vt (%path "auto.npy") :npy)))
             (%same-tensor-p vt back)))
    (check "自动检测：lisp 文件"
           (let ((back (%save+load-auto vt (%path "auto.sexp") :lisp)))
             (%same-tensor-p vt back)))))

;;; ==================================================================
;;; 7. L2 错误契约
;;; ==================================================================

(defun run-error-contract ()
  (format t "~%--- L2 错误契约 ---~%")
  ;; 参数合法性
  (check-error "vt-save: tensor 非 vt"
               (lambda () (vt-save 42 (%path "e1.npy"))))
  (check-error "vt-save: path 非路径名"
               (lambda () (vt-save (make-vt '(2) 1) 123)))
  (check-error "vt-save: as 非法（:json）"
               (lambda () (vt-save (make-vt '(2) 1) (%path "e3.npy") :as :json)))
  (check-error "vt-load: path 非路径名"
               (lambda () (vt-load 99)))
  (check-error "vt-load: as 非法（:xml）"
               (lambda () (vt-load (%path "e3.npy") :as :xml)))
  (check-error "vt-load: 文件不存在"
               (lambda () (vt-load (%path "no-such-file-xyz.npy"))))
  ;; 坏 magic
  (%write-bytes (%path "bad-magic.npy")
                (coerce '(88 89 90 90 89 89 1 0 2 0 40 41)
                        '(vector (unsigned-byte 8))))
  (check-error "vt-load: npy magic 不匹配"
               (lambda () (vt-load (%path "bad-magic.npy") :as :npy)))
  ;; 不支持 dtype（float16）：hlen=100，dict 57 字符 + 43 空格 = 100
  (%write-bytes (%path "f16.npy")
                (coerce (append '(147 78 85 77 80 89 1 0 100 0)
                                (map 'list #'char-code
                                     "{'descr': '<f2', 'fortran_order': False, 'shape': (2,), }")
                                (make-list 43 :initial-element 32)
                                '(10)
                                '(0 0 192 63 0 0 128 63))
                        '(vector (unsigned-byte 8))))
  (check-error "vt-load: 不支持的 dtype <f2"
               (lambda () (vt-load (%path "f16.npy") :as :npy)))
  ;; 数据截断：完整 header + 不完整数据
  (%write-bytes (%path "trunc.npy") (subseq *npy-fixture-f64-c* 0 148))
  (check-error "vt-load: 数据区被截断"
               (lambda () (vt-load (%path "trunc.npy") :as :npy)))
  ;; lisp 格式错误
  (with-open-file (s (%path "garbage.sexp") :direction :output :if-exists :supersede)
    (princ "(:FOO :BAR)" s))
  (check-error "vt-load: lisp 格式缺少 :CLVT-TENSOR 标签"
               (lambda () (vt-load (%path "garbage.sexp") :as :lisp)))
  (with-open-file (s (%path "short.sexp") :direction :output :if-exists :supersede)
    (princ "(:CLVT-TENSOR :SHAPE (2 3) :DTYPE :FLOAT64 :DATA #(1.0d0 2.0d0))" s))
  (check-error "vt-load: lisp 格式 data 长度不匹配"
               (lambda () (vt-load (%path "short.sexp") :as :lisp)))
  (with-open-file (s (%path "badint.sexp") :direction :output :if-exists :supersede)
    (princ "(:CLVT-TENSOR :SHAPE (2) :DTYPE :INT32 :DATA #(1 2.5))" s))
  (check-error "vt-load: 整型 dtype 收到非整数"
               (lambda () (vt-load (%path "badint.sexp") :as :lisp)))
  (with-open-file (s (%path "empty.sexp") :direction :output :if-exists :supersede)
    (princ "" s))
  (check-error "vt-load: lisp 格式空文件"
               (lambda () (vt-load (%path "empty.sexp") :as :lisp))))

;;; ==================================================================
;;; 8. 对抗补强：形状扩展 / 视图字节等价 / NaN payload / 不变性
;;;    （吸收对方测试覆盖的补充项；位模式工具为测试侧 SAP 重解释）
;;; ==================================================================

(defun %f64-from-bits (bits)
  "位模式 → double-float（SAP 重解释，与文件字节往返等价）。"
  (let ((buf (make-array 8 :element-type '(unsigned-byte 8))))
    (loop for i below 8 do (setf (aref buf i) (ldb (byte 8 (* 8 i)) bits)))
    (sb-sys:with-pinned-objects (buf)
      (sb-sys:sap-ref-double (sb-sys:vector-sap buf) 0))))

(defun %f64-bits-of (x)
  "double-float → 位模式（SAP 重解释）。"
  (let ((buf (make-array 8 :element-type '(unsigned-byte 8))))
    (sb-sys:with-pinned-objects (buf)
      (setf (sb-sys:sap-ref-double (sb-sys:vector-sap buf) 0) x))
    (loop for i below 8 sum (ash (aref buf i) (* 8 i)))))

(defun %f32-from-bits (bits)
  (let ((buf (make-array 4 :element-type '(unsigned-byte 8))))
    (loop for i below 4 do (setf (aref buf i) (ldb (byte 8 (* 8 i)) bits)))
    (sb-sys:with-pinned-objects (buf)
      (sb-sys:sap-ref-single (sb-sys:vector-sap buf) 0))))

(defun %f32-bits-of (x)
  (let ((buf (make-array 4 :element-type '(unsigned-byte 8))))
    (sb-sys:with-pinned-objects (buf)
      (setf (sb-sys:sap-ref-single (sb-sys:vector-sap buf) 0) x))
    (loop for i below 4 sum (ash (aref buf i) (* 8 i)))))

(defun run-hardening-tests ()
  (format t "~%--- 对抗补强：形状扩展/视图等价/NaN payload/不变性 ---~%")
  ;; 形状矩阵扩展：(1) 单元素 / (0) 一维空 / (2 0) 二维空 / (2 3 4) 三维
  (let ((n 0))
    (dolist (shape '((1) (0) (2 0) (2 3 4)))
      (incf n)
      (let ((vt (make-vt shape 3 :dtype :int32)))
        (check (format nil "lisp 往返 ~s" shape)
               (%roundtrip vt :lisp (format nil "hs~d" n)))
        (check (format nil "npy  往返 ~s" shape)
               (%roundtrip vt :npy (format nil "hs~d" n))))))
  ;; 负步长视图：内容一致 + 与 vt-copy 保存逐字节一致
  (let* ((base (vt-reshape (vt-arange 6 :dtype :float64) '(2 3)))
         (rev (vt-slice base '(nil nil -1))))
    (check "负步长视图 npy 保存后内容 == 视图逻辑内容"
           (let ((back (%save+load rev (%path "rev.npy") :npy)))
             (equal (vt-to-list back) (vt-to-list rev))))
    (let ((p1 (%path "rev-view.npy"))
          (p2 (%path "rev-copy.npy")))
      (vt-save rev p1 :as :npy)
      (vt-save (vt-copy rev) p2 :as :npy)
      (check "负步长视图保存 == vt-copy 保存（npy 逐字节一致）"
             (equalp (%read-file-bytes p1) (%read-file-bytes p2)))))
  ;; stride 切片视图（步长 2）
  (let* ((base (vt-arange 8 :dtype :int64))
         (evens (vt-slice base '(0 8 2))))
    (check "stride 切片视图 lisp 保存后内容一致"
           (equal (vt-to-list (%save+load evens (%path "evens.sexp") :lisp))
                  (vt-to-list evens))))
  ;; NaN payload 逐位往返（npy，f64：+payload 与 -payload）
  (let ((vt (make-vt '(2) 0 :dtype :float64)))
    (setf (aref (vt-data vt) 0) (%f64-from-bits #x7FF8000000000ABC))
    (setf (aref (vt-data vt) 1) (%f64-from-bits #xFFF8000000000ABC))
    (let ((back (%save+load vt (%path "pl64.npy") :npy)))
      (check "npy f64 NaN payload 逐位保留（含符号位）"
             (and (= (%f64-bits-of (aref (vt-data back) 0)) #x7FF8000000000ABC)
                  (= (%f64-bits-of (aref (vt-data back) 1)) #xFFF8000000000ABC)))))
  ;; NaN payload 逐位往返（npy，f32）
  (let ((vt (make-vt '(2) 0 :dtype :float32)))
    (setf (aref (vt-data vt) 0) (%f32-from-bits #x7FC00123))
    (setf (aref (vt-data vt) 1) (%f32-from-bits #xFFC0F00D))
    (let ((back (%save+load vt (%path "pl32.npy") :npy)))
      (check "npy f32 NaN payload 逐位保留（含符号位）"
             (and (= (%f32-bits-of (aref (vt-data back) 0)) #x7FC00123)
                  (= (%f32-bits-of (aref (vt-data back) 1)) #xFFC0F00D)))))
  ;; 非规格化数逐位（f64 最小非规格化 / f32 最负非规格化）
  (let ((vt (make-vt '(2) 0 :dtype :float64)))
    (setf (aref (vt-data vt) 0) (%f64-from-bits #x0000000000000001))
    (setf (aref (vt-data vt) 1) (%f64-from-bits #x800FFFFFFFFFFFFF))
    (let ((back (%save+load vt (%path "den64.npy") :npy)))
      (check "npy f64 非规格化数逐位保留"
             (and (= (%f64-bits-of (aref (vt-data back) 0)) #x0000000000000001)
                  (= (%f64-bits-of (aref (vt-data back) 1)) #x800FFFFFFFFFFFFF)))))
  (let ((vt (make-vt '(2) 0 :dtype :float32)))
    (setf (aref (vt-data vt) 0) (%f32-from-bits #x00000001))
    (setf (aref (vt-data vt) 1) (%f32-from-bits #x807FFFFF))
    (let ((back (%save+load vt (%path "den32.npy") :npy)))
      (check "npy f32 非规格化数逐位保留"
             (and (= (%f32-bits-of (aref (vt-data back) 0)) #x00000001)
                  (= (%f32-bits-of (aref (vt-data back) 1)) #x807FFFFF)))))
  ;; 输入不变性：保存不得修改源张量
  (let* ((src (vt-reshape (vt-arange 6 :dtype :int32) '(2 3)))
         (before (vt-to-list src)))
    (vt-save src (%path "imm.npy") :as :npy)
    (vt-save src (%path "imm.sexp") :as :lisp)
    (check "vt-save 不修改输入张量（npy + lisp）"
           (equal (vt-to-list src) before)))
  ;; 覆盖语义：同路径二次保存以第二次为准
  (let ((p (%path "ow.npy")))
    (vt-save (make-vt '(2) 1 :dtype :int32) p :as :npy)
    (vt-save (make-vt '(3) 2 :dtype :int32) p :as :npy)
    (let ((back (vt-load p :as :npy)))
      (check "同路径二次保存以第二次为准（supersede）"
             (and (equal (vt-shape back) '(3))
                  (equal (vt-to-list back) '(2 2 2))))))
  ;; :VERSION 标记：写入可读回、旧文件兼容（已由往返矩阵隐含）、未来版本报错
  (with-open-file (s (%path "ver1.sexp") :direction :output :if-exists :supersede)
    (prin1 (list :clvt-tensor :version 1 :shape '(2) :dtype :int32 :data #(9 8)) s))
  (check "lisp :VERSION 1 标记往返"
         (equal (vt-to-list (vt-load (%path "ver1.sexp") :as :lisp)) '(9 8)))
  (with-open-file (s (%path "ver2.sexp") :direction :output :if-exists :supersede)
    (prin1 (list :clvt-tensor :version 2 :shape '(1) :dtype :int32 :data #(1)) s))
  (check-error "lisp 未来版本确定性报错"
               (lambda () (vt-load (%path "ver2.sexp") :as :lisp)))
  ;; 解析器宽容性：双引号 header（第三方写出器），hlen=118 / 总头 128
  (let* ((dict "{\"descr\": \"<i4\", \"fortran_order\": False, \"shape\": (3), }")
         (pad (- 118 (length dict) 1)))
    (%write-bytes (%path "dq.npy")
                  (coerce (append '(147 78 85 77 80 89 1 0 118 0)
                                  (map 'list #'char-code dict)
                                  (make-list pad :initial-element 32)
                                  '(10)
                                  '(1 0 0 0 2 0 0 0 3 0 0 0))
                          '(vector (unsigned-byte 8)))))
  (check "npy 双引号 header 可读（第三方兼容）"
         (equal (vt-to-list (vt-load (%path "dq.npy") :as :npy)) '(1 2 3))))

;;; ==================================================================
;;; 主流程
;;; ==================================================================

(defun run ()
  (run-roundtrip-matrix)
  (run-shape-matrix)
  (run-view-tests)
  (run-nonfinite-tests)
  (run-fixture-tests)
  (run-python-interop)
  (run-autodetect-tests)
  (run-hardening-tests)
  (run-error-contract)
  (format t "~%通过 ~a / 失败 ~a / Skip: ~a~%" *pass* *fail* *skip*)
  (finish-output)
  (zerop *fail*))

(sb-ext:exit :code (if (run) 0 1))
