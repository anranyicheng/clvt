;;;; dtype.lisp — 元素数据类型系统（单一事实来源）

(in-package :clvt)

(defparameter *vt-dtypes*
  '(:float64 :float32 :int64 :int32 :int16 :int8 :uint8 :uint16)
  "库内识别到的 dtype 全集（前四者为物理存储类型）。")

(defparameter *vt-storage-dtypes*
  '(:float64 :float32 :int64 :int32)
  "实际可分配底层数组的元素类型。")

(declaim (inline vt-dtype-p vt-float-dtype-p vt-int-dtype-p vt-storage-dtype-p))

(defun vt-dtype->lisp-type (dtype)
  "将内部 dtype 符号映射为 Common Lisp 数组元素类型。"
  (ecase dtype
    (:float64 'double-float)
    (:float32 'single-float)
    (:int64   '(signed-byte 64))
    (:int32   '(signed-byte 32))
    (:int16   '(signed-byte 16))
    (:int8    '(signed-byte 8))
    (:uint8   '(unsigned-byte 8))
    (:uint16  '(unsigned-byte 16))))

(defun vt-dtype-itemsize (dtype)
  "返回每个元素的字节大小。"
  (ecase dtype
    (:float64 8)
    (:float32 4)
    (:int64   8)
    (:int32   4)
    (:int16   2)
    (:int8    1)
    (:uint8   1)
    (:uint16  2)))

(defun vt-dtype-p (x)
  (not (null (member x *vt-dtypes*))))

(defun vt-storage-dtype-p (x)
  (not (null (member x *vt-storage-dtypes*))))

(defun vt-float-dtype-p (dtype)
  (member dtype '(:float32 :float64)))

(defun vt-int-dtype-p (dtype)
  (member dtype '(:int8 :int16 :int32 :int64)))

(defun vt-signed-dtype-p (dtype)
  (member dtype '(:int8 :int16 :int32 :int64)))

(defun vt-unsigned-dtype-p (dtype)
  (member dtype '(:uint8 :uint16))) ; 以后扩展 uint32/uint64 时在这里加

(defun vt-int-bits (dtype)
  (ecase dtype
    ((:int8 :uint8)   8)
    ((:int16 :uint16) 16)
    (:int32           32)
    (:int64           64)))

(defun vt-bits->signed-dtype (bits)
  (ecase bits
    (8  :int8)
    (16 :int16)
    (32 :int32)
    (64 :int64)))

(defun vt-bits->unsigned-dtype (bits)
  (ecase bits
    (8  :uint8)
    (16 :uint16)
    (32 :uint32)     ; 如果以后不支持 uint32，可删除这两行
    (64 :uint64)))   ; 如果以后不支持 uint64，可删除这两行

(defun vt-next-signed-bits (unsigned-bits)
  "返回能容纳 unsigned-bits 无符号整数范围的有符号位宽；若没有则返回 nil。"
  (ecase unsigned-bits
    (8  16)
    (16 32)
    (32 64)
    (64 nil)))

(defun vt-promote-type (&rest dtypes)
  (let ((has-f64 nil)
        (has-f32 nil)
        (max-signed-bits 0)
        (max-unsigned-bits 0))
    (dolist (d dtypes)
      (cond
        ((eq d :float64)
         (setf has-f64 t))
        ((eq d :float32)
         (setf has-f32 t))
        ((vt-signed-dtype-p d)
         (setf max-signed-bits
               (max max-signed-bits (vt-int-bits d))))
        ((vt-unsigned-dtype-p d)
         (setf max-unsigned-bits
               (max max-unsigned-bits (vt-int-bits d))))))

    (cond
      ;; float64 优先
      (has-f64 :float64)

      ;; float32 与整数混合：超过 24 位有效位则提升到 float64
      (has-f32
       (if (> (max max-signed-bits max-unsigned-bits) 24)
           :float64
           :float32))

      ;; 有符号 + 无符号混合
      ((and (> max-signed-bits 0) (> max-unsigned-bits 0))
       (if (> max-signed-bits max-unsigned-bits)
           (vt-bits->signed-dtype max-signed-bits)
           (let ((next-bits (vt-next-signed-bits max-unsigned-bits)))
             (if next-bits
                 (vt-bits->signed-dtype next-bits)
                 :float64))))

      ;; 只有有符号整数
      ((> max-signed-bits 0)
       (vt-bits->signed-dtype max-signed-bits))

      ;; 只有无符号整数
      ((> max-unsigned-bits 0)
       (vt-bits->unsigned-dtype max-unsigned-bits))

      ;; 空输入或未知类型兜底
      (t :float64))))


(declaim (inline %wrap-int8 %wrap-int16 %wrap-uint8 %wrap-uint16))

(defun %wrap-int8 (v)
  "将截断后的整数按 int8 范围 [-128,127] 回绕（NumPy 语义）。"
  (let ((m (mod (truncate v) 256)))
    (if (>= m 128) (- m 256) m)))

(defun %wrap-int16 (v)
  "将截断后的整数按 int16 范围 [-32768,32767] 回绕（NumPy 语义）。"
  (let ((m (mod (truncate v) 65536)))
    (if (>= m 32768) (- m 65536) m)))

(defun %wrap-uint8 (v)
  "将截断后的整数按 uint8 范围 [0,255] 回绕（NumPy 语义）。"
  (mod (truncate v) 256))

(defun %wrap-uint16 (v)
  "将截断后的整数按 uint16 范围 [0,65535] 回绕（NumPy 语义）。"
  (mod (truncate v) 65536))

(defun vt-cast (val dtype)
  "安全类型转换。浮点转整数时执行截断，超出目标整数范围时按 NumPy 语义回绕（mod）。"
  (ecase dtype
    (:float64 (coerce val 'double-float))
    (:float32 (coerce val 'single-float))
    (:int64   (truncate val))
    (:int32   (truncate val))
    (:int16   (%wrap-int16 val))
    (:int8    (%wrap-int8 val))
    (:uint8   (%wrap-uint8 val))
    (:uint16  (%wrap-uint16 val))))

(defun vt-cast-fun (dtype)
  "返回将数值转换为 dtype 的转换函数（与 vt-cast 语义一致）。"
  (ecase dtype
    (:float64 (lambda (val) (coerce val 'double-float)))
    (:float32 (lambda (val) (coerce val 'single-float)))
    (:int64   #'truncate)
    (:int32   #'truncate)
    (:int16   #'%wrap-int16)
    (:int8    #'%wrap-int8)
    (:uint8   #'%wrap-uint8)
    (:uint16  #'%wrap-uint16)))

(defun vt-dtype-default-value (dtype)
  "返回 dtype 对应的零值。"
  (ecase dtype
    (:float64 0.0d0)
    (:float32 0.0f0)
    ((:int64 :int32 :int16 :int8 :uint8 :uint16) 0)))
