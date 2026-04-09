;;;; package.lisp

(defpackage #:clvt
  (:use #:cl)
  (:export

   :vt
   :vt-data
   :vt-strides
   :vt-offset
   :vt-element-type
   :vt-order
   :vt-shape
   :vt-size
   
   :vt-flatten-sequence
   :vt-from-sequence
   :vt-data->list
   
   :vt-zeros
   :vt-ones
   :vt-const
   :vt-arange
   :vt-random

   :vt-transpose
   :vt-reshape
   :vt-split
   :vt-ref     ;; setf 
   :vt-slice*  ;; setf
   :vt-do-each
   :vt-copy-into
   :vt-copy
   :vt-map
   :vt-contiguous
   
   :vt-+
   :vt--
   :vt-*
   :vt-/
   
   :vt-scale
   :vt-matmul
   :vt-einsum
   
   :vt-sum
   :vt-amax
   :vt-amin
   :vt-argmax
   :vt-argmin
   
   :vt->
   :vt->=
   :vt-<
   :vt-<=
   :vt-=
   :vt-/=

   :vt-sin
   :vt-cos
   :vt-tan
   :vt-asin
   :vt-acos
   :vt-atan
   :vt-sinh
   :vt-cosh
   :vt-tanh

   :vt-exp
   :vt-sqrt

   :vt-abs
   :vt-signum

   :vt-positive-p
   :vt-negative-p
   :vt-zero-p
   :vt-nonzero-p
   :vt-even-p
   :vt-odd-p
   
   :vt-expt
   :vt-mod
   :vt-rem

   :vt-atan2
   :vt-floor
   :vt-ceiling
   :vt-round
   :vt-rint

   :vt-log
   :vt-log2
   :vt-log10
   
   ))

(in-package :clvt)
(defparameter *vt-fun-list* nil)
(do-symbols (var :clvt)
  (when (search "vt-" (symbol-name var) :test #'equalp)
    (push var *vt-fun-list*)))

