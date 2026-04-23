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

   :vt-squeeze
   :vt-unsqueeze
   :vt-expand-dims
   :vt-flatten
   :vt-flatten-sequence
   :vt-from-sequence
   :vt-from-array
   :vt-data->list
   :vt-take
   
   :vt-zeros
   :vt-ones
   :vt-const
   :vt-arange
   :vt-random
   :vt-random-normal

   :vt-transpose
   :vt-reshape
   :vt-split
   :vt-ref     ;; setf 
   :vt-slice   ;; setf
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
   :vt-dot
   :vt-outer
   :vt-trace
   :vt-norm
   :vt-l1-norm
   :vt-frobenius-norm

   :vt-solve
   :vt-det
   :vt-inv
   :vt-to-2d-array
   :vt-from-2d-array
   
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
   :vt-pow
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

   :vt-mean
   :vt-std

   :vt-concatenate
   :vt-concat
   :vt-clip

   :vt-sigmoid
   :vt-relu
   :vt-leaky-relu
   :vt-swish
   :vt-softplus
   :vt-gelu
   :vt-mish
   :vt-hard-tanh
   :vt-hard-sigmoid

   :vt-softmax
   :vt-log-softmax
   :vt-mean-squared-error
   :vt-binary-cross-entropy
   :vt-cross-entropy
   
   ))

(in-package :clvt)
(defparameter *vt-fun-list* nil)
(do-symbols (var :clvt)
  (when (search "vt-" (symbol-name var) :test #'equalp)
    (push var *vt-fun-list*)))

