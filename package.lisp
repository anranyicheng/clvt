;;;; package.lisp

(defpackage #:clvt
  (:use #:cl)
  (:export
   ;; 张量结构访问器
   :vt
   :vt-shape
   :vt-strides
   :vt-offset
   :vt-data
   :vt-element-type
   :vt-order
   :vt-size
   :vt-p
   :vt-itemsize
   :vt-nbytes
   :vt-contiguous-p
   :vt-shape-to-size
   :vt-compute-strides
   :vt-compute-logical-strides

   ;; 张量创建
   :vt-zeros
   :vt-ones
   :vt-full
   :vt-empty
   :vt-zeros-like
   :vt-ones-like
   :vt-full-like
   :vt-empty-like
   :vt-const
   :vt-arange
   :vt-linspace
   :vt-logspace
   :vt-eye
   :vt-diag
   :vt-identity
   :vt-from-sequence
   :vt-from-array
   :vt-from-function
   :vt-flatten-sequence
   :vt-to-2d-array
   :vt-from-2d-array
   :vt-data->list
   :vt-tolist
   :vt-astype

   ;; 形状操作与视图
   :vt-reshape
   :vt-transpose
   :vt-squeeze
   :vt-unsqueeze
   :vt-expand-dims
   :vt-flatten
   :vt-ravel
   :vt-swapaxes
   :vt-narrow
   :vt-split
   :vt-vsplit
   :vt-hsplit
   :vt-dsplit
   :vt-stack
   :vt-vstack
   :vt-hstack
   :vt-dstack
   :vt-concatenate
   :vt-concat
   :vt-repeat
   :vt-tile
   :vt-pad
   :vt-broadcast-to
   :vt-broadcast-shapes
   :vt-broadcast-strides
   :vt-contiguous
   :vt-flip
   :vt-roll
   :vt-triu
   :vt-tril
   :vt-diagonal
   :vt-flatten-to-nested

   ;; 索引、切片与选择
   :vt-ref   ;; (setf :vt-ref
   :vt-slice ;; (setf :vt-slice
   :vt-take
   :vt-put
   :vt-where
   :vt-argwhere
   :vt-nonzero
   :vt-choose
   :vt-select
   :vt-extract
   :vt-searchsorted
   :vt-digitize
   :vt-bincount
   :vt-ensure-shape-compatible
   :vt-normalize-axis

   ;; 算术运算
   :vt-+
   :vt--
   :vt-*
   :vt-/
   :vt-add
   :vt-sub
   :vt-mul
   :vt-div
   :vt-scale
   :vt-square
   :vt-expt
   :vt-pow
   :vt-sqrt
   :vt-abs
   :vt-signum
   :vt-mod
   :vt-rem
   :vt-round
   :vt-floor
   :vt-ceiling
   :vt-trancate
   :vt-rint
   :vt-log
   :vt-log2
   :vt-log10
   :vt-exp
   :vt-clip

   ;; 三角函数与双曲函数
   :vt-sin
   :vt-cos
   :vt-tan
   :vt-asin
   :vt-acos
   :vt-atan
   :vt-atan2
   :vt-sinh
   :vt-cosh
   :vt-tanh
   :vt-hypot
   :vt-sinc
   :vt-deg2rad
   :vt-rad2deg

   ;; 比较与逻辑
   :vt-=
   :vt-/=
   :vt-<
   :vt-<=
   :vt->
   :vt->=
   :vt-positive-p
   :vt-negative-p
   :vt-zero-p
   :vt-nonzero-p
   :vt-even-p
   :vt-odd-p
   :vt-logical-and
   :vt-logical-or
   :vt-logical-not
   :vt-logical-xor
   :vt-all
   :vt-any
   :vt-isclose
   :vt-allclose
   :vt-isfinite
   :vt-isinf
   :vt-isnan

   ;; 归约与统计
   :vt-sum
   :vt-mean
   :vt-average
   :vt-std
   :vt-var
   :vt-amax
   :vt-amin
   :vt-argmax
   :vt-argmin
   :vt-prod
   :vt-cumsum
   :vt-cumprod
   :vt-median
   :vt-percentile
   :vt-quantile
   :vt-ptp
   :vt-histogram
   :vt-trapz
   :vt-gradient
   :vt-diff
   :vt-correlate
   :vt-convolve
   :vt-sort
   :vt-argsort
   :vt-maximum
   :vt-minimum

   ;; 线性代数
   :vt-matmul
   :vt-matmul-df
   :vt-@
   :vt-einsum
   :vt-dot
   :vt-outer
   :vt-trace
   :vt-norm
   :vt-l1-norm
   :vt-frobenius-norm
   :vt-solve
   :vt-inv
   :vt-det
   :vt-lu
   :vt-diag
   :vt-triu
   :vt-tril
   :vt-diagonal
   :vt-qr
   :vt-svd

   ;; 激活函数
   :vt-sigmoid
   :vt-relu
   :vt-leaky-relu
   :vt-swish
   :vt-softplus
   :vt-gelu
   :vt-mish
   :vt-hard-tanh
   :vt-hard-sigmoid

   ;; 损失函数与概率
   :vt-softmax
   :vt-log-softmax
   :vt-mean-squared-error
   :vt-binary-cross-entropy
   :vt-cross-entropy

   ;; 集合操作
   :vt-unique
   :vt-intersect1d
   :vt-union1d
   :vt-setdiff1d
   :vt-setxor1d
   :vt-in1d

   ;; 随机数生成
   :vt-random
   :vt-random-uniform
   :vt-random-normal
   :vt-random-int
   :vt-random-integers
   :vt-random-seed

   ;; 核心迭代与映射
   :vt-map
   :vt-do-each
   :vt-reduce
   :vt-copy-into
   :vt-copy
   :vt-get-contiguous-df-data

   ;; 通用辅助与宏
   :vt-normalize-axis
   :vt-broadcast-shapes
   :vt-broadcast-strides
   :vt-compute-strides
   :vt-compute-logical-strides
   :vt-ensure-shape-compatible

   ;; 打印与调试
   :print-vt-recursive
   :*vt-print-threshold*
   :*vt-print-precision*
   :*vt-indent-step*
   :*vt-fun-list*
   :*vt-einsum-parse-cache*))

(in-package :clvt)
(defparameter *vt-fun-list* nil)
(do-symbols (var :clvt)
  (when (search "vt-" (symbol-name var) :test #'equalp)
    (push var *vt-fun-list*)))

