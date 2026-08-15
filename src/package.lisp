;;;; package.lisp — clvt 包定义与公共 API 导出
;;;;
;;;; 所有函数均以 `vt-` 为前缀，便于在 Slime 中通过补全浏览。
;;;; 导出符号按功能分层组织，与 src/ 目录下的模块一一对应。

(defpackage #:clvt
  (:use #:cl)
  (:export
   ;; ------------------------------------------------------------------
   ;; 张量结构访问器 (core.lisp)
   ;; ------------------------------------------------------------------
   #:vt
   #:vt-shape
   #:vt-strides
   #:vt-offset
   #:vt-data
   #:vt-element-type
   #:vt-order
   #:vt-size
   #:vt-p
   #:vt-itemsize
   #:vt-nbytes
   #:vt-contiguous-p
   #:vt-shape-to-size
   #:vt-compute-strides
   #:vt-compute-logical-strides

   ;; ------------------------------------------------------------------
   ;; 张量创建 (creation.lisp)
   ;; ------------------------------------------------------------------
   #:vt-zeros
   #:vt-ones
   #:vt-full
   #:vt-empty
   #:vt-zeros-like
   #:vt-ones-like
   #:vt-full-like
   #:vt-empty-like
   #:vt-const
   #:vt-arange
   #:vt-linspace
   #:vt-logspace
   #:vt-eye
   #:vt-diag
   #:vt-identity
   #:vt-from-sequence
   #:vt-from-array
   #:vt-from-function
   #:vt-flatten-sequence
   #:vt-to-list
   #:vt-to-array
   #:vt-astype

   ;; ------------------------------------------------------------------
   ;; 形状操作与视图 (manip.lisp / join.lisp)
   ;; ------------------------------------------------------------------
   #:vt-view
   #:vt-reshape
   #:vt-transpose
   #:vt-squeeze
   #:vt-unsqueeze
   #:vt-expand-dims
   #:vt-flatten
   #:vt-ravel
   #:vt-swapaxes
   #:vt-rot90
   #:vt-narrow
   #:vt-split
   #:vt-vsplit
   #:vt-hsplit
   #:vt-dsplit
   #:vt-stack
   #:vt-vstack
   #:vt-hstack
   #:vt-dstack
   #:vt-concatenate
   #:vt-concat
   #:vt-repeat
   #:vt-tile
   #:vt-pad
   #:vt-broadcast-to
   #:vt-broadcast-shapes
   #:vt-broadcast-strides
   #:vt-contiguous
   #:vt-flip
   #:vt-roll
   #:vt-triu
   #:vt-tril
   #:vt-diagonal
   #:vt-flatten-to-nested
   #:vt-append
   #:vt-insert
   #:vt-delete

   ;; ------------------------------------------------------------------
   ;; 索引、切片与选择 (indexing.lisp)
   ;; ------------------------------------------------------------------
   #:vt-ref   ;; (setf vt-ref ...)
   #:vt-slice ;; (setf vt-slice ...)
   #:vt-item
   #:vt-take
   #:vt-put
   #:vt-where
   #:vt-argwhere
   #:vt-nonzero
   #:vt-choose
   #:vt-select
   #:vt-extract
   #:vt-searchsorted
   #:vt-digitize
   #:vt-bincount
   #:vt-normalize-axis

   ;; ------------------------------------------------------------------
   ;; 算术与数学 (elementwise.lisp)
   ;; ------------------------------------------------------------------
   #:vt-+
   #:vt--
   #:vt-*
   #:vt-/
   #:vt-add
   #:vt-sub
   #:vt-mul
   #:vt-div
   #:vt-scale
   #:vt-square
   #:vt-expt
   #:vt-pow
   #:vt-sqrt
   #:vt-abs
   #:vt-signum
   #:vt-mod
   #:vt-rem
   #:vt-round
   #:vt-floor
   #:vt-ceiling
   #:vt-truncate
   #:vt-rint
   #:vt-log
   #:vt-log2
   #:vt-log10
   #:vt-exp
   #:vt-clip

   ;; 三角函数与双曲函数
   #:vt-sin
   #:vt-cos
   #:vt-tan
   #:vt-asin
   #:vt-acos
   #:vt-atan
   #:vt-atan2
   #:vt-sinh
   #:vt-cosh
   #:vt-tanh
   #:vt-hypot
   #:vt-sinc
   #:vt-deg2rad
   #:vt-rad2deg
   #:vt-asinh
   #:vt-acosh
   #:vt-atanh
   #:vt-cbrt
   #:vt-reciprocal
   #:vt-negative
   #:vt-lerp

   ;; 位运算与逐元素极值
   #:vt-bit-and
   #:vt-bit-ior
   #:vt-bit-xor
   #:vt-bit-not
   #:vt-left-shift
   #:vt-right-shift
   #:vt-fmax
   #:vt-fmin
   #:vt-maximum
   #:vt-minimum

   ;; ------------------------------------------------------------------
   ;; 比较与逻辑 (elementwise.lisp)
   ;; ------------------------------------------------------------------
   #:vt-=
   #:vt-/=
   #:vt-<
   #:vt-<=
   #:vt->
   #:vt->=
   #:vt-positive-p
   #:vt-negative-p
   #:vt-zero-p
   #:vt-nonzero-p
   #:vt-even-p
   #:vt-odd-p
   #:vt-logical-and
   #:vt-logical-or
   #:vt-logical-not
   #:vt-logical-xor
   #:vt-all
   #:vt-any
   #:vt-isclose
   #:vt-allclose
   #:vt-isfinite
   #:vt-isinf
   #:vt-isnan

   ;; ------------------------------------------------------------------
   ;; 归约与统计 (reduce-stats.lisp)
   ;; ------------------------------------------------------------------
   #:vt-sum
   #:vt-mean
   #:vt-average
   #:vt-std
   #:vt-var
   #:vt-amax
   #:vt-amin
   #:vt-argmax
   #:vt-argmin
   #:vt-prod
   #:vt-cumsum
   #:vt-cumprod
   #:vt-median
   #:vt-percentile
   #:vt-quantile
   #:vt-ptp
   #:vt-histogram
   #:vt-trapz
   #:vt-gradient
   #:vt-diff
   #:vt-correlate
   #:vt-convolve
   #:vt-sort
   #:vt-argsort

   ;; nan 感知统计
   #:vt-nansum
   #:vt-nanmean
   #:vt-nanstd
   #:vt-nanvar
   #:vt-nanmax
   #:vt-nanmin
   #:vt-nanargmax
   #:vt-nanargmin
   #:vt-nanprod
   #:vt-nanmedian

   ;; ------------------------------------------------------------------
   ;; 线性代数 (linalg.lisp)
   ;; ------------------------------------------------------------------
   #:vt-matmul
   #:vt-@
   #:vt-einsum
   #:vt-dot
   #:vt-outer
   #:vt-trace
   #:vt-norm
   #:vt-l1-norm
   #:vt-frobenius-norm
   #:vt-solve
   #:vt-inv
   #:vt-det
   #:vt-lu
   #:vt-qr
   #:vt-svd
   #:vt-matrix-rank
   #:vt-cholesky
   #:vt-eig
   #:vt-pinv
   #:vt-lstsq

   ;; ------------------------------------------------------------------
   ;; 填充与插值
   ;; ------------------------------------------------------------------
   #:vt-fill
   #:vt-interp
   #:vt-kron
   #:vt-meshgrid

   ;; ------------------------------------------------------------------
   ;; 神经网络：激活 / 损失 (nn.lisp)
   ;; ------------------------------------------------------------------
   #:vt-sigmoid
   #:vt-relu
   #:vt-leaky-relu
   #:vt-swish
   #:vt-softplus
   #:vt-gelu
   #:vt-mish
   #:vt-hard-tanh
   #:vt-hard-sigmoid
   #:vt-softmax
   #:vt-log-softmax
   #:vt-mean-squared-error
   #:vt-binary-cross-entropy
   #:vt-cross-entropy

   ;; ------------------------------------------------------------------
   ;; 集合操作 (setops.lisp)
   ;; ------------------------------------------------------------------
   #:vt-unique
   #:vt-intersect1d
   #:vt-union1d
   #:vt-setdiff1d
   #:vt-setxor1d
   #:vt-in1d

   ;; ------------------------------------------------------------------
   ;; 随机数生成 (random.lisp)
   ;; ------------------------------------------------------------------
   #:vt-random
   #:vt-random-uniform
   #:vt-random-normal
   #:vt-random-int
   #:vt-random-integers
   #:vt-random-seed
   #:vt-random-choice
   #:vt-random-permutation
   #:vt-random-shuffle
   #:vt-random-multinomial

   ;; ------------------------------------------------------------------
   ;; nan / inf 相关 (nan.lisp)
   ;; ------------------------------------------------------------------
   #:vt-float-nan
   #:vt-float-nan-p
   #:vt-float-nan-=
   #:vt-float-pos-inf
   #:vt-float-neg-inf
   #:vt-float-pos-inf-p
   #:vt-float-neg-inf-p
   #:vt-float-inf-=
   #:vt-float-nan-inf-=
   #:+vt-float-nan+
   #:+vt-float-pos-inf+
   #:+vt-float-neg-inf+

   ;; ------------------------------------------------------------------
   ;; 核心迭代与映射 (map-reduce.lisp)
   ;; ------------------------------------------------------------------
   #:vt-map
   #:vt-do-each
   #:vt-reduce
   #:vt-copy-into
   #:vt-copy

   ;; ------------------------------------------------------------------
   ;; 通用辅助与宏
   ;; ------------------------------------------------------------------
   #:vt-normalize-axis
   #:vt-broadcast-shapes
   #:vt-broadcast-strides
   #:vt-compute-strides
   #:vt-compute-logical-strides
   #:with-float-safe

   ;; ------------------------------------------------------------------
   ;; 打印与调试 (io.lisp)
   ;; ------------------------------------------------------------------
   #:print-vt-recursive
   #:*vt-print-threshold*
   #:*vt-print-precision*
   #:*vt-indent-step*
   #:*vt-fun-list*
   #:*vt-einsum-parse-cache*

   ;; ------------------------------------------------------------------
   ;; 扩展功能 (extensions.lisp)
   ;; ------------------------------------------------------------------
   #:vt-count-nonzero
   #:vt-moveaxis
   #:vt-inner
   #:vt-tensordot
   #:vt-topk
   #:vt-set-print-options
   #:vt-get-print-options
   #:vt-flatnonzero
   #:vt-count
   #:vt-clip-tensor))

(in-package :clvt)

;;; 供测试与文档使用：收集所有以 `vt-` 开头的导出符号。
(defparameter *vt-fun-list* nil
  "所有以 `vt-` 开头的导出符号列表。")

(defun %refresh-vt-fun-list ()
  (setf *vt-fun-list* nil)
  (do-symbols (var :clvt)
    (when (search "vt-" (symbol-name var) :test #'equalp)
      (push var *vt-fun-list*)))
  (setf *vt-fun-list* (nreverse *vt-fun-list*)))

(%refresh-vt-fun-list)
