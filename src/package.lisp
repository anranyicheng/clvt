;;;; package.lisp — clvt 包定义与公共 API 导出
;;;;
;;;; 所有函数均以 `vt-` 为前缀，便于在 Slime 中通过补全浏览。
;;;; 导出符号按功能分层组织，与 src/ 目录下的模块一一对应。
;;;;
;;;; ==================================================================
;;;; 架构总纲：三层分离（逻辑层 / 物理层 / 执行层）
;;;; ==================================================================
;;;; 本库的全部复杂性源于三个正交维度的交织，架构上强制分离：
;;;;
;;;;   逻辑层  shape、dtype、广播规则、归约语义
;;;;           失败模式：形状不匹配、dtype 不支持
;;;;           —— dtype.lisp（类型提升）、各 API 的形状推导与归约语义
;;;;
;;;;   物理层  strides、offset、连续性、内存布局
;;;;           失败模式：越界访问、别名冲突
;;;;           —— core.lisp（strided view、广播 stride-0、连续判定、
;;;;              别名区间检测与快照保护）
;;;;
;;;;   执行层  快路径 vs 通用路径、SIMD、并行调度
;;;;           失败模式：性能退化（但不应导致错误结果）
;;;;           —— map-reduce.lisp（内联内核 + strides 遍历）、
;;;;              simd-matmul.lisp（SIMD/并行 GEMM）
;;;;
;;;; 核心原则：任何优化路径必须与通用路径在语义上完全等价。
;;;;   连续性只影响性能，绝不影响正确性：连续视图走内联/SIMD 快路径，
;;;;   非连续视图走 strides 遍历，两条路径的输出必须逐位相同。
;;;;   路径选择基于运行时检查（dtype/连续性/形状对齐），不基于编译期假设。

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
   #:vt-dtype
   #:vt-order
   #:vt-size
   #:vt-p
   #:vt-itemsize
   #:vt-nbytes
   #:vt-contiguous-p
   #:vt-shape-to-size
   #:vt-compute-strides
   #:vt-compute-logical-strides
   #:vt-broadcast-shapes
   #:vt-broadcast-strides
   #:vt-normalize-axis

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
   ;; SeedSequence
   #:vt-seed-sequence #:make-seed-sequence #:vt-seed-sequence-entropy
   #:seed-sequence-spawn #:seed-sequence-generate-state
   ;; Generator
   #:vt-generator #:make-generator #:vt-generator-state
   #:generator-from-seed-sequence #:spawn-generators
   ;; 作用域宏
   #:with-seed #:with-generator

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
   ;; v0.3.6：删除 #:+vt-float-nan+ / #:+vt-float-pos-inf+ / #:+vt-float-neg-inf+
   ;; 三个「默认 double」别名常量。请改用按 dtype 取值的
   ;; (vt-get-nan dtype) / (vt-get-pos-inf dtype) / (vt-get-neg-inf dtype)，
   ;; 或精确的 +vt-dfloat-*+ / +vt-sfloat-*+ 常量。

   ;; ------------------------------------------------------------------
   ;; 核心迭代与映射 (map-reduce.lisp)
   ;; ------------------------------------------------------------------
   #:vt-map
   #:vt-do-each
   #:vt-reduce
   #:vt-copy-into
   #:vt-copy

   ;; ------------------------------------------------------------------
   ;; 参数契约基础设施 (parcontract.lisp)
   ;; :out / :dtype 的统一校验、strides 驱动寻址、别名快照
   ;; ------------------------------------------------------------------
   #:vt-check-out
   #:vt-check-out-dtype-consistency
   #:vt-out-writable-p
   #:vt-out-contig-p
   #:vt-out-snapshot
   #:vt-write-1
   #:vt-reduce-dtypes
   #:vt-params-audit

   ;; ------------------------------------------------------------------
   ;; 通用辅助与宏
   ;; （vt-normalize-axis / vt-broadcast-shapes / vt-broadcast-strides /
   ;;  vt-compute-strides / vt-compute-logical-strides 均定义于 core.lisp，
   ;;  已归并到「张量结构访问器」区，避免重复导出）
   ;; ------------------------------------------------------------------
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
   #:vt-clip-tensor
   #:vt-clamp
   #:vt-copy-to!

   ;; extensions2.lisp
   #:vt-fliplr
   #:vt-flipud
   #:vt-ediff1d
   #:vt-geomspace
   #:vt-ravel-multi-index
   #:vt-tril-indices
   #:vt-triu-indices
   #:vt-vander
   #:vt-one-hot
   #:vt-standardize
   #:vt-layer-norm
   #:vt-apply-along-axis))

(in-package :clvt)

;;; 供测试与文档使用：收集所有以 `vt-` 开头的导出符号。
(defparameter *vt-fun-list* nil
  "所有以 `vt-` 开头的导出符号列表。")

(defun refresh-vt-fun-list ()
  (setf *vt-fun-list* nil)
  (do-symbols (var :clvt)
    (when (and (> (length (symbol-name var)) 2)
	       (search "vt-" (symbol-name var) :test #'equalp :end2 3))
      (push var *vt-fun-list*)))
  (setf *vt-fun-list* (nreverse *vt-fun-list*)))


(refresh-vt-fun-list)

