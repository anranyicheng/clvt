# clvt
clvt (common lisp vector tensor) library.
这是一个纯 common lisp 语言编写的张量库，是使用'智谱清言'AI为主，'DeepSeek' AI为辅助编写，目标就是为 common lisp 生态构建一个简洁而强大的张量计算库。虽然 lisp 社区拥有 magicl 和 numcl 这两个比较流行的库，但是 magicl 缺乏对高维张量的支持以及缺少一些重要的函数，numcl 注重类型推理并且一些函数接口和CL语言标准重合。clvt 这库的核心是 vt-einsum 计算引擎，以及 vt-map, vt-reduce 函数, 其余操作大多数都是基于这三个核心函数组合完成，易于理解，同时这个库有完美的打印输出功能。目前这个库已经实现了许多张量的基础操作，未来将进一步完善，目标是尽可能实现 numpy 众多功能。这个库的函数都是以 vt- 开头，配合 slime 一起使用非常方便，易于查看已经实现了哪些函数。

This is a tensor library written purely in Common Lisp, primarily developed using the 'Zhipu Qingyan' AI with assistance from the 'DeepSeek' AI. The goal is to build a concise yet powerful tensor computation library for the Common Lisp ecosystem. Although the Lisp community already has two relatively popular libraries, magicl and numcl, magicl lacks support for high-dimensional tensors and some important functions, while numcl focuses on type inference and overlaps with CL standard functions in some of its interfaces. The core of the clvt library consists of the `vt-einsum` computation engine, along with the `vt-map` and `vt-reduce` functions. Most other operations are composed from these three core functions, making the library easy to understand. Additionally, the library features excellent pretty-printing output. Currently, many basic tensor operations have been implemented, and future development will aim to further improve the library, with the goal of replicating as many of NumPy's features as possible. All functions in this library are prefixed with `vt-`, which works very conveniently with Slime, making it easy to see which functions have been implemented.

clvt 举例:
``` commmon lisp
CLVT> (defparameter *m* (vt-arange 27 :start 0 :step 1 :type 'fixnum))
*M*
CLVT> *m*
#<VT shape:(27) dtype:FIXNUM 
  [ 0,  1,  2,  3, ..., 23, 24, 25, 26]>
CLVT> (setf *m* (vt-reshape *m* '(3 3 3)))
#<VT shape:(3 3 3) dtype:FIXNUM 
  [[[  0,   1,   2],
    [  3,   4,   5],
    [  6,   7,   8]],
   [[  9,  10,  11],
    [ 12,  13,  14],
    [ 15,  16,  17]],
   [[ 18,  19,  20],
    [ 21,  22,  23],
    [ 24,  25,  26]]]>
CLVT> (vt-amax *m*)
26
CLVT> (vt-amax *m* :axis 0)
#<VT shape:(3 3) dtype:FIXNUM 
  [[ 18,  19,  20],
   [ 21,  22,  23],
   [ 24,  25,  26]]>
CLVT> (vt-argmax *m*)
26
CLVT> (vt-argmin *m* :axis 0)
#<VT shape:(3 3) dtype:FIXNUM 
  [[ 0,  0,  0],
   [ 0,  0,  0],
   [ 0,  0,  0]]>
CLVT> (vt-sum *m*)
351
CLVT> (vt-sum *m* :axis 0)
#<VT shape:(3 3) dtype:FIXNUM 
  [[ 27,  30,  33],
   [ 36,  39,  42],
   [ 45,  48,  51]]>
CLVT> (vt-+ *m* 5)
#<VT shape:(3 3 3) dtype:DOUBLE-FLOAT 
  [[[  5.0,   6.0,   7.0],
    [  8.0,   9.0,  10.0],
    [ 11.0,  12.0,  13.0]],
   [[ 14.0,  15.0,  16.0],
    [ 17.0,  18.0,  19.0],
    [ 20.0,  21.0,  22.0]],
   [[ 23.0,  24.0,  25.0],
    [ 26.0,  27.0,  28.0],
    [ 29.0,  30.0,  31.0]]]>
CLVT> (vt-+ *m* *m*)
#<VT shape:(3 3 3) dtype:DOUBLE-FLOAT 
  [[[  0.0,   2.0,   4.0],
    [  6.0,   8.0,  10.0],
    [ 12.0,  14.0,  16.0]],
   [[ 18.0,  20.0,  22.0],
    [ 24.0,  26.0,  28.0],
    [ 30.0,  32.0,  34.0]],
   [[ 36.0,  38.0,  40.0],
    [ 42.0,  44.0,  46.0],
    [ 48.0,  50.0,  52.0]]]>
CLVT> (vt-* *m* *m*)
#<VT shape:(3 3 3) dtype:DOUBLE-FLOAT 
  [[[   0.0,    1.0,    4.0],
    [   9.0,   16.0,   25.0],
    [  36.0,   49.0,   64.0]],
   [[  81.0,  100.0,  121.0],
    [ 144.0,  169.0,  196.0],
    [ 225.0,  256.0,  289.0]],
   [[ 324.0,  361.0,  400.0],
    [ 441.0,  484.0,  529.0],
    [ 576.0,  625.0,  676.0]]]>
CLVT> (vt-sin *m*)
#<VT shape:(3 3 3) dtype:DOUBLE-FLOAT 
  [[[     0.0,   0.8415,   0.9093],
    [  0.1411,  -0.7568,  -0.9589],
    [ -0.2794,    0.657,   0.9894]],
   [[  0.4121,   -0.544,     -1.0],
    [ -0.5366,   0.4202,   0.9906],
    [  0.6503,  -0.2879,  -0.9614]],
   [[  -0.751,   0.1499,   0.9129],
    [  0.8367,  -0.0089,  -0.8462],
    [ -0.9056,  -0.1324,   0.7626]]]>
CLVT> (vt-slice *m* 0)
#<VT shape:(3 3) dtype:FIXNUM 
  [[ 0,  1,  2],
   [ 3,  4,  5],
   [ 6,  7,  8]]>
CLVT> (setf (vt-slice *m* 1)
	    (vt-slice *m* 0))
#<VT shape:(3 3) dtype:FIXNUM 
  [[ 0,  1,  2],
   [ 3,  4,  5],
   [ 6,  7,  8]]>
CLVT> *m*
#<VT shape:(3 3 3) dtype:FIXNUM 
  [[[  0,   1,   2],
    [  3,   4,   5],
    [  6,   7,   8]],
   [[  0,   1,   2],
    [  3,   4,   5],
    [  6,   7,   8]],
   [[ 18,  19,  20],
    [ 21,  22,  23],
    [ 24,  25,  26]]]>

```

``` common lisp

;; 已经实现并导出的函数有
;; 张量结构访问器
vt
vt-shape
vt-strides
vt-offset
vt-data
vt-element-type
vt-order
vt-size
vt-p
vt-itemsize
vt-nbytes
vt-contiguous-p
vt-shape-to-size
vt-compute-strides
vt-compute-logical-strides

;; 张量创建
vt-zeros
vt-ones
vt-full
vt-empty
vt-zeros-like
vt-ones-like
vt-full-like
vt-empty-like
vt-const
vt-arange
vt-linspace
vt-logspace
vt-eye
vt-diag
vt-identity
vt-from-sequence
vt-from-array
vt-from-function
vt-flatten-sequence
vt-to-2d-array
vt-from-2d-array
vt-data->list
vt-tolist
vt-astype

;; 形状操作与视图
vt-reshape
vt-transpose
vt-squeeze
vt-unsqueeze
vt-expand-dims
vt-flatten
vt-ravel
vt-swapaxes
vt-narrow
vt-split
vt-vsplit
vt-hsplit
vt-dsplit
vt-stack
vt-vstack
vt-hstack
vt-dstack
vt-concatenate
vt-concat
vt-repeat
vt-tile
vt-pad
vt-broadcast-to
vt-broadcast-shapes
vt-broadcast-strides
vt-contiguous
vt-flip
vt-roll
vt-triu
vt-tril
vt-diagonal
vt-flatten-to-nested

;; 索引、切片与选择
vt-ref   ;; (setf vt-ref ...
vt-slice ;; (setf vt-slice ...
vt-take
vt-put
vt-where
vt-argwhere
vt-nonzero
vt-choose
vt-select
vt-extract
vt-searchsorted
vt-digitize
vt-bincount
vt-ensure-shape-compatible
vt-normalize-axis

;; 算术运算
vt-+
vt--
vt-*
vt-/
vt-add
vt-sub
vt-mul
vt-div
vt-scale
vt-square
vt-expt
vt-pow
vt-sqrt
vt-abs
vt-signum
vt-mod
vt-rem
vt-round
vt-floor
vt-ceiling
vt-trancate
vt-rint
vt-log
vt-log2
vt-log10
vt-exp
vt-clip

;; 三角函数与双曲函数
vt-sin
vt-cos
vt-tan
vt-asin
vt-acos
vt-atan
vt-atan2
vt-sinh
vt-cosh
vt-tanh
vt-hypot
vt-sinc
vt-deg2rad
vt-rad2deg

;; 比较与逻辑
vt-=
vt-/=
vt-<
vt-<=
vt->
vt->=
vt-positive-p
vt-negative-p
vt-zero-p
vt-nonzero-p
vt-even-p
vt-odd-p
vt-logical-and
vt-logical-or
vt-logical-not
vt-logical-xor
vt-all
vt-any
vt-isclose
vt-allclose
vt-isfinite
vt-isinf
vt-isnan

;; 归约与统计
vt-sum
vt-mean
vt-average
vt-std
vt-var
vt-amax
vt-amin
vt-argmax
vt-argmin
vt-prod
vt-cumsum
vt-cumprod
vt-median
vt-percentile
vt-quantile
vt-ptp
vt-histogram
vt-trapz
vt-gradient
vt-diff
vt-correlate
vt-convolve
vt-sort
vt-argsort
vt-maximum
vt-minimum

;; 线性代数
vt-matmul
vt-matmul-df
vt-@
vt-einsum
vt-dot
vt-outer
vt-trace
vt-norm
vt-l1-norm
vt-frobenius-norm
vt-solve
vt-inv
vt-det
vt-lu
vt-diag
vt-triu
vt-tril
vt-diagonal

;; 激活函数
vt-sigmoid
vt-relu
vt-leaky-relu
vt-swish
vt-softplus
vt-gelu
vt-mish
vt-hard-tanh
vt-hard-sigmoid

;; 损失函数与概率
vt-softmax
vt-log-softmax
vt-mean-squared-error
vt-binary-cross-entropy
vt-cross-entropy

;; 集合操作
vt-unique
vt-intersect1d
vt-union1d
vt-setdiff1d
vt-setxor1d
vt-in1d

;; 随机数生成
vt-random
vt-random-uniform
vt-random-normal
vt-random-int
vt-random-integers
vt-random-seed

;; 核心迭代与映射
vt-map
vt-do-each
vt-reduce
vt-copy-into
vt-copy
vt-get-contiguous-df-data

;; 通用辅助与宏
vt-normalize-axis
vt-broadcast-shapes
vt-broadcast-strides
vt-compute-strides
vt-compute-logical-strides
vt-ensure-shape-compatible

```

## License
MIT

