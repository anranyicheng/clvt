# clvt
clvt (common lisp vector tensor) library.
这是一个纯 common lisp 语言编写的张量库，是使用'智谱清言'AI(GLM5+)和'DeepSeek'(v4pro) AI共同编写，目标就是为 common lisp 生态构建一个简洁而强大的张量计算库。虽然 lisp 社区拥有 magicl 和 numcl 这两个比较流行的库，但是 magicl 缺乏对高维张量的支持以及缺少一些重要的函数，numcl 注重类型推理并且一些函数接口和CL语言标准重合。clvt 这库的核心基础是 vt-einsum，vt-map, vt-reduce 三个函数, 其余操作大多数都是基于这三个核心函数组合完成，易于理解，同时这个库有完美的打印输出功能。目前这个库已经实现了许多张量的基础操作，未来将进一步完善，目标是尽可能实现 numpy 众多功能, 部分函数功能向pytorch看齐。这个库的函数都是以 vt- 开头，配合 slime 一起使用非常方便，易于查看已经实现了哪些函数。目前在sbcl上完成了大部分的测试。

This is a tensor library written entirely in Common Lisp, collaboratively developed by the AI 'Zhipu Qingyan' (GLM5+) and 'DeepSeek' (v4pro). The goal is to build a concise yet powerful tensor computation library for the Common Lisp ecosystem. Although the Lisp community already has two relatively popular libraries, magicl and numcl, magicl lacks support for high-dimensional tensors and some important functions, while numcl emphasizes type inference and has some function interfaces that overlap with standard CL functions. The core foundation of the clvt library consists of three functions: vt-einsum, vt-map, and vt-reduce. Most other operations are built by combining these three core functions, making them easy to understand. Additionally, this library features excellent pretty-printing capabilities. Currently, the library has already implemented many basic tensor operations, and further improvements will be made in the future, aiming to implement as many NumPy features as possible, with some functions modeled after PyTorch. Functions in this library are all prefixed with vt-, which, when used with Slime, makes it very convenient to see which functions have been implemented. The majority of tests have been completed on SBCL.

clvt 举例:
``` commmon lisp
CLVT> (defparameter *m* (vt-arange 27 :start 0 :step 1 :dtype :int32))
*M*
CLVT> *m*
#<VT shape:(27) dtype:(SIGNED-BYTE 32) 
  [ 0,  1,  2, ..., 24, 25, 26]>
CLVT> (setf *m* (vt-reshape *m* '(3 3 3)))
#<VT shape:(3 3 3) dtype:(SIGNED-BYTE 32) 
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
#<VT shape:NIL dtype:(SIGNED-BYTE 32) 26>
CLVT> (vt-amax *m* :axis 0)
#<VT shape:(3 3) dtype:(SIGNED-BYTE 32) 
  [[ 18,  19,  20],
   [ 21,  22,  23],
   [ 24,  25,  26]]>
CLVT> (vt-argmin *m* :axis 0)
#<VT shape:(3 3) dtype:(SIGNED-BYTE 32) 
  [[ 0,  0,  0],
   [ 0,  0,  0],
   [ 0,  0,  0]]>
CLVT> (vt-sum *m*)
#<VT shape:NIL dtype:(SIGNED-BYTE 32) 351>
CLVT> (vt-sum *m* :axis 0)
#<VT shape:(3 3) dtype:(SIGNED-BYTE 32) 
  [[ 27,  30,  33],
   [ 36,  39,  42],
   [ 45,  48,  51]]>
CLVT> (vt-+ *m* 5)
#<VT shape:(3 3 3) dtype:(SIGNED-BYTE 64) 
  [[[  5,   6,   7],
    [  8,   9,  10],
    [ 11,  12,  13]],
   [[ 14,  15,  16],
    [ 17,  18,  19],
    [ 20,  21,  22]],
   [[ 23,  24,  25],
    [ 26,  27,  28],
    [ 29,  30,  31]]]>
CLVT> (vt-+ *m* *m*)
#<VT shape:(3 3 3) dtype:(SIGNED-BYTE 32) 
  [[[  0,   2,   4],
    [  6,   8,  10],
    [ 12,  14,  16]],
   [[ 18,  20,  22],
    [ 24,  26,  28],
    [ 30,  32,  34]],
   [[ 36,  38,  40],
    [ 42,  44,  46],
    [ 48,  50,  52]]]>
CLVT> (vt-* *m* *m*)
#<VT shape:(3 3 3) dtype:(SIGNED-BYTE 32) 
  [[[   0,    1,    4],
    [   9,   16,   25],
    [  36,   49,   64]],
   [[  81,  100,  121],
    [ 144,  169,  196],
    [ 225,  256,  289]],
   [[ 324,  361,  400],
    [ 441,  484,  529],
    [ 576,  625,  676]]]>
CLVT> (vt-* *m* 5)
#<VT shape:(3 3 3) dtype:(SIGNED-BYTE 64) 
  [[[   0,    5,   10],
    [  15,   20,   25],
    [  30,   35,   40]],
   [[  45,   50,   55],
    [  60,   65,   70],
    [  75,   80,   85]],
   [[  90,   95,  100],
    [ 105,  110,  115],
    [ 120,  125,  130]]]>
CLVT> (vt-sin *m*)
#<VT shape:(3 3 3) dtype:DOUBLE-FLOAT 
  [[[       0.0,   0.841471,   0.909297],
    [   0.14112,  -0.756802,  -0.958924],
    [ -0.279415,   0.656987,   0.989358]],
   [[  0.412118,  -0.544021,   -0.99999],
    [ -0.536573,   0.420167,   0.990607],
    [  0.650288,  -0.287903,  -0.961397]],
   [[ -0.750987,   0.149877,   0.912945],
    [  0.836656,  -0.008851,   -0.84622],
    [ -0.905578,  -0.132352,   0.762558]]]>
CLVT> (vt-slice *m* '(0))
#<VT shape:(3 3) dtype:(SIGNED-BYTE 32) 
  [[ 0,  1,  2],
   [ 3,  4,  5],
   [ 6,  7,  8]]>
CLVT> (setf (vt-slice *m* '(1))
	    (vt-slice *m* '(0)))
#<VT shape:(3 3) dtype:(SIGNED-BYTE 32) 
  [[ 0,  1,  2],
   [ 3,  4,  5],
   [ 6,  7,  8]]>
CLVT> *m*
#<VT shape:(3 3 3) dtype:(SIGNED-BYTE 32) 
  [[[  0,   1,   2],
    [  3,   4,   5],
    [  6,   7,   8]],
   [[  0,   1,   2],
    [  3,   4,   5],
    [  6,   7,   8]],
   [[ 18,  19,  20],
    [ 21,  22,  23],
    [ 24,  25,  26]]]>
CLVT> 

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
vt-to-list
vt-to-array
vt-astype

;; 形状操作与视图
vt-view
vt-reshape
vt-transpose
vt-squeeze
vt-unsqueeze
vt-expand-dims
vt-flatten
vt-ravel
vt-swapaxes
vt-rot90
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
vt-item
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
vt-qr
vt-svd
vt-matrix-rank

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

;; nan的相关
vt-float-nan
vt-float-nan-p
vt-float-nan-=
vt-float-pos-inf
vt-float-neg-inf
vt-float-pos-inf-p
vt-float-neg-inf-p
vt-float-inf-=
vt-float-nan-inf-=
+vt-float-nan+
+vt-float-pos-inf+
+vt-float-neg-inf+

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
with-float-safe


```
测试在 example/example.lisp 文件中。
``` common lisp
(load "~/quicklisp/local-projects/clvt/example/example.lisp")
(run-all-tests)
```
## License
MIT

