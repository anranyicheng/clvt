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
CLVT> (dolist (fun *vt-fun-list*)
	(print fun))
VT-ONES-LIKE 
VT-LOGICAL-AND 
VT-DSPLIT 
VT-DIAGONAL 
*VT-PRINT-PRECISION* 
VT-UNIQUE 
VT-EXTRACT 
VT-INTERSECT1D 
VT-ISFINITE 
VT-CUMSUM 
VT-WHERE 
VT-ARGSORT 
*VT-PRINT-THRESHOLD* 
VT-RANDOM-INT 
VT-EMPTY-LIKE 
*VT-INDENT-STEP* 
VT-ISINF 
VT-ISNAN 
VT-QUANTILE 
VT-ENSURE-SHAPE-COMPATIBLE 
VT-MATMUL-DF 
VT-PROD 
VT-GET-CONTIGUOUS-DF-DATA 
VT-ASTYPE 
VT-UNION1D 
VT-FULL-LIKE 
VT-SETXOR1D 
VT-TRIU 
VT-LOGICAL-OR 
VT-CHOOSE 
VT-ITEMSIZE 
VT-TOLIST 
VT-TILE 
*VT-FUN-LIST* 
VT-RAVEL 
VT-BASE 
VT-REDUCE 
VT-VAR 
VT-EYE 
VT-HSPLIT 
VT-COMPUTE-LOGICAL-STRIDES 
VT-IN1D 
VT-HSTACK 
VT-MEDIAN 
VT-P 
VT-EMPTY 
VT-ANY 
VT-BROADCAST-SHAPES 
VT-GRADIENT 
VT-CONTIGUOUS-P 
VT-@ 
VT-ARRAY-SPLIT 
VT-PERCENTILE 
VT-ZEROS-LIKE 
VT-VSTACK 
VT-FULL 
VT-NBYTES 
VT-FLATTEN-TO-NESTED 
VT-DSTACK 
VT-ISCLOSE 
VT-CUMPROD 
VT-STACK 
VT-ALL 
*VT-EINSUM-PARSE-CACHE* 
VT-VSPLIT 
VT-SETDIFF1D 
VT-MESHGRID 
VT-ALLCLOSE 
VT-PTP 
VT-LINSPACE 
VT-SHAPE-TO-SIZE 
VT-LOGICAL-NOT 
VT-NONZERO 
VT-SORT 
VT-SELECT 
VT-ARGWHERE 
VT-BROADCAST-STRIDES 
VT-EXPAND-DIMS 
VT-DIAG 
VT-REPEAT 
PRINT-VT-RECURSIVE 
VT-AVERAGE 
VT-FROM-FUNCTION 
VT-SEARCHSORTED 
VT-SQUARE 
VT-LOGICAL-XOR 
VT-TRIL 
VT-HISTOGRAM 
VT-SWAPAXES 
VT-PUT 
VT-SQUEEZE 
VT-COMPUTE-STRIDES 
VT-- 
VT-ARGMAX 
VT-FLATTEN 
VT-FLOOR 
VT-SLICE 
VT-MAP 
VT->= 
VT-ACOS 
VT-OUTER 
VT-L1-NORM 
VT-COPY 
VT-EINSUM 
VT-CONCATENATE 
VT-ATAN 
VT-SQRT 
VT-SCALE 
VT-* 
VT-INV 
VT-SIN 
VT-> 
VT-RINT 
VT-TO-2D-ARRAY 
VT-REF 
VT-RANDOM 
VT-FROBENIUS-NORM 
VT-EVEN-P 
VT-TRACE 
VT-DET 
VT-DATA 
VT-TANH 
VT-POSITIVE-P 
VT-ODD-P 
VT-LOG 
VT-ELEMENT-TYPE 
VT-SWISH 
VT-DO-EACH 
VT-< 
VT-COPY-INTO 
VT-LOG-SOFTMAX 
VT-COSH 
VT-SINH 
VT-SOFTPLUS 
VT-SIGMOID 
VT-OFFSET 
VT-SOFTMAX 
VT-DATA->LIST 
VT-LOG10 
VT-LOG2 
VT-NORM 
VT-SIZE 
VT-AMAX 
VT-ASIN 
VT-/= 
VT-SIGNUM 
VT-SHAPE 
VT-<= 
VT-/ 
VT-TAN 
VT-REM 
VT-TAKE 
VT-ZERO-P 
VT-CEILING 
VT-EXP 
VT-ZEROS 
VT-MATMUL 
VT-NEGATIVE-P 
VT-COS 
VT-SPLIT 
VT-FLATTEN-SEQUENCE 
VT-ROUND 
VT-MEAN 
VT-= 
VT-+ 
VT-EXPT 
VT-ARGMIN 
VT-FROM-SEQUENCE 
VT-CONST 
VT-ORDER 
VT-HARD-SIGMOID 
VT-CONTIGUOUS 
VT-ATAN2 
VT-MEAN-SQUARED-ERROR 
VT-SOLVE 
VT-RESHAPE 
VT-GELU 
VT-BINARY-CROSS-ENTROPY 
VT-DOT 
VT-HARD-TANH 
VT-CROSS-ENTROPY 
VT-ARANGE 
VT-ONES 
VT-MOD 
VT-FROM-2D-ARRAY 
VT-AMIN 
VT-LEAKY-RELU 
VT-MISH 
VT-ABS 
VT-RANDOM-NORMAL 
VT-TRANSPOSE 
VT-STRIDES 
VT-SUM 
VT-STD 
VT-RELU 
VT-CLIP 
VT-NONZERO-P
CLVT> 
```
## License
MIT

