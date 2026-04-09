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
VT-LOGICAL-AND 
VT-BROADCAST-SHAPES 
VT-ENSURE-SHAPE-COMPATIBLE 
VT-REPEAT 
VT-MESHGRID 
VT-ONES-LIKE 
VT-DOT 
VT-RANDOM-NORMAL 
VT-HARD-TANH 
*VT-PRINT-PRECISION* 
VT-SEARCHSORTED 
VT-P 
VT-CLIP 
VT-LOG 
VT-DSPLIT 
VT-TRIL 
VT-VSPLIT 
VT-SORT 
VT-RAVEL 
VT-INTERSECT1D 
VT-NBYTES 
VT-MEDIAN 
VT-ISCLOSE 
VT-EMPTY-LIKE 
VT-TRACE 
VT-DIAGONAL 
VT-BASE 
VT-L1-NORM 
VT-COMPUTE-LOGICAL-STRIDES 
VT-PUT 
VT-CONTIGUOUS-P 
VT-QUANTILE 
VT-LOGICAL-XOR 
VT-PTP 
VT-ISINF 
VT-SOLVE 
VT-TRIU 
VT-LOG-SOFTMAX 
VT-FULL-LIKE 
VT-GRADIENT 
VT-EXPAND-DIMS 
VT-COSH 
VT-EXTRACT 
VT-ISNAN 
VT-HSPLIT 
VT-SOFTPLUS 
VT-SWAPAXES 
VT-SELECT 
VT-ARGWHERE 
VT-PROD 
VT-ANY 
VT-HSTACK 
VT-EMPTY 
VT-CROSS-ENTROPY 
VT-SOFTMAX 
VT-STACK 
VT-COMPUTE-STRIDES 
VT-PERCENTILE 
VT-HARD-SIGMOID 
VT-HISTOGRAM 
VT-FULL 
VT-UNION1D 
VT-SWISH 
VT-LINSPACE 
VT-CONCATENATE 
VT-AVERAGE 
*VT-INDENT-STEP* 
VT-NORM 
VT-WHERE 
VT-IN1D 
VT-REDUCE 
VT-EYE 
VT-BINARY-CROSS-ENTROPY 
VT-LOGICAL-OR 
VT-TAKE 
*VT-FUN-LIST* 
VT-FLATTEN 
VT-UNIQUE 
VT-FLATTEN-TO-NESTED 
VT-FROM-2D-ARRAY 
VT-ARRAY-SPLIT 
VT-ZEROS-LIKE 
VT-DET 
VT-VAR 
VT-ITEMSIZE 
VT-GELU 
VT-DSTACK 
VT-DIAG 
VT-MISH 
VT-STD 
VT-LEAKY-RELU 
*VT-PRINT-THRESHOLD* 
VT-FROM-FUNCTION 
VT-SETDIFF1D 
VT-CUMSUM 
VT-MEAN-SQUARED-ERROR 
VT-TO-2D-ARRAY 
VT-TOLIST 
VT-ALLCLOSE 
VT-CHOOSE 
VT-TILE 
VT-SQUARE 
PRINT-VT-RECURSIVE 
VT-SETXOR1D 
VT-MEAN 
VT-ASTYPE 
*VT-EINSUM-PARSE-CACHE* 
VT-ISFINITE 
VT-INV 
VT-SHAPE-TO-SIZE 
VT-@ 
VT-LOGICAL-NOT 
VT-ALL 
VT-ARGSORT 
VT-SIGMOID 
VT-VSTACK 
VT-NONZERO 
VT-RANDOM-INT 
VT-BROADCAST-STRIDES 
VT-FROBENIUS-NORM 
VT-OUTER 
VT-CUMPROD 
VT-RELU 
VT-SQUEEZE 
VT-SIN 
VT-+ 
VT-ZEROS 
VT-DO-EACH 
VT-ARGMAX 
VT-COPY-INTO 
VT-SQRT 
VT-STRIDES 
VT-NEGATIVE-P 
VT-ORDER 
VT-CEILING 
VT-SCALE 
VT-DATA 
VT->= 
VT-NONZERO-P 
VT-CONST 
VT-EXP 
VT-COPY 
VT-RINT 
VT-ELEMENT-TYPE 
VT-REM 
VT-SUM 
VT-TRANSPOSE 
VT-ZERO-P 
VT-CONTIGUOUS 
VT-COS 
VT-RESHAPE 
VT-REF 
VT-SLICE
VT-ONES 
VT-ARANGE 
VT-ACOS 
VT-ATAN 
VT-<= 
VT-DATA->LIST 
VT-POSITIVE-P 
VT-FLATTEN-SEQUENCE 
VT-MAP 
VT-ARGMIN 
VT-SPLIT 
VT-OFFSET 
VT-ODD-P 
VT-ABS 
VT-FROM-SEQUENCE 
VT-ASIN 
VT-SIZE 
VT-LOG2 
VT-MOD 
VT-EVEN-P 
VT-AMIN 
VT-LOG10 
VT-EINSUM 
VT-AMAX 
VT-ROUND 
VT-FLOOR 
VT-EXPT 
VT-SINH 
VT-> 
VT-< 
VT-SHAPE 
VT-/= 
VT-TAN 
VT-SIGNUM 
VT-= 
VT-LOG-CLEAN 
VT-RANDOM 
VT-TANH 
VT-ATAN2 
VT-MATMUL 
VT-* 
VT-CONSH 
VT-/ 
VT--
CLVT> 
```
## License
MIT

