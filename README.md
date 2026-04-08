# clvt
common lisp vector tensor library.
这是一个 lisp 张量库，是使用智谱清言为主，DeepSeek 为辅助编写的张量库，
目标就是为 common lisp 生态构建一个简洁而强大的张量计算库。虽然 lisp 社区拥有
magicl 和 numcl 这两个比较流行的库，但是 magicl 缺乏对高维张量的支持，
numcl 繁重的类型推理，很难理解，numcl 接口和CL语言标准重合，难以论说。
这库的核心是 einsum 计算引擎，以及 vt-map, vt-reduce, 其余操作大多数
都是基于这三个核心函数组合完成，易于理解。同时这个库有完美的打印输出功能。

Common Lisp Vector Tensor Library. This is a tensor library for Lisp, primarily developed using Zhipu Qingyan and assisted by DeepSeek. The goal is to build a concise and powerful tensor computation library for the Common Lisp ecosystem. Although the Lisp community has two popular libraries, Magicl and Numcl, Magicl lacks support for high-dimensional tensors, while Numcl has heavy type inference that makes it difficult to understand. Additionally, Numcl's interface overlaps with the CL language standard, making it hard to discuss. The core of this library is the einsum computation engine, along with vt-map and vt-reduce. Most other operations are composed based on these three core functions, making them easy to understand. At the same time, this library features excellent printing and output capabilities.

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
CLVT> (vt-slice* *m* 0)
#<VT shape:(3 3) dtype:FIXNUM 
  [[ 0,  1,  2],
   [ 3,  4,  5],
   [ 6,  7,  8]]>
CLVT> (setf (vt-slice* *m* 1)
	    (vt-slice* *m* 0))
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
VT-RANDOM-NORMAL 
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
VT-DIAGONAL 
VT-BASE 
VT-COMPUTE-LOGICAL-STRIDES 
VT-PUT 
VT-CONTIGUOUS-P 
VT-QUANTILE 
VT-LOGICAL-XOR 
VT-PTP 
VT-ISINF 
VT-TRIU 
VT-FULL-LIKE 
VT-LOGICAL-NOT 
VT-GRADIENT 
VT-EXPAND-DIMS 
VT-COSH 
VT-EXTRACT 
VT-ISNAN 
VT-HSPLIT 
VT-SWAPAXES 
VT-SELECT 
VT-ARGWHERE 
VT-PROD 
VT-ANY 
VT-HSTACK 
VT-Y 
VT-EMPTY 
VT-STACK 
VT-COMPUTE-STRIDES 
VT-PERCENTILE 
VT-HISTOGRAM 
VT-FULL 
VT-UNION1D 
VT-LINSPACE 
VT-CONCATENATE 
*VT-INDENT-STEP* 
VT-WHERE 
VT-IN1D 
VT-REDUCE 
VT-EYE 
VT-AVERAGE 
VT-LOGICAL-OR 
VT-TAKE 
*VT-FUN-LIST* 
VT-FLATTEN 
VT-UNIQUE 
VT-FLATTEN-TO-NESTED 
VT-ARRAY-SPLIT 
VT-ZEROS-LIKE 
VT-VAR 
VT-ITEMSIZE 
VT-DSTACK 
VT-DIAG 
VT-STD 
*VT-PRINT-THRESHOLD* 
VT-FROM-FUNCTION 
VT-SETDIFF1D 
VT-CUMSUM 
VT-TOLIST 
VT-ALLCLOSE 
VT-CHOOSE 
VT-TILE 
VT-SQUARE 
PRINT-VT-RECURSIVE 
VT-SETXOR1D 
VT-MEAN 
VT-ASTYPE 
VT-ISFINITE 
VT-SHAPE-TO-SIZE 
VT-@ 
VT-ALL 
VT-ARGSORT 
VT-VSTACK 
VT-NONZERO 
VT-RANDOM-INT 
VT-BROADCAST-STRIDES 
VT-CUMPROD 
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
VT-SLICE* 
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
NIL
CLVT> 
```
## License
MIT

