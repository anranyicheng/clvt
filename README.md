# clvt
clvt (common lisp vector tensor) library.
这是一个纯 common lisp (sbcl) 语言编写的张量库，是使用'智谱清言'AI(GLM5+)和'DeepSeek'(v4+pro) AI等共同编写。目标就是为 common lisp 生态构建一个简洁而强大的张量计算库。clvt 这库的核心基础是 vt-einsum，vt-map 两个函数和 def-vt-reduce 宏, 其余操作大多数都是基于这三个核心功能组合完成，易于理解，同时这个库有完美的打印输出功能。目前这个库已经实现了许多张量的基础操作，未来将进一步完善，目标是尽可能实现 numpy 众多功能, 部分函数功能向pytorch看齐。这个库的函数都是以 vt- 开头，配合 slime 一起使用非常方便，易于查看已经实现了哪些函数。目前只在sbcl上运行和测试。

This is a tensor library written purely in Common Lisp (SBCL). It was co-written using "Zhipu Qingyan" AI (GLM5+) and "DeepSeek" AI (v4+pro), among others. The goal is to build a concise yet powerful tensor computation library for the Common Lisp ecosystem. The core foundation of the clvt library is the two functions vt-einsum and vt-map, plus the def-vt-reduce macro. Most of the other operations are built by combining these three core features, making them easy to understand. The library also has perfect print output functionality. At present, the library has already implemented many basic tensor operations. It will be further improved in the future, with the goal of implementing as many NumPy features as possible and aligning some function behavior with PyTorch. All functions in this library start with vt-, which makes them very convenient to use with SLIME and makes it easy to see which functions have already been implemented. Currently, it only runs and is tested on SBCL.

clvt 举例:
``` common lisp
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
vt-truncate
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

;; 反双曲函数 (新增)
vt-asinh
vt-acosh
vt-atanh

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

;; 补充算术与数学 (新增)
vt-reciprocal
vt-negative
vt-lerp
vt-cbrt

;; 位运算 (新增)
vt-bit-and
vt-bit-ior
vt-bit-xor
vt-bit-not
vt-left-shift
vt-right-shift

;; 逐元素极值 (新增)
vt-maximum
vt-minimum
vt-fmax
vt-fmin

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

;; nan 感知统计 (新增)
vt-nansum
vt-nanmean
vt-nanstd
vt-nanvar
vt-nanmax
vt-nanmin
vt-nanargmax
vt-nanargmin
vt-nanprod
vt-nanmedian

;; 线性代数
vt-matmul
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

;; 线性代数扩展 (新增)
vt-cholesky
vt-eig
vt-pinv
vt-lstsq

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

;; 填充与插值 (新增)
vt-fill
vt-interp
vt-kron
vt-meshgrid

;; 追加、插入、删除 (新增)
vt-append
vt-insert
vt-delete

;; 随机数生成
vt-random
vt-random-uniform
vt-random-normal
vt-random-int
vt-random-integers
vt-random-seed
vt-random-choice
vt-random-permutation
vt-random-shuffle
vt-random-multinomial
;; SeedSequence
#:vt-seed-sequence 
#:make-seed-sequence 
#:vt-seed-sequence-entropy
#:seed-sequence-spawn 
#:seed-sequence-generate-state
;; Generator
#:vt-generator 
#:make-generator
#:vt-generator-state
#:generator-from-seed-sequence 
#:spawn-generators
;; 作用域宏
#:with-seed #:with-generator

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

;; 通用辅助与宏
vt-normalize-axis
vt-broadcast-shapes
vt-broadcast-strides
vt-compute-strides
vt-compute-logical-strides
with-float-safe

;; 扩展功能 (extensions.lisp)
vt-count-nonzero    ;; 统计非零元素个数 (对标 numpy.count_nonzero)
vt-count            ;; 统计等于指定值的元素个数
vt-flatnonzero      ;; 展平后返回非零元素索引 (对标 numpy.flatnonzero)
vt-moveaxis         ;; 移动轴到新位置 (对标 numpy.moveaxis)
vt-inner            ;; 内积 (对标 numpy.inner)
vt-tensordot        ;; 张量缩并 (对标 numpy.tensordot)
vt-topk             ;; 获取前 k 个最大/最小值 (对标 torch.topk)
vt-clip-tensor      ;; 支持张量作为边界的裁剪
vt-set-print-options ;; 设置打印选项
vt-get-print-options ;; 获取打印选项

```
测试在 example/example.lisp 文件中。
``` common lisp
(ql:quickload :clvt)
(in-package :clvt)
(load "~/quicklisp/local-projects/clvt/example/example.lisp")
(run-all-tests)
```

自动化测试:
```bash
# 运行所有测试 (900 个测试用例)
bash test/run-tests.sh

# 运行指定测试套件
bash test/run-tests.sh --suite run_all_tests
bash test/run-tests.sh --suite nested-test

# 运行时调用 numpy 实时对比结果 (需安装 python3 + numpy)
bash test/run-tests.sh --suite numpy-compare-test

# 列出所有测试套件
bash test/run-tests.sh --list
```

## 架构

源码按职责分层组织在 `src/` 目录下：

| 文件 | 职责 |
|------|------|
| `package.lisp` | 包定义与公共 API 导出（按功能分组） |
| `dtype.lisp` | 元素数据类型系统（单一事实来源） |
| `util.lisp` | 浮点陷阱屏蔽、关键字参数解析、NaN 感知排序 |
| `nan.lisp` | NaN / Inf 常量与判定（可移植，不依赖实现内部符号） |
| `core.lisp` | 张量结构、步长、广播、连续判定、拷贝、填充 |
| `iterator.lisp` | 统一迭代原语 |
| `map-reduce.lisp` | `vt-map` / `vt-reduce` 逐元素映射与归约核心 |
| `io.lisp` | 序列/数组互转与打印 |
| `creation.lisp` | 张量创建 |
| `manip.lisp` | 形状/视图/翻转/三角/填充 |
| `indexing.lisp` | 索引、切片、选择 |
| `join.lisp` | 连接、堆叠、追加、插入、删除 |
| `elementwise.lisp` | 逐元素算术/数学/比较/逻辑 |
| `reduce-stats.lisp` | 归约、统计、排序、NaN 感知统计 |
| `setops.lisp` | 集合操作 |
| `random.lisp` | 随机数生成 |
| `linalg.lisp` | 线性代数（含 einsum） |
| `nn.lisp` | 激活 / 损失 / softmax |
| `rotate.lisp` | 图像旋转（对标 scipy.ndimage.rotate） |
| `extensions.lisp` | 扩展功能 |

## 设计约定（三层架构语义契约）

clvt 遵循"逻辑层 / 物理层 / 执行层"三层分离架构，所有公开 API 遵守以下契约：

| 层次 | 职责 | 失败模式 |
|------|------|----------|
| 逻辑层 | shape、dtype、广播规则、归约语义 | 形状不匹配、dtype 不支持 |
| 物理层 | strides、offset、连续性、内存布局 | 越界访问、别名冲突 |
| 执行层 | 快路径 vs 通用路径、SIMD、并行调度 | 性能退化（但不应导致错误结果） |

**核心原则：任何优化路径必须与通用路径在语义上完全等价。** 连续性只影响性能，
绝不影响正确性——连续视图走内联/SIMD 快路径，非连续视图走 strides 遍历，
两条路径的输出必须逐位相同。路径选择基于运行时检查，不基于编译期假设。

### 内存模型：Strided View

张量是扁平缓冲区上的视图，由 data pointer、shape、stride、dtype（外加 device）描述。
内存是一维的；一个 (3, 4) 的张量在扁平缓冲区中存储 12 个数，stride (4, 1) 表示
沿行前进跳 4 个元素、沿列前进跳 1 个。广播的物理实现是长度为 1 的轴 stride = 0。

| 操作 | 是否拷贝 | 条件 |
|------|----------|------|
| `transpose` / `permute` | 否 | 仅交换 shape 和 stride |
| `reshape` | 否（若可） | stride 兼容则视图，否则拷贝 |
| `view` | 否 | 要求连续性，非连续时报错 |
| 基础切片 `x[1:, ::2]` | 否 | 调整 base offset 和 strides |
| 高级索引 `x[[0,2]]` | 是 | 无法表达为 stride 模式 |
| `flatten` | 是 | 总是拷贝 |
| `ravel` | 否（若可） | 尽可能返回视图 |
| `contiguous` | 可能 | 已连续则返回自身（no-op） |

### 语义保证

- **广播**：右对齐、维数不足左补 1；原地操作不允许广播改变形状；
  广播视图（dim > 1 且 stride = 0）语义上只读，写入报错。
- **类型提升**：boolean < integral < floating；浮点不降级；
  int32 与 float32 混合提升为 float64（24 位尾数规则）；整数溢出回绕；
  浮点转整数截断，NaN/Inf 转整数返回 0。
- **NaN/Inf**：算术传播、比较恒假；`maximum/minimum` NaN 传播，
  `fmax/fmin` 忽略 NaN；`sum/prod` NaN 传播，`nansum/nanprod` 跳过；
  `amax/amin` 首个 NaN 立即胜出，`argmax/argmin` 遇 NaN 立即返回该位置；
  排序 NaN 稳定排在末尾；集合语义（unique 等）NaN 视为相等；
  softmax 减最大值稳定化（全 -Inf 行 → NaN，对标 PyTorch）。
- **归约**：`axis = nil` 全局归约；`keepdims` 保留归约轴为 1；
  int32 累加器提升为 int64，float32 保持 float32。
  空归约按"是否有单位元"分类：`sum→0`、`prod→1`、`all→T`、`any→NIL`
  返回单位元；`max/min` 族无单位元，空归约返回 NaN（整数结果 dtype
  提升为 float64 承载 NaN）；`argmax/argmin` 族空归约报错。
- **别名安全**：输出 `:out` 与任一输入共享底层存储且物理区间重叠时，
  先快照输入再写入；广播视图（stride = 0）与自重叠视图只读。
  重叠检测按维扩展：dim > 1 时区间向 stride 方向扩展 (dim-1)*|stride|，
  广播维不扩展区间。
- **out 契约**：硬契约（违反必报错）——形状必须等于逻辑结果形状、
  必须可写（非广播视图）、与显式 `:dtype` 冲突报错；
  软门控（只影响选路）——连续性决定快慢路径，非连续 out 必须能被正确写入。
- **einsum 路由**：纯逐元素模式 → `vt-map`；全收缩内积 → 专用累加内核；
  批量矩阵乘法 → 分块 GEMM（SIMD + 多线程）；其余 → 通用循环。

## License
MIT
