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
vt-geomspace
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
vt-fliplr
vt-flipud
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
vt-clamp
vt-vander

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

;; 标准化与差分扩展
vt-standardize
vt-ediff1d

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

;; 神经网络扩展
vt-one-hot
vt-layer-norm

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

;; 参数契约基础设施 (parcontract.lisp)
vt-check-out          ;; :out 硬契约校验（形状/dtype/可写）
vt-out-writable-p     ;; out 是否可写（非广播视图）
vt-out-contig-p       ;; out 是否连续（仅影响选路）
vt-out-snapshot       ;; 别名场景下的输入快照
vt-check-out-dtype-consistency ;; :dtype 与 :out 一致性 (H5)
vt-reduce-dtypes      ;; 归约结果 dtype 推导

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

;; 扩展功能2 (extensions2.lisp)
vt-fliplr           ;; 左右翻转 (对标 numpy.fliplr)
vt-flipud           ;; 上下翻转 (对标 numpy.flipud)
vt-geomspace        ;; 几何级数空间 (对标 numpy.geomspace)
vt-one-hot          ;; 独热编码 (对标 torch.nn.functional.one_hot)
vt-layer-norm       ;; 层归一化 (对标 torch.nn.LayerNorm)
vt-apply-along-axis ;; 沿轴应用函数 (对标 numpy.apply_along_axis)
vt-vander           ;; 范德蒙德矩阵 (对标 numpy.vander)

```
测试在 example/example.lisp 文件中。
``` common lisp
(ql:quickload :clvt)
(in-package :clvt)
(load "~/quicklisp/local-projects/clvt/example/example.lisp")
(run-all-tests)
```

自动化测试（27 个测试套件，共 1500+ 用例）:
```bash
# 运行所有测试
bash test/run-tests.sh

# 运行指定测试套件
bash test/run-tests.sh --suite run_all_tests
bash test/run-tests.sh --suite nested-test

# 运行时调用 numpy 实时对比结果 (需安装 python3 + numpy)
bash test/run-tests.sh --suite numpy-compare-test

# 列出所有测试套件
bash test/run-tests.sh --list
```

主要测试套件：

| 套件 | 用例数 | 内容 |
|------|-------:|------|
| `run_all_tests` | 143 | 基础函数测试 |
| `run_param_tests` | 155 | 3D+ 参数化测试 |
| `robustness-test` | 194 | 鲁棒性边界测试 |
| `comprehensive-test` | 119 | 综合功能测试 |
| `coverage-gap-test` | 97 | numpy/pytorch 覆盖差距测试 |
| `nan-random-test` | 89 | NaN 随机数测试 |
| `shape-degenerate-tests` | 70 | 空形状/退化形状表驱动测试 |
| `numpy-compare-test` | 69 | numpy/pytorch 实时对比测试 |
| `property-strided-tests` | 67 | 选路不变量/代数恒等式测试（strided ≡ contiguous） |
| `auto-compare-test` | 63 | JSON 自动对比测试 |
| `nested-test` | 60 | AI/ML 函数组合测试 |
| `out-contract-v2-test` | 56 | `:out` 契约严格相等测试（形状/dtype/可写/别名/非连续 out） |
| `refactor-bugfix-tests` | 53 | 重构回归测试 |
| `numpy-convention-tests` | 49 | numpy 对齐语义测试 |
| `test-ai-edge-cases` | 40 | AI 主流函数语义回归测试 |
| `extensions2-test` | 37 | 第二批扩展函数测试 |
| `nan-broadcast-test` | 31 | NaN/Inf 广播对齐 numpy 测试 |
| `out-contig-tests` | 23 | `:out` 连续/非连续写入测试 |
| `error-contract-tests` | 22 | 错误路径契约测试 |
| `test-copy-into` | 35 | `vt-copy-into` 正确性测试 |
| `test-bug0` | 37 | 已知 bug 回归测试 |
| `test-extensions` | 19 | 扩展函数回归测试 |
| `memsafety-tests` | 6 | `:out` 快路径内存安全测试 |
| `test-overlap` | 6 | 重叠拷贝回归测试 |
| `test-simd-batch-matmul` | 5 | SIMD 批量矩阵乘测试 |
| `property-test` / `simd-test` | — | 性质测试 / SIMD 路径测试 |

## 架构

源码按职责分层组织在 `src/` 目录下：

| 文件 | 职责 |
|------|------|
| `package.lisp` | 包定义与公共 API 导出（按功能分组） |
| `dtype.lisp` | 元素数据类型系统（单一事实来源） |
| `util.lisp` | 浮点陷阱屏蔽、关键字参数解析、NaN 感知排序 |
| `nan.lisp` | NaN / Inf 常量与判定（可移植，不依赖实现内部符号） |
| `core.lisp` | 张量结构、步长、广播、连续判定、拷贝、填充 |
| `parcontract.lisp` | 参数契约基础设施（`vt-check-out` 硬校验 / `vt-out-snapshot` 别名快照 / 统一 `:out` 语义） |
| `iterator.lisp` | 统一迭代原语 |
| `map-reduce.lisp` | `vt-map` / `vt-fast-map` / `vt-reduce` 逐元素映射与归约核心 |
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
| `simd-matmul.lisp` | SIMD 分块矩阵乘法快路径（2D 与批量） |
| `nn.lisp` | 激活 / 损失 / softmax |
| `rotate.lisp` | 图像旋转（对标 scipy.ndimage.rotate） |
| `extensions.lisp` | 扩展功能 |
| `extensions2.lisp` | 第二批扩展功能（fliplr/geomspace/one-hot/layer-norm 等） |

## 设计约定

clvt 遵循「逻辑层 / 物理层 / 执行层」三层分离架构。全部语义契约——三层职责与铁律、
Strided View 内存模型、dtype 与类型提升、`:out` 参数契约、NaN/Inf 语义、参数与
默认值约定、SBCL 平台陷阱、逐函数对齐要点、测试与文档约定、验收标准——
统一以**单一事实来源**为准，详见：

> **[`CONVENTIONS.md`](CONVENTIONS.md)** —— clvt 统一约定（Conventions）

该文件以 NumPy 2.3.5 实测为基准；凡冲突以 NumPy 为准，唯一例外为 `vt-arange`。
README 不再重复其内容。

## License
MIT
