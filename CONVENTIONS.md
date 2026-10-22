# clvt 统一约定（Conventions）

> **版本**：对应 clvt v0.4.0（本文件即重构基准）
> **基准对象**：NumPy **2.3.5**（本机实测，非文档推测）
> **裁决原则**：凡与 NumPy 冲突者，**一律以 NumPy 为准**。
> **唯一例外**：`vt-arange`（见 [§9.1](#91-vt-arange唯一例外)）。

---

## 0. 本文件的地位

本文件是 clvt 的**唯一语义契约来源**。`README.md`、`REFACTOR-NOTES.md`、`TEST-PLAN.md`、`CHANGELOG.md` 中的设计描述如与本文件冲突，以本文件为准；本文件与 NumPy 冲突，以 NumPy 为准。

每条约定标注三列：

| 列 | 含义 |
|---|---|
| **NumPy 行为** | 本机 NumPy 2.3.5 实测结果 |
| **clvt 现状** | 重构前 clvt v0.3.6 的行为 |
| **裁决** | 目标行为（`✅ 已对齐` / `⚠️ 需重构`） |

---

## 1. 三层架构（不可动摇的骨架）

任何函数都必须能拆成三层，且**优化路径与通用路径语义完全等价**：

| 层 | 内容 | 不变量 |
|---|---|---|
| **逻辑层** | shape / dtype / 广播 / 归约语义 | 决定"结果应该是什么" |
| **物理层** | strides / offset / 连续性 / 别名 | 决定"内存怎么放"，**只影响性能，绝不影响正确性** |
| **执行层** | 快路径（SIMD/并行/连续特化） vs 通用路径（stride 驱动） | 二者结果必须**逐位可比**（浮点按确定性归约顺序） |

> **铁律 1**：连续性只影响**选路**，不影响**结果**。任何"因为非连续所以结果不同"都是 bug。
> **铁律 2**：快路径失败必须**回退**到通用路径，而非报错或产生近似结果。

---

## 2. 内存模型：Strided View

张量 = 数据指针 + 形状 + 步长 + 偏移 + dtype。

```
vt = { data, shape, strides, offset, dtype }
```

### 2.1 广播约定（与 NumPy 完全一致）

| 规则 | NumPy 行为 | clvt 现状 | 裁决 |
|---|---|---|---|
| 从**末轴**对齐，逐轴匹配 | ✅ | ✅ | ✅ 已对齐 |
| 两轴维度相等 → 保留 | ✅ | ✅ | ✅ 已对齐 |
| 一轴为 1，另一轴为 N → 取 N | ✅ | ✅ | ✅ 已对齐 |
| 一轴缺失（秩不等）→ 视为 1 | ✅ | ✅ | ✅ 已对齐 |
| 双方均非 1 且不等 → 报错 | `ValueError` | 报错 | ✅ 已对齐 |
| **0 长度轴 vs 1 长度轴** → 结果该轴为 **0** | `(0,3)+(1,3)→(0,3)` | `(0,3)+(1,3)→(0,3)` ✅ 实测 | ✅ 已对齐 |
| **0 长度轴 vs 非 1 非 0 轴** → 报错 | `(3,0)+(1,4)` → ValueError | 报错，信息可读 ✅ 实测 | ✅ 已对齐 |

> **实现约定**：广播实现为**长度为 1 的轴 stride 置 0**（虚拟重复读）。该轴**必须只读**——若 `:out` 落在 stride=0 的轴上，属于写入歧义，必须报错（见 [§4.3](#43-别名与重叠)）。

### 2.2 连续性

| 术语 | 定义 |
|---|---|
| C 连续 | strides 从末轴起严格递减且乘积等于 shape |
| F 连续 | 转置意义上的连续（本库不单独建模，靠 strides 自然表达） |
| 连续张量 | `vt-contiguous-p` 为真 |
| 物理连续化 | `vt-contiguous`：非连续时复制出连续副本，连续时**原样返回**（零拷贝） |

`vt-contiguous` 的零拷贝语义是性能关键，重构中**必须保留**。

---

## 3. dtype 系统

### 3.1 逻辑 dtype（8 种）与物理存储（4 种）

| 类别 | dtype |
|---|---|
| 逻辑 dtype | `:float64 :float32 :int64 :int32 :int16 :int8 :uint8 :uint16` |
| 物理存储 | `:float64 :float32 :int64 :int32`（其余逻辑 dtype 以窄类型物理存储） |
| 布尔（内部） | 逻辑 `:int8` 承载，比较运算**逻辑上**返回布尔语义 |

> **约定**：本库不引入独立的 `:bool` dtype（NumPy 有 `bool_`）。比较/逻辑运算返回 `:int8`，取值 0/1，**语义上等价于 NumPy bool**。文档必须写明这一点，避免用户误当整数参与算术。

### 3.1.1 命名约定（**已确认的 API 现状**）

| 族 | clvt 实际命名 | NumPy 对应 |
|---|---|---|
| 反三角 | `vt-asin` / `vt-acos` / `vt-atan` / `vt-atan2` | `arcsin` / `arccos` / `arctan` / `arctan2` |
| 反双曲 | `vt-asinh` / `vt-acosh` / `vt-atanh` | `arcsinh` / `arccosh` / `arctanh` |

> **裁决**：clvt 用 `asin` 而非 `arcsin`。数学习惯上 `asin`/`arcsin` 等价，**不构成语义冲突**，保留现有命名，但 docstring 必须写明"对标 `np.arcsin`"以便检索。
> **注意**：实测 `vt-arcsin` / `vt-nan-to-num` **不存在**（未导出、未定义），文档与测试中**不得**引用这两个名字。

### 3.2 类型提升表（**逐格实测，共 64 格**）

NumPy 2.3.5 实测结果：

|  | int8 | int16 | int32 | int64 | uint8 | uint16 | float32 | float64 |
|---|---|---|---|---|---|---|---|---|
| **int8** | int8 | int16 | int32 | int64 | int16 | int32 | float32 | float64 |
| **int16** | int16 | int16 | int32 | int64 | int16 | int32 | float32 | float64 |
| **int32** | int32 | int32 | int32 | int64 | int32 | int32 | **float64** | float64 |
| **int64** | int64 | int64 | int64 | int64 | int64 | int64 | **float64** | float64 |
| **uint8** | int16 | int16 | int32 | int64 | uint8 | uint16 | float32 | float64 |
| **uint16** | int32 | int32 | int32 | int64 | uint16 | uint16 | float32 | float64 |
| **float32** | float32 | float32 | **float64** | **float64** | float32 | float32 | float32 | float64 |
| **float64** | float64 | float64 | float64 | float64 | float64 | float64 | float64 | float64 |

**clvt v0.3.6 现状（实测）**：

```
INT8  : INT8  INT16 INT32 INT64 INT16 INT32 FLOAT32 FLOAT64
INT16 : INT16 INT16 INT32 INT64 INT16 INT32 FLOAT32 FLOAT64
INT32 : INT32 INT32 INT32 INT64 INT32 INT32 FLOAT64 FLOAT64
INT64 : INT64 INT64 INT64 INT64 INT64 INT64 FLOAT64 FLOAT64
UINT8 : INT16 INT16 INT32 INT64 UINT8 UINT16 FLOAT32 FLOAT64
UINT16: INT32 INT32 INT32 INT64 UINT16 UINT16 FLOAT32 FLOAT64
FLOAT32:FLOAT32 FLOAT32 FLOAT64 FLOAT64 FLOAT32 FLOAT32 FLOAT32 FLOAT64
FLOAT64:FLOAT64 FLOAT64 FLOAT64 FLOAT64 FLOAT64 FLOAT64 FLOAT64 FLOAT64
```

**裁决：✅ 已对齐。** 64 格与 NumPy **完全一致**（基于"24 位尾数规则"：float32 能精确表示所有 int16 值，但不能精确表示 int32/int64，故 int32+float32→float64）。

### 3.3 提升规则的形式化描述（重构时保持）

1. **类别序**：`boolean < integral < floating`
2. **同类**：取宽度较大者；宽度相同保留（`uint8 ≠ int8`，取更宽的 int16）
3. **有符号 + 无符号整数**：取能容纳两者的类型
   - `uint8 + int8 → int16`、`uint8 + int16 → int16`、`uint16 + int8 → int32`
4. **整数 + 浮点**：如果整数**位宽 ≤ 浮点尾数位宽**，保持浮点；否则升到更高浮点
   - float32 尾数 24 位 → int8(8)/int16(16)/uint8/uint16 保持 float32；int32(32)/int64(64) → float64
5. **浮点不降级**：`float64` 永远是最高。
6. **归约默认 dtype**（NumPy）：
   - `sum/prod`：整数提升到 **int64**（`np.sum(int8) → int64`），浮点保持
   - `mean/var/std/average`：**一律 float64**（整数输入）
   - `maximum/minimum` 不做 dtype 提升以外的特化

> **重构要点**：以上 6 条必须以**表驱动 + 单一函数**实现，不得散落在各函数里。

### 3.4 神经网络的浮点语义（`nn.lisp` 专项约定）

神经网络层的输出**在数学上必然是实数**，因此 `nn.lisp` 中的激活函数与损失函数
**不得**走 §3.3 的通用提升规则（那会让「整数进 → 整数出」，把浮点结果静默截断）。

**统一规则（所有 nn 激活/损失函数必须遵守）**：

| 输入 dtype | 缺省输出 dtype | 说明 |
|---|---|---|
| `:float32` | `:float32` | 保持精度，不走 double |
| 其余（含全部整数、`:float64`） | `:float64` | 整数一律提升到 `:float64` 计算 |

> **根因（教训）**：通用路径把 lambda 直接交给 `vt-map` 且 `dtype=nil` 时，
> `vt-map` 走 `vt-promote-type` → **整数进整数出**，lambda 内算出的浮点值被写回
> 整数存储而**静默截断**。实测：`softplus([0])→0`（应 `0.693…`）、
> `gelu([1,2,3])→(0 1 2)`（应 `(0.841 1.955 2.996)`）、
> `leaky-relu([-2,3,-4])→(0 3 0)`（应 `(-0.02 3.0 -0.04)`）。
> 这是 P1-1（elementwise 数学函数整数精度）在 `nn.lisp` 的同类漏网。

**实现要求**：

1. 每个 nn 函数**必须**先经 `%nn-float-dtype` 推导结果 dtype
   （接收输入张量或标量 + 显式 `dtype` 覆盖），再用 `%nn-as-float` 把输入规范到该 dtype；
2. **不得**依赖 `vt-map` 的 `dtype=nil` 缺省推导；
3. 显式传 `:dtype` 时以显式值为准（仍须是浮点 dtype）；
4. `:out` 契约不变——若函数参数含浮点输出（如 `vt-leaky-relu` 的 `alpha`），
   `:out` 的 dtype 必须与推导结果 dtype 一致，否则报错。

**已核对符合本约定的函数**（守护断言见 `test/extensions3-test.lisp`）：
`vt-sigmoid` / `vt-relu` / `vt-leaky-relu` / `vt-swish` / `vt-softplus` / `vt-gelu` /
`vt-mish` / `vt-hard-tanh` / `vt-hard-sigmoid` / `vt-binary-cross-entropy` /
`vt-cross-entropy` / `vt-cross-entropy-logits`。

---

### 3.5 两种交叉熵语义（**不可混用**）

本库有两个交叉熵函数，语义**根本不同**，文档与调用方必须区分：

| 函数 | 输入 | 内部处理 | 对标 |
|---|---|---|---|
| `vt-cross-entropy` | **概率** `y-true`（one-hot）+ `y-pred`（通常 `vt-softmax` 之后） | `-Σ y·log(p)`，对 p 做 `[eps, 1-eps]` 裁剪 | 无（库内约定的概率式） |
| `vt-cross-entropy-logits` | **raw logits** + **整数类别索引** | log-softmax（减轴最大值 + log-sum-exp），**不裁剪** | `torch.nn.CrossEntropyLoss` |

> **⚠️ 易错点**：把未归一化的 logits 传给 `vt-cross-entropy` 会因形状不可广播而报错
> （`(2 3)` 与 `(2)` 不兼容），或对负 logits 做出无意义的 `[eps,1-eps]` 裁剪。
> 需要 torch 语义时**必须**用 `vt-cross-entropy-logits`。

**`vt-cross-entropy-logits` 契约**：

| 参数 | 说明 |
|---|---|
| `logits` | 未归一化得分，沿 `axis` 为类别维 |
| `labels` | **整数**类别索引，形状 = logits 去掉类别轴（`(N,C)`+`(N,)`；任意秩 `(N1,…,C)`+`(N1,…)`；无批次 `(C,)`+标量/长度 1） |
| `axis` | 类别轴，缺省 `-1` |
| `reduction` | `:mean`（缺省）/ `:sum` / `:none` |

- 浮点 `labels` **显式报错**（对标 torch 的 int64 labels）；
- 类别索引越界（含负数）**显式报错**，不静默取值；
- 数值稳定：`logits=(1e4,0,0)` 仍精确得 `0.0d0`；
- dtype 缺省按 logits 提升（float32→float32，其余→float64），与 §3.4 一致。

示例（与 `torch.nn.CrossEntropyLoss` 逐位一致）：

```lisp
(vt-cross-entropy-logits (vt-asarray '((1.0d0 2.0d0 3.0d0) (2.0d0 0.0d0 1.0d0)))
                         (vt-asarray '(0 1) :dtype :int64))
;; => 2.4076059644443806d0
```

---

## 4. `:out` 参数契约（**任务3 核心**）

### 4.1 numpy 实测行为

| 场景 | NumPy 行为 |
|---|---|
| 形状不匹配 | `ValueError: operands could not be broadcast together with shapes ...` |
| 只读 out | `ValueError: output array is read-only` |
| dtype 不同但可安全转换（float64 → float32） | **允许**，结果 cast 后写入 |
| dtype 不安全转换（float64 → int32） | `UFuncTypeError`（casting rule `same_kind`） |
| **非连续 out**（形状正确） | **允许**，按 out 的实际 strides 写入 |
| out 与输入**完全别名** | 允许，语义等价 `x += x` |

### 4.2 clvt 契约（v0.4.0 目标）

分为 **硬契约**（违反必报错）与 **软门控**（只影响选路）：

#### 硬契约 —— 必须在**任何计算发生之前**校验完毕

| # | 检查 | 错误信息要求 |
|---|---|---|
| H1 | `:out` 必须是 `vt` 对象 | 明确类型 |
| H2 | `(vt-shape out)` 必须等于**逻辑结果形状**（不可广播放宽） | 同时打印两者形状 |
| H3 | `(vt-dtype out)` 必须等于**结果 dtype** | 同时打印两者 dtype |
| H4 | `:out` 必须可写（不得是 `vt-broadcast-to` 产生的 stride=0 视图，除非该轴长度为 1） | 明确指出冲突轴 |
| H5 | 若同时给 `:dtype` 与 `:out`，二者必须一致 | 打印两者 |
| H6 | `:out` 不得与需要写入的自身别名冲突到无法定义结果的程度（见 §4.3） | 见下 |

> **H3 说明**（与 NumPy 的唯一口子）：NumPy 允许 float64→float32 的 out。clvt **不放开**，理由：clvt 无 `casting=` 参数，放开后无法表达"拒绝 float64→int32"这一半；且放开会与"精度解耦"（§4.4）语义纠缠。**要求 `vt-dtype out` 精确等于结果 dtype**，这比 NumPy 更严格，属"更安全的子集"，不构成冲突。

> **函数级附加约束**：个别函数可在 H1–H6 之上追加更严的形状要求，函数自身的契约更高。例：`vt-det` 要求 `:out` **必须是 0 维张量**，违反即报错。

#### 软门控 —— 只影响选路，不影响结果

| 门控 | 行为 |
|---|---|
| `vt-contiguous-p out` | 为真 → 允许 SIMD/并行快路径；为假 → 走通用 stride 路径 |
| `out` 与输入是否同一 `vt-data` | 为真 → 先 `vt-copy` 快照输入再算；为假 → 直接算 |

### 4.3 别名与重叠

**重叠定义**：两个视图的 `data` 相同，且存在一对逻辑索引映射到**同一物理位置**。

- 检测函数：`%vt-views-overlap-p`（现有实现，保留并加强：必须处理负 stride 与 offset 偏移）
- **策略**：一旦检测到输出与任一输入重叠，**先对重叠的输入做快照 `vt-copy`**，再执行计算。
- 这是**唯一**能保证 `np.add(z, z, out=z[::-1])` 类语义正确的做法（NumPy 实测输出 `[8,6,4,2]`，即先读全量再写）。

> **注意**：快路径中的别名处理必须**与通用路径一致**。`%vt-map-run` 的 `same-k` 分支、`vt-fast-map`、SIMD matmul 的 `a-c`/`b-c` 分支都必须执行同一套快照规则。当前 `simd-matmul.lisp` 只比较 `(eq (vt-data out) (vt-data a-c1))`，**只覆盖完全同一 data 指针的情况，漏掉"同 data 但 offset 不同"的重叠**——这是任务3 必须修的缺陷。

### 4.4 精度解耦（v0.3.6 引入，保留并推广）

对**归约类 / 统计类**函数（`sum/mean/var/std/average/norm` 等）：

```
1. compute-dtype : 按输入提升规则计算（如 int32 → int64；int → float64）
2. exec-dtype    : 实际累加用的 dtype（通常 = compute-dtype 或 float64）
3. write-out     : 最后一步 cast 写入 out（out.dtype 必须 == compute-dtype）
```

**约定**：累加**绝不在低精度下进行**（如 float32 累加 float32 输入时是否升 float64，取决于 NumPy：NumPy 的 `sum` 对 float32 保持 float32；本库**跟随 NumPy**，即 float32 输入 → float32 输出）。

> 实测：`np.sum(np.zeros(3,dtype=np.float32)).dtype == float32`。故本库 `vt-sum` 对 float32 保持 float32，**不得**升到 float64（除非显式 `:dtype :float64`）。

### 4.5 零尺寸与空结果

| 场景 | 约定 |
|---|---|
| `out` 尺寸为 0（任一轴为 0） | 立即返回 `out`，不做任何写操作（避免空循环歧义） |
| 结果为 0 尺寸但 out 非 0 尺寸 | H2 已拦截（形状不匹配） |
| 结果为 0 尺寸且未给 out | 返回新建 0 尺寸张量 |

### 4.6 计算路径内部的正确性（**任务3 的关键要求**）

> 用户原话：*"注意不是简单的在函数开头检查抛错误就完了，而是要在计算过程中确保正确实现"*

**这意味着**：H1–H6 前置校验只是**入场券**。真正的正确性由以下机制在**计算路径内部**保证：

#### 机制 A：ptr 步进统一由 strides 驱动

通用路径必须使用如下形式（**不允许**假定 `stride == itemsize`）：

```lisp
;; 正确：ptr 沿轴步进 = stride[axis]
(incf ptr (aref strides-vec axis))
;; 错误：假定连续
(incf ptr 1)
```

#### 机制 B：`out.offset` 必须参与寻址

```lisp
;; 正确：写入位置 = out.offset + Σ idx_i * out.strides_i
(setf (aref (vt-data out) (+ (vt-offset out) (* i (first (vt-strides out))))) v)
```

当前 `einsum-execute`（linalg.lisp:614–631）已正确使用 out 的真实 strides，**这是正面样板**，应抽象为公共原语供所有函数复用。

#### 机制 C：写入前不做"连续化假设"的取巧

不允许出现如下模式：
```lisp
;; 错误：先假设 out 连续，最后再修
(when (vt-contiguous-p out) ...)  ; 非连续分支被静默忽略或写得不同
```
必须保证**两条路径产出逐位相同的结果**，并为此写测试（见 [§10](#10-测试约定)）。

#### 机制 D：把"out 正确性"下沉为库级原语

重构新增：

```lisp
(vt-out-view shape dtype out &key op-name)
;; 1. 逐条执行 H1–H5 硬校验（op-name 用于错误信息）
;; 2. 返回 out 本身（保证 out.offset/strides 原样）
;; 3. 不做任何连续化，不复制
```

以及

```lisp
(vt-out-snapshot out inputs)
;; 检测 out 与 inputs 的重叠；对重叠的输入做 vt-copy 快照；返回新的 inputs 列表
```

**所有**带 `:out` 的函数必须经由这两个原语，杜绝各函数自行拼装检查逻辑（MECE：一个检查点，一处修改）。

#### 机制 E：einsum 路由

`vt-einsum` 按收缩模式选路，**所有路径结果必须逐位一致**（铁律 1、2 的实例）：

| 模式 | 路由 |
|---|---|
| 纯逐元素 | `vt-map` |
| 全收缩内积 | 专用累加内核 |
| 批量矩阵乘法 | 分块 GEMM（SIMD + 多线程） |
| 其余 | 通用循环 |

---

## 5. NaN / Inf 约定（**任务4 核心**）
```lisp
+vt-dfloat-nan+ +vt-dfloat-pos-inf+ +vt-dfloat-neg-inf+
+vt-sfloat-nan+ +vt-sfloat-pos-inf+ +vt-sfloat-neg-inf+
```

**迁移清单**（重构时必须全部替换）：
- `map-reduce.lisp` 的 `get-reduction-identity` 中使用 `+vt-dfloat-neg-inf+` 等 → 按目标 dtype 取
- `setops.lisp` 使用 `vt-float-nan-inf-=`（函数形式，**无需改**，仅其内部实现要改用 dtype 入口）
- `package.lisp` 导出清单（354–368 行）删除三个符号
- `README.md` / `CHANGELOG.md` 相关描述同步删除

> **验证方法**：重构后 `(grep +vt-float- (excluding dfloat/sfloat))` 必须为空。

### 5.2 算术传播

| 运算 | NumPy 行为 | 裁决 |
|---|---|---|
| 任何含 NaN 的算术 | 结果 NaN | ✅ |
| `inf + inf → inf`，`inf - inf → nan` | ✅ | ✅ |
| `inf * 0 → nan` | ✅ | ✅ |
| `1/0`（浮点） | `inf` + `RuntimeWarning` | ✅（本库不产生 warning） |

### 5.3 比较

| 运算 | NumPy | 裁决 |
|---|---|---|
| `nan == nan` | `False` | ✅ |
| `nan < x` / `nan > x` | `False` | ✅ |
| `array_equal([nan],[nan])` | `False` | ✅ |
| `isclose([nan],[nan])` | `False` | ✅ |
| `allclose([nan],[nan])` | `False` | ✅ |

> **注意**：`vt-isclose` 必须显式判断"任一为 NaN → not close"，否则 `abs(nan-nan) < atol` 为 False 恰好得到正确结果，但 `equal_nan=True` 场景需要特判。

### 5.4 极值与归约

| 函数 | NumPy 行为（实测） | 裁决 |
|---|---|---|
| `maximum/minimum` | **传播 NaN**（`maximum([nan,1],[2,3]) → [nan,3]`） | ✅ |
| `fmax/fmin` | **忽略 NaN**（`fmax([nan,1],[2,3]) → [2,3]`） | ✅ |
| `amax/amin` | **首个 NaN 胜出**（`amin([nan,1,-5]) → nan`；`amin([1,nan,-5]) → nan`） | ✅ |
| `nansum/nanprod` | 跳过 NaN | ✅ |
| `nanmax/nanmin` | 跳过 NaN | ✅ |
| `sum/prod` | 传播 | ✅ |
| `mean/var/std` | 传播 | ✅ |
| `argmax/argmin` | NaN 行为：返回**第一个 NaN 的下标**（因比较恒假） | ⚠️ 需验证并对齐 |
| `nanargmax/nanargmin` | 跳过 NaN | ✅ |

> **`amax` 的"首个 NaN 胜出"实现要点**：不能写成 `if (or (> v acc) (nan-p v))`（会把后续 NaN 也选中）。正确写法是 `if (or (> v acc) (and (not (> acc v)) ...))` 或用 NaN 检测短路：
> ```
> acc = op(acc, v) 其中 op 定义为:
>   (if (nan-p acc) acc       ; acc 已 NaN 则保持（首个 NaN 永不被打败）
>       (if (nan-p v) v acc-compare))
> ```
> 这**必须**在通用路径与快路径中**一致**实现。

### 5.5 零除与取整

| 运算 | NumPy 行为（实测） | 裁决 |
|---|---|---|
| `floor_divide(int, 0)` | `0`（不报错！） | ✅ 已对齐（`vt-div`） |
| `mod(int, 0)` | `0`（dtype 保持整型） | ✅ 已对齐 |
| `mod(float, 0)` | `nan` | ✅ |
| `floor_divide(float, 0)` | `inf` / `nan`（IEEE） | ✅ |
| `mod(-7, 3)` | `2`（符号跟随除数） | ✅ |
| `floor_divide(-7, 3)` | `-3` | ✅ |
| `nan → int` cast | `INT_MIN` 或 `0`（实现定义，本库已约定 **0**） | 保留本库约定（NumPy 未定义，属自由区） |

> **注意**：clvt 的对应关系——`vt-div` 对标 `np.floor_divide`（整数截断除法，
> 零除返回 `0`）；`vt-/` 对标 `np.true_divide`（整型提升浮点，零除按 IEEE 得
> ±Inf/NaN）。`vt-mod`（余数符号跟随除数）对标 `np.remainder`；`vt-rem`
> （余数符号跟随被除数）对标 `np.fmod`。四者整数零除均返回 `0`、**dtype 保持
> 整型**（不再提升 float64），与 numpy 实测一致。

**奇偶判定**：`vt-even-p` / `vt-odd-p` 对**非有限值**（NaN / ±Inf）一律判定为**不成立**（返回 `0`），不报错、不进入取整路径（避免 §7.1 的 SBCL 浮点陷阱）。

### 5.6 排序

| 运算 | NumPy（实测） | 裁决 |
|---|---|---|
| `sort([3,nan,1])` | `[1,3,nan]`（NaN 在**末尾**） | ✅ |
| `argsort([3,nan,1])` | `[2,0,1]` | ✅ |
| `median` 含 NaN | 传播 → NaN | ✅ |
| `nanmedian` | 跳过 NaN | ✅ |

### 5.7 NaN 与广播的交互（任务4 明确要求）

- `:out` 为 stride=0 的广播视图 → H4 报错（不可写）
- **输入**为 stride=0 的广播视图 + 含 NaN → 正常传播（遍历时多次读到同一 NaN）
- `vt-full(shape, +vt-dfloat-nan+)` 后参与运算 → 全 NaN 传播
- guard：`vt-astype` 把 NaN 转整数时**不得**触发 FLOATING-POINT-INVALID-OPERATION（SBCL 陷阱，见 §7.2）

---

## 6. 函数签名与参数约定（**任务3 的另一半**）

### 6.1 统一参数形态

| 参数类别 | 约定 |
|---|---|
| **必需参数** | 张量输入，位置参数，不限个数（如 `vt-+`）/ 固定个数（如 `vt-matmul` 的 a b） |
| **`&key dtype`** | 缺省 `nil` 表示"按输入提升"；显式指定则**先提升计算、再 cast** |
| **`&key out`** | 缺省 `nil`；非 nil 时执行 §4.2 硬契约 |
| **`&key axis / keepdims`** | 见 §6.3 |
| **`&rest / &optional`** | 仅允许在明确语义下使用（`vt-arange` 见 §9.1） |

### 6.2 参数校验统一由 `parse-vt-op-args` 承担

现有 `util.lisp` 的 `parse-vt-op-args` 已做到：
- 检测未知关键字
- 检测重复关键字
- 检测缺值关键字

**重构要求**：
1. `parse-vt-op-args` 必须能同时处理"张量 + `:dtype` + `:out` **交错给出**"和"**全 keyed**"两种调用风格（NumPy 风格与 Lisp 风格）。
2. 每个公开函数**必须**显式声明 `&key`，**不得**用 `&rest` 收所有参数后自行解析（除 varargs 运算如 `vt-+`）。
3. 校验失败的错误信息格式统一：
   ```
   vt-<func>: <参数名> <问题描述>，收到 <实际值>
   ```

### 6.3 `axis` / `keepdims` 约定

| 形式 | 语义 |
|---|---|
| `axis nil` | 归约**全部**轴 → 0 维结果 |
| `axis 2` | 单个轴 |
| `axis '(0 1)` | 多轴同时归约 |
| `axis -1` | 负轴，等价 `rank-1` |
| `axis '(0 -1)` | 混合负轴，每个独立规范化 |
| 轴越界 | 报错（不得静默忽略） |
| 轴重复 | NumPy 报错 `duplicate value in 'axis'` → 对齐报错 |
| `keepdims t` | 被归约的轴保留为 **1** |
| `keepdims nil`（默认） | 被归约的轴**移除** |

### 6.4 默认值一览（必须与 NumPy 一致）

| 函数族 | 参数 | NumPy 默认 | clvt 目标 |
|---|---|---|---|
| `reshape` | 新形状 | — | 支持 `-1`（至多一个） |
| `sum` | axis | `None` | `nil` |
| `sum` | keepdims | `False` | `nil` |
| `linspace` | endpoint | `True` | `t` |
| `sort` | 排序方向 | 升序 | 升序（NaN 末尾） |
| `argsort` | kind/stable | `'quicksort'` | 稳定排序（更安全，属"更强保证"） |
| `var/std` | ddof | `0` | `0`（需新增 `ddof` 参数） |
| `percentile` | 插值法 | `'linear'` | `:linear` |
| `clip` | a_min/a_max | 必给至少一个 | 至少给一个 |
| `concatenate` | axis | `0` | `0` |
| `stack` | axis | `0` | `0` |
| `take` | axis | `None`（展平） | `nil`（展平） |
| `put` | mode | `'raise'` | `:raise` |
| `roll` | axis | `None`（展平） | `nil` |
| `repeat` | axis | `None`（展平） | `nil` |
| `vt-arange` | — | 见 §9.1 | 见 §9.1 |

> **`argsort` 稳定**：NumPy 默认 quicksort **不保证稳定**。clvt 用稳定排序是"更强的保证"，**不构成冲突**（不会产生 NumPy 会产生的不同结果，只会更确定）。文档必须写明。

### 6.5 返回值的形状约定

| 场景 | NumPy | clvt 目标 |
|---|---|---|
| 全归约（axis=None） | 返回 **0 维 array**（`np.sum(np.zeros(3))` → 0-d array） | 返回 **0 维 vt**（非 Lisp 标量） |
| 1d @ 1d | 0 维 array | 0 维 vt |
| `vt-ref` 索引 0 维 | — | 返回 **Lisp 标量**（便利性，文档写明） |

> clvt 约定：**张量函数一律返回 vt**；只有显式取值函数（`vt-item`、`vt-ref` 全整数索引）返回 Lisp 标量。这一条**必须**在文档中突出，避免用户混用。

---

## 7. SBCL 平台陷阱（必须显式处理）

### 7.1 `with-float-traps-masked` 的局限

SBCL 对**非有限值**调用 `floor/round/ceiling/truncate` 会触发 `FLOATING-POINT-INVALID-OPERATION`，且**无法**被 `with-float-traps-masked` 屏蔽。

**约定**：凡可能接触非有限值的取整/转换路径，**必须显式拦截**：
```lisp
(if (%nan-or-inf-p x) <约定返回值> (floor x))
```

### 7.2 NaN/Inf → 整数 cast

- 统一返回 **0**（本库约定，NumPy 未定义）
- 必须走显式判定，不得依赖 `truncate` 的行为

### 7.3 浮点确定性

- 归约**累加顺序**在快路径与通用路径中必须一致（避免并行/分块改变结果）
- `*matmul-thread-count*` 并行分块**不得**改变 GEMM 的累加顺序语义（当前按 k 维顺序累加，分块须保持）

---

## 8. 宏展开与源码结构约定

### 8.1 括号检查

**教训（REFACTOR-NOTES v0.3.6 事故）**：`def-vt-reduce` 宏体内两处括号错位导致 14 个函数失效。全文件括号**净差为 0** 不代表正确。

**约定**：
- 修改文件后必须逐个核对**顶层形式边界**（建议用 `(read)` 循环读取整个文件验证）
- 修改宏体后必须**实测展开**（`macroexpand-1`）并检查生成的函数能正常调用

### 8.2 源文件加载顺序（`:serial t`）

`package → util → iterator → nan → dtype → core → map-reduce → io → creation → manip → indexing → join → elementwise → reduce-stats → setops → random → linalg → simd-matmul → nn → rotate → extensions → extensions2 → extensions3`

**约定**：新增源文件必须插入到正确位置，且**不得**引入前向依赖（除通过 `defvar` 声明的回调钩子，如 `*simd-matmul-2d-fn*`）。

### 8.3 快路径钩子模式（保留）

`linalg.lisp` 定义 `*simd-matmul-2d-fn*` / `*simd-batched-matmul-fn*`，由 `simd-matmul.lisp` 在加载时 `setf` 注册。**保留**该模式（解决加载顺序问题），但必须：
- 钩子返回 `nil` 表示**回退**，返回 `vt` 表示**接管**
- 钩子在内部失败时**必须** `return-from nil` 而非抛错

---

## 9. 逐函数约定核对清单

### 9.1 `vt-arange`（唯一例外）

**用户明确豁免**：`vt-arange` **不要求**与 NumPy 对齐。

| 项 | NumPy `np.arange` | clvt `vt-arange` | 说明 |
|---|---|---|---|
| 语义 | `arange([start,] stop[, step])`，**按 stop 计算长度** | `(vt-arange total-num &key start step dtype)`，**直接给元素个数** | ⚠️ 语义不同 |
| dtype 默认 | 由输入推断（整数→int64，浮点→float64） | `:float64` | 保留 |

> **裁决**：尊重用户豁免，**保持 clvt 现有签名**。但必须在 docstring 中：
> 1. 显式标注"**与 NumPy `arange` 语义不同**"；
> 2. 说明 `total-num` 是**元素个数**而非 `stop`；
> 3. 给出等价换算公式：`vt-arange(n, :start s, :step d)` ≡ `np.arange(s, s+n*d, d)`。
>
> **参数个数问题**：`(vt-arange total-num &key start step dtype)` 的 `total-num` 是**必需**的。文档与测试需明确"不提供 `total-num` 时行为"（当前是 `odd number of &KEY arguments` 的裸报错 → **必须**改为明确错误信息）。

### 9.2 其余函数族的对齐要点

| 函数族 | 关键对齐点 | 状态 |
|---|---|---|
| `vt-+ - * /` | varargs；`vt-/` 整型输入预提升 float64（NumPy true_divide） | ✅ 已对齐（D2 已修） |
| `vt-mod/rem` | 零除返回 0（整数，dtype 保持）/ nan（浮点）；符号跟随除数 | ✅ 已对齐 |
| `vt-div` | 对标 `np.floor_divide`：整数截断除法，整数零除返回 0（不报错）；浮点按 IEEE | ✅ 已对齐（本轮） |
| `vt-maximum/minimum/fmax/fmin` | NaN 传播 vs 忽略 | ✅ 正面样板（D10） |
| `vt-clip` | 对齐 numpy 2.3.5 `clip(a, a_min=None, a_max=None)`：两边界均可省（返回原值）；只给 max 合法；**只给 min 报错**；`a_min > a_max` → 全为 `a_max` | ✅ 已对齐（本轮） |
| `vt-hypot` | 任一 Inf → `+Inf`（即使另一为 NaN） | ✅ 已对齐（D4） |
| `vt-reciprocal` | 整数输入 → NumPy 返回 0（整型倒数恒 0） | ✅ 已对齐（D9） |
| `vt-signum` | `sign(0) → 0`；`sign(nan) → nan` | ✅ |
| `vt-sum/prod` | 整数 → int64；float32 → float32 | ✅ 正面样板（D12） |
| `vt-mean/var/std` | 整数 → float64；`ddof` 已支持（默认 0） | ✅（**缺 docstring** → 任务5） |
| `vt-argmax/argmin` | 首个极值；NaN 下标行为 | ✅ 正面样板（D14） |
| `vt-sort/argsort` | NaN 末尾 | ✅ |
| `vt-percentile/quantile` | 默认 `:linear` 插值 | ✅ |
| `vt-diff` | 默认 `n=1`、`axis=-1` | ✅ |
| `vt-gradient` | 边界一阶单侧、内部中心差分 | ✅ |
| `vt-interp` | `left`/`right` 缺省用端点值 | ✅ |
| `vt-concatenate` | dtype 按提升规则统一 | ✅ |
| `vt-take` | `axis nil` 展平；负索引支持 `mode` | ✅ |
| `vt-put` | `mode=:raise` 默认 | ✅ |
| 索引 dtype 契约 | `vt-take` / `vt-put` / `vt-choose` / `vt-insert` 的 `positions`：**非空**浮点索引报错（对标 numpy 的 TypeError/IndexError，禁止 `0.5 → 0` 静默截断）；**空索引合法**（no-op / 空结果），因为空索引没有元素可截断，且 `(vt-from-sequence '())` 这类「无索引」写法默认是 float64。统一由 `%check-integer-index` 判定 | ✅ |
| `vt-reshape` | 至多一个 `-1`；尺寸不符报错 | ✅ |
| `vt-squeeze` | 默认移除**全部**长度 1 轴（`(1,3,1)→(3)` ✅ 实测） | ✅ |
| `vt-transpose` | 默认反转全部轴 | ✅ |
| `vt-norm` | 整数输入 → float64 | ✅ |
| `vt-einsum` | out dtype 硬校验（当前已实现，**保留为样板**） | ✅ |
| `vt-matmul` | 1d 提升语义；`out` 走 `vt-copy-into` | ✅ |
| `vt-isclose` | `rtol=1e-5, atol=1e-8` | ✅ 实测默认宽松（1e-6 内判等），需在文档写明 |
| `vt-arange` | 整型溢出与 `int16/int8/uint*` 一致采用**回绕**语义（`%wrap-*`），不报错（见 §9.1） | ✅ |
| `vt-linspace` | float32 输出**以 double 精度计算后舍入存储**（对标 NumPy，避免累积漂移） | ✅ |
| `vt-random-uniform` / `vt-random-normal` | `uniform` 要求 `low`/`high` 为有限实数，`low = high` 合法（返回常量数组）；`normal` 要求 `std` 为非负有限实数，`std = 0` 返回 `mean` 填充 | ✅ |
| `vt-slice` | spec 语法须与文档统一（**无 `:range`** → D1） | ✅ 实现正确，仅需补文档 |
| `vt-where` | 单参数形式 `(vt-where cond)` 返回**按轴分组的索引列表**（长度 = rank，等价 `vt-nonzero` / `np.nonzero(cond)`）；三元形式同 `np.where(cond,x,y)` | ✅ 已对齐（见 D6） |
| 空归约 | 三分类语义 | ✅ 正面样板（D11） |
| 比较运算 | 返回 `:int8` 承载布尔（0/1） | ✅（文档需写明） |
| 三角/反三角 | `asin/acos/atan` 命名（非 arc*），越界返回 NaN | ✅（命名见 §3.1.1） |
| **docstring 覆盖** | 每个公开函数须有文档 | ⚠️ `vt-var` 等为 NIL，需全量审计（任务5） |
| 激活函数族 | `vt-sigmoid`/`vt-relu`/`vt-leaky-relu`/`vt-swish`/`vt-softplus`/`vt-gelu`/`vt-mish`/`vt-hard-tanh`/`vt-hard-sigmoid`：输出**恒为浮点**，整数输入提升 `:float64`，float32 保持 float32（见 §3.4） | ✅ 已对齐（本轮修 softplus/gelu/leaky-relu） |
| `vt-cross-entropy` | **概率/one-hot** 语义（`-Σ y·log(p)`，p 裁剪 `[eps,1-eps]`）；**不是** torch 语义 | ✅ 语义已明确（见 §3.5） |
| `vt-cross-entropy-logits` | **raw logits + 整数类别索引**，对标 `torch.nn.CrossEntropyLoss`（log-softmax，不裁剪）；浮点 labels / 越界索引报错 | ✅ 新增（见 §3.5） |

---

## 10. 测试约定

### 10.1 契约三底线（保留自 TEST-PLAN.md）

| 编号 | 底线 | 含义 |
|---|---|---|
| **L1** | 内存安全 | 任何输入（含畸形）都不得越界读写 |
| **L2** | 非法输入确定性报错 | 报错信息可读、可定位 |
| **L3** | 合法输入零检查税 | 快路径不得因检查而显著变慢 |

### 10.2 测试矩阵（C1–C10，保留）

C1 形状/秩 · C2 dtype · C3 广播 · C4 归约轴 · C5 NaN/Inf · C6 空张量 · C7 非连续 · C8 `:out` · C9 别名/重叠 · C10 边界数值

### 10.3 新增必测矩阵（任务3/4 要求）

**out × 连续性 × NaN 的笛卡尔积**，每格必须有断言：

| out 形态 | 输入含 NaN | 输入与 out 别名 | 必测 |
|---|---|---|---|
| 连续 | 否 | 否 | ✔ |
| 非连续（切片/转置/步长） | 否 | 否 | ✔ |
| 连续 | 是 | 否 | ✔ |
| 非连续 | 是 | 否 | ✔ |
| 连续 | 否 | 是（完全） | ✔ |
| 非连续 | 否 | 是（部分重叠） | ✔ |

**核心断言**：
```
(快路径结果) == (通用路径结果)  ∀ 上表每一格
```
实现方式：用 `vt-fast-map` / SIMD 强制开与强制关两次运行，比对逐位相等。

### 10.4 与 NumPy 的对照测试（保留并强化）

- 生成器：`test/gen_numpy_expected.py` + `test/ref_compute.py`
- 对照脚本：`test/numpy-compare-test.lisp`、`test/auto-compare-test.lisp`
- **要求**：每个新增/修改函数的语义必须在对照测试中有对应用例
- **基准**：NumPy **2.3.5**（更新 TEST-PLAN.md 中记录的 2.1.3）

#### 10.4.1 API 指纹与 codegen 门禁（2026-10 修订）

差分探针里有一类"自检"用例（`p0395`/`p0396`）不验证数值语义，而是验证
**包/工具链自身的状态**。这类断言最容易写成"脆弱指纹"，规则如下：

| 规则 | 说明 |
|---|---|
| **禁止把导出数量写死** | 早期 `p0395` 断言 `(length *vt-fun-list*)` 等于常量（363→371…），于是每次在 `package.lisp` 新增导出都变红 —— 而这是**正常演进**。现在断言不变量：`*vt-fun-list*` ≡ 包导出集（`vt-` 前缀）、无重复、非空 |
| **指纹口径必须唯一** | `refresh-vt-fun-list` 只能统计**导出**符号（`DO-EXTERNAL-SYMBOLS`）。用 `DO-SYMBOLS` 会把 `vt-complex-dtype-p` 这类内部辅助算进来，且结果随"何时调用"漂移 |
| **导出表不得重复** | `defpackage` 允许同名重复导出，但静态解析会重复计数（曾出现 `vt-real/vt-imag/vt-conj/vt-angle` 各两次 → 静态 371 ≠ 运行时 367，`p0395` 恒红） |
| **生成器无参 == gen** | `check_key_sync.py` 用无参调用 + 逐字节比对做漂移检查；生成器若在无参时崩溃（`IndexError`），文件不会被重写，门禁就**永远通过**——静默失效比没有门禁更危险 |
| **生成必须可复现** | 连续两次生成必须逐字节相同；随机类用例统一 `rng(seed)`，禁止裸用全局 RNG |
| **浮点格式随 numpy 版本变** | `gen_probes*.py` 用 `str(np.float64)` 输出期望值：numpy 2.4 给 `0.27279899999999996`、2.5 给 `0.272799`。**同一份代码在不同 numpy 下 codegen 必然漂移**，但数值只差 1 ulp（比较用 `:tol`）。遇到跨版本漂移时不要盲目「重新生成再提交」——先确认是格式化差异还是真语义变化（本轮就遇到过：本机 numpy 2.4.6 报漂移，上游 2.5.0 正常） |
| **重新生成后先 `git add`** | 门禁结束时 `git checkout -- <生成物>` 还原工作区，否则改动会被门禁自己抹掉 |

> 新增导出函数的完整操作：改 `package.lisp`（不重复）→ 跑 `python3 test/gen_probes2.py`
> → `git add` → `bash test/run-tests.sh --suite differential-probes-test2`。
> **不需要**修改任何期望值。

---

## 11. 文档与注释约定（任务5）

### 11.1 docstring 规范

每个公开函数必须有 docstring，包含：

1. **一行摘要**
2. **`:out` / `:dtype` 行为**（是否支持、契约要点）
3. **`axis` / `keepdims` 语义**（如适用）
4. **NaN/Inf 行为**（如与 NumPy 有差异或特别）
5. **与 NumPy 的差异**（无差异则显式写"对标 NumPy `xxx`"）
6. **示例**（至少一个简单调用）

### 11.2 必须删除/修正的内容

| 类型 | 处理 |
|---|---|
| 描述与现约定不符的注释（如"int8 加 int32 返回 int8"） | 改为正确描述 |
| 复制的样板注释（多函数重复同一段） | 提取为文件级注释 |
| 过时版本号引用（v0.3.x） | 更新或删除 |
| 注释掉的死代码 | 删除（git 有历史） |

### 11.3 保留的内容

- 架构总纲（`package.lisp` 三层分离注释）
- 数值算法关键推导（如 Householder、Jacobi 旋转公式）
- SBCL 陷阱的说明注释（具有防回归价值）
- 性能关键点的取舍说明

---

## 12. 重构验收标准

| # | 验收项 | 方法 |
|---|---|---|
| V1 | 25 个测试套件全绿 | `bash test/run-tests.sh --all` |
| V2 | 类型提升 64 格与 NumPy 一致 | 自动化对照脚本 |
| V3 | 三个 `+vt-float-*+` 常量已删除 | `grep` 无残留 |
| V4 | 所有 `:out` 函数走统一原语 | 代码审查 + grep |
| V5 | 非连续 out 与连续 out 结果逐位相同 | §10.3 矩阵测试 |
| V6 | 别名 out 与快照语义正确 | §10.3 矩阵测试 |
| V7 | NaN/Inf 全部行为与 NumPy 一致 | §5 逐条对照测试 |
| V8 | 每个公开函数有合规 docstring | 脚本扫描 |
| V9 | `vt-arange` 文档标明差异 | 人工检查 |
| V10 | example.lisp 可独立运行且输出正确 | `sbcl --load example/example.lisp` |

---

## 13. 实测缺陷清单（重构前基线，全部为**已验证**）

以下为对 clvt v0.3.6 逐条实测得到的具体缺陷，**必须**在任务2/3/4 中修复。

### D1 — `vt-slice` 的 spec **不支持** `:range` 关键字（文档/测试易踩坑）

```lisp
(vt-slice base '(:all) '(:range 0 6 2))
;; => ERROR: vt-slice: invalid slice spec (RANGE 0 6 2)
```

**实际支持且实测通过的 spec**（`indexing.lisp:58`）：

| spec | 含义 | 实测 |
|---|---|---|
| `(:all)` / `(t)` | 全选该轴 | ✅ |
| `(:newa)` | 插入新轴 | ✅ |
| `(:elli)` | 省略号（仅一次） | ✅ |
| `(idx)` | 整数索引，**降维** | ✅ |
| `(start end)` | 范围，`nil` 表示该侧不设限 | ✅ `(0 nil)` / `(nil 2)` |
| `(start end step)` | 范围 + 步长，支持**负步长** | ✅ `(3 nil -1)` → `(4 3 2 1)` |

**裁决**：⚠️ **保持 `(start end &optional step)` 列表形式**（已正确实现），但需：
1. 在 docstring 与 README 中给出**完整 spec 表**（上表）；
2. **禁止**在测试/示例中使用 `:range`；
3. `(:all)` 与 `(t)` 等价，选择其一作为主推写法（建议 `(:all)`）。

### D2 — `vt-+` 的 `:out` dtype 不校验（违反 H3）

```lisp
(vt-+ (vt-zeros '(2 3)) (vt-zeros '(2 3)) :out (vt-zeros '(2 3) :dtype :float32))
;; => ((2 3) FLOAT32)   ← 未报错！
```

结果 dtype (`float64`) 与 out dtype (`float32`) 不一致却静默通过。
**裁决**：✅ **已修复**（本轮验证）——现报错 `vt-fast-map: :out dtype FLOAT32
与结果 dtype FLOAT64 不匹配`。这是**静默数据损坏**风险：用户以为拿到
float64，实得被截断的 float32。

### D3 — `vt-floor` 家族对非有限值**行为正确**，但需补测试

```
(vt-floor    +vt-dfloat-nan+)  => NaN   ✅ 未崩溃（SBCL 陷阱已被拦截）
(vt-truncate +vt-dfloat-pos-inf+) => +Inf ✅
```
**说明**：`%floor-family-body` 的非有限值拦截**已生效**，未触发 `FLOATING-POINT-INVALID-OPERATION`。
**裁决**：✅ **实现无需改**（原"需重构"标记撤销）。⚠️ 但测试覆盖不足，须补 `(nan, inf, -inf)` × `(floor, ceil, round, truncate, rint)` 全矩阵，并在 docstring 写明约定（NaN→NaN，±Inf→±Inf）。

### D4 — `vt-hypot` 的 Inf 语义：**实测已正确**（原判断有误，撤销）

```
(vt-hypot +vt-dfloat-pos-inf+ +vt-dfloat-nan+)  =>  +Inf   ✅ 与 numpy 一致
```

**说明**：先前误用 `most-positive-double-float`（**有限最大值**，非 Inf）做测试，得到 NaN，因而误判。改用 `+vt-dfloat-pos-inf+` 后实测**返回 +Inf，已与 NumPy 对齐**。

**裁决**：✅ **无需修改**。但须补测试用例覆盖 `(Inf, NaN)` / `(NaN, Inf)` / `(-Inf, NaN)` 三种组合。

### D4b — `most-positive-double-float` 与 `+vt-dfloat-pos-inf+` 的混淆风险

`most-positive-double-float` 是 CL 内置的**有限最大值**（≈1.8e308），语义**完全不同于** Inf。
**裁决**：⚠️ 库内**一律**使用 `+vt-dfloat-pos-inf+` 表示 Inf；文档、测试、注释中不得混用。这是 D4 误判的根因，须在任务5 中全库排查。

### D5 — `vt-clip` 签名已对齐 numpy 2.3.5

```lisp
(vt-clip a)          ; → 返回 a 的副本（不裁剪），对齐 np.clip(a)
(vt-clip a nil 2.0)  ; → 仅上限，对齐 np.clip(a, None, 2)
(vt-clip a 2.0 nil)  ; → 仅下限，对齐 np.clip(a, 2, None)
(vt-clip a 2.0)      ; → 报错（对齐 numpy TypeError: missing a_max）
(vt-clip a 2.0 2.0)  ; → 全为 2.0，对齐 np.clip(a, 2, 2)
```

NumPy 2.3.5 实测：`clip(a, a_min=None, a_max=None)`——两边界**均可省略**，
但**不允许只给 `a_min`**（缺 `a_max` 抛 TypeError）。
**裁决**：✅ **已按 numpy 2.3.5 对齐**（本轮修复）。签名改为
`(vt-clip vt &optional min-val max-val &key out dtype)`；`min-val` 传 `nil`
表示不设下限、`max-val` 传 `nil` 表示不设上限；只给 `min-val` 而省略
`max-val` 时报错。

### D6 — `vt-where` 单参数形式已对齐 numpy

```lisp
(vt-where (vt-from-sequence '(1 0 1)))
;; => 长度 1 的索引列表，元素为 shape (2) 的 int64 张量 [0 2]   ✅ 对齐 np.where(cond)

(vt-where (vt-from-sequence '((1 0) (0 1))))
;; => 长度 2 的索引列表：轴0 -> [0 1]，轴1 -> [0 1]
```

numpy 文档明确 `np.where(cond)` 是 `np.asarray(cond).nonzero()` 的**简写**，
返回的是**按轴分组的索引元组**（长度 = rank），**不是** `np.argwhere` 的
`(n, rank)` 坐标堆叠。此前 clvt 返回 `(n, rank)`（理由是「CL 无元组」），
与 numpy 契约不符；CL 侧用**张量列表**承载该元组即可，这也正是库内
`vt-nonzero` 既有的返回约定，因此二者现在完全一致。

需要 `(n, rank)` 形式请直接用 `vt-argwhere`。
三元形式 `(vt-where cond x y)` 不受影响。
**裁决**：✅ **已修正为按轴分组列表**（回归见
`test/contract-regression-test.lisp`）。

### D7 — `vt-var` / `vt-std` **已有** `ddof`，但**缺 docstring**

```
(vt-var (vt-from-sequence '(1.0 2.0 3.0 4.0)) :ddof 1)  => 1.6666666666666667d0   ✅ 与 numpy 一致
(vt-std (vt-from-sequence '(1.0 2.0 3.0 4.0)) :ddof 1)  => 1.2909944487358056d0  ✅
(documentation 'vt-var 'function)                       => NIL                   ⚠️ 无文档
```

**裁决**：✅ `ddof` **无需新增**（原判断有误，已撤销）。⚠️ 但 `vt-var`/`vt-std` 等函数**缺 docstring**，属任务5 范围。

### D8 — `vt-arange` 无参调用报错信息（**豁免项，保持现状**）

```lisp
(vt-arange)
;; => ERROR: invalid number of arguments: 0
```
**裁决**：`vt-arange` 是**唯一不以 numpy 为准的例外**（用户明确豁免，见 §9.1），
其签名 `(vt-arange total-num &key start step dtype)` 与 `np.arange` 语义不同，
**保持现状不做对齐**。报错信息维持 CL 原生即可。

### D9 — `vt-reciprocal` 对整数输入返回浮点（违反 NumPy）

```lisp
(vt-reciprocal (vt-astype (vt-const '(3) 2) :int32))  ;; => (0.5d0 0.5d0 0.5d0)
```
NumPy：`np.reciprocal(np.array([2],dtype=np.int32))` → `[0]`（整型倒数恒 0，因整数除法）。
**裁决**：✅ **已对齐 NumPy**（本轮验证）——clvt `vt-reciprocal` 对 int32 输入
返回 `(0 0 0)` dtype INT32，与 numpy 一致。
**说明**：`vt-/` 保持 true_divide 语义（整型提升 float64），二者分工明确。

### D10 — `vt-maximum`/`vt-fmax` 的 NaN 语义已正确（**正面样板**）

```lisp
(vt-maximum (vt-from-sequence '(nan 1.0)) (vt-from-sequence '(2.0 3.0)))
;; => (NaN 3.0d0)   ✅ NaN 传播，与 NumPy 一致
```
**裁决**：✅ 保留，作为其他 NaN 函数的参照实现。

### D11 — 空归约错误信息已对齐 NumPy（**正面样板**）

```
(vt-amax  (vt-zeros '(0)))  => ERROR: zero-size array to reduction operation maximum which has no identity
(vt-argmax(vt-zeros '(0)))  => ERROR: attempt to get argmax of an empty sequence
(vt-sum   (vt-zeros '(0)))  => (NIL FLOAT64)   ← 0 维结果，值为 0
```

**裁决**：✅ **保留**。空归约三分类（输出为空→空结果；arg 族→ValueError；max/min 族→ValueError；单位元族→填充单位元）**已与 NumPy 一致**，重构时**不得破坏**。

### D12 — 归约 dtype 已对齐 NumPy（**正面样板**）

```
(vt-sum (vt-zeros '(3) :dtype :int8))    => INT64      ✅ numpy int64
(vt-sum (vt-zeros '(3) :dtype :float32)) => FLOAT32    ✅ numpy float32
(vt-mean int32)                          => FLOAT64    ✅ numpy float64
(vt-var int32)                           => FLOAT64    ✅ numpy float64
```

**裁决**：✅ **保留**。

### D13 — 广播的 0 长度轴已对齐 NumPy（**正面样板**）

```
(vt-+ (vt-zeros '(0 3)) (vt-zeros '(1 3)))  => shape (0 3)     ✅ numpy (0,3)
(vt-+ (vt-zeros '(3 0)) (vt-zeros '(1 4)))  => ERROR 可读       ✅ numpy ValueError
```
**裁决**：✅ **保留**。§2.1 的对应"⚠️ 需重构"标记**撤销**。

### D14 — `amax` 首个 NaN 胜出已实现（**正面样板**）

```
(vt-amax (vt-from-sequence '(nan 1.0 -5.0)))  => NaN   ✅
(vt-amax (vt-from-sequence '(1.0 nan -5.0)))  => NaN   ✅
(vt-argmax (vt-from-sequence '(1.0 nan 5.0))) => 1     ✅ 首个 NaN 下标
```
**裁决**：✅ **保留**。

### D15 — `vt-astype` 的 `most-positive-double-float` 别名

`vt-hypot` 测试中我用到 `most-positive-double-float`；库内应使用 `+vt-dfloat-pos-inf+`（Inf）而非 `most-positive-double-float`（有限最大值）。二者语义**完全不同**，文档与测试中必须区分。

### D16 — `vt-div` 整数零除抛错（违反 NumPy）

```lisp
(vt-div (vt-from-sequence '(7) :dtype :int64) 0)
;; => ERROR: arithmetic error DIVISION-BY-ZERO signalled   （改前）
```

`vt-div` 对标 `np.floor_divide`（整数截断除法）。NumPy 实测
`np.floor_divide(np.array([7]), 0)` → `[0]`（**不报错**，返回 0，dtype 保持 int64）。
**裁决**：✅ **已修复**（本轮）——整数零除返回 `0`，dtype 保持整型；
浮点零除按 IEEE 得 ±Inf/NaN。实现改用 `vt-map` + 显式零除守卫。

---

## 14. 修正后的对齐状态总览

### 14.0 Task 3 实施现状（`parcontract.lisp`）

**已落地的库级原语**（`src/parcontract.lisp`，加载顺序在 `core.lisp` 之后）：

| 原语 | 职责 |
|---|---|
| `vt-check-out` | 硬契约 H1–H4：out 是 vt / 形状精确匹配 / dtype 精确匹配 / 可写 |
| `vt-check-out-dtype-consistency` | 硬契约 H5：`:dtype` 与 `:out` 冲突检测 |
| `vt-out-writable-p` | 广播视图（dim>1 且 stride=0）判定为不可写 |
| `vt-out-snapshot` | 别名保护：重叠输入先 `vt-copy` 快照 |
| `vt-out-contig-p` | 仅作选路建议，不影响结果 |
| `vt-write` | **strides 驱动**寻址写入，零连续性假设 |
| `vt-reduce-dtypes` | 归约类精度解耦三段式 |
| `%parcontract-self-check` | 加载期自检（寻址/快照三点回归网），失败中断加载 |
| `vt-params-audit` / `vt-params-audit-report` | 参数契约审计（个数/默认值/`&key`/`&rest`/docstring） |

**参数契约实测统计（审计工具输出）**：

| 指标 | 数量 |
|---|---|
| 公开 `vt-*` 函数总数 | **341** |
| 接受 `:out` 的函数 | **139** |
| 声明 `&key` 的函数 | 220 |
| 使用 `&rest` 的函数 | 16 |
| **缺 docstring 的函数** | **174** |

> **任务3 的范围因此明确为 139 个带 `:out` 的函数**，任务5 的范围为 174 个缺 docstring 的函数。

#### 14.0.1 任务3 迁移进度（v0.3.6 收口）

`:out` 的 dtype 契约已统一为**「严格相等」**（用户裁决）：结果 dtype 由
「输入提升 + 显式 `:dtype`」决定，**不由 out 决定**；out 必须精确匹配，
否则报错。全部带 `:out` 的公开函数按下列分层完成迁移：

| 层 | 文件 | 迁移方式 | 状态 |
|---|---|---|---|
| 核心原语 | `map-reduce.lisp` | `vt-map` / `vt-fast-map` / `vt-reduce` 统一接入 `vt-check-out` + `vt-out-snapshot` | ✅ |
| 逐元素 | `elementwise.lisp` | 74 个函数经 `vt-fast-map` 统一；`vt-/` 删除「out 决定 dtype」、`vt-reciprocal` 修为 numpy 整数语义 | ✅ |
| 归约/统计 | `reduce-stats.lisp` | `def-vt-reduce` 宏（`decoupled` 恒 nil）+ `vt-mean`/`vt-var`/`vt-std`/`vt-cumulative`/`vt-average`/`vt-nanmean`/`vt-nanvar`/`vt-nanstd`/`vt-nanmedian` | ✅ |
| 线性代数 | `linalg.lisp` | `vt-einsum` 结果 dtype 不取 out；`einsum-execute` 别名改用物理区间重叠判定；`vt-det` 补 dtype 校验 | ✅ |
| SIMD 矩阵乘 | `simd-matmul.lisp` | 2d/batched 两条快路径：结果 dtype 不取 out + 别名改重叠判定 + 可写性检查 | ✅ |
| 索引 | `indexing.lisp` | `vt-where` 结果 dtype = `(or :dtype promote)` + 统一 `vt-check-out` | ✅ |
| 神经网络 | `nn.lisp` | `vt-sigmoid`/`vt-relu` 前置 `vt-check-out`（非连续 out 自动回落通用路径） | ✅ |
| 扩展 | `extensions.lisp` / `extensions2.lisp` / `extensions3.lisp` | 全部透传下游原语，已自动获得统一契约（比较/逻辑类返回 `:int8`） | ✅ |
| creation / manip / join / setops / rotate / random | 同上 | 这些函数**本无 `:out` 参数**（与 numpy 一致，如 `concatenate`/`reshape` 无 `out=`） | N/A |

**关键实现要点**（易错，已在测试中固化）：

1. **中间结果不得下传用户 out**。`vt-mean` 等会串联多个子运算；若把用户
   out 直接下传给 `vt-sum` 的 `:out`，子运算的 H5（`:dtype` 与 `:out` 一致性）
   会先于本函数的 H3 触发，报出 `vt-SUM: ...` 这类误导性错误。正确做法是：
   本函数开头用 `vt-check-out` 校验一次，之后 out 仅作最终写入目标。
2. **别名检测必须比较物理区间，不能只比 `vt-data` 指针**。同 data、异
   offset 的重叠视图（`a = base[0:64]`、`out = base[32:96]`）是真实可达的；
   只比指针会漏检，导致预清零破坏输入。统一用 `%vt-views-overlap-p`。
3. **累加式内核必须保留预清零**。`einsum-execute` 的 BMM 回退路径用
   `(incf (aref dc ...) ...)`，依赖输出缓冲区初始为 0；清零动作必须放在
   「重叠输入快照**之后**」，否则别名场景下会先破坏输入。

**任务3 验收**：`test/out-contract-v2-test.lisp` 共 56 项断言全通过，
覆盖非连续 out、转置 out、三类别名、NaN/Inf 经 out、硬契约违反、
快路径 vs 通用路径逐位对照、归约族错误信息前缀、matmul 别名。

**补充修复（任务6 阶段发现）**：`vt-fast-map` 的 dtype 检查失败回退分支
原先写成 `(vt-map op ... :out res)`，**丢失了 `:dtype`**。当显式 `:dtype`
与输入提升结果不同（如 `vt-=` 对 int64 输入要求 `:float64` 的 0/1 输出）时，
快路径的 `dtype-check` 不成立 → 回退到 `vt-map`，而 `vt-map` 按输入提升推出
`:int64`，与 out 的 `:float64` 冲突，误报 H3。修法：三处回退调用（n=1/2/3）
及别名回退分支全部补 `:dtype final-dtype`，使快慢路径 dtype 语义一致。

#### 14.0.2 任务6 测试与 example 审计（v0.3.6）

`example/example.lisp` 的 `run-all-tests` 与官方 `test/run-tests.sh`
（25 套件）已全绿。修复的测试**资产缺陷**归纳为两类：

1. **浮点断言用 `equal`/`equalp` 判等**（根本缺陷）。Common Lisp 的
   `equal` 对不同浮点格式一律判不等（`(equal 1.0 1.0d0) => NIL`），而
   `:float64` 张量元素是 double-float，期望值却常写成无后缀 single-float
   字面量 → 必然假失败。修法：新增浮点感知比较器 `vt-test-equal`
   （数值走 epsilon 近似、序列/数组递归并校验形状、其余回退 `equal`），
   并将 280 处 `(equal (vt-to-list ...) ...)` / `(equalp (vt-to-array ...) ...)`
   批量替换。`test-vt-reduce` 的局部 `check` 改为「浮点近似 / 非浮点严格」。
2. **字面量精度错误**（数值本身写错，epsilon 无法拯救）。如
   `1.0000000001`（single，被舍入为 `1.0`，令矩阵退化为真奇异）、
   `1.0e12`（single 实际值 `999999995904`，与 `1.0d12` 差 4096）、
   `(/ 2.0 3.0)`（single 结果 vs float64 实测差 2.2e-8）。修法：改为
   `d0` / `dN` 后缀字面量。

**审计结论**：算法实现正确，失败均源自测试资产本身的比较方式与字面量精度，
符合用户「错误的请完善，正确的保留」要求。

### 14.1 对齐状态表

| 类别 | 已对齐 ✅ | 需重构 ⚠️ |
|---|---|---|
| 类型提升（64 格） | 全部 | — |
| 归约 dtype | sum/prod/mean/var/std | — |
| 广播（含 0 长度轴） | 全部 | — |
| NaN 极值/排序 | maximum/minimum/fmax/fmin/amax/amin/argmax/sort/argsort | — |
| 空归约 | 全部 | — |
| `:out` 形状校验 | 全部带 `:out` 的函数 | — |
| `:out` dtype 校验（严格相等） | 全部带 `:out` 的函数（D2） | — |
| `:out` 非连续写入 | 全部（§4.6 机制 B） | — |
| `:out` 别名快照 | `vt-map` / SIMD matmul / einsum 统一用物理区间重叠判定 | — |
| 零除（整数） | `mod` / `rem` / `div`（floor_divide）均返回 0、dtype 保持（D16） | — |
| `vt-clip` 参数 | 已按 numpy 2.3.5 语义：无边/只 max 合法，只 min 报错（D5） | — |
| `vt-where` 单参 | 已对齐：`(vt-where cond)` → 按轴分组索引列表（= `vt-nonzero`，D6） | — |
| `vt-arange` 报错 | **唯一例外**，保持现状不对齐（D8，用户豁免） | — |
| `vt-reciprocal` 整数 | 已按 numpy 整数语义（D9） | — |
| docstring 覆盖 | 公开 `vt-*` 基本齐全（余 4 个 defstruct 访问器） | — |
| `most-positive-double-float` 误用 | 已排查（D4b） | — |
| 测试/example 审计 | `run-all-tests` + 27 套件全绿（任务6） | — |
| 覆盖缺口审计 | **新增 28 套件全绿**（`uncovered-coverage-test` 136 断言）；6 处缺陷修复见 §15（本轮） | — |

---

## 15. 覆盖缺口审计与缺陷修复（本轮）

本节记录对 **306 个导出 `vt-*` 函数**做的覆盖审计：用
`comm` 对比 `src/package.lisp` 导出清单与 `test/*.lisp` + `example/example.lisp`
的全部符号引用，得到 61 个**从未被任何测试或示例调用**的公开函数，
并为其补齐断言（`test/uncovered-coverage-test.lisp`）。

### 15.1 修复的实现缺陷

| # | 函数 | 缺陷类别 | 复现 | 修法 |
|---|---|---|---|---|
| F1 | `vt-identity` | **崩溃**（缺省值 NIL 下传） | `(vt-identity 3)` → `NIL fell through ECASE` | `&key (dtype :float64)` |
| F2 | `vt-vander` | **崩溃**（`&key n` 无缺省） | `(vt-vander x)` → `odd number of &KEY arguments` | `&key (n nil)` → 缺省 `len(x)` |
| F3 | `vt-atanh` | 边界值不符 numpy | `atanh(±1)` → NaN（应 ±Inf） | `±1→±Inf`，`\|x\|>1→NaN` |
| F4 | `vt-= vt-/= vt-< vt-<= vt-> vt->=` | **dtype 契约违反**（§3.1） | 返回 `float64`，应 `:int8` | 缺省 dtype `:int8`，值 1/0 |
| F5 | `vt-positive-p vt-negative-p vt-zero-p vt-nonzero-p vt-even-p vt-odd-p` | 同上 | 返回 `float64` | 缺省 dtype `:int8` |
| F6 | `vt-isnan vt-isinf vt-isfinite` | 同上 | 返回 `float64` | dtype 固定 `:int8` |

> F4–F6 的裁决依据：CONVENTIONS §3.1「比较运算返回 :int8，取值 0/1，语义上
> 等价于 NumPy bool」，§9.2「比较运算 → 返回 :int8 承载布尔」。
> NumPy 侧布尔数组与 clvt `:int8` 的**值语义完全一致**，仅 dtype 标签不同，
> 故本修复不构成与 NumPy 的冲突，而是向**本文档自身契约**收敛。
> 计数类（`vt-count-nonzero`、`vt-all`/`vt-any` 的规约输出）保持整数 dtype。

### 15.2 审计确认「正确、无需修改」的函数

以下函数一度被怀疑有缺陷，经 numpy 2.4.6 逐项对照后确认**实现正确**，
本节的结论是"补测试即可，实现不动"：

`vt-select`（condlist/choicelist 列表语义）、`vt-dsplit`（axis=2）、
`vt-vsplit`、`vt-hsplit`、`vt-moveaxis`（含负轴与多轴 `(0,1)->(2,3)`）、
`vt-lerp`（CL 参数顺序 start/end/weight）、`vt-write`（strides 驱动寻址）、
`vt-layer-norm`、`vt-apply-along-axis`、`vt-tensordot`、`vt-geomspace`、
`vt-identity` 的非缺省 dtype 分支。


---

---

## 16. 复数（complex）约定 — 第一轮进化

> **基准**：NumPy **2.4.6**（本机实测）+ SBCL **2.6.8**（本机实测）。
> **适用范围**：`dtype ∈ {:complex64, :complex128}` 的全部语义。
> **落地文件**：`src/dtype.lisp`（类型系统）、`src/core.lisp`（构造/推断）、
> `src/io.lisp`（打印/序列化入口）、`src/complex.lisp`（语义层，最后加载）。
> **对照测试**：`test/complex-test.lisp` + `test/gen_complex_expected.py`
> （期望值由 numpy 实测生成，共 **83 项差分用例 + 53 项契约断言 = 136 条断言**）。

复数不是「多加一个 dtype」那么简单：它同时打破 4 条既有隐含假设——
**全序比较**、**实数域数学函数的定义域**、**归约的单位元**、**结果 dtype = 输入 dtype**。
本章把每一处都钉死。

---

### 16.1 物理表示与 dtype（唯一存储模型）

| 项 | 规定 |
|---|---|
| dtype 符号 | `:complex64` / `:complex128`（导出为 `:complex128` 等关键字，与既有 8 种风格一致） |
| CL 元素类型 | `(complex single-float)` / `(complex double-float)` |
| itemsize | **8 / 16 字节**（= 2 × 分量宽度，与 `np.dtype('complex128').itemsize = 16` 一致） |
| 内存布局 | **interleaved**（实部、虚部相邻），不是双平面（planar） |
| strides 单位 | 仍是**元素**（1 个 stride 步 = 1 个复数 = 2 个浮点），**不是分量** |
| 逻辑 zeroness | `complex(0,0)` 即「零」，`stride-0` 广播、切片、转置、别名快照全部沿用既有机制 |

**铁律 3（本章新增）**：复数的存在**不改变物理层**。任何依赖 `data/shape/strides/offset/dtype`
的既有代码（切片、广播、`vt-copy-into`、别名快照、`vt-astype` 的非连续 run 分解）
在复数上必须**原样正确**，不得为复数分叉出第二套内存模型。

> 实测依据：`(array-element-type (vt-data x))` = `(COMPLEX DOUBLE-FLOAT)`，
> `vt-itemsize` = 16，`vt-slice`/`vt-transpose`/`vt-reshape` 在复数上与实数逐位同构。

---

### 16.2 构造与 dtype 推断

| 输入 | NumPy 行为 | clvt 裁决 |
|---|---|---|
| `1+2j` 标量 | `complex128`（0 维） | `:complex128`（0 维） |
| `#C(1.0 2.0)`（单精度分量） | `complex64` | `:complex64` |
| `[1, 2, 3+4j]` 混合列表 | `complex128`（整数/浮点被提升） | `:complex128`（`%any-complex-in-sequence-p` 判定） |
| `np.zeros(2, complex)` | `complex128` | `(vt-zeros '(2) :dtype :complex128)` |
| `re + 1j*im` 惯用式 | — | `(vt-complex re im :dtype :complex128)` |
| `np.array([1+2j], dtype=float)` | **TypeError**（`float() argument must be … not 'complex'`） | 显式实数 `:dtype` 时**按 §14.4 取实部**（不抛错）——这是 clvt 的 `vt-from-sequence` 既有「安全转换」立场，已记录为差异 D-C2 |

**约定**：`vt-zeros / vt-ones / vt-empty / vt-full / vt-const / vt-zeros-like / vt-ones-like`
的 `:dtype` **必须显式写** `:complex128` 才能得到复数（缺省仍是 `:float64`）。
`vt-const '(1) #C(1 2)` 会按 §14.4 丢虚部 —— 这是**已接受的陷阱**，
测试文件里凡是复数用例一律显式 `:dtype :complex128`。

---

### 16.3 类型提升（**100 格逐格实测**）

`vt-promote-type` 必须复现下表（`np.zeros(1,dtype=a) + np.zeros(1,dtype=b)` 的实测结果）：

| ⊔ | int8 | int16 | int32 | int64 | uint8 | uint16 | float32 | float64 | complex64 | complex128 |
|---|---|---|---|---|---|---|---|---|---|---|
| **int8** | int8 | int16 | int32 | int64 | int16 | int32 | float32 | float64 | complex64 | complex128 |
| **int16** | int16 | int16 | int32 | int64 | int16 | int32 | float32 | float64 | complex64 | complex128 |
| **int32** | int32 | int32 | int32 | int64 | int32 | int32 | float64 | float64 | complex128 | complex128 |
| **int64** | int64 | int64 | int64 | int64 | int64 | int64 | float64 | float64 | complex128 | complex128 |
| **uint8** | int16 | int16 | int32 | int64 | uint8 | uint16 | float32 | float64 | complex64 | complex128 |
| **uint16** | int32 | int32 | int32 | int64 | uint16 | uint16 | float32 | float64 | complex64 | complex128 |
| **float32** | float32 | float32 | float64 | float64 | float32 | float32 | float32 | float64 | complex64 | complex128 |
| **float64** | float64 | float64 | float64 | float64 | float64 | float64 | float64 | float64 | complex128 | complex128 |
| **complex64** | complex64 | complex64 | complex128 | complex128 | complex64 | complex64 | complex64 | complex128 | complex64 | complex128 |
| **complex128** | complex128 | complex128 | complex128 | complex128 | complex128 | complex128 | complex128 | complex128 | complex128 | complex128 |

#### 16.3.1 形式化规则（**唯一实现口径**）

> **复数的实部分量参与普通实数提升；提升结果若为 `float32` → `complex64`，
> 若为 `float64` → `complex128`。**

即 `promote(a, b, …) = map_component(realpromote(σ(a), σ(b), …))`，其中
`σ(:complex64) = :float32`、`σ(:complex128) = :float64`、其余 `σ(d) = d`，
`map_component(:float32) = :complex64`、`map_component(:float64) = :complex128`。

这一条规则**无需为复数单列任何特例**就能复现上表全部 100 格，包括最容易踩的：

| 组合 | 结果 | 原因 |
|---|---|---|
| `complex64 ⊔ int16` | `complex64` | `float32 ⊔ int16 = float32` |
| `complex64 ⊔ int32` | **`complex128`** | `float32 ⊔ int32 = float64`（24 位尾数规则） |
| `complex64 ⊔ float64` | `complex128` | `float64` 胜出 |
| `complex64 ⊔ complex128` | `complex128` | 分量 `float32 ⊔ float64 = float64` |

> 注意 `complex64 ⊔ int32 = complex128` **不是笔误**：numpy 实测确认，
> 与实数侧 `float32 ⊔ int32 = float64` 完全同源。任何「复数一律 complex128」
> 或「complex64 总是保精度」的简化都会在这四格上失真。

---

### 16.4 转换与降级（casting）

| 方向 | NumPy 行为 | clvt 裁决 |
|---|---|---|
| `complex128 → float64` | **取实部**，抛 `ComplexWarning`（可屏蔽） | **取实部，不产生警告**（差异 D-C1：CL 无警告机制；行为已在 docstring 写明） |
| `complex128 → int64` | 取实部后截断，`ComplexWarning` | 取实部 → `%coerce-int*`（截断 + 回绕，与实数一致） |
| `float64 → complex128` | 虚部补 0 | 同 |
| ufunc 显式 `dtype=float64`（如 `np.sqrt(z, dtype=float)`） | **TypeError**（safe-casting 拒绝） | **报错**：`"复数输入的结果 dtype 必须是复数类型"` |
| `np.float64(1+2j)` | **TypeError** | `vt-cast` 会取实部（差异同 D-C1） |
| `np.array([1+2j], dtype=float)` | **TypeError** | 取实部（差异 D-C2） |

统一出口：`vt-cast` / `vt-cast-fun` 对**目标为实/整**且**入参为复数**时先 `realpart`；
对**目标为复数**时按分量 `coerce`（实数入参虚部补 0）。`vt-astype` 的所有快/慢路径
都经由此出口，故 8×10 的组合矩阵无需逐格枚举。

---

### 16.5 算术与幂

| 运算 | 语义 | 备注 |
|---|---|---|
| `+ - * /` | 复数四则（分量代数） | **不做共轭**（numpy 的 `*` `/` 也不做；共轭点积必须显式 `vt-conj`） |
| 一元 `-` | 分量取负 | `vt-negative` |
| `square` | `z*z` | ≠ `|z|²`；后者是 `vt-abs` 再平方 |
| `pow` / `**` | `expt` 复数幂；指数可为实数或复数 | dtype 按 §14.3 提升 |
| `0 ** 0` | `1+0i`（numpy 实测） | 先于极点判定 |
| `0 ** 负数` | `nan+nanj`（numpy 实测） | CL 会抛条件，已拦截 |
| `reciprocal` | `1/z`；`0` → `nan+nanj` | |
| `floor_divide` / `mod` / `rem` | **无定义 → 报错** | numpy 抛 TypeError |
| 位运算 `& \| ^ ~ << >>` | **无定义 → 报错** | numpy 抛 TypeError |
| `floor/ceil/trunc/rint` | **报错** | numpy 同样不支持（复数无分量取整的全序） |
| `round` | 分量各自取整 | numpy 2.x 支持（本库 `vt-round` 保留 divisor 参数语义） |

---

### 16.6 分量与派生量

| 函数 | NumPy | clvt | 结果 dtype |
|---|---|---|---|
| `abs` | `np.abs(z)` = √(re²+im²) | `vt-abs` | complex128→**float64**，complex64→**float32** |
| `real` | `z.real`（**视图**，可写） | `vt-real` | 分量浮点；**返回副本**（差异 D-C3） |
| `imag` | `z.imag`（**视图**） | `vt-imag` | 同上；实数输入 → 全 0 |
| `conj` | `np.conj(z)` | `vt-conj` / `vt-conjugate` | 保持复数 dtype |
| `angle` | `np.angle(z[, deg])` = atan2(im, re) | `vt-angle`（`:deg`） | 实数（分量浮点） |
| `sign` | `z/|z|`（零 → 0） | `vt-signum` | 保持复数 dtype |
| `iscomplexobj` / `isrealobj` | 布尔 | `vt-iscomplexobj` / `vt-isrealobj` | 布尔 |

**差异 D-C3（real/imag 是副本而非视图）的正式理由**：
numpy 的 `z.real` 是同一缓冲的零拷贝视图（写它会改到原数组）。clvt 的 strided view
把 stride 定义在**元素**上，无法表达「同一复数缓冲区里的实数子集」（缓冲元素类型是复数），
强行支持会让 `strides/offset` 的语义分叉。因此第一轮取**副本**，并在
`vt-real/vt-imag` 的 docstring 与本节显式声明；需要写回请用
`(vt-complex new-re (vt-imag z))` 重组。

**`abs(inf + nan·i) = inf`**：与 numpy 的 hypot 语义一致（不是 nan），已单独实现。

---

### 16.7 分支切割与主值（**SBCL 与 NumPy 逐点一致**）

| 表达式 | NumPy 2.4.6 | SBCL 2.6.8 | 裁决 |
|---|---|---|---|
| `sqrt(-4+0j)` | `2j` | `#C(0 2)` | 一致（主值 + 虚部 -0.0 尊重符号） |
| `sqrt(-4-0j)` | `-2j` | `#C(0 -2)` | 一致 |
| `log(-1+0j)` | `+πi` | `#C(0 +π)` | 一致 |
| `log(-1-0j)` | `-πi` | `#C(0 -π)` | 一致 |
| `log(0+0j)` | `-inf+0j` | 需拦截（CL 抛条件） | **钳制为 `-Inf+0j`** |
| `angle(-1-0j)` | `-π` | `(atan -0.0 -1.0) = -π` | 一致 |
| `atanh(2+0j)` | `0.549…+1.571…j` | 同 | 一致 |
| `exp/asinh/acosh/asin/acos/tanh/…` | 主值 | 同 | 一致 |
| `cbrt(z)` | **TypeError** | 报错 | 一致（不支持） |

**结论**：SBCL 的复数数学实现（`sqrt/log/exp/sin/…/atanh`）在**分支切割与符号零**上
与 C99（numpy 的实现基础）逐位一致，因此 clvt 直接复用 CL 实现，
只对**极点**（`log(0)`）与**非有限值**补返回约定，不自己重写公式。

**符号零约定**：`-0.0` 的虚部是**语义信息**（决定分支），任何「规范化 `-0.0 → 0.0`」
的优化都是 bug。`%cop-sqrt` 对 `y = ±0.0` 特判以保留符号。

---

### 16.8 比较、排序与极值（**字典序**）

numpy 2.x 对复数定义了**字典序**（先实部，实部相等再比虚部），`<` / `<=` / `>` / `>=` /
`sort` / `argsort` / `min` / `max` / `argmin` / `argmax` **全部遵此语义**（实测确认）。

| 运算 | NumPy 2.4.6 | clvt | 备注 |
|---|---|---|---|
| `a == b` | 分量相等；任一分量 NaN → False | `vt-=`（`%cx-eq-flag`） | **不能**用 CL `=`：陷阱屏蔽后 `(= #C(nan 1) #C(nan 1))` 实测返回 **T** |
| `a != b` | NaN → True | `vt-/=` | |
| `a < b` | 字典序；NaN 参与 → False | `vt-<`（`%clt`） | 返回 `:int8` 的 1/0（§3.1） |
| `a <= b` | 字典序 ≤ | `vt-<=` | |
| `maximum/minimum` | 字典序；NaN 传播 | `vt-maximum/minimum` | `%cop-max/min` |
| `fmax/fmin` | 字典序；忽略 NaN | `vt-fmax/fmin` | `%cop-fmax/min` |
| `sort` | 字典序，NaN 排末尾 | `vt-sort` | `%cx-sort-axis` + `%cx-sort-less` |
| `argsort` | 同上，返回下标 | `vt-argsort` | 稳定性：`stable-sort` |
| `median` | 排序后取中位（偶数取两中位均值） | `vt-median` | 复数均值不是实数 |

**为什么必须专列**：字典序不是「任意一致的全序」——`1+9j < 2+0j` 为真（实部优先），
而按模长比较则为假。任何用 `abs` 或分量加权做比较的实现都会与 numpy 分歧。

---

### 16.9 NaN / Inf 语义

| 判定 | 规定 |
|---|---|
| `isnan(z)` | **任一分量**为 NaN 即为真（`nan+1j` 与 `1+nanj` 都是 NaN） |
| `isinf(z)` | 任一分量为 ±Inf 即为真 |
| `isfinite(z)` | 两个分量都有限才为真 |
| 算术传播 | 任一分量 NaN → 整个结果的对应该分量含 NaN（IEEE 逐分量） |
| `sum` / `prod` | NaN 传播（结果为 NaN 复数） |
| `amax` / `amin` | **NaN 胜出**（首个 NaN 元素成为结果，与 numpy 一致）；`fmax/fmin` 忽略 NaN |
| `sort` | NaN 一律排在末尾（含 `nan+nanj`） |
| `argmax/argmin` | NaN 元素的实部视为「最大」，即首个 NaN 的下标胜出 |
| `==` | `nan+0j == nan+0j` → **False** |
| `mean(空)` | `nan+nanj`（不是 0） |

---

### 16.10 归约与统计

| 函数 | NumPy | clvt | 结果 dtype |
|---|---|---|---|
| `sum` | 保 dtype（complex64→complex64） | `vt-sum` | 输入复数 dtype |
| `prod` | 同上 | `vt-prod` | 同上 |
| `mean` | 同上 | `vt-mean` | 同上（空 → `nan+nanj`） |
| `var` | `mean(|x-μ|²)` → **实数** | `vt-var` | **复数 → 分量浮点**（complex128→float64） |
| `std` | `sqrt(var)` → **实数** | `vt-std` | 同上 |
| `argmax/argmin` | `intp`（64 位平台 = int64） | `vt-argmax/argmin` | `:int64`；多轴不支持（与 numpy 同） |
| `cumsum/cumprod` | 保 dtype | `vt-cumsum/vt-cumprod` | 同上 |
| `norm` | `sqrt(Σ|z|²)` → **实数** | `vt-norm` | 复数 → 分量浮点 |
| `median` | 字典序中位 | `vt-median` | 复数 |
| `trace` | 对角线之和 | `vt-trace` | 复数 |
| `percentile/quantile` | **TypeError**（"a must be an array of real numbers"） | 报错 | — |

**空归约**（与 §5.5 一致，复数同样适用）：
`sum → 0+0j`、`prod → 1+0j`、`mean → nan+nanj`、
`amax/amin → ValueError`（"zero-size array to reduction operation maximum which has no identity"）。

**陷阱提示**：`vt-var` **不能**写成 `vt-mean(vt-square(x))` —— 那会得到复数。
正确形态是 `mean(square(abs(x - μ)))`，已封装在 `vt-var` 内。

---

### 16.11 线性代数

| 函数 | 第一轮状态 |
|---|---|
| `vt-matmul` / `vt-@` | ✅ 专用内核（1d@1d、2d@2d、2d@1d、1d@2d、批量 `...ij,...jk->...ik`） |
| `vt-dot` | ✅ 同 matmul 语义（**不取共轭**；共轭点积 = `(vt-dot (vt-conj a) b)`） |
| `vt-outer` | ✅ 分量乘法 |
| `vt-trace` | ✅ `sum(diagonal)` |
| `vt-einsum` | ❌ 报错：dtype 白名单只含 4 种实数类型，复数缓冲会错配 → 第二轮 |
| `vt-det/inv/solve/svd/eig/qr/lu/pinv/cholesky/cond` | ❌ 报错：需共轭转置与酉变换 → 第二轮 |

**为什么 matmul 不走 einsum**：`vt-einsum` 的 `supported-types` 是
`(:float64 :float32 :int64 :int32)`，其结果缓冲区按这 4 种分配。复数输入会得到
`double-float` 缓冲并在写回时报 `"is not of type DOUBLE-FLOAT"`。
第一轮为 matmul 家族写**独立专用内核**（`%cx-matmul`，先 `vt-contiguous` 落地再三重循环），
语义与 einsum 的矩阵乘完全一致；einsum 复数路径留到第二轮（需要把 dtype 白名单改成
「按 dtype 分配」而不是「按 4 种硬编码」）。

---

### 16.12 打印格式

| 场景 | NumPy | clvt |
|---|---|---|
| 数组元素 | `[1.+2.j 3.-4.j]` | `[1.0+2.0j 3.0-4.0j]` |
| 标量 | `(1+2j)` | `1.0+2.0j`（`print-object` 内联） |
| NaN 分量 | `nan+nanj` | `nan+nanj` |

**格式规定**：`<实部><符号><|虚部|>j`，实部/虚部沿用本库既有浮点打印风格
（`~,6f` 定精度去尾零、至少保留一位小数）。与 numpy 只差「尾随 `.0`」的写法
（numpy 省略 `1.`→`1.`，clvt 保留 `1.0`），**这是刻意保留的既有风格**，
不为了对齐 numpy 而改动全部浮点打印（差异 D-C4）。

---

### 16.13 错误契约（L2 底线：确定性报错）

| 触发 | 报错文本特征 |
|---|---|
| 复数入参 + 实数 `:dtype` | `"复数输入的结果 dtype 必须是复数类型"` |
| `vt-var/vt-norm` + 复数 `:dtype` | `"结果 dtype 必须是实数类型"` |
| 未支持函数 + 复数 | `"~a: 第一轮不支持复数输入（…）。见 CONVENTIONS §14.11"` |
| `amax/amin` 空归约 | `"zero-size array to reduction operation maximum which has no identity"` |
| `argmax/argmin` 空序列 | `"attempt to get argmax of an empty sequence"` |
| `argmax/argmin` 多轴 | `"不支持多轴（对标 numpy：arg 归约只接受单个整数 axis）"` |

**禁止**：静默把复数当实数算（例如 `vt-sqrt` 走实数路径返回 float64）——
这类「静默降级」是最危险的失败模式，一律改为报错或返回正确复数结果。

---

### 16.14 执行层：声明策略与实测性能（v1.1 优化后）

#### 16.14.1 声明分级（**不要无脑 `(speed 3) (safety 0)`**）

| 层级 | 位置 | 声明 | 依据 |
|---|---|---|---|
| **边界 / API** | `vt-abs`、`vt-sum`、`vt-sort` 等用户入口 | 默认安全级（不写 optimize） | 形状/dtype/`:out` 契约检查、`ensure-vt` 归一、错误信息都在这一层；关掉检查只会把错误变成静默错值 |
| **普通内核** | `%cx-*` 里的循环（`%cx-dispatch` 生成的、`%cx-abs-run`、`%cx-matmul-run`…） | `(speed 3) (safety 1)` | 索引由 `offset/size/stride` 推出，理论可证不越界；但保留边界检查实测代价 < 5%，换「形状算错立刻报错」而不是踩内存 |
| **宏展开的逐元素循环** | `map-reduce.lisp` 的 `%inline*-loop` / `%mask*-loop` | `(speed 3) (safety 0)`（既有约定） | 索引由循环上界逐元素推进（`p` 与 `op` 同步 +1），编译器能完全证明；且已由 `simple-array` 类型声明兜住 |
| **禁用** | 任何接受用户 stride/shape 的通用路径 | 不得 `safety 0` | 用户传错 stride 时必须是 Lisp 条件，不是段错误 |

#### 16.14.2 三条实测得出的硬规则

1. **`vt-fast-map` / `vt-fast-type` 的复数分支靠白名单控制**
   （`+cx-inline-ops+`）：只有「语义与元素类型无关」的算子才可进白名单，
   否则 200+ 展开点会复制出大量无效循环体（历史：8×8 全展开直接 Heap exhausted）。
2. **`declaim (inline …)` 必须在 `defun` 之前**。SBCL 只在定义时可见 inline
   声明才记录可内联展开；事后补声明不生效。实测：复数 `vt-<` 1M 元素
   84ms → **3ms**（`%cx-lt-flag`/`%clt`/`%c-lex-compare` 全链内联后与手写循环等价）。
3. **`with-float-safe` 只能在外层做一次**。`vt-float-nan-p` 每次都屏蔽/恢复陷阱
   （~200ns）；内核里每元素判 2~4 次 → 1M 元素白付 400~800ms。
   内核统一改用 `%cx-nan-p` / `%cx-inf-p` / `%cx-finite-p`（免屏蔽快判），
   并要求调用方（`vt-map`/`vt-fast-map`/`%cx-reduce`/`%cx-scan`/`%cx-sort-axis`）
   已处于 `with-float-safe` 内。边界函数仍用带屏蔽版本。

#### 16.14.3 实测（1M 元素；`test/complex-bench.lisp`，SBCL 2.6.8）

| 算子 | 优化前 | 优化后 | 倍数 | 相对 float64 |
|---|---:|---:|---:|---:|
| `add` | 135 ms | **7 ms** | 19× | 1.8× |
| `mul` | 157 ms | **7 ms** | 22× | 2.3× |
| `div` | 184 ms | **8 ms** | 23× | 2.0× |
| `sqrt` | 453 ms | **20 ms** | 23× | 4.0× |
| `exp` | 73 ms | **46 ms** | 1.6× | 5.1× |
| `abs` | 889 ms | **26 ms** | 34× | 6.5× |
| `sum` | 92 ms | **1 ms** | 90× | 1.0× |
| `sum(axis=0)` 1000² | 96 ms | **1 ms** | 96× | 0.3× |
| `cumsum` | 288 ms | **6 ms** | 48× | 2.0× |
| `sort` | 2316 ms | **136 ms** | 17× | 3.2× |
| `compare <` | 778 ms | **3 ms** | 260× | 1.5× |
| `isnan` | 402 ms | **2 ms** | 200× | 2.0× |
| `matmul` 256² | 1757 ms | **39 ms** | 45× | 39×（实数走 SIMD，复数无） |
| `argmax` | 912 ms | **5 ms** | 180× | 5.0× |

**剩余差距的来源**（不是实现缺陷，是表示与算法固有）：
- 复数元素在 SBCL 里是**堆对象**（`(complex double-float)` 按值装入槽位），
  每次 `aref` 都要构造/解构 → 算术下限约为实数的 2×；
- `sqrt`/`abs` 含 `hypot` 与除法 → 4~6× 属正常（numpy 亦然）；
- `matmul` 实数有 SIMD/并行内核，复数目前是标量三重循环 → 39× 是**下一步**（见 16.17）；
- `exp` 走 CL 的复数 exp（内部含 sin/cos），5× 属正常。

---
### 16.15 与 numpy 的已知差异清单（全部为**有意**保留）

| 编号 | 差异 | 影响面 | 处置 |
|---|---|---|---|
| D-C1 | `complex → real` 转换不产生 `ComplexWarning` | 仅提示信息 | 文档声明；行为与 numpy（取实部）一致 |
| D-C2 | `np.array([1+2j], dtype=float)` numpy 抛 TypeError，clvt 取实部 | 构造 | 文档声明（沿用 `vt-cast` 的「安全转换」立场） |
| D-C3 | `vt-real/vt-imag` 返回**副本**（numpy 返回视图） | 写回语义 | 文档声明；用 `vt-complex` 重组 |
| D-C4 | 打印保留 `1.0+2.0j`（numpy 为 `1.+2.j`） | 显示 | 沿用本库浮点风格 |
| D-C5 | 复数逐元素算子无 SIMD/内联特化 | 性能 | 第二轮 |
| D-C6 | `einsum` / LU 家族 / 百分位 不支持复数 | 功能面 | 显式报错，第二轮 |

---

### 16.16 测试约定（本章的验收方式）

1. **差分对照（强制）**：`test/gen_complex_expected.py` 用**本机 numpy** 生成期望值
   （`test/complex-expected.sexp`），Lisp 侧逐格比对值与 dtype/shape。
   **禁止手写期望值**——这是 §10.4 在复数上的具体化。
   新增复数语义时必须先在生成器里加用例，再让实现追上。
2. **dtype 提升 100 格**：`test/complex-test.lisp` 内嵌整张表，逐格断言。
3. **分支切割 8 点**：`sqrt(±0i)`、`log(±0i)`、`log(0)`、`angle(±0i)`、`atanh(2)`。
4. **契约三底线**（§10.1）：`:out` 形状/dtype/可写；非连续视图；别名 `z += z`。
5. **未支持即报错**：每个第一轮不支持的函数都有 `expect-error` 断言
   （防止未来"悄悄支持了一半"）。
6. **实数回归网**：加法/比较/排序（NaN 末尾）/sum dtype/mean/floor 六条，
   保证 `complex.lisp` 的分派层没有破坏实数路径。

运行方式：
```bash
bash test/run-tests.sh --suite complex-test        # 走统一入口（136 断言）
python3 test/gen_complex_expected.py               # 重新生成期望值（numpy 升级后必做）
```

---

### 16.17 第二轮路线图（本章的未完成项）

| 优先级 | 项目 | 关键改动 |
|---|---|---|
| P0 | `vt-einsum` 复数 | 把 dtype 白名单换成「按 `vt-promote-type` 分配」，结果缓冲用 `vt-dtype->lisp-type`（`inner/tensordot` 已绕开它实现，见 §16.18） |
| P0 | 复数归约的 `axis` 元组 + `keepdims` 全路径覆盖测试 | 复用 `%cx-reduce-values` 的多轴路径，补 3D 用例（`nan` 族已全 axis 覆盖，普通归约仍缺 3D 用例） |
| P1 | 复数内联/SIMD 特化 | `%vt-inline*-fast` 增加 `(complex double-float)` 两格；`simd-matmul` 复数 GEMM（**标量形态已对齐实数线，见 §16.27**；SIMD 仍待做） |
| P1 | LU/SVD/特征值 | 需要共轭转置（`vt-ctranspose`）与酉变换；`vt-det` 复数版（实测 numpy det 为复数） |
| P2 | `real/imag` 真视图 | 需引入「分量视图」概念（`data` + 分量偏移），会动物理层，谨慎评估（同 D-C7 的根因） |
| ~~P2~~ | ~~`.npy` 复数互操作~~ | **已完成**（§16.18：`<c16`/`<c8` 两种格式双向互操作，含 NaN payload） |
| P2 | `vt-npy` 打印/序列化的 `ComplexWarning` 等价物 | 引入可选告警通道 |

---

### 16.18 数据搬运层（第二轮进化）：转换 / 拼接 / 剪切 / 集合 / 缺失值 / 缩并

第一轮（16.1–16.14）解决的是**逐元素算术与归约**；这一轮补的是**结构层**。
落地文件 `src/complex-ops.lisp`（在 `src/complex.lisp` 之后加载），
同样遵守「实数路径零改动 + 只对含复数的调用分派」。

#### 16.18.1 本轮修复的「静默错答案」

比「不支持」更危险的是「看起来正常但错」。本轮实测发现 5 处：

| 函数 | 修复前行为 | numpy 语义 | 根因 |
|---|---|---|---|
| `vt-nanmean` | 返回**实数**（实部均值） | 复数 | 内部 `vt-astype → :float64` 按 §16.5 取实部 |
| `vt-nanvar` / `vt-nanstd` | 数值错误（按实部算） | 实数 = mean(\|x−mean\|²) | 同上 |
| `vt-nancumsum` / `vt-nancumprod` | **虚部被清零** | 保复数（NaN→0/1 后累积） | `%e3-cumulative-skip-nan` 固定 `vt-astype :float64` |
| `vt-vdot` | 不取共轭（= `vt-dot`） | **第一个操作数取共轭** | 第一轮漏做 |
| `vt-cumsum/cumprod` 的 `axis=None` | 形状保持原样（2-D 进 2-D 出） | **展平为 1-D** | `%cx-scan` 沿用了 `(vt-shape in)` |

另外 `vt-repeat` / `vt-unique` / `vt-searchsorted` / 集合运算 / `vt-nan[sum|prod|max|min|argmax|argmin|median]` /
`vt-convolve` / `vt-inner` / `vt-tensordot` / `vt-nan-to-num` / `vt-view(:dtype)` 这些
此前是**确定性报错**（类型声明里写死 real 或只列 8 种实数元素类型），本轮补齐。

#### 16.18.2 语义总表（numpy 2.4.6 实测）

**转换**

| 操作 | 复数语义 |
|---|---|
| `vt-nan-to-num` | 实部/虚部**各自**替换；`nan` 缺省 0，`posinf/neginf` 缺省为该分量精度的 ±最大有限值；结果 dtype 不变 |
| `vt-view` `:dtype` | `complex128 ↔ float64`、`complex64 ↔ float32`；`shape` 省略时**只缩放最后一维**（(2,3)→(2,6)） |
| `vt-repeat` | 标量 / 列表 repeats，`axis` 可为 nil 或单轴；dtype 保持 |
| `vt-astype` | 复数→实数/整数取实部（§16.5，仍是 astype 的既有约定） |

**拼接 / 切分**：`concatenate/concat/stack/vstack/hstack/dstack/column-stack/append/block/split/hsplit/vsplit/dsplit`
全部按 `vt-promote-type` 提升（`c128+c64→c128`、`c128+f64→c128`），切分返回的是视图。

**剪切**：`slice`（步长/负步长/负索引）、`ref`、`narrow`、`take`、`take-along-axis`、
`put`、`put-along-axis`、`compress`、`extract`、`where`、`choose`、`select`、`flip/fliplr/flipud`、
`roll/rollaxis`、`pad`、`tile`、`rot90`、`resize`、`trim-zeros` 均可用。
（`take-along-axis` / `put-along-axis` 仍要求 indices 与数组同秩，这是**库既有约束**，
与复数无关，见 §16.15 D-C8。）

**集合与搜索**

| 函数 | 语义要点 |
|---|---|
| `vt-unique` | 展平 → 字典序排序 → run-length；**多个 NaN 折叠为一个**（排末尾）；支持 `return-index/inverse/counts` |
| `union1d / intersect1d / setdiff1d / setxor1d` | 全部建立在 `unique` + 哈希判定上，结果升序去重 |
| `in1d / isin` | 哈希键 `%cx-key`：NaN 折叠为同一哨兵、其余规范化为 double-float 复数（`1.0s0` 与 `1.0d0` 同键） |
| `searchsorted` | 字典序二分；`:left` = 第一个 ≥ v，`:right` = 第一个 > v |
| `digitize` | **不支持**（numpy 亦报 `x may not be complex`），给出确定性错误 |

**缺失值（nan 族）**

| 函数 | 全 NaN 切片 | dtype |
|---|---|---|
| `nansum` / `nanprod` | `0` / `1`（单位元） | 复数 |
| `nanmax` / `nanmin` | `NaN+0j`（**不报错**） | 复数 |
| `nanargmax` / `nanargmin` | **报错** `All-NaN slice encountered` | int64 |
| `nanmean` | `NaN+NaNj` | 复数 |
| `nanvar` / `nanstd` | `NaN` | **实数** |
| `nanmedian` | `NaN+0j` | 复数 |

实现采用「掩码 + 单位元替换 + 既有复数归约内核」的向量化写法：
`sum/prod/mean` 把 NaN 换成 0/1 后直接调 `vt-sum/vt-prod`；
`max/min` 换成 `(-inf,-inf)` / `(+inf,+inf)` 哨兵（字典序下必被淘汰），
再用 `count==0` 的掩码补 `NaN`。这样一次覆盖所有 `axis/keepdims` 组合，
且复用第一轮已优化的内核，不写第二套 fiber 循环。

**缩并 / 卷积**

| 函数 | 语义 |
|---|---|
| `vt-correlate` | `c[k] = Σ a[n+k]·conj(v[n])` —— **第二操作数取共轭**（numpy 的定义） |
| `vt-convolve` | `c[k] = Σ a[n]·v[k−n]` —— **不取共轭**；实现为对 `flip(v)` 做不共轭相关 |
| `vt-inner` | 最后一维收缩，**不取共轭**（与 numpy 一致；共轭点积是 `vt-vdot`） |
| `vt-tensordot` | `transpose（收缩轴移末尾/开头）→ reshape 2D → 复数 matmul → reshape` |
| `vt-einsum` | **仍不支持复数**（dtype 白名单只含 4 种实数类型），见 §16.17 路线图 |

> `correlate` 与 `convolve` 的共轭差异在实数上不可观测，**只有复数能测出来**。
> 第一版实现漏了这一点，差分测试（`correlate-*` / `convolve-*` 用例）当场抓到。

#### 16.18.3 已知差异（D-C7 起）

| 编号 | 差异 | 原因与影响 |
|---|---|---|
| **D-C7** | `vt-view` 的 dtype 重解释返回**独立副本**（numpy 是共享内存的视图） | Common Lisp 的 `make-array :displaced-to` 要求元素类型一致，无法跨类型别名同一块内存。数值完全等价，但「改视图即改原数组」不成立——已在 docstring 与本节显式声明 |
| **D-C8** | `vt-take-along-axis` / `vt-put-along-axis` 要求 indices 与数据**同秩** | 库既有约束（numpy 会广播 indices），与复数无关，未在本轮改动 |

#### 16.18.4 声明策略（延续 §16.14.1）

本轮新增的内核只有两处带 `optimize` 声明，且都不是 `safety 0`：

| 位置 | 声明 | 理由 |
|---|---|---|
| `%cx-correlate-full` 的卷积循环 | 继承 `%cx-dispatch` 的 `(speed 3) (safety 1)` | 索引由 `k/j` 的上下界决定，可证不越界，但保留检查 |
| `%cx-unique-sorted` / `%cx-searchsorted-run` / `%cx-nan-arg-run` | 同上 | 同上 |

其余（`nan` 族的向量化实现、集合过滤、`tensordot`）全部是**组合既有算子**，
没有手写热循环，因此不需要任何 `optimize` 声明 —— 这也印证了 §16.14.1 的分级原则：
**能用既有内核组合出来的，就不要新写循环**。

#### 16.18.5 测试

- `test/complex-ops-test.lisp`：**128 条断言**
  = 97 项 numpy 差分用例（转换 4 / 拼接 17 / 剪切 17 / 集合 11 / nan 18 / 缩并卷积 9 / dtype 视图 11 / 其它 10）
  + 31 条契约断言（dtype 提升、切片别名、`:out` 硬契约、未支持组合报错、NaN 折叠、
  c64 精度保持、两种持久化格式往返、7 条实数回归网）。
- 期望值全部由 `test/gen_complex_ops_expected.py` 从本机 numpy 2.4.6 实测生成
  （`test/complex-ops-expected.sexp`），符合 §10.4「禁手写期望值」。
- 已接入 `test/run-tests.sh`（`SUITES` + `DEFAULT_ORDER`）。

---

#### 16.18.6 可移植性：浮点字面量精度取决于**读取环境**（真实缺陷复盘）

**症状**（用户环境 SBCL 2.6.9.446 / numpy 2.5.0，本机 2.6.8/2.6.9 release 无法复现）：

```
差分对照完成：PASS=97 FAIL=0
--- 契约断言 ---
Unhandled TYPE-ERROR ... The value #C(0.0 0.0) is not of type (COMPLEX SINGLE-FLOAT)
                            when binding SB-C::ITEM
```

**根因链**（三段，缺一不可）：

1. 用户 `.sbclrc` 里把 `*read-default-float-format*` 设成了 `double-float`
   （科学计算用户的常见习惯）。CL 标准只规定默认值为 `single-float`，
   源码里的**裸浮点字面量**（`0.0`、`0.5`、`180.0`）精度由**读取时**该变量决定。
2. 于是 `(complex 0.0 0.0)`（complex64 的零值）被读成 `(complex 0.0d0 0.0d0)`。
   把 double 复数填进 `(simple-array (complex single-float) (*))`：
3. SBCL 的 `vector-fill` 为**每种元素类型**预装了带类型声明的值转换器
   （`src/code/seq.lisp` 的 `init-fill-bashers`）：

   ```lisp
   (cons #'basher
         (lambda (sb-c::item)
           (declare (type ,et sb-c::item))   ; ← 报错里的 SB-C::ITEM 就是他
           ,value-transform))
   ```

   它在 `(safety 0)` 之外执行，于是抛 TYPE-ERROR。**这不是 SBCL 的 bug**：
   往 c64 数组塞 double 本来就是错的，只是错误信息里的 `#C(0.0 0.0)`
   （double 在 `*read-default-float-format*` = double 时的打印形式）
   误导性极强，看起来像「single 值被拒绝」。

> 更危险的是旁路：`fill-float` 等 `(safety 0)` 路径**不报错**，
> 直接把 double 的位模式硬写进 c64 数组 —— 静默内存错乱。

**修复（三层）**

| 层 | 改动 | 作用 |
|---|---|---|
| 系统定义 | `clvt.asd` 的 `:around-compile` 绑定 `*read-default-float-format*` = `single-float` | ASDF 包裹 `compile-file`/`load`，**read 在其动态范围内** → 无论用户 init 怎么设，库都按约定读取 |
| 源码 | 复数/单精度关键路径的裸字面量显式加 `s0`：`(complex 0.0s0 0.0s0)`、`0.5s0`、`2.0s0`、`180.0s0`、`0.6931472s0` | 即使有人直接 `load` 源文件（绕过 ASDF），语义也不漂移 |
| 测试 | `run-tests.sh` 的两处 `sbcl` 调用插入 `--eval "(setf *read-default-float-format* 'single-float)"`；`complex-ops-test` 增加 3 条**读取格式哨兵**断言 | 测试期望值不被用户 init 改成 double 而误报；哨兵在回归时立刻暴露复发 |

**约定**（写进本节，后续新增代码必须遵守）：

> **源码里的裸浮点字面量一律视为 single-float。**
> 需要 double 时必须显式写 `d0` 后缀（`0.0d0`、`1.0d-9`）；
> 需要 single 时**推荐**写 `s0`（尤其复数零值、numpy 对齐常量）。
> 任何依赖 `*read-default-float-format*` 当前值的写法都是缺陷。

**用户侧提醒**：修好后**必须强制重编译**，否则会继续加载用 double 编译的旧 fasl：

```lisp
(asdf:load-system :clvt :force t)
;; 或删除 ~/.cache/common-lisp/**/clvt/ 后重试
```

（`clvt.asd` 本身变了，正常 `asdf:load-system` 也会重编；但若你的 ASDF 版本
对 `:around-compile` 变化不敏感，用 `:force t` 最稳。）

### 16.19 线性代数与算子补齐（第三轮进化）

前两轮把「逐元素」和「结构层」做完了，但**复数线代整块仍是硬报错**，
另有一批算子在复数输入下**静默只算实部**。本轮收尾。

落地：
- `src/complex-linalg.lisp`（新增）：LU / det / solve / inv / Cholesky /
  QR / SVD / 厄米特特征值 / pinv / lstsq / matrix-rank / matrix-power
- `src/complex-ops.lisp` 第 12、13 节：补齐 numpy **支持**的算子 +
  对 numpy 也拒绝的算子给出确定性错误
- `src/linalg.lisp`：`vt-einsum` 放开复数白名单

#### 16.19.1 复数线代的唯一真分叉：转置是共轭转置

写复数线代时只有**一条**必须刻在脑子里的规则：

> 实数线代里所有的 `A^T`，在复数里都是 `A^H = conj(A^T)`。

它同时决定四件事，任一处写错都会得到「能跑、数值看起来像、但数学上错」的结果：

| 位置 | 写成 `A^T` 的后果 |
|---|---|
| `cov = X·X^T` | 得到对称但不厄米特的矩阵（对角虚部非零），`corrcoef` 跟着错 |
| Cholesky `A = L·L^T` | 分解不成立；且 `:upper` 返回 `L^T` 而不是 `L^H` |
| `pinv = V·Σ⁺·U^T` | 得到 `A⁺` 的共轭，`A·A⁺` 不再是厄米特投影算子 |
| 特征值 `A·V = V·Λ` | 只有实矩阵碰巧成立 |

`correlate` 与 `convolve` 的共轭差异同理（§16.18 已记录）：实数上不可观测，
只有复数能测出来。

#### 16.19.2 算法与关键公式

**LU（部分选主元）**：选主元用 `|a|`（复数无全序）。消元公式与实数完全一致
——乘除法在复数域封闭，不需要额外共轭。

**Cholesky（厄米特）**：

```
l_jj = sqrt( a_jj - Σ_{k<j} |l_jk|² )        ← 被开方数必须为实数且 > 0
l_ij = ( a_ij - Σ_{k<j} l_ik·conj(l_jk) ) / l_jj
```

少了 `conj` 会得到满足 `A = L·L^T` 的「L」，但那不是厄米特分解。

**QR（复 Householder）**：`H = I - β·v·v^H`，`v = x - α·e₁`，
`α = -phase(x₀)·‖x‖`（让 α 与 x₀ 反相，避免相消）。两个方向的内积**不对称**：

- 左乘 `H·R`：内层用 `conj(v)`，外层用 `v`
- 右乘 `Q·H`：内层用 `v`，外层用 `conj(v)`

> **约化 QR 的陷阱**（本轮实测踩到）：Q 只有 `k = min(m,n)` 列，而反射器
> 的支撑长度是 `m-i`，通常伸到第 k 列之外。若沿用「`Q ← Q·H_i` 顺次右乘」，
> 内积会去读 Q 不存在的列 —— 方阵与宽矩阵因为 Q 恰好是完整 `m×m` **碰巧正确**，
> 只有 tall（`m > n`）是错的。正确写法是
> `Q = H_0·H_1···H_{k-1}·I_{m×k}`，从 `i = k-1` **递减**依次**左乘**
> （每次只改行 i..m-1，截断后都存在）。

**SVD（单边 Jacobi）**：直接对 A 的列做酉旋转使之两两正交，不形成 `A^H A`，
因此没有条件数平方问题。复旋转 `J = [[c, s],[-conj(s), c]]`（c 实）的推导：

```
α = ‖a_p‖², β = ‖a_q‖² (实), γ = a_p^H a_q (复)
要求新内积为 0：  t(α-β) + γ - t²·conj(γ) = 0,   t = s/c
令 γ = |γ|e^{iφ}, u = (α-β)/|γ|, 代换 t = e^{iφ}·w 后退化为实二次式：
    w² - u·w - 1 = 0   →   w = -2 / (u + sign(u)·√(u²+4))    （|w| ≤ 1 的小根）
    t = e^{iφ}·w,   c = 1/√(1+w²),   s = t·c
```

> 直接把实数的 `t = sign(θ)/(|θ|+√(1+θ²))` 套上来，在 γ 非实时**不会收敛**。

**厄米特特征值（Jacobi）**：零化 2×2 块与 SVD 的旋转同源（同一个二次式），
但变换是 `A ← U^H·A·U`，列变换用 `(c, conj(s))`、行变换用 `(c, s)` 且第二行
取 `conj(s)` —— 这处不对称正是 `U^H` 与 `U` 的区别。

**pinv / lstsq**：`A⁺ = V·Σ⁺·U^H`。注意 `A⁺` **一般不是厄米特矩阵**
（只有 A 厄米特时才如此）；它的正确性质是 Moore–Penrose 四条。

#### 16.19.3 厄米特特征值：对角虚部必须清零

`vt-eig` 的契约与实数版一致（numpy `eigh` 的 UPLO 语义）：不校验厄米特，
只用一个三角并用**共轭**镜像到另一三角，再**把对角元的虚部清零**。

> 不清零的后果（本轮实测）：对 `[[2+1j, 1], [1-1j, 3+2j]]` 会返回
> `(3.925, 1.075)`，而 numpy 返回 `(4, 1)`。原因是 Jacobi 用
> `(realpart diag)` 当特征值候选、却让复对角元继续参与旋转 —
> 迭代仍「收敛」，只是收敛到错的值。LAPACK 的 `zheevd` 同样直接按实部使用。

#### 16.19.4 本轮修掉的「静默丢虚部」（第二轮同类，共 8 处）

| 函数 | 修复前 | numpy 2.4.6 | 根因 |
|---|---|---|---|
| `vt-cov` / `vt-corrcoef` | 返回 **float64** | 复厄米特矩阵 | `%e3-as-1d-float-transform` 首行 `ensure-vt :dtype :float64` |
| `vt-expm1` / `vt-log1p` | 返回 **float64** | 复数 | `%float-map` 只认实数元素类型 |
| `vt-nanpercentile` / `vt-nanquantile` | 返回 **float64** | **报错** | 内部 `astype :float64` 取实部 |
| `vt-interp` | 返回 **float64** | **报错** | 同上 |
| `vt-fromiter` | 返回 **float64**（虚部清零） | 复数 | 默认 `:dtype :float64` |
| `vt-nextafter` / `vt-spacing` / `vt-logaddexp` / `vt-hypot` / `vt-copysign` / `vt-signbit` / `vt-deg2rad` / `vt-rad2deg` / `vt-bincount` | 静默算实部 | **报错**（ufunc 不支持） | 无复数分派 |
| `vt-matrix-rank` | `astype :float64` 后消元 | 基于奇异值 | 取实部 |
| `vt-scale`（公共函数） | complex 数组 × 实数标量 → 结果 dtype 判成 `:float64`，写缓冲时 `%DOUBLE-FLOAT` 报错 | 保持 complex dtype | dtype 推断只列了 4 种实数类型 |

全部改为「numpy 支持的实现，numpy 拒绝的确定性报错」。

#### 16.19.5 本轮新增/放开的算子

**新支持（numpy 支持复数）**

| 函数 | 语义要点 |
|---|---|
| `vt-expm1` | `\|z\|<0.5` 走级数（12 项），否则 `exp(z)-1`；避免灾难性抵消 |
| `vt-log1p` | 同上，交替级数；`z=-1` 交给极点（`-inf+0i`） |
| `vt-sinc` | `sin(πz)/(πz)`；`\|z\|<1e-8` 用 `1-(πz)²/6` |
| `vt-sign` | `z/\|z\|`（投影到单位圆），`sign(0)=0` |
| `vt-clip` / `vt-clamp` | **字典序**整元素夹取，不按分量（numpy 的 clip 是 `minimum(maximum(a,lo),hi)`） |
| `vt-standardize` | 均值复数、标准差实数（`\|x-mean\|²` 的均值开方），结果保复数 |
| `vt-cov` / `vt-corrcoef` | `X·X^H`，厄米特对称化；对角为实数 1 |
| `vt-fromiter` | dtype 缺省时推断（含复数 → complex128）；显式实数 dtype 遇复数报错 |
| `vt-einsum` | 白名单放开 complex64/complex128，结果 dtype 由输入推导；走通用累加路径（matmul 快路径仍只服务实数） |
| `vt-lu/det/solve/inv/cholesky/qr/svd/eig/pinv/lstsq/matrix-rank/matrix-power/cond` | 见 16.19.2 |

**新拒绝（numpy 同样拒绝）**：`nextafter` `spacing` `logaddexp` `hypot`
`copysign` `signbit` `deg2rad` `rad2deg` `bincount` `histogram`
`nanpercentile` `nanquantile` `interp`（+ 已有的 `floor` 族、`mod`、位运算、
`atan2`、`even-p/odd-p`、`percentile/quantile`、`digitize`、神经网络激活）。

#### 16.19.6 已知差异（D-C9 起）

| 编号 | 差异 | 原因与影响 |
|---|---|---|
| **D-C9** | `vt-histogram` 对复数**报错**，numpy 会「跑通」 | numpy 对复平面做线性插值分箱（bins 也是复数），语义无意义；本库选择确定性拒绝。若确需沿复平面分箱，请自行 `realpart`/`imagpart` 后分别直方图 |
| **D-C10** | `vt-eig` 是 **eigh 语义**（厄米特），不是 numpy 的通用 `eig` | 与实数版一致（实数版也只支持对称）。通用非厄米特征值需要复 QR 迭代（Hessenberg + 双位移），不在本轮范围；非厄米输入与 numpy `eigh` 一样是 GIGO |
| **D-C11** | `vt-svd` 的 U/V 与 numpy **逐元素不同**（重构等价） | 奇异向量带相位自由度：`u_j → e^{iθ}u_j, v_j → e^{iθ}v_j` 仍是同一分解。断言应使用重构/正交恒等式，不要逐元素比对 |
| **D-C12** | `vt-qr` 的 R 与 numpy **逐元素不同**（重构等价） | Householder 的 α 取 `-phase(x₀)‖x‖`（numpy 用 LAPACK 的符号约定），R 的相位随之不同 |
| **D-C13** | `vt-lstsq` 的 residual 是 **2-范数**（numpy 是平方和） | 库内既有契约（`vt-norm`），保持一致；`sqrt(numpy_res)` 即得 |

#### 16.19.7 声明策略（延续 §16.14.1）

本轮新增内核的声明**只有一档**：

| 位置 | 声明 | 理由 |
|---|---|---|
| `%clu-factor!` `%clu-subst!` `%cchol-factor!` `%cqr-*` `%csvd-jacobi!` `%ceig-jacobi!` | `(speed 3) (safety 1)` | 索引由 `ld/m/n` 与循环上界推出，可证不越界；保留检查的成本在 O(n³) 算法里可忽略（<2%），而形状算错会立刻报错而不是静默踩内存 |
| 所有对外入口（`vt-svd` 等） | 不写 `optimize` | 契约检查（方阵、`:out`、正定性）与错误信息都在这一层 |

**没有**任何新的 `safety 0` —— 复数线代的索引组合（`ld` ≠ 行数）比实数更容易写错，
本轮实际因 `ld` 写错踩了 3 次（`vt-lu` 非方阵、QR 的 Q、SVD 的 U），
全靠 `safety 1` 的边界检查当场报错才定位到。

#### 16.19.8 第四轮补齐：结构算子收尾

差分对照与结构恒等式之外，第四轮又用「全导出符号逐个试调」的方式扫了一遍
剩余缺口。结论：**大部分结构算子本来就是 dtype 无关的**（eye/zeros/triu/
reshape/tile/pad/insert/delete/block/meshgrid/select/choose/compress/
where/broadcast/copy-to!/write/save-load/array-equal/isclose… 全部直接可用），
真正缺的只有三处，且都是「8 路元素类型特化漏了第 9、10 种」这一种病：

| 函数 | 症状 | 根因 |
|---|---|---|
| `vt-take-along-axis` | `不支持的输出元素类型 (COMPLEX DOUBLE-FLOAT)` | 逐元素内核按 8 种实数元素类型展开，`(t (error ...))` 兜底直接拒绝 |
| `vt-outer` | `invalid number of arguments: 3` | 复数分支写成 `(vt-reshape ... :dtype dt)` —— `vt-reshape` 只接 `(vt shape)` 两个参数。**复数外积从来没跑通过**，实数走原实现所以一直正常 |
| `vt-any` / `vt-all` | `unsupported input dtype (COMPLEX DOUBLE-FLOAT)` | 归约内核 8 路特化，复数落入 `(t (error ...))` |

修复策略体现了本轮的一条原则：**能用既有算子组合的，就不要新写内核**。

- `vt-take-along-axis`：只加两个 `(walk (complex double-float))` /
  `(walk (complex single-float))` 分支。注意内核里的 `SEL` 是**下标**（fixnum），
  元素值才是复数 —— 不受 `(type fixnum sel)` 声明影响。
- `vt-outer`：删掉多余的 `:dtype` 实参，改为先广播乘再按需 `vt-astype`，
  并补 `:out` 硬契约（`vt-check-out` + `vt-copy-into`）。
- `vt-any` / `vt-all`：**不重写归约内核** —— 先把复数转成 int8 的 0/1 真值掩码
  （`vt-nonzero-p`），再交给原实现。这样自动继承 `axis` / `keepdims` /
  `:out` / 空输入单位元（any→0、all→1）的全部既有契约。

复数真值语义（numpy 一致）：**任一分量非零即为真**，`bool(0+0j)=False`、
`bool(nan+0j)=True`。

#### 16.19.9 数值压测：为什么"差分对照"不够

差分对照只能覆盖**我事先想到的矩阵**。近奇异、病态、非方阵、秩亏、全零
这些情形一旦漏测，隐藏的收敛问题就会在用户那里爆。第四轮把压测固化成
测试的一部分（`stress-tests`，108 条断言）：

| 手法 | 内容 | 为什么有效 |
|---|---|---|
| **随机矩阵** | 确定性 LCG（`*lcg*` 固定种子 → 可复现）生成 1×1、3×3、5×5、2×5、5×2 | 覆盖"没想到的"数值组合 |
| **恒等式** | `Q·R=A`、`Q^H Q=I`、`U·S·V^H=A`、`U^H U=I`、`V^H V=I`、`A·V=V·Λ`、`A·A⁻¹=I`、`A·x=b`、Moore–Penrose 四条 | 分解不唯一（相位自由度），但恒等式**完全刻画**了分解 |
| **交叉验证** | `σ²` 必须等于 `eig(A^H A)` 的特征值 | SVD（单边 Jacobi）与 eig（厄米特 Jacobi）是**两条独立算法**，互证能抓出单边的系统性偏差 |
| **退化** | 全零矩阵（σ 全 0 / rank 0 / cond ∞ / inv 报错）、秩亏（rank 1、σ₂≈0）、1×1 | 除零、主元为零、空循环等边界 |
| **SPD 族** | `B^H B + nI`（1/2/3/6 阶）→ Cholesky + eig | 保证正定性，检验 `l_jj>0` 分支与特征值全正 |

容差按 `scale = max(1, ‖A‖∞)` 缩放（`1e-11` 相对量级），而不是写死绝对值 ——
否则病态矩阵会误报、良态矩阵又会漏报。

#### 16.19.10 测试

- `test/complex-linalg-test.lisp`：**216 条断言**
  = 46 项 numpy 2.4.6 差分用例（det/inv/solve/cholesky/SVD 奇异值/
  特征值/pinv/lstsq/rank/cond/matrix-power/einsum/范数）
  + 结构恒等式（`P·A=L·U`、`L·L^H=A`、`Q·R=A`、`Q^H Q=I`、`U·S·V^H=A`、
  `U^H U=I`、`V^H V=I`、`A·V=V·Λ`、Moore–Penrose 四条、`einsum ≡ 专用内核`）
  + 数值压测 108 条（§16.19.9：随机 1×1/3×3/5×5/2×5/5×2 + 退化 + SPD 1/2/3/6 + lstsq 超定欠定）
  + 契约断言（`:out` 形状/dtype、奇异矩阵 `solve/inv` 报错、非方阵 `det` 报错、
  非正定 `cholesky` 报错、各函数结果 dtype、complex64 升精度、实数回归网）
- 期望值由 `test/gen_complex_linalg_expected.py` 从本机 numpy 实测生成
  （`test/complex-linalg-expected.sexp`），比较用**相对容差** 1e-9
  （线代结果的最后几位随算法/求和顺序而异，逐位相等没有意义）。
- 已接入 `test/run-tests.sh`（`SUITES` + `DEFAULT_ORDER`）。

---

### 16.21 第五轮：复数 einsum 的 GEMM 快路径

#### 16.21.1 问题与实测

复数 einsum 从第三轮起就**正确**（结果 dtype 由输入统一推导，通用累加状态机
与元素类型无关），但一直**慢一个量级**：

| 表达式（256×256 complex128） | 改前 | 改后 | 倍数 |
|---|---|---|---|
| `vt-einsum "ij,jk->ik"` | **2526 ms** | **31 ms** | **81×** |
| `vt-matmul`（对照） | 33 ms | 33 ms | — |
| `vt-einsum "i,i->"` | 9 ms | **<1 ms** | ~30× |
| `vt-einsum "ij,ij->ij"`（1M 元素） | 138 ms | **8 ms** | **17×** |

根因不是「复数没优化」，而是**三条快路径的判据全是实数 dtype 谓词**
（`all-f64-p` / `all-f32-p` / `all-i64-p` / `all-i32-p`），复数直接落到
最底下的通用迭代状态机：每个输出元素都要走「补码递推 + 逐张量泛型 `aref`」，
而 `vt-matmul` 走的是元素类型特化的三重循环。

#### 16.21.2 三条快路径一并放开

| 通道 | 触发表达式 | 复数走什么 | 为什么安全 |
|---|---|---|---|
| P1-1 逐元素 | `ij,ij->ij` / `i,i->i` | `vt-*`（**不是** `vt-map`） | 复数走 `vt-*` 的元素类型特化内核；`vt-map` 对复数只能泛型 `funcall`，1M 元素实测 138ms vs 8ms（17×） |
| P1-2 全收缩 | `i,i->` | `%CX-EINSUM-DOT` | 单累加器，不取共轭，与 `numpy.einsum` 一致 |
| BMM | `ij,jk->ik` / `...ij,...jk->...ik` | `%CX-GEMM-STRIDED` | 语义 = `matmul`（不取共轭），strides 直通 |

#### 16.21.3 钩子而不是直接调用：编译顺序约束

`src/linalg.lisp` 在 `src/complex.lisp` **之前**编译（`.asd` 里复数层必须最后，
因为它要做前置分派）。所以 `einsum-execute` 不能直接写 `(%cx-gemm-strided ...)`
—— 编译期那是未定义函数。

沿用仓库既有的「SIMD 注册」模式（`*simd-matmul-2d-fn*`）：

```lisp
;; linalg.lisp（先编译）
(defvar *cx-einsum-gemm-fn* nil)   ; 复数 GEMM 快路径
(defvar *cx-einsum-dot-fn*  nil)   ; 复数全收缩内积

;; complex.lisp（后加载，顶层 setf）
(setf *cx-einsum-gemm-fn* #'%cx-gemm-strided
      *cx-einsum-dot-fn*  #'%cx-einsum-dot)
```

判据里**必须**带上钩子本身：

```lisp
(or all-f64-p all-f32-p all-i64-p all-i32-p
    (and *cx-einsum-gemm-fn* (or all-c128-p all-c64-p)))
```

否则钩子为 NIL 时仍然进入 BMM 分支，`funcall` NIL 崩掉、或 `cond` 落到
`(t nil)` 后**静默返回空结果**。这是「可选加速器」模式最容易踩的坑：
**判据要与使用点同源**。

#### 16.21.4 `%CX-GEMM-STRIDED` 的两套循环

```lisp
(defun %cx-gemm-strided (ad bd rd m k n
                         a-off b-off c-off
                         sa-i sa-j sb-j sb-k sc-i sc-k))
```

- **快分支**（`sa-j = sb-k = sc-k = 1`，即 A 行内连续 + C 连续 ——
  `ij,jk->ik` 的常规情形）：i-p-j 次序，内层 `(aref c (+ c-row jj))` 连续访存，
  与 `%CX-MATMUL-RUN` 同款。
- **通用分支**：按 strides 逐元素寻址，覆盖转置输入（`ij,kj->ik`）、
  非连续输入（列步长 >1）、非连续 `:out`（`ij,jk->ki`）。

两套循环都用 `%CX-DISPATCH` 做 **complex64 / complex128 元素类型特化** ——
`safety 1`，不是 `safety 0`（见 §16.14.1 的分级：通用 stride 路径不享受
「无条件信任」的待遇，`ld` 类错误必须让边界检查抓住）。

`%CX-EINSUM-DOT` 同理：单累加器，`%cx-dispatch` 特化，`offset` 透传。

#### 16.21.5 语义一致性验证

差分必须**同时**对齐 numpy 与 `vt-matmul`，因为 einsum 的语义是
「不取共轭」（`numpy.einsum` 一致），而 `vt-vdot` 才是共轭点积。
覆盖矩阵（complex64 / complex128 各 12 例，全部通过）：

`ij,jk->ik`、`ij,kj->ik`(NT)、strided A、`ij,jk->ki`（转置输出）、
`ij,ij->ij`、`i,i->`、`...ij,...jk->...ik`、batch 广播 `(1,4,5)@(3,5,6)`、
`:out` 提供、`k=1`、`i=1`、三张量链 `ij,jk,kl->il`。

---

### 16.22 顺带修掉的两个「静默错答案」（批量复数 matmul）

做 §16.21 的差分时，`vt-einsum "...ij,...jk->...ik"` 与 `vt-matmul` 差了 **3.3**
（量级不对，说明不是精度问题）。三方对照（einsum / matmul / 手写三重循环）
定位到 `%CX-BATCHED-MATMUL` 的两个独立缺陷：

| 缺陷 | 现象 | 根因 |
|---|---|---|
| **offset 丢失** | 每个 batch 都读到基数组开头 → 结果全错 | `(vt-slice (vt-ravel a) ...)` 得到的 1D 片段**已连续**，`vt-contiguous` 零拷贝返回它，`offset` 保留非零；而 `%CX-MATMUL-RUN` 按扁平下标从 0 起算 |
| **不支持 batch 广播** | `(1,4,5)@(3,5,6)` 直接 `重塑失败: 元素总数不一致 (旧 0, 新 20)` | 原实现用 `reshape` 切 batch，隐含「两侧 batch 形状相同」 |

这与 `manip.lisp` 里 `%flat-offset0` 一节记录的是**同一类根因**
（「扁平下标 + 裸 AREF」遇到带 offset 的视图），只是这次发生在复数线代内核里。

修复（两处都改在源头，不做调用侧补救）：

1. `%CX-MATMUL-RUN` 签名加 `a-off / b-off / c-off`，`%CX-MATMUL-2D` 透传
   `(vt-offset a)` / `(vt-offset b)` / `(vt-offset res)`；
2. `%CX-BATCHED-MATMUL` 改为**按 strides 递推 batch 偏移**（不再 reshape 切片）：
   batch 维右对齐、缺的左侧补 1、维长 1 → stride 0（广播），
   与实数 SIMD 批量路径的广播语义一致。

`%CX-DOT-1D` 也顺手补了 offset（同样的假设，只是此前没被触发）。

---

### 16.23 `vt-unique :axis`（通用特性，非复数专属）

numpy 2.x 的 `unique(axis=...)` 此前 clvt **实数路径也没有**，属于通用缺口。
实现严格照 numpy 的算法（`lib/_arraysetops_impl.py`）：

1. `moveaxis(axis, 0)` + `reshape(n, m)` —— 沿 AXIS 的每个切片是一行，
   行内元素按「其余轴的 C 序」展开；
2. 用**稳定**排序按**逐元素字典序**排行（numpy 在有 `return_index` 时用 mergesort，
   稳定排序保证「组首 = 首次出现」）；
3. 相邻行做全元素 `!=` 比较，全等才算重复。

实现取巧点：先把 AXIS 搬到首轴并落地为 C 连续副本，这样「行」在内存中连续
（行 i = `[i*m, (i+1)*m)`），比较器不需要偏移表；输出再用**逆置换**转回原轴序
（`AXIS = 0` 时省掉这次 transpose）。

**返回值的布局契约**：`vt-unique :axis` 的结果恒为 **C 连续新数组**
（`vt-contiguous-p` = T），与 numpy `unique(axis=)` 以及本库的**展平路径**
一致。`transpose` 换回轴序后必须再 `vt-contiguous` 落地 —— 否则用户会拿到
「形状对、布局是视图」的结果，与展平路径不一致（代价最多一次「唯一行」
大小的拷贝；唯一行数 ≤ 输入行数）。

#### 16.23.1 与展平路径的语义差异（同一个 numpy 里两条分支）

| 场景 | 展平（`axis=NIL`） | 按轴（`axis=k`） |
|---|---|---|
| NaN × 多个 | **折叠为一个** | **每个含 NaN 的行都保留**（`NaN != NaN`） |
| `-0.0` vs `0.0` | 相等（合并） | 相等（合并） |
| 排序 | 升序 + NaN 末尾 | 行字典序 + NaN 末尾 |
| `return-inverse` 形状 | 与输入同形（numpy ≥ 2.0） | **一维，长度 = 该轴长度** |

「NaN 行不折叠」不是 bug 而是 numpy 的实测行为：

```python
np.unique(np.array([[nan,1],[nan,1],[2,0]]), axis=0)
# [[2,0], [nan,1], [nan,1]]   ← 两行 NaN 都保留，counts 全为 1
np.unique(np.array([nan, nan, 1.]))     # [1., nan]  ← 展平则折叠
```

#### 16.23.2 退化形状（numpy 实测，逐一对齐）

| 输入 | `axis` | 结果形状 | 说明 |
|---|---|---|---|
| `(0, 3)` | 0 | `(0, 3)` | 0 个切片 |
| `(3, 0)` | 0 | `(1, 0)` | 3 个**空行**全部相等 → 合并成 1 行 |
| `(4,)` | 0 | `(1,)` | 秩 1 时 AXIS 被忽略（走展平分支）|

#### 16.23.3 复数

比较原语按 `complexp` 分派：字典序 = （实部，虚部），NaN 视为最大。
因此 `vt-unique` 的复数包装在 `axis` 给定时**直接复用实数层的 `%UNIQUE-AXIS`**，
不再写第二份实现 —— 这也验证了「按轴去重的比较是元素类型无关的」这一点。

---

### 16.24 `vt-trace :offset`（+ `:axis1` / `:axis2`）

`vt-diagonal` 早就支持 `:offset`，`vt-trace` 却没有，且硬编码 `(assert (= 2 (vt-order ...)))`。
本轮补齐完整签名 `vt-trace (matrix &key offset axis1 axis2 dtype out)`。

三个要点：

1. **收缩轴是 `-1` 而不是「全部」**。`vt-diagonal` 把对角线轴放在**末尾**，
   所以 `(vt-sum d :axis -1)` 才是「沿对角方向收缩、保留其余轴」。
   直接 `(vt-sum d)` 会把 batch 轴一起吃掉 —— 3D 输入 `axis1=1 axis2=2`
   本该得到 `(2,)`，却得到标量（差分测试第一条就抓到了）。
2. **越界 offset 返回 0**（`diag-len = 0` → 空求和），与 numpy 一致，不报错。
3. **硬契约用 `error` 而非 `assert`**：`assert` 可能被 `(safety 0)` 编译剔除
   （§16.18 已记录同类问题），秩 < 2 现在抛明确错误。

与 numpy 的**已知差异**：numpy 的 `trace` 缺省是 `axis1=0, axis2=1`，
clvt 与自家 `vt-diagonal` 保持一致，缺省取**最后两轴**。2 维输入两者等价；
batch 输入要复现 numpy 缺省需显式 `:axis1 0 :axis2 1`（已写入 docstring）。

---

### 16.25 第五轮测试

- `test/setops-axis-trace-test.lisp`：**56 条断言**
  = 37 项 numpy 2.4.6 差分（unique :axis 的 7 种轴/19 例含 index/inverse/counts、
  NaN 行、非连续视图、退化形状、复数 2 例 + trace 的 14 例含 offset 正负/越界/
  复数/3D axis1+axis2）
  + 契约断言（dtype/shape 保持、NaN 折叠差异、inverse 重建、
  axis 越界与秩不足报错、`:out` 非连续 0 维视图）
  + 实数回归网（默认路径逐位不变）
- 期望值由 `test/gen_setops_axis_expected.py` 从本机 numpy 实测生成
  （`test/setops-axis-expected.sexp`），禁手写（§10.4）。
- 已接入 `test/run-tests.sh`（`SUITES` + `DEFAULT_ORDER`），当前共 **44 个套件**。

---

### 16.26 einsum 的实数 GEMM 快通道接入 SIMD（第六轮）

**症状**：同一台机器上，`(vt-einsum "ij,jk->ik" a b)` 比 `(vt-matmul a b)`
慢 **5.7~6.4×**（float64，1024²，单线程）；多线程下内存带宽掩盖掉一部分，
仍有 1.78×。复数路径上一轮刚接通，实数这条反而漏了。

**根因（两条路径互不相通）**

| | 走的通道 | 内核 |
|---|---|---|
| `vt-matmul` 2D@2D | `*SIMD-MATMUL-2D-FN*` | `simd-matmul.lisp`：AVX 4 宽 + FMA + 64×256×256 分块 |
| `vt-einsum` BMM | `%MATMUL-DF-FAST-KERNEL` | `linalg.lisp`：4 路手写展开，**无 SIMD、无分块** |

关掉 `*SIMD-MATMUL-ENABLED*` 后两者持平（0.93~0.98×），可直接证明差异
**全部**来自这两个内核，与 einsum 的通道判定、类型提升无关。

**修法：一个裸数据钩子 + 布局归一化**

1. `linalg.lisp` 新增 `*SIMD-GEMM-RAW-FN*`（定义在前、`simd-matmul.lisp`
   注册，同 `*SIMD-MATMUL-2D-FN*` 模式 —— linalg 先编译，不能反向引用）。
   签名 `(a-data b-data c-data m k n a-off b-off c-off elem-kind)`，
   语义与 `%MATMUL-*-FAST-KERNEL` **完全一致**（`C += A·B`，行主序，
   偏移是**元素**下标）；返回 T/NIL 表示是否接管。
2. `define-einsum-gemm` 为 4 种元素类型各生成一个入口
   `%EINSUM-GEMM-{F64,F32,I64,I32}`，把任意步长归一化成
   「连续 C += 连续 A·连续 B」后进内核：

| 布局 | 处理 |
|---|---|
| ① 全连续（NN） | 直接进分块内核（SIMD 优先，NIL 落标量） |
| ② B 按 (n,k) 存（NT，如 `"ij,kj->ik"`） | 64×64 **分块转置**成 (k,n) 缓冲，走 ① |
| ③ C 非行主序（如 `"ij,jk->ki"`） | 算到连续临时 C，再 64×64 **分块 scatter** 累加回 |
| ④ 其余（A 非行连续） | 原三重循环兜底，逐位不变 |

**为什么 ②③ 必须分块**：朴素写法里内层循环的步长是「一整行」
（NT 时 `sb-k = k`，C 转置时 `so-k = m`），512² 上每步都换一条 cache line。
512² 单线程实测：③ 不分块 609 ms，分块后 36 ms —— **17×**，
比 GEMM 本体（31 ms）还贵；② 同理（381 ms → 30 ms）。

#### 16.26.1 踩坑一：不要在 macrolet 里做布局归一化

第一版把上面四条判断直接展开了 `einsum-execute` 的 `gen-bmm-logic` 里。
结果**一行没改的兜底路径**从 711 ms 退化到 947 ms（+33%）——函数体膨胀后
SBCL 对该分支的代码生成明显变差（分支更多、寄存器压力更大）。

对策：`einsum-execute` 里每个类型只留**一行函数调用**，全部逻辑进独立
`defun`。改完后兜底路径回到基线，且编译期类型检查也回来了。

> 教训：宏展开层的「多写几行」不是零成本，它会重排整个函数的寄存器分配。
> 判断标准应该是 **生成的机器码/展开体积**，不是源码行数。

#### 16.26.2 踩坑二：大对象的 `:initial-element` 是 calloc，不是 memset

临时缓冲最初写 `(make-array n :element-type 'double-float :initial-element 0.0d0)`。
512² 只有 2 MB，但「分配 + GEMM」要 **111 ms**（对照：复用缓冲 46 ms）。

原因：SBCL 对 large object 的 `:INITIAL-ELEMENT` 走 calloc/零页，**并不真正
写内存**；真正的写入由 31 个工作线程各自首次触达时触发 page fault，全部
并发抢 `mmap_lock` → 串行化。

对策：`(make-array n :element-type 'double-float)` + `(fill buf 0.0d0)`。
`FILL` 走 memset 在主线程一次 touch 完，代价 0.5 ms，总时间回到 47 ms。

（顺带：`:initial-element 0`（整数 0）还会因为与 `double-float` 元素类型
冲突而报 `Constant 0 conflicts with its asserted type`，让 COMPILE-FILE
返回 failure-p=T。所以临时缓冲的零值也按类型分别写 `0.0d0` / `0.0f0` / `0`。）

#### 16.26.3 数值一致性

四条路最终都调用同一个 `%MATMUL-*-FAST-KERNEL` 家族（或同一 SIMD 内核）、
同一分块顺序，因此 **`vt-einsum` 与 `vt-matmul` 逐位相同**（20 项差分脚本，
maxabs = 0）。与 numpy `@` 的相对差 ≤ 7.7e-15。

相对上一版（标量三重循环）的差异只有 FMA 少一次舍入：相对差
~1e-12（512²，比旧值更接近精确解），属于**改善**而非退化。

唯一非逐位相同的是批量广播 `"bij,jk->bik"`：`vt-matmul` 走
`*SIMD-BATCHED-MATMUL-FN*`（批间并行），einsum 走逐 batch GEMM，
累加顺序不同 → 相对差 2.5e-12，属正常浮点差异。

#### 16.26.4 实测（512² float64，µs，3 次最好值）

512² float64 **单线程**（µs，3 次最好值；单线程最接近核数少的机器）：

| 表达式 | 改前 | 改后 | 改前/`vt-matmul` | 改后/`vt-matmul` |
|---|---:|---:|---:|---:|
| `vt-matmul`（基准） | 31,141 | 35,596 | 1.00× | 1.00× |
| `"ij,jk->ik"`（NN） | 196,033 | **32,485** | **6.30×** | **0.91×** |
| `"ij,kj->ik"`（NT） | 381,070 | **29,818** | **12.24×** | **0.84×** |
| `"ij,jk->ki"`（转置输出） | 609,077 | **36,112** | **19.56×** | **1.01×** |

2048² **多线程**（32 核，µs）—— 规模放大后差距同样消失：

| 表达式 | 改前/`vt-matmul` | 改后/`vt-matmul` |
|---|---:|---:|
| `"ij,jk->ik"` | 1.78×（6,210,809） | **0.99×**（3,439,937） |
| `"ij,kj->ik"` | 1.71×（5,985,129） | **0.99×**（3,447,156） |
| `"ij,jk->ki"` | 13.82×（48,326,644） | **0.99×**（3,459,113） |

1024² 多线程 0.93~1.02×；256²（1.67e7，刚过 SIMD 阈值）0.95~1.02×。
小矩阵（8/16/32²，低于 `*SIMD-MATMUL-THRESHOLD*` = 50000）两者都走标量，无退化。

> 注：`"ij,jk->ki"` 的「改前」是**改动前就已经存在**的老问题（通用三重循环
> 内层跨行），不是上一轮引入的；本轮顺手按 ③ 修掉。

#### 16.26.5 回归网

`test/einsum-gemm-layout-test.lisp`（36 条断言，接入 `test/run-tests.sh`）。
**主断言写成「与 `vt-matmul` 逐位相等」**而不是硬编码期望值 —— 后者要跟着
机器/编译器变，前者同时锁住三件事：语义没变、四条布局路径一致、两条入口
最终落到同一内核。只有两处用容差（1e-10）并注明原因：

- 批量 `"bij,bjk->bik"`（每 batch 的 `m·k·n` 低于 SIMD 阈值）：einsum 逐
  batch 走标量、`vt-matmul` 的批量版按总工作量走 SIMD，累加顺序不同；
- SIMD 开/关的 A/B：FMA 与 mul+add 的舍入不同（这条对 `vt-matmul` 本来
  就成立，不是本轮引入）。

---

### 16.27 第七轮：复数 matmul 内核对齐实数线（标量形态）

#### 16.27.1 症状

`vt-matmul` 的复数 2D 路径比实数慢得离谱 —— §16.14.3 记的是 256² 下
39×（那时实数已走 SIMD，复数还是标量三重循环）。把 SIMD 关掉、只比
**标量对标准量**，复数仍慢 **2.5–3×**（n=256：复数内核 24.2ms，
实数标量内核 6.2ms），按「每元素浮点工作量」换算，复数应当只慢 4×
—— 反过来说，复数内核**每 flop 的开销是实数标量的 6 倍以上**，这已
经不是「复数算术天生贵」能解释的量级。

#### 16.27.2 消融：三项差距全在「代码形态」，与复数无关

n=256、单线程、预分配缓冲区、只计内核本身（不含 `make-vt`/`contiguous`）：

| 变体 | 耗时 | GFLOPS | 加速 |
|---|---:|---:|---:|
| 旧内核：`safety 1` + 每元素重算下标 + 无展开 | **24.2 ms** | 5.0 | 1.00× |
| ① 只把循环降到 `safety 0`（索引可证不越界） | 22.0 ms | 5.5 | 1.10× |
| ①+② 行起点只算一次，内层用 `c-ptr`/`b-ptr` 递推 | 14.1 ms | 9.5 | 1.71× |
| ①+②+③ j 维 4 路展开（对齐实数内核的展开宽度） | **13.5 ms** | 9.9 | **1.79×** |

三项收益互不重叠。**实数线 `%MATMUL-DF-ROWS` 一开始就是这三件事**
（`(safety 0)`、指针递推、4 路展开，见 `linalg.lisp`），所以差距的根因
是「复数内核是按最朴素的教科书三重循环写的」，而不是复数表示本身：

- `safety 1` 下的 `aref` 每次都带边界检查，且阻止了向量化；
- `(+ c-row j)` / `(+ b-row j)` 每元素两次加法 + 一次乘法（`p*n`），
  在内层里占掉可观的指令预算；
- 无展开 → 循环控制指令占比高，也挡住乱序执行的重叠。

#### 16.27.3 被实测否掉的两个「看上去更聪明」的方案

| 方案 | n=256 | 结论 |
|---|---:|---|
| SoA（拆 re/im 两个 `double-float` 数组）+ B 打包成 j-major + j4 | 22.9 ms | **更慢**。转换 + 打包是 O(n²) 但内层并不更快：SBCL 的复数 `*`/`+` 是**开码**的，specialized 数组里的 `(complex double-float)` 按 16 字节就地读写，不装箱、不分配 |
| SoA + i-j-p 单累加器（C 常驻寄存器） | 75.2 ms | **慢 5×**。`B[p, j]` 的 p 维步长是 n → 每取一个元素跨一行，缓存全废 |
| 把 `p=0` 折进内核（省掉清零那一趟） | 14.6 ms | **更慢**。j 展开的初始化代码要写 4 份，挤掉内层的寄存器分配；清零本身只有 O(mn) |

结论：**保持 boxed complex 缓冲区不变**，只把循环形态抄实数线。

#### 16.27.4 实现

```lisp
;; complex.lisp §11.5.1
(defmacro %def-cx-matmul-rows (name ctype) ...)   ; 生成 cdd / css 两个行内核
(%def-cx-matmul-rows %cx-matmul-cdd-rows (complex double-float))
(%def-cx-matmul-rows %cx-matmul-css-rows (complex single-float))
```

1. **行内核签名与 `%MATMUL-DF-ROWS` 完全一致**：
   `(a b c k n a-off b-off c-off i-start i-end)`，语义 `C += A@B`
   （**不清零**，由调用方负责）。这样 `vt-matmul`（先清零）与 einsum 的
   增量累加（§16.21）能共用一份内核。
2. `%cx-matmul-run` = 清零 + 按 `array-element-type` 选内核 +
   **`CALL-PARALLEL-GEMM`（直接复用 `linalg.lisp` 的调度器）**。
   于是复数与实数的并行语义（`*enable-parallel-matmul*` /
   `*matmul-parallel-threshold*` / `*matmul-thread-count*` / 行切法）
   逐项一致，不引入第二套线程策略。
3. `%cx-gemm-strided`（einsum 的 BMM 通道）新增 (a) 分支：当 strides
   与「连续 NN」逐项相同（`sa-j=1, sb-k=1, sc-k=1, sa-i=k, sb-j=n, sc-i=n`）
   时**委托同一个行内核**；否则保留原来的 (b) 行内连续/(c) 通用 strided
   两条循环。累加次序（i→p→j 递增）不变 → einsum 与 `vt-matmul`
   仍然逐位相等（§16.21.5 的验收项）。
4. `(safety 0)` 的适用性同 `%MATMUL-DF-ROWS`：索引由 i/p 维推出、
   `c-ptr`/`b-ptr` 与 j 同步 +1，上界恰为 `a-off+m·k` / `c-off+m·n` /
   `b-off+k·n`，可证不越界。**通用 strided 分支仍是 `safety 1`**
   （§16.14.1 的分级没动）。
   这与 §16.14.1 表里「普通内核 `safety 1`」并不冲突：那一行针对的是
   **接受用户 stride/shape 的通用路径**；行内核只接受「由 m/k/n 推出的
   连续布局」，信任级别与实数 `%MATMUL-DF-ROWS` 完全相同 —— 与实数线
   对齐本来就是本轮的目标，两边必须同级，否则复数永远追不上。

#### 16.27.5 实测（n=192，同进程比值，`test/complex-matmul-perf-test.lisp` 内）

| 项 | 耗时 | 说明 |
|---|---:|---|
| 新行内核（c128，单线程） | 5.32 ms | 9.6 GFLOPS |
| 旧内核形状（同上参照） | 10.18 ms | 5.0 GFLOPS → **1.92×** |
| 实数标量行内核（`%MATMUL-DF-ROWS`） | 2.46 ms | 5.8 GFLOPS → 新/实 = **2.16×** |

复数每元素 8 flop、实数 2 flop，**flop 比就是 4×**；实测 2.16× 说明
复数的**单位 flop 效率反而比实数标量高**（9.6 vs 5.8 GFLOPS），
墙钟多出来的 2.16× 小于 4× 的那部分由「复数一次 `aref` 搬 16 字节、
内层访存更省指令」抵掉了。

端到端 `vt-matmul`（含 `make-vt` + 清零，SBCL 2.6.8，32 逻辑核 / 4 核配额
的容器；绝对毫秒随机器负载波动较大，**看比值**）：

| n | 复数串行 | 复数默认（并行） | 实数 SIMD | 串行/实数 | 并行/实数 |
|---|---:|---:|---:|---:|---:|
| 256 | 14.8 ms | 2.6 ms | 1.3 ms | 11× | **2.0×** |
| 512 | 143 ms | 11.9 ms | 4.6 ms | 31× | **2.6×** |
| 1024 | 1241 ms | 342 ms | 81 ms | 15× | **4.2×** |

读法：

- **并行列**是「用户视角」的默认行为（`*enable-parallel-matmul*` 默认 T，
  与实数线同一套阈值），相对实数 SIMD 只剩 **2.0–4.2×**，1024 已经进入
  内存带宽瓶颈（复数每次乘加搬 16 字节，是实数的 8 倍）；
- **串行列 vs 实数 SIMD** 的 11–31× 是「标量 vs SIMD+分块」的差距，
  不是本轮的目标（§16.27.6）；
- 串行/并行的加速比在本沙箱里测出来是 3.6–12×，明显超过 cgroup 的
  4 核配额（`cpu.max=400000/100000`，但 `cpus_allowed=32`，实际可用核数
  随宿主负载浮动），因此**只作量级参考**。并行正确性有独立断言：
  256² 下并行与串行结果**逐位相同**（`test/complex-matmul-perf-test.lisp`）。

#### 16.27.6 还剩什么没做（按要求本轮不碰 SIMD）

剩下的差距**全部**来自「实数有 SIMD、复数没有」，分两块：

1. **宽度**：`simd-matmul.lisp` 一次处理 4 个 `double-float`（AVX）且用
   FMA，单位时间吞吐是标量的 8–10 倍；
2. **指令数**：复数每元素 4 mul + 4 add（`(ar·br − ai·bi, ar·bi + ai·br)`），
   实数 1 mul + 1 add。即使上了 SIMD，复数也要拆成两组实部/虚部向量
   做 4 次 FMA 才等价于一次复数乘。

这两块都属于 `simd-matmul.lisp` 的领域（复用它的分块与线程框架、
在 SoA 视图上做 4 条实 GEMM），是**下一轮**的事。

#### 16.27.7 测试

`test/complex-matmul-perf-test.lisp`（53 断言，已接入 `run-tests.sh`）：

- 正确性：手写三重循环参照（完全独立实现）逐位对照；非方阵 / 尾路径
  （n 取 1,2,3,5,6,7,9 专压 4 路展开的剩余列）/ 退化（k=1、m=1、n=1、
  零尺寸）/ 切片视图（offset≠0、转置视图、列切片）/ 批量与 batch 广播
  （分片对 2D `vt-matmul` 逐位）/ einsum 的 NN·NT·转置输出三条通道
  （对 `vt-matmul` 逐位）/ c64·c128 / `:out` / `:dtype`；
- 性能护栏（同进程比值，不写死毫秒）：新内核 ≥ 1.3× 旧内核形状
  （实测 1.9×）、≤ 3.5× 实数标量行内核（实测 2.2×，理论下限 4×）、
  并行开关不改结果且不慢于串行 1.2×。

---

### 16.28 第八轮：复数缺口扫描与补齐（入口 / 幂 / 归一化）

#### 16.28.1 怎么扫（这套方法请沿用）

「哪些函数没支持复数」不能靠读代码猜 —— 复数层是**前置分派**（§16.12）：
没被包过的函数会静默落到实数实现，可能报错，也可能**悄悄取实部**。
所以按三层扫：

| 层 | 做法 | 能发现什么 |
|---|---|---|
| ① 静态差集 | 从 `package.lisp` 抽 `#:vt-*` 导出清单，减去 `complex*.lisp` 里出现过的符号 | 「完全没被复数层碰过」的函数（367 个导出里 202 个被碰过、165 个没有） |
| ② 动态探针 | 对每个候选按真实签名喂复数入参，分 `:ok / :reject / :type / :other` | ② 剩下的差异：`%cx-def-unsupported` 的显式拒绝 vs 实数实现的类型断言崩 |
| ③ dtype 保真扫描 | 对**所有**可调用导出符号统一喂 `complex128`，看结果 dtype 是否变成实数 | **静默丢虚部**（最危险的一类），三层里只有这一层能抓 |

③ 的实现要点：把「按定义就该返回实数」的函数（`abs/angle/isnan/argmax/norm/
var/std/det/matrix-rank/…`）列成白名单，其余出现「复数入 → 实数出」一律当嫌疑。
本轮 ③ 报了 24 个，其中 19 个是白名单漏项（比较运算、`argsort`、`nanvar` 等），
真问题 5 个。

#### 16.28.2 本轮修掉的 6 个缺口

| # | 函数 | 症状 | 根因 |
|---|---|---|---|
| 1 | `vt-from-array` | `(complex double-float)` 数组 → **float64**（虚部丢） | `infer` 的 `cond` 里没有 complex 分支，落到末尾 `(t :float64)` |
| 2 | `vt-from-sequence` | 含复数的 list → **float64** | `:dtype` 缺省值写死 `:float64`（填充路径已支持复数，是 dtype 拖累的） |
| 3 | `vt-full` | `(vt-full s #C(3 1))` → **float64** | 同上，`:dtype` 缺省写死 `:float64` |
| 4 | `vt-float-power` | 复数 → **float64**（只算实部） | 实现里 `(coerce a 'double-float)` |
| 5 | `vt-cbrt` | 复数 → **float64** | `%float-map-pair` 强制实数内核 |
| 6 | `vt-layer-norm` | 复数 → `#C(0 0) is not of type REAL` | `final-dtype` 写死「float32 或 float64」，`(vt-+ var eps :dtype :float64)` 撞断言 |

1–3 是**数据入口**上的静默错值，4–5 是**数值算子**上的静默错值，6 是硬报错。

#### 16.28.3 三个语义决策（不是「能跑就行」）

1. **`vt-cbrt` 不改复数实现，改成拒绝。** numpy 2.3.5 实测
   `np.cbrt(np.array([8+0j]))` → `TypeError: ufunc 'cbrt' not supported for the
   input types`。所以正确行为是**确定性报错**（并入 `%cx-def-unsupported`），
   而不是补一个「复数立方根」——那会造出一个 numpy 没有的语义。
   需要复数立方根请写 `(vt-pow z 1/3)`（主值分支，与 `np.power` 一致）。
2. **`vt-float-power` 复用 `vt-pow` 的内核。** 两者在复数域是同一个 pow
   主值分支（实测 `np.float_power(2+1j, 0.5+0j)` 与 `np.power` 逐位相同），
   差别只在实数侧：`pow` 对整数保留精确整数幂、`float_power` 一律浮点。
   于是把原来内联在 `vt-pow` 里的 lambda 提成 `%CX-POW-2`，两处共用 ——
   边界（`x^0=1`、`0^负数=NaN`、`0^0=1`、NaN 传播）只有一份实现。
3. **`vt-layer-norm` 的复数定义**：沿归一化轴取**复数均值**、
   **实数方差**（`mean(|x-mean|²)`，与 numpy 的 `var(complex)` 一致）、
   实数 `std = sqrt(var+eps)`，再 `(x-mean)/std` → 复数。
   实现上只要不把中间量 `:dtype` 强制成实数即可（`final-dtype` 跟随输入
   的复数 dtype，`var/std/eps` 走对应的实 dtype）。
   torch 的 `layer_norm` 只定义在实数域，所以这一条是**本库的扩展语义**，
   用「虚部为 0 时退化为实数路径」+「归一化后 fiber 均值为 0、var≈1」
   两条自洽性断言来锁（§16.28.5）。

#### 16.28.4 顺带修掉的一个不一致

`vt-pow` 的复数路径里 NaN 判定排在「指数为 0」之前，于是
`(vt-pow #C(nan 0) 0)` → `NaN+NaNj`，而**实数路径**给 `1.0`、numpy 也给 1
（C99 的 `pow` 约定）。`%CX-POW-2` 里把零指数判定提到最前，两条路径
与 numpy 三者一致。

#### 16.28.5 测试

新增 `test/complex-gap-test.lisp`（50 断言，已接入 `run-tests.sh`，现共 47 套件）：

- **dtype 契约**：`from-array` 的 `(complex double-float)`/`(complex
  single-float)`/2D/`:fast t`；`from-sequence` 的 1D/嵌套/空序列/显式
  `:dtype`；`vt-full` 的复数/实数/整数填充；显式 `:dtype :float64` 仍按
  §16.5 取实部（不回归）；
- **数值契约**：`float_power` 用 numpy 实测值对照
  （`(2+1j)^(2+0j)=3+4j`、`(-1+0j)^0.5=6.12e-17+1j`）+ 四个边界
  （`nan^0=1`、`0^0=1`、`0^-1=NaN`）+ 实数路径回归；
- **报错契约**：`cbrt` 复数 → 报错且消息含「不支持复数」；`float_power`
  显式非复数 `:dtype` → safe-casting 拒绝；
- **layer-norm**：dtype/形状、归一化后 fiber 均值 0 与 var≈1
  （容差含 `eps`：`var/(var+eps) ≈ 1-1e-5`）、虚部为 0 时与实数路径
  数值一致、复数 `gamma`/`beta`、`:out` 契约；
- **回归网**：`conj/abs/pow/sqrt/sum/norm/matmul/take-along-axis/
  compress/kron` 的复数 dtype 抽查。

> 测试里的一个坑：**别用 `(= x x)` 判 NaN**。测试进程没有屏蔽浮点陷阱
> （库内部才有 `with-float-safe`），NaN 比较会直接抛
> `FLOATING-POINT-INVALID-OPERATION`。改用 `sb-ext:float-nan-p`（位级判定）。

#### 16.28.6 扫描结论：仍然「不支持复数」的清单（都是**有意**的）

`%cx-def-unsupported` 现在包含：`floor/ceil/trunc/rint`、`div/mod/rem`、
位运算族、`atan2`、`even-p/odd-p`、`percentile/quantile`、`hypot`、
**`cbrt`（本轮新增）**、激活函数族（`sigmoid/relu/leaky-relu/swish/
softplus/gelu/mish/hard-tanh/hard-sigmoid`）。依据都是「numpy 同样拒绝」
或「数学上无定义（复数无全序）」，不是实现偷懒。

另外 `vt-eig`/`vt-eigvals` 返回**实数**特征值**不是 bug**：本库的 `vt-eig`
是 `eigh` 语义（只吃 Hermitian/对称，用 UPLO 指定三角 + 镜像，见其 docstring），
厄米特矩阵的特征值本来就是实数。要一般复矩阵的复特征值，那需要另一套
（Hessenberg + QR 迭代）实现，属未来工作。

### 16.29 第九轮：沿轴扫描的静默错值（cumsum/cumprod 只算第一条 fiber）

#### 16.29.1 症状

```lisp
(vt-cumsum a :axis 0)   ; a 是 (3 4) complex128
;; => [[-1.5-0.5j, 0+0j, 0+0j, 0+0j],
;;     [-1.0+0j,   0+0j, 0+0j, 0+0j],
;;     [-2.0+0.25j,0+0j, 0+0j, 0+0j]]     ; ← 只有第 0 列对，其余全 0
```

`axis` 取**最后一维**时正确，取任何更靠前的维度时**只算出第一条 fiber**，
其余位置保持 `make-vt` 的 0。受影响的函数：
`vt-cumsum` / `vt-cumprod` / `vt-nancumsum` / `vt-nancumprod`（后两个复用
同一个扫描内核）。

#### 16.29.2 根因：少了一层 `AX-STRIDE` 遍历

`%CX-SCAN-RUN` 的 axis 分支用 `%CX-FIBER-GEOM` 拿到 `(head dim ax-stride)`，
然后只做了：

```
for h in 0..head:                        ; 前导块
  for j in 0..dim:                       ; 沿轴前进
    p = h*dim*ax-stride + j*ax-stride
```

漏掉了 **`ax-stride` 那一层** —— 轴之后的每个位置（如 `axis=0` 时的其余各列）
都是一条**独立 fiber**。正确形式是三层：

```
for h in 0..head:
  for ti in 0..ax-stride:                ; ← 漏的就是这层
    acc = 单位元;  p = h*dim*ax-stride + ti
    for j in 0..dim:
      acc = op(acc, s[p]);  d[p] = acc;  p += ax-stride
```

对照证据：同一份 `%CX-FIBER-GEOM` 的其它六个调用点
（`%DEF-CX-REDUCER` 家族、`%CX-ARG-REDUCE-RUN`、`nanarg`、`sort-axis`、
`nanmedian`）**都有** `ti` 循环，所以 `sum/prod/amax/median/sort/...` 沿轴
一直是对的 —— 只有扫描内核漏了。这也解释了为什么 bug 能活这么久：

- 既有测试的 `cumsum` 用例全是 1-D（`axis` 缺省或唯一位），而 1-D 时
  `ax-stride = 1`，`ti` 层退化成一次迭代，**看不出差别**；
- 复数层此前没有「沿轴扫描」的差分测试。

#### 16.29.3 修法

在 `%CX-SCAN-RUN` 的 axis 分支补上 `ti` 层（见上），并把注释写成
「为什么必须有这一层」，避免下次重构再删掉。同时保持
`(safety 1)`（§16.14.1 的分级：接受任意 axis 的通用内核不享受 safety 0）。

#### 16.29.4 本轮怎么找到它的（两层扫描，可复用）

第八轮的扫描是「dtype 保真」（复数入参 → 实数出），查不出「dtype 对、数值错」。
本轮补两层：

| 层 | 做法 | 结果 |
|---|---|---|
| A. 同形调用对照 | 对**每个**导出 `vt-*`，用同一组 20 种调用形态分别喂**实数**与**复数**镜像（形状/dtype 宽度一致），比较「实数能调通而复数报错/降级」 | 0 处（第八轮已把这类清完；`(complex in → real out)` 的 96 个原始命中全部来自一个不含复数入参的形态，属扫描自身的假阳性，已剔除） |
| B. 数值对拍 | 190 条复数表达式在 clvt 侧求值、numpy 侧复算，逐元素比对（rtol 1e-9） | 188 项一致；1 项不一致 = **本节的 cumsum/cumprod**；2 项是 QR/LU 的**约定差异**（见下） |

B 的覆盖面：逐元素/超越、归约（含 axis 元组与 keepdims）、沿轴扫描、
排序/argsort/unique（含 counts/inverse）、searchsorted、diff/gradient/trapz、
拼接/堆叠/切分/insert/delete/repeat/tile/pad/rot90/flip/roll、take/compress/
extract/take-along-axis/put-along-axis、diag/tril/triu/trace/diagonal、
matmul/dot/outer/inner/vdot/kron/cross/tensordot/multi-dot、einsum 五种模式、
det/inv/solve/pinv/cholesky/matrix_power/cond/svd-qr 的数值、复数比较与 where，
以及 complex64 一整组镜像。

> QR/LU 的两处「不一致」不是 bug：QR 分解在列上可乘任意单位模相位，
> numpy 与 clvt 的相位约定不同；用重构恒等式判：`‖QᴴQ − I‖ = 1.1e-16`、
> `‖QR − A‖ = 6.7e-16`。LU 的置换约定也不同，重构由
> `complex-linalg-test` 的恒等式覆盖。

#### 16.29.5 测试

`test/complex-scan-axis-test.lisp`（26 断言，已接入 `run-tests.sh`，共 48 套件）。
期望值由 numpy 2.3.5 实测**硬编码**（数据是二进制可精确表示的小数，
累加也精确，故要求**逐位相等**）：

- `cumsum`/`cumprod` 的 axis=0/1（3×4）+ 负轴 `-1`/`-2`；
- 3-D（2,3,4）三个轴各一组 —— 这是最能暴露「只算第一条 fiber」的形态；
- 非连续输入（转置视图）沿 axis=0；
- `nancumsum`/`nancumprod` 无 NaN 时与普通扫描逐位相同；
- complex64 与 complex128 数值一致、dtype 保持；
- 退化：轴长 1、轴长 0、`axis=nil` 展平；
- `:out`：连续目标 + **非连续目标（列切片）**；
- 实数路径回归。

---

### 16.30 第十轮：NaN/Inf 语义、median 契约与 `:out` 非连续

#### 16.30.1 方法：从「dtype 保真」走到「数值对拍」

前几轮的扫描都在查「复数入参会不会掉回实数实现」，查不出「dtype 对、数值错」。
本轮沿用第九轮的数值对拍，并把覆盖面从 190 条扩到 **262 条**，新增四个方向：

| 方向 | 条数 | 结果 |
|---|---:|---|
| NaN / Inf 全组合（nan* 家族、掩码、`nan_to_num`、含 NaN 排序、全 NaN 切片） | 37 | 抓到 **median 不传播 NaN** |
| `axis` 元组（`(0 1)`/`(1 2)`/`(0 2)`/负元组 × keepdims） | 17 | 全部一致 |
| `:out` × 非连续（列切片 / 转置视图 / 原地别名 / matmul） | 7 | 全部一致 |
| einsum 补全（`...i,...i->...`、`ii->`、`ijk,jk->i`、`ijk->kji` 等 11 种） | 11 | 全部一致 |

结论：**除 median 外，`axis` 元组、`:out` 非连续、einsum 都没有静默错值**。

#### 16.30.2 缺陷一：`vt-median` 的复数版有三处与自家实数版分叉

三处都**不报错**，都是静默错值：

| # | 症状 | 根因 |
|---|---|---|
| 1 | `(vt-median a)` 对 `(3 4)` 输入返回 **3 个中位、形状 (3)** | `(ax (if axis (vt-normalize-axis axis rank) (1- rank)))` —— `axis=nil` 被写成「末轴」，而 numpy 与**本库实数版**都是「先 ravel 再取全局中位」，返回 0 维标量。连带 `:keepdims t` 给出 `(3 1)` 而不是 `(1 1)` |
| 2 | 空 fiber（轴长 0）返回 **0+0i** | `lo = (floor (1- 0) 2) = -1`，`%cx-fiber-starts` 在 size=0 时返回 0 条，循环不进 → 结果保持 `make-vt` 的 0。实数版这边是显式 `nanv` |
| 3 | 含 NaN 时返回**某个真实元素**而不是 NaN | 复数版按字典序排序后直接取中位，没有 NaN 检查；numpy 与实数版都传播 NaN |

修法：`axis=nil` 走独立的全局分支（ravel → sort → 取中位，keepdims 用全 1 形状）；
空 fiber / 含 NaN 一律写 `NaN+NaN i`；逐 fiber 判断用新助手
`%CX-ELEM-NAN-P` / `%CX-HAS-NAN-P`。

> 顺带补了一个洞：**复数 `vt-median` 原本没包 `WITH-FLOAT-SAFE`**（实数版包了）。
> SBCL 默认开着 `OVERFLOW/INVALID/DIVIDE-BY-ZERO` 陷阱，于是 NaN 判定
> `(= x x)`、`inf + (-inf)` 都会直接抛 `FLOATING-POINT-INVALID-OPERATION`。
> 现在整个函数体与实数版一致地包在 `WITH-FLOAT-SAFE` 里。

#### 16.30.3 缺陷二：前缀扫描用「单位元」而非「首元素直拷」

numpy 的 **accumulate**（`cumsum`/`cumprod`）与 **reduce**（`sum`/`prod`）在这一点上
**方向相反**，容易抄错：

```python
np.prod([inf+0j])            # (inf+nanj)   ← reduce 从单位元 1 起算，0*inf → nan
np.cumprod([inf+0j, 2+0j])   # [inf+0j, inf+nanj]  ← accumulate 首元素是**直拷**的
np.sum([-0.0+0j])            # 0j           ← 0 + (-0.0) = +0.0
np.cumsum([-0.0+0j])         # [-0.+0.j]    ← 首元素直拷，符号位保留
```

`%CX-SCAN-RUN` 原来两条路径都从单位元起算，于是：

- `cumsum` 把输入的 `-0.0` 洗成 `+0.0`（符号位丢失）；
- `cumprod` 的首元素经 `1 * (inf+0i)` → 复数乘法 `(1·inf − 0·0, 1·0 + 0·inf)`
  = `inf + nan i`，凭空多出一个虚部 NaN，**并且这个 NaN 会污染后续所有元素**。

修法：两条路径（连续 + dtype 转换回退）都改成「首元素直拷，从第二个起才 `OP`」。
正常数据下与旧实现逐位相同（`0+x`、`1*x` 对有限值精确），只在 -0.0 / Inf 上有别 ——
这正是本轮测试锁的点。

> **遗留（非复数问题）**：实数线的 `vt-cumsum`/`vt-cumprod` 走通用
> `VT-CUMULATIVE`（`acc` 从 0/1 起算），因此实数 `cumsum([-0.0])` 仍给 `+0.0`。
> 修它要动快路径 + 通用路径 + 轴分支共四处循环，风险与本轮主题（复数）无关，
> 故列为下一轮候选，不在本轮夹带。

#### 16.30.4 已核实「不是 bug」的三类差异

数值对拍剩下的 7 项不一致，逐项归因后都不是缺陷：

| 项 | 现象 | 归因 |
|---|---|---|
| `vt-qr` 的 Q/R | 与 numpy 差一个单位模相位 | QR 列可乘任意单位模相位；用重构恒等式判：`‖QᴴQ−I‖ = 1.1e-16`、`‖QR−A‖ = 6.7e-16` |
| `vt-lu` 的置换 | 返回 `(lu perm sign)`，约定不同 | 重构由 `complex-linalg-test` 覆盖 |
| `mean`/`nanmean` 含 ±Inf | numpy 的**另一分量**变 NaN，clvt 保持 0 | numpy 用 `sum × (1/n)` 做复数除法，`0 * Inf` / `NaN * 0` 会污染另一分量；clvt 逐分量除。clvt 的行为更符合直觉，且 `isnan` 判定一致 |
| 全 NaN 切片的 `nanmin` / 含 NaN 的 `median` | NaN 出现在哪个分量不同（`NaN+0i` vs `NaN+NaN i`） | 两者 `CZ-NAN-P` 都为真；属「用哪个 NaN」的表示差异 |

这条经验值得记住：**对拍不一致 ≠ 缺陷**。先把「是不是约定差异 / 是不是对方实现里的
浮点副作用」排除掉，再决定改不改。

#### 16.30.5 测试

`test/complex-nan-inf-test.lisp`（**77 断言**，已接入 `run-tests.sh`，共 49 套件）：

- **A. median 契约**（15）：`axis=nil` 的 0 维/值/keepdims 形状、与实数路径形状一致、
  轴归约值、c64 保持 dtype、NaN 全局/逐 fiber 传播、空轴与空输入 → NaN、`nanmedian` 对照；
- **B. 前缀扫描**（8）：`-0.0` 符号位、`cumprod` 首元素 `inf+0i` 无虚部 NaN、
  第二元素 `inf+nanj`、`nancumprod`、`axis=0` 每列首元素、c64、`axis=nil` 展平形状；
- **C. NaN/Inf 归约**（17）：`nan*` 家族含 ±Inf 与全 NaN 切片、`inf + (-inf)` 相消、
  掩码三兄弟、`nan_to_num` 三替换、含 NaN 排序、`nancumsum`；
- **D1. axis 元组**（12）：形状 + **逐元素手算对照**（fiber 和/积/字典序极值/norm），
  不借道任何库函数；
- **D2. `:out`**（7）：列切片落点**逐位置**核对（含「其余位置不动」）、转置视图、
  reduce keepdims 列视图、matmul、原地别名；
- **D3. einsum**（9）：`...i,...i->...`、`ii->`、`ij->ji`、`ijk->ik`、`ijk,jk->i`、
  `i,j->ij`、`ab,bc,cd->ad`、`ijk->kji`、`...ij->...i`。

> 测试进程**故意不关 FP 陷阱**（SBCL 默认全开）：既验证这些函数在用户开着陷阱时
> 也能安全产出 NaN，也顺带抓到了 16.30.2 提到的 `WITH-FLOAT-SAFE` 缺失。

---

### 16.31 第十一轮：弱标量提升（numpy NEP 50）

#### 16.31.1 方法：换个角度扫 —— 只比 dtype

前十轮都在比**值**与「复数会不会掉回实数实现」。本轮换成**只比 dtype**：
119 条表达式全喂 `complex64`（外加若干实数 dtype 对照），逐项与 numpy 的
结果 dtype 比对。值的正确性前几轮已覆盖，这一维是空白。

结果抓到一整类问题：**窄 dtype 遇到标量就静默升宽**。

#### 16.31.2 症状

| 表达式（`*` 表示张量） | 修复前 | 修复后 = numpy | 代价 |
|---|---|---|---|
| `f32* + 1.5` | float64 | **float32** | 内存 ×2、带宽 ×2、SIMD 通道白丢 |
| `f32* + 1` | float64 | **float32** | 同上 |
| `i32* + 1` | int64 | **int32** | 同上 |
| `i8* + 1` | int64 | **int8** | 同上 |
| `u8* + 1` | int64 | **uint8** | 同上 |
| `c64* + #C(1 2)` | complex128 | **complex64** | 内存 ×2 |
| `c64* + 1.5` | complex128 | **complex64** | 同上 |
| `f32* + #C(1 2)` | complex128 | **complex64** | 同上 |
| `f32* / 2` | float64 | **float32** | 同上 |
| `c64* ** 2` | complex128 | **complex64** | 同上 |
| `maximum(c64*, 标量)` | complex128 | **complex64** | 同上 |

这批都不是「算错」，而是**分配了双倍内存、跑了一半的有效通道**，而且完全不报错 ——
用户只有在自己 print dtype 时才会发现。

#### 16.31.3 根因：`ensure-vt` 抹掉了「谁是标量」

逐元素运算的 dtype 推断是：

```lisp
(apply #'vt-promote-type (mapcar #'vt-dtype inputs))   ; inputs 已全部 ensure-vt
```

`(ensure-vt 1.5d0)` 得到 **float64 的 0 维张量**，`(ensure-vt #C(1 2))` 得到
**complex128**。于是标量以「全强度」参与 promote，`f32 ⊔ f64 = f64`。

numpy 2.x（NEP 50）的规定相反：Python 标量是**弱**类型 —— 只贡献种类
（int / float / complex），不贡献宽度；宽度由**张量**决定。若张量是整型而标量是
浮点（整型装不下该种类），才落到该种类的默认宽度（float64 / complex128）。

CL 的 `1.5d0` 对应 Python 的 `1.5`，语义正好对齐，所以按 NEP 50 修是**对的**，
不是「为了迁就某个实现」。

#### 16.31.4 修法

新增两个小函数（`src/dtype.lisp`）：

```lisp
(%scalar-kind x)        ; 张量 → NIL；Lisp 数 → :INT / :FLOAT / :COMPLEX
(%op-final-dtype kinds dtypes)
;; 张量之间仍走 VT-PROMOTE-TYPE（100 格表不动）；
;; 弱标量只按种类并入：
;;   :int     → base 不变
;;   :float   → base 是整型族 ? :float64 : base
;;   :complex → base 整型族 ? :complex128 : f32→c64 / f64→c128 / 已是复数保持
```

接进四处（都遵循「用**原始实参**判断，不能用 ensure 之后的值」）：

| 位置 | 说明 |
|---|---|
| `vt-map` | 通用路径；有 `tensors`（原始）与 `inputs`（已 ensure）两份，直接配对 |
| `vt-fast-map` 宏 | 展开时先绑 `raws`（原始）再 `(ensure-vt raw)`；顺带把 `let` 改 `let*` |
| `vt-/` | 它自己算 true_divide 的浮点目标，改成只看**张量** dtype；并把 natural-dtype 显式下传（否则下游再 ensure 一次又丢标量身份）|
| `vt-pow` / `maximum` 家族 | 各自内联的 promote 换成 `%op-final-dtype` |

`vt-/` 那处踩过一次坑：第一版只改了 `float-target` 的判定，但随后仍把
**已 ensure** 的 `(first all)` 传给 `vt-fast-map`，标量身份二次丢失 →
`f32/2` 依旧 float64。教训：**这个信息只能沿着「原始实参」传**。

##### 附带优化：单元素输入预转（`%WEAK-CAST1`）

改完 dtype 后先测了一遍性能，发现 **c64 数组 + 标量慢了 27%**。原因不是新逻辑本身，
而是「弱标量」在 ensure-vt 之后是个**另一种 dtype 的 0 维张量**：

```
结果 c64，输入是 c64 数组 + c128 标量 → %VT-INLINE*-FAST 的前置条件
「全部输入与输出同元素类型」不成立 → 整个运算掉进 vt-map 的 funcall + 泛型 aref
```

修法：在 `vt-map` / `vt-fast-map` 里，把所有条件都满足的**单元素**输入先转成结果
dtype（`%WEAK-CAST1`）。代价 O(1)，类型检查能过、内联快路径回来；
**绝不整体预转大数组**。三个条件缺一不可：

| 条件 | 为什么 |
|---|---|
| `SIZE = 1` | 只处理广播标量；大数组预转会多一次全量拷贝 |
| dtype 与结果不同 | 否则无事可做 |
| 种类秩 `源 ≤ 目标`（int< float< complex） | 只允许无损方向。反过来 `float → int` 会截断：`(vt-< f64数组 1.5)` 的结果是 **int8 掩码**，把 `1.5` 转成 int8 变成 1，`nan<1` 直接判错 |
| 整数值装得下（`%SCALAR-FITS-P`） | `(vt-> 大数组 999000)` 的标量若转进 int8 掩码会变成 -8 —— 实测让 `argwhere(gt 999000)` 从 999 变成 999911 |
| 这组输入里至少有一个真张量（`SIZE>1`） | 全标量时没有「要对齐的大数组」，预转纯属添乱：`(vt-complex 1.0d0 2.0d0)` 内部就是 `(vt-map #'complex 1.0 2.0 :dtype :complex128)`，把实参转成复数会让构造函数收到 `#C(1 0)` 而报 `is not of type REAL` |

后三道保护全是踩出来的 —— 第一版只按「单元素 + dtype 不同」转，依次挂了
`robustness-test`（`nan<1`）、`manip-interop-test`（三个 argwhere oracle）、
`complex-test`（construct 组）。每次都是「预转把一个语义前提悄悄改掉了」，
而只有跑全量套件才看得见。

实测（2M 元素，min-of-7，同机同进程内对比）：

| 表达式 | 第十轮（升宽） | 第十一轮（弱标量） | 变化 |
|---|---:|---:|---|
| `f32 数组 + 1.5` | 77 ms | **4.1 ms** | **19×** |
| `c64 数组 + 1.5` | 163 ms | **5.4 ms** | **30×** |
| `c64 数组 + c64 标量张量` | 5.9 ms | 5.9 ms | 不变（本来就同类型） |

也就是说：这一改不只是「省一半内存」，还把窄 dtype 的「+标量」从**兜底路径**
拉回了内联路径。旧实现的 78/163 ms 恰恰是因为它把结果升成 f64/c128，
输入与结果类型不一致，一直在跑 funcall 兜底。

#### 16.31.5 有意保留的三处差异（都不是本轮的问题）

| 项 | clvt | numpy | 理由 |
|---|---|---|---|
| `det/inv/solve/pinv/cholesky/qr/matrix-power/svd/eigvalsh` 的 c64 输入 | complex128 / float64 | complex64 / float32 | 复数线代**恒在 double 下计算**（`vt-det` 文档已写明「返回 0 维 complex128 张量」），精度更高；改回单精度是**降级**，不做 |
| `ptp` / `cond` | Lisp 标量（complex / double） | numpy 标量（c64 / f32） | 这两个函数返回**原生数**而非张量（§5 的既定风格），原生数没有 dtype |
| 弱整数越界（`i8* + 300`） | 回绕 | `OverflowError` | 回绕是本库既有语义，改它会牵动所有整型路径 |

#### 16.31.6 测试

`test/dtype-weak-scalar-test.lisp`（**14 断言**，已接入 `run-tests.sh`，共 50 套件）。
dtype 断言之外，值断言同时覆盖「预转不改变结果」这一点。

按要求**只收录修复前确实错的用例**（11 个 dtype 断言，逐个在修复前的构建上复验过
「当时真的是错的」）+ 3 个值正确性断言（确认「不再升宽」没把值算错：`f32+1.5` 在
float32 内算仍是 1.5、`c64+#C(1 2)` 正确、`i8+1` 不溢出）。

张量×张量的 promote 100 格、c64 的其余 dtype 契约由既有套件覆盖，此处不重复 ——
测试只用来钉住**本次修掉的行为**，不是把扫描脚本搬进仓库。

---

### 16.32 第十二轮：拆掉覆盖式复数层（单一 DEFUN 分派）

#### 16.32.1 为什么必须拆

第十二轮之前，367 个导出里有 **159 个（43.3%）** 在运行期被替换。三处实证：

| # | 现象 | 数据 |
|---|---|---|
| 1 | 规模 | `*cx-originals*` 159 条，**全部是导出符号**，内部符号 0 条 |
| 2 | 静态不可分析 | 静态扫描（重名 `defun` ∪ 宏调用）只找到 155；`vt-argmax`/`vt-argmin`/`vt-nanargmax`/`vt-nanargmin` 的实数实现是**宏生成的**，源码里没有字面 `defun`，只能问运行时注册表 |
| 3 | **重载即静默丢复数** | 只重载 `src/elementwise.lisp`（改实数实现时最常见的动作）：`(vt-sqrt #C(-1 0))` 从 `complex128 #C(-0.0d0 1.0d0)` 变成 `float64 NaN`，**不报错** |

第 3 条是根因：`(setf (symbol-function 'vt-sqrt) <匿名 lambda>)` 装的包装，会被重载的实数 `defun` 整体覆盖；覆盖之后符号直接指向实数实现，复数分支永久消失。

#### 16.32.2 三种机制（现已拆掉两种）

| 机制 | 位置 | 形态 | 影响面 |
|---|---|---|---|
| M1 宏生成包装 | `complex.lisp` / `complex-ops.lisp` | `(setf (symbol-function ',name) (lambda (&rest keys &key ...)))` | 90 个；匿名 lambda → `declaim inline` 失效 |
| M2 `defun` 重定义 | 三个 `complex*.lisp` | 同名字再写一遍 `defun`，体内 `(%cx-orig 'name)` 转发 | 65 个；语义取决于加载顺序 |
| M3 时序预捕获 | 三个文件的 `eval-when` | 手写 63 条清单，漏写即「转发给自己」挂死 | 无机械约束 |

#### 16.32.3 目标形态

```lisp
(defun vt-sqrt (vt &key out dtype)
  "逐元素平方根。"
  (when (%cx-any-p vt)                    ; ← 唯一新增
    (return-from vt-sqrt
      (let* ((in (ensure-vt vt))
             (dt (%cx-result-complex-dtype (vt-dtype in) dtype "VT-SQRT")))
        (vt-fast-map (function %cop-sqrt) in :out out :dtype dt))))
  (%float-map #'sin-impl vt out (%infer-float-dtype vt dtype)))   ; 原 body 逐字不动
```

三个理由（决定了它是 `when` + `return-from` 而不是把 body 包进 `if`）：

1. `defun` 自带同名 block，`return-from` 天然可用 —— **原 body 的缩进与括号完全不动**，159 处改造可脚本化，diff 只增不减；
2. 单一 `defun` / 单一 lambda-list / 单一 docstring，**不存在「两处 lambda-list 漂移」**（见 16.32.5 的实证）；
3. 实数路径的代价是一次内联的 dtype 判断，比原来「funcall 匿名包装 + 同样一次判断」**净减少**。

三个新增的公共件都在 `src/dispatch.lisp`（`clvt.asd` 里紧随 `core`，早于所有归属文件）：

| 名字 | 作用 |
|---|---|
| `%cx-arg-complex-p` / `%cx-any-p` | 分派判据。VT 走内联两步判断，**list/vector 进 `%cx-deep-p` 递归** |
| `%cx-result-complex-dtype` | 复数数学函数的结果 dtype 规则（原在 complex.lisp，24 处归属文件要用，必须前置） |
| `%cx-reduce-dispatch` | 归约族的算子路由（14 个符号共用一个入口） |

> 判据为什么必须认 list：`(vt-cbrt (list #C(8 0)))` 是既有测试用例。浅判据
> （只看 VT）会把它送进实数路径 —— 那是 `%cx-tensor-p` 的老问题，本轮把分派
> 判据与「VT 快判」正式分开。

#### 16.32.4 本轮完成的范围（92 / 159）

| 阶段 | 内容 | 符号数 | 落地方式 |
|---|---|---:|---|
| S0 | `src/dispatch.lisp` + `%cx-any-p` / `%cx-deep-p` / `%cx-result-complex-dtype` / `%cx-reduce-dispatch` | 0 | 新建文件；`%cx-deep-p`/`%cx-tensor-p` 从 complex.lisp 前移 |
| S1 | 拒绝型（41） | 41 | 归属 defun 插 `(when (%cx-reject …) (error "…不支持复数…"))`；删 `%cx-def-unsupported` |
| S2 | 一元超越（24） | 24 | `return-from` + `vt-fast-map`；删 `%cx-def-unary` |
| S3 | 比较 6 / 谓词 3 / 极值 4 | 13 | 同上；删三个宏 |
| S4 | 归约族（14） | 14 | **`def-vt-reduce` 拆成「`%REDUCE-<NAME>` 实现 + `def-vt-reduce-facade` 门面」**，一个门面覆盖 14 个符号 |

S4 的拆分是必须的：`all`/`any` 的复数语义是「先把复数转成 int8 真值掩码，再喂回同一个实现」。
单一定义下没有 `%cx-orig` 可以回调，只能调实现函数 —— 这就是「实现 + 门面」两层的由来
（公开符号仍然只有一个 `defun`，`%REDUCE-*` 是内部名）。

#### 16.32.5 迁移中发现的 9 处 lambda-list 漂移（新 bug 类）

对比「同一符号的两处 `defun`」的 lambda-list，9 个不一致：

| 符号 | 实数版 | 复数版 | 后果 |
|---|---|---|---|
| `vt-svd` | `:max-sweeps 50 :tol 1e-14` | `:max-sweeps 60 :tol 1e-15` | **实数路径一直在用复数版的容差**（后加载者胜），且无人察觉 |
| `vt-matrix-rank` | `(tol 1e-10)` | `(tol nil)` | 默认判秩阈值不同 |
| `vt-clamp` | `(tensor vmin vmax &key …)` | `(vt &optional min-val max-val &key …)` | 签名根本不同，按哪个调用决定行为 |
| `vt-view` | `(vt &rest args)` | `(vt new-shape)` | 前者能吃位置参数，后者不能 |
| `vt-fromiter` | `(… (dtype :float64) …)` | `(… (dtype nil dtype-p) …)` | 缺省 dtype 不同 |
| `vt-nanmedian` | 无 `dtype` | 有 `dtype` | 传 `:dtype` 时一边报错一边接受 |
| `vt-angle` / `vt-round` / `vt-@` | 参数名不同 | 参数名不同 | 仅影响关键词名一致性 |

这是覆盖机制的第二个代价：**两处定义没有任何机制强制它们签名一致**。
16.32.3 的单一 `defun` 从结构上消灭这一类（只有一个 lambda-list，改不到别处去）。

> `vt-svd` 那一条建议单独处理：需要确认期望的默认容差，不能靠「谁后加载谁赢」。

#### 16.32.6 测试

`test/complex-dispatch-test.lisp`（**15 断言**，已接入 `run-tests.sh`，共 **51 套件**）。
它断言的不是数值而是**结构契约**：

- 92 个已迁移符号的 `(symbol-function …)` 解析出的**定义名必须等于符号本身**
  （即具名 `defun`，不是 `setf` 装的匿名 lambda）—— 这一条是「重载安全」的充分必要条件；
- list 入参走复数路径（`(vt-sqrt (list #C(-1 0)))` → `complex128` 的 `i`）；
- 拒绝型守卫对 list 也生效；
- 实数路径抽样不回归（`sqrt`/`sum`/比较/`maximum` 语义）；
- 复数语义抽样（字典序极值、`any` 的真值规则、`argmax` 的 `:int64`、`c64` 不被提升、`any` 的 NaN 规则）；
- 复数归约的 `:out` 契约。

#### 16.32.7 剩余工作（65 个，S5）

65 个 M2 符号尚未迁移，其中：

| 类别 | 数量 | 说明 |
|---|---:|---|
| 纯新增符号（无 `%cx-orig`） | 6 | `vt-@ vt-angle vt-conj vt-expt vt-imag vt-real` —— 不是覆盖，是**重复实现**（两处 `defun`，后者生效，前者是死代码），应删一处 |
| lambda-list 漂移 | 9 | 见 16.32.5，需先裁决语义 |
| 机械可迁移 | 50 | 形状统一：`(if (%cx-tensor-p …) <complex> (funcall (%cx-orig …) …))` → 抽出 complex 臂做 `%cx-*`，实数文件插守卫 |

迁移顺序建议按「历史 bug 密度」：`median`/`sort`/`argsort`/`cumsum`/`cumprod`/`mean`/`var`/`std`/`average`（第九、十轮的病灶）→ 集合族 → 线代 11 个。

**验收口径**：`grep -c "setf (symbol-function" src/*.lisp` 全为 0、`*cx-originals*` 删除、
`%cx-orig` 删除、三个文件的 `eval-when` 预捕获清单删除，然后把 `complex-dispatch-test`
的 92 个名字扩到 159。

---

### 16.20 累计变更文件清单（第一～十二轮）

| 文件 | 改动 |
|---|---|
| `src/dtype.lisp` | dtype 全集/存储集 + `vt-complex-dtype-p` / `vt-complex-real-dtype` / `vt-real-dtype-of`；`vt-promote-type` 复数分量规则；`vt-cast`/`vt-cast-fun` 复数与降级；值级原语 `%creal/%cimag/%cabs/%carg/%cconj/%cisnan/%cisinf/%cisfinite/%cis-zero/%clt/%cle/%cgt/%cge/%cneq` |
| `src/core.lisp` | `make-vt` 复数分支；`ensure-vt` 复数标量/序列推断；`%any-complex-in-sequence-p` |
| `src/io.lisp` | `%format-number` 复数打印；`vt-from-sequence` 含复数序列走通用 cast |
| `src/complex.lisp` | **新增**：语义层（原实现登记 + 分派包装 + 22 个复数内核 + 归约/扫描/排序/线代内核 + 未支持报错 + 加载期自检） |
| `src/package.lisp` | 导出 `vt-complex/vt-real/vt-imag/vt-conj/vt-conjugate/vt-angle/vt-iscomplexobj/vt-isrealobj` |
| `clvt.asd` | `(:file "src/complex")` 置于最后 |
| `test/complex-test.lisp` | **新增** 136 断言 |
| `test/gen_complex_expected.py` | **新增** numpy 期望值生成器（83 用例） |
| `test/complex-expected.sexp` | **新增** 生成产物（numpy 2.4.6 实测） |
| `src/complex-ops.lisp` | **第二轮新增**：数据搬运层（nan_to_num / view:dtype / repeat / unique 与集合 / searchsorted / nan 族 / correlate-convolve / tensordot-inner / nancumsum-cumprod / vdot / digitize 拒绝） |
| `src/manip.lisp` | `vt-repeat` 的三处 8 路元素类型分派各追加 2 个复数分支 |
| `src/io.lisp` | npy 的 `<c16`/`<c8` 编解码；lisp 格式的复数编码（有限 → `#C(...)`，含非有限 → `(:CX re im)`） |
| `src/complex.lisp` | `%cx-scan` 修正 `axis=NIL` 时的输出形状（展平为 1-D） |
| `test/complex-ops-test.lisp` | **新增** 128 断言 |
| `test/gen_complex_ops_expected.py` | **新增** 期望值生成器（97 用例） |
| `test/complex-ops-expected.sexp` | **新增** 生成产物（numpy 2.4.6 实测） |
| `src/complex-linalg.lisp` | **第三轮新增**：复数线代（LU/det/solve/inv/Cholesky/QR/SVD/厄米特特征值/pinv/lstsq/matrix-rank/matrix-power） |
| `src/linalg.lisp` | `vt-einsum` 白名单放开 complex64/complex128，结果 dtype 由输入推导 |
| `src/elementwise.lisp` | `vt-scale` 的 dtype 推断补「复数数组 × 实数标量 → 保持 complex dtype」 |
| `test/complex-linalg-test.lisp` | **新增** 108 断言（46 项差分 + 结构恒等式 + 契约） |
| `test/gen_complex_linalg_expected.py` | **新增** 期望值生成器 |
| `test/complex-linalg-expected.sexp` | **新增** 生成产物（numpy 2.4.6 实测） |
| `src/extensions3.lisp` | 第四轮：`vt-take-along-axis` 的元素类型分派补 2 个复数分支 |
| `src/complex-ops.lisp` | 第四轮：`vt-any` / `vt-all` 的复数真值分派（掩码 + 原实现） |
| `src/complex.lisp` | 第四轮：修复 `vt-outer` 复数分支（`vt-reshape` 多传 `:dtype`，此前从未跑通） |
| `test/complex-linalg-test.lisp` | **新增** 216 断言（46 差分 + 结构恒等式 + 108 压测 + 契约） |
| `test/complex-ops-test.lisp` | 第四轮扩到 **149 断言**（新增 outer / any / all / take-along-axis） |
| `test/run-tests.sh` | 注册 `complex-test` / `complex-ops-test` / `complex-linalg-test`（SUITES + DEFAULT_ORDER，防止漏接线） |
| `src/linalg.lisp` | **第五轮**：`vt-einsum` 的 P1-1/P1-2/BMM 三条通道放开复数（走 `*CX-EINSUM-GEMM-FN*` / `*CX-EINSUM-DOT-FN*` 钩子，见 §16.21）；`vt-trace` 补 `:offset` / `:axis1` / `:axis2`，收缩轴改 `-1`（§16.24）。**第六轮**：新增 `*SIMD-GEMM-RAW-FN*` 钩子与 `define-einsum-gemm`（`%EINSUM-GEMM-{F64,F32,I64,I32}`），BMM 通道接入 SIMD 分块内核并做布局归一化（NN / NT 转置 / C 转置 scatter / 兜底），见 §16.26 |
| `src/complex.lisp` | **第五轮**：新增 `%cx-gemm-strided`（带 strides 的复数 GEMM 内核）与 `%cx-einsum-dot`，并注册两个 einsum 钩子；`%cx-matmul-run` 增加 offset 形参、`%cx-batched-matmul` 改按 strides 递推（支持 batch 广播）、`%cx-dot-1d` 补 offset（§16.22）；复数 `vt-trace` 透传新参数 |
| `src/setops.lisp` | **第五轮**：新增 `%unique-axis` 及比较原语 `%axis-elem-lessp` / `%axis-elem-eq`；`vt-unique` 原实现改名 `%unique-flat` 并加 `:axis` 分发（§16.23） |
| `src/simd-matmul.lisp` | **第六轮**：新增 `%simd-gemm-raw`（裸数据 2D GEMM 入口，复用已有 AVX/FMA 内核与线程调度）并注册到 `*SIMD-GEMM-RAW-FN*`，供 einsum 的批量 GEMM 通道调用（§16.26） |
| `test/einsum-gemm-layout-test.lisp` | **第六轮新增** 36 断言：四条布局路径 × 各档尺寸、批量（含广播/批量 NT）、f32/i32/i64/c128/c64、转置输入/切片/非连续 `:out`/小矩阵、SIMD 关与单线程不变性；主断言 = 与 `vt-matmul` 逐位相等（§16.26.5） |
| `test/run-tests.sh` | **第六轮**：注册 `einsum-gemm-layout-test`（SUITES + DEFAULT_ORDER），现共 45 个套件 |
| `src/complex-ops.lisp` | **第五轮**：复数 `vt-unique` 增加 `:axis`（axis 路径复用实数层实现） |
| `test/setops-axis-trace-test.lisp` | **第五轮新增** 56 断言（37 项差分 + 契约 + 实数回归） |
| `test/gen_setops_axis_expected.py` | **第五轮新增** 期望值生成器 |
| `test/setops-axis-expected.sexp` | **第五轮新增** 生成产物（numpy 2.4.6 实测） |
| `test/run-tests.sh` | 第五轮：注册 `setops-axis-trace-test`（当前共 44 个套件） |
| `src/creation.lisp` | **第八轮**：`vt-from-array` 的 dtype 推断补 complex 两格；`vt-full` 的 `:dtype` 缺省改为「复数填充 → :complex128，否则 :float64」（§16.28） |
| `src/io.lisp` | **第八轮**：`vt-from-sequence` 的 `:dtype` 缺省改为「含复数叶子 → :complex128，否则 :float64」（§16.28） |
| `src/complex.lisp` | **第七轮**：`%def-cx-matmul-rows` 行内核 / `%cx-matmul-run` 重写 / `%cx-gemm-strided` 委托（§16.27）。**第八轮**：抽出 `%cx-pow-2` 供 `vt-pow` 与 `vt-float-power` 共用（零指数判定提到 NaN 之前）；新增 `vt-float-power` 复数分派；`vt-cbrt` 并入 `%cx-def-unsupported`（§16.28） |
| `src/extensions2.lisp` | **第八轮**：`vt-layer-norm` 新增复数分支（复数均值 + 实数方差/标准差） |
| `test/complex-gap-test.lisp` | **第八轮新增** 50 断言（dtype 契约 / numpy 数值对照 / 报错契约 / layer-norm 自洽性 / 回归网），注册后共 47 套件 |
| `src/complex.lisp` | **第九轮**：`%cx-scan-run` 的 axis 分支补 `AX-STRIDE` 层遍历（cumsum/cumprod/nancumsum/nancumprod 沿非末轴只算第一条 fiber 的静默错值，§16.29） |
| `test/complex-scan-axis-test.lisp` | **第九轮新增** 26 断言（2-D 两轴 / 3-D 三轴 / 负轴 / 转置视图 / c64 / 退化 / `:out` 连续与非连续 / 实数回归），注册后共 48 套件 |
| `src/complex.lisp` | **第十轮**：`%cx-scan-run` 与 dtype 回退分支改为「首元素直拷」（§16.30.3）；`vt-median` 复分支重写 `axis=nil` 全局路径 / 空 fiber 与 NaN 传播 / 补 `with-float-safe`；新增 `%cx-elem-nan-p`、`%cx-has-nan-p` |
| `test/complex-nan-inf-test.lisp` | **第十轮新增** 77 断言（median 契约 / 扫描首元素 / NaN·Inf 归约 / axis 元组 / `:out` 非连续 / einsum 补全），注册后共 49 套件 |
| `src/dtype.lisp` | **第十一轮**：新增 `%scalar-kind` / `%int-family-dtype-p` / `%op-final-dtype`（弱标量提升，NEP 50，§16.31） |
| `src/map-reduce.lisp` | **第十一轮**：`vt-map` 与 `vt-fast-map` 宏改用 `%op-final-dtype`（宏展开先绑原始实参） |
| `src/elementwise.lisp` | **第十一轮**：`vt-/` 的浮点目标只看张量 dtype，并显式下传 natural-dtype |
| `src/complex.lisp` | **第十一轮**：`vt-pow`（复数分支 / float-power）、`maximum/minimum/fmax/fmin` 换 `%op-final-dtype` |
| `test/dtype-weak-scalar-test.lisp` | **第十一轮新增** 14 断言（只收录修复前确实错的 11 个 dtype 用例 + 3 个值正确性），注册后共 50 套件 |
| `src/dispatch.lisp` | **第十二轮新增**：复数分派的单一入口（`%cx-arg-complex-p` / `%cx-any-p` / `%cx-deep-p` / `%cx-result-complex-dtype` / `%cx-reduce-dispatch`），在 `clvt.asd` 里紧随 `core` |
| `clvt.asd` | **第十二轮**：`(:file "src/dispatch")` 插在 `core` 之后 |
| `src/elementwise.lisp` | **第十二轮**：20 个拒绝型 + 10 个比较/极值 + 18 个一元，改为体内 `when`+`return-from` 分派（§16.32.3） |
| `src/nn.lisp` | **第十二轮**：9 个激活函数加拒绝型守卫 |
| `src/extensions3.lisp` | **第十二轮**：7 个拒绝型 + 3 个一元（`expm1`/`log1p`/`sign`）加守卫 |
| `src/reduce-stats.lisp` | **第十二轮**：4 个拒绝型（`histogram`/`interp`/`percentile` 族）+ 3 个谓词（`isnan` 族）；`def-vt-reduce` 拆成 `%REDUCE-<NAME>` 实现 + `def-vt-reduce-facade` 门面（14 个符号） |
| `src/indexing.lisp` | **第十二轮**：`vt-bincount` 加拒绝型守卫 |
| `src/complex.lisp` / `src/complex-ops.lisp` | **第十二轮**：删 `%cx-def-unsupported`/`-unary`/`-compare`/`-minmax`/`-predicate`/`-reduce`/`-nan-reduce`/`-bool-reduce` 八个宏与全部调用；`%cx-deep-p`/`%cx-tensor-p`/`%cx-result-complex-dtype` 迁出 |
| `test/complex-dispatch-test.lisp` | **第十二轮新增** 15 断言（结构契约：具名 DEFUN / list 入参 / 实数不回归 / 复数抽样 / `:out`），注册后共 51 套件 |

---

## 附录 C：实测缺陷复现脚本

```lisp
;; 保存为 /tmp/probe.lisp，运行：sbcl --non-interactive --load /tmp/probe.lisp
(require :asdf)
(push #p"/workspace/clvt/" asdf:*central-registry*)
(handler-bind ((warning #'muffle-warning)) (asdf:load-system :clvt))
(in-package :clvt)
(defmacro chk (name &form)
  `(handler-case (let ((v ,&form))
                   (format t "~&~a => ~a~%" ,name
                           (if (vt-p v) (list (vt-shape v) (vt-dtype v)) v)))
     (error (e) (format t "~&~a => ERROR: ~a~%" ,name e))))

;; D2: out dtype 不校验
(chk "D2 out dtype float64->float32"
     (vt-+ (vt-zeros '(2 3)) (vt-zeros '(2 3)) :out (vt-zeros '(2 3) :dtype :float32)))
;; D4: hypot inf+nan
(chk "D4 hypot inf/nan" (vt-hypot (vt-const '(1) +vt-dfloat-pos-inf+)
                                  (vt-const '(1) +vt-dfloat-nan+)))
;; D5: clip 单边界
(chk "D5 clip 无边" (vt-clip (vt-from-sequence '(1.0 2.0 3.0))))
;; D6: where 单参
(chk "D6 where 单参" (vt-where (vt-from-sequence '(1 0 1))))
;; D8: arange 无参
(chk "D8 arange 无参" (vt-arange))
;; D9: reciprocal 整数
(chk "D9 reciprocal int" (vt-reciprocal (vt-astype (vt-const '(3) 2) :int32)))
;; D1: slice :range（应报错；正确写法见同文件注释）
(chk "D1 slice :range" (vt-slice (vt-zeros '(2 6)) '(:all) '(:range 0 6 2)))
;; D1 正确写法
(chk "D1 slice 列表式" (vt-slice (vt-zeros '(2 6)) '(:all) '(0 6 2)))
;; 任务5: docstring 覆盖
(format t "~&docstring vt-var => ~a~%" (documentation 'vt-var 'function))
```

---

## 附录 A：本文件与既有文档的关系

| 文档 | 处理 |
|---|---|
| `README.md` | 保留结构；"设计约定"章节改为**引用本文件**，避免双份维护 |
| `REFACTOR-NOTES.md` | 保留（历史记录）；新增 v0.4.0 章节记录本轮重构 |
| `TEST-PLAN.md` | 保留；更新 NumPy 基准版本号；补充 §10.3 矩阵 |
| `CHANGELOG.md` | 新增 v0.4.0 条目 |

## 附录 B：实测命令速查（复现本文件的 numPy 结论）

```bash
# 类型提升 64 格
python3 -c "
import numpy as np
from itertools import product
ds=['int8','int16','int32','int64','uint8','uint16','float32','float64']
for x,y in product(ds,ds):
    print(f'{x}+{y}->{(np.zeros(1,dtype=x)+np.zeros(1,dtype=y)).dtype}')
"

# NaN/Inf 语义
python3 -c "
import numpy as np
print(np.maximum([np.nan,1.],[2.,3.]))   # [nan  3.]
print(np.fmax([np.nan,1.],[2.,3.]))      # [2. 3.]
print(np.sort([3.,np.nan,1.]))           # [1. 3. nan]
print(np.hypot(np.inf,np.nan))           # inf
"

# out 契约
python3 -c "
import numpy as np
a=np.arange(6.).reshape(2,3); base=np.zeros((2,6)); v=base[:, ::2]
np.add(a,a,out=v); print(v)              # 非连续 out 生效
"
```
