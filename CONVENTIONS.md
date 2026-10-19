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
| P0 | `vt-einsum` 复数 | 把 dtype 白名单换成「按 `vt-promote-type` 分配」，结果缓冲用 `vt-dtype->lisp-type` |
| P0 | 复数归约的 `axis` 元组 + `keepdims` 全路径覆盖测试 | 复用 `%cx-reduce-values` 的多轴路径，补 3D 用例 |
| P1 | 复数内联/SIMD 特化 | `%vt-inline*-fast` 增加 `(complex double-float)` 两格；`simd-matmul` 复数 GEMM |
| P1 | LU/SVD/特征值 | 需要共轭转置（`vt-ctranspose`）与酉变换；`vt-det` 复数版（实测 numpy det 为复数） |
| P2 | `real/imag` 真视图 | 需引入「分量视图」概念（`data` + 分量偏移），会动物理层，谨慎评估 |
| P2 | `.npy` 复数互操作 | `<c16`/`<c8` descr + 两个 float 分量编码 |
| P2 | `vt-npy` 打印/序列化的 `ComplexWarning` 等价物 | 引入可选告警通道 |

---

### 16.18 本章变更文件清单

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
| `test/run-tests.sh` | 注册 `complex-test`（SUITES + DEFAULT_ORDER，防止漏接线） |

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
