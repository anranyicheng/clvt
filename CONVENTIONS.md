# clvt 统一约定（Conventions）

> **版本**：对应 clvt v0.4.0（本文件即重构基准）
> **基准对象**：NumPy **2.3.5**（本机实测，非文档推测）
> **裁决原则**：凡与 NumPy 冲突者，**一律以 NumPy 为准**。
> **唯一例外**：`vt-arange`（见 [§9.1](#91-vt-arange唯一例外)）。

---

## 0. 本文件的地位与文档分工

本文件是 clvt 的**唯一语义契约来源**。`README.md`、`REFACTOR-NOTES.md`、
`TEST-PLAN.md`、`CHANGELOG.md`、`HISTORY.md` 中的描述如与本文件冲突，**以本文件为准**；
本文件与 NumPy 冲突，以 NumPy 为准。

**文档分工（三类内容，三个家，不许串位）**：

| 文件 | 只放 | 不许放 |
|---|---|---|
| `CONVENTIONS.md`（本文件） | 现行契约：表、规则、不变量、"现在是什么" | 轮次叙事、性能数字的逐年变化、"曾经不支持" |
| `HISTORY.md` | 编年史：某一轮改了什么、为什么、当时的实测数字、文件清单 | 被当作契约引用 |
| `CHANGELOG.md` | 面向使用者的逐轮摘要（允许无限增长） | 细节论证（指向 HISTORY） |
| 代码注释 | 读代码**必须**知道的：不变量、陷阱、为什么不用某写法 | 轮次编号、「第 N 轮改了什么」、绝对毫秒数 |

> **为什么这么分**：三类内容的**寿命不同**——不变量永久，决策理由半永久，
> 编年史与性能数字一次性。混在一处时，腐烂的编年史会连带把不变量一起拖下水
> （读者发现一处过时就不再相信整段）。历史教训：本文件里曾同时存在三处
> 「`vt-einsum` 不支持复数」（早已支持）和一处「建议用 SoA 做复数 GEMM」
> （实测 SoA 更慢），都因"只追加、不回改"而产生。

**六条机械门禁**（`test/check_docs.py`，随 `run-tests.sh` 自动执行）：

| 门禁 | 断言 |
|---|---|
| C1 | `src/`、`test/` 的注释里不得出现「第 N 轮」（编年史只属于 HISTORY/CHANGELOG） |
| C1b | 本文件正文不得出现「本轮 / 这一轮 / 上一轮 / 前几轮」—— 读者不知道「本轮」是哪一轮，改成「这里 / 这次改动」或把过程移到 HISTORY |
| C1c | 代码注释不得出现「搬迁墓碑」（`X 已迁至 Y`）。读者 `grep` 就知道符号在哪，墓碑只会在下次搬迁时又过时；确有约束要留时改写成现状陈述（同行带「因为 / 否则」的放行） |
| C2 | 所有 `§x.y` / `§H.x` 引用必须能解析到真实小节（防改编号后锚点漂移）。扫描范围含 `HISTORY.md` / `CHANGELOG.md` —— 编年史里引用契约小节同样会漂移 |
| C3 | §16.x 三级编号必须递增（新小节追加在末尾，不许插队） |
| C4 | `HISTORY.md` 必须存在且被本文件引用（防契约与编年史再次合流） |

> **C2 的已知局限**：它只检查锚点**存在**，不检查**指向对不对**。清理过程中
> 发现过 9 处「锚点还在、但那个编号已被别的内容占用」的静默错引用。
>
> 试过一个改进方案（要求引用所在行的关键词与目标小节标题有交集），**误报率
> 54%**（41 处检查 22 处误报 —— 例如 `§4.3` 的正文没写 `:out` 字面量，但引用完全
> 合理）。结论：静态判「指向错」不可行，因为引用行与目标小节用的是不同的措辞。
> 真正有效的办法是**重排/删除小节时 grep 全库引用并逐条确认**，这条已写进下面的
> 「重排小节时的检查单」。

**重排/删除 `§x.y` 小节时的检查单**（每一条都有人踩过）：

1. `grep -rn "§x.y" src/ test/ *.md` —— 代码注释、测试、三份文档都要看；
2. 对每个命中确认的是**语义**而不只是编号存在（编号被别的内容占用最危险）；
3. 保持**被引用的编号不变**，宁可让编号跳号也别重排（重排的连带修改面是 O(引用数)）；
4. 改完跑 `python3 test/check_docs.py`，它现在也扫 `HISTORY.md` / `CHANGELOG.md`。

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

**实现（选路，不影响语义）**：非连续时的物理落地统一走 `%gather-contiguous`
—— 按元素类型特化的「末轴 DOTIMES 展开 + 其余轴 odometer」遍历，
落点固定为行主序连续。`vt-copy` 的非连续分支与 `vt-astype` 的非连续回退
走同一条路（同一个落点，保证三者逐位可比）。

`vt-astype` 在这条路之外**还有一条按 run 的融合**：非连续视图若内层
连续 run ≥ `+astype-fuse-min-run+`（切片/跨步视图，如 `a[::2,:]`），
就直接按 run 一趟写进目标数组，不落中间连续副本 —— 与 `vt-copy` 的
「先收集再转换」相比省掉一块全尺寸临时分配和两趟全量流量。
两条路的落点必须**逐位相同**（都是行主序连续、同 dtype），差别只在
搬几趟；run 太短（转置/翻转，run=1）时逐 run 的调用开销大于收益，
必须落回 `%gather-contiguous`。

> **注意（自递归）**：`%gather-contiguous` 遇到未知底层元素类型时**不得**
> 回写 `vt-contiguous`（两者互相调用会死循环），必须直接 `vt-copy-into`
> 到新分配的连续数组。

---

## 3. dtype 系统

### 3.1 逻辑 dtype（9 种）与物理存储

| 类别 | dtype |
|---|---|
| 逻辑 dtype | `:bool :float64 :float32 :int64 :int32 :int16 :int8 :uint8 :uint16` |
| 物理存储 | `:float64 :float32 :int64 :int32`（其余逻辑 dtype 以窄类型物理存储） |
| 布尔 | 独立逻辑 dtype `:bool`，物理存储 `(unsigned-byte 8)`，1 字节，取值只允许 0/1 |

> **约定**：`:bool` 是**第九种逻辑 dtype**（NumPy 的 `bool_` 对应物），完整契约见
> [§17](#17-布尔-dtypebool)。比较、逻辑、谓词、`any/all`、`isclose`、`array_equal`、
> `isin` 这些「NumPy 返回 `bool_`」的函数，缺省结果 dtype 一律是 `:bool`。
>
> 历史背景：`:bool` 引入之前，这些函数用 `:int8` 的 0/1 承载，靠「文档写一句
> 语义等价于布尔」维持约定。该做法有两个无法用文档解决的漏洞：
>
> 1. **算术泄漏** —— `:int8` 结果能被当成整数继续算：`(vt-+ mask mask)` 得 2，
>    `(vt-sum mask)` 得 int8 且会溢出，而 NumPy 的 `bool_` 参与算术按逻辑运算走、
>    `sum` 升 int64；
> 2. **类型不可区分** —— 「布尔掩码」与「真正的 int8 数据」在 dtype 上完全相同，
>    吃掩码的函数（`vt-where` / `vt-compress` / `vt-select`）无法表达、也无法校验
>    入参语义。
>
> 向后兼容：所有布尔类函数仍接受显式 `:dtype :int8`（或其它数值 dtype），此时
> 得到旧的 0/1 整型掩码，行为与引入 `:bool` 之前逐位相同。

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
- `*simd-matmul-thread-count*` 并行分块**不得**改变 GEMM 的累加顺序语义（当前按 k 维顺序累加，分块须保持）

**两条「合法的重排」**（只换遍历次序、不换任何单个输出元素的累加顺序，
因此结果逐位相同）：

| 重排 | 内容 | 为什么逐位相同 |
|---|---|---|
| GEMM 分块 | MC/KC/NC 三级分块 + 寄存器微块 | k 仍按 kc0 → kc-end 递增、块内 l 递增；kc 边界处寄存器写回 f64/f32/i32/i64 无损 |
| 归约行累加 | 内层维整体当累加器，按行推进（`%kernel-row-acc`） | 每个输出元素看到的 r 仍是 0,1,2,…；翻到外层的只是「不同输出元素之间」的循环 |

**GEMM 并行调度契约**：

| 项 | 约定 |
|---|---|
| 划分方式 | C 的 **(i, j) 二维矩形网格**（`%gemm-grid`），不得只切行 —— 只切行时每个线程都要过一遍 B，线程数翻倍则 B 的 DRAM 流量翻倍 |
| 线程数上限 | `min(16, 核数-1)`，可由 `*simd-matmul-thread-count*` 覆盖。**不是**越多越好：超过 16 后线程各自的分块工作集互相踢出 L2，实测出现过 3 倍退化 |
| 小任务 | `m·k·n < *simd-matmul-thread-threshold*` 时串行，避免线程创建/汇合摊不薄 |
| 回退 | 内核返回 NIL 表示不接管，由调用方回退（§8.3） |

**GEMM 累加器初值属于契约，不是优化项**：微内核必须**从 C 的现值开始**
累加（先载入、末尾原样写回），不得「清零后末尾再 `vaddpd`」。后者少一次
载入、实测快 3%，但舍入链变成 `((0 + a₀b₀) + …) + C`，在 C 远大于 Σab 时
会丢掉 Σab 的低位 —— 对拍出现 `-4.99e-16` 而非 0。

**f64 GEMM 的「逐位相同」边界**：`%simd-matmul-f64`（sb-simd 路径）把列分成
三段 —— 8 列微块、按 4 列对齐的整宽向量、最后 `n mod 4` 列走**标量**
`(incf c (* a b))`（两次舍入）。AVX-512 内核（§7.4 的 `%simd-matmul-f64-512`）
对这些列用的是 FMA（一次舍入），因此：

| 情形 | 要求 |
|---|---|
| `n mod 4 = 0` | **严格逐位相同**（含 1000³ 的全部 10⁶ 个元素） |
| `n mod 4 ≠ 0` | 差异**只能**落在最后 `n mod 4` 列（`j ≥ n - (n mod 4)`），且相对误差 ≤ 1e-15；512 侧更精确 |

差异一旦出现在别的列，就是内核 bug，不能归入「舍入差异」。
`test/gemm-avx512-test.lisp` 的 `%tail-start` / `check-bits` 是这条的可执行版本。

### 7.4 性能快路径清单（谁在什么时候接管）

所有快路径的共同约束：**不满足条件时返回 NIL / 落到通用路径**，且结果与
通用路径**逐位相同**（§1 铁律 1/2）。新增快路径必须同时给出「选路条件」
与「逐位一致」的测试。

| 层 | 快路径 | 选路条件 | 回退 |
|---|---|---|---|
| GEMM | `%simd-matmul-{f32,i32,i64}`（`simd-matmul.lisp`） | 2 维、dtype 属存储级、`m·k·n ≥ *simd-matmul-threshold*`、out 连续且 dtype 精确匹配 | `vt-einsum` 通用路径 |
| GEMM（f64） | `%simd-matmul-f64-512`（`gemm-avx512.lisp`，运行时发射的 AVX-512 机器码；8 行 × 16 列微块，列尾走掩码内核） | 同上一行，**且** `avx512-gemm-kernels` 非 NIL —— 即 x86-64 + `/proc/cpuinfo` 有 `avx512f` + `*avx512-gemm-enabled*` 为真。`m mod 8` 的行尾仍交回 `%simd-matmul-f64` | `%simd-matmul-f64`（sb-simd AVX2） |
| 归约（全归约） | `%kernel-global` | 输入连续、`axis=nil`、dtype 属 4 种存储级 | `%kernel-general`（stride 驱动 odometer） |
| 归约（沿轴） | `%kernel-row-acc`（`inner ≥ +row-acc-min-inner+`）/ `%kernel-single-axis` | 输入与 out 都连续，且**归约轴在 shape 里连成一段** `[a,b)` —— 满足时多层轴等价于一层：`outer=∏[0,a)`、`red=∏[a,b)`、`inner=∏[b,rank)`；单轴天然满足，`(0,1)`/`(1,2)` 满足，`(0,2)` 不满足。窄整型同条件（内层用 `%small-int-axis-form` 分派）。行累加额外要求 out 不与输入同数组 | `%kernel-general` / `%kernel-small-general`（窄整型） |
| 归约（arg 族） | `%kernel-single-axis` | 单轴 + 存储级 dtype（numpy 的 `argmax`/`argmin` 不接受轴元组；窄整型留在通用路径） | `%kernel-general` |
| 类型转换 | `vt-astype` 的 `spec` 特化（10×10 + 4×8 窄整型），提成局部函数 `%cast-run` 后**两条落点共用同一张表** | 连续视图 → 整数组一次；非连续且内层 run ≥ `+astype-fuse-min-run+` → 逐 run 一趟直接写目标 | `%gather-contiguous` → 递归 astype（run 太短或未知元素类型） |
| 连续化 | `%gather-contiguous` | 非连续视图 | 未知元素类型 → `vt-copy-into` |

### 7.5 f64 GEMM 不切 sb-simd 的 512 位通路（已实测否决）

SBCL 2.6.9 起 `sb-simd` 带 AVX-512（`(require :sb-simd)` 后有
`sb-simd-avx512f` 包，f64 侧导出 `f64.8` / `f64.8-fmadd` / `f64.8-broadcast`
等 1300 余个符号）。**它确实可以直接调用** —— `f64.8` 系列符合
`define-blocked-gemm` 的元素算子协议，一行就能生成 512 位微块，不需要任何
VOP 或机器码。但**不要**用它替换 `%simd-matmul-f64-512`：

| 1000³ 单线程 | ms | GFLOPS | 相对自写内核 |
|---|---:|---:|---:|
| **自写机器码 8×16** | **27** | **74.1** | 1.00× |
| sb-simd 512，微块 8×8 | 46 | 43.5 | 0.59× |
| sb-simd 512，微块 8×16 | 53 | 37.7 | 0.51× |
| sb-simd 512，微块 4×16 | 66 | 30.3 | 0.41× |
| sb-simd AVX2，微块 4×8 | 80 | 25.0 | 0.34× |
| numpy 单线程 | 21.7 | 92 | 1.24× |

500³ 上同此趋势（自写 3.0 ms / 83 GF，sb-simd 512 最好 8.0 ms / 31 GF，
2.7×）—— 小规模下 sb-simd 的固定开销与搬移更摊不薄。

反汇编（`disassemble` 微块函数）给出的原因很具体：`sb-simd` 的
`f64.8-fmadd` 是**纯函数 + 值语义**，结果只能回落到固定的 `ZMM0`，于是每条

```
VFMADD213PD ZMM0, ZMM1, ZMM26
VMOVUPS    ZMM26, ZMM0        ; ← 40 条 FMA 后面跟了 34 条这个
```

都带一次搬移，前端吞吐被吃掉近一半；且 zmm 的分配范围窄，spill 频繁。
自写内核用 `vfmadd231pd` **就地把累加器当目标**，一条 FMA 就是一条。

**zmm 的 VOP 也仍然写不出来**（2.6.9 复核过）：核心映像无
`sb-vm:double-avx512-reg` 这类 SC，`:scs` 写它会被当成函数调用；
`sb-assem:define-instruction` 依旧为 NIL，只有 `inst`；EVEX 编码表不在
`SB-X86-64-ASM` 里。`sb-vm::float-avx512-p` / `load-float-avx512` 这类
辅助函数存在但接不进 VOP 的 `:generator`。

结论：**自写机器码是当前唯一能拿满 512 位的路线**，sb-simd 的 512 只适合
「不想自己发字节、又能接受 0.5×」的场景。数据与踩坑见 `HISTORY.md` §H.3.25。

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
| `vt-div` | 对标 `np.floor_divide`：整数截断除法，整数零除返回 0（不报错）；浮点按 IEEE | ✅ 已对齐|
| `vt-maximum/minimum/fmax/fmin` | NaN 传播 vs 忽略 | ✅ 正面样板（D10） |
| `vt-clip` | 对齐 numpy 2.3.5 `clip(a, a_min=None, a_max=None)`：两边界均可省（返回原值）；只给 max 合法；**只给 min 报错**；`a_min > a_max` → 全为 `a_max` | ✅ 已对齐|
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
| 比较运算 | 返回 `:bool` 布尔张量（§17）；显式 `:dtype :int8` 得旧的 0/1 整型掩码 | ✅ 已对齐 |
| 三角/反三角 | `asin/acos/atan` 命名（非 arc*），越界返回 NaN | ✅（命名见 §3.1.1） |
| **docstring 覆盖** | 每个公开函数须有文档 | ⚠️ `vt-var` 等为 NIL，需全量审计（任务5） |
| 激活函数族 | `vt-sigmoid`/`vt-relu`/`vt-leaky-relu`/`vt-swish`/`vt-softplus`/`vt-gelu`/`vt-mish`/`vt-hard-tanh`/`vt-hard-sigmoid`：输出**恒为浮点**，整数输入提升 `:float64`，float32 保持 float32（见 §3.4） | ✅ 已对齐（新增 softplus/gelu/leaky-relu） |
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
| **浮点格式随 numpy 版本变** | `gen_probes*.py` 用 `str(np.float64)` 输出期望值：numpy 2.4 给 `0.27279899999999996`、2.5 给 `0.272799`。**同一份代码在不同 numpy 下 codegen 必然漂移**，但数值只差 1 ulp（比较用 `:tol`）。遇到跨版本漂移时不要盲目「重新生成再提交」——先确认是格式化差异还是真语义变化（这里遇到过：本机 numpy 2.4.6 报漂移，上游 2.5.0 正常） |
| **重新生成后先 `git add`** | 门禁结束时 `git checkout -- <生成物>` 还原工作区，否则改动会被门禁自己抹掉 |
| **门禁默认按语义比较，不按字节** | `check_key_sync.py` 默认只要求 key/dtype/shape/比较方式一致 + 数值在 dtype 级容差内（float64 1e-12、float32 1e-6），跨版本的格式化漂移只报 WARN；要钉死文本用 `--strict`。否则「换一台机器就整套测试起不来」（§16.34.3 实测） |
| **生成器退出码必须检查** | 生成器崩溃时文件不重写，`committed == regenerated` 会**假通过** —— 这条路径必须单独拦，比没有门禁更危险 |

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
**裁决**：✅ **已修复**——现报错 `vt-fast-map: :out dtype FLOAT32
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
**裁决**：✅ **已按 numpy 2.3.5 对齐**。签名改为
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
**裁决**：✅ **已对齐 NumPy**——clvt `vt-reciprocal` 对 int32 输入
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
**裁决**：✅ **已修复**——整数零除返回 `0`，dtype 保持整型；
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
| 扩展 | `extensions.lisp` / `extensions2.lisp` / `extensions3.lisp` | 全部透传下游原语，已自动获得统一契约（比较/逻辑类返回 `:bool`，见 §17） | ✅ |
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
| 覆盖缺口审计 | **新增 28 套件全绿**（`uncovered-coverage-test` 136 断言）；6 处缺陷修复见 §15| — |

---

## 15. 覆盖缺口审计与缺陷修复

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
| F4 | `vt-= vt-/= vt-< vt-<= vt-> vt->=` | **dtype 契约违反**（§3.1） | 返回 `float64`，应布尔 | 缺省 dtype `:bool`（§17） |
| F5 | `vt-positive-p vt-negative-p vt-zero-p vt-nonzero-p vt-even-p vt-odd-p` | 同上 | 返回 `float64` | 缺省 dtype `:bool`（§17） |
| F6 | `vt-isnan vt-isinf vt-isfinite` | 同上 | 返回 `float64` | dtype 固定 `:bool`（§17） |

> F4–F6 的裁决依据：§3.1 / §9.2 的「比较运算返回布尔」。**早期落地用的是
> `:int8` 的 0/1**（当时还没有布尔 dtype），值语义与 NumPy 一致但标签不同；
> 引入 `:bool` 后（§17）这些函数改回真布尔，`:dtype :int8` 仍可退回旧行为。
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
| `np.array([1+2j], dtype=float)` | **TypeError**（`float() argument must be … not 'complex'`） | 显式实数 `:dtype` 时**按 §16.4 取实部**（不抛错）——这是 clvt 的 `vt-from-sequence` 既有「安全转换」立场，已记录为差异 D-C2 |

**约定**：`vt-zeros / vt-ones / vt-empty / vt-full / vt-const / vt-zeros-like / vt-ones-like`
的 `:dtype` **必须显式写** `:complex128` 才能得到复数（缺省仍是 `:float64`）。
`vt-const '(1) #C(1 2)` 会按 §16.4 丢虚部 —— 这是**已接受的陷阱**，
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

#### 16.4.1 执行层：转换必须落在按元素类型展开的循环里

「经由此出口」指**语义**（每一格取实部 / 补零 / 回绕的规则只有一处），
**不是**指运行期调用 `vt-cast-fun` 返回的闭包。转换是逐元素算子，闭包 + 泛型
`aref` 会让每个元素都走一次「装箱 → 类型检查 → 解包」：

| 写法 | 1M 元素 c128→c64 | 结论 |
|---|---:|---|
| `(funcall (vt-cast-fun dst) (aref in p))`（泛型 aref） | 31 ms | **禁止**用于批量路径 |
| 特化 `(simple-array (complex double-float) (*))` 循环 | **2 ms** | 唯一允许的形态 |

**契约**：

1. 连续视图的 10×10 组合必须各有 `spec` 分支（`src/core.lisp` 的 `vt-astype`），
   兜底的通用 fallback 只服务「未知底层元素类型」，不服务任何已登记 dtype；
2. 非连续视图先做**同 dtype 的特化收集**（`%gather-contiguous`），再走上面的
   连续特化 cast；同 dtype 时收集结果**就是**答案（一趟）；
3. 同 dtype 且物理布局相同（含复数 interleaved）一律 `replace` 块拷贝，
   不得逐元素重建 `complex` 对象；
4. 判据：`(vt-astype 复数数组 :complex64)` 的耗时不得超过同规模
   `(vt-astype 实数数组 :float32)` 的 10 倍（`test/astype-cast-test.lisp` 有护栏）。

**同一条规则约束「拷贝」而不只是「转换」**（`vt-copy` / `vt-contiguous` /
`vt-copy-into` 共用 `%copy-generic`，它也是按元素类型展开的）：

5. `%copy-generic` 的 typed-walk 分派表必须覆盖 `(complex double-float)` 与
   `(complex single-float)`。缺这两格时复数会掉进 `%copy-generic-slow` 的泛型
   `aref`：1000×1000 转置视图 17 ms vs 6 ms（`test/test-copy-into.lisp` 有护栏）；
6. `%gather-contiguous` 与任何手写多维 walker：**末轴必须 `DOTIMES` 展开，
   其余轴才走 odometer**。全 odometer 每元素约 5 ns（索引数组 + 循环体），
   1M 元素就是 5 ms —— 与元素类型无关的固定开销；
7. 元素类型分派表的新增判据：`grep` 出所有 `(equal et 'double-float)` /
   `(equal in-et 'double-float)` 形式的表，逐张确认复数两格是否也需要；
8. 同 dtype 的**整段块拷贝**一律走 `%memcpy`（core 里那**一份**按元素类型展开
   并带两端 `(the …)` 声明的实现）。`join.lisp` 的 `%block-copy` 只是它的
   实参顺序包装。原因：`replace` 只有两端都被编译器确认为同类 specialized
   `SIMPLE-ARRAY` 时才走字节级 memmove，缺声明就退化成逐元素泛型序列操作 ——
   1M complex128 实测 **8 ms vs 0.2 ms**（f64 也白付一次 SAETP 分派）。
   判据：`grep -n "(replace " src/*.lisp | grep -v "the (simple-array"`
   应当只剩 `%memcpy` 自己的兜底分支。

**已知的固定成本**：分配一个 16 MB 缓冲区（`make-array` 的页清零）约 1~3 ms，
这是 `vt-copy` / `vt-astype` 的**下限**，与算法无关。分配本身不要用
`make-vt … 0`（多一趟全量填充，1M complex128 = 3 ms）—— 见 `%alloc-vt`。

> 早先本节写过「SBCL 没有 16 字节元素的 basher」，**那是错的**：当时看到
> 5~10 ms 是因为 `replace` 两端缺类型声明（第 8 条）。补上声明后
> 1M complex128 只要 0.2 ms，与 `double-float` 同量级。教训与 §H.3.11 一样：
> 测量结论必须能追到具体的代码形态，否则很容易把「少写一个声明」记成「平台缺陷」。

> 差异 D-C1 的两条后果在这里同样成立：复数→实数**静默**丢弃虚部（无警告），
> 且该语义在快慢两条路径上必须逐位一致（含 `-0.0` 符号位与 NaN/Inf 传播）。

---

### 16.5 算术与幂

| 运算 | 语义 | 备注 |
|---|---|---|
| `+ - * /` | 复数四则（分量代数） | **不做共轭**（numpy 的 `*` `/` 也不做；共轭点积必须显式 `vt-conj`） |
| 一元 `-` | 分量取负 | `vt-negative` |
| `square` | `z*z` | ≠ `|z|²`；后者是 `vt-abs` 再平方 |
| `pow` / `**` | `expt` 复数幂；指数可为实数或复数 | dtype 按 §16.3 提升 |
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
| `argsort` | 同上，返回下标 | `vt-argsort` | 稳定性：稳定**下标归并**内核（`%cx-argsort-fiber`） |
| `median` | 排序后取中位（偶数取两中位均值） | `vt-median` | 复数均值不是实数 |

**为什么必须专列**：字典序不是「任意一致的全序」——`1+9j < 2+0j` 为真（实部优先），
而按模长比较则为假。任何用 `abs` 或分量加权做比较的实现都会与 numpy 分歧。

#### 16.8.1 执行层：极值与排序内核的四条硬约束

上表定的是**语义**，本节定**执行形态**。三条都是踩过的坑，违反时结果正确但慢一到
两个数量级（或慢且错）：

1. **内核里的 NaN 判定只能用免屏蔽的 `%cx-nan-p`**，不能用边界版 `%cisnan` /
   `%cisinf` / `%cisfinite`。后三者每次调用都套一层 `with-float-traps-masked`
   （单次约 200 ns），极值内核每元素判两次 —— 1M 元素光「屏蔽/恢复陷阱」就是
   800 ms（`vt-maximum` 实测 804 ms → 3 ms，**280×**）。前置条件（调用方已在
   `with-float-safe` 内）由 `vt-map` / `vt-fast-map` / `%cx-reduce-*` /
   `%cx-arg-reduce-run` 共同保证，分工见 §16.14。
2. **`argsort` 必须走下标归并内核**（`%cx-argsort-fiber`），不得用
   `(stable-sort idx (lambda (a b) ...))`：后者每次比较是一次 `funcall` +
   两次跨步 `aref`（约 50 ns/比较），1M 元素约 20M 次比较 → 2.1 s。
   稳定性由「插入排序（n ≤ 16）+ 自底向上归并」共同保证，与 `stable-sort` 等价。
   归并**必须双缓冲乒乓**（每趟交换 src/dst），不得把每段归并结果 `replace` 回原
   缓冲区：n=1M 时第一趟就有 50 万次两元素拷贝，调用开销盖过数据量。
3. **`topk` 只允许算一次序**。它只需要 k 个元素，`vt-sort` + `vt-argsort` 两趟
   O(n log n) 是纯浪费；正解是 `vt-argsort` 一趟 + 沿轴按下标取回值
   （`%gather-axis-k`）。判据：`topk ≤ 1.4 × argsort`。
4. **被 `vt-fast-map` 内联调用的标量内核必须定义在 `src/complex-kernels.lisp`**
   （即早于 `map-reduce` / `elementwise*` / `extensions*` / `reduce-stats`）。
   放晚了的后果不是报错而是**静默退化成函数调用**：定义在 `complex.lisp` 时
   `vt-=` / `vt-<` / `vt-isnan` 对 1M complex128 分别是 60 / 45 / 25 ms，
   前移后 4 / 5 / 1 ms。
   **探测器**：编译日志里的
   `Proclaiming X to be INLINE, but N calls to it were previously compiled`
   —— 出现一行就说明某个内核放晚了 N 个调用点，逐条消除。本库当前为 0 条
   `%cx-*` 命中（另有 3 条与复数无关：`vt-p` / `%copy-default-rng` /
   `parse-subscript-tokens`）。
   **探测器有盲区**：它只抓「符号**已存在**、inline 声明迟到」的情形。
   若内核**根本没定义**就被 `(function %cop-xxx)` 引用（`%cop-sign` /
   `%cop-sinc` / `%cop-expm1` / `%cop-log1p` 原住在 `complex-ops.lisp`，
   而调用点在更早的 `extensions3` / `elementwise2`），编译期只当普通函数调用，
   **一条告警都不会有**。这类只能靠静态扫描：
   `(文件加载顺序, (function KERNEL) 引用位置, KERNEL 的 defun 位置)` 三元组，
   定义晚于引用即是缺陷（实测 1M complex128：`vt-sign` 104 → 26 ms）。
   `test/complex-elementary-test.lisp` 的 D 段是它的回归网。

> **`%gather-*` 的偏移契约**：`vt-contiguous` 对「已经连续」的输入**原样返回**，
> 而 `narrow` / `flip` 出来的视图可能 offset 非零却仍满足连续判定
> （单元素轴翻转后只剩 `offset=7`）。任何「取 `(vt-data t)` 后按扁平下标读写」的
> 代码都**必须**把 `(vt-offset t)` 加成基址，否则读到别的元素 —— 这是静默错值。
> 边界情形（k=1 的 `topk`）曾经因此返回全数组最小值而非最大值。

#### 16.8.2 归约与选择：两处「类型被放宽」的陷阱

§16.8.1 管的是「函数放错地方 / 用错原语」，本节管**变量类型被推断放宽**。
两者都不报错、都只在性能上现形，且都能差一个数量级。

1. **归约的累加器必须显式声明类型**。写成
   `(let ((acc (aref d 0))) … (setf acc (kernel acc v)) … (vector acc))` 时：
   `(vector acc)` 接受任意类型，加上内核的**派生返回类型是 T**（形参无声明 →
   返回 a 或 b → T），SBCL 会把 `acc` 放宽成 `T`，于是每次内核调用都在装箱/
   解包复数。补 `(declare (type (complex double-float) acc))` 后
   1M complex128 的 `vt-amax` **41 ms → 3 ms**。
   **判据**：任何归约内核的累加器都要有 `(type …)` 声明；`%def-cx-reducer`
   式的模板若不带类型只能用于「初值与步进函数都返回确定类型」的场合
   （`+`/`*` 可以，`%cop-max` 不行）。
2. **混合类型的逐元素算子必须单列特化内核**。`vt-where(mask, x, y)` 的掩码
   恒为 `:int8`、数据是 `float64` / `complex128`，三条输入的 dtype 永不相等，
   因此**永远进不去** `%vt-inline3-fast` 的「输入输出同类型」前置条件，只能
   退回 `vt-map` 的闭包（每元素一次 funcall + 一次 `vt-cast`）：
   1M complex128 实测 **32 ms → 5 ms**（`%select-fast`，按元素类型展开 10 格）。
   `vt-nansum` / `vt-nanmean` / `vt-nanvar` / `vt-nanstd` / `vt-nanmax` /
   `vt-nancumsum` 都建在 `vt-where` 上，收益直接传导（35/46/65/62/88/36 ms
   → 5/13/28/30/22/5 ms）。

> 这两条的共同点：**看起来已经特化了**（有 `simple-array` 声明、有 `the`、
> 有内联宏），但类型信息在某一处被推断放宽，热路径就退回泛型。
> 遇到「同一内核在逐元素路径快、在归约路径慢」时，先查累加器/中间变量的
> 类型，而不是查算法。

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
| `vt-einsum` | ✅ 已支持复数（原白名单只含 4 种实数类型，见下方注） |
| `vt-det/inv/solve/svd/eig/qr/lu/pinv/cholesky/cond` | ✅ 已支持复数（LU/QR/SVD/厄米特特征值等，见 §16.20） |

**matmul 为何不走 einsum**：`vt-einsum` 的 `supported-types` 一度是
`(:float64 :float32 :int64 :int32)`，结果缓冲区按这 4 种分配，复数输入会得到
`double-float` 缓冲并报 `"is not of type DOUBLE-FLOAT"`；即便白名单已放开，
matmul 仍保留**独立专用内核**（`%cx-matmul`，`vt-contiguous` 落地后三重循环），
因为它能吃 SIMD/并行调度，而 einsum 的对应通道是通用 strided 循环。
**两边语义（含累加次序）逐位一致**，见 §16.21.5 的验收项。

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
| 未支持函数 + 复数 | `"~a: 第一轮不支持复数输入（…）。见 CONVENTIONS §16.13"` |
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

#### 16.14.4 复数初等函数：**不要**把 CL 的复数超越函数当内核

CL 的 `log` / `asin` / `acos` / `asinh` / `acosh` 对复数**有定义**，语义也基本正确，
但它们走 SBCL 的通用复数算术路径，单价 200 ns~1.1 µs/元素 —— 比同形实数慢
10~50 倍。判据只有一条：**复数超越函数的内核必须用分量式重写，不得直接回调 CL**。

实测（1M complex128，SBCL 2.6.8；同形 float64 作对照）：

| 算子 | CL 回退 | 分量内核 | 相对 float64（修前 → 修后） |
|---|---:|---:|---:|
| `log` | 352 ms | **55 ms** | 43× → 6.7× |
| `log2` | 354 ms | **96 ms** | 44× → 12× |
| `log10` | 357 ms | **97 ms** | 40× → 12× |
| `log1p` | 383 ms | **65 ms** | — |
| `sign` | 104 ms | **26 ms** | 40× → 10× |
| `sinc` | 218 ms | **81 ms** | — |
| `expm1` | 116 ms | **58 ms** | — |

**分量式的代价点**（每条都是踩过的坑）：

1. **模必须走 `%cop-abs`，不能写 `(sqrt (+ (* x x) (* y y)))`**。后者在
   |x| > 1e154 时上溢成 Inf —— 与 `%cop-sqrt` 里同一处教训。`%cop-abs` 已带
   量级守卫（`< 1d150` 走快路，否则 `%cx-hypot`）。
2. **NaN 判定的优先级高于 Inf**。`log(Inf+NaNj)` 在 numpy 是 `Inf+NaNj`
   （模为 Inf 时相位无意义），若先判 `%cx-nan-p` 就退化成 `NaN+NaNj` ——
   实测 441 点网格里正是这两格不一致。
3. **相位必须用 `(atan y x)`（atan2），不能用 `(phase z)`**：`phase` 对
   `x = -0.0` 的符号零处理与 numpy 不同，`log(-0+0i)` 会丢掉 π。
4. **内核要 `declaim inline`，且声明必须早于 `defun`**（§16.14.2 第 2 条）。
   `%cop-log` 内联前后是 199 ms / 55 ms —— 差的那 144 ms 全是「每元素一次
   真实函数调用 + 内核在独立编译单元里按泛型 NUMBER 编」。

**验收方式**：`test/complex-elementary-test.lisp` 的 441 点网格（21 实部 ×
21 虚部，含 ±0 / ±1e-8 / ±1 / ±1d300 / ±Inf / NaN）× 5 个算子与 numpy 2.4.6
逐点对拍。用容差 1e-9 比数值、按**类别**比特殊值（NaN 只与 NaN 相等、Inf 比
符号），期望值由 `test/gen_complex_elementary_expected.py` 生成。

> **这六个函数**（`tan` / `tanh` / `asin` / `acos` / `asinh` / `acosh`）
> 在同一张网格上与 numpy 曾有 481/4410 处不一致（全部落在**分支切割的符号零**
> 与 **NaN 输入**上，其中 `vt-asin` 遇到含 NaN 虚部的输入会直接抛
> `(DOUBLE-FLOAT 0.0)` 类型错误）。它们已按 §16.14.5 内核化，**现在全部为 0**；
> **改动它们之前必须先跑通那张网格**，否则只是把错值换个位置。
> `atanh` 与它们同源但算法不同（见 §16.14.5 末尾），现在也是 0。

---
#### 16.14.5 三角 / 反三角族：算法必须照抄 numpy 的分支，**不能套恒等式**

`tan` / `tanh` / `asin` / `acos` / `asinh` / `acosh` 这六个函数的 CL 复数定义
语义上是对的，但 SBCL 的实现走通用复数算术：单价 200 ns~1.1 µs/元素，且在
**分支切割与 NaN 输入上与 numpy 不一致**。1M complex128 实测：

| 算子 | CL 回退 | 内核 | 修前不一致（441 点） | 修后 |
|---|---:|---:|---:|---:|
| `tan` | 203 ms | **84 ms** | 92 | **0** |
| `tanh` | 192 ms | **51 ms** | 92 | **0** |
| `asin` | 915 ms | **112 ms** | 70 | **0** |
| `acos` | 876 ms | **60 ms** | 72 | **0** |
| `asinh` | 949 ms | **92 ms** | 70 | **0** |
| `acosh` | 866 ms | **62 ms** | 67 | **0** |
| `atanh` | 75 ms | **59 ms** | 18 | **0** |

算法来源是 numpy `numpy/_core/src/npymath/npy_math_complex.c.src`：
`ctanh` 用 Kahan(1987)，`casinh`/`cacos` 共用 `_do_hard_work`
（Hull-Fairgrieve-Tang），`casin = i·conj(casinh(i·conj z))`，
`cacosh = ±i·cacos`（取实部 ≥ 0），`ctan = -i·ctanh(i·z)`。

**五条硬约束**（每条都对应一批实测错值）：

1. **不要用恒等式代替分支**。`asin = -i·log(iz + sqrt(1-z²))` 及其
   `sqrt(1+z)·sqrt(1-z)` 变体在 441 点上各有 **386** 处不一致（大 `|z|` 处
   主值支选取就错）；`tanh = (tanh x + i tan y)/(1 + i tanh x tan y)` 有
   **232** 处（`|x|` 与 `|y|` 都大时分子分母同时上溢）。Kahan 形式用
   `beta = 1+tan²y`、`rho = sqrt(1+sinh²x)` 消掉上溢，且 `|x| ≥ 22` 直接
   走渐近式。
2. **`(1- y)` 是「y 减 1」**。`_do_hard_work` 里同时出现 `1+y`、`1-y`、
   `y+1`、`y-1` 四种组合，把 `1-y` 写成 `(1- y)` 会让 `Am1` 差一个
   `hypot` 项 —— 实测 `asinh(4+4i)` 的实部从 2.426 变成 1.805。
   正确写法：`(- 1d0 y)`。
3. **`(- y y)` 不能用来构造 NaN**。内联循环是 `(speed 3)(safety 0)`，SBCL
   会把 `x-x` 代数化简成 `0` —— `tanh(-4+Infi)` 因此返回 `0+0i` 而不是
   `NaN+NaNj`。用 `(%cx-nan-of z)`。
4. **「含 NaN」分支的外层条件只能是 `(isnan x) || (isnan y)`**。写成
   「非双有限」会把 `x` 有限、`y=±Inf` 的输入也吞进去：`cacos` 那段里
   `return (x+x, -y)` 于是返回 `(2x, ∓Inf)`，441 点里 75 处错值。这类输入
   必须落到 `_clog_for_large_values` 分支。
5. **`%cx-do-hard-work` / `%cx-clog-large` 必须 `declaim inline`**。它们每个
   元素调一次，且 `%cx-do-hard-work` 有 **5 个返回值**、`%cx-clog-large` 有
   2 个；不内联时多值返回落栈 + 参数装箱，`vt-acos` 300 ms、`vt-asinh` 317 ms。
   两个都内联后 **60 ms / 92 ms**（3~5 倍）。这是本族性能的**唯一主开关** ——
   比换公式、改分支顺序的收益大一个数量级。
6. **`%cx-hypot` 对 `(Inf, Inf)` 会算出 `inf/inf = NaN`** —— 它是给
   `%cop-sqrt`/`%cop-abs` 用的（那两处在调它之前已判过 Inf）。任何直接用
   它的代码（如 `_clog_for_large_values`）**必须先判 Inf**，否则
   `log|z|` 变 NaN。

**验收方式**：`test/complex-elementary-test.lisp` 的同一张 441 点网格
（11 个算子 × 441 点 = 4851 次比较）与 numpy 2.4.6 逐点对拍，外加
「NaN 输入不抛条件」的回归断言 —— 整改前 `vt-asin`/`vt-acos`/`vt-asinh`/
`vt-acosh` 遇到虚部为 NaN 的输入会直接抛
`The value NaN is not of type (DOUBLE-FLOAT 0.0)`（SBCL `DECODE-FLOAT` 的
类型断言，`with-float-safe` 屏蔽不了）。

> **`atanh` 与上面六个不同源**：numpy 自己实现了 `npy_catanh`（分段 +
> `_real_part_reciprocal`），不是 libm。它的坑是**所有公式都用 `|x|、|y|`、
> 最后统一 `copysign` 回去** —— 直接带符号代入会在 `x<0` 处把 `log1p` 的
> 参数算成负数（`0.25·log1p(4x/den)` 在 `x = -1+1e-8i` 处退化成
> `log1p(-1) = -Inf`，而真值是 `-9.5569`）。另外大 `|x|` 时实部会退化成
> `Re(1/z)`（`|x| = 1e300` 时 `atanh ≈ 1/x = 1e-300`，用
> `0.5·(log|1+z| - log|1-z|)` 会因两个 1e300 相减得 0）。

#### 16.14.6 顺序统计量：取中位用 quickselect，取 top-k 用有界堆

`median` / `nanmedian` / `topk` 三个算子此前都是「先整份排序，再取需要的那几个」。
两个都属**算法选错**，不是常数因子问题：

- 取中位数只要 O(N)，排序是 O(N log N)；
- 取前 k 个只要 O(N log k)，排序是 O(N log N)。

**`median` / `nanmedian`（complex）**：早先复数分支直接 `(vt-sort flat)` 再取下标，
而同形 float64 走的早就是 `%SELECT-DOUBLE`（quickselect）。1M complex128 实测
**225 ms vs 28 ms（8 倍差价全部来自排序）**。新增 `%CX-SELECT-F64/F32`
（`src/complex.lisp`），与 `%SELECT-DOUBLE` 三条设计逐条对齐：

| 设计 | 理由 |
|---|---|
| **三分区**（<, =, >）而非 Lomuto 两分区 | 两分区在「大量重复值」上一趟只剥掉 pivot 一个元素。复数尤其危险：网格数据（实部虚部都取自少数几个值）重复率极高 |
| **三数取中 + 访问量上限 4N** | 三分区在坏排列上仍会退化（整体降序且 k 居中时能跑 451 趟）；超限就整段排序，把最坏情况钉在 O(N log N) |
| **比较器不判 NaN** | 调用方在收数那一趟已经数过 NaN（`%CX-FILL-FIBER` 的返回值），有 NaN 就走 NaN 语义、根本到不了 select。改用 `%C-LEX-COMPARE` 每次比较要判 4 个分量，实测贵 1.9 倍 |

配套还有两个「少一趟」：

1. **收数与 NaN 判定合成一趟**（`%CX-FILL-FIBER` 返回 NaN 个数），不再单独
   调 `%CX-HAS-NAN-P` 全扫。
2. **缓冲区必须独立分配**。`VT-RAVEL` / `VT-CONTIGUOUS` 对连续输入是零拷贝
   视图，就地重排会改到**调用方的数据** —— 这个坑比慢更严重。
3. `nanmedian` 的收数走 `%CX-COMPACT-FIBER`（丢弃 NaN 并紧收）。这里曾用
   `(array-element-type data)` 在运行期决定的通用 `AREF`，SBCL 只能走泛型
   路径，1M 元素多付约 **28 ms**；同一条语句类型特化后 216 → 30 ms 里有一半
   是它。

**`topk`**：原实现 `vt-argsort` 整份排序后切 k 个。改成**有界堆**
（`%TOPK-HEAP-*`，`src/extensions.lisp`），要点：

1. **必须与 argsort 路径逐位一致**（值 + 下标，含并列与 NaN）。所以比较器
   不重新定义语义，而是复刻 `vt-sort` 的全序：非 NaN 按数值升序、**NaN
   名次最高**（`%sort-fibers-*` 把 NaN 抽出来压末尾）、并列时按原下标升序。
   于是 largest 的次序 = `flip(narrow(argsort, N-k, N))`，smallest =
   `narrow(argsort, 0, k)`。回归测试用「同一份输入分别走堆与 argsort 比
   值和下标序列」，而不是与 numpy 比 —— 后者测不出并列序的漂移。
2. **堆根是「最差」的那个**，弹出顺序是「劣→优」，所以要**倒着填**输出
   数组（`slot = k-1-j`）。第一版正着填，k=1 全过、k≥3 全错。
3. **`flet` 换成 `labels`**：`SIFT` 要调同层的 `WORSE`/`BEATS-ROOT`，`flet`
   里兄弟函数互不可见（表现为 `undefined function CLVT::WORSE` 的
   STYLE-WARNING，而且编译仍"成功"）。
4. **阈值 k ≤ 轴长/8** 才走堆。实测 1M double：k=10 174→2.2 ms、k=1000
   183→2.9、k=100000（轴长 1/10）194→54、k=125000（1/8）65 ms；再往大
   优势收窄，所以留着 argsort 兜底。k=0 与不支持的 dtype 也走 argsort。

| 算子（1M c128 或 f64） | 修前 | 修后 |
|---|---:|---:|
| `median` complex128 | 225 ms | **30 ms** |
| `median` complex64 | 205 ms | **26 ms** |
| `median` c128 `:axis 0`（1000×1000） | 152 ms | **37 ms** |
| `nanmedian` complex128 | 216 ms | **30 ms** |
| `topk` f64 k=10 | 174 ms | **2.2 ms** |
| `topk` f64 k=1000 | 183 ms | **2.9 ms** |
| `topk` complex128 k=10 | 297 ms | **3.0 ms** |

**顺带修掉的一处静默错值**：全 NaN / 空 fiber 的复数 `nanmedian` / `nanmax` /
`nanmin` 此前返回 **NaN+0j**，numpy 是 **NaN+NaNj**（那个虚部 0 是「先取
NaN 再补 0」的副产品，不是语义）。现在统一 NaN+NaNj。

> **两处与 numpy 的已知差异**（都记在 §16.15）：
> 1. `median` 遇到**单分量 NaN**（如 `nan+1j`）时，numpy 直接返回那个 NaN
>    元素本身（它的 partition 在 NaN 上比较恒假，NaN 会停在 kth 位），本库
>    统一返回 NaN+NaNj。两者都是「结果含 NaN」。
> 2. **全 NaN fiber 沿轴归约**时 numpy 自己不一致：1-D / 展平给 `nan+nanj`，
>    2-D 沿轴给 `nan+0j`。本库统一 `nan+nanj`（只有两个分量都是 NaN 才是
>    「两边都不是数」）。

#### 16.14.7 nan 族 / `var` / `nan_to_num`：把「多趟张量流水线」压成单趟或两趟

nan 族的写法一直是「掩码 + 单位元替换 + 复用既有归约内核」：

```
nanmean = isnan + where + sum + logical-not + astype + sum + div + where   （≈7 趟）
nanvar  = 上面两倍 + map + mul + sum + map + map                           （≥10 趟）
```

好处是「语义只有一处」（掩码 + 既有内核都对，组合就对），代价是每次都要
分配 2~3 个 16 MB 临时张量并全量遍历。1M complex128 实测 `nanmean` 20 /
`nanvar` 50 / `nanmax` 23 ms，而**同形 float64 早就有单趟快路径**
（`%NAN-SUM-COUNT` / `%NAN-SUMSQ`，1.4 ms）—— 这个差价里没有一处是
「复数本来就该慢」。同样的病还有三处：`nan_to_num` 走 `vt-map` + 闭包、
`var`（复数）没有两趟版、`nansum`（实数）只有通用 `%OP-STEP` 循环。

**做法**（只在 `axis=nil` 的全归约上启用，沿轴版本保持流水线）：

| 新内核 | 用途 | 位置 |
|---|---|---|
| `%CX-NAN-SUM-COUNT` | 复数和 + 计数（单趟） | `complex-ops.lisp` |
| `%CX-NAN-EXTREMUM-COUNT` | 字典序极值 + 计数（单趟） | 同上 |
| `%CX-NAN-SUMSQ` | Σ\|z−mean\|²（跳过 NaN，单趟） | 同上 |
| `%CX-SUMSQ-ALL` | Σ\|z−mean\|²（**不跳过**，var 用） | `complex.lisp` |
| `%SUMSQ-ALL` | 实数版（var 用） | `reduce-stats.lisp` |
| `%NAN-TO-NUM-BUFFER` / `%CX-NAN-TO-NUM-BUFFER` | 逐分量替换的直线循环 | `extensions3.lisp` / `complex-ops.lisp` |

| 算子（1M） | 修前 | 修后 |
|---|---:|---:|
| `nanmean` c128 | 19.9 ms | **1.3 ms** |
| `nanvar` / `nanstd` c128 | 49.8 / 49.3 ms | **3.0 / 3.1 ms** |
| `nanmax` / `nanmin` c128 | 22.4 / 22.5 ms | **1.8 / 1.9 ms** |
| `nansum` c128 | 7.0 ms | **1.3 ms** |
| `nan-to-num` c128 | 78.9 ms | **5.5 ms**（对照 `vt-copy` 3.4 ms） |
| `nan-to-num` f64 | 21.9 ms | **1.7 ms** |
| `var` c128 | 27.2 ms | **1.2 ms** |
| `var` f64 | 6.5 ms | **1.1 ms** |
| `nansum` f64 | 8.9 ms | **0.8 ms** |

**五条坑**（每条都对应一次实测错值/崩溃）：

1. **极值扫描不能拿 `a[0]` 当种子**。NaN 参与比较时 `%CLT` 恒假，若 `a[0]` 是
   NaN，`BEST` 会永远停在 NaN 上、后面所有有限值都比不过它 —— 用 `FIRST`
   标志在「第一个非 NaN」处初始化。
2. **`n=0` 时 `(aref d off)` 越界**。空数组的 nanmax 走的是「计数 0 → NaN+NaNj」
   分支，但种子取得比它早，必须先判 `(plusp n)`。
3. **`%NAN-P` 必须在 `with-float-safe` 里调**。`(/= nan nan)` 在 SBCL 上是
   UCOMISD + invalid 陷阱，不是 C 的 quiet compare。
4. **快路径必须显式要求「浮点 dtype」**。`%E3-FLOAT-PREFER-DTYPE` 对整数输入
   原样返回 `:int64`，于是 `(eq (vt-dtype v) odt)` 对整数也成立，而
   `%NAN-TO-NUM-BUFFER` 只认 f32/f64 → 返回 **NIL 张量**（不是报错）。
5. **`nansum` 的门面是宏生成的**，不能覆盖公开符号（§16.32 禁止靠加载顺序
   覆盖）。给 `DEF-VT-REDUCE-FACADE` 加了一个可选的 `:fast` 钩子
   （5 参函数，返回 NIL 表示不接管），这样「全归约快路径」与「一个符号只有
   一个 DEFUN」两条约束同时成立。

**顺带修掉一处公开 API 崩溃**：`%CX-ABS-RUN` / `%CX-ANGLE-RUN` 的特化循环
（`%DEF-CX-TO-REAL-RUN` 生成）**没有屏蔽浮点陷阱**。内核里 `%CX-INF-P` 是
`(> v maxf)`、`%COP-ARG` 要看符号，UCOMISD 遇 NaN 会置 invalid 并触发陷阱 ——
于是 `(vt-abs z)` 与 `(vt-angle z)` 在**含 NaN 的输入上直接抛
FLOATING-POINT-INVALID-OPERATION**。这是公开 API 的崩溃（与 §16.14.5 里
`vt-asin` 那处同类），加一层 `with-float-safe` 即修好；切换 FP 控制字只在
进出各一次，不在逐元素路径上。

> **仍未融合的**：`nanprod`（单位元是 1，不是热点）、以及**所有沿轴版本**
> （`axis` 非 nil）仍是流水线。后者若要快，需要把「掩码计数 + 归约」做成
> 逐 fiber 的融合内核，收益与复杂度都远高于全归约这一批。

### 16.15 与 numpy 的已知差异清单（全部为**有意**保留）

| 编号 | 差异 | 影响面 | 处置 |
|---|---|---|---|
| D-C1 | `complex → real` 转换不产生 `ComplexWarning` | 仅提示信息 | 文档声明；行为与 numpy（取实部）一致 |
| D-C2 | `np.array([1+2j], dtype=float)` numpy 抛 TypeError，clvt 取实部 | 构造 | 文档声明（沿用 `vt-cast` 的「安全转换」立场） |
| D-C3 | `vt-real/vt-imag` 返回**副本**（numpy 返回视图） | 写回语义 | 文档声明；用 `vt-complex` 重组 |
| D-C4 | 打印保留 `1.0+2.0j`（numpy 为 `1.+2.j`） | 显示 | 沿用本库浮点风格 |
| D-C5 | 复数逐元素算子无 SIMD/内联特化 | 性能 | 第二轮 |
| D-C6 | ~~`einsum` / LU 家族~~ 已支持复数；**百分位/`sort` 的复数语义仅字典序** | 功能面 | 见 §16.10 与 §16.18 |
| D-S1 | `median` 遇**单分量 NaN**（如 `nan+1j`）返回 `NaN+NaNj`；numpy 直接把那个 NaN 元素返回（它的 partition 在 NaN 上比较恒假，NaN 会停在 kth 位） | 值 | 文档声明；两者都是「结果含 NaN」。测试里钉死本库取值（`selection-test` 的 D 节） |
| D-S2 | `nanmedian`/`nanmax`/`nanmin` 对**沿轴全 NaN 的 fiber** 返回 `NaN+NaNj`；numpy 这里返回 `nan+0j`，而 1-D/展平却返回 `nan+nanj` —— **numpy 自己不一致** | 值 | 本库统一 `nan+nanj`（只有两个分量都是 NaN 才是「两边都不是数」） |

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
7. **性能护栏只能用倍率，禁止绝对毫秒**。绝对毫秒阈值不可移植：
   `complex-elementary-test` 早期写过「`acos` c128 < 200ms」，而开发机实测
   98ms —— 只有 2 倍余量，慢一倍的机器（笔记本、共享 VM、内存带宽不同）直接
   红，报出来却是「性能回归」。可移植的两种写法：
   - **同算子 c128 / f64**（回答「复数是否被实数路径落下」）；
   - **目标算子 / 锚点算子**，锚点取成本构成相近、又不在保护范围内的那个
     （如复数超越函数之间用 `vt-exp` 当锚点；比值跨机器只漂 ±30%）。
   两边都要留 ≥3 倍余量，同时远低于「退回慢路径」时的比值，才既拦得住真
   回归、又不会被机器速度误伤。测时报出实测值便于事后判读。

运行方式：
```bash
bash test/run-tests.sh --suite complex-test        # 走统一入口（136 断言）
python3 test/gen_complex_expected.py               # 重新生成期望值（numpy 升级后必做）
```

---

### 16.17 未完成项

→ **见 §16.37（唯一清单）**。这里原先有一份路线图，其中 P0/P1
（einsum 复数、复数归约多轴、LU/SVD/特征值、`.npy` 互操作）已全部完成，
剩余项已并进 §16.37，本节不再单独维护。

---

### 16.18 数据搬运层：转换 / 拼接 / 剪切 / 集合 / 缺失值 / 缩并

本节是**结构层**（逐元素算术与归约见 §16.5–§16.14）。
落地文件 `src/complex-ops.lisp`（在 `src/complex.lisp` 之后加载），
同样遵守「实数路径零改动 + 只对含复数的调用分派」。

#### 16.18.1 曾经的「静默错答案」

比「不支持」更危险的是「看起来正常但错」。实测发现过 5 处：

| 函数 | 修复前行为 | numpy 语义 | 根因 |
|---|---|---|---|
| `vt-nanmean` | 返回**实数**（实部均值） | 复数 | 内部 `vt-astype → :float64` 按 §16.5 取实部 |
| `vt-nanvar` / `vt-nanstd` | 数值错误（按实部算） | 实数 = mean(\|x−mean\|²) | 同上 |
| `vt-nancumsum` / `vt-nancumprod` | **虚部被清零** | 保复数（NaN→0/1 后累积） | `%e3-cumulative-skip-nan` 固定 `vt-astype :float64` |
| `vt-vdot` | 不取共轭（= `vt-dot`） | **第一个操作数取共轭** | 第一轮漏做 |
| `vt-cumsum/cumprod` 的 `axis=None` | 形状保持原样（2-D 进 2-D 出） | **展平为 1-D** | `%cx-scan` 沿用了 `(vt-shape in)` |

另外 `vt-repeat` / `vt-unique` / `vt-searchsorted` / 集合运算 / `vt-nan[sum|prod|max|min|argmax|argmin|median]` /
`vt-convolve` / `vt-inner` / `vt-tensordot` / `vt-nan-to-num` / `vt-view(:dtype)` 这些
此前是**确定性报错**（类型声明里写死 real 或只列 8 种实数元素类型），现已补齐。

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
| `nanmax` / `nanmin` | `NaN+NaNj`（**不报错**） | 复数 |
| `nanargmax` / `nanargmin` | **报错** `All-NaN slice encountered` | int64 |
| `nanmean` | `NaN+NaNj` | 复数 |
| `nanvar` / `nanstd` | `NaN` | **实数** |
| `nanmedian` | `NaN+NaNj` | 复数 |

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
| `vt-einsum` | 支持复数（白名单按 dtype 分配）；缩并通道与 `vt-matmul` 逐位一致 |

> `correlate` 与 `convolve` 的共轭差异在实数上不可观测，**只有复数能测出来**。
> 第一版实现漏了这一点，差分测试（`correlate-*` / `convolve-*` 用例）当场抓到。

#### 16.18.3 已知差异（D-C7 起）

| 编号 | 差异 | 原因与影响 |
|---|---|---|
| **D-C7** | `vt-view` 的 dtype 重解释返回**独立副本**（numpy 是共享内存的视图） | Common Lisp 的 `make-array :displaced-to` 要求元素类型一致，无法跨类型别名同一块内存。数值完全等价，但「改视图即改原数组」不成立——已在 docstring 与本节显式声明 |
| **D-C8** | `vt-take-along-axis` / `vt-put-along-axis` 要求 indices 与数据**同秩** | 库既有约束（numpy 会广播 indices），与复数无关，没有一起改 |

#### 16.18.4 声明策略（延续 §16.14.1）

新增的内核只有两处带 `optimize` 声明，且都不是 `safety 0`：

| 位置 | 声明 | 理由 |
|---|---|---|
| `%cx-correlate-full` 的卷积循环 | 继承 `%cx-dispatch` 的 `(speed 3) (safety 1)` | 索引由 `k/j` 的上下界决定，可证不越界，但保留检查 |
| `%cx-unique-sorted` / `%cx-searchsorted-run` / `%cx-nan-arg-run` | 同上 | 同上 |

其余（`nan` 族的向量化实现、集合过滤、`tensordot`）全部是**组合既有算子**，
没有手写热循环，因此不需要任何 `optimize` 声明 —— 这也印证了 §16.14.1 的分级原则：
**能用既有内核组合出来的，就不要新写循环**。

#### 16.18.5 测试

套件 `complex-ops-test`。逐条断言见该文件头注释；套件说明见 `bash test/run-tests.sh --list`。

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

### 16.19 线性代数与算子补齐

本节覆盖复数线代（LU / Cholesky / QR / SVD / 厄米特特征值 …）与一批原本在
复数输入下**静默只算实部**的算子。

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

> **约化 QR 的陷阱**（实测踩到过）：Q 只有 `k = min(m,n)` 列，而反射器
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

> 不清零的后果（实测）：对 `[[2+1j, 1], [1-1j, 3+2j]]` 会返回
> `(3.925, 1.075)`，而 numpy 返回 `(4, 1)`。原因是 Jacobi 用
> `(realpart diag)` 当特征值候选、却让复对角元继续参与旋转 —
> 迭代仍「收敛」，只是收敛到错的值。LAPACK 的 `zheevd` 同样直接按实部使用。

#### 16.19.4 曾经的「静默丢虚部」（同类共 8 处）

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

#### 16.19.5 新增/放开的算子

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
| **D-C10** | `vt-eig` 是 **eigh 语义**（厄米特），不是 numpy 的通用 `eig` | 与实数版一致（实数版也只支持对称）。通用非厄米特征值需要复 QR 迭代（Hessenberg + 双位移），不在本次改动范围内；非厄米输入与 numpy `eigh` 一样是 GIGO |
| **D-C11** | `vt-svd` 的 U/V 与 numpy **逐元素不同**（重构等价） | 奇异向量带相位自由度：`u_j → e^{iθ}u_j, v_j → e^{iθ}v_j` 仍是同一分解。断言应使用重构/正交恒等式，不要逐元素比对 |
| **D-C12** | `vt-qr` 的 R 与 numpy **逐元素不同**（重构等价） | Householder 的 α 取 `-phase(x₀)‖x‖`（numpy 用 LAPACK 的符号约定），R 的相位随之不同 |
| **D-C13** | `vt-lstsq` 的 residual 是 **2-范数**（numpy 是平方和） | 库内既有契约（`vt-norm`），保持一致；`sqrt(numpy_res)` 即得 |

#### 16.19.7 声明策略（延续 §16.14.1）

新增内核的声明**只有一档**：

| 位置 | 声明 | 理由 |
|---|---|---|
| `%clu-factor!` `%clu-subst!` `%cchol-factor!` `%cqr-*` `%csvd-jacobi!` `%ceig-jacobi!` | `(speed 3) (safety 1)` | 索引由 `ld/m/n` 与循环上界推出，可证不越界；保留检查的成本在 O(n³) 算法里可忽略（<2%），而形状算错会立刻报错而不是静默踩内存 |
| 所有对外入口（`vt-svd` 等） | 不写 `optimize` | 契约检查（方阵、`:out`、正定性）与错误信息都在这一层 |

**没有**任何新的 `safety 0` —— 复数线代的索引组合（`ld` ≠ 行数）比实数更容易写错，
实际因 `ld` 写错踩过 3 次（`vt-lu` 非方阵、QR 的 Q、SVD 的 U），
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

修复策略体现了一条原则：**能用既有算子组合的，就不要新写内核**。

- `vt-take-along-axis`：只加两个 `(walk (complex double-float))` /
  `(walk (complex single-float))` 分支。注意内核里的 `SEL` 是**下标**（fixnum），
  元素值才是复数 —— 不受 `(type fixnum sel)` 声明影响。
- `vt-outer`：删掉多余的 `:dtype` 实参，改为先广播乘再按需 `vt-astype`，
  并补 `:out` 硬契约（`vt-check-out` + `vt-copy-into`）。
- `vt-any` / `vt-all`：**不重写归约内核** —— 先把复数转成 `:bool` 的真值掩码
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

套件 `complex-linalg-test`。逐条断言见该文件头注释；套件说明见 `bash test/run-tests.sh --list`。

### 16.20 累计变更文件清单 → 见 `HISTORY.md`

> 「每轮改了哪些文件」属于**编年史**，已迁到 `HISTORY.md`（§H.1）。
> 本文件只保留现行契约，不再逐步列举改动文件。

### 16.21 复数 einsum 的 GEMM 快路径

#### 16.21.1 为什么必须有这条快路径

einsum 的通用迭代状态机与元素类型无关（所以复数从第三轮起结果就**正确**），
但它每个输出元素都要走「补码递推 + 逐张量泛型 `aref`」，比元素类型特化的
三重循环慢一个量级。三条快路径的判据原本全是实数 dtype 谓词
（`all-f64-p` / `all-f32-p` / `all-i64-p` / `all-i32-p`），复数直接落到最底下。

> **约束**：新增一种 dtype 时，必须同时放开这些判据 —— 否则功能正确、
> 性能静默掉回通用路径。实测倍数见 `HISTORY.md` §H.3.7。

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

### 16.22 批量 matmul：扁平下标 + 裸 `AREF` 的陷阱

`%CX-BATCHED-MATMUL` 曾经每个 batch 都读到基数组开头 —— 因为
`(vt-slice (vt-ravel a) …)` 得到的 1D 片段**已连续**，`vt-contiguous`
零拷贝返回它、`offset` 保留非零，而内核按扁平下标从 0 起算。

**这不是复数专属的坑**：`manip.lisp` 的 `%flat-offset0` 记的是同一类根因
（§4.6 机制 B）。凡是在内核里做「展平 + 裸 `aref`」的地方，都必须显式带上
`vt-offset`；批量维的偏移要按 **strides 递推**，不能用 `reshape` 切片
（后者隐含「两侧 batch 形状相同」，批间广播会直接失败）。

完整过程（两个缺陷的现象、根因与修法）见 `HISTORY.md` §H.3.6。

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
完整签名现在是 `vt-trace (matrix &key offset axis1 axis2 dtype out)`。

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

套件 `setops-axis-trace-test`。逐条断言见该文件头注释；套件说明见
`bash test/run-tests.sh --list`。

> 注：本处原有「第五轮测试」的逐条断言枚举（编号 16.25），与套件头注释、`run-tests.sh`
> 的 `SUITES` 三处重复，已删除（编号不复用，以保持既有 `§16.2x` 引用不断）。

---

### 16.26 einsum 的实数 GEMM 快通道接入 SIMD

**两条路径必须落到同一内核**，否则同一台机器上 einsum 比 `vt-matmul` 慢数倍：

| | 走的通道 | 内核 |
|---|---|---|
| `vt-matmul` 2D@2D | `*SIMD-MATMUL-2D-FN*` | `simd-matmul.lisp`：AVX 4 宽 + FMA + 64×256×256 分块 |
| `vt-einsum` BMM | `%MATMUL-DF-FAST-KERNEL` | `linalg.lisp`：4 路手写展开，**无 SIMD、无分块** |

判据：关掉 `*SIMD-MATMUL-ENABLED*` 后两者应该持平（0.93~0.98×）——
若不等，差异来自通道而非 einsum 的判定逻辑。

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

#### 16.26.4 实测

四条布局改完后 `vt-einsum` 与 `vt-matmul` 的比值全部落到 **0.84–1.01×**
（改前 6.3–19.6×），小矩阵（低于 `*SIMD-MATMUL-THRESHOLD*`）两者都走标量、
无退化。数字与测量条件见 `HISTORY.md` §H.3.8。

#### 16.26.5 回归网

`test/einsum-gemm-layout-test.lisp`（36 条断言，接入 `test/run-tests.sh`）。
**主断言写成「与 `vt-matmul` 逐位相等」**而不是硬编码期望值 —— 后者要跟着
机器/编译器变，前者同时锁住三件事：语义没变、四条布局路径一致、两条入口
最终落到同一内核。只有两处用容差（1e-10）并注明原因：

- 批量 `"bij,bjk->bik"`（每 batch 的 `m·k·n` 低于 SIMD 阈值）：einsum 逐
  batch 走标量、`vt-matmul` 的批量版按总工作量走 SIMD，累加顺序不同；
- SIMD 开/关的 A/B：FMA 与 mul+add 的舍入不同（这条对 `vt-matmul` 本来
  就成立，不是这里引入的）。

---

### 16.27 复数 matmul 内核对齐实数线（标量形态）

#### 16.27.1 内核形态必须与实数线对齐

复数 2D matmul 的标量内核**必须**采用与实数 `%MATMUL-DF-ROWS` 相同的三件事：

| 项 | 说明 |
|---|---|
| `(safety 0)` | 索引由 i/p 维推出、`c-ptr`/`b-ptr` 与 j 同步 +1，可证不越界 |
| 指针递推 | 行起点只算一次，内层只 `incf`，不再每元素算 `(+ c-row j)` / `p*n` |
| j 维 4 路展开 | 展开宽度与实数内核一致 |

三条收益互不重叠。**差距的根因从来不是复数表示本身**，而是「复数内核按最朴素
的教科书三重循环写」—— `safety 1` 的 `aref` 带边界检查且阻止向量化、每元素两次
加法一次乘法、无展开挡住乱序执行。消融数据见 `HISTORY.md` §H.3.9。

#### 16.27.2 被实测否掉的方案

SoA（拆 re/im）、SoA + 单累加器、把 `p=0` 折进内核 —— 全部都实测更慢或持平，
数据与结论见 §16.36.2。**结论：保持 boxed complex 缓冲区不变，只抄实数线的
循环形态。**

#### 16.27.3 实现

```lisp
;; src/complex.lisp（复数 GEMM 行内核）
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
   对齐本来就是那次改动的目标，两边必须同级，否则复数永远追不上。

#### 16.27.4 实测

复数每元素 8 flop、实数 2 flop（**flop 比 4×**），而实测墙钟比是 **2.16×** ——
说明复数的单位 flop 效率反而比实数标量高（一次 `aref` 搬 16 字节，内层访存
更省指令）。端到端（含 `make-vt` + 清零，默认并行）相对实数 SIMD 是 **2.0–4.2×**。

绝对毫秒随机器负载波动很大，**只看比值**；完整表格见 `HISTORY.md` §H.3.9。

#### 16.27.5 还剩什么没做

剩下的差距**全部**来自「实数有 SIMD、复数没有」，分两块：

1. **宽度**：`simd-matmul.lisp` 一次处理 4 个 `double-float`（AVX）且用
   FMA，单位时间吞吐是标量的 8–10 倍；
2. **指令数**：复数每元素 4 mul + 4 add（`(ar·br − ai·bi, ar·bi + ai·br)`），
   实数 1 mul + 1 add。即使上了 SIMD，复数也要拆成两组实部/虚部向量
   做 4 次 FMA 才等价于一次复数乘。

这两块都属于 SIMD 内核的领域。**尚未实现**，且路线已被实测收窄
（§16.36.2）：SoA + 4 条实 GEMM 会改变累加顺序且实测更慢，剩下的可行方案
只有「在交错布局上按 j 向量化、用 shuffle 组装 re/im」这一条。

#### 16.27.6 测试

套件 `complex-matmul-perf-test`。逐条断言见该文件头注释；套件说明见 `bash test/run-tests.sh --list`。

### 16.28 复数缺口扫描与补齐（入口 / 幂 / 归一化）

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
第 ③ 层扫出 24 个，其中 19 个是白名单漏项（比较运算、`argsort`、`nanvar` 等），
真问题 5 个。

#### 16.28.2 曾经的 6 个缺口

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

套件 `complex-gap-test`。逐条断言见该文件头注释；套件说明见
`bash test/run-tests.sh --list`。其中两条是**契约级**的，改代码时别破坏：

| 契约 | 说明 |
|---|---|
| `float_power` 的边界 | `nan^0 = 1`、`0^0 = 1`、`0^-1 = NaN`（C99 的 `pow` 约定，与实数路径一致） |
| `cbrt` 的复数输入必须**报错** | 消息含「不支持复数」；不许返回一个 numpy 没有的「复数立方根」语义 |

另外测试进程**不屏蔽浮点陷阱**，判 NaN 只能用 `sb-ext:float-nan-p`（§16.34.3）。

#### 16.28.6 扫描结论：仍然「不支持复数」的清单（都是**有意**的）

`%cx-def-unsupported` 现在包含：`floor/ceil/trunc/rint`、`div/mod/rem`、
位运算族、`atan2`、`even-p/odd-p`、`percentile/quantile`、`hypot`、
**`cbrt`（新增）**、激活函数族（`sigmoid/relu/leaky-relu/swish/
softplus/gelu/mish/hard-tanh/hard-sigmoid`）。依据都是「numpy 同样拒绝」
或「数学上无定义（复数无全序）」，不是实现偷懒。

另外 `vt-eig`/`vt-eigvals` 返回**实数**特征值**不是 bug**：本库的 `vt-eig`
是 `eigh` 语义（只吃 Hermitian/对称，用 UPLO 指定三角 + 镜像，见其 docstring），
厄米特矩阵的特征值本来就是实数。要一般复矩阵的复特征值，那需要另一套
（Hessenberg + QR 迭代）实现，属未来工作。

### 16.29 沿轴扫描内核必须遍历 `ax-stride` 层

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

#### 16.29.4 怎么找到它的（两层扫描，可复用）

第八轮的扫描是「dtype 保真」（复数入参 → 实数出），查不出「dtype 对、数值错」。
在上面补两层：

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

套件 `complex-scan-axis-test`。逐条断言见该文件头注释；套件说明见 `bash test/run-tests.sh --list`。

### 16.30 NaN/Inf 语义、median 契约与 `:out` 非连续

#### 16.30.1 `vt-median` 复数版的四条契约

复数 `vt-median` 曾有三处与**自家实数版**分叉（都不报错、都是静默错值），
现全部对齐。这四条是契约：

| # | 契约 | numpy 行为 |
|---|---|---|
| 1 | `axis=nil` = **先 ravel 再取全局中位**，返回 0 维标量；`:keepdims t` 给全 1 形状 | 同（不是「末轴」） |
| 2 | 空 fiber（轴长 0）结果必须是 **NaN**（`NaN+NaN i`），不是 `0+0i` | 同 |
| 3 | 含 NaN 时**传播 NaN**，不许返回某个真实元素 | 同 |
| 4 | 整个函数体包 `WITH-FLOAT-SAFE`（与实数版一致） | — |

第 4 条尤其容易漏：SBCL 默认开着 `OVERFLOW/INVALID/DIVIDE-BY-ZERO` 陷阱，
复数版的 NaN 判定 `(= x x)`、`inf + (-inf)` 都会直接抛
`FLOATING-POINT-INVALID-OPERATION`。

三处缺陷的根因（`axis=nil` 被写成末轴、`lo = (floor (1- 0) 2) = -1` 的空 fiber、
字典序排序后没做 NaN 检查）见 `HISTORY.md` §H.3.10。

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
这正是回归测试锁的点。

> **遗留（非复数问题）**：实数线的 `vt-cumsum`/`vt-cumprod` 走通用
> `VT-CUMULATIVE`（`acc` 从 0/1 起算），因此实数 `cumsum([-0.0])` 仍给 `+0.0`。
> 修它要动快路径 + 通用路径 + 轴分支共四处循环，风险与那次改动的主题（复数）无关，
> 故列为下一轮候选，没有一起做。
>
> ✅ **已修**（§16.30.3）：三处循环全部改为「首元素直拷」，
> 并顺带发现实数 `vt-cumulative` **没有包 `with-float-safe`**——
> `(vt-cumprod #(inf 0))` 在默认陷阱下直接抛 INVALID-OPERATION，
> 而 numpy 返回 `[inf, nan]`。

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

套件 `complex-nan-inf-test`。逐条断言见该文件头注释；套件说明见 `bash test/run-tests.sh --list`。

#### 16.30.6 可能返回 NaN 的内部浮点函数：返回类型必须显式 `ftype`

**契约**：任何**允许返回 NaN** 的内部函数（`%cx-nan-*` 一族、融合的
`nansum`/`nanvar` 扫描等），若返回单一浮点，就必须在 `defun` 前显式写

```lisp
(declaim (ftype (function (t t t) (values float)) %cx-nan-sumsq))
```

理由不是风格，是**正确性**：SBCL 从 2.6.9 起会把 `(* x x)`、`(+ a b)` 的
返回类型推断成「非负浮点」`(FLOAT (0.0))`，而 **NaN 不属于该类型**（NaN 与 0
的任何序比较都为假，落不进 `(float lo hi)`）。编译器随后把这个推断记成函数的
类型，调用点就按它插一条返回值检查：

```
The value #<DOUBLE-FLOAT quiet NaN> is not of type
  (OR (MEMBER 0.0d0 0.0) (FLOAT (0.0))) from the function type declaration.
```

`nanvar([inf, -inf, 1])` 就是踩这个：`mean` 为 NaN → `Σ|z-mean|²` 为 NaN →
当场 TYPE-ERROR。这是**推断失真的编译器缺陷**（推导出的类型对那些「NaN 参与
的乘法」不成立），不该靠改语义绕开 —— 该返回 NaN 的就得返回 NaN。

| 做法 | 结果 |
|---|---|
| `(the double-float ...)` / `(sb-ext:truly-the ...)` 包住乘法或赋值 | **无效**，声明与推断取交集，NaN 仍被排除 |
| `(declare (optimize (safety 0)))` | 无效，与安全检查无关，被拦的是返回值约定 |
| 局部函数 + `(declare (notinline ...))` 藏起加法 | 有效，但每次累加多一次函数调用 |
| **`declaim ftype` 把返回类型钉回 `float`** | **有效，且零运行时开销** —— fdefn 上记的是声明类型，检查点消失，循环内的数值推导不受影响 |

判据：`(sb-introspect:function-type #'fn)` 的 `values` 分量里不得出现
`(FLOAT (0.0))` / `(MEMBER 0.0d0 0.0)` 这类下界类型。`test/nan-fused-test.lisp`
的 ±Inf 用例是这条的可执行版本（它就靠这个把回归抓下来）。

### 16.31 弱标量提升（numpy NEP 50）

#### 16.31.1 方法：换个角度扫 —— 只比 dtype

前十轮都在比**值**与「复数会不会掉回实数实现」。这里换成**只比 dtype**：
119 条表达式全喂 `complex64`（外加若干实数 dtype 对照），逐项与 numpy 的
结果 dtype 比对（值的正确性已由 §16.16 覆盖，dtype 这一维原本是空白）。

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
| 种类秩 `源 ≤ 目标`（int< float< complex） | 只允许无损方向。反过来 `float → int` 会截断：`(vt-< f64数组 1.5)` 的结果是**布尔掩码**，把 `1.5` 转成整型会变成 1，`nan<1` 直接判错 |
| 整数值装得下（`%SCALAR-FITS-P`） | `(vt-> 大数组 999000)` 的标量若转进窄整型会变成 -8 —— 实测让 `argwhere(gt 999000)` 从 999 变成 999911 |
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

#### 16.31.5 有意保留的三处差异（都不是 bug）

| 项 | clvt | numpy | 理由 |
|---|---|---|---|
| `det/inv/solve/pinv/cholesky/qr/matrix-power/svd/eigvalsh` 的 c64 输入 | complex128 / float64 | complex64 / float32 | 复数线代**恒在 double 下计算**（`vt-det` 文档已写明「返回 0 维 complex128 张量」），精度更高；改回单精度是**降级**，不做 |
| `ptp` / `cond` | Lisp 标量（complex / double） | numpy 标量（c64 / f32） | 这两个函数返回**原生数**而非张量（§5 的既定风格），原生数没有 dtype |
| 弱整数越界（`i8* + 300`） | 回绕 | `OverflowError` | 回绕是本库既有语义，改它会牵动所有整型路径 |

#### 16.31.6 测试

套件 `dtype-weak-scalar-test`。逐条断言见该文件头注释；套件说明见 `bash test/run-tests.sh --list`。

### 16.32 单一 DEFUN 分派（禁止覆盖式复数层）

> 为什么要拆、三种覆盖机制长什么样、迁移怎么分阶段做的，见 `HISTORY.md` §H.3.5。
> 本节写**现在必须遵守什么**。

#### 16.32.1 形态：`when` + `return-from`，不是用 `if` 包住 body

```lisp
(defun vt-sqrt (vt &key out dtype)
  "逐元素平方根。"
  (when (%cx-any-p vt)                    ; ← 唯一新增
    (return-from vt-sqrt …复数臂…))
  (%float-map #'sin-impl vt out (%infer-float-dtype vt dtype)))   ; 原 body 逐字不动
```

三个理由（决定了它是 `when`+`return-from` 而不是 `if` 包 body）：

1. `defun` 自带同名 block，`return-from` 天然可用 —— **原 body 的缩进与括号完全不动**，
   改造可脚本化，diff 只增不减；
2. 单一 `defun` / 单一 lambda-list / 单一 docstring，结构上**不存在**「两处
   lambda-list 漂移」这一类 bug；
3. 实数路径的代价只是一次内联的 dtype 判断，比原来「funcall 匿名包装 + 同样一次
   判断」**净减少**。

#### 16.32.2 三个公共件必须在 `dispatch.lisp`

`src/dispatch.lisp` 在 `clvt.asd` 里紧随 `core`、早于所有归属文件：

| 名字 | 作用 |
|---|---|
| `%cx-arg-complex-p` / `%cx-any-p` | 分派判据。VT 走内联两步判断，**list/vector 进 `%cx-deep-p` 递归** |
| `%cx-result-complex-dtype` | 复数数学函数的结果 dtype 规则（24 处归属文件要用，必须前置） |
| `%cx-reduce-dispatch` | 归约族的算子路由（14 个符号共用一个入口） |

**判据必须认 list**：`(vt-cbrt (list #C(8 0)))` 是既有测试用例，浅判据（只看 VT）
会把它送进实数路径。所以「分派判据」与「VT 快判」是两个不同的东西，不能合并。

#### 16.32.3 把已有复数臂搬进归属 DEFUN 的三种形态

| 形态 | 形状 | 变换 |
|---|---|---|
| A | `(if TEST <orig枝> <复数臂>)` 是顶层唯一 form | 抽臂 → `%CX-OP-X`，归属 defun 里插守卫 |
| B1 | `(let BIND (if TEST <orig枝> <复数臂>))` | 连 `BIND` 一起搬进 `%CX-OP-X`（绑定只依赖形参） |
| B2 | `(let BIND (unless … (return-from X (funcall orig…))) <复数代码>)` | 删 `unless` 块 → 改名 `%CX-OP-X` |

归约族（`all`/`any` 等 14 个）用「实现 + 门面」两层：复数语义是「先把复数转成
`:bool` 真值掩码，再喂回同一个实现」，单一定义下没有原实现可回调，只能调实现函数
（`%REDUCE-*` 是内部名，公开符号仍然只有一个 `defun`）。

#### 16.32.4 判据（必须全部输出 0）

```sh
grep -c "setf (symbol-function" src/*.lisp    # 0（注释除外）
grep -c "(%cx-orig"            src/*.lisp    # 0（注释除外）
```

`complex-reload-test` 里有对应断言；`complex-dispatch-test` 的「定义名必须等于
符号本身」断言覆盖全部导出符号 —— 新增一个分派函数时它会自动纳入。

#### 16.32.5 新写复数分支时的四条

1. 不许 `(setf (symbol-function …))`，不许同名 `defun` 再写一遍；
2. 不许用 `%cx-orig` 回调「原实现」—— 没有原实现了，实数 body 就在同一个 `defun` 里；
3. 签名以 §16.33.6 的裁决表为准（那是 lambda-list 漂移的最终结论）；
4. 新增的 `%cx-op-*` 若定义在 `dispatch.lisp` 之后，同步补 `declaim ftype`（§16.33.2），
   否则编译期出 `UNDEFINED-FUNCTION` style-warning。

---

### 16.33 复数层的结构约束与签名裁决

> 完整过程（16 个符号怎么分类、`vt-log` 的 `:base` 除零事故、重载验收的做法、
> 最终统计）见 `HISTORY.md` §H.3.4。本节只留**现行的结构约束与签名裁决**。

#### 16.33.1 覆盖不仅掩盖重复，还会掩盖错误实现

`vt-real` / `vt-imag` / `vt-conj` / `vt-angle` 在 `extensions3.lisp` 里的那份
曾经写着「本库无复数 dtype，恒返回输入本身」——**是错的**，其正确性完全靠
`clvt.asd` 的文件顺序（后加载者胜）维持。调一下顺序，`(vt-conj #C(1 2))`
就会静默返回 `#C(1 2)`。

**这是禁止覆盖式分派的第二条理由**（第一条见 §16.32）：同名定义不只是「哪份
生效」不确定，它会让**错误实现**看起来和正确实现一样正常。

#### 16.33.2 标量内核必须前移，依赖方向单向

复数分支依赖的标量内核住在 `complex-kernels.lisp`，加载位置在
`dispatch` 之后、**所有归属文件之前**：

| 层 | 文件 | 加载位置 | 内容 |
|---|---|---|---|
| 标量内核 | `src/complex-kernels.lisp` | `dispatch` 之后、归属文件之前 | `%cop-*`（sqrt/log/pow/square/recip/signum/abs）、`%cx-nan-of`/`%cx-inf-of`/`%cx-nan-complex`、`%def-cx-to-real-run` + 4 个 `%cx-*-run`、`%cx-run-to-real`、`%cx-pow-2`、`%cx-to-out`、`%cx-flat`、view 的 3 个重解释内核 |
| 张量逻辑 | `complex.lisp` / `complex-ops.lisp` / `complex-linalg.lisp` | 末尾（不变） | 需要 shape/axis/dtype 的算法 |

依赖方向**只能是**「归属文件 → 标量内核」。反过来（归属文件等复数层）就是
第十三轮之前的状态：编译期只能前向引用，`declaim ftype` 无处可写。

`dispatch.lisp` 末尾维护一份 `declaim ftype` 清单（`%cx-op-unique` /
`%cx-op-nanmedian` / `%cx-op-clip` / `%cx-op-matrix-rank`），把「这些名字由
后面的文件提供」写成**可对账的契约** —— 新增复数分支时若忘了同步，编译期
立刻出 `UNDEFINED-FUNCTION` style-warning。

#### 16.33.3 `vt-log` 的 `:base`：不要写 `(log x base)`

CL 的 `(log x base)` 是**两参数对数** `log_x(base)`，不是「以 base 为底」。

```lisp
;; 错：算成 ln(base)/ln(1) = 0.693/0，with-float-safe 下得 Inf，再乘进结果变 0+0i
(complex (/ (realpart l) (log base 1.0d0)) …)
;; 对：
(complex (/ (realpart l) (log base)) …)
```

这个 bug 在覆盖式架构下活了五轮 —— 没有任何测试覆盖「复数 + 非缺省 base」这一格。
现已在 `complex-reload-test` 的探针里锁死。

#### 16.33.4 覆盖机制已全部删除（判据）

下面三条必须全部输出 **0**（`complex-reload-test` 里有对应断言）：

```sh
grep -c "setf (symbol-function" src/*.lisp    # 0（注释除外）
grep -c "(%cx-orig"            src/*.lisp    # 0（注释除外）
grep -c "eval-when.*cx-orig"   src/*.lisp    # 0
```

删除顺序（顺序颠倒会掩盖漏改的调用点，反过来的代价是运行期
`symbol-function` 返回 NIL 而不是编译期报错）：清 `(%cx-orig …)` 调用点 →
删 `eval-when` 预捕获清单 → 删 `*cx-originals*` 表与 `%cx-orig` 函数。

#### 16.33.5 重载验收：重载任何单个源文件都不许丢复数

套件 `complex-reload-test`。它在同一 image 里逐个 `load` 12 个源文件，
每轮跑 19 个探针（3 个 §16.32 代表 + 本节 16 个符号 + 全库无
`(setf (symbol-function …))`）。覆盖式架构下它必然失败 —— 这是那套机制
唯一真正的验收手段，也是回归网里最该保留的一个。

#### 16.33.6 lambda-list 漂移的最终裁决

合并同名 `defun` 时两处参数表不一致，等于「谁生效」由加载顺序决定。
下列 7 处已裁决，**新写复数分支时必须与这里的签名一致**：

| 符号 | 裁决 | 理由 |
|---|---|---|
| `vt-view` | `&rest` + 手工分流 | 兼容 `(vt-view a shape)`，且能表达 `(vt-view a :dtype dt)` |
| `vt-average` | `&rest`；复数 + 带参 → 报错 | 复数 weights 路径未验收，显式报错优于猜（§16.13） |
| `vt-clip` | 统一签名，`min-p`/`max-p` 显式传入内核 | `supplied-p` 语义必须原样保留 |
| `vt-clamp` | `&optional`（更宽） | `(vt-clamp a)` 原先在必填签名下报错；pytorch 语义允许只给一侧 |
| `vt-fromiter` | 推断式：含复数 → `:complex128`，否则 `:float64` | 既保住旧默认，又不静默丢虚部 |
| `vt-nanmedian` | 增加可选 `dtype` | 向后兼容 |
| `vt-matrix-rank` | `(tol nil)`，实数分支内部回落 `1e-10` | 两条路径各自的缺省口径不变 |

`vt-svd` 的 `max-sweeps`/`tol` 用 `supplied-p` 分开：实数 50/1e-14、
复数 60/1e-15，各自用验证过的缺省，显式传参时以调用方为准。

#### 16.33.7 结果

单一 `DEFUN` 分派 **159/159**；`setf (symbol-function)` 2 → **0**；
`%cx-orig` 调用点 16 → **0**；`eval-when` 清单 124 条 → **0**；
跨文件同名 `defun` 14 对 → **0**。明细与行数统计见 `HISTORY.md` §H.3.4.4。

---

### 16.34 构建、门禁与测试写法（契约）

> 本节只写**现在必须满足什么**。这些规则是怎么来的（第十四轮那两个「测试/构建
> 本身坏了」的缺陷、以及裸引号事故）见 `HISTORY.md` §H.3.1。

#### 16.34.1 codegen 门禁（`test/check_key_sync.py`）

`run-tests.sh` 的第一步，任何一条不满足即 `exit 1`、后续套件一个都不跑。

| 项 | 规定 |
|---|---|
| key 集合 / dtype / shape / 比较方式（`:exact`:`tol`） | 必须完全一致，否则 FAIL |
| `:tol` 条目的数值 | 按 dtype 分级容差：float64 `1e-12`、float32 `1e-6`（跨版本实测最大漂移 float64 2.1e-16、float32 1.0e-7，均 < 1 ulp） |
| `:exact` 条目的数值 | **逐字符相同**（该模式断言的就是逐位相等，文本一变运行期必失败） |
| 生成器退出码 ≠ 0 | FAIL —— 生成器崩了文件不会被重写，「文本相同」是**假通过** |
| 生成器顺带产出的 `tmp/` | 跑完删除，不许污染工作区 |

默认按**语义**比较；`--strict` 恢复逐字节口径（要钉死文本时用）。
**为什么默认不是逐字节**：`str(np.float64)` 的格式化随 numpy 版本变，
跨版本必然漂移，字节比对会让门禁在干净机器上必红（§10.4.1）。

#### 16.34.2 编译体检门禁：`WARNING` 为 0，`STYLE-WARNING` 走白名单

跑套件前先 `asdf:compile-system :clvt :force t`。**必须真的重编译** ——
命中缓存时不产生任何告警输出，门禁会静默失效。

| 项 | 规定 |
|---|---|
| 拦截 | `^; caught WARNING:` 计数 > 0 → FAIL 并打印上下文 |
| 拦截 | `STYLE-WARNING` 未命中白名单 → FAIL（判据见下） |
| 白名单 | 只有 4 条：3 × `&OPTIONAL and &KEY found in the same lambda list`（`vt-where` / `vt-clip` / `vt-clamp` 的位置参数形态）+ 1 × `Previously compiled call to CLVT:VT-P could not be inlined`（访问器在 layout 已知前被调用） |
| 判据 | **不是"总数 ≤ 4"**（会被「一条消失 + 一条新增」绕过），而是「两类各自条数钉死，且总数 == 两类之和」——任何第 5 条都会 FAIL |
| 跳过 | `--skip-build` / `CLVT_SKIP_BUILD_CHECK=1` |
| 顺带收益 | 这次重编译的 fasl 被后续套件复用，总耗时基本不增加 |

**为什么这 4 条留着**：前 3 条是公开 API 的位置参数形态（改成 `&key` 会同时改
调用方与错误信息），第 4 条是 SBCL 对「结构体 layout 尚未可见」的自述提示。
两者都不是本库的缺陷，且**改不掉而不改语义**。白名单只允许缩，不允许加 ——
真需要加时（例如新 API 又用了 `&OPTIONAL`+`&KEY`）必须同步改门禁脚本并在
`CHANGELOG` 里写明。

**清零的过程值**：79 → 4。主体是 `%cx-dispatch` 的 `ZERO`
参数（18 条 unused + 14 条 `IGNORE declaration for outer scope` + 6 条
`reading an ignored variable`）—— 把调用点的 `(declare (ignore zero))`
改成宏内 `(ignorable ,zero)`，一次消掉 38 条。其余是散落的 unused 绑定
与 4 条**无效的** `declaim inline`（声明晚于调用点，只产生告警不产生内联）。

**实现陷阱**：不要像套件那样包 `(handler-bind ((warning #'muffle-warning)) ...)` ——
`muffle` 会把编译期告警一起吞掉，门禁恒报「0 条」。门禁必须吃原始输出。

**这条门禁为什么必须存在**：`check-parens` 查不出字符串边界错误（括号是平衡的），
数值对拍也查不出（函数照样返回正确值），唯一信号就是编译输出。

#### 16.34.3 测试里判浮点特殊值的两条硬约束

| 约束 | 原因 |
|---|---|
| **NaN 一律用 `sb-ext:float-nan-p` 判定** | 测试进程故意不屏蔽浮点陷阱（SBCL 默认全开），`(= x x)` / `(/= x x)` 会直接抛 `INVALID-OPERATION` |
| **符号零必须显式判**（`(zerop x)` 且 `(minusp (float-sign x))`） | `=` / `eql` 下 `-0.0` 与 `+0.0` 相等，任何只写数值相等的断言都抓不到保号类 bug |

#### 16.34.4 未完成项

→ 见 §16.37（唯一清单）。

---

### 16.35 累积与归约的实现约束

> 来历（沿轴累积慢两个数量级的修复、多轴 `nanmedian` 的接通）见
> `HISTORY.md` §H.3.2。本节只写**改这几处代码时必须遵守什么**。

#### 16.35.1 平面与沿轴必须共用一份内核

`vt-cumsum` 的两条路径内层**完全一样** —— 都是「从某起点按固定步长走 N 个」，
差别只在段数与起点：

| 路径 | 段数 | 起点 | 步长 |
|---|---|---|---|
| `axis=nil`（展平） | 1 | `tensor.offset` | 1（要求全连续） |
| `axis=k` | `∏_{d≠k} shape[d]` | 除 k 维外各维索引折算 | `strides[k]`（可为负） |

**不要给两条路径各写一份循环**。只给 `axis=nil` 特化、沿轴走通用回退
（`funcall` 装箱 + 逐元素 `vt-cast` + 未声明类型的泛型 `aref`），会让同一个
函数在 `(vt-cumsum a)` 与 `(vt-cumsum a :axis 0)` 上差两个数量级。

两个实现细节（都实际踩过）：

1. **用 `labels` 不用 `flet`** —— `flet` 里兄弟函数互不可见，会编译成对外部
   `CLVT::RUN` 的调用，运行期报 `The function CLVT::RUN is undefined`。
2. **`ax` 必须在 `let*` 里先算好** —— 宏在展开期求值，宏体内写 `(if ax …)` 时
   `ax` 是词法变量、编译期不可见（`The variable AX is unbound`）。
   `(if ax (run-axis) (run-flat))` 必须落在**运行期**那一侧。

#### 16.35.2 多轴 `nanmedian`：折叠，而不是另写一套遍历

`axis=(0,1)` 之类多轴归约的实现方式是

> 置换 (保留轴…, 归约轴…) → 重塑 `(H, D)` → 沿单轴 1 归约 → 重塑回目标形状

**成立的前提**：中位数对归约区内元素的**顺序不敏感**（内部本就要排序）。
实数与复数共用同一份折叠函数（折叠只做置换与重塑，与 dtype 无关）；
`vt-reshape` 对非连续视图会自动落地，所以任意 strides / 负步长 / 偏移都成立。

> **这条前提不可外推**到对顺序敏感的归约（`cumsum`、`argmax`、`max` 遇到 NaN
> 时要「首个 NaN 胜出」之类）—— 那些必须真按轴遍历。

#### 16.35.3 单趟流式读写的函数必须显式过 `vt-out-snapshot`

§4.3 的别名约定对所有函数成立，但对**单趟流式**的函数尤其关键：读到第 i 个就
写第 i 个位置，out 一旦与输入错位重叠，先写的结果会污染还没读到的输入。

| 函数 | 结构 | 重叠安全 |
|---|---|---|
| `vt-cumsum` / `vt-cumprod`（实数） | 单趟流式读改写 | **必须显式快照** |
| `vt-nancumsum` / `vt-nancumprod` | 先算进临时向量，最后一次写回 | 两趟结构，天然安全 |
| 复数 `%cx-op-cum*` | 同上 | 天然安全 |
| `vt-map` / SIMD matmul / einsum | 显式 `vt-out-snapshot` | 安全 |

`vt-out-snapshot` 内部用物理区间判重叠（`%vt-views-overlap-p`），无重叠时
零开销、不复制。**新增任何「边读边写」的循环前，先确认它属于哪一类。**

#### 16.35.4 未完成项

→ 见 §16.37（唯一清单）。

#### 16.35.5 顺序统计的选取内核必须有两条退化护栏

`median` / `percentile` / `quantile` / `nanmedian` / `nanpercentile` / `nanquantile`
共用 `%select-double`（quickselect）。它必须在**两类真实输入**上都不退化：

| 退化源 | 症状（实测） | 必需的对策 |
|---|---|---|
| **大量重复值** | Lomuto 两分区每趟只剥掉 pivot 一个元素：200k 全零 **10.7 s**、两值 3.0 s | **三分区**（`<` / `=` / `>`）：等于 pivot 的整段一趟剥掉，全等输入 O(n) → 1 ms |
| **三数取中的坏排列** | 400k 整体降序且 `k = n/2-1`：451 趟、30.2M 次访问、**142 ms**（同数组 `k = n/2` 只要 1 趟 2 ms） | **访问量上限 4n**：超限改用 `%sort-fibers` 排完整段再取第 k 个，最坏情况钉在 O(n log n) → 21 ms |

三条约束：

1. 两条护栏缺一不可 —— 它们各自对应一类输入，互不覆盖（三分区治重复值，
   上限治坏排列；只做前者时降序仍慢 7 倍，只做后者时全零仍要十几秒）；
2. 护栏只能**改变执行路径**，不得改变结果：排序兜底用的是同一个缓冲区
   （quickselect 丢弃的区间只含「一定小于/大于第 k 个」的元素，全量排序后
   第 k 个仍然是答案，§1 铁律 1）；
3. 判据（`test/order-stat-test.lisp`）：
   - 与「先排序再按位置公式取」的独立参照**逐点一致**（6 种分布 × 7 种规模 × 5 个分位点）；
   - 重复值密集 / 已排序 / 逆序输入相对**随机输入**的倍率 ≤ 8×（倍率而非绝对
     毫秒：既避开机器速度，也能抓住 5000× 与 18× 这两类回归）。

> 同样的「护栏思维」适用于任何**期望线性、最坏可能平方**的选取/分块内核。
> 写新的 quickselect / 快速幂 / 哈希探测循环时，先问「最坏输入长什么样」。

---

### 16.36 dtype 契约与一条「此路不通」的结论

#### 16.36.1 `vt-nanmedian` 的 `:dtype` 不许静默降级

| 调用 | 规定 |
|---|---|
| 实数输入 + `:dtype` ≠ `:float64` | **报错**（结果 dtype 恒为 float64；要精度请再 `vt-astype`） |
| 复数输入 + 非复数 `:dtype` | **报错**（会把虚部静默丢掉，正中 §16.13 红线） |
| 缺省 | 实数 → float64；复数 → 保持复数 dtype |

§16.4 早就为 ufunc 定了同一句话「复数输入的结果 dtype 必须是复数类型」，
`nanmedian` 曾经没接上。错误消息里必须写清原因与替代做法 —— 只报
「不支持」会让调用方去猜。

> **已知且有意保留的 numpy 差异**：`np.nanmedian(float32)` 返回 float32，
> 本库实数路径恒为 float64（§16.10）。改它属行为变更，需单独裁决。

#### 16.36.2 复数 matmul：SIMD 之外的路线都已被实测否掉

实现复数 GEMM 时**不要**再尝试下面这些（数据与测量条件见 `HISTORY.md` §H.3.3.3）：

| 方案 | 结论 |
|---|---|
| SoA（拆 re/im 两个实数组）+ B 打包成 j-major | **更慢**（22.9 ms vs 标量 13.5 ms）。SBCL 的复数 `*`/`+` 是**开码**的，specialized 数组里的 `(complex double-float)` 就地 16 字节读写，不装箱、不分配 |
| SoA + i-j-p 单累加器（C 常驻寄存器） | **慢 5×**（75.2 ms）。`B[p,j]` 的 p 维步长是 n，每取一个元素跨一行，缓存全废 |
| 把 `p=0` 折进内核（省掉清零那一趟） | **更慢**（14.6 vs 13.5 ms）。j 展开的初始化要写 4 份，挤掉内层的寄存器分配；清零本身只有 O(mn) |
| 4 个局部累加器（C 只读一次、写一次） | **无稳定收益，已回退**。192 好 1.34×、256 差 10%、384 持平 —— 在噪声内 |

结论：瓶颈在复数乘法的**标量算术**（每元素 4 乘 2 加 + 16 字节装载解包），
不在 C 的访存。唯一可行方向是 SIMD：**按 j 维向量化 + shuffle 组 re/im**
（累加次序仍是 p 递增，所以仍满足「与朴素循环逐位相等」）。

SoA + 4 条实 GEMM 虽然能吃到实数线的 SIMD/并行内核，但**会改变累加顺序**，
与那条不变量冲突 —— 要先裁决不变量，而不是先写代码。

### 16.37 未完成项（唯一清单）

> 全库**只有这一处**维护「还没做的事」。各轮小节里的旧清单已删除，
> 只留指针指向本节 —— 否则每轮都会多出一份「待办」，而没人知道哪份是最新的。

| # | 项 | 现状与数据 | 为什么难 / 下一步 |
|---|---|---|---|
| U1 | 复数 `matmul` 的 SIMD | 标量内核已对齐实数标量线（§16.27.4：同规模约 2.2×，而 flop 比是 4×）；相对**实数 SIMD** 仍差 2.0–4.2× | 唯一可行路线是「交错布局上按 j 向量化 + shuffle 组 re/im」（保持 p 递增 → 仍满足铁律 1/2）。SoA+4 路实 GEMM 已实测更慢（§16.36.2）且改变累加顺序，本地累加器也已实测无收益（§16.36.2） |
| U2 | `real/imag` 返回共享视图（D-C3） | 现在返回副本；numpy 的 `z.real` 是同一缓冲的零拷贝视图 | 需给 strided view 引入「分量视图」（`data` + 分量偏移 + 元素宽度/2），会动物理层，影响 `vt-write`/`vt-copy-into`/别名判定；风险高于收益 |
| U3 | 复数逐元素算子的 SIMD 特化（D-C5） | §16.14.3 实测：`add` 1.8×、`mul` 2.3×、`sqrt` 4.0×、`abs` 6.5×（相对 float64） | 与 U1 同源；还需要先解决 `sb-simd` 的 shuffle 组装形式 |
| U9 | 排序族的**算法**（`sort`/`argsort`/`topk`/`median` 共用归并） | `median`/`nanmedian` 已改 quickselect、`topk` 已改有界堆（见 §16.14.6）；**剩下的只有 `sort`/`argsort` 本身**：1M `sort` f64 140 / c128 199、`argsort` 158 / 284，numpy 同机 5.9 / 83 / 19 / 100。c64 与 c128 耗时几乎相同 ⇒ 不是带宽受限，而是归并每次比较的分支不可预测（随机数据约 15 周期 × 20M 次） | 要再压只能换算法，但 **LSD 基数排序已实测否决**（IEEE754 单调键 + 两半 u32 + 8 趟 8 bit，1M double 107 ms/4 趟、8 趟约 200 ms，比归并 130 ms 更慢：64 位键在 SBCL 上是 bignum 必须拆两半，且每趟要把「键+值」两条流按 256 个桶散写）。剩下的路只有 introsort（numpy 的 float 路径），但那**破坏 `argsort` 的稳定契约**。保留现状 |


> **已关闭**（保留在此仅为说明它们不再是待办）：einsum 复数、复数归约多轴
> `axis` 元组、LU/SVD/特征值/百分位、`.npy` 复数互操作、`ComplexWarning`
> 等价物（有意不做）、`vt-nanmedian` 多轴、沿轴累积性能、编译 WARNING 门禁、
> **复数极值族与 `clip`**（804→7 ms / 1600→9 ms，根因是内核误用边界版
> `%cisnan`，见 §16.8.1）、**复数 `argsort` / `topk`**（2115→283 ms /
> 490→303 ms，下标归并 + 只算一次序）、**复数比较与谓词**
> （`vt-=` 60→4 ms、`vt-<` 45→5 ms、`vt-isnan` 25→1 ms，根因是标志内核
> 定义在 `complex.lisp`、编译早于它的 `elementwise2` 只能用函数调用 ——
> 即 §16.8.1 第 4 条）、**复数 `log` 族的分量内核**（352→55 ms，同时修掉
> `log(-0±0i)` 丢失 π 相位的静默错值，见 §16.14.4）、**四个「被晚加载文件
> 调用」的逐元素内核前移**（`%cop-sign` / `%cop-sinc` / `%cop-expm1` /
> `%cop-log1p`：`vt-sign` 104→26 ms、`vt-sinc` 218→81 ms、
> `vt-expm1` 116→58 ms、`vt-log1p` 383→65 ms —— 与上一条同源但**不触发**
> 编译期探测器，见 §16.8.1 第 4 条的「盲区」）。
> 编年史见 `HISTORY.md`，各轮摘要见 `CHANGELOG.md`。

---

## 17. 布尔 dtype（`:bool`）

**一句话**：`:bool` 是第九种非复数逻辑 dtype，对标 `numpy.bool_`。凡「NumPy 返回
`bool_`」的函数，缺省结果 dtype 一律是 `:bool`，**不再**用 `:int8` 的 0/1 冒充。

### 17.1 真值语义与物理表示

| 项 | 值 |
|---|---|
| 逻辑名 | `:bool` |
| 物理存储 | `(unsigned-byte 8)`，**1 字节**（与 `numpy` 的 `bool_` 同尺寸，`np.dtype(bool).itemsize == 1`） |
| 元素类型 | `vt-dtype->lisp-type` → `(unsigned-byte 8)`；`vt-dtype-itemsize` → `1` |
| 合法取值 | **只有 0 与 1**（0 = False，1 = True）—— 见下面的不变量 |
| 存储 dtype 归属 | 不在 `*vt-storage-dtypes*` 里：它与 `:uint8` 共用同一物理表示，靠 `dtype` 槽区分 |

**「数值 → 布尔」的唯一规则是真值化，不是截断**（`%bool-flag`，`dtype.lisp`）：

| 输入 | 结果 | 依据 |
|---|---|---|
| `T` / `NIL` | 1 / 0 | Lisp 布尔 |
| 整数 `0` / 非零 | 0 / 1 | `zerop` |
| 浮点 `0.0` / 非零 | 0 / 1 | `zerop` |
| `-0.0` | **0** | IEEE 的 `-0.0 = 0`（`np.bool_(-0.0)` 为 False） |
| `NaN` | **1** | `nan ≠ 0`（`np.bool_(np.nan)` 为 True） |
| `±Inf` | 1 | 非零 |
| 复数 | 任一分量非零 → 1 | `bool(0+0j)`→False、`bool(nan+0j)`→True |

> **为什么不能复用整数截断表**：`(truncate 0.5)` 是 0，而
> `np.array(0.5).astype(bool)` 是 **True**；`(truncate nan)` 是 0，而
> `np.array(nan).astype(bool)` 也是 **True**。截断表在 `|x| < 1` 与 NaN 两处
> 全错，且错得无声。所以 `vt-astype` 在进特化表**之前**就把 `:bool` 目标
> 分流到 `%to-bool`（`core.lisp`）。实测：
>
> ```lisp
> (vt-to-list (vt-astype (vt-from-sequence #(0.5 -0.0 nan inf) ) :bool))  ; (1 0 1 1)
> ```
> 对应 `np.array([0.5,-0.0,nan,inf]).astype(bool)` → `[True False True True]`。

**不变量 B1（布尔张量只含 0/1）**：任何 `dtype = :bool` 的张量，其每个字节必为
0 或 1。所有「写进布尔数组」的路径都必须自己保证这一点：

| 写入路径 | 保证方式 |
|---|---|
| `vt-astype ... :bool` | `%to-bool` 逐元素 `%bool-flag` |
| `vt-arange` / `vt-linspace :dtype :bool` | 同上（`(vt-arange 3 :dtype :bool)` 的字节是 `0 1 1`，打印为 `[NIL T T]`，与 `np.arange(3,dtype=bool)` 一致） |
| 比较 / 逻辑 / 谓词 / 成员判定内核 | 内核返回 0/1（`%op-eq` 一族、`%op-logical-*`、`%membership-map`） |
| 归约（`vt-any` / `vt-all` / `vt-isnan` …） | `+vt-mask-out-types+` 的输出类型矩阵用 `(unsigned-byte 8)` |
| npy 读入 `\|b1` | 任何**非 0 字节**解成 1（`numpy` 只保证 0/1，但不保证别人的文件守规矩） |
| 算术算子 | 见 §17.3 —— `+`/`*` 改派逻辑内核，`-` 直接报错 |

> 之所以要把它写成不变量：`:bool` 的物理类型是无符号字节，负号或进位会**回绕**
> 成一个仍在值域内的数（`-1` → 255），而写入路径是 `(safety 0)` 的，不会报错。
> 一个含 255 的「布尔」张量会静默污染 `vt-where` 的掩码判定与 npy 落盘字节 ——
> 这与「用 `:int8` 冒充布尔」是同一类隐患，只是更难发现。

**打印与序列化**：

* 打印走 `T` / `NIL`（Lisp 布尔，**不是** numpy 的 `True` / `False`），表头写
  `dtype:BOOL`（否则与 `:uint8` 无法区分，这正是「类型不可区分」漏洞的可见形态）：
  ```
  #<VT shape:(2 2) dtype:BOOL
    [[   T,  NIL],
     [   T,    T]]>
  ```
  为什么不用 numpy 的字面量，见 §17.8。
* npy 描述符是 `|b1`（`%npy-dtype->descr` / `%npy-descr->dtype`），与
  `numpy` 的 `np.bool_` 直接互通：`np.load` 读出 `dtype('bool')`。

### 17.2 类型提升

`:bool` 在提升里**不贡献位宽**，只在「全部操作数都是布尔」时才把结果留在 `:bool`
（`%promote-real-dtype`，`dtype.lisp`）。逐格实测（NumPy 2.3.5）：

| 运算 | 结果 dtype | NumPy 实测 |
|---|---|---|
| `bool ⊔ bool` | `:bool` | `np.result_type(np.bool_, np.bool_)` → `bool` |
| `bool ⊔ int8` | `:int8` | `np.array([True])+np.array([0],np.int8)` → `int8` |
| `bool ⊔ int16/32/64` | 同名宽度 | `int16` / `int32` / `int64` |
| `bool ⊔ uint8/uint16` | 同名宽度 | `uint8` / `uint16` |
| `bool ⊔ float32` | `:float32` | `float32` |
| `bool ⊔ float64` | `:float64` | `float64` |

**弱标量（NEP 50）的例外**（`%op-final-dtype`）：弱标量本来对任何 base 都
「不改变 dtype」，但布尔 base 必须例外 —— 否则 `True + 1` 会留在 `:bool` 里存 2：

| 表达式 | 结果 | NumPy 实测 |
|---|---|---|
| `bool张量 + 1`（弱整数） | `:int64` | `int64`（`np.array([True])+1`） |
| `bool张量 + 1.0`（弱浮点） | `:float64` | `float64` |
| `bool张量 + 1j`（弱复数） | `:complex128` | `complex128` |

**归约的结果 dtype**（`%op-out-dtype`，`reduce-stats.lisp`）：

| 归约 | `:bool` 输入 → 结果 | NumPy 实测 |
|---|---|---|
| `vt-any` / `vt-all` | `:bool`（全归约 → 0 维 `:bool`；带 `axis` → `:bool` 数组） | `np.any/np.all` → `bool_` |
| `vt-sum` / `vt-prod` / `vt-nansum` / `vt-nanprod` | `:int64` | `int64` |
| `vt-cumsum` / `vt-cumprod` | `:int64` | `int64` |
| `vt-mean` / `vt-var` / `vt-std` / `vt-average` | `:float64` | `float64` |
| `vt-amax` / `vt-amin` | `:bool`（单位元 1 / 0，见 `%small-int-extreme`） | `bool` |

> **`vt-any` / `vt-all` 返回的是 0 维张量，不是 Lisp 布尔** —— 与
> `vt-array-equal` / `vt-allclose` 不同（那两个对标 NumPy 里返回 **Python** `bool`
> 的函数，故给 `T`/`NIL`）。取标量用 `(vt-item (vt-any ...))`。这条差异是
> **刻意**的：`np.any` 返回 `np.bool_`（0 维数组语义），`np.allclose` 返回
> `bool`（真标量），本库照抄两者各自的返回类别。

### 17.3 布尔参与算术时的语义

引入 `:bool` 后，`vt-+` 这类通用算术算子第一次可能**收到**布尔张量。若照整数算，
`True + True = 2` 会直接写进布尔数组，破掉 §17.1 的不变量 B1。故算术入口按
NumPy 的 `bool_` 语义分派（`elementwise.lisp`）：结果 dtype 为 `:bool` 且操作数
全是布尔张量时，改派逻辑内核或报错。

| 表达式 | NumPy 2.3.5 实测 | clvt | 落点 |
|---|---|---|---|
| `b + b` | `bool`，`True+True = True` | 逻辑或，`:bool` | `%bool-fold` + `%bool-or-op` |
| `b * b` | `bool`，逻辑与 | 逻辑与，`:bool` | `%bool-fold` + `%bool-and-op` |
| `b - b` / `-b` | **TypeError**（布尔减法未定义） | **报错**（提示用 `vt-logical-xor` / `vt-logical-not`） | `vt--` / `vt-sub` |
| `b / b` / `b / 2` | `float64`（`[1., nan]`） | `float64` | `vt-/` 的 true_divide 提升 |
| `b // b` | `int8` | `int8` | `vt-div` 显式落 `:int8` |
| `b % b` | `int8` | `int8` | `vt-mod` 显式落 `:int8` |
| `abs(b)` | `bool`（恒等） | `:bool`（恒等） | 无特化，天然正确 |
| `maximum/minimum(b,b)` | `bool` | `:bool` | 同上 |
| `b + 1` | `int64`（值 2） | `:int64`（值 2） | §17.2 弱标量例外 |
| `b @ b` / `dot` / `outer` / `inner` / `tensordot` / `einsum` | `bool`（按与/或缩并） | `:bool`（int32 计数后真值化） | §17.10 |
| `invert(b)` | `bool`（逻辑非） | `:bool`（逻辑非） | §17.10 |
| `histogram(b)` / `diff(b)` / `trapz(b)` | `int64` / `bool` / `float64` | 同左 | §17.10 |

数值目标是**逃生门**：显式 `:dtype :int8` 等仍走整数路径，得到引入 `:bool` 之前
的旧结果（`(vt-+ b b :dtype :int8)` → `[2 2 0]`），保证已有代码逐位不变。

**已知偏差（有意保留，非缺陷）**：

| 项 | NumPy | clvt | 理由 |
|---|---|---|---|
| `b ** 2` | `int8` | `:int64` | NumPy 的 `power` 有一条历史遗留的 `bool ** int → int8` 专用循环，与 `2 ** b → int64` 自相矛盾。本库只保留统一的弱标量规则（§17.2），不复制这条循环 |
| `vt-sign(b)` | **UFuncTypeError**（`sign` 无 bool 循环） | `:bool`（恒等） | 布尔没有「符号」，把它定义成恒等比抛错更符合 `sign` 的数学含义；且不破坏 B1 |
| `np.signbit(nan)` | `False`（Python 的 `nan` 是**正** QNaN） | `True` | 本库的规范 NaN 来自 x86 `0/0`，是**负** QNaN（`0xFFF8000000000000`）。`signbit` 读的是符号位本身，两边都没错；要对齐就给 NumPy 喂同一个负 NaN |

### 17.4 返回布尔的函数清单

「NumPy 返回 `bool_`」的函数，clvt 侧的缺省结果 dtype 一律 `:bool`。
`dtype` 可显式覆盖（`§4` 的 `:out` 严格相等契约照旧生效）。

| 族 | clvt 函数 | NumPy 对应 | 缺省 dtype |
|---|---|---|---|
| 真值归约 | `vt-any` / `vt-all` | `any` / `all` | `:bool` |
| 内容测试 | `vt-isnan` / `vt-isinf` / `vt-isfinite` | `isnan` / `isinf` / `isfinite` | `:bool` |
| 内容测试 | `vt-isposinf` / `vt-isneginf` / `vt-signbit` | `isposinf` / `isneginf` / `signbit` | `:bool` |
| 比较 | `vt-=` `vt-/=` `vt-<` `vt-<=` `vt->` `vt->=` | `equal` `not_equal` `less` `less_equal` `greater` `greater_equal` | `:bool` |
| 逻辑 | `vt-logical-and` / `-or` / `-not` / `-xor` | `logical_and/or/not/xor` | `:bool` |
| 谓词 | `vt-positive-p` `vt-negative-p` `vt-zero-p` `vt-nonzero-p` `vt-even-p` `vt-odd-p` | （`>0` / `<0` / `==0` / `!=0` / `%2`） | `:bool` |
| 关系检查 | `vt-isclose` | `isclose` | `:bool`（逐元素） |
| 关系检查 | `vt-allclose` / `vt-array-equal` / `vt-array-equiv` | 同名 | **Lisp `T`/`NIL`**（对标 Python `bool`） |
| 成员判定 | `vt-in1d` / `vt-isin` | `in1d` / `isin` | `:bool` |
| 构造/转换 | `vt-astype ... :bool`、`vt-arange`/`vt-linspace :dtype :bool` | `astype(bool)`、`arange(dtype=bool)` | `:bool` |

**不适用（尚无对应能力，不要硬造）**：

| NumPy | 原因 |
|---|---|
| `np.isnat` | 本库没有 datetime / timedelta dtype，`NaT` 无从谈起。等引入日期类型时再补 |
| `np.isscalar` | 本库的「标量」是 Lisp 数（`numberp` / `%scalar-kind`），不是张量属性；`np.isscalar(0-d数组)` 是 False 这条语义在 Lisp 侧没有概念承载 |
| `np.char.*` / `np.strings.*` | 本库没有字符串 dtype（元素类型全是数值/复数/布尔） |
| `np.iscomplex` / `np.isreal`（逐元素） | 元素级「是不是复数」在 `interleaved` 存储下恒真/恒假，无信息量；本库提供的是**整数组**判定 `vt-iscomplexobj` / `vt-isrealobj`（对标 `np.iscomplexobj` / `np.isrealobj`，返回 Lisp 布尔） |

### 17.5 掩码消费方与执行层

布尔张量的**唯一用途**是当掩码，因此「吃掩码」的函数必须能认出 `:bool`，
且不能因此掉进慢路径：

| 消费方 | 对 `:bool` 的处理 |
|---|---|
| `vt-where` / `vt-select` | 掩码类型矩阵 `+vt-mask-out-types+` 由 `(signed-byte 8)` 改为 `(unsigned-byte 8)`；`%select-fast` 的 `macrolet` 同时展开有符号与无符号两种掩码，否则会退回 `vt-map`（实测慢约 8×） |
| `vt-compress` | 直接按非零判定，`0/1` 与 `T/F` 等价 |
| `vt-count-nonzero` / `vt-count` | 用 `:bool` 掩码再 `vt-sum`（升 `:int64`），不再借道 `:float64` |
| `vt-nonzero` / `vt-argwhere` | 按非零判定，与 `np.nonzero` 同 |
| 归约内核 | `def-vt-reduce` 的 `all`/`any` 小整型分支按**字面量**展开（`+small-int-lts+` 逐项），不能把运行期变量名塞进类型声明 —— 那会生成 `undefined type` 的静默慢路径 |

执行层的一条硬约束（承 §1 铁律）：**`dtype` 变化只影响选路与写回宽度，不影响结果**。
`(vt-= a b)` 与 `(vt-= a b :dtype :int8)` 的布尔值必须逐个相等，差别只在落盘宽度 ——
差分探针（`differential-probes-test{,2}.lisp`）用两种 dtype 各跑一遍来钉这条。

### 17.6 向后兼容

`:bool` 是**新增** dtype，不删不改任何旧入口：

| 旧写法 | 现在的结果 |
|---|---|
| `(vt-= a b)` | 结果 dtype 由 `:int8` 变为 `:bool`（**唯一的行为变化**，值仍是 0/1） |
| `(vt-= a b :dtype :int8)` | 与引入 `:bool` 之前逐位相同 |
| 吃 `:int8` 掩码的既有代码 | `vt-where` / `vt-compress` / `vt-select` 对 `:int8` 掩码照旧工作（掩码判定只看非零） |
| `vt-promote-type` 调用点 | `:bool` 落在整数族之下，任何含数值 dtype 的提升结果与旧表完全一致（§3.2 的 64 格未变） |

> **升级提示**：若既有代码对比较结果做了算术（`(vt-+ (vt-= a b) (vt-= c d))`），
> 现在会得到 **逻辑或** 而不是 `2`。这正是 NumPy 的语义；要旧行为就显式
> `:dtype :int8`。若既有代码对比较结果做减法，现在会**报错**（§17.3）。

### 17.7 示例脚本的连带修复

`example/` 下的两个文件（`example.lisp` 6898 行 / 106 个 `test-*` 函数、
`clvt-perf-test-examples.lisp` 96 条断言 + 约 60 项基准）此前不进任何自动化，
跑通与否无人知晓。把 `:bool` 落进去之后跑这两个文件，翻出三处示例自身的缺陷
—— 全部是**示例错、实现对**，但它们此前一直静默：

| 位置 | 现象 | 根因 | 处理 |
|---|---|---|---|
| `example.lisp` 9 处 | `(assert (vt-all X))` **恒真** | `assert` 只判条件是否为 NIL，而 `vt-all` 返回 VT 对象，任何 VT 对象都不为 NIL | 新增 `vt-all-true-p` 并替换；其实现见 §17.8（`vt-item` 改返回 T/NIL 后只剩一层解包） |
| `example.lisp` 4027–4028 | 真断言一开就失败 | 「全 NaN 时 `nanmax`/`nanmin` 返回什么」的期望写成了 `±Inf`；NumPy 返回 **NaN**（`All-NaN slice encountered`），clvt 本来就一致 | 改断言为 `vt-isnan`，删掉猜测式注释 |
| `clvt-perf-test-examples.lisp` 基准 | `extract` 216783 µs，且结果**不是** `a[mask]` | 参数顺序反了：写成 `(vt-extract a mask)`，即 1M 个 float 当条件、0/1 掩码当数据源 | 改为 `(vt-extract mask a)`（对标 `numpy.extract(condition, arr)`），耗时降到 21185 µs |

由此新增门禁套件 `test/example-smoke-test.lisp`（已登记进 `test/run-tests.sh`
的 `DEFAULT_ORDER`）：把两个示例当回归跑一遍，**只跑断言、不跑基准**——基准的
绝对值随机器与 GC 漂移，做门禁会变成间歇性红灯；需要基准时手动
`(clvt-examples:run)`。

两条可以推广的教训：

* **恒真断言比没有断言更危险**。没有断言时人知道自己没测；`(assert <非 NIL
  对象>)` 看起来像在测，实际恒过 —— 它把「没测」伪装成「测过」，而且换成
  `:bool` 之后连返回值语义都变了，断言文本却一个字没改。
* **示例里的参数顺序错误会伪装成性能问题**。`(vt-extract a mask)` 慢 10 倍，
  很容易被归因成「`vt-extract` 实现差」而去动内核；真正的原因是它在拿
  1M 个 float 逐元素判真。

### 17.8 逻辑视图与物理视图

`:bool` 落地后，「从布尔张量里取一个元素」不再只有一个答案。这层区分必须写死，
因为混用会直接变成 bug：`(if (vt-item x) ...)` 在 `x` 是 `T`/`NIL` 时正确，
在 `x` 是 `1`/`0` 时**恒真** —— Common Lisp 里 0 是真值，`(if 0 'a 'b)` 给 `A`。

**分层契约**：

| 接口 | numpy 对应 | `:bool` 张量的返回 | 定位 |
|---|---|---|---|
| `vt-item` | `.item()` → Python `bool` | **`T` / `NIL`** | 逻辑视图：解包成 Lisp 标量 |
| `vt-ref` | `arr[i]` → `np.bool_` | **`T` / `NIL`** | 逻辑视图：按坐标取一个元素 |
| `print-object` | `repr` → `True`/`False` | **`T` / `NIL`** | 逻辑视图：给人看，且可 `read` 回 |
| `vt-to-list`（默认） | `.tolist()` | `(1 0 1)` | 物理视图：可重建的数值序列 |
| `vt-to-list ... :bool t` | `.tolist()` 的布尔 | `(T NIL T)` | 逻辑视图：显式要求才给 |
| `%vt-ref-raw` | 无（内部） | `1` / `0` | 物理视图：内存里的那个字节 |
| `vt-data` | `.data` | `(unsigned-byte 8)` 0/1 | 物理视图：缓冲区本身 |

一句话：**逻辑视图用 Lisp 布尔，物理视图保持 0/1 字节**。

`vt-ref` 归逻辑视图、另外拆出 `%vt-ref-raw`，是因为它同时是**内部数值代码的
取数原语**（`src/` 内 23 处读点：`linalg` 的 Gram-Schmidt、`extensions3` 的
协方差/相关系数/叉积、`reduce-stats` 的 `vt-interp` 单调性检查 …）。这些点
需要的是数字，逐个论证「bool 不会流进来」既不可靠也没必要：一律改调
`%vt-ref-raw`，语义与旧 `vt-ref` 完全一致，且省掉每元素一次 `eq` 判断。

| 位置 | 原写法 | 改法 |
|---|---|---|
| 23 处内部读点 | `(vt-ref x i)` | `(%vt-ref-raw x i)` |
| `(setf vt-ref)` | `(coerce value (vt-element-type vt))` | `:bool` 目标走 `%bool-flag`，其余不变 |

`(setf vt-ref)` 必须一起改：布尔化之后 `(setf (vt-ref b 0) (vt-ref b 1))`
成了最自然的写法，而 `(coerce nil '(unsigned-byte 8))` 会直接报
`NIL can't be converted to type (UNSIGNED-BYTE 8)`。用 `%bool-flag` 归一
还能顺带接受 `0.5`、`-0.0`、NaN 这些真值化输入，与 `vt-fill` / `vt-put` /
`vt-astype` 的布尔语义对齐。

#### `vt-item` 为什么给 T/NIL

1. **语义对应物**。`np.bool_(True).item()` 返回 Python 的 `True`（不是 `1`）；
   Lisp 里与 Python `bool` 同位的就是 `T` / `NIL`。
2. **可用性**。改完 `(when (vt-item (vt-all m)) ...)` 直接成立；此前必须写
   `(= 1 (vt-item ...))`，而漏写一次就变成恒真断言（§17.7 那 9 处的来源）。
3. **可读回**。`T` / `NIL` 被 `read` 回来还是布尔；`True` 在 `common-lisp` 包里
   根本没有这个名字，`read` 会在当前包临时 intern 一个未绑定符号。

连带改动 4 处内部消费点 —— 它们原本都假定 `vt-item` 给数字：

| 位置 | 原写法 | 改法 |
|---|---|---|
| `vt-allclose` | `(not (zerop (vt-item (vt-all ...))))` | `(vt-item (vt-all ...))`（直接是 Lisp 布尔） |
| `vt-histogram` | `(= (vt-item (vt-all ...)) 1.0d0)` | `(unless (vt-item (vt-all ...)) ...)` |
| `vt-extract`（0 维 condition） | `(/= (vt-item condition) 0)` | `(%truthy-p (vt-item condition))` |
| `vt-bincount`（取最大值） | `(vt-item (vt-amax flat))` | `(%item-number (vt-amax flat))` |

后两处不能直接丢进 `if`：`vt-extract` 的 condition 可能是 `:int8` 掩码（此时
`vt-item` 仍给 `1`/`0`，而 0 是真值）；`vt-bincount` 要对 bool 输入算出 0/1 数值
（`np.bincount([True False])` → `[1 1]`）。于是配了两个转换原语：

| 原语 | 方向 | 用途 |
|---|---|---|
| `%truthy-p` | 任意值 → `T`/`NIL` | 消费「可能是布尔也可能是数值」的单元素 |
| `%item-number` | 任意值 → 数值（`T`→1、`NIL`→0） | 消费「可能是布尔」但要算数的单元素 |

`%truthy-p` 自带 `with-float-safe`（NaN 的 `zerop` 在陷阱开启时会抛
`FLOATING-POINT-INVALID-OPERATION`），因为它挂在索引/归约的各个分散调用点上。

#### `vt-to-list` 默认为什么不布尔化

`NIL` 在 CL 里**既是布尔假又是空表**，所以「默认给 `T`/`NIL`」这条路天然带
歧义。这条歧义本身不能靠猜解决，只能靠**显式声明 dtype** 来消解：
`VT-FROM-SEQUENCE` 收到 `:dtype :bool` 时，`NIL` 一律当叶子布尔假（§17.9），
于是 `(vt-to-list b :bool t)` → `(T NIL T)` → `(vt-from-sequence … :dtype :bool)`
这条往返**现在是成立的**（`bool-dtype-test` 的 `ctor/roundtrip-bool-view`）。

即便如此，默认值仍是 `(1 0 1)`，理由有两条：

1. **契约是「可重建的数值序列」**。`vt-to-list` 是导出接口，默认给数值才符合
   「拿去喂 JSON / 写文件 / 做数值比较」的直觉；布尔视图属于表示层需求，
   显式要更清楚。
2. **改默认会静默破现有代码**。`(vt-to-list b)` 此前恒为 `(1 0 1)`，改成
   `(T NIL T)` 之后任何 `(= 1 (car ...))`、`(apply #'+ ...)` 都会在运行时炸，
   而且因为 `NIL` 是真值，`(if (car ...) ...)` 这类判断会**静默变真**。

所以：默认 `0`/`1`，需要 Lisp 布尔时显式 `(vt-to-list b :bool t)`。非布尔张量
加该关键字无副作用。

#### 打印为什么用 T/NIL

除了与 `vt-item` 一致，还有一条硬理由：`*print-readably*` 为真时，打印结果应当
能被 `read` 回来。`T` / `NIL` 可以；`True` / `False` 不行。

本库 0 维形状本来就印 `shape:NIL`（Lisp 惯例，不是 numpy 的 `()`），值却用
Python 惯例会让同一次打印里出现两套风格。列宽不用特殊处理：`print-object` 的
列宽是按 `(length (%format-number ...))` 实测的，`T`(1) / `NIL`(3) 自动右对齐。

`%format-number` 的布尔分支返回的是**字面量字符串**，`*print-circle*` 为真时
SBCL 会把它当共享结构（打印成 `#1=T, #2=NIL, #1#, #2#`），所以那里保留了
`copy-seq`。

### 17.9 布尔字面量构造与 dtype 重解释

§17.8 解决了「怎么读出一个布尔」，这里解决「怎么写进去」。三处接口此前都停在
「bool 就是 0/1 数字」的旧契约上，导致 Lisp 布尔字面量根本进不了张量。

#### 17.9.1 `vt-from-sequence` 的 `:dtype :bool` 模式

| 输入（`:dtype :bool`） | 结果 | 说明 |
|---|---|---|
| `(list t nil t)` | `[T NIL T]`，shape `(3)` | `NIL` 当叶子 |
| `(list nil)` | `[NIL]`，shape `(1)` | **不是** shape `(1 0)` 的空数组 |
| `(list nil nil)` | `[NIL NIL]`，shape `(2)` | 同上 |
| `nil`（顶层） | 空数组，shape `(0)` | 对标 `np.array([], dtype=bool)` |
| `t`（标量） | 0 维 `T`，shape `NIL` | 对标 `np.array(True)`，shape `()` |
| `(list 2 -1 0 0.5d0)` | `[1 1 0 1]` | 数值走 `%bool-flag` 真值化 |
| `(list (list t nil) (list t))` | 报「不规则嵌套」 | 布尔模式不放宽规则形状 |

规则一句话：**指定了 `:bool` 就等于声明「内容是布尔值」，`NIL` 从此只是
布尔假，不再兼任空表**。这一条只在 `:bool` 模式下生效 —— 其余 dtype 的
`(vt-from-sequence (list 1 nil 3))` 仍然报「不规则嵌套」，行为不变。

代价是 Lisp 独有的歧义：`(vt-from-sequence '(() ()) :dtype :bool)` 得到长度 2
的 `[NIL NIL]`，而 numpy 的 `np.array([[],[]], dtype=bool)` 是 shape `(2, 0)`
的空数组。两者不可能同时满足（Lisp 没有「空布尔标量」这种字面量），
要空数组请写 `(vt-zeros '(2 0) :dtype :bool)`。

连带的两处：

| 接口 | 原行为 | 现行为 |
|---|---|---|
| `vt-flatten-sequence` | `(list t nil t)` → `(T T)`，静默丢 `NIL` | 新增 `:bool t`，→ `(T NIL T)`；默认不变 |
| `vt-put` | `(vt-put b 0 t)` 静默空操作；`(vt-put b '(0 1 2) '(t nil t))` 静默给 `(1 1 1 0)` | `t`/`nil` 识别为标量值；目标为 `:bool` 时按布尔叶子展平值列表，正确得 `(1 0 1)` |

`vt-put` 的静默空操作来自 `NIL` 落进 `(listp values)` 分支被展平成空表、
`T` 三个分支都不匹配 —— 两者都让 `n-values` 为 0 后直接 `return-from`。
这类「什么都不做」的失败比报错难查得多，所以 `t` / `nil` 的判断必须排在
`listp` 之前。

#### 17.9.2 `vt-view` 的 bool 重解释：只放行一个方向

`bool`、`uint8`、`int8` 都是 1 字节，`(vt-view b :dtype :uint8)` 得到
`[1 0 1]`（形状不变）。反向**不放行**：

```lisp
(vt-view (vt-from-sequence (list 1 2) :dtype :uint8) :dtype :bool)
;; => 报错：不支持 UINT8 → BOOL 的重解释（仅 complex128↔float64、
;;          complex64↔float32、bool→uint8/int8）
```

理由：numpy 的 `view(bool)` 是**纯字节直译**，`np.array([1,2],dtype=np.uint8)
.view(bool)` 打印 `[True True]`，但值 `2` 原封不动留在数组里（`.view(np.uint8)`
回来还是 `[1,2]`）。照搬就等于允许「bool 数组里存着 2」，直接违反 §17.1 的
0/1 不变量 —— 打印、`vt-item`、`asarray` 的每条断言都会失效。需要反向时用
`vt-astype :bool`（真值化，语义明确，`2 → 1`）。

顺带修掉一个老 bug：`(vt-view a :dtype dt)` 省略 shape 时，未指定被当成
`nil` 形状（0 维），对多元素张量报「元素总数不一致 (旧 4, 新 1)」。现在用
`:none` 哨兵区分「未指定」与「显式 nil」，未指定即保持原形状，与 numpy 的
`a.view(dt)` 一致。

#### 与 numpy 的已知差异汇总

| 项 | numpy | clvt | 说明 |
|---|---|---|---|
| 打印字面量 | `True` / `False` | `T` / `NIL` | 表示层差异，语义一致 |
| `.item()` / `arr[i]` | Python `bool` / `np.bool_` | `T` / `NIL` | 同位对应物；需要数值用 `%vt-ref-raw` |
| `.tolist()` | `[True, False]` | `[1, 0]`（默认） | 见上面的理由；要布尔用 `:bool t` |
| `.view(bool)` 反向（`uint8`→`bool`） | 允许，字节直译（值 2 会留下） | **拒绝**，需 `vt-astype :bool` | 保护 §17.1 的 0/1 不变量 |
| 跨 dtype 的 view | 共享内存 | 独立副本 | CL 无法跨元素类型做位移数组（§16.18 D-C7） |
| `np.array([[],[]], dtype=bool)` | shape `(2, 0)` 空数组 | `'(() ())` → shape `(2)` 的 `[NIL NIL]` | `NIL` 兼作空表；要空数组用 `(vt-zeros '(2 0) :dtype :bool)` |
| `np.extract(0维条件, a)` | `[a[0]]`（条件被当长度 1 数组） | 整个 `a`（真）/ 空（假） | `vt-extract` 的既有解释；这里只保证布尔条件不再崩、且与 `:int8` 条件行为一致 |
| `a[mask]` 花式索引 | 支持，返回被选中元素 | **不支持**（`vt-ref` 的索引必须是整数） | 布尔掩码的消费路径是 `vt-extract`（展平语义）与 `(vt-take a (vt-nonzero mask))`，见 §17.10 |

---

### 17.10 布尔写入路径与缩并的收口

「布尔全链路」审计（numpy 2.3.5 逐例对照，152 + 66 条差分用例，另加两轮
手工探针）补齐了**写入侧**的漏洞 —— 读侧的分层契约见 §17.8 / §17.9。这些漏洞
的共同点是 `:bool` 与 `:uint8` 共用 `(unsigned-byte 8)` 存储带来的「值域失守」，
或者 `vt-item` 布尔化之后漏改的内部解包点。

分两遍扫：第一遍（17.10.1–17.10.7）问「算出来的值对不对」，第二遍
（17.10.8–17.10.9）问「布尔**字面量**传不传得进去」—— 后者挖出的四处
都是 `T fell through ETYPECASE` 一个形状。

#### 17.10.1 `vt-map` 是唯一「用户函数决定元素值」的入口

| 表达式 | 修复前 | 修复后 | numpy |
|---|---|---|---|
| `(vt-map (lambda (x) 2) b)` | `(2 2)` —— 布尔数组里躺着 2 | `(1 1)` | `astype(bool)` → True |
| `(vt-map (lambda (x) 0.5d0) b)` | `(0 0)` —— 写回时被截断成 0 | `(1 1)` | True |
| `(vt-map (lambda (x) 5) b :dtype :bool)` | `(5 5)` | `(1 1)` | True |

修法不是「写完再扫一遍归一」，而是**在写回之前**把结果包一层 `%bool-flag`：
`0.5` 走 `%coerce-uint8` 就已经变成 0，事后看数组已经救不回来了。`vt-map`
按输入个数 1..5 生成对应的包装闭包（与 `%vt-map-run` 的 `call-fn` 特化同构），
超出的走 `&rest` 兜底。非 `:bool` 结果一行不受影响 —— `(vt-map (lambda (x) 200)
uint8数组)` 仍然是 200。

#### 17.10.2 `vt-bit-not`：布尔没有「位」

`(lognot 1)` 是 `-2`，写进 `(unsigned-byte 8)` 存储就是 **254**。这既不是
`False`，也直接破掉 §17.1 的 0/1 不变量（`vt-to-list` 给 `(254 255)`、
`vt-astype :int8` 给 `(-2 -1)`）。现在布尔输入改派 `vt-logical-not`，与
numpy 的 `invert(bool)` 逐值一致；显式 `:dtype :int8` 仍是字节取反（逃生门）。

#### 17.10.3 `vt-histogram`：`vt-item` 布尔化的漏网之鱼

`(vt-item bool张量)` 返回 `T`/`NIL`（§17.8）之后，凡是要拿结果继续做算术的
内部点都必须改调 `%item-number`。`vt-histogram` 自动定 bin 范围那一支漏了：

```lisp
(setf data-min (vt-item (vt-amin tensor))   ; => NIL / T
      data-max (vt-item (vt-amax tensor)))
;; 下一行 (- data-max data-min) → "NIL is not of type NUMBER"
```

`(vt-histogram bool数组)` 因此**直接报错**，而 numpy 正常工作（按 0/1 分箱，
`[T,F,T]` 给 `[1 0 ... 0 2]`）。改 `%item-number` 即可 —— 这也是判别「该用哪个
解包原语」的规则：**后面还要参与算术的，用 `%item-number`；只当条件用的，
直接 `vt-item`（T/NIL 在 `if`/`unless` 里语义正确）。**

#### 17.10.4 `vt-diff` 布尔 ≡ 逐次异或

numpy 的 bool 减法按位回绕：`True-False = 1`、`False-True = -1`，落回 bool
都是 `True`，所以 `np.diff(bool)` 恰好等价于相邻**异或**链（`n=2` 也一致：
`[T,F,T,F]` → `[False,False]`）。clvt 的 `vt--` 对布尔是明确拒绝的（§17.3），
原来也就跟着报错，且报错文案还写成「numpy 同样拒绝」—— 这句是错的。
现在 `vt-diff` 对布尔走 `vt-logical-xor`，报错文案同步修正。

#### 17.10.5 `vt-trapz`：整数/布尔按 float64 累加

被积项是 `0.5*(l+r)*h`，结果 dtype 跟着输入走的话，`0.5` 写进 `:bool` 得 0、
写进 `:int8` 也得 0 —— `(vt-trapz bool数组)` 静默返回 **0**（numpy 给 1.0）。
现在整数与布尔输入统一按 `:float64` 累加，浮点输入行为不变。

#### 17.10.6 布尔缩并：`b @ b` 是逻辑运算，不是数值和

| 表达式 | 修复前 | 修复后 | numpy |
|---|---|---|---|
| `(vt-matmul b b)` | `:float64`，值 `[[2 1][1 2]]` | `:bool`，`[[T T][T T]]` | 同左 |
| `(vt-dot b b)` | `:float64` | `:bool` | 同左 |
| `(vt-einsum "ij,jk->ik" b b)` | `:float64` | `:bool` | 同左 |
| `(vt-outer b b)` / `(vt-inner b b)` / `(vt-tensordot b b)` | `:float64` | `:bool` | 同左 |
| `(vt-trace b)` | `:int64` | `:int64`（不变） | `np.trace(bool)` 也是 int64 |

实现：`vt-einsum`（`dot`/`matmul`/`outer`/`inner`/`tensordot` 全部汇聚到它）
在「全部输入 `:bool` 且未显式给 `:dtype`」时，以 `:int32` 累加 0/1 的乘积，
最后一次 `vt-astype :bool` 收回真值。`sum(a&b) > 0` 与 numpy 的
`or(a&b)` 逐位等价，且 int32 计数不会像 bool/int8 累加器那样越界
（200 个 1 的内积实测仍是 `T`）。

这里曾记作「有意保留的偏差」，理由是「bool 累加器没有合法表示」—— 这个理由
只对**累加器**成立，对**结果**不成立：用中间 int32 计数、末步真值化就没有
越界问题。`:out` 为 `:bool` 时也按真值写回；显式 `:dtype :float64`
等仍是数值路径（逃生门）。

#### 17.10.7 仍然不同意的三处（记录在案，不是缺陷）

| 项 | numpy | clvt | 理由 |
|---|---|---|---|
| `a[mask]` | 支持 | 不支持，`(vt-ref a mask)` 报「索引必须是整数」 | 布尔掩码的消费路径是 `vt-extract`（展平）/ `(vt-take a (vt-nonzero mask))`。`vt-ref` 的契约是「取一个标量」，返回拷贝会破坏它 |
| `b ** b` / `b ** 2` | `int8` | `:bool` / `:int64` | `power` 在 numpy 里的 bool 循环是历史包袱（见 §17.3） |
| `vt-sign(b)` | 报错 | `:bool`（恒等） | 同上 |

#### 17.10.8 `T`/`NIL` 当「值」用：`ensure-vt` 与五个值位置

第一遍审计（§17.10.1–17.10.6）收的是「算错了」；第二遍按「哪些入口收得下
布尔**字面量**」扫，收的是「根本传不进去」。四处失败长得一模一样：

| 表达式 | 修复前 | 修复后 | numpy |
|---|---|---|---|
| `(vt-insert b 0 t)` | `T fell through ETYPECASE` | 插入一个真 | `np.insert(b,0,True)` |
| `(vt-pad b '(1 1) :constant-values t)` | `invalid constant-values: T` | 四边各一个真 | `np.pad(b,1,constant_values=True)` |
| `(vt-fill-diagonal m t)` | 只填对角第一个元素 | 整条对角线 | `np.fill_diagonal(m,True)` |
| `(vt-where b t nil)` | `T fell through ETYPECASE` | `:bool`，`np.where(b,True,False)` | 同左 |

前三处的根都在 `ENSURE-VT`：它的 `ETYPECASE` 只有 `vt` / `number` / `sequence`
三个分支，`T` 是符号，一个都不匹配。第四处一半在 `VT-WHERE`（分支走
`ensure-vt`），一半在 `VT-FILL-DIAGONAL` 的「标量还是序列」判据
（`(numberp value)` 对 `T` 判否）。

**`T` 与 `NIL` 的待遇不同，这是刻意的：**

| 入口 | `T` | `NIL` |
|---|---|---|
| `ENSURE-VT` | 值 1（未给 `:dtype` 时按 `:bool` 落 0 维真） | **空表**（shape `(0)`），除非显式 `:dtype :bool` → 值 0 |
| `VT-FROM-SEQUENCE` | 叶子（需 `:dtype :bool`） | 顶层 shape `(0)`；`:bool` 下是叶子（§17.10.1） |
| 五个值位置（下表） | 值 1 | 值 0 |

`T` 没有二义性 —— 它不可能是空表，所以任何 dtype 下都当值 1
（`(ensure-vt t :dtype :int64)` → `1`），这样 `(vt-append b t)`、
`(vt-insert int数组 0 t)` 也一并通了。

`NIL` 有真二义性，且在 `ENSURE-VT` 上**沿用「空表」**：300 多个内部调用点
靠 `(ensure-vt nil)` 得到空表来表示「没有序列」，改成「假」等于让一批
可选参数悄悄变成 0。所以只在**显式声明了 `:bool`** 时才把 `NIL` 读成假。
想要「一个假元素」写 `(list nil)` 或 `0`。

**值位置清单**（这里是「一个元素」的期望位置，`NIL` 只能是假）：

| 位置 | 处理 |
|---|---|
| `VT-PUT` 的值 / 值列表 | `%BOOL-FLAG`（§17.10.1 的连带修复） |
| `VT-FILL` 的 value | `VT-CAST` → `%BOOL-FLAG` |
| `VT-INSERT` 的 values | `ENSURE-VT … :dtype :bool` |
| `VT-PAD` 的 `:constant-values` | `%PAD-CV-BOOL-TREE` 递归换 1/0（支持 `'(nil t)` 与逐维 `'((t nil)(nil t))`） |
| `VT-FILL-DIAGONAL` 的 value | 标量判据加 `T`/`NIL` 分支 |
| `VT-WHERE` 的 x / y | 分支 `T`/`NIL` → 0 维 `:bool`（两分支都是符号时结果 `:bool`） |

`VT-FILL-DIAGONAL` 那条还有个「静默错值」的味道：`T` 落进序列分支后
`(ensure-vt t)` 是 0 维、`vt-ravel` 只剩 1 个元素，于是只填对角线第一个
位置，不报错。**静默错值 > 报错**，所以它比前三处更值得修。

#### 17.10.9 `%bool-wrap` 的 `ECASE`：`T` 不是默认子句

`%bool-wrap`（§17.10.1）按输入个数 1..5 生成包装闭包、其余走 `&REST` 兜底。
第一版写的是 `ECASE`，而 `ECASE` 里的 `T` **只是一个普通键**（匹配符号 `T`），
不是 `CASE` 的默认子句 —— 于是 `(vt-map f b1 … b6)`（6 输入）在
`%vt-map-run` 那边明明有 `otherwise` 分支，却先在 `%bool-wrap` 报
`no key`。改用 `CASE` 即可。回归见 J 组 `map/six-inputs` / `map/seven-inputs`。

**这条值得单独记**：`ECASE` 配 `T` 子句能编译、能通过 5 输入以内的全部测试，
只在 n≥6 时炸，而 `vt-map` 的 6 输入是文档支持的合法用法。

#### 17.10.10 布尔缩并的第二批：`convolve` / `correlate` / `cross`

§17.10.6 修掉了矩阵乘法家族（`matmul` / `dot` / `inner` / `outer` /
`tensordot` / `einsum`），但那轮是按「矩阵乘法」这个词去找的，漏掉了
**同样是缩并、只是名字里没有 mat 的算子**。按「有没有『乘积求和』这个结构」
重扫（而不是按名字）就能捞到三个：

| 表达式 | numpy 2.3.5 | clvt（修复前） | 现在 |
|---|---|---|---|
| `(vt-convolve b b)` | `:bool` `[T F T T T T T]` | `:float64` `[1 0 2 2 1 2 1]` | `:bool`，逐值一致 |
| `(vt-correlate b b)` | `:bool` `[T T T T T T T]` | `:float64` 数值和 | `:bool`，逐值一致 |
| `(vt-cross b b)` | **报错**（boolean subtract） | `:bool` `(1 1 1)` | 报错 |

前两个是「布尔代数被当成整数和」：`or_k (a[k] and b[k])` 与
`(sum_k a[k]*b[k]) > 0` 在 bool 上**逐值等价**（乘积只可能是 0/1，
和 >0 ⇔ 存在一个为真的乘积），所以实现不用改写内层循环 —— 收尾时
`(vt-astype res :bool)` 即可，`vt-convolve` 复用 `vt-correlate` 自动受益。

`vt-cross` 属于另一类：它**含减法**（`a1*b2 - a2*b1`），而布尔减法在
本库和 numpy 里都不成立。原实现把结果数组建在输入 dtype 上
（`(or dtype (vt-dtype av))`），于是 `-1` 被截断成 1、`1` 保持 1 ——
一块 3 分量的向量全变 `(1 1 1)`，**没有任何报错**。现在是显式
`error`，要数值结果请写 `:dtype :int64`。

> 判据：找这类缺陷时，「按算子名搜」不如「按 dataflow 搜」。
> 只要函数体里出现 `(incf sum (* ...))` 或 `(vt-- ...)` 而没有
> bool 分支，就值得看一眼。

#### 17.10.11 结果 dtype 的提升表：五个「值对不上是因为类型不对」的入口

第三遍换的轴是**结果类型**而不是值：不问「这个算子的答案是多少」，
而问「它宣称返回什么类型，实际返回什么类型」。理由是 §17.10.1–17.10.6
的六类缺陷全都是「类型对、值错」，而反过来的「值看起来对、类型错」
更难发现 —— 因为 `vt-to-list` 会把 0.785 显示成 0，看上去像「算错了」，
其实是「存不下」。

| 入口 | 修复前 | 现在 | numpy |
|---|---|---|---|
| `(vt-negative b)` | `:bool`，字节 **255** | 报错（可显式 `:dtype :int8`） | 报错 |
| `(vt-atan2 b b)` | `:bool`，值 **全 0** | `:float64`，`0.785…` | `float16` |
| `(vt-atan2 int64 int64)` | `:int64`，值 **全 0** | `:float64` | `float64` |
| `(vt-pow b 2)` | `:float64` | `:int64` `(1 1 0 0)` | `int64` |
| `(vt-rem b b)` | `:bool` | `:int8` | `int8` |

四条根因，三种成因：

1. **`vt-negative` 的 255**：`vt-fast-map #'-` 把 `-1` 交给
   `%coerce-uint8`，写进 `(unsigned-byte 8)` 得 255 —— dtype 仍是 `:bool`，
   字节已越界。`(vt-sum (vt-negative b))` 实测 765。与 §17.10.2 的
   `vt-bit-not` 是同一个坑的两次踩（`lognot` 也是 254/255）。
   **凡是「布尔数组里出现非 0/1 字节」，最终都会在某个归约里放大成怪数字**
   —— 所以 `bad-bytes` 这类不变量检查比逐函数断言更值钱。

2. **`vt-atan2` 从来没做过 dtype 提升**：它的 docstring 一直写
   「整数输入输出 float64」，实现却直接把 `#'atan` 丢给 `vt-fast-map`，
   结果类型走 `%op-final-dtype` 的**纯提升**规则（`bool ⊔ bool = :bool`、
   `int64 ⊔ int64 = :int64`）。于是 0.785 写进整数数组被截断成 0。
   这不是 bool 专有缺陷 —— **整数输入也全错**，只是 bool 更容易撞上。
   修法用 `%infer-float-dtype` 的同一规则：`float32` 进 `float32` 出，
   其余一律 `float64`。

3. **`vt-pow` 的整数白名单漏了 `:bool`**：`(and int-pow-p (member in-dt
   '(:int8 … :uint16)))` 这条规则的意图是「整数基 + 正整数指数保持整数」，
   但成员表里没有 `:bool`，于是落到 `(t :float64)`。按 §17.10.1 的
   「布尔 ⊔ 弱整数标量 → `:int64`」补上 `((and int-pow-p (eq in-dt :bool))
   :int64)`。（§17.10.7 的偏差表早就把 `b ** 2` 记成「clvt 给 `:int64`」，
   而实现给的是 `:float64` —— **文档与实现两边都没错，只是不一致**，
   这类「偏差表自我固化」正是 §17.10.6 里提过的风险。）

4. **`vt-rem` 少抄了三行**：`vt-mod` / `vt-div` 都有
   `(when (and (null dtype) (%all-bool-tensors-p ...)) (setf dtype :int8))`，
   `vt-rem` 没有。于是同一族余数函数里 `(vt-mod b b)` 给 `:int8`、
   `(vt-rem b b)` 给 `:bool`。

**验收方式**：把 dtype 本身当成被断言的对象（`(vt-dtype (f ...))` 而不是
`(vt-to-list (f ...))`），并配一条「物理字节 ∈ {0,1}」的扫描。回归见 K 组
（`negative/*`、`atan2/*`、`pow/bool-*`、`rem/bool-*`、`convolve/*`、
`correlate/*`、`cross/*`，共 22 条）。

> 三遍审计的轴互不重叠，这是刻意的：
> **第一遍问「值对不对」（差分），第二遍问「字面量进不进得去」（探针），
> 第三遍问「类型宣不宣称得住」（dtype 断言 + 字节不变量）。**
> 同一批代码在三遍里各暴露不同的一部分，交集很小。

#### 17.10.12 第四遍：不问「这个函数对不对」，只问「同族/同义的两处是否一致」

前三遍都是**逐个函数**问「它算得对不对」。第四遍换了提问方式：不问单个
函数，只找**成对的东西**——同一族里走了两条内核的，和同一个语义有两个入口
的。前提是前三遍留下的一个观察：bool 的缺陷几乎从不出现在「只有一处实现」
的地方，而总是出现在「这里改了、那里没改」的接缝上（§17.10.11 的
`vt-rem` 就是），所以成对比较比逐函数穷举更省力，也更容易命中。

| 对 | 一侧 | 另一侧 | 结果 |
|---|---|---|---|
| 累积族 | `vt-cumsum` / `vt-cumprod` | `vt-nancumsum` / `vt-nancumprod` | **不一致**（下面第 1 条） |
| 布尔真值判定 | `vt-astype` 数组路径 / `(setf vt-ref)` / `%bool-flag` | `vt-cast` 标量路径 | **不一致**（下面第 2 条） |
| 缩并族 | `vt-matmul` 家族（§17.10.6） | `vt-trace` / `vt-vdot` / `vt-kron` / `vt-matrix-power` / `vt-multi-dot` | 一致 |
| 提升族 | `vt-add` 系列 | `vt-concatenate` / `vt-stack` / `vt-where` | 一致 |
| 布局 | 连续视图 | 转置 / 负步长 / 广播（stride 0） | 一致（352 次比较） |
| 序列化 | `vt-to-list` | `vt-save`/`vt-load`（`:lisp` 与 `:npy`） | 一致（npy 头是 `|b1`） |

**第 1 条：`nancumsum` / `nancumprod` 漏掉提升表 → 静默溢出**

`vt-cumsum` 走的 `%VT-CUMULATIVE` 里有一张提升表（`:bool :int8 :uint8
:int16 :uint16 :int32 → :int64`，注释还专门写了「布尔与窄整型一致」）。
`nancumsum` 走的是另一条内核 `%E3-CUMULATIVE-SKIP-NAN`，那里只有
`(or dtype (vt-dtype input))`：

| 表达式 | numpy | 修复前 | 现在 |
|---|---|---|---|
| `(vt-nancumsum b)` | `int64` `[1 2 2 2]` | `:bool`，**字节 2** | `int64` `[1 2 2 2]` |
| `(vt-nancumsum #(100 100) :dtype :int8)` | `int64` `[100 200]` | `int8` **`[100 -56]`** | `int64` `[100 200]` |
| `(vt-nancumsum #(200 200) :dtype :uint8)` | `uint64` `[200 400]` | `uint8` **`[200 144]`** | `int64` `[200 400]` |
| `(vt-nancumsum #(30000 30000) :dtype :int16)` | `int64` | `int16` **`[30000 -5536]`** | `int64` `[30000 60000]` |
| `(vt-nancumprod #(100 100) :dtype :int8)` | `int64` `[100 10000]` | `int8` **`[100 16]`** | `int64` `[100 10000]` |

注意这条**不只是 bool 的问题**：bool 那行只是「脏字节」，窄整型那三行是
**静默回绕成负数**，比 bool 更难发现——累加器溢出后值域整体翻负，输出仍
「看起来像合法数据」。修法不是给 bool 打补丁，而是把那张表抽成
`%CUMULATIVE-RESULT-DTYPE` 给两条内核共用。

> 判据：**累积类算子永远不能沿用输入的窄类型**。sum/prod/cumsum 早就
> 这么做了，`nan` 前缀的那一族是后加的，加的时候没回头看。

**第 2 条：`vt-cast` 把 `:bool` 分支放在了「剥虚部」之后**

```lisp
;; 修复前：先剥虚部，再判 :bool
(when (and (complexp val) (not (vt-complex-dtype-p dtype)))
  (setf val (realpart val)))        ; #C(0 1) → 0
(when (eq dtype :bool)
  (return-from vt-cast (%bool-flag val)))   ; 0 → False
```

于是 `(vt-cast #C(0 1) :bool)` = **0**，而同一个值的数组路径
`(vt-astype … :bool)` = **1**、`(setf (vt-ref …) #C(0 1))` = **1** ——
`%BOOL-FLAG` 的 docstring 明写「复数：任一分量非零即为真」，标量路径却
先把虚部扔了。受影响的是所有走 `vt-cast` 的写入入口：
`vt-write` / `vt-put` / `vt-fill` / `vt-pad :constant-values`。
（`(setf vt-ref)` 没受影响，因为它直接调 `%bool-flag`——这正是「成对比较」
能捞出来的原因：五个入口里四个错、一个对。）

修法是把 `:bool` 分支提到剥虚部之前。这不是 bool 专属语义的改动：
它让「标量 → 布尔」重新服从 `%BOOL-FLAG` 这一个定义，而不是在
`vt-cast` 里另立一份。

**第四遍同时确认「没有问题」的部分**（记在这里免得下次重做）：

* **归约器的关键字矩阵**：`sum`/`prod`/`mean`/`amax`/`amin`/`any`/`all`/
  `argmax`/`argmin`/`std`/`var`/`median` × `{默认, :keepdims, :axis 0,
  :axis -1, :axis '(0 2), :axis '()}` 与 numpy 逐项一致；`sum`/`prod`/
  `cumsum`/`cumprod`/`argmax`/`trace` → `:int64`，`amax`/`amin`/`any`/`all`/
  `nanmax`/`nanmin` → `:bool`，`mean`/`std`/`var`/`median` → `:float64`。
* **dtype 提升表**：`bool ⊗ {int8,int64,float32,float64}` 的算术、
  `vt-concatenate`/`vt-stack`/`vt-append`/`vt-where` 的混合输入，全部与
  numpy 一致（bool 在前时取另一侧 dtype，两侧都 bool 时仍是 bool）。
* **布局不变性**：对 42 个一元算子 × 8 种布局（转置、flip、步长 2、
  负步长、广播、组合），比较 `f(view)` 与 `f(连续副本)`，352 次全等。
* **`:out` 的严格性**：`(vt-sum b :out int8数组)` 报错而不是像 numpy 那样
  静默降级转换——这是本库早于 bool 就有的约定（§4.4），与 bool 无关。

**顺带更正一处旧结论**：§17.10.11 里写过「`np.percentile(bool)` 报错，
clvt 报错是对的」。实测 clvt **不报错**：`(vt-percentile b 50)` 给 `1.0d0`、
`(vt-quantile b 0.5)` 给 `1.0d0`（`vt-ptp` / `vt-gradient` / `vt-ediff1d`
确实报错）。这个差异是**有意的**：numpy 的分位数实现要用 `hi - lo`
（布尔减法未定义）才抛错，而本库在 float64 缓冲上取位，结果在数学上是
对的，没必要为了对齐实现细节去拒绝一个能算对的答案。契约改为：

| 表达式 | numpy | clvt（契约） |
|---|---|---|
| `median(b)` / `nanmedian(b)` | `float64` 正常 | 同 |
| `percentile(b, 50)` / `quantile(b, 0.5)` | **报错** | `float64` 正确值（有意偏差） |
| `ptp(b)` / `gradient(b)` / `ediff1d(b)` | 报错 | 报错（语义本身就是减法） |

#### 17.10.13 第五遍：不问「布尔当数据算得对不对」，问「布尔被谁读走」

前四遍都在布尔**作为操作数**的位置找问题：值（差分）、字面量（入口）、
结果类型（dtype + 字节）、成对一致性。第五遍换的轴是**消费侧**——布尔张量
被当成别的东西读走的地方：当索引、当形状、当「用户函数返回值的容器」。

这一遍不再人工挑函数，改成机械穷举：把 `:clvt` 的 375 个导出函数按参数
个数试调 `(f b)` / `(f b b)` / `(f b int64)` / `(f int64 b)`，共 ~1250 次，
逐条记录 dtype / 形状 / 值 / **物理字节**，再与 numpy 的同名函数对照。
捞出的六处都不在算术族里，按算子名扫是扫不到的。

| 入口 | 修复前 | 现在 | numpy |
|---|---|---|---|
| `(vt-reciprocal b)` | `:bool`，`(1 1 1 1)`（假值位置变真） | `:int8`，零除报错 | `int8` `[1 0 1 1]` |
| `(vt-divmod b b)` | 商 `:bool`，`(T T T T)` | 商 `:int8` `(1 0 1 1)` | `int8` `[1 0 1 1]` |
| `(vt-ptp b)` | `T is not of type NUMBER` | 布尔减法消息 | 报错 |
| `(vt-broadcast-shapes b '(2))` | `Unhandled memory fault at #x0` | 参数类型消息 | — |
| `(vt-apply-along-axis #'vt-sum 1 b)` | `:bool`，`(1 1)` | `:int64`，`(3 3)` | `int64` |
| `(vt-apply-along-axis … 返回 #(5 6))` | `:bool`，字节 **5、6** | `:int64` | 跟随返回值 |
| `(vt-vander b)` | `:bool` | `:int64` | `int64` |

**第 1、2 条是同一个缺口的两侧**：`vt-int-dtype-p` 只认 `:int8/:int16/
:int32/:int64`，不认 `:bool`（这是对的——布尔不是整数），但需要「整数语义」
的两处却用这个判定当分支条件。`vt-reciprocal` 因此落到浮点路径：
`1/0` 在 with-float-safe 下是 `+Inf`，写回 `:bool` 又被 `%bool-flag` 收成 1，
`(T NIL T T)` 静默变成 `(T T T T)`。`vt-divmod` 则是 `dtype` 缺省走提升规则
（bool ⊔ bool = bool），`floor(0/0)` 的 NaN 同样被收成 1。两处都改成
「全布尔输入 → `:int8`」，与 `vt-mod`/`vt-rem`/`vt-div`（§17.10.11 第 4 条）
并齐。

**第 3 条是偏差表自我固化的第三个实例**（前两个：`b ** 2`、`matmul`）。
§17.10.12 的表格写的是「`ptp(b)` 报错」，实现也「报错」——但报的是
`The value T is not of type NUMBER`：非 axis 分支写的是
`(- (vt-item …) (vt-item …))`，而 `vt-item` 自 §17.8 起对布尔返回 T/NIL。
**表格只记了「报不报」，没记「报什么」**，于是「报错」这一栏一直在骗人。
同一函数的 `:axis` 分支走 `vt--`，报的却是正确的布尔减法消息。

**第 4 条与新轴无关、但值得留一笔**：`vt-broadcast-shapes` /
`vt-broadcast-strides` 带 `(declare (list …))` + `(safety 0)`，传一个 VT
结构体进去时 `(length shape1)` 被编成取 car 链，读到结构体的非法槽位。
误用理应拿到一句错误消息，而不是让整个 Lisp 进程躺下。改法是**删掉 list
声明、换成运行时 `listp` 检查**——留着声明，编译器会把检查优化成 T。
代价是每次调用两次 `listp`（这两个函数在每个二元算子都会跑，实测无可测
差异）。

**第 5 条是这一遍最值钱的**：`vt-apply-along-axis` 的结果 dtype 固定取
输入 dtype，于是布尔输入下 1D→1D 分支直接 `aref` 写 5、6 ——
**布尔数组里出现越界字节**，正是 §17.1 的核心不变量。现在布尔输入的结果
dtype 由 FUNC 的返回值决定（1D→scalar 分支看返回标量/0 维张量的 dtype，
1D→1D 看返回张量的 dtype），非布尔输入维持旧契约（结果跟随输入）。这条
同时修掉了「`(vt-sum row)` 的 3 被压成 1」。

**偏差族：numpy 对布尔的超越函数落 `float16`，本库落 `float64`。**
机械扫描捞到的最大一簇不是 bug 而是这个：`sqrt` / `log` / `log2` /
`log10` / `exp` / `sin` / `cos` / `tan` / `sinh` / `cosh` / `tanh` /
`asin` / `acos` / `atan` / `asinh` / `acosh` / `atanh` / `log1p` / `expm1` /
`cbrt` / `spacing` / `deg2rad` / `rad2deg` / `atan2` / `hypot` /
`copysign` / `nextafter` / `logaddexp` / `round` / `rint` 共约 30 个，
numpy 2.3.5 对 `bool` 输入一律选 `float16` 的循环。本库一律 `float64`
（§17.10.11 的 `atan2` 就是按这条定的）。**不跟**：`float16` 只有 3 位
十进制有效数字，`sqrt(2)` 给 1.414，比「与 numpy 的 dtype 字符串一致」
重要得多。记为**有意偏差**，与 §17.10.7 的偏差表同性质。

同类但更小的三处（值相同、dtype 不同，同样不跟）：

| 表达式 | numpy | clvt | 理由 |
|---|---|---|---|
| `square(b)` / `conj(b)` | `int8` | `:bool` | numpy 自己的 `b*b` 是 bool、`square(b)` 是 int8，内部就不自洽；本库取自洽的一侧 |
| `round(b)` / `rint(b)` | `float16` | `:bool` | 布尔的 round 是恒等，保留 bool 比升 float16 更不意外 |
| `clip(b, 0, 1)` | **报错**（`positive` 无 bool 循环） | `:bool` 恒等 | 本库的 clip 不走 ufunc 的 positive，能算对就不拒绝 |

**第五遍确认「没有问题」的部分**（记在这里免得下次重做）：

* **布尔当索引/掩码/计数**：`take` / `put` / `choose` / `select` /
  `compress` / `extract` / `take-along-axis` / `put-along-axis` /
  `searchsorted` / `bincount` / `digitize` / `unique` / `isin` / `in1d` /
  `delete` / `setxor1d` / `setdiff1d` / `intersect1d` / `union1d` /
  `nonzero` / `flatnonzero` / `argwhere` / `count-nonzero` —— 23 个入口
  与 numpy 逐项一致（`take`/`put`/`choose` 按 0/1 整数索引，
  `compress`/`extract` 按掩码，两套语义并存，与 numpy 相同）。
* **写入路径的字节卫生**：`:out` 为 `:bool` 且结果 dtype 不符时一律报错
  （§4.4 的严格性）；`vt-copy-into` / `vt-copy-to!` / `(setf (vt-slice …))`
  标量分支与张量分支、`vt-fill` / `vt-fill-diagonal` / `vt-put` 全部经
  `%bool-flag`，`(setf (vt-slice b …) 2)` 得 1 而不是 2。
* **`vt-from-sequence` 的混合嵌套**：`'((t nil) (1 0))` / `'(t 1 nil)` 在
  `:dtype :bool` 下都把 T/NIL 与 0/1 读成同一个真值；不给 `:dtype` 时
  报「不规则嵌套」（与 numpy 给 object 数组不同，这是刻意的确定性）。

> 方法论上的一个坑：**机械扫描时每个用例必须重建输入**。第一版扫描复用
> 同一对 `*b*` / `*bi*`，`vt-random-shuffle` / `vt-sort` / `vt-put` 这类
> 原地函数把它们改掉之后，后面的 `vt-mul` / `vt-vstack` / `vt-take` 全部
> 报出「差异」—— 查了半小时才发现是测试夹具被污染，不是库的缺陷。
> 一个只读工具的扫描器里混进一个就地写入的函数，污染的是**之后的所有行**。

#### 17.10.14 第六遍：布尔不是「被算的」，是「被造出来的」和「被当坐标用的」

前五遍的提问对象全在**动词**一侧：布尔参与运算算得对不对（§17.10.1–7）、
传不传得进去（§17.10.8–9）、返回什么类型（§17.10.11）、两处是否一致
（§17.10.12）、被谁读走（§17.10.13）。这一遍换到**名词**一侧：
一个布尔张量是从哪来的（构造函数），以及它被放在「位置参数」位置上时
是什么含义（掩码还是坐标）。

> **numpy 对 positions / obj 参数有两套并存语义，且按 dtype 而非按值分叉。**
> 同一份 `#(1 0 1 1)`：`:uint8` 走**索引**（`delete(a, u8)` → 删掉位置
> 0 和 1），`:bool` 走**掩码**（`delete(a, b)` → 删掉 True 的位置）。
>
> | 走掩码 | 走索引 |
> |---|---|
> | `delete` / `insert` / `place` / `putmask` / `compress` / `extract` | `take` / `put` / `choose` / `bincount` / `searchsorted` / `digitize` |
>
> 判定依据是「这个参数描述的是**逐元素去留**还是**要取哪些下标**」。
> 前者必须与轴等长（numpy 会校验），后者长度任意。

**修的三处**

| 入口 | 修复前 | 现在 | numpy |
|---|---|---|---|
| `(vt-delete #(0 1 2 3) #(T NIL T T))` | `(2 3)`（当索引，删 0 和 1） | `(1)`（掩码） | `[1]` |
| `(vt-insert #(0 1 2 3) #(T NIL T T) 9)` | `(9 0 9 9 9 1 2 3)`（插 4 次） | `(9 0 1 9 2 9 3)` | 同 |
| `(vt-geomspace 1 8 3 :dtype :bool)` | 报「不支持的 dtype BOOL」 | `:bool` `(T T T)` | `[True True True]` |

前两条是同一个根因：`%check-integer-index` 只校验「是不是整数」，而
`:bool` 的底层类型是 `(unsigned-byte 8)`，于是**畅通无阻地走了整数索引
路径**。危险不在结果错，而在**静默**：`np.delete(a, a > 3)` 是最常见的
numpy 写法之一，移植过来会删掉一批完全不相干的元素，而且不报错。
修法是按 dtype 显式分叉（`%BOOL-MASK-POSITIONS`），而不是给 `:bool`
打一个「也当索引」的补丁 —— 后者会让两套语义永远分不清。

第三条是三处「按区间采样」的构造函数里唯一拒绝 bool 的：`linspace` /
`logspace` 一直支持（走 `vt-astype` 真值化），`geomspace` 的白名单漏了
`:bool`。同一族三个入口两种行为，属于 §17.10.12 那类「成对不一致」，
只是这次的对象是**创建**而不是运算。

**第六遍确认「没有问题」的部分**

* **构造函数全族**：`zeros` / `ones` / `full`（`2`、`0.5`、`T`、`NIL`
  四种值）/ `eye`（含 `:k`）/ `identity` / `tri` / `triu` / `tril` /
  `diag` / `diagflat` / `from-array` / `empty` / `empty-like` /
  `zeros-like` / `ones-like` / `full-like` / `repeat` / `tile` /
  `meshgrid` —— 结果 dtype 均为 `:bool`，物理字节全在 `{0,1}`。
  其中 `vt-full` 对 `2` / `0.5` 都真值化成 1（numpy 同），
  `vt-empty` / `vt-empty-like` 给 0 而 numpy 给未初始化内存 —— 本库更安全，
  记作有意偏差。
* **打印层**：`princ` / `print` / `format ~a` / `~s` 四条路径，
  一维与二维都渲染成 `T` / `NIL`（`[[   T,  NIL],` 对齐），
  0 维布尔张量渲染成单个 `T` / `NIL`。
* **极值尺寸**：`(0,)` / `(3 0)` / `(0 0)` 的空布尔张量在 `sum`
  （→ `int64` 0）、`reshape`、`concatenate`（空 + 非空 = 非空）、
  `vstack` 上全部正常；空归约（`amax` / `argmax`）报 numpy 同款错误；
  1 元素与 10⁶ 元素无异常。
* **布尔参与 linalg / 统计的拒绝面**：`gcd` / `lcm` 对 bool 报
  「仅支持整型输入」，numpy 报 `UFuncTypeError`（同样拒绝）；
  `random` 对 bool 报「仅支持浮点」（truncate 会静默全 0）。

> 这一遍的探针自己撒了两次谎，都值得记：一是 `vt-concatenate` 报
> 「收到 0 个张量」，查了才发现签名是 `(vt-concatenate axis &rest tensors)`
> —— axis 在前，我按 numpy 的顺序传了 list；二是拿 Python 解析 Lisp
> 打印的 `(1 0 1 1)` 时索引取错列（把形状当成了值），得出「0 处一致」
> 这种一眼假的结果。**当探针的错误率逼近它发现的缺陷率时，就该停了** ——
> 这也是这里只有三处修复的原因。

---

#### 17.10.15 第七遍：把「默认参数」这个隐藏前提拆掉

前六遍有一个共同前提，一直没被审视过：**所有调用都用默认参数**。
布尔参与运算算得对不对、传不传得进去、返回什么类型、两处是否一致、
被谁读走、从哪造出来 —— 这些问题的每一次探针，第二个参数之后都是空的。

这一遍把 `:axis` / `:keepdims` / `:dtype` / `:ddof` 打开，对 46 个接受
`axis` 的导出函数做参数矩阵（`axis ∈ {0,1,-1,-2,(0,1),(1,2),(0,2)}` ×
`keepdims ∈ {nil,t}` × `dtype ∈ {省略,:bool,:int64,:float64}`），
共 303 组。

##### 一、`dtype` 说是 `:bool`，字节就可以不是 0/1

| 表达式 | 修复前 | 现在 | numpy |
|---|---|---|---|
| `(vt-sum b :axis 0 :dtype :bool)` | `:bool`，`vt-to-list` 给 `(2 1 1 2)` | `(1 1 1 1)` | `[True True True True]` |
| `(vt-sum b :axis 1 :dtype :bool)` | `(3 3)` | `(1 1)` | `[True True]` |
| `(vt-nansum b :axis 0 :dtype :bool)` | `(2 1 1 2)` | `(1 1 1 1)` | 同 |
| `(vt-l1-norm b :axis 0 :dtype :bool)` | `(2 1 1 2)` | `(1 1 1 1)` | — |

根因是**归约内核里有两条写回路径**：

- `%KERNEL-SMALL-GENERAL` 写 `(setf (aref od out-ptr) (vt-cast acc final-out-dtype))`
  —— 经过 `vt-cast`，`%bool-flag` 会把 2 收成 1；
- `%SMALL-INT-AXIS-FORM` 生成的 `%KERNEL-ROW-ACC` / `%KERNEL-SINGLE-AXIS`
  实例写的是 `(setf (aref od out-ptr) acc)` —— **裸写**，而 `od` 声明为
  `(simple-array (unsigned-byte 8) (*))`，于是 2 原样落盘。

这不是「值不对」而是**布尔数组里出现越界字节**，破的是 §17.1 那条从第一轮
守到现在的 0/1 不变量；之后任何 `(aref data i)` 的直接读取都会拿到 2。

修法不在内核里加 `if`（要改十几个宏实例），而是在归约出口统一调
`%BOOL-CLAMP!`（按 shape/strides 只碰逻辑元素，共享缓冲的其它位置不动）。
对已经过 `vt-cast` 的路径它是幂等的，所以不必判断走了哪条内核。

##### 二、报错点名了三层以内的函数

| 用户调用 | 修复前的消息 | 现在 |
|---|---|---|
| `(vt-sum int64 :dtype :bool)` | `unsupported output dtype (UNSIGNED-BYTE 8)` | `vt-sum: 不支持「:int64 输入 → :bool 结果」…请先按数值归约，再 (vt-astype 结果 :bool)` |
| `(vt-var b :dtype :bool)` | `vt--: 布尔张量不支持减法/取负` | `vt-var: 方差结果不能是 :bool（…numpy 同样拒绝）` |
| `(vt-std b :dtype :bool)` | 同上（透出 `vt--`） | `vt-std: …` |
| `(vt-nanstd b :dtype :bool)` | **静默返回 `(0 1 1 0)`** | `vt-nanstd: …` |
| `(vt-norm b :dtype :bool)` | `vt-sum: 不支持「:float64 输入 → :bool 结果」` | `vt-norm: 范数结果不能是 :bool` |
| `(vt-standardize b :dtype :bool)` | `vt--: 布尔张量不支持减法/取负` | `vt-standardize: 标准化结果必须为浮点 dtype，收到 :bool` |

还有一处不属于 bool、但同属「报错说不到点上」的，顺手一起修了：

```
(vt-slice a :start 0 :stop 2)
  =>  vt-slice: 索引数 4 超过秩 1        （把四个关键字当成四个位置索引）
现在
  =>  vt-slice: 切片说明必须是**列表**，不是关键字（收到 :START 之类）。
      写法：(vt-slice vt '(start stop &optional step))
```

`vt-slice` 的 spec 是**列表**（`'(0 2)` 取 `[0:2]`、`'(nil nil -1)` 反转），
而 numpy / Clojure 迁过来的第一直觉都是 `:start/:stop/:step`。原消息里的
「索引数 4」对用户毫无信息量 —— 它数的是四个参数，用户只写了两个「东西」。

两件事值得单独记：

1. `(UNSIGNED-BYTE 8)` 是**物理 Lisp 元素类型**，用户手里写的是 `:dtype :bool`。
   两边对不上，报错等于没报 —— 用户只会以为自己传错了 dtype。
2. `vt-nanstd` 那条是**成对不一致**（§17.10.12 的轴）：`vt-std` 拒绝 `:bool`，
   `vt-nanstd` 却接受并把 `0.5` 经 `%bool-flag` 压成 1，静默给出「标准差
   是否为零」这种看着像答案的东西。numpy 两个都抛 `TypeError`。
   同族里 `vt-nanvar` 一并补上。

##### 三、顺带修的非布尔缺陷：`correlate` / `convolve` 的 `:valid` 与 `:same`

这是这一遍最值钱的一处，也是**「偏差表自我固化」的第四个实例**。

`vt-correlate` 的 docstring 一直写着「三种模式本身与 numpy 逐值一致
（:full/:valid/:same 均已对齐）」。实测：

| (n, m) | mode | 修复前 | 现在 = numpy |
|---|---|---|---|
| (2, 7) | `:valid` | 形状 `(0)`（空） | `(2 1 2 1 2 1)` |
| (3, 6) | `:valid` | 形状 `(0)` | 4 个值 |
| (2, 7) | `:same` | `(2 1 2 1 2 1 2)` | 同（原本就对） |
| (2, 7) | convolve `:same` | `(2 1 2 1 2 1 2)` | `(1 2 1 2 1 2 1)` |

- `:valid` 的长度写的是 `(max 0 (1+ (- n m)))` —— n 是第一个序列、m 是核。
  **n < m 时是负数，被 clamp 成 0**，于是「短序列配长核」直接得到空张量。
  numpy 的规则是 `max(n,m) - min(n,m) + 1`，与谁长谁短无关。
  起点 `offset = m-1` 同理只在 n ≥ m 时正确，应为 `min(n,m) - 1`。
- `:same` 的起点写的是 `(floor (- full-len (max n m)) 2)`。`|n-m|` 为偶数时
  它与 numpy 一致，为奇数时**方向相反**（向下取整 vs numpy 的下半区取大）。
  实测 1..60 共 400 组随机 (n,m) 拟合出两条公式：
  `correlate: min((m-1)/2, n/2)`、`convolve: min((m-1)/2, (n-1)/2)`。

第二条公式说明 **numpy 的两个入口自己就不一致**：`np.convolve(a,v,'same')`
不等于 `np.correlate(a, v[::-1], 'same')`，偶数 n + 长核时差一个位置。
所以 `vt-convolve` 不能简单委托 `vt-correlate`，要传一个内部的
`same-anchor`（`:v` / `:a`）区分。复数路径 `%CX-CORRELATE` 同样处理。

验收：实数 25×25×3×2 = 3750 组、复数 20×20×3×2 = 2400 组，
与 numpy 逐值零失配。

##### 四、确认无事的部分（写下来免得下次重做）

- **轴矩阵值层面**：`sum`/`any`/`all`/`count-nonzero`/`prod`/`amax`/`amin`
  × `axis ∈ {0,1,-1,-2,(0,1),(1,2),(0,2)}` × `keepdims` 共 82 项，
  与 numpy 逐值一致（脏字节是 dtype 层面的问题，值本身没错）。
- **`argmax`/`argmin` 接受元组轴**：numpy 直接 `TypeError`（axis 必须是
  int），本库给归约区内的线性下标。属**更宽松**，语义自洽（已核对
  `argmax(b3, axis=(0,1))` 给 `(0 1 0)`），保留。
- **`mean`/`nanmean` 的 `:dtype :bool`**：numpy 给 `[True True True True]`，
  本库同值（走 `vt-sum` 的出口收敛）。
- **`prod` 的 `:dtype :bool`**：`(1 0 0 1)`，与 numpy 的「任一为 0 则 0」一致。

##### 五、有意偏差

| 表达式 | clvt | numpy | 理由 |
|---|---|---|---|
| `(vt-sum int64数组 :dtype :bool)` | 报错（附 `vt-astype` 建议） | `[True True True]` | 布尔结果只从布尔输入直接归约。要「非零判定」先算数值再 `vt-astype` 更清楚；支持它需要给内核分派加第三条路径（任意 in-et × 小整型 out），风险大于收益 |
| `(vt-std b :dtype :int8)` | `(0 0 0 0)` | `UFuncTypeError`（sqrt 不能输出 int8） | 本库按 int8 截断（与 `var` 的 int8 行为一致），numpy 在 sqrt 处才炸。更宽松 |

---

#### 17.10.16 第八遍：`(f x … :dtype :bool)` 究竟等于什么

前七遍都把 `:dtype` 当**输入**用：我要求布尔输出，你给不给、给得对不对。
这一遍把它当**断言**来验，判据只有一条：

> `(f x … :dtype :bool)` 必须逐值等于 `(vt-astype (f x …) :bool)`

这不是新加的约定 —— `vt-map` 的 `(lambda (x) 5) :dtype :bool` 早就返回 `1`，
`vt-linspace 0 1 5 :dtype :bool` 早就返回 `(0 1 1 1 1)`（G/N 两组测的就是它们）。
既然「先按自然 dtype 算，再 astype」和「直接要 bool」在文档里是同一件事，
两者就必须相等。

用 `sb-introspect` 筛出 **139 个接受 `:dtype` 的导出函数**，各试 1/2/3 个
布尔实参，得到 144 个可比对样本：**37 个不等**，落在 35 个函数上。

##### 一、`the` 截断：0.84 变成 0，2.718 变成 2

| 表达式 | 修复前 | 现在 = astype |
|---|---|---|
| `(vt-abs #(0.5 1.5 2.0 0.25 3.0 0.75) :dtype :bool)` | 字节 `0 1 2 0 3 0`（真值 `0 1 1 0 1 0`） | `1 1 1 1 1 1` |
| `(vt-cos b :dtype :bool)`，b 为 2×3 布尔 | `(0 1 0 1 0 0)`（cos(1)=0.54 截成 0） | `(1 1 1 1 1 1)` |
| `(vt-exp b :dtype :bool)` | 字节 `2`（越界） | 字节 `1` |
| `(vt-log x :dtype :bool)` | `(0 0 0 0 0 0)` | `(1 1 1 1 1 1)` |

两类后果都出现过：截断后落在 0 的**真值反了**（0.84 → NIL），落在 ≥2 的
成了**脏字节**（2.718 → 2，破 §17.1）。

根因在 `vt-fast-map` 的派发：`dtype-check` 让「输入 dtype = 结果 dtype」的
调用进内联循环，`maskable` 又让 `:bool` 进掩码循环。两条内核都按元素类型
特化，写 `(unsigned-byte 8)` 时只有一句 `(the (unsigned-byte 8) v)` ——
那是**截断**；而 `vt-map` 的写回经 `vt-cast-fun` → `%bool-flag`，是**真值化**。
同一语义两条实现，错的那条恰好是默认路径。

修法不是给十几个内核各加一个 `if`，而是把 `:bool` 从这两条快路径里摘出去
（`dtype-check` 加 `(not (eq (vt-dtype res) :bool))`、`maskable` 去掉 `:bool`），
让布尔结果一律回落 `vt-map` 分支 —— 两条路合一条。代价只有布尔输出的映射
少一次内联；`fastmap-invariant-test` 的计时上界仍通过（float64 路径未受影响，
vt-sin 370ms ≈ 裸 vt-fast-map 370ms）。

##### 二、`%float-map` 把**输入**先降成目标 dtype

```lisp
(vt-fast-map op (%coerce-float-input IN %DT) :dtype %DT)
```

目标 `DT` 是 `:bool` 时，这一步先把 0.5 变成 1、-0.5 变成 1，再取对数 ——
算的是 `log(0/1)` 而不是 `log(x)`。`(vt-log x :dtype :bool)` 因此给出全 0。

修法：`%float-map` / `%float-map-fn` / `%float-map-lambda` 三条入口在
`DT = :bool` 时改走 `%FLOAT-MAP-BOOL` —— 按**自然浮点精度**（float32 进
float32 出）算完，再 `%to-bool`。注意 `%float-map` 的展开形状必须保持
`(if <FAST> <SLOW>)`：`fastmap-invariant-test` 按这个形状取快分支做静态检查，
布尔分流只能挂在 `<SLOW>` 里。

##### 三、累积族的第二条实现用了 `%cast-to`

`%cumulative`（cumsum / cumprod）写回是 `(vt-cast cum final-dtype)`，对；
`%e3-cumulative-skip-nan`（nancumsum / nancumprod）的展平路径走
`%e3-store-double-vector`，用 `(setf (aref dd ptr) (%cast-to lt v))` ——
`(unsigned-byte 8)` 那一支同样是截断：

```
(vt-nancumsum #(0.5 -0.5 0.25) :dtype :bool)   修复前 (0 0 0)   现在 (1 0 1)
```

修法：`%e3-store-double-vector` 里给「元素类型是 `(unsigned-byte 8)` 且
dtype 是 `:bool`」单列一支真值化写回。`(unsigned-byte 8)` 同时承载 `:uint8`
与 `:bool`，两者写回语义相反（uint8 截断、bool 真值化）—— 这是 §17 反复
出现的那个根因。

##### 四、`vt-negative` 的 `(and (null dtype) …)`

```lisp
(when (and (null dtype) (eq (vt-dtype (ensure-vt vt)) :bool)) (error …))
```

本意是给 `:dtype :int8` 留一条出口，但它把 `:dtype :bool` 也放行了：

| 调用 | 修复前 | 现在 |
|---|---|---|
| `(vt-negative b)` | 报错 | 报错 |
| `(vt-negative b :dtype :bool)` | dtype 说 BOOL，字节 **255**（-1 落进 unsigned-byte 8） | 报错 |
| `(vt-negative b :dtype :int8)` | `(-1 0 -1)` | `(-1 0 -1)`（保留） |

条件改成 `(or (null dtype) (eq dtype :bool))`：只要**结果 dtype 仍是布尔**
就拒绝，`:dtype :int8` 的出口不变。

##### 五、顺带：`&OPTIONAL` 吞掉 `&KEY`

```
(vt-clip a :dtype :bool)
  =>  :DTYPE fell through ETYPECASE expression. Wanted one of (VT NUMBER SEQUENCE).
```

CL 的 `&OPTIONAL` 不认关键字，`&KEY` 只解析位置参数耗尽之后的尾巴。
`vt-clip` 的签名是 `(vt &optional min-val max-val &key out dtype)`，于是
`:dtype` 成了 MIN-VAL 的**值**，一路走到 `%op-clip` 的 etypecase。

MIN-VAL 只可能是数值/nil，出现关键字就一定是被吞的 `&KEY` 起点 ——
按这个判据把关键字原样交回 `&KEY` 位置重新分派即可（`vt-view` 对
`(vt-view a :dtype dt)` 用的是同一条思路）。`vt-clamp` 转调 `vt-clip`，
一并修好；「只给下限」的 numpy 报错语义不受影响。

同类签名的还有 `vt-where`（`(vt-where c :dtype :bool)` 同理会把 `:dtype`
当 X）—— 但 numpy 的 `where` 本就不允许单参数形式配 `dtype`，故未改。

##### 六、确认无事的部分（写下来免得下次重做）

这一遍先试的是**别名 / 共享缓冲**轴，三条都干净：

- **`:out` 与输入别名**：`vt-add` / `vt-mul` × {自别名, 反向视图} ×
  `{:bool, :int64, :float64}` 全部正确 —— `%vt-views-overlap-p` 命中后退回
  `vt-map` 的 `vt-out-snapshot`，`vt-copy-into` 自己也有快照；
  `vt-check-out` 还会拒绝只读广播视图。
- **输入改写扫描**：60+ 函数 × 3 dtype × {1 维, 2 维}（布尔覆盖 106/134 次
  调用成功，其余是布尔本来就不接受的算子）：0 处改写输入、0 处脏字节。
- **布局等价性**：一元 / 二元 / 归约 × {连续, 切片, 转置, 翻转, 混合} 全一致。

##### 七、有意偏差

| 表达式 | clvt | numpy | 理由 |
|---|---|---|---|
| `(vt-exp b :dtype :bool)` | `(1 1 1 1 1 1)` | `TypeError: No loop matching …` | numpy 的 ufunc 没有 bool 输出循环就直接拒绝；本库允许「先算再 astype」的等价写法，但结果必须等于 astype（本节的判据） |
| `(vt-negative b :dtype :bool)` | 报错 | `TypeError` | 与 numpy 一致；`:dtype :int8` 是本库多给的出口 |

#### 17.10.17 第九遍：审「尺子」本身 —— 混型提升与三条转换入口

第八遍用 `(vt-astype (f x) :bool)` 当判据验了 144 次。这一遍反过来审那把尺子，
并顺着「提升 / 转换」这条线铺开：

- **二元算子矩阵**：11 个 dtype × 11 个 dtype × 17 个算子 → **2021 组**，
  `(dtype, 值)` 与 numpy 2.3.5 逐格一致，0 处不符（73 处报错全是两边都拒绝的
  复数 floor-div/mod 与布尔减法）。
- **提升表**：`vt-promote-type` 的二元 121 格、三元 **286 组** 全对。
- **三条转换入口**：`vt-cast`（标量）/ `vt-astype`（数组）/ `vt-map … :dtype`
  × 11×11×全部特殊值 → **605 组**，三者逐位一致。

矩阵干净的地方说明地被压过了；出问题的是**同一语义的多条实现**。这一遍
共修 5 处，全部属于这一类。

##### 一、`vt-/` 把复数降成 `:float64`

`vt-/` 的 `FLOAT-TARGET` 分支按 true_divide 的「除法恒浮点」规则，一见整型
操作数就把目标改写成 `:float64`：

```lisp
((some dt-int-p tensor-dts) (if (every … '(:float32 :int8 …)) :float32 :float64))
```

`(complex128, int8)` 命中了这一支（`int8` 是整型），于是结果 dtype 被定成
`:float64`，虚部写进实数数组：

```
(vt-/ #(1+2i 4+0i) int8数组)
  → TYPE-ERROR: The value #C(0.5d0 1.0d0) is not of type REAL
```

**24 个 (复数, 整型) 组合两个方向全崩**。而 `complex ⊔ float`、`complex ⊔ complex`、
`complex ⊔ 标量整数` 都正常 —— 差别只在「有没有一个整型**张量**」。

修法：复数参与时不做这条改写，交给 `vt-promote-type`（它已按「实部分量参与
普通提升、再映射回复数」算好 `c128 ⊔ int8 → c128`、`c64 ⊔ int8 → c64`）。
判据看**全部操作数**而不是 `tensor-dts`（后者过滤掉弱标量）。

> `#c(2 0)` 在 CL 里被规范成整数 `2`（`(complexp #c(2 0))` → `NIL`），
> 所以 `(vt-/ int8数组 #c(2 0))` 本就是 `int8数组 / 2` → `:float64`（NEP 50）。
> 这不是缺陷，是 Lisp 无法表示「零虚部的复数标量」。

##### 二、`vt-map` 的写回不认复数 → 实数

```lisp
(the double-float (coerce v 'double-float))   ; v = #C(1.0d0 2.0d0) → TYPE-ERROR
```

`(vt-map #'identity 复数张量 :dtype :float64)` 崩，而 `(vt-astype c :float64)`
与 `(vt-cast #C(1 2) :float64)` 都正常取实部（§16.5）。连同 `(vt-conj c :dtype
:float64)` 一起是 60 组。修法是加 `%REAL-WRAP`：只在「目标实数 + 有复数输入」
时把函数包一层 `realpart`，**热循环一字未动**（不在写回处加每元素判断）。

##### 三、标量入口不认 Lisp 布尔

```lisp
(vt-cast t :float64)            → TYPE-ERROR: The value T is not of type REAL
(vt-fill (vt-zeros '(2) :dtype :float64) t)  → 同上
(setf (vt-ref f 0) t)、 (vt-full '(3) t :dtype :float64)、 (vt-* f t) → 同上
```

`%bool-flag` 认 `T`/`NIL`，但只在**目标**是 `:bool` 时被调用。数值目标是
另一条路（`coerce`），于是数组入口 `(vt-astype bool张量 :float64)` 给 `1.0`、
标量入口直接崩。numpy 一律接受（`np.zeros(3)[0] = True` → `1.0`）。

修：`vt-cast` 在剥虚部之前把 `T`/`NIL` 收敛成 `1`/`0`；`(setf vt-ref)` 只在
值不是普通数（`T`/`NIL`/复数）时才绕道 `vt-cast`，数值标量仍走原 `coerce`，
逐元素写入不变慢。

顺带：`(setf (vt-ref float数组 0) #c(1 2))` 现在取实部而不是报错。
越界整数写入**仍然报错**（`(setf (vt-ref int8数组 0) 300)`）——
这与 numpy 2.x 的 `OverflowError: Python integer 300 out of bounds for int8`
一致，而 `astype` 走回绕（`np.array([300]).astype(int8)` → `44`），两边都是
numpy 自己的行为。

##### 四、一维构造快路径的 `(type real x)`

`vt-from-sequence` 的一维快路径把转换算子内联，并对元素声明 `(type real x)`。
`T` 不是 `REAL`，于是：

```
(vt-from-sequence '(t t) :dtype :float64)
  → The value T is not of type REAL when binding X
```

原代码只让 **bool 模式**绕开这条快路径（因为布尔叶子全是 `T`/`NIL`），漏了
「目标 dtype 是数值、序列里却写了 `T`」这一组合。修法是把「含复数」那个
判据扩成 `%FILL1D-INELIGIBLE-P`（复数 **或** `T`），仍是一次扫描。

> `NIL` 叶子不需要判：非 bool 模式下它是空子序列，`infer-shape` 早已以
> 「不规则嵌套」拒掉 —— `(vt-from-sequence '(t nil) :dtype :float64)` 仍报错，
> 这是 §17.8 的分层契约，不是缺陷。

##### 五、多元提升：逐对折叠 vs 一次看全部

`%op-final-dtype` 对多个张量是**逐对折叠**：

```lisp
(setf base (if base (vt-promote-type base d) d))
```

二元时与「一次看全部」等价，三元起会分叉：

| 三个 dtype | 折叠 | 一次（= numpy = `vt-promote-type` = `vt-concatenate`） |
|---|---|---|
| `int16, uint16, float32` | `int16⊔uint16 = int32` → 超 24 位 → `float64` | `float32` |
| `int16, uint16, complex64` | 同上 → `complex128` | `complex64` |

即 `(vt-+ a b c)` 的 dtype 与库自己的 `vt-promote-type`、`vt-concatenate`
都对不上。改成先收集再 `(apply #'vt-promote-type …)`。8 组受影响。

##### 六、这一遍的判据

> **同一语义的每一条入口都必须给出同一个答案**，而「同一语义」不能靠
> docstring 声明，得靠**枚举**：同一件事有几个入口（`vt-cast` / `vt-astype` /
> `vt-map`），同一套规则有几处实现（`%op-final-dtype` / `vt-promote-type` /
> `vt-concatenate` / `vt-stack`），逐个对。

第二个教训是**矩阵要打到边界值**：`int16, uint16, float32` 这种「有符号 +
无符号 + 窄浮点」的组合才暴露折叠与一次提升的差别，`float64` 参与时两者
恒等，测了也看不出来。

##### 七、有意偏差（本节新增）

| 表达式 | clvt | numpy | 理由 |
|---|---|---|---|
| `(vt-/ 复数数组 整型数组)` | `:complex128` | `complex128` | 一致（本节修复前是 `TYPE-ERROR`） |
| `(setf (vt-ref int8数组 0) 300)` | 报错 | `OverflowError` | 一致 |
| `(vt-from-sequence '(t nil) :dtype :float64)` | 「不规则嵌套」 | `[1., nan]` 之类 | Lisp 的 `NIL` 既是假又是空表（§17.8），只能靠「bool 模式」消歧 |
| `(vt-+ 复数数组 x :dtype :float64)` | 静默取实部 | `UFuncTypeError: Cannot cast …` | 与 §16.5 的「本库不产生 ComplexWarning」同一政策：显式要走实数时一律取实部，不因入口不同而有的报错有的静默 |

#### 17.10.18 第十遍：「同一件事有几个入口」要枚举成表

第九遍学会了问「同一语义有几条实现」，但那还是靠临时想起。这一遍把问题
**制度化**：先把 375 个导出可调用符号按族分组（`nan` 前缀 12 对、词干共享 15 组），
再对每一族跑「族内必须一致」，最后用一条不带 numpy 的判据扫全库的 `:out`
契约。七处缺陷（十种表现），无一例外都是**同一族里两个成员给出不同答案**。

##### 一、`nan` 族的形状与精度

| 表达式 | 修复前 | 现在 = numpy |
|---|---|---|
| `(vt-nancumsum 2×4 矩阵)` | 形状 `(2 4)`、值已展平 | `(8)`，`(1 3 6 10 15 21 28 36)` |
| `(vt-nancumsum int64 #(2^60 1 -2^60))` | `(2^60 2^60 0)` | `(2^60 2^60+1 1)` |

第一条：`%e3-cumulative-skip-nan` 的 docstring 与 `%vt-cumulative` 都写着
「AXIS 为 nil 时展平」，但前者无论如何都沿用输入形状 —— 把展平后的累积值
按 `(2 4)` 摆回去，值对、形状说谎。

第二条更隐蔽：该路径无条件 `(vt-astype input :float64)`，于是 int64 超过
2^53 就在 double 里丢低位。而整数里**根本没有 NaN 可跳**，这条「NaN 感知」
路径对整数应当与普通 `cumsum` 完全等价。修法是整数/布尔输入直接委托
`vt-cumulative`（顺带保住精确性）。

##### 二、复数 `matmul` / `dot` 的三层分叉

| 表达式 | 修复前 | 现在 = numpy |
|---|---|---|
| `(vt-matmul 2×2 向量)`（复数） | `(2 1)` | `(2)` |
| `(vt-matmul 3×2×2 向量)`（复数） | 报错 | `(3 2)` |
| `(vt-matmul 实矩阵 复矩阵)` | 报错 `非复数缓冲区` | `complex128` |
| `(vt-matmul c64 c128)` | 报错 | `complex128` |
| `(vt-matmul 复 复 :out float32)` | **静默取实部 + 降精度** | 报错 |

第三、四条是这几处里最重的：`%cx-matmul-2d` / `%cx-batched-matmul` 都按
`(vt-dtype a)` 分配结果缓冲。混型时公共 dtype 是复数、缓冲却是实数，
内核第一个元素就写不进去。numpy 的做法是**先提升两个操作数到公共 dtype
再算**，照做即可。

第五条的根因是 `:out` 走的是 `vt-copy-into`（能装就装）而不是
`vt-check-out`（形状与 dtype 精确相等）。实数路径一直用后者，所以同一个
overload 的两种输入类型，一个报错一个静默给错值。

##### 三、`vt-divmod` 与 `vt-div`

| 表达式 | 修复前 | 现在 = numpy |
|---|---|---|
| `(vt-divmod a b :out o)` | 静默忽略（docstring 却说会报错） | 报错 |
| `(vt-div #(1 -1 0) #(0 0 0))` | `(-Inf -Inf NaN)` | `(Inf -Inf NaN)` |
| `divmod` 的商 | `(NaN NaN NaN)` | 与 `vt-div` 逐位相同 |

`%floor-div-op` 用 `(or (and (plusp x) (plusp y)) (and (minusp x) (minusp y)))`
定符号，而 `(plusp 0.0d0)` 与 `(minusp 0.0d0)` 都是 `NIL` —— 分母为零时
两个分支恒假，于是 **恒返回 `-Inf`**。分母的符号必须走 `float-sign`
（`(float-sign -0.0d0)` 是 `-1.0d0`），`plusp`/`minusp` 对 `±0` 一律给 `NIL`。

`vt-divmod` 原先自带一份 floor 除法副本，两处就此分叉（商恒 `NaN`）。修法
不是对齐副本，而是**删掉副本**：商直接调 `vt-div`，与余数调 `vt-mod` 对称。

##### 四、`:out` 契约的全库扫描

判据：对每个带 `:out` 的导出函数，先取无 `:out` 的结果，再分别喂
「dtype 不符」与「形状不符」的 `out`，**两者都必须报错**。

274 次探测里只有 `vt-divmod` 一处不报 —— 它的 docstring 明写「暂不支持
`:out`（传 out 将报错）」，实现却是 `(declare (ignore out))`。这类
「docstring 声称的行为」与「代码的行为」不符，比不写还糟：用户读到的是承诺。

##### 五、留下的判据

> 族不是想出来的，是**枚举**出来的：`sb-introspect` 拿全部 `lambda-list`，
> 按签名分组、按词干分组，再逐族找那个「必须成立的等式」。
> 找不到等式时，退化输入（全 NaN / 空 / 0 除 / ±0）就是等式。

| 项 | 结果 |
|---|---|
| 探针 | 12 对 nan 变体 × 4 dtype × 7 参数 / 6 布局 / 3 维 / 168 组 numpy 锚 |
| 探针 | `:out` 契约 274 次（375 个导出可调用符号中 160 个接受 `:out`） |
| 探针 | 跨族恒等式 55 组（`cumsum` 末位 = `sum`、`std` = √`var`、`all` = `prod` 等） |
| 修复 | 7 处代码点（10 种表现：2 处 docstring 与实现相反） |
| 断言 | bool-dtype-test +64（R 组 8 节） |

##### 六、有意偏差（本节新增）

| 表达式 | clvt | numpy | 理由 |
|---|---|---|---|
| `(vt-var x :ddof k)`（k 使分母为 0 且分子非 0） | `NaN` | `inf` | 本库不区分「0/0」与「x/0」，退化到零分母一律 `NaN`；numpy 另发 RuntimeWarning |
| `(vt-median x :dtype :float32)` | 报错「未知关键字」 | 支持 | `vt-nanmedian` 有 `:dtype`/`:out` 而 `vt-median` 没有，属 API 不对称（待补） |

> 「序列 `q` 尚未实现」一条已从本表移出：`(vt-quantile x '(0.25 0.5))` 现在返回
> 前置轴结果，与 numpy 一致（§17.10.20）。

#### 17.10.19 第十一遍：打印是第四层视图，「示例」是承诺

前十遍的尺子一直是 numpy，或者族内两个成员互标。这一遍换成一个不依赖外部
的东西：**每个 docstring 里写着 `=>` 的那一行**。366 个带 docstring 的导出
函数里有 57 行 `=>`，其中 45 行是可执行表达式。逐条照抄跑一遍，撞出 8 处不实
—— 3 处是打印层缺陷，5 处是示例早已与实现脱节。

##### 一、打印层：三条规则

`%FORMAT-REAL-PART` 原来只有一条规则：`~,6f` 定精度、去尾零、保底一位小数。
这条规则在两处失效：

| 值 | 修复前 | 现在 |
|---|---|---|
| `(vt-spacing 1.0)` 的值 `2.22e-16` | `0.0` | `2.220446e-16` |
| `1.0d-300` | `0.0` | `1.0e-300` |
| NaN | `#<DOUBLE-FLOAT quiet NaN>` | `nan` |
| `+Inf` | `#.SB-EXT:DOUBLE-FLOAT-POSITIVE-INFINITY` | `inf` |
| `1.0d308` | 309 位数字 | `1.797693e+308` |

前两行是**相对误差 100%**：定点 6 位小数把 `|x| < 5e-7` 的非零值舍成 0。
第三、四行更糟 —— 它们把 SBCL 的**实现内部表示**印进了用户输出：
`#<...>` 是 print-not-readable 对象，`#.SB-EXT:...` 是读时求值宏，
两者都不该出现在张量打印里。

新规则三条：

1. NaN / ±Inf 单独走，印 `nan` / `inf` / `-inf`；
2. `|x| < 10^-P` 或 `|x| >= 10^(P+3)` 用科学计数（P = `*VT-PRINT-PRECISION*`，默认 6）；
3. 其余仍用定点。

阈值为什么按量级而不是按「相对误差 > 1e-6」：相对误差判据会误伤
`0.5000004`（印成 `0.5` 完全合理，但相对误差 8e-7）。按量级判只影响
「这一位本来就说不清」的区间：`1e-6` 仍是 `0.000001`（这一位是准的），
`1.3e-6` 已切科学计数（定点会把它压成 `0.000001`）。

**这里不承诺「可读回」。** CL 的 `1.0e-300` 默认读回 single-float，下溢成 0；
要读回 double 得写 `1.0d-300`。把指数标记改成 `d` 能解决读回，却会把
「double」写死进输出（同一个数在 single/double 下印得不一样），与「dtype 另有
位置标注」的设计冲突。所以这一层的承诺降一级：**不丢量级、不泄漏实现**，
精度由 P 位有效数字负责。§17.8 的「可读回」是布尔专有的，不适用于浮点。

##### 二、`(minusp <nan>)`：第二次踩

`%FORMAT-COMPLEX-NUMBER` 用 `(minusp im)` 决定虚部符号。而
`(minusp <nan>)` 抛 `FLOATING-POINT-INVALID-OPERATION` ——
**含 NaN 的复数根本打印不出来**，直接掉进 debugger。而它的 docstring
正举着 `nan+nanj` 这个例子。

§17.10.18 里 `%FLOOR-DIV-OP` 踩的是同一对函数的另一面（`(plusp 0.0d0)` 恒 NIL）。
两条合起来是一条规则：

> 判浮点的符号或大小，`PLUSP` / `MINUSP` 对 ±0 一律给 NIL、对 NaN 直接抛陷阱。
> 要么先过滤 NaN，要么用 `FLOAT-SIGN`。

修法是加 `%NEG-SIGN-P`：NaN 一律当非负，于是印 `nan+nanj` 而不是 `nan-nanj`。

##### 三、`(vector #c(1d0 2d0))` 的虚部：同一个坑的第二个入口

```
(vt-from-array (vector (complex 1.0d0 2.0d0)))  →  :float64，虚部消失
```

而 `vt-from-array` 的 docstring 里明明写着：

> 复数数组（v0.5.x 修复）：旧实现的 cond 里没有 complex 分支，
> `(complex double-float)` 一路落到最后的 `(t :float64)`，于是
> `(vt-from-array <复数数组>)` 把虚部静默丢掉。

那次修复加的是 `(subtypep cl-etype '(complex double-float))` 分支 —— 只对
**声明了元素类型**的数组有效。`(vector …)` 和 `#(…)` 造出来的是
`simple-vector`，`array-element-type` 是 `T`，所有 `subtypep` 判定全假，
于是又落回 `(t :float64)`。**同一个 bug、同一条注释、换了个入口。**

修法：元素类型为 `T` 时按值扫一遍，只有复数改用 `:complex128`。非复数仍维持
`:float64`（按值推断整数会波及全库，见下方偏差表）。

##### 四、文档修正

| 函数 | 原示例 | 问题 |
|---|---|---|
| `vt-array-equal` | `'(1 nan)` | `nan` 不是可求值的名字，照抄报 `The value NAN is not of type REAL` |
| `vt-nan-to-num` | `'(nan inf -inf 1.0)` | 同上 |
| `vt-isin` | `[False True False True]` | 布尔印 `T`/`NIL`（§17.8） |
| `vt-interp` | `=> 25.0d0` | 实际返回 1 元素张量 —— docstring 自己也写着「输出恒 1D」 |
| `vt-vdot` | `(vt-from-array #2A(…)) => 70` | 实际 `70.0`（`from-array` 对 `T` 元素类型给 float64） |
| `vt-compress` | `(vt-fromiter '(0 1 2 3 4))` | fromiter 缺省 float64，结果是 `[0.0 2.0 4.0]` |
| `vt-nextafter` / `vt-spacing` | `=> 1.0000000000000002` | 打印只有 6 位有效数字：值对，写法误导 |

前两条的根子是**库里没有公开的 nan/inf 构造入口**：`vt-get-nan` /
`vt-get-pos-inf` / `vt-get-neg-inf` 一直存在，却压在内部包里。numpy 有
`np.nan` / `np.inf`，本库文档也照 numpy 的写法举例，只是那个 `nan` 从来没人
定义过。这一遍把三个函数导出（**API 变更**，纯新增，不破坏兼容）。

##### 五、留下的判据

> docstring 里的 `=>` 是**断言**，不是装饰。
> 写不出可执行的示例，就把「值是什么」和「打印是什么」分开写清；
> 两者不一致时，至少要说清哪一个才是给人看的。

| 项 | 结果 |
|---|---|
| docstring 逐条执行 | 45 条可执行示例，修前 8 处不实 |
| 打印量级样本 | 24 个，修前 4 个失真（3 个 100% 误差、1 个读端问题） |
| 新增断言 | S 组 38 条 |
| 修复 | 4 处代码（打印 3 + 构造 1）、7 处文档、3 个导出 |

##### 六、偏差表（本节新增）

| 表达式 | clvt | numpy | 理由 |
|---|---|---|---|
| `(vt-from-array #(1 2 3))` | `:float64` | `int64` | 元素类型 `T` 时不做按值整数推断。改成 `int64` 会波及全库既有测试与用户代码，留作独立变更 |
| `(vt-from-sequence '(1 2))` | `:float64` | `int64` | 同上，已在 §17 的「dtype 默认」条目里 |
| `(vt-asarray '(1 2))` | `:int64` | `int64` | 唯一与 numpy 一致的构造入口 —— 要整数就显式用它 |
| `(vt-fromiter '(1 2))` | `:float64` | 必须显式给 dtype | 文档化的 v0.4 兼容默认 |
| `(format nil "~a" 1.0d-300)` | `1.0e-300` | `1e-300` | 尾随 `.0` 是本库「浮点至少一位小数」的既有风格 |

#### 17.10.20 形状、轴元组、序列参数是三件独立的事

前三节的尺子是「族内互标」和「docstring 自证」。这一节换成不挑对象的一把：
**差分模糊测试** —— Lisp 侧 `vt-save :as :npy` 落盘、Python 侧 `np.load` 逐位
比对，八个功能域（归约/操纵/算术/线代/边界/复数/布尔/杂项）共约 1800 条随机
用例。它们撞出的缺陷有一个共同形状：

> 函数的**返回形状**由某个具体数值（右端项个数、参数是否为标量）决定，
> 而不是由**输入的秩和显式参数**决定；或者某个参数有两种合法形态，
> 实现只写了其中一种。

| # | 函数 | 触发 | 修复前 | 现在 |
|---|---|---|---|---|
| 1 | `vt-solve` / `vt-inv` | `b` 形状 `(n,1)` 或 1×1 方程组 | 用 `(= nrhs 1)` 判返回秩：1×1 的 `vt-inv` 返回 shape `(1)`，`b` 的 `(n,1)` 被压成 `(n)` | 由 `b` 的**秩**决定（`b-vector-p`），`(n,1)` 恒返回 `(n,1)`，与 `np.linalg.solve` 同形 |
| 2 | `vt-ptp` | 全归约 | `(- (vt-item …) (vt-item …))` → Lisp 标量，且无 `:keepdims` | 恒走 `vt--`：全归约给 0 维张量（§6.5），并支持 `:keepdims` |
| 3 | `vt-nanmedian` | `:axis nil` + `:keepdims t` | 返回 0 维（`vt-median` 同期给 `(1 1)`） | 返回全 1 形状，二者一致 |
| 4 | `vt-median` / `vt-nanmedian` | `:axis '(0 1)` | `vt-normalize-axis` 只认整数 → `The value (0 1) is not of type …` | 轴元组先经 `%FOLD-REDUCE-AXES` 折成 `(H,D)` 再复用单轴内核（`%MULTI-AXIS-REDUCE` 同一路径） |
| 5 | `vt-percentile` / `vt-quantile` | 轴元组，以及轴元组 × 分位点序列 | 同上；且与序列 `q` 组合时报的是**内部类型泄漏**（`not of type REAL`） | 两条参数各自独立：序列先沿新首轴堆叠，轴元组再折叠 |
| 6 | `vt-left-shift` / `vt-right-shift` | 第二操作数是张量 | `(ash x shift)` 的 `shift` 是标量形参 → `not of type INTEGER` | 张量走 `vt-map` 逐元素（含广播）；dtype 按混型规则提升 |
| 7 | `vt-histogram` | `:bins` 给边界序列 | 报「必须为正整数」 | 两种语义共用一个入口：正整数 = 箱数，序列 = 边界（≥2 且严格递增） |
| 8 | `vt-sigmoid`/`vt-relu`/… 共 10 处 | — | 同一段 `%cx-reject` 守卫被粘贴两份，第一份是死代码 | 只留一份；行为不变（拒绝复数），由回归网钉住 |
| 9 | `%histogram-bucket`（边界序列的桶定位） | 数据含 `NaN` | `NaN` 与任何数的 `<`/`>` 都是假，两侧过滤全漏过 → 二分一路走到末尾，**落进最后一个箱** | 入口显式判 `%nan-p` 丢弃；`[e0, e_{n-1}]` 之外与 `NaN` 一并不计入（numpy 语义） |

##### 一、「结果秩 = 输入秩 ± 归约掉的轴」，不掺任何数值判断

`(vt-inv (vt-from-sequence '((2d0))))` 的答案是 shape `(1 1)` 的 `((0.5d0))`，
因为输入是 1×1 **矩阵**；判成 `(1)` 就把它降级成了一维数组，后续 `@` 或
广播都会按错的秩解释。修法是把「是不是向量」定义在**输入的秩**上：

```lisp
(let* ((b-vector-p (= (length b-shape) 1))
       (nrhs (if b-vector-p 1 (second b-shape)))
       …)
  (if b-vector-p (vt-reshape … (list n)) …))
```

同一个坑在 `nrhs` 上出现过第二次：`b` 形状 `(3,1)` 时 `nrhs` 恰为 1，旧代码
因此走了「向量」分支。**用「某维度是否等于 1」代替「秩是几」永远会错**，
因为 1 是合法维度。

##### 二、轴参数的两条形态（整数 / 列表）必须都走通

`vt-normalize-axes` 接受 `nil` / 整数 / 整数列表，但下游的**单轴内核**只接受
整数：`(vt-normalize-axis '(0 1) rank)` 会把整个列表当轴号，报一句内部类型
错误。规则是：

- 列表长度为 1 → 退化成整数，走单轴内核（不折，省一次转置）；
- 列表长度 ≥ 2 → `%FOLD-REDUCE-AXES` 把被归约的轴转置到尾部并合并成
  `(H, D)`，对 `D` 跑单轴内核，再 `vt-reshape` 回目标形状（`keepdims` 已在
  目标形状里）；
- 折叠只用 `transpose` / `reshape`，**不碰 dtype**，所以复数、非连续视图、
  转置输入都自动适用（无需第二条复数分支）。

`vt-sum`/`vt-mean`/`vt-amax`/`vt-amin`/`vt-argmax`/`vt-any`/`vt-all` 早就支持
轴元组，缺的只有 `vt-median`/`vt-percentile`/`vt-quantile` —— 这是「同族不同
能力」，不是 numpy 不许。回归网里有一条交叉断言：**七个归约函数在同一轴元组
上必须给出同一形状**（含 `keepdims`），任何一个退回去就会立刻变红。

##### 三、分位点序列 × 轴元组是正交的两件事

`vt-percentile` 的 `percentile` 参数可以是序列，`axis` 可以是元组，两者
**同时**用是 numpy 支持的：先对每个分位点各算一遍沿新首轴堆叠，再让每次计算
自己处理轴元组。实现顺序必须是「序列展开在前、轴折叠在后」，否则 `q` 会以
列表形态漏进单轴内核。

```lisp
;; numpy: np.percentile(a, [10, 90], axis=(0,1))  -> shape (2, …)
(vt-percentile a '(10 90) :axis '(0 1))            ; => (2 4)
(vt-percentile a '(10 90) :axis '(0 1) :keepdims t) ; => (2 1 1 4)
```

##### 四、`vt-histogram` 的两种 `:bins` 共用一个入口

numpy 的 `bins` 有两种形态，语义完全不同却都叫一个名字，因此必须在入口处
分派而不是在参数校验处拒绝：

| `:bins` | 含义 | counts 长度 | 边界 |
|---|---|---|---|
| 正整数 `k`（缺省 10） | 箱数 | `k` | 数据极值（或 `:range`）等宽切分 |
| 序列（≥2，严格递增） | 箱边界 | `n-1` | 原样返回；`[e0, e_{n-1}]` 之外**不计入** |

桶定位等价 `searchsorted(side='right') - 1`：恰好等于**内部**边界的值归右侧
箱，恰好等于**最后一条**边界的值归最后一个箱。二分条件用 `<`（不是 `<=`），
否则每个内部边界都会把值归到左侧，`[0,2,4]` 上会得到 `[3,1]` 而非 `[2,2]`。

##### 五、守卫重复本身是缺陷

`%cx-reject` 那类守卫被复制粘贴成两份时，第一份是死代码 —— 它不改变行为，
但会让「删掉一份」看起来安全，而两份的文本往往已经不同。清理后必须用一条
**行为断言**兜住：复数仍然被拒绝、实数路径仍在。只删代码不补断言，等于把
一次机械编辑变成一次不可察觉的语义变更。

| 项 | 结果 |
|---|---|
| 差分探针 | 8 个功能域、约 1800 条随机用例（Lisp `vt-save` → numpy 逐位比对） |
| 修复 | 9 组代码点 / 10 类表现（含 1 处死代码清理、1 处 NaN 入箱） |
| 连带修正 | 3 处既有断言的旧期望（`vt-ptp` 返回标量、`vt-solve` `(n,1)` 被压平），2 处示例 |
| 新增断言 | `test/reduce-shape-axis-test.lisp` 145 条（A–I 九组） |

#### 17.10.21 空维度不是「另一种数值」，是「没有数值」

上一节的尺子是随机输入。随机输入有一条天然盲区：**它几乎不会生成 0**。
randint 取到 0 长度的概率是 0，取到 0 值也只是少见的边界。于是「shape 里含 0」
这一个格子从来没被踩过，而它在实现上恰好是最危险的一格。

> 凡是「先跑一轮循环体、再判断要不要继续」的结构 —— 里程计、do-while、
> `advance` 回调 —— 在**总元素数为 0** 时都会先跑那一轮。
> 而那一轮会去 `aref` 一个长度 0 的 buffer。

这不是某一处写错，而是同一个模式在**三个独立内核**里各写了一遍，全都中招：

| # | 函数 | 触发 | 修复前 | 现在 |
|---|---|---|---|---|
| 1 | `vt-cumsum` / `vt-cumprod` | 沿轴 + 元素数 0（如 `(0,3)` 沿 axis 1） | `Invalid index 0 for (SIMPLE-ARRAY DOUBLE-FLOAT (0))` | 进入轴分支前判 `(plusp (vt-size result))`；空结果就是真空 |
| 2 | 同上，**通用 fallback** | 同上，但 dtype 需提升（如 `int8` → `int64`） | 同上（快路径接不走时由它接，同样中招） | 同一保护 |
| 3 | `vt-nancumsum` / `vt-nancumprod` | 同上 | 同上（第三个内核，`%E3-CUMULATIVE-SKIP-NAN`） | 同一保护 |
| 4 | `vt-indices` | 维度里含 0 且**不在首位**（如 `(2,0,3)`） | `cannot index into an empty tensor` | 层大小为 0 时跳过填充；stride 为 0 不再兜成 1 硬写 |
| 5 | `vt-vsplit` / `vt-hsplit` / `vt-dsplit` | 轴长为 0 | 报「不整除」（`(plusp dim)` 先短路了 `(mod dim n)`） | 0 对任何正整数都整除；返回 n 个空段 |
| 6 | `vt-cov` / `vt-corrcoef` | 零观测（`n_obs = 0`） | `FLOATING-POINT-INVALID-OPERATION` / `DIVISION-BY-ZERO` | 屏蔽陷阱，给 nan（§5.5） |

##### 一、「零元素」和「零值」是两件事，但错误消息长得一样

`(vt-cumsum (vt-zeros '(0 3)) :axis 1)` 的报错是 `Invalid index 0 for
(SIMPLE-ARRAY DOUBLE-FLOAT (0))` —— 它读起来像「索引 0 越界」，实际上是
「**没有**索引 0 可读」。这类消息特别容易在 review 里被当成参数问题放过，
因为 `axis 1` 看起来是合法的、`(0 3)` 看起来也是合法形状。

判据很简单：**空张量上，任何「输出也为空」的算子都必须成功并返回空**。
numpy 的规则是「元素数为 0 → 没有 fiber → 什么都不做」，而不是「报错」。

| 情形 | numpy | 本库 |
|---|---|---|
| `np.cumsum(np.zeros((0,3)), axis=1)` | shape `(0,3)` | 同（原先崩） |
| `np.cumsum(np.zeros((3,0)), axis=0)` | shape `(3,0)` | 同（原先崩） |
| `np.max(np.zeros((3,0)), axis=0)` | shape `(0,)`（输出为空，不需要 identity） | 同 |
| `np.max(np.zeros((0,3)), axis=0)` | 报错（输出非空、归约区为空） | 同（消息已明确） |
| `np.vsplit(np.zeros((0,3)), 2)` | 两个 `(0,3)` | 同（原先报不整除） |
| `np.indices((2,0,3))` | shape `(3,2,0,3)` | 同（原先崩） |

##### 二、角分母不许「换个分母凑合」

`vt-cov` 里有一行 `(if (<= denom 0) n-obs denom)`，读起来像防御性编程，
实际是**凭空改分母**。numpy 的写法是：

```python
fact = n_obs - ddof
if fact <= 0: fact = 0.0        # 不是「换成 n_obs」
c *= 1. / fact                  # 于是 1/0 = inf，0*inf = nan
```

两种写法在正常输入上完全一致，只在角上分叉：

| 输入 | numpy | 改成 `n_obs` 的错误答案 |
|---|---|---|
| `cov([5.])`（1 观测，ddof=1） | `nan`（0/0） | `0.0` |
| `cov([1,2], ddof=3)` | `inf`（0.5/0） | `0.25` |

`inf` 和 `nan` 是**信息**：它们告诉调用方「这个数不存在」。换成 0.25 等于
给出一个看起来合理、实际无意义的数 —— 这种错最贵，因为它不会触发任何下游
检查。

同一条在 `vt-corrcoef` 上表现为硬编码：标量方差走 `(vt-full nil 1.0d0)`。
正确的公式始终是 `C / sqrt(C²)`，它对 `C = 1` 给 1、对 `C = 0`（全等输入）
给 nan、对 `C = nan` 给 nan。numpy 的 `corrcoef([5.])` 就是 nan。

##### 三、`num = 0` 是「空采样」，不是「非法参数」

`vt-linspace` / `vt-logspace` / `vt-geomspace` 都把 `num <= 0` 当错误。
但 `num = 0` 在 numpy 里合法：`np.linspace(0, 1, 0)` 返回 `array([], dtype=float64)`。
「在区间上采 0 个点」是一个有意义且常见的请求（批量生成时 n 可能算出 0）。

只有 `num < 0` 才是非法。顺序也不能反：`np.geomspace(0, 10, 0)` **仍然报
`Geometric sequence cannot include zero`** —— 端点的零/同号校验先于数量校验。
把 `num` 判定提到前面会让一个本该报错的输入静默返回空数组。

##### 四、`weights` 让 `bincount` 换的是 dtype，不是语义

`np.bincount(x, weights=w)` 有三条容易漏的规则：

1. 结果恒为 `float64`，**即使 `w` 是整型或布尔**（文档原话：result is stored
   as a float64 array if weights is given）；
2. 负权重合法（加权直方图本来就会有负贡献）；
3. 复数权重**报 `TypeError`**（无法 cast 到 float64）—— 这一条尤其容易「出于
   好意」实现成 complex128，结果给出一个 numpy 给不出的答案。

`vt-random-choice` 的 `:p` 缺的是**同一个参数的另一半形态**：numpy 的 `p` 就是
一个 ndarray，而原实现直接对它走 `length` / `every` / `reduce` 的序列协议，
于是最常见的写法报 `The value #<VT …> is not of type SEQUENCE`。修法是把三种
形态（`nil` / Lisp 序列 / 张量）在入口归一成一个列表 —— 归一之后三条路径
**必须给出同一序列**，这条用同 seed 的等值断言钉住。

##### 五、裸 `assert` 不是错误契约

`(assert (> a 0) (a))` 报的是 `The assertion (> A 0) failed with A = 0.`：
没有函数名、没有字段名、没有「应该是什么」。`ASSERT` 在 SBCL 里是可以被
`*suppress-check-redefinition*` / 优化级别干掉的调试工具，不是给调用方看的
契约。所有对外入口的校验必须走 `(error "vt-xxx: …")`，并且消息里带上
**函数名 + 实际值 + 期望**。random 族原有的 6 处裸 `assert` 已全部改掉。

| 项 | 结果 |
|---|---|
| 换轴 | 随机输入 → **退化形状**（shape 含 0 / 0 维 / 单观测 / `num=0` / 分母 0） |
| 修复 | 11 组代码点 / 13 类表现（3 个累积内核、indices、split 族、cov、corrcoef、3 个构造函数、bincount、choice、错误契约） |
| 旁证（零缺陷） | IO 往返 188 组合、dtype 转换矩阵 121 组合、集合运算 92 条、索引族 35 条、`nn` 族恒等式 40 余条 |
| 新增断言 | `test/degenerate-empty-test.lisp` 331 条（A–J 十组） |

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
| `REFACTOR-NOTES.md` | 保留（历史记录）；新增 v0.4.0 章节记录那次重构 |
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
