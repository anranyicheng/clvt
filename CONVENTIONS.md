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

---

## 5. NaN / Inf 约定（**任务4 核心**）

### 5.1 常量体系重构（删除三个 double 常量）

**现状问题**：`nan.lisp` 定义了
```lisp
(defconstant +vt-float-nan+     +vt-dfloat-nan+)     ; 默认 double
(defconstant +vt-float-pos-inf+ +vt-dfloat-pos-inf+) ; 默认 double
(defconstant +vt-float-neg-inf+ +vt-dfloat-neg-inf+) ; 默认 double
```
这三个名字（`float` 泛指）**实际指代 double**，是误导来源。用户要求删除。

**裁决：⚠️ 需重构 —— 删除这三个常量。**

| 删除 | 替代 |
|---|---|
| `+vt-float-nan+` | `(vt-get-nan dtype)` / `(vt-float-nan &optional dtype)` |
| `+vt-float-pos-inf+` | `(vt-get-pos-inf dtype)` |
| `+vt-float-neg-inf+` | `(vt-get-neg-inf dtype)` |

**保留**（名字已明确精度，无歧义）：
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
| `floor_divide(int, 0)` | `0`（不报错！） | ⚠️ 需对齐 |
| `mod(int, 0)` | `0` | ⚠️ 需对齐 |
| `mod(float, 0)` | `nan` | ✅ |
| `floor_divide(float, 0)` | `inf` | ✅ |
| `mod(-7, 3)` | `2`（符号跟随除数） | ✅ |
| `floor_divide(-7, 3)` | `-3` | ✅ |
| `nan → int` cast | `INT_MIN` 或 `0`（实现定义，本库已约定 **0**） | 保留本库约定（NumPy 未定义，属自由区） |

> **注意**：本库 `vt-mod`/`vt-rem` 的"零除 dtype 语义 + `zero-div`"设计**与 NumPy 不符**：NumPy 整数零除返回 0 而非报错。裁决：**跟随 NumPy（返回 0）**。

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

`package → util → iterator → nan → dtype → core → map-reduce → io → creation → manip → indexing → join → elementwise → reduce-stats → setops → random → linalg → simd-matmul → nn → rotate → extensions → extensions2`

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
| `vt-+ - * /` | varargs；`vt-/` 整型输入预提升 float64（NumPy true_divide） | ✅（`:`out` dtype 校验缺失 → D2） |
| `vt-mod/rem` | 零除返回 0（整数）/ nan（浮点）；符号跟随除数 | ⚠️ 需改 |
| `vt-maximum/minimum/fmax/fmin` | NaN 传播 vs 忽略 | ✅ 正面样板（D10） |
| `vt-clip` | `a_min`/`a_max` 至少给一个；`a_min > a_max` 时结果全为 `a_max` | ⚠️ 需改（D5） |
| `vt-hypot` | 任一 Inf → `+Inf`（即使另一为 NaN） | ⚠️ 需改（D4） |
| `vt-reciprocal` | 整数输入 → NumPy 返回 0（整型倒数恒 0） | ⚠️ 需改（D9） |
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
| `vt-reshape` | 至多一个 `-1`；尺寸不符报错 | ✅ |
| `vt-squeeze` | 默认移除**全部**长度 1 轴（`(1,3,1)→(3)` ✅ 实测） | ✅ |
| `vt-transpose` | 默认反转全部轴 | ✅ |
| `vt-norm` | 整数输入 → float64 | ✅ |
| `vt-einsum` | out dtype 硬校验（当前已实现，**保留为样板**） | ✅ |
| `vt-matmul` | 1d 提升语义；`out` 走 `vt-copy-into` | ✅ |
| `vt-isclose` | `rtol=1e-5, atol=1e-8` | ✅ 实测默认宽松（1e-6 内判等），需在文档写明 |
| `vt-slice` | spec 语法须与文档统一（**无 `:range`** → D1） | ⚠️ 仅需补文档 |
| `vt-where` | 支持单参数形式返回索引（→ D6） | ⚠️ 需改 |
| 空归约 | 三分类语义 | ✅ 正面样板（D11） |
| 比较运算 | 返回 `:int8` 承载布尔（0/1） | ✅（文档需写明） |
| 三角/反三角 | `asin/acos/atan` 命名（非 arc*），越界返回 NaN | ✅（命名见 §3.1.1） |
| **docstring 覆盖** | 每个公开函数须有文档 | ⚠️ `vt-var` 等为 NIL，需全量审计（任务5） |

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
| 引用已删除常量（`+vt-float-nan+` 等）的注释 | 删除或改引用 |
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
**裁决**：⚠️ 必须报错（见 §4.2 H3）。这是**静默数据损坏**风险：用户以为拿到 float64，实得被截断的 float32。

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

### D5 — `vt-clip` 无默认边界参数（违反 NumPy 签名）

```lisp
(vt-clip (vt-from-sequence '(1.0 2.0 3.0)))
;; => ERROR: invalid number of arguments: 1
```
NumPy `np.clip(a, a_min=None, a_max=None)` 允许只给一个边界。
**裁决**：⚠️ 改为 `(vt-clip vt &optional min-val max-val ...)`，允许 `nil` 表示不限。同时需支持 `a_min > a_max` 时结果为 `a_max`（NumPy 实测 `np.clip([1,2,3],2,2) → [2,2,2]`）。

### D6 — `vt-where` 不支持单参数形式（违反 NumPy）

```lisp
(vt-where (vt-from-sequence '(1 0 1)))
;; => ERROR: invalid number of arguments: 1
```
NumPy：`np.where(cond)` 返回 `(array_of_true_indices,)`。
**裁决**：⚠️ 增加单参数重载，等价 `vt-argwhere` 的多返回值形式。

### D7 — `vt-var` / `vt-std` **已有** `ddof`，但**缺 docstring**

```
(vt-var (vt-from-sequence '(1.0 2.0 3.0 4.0)) :ddof 1)  => 1.6666666666666667d0   ✅ 与 numpy 一致
(vt-std (vt-from-sequence '(1.0 2.0 3.0 4.0)) :ddof 1)  => 1.2909944487358056d0  ✅
(documentation 'vt-var 'function)                       => NIL                   ⚠️ 无文档
```

**裁决**：✅ `ddof` **无需新增**（原判断有误，已撤销）。⚠️ 但 `vt-var`/`vt-std` 等函数**缺 docstring**，属任务5 范围。

### D8 — `vt-arange` 无参调用报错信息不可读

```lisp
(vt-arange)
;; => ERROR: invalid number of arguments: 0
```
**裁决**：⚠️ 改为 `vt-arange: 缺少必需参数 total-num（元素个数，非 stop 值）`。

### D9 — `vt-reciprocal` 对整数输入返回浮点（违反 NumPy）

```lisp
(vt-reciprocal (vt-astype (vt-const '(3) 2) :int32))  ;; => (0.5d0 0.5d0 0.5d0)
```
NumPy：`np.reciprocal(np.array([2],dtype=np.int32))` → `[0]`（整型倒数恒 0，因整数除法）。
**裁决**：⚠️ 或对齐 NumPy 返回整数 0，或在 docstring 明确标注差异。
**建议**：跟随 NumPy 返回整数 0（用户显式要求除 `vt-arange` 外全部对齐）。

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
| `vt-write-1` | **strides 驱动**寻址写入，零连续性假设 |
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
| 扩展 | `extensions.lisp` / `extensions2.lisp` | 全部透传下游原语，已自动获得统一契约 | ✅ |
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
| 零除（整数） | mod/floor-div | — |
| `vt-clip` 参数 | 已按 numpy 语义 | — |
| `vt-where` 单参 | 已对齐 | — |
| `vt-arange` 报错 | 已对齐（唯一不以 numpy 为准的例外） | — |
| `vt-reciprocal` 整数 | 已按 numpy 整数语义 | — |
| docstring 覆盖 | 公开 `vt-*` 基本齐全（余 4 个 defstruct 访问器） | — |
| `most-positive-double-float` 误用 | 已排查（D4b） | — |
| nan/inf 常量精简 | 三个 `+vt-float-*+` 已删除（§5.1） | — |
| 测试/example 审计 | `run-all-tests` + 25 套件全绿（任务6） | — |

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
