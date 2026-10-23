# CHANGELOG

本文件记录 clvt 库的每一次重大修改。

---

## 2026-10-23（十八）— 第十八轮：转置视图的复数拷贝提速（`vt-copy` 17 → 6 ms）

**行为变化**：无。只换遍历形态，落点、dtype、形状一律不变。

**性能**（1M 元素，SBCL 2.6.8）：

| 场景 | 修前 | 修后 | 同形实数对照 |
|---|---:|---:|---:|
| `vt-copy` 转置 c128T | 17 ms | **6 ms** | 3 ms |
| `vt-contiguous` 转置 c128T | 17 ms | **6 ms** | 4 ms |
| `vt-astype` 转置 c128T→c128 | 6 ms | **4 ms** | — |
| `vt-astype` 转置 c128T→c64 | 9 ms | **7 ms** | — |

两处根因，都不是上一轮猜的「访存次序」：

1. `%gather-contiguous` 用全轴 odometer，每元素约 5 ns 的固定开销（1M = 5 ms）。
   改成**末轴 `DOTIMES` 展开、其余轴走 odometer**后，转置收集 6 → 1 ms；
2. `%copy-generic`（`vt-copy`/`vt-contiguous`/`vt-copy-into` 的公共内核）的
   按元素类型分派表里**缺复数两格**，复数整片掉进泛型 `aref`。补两格即回到实数线。

**同时更正上一轮的一条错误结论**：`c128T→c128` 曾被写成「比 numpy 慢一个量级」。
复核发现 numpy 的 `astype` 缺省 `order='K'`（转置输入返回 F 连续结果 = 一次线性流），
同口径要写 `order='C'`（1.05 ms）；而「差距在访存次序」也不成立 —— 手写实验脚本
当时的「列主序 0 ms」写错了内层步长。详见 `HISTORY.md` §H.3.11 的更正与 §H.3.12。

**已知且不修**：SBCL 的 `replace` 没有 16 字节元素的 basher，`(complex double-float)`
的同型块拷贝只能逐元素（5~10 ms/1M，而 `double-float` 是 1 ms）。`sb-kernel` 的
`system-area-ub8-copy` 能到 1.5 ms 级，但属内部符号且需 pinning，与「可移植优先」
冲突，已写进 `CONVENTIONS.md` §16.4.1 作为已知固定成本。

验证：`test-copy-into` 35 → 41 断言（新增复数×非连续的 typed-walk 覆盖 + 倍率护栏），
55 套件全绿，编译体检 `caught WARNING` 0。

---

## 2026-10-23（十七）— 第十七轮：`vt-astype` 的复数转换提速 5~20 倍

**行为变化**：无。所有 dtype 组合的取值 / dtype / 形状 / 副本语义一律不变，
只换执行路径（100 格 numpy 对拍逐位相同）。

**性能**（1M 元素，SBCL 2.6.8，同进程 6 次取最优）：

| 转换 | 优化前 | 优化后 |
|---|---:|---:|
| `c128→c128`（同 dtype 的纯拷贝） | 17 ms | 3 ms |
| `c128→c64` | 31 ms | 2 ms |
| `c64→c128` | 26 ms | 3 ms |
| `c128→f64` | 25 ms | 2 ms |
| `c128→i64` | 34 ms | 3 ms |
| `f64→c128` | 27 ms | 3 ms |
| 转置视图 `c128T→c64` | 35 ms | 8 ms |

实数线（1~3 ms）不受影响。

**根因**：`vt-astype` 的特化分支只列了实数元素类型的组合，复数全部落进
「泛型 `aref` + `vt-cast-fun` 闭包」的兜底 —— 每个元素一次 `complexp` 判断
加装箱/解包。补上 34 个复数子句（同型走 `replace` 块拷贝）后即与实数线同量级。

**非连续视图**：新增 `%gather-contiguous`（按 `strides` 的同 dtype 特化收集），
转置 / 翻转 / 切片 / 广播视图由 17~35 ms 降到 3~8 ms。

**新套件**：`astype-cast-test`（663 断言 = 100 格矩阵 + 契约 + 非连续等价 +
倍率护栏），期望值由 `test/gen_astype_cast_expected.py` 从 numpy 2.4.6 实测生成。
新契约见 `CONVENTIONS.md` §16.4.1。

过程、消融与一条负结果（转置拷贝的访存次序仍压着 2~4 倍）见 `HISTORY.md` §H.3.11。

---

## 2026-10-23（十六）— 第十六轮：`:out` 重叠保护缺口 + 两处 dtype 静默降级

**行为变化（可能影响调用方）**

- 实数 `cumsum`/`cumprod` 的 `:out` 与输入**错位重叠**时，以前会算错（先写的
  结果污染还没读的输入，越滚越大），现在正确；
- `(vt-nanmedian 实数 :dtype ≠ :float64)` 与 `(vt-nanmedian 复数 :dtype 非复数)`
  由「静默忽略 / 静默取实部」改为**报错**。缺省行为不变。

**修掉的两个静默错值**

| 问题 | 影响 |
|---|---|
| 流式累积（实数 `cumsum`/`cumprod`）没做 `:out` 重叠快照 | 错位重叠时输入被自己写坏 |
| `nanmedian` 的 `:dtype` 被忽略 / 复数降级成实数 | 显式 dtype 无效；虚部无声消失（正中 §16.13 红线） |

**一条负结果**：复数 matmul 把「每元素一次 C 读改写」换成 4 个局部累加器，
A/B 实测无稳定收益，**已回退**。瓶颈在复数乘法的标量算术，不在 C 访存；
再提速只能上 SIMD。

验证：48 例 cumsum 沿轴随机对拍、退化多轴 nanmedian 与 numpy 2.4.6 一致；
`cumulative-axis-test` 70 → 78 断言；54 套件全绿，编译体检 WARNING 0。

过程、踩坑与实测表见 `HISTORY.md` §H.3.3。

---

## 2026-10-23（十五）— 第十五轮：沿轴累积提速 49× + `nanmedian` 多轴 + 编译门禁

**性能**（1000×1000，`vt-cumsum :axis`）：196 ms → **4 ms**（numpy 约 3.3 ms）。
同一个函数的展平路径只要 1 ms —— 差别只在于沿轴仍在走通用回退（每元素一次
`funcall` 装箱 + `vt-cast` + 泛型 `aref`）。把「平面」与「沿轴」统一到同一个
按元素类型特化的内核后，两条路径都吃到特化。`nancumsum :axis` 260 → 14 ms。

**新能力**：`vt-nanmedian` 支持多轴（`axis=(0,1)`），与 numpy 逐位一致。
裁决依据是「中位数对归约区内元素顺序不敏感」，因此可用「置换 → 折叠成 2-D →
沿单轴归约」复用单轴内核；实数与复数共用同一份折叠逻辑。

**新门禁**：`run-tests.sh` 跑套件前先做一次全系统编译体检，`caught WARNING` > 0
即 FAIL（`STYLE-WARNING` 不计）。理由是第十四轮那个「docstring 裸引号」事故
**运行期完全正常**，只有编译输出能发现。注意**不能**用 `muffle-warning` 包编译
——它会把告警一起吞掉，门禁恒报 0。

验证：新增 `cumulative-axis-test`（70 断言，含独立朴素参照对照 + 性能护栏
「沿轴 ≤ 40× 展平」）；54 套件全绿。

过程与消融数据见 `HISTORY.md` §H.3.2。

---

## 2026-10-23（十四）— 第十四轮：工程状态复审（4 个缺陷）

前十三轮解决「复数能不能算对」。本轮换角度：在全量重跑测试入口的过程中追问
「这条命令在一台干净的机器上真的能过吗」，抓到 4 个缺陷 + 1 个待裁决项。

| # | 问题 | 影响 |
|---|---|---|
| 1 | `elementwise1/2` 是 `elementwise` 的完整副本（只新增文件、没删原文件） | 拆分零收益（4 GB 堆仍 `Heap exhausted`），且重新引入 85 对跨文件同名 `defun` |
| 2 | docstring 里一个裸 `"` 提前闭合字符串，`is/not/of/type/REAL` 成了函数体里的 5 个 form | 20 条编译 WARNING，但**运行期正常** —— 靠未定义行为工作 |
| 3 | key-sync 门禁逐字节比对，而 `str(np.float64)` 随 numpy 版本变（差 1 ulp） | 本机必然 `exit 1`，52 个套件一个都跑不了 |
| 4 | 实数 `vt-cumulative` 漏包 `with-float-safe` | `(vt-cumprod #(inf 0))` 直接抛条件，numpy 给 `[inf, nan]` |

**行为变化**

- 实数 `cumsum`/`cumprod` 的首元素改为「直拷输入」而不是「从单位元起算」：
  `cumsum([-0.0])` 现在是 `-0.0`（不是 `+0.0`），符号位不再被洗掉；
  `cumprod([inf, 0])` 现在是 `[inf, nan]`（不是抛条件）。
- `vt-svd` 的 `max-sweeps`/`tol` 改为 `supplied-p`：实数 50/1e-14、复数 60/1e-15，
  各用自己验证过的缺省。此前复数 SVD 一直跑在实数的那组参数上。

**门禁变化**：key-sync 默认改「语义比较」（key/dtype/shape 必须一致，`:tol` 条目
按 dtype 分级容差，`:exact` 仍逐字符），`--strict` 保留字节口径；另加「生成器
退出码非 0 即 FAIL」（文件不重写的假通过比没门禁更危险）。

验证：冷编译 4 GB 从 `Heap exhausted` → 成功；`caught WARNING` 20 → 0；
跨文件同名 `defun` 85 → 0；新增 `scan-first-element-test`（38 断言，共 53 套件）。

过程、踩坑与实测表见 `HISTORY.md` §H.3.1。

---

## 2026-10-23（十三）— 第十三轮：覆盖式复数层收尾（159/159 单一 DEFUN）

第十二轮把 143/159 改成单一 `DEFUN` 分派，剩 16 个"形状不规整"的；本轮做完并
删掉整套覆盖机制。

16 个符号按"该怎么改"分四类：**纯冗余**（`vt-@`/`vt-expt`，删重复 defun）、
**错误死桩搬迁**（`vt-real`/`vt-imag`/`vt-conj`/`vt-angle` —— `extensions3.lisp`
那份写着"本库无复数 dtype，恒返回输入本身"，是错的，可靠"后加载者胜"维持）、
**匿名包装合并**（`vt-log`/`vt-float-power`）、**同名覆盖 + `%cx-orig` 转发**
（8 个，复件改名 `%cx-op-*`，归属文件插守卫，两份 lambda-list 合成一份）。

结构上把标量内核前移到新文件 `src/complex-kernels.lisp`（`dispatch` 之后、所有
归属文件之前），依赖方向变成单向「归属文件 → 标量内核」，消除倒挂。

迁移中发现 `vt-log` 的 `:base` 一直在除零：`(log base 1.0d0)` 是**两参数对数**
`log_1(base)`，导致 `(vt-log #C(8 0) :base 2)` 返回 `0+0i`（应 `3+0i`）。这个 bug
在覆盖式架构下活了五轮 —— 因为没有任何用例覆盖"复数 + 非缺省 base"这一格。

新增 `test/complex-reload-test.lisp`（14 断言）：同进程逐个重载 12 个源文件，
每轮跑 19 个探针。这是本轮**唯一真正的验收**：覆盖式架构下它必然失败。

结果：单一 `DEFUN` 分派 **159/159**；`(setf (symbol-function …))` 2 → **0**；
`%cx-orig` 调用点 16 → **0**；`eval-when` 预捕获清单 124 条 → **0**；
跨文件同名 `defun` 14 对 → **0**；套件 51 → **52**。

---

## 2026-10-23（十二）— 第十二轮：拆掉覆盖式复数层（第一阶段 92/159）

**背景**：复数支持早期靠"运行期重定义公开函数"实现（`(setf (symbol-function …))`
装匿名包装，或"同名 defun 再写一遍 + `%cx-orig` 转发"）。后果是：只重载
`src/elementwise.lisp`，`(vt-sqrt #C(-1 0))` 会从 `#C(-0.0d0 1.0d0)` 静默变成
`float64` 的 `NaN`；函数对象是匿名 LAMBDA，`declaim inline` 失效；哪些符号被覆盖
无法静态枚举。

**做法**：把复数分支下沉到归属文件的**单一 `DEFUN` 体内**：

```lisp
(defun vt-abs (vt &key out dtype)
  (when (%cx-any-p vt)                    ; 复数分流
    (return-from vt-abs (%cx-abs …)))
  …原实数实现不变…)
```

新增 `src/dispatch.lisp` 作为唯一判定入口（`%cx-arg-complex-p` / `%cx-any-p` /
`%cx-deep-p` / `%cx-result-complex-dtype` / `%cx-reduce-dispatch`），在 `clvt.asd`
里紧随 `core` 加载。本轮完成 92/159（elementwise 20 拒绝型 + 10 比较/极值 +
18 一元、nn 9 个激活、extensions3 10 个、reduce-stats 14 个门面……）。

顺带发现 9 处 lambda-list 漂移（两处同名 defun 的参数表不一致，靠加载顺序决定
谁生效），记入待办。

新增 `test/complex-dispatch-test.lisp`（结构契约：必须是具名 `DEFUN`、list 入参
走复数路径、实数不回归）。套件 50 → 51。


## 2026-10-21（十一）— 第十一轮：弱标量提升（窄 dtype 不再被标量带宽）

前十轮都在比**值**和「复数会不会掉回实数实现」。本轮换成**只比 dtype**：
119 条表达式全喂 `complex64`（外加实数 dtype 对照），逐项与 numpy 的**结果 dtype**
比对。值的正确性前几轮已覆盖，这一维是空白 —— 结果抓到一整类问题。

### 症状：窄 dtype 遇到标量就静默升宽

| 表达式（`*` = 张量） | 修复前 | 修复后 = numpy |
|---|---|---|
| `f32* + 1.5` / `f32* + 1` | float64 | **float32** |
| `i32* + 1` / `i8* + 1` / `u8* + 1` | int64 | **int32 / int8 / uint8** |
| `c64* + #C(1 2)` / `c64* + 1.5` | complex128 | **complex64** |
| `f32* + #C(1 2)` | complex128 | **complex64** |
| `f32* / 2` | float64 | **float32** |
| `c64* ** 2`、`maximum(c64*, 标量)` | complex128 | **complex64** |

不是算错，而是**分配了双倍内存、带宽翻倍、SIMD 通道白丢一半**，且完全不报错。
顺带把 `vt-gradient(c64)` 也带回了 complex64（它是自然提升的函数，之前被标量带宽）。

### 根因

逐元素运算的 dtype 一律走

```lisp
(apply #'vt-promote-type (mapcar #'vt-dtype inputs))   ; inputs 已全部 ensure-vt
```

而 `(ensure-vt 1.5d0)` 是 **float64 的 0 维张量**、`(ensure-vt #C(1 2))` 是
**complex128** —— 「谁是标量」在 ensure 之后已经没了，标量于是以全强度参与提升。

numpy 2.x（NEP 50）里 Python 标量是**弱**类型：只贡献种类（int/float/complex），
不贡献宽度；宽度由张量决定，只有「整型张量 + 浮点标量」才落到该种类的默认宽度。
CL 的 `1.5d0` 对应 Python 的 `1.5`，语义正好对齐，故按 NEP 50 修
（CONVENTIONS §0：与 NumPy 冲突以 NumPy 为准）。

### 修法

`src/dtype.lisp` 新增 `%SCALAR-KIND`（张量→NIL，Lisp 数→:INT/:FLOAT/:COMPLEX）
与 `%OP-FINAL-DTYPE`（张量之间仍走原 100 格 promote 表，弱标量只按种类并入）。
接进 `vt-map`、`vt-fast-map` 宏、`vt-/`、`vt-pow`（含 float-power）、
`maximum/minimum/fmax/fmin`。

一处关键细节：**标量身份只能沿着「原始实参」传**。`vt-/` 第一版只改了自身判定，
随后又把已 ensure 的值传给 `vt-fast-map`，标量身份二次丢失，`f32/2` 依旧 float64。
`vt-fast-map` 宏因此改成「先绑原始实参、再 ensure-vt」。

### 附带优化：单元素输入预转（让内联快路径回来）

改完 dtype 先测性能，发现 c64 数组 + 标量反而慢 27% —— 因为「弱标量」在 ensure-vt
之后是**另一种 dtype 的 0 维张量**，`%VT-INLINE*-FAST` 的「输入与输出同元素类型」
前置条件不成立，整个运算掉进 funcall + 泛型 aref 兜底。

修法：对满足五个条件的**单元素**输入预转成结果 dtype（`%WEAK-CAST1`）：
SIZE=1、dtype 与结果不同、种类秩 `源 ≤ 目标`（int<float<complex 只走无损方向）、
整数值要装得下、**这组输入里至少有一个真张量**。不整体预转大数组。

后三个条件是踩出来的，每一条都对应一次「预转把语义前提悄悄改掉」：
* 只按「单元素 + dtype 不同」转 → 比较运算的标量进了 int8 掩码，`nan<1` 判错；
* 加上「同大类」→ 仍会把 `999000` 转进 int8（变成 -8），argwhere 数出 999911；
* 加上「值装得下」→ 全标量的 `(vt-complex 1.0 2.0)` 把实参转成复数，
  构造函数收到 `#C(1 0)` 报 `is not of type REAL`。
三次都是跑全量套件才发现的 —— 局部看都对。

实测 2M 元素（min-of-7）：

| 表达式 | 第十轮 | 第十一轮 |
|---|---:|---:|
| `f32 数组 + 1.5` | 77 ms | **4.1 ms**（19×）|
| `c64 数组 + 1.5` | 163 ms | **5.4 ms**（30×）|

旧实现慢正是因为升宽导致输入/结果类型不一致、一直在跑兜底路径。

### 有意保留的三处差异

* `det/inv/solve/pinv/cholesky/qr/matrix-power/svd/eigvalsh` 的 c64 输入仍返回
  complex128 / float64：复数线代**恒在 double 下计算**（`vt-det` 文档已写明），
  精度更高，改回单精度是降级；
* `ptp` / `cond` 返回 Lisp 标量而非张量（§5 既定风格），原生数没有 dtype；
* 弱整数越界（`i8* + 300`）本库回绕，numpy 抛 OverflowError。

### 测试

`test/dtype-weak-scalar-test.lisp`（**14 断言**，已接入 `run-tests.sh`，共 50 套件）。
按要求**只收录修复前确实错的用例**：11 个 dtype 断言（每一个都在修复前的构建上
复验过「当时真的是错的」）+ 3 个值正确性断言（确认不再升宽没把值算错）。
张量×张量的 promote 100 格与 c64 其余 dtype 契约由既有套件覆盖，不重复。

全量 **50 套件全绿**；`manip-interop-test` 的 195 例 Lisp↔numpy 逐位对照 0 差异。

文档：CONVENTIONS §16.31（方法 / 症状 / 根因 / 修法 / 保留差异）、
§16.20 变更清单。

---

## 2026-10-21（十）— 第十轮：median 契约、前缀扫描首元素与 NaN/Inf 语义

第九轮补了「数值对拍」这条扫描线，本轮把表达式从 190 条扩到 **262 条**，
新增 NaN/Inf 全组合、`axis` 元组、`:out` × 非连续、einsum 补全四个方向。
结果：**除 `vt-median` 外这三块都没有静默错值**，而 `median` 一次暴露三处。

### 缺陷一：`vt-median` 复数版与自家实数版三处分叉（都不报错）

```lisp
(vt-median a)   ; a = (3 4) complex128
;; 修复前 → 3 个中位、形状 (3);  修复后 → 0 维标量（与实数路径、numpy 一致）
```

| # | 症状 | 根因 |
|---|---|---|
| 1 | `axis=nil` 按**末轴**归约（`(3 4)` → 形状 `(3)`，`:keepdims` → `(3 1)`） | 缺省轴写成 `(1- rank)`；numpy 与实数版都是「先 ravel 再取全局中位」 |
| 2 | 空 fiber（轴长 0）返回 **0+0i** | `lo = (floor -1 2) = -1` + `%cx-fiber-starts` 在 size=0 时返回 0 条 → 结果保持 `make-vt` 的 0 |
| 3 | 含 NaN 返回**某个真实元素** | 复数版没有 NaN 检查；numpy 与实数版都传播 NaN |

顺带补了个洞：**复数 `vt-median` 原来没包 `WITH-FLOAT-SAFE`**（实数版包了）。
SBCL 默认开着 `OVERFLOW/INVALID/DIVIDE-BY-ZERO` 陷阱，于是 NaN 判定 `(= x x)`
与 `inf + (-inf)` 会直接抛 `FLOATING-POINT-INVALID-OPERATION`。

### 缺陷二：前缀扫描用「单位元」而非「首元素直拷」

numpy 的 **accumulate** 与 **reduce** 在这里方向相反：

```python
np.prod([inf+0j])            # (inf+nanj)          ← reduce 从 1 起算
np.cumprod([inf+0j, 2+0j])   # [inf+0j, inf+nanj]  ← accumulate 首元素直拷
np.cumsum([-0.0+0j])         # [-0.+0.j]           ← 首元素直拷，符号位保留
```

`%CX-SCAN-RUN` 两条路径都从单位元起算，于是：

* `cumsum` 把输入的 `-0.0` 洗成 `+0.0`（符号位丢失）；
* `cumprod` 首元素走 `1 * (inf+0i)` → `(1·inf − 0·0, 1·0 + 0·inf)` = `inf + nan i`，
  凭空多一个虚部 NaN，并被后续所有元素继承。

修法：连续路径与 dtype 回退路径都改成「首元素直拷，从第二个起才 OP」。
有限数据下与旧实现逐位相同，只在 -0.0 / Inf 上有别。

> 遗留：实数线的 `vt-cumsum` 走通用 `VT-CUMULATIVE`（acc 从 0 起算），
> `cumsum([-0.0])` 仍给 `+0.0`。修它要动四处循环，与本轮主题无关，列为下一轮候选。

### 已核实「不是 bug」的差异

对拍剩下的 7 项不一致逐项归因：QR 的相位约定（重构误差 1.1e-16 / 6.7e-16）、
LU 的置换约定、numpy 复数均值用 `sum × (1/n)` 导致 `0*Inf` 污染另一分量、
全 NaN 切片里「用哪个 NaN」（`NaN+0i` vs `NaN+NaN i`）。
**对拍不一致 ≠ 缺陷** —— 先排除约定差异与对方的浮点副作用，再决定改不改。

### 测试

新增 `test/complex-nan-inf-test.lisp`（**77 断言**，已接入 `run-tests.sh`，现共 **49 套件**）：

- median 契约 15 项（`axis=nil` 形状/值/keepdims、与实数路径形状一致、NaN 全局与逐 fiber 传播、空轴/空输入、c64、`nanmedian` 对照）；
- 前缀扫描 8 项（`-0.0` 符号位、`cumprod` 首元素 `inf+0i`、`axis=0` 每列首元素、c64、`axis=nil` 展平）；
- NaN/Inf 归约 17 项（`nan*` 家族含 ±Inf 与全 NaN 切片、`inf+(-inf)` 相消、掩码三兄弟、`nan_to_num` 三替换、含 NaN 排序）；
- `axis` 元组 12 项（形状 + **逐元素手算**对照，不借道库函数）；
- `:out` 7 项（列切片**逐位置**核对、转置视图、reduce keepdims 列视图、matmul、原地别名）；
- einsum 9 项（`...i,...i->...`、`ii->`、`ij->ji`、`ijk->ik`、`ijk,jk->i`、`i,j->ij`、`ab,bc,cd->ad`、`ijk->kji`、`...ij->...i`）。

测试进程**故意不关 FP 陷阱**（SBCL 默认全开），既验证开陷阱时也能安全产出 NaN，
也顺带抓到了 `WITH-FLOAT-SAFE` 的缺失。全量 **49 套件全绿**。

文档：CONVENTIONS §16.30（方法 / 两个缺陷 / 三类「不是 bug」/ 测试）、§16.20 变更清单。

---

## 2026-10-21（九）— 第九轮：沿轴扫描的静默错值（cumsum/cumprod 只算第一条 fiber）

上一轮修的是「dtype 静默降级」，这一轮补上「**数值静默错值**」这条线 ——
两层新扫描（同形调用对照 + 与 numpy 的数值对拍）跑下来，抓到 1 个真 bug。

### 症状

```lisp
(vt-cumsum a :axis 0)     ; a = (3 4) complex128
;; => [[-1.5-0.5j,   0+0j,  0+0j, 0+0j],
;;     [-1.0+0j,      0+0j,  0+0j, 0+0j],
;;     [-2.0+0.25j,   0+0j,  0+0j, 0+0j]]   ; 只有第 0 列对
```

`axis` 是**最后一维**时正确，任何更靠前的轴都只算第一条 fiber，其余留 0。
受影响：`vt-cumsum` / `vt-cumprod` / `vt-nancumsum` / `vt-nancumprod`。

### 根因

`%CX-SCAN-RUN` 的 axis 分支拿到 `(head dim ax-stride)` 后只循环了
`head × dim`，**漏掉 `ax-stride` 那一层** —— 轴之后的每个位置（`axis=0`
时的其余各列）都是一条独立 fiber。

同一份 `%CX-FIBER-GEOM` 的另外六个调用点（`%DEF-CX-REDUCER` 家族、
`argmax/argmin`、`nanarg`、`sort-axis`、`nanmedian`）**都有**这层循环，
所以 `sum/prod/amax/median/sort/diff/...` 沿轴一直是对的，只有扫描内核漏了。

**为什么一直没被发现**：所有既有 `cumsum` 测试都是 1-D，而 1-D 时
`ax-stride = 1`，漏掉的那层退化成一次迭代，数值完全一样。

### 修法

补上 `ti` 层遍历（三层循环：前导块 → ax-stride 内偏移 → 沿轴），
并在注释里写明「这层不能删」的原因。

### 本轮的两层扫描方法（已写入 CONVENTIONS §16.29.4）

| 层 | 做法 | 结果 |
|---|---|---|
| A. 同形调用对照 | 对每个导出 `vt-*`，用同一组 20 种调用形态分别喂实数/复数镜像，比「实数通、复数失败/降级」 | 0 处（第八轮已清完；上一轮报的 96 个 `complex→real` 全是某个不含复数入参的形态造成的假阳性） |
| B. 数值对拍 | 190 条复数表达式 clvt 求值 vs numpy 复算，逐元素比对 | 188 一致，1 不一致（= 本 bug），2 项是 QR/LU 约定差异 |

B 覆盖：逐元素/超越、归约（axis 元组/keepdims）、沿轴扫描、排序/argsort/
unique（含 counts/inverse）、searchsorted、diff/gradient/trapezoid、
拼接/堆叠/切分/insert/delete/repeat/tile/pad/rot90/flip/roll、take/compress/
extract/take-along-axis/put-along-axis、diag/tril/triu/trace/diagonal、
matmul/dot/outer/inner/vdot/kron/cross/tensordot/multi-dot、einsum 五种模式、
det/inv/solve/pinv/cholesky/matrix_power/cond/svd-qr、复数比较与 where，
外加 complex64 一整组镜像。

> QR/LU 的两处「不一致」**不是 bug**：QR 的列可乘任意单位模相位，
> 验证用重构恒等式（‖QᴴQ−I‖ = 1.1e-16、‖QR−A‖ = 6.7e-16）；
> LU 的置换约定不同，重构由 `complex-linalg-test` 覆盖。

### 测试

新增 `test/complex-scan-axis-test.lisp`（**26 断言**，已接入 `run-tests.sh`，
现共 **48 套件**）。期望值由 numpy 2.3.5 实测硬编码，数据是二进制可精确
表示的小数、累加也精确，故要求**逐位相等**：

- `cumsum`/`cumprod` 的 axis=0/1 + 负轴 `-1`/`-2`；
- 3-D（2,3,4）三个轴各一组（最能暴露「只算第一条 fiber」）；
- 非连续输入（转置视图）沿 axis=0；
- `nancumsum`/`nancumprod` 无 NaN 时与普通扫描逐位相同；
- complex64 与 complex128 一致、dtype 保持；
- 退化：轴长 1 / 轴长 0 / `axis=nil` 展平；
- `:out`：连续目标 + 非连续目标（列切片）；
- 实数路径回归。

全量 **48 个套件全绿**。

---

## 2026-10-21（八）— 第八轮：复数缺口扫描 + 6 个补丁（3 个静默错值）

上一轮把复数 matmul 的内核修好之后，做了一次**系统性扫描**：导出清单 367 个
`vt-*` 里 202 个已被复数层碰过，剩下 165 个逐个用复数入参探针跑；
再加一层「所有可调用导出符号统一喂 complex128，看结果 dtype 是否变实数」的
保真扫描（唯一能抓「静默丢虚部」的手段）。

结论：**6 个缺口**，其中 3 个是静默错值（不报错、悄悄丢虚部）：

| 函数 | 改前 | 改后 |
|---|---|---|
| `vt-from-array` | `(complex double-float)` 数组 → **float64** | **complex128**（complex-single → complex64） |
| `vt-from-sequence` | 含复数的 list → **float64** | **complex128** |
| `vt-full` | `#C(3 1)` 填充 → **float64** | **complex128** |
| `vt-float-power` | 复数 → **float64**（只算实部） | 复数主值幂（`(2+1j)^2 = 3+4j`） |
| `vt-cbrt` | 复数 → **float64** | **确定性报错**（numpy 亦拒绝） |
| `vt-layer-norm` | 复数 → `#C(0 0) is not of type REAL` | **complex128**（复数均值 + 实数方差） |

### 三个语义决策

1. **`cbrt` 不做复数实现，改成拒绝**：numpy 2.3.5 实测
   `np.cbrt(np.array([8+0j]))` 直接 `TypeError: ufunc 'cbrt' not supported`。
   补一个 numpy 没有的「复数立方根」是造语义，不如对齐；
   要用请 `(vt-pow z 1/3)`。
2. **`float_power` 复用 `pow` 的内核**：两者在复数域是同一个主值分支
   （实测 `np.float_power` 与 `np.power` 逐位相同），差别只在实数侧。
   把 `vt-pow` 里内联的 lambda 提成 `%CX-POW-2` 两处共用，
   边界（`x^0=1` / `0^负=NaN` / `0^0=1` / NaN 传播）只留一份。
3. **`layer_norm` 的复数语义**（本库扩展，torch 只定义实数域）：
   复数均值 + 实数方差（`mean(|x-mean|²)`，与 `np.var(complex)` 一致）
   + 实数 `std = sqrt(var+eps)`，结果复数。用「虚部为 0 时退化为实数路径」
   与「归一化后 fiber 均值 0、var≈1」两条自洽性断言锁住。

### 顺带修掉的不一致

`vt-pow` 的复数路径把 NaN 判定排在「指数为 0」之前，于是
`(vt-pow #C(nan 0) 0)` → `NaN`，而**同一函数的实数路径**给 `1.0`、numpy 也给 1。
`%CX-POW-2` 里把零指数判定提到最前，三条路径一致。

### 改动

- `src/creation.lisp`：`vt-from-array` 的 dtype 推断补 complex 两格；
  `vt-full` 的 `:dtype` 缺省改为「复数填充 → :complex128，否则 :float64」
  （不按整数推断，避免 `(vt-full s 3)` 从 float64 变 int64）；
- `src/io.lisp`：`vt-from-sequence` 的 `:dtype` 缺省同理
  （复用已有的 `%any-complex-in-sequence-p`）；显式 `:dtype` 优先级不变；
- `src/complex.lisp`：抽出 `%cx-pow-2`、新增 `vt-float-power` 复数分派、
  `vt-cbrt` 并入 `%cx-def-unsupported`；
- `src/extensions2.lisp`：`vt-layer-norm` 新增复数分支（提前 return，
  实数路径逐字不动）；
- `test/complex-gap-test.lisp`：**新增套件（50 断言）**，已接入
  `test/run-tests.sh`（现共 47 套件）。

### 测试

- dtype 契约：`from-array`（c128/c64/2D/`:fast t`/显式 `:dtype :float64`
  仍取实部）、`from-sequence`（1D/嵌套/空/显式 dtype）、`vt-full`
  （复数/实数/整数填充）；
- 数值契约：`float_power` 用 numpy 实测值（`(2+1j)^(2+0j)=3+4j`、
  `(-1+0j)^0.5=6.12e-17+1j`）+ 边界（`nan^0=1`、`0^0=1`、`0^-1=NaN`）；
- 报错契约：`cbrt` 复数报错且消息含「不支持复数」；显式非复数 `:dtype` →
  safe-casting 拒绝；
- `layer-norm`：dtype/形状、均值 0、var≈1（容差含 eps）、虚部 0 时与实数
  路径一致、复数 gamma/beta、`:out`；
- 回归网：`conj/abs/pow/sqrt/sum/norm/matmul/take-along-axis/compress/kron`；
- 全量 **47 个套件全绿**。

> 踩坑记录：测试里**别用 `(= x x)` 判 NaN** —— 测试进程没有屏蔽浮点陷阱，
> NaN 比较会抛 `FLOATING-POINT-INVALID-OPERATION`；用 `sb-ext:float-nan-p`。

### 扫描结论：仍然「不支持复数」的清单（都是有意为之）

`floor/ceil/trunc/rint`、`div/mod/rem`、位运算族、`atan2`、`even-p/odd-p`、
`percentile/quantile`、`hypot`、`cbrt`（本轮新增）、激活函数族 —— 依据都是
「numpy 同样拒绝」或「复数无全序」。

另外 `vt-eig`/`vt-eigvals` 返回**实数**特征值是**契约**（本库 `vt-eig` 是
`eigh` 语义：只吃 Hermitian/对称，用 UPLO 指定三角 + 镜像），不是 bug；
一般复矩阵的复特征值需要另一套 Hessenberg + QR 实现，属未来工作。

---

## 2026-10-21（七）— 第七轮：复数 matmul 内核对齐实数线（标量形态，1.9×）

`vt-matmul` 的复数 2D 路径一直是**教科书三重循环**：每个元素都从 `(i,j,p)`
重算扁平下标，整段落在 `%CX-DISPATCH` 的 `(safety 1)` 里，且没有 j 维展开。
实数线 `%MATMUL-DF-ROWS` 从第一版起就是 `(safety 0)` + 指针递推 + 4 路展开，
所以同一尺寸下复数比**实数标量**内核慢 2.5–3×（比实数 SIMD 慢 10× 以上）。

本轮**不动 SIMD**（按要求），只把循环形态抄成实数线的样子。

### 消融（n=256，单线程，预分配，只计内核）

| 变体 | 耗时 | GFLOPS |
|---|---:|---:|
| 旧内核（`safety 1` + 下标重算 + 无展开） | 24.2 ms | 5.0 |
| ① 只降到 `safety 0` | 22.0 ms | 5.5 |
| ①+② 行起点只算一次、内层指针递推 | 14.1 ms | 9.5 |
| ①+②+③ j 维 4 路展开 | **13.5 ms** | **9.9** |

三项收益互不重叠，**全是代码形态问题**，与复数表示无关。

### 性能（n=192 同进程比值，`test/complex-matmul-perf-test.lisp` 内）

| 项 | 耗时 | 相对 |
|---|---:|---:|
| 新行内核（c128 单线程） | 5.3 ms | 9.6 GFLOPS |
| 旧内核形状（同参照） | 10.2 ms | **1.9×** 慢于新内核 |
| 实数标量行内核 `%MATMUL-DF-ROWS` | 2.5 ms | 新/实 = **2.1×** |

复数每元素 8 flop、实数 2 flop，**flop 比就是 4×**；实测 2.1× 说明复数的
**单位 flop 效率已超过实数标量内核**（9.6 vs 5.8 GFLOPS）。默认并行下
（`*enable-parallel-matmul*` 与实数线同一套阈值）端到端相对实数 SIMD 只剩
**2.0–4.2×**，1024² 已进入内存带宽瓶颈。

### 改动

- `complex.lisp`：新增 `%def-cx-matmul-rows` 宏，生成
  `%CX-MATMUL-CDD-ROWS` / `%CX-MATMUL-CSS-ROWS` 两个行内核，
  **签名与 `%MATMUL-DF-ROWS` 完全一致**（`C += A@B`，不清零）；
- `complex.lisp`：`%CX-MATMUL-RUN` 改为「清零 + 选内核 + `CALL-PARALLEL-GEMM`」，
  直接复用 `linalg.lisp` 的调度器 → 复数与实数的并行语义（阈值/线程数/行切法）
  逐项一致，不引入第二套线程策略；
- `complex.lisp`：`%CX-GEMM-STRIDED`（einsum 的 BMM 通道）新增 (a) 分支 ——
  布局与「连续 NN」逐项相同时委托同一个行内核；(b) 行内连续 / (c) 通用 strided
  两条循环保留。累加次序不变，**einsum 与 `vt-matmul` 仍逐位相等**；
- `test/complex-matmul-perf-test.lisp`：**新增套件（53 条断言）**，已接入
  `test/run-tests.sh`（现共 46 个套件）；
- `test/complex-bench.lisp`：matmul 段扩成 128/256/512 三档 × 串行/并行。

### 被实测否掉的两个方案（记下来免得下次再试）

| 方案 | n=256 | 结论 |
|---|---:|---|
| SoA（拆 re/im）+ B 打包成 j-major + j4 | 22.9 ms | 更慢。SBCL 的复数 `*`/`+` 是**开码**的，specialized 数组里的 `(complex double-float)` 就地读写、不装箱 |
| SoA + i-j-p 单累加器 | 75.2 ms | 慢 5×。`B[p,j]` 的 p 步长是 n，每取一个元素跨一行，缓存全废 |
| 把 `p=0` 折进内核（省掉清零趟） | 14.6 ms | 更慢。j 展开的初始化代码写 4 份，挤掉内层寄存器分配 |

### 数值一致性

行内核的累加次序仍是 i→p→j 递增，与旧内核**逐位相同**；
`vt-matmul`（串行/并行）与 `vt-einsum`（NN/NT/转置输出）三方差分 maxabs = 0。
三张量链 `ij,jk,kl->il` 与 `(a@b)@c` 因缩并次序不同只保证 ~2e-15 一致。

### 测试

- **新增 `test/complex-matmul-perf-test.lisp`（53 断言）**：手写三重循环参照
  逐位对照、尾路径（n=1,2,3,5,6,7,9 专压 4 路展开的剩余列）、退化与零尺寸、
  切片视图（offset≠0 / 转置视图 / 列切片）、批量与 batch 广播分片逐位、
  einsum 三通道逐位、c64/c128、`:out`、并行（含真正超过阈值起线程的 200³ 规模）；
  性能护栏用**同进程比值**：新内核 ≥ 1.3× 旧形状（实测 1.9×）、
  ≤ 3.5× 实数标量行内核（实测 2.1×，理论下限 4×）、并行不改结果且不慢于串行；
- 全量 **46 个套件全绿**（`bash test/run-tests.sh`）。

> 遗留：SIMD 版复数 GEMM（拆 SoA 视图 + 4 条实 GEMM 复用 `simd-matmul.lisp`
> 的分块与线程框架）见 CONVENTIONS §16.27.6，是下一轮的事。

---

## 2026-10-21（六）— 第六轮：einsum 的实数 GEMM 接入 SIMD（5.7× → 1.0×）

第五轮把**复数** einsum 接到了 matmul 快路径上，但**实数**那条一直是另一套：
`vt-matmul` 走 `simd-matmul.lisp` 的 AVX/FMA + 64×256×256 分块内核，
`vt-einsum` 的批量 GEMM 通道却只调 `linalg.lisp` 的标量内核（4 路手写展开、
无 SIMD、无分块）。于是同一台机器上 einsum 比 matmul 慢 5.7~6.4×（单线程）。

本轮把两条路径接通，并顺带修掉两个既有的布局相关性能坑。

### 性能（512² float64，**单线程**，µs，3 次最好值）

| 表达式 | 改前 | 改后 | 改前 vs `vt-matmul` | 改后 vs `vt-matmul` |
|---|---:|---:|---:|---:|
| `vt-matmul`（基准） | 31,141 | 35,596 | 1.00× | 1.00× |
| `"ij,jk->ik"`（NN） | 196,033 | **32,485** | **6.30×** | **0.91×** |
| `"ij,kj->ik"`（NT） | 381,070 | **29,818** | **12.24×** | **0.84×** |
| `"ij,jk->ki"`（转置输出） | 609,077 | **36,112** | **19.56×** | **1.01×** |

2048² 多线程（32 核）同样收敛：1.78× / 1.71× / 13.82× → **0.99× / 0.99× / 0.99×**
（转置输出从 48.3 s 降到 3.46 s）。1024² 多线程 0.93~1.02×；256² 0.95~1.02×；
小矩阵（8/16/32²，低于 SIMD 阈值）两者都走标量，无退化。
复数路径与上一轮持平（0.96~1.03×）。

> 转置输出（`"ij,jk->ki"`）的「改前」是**本轮之前就存在**的老问题 ——
> 通用三重循环的内层沿 `so-k`（= 整行）走，每步换一条 cache line。
> 本轮按「临时连续 C + 分块 scatter」一并解决，不是本轮引入的回归。

### 改动

- `linalg.lisp`：新增裸数据 GEMM 钩子 `*SIMD-GEMM-RAW-FN*`（`simd-matmul.lisp`
  注册 `%SIMD-GEMM-RAW`），语义与 `%MATMUL-*-FAST-KERNEL` 一致（`C += A·B`）；
- `linalg.lisp`：新增 `define-einsum-gemm`，生成 `%EINSUM-GEMM-{F64,F32,I64,I32}`，
  把 einsum 的任意步长归一化成「连续 C += 连续 A·连续 B」：
  NN 直通 / NT 分块转置 B / C 非行主序走临时缓冲 + 分块 scatter / 其余兜底；
- `einsum-execute` 的 `gen-bmm-logic` 每个类型只留一行调用（不再内联判断）。

### 两个踩过的坑（详见 CONVENTIONS §16.26）

1. **别在 macrolet 里做布局归一化**：第一版把判断就地展开，结果**一行没改的
   兜底路径**从 711 ms 退化到 947 ms（+33%）—— 函数体膨胀后 SBCL 对它的
   寄存器分配变差。挪成独立 `defun` 后回到基线。
2. **大对象的 `:initial-element` 是 calloc 不是 memset**：2 MB 临时缓冲用
   `:initial-element 0.0d0` 时「分配 + GEMM」要 111 ms（复用缓冲只要 46 ms）
   —— 零页由 31 个工作线程各自首次触达时 fault，并发抢 `mmap_lock`。
   改成 `(make-array ...)` + `(fill buf 0.0d0)` 后回到 47 ms。

### 数值一致性

四条路最终都进同一个内核、同一分块顺序，所以 **einsum 与 `vt-matmul` 逐位
相同**（20 项差分，maxabs = 0），与 numpy `@` 相对差 ≤ 7.7e-15。相对旧的标量
三重循环只有 FMA 少一次舍入（~1e-12，更接近精确解）。

### 测试

- **新增套件 `test/einsum-gemm-layout-test.lisp`（36 条断言，已接入
  `test/run-tests.sh`，现共 45 个套件）**：四条布局路径 × 各档尺寸、
  批量（含广播与批量 NT）、f32/i32/i64/c128/c64、转置输入 / 带 offset 切片 /
  非连续 `:out` / 小矩阵、以及 SIMD 关与单线程下的不变性。
  主断言是「与 `vt-matmul` 逐位相等」——不依赖外部期望值文件（§10.4），
  且同时锁住「语义没变」和「四条路径数值一致」；
- 另写一次性差分脚本做 numpy 逐值对照（NN / NT / 转置输出，最大相对差 7.7e-15），
  以及 SIMD 开/关的 A/B 性能对照（关掉 `*SIMD-MATMUL-ENABLED*` 后两条路径
  持平，直接证明差异来自内核而非 einsum 的通道判定）；
- 全量 **45 个套件、4655 条用例全绿**；冷编译 27 文件（4 GB 动态空间，
  1 分 53 秒，峰值 3.0 GB），0 条非 style 警告。

---

## 2026-10-20（五）— 第五轮：einsum 快路径 + 两个通用特性缺口

前四轮解决「对不对」，本轮解决「快不快」和「有没有」：复数 einsum 提速 81×，
顺手挖出并修掉批量复数 matmul 的两个静默错误，再补两个 numpy 有而 clvt 没有的
通用特性（与复数无关）。

### 性能：复数 einsum 走 matmul 快路径（§16.21）

| 表达式（256×256 complex128） | 改前 | 改后 | 倍数 |
|---|---|---|---|
| `vt-einsum "ij,jk->ik"` | 2526 ms | **31 ms** | **81×** |
| `vt-einsum "i,i->"` | 9 ms | **<1 ms** | ~30× |
| `vt-einsum "ij,ij->ij"`（1M 元素） | 138 ms | **8 ms** | **17×** |

根因：三条快路径（逐元素 / 全收缩 / BMM）的判据全是实数 dtype 谓词，
复数直接落到通用迭代状态机（每个元素「补码递推 + 逐张量泛型 aref」）。

修法：新增 `%CX-GEMM-STRIDED`（带 strides 的复数 GEMM，含连续快分支与
通用 strided 分支）与 `%CX-EINSUM-DOT`，由 `complex.lisp` 在加载期注册到
`linalg.lisp` 的两个钩子上（`*CX-EINSUM-GEMM-FN*` / `*CX-EINSUM-DOT-FN*`）
——因为 `linalg.lisp` 先编译，不能直接引用复数层（同 `*SIMD-MATMUL-2D-FN*` 模式）。
判据里必须带上钩子本身，否则钩子为 NIL 时会静默返回空结果。

### 修复：批量复数 matmul 的两个静默错误（§16.22）

做差分时发现 `einsum "...ij,...jk->...ik"` 与 `vt-matmul` 差 3.3（不是精度问题）。
三方对照（einsum / matmul / 手写三重循环）定位：

| 缺陷 | 现象 | 根因 |
|---|---|---|
| offset 丢失 | 每个 batch 都读到基数组开头 | `(vt-slice (vt-ravel a) ...)` 的片段已连续 → `vt-contiguous` 零拷贝 → offset 非零，而 2D 内核按扁平下标从 0 起算 |
| 不支持 batch 广播 | `(1,4,5)@(3,5,6)` 报「重塑失败」 | 用 reshape 切 batch，隐含两侧 batch 形状相同 |

与 `manip.lisp` 的 `%flat-offset0` 一节同一类根因。改为「strides 递推 batch 偏移 +
offset 透传」，广播按「维长 1 → stride 0」处理，与实数 SIMD 路径一致。
`%CX-DOT-1D` 顺手补了同样的 offset 假设。

### 新特性：`vt-unique :axis`（§16.23，通用）

numpy 2.x 的按轴去重，此前**实数路径也没有**。严格照 numpy 算法实现
（moveaxis → reshape 成行 → 稳定字典序排序 → 相邻行全元素比较），
并把 numpy 内部的两条分支如实分层：

- 展平模式：多个 NaN **折叠为一个**；
- 按轴模式：含 NaN 的行**各自保留**（`NaN != NaN`），`counts` 全为 1。

`return-inverse` 在 axis 模式下是**一维**（长度 = 该轴长度），
可用 `(vt-take u inv :axis axis)` 重建输入。复数直接复用同一实现
（比较原语按 `complexp` 分派，字典序 = 实部→虚部）。

### 新特性：`vt-trace :offset`（+`:axis1`/`:axis2`）（§16.24）

`vt-diagonal` 早就支持 `:offset`，`vt-trace` 却硬编码 2 维。
本轮补齐 `(matrix &key offset axis1 axis2 dtype out)`，关键点是
**收缩轴是 `-1` 而不是全部**（`vt-diagonal` 把对角轴放末尾），
否则 3D 输入的 batch 轴会被一起吃掉。越界 offset 返回 0（同 numpy），
秩 < 2 用 `error` 报错（`assert` 会被 `(safety 0)` 剔除）。

### 测试

- 新增 `test/setops-axis-trace-test.lisp`：**56 断言**（37 项 numpy 2.4.6 差分
  + 契约 + 实数回归网），期望值由 `test/gen_setops_axis_expected.py` 生成。
- 全量：**44 个套件全绿**。

### 合入前审阅与独立验证

本轮补丁在合入前做过一次独立复核（不依赖作者自述，另写了一套差分脚本
重算期望值），结论：**取证通过，可合入**；顺带修正 1 处、记录如下。

**独立 numpy 差分（40 项，逐值一致）**

| 组 | 覆盖 | 结果 |
|---|---|---|
| `vt-unique :axis` | 19 项：f64/c128/i64、`axis=0/-2/1/2/-1`、`return-index`/`inverse`/`counts` 任意组合、含 NaN 行、`(0,3)`、`(2,0)`、秩 1（axis 被忽略）、3D 各轴 | 全一致 |
| `vt-trace` | 8 项：offset `0/+1/-1/+9(越界→0)`、3D `axis1/axis2` 两种组合、复数 | 全一致 |
| 批量 matmul / einsum | 13 项：`(2,3,4)@(2,4,5)`、batch 广播 `(1,4,5)@(3,5,6)`、**带非零 offset 的切片**、2D@3D、`ij,jk->ik`、`i,i->`、`ij,ij->ij`、`ij,ji->`、`ij,jk->`、`ij->ji`、转置输入（strided 分支） | 全一致 |

追加 einsum 形态（complex128 / complex64 各 5 项，全一致）：
`ij,kj->ik`（NT）、`ij,jk->ki`（转置输出）、`i=1`、`k=1`、`ij,jk,kl->il`（三张量链）。

**性能 A/B（同进程内把钩子置 NIL 复现「改前」）**

| 表达式 | 钩子关（改前） | 钩子开（改后） | 倍数 |
|---|---:|---:|---:|
| `vt-einsum "ij,jk->ik"` 256×256 c128 | 1,629 ms | **19.3 ms** | **84.6×** |
| `vt-matmul` 同规模（对照） | — | 19.3 ms | 与 einsum 持平 ✓ |
| `vt-einsum "i,i->"` 1M c128 | 156 ms | **0.94 ms** | **166×** |

（上表为审阅机实测；作者原表 2526→31ms / 81× 为另一台机器，比值一致。）

**审阅修正 1 处**：`vt-unique :axis` 在 `axis≠0` 时原本返回 `vt-transpose` 的
**视图**（`vt-contiguous-p` = NIL），而展平路径与 numpy 都返回新数组。
已补 `vt-contiguous` 落地（最多一次「唯一行」大小的拷贝），两条路径一致。

**边界行为复核**：`:out` 预填 999 时结果被**覆盖**而非累加；非连续 `:out`
（转置视图）走 strided 分支且与 `vt-matmul` 一致；batch 不可广播时报
`vt-matmul: 批量维不可广播：(2) vs (3)`；`(0,3)@(3,2)` 得 `(0,2)` 空结果；
`vt-trace` 秩 < 2 报错、`axis` 越界报错、`(2,0)` 的 unique 得 `(1,0)`。

---

## 2026-10-20（四）— 第四轮：结构算子收尾 + 数值压测固化

第三轮把线代补完；本轮用「全导出符号逐个试调」扫尾，并把随机压测写进测试。

### 补齐（3 处，都是「8 路元素类型特化漏了第 9、10 种」）

| 函数 | 修复前 | 根因 |
|---|---|---|
| `vt-outer` | `invalid number of arguments: 3` | 复数分支写成 `(vt-reshape ... :dtype dt)`，而 `vt-reshape` 只接 `(vt shape)` —— **复数外积从来没跑通过**，实数走原实现所以一直正常 |
| `vt-any` / `vt-all` | `unsupported input dtype (COMPLEX DOUBLE-FLOAT)` | 归约内核 8 路特化，复数落入 `(t (error ...))` |
| `vt-take-along-axis` | `不支持的输出元素类型 (COMPLEX DOUBLE-FLOAT)` | 逐元素内核只展开 8 种实数元素类型 |

修复策略：**能用既有算子组合的就不新写内核**。
`vt-any/vt-all` 先把复数转成 int8 真值掩码（`vt-nonzero-p`）再交给原实现，
自动继承 `axis`/`keepdims`/`:out`/空输入单位元（any→0、all→1）；
`vt-outer` 改为广播乘 + 按需 astype，并补 `:out` 硬契约。

复数真值语义（numpy 一致）：任一分量非零即为真，`bool(0+0j)=False`、
`bool(nan+0j)=True`。

### 数值压测固化（108 条断言，`stress-tests`）

差分对照只能覆盖"我事先想到的矩阵"。新增：
- **随机矩阵**：确定性 LCG（固定种子，可复现）1×1 / 3×3 / 5×5 / 2×5 / 5×2
- **恒等式**：`Q·R=A`、`Q^H Q=I`、`U·S·V^H=A`、`U^H U=I`、`V^H V=I`、
  `A·V=V·Λ`、`A·A⁻¹=I`、`A·x=b`、Moore–Penrose 四条
- **交叉验证**：`σ²` 必须等于 `eig(A^H A)` —— SVD（单边 Jacobi）与
  eig（厄米特 Jacobi）是两条独立算法，互证可抓单边系统性偏差
- **退化**：全零矩阵（σ 全 0 / rank 0 / cond ∞ / inv 报错）、秩亏（σ₂≈0）、1×1
- **SPD 族**：`B^H B + nI`（1/2/3/6 阶）→ Cholesky + eig，检验 `l_jj>0` 与特征值全正
- 容差按 `scale = max(1, ‖A‖∞)` 缩放，避免病态误报 / 良态漏报

### 顺带确认「本来就是好的」的算子

eye/identity/ones/full/*-like/tri/triu/tril/triu-indices/tril-indices/
reshape/ravel/flatten/squeeze/expand-dims/narrow/swapaxes/moveaxis/rollaxis/
rot90/tile/pad/insert/delete/append/block/repeat/resize/trim-zeros/meshgrid/
select/choose/compress/extract/where/broadcast-to/broadcast-arrays/
copy-to!/write/fill/stack/vstack/hstack/dstack/column-stack/split/hsplit/vsplit/
concatenate/concat/dot/multi-dot/matmul/outer/kron/vander/diag/diagonal/diagflat/
fill-diagonal/trace/topk/sort/argsort/median/nonzero/argwhere/flatnonzero/
count-nonzero/zero-p/nonzero-p/array-equal/array-equiv/isclose/allclose/
logical-and/or/not/xor/to-list/to-array/print —— 全部直接可用（dtype 无关）。

### 文档

- `CONVENTIONS.md` §16.19.8（第四轮补齐：三处缺口的症状/根因/修复策略）、
  §16.19.9（为什么差分对照不够：压测的五种手法与容差口径）、
  §16.19.10 测试计数更新、§16.20 变更文件清单

### 测试与回归

- complex-linalg-test **108 → 216**、complex-ops-test **131 → 149**
- 其余套件全绿：complex 136、core 284、test-all 50（642 断言）、io-interop 77、
  probes 496、probes2 405、extensions3 264、legacy 284、numpy-convention 69、
  contract 136、out-contract-v2 56、nan-broadcast 31

---

## 2026-10-20（三）— 第三轮进化：复数线性代数 + 算子补齐（收尾）

前两轮做完「逐元素」与「结构层」；本轮把**复数线代整块**从硬报错变成可用，
并修掉 8 处「复数输入静默只算实部」。

### 新增：src/complex-linalg.lisp（复数线代）

| 函数 | 算法 | 说明 |
|---|---|---|
| `vt-lu` | 部分选主元 LU | 选主元按 `\|a\|`（复数无全序） |
| `vt-det` / `vt-solve` / `vt-inv` | 基于 LU | 奇异 → det=0 / solve·inv 报错 |
| `vt-cholesky` | 厄米特 Cholesky | `A = L·L^H`；`:upper` 返回 **L^H**（不是 L^T） |
| `vt-qr` | 复 Householder | `:reduced`/`:full`；Q 用**逆序左乘**累积 |
| `vt-svd` | 单边 Jacobi（复旋转） | 不形成 A^H A，无条件数平方问题 |
| `vt-eig` / `vt-eigvals` / `vt-eigvalsh` | 厄米特 Jacobi | 特征值实数、降序；对角虚部清零 |
| `vt-pinv` / `vt-lstsq` | 基于 SVD | `A⁺ = V·Σ⁺·U^H` |
| `vt-matrix-rank` / `vt-cond` / `vt-matrix-power` | SVD / 复用 | rank 改用奇异值（原实现取实部） |

### 修掉的「静默丢虚部」（8 处）

| 函数 | 修复前 | numpy | 根因 |
|---|---|---|---|
| `vt-cov` / `vt-corrcoef` | float64 | 复厄米特 | `%e3-as-1d-float-transform` 首行 `ensure-vt :float64` |
| `vt-expm1` / `vt-log1p` | float64 | 复数 | `%float-map` 只认实数 |
| `vt-nanpercentile` / `vt-nanquantile` / `vt-interp` | float64 | **报错** | `astype :float64` 取实部 |
| `vt-fromiter` | float64 | 复数 | 默认 `:dtype :float64` |
| `vt-nextafter` 等 9 个 ufunc | 静默算实部 | **报错** | 无复数分派 |
| `vt-matrix-rank` | 取实部消元 | 奇异值 | 同上 |
| `vt-scale`（公共函数） | complex×实数标量 → dtype 判成 float64，写缓冲报 `%DOUBLE-FLOAT` | 保持 complex | dtype 推断只列 4 种实数类型 |

### 新增/放开的算子

- **新支持**：`vt-expm1` `vt-log1p` `vt-sinc` `vt-sign`（z/|z|）
  `vt-clip`/`vt-clamp`（**字典序**整元素夹取）`vt-standardize`
  `vt-cov`/`vt-corrcoef`（`X·X^H`）`vt-fromiter`（dtype 推断）
  `vt-einsum`（放开白名单，走通用累加路径）
- **新拒绝**（numpy 亦拒绝）：`nextafter` `spacing` `logaddexp` `hypot`
  `copysign` `signbit` `deg2rad` `rad2deg` `bincount` `histogram`
  `nanpercentile` `nanquantile` `interp`

### 三个「只在非方阵/复数上暴露」的真实坑

1. **约化 QR 的 Q 不能截断后顺次右乘** —— 反射器支撑长度 `m-i` 通常超过
   `k = min(m,n)` 列，内积会读不存在的列。方阵/宽矩阵因为 Q 恰好是完整
   `m×m` **碰巧正确**，只有 tall（m>n）错。正解：`Q = H_0···H_{k-1}·I_{m×k}`，
   从 `i = k-1` **递减左乘**。
2. **`ld`（行距）≠ 行数** —— 行主序 m×n 的行距是 **n**。本轮因写错 `ld`
   踩了 3 次（`vt-lu` 非方阵、QR 的 Q、SVD 的 U），全靠 `(safety 1)` 的
   边界检查当场报错才定位到。
3. **Cholesky 的 `:upper` 必须是 `L^H`** —— 实数下 `L^T = L^H`，
   复数下写成 `transpose` 会静默返回 `L^T`。

另：`vt-eig` 的对角虚部必须清零，否则非厄米输入会收敛到**错的特征值**
（`[[2+1j,1],[1-1j,3+2j]]` 给出 3.925/1.075，numpy 是 4/1）。

### 文档

- `CONVENTIONS.md` **§16.19 线性代数与算子补齐**：共轭转置这条唯一真分叉、
  LU/Cholesky/Householder/Jacobi 的复公式、约化 QR 陷阱、对角虚部清零原因、
  8 处静默错答案清单、已知差异 **D-C9..D-C13**（histogram 拒绝 / eig 是 eigh 语义 /
  SVD·QR 向量带相位自由度 / lstsq residual 是 2-范数）、声明策略

### 测试

- `test/complex-linalg-test.lisp` **108 断言** = 46 项 numpy 2.4.6 差分对照
  + 结构恒等式（`P·A=L·U`、`L·L^H=A`、`Q·R=A`、`Q^H Q=I`、`U·S·V^H=A`、
  `U^H U=I`、`V^H V=I`、`A·V=V·Λ`、Moore–Penrose 四条、`einsum ≡ 专用内核`）
  + 契约（`:out`、奇异/非方阵/非正定报错、dtype、实数回归网）
- `test/gen_complex_linalg_expected.py` 生成 `test/complex-linalg-expected.sexp`
  （禁手写期望值；线代比较用相对容差 1e-9，逐位相等无意义）
- 已接入 `test/run-tests.sh`

### 回归

- complex-linalg 108/108、complex 136/136、complex-ops 131/131、core 284/284、
  test-all 50/50（642 断言）、io-interop 77/77、differential-probes 496/496、
  probes2 405/405、extensions3 264/264、legacy 284/284、numpy-convention 69/69、
  contract-regression 136/136、out-contract-v2 56/56、nan-broadcast 31/31 —— 全绿
- 过程中差分探针 **p0109/p0110** 抓到 `vt-matrix-rank` 的新增 `&optional tol`
  把实数默认值 1e-10 覆盖成 NIL 的回归，已修

---

## 2026-10-20（二）— 修复：`*read-default-float-format*` = double-float 时 complex64 崩溃

### 症状（用户环境 SBCL 2.6.9.446 / numpy 2.5.0）

```
差分对照完成：PASS=97 FAIL=0
--- 契约断言 ---
Unhandled TYPE-ERROR ... The value #C(0.0 0.0) is not of type (COMPLEX SINGLE-FLOAT)
                            when binding SB-C::ITEM
```

### 根因（三段链，已用用户同版本 SBCL 精确复现）

1. 用户 `.sbclrc` 把 `*read-default-float-format*` 设为 `double-float`；
   源码里的**裸浮点字面量**（`0.0`/`0.5`/`180.0`）精度由读取时该变量决定。
2. 于是 `(complex 0.0 0.0)`（complex64 的零值）被读成 `(complex 0.0d0 0.0d0)`。
3. 往 `(simple-array (complex single-float) (*))` 填 double 复数时，SBCL 的
   `vector-fill` 为每种元素类型预装的转换器
   `(lambda (sb-c::item) (declare (type (complex single-float) sb-c::item)) ...)`
   抛 TYPE-ERROR —— 报错里的 `#C(0.0 0.0)` 正是 double 在该读取格式下的打印形式，
   具有极强误导性。（`(safety 0)` 旁路更糟：不报错，直接写错位模式。）

### 修复（三层）

- **`clvt.asd`**：新增 `:around-compile`，在编译/加载每个组件时绑定
  `*read-default-float-format*` = `single-float`（read 发生在该动态范围内），
  从根上消除对用户 init 的依赖。
- **源码**：复数/单精度关键路径的裸字面量显式加 `s0`
  （`(complex 0.0s0 0.0s0)`、`0.5s0`、`2.0s0`、`180.0s0`、`0.6931472s0`、
  `0.31415927s1`、`(cval 0.0s0)`、`(/ 1.0s0 norm)`、`0.5s0` 直方图边界 等），
  即使绕过 ASDF 直接 `load` 源文件也不漂移。
- **测试**：`run-tests.sh` 两处 `sbcl` 调用插入
  `--eval "(setf *read-default-float-format* 'single-float)"`；
  `complex-ops-test` 增加 3 条「读取格式哨兵」断言
  （`(type-of (vt-dtype-default-value :complex64))` 等），复发时立刻报红。

### 文档

- `CONVENTIONS.md` §16.18.6：完整复盘（症状/根因链/三层修复/用户侧 fasl 缓存提醒），
  并固化约定：**源码里的裸浮点字面量一律视为 single-float；double 必须写 `d0`**。

### 测试

- 在**用户同版本** SBCL（2.6.9.446-9e249df9d）上复现并验证修复
- double-float 读取环境：complex-ops 131/131、complex 136/136、core 284/284
- 默认环境：全量 13 套件全绿

> 用户侧注意：更新后需 `(asdf:load-system :clvt :force t)`（或清
> `~/.cache/common-lisp`），否则可能继续加载用 double 编译的旧 fasl。

---

## 2026-10-20 — 第二轮进化：复数数据搬运层（转换 / 拼接 / 剪切 / 集合 / 缺失值 / 缩并）

第一轮（§16）解决逐元素算术与归约；本轮补结构层，新增 `src/complex-ops.lisp`
（紧随 `src/complex.lisp` 加载，同样「实数路径零改动 + 只对含复数调用分派」）。

### 修掉的「静默错答案」（比不支持更危险）

| 函数 | 修复前 | numpy 语义 |
|---|---|---|
| `vt-nanmean` | 返回**实数**（实部均值） | 复数 |
| `vt-nanvar` / `vt-nanstd` | 数值错误 | 实数 = mean(\|x−mean\|²) |
| `vt-nancumsum` / `vt-nancumprod` | **虚部清零** | 保复数（NaN→0/1 后累积） |
| `vt-vdot` | 不取共轭 | **第一操作数取共轭** |
| `vt-cumsum/cumprod` 的 `axis=nil` | 形状不展平（2-D 进 2-D 出） | 展平为 1-D |

### 新增能力

**转换**
- `vt-nan-to-num`：实部/虚部各自替换（`nan`→0、`posinf/neginf`→该精度 ±最大有限值）
- `vt-view` 支持 `:dtype` 重解释：`complex128 ↔ float64`、`complex64 ↔ float32`，
  shape 省略时按 numpy 规则只缩放最后一维；三种调用形式
  `(vt-view a shape)` / `(vt-view a shape :dtype dt)` / `(vt-view a :dtype dt)`
- `vt-repeat`：三处 8 路元素类型分派各追加 2 个复数分支（标量/列表 repeats、任意 axis）
- 持久化：npy 新增 `<c16` / `<c8` 编解码（与 numpy 双向互通，含 NaN payload）；
  lisp 格式复数编码（分量全有限 → `#C(re im)`，否则 `(:CX re im)` 保证可读性）

**集合 / 搜索**
- `vt-unique`（含 `return-index/inverse/counts`）、`union1d`、`intersect1d`、
  `setdiff1d`、`setxor1d`、`in1d`、`isin`、`searchsorted`（`:left`/`:right`）
- 统一走「字典序排序 + `%cx-key` 哈希」：NaN 折叠为一个、`1.0s0` 与 `1.0d0` 同键
- `vt-digitize` 对复数给出确定性错误（numpy 亦报 `x may not be complex`）

**缺失值（nan 族）**：`nansum/nanprod/nanmax/nanmin/nanargmax/nanargmin/nanmean/nanvar/nanstd/nanmedian`
- 全 NaN 切片：sum→0、prod→1、max/min→`NaN+0j`、mean→`NaN+NaNj`、var/std→`NaN`、arg→报错
- 实现为「掩码 + 单位元替换 + 既有复数归约内核」的向量化写法（一次覆盖所有 axis/keepdims）

**缩并 / 卷积**：`vt-inner`、`vt-tensordot`（transpose+reshape+复数 matmul）、
`vt-correlate`（第二操作数**取共轭**）、`vt-convolve`（**不取共轭**，对 flip(v) 相关）

### 文档
- `CONVENTIONS.md` **§16.18 数据搬运层**：静默错答案清单、语义总表（按 numpy 2.4.6 实测）、
  已知差异 **D-C7**（view 重解释是副本，非共享视图）、**D-C8**（take-along-axis 同秩约束）、
  声明策略（能用既有内核组合就不新写循环）、测试口径
- §16.17 路线图：标注 `.npy` 复数互操作已完成；§16.19 变更文件清单补齐

### 测试
- `test/complex-ops-test.lisp` **128 断言** = 97 项 numpy 2.4.6 差分用例 + 31 条契约断言
  （dtype 提升 / 切片别名 / `:out` 硬契约 / 未支持组合报错 / NaN 折叠 / c64 精度 /
  两种格式往返 / 7 条实数回归网）
- `test/gen_complex_ops_expected.py` 生成 `test/complex-ops-expected.sexp`（禁手写期望值）
- 已接入 `test/run-tests.sh`（`SUITES` + `DEFAULT_ORDER`）

### 回归
- complex 136/136、complex-ops 128/128、core 284/284、io-interop 77/77、test-all 50/50 全绿

---

## 2026-10-19（三）— 复数层性能优化（第一轮进化 v1.1）

优化目标：把「第一轮只为正确性、全部走泛型 walker」的复数实现拉到接近实数快路径，
同时**不引入无脑 `(speed 3) (safety 0)`**。1M 元素实测（`test/complex-bench.lisp`）：

| 算子 | 优化前 | 优化后 | 倍数 |
|---|---:|---:|---:|
| add | 135 ms | 7 ms | 19× |
| mul | 157 ms | 7 ms | 22× |
| div | 184 ms | 8 ms | 23× |
| sqrt | 453 ms | 20 ms | 23× |
| abs | 889 ms | 26 ms | 34× |
| sum | 92 ms | 1 ms | 90× |
| cumsum | 288 ms | 6 ms | 48× |
| sort | 2316 ms | 136 ms | 17× |
| compare `<` | 778 ms | 3 ms | 260× |
| isnan | 402 ms | 2 ms | 200× |
| matmul 256² | 1757 ms | 39 ms | 45× |
| argmax | 912 ms | 5 ms | 173× |

### 改动

**src/map-reduce.lisp**
- 新增 `+cx-elem-types+` + `+cx-inline-ops+` 白名单与 `%cx-inline-op-p`；
  `%elem-type-cond(-strided)`、`%vt-inline{1,2,3}[-strided]-fast`、`%vt-mask{1,2}[‑strided]‑fast`
  在**算子命中白名单**时追加 2 个复数分支 → 复数算术/数学/比较/谓词进入
  「类型声明驱动 + 算子内联」的循环（原来一律落到 `vt-map` 的 funcall walker）。
  白名单机制避免 200+ 展开点被 10 路广播撑爆（历史教训：8×8 全展开 Heap exhausted）。

**src/dtype.lisp**
- 新增免屏蔽快判 `%cx-nan-p` / `%cx-inf-p` / `%cx-finite-p`：内核在
  `with-float-safe` 之下运行，不必每元素再屏蔽/恢复陷阱
  （`vt-float-nan-p` 单次约 200ns，每元素判 2~4 次 → 1M 元素白付 400~800ms）。
- `%c-lex-compare` / `%clt` / `%cle` / `%cgt` / `%cge` / `%cneq` 全部 **inline**
  （必须写在 `defun` 之前，SBCL 才记录内联展开）。

**src/complex.lisp**
- `%cx-def-unary` / `%cx-def-compare` / `%cx-def-predicate` / `vt-conj` 改为走
  `vt-fast-map` 宏（原来走 `vt-map` 函数，编译期拿不到算子字面量 → 无法内联）。
- 新增 `%def-cx-to-real-run`（complex→real 专用内核，`(speed 3) (safety 1)`）+
  `%cx-abs-run` / `%cx-real-run` / `%cx-imag-run` / `%cx-angle-run`：
  abs/real/imag/angle 是跨 dtype 映射，进不了 `dtype-check` 快路径。
- 归约重写：`%cx-dispatch`（按元素类型绑定 simple-array）+ `%def-cx-reducer`
  （全局单循环 / 单轴 fiber 循环）+ `%cx-arg-reduce-run`；多轴仍走通用 odometer 兜底。
- 扫描 `%cx-scan-run`、矩阵乘 `%cx-matmul-run`（i-k-j 次序，内层连续）类型特化。
- 排序重写：`%def-cx-fiber-sorter`（插入排序 + 自底向上归并，比较器内联、零 cons），
  `%cx-sort-fiber-strided` 用「收拢→排→写回」处理任意步长 fiber；
  argsort 改为 fixnum 下标向量 + STABLE-SORT（不再每元素 cons 一个 (值 . 下标)）。
- `%cop-sqrt` 换成 C99 公式（hypot + 1 sqrt + 2 除法，350ns → 20ns/元素），
  `%cop-abs` 加常规量级快路径（极端量级才走带缩放的 hypot）。
- `%cx-reduce` / `%cx-scan` / `%cx-sort-axis` / `%cx-arg-reduce` 外层统一
  `with-float-safe`（配合内核里的免屏蔽快判）。

### 声明策略（写进 CONVENTIONS §16.14.1）

| 层级 | 声明 | 理由 |
|---|---|---|
| 用户入口 / 边界 | 默认安全级 | 契约检查与错误信息都在这里 |
| 普通内核（`%cx-*`） | `(speed 3) (safety 1)` | 索引可证不越界，但保留检查实测 < 5%，避免静默踩内存 |
| 宏展开逐元素循环（map-reduce） | `(speed 3) (safety 0)`（既有） | 索引由循环上界逐元素推进，编译器可完全证明 |
| 通用 stride 路径 | 禁止 `safety 0` | 用户 stride/shape 错误必须是 Lisp 条件 |

### 测试

- 新增 `test/complex-bench.lisp`（1M 元素 × 14 项，含 float64 对照）并登记为基准套件
  （`SUITES[complex-bench]` + `BENCH_SUITES`，默认跳过、`--all` 时运行）。
- 回归全绿：complex-test 136/136、core-test 284/284、differential-probes 496/496、
  differential-probes-test2 405/405、extensions3 264/264、legacy 284/284、
  io-interop 77/77、nan-broadcast 31/31、test-all 50/50。

---

## 2026-10-19（二）— 修复：新增导出导致 p0395/p0392 失败 + codegen 门禁静默失效

### 现象
上游合并复数补丁后 `differential-probes-test2` 恒红：`403/405`，`p0392`（data）与
`p0395`（data）。

### 根因（四条，全部实测确认）

1. **`p0395` 把导出数量写死成常量 371**，而运行时 `*vt-fun-list*` 只有 367：
   `vt-real / vt-imag / vt-conj / vt-angle` 早在 `extensions3.lisp` 就存在并已导出，
   复数层又**重复导出**了一次。`defpackage` 允许重复，静态解析因此多算 4 个。
2. **`refresh-vt-fun-list` 用 `DO-SYMBOLS`**（把内部辅助符号也算进来），与其
   docstring「导出符号」矛盾，且结果依赖"何时调用"——运行时重刷一次就会从 367 涨到 406。
3. **`p0392` 是真语义 bug**：`%cimag` 的实数分支写成 `(* 0 x)`，对负数产生 **-0.0**，
   而 numpy 的 `np.imag(实数)` 恒为 **+0.0**；探针 `:exact` 做位级比较（含符号位）故判不等。
   同源的 `%carg` 也把 `-0.0` 漏判成 0（numpy: `angle(-0.0) = π`）。
4. **codegen 门禁静默失效**：`check_key_sync.py` 用「无参调用生成器」做漂移检查，
   而两个生成器都是 `sys.argv[1]` → 无参即 `IndexError`，文件未被重写，比对自然"通过"。
   后果：9 行过期浮点期望值（旧 `str()` 口径 `0.272799`，现应为 `0.27279899999999996`）
   一直没被发现。

### 修复

- `src/dtype.lisp`：`%cimag` 实数分支返回同类 **+0.0**；`%carg` 增加 NaN 传播与
  **符号零**判定（`float-sign` 判 `-0.0` → π，返回 +0.0）。
- `src/package.lisp`：删除 4 个重复导出并加注释；`refresh-vt-fun-list` 改用
  `DO-EXTERNAL-SYMBOLS`（口径唯一、与调用时机无关）。
- `test/gen_probes*.py`：无参默认 `gen`（门禁真正生效）；`p0395` 由"数量常量"
  改为**不变量断言**（列表 ≡ 导出集、无重复、非空），新增导出不再变红；
  删除失效的 `_static_fun_count`。重新生成 `differential-probes-test2.lisp`
  （顺带把 9 行过期浮点期望值规范化）。
- `test/README-tools.md`：新增「codegen 门禁的两条硬约束」与「新增导出函数怎么操作」。
- `CONVENTIONS.md` §10.4.1：把 API 指纹规则固化（禁写死数量 / 口径唯一 /
  导出不重复 / 无参=gen / 生成可复现 / 重新生成后先 add）。
- `.gitignore`：忽略 `tmp/`（生成器中间产物）。

### 验证

- `differential-probes-test2`：**405/405**（原 403/405）
- `complex-test` 136/136、`core-test` 284/284、`differential-probes-test` 496/496
- `check_key_sync.py` 全绿且**真实生效**（人为制造漂移可被测出）
- 不变量探针非空转：列表污染 / 含重复 / 为空 / 晚调用 refresh 四种破坏各自返回 0

---

## 2026-10-19 — 第一轮进化：复数（complex64 / complex128）支持

### 新增 dtype（src/dtype.lisp）
- `:complex128` / `:complex64`（CL 元素类型 `(complex double-float)` / `(complex single-float)`，
  itemsize 16 / 8，interleaved 存储；strides 仍以**元素**为单位）
- 提升规则：**复数的实部分量参与普通实数提升**，结果映射回 complex64/complex128。
  该单一规则复现 numpy 2.4.6 的 **100 格**提升表（含 `complex64 ⊔ int32 → complex128`）
- 值级原语（唯一语义出口）：`%creal %cimag %cabs %carg %cconj %cisnan %cisinf
  %cisfinite %cis-zero %clt %cle %cgt %cge %cneq`（字典序比较 + NaN 规则）
- `vt-cast` / `vt-cast-fun`：complex→real/int **取实部**（对标 numpy astype 的
  ComplexWarning 语义，本库不产生警告）；real→complex 虚部补 0

### 构造与推断（src/core.lisp、src/io.lisp）
- `ensure-vt`：复数标量 → complex64/complex128（按分量宽度）；含复数的列表 → complex128
- `make-vt` / `vt-zeros` / `vt-ones` / `vt-const` 支持复数 dtype
- 打印：`1.0+2.0j` 形式（`%format-number` 新增 :complex 分类）

### 语义层（src/complex.lisp，新增，最后加载）
- 分派机制：加载时捕获各函数的**原实现**（`%cx-orig`，预捕获以避免自我递归），
  仅「输入含复数」的调用进复数内核，实数路径零改动
- 一元数学 21 个（sqrt/exp/log/log2/log10/三角/双曲/反三角/反双曲/square/
  reciprocal/signum/negative）+ 幂（复指数、`0**0=1`、`0**负=nan+nanj`）
- 分量族：`vt-abs`（→实数 dtype）、`vt-real` / `vt-imag`（副本）、`vt-conj` /
  `vt-conjugate`、`vt-angle`、`vt-complex`、`vt-iscomplexobj` / `vt-isrealobj`
- 比较：`== != < <= > >=` 全部**字典序**（numpy 2.x 语义），NaN 参与恒假
- 极值/排序：`maximum/minimum/fmax/fmin`、`sort/argsort`（字典序 + NaN 末尾）、`median`
- 判定：`isnan/isinf/isfinite`（任一分量为真即真）
- 归约：`sum/prod/mean`（保复数 dtype）、`var/std/norm`（**恒实数**）、
  `argmax/argmin`（int64）、`cumsum/cumprod`、`trace`、独立 stride 驱动内核
- 线代：`matmul/@/dot/outer` 专用内核（不复用 einsum —— 其 dtype 白名单只含 4 种实数类型）
- 未支持即报错：floor/ceil/trunc/rint、floor_divide/mod/rem、位运算、atan2、
  even/odd-p、percentile/quantile、einsum、LU/SVD/eig/inv/solve/cholesky/qr、
  神经网络激活（均给出确定性错误，绝不静默丢虚部）

### 文档
- `CONVENTIONS.md` 新增 **§16 复数（complex）约定**（约 380 行）：物理表示、
  100 格提升表与形式化规则、casting、分支切割（SBCL 与 numpy 逐点对照）、
  字典序比较、NaN/Inf、归约、线代、打印、错误契约、执行层定位、已知差异 D-C1..D-C6、
  第二轮路线图
- 记录 SBCL 平台陷阱：`&rest` + `(safety 0)` + call-site `apply` 三者同时成立会段错误

### 测试
- `test/complex-test.lisp`（**136 断言**）+ `test/gen_complex_expected.py`
  （numpy 2.4.6 实测生成 **83 项**差分期望，`test/complex-expected.sexp`）
- 已接入 `test/run-tests.sh`（SUITES + DEFAULT_ORDER，防漏接线）

### 回归
- core-test 284/284、run_all_tests 143/143、nan-broadcast-test 31/31、
  numpy-convention-tests 69/69、contract-regression-test 136/136、
  out-contract-v2-test 56/56 全部通过（实数路径未受影响）

---

## 2026-10-10 测试工具链防漂移 + 遗产用例盘活 + vt-narrow 负索引

### vt-narrow 语义修复（src/manip.lisp）
- `vt-narrow` 的 start/end 支持 **numpy 切片负索引语义**：负值先加轴长，
  规约后自动 clip 到 `[0, dim-size]`（此前负索引/越界直接报错）。
  规约后 `end < start` 仍报错。core-test 新增 52b/52c 两条断言。

### 测试工具链
- 删除旧架构孤儿生成器 `gen_all_tests.py` / `gen_param_tests.py`
  （静态 JSON 迁移遗留，产出物无任何消费者，且旧 key 风格易误导维护者）
- 新增 `test/check_key_sync.py` 防漂移门禁并挂入 `run-tests.sh` 前导：
  5 套件查询 key 与 ref_compute 供给机械核对（缺失→FAIL+报因，死供给→WARN），
  探针 codegen 与提交 .lisp 逐字节校验
- 遗产盘活：架构迁移时仅 280/560 例被接线，其余 281 例从未被任何套件消费。
  现全部恢复：oracle 并入 `ref_compute.py`（651 key，0 缺失 0 死供给），
  lisp 调用补入新套件 `test/legacy-coverage-test.lisp`（284/284 通过）
- 新增 `test/README-tools.md` 工具链地图（各 py/lisp 角色、消费方、修改清单）
- 盘活中确认的语义差异（已在 README-tools.md 登记）：
  `vt-det` 暂不支持 3D 批量；`vt-diagonal` 3D 为 pytorch 末两维风格；
  `vt-gradient` 1D 返回裸 VT

---

## 2026-10-06（四）— 新增 `vt-cross-entropy-logits`：真正的 torch.nn.CrossEntropyLoss 语义

### 问题

`vt-cross-entropy` 的文档字符串声称「对标 torch.nn.CrossEntropyLoss」，
但实现是**概率/one-hot 输入**语义（`-Σ y·log(p)`，且对 p 做
`[eps, 1-eps]` 裁剪）。两者并不一致：

- torch 的 `CrossEntropyLoss` 收 **raw logits + 整数类别索引**，
  内部做 log-softmax；
- 该函数收 **概率 + one-hot**，且 `vt-clip(p, eps, 1-eps)` 对 logits
  完全不适用（logits 可为负、可远大于 1）。

以「对标 torch.nn.CrossEntropyLoss」的写法调用会因形状不可广播而
报错，例如：

```lisp
(vt-cross-entropy (vt-asarray '((1.0d0 2.0d0 3.0d0) (2.0d0 0.0d0 1.0d0)))
                  (vt-asarray '(0 1) :dtype :int64))
;; => error: 形状 (2 3) 和 (2) 无法广播：维度 3 与 2 不兼容
```

### 修复

**新增 `vt-cross-entropy-logits`**（`src/nn.lisp`，已导出），真正对标
`torch.nn.CrossEntropyLoss`：

| 参数 | 说明 |
|---|---|
| `logits` | 未归一化得分，沿 `axis` 为类别维（缺省 -1） |
| `labels` | **整数**类别索引，形状 = logits 去掉类别轴（`(N,C)` 配 `(N,)`；支持任意秩 `(N1,N2,...,C)` 配 `(N1,N2,...)`；也支持无批次 `(C,)` 配标量/长度 1） |
| `axis` | 类别轴，缺省 -1 |
| `reduction` | `:mean`（缺省）/ `:sum` / `:none` |

关键设计：

- **数值稳定**：内部用 log-softmax（减该轴最大值 + log-sum-exp），
  大 logits 不溢出（`logits=(1e4,0,0)` 仍精确得 `0.0d0`）；
- **不做 `[eps, 1-eps]` 裁剪**——logits 语义下裁剪会破坏数值；
- **类别索引越界即报错**（含负数索引），非静默取值；
- 浮点 `labels` 明确报错（要求整型，对标 torch 的 int64 labels）；
- dtype 缺省按 logits 提升：float32 → float32，其余 → float64。

示例（与 `torch.nn.CrossEntropyLoss` 逐位一致）：

```lisp
(vt-cross-entropy-logits (vt-asarray '((1.0d0 2.0d0 3.0d0) (2.0d0 0.0d0 1.0d0)))
                         (vt-asarray '(0 1) :dtype :int64))
;; => 2.4076059644443806d0
```

**`vt-cross-entropy` 语义保持不变**（现有 5 处调用依赖其概率语义，
含 `example/example.lisp` 的 one-hot 用法），仅**修正文档字符串**，
明确它是概率输入并指向新函数，消除「对标」与实现不一致的误导。

### 测试

- `test/extensions3-test.lisp`：新增 18 条 `CE-LOGITS` 断言
  （用户原文调用、reduction 三态、任意秩、axis、数值稳定、
  与 softmax+log 对照、dtype、`:out`、5 条错误路径、
  旧接口未受影响），套件 243 → **261** 条全绿。
- `example/example.lisp`：`test-activation-loss` 补 `vt-cross-entropy-logits` 用例。
- 全量 **31/31 套件通过，0 失败**。

### 约定文档同步（避免后人重踩）

- `CONVENTIONS.md` **新增 §3.4「神经网络的浮点语义」**：明确 nn 激活/损失函数
  **不得**走通用提升规则（整数进整数出会静默截断），统一规则为
  float32→float32、其余（含整数）→float64，并记录根因与实现要求
  （必须经 `%nn-float-dtype`/`%nn-as-float`，不得依赖 `vt-map` 的 `dtype=nil`）。
- `CONVENTIONS.md` **新增 §3.5「两种交叉熵语义（不可混用）」**：并列
  `vt-cross-entropy`（概率/one-hot）与 `vt-cross-entropy-logits`（raw logits，
  对标 torch），标注易错点与 `vt-cross-entropy-logits` 的完整契约。
- `CONVENTIONS.md` §9.2 核对清单补「激活函数族 / `vt-cross-entropy` /
  `vt-cross-entropy-logits`」三行。
- `README.md` 损失函数清单补 `vt-cross-entropy-logits` 并注明两者语义差异。

---
## 2026-10-06（三）— 第二轮审查核实：修复 nn.lisp 整数提升链（softplus/gelu/leaky-relu）+ 澄清 R-1/R-2

对第二轮审查报告逐条在最新 `master`（`471d005`）上实测复核。

### 真实缺陷（已修复）

| # | 位置 | 缺陷 | 修法 |
|---|------|------|------|
| R-3 | `vt-softplus` | 整数输入返回**整数** dtype 且数值被静默截断。`softplus([0])` → `0`（应 `0.693…`）；`softplus([-1])` → `0`（应 `0.313…`）；`softplus([-5,0,5,25])` → `(0 0 5 25)`（应 `(0.007 0.693 5.007 25.0)`）；配 `:out float64` 直接报 dtype 不匹配 | `%nn-float-dtype` 推导浮点结果 dtype（float32→float32，其余含整数→float64），`%nn-as-float` 先把输入提升再映射 |
| R-3b | `vt-gelu` | **报告未提及**。整数输入同样返回整数 dtype 且数值截断：`gelu([1,2,3])` → `(0 1 2)`（应 `(0.841 1.955 2.996)`） | 同上 |
| R-3c | `vt-leaky-relu` | **报告未提及**。整数输入负值被截断：`leaky-relu([-2,3,-4])` → `(0 3 0)`（应 `(-0.02 3.0 -0.04)`） | 同上 |

> 三个函数同属一条「整数提升缺失链」：通用路径把 lambda 直接交给
> `vt-map` 且 `dtype=nil`，`vt-map` 走 `vt-promote-type` → 整数进整数出，
> lambda 内算出的浮点值被写回整数存储而静默截断。
> 这是 P1-1（elementwise 数学函数）在 `nn.lisp` 的**同类漏网**。

已排查并**确认无此问题**的同族函数（补断言守护）：
`vt-sigmoid` / `vt-swish` / `vt-mish` / `vt-hard-tanh` / `vt-hard-sigmoid` /
`vt-binary-cross-entropy`（均正确返回 float64）。

### 经核实为「误报」或「已修复」（未改动）

| 报告条目 | 报告结论 | 实际核实 |
|---|---|---|
| P0-1 `vt-relu` 整数段错误 | 仍存在，段错误崩溃 | **已修复**（`471d005`）：实测 `(vt-relu (vt-asarray '(1 2 3 4)))` → `(:FLOAT64 (1.0 2.0 3.0 4.0))`，int8/int32/uint8/负值/float32 全部正常 |
| P1-1 整数数学函数 float32 精度 | 15 项失败 | **已修复**（`471d005`）：`sin int64 → 0.8414709848078965d0`，与 numpy 逐位一致 |
| P1-2 `vt-expm1`/`vt-log1p` 不提升 | 返回 int64 | **已修复**（`471d005`）：两者整数输入均返回 `:FLOAT64` |
| **R-1** 非连续 `:out` 写入不落盘 | 维持原判，称「上游测试只验证通过 out 读回，没验证通过 base 读回」 | **误报，且报告对上游测试的描述与事实相反**。实测：`(vt-add m23 m23 :out view)` 后 **`base` 正确落盘**为 `((0 0 2 0 4 0) (6 0 8 0 10 0))`。`test/out-contig-tests.lisp` 中 `"2b base 偶数位被写入"`、`"2c base 奇数位保持 0"`、`"3c 写入位置正确"`、`"3d 未写位置保持 0"` 等断言**全部作用在 `base` 上**，正是报告声称「缺失」的那类验证 |
| **R-2** `vt-cross-entropy` 不支持分类调用 | P1 新发现，且「传 3 元素 labels 静默给出错误数值 56.41（正确 4.815）」 | **非缺陷，属设计取向**。该函数文档与测试均明确为**概率输入**（`test/test-all.lisp`：`"vt-cross-entropy 同 bce（概率输入）"`），与 `vt-softmax` 配套使用。报告要求的 raw-logits 语义属**功能诉求**（可考虑新增 `vt-cross-entropy-logits`），非「实现错误」。另：报告给出的数值 `4.815`/`56.41` 用 one-hot 语义（`1.4076`）与广播语义（`0.4076`）**均无法复现**，数值来源不可信 |

### 测试

- `test/extensions3-test.lisp`：新增 21 条「R-3 整数提升回归」断言
  （softplus dtype×3 + 数值×5 + `:out`×1；gelu dtype×3 + 数值×1；
  leaky-relu dtype×1 + 负值不截断×1 + float32 保持×1；同族守护×5），
  套件由 222 → **243** 条全绿。
- 全量 **31/31 套件通过，0 失败**（含 `fastmap-invariant-test`）。

---
## 2026-10-06（二）— 修复 P1-1 精度修复引入的性能回归：`%float-map` 宏化，恢复 `vt-fast-map` 内联快路径

上一轮为修复 P1-1（整数输入仅 float32 精度）引入了函数 `%float-map-fn`，
它无条件把输入交给 `vt-map`。**这是一个性能回归**：`vt-fast-map` 是宏，
编译期把字面算子（`#'sin`）内联成特化循环；而 `vt-map` 经 `funcall` 调用，
单元素迭代器慢约 **1.6–1.8×**。结果是连**本就无需提升的连续 float64/float32
输入**也被拖到慢路径上。

实测（100k 元素 × 1000 次，SBCL 2.6.8）：

| 调用 | 回归态（`%float-map-fn`） | 修复后（`%float-map` 宏） | 原始基线 |
|---|---:|---:|---:|
| `vt-sin` float64 | 2804 ms | **1182 ms** | 1206 ms |
| `vt-exp` float64 | 1216 ms | **623 ms** | 614 ms |
| `vt-sin` int64 | 2514 ms | **1287 ms** | — |
| `vt-fast-map #'sin` 裸调用参考 | — | 1205 ms | — |
| `vt-map #'sin` 裸调用参考 | — | 2207 ms | — |

### 修法

把 `%float-map-fn` 拆分/替换为两个宏，**dtype 判定外提为运行时分枝，
但两条分支都在编译期展开为快路径**：

| 新入口 | 适用算子 | 展开后的快路径 |
|---|---|---|
| `%float-map` | `(function <symbol>)` 字面算子（`#'sin`/`#'cos`/`#'asinh`/`#'exp`/`#'%e3-expm1`/`#'%e3-log1p` 等） | **`vt-fast-map` 内联循环**（已是目标浮点 → 零拷贝直通；整数 → 先 astype 再内联） |
| `%float-map-lambda` | lambda 算子（`vt-asin`/`vt-acos`/`vt-acosh`/`vt-atanh`/`vt-sqrt` 需逐元素判 NaN） | `vt-map`（`vt-fast-map` 无法内联 lambda）+ 同样的 dtype 提升分支 |
| `%float-map-fn`（保留为兜底） | 任意运行时 FN | `vt-map` |

两宏展开形如：

```lisp
(let* ((%in (ensure-vt vt)) (%dt dt))
  (if (eq (vt-dtype %in) %dt)        ; 已浮点：直通，不做多余 astype
      (vt-fast-map #'op %in :out out :dtype %dt)
      (vt-fast-map #'op (%coerce-float-input %in %dt) :out out :dtype %dt)))
```

改动点：

- `src/elementwise.lisp`：新增 `%float-map`（宏，带 `(function <symbol>)`
  入参校验）、`%float-map-lambda`（宏）；`%float-map-fn` 降级为文档化的
  兜底入口。9 个字面算子调用点（sin/cos/tan/atan/sinh/cosh/tanh/asinh/exp）
  改用 `%float-map`，5 个 lambda 调用点（asin/acos/acosh/atanh/sqrt）改用
  `%float-map-lambda`。
- `src/extensions3.lisp`：`vt-expm1` / `vt-log1p` 由手写
  `vt-map + 预 astype` 改为 `%float-map #'%e3-expm1 / #'%e3-log1p`，
  同样拿到内联快路径，并消除「调用方重复 astype」的多余拷贝。

### 正确性

P1-1 精度契约完好：`(vt-sin (vt-asarray '(1 2 3 4)))` 第 0 元素
`0.8414709848078965d0`（float64 全精度）；`vt-expm1`/`vt-log1p`
整数输入仍返回 `:float64`。全量测试 **30/30 套件通过，0 失败**。

---
## 2026-10-06（一）— 第三方审查报告核实：修复 6 类真实缺陷 + 澄清 4 项报告误报

对 `clvt-review-report.md`（对照 numpy 2.5.3 / SBCL 2.6.8）逐条复核。
报告称 2 个 P0、2 类 P1、一批 P2；经在当前 `master` 上以 numpy 2.4.6
实测复现，**确认其中 6 类为真实缺陷并修复，另有 4 项经核实为报告误报
或已在 v0.4.0 修复**。

### 真实缺陷（已修复）

| # | 位置 | 缺陷 | 修法 |
|---|------|------|------|
| P0-1 | `vt-relu` | **段错误**：整数输入走通用路径，把 int64 元素喂给声明 `(double-float x)(safety 0)` 的 `%relu-double`，fixnum 被当 boxed double 解引用非法地址，SBCL 镜像崩溃 | 通用路径改为无类型假设的 `lambda` + 先 `vt-astype` 提升到目标浮点 dtype |
| P1-1 | `vt-sin/cos/tan/asin/acos/atan/sinh/cosh/tanh/asinh/acosh/atanh/exp/log/sqrt` | 整数输入数值仅 **float32 精度**：CL 对整数参数 `(sin 1)` 只返回 single-float（0.84147096），却被写入标称 float64 的结果 | 新增 `%coerce-float-input` / `%float-map-fn`，映射前把整数输入提升到目标浮点 dtype（float32 输入仍保持 float32） |
| P1-2 | `vt-expm1` / `vt-log1p` | 整数输入**不提升**，返回被截断的 int64（`expm1([1,2,3])` → `(1 6 19)`） | `%e3-float-prefer-dtype` 缺省不再返回 NIL，改为按输入推导（float32→float32，其余→float64） |
| P2-1 | `vt-argmax/argmin/nanargmax/nanargmin` | 返回 `:int32`，numpy 返回 `intp`(int64) | `%op-out-dtype` 改 `:int64` |
| P2-2 | `vt-logical-and/or/not/xor` | 默认返回 `:float64`，违反 CONVENTIONS §3.1「比较/逻辑返回 `:int8`」 | 缺省 dtype 改 `:int8`，返回 1/0 |
| P2-3 | `vt-cov` | `ddof` 语义与 numpy 不符：numpy 的 `ddof` **缺省为 1**（`np.cov(a)` 除以 N-1，`np.cov(a, ddof=0)` 除以 N），原实现恒用 N-1-ddof | 用 `ddof-p` 哨兵区分「未传」与「显式 0」，分母统一为 `N - ddof`（缺省 ddof=1） |
| P2-4 | `vt-histogram` | counts 返回 `:float64`，numpy 返回 int64 | counts 改 `:int64`（density=t 仍为 float64） |
| P2-6 | `vt-pow` | 整数基 + 正整数指数返回 float64（numpy 返回 int64）；传张量指数**静默返回 NaN** | 整数基 + 正整数指数保持整数 dtype；显式拒绝张量指数并给出清晰错误 |

> P0-1 是本轮最严重缺陷：非崩溃代码路径在正常输入（int 张量）下即触发
> 内存越界并杀死进程。已在独立子进程复现（`Memory fault ... in
> CLVT::%RELU-DOUBLE`），修复后 int/int8/int32/uint8 全部正常。

### 经核实为「报告误报」或「上游已修复」（未改动）

| 报告条目 | 报告结论 | 实际核实 |
|---|---|---|
| P0-2 `:out` 非连续视图写入丢失 | 称 base 仍全 0 | **误报**：当前 master 上 `vt-add(m,m,:out=v)` 正确落到底层 base，与 numpy 逐位一致（已有 `out-contig-tests` 23 条守护） |
| P2#5 `vt-trace` 整数输入返回 float64 | 称应返回 int64 | **误报**：实测 int64→int64、float32→float32、int8→int64，与 numpy 完全一致 |
| P2#8 `vt-eigvalsh` 未升序 | 称应升序 | **上游已修复**（v0.4.0）：现 `vt-eigvalsh` 升序、`vt-eigvals` 降序，均已符合各自 numpy 约定 |
| P2#3 描述「除以 n-ddof-1」 | 描述 | 描述本身正确，但报告未指出 numpy 的 **`ddof` 缺省值为 1** 这一关键点；已按 numpy 真实语义修复（见上表 P2-3） |

### 未处理（API 缺口，非缺陷）

报告 P2 表中另有约 12 项属**功能缺口或有意设计**，非缺陷，本次不改：
`vt-percentile` 不支持数组分位数、`vt-det` 不支持批量、`vt-bincount` 无
`:weights`、`vt-norm` 无 `ord`、`vt-sort` 无降序开关、`vt-array-equal`/
`vt-allclose` 无 `equal-nan`、`vt-repeat` 不支持逐元素 repeats、`vt-delete`/
`vt-ravel-multi-index`/`vt-random-choice :p` 不接受 VT（要求 list）、
`vt-lstsq :rcond nil`、布尔掩码写入 API 缺失、`vt-layer-norm` 为 torch 风格。
这些是"尚未实现"而非"实现错误"，建议列入后续路线图而非缺陷修复。

### 测试

- `test/extensions3-test.lisp`：新增 38 条「审查修复回归」断言
  （P0-1×6、P1-1×8、P1-2×4、P2-1×5、P2-2×5、P2-3×5、P2-6×5），
  套件由 184 → **222** 条全绿。
- `test/out-contig-tests.lisp`：argmax 的 `:out` 由 int32 改 int64（随 P2-1 契约更新）。
- `test/coverage-gap-test.lisp`：histogram counts 期望值改整数（随 P2-4）。
- 全量 **30/30 套件通过，0 失败**；`example/example.lisp` 打印 `all test passed`。

---
## 2026-10-05 — 覆盖缺口审计：补齐未覆盖函数测试 + 修复 6 处实现缺陷

在 SBCL 2.6.8 + Quicklisp + numpy 2.4.6 环境上，以 CONVENTIONS.md 为语义
契约运行全部 27 个测试套件与 example/example.lisp（均全绿），随后做
**导出函数覆盖审计**：对比 `src/package.lisp` 的 306 个 `vt-*` 导出与
`test/*.lisp` + `example/example.lisp` 的全部引用，发现 61 个公开函数
从未被任何测试或示例调用。

据此新增 `test/uncovered-coverage-test.lisp`（136 条断言，期望值取自
numpy 2.4.6 实测或 CONVENTIONS 契约），并为其中暴露的缺陷做修复。

### 修复的实现缺陷（6 处）

| # | 函数 | 缺陷 | 修法 |
|---|------|------|------|
| 1 | `vt-identity` | **崩溃**：`(vt-identity 3)` 缺省 `dtype` 为 NIL，下传 `vt-eye` 触发 `NIL fell through ECASE` | `&key (dtype :float64)` 对齐 `vt-eye` 缺省值 |
| 2 | `vt-vander` | **崩溃**：`(vt-vander x)` 缺省 `n` 为 NIL → `odd number of &KEY arguments` | `&key (n nil)`，缺省取 `len(x)`（对标 `np.vander` N=len(x)） |
| 3 | `vt-atanh` | `\|x\|=1` 返回 NaN，与 numpy 不符 | `x=+1→+Inf`、`x=-1→-Inf`、`\|x\|>1→NaN`（补 `vt-get-pos-inf`/`vt-get-neg-inf`） |
| 4 | `vt-= vt-/= vt-< vt-<= vt-> vt->=` | 返回 `float64`，违反 CONVENTIONS §3.1「比较运算返回 :int8 承载布尔」 | 缺省 dtype 改 `:int8`，返回值 1/0（numpy 语义不变，dtype 对齐） |
| 5 | `vt-positive-p vt-negative-p vt-zero-p vt-nonzero-p vt-even-p vt-odd-p` | 同上，返回 `float64` | 缺省 dtype 改 `:int8` |
| 6 | `vt-isnan vt-isinf vt-isfinite` | 返回 `float64` 布尔 | dtype 固定 `:int8` |

> 缺陷 4–6 是**契约违反**：CONVENTIONS §3.1/§9.2 明确"比较/逻辑运算返回
> `:int8`，取值 0/1"，而实现长期返回 `float64`。属"文档正确、实现偏差"，
> 按 CONVENTIONS「与 NumPy 冲突以 NumPy 为准、与文档冲突修实现」处理。
> `vt-all`/`vt-any`/`vt-count-nonzero` 返回整数计数（`int64`），非布尔，
> 保持不动；`vt-def-vt-reduce` 的 `:all`/`:any` 分支亦不受影响。

### 同步更新的既有测试（期望值随契约修正）

- `test/comprehensive-test.lisp`：`a<b` / `a==b` 期望由 `(1.0 1.0 0.0 ...)`
  改为 `(1 1 0 ...)`（int8）。
- `test/nan-random-test.lisp`：`isnan/isinf/isfinite` 期望改为整数 0/1。

### 新增测试套件

- `test/uncovered-coverage-test.lisp`（136 断言，已注册进 `run-tests.sh`）
  覆盖：二元算术别名、反三角/反双曲、谓词族、逻辑/位运算、创建族
  （含 identity 回归）、拼接/维度、clamp/select/copy-to!、vander 回归、
  结构原语（strides/normalize-axis/out 可写性/strides 驱动写入）、
  扩展函数、extensions2、随机数底层对象、最小二乘。

**验收**：`bash test/run-tests.sh --quick` → 28 套件全绿；
`sbcl --load example/example.lisp` + `run-all-tests` → 全绿。

## 2026-10-05 — extensions3：补充第三批 NumPy 重要缺失函数（约 50 个）

新增 `src/extensions3.lisp`（约 1100 行），按 CONVENTIONS.md 的语义契约
（`&key dtype out`、`ensure-vt` 归一化输入、比较/逻辑返回 `:int8`、
NaN/Inf 对齐 numpy 2.4.6、`:out` 硬契约 H1–H6）一次性补齐 NumPy 高频函数，
并配套新增 `test/extensions3-test.lisp`（184 条断言，全部通过）。

### 新增函数（按类分组）

| 类别 | 函数 |
|------|------|
| 创建 | `vt-asarray` `vt-fromiter` `vt-tri` `vt-diagflat` `vt-trim-zeros` |
| 形状 | `vt-rollaxis` `vt-column-stack` `vt-block` `vt-broadcast-arrays` `vt-resize` |
| 索引 | `vt-take-along-axis` `vt-put-along-axis` `vt-compress` `vt-indices` `vt-fill-diagonal` |
| 数学 | `vt-absolute` `vt-sign` `vt-positive` `vt-expm1` `vt-log1p` `vt-logaddexp` `vt-float-power` `vt-copysign` `vt-signbit` `vt-nextafter` `vt-spacing` `vt-gcd` `vt-lcm` `vt-divmod` `vt-nan-to-num` `vt-real` `vt-imag` `vt-conj` `vt-angle` |
| 统计 | `vt-nancumsum` `vt-nancumprod` `vt-nanpercentile` `vt-nanquantile` `vt-cov` `vt-corrcoef` `vt-cross` |
| 线代 | `vt-vdot` `vt-eigvals` `vt-eigvalsh` `vt-matrix-power` `vt-cond` `vt-multi-dot` |
| 逻辑 | `vt-array-equal` `vt-array-equiv` `vt-isposinf` `vt-isneginf` |
| 集合 | `vt-isin` |

### 关键实现要点（对齐 NumPy 语义）

- **`vt-take-along-axis` / `vt-put-along-axis`**：非 axis 维按 NumPy
  **广播规则**处理（长度相等或为 1），而非要求严格相等；输出形状中 axis 维
  取自 `indices`、其余维取自输入张量。原实现误用严格相等 + 线性下标，
  已改为基于 `strides` 的坐标寻址，并修正 `vt-ravel` 后 1 维视图的混用。
- **`vt-spacing`**：`numpy.spacing(x) = nextafter(|x|, +inf) - |x|`；
  原实现误取「向 `most-positive-double-float` 的 nextafter 值」。
- **`vt-cross`**：结果形状对齐 NumPy——1 维输入去掉 batch 轴（2 分量 → 0 维
  标量）；2 维批量输入保留 batch 轴（2 分量 → 去掉分量轴）。修正了
  `%e3-as2d` 对 `axis=-1` 误转置的缺陷。
- **`vt-nancumsum` / `vt-nancumprod`**：`axis` 分支的输出坐标需以 axis 位置的
  累积下标**替换**该维坐标，且循环结束后返回 `result`（原实现返回 `nil`）。
- **`vt-sign`**：NaN 分支须包 `with-float-safe`（`%nan-p` 的 `x /= x`
  否则触发 `FLOATING-POINT-INVALID-OPERATION`）。
- **`vt-expm1` / `vt-log1p`**：x≈0 时用 Taylor 级数（`%e3-expm1` /
  `%e3-log1p`）保证与 numpy 一致的精度（`1e-10` 量级逐位对齐）。
- **`vt-signbit`**：用 `(minusp (float-sign xf))` 正确识别 `-0.0`。

### 测试

- `test/extensions3-test.lisp`：184 条断言覆盖全部新函数，含
  `:out` 硬契约（形状/dtype 精确匹配、可写）、非连续输入、dtype 提升、
  NaN/Inf 语义与错误路径。
- 测试辅助 `->list`/`approx` 增加 NaN 安全处理（`with-float-traps-masked`），
  避免浮点陷阱在断言阶段误报。

---

## 2026-10-05 — v0.3.6 numpy 2.1.3 语义对齐：空归约重校准 / :out 精度解耦 / true_divide / random 校验

以 numpy 2.1.3 实测为基准（逐项探针对表），修正 v0.3.5 分歧表中三处
"设计约定"与 numpy 的偏差，并收敛 TEST-PLAN §6 两项契约缺口。
新增回归套件 test/numpy-convention-tests.lisp（46 条断言），全部通过。

### 方向 1：空归约语义重校准（reduce-stats.lisp，def-vt-reduce 统一生成）

| 场景 | v0.3.5（NaN 填充约定） | v0.3.6（numpy 对齐） |
|------|------------------------|----------------------|
| 输出为空（`(3 0)` axis=0 等） | NaN 填充 / arg 报错 | **所有算子返回空结果**（含 argmax/argmin） |
| max/min 族输出非空且归约区空 | NaN 填充（整数提升 float64） | **ValueError**（zero-size array to reduction operation ... which has no identity） |
| arg 族输出非空且归约区空 | 报错 | **ValueError**（attempt to get argmax of an empty sequence，消息对齐 numpy） |
| 单位元族（sum/prod/all/any/nan*） | 填单位元 | 不变 |
| mean 空归约 | NaN | NaN（numpy 同款，保留） |

### 方向 2：`:out` 精度解耦（TEST-PLAN §6 #2 收敛）

旧行为：`:out` dtype 决定**计算精度**（float64 输入 + int32 out 按 int32
累加，静默错值）。新行为：**按输入提升计算**（compute-dtype，与无 :out
调用逐位一致），最后一步 cast 写入 `:out`（decoupled 预路径）。

- `def-vt-reduce`（sum/prod/amax/amin/all/any/nan*/arg* 共 14 函数）：
  三条内核路径统一——decoupled 时结果先写入 compute-dtype 临时缓冲，
  尾部 `vt-copy-into` 跨 dtype 写入 out
- `vt-mean`/`vt-var`/`vt-std`/`vt-nanmean`/`vt-nanvar`/`vt-nanstd`：
  compute-dtype / exec-dtype / write-out 三段式解耦（显式 `:dtype` 仍按
  numpy 语义直接以该 dtype 计算）
- `:out` 与 `:dtype` 冲突显式报错；`:out` 广播视图（dim>1 且 stride=0）
  拒绝写入（既有契约保留）

### 方向 3：vt-mod/vt-rem 零除 dtype 语义（elementwise.lisp）

旧行为：零除一律返回 0（库约定）。新行为：**按除数 dtype 语义**——
浮点语境（除数为浮点标量或张量）→ NaN（numpy mod/fmod 同款），
整型语境 → 0。混合输入按被除数语境判定。

### 方向 4：vt-/ true_divide 语义（elementwise.lisp）

旧行为：整数除零报 CL division-by-zero；整数/整数返回整型截断。
新行为（numpy true_divide 同款）：**整数输入先提升 float64 再除**，
零除按 IEEE 得 ±Inf/NaN（`(vt-/ 5 0)` → +Inf、`(vt-/ 0 0)` → NaN），
`(vt-/ 5 2)` → 2.5d0（float64）；显式 `:dtype`/`:out` 浮点目标一致处理；
标量除数不再触发类型崩溃。

### 方向 5：random NaN/Inf 参数校验前移（random.lisp，TEST-PLAN §6 #4 收敛）

`vt-random-uniform`/`vt-random-normal`：`numberp` 检查在最外层，
NaN/Inf 判定（`%nan-p`/`%inf-p`）前移进 `with-float-safe`（屏蔽
FP-INVALID-OPERATION）——保证报出干净的参数错误而非 FP 异常。
本轮补齐 `vt-random-normal` 缺口：mean 有限性校验（此前完全缺失）、
std 非负**且有限**校验（此前 `>= 0` 放过 +Inf）。

### 测试

- 新增 test/numpy-convention-tests.lisp：46 条断言（空归约三分类 ×17、
  `:out` 精度解耦 ×8、mod/rem 零除 ×6、true_divide ×6、random 校验 ×9），
  已注册 run-tests.sh
- 更新 shape-degenerate-tests.lisp（amax/argmax 空 (3 0) 断言改 numpy 语义、
  softmax (0) 改期望报错——scipy.special.softmax 实测对空数组抛同款
  ValueError）、error-contract-tests.lisp（整数除零改 IEEE 断言：0-d 输入
  + 测试侧 FP 陷阱屏蔽的 NaN 检测）、refactor-bugfix-tests.lisp（mod 零除
  两断言改 NaN 期望）、test-ai-edge-cases.lisp（BUG-2b softmax 空向量改
  期望报错）
- 修复 def-vt-reduce 宏体两处括号错位（patch 期间引入：空归约块提前闭合
  主 let* 导致 14 个生成函数缺失/旧代码残留，表现为运行时 IN-ET unbound）
- 清理 vt-var 冗余 write-out 绑定（emit 统一出口后遗留）

---

## 2026-10-04 — v0.3.5 三层分离复审：非有限值传播 / numpy 语义修复 / 内存安全校验

在 SBCL 2.6.8 + Quicklisp 环境下按三层分离约定逐文件复审全部源码，
以运行时探针逐项对表 numpy 实测语义，确认并修复 8 类 bug。
新增回归套件 test/refactor-bugfix-tests.lisp（53 条断言），全部通过。

### 方向 1：非有限值（NaN/±Inf）传播（执行层泄漏底层陷阱 → 逻辑层显式处理）

`vt-floor`/`vt-ceiling`/`vt-truncate`/`vt-round`/`vt-rint`/`vt-mod`/`vt-rem`
对 NaN/±Inf 输入直接泄漏 `FLOATING-POINT-INVALID-OPERATION`（SBCL 对非有限值
调用 floor/round 会 signal，浮点陷阱屏蔽无法避免），而 numpy 语义为传播：
floor(nan)=nan、floor(±inf)=±inf、mod 含 NaN → NaN、mod(±Inf, 有限) → NaN、
mod(有限, ±Inf) → 被除数本身。
修复（elementwise.lisp）：`%floor-family-body` 宏统一在进入 CL 取整前拦截
非有限值并返回原值；mod/rem 增加同一约定（除 0 返 0 的库约定保持不变）。

### 方向 2：vt-insert 对齐 numpy.insert（逻辑层语义修复）

旧实现降序插入且负索引按"演化中尺寸"解析，与 numpy 相比有两处偏差：
重复索引处值逆序（`insert([0,1,2],[1,1],[10,20])` 旧 `[0,20,10,1,2]`
→ numpy `[0,10,20,1,2]`）；混合正负索引错位。
修复（join.lisp）：flat/axis 两模式统一为 numpy 算法——
先按原尺寸归一化负索引并越界检查，再升序稳定排序（值跟随索引、
同位置按 values 顺序），插入位置 = 归一化位置 + 已插入个数。

### 方向 3：内存安全与输入校验（L1/L2 底线）

| 问题 | 修复 | 文件 |
|------|------|------|
| `vt-solve` 不校验 b 行数：b 行数 < n 时消元循环按 n 行读写底层缓冲区（实测 SBCL 报 "Invalid index 2 for (SIMPLE-ARRAY DOUBLE-FLOAT (2))"，越界尝试） | 逻辑层显式校验 b 行数 = 系数矩阵阶数，违反报可定位错误 | linalg.lisp |
| `vt-tensordot` 负轴不归一化：被当自由轴处理，产生错误结果或 "%vt-einsum-string" 内部错误 | 负轴按各自秩归一化 + 越界显式报错；整数 axes 超出可收缩范围报错 | extensions.lisp |
| `vt-inner` 0 维输入报 subseq 内部错误 | 0 维输入（任一侧）走广播逐元素乘（对标 numpy.inner） | extensions.lisp |
| `vt-mod`/`vt-rem` 仅接受标量除数，张量除数报类型错误 | 除数可为张量，广播逐元素取模（语义同标量路径） | elementwise.lisp |

### 方向 4：dtype 单一事实来源补全（逻辑层）

| 问题 | 修复 | 文件 |
|------|------|------|
| `vt-from-array` 对 (signed-byte 16/8)、(unsigned-byte 8/16) CL 数组一律静默推断 :int32（subtypep 宽类型命中） | 按窄类型优先精确推断 :int8/:int16/:uint8/:uint16/:int32/:int64/:float32/:float64，显式 :dtype 仍可覆盖 | creation.lisp |
| `vt-cast-fun :uint16` 走 `%wrap-uint16`（v0.3.1 统一政策的遗漏项） | 统一为 `%coerce-uint16` | dtype.lisp |
| package.lisp 5 个 core.lisp 辅助函数重复导出 3 处 | 归并到「张量结构访问器」区唯一导出 | package.lisp |
| `vt-rem` docstring 误标 numpy.remainder（CL:rem 为截断除法余数 = numpy.fmod） | 更正为 numpy.fmod | elementwise.lisp |

### 测试

- 新增 test/refactor-bugfix-tests.lisp：53 条断言覆盖上述全部修复
  （floor 族 NaN/Inf ×6、mod/rem 张量除数与非有限值 ×6、insert numpy 语义
  ×8、solve 校验 ×3、inner 0 维 ×3、tensordot 负轴 ×5、from-array 推断
  ×7、cast-fun ×2 及相关回归），已注册 run-tests.sh
- numpy 实测对照：np.insert / np.mod / np.inner / np.tensordot 逐例验证
- 全量回归 `bash test/run-tests.sh`：24/24 套件全部通过（含既有
  shape-degenerate 70 / property-strided 67 / error-contract 19 等套件）

---

## 2026-10-03 — v0.3.4 测试规划落地 / vt-copy-into 零尺寸写入修复

按 TEST-PLAN.md（新增，test/TEST-PLAN.md）建立"类别矩阵 > 函数覆盖 > 断言数量"
的覆盖策略，并以探针实测（tmp/probe-edge.lisp 方法）逐项对表 numpy 语义。

### 方向 1：vt-copy-into 零尺寸写入早退（bug 回归：stack axis=1 堆叠空 1D）

`(vt-stack 1 (vt-zeros '(0)) (vt-zeros '(0)))` 泄漏 CL 底层序列错误
"bounding indices 1 and 1 are bad for a sequence of length 0"：
vt-concatenate 向 (0 2) 结果的 [:, 1:2] 空视图写入时 offset=1、底层
存储长度=0，快路径按 (offset+size) 计算平坦边界触发 replace 越界。
修复（core.lisp）：vt-copy-into 在形状兼容性检查后、快路径前增加
零尺寸早退（v0.3.3 空 strided 问题的写入侧）。零元素写入是 no-op，
但非法形状仍照常报错——只跳过平坦边界计算，不放宽契约。

### 方向 2：三个新套件（156 条断言，全部通过）

| 套件 | 类目 | 断言 | 内容 |
|------|------|------|------|
| test/shape-degenerate-tests.lisp | C1/C2 | 70 | 全 op 家族 × 空形状/退化形状表驱动：创建、元素级、归约（sum/prod 单位元、mean/amax NaN 约定、argmax 报错约定）、matmul/dot/outer/trace、变形、索引、nn、集合运算；含 bug-stack-empty-ax1 回归 |
| test/property-strided-tests.lisp | C8 | 67 | strided≡contiguous 选路不变量（8 随机形状 × 元素级/归约/转置/argmax/matmul，固定种子确定性）；flip/transpose/reshape 往返、concat 可逆、sum 拆解、sort≡take(argsort) 等 12 项代数恒等式 |
| test/error-contract-tests.lisp | C7 | 19 | 形状/秩/内维/perm/越界/切片步长/one-hot/einsum/uniform 参数域/整除 0/广播 out 只读等契约错误逐项断言 |

三个套件均已登记 test/run-tests.sh。回归确认无副作用：
test-copy-into 35、out-contig-tests 23、test-overlap 6、
test-ai-edge-cases 40、memsafety-tests 6 全部通过。

### 已知分歧与缺口（TEST-PLAN.md §5/§6 记录，不阻塞）

- mean/median/amax/amin 空归约填 NaN、argmax 空输入报错：文档化设计约定，
  与 numpy（ValueError/空输出）不同；
- reduce 族 `:out` dtype 决定计算精度（float64 输入 + int32 out 按
  int32 累加）：契约缺口，需专项决策（对齐 numpy 提升语义或文档化）；
- vt-random-uniform NaN 参数报 FP 异常而非参数校验错误：L2 达标，
  错误类型欠佳。

---

## 2026-10-02 — v0.3.2 首次 SBCL 实测回归 / reduce 小整型内核补全

v0.3.0/v0.3.1 两轮重构均为静态验证（无 SBCL 环境）。本轮在 SBCL 2.6.8 +
Quicklisp + sb-simd 真实环境下首次全量加载与回归，修复实测暴露的问题，
并补全三层分离在执行层的最后一处 dtype 契约缺口。

### 方向 1：reduce 家族小整型 dtype 补全（执行层选路规则落地）

`def-vt-reduce` 生成的类型特化内核此前只覆盖 4 种存储级元素类型
（double/single/int64/int32），导致 `vt-sum`/`vt-amax` 等对
int16/int8/uint8/uint16 张量直接抛 "unsupported input/output dtype"——
违背 dtype.lisp 声明的 8 种逻辑 dtype 单一事实来源（test-bug0 用例 5 暴露）。

修复（reduce-stats.lisp）：

| 项目 | 设计 |
|------|------|
| 特化内核范围 | 保持 4 种存储级类型不变（对应 `*vt-storage-dtypes*`），避免内核全展开（8×8 组合实测在 1GB 堆下编译器耗尽） |
| 小整型路径 | 新增 `%kernel-small-general`：每算子仅展开一份的通用内核（路径 3 结构），读取经通用 aref，sum/prod 按 numpy 语义提升 int64 累加，max/min 以输入 dtype 极值初始化，写出经 vt-cast 按结果 dtype 转换 |
| 选路守卫 | 路径 1（连续+全局）/路径 2（连续+单轴）增加存储级 dtype 守卫，小整型统一由路径 3 兜底——连续性/特化只影响性能，不影响正确性 |
| dtype 映射 | `%dtype->lt`/`%et->lt`/`%lt-rank`/`%cast-form` 补全 8 种类型；`%op-acc-lt` 小整型 sum/prod 提升 int64；`%op-init` 补全小整型 max/min 极值初始化 |
| 空归约 | 单位元填充的编译期表覆盖全部 8 种输入类型（此前 prod 空小整型会错误返回 0） |

语义对齐 numpy：`sum(int8)→int64`、`amax(int8)→int8`（保 dtype）、
`amax(uint16)→65535` 无回绕。

### 方向 2：随机数静默陷阱显式化

| 问题 | 修复 | 文件 |
|------|------|------|
| `vt-random` 整型 dtype 下 truncate(U[0,1)) 恒为 0（静默全 0 陷阱，test-bug0 用例 17 标注） | 显式报错并指引 `vt-random-int` / `vt-random-integers`；仅接受 :float64/:float32 | random.lisp |

### 方向 3：测试基础设施修复（实测暴露）

| 问题 | 修复 | 文件 |
|------|------|------|
| 7 个测试文件开头 `(ql:quickload :clvt)`，而 run-tests.sh 只经 ASDF 加载系统、不加载 Quicklisp，导致 7 个套件必崩 | 改为 `(require :asdf)` + `#+quicklisp (ql:quickload ...)` + `(asdf:load-system :clvt)`，Quicklisp 存在与否均可运行 | test/*.lisp |
| `set -u` 下失败汇总 `${SUITES[$name]}` 对未注册套件报 unbound variable，吞掉失败清单 | 补默认值 `:-未知测试`；补齐 7 个套件的 SUITES 注册 | test/run-tests.sh |
| comprehensive-test "b/a(int)" 假失败：期望值 `2.3333333333333335` 缺 `d0` 被读成单精度，与正确的 float64 输出差 ~1.6e-7 > 1e-10 容差 | 期望值改双精度字面量 | test/comprehensive-test.lisp |
| test-bug0 以 `(and (= *checks* 28) (= *failures* 2))` 固化 2 个已知失败 | 修复后扩充为 37 项检查、0 失败，新增 9 项小整型归约回归（含 axis 归约与转置视图） | test/test-bug0.lisp |

### 验证

- 环境：SBCL 2.6.8（x86-64-linux binary）+ ASDF 3.3.1 + Quicklisp + sb-simd
- `bash test/run-tests.sh`：**19/19 套件全部通过**（run_all_tests 143、
  run_param_tests 155、robustness 194、coverage-gap 97、comprehensive 119、
  auto-compare 63、numpy-compare 69、test-bug0 37 等约 1100+ 断言）
- 冒烟覆盖：int8/uint8/int16/uint16 的 sum/amax/amin/argmax/prod/nansum/
  nanmax/all/any、axis 归约、转置视图、:out 写入、:dtype 覆盖、空轴归约
- einsum 小整型输入经通用路径提升 float64 验证通过（显式 :dtype/:out 仍限
  存储级类型并显式报错——已文档化的边界，不属静默错误）

---

## 2026-10-01 — v0.3.1 数据类型统一 / NaN·Inf 补全 / out 契约加固

在 v0.3.0 三层架构文档化基础上，针对三个遗留问题域做第二轮重构。

### 方向 1：数据类型统一（dtype.lisp 单一事实来源）

| 问题 | 修复 | 文件 |
|------|------|------|
| `vt-cast-fun` 对 int8/uint8 走 `%wrap-*`，int16/int32 走 `%coerce-*`，风格分裂 | 统一为 `%coerce-*`（语义一致，含 typep 快速路径） | dtype.lisp |
| `vt-arange` int64/int32 溢出直接存储（触发数组元素类型错误），int16/int8/uint 却回绕 | int64/int32 统一 `%wrap-*` 回绕（§9.2） | creation.lisp |
| `vt-linspace` float32 用 float32 累加，长序列漂移 | 以 double 精度计算、存储时舍入（对标 NumPy） | creation.lisp |
| `vt-bits->unsigned-dtype` 返回 `:uint32/:uint64`（非存储类型） | 文档标注保留符号状态 | dtype.lisp |

### 方向 2：NaN / Inf 特殊值补全

| 问题 | 修复 | 文件 |
|------|------|------|
| `vt-mod`/`vt-rem` 除 0：整数路径硬报错、浮点路径实现定义（NumPy 返回 0） | 除 0 返回 0 | elementwise.lisp |
| `vt-even-p`/`vt-odd-p` 对 NaN/±Inf 触发 `evenp`/`floor` 类型错误 | 非有限值判定为不成立（返回 0） | elementwise.lisp |
| `vt-hypot(NaN, ±Inf)` 结果依赖 SBCL `max` 的实现定义 | 任一 ±Inf → +Inf；NaN 单独出现时传播 | elementwise.lisp |
| `vt-random-uniform`：`assert(< low high)` 拒绝 low=high（NumPy 允许），NaN 边界报错误导 | 显式校验有限性；low=high → 常量数组 | random.lisp |
| `vt-random-normal`：负/NaN std 静默产生镜像分布 | 校验非负有限；std=0 → mean 填充（对标 NumPy scale=0） | random.lisp |

### 方向 3：:out 内存布局契约加固

| 问题 | 修复 | 文件 |
|------|------|------|
| `vt-det` 接受任意形状 :out 并被 `vt-fill` 静默填满 | :out 必须为 0 维张量，违反即报错（§8.2） | linalg.lisp |

审计确认已正确、无需改动的部分：`vt-map`/`vt-fast-map` 非连续 out 的 strides 写入与重叠快照保护；`def-vt-reduce` 家族（out-contig-tests 回归覆盖）；einsum 的 out strides 与重叠拷贝；`vt-where`/`vt-take`/`vt-pad`/`vt-triu`/`vt-tril`；`vt-sigmoid`/`vt-relu` 快路径连续性守卫；`log`/`sqrt`/`asin`/`acos`/`atanh`/`pow` 的复数域守卫与 NaN 返回；`maximum`/`minimum`/`fmax`/`fmin` 的 NaN 语义；median/percentile 的 NaN 传播。

---


## 2026-10-01 — v0.3.0 架构重构：三层分离语义契约落地

按《高性能张量库设计约定》完成整体重构审查，源码语义与文档对齐三层架构
（逻辑层 / 物理层 / 执行层）。全部修改不影响既有公开 API 签名。

### 修正

1. **`%inline1-strided` / `%inline2-strided` 误导性文档（关键修复）**
   两个宏自实现起就遍历结果张量自身 strides，一直支持任意 strides（含非连续 RES、
   广播输入），但文档错误标注"RES 必须连续"，差点误导执行层选路架构。
   已重写文档并显式声明"优化路径与通用路径输出必须逐位相同"。

2. **空归约语义（reduce-stats.lisp def-vt-reduce）**
   旧行为：空张量/空轴上 max/min 返回 ±Inf 哨兵、argmax/argmin 填 0。
   新行为按设计约定 §6.2：
   - 有单位元的操作返回单位元：`sum→0`、`prod→1`、`all→1`、`any→0`、
     `nansum→0`、`nanprod→1`；
   - `max/min/nanmax/nanmin` 无单位元：填充 NaN；整数结果 dtype 提升为
     float64 承载 NaN；显式整数 `:dtype`/整数 `:out` 与 NaN 结果冲突时报错；
   - `argmax/argmin/nanargmax/nanargmin` 无单位元且无哨兵值：一律报错。

3. **einsum `:out` 硬契约（linalg.lisp）**
   形状/dtype 校验由 `assert` 改为显式 `error`：assert 可能被
   `(safety 0)` 编译剔除，硬契约必须保证触发。

### 文档

- `package.lisp` 增加三层架构总纲（各层职责、失败模式与核心原则）。
- `core.lisp` vt 结构文档补全物理层四要素与 stride-0 广播只读语义。
- `map-reduce.lisp` 模块头写入执行层三条路径（快路径/strides 通用路径/vt-map）
  的选路规则与逐位等价承诺。
- `nn.lisp` `vt-softmax` 补充数值稳定化文档（全 -Inf 行 → NaN，对标 PyTorch）。
- `README.md` 新增"设计约定（三层架构语义契约）"章节：
  内存模型、视图/拷贝表、广播、类型提升、NaN 语义、归约、别名安全、
  out 契约、einsum 路由。

---

## 2026-09-05 — 测试体系重构与性能优化

### 测试体系重构：静态 JSON → 实时 NumPy 参考生成

**问题**：原测试体系依赖三份静态预先生成的 JSON 文件（`all_expected.json`、`param_expected.json`、`expected_numpy.json`）作为参考值对比基准，存在历史数据过期、维护困难、强依赖 PyTorch 等问题。

**修复**：全部 8 个测试套件中的数值对比测试改为**运行时实时调用 Python + NumPy 生成参考值**，不再依赖任何历史静态 JSON 数据。

#### 修改的测试文件

| 文件 | 改动 |
|------|------|
| `test/ref_compute.py` | 重写：核心用纯 NumPy 实现，覆盖 8 个测试套件约 900 个用例；保留 PyTorch 交叉验证层（76 项核心算子独立验证） |
| `test/run_all_tests.lisp` | 移除对 `all_expected.json` 的读取，改为实时调用 Python 生成参考值；内置 JSON 解析器；改进 approx 跨类型比较 |
| `test/run_param_tests.lisp` | 移除对 `param_expected.json` 的依赖，改为实时调用；修复 `vt-narrow` 参数语义（start/end 而非 start/length）；删除重复 convolve 项 |
| `test/auto-compare-test.lisp` | 移除对 `expected_numpy.json` 的依赖，改为实时调用；修复键名冲突（`roll_2` / `flip_axis0` 等分配唯一键） |
| `test/numpy-compare-test.lisp` | 改为调用更新后的 `ref_compute.py`；改进数值比较容差 |
| `test/run-tests.sh` | 不再强制要求 PyTorch，仅检查 NumPy 可用性 |

#### 删除的过时文件
- ❌ `test/all_expected.json`（约 3500 行历史硬编码数据）
- ❌ `test/param_expected.json`（约 9500 行历史硬编码数据）
- ❌ `test/expected_numpy.json`（约 1200 行历史硬编码数据）

### Bug 修复

| Bug | 位置 | 严重程度 | 修复 |
|-----|------|----------|------|
| `vt-nanmean` / `vt-nanvar` 类型转换崩溃（浮点布尔掩码强制转 int64 导致 TYPE-ERROR） | `reduce-stats.lisp` | 🔴 严重 | count 改用与结果相同的浮点 dtype，zerop/divisor 全部浮点化 |
| `(vt-/ x)` 一元倒数走普通 vt-map 带 lambda 开销 | `elementwise.lisp` | 🟡 中等 | 改用 `vt-fast-map` 标量广播除法快路径 |

### 性能优化

#### 1. vt-sum / vt-mean 全局归约连续内存快路径（≈30×）
**位置**：`src/map-reduce.lisp`

为 **连续内存 + 全局归约（axis=nil）** 添加平面 dotimes 循环快路径，使用 macrolet 生成按元素类型特化的循环，消除通用递归 stride 遍历和每元素 funcall 开销。

| 操作 | 优化前 | 优化后 | 提升 |
|------|--------|--------|------|
| `vt-sum`（1M 元素全局归约） | 1452 ms | **48 ms** | **30×** |
| `vt-mean`（1M 元素全局归约） | ~1450 ms | **~50 ms** | **~29×** |

#### 2. vt-sigmoid 内联快路径（≈5.6×）
**位置**：`src/nn.lisp`

为连续 float 张量添加 `%sigmoid-fast` 内联快路径，直接展开 `1.0 / (1.0 + exp(-x))` 计算。

| 操作 | 优化前 | 优化后 | 提升 |
|------|--------|--------|------|
| `vt-sigmoid`（1M 元素） | 472 ms | **84 ms** | **5.6×** |

#### 3. vt-relu 内联快路径（≈7.9×）
**位置**：`src/nn.lisp`

为连续 float 张量添加 `%relu-fast` 内联快路径，直接展开 `(max 0.0 x)` 计算。

| 操作 | 优化前 | 优化后 | 提升 |
|------|--------|--------|------|
| `vt-relu`（1M 元素） | 472 ms | **60 ms** | **7.9×** |

### 新增功能（extensions.lisp）

| 函数 | 对标 | 说明 |
|------|------|------|
| `vt-clamp` | `torch.clamp` | `vt-clip` 的 PyTorch 风格别名 |
| `vt-copy-to!` | `tensor.copy_` | PyTorch 风格原地拷贝，便于链式调用和内存复用 |

### 新增测试 / 基准工具

| 文件 | 说明 |
|------|------|
| `test/performance-bench.lisp` | Common Lisp 性能基准套件：算术 / 激活函数 / 归约 / 矩阵乘法 |
| `test/torch-compare.py` | 独立 PyTorch 对比测试 + 性能基线（18 项核心算子） |

### 测试结果

修改后全部 **8 个测试套件、约 900 个测试用例** 全部通过 ✅：

```
=== 测试总结 ===
  运行套件: 8
  通过:     8
  失败:     0
```

| 测试套件 | 用例数 | 说明 |
|----------|--------|------|
| `run_all_tests` | 143 | 基础函数（实时 NumPy 对比） |
| `run_param_tests` | 154 | 3D+ 参数化（实时 NumPy 对比） |
| `nested-test` | 60 | AI/ML 函数组合 |
| `robustness-test` | 194 | 鲁棒性边界 |
| `coverage-gap-test` | 97 | NumPy/PyTorch 覆盖差距 |
| `comprehensive-test` | 119 | 综合功能 |
| `auto-compare-test` | 63 | 自动对比（实时 NumPy 对比） |
| `numpy-compare-test` | 69 | NumPy/PyTorch 实时对比 |

---

## 2026-08-15 — 架构重构与性能优化

### 架构重构

从零重构了整个库，将原来职责混杂的 17 个顶层 `.lisp` 文件重组为 `src/` 下按职责分层的 20 个模块（详见 README「架构」一节），保持全部 `vt-*` 公共 API 签名兼容：

- **内核层**：`dtype.lisp`（类型系统单一事实来源）、`core.lisp`（结构/步长/广播/连续判定/拷贝/填充）、`iterator.lisp`、`map-reduce.lisp`
- **功能层**：`creation` / `manip` / `indexing` / `join` / `elementwise` / `reduce-stats` / `setops` / `linalg` / `nn` / `random` / `rotate` / `io` / `extensions`

### 规范性修正

- 补齐 `vt-compute-logical-strides` 等「已导出但未定义」的悬空符号
- `package.lisp` 导出按功能分组，`*vt-fun-list*` 自动收集
- 修复 `benchmark-copy.lisp` 中非法 FORMAT 指令 `~30-50x`

### 性能优化

- `vt-map` 重写为按输出元素类型特化（`(the (simple-array double-float (*)) ...)`），消除浮点装箱
- 恢复 `vt-fast-map` 编译期内联（`%inline1-loop` / `%inline2-loop` / `%cast-to`），`vt-+`/`vt-*`/`vt--`/`vt-/`、`vt-add`/`vt-sub`/`vt-mul`/`vt-div`/`vt-scale` 及一元数学函数（`vt-sin`/`vt-cos`/`vt-exp` 等）均内联算子，避免 `funcall` 装箱
- 基准（100 万元素，暖机后）：`vt-+` ≈ 0.004s、`vt-add` ≈ 0.009s、`vt-sin` ≈ 0.027s，达到或超越重构前水平；内联路径装箱量从 ~56MB 降至 ~8MB（仅结果张量本身）
- `vt-reduce` 内层递归循环按输入/输出元素类型特化（`macrolet` 生成 `(the (simple-array <type> (*)) ...)` 的 `recurse`，整数走 `truncate`、浮点走 `coerce`），消除 `vt-cast` 运行时 `ecase` 分派与通用 `aref` 的逐元素装箱；100 万元素下 `vt-sum` 提速约 1.5–1.75×、`vt-mean` 约 1.5–1.7×、`vt-amax` 约 1.3–1.5×
- `vt-ref` / `(setf vt-ref)` 按数组元素类型特化最终 `aref`，用 `(some #'zerop shape)` 替代 `vt-size` 全量 `reduce`，并补上显式越界检查（负索引归一化后校验范围，越界确定性报错，不再依赖 `aref` 的 safety 相关边界检查）；1D/2D/3D 标量访问提速约 1.3–1.6×

### 测试体系

- 新增 `test/numpy-compare-test.lisp` + `test/ref_compute.py`：SBCL 运行时调用 `python3` 实时生成 numpy / pytorch 参考结果并对比（69 项，覆盖创建/逐元素/归约/线性代数/神经网络/集合/形状/torch.topk）
- `test/run-tests.sh` 更新：纳入新套件、python3/numpy/torch 可用性检查
- 总计 **900 个测试用例**（831 原有 + 69 新增实时对比），全部通过 ✅

### 回归修复

通过 `example.lisp` 额外捕获并修复了重写引入的 4 处回归：

| 问题 | 位置 | 修复 |
|------|------|------|
| `vt-gradient` 切片规格多余嵌套 / 引号变量 | `reduce-stats.lisp` | `'((1 2))`→`'(1 2)`、`'(2 n)`→`(list 2 n)` |
| `vt-histogram` 缺失 `with-float-safe` 触发 NaN/Inf 浮点陷阱 | `reduce-stats.lisp` | 恢复包装 |
| `vt-unique` 缺失 `with-float-safe` 触发 `(= +inf NaN)` 陷阱 | `setops.lisp` | 恢复包装 |
| `vt-delete` 错误消息格式串与参数个数不匹配 | `join.lisp` | 补全 `~d` 占位符 |

### Bug 修复（数值转换与返回类型）

- `vt-cast` 对 `:int8`/`:int16`/`:uint8`/`:uint16` 增加 NumPy 语义的回绕（mod）：超出范围的值不再直接返回导致 TYPE-ERROR，而是回绕到目标范围内（如 `(vt-cast 200.0 :int8)` → -56、`(vt-cast 256 :uint8)` → 0、`(vt-cast 65536 :uint16)` → 0）
- `vt-cast-fun` 对 `:uint8`/`:uint16`/`:int8`/`:int16` 返回与 `vt-cast` 语义一致的转换函数（此前返回 `#'truncate`，负数行为不一致）
- `vt-sinc` 整数输入时按 `%infer-float-dtype` 推断浮点结果类型（此前沿用整数 dtype 导致精度丢失）
- `vt-gradient` 当 `axis=nil` 且输入为一维时返回单个 VT 对象（此前返回含单元素的列表，与 NumPy 不一致）

---

## 2026-08-10 — MiMo (Xiaomi AI) 项目审查与功能补充

### 项目审查

对整个 clvt 项目进行了全面审查，覆盖全部 15 个源文件（约 8000 行代码），识别出 NumPy/PyTorch 生态中常用但缺失的关键函数，并补充实现。

### 新增功能 (extensions.lisp)

| 函数 | 对标 | 说明 |
|------|------|------|
| `vt-count-nonzero` | `numpy.count_nonzero` | 统计非零元素个数，支持 axis/keepdims |
| `vt-count` | — | 统计等于指定值的元素个数 |
| `vt-flatnonzero` | `numpy.flatnonzero` | 展平后返回非零元素的一维索引 |
| `vt-moveaxis` | `numpy.moveaxis` | 将轴从源位置移动到目标位置，返回零拷贝视图 |
| `vt-inner` | `numpy.inner` | 内积，沿最后一个轴收缩 |
| `vt-tensordot` | `numpy.tensordot` | 张量缩并，支持整数轴和显式轴对两种模式 |
| `vt-topk` | `torch.topk` | 沿指定轴获取前 k 个最大/最小值及其索引 |
| `vt-clip-tensor` | — | 支持张量作为上下边界的裁剪（`vt-clip` 的扩展版） |
| `vt-set-print-options` | `numpy.set_print_options` | 设置打印阈值、精度、缩步长 |
| `vt-get-print-options` | — | 获取当前打印选项 |

### Bug 修复

| Bug | 严重程度 | 修复方式 |
|-----|----------|----------|
| README.md 中 `vt-trancate` 拼写错误 | 🟢 轻微 | 修正为 `vt-truncate`，与实际导出名一致 |

### 构建与环境

- SBCL 安装：从 SourceForge 下载 SBCL 2.6.7 二进制包，安装到 `~/.local/`
- Quicklisp：自动安装并配置 `~/quicklisp/local-projects/clvt` 软链接
- Python3 + NumPy：确认系统自带可用

### 测试

- 新增 `test/test-extensions.lisp`，包含 **19 个测试用例**，覆盖全部新增函数
- 原有 **815 个测试用例** 全部通过，无回归
- 总计 **834 个测试用例**，全部通过 ✅

---

## 2026-08-01 — MiMo (Xiaomi AI) 系统性测试与修复

### Bug 修复

| Bug | 严重程度 | 修复方式 |
|-----|----------|----------|
| `vt-copy-into` 慢速路径对非连续视图产生错误结果 | 🔴 严重 | 递归→显式迭代器，修复 SBCL 编译器对递归闭包的优化问题 |
| `vt-qr` Householder QR 分解崩溃 | 🔴 严重 | einsum→直接循环，分离 w 计算与 R 更新 |
| `vt-diagonal` 3D 张量数组越界 | 🔴 严重 | 重写为迭代实现，修复 out-ptr 不传播问题 |
| `vt-eig` Jacobi 特征值分解错误 | 🔴 严重 | 修正旋转角计算 (atan)、V 更新公式、off-diagonal 更新顺序 |
| `vt-convolve` 崩溃 | 🟡 中等 | 添加 `vt-contiguous` 保护负步长视图 |
| `vt-reduce` 空张量 prod 返回 0 | 🟡 中等 | 空张量防御改为返回 `init-val` |

### 性能优化

- `vt-copy-into` 慢速路径：递归→显式迭代器，性能持平或略优（大张量快 4%），修复正确性 bug
- `vt-qr`：用直接循环替代 einsum 调用，消除非连续视图的 stride 处理问题

### 新增功能

- `vt-cholesky`：Cholesky 分解（正定矩阵 → 下/上三角）
- `vt-eig`：对称矩阵特征值分解（Jacobi 旋转法 + atan）
- `vt-pinv`：Moore-Penrose 伪逆（基于 SVD）
- `vt-lstsq`：最小二乘解（基于 SVD）

### 补充导出

以下函数在原代码中已实现但未导出，本次补充到 package.lisp：

- `vt-asinh` / `vt-acosh` / `vt-atanh`：反双曲函数
- `vt-reciprocal` / `vt-negative` / `vt-lerp` / `vt-cbrt`：补充数学函数
- `vt-bit-and` / `vt-bit-ior` / `vt-bit-xor` / `vt-bit-not` / `vt-left-shift` / `vt-right-shift`：位运算
- `vt-fmax` / `vt-fmin`：NaN 忽略的逐元素极值
- `vt-nansum` / `vt-nanmean` / `vt-nanstd` / `vt-nanvar` / `vt-nanmax` / `vt-nanmin`：NaN 感知统计
- `vt-fill` / `vt-interp` / `vt-kron` / `vt-meshgrid`：填充、插值、克罗内克积、网格生成
- `vt-append` / `vt-insert` / `vt-delete`：追加、插入、删除

### 测试体系

构建了全面的自动化测试体系，共 **815 个测试用例**，全部通过：

| 测试套件 | 数量 | 覆盖范围 |
|----------|------|----------|
| `run_all_tests.lisp` | 143 | 基础函数 + NumPy 对比 |
| `run_param_tests.lisp` | 155 | 3D+ 参数化测试（多 axis、keepdims） |
| `nested-test.lisp` | 60 | AI/ML 函数组合（Linear、Attention、BatchNorm、MLP 等） |
| `robustness-test.lisp` | 178 | 边界条件（标量、空张量、NaN/Inf、数值稳定性） |
| `coverage-gap-test.lisp` | 97 | numpy/pytorch 覆盖差距（布尔索引、einsum 高级、torch.nn 模式） |
| `comprehensive-test.lisp` | 119 | 综合功能测试 |
| `auto-compare-test.lisp` | 63 | JSON 驱动自动对比 |

测试可通过 shell 脚本一键运行：
```bash
bash test/run-tests.sh          # 运行所有测试
bash test/run-tests.sh --list   # 列出所有测试套件
bash test/run-tests.sh --suite run_all_tests  # 运行指定套件
```

---

## 2026-07-30 — 初始版本

由'智谱清言'AI(GLM5+)和'DeepSeek'(v4pro) AI 共同编写。

### 核心架构

- 三大核心函数：`vt-einsum`、`vt-map`、`vt-reduce`
- 所有函数以 `vt-` 前缀命名
- 支持 4 种数据类型：`:int32`、`:int64`、`:float32`、`:float64`
- 完美的打印输出功能
- 零拷贝视图操作

### 已实现功能

- 张量创建（arange、linspace、zeros、ones、eye 等）
- 形状操作（reshape、transpose、squeeze、concatenate、stack 等）
- 索引与切片（ref、slice、where、nonzero 等）
- 算术运算（四则运算、三角函数、指数对数等）
- 比较与逻辑
- 归约与统计（sum、mean、std、var、median、percentile 等）
- 线性代数（matmul、solve、inv、det、qr、svd 等）
- einsum 爱因斯坦求和
- 激活函数（sigmoid、relu、tanh、gelu 等）
- 损失函数（softmax、cross-entropy、mse 等）
- 集合操作
- 随机数生成
- NaN/Inf 处理
