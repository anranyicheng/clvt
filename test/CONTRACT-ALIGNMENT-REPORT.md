# clvt 与 numpy / pytorch 契约对齐报告

本报告记录一轮针对 `indexing` / `manip` / `join` 三类结构操作的**契约审计与修正**：
逐条核对 clvt 的行为与 NumPy / PyTorch 的公开契约是否一致，凡是会造成
**静默错值**或**契约缺口**的，一律在源码层面对齐。

**核验方法**：每条结论先在 **SBCL + clvt** 下实测复现，再用 **NumPy 2.4.6** 跑等价
表达式取得权威期望值（`np.insert` / `np.where` / `np.diagonal` / `np.digitize` /
`np.select` / `np.choose` / `np.split` 族的源码与实测行为均逐行对照）。
判定标准只有一条：**是否与 numpy/pytorch 契约一致**，与测试是否通过无关。

## 一、对齐结果总表

| 函数 | 契约点 | 修正前实测行为 | 修正位置 |
| --- | --- | --- | --- |
| `(setf vt-ref)`、`(setf vt-slice)` | 广播视图只读（H4） | 对 `vt-broadcast-to` 出的列 `stride=0` 视图写入**不报错**且写穿底层：`base` 第 0 行由 `(0 1 2 3)` 变 `(0 1 99 3)`；标量分支同样漏防 | `src/indexing.lisp` |
| `vt-put` | 广播视图只读（H4） | 判定过严：`vt-slice` 的 `(:newa)` 产生 `size=1` 且 `stride=0` 的轴本不构成广播，却被误拒（numpy 允许 `a[np.newaxis].put(...)`） | `src/indexing.lisp` |
| `vt-roll` | 返回值别名语义 | 位移取模为 0（`shift=0`、`shift` 为轴长整数倍）或轴长为 0 时**直接返回入参本身**：`(eq (vt-data 结果) (vt-data 输入))` 为 `T`，写结果会回写输入 | `src/manip.lisp` |
| `vt-diagonal` | 支持 `:axis1`/`:axis2` | 只有 `offset`，`(vt-diagonal a :axis1 0 :axis2 1)` 抛 `Unknown &KEY argument`；无法表达 numpy 的任意对角平面 | `src/manip.lisp` |
| `vt-hsplit` | 整数段数需整除 | `(vt-hsplit (2,5) 2)` 返回 `(2,3)+(2,2)`；numpy 抛 `ValueError: array split does not result in an equal division` | `src/manip.lisp` |
| `vt-dsplit` | 整数段数需整除 + `rank>=3` | 轴长 4 分 3 段返回 `2+1+1`（numpy 抛 `ValueError`）；2-D 输入报的是轴越界而非 numpy 的维度错误 | `src/manip.lisp` |
| `vt-where`（单参） | 返回形态 | 返回 `(n, rank)` 坐标（argwhere 语义）；numpy 文档明确 `np.where(cond)` ≡ `np.asarray(cond).nonzero()`，应返回**按轴分组的索引元组** | `src/indexing.lisp` |
| `vt-select` | 接受标量 choice | `(vt-select (list c) (list 99))` 抛 `The value 99 is not of type CLVT:VT`；`np.select` 允许 `choicelist` 含标量 | `src/indexing.lisp` |
| `vt-choose` | 索引必须为整数 | 浮点 `indices` 被 `truncate` 静默截断（`0.5 → 0`）；numpy 抛 `TypeError: 'float' object cannot be interpreted as an integer` | `src/indexing.lisp` |
| `vt-take`、`vt-put` | 索引必须为整数 | 浮点索引被静默截断（`0.9 → 0`）；numpy 抛 `IndexError: arrays used as indices must be of integer (or boolean) type` | `src/indexing.lisp` |
| `vt-digitize` | 保持输入形状 | `(3,4)` 输入返回 `shape (12)`（被展平）；numpy 返回 `(3,4)` | `src/indexing.lisp` |
| `vt-digitize` | `bins` 接受序列 | `(vt-digitize x '(4 8))` 抛 `The value (4 8) is not of type CLVT:VT`；numpy 接受 list/sequence | `src/indexing.lisp` |
| `vt-insert` | `axis` 路径的 `moveaxis` | 用行主序 `vt-reshape` 顶替 `np.moveaxis(values, 0, axis)`。当 `values` 元素总数恰等于块元素数（`(2,1000)` 与块 `(1000,2)` 都是 2000）时形状校验通过，**静默**给出不同结果：1000×1002 输出 1998 处不同，首处 `(0,8)` clvt `1.0` vs numpy `1000.0` | `src/join.lisp` |

## 二、修正过程中额外对齐的契约

在对照 `numpy/lib/_function_base_impl.py::digitize` 源码时，发现原实现还有两处
未覆盖的 numpy 契约，一并修正：

1. **单调递减的 `bins`**。numpy 对单调不增的 `bins` 走
   `len(bins) - searchsorted(bins[::-1], x, side)`。例：
   `np.digitize([0,4,5,8,9], [8,4])` → `[2,1,1,0,0]`，原实现按递增处理给 `[0,0,0,2,2]`。
   现改为与 numpy 同构的二分实现（`side = 'left' if right else 'right'`）。
2. **单调性判定的边界**。numpy 对「全相等 `bins`」「空 `bins`」**不报错**（视为递增），
   仅当同时存在递增与递减跳变时才抛 `ValueError`。现按此语义实现：
   `np.digitize([0,1,3,9],[1,1,1])` = `[0,3,3,3]`。

## 三、刻意保留的分歧（已文档化）

- **`vt-select` 的 `condlist` 不强制布尔**：numpy 2.x 要求 `condlist` 元素为
  `boolean ndarray`（否则 `TypeError`），clvt 仍接受任意 dtype 并按非零真值处理。
  这是**宽松超集**，不会产生错值；强制会让大量既有 clvt 代码（用 `int8`/`int64`
  条件张量）失效，故保留并在 docstring 注明。
- **`vt-diagonal` 默认仍是 pytorch 末两轴**：numpy 的 `np.diagonal(a)` 缺省
  `axis1=0 / axis2=1` 属历史特例，batch 下与 torch 不同。默认保持 pytorch 兼容
  （不破坏既有调用），显式传 `:axis1`/`:axis2` 即得 numpy 语义。

## 四、因语义变更而同步更新的既有断言

这些断言原先**固化了修正前的旧行为**，属于「测试编码了错误契约」，已改写到正确的
numpy 期望上（不是为了让测试通过而放宽标准）：

| 文件 | 原断言 | 更新后 |
| --- | --- | --- |
| `test/core-test.lisp` | `vt-where(82): 单参 → (n,rank) 坐标` | 单参 → 按轴分组元组（= `vt-nonzero`） |
| `test/core-test.lisp` | `vt-take`/`vt-put` 用 `(vt-from-sequence '(0 2))`（默认 float64）作索引 | 显式 `:dtype :int64`（原断言依赖静默截断） |
| `test/numpy-convention-tests.lisp` | 「where 单参 → (n,1) / (n,2) 坐标」「与 argwhere 一致」 | 单参 → 按轴分组元组；并保留 `argwhere` 仍为 `(n,rank)` 的断言 |
| `test/test-all.lisp` | `vt-where 单参 → (n rank) 坐标` | 单参 → 按轴分组元组，且与 `vt-nonzero` 一致 |
| `CONVENTIONS.md` | D6 节把 `(n,rank)` 记为「已对齐 numpy」 | 更正为按轴分组列表（= `vt-nonzero`） |

`example/example.lisp` 无需改动（原本就用整数索引 / 整数列表）。

## 五、测试与验证

| 项 | 内容 |
| --- | --- |
| 新增套件 | `test/contract-regression-test.lisp`（**136** 断言，按行为分 11 节；期望来自 NumPy 实测；断言与分节均**异常安全**，实现损坏时报出全部失败而不中断） |
| 注册 | `test/run-tests.sh`（唯一入口）的 `SUITES` + `DEFAULT_ORDER` |
| 全量回归 | `bash test/run-tests.sh` → **40 / 40 套件通过**；`--all` → **42 / 42**（含 `benchmark-copy`、`performance-bench`） |
| 跨语言互操作 | `test/vt-manip-interop-test.lisp` + `test/verify_manip_numpy.py` → **195 / 195 通过、0 已知差异**（规模基准 `(vt-reshape (vt-arange 1000000) '(1000 1000))`，经 `vt-save` 落盘后由 NumPy 逐位校验） |
| **反向验证** | 把同一套件放到修正前的源码上运行 → **通过 49 / 失败 51**（退出码 1）；修正后 → **通过 115 / 失败 0**。证明断言确实捕获上表各行为差异，而非「为通过而写」 |
| 补丁可应用性 | 在基线 `6514cf9` 的干净克隆上 `git apply` → 干净应用，应用后 `bash test/run-tests.sh` **40 / 40 通过** |

### 5.1 测试入口统一到 `test/run-tests.sh`

`test/` 下的套件此前存在「已注册却从未被运行」与「需要另起脚本」两类问题，
本轮把**全部测试入口统一到 `test/run-tests.sh`**：

| 问题 | 处理 |
| --- | --- |
| 互操作套件是「Lisp 生成 + NumPy 校验」两阶段，原先只能靠独立脚本 `test/run-manip-interop.sh` 运行 | 在 `run-tests.sh` 新增 `PIPELINE_SUITES` 机制，注册为套件 `manip-interop-test` 并纳入默认列表；产物默认写临时目录、跑完自动清理（`CLVT_TEST_OUT` 可指定以保留）；**删除** `test/run-manip-interop.sh` |
| `performance-bench.lisp` 只能手工 `sbcl --load` | 注册为套件 `performance-bench`（与 `benchmark-copy` 同归 `BENCH_SUITES`，默认跳过、`--all` 执行） |
| `legacy-coverage-test` 已注册在 `SUITES`，却不在任何运行列表中 | 补进 `DEFAULT_ORDER`（实测 284/284 通过） |
| 运行列表是手写清单，容易与新套件漂移 | 重构为 `DEFAULT_ORDER` + `BENCH_SUITES`，并新增**启动期自检** `check_suite_wiring`：二者之并必须恰好覆盖 `SUITES` 全部条目，且流水线套件的两个阶段文件必须存在；不通过则直接报错退出、不跑任何套件 |
| 套件异常退出时只报 `?/?`，失败无据可查 | 失败分支改为输出 `exit=<code>`；当计数完全解析不出（典型的进程异常退出）时，额外打印原始输出末尾 20 行；流水线套件阶段 1 失败时同样打印输出尾部（含 Lisp 回溯） |
| `--list` 只列名字，看不出跳过与流水线 | 改为按「默认执行 / 性能基准」分组，并标注 `[两阶段: Lisp+NumPy]` |

自检生效性用「注入一个只登记不接线的假套件」实测验证：

```
[FAIL] 以下套件已注册但未接入运行列表，请补进 DEFAULT_ORDER 或 BENCH_SUITES:
  - forgot-to-wire-test
[FAIL] 套件接线自检未通过（见上方输出）
```

## 六、改动文件清单

**源码**
- `src/indexing.lisp` — 广播视图写入拦截（`(setf vt-ref)` / `(setf vt-slice)`，并放宽
  `vt-put` 的只读判定）、`vt-where` 单参返回形态、`vt-select` 标量 choice、
  `vt-choose` / `vt-take` / `vt-put` 整数索引校验（统一走新增的
  `%check-integer-index`，**空索引放行**，见 §七）、`vt-digitize` 形状保持 +
  序列 `bins` + 递减 `bins`（新增 `%bins-monotonicity` / `%digitize-searchsorted`）
- `src/manip.lisp` — `vt-roll` 拷贝语义、`vt-diagonal` 轴参数、
  `vt-hsplit` / `vt-dsplit` 整除与维度校验（新增 `%require-equal-division`）
- `src/join.lisp` — `%insert-axis` 按 numpy 语义重写（新增
  `%shape-broadcastable-to-p`），并区分标量 `obj` 与列表 `obj=[k]`；
  `vt-insert` 的 `positions` 改用 `%check-integer-index`（空 positions 放行）
- `CONVENTIONS.md` — 更正 `vt-where` 单参的契约描述

**测试**
- 新增 `test/contract-regression-test.lisp`、`test/CONTRACT-ALIGNMENT-REPORT.md`、
  `test/vt-manip-interop-test.lisp`、`test/verify_manip_numpy.py`、
  `test/MANIP-INTEROP.md`
- 更新 `test/run-tests.sh`（统一测试入口：新增两阶段流水线套件机制，
  把 `manip-interop-test` 接入默认运行列表；登记 `performance-bench`；
  补 `legacy-coverage-test` 到默认列表；新增启动期「运行列表必须覆盖全部
  已注册套件」的自检，防止漏接线）、`test/core-test.lisp`、
  `test/numpy-convention-tests.lisp`、`test/test-all.lisp`

## 七、上游合并后的回归修复：整数索引契约误伤「空索引」

上一轮把 `vt-take` / `vt-put` / `vt-choose` / `vt-insert` 的浮点索引改成报错
（对齐 numpy），但判断条件写成了「dtype 是浮点就报错」，于是把**空索引**也一并
拒了 —— 而 CL 侧表示「无索引」的自然写法默认正是 float64：

| 调用 | 修复前 | numpy 对应写法 | 应有行为 |
| --- | --- | --- | --- |
| `(vt-take a '())` / `#()` | 报错 | `np.take(a, [])` | 空结果 |
| `(vt-take a (vt-from-sequence '()))` | 报错 | —— | 空结果 |
| `(vt-take a (vt-zeros '(0)))` | 报错 | —— | 空结果 |
| `(vt-put z '() '())` | 报错 | `z.put([], [])` | no-op |
| `(vt-insert a '() v)` | 报错 | `np.insert(a, [], v)` | 原样返回 |

**修复**：抽出统一辅助 `%check-integer-index`（`src/indexing.lisp`），规则改为
**「元素数为 0 的索引不做整数检查」**，四个入口（`vt-take` / `vt-put` /
`vt-choose` / `vt-insert` 的 `positions`）统一调用。非空浮点索引仍然一律报错，
上一轮的目的（禁止 `0.5 → 0` 这类静默截断）完全保留。

理由两点：
1. 空索引没有任何元素会被截断 ⇒ 不存在静默错值风险，拒绝它不带来任何安全性收益；
2. 拒绝只会把「索引为空」这一正常用法判成错误，属纯粹的误伤。

### 与 numpy 的窄分歧（新增，方向为「更宽松且不影响正确性」）

numpy 的精确行为其实是「空 **list** 放行、空 **float64 数组** 报错」：

| | 空 list | 空 float64 数组 |
| --- | --- | --- |
| `np.take` | OK（空结果） | ERR（casting） |
| `ndarray.put` | OK（no-op） | ERR（casting） |
| `np.choose` | ERR（casting） | ERR（casting） |
| `np.insert` | OK（原样返回） | ERR（IndexError） |

clvt 对两者**统一放行**（顺序 API 取其宽松侧）。多出的那部分放宽是刻意的：
numpy 只对空 float64 数组报错是 `float64 → int64` 的 casting 副作用，并非有意
契约；而 CL 侧 `(vt-from-sequence '())` 这类「空索引」的自然写法恰好就是
float64，照搬 numpy 的 casting 行为会让用户无法用统一写法表达「无索引」。

### 验证

- `test/contract-regression-test.lisp` 新增一节 21 条断言：空 `list` / `#()` /
  空 float64 张量 / `(vt-zeros '(0))` / 带 `:axis` 的空索引 / 非空浮点仍报错 /
  浮点标量仍报错 / 正常整数索引不受影响；套件 115 → **136**。
- `test/vt-manip-interop-test.lisp` 新增 L 组 5 个用例（规模 1000×1000），
  由 NumPy 逐位校验，对应 `np.take(M, [])` / `np.take(M, [], axis=0|1)` /
  `M.copy().put([], [])` / `np.insert(M, [], M[0:1], axis=0)`：
  190 → **195**，全部通过；Lisp 端自检同步增加 8 条空索引 oracle（55 → 63）。
- 空张量的 `vt-save`（`:npy` / `:lisp` 两种格式）→ `vt-load` 往返正常，
  写出的 `.npy` 也能被 NumPy 直接 `np.load`（shape / dtype 一致）。
- **反向验证**：把新断言放到未修复的源码（上游合并提交 `fdc77fa`）上运行 →
  空索引一节 **11 条失败**（整体 125 通过 / 11 失败），而「非空浮点仍报错」
  的各条全部通过 —— 正是本次修复的差异面；修复后 136 / 0 全部通过。
- **顺带修掉测试基础设施的一处偶发崩溃**：冷编译整个库（新克隆目录 / CI 干净
  容器首次运行）时，SBCL 默认动态空间（实测 1 GB）会偶发
  `Heap exhausted, game over` 直接崩溃，进程带**空输出**退出 —— 运行器只能
  报一句无法追查的 `core-test: ?/? 通过, ? 失败`。冷编译实测 3 次：默认堆
  1 次崩溃；`--dynamic-space-size 4096` 3 次全稳。`test/run-tests.sh` 现统一
  传 `--dynamic-space-size`（默认 4096 MB，可用环境变量
  `SBCL_DYNAMIC_SPACE_SIZE` 覆盖），并在 `--help` 中写明。
- 全量 `bash test/run-tests.sh` → **40 / 40 套件通过**。
