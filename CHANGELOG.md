# CHANGELOG

本文件记录 clvt 库的每一次重大修改。

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
| `vt-arange` int64/int32 溢出直接存储（触发数组元素类型错误），int16/int8/uint 却回绕 | int64/int32 统一 `%wrap-*` 回绕（§8.5） | creation.lisp |
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

- 补齐 `vt-float-nan` / `vt-float-pos-inf` / `vt-float-neg-inf` / `vt-compute-logical-strides` 等「已导出但未定义」的悬空符号
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
