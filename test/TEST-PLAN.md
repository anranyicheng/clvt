# clvt 测试规划（TEST-PLAN）

版本：v0.3.4 起 ｜ 日期：2026-10-03 ｜ 语义基准：numpy（对齐处注明分歧）

本文档回答三个问题：**测什么**（类别矩阵）、**怎么算测过**（契约底线）、
**何时算发现回归**（逃逸率，而非断言数）。

---

## 1. 契约三底线（一切测试的判定依据）

| # | 底线 | 含义 | 反例（历史上真实出现过） |
|---|------|------|--------------------------|
| L1 | **内存安全无条件保证** | 任何输入（含非法）不得崩溃解释器、不得越界写入 | `(0) op (1)` 崩溃（v0.3.3 已修）；stack 空 1D 泄漏 CL 序列错误（本轮已修） |
| L2 | **非法输入 → 确定性报错，绝不静默错值** | 违反契约必须 signal，且错误信息可定位 | matmul `(6)` 静默错值、BCE NaN→1e-7（已修）；`sum float64 :out int32` 静默降精度（见 §5 缺口） |
| L3 | **合法输入 → 零检查税，选路不影响结果** | 优化路径与通用路径输出逐位一致；检查只允许 O(1) 且只做一次 | strided 循环进位错误、0-size 快路径边界（已修） |

## 2. 覆盖策略：类别矩阵 > 函数覆盖 > 断言数量

断言数量不构成鲁棒性证据——历史上全部已修 bug 都活在"全绿套件"的
盲区里，因为缺的不是某条断言，而是**整个概念类目**。因此规定：
每个公开 op 家族，至少在下列类目中各有一条用例：

| 类目 | 内容 | 主责套件 |
|------|------|----------|
| C1 | 空形状 / 退化形状（任一 dim=0、(1 1)、内部空维） | **shape-degenerate-tests**（本轮新增，表驱动） |
| C2 | size-1 广播 / 零 stride 读视图 | shape-degenerate-tests + run_param_tests |
| C3 | 非连续输入（slice/transpose 视图） | **property-strided-tests**（本轮新增）+ run_all_tests |
| C4 | `:out` 非连续 / 别名重叠 | out-contig-tests、test-overlap |
| C5 | dtype 边界（int8 回绕、小整型提升、混合 promote） | memsafety-tests、coverage-gap-test |
| C6 | NaN / ±Inf 传播 | nan-random-test、robustness-test |
| C7 | 错误路径（非法参数必须确定性报错） | **error-contract-tests**（本轮新增） |
| C8 | 选路不变量：strided ≡ contiguous（随机形状性质测试） | **property-strided-tests**（本轮新增） |
| C9 | 大形状 / 进位（>2^16 元素、高维 carry） | comprehensive-test、simd-test |
| C10 | 幂等 / 对合（flip∘flip、双转置、reshape 往返） | property-strided-tests |

## 3. 套件清单

**本轮新增（3 个）：**

| 套件 | 类目 | 方法 |
|------|------|------|
| `test/shape-degenerate-tests.lisp` | C1 C2 | 表驱动：op × {空、(3 0)、(0 4)、(1 0 2)、strided 空} 矩阵；期望值全部来自本轮探针**实测**并与 numpy 语义逐项对表 |
| `test/property-strided-tests.lisp` | C8 C10 C3 | 性质测试：固定种子随机形状上断言 `strided≡contiguous`；代数对合恒等式 |
| `test/error-contract-tests.lisp` | C7 | 每条已文档化契约错误一个 `check-error`；期望与探针实测一致 |

**v0.3.6 新增：**

| 套件 | 类目 | 方法 |
|------|------|------|
| `test/numpy-convention-tests.lisp` | C1 C5 C7 | numpy 2.1.3 对齐语义专项：空归约三分类、`:out` 精度解耦、mod/rem 零除 dtype 语义、true_divide IEEE、random 参数校验（46 断言） |

**既有套件职责（不变）：** run_all_tests（全函数基础值）、run_param_tests
（3D+ 参数化）、nested-test（组合）、robustness-test（C1/C6 部分形态）、
coverage-gap-test、comprehensive-test、auto-compare/numpy-compare（差异化
对照）、out-contig-tests（C4 回归）、memsafety-tests（C5）、nan-random-test
（C6）、benchmark-copy/performance-bench（性能基线，不入 CI 快车道）。

## 4. 新增规则

1. **bug 修复必须带回归测试进门**，命名 `bug-<主题>`（沿用 out-contig-tests
   风格）。无测试的修复视为未完成。
2. **语义分歧必须显式登记**在 §4 分歧表，禁止在套件里"悄悄按当前行为写期望"。
3. **度量用逃逸率**：每轮审查发现的 bug 数 / 测试本应拦住的数量；断言总数
   仅作参考，不作为完成度指标。
4. 任何套件失败即整体失败（run-tests.sh 以退出码聚合）。

## 5. 与 numpy 的已知分歧（探针实测登记）

**v0.3.6 重校准**：下表旧版中标记"设计约定"的三行已于 v0.3.6 全部向
numpy 2.1.3 收敛（实测基准：numpy 2.1.3 探针逐项对表），语义回归由
**numpy-convention-tests** 套件固化：

| 场景 | v0.3.5 行为 | numpy 2.1.3 实测 | v0.3.6 状态 |
|------|-------------|------------------|-------------|
| `mean` 空张量（全局或空轴） | NaN | NaN（+warning） | ✅ 一致（NaN 保留） |
| `amax/amin` 空张量 | NaN 填充（整数提升 float64） | **ValueError**（zero-size → no identity） | ✅ **改为 ValueError** |
| `amax/amin` 输出为空（如 `(3 0)` axis=0） | NaN 填充 | **空结果 shape (0)** | ✅ **改为返回空结果** |
| `argmax/argmin` 输出非空且归约区空（`(0)`、`(3 0)` axis=1） | 报错 | **ValueError**（attempt to get argmax of an empty sequence） | ✅ 一致（保留报错） |
| `argmax/argmin` 输出为空（`(3 0)` axis=0） | 一律报错 | **空结果 shape (0)** | ✅ **改为返回空结果** |
| 整数除以 0 | CL `division-by-zero` 错误 | 提升浮点，IEEE ±Inf/NaN | ✅ **改为 true_divide 语义**（IEEE，不再报错） |
| `vt-random-uniform` NaN low/high | FP-INVALID-OPERATION | — | ✅ **改为干净参数错误**（校验前移 + 屏蔽 FP 陷阱） |

**v0.3.6 仍保留的分歧（决策：保留）：**

| 场景 | clvt 行为 | numpy 行为 | 状态 |
|------|-----------|------------|------|
| `mod/rem` 零除 | 浮点语境 NaN / 整型语境 0（按 dtype 语义） | fmod/mod 混合规则 | ✅ 与 numpy 逐 dtype 对齐后一致 |

## 6. 契约缺口清单（探针发现）

| # | 缺口 | 现状 | 说明 |
|---|------|------|------|
| 1 | `stack` axis=1 堆叠空 1D 崩溃，泄漏 CL 序列错误 | **已修**（v0.3.5，`vt-copy-into` 零尺寸写入早退） | 回归用例在 shape-degenerate-tests |
| 2 | reduce 族 `:out` dtype 决定计算精度（float64 输入 + int32 out 按 int32 累加） | **已修**（v0.3.6）：按输入提升计算（compute-dtype），最后一步 cast 写入 `:out`；`def-vt-reduce` 三条内核路径统一 decoupled 预路径，mean/var/std/nanmean/nanvar/nanstd 同款解耦 | 与无 `:out` 结果逐位一致，numpy-convention-tests §2 固化 |
| 3 | `nonzero` 返回结构与 numpy tuple 形态一致，但文档未写明 | 行为正确 | docstring 补返回结构说明 |
| 4 | `vt-random-uniform/normal` NaN 参数报 FP 异常而非参数校验错误 | **已修**（v0.3.6）：`numberp` 检查在最外层，NaN/Inf 判定前移进 `with-float-safe`（屏蔽 FP-INVALID）；normal 补齐 mean 有限性 + std 非负有限校验 | error-contract-tests + numpy-convention-tests §5 固化 |

## 7. 运行方式

```bash
SBCL=tmp/sbcl-2.6.8-install/bin/sbcl   # 本工作区环境
# 单套件（加载即运行，退出码 0=全过）
$SBCL --noinform --non-interactive \
  --load clvt/test/shape-degenerate-tests.lisp
# 全量
bash clvt/test/run-tests.sh --quick
```

## 8. 类目矩阵推进方式

C1/C7/C8 本轮落地为表驱动套件后，**新 op 进入库的门槛** = 在三张表中
各加一行。矩阵空缺即工作队列，不再依赖审查者的记忆完整。
