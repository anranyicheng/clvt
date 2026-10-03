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

## 5. 与 numpy 的已知分歧（本轮探针实测登记）

以下均为**有源码注释的设计约定**，测试按 clvt 当前契约编码；
是否向 numpy 收敛留给维护者决策：

| 场景 | clvt 行为 | numpy 行为 | 状态 |
|------|-----------|------------|------|
| `mean/median` 空张量（全局或空轴） | 返回 NaN | ValueError（+warning） | 设计约定：NaN 作为空归约错误信号 |
| `amax/amin` 空张量（全局或空轴） | NaN 填充（整数提升 float64） | ValueError | 同上 |
| `argmax/argmin` 输入含空（即使输出为空，如 `(3 0)` axis=0） | 一律报错 | 仅归约轴为空时报错；`(3 0)` axis=0 返回 `(0)` | 设计约定：arg 归约空输入无定义 |
| 整数除以 0 | CL `division-by-zero` 错误 | 返回 0（+warning） | 更严格，可接受 |
| `vt-random-uniform` NaN low/high | FP-INVALID-OPERATION（非契约错误类型） | — | 报错达标，错误类型欠佳（§6 #4） |

## 6. 契约缺口清单（探针发现，待处理）

| # | 缺口 | 现状 | 建议 |
|---|------|------|------|
| 1 | `stack` axis=1 堆叠空 1D 崩溃，泄漏 CL 序列错误 | **本轮已修**（`vt-copy-into` 零尺寸写入早退，v0.3.3 同类写入侧） | 回归用例在 shape-degenerate-tests |
| 2 | `sum` 等 reduce：`:out` dtype 决定计算精度（float64 输入 + int32 out 按 int32 累加） | 静默，未报错 | 与 reduce 精度模型耦合，需专项决策：对齐 numpy（计算按输入提升，写入 cast 校验）或文档化 |
| 3 | `nonzero` 返回结构（单 (0) 张量的 list）与 numpy tuple 形态一致，但文档未写明 | 行为正确 | docstring 补返回结构说明 |
| 4 | `vt-random-uniform` NaN 参数报 FP 异常而非参数校验错误 | 仍是报错（L2 达标） | 把参数校验前移到任何浮点求值之前 |

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
