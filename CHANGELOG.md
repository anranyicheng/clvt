# CHANGELOG

本文件记录 clvt 库的每一次重大修改。

---

## 2026-08-15 — 架构重构与性能优化

### 架构重构

从零重构了整个库，将原来职责混杂的 17 个顶层 `.lisp` 文件重组为 `src/` 下按职责分层的 20 个模块（详见 README「架构」一节），保持全部 `vt-*` 公共 API 签名兼容：

- **内核层**：`dtype.lisp`（类型系统单一事实来源）、`core.lisp`（结构/步长/广播/连续判定/拷贝/填充）、`iterator.lisp`、`map-reduce.lisp`
- **功能层**：`creation` / `manip` / `indexing` / `join` / `elementwise` / `reduce-stats` / `setops` / `linalg` / `nn` / `random` / `rotate` / `io` / `extensions`

### 规范性修正

- `with-float-safe` 改为 `#+sbcl` / `#-sbcl` 可移植实现
- `nan.lisp` 不再依赖 `sb-kernel::float-nan-p` 等 SBCL 内部符号，改用 IEEE 754 可移植判定
- 补齐 `vt-float-nan` / `vt-float-pos-inf` / `vt-float-neg-inf` / `vt-compute-logical-strides` 等「已导出但未定义」的悬空符号
- `package.lisp` 导出按功能分组，`*vt-fun-list*` 自动收集
- 修复 `benchmark-copy.lisp` 中非法 FORMAT 指令 `~30-50x`

### 性能优化

- `vt-map` 重写为按输出元素类型特化（`(the (simple-array double-float (*)) ...)`），消除浮点装箱
- 恢复 `vt-fast-map` 编译期内联（`%inline1-loop` / `%inline2-loop` / `%cast-to`），`vt-+`/`vt-*`/`vt--`/`vt-/`、`vt-add`/`vt-sub`/`vt-mul`/`vt-div`/`vt-scale` 及一元数学函数（`vt-sin`/`vt-cos`/`vt-exp` 等）均内联算子，避免 `funcall` 装箱
- 基准（100 万元素，暖机后）：`vt-+` ≈ 0.004s、`vt-add` ≈ 0.009s、`vt-sin` ≈ 0.027s，达到或超越重构前水平；内联路径装箱量从 ~56MB 降至 ~8MB（仅结果张量本身）
- `vt-reduce` 内层递归循环按输入/输出元素类型特化（`macrolet` 生成 `(the (simple-array <type> (*)) ...)` 的 `recurse`，整数走 `truncate`、浮点走 `coerce`），消除 `vt-cast` 运行时 `ecase` 分派与通用 `aref` 的逐元素装箱；100 万元素下 `vt-sum` 提速约 1.5–1.75×、`vt-mean` 约 1.5–1.7×、`vt-amax` 约 1.3–1.5×

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
