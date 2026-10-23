# HISTORY — clvt 编年史

> **定位**：本文件是**编年史**，不是契约。
> 契约（现行语义、不变量、错误文本）只在 `CONVENTIONS.md`；本文件记录
> 「某一轮改了什么、为什么那样改、当时的实测数字」，允许无限增长，
> **不保证与当前实现一致** —— 与 `CONVENTIONS.md` 冲突时以
> `CONVENTIONS.md` 为准（`CONVENTIONS.md` §0 的裁决原则）。
>
> 为什么单独拆出来：契约与编年史混在一份文件里时，历史状态会被当成现行契约
> 阅读（例如「einsum 不支持复数」这种已被推翻的表述会被当成事实）。
>
> 各轮的**摘要**同时存在于 `CHANGELOG.md`（按时间倒序、面向使用者）；
> 本文件保留的是**实现细节与文件清单**（面向维护者）。

---

## H.1 累计变更文件清单（第一～十六轮）

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
| `src/linalg.lisp` | **第十二轮**：`vt-dot`/`vt-matmul`/`vt-outer`/`vt-trace`/`vt-norm` 加体内分派；线代 9 个（`cholesky det eig inv lstsq lu pinv qr svd`）+ `vt-solve` 抽复数臂为 `%cx-op-*` |
| `src/setops.lisp` | **第十二轮**：集合族（`in1d isin intersect1d setdiff1d setxor1d union1d`）加体内分派 |
| `src/extensions.lisp` | **第十二轮**：`vt-inner`/`vt-tensordot` 加体内分派 |
| `src/extensions2.lisp` | **第十二轮**：`vt-standardize` 加体内分派 |
| `src/manip.lisp` | **第十二轮**：`vt-repeat` 加体内分派 |
| `src/complex-linalg.lisp` | **第十二轮**：10 个公开 `defun` 抽成 `%cx-op-*` 内部实现（仅 `vt-matrix-rank` 因 `&optional` 未迁） |
| `test/complex-dispatch-test.lisp` | **第十二轮新增** 15 断言（结构契约：143 个符号必须是具名 DEFUN / list 入参 / 实数不回归 / 复数抽样 / `:out`），注册后共 51 套件 |
| `src/elementwise.lisp` | **第十四轮**：删掉与 `elementwise1/2` 重复的 85 个定义（原 1530 行 → 614 行），文件尾补三段分工与「跨文件同名 defun 必须为 0」的铁律说明（§16.34.1） |
| `src/map-reduce.lisp` | **第十四轮**：`%weak-cast-needed-p` 的 docstring 里裸 `"` 转义为 `\"`，消除 5 个未定义变量混入函数体（20 条编译 WARNING → 0，§16.34.2） |
| `test/check_key_sync.py` | **第十四轮**：codegen 门禁改「语义比较 + `--strict` 逐字节」双口径；`:tol` 按 dtype 分级容差；生成器退出码非 0 直接 FAIL；清理生成器遗留的 `tmp/`（§16.34.3） |
| `src/reduce-stats.lisp` | **第十四轮**：`vt-cumulative` 拆「外壳（with-float-safe）+ `%vt-cumulative` 内核」；快路径 / axis / 展平三处循环改为**首元素直拷**（§16.34.2、§16.34.4） |
| `src/extensions3.lisp` | **第十四轮**：`%e3-cumulative-skip-nan` 的 axis 与展平两处循环改为「掩码后首元素直拷」，`nancumsum([-0.0])` 保住符号位（§16.34.4） |
| `src/linalg.lisp` | **第十四轮**：`vt-svd` 的 `max-sweeps`/`tol` 改 `supplied-p`，实数 50/1e-14 与复数 60/1e-15 各用自己验证过的缺省（§16.34.5） |
| `test/scan-first-element-test.lisp` | **第十四轮新增** 38 断言（实数 cumsum/cumprod 三条路径的符号零 + nan 族 + dtype 提升 + `:out` 非连续落点 + 复数回归），注册后共 53 套件 |
| `src/reduce-stats.lisp` | **第十五轮**：`%vt-cumulative` 的平面/沿轴两条路径合并到同一个按元素类型特化的 `RUN`（沿轴 196 → 4 ms）；新增 `%nanmedian-fold-axes` / `%nanmedian-multi`，`vt-nanmedian` 接通 axis 列表（§16.35.2、§16.35.4） |
| `src/extensions3.lisp` | **第十五轮**：`%e3-cumulative-skip-nan` 的 axis 分支改 strides 指针写回、op 内联、float64 直存（260 → 14 ms）；删除已成死代码的 `cumstep` flet（§16.35.3） |
| `src/complex-ops.lisp` | **第十五轮**：新增 `%cx-nanmedian-multi`，`%cx-op-nanmedian` 入口分流 axis 列表（复数多轴，复用实数侧折叠） |
| `test/run-tests.sh` | **第十五轮**：新增「全系统零 WARNING」编译体检门禁（`--skip-build` 可跳过）+ `cumulative-axis-test` 注册，共 54 套件 |
| `test/cumulative-axis-test.lisp` | **第十五轮新增** 70 断言（沿轴累积的朴素参照对照 + 性能护栏 + nanmedian 多轴/复数/`:out` 契约） |
| `src/reduce-stats.lisp` | **第十六轮**：`%vt-cumulative` 补 `vt-out-snapshot`（流式累积的 `:out` 错位重叠会把输入写坏，§16.36.1）；`vt-nanmedian` 入口拦截两处 dtype 静默降级（实数忽略 `:dtype`、复数取实部丢虚部，§16.36.2） |
| `test/cumulative-axis-test.lisp` | **第十六轮**：70 → **78 断言**（新增 C 组：`:out` 偏移重叠 2 + 完全别名 1 + dtype 拦截 4 + 退化多轴 1） |

---

---

## H.2 架构演进：覆盖式分派 → 单一口径分派（第一～十三轮）

> 本节是从 `src/complex.lisp`、`src/complex-kernels.lisp`、`src/dispatch.lisp`
> 的文件头注释里迁出来的**历史叙述**。代码里只保留现行设计的一句话说明。

早期（第一～十一轮）复数支持靠「运行期重定义公开函数」实现，先后用过三种机制：

1. `%CX-DEF-*` 宏 + `(setf (symbol-function …))` 装匿名包装（三个文件里 159 个
   导出符号中的一部分）；
2. 「同名 DEFUN 再写一遍 + `(%cx-orig 'name)` 转发」；
3. 一份 63 个名字的 `eval-when` 预捕获清单 + `*cx-originals*` 哈希表。

后果（实测）：只重载 `src/elementwise.lisp`，`(vt-sqrt #C(-1 0))` 会从
`#C(-0.0d0 1.0d0)` 静默变成 `float64` 的 `NaN` 且不报错；函数对象是匿名
LAMBDA，`declaim inline` 对调用点失效；哪些符号被覆盖无法静态枚举。

第十二～十三轮把它们全部拆成「归属文件里单一 DEFUN 体内分流」，删除顺序有讲究：

1. 先清掉所有 `(%cx-orig 'vtx)` 调用点（2 处，`vt-sum`）；
2. 再删三个文件的 `eval-when` 预捕获清单；
3. 最后删表与函数 —— 顺序颠倒会让「忘记删的调用点」在运行期变成
   「symbol-function 返回 NIL」而不是编译期就报错。

判据（全部应为 0）：`grep -c "setf (symbol-function" src/*.lisp`、
`grep -c "(%cx-orig" src/*.lisp`。
