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

## H.1 累计变更文件清单（第一～十七轮）

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
| `src/elementwise.lisp` | **第十二轮**：20 个拒绝型 + 10 个比较/极值 + 18 个一元，改为体内 `when`+`return-from` 分派（§16.32.1） |
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
| `src/linalg.lisp` | **第十四轮**：`vt-svd` 的 `max-sweeps`/`tol` 改 `supplied-p`，实数 50/1e-14 与复数 60/1e-15 各用自己验证过的缺省（§16.33.6） |
| `test/scan-first-element-test.lisp` | **第十四轮新增** 38 断言（实数 cumsum/cumprod 三条路径的符号零 + nan 族 + dtype 提升 + `:out` 非连续落点 + 复数回归），注册后共 53 套件 |
| `src/reduce-stats.lisp` | **第十五轮**：`%vt-cumulative` 的平面/沿轴两条路径合并到同一个按元素类型特化的 `RUN`（沿轴 196 → 4 ms）；新增 `%nanmedian-fold-axes` / `%nanmedian-multi`，`vt-nanmedian` 接通 axis 列表（§16.35.2、§16.35.4） |
| `src/extensions3.lisp` | **第十五轮**：`%e3-cumulative-skip-nan` 的 axis 分支改 strides 指针写回、op 内联、float64 直存（260 → 14 ms）；删除已成死代码的 `cumstep` flet（§16.35.3） |
| `src/complex-ops.lisp` | **第十五轮**：新增 `%cx-nanmedian-multi`，`%cx-op-nanmedian` 入口分流 axis 列表（复数多轴，复用实数侧折叠） |
| `test/run-tests.sh` | **第十五轮**：新增「全系统零 WARNING」编译体检门禁（`--skip-build` 可跳过）+ `cumulative-axis-test` 注册，共 54 套件 |
| `test/cumulative-axis-test.lisp` | **第十五轮新增** 70 断言（沿轴累积的朴素参照对照 + 性能护栏 + nanmedian 多轴/复数/`:out` 契约） |
| `src/reduce-stats.lisp` | **第十六轮**：`%vt-cumulative` 补 `vt-out-snapshot`（流式累积的 `:out` 错位重叠会把输入写坏，§16.36.1）；`vt-nanmedian` 入口拦截两处 dtype 静默降级（实数忽略 `:dtype`、复数取实部丢虚部，§16.36.2） |
| `test/cumulative-axis-test.lisp` | **第十六轮**：70 → **78 断言**（新增 C 组：`:out` 偏移重叠 2 + 完全别名 1 + dtype 拦截 4 + 退化多轴 1） |
| `src/core.lisp` | **第十七轮**：`%astype-cast-form` 增加源类型参数（复数→实取实部、实→复虚部补 0）；`vt-astype` 连续分支补 34 个复数子句（含同型 `replace` 块拷贝）；非连续回退改走新的 `%gather-contiguous`（§16.4.1） |
| `CONVENTIONS.md` | **第十七轮**：新增 §16.4.1「转换必须落在按元素类型展开的循环里」+ 判据 |
| `test/astype-cast-test.lisp` | **第十七轮新增** 663 断言（100 格矩阵 / 契约 / 非连续等价 / 倍率护栏），注册后共 55 套件 |
| `test/gen_astype_cast_expected.py` | **第十七轮新增** 期望值生成器（100 用例，numpy 2.4.6 实测） |
| `test/astype-cast-expected.sexp` | **第十七轮新增** 生成产物 |
| `test/run-tests.sh` | **第十七轮**：注册 `astype-cast-test`（SUITES + DEFAULT_ORDER） |

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

---

## H.3 §16 后半章的详细叙事（第五～十七轮）

> 为什么单独成节：这些内容原先住在 `CONVENTIONS.md` §16.21–§16.36，按
> 「症状 → 根因 → 修法 → 实测」写，与 `CHANGELOG.md` 的对应条目是**同一件事
> 的两个版本**，且其中的性能数字随机器/版本腐烂得最快。
>
> 现在 `CONVENTIONS.md` 只留**现行契约**（各节开头一句「来历见 HISTORY §H.3.x」），
> 过程、踩坑、消融实验与实测数据放这里。
>
> **本节的数字一律带测量条件**（机器、并发、是否含包装开销）。引用性能数字时
> 请写相对关系（「快一个数量级」），不要复制绝对值 —— 它们会腐烂。

### H.3.1 第十四轮：拆分的收尾、两处静默错值与门禁的可用性

前十三轮把「复数能不能算对」基本做完了（159/159 单一 DEFUN 分派、52 个套件全绿）。
本轮换了个角度：**不复盘语义，复盘工程状态** —— 全量重跑测试入口，逐条追问
「这条命令在一台干净的机器上真的能过吗」。

结论是 4 个缺陷，其中 2 个属于「测试/构建本身坏了」，1 个是静默错值，1 个是
会在用户进程里崩的陷阱问题。

#### H.3.1.1 A：`elementwise1/2` 是 `elementwise` 的完整副本（拆了个寂寞）

提交 `59ff055`（"拆分elementwise文件，避免编译需要过大内存"）**只加了两个文件，
一行没从原文件删**：

| 事实 | 数据 |
|---|---|
| `elementwise.lisp` 顶层定义 | 133 个（1530 行） |
| `elementwise1.lisp` + `elementwise2.lisp` | 39 + 46 = 85 个 |
| 这 85 个在 `elementwise.lisp` 里**是否还存在** | **100% 存在**（逐行归一化比对：422/422、361/361 行命中） |
| 唯一差异 | ew1 多 `(in-package :clvt)`；ew2 多一行 `(sb-ext::gc :full t)` |

后果有两条，都不是"只是占地方"：

1. **拆分目标完全没达成**。`elementwise.lisp` 仍是 1530 行，4 GB 堆照样在
   编译它时 `Heap exhausted, game over` —— 而 `run-tests.sh` 的默认值正是
   4096 MB。也就是说这个提交的全部收益为零，成本是 942 行死代码。
2. **重新引入了刚刚花两轮消灭的病根**。85 个 `DEFUN` 各存在两份，后加载者胜；
   `elementwise.lisp` 里那份改动**永远不生效**。这与 §16.32/§16.33 描述的
   「覆盖式分派」是同一个机制（`clvt.asd` 的顺序决定谁赢），只是从"符号函数被
   setf 覆盖"变成了"符号函数被后一个文件重定义"。§16.33.7 花一整列记下的
   「跨文件同名 defun：14 对 → **0**」在同一个提交里回到了 85 对。

**修法**：把 `elementwise.lisp` 第 596 行之后（`%pow-odd-int-p` 起，即 ew1+ew2
承载的整段）删掉，保留前 48 个定义，并在文件尾写清三段分工与这条铁律。

验证（`sbcl --dynamic-space-size 4096`，清空 fasl 缓存冷编译）：

| 指标 | 修前 | 修后 |
|---|---|---|
| 冷编译 | **Heap exhausted（4 GB）** | **成功** |
| `caught WARNING` | 20 | **0** |
| `caught STYLE-WARNING` | 96 | 96（均为既有的 unused-variable 类） |
| 跨文件同名 `defun` | 85 对 | **0** |

#### H.3.1.2 B：docstring 里一个裸 `"` 让函数体混进了 5 个未定义变量

`src/map-reduce.lisp` 的 `%weak-cast-needed-p` docstring 结尾写着：

```lisp
      构造函数收到 #C(1 0) 而报 "is not of type REAL"。"
```

中间那个 `"` 把字符串**提前闭合**了。于是 `is`、`not`、`of`、`type`、`REAL`
五个 token 成了**函数体里的五个 form**，真正的 docstring 只剩末尾一个 `"。"`：

```lisp
(BLOCK %WEAK-CAST-NEEDED-P
  CLVT::IS NOT CLVT::OF TYPE REAL "。" (AND (= (VT-SIZE X) 1) ...))
```

编译器报了 5 条 `undefined variable` + 「294 more uses」（SBCL 按组件累计），
共 20 条 WARNING。**危险的地方在于它不报错**：SBCL 把未定义变量当 special 变量
读，实测 `(%weak-cast-needed-p ...)` 仍返回 T —— 也就是说这个函数一直在依赖
「读未绑定 special 变量」的未定义行为工作。

**修法**：写成 `\"is not of type REAL\"`（转义）。修完 `caught WARNING` 归零。

> 教训：`check-parens` 查不出这类问题 —— 括号是平衡的，坏的是**字符串边界**。
> 唯一可靠的信号是编译期的 `undefined variable` 警告，所以「编译零 WARNING」
> 应当作为门禁，而不是当成噪音 muffle 掉（`run-tests.sh` 现在正是
> `(handler-bind ((warning #'muffle-warning)) ...)`）。

#### H.3.1.3 C：key-sync 门禁在 numpy 2.4.6 上硬失败，整套测试起不来

`run-tests.sh` 的第一步是 `check_key_sync.py`，它把「重新生成的探针文件」
与「提交的探针文件」做**逐字节比对**。而 `gen_probes*.py` 用 `str(np.float64)`
输出期望值 —— 这个格式化随 numpy 版本变：

```
numpy 2.5.0：0.529475129
numpy 2.4.6：0.5294751290000002      （同一条用例，差 1 ulp）
```

本机 numpy 2.4.6（也就是 §16 复数章的基准版本）因此必然报「codegen 漂移」，
门禁 `exit 1`，**52 个套件一个都跑不了**。§10.4.1 其实已经记录了这个现象
（"不要盲目重新生成再提交"），但门禁本身没有相应放宽。

**修法**：门禁默认改成**语义比较**，`--strict` 保留逐字节口径：

| 项 | 规则 |
|---|---|
| key 集合 / dtype / shape / 比较方式（`:exact`/`:tol`） | 必须完全一致，否则 FAIL |
| `:tol` 条目的数值 | 按 dtype 分级容差：float64 `1e-12`、float32 `1e-6`（实测跨版本最大漂移 float64 2.1e-16、float32 1.0e-7，均 <1 ulp） |
| `:exact` 条目的数值 | **逐字符相同**（该模式断言的就是逐位相等，文本一变必然运行期失败） |
| 生成器退出码非 0 | FAIL（否则文件不重写 → 「文本相同」的假通过，见 §10.4.1 的"静默失效"） |
| 生成的 `tmp/` 目录 | 跑完删掉（提交 60336c9 已把它从仓库移除，不该每次跑测试又冒出来） |

修后本机输出：`[key-sync] OK: ref_compute 651 key，5 套件查询 651，缺失 0`
+ 两条「格式化漂移」WARN（7 条与 8 条），套件得以正常启动。

#### H.3.1.4 D：实数线 `vt-cumulative` 没包 `with-float-safe`（用户进程会崩）

写 §16.34.2 的回归测试时顺手试了 `(vt-cumprod #(inf 0))`：

```
Unhandled FLOATING-POINT-INVALID-OPERATION ...
Operation was (* #.SB-EXT:DOUBLE-FLOAT-POSITIVE-INFINITY 0.0d0).
```

numpy 的行为是 `np.cumprod([inf, 0]) → [inf, nan]`（附 RuntimeWarning）。
按 §5.2「算术传播」与 §7 的平台陷阱约定，本库必须返回 nan 而不是抛条件。
同类问题在复数 `vt-median` 上踩过一次（漏包 `WITH-FLOAT-SAFE`，见 §16.30.1），这次是实数累积。

**修法**：拆成「外壳 + 内核」两个函数 ——

```lisp
(defun vt-cumulative (tensor op init-val &key axis dtype out)
  "…（契约不变）…"
  (with-float-safe
   (%vt-cumulative tensor op init-val :axis axis :dtype dtype :out out)))
```

不用「把 150 行循环体整体缩进一级」的写法，是因为那样每次语义改动都会淹没在
重排 diff 里。内核改名后必须同步 `(return-from %vt-cumulative result)` ——
漏改会得到 `return for unknown block: VT-CUMULATIVE`（本轮实测踩到）。

#### H.3.1.5 顺带：`vt-svd` 的死缺省（§16.33.6 待办的收尾）

第十三轮合并两处 `defun` 时，公开签名取的是实数线的 `50 / 1e-14`，而分派处写的是
`(%cx-op-svd matrix :max-sweeps max-sweeps :tol tol)` —— 于是 `%cx-op-svd`
自己的 `60 / 1e-15` 缺省**永远是死代码**，复数 SVD 实际一直跑在 50/1e-14 上，
与它被压测验证的参数不一致。

**修法**：改 `supplied-p`，两条路径各回各家：

```lisp
(defun vt-svd (matrix &key (full-matrices nil)
                          (max-sweeps nil max-sweeps-p) (tol nil tol-p))
  (when (%cx-any-p matrix)
    (return-from vt-svd (%cx-op-svd matrix :full-matrices full-matrices
                                          :max-sweeps (if max-sweeps-p max-sweeps 60)
                                          :tol (if tol-p tol 1e-15))))
  (let ((max-sweeps (if max-sweeps-p max-sweeps 50))
        (tol (if tol-p tol 1e-14)))
    ...))
```

不把两条路径统一到同一组数值，是因为两侧的缺省都是**各自验证过**的：实数线的
SIMD 内核用 1e-14 做过性能/收敛平衡，复数 Jacobi 的压测按 1e-15 通过。统一到
任一侧都是拿一侧的验证结论去覆盖另一侧。

#### H.3.1.6 测试

新增 `test/scan-first-element-test.lisp`（**38 断言**，已接入 `run-tests.sh`，
当前共 **53 个套件**）。它锁的是 A/B/C 之外的**语义**改动（首元素直拷 + 陷阱），
结构修复由既有的 `complex-dispatch-test`（跨文件同名 defun）与「编译零 WARNING」覆盖：

| 组 | 覆盖 |
|---|---|
| 1 实数 cumsum（19） | `[-0.0]` 的符号位、`[-0.0,0.0]`、2-D 三向（axis=0/1/nil）的**符号表 + 数值**、float32、轴长 1、负步长视图、空输入 |
| 2 实数 cumprod（5） | `[-0.0]`、`[inf,0] → [inf,nan]`、2-D axis=0 符号表 |
| 3 nan 族（6） | `nancumsum([-0.0])` 保号、`[nan,-0.0] → [+0,+0]`、`nancumprod([nan]) → 1`、2-D axis=0 |
| 4 dtype 契约（6） | int8 → int64 且不溢出、显式 `:dtype` 生效且不破坏保号、复数路径回归 |
| 5 `:out` 非连续（3） | 列视图落点逐位置核对、首元素符号、**其余位置不动** |

两条测试写法硬约束（NaN 用 `sb-ext:float-nan-p`、符号零显式判）已提炼为
`CONVENTIONS.md` §16.34.3。

### H.3.2 第十五轮：把第十四轮的三条遗留全部收口

第十四轮最后留了一张「已定位、未修」的表（§16.34.4）：沿轴累积慢 39×、
`nanmedian` 不支持多轴、编译 WARNING 无人管。本轮把三条做完，并顺带修掉
一条同源的隐性开销（`nancumsum` 的逐元素写回）。

#### H.3.2.1 出发点：同一个函数的两条路径差 196 倍

```
(vt-cumsum a)            ; 1M 元素，展平  → 1 ms
(vt-cumsum a :axis 0)    ; 1000×1000      → 196 ms     ← 同一份语义
```

第十四轮给 `axis=nil` 做了「按元素类型特化 + 首元素直拷」的循环，但沿轴那条
仍在跑通用回退：`(funcall op ...)` 每元素一次装箱、`(vt-cast cum final-dtype)`
每元素一次 ecase、`aref` 打在**未声明元素类型**的数组上走泛型分派。
三个 200ns/元素 叠起来，正好是 196 ms。

#### H.3.2.2 修法：平面与沿轴共用同一个 `RUN`

关键观察：两条路径的**内层**完全一样 —— 都是「从某起点，按固定步长走 N 个」。
差别只在「有多少条这样的段、每段起点怎么算」：

| 路径 | 段数 | 起点 | 步长 |
|---|---|---|---|
| `axis=nil`（展平） | 1 | `tensor.offset` | 1（要求全连续） |
| `axis=k` | `∏_{d≠k} shape[d]` | 除 k 维之外各维索引折算 | `strides[k]`（可为负） |

于是只写一份内核：

```lisp
(labels ((run (sp dp cnt istep ostep)   ; 从 SP 走 CNT 个到 DP，步长 ISTEP/OSTEP
           ...（首元素直拷 + 特化累加）...)
         (run-flat () (run in-offset out-offset (vt-size result) 1 1))
         (run-axis ()                       ; 除 AX 外各维做里程计
           (loop ... (run sp dp dim istep ostep) ...)))
  (if ax (run-axis) (run-flat)))
```

两个实现细节值得记：

1. **`labels` 而不是 `flet`**。`run-flat`/`run-axis` 要调 `run`，`flet` 里
   兄弟函数**互不可见**（会编译成对外部 `CLVT::RUN` 的调用，运行期
   `The function CLVT::RUN is undefined`）。这个坑当场踩了一次。
2. **`ax` 必须提前到 `let*` 里算**。宏在展开期求值，若在宏体内写
   `(if ax ...)`，`ax` 是词法变量、编译期不可见 → `The variable AX is unbound`。
   现在宏展开出的是**运行期** `(if ax (run-axis) (run-flat))`。

#### H.3.2.3 实测（1000×1000 / 1M 元素，SBCL 2.6.8，同进程取 5 次最优）

| 调用 | 修前 | 修后 | numpy 2.4.6 | 修前/后 |
|---|---:|---:|---:|---:|
| `cumsum axis=0` | 196 ms | **4 ms** | 3.3 ms | 49× |
| `cumsum axis=1` | 196 ms | **4 ms** | 3.3 ms | 49× |
| `cumprod axis=0` | 197 ms | **3 ms** | 3.8 ms | 66× |
| `cumsum axis=1`（float32） | 196 ms | **1 ms** | — | — |
| `cumsum axis=1`（int64） | 191 ms | **1 ms** | — | — |
| `cumsum axis=0`（转置视图，stride 非 1） | 196 ms | **2 ms** | — | — |
| `nancumsum axis=0` | 260 ms | **14 ms** | 6.2 ms | 18× |
| `cumsum`（展平，回归项） | 1–4 ms | **1–3 ms** | — | — |

`nancumsum` 那条另有三处改动（`%E3-CUMULATIVE-SKIP-NAN` 的 axis 分支）：
把 `(setf (apply #'vt-ref result coords) ...)` 换成 strides 驱动的指针（省掉
每元素一次坐标列表 cons + 一次通用坐标折算）、把 `(funcall op-fn ...)` 内联成
`(if sum-p (+ cum x) (* cum x))`、结果 dtype 是 float64 时直接存（省掉
`vt-cast`）。顺带删掉了变成死代码的 `cumstep` flet。

#### H.3.2.4 `nanmedian` 多轴：折叠成 2-D，复用单轴内核

numpy 的 `nanmedian(a, axis=(0,1))` 合法，而 clvt 两侧（实数 `vt-nanmedian`、
复数 `%cx-op-nanmedian`）都只支持单轴，多轴直接报「暂不支持」。

**正确性前提（本轮的裁决依据）**：中位数对归约区内元素的**顺序不敏感**
（内部本来就要排序）。所以

> 置换 (保留轴…, 归约轴…) → 重塑成 `(H, D)` → 沿单轴 1 归约 → 重塑回目标形状

与「直接沿多轴取中位」等价。这条前提让改动量从「再写一套多维 fiber 遍历」
降到「一个 20 行的折叠函数」，并且复数侧可以**完整复用**同一份折叠逻辑
（折叠只做置换与重塑，与 dtype 无关）：

```lisp
(defun %nanmedian-fold-axes (tensor axes keepdims) ...)  ; 共用
(defun %nanmedian-multi     (tensor axis keepdims out) ...)      ; 实数
(defun %cx-nanmedian-multi  (in axes keepdims out dtype) ...)    ; 复数
```

`vt-reshape` 对非连续视图会自动落地，所以任意 strides / 负步长 / 偏移的输入
都成立（实测转置输入与连续副本结果一致）。

**验证**（numpy 2.4.6 实测值，数据用 `a[i,j,k]=100i+10j+k` 保证可精确表示）：

| 用例 | numpy | clvt |
|---|---|---|
| `axis=(0,1)` | `[60 61 62 63]` | 一致 |
| `axis=(1,2)` | `[11.5 111.5]` | 一致 |
| `axis=(0,2)` | `[51.5 61.5 71.5]` | 一致 |
| `axis=(0,1,2)` | `61.5`（0 维） | 一致 |
| `axis=(0,1) keepdims` | `(1,1,4)` | 一致 |
| `b[0,0,0]=NaN, axis=(0,1)` | `[100 61 62 63]` | 一致 |
| 复数 `z=a+i(i+2j+3k), axis=(0,2)` | `[51.5+5j 61.5+7j 71.5+9j]` | 一致（字典序） |
| 负数轴 `(-2,-1)` / 乱序 `(2,1)` | 与 `(1,2)` 相同 | 一致 |
| 非连续 `:out` | — | 落点正确、未越界 |

#### H.3.2.5 编译体检门禁：WARNING 归零才算过

第十四轮那个「裸引号」事故的本质是：**编译器报了 20 条 WARNING，运行期完全
正常**（SBCL 把未定义变量当 special 变量读，函数照样返回 T）。只要没人看编译
输出，这类缺陷可以永久潜伏 —— 而 `check-parens` 查不出（括号是平衡的），
numpy 对拍也查不出（数值是对的）。

`run-tests.sh` 现在在跑套件前先做一次全系统编译体检：

| 项 | 规定 |
|---|---|
| 触发 | 每次运行都做（`--skip-build` / `CLVT_SKIP_BUILD_CHECK=1` 可跳过） |
| 口径 | `asdf:compile-system :clvt :force t`，**必须真的重编译** |
| 拦截 | `^; caught WARNING:` 计数 > 0 → FAIL 并打印上下文 |
| 放行 | `STYLE-WARNING`（unused variable 类，现有 96 条）不计 —— 噪声化的门禁等于没门禁 |
| 顺带收益 | 这一次重编译产出的 fasl 被后续套件直接复用，总耗时基本不额外增加 |

**踩到的坑（值得记）**：第一版把编译包在
`(handler-bind ((warning #'muffle-warning)) ...)` 里（照抄套件里的写法）——
结果 96 条 STYLE-WARNING 全部消失、门禁恒报「0 条」。**muffle 把编译期告警
一起吞了**。门禁必须吃原始输出。这一条已写进脚注里，防止后人"顺手加个 muffle
让输出干净点"。

反向验证（故意引入 `(progn undefined-scratch-variable 1)`）：

```
[FAIL] 编译产生 1 条 WARNING（STYLE-WARNING 96 条，不计）
    17526-;   undefined variable: CLVT::UNDEFINED-SCRATCH-VARIABLE
[FAIL] 编译体检未通过（先消除 WARNING 再跑套件；临时跳过用 --skip-build）
```

### H.3.3 第十六轮：`:out` 重叠保护的缺口、两处 dtype 静默降级，以及一次负结果

第十五轮把「能算对」和「跑得快」两条线都推到了头，剩下的三条（复数 matmul
SIMD、`real/imag` 视图、复数逐元素 SIMD）都是内核级工作。本轮换到**契约层**
扫了一遍 —— 抓到了两个真缺陷，都在「不报错、但结果是错的」这一类。

#### H.3.3.1 A：流式累积没有 `:out` 重叠保护（数据被自己写坏）

§4.3 的硬约定是「输出与任一输入在物理区间上重叠 → 先做 `vt-copy` 快照」，
理由是 `np.add(z, z, out=z[::-1])` 这类调用必须等价于「先读全量、再写」。
`vt-map` 与 SIMD matmul 都走 `vt-out-snapshot`，但 `reduce-stats` 的
`vt-cumulative` 是**流式**的（读到第 i 个就写第 i 个位置），一旦 out 与输入
错位重叠，先写的结果就会污染还没读到的输入：

```lisp
;; base = (1 2 3 4 5 0 0 0)；in = base[0:6]，out = base[1:7]
(vt-cumsum in :out out)
;; 修前 → (1 1 2 4 8 16 32 0)      ← 输入被自己覆盖，越滚越大
;; 修后 → (1 1 3 6 10 15 15 0)     ← 先快照，等于 cumsum((1 2 3 4 5 0))
```

为什么只有实数 `cumsum`/`cumprod` 中招 —— 这一栏值得记住，它解释了「为什么
同样的契约在别的函数上没出问题」：

| 函数 | 结构 | 重叠安全 |
|---|---|---|
| `vt-cumsum` / `vt-cumprod`（实数） | 单趟流式读改写 | ❌ → 本轮修 |
| `vt-nancumsum` / `vt-nancumprod` | 先算进临时 double 向量，最后一次写回 | ✅ 天然安全 |
| 复数 `%cx-op-cum*` | 同样两趟 | ✅ |
| `vt-map` / SIMD matmul / einsum | 显式 `vt-out-snapshot` | ✅ |

**修法**：`%vt-cumulative` 开头把输入过一遍 `vt-out-snapshot`：

```lisp
(let* ((tensor (if out (first (vt-out-snapshot out (list tensor))) tensor))
       ...)
```

`vt-out-snapshot` 内部用物理区间判重叠（`%vt-views-overlap-p`），无重叠时
零开销、不复制。

#### H.3.3.2 B：`nanmedian` 的两处 dtype 静默降级

| 调用 | 修前 | 问题 |
|---|---|---|
| `(vt-nanmedian 实数数组 :dtype :int32)` | 静默返回 float64 | 显式 `:dtype` 被**完全忽略** |
| `(vt-nanmedian 复数数组 :dtype :float64)` | 静默返回 float64（**取实部**） | 虚部消失，正中 §16.13 红线 |

第二条是最危险的一类：用户拿到的是「数值看起来合理」的实数结果，而虚部已经
无声无息地丢了。§16.4 早已为 ufunc 定了同一句话——「复数输入的结果 dtype
必须是复数类型」——`nanmedian` 当时没接上。

**修法**：公开入口做两个前置判定（消息里写清原因与替代做法）：

```lisp
(实数输入 + :dtype ≠ :float64)  → 报错「实数输入的结果 dtype 恒为 :float64…」
(复数输入 + :dtype 非复数)      → 报错「用实数 dtype 会把虚部静默丢掉」
```

缺省行为不变（实数 → float64，复数 → 保持复数 dtype），所以不影响既有调用。

> 顺带记录一个**未动**的 numpy 差异：`np.nanmedian(float32 数组)` 返回
> float32，而本库实数路径恒为 float64。这是既有契约（§16.10 明写），
> 改它属于行为变更，需要单独裁决；本轮只堵住「显式 `:dtype` 被忽略」。

#### H.3.3.3 C：复数 matmul 的一次负结果（省下一轮重复投入）

`%cx-matmul-*-rows` 的内层是 `(incf (aref c …) (* av (aref b …)))` —— 每个
`(p,j)` 都对 C 做一次读-改-写，k=256 时同一组元素被访问 256 次。直觉上换成
「4 个局部累加器、只在最后写回」应该大幅减少访存，而且累加次序仍是 p 递增、
结果**逐位相同**（§16.27.4 的 bit-exact 断言能直接验证）。

实测（禁用并行 GEMM，单线程，SBCL 2.6.8）：

| n | 旧内核（逐元素 incf） | 新内核（4 局部累加器） | 结论 |
|---:|---:|---:|---|
| 192 | 8.60 ms | **6.40 ms** | 好 1.34× |
| 256 | **15.67 ms** | 17.33 ms | 反而差 10% |
| 384 | 66.00 ms | 68.00 ms | 持平 |

三档没有一致方向、差异在测量噪声内 —— **无稳定收益，已回退**（不留代码）。
并行路径（默认开启）下两者都是 3 ms 级，差异被完全掩盖。

结论：这条路的瓶颈在复数乘法的**标量算术**（每元素 4 乘 2 加，且 SBCL 的
`(complex double-float)` 读写要经 16 字节装载/解包），不在 C 的访存。
要再快只有两条路：

1. **SIMD**：按 j 维向量化（f64.4 处理 2 个复数，需 shuffle 组 re/im），
   累加次序不变 → 仍满足铁律 1/2；
2. **SoA + 4 条实 GEMM**：算术量不变但把复数乘法拆成纯实数 FMA，能吃到
   实数线的 SIMD/并行内核；代价是**累加顺序改变**，与「与朴素循环逐位相等」
   冲突，需要先裁决这条不变量。

§16.36.2 已实测 SoA/打包比标量更慢（22.9 ms vs 13.5 ms），本轮又排除了
「局部累加器」这一条 —— 剩下能动的只有第 1 条。

---

### H.3.4 第十三轮：覆盖式复数层的收尾（159 / 159）

> 本轮的**契约结论**（结构约束、签名裁决、删除判据）留在
> `CONVENTIONS.md` §16.33；这里是完整过程与统计。

#### H.3.4.1 16 个符号按「该怎么改」分四类

| 类 | 数量 | 符号 | 处理方式 |
|---|---:|---|---|
| A 纯冗余 | 2 | `vt-@` `vt-expt` | 两处 `defun` 语义完全相同（都只是委托 `vt-matmul` / `vt-pow`），删 `complex.lisp` 那份 |
| B 错误死桩搬迁 | 4 | `vt-real` `vt-imag` `vt-conj` `vt-angle` | `extensions3.lisp` 那份写着「本库无复数 dtype，恒返回输入本身」——**是错的**；把 `complex.lisp` 的权威实现搬到归属文件，删桩 |
| C M1 匿名包装合并 | 2 | `vt-log` `vt-float-power` | 包装体（含 `:base` / 复数幂）并进归属文件的 `defun`，`return-from` 分派 |
| D M2 同名覆盖 + `%cx-orig` 转发 | 8 | `vt-view` `vt-average` `vt-clip` `vt-clamp` `vt-fromiter` `vt-unique` `vt-nanmedian` `vt-matrix-rank` | 复数侧实现改名 `%cx-op-*`，归属文件插守卫；**同时把两处 lambda-list 合并成一份** |

B 类是这一轮最值得记的一笔：那 4 个桩的正确性完全靠「后加载者胜」维持。
有人调整 `clvt.asd` 的文件顺序，`(vt-conj #C(1 2))` 就会静默返回 `#C(1 2)`。
**覆盖不仅掩盖重复，还会掩盖错误实现** —— 这是它比死代码更危险的地方。

#### H.3.4.2 迁移中发现的新缺陷：`vt-log` 的 `:base` 一直在除零

```
原实现: (complex (/ (realpart l) (log base 1.0d0)) …)
```

CL 的 `(log x base)` 是**两参数对数** `log_x(base)`，于是这里算成了
`ln(base) / ln(1) = 0.693 / 0`。浮点陷阱被 `with-float-safe` 屏蔽，除法得到 `Inf`，
再乘进结果变成 `0+0i`：

| 表达式 | 整改前 | 整改后 | numpy |
|---|---|---|---|
| `(vt-log #C(8 0) :base 2)` | `#C(0.0d0 0.0d0)` | `#C(3.0d0 0.0d0)` | `3+0j` |

正确写法是 `(log base)`（自然对数）。**这个 bug 在覆盖式架构下活了整整五轮**，
原因是没有任何测试覆盖「复数 + 非缺省 base」这一格 —— 而它属于最常用的换底调用。
本轮把它补进 `complex-reload-test` 的探针，并修正实现。

#### H.3.4.3 S7 验收：重载单文件不丢复数（新套件）

`test/complex-reload-test.lisp`（14 断言）—— 这是本轮**唯一真正的验收**。
同一 image 里逐个 `load` 12 个源文件（`dispatch` / `complex-kernels` / `manip` /
`elementwise` / `reduce-stats` / `setops` / `linalg` / `extensions` / `extensions3` /
`complex` / `complex-ops` / `complex-linalg`），每轮跑 19 个探针：

| 探针组 | 覆盖 |
|---|---|
| 第十二轮代表（3） | `sqrt(-1+0i)`、`sum`、`maximum` 字典序 |
| 第十三轮 16 个符号 | `@` `expt` `real` `imag` `conj` `angle` `log`（含 `:base`）`float-power` `view` `average` `clip` `clamp` `fromiter` `unique` `nanmedian` `matrix-rank` |
| 结构 | 全库无 `(setf (symbol-function …))` 代码行 |

在整改前，这一条在「重载 `elementwise.lisp`」那一轮**必然失败**（`sqrt` 变 NaN float64）。
现在 14/14 全过。

#### H.3.4.4 结果

| 项 | 第十二轮后 | 本轮后 |
|---|---|---|
| 单一 `DEFUN` 分派的公开符号 | 143 / 159 | **159 / 159** |
| `(setf (symbol-function …))` 代码处 | 2 | **0** |
| `%cx-orig` 调用点 | 16 | **0** |
| `eval-when` 预捕获清单 | 3 个文件、124 条 | **0** |
| 跨文件同名 `defun` | 14 对 | **0** |
| `%cx-def-*` 宏 | 0 | 0 |
| 测试套件 | 51 | **52**（新增 `complex-reload-test`） |
| 源码行数 | 24,624 | 24,878 |

导出符号数不变（367），`src/` 净增 254 行 —— 增量全部是注释与 docstring：
删掉 659 行覆盖代码与死桩，新增 373 行分派守卫、内核文件与新测试。

---

### H.3.5 第十二轮：拆掉覆盖式复数层（第一阶段）

#### H.3.5.1 为什么必须拆

第十二轮之前，367 个导出里有 **159 个（43.3%）** 在运行期被替换。三处实证：

| # | 现象 | 数据 |
|---|---|---|
| 1 | 规模 | `*cx-originals*` 159 条，**全部是导出符号**，内部符号 0 条 |
| 2 | 静态不可分析 | 静态扫描（重名 `defun` ∪ 宏调用）只找到 155；`vt-argmax`/`vt-argmin`/`vt-nanargmax`/`vt-nanargmin` 的实数实现是**宏生成的**，源码里没有字面 `defun`，只能问运行时注册表 |
| 3 | **重载即静默丢复数** | 只重载 `src/elementwise.lisp`（改实数实现时最常见的动作）：`(vt-sqrt #C(-1 0))` 从 `complex128 #C(-0.0d0 1.0d0)` 变成 `float64 NaN`，**不报错** |

第 3 条是根因：`(setf (symbol-function 'vt-sqrt) <匿名 lambda>)` 装的包装，会被重载的实数 `defun` 整体覆盖；覆盖之后符号直接指向实数实现，复数分支永久消失。

#### H.3.5.2 三种机制（现已拆掉两种）

| 机制 | 位置 | 形态 | 影响面 |
|---|---|---|---|
| M1 宏生成包装 | `complex.lisp` / `complex-ops.lisp` | `(setf (symbol-function ',name) (lambda (&rest keys &key ...)))` | 90 个；匿名 lambda → `declaim inline` 失效 |
| M2 `defun` 重定义 | 三个 `complex*.lisp` | 同名字再写一遍 `defun`，体内 `(%cx-orig 'name)` 转发 | 65 个；语义取决于加载顺序 |
| M3 时序预捕获 | 三个文件的 `eval-when` | 手写 63 条清单，漏写即「转发给自己」挂死 | 无机械约束 |

#### H.3.5.3 目标形态

```lisp
(defun vt-sqrt (vt &key out dtype)
  "逐元素平方根。"
  (when (%cx-any-p vt)                    ; ← 唯一新增
    (return-from vt-sqrt
      (let* ((in (ensure-vt vt))
             (dt (%cx-result-complex-dtype (vt-dtype in) dtype "VT-SQRT")))
        (vt-fast-map (function %cop-sqrt) in :out out :dtype dt))))
  (%float-map #'sin-impl vt out (%infer-float-dtype vt dtype)))   ; 原 body 逐字不动
```

三个理由（决定了它是 `when` + `return-from` 而不是把 body 包进 `if`）：

1. `defun` 自带同名 block，`return-from` 天然可用 —— **原 body 的缩进与括号完全不动**，159 处改造可脚本化，diff 只增不减；
2. 单一 `defun` / 单一 lambda-list / 单一 docstring，**不存在「两处 lambda-list 漂移」**（见 16.32.5 的实证）；
3. 实数路径的代价是一次内联的 dtype 判断，比原来「funcall 匿名包装 + 同样一次判断」**净减少**。

三个新增的公共件都在 `src/dispatch.lisp`（`clvt.asd` 里紧随 `core`，早于所有归属文件）：

| 名字 | 作用 |
|---|---|
| `%cx-arg-complex-p` / `%cx-any-p` | 分派判据。VT 走内联两步判断，**list/vector 进 `%cx-deep-p` 递归** |
| `%cx-result-complex-dtype` | 复数数学函数的结果 dtype 规则（原在 complex.lisp，24 处归属文件要用，必须前置） |
| `%cx-reduce-dispatch` | 归约族的算子路由（14 个符号共用一个入口） |

> 判据为什么必须认 list：`(vt-cbrt (list #C(8 0)))` 是既有测试用例。浅判据
> （只看 VT）会把它送进实数路径 —— 那是 `%cx-tensor-p` 的老问题，本轮把分派
> 判据与「VT 快判」正式分开。

#### H.3.5.5 迁移中发现的 9 处 lambda-list 漂移（新 bug 类）

对比「同一符号的两处 `defun`」的 lambda-list，9 个不一致：

| 符号 | 实数版 | 复数版 | 后果 |
|---|---|---|---|
| `vt-svd` | `:max-sweeps 50 :tol 1e-14` | `:max-sweeps 60 :tol 1e-15` | **实数路径一直在用复数版的容差**（后加载者胜），且无人察觉 |
| `vt-matrix-rank` | `(tol 1e-10)` | `(tol nil)` | 默认判秩阈值不同 |
| `vt-clamp` | `(tensor vmin vmax &key …)` | `(vt &optional min-val max-val &key …)` | 签名根本不同，按哪个调用决定行为 |
| `vt-view` | `(vt &rest args)` | `(vt new-shape)` | 前者能吃位置参数，后者不能 |
| `vt-fromiter` | `(… (dtype :float64) …)` | `(… (dtype nil dtype-p) …)` | 缺省 dtype 不同 |
| `vt-nanmedian` | 无 `dtype` | 有 `dtype` | 传 `:dtype` 时一边报错一边接受 |
| `vt-angle` / `vt-round` / `vt-@` | 参数名不同 | 参数名不同 | 仅影响关键词名一致性 |

这是覆盖机制的第二个代价：**两处定义没有任何机制强制它们签名一致**。
16.32.3 的单一 `defun` 从结构上消灭这一类（只有一个 lambda-list，改不到别处去）。

> `vt-svd` 那一条建议单独处理：需要确认期望的默认容差，不能靠「谁后加载谁赢」。

#### H.3.5.6 测试

套件 `complex-dispatch-test`。逐条断言见该文件头注释；套件说明见 `bash test/run-tests.sh --list`。

> 第十三轮把剩下的 16 个做完，最终 **159/159**；进度与统计见 §H.3.4。

---

### H.3.6 批量复数 matmul 的两个静默错答案

做 §16.21 的差分时，`vt-einsum "...ij,...jk->...ik"` 与 `vt-matmul` 差了 **3.3**
（量级不对，说明不是精度问题）。三方对照（einsum / matmul / 手写三重循环）
定位到 `%CX-BATCHED-MATMUL` 的两个独立缺陷：

| 缺陷 | 现象 | 根因 |
|---|---|---|
| **offset 丢失** | 每个 batch 都读到基数组开头 → 结果全错 | `(vt-slice (vt-ravel a) ...)` 得到的 1D 片段**已连续**，`vt-contiguous` 零拷贝返回它，`offset` 保留非零；而 `%CX-MATMUL-RUN` 按扁平下标从 0 起算 |
| **不支持 batch 广播** | `(1,4,5)@(3,5,6)` 直接 `重塑失败: 元素总数不一致 (旧 0, 新 20)` | 原实现用 `reshape` 切 batch，隐含「两侧 batch 形状相同」 |

这与 `manip.lisp` 里 `%flat-offset0` 一节记录的是**同一类根因**
（「扁平下标 + 裸 AREF」遇到带 offset 的视图），只是这次发生在复数线代内核里。

修复（两处都改在源头，不做调用侧补救）：

1. `%CX-MATMUL-RUN` 签名加 `a-off / b-off / c-off`，`%CX-MATMUL-2D` 透传
   `(vt-offset a)` / `(vt-offset b)` / `(vt-offset res)`；
2. `%CX-BATCHED-MATMUL` 改为**按 strides 递推 batch 偏移**（不再 reshape 切片）：
   batch 维右对齐、缺的左侧补 1、维长 1 → stride 0（广播），
   与实数 SIMD 批量路径的广播语义一致。

`%CX-DOT-1D` 也顺手补了 offset（同样的假设，只是此前没被触发）。

---

---

### H.3.7 复数 einsum 的 GEMM 快路径（为什么慢一个量级）

#### 问题与实测

复数 einsum 从第三轮起就**正确**（结果 dtype 由输入统一推导，通用累加状态机
与元素类型无关），但一直**慢一个量级**：

| 表达式（256×256 complex128） | 改前 | 改后 | 倍数 |
|---|---|---|---|
| `vt-einsum "ij,jk->ik"` | **2526 ms** | **31 ms** | **81×** |
| `vt-matmul`（对照） | 33 ms | 33 ms | — |
| `vt-einsum "i,i->"` | 9 ms | **<1 ms** | ~30× |
| `vt-einsum "ij,ij->ij"`（1M 元素） | 138 ms | **8 ms** | **17×** |

根因不是「复数没优化」，而是**三条快路径的判据全是实数 dtype 谓词**
（`all-f64-p` / `all-f32-p` / `all-i64-p` / `all-i32-p`），复数直接落到
最底下的通用迭代状态机：每个输出元素都要走「补码递推 + 逐张量泛型 `aref`」，
而 `vt-matmul` 走的是元素类型特化的三重循环。

---

### H.3.8 einsum 实数 GEMM 接 SIMD（症状、根因与实测）

#### einsum 的实数 GEMM 快通道接入 SIMD

**症状**：同一台机器上，`(vt-einsum "ij,jk->ik" a b)` 比 `(vt-matmul a b)`
慢 **5.7~6.4×**（float64，1024²，单线程）；多线程下内存带宽掩盖掉一部分，
仍有 1.78×。复数路径上一轮刚接通，实数这条反而漏了。

**根因（两条路径互不相通）**

| | 走的通道 | 内核 |
|---|---|---|
| `vt-matmul` 2D@2D | `*SIMD-MATMUL-2D-FN*` | `simd-matmul.lisp`：AVX 4 宽 + FMA + 64×256×256 分块 |
| `vt-einsum` BMM | `%MATMUL-DF-FAST-KERNEL` | `linalg.lisp`：4 路手写展开，**无 SIMD、无分块** |

关掉 `*SIMD-MATMUL-ENABLED*` 后两者持平（0.93~0.98×），可直接证明差异
**全部**来自这两个内核，与 einsum 的通道判定、类型提升无关。

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

#### 实测（512² float64，µs，3 次最好值）

512² float64 **单线程**（µs，3 次最好值；单线程最接近核数少的机器）：

| 表达式 | 改前 | 改后 | 改前/`vt-matmul` | 改后/`vt-matmul` |
|---|---:|---:|---:|---:|
| `vt-matmul`（基准） | 31,141 | 35,596 | 1.00× | 1.00× |
| `"ij,jk->ik"`（NN） | 196,033 | **32,485** | **6.30×** | **0.91×** |
| `"ij,kj->ik"`（NT） | 381,070 | **29,818** | **12.24×** | **0.84×** |
| `"ij,jk->ki"`（转置输出） | 609,077 | **36,112** | **19.56×** | **1.01×** |

2048² **多线程**（32 核，µs）—— 规模放大后差距同样消失：

| 表达式 | 改前/`vt-matmul` | 改后/`vt-matmul` |
|---|---:|---:|
| `"ij,jk->ik"` | 1.78×（6,210,809） | **0.99×**（3,439,937） |
| `"ij,kj->ik"` | 1.71×（5,985,129） | **0.99×**（3,447,156） |
| `"ij,jk->ki"` | 13.82×（48,326,644） | **0.99×**（3,459,113） |

1024² 多线程 0.93~1.02×；256²（1.67e7，刚过 SIMD 阈值）0.95~1.02×。
小矩阵（8/16/32²，低于 `*SIMD-MATMUL-THRESHOLD*` = 50000）两者都走标量，无退化。

> 注：`"ij,jk->ki"` 的「改前」是**改动前就已经存在**的老问题（通用三重循环
> 内层跨行），不是上一轮引入的；本轮顺手按 ③ 修掉。

---

### H.3.9 复数 matmul 内核对齐实数线（消融与实测）

#### 症状

`vt-matmul` 的复数 2D 路径比实数慢得离谱 —— §16.14.3 记的是 256² 下
39×（那时实数已走 SIMD，复数还是标量三重循环）。把 SIMD 关掉、只比
**标量对标准量**，复数仍慢 **2.5–3×**（n=256：复数内核 24.2ms，
实数标量内核 6.2ms），按「每元素浮点工作量」换算，复数应当只慢 4×
—— 反过来说，复数内核**每 flop 的开销是实数标量的 6 倍以上**，这已
经不是「复数算术天生贵」能解释的量级。

#### 消融：三项差距全在「代码形态」，与复数无关

n=256、单线程、预分配缓冲区、只计内核本身（不含 `make-vt`/`contiguous`）：

| 变体 | 耗时 | GFLOPS | 加速 |
|---|---:|---:|---:|
| 旧内核：`safety 1` + 每元素重算下标 + 无展开 | **24.2 ms** | 5.0 | 1.00× |
| ① 只把循环降到 `safety 0`（索引可证不越界） | 22.0 ms | 5.5 | 1.10× |
| ①+② 行起点只算一次，内层用 `c-ptr`/`b-ptr` 递推 | 14.1 ms | 9.5 | 1.71× |
| ①+②+③ j 维 4 路展开（对齐实数内核的展开宽度） | **13.5 ms** | 9.9 | **1.79×** |

三项收益互不重叠。**实数线 `%MATMUL-DF-ROWS` 一开始就是这三件事**
（`(safety 0)`、指针递推、4 路展开，见 `linalg.lisp`），所以差距的根因
是「复数内核是按最朴素的教科书三重循环写的」，而不是复数表示本身：

- `safety 1` 下的 `aref` 每次都带边界检查，且阻止了向量化；
- `(+ c-row j)` / `(+ b-row j)` 每元素两次加法 + 一次乘法（`p*n`），
  在内层里占掉可观的指令预算；
- 无展开 → 循环控制指令占比高，也挡住乱序执行的重叠。
#### 实测（n=192，同进程比值，`test/complex-matmul-perf-test.lisp` 内）

| 项 | 耗时 | 说明 |
|---|---:|---|
| 新行内核（c128，单线程） | 5.32 ms | 9.6 GFLOPS |
| 旧内核形状（同上参照） | 10.18 ms | 5.0 GFLOPS → **1.92×** |
| 实数标量行内核（`%MATMUL-DF-ROWS`） | 2.46 ms | 5.8 GFLOPS → 新/实 = **2.16×** |

复数每元素 8 flop、实数 2 flop，**flop 比就是 4×**；实测 2.16× 说明
复数的**单位 flop 效率反而比实数标量高**（9.6 vs 5.8 GFLOPS），
墙钟多出来的 2.16× 小于 4× 的那部分由「复数一次 `aref` 搬 16 字节、
内层访存更省指令」抵掉了。

端到端 `vt-matmul`（含 `make-vt` + 清零，SBCL 2.6.8，32 逻辑核 / 4 核配额
的容器；绝对毫秒随机器负载波动较大，**看比值**）：

| n | 复数串行 | 复数默认（并行） | 实数 SIMD | 串行/实数 | 并行/实数 |
|---|---:|---:|---:|---:|---:|
| 256 | 14.8 ms | 2.6 ms | 1.3 ms | 11× | **2.0×** |
| 512 | 143 ms | 11.9 ms | 4.6 ms | 31× | **2.6×** |
| 1024 | 1241 ms | 342 ms | 81 ms | 15× | **4.2×** |

读法：

- **并行列**是「用户视角」的默认行为（`*enable-parallel-matmul*` 默认 T，
  与实数线同一套阈值），相对实数 SIMD 只剩 **2.0–4.2×**，1024 已经进入
  内存带宽瓶颈（复数每次乘加搬 16 字节，是实数的 8 倍）；
- **串行列 vs 实数 SIMD** 的 11–31× 是「标量 vs SIMD+分块」的差距，
  不是本轮的目标（§16.27.6）；
- 串行/并行的加速比在本沙箱里测出来是 3.6–12×，明显超过 cgroup 的
  4 核配额（`cpu.max=400000/100000`，但 `cpus_allowed=32`，实际可用核数
  随宿主负载浮动），因此**只作量级参考**。并行正确性有独立断言：
  256² 下并行与串行结果**逐位相同**（`test/complex-matmul-perf-test.lisp`）。

---

### H.3.10 Complex median 的三处静默错值（方法、症状、根因）

#### 方法：从「dtype 保真」走到「数值对拍」

前几轮的扫描都在查「复数入参会不会掉回实数实现」，查不出「dtype 对、数值错」。
本轮沿用第九轮的数值对拍，并把覆盖面从 190 条扩到 **262 条**，新增四个方向：

| 方向 | 条数 | 结果 |
|---|---:|---|
| NaN / Inf 全组合（nan* 家族、掩码、`nan_to_num`、含 NaN 排序、全 NaN 切片） | 37 | 抓到 **median 不传播 NaN** |
| `axis` 元组（`(0 1)`/`(1 2)`/`(0 2)`/负元组 × keepdims） | 17 | 全部一致 |
| `:out` × 非连续（列切片 / 转置视图 / 原地别名 / matmul） | 7 | 全部一致 |
| einsum 补全（`...i,...i->...`、`ii->`、`ijk,jk->i`、`ijk->kji` 等 11 种） | 11 | 全部一致 |

结论：**除 median 外，`axis` 元组、`:out` 非连续、einsum 都没有静默错值**。

#### 缺陷一：`vt-median` 的复数版有三处与自家实数版分叉

三处都**不报错**，都是静默错值：

| # | 症状 | 根因 |
|---|---|---|
| 1 | `(vt-median a)` 对 `(3 4)` 输入返回 **3 个中位、形状 (3)** | `(ax (if axis (vt-normalize-axis axis rank) (1- rank)))` —— `axis=nil` 被写成「末轴」，而 numpy 与**本库实数版**都是「先 ravel 再取全局中位」，返回 0 维标量。连带 `:keepdims t` 给出 `(3 1)` 而不是 `(1 1)` |
| 2 | 空 fiber（轴长 0）返回 **0+0i** | `lo = (floor (1- 0) 2) = -1`，`%cx-fiber-starts` 在 size=0 时返回 0 条，循环不进 → 结果保持 `make-vt` 的 0。实数版这边是显式 `nanv` |
| 3 | 含 NaN 时返回**某个真实元素**而不是 NaN | 复数版按字典序排序后直接取中位，没有 NaN 检查；numpy 与实数版都传播 NaN |

修法：`axis=nil` 走独立的全局分支（ravel → sort → 取中位，keepdims 用全 1 形状）；
空 fiber / 含 NaN 一律写 `NaN+NaN i`；逐 fiber 判断用新助手
`%CX-ELEM-NAN-P` / `%CX-HAS-NAN-P`。

> 顺带补了一个洞：**复数 `vt-median` 原本没包 `WITH-FLOAT-SAFE`**（实数版包了）。
> SBCL 默认开着 `OVERFLOW/INVALID/DIVIDE-BY-ZERO` 陷阱，于是 NaN 判定
> `(= x x)`、`inf + (-inf)` 都会直接抛 `FLOATING-POINT-INVALID-OPERATION`。
> 现在整个函数体与实数版一致地包在 `WITH-FLOAT-SAFE` 里。

---

### H.3.11 第十七轮：`vt-astype` 的复数转换为什么慢 8~15 倍

#### 症状

用户反馈「`vt-astype` 对复数的转换太慢」。基线（1M 元素，SBCL 2.6.8，同进程
6 次取最优）：

| 转换 | 耗时 | 同规模实数对照 |
|---|---:|---:|
| `c128→c128`（同 dtype！） | 17 ms | `f64→f64` 1 ms |
| `c128→c64` | 31 ms | `f64→f32` 1 ms |
| `c64→c128` | 26 ms | — |
| `c128→f64` | 25 ms | — |
| `c128→i64` | 34 ms | `f64→i64` 3 ms |
| `f64→c128` | 27 ms | — |
| `f64→c64` | 41 ms | — |
| 1000×1000 转置视图 `c128T→c64` | 35 ms | `f64T` 同类 3~4 ms |

numpy 2.4.6 同机同规模为 0.2~0.5 ms（内存带宽下限），但这不作为目标 ——
目标是「与自家实数线同量级」。

#### 根因：兜底分支吃掉了全部复数组合

`vt-astype` 的连续视图分支是一条手写的 `cond`，按 `(源元素类型, 目标元素类型)`
展开 `spec` 特化循环。它列了 8 种实数元素类型的组合，**一种复数都没有**：

```lisp
((and (equal in-et 'double-float) (equal new-lisp-type '(signed-byte 64)))
 (spec double-float (signed-byte 64)))
...
(t                                  ; ← 复数全部落在这里
 (let ((conv (vt-cast-fun new-dtype)))
   (dotimes (i size)
     (setf (aref new-data i) (funcall conv (aref in-data p))))))
```

兜底分支有三个叠加的代价：`in-data` / `new-data` 没有类型声明（泛型 `aref`
每次做数组分派）、`vt-cast-fun` 返回的是**闭包**（`funcall` 不可内联）、
闭包体内每个元素都要 `complexp` 判断 + `realpart`/`imagpart` 解包再 `coerce`
重组。1M 元素 = 1M 次装箱。

顺带暴露的第二处：**同 dtype 的复数转换也没有块拷贝分支**，于是
`(vt-astype z :complex128)` 这种「纯拷贝」比 `f64→f64` 慢 17 倍。

#### 修法（三层，都在 `src/core.lisp`）

1. **`%astype-cast-form` 增加源类型参数**，把 §16.4 的三条语义编进代码生成：
   目标复数 + 源复数 → 分量各自 `coerce`；目标复数 + 源实数 → 虚部补 `0.0d0/0.0s0`；
   目标实/整 + 源复数 → 先 `realpart` 再走原来的 `%coerce-int*` / `coerce`。
   实数路径生成的代码逐字不变（`spec` 只多传一个 `in-lt`）。
2. **补 34 个分支**：同型复数 2 个 `memcpy`（`replace` 块拷贝，interleaved 布局
   逐位相同）、复数↔复数 2 个、复数→实/整 16 个、实/整→复数 16 个。
3. **非连续视图新增 `%gather-contiguous`**：按 `strides` 做**同 dtype 特化收集**。
   同 dtype 时它就是答案（一趟）；跨 dtype 时作为中间副本，交给连续 cast。
   原来「逐元素 `funcall` + 泛型 aref」的游走只剩「未知元素类型」一条退路。

> 为什么不是「先 `vt-contiguous` 再递归」：`vt-contiguous` 走 `vt-copy-into`，
> 对非连续复数视图同样是泛型 aref —— 实测 `(vt-copy c128T)` 就要 17 ms，
> 先物化再转换合计反而更慢。所以收集这一步必须自己特化。

#### 实测（1M 元素，同进程 6 次取最优）

| 转换 | 优化前 | 优化后 | 倍数 |
|---|---:|---:|---:|
| `c128→c128` | 17 ms | 3 ms | 5.7× |
| `c128→c64` | 31 ms | 2 ms | 15.5× |
| `c64→c128` | 26 ms | 3 ms | 8.7× |
| `c64→c64` | 12 ms | 2 ms | 6.0× |
| `c128→f64` | 25 ms | 2 ms | 12.5× |
| `c128→f32` | 21 ms | 1 ms | 21× |
| `c128→i64` | 34 ms | 3 ms | 11× |
| `f64→c128` | 27 ms | 3 ms | 9.0× |
| `f64→c64` | 41 ms | 2 ms | 20× |
| `i64→c128` | 19 ms | 3 ms | 6.3× |
| 转置 `c128T→c128` | 21 ms | 5 ms | 4.2× |
| 转置 `c128T→c64` | 35 ms | 8 ms | 4.4× |
| 转置 `c64T→c128` | 28 ms | 8 ms | 3.5× |

实数线（`f64→f64` 1 ms、`f64→f32` 1 ms、`f64→i64` 3 ms）**不受影响**：
新分支只按元素类型追加，实数组合命中的仍是原来那些子句。

#### 一条负结果：转置视图的 16 MB 拷贝仍比 numpy 慢一个量级

`c128T→c128` 现在 5 ms，numpy 0.44 ms。原因不是元素装箱（已消除），而是
访存次序：转置视图按行主序走 = 读一个 16 B 元素跳 16 KB，每元素一次 cache line。
手写实验（`test` 外的临时脚本）证实：同样 1M 元素，**列主序走（读连续、写跨行）
0 ms 级、行主序走 2 ms**，而 `vt-copy` 的泛型 walker 17 ms —— 也就是
「特化」解决了 8 倍，「循环次序」还压着 2~4 倍。

后者要动 `vt-copy-into` 的循环次序选择（按 strides 选「读连续」的轴序，
或分块转置），影响面覆盖全库所有 strided 拷贝，**本轮不做**，留在
`§16.37` 的未完成项里。

#### 顺带修掉的一处文档口径

§16.4 原来写「`vt-astype` 的所有快/慢路径都经由此出口」—— 读者会理解成
「运行期调用 `vt-cast-fun` 的闭包也行」。实际契约是**语义统一**、**执行特化**，
已在 §16.4.1 写清，并给出判据（复数转换 ≤ 10× 实数线）。

#### 测试

新增 `test/astype-cast-test.lisp`（663 断言）：

| 组 | 内容 |
|---|---|
| 矩阵 | 10 dtype × 10 dtype = 100 格，每格断言 dtype / shape / 连续性 / 独立缓冲 / 逐元素值 |
| 契约 | 取实部、虚部补 +0.0、`-0.0` 保号、NaN 传播、NaN/Inf→整数 0、rank-0、空张量、广播视图 |
| 非连续 | 转置 / 翻转 / 切片 / 广播 4 种视图 × 8 目标 dtype，与「先物化再转换」逐位一致 |
| 护栏 | 复数转换 ≤ 8~10× 实数线（按倍率，不看绝对毫秒） |

期望值由 `test/gen_astype_cast_expected.py` 从 numpy 2.4.6 实测生成
（`test/astype-cast-expected.sexp`，100 条）；源值全部取 |v| ≤ 100 的整数/半整数，
使窄整数目标的回绕在 numpy 与 CL 两侧同义；非有限值不进矩阵（numpy 对
NaN/Inf→整数是平台相关的 C cast，本库契约为 0，用契约断言单独覆盖）。
