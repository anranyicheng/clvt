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

## H.1 累计变更文件清单（第一～二十一轮）

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
| `src/core.lisp` | **第十八轮**：`%gather-contiguous` 的遍历改为「末轴 `DOTIMES` + 其余轴 odometer」（1M 转置收集 6 → 1 ms）；`%copy-generic` 的 typed-walk 补 `(complex double-float)` / `(complex single-float)` 两格（`vt-copy` 转置 17 → 6 ms，§16.4.1 第 5–7 条） |
| `CONVENTIONS.md` | **第十八轮**：§16.4.1 补 5/6/7 三条（拷贝同受约束、末轴必须展开、新增分派表的检查方法）+ 「16 字节 basher 缺失」的已知成本声明 |
| `src/reduce-stats.lisp` | **第十九轮**：`%select-double` 改三分区 + 访问量上限 4n（全零 200k 10.7 s → 1 ms，降序 400k 142 → 21 ms，§16.35.5） |
| `src/core.lisp` | **第十九轮**：新增 `%alloc-vt`（不填充分配）与 `%memcpy`（带 `(the …)` 声明的按元素类型块拷贝）；`vt-copy` / `vt-contiguous` / `vt-copy-into` 快路径改走它们（连续 c128 拷贝 12 → 4 ms，`flatten`/`roll`/`tile`/`pad` 同步受益，§16.4.1 第 8 条） |
| `src/join.lisp` | **第十九轮**：`%block-copy` 改为委托 `%memcpy`（保留原实参顺序），顺带获得复数两格 |
| `src/complex-kernels.lisp` | **第十九轮**：`%cop-max` / `%cop-min` / `%cop-fmax` / `%cop-fmin` 由 `complex.lisp` 前移至此（§16.33.2 的「标量内核前移」），带 `declaim inline` |
| `CONVENTIONS.md` | **第十九轮**：新增 §16.35.5（顺序统计的两条退化护栏）；§16.4.1 补第 8 条（块拷贝必须带类型声明）并更正「没有 16 字节 basher」的错误结论；§16.37 增补 U5–U8 |
| `test/order-stat-test.lisp` | **第十九轮新增** 266 断言（独立参照对拍 + 退化倍率护栏），注册后共 56 套件 |
| `test/run-tests.sh` | **第十九轮**：注册 `order-stat-test` |
| `src/complex-kernels.lisp` | **第二十轮**：极值内核的 NaN 判定由边界版 `%cisnan` 改为免屏蔽 `%cx-nan-p`（`vt-maximum` 804 → 7 ms）；九个比较/谓词标志内核（`%cx-eq-flag` 等）由 `complex.lisp` 前移至此（`vt-=` 60 → 4 ms，§16.8.1 第 1/4 条） |
| `src/map-reduce.lisp` | **第二十轮**：`%cast-to` 的复数分支加 `typep` 同宽快通道（「结果类型未知」时不再 2×realpart + 2×coerce，1M 7 → 3 ms）；`+cx-inline-ops+` / `+cx-only-ops+` 登记极值族 |
| `src/elementwise2.lisp` | **第二十轮**：`vt-maximum`/`vt-minimum`/`vt-fmax`/`vt-fmin` 的复数分支由 `vt-map` 改为 `vt-fast-map`（内联） |
| `src/complex.lisp` | **第二十轮**：新增 `%def-cx-argsort-kernel` / `%cx-argsort-fiber`（下标归并内核，`vt-argsort` 2115 → 273 ms），两处 `stable-sort` 调用点改走它；九个标志内核移出 |
| `src/extensions.lisp` | **第二十轮**：新增 `%gather-axis-k`（允许两侧轴长不同的沿轴 gather）；`vt-topk` 改为「只算一次序 + gather」（490 → 303 ms）；`%gather-along-axis` / `%gather-axis-k` 补 `(vt-offset …)` 基址（修 `k=1` 的静默错值） |
| `CONVENTIONS.md` | **第二十轮**：新增 §16.8.1（极值/排序内核的四条执行层硬约束 + 「previously compiled」探测器）；§16.37 关闭 U5/U6/U8 |
| `test/cx-extreme-argsort-test.lisp` | **第二十轮新增** 74 断言（极值语义/布局/`clip`/`argsort`/`topk` + 四条倍率护栏），注册后共 57 套件 |
| `test/run-tests.sh` | **第二十轮**：注册 `cx-extreme-argsort-test` |
| `src/complex.lisp` | **第二十一轮**：新增 `%def-cx-extremum-kernel`（按精度生成的极值归约内核，累加器带类型声明 + 首元素播种），`%cx-reduce-max/-min` 改为元素类型分派（`vt-amax` 41 → 3 ms，§16.8.2 第 1 条）；归并排序改双缓冲乒乓（§16.8.1 第 2 条） |
| `src/map-reduce.lisp` | **第二十一轮**：新增 `%select-loop` / `%select-fast`（int8 掩码 + 同类型数据的三元选择，10 格分派） |
| `src/indexing.lisp` | **第二十一轮**：`vt-where` 接入 `%select-fast` 快路径，分配由 `vt-zeros` 改为 `%alloc-vt`（32 → 5 ms） |
| `CONVENTIONS.md` | **第二十一轮**：新增 §16.8.2（归约累加器 / 混合类型逐元素算子的两处类型放宽陷阱）；§16.8.1 第 2 条补「归并必须双缓冲乒乓」；§16.37 更新 U7、新增 U9（基数排序） |
| `test/cx-reduce-where-test.lisp` | **第二十一轮新增** 119 断言（`vt-where` 10 dtype × 4 掩码、布局、极值归约、排序乒乓边界、四条护栏），注册后共 58 套件 |
| `test/run-tests.sh` | **第二十一轮**：注册 `cx-reduce-where-test` |


| `test/test-copy-into.lisp` | **第十八轮**：+6 断言（复数×非连续 typed-walk：转置/翻转/3-D 置换/非连续 dest + `c128T vt-copy ≤ 8× f64T` 护栏），35 → 41 |

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

## H.3 §16 后半章的详细叙事（第五～二十一轮）

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

#### 当时的「负结果」有一条是**错的口径**（后续更正）

本节初稿写「`c128T→c128` 5 ms vs numpy 0.44 ms，差距在访存次序」。
后续复核发现两处错误，两者都在 §H.3.12 里逐条更正：

1. **0.44 ms 不是同口径**：`ndarray.astype` 缺省 `order='K'`，对转置输入返回
   **F 连续**结果（等价于一次线性内存流），根本没有转置。同口径必须写
   `order='C'`：numpy 实测 1.05 ms。
2. **差距不在访存次序**：手写实验里「列主序 0 ms」是脚本写错了（内层步长写成 1，
   实际是平移后的顺序拷贝）。按正确写法，读跨行 1 ms、写跨行 3 ms，
   而当时的全 odometer walker 6 ms —— 差的那 5 ms 是**odometer 的每元素开销**，
   与访存次序无关。

结论：`test` 里的性能对照，被比较的两个实现必须**同样的落点布局**（都 C 连续），
否则测的是两件事。

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

---

### H.3.12 第十八轮：把「转置视图慢」追到底 —— odometer 开销与两张缺格的分派表

上一轮（§H.3.11）把复数转换从 31 ms 拉到 2 ms，留了一句「转置视图仍慢一个量级」。
本轮把它追到底，结论是：**那句话的两个前提都不成立**，真正的瓶颈是另外两处。

#### 起点：三个数字，两个是错的

| 当时的说法 | 复核结果 |
|---|---|
| numpy 转置 astype 0.44 ms | **口径不同**：`astype` 缺省 `order='K'`，转置输入返回 F 连续结果 = 一次线性流。同口径 `order='C'` 是 1.05 ms |
| 手写「列主序走」0 ms | **脚本写错**：内层步长写成 1，实际是平移后的顺序拷贝。正确写法是 3 ms |
| 「循环次序还压着 2~4 倍」 | **归因错误**：读跨行 1 ms、写跨行 3 ms，都远快于当时的 6 ms；差的是 odometer 每元素开销 |

这两条已在 §H.3.11 就地更正（留着是为了记住教训：性能对照必须先对齐**落点布局**）。

#### 缺陷一：`%gather-contiguous` 的 odometer 每元素 5 ns

原实现是「全轴 odometer」：每个元素都做一次索引数组自增 + 进位判断 + `svref`。
1M 元素 ≈ 5 ms 的**固定开销**，与元素类型、stride 都无关。

修法：**末轴用 `DOTIMES` 展开，其余轴才走 odometer**。落点是行主序，所以末轴在
目标上连续，内层循环只要源指针按 `s[末轴]` 步进；odometer 的迭代次数从 `size`
降到 `size / d[末轴]`（1000×1000 转置视图：1M → 1000 次）。

| 1000×1000 c128 转置收集 | 耗时 |
|---|---:|
| 全 odometer（修前） | 6 ms |
| 末轴 DOTIMES（修后） | 1 ms |
| 手写 2 层循环（同次序下限参照） | 1 ms |

#### 缺陷二：`%copy-generic` 的分派表缺复数两格

`vt-copy` / `vt-contiguous` / `vt-copy-into` 共用 `%copy-generic`，它**已经有**
按元素类型展开的 typed-walk（8 种实数），但表里没有
`(complex double-float)` / `(complex single-float)` —— 复数一律掉到
`%copy-generic-slow` 的泛型 `aref`（每元素装箱/解包）。

补两格即可，`vt-copy` 的转置复数拷贝 17 ms → 6 ms（同形 f64T 是 3 ms，2 倍数据量
对应 2 倍耗时，已到该访存次序的下限）。

#### 缺陷三（**不修**）：SBCL 没有 16 字节元素的 `replace` basher

拆解 `vt-copy` 连续路径（1M 元素）：

| 段 | c128（16 MB） | f64（8 MB） |
|---|---:|---:|
| `make-array` 裸分配 | 0 ms | 0 ms |
| `make-vt` 的 0 填充 | 3 ms | 1 ms |
| `replace` | 5~10 ms | 1 ms |
| 逐元素 `dotimes` 参照 | 11 ms | 7 ms |

`replace` 对 `single-float`/`double-float`/`(complex single-float)` 都是字节级
memmove（0~1 ms），唯独 `(complex double-float)` 退化成逐元素循环 —— SBCL 的 basher
表只覆盖 1/2/4/8 字节。

可选方案是 `sb-kernel:system-area-ub8-copy`（需 `with-pinned-objects`），实测能到
1.5 ms 级，但那是**内部符号**：跨 SBCL 版本无保证，且本库把「可移植 + 无未定义行为」
排在性能前面（§16.14.1 的分级同理）。**决定不引入**，作为已知固定成本写进
§16.4.1 的末尾，谁要动它需要先裁决这条原则。

#### 实测汇总（1M 元素，同进程 8 次取最优）

| 场景 | 修前 | 修后 | 同形实数对照 |
|---|---:|---:|---:|
| `vt-copy` 连续 c128 | 13 ms | 13 ms | 2 ms（差距全在 `replace`，见上表） |
| `vt-copy` 转置 c128T | 17 ms | 6 ms | 3 ms |
| `vt-contiguous` 转置 c128T | 17 ms | 6 ms | 4 ms |
| `vt-astype` 转置 c128T→c128 | 6 ms | 4 ms | — |
| `vt-astype` 转置 c128T→c64 | 9 ms | 7 ms | — |

转置 cross-dtype 仍要「收集 + 连续 cast」两趟（各一次 16 MB 分配）。把它合成一趟
（strided cast）需要在非连续路径再展开 100 格分派表 —— 收益约 3 ms/1M，
不值得让 `core.lisp` 再长 300 行；这一条留作**明确不做**。

#### 测试

- `test/test-copy-into.lisp`：+6 断言（复数×非连续的 typed-walk 覆盖）——
  2-D 转置（c128/c64）、负步长视图、3-D 置换（与逐点公式的独立参照对比）、
  非连续 dest 写入，外加一条 `c128T vt-copy ≤ 8× f64T` 的倍率护栏（实测 2.3×）；
- `test/astype-cast-test.lisp`：663 断言不变（遍历形态改了，落点必须逐位一致，
  已有的「非连续等价」组正好覆盖）。

---

### H.3.13 第十九轮：性能巡检 —— 找出「毫秒级 vs 秒级」的那几处

前两轮都是「用户指哪儿打哪儿」。本轮换方法：**不看代码，先测**。写一个覆盖
8 个函数族、50 个算子的巡检脚本，对同一算子分别跑 float64 与 complex128
（同元素数），把「复数耗时 / 实数耗时」全部打出来。比值 ≥ 4 的逐个查根因。

这个方法立刻露出四个前两轮完全没看到的问题（都在**实数线**或**非转换路径**上）：

| 算子（1M 元素 / 200k 取样） | 修前 | 修后 | 根因 |
|---|---:|---:|---|
| `median`（全零数组）200k | 10 700 ms | **1 ms** | quickselect 两分区 → O(n²) |
| `median`（整体降序）400k | 142 ms | **21 ms** | 三分区的坏排列，每趟只削极窄一段 |
| `vt-copy`（连续 c128） | 12 ms | **4 ms** | `replace` 缺类型声明 + 零填充 |
| `vt-copy` / `vt-flatten` / `roll` / `tile` / `pad` | 3~12× | **1.5~2×** | 同上 |

#### 缺陷一：顺序统计的 quickselect 有两类退化（最严重，秒级）

`median` / `percentile` / `quantile` / `nanmedian` 共用 `%select-double`。
实测曲线（200k，先构造好数据再计时）：

| 输入 | 修前 | 修后 |
|---|---:|---:|
| 全等（全零） | **10 700 ms** | 1 ms |
| 只有两个值 | 2 980 ms | 2 ms |
| 7 个值循环 | 364 ms（400k） | 2 ms（400k） |
| 均匀随机 | 3 ms | 6 ms |
| 已升序 | 0 ms | 2 ms |
| 整体降序（`k = n/2-1`） | 142 ms（400k） | 21 ms（400k） |

两处根因、两套对策（详见 §16.35.5 的契约）：

1. **Lomuto 两分区 + 重复值** = O(n²)：每趟只剥掉 pivot 一个元素。改三分区
   （`<` / `=` / `>`），等于 pivot 的整段一趟剥掉；
2. **三分区 + 三数取中的坏排列**（该用例是「整体降序且 k 取中线左侧一格」）：
   451 趟、30.2M 次访问。加**访问量上限 4n**，超限改用 `%sort-fibers`
   排完整段再取第 k 个（缓冲区仍含全部元素，所以全量排序后第 k 个就是答案）。

代价：随机输入 3 → 6 ms（多一次 `incf` 计数与三分区的一次额外比较），
换来两类秒级退化消失。

#### 缺陷二：`replace` 缺类型声明 = 逐元素泛型序列操作

拆 `vt-copy` 的连续路径（1M complex128，16 MB）：

| 段 | 修前 | 修后 |
|---|---:|---:|
| `make-vt … 0`（零填充） | 3 ms | 0（改用 `%alloc-vt` 不填充） |
| `replace` 无类型声明 | 8 ms | — |
| `replace` 带 `(the (simple-array …))` | — | **0.2 ms** |

`replace` 只有在**两端都被编译器确认为同类 specialized `SIMPLE-ARRAY`** 时才
展开成字节级 memmove；少一个声明就退回 `sb-sequence` 的逐元素路径（每元素一次
SAETP 分派 + 类型检查）。这解释了前两轮一直以为的「SBCL 没有 16 字节 basher」
—— 其实有，只是要类型信息。新增 `%memcpy`（core 内唯一一份按元素类型展开 +
带声明的块拷贝），`vt-copy` / `vt-contiguous` / `vt-copy-into` 快路径与
`join.lisp` 的 `%block-copy` 全部接到它上面（顺带给 `%block-copy` 补上复数两格，
它原来只有 8 种实数）。

顺带把 `make-vt … 0` 换成新的 `%alloc-vt`（不填充）：`vt-copy` / `vt-contiguous`
/ `%gather-contiguous` 都是「先分配、再写满每个元素」，零值填充是纯浪费 ——
它不但多一趟写，还把目标页弄脏，让随后的真实写入付 RFO。1M complex128 = 3 ms。

#### 一条**负结果**：复数极值族（maximum/minimum/fmax/fmin/clip）

巡检把 `vt-clip` 的复数版打成 **1637 ms / 1M**（实数 3 ms）。根因链：

- `vt-clip` 复数分支 = `vt-maximum` + `vt-minimum`，两者都走
  `(vt-map (function %cop-max) …)` —— `vt-map` 的通用路径每元素
  `funcall` + `vt-cast` 写回，而 `vt-cast` 内部**每次调用都 `with-float-safe`**
  （屏蔽/恢复浮点陷阱约 200 ns ×2）→ 800 ns/元素。

按 §16.14.2 的正解应该是走 `vt-fast-map` 内联（`+`/`*` 就是这么降到 5 ms 的）。
试了两版，**都更慢**：

| 方案 | 1M complex128 |
|---|---:|
| `vt-map %cop-max`（原状） | 834 ms |
| `vt-fast-map %cop-max`（内核仍在 complex.lisp） | 1592 ms |
| 内核前移到 complex-kernels + `declaim inline` 后再 `vt-fast-map` | 1469 ms |

内联路径反而更慢，说明生成的循环里每元素仍发生一次「返回值类型未知 →
写回复数槽位前的运行期转换」—— `+`/`*` 没有这个问题是因为 SBCL 对 CL 内置
函数有类型推导，而 `%cop-max` 的返回类型取决于分量宽度（c64/c128 两种）。

**已回滚**到 `vt-map`（不留更慢的代码）。下一步的正确形态是给 c64/c128 各写
一个**带 ftype 返回类型**的内核（`%cop-max-f64` / `%cop-max-f32`），让内联循环
拿到确定的类型；或者按 `%cx-dispatch` 的形态写一个专用的二元内核宏。
在做到之前，`clip` 的复数版维持在 ~1.6 s/1M，属**已知未修**。

#### 巡检顺带量到的其他已知项（本轮不动，数字留档）

| 算子（1M 复数） | 耗时 | 相对实数 | 性质 |
|---|---:|---:|---|
| `argsort` / `topk` | 1062 / 1083 ms | 21× / 13× | `stable-sort` + lambda 比较器，每比较一次 `funcall` + 两次 `aref`（~50 ns × 20M 次） |
| `einsum`（GEMM） | 845 ms | 13.6× | 复数 GEMM 无 SIMD（§16.36.2 已裁决） |
| `matmul` / `inner` | 278 ms | 4.3× | 同上 |
| `nanvar` / `nanmean` / `nansum` | 117 / 101 / 86 ms | ~100× | 掩码 + `vt-cast` + 复算，组合算子偏多 |
| `abs` | 17 ms | 8.5× | hypot，numpy 亦慢 |

这些都已写进 §16.37 的未完成项，按「绝对耗时 × 修复代价」排序。

#### 测试

新增 `test/order-stat-test.lisp`（266 断言）：

| 组 | 内容 |
|---|---|
| 正确性 | `median` / `percentile(0,25,50,75,100)` / `nanmedian` 与「先 `vt-sort` 再按 numpy 位置公式取」的**独立参照**逐点一致（6 种分布 × 7 种规模：1/2/3/7/8/100/1001） |
| 护栏 | 全等 / 两值 / 7 值 / 升序 / 降序 相对**随机输入**的倍率 ≤ 8×（倍率而非绝对毫秒，避开机器速度；同时抓 5000× 与 18× 两类回归），另加全等数组的 200 ms 绝对上限 |

> 写这套测试时踩了一个坑：第一版把 `make-data` 放进了计时区间，随机数生成
> （200k ≈ 40 ms）盖过了被测算子，所有倍率都打成 1.0×。性能用例的
> **基准与被测项必须同口径**，这条在 §H.3.11 已经记过一次，这是第二例。

---

### H.3.14 第二十轮：把巡检留下的三项待办全部关掉 —— 三处都是「放错地方」

§16.37 的待办表里，U5（复数极值族）、U6（复数 argsort/topk）、U8（复数比较）
挂着三行数字。本轮把它们全部关掉。共同点：**都不是算法错，而是「用错了变体」
或「放错了文件」**，代价 10~300 倍。

| 场景（1M 元素） | 修前 | 修后 | 同形实数线 |
|---|---:|---:|---:|
| `vt-maximum` c128 | 804 ms | **7 ms** | 2 ms |
| `vt-clip` c128 | 1600 ms | **9 ms** | 3 ms |
| `vt-argsort` c128 | 2115 ms | **273 ms** | 166 ms |
| `vt-topk k=10` c128 | 490 ms | **303 ms** | 168 ms |
| `vt-=` c128 | 60 ms | **4 ms** | 1 ms |
| `vt-<` c128 | 45 ms | **5 ms** | 5 ms |
| `vt-isnan` c128 | 25 ms | **1 ms** | 1 ms |

#### 一、极值族：`%cisnan` 每元素两次「屏蔽/恢复浮点陷阱」

上一轮把这条归因成「返回值类型未知导致写回前多一次运行期转换」，**错了**。
先用三段式拆开（同一内核只改写回点）：

| 写法 | 1M c128 |
|---|---:|
| 裸内核 + 裸写回 | 843 ms |
| 裸内核 + `%cast-to` 复数构造 | 1661 ms |
| 裸内核 + `(the (complex double-float) …)` | 852 ms |

裸写回就要 843 ms —— 说明瓶颈在内核本身。把 `%cisnan` 换成免屏蔽的
`%cx-nan-p`（其余不动）：**839 ms → 3 ms**。根因是 `%cisnan` 每次都套一层
`with-float-traps-masked`（约 200 ns），极值内核每元素判两次 = 400 ns/元素。
dtype.lisp 里那组快判原语的注释早就写了「内核热路径请用 %CX-NAN-P」，
这条内核没遵守 —— 而 `%cx-arg-reduce-run`、`%cx-sort-less` 都遵守了。

修完内核，`%cast-to` 的问题才浮出来（上一轮看到的就是这一层）：
复数的 `coerce` 分支在「实参类型未知」时要 2 次 `realpart` + 2 次 `coerce`，
约 800 ns/元素。改成**先 `typep` 判同宽、命中直接返回**：

```lisp
(let ((z <form>))
  (if (typep z '(complex double-float))
      z
      (complex (coerce (realpart z) 'double-float)
               (coerce (imagpart z) 'double-float))))
```

这一改是**通用**的（任何「复数目标、结果类型未知」的内联算子都受益），
且**不撒谎**（`typep` 保证安全，不是 `the` 断言），1M 上 7 ms → 3 ms，
已与手写循环同速。顺带解释清楚上一轮 `vt-fast-map` 反而更慢的原因：
内核慢 800 ms + 写回慢 800 ms，两笔叠加。

#### 二、`argsort`：`stable-sort` + lambda 比较器

旧实现给 fixnum 下标向量做 `(stable-sort idx (lambda (a b) (%cx-sort-less …)))`，
每次比较 = 一次 `funcall` + 两次跨步 `aref`（约 50 ns），1M 元素约 20M 次比较。
新增 `%def-cx-argsort-kernel`（照 `%def-cx-fiber-sorter` 的形态）：插入排序
（n ≤ 16）+ 自底向上稳定归并，比较器**内联**进归并循环，零 cons、零 funcall。
同一份数据上 A/B：c128 **2115 → 273 ms**、c64 1598 → 229 ms，两边给出的排列
逐位相同（`equalp` 校验），重复值数组的稳定性单独验过。

#### 三、`topk`：既排值又排下标（两趟 O(n log n)）

`vt-topk` 只要 k 个元素，却同时调 `vt-sort` 与 `vt-argsort`。改成「只算一次序 +
按下标取回值」，新增 `%gather-axis-k`（沿轴 gather，允许两侧轴长不同 ——
既有的 `%gather-along-axis` 要求形状完全相同，用不了）。490 → 303 ms。

**顺带挖出一个静默错值**：`%gather-axis-k` 最初假设两边视图的 `offset` 为 0。
但 `vt-contiguous` 对「已经连续」的输入**原样返回**，而 `narrow` + `flip` 出来的
单元素视图 offset=7 却仍满足连续判定 —— 于是 `k=1` 的 `topk` 返回了全数组的
**最小值**（下标对、值错）。既有 `%gather-along-axis` 有同一处隐患，一并补上
`(vt-offset …)` 基址。这类「视图 offset」错误不报错、只在边界尺寸上现形，
测试里专门加了 `k = 0 / 1 / n` 三档。

#### 四、比较与谓词：内核放晚了一个文件

`vt-=` 60 ms、`vt-<` 45 ms、`vt-isnan` 25 ms（实数线 1~5 ms）—— 而 §16.14.3
记录的曾经是 3 ms。它们在 `elementwise2.lisp` 里通过 `vt-fast-map` 内联调用，
但 `%cx-eq-flag` 等九个标志内核定义在 **`complex.lisp`**，而 `complex.lisp` 是
**最后**加载的（它要捕获各函数的实数原实现）。于是编译 `elementwise2` 时这些
符号还不存在，12 个调用点全部编成真实函数调用；事后 `declaim inline` 也无效。

编译日志里一直有 9 条告警在说这件事：

```
Proclaiming CLVT::%CX-EQ-FLAG to be INLINE, but 12 calls to it were previously compiled.
```

把这九条当成噪音忽略了很久。它们其实是**缺陷信号**：出现一行就说明某个内核
放晚了 N 个调用点。把九个内核移到 `src/complex-kernels.lisp`（`§16.33.2`
「标量内核前移」的同一招，上一轮对极值族用过）→ `STYLE-WARNING` 97 → 88，
`vt-=` 60 → 4 ms、`vt-<` 45 → 5 ms、`vt-isnan` 25 → 1 ms。
剩下 3 条 `previously compiled` 与复数无关（`vt-p` / `%copy-default-rng` /
`parse-subscript-tokens`），留给 U4 一起清。

#### 五、测试

新增 `test/cx-extreme-argsort-test.lisp`（74 断言）：

| 组 | 内容 |
|---|---|
| 极值语义 | `maximum`/`minimum`/`fmax`/`fmin` × 2 dtype × 16×16 值对，与「按字典序逐对折叠」的独立参照（只用 `realpart`/`imagpart`/`float-nan-p`）逐点一致 |
| 布局 | 连续 / 广播标量 / 转置视图 / 步进视图 四种落点同语义（快路径与 `vt-map` 兜底的选择随连续性而变，语义不能跟着变） |
| `clip` | `= min(max(x, lo), hi)`；只给上限；全省略 = 副本；返回新缓冲 |
| `argsort` | 双射、施加上标 == `vt-sort`、非降序、**稳定性**（等值下标递增）、`axis=0/1/nil`、转置视图与物化副本同结果、NaN 排末尾 |
| `topk` | `k ∈ {0,1,3,n}` × `largest` × `sorted` 与「排序切片」一致、下标回指、二维沿轴 |
| 护栏 | 极值 c128/f64 ≤ 8×、argsort c128/f64 ≤ 3×、`topk ≤ 1.4×argsort`、`argsort ≤ 2.5×sort` |

> 护栏的规模从 200k 抬到 1M：200k 时实数线只要 0.2~0.6 ms，落到
> `get-internal-real-time` 的有效分辨率（约 1 ms）里，倍率会算成 1000×+。
> **性能护栏的规模必须让「基准项」也远超计时噪声**，这是第三次记同类教训
> （前两次：§H.3.11 的落点口径、§H.3.13 的构造计入计时）。

---

### H.3.15 第二十一轮：巡检第二遍 —— 三处「类型被推断放宽」

上一轮把 U5/U6/U8 关掉后，重跑一遍巡检（8 个族、约 30 个算子，按 c128/f64 比值
排序）。这次的异常项集中在**归约**与**掩码选择**上，根因全是同一件事：
**变量类型在某一处被推断放宽**，热路径静默退回泛型。

| 场景（1M 元素） | 修前 | 修后 | 同形实数线 |
|---|---:|---:|---:|
| `vt-amax` c128 | 41 ms | **3 ms** | <1 ms |
| `vt-amin` c128 | 41 ms | **2 ms** | <1 ms |
| `vt-where(mask,a,b)` c128 | 32 ms | **5 ms** | 2 ms |
| `vt-nansum` | 35 ms | **5 ms** | 1 ms |
| `vt-nanmean` | 46 ms | **13 ms** | <1 ms |
| `vt-nanvar` | 65 ms | **28 ms** | 1 ms |
| `vt-nanstd` | 62 ms | **30 ms** | 1 ms |
| `vt-nanmax` | 88 ms | **22 ms** | 1 ms |
| `vt-nancumsum` | 36 ms | **5 ms** | 13 ms |

#### 一、归约累加器被放宽成 `T`

先按上一轮的套路拆（`%cx-reduce-max` 40 ms、`%cx-reduce-values :max` 43 ms、
`vt-amax` 端到端 53 ms），确定瓶颈在内核本身；同一条宏生成的 `%cx-reduce-sum`
却是 <1 ms。同一个 `%cx-dispatch`、同一条循环骨架，差别只有步进表达式。

抽出最小复现，两个函数只差一行 `(declare (type (complex double-float) acc))`：

| 写法 | 1M c128 |
|---|---:|
| `(let ((acc (aref d 0))) … (setf acc (%cop-max acc v)) … (vector acc))` | 40 ms |
| 同上 + `(declare (type (complex double-float) acc))` | **3 ms** |

原因：`(vector acc)` 接受任意类型，而 `%cop-max` 的**派生返回类型是 `T`**
（它的形参没有声明 → `(cond ((%cx-nan-p a) a) …)` 返回 `a` 或 `b` → `T`）。
SBCL 于是把 `acc` 的变量类型放宽到 `T`，每次调用都要装箱/解包复数。

> 为什么**逐元素**路径没这个问题：`%inline2-loop` 里的实参是已知类型的
> `aref`，内联展开后按**上下文**重新推导类型，不受函数派生类型的拖累。
> 于是出现了「同一内核、逐元素 5 ms、归约 41 ms」的怪象 —— 这条差异上一轮
> 完全没看出来。

修法：新增 `%def-cx-extremum-kernel`，按精度各生成一份，累加器带
`(type (complex double-float) acc)` / `(type (complex single-float) acc)`，
`%cx-reduce-max` / `-min` 只做元素类型分派。顺带把 `(if (null acc) v …)` 换成
「首元素播种」，空归约区仍返回 `(vector nil)`，与调用方契约一致。

（中间踩了一个坑：把 `%def-cx-reducer` 的 `flet` 改成三个互相调用的局部函数时
必须用 `labels` —— `flet` 是并列绑定，兄弟函数互相不可见，直接报
「The function STEP-AT is undefined」。）

#### 二、`vt-where` 永远进不去内联快路径

`vt-where` 是三元选择，走 `(vt-map (lambda (c a b) (if (/= c 0) a b)) …)` ——
闭包 + 每元素 `vt-cast`。数值上一看就知道：`vt-isnan` 2 ms、`vt-sum` <1 ms，
而 `vt-where(mask, 0, z)` 32 ms。

根因不是「没特化」，而是**永远进不去**：`%vt-inline3-fast` 的前置条件是
「三个输入与结果的元素类型全同」，而掩码恒为 `:int8`、数据是 `:complex128` ——
dtype 永远不相等。这类「混合类型逐元素算子」必须单独列一张按元素类型展开的
表（和 `%mask*-loop` 同一个理由）。

新增 `%select-loop`（掩码 int8 + 两个同类型数据 + 长度 1 广播）与
`%select-fast`（10 格分派），`vt-where` 在「结果连续 + 每输入要么长度 1 要么
连续同形状」时走它，否则退回原路。1M c128 **32 → 5 ms**。

nan 族全部建在 `vt-where` 上，收益直接传导：`nansum` 35→5、`nanmean` 46→13、
`nanvar` 65→28、`nanstd` 62→30、`nanmax` 88→22、`nancumsum` 36→5。
顺带把 `vt-where` 的分配从 `vt-zeros` 换成 `%alloc-vt`（两条路都会写满每个
元素，零填充是纯浪费）。

#### 三、排序的双缓冲乒乓（只有 10%，但先把话说准）

`vt-sort` 的归并每合并一段就 `replace` 回原缓冲区 —— n=1M 时第一趟有 50 万次
两元素拷贝。改成「每趟把 src 归并进 dst 后交换」，理论上该省掉这些调用。

实测只从 200 → 181 ms（c128）、132 → 130 ms（f64）。**说明 `replace` 的调用
开销不是主因**（SBCL 对小 `replace` 的展开比预期便宜）。真正的成本结构是：
c64（8 字节）176 ms 与 c128（16 字节）181 ms 几乎一样 → **不是内存带宽受限**，
而是每次比较的**分支不可预测**（随机数据上归并的 if 是抛硬币，约 15 周期/次，
20M 次 ≈ 100 ms）。

要再往下只能换算法（LSD 基数排序，numpy 对 f64 用的就是它：5.9 ms vs 我们的
130 ms）。复数要按「先虚部后实部两次稳定基数」实现，且 NaN 必须映到最大键值；
这是一次算法替换而不是缺陷修复，收益大但风险也大 —— 登记进 §16.37，本轮不做。
乒乓本身仍然保留：它是标准形态，且省掉了每段的 `replace` 调用。

#### 四、测试

新增 `test/cx-reduce-where-test.lisp`（119 断言）：

| 组 | 内容 |
|---|---|
| `vt-where` | 8 个实数 dtype + 2 个复数 dtype × 4 种掩码（全 0/全 1/交替/负值），与逐元素参考逐点一致 |
| 布局 | 标量广播、转置视图与物化副本同结果、`:out` 复用写满、单参数形式 = `nonzero`、参数不配对/`dtype` 冲突报错 |
| 极值归约 | NaN 传播（max 与 min 都要）、无 NaN 时的字典序取值、单轴 `axis=0/1`、c64 保 dtype、空归约报错 |
| 排序乒乓 | n ∈ {1,2,3,16,17,18,33,64,65,1000}（跨过插入排序阈值与奇偶趟数）：非降序、`argsort` 是排列、按下标取值 == `vt-sort` |
| 护栏 | `where` c128/f64 ≤ 4×、`amax` c128/f64 ≤ 8×、`amax` 绝对 < 40 ms |

> 写这套测试时又踩了「字典序」：先按直觉把 `amin{(1+2i),(0+9i),(2+0i)}` 写成
> `1+2i`，实际最小是 `0+9i`（实部优先）。§16.8 那句「字典序不是任意一致的全序」
> 值得每个写测试的人读一遍。

---

### H.3.16 第二十二轮：复数 `log` 族为什么慢 43 倍 —— 以及一次「差点写错」的教训

#### 一、起点不是「猜」，是巡检比值

沿用第十九/二十轮的方法：对同一算子跑 float64 与 complex128（同元素数 1M），
把比值打出来。这一遍扩到 45 个算子，26 个比值 ≥ 4，最高两个是
`log` **42.4×**、`sign` **40.3×**。这两个数都不在 §16.14.3 的已知表里
（那张表只覆盖 `add/mul/div/sqrt/exp/abs/sum/cumsum/sort/matmul`），
说明它们从未被量过。

第一个动作是**先量 CL 基线**，而不是先改代码：

| 写法（1M，手写循环） | 耗时 |
|---|---:|
| `(log z)` CL 复数 | 329 ms |
| `(complex (log (sqrt (+ x² y²))) (atan y x))` | 88 ms |
| `(complex (log (hypot x y)) (atan y x))` | 54 ms |
| 库内 `vt-log`（走 CL） | 352 ms |

结论明确：**352 ms 里 329 ms 是 CL 自己的复数 log**，不是我们的调度或装箱。
于是问题变成「分量式能不能既快又和 numpy 一致」。

#### 二、一张 441 点网格，把「快」和「对」同时钉住

只比速度很容易写出更快的错代码。这里先建网格：21 个实部 × 21 个虚部，
取值刻意包含 `±0.0`（符号零）、`±1e-8`、`±0.5`、`±1`、`±1.5`、`±2`、`±4`、
`±1e300`、`±Inf`、`NaN`，从 numpy 2.4.6 导出期望值
（`test/gen_complex_elementary_expected.py`），然后在 Lisp 侧逐点比。

**网格立刻给出一个意外**：现有实现（CL 回退）在 `log` 上就有 6 处与
numpy 不一致，`tan`/`tanh` 各 92 处，`asin`/`acos`/`asinh`/`acosh` 各 67~72 处，
合计 **481/4410**。也就是说本轮之前，这些函数在分支切割与 NaN 输入上
就已经和 numpy 有系统偏差 —— 只是没有测试盯着。其中：

```
log(-0+0i)：CL 给 -Inf+0i，numpy 给 -Inf+πi
asin(-4-0i)：CL 给 -1.5708+2.0634i，numpy 给 -1.5708-2.0634i（切割上的虚部符号反了）
```

第二条是硬缺陷：`vt-asin` 遇到「虚部是 NaN」的输入直接
`The value NaN is not of type (DOUBLE-FLOAT 0.0)` —— 一个既能被用户触发、
又不是 Lisp 语义所要求的崩溃（`with-float-safe` 只屏蔽浮点陷阱，屏蔽不了
SBCL 的 `DECODE-FLOAT` 类型断言）。

#### 三、`%cop-log` 的三个必须

改完分量式后网格从 6 处降到 0，靠的不是公式本身，而是三条细节：

1. **模走 `%cop-abs`**。第一版写 `(sqrt (+ (* x x) (* y y)))`，
   `z = 1e300+1e300i` 直接 `FLOATING-POINT-OVERFLOW` —— 和 `%cop-sqrt` 里
   踩过的是同一个坑（`x²` 上溢）。`%cop-abs` 已带 `< 1d150` 的量级守卫。
2. **Inf 判定必须早于 NaN**。`log(Inf+NaNj)` 在 numpy 是 `Inf+NaNj`
   （模为 Inf 时相位无意义，但模本身仍是 Inf）；先判 `%cx-nan-p` 就退化成
   `NaN+NaNj`。网格里正好这两格（#398/#419）报错，改过来即 0。
3. **相位用 `(atan y x)`**。`(phase z)` 看起来等价，但它在 `x = -0.0` 上
   丢掉符号零带来的 π —— `log(-0+0i)` 就错成 `-Inf+0i`。

#### 四、`declaim inline` 的 144 ms，以及探测器的盲区

把内核写对之后 `vt-log` 仍有 199 ms。用 `vt-map` / `vt-fast-map` / 手写循环
三路对比：

| 调用方式 | 耗时 |
|---|---:|
| `vt-map #'%cop-log`（闭包） | 210 ms |
| `vt-fast-map #'%cop-log`（无内联声明） | 202 ms |
| `vt-fast-map #'%cop-log`（加 `declaim inline`） | 55 ms |
| 手写循环 + 分量公式 | 23 ms |

`vt-fast-map` 与 `vt-map` 差不多，说明**快路径根本没进去**：不是白名单问题
（`%cop-log` 一直在 `+cx-inline-ops+` 里），而是缺 `declaim inline` ——
和 §16.14.2 第 2 条同一件事，只是这次漏在了 log 上。

顺手把同族四个内核（`%cop-expm1` / `%cop-log1p` / `%cop-sinc` / `%cop-sign`）
搬到 `complex-kernels.lisp`。它们原本住在 `complex-ops.lisp`，而调用点在
更早加载的 `extensions3` / `elementwise2` 里 —— **编译调用点时符号还不存在**，
`(function %cop-sign)` 只能编成真实函数调用，内核本体被独立编译成
「泛型 NUMBER」版本。实测 `vt-sign` 104 ms，`vt-signum` 17 ms（同一个公式、
只是内联与否），差距一目了然。

> **一个探测器抓不到的盲区**：§16.8.1 第 4 条让我们盯
> `Proclaiming X to be INLINE, but N calls …`，但那条告警只在「符号已存在、
> inline 声明迟到」时出现。这里的四个内核**压根还没定义**，编译期只当普通
> 函数调用，一条告警都不会有。补了一个静态扫描（文件加载顺序 ×
> `(function KERNEL)` 引用点 × `defun` 位置），全库只剩 4 处命中，就是这四个。

#### 五、回滚：`asin`/`tanh` 的两版更差的尝试

既然 `asin` 是 915 ms（最慢的一个），顺手试了分量式：

| 方案 | 441 点不一致数 |
|---|---:|
| 现状（CL 回退） | 70 |
| `asin = -i·log(iz + sqrt(1-z²))` | 386 |
| `asin = -i·log(iz + sqrt(1+z)·sqrt(1-z))` | 386 |
| `tanh = (tanh x + i·tan y)/(1 + i·tanh x·tan y)` | 232 |

都比现状差 —— 大 `|z|` 处（1e300）主值支选取与 numpy 不同，`|y|` 巨大时
numpy 走 `copysign(0,y)` 的极限分支而不是公式。**两版都已回滚**，数据写进
§16.37 的 U10，避免下一轮从零再试一遍。教训：这六个函数要复刻 numpy 的
Kahan 分支（`npy_casin` / `npy_ctanh` 的源码分支），不是套恒等式。

#### 六、测试

新增 `test/complex-elementary-test.lisp`（68 断言）：

| 组 | 内容 |
|---|---|
| A 网格 | 441 点 × 5 算子（log/log2/log10/sign/sinc）与 numpy 2.4.6 逐点对拍；sign/sinc 在 `\|分量\| ≥ 1e300` 处跳过并说明原因 |
| B 语义 | `log(±0±0i)` 的极点与分支切割（按**位模式**比较，不吃容差）、负实轴主值、模为 Inf、NaN 组合传播、`sign`/`sinc` 极点 |
| C 契约 | complex64 保持、2-D 形状、`:out` 复用、`log(:base 2) = log2`、空数组、整数输入走实数通路 |
| D 前移内核 | `expm1`/`log1p`/`sinc`/`sign` 与标量参考一致 + complex64 结果 dtype 不被 `0.0d0` 常量拉回 complex128 |
| E 护栏 | log c128/f64 ≤ 12×、sign ≤ 15×、log 绝对 < 150 ms、sign 绝对 < 60 ms |

> **测试自身的一个既有缺陷也在这里暴露**：`cx-reduce-where-test` 的 `ms-of`
> 用 `get-internal-real-time`，而 SBCL 的该时钟只有 1ms tick —— `vt-amax`
> float64 1M 约 0.8ms，被量成 0，倍率护栏的分母落到 0.05 下限，于是
> 「c128/f64 ≤ 8x」偶发报 60x。两次全绿、第三次红的间歇性失败**不是被测代码
> 的问题**，是量尺的问题。两套件统一改用 `sb-ext:get-time-of-day`（µs）。

> 写网格测试时踩了两个「自己写的坑」：`comp=` 里 `((floatp want) … nil)`
> 少写了一个 `and`，把「want 是浮点」直接当成「不相等」，于是**每个点都失败**
> 而打印出来的 got/want 一模一样；另一次是把 `1e300` 写成单精度字面量
> （本库锁定 `*read-default-float-format*` 为 single-float）导致 reader 直接
> 上溢。两处都由「测试全红但差异不可见」暴露，值得记一笔。

---

### H.3.17 第二十三轮：三角 / 反三角六族 —— 为什么"套恒等式"是错的

#### 一、上一轮已经把坑踩过了

第二十二轮收尾时试过两版 `asin` 与一版 `tanh` 的分量式，分别有 386/386/232 处
不一致，回滚并登记为 U10。这一轮把那张表当成路标：**不要从恒等式出发，
从 numpy 的源码分支出发**。

拉 `numpy/_core/src/npymath/npy_math_complex.c.src`，六个函数的关系是：

```
ctan   = -i·ctanh(i·z)            完全委托
casin  = i·conj(casinh(i·conj z)) 完全委托
cacosh = ±i·cacos                 取实部 ≥ 0
casinh ┐
cacos  ┴─ 共用 _do_hard_work + _clog_for_large_values
ctanh  ──  Kahan(1987) + |x| ≥ 22 的渐近分支
```

也就是说只要把 `casinh`/`cacos`/`ctanh` 三个写对，另外三个是五行变换。
`_do_hard_work` 是 Hull-Fairgrieve-Tang 的区域划分（`A_crossover = 10`、
`B_crossover = 0.6417`、`FOUR_SQRT_MIN`），加上 `_f(a,b,h) = (hypot-b)/2`
的无抵消形式 —— 这些常数和分支一个都不能省。

#### 二、四次"看起来对但错"的经历

写的过程里每个坑都表现为「网格上几十处错值」，而不是编译错误：

1. **`(1- y)` 是「y 减 1」**。`_do_hard_work` 里同时有 `1+y`、`1-y`、`y+1`、
   `y-1` 四种组合，我第一遍把 `1-y` 写成 `(1- y)`，`Am1` 少了一个 hypot 项，
   `asinh(4+4i)` 的实部从 2.426 变 1.805，asinh/acos 各 77 处错。
2. **`(- y y)` 会被代数化简成 0**。numpy 的 `return (y-y, y-y)` 用来造 NaN，
   但在 `(speed 3)(safety 0)` 的内联循环里 SBCL 直接把 `x-x` 折成 0 ——
   `tanh(-4+Infi)` 于是返回 `0+0i`。改用 `(%cx-nan-of z)`。
3. **「非双有限」不能代替「含 NaN」**。`cacos` 的非有限段是
   `if (isnan(x) || isnan(y)) { … }`，我写成 `(not (and (finite x) (finite y)))`，
   于是「x 有限 + y=±Inf」也进了那段，返回 `(2x, ∓Inf)`：`acos(-4+Infi)` 给出
   `-8.0-Inf i`。75 处错值。
4. **`%cx-hypot` 对 `(Inf, Inf)` 得 NaN**。它是给 `%cop-sqrt`/`%cop-abs` 用的，
   那两处在调它之前已经判过 Inf；`_clog_for_large_values` 直接用就
   `inf/inf = NaN`，`log|z|` 变 NaN，`asin(±Inf±Infi)` 全错（4 个函数共 12 处）。

#### 三、符号零：这一族真正的难点

`y = ±0` 的符号位决定分支切割两侧的虚部符号，而 CL 的实现里 `phase`/`decode-float`
一类的路径会把它抹平。numpy 的做法是**每个出口都 `copysign`**：
`copysign(rx, x)`、`copysign(ry, y)`。我第一版把 `copysign(wy, y)` 写成
`(if (minusp (float-sign y)) (- wy) wy)` —— 这是 `-copysign`，符号恰好反了，
`asin(-4-0i)` 于是给出 `-1.5708+2.0634i`（正确是 `-1.5708-2.0634i`）。
改成 `(if (minusp (float-sign src)) (- (abs mag)) (abs mag))` 之后归零。

#### 四、验收

441 点网格（21 实部 × 21 虚部，取值刻意含 `±0.0`、`±1e-8`、`±1`、`±2`、`±4`、
`±1e300`、`±Inf`、`NaN`）上：

| 算子 | 修前 | 修后 |
|---|---:|---:|
| `tan` / `tanh` | 92 / 92 | **0 / 0** |
| `asin` / `acos` | 70 / 72 | **0 / 0** |
| `asinh` / `acosh` | 70 / 67 | **0 / 0** |
| `atanh` | 18 | 18（仍走 CL，登记 U11） |

性能同时改善 2.4~3.8 倍（`tan` 203→84、`tanh` 192→51、`asin` 915→340、
`acos` 876→300、`asinh` 949→317、`acosh` 866→315 ms）。

#### 五、留下的事

- **`atanh` 18 处**：`atanh(-1±0i)` 应是 `-Inf+0i`，CL 给 `-177.099+0.785i`
  —— CL 把极点算成了有限值。glibc 的 `catanh` 也没有 Kahan 分支可抄，
  要按 C99 的角例表自己写。
- **反三角族仍慢**：300~340 ms vs float64 的 20 ms。瓶颈是
  `_do_hard_work` 的 5 个返回值（多值返回在内联循环里会落栈），以及
  `_clog_for_large_values` 未内联。可行的下一步是拆成两个单值函数。
  两件都写进 §16.37 的 U11。

> 又一次印证了第二十轮那句话：**这类问题没有"聪明的写法"，只有"照对的写"**。
> 恒等式看起来更短，但在 `|z| = 1e300` 与 `y = ±0` 这两个极端上，主值支和
> 符号零的选取全在细节里 —— 而细节只存在于参考实现的分支里。

---

### H.3.18 第二十四轮：`atanh` 的三次"公式对了但值错了"，与一个 5 倍的开关

#### 一、`atanh` 不是 libm，是 numpy 自己写的

上一轮以为 `atanh` 要复刻 glibc 的 C99 角例表，拉源码才发现 numpy 有自己完整的
`npy_catanh`（分段 + `_real_part_reciprocal` + `_sum_squares`）。四个分支：

```
y = 0 且 |x| ≤ 1        → (atanh(x), y)            实数退化，含 ±1 → ±Inf
x = 0                   → (x, atan(y))             纯虚数，保住 atan 的精度
|x| 或 |y| > 1/eps      → (Re(1/z), copysign(π/2, y))
|x|,|y| 都极小          → z
其余                    → log1p(4|x| / ((|x|-1)²+|y|²)) / 4  +  三种 atan2
```

我按直觉写了三个版本，每个都"公式看着对"：

**第一版**：`real = 0.25·log1p(4x/((1-x)²+y²))`（带符号的 x）。看起来是恒等式
`log(num/den) = log1p((num-den)/den)`、`num-den = 4x` 的直接推论。但
`x = -1+1e-8i` 时 `4x/den = -1`，`log1p(-1) = -Inf`，而真值是 `-9.5569`。
numpy 用 `|x|` 是有原因的：`4|x|/den` 恒 ≥ 0，永远不会踩到 `log1p` 的极点。

**第二版**：改成 `0.5·(log|1+z| - log|1-z|)`。`x = -1` 附近对了，但
`x = 1e300` 时两个 `1e300` 的 log 相减得 0，而真值是 `1e-300`（`atanh ≈ 1/x`）。
numpy 在这一段根本不走这个公式 —— `|x| > 1/eps` 时实部直接退化成 `Re(1/z)`。

**第三版**：照抄分段，一次通过 `0/441`。

第三次的差别只在于「哪些输入走哪一段」，而这三段的边界（`1/eps`、
`sqrt(3·eps)/2`、`|x|=1 && |y|<eps`）没有一个是我能凭数学直觉推出来的。

#### 二、一个 `declaim inline` 值 3~5 倍

上一轮结束后 `asin/acos/asinh/acosh` 是 300~340 ms，仍在 U11 里记着
"瓶颈是 `_do_hard_work` 的 5 个返回值"。这一轮先做最小实验：
把 `%cx-do-hard-work`、`%cx-clog-large`、`%cx-f` 三个都 `declaim inline`，
不动任何算法：

| 算子 | 内联前 | 内联后 |
|---|---:|---:|
| `vt-acos` | 300 ms | **60 ms** |
| `vt-asinh` | 317 ms | **92 ms** |
| `vt-acosh` | 315 ms | **62 ms** |
| `vt-asin` | 340 ms | **112 ms** |

一次一行改动、5 倍。原因和 §16.14.2 第 2 条完全一样：不内联时
「5 个返回值 + 4 个形参」的组合只能落栈，且内核在独立编译单元里看不到
实参的 `(complex double-float)` 类型，参数要装箱。

**这条比"把多值返回拆成单值函数"便宜得多** —— 后者是我上一轮写在 U11 里的
"下一步"，实际上根本不需要：SBCL 内联展开后，多值返回在寄存器里流转。

#### 三、结果

整族 10 个算子（log/log2/log10/tan/tanh/asin/acos/asinh/acosh/atanh）
在同一张 441 点网格上**偏差全部为 0**；`atanh` 从 75 → 59 ms，
反三角族从 300~340 → 60~112 ms（对 float64 的倍率从 15~20× 降到 3~6×，
落在复数算术的固有区间内）。

> 教训和 H.3.17 一样：**边界条件不是推导出来的，是读出来的**。
> 而性能上，先量"最便宜的可疑项"（这里是一行 `declaim inline`），
> 再去动结构 —— 上一轮我把它写成了"拆函数"，方向对但代价高了一个数量级。

---

### H.3.19 第二十五轮：79 条 STYLE-WARNING 的来源，与一次"先量再改"救回来的重写

#### 一、先给 U9 判死刑（省下一轮的工作量）

§16.37 的 U9 写着「排序族换 LSD 基数排序」，理由也充分：numpy 对整数就是用
基数排序，而归并每次比较的分支不可预测（随机数据上约 15 周期 × 20M 次）。
按惯例先写原型量一量，再动手改库。

原型：IEEE754 double → 单调无符号键（负数全位取反、非负数翻符号位），
按 8 bit × 8 趟做 LSD，值数组与键数组一起乒乓。

| 实现 | 1M double |
|---|---:|
| radix，64 位整数键 | 384 ms |
| radix，两半 u32（8 bit × 4 趟 × 2） | **107 ms / 4 趟** → 8 趟约 200 ms |
| 现有自底向上归并 | **130 ms** |

第一条曲线暴露了 SBCL 的一个硬事实：**64 位整数是 bignum**（fixnum 只有 62 位），
`(ash key shift)` 每元素一次分配，直接把基数排序拖到 384 ms。拆成两半 u32 后
降到 107 ms/4 趟 —— 但仍然打不过归并：每趟都要把「键 + 值」两条流按 256 个桶
散写，缓存不友好；而归并是两条顺序流的顺序读写。

结论写进 U9：**这条路的实测上限就在 130 ms 附近，换算法没有收益**。若要再压
只剩 introsort（numpy 的 float 路径），但那会破坏 `argsort` 的稳定契约
（§16.8 与 `cx-extreme-argsort-test` 都钉着稳定性）。于是掉头做 U4。

#### 二、79 条告警只有五类

清单出来后发现，79 条里 **38 条来自同一个宏**：`%cx-dispatch` 无条件绑定
`ZERO`（该精度的 0+0i），而大部分内核用不到它。调用点于是写
`(declare (ignore zero))` —— 但这个 declare 落在**内层 `macrolet` 的 body 里**，
对 `zero` 来说是「外层变量」，SBCL 既报 `IGNORE declaration for a variable
from outer scope`，又照旧报 `ZERO is defined but never used`。少数内核真的用它，
那些地方再报 `reading an ignored variable`。

**一次改动消掉 38 条**：把 `(ignorable ,zero)` 写进宏的绑定处，调用点的
`(declare (ignore zero))` 全部删掉。`ignorable` 的语义正是「用不用都行」。

剩下 41 条分四类：散落的未使用绑定（26）、**无效的** `declaim inline`（5）、
赋值后不读（6）、结构性 4 条。前两类都是"删"或"补声明"，逐个看语义即可；
4 处是反向错误 —— 声明了 `(declare (ignore x))` 却仍在读 `x`
（`vt-nan-to-num` 的 `copy`、`vt-isin` 的 `assume-unique`），删掉声明即可。

`declaim inline` 那 5 条值得单说：它们**从来没有生效过**。声明写在定义之前、
但调用点更早（跨文件或同文件内提前调用），SBCL 只能当普通调用编。留着声明
唯一的作用就是每次编译吐一条告警。删掉即可 —— 那 1~2 个调用点本来就没内联。
事后按排序族复核了一遍（`vt-sort` f64 139 / c128 201、`vt-argsort` 182、
`vt-median` 27、`vt-copy` 1.7 ms），与 §16.37 U9 记录的 130/181 基线同一量级，
差异在本机噪声内。

#### 三、门禁怎么收紧才算数

原本门禁是「`WARNING` 必须 0，`STYLE-WARNING` 不看」。现在要管后者，
最省事的写法是「`STYLE-WARNING` ≤ 4」—— 但这样「修掉一条、新增一条」
仍然能过。所以判据写成：

```
n_api == 3 && n_vtp == 1 && n_style == n_api + n_vtp
```

即「两类白名单各自条数钉死，且总数恰好等于两者之和」。任何第 5 条、或
第 4 条换了个种类，都会 FAIL 并打印全部 `STYLE-WARNING` 上下文。

留下的 4 条是**改不掉而不改语义**的：3 条是 `vt-where`/`vt-clip`/`vt-clamp`
的 `&OPTIONAL`+`&KEY` 形态（改成 `&key` 要同时改调用方和错误信息），
1 条是 SBCL 对「结构体 layout 尚未可见」的自述提示。

> 这一轮没有一行运行期代码被改，全量 59 套件照旧全绿。
> 但它的价值在后面几轮：**从此任何新告警都会在 CI 里直接红**，
> 而不是像以前那样被 79 条噪音淹没。

---

### H.3.20 第二十六轮：三个「排序取一个」的算子，与一次「结果对但语义错」的收尾

#### 一、起点还是比值，这次是 8 倍

上一轮收尾时 U9 那行留着三个数字：`median` f64 25 / c128 208 ms。25 对 208
不是「复数贵一点」的量级 —— 复数确实要贵 1.4 倍（`sort` f64 140 / c128 199），
但 8.3 倍说明**两条路走的算法不同**。翻代码确认：float64 的 `median` 在 v0.4.10
就换成了 `%SELECT-DOUBLE`（quickselect），复数分支还在 `(vt-sort flat)` 之后
按 `(floor n 2)` 取下标 —— 取一个中位数，排了整份数据。

于是本轮的第一件事是把 quickselect 补到复数上。`%CX-SELECT-F64/F32` 与
`%SELECT-DOUBLE` 同构，三条设计一条都不能省（三分区 / 访问量上限 4N /
比较器不判 NaN），它们的理由在 §16.14.6 里逐条写了实测数据。

#### 二、顺手看到 topk 更离谱

既然 `median` 的问题是「排序取一个」，就顺手量了 `topk`：1M double 取 k=10
要 **174 ms** —— 比整份 `sort`（140 ms）还贵，因为走的是 `argsort`。而「取
前 10 个」只要 O(N log k)。原型（纯内核、不含封装）是 **1.1 ms**，158 倍；
接进 `vt-topk` 后含输出构造是 **2.2 ms**，仍快 79 倍。

但 topk 的难点不是性能，是**不能改语义**。它的输出不仅是「哪 k 个」，
还有顺序：`largest` 时按值降序，**并列时按下标降序**；`smallest` 时按下标
升序。这是 `vt-argsort` 稳定排序 + `flip` 的直接后果，用户可能已经依赖它。
所以堆的比较器不重新定义全序，而是把 `vt-sort` 那个全序原样复刻成
「名次 -1/0/1」：非 NaN 按值、NaN 名次最高（`%sort-fibers-*` 就是这么做的）、
并列比下标。

回归测试也因此不能与 numpy 比 —— numpy 的 topk 不保证并列序。改成
「同一份输入分别走堆与 `argsort` 参照，比较**值和下标序列**」，6 种 dtype ×
并列/NaN × 3 种维度 × `sorted=nil`，一共 2000 多条断言。

#### 三、两个「结果对但语义错」

写测试时撞到两处不是本轮引入、但一直没人钉住的东西：

1. **全 NaN 的复数 `nanmedian` / `nanmax` / `nanmin` 返回 `NaN+0j`**。
   代码里是 `(complex (%cx-nan-of raw) 0)` —— 虚部那个 `0` 是「先取 NaN，
   再补 0」的副产品，但 numpy 给的是 `nan+nanj`。改成 `nan+nanj`。
2. **numpy 自己在「沿轴全 NaN」上不一致**：1-D 或展平给 `nan+nanj`，2-D 沿轴
   给 `nan+0j`。既然参考实现自相矛盾，就只能自己定一个：本库统一 `nan+nanj`
   （只有两个分量都是 NaN 才真的「两边都不是数」），并在 §16.15 登记为 D-S2。

第一处的修法只有一行，但它暴露的问题值得记：**「返回一个 NaN」与「返回
NaN+NaN j」看起来都能过测试**，只有当测试把虚部也断言下来时才会红。

#### 四、三个坑

1. **`vt-ravel` 是零拷贝视图**。quickselect 就地重排，第一版直接拿
   `(vt-ravel in)` 的 data 开工 —— 会把**调用方的输入**改掉。测试里
   `median` 之后复查输入才发现。必须独立分配缓冲区。
2. **堆根是「最差」的那个**，弹出顺序是「劣→优」，输出要**倒着填**
   （`slot = k-1-j`）。第一版正着填：k=1 全过、k≥3 全错 —— 因为 k=1 时
   唯一那个元素填哪都对。
3. **`flet` 里兄弟函数互不可见**。`sift` 调 `worse`，`flet` 编成
   「未定义函数 CLVT::WORSE」，但**编译不报错、只出 STYLE-WARNING**，
   于是上一轮刚立的「STYLE-WARNING 白名单」门禁立刻把它抓住（4 → 8）。
   改成 `labels` 即可。这条顺带证明了第二轮那个门禁的价值。

#### 五、结果

| 算子（1M） | 修前 | 修后 | 倍数 |
|---|---:|---:|---:|
| `median` complex128 | 225 ms | 30 ms | 7.5× |
| `median` c128 `:axis 0`（1000×1000） | 152 ms | 37 ms | 4.1× |
| `nanmedian` complex128 | 216 ms | 30 ms | 7.2× |
| `topk` f64 k=10 | 174 ms | 2.2 ms | 79× |
| `topk` complex128 k=10 | 297 ms | 3.0 ms | 99× |

`median` c128 现在是 30 ms，同形 f64 是 27 ms —— **倍率从 8.3 降到 1.1**，
这才是「算法对齐」该有的样子；`sort`/`argsort` 本身仍是 1.4 倍，那是
复数元素宽度的固有代价。

> 这一轮的教训和上一轮互补：上一轮是「先量最便宜的可疑项」，这一轮是
> **「先看两边走的算法是不是同一条」**。同一个算子、同一个库、两种 dtype，
> 差价 8 倍时不该去调常数因子 —— 先去看 `grep` 出来的两条代码路径。

---

### H.3.21 第二十七轮：四天前刚立的门禁，抓到了它自己那一类问题

#### 一、从一条「同族不同命运」的对比开始

U7 那行写着「复数 nan 族剩余的多趟成本」。真正让人下决心的是把两行数字并排：

| | 1M 元素 |
|---|---:|
| `nanmean` **float64** | 0.69 ms |
| `nanmean` **complex128** | 19.9 ms |

同一个算子、同一个库、同一个「跳过 NaN 求平均」，差 29 倍。查代码：实数侧
在 v0.4.9 就换成了 `%NAN-SUM-COUNT` + `%NAN-SUMSQ` 两个单趟扫描，复数侧
还停在「掩码 + 单位元替换 + 复用归约内核」的流水线（≈7 趟、2~3 个 16 MB 临时）。

沿着这条线量下去，同族的病灶一共四处：

- `nanvar` / `nanstd`：≥10 趟 → 49.8 ms（实数侧 1.4 ms，**35 倍**）
- `nanmax` / `nanmin`：22.4 ms
- `nansum`（复数）7.0、`nansum`（实数）8.9 —— 而 `nanmean`（实数）0.69。
  同一个「跳过 NaN 求和」，两条路差 12 倍，因为 `nansum` 走的是通用
  `%OP-STEP` 循环。
- `nan_to_num`：78.9 ms，因为它用的是 `vt-map` + **闭包**（每元素一次 funcall
  + 装箱）。对照 `vt-copy` 只要 3.4 ms。

#### 二、做法：五个单趟 / 直线内核

| 内核 | 形态 |
|---|---|
| `%CX-NAN-SUM-COUNT` | 复数和 + 计数，一趟 |
| `%CX-NAN-EXTREMUM-COUNT` | 字典序极值 + 计数，一趟 |
| `%CX-NAN-SUMSQ` / `%CX-SUMSQ-ALL` | Σ\|z−mean\|²，跳过 NaN / 不跳过 |
| `%NAN-TO-NUM-BUFFER` / `%CX-NAN-TO-NUM-BUFFER` | 逐分量替换的直线循环 |

只在 `axis=nil` 的全归约上启用 —— 与实数侧 `%NAN-FAST-P` 的判据完全一致，
沿轴版本继续走流水线（那条路的正确性已有差分测试盯着，不值得为它再写一套
逐 fiber 融合）。

`vt-nansum` 有个额外的麻烦：它的门面是 `DEF-VT-REDUCE-FACADE` 宏生成的。
最省事的写法是「先 `(def-vt-reduce-facade nansum …)`，再同名 `defun` 覆盖」，
但那正是 §16.32 明令禁止的「靠加载顺序覆盖」（历史上出过静默错值的桩）。
于是给宏加了一个可选的 `:fast` 钩子（5 参函数，返回 NIL 表示不接管），
「全归约快路径」与「一个符号只有一个 DEFUN」两条约束同时成立。

#### 三、五个坑，每个都对应一次实测

1. **极值扫描不能拿 `a[0]` 当种子**。NaN 参与比较时 `%CLT` 恒假 ——
   若 `a[0]` 是 NaN，`BEST` 会永远停在 NaN 上，后面所有有限值都比不过它。
   用 `FIRST` 标志在「第一个非 NaN」处初始化。
2. **`n=0` 时种子先越界**。空数组的 nanmax 本该走「计数 0 → NaN+NaNj」，
   但 `(aref d off)` 在判计数之前就执行了。
3. **`%NAN-P` 必须在 `with-float-safe` 里调**。`(/= nan nan)` 在 SBCL 上是
   UCOMISD + invalid 陷阱，不是 C 的 quiet compare。`vt-var` 的快分支里
   第一版忘了包，测试立刻炸在「只有 ±Inf」这个用例上。
4. **快路径必须显式要求浮点 dtype**。`%E3-FLOAT-PREFER-DTYPE` 对整数输入
   原样返回 `:int64`，于是 `(eq (vt-dtype v) odt)` 对整数也成立，而
   `%NAN-TO-NUM-BUFFER` 只认 f32/f64 —— 它 `cond` 不中任何分支、返回 **NIL**，
   于是 `(vt-nan-to-num #(1 2 3))` 返回 NIL 而不是报错。这条是测试里
   「整数输入」那个用例抓出来的。
5. **宏生成的门面不能覆盖**（上面已说）。

#### 四、顺手抓到一个公开 API 的崩溃

写测试的「复数一元算子遇 NaN 不抛条件」那一节时，`vt-abs` 直接抛
`FLOATING-POINT-INVALID-OPERATION`。追下去是 `%DEF-CX-TO-REAL-RUN` 生成的
特化循环**没有屏蔽浮点陷阱**：内核里 `%CX-INF-P` 是 `(> v maxf)`、
`%COP-ARG` 要看符号，UCOMISD 遇 NaN 会置 invalid 并触发。

也就是说：**`(vt-abs z)` / `(vt-angle z)` 只要输入里有一个 NaN 就抛条件** ——
而 NaN 恰恰是这两个函数最该正常处理的输入。这跟第二十三轮修的 `vt-asin`
是同一个病（特化循环忘了自己也要屏蔽），只是活了更久，因为没人测过
「abs 一个含 NaN 的数组」。加一层 `with-float-safe` 即修好。

> 这一轮和第三轮（`log` 族）有点像：都不是「算法错」，而是**同一个库里的两条
> 路只优化了一条**。判断标准很朴素 —— 同形 dtype 之间的倍率超过 5 倍，
> 基本就不是「复数固有代价」，而是「有一边没跟上」。

## H.3.22 第二十九轮：把「搬运与归约」从内存带宽里捞出来

> 第二十八轮只改测试（性能护栏改倍率），细节见 `CHANGELOG.md`，本节不重复。

#### 一、起点：四条路里只有两条真的在算

用户点名的六个对象（矩阵乘、连续性判断/连续化、dtype 转换、切片、拼接、求和）
先按「1M 元素、1000×1000、float64」量了一遍。结论分成三类：

| 操作 | 实测 | 判断 |
|---|---:|---|
| `vt-matmul` 1000³ f64 | 98 ms（默认线程） | 慢 3~4 倍，且在 31 线程下崩溃 |
| `vt-sum :axis 0` | 3 ms | 与同数据 numpy 差 20 倍 |
| `vt-amax :axis 0` | 20 ms | 同上，且更糟 |
| `vt-sum` int16 | 7 ms | 掉进通用 odometer |
| `vt-astype` int8→f64 | 8 ms | 每元素一次 `funcall` |
| `vt-contiguous`（转置视图） | 3.4 ms | 通用 walker 每元素重算 stride |
| `vt-concatenate :axis 0` | 0.2 ms | 已经是块拷贝，无空间 |
| `vt-slice` 任意 spec | 0.1 µs | **纯视图构造，本来就不用优化** |

「切片」这一项是虚惊：它是零拷贝的 stride 重写，1M 张量切一刀 0.1µs，
比一次函数调用还便宜。真正贵的是「切片之后干了什么」（连续化/转换），
所以力气全花在那两条路上。

#### 二、矩阵乘：先分清是「算得慢」还是「搬得慢」

旧的 `define-simd-matmul-rows` 是「1 行 × VW 列」的 gaxpy：每个 k 步读一遍
C 的一整行、写一遍；B 的每一行被 m 次重读。1000³ 时 B 要从内存过 8 GB ——
单线程 235 ms 与「纯内存带宽上限」对得上，与算力无关。

换成标准三层分块后：

- 寄存器微块 4×8（f64）/ 4×16（f32）：C 的一块在整段 k 里驻留寄存器，
  C 只在 kc 边界读写一次；B 的每个向量被 4 行复用；
- 缓存分块 MC/KC/NC = 64/256/256：A 128KB + B 512KB + C 128KB，一起进 L2；
- 内层指针步进：每个 k 步只有 4 次 A 指针自增 + 1 次 B 指针自增。

单线程 235 ms → 82 ms。**这就是这台机器 AVX2 的上限**：同一份数据
f32 只要 39 ms（8 宽对 4 宽，正好差 2 倍），说明两条都贴住了峰值；
而 numpy 单线程 1000³ 是 22 ms —— 它用的是 AVX-512（本机 numpy 的
`show_config` 报 `AVX512_ICL`）。sb-simd 只有 AVX2，所以单线程没得再挖，
剩下的只能靠线程。

#### 三、线程数：31 线程比 8 线程慢 3 倍

原默认是 `(- *processor-number* 1)` = 31。实测（1000³ f64）：

| 线程 | 1 | 2 | 4 | 8 | 12 | 16 | 24 | 31 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ms | 93 | 46 | 26 | 16 | 13 | 12 | 11~16 | 11~87 |

两件事同时成立：

1. **只切行是错的**。每个线程都要把整个 B 过一遍，线程数翻倍、B 的
   DRAM 流量也翻倍。改成 (i, j) 二维网格后，总流量 ≈
   `(m/MC)·k·n + (n/NC)·m·k`，与线程数几乎无关。
2. **线程数不是越多越好**。31 线程出现过 87 ms 的崩溃式退化（30 个线程
   各自的工作集互相踢出 L2，而线程创建/汇合的开销摊不薄 8 ms 的活）。
   默认上限因此取 `min(16, 核数-1)`。

默认线程下 1000³ f64：98 ms → 12~16 ms（6~8 倍）。

#### 四、归约：把「按列跨 stride」换成「整行推进」

`%kernel-single-axis` 的内层是「固定列 i，沿归约轴走」，步长 = INNER。
axis=0 时 INNER = n（8KB）：每个元素换一条 cache line，整列扫完再扫下一列
时那些行早被挤出 L2。1M f64 求和 3 ms，numpy 0.15 ms。

新的 `%kernel-row-acc` 把内层维整体当累加器（`acc[0..INNER)` 常驻 L1），
按行推进，两个流都是顺序的。**关键是它没有改变任何求和顺序**：
同一个输出元素看到的仍是 r = 0,1,2,…，只是把「不同输出元素之间」的循环
翻到了外层，所以结果与旧路径**逐位相同**（当场用独立三重循环对拍验证）。

1M f64 axis=0：sum 3→0.45 ms，amax 20→1.7 ms，prod/nansum/mean 同量级。

顺带把 4 种窄整型（int8/int16/uint8/uint16）也接进这条路径：它们原先
只能走路径 3 的 odometer（每元素重算多维下标），1M int16 求和 7 ms →
0.5 ms。

还多接了一类：**归约轴是连续区间**的多轴归约（`(0,1)`、`(1,2)` …）。
只要归约轴在 shape 里连成一段 `[a,b)`，多层轴就等价于一层
（`outer=∏[0,a)`、`red=∏[a,b)`、`inner=∏[b,rank)`），行累加内核直接可用；
`inner` 太窄时退回列扫内核。`(50,100,200)` 求 `axis=(0,1)`：
9 ms → 0.5 ms，与朴素三重循环参考逐位一致。

**收尾：三块分派合成一块，并修掉两个自己埋的坑。**

第一版把「单轴存储级」「单轴窄整型」「多轴」写成了三条独立 clause，其中
多轴那条为了在四层嵌套的反引号模板里插 clause，用 `(list (list (list 'and …)))`
手工拼了二十多行代码数据 —— 既读不懂，也确实出了两次括号事故。改法：

- 几何条件抽成 `%CONTIG-AXIS-TEST`（4 条断言，`(consp axes)` + 双连续 +
  「归约轴连成一段」的 loop）；
- clause 的 body 抽成 `%CONTIG-AXIS-BODY`，用**一层反引号**写真代码；
- `%CONTIG-AXIS-CLAUSE` 只做 `(cons test body)` 的装配，模板里剩一行
  `,@(%contig-axis-clause op arg-p #'dispatch)`。

三块合一之后顺带暴露并修掉两个 bug：

1. **窄整型多轴直接抛错**。旧的多轴 clause 只检查几何条件、不检查 dtype，
   于是 `(vt-sum int16-3d :axis '(0 1))` 落进存储级 dispatch，报
   `unsupported input dtype (SIGNED-BYTE 16)` —— 这条路以前走路径 3，
   是我把它拉进快路径时掐断的。现在窄整型有 `%SMALL-INT-AXIS-FORM`
   统一分派（行累加 / 列扫）。
2. **窄整型 `amax` 报 `unsupported output dtype`**。窄整型的 max/min
   结果**保持**输入 dtype（int8 进 int8 出），而分派表只列了 4 种存储级
   输出类型。改成遍历 `+ALL-LTS+`（存储级 + 窄整型）即可。

对拍：8 种 dtype × 3 种 shape × 全部轴组合（单轴/多轴/不连续轴）×
4 个算子 + argmax/argmin，共 592 组，与「按 C 序累加进输出桶」的朴素参考
**逐位相同**（float32 的参考也用 single 累加，否则比的是精度不是实现）。

#### 五、一条 `declaim inline` 值 10 倍

`amax` 换成行累加后仍要 14.6 ms，而 sum 只要 0.45 ms —— 同样的循环、
同样的访存，差别只在 `%op-step` 的极值分支里有两次 `%nan-p` 调用。

`%nan-p` 的形参类型是 `T`（`(and (floatp x) (not (= x x)))`）。不内联时
函数体只能按泛型数值编译，`=` 走 `TWO-ARG-=` 的运行时派发，一次调用
20~30 个周期；每元素两次，正好是实测的 14 ns/元素。

`(declaim (inline %nan-p %nan-or-inf-p))` 之后，调用点已知元素类型，
`(floatp x)` 折成常量 T、`(= x x)` 折成一条 COMISD：14.6 ms → 1.2 ms。

这条**必须**放在 `util.lisp` 而不是 `nan.lisp`：`util.lisp` 里的
`vt-numpy-sort` 已经调用了它，声明晚于调用点会得到
「could not be inlined because its source code was not saved」，
内联**静默失效**（只有一个 STYLE-WARNING）。两个谓词的定义因此整体
搬到了 `util.lisp`（它们不依赖 nan.lisp 的任何东西）。

#### 六、astype 与连续化：把「逐元素 funcall」换成特化循环

- 窄整型源（4 种）原先没有特化分支，落到通用 fallback：每元素一次
  `(funcall (vt-cast-fun ...))`。补上 4×8 个 `spec` 实例后
  int8→f64 8 ms → 0.7 ms。
- `vt-contiguous` / `vt-copy` 的非连续分支原走 `vt-copy-into` 的通用
  walker（每元素重算 shape/stride）。改走 `%gather-contiguous`
  （末轴展开 + 其余轴 odometer、按元素类型特化）：转置 1M 3.4 → 1.2 ms。
  注意 `%gather-contiguous` 的「未知元素类型」兜底原本 `return-from`
  到 `vt-contiguous` —— 现在两者互相调用，那条兜底必须改成直接
  `vt-copy-into`，否则就是死循环。

#### 七、三个坑

1. **`ql:quickload` 会骗人**。改完文件在同一秒内编译，ASDF 按秒级
   时间戳判定 fasl 是新的，于是「LOADED」但跑的是旧代码；等测试套件
   强制重编时才炸（`undefined variable N-NON`）。判断编译是否真的发生，
   要看输出里有没有 `; wrote ...fasl`，不能只看「没报错」。
2. **`(setf ,@pairs)` 的 pairs 必须是扁平序列**。第一版写成
   `((place value) (place value))`，展开成 `(setf (p v) (p v))`，
   SBCL 报的是「不是合法的函数名」—— 因为 `(setf (p v))` 只有一个参数时
   会被当成 `(function (setf p))` 的查找。
3. **`cond` 的 clause 不能「中途失败再落到下一条」**。窄整型的快路径
   先写成了 `(clause (if 满足 (行累加) (通用内核)))`，而通用内核要的
   `n-red/n-non/...` 只在路径 3 的 `let*` 里绑定 —— 报 undefined variable。
   正确写法是把「满足条件」并进 clause 自己的条件里，不满足时这条 clause
   干脆不匹配，自然落到路径 3。

#### 八、这一轮改到的文件

| 文件 | 改动 |
|---|---|
| `src/simd-matmul.lisp` | 重写 GEMM 内核（分块宏 `DEFINE-BLOCKED-GEMM` + 标量/向量行尾块 + 二维网格调度）；线程数默认上限 16 |
| `src/reduce-stats.lisp` | 新增 `%KERNEL-ROW-ACC`（路径 2b）与窄整型路径 2c；`+ROW-ACC-MIN-INNER+` 门槛 |
| `src/util.lisp` | `%NAN-P` / `%NAN-OR-INF-P` 移入并 `declaim inline`（必须先于所有调用点） |
| `src/nan.lisp` | 迁出上述两个谓词（其余不变） |
| `src/core.lisp` | `vt-astype` 补窄整型 ×8 目标特化；`vt-contiguous` / `vt-copy` 非连续分支改走 `%GATHER-CONTIGUOUS` |
| `CONVENTIONS.md` | §2.2 补连续化实现；§7.3 改线程数/网格契约；新增 §7.4 快路径清单 |

## H.3.23 第三十轮：`vt-astype` 的「两趟」是怎么被找出来的

> 上一轮把 astype / 连续化 / 归约的搬运层统一到 `%MEMCPY` + `%GATHER-CONTIGUOUS`，
> 细节见 `CHANGELOG.md`（二十九），本节不重复。

#### 一、表里的倍率对不上

上一轮的验收表里，转置视图的 `vt-contiguous` 是「3.4 ms → 1.2 ms」，而
`vt-copy` 的转置是「2.6 ms → 1.2 ms」—— 两条都落在 1.2 ms，看着很一致。
但把同一份数据做成**切片视图**（`a[::2,:]`，500×1000，内层 run = 1000）再量：

| 视图 | `vt-astype → f32` |
|---|---:|
| 连续 1000×1000 | 0.55 ms |
| 切片 `a[::2,:]`（半张表） | **1.23 ms** |
| 转置（半趟等效数据） | 1.92 ms |

半张表的转换比整张表还慢 —— 只能是「趟数」不对。读代码确认：非连续分支
`(%inner-run-suffix ...)` 只用来判「run 够不够长」，够长时**跨 dtype** 仍走
`(vt-astype (vt-contiguous tensor) new-dtype)`，也就是「先收集成连续副本，
再整数组转换」：一块 4 MB 临时分配 + 两趟全量流量。同一份数据上，
转置视图（run = 1）走 `%gather-contiguous` 只要一趟，切片视图反而要两趟。

#### 二、为什么不能只加一条分支

给「跨 dtype + 长 run」加一条融合分支，要的是同一张 10×10 特化表的
**另一套落点**。当时那张表是写在 `(macrolet ((spec ...)) (cond ...))` 里的
62 条手写 clause，`spec` 展开成「`(dotimes (i size))` 写满整个 `new-data`」，
没法按 run 复用 —— 直接复制一份就是 62 条 clause 的手抄，正式把
「快路径与通用路径各写一套」这个坑再挖一遍（§7.4 的原话）。

改法是把循环的**起止**和**落点**参数化：`spec` 展开成
「`dp` 从 `p` 起、`dq` 从 `q` 起、共 `len` 个」，然后整张表提成局部函数
`%cast-run (dst p q len)`：

- 连续视图：`(%cast-run new-data in-offset 0 size)` —— 一次调用，参数全常量；
- 切片视图：`%map-inner-runs` 每段 run 调一次 `(%cast-run new-data p q run-len)`。

62 条 clause 一字未改（只把 `(aref dst i)` 换成 `(aref d dq)`，编译器对
`dq` 与对 `i` 做的是同一件事：strength-reduce 成指针自增），所以连续路径
的耗时实测不变（0.55 ms，改前 0.55 ms）。

#### 三、两个当场踩到的坑

1. **`return-from` 不是可选项**。融合分支写完后，`%gather-contiguous` 那条
   分支被我顺手写成了 `(if ... tmp (vt-astype tmp new-dtype))` —— 看着等价，
   但函数末尾还有一句 `new`（调用方预分配的那个数组）。于是所有非连续
   **短 run** 路径都返回了未写满的 `new`：34 条「转置/翻转/切片/广播视图
   与连续副本一致」全红，而 100 格连续矩阵全绿 —— 正好是「非连续分支唯一
   出口」这条约定（§2.2）漏掉的表现。
2. **`(declare (ignorable ,lt))` 不是「忽略宏参数」的写法**。`memcpy` 宏的
   形参是类型说明符 `(signed-byte 64)`，展开成 `(declare (ignorable (signed-byte 64)))`
   直接被编译器判为「不是合法的变量名」。宏参数不用就不会有 style-warning，
   什么都不写才对。

#### 四、验收

| 场景 | 改前 | 改后 |
|---|---:|---:|
| `astype a[::2,:]` f64→f32 | 1.23 ms | 0.33 ms |
| `astype a[::2,:]` f64→i32 | 1.36 ms | 0.87 ms |
| `astype a[::2,:]` c128→c64 | 1.63 ms | 0.64 ms |
| `vt-copy a[::2,:] :dtype :float32` | 0.82 ms | 0.33 ms |
| `astype a[::2,:]` f64→f64（同型） | 0.23 ms | 0.24 ms |
| `astype` 连续 1000×1000 f64→f32 | 0.55 ms | 0.55 ms |
| `astype` 转置 f64→f32 | 1.92 ms | 1.92 ms |

62 套件全绿；新增护栏 G8 把「切片视图不得退回双趟」钉成倍率断言
（`≤ 1.8×`，双趟是 4×）。

---

## H.3.24 第三十一轮：AVX-512 GEMM —— 在没有 VOP 的 SBCL 上发机器码

> 结果与对外承诺见 `CHANGELOG.md`（三十一），本节写**怎么把它试出来的**：
> 哪些路走不通、字段是怎么反推的、以及那些只有跑起来才会露头的坑。

#### 一、先说走不通的路

**路线：写 VOP。** 目标是在 `:generator` 里发 zmm 指令。三个硬阻塞：

1. SBCL 2.6.8 的**目标映像里没有** `sb-assem:define-instruction`（发行版构建
   时 `sb-assem` 只保留了 `inst` 等少数符号）；
2. 没有 `simd-pack-512` 的 primitive type，也就没有 zmm 的 TN，VOP 的
   `:operands` 写不出来；
3. `(sb-assem:inst 'vmovupd ...)` 要求目标符号 intern 在 `SB-X86-64-ASM` 且
   编码表里有 EVEX 编码 —— 两者都没有，加进去要重建 SBCL 本身。

**路线：等 sb-simd。** sb-simd 的 512 支持没有时间表，而且它走的是
`define-intrinsics` 那套，等于同样依赖 1、3 两条。

**剩下的路：运行时自发机器码**。不需要编译器、不需要重建 SBCL，代价是要
自己保证编码正确 —— 而这正是测试（`test/gemm-avx512-test.lisp` 49 项）与
C 对拍（660/660）存在的意义。

#### 二、EVEX 字段是反推出来的，不是抄来的

`62 P0 P1 P2 opcode modrm [sib] [disp]`。按手册写的第一版，生成的字节和
gas 对不上，而且差得没有规律。于是反过来做：用 gas 生成 32 个 zmm 寄存器
两两组合的 `vaddpd`，把每个字段的每一位拆出来看它随寄存器号怎么变。
结论：

| 字段 | 语义 |
|---|---|
| `R̄` `X̄` `B̄` | **取反**（寄存器号第 3/2/1 位取反后填） |
| `R'` `V'` | 对外部寄存器更**取反** |
| `v` (4 位) | ⚠️ **不是**直接放 `vvvv`：而是 `~(vvvv << 1) & 15` |
| `L'L` `b` `aaa` `z` | 与手册一致（512 位时 `L'L=10`） |

最后那条「v 字段是左移一位再取反」是最反直觉的一处，也是第一版全部指令
都错位一格的原因。改完之后，发射器原型输出的字节与 gas 产出**逐条一致，
差异 0**（`objdump -d` 比对 200 多条）。

#### 三、四个只有跑起来才露头的坑

1. **`push %r13/%r14/%r15` 漏了 ⇒ `free(): invalid size`。** 内核用了 r12–r15
   当行指针，但只保存了 rbx/rbp。SysV 里 r12–r15 是**被调用者保存**，不恢复就
   把 SBCL 的运行栈/GC 状态改坏了 —— 表现不是算错，而是几十次调用之后 glibc
   在 `free` 里报堆损坏。补 push/pop 之后，栈偏移全部要重算
   （6 个 push 之后，`[rsp+8]` 的列块数变成 `56(%rsp)`）。
2. **`add $0x80` 被编码成 imm8。** `83 /0 ib` 的立即数是有符号 8 位，
   `0x80` 即 -128。A 的行距一旦超过 127 字节（任何真实矩阵都是），指针就
   往回跳 128 字节。改成「能塞进 imm8 才用 83，否则 `81 /0 id`」。
3. **base 是 rsp/r12 时必须带 SIB。** ModRM 的 rm=100 表示「后面还有 SIB」，
   不是「寄存器 rsp」。第一版在 `(%r10,%riz,2)` 上生成了一条垃圾指令，
   跑起来结果随机。
4. **掩码 16 位不能一次塞进 k1。** `kmovw` 只有 16 位，但列掩码是
   8+8 两段，第一版把高 8 位左移 8 塞进同一个 k1 —— 第二段载入用了错误的
   掩码，越界写。改成 k1 = 低 8 位、k2 = 高 8 位，两个载入各自带 `{k}{z}`。

#### 四、数值上唯一一次「差点走偏」

第一版内核把累加器**清零**，末尾用 `vaddpd` 加回 C —— 少一次载入、少一份
寄存器压力，实测快 3%。对拍立刻炸：同一个元素 C 侧是 `0`，Lisp 侧是
`-4.99e-16`。原因是 sb-simd 路径的链是 `((C + a₀b₀) + a₁b₁)…`，而清零版本
是 `((0 + a₀b₀) + …) + C` —— 后者在 C 远大于 Σab 时会丢掉整个 Σab 的低位。
改回「从 C 载入、末尾原样写回」后，1000³ 上 1000000 个元素 0 差异。

这条经验后来写成了 `CONVENTIONS` §7.3 的一句话：**GEMM 的累加器初值属于
契约的一部分，不是优化项**。

#### 五、为什么小 n 会有 1 ulp 差异（以及为什么它不是 bug）

逐位对照在 `n = 8/16/300` 全绿，在 `n = 5/7/15/17/31` 却有零星几个元素差
1~2 ulp。差异位置有严格规律：**只有最后 `n % 4` 列**。

读 `build-driver` 就明白了：sb-simd 把列分成三段 —— 8 列微块、按 4 列对齐
的整宽向量、以及最后 `< 4` 列走**标量** `(incf c (* a b))`（乘、加各一次
舍入）。512 内核对这些列用的是 FMA（一次舍入）。所以差异不是内核算错，
而是**旧的标量尾列本来精度就低一点**。

于是测试的判据不是「放宽阈值」，而是**结构化的**：差异必须全部落在
`j ≥ n - (n % 4)`，且相对误差 ≤ 1e-15；`n % 4 = 0` 时这个集合为空，自动
退化成严格逐位相同。差异一旦跑到别的列，立刻是红的 —— 那才是真 bug。

#### 六、验收清单

| 项 | 结果 |
|---|---|
| 与 sb-simd 逐位（`n % 4 = 0`，含 1000³） | 0 差异 |
| 与 sb-simd 逐位（`n % 4 ≠ 0`） | 差异仅在最末 `n % 4` 列，≤ 4.2e-16 |
| C 侧对拍（`kdrv.c`，660 组形状） | 660/660 |
| 写边界（前后哨兵 + 非对齐 C） | 未越界 |
| 500³ / 1000³ / 1500³ 单线程 vs numpy | 1.02× / 0.77× / 0.74× |
| 测试套件 | 63 套件、新增 49 项、护栏 G9 |

#### 七、还没做的

- **行尾 MR=4 的 512 内核**：现在 `m % 8` 的余数交回 sb-simd，占整块算力
  1% 以内，但 500³ 这种「m 小」的场景会放大（实测差距主要在固定开销）。
- **对齐**：SBCL 的数组数据只保证 16 字节对齐（实测 1000×1000 f64 的 SAP
  低 6 位 = 16），内核用的是 `vmovupd`。改成对齐载入需要在调用前多一次
  拷贝，不划算。
- **f32 / 复数**：同样的发射器可以直接扩到 f32（`vfmadd231ps`，`{1to16}`），
  但 f32 GEMM 目前的占比不足以证明这笔投入。

---

## H.3.25 第三十二轮：SBCL 2.6.9 的两个面 —— sb-simd 的 512「能用但慢」，与一次推断失真导致的 NaN 崩溃

> 对外结论见 `CHANGELOG.md`（三十二），契约见 `CONVENTIONS.md` §7.5 / §16.30.6。

#### 一、先回答「能不能直接调」

2.6.9 的 NEWS 第一条就是 `enhancement: addition of some AVX512 support to
SB-SIMD`。装上之后 `(require :sb-simd)`，`sb-simd-avx512f` 包导出 1370 个
符号，f64 一侧 `f64.8` / `f64.8-fmadd` / `f64.8-broadcast` / 比较与水平归约
与 AVX2 的 `f64.4` 一族**完全同构**，`sb-ext:simd-pack-512` 也是内建类，
`(sb-ext:simd-pack-512 double-float)` 是精确类型。

也就是说：**不需要 VOP、不需要发机器码**，`define-blocked-gemm` 的算子参数
传 `sb-simd-avx512f:f64.8*` 就能生成 512 位微块，改两行的事。于是先量：

| 1000³ 单线程 | ms |
|---|---:|
| sb-simd 512，8×8 | 45 |
| sb-simd 512，8×16 | 52 |
| sb-simd 512，4×16 | 66 |
| 自写机器码 8×16 | **27** |
| sb-simd AVX2 4×8 | 82 |

最快的 sb-simd 形态（45 ms）仍只有自写内核（27 ms）的 0.6×，而且**比 AVX2
的 8×8 还慢** —— 如果只是「512 位更宽所以更快」的朴素预期，这里就该警觉。

#### 二、反汇编把原因钉死在一行上

反汇编 8×8 微块，内层每 5 条指令一组：

```
VFMADD213PD ZMM0, ZMM1, ZMM26
VMOVUPS    ZMM26, ZMM0
VFMADD213PD ZMM0, ZMM0, ZMM27
VMOVUPS    ZMM27, ZMM0
...
```

40 条 `VFMADD213PD` 后面跟了 34 条 `VMOVUPS`。根因是**语义模型**而不是实现
质量：`f64.8-fmadd` 是纯函数（三参一返、`movable`+`flushable`），值语义要求
结果落到约定的返回寄存器 `ZMM0`，而累加器住在另一个 zmm 里，于是每条 FMA
都要搬一次。自写内核里 `vfmadd231pd zmm2, zmm1, [mem]{1to8}` 的第三个操作数
**就是累加器本身**（读-改-写），一条指令顶两条。

搬移还在前端：每条 FMA 多一个 uop，正好把 FMA 端口留出的余量吃回去。这解释了
为什么 512 位反而打不过 256 位 —— 8×8 的 AVX2 微块每条 FMA 也是 2 uop，但
寄存器压力小得多、没有 EVEX 前缀的取指代价。

**顺便复核了 zmm 的 VOP**（第一轮的结论是否在 2.6.9 变了）：没有。
`:scs (sb-vm::double-avx512-reg)` 能「编译通过」，是因为这个名字
`fboundp` —— 编译器把它当成一次**函数调用**，而不是一个存储类；换成精确的
`:arg-types` 后立刻报「不是合法的 SC」。`sb-assem:define-instruction` 依旧
为 NIL，只有 `inst`；`SB-X86-64-ASM` 里没有 EVEX 编码表。

结论：**自写机器码仍是唯一能拿满 512 位的路线**，与「2.6.9 是否支持 AVX-512」
无关 —— 支持的是 sb-simd 的**函数式**通路，不是可发射的指令层。

#### 三、第二个面：一个 NaN 让整套测试崩在 62/63

装好 2.6.9 跑全量：63 套件、62 过、1 红（`nan-fused-test`），其余
（包括 49 项 AVX-512 GEMM 测试）全绿 —— 自写机器码内核与编译器版本无关，
这点先确认了，否则后面分不清是内核坏了还是环境变了。

红灯现场：

```
Unhandled TYPE-ERROR in thread "main thread":
  The value
    #<DOUBLE-FLOAT quiet NaN>
  is not of type
    (OR (MEMBER 0.0d0 0.0) (FLOAT (0.0)))
  from the function type declaration.
0: (%CX-NAN-VAR-GLOBAL "VT-NANVAR" #<VT ... [+inf, -inf, 1.0-2.0j]> ...)
```

输入是 `[+inf+0i, -inf+0i, 1-2i]`：`mean` 的实部 `(+inf + -inf + 1)/3` 是
NaN，`Σ|z-mean|²` 自然也是 NaN —— 这正是 `CONVENTIONS` §16.30 要求的传播。
**错的是类型，不是值。**

容易走错的一步：backtrace 顶层是 `%CX-NAN-VAR-GLOBAL`，看着像它的返回类型
有问题。一看 `(sb-introspect:function-type #'%cx-nan-var-global)` 是
`(FUNCTION (T T T T T T T) *)`，没事；有问题的是**被它调用的**
`%cx-nan-sumsq`：

```
(FUNCTION (T T T) (VALUES (OR (MEMBER 0.0d0 0.0) (FLOAT (0.0))) &OPTIONAL))
```

检查是在**调用点**做的（内联进调用者帧，所以 backtrace 顶层落在调用者身上）。
根因链：2.6.9 起 `(* x x)` 的返回类型被推断成「非负浮点」→ `(+ a (* dr dr))`
仍是「非负」→ 整个 `let` 的 `a` 与函数返回值都被压成 `(FLOAT (0.0))`。
而 **NaN 不属于任何 `(float lo hi)` 类型**（它与 0 的三种序比较全为假），
于是这条推断在「参与乘法的量是 NaN」时不成立 —— 是**推断失真**，不是我们的
语义错了。

#### 四、四试一成才找到不花钱的修法

`a` 明明声明了 `(type ,rtype a)` 也没用：声明与推断**取交集**，NaN 照样被
排除。于是逐个试：

| 写法 | 结果 |
|---|---|
| `(the double-float (* x x))` | 无效 |
| `(sb-ext:truly-the double-float (incf a ...))` / 包住乘法 | 无效（`truly-the` 只在 NODE 层截断，回写变量时又并回推断） |
| `(declare (optimize (safety 0)))` | 无效 —— 被拦的不是数组/类型安全，是「函数返回值约定」 |
| 局部 `flet` + `(declare (notinline acc))` | **有效**，但每元素多一次函数调用 |
| `(declaim (ftype (function (t t t) (values float)) %cx-nan-sumsq))` | **有效，且零运行时开销** |

选最后一条：`ftype` 决定 fdefn 上记录的类型（调用点据此插检查），循环内的
数值推导照旧，`(speed 3)` 的代码质量不受影响。`(values float)` 是**真话** ——
NaN 是 float，两种精度都被覆盖；比 `(values t)` 保留的信息多，比
`(values double-float)` 不撒谎（单精度输入也会走这里）。

一句话记住：**凡是允许返回 NaN 的内部浮点函数，都要显式写 `ftype`；
让编译器「推断」出来的返回类型去当契约，迟早会被一次 NaN 打穿。**
这条已写进 `CONVENTIONS` §16.30.6。

#### 五、验收

| 项 | 2.6.9 |
|---|---|
| `nan-fused-test` | 302 项全过（修前崩在 `TEST-NANVAR`） |
| `gemm-avx512-test` | 49/49、1000³ 27 ms（与 2.6.8 逐位、逐毫秒一致） |
| 全量套件 | 63 套件全绿 |
| 编译体检 | WARNING 0、STYLE-WARNING 4（白名单原样） |
| 文档门禁 | `check_docs.py` 196 个锚点全解析 |

环境侧：2.6.9 用 `./install.sh --prefix=/usr/local` 覆盖安装，2.6.8 保留为
`/usr/local/bin/sbcl-2.6.8` + `/usr/local/lib/sbcl-2.6.8`（需
`SBCL_HOME=/usr/local/lib/sbcl-2.6.8` 调用），回退只需换路径。

---

## H.3.26 第三十三轮：布尔不是一种整数 —— 把 `:int8` 的 0/1 换成真的 `:bool`

> 对外结论见 `CHANGELOG.md`（三十三），契约见 `CONVENTIONS.md` §17。

#### 一、问题不是「少一个 dtype」，而是「两件事被同一个表示占了」

库里有九类函数按 NumPy 的语义**必须返回 `bool_`**：比较（6 个）、逻辑（4 个）、
谓词（6 个）、`any`/`all`、`isnan`/`isinf`/`isfinite`/`isposinf`/`isneginf`/
`signbit`、`isclose`、`isin`/`in1d`。但它们此前统一返回 `:int8` 的 0/1，
靠文档里一句「语义上等价于布尔」维持。

这句话挡不住两种事：

1. **算术泄漏**。`:int8` 结果天然能继续参与算术：`(vt-+ mask mask)` 得 2，
   `(vt-sum mask)` 得 `:int8` 且 128 个真就溢出；而 NumPy 的 `bool_` 在算术里
   按逻辑运算走（`True+True=True`）、`sum` 升 `int64`。文档写得再清楚，
   也拦不住调用方把掩码当整数用 —— 因为**类型系统没说不能**。
2. **类型不可区分**。「布尔掩码」与「真 int8 数据」dtype 完全相同，
   吃掩码的 `vt-where` / `vt-compress` / `vt-select` 无法表达、也无法校验
   入参语义；打印出来两者也长得一模一样。

#### 二、决策：新增第九种逻辑 dtype，物理 1 字节

`*vt-dtypes*` 加 `:bool`，物理存储 `(unsigned-byte 8)`，`vt-dtype-itemsize` 给 1
（与 `np.dtype(bool).itemsize == 1` 一致）。它**不进** `*vt-storage-dtypes*`：
存储层没有新类型，只是与 `:uint8` 共用 `(unsigned-byte 8)`，靠 dtype 槽区分。

`%dtype-kind-rank` 给 `bool = -1`（int=0 < float=1 < complex=2）。这一位的用途
不是排序好看，而是单元素预转的放行条件：任何数值 dtype 都能无损承载 0/1，
所以 `bool → 任意数值` 永远放行；反过来 `float → bool` 是**真值化**而不是截断，
不能当普通预转处理。

#### 三、一个必须在表之前拦下的转换：真值化 ≠ 截断

`vt-astype` 原本只有一张按整数语义截断的 10×10 特化表。把 `:bool` 目标丢进去，
`0.5` 会被 `truncate` 成 0 —— 而 `np.array(0.5).astype(bool)` 是 **True**；
`nan` 会被截成 0 —— 而 `np.array(nan).astype(bool)` 也是 **True**。两处都错在
「小于 1 的值」和 NaN，且**错得无声**。

于是 `vt-astype` 在进表之前先分流：`:bool` 目标走 `%to-bool` → 逐元素
`%bool-flag`。规则一共四条：Lisp 布尔按 T/NIL；整数/浮点按 `zerop`
（`-0.0` → False，与 `np.bool_(-0.0)` 一致，`zerop` 天然正确）；NaN/±Inf →
True（`nan ≠ 0`）；复数**任一分量非零**即为真。

同一套规则也用在 `vt-arange` / `vt-linspace` 的 `:bool` 分支与 npy `|b1` 解码上
（后者对任何非 0 字节一律真值化，不信任别人的文件守 0/1）。

#### 四、新 dtype 自己带来的坑：`True + True = 2`

`:bool` 落地后，`vt-+` 这类通用算术算子第一次可能**收到**布尔张量。当时的
`vt-+` 是 dtype 无关的 —— `%op-final-dtype` 判 `bool ⊔ bool → :bool`（对，NumPy
如此），然后内核用 `#'+` 相加，于是 `(1+1)=2` 被**原样写进 `:bool` 数组**。

这条最危险的地方在于它是静默的：写回路径是 `(safety 0)` 的特化 aref，
`(unsigned-byte 8)` 装 2 不报错；但「布尔数组只含 0/1」这条不变量一破，
下游 `vt-where` 的掩码判定、npy 落盘的字节、打印的 True/False 就全都建立在
一个假前提上。实测确认：

```lisp
(vt-to-list (vt-+ b c))   ; → (2 2 0)，dtype BOOL —— 修前
```

修法是按 NumPy 的 `bool_` 语义**分派**，而不是靠 clamp：

| 算式 | NumPy 2.3.5 | 落点 |
|---|---|---|
| `b + b` | `bool`（`True+True=True`） | `%bool-fold` + `%bool-or-op` |
| `b * b` | `bool`（逻辑与） | `%bool-fold` + `%bool-and-op` |
| `b - b` / `-b` | **TypeError** | 报错并提示 `vt-logical-xor` / `-not` |
| `b / b` | `float64`（`[1., nan]`） | `vt-/` 把 `:bool` 并入 true_divide 提升族 |
| `b // b`、`b % b` | `int8` | `vt-div` / `vt-mod` 显式落 `:int8` |

两个实现细节踩过坑：

* `%bool-fold` 必须是**宏**。`vt-fast-map` 只在 `(function <symbol>)` 字面量形式
  下走编译期内联，传函数值会退化成 `apply #'vt-map` —— 而那条回退路径把
  函数值当成了「张量」，直接报 `参数解析错误`。
* `vt-add` / `vt-sub` / `vt-mul` 是绕过 `vt-+` / `vt--` / `vt-*` 的二元专用入口，
  守卫要单独加一遍；只修通用路径会留下后门。

#### 五、执行层：掩码类型从 sb8 改 ub8，别掉进慢路径

`+vt-mask-out-types+` 原来写的是 `(signed-byte 8)`（`:int8` 掩码），改为
`(unsigned-byte 8)`。但 `%select-fast` 的 `macrolet` 只展开了有符号掩码，
`vt-where` 拿 `:bool` 掩码会**退回 `vt-map`**（实测慢约 8×）——
所以那里同时生成两种掩码类型的循环。

归约侧另有一坑：`def-vt-reduce` 里 `all`/`any` 的小整型分支如果写成
`(,kernel-macro ,op ,lt res-lt)`，`res-lt` 是**生成代码里的运行时变量名**，
宏展开期就变成 `undefined type CLVT::RES-LT`（512 处 style-warning）；
改成按 `+small-int-lts+` 逐项**字面量**展开才对。

#### 六、测试策略：期望值必须与 clvt 的规范 NaN 逐位一致

`test/gen_bool_expected.py` 生成 52 例 NumPy 期望（2.3.5 实测），输入刻意取
`[-1, 0, 0.5, 1, nan, inf, -inf, -0.0, 3]` 这一组「每个分支都踩到」的值。

第一版有 3 例不符（`stringbit-1d` / `logical-or` / `logical-xor`）。根因不是
布尔语义，而是 **NaN 的符号位**：clvt 的规范 NaN 来自 x86 `0/0`，位模式是
**负** QNaN `0xFFF8000000000000`；Python 的 `float("nan")` 是正 QNaN
`0x7FF8000000000000`。两者 `isnan` 一致，但 `signbit` 与真值化会分叉。

处理方式是**改期望、不改实现**：生成器显式构造同一个负 NaN
（`np.array([0xFFF8000000000000], np.uint64).view(np.float64)`）。这条差异本身
写进了 §17.3 的「已知偏差」表，因为它是表示层的差异，不是语义缺陷。

差分探针（`differential-probes-test{,2}.lisp`）里 19 + 23 条期望 dtype 由
`:int8` 就地改成 `:bool`。**没有整体重生成** —— 重生成会混入 numpy 版本差异的
格式化漂移，把一次语义修复变成一次「文本全变」的不可审查提交。

#### 七、把旧契约的测试也改掉（这一项最容易漏）

改 dtype 之后 `test-all` / `core-test` / `uncovered-coverage-test` /
`extensions3-test` / `complex-dispatch-test` / `complex-test` /
`complex-ops-test` 七个套件全红 —— 它们断言的是**旧契约**（`(eq :int8 (vt-dtype
(vt-= ...)))`）。这不是回归，是契约变了而断言没跟着改。

处理原则：每处 `:int8` → `:bool`，并在同一位置**补一条 `:dtype :int8` 仍可用的
断言**——否则「向后兼容」只写在文档里，没有任何机器守着。复数两套期望文件
（`complex-expected.sexp` / `complex-ops-expected.sexp`）连同生成器
（`gen_*.py`）一起改，避免下次重生成又漂回 `:int8`。

#### 八、验收

| 项 | 结果 |
|---|---|
| `bool-dtype-test` | 239 断言全过（A 52 例 numpy 对照 / B dtype 契约 / C 0-1 不变量 / D 互操作 / E 打印 / F npy `\|b1` / G 布尔算术） |
| 差分探针 | `PASS=496/496`、`PASS=405/405` |
| 全量套件 | 64 套件全绿 |
| `check_docs.py` | 203 个锚点全解析 |
| `check_key_sync.py` | `codegen 同步 ✓`，仅剩 numpy 1e-12 级格式化 WARN（pristine HEAD 同样如此） |

一条离线教训：**给一个「语义上等价」的约定配上真类型，才会暴露那些此前
只是「文档没写清」的洞**。`:int8` 冒充布尔的这些年里没人踩到 `True+True=2`，
不是因为不存在，而是因为没人会去加两个掩码；换成 `:bool` 之后，第一次
`(vt-+ b c)` 就把它照出来了。

#### 九、把 `example/` 跑一遍：翻出的是示例自己的三个洞

收尾时跑了 `example/` 下两个文件（`example.lisp` 6898 行 / 106 个 `test-*`
函数、`clvt-perf-test-examples.lisp` 96 条断言 + 约 60 项基准）。这两个文件
此前**从没进过任何自动化**，跑得通跑不通没人知道。

第一次运行：两个都「通过」。但这个「通过」是假的：

1. **9 处 `(assert (vt-all X))` 恒真**。`assert` 只判条件是否为 NIL，而
   `vt-all` 返回的是 VT 对象 —— 任何 VT 对象都不为 NIL。也就是说从
   `:int8` 时代起，这 9 条断言**一秒钟都没检查过任何东西**。换成 `:bool`
   之后更隐蔽：返回值仍然不是 NIL，断言文本却写着「全为真」。
   修法是加一个 `vt-all-true-p`（`(= 1 (vt-item (vt-all x)))`），把 0 维
   布尔张量真正解包成 Lisp 布尔。
2. **解开之后立刻有一条失败**：`(vt-all (vt-isinf mx))` 里 `mx` 是
   `(vt-nanmax 全-NaN)`。作者当年拿不准，注释写的是
   「+inf? 看实现，应该是 most-positive-double-float」。实测两边都不是：
   NumPy 返回 **NaN**（`All-NaN slice encountered`），clvt 也返回 NaN。
   是**示例的期望写错**，实现从一开始就是对的 —— 但恒真断言把它藏了这些年。
3. **基准里 `extract` 测的是错的东西**。`(bench "extract" 3 (vt-extract a mask))`
   把参数写反了：`numpy.extract(condition, arr)` 是条件在前，这里却是
   1M 个 float 当条件、0/1 掩码当数据源。结果既不是 `a[mask]`（长度 999999
   而不是 500000），耗时 216783 µs —— 而语义等价的 `compress` 只要 8479 µs。
   改成 `(vt-extract mask a)` 后落到 21185 µs。**一个参数顺序错误伪装成了
   「`vt-extract` 实现差 25 倍」**，如果直接去动内核，就是拿错误的尺子量。

修完给两个示例补了门禁：`test/example-smoke-test.lisp` 登记进
`run-tests.sh` 的 `DEFAULT_ORDER`，把两个示例当回归跑一遍，**只跑断言、
不跑基准** —— 基准的绝对值随机器与 GC 漂移，做门禁只会变成间歇性红灯。
`clvt-perf-test-examples.lisp` 的断言区另补了 9 条 `:bool` 契约
（dtype / 0-1 取值 / compress≡extract / sum 升 int64 / any-all 形状 /
where 吃 `:bool` 掩码 / `bool+bool` 是逻辑或 / `bool-bool` 必须报错），
总数从 87 提到 96。

顺带修掉一个由 `:bool` 打印引入的小瑕疵：`%format-number` 的布尔分支返回
字面量 `"True"`/`"False"`，是**同一个常量对象**，SBCL 在 `*print-circle*`
为真时会把它当共享结构，打印成 `#1=True, #2=False, #1#, #2#`。浮点分支走
`format` 天然是新字符串，所以只有布尔撞上。加 `copy-seq` 即解。

这一节的教训和第八节那条正好互为正反面：**把「看起来在测」的东西变成
真的在测，比多写十条测试更有价值**。恒真断言不是「弱断言」，它是把
「没测」伪装成「测过」—— 而且在语义变更（`:int8` → `:bool`）时，它的
文字不需要改一个字就继续「通过」。

---

## H.3.27 第三十四轮：布尔的三层视图 —— `item` 给 T/NIL、打印给 T/NIL、`to-list` 保持 0/1

> 对外结论见 `CHANGELOG.md`（三十四），契约见 `CONVENTIONS.md` §17.8。

#### 一、先到来的不是需求，是两个红灯

上游合并补丁后跑全量，两个差分探针红了（`496` 与 `405` 条全表的套件）。
第一反应是「`:bool` 改动破坏了什么」，但本地同样代码全绿。查下去发现根因不在
语义：**探针的期望表是内嵌在 `.lisp` 里的字面量**，由 `gen_probes*.py` 生成；
换了机器或 numpy 版本后，「仓库里的期望」与「本机 numpy 的期望」会差在第 16
位有效数字上。跑套件前没有人重新生成，红灯就被当成回归去排查了。

`check_key_sync.py` 明明每次都在跑，为什么没解决？因为它的职责是**比对**：
重新生成 → 逐字节/语义比对 → `finally` 里 `git checkout --` 把工作区还原成
提交状态。它正确地报告了「格式化漂移 7 + 8 条」，然后**把新生成的版本扔了**。
门禁要的是「发现问题」，而这里要的是「留下新基线」，是两件事。

于是 `run-tests.sh` 多了一个阶段，位置必须卡在 `check_key_sync.py` **之后**
（否则写的基线会被它的 `git checkout` 抹掉）：

```bash
check_key_sync.py   # 比对语义漂移，FAIL 即退出
regen_probes        # 按本机 numpy 重生成并「留下」（新增）
check_docs.py
check_build_warnings
→ 跑套件
```

边界写进了函数注释，因为这一步天然有「自证」风险 —— 自己生成期望、自己跑、
当然绿：

* 语义漂移在上一阶段就 FAIL 退出，走不到这里，所以这里只可能刷新格式化差异；
* 刷新了就把**改了多少行**报出来，并明说「本轮是自洽基线，不是对提交期望的回归」；
* `--no-regen-probes` / `CLVT_SKIP_PROBE_REGEN=1` 走严格模式（CI 用这个）；
* 中间产物通过 `FUZZ_TMP` 指到临时目录，跑完删掉，工作区不会多出 `tmp/`。

#### 二、`True` / `False` 是 Lisp 里的错字，不只是风格

用户提的第二个问题：布尔张量打印成 `True` / `False`，不符合 Lisp 惯例。
这不只是审美 ——

```lisp
(read-from-string "True")   ; → 符号 TRUE，当前包里临时 intern，未绑定
(read-from-string "T")      ; → T
```

`common-lisp` 包里没有 `TRUE` 这个名字。所以 `*print-readably*` 为真时，
`True` 根本读不回来；`T` / `NIL` 可以。而且本库的 0 维形状本来就印
`shape:NIL`（Lisp 惯例，不是 numpy 的 `()`），值却用 Python 惯例，同一次打印里
两套风格。

改动只有 `%format-number` 的一个分支。列宽不用管 —— `print-object` 的
`scan-axis` 是按 `(length (%format-number ...))` 实测的，`T`(1) / `NIL`(3)
自动右对齐：

```
#<VT shape:(2 2) dtype:BOOL
  [[   T,  NIL],
   [   T,    T]]>
```

`*print-circle*` 那个坑照旧要防（字面量字符串是同一个常量对象），`copy-seq` 留着。

#### 三、`vt-item` 给 T/NIL：一行改动，四个地方会崩

`np.bool_(True).item()` 返回 Python 的 `True`，不是 `1`。Lisp 里与 Python
`bool` 同位的就是 `T` / `NIL`。改完 `(when (vt-item (vt-all m)) ...)` 直接成立，
上一轮为了修恒真断言而加的 `(= 1 (vt-item ...))` 这层包装也就可以拆掉了。

但 `vt-item` 有 20 处内部调用，其中 4 处会因为返回类型变化而当场出事：

| 位置 | 原写法 | 为什么崩 |
|---|---|---|
| `vt-allclose` | `(not (zerop (vt-item ...)))` | `zerop` 收到 `T` |
| `vt-histogram` | `(= (vt-item ...) 1.0d0)` | `=` 收到 `T` |
| `vt-extract`（0 维 condition） | `(/= (vt-item ...) 0)` | `/=` 收到 `T` |
| `vt-bincount`（取最大值） | `(1+ (vt-item (vt-amax flat)))` | `1+` 收到 `T` |

前两处改完反而更简洁（返回值本身就是 Lisp 布尔）。后两处不能直接丢进 `if`
或算术：

* `vt-extract` 的 condition 可能仍是 `:int8` 掩码 —— 那时 `vt-item` 给 `1`/`0`，
  而 **CL 里 0 是真值**，`(if 0 'a 'b)` 给 `A`。必须过一层 `%truthy-p`。
* `vt-bincount` 要算 0/1 数值（`np.bincount([True False])` → `[1 1]`），
  过 `%item-number`。

这两个转换原语就放在 `%bool-flag` 旁边，互为反向：`%bool-flag` 收敛到物理
0/1，`%truthy-p` 收敛到 Lisp T/NIL，`%item-number` 把 `T`/`NIL` 推回数值。
`%truthy-p` 自带 `with-float-safe` —— 它挂在索引/归约的分散调用点上，指望每个
调用方记得包一层不现实，而 NaN 的 `zerop` 在陷阱开启时会抛
`FLOATING-POINT-INVALID-OPERATION`（探针实测踩到过）。

#### 四、`vt-to-list` 是这次唯一说「不」的地方

按一致性，`vt-to-list` 也该返回 `(T NIL T)`。但它不能，理由是个 Lisp 特有的
歧义：

```lisp
(vt-from-sequence (list t nil t))
;; => 报错：不规则嵌套
```

`NIL` 既是布尔假、又是空表，`vt-from-sequence` 无法区分「元素 NIL」和「嵌套的
空列表」。只要 `vt-to-list` 返回 T/NIL，`(vt-from-sequence (vt-to-list x))` 这条
往返就**永久不成立** —— 而 `vt-to-list` 的契约恰恰是「可重建的数值序列」。

所以定了三层：`vt-item`（标量）与打印走 Lisp 布尔，`vt-to-list` 默认保持
`0`/`1`，要布尔视图显式 `:bool t`。`vt-ref` / `vt-data` 保持物理字节不动 ——
它们是内存视图，改成 T/NIL 会让所有按字节遍历的内核失效。

一句话总结这张表：**逻辑视图用 Lisp 布尔，物理视图保持 0/1 字节**。

#### 五、连带修改：8 个测试文件的 18 条断言

`vt-item` 换返回类型，直接踩中所有「拿 `vt-item` 结果和数字比」的断言：

| 文件 | 条数 | 改法 |
|---|---|---|
| `core-test` | 2 | `(= (vt-item (vt-all ...)) 1)` → 直接布尔 |
| `numpy-convention-tests` | 2 | 期望 `1`/`0` → `t`/`nil` |
| `test-all` | 6 | 同上 |
| `test-bug0` | 1 | 去掉 `plusp` 包装 |
| `shape-degenerate-tests` | 2 | `'(:ok :scalar 1)` → `'(:ok :scalar t)` |
| `robustness-test` | 2 | 数据驱动的 `T!` 用数值比较，显式 `(if ... 1.0d0 0.0d0)` |
| `run_all_tests` | 2 | 同上（期望来自 JSON 的数值） |
| `complex-dispatch-test` | 1 | `(= 0 (vt-item (vt-any ...)))` → `(null (vt-item ...))` |
| `example.lisp` | 4 | `vt-all-true-p` 简化成 `(vt-item (vt-all x))`，三处 `(= ... 1.0d0)` 改成直接判 |

其中前六个是「测试跟着契约走」，最后一个是「示例跟着契约走」。区分标准：
测试改断言是**对的**（旧契约变了），但如果改的是**被测语义**那才是掩盖回归 ——
这批全部属于前者，因为 `vt-item` 的返回值本身就是被测对象。

#### 六、验收

| 项 | 结果 |
|---|---|
| `bool-dtype-test` | 282 断言全过（新增 H 组 43 条：视图分层/0-1 物理/往返/打印/四个消费点） |
| 全量套件 | 65 套件全绿（详见 `CHANGELOG.md` 三十四） |
| `check_docs.py` | 205 个锚点全解析 |
| `check_key_sync.py` | `codegen 同步 ✓`（格式化漂移由新的 regen 阶段消化） |

一条可以推广的教训：**「门禁报的问题」和「门禁该做的事」不是一回事**。
`check_key_sync.py` 一直在正确报告格式化漂移，但它按设计要把工作区还原 ——
于是每次跑测试都看见红灯，每次都不知道该拿它怎么办。修法不是放宽门禁，
而是在它后面补一个「负责留下结果」的阶段，并且把这个阶段的「自证」边界
写成注释和开关。

## H.3.28 第三十五轮：布尔写不进去 —— `vt-ref` 的 setter、`vt-view` 的白名单、`vt-from-sequence` 的 NIL 叶子

> 对外结论见 `CHANGELOG.md`（三十五），契约见 `CONVENTIONS.md` §17.9。

#### 一、上游合了补丁，但读写两端只通了「读」

上一轮把逻辑视图（`vt-item` / 打印）改成了 `T`/`NIL`，读的方向通了。这一轮
用户提的三件事其实是同一个缺口的三面：**写进去的路还停在「bool 就是 0/1
数字」的旧契约上**。先把现状实测了一遍，避免凭印象改：

| 场景 | 改前实际行为 |
|---|---|
| `(vt-ref b 0)` | `1`，`(type-of …)` 是 **`BIT`** |
| `(setf (vt-ref b 0) nil)` | 报错 `NIL can't be converted to (UNSIGNED-BYTE 8)` |
| `(vt-put b 0 t)` | **静默空操作**，数组没有任何变化 |
| `(vt-put b '(0 1 2) '(t nil t))` | 得到 `(1 1 1 0)` —— 期望是 `(1 0 1 0)` |
| `(vt-flatten-sequence '(t nil t))` | 得到 `(T T)`，静默丢掉 `NIL` |
| `(vt-from-sequence '(t nil t) :dtype :bool)` | 报「不规则嵌套」 |
| `(vt-from-sequence (list nil) :dtype :bool)` | 得到 shape `(1 0)` 的**空数组** |
| `(vt-from-sequence t :dtype :bool)` | 报「无法从 T 创建张量」 |
| `(vt-view b :dtype :uint8)` | 报「仅支持复数两对」 |
| `(vt-view b :dtype :bool)` | 报「元素总数不一致 (旧 4, 新 1)」 |

最后两条不在用户提的清单里，是排查时撞见的：`vt-view` 省略 `:dtype` 时把
「未指定形状」当成 `nil` 形状（0 维），`%resolve-minus-one` 于是给 `(1)`，
对任何多元素张量都报元素总数不一致。**只要 `:dtype` 与自身相同就会中招**，
与 bool 无关，属于老 bug。

#### 二、三个决策，以及每个决策否掉了什么

**1. `vt-ref` 布尔化，并把内部读点分流到 `%vt-ref-raw`。**

`vt-ref` 与 `vt-item` 是同一类东西（按坐标取一个元素），给不同表示没有道理。
但 `vt-ref` 同时是库内部数值代码的取数原语 —— `src/` 里 23 处读点，分布在
`linalg` 的 Gram-Schmidt、`extensions3` 的协方差/相关系数/叉积、
`reduce-stats` 的 `vt-interp` 单调性检查。这些点要的是数字，改完如果还调
`vt-ref`，就得逐个论证「bool 不会流进来」；而这类论证过三个月就失效。

所以拆成两层：`%vt-ref-raw` 是原来的实现（永远 0/1，内部专用），`vt-ref`
在它外面加一层 dtype 判断。23 个读点全部改调前者，好处是内部数值路径连一次
`eq` 都不用付，也不必依赖任何「不会传入 bool」的假设。

`(setf vt-ref)` 必须一起改。布尔化之后 `(setf (vt-ref b 0) (vt-ref b 1))`
成了最自然的写法，而它原来走 `(coerce value (vt-element-type vt))` ——
`(coerce nil '(unsigned-byte 8))` 直接报类型错误，等于「读得出来、写不回去」。
改用 `%bool-flag` 归一，顺带接受 `0.5`、`-0.0`、NaN，与 `vt-fill` / `vt-put` /
`vt-astype` 的布尔语义对齐。

**2. `vt-view` 只放行 `bool → uint8/int8`，反向要求走 `vt-astype`。**

numpy 是双向都允许的，为什么这里只放一个方向？因为 `uint8.view(bool)` 是
**纯字节直译**：

```python
np.array([1, 2, 0], dtype=np.uint8).view(bool)   # [ True  True False]
np.array([1, 2, 0], dtype=np.uint8).view(bool).view(np.uint8)   # [1 2 0]
```

打印是 `True`，值 `2` 原封不动躺在数组里。照搬就等于允许「bool 数组里存着
2」，而 §17.1 的 0/1 不变量是打印、`vt-item`、npy 往返、`sum` 升 int64 这一整
套断言的地基。反向若要做真值化，那是 `astype` 的语义（`2 → 1`），不该藏在
`view` 这个名字底下。所以白名单只加安全方向，报错信息里点明
「bool→uint8/int8」，让人知道替代路径。

**3. `:dtype :bool` 下 `nil` 一律是叶子。**

这是唯一有真歧义的地方，也是唯一必须靠「显式声明」消解的地方。`NIL` 在 CL
里既是布尔假又是空表，不可能两全。选择是：**指定了 `:bool` 就等于声明
「内容是布尔值」，`NIL` 从此只是假**。于是 `'(t nil t)` → `[T NIL T]`、
`(list nil)` → shape `(1)`、`(list nil nil)` → `[NIL NIL]`，而顶层 `nil`
仍然是 shape `(0)` 的空数组（对标 `np.array([], dtype=bool)`）。

代价明确记录在案：`'(() ())` 得到 `[NIL NIL]` 而不是 numpy 的 `(2, 0)` 空数组。
要空数组写 `(vt-zeros '(2 0) :dtype :bool)`。另一条路（加 `:nil-is-element t`
开关、默认沿用旧解释）被否掉了 —— 那等于把坑留着，每次都得记得加开关。

非 `:bool` 模式一行没动：`(vt-from-sequence (list 1 nil 3))` 仍然报「不规则
嵌套」，与改前一致。

#### 三、连带修的两处静默失败

`vt-put` 的布尔标量不在这轮的需求清单里，但它是「`vt-ref` 布尔化」的直接
下游 —— 读能给 `T` 了，写就不会再有人写 `1`。而 `(vt-put b 0 t)` 的实际行为
是**什么都不做**：`T` 不匹配 `numberp` / `listp` / `arrayp` / `vt-p` 任何一个
分支，`val-list` 为 `nil`，`n-values` 为 0，函数在 `(when (zerop n-values)
(return-from vt-put tensor))` 处安静返回。`nil` 更糟，它落进 `(listp values)`
被展平成空表，同样静默返回。

值列表那条更隐蔽：`(vt-put b '(0 1 2) '(t nil t))` 里 `vt-flatten-sequence`
把 `(T NIL T)` 压成 `(T T)`，值列表短一个，`(nth (mod vi n-values) val-list)`
就循环复用，得到 `(1 1 1 0)` —— 一个不报错、也不对的答案。所以
`vt-flatten-sequence` 加了 `:bool` 开关，`vt-put` 在目标为 `:bool` 时自动启用。

**「静默什么都不做」比报错难查一个数量级**：报错有栈，空操作只有「怎么没
生效」。这批修复的验收标准因此不是「新功能可用」，而是「旧写法要么正确、
要么报错」。

#### 四、仓库卫生

用户提到 `git add -f *` 把 `tmp/` 加进了仓库。确认下来是 `gen_probes*.py`
直跑时的默认输出目录（`tmp/{cases,cases2}.json` + `tmp/expected{,2}.txt`），
以及 `test/__pycache__/gen_probes.cpython-312.pyc`。这些对任何流程都不需要：
探针期望内嵌在 `differential-probes-test*.lisp` 里，测试运行时不读
`tmp/expected.txt`（那只在 `gen_probes.py compare` 子命令里用）。

顺带发现上一轮加的 `/tmp/` 忽略规则被上游删掉了，恢复并补上 `*.pyc`。
`git add -f` 能绕过忽略规则，但 `.gitignore` 至少能让 `git add .` 不再捕获。

#### 五、验收

| 项 | 结果 |
|---|---|
| `bool-dtype-test` | **327** 断言全过（新增 I 组 33 条 + H 组由 43 扩到 55，另改写 4 条 astype 断言） |
| 两个差分探针 | `496/496`、`405/405`（严格模式 `--no-regen-probes`） |
| 全量套件 | 65 套件全绿（详见 `CHANGELOG.md` 三十五） |
| `check_docs.py` | 208 个锚点全解析 |
| 内部读点 | 23 处改调 `%vt-ref-raw`，13 处写点走改后的 setter |

一条值得记住的判据：**判断一个视图接口属于逻辑层还是物理层，看它返回的东西
能不能直接参与算术**。`vt-ref` 原来能（所以它是物理视图的伪装），`vt-item`
改完不能（所以它是逻辑视图）。同一个张量上两个接口给不同表示，本身就是错的 ——
这一轮把 `vt-ref` 拉齐到 `vt-item`，才让 §17.8 的分层自洽。

---

## H.3.29 第三十六轮：布尔到底能不能「算」—— 两遍审计、六类静默错值，以及 `T` 传不进去的四个入口

> 对外结论见 `CHANGELOG.md`（三十六），契约见 `CONVENTIONS.md` §17.10。

#### 一、审计怎么设计的

上一轮（H.3.28）修的是「写不进去」，这一轮用户要求「对 bool 的函数改动做
一次深度检查，尽可能找 bug」。没有现成清单，所以先按**两条彼此独立的轴**
各扫一遍，因为这两类缺陷的发现手段完全不同：

| 遍 | 问题 | 手段 | 产出 |
|---|---|---|---|
| 第一遍 | 值算错了吗？ | numpy 逐例差分（152 + 66 条），比对 dtype / shape / 值 + 0/1 不变量 | 6 类缺陷（§17.10.1–17.10.6） |
| 第二遍 | 布尔字面量传得进去吗？ | 30+ 个入口逐个喂 `T`/`NIL`/小样本，人工比对 numpy | 4 个入口 + 1 个 `ECASE` 陷阱（§17.10.8–17.10.9） |

第一遍的框架是「python 生成期望 → lisp 逐条比对」，适合**有明确 numpy 对应物**
的算子；第二遍的输入是「`(vt-xxx b t)` 这种一行调用」，差分框架反而笨重，
直接写探针看输出更快 —— 而且这遍要找的是**能不能进函数**，不是算得对不对。

两遍的共同前提是先把 numpy 侧的行为钉死。有几个结论和直觉相反，值得记：

- `np.diff(bool)` **不报错**，等价于相邻异或（`True-False` 按位回绕成 1，
  落回 bool 仍是 True）；
- `np.invert(bool)` 是逻辑非，而 `(lognot 1)` 写进 `(unsigned-byte 8)` 是 254；
- `bool @ bool` 是**按位与再或**，结果仍是 bool，不是数值和；
- `np.trapezoid(bool)` 给 `float64`（1.5），跟着输入走会截断成 0；
- `np.percentile(bool)` / `np.gradient(bool)` / `np.ptp(bool)` **确实报错**
  （布尔减法未定义），所以这三处 clvt 报错是对的，不该"修"。

#### 二、六类静默错值

按「静默错值 > 报错 > 缺失」排优先级，六类里只有两类会当场报错，其余都是
算出一个看起来合法的错数：

| # | 症状 | 根因 | 修法 |
|---|---|---|---|
| 1 | `(vt-map (lambda (x) 2) b)` → `(2 2)` | 布尔数组里躺着 2，破 0/1 不变量 | 写回**之前**包一层 `%bool-flag` |
| 2 | `(vt-bit-not b)` → `(254 255)` | `lognot` 是字节取反 | bool 分支改派 `vt-logical-not` |
| 3 | `(vt-histogram b)` 报 `NIL is not of type NUMBER` | 自动定 bin 范围取了 `vt-item` 的 T/NIL | 改 `%item-number` |
| 4 | `(vt-diff b)` 报「布尔减法未定义」 | `vt--` 拒绝 bool（且文案说"numpy 同样拒绝"，是错的） | bool 走 `vt-logical-xor`，文案同步改 |
| 5 | `(vt-trapz b)` → **0** | `0.5*(l+r)*h` 按输入 dtype 写回，`.5` 截断成 0 | 整数/布尔按 `:float64` 累加 |
| 6 | `bool @ bool` → `:float64`，值 `[[2 1][1 2]]` | einsum 走数值和 | int32 计数 + 末步真值化 |

第 6 类上一轮曾被记成「有意保留的偏差」，理由是「bool 累加器没有合法表示」。
这个理由只对**累加器**成立，对**结果**不成立：中间用 int32 计数、最后
`vt-astype :bool` 就没有越界问题（200 个 1 的内积实测仍是 `T`）。这种
「局部结论被当成整体结论」是偏差表最危险的写法 —— 它会自我固化。

第 1 类的修法也值得记一笔：第一版写成「写完再扫一遍归零」，实测
`(vt-map (lambda (x) 0.5) b)` 给 `False` —— `0.5` 过 `%coerce-uint8` 已经
变成 0，事后救不回来。必须**在写回之前**归一。

#### 三、`T` 传不进去的四个入口

第二遍一扫，四处失败长得一模一样，都是 `T fell through ETYPECASE`：

| 表达式 | 现象 |
|---|---|
| `(vt-insert b 0 t)` | ETYPECASE |
| `(vt-pad b '(1 1) :constant-values t)` | `invalid constant-values: T` |
| `(vt-where b t nil)` | ETYPECASE |
| `(vt-fill-diagonal m t)` | **静默只填对角第一个元素** |

根在 `ensure-vt` 的 `etypecase`：只有 `vt` / `number` / `sequence`，`T` 是符号。
`vt-fill-diagonal` 那条更隐蔽 —— 它有「标量还是序列」的判据
`(numberp value)`，`T` 判否 → 走序列分支 → `(ensure-vt t)` 是 0 维 →
`vt-ravel` 后只剩 1 个元素 → 只填第一个位置，**不报错**。

`NIL` 的处置要单独说。`T` 没有二义性（不可能是空表），直接在 `ensure-vt`
里当值 1；`NIL` 二义，而 `(ensure-vt nil)` = shape `(0)` 的空表是 300 多个
内部调用点依赖的既定语义（用来表示「没有序列」）。所以只在**显式
`:dtype :bool`** 时把 `NIL` 读成假，其余保持空表。代价明确：`(vt-insert b 0 nil)`
要写 `0` 或 `(list nil)` 才稳妥 —— 但换来的是一批可选参数不会悄悄变成 0。

#### 四、`ECASE` 配 `T`：一个只在 n≥6 炸的陷阱

`%bool-wrap` 按输入个数 1..5 生成包装闭包，其余走 `&REST` 兜底，第一版用的
是 `ECASE`：

```lisp
(ecase n
  (1 …) (2 …) (3 …) (4 …) (5 …)
  (t (lambda (&rest xs) …)))   ; ← T 只是「匹配符号 T」，不是默认子句
```

`CASE` 里 `T` 是默认子句，`ECASE` 里**不是**。它能编译、能过 5 输入以内的
全部测试，只在 `(vt-map f b1 … b6)` 时报 `no key` —— 而 `vt-map` 的 6 输入
是 `%vt-map-run` 的 `otherwise` 分支明确支持的合法用法。改成 `CASE`。

#### 五、验收

| 项 | 结果 |
|---|---|
| `bool-dtype-test` | **388** 断言全过（J 组由 52 扩到 71：新增 J7「值位置的 T/NIL」15 条 + 6/7 输入 2 条） |
| 全量套件 | 65 套件全绿（`--no-regen-probes` 严格模式） |
| 两个差分探针 | 496/496、405/405（严格模式，按仓库内期望判定） |
| `check_docs.py` | 218 个 §锚点全部可解析 |
| 编译体检 | 零 WARNING；STYLE-WARNING 4 条 = 白名单 4 条 |

一条值得记住的判据：**审计「某类型支持得对不对」，要分别问「算得对不对」
和「传不传得进去」**。前者靠差分，后者靠把字面量一个个塞进入口；两者
交集很小 —— 六类静默错值里没有一个是「传不进去」，四个传不进去的入口里
也只有一个是静默错值。

---

## H.3.30 第三十七轮：第三遍换轴 —— 不问「值」问「类型」，不问「名字」问「dataflow」

> 对外结论见 `CHANGELOG.md`（三十七），契约见 `CONVENTIONS.md` §17.10.10 / §17.10.11。

#### 一、为什么还要再来一遍

前两遍的轴是「值」（numpy 差分）和「入口」（字面量探针），各修掉六类和四类。
再想找出新东西，必须换**被断言的对象**，而不是再换一批用例：

| 遍 | 被断言的对象 | 手段 | 产出 |
|---|---|---|---|
| 一 | 值 | numpy 逐例差分 152 + 66 条 | 6 类静默错值 |
| 二 | 「能不能进函数」 | 30+ 入口喂 `T`/`NIL` | 4 个入口 + 1 个 `ECASE` |
| **三** | **结果 dtype 本身 + 物理字节** | `(vt-dtype (f ...))` 断言 + 0/1 字节扫描 | **6 处（本轮）** |

第三遍能成立的前提是一个容易被忽略的事实：**`vt-to-list` 会把 0.785 显示成
`0`**。所以「值不对」这个现象，可能来自算错，也可能来自「算对了但存不下」。
前两遍都默认了前者。

同时换了找缺陷的检索方式：第一、二遍是「挑函数来测」，第三遍先**静态搜
结构**——`(incf sum (* ...))`、`(vt-- ...)`、`'(:int8 … :uint16)`——
再顺藤摸瓜。按名字搜会漏（`convolve` 不叫 `matmul`），按结构搜不会。

#### 二、按 dataflow 搜出的三个缩并

§17.10.6 修完 `matmul` 家族后，我当时的判断是「布尔缩并就这些了」。
这个判断错了，错在**按词找**：修的是名字里带 mat/dot/inner/outer 的那批。
换成「函数体里有没有乘积求和」再扫，立刻捞到 `vt-convolve` /
`vt-correlate` / `vt-cross`。

| 表达式 | numpy | 修复前 | 现在 |
|---|---|---|---|
| `(vt-convolve b b)` | `:bool` `[T F T T T T T]` | `:float64` `[1 0 2 2 1 2 1]` | `:bool`，逐值一致 |
| `(vt-correlate b b)` | `:bool` 全 `T` | `:float64` 数值和 | `:bool` |
| `(vt-cross b b)` | 报错 | `:bool` `(1 1 1)` | 报错 |

`cross` 那条最值得记：它不是「类型错」，是**静默的错值**。叉积
`a1*b2 - a2*b1` 里 `-1` 被截成 1、`1` 保持 1，三个分量全变 1 ——
输出一块看起来非常整齐的 `(1 1 1)`，不报错、不警告。而 numpy 在这里
直接抛 `TypeError`。

#### 三、按 dtype 搜出的四条

这四条的共同点是「docstring / 偏差表说的和做的不一样」，而**两边单看都对**：

| 入口 | 宣称 | 修复前实际 | 现在 |
|---|---|---|---|
| `(vt-negative b)` | numpy 拒绝 | `:bool`，字节 255 | 报错 |
| `(vt-atan2 int int)` | 「整数输入输出 float64」 | `:int64`，值全 0 | `:float64` |
| `(vt-pow b 2)` | §17.10.7 记「clvt 给 `:int64`」 | `:float64` | `:int64` |
| `(vt-rem b b)` | 与 `vt-mod` 同族 | `:bool` | `:int8` |

`vt-negative` 的 255 值得单独说：它与 §17.10.2 的 `vt-bit-not`（254/255）
是同一个坑的**第二次踩**。上一轮修 `bit-not` 时我以为「布尔取反只有这一处」，
但真正的根因是「`-1` 经 `%coerce-uint8` 写进 `(unsigned-byte 8)`」，
而这条路径对所有产生负值的算子都成立。所以这轮之后加了一条自检：
**任何 `:bool` 结果的物理字节必须 ∈ {0,1}**，扫一遍比逐函数断言有效。

`vt-atan2` 最有意思：它根本不是 bool 缺陷 —— **整数输入也全错**
（`atan2(2,2)` 给 0）。只是 bool 更容易撞上：整数至少还能装下 0，
bool 连 0.785 都装不下。发现它是因为我把 `atan2` 放进了「浮点算子 ×
{bool, int64, float32}」的矩阵里，而 int64 那格同样红。

`vt-pow` 那条是「偏差表自我固化」的实例：§17.10.7 早写「`b ** 2` clvt 给
`:int64`」，实现给 `:float64` —— 文档和代码**各自都自洽**，只是从没对齐。
偏差表一旦写成「记录在案，不是缺陷」，就很容易变成不再复验的化石。

`vt-rem` 最朴素：`vt-mod` / `vt-div` 都有那三行 bool → `:int8`，
`vt-rem` 没有。**同族函数里「只改了一个」是最常见的形态**，修的时候
要顺手 grep 一遍兄弟函数。

#### 四、验收

| 项 | 结果 |
|---|---|
| `bool-dtype-test` | **410** 断言全过（新增 K 组 22 条） |
| 全量套件 | 65 套件全绿（`--no-regen-probes` 严格模式） |
| `check_docs.py` | 220 个 §锚点全部可解析 |
| 编译体检 | 零 WARNING |

一条留给下次的判据：**审计某类型支持得对不对，第三遍要问「它说自己返回
什么，实际返回什么」**。前两遍都在验证「算得对不对」，而 dtype 是**契约**，
不是结果的副产品 —— 契约错的时候，值往往是对的（如 `b ** 2` 的 1/0），
所以永远抓不到。

---

## H.3.31 第三十八轮：第四遍只找「成对的不一致」—— 一个静默溢出，一个先剥虚部

> 对外结论见 `CHANGELOG.md`（三十八），契约见 `CONVENTIONS.md` §17.10.12。

#### 一、为什么第四遍不再逐函数测

前三遍各有一个被断言的对象（值 / 入口 / dtype），每次都靠「再造一批用例」
推进。到第四遍这条路的边际收益已经很薄：把 `vt-sum` 的 `:axis`、`:keepdims`
组合全测一遍，全部正确；把 42 个一元算子 × 8 种布局全比一遍，352 次全等。
继续加用例只是在证明「已经对的地方还对」。

于是换提问方式：**不问单个函数对不对，只问成对的东西一不一致**。这个
想法来自前三遍的一个统计规律——bool 的十六处缺陷里，没有一处是「唯一
实现写错了」，全部是「这里改了、那里没改」：

| 缺陷 | 被改的一侧 | 漏掉的一侧 |
|---|---|---|
| §17.10.1 `vt-map` 不真值化 | `vt-bit-not` 有 bool 分支 | `vt-map` 的写回 |
| §17.10.2 `vt-bit-not` | 逻辑算子 | 位算子 |
| §17.10.11 `vt-rem` | `vt-mod` / `vt-div` 有 `:int8` 分流 | `vt-rem` 没有 |
| §17.10.11 `vt-pow` | 整数白名单 | 白名单里没有 `:bool` |

所以第四遍的清单是「同一族的两个实现」和「同一语义的两个入口」，一共
六组。命中两组，四组确认一致。

#### 二、累积族：一个把计数写进布尔数组，三个把溢出藏成负数

`vt-cumsum` 有提升表，`nancumsum` 没有。差异不在 bool 这一行有多严重，
而在**同一条判据上两条内核给了不同答案**：

```lisp
;; %VT-CUMULATIVE（cumsum 走这条）
(final-dtype (or dtype (case (vt-dtype tensor)
                         ((:bool :int8 :uint8 :int16 :uint16 :int32) :int64)
                         (t (vt-dtype tensor)))))
;; %E3-CUMULATIVE-SKIP-NAN（nancumsum 走这条）
(final-dtype (or dtype (vt-dtype input)))
```

bool 那行的后果是**脏字节**：`(vt-nancumsum b)` 给 `:bool` 数组，里面躺着
`2`，`b=2`。这个用字节扫描一眼能看见。窄整型那三行更麻烦：
`(vt-nancumsum #(100 100) :dtype :int8)` 给 `[100 -56]` —— 累加器在
`(signed-byte 8)` 里回绕，输出仍是「合法的 int8」，没有任何信号。

`vt-cumsum` 的注释里其实早写过这件事：「v0.4.x 修复：numpy cumsum/cumprod
对精度低于平台整型的整型输入提升为 int64……原实现保持 int8 导致 100+100
溢出回绕为 -56」。也就是说同一个 bug 在 `cumsum` 上修过一次，`nancumsum`
是后加的，加的时候没回头看。修法因此不是打补丁，是把表抽成
`%CUMULATIVE-RESULT-DTYPE`，两条内核共用一份。

#### 三、真值判定：五个写入入口，四个错一个对

`vt-cast` 的开头是「先剥虚部，再判 `:bool`」：

```lisp
(when (and (complexp val) (not (vt-complex-dtype-p dtype)))
  (setf val (realpart val)))          ; #C(0 1) → 0
(when (eq dtype :bool)
  (return-from vt-cast (%bool-flag val)))     ; → False
```

而 `%bool-flag` 自己的 docstring 写着「复数：**任一分量非零**即为真」。
两句话都写在同一个文件里，隔了十行。

受影响的入口正好是「所有走 `vt-cast` 的值写入」：`vt-write` / `vt-put` /
`vt-fill` / `vt-pad :constant-values` 全给 0；唯一正确的是
`(setf (vt-ref …))`，因为它绕过 `vt-cast` 直接调 `%bool-flag`。这就是
「成对比较」的价值：只测 `vt-astype`（数组路径）会看到正确结果，
只测 `vt-write` 会以为「复数写布尔本来就该取实部」，只有把五个入口并排
才看得出是四个对不上一个。

修法是把 `:bool` 分支提到剥虚部之前——让「标量 → 布尔」服从
`%bool-flag` 这一个定义，而不是在 `vt-cast` 里再立一份。

#### 四、另外四组为什么「一致」值得记

确认一致的四组也写进了 §17.10.12，理由是**它们的正确性不是显然的**，
下次审计很容易重做一遍：

- **缩并族**：`trace`/`vdot`/`kron`/`matrix-power`/`multi-dot` 全部与
  numpy 一致（`trace` → `:int64`，其余 → `:bool`）。上一轮按「名字里有
  mat」找过一遍，这轮按「有没有乘积求和」补扫，没再捞到东西。
- **提升族**：`bool ⊗ {int8,int64,float32,float64}` 与
  `concatenate`/`stack`/`where` 的混合输入全对。这条尤其反直觉：
  `(vt-concatenate 0 b i64)` 必须给 `:int64`（否则 5/6/7/8 会被压成 1），
  实现是取「另一侧」而不是取「第一个」。
- **布局**：42 个算子 × 8 种布局 352 次全等。这是第二轮 `%bool-normalize!`
  里那条「非连续走逻辑位置」分支的直接验证。
- **序列化**：npy 头是 `|b1`，numpy 读回来就是 `bool`；`:lisp` 往返带 dtype。

#### 五、顺带更正一处旧结论

§17.10.11 写过「`np.percentile(bool)` 报错，clvt 报错是对的」。实测 clvt
**不报错**：`(vt-percentile b 50)` 给 `1.0d0`。当时只测了 `vt-ptp` /
`vt-gradient` 就顺手把 `percentile` 归进同一类，没验证——这是「偏差表
自我固化」的第三次实例（前两次是 `b ** 2` 的 dtype 和 `matmul` 家族）。

行为不改：numpy 抛错是因为它的分位数实现要算 `hi - lo`，本库在 float64
缓冲上取位，给的 1.0 在数学上是对的。**为了对齐实现细节去拒绝一个算得对的
答案，是负收益**。契约表已在 §17.10.12 改写清楚。

#### 六、验收

| 项 | 结果 |
|---|---|
| `bool-dtype-test` | **432** 断言全过（新增 L 组 22 条） |
| 全量套件 | 65 套件全绿（`--no-regen-probes` 严格模式） |
| `check_docs.py` | 221 个 §锚点全部可解析 |
| 编译体检 | 零 WARNING |

一条留给下次的判据：**当「再加一批用例」开始全部通过时，换的不是用例，
是被比较的对象**。第四遍没有测任何新函数，只是把已经测过的函数两两配对，
就找出两处——因为它们都不在「函数」这一层，而在「函数之间的关系」上。

---

## H.3.32 第三十九轮：第五遍不看「布尔怎么算」，看「布尔被谁读走」

> 对外结论见 `CHANGELOG.md`（三十九），契约见 `CONVENTIONS.md` §17.10.13。

#### 一、这一遍换的是提问对象，也是干活的方式

前四遍有一个共同点：都是我**挑**函数去测。挑的依据从「名字像不像算术」
到「有没有乘积求和」，再到「同族的两处是否一致」——方法在变，但「哪些函数
进入测试集」始终由我决定。第四遍末尾写过一句判据：*当「再加一批用例」
开始全部通过时，换的不是用例，是被比较的对象*。这一遍索性把「挑」这个
动作本身也交出去：**用程序遍历全部 375 个导出符号**，按参数个数试调

```
(f b)  (f b b)  (f b int64向量)  (f int64向量 b)
```

每个调用记录 dtype / 形状 / 值 / **物理字节**，再拿函数名去 numpy 里找同名
实现对照。1250 次调用，565 次正常返回，其余是参数语义不匹配（把张量传给
`vt-zeros` 之类）。

这不是「更全面地测一遍」，而是把抽样换成穷举——**抽样靠直觉，穷举靠机器，
两者的盲区不重叠**。六处缺陷里没有一处是前四遍「差一点就测到」的，它们
分布在动词的另一侧：不是「布尔参与运算」，而是「布尔被当成别的东西」。

#### 二、第一版扫描的假差异：夹具被就地写入污染

第一版跑出来的差异清单长得吓人：`vt-mul(b, i64)` 给 `(1 0 1 0)` 而
`vt-*(b, i64)` 给 `(1 0 1 1)`；`vt-vstack(b, i64)` 第二行是 `(1 1 1 0)`。
两个「等价」入口给出不同答案，看起来是本轮最大的鱼。

查了半小时才发现：扫描器对所有函数复用了同一对输入张量，而
`vt-random-shuffle`（在字母序里排在 `vt-mul` 之前）**就地打乱**了它。
`vt-sort` / `vt-put` / `vt-fill-diagonal` 同理。夹具被污染之后，后面每一行
都是错的——包括那些看起来「很有说服力」的差异。

改成每个用例重建输入，差异全部消失（`vt-mul` 与 `vt-*` 逐位一致）。
教训写进了 §17.10.13 的注解：**一个只读工具的扫描器里混进一个就地写入的
函数，污染的是之后的所有行**。这类错误的危险在于它不是随机噪声，而是
一批看起来自洽、能编出解释的假信号。

#### 三、六处缺陷：都在算术族之外

| 入口 | 现象 | 根因 |
|---|---|---|
| `vt-reciprocal` | `(T NIL T T)` → `(T T T T)` | `vt-int-dtype-p :bool` 为假 → 落浮点路径 → `1/0=+Inf` → `%bool-flag` 收成 1 |
| `vt-divmod` | 商是 `:bool` `(T T T T)` | `dtype` 缺省走提升规则，`floor(0/0)=NaN` 同样收成 1 |
| `vt-ptp` | `The value T is not of type NUMBER` | 非 axis 分支把 `(vt-item …)` 的 T 丢进 `-` |
| `vt-broadcast-shapes` / `-strides` | `Unhandled memory fault at #x0` | `(declare (list …))` + `(safety 0)` → 对结构体取 car 链 |
| `vt-apply-along-axis` | `(vt-sum row)` 的 3 变 1；返回值 `#(5 6)` 写进 bool 得字节 **5、6** | 结果 dtype 固定取输入 dtype，1D→1D 分支直接 `aref` |
| `vt-vander` | `:bool`（numpy `int64`） | dtype 直接继承输入，没有 bool 分支 |

前两条是同一个缺口的两个侧面。`vt-int-dtype-p` 只认四个有符号整数类型、
不认 `:bool`——这个判定本身是对的（布尔不是整数），但**需要「整数语义」的
地方拿它当分支条件**就错了：布尔要的是「按 0/1 参与整数运算，但结果不留在
bool 里」。§17.10.11 修 `vt-rem` 时用的正是这条，只是没回头检查同族还有谁
在用同一个判定（只有 `vt-reciprocal` 一处，但它在一个完全不同的文件里）。

第三条是「偏差表自我固化」的第三个实例。§17.10.12 的表格写着
「`ptp(b)` 报错」，我核对时看到确实报错就打了勾——**没有看报的是什么错**。
实际报的是 `T is not of type NUMBER`：同一函数的 `:axis` 分支走 `vt--`
（正确的布尔减法消息），非 axis 分支走 `-`。表格只记了「报不报」，
这一栏就一直在骗人。

第五条最值钱：`vt-apply-along-axis` 的结果 dtype 固定取输入 dtype。布尔
输入下 1D→scalar 把 3 压成 1（静默错值），1D→1D 直接 `aref` 写 5、6 ——
**布尔数组里出现越界字节**，破的是 §17.1 那条从第一轮守到现在的核心不变量。
同一个函数里两条分支犯两种错，因为一条经过 `vt-cast`（会真值化），
另一条不经过。修法不是给两条分支各打一个补丁，而是把结果 dtype 的来源
改对：布尔输入时由 FUNC 的返回值决定（numpy 也是这么做的），非布尔输入
维持旧契约——这样两条分支自动都对了。

#### 四、最大的一簇「差异」不是 bug

机械扫描最有价值的产出其实是一个**否定结论**：约 30 个超越函数（`sqrt` /
`log*` / `exp*` / 三角函数 / `spacing` / `deg2rad` / `atan2` / `hypot` /
`nextafter` …）在 numpy 里对 `bool` 输入一律给 `float16`，本库给 `float64`。

按「与 numpy 一致」的字面标准，这是 30 处偏差；按用户收益，`float16` 只有
3 位十进制有效数字，`sqrt(2)` 给 1.414。**不跟**。这条已和 §17.10.7 的偏差
表并列，理由是同一个：*对齐实现细节不如算对*。

同性质的三处更小：`square(b)` / `conj(b)` numpy 给 `int8`（而它自己的
`b*b` 给 bool，内部就不自洽）、`round(b)` numpy 给 `float16`、`clip(b,0,1)`
numpy 直接报错。本库都取「值正确且类型自洽」的一侧。

#### 五、确认无误的 23 个「布尔消费」入口

这一遍把「布尔当索引/掩码/计数」的入口全测了：`take` / `put` / `choose` /
`select` / `compress` / `extract` / `take-along-axis` / `put-along-axis` /
`searchsorted` / `bincount` / `digitize` / `unique` / `isin` / `in1d` /
`delete` / `setxor1d` / `setdiff1d` / `intersect1d` / `union1d` / `nonzero` /
`flatnonzero` / `argwhere` / `count-nonzero`，与 numpy 逐项一致。

值得一提的是 numpy 在这里有**两套并存的语义**：`take`/`put`/`choose` 把
布尔数组当 0/1 整数索引（`a.take([True,False])` → `[a[1],a[0]]`），
`compress`/`extract` 把它当掩码。本库原样保留了这个分裂——因为它是
numpy 的既定契约，改「更合理」的语义反而会让迁移代码静默出错。

写入路径的字节卫生也重新扫了一遍：`:out` 为 `:bool` 且结果 dtype 不符时
一律报错（§4.4 的严格性，早于 bool 就有）；`copy-into` / `copy-to!` /
`(setf (vt-slice …))` 的两个分支 / `vt-fill` / `vt-fill-diagonal` / `vt-put`
全部经 `%bool-flag`，`(setf (vt-slice b …) 2)` 得 1 而不是 2。

#### 六、验收

| 项 | 结果 |
|---|---|
| `bool-dtype-test` | **461** 断言全过（新增 M 组 29 条） |
| 全量套件 | 65 套件全绿（`--no-regen-probes` 严格模式） |
| `check_docs.py` | 222 个 §锚点全部可解析 |
| 编译体检 | 零 WARNING |

一条留给下次的判据：**当「挑函数来测」这件事本身变成瓶颈时，把挑的动作
也交出去**。这一遍六处缺陷没有一处来自「更仔细地看代码」，全部来自
「让程序去调用我根本想不起来要测的入口」。

---

## H.3.33 第六轮：换到「名词」一侧 —— 布尔是从哪造出来的、被当坐标用时是什么

> 对外结论见 `CHANGELOG.md`（四十），契约见 `CONVENTIONS.md` §17.10.14。

#### 一、这一遍的起点，是上一遍欠的一笔债

第五遍末尾我写过一句话：1250 次机械调用里 565 次正常返回，「其余是参数
语义不匹配（把张量传给 `vt-zeros` 之类）」。**这句是抽样判断，不是逐条
核实**——685 条报错我扫了一眼错误消息的形状就归了类。

这一遍先把这笔债还了：重跑扫描，并且**给每种形态配上 int64 / float64
基线**（上一版没有基线，导致「bool 挂了」根本无法区分是 bool 特有还是
调用本身不合法）。然后做三元对比：

- 「bool 挂 + 两个基线都通」→ 查出 7 个，全部是减法族（`vt--` /
  `vt-negative` / `vt-ptp` / `ediff1d` / `gradient` / `softmax` /
  `log-softmax`）的**刻意拒绝**，消息里就写着「numpy 同样拒绝」。这反而
  验证了前几轮的设计：布尔减法不该给一个「看起来像减法」的数。
- 「bool 通 + 基线报错」→ 0 个。
- 脏字节（dtype 是 BOOL 但物理字节 ∉ {0,1}）→ **0 个**，全形态 1296 次
  调用。前几轮守的那条不变量是稳的。

还完债的结论是：**这条路上已经没有东西了**。于是换到前五遍都没碰过的
一侧——前五遍问的都是动词（算、传、返、比、读），这一遍问名词：
**一个布尔张量是从哪来的，以及它被放在「位置」参数上时是什么含义。**

#### 二、探针自己撒了两次谎

第一次：`(vt-concatenate (list a b))` 报「至少需要一个张量，收到 0 个」。
我一度以为抓到了空张量拼接的 bug，写了一段分析。查签名才发现是
`(vt-concatenate axis &rest tensors)`——**axis 是第一个位置参数**，我按
numpy 的顺序把 list 传进去了。用正确签名重测，空 + 非空、空 + 空、
二维空行全部正常。

第二次：拿 Python 解析 Lisp 打印的 `(1 0 1 1)` 做 numpy 对照，得出
「bool×bool 一致 0 例」。这个数字一眼就是假的——不是「找到了 0 处差异」，
是「一例都没比对成功」。原因是 `rest` 的第一段是**形状**不是值，我索引
取错了一列。

两次都不是库的问题，是探针的问题。上一遍记的是「夹具被就地写入污染」，
这一遍是「签名猜错」和「解析取错列」。三次都指向同一件事：
**测量工具本身需要先被测量**。

#### 三、三处修复：两处同一个根因，一处是漏网

| 入口 | 修复前 | 现在 | numpy |
|---|---|---|---|
| `(vt-delete #(0 1 2 3) #(T NIL T T))` | `(2 3)` | `(1)` | `[1]` |
| `(vt-insert #(0 1 2 3) #(T NIL T T) 9)` | `(9 0 9 9 9 1 2 3)` | `(9 0 1 9 2 9 3)` | 同 |
| `(vt-geomspace 1 8 3 :dtype :bool)` | 报「不支持的 dtype」 | `(T T T)` | `[True True True]` |

前两条的根因值得单独说：`%check-integer-index` 只校验「是不是整数」，
而 `:bool` 的底层类型就是 `(unsigned-byte 8)`，于是**一路畅通地走了整数
索引路径**。危险不在结果错，而在静默——`np.delete(a, a > 3)` 是 numpy
里最常见的写法之一，移植过来不会报错，只会删掉一批不相干的元素。

第五遍的「23 个布尔消费入口确认一致」里就包含 `delete`。那份确认是用
**整数索引**测的，没测 bool 输入。这不是测漏了，是**测的时候没想到同一
个参数位置会因为 dtype 换语义**。

numpy 的分叉规则很清楚：描述「逐元素去留」的参数（`delete` / `insert` /
`place` / `putmask` / `compress` / `extract`）走掩码，必须与轴等长；
描述「要取哪些下标」的参数（`take` / `put` / `choose` / `bincount` /
`searchsorted` / `digitize`）走索引，长度任意。本库在后者全对，在前者
错了两个。

修法是按 dtype 显式分叉（`%BOOL-MASK-POSITIONS`），不是给 `:bool` 打
一个「也当索引」的补丁——后者会让两套语义永远糊在一起。同一份
`#(1 0 1 1)`，`:uint8` 仍然走索引（`(2 3)`），`:bool` 走掩码（`(1)`），
测试里专门钉了这两条并排的断言。

第三条是「三处按区间采样的构造函数里唯一拒绝 bool 的那个」：
`linspace` / `logspace` 一直支持（走 `vt-astype` 真值化），`geomspace`
的白名单漏了 `:bool`。同族三个入口两种行为，是 §17.10.12「成对不一致」
的又一例，只是对象换成了**创建**。

#### 四、这一遍确认无事的部分（写下来免得下次重做）

构造函数全族 19 个入口（`zeros`/`ones`/`full` 的四种值/`eye` 含 `:k`/
`identity`/`tri`/`triu`/`tril`/`diag`/`diagflat`/`from-array`/`empty`/
三个 `-like`/`repeat`/`tile`/`meshgrid`）dtype 与字节全对；`vt-full`
对 `2` 和 `0.5` 都真值化成 1（numpy 同）；`vt-empty` 给 0 而 numpy 给
未初始化内存（本库更安全，记作偏差）。

打印层四条路径（`princ`/`print`/`format ~a`/`~s`）都渲染 `T`/`NIL`，
二维对齐，0 维渲染单个 `T`/`NIL`。

极值尺寸：`(0,)`/`(3 0)`/`(0 0)` 空布尔在 `sum`/`reshape`/`concatenate`/
`vstack` 上正常，空归约报 numpy 同款错误，1 元素与 10⁶ 元素无异常。

`gcd`/`lcm` 对 bool 报「仅支持整型输入」，numpy 报 `UFuncTypeError`
（同样拒绝）；`random` 对 bool 报「仅支持浮点」（truncate 会静默全 0）。

#### 五、验收

| 项 | 结果 |
|---|---|
| `bool-dtype-test` | **491** 断言全过（新增 N 组 30 条） |
| 全量套件 | 65 套件全绿（`--no-regen-probes` 严格模式） |
| `check_docs.py` | 223 个 §锚点全部可解析 |
| 编译体检 | 零 WARNING |

一条留给下次的判据：**当「上一遍的否定结论」是抽样得出的，它就是一个
待还的债**。第五遍那句「其余是参数语义不匹配」看着无害，实际上把一个
685 条的样本空间整块划掉了。这一遍把它逐条核实完，只捞回三处——但
「核实完」这件事本身，才是这一遍真正的产出。

---

## H.3.34 第七轮：七遍里的第一个共同前提 —— 「所有调用都用默认参数」

> 对外结论见 `CHANGELOG.md`（四十一），契约见 `CONVENTIONS.md` §17.10.15。

#### 一、这一遍之前，六遍共享一个没人质疑的假设

回看六遍的探针，它们长得都不一样：第一遍是 numpy 差分，第二遍扫入口，
第三遍扫结果 dtype 与物理字节，第四遍做成对比较，第五遍用程序遍历
375 个导出符号，第六遍问构造函数与位置参数。

但把它们并排放，会看到一条共同的地基：**每一处调用，第二个参数之后都是空的**。
`(vt-sum b)`、`(vt-correlate a v)`、`(vt-delete a b)` —— 全是默认路径。

这不是疏忽，是「先测最常用的」。但七遍之后再回来看，最常用的那一层已经
被犁了六次，而 `:axis`、`:keepdims`、`:dtype`、`:ddof` 这四条分支一次都
没进去过。**默认参数不是「简单情形」，它是「另一条代码路径」**：归约器里
`axis=nil` 走全归约内核、`axis=0` 走列扫/行累加内核，两者是不同的函数。

于是这一遍的轴定成：用反射筛出 46 个接受 `axis` 的导出函数，对每个做
参数矩阵（7 种轴 × 2 种 keepdims × 4 种 dtype），共 303 组。

#### 二、第一条线索：两条写回路径，一条经过 cast、一条裸写

矩阵跑完先看脏字节（dtype 说是 BOOL，物理字节 ∉ {0,1}）。三条命中，
都是 `:dtype :bool`：

```
(vt-sum    b :axis 0 :dtype :bool)  =>  (2 1 1 2)
(vt-nansum b :axis 0 :dtype :bool)  =>  (2 1 1 2)
(vt-l1-norm b :axis 0 :dtype :bool) =>  (2 1 1 2)
```

同一个函数、同一个参数，`axis=nil` 时给 `1`（正确），`axis=0` 时给 `2`。
差别不在算子，在**选到了哪条写回路径**：

- 全归约 → `%KERNEL-SMALL-GENERAL` → `(setf (aref od i) (vt-cast acc dt))`，
  经过 `vt-cast`，`%bool-flag` 把 2 收成 1；
- 沿轴归约（窄整型）→ `%SMALL-INT-AXIS-FORM` 生成的 `%KERNEL-ROW-ACC`
  → `(setf (aref od i) acc)`，**裸写**，`od` 是 `(unsigned-byte 8)` 数组，
  2 原样落盘。

这就是「同一语义两条实现」的代价：一条走了类型转换，一条没走，而
**测试只测了默认路径的那一条**。

修法上我犹豫了一下：在十几个内核宏实例里各加一个 `if` 显然不行（要把
`res-lt` 是不是 bool 的判断塞进每个宏展开）；把 `dtype :bool` 直接禁掉
又太粗（numpy 允许 `np.sum(b, axis=0, dtype=bool)`）。最后选的是
**在归约出口统一收敛**：`%BOOL-CLAMP!` 按 shape/strides 只碰逻辑元素
（归约结果常是 `:out` 传进来的视图，不能整块扫缓冲），对已经过 `vt-cast`
的路径是幂等的，所以不必判断走了哪条内核。

#### 三、第二条线索：报错点错了人

矩阵里 39 条报错，我按消息分组看，发现两类「点错人」：

**第一类：把物理类型当 dtype 报。** `(vt-sum int64数组 :dtype :bool)` 报

```
unsupported output dtype (UNSIGNED-BYTE 8)
```

`(UNSIGNED-BYTE 8)` 是 Lisp 元素类型，用户手里写的是 `:dtype :bool`。
两边对不上，用户看到只会以为自己传错了 dtype —— 而真正的原因（本库不做
「整数归约到布尔」）一个字都没说。

**第二类：报错来自用户从没调用的函数。**

| 用户调用 | 报的是 |
|---|---|
| `(vt-var b :dtype :bool)` | `vt--: 布尔张量不支持减法/取负` |
| `(vt-std b :dtype :bool)` | 同上 |
| `(vt-norm b :dtype :bool)` | `vt-sum: 不支持「:float64 输入 → :bool 结果」` |
| `(vt-standardize b :dtype :bool)` | `vt--: 布尔张量不支持减法/取负` |

四条都是「参数没在最外层校验，一路走到内部才炸」。`vt-var` 要先算
`(x - mean)`，于是 `vt--` 先炸；`vt-norm` 要先 `square` 再 `sum`，于是
`vt-sum` 先炸。**错误消息里出现的那层函数，恰好是用户没有调用的那层。**

这不算「值错」，但它让定位成本从「看一眼」变成「读一遍调用链」。修法是
在每个公开入口前置一条 `(when (eq dtype :bool) (error "vt-xxx: ..."))`。

#### 四、第三条线索：`vt-nanstd` 答应了 `vt-std` 拒绝的事

补前置校验时发现一处**成对不一致**（第四遍的轴，换个场景又出现）：

```
(vt-std    b :dtype :bool)  =>  报错
(vt-nanstd b :dtype :bool)  =>  :bool (0 1 1 0)
```

同一份数据、同一个参数，一个报错一个静默。`nanstd` 的 `(0 1 1 0)` 读起来
还挺像答案 —— 它是「标准差是否为零」的真值化，但用户要的是 0.0/0.5/0.5/0.0。
numpy 两个都抛 `TypeError`。这条和 `vt-nanvar` 一起补上。

有意思的是 `vt-std` 那份「拒绝」也不完全是我这一轮加的 —— 它是本来就拒绝的，
只是拒绝得**理由不对**（通过 `vt--` 拒绝）。所以这一轮实际做的是：
把「偶然的拒绝」变成「有意的拒绝」，再让同族的另一个跟上。

#### 五、顺带修的非布尔缺陷：一次被 docstring 骗过去的核对

用户这一轮说「发现的问题都解决掉，不一定非得是 bool」。我先去翻前几轮
探查里留下的可疑点，其中一条是 `vt-correlate` 的 `:mode :valid`。

docstring 里明明白白写着：

> 三种模式本身与 numpy 逐值一致（:full/:valid/:same 均已对齐）

我前几轮核对时看到这句就放过了。这一遍随手跑了一组 ——

```
(vt-correlate #(1 2) #(1 0 1 0 1 0 1) :mode :valid)  =>  shape (0)
```

空张量。numpy 给 `(2 1 2 1 2 1)`。

顺着查下去，两种模式各有一处错：

| 模式 | 原来的写法 | 错在哪 |
|---|---|---|
| `:valid` | `(max 0 (1+ (- n m)))` | n<m 时是负数，被 clamp 成 0 |
| `:same` | `(floor (- full-len (max n m)) 2)` | `|n-m|` 为奇数时取整方向与 numpy 相反 |

为了定 `:same` 的公式，我拿 1..60 的两两组合跑了 400 组随机，反推出
`start = min((m-1)/2, n/2)`。然后发现 `np.convolve` 不服从这个公式 ——
它用 `(n-1)/2`。**numpy 的两个入口自己就不一致**：偶数 n + 长核时
`np.convolve(a,v,'same')` 与 `np.correlate(a, v[::-1], 'same')` 差一个位置。

所以 `vt-convolve` 不能简单委托 `vt-correlate`，得从内部传一个
`same-anchor` 区分。复数路径 `%CX-CORRELATE` 是同一份逻辑的另一份拷贝，
同样的 bug，同样修。

修完实数 3750 组、复数 2400 组，与 numpy 逐值零失配。

这是「偏差表自我固化」的**第四个实例**（前三个：`b ** 2`、`matmul`、
`percentile`）。四次都是同一个形状：**文档里写了一句「已验证」，下次就不再
验了**。区别是这次它藏在 docstring 里，比藏在小节表格里更难发现 —— 谁会
去怀疑一句「均已对齐」呢。

#### 六、也确认了一批「无事」的

- 轴矩阵的**值**层面 82 项全对（脏字节是 dtype 层的问题，值本身没错）；
- `argmax`/`argmin` 接受元组轴（numpy 直接 `TypeError`），给归约区内的
  线性下标，语义自洽，保留为更宽松；
- `mean`/`nanmean` 的 `:dtype :bool` 与 numpy 同值。

#### 七、验收

| 项 | 结果 |
|---|---|
| `bool-dtype-test` | **524** 断言全过（新增 O 组 33 条） |
| 全量套件 | 65 套件全绿（`--no-regen-probes` 严格模式） |
| `check_docs.py` | 224 个 §锚点全部可解析 |
| 编译体检 | 无新增 WARNING |

一条留给下次的判据：**「默认参数」不是简单情形，是另一条代码路径**。
每一轮的探针都该问一句「我这次传了几个参数」，如果和上一轮一样，那这轮
大概率在重复上一轮。

---

## H.3.35 第八轮：把「我要 bool」当成断言，而不是请求

> 对外结论见 `CHANGELOG.md`（四十二），契约见 `CONVENTIONS.md` §17.10.16。

#### 一、这一遍的轴换得比前七遍都彻底：不再问「给不给」，改问「等不等于」

前七遍的每一处探针，形式上都是「我传一个布尔，看它怎么算/怎么报错/返回什么
dtype」。这条线走到第七遍已经很细：默认参数、非默认参数、成对一致、消费者、
构造、位置参数 —— 都犁过。

第八遍一开始也顺着惯性找了两个轴（别名/共享缓冲、输入改写），都干净：

- `:out` 指向输入自己、指向输入的反向视图、指向输入的切片，`vt-add`/`vt-mul`
  × 三种 dtype 全对；
- 60+ 函数 × 3 dtype × 1 维/2 维扫「谁偷偷改了输入」，0 处；
- 同一逻辑数组换六种物理布局（连续/切片/转置/翻转/混合），一元/二元/归约
  结果全一致。

三条都干净，说明这块地被前面几轮顺带压过了。于是换了问法：

> 我一直把 `:dtype :bool` 当**请求**（「我要布尔」）。它能不能当**断言**？
> 即：`(f x … :dtype :bool)` 的结果，是不是恰好等于 `(vt-astype (f x …) :bool)`？

这个问题之所以成立，是因为库自己早就把两者定义成同一件事：`vt-map` 的
`(lambda (x) 5) :dtype :bool` 返回 `1`，`vt-linspace 0 1 5 :dtype :bool` 返回
`(0 1 1 1 1)`，`bool-dtype-test` 的 G/N 两组就在测它。既然「先算再 astype」
是文档语义，那「直接要 bool」就必须逐值相等 —— 不等就是 bug，而且不需要
跟 numpy 对照，**用库自己的另一条路径当判据**。

#### 二、144 个样本，37 个不等

`sb-introspect` 筛出 139 个接受 `:dtype` 的导出函数，各试 1/2/3 个布尔实参，
得到 144 个可比对样本（自然结果与 bool 结果都拿得到）。37 个不等，落在 35 个
函数上，全部是浮点算子：`abs` / `sin` / `cos` / `tanh` / `log*` / `exp` /
`acos*` / `asinh` / `atan` / `conj` / `square` / `nancumsum` / `negative` …

看第一个就明白了：

```
(vt-abs #(0.5 1.5 2.0 0.25 3.0 0.75) :dtype :bool)
  修复前  字节 0 1 2 0 3 0     → 真值 0 1 1 0 1 0
  现在    字节 1 1 1 1 1 1
```

同一个表达式里有两种错：`0.5 → 0` 是**真值反了**（该 True 给了 False），
`2.0 → 2` 是**脏字节**（破 §17.1）。两种错都来自同一句
`(the (unsigned-byte 8) v)` —— `the` 对浮点做的是截断，不是真值化。

有意思的是这个坑的形状：`vt-fast-map` 有 `dtype-check`（输入 dtype = 结果
dtype 就走内联）和 `maskable`（`:bool`/`:int8`/… 走掩码内核）两条快路径，
两条都按元素类型特化，都只做 `the` 截断；而慢路径 `vt-map` 经
`vt-cast-fun` → `%bool-flag`，是真值化。**同语义两条实现，默认路径是错的
那条** —— 这已经是本项目第七次遇到同一个形状（§17.10.12 的成对、§17.10.15
的 `%KERNEL-ROW-ACC` 裸写都是它）。

修法选择了「合并」而不是「打补丁」：把 `:bool` 从两条快路径里摘出去
（`dtype-check` 加一个 `(not (eq (vt-dtype res) :bool))`、`maskable` 去掉
`:bool`），布尔结果一律落 `vt-map`。这样只剩一条实现，永远不会再分叉。
代价是布尔输出的映射少一次内联 —— 布尔 elementwise 本来就是冷路径，而
`fastmap-invariant-test` 的计时上界（float64 路径）没有变化。

#### 三、修完还剩 5 个，暴露了第二层

合并快慢路径之后 37 → 5。剩下的是 `acos` / `acosh` / `log` / `log10` / `log2`
的实数入参。它们不走 `vt-fast-map` 的布尔分支了，为什么还错？

因为 `%float-map` 的常规分支是

```lisp
(vt-fast-map op (%coerce-float-input IN %DT) :dtype %DT)
```

—— 它先把**输入**降到目标 dtype 再算。目标恰好是 `:bool` 时，这一步把 0.5
变成 1、-0.5 变成 1，于是算的是 `log(0/1)` 而不是 `log(x)`，结果全 0。

这一处的修法不能是「也合并」，因为它是**精度契约**的实现：整数输入必须先
提升到浮点再算（P1-1）。只能加一条专用分支 `%FLOAT-MAP-BOOL`：按自然浮点
精度算完，再 `%to-bool`。

写这条分支时踩了一个自己的坑：`fastmap-invariant-test` 是**静态**检查，
它 `(macroexpand-1 '(%float-map …))` 之后按固定形状 `(let* … (if <FAST> <SLOW>))`
取 `(third (third form))` 当快分支。我第一版把布尔分流包在**外层** if 上，
测试立刻报 9 个「SIN 快路径为 vt-fast-map（非 vt-map）」—— 它看到的
「快分支」变成了 `%FLOAT-MAP-BOOL`。这不是测试太脆，而是它在替我守
「别把快路径改没了」；正确的做法是保持形状、把分流挂进 `<SLOW>`。

#### 四、累积族的第二条实现：`%cast-to` vs `vt-cast`

`(vt-cumsum b :dtype :bool)` 一直是对的（`(1 1 1 1 1 1)`），
`(vt-nancumsum x :dtype :bool)` 却是 `(0 0 0)`。差别不在算子，在写回：

- `%cumulative`（cumsum/cumprod）写 `(vt-cast cum final-dtype)`；
- `%e3-cumulative-skip-nan`（nancumsum/nancumprod）的展平路径把结果先算进
  一个 double vector，再 `%e3-store-double-vector` 按元素类型 8 路写回，
  `(unsigned-byte 8)` 那一支是 `(%cast-to lt v)` —— 截断。

又是 `(unsigned-byte 8)` 同时承载 `:uint8` 与 `:bool`、而两者写回语义相反
（uint8 要截断、bool 要真值化）这个根因，第三次出现在本轮里。修法是给
`%e3-store-double-vector` 单列一支真值化写回。

#### 五、剩下两处是「检查写歪了」和「签名吞了关键字」

`vt-negative` 的布尔拒绝写成 `(and (null dtype) …)`。本意是给
`:dtype :int8` 留出口（docstring 里明说了「若确实要数值取负，请显式给出
`:dtype :int8`」），但它顺手把 `:dtype :bool` 也放行了：

```
(vt-negative b :dtype :bool)  =>  dtype 说 BOOL，字节 255
```

`-1` 写进 `(unsigned-byte 8)` 就是 255，比不加 `:dtype` 的调用还糟。条件改成
`(or (null dtype) (eq dtype :bool))` 即可 —— 判据是「**结果** dtype 是不是
布尔」，而不是「用户有没有显式说话」。

最后一处跟布尔无关，但是同一个入口上撞见的：

```
(vt-clip a :dtype :bool)
  =>  :DTYPE fell through ETYPECASE expression. Wanted one of (VT NUMBER SEQUENCE).
```

`vt-clip` 签名是 `(vt &optional min-val max-val &key out dtype)`。CL 的
`&OPTIONAL` 不认关键字，于是 `:dtype` 成了 MIN-VAL 的**值**、`:bool` 成了
MAX-VAL 的值，一路走到 `%op-clip` 的 etypecase。用户看到的错误里没有
「你少写了两个 nil」这条信息。

修法：MIN-VAL 只可能是数值/nil，出现关键字就一定整体是 `&KEY`，原样交回
`&KEY` 位置重新分派（`vt-view` 处理 `(vt-view a :dtype dt)` 用的是同一条
思路）。`vt-clamp` 转调 `vt-clip`，一并好了；「只给下限」的 numpy 报错语义
没动。同类签名还有 `vt-where`，但 numpy 的 `where` 本就不允许单参数形式配
`dtype`，故只记录不改。

#### 六、这一遍留下的判据

**「同语义两条实现」可以反过来当测试工具用。** 这一轮的判据不是 numpy，
而是库自己的另一条路径：`(f x :dtype :bool)` vs `(vt-astype (f x) :bool)`。
前七遍一直拿 numpy 当唯一外部标尺，但库内只要存在两条实现同一件事的路径，
它们互为标尺 —— 而且这种标尺比 numpy 更严：numpy 对 `np.exp(b, dtype=bool)`
直接 TypeError（没有 bool 循环），根本没机会暴露「截断 vs 真值化」的差别。

第二个判据是**覆盖率必须打印**。这一轮前三个探针都报「0 处失败」，
但那是 106/134 —— 剩下 28 次是调用直接报错，被静默归入「没问题」。
后来每次扫描都把「成功/报错」计数打出来，才看清哪些函数是真的没测到。

| 项 | 结果 |
|---|---|
| 探针 | 139 个接受 `:dtype` 的导出函数 × 1/2/3 实参 → 144 样本 |
| 修复 | 37 处不等 → 0 |
| 断言 | bool-dtype-test 549（新增 P 组 25 条） |
| 套件 | 65/65（278s） |
| 编译体检 | 无新增 WARNING（`fastmap-invariant-test` 24/24） |

---

## H.3.36 第九轮：去审那把尺子

> 对外结论见 `CHANGELOG.md`（四十三），契约见 `CONVENTIONS.md` §17.10.17。

#### 一、这一轮的起点是上一轮的判据

第八遍我拿 `(vt-astype (f x) :bool)` 当尺子量了 144 次，结论是「37 处不等 → 0」。
可那把尺子本身从没被量过：如果 `vt-astype → :bool` 在某个 dtype 或某个特殊
值上就是错的，那 144 次「相等」只证明了两条路径**一起错**。

所以先审尺子，再顺着「dtype 提升 / 转换」这条线往下走：

| 探针 | 规模 | 结果 |
|---|---|---|
| 值 → 布尔：11 dtype × 特殊值 × 三条入口 | 55 组 × 5 判据 | 全对 |
| bool 往返：`:bool → D → :bool` | 11 | 全恒等 |
| 二元算子 × 混型 | 11×11×17 = 2021 | (dtype, 值) 全对 |
| 提升表 | 二元 121 + 三元 286 | 全对 |
| 三条转换入口 `vt-cast`/`vt-astype`/`vt-map` | 605 | 全一致 |
| 合并/选择/三元算术 | 847 | 先 8 处不符 → 修后 0 |

尺子是准的。矩阵也是干净的 —— 说明这块地被前八轮顺带压过了。真正出问题的是
**「同一语义有几条实现」**：五处缺陷，无一例外都是两条路径给出不同答案。

#### 二、`vt-/` 见到整型就改判 dtype

```
(vt-/ #(1+2i 4+0i) int8数组)
  → TYPE-ERROR: The value #C(0.5d0 1.0d0) is not of type REAL
```

`vt-/` 按 true_divide 的「除法恒浮点」规则，只要有一个整型**张量**参与，就把
结果 dtype 改写成 `:float64`；`(complex128, int8)` 命中这一支，虚部于是被写进
实数数组。**24 个 (复数, 整型) 组合，两个方向全崩**，而 `complex ⊔ float`、
`complex ⊔ complex`、`complex ⊔ 标量整数` 都正常。

有意思的是这一支的注释写着「int8/int16/uint8/uint16 与 float32 混合 → float32」——
它是照 numpy 的 promote 规则写的，只是**没考虑复数已经在 `vt-promote-type`
里被正确提升过**，再改一次就把它改没了。

#### 三、`vt-map` 的写回不认复数

```lisp
(the double-float (coerce v 'double-float))   ; v = #C(1.0d0 2.0d0)
```

`(vt-map #'identity 复数张量 :dtype :float64)` 崩，`(vt-conj c :dtype :float64)`
一并崩 —— 而同一件事的另两个入口 `(vt-astype c :float64)`、`(vt-cast #C(1 2)
:float64)` 都正常取实部。§16.5 早就写明「复数 → 实数取实部」，`vt-cast-fun` 的
`:float64` 分支甚至专门写了 `(if (complexp val) (realpart val) val)` —— 只是
`%vt-map-run` 没走 `vt-cast-fun`。

修法上犹豫过：直接在写回处加 `(if (complexp v) …)` 最省事，但那个 `dotimes`
是全库最热的循环（2M 元素级），每元素多一次类型判断会摊到所有实数路径上。
最后照第八遍 `%bool-wrap` 的样子加了 `%REAL-WRAP`，只在「目标实数 + 有复数输入」
时把函数包一层，热路径一字未动。

#### 四、标量入口不认 Lisp 布尔（一串）

```lisp
(vt-cast t :float64)                            ; The value T is not of type REAL
(vt-fill (vt-zeros '(2) :dtype :float64) t)
(setf (vt-ref f 0) t)
(vt-full '(3) t :dtype :float64)
(vt-* f t)
```

`%bool-flag` 认 `T`/`NIL`，但只在**目标**是 `:bool` 时被调用。数值目标是另一条
`coerce` 的路，于是数组入口 `(vt-astype bool张量 :float64)` 给 `1.0`、标量入口
直接崩。这是本轮第四次看到同一个形状：**同一个值，两条入口，两种答案**。

顺手把 `(setf (vt-ref float数组 0) #c(1 2))` 也从「报错」改成「取实部」。
越界整数写入**故意保留报错** —— numpy 2.x 的 `a[0] = 300` 也是
`OverflowError`，而 `astype` 走回绕（`np.array([300]).astype(int8)` → `44`）。
两边都是 numpy 自己的行为，不该「统一」掉。

#### 五、`(type real x)` 那个声明

`vt-from-sequence` 的一维快路径把转换算子内联，并对元素声明 `(type real x)`：

```
(vt-from-sequence '(t t) :dtype :float64)
  → The value T is not of type REAL when binding X
```

原代码只让 bool 模式绕开这条快路径（布尔叶子全是 `T`/`NIL`），漏了「目标
dtype 是数值、序列里却写了 `T`」这一组合。判据扩成 `%FILL1D-INELIGIBLE-P`
（复数**或** `T`），仍是一次扫描 —— 原来那个「含复数」的判据本来就要扫一遍。

#### 六、三元的提升：折叠 vs 一次看全部

`%op-final-dtype` 对多个张量是逐对折叠：

```lisp
(setf base (if base (vt-promote-type base d) d))
```

二元时与「一次看全部」等价，三元起分叉：

```
int16 ⊔ uint16 = int32 → int32 超 24 位 → float64
而 numpy / vt-promote-type / vt-concatenate 都给 float32
```

也就是说 `(vt-+ a b c)` 的结果 dtype 与库自己的提升表都对不上 ——
`vt-concatenate` 用的正是 `(apply #'vt-promote-type …)`。改成同样的一次提升。

这条之所以能藏这么久：`float64` 参与时两种算法恒等，只有「有符号 + 无符号 +
窄浮点」这种组合才露馅。**矩阵要打到边界值**，这是这一轮学到的第二条。

#### 七、留下的判据

> 判据不能只问「这个函数对不对」，要问「同一件事有几个入口」。
> 入口数得靠枚举（`vt-cast` / `vt-astype` / `vt-map`），规则实现数也得靠枚举
> （`%op-final-dtype` / `vt-promote-type` / `vt-concatenate` / `vt-stack`），
> 然后逐个对。这一轮的 5 处缺陷全部落在这张网里。

| 项 | 结果 |
|---|---|
| 混型算子矩阵 | 2021 组，(dtype, 值) 与 numpy 零失配（报错 73 处均为两边共同拒绝） |
| 提升表 | 二元 121 格 + 三元 286 组，零失配 |
| 三条转换入口 | 605 组，零分歧 |
| 合并/选择/三元 | 847 组，修前 8 处不符，修后 0 |
| 修复 | 5 处（复数真除法 24 例、复数→实数写回 60 例、布尔标量入口 8 类、T 叶子构造、三元提升 8 组） |
| 断言 | bool-dtype-test 575（新增 Q 组 26 条） |
| 套件 | 65/65（275s） |
