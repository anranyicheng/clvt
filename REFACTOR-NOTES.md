# clvt v0.3.0 重构说明（对照《高性能张量库设计约定》）

本次重构将既有实现逐条对照设计约定审查，修正偏差、固化契约、修正误导性文档。
以下为"约定条目 → 代码落点 → 状态"的完整映射。

## 一、三层分离

| 约定 | 代码落点 | 状态 |
|------|----------|------|
| 逻辑层：shape/dtype/广播/归约语义 | dtype.lisp（类型提升单一事实来源）、各 API 形状推导 | 已符合，文档化于 package.lisp 架构总纲 |
| 物理层：strides/offset/连续性/别名 | core.lisp（vt 结构、stride-0 广播、连续判定、区间重叠检测） | 已符合，vt 结构文档补全 |
| 执行层：快/慢路径、SIMD、并行 | map-reduce.lisp、simd-matmul.lisp | 已符合，模块头写入选路规则 |
| 优化路径与通用路径逐位等价 | vt-fast-map 三路分派：dtype 匹配+连续 → 内联；dtype 匹配+非连续/广播 → strided 宏；其余 → vt-map | 已符合，文档显式声明 |
| 连续性只影响性能不影响正确性 | strided 宏支持任意 strides | **文档修复**（本次最重要修改，见下） |

### 关键修复：%inline*-strided 文档

`%inline1-strided` / `%inline2-strided` 的实现从一开始就按结果张量自身
strides 遍历写入（`rs` 向量），天然支持非连续 RES 与广播输入；但文档错误
标注"RES 必须连续"。这与事实相反——若按错误文档把非连续 RES 改走
"先 contiguous 再写"的路线，会平添一次全量拷贝并误导后续优化。本次已重写
两个宏的文档字符串，并在 `%inline3-strided` 补充同义注释，
同时把"两条路径输出必须逐位相同"写入 map-reduce.lisp 模块头。

## 二、内存模型（Strided View）

| 约定 | 代码落点 | 状态 |
|------|----------|------|
| 张量 = data/shape/stride/dtype(+offset) 视图 | core.lisp vt 结构 | 已符合 |
| transpose/permute 零拷贝 | manip.lisp vt-transpose | 已符合 |
| reshape 视图优先否则拷贝；view 要求连续否则报错 | manip.lisp vt-reshape/vt-view | 已符合 |
| 基础切片零拷贝；高级索引拷贝 | indexing.lisp | 已符合 |
| flatten 总拷贝；ravel 视图优先 | manip.lisp vt-flatten/vt-ravel | 已符合 |
| contiguous() 已连续返回自身 | core.lisp | 已符合 |
| 广播 = stride-0 虚拟重复读 | core.lisp vt-broadcast-strides | 已符合 |

## 三、广播规则

| 约定 | 代码落点 | 状态 |
|------|----------|------|
| 右对齐、左补 1、逐维取大 | core.lisp vt-broadcast-shapes | 已符合 |
| 原地不允许广播改变形状 | core.lisp vt-copy-into 形状校验 | 已符合 |
| 广播视图（dim>1 且 stride=0）语义只读 | vt-fill / vt-copy-into / def-vt-reduce :out 校验 | 已符合 |
| 广播性能（循环重排/分块） | strided 宏按 res strides 顺序遍历，广播维 stride=0 不产生额外访存抖动 | 已符合 |

## 四、类型系统

| 约定 | 代码落点 | 状态 |
|------|----------|------|
| 提升层次 integral < floating | dtype.lisp vt-promote-dtype | 已符合 |
| 浮点不降级；int 位宽决定浮点精度（int32+float32→float64） | dtype.lisp | 已符合 |
| 整数溢出回绕 | dtype.lisp %wrap-int* | 已符合 |
| 浮点转整数截断；NaN/Inf → 0 | nan.lisp %safe-truncate、%coerce-int* | 已符合 |

## 五、NaN/Inf 语义

| 约定 | 代码落点 | 状态 |
|------|----------|------|
| 算术传播、比较恒假（陷阱屏蔽下 IEEE 语义） | util.lisp with-float-safe、elementwise.lisp | 已符合 |
| maximum/minimum：NaN 传播；fmax/fmin：忽略 NaN | elementwise.lisp | 已符合 |
| sum/prod 传播；nansum/nanprod 跳过 | reduce-stats.lisp %op-step | 已符合 |
| amax/amin 首个 NaN 胜出；argmax 遇 NaN 即返回 | reduce-stats.lisp %op-step/%op-arg-step | 已符合 |
| nanmax/nanmin 全 NaN 报错；nanarg* 全 NaN 报错 | reduce-stats.lisp 内核 | 已符合 |
| 排序 NaN 稳定排末尾 | util.lisp vt-numpy-sort | 已符合 |
| unique NaN 视为相等（集合语义） | setops.lisp vt-float-nan-inf-= | 已符合 |
| softmax 减最大值稳定化；全 -Inf 行 → NaN（对标 PyTorch） | nn.lisp vt-softmax | 已符合，本次补充文档 |

## 六、归约语义

| 约定 | 代码落点 | 状态 |
|------|----------|------|
| axis=nil 全局 / 单轴 / 多轴 / keepdims | vt-normalize-axes + def-vt-reduce | 已符合 |
| int32 累加器提升 int64；float32 保持 float32 | %op-acc-lt / %op-out-dtype | 已符合 |
| 空归约：有单位元返回单位元 | def-vt-reduce 空分支 | **语义修正**（原 sum/prod/all/any 正确） |
| 空归约：max/min 族返回 NaN（int 结果提升 float64） | def-vt-reduce 空分支 | **语义修正**（原返回 ±Inf 哨兵） |
| 空归约：arg 族报错 | def-vt-reduce 空分支 | **语义修正**（原 argmax/argmin 填 0） |

## 七、内存布局与别名

| 约定 | 代码落点 | 状态 |
|------|----------|------|
| 行主序默认 | vt-compute-strides | 已符合 |
| out 与输入重叠 → 先快照输入 | vt-map / vt-copy-into / einsum-execute 的 %vt-views-overlap-p 分支 | 已符合 |
| 重叠检测：dim>1 区间扩展 (dim-1)*stride，广播维不扩展 | core.lisp %vt-view-span | 已符合 |
| 广播视图与自重叠视图写入报错/只读 | vt-copy-into | 已符合 |

## 八、API 设计约定

| 约定 | 代码落点 | 状态 |
|------|----------|------|
| axis/keepdims/dtype/out 命名一致 | 全库 | 已符合 |
| out 硬契约：形状/可写/dtype 冲突必报错 | def-vt-reduce、einsum（本次 assert→error）、vt-fast-map | **einsum 修正**，其余已符合 |
| out 软门控：连续性只影响选路 | 各内核非连续 out 路径（out-contig-tests 回归覆盖） | 已符合 |
| 创建函数默认 float64 | creation.lisp | 已符合 |
| empty 未初始化（退化为 zeros） | creation.lisp vt-empty | 已符合 |

## 九、性能优化

| 约定 | 代码落点 | 状态 |
|------|----------|------|
| 快路径/通用路径双路径 + 运行时选路 | vt-fast-map 分派树 | 已符合 |
| SIMD 仅在 dtype+连续满足时启用，否则回退 | simd-matmul.lisp dispatch → NIL 回退 einsum | 已符合 |
| 并行 GEMM 工作量阈值 | *matmul-parallel-threshold* = 5e6 | 已符合 |
| 热路径宏内联消除 funcall 装箱 | %inline*-loop / %inline*-strided | 已符合 |
| 缓存友好遍历（末轴内层） | 内核按行主序末轴连续遍历；转置场景走 strided 路径保正确性 | 已符合 |
| einsum 路由四分类 | einsum-execute：逐元素→vt-map；全收缩→专用累加；BMM→NN/NT 分块 GEMM；其余→通用循环 | 已符合 |

## 验证方式

本次重构环境无 SBCL，采用静态验证：
1. Lisp 感知括号平衡检查（跳过字符串/注释/字符字面量）——全部 22 个源文件通过；
2. 全部修改以 patch 脚本落盘（scripts/apply_refactor.py、refine_empty_reduce.py），
   每处修改要求唯一匹配，失败即中止；
3. diff 审查全部改动；
4. 确认测试套件无依赖旧空归约行为的用例。

建议在 SBCL 环境执行 `bash test/run-tests.sh` 做最终回归确认。


---

# 第二轮重构（v0.3.1）—— 数据类型统一 / NaN·Inf 补全 / out 契约加固

上游已合入第一轮补丁（66a090f、95baf94）。本轮针对三个遗留问题域：

## 一、数据类型混乱 → 统一走 dtype.lisp 单一事实来源

| 问题 | 修复 | 原则 |
|------|------|------|
| vt-cast-fun 对 int8/uint8 走 %wrap-* 慢路径，int16/int32 走 %coerce-* 快路径，风格分裂 | 统一为 %coerce-*（语义一致，含 typep 快速路径） | §8.5 |
| vt-arange int64/int32 溢出直接存储（会触发 SBCL 类型错误），int16/int8/uint 却回绕 | int64/int32 统一 %wrap-* 回绕 | §8.5 回绕 |
| vt-linspace float32 用 float32 累加产生漂移 | 以 double 计算、存储时舍入（对标 NumPy） | 一、路径等价 |
| vt-bits->unsigned-dtype 返回 :uint32/:uint64（非存储类型） | 文档标注保留符号状态 | 二、单一事实来源 |

## 二、NaN/Inf 处理补全

按 §5.2 分层语义逐项核对：传播层（算术/maximum/minimum/median）、
跳过层（nan-* 族）、错误层（空 arg 归约、全 NaN 片段）。本轮修复
mod/rem 零除、even-p/odd-p 非有限值、hypot 混合特殊值、随机数边界校验四处。

## 三、:out 连续/非连续内存处理

逐一核对了 117 个接受 :out 的函数的写入路径：
- 已正确（strides 写入/契约拒绝）：vt-map、vt-fast-map（含 %inline*-strided）、
  def-vt-reduce 家族、einsum、vt-where、vt-take（内部 out）、vt-pad、
  vt-triu/vt-tril（fresh copy + strided 写）、vt-sigmoid/vt-relu（连续守卫 + 回退）
- 本轮修复：vt-det 的 :out 形状硬契约（此前任意形状被静默填满）

## 验证方式

环境无 SBCL，采用静态验证：
1. Lisp 感知括号平衡检查全部通过（22 源文件 + asd）；
2. 全部修改经唯一匹配 patch 脚本落盘（scripts/apply_round2.py + apply_completion.py）；
3. 确认测试套件无依赖旧行为的用例（random-uniform 的 low<high assert 除外——
   现有测试若使用 low=high 用例将获得新的合法行为）。

建议在 SBCL 环境执行 `bash test/run-tests.sh` 做最终回归确认。

---

# v0.3.2 第三轮重构说明（SBCL 2.6.8 实测回归）

前两轮的静态验证遗留了三类问题：执行层与逻辑层的 dtype 契约缺口、
静默错误行为、测试基础设施缺陷。本轮在真实环境（SBCL 2.6.8 + Quicklisp +
sb-simd）下首次全量加载与回归，逐项修复。详细修复清单见 CHANGELOG.md
v0.3.2 条目，此处只记录与三层分离约定的对照结论：

| 约定 | 实测结论 | 本轮落点 |
|------|----------|----------|
| 逻辑层声明的 dtype 必须可用（单一事实来源） | 违反：reduce 家族内核只实现 4 种存储级类型，小整型直接抛错 | 新增 `%kernel-small-general` 通用内核 + dtype 映射/累加/单位元补全 |
| 执行层选路只影响性能不影响正确性 | 落实：小整型统一路由路径 3 通用内核，路径 1/2 加存储级守卫 | reduce-stats.lisp `def-vt-reduce` |
| 类型特化与通用路径输出一致 | 新旧路径并存，test-bug0 新增 9 项回归（含转置视图/axis/:out）锁定 | test/test-bug0.lisp |
| 错误层：静默错误必须显式报错 | 违反：`vt-random` 整型 dtype 恒返回全 0（截断陷阱） | random.lisp 显式报错并指引替代 API |
| einsum dtype 边界 | 确认为已文档化的显式边界（非静默）：输入经通用路径提升 float64，显式 :dtype/:out 限存储级类型 | 无需改动 |

环境差异带来的新认识（写入设计约定）：`def-vt-reduce` 的内核展开规模受
编译器堆约束（8 输入 × 8 输出 × 3 路径 × 14 算子在 1GB 动态空间下编译耗尽），
执行层特化必须与物理层存储类型对齐——这正是三层分离的工程价值：
逻辑 dtype 全集可用、物理存储类型特化、逻辑级 dtype 由通用内核兜底。

回归结果：`bash test/run-tests.sh` 19/19 套件全部通过。

