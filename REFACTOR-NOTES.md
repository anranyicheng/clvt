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
