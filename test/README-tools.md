# test/ 测试工具链地图

> 面向维护者（含 AI 协作者）的一页速查。改任何测试前先看这里，避免重建已废弃的链路。
> `check_key_sync.py` 门禁会在 `run-tests.sh` 开头自动执行；它失败时先读它的报因，不要先改 lisp。

## 参考值供给：唯一枢纽

```
                    ┌─ run_all_tests.lisp          (funcall E "key")
                    ├─ run_param_tests.lisp        (E J "key")
ref_compute.py ────┼─ numpy-compare-test.lisp     (E "key")
 (python3 子进程)   ├─ auto-compare-test.lisp      (E "key")
                    └─ legacy-coverage-test.lisp   (E J "key")
```

**`ref_compute.py` 是唯一真理源**：运行时由 lisp 套件以子进程调用，stdout 输出
`{"key": {"t":"a","s":[..],"d":"..","v":[..]}}` 形式的 numpy 参考值（可选 PyTorch 交叉验证）。
lisp 侧的查询 key 与此文件中的 `R["..."]` **只靠命名约定同步**——新增用例时两边必须同时改，
`check_key_sync.py` 会在跑测试前机械校验（缺失→FAIL 并列出 key；死供给→WARN）。

## 各文件角色与消费方

| 文件 | 角色 | 消费方 | 注意 |
|---|---|---|---|
| `ref_compute.py` | numpy 参考值唯一供给（384 key） | 4 个 lisp 套件运行时 | 改 key 必须 4 套件同步检查 |
| `check_key_sync.py` | 防漂移门禁（key 约定 + codegen 同步） | `run-tests.sh` 前导 | 静态秒级；5 套件 651 key 全覆盖 |
| `legacy-coverage-test.lisp` | 静态 JSON 时代未接线用例盘活（284 例） | run-tests.sh | 生成后人工校准的固定表；期望值运行时现算 |
| `gen_probes.py` | codegen → `differential-probes-test.lisp`（期望内嵌，自包含） | run-tests.sh | **生成物勿手改**；改后跑门禁校验同步 |
| `gen_probes2.py` | codegen → `differential-probes-test2.lisp`（#200-#377 差分探针） | run-tests.sh | 同上 |
| `gen_report.py` / `gen_report2.py` | 离线诊断：探针失败 → 报告/最小复现 | 人工 | 不在 CI 链路 |
| `gen_numpy_expected.py` | 已废弃，仅重定向到 ref_compute.py | 兼容入口 | 勿扩展 |
| `torch-compare.py` | 独立 PyTorch 对比 + 性能基线（18 算子） | 人工（可选） | 不在 run-tests.sh 链路 |

## 已删除的旧架构（勿重建）

静态预生成 JSON 时代的三件套 `all_expected.json` / `param_expected.json` /
`expected_numpy.json` 及其生成器 `gen_all_tests.py` / `gen_param_tests.py`
已于架构迁移时废弃（见 CHANGELOG「参考值实时生成」一节），2026-10 清理删除。
它们用旧 key 风格（如 `arange_10`），与现行枢纽（`arange`）不同——**不要参考它们推测 key 约定**。

## 2026-10 盘活记录（560 遗产用例去向）

架构迁移时仅 142/260 + 138/300 例被接线，其余 281 例从未被任何套件消费。
已全部盘活：oracle 并入 `ref_compute.py`（`_legacy_all_block`/`_legacy_param_block`），
lisp 调用补入 `legacy-coverage-test.lisp`（284 例含 3 条 narrow 语义回归）。
期间发现并处理的 clvt 语义点：
- `vt-narrow` 负切片索引/越界 clip 已实现（src/manip.lisp，numpy 对齐）
- `vt-det` 不支持 3D 批量（det_batch 以逐批等效调用覆盖，待上游扩展）
- `vt-diagonal` 3D 为 pytorch 末两维风格（numpy.trace 默认前两轴，trace_3d 以 transpose 等效）
- `vt-gradient` 1D 返回裸 VT、多轴返回 list（grad-first helper 统一）

## 修改清单（按场景）

- **给现有套件加数值用例**：在 `ref_compute.py` 加 `R["fn_场景"] = ...`，在对应 lisp 加
  `(T! "..." (funcall E "fn_场景") (vt-...))`，跑 `python3 test/check_key_sync.py`。
- **改探针**：只改 `gen_probes*.py`，跑 `python3 test/gen_probes*.py` 重新生成，门禁会校验一致性。
- **新套件接参考值**：仿照 `auto-compare-test.lisp` 的子进程 + 内置 JSON 解析器结构，
  并把新文件的 key 提取正则登记进 `check_key_sync.py` 的 `CONSUMERS`。
