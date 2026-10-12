# clvt 形状/视图/拼接/切片 跨语言互操作测试

用 **SBCL + clvt** 生成结果并用 `vt-save` 落盘，再用 **Python + NumPy** 独立重建、
逐项比对，验证「结构类」张量操作（**不含数值计算部分**）的正确性。

规模基准（按要求固定）：

```lisp
(vt-reshape (vt-arange 1000000) '(1000 1000))   ; float64, 形如 m[i][j] = i*1000 + j
```

## 文件

| 文件 | 作用 |
| --- | --- |
| `vt-manip-interop-test.lisp` | Lisp 端（阶段 1）：195 个用例 + Lisp 自检（oracle/错误契约）+ `vt-save` 落盘 + `manifest.json` |
| `verify_manip_numpy.py` | NumPy 端（阶段 2）：逐用例重建期望值，比对 shape / dtype / 数值 |

本套件是**两阶段流水线**，已作为 `manip-interop-test` 注册进统一测试入口
`test/run-tests.sh`（`PIPELINE_SUITES`），不再需要独立脚本。

## 运行

```bash
# 1) 安装 SBCL（可用 gitee.com/xizangqt/sbcl-binary 的二进制包）
tar -xjf sbcl-2.6.8-x86-64-linux-binary.tar.bz2
cd sbcl-2.6.8-x86-64-linux && sh install.sh --prefix=/usr/local && cd -

# 2) 安装 Quicklisp
curl -O https://beta.quicklisp.org/quicklisp.lisp
sbcl --non-interactive --load quicklisp.lisp --eval '(quicklisp-quickstart:install)'
#   再把 ql 初始化片段写入 ~/.sbclrc；然后
#   ln -s "$(pwd)" ~/quicklisp/local-projects/clvt
sbcl --non-interactive --eval '(ql:quickload :sb-simd)'   # 首次拉取依赖

# 3) 升级 numpy
python3 -m pip install --upgrade numpy

# 4) 经统一入口跑本套件（阶段 1 + 阶段 2）
bash test/run-tests.sh --suite manip-interop-test
#   或随全量回归一起跑：bash test/run-tests.sh
```

`run-tests.sh` 会为流水线套件建一个临时输出目录并在跑完后**自动清理**；
若要保留产物（如 `numpy_verify_report.json`、各 `.npy`），设置 `CLVT_TEST_OUT`：

```bash
CLVT_TEST_OUT=/tmp/clvt-out bash test/run-tests.sh --suite manip-interop-test
```

产物内容：
每个用例输出 `<name>.npy`（与 `np.save` 逐字节兼容）；结果元素数 ≤ 250000 的用例
额外输出 `<name>.sexp`（clvt 原生 s-expression 格式）并做 `vt-save → vt-load` 往返自检。

## 覆盖的函数（48 个）

| 分组 | 函数 |
| --- | --- |
| 形状/视图 | `vt-view` `vt-reshape` `vt-transpose` `vt-squeeze` `vt-unsqueeze` `vt-expand-dims` `vt-flatten` `vt-ravel` `vt-swapaxes` `vt-rot90` `vt-narrow` `vt-split` `vt-vsplit` `vt-hsplit` `vt-dsplit` |
| 拼接/构造 | `vt-stack` `vt-vstack` `vt-hstack` `vt-dstack` `vt-concatenate` `vt-concat` `vt-repeat` `vt-tile` `vt-pad` `vt-broadcast-to` `vt-contiguous` `vt-flip` `vt-roll` `vt-triu` `vt-tril` `vt-diagonal` `vt-flatten-to-nested` `vt-append` `vt-insert` `vt-delete` |
| 索引/选择 | `vt-slice`(+`setf`) `vt-item` `vt-take` `vt-put` `vt-where` `vt-argwhere` `vt-nonzero` `vt-choose` `vt-select` `vt-extract` `vt-searchsorted` `vt-digitize` `vt-bincount` |
| 类型 | `vt-astype` |

另有：`vt-save` / `vt-load`（`:npy` 与 `:lisp` 双格式往返）、`vt-contiguous-p`、`vt-array-equal`。

## 用例分布

- **A–L 组，195 个用例**：A(12) view/reshape · B(20) transpose/rot90/flip/roll ·
  C(13) squeeze/flatten/ravel · D(15) narrow/slice · E(11) split 族 ·
  F(15) stack/concatenate 族 · G(21) repeat/tile/pad/broadcast/contiguous ·
  H(10) triu/tril/diagonal/flatten-to-nested · I(15) append/insert/delete ·
  J(33) take/put/where/nonzero/choose/select/extract/searchsorted/digitize/bincount/astype ·
  K(20) **多算子组合**（如 `transpose→reshape→contiguous`、`slice→transpose→flatten`、
  `stack→reshape`、`split→vstack` 往返、`pad→flip→roll`…）。
- **L 组（5 个用例）空索引**：`vt-take` 空索引（含 `:axis`）返回空结果、`vt-put` 空索引为 no-op、`vt-insert` 空 `positions` 原样返回 —— 与 `np.take(a, [])` / `a.put([], [])` / `np.insert(a, [], v)` 逐位一致。
- **Lisp 自检 63 项**：元素位置 oracle（如 `transpose[7][5] == m[5][7]`）、
  连续性判定、形状推导，以及 15 条错误契约（越界、形状不匹配、非法参数等）断言。

## 结果

```
总计: 195   通过: 195   失败: 0   已知差异: 0   缺参考: 0
结论: 全部通过 ✅
```

（原始版本有 185 例、1 项已知差异 —— `vt-insert` 的 axis 路径问题；
该缺陷已在源码中修复，用例改名为 `I15_insert_ax1_moveaxis` 并转入正常判定，
同时新增 `H11–H13`（batch 对角线显式轴）、`G22/G23`（roll 位移取模 0）
共 5 个用例；后又补 L 组 5 个空索引用例，故为 195 例。差异详情与修复见 `test/CONTRACT-ALIGNMENT-REPORT.md`。）

## 已修复的差异（原「已知差异」）

### `vt-insert`：`axis` 给定时未施加 `moveaxis(values, 0, axis)` —— 已修复

- **原现象**：`(vt-insert m 7 (vt-narrow m 0 0 2) :axis 1)`（`values` 形状 `(2,1000)`）
  与 `np.insert(M, 7, M[0:2], axis=1)` 结果不同；首个差异位于 `(0,8)`，
  clvt `1.0` vs numpy `1000.0`（1000×1002 的输出里有 1998 处不同）。
- **根因**：numpy 对**标量** `obj` 会先做 `values = np.moveaxis(values, 0, axis)`
  再作为整块写入；clvt 用行主序 `vt-reshape` 顶替该变换。由于形状校验只看
  「元素总数」，当 `values` 元素总数恰等于块元素数（`(2,1000)` 与块 `(1000,2)`
  都是 2000 个）时校验通过，从而**静默**给出不同结果。
- **修复**：`src/join.lisp` 的 `%insert-axis` 改为完整复刻 numpy 2.x：
  `ndmin(values) → 标量 obj 时 moveaxis(0→axis) → numnew = values.shape[axis]
  → 广播写入块`；并区分标量 `obj` 与列表 `obj=[k]`（二者 numpy 结果不同）。
- **验证**：`test/contract-regression-test.lisp`（小尺寸逐值 + 1000×1000 公式 oracle）
  与 `test/vt-manip-interop-test.lisp` 用例 `I15_insert_ax1_moveaxis`（1000×1000，
  与 numpy 逐位比对）。

### 说明：`vt-diagonal` 的 batch 约定

`vt-diagonal` 对 ≥3 维输入取**最后两轴**的对角线（同 `torch.diagonal` /
`np.diagonal(a, axis1=-2, axis2=-1)`）；NumPy 的 `np.diagonal(a)` 默认是
`axis1=0, axis2=1`（历史特例），batch 下两者不同。二维输入两者一致。
这不是缺陷，校验脚本已按 clvt 的约定（显式 `-2/-1`）比对。
