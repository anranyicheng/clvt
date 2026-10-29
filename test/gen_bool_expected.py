"""生成 bool-expected.sexp —— clvt 布尔 dtype（CONVENTIONS §17）与 numpy 的对照表。

设计要点：**只存期望值，不存输入**。输入由 `BX` 这份固定字面量给出，
`test/bool-dtype-test.lisp` 里有逐位等价的一份（同样的 -0.0 / nan / ±inf）。
期望值本体全是 0/1 整数（布尔数组展平）或整数（计数），因此文件很小。

用法：python3 test/gen_bool_expected.py
"""
import numpy as np

# 与 clvt 的规范 NaN **逐位相同**：本库的 NaN 来自 x86 的 0/0，
# 位模式是 0xFFF8000000000000（**负** QNaN），而 Python 的 float("nan") 是
# 正的 0x7FF8000000000000。两者在 isnan 上一致，但 signbit / 真值化会分叉，
# 所以这里显式构造同一个负 NaN —— 期望值才是在测布尔语义，而不是在测
# 「谁的 NaN 符号位是什么」。
NAN = np.array([0xFFF8000000000000], dtype=np.uint64).view(np.float64)[0]
INF = float("inf")

# 与 lisp 侧 *BX* 逐位相同的一维输入：覆盖负数、+0.0、小数、1.0、
# NaN、±Inf、-0.0、正数。
BX = np.array([-1.0, 0.0, 0.5, 1.0, NAN, INF, -INF, -0.0, 3.0])

# 与 lisp 侧 *BM* 相同的 3×3 输入（按 C 序展平即 BX）。
BM = BX.reshape(3, 3)

# 另一条用于二元运算的输入（与 BX 同形，值不同：三正三负三零）。
BY = np.array([2.0, -1.0, 0.0, 1.0, 0.0, -3.0, 0.0, 0.5, -0.0])


def flat(a):
    """展平并按「布尔 → 0/1 整数」输出。"""
    a = np.asarray(a)
    if a.dtype == np.bool_:
        return [int(v) for v in a.ravel()]
    if a.dtype.kind in "iu":
        return [int(v) for v in a.ravel()]
    raise SystemExit(f"期望值只允许布尔/整数，收到 {a.dtype}")


CASES = []


def add(name, result):
    """登记一个用例：名字 + 结果数组（dtype/shape 一并记录）。"""
    r = np.asarray(result)
    dt = {"b": "BOOL", "i": "INT64" if r.dtype.itemsize == 8 else "INT32"}.get(r.dtype.kind)
    if dt is None:
        raise SystemExit(f"{name}: 不支持的期望 dtype {r.dtype}")
    CASES.append((name, list(r.shape), dt, flat(r)))


# ---------------- 🔍 数组内容测试 ----------------
add("isnan-1d", np.isnan(BX))
add("isinf-1d", np.isinf(BX))
add("isfinite-1d", np.isfinite(BX))
add("isposinf-1d", np.isposinf(BX))
add("isneginf-1d", np.isneginf(BX))
add("signbit-1d", np.signbit(BX))
add("isnan-2d", np.isnan(BM))
add("isfinite-2d", np.isfinite(BM))

# ---------------- ⚖️ 比较运算 ----------------
add("eq-self-1d", BX == BX)            # nan != nan → False
add("ne-self-1d", BX != BX)
add("gt-zero-1d", BX > 0.0)
add("ge-zero-1d", BX >= 0.0)
add("lt-zero-1d", BX < 0.0)
add("le-zero-1d", BX <= 0.0)
add("eq-xy-1d", BX == BY)
add("lt-xy-1d", BX < BY)
add("gt-xy-1d", BX > BY)

# ---------------- 🔗 逻辑运算 ----------------
M1 = np.isfinite(BX)
M2 = np.signbit(BX)
add("logical-and", np.logical_and(M1, M2))
add("logical-or", np.logical_or(M1, M2))
add("logical-not", np.logical_not(M1))
add("logical-xor", np.logical_xor(M1, M2))
# 数值入参按「非零即真」
add("logical-and-num", np.logical_and(BX, BY))

# ---------------- 🧭 any / all（整体 + 沿轴） ----------------
add("any-1d", np.any(BX))
add("all-1d", np.all(BX))
add("any-2d", np.any(BM))
add("all-2d", np.all(BM))
add("any-axis0", np.any(BM, axis=0))
add("all-axis0", np.all(BM, axis=0))
add("any-axis1", np.any(BM, axis=1))
add("all-axis1", np.all(BM, axis=1))
add("any-axis0-keep", np.any(BM, axis=0, keepdims=True))
add("all-axis1-keep", np.all(BM, axis=1, keepdims=True))
add("any-empty", np.any(np.array([0.0, 0.0, 0.0])))
add("all-empty", np.all(np.array([0.0, 0.0, 0.0])))
add("any-nan", np.any(np.array([NAN])))
add("all-nan", np.all(np.array([NAN])))

# ---------------- 🤝 数组关系检查 ----------------
add("isclose-same", np.isclose(BX, BX))          # inf==inf 为 True、nan 为 False
add("isclose-near", np.isclose([1.0, 1.000001], [1.0, 1.0]))
add("isclose-far", np.isclose([1.0, 2.0], [1.0, 1.0]))
add("array-equal", np.array([np.array_equal(BX, BX.copy())], dtype=np.bool_))
add("array-equal-nan", np.array([np.array_equal(BX, BX)], dtype=np.bool_))
add("allclose-true", np.array([np.allclose([1.0, 2.0], [1.0, 2.0])], dtype=np.bool_))
add("allclose-false", np.array([np.allclose([1.0, 2.0], [1.0, 2.5])], dtype=np.bool_))

# ---------------- 📚 集合 / 转型 / 归约 ----------------
add("isin", np.isin(BX, [0.0, 1.0]))
add("isin-invert", np.isin(BX, [0.0, 1.0], invert=True))
add("in1d", np.isin(BX, [2.0, 3.0]))
add("astype-bool", BX.astype(bool))
add("astype-bool-2d", BM.astype(bool))
add("sum-bool", np.sum(BX.astype(bool)))
add("sum-bool-axis0", np.sum(BM.astype(bool), axis=0))
add("count-nonzero", np.count_nonzero(BX))

# ---------------- 🧪 布尔掩码参与选择 -------------#
add("any-of-compare", np.any(BX > 0.0))

lines = [";;; 由 test/gen_bool_expected.py 生成，勿手改。",
         ";;; 格式：((用例名 形状 dtype (展平值 …)) …)",
         "(",]
for i, (name, shape, dt, vals) in enumerate(CASES):
    tail = ")" if i == len(CASES) - 1 else ""
    lines.append(' ("%s" (%s) :%s (%s))%s'
                 % (name, " ".join(str(d) for d in shape), dt.lower(),
                    " ".join(str(v) for v in vals), tail))
lines.append(")")
open("test/bool-expected.sexp", "w", encoding="utf-8").write("\n".join(lines) + "\n")
print(f"wrote test/bool-expected.sexp: {len(CASES)} cases")
