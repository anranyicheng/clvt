"""生成 selection-expected.sexp —— median / nanmedian 与 numpy 的对照表。

设计要点：**只存期望值，不存输入**。输入由下面的 `gen_*` 公式生成，
`test/selection-test.lisp` 里有一份逐位等价的 Lisp 实现（整数取模 + 一次
double 除法，两边都是 IEEE 舍入，结果确定）。这样期望值文件只有几 KB，
而不是几百 KB 的输入副本。

用法：python3 test/gen_selection_expected.py
"""
import numpy as np

NAN = float("nan")
INF = float("inf")


def gen_ramp(n):
    """互不相同的值：(i·37 mod 1009)/1009 + i·((i·61 mod 1013)/1013)。"""
    return np.array([complex(((i * 37) % 1009) / 1009.0,
                             ((i * 61) % 1013) / 1013.0) for i in range(n)])


def gen_grid(n):
    """网格重值：实部虚部各取自 4 个值 —— 用来打死两分区 quickselect。"""
    cols = [-0.5, 0.0, 0.25, 1.0]
    out = []
    for i in range(n):
        out.append(complex(cols[i % 4], cols[(i // 4) % 4]))
    return np.array(out, dtype=np.complex128)


def gen_alleq(n):
    return np.full(n, complex(0.5, -0.5))


def gen_desc(n):
    return np.array([complex((n - i) / n, -(n - i) / n) for i in range(n)])


def gen_nan(n):
    """每隔 7 个放一个 NaN+NaNj。"""
    a = gen_ramp(n)
    a[::7] = complex(NAN, NAN)
    return a


def gen_nan_re(n):
    """NaN 只落在实部（单分量 NaN）。"""
    a = gen_ramp(n)
    a[::5] = complex(NAN, 0.25)
    return a


def gen_dense_nan(shape):
    a = np.array([complex(((i * 37) % 1009) / 1009.0, ((i * 61) % 1013) / 1013.0)
                  for i in range(int(np.prod(shape)))]).reshape(shape)
    flat = a.reshape(-1)
    flat[::4] = complex(NAN, NAN)
    flat[1::11] = complex(NAN, 0.75)
    return a


def gen_allnan(n):
    return np.full(n, complex(NAN, NAN))


def pair(z):
    def num(v):
        v = float(v)
        if np.isnan(v):
            return "nan"
        if np.isposinf(v):
            return "posinf"
        if np.isneginf(v):
            return "neginf"
        return f"{v:.17e}".replace("e", "d")
    return f"({num(z.real)} {num(z.imag)})"


CASES = [
    # (名字, 生成器/形状, 归约函数, axis, keepdims)
    ("median-flat-n1", ("ramp", (1,)), "median", None, False),
    ("median-flat-n2", ("ramp", (2,)), "median", None, False),
    ("median-flat-n3", ("ramp", (3,)), "median", None, False),
    ("median-flat-n7", ("ramp", (7,)), "median", None, False),
    ("median-flat-n8", ("ramp", (8,)), "median", None, False),
    ("median-flat-n999", ("ramp", (999,)), "median", None, False),
    ("median-flat-n1000", ("ramp", (1000,)), "median", None, False),
    ("median-flat-n1001", ("ramp", (1001,)), "median", None, False),
    ("median-flat-grid400", ("grid", (400,)), "median", None, False),
    ("median-flat-grid401", ("grid", (401,)), "median", None, False),
    ("median-flat-alleq500", ("alleq", (500,)), "median", None, False),
    ("median-flat-desc600", ("desc", (600,)), "median", None, False),
    ("median-flat-nan300", ("nan", (300,)), "median", None, False),
    ("median-flat-nanre200", ("nanre", (200,)), "median", None, False),
    ("median-flat-empty", ("ramp", (0,)), "median", None, False),
    ("median-2d-17x13-axis0", ("ramp", (17, 13)), "median", 0, False),
    ("median-2d-17x13-axis1", ("ramp", (17, 13)), "median", 1, False),
    ("median-2d-17x13-axis1-kd", ("ramp", (17, 13)), "median", 1, True),
    ("median-2d-grid-axis0", ("grid", (17, 13)), "median", 0, False),
    ("median-2d-grid-axis1", ("grid", (17, 13)), "median", 1, False),
    ("median-2d-nan-axis0", ("nan", (24, 11)), "median", 0, False),
    ("median-2d-nan-axis1", ("nan", (24, 11)), "median", 1, False),
    ("median-3d-axis2", ("ramp", (5, 4, 9)), "median", 2, False),
    ("median-3d-axis0", ("ramp", (5, 4, 9)), "median", 0, False),
    # ---- nanmedian ----
    ("nanmedian-flat-n999", ("ramp", (999,)), "nanmedian", None, False),
    ("nanmedian-flat-nan300", ("nan", (300,)), "nanmedian", None, False),
    ("nanmedian-flat-nanre200", ("nanre", (200,)), "nanmedian", None, False),
    ("nanmedian-flat-allnan20", ("allnan", (20,)), "nanmedian", None, False),
    ("nanmedian-2d-nan-axis0", ("nan", (24, 11)), "nanmedian", 0, False),
    ("nanmedian-2d-nan-axis1", ("nan", (24, 11)), "nanmedian", 1, False),
    ("nanmedian-2d-grid-axis1", ("grid", (17, 13)), "nanmedian", 1, False),
    ("nanmedian-3d-dense-axis1", ("dense", (9, 7, 4)), "nanmedian", 1, False),
    ("nanmedian-3d-dense-axis0", ("dense", (9, 7, 4)), "nanmedian", 0, False),
    ("nanmedian-3d-dense-flat", ("dense", (9, 7, 4)), "nanmedian", None, False),
]


def build(kind, shape):
    n = int(np.prod(shape)) if shape else 1
    if kind == "ramp":
        a = gen_ramp(n) if n else np.array([], dtype=np.complex128)
    elif kind == "grid":
        a = gen_grid(n)
    elif kind == "alleq":
        a = gen_alleq(n)
    elif kind == "desc":
        a = gen_desc(n)
    elif kind == "nan":
        a = gen_nan(n)
    elif kind == "nanre":
        a = gen_nan_re(n)
    elif kind == "allnan":
        a = gen_allnan(n)
    elif kind == "dense":
        a = gen_dense_nan(shape)
    else:
        raise ValueError(kind)
    if shape:
        a = a.reshape(shape)
    return a


lines = ["(("]
for name, (kind, shape), fn, axis, keep in CASES:
    a = build(kind, shape)
    f = np.median if fn == "median" else np.nanmedian
    with np.errstate(all="ignore"):
        if axis is None:
            r = f(a, keepdims=keep)
        else:
            r = f(a, axis=axis, keepdims=keep)
    ra = np.asarray(r, dtype=np.complex128).reshape(-1)
    shp = " ".join(str(d) for d in a.shape)
    vals = "\n   ".join(pair(v) for v in ra) if ra.size else ""
    lines.append(f'("{name}" "{kind}" ({shp}) {"nil" if axis is None else axis} '
                 f'{"t" if keep else "nil"}')
    lines.append(f" ({vals}))" if vals else " ()")
lines.append("))")
open("test/selection-expected.sexp", "w").write("\n".join(lines) + "\n")
print("cases:", len(CASES))
