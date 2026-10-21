#!/usr/bin/env python3
"""生成「按轴去重 / 迹」差分测试的期望值（test/setops-axis-expected.sexp）。

覆盖第五轮新增的两个通用（非复数专属）特性：
  · vt-unique :axis —— numpy 2.x 的按轴去重（含 return-index/inverse/counts）
  · vt-trace :offset / :axis1 / :axis2 —— numpy.trace 的完整签名

所有期望值都由**本机 numpy 实测**产生（CONVENTIONS §10.4：禁手写期望值），
本文件只负责把 numpy 结果序列化成 clvt 可读的 s-expression，格式与
gen_complex_ops_expected.py 一致（:VALUE 恒为扁平列表 + :SHAPE）。

用法: python3 test/gen_setops_axis_expected.py [输出路径]
"""

import math
import re
import sys

import numpy as np

OUT = sys.argv[1] if len(sys.argv) > 1 else "test/setops-axis-expected.sexp"

CASES = []

NAN = float("nan")
INF = float("inf")


def real_lit(x):
    x = float(x)
    if math.isnan(x):
        return ":NAN"
    if math.isinf(x):
        return ":INF" if x > 0 else ":NEG-INF"
    if x == int(x) and abs(x) < 1e15:
        return "%dd0" % int(x)
    r = repr(x)
    if "e" in r or "E" in r:
        mant, exp = re.split("[eE]", r)
        if "." not in mant:
            mant += ".0"
        return "%sd%s" % (mant, exp if exp.startswith(("+", "-")) else "+" + exp)
    return r + "d0"


def lit(x):
    if isinstance(x, (complex, np.complexfloating)):
        return "(:CX %s %s)" % (real_lit(x.real), real_lit(x.imag))
    if isinstance(x, (float, np.floating)):
        return real_lit(x)
    return str(int(x))


def lit_str(s):
    return '"%s"' % s.replace("\\", "\\\\").replace('"', '\\"')


def dtype_kw(dt):
    return ":" + str(np.dtype(dt)).upper()


def case(name, form, value, dtype=None, shape=None):
    arr = np.asarray(value)
    if dtype is None:
        dtype = arr.dtype
    if shape is None:
        shape = list(arr.shape)
    flat = "(" + " ".join(lit(e) for e in arr.ravel()) + ")" if arr.size else "()"
    CASES.append(dict(name=name, form=form, value=flat, dtype=dtype_kw(dtype),
                      shape="(" + " ".join(str(d) for d in shape) + ")"))


# ---------------------------------------------------------------------------
# 输入数据（与 test/setops-axis-trace-test.lisp 一一对应）
# ---------------------------------------------------------------------------
UA = np.array([[2, 0, 1, 5],
               [1, 5, 2, 0],
               [2, 0, 1, 5]], dtype=np.int64)          # 行 0 与行 2 相同

UB = np.array([[3, 1, 3],
               [1, 2, 2],
               [3, 1, 3],
               [5, 5, 5]], dtype=np.int64)             # 列 0 与列 2 相同

UC = np.array([[NAN, 1.0],
               [NAN, 1.0],
               [2.0, 0.0]], dtype=np.float64)          # NaN 行不合并

UD = np.array([[[0, 1], [0, 1], [5, 5]],
               [[9, 9], [0, 1], [5, 5]]], dtype=np.float64)

UE = np.array([[1 + 2j, 0 + 0j],
               [1 + 2j, 0 + 0j],
               [0 + 1j, 3 + 0j]], dtype=np.complex128)

UE2 = np.array([[1 + 2j, 0 + 0j, 5 - 1j],
                [0 + 1j, 3 + 0j, 5 - 1j],
                [1 + 2j, 0 + 0j, 5 - 1j]], dtype=np.complex128)

UEMPTY = np.zeros((0, 3), dtype=np.float64)
UZERO = np.zeros((3, 0), dtype=np.float64)

TM = np.array([[1.0, 2.0, 3.0, 4.0],
               [5.0, 6.0, 7.0, 8.0],
               [9.0, 10.0, 11.0, 12.0],
               [13.0, 14.0, 15.0, 16.0]], dtype=np.float64)

TR = np.array([[1.0, 2.0, 3.0, 4.0, 5.0],
               [6.0, 7.0, 8.0, 9.0, 10.0],
               [11.0, 12.0, 13.0, 14.0, 15.0]], dtype=np.float64)   # 3x5

TC = np.array([[1 + 2j, 3 - 1j, 0 + 0j],
               [2 + 0j, 4 + 4j, 1 - 1j],
               [0 + 3j, 5 + 0j, 6 + 2j]], dtype=np.complex128)

TD = np.arange(18, dtype=np.float64).reshape(2, 3, 3)

# ---------------------------------------------------------------- unique :axis
case("ua-axis0", "(vt-unique UA :axis 0)", np.unique(UA, axis=0))
case("ua-axis1", "(vt-unique UA :axis 1)", np.unique(UA, axis=1))
case("ua-axis-1", "(vt-unique UA :axis -1)", np.unique(UA, axis=-1))
case("ub-axis0", "(vt-unique UB :axis 0)", np.unique(UB, axis=0))
case("ub-axis1", "(vt-unique UB :axis 1)", np.unique(UB, axis=1))
case("uc-axis0-nan", "(vt-unique UC :axis 0)", np.unique(UC, axis=0))
case("ud-axis1-3d", "(vt-unique UD :axis 1)", np.unique(UD, axis=1))
case("ud-axis2-3d", "(vt-unique UD :axis 2)", np.unique(UD, axis=2))
case("ue-axis0", "(vt-unique UE :axis 0)", np.unique(UE, axis=0))
case("ue2-axis1", "(vt-unique UE2 :axis 1)", np.unique(UE2, axis=1))
case("ua-axis0-idx", "(nth-value 1 (vt-unique UA :axis 0 :return-index t))",
     np.unique(UA, axis=0, return_index=True)[1])
case("ua-axis0-inv", "(nth-value 2 (vt-unique UA :axis 0 :return-inverse t))",
     np.unique(UA, axis=0, return_inverse=True)[1])
case("ua-axis0-cnt", "(nth-value 3 (vt-unique UA :axis 0 :return-counts t))",
     np.unique(UA, axis=0, return_counts=True)[1])
case("ub-axis1-idx", "(nth-value 1 (vt-unique UB :axis 1 :return-index t))",
     np.unique(UB, axis=1, return_index=True)[1])
case("ud-axis1-inv", "(nth-value 2 (vt-unique UD :axis 1 :return-inverse t))",
     np.unique(UD, axis=1, return_inverse=True)[1])
# 非连续视图输入
case("transposed-axis1",
     "(vt-unique (vt-transpose UA) :axis 1)",
     np.unique(UA.T, axis=1))
# 退化形状
case("empty-rows", "(vt-unique UEMPTY :axis 0)", np.unique(UEMPTY, axis=0))
case("zero-cols", "(vt-unique UZERO :axis 0)", np.unique(UZERO, axis=0))
# 1 维：axis 被忽略（numpy 同样走展平分支）
case("flat-1d-axis0",
     "(vt-unique (vt-from-sequence (list 3.0d0 1.0d0 3.0d0) :dtype :float64) :axis 0)",
     np.unique(np.array([3.0, 1.0, 3.0]), axis=0))

# ---------------------------------------------------------------- trace :offset
case("trace-plain", "(vt-trace TM)", np.trace(TM))
for k in (1, 2, 3, 4, -1, -2, -3, -4):
    case("trace-off%d" % k if k >= 0 else "trace-offm%d" % -k,
         "(vt-trace TM :offset %d)" % k, np.trace(TM, offset=k))
case("trace-rect-off1", "(vt-trace TR :offset 1)", np.trace(TR, offset=1))
case("trace-rect-offm1", "(vt-trace TR :offset -1)", np.trace(TR, offset=-1))
case("trace-rect-off5", "(vt-trace TR :offset 5)", np.trace(TR, offset=5))
case("trace-cplx", "(vt-trace TC)", np.trace(TC))
case("trace-cplx-off1", "(vt-trace TC :offset 1)", np.trace(TC, offset=1))
case("trace-cplx-offm1", "(vt-trace TC :offset -1)", np.trace(TC, offset=-1))
case("trace-3d-ax12", "(vt-trace TD :axis1 1 :axis2 2)",
     np.trace(TD, axis1=1, axis2=2))
case("trace-3d-ax12-off1", "(vt-trace TD :offset 1 :axis1 1 :axis2 2)",
     np.trace(TD, offset=1, axis1=1, axis2=2))
case("trace-3d-ax02", "(vt-trace TD :axis1 0 :axis2 2)",
     np.trace(TD, axis1=0, axis2=2))


def emit():
    lines = []
    lines.append(";;; 由 test/gen_setops_axis_expected.py 自动生成，请勿手改。")
    lines.append(";;; 生成器：numpy %s" % np.__version__)
    lines.append("(:GENERATOR %s" % lit_str(np.__version__))
    lines.append(" :CASES")
    lines.append(" (")
    for c in CASES:
        lines.append(
            "  (:NAME %s :FORM %s :VALUE %s :DTYPE %s :SHAPE %s)"
            % (lit_str(c["name"]), lit_str(c["form"]), c["value"], c["dtype"],
               c["shape"]))
    lines.append(" )")
    lines.append(")")
    text = "\n".join(lines) + "\n"
    with open(OUT, "w") as f:
        f.write(text)
    print("wrote %s (%d cases, numpy %s)" % (OUT, len(CASES), np.__version__))


if __name__ == "__main__":
    emit()
