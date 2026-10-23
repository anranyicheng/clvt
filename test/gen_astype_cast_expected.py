#!/usr/bin/env python3
"""gen_astype_cast_expected.py — 生成 vt-astype 转换矩阵期望值（numpy 实测）

产物: test/astype-cast-expected.sexp
用法: python3 test/gen_astype_cast_expected.py

设计要点（CONVENTIONS §16.4 / §7.2）：
  · 覆盖 10 种逻辑 dtype × 10 种目标 dtype 的完整 100 格；
  · 源值全部取「各位宽都能精确表示、且窄整数目标下不触发 C 未定义行为」的
    小整数/半整数（|v| ≤ 100），因此整数目标的回绕在 numpy 与 CL 两侧同义；
  · 非有限值（NaN/±Inf）不进入矩阵：numpy 对 NaN/Inf → 整数的结果是平台相关的
    C cast，而本库契约为「统一返回 0」（§7.2），这类以契约断言单独覆盖；
  · 浮点/复数期望值按 double 打印（float32 值 → double 是无损的），
    比较时按 dtype 分级容差（float64 1e-12、float32/complex64 1e-6）。
"""
import pathlib
import warnings

import numpy as np

DTYPES = [
    ":complex128", ":complex64",
    ":float64", ":float32",
    ":int64", ":int32", ":int16", ":int8", ":uint8", ":uint16",
]

NP = {
    ":complex128": np.complex128, ":complex64": np.complex64,
    ":float64": np.float64, ":float32": np.float32,
    ":int64": np.int64, ":int32": np.int32, ":int16": np.int16,
    ":int8": np.int8, ":uint8": np.uint8, ":uint16": np.uint16,
}

# 复数源：实部/虚部都取半整数（能被所有位宽精确表示）
CX_VALUES = [(1.5, 2.5), (-3.25, 0.75), (0.0, 0.0), (7.0, -0.0),
             (-2.0, -1.0), (100.0, -100.0)]
# 浮点源
FL_VALUES = [1.5, -2.25, 0.0, -0.0, 7.0, -100.0]
# 整数源（无符号源不能用负值：numpy 2.x 会拒绝越界的 Python 整数）
IN_VALUES = [1, -2, 0, 7, -100, 100]
UIN_VALUES = [1, 2, 0, 7, 100, 200]


def lisp_num(v):
    """把数值打印成 Lisp 字面量（显式 d0 后缀，见 §16.18.6）。"""
    if isinstance(v, (complex, np.complexfloating)):
        return "#C(%sd0 %sd0)" % (repr(float(v.real)), repr(float(v.imag)))
    if isinstance(v, (float, np.floating)):
        return "%sd0" % repr(float(v))
    return str(int(v))


def is_complex(dt):
    return dt.startswith(":complex")


def is_int(dt):
    return dt in (":int64", ":int32", ":int16", ":int8", ":uint8", ":uint16")


def main():
    out = []
    for src in DTYPES:
        if is_complex(src):
            vals = [complex(r, i) for (r, i) in CX_VALUES]
        elif src in (":uint8", ":uint16"):
            vals = list(UIN_VALUES)
        elif is_int(src):
            vals = list(IN_VALUES)
        else:
            vals = list(FL_VALUES)
        a = np.array(vals, dtype=NP[src])
        for dst in DTYPES:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                b = a.astype(NP[dst])
            key = "%s->%s" % (src[1:], dst[1:])
            mode = ":exact" if is_int(dst) else ":tol"
            got = " ".join(lisp_num(v) for v in b)
            out.append('("%s" %s %s %s (%s))' % (key, src, dst, mode, got))
    dst = pathlib.Path(__file__).resolve().parent / "astype-cast-expected.sexp"
    header = (";;;; astype-cast-expected.sexp — 由 gen_astype_cast_expected.py 从本机\n"
              ";;;; numpy %s 实测生成，**请勿手改**。\n"
              ";;;; 每项：(key src-dtype dst-dtype cmp-mode (期望值...))\n" % np.__version__)
    dst.write_text(header + "\n".join(out) + "\n", encoding="utf-8")
    print("生成 %d 条用例 → %s" % (len(out), dst.name))


if __name__ == "__main__":
    main()
