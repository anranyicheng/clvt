#!/usr/bin/env python3
"""gen_complex_elementary_expected.py — 生成复数初等函数（log / log2 / log10 /
sign / sinc / expm1 / log1p）的 numpy 2.4.6 参照值。

输出 test/complex-elementary-expected.sexp，格式：

    ((inputs        (re im) ...)
     (log           (re im) ...)
     (log2          (re im) ...)
     ...)

  · re/im 为 double-float 字面量（写成 1.23d0 形式，Lisp 直接 read 即可）；
  · NaN / ±Inf 分别写符号 nan / pos-inf / neg-inf，测试侧按语义比较
    （NaN 只与 NaN 相等，Inf 比符号），避免依赖具体 payload 位模式。

网格：21 个实部 × 21 个虚部 = 441 点，刻意覆盖
  ±0.0（符号零）、±1e-8、±0.5、±1、±1.5、±2、±4、±1e300、±Inf、NaN，
既包含分支切割（y=±0 且 |x|>1）与极点（z=0），也包含溢出边界。

用法：python3 test/gen_complex_elementary_expected.py
"""

import itertools

import numpy as np

VALS = [-1e300, -4.0, -2.0, -1.5, -1.0, -0.5, -0.1, -1e-8, -0.0, 0.0, 1e-8,
        0.1, 0.5, 1.0, 1.5, 2.0, 4.0, 1e300, np.inf, -np.inf, np.nan]

FUNCS = [
    ("log", np.log),
    ("log2", np.log2),
    ("log10", np.log10),
    ("sign", np.sign),
    ("sinc", np.sinc),
]


def num(x):
    x = float(x)
    if np.isnan(x):
        return "nan"
    if np.isposinf(x):
        return "pos-inf"
    if np.isneginf(x):
        return "neg-inf"
    mant, _, expo = f"{x:.17e}".partition("e")
    return f"{mant}d{int(expo)}"


def pair(c):
    return f"({num(c.real)} {num(c.imag)})"


def main():
    pts = [complex(a, b) for a, b in itertools.product(VALS, VALS)]
    z = np.array(pts, dtype=np.complex128)

    blocks = [("inputs", [pair(p) for p in pts])]
    for name, fn in FUNCS:
        with np.errstate(all="ignore"):
            res = fn(z)
        blocks.append((name, [pair(p) for p in res]))

    out = ["("]
    for idx, (key, rows) in enumerate(blocks):
        head = "(" if idx == 0 else " ("
        out.append(f"{head}{key}")
        out.append("\n  " + "\n  ".join(rows))
        out.append(" )")
    out.append(")")
    out.append("")

    with open("test/complex-elementary-expected.sexp", "w") as fh:
        fh.write("\n".join(out))
    print(f"{len(pts)} 点 × {len(FUNCS)} 函数已写入 "
          f"test/complex-elementary-expected.sexp")


if __name__ == "__main__":
    main()
