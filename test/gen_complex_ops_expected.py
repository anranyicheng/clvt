#!/usr/bin/env python3
"""生成复数「数据搬运层」差分测试的期望值（test/complex-ops-expected.sexp）。

所有期望值都由**本机 numpy 实测**产生（CONVENTIONS §10.4：禁手写期望值），
本文件只负责把 numpy 结果序列化成 clvt 可读的 s-expression。

输出格式（与 gen_complex_expected.py 一致，便于共用阅读习惯）：

    (:GENERATOR "2.4.6"
     :CASES
     (
      (:NAME "..." :FORM "(vt-nansum CX-NA :axis 0)"
       :VALUE (v1 v2 ...) :DTYPE :COMPLEX128 :SHAPE (3))
      ...
     ))

要点：
  · :VALUE **恒为扁平列表**（配合 :SHAPE 由测试侧逐元素比对）——
    避免嵌套列表在 Lisp 侧还要区分「复数标量 (:CX re im)」与「数组」。
  · 复数元素写成 (:CX re im)，实数写 double-float 字面量，
    NaN/±Inf 写 :NAN/:INF/:NEG-INF（`#C(:NAN 1d0)` 不是合法 Lisp 字面量）。
"""

import math
import re
import sys

import numpy as np

OUT = sys.argv[1] if len(sys.argv) > 1 else "test/complex-ops-expected.sexp"

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
    return ":" + str(np.dtype(dt)).upper().replace("COMPLEX", "COMPLEX")


def case(name, form, value, dtype=None, shape=None):
    """数值用例：VALUE 为 numpy 标量/数组，DTYPE 缺省由它的 dtype 推出。"""
    arr = np.asarray(value)
    if dtype is None:
        dtype = arr.dtype
    if shape is None:
        shape = list(arr.shape)
    flat = "(" + " ".join(lit(e) for e in arr.ravel()) + ")" if arr.size else "()"
    CASES.append(dict(name=name, form=form, value=flat,
                      dtype=dtype_kw(dtype), shape="(" + " ".join(str(d) for d in shape) + ")"))


def case_symbol(name, form, symbol):
    CASES.append(dict(name=name, form=form, value=symbol, dtype=":SYMBOL",
                      shape="NIL", kind="symbol"))


# ---------------------------------------------------------------------------
# 与 test/complex-ops-test.lisp 严格一致的输入数据
# ---------------------------------------------------------------------------
A = np.array([1 + 2j, 3 - 4j])
B = np.array([2 - 1j, 1 + 1j])
C = np.array([1 + 2j, 3 - 4j, 5 + 0j])
NA = np.array([[1 + 2j, complex(NAN, 0), 3 - 4j],
               [5 + 0j, 6 + 1j, complex(0, NAN)]])
M2 = np.array([[1 + 1j, 2 + 0j], [3 + 0j, 0 + 4j]])
DUP = np.array([3 + 1j, 1 + 5j, 1 + 2j, 3 + 1j])
FLAT4 = np.array([1.0, 2.0, 3.0, 4.0])

# ---------------------------------------------------------------- 转换
case("view-c2f", "(vt-view CX-A nil :dtype :float64)", A.view(np.float64))
case("view-c2f-2d", "(vt-view CX-M2 nil :dtype :float64)", M2.view(np.float64))
case("view-f2c", "(vt-view FLAT4 nil :dtype :complex128)", FLAT4.view(np.complex128))
case("nan-to-num",
     "(vt-nan-to-num (vt-from-sequence (list (complex +vt-dfloat-nan+ 0.0d0) #C(1.0d0 2.0d0)) :dtype :complex128))",
     np.nan_to_num(np.array([complex(NAN, 0), 1 + 2j])))
case("nan-to-num-m", "(vt-nan-to-num CX-NA)", np.nan_to_num(NA))
case("repeat-1d", "(vt-repeat CX-A 2)", np.repeat(A, 2))
case("repeat-list", "(vt-repeat CX-A (list 1 3))", np.repeat(A, [1, 3]))
case("repeat-axis", "(vt-repeat CX-M2 2 :axis 0)", np.repeat(M2, 2, axis=0))
case("repeat-axis-1", "(vt-repeat CX-M2 2 :axis 1)", np.repeat(M2, 2, axis=1))

# ---------------------------------------------------------------- 拼接
case("concat-0", "(vt-concatenate 0 CX-A CX-B)", np.concatenate([A, B]))
case("concat-1", "(vt-concatenate 1 CX-M2 CX-M2)", np.concatenate([M2, M2], axis=1))
case("stack-0", "(vt-stack 0 CX-A CX-B)", np.stack([A, B], axis=0))
case("stack-1", "(vt-stack 1 CX-A CX-B)", np.stack([A, B], axis=1))
case("vstack", "(vt-vstack CX-A CX-B)", np.vstack([A, B]))
case("hstack", "(vt-hstack CX-A CX-B)", np.hstack([A, B]))
case("dstack", "(vt-dstack CX-A CX-B)", np.dstack([A, B]))
case("column-stack", "(vt-column-stack (list CX-A CX-B))", np.column_stack([A, B]))
case("append", "(vt-append CX-A CX-B)", np.append(A, B))
case("split", "(first (vt-split CX-C 3))",
     np.split(C, 3)[0])
case("hsplit", "(first (vt-hsplit CX-M2 2))", np.hsplit(M2, 2)[0])
case("vsplit", "(first (vt-vsplit CX-M2 2))", np.vsplit(M2, 2)[0])
case("insert", "(vt-insert CX-C 1 #C(9.0d0 9.0d0))", np.insert(C, 1, 9 + 9j))
case("delete", "(vt-delete CX-C 0)", np.delete(C, 0))

# ---------------------------------------------------------------- 剪切
case("slice-step", "(vt-slice CX-C (list 0 3 2))", C[0:3:2])
case("slice-rev", "(vt-slice CX-C (list nil nil -1))", C[::-1])
case("take", "(vt-take CX-C (list 0 2))", np.take(C, [0, 2]))
case("compress", "(vt-compress (vt-from-sequence (list 1 0 1) :dtype :int8) CX-C)",
     np.compress([True, False, True], C))
case("where", "(vt-where (vt-from-sequence (list 1 0 1) :dtype :int8) CX-C CX-Z3)",
     np.where([1, 0, 1], C, np.zeros(3, dtype=complex)))
case("flip", "(vt-flip CX-C)", np.flip(C))
case("fliplr", "(vt-fliplr CX-M2)", np.fliplr(M2))
case("flipud", "(vt-flipud CX-M2)", np.flipud(M2))
case("roll", "(vt-roll CX-C 1)", np.roll(C, 1))
case("tile", "(vt-tile CX-A 2)", np.tile(A, 2))
case("pad", "(vt-pad CX-A 1)", np.pad(A, 1))
case("rot90", "(vt-rot90 CX-M2)", np.rot90(M2))
case("transpose", "(vt-transpose CX-M2)", M2.T)
case("reshape", "(vt-reshape CX-M2 (list 4))", M2.reshape(4))
case("squeeze", "(vt-squeeze (vt-reshape CX-A (list 1 2)))", A.reshape(1, 2).squeeze())
case("expand-dims", "(vt-expand-dims CX-A 0)", np.expand_dims(A, 0))
case("broadcast-to", "(vt-broadcast-to CX-A (list 3 2))", np.broadcast_to(A, (3, 2)))

# ---------------------------------------------------------------- 集合 / 搜索
case("unique", "(vt-unique CX-DUP)", np.unique(DUP))
case("unique-counts", "(nth-value 3 (vt-unique CX-DUP :return-counts t))",
     np.unique(DUP, return_counts=True)[1])
case("unique-index", "(nth-value 1 (vt-unique CX-DUP :return-index t))",
     np.unique(DUP, return_index=True)[1])
case("unique-inverse", "(nth-value 2 (vt-unique CX-DUP :return-inverse t))",
     np.unique(DUP, return_inverse=True)[1])
case("union1d", "(vt-union1d CX-A CX-B)", np.union1d(A, B))
case("intersect1d", "(vt-intersect1d CX-A CX-DUP)", np.intersect1d(A, DUP))
case("setdiff1d", "(vt-setdiff1d CX-A CX-DUP)", np.setdiff1d(A, DUP))
case("setxor1d", "(vt-setxor1d CX-A CX-B)", np.setxor1d(A, B))
case("in1d", "(vt-in1d CX-A CX-B)", np.isin(A, B).astype(np.int8), dtype=np.int8)
case("isin", "(vt-isin CX-A CX-DUP)", np.isin(A, DUP).astype(np.int8), dtype=np.int8)
case("searchsorted-left",
     "(vt-searchsorted (vt-sort CX-DUP) (vt-from-sequence (list #C(1.0d0 1.0d0) #C(2.0d0 0.0d0) #C(3.0d0 0.0d0) #C(5.0d0 0.0d0)) :dtype :complex128))",
     np.searchsorted(np.sort(DUP), [1 + 1j, 2 + 0j, 3 + 0j, 5 + 0j]))
case("searchsorted-right",
     "(vt-searchsorted (vt-sort CX-DUP) (vt-from-sequence (list #C(1.0d0 1.0d0) #C(2.0d0 0.0d0) #C(3.0d0 0.0d0) #C(5.0d0 0.0d0)) :dtype :complex128) :side :right)",
     np.searchsorted(np.sort(DUP), [1 + 1j, 2 + 0j, 3 + 0j, 5 + 0j], side="right"))

# ---------------------------------------------------------------- nan 族
case("nansum-ax0", "(vt-nansum CX-NA :axis 0)", np.nansum(NA, axis=0))
case("nansum-ax1", "(vt-nansum CX-NA :axis 1)", np.nansum(NA, axis=1))
case("nansum-all", "(vt-nansum CX-NA)", np.nansum(NA))
case("nanprod-ax1", "(vt-nanprod CX-NA :axis 1)", np.nanprod(NA, axis=1))
case("nanmax-ax1", "(vt-nanmax CX-NA :axis 1)", np.nanmax(NA, axis=1))
case("nanmin-ax1", "(vt-nanmin CX-NA :axis 1)", np.nanmin(NA, axis=1))
# 全 NaN 时 numpy 的 nanmax/nanmin 返回**输入里的某个 NaN 元素**（所有比较恒假，
# 于是 np.max 落到第一个元素），所以结果里那个「非 NaN 分量」是输入带来的、
# 不是语义。这里用本库自己的规范复数 NaN（NaN+NaNj，见 %CX-COMPLEX-NAN-OF）
# 作输入，期望值才有意义：clvt 统一返回 NaN+NaNj（§16.15 的 D-S2）。
case("nanmax-allnan",
     "(vt-nanmax (vt-from-sequence (list (complex +vt-dfloat-nan+ +vt-dfloat-nan+) (complex +vt-dfloat-nan+ +vt-dfloat-nan+)) :dtype :complex128))",
     np.nanmax(np.array([complex(NAN, NAN), complex(NAN, NAN)])))
case("nanargmax-ax1", "(vt-nanargmax CX-NA :axis 1)", np.nanargmax(NA, axis=1))
case("nanargmin-ax1", "(vt-nanargmin CX-NA :axis 1)", np.nanargmin(NA, axis=1))
case("nanmean-ax1", "(vt-nanmean CX-NA :axis 1)", np.nanmean(NA, axis=1))
case("nanmean-all", "(vt-nanmean CX-NA)", np.nanmean(NA))
case("nanvar-ax1", "(vt-nanvar CX-NA :axis 1)", np.nanvar(NA, axis=1))
case("nanstd-ax1", "(vt-nanstd CX-NA :axis 1)", np.nanstd(NA, axis=1))
case("nanvar-ddof1", "(vt-nanvar CX-NA :axis 1 :ddof 1)", np.nanvar(NA, axis=1, ddof=1))
case("nanmedian-ax1", "(vt-nanmedian CX-NA :axis 1)", np.nanmedian(NA, axis=1))
case("nanmedian-odd", "(vt-nanmedian CX-C)", np.nanmedian(C))
case("nancumsum", "(vt-nancumsum CX-NA)", np.nancumsum(NA))
case("nancumprod", "(vt-nancumprod CX-NA)", np.nancumprod(NA))

# ---------------------------------------------------------------- 缩并 / 卷积
case("tensordot-1", "(vt-tensordot CX-M2 CX-M2 :axes 1)",
     np.tensordot(M2, M2, axes=1))
case("tensordot-2", "(vt-tensordot CX-M2 CX-M2 :axes 2)",
     np.tensordot(M2, M2, axes=2))
case("tensordot-pair", "(vt-tensordot CX-M2 CX-M2 :axes (list (list 1) (list 0)))",
     np.tensordot(M2, M2, axes=([1], [0])))
case("inner-1d", "(vt-inner CX-A CX-B)", np.inner(A, B))
case("inner-2d", "(vt-inner CX-M2 CX-M2)", np.inner(M2, M2))
case("vdot", "(vt-vdot CX-A CX-B)", np.vdot(A, B))
case("correlate-full", "(vt-correlate CX-A CX-B :mode :full)",
     np.correlate(A, B, "full"))
case("correlate-valid", "(vt-correlate CX-C CX-B :mode :valid)",
     np.correlate(C, B, "valid"))
case("correlate-same", "(vt-correlate CX-C CX-B :mode :same)",
     np.correlate(C, B, "same"))
case("convolve-full", "(vt-convolve CX-A CX-B :mode :full)",
     np.convolve(A, B, "full"))
case("convolve-same", "(vt-convolve CX-C CX-B :mode :same)",
     np.convolve(C, B, "same"))
case("kron", "(vt-kron CX-A CX-B)", np.kron(A, B))
case("diff", "(vt-diff CX-C)", np.diff(C))
case("trapz", "(vt-trapz CX-C)", np.trapezoid(C))

# ---------------------------------------------------------------- dtype 视图
case_symbol("nanmean-dtype", "(vt-dtype (vt-nanmean CX-NA))", ":COMPLEX128")
case_symbol("nanvar-dtype", "(vt-dtype (vt-nanvar CX-NA))", ":FLOAT64")
case_symbol("nanstd-dtype", "(vt-dtype (vt-nanstd CX-NA))", ":FLOAT64")
case_symbol("nanmedian-dtype", "(vt-dtype (vt-nanmedian CX-NA))", ":COMPLEX128")
case_symbol("unique-dtype", "(vt-dtype (vt-unique CX-DUP))", ":COMPLEX128")
case_symbol("repeat-dtype", "(vt-dtype (vt-repeat CX-A 2))", ":COMPLEX128")
case_symbol("view-dtype", "(vt-dtype (vt-view CX-A nil :dtype :float64))", ":FLOAT64")
case_symbol("tensordot-dtype", "(vt-dtype (vt-tensordot CX-M2 CX-M2 :axes 1))",
            ":COMPLEX128")
case_symbol("c64-nansum-dtype",
            "(vt-dtype (vt-nansum (vt-astype CX-NA :complex64)))", ":COMPLEX64")
case_symbol("c64-nanvar-dtype",
            "(vt-dtype (vt-nanvar (vt-astype CX-NA :complex64)))", ":FLOAT32")
case_symbol("shape-view-2d", "(vt-shape (vt-view CX-M2 :dtype :float64))", "(2 4)")
case_symbol("shape-view-2d-explicit", "(vt-shape (vt-view CX-M2 nil :dtype :float64))", "(2 4)")
case_symbol("shape-tensordot-2", "(vt-shape (vt-tensordot CX-M2 CX-M2 :axes 2))", "NIL")


# ---------------------------------------------------------------- 第四轮：outer / any / all / take-along-axis
# vt-outer 的复数分支此前从未跑通（vt-reshape 多传了 :dtype），
# vt-any/vt-all 是 8 路实数特化、遇复数直接报错。
case("outer", "(vt-outer CX-A CX-B)", np.outer(A, B))
case("outer-c64", "(vt-outer (vt-astype CX-A :complex64) (vt-astype CX-B :complex64))",
     np.outer(A, B).astype(np.complex64))
case("outer-shape", "(vt-shape (vt-outer CX-A CX-B))", np.array([2, 2]), dtype="int64")
case_symbol("outer-dtype", "(vt-dtype (vt-outer CX-A CX-B))", ":COMPLEX128")

# any / all：复数的真值 = 任一分量非零（NaN 为真）
case("any-mixed", "(vt-any CX-C)", np.int8(np.any(C)), dtype=np.int8)
case("all-mixed", "(vt-all CX-C)", np.int8(np.all(C)), dtype=np.int8)
case("any-zero", "(vt-any CX-Z3)", np.int8(np.any(np.zeros(3, dtype=complex))),
     dtype=np.int8)
case("all-zero", "(vt-all CX-Z3)", np.int8(np.all(np.zeros(3, dtype=complex))),
     dtype=np.int8)
case("any-axis0", "(vt-any CX-M2 :axis 0)",
     np.any(M2, axis=0).astype(np.int8), dtype=np.int8)
case("all-axis1", "(vt-all CX-M2 :axis 1)",
     np.all(M2, axis=1).astype(np.int8), dtype=np.int8)
case("any-keepdims", "(vt-any CX-M2 :axis 1 :keepdims t)",
     np.any(M2, axis=1, keepdims=True).astype(np.int8), dtype=np.int8)
case("any-nan", "(vt-any CX-NANC)",
     np.int8(np.any(np.array([complex(np.nan, 0)]))), dtype=np.int8)
case("all-nan", "(vt-all CX-NANC)",
     np.int8(np.all(np.array([complex(np.nan, 0)]))), dtype=np.int8)
case("any-empty", "(vt-any CX-EMPTYC)",
     np.int8(np.any(np.array([], dtype=complex))), dtype=np.int8)
case("all-empty", "(vt-all CX-EMPTYC)",
     np.int8(np.all(np.array([], dtype=complex))), dtype=np.int8)
case_symbol("any-dtype", "(vt-dtype (vt-any CX-C))", ":INT8")

# take-along-axis：8 路元素类型分派此前漏了复数
case("take-along", "(vt-take-along-axis CX-M2 CX-IDXCOL 1)",
     np.take_along_axis(M2, np.array([[1], [0]]), axis=1))
case("take-along-axis0", "(vt-take-along-axis CX-M2 CX-IDXROW 0)",
     np.take_along_axis(M2, np.array([[1, 0]]), axis=0))


def emit():
    lines = []
    lines.append(";;; 由 test/gen_complex_ops_expected.py 自动生成，请勿手改。")
    lines.append(";;; 生成器：numpy %s" % np.__version__)
    lines.append("(:GENERATOR %s" % lit_str(np.__version__))
    lines.append(" :CASES")
    lines.append(" (")
    for c in CASES:
        lines.append(
            "  (:NAME %s :FORM %s :VALUE %s :DTYPE %s :SHAPE %s%s)"
            % (lit_str(c["name"]), lit_str(c["form"]), c["value"], c["dtype"],
               c["shape"], " :KIND :SYMBOL" if c.get("kind") == "symbol" else ""))
    lines.append(" )")
    lines.append(")")
    text = "\n".join(lines) + "\n"
    with open(OUT, "w") as f:
        f.write(text)
    print("wrote %s (%d cases, numpy %s)" % (OUT, len(CASES), np.__version__))


if __name__ == "__main__":
    emit()
