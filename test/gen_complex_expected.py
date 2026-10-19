#!/usr/bin/env python3
"""gen_complex_expected.py — 复数语义差分测试的期望值生成器

用法：python3 test/gen_complex_expected.py [输出路径]

输出一个 Lisp 可读的数据文件（默认 test/complex-expected.sexp），形如：

  (:GENERATOR "numpy 2.4.6" :CASES
   ((:NAME "add" :FORM "(vt-+ CX-A CX-B)" :DTYPE :COMPLEX128 :SHAPE (2)
     :VALUE (#C(3.0d0 1.0d0) #C(4.0d0 -3.0d0)))
    ...))

其中
  :FORM  是要在 clvt 包里求值的形式（可以引用 CX-A / CX-B 等预置张量）
  :VALUE 是 numpy 实测结果，复数写成 #C(re im)，NaN/±Inf 写成
         :NAN / :INF / :NEG-INF（Lisp 侧按目标 dtype 还原）。
标量结果的 :SHAPE 为 NIL。

设计原则：**每一格期望值都来自本机 numpy 实测**，不允许手写（CONVENTIONS
§10.4「每个新增/修改函数的语义必须在对照测试中有对应用例」）。
"""

import math
import re
import sys
import numpy as np

OUT = sys.argv[1] if len(sys.argv) > 1 else "test/complex-expected.sexp"

# 与 Lisp 侧 test/complex-test.lisp 的预置变量保持一致
A = np.array([1 + 2j, 3 - 4j])
B = np.array([2 - 1j, 1 + 1j])
C = np.array([3 + 1j, 1 + 5j, 1 + 2j])          # 排序 / 极值用例
D = np.array([1 + 2j, np.nan + 0j, np.inf + 0j])  # NaN / Inf 用例


def lit(x):
    """numpy 值 → Lisp 可读数文本。

    复数写成 (:CX <re> <im>)，而**不是** #C(...)：`#C(:NAN 1d0)` 不是合法
    Lisp 字面量（#C 的读入要求两个实数），而 numpy 的复结果里 NaN/Inf
    出现在分量上是常态（如 sort 的 NaN 末尾用例）。Lisp 侧读入后由
    NORMALIZE-VALUE 把 (:CX re im) 还原成 complex。
    """
    if isinstance(x, (complex, np.complexfloating)):
        return "(:CX %s %s)" % (real_lit(x.real), real_lit(x.imag))
    if isinstance(x, (float, np.floating)):
        return real_lit(x)
    return str(int(x))


def real_lit(x):
    """实数 → Lisp double-float 字面量（NaN/±Inf 用关键字哨兵）。"""
    x = float(x)
    if math.isnan(x):
        return ":NAN"
    if math.isinf(x):
        return ":INF" if x > 0 else ":NEG-INF"
    if x == int(x) and abs(x) < 1e15:
        return "%sd0" % int(x)
    r = repr(x)
    if "e" in r or "E" in r:
        mant, exp = re.split("[eE]", r)
        if "." not in mant:
            mant += ".0"
        return "%sd%s" % (mant, exp if exp.startswith(("+", "-")) else "+" + exp)
    return r + "d0"


def val_lit(v):
    """数组/标量 → Lisp 值文本（元素用 lit()，整体为列表）。"""
    arr = np.asarray(v)
    if arr.ndim == 0:
        return lit(arr.item())
    return "(" + " ".join(lit(e) for e in arr.ravel()) + ")"


def dtype_kw(dt):
    return ":" + str(np.dtype(dt)).upper()


def shape_lit(shape):
    if not shape:
        return "NIL"
    return "(" + " ".join(str(int(d)) for d in shape) + ")"


CASES = []


def case(name, form, value, dtype=None, shape=None):
    arr = np.asarray(value)
    CASES.append(
        dict(
            name=name,
            form=form,
            value=val_lit(value),
            dtype=dtype_kw(dtype if dtype is not None else arr.dtype),
            shape=shape_lit(arr.shape if shape is None else shape),
        )
    )


def case_symbol(name, form, symbol):
    """dtype/符号类断言：期望值是关键字而非数值。"""
    CASES.append(dict(name=name, form=form, value=symbol, dtype=":SYMBOL",
                      shape="NIL", kind="symbol"))


def c64(x):
    return np.asarray(x, dtype=np.complex64)


# ---------------------------------------------------------------- 算术
case("add", "(vt-+ CX-A CX-B)", A + B)
case("sub", "(vt-- CX-A CX-B)", A - B)
case("mul", "(vt-* CX-A CX-B)", A * B)
case("div", "(vt-/ CX-A CX-B)", A / B)
case("neg", "(vt-negative CX-A)", -A)
case("square", "(vt-square CX-A)", A ** 2)
case("pow-int", "(vt-pow CX-A 2)", A ** 2)
case("pow-half", "(vt-pow CX-A 0.5d0)", A ** 0.5)
case("add-real-scalar", "(vt-+ CX-A 1.0d0)", A + 1.0)
case("mul-imag-unit", "(vt-* CX-A #C(0.0d0 1.0d0))", A * 1j)
case("mixed-float-vec", "(vt-+ CX-A (vt-from-sequence (list 1.0d0 2.0d0)))", A + np.array([1.0, 2.0]))
case("reciprocal", "(vt-reciprocal CX-A)", np.reciprocal(A))
case("signum", "(vt-signum CX-A)", np.sign(A))
case("round", "(vt-round CX-A)", np.round(A))   # numpy 2.x：分量各自取整
# complex64 保持（numpy：complex64+complex64 → complex64）
case("add-c64", "(vt-+ CX-C64-A CX-C64-A)", c64(A) + c64(A), dtype=np.complex64)

# ---------------------------------------------------------------- 一元数学
for name, npf in [
    ("sqrt", np.sqrt),
    ("exp", np.exp),
    ("log", np.log),
    ("log2", np.log2),
    ("log10", np.log10),
    ("sin", np.sin),
    ("cos", np.cos),
    ("tan", np.tan),
    ("asin", np.arcsin),
    ("acos", np.arccos),
    ("atan", np.arctan),
    ("sinh", np.sinh),
    ("cosh", np.cosh),
    ("tanh", np.tanh),
    ("asinh", np.arcsinh),
    ("acosh", np.arccosh),
    ("atanh", np.arctanh),
]:
    case(name, "(vt-%s CX-A)" % name, npf(A))

# ---------------------------------------------------------------- 分量 / 派生
case("abs", "(vt-abs CX-A)", np.abs(A))
case("real", "(vt-real CX-A)", A.real)
case("imag", "(vt-imag CX-A)", A.imag)
case("conj", "(vt-conj CX-A)", np.conj(A))
case("angle", "(vt-angle CX-A)", np.angle(A))
case("angle-deg", "(vt-angle CX-A :deg t)", np.angle(A, deg=True))
case("abs-c64", "(vt-abs CX-C64-A)", np.abs(c64(A)))
case("conj-c64", "(vt-conj CX-C64-A)", np.conj(c64(A)))

# ---------------------------------------------------------------- 比较
case("eq", "(vt-= CX-A CX-A)", (A == A).astype(np.int8), dtype=np.int8)
case("ne", "(vt-/= CX-A CX-B)", (A != B).astype(np.int8), dtype=np.int8)
case("lt", "(vt-< CX-A CX-B)", (A < B).astype(np.int8), dtype=np.int8)
case("le", "(vt-<= CX-C CX-C)", (C <= C).astype(np.int8), dtype=np.int8)
case("gt", "(vt-> CX-C (vt-const (list 3) #C(1.0d0 2.0d0) :dtype :complex128))",
     (C > np.array([1 + 2j] * 3)).astype(np.int8), dtype=np.int8)
case("ge", "(vt->= CX-C CX-C)", (C >= C).astype(np.int8), dtype=np.int8)
case("lt-lex-imag", "(vt-< (vt-const (list 1) #C(1.0d0 1.0d0) :dtype :complex128) (vt-const (list 1) #C(1.0d0 2.0d0) :dtype :complex128))",
     (np.array([1 + 1j]) < np.array([1 + 2j])).astype(np.int8), dtype=np.int8)

# ---------------------------------------------------------------- 判定
case("eq-nan", "(vt-= CX-D CX-D)", (D == D).astype(np.int8), dtype=np.int8)
case("ne-nan", "(vt-/= CX-D CX-D)", (D != D).astype(np.int8), dtype=np.int8)
case("isnan", "(vt-isnan CX-D)", np.isnan(D).astype(np.int8), dtype=np.int8)
case("isinf", "(vt-isinf CX-D)", np.isinf(D).astype(np.int8), dtype=np.int8)
case("isfinite", "(vt-isfinite CX-D)", np.isfinite(D).astype(np.int8), dtype=np.int8)

# ---------------------------------------------------------------- 极值 / 排序
case("maximum", "(vt-maximum CX-A CX-B)", np.maximum(A, B))
case("minimum", "(vt-minimum CX-A CX-B)", np.minimum(A, B))
case("amax", "(vt-amax CX-C)", np.amax(C))
case("amin", "(vt-amin CX-C)", np.amin(C))
case("argmax", "(vt-argmax CX-C)", np.argmax(C), dtype=np.int64)
case("argmin", "(vt-argmin CX-C)", np.argmin(C), dtype=np.int64)
case("sort", "(vt-sort CX-C)", np.sort(C))
case("argsort", "(vt-argsort CX-C)", np.argsort(C), dtype=np.int64)
case("sort-nan", "(vt-sort CX-D)", np.sort(D))   # NaN 排末尾

# ---------------------------------------------------------------- 归约
case("sum", "(vt-sum CX-A)", np.sum(A))
case("prod", "(vt-prod CX-A)", np.prod(A))
case("mean", "(vt-mean CX-A)", np.mean(A))
case("var", "(vt-var CX-A)", np.var(A))
case("std", "(vt-std CX-A)", np.std(A))
case("var-ddof", "(vt-var CX-A :ddof 1)", np.var(A, ddof=1))
M = np.array([[1 + 1j, 2 + 0j], [3 + 0j, 0 + 4j]])
case("sum-axis1", "(vt-sum CX-M :axis 1)", np.sum(M, axis=1))
case("sum-axis0-keepdims", "(vt-sum CX-M :axis 0 :keepdims t)",
     np.sum(M, axis=0, keepdims=True))
case("cumsum", "(vt-cumsum CX-A)", np.cumsum(A))
case("cumprod", "(vt-cumprod CX-A)", np.cumprod(A))
case("norm", "(vt-norm CX-A)", np.linalg.norm(A))
case("sum-c64", "(vt-sum CX-C64-A)", np.sum(c64(A)), dtype=np.complex64)

# ---------------------------------------------------------------- 线性代数
case("dot", "(vt-dot CX-A CX-B)", np.dot(A, B))
case("matmul", "(vt-matmul CX-M CX-M)",
     np.matmul(np.array([[1 + 1j, 2 + 0j], [3 + 0j, 0 + 4j]]),
               np.array([[1 + 1j, 2 + 0j], [3 + 0j, 0 + 4j]])))
case("trace", "(vt-trace CX-M)",
     np.trace(np.array([[1 + 1j, 2 + 0j], [3 + 0j, 0 + 4j]])))

# ---------------------------------------------------------------- dtype 视图
case_symbol("abs-dtype-c128", "(vt-dtype (vt-abs CX-A))", ":FLOAT64")
case_symbol("var-dtype", "(vt-dtype (vt-var CX-A))", ":FLOAT64")
case_symbol("promote-c64-i16", "(vt-promote-type :complex64 :int16)", ":COMPLEX64")
case_symbol("promote-c64-i32", "(vt-promote-type :complex64 :int32)", ":COMPLEX128")
case_symbol("promote-c64-f64", "(vt-promote-type :complex64 :float64)", ":COMPLEX128")
case_symbol("promote-c64-c128", "(vt-promote-type :complex64 :complex128)", ":COMPLEX128")
case_symbol("array-dtype-complex",
            "(vt-dtype (vt-from-sequence (list #C(1.0d0 2.0d0)) :dtype :complex128))",
            ":COMPLEX128")


def emit():
    """写出 Lisp 数据文件：(:GENERATOR "x" :CASES ( 用例... ))。"""
    lines = []
    lines.append(";;; 由 test/gen_complex_expected.py 自动生成，请勿手改。")
    lines.append(";;; 生成器：numpy %s" % np.__version__)
    lines.append("(:GENERATOR %s" % lit_str(np.__version__))
    lines.append(" :CASES")
    lines.append(" (")
    for c in CASES:
        lines.append(
            "  (:NAME %s :FORM %s :VALUE %s :DTYPE %s :SHAPE %s%s)"
            % (
                lit_str(c["name"]),
                lit_str(c["form"]),
                c["value"],
                c["dtype"],
                c["shape"],
                " :KIND :SYMBOL" if c.get("kind") == "symbol" else "",
            )
        )
    lines.append(" )")
    lines.append(")")
    text = "\n".join(lines) + "\n"
    with open(OUT, "w") as f:
        f.write(text)
    print("wrote %s (%d cases, numpy %s)" % (OUT, len(CASES), np.__version__))


def lit_str(s):
    return '"' + s.replace('"', '\\"') + '"'


if __name__ == "__main__":
    emit()
