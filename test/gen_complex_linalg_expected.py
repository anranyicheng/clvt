#!/usr/bin/env python3
"""生成「复数线性代数」差分期望值（供 test/complex-linalg-test.lisp 使用）。

约定（与 gen_complex_ops_expected.py 一致）：
  * 期望值**只能**由 numpy 实测生成，禁止手写（CONVENTIONS §10.4）；
  * 复数写成 (:CX <re> <im>)，因为 `#C(:NAN 1d0)` 不是合法 Lisp 字面量；
  * dtype / 形状类断言用 :KIND :SYMBOL（期望值是关键字或整数表）。

用法：python3 test/gen_complex_linalg_expected.py [输出路径]
"""
import math
import re
import sys

import numpy as np

OUT = sys.argv[1] if len(sys.argv) > 1 else "test/complex-linalg-expected.sexp"

CASES = []


def real_lit(x):
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


def lit(x):
    if isinstance(x, (complex, np.complexfloating)):
        return "(:CX %s %s)" % (real_lit(x.real), real_lit(x.imag))
    if isinstance(x, (float, np.floating)):
        return real_lit(x)
    return str(int(x))


def val_lit(v):
    arr = np.asarray(v)
    if arr.ndim == 0:
        return lit(arr.item())
    return "(" + " ".join(lit(e) for e in arr.ravel()) + ")"


def dtype_kw(dt):
    return ":" + str(np.dtype(dt)).upper().replace("COMPLEX128", "COMPLEX128")


def shape_lit(v):
    return "(" + " ".join(str(d) for d in np.asarray(v).shape) + ")"


def case(name, form, value, dtype=None, tol=None):
    """数值型断言：VALUE 会被压平逐元素比较（相对容差 TOL）。"""
    arr = np.asarray(value)
    e = dict(name=name, form=form, value=val_lit(value),
             dtype=dtype_kw(dtype if dtype is not None else arr.dtype),
             shape=shape_lit(value))
    if tol is not None:
        e["tol"] = real_lit(tol)
    CASES.append(e)


def case_symbol(name, form, symbol, shape="NIL"):
    CASES.append(dict(name=name, form=form, value=symbol,
                      dtype=":SYMBOL", shape=shape, kind="symbol"))


# ---------------------------------------------------------------- 输入
A = np.array([[2 + 1j, 1 + 0j], [1 - 1j, 3 + 2j]], dtype=np.complex128)
H = np.array([[4 + 0j, 1 + 1j], [1 - 1j, 3 + 0j]], dtype=np.complex128)
W = np.array([[1 + 1j, 2 + 0j, 0 - 1j],
              [3 + 0j, 1 + 2j, 1 + 0j]], dtype=np.complex128)
T = W.T.copy()                                   # 3×2
B1 = np.array([1 + 0j, 2 + 0j], dtype=np.complex128)
B2 = np.eye(2, dtype=np.complex128)
V1 = np.array([1 + 2j, 3 - 4j, 5 + 0j], dtype=np.complex128)
V2 = np.array([2 - 1j, 1 + 1j, 0 + 2j], dtype=np.complex128)
M2 = np.array([[1 + 1j, 2 + 0j], [3 + 0j, 0 + 4j]], dtype=np.complex128)
SING = np.array([[1 + 0j, 2 + 0j], [2 + 0j, 4 + 0j]], dtype=np.complex128)

# ---------------------------------------------------------------- det / inv / solve
case("det", "(vt-item (vt-det CX-LA-A))", np.linalg.det(A))
case("det-singular", "(vt-item (vt-det CX-LA-SING))", np.linalg.det(SING))
case_symbol("det-shape", "(vt-shape (vt-det CX-LA-A))", "NIL")
case("inv", "(vt-inv CX-LA-A)", np.linalg.inv(A))
case("solve-1d", "(vt-solve CX-LA-A CX-LA-B1)", np.linalg.solve(A, B1))
case("solve-2d", "(vt-solve CX-LA-A CX-LA-B2)", np.linalg.solve(A, B2))
case("inv-of-square-product", "(vt-inv (vt-matmul CX-LA-A CX-LA-A))",
     np.linalg.inv(A @ A))

# ---------------------------------------------------------------- cholesky
case("chol", "(vt-cholesky CX-LA-H)", np.linalg.cholesky(H))
# :upper 返回 L^H（共轭转置），不是 L^T —— 复数下二者不同
case("chol-upper", "(vt-cholesky CX-LA-H :upper t)",
     np.linalg.cholesky(H).conj().T)
case_symbol("chol-upper-dtype", "(vt-dtype (vt-cholesky CX-LA-H :upper t))", ":COMPLEX128")

# ---------------------------------------------------------------- SVD
_u, _s, _vh = np.linalg.svd(A)
case("svd-s-sq", "(nth-value 1 (vt-svd CX-LA-A))", _s, dtype="float64")
case("svd-s-tall", "(nth-value 1 (vt-svd CX-LA-T))", np.linalg.svd(T)[1],
     dtype="float64")
case("svd-s-wide", "(nth-value 1 (vt-svd CX-LA-W))", np.linalg.svd(W)[1],
     dtype="float64")
case_symbol("svd-s-dtype", "(vt-dtype (nth-value 1 (vt-svd CX-LA-A)))", ":FLOAT64")
case_symbol("svd-u-shape-sq", "(vt-shape (nth-value 0 (vt-svd CX-LA-A)))", "(2 2)")
case_symbol("svd-u-shape-tall", "(vt-shape (nth-value 0 (vt-svd CX-LA-T)))", "(3 2)")
case_symbol("svd-vh-shape-wide", "(vt-shape (nth-value 2 (vt-svd CX-LA-W)))", "(2 3)")
case_symbol("svd-full-u-shape", "(vt-shape (nth-value 0 (vt-svd CX-LA-T :full-matrices t)))",
            "(3 3)")
case_symbol("svd-full-vh-shape", "(vt-shape (nth-value 2 (vt-svd CX-LA-T :full-matrices t)))",
            "(2 2)")

# ---------------------------------------------------------------- 特征值（厄米特）
case("eig-h-vals", "(nth-value 0 (vt-eig CX-LA-H))",
     np.linalg.eigvalsh(H)[::-1], dtype="float64")     # clvt 契约：降序
case("eig-a-vals", "(nth-value 0 (vt-eig CX-LA-A))",
     np.linalg.eigvalsh(np.tril(A) + np.tril(A, -1).conj().T)[::-1],
     dtype="float64")
case("eigvalsh-asc", "(vt-eigvalsh CX-LA-H)", np.linalg.eigvalsh(H), dtype="float64")
case_symbol("eig-vals-dtype", "(vt-dtype (nth-value 0 (vt-eig CX-LA-H)))", ":FLOAT64")
case_symbol("eig-vecs-dtype", "(vt-dtype (nth-value 1 (vt-eig CX-LA-H)))", ":COMPLEX128")

# ---------------------------------------------------------------- pinv / lstsq / rank / cond
case("pinv", "(vt-pinv CX-LA-A)", np.linalg.pinv(A))
case("pinv-tall", "(vt-pinv CX-LA-T)", np.linalg.pinv(T))
_sol = np.linalg.lstsq(T, np.array([1 + 0j, 2 + 0j, 3 + 0j]), rcond=None)
case("lstsq-x", "(nth-value 0 (vt-lstsq CX-LA-T CX-LA-B3))", _sol[0])
# residual 已对齐 numpy：**平方和**（不是 2-范数），形状 (K,)。
# 旧版本此处取 sqrt 并写死「2-范数是库内契约」（D-C13），§17.10.22 已废除。
case("lstsq-res", "(nth-value 1 (vt-lstsq CX-LA-T CX-LA-B3))",
     np.asarray(_sol[1], dtype=np.float64))
case("lstsq-rank", "(nth-value 2 (vt-lstsq CX-LA-T CX-LA-B3))",
     np.array(_sol[2]), dtype="int64")
case("rank", "(vt-matrix-rank CX-LA-A)", np.array(np.linalg.matrix_rank(A)),
     dtype="int64")
case("rank-singular", "(vt-matrix-rank CX-LA-SING)",
     np.array(np.linalg.matrix_rank(SING)), dtype="int64")
case("cond", "(vt-cond CX-LA-A)", np.linalg.cond(A), dtype="float64")
case("cond-p1", "(vt-cond CX-LA-A :p 1)", np.linalg.cond(A, 1), dtype="float64")
case("cond-inf", "(vt-cond CX-LA-A :p :inf)", np.linalg.cond(A, np.inf), dtype="float64")

# ---------------------------------------------------------------- matrix-power
case("pow2", "(vt-matrix-power CX-LA-A 2)", np.linalg.matrix_power(A, 2))
case("pow0", "(vt-matrix-power CX-LA-A 0)", np.linalg.matrix_power(A, 0))
case("pow-neg1", "(vt-matrix-power CX-LA-A -1)", np.linalg.matrix_power(A, -1))
case("pow3-tall-ish", "(vt-matrix-power CX-LA-M2 3)", np.linalg.matrix_power(M2, 3))

# ---------------------------------------------------------------- einsum
case("einsum-dot", "(vt-einsum \"i,i->\" CX-LA-V1 CX-LA-V2)",
     np.einsum("i,i->", V1, V2))
case("einsum-trace", "(vt-einsum \"ii->\" CX-LA-M2)", np.einsum("ii->", M2))
case("einsum-sum", "(vt-einsum \"ij->\" CX-LA-M2)", np.einsum("ij->", M2))
case("einsum-matmul", "(vt-einsum \"ij,jk->ik\" CX-LA-M2 CX-LA-M2)",
     np.einsum("ij,jk->ik", M2, M2))
case("einsum-outer", "(vt-einsum \"i,j->ij\" CX-LA-V1 CX-LA-V2)",
     np.einsum("i,j->ij", V1, V2))
case_symbol("einsum-dtype-c64",
            "(vt-dtype (vt-einsum \"i,i->\" (vt-astype CX-LA-V1 :complex64) (vt-astype CX-LA-V2 :complex64)))",
            ":COMPLEX64")

# ---------------------------------------------------------------- 范数 / 迹
case("norm-frob", "(vt-item (vt-frobenius-norm CX-LA-A))",
     np.linalg.norm(A), dtype="float64")
case("trace", "(vt-item (vt-trace CX-LA-A))", np.trace(A))


def lit_str(s):
    return '"' + s.replace("\\", "\\\\").replace('"', '\\"') + '"'


def emit():
    lines = []
    lines.append(";;; 由 test/gen_complex_linalg_expected.py 自动生成，请勿手改。")
    lines.append(";;; 生成器：numpy %s" % np.__version__)
    lines.append("(:GENERATOR %s" % lit_str(np.__version__))
    lines.append(" :CASES")
    lines.append(" (")
    for c in CASES:
        extra = ""
        if c.get("kind") == "symbol":
            extra = " :KIND :SYMBOL"
        elif c.get("tol"):
            extra = " :TOL %s" % c["tol"]
        lines.append(
            "  (:NAME %s :FORM %s :VALUE %s :DTYPE %s :SHAPE %s%s)"
            % (lit_str(c["name"]), lit_str(c["form"]), c["value"],
               c["dtype"], c["shape"], extra))
    lines.append(" )")
    lines.append(")")
    text = "\n".join(lines) + "\n"
    with open(OUT, "w") as f:
        f.write(text)
    print("wrote %s (%d cases, numpy %s)" % (OUT, len(CASES), np.__version__))


emit()
