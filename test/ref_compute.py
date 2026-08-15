#!/usr/bin/env python3
"""clvt 参考计算：用 numpy 与 pytorch 生成确定性参考结果，输出 JSON 到 stdout。"""
import json
import numpy as np
import torch

R = {}

def add(key, val):
    if isinstance(val, np.ndarray):
        val = val.tolist()
    if isinstance(val, torch.Tensor):
        val = val.detach().cpu().tolist()
    if isinstance(val, (np.floating,)):
        val = float(val)
    if isinstance(val, (np.integer,)):
        val = int(val)
    R[key] = val

# ---------------- 创建 ----------------
add("arange", np.arange(10))
add("linspace", np.linspace(0.0, 1.0, 5))
add("logspace", np.logspace(0, 3, 4))
add("eye", np.eye(3))
add("diag", np.diag([1, 2, 3]))

# ---------------- 逐元素（numpy） ----------------
a = np.array([1.0, 2.0, 3.0, 4.0])
b = np.array([5.0, 6.0, 7.0, 8.0])
x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
add("add", a + b)
add("sub", b - a)
add("mul", a * b)
add("div", b / a)
add("sin", np.sin(x))
add("cos", np.cos(x))
add("tanh", np.tanh(x))
add("exp", np.exp(x))
add("log", np.log(np.array([1.0, 2.0, 3.0, 4.0])))
add("sqrt", np.sqrt(np.array([1.0, 4.0, 9.0, 16.0])))
add("abs", np.abs(np.array([-3.0, -1.0, 0.0, 1.0, 3.0])))
add("clip", np.clip(np.array([1.0, 2.0, 3.0, 4.0]), 2.0, 3.0))
add("maximum", np.maximum(a, b))
add("minimum", np.minimum(a, b))
add("pow", np.power(a, 2.0))
add("reciprocal", np.reciprocal(a))
add("cbrt", np.cbrt(np.array([-8.0, -1.0, 0.0, 1.0, 8.0])))

# ---------------- 归约 / 统计（numpy） ----------------
m = np.arange(12, dtype=np.float64).reshape(3, 4)
add("sum_all", float(m.sum()))
add("sum_axis0", m.sum(axis=0))
add("sum_axis1", m.sum(axis=1))
add("mean_all", float(m.mean()))
add("mean_axis1", m.mean(axis=1))
add("var_all", float(m.var()))
add("std_all", float(m.std()))
add("max_axis0", m.max(axis=0))
add("min_axis1", m.min(axis=1))
add("argmax_axis1", m.argmax(axis=1))
add("argmin_axis0", m.argmin(axis=0))
add("prod_all", float(m.prod()))
add("cumsum", np.cumsum(np.array([1.0, 2.0, 3.0, 4.0])))
add("cumprod", np.cumprod(np.array([1.0, 2.0, 3.0, 4.0])))
add("median", float(np.median(np.array([3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0]))))
add("percentile", float(np.percentile(np.array([1.0, 2.0, 3.0, 4.0, 5.0]), 90)))
add("ptp", float(np.ptp(np.array([3.0, 1.0, 4.0, 1.0, 5.0]))))
add("sort", np.sort(np.array([3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0])))
add("argsort", np.argsort(np.array([3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0])))
add("diff", np.diff(np.array([1.0, 3.0, 6.0, 10.0, 15.0])))

# ---------------- 线性代数（numpy） ----------------
M = np.array([[1.0, 2.0], [3.0, 4.0]])
N = np.array([[5.0, 6.0], [7.0, 8.0]])
add("matmul", M @ N)
add("dot", float(np.dot(np.array([1.0, 2.0, 3.0]), np.array([4.0, 5.0, 6.0]))))
add("outer", np.outer(np.array([1.0, 2.0, 3.0]), np.array([4.0, 5.0])))
add("trace", float(np.trace(np.arange(9, dtype=np.float64).reshape(3, 3))))
add("norm", float(np.linalg.norm(np.array([3.0, 4.0]))))
add("det", float(np.linalg.det(M)))
add("inv", np.linalg.inv(M))
add("solve", np.linalg.solve(np.array([[2.0, 1.0], [1.0, 3.0]]), np.array([7.0, 8.0])))
add("cholesky", np.linalg.cholesky(np.array([[4.0, 2.0], [2.0, 3.0]])))
add("eigvals", np.sort(np.linalg.eigvalsh(np.array([[2.0, 1.0], [1.0, 3.0]])))[::-1])
add("matrix_rank", int(np.linalg.matrix_rank(np.array([[1.0, 2.0], [2.0, 4.0]]))))
add("einsum_dot", float(np.einsum("i,i->", np.array([1.0, 2.0, 3.0]), np.array([4.0, 5.0, 6.0]))))
add("einsum_outer", np.einsum("i,j->ij", np.array([1.0, 2.0, 3.0]), np.array([4.0, 5.0])))
add("einsum_trace", float(np.einsum("ii->", np.arange(9, dtype=np.float64).reshape(3, 3))))

# ---------------- 神经网络（numpy 近似 + torch） ----------------
xt = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=torch.float64)
add("sigmoid", torch.sigmoid(xt))
add("relu", torch.relu(xt))
add("gelu", torch.nn.functional.gelu(xt, approximate="tanh"))
add("softmax", torch.softmax(torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64), dim=-1))
add("log_softmax", torch.log_softmax(torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64), dim=-1))

# ---------------- 集合（numpy） ----------------
add("unique", np.unique(np.array([1, 2, 2, 3, 3, 3])))
add("intersect1d", np.intersect1d(np.array([1, 2, 3, 4, 5]), np.array([3, 4, 5, 6, 7])))

# ---------------- 形状（numpy） ----------------
add("reshape", np.arange(6).reshape(2, 3))
add("transpose", np.arange(6).reshape(2, 3).T)
add("concatenate", np.concatenate([np.array([[1, 2], [3, 4]]), np.array([[5, 6], [7, 8]])], axis=0))
add("broadcast", np.broadcast_to(np.array([1, 2, 3]), (2, 3)))

# ---------------- torch topk ----------------
tk = torch.tensor([[1.0, 5.0, 3.0], [7.0, 2.0, 9.0]], dtype=torch.float64)
tv, ti = torch.topk(tk, 2, dim=-1)
add("topk_vals", tv)
add("topk_idx", ti)

print(json.dumps(R))
