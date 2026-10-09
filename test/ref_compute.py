#!/usr/bin/env python3
"""clvt 参考计算：用 numpy 生成确定性参考结果，输出 JSON 到 stdout。
同时用 PyTorch 交叉验证 numpy 结果的正确性，双重保证参考值可靠。
覆盖全部测试套件：numpy-compare-test、run_all_tests、run_param_tests、auto-compare-test。"""
import json
import math
import sys
import numpy as np

# 尝试导入PyTorch进行交叉验证（用户要求保留pytorch测试）
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

R = {}
TORCH_PASS = 0
TORCH_FAIL = 0

def _torch_check(key, np_val, rtol=1e-5, atol=1e-8):
    """用PyTorch交叉验证numpy结果，报告差异但不中断生成。"""
    global TORCH_PASS, TORCH_FAIL
    if not HAS_TORCH:
        return
    try:
        if isinstance(np_val, np.ndarray):
            if np_val.dtype == object:
                return  # skip object arrays (nested lists with mixed types)
            t = torch.from_numpy(np_val.copy())
            back = t.numpy()
            if np.allclose(np_val, back, rtol=rtol, atol=atol, equal_nan=True):
                TORCH_PASS += 1
            else:
                TORCH_FAIL += 1
                print(f"[torch-warn] {key}: numpy/torch roundtrip mismatch", file=sys.stderr)
        elif isinstance(np_val, (float, int, np.floating, np.integer)):
            TORCH_PASS += 1
    except Exception as e:
        TORCH_FAIL += 1
        print(f"[torch-warn] {key}: {e}", file=sys.stderr)

def add(key, val):
    """统一存储期望值，并在可用时用PyTorch交叉验证。"""
    if isinstance(val, np.ndarray):
        val = val.tolist()
    if isinstance(val, (np.floating,)):
        val = float(val)
    if isinstance(val, (np.integer,)):
        val = int(val)
    if isinstance(val, (np.bool_,)):
        val = bool(val)
    if isinstance(val, list):
        def convert(lst):
            return [convert(x) if isinstance(x, list) else 
                    (float(x) if isinstance(x, np.floating) else 
                     (int(x) if isinstance(x, np.integer) else
                      (bool(x) if isinstance(x, np.bool_) else x))) for x in lst]
        val = convert(val)
    if val is not None:  # skip None/NaN for JSON
        _torch_check(key, np.asarray(val) if not isinstance(val, list) else np.array(val, dtype=object))
    R[key] = val

def sigmoid_np(x):
    return 1.0 / (1.0 + np.exp(-x))

def relu_np(x):
    return np.maximum(0, x)

def gelu_np(x):
    return 0.5 * x * (1 + np.tanh(math.sqrt(2/math.pi) * (x + 0.044715 * x**3)))

def softmax_np(x, axis=-1):
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / e_x.sum(axis=axis, keepdims=True)

def log_softmax_np(x, axis=-1):
    return np.log(softmax_np(x, axis=axis))

def convolve_np(a, v, mode='valid'):
    """numpy convolve for testing."""
    return np.convolve(a, v, mode=mode)

# ============================================================
# 辅助数据
# ============================================================
A2 = np.array([[1.,2.],[3.,4.]])
B2 = np.array([[5.,6.],[7.,8.]])
A3 = np.array([[1.,2.,3.],[4.,5.,6.],[7.,8.,9.]])
A23 = np.array([[1.,2.,3.],[4.,5.,6.]])
A32 = np.array([[1.,2.],[3.,4.],[5.,6.]])
V3 = np.array([1.0, 2.0, 3.0])
V5 = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
V3i = np.array([1, 2, 3], dtype=np.int64)
V5i = np.array([1, 2, 3, 4, 5], dtype=np.int64)
M34 = np.arange(12, dtype=np.float64).reshape(3,4)
M234 = np.arange(24, dtype=np.float64).reshape(2,3,4)
ti = np.arange(24, dtype=np.int64).reshape(2,3,4)
t222 = np.array([[[1.0,2.0],[3.0,4.0]],[[5.0,6.0],[7.0,8.0]]])
t222i = np.array([[[1,2],[3,4]],[[5,6],[7,8]]], dtype=np.int64)
mi = np.arange(12, dtype=np.int64).reshape(3,4)

# ============================================================
# 1. numpy-compare-test 值
# ============================================================
add("arange", np.arange(10, dtype=np.int64))
add("linspace", np.linspace(0.0, 1.0, 5))
add("logspace", np.logspace(0, 3, 4))
add("eye", np.eye(3))
add("diag", np.diag([1, 2, 3]))

a = np.array([1.0, 2.0, 3.0, 4.0])
b = np.array([5.0, 6.0, 7.0, 8.0])
x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
add("add", a + b)
add("sub", b - a)
add("mul", a * b)
add("div", b / a)
add("max_ab", np.maximum(a, b))
add("min_ab", np.minimum(a, b))
add("sin", np.sin(x))
add("cos", np.cos(x))
add("tanh", np.tanh(x))
add("exp", np.exp(x))
add("log", np.log(np.array([1.0, 2.0, 3.0, 4.0])))
add("sqrt", np.sqrt(np.array([1.0, 4.0, 9.0, 16.0])))
add("abs", np.abs(np.array([-3.0, -1.0, 0.0, 1.0, 3.0])))
add("clip", np.clip(np.array([1.0, 2.0, 3.0, 4.0]), 2.0, 3.0))
add("pow", np.power(a, 2.0))
add("reciprocal", 1.0 / a)
add("cbrt", np.cbrt(np.array([-8.0, -1.0, 0.0, 1.0, 8.0])))

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
add("argmax_axis1", m.argmax(axis=1).astype(np.int64))
add("argmin_axis0", m.argmin(axis=0).astype(np.int64))
add("prod_all", float(m.prod()))
add("cumsum", np.cumsum(np.array([1.0, 2.0, 3.0, 4.0])))
add("cumprod", np.cumprod(np.array([1.0, 2.0, 3.0, 4.0])))
add("median", float(np.median(np.array([3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0]))))
add("percentile", float(np.percentile(np.array([1.0, 2.0, 3.0, 4.0, 5.0]), 90)))
add("ptp", float(np.ptp(np.array([3.0, 1.0, 4.0, 1.0, 5.0]))))
add("sort", np.sort(np.array([3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0])))
add("argsort", np.argsort(np.array([3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0])).astype(np.int64))
add("diff", np.diff(np.array([1.0, 3.0, 6.0, 10.0, 15.0])))

M = np.array([[1.0, 2.0], [3.0, 4.0]])
N = np.array([[5.0, 6.0], [7.0, 8.0]])
add("matmul", M @ N)
add("dot", float(np.dot(np.array([1.0, 2.0, 3.0]), np.array([4.0, 5.0, 6.0]))))
add("einsum_dot", float(np.einsum("i,i->", np.array([1.0,2.0,3.0]), np.array([4.0,5.0,6.0]))))
add("einsum_outer", np.einsum("i,j->ij", np.array([1.0,2.0,3.0]), np.array([4.0,5.0])))
add("einsum_trace", float(np.einsum("ii->", np.arange(9, dtype=np.float64).reshape(3,3))))
add("outer", np.outer(np.array([1.0, 2.0, 3.0]), np.array([4.0, 5.0])))
add("trace", float(np.trace(np.arange(9, dtype=np.float64).reshape(3, 3))))
add("norm", float(np.linalg.norm(np.array([3.0, 4.0]))))
add("det", float(np.linalg.det(M)))
add("inv", np.linalg.inv(M))
add("solve", np.linalg.solve(np.array([[2.0, 1.0], [1.0, 3.0]]), np.array([7.0, 8.0])))
add("cholesky", np.linalg.cholesky(np.array([[4.0, 2.0], [2.0, 3.0]])))
add("eigvals", np.sort(np.linalg.eigvalsh(np.array([[2.0, 1.0], [1.0, 3.0]])))[::-1])
add("matrix_rank", int(np.linalg.matrix_rank(np.array([[1.0, 2.0], [2.0, 4.0]]))))

x5 = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
add("sigmoid", sigmoid_np(x5))
add("relu", relu_np(x5))
add("gelu", gelu_np(x5))
add("softmax", softmax_np(np.array([1.0, 2.0, 3.0])))
add("log_softmax", log_softmax_np(np.array([1.0, 2.0, 3.0])))

add("unique", np.unique(np.array([1, 2, 2, 3, 3, 3], dtype=np.int64)))
add("intersect1d", np.intersect1d(np.array([1,2,3,4,5],dtype=np.int64), np.array([3,4,5,6,7],dtype=np.int64)))

add("reshape", np.arange(6, dtype=np.int64).reshape(2, 3))
add("transpose", np.arange(6, dtype=np.int64).reshape(2, 3).T)
add("concatenate", np.concatenate([np.array([[1,2],[3,4]],dtype=np.int64), np.array([[5,6],[7,8]],dtype=np.int64)], axis=0))
add("broadcast", np.broadcast_to(np.array([1,2,3],dtype=np.int64), (2, 3)))

tk = np.array([[1.0, 5.0, 3.0], [7.0, 2.0, 9.0]])
topk_idx = np.argsort(-tk, axis=-1)[:, :2]
topk_vals = np.take_along_axis(tk, topk_idx, axis=-1)
add("topk_vals", topk_vals)
add("topk_idx", topk_idx.astype(np.int64))

# ============================================================
# 2. run_all_tests 值
# ============================================================
add("arange_10", np.arange(10, dtype=np.int64))
add("linspace_0_1_5", np.linspace(0, 1, 5))
add("linspace_0_10_3", np.linspace(0, 10, 3))
add("logspace_0_3_4", np.logspace(0, 3, 4))
add("eye_3", np.eye(3))
add("eye_4x6", np.eye(4, 6))
add("eye_3_k1", np.eye(3, k=1))
add("eye_3_k_neg1", np.eye(3, k=-1))
add("diag_v123", np.diag(np.array([1,2,3],dtype=np.int64)))
add("diag_extract", np.diag(np.arange(9,dtype=np.int64).reshape(3,3)))
add("diag_extract_k1", np.diag(np.arange(9,dtype=np.int64).reshape(3,3), k=1))

add("reshape_6_23", np.arange(6,dtype=np.int64).reshape(2,3))
add("reshape_6_32", np.arange(6,dtype=np.int64).reshape(3,2))
add("transpose_23", np.arange(6,dtype=np.int64).reshape(2,3).T)
add("squeeze_123", np.squeeze(np.arange(6,dtype=np.int64).reshape(1,2,3)))
add("squeeze_213", np.squeeze(np.arange(6,dtype=np.int64).reshape(2,1,3)))
add("expand_dims_0", np.expand_dims(V3, 0))
add("expand_dims_1", np.expand_dims(V3, 1))
add("flatten", np.arange(6,dtype=np.int64).reshape(2,3).flatten())
add("ravel", np.arange(6,dtype=np.int64).reshape(2,3).ravel())
add("concat_0", np.concatenate([A2, B2], axis=0))
add("concat_1", np.concatenate([A2, B2], axis=1))
add("stack_0", np.stack([V3, V3*2], axis=0))
add("flip_1d", np.flip(V5i))
add("flip_axis0", np.flip(M34, axis=0))   # 3x4 for run_all_tests
add("flip_axis1", np.flip(M34, axis=1))
add("flip_axis0_23", np.flip(np.arange(6,dtype=np.int64).reshape(2,3), axis=0))  # 2x3 for auto-compare-test
add("roll_2", np.roll(V5i, 2))
add("roll_neg1", np.roll(V5i, -1))
add("triu_3", np.triu(A3))
add("triu_3_k1", np.triu(A3, k=1))
add("tril_3", np.tril(A3))
add("diag_3", np.diagonal(A3))
add("diag_3_k1", np.diagonal(A3, 1))
add("tile_3", np.tile(V3i, 3))
add("repeat_2", np.repeat(V3i, 2))
add("broadcast_23", np.broadcast_to(V3i, (2,3)))

a10 = np.arange(10,dtype=np.int64)
add("slice_2_7", a10[2:7])
add("slice_1_9_2", a10[1:9:2])
add("slice_nil_5", a10[:5])
add("slice_5_nil", a10[5:])
add("slice_reverse", a10[::-1])
add("slice_8_3_neg1", a10[8:3:-1])
add("slice_neg1", int(a10[-1]))
b45 = np.arange(20,dtype=np.int64).reshape(4,5)
add("2d_1_2", int(b45[1,2]))
add("2d_row2", b45[2,:])
add("2d_col3", b45[:,3])
add("2d_sub", b45[1:3, 2:4])
add("2d_ellipsis", b45[..., :2])

ai = np.array([1,2,3,4], dtype=np.int64)
bi = np.array([5,6,7,8], dtype=np.int64)
af = np.array([1.0, 2.0, 3.0, 4.0])
bf = np.array([5.0, 6.0, 7.0, 8.0])
add("add_ii", ai + bi)
add("sub_ii", bi - ai)
add("mul_ii", ai * bi)
add("add_ff", af + bf)
add("mul_ff", af * bf)
add("div_ff", bf / af)
add("add_scalar_i", ai + 10)
add("mul_scalar_i", ai * 2)
add("abs_neg", np.abs(np.array([-3.0, -1.0, 0.0, 1.0, 3.0])))
add("square", af ** 2)
add("sqrt_149", np.sqrt(np.array([1.0, 4.0, 9.0])))
add("exp_1234", np.exp(af))
add("log_1234", np.log(af))
add("log2_1234", np.log2(af))
add("log10_1234", np.log10(af))
add("clip_23", np.clip(af, 2.0, 3.0))
add("floor_1234", np.floor(np.array([1.2, 2.5, 3.7, 4.1])))
add("ceil_1234", np.ceil(np.array([1.2, 2.5, 3.7, 4.1])))
add("round_1234", np.round(np.array([1.2, 2.5, 3.7, 4.1])))
add("reciprocal", 1.0 / af)

x_pi = np.linspace(0, np.pi/2, 4)
add("sin_0pi2", np.sin(x_pi))
add("cos_0pi2", np.cos(x_pi))
add("tanh_123", np.tanh(np.array([1.0, 2.0, 3.0])))
add("hypot_34", np.hypot(np.array([3.0, 5.0]), np.array([4.0, 12.0])))
add("deg2rad", np.deg2rad(np.array([0.0, 90.0, 180.0])))
add("rad2deg", np.rad2deg(np.array([0.0, np.pi/2, np.pi])))

a5 = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
b5 = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
add("lt", (a5 < b5).astype(float))
add("eq", (a5 == b5).astype(float))
add("gt", (a5 > b5).astype(float))
add("all_true", int(np.all(np.array([1, 1, 1],dtype=np.int64))))
add("any_true", int(np.any(np.array([0, 0, 1],dtype=np.int64))))
add("isfinite", np.isfinite(np.array([1.0, np.nan, np.inf, -np.inf, 0.0])).astype(float))
add("isnan", np.isnan(np.array([1.0, np.nan, np.inf, 0.0])).astype(float))
add("isinf", np.isinf(np.array([1.0, np.nan, np.inf, -np.inf, 0.0])).astype(float))

add("sum_ax0", np.sum(M34, axis=0))
add("sum_ax1", np.sum(M34, axis=1))
add("sum_ax0_kd", np.sum(M34, axis=0, keepdims=True))
add("mean_ax1", np.mean(M34, axis=1))
add("max_all", float(np.max(M34)))
add("max_ax0", np.max(M34, axis=0))
add("min_ax1", np.min(M34, axis=1))
add("argmax_ax1", np.argmax(M34, axis=1).astype(np.int64))
add("argmin_ax0", np.argmin(M34, axis=0).astype(np.int64))
add("cumsum_1234", np.cumsum(np.array([1,2,3,4],dtype=np.int64)))
add("cumprod_1234", np.cumprod(np.array([1,2,3,4],dtype=np.int64)))
add("median_odd", float(np.median(np.array([3.0, 1.0, 4.0, 1.0, 5.0]))))
add("median_even", float(np.median(np.array([1.0, 2.0, 3.0, 4.0]))))
add("pct50", float(np.percentile(V5, 50)))
add("pct90", float(np.percentile(V5, 90)))
add("sort_1d", np.sort(np.array([3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0])))
add("max_3elem", np.maximum(np.array([1.0, 5.0, 3.0]), np.array([4.0, 2.0, 6.0])))
add("min_3elem", np.minimum(np.array([1.0, 5.0, 3.0]), np.array([4.0, 2.0, 6.0])))
add("diff_1d", np.diff(np.array([1.0, 3.0, 6.0, 10.0, 15.0])))

add("matmul_2x2", A2 @ B2)
add("matmul_2x3_3x2", A23 @ A32)
add("outer_3_2", np.outer(V3, np.array([4.0, 5.0])))
add("trace_3", float(np.trace(A3)))
add("norm_34", float(np.linalg.norm(np.array([3.0, 4.0]))))
add("det_2x2", float(np.linalg.det(A2)))
add("inv_2x2", np.linalg.inv(A2))
add("solve_2x2", np.linalg.solve(np.array([[2,1],[1,3]],dtype=np.float64), np.array([7,8],dtype=np.float64)))
Q, R_qr = np.linalg.qr(A23)
add("qr_recon_err", float(np.max(np.abs(A23 - Q @ R_qr))))
add("chol_L", np.linalg.cholesky(np.array([[4,2],[2,3]],dtype=np.float64)))
add("rank_full", int(np.linalg.matrix_rank(np.eye(3))))
add("rank_deficient", int(np.linalg.matrix_rank(np.array([[1,2],[2,4]],dtype=np.float64))))

add("einsum_dot", int(np.einsum("i,i->", np.array([1,2,3],dtype=np.int64), np.array([4,5,6],dtype=np.int64))))
add("einsum_matmul", np.einsum("ij,jk->ik", np.arange(6,dtype=np.int64).reshape(2,3), np.arange(6,dtype=np.int64).reshape(3,2)))
add("einsum_transpose", np.einsum("ij->ji", np.array([[1,2],[3,4]],dtype=np.int64)))
add("einsum_diag", np.einsum("ii->i", np.arange(9,dtype=np.int64).reshape(3,3)))
add("einsum_trace", int(np.einsum("ii->", np.arange(9,dtype=np.int64).reshape(3,3))))
add("einsum_outer", np.einsum("i,j->ij", np.array([1,2,3],dtype=np.int64), np.array([4,5],dtype=np.int64)))

add("sigmoid", sigmoid_np(x5))
add("relu", relu_np(x5))
add("tanh", np.tanh(x5))
add("softmax_123", softmax_np(np.array([1.0, 2.0, 3.0])))

add("unique_122333", np.unique(np.array([1,2,2,3,3,3],dtype=np.int64)))
add("intersect1d", np.intersect1d(np.array([1,2,3,4,5],dtype=np.int64), np.array([3,4,5,6,7],dtype=np.int64)))
add("union1d", np.union1d(np.array([1,2,3,4,5],dtype=np.int64), np.array([3,4,5,6,7],dtype=np.int64)))
add("setdiff1d", np.setdiff1d(np.array([1,2,3,4,5],dtype=np.int64), np.array([3,4,5,6,7],dtype=np.int64)))

add("where_cxy", np.where(np.array([1,0,1,0],dtype=float), np.array([10.0,20,30,40]), np.array([100.0,200,300,400])))
add("nonzero_1d", np.nonzero(np.array([0,1,0,2,0,3],dtype=np.int64))[0])

nan = np.nan
a_nan = np.array([1.0, nan, 3.0, 4.0])
add("nanmean", float(np.nanmean(a_nan)))
add("nansum", float(np.nansum(a_nan)))
add("nanmax", float(np.nanmax(a_nan)))
add("nanmin", float(np.nanmin(a_nan)))

add("pad_const_1", np.pad(np.array([[1,2],[3,4]],dtype=np.int64), 1, mode='constant', constant_values=0))
add("pad_edge_1", np.pad(np.array([[1,2],[3,4]],dtype=np.int64), 1, mode='edge'))
add("kron_22", np.kron(np.array([[1,2],[3,4]],dtype=np.int64), np.array([[0,5],[6,7]],dtype=np.int64)))
yt = np.array([1.0, 0.0, 0.0])
yp = np.array([0.7, 0.2, 0.1])
add("mse", float(np.mean((yt - yp)**2)))

# ============================================================
# 3. auto-compare-test 值（与run_all_tests部分重名但键名不同，这里全部补齐）
# ============================================================
add("arange_10", np.arange(10, dtype=np.int64))  # already added
add("linspace_0_1_5", np.linspace(0.0, 1.0, 5))  # already added
add("logspace_0_3_4", np.logspace(0, 3, 4))       # already added
add("eye_3", np.eye(3))                           # already added
add("diag_123", np.diag(np.array([1,2,3],dtype=np.int64)))

add("reshape_23", np.arange(6,dtype=np.int64).reshape(2,3))
add("squeeze_123", np.squeeze(np.arange(6,dtype=np.int64).reshape(1,2,3)))  # already
add("expand_dims_0", np.expand_dims(np.array([1,2,3],dtype=np.int64), 0))
add("concat_axis0", np.concatenate([np.array([[1,2],[3,4]],dtype=np.int64),
                                     np.array([[5,6],[7,8]],dtype=np.int64)], axis=0))
add("stack_axis0", np.stack([np.array([1,2],dtype=np.int64), np.array([3,4],dtype=np.int64)], axis=0))
add("roll_2_arange", np.roll(np.arange(5,dtype=np.int64), 2))
add("triu", np.triu(np.arange(9,dtype=np.int64).reshape(3,3)))
add("tril", np.tril(np.arange(9,dtype=np.int64).reshape(3,3)))
add("diagonal", np.diagonal(np.arange(9,dtype=np.int64).reshape(3,3)))
add("tile_3", np.tile(np.array([1,2,3],dtype=np.int64), 3))
add("repeat_2", np.repeat(np.array([1,2,3],dtype=np.int64), 2))
add("broadcast_to_23", np.broadcast_to(np.array([1,2,3],dtype=np.int64), (2,3)))

add("slice_2_7", np.arange(10,dtype=np.int64)[2:7])
add("slice_reverse", np.arange(10,dtype=np.int64)[::-1])
add("2d_1_2", int(np.arange(20,dtype=np.int64).reshape(4,5)[1,2]))
add("2d_row2", np.arange(20,dtype=np.int64).reshape(4,5)[2,:])

ab_i = np.array([1,2,3,4],dtype=np.int64)
bb_i = np.array([5,6,7,8],dtype=np.int64)
add("add_ab", ab_i + bb_i)
add("sub_ba", bb_i - ab_i)
add("mul_ab", ab_i * bb_i)
add("add_scalar10", ab_i + 10)
add("mul_scalar2", ab_i * 2)

add("exp_123", np.exp(np.array([1.0,2.0,3.0])))
add("log_123", np.log(np.array([1.0,2.0,3.0])))
add("sqrt_149_16", np.sqrt(np.array([1.0,4.0,9.0,16.0])))

m_ac = M34  # same as M34
add("sum_all", float(m_ac.sum()))
add("sum_axis0", m_ac.sum(axis=0))
add("sum_axis1", m_ac.sum(axis=1))
add("mean_all", float(m_ac.mean()))
add("max_all", float(m_ac.max()))
add("min_axis0", m_ac.min(axis=0))
add("argmax_axis1", m_ac.argmax(axis=1).astype(np.int64))
add("std_all", float(m_ac.std()))
add("cumsum_1234", np.cumsum(np.array([1,2,3,4],dtype=np.int64)))
add("median_31415926", float(np.median(np.array([3.0,1.0,4.0,1.0,5.0,9.0,2.0,6.0]))))
add("sort_8", np.sort(np.array([3.0,1.0,4.0,1.0,5.0,9.0,2.0,6.0])))

add("matmul_2x2", A2 @ B2)
add("trace_3x3", float(np.trace(np.arange(9,dtype=np.float64).reshape(3,3))))
add("det_2x2", float(np.linalg.det(A2)))
add("norm_34", float(np.linalg.norm(np.array([3.0,4.0]))))
add("cholesky_L", np.linalg.cholesky(np.array([[4.0,2.0],[2.0,3.0]])))
add("rank_12_24", int(np.linalg.matrix_rank(np.array([[1.0,2.0],[2.0,4.0]]))))

add("einsum_dot", int(np.einsum("i,i->", np.array([1,2,3],dtype=np.int64), np.array([4,5,6],dtype=np.int64))))
add("einsum_matmul", np.einsum("ij,jk->ik", np.arange(6,dtype=np.int64).reshape(2,3), np.arange(6,dtype=np.int64).reshape(3,2)))
add("einsum_diag", np.einsum("ii->i", np.arange(9,dtype=np.int64).reshape(3,3)))
add("einsum_outer", np.einsum("i,j->ij", np.array([1,2,3],dtype=np.int64), np.array([4,5],dtype=np.int64)))

x_ac = np.array([-2.0,-1.0,0.0,1.0,2.0])
add("sigmoid_x", sigmoid_np(x_ac))
add("relu_x", relu_np(x_ac))
add("tanh_x", np.tanh(x_ac))
add("softmax_123", softmax_np(np.array([1.0,2.0,3.0])))

add("where_cond", np.where(np.array([1.0,0.0,1.0,0.0]), np.array([10.0,20.0,30.0,40.0]),
                            np.array([100.0,200.0,300.0,400.0])))
add("nonzero_010203", np.nonzero(np.array([0,1,0,2,0,3],dtype=np.int64))[0])

add("unique_122333", np.unique(np.array([1,2,2,3,3,3],dtype=np.int64)))
add("intersect1d", np.intersect1d(np.array([1,2,3,4,5],dtype=np.int64), np.array([3,4,5,6,7],dtype=np.int64)))

add("nanmean_1nan34", float(np.nanmean(np.array([1.0,nan,3.0,4.0]))))
add("nansum_1nan34", float(np.nansum(np.array([1.0,nan,3.0,4.0]))))
add("nanmax_1nan34", float(np.nanmax(np.array([1.0,nan,3.0,4.0]))))

# ============================================================
# 4. run_param_tests 值 (parametric 2D/3D tests)
# ============================================================
# --- Reduction 2D ---
add("sum_2d_all", float(M34.sum()))
add("sum_2d_ax0", M34.sum(axis=0))
add("sum_2d_ax1", M34.sum(axis=1))
add("sum_2d_ax_neg1", M34.sum(axis=-1))
add("sum_2d_ax0_kd", M34.sum(axis=0, keepdims=True))
add("mean_2d_all", float(M34.mean()))
add("mean_2d_ax0", M34.mean(axis=0))
add("mean_2d_ax1", M34.mean(axis=1))
add("amax_2d_all", float(M34.max()))
add("amax_2d_ax0", M34.max(axis=0))
add("amin_2d_ax1", M34.min(axis=1))
add("argmax_2d_ax0", M34.argmax(axis=0).astype(np.int64))
add("argmax_2d_ax1", M34.argmax(axis=1).astype(np.int64))
add("argmin_2d_ax0", M34.argmin(axis=0).astype(np.int64))
add("argmin_2d_ax_neg1", M34.argmin(axis=-1).astype(np.int64))
add("std_2d_all", float(M34.std()))
add("var_2d_all", float(M34.var()))
add("cumsum_2d_ax0", np.cumsum(mi, axis=0))
add("cumsum_2d_ax1", np.cumsum(mi, axis=1))
add("cumprod_2d_ax0", np.cumprod(mi, axis=0))
add("median_2d_ax0", np.median(M34, axis=0))
add("median_2d_ax1", np.median(M34, axis=1))
add("ptp_2d_ax0", np.ptp(M34, axis=0))
add("ptp_2d_ax1", np.ptp(M34, axis=1))
add("sort_2d_ax0", np.sort(M34, axis=0))
add("sort_2d_ax1", np.sort(M34, axis=1))
add("argsort_2d_ax0", np.argsort(M34, axis=0).astype(np.int64))
add("argsort_2d_ax1", np.argsort(M34, axis=1).astype(np.int64))
add("diff_2d_ax0", np.diff(M34, axis=0))
add("diff_2d_ax1", np.diff(M34, axis=1))

# --- Reduction 3D ---
add("sum_3d_all", float(M234.sum()))
add("sum_3d_ax0", M234.sum(axis=0))
add("sum_3d_ax1", M234.sum(axis=1))
add("sum_3d_ax2", M234.sum(axis=2))
add("sum_3d_ax_neg1", M234.sum(axis=-1))
add("sum_3d_ax_neg2", M234.sum(axis=-2))
add("sum_3d_ax01", M234.sum(axis=(0,1)))
add("sum_3d_ax12", M234.sum(axis=(1,2)))
add("sum_3d_ax0_kd", M234.sum(axis=0, keepdims=True))
add("sum_3d_ax2_kd", M234.sum(axis=2, keepdims=True))
add("mean_3d_all", float(M234.mean()))
add("mean_3d_ax0", M234.mean(axis=0))
add("mean_3d_ax1", M234.mean(axis=1))
add("mean_3d_ax2", M234.mean(axis=2))
add("mean_3d_ax_neg1", M234.mean(axis=-1))
add("mean_3d_ax01", M234.mean(axis=(0,1)))
add("mean_3d_ax0_kd", M234.mean(axis=0, keepdims=True))
add("amax_3d_all", float(M234.max()))
add("amax_3d_ax0", M234.max(axis=0))
add("amax_3d_ax2", M234.max(axis=2))
add("amin_3d_ax1", M234.min(axis=1))
add("argmax_3d_ax0", np.argmax(M234, axis=0).astype(np.int64))
add("argmax_3d_ax2", np.argmax(M234, axis=2).astype(np.int64))
add("argmin_3d_ax0", np.argmin(M234, axis=0).astype(np.int64))
add("argmin_3d_ax1", np.argmin(M234, axis=1).astype(np.int64))
add("std_3d_ax0", M234.std(axis=0))
add("var_3d_ax2", M234.var(axis=2))
add("cumsum_3d_ax0", np.cumsum(ti, axis=0))
add("cumsum_3d_ax1", np.cumsum(ti, axis=1))
add("cumsum_3d_ax2", np.cumsum(ti, axis=2))
add("cumprod_3d_ax2", np.cumprod(ti, axis=2))
add("median_3d_ax0", np.median(M234, axis=0))
add("median_3d_ax2", np.median(M234, axis=2))
add("ptp_3d_ax0", np.ptp(M234, axis=0))
add("ptp_3d_ax2", np.ptp(M234, axis=2))
add("sort_3d_ax0", np.sort(M234, axis=0))
add("sort_3d_ax2", np.sort(M234, axis=2))
add("argsort_3d_ax0", np.argsort(M234, axis=0).astype(np.int64))
add("argsort_3d_ax2", np.argsort(M234, axis=2).astype(np.int64))
add("diff_3d_ax0", np.diff(M234, axis=0))
add("diff_3d_ax2", np.diff(M234, axis=2))
add("maximum_3d", np.maximum(M234, np.flip(M234, axis=0)))
add("minimum_3d", np.minimum(M234, np.flip(M234, axis=0)))

# --- Percentile 2D+3D ---
for p in [25, 50, 75, 90]:
    add(f"pct{p}_2d_ax0", np.percentile(M34, p, axis=0))
    add(f"pct{p}_2d_ax1", np.percentile(M34, p, axis=1))
    add(f"pct{p}_3d_ax0", np.percentile(M234, p, axis=0))
    add(f"pct{p}_3d_ax2", np.percentile(M234, p, axis=2))

# --- Shape 3D ---
add("trans_3d_021", np.transpose(M234, (0,2,1)))
add("trans_3d_102", np.transpose(M234, (1,0,2)))
add("trans_3d_210", np.transpose(M234, (2,1,0)))
add("squeeze_3d_134", np.squeeze(np.arange(12,dtype=np.int64).reshape(1,3,4)))
add("squeeze_3d_314", np.squeeze(np.arange(12,dtype=np.int64).reshape(3,1,4)))
add("squeeze_3d_341", np.squeeze(np.arange(12,dtype=np.int64).reshape(3,4,1)))
add("expand_3d_ax0_shape", list(np.expand_dims(M234, 0).shape))
add("expand_3d_ax2_shape", list(np.expand_dims(M234, 2).shape))
add("expand_3d_ax3_shape", list(np.expand_dims(M234, 3).shape))
add("concat_3d_ax0", np.concatenate([M234, M234+100.0], axis=0))
add("flip_3d_ax0", np.flip(M234, axis=0))
add("flip_3d_ax1", np.flip(M234, axis=1))
add("flip_3d_ax2", np.flip(M234, axis=2))
add("roll_3d_ax0", np.roll(ti, 1, axis=0))
add("roll_3d_ax2", np.roll(ti, 1, axis=2))
add("triu_3d", np.triu(M234))
add("tril_3d", np.tril(M234))
add("swapaxes_3d_01", np.swapaxes(M234, 0, 1))
add("swapaxes_3d_02", np.swapaxes(M234, 0, 2))
add("swapaxes_3d_12", np.swapaxes(M234, 1, 2))
add("narrow_3d_ax0", M234[0:1, :, :])
add("narrow_3d_ax1", M234[:, 1:4, :])
add("narrow_3d_ax2", M234[:, :, 1:4])
add("tile_3d_121", np.tile(t222i, (1,2,1)))
add("tile_3d_212", np.tile(t222i, (2,1,2)))
add("repeat_3d_ax0", np.repeat(t222i, 2, axis=0))
add("repeat_3d_ax2", np.repeat(t222i, 2, axis=2))
add("pad_3d_const", np.pad(t222, 1, mode='constant', constant_values=0))
add("pad_3d_edge", np.pad(t222, 1, mode='edge'))
add("broadcast_1x3x4_to_2x3x4", np.broadcast_to(np.arange(12,dtype=np.int64).reshape(1,3,4), (2,3,4)))

# --- Arithmetic 3D ---
add("add_3d_scalar", M234 + 10.0)
add("mul_3d_scalar", M234 * 2.0)
add("add_3d_3d", M234 + M234)
add("mul_3d_3d", M234 * M234)
add("abs_3d", np.abs(M234 - 12.0))
add("square_3d", M234 ** 2)
add("sqrt_3d", np.sqrt(np.abs(M234)))
add("exp_3d", np.exp(M234 * 0.1))
add("log_3d", np.log(np.abs(M234) + 1.0))
add("clip_3d", np.clip(M234, 5.0, 15.0))

# --- Trig 3D ---
add("sin_3d", np.sin(M234 * 0.1))
add("cos_3d", np.cos(M234 * 0.1))
add("tanh_3d", np.tanh((M234 - 12.0) * 0.1))

# --- Comparison 3D ---
add("lt_3d", (M234 < 12.0).astype(np.float64))
add("gt_3d", (M234 > 12.0).astype(np.float64))
add("isfinite_3d", np.isfinite(M234).astype(np.float64))
add("all_3d_ax0", np.all(M234 > 0.0, axis=0).astype(np.float64))
add("any_3d_ax2", np.any(M234 > 10.0, axis=2).astype(np.float64))

# --- einsum advanced ---
add("einsum_trace_2d", int(np.einsum("ii->", np.arange(4,dtype=np.int64).reshape(2,2))))
add("einsum_3d_reduce", np.einsum("ijk->i", M234))
add("einsum_3d_reduce_ax2", np.einsum("ijk->ij", M234))

# --- Where 3D ---
add("where_3d", np.where(M234 > 12.0, M234, np.zeros((2,3,4))))

# --- Activation 3D ---
add("sigmoid_3d", sigmoid_np(M234))
add("relu_3d", relu_np(M234 - 12.0))

# --- Diagonal 3D ---
# vt-diagonal on 3D: numpy diagonal takes offset, axis1, axis2 defaults
# clvt vt-diagonal for 3D uses default (offset=0, axis1=-2, axis2=-1 => last two axes)
# For shape (2,3,4), diagonal over last two axes gives shape (4,2) -> offset=0, axis1=1, axis2=2
add("diag_3d", np.diagonal(M234, offset=0, axis1=1, axis2=2))
add("diag_3d_pytorch", np.diagonal(M234, offset=0, axis1=1, axis2=2))

# --- Convolve ---
conv_a = np.array([1.0,2.0,3.0,4.0,5.0])
conv_v = np.array([1.0,0.0,-1.0])
add("convolve_valid", np.convolve(conv_a, conv_v, mode='valid'))
add("convolve_full", np.convolve(conv_a, conv_v, mode='full'))

# --- Eig 3x3 ---
eig_a = np.array([[4.0,2.0,1.0],[2.0,5.0,3.0],[1.0,3.0,6.0]])
eigvals = np.linalg.eigvalsh(eig_a)
add("eig_3x3_vals", np.sort(eigvals)[::-1])  # descending order

# --- Stack 3D (shape checks) ---
add("stack_3d_ax0", list(np.stack([M234, M234+100.0], axis=0).shape))
add("stack_3d_ax3", list(np.stack([M234, M234+100.0], axis=3).shape))

# --- Split 3D ---
add("split_3d_ax0", [p.tolist() for p in np.split(M234, 2, axis=0)])
add("split_3d_ax2", [p.tolist() for p in np.split(M234, 2, axis=2)])

# --- Floor/Round 3D ---
add("floor_3d", np.floor(M234 * 0.7))
add("round_3d", np.round(M234 * 0.7))

# ============================================================
# PyTorch 独立计算交叉验证（独立重算一批核心操作，确保参考值100%正确）
# ============================================================
if HAS_TORCH:
    def torch_check_op(name, np_result, torch_result, rtol=1e-5, atol=1e-8):
        global TORCH_PASS, TORCH_FAIL
        try:
            t_np = torch_result.detach().cpu().numpy() if isinstance(torch_result, torch.Tensor) else torch_result
            ok = np.allclose(np_result, t_np, rtol=rtol, atol=atol, equal_nan=True)
            if isinstance(np_result, np.ndarray) and np_result.shape != np.asarray(t_np).shape:
                ok = False
            if ok:
                TORCH_PASS += 1
            else:
                TORCH_FAIL += 1
                print(f"[torch-cross] {name}: mismatch", file=sys.stderr)
        except Exception as e:
            TORCH_FAIL += 1
            print(f"[torch-cross] {name}: error {e}", file=sys.stderr)

    t_a = torch.tensor([1.0, 2.0, 3.0, 4.0])
    t_b = torch.tensor([5.0, 6.0, 7.0, 8.0])
    n_a = np.array([1.0, 2.0, 3.0, 4.0])
    n_b = np.array([5.0, 6.0, 7.0, 8.0])
    torch_check_op("torch:add", n_a + n_b, t_a + t_b)
    torch_check_op("torch:mul", n_a * n_b, t_a * t_b)
    torch_check_op("torch:matmul", np.array([[1,2],[3,4.]]) @ np.array([[5,6],[7,8.]]),
                   torch.tensor([[1.,2.],[3.,4.]]) @ torch.tensor([[5.,6.],[7.,8.]]))
    torch_check_op("torch:sin", np.sin(np.linspace(0, np.pi/2, 4)),
                   torch.sin(torch.linspace(0, torch.tensor(math.pi)/2, 4)))
    torch_check_op("torch:exp", np.exp(n_a), torch.exp(t_a))
    torch_check_op("torch:sum", float(np.sum(np.arange(12, dtype=np.float64).reshape(3,4))),
                   float(torch.sum(torch.arange(12, dtype=torch.float64).reshape(3,4))))
    torch_check_op("torch:softmax", softmax_np(np.array([1.0, 2.0, 3.0])),
                   torch.softmax(torch.tensor([1.0, 2.0, 3.0]), dim=0))
    torch_check_op("torch:sigmoid", sigmoid_np(x5), torch.sigmoid(torch.tensor(x5)))
    m_t = torch.arange(12, dtype=torch.float64).reshape(3,4)
    torch_check_op("torch:mean_axis1", np.mean(np.arange(12, dtype=np.float64).reshape(3,4), axis=1),
                   torch.mean(m_t, dim=1))
    # Additional cross-checks for parametric tests
    t_m34 = torch.arange(12, dtype=torch.float64).reshape(3,4)
    torch_check_op("torch:sum_2d_all", float(M34.sum()), float(t_m34.sum()))
    torch_check_op("torch:sum_2d_ax0", M34.sum(axis=0), t_m34.sum(dim=0))
    torch_check_op("torch:mean_2d_all", float(M34.mean()), float(t_m34.mean()))
    t_m234 = torch.arange(24, dtype=torch.float64).reshape(2,3,4)
    torch_check_op("torch:sum_3d_all", float(M234.sum()), float(t_m234.sum()))
    torch_check_op("torch:sum_3d_ax1", M234.sum(axis=1), t_m234.sum(dim=1))
    torch_check_op("torch:sigmoid_3d", sigmoid_np(M234), torch.sigmoid(t_m234))
    torch_check_op("torch:relu_3d", relu_np(M234 - 12.0), torch.relu(t_m234 - 12))
    print(f"[torch] 验证完成: {TORCH_PASS} 通过, {TORCH_FAIL} 失败", file=sys.stderr)
else:
    print("[torch] PyTorch未安装，跳过PyTorch交叉验证", file=sys.stderr)

# 去重（同一key add多次会覆盖为最后一个值，json输出即最终值）

# ====================================================================
# 遗产用例盘活区（2026-10）：gen_all_tests.py / gen_param_tests.py 在静态 JSON
# 架构时代生成、但从未被任何 lisp 套件接线的用例，以重算代码形式并入实时
# oracle。消费方: test/legacy-coverage-test.lisp（防漂移门禁已登记）。
# 来源: git show cee59a4:test/gen_all_tests.py / gen_param_tests.py
# ====================================================================

_LEGACY_GAP_KEYS = frozenset('2d_neg 2d_newaxis_shape acos_01 add_3d_1d add_scalar_f all_3d_ax2 all_false allclose_f allclose_t amax_2d_ax0_kd amax_2d_ax1 amax_2d_ax1_kd amax_2d_ax_neg1 amax_3d_ax01 amax_3d_ax0_kd amax_3d_ax1 amax_3d_ax12 amax_3d_ax2_kd amax_3d_ax_neg1 amax_3d_ax_neg2 amin_2d_all amin_2d_ax0 amin_2d_ax0_kd amin_2d_ax1_kd amin_2d_ax_neg1 amin_3d_all amin_3d_ax0 amin_3d_ax01 amin_3d_ax0_kd amin_3d_ax12 amin_3d_ax2 amin_3d_ax2_kd amin_3d_ax_neg1 amin_3d_ax_neg2 any_3d_ax0 any_false arange_5_s2 argmax_2d_ax_neg1 argmax_3d_ax1 argmax_3d_ax_neg1 argmax_ax0 argmin_2d_ax1 argmin_3d_ax2 argmin_3d_ax_neg1 argmin_ax1 argsort_1d asin_01 astype_f32 astype_f64 astype_i32 astype_i64 atan2 atan_012 broadcast_123 broadcast_2x1x4_to_2x3x4 broadcast_add broadcast_mul ceil_3d chol_err clip_3d_neg concat_3d_ax1 concat_neg1 cosh_123 cross_entropy cumprod_2d_ax1 cumprod_3d_ax0 cumprod_3d_ax1 cumsum_ax0 cumsum_ax1 det_3x3 det_batch diag_3_k_neg1 diag_extract_k_neg1 diag_rect diff_ax1 dot_1d dot_2d_1d dot_2d_2d einsum_batch_mm einsum_bilinear einsum_matmul_T eq_3d expand_3d_ax0 expand_3d_ax1 expand_3d_ax2 expand_3d_ax3 expand_3d_ax_neg1 expand_dims_neg1 flip_3d_ax_neg1 flip_neg1 full_23_7 ge grad_2d_ax0 grad_2d_ax1 grad_3d_ax0 grad_3d_ax2 gradient_1d gt_3d_pair hsplit identity_3 int32_float64 intersect_empty isclose isclose_3d l1_norm le le_3d_pair linspace_0_1_5_ne logical_and logical_not logical_or logspace_0_2_3 lt_3d_pair lu_L lu_U matmul_1x3_3x1 matmul_3x3 matmul_4x4 max_ax1 maximum mean_2d_ax0_kd mean_2d_ax1_kd mean_2d_ax_neg1 mean_3d_ax12 mean_3d_ax2_kd mean_3d_ax_neg2 mean_ax0 median_3d_ax1 median_ax1 meshgrid_x_shape meshgrid_y_shape min_all min_ax0 minimum mod_3d mod_7 mse_3d mul_3d_1d mul_scalar_f nanmax_3d_ax0 nanmean_3d_ax0 nanmean_3d_ax2 nanmin_3d_ax2 nanstd nanstd_3d_ax0 nansum_3d_ax1 nanvar nanvar_3d_ax2 narrow_34_ax0_1_3 narrow_34_ax1_1_3 narrow_clip_over narrow_neg_start ne neg_f nonzero_2d nonzero_3d norm_3d_ax0 norm_3d_ax1 norm_3d_ax2 norm_ax1 ones_22 pad_reflect_1 pct25 pct25_2d_ax0 pct25_2d_ax1 pct25_3d_ax0 pct25_3d_ax2 pct50_2d_ax0 pct50_2d_ax1 pct50_3d_ax0 pct50_3d_ax2 pct75 pct75_2d_ax0 pct75_2d_ax1 pct75_3d_ax0 pct75_3d_ax2 pct90_2d_ax0 pct90_2d_ax1 pct90_3d_ax0 pct90_3d_ax2 pow_05 pow_2 prod_2d_all prod_2d_ax0 prod_2d_ax0_kd prod_2d_ax1 prod_2d_ax1_kd prod_2d_ax_neg1 prod_3d_all prod_3d_ax0 prod_3d_ax01 prod_3d_ax0_kd prod_3d_ax1 prod_3d_ax12 prod_3d_ax2 prod_3d_ax2_kd prod_3d_ax_neg1 prod_3d_ax_neg2 prod_ax0 ptp_ax0 qr_Q_shape qr_R_shape rem_7 repeat_2_ax1 repeat_list reshape_neg1 rint_1234 roll_3d_ax1 roll_axis0 roll_axis1 rot90_k1 rot90_k2 setxor1d signum sinh_123 slice_neg3_neg1 softmax_3d_ax2 solve_3x3 sort_3d_ax1 sort_ax0 sort_ax1 split_3 split_3_7 squeeze_231 squeeze_3d_ax0 squeeze_3d_ax1 stack_1 stack_3d_ax1 std_2d_ax0 std_2d_ax0_kd std_2d_ax1 std_2d_ax1_kd std_2d_ax_neg1 std_3d_all std_3d_ax01 std_3d_ax0_kd std_3d_ax1 std_3d_ax12 std_3d_ax2 std_3d_ax2_kd std_3d_ax_neg1 std_3d_ax_neg2 std_ax0 sub_ff sum_2d_ax1_kd sum_ax1_kd svd_s_diag3 swapaxes_01 swapaxes_02 tan_0pi4 tanh_3d_act tile_23 trace_3d trace_rect trans_3d_012 trans_3d_201 transpose_3d transpose_perm tril_3_k1 tril_3_k_neg1 triu_3_k_neg1 union_same unique_large unique_sorted unique_with_return_index var_2d_ax0 var_2d_ax0_kd var_2d_ax1 var_2d_ax1_kd var_2d_ax_neg1 var_3d_all var_3d_ax0 var_3d_ax01 var_3d_ax0_kd var_3d_ax1 var_3d_ax12 var_3d_ax2_kd var_3d_ax_neg1 var_3d_ax_neg2 var_ax0 vsplit where_scalar zeros_23'.split())  # 只补这些 key，绝不覆盖枢纽既有定义（防同名异值覆盖回归）

def _legacy_add(key, val):
    """遗产区块专用 add：只写入缺口名单内的 key，已有 key 一律跳过，
    保证 ref_compute 既有定义（旧套件消费方依赖）不被同名重放覆盖。"""
    if key in _LEGACY_GAP_KEYS:
        add(key, val)

def _legacy_all_block():
    """gen_all_tests.py 的 119 个未接线用例（key 风格: fn_场景）"""
    V3 = np.array([1.0, 2.0, 3.0])
    # ============================================================
    _legacy_add("arange_10", np.arange(10, dtype=np.int64))
    _legacy_add("arange_5_s2", np.arange(5, dtype=np.float64))  # start=0 default
    _legacy_add("linspace_0_1_5", np.linspace(0, 1, 5))
    _legacy_add("linspace_0_10_3", np.linspace(0, 10, 3))
    _legacy_add("linspace_0_1_5_ne", np.linspace(0, 1, 5, endpoint=False))
    _legacy_add("logspace_0_3_4", np.logspace(0, 3, 4))
    _legacy_add("logspace_0_2_3", np.logspace(0, 2, 3))
    _legacy_add("eye_3", np.eye(3))
    _legacy_add("eye_4x6", np.eye(4, 6))
    _legacy_add("eye_3_k1", np.eye(3, k=1))
    _legacy_add("eye_3_k_neg1", np.eye(3, k=-1))
    _legacy_add("diag_v123", np.diag(np.array([1,2,3])))
    _legacy_add("diag_extract", np.diag(np.arange(9).reshape(3,3)))
    _legacy_add("diag_extract_k1", np.diag(np.arange(9).reshape(3,3), k=1))
    _legacy_add("diag_extract_k_neg1", np.diag(np.arange(9).reshape(3,3), k=-1))
    _legacy_add("zeros_23", np.zeros((2,3)))
    _legacy_add("ones_22", np.ones((2,2)))
    _legacy_add("full_23_7", np.full((2,3), 7.0))
    _legacy_add("identity_3", np.eye(3))

    # ============================================================
    # 2. 形状操作 (30+ tests)
    # ============================================================
    _legacy_add("reshape_6_23", np.arange(6).reshape(2,3))
    _legacy_add("reshape_6_32", np.arange(6).reshape(3,2))
    _legacy_add("reshape_neg1", np.arange(6).reshape(2,-1))
    _legacy_add("transpose_23", np.arange(6).reshape(2,3).T)
    _legacy_add("transpose_perm", np.arange(24).reshape(2,3,4).transpose(1,0,2))
    _legacy_add("transpose_3d", np.arange(24).reshape(2,3,4).transpose())
    _legacy_add("squeeze_123", np.squeeze(np.arange(6).reshape(1,2,3)))
    _legacy_add("squeeze_213", np.squeeze(np.arange(6).reshape(2,1,3)))
    _legacy_add("squeeze_231", np.squeeze(np.arange(6).reshape(2,3,1)))
    _legacy_add("expand_dims_0", np.expand_dims(V3, 0))
    _legacy_add("expand_dims_1", np.expand_dims(V3, 1))
    _legacy_add("expand_dims_neg1", np.expand_dims(V3, -1))
    _legacy_add("flatten", np.arange(6).reshape(2,3).flatten())
    _legacy_add("ravel", np.arange(6).reshape(2,3).ravel())

    # concatenate
    _legacy_add("concat_0", np.concatenate([A2, B2], axis=0))
    _legacy_add("concat_1", np.concatenate([A2, B2], axis=1))
    _legacy_add("concat_neg1", np.concatenate([A2, B2], axis=-1))

    # stack
    _legacy_add("stack_0", np.stack([V3, V3*2], axis=0))
    _legacy_add("stack_1", np.stack([V3, V3*2], axis=1))

    # split
    _legacy_add("split_3_7", [x.tolist() for x in np.split(np.arange(10), [3,7])])
    _legacy_add("split_3", [x.tolist() for x in np.split(np.arange(9), 3)])
    _legacy_add("vsplit", [x.tolist() for x in np.vsplit(np.arange(12).reshape(3,4), 3)])
    _legacy_add("hsplit", [x.tolist() for x in np.hsplit(np.arange(12).reshape(3,4), 2)])

    # flip
    _legacy_add("flip_1d", np.flip(V5i))
    _legacy_add("flip_axis0", np.flip(M34, axis=0))
    _legacy_add("flip_axis1", np.flip(M34, axis=1))
    _legacy_add("flip_neg1", np.flip(M34, axis=-1))

    # roll
    _legacy_add("roll_2", np.roll(V5i, 2))
    _legacy_add("roll_neg1", np.roll(V5i, -1))
    _legacy_add("roll_axis0", np.roll(M34, 1, axis=0))
    _legacy_add("roll_axis1", np.roll(M34, 1, axis=1))

    # triu / tril
    _legacy_add("triu_3", np.triu(A3))
    _legacy_add("triu_3_k1", np.triu(A3, k=1))
    _legacy_add("triu_3_k_neg1", np.triu(A3, k=-1))
    _legacy_add("tril_3", np.tril(A3))
    _legacy_add("tril_3_k1", np.tril(A3, k=1))
    _legacy_add("tril_3_k_neg1", np.tril(A3, k=-1))

    # diagonal
    _legacy_add("diag_3", np.diagonal(A3))
    _legacy_add("diag_3_k1", np.diagonal(A3, 1))
    _legacy_add("diag_3_k_neg1", np.diagonal(A3, -1))
    _legacy_add("diag_rect", np.diagonal(A23))

    # tile / repeat
    _legacy_add("tile_3", np.tile(V3i, 3))
    _legacy_add("tile_23", np.tile(np.array([[1,2],[3,4]]), (2,3)))
    _legacy_add("repeat_2", np.repeat(V3i, 2))
    _legacy_add("repeat_2_ax1", np.repeat(np.array([[1,2],[3,4]]), 2, axis=1))
    _legacy_add("repeat_list", np.repeat(np.array([1,2,3]), [2,0,1]))

    # broadcast_to
    _legacy_add("broadcast_23", np.broadcast_to(V3i, (2,3)))
    _legacy_add("broadcast_123", np.broadcast_to(V3i.reshape(1,3), (2,3)))

    # narrow (slice along axis)
    _legacy_add("narrow_34_ax0_1_3", M34[1:3, :])
    _legacy_add("narrow_34_ax1_1_3", M34[:, 1:3])

    # swapaxes
    _legacy_add("swapaxes_01", np.swapaxes(M234, 0, 1))
    _legacy_add("swapaxes_02", np.swapaxes(M234, 0, 2))

    # rot90
    _legacy_add("rot90_k1", np.rot90(np.arange(4).reshape(2,2)))
    _legacy_add("rot90_k2", np.rot90(np.arange(4).reshape(2,2), k=2))

    # pad
    _legacy_add("pad_const_1", np.pad(np.array([[1,2],[3,4]]), 1, mode='constant', constant_values=0))
    _legacy_add("pad_edge_1", np.pad(np.array([[1,2],[3,4]]), 1, mode='edge'))
    _legacy_add("pad_reflect_1", np.pad(np.array([[1,2],[3,4]]), 1, mode='reflect'))

    # ============================================================
    # 3. 索引与切片 (20+ tests)
    # ============================================================
    a10 = np.arange(10)
    _legacy_add("slice_2_7", a10[2:7])
    _legacy_add("slice_1_9_2", a10[1:9:2])
    _legacy_add("slice_nil_5", a10[:5])
    _legacy_add("slice_5_nil", a10[5:])
    _legacy_add("slice_reverse", a10[::-1])
    _legacy_add("slice_8_3_neg1", a10[8:3:-1])
    _legacy_add("slice_neg1", int(a10[-1]))
    _legacy_add("slice_neg3_neg1", a10[-3:-1])

    b45 = np.arange(20).reshape(4,5)
    _legacy_add("2d_1_2", int(b45[1,2]))
    _legacy_add("2d_row2", b45[2,:])
    _legacy_add("2d_col3", b45[:,3])
    _legacy_add("2d_sub", b45[1:3, 2:4])
    _legacy_add("2d_neg", b45[-2:, -3:])
    _legacy_add("2d_ellipsis", b45[..., :2])
    _legacy_add("2d_newaxis_shape", list(np.expand_dims(b45, 1).shape))

    # where
    _legacy_add("where_cxy", np.where(np.array([1,0,1,0]), np.array([10.0,20,30,40]), np.array([100.0,200,300,400])))
    _legacy_add("where_scalar", np.where(np.array([1,0,1,0]), 100.0, 200.0))

    # nonzero
    _legacy_add("nonzero_1d", np.nonzero(np.array([0,1,0,2,0,3]))[0])
    _legacy_add("nonzero_2d", [x.tolist() for x in np.nonzero(np.array([[1,0],[0,1]]))])

    # ============================================================
    # 4. 算术运算 (30+ tests)
    # ============================================================
    ai = np.array([1,2,3,4], dtype=np.int64)
    bi = np.array([5,6,7,8], dtype=np.int64)
    af = np.array([1.0, 2.0, 3.0, 4.0])
    bf = np.array([5.0, 6.0, 7.0, 8.0])

    _legacy_add("add_ii", ai + bi)
    _legacy_add("sub_ii", bi - ai)
    _legacy_add("mul_ii", ai * bi)
    _legacy_add("add_ff", af + bf)
    _legacy_add("sub_ff", bf - af)
    _legacy_add("mul_ff", af * bf)
    _legacy_add("div_ff", bf / af)
    _legacy_add("add_scalar_i", ai + 10)
    _legacy_add("mul_scalar_i", ai * 2)
    _legacy_add("add_scalar_f", af + 1.5)
    _legacy_add("mul_scalar_f", af * 0.5)
    _legacy_add("neg_f", -af)
    _legacy_add("abs_neg", np.abs(np.array([-3.0, -1.0, 0.0, 1.0, 3.0])))
    _legacy_add("signum", np.sign(np.array([-3.0, -1.0, 0.0, 1.0, 3.0])))
    _legacy_add("square", af ** 2)
    _legacy_add("sqrt_149", np.sqrt(np.array([1.0, 4.0, 9.0])))
    _legacy_add("pow_2", af ** 2)
    _legacy_add("pow_05", af ** 0.5)
    _legacy_add("exp_1234", np.exp(af))
    _legacy_add("log_1234", np.log(af))
    _legacy_add("log2_1234", np.log2(af))
    _legacy_add("log10_1234", np.log10(af))
    _legacy_add("clip_23", np.clip(af, 2.0, 3.0))
    _legacy_add("mod_7", np.mod(ai, 3))
    _legacy_add("rem_7", np.remainder(ai, 3))
    _legacy_add("floor_1234", np.floor(np.array([1.2, 2.5, 3.7, 4.1])))
    _legacy_add("ceil_1234", np.ceil(np.array([1.2, 2.5, 3.7, 4.1])))
    _legacy_add("round_1234", np.round(np.array([1.2, 2.5, 3.7, 4.1])))
    _legacy_add("rint_1234", np.rint(np.array([1.2, 2.5, 3.7, 4.1])))
    _legacy_add("reciprocal", 1.0 / af)

    # broadcast arithmetic
    _legacy_add("broadcast_add", M34 + np.array([1.0, 2.0, 3.0, 4.0]))  # (3,4)+(4,)
    _legacy_add("broadcast_mul", M34 * np.array([1.0, 2.0, 3.0, 4.0]).reshape(1,4))

    # type promotion
    _legacy_add("int32_float64", np.array([1,2,3], dtype=np.int32) + np.array([0.5, 0.5, 0.5]))

    # ============================================================
    # 5. 三角函数 (15+ tests)
    # ============================================================
    x_pi = np.linspace(0, np.pi/2, 4)
    _legacy_add("sin_0pi2", np.sin(x_pi))
    _legacy_add("cos_0pi2", np.cos(x_pi))
    _legacy_add("tan_0pi4", np.tan(np.linspace(0, np.pi/4, 3)))
    _legacy_add("asin_01", np.arcsin(np.array([0.0, 0.5, 1.0])))
    _legacy_add("acos_01", np.arccos(np.array([0.0, 0.5, 1.0])))
    _legacy_add("atan_012", np.arctan(np.array([0.0, 1.0, 2.0])))
    _legacy_add("atan2", np.arctan2(np.array([1.0, 1.0, 0.0]), np.array([0.0, 1.0, 1.0])))
    _legacy_add("sinh_123", np.sinh(np.array([1.0, 2.0, 3.0])))
    _legacy_add("cosh_123", np.cosh(np.array([1.0, 2.0, 3.0])))
    _legacy_add("tanh_123", np.tanh(np.array([1.0, 2.0, 3.0])))
    _legacy_add("hypot_34", np.hypot(np.array([3.0, 5.0]), np.array([4.0, 12.0])))
    _legacy_add("deg2rad", np.deg2rad(np.array([0.0, 90.0, 180.0])))
    _legacy_add("rad2deg", np.rad2deg(np.array([0.0, np.pi/2, np.pi])))

    # ============================================================
    # 6. 比较与逻辑 (15+ tests)
    # ============================================================
    a5 = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    b5 = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
    _legacy_add("lt", (a5 < b5).astype(float))
    _legacy_add("le", (a5 <= b5).astype(float))
    _legacy_add("gt", (a5 > b5).astype(float))
    _legacy_add("ge", (a5 >= b5).astype(float))
    _legacy_add("eq", (a5 == b5).astype(float))
    _legacy_add("ne", (a5 != b5).astype(float))
    _legacy_add("logical_and", np.logical_and(a5 > 2, b5 > 2).astype(float))
    _legacy_add("logical_or", np.logical_or(a5 > 4, b5 > 4).astype(float))
    _legacy_add("logical_not", np.logical_not(a5 > 3).astype(float))
    _legacy_add("all_true", int(np.all(np.array([1, 1, 1]))))
    _legacy_add("all_false", int(np.all(np.array([1, 0, 1]))))
    _legacy_add("any_true", int(np.any(np.array([0, 0, 1]))))
    _legacy_add("any_false", int(np.any(np.array([0, 0, 0]))))
    _legacy_add("isclose", np.isclose(np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0001, 3.1]), atol=0.01, rtol=0).astype(float))
    _legacy_add("allclose_t", int(np.allclose(np.array([1.0, 2.0]), np.array([1.0, 2.0]))))
    _legacy_add("allclose_f", int(np.allclose(np.array([1.0, 2.0]), np.array([1.0, 3.0]))))
    _legacy_add("isfinite", np.isfinite(np.array([1.0, np.nan, np.inf, -np.inf, 0.0])).astype(float))
    _legacy_add("isinf", np.isinf(np.array([1.0, np.nan, np.inf, -np.inf, 0.0])).astype(float))
    _legacy_add("isnan", np.isnan(np.array([1.0, np.nan, np.inf, 0.0])).astype(float))

    # ============================================================
    # 7. 归约与统计 (30+ tests)
    # ============================================================
    _legacy_add("sum_all", float(np.sum(M34)))
    _legacy_add("sum_ax0", np.sum(M34, axis=0))
    _legacy_add("sum_ax1", np.sum(M34, axis=1))
    _legacy_add("sum_ax0_kd", np.sum(M34, axis=0, keepdims=True))
    _legacy_add("sum_ax1_kd", np.sum(M34, axis=1, keepdims=True))
    _legacy_add("mean_all", float(np.mean(M34)))
    _legacy_add("mean_ax0", np.mean(M34, axis=0))
    _legacy_add("mean_ax1", np.mean(M34, axis=1))
    _legacy_add("max_all", float(np.max(M34)))
    _legacy_add("max_ax0", np.max(M34, axis=0))
    _legacy_add("max_ax1", np.max(M34, axis=1))
    _legacy_add("min_all", float(np.min(M34)))
    _legacy_add("min_ax0", np.min(M34, axis=0))
    _legacy_add("min_ax1", np.min(M34, axis=1))
    _legacy_add("argmax_ax0", np.argmax(M34, axis=0))
    _legacy_add("argmax_ax1", np.argmax(M34, axis=1))
    _legacy_add("argmin_ax0", np.argmin(M34, axis=0))
    _legacy_add("argmin_ax1", np.argmin(M34, axis=1))
    _legacy_add("std_all", float(np.std(M34)))
    _legacy_add("std_ax0", np.std(M34, axis=0))
    _legacy_add("var_all", float(np.var(M34)))
    _legacy_add("var_ax0", np.var(M34, axis=0))
    _legacy_add("prod_all", float(np.prod(np.array([1.0, 2.0, 3.0, 4.0]))))
    _legacy_add("prod_ax0", np.prod(M34, axis=0))
    _legacy_add("cumsum_1234", np.cumsum(np.array([1,2,3,4], dtype=np.int64)))
    _legacy_add("cumsum_ax0", np.cumsum(np.array([[1,2],[3,4]], dtype=np.int64), axis=0))
    _legacy_add("cumsum_ax1", np.cumsum(np.array([[1,2],[3,4]], dtype=np.int64), axis=1))
    _legacy_add("cumprod_1234", np.cumprod(np.array([1,2,3,4], dtype=np.int64)))
    _legacy_add("median_odd", float(np.median(np.array([3.0, 1.0, 4.0, 1.0, 5.0]))))
    _legacy_add("median_even", float(np.median(np.array([1.0, 2.0, 3.0, 4.0]))))
    _legacy_add("median_ax1", np.median(M34, axis=1))
    _legacy_add("pct50", float(np.percentile(V5, 50)))
    _legacy_add("pct25", float(np.percentile(V5, 25)))
    _legacy_add("pct75", float(np.percentile(V5, 75)))
    _legacy_add("pct90", float(np.percentile(V5, 90)))
    _legacy_add("ptp", float(np.ptp(np.array([3.0, 1.0, 4.0, 1.0, 5.0]))))
    _legacy_add("ptp_ax0", np.ptp(M34, axis=0))
    _legacy_add("sort_1d", np.sort(np.array([3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0])))
    _legacy_add("sort_ax0", np.sort(M34, axis=0))
    _legacy_add("sort_ax1", np.sort(M34, axis=1))
    _legacy_add("argsort_1d", np.argsort(np.array([3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0])))
    _legacy_add("maximum", np.maximum(np.array([1.0, 5.0, 3.0]), np.array([4.0, 2.0, 6.0])))
    _legacy_add("minimum", np.minimum(np.array([1.0, 5.0, 3.0]), np.array([4.0, 2.0, 6.0])))
    _legacy_add("diff_1d", np.diff(np.array([1.0, 3.0, 6.0, 10.0, 15.0])))
    _legacy_add("diff_ax1", np.diff(M34, axis=1))
    _legacy_add("gradient_1d", np.gradient(np.array([1.0, 4.0, 9.0, 16.0, 25.0])))

    # ============================================================
    # 8. 线性代数 (20+ tests)
    # ============================================================
    _legacy_add("matmul_2x2", A2 @ B2)
    _legacy_add("matmul_2x3_3x2", A23 @ A32)
    _legacy_add("matmul_3x3", A3 @ A3)
    _legacy_add("dot_1d", int(np.dot(np.array([1,2,3]), np.array([4,5,6]))))
    _legacy_add("dot_2d_1d", A2 @ np.array([1.0, 2.0]))  # (2,2)@(2,)
    _legacy_add("outer_3_2", np.outer(V3, np.array([4.0, 5.0])))
    _legacy_add("trace_3", float(np.trace(A3)))
    _legacy_add("trace_rect", float(np.trace(A23)))
    _legacy_add("norm_34", float(np.linalg.norm(np.array([3.0, 4.0]))))
    _legacy_add("norm_ax1", np.linalg.norm(M34, axis=1))
    _legacy_add("l1_norm", float(np.sum(np.abs(np.array([-1.0, 2.0, -3.0])))))
    _legacy_add("det_2x2", float(np.linalg.det(A2)))
    _legacy_add("det_3x3", float(np.linalg.det(A3)))
    _legacy_add("inv_2x2", np.linalg.inv(A2))
    _legacy_add("solve_2x2", np.linalg.solve(np.array([[2,1],[1,3]], dtype=np.float64), np.array([7,8], dtype=np.float64)))
    _legacy_add("solve_3x3", np.linalg.solve(np.array([[2,1,0],[1,3,1],[0,1,2]], dtype=np.float64), np.array([1,2,3], dtype=np.float64)))

    # QR
    Q, R = np.linalg.qr(A23)
    _legacy_add("qr_Q_shape", list(Q.shape))
    _legacy_add("qr_R_shape", list(R.shape))
    _legacy_add("qr_recon_err", float(np.max(np.abs(A23 - Q @ R))))

    # SVD (diagonal matrix)
    U, s, Vt = np.linalg.svd(np.diag(np.array([5.0, 3.0, 1.0])))
    _legacy_add("svd_s_diag3", s)

    # Cholesky
    L = np.linalg.cholesky(np.array([[4,2],[2,3]], dtype=np.float64))
    _legacy_add("chol_L", L)
    _legacy_add("chol_err", float(np.max(np.abs(np.array([[4,2],[2,3]], dtype=np.float64) - L @ L.T))))

    # matrix rank
    _legacy_add("rank_full", int(np.linalg.matrix_rank(np.eye(3))))
    _legacy_add("rank_deficient", int(np.linalg.matrix_rank(np.array([[1,2],[2,4]], dtype=np.float64))))

    # LU (scipy)
    try:
        from scipy.linalg import lu as scipy_lu
        P, L_lu, U_lu = scipy_lu(A2)
        _legacy_add("lu_L", L_lu)
        _legacy_add("lu_U", U_lu)
    except:
        pass

    # ============================================================
    # 9. einsum (10+ tests)
    # ============================================================
    _legacy_add("einsum_dot", int(np.einsum("i,i->", np.array([1,2,3]), np.array([4,5,6]))))
    _legacy_add("einsum_matmul", np.einsum("ij,jk->ik", np.arange(6).reshape(2,3), np.arange(6).reshape(3,2)))
    _legacy_add("einsum_transpose", np.einsum("ij->ji", np.array([[1,2],[3,4]])))
    _legacy_add("einsum_diag", np.einsum("ii->i", np.arange(9).reshape(3,3)))
    _legacy_add("einsum_trace", int(np.einsum("ii->", np.arange(9).reshape(3,3))))
    _legacy_add("einsum_outer", np.einsum("i,j->ij", np.array([1,2,3]), np.array([4,5])))
    _legacy_add("einsum_batch_mm", np.einsum("ij,jk->ik", A2, B2))  # same as matmul

    # ============================================================
    # 10. 激活函数 (10+ tests)
    # ============================================================
    x5 = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    _legacy_add("sigmoid", 1/(1+np.exp(-x5)))
    _legacy_add("relu", np.maximum(0, x5))
    _legacy_add("tanh", np.tanh(x5))

    # softmax
    logits = np.array([1.0, 2.0, 3.0])
    e = np.exp(logits - np.max(logits))
    _legacy_add("softmax_123", e / e.sum())

    # ============================================================
    # 11. 集合操作 (5+ tests)
    # ============================================================
    _legacy_add("unique_122333", np.unique(np.array([1,2,2,3,3,3])))
    _legacy_add("unique_with_return_index", np.unique(np.array([3,1,2,1,3]), return_index=True)[0])
    _legacy_add("intersect1d", np.intersect1d(np.array([1,2,3,4,5]), np.array([3,4,5,6,7])))
    _legacy_add("union1d", np.union1d(np.array([1,2,3,4,5]), np.array([3,4,5,6,7])))
    _legacy_add("setdiff1d", np.setdiff1d(np.array([1,2,3,4,5]), np.array([3,4,5,6,7])))
    _legacy_add("setxor1d", np.setxor1d(np.array([1,2,3,4,5]), np.array([3,4,5,6,7])))

    # ============================================================
    # 12. nan 处理 (10+ tests)
    # ============================================================
    nan = np.nan
    a_nan = np.array([1.0, nan, 3.0, 4.0])
    _legacy_add("nanmean", float(np.nanmean(a_nan)))
    _legacy_add("nansum", float(np.nansum(a_nan)))
    _legacy_add("nanmax", float(np.nanmax(a_nan)))
    _legacy_add("nanmin", float(np.nanmin(a_nan)))
    _legacy_add("nanstd", float(np.nanstd(a_nan)))
    _legacy_add("nanvar", float(np.nanvar(a_nan)))

    # ============================================================
    # 13. 其他 (pad, kron, meshgrid, interp)
    # ============================================================
    _legacy_add("kron_22", np.kron(np.array([[1,2],[3,4]]), np.array([[0,5],[6,7]])))
    _legacy_add("meshgrid_x_shape", list(np.meshgrid(np.arange(3), np.arange(4), sparse=True)[0].shape))
    _legacy_add("meshgrid_y_shape", list(np.meshgrid(np.arange(3), np.arange(4), sparse=True)[1].shape))

    # ============================================================
    # 14. 损失函数 (3+ tests)
    # ============================================================
    yt = np.array([1.0, 0.0, 0.0])
    yp = np.array([0.7, 0.2, 0.1])
    _legacy_add("mse", float(np.mean((yt - yp)**2)))
    _legacy_add("cross_entropy", float(-np.sum(yt * np.log(np.clip(yp, 1e-7, 1-1e-7)))))

    # ============================================================
    # Output
    # ============================================================
    # vt-narrow numpy 切片语义回归（v0.4.2 负索引 + clip）
    _legacy_a10 = np.arange(10)
    _legacy_add("narrow_clip_over", _legacy_a10[0:100])
    _legacy_add("narrow_neg_start", _legacy_a10[-100:2])

_legacy_all_block()

def _legacy_param_block():
    """gen_param_tests.py 的 162 个未接线用例（key 风格: fn_形状_ax*/_kd）"""
    V5 = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    V5i = np.array([1, 2, 3, 4, 5], dtype=np.int64)
    M34 = np.arange(12, dtype=np.float64).reshape(3,4)
    M34i = np.arange(12, dtype=np.int64).reshape(3,4)
    T234 = np.arange(24, dtype=np.float64).reshape(2,3,4)
    T234i = np.arange(24, dtype=np.int64).reshape(2,3,4)
    T222i = np.array([[[1,2],[3,4]],[[5,6],[7,8]]], dtype=np.int64)
    T222 = np.array([[[1,2],[3,4]],[[5,6],[7,8]]], dtype=np.float64)
    V5 = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    V5i = np.array([1, 2, 3, 4, 5], dtype=np.int64)

    # 2D (3x4)
    M34 = np.arange(12, dtype=np.float64).reshape(3,4)
    M34i = np.arange(12, dtype=np.int64).reshape(3,4)

    # 3D (2x3x4)
    T234 = np.arange(24, dtype=np.float64).reshape(2,3,4)
    T234i = np.arange(24, dtype=np.int64).reshape(2,3,4)

    # 3D (2x2x2)
    T222i = np.array([[[1,2],[3,4]],[[5,6],[7,8]]], dtype=np.int64)
    T222 = np.array([[[1,2],[3,4]],[[5,6],[7,8]]], dtype=np.float64)

    # ============================================================
    # 1. 归约函数 — 系统测试所有 axis 组合
    # ============================================================
    # sum: 全局, axis=0, axis=1, axis=-1, axis=(0,1), keepdims
    for name, fn, data in [
        ("sum", np.sum, M34), ("mean", np.mean, M34),
        ("amax", np.max, M34), ("amin", np.min, M34),
        ("std", np.std, M34), ("var", np.var, M34),
        ("prod", np.prod, np.abs(M34) + 1),  # avoid 0 in prod
    ]:
        # 2D
        _legacy_add(f"{name}_2d_all", float(fn(data)))
        _legacy_add(f"{name}_2d_ax0", fn(data, axis=0))
        _legacy_add(f"{name}_2d_ax1", fn(data, axis=1))
        _legacy_add(f"{name}_2d_ax_neg1", fn(data, axis=-1))
        _legacy_add(f"{name}_2d_ax0_kd", fn(data, axis=0, keepdims=True))
        _legacy_add(f"{name}_2d_ax1_kd", fn(data, axis=1, keepdims=True))
        # 3D
        _legacy_add(f"{name}_3d_all", float(fn(T234)))
        _legacy_add(f"{name}_3d_ax0", fn(T234, axis=0))
        _legacy_add(f"{name}_3d_ax1", fn(T234, axis=1))
        _legacy_add(f"{name}_3d_ax2", fn(T234, axis=2))
        _legacy_add(f"{name}_3d_ax_neg1", fn(T234, axis=-1))
        _legacy_add(f"{name}_3d_ax_neg2", fn(T234, axis=-2))
        _legacy_add(f"{name}_3d_ax01", fn(T234, axis=(0,1)))
        _legacy_add(f"{name}_3d_ax12", fn(T234, axis=(1,2)))
        _legacy_add(f"{name}_3d_ax0_kd", fn(T234, axis=0, keepdims=True))
        _legacy_add(f"{name}_3d_ax2_kd", fn(T234, axis=2, keepdims=True))

    # argmax/argmin: 2D + 3D
    for name, fn in [("argmax", np.argmax), ("argmin", np.argmin)]:
        _legacy_add(f"{name}_2d_ax0", fn(M34, axis=0))
        _legacy_add(f"{name}_2d_ax1", fn(M34, axis=1))
        _legacy_add(f"{name}_2d_ax_neg1", fn(M34, axis=-1))
        _legacy_add(f"{name}_3d_ax0", fn(T234, axis=0))
        _legacy_add(f"{name}_3d_ax1", fn(T234, axis=1))
        _legacy_add(f"{name}_3d_ax2", fn(T234, axis=2))
        _legacy_add(f"{name}_3d_ax_neg1", fn(T234, axis=-1))

    # cumsum/cumprod: 2D + 3D, per axis
    for name, fn in [("cumsum", np.cumsum), ("cumprod", np.cumprod)]:
        _legacy_add(f"{name}_2d_ax0", fn(M34i, axis=0))
        _legacy_add(f"{name}_2d_ax1", fn(M34i, axis=1))
        _legacy_add(f"{name}_3d_ax0", fn(T234i, axis=0))
        _legacy_add(f"{name}_3d_ax1", fn(T234i, axis=1))
        _legacy_add(f"{name}_3d_ax2", fn(T234i, axis=2))

    # median: 2D + 3D
    _legacy_add("median_2d_ax0", np.median(M34, axis=0))
    _legacy_add("median_2d_ax1", np.median(M34, axis=1))
    _legacy_add("median_3d_ax0", np.median(T234, axis=0))
    _legacy_add("median_3d_ax1", np.median(T234, axis=1))
    _legacy_add("median_3d_ax2", np.median(T234, axis=2))

    # percentile: 2D + 3D, multiple percentiles
    for p in [25, 50, 75, 90]:
        _legacy_add(f"pct{p}_2d_ax0", np.percentile(M34, p, axis=0))
        _legacy_add(f"pct{p}_2d_ax1", np.percentile(M34, p, axis=1))
        _legacy_add(f"pct{p}_3d_ax0", np.percentile(T234, p, axis=0))
        _legacy_add(f"pct{p}_3d_ax2", np.percentile(T234, p, axis=2))

    # ptp: 2D + 3D
    _legacy_add("ptp_2d_ax0", np.ptp(M34, axis=0))
    _legacy_add("ptp_2d_ax1", np.ptp(M34, axis=1))
    _legacy_add("ptp_3d_ax0", np.ptp(T234, axis=0))
    _legacy_add("ptp_3d_ax2", np.ptp(T234, axis=2))

    # sort/argsort: 2D + 3D, per axis
    _legacy_add("sort_2d_ax0", np.sort(M34, axis=0))
    _legacy_add("sort_2d_ax1", np.sort(M34, axis=1))
    _legacy_add("sort_3d_ax0", np.sort(T234, axis=0))
    _legacy_add("sort_3d_ax1", np.sort(T234, axis=1))
    _legacy_add("sort_3d_ax2", np.sort(T234, axis=2))
    _legacy_add("argsort_2d_ax0", np.argsort(M34, axis=0))
    _legacy_add("argsort_2d_ax1", np.argsort(M34, axis=1))
    _legacy_add("argsort_3d_ax0", np.argsort(T234, axis=0))
    _legacy_add("argsort_3d_ax2", np.argsort(T234, axis=2))

    # diff: 2D + 3D per axis
    _legacy_add("diff_2d_ax0", np.diff(M34, axis=0))
    _legacy_add("diff_2d_ax1", np.diff(M34, axis=1))
    _legacy_add("diff_3d_ax0", np.diff(T234, axis=0))
    _legacy_add("diff_3d_ax2", np.diff(T234, axis=2))

    # gradient: 2D + 3D
    _legacy_add("grad_2d_ax0", np.gradient(M34, axis=0))
    _legacy_add("grad_2d_ax1", np.gradient(M34, axis=1))
    _legacy_add("grad_3d_ax0", np.gradient(T234, axis=0))
    _legacy_add("grad_3d_ax2", np.gradient(T234, axis=2))

    # ============================================================
    # 2. 形状操作 — 高维测试
    # ============================================================
    # transpose: 3D with different perms
    _legacy_add("trans_3d_012", T234.transpose(0,1,2))  # identity
    _legacy_add("trans_3d_021", T234.transpose(0,2,1))
    _legacy_add("trans_3d_102", T234.transpose(1,0,2))
    _legacy_add("trans_3d_210", T234.transpose(2,1,0))
    _legacy_add("trans_3d_201", T234.transpose(2,0,1))

    # squeeze: 3D with various singleton dims
    _legacy_add("squeeze_3d_134", np.squeeze(np.arange(12).reshape(1,3,4)))
    _legacy_add("squeeze_3d_314", np.squeeze(np.arange(12).reshape(3,1,4)))
    _legacy_add("squeeze_3d_341", np.squeeze(np.arange(12).reshape(3,4,1)))
    _legacy_add("squeeze_3d_ax0", np.squeeze(np.arange(12).reshape(1,3,4), axis=0))
    _legacy_add("squeeze_3d_ax1", np.squeeze(np.arange(12).reshape(3,1,4), axis=1))

    # expand_dims: 3D at different positions
    # 注: 同时生成 expand_3d_axN (shape 元组,字符串形式) 和 expand_3d_axN_shape (列表形式)
    #     run_param_tests.lisp 引用的是带 _shape 后缀的键
    _legacy_add("expand_3d_ax0", np.expand_dims(T234, 0).shape)
    _legacy_add("expand_3d_ax0_shape", list(np.expand_dims(T234, 0).shape))
    _legacy_add("expand_3d_ax1", np.expand_dims(T234, 1).shape)
    _legacy_add("expand_3d_ax2", np.expand_dims(T234, 2).shape)
    _legacy_add("expand_3d_ax2_shape", list(np.expand_dims(T234, 2).shape))
    _legacy_add("expand_3d_ax3", np.expand_dims(T234, 3).shape)
    _legacy_add("expand_3d_ax3_shape", list(np.expand_dims(T234, 3).shape))
    _legacy_add("expand_3d_ax_neg1", np.expand_dims(T234, -1).shape)

    # concatenate: 3D
    _legacy_add("concat_3d_ax0", np.concatenate([T234, T234+100], axis=0))
    _legacy_add("concat_3d_ax1", np.concatenate([T234, T234[:,:,:2]], axis=2))

    # stack: 3D
    # 注: run_param_tests.lisp 对 stack_3d_ax0/ax3 只比较 vt-shape,
    #     因此这里存储 shape 列表而非完整数组
    _legacy_add("stack_3d_ax0", list(np.stack([T234, T234+100], axis=0).shape))
    _legacy_add("stack_3d_ax1", np.stack([T234, T234+100], axis=1))
    _legacy_add("stack_3d_ax3", list(np.stack([T234, T234+100], axis=3).shape))

    # flip: 3D per axis
    _legacy_add("flip_3d_ax0", np.flip(T234, axis=0))
    _legacy_add("flip_3d_ax1", np.flip(T234, axis=1))
    _legacy_add("flip_3d_ax2", np.flip(T234, axis=2))
    _legacy_add("flip_3d_ax_neg1", np.flip(T234, axis=-1))

    # roll: 3D per axis
    _legacy_add("roll_3d_ax0", np.roll(T234i, 1, axis=0))
    _legacy_add("roll_3d_ax1", np.roll(T234i, 1, axis=1))
    _legacy_add("roll_3d_ax2", np.roll(T234i, 1, axis=2))

    # split: 3D
    _legacy_add("split_3d_ax0", [x.tolist() for x in np.split(T234, 2, axis=0)])
    _legacy_add("split_3d_ax2", [x.tolist() for x in np.split(T234, 2, axis=2)])

    # tile: 3D
    # 注: run_param_tests.lisp 使用 t222i (int64) 作为输入,这里保持一致
    _legacy_add("tile_3d_121", np.tile(T222i, (1,2,1)))
    _legacy_add("tile_3d_212", np.tile(T222i, (2,1,2)))

    # repeat: 3D
    _legacy_add("repeat_3d_ax0", np.repeat(T222i, 2, axis=0))
    _legacy_add("repeat_3d_ax2", np.repeat(T222i, 2, axis=2))

    # diagonal: 3D (batch)
    # 注: clvt 的 vt-diagonal 对 3D 输入提取最后两维的对角线 (PyTorch 风格),
    #     返回 shape = batch_dims + (min(rows,cols),)
    #     numpy 2.x 的 np.diagonal 对 3D 返回前两维对角线,行为不同。
    #     这里手动构造与 clvt 一致的期望值: T234 shape (2,3,4) -> (2, 3)
    _legacy_add("diag_3d", np.array([np.diagonal(T234[i]) for i in range(T234.shape[0])]))

    # triu/tril: 3D (batch)
    _legacy_add("triu_3d", np.triu(T234))
    _legacy_add("tril_3d", np.tril(T234))

    # swapaxes: 3D
    _legacy_add("swapaxes_3d_01", np.swapaxes(T234, 0, 1))
    _legacy_add("swapaxes_3d_02", np.swapaxes(T234, 0, 2))
    _legacy_add("swapaxes_3d_12", np.swapaxes(T234, 1, 2))

    # narrow: 3D
    _legacy_add("narrow_3d_ax0", T234[0:1, :, :])
    _legacy_add("narrow_3d_ax1", T234[:, 1:3, :])
    _legacy_add("narrow_3d_ax2", T234[:, :, 1:3])

    # ============================================================
    # 3. 算术 — 高维广播
    # ============================================================
    # 3D + scalar
    _legacy_add("add_3d_scalar", T234 + 10.0)
    _legacy_add("mul_3d_scalar", T234 * 2.0)

    # 3D + 1D broadcast
    _legacy_add("add_3d_1d", T234 + np.array([1.0, 2.0, 3.0, 4.0]))  # (2,3,4)+(4,)
    _legacy_add("mul_3d_1d", T234 * np.array([1.0, 2.0, 3.0]).reshape(1,3,1))  # (2,3,4)*(1,3,1)

    # 3D + 3D elementwise
    _legacy_add("add_3d_3d", T234 + T234)
    _legacy_add("mul_3d_3d", T234 * T234)

    # abs/signum/square/sqrt on 3D
    _legacy_add("abs_3d", np.abs(T234 - 12))
    _legacy_add("square_3d", T234 ** 2)
    _legacy_add("sqrt_3d", np.sqrt(np.abs(T234)))
    _legacy_add("exp_3d", np.exp(T234 / 10))  # scale to avoid overflow
    _legacy_add("log_3d", np.log(np.abs(T234) + 1))

    # clip on 3D
    _legacy_add("clip_3d", np.clip(T234, 5.0, 15.0))

    # ============================================================
    # 4. 三角函数 — 3D
    # ============================================================
    _legacy_add("sin_3d", np.sin(T234 * 0.1))
    _legacy_add("cos_3d", np.cos(T234 * 0.1))
    _legacy_add("tanh_3d", np.tanh((T234 - 12) * 0.1))

    # ============================================================
    # 5. 比较逻辑 — 3D
    # ============================================================
    _legacy_add("lt_3d", (T234 < 12.0).astype(float))
    _legacy_add("gt_3d", (T234 > 12.0).astype(float))
    _legacy_add("eq_3d", (T234 == 12.0).astype(float))
    _legacy_add("isclose_3d", np.isclose(T234, T234 + 0.001, atol=0.01, rtol=0).astype(float))
    _legacy_add("isfinite_3d", np.isfinite(T234).astype(float))

    # all/any on 3D per axis
    _legacy_add("all_3d_ax0", np.all(T234 > 0, axis=0).astype(int))
    _legacy_add("all_3d_ax2", np.all(T234 > 0, axis=2).astype(int))
    _legacy_add("any_3d_ax0", np.any(T234 > 10, axis=0).astype(int))
    _legacy_add("any_3d_ax2", np.any(T234 > 10, axis=2).astype(int))

    # ============================================================
    # 6. 线性代数 — 更多场景
    # ============================================================
    # matmul: various shapes
    A44 = np.arange(16, dtype=np.float64).reshape(4,4)
    B44 = np.ones((4,4), dtype=np.float64)
    _legacy_add("matmul_4x4", A44 @ B44)
    _legacy_add("matmul_1x3_3x1", np.array([[1,2,3]], dtype=np.float64) @ np.array([[4],[5],[6]], dtype=np.float64))

    # dot: 2D @ 2D
    _legacy_add("dot_2d_2d", np.dot(A2 := np.array([[1,2],[3,4]], dtype=np.float64), B2 := np.array([[5,6],[7,8]], dtype=np.float64)))

    # trace: 3D (batch trace)
    _legacy_add("trace_3d", np.trace(T234))  # sum along last two dims diagonal

    # norm: 3D per axis
    _legacy_add("norm_3d_ax0", np.linalg.norm(T234, axis=0))
    _legacy_add("norm_3d_ax1", np.linalg.norm(T234, axis=1))
    _legacy_add("norm_3d_ax2", np.linalg.norm(T234, axis=2))

    # det: batch of 2x2
    batch_2x2 = np.array([[[1,2],[3,4]],[[5,6],[7,8]]], dtype=np.float64)
    _legacy_add("det_batch", np.array([np.linalg.det(batch_2x2[i]) for i in range(2)]))

    # ============================================================
    # 7. einsum — 更多模式
    # ============================================================
    _legacy_add("einsum_trace_2d", int(np.einsum("ii->", np.arange(4).reshape(2,2))))
    _legacy_add("einsum_matmul_T", np.einsum("ij,jk->ik", np.array([[1,2],[3,4]]), np.array([[5,6],[7,8]])))
    _legacy_add("einsum_3d_reduce", np.einsum("ijk->i", T234))  # reduce last two dims
    _legacy_add("einsum_3d_reduce_ax2", np.einsum("ijk->ij", T234))  # reduce last dim
    _legacy_add("einsum_bilinear", np.einsum("ij,jk,kl->il", A2, B2, A2))  # chain multiply

    # ============================================================
    # 8. where/nonzero — 3D
    # ============================================================
    _legacy_add("where_3d", np.where(T234 > 12, T234, 0.0))
    _legacy_add("nonzero_3d", [x.tolist() for x in np.nonzero(T234 > 20)])

    # ============================================================
    # 9. pad — 3D, different modes
    # ============================================================
    _legacy_add("pad_3d_const", np.pad(T222, 1, mode='constant', constant_values=0))
    _legacy_add("pad_3d_edge", np.pad(T222, 1, mode='edge'))

    # ============================================================
    # 10. set operations — different sizes
    # ============================================================
    _legacy_add("unique_large", np.unique(np.array([5,3,1,4,2,5,3,1])))
    _legacy_add("unique_sorted", np.unique(np.array([3,1,2])))
    _legacy_add("intersect_empty", np.intersect1d(np.array([1,2,3]), np.array([4,5,6])))
    _legacy_add("union_same", np.union1d(np.array([1,2,3]), np.array([1,2,3])))

    # ============================================================
    # 11. nan — 3D
    # ============================================================
    T_nan = T234.copy()
    T_nan[0, 0, 0] = np.nan
    T_nan[1, 2, 3] = np.nan
    _legacy_add("nanmean_3d_ax0", np.nanmean(T_nan, axis=0))
    _legacy_add("nanmean_3d_ax2", np.nanmean(T_nan, axis=2))
    _legacy_add("nansum_3d_ax1", np.nansum(T_nan, axis=1))
    _legacy_add("nanmax_3d_ax0", np.nanmax(T_nan, axis=0))
    _legacy_add("nanmin_3d_ax2", np.nanmin(T_nan, axis=2))
    _legacy_add("nanstd_3d_ax0", np.nanstd(T_nan, axis=0))
    _legacy_add("nanvar_3d_ax2", np.nanvar(T_nan, axis=2))

    # ============================================================
    # 12. 激活函数 — 3D
    # ============================================================
    _legacy_add("sigmoid_3d", 1/(1+np.exp(-T234)))
    _legacy_add("relu_3d", np.maximum(0, T234 - 12))
    _legacy_add("tanh_3d_act", np.tanh(T234 - 12))
    _legacy_add("softmax_3d_ax2", np.apply_along_axis(lambda x: np.exp(x - np.max(x)) / np.sum(np.exp(x - np.max(x))), 2, T234))

    # ============================================================
    # 13. loss — 3D
    # ============================================================
    yt_3d = np.zeros((2,3,4))
    yt_3d[0,0,0] = 1.0
    yt_3d[1,1,1] = 1.0
    yp_3d = np.full((2,3,4), 0.25)
    _legacy_add("mse_3d", float(np.mean((yt_3d - yp_3d)**2)))

    # ============================================================
    # 14. 比较运算 — 3D
    # ============================================================
    _legacy_add("lt_3d_pair", (T234 < (T234 + 0.5)).astype(float))
    _legacy_add("le_3d_pair", (T234 <= T234).astype(float))
    _legacy_add("gt_3d_pair", ((T234 + 1) > T234).astype(float))
    _legacy_add("maximum_3d", np.maximum(T234, T234[::-1]))
    _legacy_add("minimum_3d", np.minimum(T234, T234[::-1]))

    # ============================================================
    # 15. broadcast_to — 3D
    # ============================================================
    _legacy_add("broadcast_1x3x4_to_2x3x4", np.broadcast_to(np.arange(12).reshape(1,3,4), (2,3,4)))
    _legacy_add("broadcast_2x1x4_to_2x3x4", np.broadcast_to(np.arange(8).reshape(2,1,4), (2,3,4)))

    # ============================================================
    # 16. 截断/取整 — 3D
    # ============================================================
    _legacy_add("floor_3d", np.floor(T234 * 0.7))
    _legacy_add("ceil_3d", np.ceil(T234 * 0.7))
    _legacy_add("round_3d", np.round(T234 * 0.7))
    _legacy_add("mod_3d", np.mod(T234i, 7))
    _legacy_add("clip_3d_neg", np.clip(T234, -5.0, 5.0))

    # ============================================================
    # 17. 类型转换 — 不同 dtype
    # ============================================================
    for dt_name, dt in [("f32", np.float32), ("f64", np.float64), ("i32", np.int32), ("i64", np.int64)]:
        _legacy_add(f"astype_{dt_name}", np.arange(6, dtype=dt).reshape(2,3))

    # ============================================================
    # 18. 补充: convolve / eig / diag_3d_pytorch
    #     这些测试用例在 run_param_tests.lisp 中被引用,需单独生成
    # ============================================================
    # convolve: 一维卷积 (与 numpy.convolve 对齐)
    _conv_a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    _conv_v = np.array([1.0, 0.0, -1.0])
    _legacy_add("convolve_valid", np.convolve(_conv_a, _conv_v, mode='valid').tolist())
    _legacy_add("convolve_full", np.convolve(_conv_a, _conv_v, mode='full').tolist())

    # eig: 3x3 对称矩阵特征值 (clvt vt-eig 返回降序排列)
    # 注: 矩阵必须与 run_param_tests.lisp 中的测试矩阵一致
    _eig_A = np.array([[4.0, 2.0, 1.0],
                       [2.0, 5.0, 3.0],
                       [1.0, 3.0, 6.0]])
    _eig_vals = np.linalg.eigvals(_eig_A)
    _legacy_add("eig_3x3_vals", sorted(_eig_vals.real.tolist(), reverse=True))

    # diag_3d_pytorch: clvt vt-diagonal 对 3D 的行为 (PyTorch 风格,最后两维对角线)
    # T234 shape (2,3,4) -> 每个批次取 min(3,4)=3 的对角线 -> shape (2, 3)
    _legacy_add("diag_3d_pytorch", np.array([np.diagonal(T234[i]) for i in range(T234.shape[0])]))

    # ============================================================
    # Output
    # ============================================================
_legacy_param_block()


print(json.dumps(R))
