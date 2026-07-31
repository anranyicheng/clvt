#!/usr/bin/env python3
"""
gen_all_tests.py — 为 clvt 所有函数生成全面测试用例
每个函数测试多种参数组合，输出 test/all_expected.json
"""
import numpy as np
import json, sys

RESULTS = {}

def add(key, val):
    """统一存储期望值"""
    if isinstance(val, np.ndarray):
        RESULTS[key] = {"t": "a", "s": list(val.shape), "d": str(val.dtype), "v": val.tolist()}
    elif isinstance(val, (int, np.integer)):
        RESULTS[key] = {"t": "i", "v": int(val)}
    elif isinstance(val, (float, np.floating)):
        RESULTS[key] = {"t": "f", "v": float(val)}
    elif isinstance(val, list):
        RESULTS[key] = {"t": "l", "v": val}
    elif val is None:
        RESULTS[key] = {"t": "n", "v": None}
    else:
        try:
            RESULTS[key] = {"t": "f", "v": float(val)}
        except:
            RESULTS[key] = {"t": "s", "v": str(val)}

# ============================================================
# 辅助数据
# ============================================================
A2 = np.array([[1,2],[3,4]], dtype=np.float64)
B2 = np.array([[5,6],[7,8]], dtype=np.float64)
A3 = np.array([[1,2,3],[4,5,6],[7,8,9]], dtype=np.float64)
A23 = np.array([[1,2,3],[4,5,6]], dtype=np.float64)
A32 = np.array([[1,2],[3,4],[5,6]], dtype=np.float64)
V3 = np.array([1.0, 2.0, 3.0])
V5 = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
V3i = np.array([1, 2, 3], dtype=np.int64)
V5i = np.array([1, 2, 3, 4, 5], dtype=np.int64)
M34 = np.arange(12, dtype=np.float64).reshape(3,4)
M234 = np.arange(24, dtype=np.float64).reshape(2,3,4)

# ============================================================
# 1. 张量创建 (20+ tests)
# ============================================================
add("arange_10", np.arange(10, dtype=np.int64))
add("arange_5_s2", np.arange(5, dtype=np.float64))  # start=0 default
add("linspace_0_1_5", np.linspace(0, 1, 5))
add("linspace_0_10_3", np.linspace(0, 10, 3))
add("linspace_0_1_5_ne", np.linspace(0, 1, 5, endpoint=False))
add("logspace_0_3_4", np.logspace(0, 3, 4))
add("logspace_0_2_3", np.logspace(0, 2, 3))
add("eye_3", np.eye(3))
add("eye_4x6", np.eye(4, 6))
add("eye_3_k1", np.eye(3, k=1))
add("eye_3_k_neg1", np.eye(3, k=-1))
add("diag_v123", np.diag(np.array([1,2,3])))
add("diag_extract", np.diag(np.arange(9).reshape(3,3)))
add("diag_extract_k1", np.diag(np.arange(9).reshape(3,3), k=1))
add("diag_extract_k_neg1", np.diag(np.arange(9).reshape(3,3), k=-1))
add("zeros_23", np.zeros((2,3)))
add("ones_22", np.ones((2,2)))
add("full_23_7", np.full((2,3), 7.0))
add("identity_3", np.eye(3))

# ============================================================
# 2. 形状操作 (30+ tests)
# ============================================================
add("reshape_6_23", np.arange(6).reshape(2,3))
add("reshape_6_32", np.arange(6).reshape(3,2))
add("reshape_neg1", np.arange(6).reshape(2,-1))
add("transpose_23", np.arange(6).reshape(2,3).T)
add("transpose_perm", np.arange(24).reshape(2,3,4).transpose(1,0,2))
add("transpose_3d", np.arange(24).reshape(2,3,4).transpose())
add("squeeze_123", np.squeeze(np.arange(6).reshape(1,2,3)))
add("squeeze_213", np.squeeze(np.arange(6).reshape(2,1,3)))
add("squeeze_231", np.squeeze(np.arange(6).reshape(2,3,1)))
add("expand_dims_0", np.expand_dims(V3, 0))
add("expand_dims_1", np.expand_dims(V3, 1))
add("expand_dims_neg1", np.expand_dims(V3, -1))
add("flatten", np.arange(6).reshape(2,3).flatten())
add("ravel", np.arange(6).reshape(2,3).ravel())

# concatenate
add("concat_0", np.concatenate([A2, B2], axis=0))
add("concat_1", np.concatenate([A2, B2], axis=1))
add("concat_neg1", np.concatenate([A2, B2], axis=-1))

# stack
add("stack_0", np.stack([V3, V3*2], axis=0))
add("stack_1", np.stack([V3, V3*2], axis=1))

# split
add("split_3_7", [x.tolist() for x in np.split(np.arange(10), [3,7])])
add("split_3", [x.tolist() for x in np.split(np.arange(9), 3)])
add("vsplit", [x.tolist() for x in np.vsplit(np.arange(12).reshape(3,4), 3)])
add("hsplit", [x.tolist() for x in np.hsplit(np.arange(12).reshape(3,4), 2)])

# flip
add("flip_1d", np.flip(V5i))
add("flip_axis0", np.flip(M34, axis=0))
add("flip_axis1", np.flip(M34, axis=1))
add("flip_neg1", np.flip(M34, axis=-1))

# roll
add("roll_2", np.roll(V5i, 2))
add("roll_neg1", np.roll(V5i, -1))
add("roll_axis0", np.roll(M34, 1, axis=0))
add("roll_axis1", np.roll(M34, 1, axis=1))

# triu / tril
add("triu_3", np.triu(A3))
add("triu_3_k1", np.triu(A3, k=1))
add("triu_3_k_neg1", np.triu(A3, k=-1))
add("tril_3", np.tril(A3))
add("tril_3_k1", np.tril(A3, k=1))
add("tril_3_k_neg1", np.tril(A3, k=-1))

# diagonal
add("diag_3", np.diagonal(A3))
add("diag_3_k1", np.diagonal(A3, 1))
add("diag_3_k_neg1", np.diagonal(A3, -1))
add("diag_rect", np.diagonal(A23))

# tile / repeat
add("tile_3", np.tile(V3i, 3))
add("tile_23", np.tile(np.array([[1,2],[3,4]]), (2,3)))
add("repeat_2", np.repeat(V3i, 2))
add("repeat_2_ax1", np.repeat(np.array([[1,2],[3,4]]), 2, axis=1))
add("repeat_list", np.repeat(np.array([1,2,3]), [2,0,1]))

# broadcast_to
add("broadcast_23", np.broadcast_to(V3i, (2,3)))
add("broadcast_123", np.broadcast_to(V3i.reshape(1,3), (2,3)))

# narrow (slice along axis)
add("narrow_34_ax0_1_3", M34[1:3, :])
add("narrow_34_ax1_1_3", M34[:, 1:3])

# swapaxes
add("swapaxes_01", np.swapaxes(M234, 0, 1))
add("swapaxes_02", np.swapaxes(M234, 0, 2))

# rot90
add("rot90_k1", np.rot90(np.arange(4).reshape(2,2)))
add("rot90_k2", np.rot90(np.arange(4).reshape(2,2), k=2))

# pad
add("pad_const_1", np.pad(np.array([[1,2],[3,4]]), 1, mode='constant', constant_values=0))
add("pad_edge_1", np.pad(np.array([[1,2],[3,4]]), 1, mode='edge'))
add("pad_reflect_1", np.pad(np.array([[1,2],[3,4]]), 1, mode='reflect'))

# ============================================================
# 3. 索引与切片 (20+ tests)
# ============================================================
a10 = np.arange(10)
add("slice_2_7", a10[2:7])
add("slice_1_9_2", a10[1:9:2])
add("slice_nil_5", a10[:5])
add("slice_5_nil", a10[5:])
add("slice_reverse", a10[::-1])
add("slice_8_3_neg1", a10[8:3:-1])
add("slice_neg1", int(a10[-1]))
add("slice_neg3_neg1", a10[-3:-1])

b45 = np.arange(20).reshape(4,5)
add("2d_1_2", int(b45[1,2]))
add("2d_row2", b45[2,:])
add("2d_col3", b45[:,3])
add("2d_sub", b45[1:3, 2:4])
add("2d_neg", b45[-2:, -3:])
add("2d_ellipsis", b45[..., :2])
add("2d_newaxis_shape", list(np.expand_dims(b45, 1).shape))

# where
add("where_cxy", np.where(np.array([1,0,1,0]), np.array([10.0,20,30,40]), np.array([100.0,200,300,400])))
add("where_scalar", np.where(np.array([1,0,1,0]), 100.0, 200.0))

# nonzero
add("nonzero_1d", np.nonzero(np.array([0,1,0,2,0,3]))[0])
add("nonzero_2d", [x.tolist() for x in np.nonzero(np.array([[1,0],[0,1]]))])

# ============================================================
# 4. 算术运算 (30+ tests)
# ============================================================
ai = np.array([1,2,3,4], dtype=np.int64)
bi = np.array([5,6,7,8], dtype=np.int64)
af = np.array([1.0, 2.0, 3.0, 4.0])
bf = np.array([5.0, 6.0, 7.0, 8.0])

add("add_ii", ai + bi)
add("sub_ii", bi - ai)
add("mul_ii", ai * bi)
add("add_ff", af + bf)
add("sub_ff", bf - af)
add("mul_ff", af * bf)
add("div_ff", bf / af)
add("add_scalar_i", ai + 10)
add("mul_scalar_i", ai * 2)
add("add_scalar_f", af + 1.5)
add("mul_scalar_f", af * 0.5)
add("neg_f", -af)
add("abs_neg", np.abs(np.array([-3.0, -1.0, 0.0, 1.0, 3.0])))
add("signum", np.sign(np.array([-3.0, -1.0, 0.0, 1.0, 3.0])))
add("square", af ** 2)
add("sqrt_149", np.sqrt(np.array([1.0, 4.0, 9.0])))
add("pow_2", af ** 2)
add("pow_05", af ** 0.5)
add("exp_1234", np.exp(af))
add("log_1234", np.log(af))
add("log2_1234", np.log2(af))
add("log10_1234", np.log10(af))
add("clip_23", np.clip(af, 2.0, 3.0))
add("mod_7", np.mod(ai, 3))
add("rem_7", np.remainder(ai, 3))
add("floor_1234", np.floor(np.array([1.2, 2.5, 3.7, 4.1])))
add("ceil_1234", np.ceil(np.array([1.2, 2.5, 3.7, 4.1])))
add("round_1234", np.round(np.array([1.2, 2.5, 3.7, 4.1])))
add("rint_1234", np.rint(np.array([1.2, 2.5, 3.7, 4.1])))
add("reciprocal", 1.0 / af)

# broadcast arithmetic
add("broadcast_add", M34 + np.array([1.0, 2.0, 3.0, 4.0]))  # (3,4)+(4,)
add("broadcast_mul", M34 * np.array([1.0, 2.0, 3.0, 4.0]).reshape(1,4))

# type promotion
add("int32_float64", np.array([1,2,3], dtype=np.int32) + np.array([0.5, 0.5, 0.5]))

# ============================================================
# 5. 三角函数 (15+ tests)
# ============================================================
x_pi = np.linspace(0, np.pi/2, 4)
add("sin_0pi2", np.sin(x_pi))
add("cos_0pi2", np.cos(x_pi))
add("tan_0pi4", np.tan(np.linspace(0, np.pi/4, 3)))
add("asin_01", np.arcsin(np.array([0.0, 0.5, 1.0])))
add("acos_01", np.arccos(np.array([0.0, 0.5, 1.0])))
add("atan_012", np.arctan(np.array([0.0, 1.0, 2.0])))
add("atan2", np.arctan2(np.array([1.0, 1.0, 0.0]), np.array([0.0, 1.0, 1.0])))
add("sinh_123", np.sinh(np.array([1.0, 2.0, 3.0])))
add("cosh_123", np.cosh(np.array([1.0, 2.0, 3.0])))
add("tanh_123", np.tanh(np.array([1.0, 2.0, 3.0])))
add("hypot_34", np.hypot(np.array([3.0, 5.0]), np.array([4.0, 12.0])))
add("deg2rad", np.deg2rad(np.array([0.0, 90.0, 180.0])))
add("rad2deg", np.rad2deg(np.array([0.0, np.pi/2, np.pi])))

# ============================================================
# 6. 比较与逻辑 (15+ tests)
# ============================================================
a5 = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
b5 = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
add("lt", (a5 < b5).astype(float))
add("le", (a5 <= b5).astype(float))
add("gt", (a5 > b5).astype(float))
add("ge", (a5 >= b5).astype(float))
add("eq", (a5 == b5).astype(float))
add("ne", (a5 != b5).astype(float))
add("logical_and", np.logical_and(a5 > 2, b5 > 2).astype(float))
add("logical_or", np.logical_or(a5 > 4, b5 > 4).astype(float))
add("logical_not", np.logical_not(a5 > 3).astype(float))
add("all_true", int(np.all(np.array([1, 1, 1]))))
add("all_false", int(np.all(np.array([1, 0, 1]))))
add("any_true", int(np.any(np.array([0, 0, 1]))))
add("any_false", int(np.any(np.array([0, 0, 0]))))
add("isclose", np.isclose(np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0001, 3.1]), atol=0.01, rtol=0).astype(float))
add("allclose_t", int(np.allclose(np.array([1.0, 2.0]), np.array([1.0, 2.0]))))
add("allclose_f", int(np.allclose(np.array([1.0, 2.0]), np.array([1.0, 3.0]))))
add("isfinite", np.isfinite(np.array([1.0, np.nan, np.inf, -np.inf, 0.0])).astype(float))
add("isinf", np.isinf(np.array([1.0, np.nan, np.inf, -np.inf, 0.0])).astype(float))
add("isnan", np.isnan(np.array([1.0, np.nan, np.inf, 0.0])).astype(float))

# ============================================================
# 7. 归约与统计 (30+ tests)
# ============================================================
add("sum_all", float(np.sum(M34)))
add("sum_ax0", np.sum(M34, axis=0))
add("sum_ax1", np.sum(M34, axis=1))
add("sum_ax0_kd", np.sum(M34, axis=0, keepdims=True))
add("sum_ax1_kd", np.sum(M34, axis=1, keepdims=True))
add("mean_all", float(np.mean(M34)))
add("mean_ax0", np.mean(M34, axis=0))
add("mean_ax1", np.mean(M34, axis=1))
add("max_all", float(np.max(M34)))
add("max_ax0", np.max(M34, axis=0))
add("max_ax1", np.max(M34, axis=1))
add("min_all", float(np.min(M34)))
add("min_ax0", np.min(M34, axis=0))
add("min_ax1", np.min(M34, axis=1))
add("argmax_ax0", np.argmax(M34, axis=0))
add("argmax_ax1", np.argmax(M34, axis=1))
add("argmin_ax0", np.argmin(M34, axis=0))
add("argmin_ax1", np.argmin(M34, axis=1))
add("std_all", float(np.std(M34)))
add("std_ax0", np.std(M34, axis=0))
add("var_all", float(np.var(M34)))
add("var_ax0", np.var(M34, axis=0))
add("prod_all", float(np.prod(np.array([1.0, 2.0, 3.0, 4.0]))))
add("prod_ax0", np.prod(M34, axis=0))
add("cumsum_1234", np.cumsum(np.array([1,2,3,4], dtype=np.int64)))
add("cumsum_ax0", np.cumsum(np.array([[1,2],[3,4]], dtype=np.int64), axis=0))
add("cumsum_ax1", np.cumsum(np.array([[1,2],[3,4]], dtype=np.int64), axis=1))
add("cumprod_1234", np.cumprod(np.array([1,2,3,4], dtype=np.int64)))
add("median_odd", float(np.median(np.array([3.0, 1.0, 4.0, 1.0, 5.0]))))
add("median_even", float(np.median(np.array([1.0, 2.0, 3.0, 4.0]))))
add("median_ax1", np.median(M34, axis=1))
add("pct50", float(np.percentile(V5, 50)))
add("pct25", float(np.percentile(V5, 25)))
add("pct75", float(np.percentile(V5, 75)))
add("pct90", float(np.percentile(V5, 90)))
add("ptp", float(np.ptp(np.array([3.0, 1.0, 4.0, 1.0, 5.0]))))
add("ptp_ax0", np.ptp(M34, axis=0))
add("sort_1d", np.sort(np.array([3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0])))
add("sort_ax0", np.sort(M34, axis=0))
add("sort_ax1", np.sort(M34, axis=1))
add("argsort_1d", np.argsort(np.array([3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0])))
add("maximum", np.maximum(np.array([1.0, 5.0, 3.0]), np.array([4.0, 2.0, 6.0])))
add("minimum", np.minimum(np.array([1.0, 5.0, 3.0]), np.array([4.0, 2.0, 6.0])))
add("diff_1d", np.diff(np.array([1.0, 3.0, 6.0, 10.0, 15.0])))
add("diff_ax1", np.diff(M34, axis=1))
add("gradient_1d", np.gradient(np.array([1.0, 4.0, 9.0, 16.0, 25.0])))

# ============================================================
# 8. 线性代数 (20+ tests)
# ============================================================
add("matmul_2x2", A2 @ B2)
add("matmul_2x3_3x2", A23 @ A32)
add("matmul_3x3", A3 @ A3)
add("dot_1d", int(np.dot(np.array([1,2,3]), np.array([4,5,6]))))
add("dot_2d_1d", A2 @ np.array([1.0, 2.0]))  # (2,2)@(2,)
add("outer_3_2", np.outer(V3, np.array([4.0, 5.0])))
add("trace_3", float(np.trace(A3)))
add("trace_rect", float(np.trace(A23)))
add("norm_34", float(np.linalg.norm(np.array([3.0, 4.0]))))
add("norm_ax1", np.linalg.norm(M34, axis=1))
add("l1_norm", float(np.sum(np.abs(np.array([-1.0, 2.0, -3.0])))))
add("det_2x2", float(np.linalg.det(A2)))
add("det_3x3", float(np.linalg.det(A3)))
add("inv_2x2", np.linalg.inv(A2))
add("solve_2x2", np.linalg.solve(np.array([[2,1],[1,3]], dtype=np.float64), np.array([7,8], dtype=np.float64)))
add("solve_3x3", np.linalg.solve(np.array([[2,1,0],[1,3,1],[0,1,2]], dtype=np.float64), np.array([1,2,3], dtype=np.float64)))

# QR
Q, R = np.linalg.qr(A23)
add("qr_Q_shape", list(Q.shape))
add("qr_R_shape", list(R.shape))
add("qr_recon_err", float(np.max(np.abs(A23 - Q @ R))))

# SVD (diagonal matrix)
U, s, Vt = np.linalg.svd(np.diag(np.array([5.0, 3.0, 1.0])))
add("svd_s_diag3", s)

# Cholesky
L = np.linalg.cholesky(np.array([[4,2],[2,3]], dtype=np.float64))
add("chol_L", L)
add("chol_err", float(np.max(np.abs(np.array([[4,2],[2,3]], dtype=np.float64) - L @ L.T))))

# matrix rank
add("rank_full", int(np.linalg.matrix_rank(np.eye(3))))
add("rank_deficient", int(np.linalg.matrix_rank(np.array([[1,2],[2,4]], dtype=np.float64))))

# LU (scipy)
try:
    from scipy.linalg import lu as scipy_lu
    P, L_lu, U_lu = scipy_lu(A2)
    add("lu_L", L_lu)
    add("lu_U", U_lu)
except:
    pass

# ============================================================
# 9. einsum (10+ tests)
# ============================================================
add("einsum_dot", int(np.einsum("i,i->", np.array([1,2,3]), np.array([4,5,6]))))
add("einsum_matmul", np.einsum("ij,jk->ik", np.arange(6).reshape(2,3), np.arange(6).reshape(3,2)))
add("einsum_transpose", np.einsum("ij->ji", np.array([[1,2],[3,4]])))
add("einsum_diag", np.einsum("ii->i", np.arange(9).reshape(3,3)))
add("einsum_trace", int(np.einsum("ii->", np.arange(9).reshape(3,3))))
add("einsum_outer", np.einsum("i,j->ij", np.array([1,2,3]), np.array([4,5])))
add("einsum_batch_mm", np.einsum("ij,jk->ik", A2, B2))  # same as matmul

# ============================================================
# 10. 激活函数 (10+ tests)
# ============================================================
x5 = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
add("sigmoid", 1/(1+np.exp(-x5)))
add("relu", np.maximum(0, x5))
add("tanh", np.tanh(x5))

# softmax
logits = np.array([1.0, 2.0, 3.0])
e = np.exp(logits - np.max(logits))
add("softmax_123", e / e.sum())

# ============================================================
# 11. 集合操作 (5+ tests)
# ============================================================
add("unique_122333", np.unique(np.array([1,2,2,3,3,3])))
add("unique_with_return_index", np.unique(np.array([3,1,2,1,3]), return_index=True)[0])
add("intersect1d", np.intersect1d(np.array([1,2,3,4,5]), np.array([3,4,5,6,7])))
add("union1d", np.union1d(np.array([1,2,3,4,5]), np.array([3,4,5,6,7])))
add("setdiff1d", np.setdiff1d(np.array([1,2,3,4,5]), np.array([3,4,5,6,7])))
add("setxor1d", np.setxor1d(np.array([1,2,3,4,5]), np.array([3,4,5,6,7])))

# ============================================================
# 12. nan 处理 (10+ tests)
# ============================================================
nan = np.nan
a_nan = np.array([1.0, nan, 3.0, 4.0])
add("nanmean", float(np.nanmean(a_nan)))
add("nansum", float(np.nansum(a_nan)))
add("nanmax", float(np.nanmax(a_nan)))
add("nanmin", float(np.nanmin(a_nan)))
add("nanstd", float(np.nanstd(a_nan)))
add("nanvar", float(np.nanvar(a_nan)))

# ============================================================
# 13. 其他 (pad, kron, meshgrid, interp)
# ============================================================
add("kron_22", np.kron(np.array([[1,2],[3,4]]), np.array([[0,5],[6,7]])))
add("meshgrid_x_shape", list(np.meshgrid(np.arange(3), np.arange(4), sparse=True)[0].shape))
add("meshgrid_y_shape", list(np.meshgrid(np.arange(3), np.arange(4), sparse=True)[1].shape))

# ============================================================
# 14. 损失函数 (3+ tests)
# ============================================================
yt = np.array([1.0, 0.0, 0.0])
yp = np.array([0.7, 0.2, 0.1])
add("mse", float(np.mean((yt - yp)**2)))
add("cross_entropy", float(-np.sum(yt * np.log(np.clip(yp, 1e-7, 1-1e-7)))))

# ============================================================
# Output
# ============================================================
with open("test/all_expected.json", "w") as f:
    json.dump(RESULTS, f, indent=1, ensure_ascii=False)

# Stats
n_scalar = sum(1 for v in RESULTS.values() if v.get("t") in ("i", "f"))
n_array = sum(1 for v in RESULTS.values() if v.get("t") == "a")
n_list = sum(1 for v in RESULTS.values() if v.get("t") == "l")
print(f"Generated {len(RESULTS)} tests: {n_scalar} scalars, {n_array} arrays, {n_list} lists")
