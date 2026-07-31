#!/usr/bin/env python3
"""
gen_numpy_expected.py — 为 clvt 每个函数生成 numpy/pytorch 期望结果
输出: expected_results.json
用法: python3 gen_numpy_expected.py
"""
import numpy as np
import json, sys

RESULTS = {}  # results dict

def add(key, val):
    if isinstance(val, np.ndarray):
        RESULTS[key] = {"shape": list(val.shape), "dtype": str(val.dtype), "data": val.tolist()}
    elif isinstance(val, (float, int)):
        RESULTS[key] = {"scalar": val}
    elif hasattr(val, 'item'):
        RESULTS[key] = {"scalar": val.item()}
    elif isinstance(val, list):
        RESULTS[key] = {"data": val}
    else:
        try:
            RESULTS[key] = {"scalar": float(val)}
        except:
            RESULTS[key] = {"data": str(val)}

# ============================================================
# 1. 张量创建
# ============================================================
add("arange_10", np.arange(10))
add("arange_5_start2_step3", np.arange(5) * 3 + 2)
add("linspace_0_1_5", np.linspace(0, 1, 5))
add("linspace_0_1_5_noendpoint", np.linspace(0, 1, 5, endpoint=False))
add("logspace_0_3_4", np.logspace(0, 3, 4))
add("eye_3", np.eye(3))
add("eye_3x5_k1", np.eye(3, 5, k=1))
add("diag_123", np.diag([1,2,3]))
add("diag_extract_k1", np.diag(np.arange(9).reshape(3,3), k=1))
add("zeros_23", np.zeros((2,3)))
add("ones_22", np.ones((2,2)))
add("full_23_7", np.full((2,3), 7.0))

# ============================================================
# 2. 形状操作
# ============================================================
a = np.arange(6)
add("reshape_23", a.reshape(2,3))
add("reshape_neg1", a.reshape(2,-1))  # -1 inference
add("transpose_23", a.reshape(2,3).T)
add("transpose_perm", np.arange(24).reshape(2,3,4).transpose(1,0,2))
add("squeeze_123", np.squeeze(np.arange(6).reshape(1,2,3)))
add("squeeze_axis1", np.squeeze(np.arange(6).reshape(2,1,3), axis=1))
add("expand_dims_0", np.expand_dims(np.array([1,2,3]), 0))
add("expand_dims_neg1", np.expand_dims(np.array([1,2,3]), -1))

# concatenate / stack
a2 = np.array([[1,2],[3,4]])
b2 = np.array([[5,6],[7,8]])
add("concat_axis0", np.concatenate([a2, b2], axis=0))
add("concat_axis1", np.concatenate([a2, b2], axis=1))
add("stack_axis0", np.stack([np.array([1,2]), np.array([3,4])], axis=0))
add("stack_axis1", np.stack([np.array([1,2]), np.array([3,4])], axis=1))

# split
add("split_3_7", [x.tolist() for x in np.split(np.arange(10), [3,7])])

# flip / roll
add("flip_axis0", np.flip(np.arange(6).reshape(2,3), axis=0))
add("flip_axis1", np.flip(np.arange(6).reshape(2,3), axis=1))
add("roll_2", np.roll(np.arange(5), 2))
add("roll_neg1", np.roll(np.arange(5), -1))

# triu / tril / diagonal
m3 = np.arange(9).reshape(3,3)
add("triu", np.triu(m3))
add("triu_k1", np.triu(m3, k=1))
add("tril", np.tril(m3))
add("tril_k_neg1", np.tril(m3, k=-1))
add("diagonal", np.diagonal(m3))
add("diagonal_k1", np.diagonal(m3, offset=1))
add("diagonal_k_neg1", np.diagonal(m3, offset=-1))

# tile / repeat
add("tile_3", np.tile(np.array([1,2,3]), 3))
add("repeat_2", np.repeat(np.array([1,2,3]), 2))
add("tile_23", np.tile(np.array([[1,2],[3,4]]), (2,3)))
add("repeat_2_axis1", np.repeat(np.array([[1,2],[3,4]]), 2, axis=1))

# broadcast
add("broadcast_to_23", np.broadcast_to(np.array([1,2,3]), (2,3)))

# ============================================================
# 3. 索引与切片
# ============================================================
a10 = np.arange(10)
add("slice_2_7", a10[2:7])
add("slice_1_9_2", a10[1:9:2])
add("slice_reverse", a10[::-1])
add("slice_neg1", int(a10[-1]))
add("slice_neg3_neg1", a10[-3:-1])

b20 = np.arange(20).reshape(4,5)
add("2d_1_2", int(b20[1,2]))
add("2d_row2", b20[2,:])
add("2d_col3", b20[:,3])
add("2d_sub_13_24", b20[1:3, 2:4])
add("2d_neg_idx", b20[-2:, -3:])

# ============================================================
# 4. 算术
# ============================================================
ai = np.array([1,2,3,4], dtype=np.int64)
bi = np.array([5,6,7,8], dtype=np.int64)
add("add_ab", ai + bi)
add("sub_ba", bi - ai)
add("mul_ab", ai * bi)
add("div_ba_float", bi.astype(float) / ai.astype(float))  # float division
add("add_scalar10", ai + 10)
add("mul_scalar2", ai * 2)
add("int32_float64_promote", np.array([1,2,3], dtype=np.int32) + np.array([0.5,0.5,0.5], dtype=np.float64))

# ============================================================
# 5. 三角 / 指数 / 对数
# ============================================================
x = np.linspace(0, np.pi/2, 4)
add("sin_0_pi2", np.sin(x))
add("cos_0_pi2", np.cos(x))
add("exp_123", np.exp(np.array([1.0, 2.0, 3.0])))
add("log_123", np.log(np.array([1.0, 2.0, 3.0])))
add("sqrt_149_16", np.sqrt(np.array([1.0, 4.0, 9.0, 16.0])))

# ============================================================
# 6. 归约
# ============================================================
a12 = np.arange(12, dtype=np.float64).reshape(3,4)
add("sum_all", float(np.sum(a12)))
add("sum_axis0", np.sum(a12, axis=0))
add("sum_axis1", np.sum(a12, axis=1))
add("mean_all", float(np.mean(a12)))
add("mean_axis1", np.mean(a12, axis=1))
add("max_all", float(np.max(a12)))
add("min_axis0", np.min(a12, axis=0))
add("argmax_axis1", np.argmax(a12, axis=1))
add("argmin_axis0", np.argmin(a12, axis=0))
add("std_all", float(np.std(a12)))
add("var_axis0_ddof0", np.var(a12, axis=0))  # ddof=0 default
add("sum_axis0_keepdims", np.sum(a12, axis=0, keepdims=True))
add("mean_axis1_keepdims", np.mean(a12, axis=1, keepdims=True))

# cumsum / cumprod
add("cumsum_1234", np.cumsum(np.array([1,2,3,4], dtype=np.int64)))
add("cumprod_1234", np.cumprod(np.array([1,2,3,4], dtype=np.int64)))

# median / percentile
add("median_31415926", float(np.median(np.array([3.0,1,4,1,5,9,2,6]))))
add("pct50_12345", float(np.percentile(np.array([1.0,2,3,4,5]), 50)))
add("pct90_12345", float(np.percentile(np.array([1.0,2,3,4,5]), 90)))

# sort / argsort
v8 = np.array([3.0,1,4,1,5,9,2,6])
add("sort_8", np.sort(v8))
add("argsort_8", np.argsort(v8))

# ============================================================
# 7. 线性代数
# ============================================================
A2 = np.array([[1,2],[3,4]], dtype=np.float64)
B2 = np.array([[5,6],[7,8]], dtype=np.float64)
add("matmul_2x2", A2 @ B2)

A23 = np.array([[1,2,3],[4,5,6]], dtype=np.float64)
B32 = np.array([[7,8],[9,10],[11,12]], dtype=np.float64)
add("matmul_2x3_3x2", A23 @ B32)

add("trace_3x3", float(np.trace(np.arange(9).reshape(3,3).astype(float))))
add("det_2x2", float(np.linalg.det(A2)))
add("solve_2x2", np.linalg.solve(np.array([[2,1],[1,3]], dtype=np.float64), np.array([7,8], dtype=np.float64)))
add("inv_2x2", np.linalg.inv(np.array([[4,7],[2,6]], dtype=np.float64)))
add("norm_34", float(np.linalg.norm(np.array([3.0,4.0]))))

# QR
Q_r, R_mat = np.linalg.qr(np.array([[1,2],[3,4],[5,6]], dtype=np.float64))
add("qr_recon_err", float(np.max(np.abs(np.array([[1,2],[3,4],[5,6]], dtype=np.float64) - Q_r @ R_mat))))

# SVD (diagonal matrix for reliability)
U_svd, s_svd, Vt_svd = np.linalg.svd(np.array([[3,0],[0,2]], dtype=np.float64))
add("svd_s_diag", s_svd)

# Cholesky
L_mat = np.linalg.cholesky(np.array([[4,2],[2,3]], dtype=np.float64))
add("cholesky_L", L_mat)
add("cholesky_recon_err", float(np.max(np.abs(np.array([[4,2],[2,3]], dtype=np.float64) - L_mat @ L_mat.T))))

# eigenvalues (diagonal for reliability)
add("eig_diag_52", np.array([5.0, 2.0]))

# matrix rank
add("rank_12_24", int(np.linalg.matrix_rank(np.array([[1,2],[2,4]], dtype=np.float64))))

# ============================================================
# 8. einsum
# ============================================================
add("einsum_dot", int(np.einsum("i,i->", np.array([1,2,3]), np.array([4,5,6]))))
add("einsum_matmul", np.einsum("ij,jk->ik", np.arange(6).reshape(2,3), np.arange(6).reshape(3,2)))
add("einsum_transpose", np.einsum("ij->ji", np.array([[1,2],[3,4]])))
add("einsum_diag", np.einsum("ii->i", np.arange(9).reshape(3,3)))
add("einsum_trace", int(np.einsum("ii->", np.arange(9).reshape(3,3))))
add("einsum_outer", np.einsum("i,j->ij", np.array([1,2,3]), np.array([4,5])))

# ============================================================
# 9. 比较
# ============================================================
a5 = np.array([1.0,2,3,4,5])
b5 = np.array([5.0,4,3,2,1])
add("lt_ab", (a5 < b5).astype(float))
add("eq_ab", (a5 == b5).astype(float))

# ============================================================
# 10. 激活函数
# ============================================================
x5 = np.array([-2.0,-1,0,1,2])
add("sigmoid_x", 1/(1+np.exp(-x5)))
add("relu_x", np.maximum(0, x5))
add("tanh_x", np.tanh(x5))
logits = np.array([1.0,2,3])
e = np.exp(logits - np.max(logits))
add("softmax_123", e / e.sum())

# ============================================================
# 11. where / nonzero
# ============================================================
add("where_cond", np.where(np.array([1,0,1,0]), np.array([10.0,20,30,40]), np.array([100.0,200,300,400])))
add("nonzero_010203", np.nonzero(np.array([0,1,0,2,0,3]))[0])

# ============================================================
# 12. 集合
# ============================================================
add("unique_122333", np.unique([1,2,2,3,3,3]))
add("intersect1d", np.intersect1d(np.array([1,2,3,4,5]), np.array([3,4,5,6,7])))
add("union1d", np.union1d(np.array([1,2,3,4,5]), np.array([3,4,5,6,7])))
add("setdiff1d", np.setdiff1d(np.array([1,2,3,4,5]), np.array([3,4,5,6,7])))

# ============================================================
# 13. pad
# ============================================================
add("pad_constant_1", np.pad(np.array([[1,2],[3,4]]), 1, mode='constant', constant_values=0))
add("pad_edge_1", np.pad(np.array([[1,2],[3,4]]), 1, mode='edge'))

# ============================================================
# 14. diff / gradient
# ============================================================
add("diff_1361015", np.diff(np.array([1.0,3,6,10,15])))
add("gradient_1491625", np.gradient(np.array([1.0,4,9,16,25])))

# ============================================================
# 15. nan
# ============================================================
nan = np.nan
a_nan = np.array([1.0, nan, 3.0, 4.0])
add("nanmean_1nan34", float(np.nanmean(a_nan)))
add("nansum_1nan34", float(np.nansum(a_nan)))
add("nanmax_1nan34", float(np.nanmax(a_nan)))
add("nanmin_1nan34", float(np.nanmin(a_nan)))

# ============================================================
# 16. meshgrid / kron
# ============================================================
add("meshgrid_sparse_X_shape", [1, 3])
add("meshgrid_sparse_Y_shape", [4, 1])
add("kron_22_22", np.kron(np.array([[1,2],[3,4]]), np.array([[0,5],[6,7]])))

# ============================================================
# 17. outer
# ============================================================
add("outer_123_45", np.outer(np.array([1.0,2,3]), np.array([4.0,5])))

# ============================================================
# 18. MSE
# ============================================================
add("mse_100_721", float(np.mean((np.array([1.0,0,0]) - np.array([0.7,0.2,0.1]))**2)))

# ============================================================
# Output
# ============================================================
with open("test/expected_numpy.json", "w") as f:
    json.dump(RESULTS, f, indent=2, ensure_ascii=False)

print(f"Generated {len(RESULTS)} expected results -> test/expected_numpy.json")
