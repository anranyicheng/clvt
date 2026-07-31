#!/usr/bin/env python3
"""
gen_param_tests.py — 系统化参数测试生成器
每个函数测试多种参数组合：不同维度(1D/2D/3D)、不同axis、keepdims、dtype等
输出: test/param_expected.json
"""
import numpy as np
import json

R = {}

def add(key, val):
    if isinstance(val, np.ndarray):
        R[key] = {"t":"a","s":list(val.shape),"d":str(val.dtype),"v":val.tolist()}
    elif isinstance(val, (int, np.integer)):
        R[key] = {"t":"i","v":int(val)}
    elif isinstance(val, (float, np.floating)):
        R[key] = {"t":"f","v":float(val)}
    elif isinstance(val, list):
        R[key] = {"t":"l","v":val}
    elif val is None:
        R[key] = {"t":"n","v":None}
    else:
        try: R[key] = {"t":"f","v":float(val)}
        except: R[key] = {"t":"s","v":str(val)}

# ============================================================
# 测试数据定义
# ============================================================
# 1D
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
    add(f"{name}_2d_all", float(fn(data)))
    add(f"{name}_2d_ax0", fn(data, axis=0))
    add(f"{name}_2d_ax1", fn(data, axis=1))
    add(f"{name}_2d_ax_neg1", fn(data, axis=-1))
    add(f"{name}_2d_ax0_kd", fn(data, axis=0, keepdims=True))
    add(f"{name}_2d_ax1_kd", fn(data, axis=1, keepdims=True))
    # 3D
    add(f"{name}_3d_all", float(fn(T234)))
    add(f"{name}_3d_ax0", fn(T234, axis=0))
    add(f"{name}_3d_ax1", fn(T234, axis=1))
    add(f"{name}_3d_ax2", fn(T234, axis=2))
    add(f"{name}_3d_ax_neg1", fn(T234, axis=-1))
    add(f"{name}_3d_ax_neg2", fn(T234, axis=-2))
    add(f"{name}_3d_ax01", fn(T234, axis=(0,1)))
    add(f"{name}_3d_ax12", fn(T234, axis=(1,2)))
    add(f"{name}_3d_ax0_kd", fn(T234, axis=0, keepdims=True))
    add(f"{name}_3d_ax2_kd", fn(T234, axis=2, keepdims=True))

# argmax/argmin: 2D + 3D
for name, fn in [("argmax", np.argmax), ("argmin", np.argmin)]:
    add(f"{name}_2d_ax0", fn(M34, axis=0))
    add(f"{name}_2d_ax1", fn(M34, axis=1))
    add(f"{name}_2d_ax_neg1", fn(M34, axis=-1))
    add(f"{name}_3d_ax0", fn(T234, axis=0))
    add(f"{name}_3d_ax1", fn(T234, axis=1))
    add(f"{name}_3d_ax2", fn(T234, axis=2))
    add(f"{name}_3d_ax_neg1", fn(T234, axis=-1))

# cumsum/cumprod: 2D + 3D, per axis
for name, fn in [("cumsum", np.cumsum), ("cumprod", np.cumprod)]:
    add(f"{name}_2d_ax0", fn(M34i, axis=0))
    add(f"{name}_2d_ax1", fn(M34i, axis=1))
    add(f"{name}_3d_ax0", fn(T234i, axis=0))
    add(f"{name}_3d_ax1", fn(T234i, axis=1))
    add(f"{name}_3d_ax2", fn(T234i, axis=2))

# median: 2D + 3D
add("median_2d_ax0", np.median(M34, axis=0))
add("median_2d_ax1", np.median(M34, axis=1))
add("median_3d_ax0", np.median(T234, axis=0))
add("median_3d_ax1", np.median(T234, axis=1))
add("median_3d_ax2", np.median(T234, axis=2))

# percentile: 2D + 3D, multiple percentiles
for p in [25, 50, 75, 90]:
    add(f"pct{p}_2d_ax0", np.percentile(M34, p, axis=0))
    add(f"pct{p}_2d_ax1", np.percentile(M34, p, axis=1))
    add(f"pct{p}_3d_ax0", np.percentile(T234, p, axis=0))
    add(f"pct{p}_3d_ax2", np.percentile(T234, p, axis=2))

# ptp: 2D + 3D
add("ptp_2d_ax0", np.ptp(M34, axis=0))
add("ptp_2d_ax1", np.ptp(M34, axis=1))
add("ptp_3d_ax0", np.ptp(T234, axis=0))
add("ptp_3d_ax2", np.ptp(T234, axis=2))

# sort/argsort: 2D + 3D, per axis
add("sort_2d_ax0", np.sort(M34, axis=0))
add("sort_2d_ax1", np.sort(M34, axis=1))
add("sort_3d_ax0", np.sort(T234, axis=0))
add("sort_3d_ax1", np.sort(T234, axis=1))
add("sort_3d_ax2", np.sort(T234, axis=2))
add("argsort_2d_ax0", np.argsort(M34, axis=0))
add("argsort_2d_ax1", np.argsort(M34, axis=1))
add("argsort_3d_ax0", np.argsort(T234, axis=0))
add("argsort_3d_ax2", np.argsort(T234, axis=2))

# diff: 2D + 3D per axis
add("diff_2d_ax0", np.diff(M34, axis=0))
add("diff_2d_ax1", np.diff(M34, axis=1))
add("diff_3d_ax0", np.diff(T234, axis=0))
add("diff_3d_ax2", np.diff(T234, axis=2))

# gradient: 2D + 3D
add("grad_2d_ax0", np.gradient(M34, axis=0))
add("grad_2d_ax1", np.gradient(M34, axis=1))
add("grad_3d_ax0", np.gradient(T234, axis=0))
add("grad_3d_ax2", np.gradient(T234, axis=2))

# ============================================================
# 2. 形状操作 — 高维测试
# ============================================================
# transpose: 3D with different perms
add("trans_3d_012", T234.transpose(0,1,2))  # identity
add("trans_3d_021", T234.transpose(0,2,1))
add("trans_3d_102", T234.transpose(1,0,2))
add("trans_3d_210", T234.transpose(2,1,0))
add("trans_3d_201", T234.transpose(2,0,1))

# squeeze: 3D with various singleton dims
add("squeeze_3d_134", np.squeeze(np.arange(12).reshape(1,3,4)))
add("squeeze_3d_314", np.squeeze(np.arange(12).reshape(3,1,4)))
add("squeeze_3d_341", np.squeeze(np.arange(12).reshape(3,4,1)))
add("squeeze_3d_ax0", np.squeeze(np.arange(12).reshape(1,3,4), axis=0))
add("squeeze_3d_ax1", np.squeeze(np.arange(12).reshape(3,1,4), axis=1))

# expand_dims: 3D at different positions
add("expand_3d_ax0", np.expand_dims(T234, 0).shape)
add("expand_3d_ax1", np.expand_dims(T234, 1).shape)
add("expand_3d_ax2", np.expand_dims(T234, 2).shape)
add("expand_3d_ax3", np.expand_dims(T234, 3).shape)
add("expand_3d_ax_neg1", np.expand_dims(T234, -1).shape)

# concatenate: 3D
add("concat_3d_ax0", np.concatenate([T234, T234+100], axis=0))
add("concat_3d_ax1", np.concatenate([T234, T234[:,:,:2]], axis=2))

# stack: 3D
add("stack_3d_ax0", np.stack([T234, T234+100], axis=0))
add("stack_3d_ax1", np.stack([T234, T234+100], axis=1))
add("stack_3d_ax3", np.stack([T234, T234+100], axis=3))

# flip: 3D per axis
add("flip_3d_ax0", np.flip(T234, axis=0))
add("flip_3d_ax1", np.flip(T234, axis=1))
add("flip_3d_ax2", np.flip(T234, axis=2))
add("flip_3d_ax_neg1", np.flip(T234, axis=-1))

# roll: 3D per axis
add("roll_3d_ax0", np.roll(T234i, 1, axis=0))
add("roll_3d_ax1", np.roll(T234i, 1, axis=1))
add("roll_3d_ax2", np.roll(T234i, 1, axis=2))

# split: 3D
add("split_3d_ax0", [x.tolist() for x in np.split(T234, 2, axis=0)])
add("split_3d_ax2", [x.tolist() for x in np.split(T234, 2, axis=2)])

# tile: 3D
add("tile_3d_121", np.tile(T222, (1,2,1)))
add("tile_3d_212", np.tile(T222, (2,1,2)))

# repeat: 3D
add("repeat_3d_ax0", np.repeat(T222i, 2, axis=0))
add("repeat_3d_ax2", np.repeat(T222i, 2, axis=2))

# diagonal: 3D (batch)
add("diag_3d", np.diagonal(T234))  # shape (2, 4, 3) -> (2, 3) last two dims

# triu/tril: 3D (batch)
add("triu_3d", np.triu(T234))
add("tril_3d", np.tril(T234))

# swapaxes: 3D
add("swapaxes_3d_01", np.swapaxes(T234, 0, 1))
add("swapaxes_3d_02", np.swapaxes(T234, 0, 2))
add("swapaxes_3d_12", np.swapaxes(T234, 1, 2))

# narrow: 3D
add("narrow_3d_ax0", T234[0:1, :, :])
add("narrow_3d_ax1", T234[:, 1:3, :])
add("narrow_3d_ax2", T234[:, :, 1:3])

# ============================================================
# 3. 算术 — 高维广播
# ============================================================
# 3D + scalar
add("add_3d_scalar", T234 + 10.0)
add("mul_3d_scalar", T234 * 2.0)

# 3D + 1D broadcast
add("add_3d_1d", T234 + np.array([1.0, 2.0, 3.0, 4.0]))  # (2,3,4)+(4,)
add("mul_3d_1d", T234 * np.array([1.0, 2.0, 3.0]).reshape(1,3,1))  # (2,3,4)*(1,3,1)

# 3D + 3D elementwise
add("add_3d_3d", T234 + T234)
add("mul_3d_3d", T234 * T234)

# abs/signum/square/sqrt on 3D
add("abs_3d", np.abs(T234 - 12))
add("square_3d", T234 ** 2)
add("sqrt_3d", np.sqrt(np.abs(T234)))
add("exp_3d", np.exp(T234 / 10))  # scale to avoid overflow
add("log_3d", np.log(np.abs(T234) + 1))

# clip on 3D
add("clip_3d", np.clip(T234, 5.0, 15.0))

# ============================================================
# 4. 三角函数 — 3D
# ============================================================
add("sin_3d", np.sin(T234 * 0.1))
add("cos_3d", np.cos(T234 * 0.1))
add("tanh_3d", np.tanh((T234 - 12) * 0.1))

# ============================================================
# 5. 比较逻辑 — 3D
# ============================================================
add("lt_3d", (T234 < 12.0).astype(float))
add("gt_3d", (T234 > 12.0).astype(float))
add("eq_3d", (T234 == 12.0).astype(float))
add("isclose_3d", np.isclose(T234, T234 + 0.001, atol=0.01, rtol=0).astype(float))
add("isfinite_3d", np.isfinite(T234).astype(float))

# all/any on 3D per axis
add("all_3d_ax0", np.all(T234 > 0, axis=0).astype(int))
add("all_3d_ax2", np.all(T234 > 0, axis=2).astype(int))
add("any_3d_ax0", np.any(T234 > 10, axis=0).astype(int))
add("any_3d_ax2", np.any(T234 > 10, axis=2).astype(int))

# ============================================================
# 6. 线性代数 — 更多场景
# ============================================================
# matmul: various shapes
A44 = np.arange(16, dtype=np.float64).reshape(4,4)
B44 = np.ones((4,4), dtype=np.float64)
add("matmul_4x4", A44 @ B44)
add("matmul_1x3_3x1", np.array([[1,2,3]], dtype=np.float64) @ np.array([[4],[5],[6]], dtype=np.float64))

# dot: 2D @ 2D
add("dot_2d_2d", np.dot(A2 := np.array([[1,2],[3,4]], dtype=np.float64), B2 := np.array([[5,6],[7,8]], dtype=np.float64)))

# trace: 3D (batch trace)
add("trace_3d", np.trace(T234))  # sum along last two dims diagonal

# norm: 3D per axis
add("norm_3d_ax0", np.linalg.norm(T234, axis=0))
add("norm_3d_ax1", np.linalg.norm(T234, axis=1))
add("norm_3d_ax2", np.linalg.norm(T234, axis=2))

# det: batch of 2x2
batch_2x2 = np.array([[[1,2],[3,4]],[[5,6],[7,8]]], dtype=np.float64)
add("det_batch", np.array([np.linalg.det(batch_2x2[i]) for i in range(2)]))

# ============================================================
# 7. einsum — 更多模式
# ============================================================
add("einsum_trace_2d", int(np.einsum("ii->", np.arange(4).reshape(2,2))))
add("einsum_matmul_T", np.einsum("ij,jk->ik", np.array([[1,2],[3,4]]), np.array([[5,6],[7,8]])))
add("einsum_3d_reduce", np.einsum("ijk->i", T234))  # reduce last two dims
add("einsum_3d_reduce_ax2", np.einsum("ijk->ij", T234))  # reduce last dim
add("einsum_bilinear", np.einsum("ij,jk,kl->il", A2, B2, A2))  # chain multiply

# ============================================================
# 8. where/nonzero — 3D
# ============================================================
add("where_3d", np.where(T234 > 12, T234, 0.0))
add("nonzero_3d", [x.tolist() for x in np.nonzero(T234 > 20)])

# ============================================================
# 9. pad — 3D, different modes
# ============================================================
add("pad_3d_const", np.pad(T222, 1, mode='constant', constant_values=0))
add("pad_3d_edge", np.pad(T222, 1, mode='edge'))

# ============================================================
# 10. set operations — different sizes
# ============================================================
add("unique_large", np.unique(np.array([5,3,1,4,2,5,3,1])))
add("unique_sorted", np.unique(np.array([3,1,2])))
add("intersect_empty", np.intersect1d(np.array([1,2,3]), np.array([4,5,6])))
add("union_same", np.union1d(np.array([1,2,3]), np.array([1,2,3])))

# ============================================================
# 11. nan — 3D
# ============================================================
T_nan = T234.copy()
T_nan[0, 0, 0] = np.nan
T_nan[1, 2, 3] = np.nan
add("nanmean_3d_ax0", np.nanmean(T_nan, axis=0))
add("nanmean_3d_ax2", np.nanmean(T_nan, axis=2))
add("nansum_3d_ax1", np.nansum(T_nan, axis=1))
add("nanmax_3d_ax0", np.nanmax(T_nan, axis=0))
add("nanmin_3d_ax2", np.nanmin(T_nan, axis=2))
add("nanstd_3d_ax0", np.nanstd(T_nan, axis=0))
add("nanvar_3d_ax2", np.nanvar(T_nan, axis=2))

# ============================================================
# 12. 激活函数 — 3D
# ============================================================
add("sigmoid_3d", 1/(1+np.exp(-T234)))
add("relu_3d", np.maximum(0, T234 - 12))
add("tanh_3d_act", np.tanh(T234 - 12))
add("softmax_3d_ax2", np.apply_along_axis(lambda x: np.exp(x - np.max(x)) / np.sum(np.exp(x - np.max(x))), 2, T234))

# ============================================================
# 13. loss — 3D
# ============================================================
yt_3d = np.zeros((2,3,4))
yt_3d[0,0,0] = 1.0
yt_3d[1,1,1] = 1.0
yp_3d = np.full((2,3,4), 0.25)
add("mse_3d", float(np.mean((yt_3d - yp_3d)**2)))

# ============================================================
# 14. 比较运算 — 3D
# ============================================================
add("lt_3d_pair", (T234 < (T234 + 0.5)).astype(float))
add("le_3d_pair", (T234 <= T234).astype(float))
add("gt_3d_pair", ((T234 + 1) > T234).astype(float))
add("maximum_3d", np.maximum(T234, T234[::-1]))
add("minimum_3d", np.minimum(T234, T234[::-1]))

# ============================================================
# 15. broadcast_to — 3D
# ============================================================
add("broadcast_1x3x4_to_2x3x4", np.broadcast_to(np.arange(12).reshape(1,3,4), (2,3,4)))
add("broadcast_2x1x4_to_2x3x4", np.broadcast_to(np.arange(8).reshape(2,1,4), (2,3,4)))

# ============================================================
# 16. 截断/取整 — 3D
# ============================================================
add("floor_3d", np.floor(T234 * 0.7))
add("ceil_3d", np.ceil(T234 * 0.7))
add("round_3d", np.round(T234 * 0.7))
add("mod_3d", np.mod(T234i, 7))
add("clip_3d_neg", np.clip(T234, -5.0, 5.0))

# ============================================================
# 17. 类型转换 — 不同 dtype
# ============================================================
for dt_name, dt in [("f32", np.float32), ("f64", np.float64), ("i32", np.int32), ("i64", np.int64)]:
    add(f"astype_{dt_name}", np.arange(6, dtype=dt).reshape(2,3))

# ============================================================
# Output
# ============================================================
with open("test/param_expected.json", "w") as f:
    json.dump(R, f, indent=1, ensure_ascii=False)

n_s = sum(1 for v in R.values() if v.get("t") in ("i","f"))
n_a = sum(1 for v in R.values() if v.get("t") == "a")
n_l = sum(1 for v in R.values() if v.get("t") == "l")
print(f"Generated {len(R)} parametric tests: {n_s} scalars, {n_a} arrays, {n_l} lists")
