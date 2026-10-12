#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
verify_manip_numpy.py
================================================================================
clvt 形状/视图/拼接/切片类 API 的跨语言互操作测试 —— NumPy 端校验器

读取 vt-manip-interop-test.lisp 用 `vt-save` 落盘的 .npy 文件，
在 NumPy 中独立重建同一用例的期望结果，逐项比对：
    - shape 是否一致
    - dtype 是否一致
    - 数值是否逐位（精确）一致

基准数据（与 Lisp 端完全一致）：
    BASE = np.arange(1000000, dtype=np.float64)      # (vt-arange 1000000)
    M    = BASE.reshape(1000, 1000)                  # (vt-reshape ... '(1000 1000))
    M3   = BASE.reshape(100, 100, 100)

用法：
    CLVT_TEST_OUT=/tmp/clvt-interop-out python3 test/verify_manip_numpy.py
退出码：0 = 全部通过；1 = 存在失败
================================================================================
"""
import json
import os
import sys

import numpy as np

OUT = os.environ.get("CLVT_TEST_OUT", "/tmp/clvt-interop-out")

# --- 基准数据 -------------------------------------------------------------
BASE = np.arange(1000000, dtype=np.float64)
M = BASE.reshape(1000, 1000)
M3 = BASE.reshape(100, 100, 100)

# ---------------------------------------------------------------------------
# 用例注册表：name -> callable() -> ndarray | list[ndarray]
# 每个用例语义与 test/vt-manip-interop-test.lisp 中的 `defcase` 一一对应。
# ---------------------------------------------------------------------------
CASES = {}


def case(name):
    def deco(fn):
        CASES[name] = fn
        return fn
    return deco


# === A 组：view / reshape ==================================================
@case("A01_view_same_shape")
def _():
    return M.reshape(1000, 1000)

@case("A02_view_flat")
def _():
    return M.reshape(1000000)

@case("A03_view_500x2000")
def _():
    return M.reshape(500, 2000)

@case("A04_view_100x100x100")
def _():
    return M.reshape(100, 100, 100)

@case("A05_view_2000x500")
def _():
    return M.reshape(2000, 500)

@case("A06_reshape_same")
def _():
    return np.reshape(M, (1000, 1000))

@case("A07_reshape_flat")
def _():
    return np.reshape(M, (1000000,))

@case("A08_reshape_4d")
def _():
    return np.reshape(M, (10, 10, 100, 100))

@case("A09_reshape_transposed_flat")
def _():
    return np.reshape(M.T, (1000000,))

@case("A10_reshape_transposed_500x2000")
def _():
    return np.reshape(M.T, (500, 2000))

@case("A11_reshape_narrowed")
def _():
    return np.reshape(M[100:700], (600, 1000))

@case("A12_reshape_strided_cols")
def _():
    return np.reshape(M[:, 0:1000:2], (1000, 500))


# === B 组：transpose / swapaxes / rot90 / flip / roll ======================
@case("B01_transpose_rev")
def _():
    return M.T

@case("B02_transpose_perm_201")
def _():
    return M3.transpose(2, 0, 1)

@case("B03_transpose_perm_120")
def _():
    return M3.transpose(1, 2, 0)

@case("B04_transpose_3d_rev")
def _():
    return M3.transpose(2, 1, 0)

@case("B05_transpose_of_narrow")
def _():
    return np.transpose(M[100:700])

@case("B06_swapaxes_0_1")
def _():
    return M.swapaxes(0, 1)

@case("B07_swapaxes_m3_0_2")
def _():
    return M3.swapaxes(0, 2)

@case("B08_rot90_k1")
def _():
    return np.rot90(M)

@case("B09_rot90_k2")
def _():
    return np.rot90(M, 2)

@case("B10_rot90_k3")
def _():
    return np.rot90(M, 3)

@case("B11_rot90_k-1")
def _():
    return np.rot90(M, -1)

@case("B12_rot90_3d_axes12")
def _():
    return np.rot90(M3, axes=(1, 2))

@case("B13_flip_ax0")
def _():
    return np.flip(M, 0)

@case("B14_flip_ax1")
def _():
    return np.flip(M, 1)

@case("B15_flip_all")
def _():
    return np.flip(M)

@case("B16_roll_ax0_5")
def _():
    return np.roll(M, 5, axis=0)

@case("B17_roll_ax1_-3")
def _():
    return np.roll(M, -3, axis=1)

@case("B18_roll_two_axes")
def _():
    return np.roll(M, (5, 7), axis=(0, 1))

@case("B19_roll_flat")
def _():
    return np.roll(M, 123)

@case("B20_roll_flat_list")
def _():
    return np.roll(M, 1000000 + 5)


# === C 组：squeeze / unsqueeze / expand-dims / flatten / ravel =============
@case("C01_squeeze_all")
def _():
    return np.squeeze(M.reshape(1, 1000, 1000, 1))

@case("C02_squeeze_axis0")
def _():
    return np.squeeze(M.reshape(1, 1000, 1000), axis=0)

@case("C03_squeeze_axis2")
def _():
    return np.squeeze(M.reshape(1000, 1000, 1), axis=2)

@case("C04_unsqueeze0")
def _():
    return np.expand_dims(M, 0)

@case("C05_unsqueeze1")
def _():
    return np.expand_dims(M, 1)

@case("C06_unsqueeze_neg1")
def _():
    return np.expand_dims(M, -1)

@case("C07_expand_dims0")
def _():
    return np.expand_dims(M, 0)

@case("C08_expand_dims2")
def _():
    return np.expand_dims(M, 2)

@case("C09_flatten")
def _():
    return M.flatten()

@case("C10_flatten_transposed")
def _():
    return M.T.flatten()

@case("C11_ravel")
def _():
    return np.ravel(M)

@case("C12_ravel_transposed")
def _():
    return np.ravel(M.T)

@case("C13_ravel_strided_slice")
def _():
    return np.ravel(M[0:1000:3])


# === D 组：narrow / slice ==================================================
@case("D01_narrow_rows")
def _():
    return M[100:900]

@case("D02_narrow_cols")
def _():
    return M[:, 0:500]

@case("D03_narrow_neg_start")
def _():
    return M[-100:1000]

@case("D04_narrow_3d_axis1")
def _():
    return M3[:, 10:90]

@case("D05_slice_rows")
def _():
    return M[100:900, :]

@case("D06_slice_step_rows")
def _():
    return M[0:1000:3, :]

@case("D07_slice_step_cols")
def _():
    return M[:, 0:1000:7]

@case("D08_slice_both_step")
def _():
    return M[100:200:5, 0:500:2]

@case("D09_slice_int_index")
def _():
    return M[500, :]

@case("D10_slice_neg_index")
def _():
    return M[-1, :]

@case("D11_slice_newa")
def _():
    return M[np.newaxis, :, :]

@case("D12_slice_elli")
def _():
    return M3[..., 0:50]

@case("D13_slice_neg_range")
def _():
    return M[-200:-100, :]

@case("D14_slice_reverse_cols")
def _():
    return M[:, ::-1]

@case("D15_slice_reverse_step3")
def _():
    return M[:, ::-3]


# === E 组：split / vsplit / hsplit / dsplit ================================
@case("E01_split4_ax0")
def _():
    return list(np.split(M, 4, axis=0))

@case("E02_split3_ax0")
def _():
    return list(np.array_split(M, 3, axis=0))

@case("E03_split_indices_ax0")
def _():
    return list(np.split(M, [250, 750], axis=0))

@case("E04_split5_ax1")
def _():
    return list(np.array_split(M, 5, axis=1))

@case("E05_split_m3_ax2")
def _():
    return list(np.array_split(M3, 4, axis=2))

@case("E06_vsplit4")
def _():
    return list(np.vsplit(M, 4))

@case("E07_vsplit_indices")
def _():
    return list(np.vsplit(M, [300, 700]))

@case("E08_hsplit2")
def _():
    return list(np.hsplit(M, 2))

@case("E09_hsplit_1d")
def _():
    return list(np.hsplit(BASE, 4))

@case("E10_dsplit2")
def _():
    return list(np.dsplit(M3, 2))

@case("E11_dsplit4")
def _():
    return list(np.dsplit(M3, 4))


# === F 组：stack / vstack / hstack / dstack / concatenate ==================
@case("F01_stack0")
def _():
    return np.stack([M[0:4], M[4:8]], axis=0)

@case("F02_stack1")
def _():
    return np.stack([M[0:4], M[4:8]], axis=1)

@case("F03_stack_neg1")
def _():
    return np.stack([M[0:4], M[4:8]], axis=-1)

@case("F04_stack_three")
def _():
    return np.stack([M[0:4], M[4:8], M[8:12]], axis=0)

@case("F05_vstack")
def _():
    return np.vstack([M[0:4], M[4:8]])

@case("F06_vstack_1d")
def _():
    return np.vstack([np.arange(1000, dtype=np.float64),
                      np.arange(1000, dtype=np.float64) + 1000])

@case("F07_hstack")
def _():
    return np.hstack([M[:, 0:4], M[:, 4:8]])

@case("F08_hstack_1d")
def _():
    return np.hstack([BASE[0:1000], BASE[1000:2000]])

@case("F09_dstack")
def _():
    return np.dstack([M[0:4], M[4:8]])

@case("F10_concatenate_ax0")
def _():
    return np.concatenate([M[0:4], M[4:8]], axis=0)

@case("F11_concatenate_ax1")
def _():
    return np.concatenate([M[:, 0:4], M[:, 4:8]], axis=1)

@case("F12_concat_neg_axis")
def _():
    return np.concatenate([M[:, 0:4], M[:, 4:8]], axis=-1)

@case("F13_concatenate_noncontig")
def _():
    return np.concatenate([M.T, M.T], axis=0)

@case("F14_concatenate_three")
def _():
    return np.concatenate([M[0:4], M[4:8], M[8:12]], axis=0)

@case("F15_concatenate_interleaved")
def _():
    return np.concatenate([M[0:1000:2], M[1:1000:2]], axis=1)


# === G 组：repeat / tile / pad / broadcast-to / contiguous =================
@case("G01_repeat_ax0")
def _():
    return np.repeat(M[0:4], 2, axis=0)

@case("G02_repeat_ax1")
def _():
    return np.repeat(M[0:4], 3, axis=1)

@case("G03_repeat_list")
def _():
    return np.repeat(M[0:4], [0, 1, 2, 3], axis=0)

@case("G04_repeat_flat")
def _():
    return np.repeat(M[0:4], 2)

@case("G05_tile_scalar")
def _():
    return np.tile(M[0:4], 2)

@case("G06_tile_2x3")
def _():
    return np.tile(M[0:4], (2, 3))

@case("G07_tile_3d_reps")
def _():
    return np.tile(M[0:4], (2, 1, 2))

@case("G08_pad_constant1")
def _():
    return np.pad(M[0:4], 1)

@case("G09_pad_perdim")
def _():
    return np.pad(M[0:4], ((2, 1), (3, 0)))

@case("G10_pad_edge")
def _():
    return np.pad(M[0:4], 3, mode="edge")

@case("G11_pad_wrap")
def _():
    return np.pad(M[0:4], 3, mode="wrap")

@case("G12_pad_reflect")
def _():
    return np.pad(M[0:4], 3, mode="reflect")

@case("G13_pad_symmetric")
def _():
    return np.pad(M[0:4], 3, mode="symmetric")

@case("G14_pad_constant_value")
def _():
    return np.pad(M[0:4], 2, constant_values=-1)

@case("G15_broadcast_col")
def _():
    return np.broadcast_to(M[:, 0:1].reshape(1000, 1), (1000, 1000))

@case("G16_broadcast_row")
def _():
    return np.broadcast_to(M[0:1], (1000, 1000))

@case("G17_broadcast_3d")
def _():
    return np.broadcast_to(M[:, 0:1].reshape(1, 1000, 1), (2, 1000, 1000))

@case("G18_broadcast_scalar")
def _():
    return np.broadcast_to(np.array([7.0]), (1000, 1000))

@case("G19_contiguous_transposed")
def _():
    return np.ascontiguousarray(M.T)

@case("G20_contiguous_strided")
def _():
    return np.ascontiguousarray(M[0:1000:2])

@case("G21_contiguous_already")
def _():
    return np.ascontiguousarray(M)

# 位移取模为 0 时结果值不变（拷贝语义由 contract-regression-test 断言）
@case("G22_roll_zero_axis0")
def _():
    return np.roll(M, 0, axis=0)

@case("G23_roll_axislen_multiple")
def _():
    return np.roll(M, 1000, axis=0)


# === H 组：triu / tril / diagonal / flatten-to-nested ======================
@case("H01_triu")
def _():
    return np.triu(M)

@case("H02_triu_k1")
def _():
    return np.triu(M, 1)

@case("H03_tril")
def _():
    return np.tril(M)

@case("H04_tril_k-1")
def _():
    return np.tril(M, -1)

@case("H05_diagonal")
def _():
    return np.diagonal(M).copy()

@case("H06_diagonal_off_pos")
def _():
    return np.diagonal(M, offset=5).copy()

@case("H07_diagonal_off_neg")
def _():
    return np.diagonal(M, offset=-5).copy()

@case("H08_diagonal_batch")
def _():
    # clvt 的 batch 对角线取「最后两轴」(同 torch.diagonal)；
    # numpy 的 np.diagonal(a) 默认 axis1=0/axis2=1（历史特例），需显式指定 -2/-1。
    return np.diagonal(M3, axis1=-2, axis2=-1).copy()

@case("H09_flatten_to_nested_2d")
def _():
    # vt-flatten-to-nested '(1000 1000) 再 vt-from-sequence 回建 => 原矩阵
    return M.reshape(1000, 1000)

@case("H10_flatten_to_nested_3d")
def _():
    return M3.reshape(100, 100, 100)

# batch 对角线显式轴（numpy 语义；默认仍取末两轴）
@case("H11_diagonal_axis1_0_axis2_1")
def _():
    return np.diagonal(M3, axis1=0, axis2=1).copy()

@case("H12_diagonal_axis1_0_axis2_2")
def _():
    return np.diagonal(M3, axis1=0, axis2=2).copy()

@case("H13_diagonal_axis1_1_axis2_2_off3")
def _():
    return np.diagonal(M3, offset=3, axis1=1, axis2=2).copy()


# === I 组：append / insert / delete ========================================
@case("I01_append_ax0")
def _():
    return np.append(M, M[0:3], axis=0)

@case("I02_append_ax1")
def _():
    return np.append(M, M[:, 0:3], axis=1)

@case("I03_append_flat")
def _():
    return np.append(M, np.array([1.0, 2.0, 3.0]), axis=None)

@case("I04_append_flat_views")
def _():
    return np.append(M[0:2], M[2:5], axis=None)

@case("I05_insert_ax0")
def _():
    return np.insert(M, 5, M[0:3], axis=0)

@case("I06_insert_ax1")
def _():
    return np.insert(M, 7, M[0:1], axis=1)

# I15：vt-insert axis=1 多行 values 施加 moveaxis(values, 0, axis)
@case("I15_insert_ax1_moveaxis")
def _():
    return np.insert(M, 7, M[0:2], axis=1)

@case("I07_insert_flat")
def _():
    return np.insert(M[0:10], 3, np.array([100.0, 200.0]), axis=None)

@case("I08_insert_multi_pos")
def _():
    return np.insert(M[0:6], [1, 3], M[0:2], axis=0)

@case("I09_delete_ax0")
def _():
    return np.delete(M, 5, axis=0)

@case("I10_delete_ax1")
def _():
    return np.delete(M, 5, axis=1)

@case("I11_delete_list")
def _():
    return np.delete(M, [1, 3, 5], axis=0)

@case("I12_delete_neg")
def _():
    return np.delete(M, -1, axis=0)

@case("I13_delete_flat")
def _():
    return np.delete(M, 0, axis=None)

@case("I14_delete_flat_list")
def _():
    return np.delete(M[0:2], [0, 1, 2], axis=None)


# === J 组：take/put/where/argwhere/nonzero/choose/select/extract/  =========
#            searchsorted/digitize/bincount/astype
@case("J01_take_ax0")
def _():
    return M[[0, 5, 999, 500], :]

@case("J02_take_ax1")
def _():
    return M[:, [0, 999]]

@case("J03_take_flat")
def _():
    return M.reshape(-1)[[0, 999999, 500000, 123]]

@case("J04_take_duplicates")
def _():
    return M[[3, 3, 3], :]

@case("J05_put_flat")
def _():
    c = M[0:3].copy()
    np.put(c, [0, 2999, 1500], [-1.0, -2.0, -3.0])
    return c

@case("J06_put_duplicate_indices")
def _():
    c = M[0:3].copy()
    np.put(c, [0, 0, 1], [10.0, 20.0, 30.0])
    return c

@case("J07_where_cond_only_gt")
def _():
    # np.where(cond) == np.asarray(cond).nonzero()，返回按轴分组的索引
    # 元组（长度 = rank），不是 np.argwhere 的 (n, rank)。
    return list(np.nonzero(M > 999000))

@case("J08_where_cond_only_eq")
def _():
    # 同上：单参 where 对齐 np.nonzero
    return list(np.nonzero(M == 123456))

@case("J09_where_three_arg")
def _():
    return np.where(M > 500000, M, np.flip(M, 1))

@case("J10_where_scalar_branch")
def _():
    return np.where(M < 5, M, -1.0)

@case("J11_argwhere_eq")
def _():
    return np.argwhere(M == 123456)

@case("J12_argwhere_gt")
def _():
    return np.argwhere(M > 999000)

@case("J13_argwhere_range")
def _():
    return np.argwhere((M >= 250000) & (M < 250010))

@case("J14_nonzero_eq")
def _():
    return list(np.nonzero(M == 123456))

@case("J15_nonzero_gt")
def _():
    return list(np.nonzero(M > 999000))

@case("J16_nonzero_3d")
def _():
    return list(np.nonzero(M3 == 500000))

@case("J17_choose")
def _():
    return np.choose((M > 500000).astype(np.int64), [M, np.flip(M, 1)])

@case("J18_choose_clip")
def _():
    idx = (M > 500000).astype(np.int64) - 1
    return np.choose(idx, [M, np.flip(M, 1)], mode="clip")

@case("J19_select")
def _():
    return np.select([M > 999000, M < 5], [M, np.flip(M, 1)], default=-1.0)

@case("J20_extract_eq")
def _():
    return np.extract(M == 123456, M)

@case("J21_extract_gt")
def _():
    return np.extract(M > 999000, M)

@case("J22_searchsorted_left")
def _():
    return np.searchsorted(BASE, np.array([0.0, 500000.0, 999999.0, 123456.0]),
                           side="left")

@case("J23_searchsorted_right")
def _():
    return np.searchsorted(BASE, np.array([0.0, 500000.0, 999999.0, 123456.0]),
                           side="right")

@case("J24_searchsorted_vector")
def _():
    # clvt: (vt-searchsorted (vt-arange 1000) (vt-narrow m 0 0 5))
    # 第二参为 M[0:5]（形状 (5,1000)），numpy.searchsorted 支持多维 values 并保持形状。
    return np.searchsorted(np.arange(1000, dtype=np.float64), M[0:5])

@case("J25_digitize")
def _():
    bins = np.arange(1001, dtype=np.float64) * 1000.0
    x = np.array([-1.0, 0.0, 500.0, 999.5, 1000.0, 1000000.0, 2000000.0])
    return np.digitize(x, bins)

@case("J26_digitize_right")
def _():
    bins = np.arange(1001, dtype=np.float64) * 1000.0
    x = np.array([-1.0, 0.0, 500.0, 999.5, 1000.0, 1000000.0, 2000000.0])
    return np.digitize(x, bins, right=True)

@case("J27_bincount")
def _():
    return np.bincount((np.arange(100000, dtype=np.int64) * 37) % 20)

@case("J28_bincount_minlength")
def _():
    return np.bincount(np.array([1, 3, 5]), minlength=10)

@case("J29_astype_int32")
def _():
    return M.astype(np.int32)

@case("J30_astype_float32")
def _():
    return M.astype(np.float32)

@case("J31_astype_int16")
def _():
    return M[0:2].astype(np.int16)

@case("J32_astype_noncontig")
def _():
    return M.T.astype(np.float32)

@case("J33_astype_int64")
def _():
    return M.astype(np.int64)


# === K 组：多算子组合 ======================================================
@case("K01_combo_trans_reshape_contig")
def _():
    return np.ascontiguousarray(np.reshape(M.T, (500, 2000)))

@case("K02_combo_slice_trans_flatten")
def _():
    return np.transpose(M[0:1000:2]).flatten()

@case("K03_combo_reshape_narrow_trans")
def _():
    return np.transpose(M.reshape(100, 100, 100)[:, 10:90])

@case("K04_combo_pad_flip_roll")
def _():
    return np.roll(np.flip(np.pad(M[0:4], 2), axis=1), 3, axis=1)

@case("K05_combo_stack_reshape")
def _():
    return np.reshape(np.stack([M[0:2], M[2:4]], axis=0), (4, 1000))

@case("K06_combo_split_vstack_roundtrip")
def _():
    return np.vstack(np.split(M[0:8], 4, axis=0))

@case("K07_combo_trans_swapaxes")
def _():
    return np.transpose(M3, (2, 0, 1)).swapaxes(0, 1)

@case("K08_combo_tile_transpose")
def _():
    return np.transpose(np.tile(M[0:4], (1, 2)))

@case("K09_combo_broadcast_concat")
def _():
    return np.concatenate([np.broadcast_to(M[:, 0:1].reshape(1000, 1), (1000, 500)),
                           M[:, 0:500]], axis=1)

@case("K10_combo_pad_flatten")
def _():
    return np.pad(M[0:4], 1).flatten()

@case("K11_combo_rot90_reshape_flat")
def _():
    return np.rot90(M).reshape(1000000)

@case("K12_combo_concat_repeat_narrow")
def _():
    return np.repeat(np.concatenate([M[0:2], M[998:1000]], axis=0), 2, axis=0)

@case("K13_combo_where_flatten")
def _():
    return np.where(M > 999000, M, 0.0).flatten()

@case("K14_combo_squeeze_unsqueeze_roundtrip")
def _():
    return np.squeeze(np.expand_dims(M, 1), axis=1)

@case("K15_combo_diagonal_take")
def _():
    return np.diagonal(M)[[0, 500, 999]]

@case("K16_combo_flip_both_axes")
def _():
    return np.flip(np.flip(M, 0), 1)

@case("K17_combo_strided_interleave_concat")
def _():
    return np.concatenate([M[0:1000:2], M[1:1000:2]], axis=0)

@case("K18_combo_pad_trans_contig")
def _():
    return np.ascontiguousarray(np.transpose(np.pad(M[:, 0:4], ((0, 0), (2, 3)))))

@case("K19_combo_4d_perm_reshape")
def _():
    return np.reshape(np.transpose(M.reshape(10, 100, 1000), (1, 2, 0)), (1000, 1000))

@case("K20_combo_insert_delete_roundtrip")
def _():
    return np.delete(np.insert(M[0:6], 2, M[0:1], axis=0), 2, axis=0)


# ---------------------------------------------------------------------------
# L 组：空索引（整数索引契约不得误伤「空索引」）
#   空索引没有任何元素会被截断，numpy 对空 *list* 同样放行：
#   np.take(a, []) → 空结果、a.put([], []) → no-op、np.insert(a, [], v)
#   → 原样返回。故这里以 numpy 的空 list 作为参照。
#   （clvt 对「空 float64 索引张量」也放行；numpy 在此因 float64→int64 的
#     casting 规则报错，属 casting 副作用，已在 CONTRACT-ALIGNMENT-REPORT
#     中记为刻意分歧。）
# ---------------------------------------------------------------------------
@case("L01_take_empty_index")
def _():
    return np.take(M, [])

@case("L02_take_empty_axis0")
def _():
    return np.take(M, [], axis=0)

@case("L03_take_empty_axis1")
def _():
    return np.take(M, [], axis=1)

@case("L04_put_empty_index_noop")
def _():
    out = M.copy()
    out.put([], [])
    return out

@case("L05_insert_empty_positions")
def _():
    return np.insert(M, [], M[0:1], axis=0)


# ---------------------------------------------------------------------------
# 已知差异（KNOWN DEVIATIONS）
# 这些用例的 clvt 行为与 numpy 契约不一致，已由 Lisp 端/NumPy 端双向确认。
# 它们被单独统计，不计入失败，但会在报告中显式列出，便于上游修复。
# ---------------------------------------------------------------------------
KNOWN_DEVIATIONS = {
    # 当前无未修复的已知差异。
    # （vt-insert 的 moveaxis 问题已修正，原先归类为「已知差异」的用例已改名
    #   I15_insert_ax1_moveaxis 并转入正常判定；详见
    #   test/contract-regression-test.lisp 与 src/join.lisp。）
}


# ---------------------------------------------------------------------------
# 比对逻辑
# ---------------------------------------------------------------------------
def as_list(r):
    return list(r) if isinstance(r, (list, tuple)) else [r]


def compare_one(name, idx, got, want):
    """返回 (ok, msg)。"""
    if got.shape != want.shape:
        return False, f"shape 不符: clvt={got.shape} numpy={want.shape}"
    if got.dtype != want.dtype:
        return False, f"dtype 不符: clvt={got.dtype} numpy={want.dtype}"
    if np.array_equal(got, want):
        return True, ""
    # 定位首个差异
    diff = np.argwhere(got != want)
    pos = tuple(diff[0]) if len(diff) else ("?",)
    return False, (f"数值不符: {len(diff)} 处不同, 首个位置 {pos} "
                   f"clvt={got[pos]} numpy={want[pos]}")


def main():
    manifest_path = os.path.join(OUT, "manifest.json")
    if not os.path.exists(manifest_path):
        print(f"[错误] 找不到 manifest: {manifest_path}")
        print("       请先运行 test/vt-manip-interop-test.lisp 生成数据。")
        return 1

    with open(manifest_path, encoding="utf-8") as f:
        man = json.load(f)

    print("=" * 78)
    print("clvt <-> NumPy 形状/视图/拼接/切片 互操作校验")
    print("=" * 78)
    print(f"NumPy {np.__version__}")
    print(f"manifest: {manifest_path}")
    print(f"基准: (vt-reshape (vt-arange 1000000) '(1000 1000))  ->  "
          f"shape={man.get('base_shape')} dtype={man.get('base_dtype')}")
    print(f"clvt 版本: {man.get('clvt_version')}   SBCL {man.get('sbcl')}")
    print(f"用例数: {man.get('case_count')}")
    print("-" * 78)

    results = []
    n_pass = n_fail = n_missing = n_dev = n_dev_fixed = 0

    for c in man["cases"]:
        name = c["name"]
        fn = CASES.get(name)
        if fn is None:
            n_missing += 1
            results.append({"name": name, "status": "NO_REF",
                            "detail": "验证器缺少该用例的 NumPy 参考实现"})
            print(f"[ ?    ] {name}  <-- 验证器未实现该用例")
            continue

        want = as_list(fn())
        outs = c["outputs"]
        if len(want) != len(outs):
            n_fail += 1
            results.append({"name": name, "status": "FAIL",
                            "detail": f"输出个数不符: clvt={len(outs)} numpy={len(want)}"})
            print(f"[FAIL  ] {name}  输出个数不符 clvt={len(outs)} numpy={len(want)}")
            continue

        ok_all, msgs = True, []
        for i, (meta, w) in enumerate(zip(outs, want)):
            path = os.path.join(OUT, meta["file"])
            if not os.path.exists(path):
                ok_all = False
                msgs.append(f"#{i} 文件缺失 {meta['file']}")
                continue
            got = np.load(path)
            ok, msg = compare_one(name, i, got, np.asarray(w))
            if not ok:
                ok_all = False
                msgs.append(f"#{i} {msg}")

        # 附加：manifest 声明的 shape/dtype 与 NumPy 期望是否一致
        for i, (meta, w) in enumerate(zip(outs, want)):
            w = np.asarray(w)
            if list(w.shape) != list(meta["shape"]):
                ok_all = False
                msgs.append(f"#{i} manifest shape={meta['shape']} != numpy {list(w.shape)}")
            if str(w.dtype) != meta["dtype"]:
                ok_all = False
                msgs.append(f"#{i} manifest dtype={meta['dtype']} != numpy {w.dtype}")

        is_dev = name in KNOWN_DEVIATIONS
        if ok_all and not is_dev:
            n_pass += 1
            results.append({"name": name, "status": "PASS",
                            "outputs": [o["file"] for o in outs],
                            "shape": outs[0]["shape"], "dtype": outs[0]["dtype"]})
            print(f"[PASS  ] {name}  shape={outs[0]['shape']} dtype={outs[0]['dtype']}"
                  + (f"  ({len(outs)} 输出)" if len(outs) > 1 else ""))
        elif ok_all and is_dev:
            n_dev_fixed += 1
            results.append({"name": name, "status": "DEV-FIXED",
                            "note": "此前记录的已知差异已消失（clvt 与 numpy 现已一致）",
                            "deviation": KNOWN_DEVIATIONS[name]})
            print(f"[FIXED ] {name}  已知差异已消失，clvt 与 numpy 现已一致")
        elif (not ok_all) and is_dev:
            n_dev += 1
            results.append({"name": name, "status": "KNOWN-DEV",
                            "detail": "; ".join(msgs),
                            "deviation": KNOWN_DEVIATIONS[name]})
            print(f"[DEVI  ] {name}  已知差异（不计失败）")
            print(f"         {KNOWN_DEVIATIONS[name]}")
            for m in msgs:
                print(f"         {m}")
        else:
            n_fail += 1
            results.append({"name": name, "status": "FAIL", "detail": "; ".join(msgs)})
            print(f"[FAIL  ] {name}")
            for m in msgs:
                print(f"         {m}")

    print("-" * 78)
    total = n_pass + n_fail + n_missing + n_dev + n_dev_fixed
    print(f"总计: {total}   通过: {n_pass}   失败: {n_fail}   "
          f"已知差异: {n_dev}   缺参考: {n_missing}")
    if n_dev_fixed:
        print(f"       (另有 {n_dev_fixed} 个已知差异用例现已修复)")
    verdict = "全部通过 ✅" if (n_fail == 0 and n_missing == 0) else "存在不一致 ❌"
    print(f"结论: {verdict}")
    print("=" * 78)

    report = {
        "numpy_version": np.__version__,
        "clvt_version": man.get("clvt_version"),
        "sbcl": man.get("sbcl"),
        "base_expression": man.get("base_expression"),
        "total": total, "pass": n_pass, "fail": n_fail, "no_ref": n_missing,
        "known_deviations": n_dev, "dev_fixed": n_dev_fixed,
        "verdict": verdict,
        "results": results,
    }
    rep_path = os.path.join(OUT, "numpy_verify_report.json")
    with open(rep_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"报告已写入: {rep_path}")

    return 0 if (n_fail == 0 and n_missing == 0) else 1


if __name__ == "__main__":
    sys.exit(main())
