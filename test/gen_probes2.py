#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
clvt vs numpy 差分模糊测试生成器 + 比对器 —— 第 2 阶段
覆盖 package.lisp 导出符号第 200 个（vt-nanmax）到第 377 个（vt-isin），
前 199 个已由 gen_probes.py / gen_report.py 完成测试。

约定基准：CONVENTIONS.md（凡与 NumPy 冲突一律以 NumPy 为准，唯一例外 vt-arange）。

  gen     : 生成 test/differential-probes-test2.lisp + <TMP>/expected2.txt + <TMP>/cases2.json
  compare : 读 <TMP>/actual2.txt（clvt 实际输出）与 expected2 比对，输出不匹配报告

随机数族（vt-random*、SeedSequence、Generator、with-seed/with-generator）与
LAPACK 分解族（lu/qr/svd/eig 符号歧义）无法与 numpy 逐值对标，
采用「可复现性 / 值域 / 形状 / 结构恒等式」属性探针（布尔），numpy 侧恒为 True。
"""
import sys, os, json, math, re, warnings
import numpy as np
warnings.filterwarnings('ignore')
np.seterr(all='ignore')

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TMP = os.environ.get('FUZZ_TMP') or os.path.join(os.path.dirname(_SCRIPT_DIR), 'tmp')
PROBE_OUT = os.path.join(_SCRIPT_DIR, 'differential-probes-test2.lisp')

# 复用第 1 阶段的基础设施：数据生成 / 字面量 / OP / probe 前导码
import gen_probes as gp
from gen_probes import CLD, gen, fmtnum, lit, OP

import ast as _ast
def _extract_prelude(path):
    # AST 提取 prelude 字面量的运行时值（正确处理反斜杠转义）
    tree = _ast.parse(open(path, encoding='utf-8').read())
    for node in _ast.walk(tree):
        if isinstance(node, _ast.Assign):
            for t in node.targets:
                if isinstance(t, _ast.Name) and t.id == 'prelude':
                    return _ast.literal_eval(node.value)
    raise RuntimeError('prelude not found')
PRELUDE = _extract_prelude(gp.__file__)

CASES = []  # (cid, desc, lisp_form, np_thunk, policy, expect)  expect: None|'err'
def add(desc, lisp, npfn, policy='exact', expect=None):
    cid = f"p{len(CASES):04d}"
    CASES.append((cid, desc, lisp, npfn, policy, expect))
    return cid

def rng(seed):
    return np.random.RandomState(seed)

# ---------------- 工具 ----------------
def B(cond):
    """布尔 → 0/1 整数数组（LIST 探针里禁止 T/NIL token）"""
    return np.asarray(1 if cond else 0)

# =====================================================================
# G1  NaN 统计（200 vt-nanmax, 201 vt-nanmin, 202 nanargmax, 203 nanargmin,
#              204 nanprod, 205 nanmedian, 360 nancumsum, 361 nancumprod,
#              362 nanpercentile, 363 nanquantile）
# =====================================================================
def g1():
    D = np.array([[1.0, np.nan, 3.0], [4.0, 5.0, np.nan]])
    L = "(list (list 1.0d0 +nan+ 3.0d0) (list 4.0d0 5.0d0 +nan+))"
    lt = f"(vt-from-sequence {L} :dtype :float64)"
    add('nanmax 2d 全归约', f"(vt-nanmax {lt})", lambda: np.nanmax(D), 'tol')
    add('nanmax axis0', f"(vt-nanmax {lt} :axis 0)", lambda: np.nanmax(D, axis=0), 'tol')
    add('nanmax axis1', f"(vt-nanmax {lt} :axis 1)", lambda: np.nanmax(D, axis=1), 'tol')
    add('nanmax axis0 keepdims', f"(vt-nanmax {lt} :axis 0 :keepdims t)",
        lambda: np.nanmax(D, axis=0, keepdims=True), 'tol')
    F32 = D.astype(np.float32)
    lf32 = f"(vt-from-sequence {lit(F32, 'float32')} :dtype :float32)"
    add('nanmax float32 axis0', f"(vt-nanmax {lf32} :axis 0)",
        lambda: np.nanmax(F32, axis=0), 'tol')
    A = np.full((2, 2), np.nan)
    lA = "(vt-from-sequence (list (list +nan+ +nan+) (list +nan+ +nan+)) :dtype :float64)"
    add('nanmax 全NaN → NaN', f"(vt-nanmax {lA})", lambda: np.nanmax(A), 'tol')
    I = np.array([[1, 2], [3, 4]], dtype=np.int64)
    lI = "(vt-from-sequence (list (list 1 2) (list 3 4)) :dtype :int64)"
    add('nanmax int64', f"(vt-nanmax {lI})", lambda: np.nanmax(I))
    add('nanmin 2d', f"(vt-nanmin {lt})", lambda: np.nanmin(D), 'tol')
    add('nanmin axis1', f"(vt-nanmin {lt} :axis 1)", lambda: np.nanmin(D, axis=1), 'tol')
    add('nanmin 全NaN', f"(vt-nanmin {lA})", lambda: np.nanmin(A), 'tol')
    add('nanargmax 2d', f"(vt-nanargmax {lt})", lambda: np.nanargmax(D))
    add('nanargmax axis0', f"(vt-nanargmax {lt} :axis 0)", lambda: np.nanargmax(D, axis=0))
    add('nanargmax axis1', f"(vt-nanargmax {lt} :axis 1)", lambda: np.nanargmax(D, axis=1))
    add('nanargmin 2d', f"(vt-nanargmin {lt})", lambda: np.nanargmin(D))
    add('nanargmin axis1', f"(vt-nanargmin {lt} :axis 1)", lambda: np.nanargmin(D, axis=1))
    V = np.array([np.nan, 2.0, 3.0])
    lV = "(vt-from-sequence (list +nan+ 2.0d0 3.0d0) :dtype :float64)"
    add('nanprod [nan 2 3]', f"(vt-nanprod {lV})", lambda: np.nanprod(V), 'tol')
    add('nanprod 2d axis0', f"(vt-nanprod {lt} :axis 0)", lambda: np.nanprod(D, axis=0), 'tol')
    add('nanprod float32', f"(vt-nanprod {lf32})", lambda: np.nanprod(F32), 'tol')
    add('nanmedian 奇数个', f"(vt-nanmedian {lV})", lambda: np.nanmedian(V), 'tol')
    E = np.array([1.0, 2.0, np.nan, 4.0])
    lE = "(vt-from-sequence (list 1.0d0 2.0d0 +nan+ 4.0d0) :dtype :float64)"
    add('nanmedian 偶数个', f"(vt-nanmedian {lE})", lambda: np.nanmedian(E), 'tol')
    add('nanmedian 2d axis0', f"(vt-nanmedian {lt} :axis 0)", lambda: np.nanmedian(D, axis=0), 'tol')
    add('nanmedian 全NaN', f"(vt-nanmedian {lA})", lambda: np.nanmedian(A), 'tol')
    C = np.array([np.nan, 1.0, 2.0, np.nan, 4.0])
    lC = "(vt-from-sequence (list +nan+ 1.0d0 2.0d0 +nan+ 4.0d0) :dtype :float64)"
    add('nancumsum 跳过NaN', f"(vt-nancumsum {lC})", lambda: np.nancumsum(C), 'tol')
    add('nancumsum 2d axis1', f"(vt-nancumsum {lt} :axis 1)", lambda: np.nancumsum(D, axis=1), 'tol')
    CI = np.array([1, 2, 3], dtype=np.int64)
    add('nancumsum int64', f"(vt-nancumsum {lI} :axis 0)", lambda: np.nancumsum(I, axis=0))
    P = np.array([np.nan, 2.0, 3.0])
    add('nancumprod 跳过NaN', f"(vt-nancumprod {lV})", lambda: np.nancumprod(P), 'tol')
    add('nancumprod 2d axis0', f"(vt-nancumprod {lt} :axis 0)", lambda: np.nancumprod(D, axis=0), 'tol')
    Q1 = np.array([1.0, np.nan, 3.0, 5.0, 7.0])
    lQ = "(vt-from-sequence (list 1.0d0 +nan+ 3.0d0 5.0d0 7.0d0) :dtype :float64)"
    add('nanpercentile 50', f"(vt-nanpercentile {lQ} 50.0d0)", lambda: np.nanpercentile(Q1, 50), 'tol')
    add('nanpercentile 2d axis0', f"(vt-nanpercentile {lt} 50.0d0 :axis 0)",
        lambda: np.nanpercentile(D, 50, axis=0), 'tol')
    add('nanpercentile 插值lower', f"(vt-nanpercentile {lQ} 40.0d0 :interpolation :lower)",
        lambda: np.nanpercentile(Q1, 40, method='lower'), 'tol')
    add('nanquantile 0.25', f"(vt-nanquantile {lQ} 0.25d0)", lambda: np.nanquantile(Q1, 0.25), 'tol')
    add('nanquantile 插值nearest', f"(vt-nanquantile {lQ} 0.9d0 :interpolation :nearest)",
        lambda: np.nanquantile(Q1, 0.9, method='nearest'), 'tol')
    add('nanpercentile 越界报错', f"(vt-nanpercentile {lQ} 101.0d0)", None, expect='err')
    add('nanquantile 越界报错', f"(vt-nanquantile {lQ} 1.5d0)", None, expect='err')

# =====================================================================
# G2  矩阵乘 / 点积（206 matmul, 207 @, 209 dot, 210 outer,
#                     304 inner, 305 tensordot）
# =====================================================================
def g2():
    A = rng(61).rand(3, 4).round(3)
    Bm = rng(62).rand(4, 5).round(3)
    la = f"(vt-from-sequence {lit(A, 'float64')} :dtype :float64)"
    lb = f"(vt-from-sequence {lit(Bm, 'float64')} :dtype :float64)"
    add('matmul 2d@2d f64', f"(vt-matmul {la} {lb})", lambda: A @ Bm, 'tol')
    Ai = rng(63).randint(-5, 5, (3, 4)).astype(np.int64)
    Bi = rng(64).randint(-5, 5, (4, 2)).astype(np.int64)
    lai = f"(vt-from-sequence {lit(Ai, 'int64')} :dtype :int64)"
    lbi = f"(vt-from-sequence {lit(Bi, 'int64')} :dtype :int64)"
    add('matmul 2d@2d int64', f"(vt-matmul {lai} {lbi})", lambda: Ai @ Bi)
    A32 = A.astype(np.float32); B32 = Bm.astype(np.float32)
    la32 = f"(vt-from-sequence {lit(A32, 'float32')} :dtype :float32)"
    lb32 = f"(vt-from-sequence {lit(B32, 'float32')} :dtype :float32)"
    add('matmul 2d@2d f32', f"(vt-matmul {la32} {lb32})",
        lambda: A32 @ B32, 'tol')
    v = rng(65).rand(4).round(3)
    lv = f"(vt-from-sequence {lit(v, 'float64')} :dtype :float64)"
    w = rng(66).rand(4).round(3)
    lw = f"(vt-from-sequence {lit(w, 'float64')} :dtype :float64)"
    add('matmul 1d@1d → 0维', f"(vt-matmul {lv} {lw})", lambda: np.asarray(v @ w), 'tol')
    add('matmul 1d@2d', f"(vt-matmul {lv} {lb})", lambda: v @ Bm, 'tol')
    add('matmul 2d@1d', f"(vt-matmul {la} {lv})", lambda: A @ v, 'tol')
    A3 = rng(67).rand(2, 3, 4).round(3)
    B3 = rng(68).rand(2, 4, 5).round(3)
    la3 = f"(vt-from-sequence {lit(A3, 'float64')} :dtype :float64)"
    lb3 = f"(vt-from-sequence {lit(B3, 'float64')} :dtype :float64)"
    add('matmul 批量 3d@3d', f"(vt-matmul {la3} {lb3})", lambda: A3 @ B3, 'tol')
    B3b = rng(69).rand(1, 4, 5).round(3)
    lb3b = f"(vt-from-sequence {lit(B3b, 'float64')} :dtype :float64)"
    add('matmul 广播 3d@1xk', f"(vt-matmul {la3} {lb3b})", lambda: A3 @ B3b, 'tol')
    At = f"(vt-transpose {la})"  # (4,3) 非连续视图
    Bt = rng(70).rand(3, 2).round(3)
    lbt = f"(vt-from-sequence {lit(Bt, 'float64')} :dtype :float64)"
    add('matmul 转置视图', f"(vt-matmul {At} {lbt})", lambda: A.T @ Bt, 'tol')
    Bbad = rng(71).rand(3, 5).round(3)
    lbbad = f"(vt-from-sequence {lit(Bbad, 'float64')} :dtype :float64)"
    add('matmul 形状不匹配报错', f"(vt-matmul {la} {lbbad})", None, expect='err')
    add('@ 别名 2d@2d', f"(vt-@ {la} {lb})", lambda: A @ Bm, 'tol')
    add('@ 别名 1d@2d', f"(vt-@ {lv} {lb})", lambda: v @ Bm, 'tol')
    add('dot 1d·1d → 0维', f"(vt-dot {lv} {lw})", lambda: np.asarray(np.dot(v, w)), 'tol')
    add('dot 2d@1d', f"(vt-dot {la} {lv})", lambda: np.dot(A, v), 'tol')
    add('dot 1d@2d', f"(vt-dot {lv} {lb})", lambda: np.dot(v, Bm), 'tol')
    add('dot 2d@2d', f"(vt-dot {la} {lb})", lambda: np.dot(A, Bm), 'tol')
    add('dot 批量 3d@3d', f"(vt-dot {la3} {lb3})", lambda: np.matmul(A3, B3), 'tol')
    add('outer 1d×1d', f"(vt-outer {lv} {lw})", lambda: np.outer(v, w), 'tol')
    a2 = rng(72).rand(2, 3).round(3)
    b2 = rng(73).rand(3, 2).round(3)
    la2 = f"(vt-from-sequence {lit(a2, 'float64')} :dtype :float64)"
    lb2 = f"(vt-from-sequence {lit(b2, 'float64')} :dtype :float64)"
    add('outer 2d×2d 展平', f"(vt-outer {la2} {lb2})", lambda: np.outer(a2, b2), 'tol')
    a3 = rng(74).rand(2, 2, 2).round(3)
    la3s = f"(vt-from-sequence {lit(a3, 'float64')} :dtype :float64)"
    add('outer 3d×1d 展平', f"(vt-outer {la3s} {lw})", lambda: np.outer(a3, w), 'tol')
    add('outer flatten=nil', f"(vt-outer {la2} {lw} :flatten nil)",
        lambda: (a2.reshape(2, 3, 1) * w), 'tol')
    add('inner 1d·1d', f"(vt-inner {lv} {lw})", lambda: np.asarray(np.inner(v, w)), 'tol')
    add('inner 2d×2d', f"(vt-inner {la2} {lb2})", lambda: np.inner(a2, b2), 'tol')
    T1 = rng(75).rand(4, 3, 2).round(3)
    T2 = rng(76).rand(3, 2, 5).round(3)
    lt1 = f"(vt-from-sequence {lit(T1, 'float64')} :dtype :float64)"
    lt2 = f"(vt-from-sequence {lit(T2, 'float64')} :dtype :float64)"
    add('tensordot axes=2', f"(vt-tensordot {lt1} {lt2} :axes 2)",
        lambda: np.tensordot(T1, T2, axes=2), 'tol')
    add('tensordot axes=(1,0)', f"(vt-tensordot {la} {lb} :axes (list (list 1) (list 0)))",
        lambda: np.tensordot(A, Bm, axes=([1], [0])), 'tol')
    add('tensordot axes=(0,1)', f"(vt-tensordot {la} {lb} :axes (list (list 0) (list 1)))",
        lambda: np.tensordot(A, Bm, axes=([0], [1])), 'tol')

# =====================================================================
# G3  einsum（208）+ 解析缓存（301）
# =====================================================================
def g3():
    Ei = np.arange(6).reshape(2, 3)
    Ej = np.arange(6, 12).reshape(3, 2)
    lei = f"(vt-from-sequence {lit(Ei, 'int64')} :dtype :int64)"
    lej = f"(vt-from-sequence {lit(Ej, 'int64')} :dtype :int64)"
    Ef = Ei.astype(np.float64)
    lf = f"(vt-from-sequence {lit(Ef, 'float64')} :dtype :float64)"
    add('einsum ij,jk->ik int', f'(vt-einsum "ij,jk->ik" {lei} {lej})', lambda: np.einsum('ij,jk->ik', Ei, Ej))
    Ejf = Ej.astype(np.float64)
    lejf = f"(vt-from-sequence {lit(Ejf, 'float64')} :dtype :float64)"
    add('einsum ij,jk->ki', f'(vt-einsum "ij,jk->ki" {lf} {lejf})',
        lambda: np.einsum('ij,jk->ki', Ef, Ej.astype(np.float64)), 'tol')
    v6 = np.array([1.0, 2.0, 3.0])
    lv6 = "(vt-from-sequence (list 1.0d0 2.0d0 3.0d0) :dtype :float64)"
    w6 = np.array([4.0, 5.0, 6.0])
    lw6 = "(vt-from-sequence (list 4.0d0 5.0d0 6.0d0) :dtype :float64)"
    add('einsum i,i-> 点积', f'(vt-einsum "i,i->" {lv6} {lw6})',
        lambda: np.asarray(np.einsum('i,i->', v6, w6)), 'tol')
    add('einsum i-> 求和', f'(vt-einsum "i->" {lv6})', lambda: np.asarray(np.einsum('i->', v6)), 'tol')
    add('einsum ij->ji 转置', f'(vt-einsum "ij->ji" {lf})', lambda: np.einsum('ij->ji', Ef), 'tol')
    M = np.array([[1.0, 2.0], [3.0, 4.0]])
    lM = f"(vt-from-sequence {lit(M, 'float64')} :dtype :float64)"
    add('einsum ii->i 对角线', f'(vt-einsum "ii->i" {lM})', lambda: np.einsum('ii->i', M), 'tol')
    add('einsum ii-> trace', f'(vt-einsum "ii->" {lM})', lambda: np.asarray(np.einsum('ii->', M)), 'tol')
    A5 = np.arange(6.0).reshape(2, 3)
    B5 = np.arange(6.0, 12.0).reshape(3, 2)
    C5 = np.arange(12.0, 16.0).reshape(2, 2)
    la5 = f"(vt-from-sequence {lit(A5, 'float64')} :dtype :float64)"
    lb5 = f"(vt-from-sequence {lit(B5, 'float64')} :dtype :float64)"
    lc5 = f"(vt-from-sequence {lit(C5, 'float64')} :dtype :float64)"
    add('einsum 三矩阵链', f'(vt-einsum "ij,jk,kl->il" {la5} {lb5} {lc5})',
        lambda: np.einsum('ij,jk,kl->il', A5, B5, C5), 'tol')
    Ab = np.arange(8.0).reshape(2, 2, 2)
    Bb = np.arange(8.0, 16.0).reshape(2, 2, 2)
    lab = f"(vt-from-sequence {lit(Ab, 'float64')} :dtype :float64)"
    lbb = f"(vt-from-sequence {lit(Bb, 'float64')} :dtype :float64)"
    add('einsum bij,bjk->bik', f'(vt-einsum "bij,bjk->bik" {lab} {lbb})',
        lambda: np.einsum('bij,bjk->bik', Ab, Bb), 'tol')
    add('einsum 省略号', f'(vt-einsum "...ij,...jk->...ik" {lab} {lbb})',
        lambda: np.einsum('...ij,...jk->...ik', Ab, Bb), 'tol')
    add('einsum 秩不符报错', f'(vt-einsum "ijk->" {lf})', None, expect='err')
    add('einsum 独占下标求和', f'(vt-einsum "ij,kl->ij" {lf} {lejf})',
        lambda: np.einsum('ij,kl->ij', Ef, Ej.astype(np.float64)), 'tol')
    add('einsum 解析缓存生效',
        f'(progn (vt-einsum "ij,jk->ik" {lei} {lej}) (if *vt-einsum-parse-cache* 1 0))',
        lambda: B(True))

# =====================================================================
# G4  trace / norm（211-214）
# =====================================================================
def g4():
    Mf = np.array([[1.0, 2.0], [3.0, 4.0]])
    lMf = "(vt-from-sequence (list (list 1.0d0 2.0d0) (list 3.0d0 4.0d0)) :dtype :float64)"
    add('trace 2d f64', f"(vt-trace {lMf})", lambda: np.asarray(np.trace(Mf)))
    Mi = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.int64)
    lMi = "(vt-from-sequence (list (list 1 2 3) (list 4 5 6) (list 7 8 9)) :dtype :int64)"
    add('trace 3x3 int64', f"(vt-trace {lMi})", lambda: np.asarray(np.trace(Mi)))
    v = np.array([3.0, 4.0])
    lv = "(vt-from-sequence (list 3.0d0 4.0d0) :dtype :float64)"
    add('norm 1d L2', f"(vt-norm {lv})", lambda: np.asarray(np.linalg.norm(v)), 'tol')
    add('norm 1d keepdims', f"(vt-norm {lv} :keepdims t)",
        lambda: np.linalg.norm(v, keepdims=True), 'tol')
    M = rng(81).rand(3, 4).round(3)
    lM = f"(vt-from-sequence {lit(M, 'float64')} :dtype :float64)"
    add('norm 2d 全展平 L2', f"(vt-norm {lM})", lambda: np.asarray(np.linalg.norm(M)), 'tol')
    add('norm axis1', f"(vt-norm {lM} :axis 1)", lambda: np.linalg.norm(M, axis=1), 'tol')
    add('norm axis0 keepdims', f"(vt-norm {lM} :axis 0 :keepdims t)",
        lambda: np.linalg.norm(M, axis=0, keepdims=True), 'tol')
    add('norm int → float64', f"(vt-norm (vt-from-sequence (list 3 4) :dtype :int64))",
        lambda: np.linalg.norm(np.array([3, 4])), 'tol')
    add('l1-norm 1d', f"(vt-l1-norm {lv})", lambda: np.asarray(np.abs(v).sum()), 'tol')
    add('l1-norm axis0', f"(vt-l1-norm {lM} :axis 0)", lambda: np.abs(M).sum(axis=0), 'tol')
    add('frobenius-norm 2d', f"(vt-frobenius-norm {lM})",
        lambda: np.asarray(np.linalg.norm(M, 'fro')), 'tol')
    add('frobenius-norm axis0', f"(vt-frobenius-norm {lM} :axis 0 :keepdims t)",
        lambda: np.linalg.norm(M, axis=0, keepdims=True), 'tol')

# =====================================================================
# G5  solve / inv / det（215-217）
# =====================================================================
def g5():
    S = np.array([[3.0, 1.0, 1.0], [1.0, 2.0, 0.0], [1.0, 0.0, 1.0]])
    lS = f"(vt-from-sequence {lit(S, 'float64')} :dtype :float64)"
    b1 = np.array([1.0, 2.0, 3.0])
    lb1 = f"(vt-from-sequence {lit(b1, 'float64')} :dtype :float64)"
    add('solve 3x3 b=1d', f"(vt-solve {lS} {lb1})", lambda: np.linalg.solve(S, b1), 'tol')
    b2 = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    lb2 = f"(vt-from-sequence {lit(b2, 'float64')} :dtype :float64)"
    add('solve 3x3 b=2d', f"(vt-solve {lS} {lb2})", lambda: np.linalg.solve(S, b2), 'tol')
    Si = np.array([[2, 1], [1, 3]])
    bi = np.array([5, 10], dtype=np.int64)
    lSi = "(vt-from-sequence (list (list 2 1) (list 1 3)) :dtype :int64)"
    lbi = "(vt-from-sequence (list 5 10) :dtype :int64)"
    add('solve int → float64', f"(vt-solve {lSi} {lbi})",
        lambda: np.linalg.solve(Si.astype(np.float64), bi.astype(np.float64)), 'tol')
    Sb = rng(91).rand(3, 2).round(3)
    lSb = f"(vt-from-sequence {lit(Sb, 'float64')} :dtype :float64)"
    add('solve 非方阵报错', f"(vt-solve {lSb} {lb1})", None, expect='err')
    I2 = np.array([[1.0, 2.0], [3.0, 4.0]])
    lI2 = f"(vt-from-sequence {lit(I2, 'float64')} :dtype :float64)"
    add('inv 2x2', f"(vt-inv {lI2})", lambda: np.linalg.inv(I2), 'tol')
    I3 = rng(92).rand(3, 3).round(3) + 3 * np.eye(3)
    lI3 = f"(vt-from-sequence {lit(I3, 'float64')} :dtype :float64)"
    add('inv 3x3', f"(vt-inv {lI3})", lambda: np.linalg.inv(I3), 'tol')
    IS = np.array([[1.0, 2.0], [2.0, 4.0]])
    lIS = "(vt-from-sequence (list (list 1.0d0 2.0d0) (list 2.0d0 4.0d0)) :dtype :float64)"
    add('inv 奇异报错', f"(vt-inv {lIS})", None, expect='err')
    D2 = np.array([[1.0, 2.0], [3.0, 4.0]])
    lD2 = f"(vt-from-sequence {lit(D2, 'float64')} :dtype :float64)"
    add('det 2x2', f"(vt-det {lD2})", lambda: np.asarray(np.linalg.det(D2)), 'tol')
    D3 = rng(93).rand(3, 3).round(3)
    lD3 = f"(vt-from-sequence {lit(D3, 'float64')} :dtype :float64)"
    add('det 3x3', f"(vt-det {lD3})", lambda: np.asarray(np.linalg.det(D3)), 'tol')
    DS = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.int64)
    lDS = "(vt-from-sequence (list (list 1 2 3) (list 4 5 6) (list 7 8 9)) :dtype :int64)"
    add('det 奇异 → 0', f"(vt-det {lDS})", lambda: np.asarray(np.linalg.det(DS.astype(np.float64))), 'tol')
    add('det 1x1', f"(vt-det (vt-from-sequence (list (list 5.0d0)) :dtype :float64))",
        lambda: np.asarray(np.linalg.det(np.array([[5.0]]))), 'tol')

# =====================================================================
# G6  分解（218 lu, 219 qr, 220 svd, 221 matrix-rank, 222 cholesky,
#             223 eig, 224 pinv, 225 lstsq, 368 eigvals, 369 eigvalsh,
#             370 matrix-power, 371 cond, 372 multi-dot, 367 vdot,
#             366 cross, 364 cov, 365 corrcoef）
# LAPACK/雅可比族存在符号/顺序歧义 → 结构恒等式 + 形状 + 特征值对标。
# =====================================================================
def g6():
    # ---- lu：L@U = P@A 结构恒等式（clvt 返回 (values lu piv sign)）----
    Lu = rng(101).rand(3, 3).round(3)
    lLu = f"(vt-from-sequence {lit(Lu, 'float64')} :dtype :float64)"
    add('lu 分解 L@U=P@A',
        f"""(multiple-value-bind (lu piv sign)
                (vt-lu {lLu})
              (let* ((n 3)
                     (perm (coerce piv 'vector))
                     (pa (make-array '(3 3) :initial-element 0.0d0))
                     (a-data (coerce (vt-to-list (vt-flatten {lLu})) 'vector)))
                (dotimes (i n)
                  (dotimes (j n)
                    (setf (aref pa i j) (aref a-data (+ (* (aref perm i) n) j)))))
                (let ((l (make-array '(3 3) :initial-element 0.0d0))
                      (u (make-array '(3 3) :initial-element 0.0d0))
                      (lu-data (coerce (vt-to-list (vt-flatten lu)) 'vector))
                      (diag 1.0d0))
                  (dotimes (i n)
                    (setf (aref l i i) 1.0d0)
                    (dotimes (j n)
                      (if (< j i)
                          (setf (aref l i j) (aref lu-data (+ (* i n) j)))
                          (setf (aref u i j) (aref lu-data (+ (* i n) j)))))
                    (setf diag (* diag (aref u i i))))
                  (let ((err 0.0d0))
                    (dotimes (i n)
                      (dotimes (j n)
                        (let ((s 0.0d0))
                          (dotimes (k n) (incf s (* (aref l i k) (aref u k j))))
                          (setf err (max err (abs (- s (aref pa i j))))))))
                    (and (< err 1e-9)
                         (< (abs (- (* (coerce sign 'double-float) diag)
                                    (vt-item (vt-det {lLu}))))
                            1e-9))))))""",
        lambda: B(True))
    add('lu 返回形状', f"(multiple-value-bind (lu piv sign) (vt-lu {lLu}) (list (vt-shape lu)))",
        lambda: np.array([3, 3]))
    # ---- qr ----
    R = rng(102).rand(4, 3).round(3)
    lR = f"(vt-from-sequence {lit(R, 'float64')} :dtype :float64)"
    q_r = np.linalg.qr(R, mode='reduced')
    add('qr reduced Q@R=A', f"(multiple-value-bind (q r) (vt-qr {lR}) (vt-matmul q r))",
        lambda: R, 'tol')
    add('qr reduced 正交性',
        f"""(multiple-value-bind (q r) (vt-qr {lR})
              (let ((m (vt-transpose q)))
                (let ((e (vt-- (vt-matmul m q) (vt-eye 3 :dtype :float64))))
                  (< (vt-item (vt-norm (vt-flatten e))) 1e-9))))""",
        lambda: B(True))
    add('qr reduced 形状',
        f"(multiple-value-bind (q r) (vt-qr {lR}) (append (vt-shape q) (vt-shape r)))",
        lambda: np.array([q_r[0].shape[0], q_r[0].shape[1], q_r[1].shape[0], q_r[1].shape[1]]))
    q_f = np.linalg.qr(R, mode='complete')
    add('qr full Q@R=A', f"(multiple-value-bind (q r) (vt-qr {lR} :mode :full) (vt-matmul q r))",
        lambda: R, 'tol')
    add('qr full 形状',
        f"(multiple-value-bind (q r) (vt-qr {lR} :mode :full) (append (vt-shape q) (vt-shape r)))",
        lambda: np.array([q_f[0].shape[0], q_f[0].shape[1], q_f[1].shape[0], q_f[1].shape[1]]))
    # ---- svd ----
    S = rng(103).rand(4, 3).round(3)
    lS = f"(vt-from-sequence {lit(S, 'float64')} :dtype :float64)"
    u_s_vh = np.linalg.svd(S, full_matrices=False)
    add('svd 奇异值降序', f"(multiple-value-bind (u s v) (vt-svd {lS}) s)",
        lambda: u_s_vh[1], 'tol')
    add('svd thin U diag(s) V = A',
        f"""(multiple-value-bind (u s v) (vt-svd {lS})
              (let ((diag (make-array '(3 3) :initial-element 0.0d0)))
                (dotimes (i 3) (setf (aref diag i i) (vt-item (vt-ref s i))))
                (let ((s-mat (vt-from-array diag :dtype :float64)))
                  (vt-- (vt-matmul u (vt-matmul s-mat v)) {lS}))))""",
        lambda: np.zeros((4, 3)), 'tol')
    add('svd thin 形状',
        f"(multiple-value-bind (u s v) (vt-svd {lS}) (append (vt-shape u) (vt-shape s) (vt-shape v)))",
        lambda: np.array([u_s_vh[0].shape[0], u_s_vh[0].shape[1],
                          u_s_vh[1].shape[0], u_s_vh[2].shape[0], u_s_vh[2].shape[1]]))
    add('svd 正交性(U列)',
        f"""(multiple-value-bind (u s v) (vt-svd {lS})
              (< (vt-item (vt-norm (vt-flatten
                     (vt-- (vt-matmul (vt-transpose u) u)
                           (vt-eye 3 :dtype :float64)))))
                 1e-9))""",
        lambda: B(True))
    add('svd full 形状',
        f"(multiple-value-bind (u s v) (vt-svd {lS} :full-matrices t) (append (vt-shape u) (vt-shape s) (vt-shape v)))",
        lambda: np.array([4, 4, 3, 3, 3]))
    add('svd 方阵', f"(multiple-value-bind (u s v) (vt-svd {lS3x3}) s)",
        lambda: np.linalg.svd(S3x3, full_matrices=False)[1], 'tol')
    # ---- matrix-rank ----
    FR = rng(104).rand(3, 3).round(3) + np.eye(3)
    lFR = f"(vt-from-sequence {lit(FR, 'float64')} :dtype :float64)"
    add('matrix-rank 满秩', f"(vt-matrix-rank {lFR})", lambda: np.asarray(np.linalg.matrix_rank(FR)))
    RD = np.array([[1.0, 2.0, 3.0], [2.0, 4.0, 6.0], [1.0, 1.0, 1.0]])
    lRD = "(vt-from-sequence (list (list 1.0d0 2.0d0 3.0d0) (list 2.0d0 4.0d0 6.0d0) (list 1.0d0 1.0d0 1.0d0)) :dtype :float64)"
    add('matrix-rank 缺秩', f"(vt-matrix-rank {lRD})", lambda: np.asarray(np.linalg.matrix_rank(RD)))
    add('matrix-rank 显式tol', f"(vt-matrix-rank {lRD} 1e-10)",
        lambda: np.asarray(np.linalg.matrix_rank(RD, tol=1e-10)))
    # ---- cholesky ----
    C = np.array([[4.0, 2.0], [2.0, 3.0]])
    lC = "(vt-from-sequence (list (list 4.0d0 2.0d0) (list 2.0d0 3.0d0)) :dtype :float64)"
    add('cholesky 下三角', f"(vt-cholesky {lC})", lambda: np.linalg.cholesky(C), 'tol')
    add('cholesky 上三角', f"(vt-cholesky {lC} :upper t)",
        lambda: np.linalg.cholesky(C).T, 'tol')
    CNP = rng(105).rand(3, 3) + 3 * np.eye(3)
    CNP = CNP @ CNP.T
    lCNP = f"(vt-from-sequence {lit(CNP, 'float64')} :dtype :float64)"
    add('cholesky 3x3 SPD', f"(vt-cholesky {lCNP})", lambda: np.linalg.cholesky(CNP), 'tol')
    add('cholesky 非正定报错',
        "(vt-cholesky (vt-from-sequence (list (list 1.0d0 2.0d0) (list 2.0d0 1.0d0)) :dtype :float64))",
        None, expect='err')
    # ---- eig / eigvals / eigvalsh（对称矩阵）----
    Sy = np.array([[2.0, 1.0], [1.0, 3.0]])
    lSy = "(vt-from-sequence (list (list 2.0d0 1.0d0) (list 1.0d0 3.0d0)) :dtype :float64)"
    add('eig 特征值升序', f"(multiple-value-bind (vals vec) (vt-eig {lSy}) (vt-to-list (vt-sort vals)))",
        lambda: np.sort(np.linalg.eigvalsh(Sy)), 'tol')
    add('eig 残差 A·v=λv',
        f"""(multiple-value-bind (vals vec) (vt-eig {lSy})
              (let ((d (make-array '(2 2) :initial-element 0.0d0)))
                (dotimes (i 2) (setf (aref d i i) (vt-item (vt-ref vals i))))
                (< (vt-item (vt-norm (vt-flatten
                       (vt-- (vt-matmul {lSy} vec)
                             (vt-matmul vec (vt-from-array d :dtype :float64))))))
                   1e-8)))""",
        lambda: B(True))
    add('eigvalsh 升序', f"(vt-eigvalsh {lSy})", lambda: np.linalg.eigvalsh(Sy), 'tol')
    Sy3 = rng(106).rand(3, 3); Sy3 = (Sy3 + Sy3.T) / 2
    lSy3 = f"(vt-from-sequence {lit(Sy3, 'float64')} :dtype :float64)"
    add('eigvalsh 3x3', f"(vt-eigvalsh {lSy3})", lambda: np.linalg.eigvalsh(Sy3), 'tol')
    add('eigvals 对称矩阵', f"(vt-to-list (vt-sort (vt-eigvals {lSy3})))",
        lambda: np.sort(np.linalg.eigvalsh(Sy3)), 'tol')
    add('eigvals 形状', f"(vt-shape (vt-eigvals {lSy3}))", lambda: np.array([3]))
    # ---- pinv ----
    P = rng(107).rand(3, 4).round(3)
    lP = f"(vt-from-sequence {lit(P, 'float64')} :dtype :float64)"
    add('pinv 3x4 数值', f"(vt-pinv {lP})", lambda: np.linalg.pinv(P), 'tol')
    add('pinv 恒等式 A·pinv·A=A',
        f"""(let* ((a {lP}) (ap (vt-pinv a)))
          (< (vt-item (vt-norm (vt-flatten
                 (vt-- (vt-matmul (vt-matmul a ap) a) a)))) 1e-9))""",
        lambda: B(True))
    add('pinv 恒等式 pinv·A·pinv=pinv',
        f"""(let* ((a {lP}) (ap (vt-pinv a)))
          (< (vt-item (vt-norm (vt-flatten
                 (vt-- (vt-matmul (vt-matmul ap a) ap) ap)))) 1e-8))""",
        lambda: B(True))
    # ---- lstsq ----
    LS_A = rng(108).rand(5, 3).round(3)
    LS_b = rng(109).rand(5).round(3)
    lLS_A = f"(vt-from-sequence {lit(LS_A, 'float64')} :dtype :float64)"
    lLS_b = f"(vt-from-sequence {lit(LS_b, 'float64')} :dtype :float64)"
    npx, _, nprank, nps = np.linalg.lstsq(LS_A, LS_b, rcond=None)
    add('lstsq 解 残差', f"(multiple-value-bind (x res rank s) (vt-lstsq {lLS_A} {lLS_b}) (< (vt-item (vt-norm (vt-matmul (vt-transpose {lLS_A}) (vt-- (vt-matmul {lLS_A} (vt-flatten x)) {lLS_b})))) 1e-9))",
        lambda: B(True))
    add('lstsq 秩', f"(multiple-value-bind (x res rank s) (vt-lstsq {lLS_A} {lLS_b}) rank)",
        lambda: np.asarray(nprank))
    add('lstsq 奇异值', f"(multiple-value-bind (x res rank s) (vt-lstsq {lLS_A} {lLS_b}) s)",
        lambda: nps, 'tol')
    # ---- vdot ----
    add('vdot 实向量', f"(vt-vdot {lv6 if False else lVd1} {lVd2})",
        lambda: np.asarray(np.vdot(Vd1, Vd2)), 'tol')
    # ---- cross ----
    X1 = np.array([1.0, 2.0, 3.0]); X2 = np.array([4.0, 5.0, 6.0])
    lX1 = "(vt-from-sequence (list 1.0d0 2.0d0 3.0d0) :dtype :float64)"
    lX2 = "(vt-from-sequence (list 4.0d0 5.0d0 6.0d0) :dtype :float64)"
    add('cross 3d向量', f"(vt-cross {lX1} {lX2})", lambda: np.cross(X1, X2), 'tol')
    X3 = rng(110).rand(2, 3).round(3)
    X4 = rng(111).rand(2, 3).round(3)
    lX3 = f"(vt-from-sequence {lit(X3, 'float64')} :dtype :float64)"
    lX4 = f"(vt-from-sequence {lit(X4, 'float64')} :dtype :float64)"
    add('cross 批量', f"(vt-cross {lX3} {lX4})", lambda: np.cross(X3, X4), 'tol')
    # ---- cov / corrcoef ----
    CV = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    lCV = "(vt-from-sequence (list (list 1.0d0 2.0d0 3.0d0) (list 4.0d0 5.0d0 6.0d0)) :dtype :float64)"
    add('cov 2d rowvar', f"(vt-cov {lCV})", lambda: np.cov(CV), 'tol')
    add('cov rowvar=nil', f"(vt-cov {lCV} :rowvar nil)", lambda: np.cov(CV, rowvar=False), 'tol')
    add('cov ddof=0', f"(vt-cov {lCV} :ddof 0)", lambda: np.cov(CV, ddof=0), 'tol')
    add('cov bias=t', f"(vt-cov {lCV} :bias t)", lambda: np.cov(CV, bias=True), 'tol')
    add('cov 1d', f"(vt-cov {lVd1})", lambda: np.asarray(np.cov(Vd1)), 'tol')
    add('cov y 参数', f"(vt-cov {lVd1} :y {lVd2})", lambda: np.cov(Vd1, Vd2), 'tol')
    CC = rng(112).rand(4, 3).round(3)
    lCC = f"(vt-from-sequence {lit(CC, 'float64')} :dtype :float64)"
    add('corrcoef 2d', f"(vt-corrcoef {lCC})", lambda: np.corrcoef(CC), 'tol')
    add('corrcoef 1d', f"(vt-corrcoef {lVd1})", lambda: np.asarray(np.corrcoef(Vd1)), 'tol')
    # ---- matrix-power / cond / multi-dot ----
    MP = np.array([[2.0, 1.0], [1.0, 3.0]])
    lMP = "(vt-from-sequence (list (list 2.0d0 1.0d0) (list 1.0d0 3.0d0)) :dtype :float64)"
    add('matrix-power 0 → I', f"(vt-matrix-power {lMP} 0)", lambda: np.linalg.matrix_power(MP, 0), 'tol')
    add('matrix-power 2', f"(vt-matrix-power {lMP} 2)", lambda: np.linalg.matrix_power(MP, 2), 'tol')
    add('matrix-power 3', f"(vt-matrix-power {lMP} 3)", lambda: np.linalg.matrix_power(MP, 3), 'tol')
    add('matrix-power -1', f"(vt-matrix-power {lMP} -1)", lambda: np.linalg.matrix_power(MP, -1), 'tol')
    CD = rng(113).rand(4, 4).round(3) + np.eye(4)
    lCD = f"(vt-from-sequence {lit(CD, 'float64')} :dtype :float64)"
    add('cond 2范数', f"(vt-cond {lCD})", lambda: np.asarray(np.linalg.cond(CD)), 'tol')
    add('cond p=1', f"(vt-cond {lCD} :p 1)", lambda: np.asarray(np.linalg.cond(CD, 1)), 'tol')
    MD1 = np.arange(4.0).reshape(2, 2)
    MD2 = np.arange(4.0, 8.0).reshape(2, 2)
    MD3 = np.arange(8.0, 12.0).reshape(2, 2)
    lMD1 = f"(vt-from-sequence {lit(MD1, 'float64')} :dtype :float64)"
    lMD2 = f"(vt-from-sequence {lit(MD2, 'float64')} :dtype :float64)"
    lMD3 = f"(vt-from-sequence {lit(MD3, 'float64')} :dtype :float64)"
    add('multi-dot 3矩阵', f"(vt-multi-dot (list {lMD1} {lMD2} {lMD3}))",
        lambda: np.linalg.multi_dot([MD1, MD2, MD3]), 'tol')
    MD4 = np.arange(12.0, 16.0).reshape(2, 2)
    lMD4 = f"(vt-from-sequence {lit(MD4, 'float64')} :dtype :float64)"
    add('multi-dot 4矩阵', f"(vt-multi-dot (list {lMD1} {lMD2} {lMD3} {lMD4}))",
        lambda: np.linalg.multi_dot([MD1, MD2, MD3, MD4]), 'tol')

# G6 专用数据（提前定义，避免惰性初始化混乱）
Vd1 = np.array([1.0, 2.0, 3.0]); Vd2 = np.array([4.0, 5.0, 6.0])
lVd1 = "(vt-from-sequence (list 1.0d0 2.0d0 3.0d0) :dtype :float64)"
lVd2 = "(vt-from-sequence (list 4.0d0 5.0d0 6.0d0) :dtype :float64)"
S3x3 = np.array([[4.0, 1.0, 0.0], [1.0, 3.0, 1.0], [0.0, 1.0, 2.0]])
lS3x3 = "(vt-from-sequence (list (list 4.0d0 1.0d0 0.0d0) (list 1.0d0 3.0d0 1.0d0) (list 0.0d0 1.0d0 2.0d0)) :dtype :float64)"

# =====================================================================
# G7  fill / interp / kron / meshgrid（226-229）
# =====================================================================
def g7():
    add('fill 2d', "(vt-fill (vt-zeros '(2 3)) 7)",
        lambda: (lambda z: (z.fill(7.0), z)[1])(np.zeros((2, 3))))
    add('fill float', "(vt-fill (vt-zeros '(2 2) :dtype :float64) 2.5d0)",
        lambda: np.full((2, 2), 2.5))
    add('fill int dtype', "(vt-fill (vt-zeros '(3) :dtype :int64) 9)",
        lambda: np.full(3, 9, dtype=np.int64))
    X = np.array([0.0, 1.0, 2.0, 3.0]); XP = np.array([0.0, 2.0, 4.0]); FP = np.array([10.0, 20.0, 30.0])
    lX = "(vt-from-sequence (list 0.0d0 1.0d0 2.0d0 3.0d0) :dtype :float64)"
    lXP = "(vt-from-sequence (list 0.0d0 2.0d0 4.0d0) :dtype :float64)"
    lFP = "(vt-from-sequence (list 10.0d0 20.0d0 30.0d0) :dtype :float64)"
    add('interp 线性', f"(vt-interp {lX} {lXP} {lFP})", lambda: np.interp(X, XP, FP), 'tol')
    add('interp 端点截断', f"(vt-interp {lX} {lXP} {lFP} :left -1.0d0 :right 99.0d0)",
        lambda: np.interp(X, XP, FP, left=-1.0, right=99.0), 'tol')
    add('interp 默认端点', f"(vt-interp {lX} {lXP} {lFP})",
        lambda: np.interp(X, XP, FP), 'tol')
    K1 = np.array([[1, 2], [3, 4]], dtype=np.int64)
    K2 = np.array([[0, 1], [1, 0]], dtype=np.int64)
    lK1 = f"(vt-from-sequence {lit(K1, 'int64')} :dtype :int64)"
    lK2 = f"(vt-from-sequence {lit(K2, 'int64')} :dtype :int64)"
    add('kron 2d×2d int', f"(vt-kron {lK1} {lK2})", lambda: np.kron(K1, K2))
    Ka = np.array([1.0, 2.0]); Kb = np.array([10.0, 20.0])
    lKa = "(vt-from-sequence (list 1.0d0 2.0d0) :dtype :float64)"
    lKb = "(vt-from-sequence (list 10.0d0 20.0d0) :dtype :float64)"
    add('kron 1d×1d', f"(vt-kron {lKa} {lKb})", lambda: np.kron(Ka, Kb))
    KI = np.eye(2, dtype=np.int64)
    K3 = np.array([[5, 6], [7, 8], [9, 10]], dtype=np.int64)
    lKI = "(vt-from-sequence (list (list 1 0) (list 0 1)) :dtype :int64)"
    lK3 = f"(vt-from-sequence {lit(K3, 'int64')} :dtype :int64)"
    add('kron I×3x2', f"(vt-kron {lKI} {lK3})", lambda: np.kron(KI, K3))
    MG1 = np.array([1.0, 2.0, 3.0]); MG2 = np.array([10.0, 20.0])
    lMG1 = "(vt-from-sequence (list 1.0d0 2.0d0 3.0d0) :dtype :float64)"
    lMG2 = "(vt-from-sequence (list 10.0d0 20.0d0) :dtype :float64)"
    add('meshgrid xy', f"(vt-meshgrid (list {lMG1} {lMG2}))",
        lambda: np.meshgrid(MG1, MG2), 'tol')
    add('meshgrid ij', f"(vt-meshgrid (list {lMG1} {lMG2}) :indexing :ij)",
        lambda: np.meshgrid(MG1, MG2, indexing='ij'), 'tol')
    add('meshgrid sparse xy', f"(vt-meshgrid (list {lMG1} {lMG2}) :sparse t)",
        lambda: np.meshgrid(MG1, MG2, sparse=True), 'tol')

# =====================================================================
# G8  NN 激活（230-240）
# =====================================================================
def g8():
    X = np.array([-3.0, -1.0, -0.5, 0.0, 0.5, 1.0, 3.0, 30.0, -30.0, np.nan])
    lX = ("(vt-from-sequence (list -3.0d0 -1.0d0 -0.5d0 0.0d0 0.5d0 1.0d0 3.0d0 30.0d0 -30.0d0 +nan+) :dtype :float64)")
    add('sigmoid', f"(vt-sigmoid {lX})", lambda: 1 / (1 + np.exp(-X)), 'tol')
    add('relu', f"(vt-relu {lX})", lambda: np.maximum(X, 0.0), 'tol')
    add('relu 整数输入', "(vt-relu (vt-from-sequence (list -2 0 3) :dtype :int64))",
        lambda: np.maximum(np.array([-2, 0, 3], dtype=np.int64), np.int64(0)))
    add('leaky-relu 默认alpha', f"(vt-leaky-relu {lX})",
        lambda: np.where(X > 0, X, 0.01 * X), 'tol')
    add('leaky-relu alpha=0.2', f"(vt-leaky-relu {lX} :alpha 0.2d0)",
        lambda: np.where(X > 0, X, 0.2 * X), 'tol')
    add('leaky-relu 整数 → f64', "(vt-leaky-relu (vt-from-sequence (list -2 3) :dtype :int64))",
        lambda: np.where(np.array([-2, 3]) > 0, np.array([-2, 3]), 0.01 * np.array([-2, 3])), 'tol')
    add('swish', f"(vt-swish {lX})", lambda: X * (1 / (1 + np.exp(-X))), 'tol')
    add('softplus', f"(vt-softplus {lX})", lambda: np.logaddexp(0, X), 'tol')
    def gelu_tanh(x):
        return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x ** 3)))
    add('gelu tanh近似', f"(vt-gelu {lX})", lambda: gelu_tanh(X), 'tol')
    add('mish', f"(vt-mish {lX})", lambda: X * np.tanh(np.logaddexp(0, X)), 'tol')
    add('hard-tanh', f"(vt-hard-tanh {lX})", lambda: np.clip(X, -1.0, 1.0), 'tol')
    add('hard-sigmoid', f"(vt-hard-sigmoid {lX})", lambda: np.clip(X / 5 + 0.5, 0, 1), 'tol')
    S1 = np.array([1.0, 2.0, 3.0])
    lS1 = "(vt-from-sequence (list 1.0d0 2.0d0 3.0d0) :dtype :float64)"
    add('softmax 1d', f"(vt-softmax {lS1})",
        lambda: np.exp(S1 - S1.max()) / np.exp(S1 - S1.max()).sum(), 'tol')
    S2 = rng(121).rand(2, 4).round(3)
    lS2 = f"(vt-from-sequence {lit(S2, 'float64')} :dtype :float64)"
    def softmax(x, ax):
        e = np.exp(x - x.max(axis=ax, keepdims=True))
        return e / e.sum(axis=ax, keepdims=True)
    add('softmax 2d axis-1', f"(vt-softmax {lS2} :axis -1)", lambda: softmax(S2, -1), 'tol')
    add('softmax 2d axis0', f"(vt-softmax {lS2} :axis 0)", lambda: softmax(S2, 0), 'tol')
    add('softmax 整数输入', "(vt-softmax (vt-from-sequence (list 1 2 3) :dtype :int64))",
        lambda: softmax(np.array([1, 2, 3], dtype=np.float64), -1), 'tol')
    add('log-softmax 1d', f"(vt-log-softmax {lS1})",
        lambda: S1 - S1.max() - np.log(np.exp(S1 - S1.max()).sum()), 'tol')
    add('log-softmax 2d axis0', f"(vt-log-softmax {lS2} :axis 0)",
        lambda: np.log(softmax(S2, 0)), 'tol')
    add('sigmoid float32 保持', "(vt-dtype (vt-sigmoid (vt-from-sequence (list 1.0) :dtype :float32)))",
        lambda: B(True))

# =====================================================================
# G9  NN 损失（241-244）
# =====================================================================
def g9():
    YT = np.array([[0.0, 1.0], [1.0, 0.0]])
    YPr = np.array([[0.2, 0.8], [0.7, 0.3]])
    lYT = "(vt-from-sequence (list (list 0.0d0 1.0d0) (list 1.0d0 0.0d0)) :dtype :float64)"
    lYPr = "(vt-from-sequence (list (list 0.2d0 0.8d0) (list 0.7d0 0.3d0)) :dtype :float64)"
    add('mse', f"(vt-mean-squared-error {lYT} {lYPr})",
        lambda: np.asarray(((YT - YPr) ** 2).mean()), 'tol')
    YT2 = rng(122).rand(3, 4).round(3)
    YPr2 = rng(123).rand(3, 4).round(3)
    lYT2 = f"(vt-from-sequence {lit(YT2, 'float64')} :dtype :float64)"
    lYPr2 = f"(vt-from-sequence {lit(YPr2, 'float64')} :dtype :float64)"
    add('mse 3x4', f"(vt-mean-squared-error {lYT2} {lYPr2})",
        lambda: np.asarray(((YT2 - YPr2) ** 2).mean()), 'tol')
    add('mse 整数输入', f"(vt-mean-squared-error (vt-from-sequence (list 1 2) :dtype :int64) (vt-from-sequence (list 3 5) :dtype :int64))",
        lambda: np.asarray(((np.array([1.0, 2.0]) - np.array([3.0, 5.0])) ** 2).mean()), 'tol')
    add('mse 广播 (2,2) vs (1,)', f"(vt-mean-squared-error {lYT} (vt-from-sequence (list 1.0d0) :dtype :float64))",
        lambda: np.asarray(np.mean((YT - 1.0) ** 2)), 'tol')
    EPS = 1e-7
    add('bce', f"(vt-binary-cross-entropy {lYT} {lYPr})",
        lambda: np.asarray(-(YT * np.log(np.clip(YPr, EPS, 1 - EPS)) +
                             (1 - YT) * np.log(np.clip(1 - YPr, EPS, 1 - EPS))).mean()), 'tol')
    add('bce eps=1e-3', f"(vt-binary-cross-entropy {lYT} {lYPr} :eps 1.0d-3)",
        lambda: np.asarray(-(YT * np.log(np.clip(YPr, 1e-3, 1 - 1e-3)) +
                             (1 - YT) * np.log(np.clip(1 - YPr, 1e-3, 1 - 1e-3))).mean()), 'tol')
    CE_t = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    CE_p = np.array([[0.7, 0.2, 0.1], [0.1, 0.3, 0.6]])
    lCE_t = "(vt-from-sequence (list (list 1.0d0 0.0d0 0.0d0) (list 0.0d0 0.0d0 1.0d0)) :dtype :float64)"
    lCE_p = "(vt-from-sequence (list (list 0.7d0 0.2d0 0.1d0) (list 0.1d0 0.3d0 0.6d0)) :dtype :float64)"
    add('cross-entropy', f"(vt-cross-entropy {lCE_t} {lCE_p})",
        lambda: np.asarray(-(CE_t * np.log(np.clip(CE_p, EPS, 1))).sum(-1).mean()), 'tol')
    LG = rng(124).rand(4, 3).round(3) * 3
    LB = np.array([0, 2, 1, 2], dtype=np.int64)
    lLG = f"(vt-from-sequence {lit(LG, 'float64')} :dtype :float64)"
    lLB = f"(vt-from-sequence {lit(LB, 'int64')} :dtype :int64)"
    def xent_logits(logits, labels, ax=-1):
        ax = ax % logits.ndim
        m = logits.max(axis=ax, keepdims=True)
        ls = logits - m - np.log(np.exp(logits - m).sum(axis=ax, keepdims=True))
        if ax == logits.ndim - 1:
            idx = (np.arange(len(labels)), labels)
        else:
            idx = (labels, np.arange(len(labels)))
        return -ls[idx]
    add('ce-logits mean', f"(vt-cross-entropy-logits {lLG} {lLB})",
        lambda: np.asarray(xent_logits(LG, LB).mean()), 'tol')
    add('ce-logits sum', f"(vt-cross-entropy-logits {lLG} {lLB} :reduction :sum)",
        lambda: np.asarray(xent_logits(LG, LB).sum()), 'tol')
    add('ce-logits none', f"(vt-cross-entropy-logits {lLG} {lLB} :reduction :none)",
        lambda: xent_logits(LG, LB), 'tol')
    LGA = rng(125).rand(3, 4).round(3) * 2
    LBA = np.array([1, 3, 0, 2], dtype=np.int64)
    lLGA = f"(vt-from-sequence {lit(LGA, 'float64')} :dtype :float64)"
    lLBA = f"(vt-from-sequence {lit(LBA, 'int64')} :dtype :int64)"
    add('ce-logits axis0', f"(vt-cross-entropy-logits {lLGA} {lLBA} :axis 0)",
        lambda: np.asarray(xent_logits(LGA, LBA, ax=0).mean()), 'tol')

# =====================================================================
# G10 集合运算（245-250, 373 array-equal, 374 array-equiv, 377 isin）
# =====================================================================
def g10():
    U = np.array([3, 1, 2, 1, 3, 2], dtype=np.int64)
    lU = "(vt-from-sequence (list 3 1 2 1 3 2) :dtype :int64)"
    add('unique int', f"(vt-unique {lU})", lambda: np.unique(U))
    add('unique return_index', f"(multiple-value-bind (u i inv c) (vt-unique {lU} :return-index t) (list (vt-to-list u) (vt-to-list i)))",
        lambda: tuple(np.unique(U, return_index=True)[:2]))
    add('unique return_inverse', f"(multiple-value-bind (u i inv c) (vt-unique {lU} :return-inverse t) (list (vt-to-list u) (vt-to-list inv)))",
        lambda: np.unique(U, return_inverse=True)[:2])
    add('unique return_counts', f"(multiple-value-bind (u i inv c) (vt-unique {lU} :return-counts t) (list (vt-to-list u) (vt-to-list c)))",
        lambda: np.unique(U, return_counts=True))
    UN = np.array([np.nan, 1.0, np.nan, 2.0])
    lUN = "(vt-from-sequence (list +nan+ 1.0d0 +nan+ 2.0d0) :dtype :float64)"
    add('unique NaN合并(numpy2.x equal_nan)', f"(vt-unique {lUN})", lambda: np.unique(UN), 'tol')
    U2 = np.array([[1, 2], [2, 1], [1, 2]], dtype=np.int64)
    lU2 = "(vt-from-sequence (list (list 1 2) (list 2 1) (list 1 2)) :dtype :int64)"
    add('unique 2d展平', f"(vt-unique {lU2})", lambda: np.unique(U2))
    A1 = np.array([1, 2, 3, 4], dtype=np.int64)
    A2 = np.array([3, 4, 5], dtype=np.int64)
    lA1 = f"(vt-from-sequence {lit(A1, 'int64')} :dtype :int64)"
    lA2 = f"(vt-from-sequence {lit(A2, 'int64')} :dtype :int64)"
    add('intersect1d', f"(vt-intersect1d {lA1} {lA2})", lambda: np.intersect1d(A1, A2))
    add('union1d', f"(vt-union1d {lA1} {lA2})", lambda: np.union1d(A1, A2))
    add('setdiff1d', f"(vt-setdiff1d {lA1} {lA2})", lambda: np.setdiff1d(A1, A2))
    add('setxor1d', f"(vt-setxor1d {lA1} {lA2})", lambda: np.setxor1d(A1, A2))
    add('in1d', f"(vt-in1d {lA1} {lA2})", lambda: np.isin(A1, A2).astype(np.int8))
    F1 = np.array([1.5, 2.5, 3.5]); F2 = np.array([2.5, 5.0])
    lF1 = "(vt-from-sequence (list 1.5d0 2.5d0 3.5d0) :dtype :float64)"
    lF2 = "(vt-from-sequence (list 2.5d0 5.0d0) :dtype :float64)"
    add('intersect1d float', f"(vt-intersect1d {lF1} {lF2})", lambda: np.intersect1d(F1, F2), 'tol')
    E = np.array([0, 1, 2, 5, 0], dtype=np.int64)
    T = np.array([0, 2], dtype=np.int64)
    lE = f"(vt-from-sequence {lit(E, 'int64')} :dtype :int64)"
    lT = f"(vt-from-sequence {lit(T, 'int64')} :dtype :int64)"
    add('isin 数组×数组', f"(vt-isin {lE} {lT})", lambda: np.isin(E, T).astype(np.int8))
    add('isin invert', f"(vt-isin {lE} {lT} :invert t)", lambda: (~np.isin(E, T)).astype(np.int8))
    add('isin 标量元素', f"(vt-isin 2 {lT})", lambda: np.isin(2, T).astype(np.int8))
    E2 = np.array([[1, 2], [3, 4]], dtype=np.int64)
    lE2 = "(vt-from-sequence (list (list 1 2) (list 3 4)) :dtype :int64)"
    add('isin 2d元素', f"(vt-isin {lE2} {lT})", lambda: np.isin(E2, T).astype(np.int8))
    add('array-equal 相等', f"(vt-array-equal {lA1} (vt-from-sequence (list 1 2 3 4) :dtype :int64))",
        lambda: np.array_equal(A1, np.array([1, 2, 3, 4])))
    add('array-equal 不等', f"(vt-array-equal {lA1} {lA2})",
        lambda: np.array_equal(A1, A2))
    add('array-equal 跨dtype', f"(vt-array-equal (vt-from-sequence (list 1.0d0) :dtype :float64) (vt-from-sequence (list 1) :dtype :int64))",
        lambda: np.array_equal(np.array([1.0]), np.array([1])))
    add('array-equal 形状不同', f"(vt-array-equal (vt-reshape {lA1} '(2 2)) {lA1})",
        lambda: np.array_equal(A1.reshape(2, 2), A1))
    AN = np.array([np.nan])
    lAN = "(vt-from-sequence (list +nan+) :dtype :float64)"
    add('array-equal NaN', f"(vt-array-equal {lAN} {lAN})", lambda: np.array_equal(AN, AN))
    add('array-equiv 同形状', f"(vt-array-equiv {lA1} (vt-from-sequence (list 1 2 3 4) :dtype :int64))",
        lambda: np.array_equiv(A1, np.array([1, 2, 3, 4])))
    add('array-equiv (1,3)/(3,)', f"(vt-array-equiv (vt-reshape (vt-from-sequence (list 1 2 3) :dtype :int64) '(1 3)) (vt-from-sequence (list 1 2 3) :dtype :int64))",
        lambda: np.array_equiv(np.array([[1, 2, 3]]), np.array([1, 2, 3])))
    add('array-equiv 不等', f"(vt-array-equiv (vt-from-sequence (list 1 2 3 4 5 6) :dtype :int64) (vt-from-sequence (list 1 2 3) :dtype :int64))",
        lambda: np.array_equiv(np.array([1, 2, 3, 4, 5, 6]), np.array([1, 2, 3])))

# =====================================================================
# G11 随机数族（251-272）—— 可复现性/值域/形状/统计属性探针
# =====================================================================
def g11():
    add('random 值域[0,1)',
        "(every (lambda (x) (and (>= x 0.0d0) (< x 1.0d0))) (vt-to-list (vt-random '(1000))))",
        lambda: B(True))
    add('random 形状', "(vt-shape (vt-random '(3 4)))", lambda: np.array([3, 4]))
    add('random 可复现',
        "(equalp (vt-to-list (vt-random '(5) :rng (make-generator 42))) (vt-to-list (vt-random '(5) :rng (make-generator 42))))",
        lambda: B(True))
    add('random 不同seed 不同序列',
        "(not (equalp (vt-to-list (vt-random '(5) :rng (make-generator 42))) (vt-to-list (vt-random '(5) :rng (make-generator 43)))))",
        lambda: B(True))
    add('random 流连续性',
        "(let ((g (make-generator 7)) (g2 (make-generator 7))) (let ((a (vt-to-list (vt-random '(4) :rng g))) (b (vt-to-list (vt-random '(4) :rng g)))) (equalp (append a b) (vt-to-list (vt-random '(8) :rng g2)))))",
        lambda: B(True))
    add('random int dtype 报错', "(vt-random '(3) :dtype :int64)", None, expect='err')
    add('random-uniform 值域',
        "(every (lambda (x) (and (>= x -2.0d0) (< x 3.0d0))) (vt-to-list (vt-random-uniform '(1000) :low -2.0d0 :high 3.0d0)))",
        lambda: B(True))
    add('random-uniform low=high 常量',
        "(equalp (vt-to-list (vt-random-uniform '(3) :low 7.0d0 :high 7.0d0)) (list 7.0d0 7.0d0 7.0d0))",
        lambda: B(True))
    add('random-uniform 整数dtype截断',
        "(every (lambda (x) (and (>= x 0) (<= x 3))) (vt-to-list (vt-random-uniform '(1000) :low 0.9d0 :high 3.1d0 :dtype :int64)))",
        lambda: B(True))
    add('random-uniform low>high 报错', "(vt-random-uniform '(3) :low 5.0d0 :high 1.0d0)",
        None, expect='err')
    add('random-uniform NaN low 报错', "(vt-random-uniform '(3) :low +nan+ :high 1.0d0)",
        None, expect='err')
    add('random-uniform 整数size', "(vt-shape (vt-random-uniform 3 :low 0.0d0 :high 1.0d0))",
        lambda: np.array([3]))
    add('random-normal 均值标准差',
        "(let ((v (vt-from-sequence (vt-to-list (vt-random-normal '(100000)))))) (and (< (abs (vt-item (vt-mean v))) 0.05) (< (abs (- (vt-item (vt-std v)) 1.0d0)) 0.05)))",
        lambda: B(True))
    add('random-normal std=0 常量',
        "(equalp (vt-to-list (vt-random-normal '(3) :mean 5.0d0 :std 0.0d0)) (list 5.0d0 5.0d0 5.0d0))",
        lambda: B(True))
    add('random-normal 均值偏移',
        "(let ((v (vt-from-sequence (vt-to-list (vt-random-normal '(100000) :mean 5.0d0 :std 2.0d0))))) (< (abs (- (vt-item (vt-mean v)) 5.0d0)) 0.1))",
        lambda: B(True))
    add('random-normal std<0 报错', "(vt-random-normal '(3) :std -1.0d0)", None, expect='err')
    add('random-int 值域',
        "(every (lambda (x) (and (>= x 2) (< x 10))) (vt-to-list (vt-random-int 2 10 :size '(1000))))",
        lambda: B(True))
    add('random-int size=nil 0维', "(let ((x (vt-random-int 2 10))) (and (null (vt-shape x)) (>= (vt-item x) 2) (< (vt-item x) 10)))", lambda: B(True))
    add('random-int size整数', "(vt-shape (vt-random-int 2 10 :size 4))", lambda: np.array([4]))
    add('random-int low=high 常量',
        "(equalp (vt-to-list (vt-random-int 5 5 :size '(3))) (list 5 5 5))", lambda: B(True))
    add('random-int high<low 报错', "(vt-random-int 10 2 :size '(3))", None, expect='err')
    add('random-integers 值域',
        "(every (lambda (x) (and (>= x 0) (< x 5))) (vt-to-list (vt-random-integers 0 5 :size '(100))))",
        lambda: B(True))
    add('random-seed 可复现',
        "(progn (vt-random-seed 42) (let ((x (vt-to-list (vt-random '(3))))) (vt-random-seed 42) (equalp x (vt-to-list (vt-random '(3))))))",
        lambda: B(True))
    add('random-seed 全局隔离',
        "(progn (vt-random-seed 99) (vt-random '(3)) (let ((x2 (vt-to-list (vt-random '(3))))) (with-seed (1) (vt-random '(100))) (vt-random-seed 99) (vt-random '(3)) (equalp x2 (vt-to-list (vt-random '(3))))))",
        lambda: B(True))
    add('random-choice 值域',
        "(every (lambda (x) (member x '(10 20 30))) (vt-to-list (vt-random-choice (vt-from-sequence (list 10 20 30) :dtype :int64) :size 100)))",
        lambda: B(True))
    add('random-choice 无放回去重',
        "(let ((r (vt-to-list (vt-random-choice (vt-from-sequence (list 1 2 3 4 5) :dtype :int64) :size 5 :replace nil)))) (equalp (length r) (length (remove-duplicates r))))",
        lambda: B(True))
    add('random-choice p 加权',
        "(let ((r (vt-to-list (vt-random-choice (vt-from-sequence (list 0 1) :dtype :int64) :size 10000 :p (list 0.9d0 0.1d0))))) (> (/ (count 0 r :test #'=) 10000.0d0) 0.8d0))",
        lambda: B(True))
    add('random-permutation int 排列',
        "(let ((r (vt-to-list (vt-random-permutation 5)))) (equalp (sort r #'<) (list 0 1 2 3 4)))",
        lambda: B(True))
    add('random-permutation 多重集不变',
        f"""(let* ((a {lA1d}) (r (vt-random-permutation a)))
          (and (equalp (sort (vt-to-list r) #'<) (sort (vt-to-list a) #'<))
               (equalp (vt-to-list a) (list 1 2 3 4))))""",
        lambda: B(True))
    add('random-shuffle 原地',
        f"""(let* ((a (vt-from-sequence (list 1 2 3 4 5 6 7 8) :dtype :int64))
               (r (vt-random-shuffle a)))
          (and (eq r a)
               (equalp (sort (vt-to-list a) #'<) (list 1 2 3 4 5 6 7 8))))""",
        lambda: B(True))
    add('random-shuffle axis0 行多重集',
        f"""(let* ((a (vt-from-sequence (list (list 1 2) (list 3 4) (list 5 6)) :dtype :int64))
               (sh (vt-random-shuffle a :axis 0)))
          (equalp (sort (vt-to-list (vt-flatten sh)) #'<) (list 1 2 3 4 5 6)))""",
        lambda: B(True))
    add('random-multinomial 行和=n',
        "(let ((r (vt-random-multinomial 10 (vt-from-sequence (list 0.2d0 0.3d0 0.5d0) :dtype :float64) :size '(100 3)))) (= (vt-item (vt-sum (vt-flatten r))) (* 10 100 3)))",
        lambda: B(True))
    add('random-multinomial 概率偏置',
        "(let* ((r (vt-random-multinomial 1000 (vt-from-sequence (list 0.9d0 0.1d0) :dtype :float64) :size '(50 2))) (flat (vt-to-list (vt-flatten r))) (s0 0) (s1 0)) (loop for x in flat for i from 0 do (if (evenp i) (incf s0 x) (incf s1 x))) (> s0 s1))",
        lambda: B(True))
    add('seed-sequence entropy', "(vt-seed-sequence-entropy (make-seed-sequence 42))",
        lambda: np.asarray(42))
    add('seed-sequence 可复现生成',
        "(equalp (vt-to-list (vt-random '(4) :rng (generator-from-seed-sequence (make-seed-sequence 42)))) (vt-to-list (vt-random '(4) :rng (generator-from-seed-sequence (make-seed-sequence 42)))))",
        lambda: B(True))
    add('seed-sequence spawn 独立流',
        "(let ((ss (make-seed-sequence 42))) (let ((gs (seed-sequence-spawn ss 2))) (not (equalp (vt-to-list (vt-random '(4) :rng (generator-from-seed-sequence (first gs)))) (vt-to-list (vt-random '(4) :rng (generator-from-seed-sequence (second gs))))))))",
        lambda: B(True))
    add('seed-sequence spawn 可复现',
        "(let ((ss (make-seed-sequence 7))) (let ((gs1 (seed-sequence-spawn ss 2)) (gs2 (seed-sequence-spawn ss 2))) (equalp (vt-to-list (vt-random '(4) :rng (generator-from-seed-sequence (first gs1)))) (vt-to-list (vt-random '(4) :rng (generator-from-seed-sequence (first gs2)))))))",
        lambda: B(True))
    add('generate-state+generator 可复现',
        "(let ((ss (make-seed-sequence 3))) (equalp (seed-sequence-generate-state ss) (seed-sequence-generate-state ss)))",
        lambda: B(True))
    add('with-seed 作用域',
        "(equalp (with-seed (42) (vt-to-list (vt-random '(3)))) (with-seed (42) (vt-to-list (vt-random '(3)))))",
        lambda: B(True))
    add('with-generator 流连续',
        "(let ((g (make-generator 9)) (g2 (make-generator 9))) (let ((a (with-generator (g) (vt-to-list (vt-random '(3))))) (b (with-generator (g) (vt-to-list (vt-random '(3)))))) (equalp (append a b) (vt-to-list (vt-random '(6) :rng g2)))))",
        lambda: B(True))

# G11 专用数据
lA1d = "(vt-from-sequence (list 1 2 3 4) :dtype :int64)"

# =====================================================================
# G12 NaN/Inf 标量助手（273-281）
# =====================================================================
def g12():
    add('float-nan 是NaN', "(vt-float-nan-p (vt-float-nan))", lambda: np.isnan(np.nan))
    add('float-nan-p 1.0', "(vt-float-nan-p 1.0d0)", lambda: np.isnan(np.float64(1.0)))
    add('float-nan-p 整数', "(vt-float-nan-p 3)", lambda: np.isnan(np.float64(3)))
    add('float-nan= nan/nan', "(vt-float-nan-= (vt-float-nan) (vt-float-nan))", lambda: B(True))
    add('float-nan= nan/1', "(vt-float-nan-= (vt-float-nan) 1.0d0)", lambda: B(False))
    add('float-nan= 1/1 非NaN', "(vt-float-nan-= 1.0d0 1.0d0)", lambda: B(False))
    add('float-pos-inf 标量', "(+ 0.0d0 (vt-float-pos-inf))", lambda: np.asarray(np.inf), 'tol')
    add('float-neg-inf 标量', "(+ 0.0d0 (vt-float-neg-inf))", lambda: np.asarray(-np.inf), 'tol')
    add('float-pos-inf-p', "(vt-float-pos-inf-p (vt-float-pos-inf))", lambda: np.isposinf(np.inf))
    add('float-neg-inf-p', "(vt-float-neg-inf-p (vt-float-neg-inf))", lambda: np.isneginf(-np.inf))
    add('float-pos-inf-p 1.0', "(vt-float-pos-inf-p 1.0d0)", lambda: np.isposinf(np.float64(1.0)))
    add('float-inf= inf/inf', "(vt-float-inf-= (vt-float-pos-inf) (vt-float-pos-inf))", lambda: B(True))
    add('float-inf= inf/-inf', "(vt-float-inf-= (vt-float-pos-inf) (vt-float-neg-inf))", lambda: B(False))
    add('float-inf= inf/1', "(vt-float-inf-= (vt-float-pos-inf) 1.0d0)", lambda: B(False))
    add('float-inf= -inf/-inf', "(vt-float-inf-= (vt-float-neg-inf) (vt-float-neg-inf))", lambda: B(True))
    add('float-nan-inf= nan/nan', "(vt-float-nan-inf-= (vt-float-nan) (vt-float-nan))", lambda: B(True))
    add('float-nan-inf= nan/1', "(vt-float-nan-inf-= (vt-float-nan) 1.0d0)", lambda: B(False))
    add('float-nan-inf= 1/1', "(vt-float-nan-inf-= 1.0d0 1.0d0)", lambda: B(True))
    add('float-nan-inf= inf/1', "(vt-float-nan-inf-= (vt-float-pos-inf) 1.0d0)", lambda: B(False))
    add('float-nan-inf= inf/inf', "(vt-float-nan-inf-= (vt-float-pos-inf) (vt-float-pos-inf))", lambda: B(True))

# =====================================================================
# G13 map/reduce 基元（282-286）+ parcontract（287-295）+ with-float-safe
# =====================================================================
def g13():
    V = np.array([1.0, 2.0, 3.0, 4.0])
    lV = "(vt-from-sequence (list 1.0d0 2.0d0 3.0d0 4.0d0) :dtype :float64)"
    add('map 一元平方', f"(vt-map (lambda (x) (* x x)) {lV})", lambda: V ** 2, 'tol')
    add('map 二元加法', f"(vt-map (lambda (a b) (+ a b)) {lV} {lV})", lambda: V + V, 'tol')
    add('map 三元', f"(vt-map (lambda (a b c) (* a b c)) {lV} {lV} {lV})", lambda: V * V * V, 'tol')
    add('map 广播 (2,2)+(2,1)',
        "(vt-map (lambda (a b) (+ a b)) (vt-reshape (vt-from-sequence (list 1.0d0 2.0d0 3.0d0 4.0d0) :dtype :float64) '(2 2)) (vt-reshape (vt-from-sequence (list 10.0d0 20.0d0) :dtype :float64) '(2 1)))",
        lambda: np.array([[1.0, 2.0], [3.0, 4.0]]) + np.array([[10.0], [20.0]]), 'tol')
    add('map dtype=int8 转换', f"(vt-map (lambda (x) (* x 2)) (vt-from-sequence (list 1 2) :dtype :int64) :dtype :int8)",
        lambda: (np.array([1, 2]) * 2).astype(np.int8))
    add('map out 非连续写入',
        "(let ((base (vt-zeros '(2 6))) (a (vt-from-sequence (list (list 1.0d0 2.0d0) (list 3.0d0 4.0d0)) :dtype :float64))) (vt-map (lambda (x) (* x x)) a :out (vt-slice base '(:all) '(0 6 2))) base)",
        lambda: (lambda b: (b.__setitem__((slice(None), slice(0, 6, 2)), np.array([[1.0, 4.0], [9.0, 16.0]])), b)[1])(np.zeros((2, 6))), 'tol')
    add('do-each 求和',
        f"(let ((s 0.0d0)) (vt-do-each (i x {lV}) (incf s x)) s)",
        lambda: np.asarray(V.sum()), 'tol')
    add('do-each 下标访问',
        "(let ((s 0)) (vt-do-each (i x (vt-from-sequence (list 1 2 3) :dtype :int64)) (incf s i)) s)",
        lambda: np.asarray(0 + 1 + 2))
    add('reduce 折叠求和', f"(vt-reduce {lV} nil 0.0d0 (lambda (acc x) (+ acc x)))",
        lambda: np.asarray(V.sum()), 'tol')
    add('reduce max初值', f"(vt-reduce (vt-from-sequence (list (list 1.0d0 5.0d0) (list 3.0d0 2.0d0)) :dtype :float64) 0 0.0d0 (lambda (acc x) (values (if (> x acc) x acc) t)))",
        lambda: np.max(np.array([[1.0, 5.0], [3.0, 2.0]]), axis=0), 'tol')
    add('copy-into 全量',
        f"(let ((dst (vt-zeros '(2 2))) (src (vt-from-sequence (list (list 1.0d0 2.0d0) (list 3.0d0 4.0d0)) :dtype :float64))) (vt-copy-into dst src) dst)",
        lambda: np.array([[1.0, 2.0], [3.0, 4.0]]), 'tol')
    add('copy-into 非连续dest',
        "(let ((base (vt-zeros '(2 4))) (src (vt-from-sequence (list (list 7.0d0 8.0d0) (list 9.0d0 10.0d0)) :dtype :float64))) (vt-copy-into (vt-slice base '(:all) '(1 4 2)) src) base)",
        lambda: (lambda b: (np.copyto(b[:, 1:4:2], np.array([[7, 8], [9, 10]])), b)[1])(np.zeros((2, 4))), 'tol')
    add('copy dtype转换', f"(vt-copy (vt-from-sequence (list 1.7d0 2.7d0) :dtype :float64) :dtype :int64)",
        lambda: np.array([1.7, 2.7]).astype(np.int64))
    add('copy 转置视图落地',
        "(vt-to-list (vt-copy (vt-transpose (vt-reshape (vt-arange 4 :dtype :int64) '(2 2)))))",
        lambda: np.arange(4).reshape(2, 2).T.copy().ravel())
    add('check-out 合法透传', "(let ((o (vt-zeros '(2 3)))) (if (eq (vt-check-out o '(2 3) :float64 :op-name \"t\") o) 1 0))",
        lambda: B(True))
    add('check-out 形状不符报错', "(vt-check-out (vt-zeros '(2 2)) '(2 3) :float64 :op-name \"t\")",
        None, expect='err')
    add('check-out dtype不符报错', "(vt-check-out (vt-zeros '(2 3) :dtype :int64) '(2 3) :float64 :op-name \"t\")",
        None, expect='err')
    add('check-out 广播视图报错(H4)',
        "(vt-check-out (vt-broadcast-to (vt-from-sequence '(1)) '(2 2)) '(2 2) :int64 :op-name \"t\")",
        None, expect='err')
    add('check-out-dtype 一致→nil',
        "(if (vt-check-out-dtype-consistency \"t\" :float64 (vt-zeros '(2) :dtype :float64)) 1 0)",
        lambda: np.asarray(0))
    add('check-out-dtype 不一致报错',
        "(vt-check-out-dtype-consistency \"t\" :float64 (vt-zeros '(2) :dtype :int64))",
        None, expect='err')
    add('out-writable-p 连续', "(if (vt-out-writable-p (vt-zeros '(2 3))) 1 0)", lambda: np.asarray(1))
    add('out-writable-p 广播视图',
        "(if (vt-out-writable-p (vt-broadcast-to (vt-from-sequence '(1 2)) '(2 2))) 1 0)",
        lambda: np.asarray(0))
    add('out-contig-p 连续', "(if (vt-out-contig-p (vt-zeros '(2 3))) 1 0)", lambda: np.asarray(1))
    add('out-contig-p 步进切片',
        "(if (vt-out-contig-p (vt-slice (vt-zeros '(2 4)) '(:all) '(0 4 2))) 1 0)",
        lambda: np.asarray(0))
    add('out-snapshot 别名快照',
        "(let* ((a (vt-from-sequence (list 1.0d0 2.0d0 3.0d0) :dtype :float64)) (ins (vt-out-snapshot a (list a)))) (list (if (eq (car ins) a) 0 1) (vt-to-list (car ins))))",
        lambda: np.concatenate([[1], np.array([1, 2, 3])]), 'tol')
    add('out-snapshot 无重叠透传',
        "(let* ((a (vt-from-sequence (list 1.0d0) :dtype :float64)) (o (vt-zeros '(1) :dtype :float64)) (ins (vt-out-snapshot o (list a)))) (if (eq (car ins) a) 1 0))",
        lambda: np.asarray(1))
    add('write 单元素写',
        "(let ((o (vt-zeros '(3) :dtype :int64))) (vt-write o (list 1) 42) (vt-to-list o))",
        lambda: np.array([0, 42, 0]))
    add('reduce-dtypes int8→int64', "(multiple-value-bind (c e) (vt-reduce-dtypes (list (vt-zeros '(1) :dtype :int8)) nil) (if (eq c :int64) 1 0))", lambda: np.asarray(1))
    add('reduce-dtypes float32保持', "(multiple-value-bind (c e) (vt-reduce-dtypes (list (vt-zeros '(1) :dtype :float32)) nil) (if (eq c :float32) 1 0))", lambda: np.asarray(1))
    add('reduce-dtypes int8+float64', "(multiple-value-bind (c e) (vt-reduce-dtypes (list (vt-zeros '(1) :dtype :int8) (vt-zeros '(1) :dtype :float64)) nil) (if (eq c :float64) 1 0))", lambda: np.asarray(1))
    add('reduce-dtypes uint8→int64', "(multiple-value-bind (c e) (vt-reduce-dtypes (list (vt-zeros '(1) :dtype :uint8)) nil) (if (eq c :int64) 1 0))", lambda: np.asarray(1))
    add('params-audit 摘要结构',
        "(let ((r (vt-params-audit :clvt))) (list (if (consp r) 1 0) (if (assoc 'vt-sum r) 1 0)))", lambda: np.array([1, 1]))
    add('with-float-safe 除零→inf', "(with-float-safe (/ 1.0d0 0.0d0))", lambda: np.asarray(np.inf), 'tol')
    add('with-float-safe 0/0→nan', "(with-float-safe (/ 0.0d0 0.0d0))", lambda: np.asarray(np.nan), 'tol')

# =====================================================================
# G14 extensions1（302 count-nonzero, 303 moveaxis, 306 topk,
#                307/308 print-options, 309 flatnonzero, 310 count,
#                311 clip-tensor, 312 clamp, 313 copy-to!）
# =====================================================================
def g14():
    C1 = np.array([1, 0, 2, 0, 3], dtype=np.int64)
    lC1 = "(vt-from-sequence (list 1 0 2 0 3) :dtype :int64)"
    add('count-nonzero 1d', f"(vt-count-nonzero {lC1})", lambda: np.count_nonzero(C1))
    C2 = np.array([[1, 0, 2], [0, 3, 0]], dtype=np.int64)
    lC2 = "(vt-from-sequence (list (list 1 0 2) (list 0 3 0)) :dtype :int64)"
    add('count-nonzero 2d axis0', f"(vt-count-nonzero {lC2} :axis 0)", lambda: np.count_nonzero(C2, axis=0))
    add('count-nonzero axis1 keepdims', f"(vt-count-nonzero {lC2} :axis 1 :keepdims t)",
        lambda: np.count_nonzero(C2, axis=1, keepdims=True))
    M3 = rng(131).rand(2, 3, 4).round(3)
    lM3 = f"(vt-from-sequence {lit(M3, 'float64')} :dtype :float64)"
    add('moveaxis 0→2', f"(vt-moveaxis {lM3} 0 2)", lambda: np.moveaxis(M3, 0, 2), 'tol')
    add('moveaxis (0,1)→(2,0)', f"(vt-moveaxis {lM3} (list 0 1) (list 2 0))",
        lambda: np.moveaxis(M3, [0, 1], [2, 0]), 'tol')
    TK = np.array([1.0, 5.0, 2.0, 8.0, 3.0, 9.0, 4.0])
    lTK = "(vt-from-sequence (list 1.0d0 5.0d0 2.0d0 8.0d0 3.0d0 9.0d0 4.0d0) :dtype :float64)"
    add('topk 最大3个', f"(multiple-value-bind (v i) (vt-topk {lTK} 3) v)",
        lambda: np.sort(TK)[-3:][::-1], 'tol')
    add('topk 最大3个下标', f"(multiple-value-bind (v i) (vt-topk {lTK} 3) i)",
        lambda: np.argsort(TK)[-3:][::-1].astype(np.int64), 'tol')
    add('topk 最小2个', f"(multiple-value-bind (v i) (vt-topk {lTK} 2 :largest nil) v)",
        lambda: np.sort(TK)[:2], 'tol')
    TK2 = np.array([[1.0, 9.0, 4.0], [7.0, 2.0, 8.0]])
    lTK2 = "(vt-from-sequence (list (list 1.0d0 9.0d0 4.0d0) (list 7.0d0 2.0d0 8.0d0)) :dtype :float64)"
    add('topk 2d axis0', f"(multiple-value-bind (v i) (vt-topk {lTK2} 1 :axis 0) v)",
        lambda: np.sort(TK2, axis=0)[-1:], 'tol')
    add('topk k越界报错', f"(vt-topk {lTK} 99)", None, expect='err')
    add('set/get print-options roundtrip',
        "(let ((saved (vt-get-print-options))) (unwind-protect (progn (vt-set-print-options :threshold 6 :precision 3 :indent-step 2) (vt-get-print-options)) (apply #'vt-set-print-options (mapcan (lambda (k v) (list (intern (string-upcase k) :keyword) v)) '(threshold precision indent-step) saved))))",
        lambda: np.array([6, 3, 2]))
    add('flatnonzero', f"(vt-flatnonzero {lC1})", lambda: np.flatnonzero(C1))
    add('flatnonzero 2d展平', f"(vt-flatnonzero {lC2})", lambda: np.flatnonzero(C2))
    add('count 指定值', f"(vt-count {lC1} 0)", lambda: np.count_nonzero(C1 == 0))
    add('count 2d axis0', f"(vt-count {lC2} 0 :axis 0)", lambda: np.count_nonzero(C2 == 0, axis=0))
    CT = rng(132).rand(2, 3).round(3)
    lCT = f"(vt-from-sequence {lit(CT, 'float64')} :dtype :float64)"
    add('clip-tensor', f"(vt-clip-tensor {lCT} 0.3d0 0.7d0)", lambda: np.clip(CT, 0.3, 0.7), 'tol')
    add('clamp 别名', f"(vt-clamp {lCT} 0.3d0 0.7d0)", lambda: np.clip(CT, 0.3, 0.7), 'tol')
    add('copy-to! 切片写入',
        "(let ((dst (vt-zeros '(2 3))) (src (vt-from-sequence (list (list 1.0d0 2.0d0) (list 3.0d0 4.0d0)) :dtype :float64))) (vt-copy-to! (vt-slice dst '(:all) '(0 3 2)) src) dst)",
        lambda: (lambda b: (np.copyto(b[:, 0:3:2], np.array([[1, 2], [3, 4]])), b)[1])(np.zeros((2, 3))), 'tol')

# =====================================================================
# G15 extensions2（314 fliplr, 315 flipud, 316 ediff1d, 317 geomspace,
#                 318 ravel-multi-index, 319/320 tril/triu-indices,
#                 321 vander, 322 one-hot, 323 standardize,
#                 324 layer-norm, 325 apply-along-axis）
# =====================================================================
def g15():
    M = np.arange(6, dtype=np.int64).reshape(2, 3)
    lM = "(vt-from-sequence (list (list 0 1 2) (list 3 4 5)) :dtype :int64)"
    add('fliplr', f"(vt-fliplr {lM})", lambda: np.fliplr(M))
    add('flipud', f"(vt-flipud {lM})", lambda: np.flipud(M))
    E = np.array([1, 4, 7, 10], dtype=np.int64)
    lE = "(vt-from-sequence (list 1 4 7 10) :dtype :int64)"
    add('ediff1d', f"(vt-ediff1d {lE})", lambda: np.ediff1d(E))
    add('ediff1d to-end/to-beginning',
        f"(vt-ediff1d {lE} :to-end (list 99) :to-beginning (list -1 -2))",
        lambda: np.ediff1d(E, to_end=[99], to_begin=[-1, -2]))
    G = np.geomspace(1, 1000, 4)
    lG = "(vt-geomspace 1.0d0 1000.0d0 4)"
    add('geomspace 1→1000', lG, lambda: G, 'tol')
    add('geomspace 2点', "(vt-geomspace 1.0d0 4.0d0 2)", lambda: np.geomspace(1, 4, 2), 'tol')
    RMI = np.array([[1, 2], [0, 1]], dtype=np.int64)
    lRMI = "(vt-from-sequence (list (list 1 2) (list 0 1)) :dtype :int64)"
    add('ravel-multi-index', f"(vt-ravel-multi-index (list (list 1 0) (list 2 1)) '(3 4))",
        lambda: np.ravel_multi_index((np.array([1, 0]), np.array([2, 1])), (3, 4)))
    add('tril-indices 4x4 k=0', "(multiple-value-bind (i j) (vt-tril-indices 4) (append (vt-to-list i) (vt-to-list j)))",
        lambda: np.concatenate([x.ravel() for x in np.tril_indices(4)]))
    add('tril-indices k=1', "(multiple-value-bind (i j) (vt-tril-indices 4 :k 1) (append (vt-to-list i) (vt-to-list j)))",
        lambda: np.concatenate([x.ravel() for x in np.tril_indices(4, 1)]))
    add('triu-indices 3x3 k=-1', "(multiple-value-bind (i j) (vt-triu-indices 3 :k -1) (append (vt-to-list i) (vt-to-list j)))",
        lambda: np.concatenate([x.ravel() for x in np.triu_indices(3, -1)]))
    VD = np.array([1.0, 2.0, 3.0])
    lVD = "(vt-from-sequence (list 1.0d0 2.0d0 3.0d0) :dtype :float64)"
    add('vander 默认降幂', f"(vt-vander {lVD})", lambda: np.vander(VD), 'tol')
    add('vander n=4 increasing', f"(vt-vander {lVD} :n 4 :increasing t)",
        lambda: np.vander(VD, 4, increasing=True), 'tol')
    OH = np.array([1, 3, 0], dtype=np.int64)
    lOH = "(vt-from-sequence (list 1 3 0) :dtype :int64)"
    add('one-hot depth=4', f"(vt-one-hot {lOH} 4)", lambda: np.eye(4)[OH])
    add('one-hot 越界报错', f"(vt-one-hot (vt-from-sequence (list 4) :dtype :int64) 4)",
        None, expect='err')
    ST = rng(133).rand(2, 3).round(3)
    lST = f"(vt-from-sequence {lit(ST, 'float64')} :dtype :float64)"
    add('standardize 全局', f"(vt-standardize {lST})",
        lambda: (ST - ST.mean()) / ST.std(), 'tol')
    add('standardize axis0', f"(vt-standardize {lST} :axis 0)",
        lambda: (ST - ST.mean(axis=0)) / ST.std(axis=0), 'tol')
    LN = rng(134).rand(2, 4).round(3)
    lLN = f"(vt-from-sequence {lit(LN, 'float64')} :dtype :float64)"
    add('layer-norm 最后一维',
        f"(vt-layer-norm {lLN} (list 4))",
        lambda: (LN - LN.mean(-1, keepdims=True)) / np.sqrt(LN.var(-1, keepdims=True) + 1e-5), 'tol')
    AAA = np.arange(6, dtype=np.int64).reshape(2, 3)
    lAAA = "(vt-from-sequence (list (list 0 1 2) (list 3 4 5)) :dtype :int64)"
    add('apply-along-axis 反转', f"(vt-apply-along-axis (lambda (v) (vt-take v (vt-from-sequence (list 2 1 0) :dtype :int64))) 1 {lAAA})",
        lambda: np.apply_along_axis(lambda v: v[::-1], 1, AAA))
    add('apply-along-axis 求和', f"(vt-apply-along-axis (lambda (v) (vt-item (vt-sum v))) 0 {lAAA})",
        lambda: np.apply_along_axis(np.sum, 0, AAA))

# =====================================================================
# G16 extensions3 创建类（326 asarray, 327 fromiter, 328 tri,
#      329 diagflat, 330 trim-zeros, 331 rollaxis, 332 column-stack,
#      333 block, 334 broadcast-arrays, 335 resize）
# =====================================================================
def g16():
    add('asarray 整数嵌套', "(vt-asarray (list (list 1 2) (list 3 4)))",
        lambda: np.asarray([[1, 2], [3, 4]]))
    add('asarray 浮点', "(vt-asarray (list 1.5d0 2.5d0))", lambda: np.asarray([1.5, 2.5]), 'tol')
    add('asarray dtype指定', "(vt-asarray (list 1.7d0 2.2d0) :dtype :int64)",
        lambda: np.asarray([1.7, 2.2], dtype=np.int64))
    add('asarray 从vt', "(vt-asarray (vt-from-sequence (list 1 2) :dtype :int64))",
        lambda: np.asarray([1, 2]))
    add('fromiter 列表', "(vt-fromiter (list 1 4 9) :dtype :int64)", lambda: np.fromiter([1, 4, 9], dtype=np.int64))
    add('fromiter 浮点', "(vt-fromiter (list 1.5d0 2.5d0) :dtype :float64)",
        lambda: np.fromiter([1.5, 2.5], dtype=np.float64), 'tol')
    add('tri 3x4', "(vt-tri 3 :m 4)", lambda: np.tri(3, 4))
    add('tri k=1', "(vt-tri 3 :m 3 :k 1)", lambda: np.tri(3, 3, 1))
    add('diagflat', "(vt-diagflat (list 1 2 3))", lambda: np.diagflat([1, 2, 3]))
    add('diagflat k=1', "(vt-diagflat (list 1 2) :k 1)", lambda: np.diagflat([1, 2], 1))
    add('trim-zeros 两端', "(vt-trim-zeros (vt-from-sequence (list 0 1 2 0 0) :dtype :int64))",
        lambda: np.trim_zeros(np.array([0, 1, 2, 0, 0])))
    add('trim-zeros 仅前部', "(vt-trim-zeros (vt-from-sequence (list 0 1 2 0) :dtype :int64) :trim :f)",
        lambda: np.trim_zeros(np.array([0, 1, 2, 0]), 'f'))
    RA = np.arange(6, dtype=np.int64).reshape(2, 3)
    lRA = "(vt-from-sequence (list (list 0 1 2) (list 3 4 5)) :dtype :int64)"
    RA3 = np.arange(24, dtype=np.int64).reshape(2, 3, 4)
    lRA3 = f"(vt-from-sequence {lit(RA3, 'int64')} :dtype :int64)"
    add('rollaxis 2→0', f"(vt-rollaxis {lRA3} 2)", lambda: np.rollaxis(RA3, 2))
    add('rollaxis 3d 1→0', f"(vt-rollaxis {lRA3} 1)", lambda: np.rollaxis(RA3, 1))
    add('rollaxis start=1', f"(vt-rollaxis {lRA3} 2 1)", lambda: np.rollaxis(RA3, 2, 1))
    V1 = np.array([1, 2], dtype=np.int64); V2 = np.array([3, 4], dtype=np.int64)
    lV1 = f"(vt-from-sequence {lit(V1, 'int64')} :dtype :int64)"
    lV2 = f"(vt-from-sequence {lit(V2, 'int64')} :dtype :int64)"
    add('column-stack 1d×2', f"(vt-column-stack (list {lV1} {lV2}))",
        lambda: np.column_stack([V1, V2]))
    M22 = np.array([[5, 6], [7, 8]], dtype=np.int64)
    lM22 = f"(vt-from-sequence {lit(M22, 'int64')} :dtype :int64)"
    add('column-stack 1d+2d', f"(vt-column-stack (list {lV1} {lM22}))",
        lambda: np.column_stack([V1, M22]))
    B11 = np.array([[1, 2], [3, 4]], dtype=np.int64)
    B12 = np.array([[5, 6], [7, 8]], dtype=np.int64)
    B21 = np.array([[9, 10], [11, 12]], dtype=np.int64)
    B22 = np.array([[13, 14], [15, 16]], dtype=np.int64)
    lB11 = f"(vt-from-sequence {lit(B11, 'int64')} :dtype :int64)"
    lB12 = f"(vt-from-sequence {lit(B12, 'int64')} :dtype :int64)"
    lB21 = f"(vt-from-sequence {lit(B21, 'int64')} :dtype :int64)"
    lB22 = f"(vt-from-sequence {lit(B22, 'int64')} :dtype :int64)"
    add('block 2x2 块', f"(vt-block (list (list {lB11} {lB12}) (list {lB21} {lB22})))",
        lambda: np.block([[B11, B12], [B21, B22]]))
    add('block 1d 拼接', f"(vt-block (list {lV1} {lV2}))", lambda: np.block([V1, V2]))
    BA1 = rng(135).rand(3, 1).round(3)
    BA2 = rng(136).rand(1, 4).round(3)
    lBA1 = f"(vt-from-sequence {lit(BA1, 'float64')} :dtype :float64)"
    lBA2 = f"(vt-from-sequence {lit(BA2, 'float64')} :dtype :float64)"
    add('broadcast-arrays (3,1)/(1,4)',
        f"(let ((r (vt-broadcast-arrays (list {lBA1} {lBA2})))) (list (vt-to-list (first r)) (vt-to-list (second r))))",
        lambda: np.broadcast_arrays(BA1, BA2), 'tol')
    add('broadcast-arrays 形状',
        f"(let ((r (vt-broadcast-arrays (list {lBA1} {lBA2})))) (append (vt-shape (first r)) (vt-shape (second r))))",
        lambda: np.array([3, 4, 3, 4]))
    add('resize 放大重复', f"(vt-resize {lV1} '(4))", lambda: np.resize(V1, 4))
    add('resize 缩小截断', f"(vt-resize {lRA} '(2 2))", lambda: np.resize(RA, (2, 2)))
    add('resize 2d→3d', f"(vt-resize {lV1} '(2 2))", lambda: np.resize(V1, (2, 2)))

# =====================================================================
# G17 extensions3 元素/索引类（336 take-along-axis, 337 put-along-axis,
#      338 compress, 339 indices, 340 fill-diagonal,
#      341 absolute, 342 sign, 343 positive, 344 expm1, 345 log1p,
#      346 logaddexp, 347 float-power, 348 copysign, 349 signbit,
#      350 nextafter, 351 spacing, 352 gcd, 353 lcm, 354 divmod,
#      355 nan-to-num, 356 real, 357 imag, 358 conj, 359 angle）
# =====================================================================
def g17():
    TA = np.array([[10, 20, 30], [40, 50, 60]], dtype=np.int64)
    TI = np.array([[0, 2, 1], [1, 0, 2]], dtype=np.int64)
    lTA = f"(vt-from-sequence {lit(TA, 'int64')} :dtype :int64)"
    lTI = f"(vt-from-sequence {lit(TI, 'int64')} :dtype :int64)"
    add('take-along-axis axis1', f"(vt-take-along-axis {lTA} {lTI} 1)",
        lambda: np.take_along_axis(TA, TI, 1))
    add('put-along-axis 原位',
        f"(let ((r (vt-put-along-axis {lTA} {lTI} (vt-from-sequence (list (list 1 2 3) (list 4 5 6)) :dtype :int64) 1))) (list (vt-to-list r) (vt-to-list {lTA})))",
        lambda: (lambda c: (np.put_along_axis(c, TI, np.array([[1, 2, 3], [4, 5, 6]]), 1), np.concatenate([c.ravel(), TA.ravel()]))[1])(TA.copy()))
    CA = np.arange(8, dtype=np.int64)
    lCA = f"(vt-from-sequence {lit(CA, 'int64')} :dtype :int64)"
    COND = np.array([1, 0, 1, 0, 1, 0, 1, 0], dtype=np.int64)
    lCOND = f"(vt-from-sequence {lit(COND, 'int64')} :dtype :int64)"
    add('compress axis0', f"(vt-compress {lCOND} {lCA} :axis 0)",
        lambda: np.compress(COND.astype(bool), CA))
    add('compress 2d axis1',
        f"(vt-compress (vt-from-sequence (list 1 0 1) :dtype :int64) {lRA16} :axis 1)",
        lambda: np.compress(np.array([1, 0, 1], dtype=bool), RA16, axis=1))
    add('indices (2,3)', "(vt-indices '(2 3))",
        lambda: np.stack(np.indices((2, 3))))
    FD = np.zeros((3, 3))
    lFD = "(vt-fill-diagonal (vt-zeros '(3 3)) 7.0d0)"
    add('fill-diagonal 返回矩阵', lFD,
        lambda: (lambda m: (np.fill_diagonal(m, 7.0), m)[1])(np.zeros((3, 3))), 'tol')
    FD2 = np.zeros((3, 3), dtype=np.int64)
    add('fill-diagonal int', "(let ((m (vt-zeros '(3 3) :dtype :int64))) (vt-fill-diagonal m 5) m)",
        lambda: (lambda m: (np.fill_diagonal(m, 5), m)[1])(FD2))
    X = np.array([-1.5, 0.0, 2.5, -0.0])
    lX = "(vt-from-sequence (list -1.5d0 0.0d0 2.5d0 -0.0d0) :dtype :float64)"
    add('absolute', f"(vt-absolute {lX})", lambda: np.absolute(X))
    add('sign', f"(vt-sign {lX})", lambda: np.sign(X))
    add('positive', f"(vt-positive {lX})", lambda: np.positive(X))
    E1 = np.array([1e-10, 1.0, 2.0, np.inf, -0.0])
    lE1 = "(vt-from-sequence (list 1.0d-10 1.0d0 2.0d0 +inf+ -0.0d0) :dtype :float64)"
    add('expm1', f"(vt-expm1 {lE1})", lambda: np.expm1(E1), 'tol')
    add('log1p', f"(vt-log1p (vt-from-sequence (list 1.0d-10 1.0d0 0.0d0 -1.0d0) :dtype :float64))",
        lambda: np.log1p(np.array([1e-10, 1.0, 0.0, -1.0])), 'tol')
    A2 = np.array([1.0, 2.0, 1e10]); B2 = np.array([2.0, 1.0, 1e-10])
    lA2 = "(vt-from-sequence (list 1.0d0 2.0d0 1.0d10) :dtype :float64)"
    lB2 = "(vt-from-sequence (list 2.0d0 1.0d0 1.0d-10) :dtype :float64)"
    add('logaddexp', f"(vt-logaddexp {lA2} {lB2})", lambda: np.logaddexp(A2, B2), 'tol')
    FP1 = np.array([2.0, -2.0, 4.0])
    lFP1 = "(vt-from-sequence (list 2.0d0 -2.0d0 4.0d0) :dtype :float64)"
    FP2 = np.array([3.0, 0.5, 0.25])
    lFP2 = "(vt-from-sequence (list 3.0d0 0.5d0 0.25d0) :dtype :float64)"
    add('float-power', f"(vt-float-power {lFP1} {lFP2})", lambda: np.float_power(FP1, FP2), 'tol')
    CS1 = np.array([1.0, 2.0, np.nan, -0.0]); CS2 = np.array([-0.0, 1.0, -1.0, 1.0])
    lCS1 = "(vt-from-sequence (list 1.0d0 2.0d0 +nan+ -0.0d0) :dtype :float64)"
    lCS2 = "(vt-from-sequence (list -0.0d0 1.0d0 -1.0d0 1.0d0) :dtype :float64)"
    add('copysign', f"(vt-copysign {lCS1} {lCS2})", lambda: np.copysign(CS1, CS2), 'tol')
    add('signbit', f"(vt-signbit (vt-from-sequence (list -0.0d0 1.0d0 -1.0d0 (sb-int:with-float-traps-masked (:invalid :divide-by-zero) (/ -0.0d0 0.0d0))) :dtype :float64))",
        lambda: np.signbit(np.array([-0.0, 1.0, -1.0, -np.nan])).astype(np.int8))
    NA = np.array([1.0, 0.0]); NB = np.array([2.0, -1.0])
    lNA = "(vt-from-sequence (list 1.0d0 0.0d0) :dtype :float64)"
    lNB = "(vt-from-sequence (list 2.0d0 -1.0d0) :dtype :float64)"
    add('nextafter', f"(vt-nextafter {lNA} {lNB})", lambda: np.nextafter(NA, NB), 'tol')
    add('spacing', f"(vt-spacing (vt-from-sequence (list 1.0d0 2.0d0) :dtype :float64))",
        lambda: np.spacing(np.array([1.0, 2.0])), 'tol')
    G1 = np.array([12, 18, 0, -4], dtype=np.int64); G2 = np.array([8, 24, 0, 6], dtype=np.int64)
    lG1 = "(vt-from-sequence (list 12 18 0 -4) :dtype :int64)"
    lG2 = "(vt-from-sequence (list 8 24 0 6) :dtype :int64)"
    add('gcd 含负数', f"(vt-gcd {lG1} {lG2})", lambda: np.gcd(G1, G2))
    add('lcm 含负数', f"(vt-lcm {lG1} {lG2})", lambda: np.lcm(G1, G2))
    DM1 = np.array([7, -7, 8, -8], dtype=np.int64); DM2 = np.array([3, 3, -3, -3], dtype=np.int64)
    lDM1 = "(vt-from-sequence (list 7 -7 8 -8) :dtype :int64)"
    lDM2 = "(vt-from-sequence (list 3 3 -3 -3) :dtype :int64)"
    add('divmod floor语义', f"(multiple-value-bind (q r) (vt-divmod {lDM1} {lDM2}) (list (vt-to-list q) (vt-to-list r)))",
        lambda: np.divmod(DM1, DM2))
    NT = np.array([np.nan, np.inf, -np.inf, 1.0])
    lNT = "(vt-from-sequence (list +nan+ +inf+ +ninf+ 1.0d0) :dtype :float64)"
    add('nan-to-num 默认', "(vt-nan-to-num (vt-from-sequence (list +nan+ +inf+ +ninf+ 1.0d0) :dtype :float64))", lambda: np.nan_to_num(NT), 'tol')
    add('nan-to-num 自定义替换',
        "(vt-nan-to-num (vt-from-sequence (list +nan+ +inf+ +ninf+) :dtype :float64) :nan -1.0d0 :posinf 99.0d0 :neginf -99.0d0)",
        lambda: np.nan_to_num(np.array([np.nan, np.inf, -np.inf]), nan=-1.0, posinf=99.0, neginf=-99.0), 'tol')
    RV = np.array([1.0, -2.0, 0.0])
    lRV = "(vt-from-sequence (list 1.0d0 -2.0d0 0.0d0) :dtype :float64)"
    add('real', f"(vt-real {lRV})", lambda: np.real(RV))
    add('imag', f"(vt-imag {lRV})", lambda: np.imag(RV))
    add('conj', f"(vt-conj {lRV})", lambda: np.conj(RV))
    ANG = np.array([1.0, -1.0, 0.5])
    lANG = "(vt-from-sequence (list 1.0d0 -1.0d0 0.5d0) :dtype :float64)"
    add('angle', f"(vt-angle {lANG})", lambda: np.angle(ANG), 'tol')

# G17 专用数据
RA16 = np.arange(6, dtype=np.int64).reshape(2, 3)
lRA16 = "(vt-from-sequence (list (list 0 1 2) (list 3 4 5)) :dtype :int64)"

# =====================================================================
# G18 打印/内省杂项（296 print-vt-recursive, 297-299 打印变量,
#                     300 *vt-fun-list*）
# =====================================================================
def g18():
    add('*vt-fun-list* 规模', "(* (length *vt-fun-list*) 1)", lambda: np.asarray(FUN_COUNT))
    add('*vt-fun-list* 含vt-sum', "(if (member 'clvt:vt-sum *vt-fun-list*) 1 0)", lambda: np.asarray(1))
    # 注：搜索串不带 d0 后缀——与浮点表示形式无关（print-vt-recursive
    # 经 %format-number 的 ~,vF 分支输出 "1.0"，PRIN1 兜底输出 "1.0d0"，
    # 两者均包含 "1.0"），跨 SBCL/clvt 版本稳定。
    add('print-vt-recursive 1d 输出',
        "(let ((s (make-string-output-stream))) (print-vt-recursive (vt-from-sequence (list 1.0d0 2.0d0) :dtype :float64) 0 nil 2 8 :float64 s) (if (search \"1.0\" (get-output-stream-string s)) 1 0))",
        lambda: np.asarray(1))
    add('print 1d 包含数据',
        "(let ((s (make-string-output-stream))) (print-vt-recursive (vt-from-sequence (list 1.0d0 2.0d0) :dtype :float64) 0 nil 2 8 :float64 s) (if (search \"2.0\" (get-output-stream-string s)) 1 0))",
        lambda: np.asarray(1))
    add('print-vt-recursive 2d 输出',
        "(let ((s (make-string-output-stream))) (print-vt-recursive (vt-reshape (vt-from-sequence (list 1.0d0 2.0d0 3.0d0 4.0d0) :dtype :float64) '(2 2)) 0 nil 2 8 :float64 s) (if (search \"3.0\" (get-output-stream-string s)) 1 0))",
        lambda: np.asarray(1))
    add('set-print-options 生效',
        "(let ((saved (vt-get-print-options))) (unwind-protect (progn (vt-set-print-options :precision 5) (nth 1 (vt-get-print-options))) (apply #'vt-set-print-options (mapcan (lambda (k v) (list (intern (string-upcase k) :keyword) v)) '(threshold precision indent-step) saved))))",
        lambda: np.asarray(5))
    add('get-print-options 默认列表',
        "(let ((o (vt-get-print-options))) (list (if (nth 0 o) 1 0) (if (nth 1 o) 1 0) (if (nth 2 o) 1 0)))",
        lambda: np.array([1, 1, 1]))

# *vt-fun-list* 期望长度：以 SBCL 实测为准（见 discover_fun_count()）
def _static_fun_count():
    import re as _re
    _p = os.path.join(_SCRIPT_DIR, '..', 'src', 'package.lisp')
    _s = open(_p, encoding='utf-8').read()
    _i = _s.find('(:export'); _j = _s.find('(in-package', _i)
    _syms = _re.findall(r'#:([A-Za-z0-9@!*\-+<>=/_]+)', _s[_i:_j])
    return sum(1 for s in _syms if len(s) > 2 and s[:3].lower() == 'vt-')
FUN_COUNT = _static_fun_count()

# =====================================================================
# G19 extensions3 收尾（375 isposinf, 376 isneginf + 补充鲁棒性）
# =====================================================================
def g19():
    INF = np.array([np.inf, -np.inf, 0.0, np.nan, 1.0])
    lINF = "(vt-from-sequence (list +inf+ +ninf+ 0.0d0 +nan+ 1.0d0) :dtype :float64)"
    add('isposinf', f"(vt-isposinf {lINF})", lambda: np.isposinf(INF).astype(np.int8))
    add('isneginf', f"(vt-isneginf {lINF})", lambda: np.isneginf(INF).astype(np.int8))
    add('isposinf int输入', f"(vt-isposinf (vt-from-sequence (list 1 2) :dtype :int64))",
        lambda: np.isposinf(np.array([1, 2], dtype=np.int64)).astype(np.int8))

# =====================================================================
# 生成与比对（沿用第 1 阶段文件布局，产物加 2 后缀）
# =====================================================================
def tok1(v):
    v = float(v)
    if math.isnan(v): return 'nan'
    if math.isinf(v): return 'inf' if v > 0 else '-inf'
    return repr(float(v))

def cmd_gen():
    g1(); g2(); g3(); g4(); g5(); g6(); g7(); g8(); g9(); g10(); g11(); g12(); g13()
    g14(); g15(); g16(); g17(); g18(); g19()
    os.makedirs(TMP, exist_ok=True)
    exps = {}
    for cid, desc, lisp, npfn, policy, expect in CASES:
        if expect == 'err':
            exps[cid] = ('MUST-ERR', None, None, None); continue
        try:
            with np.errstate(all='ignore'):
                r = npfn()
            if (isinstance(r, (list, tuple)) and r
                    and all(isinstance(x, np.ndarray) for x in r)):
                r = np.concatenate([np.asarray(x).ravel() for x in r])
            else:
                r = np.asarray(r)
            if r.dtype == np.bool_:
                r = r.astype(np.int8)
            exps[cid] = ('VT', CLD[str(r.dtype)],
                         '(' + ' '.join(map(str, r.shape)) + ')',
                         [tok1(x) for x in r.reshape(-1)])
        except Exception:
            exps[cid] = ('ERR', None, None, None)
    # 括号配平自检（基于运行时 lisp 字符串，防止读取器读到文件尾）
    for cid, desc, lisp, npfn, policy, expect in CASES:
        if not lisp:
            continue
        _d = 0
        for _ch in re.sub(r'"(?:[^"\\\\]|\\\\.)*"', '""', lisp):
            _d += (_ch == '(') - (_ch == ')')
        if _d != 0:
            raise SyntaxError(f'{cid} {desc}: lisp 括号不配平({_d})')
    with open(PROBE_OUT, 'w') as f:
        f.write(PRELUDE)
        f.write('(setf *fz-expected* (list\n')
        for cid, desc, lisp, npfn, policy, expect in CASES:
            k, dt, sh, toks = exps[cid]
            if k == 'VT':
                toks_s = ' '.join(f'"{t}"' for t in toks)
                f.write(f' (list "{cid}" "VT" "{dt}" "{sh}" (list {toks_s}) :{policy})\n')
            else:
                f.write(f' (list "{cid}" "{k}" nil nil nil :{policy})\n')
        f.write('))\n')
        for cid, desc, lisp, npfn, policy, expect in CASES:
            if lisp:
                f.write(f'(probe "{cid}" {lisp})\n')
        f.write('(fz-summary)\n')
        f.write('(sb-ext:exit :code (if (zerop *fz-n-fail*) 0 1))\n')
    with open(f'{TMP}/expected2.txt', 'w') as f:
        for cid, desc, lisp, npfn, policy, expect in CASES:
            k, dt, sh, toks = exps[cid]
            if k == 'MUST-ERR':
                f.write(f"{cid}|MUST-ERR|\n")
            elif k == 'ERR':
                f.write(f"{cid}|ERR|\n")
            else:
                f.write(f"{cid}|VT|dtype={dt} shape={sh} data=[{' '.join(toks)}]\n")
    meta = [{'id': c[0], 'desc': c[1], 'lisp': c[2], 'policy': c[4], 'expect': c[5]}
            for c in CASES]
    json.dump(meta, open(f'{TMP}/cases2.json', 'w'), ensure_ascii=False)
    print(f"生成 {len(CASES)} 个探针 → {PROBE_OUT}, {TMP}/expected2.txt")

def cmd_compare():
    exp = {}
    for line in open(f'{TMP}/expected2.txt'):
        line = line.strip()
        if not line: continue
        cid, rest = line.split('|', 1)
        exp[cid] = rest
    act = {}
    for line in open(f'{TMP}/actual2.txt'):
        line = line.strip()
        if not line: continue
        cid, rest = line.split('|', 1)
        act[cid] = rest
    fails = {'missing': [], 'kind': [], 'dtype': [], 'shape': [], 'len': [],
             'data-exact': [], 'data-tol': [], 'err-not-raised': [], 'numpy-err-ok': []}
    for cid, es in exp.items():
        if cid not in act:
            fails['missing'].append((cid, '', '', es, ''))
            continue
        as_ = act[cid]
        ek, _, esh, edata = (es.split('|', 1) + [''])[:1] + (None,) * 3
        # 细拆
        ekind = es.split('|')[0] if '|' in es else es
        akind = as_.split('|')[0] if '|' in as_ else as_
        if ekind == 'MUST-ERR':
            (fails['missing' if akind != 'ERR' else 'err-not-raised'
                   if False else 'ok'].append((cid, '', '', es, as_))
             if akind != 'ERR' else None)
            if akind == 'ERR':
                continue
            fails['err-not-raised'].append((cid, '', '', es, as_))
            continue
        if ekind == 'ERR':
            if akind == 'ERR':
                continue
            fails['numpy-err-ok'].append((cid, '', '', es, as_))
            continue
        if akind == 'ERR':
            fails['kind'].append((cid, '', '', es, as_))
            continue
        efield = dict(kv.split('=', 1) for kv in es.split(' ') if '=' in kv)
        afield = dict(kv.split('=', 1) for kv in as_.split(' ') if '=' in kv)
        if 'LIST' not in (ekind, akind) and ekind != akind and not (ekind == 'VT' and efield.get('shape') == '()'):
            fails['kind'].append((cid, '', '', es, as_)); continue
        if 'LIST' in (ekind, akind):
            pass
        elif efield.get('shape') != afield.get('shape'):
            fails['shape'].append((cid, '', '', es, as_)); continue
        edata = es[es.find('data=[') + 6: es.find(']')] if 'data=[' in es else ''
        adata = as_[as_.find('data=[') + 6: as_.find(']')] if 'data=[' in as_ else ''
        et, at = edata.split(' '), adata.split(' ')
        if len(et) != len(at):
            fails['len'].append((cid, '', '', es, as_)); continue
        bad = []
        pol = 'exact'
        for x, y in zip(et, at):
            fx, fy = _num(x), _num(y)
            ok = (_bit_eq(fx, fy) if pol == 'exact' else _tol_eq(fx, fy))
            if not ok: bad.append((x, y))
        if bad:
            fails['data-exact'].append((cid, '', '', es, as_))
    total = len(exp)
    nfail = sum(len(v) for v in fails.values())
    print(f"{'='*70}\n差分比对(阶段2): 期望={total}  不匹配={nfail}\n{'='*70}")
    for k, items in fails.items():
        if not items or k == 'ok': continue
        print(f"\n### {k} ({len(items)})")
        for cid, desc, lisp, es, as_ in items[:60]:
            print(f"  [{cid}] 期望: {es[:110]}")
            print(f"        实际: {as_[:110]}")

def _num(s):
    if s == 'nan': return ('nan',)
    if s == 'inf': return ('inf',)
    if s == '-inf': return ('ninf',)
    return ('v', float(s))

def _bit_eq(a, b):
    if a[0] == 'nan' and b[0] == 'nan': return True
    if a[0] == 'nan' or b[0] == 'nan': return False
    if a[0] in ('inf', 'ninf') or b[0] in ('inf', 'ninf'): return a == b
    return a[1] == b[1] or (a[1] == b[1] == 0.0)

def _tol_eq(a, b, rt=1e-11, at=1e-12):
    if a[0] == 'nan' and b[0] == 'nan': return True
    if a[0] == 'nan' or b[0] == 'nan': return False
    if a[0] in ('inf', 'ninf') or b[0] in ('inf', 'ninf'): return a == b
    fa, fb = a[1], b[1]
    return abs(fa - fb) <= at + rt * max(abs(fa), abs(fb))

if __name__ == '__main__':
    if sys.argv[1] == 'gen':
        cmd_gen()
    else:
        cmd_compare()
