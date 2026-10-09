#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
clvt vs numpy 差分模糊测试生成器 + 比对器
  gen     : 生成 tmp/probes.lisp（Lisp 探针）+ tmp/expected.txt（numpy 期望）+ tmp/cases.json
  compare : 读 tmp/actual.txt（clvt 实际输出）与 expected 比对，输出不匹配报告
"""
import sys, os, json, math, warnings
import numpy as np
warnings.filterwarnings('ignore')
np.seterr(all='ignore')

# 输出目录：默认当前目录下的 tmp/，可用环境变量 FUZZ_TMP 覆盖
# 本脚本位于 test/ 下：中间产物默认放仓库 tmp/（可用 FUZZ_TMP 覆盖）
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TMP  = os.environ.get('FUZZ_TMP') or os.path.join(os.path.dirname(_SCRIPT_DIR), 'tmp')
PROBE_OUT = os.path.join(_SCRIPT_DIR, 'differential-probes-test.lisp')

CLD = {'float64':':float64','float32':':float32','int8':':int8','int16':':int16',
       'int32':':int32','int64':':int64','uint8':':uint8','uint16':':uint16'}

CASES = []  # (cid, desc, lisp_form, np_thunk, policy, expect)  expect: None|'err'
def add(desc, lisp, npfn, policy='exact', expect=None):
    cid = f"c{len(CASES):04d}"
    CASES.append((cid, desc, lisp, npfn, policy, expect))
    return cid

# ---------- 数据与字面量 ----------
def gen(shape, dt, seed, lo=None, hi=None):
    rng = np.random.RandomState(seed)
    if dt in ('float32','float64'):
        a = np.round(rng.standard_normal(shape), 3)
        return a.astype(np.float32) if dt=='float32' else a
    if lo is None: lo, hi = {'int8':(-50,50),'uint8':(0,50)}.get(dt,(-50,50))
    return rng.randint(lo, hi, size=shape).astype(dt)

def fmtnum(v, dt):
    if dt.startswith('float'):
        v = float(v)
        if np.isnan(v): return '+nan+'
        if np.isposinf(v): return '+inf+'
        if np.isneginf(v): return '+ninf+'
        if dt=='float64': return f"{v:.16e}".replace('e','d',1)
        return f"{v:.8e}"
    return str(int(v))

def lit(base, dt):
    b = np.ascontiguousarray(base)
    def rec(a):
        if a.ndim==0: return fmtnum(a[()], dt)
        return '(list ' + ' '.join(rec(a[i]) for i in range(a.shape[0])) + ')'
    return rec(b)

def OP(spec, seed, data=None):
    """返回 (numpy数组[可能是视图], lisp构造式)"""
    k = spec[0]
    if k=='c0':
        dt,val = spec[1],spec[2]
        a = np.array(val, dtype=dt)
        return a, f"(vt-const nil {fmtnum(val,dt)} :dtype {CLD[dt]})"
    if k=='c':
        shape,dt = spec[1],spec[2]
        a = data if data is not None else gen(shape,dt,seed)
        return a, f"(vt-from-sequence {lit(a,dt)} :dtype {CLD[dt]})"
    if k=='t':
        shape,dt = spec[1],spec[2]
        a = data if data is not None else gen(shape,dt,seed)
        return a.T, f"(vt-transpose (vt-from-sequence {lit(a,dt)} :dtype {CLD[dt]}))"
    if k=='s':
        bshape,dt,sl = spec[1],spec[2],spec[3]
        a = data if data is not None else gen(bshape,dt,seed)
        py = tuple(slice(None) if s=='a' else (slice(*s) if isinstance(s,tuple) else s) for s in sl)
        parts=[]
        for s in sl:
            if s=='a': parts.append("'(:all)")
            elif isinstance(s,tuple): parts.append("'(" + " ".join(map(str,s)) + ")")
            else: parts.append(f"'({s})")
        return a[py], f"(vt-slice (vt-from-sequence {lit(a,dt)} :dtype {CLD[dt]}) {' '.join(parts)})"
    if k=='b':
        bshape,dt,target = spec[1],spec[2],spec[3]
        a = data if data is not None else gen(bshape,dt,seed)
        lsp = '(' + ' '.join(map(str, target)) + ')'
        return np.broadcast_to(a,target), f"(vt-broadcast-to (vt-from-sequence {lit(a,dt)} :dtype {CLD[dt]}) '{lsp})"
    raise ValueError(k)

def op2(specs, seed, fn, desc, policy='exact', expect=None, datas=None):
    """双操作数便捷封装: fn(a0,a1)->np结果"""
    datas = datas or [None,None]
    a0,l0 = OP(specs[0], seed,   datas[0])
    a1,l1 = OP(specs[1], seed+1, datas[1])
    add(desc, fn(a0,a1,l0,l1), (lambda a0=a0,a1=a1,fn=fn: fn(a0,a1)), policy, expect)

# ---------- F1 提升 8x8 ----------
def f1():
    for d1 in CLD:
        for d2 in CLD:
            v1, v2 = (1.5 if d1.startswith('float') else 2), (2.5 if d2.startswith('float') else 3)
            a1 = np.array([v1], dtype=d1); a2 = np.array([v2], dtype=d2)
            l1 = f"(vt-from-sequence '({fmtnum(v1,d1)}) :dtype {CLD[d1]})"
            l2 = f"(vt-from-sequence '({fmtnum(v2,d2)}) :dtype {CLD[d2]})"
            add(f"add提升 {d1}+{d2}", f"(vt-+ {l1} {l2})", (lambda a1=a1,a2=a2: np.add(a1,a2)))
    for d1 in CLD:
        for d2 in CLD:
            v1, v2 = (1.5 if d1.startswith('float') else 7), (2.5 if d2.startswith('float') else 4)
            a1 = np.array([v1], dtype=d1); a2 = np.array([v2], dtype=d2)
            l1 = f"(vt-from-sequence '({fmtnum(v1,d1)}) :dtype {CLD[d1]})"
            l2 = f"(vt-from-sequence '({fmtnum(v2,d2)}) :dtype {CLD[d2]})"
            add(f"div提升 true_divide {d1}/{d2}", f"(vt-/ {l1} {l2})", (lambda a1=a1,a2=a2: np.true_divide(a1,a2)))

# ---------- F2 基本算术 ----------
def f2():
    for dt in ['int64','float64','float32','uint8']:
        for shape in [(6,),(2,3)]:
            seed = 5000 + hash(shape)%97
            pairs = [
                ('add',  lambda a,b,l0,l1: f"(vt-+ {l0} {l1})", np.add, None, None),
                ('sub',  lambda a,b,l0,l1: f"(vt-- {l0} {l1})", np.subtract, None, None),
                ('mul',  lambda a,b,l0,l1: f"(vt-* {l0} {l1})", np.multiply, (-9,9), (-9,9)),
                ('floordiv', lambda a,b,l0,l1: f"(vt-div {l0} {l1})", np.floor_divide, None, (1,9)),
                ('truediv',  lambda a,b,l0,l1: f"(vt-/ {l0} {l1})", np.true_divide, None, (1,9)),
                ('mod',  lambda a,b,l0,l1: f"(vt-mod {l0} {l1})", np.remainder, None, (1,9)),
                ('rem',  lambda a,b,l0,l1: f"(vt-rem {l0} {l1})", np.fmod, None, (1,9)),
            ]
            for name, lf, nf, ra, rb in pairs:
                a = gen(shape, dt, seed+1, *ra) if ra else gen(shape, dt, seed+1)
                b = gen(shape, dt, seed+2, *rb) if rb else gen(shape, dt, seed+2)
                pol = 'tol' if dt.startswith('float') and name in ('truediv','mod','rem','expt','expt-int','pow','power') else 'exact'
                add(f"{name} {dt} {shape}", lf(a,b,'A','B'),
                    (lambda a=a,b=b,nf=nf: nf(a,b)), pol,
                    datas=[a,b] and None) if False else None
                a0,l0 = OP(('c',shape,dt), seed+1, data=a)
                a1,l1 = OP(('c',shape,dt), seed+2, data=b)
                add(f"{name} {dt} {shape}", lf(a0,a1,l0,l1), (lambda a0=a0,a1=a1,nf=nf: nf(a0,a1)), pol)
            # 一元与标量
            if dt in ('int8','uint8'):
                a = gen(shape, dt, seed+3, -9, 9)
            else:
                a = gen(shape, dt, seed+3)
            a0,l0 = OP(('c',shape,dt), seed+3, data=a)
            add(f"scale {dt} {shape}", f"(vt-scale {l0} 3.0d0)" if dt.startswith('float') else f"(vt-scale {l0} 3)",
                (lambda a0=a0: a0*3), 'exact')
            add(f"negative {dt} {shape}", f"(vt-- {l0})", (lambda a0=a0: np.negative(a0)), 'exact')
            add(f"abs {dt} {shape}", f"(vt-abs {l0})", (lambda a0=a0: np.abs(a0)), 'exact')
            add(f"square {dt} {shape}", f"(vt-square {l0})", (lambda a0=a0: np.square(a0)), 'exact')
            add(f"expt-int {dt} {shape}", f"(vt-expt {l0} 3)", (lambda a0=a0: np.power(a0,3)), 'tol')  # 浮点幂跨实现 ULP 噪声

# ---------- F3 整型溢出 ----------
def f3():
    V = lambda vals, dt: (np.array(vals, dtype=dt), f"(vt-from-sequence '({' '.join(str(v) for v in vals)}) :dtype {CLD[dt]})")
    cases = [
        ('i8 add 127+1', [127],[1], np.add),
        ('i8 add -128+-1', [-128],[-1], np.add),
        ('i8 sub -128-1', [-128],[1], np.subtract),
        ('i8 mul 127*2', [127],[2], np.multiply),
        ('i8 square 127', [127],[0], None),
        ('i8 abs -128', [-128],[0], None),
        ('i8 negative -128', [-128],[0], None),
        ('u8 add 250+10', [250],[10], np.add),
        ('u8 sub 0-1', [0],[1], np.subtract),
        ('i16 add 32767+1', [32767],[1], np.add),
        ('i64 add max+1', [9223372036854775807],[1], np.add),
    ]
    for name, xs, ys, nf in cases:
        dt = 'int8' if name.startswith('i8') else 'uint8' if name.startswith('u8') else 'int16' if name.startswith('i16') else 'int64'
        a0,l0 = V(xs, dt)
        if nf:
            a1,l1 = V(ys, dt)
            add(name, f"(vt-+ {l0} {l1})" if 'add' in name else f"(vt-- {l0} {l1})" if 'sub' in name else f"(vt-* {l0} {l1})",
                (lambda a0=a0,a1=a1,nf=nf: nf(a0,a1)))
        elif 'square' in name:
            add(name, f"(vt-square {l0})", (lambda a0=a0: np.square(a0)))
        elif 'abs' in name:
            add(name, f"(vt-abs {l0})", (lambda a0=a0: np.abs(a0)))
        else:
            add(name, f"(vt-- {l0})", (lambda a0=a0: np.negative(a0)))

# ---------- F4 舍入与特殊浮点 ----------
def f4():
    base = [-3.7,-2.5,-1.5,-0.5,0.5,1.5,2.5,3.7]
    sp   = [float('nan'), float('inf'), float('-inf'), -0.0]
    V = base + sp
    a0,l0 = OP(('c',(len(V),),'float64'), 1, data=np.array(V))
    for name, lf, nf in [
        ('floor', 'vt-floor', np.floor), ('ceil', 'vt-ceiling', np.ceil),
        ('trunc', 'vt-truncate', np.trunc), ('rint', 'vt-rint', np.rint),
        ('round', 'vt-round', np.round)]:
        add(f"{name} 含NaN/Inf/-0.0", f"({lf} {l0})", (lambda a0=a0,nf=nf: nf(a0)))
    def unary(name, vals, nf, lfn, policy='exact'):
        a0,l0 = OP(('c',(len(vals),),'float64'), 1, data=np.array(vals))
        add(name, f"({lfn} {l0})", (lambda a0=a0,nf=nf: nf(a0)), policy)
    F,N,I = float('nan'), float('inf'), float('-inf')
    unary('sqrt 特殊值', [-0.0,-1.0,0.0,4.0,F,I], np.sqrt, 'vt-sqrt')
    unary('reciprocal 特殊值', [0.0,-0.0,4.0,F], np.reciprocal, 'vt-reciprocal')
    unary('signum 特殊值', [-0.0,F,-3.0,0.0,3.0], np.sign, 'vt-signum')
    unary('cbrt 特殊值', [-8.0,0.0,27.0,-27.0,F,I], np.cbrt, 'vt-cbrt', 'tol')
    unary('log 特殊值', [0.0,-1.0,1.0,F,I], np.log, 'vt-log', 'tol')
    unary('log2 特殊值', [0.0,-1.0,8.0,F], np.log2, 'vt-log2', 'tol')
    unary('log10 特殊值', [0.0,-1.0,1000.0,F], np.log10, 'vt-log10', 'tol')
    unary('exp 特殊值', [0.0,1.0,1000.0,-1000.0], np.exp, 'vt-exp', 'tol')
    unary('asin 特殊值', [2.0,1.0,-2.0,0.5], np.arcsin, 'vt-asin', 'tol')
    unary('acos 特殊值', [2.0,-2.0,0.5], np.arccos, 'vt-acos', 'tol')
    unary('atan 特殊值', [0.0,-0.0,1.0,I,F], np.arctan, 'vt-atan', 'tol')
    unary('atanh 特殊值', [1.0,-1.0,2.0,0.5], np.arctanh, 'vt-atanh', 'tol')
    unary('acosh 特殊值', [0.5,1.0,2.0], np.arccosh, 'vt-acosh', 'tol')
    unary('sinh 特殊值', [0.0,1.0,-1.0,1000.0], np.sinh, 'vt-sinh', 'tol')
    unary('cosh 特殊值', [0.0,1.0,-1.0,1000.0], np.cosh, 'vt-cosh', 'tol')
    unary('tanh 特殊值', [0.0,1.0,-1.0], np.tanh, 'vt-tanh', 'tol')
    unary('sinc 特殊值', [0.0,0.5,1.0], np.sinc, 'vt-sinc', 'tol')
    unary('deg2rad', [180.0,-90.0,45.0], np.deg2rad, 'vt-deg2rad', 'tol')
    unary('rad2deg', [3.141592653589793,-1.5707963267948966], np.rad2deg, 'vt-rad2deg', 'tol')
    # 二元
    def binop(name, xs, ys, nf, lfn, policy='exact'):
        a0,l0 = OP(('c',(len(xs),),'float64'), 1, data=np.array(xs))
        a1,l1 = OP(('c',(len(ys),),'float64'), 2, data=np.array(ys))
        add(name, f"({lfn} {l0} {l1})", (lambda a0=a0,a1=a1,nf=nf: nf(a0,a1)), policy)
    binop('hypot 1e300/Inf/NaN', [1e300,I,3.0], [1e300,F,4.0], np.hypot, 'vt-hypot', 'tol')
    binop('atan2 特殊值', [0.0,1.0,-0.0,-1.0], [1.0,1.0,1.0,1.0], np.arctan2, 'vt-atan2', 'tol')
    binop('expt 浮点', [-8.0,2.0,0.0,-2.0], [1/3,0.5,-1.0,2.0], np.power, 'vt-expt', 'tol')
    binop('maximum NaN传播', [F,1.0,3.0], [2.0,F,1.0], np.maximum, 'vt-maximum')
    binop('minimum NaN传播', [F,1.0,3.0], [2.0,F,1.0], np.minimum, 'vt-minimum')
    binop('fmax 忽略NaN', [F,1.0,3.0], [2.0,F,1.0], np.fmax, 'vt-fmax')
    binop('fmin 忽略NaN', [F,1.0,3.0], [2.0,F,1.0], np.fmin, 'vt-fmin')
    binop('mod 浮点符号', [-7.5,7.5,-3.0,3.0], [3.0,-3.0,-2.0,2.0], np.remainder, 'vt-mod', 'tol')
    binop('rem/fmod 浮点符号', [-7.5,7.5,-3.0,3.0], [3.0,-3.0,-2.0,2.0], np.fmod, 'vt-rem', 'tol')
    # lerp
    a0,l0 = OP(('c',(2,),'float64'), 1, data=np.array([0.0,2.0]))
    a1,l1 = OP(('c',(2,),'float64'), 2, data=np.array([10.0,4.0]))
    add('lerp 0,10 w=.25 / 2,4 w=.5', f"(vt-lerp {l0} {l1} 0.25d0)",
        (lambda a0=a0,a1=a1: a0 + 0.25*(a1-a0)), 'tol')

# ---------- F5 astype ----------
def f5():
    vals = [float('nan'), float('inf'), float('-inf'), 3.7, -3.7, 1e300]
    a0,l0 = OP(('c',(6,),'float64'), 1, data=np.array(vals))
    # NaN/Inf/1e300→int: NumPy 未定义，按 CONVENTIONS §7.2 期望 0
    for dt in ['int8','int32','int64','uint8']:
        doc = np.array([0,0,0,3,-3,0], dtype=np.int16).astype(dt)
        add(f"astype f64→{dt} NaN/Inf→0(§7.2)", f"(vt-astype {l0} {CLD[dt]})",
            (lambda doc=doc: doc))
    a32 = np.array(vals, dtype=np.float32)
    add('astype f64→f32', f"(vt-astype {l0} :float32)", (lambda a0=a0: a0.astype(np.float32)))
    iv = [200,-56,127,128,255,-1]
    b0,lb = OP(('c',(7,),'int64'), 2, data=np.array(iv))
    add('astype i64→i8 溢出截断', f"(vt-astype {lb} :int8)", (lambda b0=b0: b0.astype(np.int8)))
    add('astype i64→u8', f"(vt-astype {lb} :uint8)", (lambda b0=b0: b0.astype(np.uint8)))
    c0,lc = OP(('c',(2,),'uint8'), 3, data=np.array([255,128],dtype=np.uint8))
    add('astype u8→i8', f"(vt-astype {lc} :int8)", (lambda c0=c0: c0.astype(np.int8)))

# ---------- F6 空数组归约 ----------
def f6():
    for dt in ['int64','float64']:
        z,lz = f"(vt-zeros '(0) :dtype {CLD[dt]})", None
        z = lz = f"(vt-zeros '(0) :dtype {CLD[dt]})"
        za = np.zeros(0, dtype=dt)
        ops = [('sum',np.sum,'vt-sum'),('prod',np.prod,'vt-prod'),('mean',np.mean,'vt-mean'),
               ('var',np.var,'vt-var'),('std',np.std,'vt-std'),('amax',np.max,'vt-amax'),
               ('amin',np.min,'vt-amin'),('argmax',np.argmax,'vt-argmax'),('argmin',np.argmin,'vt-argmin'),
               ('all',np.all,'vt-all'),('any',np.any,'vt-any'),('median',np.median,'vt-median'),
               ('ptp',np.ptp,'vt-ptp'),('cumsum',np.cumsum,'vt-cumsum'),('cumprod',np.cumprod,'vt-cumprod')]
        for name, nf, lf in ops:
            pol = 'tol' if dt=='float64' and name in ('mean','var','std') else 'exact'
            add(f"空数组 {name} {dt}", f"({lf} {z})", (lambda za=za,nf=nf: nf(za)), pol)
        add(f"空数组 percentile {dt}", f"(vt-percentile {z} 50)", (lambda za=za: np.percentile(za,50)))

# ---------- F7 NaN 归约 ----------
def f7():
    vals = [float('nan'),1.0,3.0]
    a0,l0 = OP(('c',(3,),'float64'), 1, data=np.array(vals))
    full = [('sum',np.sum,'vt-sum'),('mean',np.mean,'vt-mean'),('median',np.median,'vt-median'),
            ('ptp',np.ptp,'vt-ptp'),('amax',np.max,'vt-amax'),('amin',np.min,'vt-amin'),
            ('argmax',np.argmax,'vt-argmax'),('argmin',np.argmin,'vt-argmin'),
            ('cumsum',np.cumsum,'vt-cumsum'),('nansum',np.nansum,'vt-nansum'),
            ('nanmean',np.nanmean,'vt-nanmean'),('nanvar',np.nanvar,'vt-nanvar'),
            ('nanstd',np.nanstd,'vt-nanstd'),('nanmax',np.nanmax,'vt-nanmax'),
            ('nanmin',np.nanmin,'vt-nanmin')]
    for name, nf, lf in full:
        pol = 'tol' if name in ('sum','mean','nansum','nanmean','nanvar','nanstd','cumsum','median') else 'exact'
        add(f"NaN传播 {name}", f"({lf} {l0})", (lambda a0=a0,nf=nf: nf(a0)), pol)
    add('NaN sort', f"(vt-sort {l0})", (lambda a0=a0: np.sort(a0)))
    add('NaN argsort', f"(vt-argsort {l0})", (lambda a0=a0: np.argsort(a0)))
    b0,lb = OP(('c',(2,),'float64'), 2, data=np.array([float('nan'),float('nan')]))
    add('全NaN nansum', f"(vt-nansum {lb})", (lambda b0=b0: np.nansum(b0)))
    add('全NaN nanmean', f"(vt-nanmean {lb})", (lambda b0=b0: np.nanmean(b0)))
    add('全NaN nanmax', f"(vt-nanmax {lb})", (lambda b0=b0: np.nanmax(b0)))

# ---------- F8 percentile 模式 ----------
def f8():
    a0,l0 = OP(('c',(4,),'float64'), 1, data=np.array([1.0,2.0,3.0,4.0]))
    for m in ['linear','lower','higher','midpoint','nearest']:
        add(f"percentile 25 {m}", f"(vt-percentile {l0} 25 :interpolation :{m})",
            (lambda a0=a0,m=m: np.percentile(a0,25,method=m)), 'tol')
        add(f"quantile .25 {m}", f"(vt-quantile {l0} 0.25d0 :interpolation :{m})",
            (lambda a0=a0,m=m: np.quantile(a0,0.25,method=m)), 'tol')
    add('percentile 边界0/100', f"(vt-percentile {l0} 0)",
        (lambda a0=a0: np.percentile(a0,0)), 'tol')

# ---------- F9 索引与查询 ----------
def f9():
    v,l = OP(('c',(5,),'int64'), 1, data=np.array([10,20,30,40,50]))
    ivt = lambda vals: f"(vt-from-sequence '({' '.join(map(str,vals))}) :dtype :int64)"
    ivt_arr = lambda vals: np.array(vals, dtype=np.int64)
    add('take 负索引', f"(vt-take {l} {ivt([-1,-3])})", (lambda v=v: np.take(v,[-1,-3])))
    add('take 越界', f"(vt-take {v[1] if False else l} {ivt([2,99])})", (lambda v=v: np.take(v,[2,99])))
    m,lm = OP(('c',(2,3),'int64'), 2, data=np.array([[1,2,3],[4,5,6]]))
    add('take axis=1', f"(vt-take {lm} {ivt([0,2])} :axis 1)", (lambda m=m: np.take(m,[0,2],axis=1)))
    add('put wrap', f"(vt-put (vt-zeros '(4) :dtype :int64) {ivt([5,-1])} 7 :mode :wrap)",
        (lambda: (lambda a: (np.put(a,[5,-1],7,mode='wrap'), a)[1])(np.zeros(4,dtype=np.int64))))
    add('put clip', f"(vt-put (vt-zeros '(4) :dtype :int64) {ivt([5,-1])} 7 :mode :clip)",
        (lambda: (lambda a: (np.put(a,[5,-1],7,mode='clip'), a)[1])(np.zeros(4,dtype=np.int64))))
    add('put 列表值', f"(vt-put (vt-zeros '(3) :dtype :int64) {ivt([0,2])} {ivt([7,8])})",
        (lambda: (lambda a: (np.put(a,[0,2],[7,8]), a)[1])(np.zeros(3,dtype=np.int64))))
    c0,lc0 = OP(('c',(2,),'int64'), 3, data=np.array([1,2]))
    c1,lc1 = OP(('c',(2,),'int64'), 4, data=np.array([10,20]))
    idx,lidx = OP(('c',(2,),'int64'), 5, data=np.array([1,0]))
    add('choose', f"(vt-choose (list {lc0} {lc1}) {lidx})", (lambda c0=c0,c1=c1,idx=idx: np.choose(idx,[c0,c1])))
    add('choose wrap', f"(vt-choose (list {lc0} {lc1}) {ivt([2,-1])} :mode :wrap)",
        (lambda c0=c0,c1=c1: np.choose(ivt_arr([2,-1]),[c0,c1],mode='wrap')))
    add('choose clip', f"(vt-choose (list {lc0} {lc1}) {ivt([5,-5])} :mode :clip)",
        (lambda c0=c0,c1=c1: np.choose(ivt_arr([5,-5]),[c0,c1],mode='clip')))
    s,ls = OP(('c',(4,),'float64'), 6, data=np.array([10.0,20.0,30.0,40.0]))
    sv,lsv = OP(('c',(6,),'float64'), 7, data=np.array([5.0,10.0,20.0,25.0,30.0,35.0]))
    add('searchsorted left', f"(vt-searchsorted {ls} {lsv})", (lambda s=s,sv=sv: np.searchsorted(s,sv)))
    add('searchsorted right', f"(vt-searchsorted {ls} {lsv} :side :right)", (lambda s=s,sv=sv: np.searchsorted(s,sv,side='right')))
    b,lb = OP(('c',(5,),'float64'), 8, data=np.array([0.0,1.0,2.5,4.0,10.0]))
    x,lx = OP(('c',(4,),'float64'), 9, data=np.array([0.2,6.4,3.0,1.6]))
    add('digitize left', f"(vt-digitize {lx} {lb})", (lambda x=x,b=b: np.digitize(x,b)))
    add('digitize right', f"(vt-digitize {lx} {lb} :right t)", (lambda x=x,b=b: np.digitize(x,b,right=True)))
    add('bincount 负数', f"(vt-bincount {ivt([-1,0,1])})", (lambda: np.bincount(np.array([-1,0,1]))))
    add('bincount minlength', f"(vt-bincount {ivt([0,0,2])} :minlength 5)", (lambda: np.bincount(np.array([0,0,2]),minlength=5)))
    v3,l3 = OP(('c',(4,),'int64'), 10, data=np.array([1,2,3,4]))
    add('insert 重复位置', f"(vt-insert {l3} 1 {ivt([10,20])})", (lambda v3=v3: np.insert(v3,1,[10,20])))
    add('insert 负位置', f"(vt-insert {l3} -1 99)", (lambda v3=v3: np.insert(v3,-1,99)))
    add('insert 越界', f"(vt-insert {l3} 9 99)", (lambda v3=v3: np.insert(v3,9,99)))
    add('insert 多位置', f"(vt-insert {l3} {ivt([0,2])} {ivt([7,8])})", (lambda v3=v3: np.insert(v3,[0,2],[7,8])))
    add('delete 列表', f"(vt-delete {l3} {ivt([0,2])})", (lambda v3=v3: np.delete(v3,[0,2])))
    add('delete 负', f"(vt-delete {l3} -1)", (lambda v3=v3: np.delete(v3,-1)))
    m2,lm2 = OP(('c',(2,3),'int64'), 11, data=np.array([[1,2,3],[4,5,6]]))
    m3,lm3 = OP(('c',(1,3),'int64'), 12, data=np.array([[7,8,9]]))
    add('append axis=0', f"(vt-append {lm2} {lm3} :axis 0)", (lambda m2=m2,m3=m3: np.append(m2,m3,axis=0)))
    add('append 展平', f"(vt-append {lm2} {lm3})", (lambda m2=m2,m3=m3: np.append(m2,m3)))
    v5,l5 = OP(('c',(6,),'int64'), 13, data=np.array([0,1,2,3,4,5]))
    add('split 段数不整除', f"(vt-split {l5} 3)", (lambda v5=v5: np.array_split(v5,3)))
    add('split 索引列表', f"(vt-split {l5} '(1 3))", (lambda v5=v5: np.array_split(v5,[1,3])))
    m4,lm4 = OP(('c',(3,4),'int64'), 14, data=np.arange(12,dtype=np.int64).reshape(3,4))
    add('vsplit 不整除', f"(vt-vsplit {lm4} 2)", (lambda m4=m4: np.vsplit(m4,2)))
    add('split axis=1', f"(vt-split {lm4} 2 :axis 1)", (lambda m4=m4: np.array_split(m4,2,axis=1)))
    # where 三参广播
    cond,lcond = OP(('c',(2,1),'int8'), 15, data=np.array([[1],[0]],dtype=np.int8))
    xv,lxv = OP(('c',(3,),'float64'), 16, data=np.array([1.0,2.0,3.0]))
    add('where 三参广播', f"(vt-where {lcond} {lxv} 7.5d0)",
        (lambda cond=cond,xv=xv: np.where(cond,xv,7.5)))
    add('argwhere', f"(vt-argwhere {lcond})", (lambda cond=cond: np.argwhere(cond)))
    add('extract', f"(vt-extract {lcond} {lxv})", (lambda cond=cond,xv=xv: np.extract(cond,xv)))
    ch1,lch1 = OP(('c',(2,3),'int64'), 17, data=np.array([[1,2,3],[4,5,6]]))
    ch2,lch2 = OP(('c',(2,3),'int64'), 18, data=np.array([[10,20,30],[40,50,60]]))
    cidx,lcidx = OP(('c',(2,3),'int64'), 19, data=np.array([[0,1,1],[1,0,0]]))
    add('select 优先级', f"(vt-select (list {lcond} {lcidx}) (list {lxv} {lch1}) :default -1)",
        (lambda cond=cond,cidx=cidx,xv=xv,ch1=ch1: np.select([cond.astype(bool),cidx.astype(bool)],[xv,ch1],default=-1)))
    # ref / slice
    add('ref 负索引', f"(vt-ref {lm2} -1 -1)", (lambda m2=m2: m2[-1,-1]))
    add('ref 越界', f"(vt-ref {lm2} 5 0)", (lambda m2=m2: m2[5,0]))
    add('ref 标量', f"(vt-ref {lm2} 1 2)", (lambda m2=m2: m2[1,2]))
    add('slice 负起点(索引语义)', f"(vt-slice {l5} '(-2))", (lambda v5=v5: v5[-2]))
    add('slice 步长2', f"(vt-slice {l5} '(1 6 2))", (lambda v5=v5: v5[1:6:2]))
    add('slice 负步长', f"(vt-slice {l5} '(4 0 -1))", (lambda v5=v5: v5[4:0:-1]))
    add('slice 2d 双spec', f"(vt-slice {lm4} '(1) '(1 4))", (lambda m4=m4: m4[1,1:4]))

# ---------- F10 形状操作 ----------
def f10():
    m34,lm34 = OP(('c',(3,4),'int64'), 20, data=np.arange(12,dtype=np.int64).reshape(3,4))
    add('reshape 非连续', f"(vt-reshape (vt-transpose {lm34}) '(2 6))",
        (lambda m34=m34: m34.T.reshape(2,6)))
    add('rot90 k=2', f"(vt-rot90 {lm34} :k 2)", (lambda m34=m34: np.rot90(m34,2)))
    add('rot90 k=-1', f"(vt-rot90 {lm34} :k -1)", (lambda m34=m34: np.rot90(m34,-1)))
    add('rot90 k=3', f"(vt-rot90 {lm34} :k 3)", (lambda m34=m34: np.rot90(m34,3)))
    m2,lm2 = OP(('c',(2,3,4),'int64'), 21, data=np.arange(24,dtype=np.int64).reshape(2,3,4))
    add('rot90 3d axes(1,2)', f"(vt-rot90 {lm2} :k 1 :axes '(1 2))", (lambda m2=m2: np.rot90(m2,1,(1,2))))
    add('roll axis=1 负', f"(vt-roll {lm34} -1 :axis 1)", (lambda m34=m34: np.roll(m34,-1,axis=1)))
    add('roll axis=0', f"(vt-roll {lm34} 2 :axis 0)", (lambda m34=m34: np.roll(m34,2,axis=0)))
    add('flip axis=1', f"(vt-flip {lm34} :axis 1)", (lambda m34=m34: np.flip(m34,1)))
    add('flip 全轴', f"(vt-flip {lm34} :axis nil)", (lambda m34=m34: np.flip(m34)))
    add('narrow', f"(vt-narrow {lm34} 1 1 3)", (lambda m34=m34: m34[:,1:3]))
    t2,lt2 = OP(('c',(2,),'int64'), 22, data=np.array([1,2]))
    t3,lt3 = OP(('c',(2,),'int64'), 23, data=np.array([3,4]))
    add('tile 标量', f"(vt-tile {lt2} 2)", (lambda t2=t2: np.tile(t2,2)))
    add('tile 列表(2,1)', f"(vt-tile {lt2} '(2 1))", (lambda t2=t2: np.tile(t2,(2,1))))
    add('tile 列表(1,2)', f"(vt-tile {lt2} '(1 2))", (lambda t2=t2: np.tile(t2,(1,2))))
    add('repeat axis=0 1d', f"(vt-repeat {lt2} 2 :axis 0)", (lambda t2=t2: np.repeat(t2,2,axis=0)))
    add('repeat 展平 2d', f"(vt-repeat {lm34} 2)", (lambda m34=m34: np.repeat(m34,2)))
    add('repeat axis=1', f"(vt-repeat {lm34} 2 :axis 1)", (lambda m34=m34: np.repeat(m34,2,axis=1)))
    fv,lfv = OP(('c',(3,),'float64'), 24, data=np.array([1.0,2.0,3.0]))
    add('pad 常量9', f"(vt-pad {lfv} 1 :constant-values 9.0d0)", (lambda fv=fv: np.pad(fv,1,constant_values=9)))
    add('pad 宽2', f"(vt-pad {lfv} 2)", (lambda fv=fv: np.pad(fv,2)))
    m2b,l = OP(('c',(2,2),'int64'), 26, data=np.array([[5,6],[7,8]]))
    add('stack axis=0', f"(vt-stack 0 {l} {l})", (lambda m2b=m2b: np.stack([m2b,m2b],0)))
    add('stack axis=1', f"(vt-stack 1 {l} {l})", (lambda m2b=m2b: np.stack([m2b,m2b],1)))
    add('stack axis=-1', f"(vt-stack -1 {l} {l})", (lambda m2b=m2b: np.stack([m2b,m2b],-1)))
    add('concatenate axis=1', f"(vt-concatenate 1 {l} {l})", (lambda m2b=m2b: np.concatenate([m2b,m2b],1)))
    add('vstack', f"(vt-vstack {l} {l})", (lambda m2b=m2b: np.vstack([m2b,m2b])))
    add('hstack', f"(vt-hstack {l} {l})", (lambda m2b=m2b: np.hstack([m2b,m2b])))
    add('squeeze 全部', f"(vt-squeeze (vt-reshape {lt2} '(1 2 1)))", (lambda t2=t2: np.squeeze(t2.reshape(1,2,1))))
    add('squeeze 指定轴', f"(vt-squeeze (vt-reshape {lt2} '(1 2 1)) :axis 2)", (lambda t2=t2: np.squeeze(t2.reshape(1,2,1),axis=2)))
    add('squeeze 非1轴报错', f"(vt-squeeze {l} :axis 0)", (lambda m2b=m2b: np.squeeze(m2b,axis=0)))
    add('unsqueeze axis=1', f"(vt-unsqueeze {lt2} 1)", (lambda t2=t2: np.expand_dims(t2,1)))
    add('unsqueeze axis=-1', f"(vt-unsqueeze {lt2} -1)", (lambda t2=t2: np.expand_dims(t2,-1)))
    add('unsqueeze 越界', f"(vt-unsqueeze {lt2} 3)", (lambda t2=t2: np.expand_dims(t2,3)))
    add('broadcast-to 失败', f"(vt-broadcast-to {lt2} '(3 2))", (lambda t2=t2: np.broadcast_to(t2,(3,2))))
    add('broadcast-shapes 标量', f"(vt-broadcast-shapes nil '(3))", (lambda: np.broadcast_shapes((),(3,))))
    add('broadcast-shapes 冲突', f"(vt-broadcast-shapes '(3) '(4))", (lambda: np.broadcast_shapes((3,),(4,))))
    add('flatten-to-nested', f"(vt-flatten-to-nested '(2 2) (make-array 4 :initial-contents '(1 2 3 4)))",
        (lambda: np.array([1,2,3,4]).tolist()))
    add('swapaxes', f"(vt-swapaxes {lm2} 0 2)", (lambda m2=m2: np.swapaxes(m2,0,2)))
    add('transpose 3d perm', f"(vt-transpose {lm2} '(2 0 1))", (lambda m2=m2: m2.transpose(2,0,1)))
    add('diag 1d→矩阵', f"(vt-diag {lt2})", (lambda t2=t2: np.diag(t2)))
    add('diag 矩阵→对角线', f"(vt-diag {l})", (lambda m2b=m2b: np.diag(m2b)))
    add('eye k=1', f"(vt-eye 3 :k 1 :dtype :int64)", (lambda: np.eye(3,k=1,dtype=np.int64)))
    add('eye 2x4', f"(vt-eye 2 :cols 4)", (lambda: np.eye(2,4)))
    add('eye value=7', f"(vt-eye 2 :value 7)", (lambda: np.eye(2)*7))
    add('triu k=1', f"(vt-triu {l} :k 1)", (lambda m2b=m2b: np.triu(m2b,1)))
    add('tril k=-1', f"(vt-tril {l} :k -1)", (lambda m2b=m2b: np.tril(m2b,-1)))
    add('diagonal offset=1', f"(vt-diagonal {l} :offset 1)", (lambda m2b=m2b: np.diagonal(m2b,1)))

# ---------- F11 :out 与别名 ----------
def f11():
    a,la = OP(('c',(2,3),'float64'), 30, data=np.array([[1.0,2.0,3.0],[4.0,5.0,6.0]]))
    add('out dtype不匹配(契约H3必须报错)',
        f"(vt-add {la} {la} :out (vt-zeros '(2 3) :dtype :float32))", None, expect='err')
    add('out stride0只读(契约H4必须报错)',
        f"(vt-add {la} {la} :out (vt-broadcast-to (vt-zeros '(3)) '(2 3)))", None, expect='err')
    base = np.zeros((2,6))
    lbase = f"(let ((base (vt-zeros '(2 6)))) (vt-add {la} {la} :out (vt-slice base '(:all) '(0 6 2))) base)"
    add('out 非连续写入(契约H6)', lbase,
        (lambda a=a: (lambda b: (np.add(a,a,out=b[:,0:6:2]), b)[1])(np.zeros((2,6)))))
    add('out 别名快照', f"(let ((a {la})) (vt-add a a :out a) a)",
        (lambda a=a: (lambda x: (np.add(x,x,out=x), x)[1])(a.copy())))
    af,lf = OP(('c',(2,),'float64'), 31, data=np.array([1.5,2.5]))
    add('dtype= 结果astype语义', f"(vt-add {lf} {lf} :dtype :int32)",
        (lambda af=af: np.add(af,af).astype(np.int32)))
    add('写穿透切片视图', f"(let ((a (vt-from-sequence '(1 2 3) :dtype :int64))) (vt-put (vt-slice a '(1 3)) 0 99) a)",
        (lambda: (lambda x: (np.put(x[1:3],0,99), x)[1])(np.array([1,2,3]))))
    add('写广播视图报错', f"(vt-put (vt-broadcast-to (vt-from-sequence '(1 2)) '(2 2)) 0 9)",
        (lambda: (lambda x: (np.put(np.broadcast_to(x,(2,2)),0,9), np.broadcast_to(x,(2,2)))[1])(np.array([1,2]))))
    add('copy 独立性', f"(let* ((a (vt-from-sequence '(1 2) :dtype :int64)) (c (vt-copy a))) (vt-put c 0 99) a)",
        (lambda: (lambda x,c: (np.put(c,0,99), x)[1])(np.array([1,2]), np.array([1,2]).copy())))

# ---------- F12 非连续输入 ----------
def f12():
    dat = np.array([[1.0,float('nan'),3.0,4.0],[5.0,6.0,7.0,8.0],[9.0,10.0,11.0,float('nan')]])
    T,lT = OP(('t',(3,4),'float64'), 40, data=dat)
    add('转置视图 add', f"(vt-+ {lT} {lT})", (lambda T=T: T+T))
    add('转置视图 sum axis0', f"(vt-sum {lT} :axis 0)", (lambda T=T: T.sum(0)), 'tol')
    add('转置视图 sum 全', f"(vt-sum {lT})", (lambda T=T: T.sum()), 'tol')
    add('转置视图 amax', f"(vt-amax {lT})", (lambda T=T: T.max()))
    add('转置视图 argmax', f"(vt-argmax {lT})", (lambda T=T: T.argmax()))
    add('转置视图 clip', f"(vt-clip {lT} 2.0d0 8.0d0)", (lambda T=T: np.clip(T,2.0,8.0)))
    add('转置视图 isnan', f"(vt-isnan {lT})", (lambda T=T: np.isnan(T)))
    add('转置视图 sort axis0', f"(vt-sort {lT} :axis 0)", (lambda T=T: np.sort(T,0)))
    add('转置视图 cumsum axis0', f"(vt-cumsum {lT} :axis 0)", (lambda T=T: np.cumsum(T,0)), 'tol')
    add('转置视图 mean axis1', f"(vt-mean {lT} :axis 1)", (lambda T=T: T.mean(1)), 'tol')
    add('双重转置还原', f"(vt-transpose {lT})", (lambda T=T: T.T))
    add('转置视图 scale', f"(vt-scale {lT} 2.0d0)", (lambda T=T: T*2))
    B, lB = OP(('b',(1,4),'float64',(3,4)), 41, data=dat[1:2,:])
    add('广播视图 + 转置视图', f"(vt-+ {lB} {lT})", (lambda B=B,T=T: B+T))
    add('广播视图 sum axis0', f"(vt-sum {lB} :axis 0)", (lambda B=B: B.sum(0)), 'tol')
    add('广播视图 amax', f"(vt-amax {lB})", (lambda B=B: B.max()))
    dat64 = np.arange(12,dtype=np.int64).reshape(3,4)
    S,lS = OP(('s',(3,4),'int64',[(0,4,2)]), 42, data=dat64)
    add('切片视图 add', f"(vt-+ {lS} {lS})", (lambda S=S: S+S))
    add('切片视图 floordiv', f"(vt-div {lS} 2)",
        (lambda S=S: S//np.array(2)))
    add('切片视图 mod', f"(vt-mod {lS} 3)",
        (lambda S=S: S % np.array(3)))
    add('切片视图 argmax axis1', f"(vt-argmax {lS} :axis 1)", (lambda S=S: S.argmax(1)))
    add('切片视图 astype f64', f"(vt-astype {lS} :float64)", (lambda S=S: S.astype(np.float64)))
    add('切片视图 take', f"(vt-take {lS} (vt-from-sequence '(0 5) :dtype :int64))",
        (lambda S=S: np.take(S,[0,5])))
    add('0维+矩阵广播', f"(vt-+ (vt-const nil 2.5d0) {lT})", (lambda T=T: T+np.float64(2.5)))
    m4,lm4 = OP(('c',(4,),'int64'), 43, data=np.array([3,1,4,1]))
    add('argmax axis=? 越界报错', f"(vt-argmax {lm4} :axis 5)", (lambda m4=m4: m4.argmax(axis=5)))
    add('sum axis 越界报错', f"(vt-sum {lm4} :axis 2)", (lambda m4=m4: m4.sum(axis=2)))

# ---------- F13 零长度广播 D13 ----------
def f13():
    add('(3,0)+(1,4) 报错', f"(vt-+ (vt-zeros '(3 0)) (vt-zeros '(1 4)))",
        (lambda: np.add(np.zeros((3,0)), np.zeros((1,4)))))
    add('(0,3)+(1,3)', f"(vt-+ (vt-zeros '(0 3)) (vt-zeros '(1 3)))",
        (lambda: np.add(np.zeros((0,3)), np.zeros((1,3)))))
    add('sum (0,3) axis0', f"(vt-sum (vt-zeros '(0 3)) :axis 0)",
        (lambda: np.zeros((0,3)).sum(0)))
    add('sum (3,0) axis1', f"(vt-sum (vt-zeros '(3 0)) :axis 1)",
        (lambda: np.zeros((3,0)).sum(1)))
    add('amax (3,0) axis1 报错', f"(vt-amax (vt-zeros '(3 0)) :axis 1)",
        (lambda: np.zeros((3,0)).max(1)))
    add('sum 空一维', f"(vt-sum (vt-zeros '(0)))", (lambda: np.zeros(0).sum()))

# ---------- F14 isclose/allclose ----------
def f14():
    F,I = float('nan'), float('inf')
    a0,l0 = OP(('c',(3,),'float64'), 50, data=np.array([I,I,1.0]))
    b0,l1 = OP(('c',(3,),'float64'), 51, data=np.array([I,1.0,1.0000001]))
    add('isclose inf/inf', f"(vt-isclose {l0} {l1})", (lambda a0=a0,b0=b0: np.isclose(a0,b0)))
    n0,ln = OP(('c',(1,),'float64'), 52, data=np.array([F]))
    add('isclose nan/nan→F', f"(vt-isclose {ln} {ln})", (lambda n0=n0: np.isclose(n0,n0)))
    add('allclose nan/nan→T(equal_nan)', f"(vt-allclose {ln} {ln})", (lambda n0=n0: np.allclose(n0,n0)))
    c0,lc = OP(('c',(1,),'float64'), 53, data=np.array([1.000000001]))
    d0,ld = OP(('c',(1,),'float64'), 54, data=np.array([1.0]))
    add('isclose 自定义rtol', f"(vt-isclose {lc} {ld} :rtol 1.0d-9 :atol 0.0d0)",
        (lambda c0=c0,d0=d0: np.isclose(c0,d0,rtol=1e-9,atol=0.0)))
    e0,le = OP(('c',(3,),'float64'), 55, data=np.array([1000.0,1001.0,0.5]))
    f0,lf2 = OP(('c',(3,),'float64'), 56, data=np.array([1001.0,1000.0,0.7]))
    add('isclose 广播值', f"(vt-isclose {le} {lf2})", (lambda e0=e0,f0=f0: np.isclose(e0,f0)))

# ---------- F15 创建 ----------
def f15():
    add('arange 浮点step累积', "(vt-arange 8 :start 0.0d0 :step 0.1d0)",
        (lambda: np.arange(0.0, 0.8, 0.1)), 'exact')
    add('arange 整数', "(vt-arange 4 :start 5 :step 2 :dtype :int64)",
        (lambda: np.arange(5, 13, 2)), 'exact')
    add('arange 0元素(§9.1例外:缺省float64)', "(vt-arange 0)", (lambda: np.arange(0, dtype=np.float64)))
    add('linspace 7点', "(vt-linspace 0.0d0 1.0d0 7)", (lambda: np.linspace(0,1,7)), 'tol')
    add('linspace num=1', "(vt-linspace 2.0d0 9.0d0 1)", (lambda: np.linspace(2,9,1)), 'tol')
    add('linspace endpoint=f', "(vt-linspace 0.0d0 1.0d0 5 :endpoint nil)",
        (lambda: np.linspace(0,1,5,endpoint=False)), 'tol')
    add('logspace base=2', "(vt-logspace 0 3 4 :base 2.0d0)", (lambda: np.logspace(0,3,4,base=2.0)), 'tol')

# ---------- F16 逻辑与谓词 ----------
def f16():
    a0,l0 = OP(('c',(5,),'int64'), 60, data=np.array([-2,-1,0,1,2]))
    for name, lf, nf in [('logical-and','vt-logical-and',np.logical_and),
                         ('logical-or','vt-logical-or',np.logical_or),
                         ('logical-xor','vt-logical-xor',np.logical_xor)]:
        add(f"{name} 整数真值", f"({lf} {l0} {l0})", (lambda a0=a0,nf=nf: nf(a0,a0)))
    add('logical-not', f"(vt-logical-not {l0})", (lambda a0=a0: np.logical_not(a0)))
    fv,lfv = OP(('c',(7,),'float64'), 61, data=np.array([2.0,3.0,2.5,-4.0,float('nan'),float('inf'),0.0]))
    add('even-p 浮点/NaN/Inf', f"(vt-even-p {lfv})",
        (lambda fv=fv: np.equal(np.mod(fv,2),0)))
    add('odd-p 浮点/NaN/Inf', f"(vt-odd-p {lfv})",
        (lambda fv=fv: np.equal(np.mod(fv,2),1)))
    add('zero-p 含-0.0', f"(vt-zero-p (vt-from-sequence '(-1.0d0 0.0d0 0.5d0 -0.0d0)))",
        (lambda: np.equal(np.array([-1.0,0.0,0.5,-0.0]),0)))
    add('positive-p', f"(vt-positive-p {lfv})", (lambda fv=fv: np.greater(fv,0)))

# ---------- F17 cumsum提升/keepdims/clip ----------
def f17():
    add('cumsum int8→int64', f"(vt-cumsum (vt-from-sequence '(100 100) :dtype :int8))",
        (lambda: np.cumsum(np.array([100,100],dtype=np.int8))))
    add('prod int8→int64', f"(vt-prod (vt-from-sequence '(3 4) :dtype :int8))",
        (lambda: np.prod(np.array([3,4],dtype=np.int8))))
    m,lm = OP(('c',(2,3),'float64'), 62, data=np.array([[1.0,2.0,3.0],[4.0,5.0,6.0]]))
    add('sum keepdims axis0', f"(vt-sum {lm} :axis 0 :keepdims t)", (lambda m=m: m.sum(0,keepdims=True)), 'tol')
    add('all keepdims axis1', f"(vt-all (vt-ones '(2 3)) :axis 1 :keepdims t)",
        (lambda: np.ones((2,3)).all(1,keepdims=True)))
    add('amax keepdims', f"(vt-amax {lm} :axis 1 :keepdims t)", (lambda m=m: m.max(1,keepdims=True)))
    add('clip min>max 顺序', f"(vt-clip (vt-const '(1) 5.0d0) 3.0d0 1.0d0)",
        (lambda: np.clip(np.array([5.0]),3,1)))
    add('clip NaN', f"(vt-clip (vt-const '(1) +nan+) 0.0d0 1.0d0)",
        (lambda: np.clip(np.array([float('nan')]),0,1)))
    add('4元加法两两结合', f"(vt-+ (vt-const '(1) 1.0d0) (vt-const '(1) 2.0d0) (vt-const '(1) +nan+) (vt-const '(1) 4.0d0))",
        (lambda: np.add(np.add(np.array([1.0]),np.array([2.0])), np.add(np.array([float('nan')]),np.array([4.0])))))

# ---------- F18 0维与杂项 ----------
def f18():
    add('0d sum', f"(vt-sum (vt-const nil 2.5d0))", (lambda: np.asarray(np.sum(np.array(2.5)))), 'tol')
    add('0d ref', f"(vt-ref (vt-const nil 7.5d0))", (lambda: np.asarray(7.5)))
    add('0d astype', f"(vt-astype (vt-const nil 2.7d0) :int32)", (lambda: np.asarray(np.int32(np.array(2.7).astype(np.int32)))))
    add('normalize-axis -1', f"(vt-normalize-axis -1 2)", (lambda: 1))
    add('normalize-axis 越界', f"(vt-normalize-axis 2 2)", None, expect='err')
    add('itemsize f32', f"(vt-itemsize (vt-zeros '(1) :dtype :float32))", (lambda: np.asarray(4)))
    add('contiguous 转置拷贝', f"(vt-to-list (vt-contiguous (vt-transpose (vt-reshape (vt-arange 6 :dtype :int64) '(2 3)))))",
        (lambda: np.arange(6,dtype=np.int64).reshape(2,3).T.copy().reshape(-1)))
    add('item 1元素2d', f"(vt-item (vt-reshape (vt-from-sequence '(9) :dtype :int64) '(1 1)))", (lambda: np.asarray(9)))

# 生成 lisp 探针文件与期望
def cmd_gen():
    f1(); f2(); f3(); f4(); f5(); f6(); f7(); f8(); f9(); f10(); f11(); f12(); f13(); f14(); f15(); f16(); f17(); f18()
    prelude = ''';;; 自动生成差分探针 — clvt vs numpy
(require :asdf)
;;; ---- 定位 clvt 仓库根目录（含 clvt.asd）：CLVT_HOME → 本脚本目录 → 当前目录 ----
(defun %probe-clvt-dir (dir)
  (and dir (probe-file (merge-pathnames "clvt.asd" dir)) dir))  ; 命中时返回目录本身
(defun %find-clvt-root ()
  (let ((env (uiop:getenv "CLVT_HOME")))
    (or (and env (plusp (length env))
             (let ((d (uiop:parse-unix-namestring env :ensure-directory t)))
               (%probe-clvt-dir d)))
        (let ((d (and *load-pathname*
                      (uiop:pathname-directory-pathname *load-pathname*))))
          (or (%probe-clvt-dir d)
              (and d (%probe-clvt-dir (uiop:pathname-parent-directory-pathname d)))))
        (%probe-clvt-dir *default-pathname-defaults*))))
(let ((root (%find-clvt-root)))
  (unless root
    (error "未找到 clvt/clvt.asd：请设置环境变量 CLVT_HOME=clvt 仓库根目录，或将本探针文件放入 clvt 仓库根目录（与 clvt.asd 同级）后运行。"))
  (pushnew root asdf:*central-registry* :test #'equal))
(handler-bind ((warning #'muffle-warning)) (asdf:load-system :clvt))
(defpackage :fz (:use :cl :clvt))
(in-package :fz)
(defparameter +inf+ sb-ext:double-float-positive-infinity)
(defparameter +ninf+ sb-ext:double-float-negative-infinity)
(defparameter +nan+ (sb-int:with-float-traps-masked (:invalid :divide-by-zero) (/ 0.0d0 0.0d0)))
(defun fz-flat (x) (cond ((null x) nil) ((vt-p x) (fz-flat (vt-to-list x))) ((atom x) (list x)) (t (append (fz-flat (car x)) (fz-flat (cdr x))))))
(defun fz-token (x)
  (typecase x
    (single-float (fz-token (coerce x 'double-float)))
    (double-float (cond ((sb-ext:float-nan-p x) "nan")
                        ((and (sb-ext:float-infinity-p x) (plusp x)) "inf")
                        ((sb-ext:float-infinity-p x) "-inf")
                        (t (format nil "~,16E" x))))
    (integer (format nil "~D" x))
    (ratio (fz-token (coerce x 'double-float)))
    (t (format nil "~A" x))))
(defvar *fz-expected* nil)  ; 期望表由生成器填入：(cid kind dtype shape toks policy)
(defparameter *fz-id* "?")
(defparameter *fz-n-pass* 0)
(defparameter *fz-n-fail* 0)
(defparameter *fz-fails* nil)
(defparameter *fz-cats* (list (cons :clvt-err 0) (cons :numpy-err-ok 0) (cons :err-not-raised 0)
                              (cons :dtype 0) (cons :shape 0) (cons :len 0) (cons :data 0) (cons :missing 0)))

(defun fz-emit (k p) (format t "~a|~a|~a~%" *fz-id* k p))

(defun fz-split-kind (token)
  (let ((p (position #\\| token)))
    (if p (values (subseq token 0 p) (subseq token (1+ p)))
        (values token ""))))

(defun fz-field (payload key)
  "取 key=value 的 value 部分（值不含空格），key 自动补 = 号"
  (let* ((kv (concatenate 'string key "="))
         (p (search kv payload)))
    (when p
      (let* ((s (+ p (length kv)))
             (e (position #\\space payload :start s)))
        (subseq payload s (or e (length payload)))))))

(defun fz-shape-str (payload)
  (let ((p (search "shape=" payload)))
    (when p
      (let ((s (+ p 6)) e)
        (if (and (< s (length payload)) (char= (char payload s) #\\())
            (progn (setf e (position #\\) payload :start s))
                   (when e (subseq payload s (1+ e))))
            (progn (setf e (position #\\space payload :start s))
                   (subseq payload s (or e (length payload)))))))))

(defun fz-data-str (payload)
  (let ((p (search "data=[" payload)))
    (when p
      (let ((s (+ p 6))
            (e (position #\\] payload :start p)))
        (when e (subseq payload s e))))))

(defun fz-split-chars (s ch)
  (loop with start = 0
        for i from 0 to (length s)
        when (or (= i (length s)) (char= (char s i) ch))
          collect (subseq s start i)
          and do (setf start (1+ i))))

(defun fz-toks (s)
  (when (and s (plusp (length s)))
    (remove "" (mapcar (lambda (x) (string-trim " " x)) (fz-split-chars s #\\space))
            :test #'string=)))

(defun fz-shape-toks (s)
  (when (and s (plusp (length s)))
    (let ((inner (string-trim "()" s)))
      (when (plusp (length inner))
        (mapcar #'parse-integer (fz-toks inner))))))

(defun fz-num (s)
  "token → number / :nan / :inf / :ninf（float 一律按双精度读入）"
  (cond ((string= s "nan") :nan)
        ((string= s "inf") :inf)
        ((string= s "-inf") :ninf)
        ((find #\\d s :test #'char-equal) (read-from-string s))
        ((find #\\e s :test #'char-equal)
         (read-from-string (nsubstitute #\\d #\\e (copy-seq s) :test #'char-equal)))
        ((find #\\. s) (read-from-string (concatenate 'string s "d0")))
        (t (parse-integer s))))

(defun fz-bit= (a b)
  (flet ((f (x) (coerce x 'double-float)))
    (cond ((and (eq a :nan) (eq b :nan)) t)
          ((or (eq a :nan) (eq b :nan)) nil)
          ((and (eq a :inf) (eq b :inf)) t)
          ((and (eq a :ninf) (eq b :ninf)) t)
          ((or (eq a :inf) (eq a :ninf) (eq b :inf) (eq b :ninf)) nil)
          ((and (integerp a) (integerp b)) (= a b))
          (t (let ((fa (f a)) (fb (f b)))
               (if (and (zerop fa) (zerop fb))
                   (= (float-sign fa) (float-sign fb))
                   (= fa fb)))))))

(defun fz-tol= (a b &optional (rt 1d-11) (at 1d-13))
  (flet ((f (x) (coerce x 'double-float)))
    (cond ((and (eq a :nan) (eq b :nan)) t)
          ((or (eq a :nan) (eq b :nan)) nil)
          ((or (eq a :inf) (eq a :ninf) (eq b :inf) (eq b :ninf)) (eq a b))
          (t (let ((fa (f a)) (fb (f b)))
               (<= (abs (- fa fb)) (+ at (* rt (max (abs fa) (abs fb))))))))))

(defun fz-check (rec act)
  "rec=(cid kind dtype shape toks policy)，act=完整 token → (values ok 类别)"
  (destructuring-bind (cid ekind edt esh eda epol) rec
    (declare (ignore cid))
    (multiple-value-bind (akind ap) (fz-split-kind act)
      (cond
        ((string= ekind "MUST-ERR")
         (if (string= akind "ERR") (values t nil) (values nil :err-not-raised)))
        ((string= ekind "ERR")
         (if (string= akind "ERR") (values t nil) (values nil :numpy-err-ok)))
        ((string= akind "ERR") (values nil :clvt-err))
        (t
         (let ((adt (fz-field ap "dtype"))
               (ash (fz-shape-str ap))
               (ada (fz-toks (fz-data-str ap))))
           (cond
             ;; kind 兼容：SCALAR 与 0 维 VT 等价；含 LIST 跳过类型检查
             ((and (not (or (string= ekind "LIST") (string= akind "LIST")))
                   (not (and (string= ekind "VT") (string= esh "()")))
                   (not (string= ekind akind)))
              (values nil :dtype))
             ((and (string= ekind "VT") (string= akind "VT") edt adt
                   (not (string= esh "()")) (not (string= edt adt)))
              (values nil :dtype))
             ((and (not (or (string= ekind "LIST") (string= akind "LIST")))
                   (not (equal (fz-shape-toks esh) (fz-shape-toks ash))))
              (values nil :shape))
             ((/= (length eda) (length ada)) (values nil :len))
             (t
              ;; 容差按 dtype 适配：float32 的 1-ULP 相对差 ~6e-8
              (let* ((rt (if (search "float32" adt) 1d-5 1d-11))
                     (bad (loop for x in eda for y in ada
                               unless (if (eq epol :tol)
                                          (fz-tol= (fz-num x) (fz-num y) rt)
                                          (fz-bit= (fz-num x) (fz-num y)))
                               collect x)))
                (if bad (values nil :data) (values t nil)))))))))))

(defun fz-clip (s) (if (<= (length s) 110) s (concatenate 'string (subseq s 0 110) "…")))

(defun fz-result (token)
  (format t "~a|~a~%" *fz-id* token)
  (let ((rec (assoc *fz-id* *fz-expected* :test #'string=)))
    (if rec
        (multiple-value-bind (ok cat) (fz-check rec token)
          (if ok
              (incf *fz-n-pass*)
              (progn (incf *fz-n-fail*)
                     (incf (cdr (assoc cat *fz-cats*)))
                     (push (list *fz-id* cat
                                 (fz-clip (format nil "~{~a~^ | ~}" (cddr rec)))
                                 (fz-clip token))
                           *fz-fails*))))
        (progn (incf *fz-n-fail*)
               (incf (cdr (assoc :missing *fz-cats*)))
               (push (list *fz-id* :missing "" (fz-clip token)) *fz-fails*)))))

(defmacro probe (id form)
  `(let ((*fz-id* ,id))
     (handler-case
         (let ((r (progn ,form)))
           (typecase r
             (vt (let ((sh (vt-shape r)))
                   (if sh
                       (fz-result (format nil "VT|dtype=~(~s~) shape=(~{~a~^ ~}) data=[~{~a~^ ~}]"
                                          (vt-dtype r) sh (mapcar #'fz-token (fz-flat (vt-to-list r)))))
                       (fz-result (format nil "VT|dtype=~(~s~) shape=() data=[~a]"
                                          (vt-dtype r) (fz-token (vt-item r)))))))
             (symbol (fz-result (format nil "SCALAR|data=[~a]" (if r 1 0))))
             (number (fz-result (format nil "SCALAR|data=[~a]" (fz-token r))))
             (list (fz-result (format nil "LIST|data=[~{~a~^ ~}]" (mapcar #'fz-token (fz-flat r)))))
             (t (fz-result (format nil "OTHER|~s" r)))))
       (error (e) (fz-result (format nil "ERR|msg=~a" e))))))

(defun fz-summary ()
  (format t "~%======================================================================~%")
  (format t "差分探针汇总: PASS=~d  FAIL=~d  总计=~d~%"
          *fz-n-pass* *fz-n-fail* (+ *fz-n-pass* *fz-n-fail*))
  (format t "失败分类: ~{~(~a~)=~d~^  ~}~%"
          (mapcan (lambda (c) (list (car c) (cdr c))) *fz-cats*))
  (when *fz-fails*
    (format t "~%失败用例:~%")
    (dolist (f (reverse *fz-fails*))
      (format t "  ❌ ~a [~(~a~)]~%      期望: ~a~%      实际: ~a~%"
              (first f) (second f) (third f) (fourth f))))
  (format t "结论: ~a~%"
          (if (zerop *fz-n-fail*) "全部通过 ✔" "存在不匹配 ✘（退出码 1）")))
'''
    pass  # prelude 已内建可移植的 clvt 根目录探测
    os.makedirs(TMP, exist_ok=True)
    # ---------- 期望统一计算：expected.txt 与嵌入探针的期望表共用 ----------
    exps = {}
    for cid, desc, lisp, npfn, policy, expect in CASES:
        if expect == 'err':
            exps[cid] = ('MUST-ERR', None, None, None); continue
        try:
            with np.errstate(all='ignore'):
                r = npfn()
            # split/vsplit 返回「数组列表」：clvt LIST 侧按展平数据比较，
            # numpy 侧同样展平拼接（异形数组 asarray 会抛 inhomogeneous）
            if (isinstance(r, (list, tuple)) and r
                    and all(isinstance(x, np.ndarray) for x in r)):
                r = np.concatenate([np.asarray(x).ravel() for x in r]) \
                    if r else np.array([], dtype=np.float64)
            else:
                r = np.asarray(r)
            if r.dtype == np.bool_: r = r.astype(np.int8)
            if r.ndim == 0:
                exps[cid] = ('VT', CLD[str(r.dtype)], '()', [tok1(r[()])])
            else:
                exps[cid] = ('VT', CLD[str(r.dtype)],
                             '(' + ' '.join(map(str, r.shape)) + ')',
                             [tok1(x) for x in r.reshape(-1)])
        except Exception:
            exps[cid] = ('ERR', None, None, None)
    with open(PROBE_OUT,'w') as f:
        f.write(prelude)
        # 期望表内嵌 → 探针运行时即时判定 PASS/FAIL 并汇总
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
    with open(f'{TMP}/expected.txt','w') as f:
        for cid, desc, lisp, npfn, policy, expect in CASES:
            k, dt, sh, toks = exps[cid]
            if k == 'MUST-ERR':
                f.write(f"{cid}|MUST-ERR|\n")
            elif k == 'ERR':
                f.write(f"{cid}|ERR|\n")
            else:
                f.write(f"{cid}|VT|dtype={dt} shape={sh} data=[{' '.join(toks)}]\n")
    meta = [{'id':c[0],'desc':c[1],'lisp':c[2],'policy':c[4],'expect':c[5]} for c in CASES]
    json.dump(meta, open(f'{TMP}/cases.json','w'), ensure_ascii=False)
    print(f"生成 {len(CASES)} 个探针 → {PROBE_OUT}, {TMP}/expected.txt")

def tok1(v):
    v = float(v)
    if math.isnan(v): return 'nan'
    if math.isinf(v): return 'inf' if v>0 else '-inf'
    return repr(float(v))

# ---------- 比对 ----------
def parse_line(kind, payload):
    if kind in ('ERR','MUST-ERR','MISSING'): return (kind, None, None, None)
    d = {}
    # data=[...] 含空格，须整体截取；修复：原按空格 split 导致 data 只保留首元素，
    # 比对实际只覆盖每个用例的第一个元素（此前"全绿"覆盖不足）
    di = payload.find('data=[')
    data = ''
    head = payload
    if di >= 0:
        sj = payload.find(']', di)
        data = payload[di+6:sj] if sj >= 0 else payload[di+6:]
        head = payload[:di]
    for part in head.split(' '):
        if '=' in part:
            k,_,v = part.partition('='); d[k]=v
    dt = d.get('dtype'); sh = d.get('shape','()')
    shape = tuple(int(x) for x in sh.strip('()').split()) if sh.strip('()') else ()
    toks = data.split() if data else []
    return (kind, dt, shape, toks)

def val(t):
    t = t.strip()
    if t == 'nan': return float('nan')
    if t == 'inf': return float('inf')
    if t == '-inf': return float('-inf')
    if 'd' in t or 'D' in t: t = t.replace('d','e').replace('D','e')  # Lisp d-指数
    if '.' in t or 'e' in t: return float(t)
    return int(t)
    if t in ('nan','inf','-inf'): return float(t)
    if t is None: return None
    try:
        if ('.' in t) or ('e' in t): return float(t)
        return int(t)
    except ValueError:
        return t

def bit_eq(a,b):
    if isinstance(a,int) and isinstance(b,int): return a==b
    fa,fb = float(a),float(b)
    if math.isnan(fa) and math.isnan(fb): return True
    if math.isinf(fa) or math.isinf(fb): return fa==fb
    if fa==0 and fb==0: return math.copysign(1,fa)==math.copysign(1,fb)
    return fa==fb

def tol_eq(a,b,rt=1e-11,at=1e-13):
    fa,fb = float(a),float(b)
    if math.isnan(fa) and math.isnan(fb): return True
    if math.isinf(fa) or math.isinf(fb): return fa==fb
    return abs(fa-fb) <= at + rt*max(abs(fa),abs(fb))

def cmd_compare():
    exp = {}
    for line in open(f'{TMP}/expected.txt'):
        p = line.rstrip('\n').split('|',2)
        exp[p[0]] = parse_line(p[1], p[2] if len(p)>2 else '')
    act = {}
    for line in open(f'{TMP}/actual.txt'):
        p = line.rstrip('\n').split('|',2)
        if not (len(p)==3 and p[0].startswith('c') and p[0][1:].isdigit()): continue
        act[p[0]] = parse_line(p[1], p[2] if len(p)>2 else '')
    meta = {c['id']: c for c in json.load(open(f'{TMP}/cases.json'))}
    fails = {'clvt-err':[], 'numpy-err-clvt-ok':[], 'doc-err-not-raised':[],
             'dtype':[], 'shape':[], 'len':[], 'data-exact':[], 'data-tol':[], 'missing':[]}
    for cid in sorted(exp):
        ek,edt,esh,eda = exp[cid]; ak,adt,ash,ada = act.get(cid,('MISSING',None,None,None))
        m = meta[cid]; desc = m['desc']; lisp = m['lisp']; pol = m['policy']
        if ak == 'MISSING':
            fails['missing'].append((cid,desc,lisp,'','')); continue
        if ek=='MUST-ERR':
            if ak!='ERR': fails['doc-err-not-raised'].append((cid,desc,lisp,'clvt未报错: '+str(ada)[:10],''))
            continue
        if ek=='ERR':
            if ak!='ERR': fails['numpy-err-clvt-ok'].append((cid,desc,lisp,'clvt值='+str(ada)[:12],''))
            continue
        if ak=='ERR':
            fails['clvt-err'].append((cid,desc,lisp,'numpy='+' '.join(eda or [])[:12],'')); continue
        # SCALAR 与 0维VT 等价
        if ek=='SCALAR' and ak=='VT' and ash==(): pass
        elif ak=='SCALAR' and ek=='VT' and esh==(): pass
        elif ek=='LIST' or ak=='LIST':
            pass
        elif ek!=ak:
            fails['dtype'].append((cid,desc,lisp,f'kind {ek}',f'kind {ak}')); continue
        if edt and adt and edt!=adt and ek=='VT' and ak=='VT' and esh!=():
            fails['dtype'].append((cid,desc,lisp,edt,adt))
        if ek!='LIST' and ak!='LIST' and esh!=ash:
            fails['shape'].append((cid,desc,lisp,str(esh),str(ash))); continue
        if len(eda or [])!=len(ada or []):
            fails['len'].append((cid,desc,lisp,f'n={len(eda or [])}',f'n={len(ada or [])}')); continue
        bad = []
        # 容差按 dtype 适配：float32 的 1-ULP 相对差 ~6e-8，rt=1e-11 对其过严
        rt = 1e-5 if adt in (':float32', 'float32') else 1e-11
        for i,(x,y) in enumerate(zip(eda or [], ada or [])):
            vx,vy = val(x), val(y)
            ok = bit_eq(vx,vy) if pol=='exact' else tol_eq(vx,vy,rt)
            if not ok: bad.append((i,x,y))
        if bad:
            key = 'data-exact' if pol=='exact' else 'data-tol'
            fails[key].append((cid,desc,lisp,'','; '.join(f"[{i}]{x}≠{y}" for i,x,y in bad[:5])))
    total = len(exp)
    nfail = sum(len(v) for v in fails.values())
    print(f"{'='*70}\n差分比对: 探针={total}  不匹配={nfail}\n{'='*70}")
    for k,items in fails.items():
        if not items: continue
        print(f"\n### {k} ({len(items)})")
        for cid,desc,lisp,es,as_ in items[:60]:
            print(f"  [{cid}] {desc}")
            print(f"      lisp: {lisp[:150]}")
            if es: print(f"      期望: {es}")
            if as_: print(f"      实际: {as_}")
    print(f"\n{'='*70}\n各分类: " + ", ".join(f"{k}={len(v)}" for k,v in fails.items()))

if __name__ == '__main__':
    if sys.argv[1] == 'gen': cmd_gen()
    else: cmd_compare()
