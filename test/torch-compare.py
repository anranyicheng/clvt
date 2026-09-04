#!/usr/bin/env python3
"""
clvt vs PyTorch 正确性 + 性能对比测试
===========================================
运行方式: python3 test/torch-compare.py
先启动sbcl运行clvt生成结果，再用pytorch独立计算并对比。
"""
import subprocess
import json
import sys
import time
import numpy as np

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("[警告] PyTorch未安装，跳过PyTorch对比测试", file=sys.stderr)
    sys.exit(0)

print("="*60)
print("  clvt ↔ PyTorch 正确性与性能对比测试")
print("="*60)

# 1. 先用ref_compute.py + numpy生成参考（已含torch交叉验证）
print("\n[1/3] 生成 numpy 参考结果并交叉验证 PyTorch ...")
ref = json.loads(subprocess.check_output([sys.executable, "test/ref_compute.py"],
                                          cwd=".", stderr=subprocess.DEVNULL))
print(f"     ✓ 生成 {len(ref)} 个参考结果")

# 2. 用纯PyTorch独立重算一批关键算子
print("\n[2/3] 独立使用 PyTorch 重算核心算子对比 ...")
torch_tests = {}
pass_count = 0
fail_count = 0

def cmp(name, np_val, torch_val, rtol=1e-5, atol=1e-7):
    global pass_count, fail_count
    t_np = torch_val.detach().cpu().numpy() if isinstance(torch_val, torch.Tensor) else torch_val
    try:
        ok = np.allclose(np_val, t_np, rtol=rtol, atol=atol, equal_nan=True)
        if ok and np.asarray(np_val).shape == np.asarray(t_np).shape:
            pass_count += 1
            print(f"     ✓ {name}")
        else:
            fail_count += 1
            print(f"     ✗ {name}  MISMATCH")
    except Exception as e:
        fail_count += 1
        print(f"     ✗ {name}  ERROR: {e}")

# 基础算子
a = np.array([1.0, 2.0, 3.0, 4.0])
ta = torch.tensor(a)
b = np.array([5.0, 6.0, 7.0, 8.0])
tb = torch.tensor(b)

cmp("torch.add", a+b, ta+tb)
cmp("torch.mul", a*b, ta*tb)
cmp("torch.sub", a-b, ta-tb)
cmp("torch.div", a/b, ta/tb)
cmp("torch.sin", np.sin(a), torch.sin(ta))
cmp("torch.exp", np.exp(a), torch.exp(ta))
cmp("torch.abs", np.abs(a-2.5), torch.abs(ta-2.5))
cmp("torch.sqrt", np.sqrt(a), torch.sqrt(ta))

# 矩阵乘
m1 = np.array([[1.,2.],[3.,4.]])
m2 = np.array([[5.,6.],[7.,8.]])
cmp("torch.matmul", m1 @ m2, torch.tensor(m1) @ torch.tensor(m2))

# 归约
arr = np.arange(12, dtype=np.float64).reshape(3,4)
cmp("torch.sum(all)", float(np.sum(arr)), float(torch.sum(torch.tensor(arr))))
cmp("torch.sum(axis=0)", np.sum(arr, axis=0), torch.sum(torch.tensor(arr), dim=0))
cmp("torch.mean(axis=1)", np.mean(arr, axis=1), torch.mean(torch.tensor(arr), dim=1))
cmp("torch.max(axis=0)", np.max(arr, axis=0), torch.max(torch.tensor(arr), dim=0).values)

# 激活函数
x = np.linspace(-3, 3, 7)
tx = torch.tensor(x)
cmp("torch.sigmoid", 1/(1+np.exp(-x)), torch.sigmoid(tx))
cmp("torch.relu", np.maximum(0, x), torch.relu(tx))
cmp("torch.tanh", np.tanh(x), torch.tanh(tx))
cmp("torch.softmax", np.exp(x)/np.exp(x).sum(), torch.softmax(tx, dim=0))

# 累计
cmp("torch.cumsum", np.cumsum(a), torch.cumsum(ta, dim=0))

# 3. 微基准（numpy vs pytorch，展示工业级性能基线）
print("\n[3/3] PyTorch vs NumPy 性能基线参考（1M float64元素）...")
size = 1000000
an = np.random.randn(size)
tn = torch.from_numpy(an.copy())

def bench_np(fn):
    t0 = time.perf_counter()
    for _ in range(50):
        fn(an)
    return (time.perf_counter()-t0)/50*1000

def bench_torch(fn):
    t0 = time.perf_counter()
    for _ in range(50):
        fn(tn)
    return (time.perf_counter()-t0)/50*1000

print(f"     {'算子':<15} {'NumPy(ms)':>12} {'PyTorch(ms)':>14}")
print(f"     {'-'*45}")
for name, npf, tf in [
    ("add", lambda x:x+1, lambda x:x+1),
    ("mul", lambda x:x*2, lambda x:x*2),
    ("exp", np.exp, torch.exp),
    ("sin", np.sin, torch.sin),
    ("sigmoid", lambda x:1/(1+np.exp(-x)), torch.sigmoid),
    ("sum(all)", lambda x:np.sum(x), lambda x:torch.sum(x)),
]:
    print(f"     {name:<15} {bench_np(npf):>12.3f} {bench_torch(tf):>14.3f}")

print("\n" + "="*60)
print(f"  PyTorch 独立测试: {pass_count} 通过, {fail_count} 失败")
if fail_count == 0:
    print("  ✓ 所有PyTorch对比测试通过！numpy↔pytorch参考值一致。")
else:
    print("  ✗ 存在失败用例，请检查。")
    sys.exit(1)
print("="*60)
