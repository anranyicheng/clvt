#!/usr/bin/env python3
"""check_key_sync.py — 测试工具链防漂移门禁（静态检查，秒级，无 numpy 依赖）

背景: test/ 下 5 个 lisp 套件运行时通过 (E "key") / (funcall E "key") / (E J "key")
向 ref_compute.py 请求 numpy 参考值。ref_compute.py 的 key 与 lisp 的查询 key
之间只靠命名约定同步——历史上已发生两次漂移（gen_all_tests.py / gen_param_tests.py
两个旧架构生成器的 key 约定与现行枢纽不一致，误导维护者）。

本脚本机械化检查三类漂移，任何一类都直接退出码 1:
  1. lisp 查询了 ref_compute 没有的 key        → 漂移（会以 "exp: NIL" 失败，报因晦涩）
  2. ref_compute 供给但无人查询的 key          → 死供给（提示清理）
  3. 探针 codegen 与提交的 .lisp 不一致         → 有人手改了生成物

关于第 3 类：**"文本不一致" 不等于 "语义漂移"**。
`gen_probes*.py` 用 `str(np.float64)` 输出期望值，而该格式随 numpy 版本变
（同一数值 2.4.6 给 `0.5294751290000002`、2.5.0 给 `0.529475129`，只差 1 ulp，
比较时都走 `:tol`）。若把「文本逐字节相同」当门禁，换一台机器/换一个 numpy
就必然变红 —— 而这是**正常演进**，按 CONVENTIONS §10.4.1 的裁决：
先分辨「格式化差异」还是「真语义变化」，不要盲目重生成再提交。

因此默认口径是**语义比较**：
  · key 集合、dtype、shape、比较方式（:exact/:tol）必须完全一致；
  · 数值字段按 **float 解析**后比较：完全相等 → 通过；相对差 ≤ 1e-12
    （纯格式化/最后一位）→ 记为「格式化漂移」的 WARN，不失败；
  · 其余（数值真变了、条目增删、结构不同）→ FAIL。
需要钉死文本时用 `--strict`（CI 锁定单一 numpy 版本时用它）。

用法: python3 test/check_key_sync.py            （由 run-tests.sh 自动调用）
      python3 test/check_key_sync.py --probe-only   （只查 codegen 同步）
      python3 test/check_key_sync.py --strict       （文本逐字节一致才通过）
"""
import json
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REF_SCRIPT = HERE / "ref_compute.py"

# (lisp 文件, key 提取正则) —— 覆盖 5 套件现存的全部查询写法
CONSUMERS = [
    ("run_all_tests.lisp",        r'(?:funcall E|\(E)\s+"([^"]+)"'),
    ("run_param_tests.lisp",      r'\(E J\s+"([^"]+)"'),
    ("numpy-compare-test.lisp",   r'\(E\s+"([^"]+)"'),
    ("auto-compare-test.lisp",    r'\(E\s+"([^"]+)"'),
    ("legacy-coverage-test.lisp", r'\(E J\s+"([^"]+)"'),
]

# 探针生成器与其提交产物（期望逐字节一致）
CODEGEN = [
    ("gen_probes.py",  "differential-probes-test.lisp"),
    ("gen_probes2.py", "differential-probes-test2.lisp"),
]


# 门禁产生的子进程一律不写字节码缓存（gen_probes2 import gen_probes 会触发
# CPython 为被 import 模块写 test/__pycache__，弄脏工作区）
_ENV = {**__import__("os").environ, "PYTHONDONTWRITEBYTECODE": "1"}

# 探针条目形态（gen_probes*.py 每行一条）：
#   (list "c0163" "VT" ":float64" "(6)" (list "v1" "v2" ...) :tol)
_PROBE_RE = re.compile(
    r'\(list\s+"([^"]+)"\s+"([^"]*)"\s+"([^"]*)"\s+"([^"]*)"\s+\(list\s+([^()]*)\)\s+:(\w+)\)')

# 纯格式化差异的容差（按声明的结果 dtype 分级，只吸收「str(float) 末位」）。
# 依据：本机实测跨 numpy 版本的最大漂移为 float64 2.1e-16（1 ulp）、
# float32 1.0e-7（<1 ulp）。任何真正的语义变化都远大于此。
_FMT_RTOL = {":float64": 1e-12, ":float32": 1e-6}
_FMT_RTOL_DEFAULT = 0.0            # 整型/布尔等：必须逐字符相同


def parse_probes(text):
    """把生成的探针文件解析成 {key: (kind, dtype, shape, values, mode)}。

    只认识 `(list "cNNNN" ...)` 条目，其余行（注释/辅助函数）忽略 —— 它们由
    「生成器是否崩溃」与「结构是否一致」另行保证。
    """
    out = {}
    for line in text.splitlines():
        m = _PROBE_RE.search(line)
        if not m:
            continue
        key, kind, dtype, shape, vals, mode = m.groups()
        values = re.findall(r'"([^"]*)"', vals)
        out[key] = (kind, dtype, shape, values, mode)
    return out


def _as_float(s):
    """数值字符串 → float；nan / ±inf 也能解析（CL 与 numpy 都这么打印）。"""
    try:
        return float(s)
    except ValueError:
        return None


def _same_number(a, b, rtol):
    """'4.291015625' vs '4.2910156250000001' → True（格式化）；'4.2' vs '4.3' → False。

    rtol = 0 表示要求逐字符相同（整型/布尔/字符串）。"""
    if a == b:
        return True
    fa, fb = _as_float(a), _as_float(b)
    if fa is None or fb is None or rtol <= 0:
        return False                        # 非数值，或该 dtype 不容许任何差异
    if fa != fa and fb != fb:               # 都是 nan
        return True
    if fa != fa or fb != fb:                # 只有一边 nan
        return False
    if fa in (float("inf"), float("-inf")) or fb in (float("inf"), float("-inf")):
        return False
    scale = max(abs(fa), abs(fb), 1e-300)
    return abs(fa - fb) <= rtol * scale


def diff_probes(committed_text, regenerated_text):
    """比较两份探针文本，返回 (errors, fmt_drift_keys)。"""
    old, new = parse_probes(committed_text), parse_probes(regenerated_text)
    errors, drift = [], []

    only_old = sorted(set(old) - set(new))
    only_new = sorted(set(new) - set(old))
    if only_old:
        errors.append(f"  生成器少了 {len(only_old)} 条探针: {', '.join(only_old[:8])}")
    if only_new:
        errors.append(f"  生成器多了 {len(only_new)} 条探针: {', '.join(only_new[:8])}")

    for key in sorted(set(old) & set(new)):
        o, n = old[key], new[key]
        if o[:3] != n[:3] or o[4] != n[4]:
            errors.append(f"  {key}: 结构变了 {o[:3]}+{o[4]} → {n[:3]}+{n[4]}")
            continue
        if len(o[3]) != len(n[3]):
            errors.append(f"  {key}: 元素个数 {len(o[3])} → {len(n[3])}")
            continue
        # :exact 条目断言的是「逐位相等」，文本一变就一定会运行期失败，
        # 所以只允许逐字符相同；:tol 条目才按 dtype 吸收末位差异。
        rtol = 0.0 if o[4] == "exact" else _FMT_RTOL.get(o[1], _FMT_RTOL_DEFAULT)
        bad = [(x, y) for x, y in zip(o[3], n[3]) if not _same_number(x, y, rtol)]
        if bad:
            x, y = bad[0]
            errors.append(f"  {key}: 数值真变了 {x} → {y}（{len(bad)} 个分量）")
            continue
        if o[3] != n[3]:                   # 数值等价但文本不同 = 纯格式化
            drift.append(f"{key}({o[1]})")
    return errors, drift


def check_codegen(gen, lisp, strict, errors, warns):
    """重新生成 → 与提交物比对 → 还原工作区。"""
    path = HERE / lisp
    committed = path.read_text()
    side = HERE.parent / "tmp"                 # 生成器顺带写出的中间产物目录
    side_existed = side.exists()
    try:
        proc = subprocess.run([sys.executable, str(HERE / gen)],
                              capture_output=True, text=True, cwd=HERE.parent,
                              timeout=600, env=_ENV)
        # 生成器崩溃时文件不会被重写，此时「文本相同」是假通过 —— 必须单独拦。
        if proc.returncode != 0:
            errors.append(
                f"codegen 执行失败: python3 test/{gen} 退出码 {proc.returncode}\n"
                f"  {proc.stderr.strip().splitlines()[-1:] or ['(无 stderr)']}\n"
                f"  → 生成器坏了比没有门禁更危险（文件不重写，门禁会永远通过）。")
            return
        regenerated = path.read_text()

        if committed == regenerated:
            return
        if strict:
            errors.append(
                f"codegen 漂移（--strict 文本比对）: {gen} → {lisp}\n"
                f"  → 修复: python3 test/{gen} 后 git add {lisp}；或还原手改。")
            return

        errs, drift = diff_probes(committed, regenerated)
        if errs:
            errors.append(
                f"codegen 语义漂移: {gen} → {lisp}（不是格式化差异）\n"
                + "\n".join(errs[:6])
                + f"\n  → 先确认是生成的期望值本身变了，再决定改生成器还是改实现。")
        elif drift:
            warns.append(
                f"codegen 格式化漂移: {gen} → {lisp} 共 {len(drift)} 条数值文本不同"
                f"（相对差 ≤ float64 1e-12 / float32 1e-6，比较走 :tol，不影响判定）\n"
                f"  例: {', '.join(drift[:5])}\n"
                f"  → 通常是 numpy 版本换了 str(float) 的格式化方式；"
                f"需要钉死文本时用 python3 test/check_key_sync.py --strict。")
    finally:
        # 生成器会顺带写根目录 tmp/（提交 60336c9 已把该目录从仓库删除）。
        # 门禁跑完必须把它清掉，否则每跑一次测试就多一个未跟踪目录。
        if not side_existed and side.exists():
            import shutil
            shutil.rmtree(side, ignore_errors=True)


def ref_keys():
    out = subprocess.run([sys.executable, str(REF_SCRIPT)],
                         capture_output=True, text=True, timeout=600, env=_ENV)
    if out.returncode != 0:
        sys.exit(f"[key-sync] FAIL: ref_compute.py 运行失败:\n{out.stderr[-800:]}")
    return set(json.loads(out.stdout)), out.stdout


def main():
    probe_only = "--probe-only" in sys.argv
    strict = "--strict" in sys.argv
    errors, warns = [], []

    # --- 检查 3: 探针 codegen 与提交物一致（先做，避免污染工作区状态）---
    for gen, lisp in CODEGEN:
        try:
            check_codegen(gen, lisp, strict, errors, warns)
        finally:
            # 还原工作区到提交状态，避免检查本身弄脏 git status
            subprocess.run(["git", "checkout", "-q", "--", str(HERE / lisp)],
                           capture_output=True, text=True, cwd=HERE.parent)

    if probe_only:
        if errors:
            print("[key-sync] FAIL")
            print("\n".join(errors)); sys.exit(1)
        for w in warns:
            print("[key-sync] " + w)
        print("[key-sync] codegen 同步 ✓"); sys.exit(0)

    # --- 检查 1/2: ref_compute 供给 vs lisp 查询 ---
    keys, _raw = ref_keys()
    used = set()
    for lisp, pat in CONSUMERS:
        text = (HERE / lisp).read_text()
        q = set(re.findall(pat, text))
        used |= q
        missing = sorted(q - keys)
        if missing:
            errors.append(
                f"{lisp} 查询了 ref_compute.py 未供给的 {len(missing)} 个 key:\n"
                f"  {', '.join(missing[:10])}\n"
                f"  → 运行时会以 exp: NIL 形式失败。请在 ref_compute.py 补充对应 R[\"...\"] = ...")
    dead = sorted(keys - used)
    if dead:
        warns.append(
            f"ref_compute.py 供给的 {len(dead)} 个 key 无任何套件查询（死供给，可清理）:\n"
            f"  {', '.join(dead[:10])}")

    if errors:
        print("[key-sync] FAIL")
        for e in errors:
            print("  " + e)
        for w in warns:
            print("[key-sync] " + w)
        sys.exit(1)
    print(f"[key-sync] OK: ref_compute {len(keys)} key，5 套件查询 {len(used)}，缺失 0")
    for w in warns:
        print("[key-sync] " + w)
    sys.exit(0)


if __name__ == "__main__":
    main()
