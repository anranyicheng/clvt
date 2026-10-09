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

用法: python3 test/check_key_sync.py            （由 run-tests.sh 自动调用）
      python3 test/check_key_sync.py --probe-only   （只查 codegen 同步）
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


def ref_keys():
    out = subprocess.run([sys.executable, str(REF_SCRIPT)],
                         capture_output=True, text=True, timeout=600, env=_ENV)
    if out.returncode != 0:
        sys.exit(f"[key-sync] FAIL: ref_compute.py 运行失败:\n{out.stderr[-800:]}")
    return set(json.loads(out.stdout)), out.stdout


def main():
    probe_only = "--probe-only" in sys.argv
    errors, warns = [], []

    # --- 检查 3: 探针 codegen 与提交物一致（先做，避免污染工作区状态）---
    for gen, lisp in CODEGEN:
        committed = (HERE / lisp).read_text()
        subprocess.run([sys.executable, str(HERE / gen)],
                       capture_output=True, text=True, cwd=HERE.parent, timeout=600, env=_ENV)
        regenerated = (HERE / lisp).read_text()
        if committed != regenerated:
            errors.append(
                f"codegen 漂移: {gen} 重新生成的 {lisp} 与仓库中提交的版本不一致\n"
                f"  → 有人手改了生成物而未更新生成器，或反之。\n"
                f"  → 修复: python3 test/{gen} 后 git add {lisp}；或还原手改。")
        # 还原工作区到提交状态，避免检查本身弄脏 git status
        subprocess.run(["git", "checkout", "-q", "--", str(HERE / lisp)],
                       capture_output=True, text=True, cwd=HERE.parent)

    if probe_only:
        if errors:
            print("\n".join(errors)); sys.exit(1)
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
        sys.exit(1)
    print(f"[key-sync] OK: ref_compute {len(keys)} key，5 套件查询 {len(used)}，缺失 0")
    for w in warns:
        print("[key-sync] " + w)
    sys.exit(0)


if __name__ == "__main__":
    main()
