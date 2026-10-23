#!/usr/bin/env python3
"""check_docs.py — 文档/注释一致性门禁（静态，秒级，无第三方依赖）

背景：本库的文档与注释长期由「每轮追加一段」的方式增长，结果是
**同一事实散在四处、且互相矛盾**（例如 `vt-einsum` 已支持复数，但
`CONVENTIONS.md` 里三处仍写着「不支持」）。根因不是注释多，而是：

  1. 契约（现行语义）与编年史（第 N 轮改了什么）混在同一份文件里；
  2. 代码注释里带轮次编号，读者无法判断哪条是现状；
  3. 小节编号被反复插入，已经出现 §16.20 排在 §16.36 之后的情况。

本门禁把这三类问题变成**可机械检查**的断言：

  C1  代码注释里不得出现「第 N 轮」        → 编年史只允许在 HISTORY.md / CHANGELOG.md
  C1b CONVENTIONS 正文不得出现「本轮」     → 契约不谈时序，读者不知道「本轮」是哪一轮
  C2  `§x.y` 引用必须能解析到真实小节       → 防锚点漂移（改编号后忘了改引用）
  C3  CONVENTIONS 的 §16.x 三级编号必须递增 → 防再次插入错位
  C4  HISTORY.md 必须存在且被 CONVENTIONS 引用 → 防契约与编年史再次合流

用法：python3 test/check_docs.py      （由 run-tests.sh 自动调用）
"""

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CONV = ROOT / "CONVENTIONS.md"
HIST = ROOT / "HISTORY.md"
CODE_GLOBS = ["src/*.lisp", "test/*.lisp"]

ROUND_RE = re.compile(r"第[一二三四五六七八九十]+轮")
NOW_RE = re.compile(r"本轮|这一轮|上一轮|前几轮")
HEAD_RE = re.compile(r"^#{2,4}\s+(\d+(?:\.\d+)*)\b", re.M)
REF_RE = re.compile(r"§\s*(\d+\.\d+(?:\.\d+)?)")


def read(p):
    return p.read_text(encoding="utf-8")


def main():
    errors, warns = [], []
    conv = read(CONV) if CONV.exists() else ""

    # ---------- C1: 代码注释里不得有轮次编号 ----------
    hits = []
    for pat in CODE_GLOBS:
        for f in sorted(ROOT.glob(pat)):
            for i, line in enumerate(read(f).splitlines(), 1):
                if ROUND_RE.search(line) and line.lstrip().startswith(";"):
                    hits.append(f"{f.relative_to(ROOT)}:{i}: {line.strip()[:70]}")
    if hits:
        errors.append(
            "代码注释里出现「第 N 轮」（编年史应只写在 HISTORY.md / CHANGELOG.md，"
            "代码注释请改写成无时序的现状陈述）：\n    " + "\n    ".join(hits[:10])
            + (f"\n    …共 {len(hits)} 处" if len(hits) > 10 else ""))

    # ---------- C1b: CONVENTIONS 正文不得有「本轮」 ----------
    # 跳过门禁自身的说明行（§0 的规则表里必须写出这些词才能描述规则）
    now_hits = [f"CONVENTIONS.md:{i}: {l.strip()[:70]}"
                for i, l in enumerate(conv.splitlines(), 1)
                if NOW_RE.search(l) and "C1b" not in l]
    if now_hits:
        errors.append(
            "CONVENTIONS.md 出现「本轮 / 这一轮 / 上一轮 / 前几轮」"
            "（契约不写时序：改成「这里 / 这次改动」，或把过程移到 HISTORY.md）：\n    "
            + "\n    ".join(now_hits[:8])
            + (f"\n    …共 {len(now_hits)} 处" if len(now_hits) > 8 else ""))

    # ---------- C2: §x.y 引用必须可解析 ----------
    anchors = set(HEAD_RE.findall(conv))
    # 允许引用 HISTORY 的 H.x
    hist = read(HIST) if HIST.exists() else ""
    anchors |= set(re.findall(r"^#{2,4}\s+(H\.\d+)\b", hist, re.M))

    bad_refs = {}
    for pat in ["CONVENTIONS.md", "README.md"] + CODE_GLOBS:
        for f in sorted(ROOT.glob(pat)):
            for i, line in enumerate(read(f).splitlines(), 1):
                for ref in REF_RE.findall(line):
                    if ref not in anchors:
                        bad_refs.setdefault(ref, []).append(
                            f"{f.relative_to(ROOT)}:{i}")
    if bad_refs:
        detail = "\n".join(
            f"    §{ref} → {', '.join(v[:3])}" for ref, v in sorted(bad_refs.items()))
        errors.append("引用了不存在的 §编号（改编号后忘了改引用？）：\n" + detail)

    # ---------- C3: §16.x 三级编号递增 ----------
    nums = [int(m.group(1))
            for m in re.finditer(r"^###\s+16\.(\d+)\b", conv, re.M)]
    for prev, cur in zip(nums, nums[1:]):
        if cur <= prev:
            errors.append(
                f"CONVENTIONS.md 的 §16.x 编号非递增：… 16.{prev} 之后出现 16.{cur}"
                "（新小节请追加在末尾，不要插队）")
            break

    # ---------- C4: HISTORY.md 必须存在且被引用 ----------
    if not HIST.exists():
        errors.append("缺少 HISTORY.md（契约与编年史必须分家）")
    elif "HISTORY.md" not in conv:
        errors.append("CONVENTIONS.md 未引用 HISTORY.md（读者找不到编年史）")

    # ---------- 报告 ----------
    if errors:
        print("[docs] FAIL")
        for e in errors:
            print("  " + e)
        for w in warns:
            print("[docs] " + w)
        sys.exit(1)

    n_sections = len(anchors)
    print(f"[docs] OK: 代码注释无轮次编号；{n_sections} 个 §锚点全部可解析；"
          f"§16.x 递增；无时序用语；HISTORY.md 已接入")
    for w in warns:
        print("[docs] " + w)
    sys.exit(0)


if __name__ == "__main__":
    main()
