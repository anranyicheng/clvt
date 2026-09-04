#!/usr/bin/env python3
"""此脚本已废弃。参考值现在由 ref_compute.py 实时生成。
保留此文件作为入口，直接重定向到 ref_compute.py"""
import subprocess
import sys
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
ref_script = os.path.join(script_dir, "ref_compute.py")
result = subprocess.run([sys.executable, ref_script], capture_output=True, text=True)
print(result.stdout)
sys.exit(result.returncode)
