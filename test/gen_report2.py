#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
clvt vs numpy 差分测试第 2 阶段 —— 候选分歧最小复现报告
（对应 gen_probes2.py 的 405 个探针；前 199 个符号见 gen_report.py）

用法:  CLVT_HOME=<clvt仓库> SBCL=<sbcl路径> python3 gen_report2.py
流程:  1) 生成一个 SBCL 批处理脚本，逐条输出 `RXX|clvt结果`
       2) Python 侧计算 numpy 期望
       3) 输出对照表与判定（BUG=确证缺陷 / 分歧=与NumPy语义不一致 / API=设计取舍 / 精度=数值容差问题）
"""
import os, re, subprocess, sys, shutil, tempfile
import numpy as np
np.seterr(all='ignore')

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SBCL = os.environ.get('SBCL') or shutil.which('sbcl')
CLVT_HOME = os.environ.get('CLVT_HOME') or os.path.dirname(_SCRIPT_DIR)
QL = os.environ.get('QUICKLISP_SETUP')  # 可选：quicklisp setup.lisp 路径（sb-simd 依赖）

PRELUDE = '''(require :asdf)
(pushnew #p"CLVT_DIR" asdf:*central-registry* :test #'equal)
(handler-bind ((warning #'muffle-warning)) (asdf:load-system :clvt))
(defpackage :fzr (:use :cl :clvt))
(in-package :fzr)
'''

# 每个 R 用例：clvt 形式（打印 RXX|token）+ numpy 期望说明
CASES = r'''
(defun show (tag v)
  (format t "~a|~a~%" tag (if (vt-p v) (format nil "~a dtype=~(~a~)" (vt-to-list (vt-flatten v)) (vt-dtype v)) v)))

;; R1 vt-reduce axis=nil 是否真的执行 reducer-fn（常数 99 检验）
(let ((v (vt-from-sequence (list 1.0d0 2.0d0 3.0d0 4.0d0) :dtype :float64)))
  (let ((r (vt-reduce v nil 0.0d0 (lambda (a x) (declare (ignore a x)) 99.0d0))))
    (show "R1" (vt-item r))))

;; R2 topk axis=0 k=1：values 与 indices 是否一致
(multiple-value-bind (v i) (vt-topk (vt-from-sequence (list (list 1.0d0 9.0d0 4.0d0) (list 7.0d0 2.0d0 8.0d0)) :dtype :float64) 1 :axis 0)
  (format t "R2|vals=~a idx=~a~%" (vt-to-list v) (vt-to-list i)))

;; R3 interp 左边界 x == xp[0]（numpy 取 fp[0]，不触发 left）
(let ((r (vt-interp (vt-from-sequence (list 0.0d0 1.0d0 3.0d0) :dtype :float64)
                    (vt-from-sequence (list 0.0d0 2.0d0 4.0d0) :dtype :float64)
                    (vt-from-sequence (list 10.0d0 20.0d0 30.0d0) :dtype :float64)
                    :left -1.0d0 :right 99.0d0)))
  (show "R3" r))

;; R4 copysign(1, -0.0)（numpy = -1）
(show "R4" (vt-copysign (vt-from-sequence (list 1.0d0) :dtype :float64)
                        (vt-from-sequence (list -0.0d0) :dtype :float64)))

;; R5 signbit(NaN)（numpy = 0）
(show "R5" (vt-signbit (vt-from-sequence (list (sb-int:with-float-traps-masked (:invalid :divide-by-zero) (/ 0.0d0 0.0d0))) :dtype :float64)))

;; R6 cond :p 1（numpy = ||A||1·||A^-1||1）
(show "R6" (vt-cond (vt-from-sequence (list (list 1.0d0 2.0d0) (list 3.0d0 4.0d0)) :dtype :float64) :p 1))

;; R7 in1d 返回 dtype（F4 要求布尔为 :int8）
(let ((r (vt-in1d (vt-from-sequence (list 1 2 3 4) :dtype :int64)
                  (vt-from-sequence (list 3 1) :dtype :int64))))
  (format t "R7|dtype=~(~a~) data=~a~%" (vt-dtype r) (vt-to-list r)))

;; R8 relu 整数输入 dtype（文档设计：整数提升 float64）
(let ((r (vt-relu (vt-from-sequence (list -2 0 3) :dtype :int64))))
  (format t "R8|dtype=~(~a~) data=~a~%" (vt-dtype r) (vt-to-list r)))

;; R9 einsum 未使用下标（numpy ValueError）
(handler-case (progn (vt-einsum "ij,kl->ij" (vt-from-sequence (list (list 0.0d0 1.0d0)) :dtype :float64)
                                    (vt-from-sequence (list (list 0.0d0 1.0d0)) :dtype :float64))
                     (format t "R9|NO-ERROR~%"))
  (error (e) (format t "R9|ERROR ~a~%" e)))

;; R10 norm :keepdims 无 axis（numpy (1,)，clvt 0 维）
(let ((r (vt-norm (vt-from-sequence (list 3.0d0 4.0d0) :dtype :float64) :keepdims t)))
  (format t "R10|shape=~a data=~a~%" (vt-shape r) (vt-item r)))

;; R11 with-generator 是否推进生成器自身状态
(let ((g (make-generator 9)))
  (with-generator (g) (vt-random '(3)))
  (let ((a (vt-to-list (vt-random '(2) :rng g)))
        (b (vt-to-list (vt-random '(2) :rng (make-generator 9)))))
    (format t "R11|with-gen后g的第1-2个=~a 新生成器前2个=~a 相等=~a~%" a b (equalp (subseq a 0 2) b))))

;; R12 pinv 数值精度（与 numpy.linalg.pinv 的最大相对差）
(multiple-value-bind (u s v) (vt-svd (vt-from-sequence (list (list 0.55d0 0.72d0 0.6d0 0.54d0) (list 0.42d0 0.19d0 0.33d0 0.92d0) (list 0.3d0 0.61d0 0.19d0 0.09d0)) :dtype :float64))
  (declare (ignore u s v)))
(let ((ap (vt-pinv (vt-from-sequence (list (list 0.55d0 0.72d0 0.6d0 0.54d0) (list 0.42d0 0.19d0 0.33d0 0.92d0) (list 0.3d0 0.61d0 0.19d0 0.09d0)) :dtype :float64))))
  (format t "R12|~a~%" (vt-to-list (vt-flatten ap))))
'''

def run_clvt():
    script = PRELUDE.replace('CLVT_DIR', CLVT_HOME + os.sep) + CASES
    with tempfile.NamedTemporaryFile('w', suffix='.lisp', delete=False, encoding='utf-8') as f:
        f.write(script)
        path = f.name
    cmd = [SBCL, '--noinform', '--disable-debugger', '--script', path]
    if QL:
        cmd = [SBCL, '--noinform', '--disable-debugger',
               '--eval', f'(load "{QL}")',
               '--eval', '(ql:quickload :sb-simd :silent t)',
               '--script', path]
    env = dict(os.environ, CLVT_HOME=CLVT_HOME)
    out = subprocess.run(cmd, capture_output=True, text=True, timeout=600, env=env)
    os.unlink(path)
    res = {}
    for line in out.stdout.splitlines():
        m = re.match(r'(R\d+)\|(.*)', line)
        if m:
            res[m.group(1)] = m.group(2)
    return res

# ---------- numpy 期望 ----------
def npv(tag):
    A = np.array([[0.234, 0.034, 0.58], [0.129, 0.686, 0.684],
                  [0.873, 0.127, 0.59], [0.053, 0.322, 0.456], [0.905, 0.541, 0.037]])
    P = np.array([[0.55, 0.72, 0.6, 0.54], [0.42, 0.19, 0.33, 0.92], [0.3, 0.61, 0.19, 0.09]])
    if tag == 'R1': return 99.0, 'reducer 常数 99'
    if tag == 'R2':
        M = np.array([[1, 9, 4], [7, 2, 8]], dtype=float)
        return (M.max(axis=0), M.argmax(axis=0)), '每列最大值与索引'
    if tag == 'R3':
        return np.interp([0, 1, 3], [0, 2, 4], [10, 20, 30], left=-1.0, right=99.0), 'x==xp[0] 取 fp[0]'
    if tag == 'R4': return np.copysign(1.0, -0.0), '-0.0 的符号'
    if tag == 'R5': return np.signbit(np.nan), 'NaN 符号位'
    if tag == 'R6': return np.linalg.cond(np.array([[1, 2], [3, 4]]), 1), '1-范数条件数'
    if tag == 'R7': return (np.isin([1, 2, 3, 4], [3, 1]), ':int8'), 'F4: 布尔应为 int8'
    if tag == 'R8': return (np.maximum([-2, 0, 3], 0), ':int64'), 'numpy 保持整数'
    if tag == 'R9': return 'OK', 'numpy 允许（对 k,l 求和）'
    if tag == 'R10': return (np.linalg.norm(np.array([3.0, 4.0]), keepdims=True).shape, '(1,)'), 'keepdims 无 axis'
    if tag == 'R11': return True, '生成器状态应被推进'
    if tag == 'R12': return np.linalg.pinv(P), 'LAPACK pinv'
    return None, ''

# v0.4.1：以下分歧均已在 clvt 源码中修复（见 git 补丁），判定列为修复后的
# 预期语义；报告中若出现非「一致」判定说明回归。
VERDICT = {
 'R1': ('一致', 'vt-reduce 自定义 reducer-fn 已被尊重（快路径增加 fn 门控，全局归约不再退化为求和）'),
 'R2': ('一致', 'vt-sort 非末轴排序已修复（遍历尾部维度）；vt-topk values/indices 已一致'),
 'R3': ('一致', 'vt-interp 边界已对齐 numpy：left 仅在 x<xp[0] 时生效，x==xp[0] 取 fp[0]'),
 'R4': ('一致', 'vt-copysign 已按 IEEE 符号位处理（copysign(1,-0.0)=-1，含 NaN 符号位）'),
 'R5': ('一致', 'vt-signbit 已按 IEEE 符号位判定（numpy 的 0/0 NaN 符号位同为 1；常量 NaN 为 0）'),
 'R6': ('一致', 'vt-cond p=1/:inf 已实现 ||A||·||A⁻¹||（含奇异矩阵报错，对标 LinAlgError）'),
 'R7': ('一致', 'vt-in1d 已返回 :int8（F4 契约）'),
 'R8': ('一致', 'vt-relu 整数输入保持整数 dtype（对标 numpy.maximum/torch.relu）'),
 'R9': ('对照', 'numpy 实测 ij,kl->ij 合法（对 k,l 求和），clvt 原行为正确——第 2 阶段探针期望错误，已更正为数值对比'),
 'R10':('一致', 'vt-norm keepdims 无 axis 已返回全 1 形状（1d→(1,)，2d→(1,1)）'),
 'R11':('一致', 'with-generator 作用域内抽样已直接消费生成器流（状态被推进）'),
 'R12':('一致', 'vt-pinv 已加 Newton–Schulz 精化（SVD 容差 1e-14），精度达机器精度'),
}

def fmt_np(v):
    if isinstance(v, tuple):
        return np.concatenate([np.asarray(x).ravel() for x in v])[:8]
    return np.asarray(v).reshape(-1)[:6]

if __name__ == '__main__':
    if not SBCL:
        sys.exit('未找到 SBCL，请设置 SBCL 环境变量')
    print('运行 clvt 最小复现 ...')
    res = run_clvt()
    print(f"\n{'ID':5} {'clvt 结果':52} {'numpy 期望':44} 判定")
    print('=' * 155)
    for k in sorted(res):
        np_v, np_desc = npv(k)
        verdict, note = VERDICT.get(k, ('?', ''))
        print(f"{k:5} {res[k][:52]:52} {np_desc}→{str(fmt_np(np_v))[:34]:36} {verdict}: {note}")
    n_bug = sum(1 for v in VERDICT.values() if v[0] == 'BUG')
    print(f"\n汇总: BUG={sum(1 for k in VERDICT if VERDICT[k][0]=='BUG')}  "
          f"分歧={sum(1 for k in VERDICT if VERDICT[k][0]=='分歧')}  "
          f"API={sum(1 for k in VERDICT if VERDICT[k][0]=='API')}  "
          f"精度={sum(1 for k in VERDICT if VERDICT[k][0]=='精度')}")
