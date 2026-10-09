#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""最小复现验证：对每个候选 bug，用干净的最小用例对照 clvt 与 numpy"""
import os, re, subprocess, json, math, shutil
import numpy as np
np.seterr(all='ignore')

# 输入/输出目录：默认当前目录下的 tmp/，可用环境变量 FUZZ_TMP 覆盖
# 本脚本位于 test/ 下：输入/输出默认放仓库 tmp/（可用 FUZZ_TMP 覆盖）
TMP = os.environ.get('FUZZ_TMP') or os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'tmp')
SBCL = os.environ.get('SBCL') or shutil.which('sbcl')

# ---------- 从 c0181 的 lisp 字面量还原 float32 操作数 ----------
cases = {c['id']: c for c in json.load(open(f'{TMP}/cases.json'))}
def lisp_floats(s):
    return [float(x.replace('d','e')) for x in re.findall(r'-?\d+\.\d+(?:[deDE][+-]?\d+)?', s)]
l181 = cases['c0181']['lisp']
nums = lisp_floats(l181)
a32 = np.array(nums[:6], dtype=np.float32)
b32 = np.array(nums[6:12], dtype=np.float32)

REPROS = r'''
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
          (%probe-clvt-dir d))
        (%probe-clvt-dir *default-pathname-defaults*))))
(let ((root (%find-clvt-root)))
  (unless root
    (error "未找到 clvt/clvt.asd：请设置环境变量 CLVT_HOME=clvt 仓库根目录运行。"))
  (pushnew root asdf:*central-registry* :test #'equal))
(handler-bind ((warning #'muffle-warning)) (asdf:load-system :clvt))
(defpackage :vr (:use :cl :clvt)) (in-package :vr)
(defparameter +nan+ (sb-int:with-float-traps-masked (:invalid) (/ 0.0d0 0.0d0)))
(defparameter +inf+ sb-ext:double-float-positive-infinity)
(defparameter +ninf+ sb-ext:double-float-negative-infinity)
(defun rep (id fn)
  (handler-case
      (let ((r (funcall fn)))
        (typecase r
          (vt (format t "~a|VT|dtype=~(~s~) shape=(~{~a~^ ~}) data=[~{~a~^ ~}]~%"
                  id (vt-dtype r) (vt-shape r)
                  (mapcar (lambda (x) (typecase x
                            (double-float (cond ((sb-ext:float-nan-p x) "nan")
                                                ((sb-ext:float-infinity-p x) (if (plusp x) "inf" "-inf"))
                                                (t (format nil "~,16E" x))))
                            (single-float (rep id (lambda () (coerce x 'double-float))) nil)
                            (t (format nil "~a" x))))
                          (let ((l (vt-to-list r)))
                            (if (and (vt-shape r) (vt-shape r)) (alexandria:flatten l) (list (vt-item r)))))))
          (t (format t "~a|SCALAR|~s~%" id r))))
    (error (e) (format t "~a|ERR|~a~%" id (substitute #\space #\newline (format nil "~a" e))))))
'''  # 占位，实际在下方重写更稳妥

# 上面 rep 对 single-float 的处理太绕，直接统一转 double 再打
REPROS = r'''
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
          (%probe-clvt-dir d))
        (%probe-clvt-dir *default-pathname-defaults*))))
(let ((root (%find-clvt-root)))
  (unless root
    (error "未找到 clvt/clvt.asd：请设置环境变量 CLVT_HOME=clvt 仓库根目录运行。"))
  (pushnew root asdf:*central-registry* :test #'equal))
(handler-bind ((warning #'muffle-warning)) (asdf:load-system :clvt))
(defpackage :vr (:use :cl :clvt)) (in-package :vr)
(defparameter +nan+ (sb-int:with-float-traps-masked (:invalid) (/ 0.0d0 0.0d0)))
(defparameter +inf+ sb-ext:double-float-positive-infinity)
(defparameter +ninf+ sb-ext:double-float-negative-infinity)

(defun tok (x)
  (typecase x
    (single-float (tok (coerce x 'double-float)))
    (double-float (cond ((sb-ext:float-nan-p x) "nan")
                        ((sb-ext:float-infinity-p x) (if (plusp x) "inf" "-inf"))
                        (t (format nil "~,16E" x))))
    (integer (format nil "~a" x))
    (t (format nil "~a" x))))
(defun flat (x) (cond ((null x) nil) ((atom x) (list x)) (t (append (flat (car x)) (flat (cdr x))))))

(defmacro rep (id form)
  `(handler-case
       (let ((r ,form))
         (typecase r
           (vt (format t "~a|VT|dtype=~(~s~) shape=(~{~a~^ ~}) data=[~{~a~^ ~}]~%"
                       ,id (vt-dtype r) (vt-shape r)
                       (mapcar #'tok (flat (vt-to-list r)))))
           (number (format t "~a|SCALAR|~a~%" ,id (tok r)))
           (t (format t "~a|OTHER|~s~%" ,id r))))
     (error (e) (format t "~a|ERR|~a~%" ,id
                        (substitute #\space #\newline (format nil "~a" e))))))

(rep R01 (vt-/ (vt-const '(1) 7 :dtype :uint8) (vt-const '(1) 4 :dtype :uint8)))
(rep R02 (vt-/ (vt-const '(1) 7 :dtype :uint16) (vt-const '(1) 4 :dtype :uint16)))
(rep R03 (vt-/ (vt-const '(1) 1.5d0 :dtype :float64) (vt-const '(1) 2.5f0 :dtype :float32)))
(rep R04 (vt-/ (vt-const '(1) 7 :dtype :int64) (vt-const '(1) 2.5f0 :dtype :float32)))
(rep R05 (vt-/ (vt-const '(1) 7 :dtype :int64) (vt-const '(1) 2.5d0 :dtype :float64)))
(rep R06 (vt-div (vt-const '(1) 0.864d0) (vt-const '(1) 0.15d0)))
(rep R07 (vt-div (vt-const '(2) -3.7d0) (vt-const '(2) 1.5d0)))
(rep R08 (vt-cbrt (vt-const '(1) -27.0d0)))
(rep R09 (vt-cbrt (vt-const '(1) -8.0d0)))
(rep R10 (vt-expt (vt-const '(1) -8.0d0) (/ 1.0d0 3.0d0)))
(rep R11 (vt-put (vt-broadcast-to (vt-from-sequence '(1 2) :dtype :int64) '(2 2)) 0 9))
(rep R12 (vt-item (vt-reshape (vt-const '(1 1) 9 :dtype :int64) '(1 1))))
(rep R13 (vt-isclose (vt-const '(1) 1.000000001d0) (vt-const '(1) 1.0d0) :rtol 1.0d-9 :atol 0.0d0))
(rep R14 (vt-slice (vt-arange 6 :dtype :int64) '(-2)))
(rep R15 (vt-insert (vt-arange 4 :dtype :int64) (vt-from-sequence '(0 2) :dtype :int64) (vt-const '(2) 7 :dtype :int64)))
(rep R16 (vt-delete (vt-arange 4 :dtype :int64) (vt-from-sequence '(0 2) :dtype :int64)))
(rep R17 (vt-percentile (vt-zeros '(0)) 50))
(rep R18 (vt-vsplit (vt-reshape (vt-arange 12 :dtype :int64) '(3 4)) 2))
(rep R19 (vt-mod (vt-from-sequence __LITA__ :dtype :float32) (vt-from-sequence __LITB__ :dtype :float32)))
(rep R20 (vt-select (list (vt-const '(2 1) 1 :dtype :int8)) (list (vt-ones '(2 3) :dtype :int64)) :default -1))
(rep R21 (vt-cbrt (vt-const '(1) +ninf+)))
(rep R22 (vt-cbrt (vt-const '(1) +nan+)))
(rep R23 (vt-isclose (vt-const '(1) 1.0d0) (vt-const '(1) 1.0d0)))
(rep R24 (vt-isclose (vt-const '(1) 1.0d0) (vt-const '(1) 1.05d0) :rtol 0.1d0 :atol 0.0d0))
'''

# R19 由 python 注入实际数组字面量
lit = '(list ' + ' '.join(f"{float(v):.8e}" for v in a32) + ')'
litb = '(list ' + ' '.join(f"{float(v):.8e}" for v in b32) + ')'
REPROS = re.sub(r'\(rep (R\d+)', r'(rep "\1"', REPROS)  # id 改为字符串，避免符号求值
REPROS = REPROS.replace('__LITA__', lit).replace('__LITB__', litb)

os.makedirs(TMP, exist_ok=True)
with open(f'{TMP}/verify.lisp','w') as f: f.write(REPROS)
assert SBCL, "未找到 SBCL：请安装 sbcl 或用环境变量 SBCL 指定可执行文件路径"
out = subprocess.run([SBCL,'--script',f'{TMP}/verify.lisp'],
                     capture_output=True, text=True, timeout=240).stdout
res = {}
for line in out.splitlines():
    if '|' in line and re.match(r'^R\d+\|', line):
        k,_,p = line.partition('|'); res[k] = line[len(k)+1:]

# ---------- numpy 侧 ----------
def npv(name):
    v = {
      'R01': np.true_divide(np.array([7],np.uint8), np.array([4],np.uint8)),
      'R02': np.true_divide(np.array([7],np.uint16), np.array([4],np.uint16)),
      'R03': np.true_divide(np.array([1.5],np.float64), np.array([2.5],np.float32)),
      'R04': np.true_divide(np.array([7],np.int64), np.array([2.5],np.float32)),
      'R05': np.true_divide(np.array([7],np.int64), np.array([2.5],np.float64)),
      'R06': np.floor_divide(np.array([0.864]), np.array([0.15])),
      'R07': np.floor_divide(np.array([-3.7,3.7]), np.array([1.5,1.5])),
      'R08': np.cbrt(np.array([-27.0])),
      'R09': np.cbrt(np.array([-8.0])),
      'R10': np.power(np.array([-8.0]), np.array([1/3])),
      'R11': 'ERR(read-only)',
      'R12': np.array([[9]]).item(),
      'R13': np.isclose(1.000000001, 1.0, rtol=1e-9, atol=0.0),
      'R14': np.arange(6)[-2],
      'R15': np.insert(np.arange(4), np.array([0,2]), np.array([7,7])),
      'R16': np.delete(np.arange(4), np.array([0,2])),
      'R17': 'ERR(empty)',
      'R18': 'ERR(non-div)',
      'R19': np.mod(a32, b32),
      'R20': np.select([np.ones((2,1),dtype=bool)], [np.ones((2,3),np.int64)], default=-1),
      'R21': np.cbrt(np.array([-np.inf])),
      'R22': np.cbrt(np.array([np.nan])),
      'R23': np.isclose(np.array([1.0]), np.array([1.0])),
      'R24': np.isclose(np.array([1.0]), np.array([1.05]), rtol=0.1, atol=0.0),
    }[name]
    return v

def fmt_np(v):
    try:
        a = np.asarray(v)
        if a.dtype == np.bool_: a = a.astype(np.int8)
        if a.ndim == 0: return f"SCALAR {float(a[()])!r}" if a.dtype.kind=='f' else f"SCALAR {a[()]}"
        return f"VT {a.dtype} {a.shape} {[float(x) for x in a.reshape(-1)] if a.dtype.kind=='f' else [int(x) for x in a.reshape(-1)]}"
    except Exception:
        return v

VERDICT = {
 'R01':('BUG','vt-/ uint8/uint8 做整数除法，numpy 返回 float64 1.75'),
 'R02':('BUG','vt-/ uint16 同上'),
 'R03':('BUG','vt-/ float64+float32 提升为 float32，numpy 为 float64'),
 'R04':('BUG','vt-/ int64+float32 → float32，numpy 为 float64'),
 'R05':('对照','int64/float64 → float64 正确'),
 'R06':('BUG','vt-div 浮点未做 floor，numpy floor_divide=5.0'),
 'R07':('BUG','vt-div 浮点负数同样未 floor'),
 'R08':('对照','cbrt(-27)=-3 正确（fuzz 错误源于 -inf/NaN，见 R21/R22）'),
 'R09':('对照','cbrt(-8)=-2 正确'),
 'R10':('API','vt-expt 仅支持标量指数（nan 语义本身正确）；张量指数 unsupported'),
 'R11':('BUG','vt-put 写 stride-0 广播视图未报错（违反 H4）'),
 'R12':('BUG','vt-item 对 (1,1) 二维返回张量而非标量'),
 'R13':('BUG','vt-isclose 带 :rtol/:atol 时返回 :float64，违反 F4（应为 :int8）'),
 'R14':('确认','单元素 spec = 索引语义（与 numpy a[-2] 一致，非 bug）'),
 'R15':('API','vt-insert 不接受 VT 索引（numpy 接受数组）'),
 'R16':('API','vt-delete 不接受 VT 索引'),
 'R17':('分歧','空数组 percentile：clvt=nan，numpy 报错'),
 'R18':('分歧','vsplit 不整除：clvt array_split 语义，numpy 报错'),
 'R19':('BUG','float32 mod 与 numpy fmod 结果不一致（fmod 应逐位精确）'),
 'R20':('对照','探针修正：全 1 条件下结果一致；广播条件场景在 numpy 需 bool，无法直接对标'),
 'R21':('BUG?','cbrt(-inf) 行为待定，numpy=-inf'),
 'R22':('BUG?','cbrt(nan) 行为待定，numpy=nan'),
 'R23':('BUG?','isclose 默认参数返回 dtype 应为 :int8'),
 'R24':('BUG?','isclose :rtol 0.1 是否生效待定，numpy=True'),
}

print(f"{'ID':5} {'clvt 结果':58} {'numpy 期望':50} 判定")
print('='*150)
for k in sorted(res):
    print(f"{k:5} {res[k][:58]:58} {str(fmt_np(npv(k)))[:50]:50} {VERDICT.get(k,('?',''))[0]}: {VERDICT.get(k,('',''))[1]}")
