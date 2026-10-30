# CHANGELOG

本文件记录 clvt 库的每一次重大修改。

---

## 2026-10-30（四十七）— fix: 退化形状（shape 含 0 / 0 维 / 单观测 / num=0）暴露的十三类缺陷 —— 零元素不是零值

上一遍用随机输入找问题，而随机输入**几乎不会生成 0**：`randint` 取到 0 长度
的概率是 0。于是「shape 里含 0」这一格从未被踩过，而它在实现上恰好最危险。
这一遍把维度推到边界，撞出的一类缺陷有共同形状：

> 凡是「先跑一轮循环体、再判断要不要继续」的结构（里程计 / do-while /
> `advance` 回调），在**总元素数为 0** 时都会先跑那一轮 —— 而那一轮会去
> `aref` 一个长度 0 的 buffer。

**空维度（此前直接崩或误报）**：

- `vt-cumsum` / `vt-cumprod`：沿轴 + 元素数 0 时报
  `Invalid index 0 for (SIMPLE-ARRAY DOUBLE-FLOAT (0))`。**两个内核各中一次**
  （类型特化的 `run-axis` 与 dtype 提升时的通用 fallback），
  现统一在进入轴分支前判 `(plusp (vt-size result))`。numpy 对
  `np.cumsum(np.zeros((0,3)), axis=1)` 返回同形状空数组。
- `vt-nancumsum` / `vt-nancumprod`：走的是第三个内核
  （`%E3-CUMULATIVE-SKIP-NAN`），同一处 do-while 结构，同一保护。
- `vt-indices`：维度里含 0 且不在首位（如 `(2,0,3)`）时报
  `cannot index into an empty tensor`。原因是该层的 stride 算成 0 后被兜成 1，
  然后照常写满 `outer × axis-len`。现按层大小为 0 跳过填充；
  `np.indices((2,0,3)).shape == (3,2,0,3)`。
- `vt-vsplit` / `vt-hsplit` / `vt-dsplit`：轴长为 0 时误报「不整除」——
  `%require-equal-division` 的 `(plusp dim)` 先短路了 `(mod dim n)`，把 0
  归进「不整除」。0 对任何正整数都整除，`np.vsplit(np.zeros((0,3)), 2)` 给
  两个 `(0,3)`。

**角参数（此前抛浮点陷阱或给出「看起来合理」的错数）**：

- `vt-cov` / `vt-corrcoef`：零观测时 `FLOATING-POINT-INVALID-OPERATION` /
  `DIVISION-BY-ZERO`（未屏蔽陷阱，§5.5）；现给 nan。
- `vt-cov` 的角分母：原实现 `(if (<= denom 0) n-obs denom)` 是**凭空改分母**。
  numpy 的写法是 `fact = max(n_obs - ddof, 0)` 然后 `c *= 1/fact` —— 于是
  `cov([5.])` 是 nan（0/0，本库原给 0.0）、`cov([1,2], ddof=3)` 是 inf
  （0.5/0，原给 0.25）。inf 与 nan 是**信息**，换成有限数等于给出无意义的数。
- `vt-corrcoef` 的标量分支硬编码 1.0，正确公式始终是 `C / sqrt(C²)`：
  `np.corrcoef([5.])` 是 nan（本库原给 1.0）。
- `vt-linspace` / `vt-logspace` / `vt-geomspace`：`num = 0` 报
  「num 必须大于 0」，而 numpy 返回空数组（空采样是合法请求）。
  `num < 0` 仍报错；且 `vt-geomspace` 的端点零/同号校验**先于**数量校验
  （`np.geomspace(0,10,0)` 仍报「不能含零」）。

**API 补齐（一个参数只写了一半形态）**：

- `vt-bincount` 新增 `:weights`。三条容易漏的 numpy 规则都对齐：结果恒为
  `float64`（即使权重是整型或布尔）、负权重合法、复数权重**报错**
  （numpy 在这里 TypeError，本库不「出于好意」给出 complex128）。
- `vt-random-choice` 的 `:p` 接受张量。numpy 的 `p` 本来就是 ndarray，原实现
  直接走序列协议（`length`/`every`/`reduce`），最常见的写法报
  `The value #<VT …> is not of type SEQUENCE`。三种形态（nil / Lisp 序列 /
  张量）归一到同一列表，同 seed 下必须给出同一序列。

**错误契约**：random 族的 6 处裸 `ASSERT` 改成带函数名的消息。`(assert (> a 0)
(a))` 报的是 `The assertion (> A 0) failed with A = 0.` —— 没有函数名、没有
期望值，且 `ASSERT` 本身是可被优化级别干掉的调试工具，不是给调用方看的契约。

**旁证（零缺陷，已固化为断言）**：IO 往返 188 组合（11 dtype × 7 形状 +
非连续视图 + NaN/Inf/±0 + 复数，`:lisp` 与 `:npy` 双格式，逐**位模式**比）、
dtype 转换矩阵 121 组合（11×11，含向零截断/窄化回绕/无符号回绕/真值化/
复数丢虚部）、集合运算 92 条、索引族 35 条、`nn` 族恒等式 40 余条。

**新增套件**：`test/degenerate-empty-test.lisp` 331 条断言（A–J 十组）。
期望值全部 numpy 2.4.6 实测。

**文档**：`CONVENTIONS.md` 新增 §17.10.21（空维度不是「另一种数值」，
是「没有数值」）。

---

## 2026-10-30（四十六）— fix: 差分模糊测试（约 1800 条随机用例）挖出的九类缺陷 —— 结果秩由数值决定、轴元组全线缺失、两种参数形态只写了一种

这一遍的尺子不挑对象：**差分模糊测试**。Lisp 侧 `vt-save :as :npy` 落盘、
Python 侧 `np.load` 逐位比对，八个功能域（归约/操纵/算术/线代/边界/复数/
布尔/杂项）共约 1800 条随机用例。撞出的缺陷有同一个形状：**返回形状由某个
具体数值决定**（右端项个数、参数是不是标量），或者**一个参数有两种合法形态，
实现只写了一种**（轴元组、边界序列、分位点序列、张量位移）。

**破坏性变更（`vt-ptp` 的返回值从 Lisp 数变成 0 维张量）**：

- `(vt-ptp a)` 原先返回 Lisp 标量，与同族的 `vt-amax`/`vt-amin`/`vt-sum`
  不一致（它们都返回 vt），也让 `:keepdims` 无处可放 —— 参数加了也没语义。
  现在恒走 `vt--`，全归约给 0 维张量（§6.5）。要标量请 `(vt-item …)`。
  同批修掉：`vt-ptp` 新增 `:keepdims`；布尔输入的两个分支现在报**同一条**
  消息（此前 `:axis` 分支报「布尔减法未定义」，非 axis 分支把 `T` 丢进 `-`
  报「T is not of type NUMBER」）。
- `(vt-solve A b)` / `(vt-inv A)` 的结果秩改由**输入的秩**决定。
  `b` 形状 `(n,1)` 时不再被压成 `(n)`；1×1 方程组的 `vt-inv` 返回
  `(1 1)` 而不是 `(1)`（对标 `np.linalg.solve` / `np.linalg.inv`）。
  判据从 `(= nrhs 1)` 改成 `b-vector-p`：**用「某维度是否等于 1」代替
  「秩是几」永远会错**，因为 1 是合法维度。

**API 补齐（此前报「必须为正整数 / 必须为实数 / not of type」的地方）**：

- `vt-median` / `vt-nanmedian`：`:axis` 支持**轴元组**（`'(0 1)`、负轴、
  任意顺序）。此前 `(vt-median a :axis '(0 1))` 抛
  `The value (0 1) is not of type …`，而 `vt-sum`/`vt-mean`/`vt-amax` 一族
  早就支持 —— 属「同族不同能力」。
- `vt-percentile` / `vt-quantile`：`:axis` 同样支持轴元组；且**轴元组 ×
  分位点序列**可同时使用（先按分位点堆叠新首轴，再折轴），
  `(vt-percentile a '(10 90) :axis '(0 1))` 给 `(2 4)`，与 numpy 同形。
- `vt-percentile` / `vt-quantile`：`percentile` / `q` 支持**序列**，
  沿新首轴堆叠（`(vt-percentile a '(25 50 75))` → shape `(3)`）。
- `vt-histogram`：`:bins` 支持**边界序列**（≥2 且严格递增），counts 长
  `n-1`，`[e0, e_{n-1}]` 之外的值不计入，桶定位等价
  `searchsorted(side='right')-1`。此前只认正整数，序列报「必须为正整数」。
  同批修掉：`NaN` 会从两侧过滤里漏过去（`NaN` 与任何数的 `<`/`>` 都是假），
  一路落进**最后一个箱** —— numpy 是丢弃。
- `vt-left-shift` / `vt-right-shift`：第二操作数可以是**张量**（逐元素 +
  广播）。此前报 `not of type INTEGER`，numpy 支持。dtype 按混型规则提升
  （`uint8 << int64张量` → `int64`，而 `uint8 << 1` 仍是 `uint8`）。
- `vt-nanmedian`：全局归约 + `:keepdims t` 返回全 1 形状（原先返回 0 维，
  与 `vt-median` 的 `(1 1)` 不一致）。

**清理**：`src/nn.lisp` 9 处 + `src/reduce-stats.lisp` 1 处的 `%cx-reject`
守卫各被粘贴成两份（第一份是死代码，缩进顶格）。只留一份，行为不变。

**连带修正的既有断言**（它们把旧缺陷写成了期望）：`bool-dtype-test` 的
`ptp/int-unchanged`（原先断言返回 Lisp 标量 8.0d0）、
`linalg-fast-path-test` 的 `ref/solve-general3` 与 `numpy/solve-general3`
（原先靠 `(vt-to-list <(3,1) 结果>)` 得 `(5.75 -5.5 2.75)` 的假象）、
`example/example.lisp` 的 `(assert (= (vt-ptp a) 4.0d0))`。

**新增套件**：`test/reduce-shape-axis-test.lisp` 145 条断言（A–I 九组：
solve 秩 / ptp / nanmedian keepdims / median 多轴 / percentile 多轴×序列 /
位移张量 / histogram 边界 / 复数守卫去重 / 多轴归约族形状一致性交叉网）。
期望值全部 numpy 2.4.6 实测。

**文档**：`CONVENTIONS.md` 新增 §17.10.20（形状、轴元组、序列参数是三件
独立的事），并把「序列 `q` 尚未实现」从 §17.10.18 的偏差表里移出。

---

## 2026-10-30（四十五）— fix: 第十一遍审计（最后一轮，换轴：docstring 示例）—— 打印层三条规则、含 NaN 的复数打印崩溃、`(vector #c)` 丢虚部

这一遍的尺子不是 numpy，而是**每个 docstring 里写着 `=>` 的那一行**。
366 个带 docstring 的导出函数里有 57 行 `=>`，45 行可执行，逐条照抄跑一遍，
撞出 8 处不实：3 处打印层真缺陷 + 5 处示例与实现脱节。

**破坏性变更（打印输出变了）**：

- **小量不再退化成 0**。`%format-real-part` 原先恒用 `~,6f`，`|x| < 5e-7` 的
  非零值一律舍成 `0.0`（相对误差 100%）：`(vt-spacing 1.0)` 的值是
  `2.22e-16`，打印出来却是 `0.0`。现在按量级切换科学计数：
  `2.220446e-16`、`1.0e-300`。
- **大数不再印满一行**。`1.0d308` 原先印出 309 位数字，现在 `1.797693e+308`。
- **NaN / ±Inf 印 `nan` / `inf` / `-inf`**。原先分别印成
  `#<DOUBLE-FLOAT quiet NaN>` 与 `#.SB-EXT:DOUBLE-FLOAT-POSITIVE-INFINITY`
  —— 把 SBCL 的实现内部表示泄漏进用户输出，既不可读回也不可移植。

阈值：`|x| < 10^-P` 或 `|x| >= 10^(P+3)` 用科学计数，P = `*vt-print-precision*`
（默认 6）。`1e-6` 仍是 `0.000001`（这一位是准的），`1.3e-6` 切科学计数。
**注意这一层不承诺「可读回」**：CL 的 `1.0e-300` 默认读回 single-float 会下溢，
要读回 double 得写 `1.0d-300`；把指数写成 `d` 会把 dtype 写死进输出。
这里的承诺是「不丢量级、不泄漏实现」（§17.10.19）。

**破坏性变更（含 NaN 的复数以前根本打印不出来）**：

- `(format nil "~a" <含 NaN 的复数张量>)` 原先抛
  `FLOATING-POINT-INVALID-OPERATION`（`%format-complex-number` 用
  `(minusp im)` 判虚部符号，而 `(minusp <nan>)` 抛陷阱）。现在印 `nan+nanj`，
  与 docstring 举的例子一致。

**破坏性变更（静默丢数据）**：

- `(vt-from-array (vector #c(1d0 2d0)))` / `(vt-from-array #(#c(1d0 2d0)))`
  原先得 `:float64`，**虚部静默消失**。`(vector …)` 与 `#(…)` 的元素类型是
  `T`，所有 `subtypep` 判定全假 → 落到兜底的 `:float64`。v0.5.x 修过同一个
  坑，但只补了「元素类型声明为 complex」的分支。现在元素类型为 `T` 时按值扫
  一遍，含复数则 `:complex128`。非复数维持 `:float64`（见偏差表）。

**API 新增（导出三个此前压在内部包里的函数）**：

- `vt-get-nan` / `vt-get-pos-inf` / `vt-get-neg-inf`，对标 `np.nan` /
  `np.inf` / `-np.inf`。文档原先拿裸符号 `nan` / `inf` 举例，而这两个名字
  库里从来没有定义过 —— 用户照抄得到 `The value NAN is not of type REAL`。

**文档修正 7 处**：`vt-array-equal` / `vt-nan-to-num`（示例改用
`vt-get-nan` 一族，原先照抄必崩）、`vt-isin`（`[False True …]` → `[NIL T …]`，
布尔印 T/NIL 是 §17.8 定的）、`vt-interp`（`=> 25.0d0` → 实际 1 元素张量，
而它自己的 docstring 写着「输出恒 1D」）、`vt-vdot`（`70` → `70.0`，根因是
`from-array` 对 `T` 元素类型给 float64）、`vt-compress`（补 `:dtype :int64`）、
`vt-nextafter` / `vt-spacing`（示例改成 `(vt-item …)` 形式，表达精确值而不是
6 位打印）。

**未改（有意，记入 §17.10.19 偏差表）**：`(vt-from-array #(1 2 3))` 仍是
`:float64` 而 numpy 是 `int64`；`(vt-from-sequence '(1 2))` 同理；
`(vt-asarray '(1 2))` 是唯一给 `int64` 的构造入口。改成按值推断整数会波及
全库既有测试与用户代码，留作独立变更。

**验收**：`bool-dtype-test` 677 断言（新增 S 组 38 条）；65 套件全绿；
`check_docs.py` 通过（228 个 §锚点）。契约见 `CONVENTIONS.md` §17.10.19，
过程见 `HISTORY.md` H.3.38。

---

## 2026-10-30（四十四）— fix: 第十遍审计（换轴：同义入口族）—— 复数 matmul 秩、`nancumsum` 形状与精度、`divmod` 的 `:out`、零除符号、`dot` 的秩

这一遍的判据不是「这个函数对不对」，而是「**同一件事有几个入口**」。
用 `sb-introspect` 把全部导出符号按签名和词干分组，得到六张族表，每族写一条
必须成立的等式，然后让族内两个成员互为标尺（不依赖 numpy，也就不怕两边一起错）。
等式写不出来的族用退化输入逼（全 NaN / 空 / 0 除 / ±0）。六张表压完，7 处不符。

**破坏性变更（下面这些调用以前报错，现在给出结果）**：

- `(vt-matmul 复数矩阵 向量)`、`(vt-matmul 复数批量 向量)`：现在按 numpy 的
  1-D 提升规则返回 `(2)` / `(3 2)`。修复前前者给 `(2 1)`（多一维），后者
  `%cx-batched-matmul` 里 `nth` 越界直接报错。
- `(vt-matmul 实矩阵 复矩阵)`、`(vt-matmul c64 c128)`、`(vt-matmul f32 c128)`：
  现在先提升到 `vt-promote-type` 的公共 dtype 再进内核，结果 `:complex128`。
  修复前报 `%cx-dispatch: 非复数缓冲区 DOUBLE-FLOAT` —— `%cx-matmul-2d` 按
  `(vt-dtype a)` 分配结果缓冲。
- `(vt-dot 三维数组 向量)` / `(vt-dot 向量 三维数组)`（实数）：现在返回
  `(3 2)` / `(2 3)`。修复前报 `unsupported dimensions`，而复数路径已支持 ——
  同一个函数换个 dtype 就一个能算一个报错。
- `(vt-nancumsum 二维数组)`：形状现在是 `(8)`，与 `vt-cumsum` 及 numpy 一致。
  修复前值对、形状是 `(2 4)`：`%e3-cumulative-skip-nan` 建缓冲时无条件用了
  输入 `shape`，而它的 docstring 写着「AXIS 为 nil 时展平」。

**行为修正（数值变化）**：

- `(vt-nancumsum int64大值)` 不再丢低位。修复前无条件
  `(vt-astype input :float64)`，int64 超过 2^53 进 double 就损：
  `#(2^60 1 -2^60)` 给 `(2^60 2^60 0)`，现在是 `(2^60 2^60+1 1)`。
  整数/布尔输入里没有 NaN 可跳，直接委托 `vt-cumulative` —— 顺带修掉
  int8/uint8 的回绕（第九轮 Q 组记下的待办）。
- `(vt-div #(1 -1 0) #(0 0 0))` 首元素从 `-Inf` 变 `Inf`。`%floor-div-op` 用
  `plusp`/`minusp` 判符号，而它们对 ±0 一律给 `NIL` → 恒 `-Inf`。改走
  `float-sign`（`-0.0` 的 `float-sign` 是 `-1.0`）。
- `(nth-value 0 (vt-divmod a b))` 不再恒为 `NaN`。原实现自带一份 floor 除法
  副本，副本里 `(if (or nan-p nan-p) nan nan)` 两支相同（残迹），去掉副本、
  商改调 `vt-div`，与余数调 `vt-mod` 对称。

**契约收紧（从静默改为报错）**：

- `(vt-divmod a b :out o)`：现在报错。docstring 一直写着「传 `:out` 将报错」，
  实现是 `(declare (ignore out))` —— 静默吞掉，`out` 原封不动。这是全库
  `:out` 契约扫描（274 次探测）里唯一的漏网。
- `(vt-matmul 复 复 :out 实数数组)`：现在报错。原先走 `vt-copy-into`
  （能装就装）静默取实部降精度，与 §4.4 的「形状+dtype 精确相等」不符；
  改走 `vt-check-out`。

**未改（有意）**：`vt-var :ddof` 在 `ddof >= n` 时给 `NaN` 而 numpy 给 `inf`；
`vt-quantile` 不接受序列 `q`；`vt-median` 缺 `:dtype`/`:out`（与 `vt-nanmedian`
不对称，待补）。三条记入 §17.10.18 的偏差表。

**验收**：`bool-dtype-test` 639 断言（新增 R 组 64 条）；65 套件全绿；
`check_docs.py` 通过（227 个 §锚点）。契约见 `CONVENTIONS.md` §17.10.18，
过程见 `HISTORY.md` H.3.37。

---

## 2026-10-30（四十三）— fix: 第九遍审计（换轴：混型提升与转换入口）—— 复数÷整型、复数→实数写回、布尔标量、三元提升折叠

这一遍先去审第八遍用的那把尺子（`vt-astype … :bool`），再沿「dtype 提升 /
转换」铺开：混型二元算子 2021 组、提升表二元 121 + 三元 286、三条转换入口
605 组、合并/选择/三元算术 847 组，全部与 numpy 2.3.5 逐格对照。矩阵是干净的；
出问题的是**同一语义的多条实现**，共 5 处。

**破坏性变更（下面这些调用以前是报错，现在给出结果）**：

- `(vt-/ 复数数组 整型数组)` 与反向：结果 dtype 现在是 `:complex128` /
  `:complex64`。**修复前 24 个 (复数, 整型) 组合两个方向全部
  `TYPE-ERROR: The value #C(0.5d0 1.0d0) is not of type REAL`** —— `vt-/`
  的 `FLOAT-TARGET` 分支一见整型张量就把目标改写成 `:float64`，虚部无处可写。
  复数参与时改为交给 `vt-promote-type`（`c128 ⊔ int8 → c128`）。
- `(vt-map #'identity 复数张量 :dtype :float64)` / `(vt-conj c :dtype :float64)`：
  现在取实部（§16.5），不再 `TYPE-ERROR`。修法是新增 `%REAL-WRAP`，只在
  「目标实数 + 有复数输入」时包一层 `realpart`，热循环未改。
- `(vt-cast t :float64)`、`(vt-fill float数组 t)`、`(vt-full '(3) t :dtype
  :float64)`、`(setf (vt-ref float数组 0) t)`、`(vt-* float数组 t)`：`T`/`NIL`
  现在按 1/0 收敛，不再报 `The value T is not of type REAL`。数组入口
  （`vt-astype`）一直是对的，只有标量入口缺这条。
- `(setf (vt-ref float数组 0) #c(1 2))`：现在取实部（对标 numpy 的
  ComplexWarning 静默丢弃虚部），不再报错。
- `(vt-from-sequence '(t t) :dtype :float64)`：现在得到 `(1.0d0 1.0d0)`。
  修复前报 `The value T is not of type REAL when binding X` —— 一维构造的
  快路径对元素声明了 `(type real x)`。

**行为修正（结果 dtype 变化）**：

- `(vt-+ a b c)` 等三元及以上算术的 dtype。`%op-final-dtype` 原先是**逐对
  折叠** `promote(promote(a,b),c)`，三元起与「一次看全部」分叉：
  `int16, uint16, float32` 折叠得 `:float64`，而 numpy / `vt-promote-type` /
  `vt-concatenate` 都给 `:float32`。8 组受影响。

**未改（有意）**：`(setf (vt-ref int8数组 0) 300)` 仍报错 —— numpy 2.x 的
`a[0] = 300` 也是 `OverflowError`，而 `astype` 走回绕（`np.array([300]).astype(int8)`
→ `44`），两边都照 numpy。

**验收**：`bool-dtype-test` 575 断言（新增 Q 组 26 条）；65 套件全绿（275s）；
`check_docs.py` 通过。契约见 `CONVENTIONS.md` §17.10.17，过程见 `HISTORY.md` H.3.36。

---

## 2026-10-30（四十二）— fix: 第八遍审计（换轴：`:dtype` 当断言）—— 布尔输出被 `the` 截断、`%float-map` 预降输入、`&optional` 吞关键字

> 契约见 `CONVENTIONS.md` §17.10.16；决策过程见 `HISTORY.md` H.3.35。
> 判据：`(f x … :dtype :bool)` 必须等于 `(vt-astype (f x …) :bool)`
> —— 即**真值化**，不是朝零截断。反射筛出 139 个接受 `:dtype` 的导出
> 函数，144 个样本里 37 个不等，落在 35 个函数上。

**破坏性变更（35 个函数的 `:dtype :bool` 结果 + `vt-clip`/`vt-clamp` 的关键字解析）**：

| 表达式 | 修复前 | 现在 |
|---|---|---|
| `(vt-abs #(0.5 1.5 2.0 0.25 3.0 0.75) :dtype :bool)` | 字节 `0 1 2 0 3 0` | `1 1 1 1 1 1` |
| `(vt-cos b :dtype :bool)` | `(0 1 0 1 0 0)`（cos(1)=0.54 截成 0） | `(1 1 1 1 1 1)` |
| `(vt-exp b :dtype :bool)` | 字节 `2`（越界） | 字节 `1` |
| `(vt-log x :dtype :bool)` | `(0 0 0 0 0 0)` | `(1 1 1 1 1 1)` |
| `(vt-nancumsum #(0.5 -0.5 0.25) :dtype :bool)` | `(0 0 0)` | `(1 0 1)` |
| `(vt-negative b :dtype :bool)` | dtype 说 BOOL，字节 **255**（-1 落进 unsigned-byte 8） | 报错（同 numpy） |
| `(vt-clip a :dtype :bool)` | `:DTYPE fell through ETYPECASE …` | 正常返回（关键字归位） |

**根因一：内联/掩码内核用 `(the (unsigned-byte 8) v)` 收尾。**
`vt-fast-map` 的 `dtype-check`（输入 dtype = 结果 dtype 走内联）与
`maskable`（`:bool`/`:int8`/… 走掩码内核）让布尔结果进了特化循环，
写回只有一句 `the` —— 对浮点就是截断：0.84 → 0（**真值反了**）、
2.718 → 2（**脏字节**，破 §17.1）。而慢路径 `vt-map` 经 `vt-cast-fun`
→ `%bool-flag` 是真值化。修法是把 `:bool` 从两条快路径摘出去、一律回落
`vt-map`（两条实现合一），而不是给内核各加一个 `if`。
`fastmap-invariant-test` 24/24 仍通过（float64 路径未受影响）。

**根因二：`%float-map` 先把输入降成目标 dtype 再算。**
`(vt-fast-map op (%coerce-float-input IN %DT) :dtype %DT)` 在 `DT = :bool`
时把 0.5 先变成 1，于是 `log(0.5)` 算成了 `log(1)`。新增
`%FLOAT-MAP-BOOL`：按自然浮点精度（float32 进 float32 出）算完再 `%to-bool`。
注意 `%float-map` 的展开形状保持 `(if <FAST> <SLOW>)`，布尔分流挂在
`<SLOW>` 里 —— 否则 `fastmap-invariant-test` 会判为「快路径被改没」。

**根因三：累积族两条实现，一条 `vt-cast`、一条 `%cast-to`。**
`%cumulative` 写 `(vt-cast cum dtype)`（对），`%e3-cumulative-skip-nan` 的
展平路径走 `%e3-store-double-vector` 的 `(%cast-to lt v)`（截断）。给
`(unsigned-byte 8)` + `:bool` 单列一支真值化写回。

**另两处**：
- `vt-negative` 的拒绝条件写成 `(and (null dtype) …)`，`:dtype :bool`
  绕过检查。改为 `(or (null dtype) (eq dtype :bool))`；`:dtype :int8`
  这条「我知道自己在做什么」的出口保留。
- `vt-clip` / `vt-clamp` 的 `(vt &optional min max &key out dtype)`：
  CL 的 `&OPTIONAL` 不认关键字，`(vt-clip a :dtype :bool)` 把 `:dtype`
  当成了 MIN-VAL 的值。现在检测「MIN-VAL 是关键字」即把整段交回 `&KEY`
  重新分派（`vt-view` 的同款做法）。「只给下限」的报错语义不变。

**这一遍确认无事的轴**（记录以免重做）：`:out` 与输入别名（自别名/反向
视图/切片 × 3 dtype）全部正确；60+ 函数 × 3 dtype 扫「谁改写了输入」
0 处；同一逻辑数组换六种物理布局，一元/二元/归约结果全一致。

验收：bool-dtype-test 549 断言（新增 P 组 25 条）；65 套件全绿（278s）；
`fastmap-invariant-test` 24/24；check_docs 225 锚点。

---


> 契约见 `CONVENTIONS.md` §17.10.15；决策过程见 `HISTORY.md` H.3.34。
> 前六遍的探针有个共同前提：**所有调用都用默认参数**。这一遍把
> `:axis` / `:keepdims` / `:dtype` / `:ddof` 打开，对 46 个接受 `axis`
> 的导出函数做参数矩阵，共 303 组。

**破坏性变更（八处）**：

| 表达式 | 修复前 | 现在 | numpy |
|---|---|---|---|
| `(vt-sum b :axis 0 :dtype :bool)` | `:bool` `(2 1 1 2)`（脏字节） | `(1 1 1 1)` | `[T T T T]` |
| `(vt-sum b :axis 1 :dtype :bool)` | `(3 3)` | `(1 1)` | `[T T]` |
| `(vt-nansum b :axis 0 :dtype :bool)` | `(2 1 1 2)` | `(1 1 1 1)` | 同 |
| `(vt-l1-norm b :axis 0 :dtype :bool)` | `(2 1 1 2)` | `(1 1 1 1)` | — |
| `(vt-correlate #(1 2) #(1 0 1 0 1 0 1) :mode :valid)` | 形状 `(0)`（空） | `(2 1 2 1 2 1)` | 同 |
| `(vt-convolve … :mode :same)`（n 偶、核长） | 偏移差一位 | 与 numpy 同 | — |
| `(vt-nanstd b :dtype :bool)` | **静默** `:bool` `(0 1 1 0)` | 报错（同 numpy） | `TypeError` |
| `(vt-var/vt-std/vt-norm/vt-standardize b :dtype :bool)` | 报错但**点名内部函数**（`vt--` / `vt-sum`） | 点名调用者 | — |

**根因一：归约内核有两条写回路径，一条经过 `vt-cast`、一条裸写。**
全归约走 `%KERNEL-SMALL-GENERAL`，写回是 `(vt-cast acc dt)`，`%bool-flag`
把 2 收成 1；沿轴归约（窄整型）走 `%SMALL-INT-AXIS-FORM` 生成的
`%KERNEL-ROW-ACC`，写回是 `(setf (aref od i) acc)` —— `od` 声明为
`(unsigned-byte 8)`，2 原样落盘。于是布尔数组里出现**越界字节**，破
§17.1 的 0/1 不变量。修法不在内核里加 `if`（十几个宏实例），而是在
归约出口统一调 `%BOOL-CLAMP!`（按 shape/strides 只碰逻辑元素，对已
cast 过的路径幂等）。

**根因二：`correlate`/`convolve` 的 `:valid` 与 `:same` 裁切算错。**

- `:valid` 长度写成 `(max 0 (1+ (- n m)))` —— n<m 时是负数被 clamp 成 0，
  「短序列配长核」直接得空张量。numpy 的规则是 `|n-m|+1`，与谁长无关；
  起点应为 `min(n,m)-1`（原来用 `offset = m-1`）。
- `:same` 起点写成 `(floor (- full-len (max n m)) 2)`，`|n-m|` 为奇数时
  方向与 numpy 相反。实测 1..60 共 400 组拟合出**两条**公式：
  `correlate: min((m-1)/2, n/2)`、`convolve: min((m-1)/2, (n-1)/2)`
  —— numpy 的两个入口自己就不一致，所以 `vt-convolve` 不能直接委托，
  要传内部参数 `same-anchor`。复数路径 `%CX-CORRELATE` 是同一份逻辑的
  另一份拷贝，同样修。

验收：实数 3750 组、复数 2400 组与 numpy 逐值零失配。

**根因三：报错点错了人。** `unsupported output dtype (UNSIGNED-BYTE 8)`
把物理 Lisp 类型当 dtype 报给用户（用户写的是 `:dtype :bool`）；
`vt-var`/`vt-std`/`vt-standardize` 的 `:bool` 错误从内部的 `vt--` 冒出，
`vt-norm` 的从 `vt-sum` 冒出 —— 错误消息里的函数恰好是用户没调用的那个。
改为在公开入口前置校验，并在消息里给出下一步建议。

**成对不一致（第四遍的轴又出现）**：`vt-std` 拒绝 `:bool` 而 `vt-nanstd`
静默真值化成 `(0 1 1 0)`（读起来还挺像「标准差是否为零」）。numpy 两个都
抛 `TypeError`。`vt-nanvar` 一并补上。

**偏差（记入 §17.10.15）**：`(vt-sum int64数组 :dtype :bool)` 报错并建议
先数值归约再 `vt-astype`，numpy 则直接给真值化结果 —— 支持它需要给内核
分派加第三条路径（任意 in-et × 小整型 out），风险大于收益。
`(vt-std b :dtype :int8)` 本库按 int8 截断给 `(0 0 0 0)`，numpy 在 sqrt
处抛 `UFuncTypeError`（本库更宽松）。

**同时确认无事的**：轴矩阵的**值**层面 82 项全对；`argmax`/`argmin` 接受
元组轴（numpy 直接 `TypeError`）并给归约区内线性下标，语义自洽；
`mean`/`nanmean`/`prod` 的 `:dtype :bool` 与 numpy 同值。

**验收**：`bool-dtype-test` 524 断言全过（新增 O 组 33 条）；65 套件全绿
（严格模式）；`check_docs.py` 224 锚点全解析；无新增 WARNING。

---

## 2026-10-30（四十）— fix: 布尔当「位置参数」的两分法（delete/insert 掩码）+ geomspace 收 bool

> 契约见 `CONVENTIONS.md` §17.10.14；决策过程见 `HISTORY.md` H.3.33。
> 这一轮换到前五轮没碰过的一侧：前五轮问的都是动词（算、传、返、比、读），
> 这一轮问名词 —— 布尔张量**从哪造出来**、被放在**位置参数**上是什么含义。

**破坏性变更（三处）**：

| 表达式 | 修复前 | 现在 | numpy |
|---|---|---|---|
| `(vt-delete #(0 1 2 3) #(T NIL T T))` | `(2 3)`（当索引，删掉 0 和 1） | `(1)`（掩码） | `[1]` |
| `(vt-insert #(0 1 2 3) #(T NIL T T) 9)` | `(9 0 9 9 9 1 2 3)`（插 4 次） | `(9 0 1 9 2 9 3)` | 同 |
| `(vt-geomspace 1 8 3 :dtype :bool)` | 报「不支持的 dtype BOOL」 | `:bool` `(T T T)` | `[True True True]` |

**根因一：`%check-integer-index` 只校验「是不是整数」，而 `:bool` 的底层
类型就是 `(unsigned-byte 8)`**。于是布尔张量一路畅通地走了整数索引路径，
把掩码当成了「要删哪些下标」。危险不在结果错，而在**静默** ——
`np.delete(a, a > 3)` 是 numpy 最常见写法之一，移植过来不报错，只会删掉
一批完全不相干的元素。

numpy 对 `positions` / `obj` 参数有**两套并存语义，且按 dtype 而非按值
分叉**。同一份 `#(1 0 1 1)`，`:uint8` 走索引、`:bool` 走掩码：

| 走掩码（长度必须等于轴长） | 走索引（长度任意） |
|---|---|
| `delete` / `insert` / `place` / `putmask` / `compress` / `extract` | `take` / `put` / `choose` / `bincount` / `searchsorted` / `digitize` |

本库在索引侧全对，在掩码侧错了两个。修法是按 dtype 显式分叉
（`%BOOL-MASK-POSITIONS`），不是给 `:bool` 打个「也当索引」的补丁。
掩码形状不合规时报错（照抄 numpy 的措辞：必须一维且长度与被索引轴一致）：
`(vt-delete a #(T NIL))`、二维掩码、0 维掩码都拒绝。

**根因二（漏网）**：三处「按区间采样」的构造函数里，`linspace` /
`logspace` 一直支持 `:dtype :bool`（走 `vt-astype` 真值化），`geomspace`
的白名单漏了 `:bool`。同族三个入口两种行为。

**这一轮同时核实完的（第五遍的抽样结论）**：上一轮 1250 次机械调用里
685 条报错，当时用「参数语义不匹配」一句带过。这一轮补上 int64 /
float64 基线做三元对比，逐条核实：7 个「bool 挂 + 基线通」全部是减法族
（`vt--`/`negative`/`ptp`/`ediff1d`/`gradient`/`softmax`/`log-softmax`）的
刻意拒绝；「bool 通 + 基线报错」0 个；脏字节（dtype BOOL 但字节 ∉{0,1}）
**0 个**（1296 次调用）。

**确认无事的**：构造函数全族 19 个入口（含 `full` 的 `2`/`0.5`/`T`/`NIL`
四种值、`eye :k`、三个 `-like`、`meshgrid`）dtype 与字节全对；打印层四条
路径都渲染 `T`/`NIL`（二维对齐、0 维单个）；`(0,)`/`(3 0)`/`(0 0)` 空布尔
在 `sum`/`reshape`/`concatenate`/`vstack` 正常、空归约报 numpy 同款错误；
`gcd`/`lcm` 拒绝 bool 与 numpy 一致。

**偏差（记入 §17.10.14）**：`vt-empty`/`vt-empty-like` 给 0，numpy 给
未初始化内存（本库更安全）；`vt-arange :dtype :bool` 给结果，numpy 报错
（签名本就不同族）。

**验收**：`bool-dtype-test` 491 断言全过（新增 N 组 30 条）；65 套件全绿
（严格模式）；`check_docs.py` 223 锚点全解析；编译零 WARNING。

---

## 2026-10-30（三十九）— fix: 布尔「被消费」的六个入口（倒数 / 整除 / 峰谷差 / 形状工具 / along-axis / vander）

> 契约见 `CONVENTIONS.md` §17.10.13；决策过程见 `HISTORY.md` H.3.32。
> 这一轮不再挑函数，改用程序遍历全部 **375 个导出符号**（4 种调用形态，
> ~1250 次），记录 dtype / 值 / 物理字节并与 numpy 同名实现对照。

**破坏性变更（六处，全部是「原来给错答案或直接崩」）**：

| 表达式 | 修复前 | 现在 | numpy |
|---|---|---|---|
| `(vt-reciprocal #(T NIL T T))` | `:bool` **`(1 1 1 1)`**（假值变真） | `:int8`；含假值时报「整数零除」 | `int8` `[1 0 1 1]` |
| `(vt-divmod b b)` | 商 `:bool` **`(T T T T)`** | 商 `:int8` `(1 0 1 1)` | `int8` |
| `(vt-ptp b)` | `The value T is not of type NUMBER` | 「布尔张量不支持峰谷差」 | 报错 |
| `(vt-broadcast-shapes b '(2))` | **`Unhandled memory fault at #x0`** | 「参数必须是形状列表」 | — |
| `(vt-broadcast-strides b '(2) '(1))` | **内存错误** | 同上 | — |
| `(vt-apply-along-axis #'vt-sum 1 b)` | `:bool` **`(1 1)`**（3 被压成 1） | `:int64` `(3 3)` | `int64` |
| `(vt-apply-along-axis … 返回 #(5 6))` | `:bool`，字节 **5、6** | `:int64`，值 `5 6` | 跟随返回值 |
| `(vt-vander b)` | `:bool` | `:int64` | `int64` |

两条根因：

1. **`vt-int-dtype-p` 不认 `:bool`，却被当成「整数语义」的分支条件**。
   `vt-reciprocal` 因此落到浮点路径：`1/0` 在 with-float-safe 下是 `+Inf`，
   写回 `:bool` 被 `%bool-flag` 收成 1。`vt-divmod` 是 `dtype` 缺省走提升
   规则（bool ⊔ bool = bool），`floor(0/0)` 的 NaN 同样收成 1。两处都改成
   「全布尔输入 → `:int8`」，与 `vt-mod`/`vt-rem`/`vt-div` 并齐。
2. **`vt-apply-along-axis` 的结果 dtype 固定取输入 dtype**。布尔输入下
   1D→scalar 把 `(vt-sum row)` 的 3 压成 1，1D→1D 直接 `aref` 写 5、6 ——
   **布尔数组里出现越界字节**（破 §17.1 的 0/1 不变量）。现在布尔输入按
   FUNC 的返回类型定结果 dtype，非布尔输入维持旧契约。

**顺带修的两处健壮性**：`vt-ptp` 的布尔检查（原来非 `:axis` 分支报的是
内部类型泄漏，同一函数的 `:axis` 分支报的却是正确的布尔消息）；
`vt-broadcast-shapes`/`-strides` 去掉 `(declare (list …))`、换成运行时
`listp` 检查——保留声明时编译器会把这个检查优化掉，而传张量进来会读到
结构体的非法槽位，直接把进程打掉。

**偏差（不跟 numpy，已记入 §17.10.13）**：约 30 个超越函数在 numpy 里对
`bool` 输入给 `float16`，本库给 `float64`。`float16` 只有 3 位十进制有效
数字（`sqrt(2)` → 1.414），对齐 dtype 字符串不如算对。同性质三处：
`square`/`conj`（numpy `int8`，但它自己的 `b*b` 是 bool，内部不自洽）、
`round`/`rint`（numpy `float16`）、`clip(b,0,1)`（numpy 报错）。

**这一轮同时确认无误的**（记在 §17.10.13，免得下次重做）：23 个「布尔当
索引/掩码/计数」的入口（`take`/`put`/`choose`/`select`/`compress`/`extract`/
`take-along-axis`/`put-along-axis`/`searchsorted`/`bincount`/`digitize`/
`unique`/`isin`/`in1d`/`delete`/`setxor1d`/`setdiff1d`/`intersect1d`/
`union1d`/`nonzero`/`flatnonzero`/`argwhere`/`count-nonzero`）与 numpy 逐项
一致；`:out`/`copy-into`/`copy-to!`/`(setf (vt-slice …))`/`vt-fill`/
`vt-fill-diagonal`/`vt-put` 的字节卫生；`vt-from-sequence` 的 T/NIL 与 0/1
混合嵌套。

**验收**：`bool-dtype-test` 461 断言全过（新增 M 组 29 条）；65 套件全绿
（严格模式）；`check_docs.py` 222 锚点全解析；编译零 WARNING。

---

## 2026-10-30（三十八）— fix: 累积提升表与「标量 → 布尔」的两处不一致

> 契约见 `CONVENTIONS.md` §17.10.12；决策过程见 `HISTORY.md` H.3.31。
> 这一轮不测新函数，只把**同一族的两个实现**和**同一语义的两个入口**
> 并排比较。

**破坏性变更（一处，bool 与窄整型都受影响）**：

| 表达式 | 修复前 | 现在 | numpy |
|---|---|---|---|
| `(vt-nancumsum b)` | `:bool`，物理字节 **2** | `:int64` `[1 2 2 2]` | `int64` |
| `(vt-nancumsum #(100 100) :dtype :int8)` | `:int8` **`[100 -56]`**（回绕） | `:int64` `[100 200]` | `int64` |
| `(vt-nancumsum #(200 200) :dtype :uint8)` | `:uint8` **`[200 144]`** | `:int64` `[200 400]` | `uint64` |
| `(vt-nancumsum #(30000 30000) :dtype :int16)` | `:int16` **`[30000 -5536]`** | `:int64` `[30000 60000]` | `int64` |
| `(vt-nancumprod #(100 100) :dtype :int8)` | `:int8` **`[100 16]`** | `:int64` `[100 10000]` | `int64` |

根因：`vt-cumsum`/`vt-cumprod` 走 `%VT-CUMULATIVE`，里面有一张
`:bool :int8 :uint8 :int16 :uint16 :int32 → :int64` 的提升表；
`vt-nancumsum`/`vt-nancumprod` 走另一条内核 `%E3-CUMULATIVE-SKIP-NAN`，
那里只有 `(or dtype (vt-dtype input))`。同一判据两条内核给不同答案。
修法是把表抽成 `%CUMULATIVE-RESULT-DTYPE` 共用，而不是给 bool 单打补丁。

**修好的「同一个值、五个入口、四个答案」**：

| 入口 | 修复前 `(… #C(0 1) …)` | 现在 |
|---|---|---|
| `(vt-cast #C(0 1) :bool)` | `0` | `1` |
| `(vt-write x '(0) #C(0 1))` | `0` | `1` |
| `(vt-put x '(0) (list #C(0 1)))` | `0` | `1` |
| `(vt-fill x #C(0 1))` | `0` | `1` |
| `(vt-pad … :constant-values #C(0 1))` | `0` | `1` |
| `(setf (vt-ref x 0) #C(0 1))` | `1`（唯一正确的） | `1` |

`vt-cast` 原先「先剥虚部、再判 `:bool`」，于是 `#C(0 1)` 变成 `0`；
而 `%BOOL-FLAG` 的契约是「复数任一分量非零即为真」。把 `:bool` 分支
提到剥虚部之前，标量路径重新服从同一个定义。

**文档更正（不改行为）**：§17.10.11 曾写「`np.percentile(bool)` 报错，
clvt 报错是对的」。实测 clvt **不报错**且给出正确值（`1.0d0`）。
numpy 抛错只是因为它内部要算 `hi - lo`；本库在 float64 缓冲上取位，
没有理由为了对齐实现细节去拒绝一个算对的答案。契约已在 §17.10.12 写明：
`median`/`percentile`/`quantile` 对 bool 返回 `float64`（有意偏差），
`ptp`/`gradient`/`ediff1d` 报错（语义本身就是减法）。

**这一轮同时确认无误的**（记在 §17.10.12，免得下次重做）：归约器的
`:axis`/`:keepdims`/`:axis '(0 2)`/`:axis '()` 全矩阵；`bool ⊗
{int8,int64,float32,float64}` 与 `concatenate`/`stack`/`where` 的提升表；
42 个算子 × 8 种布局的 352 次布局不变性；npy（`|b1`）与 `:lisp` 的往返。

**验收**：`bool-dtype-test` 432 断言全过（新增 L 组 22 条）；65 套件全绿
（严格模式）；`check_docs.py` 221 锚点全解析；编译零 WARNING。

---

## 2026-10-30（三十七）— fix(bool): 第三遍审计（换轴：结果 dtype 与物理字节）

> 契约见 `CONVENTIONS.md` §17.10.10 / §17.10.11；决策过程见 `HISTORY.md` H.3.30。
> 前两遍的轴是「值」与「入口」，这一遍的被断言对象是**结果 dtype 本身**，
> 外加一条「`:bool` 结果的物理字节必须 ∈ {0,1}」的不变量扫描。

**破坏性变更（四处，全部是「原来给错答案」）**：

| 旧 | 现在 | 影响面 |
|---|---|---|
| `(vt-negative bool数组)` → `:bool`，字节 **255** | 报错（可显式 `:dtype :int8`） | 只影响布尔输入 |
| `(vt-atan2 int数组 int数组)` / `(vt-atan2 bool…)` → `:int64` / `:bool`，值 **全 0** | 一律 `:float64`（`float32` 仍进 `float32` 出） | 只影响整数/布尔输入；浮点路径逐位不变 |
| `(vt-convolve b b)` / `(vt-correlate b b)` → `:float64` 数值和 | → `:bool`，按逻辑与/或（逐值与 numpy 一致） | 只影响两侧都是 `:bool` 的调用 |
| `(vt-cross b b)` → `:bool` `(1 1 1)` | 报错（numpy 同样拒绝） | 只影响布尔输入；显式 `:dtype` 仍可用 |

**其余两处**：`(vt-pow b 2)` 由 `:float64` 改为 `:int64`（与 §17.10.7
偏差表记的一致，也与 numpy 一致）；`(vt-rem b b)` 由 `:bool` 改为 `:int8`
（与同族的 `vt-mod` / `vt-div` 对齐）。

**为什么这一遍才发现**：`vt-to-list` 会把 `0.785` 显示成 `0`，所以
「值不对」既可能是算错、也可能是「算对了但类型存不下」。前两遍默认了前者。
另外 `vt-negative` 的 255 是 `vt-bit-not`（254/255）那个坑的第二次踩 ——
根因不是某个算子，而是「负值经 `%coerce-uint8` 写进 `(unsigned-byte 8)`」
这条路径本身。

**没改的**：`b ** 0` 仍返回 `:float64`（numpy 给 `int64`，值相同）；
`vt-sign(b)`、`b ** b` 两条偏差保持原样（§17.10.7）。

**验收**：`bool-dtype-test` 410 断言全过（新增 K 组 22 条）；65 套件全绿
（严格模式）；`check_docs.py` 220 锚点全解析；编译零 WARNING。

---

## 2026-10-30（三十六）— fix(bool): 布尔全链路审计（两遍）—— 六类静默错值 + `T`/`NIL` 值位置

> 契约见 `CONVENTIONS.md` §17.10（17.10.1–17.10.9）；实现细节见 `HISTORY.md` H.3.29。
> 本轮的六类问题全部有 numpy 2.3.5 逐例对照（152 + 66 条差分用例），
> 另加两轮手工探针扫「字面量传不传得进去」。

**破坏性变更（三处，都是「原来给错答案」）**：

| 旧 | 现在 | 影响面 |
|---|---|---|
| `(vt-bit-not bool数组)` → `(254 255)` | → 逻辑非 `(0 1)`，dtype `:bool` | 只影响布尔输入；显式 `:dtype :int8` 仍是字节取反 |
| `(vt-matmul b b)` / `dot` / `outer` / `inner` / `tensordot` / `einsum` → `:float64`，值 `[[2 1][1 2]]` | → `:bool`，值 `[[T T][T T]]`（按与/或缩并） | 只影响「全部输入 `:bool` 且未显式给 `:dtype`」；显式 `:dtype` 仍是数值路径 |
| `(vt-map (lambda (x) 2) bool数组)` → `(2 2)` | → `(1 1)` | 只影响 `:bool` 结果；`uint8` 的 200 不受影响 |

**修好的六类静默错值**：

| 表达式 | 修复前 | 修复后 | numpy |
|---|---|---|---|
| `(vt-map (lambda (x) 0.5d0) b)` | `(0 0)`（截断） | `(1 1)` | `True` |
| `(vt-bit-not b)` | `(254 255)` | `(0 1)` | `invert` |
| `(vt-histogram b)` | 报 `NIL is not of type NUMBER` | `(2 0 … 0 2)` | 同左 |
| `(vt-diff b)` | 报「布尔减法未定义」 | 相邻异或 | `np.diff` |
| `(vt-trapz b)` | **0** | `1.5d0`（`:float64`） | `1.5` |
| `b @ b` | `:float64` 值 `2` | `:bool` | 同左 |

**修好的「传不进去」**（`T`/`NIL` 作为值）：

```lisp
(vt-insert b 0 t)                       ; 原来 T fell through ETYPECASE
(vt-pad b '(1 1) :constant-values t)    ; 原来 invalid constant-values: T
(vt-pad b '(1 1) :constant-values '(nil t))
(vt-where b t nil)                      ; 原来 ETYPECASE；现在结果 :bool
(vt-fill-diagonal m t)                  ; 原来静默只填对角第一个元素
(vt-append b t)
(vt-map f b1 b2 b3 b4 b5 b6)            ; 6 输入：原来 no key（ECASE 的 T 子句）
```

**`NIL` 的规则（唯一需要记的）**：

| 位置 | `NIL` 的含义 |
|---|---|
| `vt-from-sequence` 顶层 | shape `(0)` 的空数组；`:dtype :bool` 下才是叶子 |
| `ensure-vt`（无 `:dtype`） | 仍是空表（300+ 内部调用点依赖） |
| `ensure-vt … :dtype :bool` | 值 0 |
| `vt-put` / `vt-fill` / `vt-insert` / `vt-pad` / `vt-where` 的**值位置** | 值 0 |

**没改的三处**（numpy 同样报错，clvt 报错是对的）：`np.percentile(bool)`、
`np.gradient(bool)`、`np.ptp(bool)`。另外 `a[mask]`（不走 `vt-ref`）、
`b ** 2` 的 dtype、`vt-sign(b)` 三条偏差仍记录在 §17.10.7。

**验收**：`bool-dtype-test` 388 断言全过；65 套件全绿（严格模式）；
两个差分探针 496/496、405/405；`check_docs.py` 218 锚点全解析；编译零 WARNING。

---

## 2026-10-29（三十五）— feat(bool): 布尔值「写得进去」—— `vt-ref` 布尔化、`vt-view` 放行 bool→uint8/int8、`vt-from-sequence` 认 T/NIL 字面量

> **取代（三十四）中的两条结论**：`vt-ref` 不再返回 0/1 字节（改返回 `T`/`NIL`）；
> `(vt-to-list b :bool t)` → `(vt-from-sequence … :dtype :bool)` 这条往返**现在成立**。
> 括号内的历史叙述保留原样，以避免改写流水账。

**破坏性变更（四处）**：

| 旧 | 现在 |
|---|---|
| `(vt-ref bool张量 i)` → `1` / `0` | → **`T` / `NIL`**（与 `vt-item` 一致） |
| `(setf (vt-ref b 0) nil)` → 报 `can't be converted to (UNSIGNED-BYTE 8)` | → 接受 `T`/`NIL`/任意数值（按 `%bool-flag` 真值化） |
| `(vt-from-sequence '(t nil t) :dtype :bool)` → 报「不规则嵌套」 | → `[T NIL T]`；`NIL` 一律是叶子 |
| `(vt-view b :dtype :uint8)` → 报「仅支持复数两对」 | → `[1 0 1]`，形状不变 |

**不受影响**：`vt-data` 仍是 `(unsigned-byte 8)` 的 0/1 缓冲区；`vt-to-list`
默认仍是 `(1 0)`；npy 落盘的 `|b1` 字节不变；非 `:bool` dtype 的一切行为不变
（`(vt-from-sequence (list 1 nil 3))` 仍报「不规则嵌套」）。

**为什么改**：

1. `vt-ref` 与 `vt-item` 是同一类接口（按坐标取一个元素），给不同表示本身
   就是错的。`arr[i]` 在 numpy 里返回 `np.bool_`，Lisp 的同位对应物是 `T`/`NIL`。
   改完 `(when (vt-ref mask i) ...)` 直接可用 —— 而 `0` 在 CL 里是真值，
   漏写一次 `(= 1 ...)` 就变成**恒真判断**。
2. 布尔化之后 `(setf (vt-ref b 0) (vt-ref b 1))` 成了最自然的写法，原实现走
   `(coerce value (vt-element-type vt))` 会直接报类型错误 —— 等于「读得出来、
   写不回去」。改用 `%bool-flag` 归一，顺带接受 `0.5` / `-0.0` / NaN，与
   `vt-fill`、`vt-put`、`vt-astype` 的布尔语义对齐。
3. `vt-from-sequence` 是 Lisp 列表 → 张量的主入口，不认 `T`/`NIL` 就意味着
   布尔张量只能从 `0`/`1` 或者 `vt-astype` 绕进来。
4. `bool`、`uint8`、`int8` 都是 1 字节，`view` 到整型本该逐位等价。

**`vt-view` 为什么只放行 `bool → uint8/int8`，反向不放**：numpy 的
`uint8.view(bool)` 是**纯字节直译** —— `np.array([1,2,0], dtype=np.uint8)
.view(bool)` 打印 `[True True False]`，但值 `2` 原封不动留在数组里。照搬就
等于允许「bool 数组里存着 2」，直接违反 0/1 不变量（打印、`vt-item`、npy 往返、
`sum` 升 int64 全依赖它）。反向请用 `vt-astype :bool`：真值化 `2 → 1`，
语义明确，不该藏在 `view` 这个名字底下。

**迁移**：

```lisp
;; 旧
(= 1 (vt-ref mask i))
(setf (vt-ref b 0) 1)
(vt-from-sequence (list 0 1 0) :dtype :bool)

;; 新：逻辑值直接当布尔用
(when (vt-ref mask i) ...)
(setf (vt-ref b 0) t)                ; 0/1 也仍然接受
(vt-from-sequence (list t nil t) :dtype :bool)

;; 需要 0/1 数值时
(if (vt-ref b i) 1 0)                ; 或内部用 clvt::%vt-ref-raw
```

**内部改动**：`vt-ref` 拆成两层 —— `%vt-ref-raw` 保留原实现（永远 0/1，
供库内数值代码使用），`vt-ref` 在它外面加一层 dtype 判断。`src/` 内 23 处
读点全部改调 `%vt-ref-raw`（`linalg` 的 Gram-Schmidt、`extensions3` 的协方差/
相关系数/叉积、`reduce-stats` 的 `vt-interp` 单调性检查等），理由是这些点要的
是数字，靠「bool 不会流进来」的假设兜底迟早失效；分流后数值热路径连一次 `eq`
都不用付。

**连带修的三处静默失败**（都不报错、只是结果不对，比报错难查得多）：

| 场景 | 旧行为 | 现在 |
|---|---|---|
| `(vt-put b 0 t)` | 静默空操作（`t` 不匹配任何分支，`n-values`=0 直接返回） | 写入 1 |
| `(vt-put b 1 nil)` | 静默空操作（`nil` 落进 `listp` 分支被展平成空表） | 写入 0 |
| `(vt-put b '(0 1 2) '(t nil t))` | 得到 `(1 1 1 0)`（`vt-flatten-sequence` 把值列表压成 `(T T)` 后循环复用） | 得到 `(1 0 1 0)` |

`vt-flatten-sequence` 因此新增 `&key bool`（默认行为不变，仍是 `(T T)`），
`vt-put` 在目标为 `:bool` 时自动启用。

**`vt-view` 顺带修的老 bug**：`(vt-view a :dtype dt)` 省略 shape 时，未指定被
当成 `nil` 形状（0 维），对任何多元素张量都报「元素总数不一致 (旧 4, 新 1)」。
现在用 `:none` 哨兵区分「未指定」与「显式 nil」，未指定即保持原形状，与 numpy
的 `a.view(dt)` 一致。

**仓库卫生**：删掉 `git add -f *` 带进来的 `tmp/{cases,cases2}.json`、
`tmp/expected{,2}.txt` 与 `test/__pycache__/gen_probes.cpython-312.pyc`，恢复
`/tmp/` 忽略规则并补 `*.pyc`。这些对任何流程都不必要：探针期望内嵌在
`differential-probes-test*.lisp` 里，`tmp/` 只在 `gen_probes.py compare`
子命令里用。

**测试**：`bool-dtype-test` 增至 **327** 条（新增 I 组 33 条：布尔字面量构造 /
0 维标量 / `NIL` 叶子 / `vt-view` 双向拒绝 / `flatten :bool` / `vt-put` 布尔
标量；H 组由 43 扩到 55：`vt-ref` 布尔化与 setter）；两个差分探针
`496/496` + `405/405`；65 套件全绿。契约写进 `CONVENTIONS.md` §17.9，
分层表的 `vt-ref` 行改归逻辑视图并新增 `%vt-ref-raw` 行。

## 2026-10-29（三十四）— feat(bool): 布尔值的「逻辑视图」改回 Lisp 惯例，run-tests 自动刷新探针基线

**破坏性变更（两处，都在解包/显示层）**：

| 旧 | 现在 |
|---|---|
| `(vt-item bool张量)` → `1` / `0` | → **`T` / `NIL`** |
| 打印 `[ True, False]` | → `[   T,  NIL]` |

**不受影响**：`vt-to-list` 默认仍返回 `(1 0)`；`vt-ref` / `vt-data` 仍返回 0/1
字节；npy 落盘的 `|b1` 字节不变。

**为什么改**：

1. `np.bool_(True).item()` 返回 Python 的 `True`，Lisp 里与 `bool` 同位的对应物
   就是 `T` / `NIL`。此前 `(when (vt-item (vt-all m)) ...)` 不成立，必须写
   `(= 1 ...)`；漏写一次就变成**恒真断言**（上一轮 9 处 `(assert (vt-all ...))`
   正是这么来的）。
2. `T` / `NIL` 在 `*print-readably*` 下能被 `read` 回来；`True` 读回来是当前包
   临时 intern 的未绑定符号（`common-lisp` 里没有 `TRUE` 这个名字）。
3. 本库 0 维形状本来就印 `shape:NIL`（Lisp 惯例，不是 numpy 的 `()`），值却用
   Python 惯例，同一次打印里两套风格。

**迁移**：

```lisp
;; 旧
(= 1 (vt-item (vt-all m)))
;; 新：直接当布尔用
(vt-item (vt-all m))

;; 需要 0/1 数值时显式归一
(if (vt-item x) 1 0)
```

**`vt-to-list` 为什么没跟着改**：`NIL` 在 CL 里既是布尔假、又是空表，
`(vt-from-sequence (list t nil t))` 无法区分「元素 NIL」与「嵌套的空列表」
（实测直接报「不规则嵌套」）。若 `vt-to-list` 返回 T/NIL，
`(vt-from-sequence (vt-to-list x))` 这条往返就永久不成立 —— 而它的契约恰恰是
「可重建的数值序列」。故默认保持 0/1，另提供 `(vt-to-list b :bool t)`。

**`run-tests.sh` 新增「探针基线」阶段**：跑套件前按本机 numpy 重生成两个差分
探针的期望表（`gen_probes.py` / `gen_probes2.py`）并保留结果。换机器 / 换 numpy
后不再因为第 16 位有效数字的格式化差异误报红。四个边界：

* 语义漂移（不是格式化）仍由 `check_key_sync.py` 在**此之前**拦下并退出；
* 刷新时报告改动行数，并明说「本轮是自洽基线，不是对提交期望的回归」；
* `--no-regen-probes` / `CLVT_SKIP_PROBE_REGEN=1` 走严格模式（CI 建议）；
* 生成器的中间产物走临时目录，跑完删除，工作区不会多出 `tmp/`。

**测试**：`bool-dtype-test` 增至 **282** 条（新增 H 组：视图分层 / 0-1 物理视图 /
往返不变性 / 打印字面量 / `vt-extract`·`vt-compress`·`vt-bincount`·`vt-histogram`
四个内部消费点 / `%truthy-p` 与 `%item-number`）；另有 8 个套件的 18 条断言
随契约更新。契约写进 `CONVENTIONS.md` §17.8。

**验收**：65 套件全绿（详见本轮运行报告）；`check_docs.py` 205 锚点全解析。

---

## 2026-10-29（三十三）— feat(dtype): 引入 `:bool` 布尔 dtype，比较/逻辑/谓词类函数返回真布尔

**行为变化（唯一一处）**：`vt-=` / `vt-/=` / `vt-<` / `vt-<=` / `vt->` / `vt->=`
以及逻辑、谓词、`any`/`all`、`isnan` 族、`isclose`、`isin`/`in1d` 的**缺省结果
dtype 由 `:int8` 变为 `:bool`**，值仍是 0/1。这是新增 dtype 带来的升级点：

| 旧写法 | 现在 |
|---|---|
| `(vt-= a b)` | 返回 `:bool` 张量（打印为 `True`/`False`） |
| `(vt-= a b :dtype :int8)` | 与升级前逐位相同（逃生门，全部布尔类函数都支持） |
| `(vt-+ (vt-= a b) (vt-= c d))` | 现在是**逻辑或**（NumPy 语义），不再是 2 |
| `(vt-- (vt-= a b) (vt-= c d))` | 现在**报错**（NumPy 的 `bool_` 不支持减法） |

**背景**：库里有九类函数按 NumPy 语义必须返回 `bool_`，此前统一用 `:int8` 的 0/1
承载，靠文档一句「语义上等价于布尔」维持。两个漏洞文档堵不住：① **算术泄漏** ——
掩码能继续当整数算（`True+True=2`、`sum` 得 `int8` 且会溢出），而 NumPy 的
`bool_` 在算术里走逻辑运算、`sum` 升 `int64`；② **类型不可区分** —— 布尔掩码与
真的 `int8` 数据 dtype 相同，`vt-where` / `vt-compress` 无法校验入参语义。

**做了什么**：

- `:bool` 成为第九种非复数逻辑 dtype，物理存储 `(unsigned-byte 8)`、1 字节，
  合法值只有 0/1（与 `np.dtype(bool).itemsize == 1` 一致）；
- 「数值 → 布尔」走**真值化**而不是整数截断：`0.5 → True`、`nan → True`、
  `-0.0 → False`、复数任一分量非零即真；`vt-astype` 在进截断表之前分流；
- 类型提升对齐 NumPy：`bool ⊔ bool → bool`、`bool ⊔ int8 → int8`、
  `bool ⊔ float32 → float32`；弱标量 `bool + 1 → int64`、`+ 1.0 → float64`；
  `sum/prod/cumsum → int64`、`mean/std → float64`、`max/min → bool`；
- 布尔参与算术时按 NumPy 分派：`+` 逻辑或、`*` 逻辑与、`-` 报错、
  `/` 升 `float64`、`//` 与 `%` 落 `int8`（详见 `CONVENTIONS.md` §17.3）；
- 打印 `True`/`False` 且表头 `dtype:BOOL`；npy 用 `|b1` 描述符，与
  `np.load` / `np.save` 直接互通；
- 契约写进 `CONVENTIONS.md` §17（真值语义 / 类型提升 / 算术语义 / 函数清单 /
  掩码消费方 / 向后兼容），§3.1 的 dtype 表同步；`HISTORY.md` H.3.26 记过程。

**example/ 的连带修复**：把 `example/` 下两个脚本跑进自动化后，翻出三处
**示例自身**的缺陷（实现对、示例错，但此前一直静默）：

- `example.lisp` 里 9 处 `(assert (vt-all X))` 恒真 —— `assert` 只判 NIL，
  而 `vt-all` 返回 VT 对象，任何 VT 对象都不为 NIL。这 9 条断言从 `:int8`
  时代起就没检查过任何东西。改用 `vt-all-true-p` 解包成 Lisp 布尔；
- 解开后立刻暴露一条错期望：全 NaN 时 `nanmax`/`nanmin` 的期望被写成 `±Inf`，
  而 numpy 与 clvt 都返回 **NaN**；
- 基准里 `(vt-extract a mask)` 参数顺序反了（应为 `numpy.extract(condition, arr)`），
  测出来的不是 `a[mask]`，还把 216783 µs 的错测量记成了 `extract` 的性能。
  改为 `(vt-extract mask a)` 后是 21185 µs。

新增门禁套件 `example-smoke-test`（`test/example-smoke-test.lisp`，已进
`DEFAULT_ORDER`），只跑断言不跑基准；断言区另补 9 条 `:bool` 契约（87 → 96）。
顺带修掉 `*print-circle*` 为真时布尔张量打印出 `#1=True, #2=False, #1#, #2#`
的噪音（`%format-number` 返回字面量字符串所致，加 `copy-seq`）。

**验收**：`bool-dtype-test` 239 断言（含 52 例 numpy 2.3.5 逐例对照）全过；
差分探针 `496/496` + `405/405`；全量 **65 套件全绿**（新增 `example-smoke-test`
98/98）；`check_docs.py` 204 锚点全解析。

---

## 2026-10-29（三十二）— compat: SBCL 2.6.9 验证（sb-simd 的 512 实测否决 + 一处 NaN 类型回归）

**行为变化**：对外 API / 数值一律不变。修的是**在 SBCL 2.6.9 上**的一个
崩溃：`vt-nanvar` / `vt-nanstd` 遇到 `mean` 为 NaN 的输入（如 `[+inf, -inf, 1]`）
抛 `TYPE-ERROR`，2.6.8 正常。修后 `nan-fused-test` 302 项全过，63 套件全绿。

**背景**：上游 2.6.9 的 NEWS 首条即「AVX512 support to SB-SIMD」，于是有两问：
① 能不能直接调 sb-simd 的 512 替换自写内核；② 我们的内核在新版本上还准不准。

**结论一：能调，但只有自写内核的 0.6×，不换。**

`(require :sb-simd)` 后 `sb-simd-avx512f` 的 `f64.8` 一族与 AVX2 的 `f64.4`
完全同构，`sb-ext:simd-pack-512` 是内建类 —— **不需要 VOP、不需要机器码**，
`define-blocked-gemm` 的算子参数换成 `sb-simd-avx512f:f64.8*` 就能生成 512
位微块。但 1000³ 单线程实测：

| 形态 | ms | GFLOPS | 相对自写内核 |
|---|---:|---:|---:|
| **自写机器码 8×16** | **27** | **74** | 1.00× |
| sb-simd 512，8×8 | 46 | 44 | 0.59× |
| sb-simd 512，8×16 | 53 | 38 | 0.51× |
| sb-simd 512，4×16 | 66 | 30 | 0.41× |
| sb-simd AVX2 4×8 | 80 | 25 | 0.34× |

反汇编给出的原因只有一条：`f64.8-fmadd` 是**纯函数 + 值语义**，结果必须回落到
固定的 `ZMM0`，于是每条 `VFMADD213PD` 后面都跟一条 `VMOVUPS`（40 条 FMA / 34 条
搬移），前端吞吐被吃掉近一半。自写内核的 `vfmadd231pd` 拿累加器当目标操作数，
一条顶两条。zmm 的 VOP 在 2.6.9 上**依然写不出来**（无 `double-avx512-reg`
这类 SC、`define-instruction` 仍为 NIL、无 EVEX 编码表），结论与上一轮一致。

**结论二：一处回归，根因是编译器推断失真。**

2.6.9 起 `(* x x)` 的返回类型被推断成「非负浮点」`(FLOAT (0.0))`，而 **NaN 不
属于该类型**。`%cx-nan-sumsq` 的返回值因此被压成
`(OR (MEMBER 0.0d0 0.0) (FLOAT (0.0)))`，调用点按它插的返回值检查直接把
quiet NaN 判成 `TYPE-ERROR`。

**改法**（`src/complex-ops.lisp`，1 行声明 + 5 行注释）：

```lisp
(declaim (ftype (function (t t t) (values float)) %cx-nan-sumsq))
```

把 fdefn 上记录的类型钉回 `float`（NaN 是 float），检查点消失，循环内的数值
推导与 `(speed 3)` 代码质量不受影响 —— **零运行时开销**。`the` /
`sb-ext:truly-the` / `(safety 0)` 全部无效（声明与推断取交集，NaN 仍被排除），
`notinline` 有效但每元素多一次调用。契约见 `CONVENTIONS` §16.30.6：**凡是允许
返回 NaN 的内部浮点函数，都要显式写 `ftype`**。

**保留判定**：自写机器码内核在 2.6.9 上功能与性能**完全不变**（49/49、
1000³ 27 ms，与 2.6.8 逐位一致），不依赖编译器版本 —— 这是**不**切回 sb-simd
的第二个理由。数据与排查过程见 `HISTORY.md` §H.3.25，对照见 §7.5。

**测试**：未新增用例 —— `nan-fused-test` 的 ±Inf 用例本来就是这条的可执行版本
（它在 2.6.8 上一直绿，是版本升级把它变成红的）。

---

## 2026-10-29（三十一）— perf: f64 GEMM 改走运行时发射的 AVX-512 内核

**行为变化**：单线程 f64 `vt-matmul` 快 3 倍（1000³：85 ms → 28 ms），
数值只在 `n % 4 ≠ 0` 的**最后 `n % 4` 列**差 1~2 ulp（512 侧更精确，
见下「契约」）。新增 49 项测试 + 护栏 G9，63 套件全绿。

**问题**：前面几轮已经把 GEMM 从「比 numpy 慢十倍」推到「约 25% of numpy」，
分块、线程、内存都没问题，卡在**指令层**：sb-simd 只到 AVX2（256 位），每个 k 步
的「广播 + FMA」要拆成 `vbroadcastsd` + `vfmadd231pd` 两条，内层每 8 次 FMA
吃掉 47 个 uop，前端先饱和，实测 16 FMA/迭代要 3.98 cycles（FMA 端口理论
下限 4），加载入不加 cycles —— 瓶颈是**指令条数**而不是带宽或端口。

**为什么没写 VOP**：SBCL 2.6.8 的目标映像里 `sb-assem:define-instruction`
根本不存在，且没有 `simd-pack-512` 的 primitive type、没有 zmm 的 TN。
常规做法（`(sb-assem:inst 'vmovupd ...)`）要求目标符号已 intern 到
`SB-X86-64-ASM` 且编码表里有 EVEX —— 都没有。于是直接**自己发字节**。

**改法**（新文件 `src/gemm-avx512.lisp`，316 行）：

- 一个只覆盖所需指令的字节发射器（`%e-evex` / `%e-vmovupd-{load,store}` /
  `%e-vfmadd231pd` / `%e-kmov-to-k` / `%e-lea` / `%e-jcc-rel32` …）；
  EVEX 的字段语义是拿 gas 生成 32 个寄存器的组合**逐字节反推**出来的，
  不是照手册写的（`R̄/R'/B̄/X̄/v̄/V'` 全是取反位，唯独 v 字段那 4 位是
  「寄存器号左移一位再取反」）；
- 内核形状 8 行 × 16 列：`zmm2..zmm17` 是 16 个累加器，每个 k 步 16 条
  `vfmadd231pd`，A 的广播直接取内存操作数 `{1to8}`（不额外占指令）；
- 列尾（`n % 16`）用**掩码内核**（k1/k2 各管低/高 8 列，`{k}{z}` 载入
  B，写回也带掩码 ⇒ 不越界写）；行尾（`m % 8`）交回 sb-simd 微块
  （≤7 行，占整块算力 1% 以内，不值得再发射一份 MR=4 的内核）；
- `sb-sys:allocate-system-memory` + `mprotect(PROT_READ|WRITE|EXEC)` +
  `sb-alien:sap-alien` 得到可直接 `alien-funcall` 的函数对象，首次调用时
  发射一次并缓存；
- 平台探测只读 `/proc/cpuinfo` 的 ` avx512f `（Linux 填这张表前已用
  XGETBV 过滤过）；**不发探测代码**——探测失败是 SIGILL 杀进程，不是返回 NIL。
  探测不到时 `avx512-gemm-kernels` 返回 NIL，调用方原样走 sb-simd。

**数值语义**（`CONVENTIONS` §7.3）：累加器**从 C 的值开始**（先 `vmovupd`
载入、末尾原样写回），k 仍按 0,1,2,… 递增且单次舍入（FMA），与 sb-simd 路径
同一条舍入链。曾经试过「清零 + 末尾 `vaddpd`」—— 快一点点，但对拍出现
`-4.99e-16` 而非 0，违反逐位一致，已改回。

**性能**（AMD EPYC 9K65 / Zen5，`taskset -c 2` 单线程，取 5 次最优；
numpy 为同一核上的 OpenBLAS AVX-512）：

| 规模 | numpy | 改后（AVX-512） | 改前（sb-simd） | 相对 numpy | 加速 |
|---|---:|---:|---:|---:|---:|
| 500³ | 3.05 ms | **3.00 ms**（83 GF） | 10.00 ms（25 GF） | **1.02×** | 3.3× |
| 1000³ | 21.67 ms | **28.00 ms**（71 GF） | 85.00 ms（24 GF） | **0.77×** | 3.0× |
| 1500³ | 71.42 ms | **96.00 ms**（70 GF） | 320.01 ms（21 GF） | **0.74×** | 3.3× |

1000³ 上 `%simd-matmul-f64-512` 与 `%simd-matmul-f64` **逐位相同**
（1000000 个元素 0 差异），`vt-matmul` 与 `vt-einsum` 也逐位相同。

**契约**：`CONVENTIONS` §7.3 新增「f64 GEMM 的逐位相同边界」——sb-simd 的
列尾最后 `n % 4` 列走标量 `(incf c (* a b))`（两次舍入），512 内核用 FMA
（一次舍入），所以这些列差 1~2 ulp；测试据此把「逐位相同」放宽成
「差异只能落在 `j ≥ n - (n % 4)` 且相对误差 ≤ 1e-15」，`n % 4 = 0` 时
自动退化为严格逐位。§7.4 的 GEMM 一行补上 512 内核及其回退条件。

**测试**：新增 `test/gemm-avx512-test.lisp`（49 项）——16 种尾块形态、
视图/对齐（偏移 1/3/5 元素、跨步、转置、3D 批量）、**写边界**（大缓冲
前后放哨兵 + C 指针故意非对齐到 264 字节，验证内核只写它该写的元素）、
量级（1e150²、1e-160² 次正规、全零 ±0.0、NaN 位型）、开关回退。
新增护栏 G9（`t(512) ÷ t(sb-simd) ≤ 0.6`，实测 0.33）防「内核没加载/
开关被关」的静默退化：那种情况结果全对，只是慢 3 倍。

---

## 2026-10-28（三十）— perf: `vt-astype` 的切片/跨步视图改「逐 run 单趟」

**行为变化**：无。10×10 转换矩阵的取值 / dtype / 形状 / 副本语义一律不变，
663 条 astype 断言 + 62 套件全绿。

**问题**：上一轮把非连续视图统一接到 `%gather-contiguous`，对**转置/翻转**
视图（内层 run = 1）收益明确；但**切片/跨步**视图（内层 run ≥ 16，如
`a[::2,:]`）走的仍是「先收集成连续临时副本 → 再整数组转换」，也就是
两趟全量流量 + 一块全尺寸临时分配。同一份数据，转置视图一趟就够、
切片视图要两趟，`astype` 在这类视图上因此比 numpy 慢 2~4 倍。

**改法**：把 `vt-astype` 的 10×10 特化表提成一个局部函数
`%cast-run (dst p q len)`，让**两条落点共用同一张表**（不再「快路径各写一套」）：

- 同 dtype 的 run 走 `%memcpy`（库内唯一块拷贝实现，§2.2 / §16.4.1）——
  原先 10 份 `(replace (the (simple-array …)))` 合并成一处；
- 跨 dtype 的 run 走逐元素特化 cast，落点**直接是目标数组**；
- run 太短（< `+astype-fuse-min-run+` = 16）或未知元素类型，仍走
  `%gather-contiguous` → 递归 astype。

`vt-copy` 的 `:dtype` 分支本就委托 `vt-astype`，因此自动拿到同一条路。

**性能**（1000×1000 f64，视图 = `a[::2,:]` 即 500×1000，SBCL 2.6.8，同机取最优）：

| 场景 | 改前 | 改后 | 倍数 |
|---|---:|---:|---:|
| `vt-astype a[::2,:]` f64→f32 | 1.23 ms | 0.33 ms | **3.7×** |
| `vt-astype a[::2,:]` f64→i32 | 1.36 ms | 0.87 ms | 1.6× |
| `vt-astype a[::2,:]` c128→c64 | 1.63 ms | 0.64 ms | 2.5× |
| `vt-copy a[::2,:] :dtype :float32` | 0.82 ms | 0.33 ms | 2.5× |
| `vt-astype a[::2,:]` f64→f64（同型，纯块拷贝） | 0.23 ms | 0.24 ms | 1.0 |

连续视图与转置视图的耗时不变（0.55 ms / 1.9 ms）—— 两条路都不经过新代码，
这也是「改快路径不许动老路径」的可执行验收。

**为什么不在转置视图上也融合**：转置视图的 run = 1，逐 run 等于逐元素，
调用开销会吞掉全部收益；而 `%gather-contiguous` 的末轴展开已经是该类
视图的下限（同数据手写两层循环同速）。两条路的分界因此由
`+astype-fuse-min-run+` 明确写死，与 `vt-copy-into` 的块拷贝门槛取同一个数。

**契约**：`CONVENTIONS` §2.2 补「按 run 融合」的落点等价性要求；
§7.4 的「类型转换」一行改成两条落点；新增护栏 G8
（`astype(切片) ÷ astype(同规模连续) ≤ 1.8`，双趟实现是 4×）。

---

## 2026-10-27（二十九）— perf: 矩阵乘 / 连续化 / 类型转换 / 归约的搬运层重做

**行为变化**：无。结果与优化前**逐位相同**（GEMM 的 k 累加序、归约的 r 累加序
都没变），只有耗时变了。

**起因**：按「1M 元素 / 1000×1000 / float64」把非复数路径全量量了一遍，
有四条路明显不在算力上限上：

| 操作 | 改前 | 改后 | 倍数 |
|---|---:|---:|---:|
| `vt-matmul` 1000³ f64（默认线程） | 98 ms | 12~16 ms | **6~8×** |
| 同上，单线程 | 235 ms | 82 ms | 2.9× |
| `vt-sum :axis 0` 1M f64 | 3 ms | 0.45 ms | 6.7× |
| `vt-amax :axis 0` 1M f64 | 20 ms | 1.7 ms | 12× |
| `vt-sum :axis 0` 1M int16 | 7 ms | 0.5 ms | 14× |
| `vt-astype` int8→f64 1M | 8 ms | 0.7 ms | 11× |
| `vt-contiguous` 转置 1M | 3.4 ms | 1.2 ms | 2.8× |
| `vt-copy` 转置 1M | 2.6 ms | 1.2 ms | 2.2× |

**四件事**：

1. **GEMM 换分块内核**：旧内核是「1 行 × VW 列」的 gaxpy，C 每个 k 步读+写
   一遍、B 的每行被 m 次重读（1000³ 时 B 过内存 8 GB）。改为
   寄存器微块（f64 4×8 / f32 4×16）+ MC/KC/NC 分块 + 指针步进内层。
   单线程因此贴住 AVX2 峰值（f32 恰好快 2 倍 = 位宽比，可作旁证）。
2. **并行调度改二维网格 + 线程数上限 16**。只切行时每个线程都要过一遍 B，
   线程数翻倍流量也翻倍；31 线程实测比 8 线程慢 3 倍（最坏 87 ms）。
3. **归约新增「行累加」路径**：把内层维整体当累加器、按行推进，替掉
   「固定列、跨 stride 走」的列扫（axis=0 时每元素换一条 cache line）。
   累加次序不变 ⇒ 与旧路径逐位相同；窄整型与「归约轴是连续区间」的多轴
   情形（`axis=(0,1)` 等）也一并接入（多轴 1M f64：9 ms → 0.5 ms）。
4. **`%nan-p` / `%nan-or-inf-p` 改 inline**（定义移到 `util.lisp` 以保证
   先于所有调用点）。形参类型是 `T`，不内联时函数体只能按泛型数值编译，
   `=` 走运行时派发 ≈ 20~30 周期/次，而极值类归约每元素要调 2 次。

**契约**：`CONVENTIONS` §2.2 补「连续化的实现与零拷贝语义」，
§7.3 重写为「GEMM 线程数/网格 + 累加顺序不变」，新增 §7.4
「性能快路径清单」（哪条路在什么时候接管、失败如何回退），
避免下一轮重新踩「快路径与通用路径各写一套」的坑。

**顺带修掉两个自己埋的坑**（都是把归约接进快路径时掐断的）：

1. `(vt-sum int16-3d :axis '(0 1))` 报 `unsupported input dtype` ——
   多轴 clause 只判几何、没判 dtype，窄整型落进了存储级分派表；
2. `(vt-amax int8-array)` 报 `unsupported output dtype` —— 窄整型的
   max/min 保持输入 dtype，而分派表只列了 4 种存储级输出类型。

两处都已修，并补了对拍：8 dtype × 3 shape × 全部轴组合 × 4 算子
+ argmax/argmin，共 592 组与朴素参考**逐位相同**。

**没动的**：`vt-slice`（纯视图构造，0.1 µs）与 `vt-concatenate :axis 0`
（已是整块 memcpy）本来就在最优形态；`vt-sum :axis 1`（内层维 = 1）
受累加依赖链限制，除非改变求和顺序（违反 §7.3）否则无法再快。

---

## 2026-10-26（二十八）— 修：性能护栏里的绝对毫秒阈值不可移植

**行为变化**：无（只改测试）。

**问题**：`complex-elementary-test` 在别的机器上红一条（180/181）。原因是
护栏用了绝对毫秒阈值，而阈值是按开发机标定的：

| 断言 | 开发机实测 | 阈值 | 余量 |
|---|---:|---:|---:|
| `acos c128` | 98 ms | < 200 | **2.0×** |
| `acosh c128` | 97 ms | < 200 | **2.1×** |
| `asin c128` | 123 ms | < 250 | **2.0×** |

慢一倍的机器（笔记本 / 共享 VM / 内存带宽不同）就会顶到阈值，而报出来的
信息是「性能回归」—— 这是**量尺**的问题，不是被测代码的问题（与第三轮
`get-internal-real-time` 只有 1ms tick 那次同类）。

**改法**：全部换成**机器无关的倍率**，两类各 3/6 条：

- **同算子 c128 / f64**：`log ≤ 12×`、`sign ≤ 15×`、`tanh ≤ 8×`
  （修前分别是 43 / 40 / 11.5）；
- **目标算子 / 锚点 `vt-exp`**：`log ≤ 5×`、`tanh ≤ 4×`、`asin`/`acos`/
  `asinh`/`acosh ≤ 10×`（修前是 10.2 / 5.6 / 26.5 / 25.4 / 27.5 / 25.1）。
  选 `vt-exp` 当锚点：同为「一次复数超越函数 + 少量算术」，成本构成相近，
  自身又不在这些断言的保护范围内；比值跨机器只漂 ±30% 左右。

两边都留 ≥3 倍余量、又远低于「退回 CL 回退」时的比值，所以既拦得住真回归，
又不会被机器速度误伤。运行时打印锚点耗时，失败时打印实测比值与阈值。

**契约**：`CONVENTIONS §16.16` 新增第 7 条「性能护栏只能用倍率，禁止绝对
毫秒」，说明两类写法与余量要求，避免下一轮再犯。

---

## 2026-10-25（二十七）— 第二十七轮：nan 族 / `var` / `nan_to_num` 的多趟流水线压成单趟

**行为变化**：有 —— 一处**修错**（两个公开 API 的崩溃）。

| 场景 | 修前 | 修后 |
|---|---|---|
| `(vt-abs z)` / `(vt-angle z)`，`z` 含 NaN | 抛 `FLOATING-POINT-INVALID-OPERATION` | 正常返回（NaN 传播） |

**性能**（1M，SBCL 2.6.8，同进程 3 次取最优）：

| 算子 | 修前 | 修后 |
|---|---:|---:|
| `nanmean` c128 | 19.9 ms | **1.3 ms** |
| `nanvar` / `nanstd` c128 | 49.8 / 49.3 ms | **3.0 / 3.1 ms** |
| `nanmax` / `nanmin` c128 | 22.4 / 22.5 ms | **1.8 / 1.9 ms** |
| `nansum` c128 | 7.0 ms | **1.3 ms** |
| `nan-to-num` c128 | 78.9 ms | **5.5 ms**（对照 `vt-copy` 3.4 ms） |
| `nan-to-num` f64 | 21.9 ms | **1.7 ms** |
| `var` c128 | 27.2 ms | **1.2 ms** |
| `var` f64 | 6.5 ms | **1.1 ms** |
| `nansum` f64 | 8.9 ms | **0.8 ms** |

**根因**：nan 族一直是「掩码 + 单位元替换 + 复用既有归约内核」的张量流水线
（`nanmean` ≈ 7 趟、`nanvar` ≥ 10 趟，每次分配 2~3 个 16 MB 临时），
而同形 float64 早就有单趟快路径（1.4 ms）—— 差价里没有一处是「复数本来就该慢」。
同样的病还有三处：`nan_to_num` 走 `vt-map` + 闭包（每元素一次 funcall + 装箱）、
复数 `var` 没有两趟版（修完后它一度是 `nanvar` 的 8 倍）、
实数 `nansum` 只有通用 `%OP-STEP` 循环（比同族 `nanmean` 慢 12 倍）。

**做法**：补五个单趟扫描 / 直线循环（求和+计数、字典序极值+计数、
Σ\|z−mean\|² 的跳过版与不跳过版、`nan_to_num` 的逐分量替换），
只在 `axis=nil` 的全归约上启用；沿轴版本保持流水线。`vt-nansum` 的门面是
宏生成的，为了不覆盖公开符号（§16.32），给 `DEF-VT-REDUCE-FACADE` 加了
可选的 `:fast` 钩子。

**五条坑**（写进 §16.14.7）：极值扫描不能拿 `a[0]` 当种子（NaN 比较恒假，
`BEST` 会永远停在 NaN 上）；`n=0` 时种子先越界；`%NAN-P` 必须在
`with-float-safe` 里调；快路径必须显式要求浮点 dtype（
`%E3-FLOAT-PREFER-DTYPE` 对整数原样返回 `:int64`，少了这条判据会返回
**NIL 张量**）；宏生成的门面不能覆盖。

**顺带修掉一处公开 API 崩溃**：`%CX-ABS-RUN` / `%CX-ANGLE-RUN` 的特化循环
没有屏蔽浮点陷阱 —— 内核里 `%CX-INF-P` 是 `(> v maxf)`、`%COP-ARG` 要看符号，
UCOMISD 遇 NaN 触发陷阱，于是 `(vt-abs z)` 与 `(vt-angle z)` 在含 NaN 的输入上
直接抛 `FLOATING-POINT-INVALID-OPERATION`（与 §16.14.5 的 `vt-asin` 同类）。

**测试**：新增 `nan-fused-test`（**302 断言**）—— 把「旧流水线」在测试里重写
一遍（只用公开算子），与新内核逐例对拍：6 个算子 × c128/c64 × 11 种输入
（含 `a[0]` 是 NaN、全 NaN、空、单元素、±Inf）× `ddof` 0/1/2，加 0 维输入、
`vt-var`/`vt-std` 的实数与复数、`nan_to_num` 四 dtype 的**逐位**比较
（含 `:out` 与整数输入）、以及「复数一元算子遇 NaN 不抛条件」。
已接入 `run-tests.sh`（61 套件全绿）。

---

## 2026-10-25（二十六）— 第二十六轮：顺序统计量换算法（median 7.7x、topk 79x）

**行为变化**：有 —— 一处**修错**（全 NaN 的复数结果虚部）+ 两处已登记的
已知差异（§16.15 的 D-S1/D-S2）。

| 场景 | 修前 | 修后（= numpy 2.4.6） |
|---|---|---|
| 全 NaN 的 `nanmedian` / `nanmax` / `nanmin` | `NaN+0j` | **`NaN+NaNj`** |

**性能**（1M，SBCL 2.6.8，同进程 3 次取最优）：

| 算子 | 修前 | 修后 |
|---|---:|---:|
| `median` complex128 | 225 ms | **30 ms** |
| `median` complex64 | 205 ms | **26 ms** |
| `median` c128 `:axis 0`（1000×1000） | 152 ms | **37 ms** |
| `nanmedian` complex128 | 216 ms | **30 ms** |
| `topk` f64 k=10 / k=1000 | 174 / 183 ms | **2.2 / 2.9 ms** |
| `topk` complex128 k=10 | 297 ms | **3.0 ms** |

**根因是算法选错，不是常数因子**：三个算子都在「整份排序再取要的那几个」。

1. **`median` / `nanmedian`（复数）** 走 `(vt-sort flat)` 再取下标，而同形
   float64 早在 v0.4.10 就换成了 `%SELECT-DOUBLE`（quickselect）。新增
   `%CX-SELECT-F64/F32`，与 `%SELECT-DOUBLE` 三条设计逐条对齐：三分区（大量
   重复值）、三数取中 + 访问量上限 4N（坏排列）、比较器**不判 NaN**（调用方
   收数时已数过，实测判 NaN 贵 1.9 倍）。另有两处「少一趟」：收数与 NaN 判定
   合成一趟；`nanmedian` 的收数从运行期 `(array-element-type ...)` 决定的
   通用 `AREF` 改成类型特化（这一条本身就值 28 ms）。
2. **`topk`** 改成**有界堆** O(N log k)。硬约束是「必须与 argsort 路径逐位
   一致」（值 + 下标，含并列与 NaN），所以比较器复刻 `vt-sort` 的全序
   （NaN 名次最高、并列按原下标），而不是重新定义语义。阈值 k ≤ 轴长/8 才
   走堆，其余维持 argsort。

**三个坑**（写进 §16.14.6）：`(vt-ravel ...)` 对连续输入是零拷贝视图，
quickselect 就地重排会**改到调用方的数据**；堆根是「最差」的那个，弹出是
「劣→优」，输出必须倒着填（第一版正着填，k=1 全过、k≥3 全错）；`flet` 里
兄弟函数互不可见，`SIFT` 调 `WORSE` 会静默编成「未定义函数」（要 `labels`）。

**测试**：新增 `selection-test`（**2170 断言**）= median/nanmedian 与 numpy 2.4.6
逐例对拍（1/2/3 维 × axis/keepdims，输入由「整数取模 + 一次 double 除法」的
公式生成，两边逐位相同，**期望值文件只有 16 KB**）+ topk 与 argsort 路径
**逐位一致**（6 dtype × 并列/NaN × 1/2/3 维 × `sorted=nil` × 非连续输入）+
全 NaN/单分量 NaN 语义 + 四条护栏。已接入 `run-tests.sh`。

**顺带把 U9 缩小**：`median` / `nanmedian` / `topk` 三项从 U9 移出（已解决），
U9 只剩 `sort` / `argsort` 本身，并保留「LSD 基数排序已实测否决」的结论。

---

## 2026-10-25（二十五）— 第二十五轮：`STYLE-WARNING` 清零 79→4 + 门禁收紧

**行为变化**：无（全是编译期声明与门禁，运行期语义逐位不变）。

**门禁**：编译体检从「只看 `WARNING`」收紧到「`STYLE-WARNING` 必须全部命中
白名单」。判据不是「总数 ≤ 4」（会被「一条消失 + 一条新增」绕过），而是
「两类各自条数钉死 + 总数 == 两类之和」，任何第 5 条直接 FAIL。

**79 → 4 的构成**：

| 类别 | 条数 | 处置 |
|---|---:|---|
| `%cx-dispatch` 的 `ZERO`（unused + `IGNORE for outer scope` + `reading an ignored`） | 38 | 调用点的 `(declare (ignore zero))` 改为宏内 `(ignorable ,zero)` |
| 散落的未使用绑定（`run`/`dts`/`fn-name`/`keepdims`/`ob-s1`/`l-s0`/`v-data`/`m`/`dt`/`what`/`copy`/`assume-unique`/`k`/`j` 等） | 26 | 删除绑定、或按语义补 `ignore`/`ignorable`（4 处是「声明了 ignore 却又读取」的反向错误） |
| **无效的** `declaim inline`（声明晚于调用点） | 5 | 去掉声明（`%nan-p` / `vt-p` / `%copy-default-rng` / `parse-subscript-tokens`）—— 留着只会告警，内联本来就不生效 |
| 零散 `never used`（`axisc` 等赋值后不读） | 6 | 改 `ignorable` 或删赋值 |
| 结构性、留白名单 | 4 | 3 × `&OPTIONAL and &KEY`（`vt-where`/`vt-clip`/`vt-clamp` 的位置参数形态）+ 1 × `VT-P` 布局提示 |

**顺带的一个负结果**：本想做 U9（排序族换 LSD 基数排序），先写原型实测 ——
IEEE754 单调键 + 两半 u32 + 8 趟 8 bit，1M double 上 **107 ms/4 趟**
（8 趟约 200 ms），比现有自底向上归并的 **130 ms 更慢**。原因两条：64 位键在
SBCL 上是 bignum（必须拆两半），且每趟都要把「键 + 值」两条流按 256 个桶散写。
结论写进 §16.37 的 U9，避免下一轮重复这条路。

**测试**：`run-tests.sh` 的编译体检段重写（白名单判据 + 失败时打印全部
STYLE-WARNING 上下文）。套件数与断言数不变，59/59 全绿。

---

## 2026-10-25（二十四）— 第二十四轮：`atanh` 内核化 + 反三角族内联提速 3~5 倍

**行为变化**：有，且是**修错**（极点被算成有限值）。

| 场景 | 修前（CL） | 修后（= numpy 2.4.6） |
|---|---|---|
| `atanh(-1+0i)` | `-177.099+0.785i` | **`-Inf+0i`** |
| `atanh(-1-0i)` | `-177.099-0.785i` | **`-Inf-0i`** |
| `atanh(-0+NaNj)` | `NaN+NaNj` | **`-0+NaNj`** |

**性能**（1M complex128，SBCL 2.6.8，同进程 3 次取最优）：

| 算子 | 上轮 | 本轮 | 同形 float64 |
|---|---:|---:|---:|
| `vt-atanh` | 75 ms | **59 ms** | 15 ms |
| `vt-asin` | 340 ms | **112 ms** | 20 ms |
| `vt-acos` | 300 ms | **60 ms** | 20 ms |
| `vt-asinh` | 317 ms | **92 ms** | 22 ms |
| `vt-acosh` | 315 ms | **62 ms** | 22 ms |

**两件事**：

1. **`atanh` 换内核**。numpy 的 `npy_catanh` 不是 libm，是自己分段的实现
   （实数退化 / 纯虚数 / `|x|` 或 `|y| > 1/eps` 时实部退化成 `Re(1/z)` /
   一般式 `log1p(4|x|/((|x|-1)²+|y|²))/4` + 三种 `atan2`）。三个关键点：
   公式一律用 **|x|、|y|**、最后统一 `copysign`（否则 `x<0` 时 `log1p`
   参数变负：`x=-1+1e-8i` 处会退化成 `log1p(-1)=-Inf`，真值 `-9.5569`）；
   大 `|x|` 必须走 `Re(1/z)`（`atanh(1e300) ≈ 1e-300`，用
   `0.5(log|1+z|-log|1-z|)` 会因两个 1e300 相减得 0）；`±1` 要显式给 ±Inf。
   改完 441 点 **18 → 0**，一次通过。

2. **反三角族的性能主开关是内联，不是公式**。`%cx-do-hard-work`（5 个返回值）
   与 `%cx-clog-large`（2 个）原先没内联，多值返回落栈 + 装箱：
   `vt-acos` 300 ms、`vt-asinh` 317 ms。两个都 `declaim inline` 之后
   **60 / 92 ms**。加上 `vt-atanh` 的内核化，整族 10 个算子在同一张 441 点
   网格上与 numpy 的偏差**全部归零**。

**测试**：`complex-elementary-test` 164 → **181 断言**（网格扩到 12 个算子；
新增 `atanh` 的 NaN/极点安全断言与 2 条绝对护栏；`asin` 的护栏从 500 ms
收紧到 250 ms）。期望值由 `test/gen_complex_elementary_expected.py` 重新生成
（441 × 12 点）。契约见 `CONVENTIONS.md` §16.14.5 第 5 条（内联是主开关）。

---

## 2026-10-25（二十三）— 第二十三轮：三角 / 反三角六族内核化（481 处错值归零）

**行为变化**：有，且是**修错**（分支切割的符号零 + 一处直接崩溃）。

| 场景 | 修前 | 修后（= numpy 2.4.6） |
|---|---|---|
| `asin(-4-0i)` | `-1.5708+2.0634i`（虚部符号反） | **`-1.5708-2.0634i`** |
| `asin(0+NaNj)` | 抛 `NaN is not of type (DOUBLE-FLOAT 0.0)` | **`0+NaNj`** |
| `acos(-4+Infi)` | `-8.0-Inf i`（把 `2x` 当结果） | **`π/2-Inf i`** |
| `tanh(-4+Infi)` | `0.0+0.0i` | **`NaN+NaNj`** |
| `acosh(0+NaNj)` | `NaN+NaNj` | **`NaN+π/2·i`** |

**性能**（1M complex128，SBCL 2.6.8，同进程 3 次取最优）：

| 算子 | 修前 | 修后 | 同形 float64 |
|---|---:|---:|---:|
| `vt-tan` | 203 ms | **84 ms** | 14 ms |
| `vt-tanh` | 192 ms | **51 ms** | 17 ms |
| `vt-asin` | 915 ms | **340 ms** | 20 ms |
| `vt-acos` | 876 ms | **300 ms** | 20 ms |
| `vt-asinh` | 949 ms | **317 ms** | 22 ms |
| `vt-acosh` | 866 ms | **315 ms** | 22 ms |

**根因**：CL 对这六个函数都有复数定义，SBCL 的实现走通用复数算术（单价
200 ns~1.1 µs/元素），而且**在分支切割与 NaN 输入上与 numpy 系统性不一致** ——
441 点网格（21 实部 × 21 虚部，含 ±0/±Inf/NaN/±1d300）上合计 **481/4410**：
`tan` 92、`tanh` 92、`asin` 70、`acos` 72、`asinh` 70、`acosh` 67。
其中「NaN 虚部 → 抛 `DECODE-FLOAT` 类型断言」是**崩溃**，不是错值。

**做法**：逐分支照抄 numpy `npy_math_complex.c.src` ——
`ctanh` 用 Kahan(1987)；`casinh`/`cacos` 共用 `_do_hard_work`
（Hull-Fairgrieve-Tang）+ `_clog_for_large_values`；`casin` 由 `casinh`
变换；`cacosh` 由 `cacos` 旋转；`ctan` 完全委托 `ctanh`。
**上一轮试过并回滚的恒等式全部弃用**（`asin = -i·log(iz+sqrt(1-z²))` 两版各有
386 处错、`tanh` 的 tan 形式 232 处）。

**四个坑**（都写进了 §16.14.5）：`(1- y)` 是「y 减 1」不是「1 减 y」；
`(- y y)` 在内联循环里会被代数化简成 0（NaN 要用常量）；
「含 NaN」分支的外层条件只能是 `isnan(x)||isnan(y)`，写成「非双有限」会吞掉
「x 有限 + y=±Inf」并返回 `2x`；`%cx-hypot` 对 `(Inf,Inf)` 得 NaN，
`_clog_for_large_values` 用之前必须先判 Inf。

**测试**：`complex-elementary-test` 68 → **164 断言**（网格从 5 算子扩到 11 个，
加 §D2「NaN 输入不抛条件」的 72 条回归 + 3 条极点语义 + 3 条新护栏）。
期望值由 `test/gen_complex_elementary_expected.py` 重新生成（441 × 11 点）。
新契约见 `CONVENTIONS.md` §16.14.5；`atanh` 的 18 处遗留登记为 §16.37 的 U11。

---

## 2026-10-25（二十二）— 第二十二轮：复数 `log` 族内核化 + 四个「晚加载」内核前移

**行为变化**：有，且是**修错**（`log` 在分支切割上的相位）。其余为执行路径。

| 场景 | 修前 | 修后 |
|---|---|---|
| `log(-0+0i)` | `-Inf + 0i` | **`-Inf + πi`**（= numpy） |
| `log(-0-0i)` | `-Inf - 0i` | **`-Inf - πi`** |
| `log(Inf+NaNj)` | `NaN + NaNj` | **`Inf + NaNj`** |

**性能**（1M complex128，SBCL 2.6.8，同进程 3 次取最优）：

| 场景 | 修前 | 修后 | 同形 float64 |
|---|---:|---:|---:|
| `vt-log` | 352 ms | **55 ms** | 8 ms |
| `vt-log2` | 354 ms | **96 ms** | 8 ms |
| `vt-log10` | 357 ms | **97 ms** | 9 ms |
| `vt-log1p` | 383 ms | **65 ms** | — |
| `vt-sign` | 104 ms | **26 ms** | 2.6 ms |
| `vt-sinc` | 218 ms | **81 ms** | — |
| `vt-expm1` | 116 ms | **58 ms** | — |

**两处根因**（都不是算法问题）：

1. **`%cop-log` 直接回调 CL 的 `(log z)`**。SBCL 的复数 log 走通用复数算术，
   332 ns/元素；分量式 `log|z| + i·atan2(y,x)` 只需 abs + log + atan2。
   顺带修掉三处相位错值 —— 根因是 CL 的 `log` 先判 NaN 再用自己的 `phase`，
   把「模为 Inf」与「符号零决定的相位」两种情况都抹平了。改法要点：模走
   `%cop-abs`（带量级守卫，避免 `x²` 上溢）、Inf 判定优先于 NaN、
   相位用 `atan` 两参版（`phase` 对 `-0.0` 的处理与 numpy 不同）。
   **`%cop-log` 还必须 `declaim inline`**：内联前后是 199 / 55 ms。
2. **四个逐元素内核被晚加载的文件定义**。`%cop-expm1` / `%cop-log1p` /
   `%cop-sinc` / `%cop-sign` 原住在 `complex-ops.lisp`，而调用点
   （`extensions3` 的 `vt-sign`/`vt-expm1`/`vt-log1p`、`elementwise2` 的
   `vt-sinc`）**先于它**编译。符号在编译调用点时还不存在 → `(function …)`
   编成真实函数调用，内核本体被独立编译成「泛型 NUMBER」版本。
   前移到 `complex-kernels.lisp` 即回到内联快路径。
   **注意这类缺陷 `Proclaiming … to be INLINE, but N calls …` 探测器抓不到**
   （它只抓「符号已存在、声明迟到」），只能静态扫描加载顺序 —— 已写进
   `CONVENTIONS.md` §16.8.1 第 4 条的「盲区」。

**同时量清了另一件事**：剩下 6 个复数超越函数（`tan`/`tanh`/`asin`/`acos`/
`asinh`/`acosh`）仍走 CL 回退，在同一张 441 点网格上与 numpy 有 **481/4410**
处不一致（其中 `vt-asin` 对含 NaN 虚部的输入会**直接抛类型错误**）。
其中 67~92 处/函数落在「分支切割的符号零」与 NaN 输入上 —— 属已有静默错值，
不是本轮引入。试过两版 `asin` 与一版 `tanh` 分量式，均**更差**（386/386/232
处不一致），已回滚并把数据写进 §16.37 的 U10，避免下一轮重复踩。

**测试**：新增 `complex-elementary-test`（68 断言 = 441 点网格 × 5 算子与
numpy 2.4.6 逐点对拍 + 分支切割/极点/NaN 组合的显式断言 + dtype/shape/`:out`/
`:base`/空数组契约 + 4 个前移内核的标量参考回归 + 四条倍率护栏）。
期望值由 `test/gen_complex_elementary_expected.py` 生成。

**顺带修掉测试侧的一个既有缺陷**：`cx-reduce-where-test` 的 `ms-of` 用
`get-internal-real-time` 计时，而它在 SBCL 上只有 **1ms tick** ——
比 tick 短的操作（`vt-amax` float64 1M 约 0.8ms）一律量成 0，护栏的分母
落到下限 0.05，于是「c128/f64 ≤ 8x」偶发报 **60x 假失败**（实测 3 次里 2 次）。
两套件统一换成 `sb-ext:get-time-of-day`（µs 分辨率）。
新契约见 `CONVENTIONS.md` §16.14.4、§16.8.1（盲区）。

---

## 2026-10-24（二十一）— 第二十一轮：归约 / 掩码选择，三处「类型被推断放宽」

**行为变化**：无（唯一可见差异是 `vt-where` 的分配不再先清零，结果值不变）。

**性能**（1M 元素）：

| 场景 | 修前 | 修后 | 同形实数线 |
|---|---:|---:|---:|
| `vt-amax` c128 | 41 ms | **3 ms** | <1 ms |
| `vt-amin` c128 | 41 ms | **2 ms** | <1 ms |
| `vt-where(mask,a,b)` c128 | 32 ms | **5 ms** | 2 ms |
| `vt-nansum` | 35 ms | **5 ms** | 1 ms |
| `vt-nanmean` | 46 ms | **13 ms** | <1 ms |
| `vt-nanvar` | 65 ms | **28 ms** | 1 ms |
| `vt-nanstd` | 62 ms | **30 ms** | 1 ms |
| `vt-nanmax` | 88 ms | **22 ms** | 1 ms |
| `vt-nancumsum` | 36 ms | **5 ms** | 13 ms |

两处根因都是**变量类型被推断放宽**，热路径静默退回泛型：

1. **归约累加器**。`(let ((acc (aref d 0))) … (vector acc))` 里 `(vector acc)`
   接受任意类型，而 `%cop-max` 的派生返回类型是 `T`（形参无声明 → 返回 a 或 b）→
   SBCL 把 `acc` 放宽成 `T`，每次调用都装箱/解包复数。补
   `(declare (type (complex double-float) acc))` 即 41 → 3 ms。
   新增 `%def-cx-extremum-kernel` 按精度各生成一份（累加器带类型 + 首元素播种）。
   > 逐元素路径没有这个问题（内联展开后按上下文重推类型），所以出现了
   > 「同一内核、逐元素 5 ms、归约 41 ms」的怪象。
2. **`vt-where` 永远进不去内联快路径**。`%vt-inline3-fast` 要求「三个输入与
   结果元素类型全同」，而掩码恒为 `:int8`、数据是 `:complex128` —— 永不相等，
   只能走 `vt-map` 的闭包。新增 `%select-loop` / `%select-fast`（10 格分派），
   32 → 5 ms。nan 族全建在它上面，收益直接传导（见上表）。

**排序**：归并改为双缓冲乒乓（不再每段 `replace` 回原缓冲区）。200 → 181 ms
（c128）、132 → 130 ms（f64）—— 收益只有 10%，因为真正的成本是**比较分支
不可预测**（c64 与 c128 耗时几乎相同 ⇒ 不是带宽受限）。要再压须换基数排序，
作为 U9 登记在 §16.37，本轮不做。

**测试**：新增 `cx-reduce-where-test`（119 断言 = `vt-where` 10 dtype × 4 掩码
对逐元素参考 + 广播/转置/`:out`/错误路径 + `amax`/`amin` 的 NaN 传播与单轴 +
排序乒乓的 1..1000 规模边界 + 四条倍率护栏）。新契约见 `CONVENTIONS.md` §16.8.2。

过程、最小复现与「字典序」踩坑见 `HISTORY.md` §H.3.15。

---

## 2026-10-23（二十）— 第二十轮：极值族 / argsort / topk / 比较族，10~300 倍

**行为变化**：无（`topk` 的 `k=1` 顺带修掉一个静默错值，见下）。

**性能**（1M 元素）：

| 场景 | 修前 | 修后 | 同形实数线 |
|---|---:|---:|---:|
| `vt-maximum` c128 | 804 ms | **7 ms** | 2 ms |
| `vt-clip` c128 | 1600 ms | **9 ms** | 3 ms |
| `vt-argsort` c128 | 2115 ms | **273 ms** | 166 ms |
| `vt-topk k=10` c128 | 490 ms | **303 ms** | 168 ms |
| `vt-=` c128 | 60 ms | **4 ms** | 1 ms |
| `vt-<` c128 | 45 ms | **5 ms** | 5 ms |
| `vt-isnan` c128 | 25 ms | **1 ms** | 1 ms |

四处根因，**没有一处是算法错**，全是「用错变体」或「放错文件」：

1. 极值内核用边界版 `%cisnan` 判 NaN —— 它每次调用都套一层
   `with-float-traps-masked`（约 200 ns），每元素两次 = 400 ns。换免屏蔽的
   `%cx-nan-p` 即 839 → 3 ms。上一轮把这个归因成「返回值类型未知」，是错的。
2. `%cast-to` 的复数分支在「结果类型未知」时要 2 次 `realpart` + 2 次 `coerce`。
   加 `typep` 同宽快通道（命中直接返回，**不用 `the` 撒谎**），1M 7 → 3 ms，
   与手写循环同速。这是**通用**修复，所有「复数目标、类型未知」的内联算子受益。
3. `argsort` 用 `stable-sort` + lambda 比较器（每次比较一次 `funcall`）。
   换成下标归并内核（插入排序 + 自底向上归并，比较器内联）：2115 → 273 ms，
   排列逐位相同、稳定性单独验过。
4. `topk` 同时 `vt-sort` + `vt-argsort`（两趟 O(n log n)），但它只要 k 个元素。
   改为「一次 argsort + 沿轴 gather」：490 → 303 ms。

**第五处（顺带发现）**：`vt-=` / `vt-<` / `vt-isnan` 慢是因为九个标志内核定义在
**`complex.lisp`**，而它最后加载 —— 编译 `elementwise2` 时符号还不存在，12 个
调用点全编成真实函数调用。编译日志里 9 条
`Proclaiming %CX-EQ-FLAG to be INLINE, but 12 calls were previously compiled`
一直在说这件事，被当成噪音忽略了。前移到 `complex-kernels.lisp` 后
`STYLE-WARNING` 97 → 88，三个算子分别 60→4 / 45→5 / 25→1 ms。

**一处静默错值修复**：新增的 `%gather-axis-k` 最初假设视图 `offset` 为 0，
但 `vt-contiguous` 对「已连续」的输入原样返回，`narrow`+`flip` 的单元素视图
`offset=7` 仍被判连续 → `topk(k=1)` 返回全数组最小值（下标对、值错）。
既有 `%gather-along-axis` 同隐患，一并补 `(vt-offset …)` 基址。

**测试**：新增 `cx-extreme-argsort-test`（74 断言 = 极值语义独立参照 16×16×2 +
四种布局同语义 + `clip` + `argsort` 双射/非降序/稳定性/多轴 + `topk` 边界 k +
四条倍率护栏）。新契约见 `CONVENTIONS.md` §16.8.1（四条执行层硬约束 +
「previously compiled」探测器）。

过程与消融见 `HISTORY.md` §H.3.14；§16.37 的 U5 / U6 / U8 全部关闭。

---

## 2026-10-23（十九）— 第十九轮：性能巡检 —— `median` 秒级退化与 `replace` 缺声明

**方法**：不再「指哪儿打哪儿」，先写一个覆盖 8 个族、50 个算子的巡检脚本，
对同一算子分别跑 float64 / complex128（同元素数），把比值全打出来，≥4× 的逐个
查根因。立刻抓到四个前两轮没看到的问题。

**行为变化**：无（全是执行路径；`median`/`percentile` 的取值由独立参照逐点核对）。

**性能**：

| 场景 | 修前 | 修后 |
|---|---:|---:|
| `median` 全零数组 200k | **10 700 ms** | **1 ms** |
| `median` 只有两值 200k | 2 980 ms | 2 ms |
| `median` 整体降序 400k | 142 ms | 21 ms |
| `median` 均匀随机 200k | 3 ms | 6 ms |
| `vt-copy` 连续 c128 1M | 12 ms | 4 ms |
| `vt-copy`/`vt-flatten`/`roll`/`tile`/`pad`（复数 vs 实数倍率） | 3~12× | 1.5~2× |

**三处根因**：

1. **quickselect 两分区 + 重复值 = O(n²)**：每趟只剥掉 pivot 一个元素。改
   三分区（`<`/`=`/`>`）后全等数组 O(n) 结束；
2. **三分区 + 三数取中的坏排列**：整体降序且 `k = n/2-1` 时每趟只削极窄一段
   （451 趟、30.2M 次访问）。加**访问量上限 4n**，超限改用 `%sort-fibers`
   排完再取第 k 个，最坏钉在 O(n log n)；
3. **`replace` 缺类型声明 = 逐元素泛型序列操作**：两端都带
   `(the (simple-array …))` 时才走字节级 memmove —— 1M c128 **8 ms → 0.2 ms**。
   新增 `%memcpy` 作为库内唯一块拷贝实现，`vt-copy`/`vt-contiguous`/
   `vt-copy-into`/`join` 的 `%block-copy` 全部接上（顺带补上复数两格）；
   另新增 `%alloc-vt`（不填充的分配），省掉 `vt-copy` 那趟 3 ms 零填充。

**一条负结果**：复数 `clip` 1M 要 1.6 s（`maximum`+`minimum` 各 834 ms）。试了
两版 `vt-fast-map` 内联都**更慢**（1469~1592 ms），已回滚 —— 根因是内核返回类型
取决于分量宽度、内联循环拿不到确定类型。连同 `argsort`/`topk`（1.06 s）、
nan 族（~100 ms）一起登记进 §16.37 的未完成项，附数据与下一步方案。

**新套件**：`order-stat-test`（266 断言 = 6 种分布 × 7 种规模 × 5 个分位点的
独立参照对拍 + 退化倍率护栏）。新契约见 `CONVENTIONS.md` §16.35.5。

过程、消融与被回滚的两个方案见 `HISTORY.md` §H.3.13。

---

## 2026-10-23（十八）— 第十八轮：转置视图的复数拷贝提速（`vt-copy` 17 → 6 ms）

**行为变化**：无。只换遍历形态，落点、dtype、形状一律不变。

**性能**（1M 元素，SBCL 2.6.8）：

| 场景 | 修前 | 修后 | 同形实数对照 |
|---|---:|---:|---:|
| `vt-copy` 转置 c128T | 17 ms | **6 ms** | 3 ms |
| `vt-contiguous` 转置 c128T | 17 ms | **6 ms** | 4 ms |
| `vt-astype` 转置 c128T→c128 | 6 ms | **4 ms** | — |
| `vt-astype` 转置 c128T→c64 | 9 ms | **7 ms** | — |

两处根因，都不是上一轮猜的「访存次序」：

1. `%gather-contiguous` 用全轴 odometer，每元素约 5 ns 的固定开销（1M = 5 ms）。
   改成**末轴 `DOTIMES` 展开、其余轴走 odometer**后，转置收集 6 → 1 ms；
2. `%copy-generic`（`vt-copy`/`vt-contiguous`/`vt-copy-into` 的公共内核）的
   按元素类型分派表里**缺复数两格**，复数整片掉进泛型 `aref`。补两格即回到实数线。

**同时更正上一轮的一条错误结论**：`c128T→c128` 曾被写成「比 numpy 慢一个量级」。
复核发现 numpy 的 `astype` 缺省 `order='K'`（转置输入返回 F 连续结果 = 一次线性流），
同口径要写 `order='C'`（1.05 ms）；而「差距在访存次序」也不成立 —— 手写实验脚本
当时的「列主序 0 ms」写错了内层步长。详见 `HISTORY.md` §H.3.11 的更正与 §H.3.12。

**已知且不修**：SBCL 的 `replace` 没有 16 字节元素的 basher，`(complex double-float)`
的同型块拷贝只能逐元素（5~10 ms/1M，而 `double-float` 是 1 ms）。`sb-kernel` 的
`system-area-ub8-copy` 能到 1.5 ms 级，但属内部符号且需 pinning，与「可移植优先」
冲突，已写进 `CONVENTIONS.md` §16.4.1 作为已知固定成本。

验证：`test-copy-into` 35 → 41 断言（新增复数×非连续的 typed-walk 覆盖 + 倍率护栏），
55 套件全绿，编译体检 `caught WARNING` 0。

---

## 2026-10-23（十七）— 第十七轮：`vt-astype` 的复数转换提速 5~20 倍

**行为变化**：无。所有 dtype 组合的取值 / dtype / 形状 / 副本语义一律不变，
只换执行路径（100 格 numpy 对拍逐位相同）。

**性能**（1M 元素，SBCL 2.6.8，同进程 6 次取最优）：

| 转换 | 优化前 | 优化后 |
|---|---:|---:|
| `c128→c128`（同 dtype 的纯拷贝） | 17 ms | 3 ms |
| `c128→c64` | 31 ms | 2 ms |
| `c64→c128` | 26 ms | 3 ms |
| `c128→f64` | 25 ms | 2 ms |
| `c128→i64` | 34 ms | 3 ms |
| `f64→c128` | 27 ms | 3 ms |
| 转置视图 `c128T→c64` | 35 ms | 8 ms |

实数线（1~3 ms）不受影响。

**根因**：`vt-astype` 的特化分支只列了实数元素类型的组合，复数全部落进
「泛型 `aref` + `vt-cast-fun` 闭包」的兜底 —— 每个元素一次 `complexp` 判断
加装箱/解包。补上 34 个复数子句（同型走 `replace` 块拷贝）后即与实数线同量级。

**非连续视图**：新增 `%gather-contiguous`（按 `strides` 的同 dtype 特化收集），
转置 / 翻转 / 切片 / 广播视图由 17~35 ms 降到 3~8 ms。

**新套件**：`astype-cast-test`（663 断言 = 100 格矩阵 + 契约 + 非连续等价 +
倍率护栏），期望值由 `test/gen_astype_cast_expected.py` 从 numpy 2.4.6 实测生成。
新契约见 `CONVENTIONS.md` §16.4.1。

过程、消融与一条负结果（转置拷贝的访存次序仍压着 2~4 倍）见 `HISTORY.md` §H.3.11。

---

## 2026-10-23（十六）— 第十六轮：`:out` 重叠保护缺口 + 两处 dtype 静默降级

**行为变化（可能影响调用方）**

- 实数 `cumsum`/`cumprod` 的 `:out` 与输入**错位重叠**时，以前会算错（先写的
  结果污染还没读的输入，越滚越大），现在正确；
- `(vt-nanmedian 实数 :dtype ≠ :float64)` 与 `(vt-nanmedian 复数 :dtype 非复数)`
  由「静默忽略 / 静默取实部」改为**报错**。缺省行为不变。

**修掉的两个静默错值**

| 问题 | 影响 |
|---|---|
| 流式累积（实数 `cumsum`/`cumprod`）没做 `:out` 重叠快照 | 错位重叠时输入被自己写坏 |
| `nanmedian` 的 `:dtype` 被忽略 / 复数降级成实数 | 显式 dtype 无效；虚部无声消失（正中 §16.13 红线） |

**一条负结果**：复数 matmul 把「每元素一次 C 读改写」换成 4 个局部累加器，
A/B 实测无稳定收益，**已回退**。瓶颈在复数乘法的标量算术，不在 C 访存；
再提速只能上 SIMD。

验证：48 例 cumsum 沿轴随机对拍、退化多轴 nanmedian 与 numpy 2.4.6 一致；
`cumulative-axis-test` 70 → 78 断言；54 套件全绿，编译体检 WARNING 0。

过程、踩坑与实测表见 `HISTORY.md` §H.3.3。

---

## 2026-10-23（十五）— 第十五轮：沿轴累积提速 49× + `nanmedian` 多轴 + 编译门禁

**性能**（1000×1000，`vt-cumsum :axis`）：196 ms → **4 ms**（numpy 约 3.3 ms）。
同一个函数的展平路径只要 1 ms —— 差别只在于沿轴仍在走通用回退（每元素一次
`funcall` 装箱 + `vt-cast` + 泛型 `aref`）。把「平面」与「沿轴」统一到同一个
按元素类型特化的内核后，两条路径都吃到特化。`nancumsum :axis` 260 → 14 ms。

**新能力**：`vt-nanmedian` 支持多轴（`axis=(0,1)`），与 numpy 逐位一致。
裁决依据是「中位数对归约区内元素顺序不敏感」，因此可用「置换 → 折叠成 2-D →
沿单轴归约」复用单轴内核；实数与复数共用同一份折叠逻辑。

**新门禁**：`run-tests.sh` 跑套件前先做一次全系统编译体检，`caught WARNING` > 0
即 FAIL（`STYLE-WARNING` 不计）。理由是第十四轮那个「docstring 裸引号」事故
**运行期完全正常**，只有编译输出能发现。注意**不能**用 `muffle-warning` 包编译
——它会把告警一起吞掉，门禁恒报 0。

验证：新增 `cumulative-axis-test`（70 断言，含独立朴素参照对照 + 性能护栏
「沿轴 ≤ 40× 展平」）；54 套件全绿。

过程与消融数据见 `HISTORY.md` §H.3.2。

---

## 2026-10-23（十四）— 第十四轮：工程状态复审（4 个缺陷）

前十三轮解决「复数能不能算对」。本轮换角度：在全量重跑测试入口的过程中追问
「这条命令在一台干净的机器上真的能过吗」，抓到 4 个缺陷 + 1 个待裁决项。

| # | 问题 | 影响 |
|---|---|---|
| 1 | `elementwise1/2` 是 `elementwise` 的完整副本（只新增文件、没删原文件） | 拆分零收益（4 GB 堆仍 `Heap exhausted`），且重新引入 85 对跨文件同名 `defun` |
| 2 | docstring 里一个裸 `"` 提前闭合字符串，`is/not/of/type/REAL` 成了函数体里的 5 个 form | 20 条编译 WARNING，但**运行期正常** —— 靠未定义行为工作 |
| 3 | key-sync 门禁逐字节比对，而 `str(np.float64)` 随 numpy 版本变（差 1 ulp） | 本机必然 `exit 1`，52 个套件一个都跑不了 |
| 4 | 实数 `vt-cumulative` 漏包 `with-float-safe` | `(vt-cumprod #(inf 0))` 直接抛条件，numpy 给 `[inf, nan]` |

**行为变化**

- 实数 `cumsum`/`cumprod` 的首元素改为「直拷输入」而不是「从单位元起算」：
  `cumsum([-0.0])` 现在是 `-0.0`（不是 `+0.0`），符号位不再被洗掉；
  `cumprod([inf, 0])` 现在是 `[inf, nan]`（不是抛条件）。
- `vt-svd` 的 `max-sweeps`/`tol` 改为 `supplied-p`：实数 50/1e-14、复数 60/1e-15，
  各用自己验证过的缺省。此前复数 SVD 一直跑在实数的那组参数上。

**门禁变化**：key-sync 默认改「语义比较」（key/dtype/shape 必须一致，`:tol` 条目
按 dtype 分级容差，`:exact` 仍逐字符），`--strict` 保留字节口径；另加「生成器
退出码非 0 即 FAIL」（文件不重写的假通过比没门禁更危险）。

验证：冷编译 4 GB 从 `Heap exhausted` → 成功；`caught WARNING` 20 → 0；
跨文件同名 `defun` 85 → 0；新增 `scan-first-element-test`（38 断言，共 53 套件）。

过程、踩坑与实测表见 `HISTORY.md` §H.3.1。

---

## 2026-10-23（十三）— 第十三轮：覆盖式复数层收尾（159/159 单一 DEFUN）

第十二轮把 143/159 改成单一 `DEFUN` 分派，剩 16 个"形状不规整"的；本轮做完并
删掉整套覆盖机制。

16 个符号按"该怎么改"分四类：**纯冗余**（`vt-@`/`vt-expt`，删重复 defun）、
**错误死桩搬迁**（`vt-real`/`vt-imag`/`vt-conj`/`vt-angle` —— `extensions3.lisp`
那份写着"本库无复数 dtype，恒返回输入本身"，是错的，可靠"后加载者胜"维持）、
**匿名包装合并**（`vt-log`/`vt-float-power`）、**同名覆盖 + `%cx-orig` 转发**
（8 个，复件改名 `%cx-op-*`，归属文件插守卫，两份 lambda-list 合成一份）。

结构上把标量内核前移到新文件 `src/complex-kernels.lisp`（`dispatch` 之后、所有
归属文件之前），依赖方向变成单向「归属文件 → 标量内核」，消除倒挂。

迁移中发现 `vt-log` 的 `:base` 一直在除零：`(log base 1.0d0)` 是**两参数对数**
`log_1(base)`，导致 `(vt-log #C(8 0) :base 2)` 返回 `0+0i`（应 `3+0i`）。这个 bug
在覆盖式架构下活了五轮 —— 因为没有任何用例覆盖"复数 + 非缺省 base"这一格。

新增 `test/complex-reload-test.lisp`（14 断言）：同进程逐个重载 12 个源文件，
每轮跑 19 个探针。这是本轮**唯一真正的验收**：覆盖式架构下它必然失败。

结果：单一 `DEFUN` 分派 **159/159**；`(setf (symbol-function …))` 2 → **0**；
`%cx-orig` 调用点 16 → **0**；`eval-when` 预捕获清单 124 条 → **0**；
跨文件同名 `defun` 14 对 → **0**；套件 51 → **52**。

---

## 2026-10-23（十二）— 第十二轮：拆掉覆盖式复数层（第一阶段 92/159）

**背景**：复数支持早期靠"运行期重定义公开函数"实现（`(setf (symbol-function …))`
装匿名包装，或"同名 defun 再写一遍 + `%cx-orig` 转发"）。后果是：只重载
`src/elementwise.lisp`，`(vt-sqrt #C(-1 0))` 会从 `#C(-0.0d0 1.0d0)` 静默变成
`float64` 的 `NaN`；函数对象是匿名 LAMBDA，`declaim inline` 失效；哪些符号被覆盖
无法静态枚举。

**做法**：把复数分支下沉到归属文件的**单一 `DEFUN` 体内**：

```lisp
(defun vt-abs (vt &key out dtype)
  (when (%cx-any-p vt)                    ; 复数分流
    (return-from vt-abs (%cx-abs …)))
  …原实数实现不变…)
```

新增 `src/dispatch.lisp` 作为唯一判定入口（`%cx-arg-complex-p` / `%cx-any-p` /
`%cx-deep-p` / `%cx-result-complex-dtype` / `%cx-reduce-dispatch`），在 `clvt.asd`
里紧随 `core` 加载。本轮完成 92/159（elementwise 20 拒绝型 + 10 比较/极值 +
18 一元、nn 9 个激活、extensions3 10 个、reduce-stats 14 个门面……）。

顺带发现 9 处 lambda-list 漂移（两处同名 defun 的参数表不一致，靠加载顺序决定
谁生效），记入待办。

新增 `test/complex-dispatch-test.lisp`（结构契约：必须是具名 `DEFUN`、list 入参
走复数路径、实数不回归）。套件 50 → 51。


## 2026-10-21（十一）— 第十一轮：弱标量提升（窄 dtype 不再被标量带宽）

前十轮都在比**值**和「复数会不会掉回实数实现」。本轮换成**只比 dtype**：
119 条表达式全喂 `complex64`（外加实数 dtype 对照），逐项与 numpy 的**结果 dtype**
比对。值的正确性前几轮已覆盖，这一维是空白 —— 结果抓到一整类问题。

### 症状：窄 dtype 遇到标量就静默升宽

| 表达式（`*` = 张量） | 修复前 | 修复后 = numpy |
|---|---|---|
| `f32* + 1.5` / `f32* + 1` | float64 | **float32** |
| `i32* + 1` / `i8* + 1` / `u8* + 1` | int64 | **int32 / int8 / uint8** |
| `c64* + #C(1 2)` / `c64* + 1.5` | complex128 | **complex64** |
| `f32* + #C(1 2)` | complex128 | **complex64** |
| `f32* / 2` | float64 | **float32** |
| `c64* ** 2`、`maximum(c64*, 标量)` | complex128 | **complex64** |

不是算错，而是**分配了双倍内存、带宽翻倍、SIMD 通道白丢一半**，且完全不报错。
顺带把 `vt-gradient(c64)` 也带回了 complex64（它是自然提升的函数，之前被标量带宽）。

### 根因

逐元素运算的 dtype 一律走

```lisp
(apply #'vt-promote-type (mapcar #'vt-dtype inputs))   ; inputs 已全部 ensure-vt
```

而 `(ensure-vt 1.5d0)` 是 **float64 的 0 维张量**、`(ensure-vt #C(1 2))` 是
**complex128** —— 「谁是标量」在 ensure 之后已经没了，标量于是以全强度参与提升。

numpy 2.x（NEP 50）里 Python 标量是**弱**类型：只贡献种类（int/float/complex），
不贡献宽度；宽度由张量决定，只有「整型张量 + 浮点标量」才落到该种类的默认宽度。
CL 的 `1.5d0` 对应 Python 的 `1.5`，语义正好对齐，故按 NEP 50 修
（CONVENTIONS §0：与 NumPy 冲突以 NumPy 为准）。

### 修法

`src/dtype.lisp` 新增 `%SCALAR-KIND`（张量→NIL，Lisp 数→:INT/:FLOAT/:COMPLEX）
与 `%OP-FINAL-DTYPE`（张量之间仍走原 100 格 promote 表，弱标量只按种类并入）。
接进 `vt-map`、`vt-fast-map` 宏、`vt-/`、`vt-pow`（含 float-power）、
`maximum/minimum/fmax/fmin`。

一处关键细节：**标量身份只能沿着「原始实参」传**。`vt-/` 第一版只改了自身判定，
随后又把已 ensure 的值传给 `vt-fast-map`，标量身份二次丢失，`f32/2` 依旧 float64。
`vt-fast-map` 宏因此改成「先绑原始实参、再 ensure-vt」。

### 附带优化：单元素输入预转（让内联快路径回来）

改完 dtype 先测性能，发现 c64 数组 + 标量反而慢 27% —— 因为「弱标量」在 ensure-vt
之后是**另一种 dtype 的 0 维张量**，`%VT-INLINE*-FAST` 的「输入与输出同元素类型」
前置条件不成立，整个运算掉进 funcall + 泛型 aref 兜底。

修法：对满足五个条件的**单元素**输入预转成结果 dtype（`%WEAK-CAST1`）：
SIZE=1、dtype 与结果不同、种类秩 `源 ≤ 目标`（int<float<complex 只走无损方向）、
整数值要装得下、**这组输入里至少有一个真张量**。不整体预转大数组。

后三个条件是踩出来的，每一条都对应一次「预转把语义前提悄悄改掉」：
* 只按「单元素 + dtype 不同」转 → 比较运算的标量进了 int8 掩码，`nan<1` 判错；
* 加上「同大类」→ 仍会把 `999000` 转进 int8（变成 -8），argwhere 数出 999911；
* 加上「值装得下」→ 全标量的 `(vt-complex 1.0 2.0)` 把实参转成复数，
  构造函数收到 `#C(1 0)` 报 `is not of type REAL`。
三次都是跑全量套件才发现的 —— 局部看都对。

实测 2M 元素（min-of-7）：

| 表达式 | 第十轮 | 第十一轮 |
|---|---:|---:|
| `f32 数组 + 1.5` | 77 ms | **4.1 ms**（19×）|
| `c64 数组 + 1.5` | 163 ms | **5.4 ms**（30×）|

旧实现慢正是因为升宽导致输入/结果类型不一致、一直在跑兜底路径。

### 有意保留的三处差异

* `det/inv/solve/pinv/cholesky/qr/matrix-power/svd/eigvalsh` 的 c64 输入仍返回
  complex128 / float64：复数线代**恒在 double 下计算**（`vt-det` 文档已写明），
  精度更高，改回单精度是降级；
* `ptp` / `cond` 返回 Lisp 标量而非张量（§5 既定风格），原生数没有 dtype；
* 弱整数越界（`i8* + 300`）本库回绕，numpy 抛 OverflowError。

### 测试

`test/dtype-weak-scalar-test.lisp`（**14 断言**，已接入 `run-tests.sh`，共 50 套件）。
按要求**只收录修复前确实错的用例**：11 个 dtype 断言（每一个都在修复前的构建上
复验过「当时真的是错的」）+ 3 个值正确性断言（确认不再升宽没把值算错）。
张量×张量的 promote 100 格与 c64 其余 dtype 契约由既有套件覆盖，不重复。

全量 **50 套件全绿**；`manip-interop-test` 的 195 例 Lisp↔numpy 逐位对照 0 差异。

文档：CONVENTIONS §16.31（方法 / 症状 / 根因 / 修法 / 保留差异）、
§16.20 变更清单。

---

## 2026-10-21（十）— 第十轮：median 契约、前缀扫描首元素与 NaN/Inf 语义

第九轮补了「数值对拍」这条扫描线，本轮把表达式从 190 条扩到 **262 条**，
新增 NaN/Inf 全组合、`axis` 元组、`:out` × 非连续、einsum 补全四个方向。
结果：**除 `vt-median` 外这三块都没有静默错值**，而 `median` 一次暴露三处。

### 缺陷一：`vt-median` 复数版与自家实数版三处分叉（都不报错）

```lisp
(vt-median a)   ; a = (3 4) complex128
;; 修复前 → 3 个中位、形状 (3);  修复后 → 0 维标量（与实数路径、numpy 一致）
```

| # | 症状 | 根因 |
|---|---|---|
| 1 | `axis=nil` 按**末轴**归约（`(3 4)` → 形状 `(3)`，`:keepdims` → `(3 1)`） | 缺省轴写成 `(1- rank)`；numpy 与实数版都是「先 ravel 再取全局中位」 |
| 2 | 空 fiber（轴长 0）返回 **0+0i** | `lo = (floor -1 2) = -1` + `%cx-fiber-starts` 在 size=0 时返回 0 条 → 结果保持 `make-vt` 的 0 |
| 3 | 含 NaN 返回**某个真实元素** | 复数版没有 NaN 检查；numpy 与实数版都传播 NaN |

顺带补了个洞：**复数 `vt-median` 原来没包 `WITH-FLOAT-SAFE`**（实数版包了）。
SBCL 默认开着 `OVERFLOW/INVALID/DIVIDE-BY-ZERO` 陷阱，于是 NaN 判定 `(= x x)`
与 `inf + (-inf)` 会直接抛 `FLOATING-POINT-INVALID-OPERATION`。

### 缺陷二：前缀扫描用「单位元」而非「首元素直拷」

numpy 的 **accumulate** 与 **reduce** 在这里方向相反：

```python
np.prod([inf+0j])            # (inf+nanj)          ← reduce 从 1 起算
np.cumprod([inf+0j, 2+0j])   # [inf+0j, inf+nanj]  ← accumulate 首元素直拷
np.cumsum([-0.0+0j])         # [-0.+0.j]           ← 首元素直拷，符号位保留
```

`%CX-SCAN-RUN` 两条路径都从单位元起算，于是：

* `cumsum` 把输入的 `-0.0` 洗成 `+0.0`（符号位丢失）；
* `cumprod` 首元素走 `1 * (inf+0i)` → `(1·inf − 0·0, 1·0 + 0·inf)` = `inf + nan i`，
  凭空多一个虚部 NaN，并被后续所有元素继承。

修法：连续路径与 dtype 回退路径都改成「首元素直拷，从第二个起才 OP」。
有限数据下与旧实现逐位相同，只在 -0.0 / Inf 上有别。

> 遗留：实数线的 `vt-cumsum` 走通用 `VT-CUMULATIVE`（acc 从 0 起算），
> `cumsum([-0.0])` 仍给 `+0.0`。修它要动四处循环，与本轮主题无关，列为下一轮候选。

### 已核实「不是 bug」的差异

对拍剩下的 7 项不一致逐项归因：QR 的相位约定（重构误差 1.1e-16 / 6.7e-16）、
LU 的置换约定、numpy 复数均值用 `sum × (1/n)` 导致 `0*Inf` 污染另一分量、
全 NaN 切片里「用哪个 NaN」（`NaN+0i` vs `NaN+NaN i`）。
**对拍不一致 ≠ 缺陷** —— 先排除约定差异与对方的浮点副作用，再决定改不改。

### 测试

新增 `test/complex-nan-inf-test.lisp`（**77 断言**，已接入 `run-tests.sh`，现共 **49 套件**）：

- median 契约 15 项（`axis=nil` 形状/值/keepdims、与实数路径形状一致、NaN 全局与逐 fiber 传播、空轴/空输入、c64、`nanmedian` 对照）；
- 前缀扫描 8 项（`-0.0` 符号位、`cumprod` 首元素 `inf+0i`、`axis=0` 每列首元素、c64、`axis=nil` 展平）；
- NaN/Inf 归约 17 项（`nan*` 家族含 ±Inf 与全 NaN 切片、`inf+(-inf)` 相消、掩码三兄弟、`nan_to_num` 三替换、含 NaN 排序）；
- `axis` 元组 12 项（形状 + **逐元素手算**对照，不借道库函数）；
- `:out` 7 项（列切片**逐位置**核对、转置视图、reduce keepdims 列视图、matmul、原地别名）；
- einsum 9 项（`...i,...i->...`、`ii->`、`ij->ji`、`ijk->ik`、`ijk,jk->i`、`i,j->ij`、`ab,bc,cd->ad`、`ijk->kji`、`...ij->...i`）。

测试进程**故意不关 FP 陷阱**（SBCL 默认全开），既验证开陷阱时也能安全产出 NaN，
也顺带抓到了 `WITH-FLOAT-SAFE` 的缺失。全量 **49 套件全绿**。

文档：CONVENTIONS §16.30（方法 / 两个缺陷 / 三类「不是 bug」/ 测试）、§16.20 变更清单。

---

## 2026-10-21（九）— 第九轮：沿轴扫描的静默错值（cumsum/cumprod 只算第一条 fiber）

上一轮修的是「dtype 静默降级」，这一轮补上「**数值静默错值**」这条线 ——
两层新扫描（同形调用对照 + 与 numpy 的数值对拍）跑下来，抓到 1 个真 bug。

### 症状

```lisp
(vt-cumsum a :axis 0)     ; a = (3 4) complex128
;; => [[-1.5-0.5j,   0+0j,  0+0j, 0+0j],
;;     [-1.0+0j,      0+0j,  0+0j, 0+0j],
;;     [-2.0+0.25j,   0+0j,  0+0j, 0+0j]]   ; 只有第 0 列对
```

`axis` 是**最后一维**时正确，任何更靠前的轴都只算第一条 fiber，其余留 0。
受影响：`vt-cumsum` / `vt-cumprod` / `vt-nancumsum` / `vt-nancumprod`。

### 根因

`%CX-SCAN-RUN` 的 axis 分支拿到 `(head dim ax-stride)` 后只循环了
`head × dim`，**漏掉 `ax-stride` 那一层** —— 轴之后的每个位置（`axis=0`
时的其余各列）都是一条独立 fiber。

同一份 `%CX-FIBER-GEOM` 的另外六个调用点（`%DEF-CX-REDUCER` 家族、
`argmax/argmin`、`nanarg`、`sort-axis`、`nanmedian`）**都有**这层循环，
所以 `sum/prod/amax/median/sort/diff/...` 沿轴一直是对的，只有扫描内核漏了。

**为什么一直没被发现**：所有既有 `cumsum` 测试都是 1-D，而 1-D 时
`ax-stride = 1`，漏掉的那层退化成一次迭代，数值完全一样。

### 修法

补上 `ti` 层遍历（三层循环：前导块 → ax-stride 内偏移 → 沿轴），
并在注释里写明「这层不能删」的原因。

### 本轮的两层扫描方法（已写入 CONVENTIONS §16.29.4）

| 层 | 做法 | 结果 |
|---|---|---|
| A. 同形调用对照 | 对每个导出 `vt-*`，用同一组 20 种调用形态分别喂实数/复数镜像，比「实数通、复数失败/降级」 | 0 处（第八轮已清完；上一轮报的 96 个 `complex→real` 全是某个不含复数入参的形态造成的假阳性） |
| B. 数值对拍 | 190 条复数表达式 clvt 求值 vs numpy 复算，逐元素比对 | 188 一致，1 不一致（= 本 bug），2 项是 QR/LU 约定差异 |

B 覆盖：逐元素/超越、归约（axis 元组/keepdims）、沿轴扫描、排序/argsort/
unique（含 counts/inverse）、searchsorted、diff/gradient/trapezoid、
拼接/堆叠/切分/insert/delete/repeat/tile/pad/rot90/flip/roll、take/compress/
extract/take-along-axis/put-along-axis、diag/tril/triu/trace/diagonal、
matmul/dot/outer/inner/vdot/kron/cross/tensordot/multi-dot、einsum 五种模式、
det/inv/solve/pinv/cholesky/matrix_power/cond/svd-qr、复数比较与 where，
外加 complex64 一整组镜像。

> QR/LU 的两处「不一致」**不是 bug**：QR 的列可乘任意单位模相位，
> 验证用重构恒等式（‖QᴴQ−I‖ = 1.1e-16、‖QR−A‖ = 6.7e-16）；
> LU 的置换约定不同，重构由 `complex-linalg-test` 覆盖。

### 测试

新增 `test/complex-scan-axis-test.lisp`（**26 断言**，已接入 `run-tests.sh`，
现共 **48 套件**）。期望值由 numpy 2.3.5 实测硬编码，数据是二进制可精确
表示的小数、累加也精确，故要求**逐位相等**：

- `cumsum`/`cumprod` 的 axis=0/1 + 负轴 `-1`/`-2`；
- 3-D（2,3,4）三个轴各一组（最能暴露「只算第一条 fiber」）；
- 非连续输入（转置视图）沿 axis=0；
- `nancumsum`/`nancumprod` 无 NaN 时与普通扫描逐位相同；
- complex64 与 complex128 一致、dtype 保持；
- 退化：轴长 1 / 轴长 0 / `axis=nil` 展平；
- `:out`：连续目标 + 非连续目标（列切片）；
- 实数路径回归。

全量 **48 个套件全绿**。

---

## 2026-10-21（八）— 第八轮：复数缺口扫描 + 6 个补丁（3 个静默错值）

上一轮把复数 matmul 的内核修好之后，做了一次**系统性扫描**：导出清单 367 个
`vt-*` 里 202 个已被复数层碰过，剩下 165 个逐个用复数入参探针跑；
再加一层「所有可调用导出符号统一喂 complex128，看结果 dtype 是否变实数」的
保真扫描（唯一能抓「静默丢虚部」的手段）。

结论：**6 个缺口**，其中 3 个是静默错值（不报错、悄悄丢虚部）：

| 函数 | 改前 | 改后 |
|---|---|---|
| `vt-from-array` | `(complex double-float)` 数组 → **float64** | **complex128**（complex-single → complex64） |
| `vt-from-sequence` | 含复数的 list → **float64** | **complex128** |
| `vt-full` | `#C(3 1)` 填充 → **float64** | **complex128** |
| `vt-float-power` | 复数 → **float64**（只算实部） | 复数主值幂（`(2+1j)^2 = 3+4j`） |
| `vt-cbrt` | 复数 → **float64** | **确定性报错**（numpy 亦拒绝） |
| `vt-layer-norm` | 复数 → `#C(0 0) is not of type REAL` | **complex128**（复数均值 + 实数方差） |

### 三个语义决策

1. **`cbrt` 不做复数实现，改成拒绝**：numpy 2.3.5 实测
   `np.cbrt(np.array([8+0j]))` 直接 `TypeError: ufunc 'cbrt' not supported`。
   补一个 numpy 没有的「复数立方根」是造语义，不如对齐；
   要用请 `(vt-pow z 1/3)`。
2. **`float_power` 复用 `pow` 的内核**：两者在复数域是同一个主值分支
   （实测 `np.float_power` 与 `np.power` 逐位相同），差别只在实数侧。
   把 `vt-pow` 里内联的 lambda 提成 `%CX-POW-2` 两处共用，
   边界（`x^0=1` / `0^负=NaN` / `0^0=1` / NaN 传播）只留一份。
3. **`layer_norm` 的复数语义**（本库扩展，torch 只定义实数域）：
   复数均值 + 实数方差（`mean(|x-mean|²)`，与 `np.var(complex)` 一致）
   + 实数 `std = sqrt(var+eps)`，结果复数。用「虚部为 0 时退化为实数路径」
   与「归一化后 fiber 均值 0、var≈1」两条自洽性断言锁住。

### 顺带修掉的不一致

`vt-pow` 的复数路径把 NaN 判定排在「指数为 0」之前，于是
`(vt-pow #C(nan 0) 0)` → `NaN`，而**同一函数的实数路径**给 `1.0`、numpy 也给 1。
`%CX-POW-2` 里把零指数判定提到最前，三条路径一致。

### 改动

- `src/creation.lisp`：`vt-from-array` 的 dtype 推断补 complex 两格；
  `vt-full` 的 `:dtype` 缺省改为「复数填充 → :complex128，否则 :float64」
  （不按整数推断，避免 `(vt-full s 3)` 从 float64 变 int64）；
- `src/io.lisp`：`vt-from-sequence` 的 `:dtype` 缺省同理
  （复用已有的 `%any-complex-in-sequence-p`）；显式 `:dtype` 优先级不变；
- `src/complex.lisp`：抽出 `%cx-pow-2`、新增 `vt-float-power` 复数分派、
  `vt-cbrt` 并入 `%cx-def-unsupported`；
- `src/extensions2.lisp`：`vt-layer-norm` 新增复数分支（提前 return，
  实数路径逐字不动）；
- `test/complex-gap-test.lisp`：**新增套件（50 断言）**，已接入
  `test/run-tests.sh`（现共 47 套件）。

### 测试

- dtype 契约：`from-array`（c128/c64/2D/`:fast t`/显式 `:dtype :float64`
  仍取实部）、`from-sequence`（1D/嵌套/空/显式 dtype）、`vt-full`
  （复数/实数/整数填充）；
- 数值契约：`float_power` 用 numpy 实测值（`(2+1j)^(2+0j)=3+4j`、
  `(-1+0j)^0.5=6.12e-17+1j`）+ 边界（`nan^0=1`、`0^0=1`、`0^-1=NaN`）；
- 报错契约：`cbrt` 复数报错且消息含「不支持复数」；显式非复数 `:dtype` →
  safe-casting 拒绝；
- `layer-norm`：dtype/形状、均值 0、var≈1（容差含 eps）、虚部 0 时与实数
  路径一致、复数 gamma/beta、`:out`；
- 回归网：`conj/abs/pow/sqrt/sum/norm/matmul/take-along-axis/compress/kron`；
- 全量 **47 个套件全绿**。

> 踩坑记录：测试里**别用 `(= x x)` 判 NaN** —— 测试进程没有屏蔽浮点陷阱，
> NaN 比较会抛 `FLOATING-POINT-INVALID-OPERATION`；用 `sb-ext:float-nan-p`。

### 扫描结论：仍然「不支持复数」的清单（都是有意为之）

`floor/ceil/trunc/rint`、`div/mod/rem`、位运算族、`atan2`、`even-p/odd-p`、
`percentile/quantile`、`hypot`、`cbrt`（本轮新增）、激活函数族 —— 依据都是
「numpy 同样拒绝」或「复数无全序」。

另外 `vt-eig`/`vt-eigvals` 返回**实数**特征值是**契约**（本库 `vt-eig` 是
`eigh` 语义：只吃 Hermitian/对称，用 UPLO 指定三角 + 镜像），不是 bug；
一般复矩阵的复特征值需要另一套 Hessenberg + QR 实现，属未来工作。

---

## 2026-10-21（七）— 第七轮：复数 matmul 内核对齐实数线（标量形态，1.9×）

`vt-matmul` 的复数 2D 路径一直是**教科书三重循环**：每个元素都从 `(i,j,p)`
重算扁平下标，整段落在 `%CX-DISPATCH` 的 `(safety 1)` 里，且没有 j 维展开。
实数线 `%MATMUL-DF-ROWS` 从第一版起就是 `(safety 0)` + 指针递推 + 4 路展开，
所以同一尺寸下复数比**实数标量**内核慢 2.5–3×（比实数 SIMD 慢 10× 以上）。

本轮**不动 SIMD**（按要求），只把循环形态抄成实数线的样子。

### 消融（n=256，单线程，预分配，只计内核）

| 变体 | 耗时 | GFLOPS |
|---|---:|---:|
| 旧内核（`safety 1` + 下标重算 + 无展开） | 24.2 ms | 5.0 |
| ① 只降到 `safety 0` | 22.0 ms | 5.5 |
| ①+② 行起点只算一次、内层指针递推 | 14.1 ms | 9.5 |
| ①+②+③ j 维 4 路展开 | **13.5 ms** | **9.9** |

三项收益互不重叠，**全是代码形态问题**，与复数表示无关。

### 性能（n=192 同进程比值，`test/complex-matmul-perf-test.lisp` 内）

| 项 | 耗时 | 相对 |
|---|---:|---:|
| 新行内核（c128 单线程） | 5.3 ms | 9.6 GFLOPS |
| 旧内核形状（同参照） | 10.2 ms | **1.9×** 慢于新内核 |
| 实数标量行内核 `%MATMUL-DF-ROWS` | 2.5 ms | 新/实 = **2.1×** |

复数每元素 8 flop、实数 2 flop，**flop 比就是 4×**；实测 2.1× 说明复数的
**单位 flop 效率已超过实数标量内核**（9.6 vs 5.8 GFLOPS）。默认并行下
（`*enable-parallel-matmul*` 与实数线同一套阈值）端到端相对实数 SIMD 只剩
**2.0–4.2×**，1024² 已进入内存带宽瓶颈。

### 改动

- `complex.lisp`：新增 `%def-cx-matmul-rows` 宏，生成
  `%CX-MATMUL-CDD-ROWS` / `%CX-MATMUL-CSS-ROWS` 两个行内核，
  **签名与 `%MATMUL-DF-ROWS` 完全一致**（`C += A@B`，不清零）；
- `complex.lisp`：`%CX-MATMUL-RUN` 改为「清零 + 选内核 + `CALL-PARALLEL-GEMM`」，
  直接复用 `linalg.lisp` 的调度器 → 复数与实数的并行语义（阈值/线程数/行切法）
  逐项一致，不引入第二套线程策略；
- `complex.lisp`：`%CX-GEMM-STRIDED`（einsum 的 BMM 通道）新增 (a) 分支 ——
  布局与「连续 NN」逐项相同时委托同一个行内核；(b) 行内连续 / (c) 通用 strided
  两条循环保留。累加次序不变，**einsum 与 `vt-matmul` 仍逐位相等**；
- `test/complex-matmul-perf-test.lisp`：**新增套件（53 条断言）**，已接入
  `test/run-tests.sh`（现共 46 个套件）；
- `test/complex-bench.lisp`：matmul 段扩成 128/256/512 三档 × 串行/并行。

### 被实测否掉的两个方案（记下来免得下次再试）

| 方案 | n=256 | 结论 |
|---|---:|---|
| SoA（拆 re/im）+ B 打包成 j-major + j4 | 22.9 ms | 更慢。SBCL 的复数 `*`/`+` 是**开码**的，specialized 数组里的 `(complex double-float)` 就地读写、不装箱 |
| SoA + i-j-p 单累加器 | 75.2 ms | 慢 5×。`B[p,j]` 的 p 步长是 n，每取一个元素跨一行，缓存全废 |
| 把 `p=0` 折进内核（省掉清零趟） | 14.6 ms | 更慢。j 展开的初始化代码写 4 份，挤掉内层寄存器分配 |

### 数值一致性

行内核的累加次序仍是 i→p→j 递增，与旧内核**逐位相同**；
`vt-matmul`（串行/并行）与 `vt-einsum`（NN/NT/转置输出）三方差分 maxabs = 0。
三张量链 `ij,jk,kl->il` 与 `(a@b)@c` 因缩并次序不同只保证 ~2e-15 一致。

### 测试

- **新增 `test/complex-matmul-perf-test.lisp`（53 断言）**：手写三重循环参照
  逐位对照、尾路径（n=1,2,3,5,6,7,9 专压 4 路展开的剩余列）、退化与零尺寸、
  切片视图（offset≠0 / 转置视图 / 列切片）、批量与 batch 广播分片逐位、
  einsum 三通道逐位、c64/c128、`:out`、并行（含真正超过阈值起线程的 200³ 规模）；
  性能护栏用**同进程比值**：新内核 ≥ 1.3× 旧形状（实测 1.9×）、
  ≤ 3.5× 实数标量行内核（实测 2.1×，理论下限 4×）、并行不改结果且不慢于串行；
- 全量 **46 个套件全绿**（`bash test/run-tests.sh`）。

> 遗留：SIMD 版复数 GEMM（拆 SoA 视图 + 4 条实 GEMM 复用 `simd-matmul.lisp`
> 的分块与线程框架）见 CONVENTIONS §16.27.6，是下一轮的事。

---

## 2026-10-21（六）— 第六轮：einsum 的实数 GEMM 接入 SIMD（5.7× → 1.0×）

第五轮把**复数** einsum 接到了 matmul 快路径上，但**实数**那条一直是另一套：
`vt-matmul` 走 `simd-matmul.lisp` 的 AVX/FMA + 64×256×256 分块内核，
`vt-einsum` 的批量 GEMM 通道却只调 `linalg.lisp` 的标量内核（4 路手写展开、
无 SIMD、无分块）。于是同一台机器上 einsum 比 matmul 慢 5.7~6.4×（单线程）。

本轮把两条路径接通，并顺带修掉两个既有的布局相关性能坑。

### 性能（512² float64，**单线程**，µs，3 次最好值）

| 表达式 | 改前 | 改后 | 改前 vs `vt-matmul` | 改后 vs `vt-matmul` |
|---|---:|---:|---:|---:|
| `vt-matmul`（基准） | 31,141 | 35,596 | 1.00× | 1.00× |
| `"ij,jk->ik"`（NN） | 196,033 | **32,485** | **6.30×** | **0.91×** |
| `"ij,kj->ik"`（NT） | 381,070 | **29,818** | **12.24×** | **0.84×** |
| `"ij,jk->ki"`（转置输出） | 609,077 | **36,112** | **19.56×** | **1.01×** |

2048² 多线程（32 核）同样收敛：1.78× / 1.71× / 13.82× → **0.99× / 0.99× / 0.99×**
（转置输出从 48.3 s 降到 3.46 s）。1024² 多线程 0.93~1.02×；256² 0.95~1.02×；
小矩阵（8/16/32²，低于 SIMD 阈值）两者都走标量，无退化。
复数路径与上一轮持平（0.96~1.03×）。

> 转置输出（`"ij,jk->ki"`）的「改前」是**本轮之前就存在**的老问题 ——
> 通用三重循环的内层沿 `so-k`（= 整行）走，每步换一条 cache line。
> 本轮按「临时连续 C + 分块 scatter」一并解决，不是本轮引入的回归。

### 改动

- `linalg.lisp`：新增裸数据 GEMM 钩子 `*SIMD-GEMM-RAW-FN*`（`simd-matmul.lisp`
  注册 `%SIMD-GEMM-RAW`），语义与 `%MATMUL-*-FAST-KERNEL` 一致（`C += A·B`）；
- `linalg.lisp`：新增 `define-einsum-gemm`，生成 `%EINSUM-GEMM-{F64,F32,I64,I32}`，
  把 einsum 的任意步长归一化成「连续 C += 连续 A·连续 B」：
  NN 直通 / NT 分块转置 B / C 非行主序走临时缓冲 + 分块 scatter / 其余兜底；
- `einsum-execute` 的 `gen-bmm-logic` 每个类型只留一行调用（不再内联判断）。

### 两个踩过的坑（详见 CONVENTIONS §16.26）

1. **别在 macrolet 里做布局归一化**：第一版把判断就地展开，结果**一行没改的
   兜底路径**从 711 ms 退化到 947 ms（+33%）—— 函数体膨胀后 SBCL 对它的
   寄存器分配变差。挪成独立 `defun` 后回到基线。
2. **大对象的 `:initial-element` 是 calloc 不是 memset**：2 MB 临时缓冲用
   `:initial-element 0.0d0` 时「分配 + GEMM」要 111 ms（复用缓冲只要 46 ms）
   —— 零页由 31 个工作线程各自首次触达时 fault，并发抢 `mmap_lock`。
   改成 `(make-array ...)` + `(fill buf 0.0d0)` 后回到 47 ms。

### 数值一致性

四条路最终都进同一个内核、同一分块顺序，所以 **einsum 与 `vt-matmul` 逐位
相同**（20 项差分，maxabs = 0），与 numpy `@` 相对差 ≤ 7.7e-15。相对旧的标量
三重循环只有 FMA 少一次舍入（~1e-12，更接近精确解）。

### 测试

- **新增套件 `test/einsum-gemm-layout-test.lisp`（36 条断言，已接入
  `test/run-tests.sh`，现共 45 个套件）**：四条布局路径 × 各档尺寸、
  批量（含广播与批量 NT）、f32/i32/i64/c128/c64、转置输入 / 带 offset 切片 /
  非连续 `:out` / 小矩阵、以及 SIMD 关与单线程下的不变性。
  主断言是「与 `vt-matmul` 逐位相等」——不依赖外部期望值文件（§10.4），
  且同时锁住「语义没变」和「四条路径数值一致」；
- 另写一次性差分脚本做 numpy 逐值对照（NN / NT / 转置输出，最大相对差 7.7e-15），
  以及 SIMD 开/关的 A/B 性能对照（关掉 `*SIMD-MATMUL-ENABLED*` 后两条路径
  持平，直接证明差异来自内核而非 einsum 的通道判定）；
- 全量 **45 个套件、4655 条用例全绿**；冷编译 27 文件（4 GB 动态空间，
  1 分 53 秒，峰值 3.0 GB），0 条非 style 警告。

---

## 2026-10-20（五）— 第五轮：einsum 快路径 + 两个通用特性缺口

前四轮解决「对不对」，本轮解决「快不快」和「有没有」：复数 einsum 提速 81×，
顺手挖出并修掉批量复数 matmul 的两个静默错误，再补两个 numpy 有而 clvt 没有的
通用特性（与复数无关）。

### 性能：复数 einsum 走 matmul 快路径（§16.21）

| 表达式（256×256 complex128） | 改前 | 改后 | 倍数 |
|---|---|---|---|
| `vt-einsum "ij,jk->ik"` | 2526 ms | **31 ms** | **81×** |
| `vt-einsum "i,i->"` | 9 ms | **<1 ms** | ~30× |
| `vt-einsum "ij,ij->ij"`（1M 元素） | 138 ms | **8 ms** | **17×** |

根因：三条快路径（逐元素 / 全收缩 / BMM）的判据全是实数 dtype 谓词，
复数直接落到通用迭代状态机（每个元素「补码递推 + 逐张量泛型 aref」）。

修法：新增 `%CX-GEMM-STRIDED`（带 strides 的复数 GEMM，含连续快分支与
通用 strided 分支）与 `%CX-EINSUM-DOT`，由 `complex.lisp` 在加载期注册到
`linalg.lisp` 的两个钩子上（`*CX-EINSUM-GEMM-FN*` / `*CX-EINSUM-DOT-FN*`）
——因为 `linalg.lisp` 先编译，不能直接引用复数层（同 `*SIMD-MATMUL-2D-FN*` 模式）。
判据里必须带上钩子本身，否则钩子为 NIL 时会静默返回空结果。

### 修复：批量复数 matmul 的两个静默错误（§16.22）

做差分时发现 `einsum "...ij,...jk->...ik"` 与 `vt-matmul` 差 3.3（不是精度问题）。
三方对照（einsum / matmul / 手写三重循环）定位：

| 缺陷 | 现象 | 根因 |
|---|---|---|
| offset 丢失 | 每个 batch 都读到基数组开头 | `(vt-slice (vt-ravel a) ...)` 的片段已连续 → `vt-contiguous` 零拷贝 → offset 非零，而 2D 内核按扁平下标从 0 起算 |
| 不支持 batch 广播 | `(1,4,5)@(3,5,6)` 报「重塑失败」 | 用 reshape 切 batch，隐含两侧 batch 形状相同 |

与 `manip.lisp` 的 `%flat-offset0` 一节同一类根因。改为「strides 递推 batch 偏移 +
offset 透传」，广播按「维长 1 → stride 0」处理，与实数 SIMD 路径一致。
`%CX-DOT-1D` 顺手补了同样的 offset 假设。

### 新特性：`vt-unique :axis`（§16.23，通用）

numpy 2.x 的按轴去重，此前**实数路径也没有**。严格照 numpy 算法实现
（moveaxis → reshape 成行 → 稳定字典序排序 → 相邻行全元素比较），
并把 numpy 内部的两条分支如实分层：

- 展平模式：多个 NaN **折叠为一个**；
- 按轴模式：含 NaN 的行**各自保留**（`NaN != NaN`），`counts` 全为 1。

`return-inverse` 在 axis 模式下是**一维**（长度 = 该轴长度），
可用 `(vt-take u inv :axis axis)` 重建输入。复数直接复用同一实现
（比较原语按 `complexp` 分派，字典序 = 实部→虚部）。

### 新特性：`vt-trace :offset`（+`:axis1`/`:axis2`）（§16.24）

`vt-diagonal` 早就支持 `:offset`，`vt-trace` 却硬编码 2 维。
本轮补齐 `(matrix &key offset axis1 axis2 dtype out)`，关键点是
**收缩轴是 `-1` 而不是全部**（`vt-diagonal` 把对角轴放末尾），
否则 3D 输入的 batch 轴会被一起吃掉。越界 offset 返回 0（同 numpy），
秩 < 2 用 `error` 报错（`assert` 会被 `(safety 0)` 剔除）。

### 测试

- 新增 `test/setops-axis-trace-test.lisp`：**56 断言**（37 项 numpy 2.4.6 差分
  + 契约 + 实数回归网），期望值由 `test/gen_setops_axis_expected.py` 生成。
- 全量：**44 个套件全绿**。

### 合入前审阅与独立验证

本轮补丁在合入前做过一次独立复核（不依赖作者自述，另写了一套差分脚本
重算期望值），结论：**取证通过，可合入**；顺带修正 1 处、记录如下。

**独立 numpy 差分（40 项，逐值一致）**

| 组 | 覆盖 | 结果 |
|---|---|---|
| `vt-unique :axis` | 19 项：f64/c128/i64、`axis=0/-2/1/2/-1`、`return-index`/`inverse`/`counts` 任意组合、含 NaN 行、`(0,3)`、`(2,0)`、秩 1（axis 被忽略）、3D 各轴 | 全一致 |
| `vt-trace` | 8 项：offset `0/+1/-1/+9(越界→0)`、3D `axis1/axis2` 两种组合、复数 | 全一致 |
| 批量 matmul / einsum | 13 项：`(2,3,4)@(2,4,5)`、batch 广播 `(1,4,5)@(3,5,6)`、**带非零 offset 的切片**、2D@3D、`ij,jk->ik`、`i,i->`、`ij,ij->ij`、`ij,ji->`、`ij,jk->`、`ij->ji`、转置输入（strided 分支） | 全一致 |

追加 einsum 形态（complex128 / complex64 各 5 项，全一致）：
`ij,kj->ik`（NT）、`ij,jk->ki`（转置输出）、`i=1`、`k=1`、`ij,jk,kl->il`（三张量链）。

**性能 A/B（同进程内把钩子置 NIL 复现「改前」）**

| 表达式 | 钩子关（改前） | 钩子开（改后） | 倍数 |
|---|---:|---:|---:|
| `vt-einsum "ij,jk->ik"` 256×256 c128 | 1,629 ms | **19.3 ms** | **84.6×** |
| `vt-matmul` 同规模（对照） | — | 19.3 ms | 与 einsum 持平 ✓ |
| `vt-einsum "i,i->"` 1M c128 | 156 ms | **0.94 ms** | **166×** |

（上表为审阅机实测；作者原表 2526→31ms / 81× 为另一台机器，比值一致。）

**审阅修正 1 处**：`vt-unique :axis` 在 `axis≠0` 时原本返回 `vt-transpose` 的
**视图**（`vt-contiguous-p` = NIL），而展平路径与 numpy 都返回新数组。
已补 `vt-contiguous` 落地（最多一次「唯一行」大小的拷贝），两条路径一致。

**边界行为复核**：`:out` 预填 999 时结果被**覆盖**而非累加；非连续 `:out`
（转置视图）走 strided 分支且与 `vt-matmul` 一致；batch 不可广播时报
`vt-matmul: 批量维不可广播：(2) vs (3)`；`(0,3)@(3,2)` 得 `(0,2)` 空结果；
`vt-trace` 秩 < 2 报错、`axis` 越界报错、`(2,0)` 的 unique 得 `(1,0)`。

---

## 2026-10-20（四）— 第四轮：结构算子收尾 + 数值压测固化

第三轮把线代补完；本轮用「全导出符号逐个试调」扫尾，并把随机压测写进测试。

### 补齐（3 处，都是「8 路元素类型特化漏了第 9、10 种」）

| 函数 | 修复前 | 根因 |
|---|---|---|
| `vt-outer` | `invalid number of arguments: 3` | 复数分支写成 `(vt-reshape ... :dtype dt)`，而 `vt-reshape` 只接 `(vt shape)` —— **复数外积从来没跑通过**，实数走原实现所以一直正常 |
| `vt-any` / `vt-all` | `unsupported input dtype (COMPLEX DOUBLE-FLOAT)` | 归约内核 8 路特化，复数落入 `(t (error ...))` |
| `vt-take-along-axis` | `不支持的输出元素类型 (COMPLEX DOUBLE-FLOAT)` | 逐元素内核只展开 8 种实数元素类型 |

修复策略：**能用既有算子组合的就不新写内核**。
`vt-any/vt-all` 先把复数转成 int8 真值掩码（`vt-nonzero-p`）再交给原实现，
自动继承 `axis`/`keepdims`/`:out`/空输入单位元（any→0、all→1）；
`vt-outer` 改为广播乘 + 按需 astype，并补 `:out` 硬契约。

复数真值语义（numpy 一致）：任一分量非零即为真，`bool(0+0j)=False`、
`bool(nan+0j)=True`。

### 数值压测固化（108 条断言，`stress-tests`）

差分对照只能覆盖"我事先想到的矩阵"。新增：
- **随机矩阵**：确定性 LCG（固定种子，可复现）1×1 / 3×3 / 5×5 / 2×5 / 5×2
- **恒等式**：`Q·R=A`、`Q^H Q=I`、`U·S·V^H=A`、`U^H U=I`、`V^H V=I`、
  `A·V=V·Λ`、`A·A⁻¹=I`、`A·x=b`、Moore–Penrose 四条
- **交叉验证**：`σ²` 必须等于 `eig(A^H A)` —— SVD（单边 Jacobi）与
  eig（厄米特 Jacobi）是两条独立算法，互证可抓单边系统性偏差
- **退化**：全零矩阵（σ 全 0 / rank 0 / cond ∞ / inv 报错）、秩亏（σ₂≈0）、1×1
- **SPD 族**：`B^H B + nI`（1/2/3/6 阶）→ Cholesky + eig，检验 `l_jj>0` 与特征值全正
- 容差按 `scale = max(1, ‖A‖∞)` 缩放，避免病态误报 / 良态漏报

### 顺带确认「本来就是好的」的算子

eye/identity/ones/full/*-like/tri/triu/tril/triu-indices/tril-indices/
reshape/ravel/flatten/squeeze/expand-dims/narrow/swapaxes/moveaxis/rollaxis/
rot90/tile/pad/insert/delete/append/block/repeat/resize/trim-zeros/meshgrid/
select/choose/compress/extract/where/broadcast-to/broadcast-arrays/
copy-to!/write/fill/stack/vstack/hstack/dstack/column-stack/split/hsplit/vsplit/
concatenate/concat/dot/multi-dot/matmul/outer/kron/vander/diag/diagonal/diagflat/
fill-diagonal/trace/topk/sort/argsort/median/nonzero/argwhere/flatnonzero/
count-nonzero/zero-p/nonzero-p/array-equal/array-equiv/isclose/allclose/
logical-and/or/not/xor/to-list/to-array/print —— 全部直接可用（dtype 无关）。

### 文档

- `CONVENTIONS.md` §16.19.8（第四轮补齐：三处缺口的症状/根因/修复策略）、
  §16.19.9（为什么差分对照不够：压测的五种手法与容差口径）、
  §16.19.10 测试计数更新、§16.20 变更文件清单

### 测试与回归

- complex-linalg-test **108 → 216**、complex-ops-test **131 → 149**
- 其余套件全绿：complex 136、core 284、test-all 50（642 断言）、io-interop 77、
  probes 496、probes2 405、extensions3 264、legacy 284、numpy-convention 69、
  contract 136、out-contract-v2 56、nan-broadcast 31

---

## 2026-10-20（三）— 第三轮进化：复数线性代数 + 算子补齐（收尾）

前两轮做完「逐元素」与「结构层」；本轮把**复数线代整块**从硬报错变成可用，
并修掉 8 处「复数输入静默只算实部」。

### 新增：src/complex-linalg.lisp（复数线代）

| 函数 | 算法 | 说明 |
|---|---|---|
| `vt-lu` | 部分选主元 LU | 选主元按 `\|a\|`（复数无全序） |
| `vt-det` / `vt-solve` / `vt-inv` | 基于 LU | 奇异 → det=0 / solve·inv 报错 |
| `vt-cholesky` | 厄米特 Cholesky | `A = L·L^H`；`:upper` 返回 **L^H**（不是 L^T） |
| `vt-qr` | 复 Householder | `:reduced`/`:full`；Q 用**逆序左乘**累积 |
| `vt-svd` | 单边 Jacobi（复旋转） | 不形成 A^H A，无条件数平方问题 |
| `vt-eig` / `vt-eigvals` / `vt-eigvalsh` | 厄米特 Jacobi | 特征值实数、降序；对角虚部清零 |
| `vt-pinv` / `vt-lstsq` | 基于 SVD | `A⁺ = V·Σ⁺·U^H` |
| `vt-matrix-rank` / `vt-cond` / `vt-matrix-power` | SVD / 复用 | rank 改用奇异值（原实现取实部） |

### 修掉的「静默丢虚部」（8 处）

| 函数 | 修复前 | numpy | 根因 |
|---|---|---|---|
| `vt-cov` / `vt-corrcoef` | float64 | 复厄米特 | `%e3-as-1d-float-transform` 首行 `ensure-vt :float64` |
| `vt-expm1` / `vt-log1p` | float64 | 复数 | `%float-map` 只认实数 |
| `vt-nanpercentile` / `vt-nanquantile` / `vt-interp` | float64 | **报错** | `astype :float64` 取实部 |
| `vt-fromiter` | float64 | 复数 | 默认 `:dtype :float64` |
| `vt-nextafter` 等 9 个 ufunc | 静默算实部 | **报错** | 无复数分派 |
| `vt-matrix-rank` | 取实部消元 | 奇异值 | 同上 |
| `vt-scale`（公共函数） | complex×实数标量 → dtype 判成 float64，写缓冲报 `%DOUBLE-FLOAT` | 保持 complex | dtype 推断只列 4 种实数类型 |

### 新增/放开的算子

- **新支持**：`vt-expm1` `vt-log1p` `vt-sinc` `vt-sign`（z/|z|）
  `vt-clip`/`vt-clamp`（**字典序**整元素夹取）`vt-standardize`
  `vt-cov`/`vt-corrcoef`（`X·X^H`）`vt-fromiter`（dtype 推断）
  `vt-einsum`（放开白名单，走通用累加路径）
- **新拒绝**（numpy 亦拒绝）：`nextafter` `spacing` `logaddexp` `hypot`
  `copysign` `signbit` `deg2rad` `rad2deg` `bincount` `histogram`
  `nanpercentile` `nanquantile` `interp`

### 三个「只在非方阵/复数上暴露」的真实坑

1. **约化 QR 的 Q 不能截断后顺次右乘** —— 反射器支撑长度 `m-i` 通常超过
   `k = min(m,n)` 列，内积会读不存在的列。方阵/宽矩阵因为 Q 恰好是完整
   `m×m` **碰巧正确**，只有 tall（m>n）错。正解：`Q = H_0···H_{k-1}·I_{m×k}`，
   从 `i = k-1` **递减左乘**。
2. **`ld`（行距）≠ 行数** —— 行主序 m×n 的行距是 **n**。本轮因写错 `ld`
   踩了 3 次（`vt-lu` 非方阵、QR 的 Q、SVD 的 U），全靠 `(safety 1)` 的
   边界检查当场报错才定位到。
3. **Cholesky 的 `:upper` 必须是 `L^H`** —— 实数下 `L^T = L^H`，
   复数下写成 `transpose` 会静默返回 `L^T`。

另：`vt-eig` 的对角虚部必须清零，否则非厄米输入会收敛到**错的特征值**
（`[[2+1j,1],[1-1j,3+2j]]` 给出 3.925/1.075，numpy 是 4/1）。

### 文档

- `CONVENTIONS.md` **§16.19 线性代数与算子补齐**：共轭转置这条唯一真分叉、
  LU/Cholesky/Householder/Jacobi 的复公式、约化 QR 陷阱、对角虚部清零原因、
  8 处静默错答案清单、已知差异 **D-C9..D-C13**（histogram 拒绝 / eig 是 eigh 语义 /
  SVD·QR 向量带相位自由度 / lstsq residual 是 2-范数）、声明策略

### 测试

- `test/complex-linalg-test.lisp` **108 断言** = 46 项 numpy 2.4.6 差分对照
  + 结构恒等式（`P·A=L·U`、`L·L^H=A`、`Q·R=A`、`Q^H Q=I`、`U·S·V^H=A`、
  `U^H U=I`、`V^H V=I`、`A·V=V·Λ`、Moore–Penrose 四条、`einsum ≡ 专用内核`）
  + 契约（`:out`、奇异/非方阵/非正定报错、dtype、实数回归网）
- `test/gen_complex_linalg_expected.py` 生成 `test/complex-linalg-expected.sexp`
  （禁手写期望值；线代比较用相对容差 1e-9，逐位相等无意义）
- 已接入 `test/run-tests.sh`

### 回归

- complex-linalg 108/108、complex 136/136、complex-ops 131/131、core 284/284、
  test-all 50/50（642 断言）、io-interop 77/77、differential-probes 496/496、
  probes2 405/405、extensions3 264/264、legacy 284/284、numpy-convention 69/69、
  contract-regression 136/136、out-contract-v2 56/56、nan-broadcast 31/31 —— 全绿
- 过程中差分探针 **p0109/p0110** 抓到 `vt-matrix-rank` 的新增 `&optional tol`
  把实数默认值 1e-10 覆盖成 NIL 的回归，已修

---

## 2026-10-20（二）— 修复：`*read-default-float-format*` = double-float 时 complex64 崩溃

### 症状（用户环境 SBCL 2.6.9.446 / numpy 2.5.0）

```
差分对照完成：PASS=97 FAIL=0
--- 契约断言 ---
Unhandled TYPE-ERROR ... The value #C(0.0 0.0) is not of type (COMPLEX SINGLE-FLOAT)
                            when binding SB-C::ITEM
```

### 根因（三段链，已用用户同版本 SBCL 精确复现）

1. 用户 `.sbclrc` 把 `*read-default-float-format*` 设为 `double-float`；
   源码里的**裸浮点字面量**（`0.0`/`0.5`/`180.0`）精度由读取时该变量决定。
2. 于是 `(complex 0.0 0.0)`（complex64 的零值）被读成 `(complex 0.0d0 0.0d0)`。
3. 往 `(simple-array (complex single-float) (*))` 填 double 复数时，SBCL 的
   `vector-fill` 为每种元素类型预装的转换器
   `(lambda (sb-c::item) (declare (type (complex single-float) sb-c::item)) ...)`
   抛 TYPE-ERROR —— 报错里的 `#C(0.0 0.0)` 正是 double 在该读取格式下的打印形式，
   具有极强误导性。（`(safety 0)` 旁路更糟：不报错，直接写错位模式。）

### 修复（三层）

- **`clvt.asd`**：新增 `:around-compile`，在编译/加载每个组件时绑定
  `*read-default-float-format*` = `single-float`（read 发生在该动态范围内），
  从根上消除对用户 init 的依赖。
- **源码**：复数/单精度关键路径的裸字面量显式加 `s0`
  （`(complex 0.0s0 0.0s0)`、`0.5s0`、`2.0s0`、`180.0s0`、`0.6931472s0`、
  `0.31415927s1`、`(cval 0.0s0)`、`(/ 1.0s0 norm)`、`0.5s0` 直方图边界 等），
  即使绕过 ASDF 直接 `load` 源文件也不漂移。
- **测试**：`run-tests.sh` 两处 `sbcl` 调用插入
  `--eval "(setf *read-default-float-format* 'single-float)"`；
  `complex-ops-test` 增加 3 条「读取格式哨兵」断言
  （`(type-of (vt-dtype-default-value :complex64))` 等），复发时立刻报红。

### 文档

- `CONVENTIONS.md` §16.18.6：完整复盘（症状/根因链/三层修复/用户侧 fasl 缓存提醒），
  并固化约定：**源码里的裸浮点字面量一律视为 single-float；double 必须写 `d0`**。

### 测试

- 在**用户同版本** SBCL（2.6.9.446-9e249df9d）上复现并验证修复
- double-float 读取环境：complex-ops 131/131、complex 136/136、core 284/284
- 默认环境：全量 13 套件全绿

> 用户侧注意：更新后需 `(asdf:load-system :clvt :force t)`（或清
> `~/.cache/common-lisp`），否则可能继续加载用 double 编译的旧 fasl。

---

## 2026-10-20 — 第二轮进化：复数数据搬运层（转换 / 拼接 / 剪切 / 集合 / 缺失值 / 缩并）

第一轮（§16）解决逐元素算术与归约；本轮补结构层，新增 `src/complex-ops.lisp`
（紧随 `src/complex.lisp` 加载，同样「实数路径零改动 + 只对含复数调用分派」）。

### 修掉的「静默错答案」（比不支持更危险）

| 函数 | 修复前 | numpy 语义 |
|---|---|---|
| `vt-nanmean` | 返回**实数**（实部均值） | 复数 |
| `vt-nanvar` / `vt-nanstd` | 数值错误 | 实数 = mean(\|x−mean\|²) |
| `vt-nancumsum` / `vt-nancumprod` | **虚部清零** | 保复数（NaN→0/1 后累积） |
| `vt-vdot` | 不取共轭 | **第一操作数取共轭** |
| `vt-cumsum/cumprod` 的 `axis=nil` | 形状不展平（2-D 进 2-D 出） | 展平为 1-D |

### 新增能力

**转换**
- `vt-nan-to-num`：实部/虚部各自替换（`nan`→0、`posinf/neginf`→该精度 ±最大有限值）
- `vt-view` 支持 `:dtype` 重解释：`complex128 ↔ float64`、`complex64 ↔ float32`，
  shape 省略时按 numpy 规则只缩放最后一维；三种调用形式
  `(vt-view a shape)` / `(vt-view a shape :dtype dt)` / `(vt-view a :dtype dt)`
- `vt-repeat`：三处 8 路元素类型分派各追加 2 个复数分支（标量/列表 repeats、任意 axis）
- 持久化：npy 新增 `<c16` / `<c8` 编解码（与 numpy 双向互通，含 NaN payload）；
  lisp 格式复数编码（分量全有限 → `#C(re im)`，否则 `(:CX re im)` 保证可读性）

**集合 / 搜索**
- `vt-unique`（含 `return-index/inverse/counts`）、`union1d`、`intersect1d`、
  `setdiff1d`、`setxor1d`、`in1d`、`isin`、`searchsorted`（`:left`/`:right`）
- 统一走「字典序排序 + `%cx-key` 哈希」：NaN 折叠为一个、`1.0s0` 与 `1.0d0` 同键
- `vt-digitize` 对复数给出确定性错误（numpy 亦报 `x may not be complex`）

**缺失值（nan 族）**：`nansum/nanprod/nanmax/nanmin/nanargmax/nanargmin/nanmean/nanvar/nanstd/nanmedian`
- 全 NaN 切片：sum→0、prod→1、max/min→`NaN+0j`、mean→`NaN+NaNj`、var/std→`NaN`、arg→报错
- 实现为「掩码 + 单位元替换 + 既有复数归约内核」的向量化写法（一次覆盖所有 axis/keepdims）

**缩并 / 卷积**：`vt-inner`、`vt-tensordot`（transpose+reshape+复数 matmul）、
`vt-correlate`（第二操作数**取共轭**）、`vt-convolve`（**不取共轭**，对 flip(v) 相关）

### 文档
- `CONVENTIONS.md` **§16.18 数据搬运层**：静默错答案清单、语义总表（按 numpy 2.4.6 实测）、
  已知差异 **D-C7**（view 重解释是副本，非共享视图）、**D-C8**（take-along-axis 同秩约束）、
  声明策略（能用既有内核组合就不新写循环）、测试口径
- §16.17 路线图：标注 `.npy` 复数互操作已完成；§16.19 变更文件清单补齐

### 测试
- `test/complex-ops-test.lisp` **128 断言** = 97 项 numpy 2.4.6 差分用例 + 31 条契约断言
  （dtype 提升 / 切片别名 / `:out` 硬契约 / 未支持组合报错 / NaN 折叠 / c64 精度 /
  两种格式往返 / 7 条实数回归网）
- `test/gen_complex_ops_expected.py` 生成 `test/complex-ops-expected.sexp`（禁手写期望值）
- 已接入 `test/run-tests.sh`（`SUITES` + `DEFAULT_ORDER`）

### 回归
- complex 136/136、complex-ops 128/128、core 284/284、io-interop 77/77、test-all 50/50 全绿

---

## 2026-10-19（三）— 复数层性能优化（第一轮进化 v1.1）

优化目标：把「第一轮只为正确性、全部走泛型 walker」的复数实现拉到接近实数快路径，
同时**不引入无脑 `(speed 3) (safety 0)`**。1M 元素实测（`test/complex-bench.lisp`）：

| 算子 | 优化前 | 优化后 | 倍数 |
|---|---:|---:|---:|
| add | 135 ms | 7 ms | 19× |
| mul | 157 ms | 7 ms | 22× |
| div | 184 ms | 8 ms | 23× |
| sqrt | 453 ms | 20 ms | 23× |
| abs | 889 ms | 26 ms | 34× |
| sum | 92 ms | 1 ms | 90× |
| cumsum | 288 ms | 6 ms | 48× |
| sort | 2316 ms | 136 ms | 17× |
| compare `<` | 778 ms | 3 ms | 260× |
| isnan | 402 ms | 2 ms | 200× |
| matmul 256² | 1757 ms | 39 ms | 45× |
| argmax | 912 ms | 5 ms | 173× |

### 改动

**src/map-reduce.lisp**
- 新增 `+cx-elem-types+` + `+cx-inline-ops+` 白名单与 `%cx-inline-op-p`；
  `%elem-type-cond(-strided)`、`%vt-inline{1,2,3}[-strided]-fast`、`%vt-mask{1,2}[‑strided]‑fast`
  在**算子命中白名单**时追加 2 个复数分支 → 复数算术/数学/比较/谓词进入
  「类型声明驱动 + 算子内联」的循环（原来一律落到 `vt-map` 的 funcall walker）。
  白名单机制避免 200+ 展开点被 10 路广播撑爆（历史教训：8×8 全展开 Heap exhausted）。

**src/dtype.lisp**
- 新增免屏蔽快判 `%cx-nan-p` / `%cx-inf-p` / `%cx-finite-p`：内核在
  `with-float-safe` 之下运行，不必每元素再屏蔽/恢复陷阱
  （`vt-float-nan-p` 单次约 200ns，每元素判 2~4 次 → 1M 元素白付 400~800ms）。
- `%c-lex-compare` / `%clt` / `%cle` / `%cgt` / `%cge` / `%cneq` 全部 **inline**
  （必须写在 `defun` 之前，SBCL 才记录内联展开）。

**src/complex.lisp**
- `%cx-def-unary` / `%cx-def-compare` / `%cx-def-predicate` / `vt-conj` 改为走
  `vt-fast-map` 宏（原来走 `vt-map` 函数，编译期拿不到算子字面量 → 无法内联）。
- 新增 `%def-cx-to-real-run`（complex→real 专用内核，`(speed 3) (safety 1)`）+
  `%cx-abs-run` / `%cx-real-run` / `%cx-imag-run` / `%cx-angle-run`：
  abs/real/imag/angle 是跨 dtype 映射，进不了 `dtype-check` 快路径。
- 归约重写：`%cx-dispatch`（按元素类型绑定 simple-array）+ `%def-cx-reducer`
  （全局单循环 / 单轴 fiber 循环）+ `%cx-arg-reduce-run`；多轴仍走通用 odometer 兜底。
- 扫描 `%cx-scan-run`、矩阵乘 `%cx-matmul-run`（i-k-j 次序，内层连续）类型特化。
- 排序重写：`%def-cx-fiber-sorter`（插入排序 + 自底向上归并，比较器内联、零 cons），
  `%cx-sort-fiber-strided` 用「收拢→排→写回」处理任意步长 fiber；
  argsort 改为 fixnum 下标向量 + STABLE-SORT（不再每元素 cons 一个 (值 . 下标)）。
- `%cop-sqrt` 换成 C99 公式（hypot + 1 sqrt + 2 除法，350ns → 20ns/元素），
  `%cop-abs` 加常规量级快路径（极端量级才走带缩放的 hypot）。
- `%cx-reduce` / `%cx-scan` / `%cx-sort-axis` / `%cx-arg-reduce` 外层统一
  `with-float-safe`（配合内核里的免屏蔽快判）。

### 声明策略（写进 CONVENTIONS §16.14.1）

| 层级 | 声明 | 理由 |
|---|---|---|
| 用户入口 / 边界 | 默认安全级 | 契约检查与错误信息都在这里 |
| 普通内核（`%cx-*`） | `(speed 3) (safety 1)` | 索引可证不越界，但保留检查实测 < 5%，避免静默踩内存 |
| 宏展开逐元素循环（map-reduce） | `(speed 3) (safety 0)`（既有） | 索引由循环上界逐元素推进，编译器可完全证明 |
| 通用 stride 路径 | 禁止 `safety 0` | 用户 stride/shape 错误必须是 Lisp 条件 |

### 测试

- 新增 `test/complex-bench.lisp`（1M 元素 × 14 项，含 float64 对照）并登记为基准套件
  （`SUITES[complex-bench]` + `BENCH_SUITES`，默认跳过、`--all` 时运行）。
- 回归全绿：complex-test 136/136、core-test 284/284、differential-probes 496/496、
  differential-probes-test2 405/405、extensions3 264/264、legacy 284/284、
  io-interop 77/77、nan-broadcast 31/31、test-all 50/50。

---

## 2026-10-19（二）— 修复：新增导出导致 p0395/p0392 失败 + codegen 门禁静默失效

### 现象
上游合并复数补丁后 `differential-probes-test2` 恒红：`403/405`，`p0392`（data）与
`p0395`（data）。

### 根因（四条，全部实测确认）

1. **`p0395` 把导出数量写死成常量 371**，而运行时 `*vt-fun-list*` 只有 367：
   `vt-real / vt-imag / vt-conj / vt-angle` 早在 `extensions3.lisp` 就存在并已导出，
   复数层又**重复导出**了一次。`defpackage` 允许重复，静态解析因此多算 4 个。
2. **`refresh-vt-fun-list` 用 `DO-SYMBOLS`**（把内部辅助符号也算进来），与其
   docstring「导出符号」矛盾，且结果依赖"何时调用"——运行时重刷一次就会从 367 涨到 406。
3. **`p0392` 是真语义 bug**：`%cimag` 的实数分支写成 `(* 0 x)`，对负数产生 **-0.0**，
   而 numpy 的 `np.imag(实数)` 恒为 **+0.0**；探针 `:exact` 做位级比较（含符号位）故判不等。
   同源的 `%carg` 也把 `-0.0` 漏判成 0（numpy: `angle(-0.0) = π`）。
4. **codegen 门禁静默失效**：`check_key_sync.py` 用「无参调用生成器」做漂移检查，
   而两个生成器都是 `sys.argv[1]` → 无参即 `IndexError`，文件未被重写，比对自然"通过"。
   后果：9 行过期浮点期望值（旧 `str()` 口径 `0.272799`，现应为 `0.27279899999999996`）
   一直没被发现。

### 修复

- `src/dtype.lisp`：`%cimag` 实数分支返回同类 **+0.0**；`%carg` 增加 NaN 传播与
  **符号零**判定（`float-sign` 判 `-0.0` → π，返回 +0.0）。
- `src/package.lisp`：删除 4 个重复导出并加注释；`refresh-vt-fun-list` 改用
  `DO-EXTERNAL-SYMBOLS`（口径唯一、与调用时机无关）。
- `test/gen_probes*.py`：无参默认 `gen`（门禁真正生效）；`p0395` 由"数量常量"
  改为**不变量断言**（列表 ≡ 导出集、无重复、非空），新增导出不再变红；
  删除失效的 `_static_fun_count`。重新生成 `differential-probes-test2.lisp`
  （顺带把 9 行过期浮点期望值规范化）。
- `test/README-tools.md`：新增「codegen 门禁的两条硬约束」与「新增导出函数怎么操作」。
- `CONVENTIONS.md` §10.4.1：把 API 指纹规则固化（禁写死数量 / 口径唯一 /
  导出不重复 / 无参=gen / 生成可复现 / 重新生成后先 add）。
- `.gitignore`：忽略 `tmp/`（生成器中间产物）。

### 验证

- `differential-probes-test2`：**405/405**（原 403/405）
- `complex-test` 136/136、`core-test` 284/284、`differential-probes-test` 496/496
- `check_key_sync.py` 全绿且**真实生效**（人为制造漂移可被测出）
- 不变量探针非空转：列表污染 / 含重复 / 为空 / 晚调用 refresh 四种破坏各自返回 0

---

## 2026-10-19 — 第一轮进化：复数（complex64 / complex128）支持

### 新增 dtype（src/dtype.lisp）
- `:complex128` / `:complex64`（CL 元素类型 `(complex double-float)` / `(complex single-float)`，
  itemsize 16 / 8，interleaved 存储；strides 仍以**元素**为单位）
- 提升规则：**复数的实部分量参与普通实数提升**，结果映射回 complex64/complex128。
  该单一规则复现 numpy 2.4.6 的 **100 格**提升表（含 `complex64 ⊔ int32 → complex128`）
- 值级原语（唯一语义出口）：`%creal %cimag %cabs %carg %cconj %cisnan %cisinf
  %cisfinite %cis-zero %clt %cle %cgt %cge %cneq`（字典序比较 + NaN 规则）
- `vt-cast` / `vt-cast-fun`：complex→real/int **取实部**（对标 numpy astype 的
  ComplexWarning 语义，本库不产生警告）；real→complex 虚部补 0

### 构造与推断（src/core.lisp、src/io.lisp）
- `ensure-vt`：复数标量 → complex64/complex128（按分量宽度）；含复数的列表 → complex128
- `make-vt` / `vt-zeros` / `vt-ones` / `vt-const` 支持复数 dtype
- 打印：`1.0+2.0j` 形式（`%format-number` 新增 :complex 分类）

### 语义层（src/complex.lisp，新增，最后加载）
- 分派机制：加载时捕获各函数的**原实现**（`%cx-orig`，预捕获以避免自我递归），
  仅「输入含复数」的调用进复数内核，实数路径零改动
- 一元数学 21 个（sqrt/exp/log/log2/log10/三角/双曲/反三角/反双曲/square/
  reciprocal/signum/negative）+ 幂（复指数、`0**0=1`、`0**负=nan+nanj`）
- 分量族：`vt-abs`（→实数 dtype）、`vt-real` / `vt-imag`（副本）、`vt-conj` /
  `vt-conjugate`、`vt-angle`、`vt-complex`、`vt-iscomplexobj` / `vt-isrealobj`
- 比较：`== != < <= > >=` 全部**字典序**（numpy 2.x 语义），NaN 参与恒假
- 极值/排序：`maximum/minimum/fmax/fmin`、`sort/argsort`（字典序 + NaN 末尾）、`median`
- 判定：`isnan/isinf/isfinite`（任一分量为真即真）
- 归约：`sum/prod/mean`（保复数 dtype）、`var/std/norm`（**恒实数**）、
  `argmax/argmin`（int64）、`cumsum/cumprod`、`trace`、独立 stride 驱动内核
- 线代：`matmul/@/dot/outer` 专用内核（不复用 einsum —— 其 dtype 白名单只含 4 种实数类型）
- 未支持即报错：floor/ceil/trunc/rint、floor_divide/mod/rem、位运算、atan2、
  even/odd-p、percentile/quantile、einsum、LU/SVD/eig/inv/solve/cholesky/qr、
  神经网络激活（均给出确定性错误，绝不静默丢虚部）

### 文档
- `CONVENTIONS.md` 新增 **§16 复数（complex）约定**（约 380 行）：物理表示、
  100 格提升表与形式化规则、casting、分支切割（SBCL 与 numpy 逐点对照）、
  字典序比较、NaN/Inf、归约、线代、打印、错误契约、执行层定位、已知差异 D-C1..D-C6、
  第二轮路线图
- 记录 SBCL 平台陷阱：`&rest` + `(safety 0)` + call-site `apply` 三者同时成立会段错误

### 测试
- `test/complex-test.lisp`（**136 断言**）+ `test/gen_complex_expected.py`
  （numpy 2.4.6 实测生成 **83 项**差分期望，`test/complex-expected.sexp`）
- 已接入 `test/run-tests.sh`（SUITES + DEFAULT_ORDER，防漏接线）

### 回归
- core-test 284/284、run_all_tests 143/143、nan-broadcast-test 31/31、
  numpy-convention-tests 69/69、contract-regression-test 136/136、
  out-contract-v2-test 56/56 全部通过（实数路径未受影响）

---

## 2026-10-10 测试工具链防漂移 + 遗产用例盘活 + vt-narrow 负索引

### vt-narrow 语义修复（src/manip.lisp）
- `vt-narrow` 的 start/end 支持 **numpy 切片负索引语义**：负值先加轴长，
  规约后自动 clip 到 `[0, dim-size]`（此前负索引/越界直接报错）。
  规约后 `end < start` 仍报错。core-test 新增 52b/52c 两条断言。

### 测试工具链
- 删除旧架构孤儿生成器 `gen_all_tests.py` / `gen_param_tests.py`
  （静态 JSON 迁移遗留，产出物无任何消费者，且旧 key 风格易误导维护者）
- 新增 `test/check_key_sync.py` 防漂移门禁并挂入 `run-tests.sh` 前导：
  5 套件查询 key 与 ref_compute 供给机械核对（缺失→FAIL+报因，死供给→WARN），
  探针 codegen 与提交 .lisp 逐字节校验
- 遗产盘活：架构迁移时仅 280/560 例被接线，其余 281 例从未被任何套件消费。
  现全部恢复：oracle 并入 `ref_compute.py`（651 key，0 缺失 0 死供给），
  lisp 调用补入新套件 `test/legacy-coverage-test.lisp`（284/284 通过）
- 新增 `test/README-tools.md` 工具链地图（各 py/lisp 角色、消费方、修改清单）
- 盘活中确认的语义差异（已在 README-tools.md 登记）：
  `vt-det` 暂不支持 3D 批量；`vt-diagonal` 3D 为 pytorch 末两维风格；
  `vt-gradient` 1D 返回裸 VT

---

## 2026-10-06（四）— 新增 `vt-cross-entropy-logits`：真正的 torch.nn.CrossEntropyLoss 语义

### 问题

`vt-cross-entropy` 的文档字符串声称「对标 torch.nn.CrossEntropyLoss」，
但实现是**概率/one-hot 输入**语义（`-Σ y·log(p)`，且对 p 做
`[eps, 1-eps]` 裁剪）。两者并不一致：

- torch 的 `CrossEntropyLoss` 收 **raw logits + 整数类别索引**，
  内部做 log-softmax；
- 该函数收 **概率 + one-hot**，且 `vt-clip(p, eps, 1-eps)` 对 logits
  完全不适用（logits 可为负、可远大于 1）。

以「对标 torch.nn.CrossEntropyLoss」的写法调用会因形状不可广播而
报错，例如：

```lisp
(vt-cross-entropy (vt-asarray '((1.0d0 2.0d0 3.0d0) (2.0d0 0.0d0 1.0d0)))
                  (vt-asarray '(0 1) :dtype :int64))
;; => error: 形状 (2 3) 和 (2) 无法广播：维度 3 与 2 不兼容
```

### 修复

**新增 `vt-cross-entropy-logits`**（`src/nn.lisp`，已导出），真正对标
`torch.nn.CrossEntropyLoss`：

| 参数 | 说明 |
|---|---|
| `logits` | 未归一化得分，沿 `axis` 为类别维（缺省 -1） |
| `labels` | **整数**类别索引，形状 = logits 去掉类别轴（`(N,C)` 配 `(N,)`；支持任意秩 `(N1,N2,...,C)` 配 `(N1,N2,...)`；也支持无批次 `(C,)` 配标量/长度 1） |
| `axis` | 类别轴，缺省 -1 |
| `reduction` | `:mean`（缺省）/ `:sum` / `:none` |

关键设计：

- **数值稳定**：内部用 log-softmax（减该轴最大值 + log-sum-exp），
  大 logits 不溢出（`logits=(1e4,0,0)` 仍精确得 `0.0d0`）；
- **不做 `[eps, 1-eps]` 裁剪**——logits 语义下裁剪会破坏数值；
- **类别索引越界即报错**（含负数索引），非静默取值；
- 浮点 `labels` 明确报错（要求整型，对标 torch 的 int64 labels）；
- dtype 缺省按 logits 提升：float32 → float32，其余 → float64。

示例（与 `torch.nn.CrossEntropyLoss` 逐位一致）：

```lisp
(vt-cross-entropy-logits (vt-asarray '((1.0d0 2.0d0 3.0d0) (2.0d0 0.0d0 1.0d0)))
                         (vt-asarray '(0 1) :dtype :int64))
;; => 2.4076059644443806d0
```

**`vt-cross-entropy` 语义保持不变**（现有 5 处调用依赖其概率语义，
含 `example/example.lisp` 的 one-hot 用法），仅**修正文档字符串**，
明确它是概率输入并指向新函数，消除「对标」与实现不一致的误导。

### 测试

- `test/extensions3-test.lisp`：新增 18 条 `CE-LOGITS` 断言
  （用户原文调用、reduction 三态、任意秩、axis、数值稳定、
  与 softmax+log 对照、dtype、`:out`、5 条错误路径、
  旧接口未受影响），套件 243 → **261** 条全绿。
- `example/example.lisp`：`test-activation-loss` 补 `vt-cross-entropy-logits` 用例。
- 全量 **31/31 套件通过，0 失败**。

### 约定文档同步（避免后人重踩）

- `CONVENTIONS.md` **新增 §3.4「神经网络的浮点语义」**：明确 nn 激活/损失函数
  **不得**走通用提升规则（整数进整数出会静默截断），统一规则为
  float32→float32、其余（含整数）→float64，并记录根因与实现要求
  （必须经 `%nn-float-dtype`/`%nn-as-float`，不得依赖 `vt-map` 的 `dtype=nil`）。
- `CONVENTIONS.md` **新增 §3.5「两种交叉熵语义（不可混用）」**：并列
  `vt-cross-entropy`（概率/one-hot）与 `vt-cross-entropy-logits`（raw logits，
  对标 torch），标注易错点与 `vt-cross-entropy-logits` 的完整契约。
- `CONVENTIONS.md` §9.2 核对清单补「激活函数族 / `vt-cross-entropy` /
  `vt-cross-entropy-logits`」三行。
- `README.md` 损失函数清单补 `vt-cross-entropy-logits` 并注明两者语义差异。

---
## 2026-10-06（三）— 第二轮审查核实：修复 nn.lisp 整数提升链（softplus/gelu/leaky-relu）+ 澄清 R-1/R-2

对第二轮审查报告逐条在最新 `master`（`471d005`）上实测复核。

### 真实缺陷（已修复）

| # | 位置 | 缺陷 | 修法 |
|---|------|------|------|
| R-3 | `vt-softplus` | 整数输入返回**整数** dtype 且数值被静默截断。`softplus([0])` → `0`（应 `0.693…`）；`softplus([-1])` → `0`（应 `0.313…`）；`softplus([-5,0,5,25])` → `(0 0 5 25)`（应 `(0.007 0.693 5.007 25.0)`）；配 `:out float64` 直接报 dtype 不匹配 | `%nn-float-dtype` 推导浮点结果 dtype（float32→float32，其余含整数→float64），`%nn-as-float` 先把输入提升再映射 |
| R-3b | `vt-gelu` | **报告未提及**。整数输入同样返回整数 dtype 且数值截断：`gelu([1,2,3])` → `(0 1 2)`（应 `(0.841 1.955 2.996)`） | 同上 |
| R-3c | `vt-leaky-relu` | **报告未提及**。整数输入负值被截断：`leaky-relu([-2,3,-4])` → `(0 3 0)`（应 `(-0.02 3.0 -0.04)`） | 同上 |

> 三个函数同属一条「整数提升缺失链」：通用路径把 lambda 直接交给
> `vt-map` 且 `dtype=nil`，`vt-map` 走 `vt-promote-type` → 整数进整数出，
> lambda 内算出的浮点值被写回整数存储而静默截断。
> 这是 P1-1（elementwise 数学函数）在 `nn.lisp` 的**同类漏网**。

已排查并**确认无此问题**的同族函数（补断言守护）：
`vt-sigmoid` / `vt-swish` / `vt-mish` / `vt-hard-tanh` / `vt-hard-sigmoid` /
`vt-binary-cross-entropy`（均正确返回 float64）。

### 经核实为「误报」或「已修复」（未改动）

| 报告条目 | 报告结论 | 实际核实 |
|---|---|---|
| P0-1 `vt-relu` 整数段错误 | 仍存在，段错误崩溃 | **已修复**（`471d005`）：实测 `(vt-relu (vt-asarray '(1 2 3 4)))` → `(:FLOAT64 (1.0 2.0 3.0 4.0))`，int8/int32/uint8/负值/float32 全部正常 |
| P1-1 整数数学函数 float32 精度 | 15 项失败 | **已修复**（`471d005`）：`sin int64 → 0.8414709848078965d0`，与 numpy 逐位一致 |
| P1-2 `vt-expm1`/`vt-log1p` 不提升 | 返回 int64 | **已修复**（`471d005`）：两者整数输入均返回 `:FLOAT64` |
| **R-1** 非连续 `:out` 写入不落盘 | 维持原判，称「上游测试只验证通过 out 读回，没验证通过 base 读回」 | **误报，且报告对上游测试的描述与事实相反**。实测：`(vt-add m23 m23 :out view)` 后 **`base` 正确落盘**为 `((0 0 2 0 4 0) (6 0 8 0 10 0))`。`test/out-contig-tests.lisp` 中 `"2b base 偶数位被写入"`、`"2c base 奇数位保持 0"`、`"3c 写入位置正确"`、`"3d 未写位置保持 0"` 等断言**全部作用在 `base` 上**，正是报告声称「缺失」的那类验证 |
| **R-2** `vt-cross-entropy` 不支持分类调用 | P1 新发现，且「传 3 元素 labels 静默给出错误数值 56.41（正确 4.815）」 | **非缺陷，属设计取向**。该函数文档与测试均明确为**概率输入**（`test/test-all.lisp`：`"vt-cross-entropy 同 bce（概率输入）"`），与 `vt-softmax` 配套使用。报告要求的 raw-logits 语义属**功能诉求**（可考虑新增 `vt-cross-entropy-logits`），非「实现错误」。另：报告给出的数值 `4.815`/`56.41` 用 one-hot 语义（`1.4076`）与广播语义（`0.4076`）**均无法复现**，数值来源不可信 |

### 测试

- `test/extensions3-test.lisp`：新增 21 条「R-3 整数提升回归」断言
  （softplus dtype×3 + 数值×5 + `:out`×1；gelu dtype×3 + 数值×1；
  leaky-relu dtype×1 + 负值不截断×1 + float32 保持×1；同族守护×5），
  套件由 222 → **243** 条全绿。
- 全量 **31/31 套件通过，0 失败**（含 `fastmap-invariant-test`）。

---
## 2026-10-06（二）— 修复 P1-1 精度修复引入的性能回归：`%float-map` 宏化，恢复 `vt-fast-map` 内联快路径

上一轮为修复 P1-1（整数输入仅 float32 精度）引入了函数 `%float-map-fn`，
它无条件把输入交给 `vt-map`。**这是一个性能回归**：`vt-fast-map` 是宏，
编译期把字面算子（`#'sin`）内联成特化循环；而 `vt-map` 经 `funcall` 调用，
单元素迭代器慢约 **1.6–1.8×**。结果是连**本就无需提升的连续 float64/float32
输入**也被拖到慢路径上。

实测（100k 元素 × 1000 次，SBCL 2.6.8）：

| 调用 | 回归态（`%float-map-fn`） | 修复后（`%float-map` 宏） | 原始基线 |
|---|---:|---:|---:|
| `vt-sin` float64 | 2804 ms | **1182 ms** | 1206 ms |
| `vt-exp` float64 | 1216 ms | **623 ms** | 614 ms |
| `vt-sin` int64 | 2514 ms | **1287 ms** | — |
| `vt-fast-map #'sin` 裸调用参考 | — | 1205 ms | — |
| `vt-map #'sin` 裸调用参考 | — | 2207 ms | — |

### 修法

把 `%float-map-fn` 拆分/替换为两个宏，**dtype 判定外提为运行时分枝，
但两条分支都在编译期展开为快路径**：

| 新入口 | 适用算子 | 展开后的快路径 |
|---|---|---|
| `%float-map` | `(function <symbol>)` 字面算子（`#'sin`/`#'cos`/`#'asinh`/`#'exp`/`#'%e3-expm1`/`#'%e3-log1p` 等） | **`vt-fast-map` 内联循环**（已是目标浮点 → 零拷贝直通；整数 → 先 astype 再内联） |
| `%float-map-lambda` | lambda 算子（`vt-asin`/`vt-acos`/`vt-acosh`/`vt-atanh`/`vt-sqrt` 需逐元素判 NaN） | `vt-map`（`vt-fast-map` 无法内联 lambda）+ 同样的 dtype 提升分支 |
| `%float-map-fn`（保留为兜底） | 任意运行时 FN | `vt-map` |

两宏展开形如：

```lisp
(let* ((%in (ensure-vt vt)) (%dt dt))
  (if (eq (vt-dtype %in) %dt)        ; 已浮点：直通，不做多余 astype
      (vt-fast-map #'op %in :out out :dtype %dt)
      (vt-fast-map #'op (%coerce-float-input %in %dt) :out out :dtype %dt)))
```

改动点：

- `src/elementwise.lisp`：新增 `%float-map`（宏，带 `(function <symbol>)`
  入参校验）、`%float-map-lambda`（宏）；`%float-map-fn` 降级为文档化的
  兜底入口。9 个字面算子调用点（sin/cos/tan/atan/sinh/cosh/tanh/asinh/exp）
  改用 `%float-map`，5 个 lambda 调用点（asin/acos/acosh/atanh/sqrt）改用
  `%float-map-lambda`。
- `src/extensions3.lisp`：`vt-expm1` / `vt-log1p` 由手写
  `vt-map + 预 astype` 改为 `%float-map #'%e3-expm1 / #'%e3-log1p`，
  同样拿到内联快路径，并消除「调用方重复 astype」的多余拷贝。

### 正确性

P1-1 精度契约完好：`(vt-sin (vt-asarray '(1 2 3 4)))` 第 0 元素
`0.8414709848078965d0`（float64 全精度）；`vt-expm1`/`vt-log1p`
整数输入仍返回 `:float64`。全量测试 **30/30 套件通过，0 失败**。

---
## 2026-10-06（一）— 第三方审查报告核实：修复 6 类真实缺陷 + 澄清 4 项报告误报

对 `clvt-review-report.md`（对照 numpy 2.5.3 / SBCL 2.6.8）逐条复核。
报告称 2 个 P0、2 类 P1、一批 P2；经在当前 `master` 上以 numpy 2.4.6
实测复现，**确认其中 6 类为真实缺陷并修复，另有 4 项经核实为报告误报
或已在 v0.4.0 修复**。

### 真实缺陷（已修复）

| # | 位置 | 缺陷 | 修法 |
|---|------|------|------|
| P0-1 | `vt-relu` | **段错误**：整数输入走通用路径，把 int64 元素喂给声明 `(double-float x)(safety 0)` 的 `%relu-double`，fixnum 被当 boxed double 解引用非法地址，SBCL 镜像崩溃 | 通用路径改为无类型假设的 `lambda` + 先 `vt-astype` 提升到目标浮点 dtype |
| P1-1 | `vt-sin/cos/tan/asin/acos/atan/sinh/cosh/tanh/asinh/acosh/atanh/exp/log/sqrt` | 整数输入数值仅 **float32 精度**：CL 对整数参数 `(sin 1)` 只返回 single-float（0.84147096），却被写入标称 float64 的结果 | 新增 `%coerce-float-input` / `%float-map-fn`，映射前把整数输入提升到目标浮点 dtype（float32 输入仍保持 float32） |
| P1-2 | `vt-expm1` / `vt-log1p` | 整数输入**不提升**，返回被截断的 int64（`expm1([1,2,3])` → `(1 6 19)`） | `%e3-float-prefer-dtype` 缺省不再返回 NIL，改为按输入推导（float32→float32，其余→float64） |
| P2-1 | `vt-argmax/argmin/nanargmax/nanargmin` | 返回 `:int32`，numpy 返回 `intp`(int64) | `%op-out-dtype` 改 `:int64` |
| P2-2 | `vt-logical-and/or/not/xor` | 默认返回 `:float64`，违反 CONVENTIONS §3.1「比较/逻辑返回 `:int8`」 | 缺省 dtype 改 `:int8`，返回 1/0 |
| P2-3 | `vt-cov` | `ddof` 语义与 numpy 不符：numpy 的 `ddof` **缺省为 1**（`np.cov(a)` 除以 N-1，`np.cov(a, ddof=0)` 除以 N），原实现恒用 N-1-ddof | 用 `ddof-p` 哨兵区分「未传」与「显式 0」，分母统一为 `N - ddof`（缺省 ddof=1） |
| P2-4 | `vt-histogram` | counts 返回 `:float64`，numpy 返回 int64 | counts 改 `:int64`（density=t 仍为 float64） |
| P2-6 | `vt-pow` | 整数基 + 正整数指数返回 float64（numpy 返回 int64）；传张量指数**静默返回 NaN** | 整数基 + 正整数指数保持整数 dtype；显式拒绝张量指数并给出清晰错误 |

> P0-1 是本轮最严重缺陷：非崩溃代码路径在正常输入（int 张量）下即触发
> 内存越界并杀死进程。已在独立子进程复现（`Memory fault ... in
> CLVT::%RELU-DOUBLE`），修复后 int/int8/int32/uint8 全部正常。

### 经核实为「报告误报」或「上游已修复」（未改动）

| 报告条目 | 报告结论 | 实际核实 |
|---|---|---|
| P0-2 `:out` 非连续视图写入丢失 | 称 base 仍全 0 | **误报**：当前 master 上 `vt-add(m,m,:out=v)` 正确落到底层 base，与 numpy 逐位一致（已有 `out-contig-tests` 23 条守护） |
| P2#5 `vt-trace` 整数输入返回 float64 | 称应返回 int64 | **误报**：实测 int64→int64、float32→float32、int8→int64，与 numpy 完全一致 |
| P2#8 `vt-eigvalsh` 未升序 | 称应升序 | **上游已修复**（v0.4.0）：现 `vt-eigvalsh` 升序、`vt-eigvals` 降序，均已符合各自 numpy 约定 |
| P2#3 描述「除以 n-ddof-1」 | 描述 | 描述本身正确，但报告未指出 numpy 的 **`ddof` 缺省值为 1** 这一关键点；已按 numpy 真实语义修复（见上表 P2-3） |

### 未处理（API 缺口，非缺陷）

报告 P2 表中另有约 12 项属**功能缺口或有意设计**，非缺陷，本次不改：
`vt-percentile` 不支持数组分位数、`vt-det` 不支持批量、`vt-bincount` 无
`:weights`、`vt-norm` 无 `ord`、`vt-sort` 无降序开关、`vt-array-equal`/
`vt-allclose` 无 `equal-nan`、`vt-repeat` 不支持逐元素 repeats、`vt-delete`/
`vt-ravel-multi-index`/`vt-random-choice :p` 不接受 VT（要求 list）、
`vt-lstsq :rcond nil`、布尔掩码写入 API 缺失、`vt-layer-norm` 为 torch 风格。
这些是"尚未实现"而非"实现错误"，建议列入后续路线图而非缺陷修复。

### 测试

- `test/extensions3-test.lisp`：新增 38 条「审查修复回归」断言
  （P0-1×6、P1-1×8、P1-2×4、P2-1×5、P2-2×5、P2-3×5、P2-6×5），
  套件由 184 → **222** 条全绿。
- `test/out-contig-tests.lisp`：argmax 的 `:out` 由 int32 改 int64（随 P2-1 契约更新）。
- `test/coverage-gap-test.lisp`：histogram counts 期望值改整数（随 P2-4）。
- 全量 **30/30 套件通过，0 失败**；`example/example.lisp` 打印 `all test passed`。

---
## 2026-10-05 — 覆盖缺口审计：补齐未覆盖函数测试 + 修复 6 处实现缺陷

在 SBCL 2.6.8 + Quicklisp + numpy 2.4.6 环境上，以 CONVENTIONS.md 为语义
契约运行全部 27 个测试套件与 example/example.lisp（均全绿），随后做
**导出函数覆盖审计**：对比 `src/package.lisp` 的 306 个 `vt-*` 导出与
`test/*.lisp` + `example/example.lisp` 的全部引用，发现 61 个公开函数
从未被任何测试或示例调用。

据此新增 `test/uncovered-coverage-test.lisp`（136 条断言，期望值取自
numpy 2.4.6 实测或 CONVENTIONS 契约），并为其中暴露的缺陷做修复。

### 修复的实现缺陷（6 处）

| # | 函数 | 缺陷 | 修法 |
|---|------|------|------|
| 1 | `vt-identity` | **崩溃**：`(vt-identity 3)` 缺省 `dtype` 为 NIL，下传 `vt-eye` 触发 `NIL fell through ECASE` | `&key (dtype :float64)` 对齐 `vt-eye` 缺省值 |
| 2 | `vt-vander` | **崩溃**：`(vt-vander x)` 缺省 `n` 为 NIL → `odd number of &KEY arguments` | `&key (n nil)`，缺省取 `len(x)`（对标 `np.vander` N=len(x)） |
| 3 | `vt-atanh` | `\|x\|=1` 返回 NaN，与 numpy 不符 | `x=+1→+Inf`、`x=-1→-Inf`、`\|x\|>1→NaN`（补 `vt-get-pos-inf`/`vt-get-neg-inf`） |
| 4 | `vt-= vt-/= vt-< vt-<= vt-> vt->=` | 返回 `float64`，违反 CONVENTIONS §3.1「比较运算返回 :int8 承载布尔」 | 缺省 dtype 改 `:int8`，返回值 1/0（numpy 语义不变，dtype 对齐） |
| 5 | `vt-positive-p vt-negative-p vt-zero-p vt-nonzero-p vt-even-p vt-odd-p` | 同上，返回 `float64` | 缺省 dtype 改 `:int8` |
| 6 | `vt-isnan vt-isinf vt-isfinite` | 返回 `float64` 布尔 | dtype 固定 `:int8` |

> 缺陷 4–6 是**契约违反**：CONVENTIONS §3.1/§9.2 明确"比较/逻辑运算返回
> `:int8`，取值 0/1"，而实现长期返回 `float64`。属"文档正确、实现偏差"，
> 按 CONVENTIONS「与 NumPy 冲突以 NumPy 为准、与文档冲突修实现」处理。
> `vt-all`/`vt-any`/`vt-count-nonzero` 返回整数计数（`int64`），非布尔，
> 保持不动；`vt-def-vt-reduce` 的 `:all`/`:any` 分支亦不受影响。

### 同步更新的既有测试（期望值随契约修正）

- `test/comprehensive-test.lisp`：`a<b` / `a==b` 期望由 `(1.0 1.0 0.0 ...)`
  改为 `(1 1 0 ...)`（int8）。
- `test/nan-random-test.lisp`：`isnan/isinf/isfinite` 期望改为整数 0/1。

### 新增测试套件

- `test/uncovered-coverage-test.lisp`（136 断言，已注册进 `run-tests.sh`）
  覆盖：二元算术别名、反三角/反双曲、谓词族、逻辑/位运算、创建族
  （含 identity 回归）、拼接/维度、clamp/select/copy-to!、vander 回归、
  结构原语（strides/normalize-axis/out 可写性/strides 驱动写入）、
  扩展函数、extensions2、随机数底层对象、最小二乘。

**验收**：`bash test/run-tests.sh --quick` → 28 套件全绿；
`sbcl --load example/example.lisp` + `run-all-tests` → 全绿。

## 2026-10-05 — extensions3：补充第三批 NumPy 重要缺失函数（约 50 个）

新增 `src/extensions3.lisp`（约 1100 行），按 CONVENTIONS.md 的语义契约
（`&key dtype out`、`ensure-vt` 归一化输入、比较/逻辑返回 `:int8`、
NaN/Inf 对齐 numpy 2.4.6、`:out` 硬契约 H1–H6）一次性补齐 NumPy 高频函数，
并配套新增 `test/extensions3-test.lisp`（184 条断言，全部通过）。

### 新增函数（按类分组）

| 类别 | 函数 |
|------|------|
| 创建 | `vt-asarray` `vt-fromiter` `vt-tri` `vt-diagflat` `vt-trim-zeros` |
| 形状 | `vt-rollaxis` `vt-column-stack` `vt-block` `vt-broadcast-arrays` `vt-resize` |
| 索引 | `vt-take-along-axis` `vt-put-along-axis` `vt-compress` `vt-indices` `vt-fill-diagonal` |
| 数学 | `vt-absolute` `vt-sign` `vt-positive` `vt-expm1` `vt-log1p` `vt-logaddexp` `vt-float-power` `vt-copysign` `vt-signbit` `vt-nextafter` `vt-spacing` `vt-gcd` `vt-lcm` `vt-divmod` `vt-nan-to-num` `vt-real` `vt-imag` `vt-conj` `vt-angle` |
| 统计 | `vt-nancumsum` `vt-nancumprod` `vt-nanpercentile` `vt-nanquantile` `vt-cov` `vt-corrcoef` `vt-cross` |
| 线代 | `vt-vdot` `vt-eigvals` `vt-eigvalsh` `vt-matrix-power` `vt-cond` `vt-multi-dot` |
| 逻辑 | `vt-array-equal` `vt-array-equiv` `vt-isposinf` `vt-isneginf` |
| 集合 | `vt-isin` |

### 关键实现要点（对齐 NumPy 语义）

- **`vt-take-along-axis` / `vt-put-along-axis`**：非 axis 维按 NumPy
  **广播规则**处理（长度相等或为 1），而非要求严格相等；输出形状中 axis 维
  取自 `indices`、其余维取自输入张量。原实现误用严格相等 + 线性下标，
  已改为基于 `strides` 的坐标寻址，并修正 `vt-ravel` 后 1 维视图的混用。
- **`vt-spacing`**：`numpy.spacing(x) = nextafter(|x|, +inf) - |x|`；
  原实现误取「向 `most-positive-double-float` 的 nextafter 值」。
- **`vt-cross`**：结果形状对齐 NumPy——1 维输入去掉 batch 轴（2 分量 → 0 维
  标量）；2 维批量输入保留 batch 轴（2 分量 → 去掉分量轴）。修正了
  `%e3-as2d` 对 `axis=-1` 误转置的缺陷。
- **`vt-nancumsum` / `vt-nancumprod`**：`axis` 分支的输出坐标需以 axis 位置的
  累积下标**替换**该维坐标，且循环结束后返回 `result`（原实现返回 `nil`）。
- **`vt-sign`**：NaN 分支须包 `with-float-safe`（`%nan-p` 的 `x /= x`
  否则触发 `FLOATING-POINT-INVALID-OPERATION`）。
- **`vt-expm1` / `vt-log1p`**：x≈0 时用 Taylor 级数（`%e3-expm1` /
  `%e3-log1p`）保证与 numpy 一致的精度（`1e-10` 量级逐位对齐）。
- **`vt-signbit`**：用 `(minusp (float-sign xf))` 正确识别 `-0.0`。

### 测试

- `test/extensions3-test.lisp`：184 条断言覆盖全部新函数，含
  `:out` 硬契约（形状/dtype 精确匹配、可写）、非连续输入、dtype 提升、
  NaN/Inf 语义与错误路径。
- 测试辅助 `->list`/`approx` 增加 NaN 安全处理（`with-float-traps-masked`），
  避免浮点陷阱在断言阶段误报。

---

## 2026-10-05 — v0.3.6 numpy 2.1.3 语义对齐：空归约重校准 / :out 精度解耦 / true_divide / random 校验

以 numpy 2.1.3 实测为基准（逐项探针对表），修正 v0.3.5 分歧表中三处
"设计约定"与 numpy 的偏差，并收敛 TEST-PLAN §6 两项契约缺口。
新增回归套件 test/numpy-convention-tests.lisp（46 条断言），全部通过。

### 方向 1：空归约语义重校准（reduce-stats.lisp，def-vt-reduce 统一生成）

| 场景 | v0.3.5（NaN 填充约定） | v0.3.6（numpy 对齐） |
|------|------------------------|----------------------|
| 输出为空（`(3 0)` axis=0 等） | NaN 填充 / arg 报错 | **所有算子返回空结果**（含 argmax/argmin） |
| max/min 族输出非空且归约区空 | NaN 填充（整数提升 float64） | **ValueError**（zero-size array to reduction operation ... which has no identity） |
| arg 族输出非空且归约区空 | 报错 | **ValueError**（attempt to get argmax of an empty sequence，消息对齐 numpy） |
| 单位元族（sum/prod/all/any/nan*） | 填单位元 | 不变 |
| mean 空归约 | NaN | NaN（numpy 同款，保留） |

### 方向 2：`:out` 精度解耦（TEST-PLAN §6 #2 收敛）

旧行为：`:out` dtype 决定**计算精度**（float64 输入 + int32 out 按 int32
累加，静默错值）。新行为：**按输入提升计算**（compute-dtype，与无 :out
调用逐位一致），最后一步 cast 写入 `:out`（decoupled 预路径）。

- `def-vt-reduce`（sum/prod/amax/amin/all/any/nan*/arg* 共 14 函数）：
  三条内核路径统一——decoupled 时结果先写入 compute-dtype 临时缓冲，
  尾部 `vt-copy-into` 跨 dtype 写入 out
- `vt-mean`/`vt-var`/`vt-std`/`vt-nanmean`/`vt-nanvar`/`vt-nanstd`：
  compute-dtype / exec-dtype / write-out 三段式解耦（显式 `:dtype` 仍按
  numpy 语义直接以该 dtype 计算）
- `:out` 与 `:dtype` 冲突显式报错；`:out` 广播视图（dim>1 且 stride=0）
  拒绝写入（既有契约保留）

### 方向 3：vt-mod/vt-rem 零除 dtype 语义（elementwise.lisp）

旧行为：零除一律返回 0（库约定）。新行为：**按除数 dtype 语义**——
浮点语境（除数为浮点标量或张量）→ NaN（numpy mod/fmod 同款），
整型语境 → 0。混合输入按被除数语境判定。

### 方向 4：vt-/ true_divide 语义（elementwise.lisp）

旧行为：整数除零报 CL division-by-zero；整数/整数返回整型截断。
新行为（numpy true_divide 同款）：**整数输入先提升 float64 再除**，
零除按 IEEE 得 ±Inf/NaN（`(vt-/ 5 0)` → +Inf、`(vt-/ 0 0)` → NaN），
`(vt-/ 5 2)` → 2.5d0（float64）；显式 `:dtype`/`:out` 浮点目标一致处理；
标量除数不再触发类型崩溃。

### 方向 5：random NaN/Inf 参数校验前移（random.lisp，TEST-PLAN §6 #4 收敛）

`vt-random-uniform`/`vt-random-normal`：`numberp` 检查在最外层，
NaN/Inf 判定（`%nan-p`/`%inf-p`）前移进 `with-float-safe`（屏蔽
FP-INVALID-OPERATION）——保证报出干净的参数错误而非 FP 异常。
本轮补齐 `vt-random-normal` 缺口：mean 有限性校验（此前完全缺失）、
std 非负**且有限**校验（此前 `>= 0` 放过 +Inf）。

### 测试

- 新增 test/numpy-convention-tests.lisp：46 条断言（空归约三分类 ×17、
  `:out` 精度解耦 ×8、mod/rem 零除 ×6、true_divide ×6、random 校验 ×9），
  已注册 run-tests.sh
- 更新 shape-degenerate-tests.lisp（amax/argmax 空 (3 0) 断言改 numpy 语义、
  softmax (0) 改期望报错——scipy.special.softmax 实测对空数组抛同款
  ValueError）、error-contract-tests.lisp（整数除零改 IEEE 断言：0-d 输入
  + 测试侧 FP 陷阱屏蔽的 NaN 检测）、refactor-bugfix-tests.lisp（mod 零除
  两断言改 NaN 期望）、test-ai-edge-cases.lisp（BUG-2b softmax 空向量改
  期望报错）
- 修复 def-vt-reduce 宏体两处括号错位（patch 期间引入：空归约块提前闭合
  主 let* 导致 14 个生成函数缺失/旧代码残留，表现为运行时 IN-ET unbound）
- 清理 vt-var 冗余 write-out 绑定（emit 统一出口后遗留）

---

## 2026-10-04 — v0.3.5 三层分离复审：非有限值传播 / numpy 语义修复 / 内存安全校验

在 SBCL 2.6.8 + Quicklisp 环境下按三层分离约定逐文件复审全部源码，
以运行时探针逐项对表 numpy 实测语义，确认并修复 8 类 bug。
新增回归套件 test/refactor-bugfix-tests.lisp（53 条断言），全部通过。

### 方向 1：非有限值（NaN/±Inf）传播（执行层泄漏底层陷阱 → 逻辑层显式处理）

`vt-floor`/`vt-ceiling`/`vt-truncate`/`vt-round`/`vt-rint`/`vt-mod`/`vt-rem`
对 NaN/±Inf 输入直接泄漏 `FLOATING-POINT-INVALID-OPERATION`（SBCL 对非有限值
调用 floor/round 会 signal，浮点陷阱屏蔽无法避免），而 numpy 语义为传播：
floor(nan)=nan、floor(±inf)=±inf、mod 含 NaN → NaN、mod(±Inf, 有限) → NaN、
mod(有限, ±Inf) → 被除数本身。
修复（elementwise.lisp）：`%floor-family-body` 宏统一在进入 CL 取整前拦截
非有限值并返回原值；mod/rem 增加同一约定（除 0 返 0 的库约定保持不变）。

### 方向 2：vt-insert 对齐 numpy.insert（逻辑层语义修复）

旧实现降序插入且负索引按"演化中尺寸"解析，与 numpy 相比有两处偏差：
重复索引处值逆序（`insert([0,1,2],[1,1],[10,20])` 旧 `[0,20,10,1,2]`
→ numpy `[0,10,20,1,2]`）；混合正负索引错位。
修复（join.lisp）：flat/axis 两模式统一为 numpy 算法——
先按原尺寸归一化负索引并越界检查，再升序稳定排序（值跟随索引、
同位置按 values 顺序），插入位置 = 归一化位置 + 已插入个数。

### 方向 3：内存安全与输入校验（L1/L2 底线）

| 问题 | 修复 | 文件 |
|------|------|------|
| `vt-solve` 不校验 b 行数：b 行数 < n 时消元循环按 n 行读写底层缓冲区（实测 SBCL 报 "Invalid index 2 for (SIMPLE-ARRAY DOUBLE-FLOAT (2))"，越界尝试） | 逻辑层显式校验 b 行数 = 系数矩阵阶数，违反报可定位错误 | linalg.lisp |
| `vt-tensordot` 负轴不归一化：被当自由轴处理，产生错误结果或 "%vt-einsum-string" 内部错误 | 负轴按各自秩归一化 + 越界显式报错；整数 axes 超出可收缩范围报错 | extensions.lisp |
| `vt-inner` 0 维输入报 subseq 内部错误 | 0 维输入（任一侧）走广播逐元素乘（对标 numpy.inner） | extensions.lisp |
| `vt-mod`/`vt-rem` 仅接受标量除数，张量除数报类型错误 | 除数可为张量，广播逐元素取模（语义同标量路径） | elementwise.lisp |

### 方向 4：dtype 单一事实来源补全（逻辑层）

| 问题 | 修复 | 文件 |
|------|------|------|
| `vt-from-array` 对 (signed-byte 16/8)、(unsigned-byte 8/16) CL 数组一律静默推断 :int32（subtypep 宽类型命中） | 按窄类型优先精确推断 :int8/:int16/:uint8/:uint16/:int32/:int64/:float32/:float64，显式 :dtype 仍可覆盖 | creation.lisp |
| `vt-cast-fun :uint16` 走 `%wrap-uint16`（v0.3.1 统一政策的遗漏项） | 统一为 `%coerce-uint16` | dtype.lisp |
| package.lisp 5 个 core.lisp 辅助函数重复导出 3 处 | 归并到「张量结构访问器」区唯一导出 | package.lisp |
| `vt-rem` docstring 误标 numpy.remainder（CL:rem 为截断除法余数 = numpy.fmod） | 更正为 numpy.fmod | elementwise.lisp |

### 测试

- 新增 test/refactor-bugfix-tests.lisp：53 条断言覆盖上述全部修复
  （floor 族 NaN/Inf ×6、mod/rem 张量除数与非有限值 ×6、insert numpy 语义
  ×8、solve 校验 ×3、inner 0 维 ×3、tensordot 负轴 ×5、from-array 推断
  ×7、cast-fun ×2 及相关回归），已注册 run-tests.sh
- numpy 实测对照：np.insert / np.mod / np.inner / np.tensordot 逐例验证
- 全量回归 `bash test/run-tests.sh`：24/24 套件全部通过（含既有
  shape-degenerate 70 / property-strided 67 / error-contract 19 等套件）

---

## 2026-10-03 — v0.3.4 测试规划落地 / vt-copy-into 零尺寸写入修复

按 TEST-PLAN.md（新增，test/TEST-PLAN.md）建立"类别矩阵 > 函数覆盖 > 断言数量"
的覆盖策略，并以探针实测（tmp/probe-edge.lisp 方法）逐项对表 numpy 语义。

### 方向 1：vt-copy-into 零尺寸写入早退（bug 回归：stack axis=1 堆叠空 1D）

`(vt-stack 1 (vt-zeros '(0)) (vt-zeros '(0)))` 泄漏 CL 底层序列错误
"bounding indices 1 and 1 are bad for a sequence of length 0"：
vt-concatenate 向 (0 2) 结果的 [:, 1:2] 空视图写入时 offset=1、底层
存储长度=0，快路径按 (offset+size) 计算平坦边界触发 replace 越界。
修复（core.lisp）：vt-copy-into 在形状兼容性检查后、快路径前增加
零尺寸早退（v0.3.3 空 strided 问题的写入侧）。零元素写入是 no-op，
但非法形状仍照常报错——只跳过平坦边界计算，不放宽契约。

### 方向 2：三个新套件（156 条断言，全部通过）

| 套件 | 类目 | 断言 | 内容 |
|------|------|------|------|
| test/shape-degenerate-tests.lisp | C1/C2 | 70 | 全 op 家族 × 空形状/退化形状表驱动：创建、元素级、归约（sum/prod 单位元、mean/amax NaN 约定、argmax 报错约定）、matmul/dot/outer/trace、变形、索引、nn、集合运算；含 bug-stack-empty-ax1 回归 |
| test/property-strided-tests.lisp | C8 | 67 | strided≡contiguous 选路不变量（8 随机形状 × 元素级/归约/转置/argmax/matmul，固定种子确定性）；flip/transpose/reshape 往返、concat 可逆、sum 拆解、sort≡take(argsort) 等 12 项代数恒等式 |
| test/error-contract-tests.lisp | C7 | 19 | 形状/秩/内维/perm/越界/切片步长/one-hot/einsum/uniform 参数域/整除 0/广播 out 只读等契约错误逐项断言 |

三个套件均已登记 test/run-tests.sh。回归确认无副作用：
test-copy-into 35、out-contig-tests 23、test-overlap 6、
test-ai-edge-cases 40、memsafety-tests 6 全部通过。

### 已知分歧与缺口（TEST-PLAN.md §5/§6 记录，不阻塞）

- mean/median/amax/amin 空归约填 NaN、argmax 空输入报错：文档化设计约定，
  与 numpy（ValueError/空输出）不同；
- reduce 族 `:out` dtype 决定计算精度（float64 输入 + int32 out 按
  int32 累加）：契约缺口，需专项决策（对齐 numpy 提升语义或文档化）；
- vt-random-uniform NaN 参数报 FP 异常而非参数校验错误：L2 达标，
  错误类型欠佳。

---

## 2026-10-02 — v0.3.2 首次 SBCL 实测回归 / reduce 小整型内核补全

v0.3.0/v0.3.1 两轮重构均为静态验证（无 SBCL 环境）。本轮在 SBCL 2.6.8 +
Quicklisp + sb-simd 真实环境下首次全量加载与回归，修复实测暴露的问题，
并补全三层分离在执行层的最后一处 dtype 契约缺口。

### 方向 1：reduce 家族小整型 dtype 补全（执行层选路规则落地）

`def-vt-reduce` 生成的类型特化内核此前只覆盖 4 种存储级元素类型
（double/single/int64/int32），导致 `vt-sum`/`vt-amax` 等对
int16/int8/uint8/uint16 张量直接抛 "unsupported input/output dtype"——
违背 dtype.lisp 声明的 8 种逻辑 dtype 单一事实来源（test-bug0 用例 5 暴露）。

修复（reduce-stats.lisp）：

| 项目 | 设计 |
|------|------|
| 特化内核范围 | 保持 4 种存储级类型不变（对应 `*vt-storage-dtypes*`），避免内核全展开（8×8 组合实测在 1GB 堆下编译器耗尽） |
| 小整型路径 | 新增 `%kernel-small-general`：每算子仅展开一份的通用内核（路径 3 结构），读取经通用 aref，sum/prod 按 numpy 语义提升 int64 累加，max/min 以输入 dtype 极值初始化，写出经 vt-cast 按结果 dtype 转换 |
| 选路守卫 | 路径 1（连续+全局）/路径 2（连续+单轴）增加存储级 dtype 守卫，小整型统一由路径 3 兜底——连续性/特化只影响性能，不影响正确性 |
| dtype 映射 | `%dtype->lt`/`%et->lt`/`%lt-rank`/`%cast-form` 补全 8 种类型；`%op-acc-lt` 小整型 sum/prod 提升 int64；`%op-init` 补全小整型 max/min 极值初始化 |
| 空归约 | 单位元填充的编译期表覆盖全部 8 种输入类型（此前 prod 空小整型会错误返回 0） |

语义对齐 numpy：`sum(int8)→int64`、`amax(int8)→int8`（保 dtype）、
`amax(uint16)→65535` 无回绕。

### 方向 2：随机数静默陷阱显式化

| 问题 | 修复 | 文件 |
|------|------|------|
| `vt-random` 整型 dtype 下 truncate(U[0,1)) 恒为 0（静默全 0 陷阱，test-bug0 用例 17 标注） | 显式报错并指引 `vt-random-int` / `vt-random-integers`；仅接受 :float64/:float32 | random.lisp |

### 方向 3：测试基础设施修复（实测暴露）

| 问题 | 修复 | 文件 |
|------|------|------|
| 7 个测试文件开头 `(ql:quickload :clvt)`，而 run-tests.sh 只经 ASDF 加载系统、不加载 Quicklisp，导致 7 个套件必崩 | 改为 `(require :asdf)` + `#+quicklisp (ql:quickload ...)` + `(asdf:load-system :clvt)`，Quicklisp 存在与否均可运行 | test/*.lisp |
| `set -u` 下失败汇总 `${SUITES[$name]}` 对未注册套件报 unbound variable，吞掉失败清单 | 补默认值 `:-未知测试`；补齐 7 个套件的 SUITES 注册 | test/run-tests.sh |
| comprehensive-test "b/a(int)" 假失败：期望值 `2.3333333333333335` 缺 `d0` 被读成单精度，与正确的 float64 输出差 ~1.6e-7 > 1e-10 容差 | 期望值改双精度字面量 | test/comprehensive-test.lisp |
| test-bug0 以 `(and (= *checks* 28) (= *failures* 2))` 固化 2 个已知失败 | 修复后扩充为 37 项检查、0 失败，新增 9 项小整型归约回归（含 axis 归约与转置视图） | test/test-bug0.lisp |

### 验证

- 环境：SBCL 2.6.8（x86-64-linux binary）+ ASDF 3.3.1 + Quicklisp + sb-simd
- `bash test/run-tests.sh`：**19/19 套件全部通过**（run_all_tests 143、
  run_param_tests 155、robustness 194、coverage-gap 97、comprehensive 119、
  auto-compare 63、numpy-compare 69、test-bug0 37 等约 1100+ 断言）
- 冒烟覆盖：int8/uint8/int16/uint16 的 sum/amax/amin/argmax/prod/nansum/
  nanmax/all/any、axis 归约、转置视图、:out 写入、:dtype 覆盖、空轴归约
- einsum 小整型输入经通用路径提升 float64 验证通过（显式 :dtype/:out 仍限
  存储级类型并显式报错——已文档化的边界，不属静默错误）

---

## 2026-10-01 — v0.3.1 数据类型统一 / NaN·Inf 补全 / out 契约加固

在 v0.3.0 三层架构文档化基础上，针对三个遗留问题域做第二轮重构。

### 方向 1：数据类型统一（dtype.lisp 单一事实来源）

| 问题 | 修复 | 文件 |
|------|------|------|
| `vt-cast-fun` 对 int8/uint8 走 `%wrap-*`，int16/int32 走 `%coerce-*`，风格分裂 | 统一为 `%coerce-*`（语义一致，含 typep 快速路径） | dtype.lisp |
| `vt-arange` int64/int32 溢出直接存储（触发数组元素类型错误），int16/int8/uint 却回绕 | int64/int32 统一 `%wrap-*` 回绕（§9.2） | creation.lisp |
| `vt-linspace` float32 用 float32 累加，长序列漂移 | 以 double 精度计算、存储时舍入（对标 NumPy） | creation.lisp |
| `vt-bits->unsigned-dtype` 返回 `:uint32/:uint64`（非存储类型） | 文档标注保留符号状态 | dtype.lisp |

### 方向 2：NaN / Inf 特殊值补全

| 问题 | 修复 | 文件 |
|------|------|------|
| `vt-mod`/`vt-rem` 除 0：整数路径硬报错、浮点路径实现定义（NumPy 返回 0） | 除 0 返回 0 | elementwise.lisp |
| `vt-even-p`/`vt-odd-p` 对 NaN/±Inf 触发 `evenp`/`floor` 类型错误 | 非有限值判定为不成立（返回 0） | elementwise.lisp |
| `vt-hypot(NaN, ±Inf)` 结果依赖 SBCL `max` 的实现定义 | 任一 ±Inf → +Inf；NaN 单独出现时传播 | elementwise.lisp |
| `vt-random-uniform`：`assert(< low high)` 拒绝 low=high（NumPy 允许），NaN 边界报错误导 | 显式校验有限性；low=high → 常量数组 | random.lisp |
| `vt-random-normal`：负/NaN std 静默产生镜像分布 | 校验非负有限；std=0 → mean 填充（对标 NumPy scale=0） | random.lisp |

### 方向 3：:out 内存布局契约加固

| 问题 | 修复 | 文件 |
|------|------|------|
| `vt-det` 接受任意形状 :out 并被 `vt-fill` 静默填满 | :out 必须为 0 维张量，违反即报错（§8.2） | linalg.lisp |

审计确认已正确、无需改动的部分：`vt-map`/`vt-fast-map` 非连续 out 的 strides 写入与重叠快照保护；`def-vt-reduce` 家族（out-contig-tests 回归覆盖）；einsum 的 out strides 与重叠拷贝；`vt-where`/`vt-take`/`vt-pad`/`vt-triu`/`vt-tril`；`vt-sigmoid`/`vt-relu` 快路径连续性守卫；`log`/`sqrt`/`asin`/`acos`/`atanh`/`pow` 的复数域守卫与 NaN 返回；`maximum`/`minimum`/`fmax`/`fmin` 的 NaN 语义；median/percentile 的 NaN 传播。

---


## 2026-10-01 — v0.3.0 架构重构：三层分离语义契约落地

按《高性能张量库设计约定》完成整体重构审查，源码语义与文档对齐三层架构
（逻辑层 / 物理层 / 执行层）。全部修改不影响既有公开 API 签名。

### 修正

1. **`%inline1-strided` / `%inline2-strided` 误导性文档（关键修复）**
   两个宏自实现起就遍历结果张量自身 strides，一直支持任意 strides（含非连续 RES、
   广播输入），但文档错误标注"RES 必须连续"，差点误导执行层选路架构。
   已重写文档并显式声明"优化路径与通用路径输出必须逐位相同"。

2. **空归约语义（reduce-stats.lisp def-vt-reduce）**
   旧行为：空张量/空轴上 max/min 返回 ±Inf 哨兵、argmax/argmin 填 0。
   新行为按设计约定 §6.2：
   - 有单位元的操作返回单位元：`sum→0`、`prod→1`、`all→1`、`any→0`、
     `nansum→0`、`nanprod→1`；
   - `max/min/nanmax/nanmin` 无单位元：填充 NaN；整数结果 dtype 提升为
     float64 承载 NaN；显式整数 `:dtype`/整数 `:out` 与 NaN 结果冲突时报错；
   - `argmax/argmin/nanargmax/nanargmin` 无单位元且无哨兵值：一律报错。

3. **einsum `:out` 硬契约（linalg.lisp）**
   形状/dtype 校验由 `assert` 改为显式 `error`：assert 可能被
   `(safety 0)` 编译剔除，硬契约必须保证触发。

### 文档

- `package.lisp` 增加三层架构总纲（各层职责、失败模式与核心原则）。
- `core.lisp` vt 结构文档补全物理层四要素与 stride-0 广播只读语义。
- `map-reduce.lisp` 模块头写入执行层三条路径（快路径/strides 通用路径/vt-map）
  的选路规则与逐位等价承诺。
- `nn.lisp` `vt-softmax` 补充数值稳定化文档（全 -Inf 行 → NaN，对标 PyTorch）。
- `README.md` 新增"设计约定（三层架构语义契约）"章节：
  内存模型、视图/拷贝表、广播、类型提升、NaN 语义、归约、别名安全、
  out 契约、einsum 路由。

---

## 2026-09-05 — 测试体系重构与性能优化

### 测试体系重构：静态 JSON → 实时 NumPy 参考生成

**问题**：原测试体系依赖三份静态预先生成的 JSON 文件（`all_expected.json`、`param_expected.json`、`expected_numpy.json`）作为参考值对比基准，存在历史数据过期、维护困难、强依赖 PyTorch 等问题。

**修复**：全部 8 个测试套件中的数值对比测试改为**运行时实时调用 Python + NumPy 生成参考值**，不再依赖任何历史静态 JSON 数据。

#### 修改的测试文件

| 文件 | 改动 |
|------|------|
| `test/ref_compute.py` | 重写：核心用纯 NumPy 实现，覆盖 8 个测试套件约 900 个用例；保留 PyTorch 交叉验证层（76 项核心算子独立验证） |
| `test/run_all_tests.lisp` | 移除对 `all_expected.json` 的读取，改为实时调用 Python 生成参考值；内置 JSON 解析器；改进 approx 跨类型比较 |
| `test/run_param_tests.lisp` | 移除对 `param_expected.json` 的依赖，改为实时调用；修复 `vt-narrow` 参数语义（start/end 而非 start/length）；删除重复 convolve 项 |
| `test/auto-compare-test.lisp` | 移除对 `expected_numpy.json` 的依赖，改为实时调用；修复键名冲突（`roll_2` / `flip_axis0` 等分配唯一键） |
| `test/numpy-compare-test.lisp` | 改为调用更新后的 `ref_compute.py`；改进数值比较容差 |
| `test/run-tests.sh` | 不再强制要求 PyTorch，仅检查 NumPy 可用性 |

#### 删除的过时文件
- ❌ `test/all_expected.json`（约 3500 行历史硬编码数据）
- ❌ `test/param_expected.json`（约 9500 行历史硬编码数据）
- ❌ `test/expected_numpy.json`（约 1200 行历史硬编码数据）

### Bug 修复

| Bug | 位置 | 严重程度 | 修复 |
|-----|------|----------|------|
| `vt-nanmean` / `vt-nanvar` 类型转换崩溃（浮点布尔掩码强制转 int64 导致 TYPE-ERROR） | `reduce-stats.lisp` | 🔴 严重 | count 改用与结果相同的浮点 dtype，zerop/divisor 全部浮点化 |
| `(vt-/ x)` 一元倒数走普通 vt-map 带 lambda 开销 | `elementwise.lisp` | 🟡 中等 | 改用 `vt-fast-map` 标量广播除法快路径 |

### 性能优化

#### 1. vt-sum / vt-mean 全局归约连续内存快路径（≈30×）
**位置**：`src/map-reduce.lisp`

为 **连续内存 + 全局归约（axis=nil）** 添加平面 dotimes 循环快路径，使用 macrolet 生成按元素类型特化的循环，消除通用递归 stride 遍历和每元素 funcall 开销。

| 操作 | 优化前 | 优化后 | 提升 |
|------|--------|--------|------|
| `vt-sum`（1M 元素全局归约） | 1452 ms | **48 ms** | **30×** |
| `vt-mean`（1M 元素全局归约） | ~1450 ms | **~50 ms** | **~29×** |

#### 2. vt-sigmoid 内联快路径（≈5.6×）
**位置**：`src/nn.lisp`

为连续 float 张量添加 `%sigmoid-fast` 内联快路径，直接展开 `1.0 / (1.0 + exp(-x))` 计算。

| 操作 | 优化前 | 优化后 | 提升 |
|------|--------|--------|------|
| `vt-sigmoid`（1M 元素） | 472 ms | **84 ms** | **5.6×** |

#### 3. vt-relu 内联快路径（≈7.9×）
**位置**：`src/nn.lisp`

为连续 float 张量添加 `%relu-fast` 内联快路径，直接展开 `(max 0.0 x)` 计算。

| 操作 | 优化前 | 优化后 | 提升 |
|------|--------|--------|------|
| `vt-relu`（1M 元素） | 472 ms | **60 ms** | **7.9×** |

### 新增功能（extensions.lisp）

| 函数 | 对标 | 说明 |
|------|------|------|
| `vt-clamp` | `torch.clamp` | `vt-clip` 的 PyTorch 风格别名 |
| `vt-copy-to!` | `tensor.copy_` | PyTorch 风格原地拷贝，便于链式调用和内存复用 |

### 新增测试 / 基准工具

| 文件 | 说明 |
|------|------|
| `test/performance-bench.lisp` | Common Lisp 性能基准套件：算术 / 激活函数 / 归约 / 矩阵乘法 |
| `test/torch-compare.py` | 独立 PyTorch 对比测试 + 性能基线（18 项核心算子） |

### 测试结果

修改后全部 **8 个测试套件、约 900 个测试用例** 全部通过 ✅：

```
=== 测试总结 ===
  运行套件: 8
  通过:     8
  失败:     0
```

| 测试套件 | 用例数 | 说明 |
|----------|--------|------|
| `run_all_tests` | 143 | 基础函数（实时 NumPy 对比） |
| `run_param_tests` | 154 | 3D+ 参数化（实时 NumPy 对比） |
| `nested-test` | 60 | AI/ML 函数组合 |
| `robustness-test` | 194 | 鲁棒性边界 |
| `coverage-gap-test` | 97 | NumPy/PyTorch 覆盖差距 |
| `comprehensive-test` | 119 | 综合功能 |
| `auto-compare-test` | 63 | 自动对比（实时 NumPy 对比） |
| `numpy-compare-test` | 69 | NumPy/PyTorch 实时对比 |

---

## 2026-08-15 — 架构重构与性能优化

### 架构重构

从零重构了整个库，将原来职责混杂的 17 个顶层 `.lisp` 文件重组为 `src/` 下按职责分层的 20 个模块（详见 README「架构」一节），保持全部 `vt-*` 公共 API 签名兼容：

- **内核层**：`dtype.lisp`（类型系统单一事实来源）、`core.lisp`（结构/步长/广播/连续判定/拷贝/填充）、`iterator.lisp`、`map-reduce.lisp`
- **功能层**：`creation` / `manip` / `indexing` / `join` / `elementwise` / `reduce-stats` / `setops` / `linalg` / `nn` / `random` / `rotate` / `io` / `extensions`

### 规范性修正

- 补齐 `vt-compute-logical-strides` 等「已导出但未定义」的悬空符号
- `package.lisp` 导出按功能分组，`*vt-fun-list*` 自动收集
- 修复 `benchmark-copy.lisp` 中非法 FORMAT 指令 `~30-50x`

### 性能优化

- `vt-map` 重写为按输出元素类型特化（`(the (simple-array double-float (*)) ...)`），消除浮点装箱
- 恢复 `vt-fast-map` 编译期内联（`%inline1-loop` / `%inline2-loop` / `%cast-to`），`vt-+`/`vt-*`/`vt--`/`vt-/`、`vt-add`/`vt-sub`/`vt-mul`/`vt-div`/`vt-scale` 及一元数学函数（`vt-sin`/`vt-cos`/`vt-exp` 等）均内联算子，避免 `funcall` 装箱
- 基准（100 万元素，暖机后）：`vt-+` ≈ 0.004s、`vt-add` ≈ 0.009s、`vt-sin` ≈ 0.027s，达到或超越重构前水平；内联路径装箱量从 ~56MB 降至 ~8MB（仅结果张量本身）
- `vt-reduce` 内层递归循环按输入/输出元素类型特化（`macrolet` 生成 `(the (simple-array <type> (*)) ...)` 的 `recurse`，整数走 `truncate`、浮点走 `coerce`），消除 `vt-cast` 运行时 `ecase` 分派与通用 `aref` 的逐元素装箱；100 万元素下 `vt-sum` 提速约 1.5–1.75×、`vt-mean` 约 1.5–1.7×、`vt-amax` 约 1.3–1.5×
- `vt-ref` / `(setf vt-ref)` 按数组元素类型特化最终 `aref`，用 `(some #'zerop shape)` 替代 `vt-size` 全量 `reduce`，并补上显式越界检查（负索引归一化后校验范围，越界确定性报错，不再依赖 `aref` 的 safety 相关边界检查）；1D/2D/3D 标量访问提速约 1.3–1.6×

### 测试体系

- 新增 `test/numpy-compare-test.lisp` + `test/ref_compute.py`：SBCL 运行时调用 `python3` 实时生成 numpy / pytorch 参考结果并对比（69 项，覆盖创建/逐元素/归约/线性代数/神经网络/集合/形状/torch.topk）
- `test/run-tests.sh` 更新：纳入新套件、python3/numpy/torch 可用性检查
- 总计 **900 个测试用例**（831 原有 + 69 新增实时对比），全部通过 ✅

### 回归修复

通过 `example.lisp` 额外捕获并修复了重写引入的 4 处回归：

| 问题 | 位置 | 修复 |
|------|------|------|
| `vt-gradient` 切片规格多余嵌套 / 引号变量 | `reduce-stats.lisp` | `'((1 2))`→`'(1 2)`、`'(2 n)`→`(list 2 n)` |
| `vt-histogram` 缺失 `with-float-safe` 触发 NaN/Inf 浮点陷阱 | `reduce-stats.lisp` | 恢复包装 |
| `vt-unique` 缺失 `with-float-safe` 触发 `(= +inf NaN)` 陷阱 | `setops.lisp` | 恢复包装 |
| `vt-delete` 错误消息格式串与参数个数不匹配 | `join.lisp` | 补全 `~d` 占位符 |

### Bug 修复（数值转换与返回类型）

- `vt-cast` 对 `:int8`/`:int16`/`:uint8`/`:uint16` 增加 NumPy 语义的回绕（mod）：超出范围的值不再直接返回导致 TYPE-ERROR，而是回绕到目标范围内（如 `(vt-cast 200.0 :int8)` → -56、`(vt-cast 256 :uint8)` → 0、`(vt-cast 65536 :uint16)` → 0）
- `vt-cast-fun` 对 `:uint8`/`:uint16`/`:int8`/`:int16` 返回与 `vt-cast` 语义一致的转换函数（此前返回 `#'truncate`，负数行为不一致）
- `vt-sinc` 整数输入时按 `%infer-float-dtype` 推断浮点结果类型（此前沿用整数 dtype 导致精度丢失）
- `vt-gradient` 当 `axis=nil` 且输入为一维时返回单个 VT 对象（此前返回含单元素的列表，与 NumPy 不一致）

---

## 2026-08-10 — MiMo (Xiaomi AI) 项目审查与功能补充

### 项目审查

对整个 clvt 项目进行了全面审查，覆盖全部 15 个源文件（约 8000 行代码），识别出 NumPy/PyTorch 生态中常用但缺失的关键函数，并补充实现。

### 新增功能 (extensions.lisp)

| 函数 | 对标 | 说明 |
|------|------|------|
| `vt-count-nonzero` | `numpy.count_nonzero` | 统计非零元素个数，支持 axis/keepdims |
| `vt-count` | — | 统计等于指定值的元素个数 |
| `vt-flatnonzero` | `numpy.flatnonzero` | 展平后返回非零元素的一维索引 |
| `vt-moveaxis` | `numpy.moveaxis` | 将轴从源位置移动到目标位置，返回零拷贝视图 |
| `vt-inner` | `numpy.inner` | 内积，沿最后一个轴收缩 |
| `vt-tensordot` | `numpy.tensordot` | 张量缩并，支持整数轴和显式轴对两种模式 |
| `vt-topk` | `torch.topk` | 沿指定轴获取前 k 个最大/最小值及其索引 |
| `vt-clip-tensor` | — | 支持张量作为上下边界的裁剪（`vt-clip` 的扩展版） |
| `vt-set-print-options` | `numpy.set_print_options` | 设置打印阈值、精度、缩步长 |
| `vt-get-print-options` | — | 获取当前打印选项 |

### Bug 修复

| Bug | 严重程度 | 修复方式 |
|-----|----------|----------|
| README.md 中 `vt-trancate` 拼写错误 | 🟢 轻微 | 修正为 `vt-truncate`，与实际导出名一致 |

### 构建与环境

- SBCL 安装：从 SourceForge 下载 SBCL 2.6.7 二进制包，安装到 `~/.local/`
- Quicklisp：自动安装并配置 `~/quicklisp/local-projects/clvt` 软链接
- Python3 + NumPy：确认系统自带可用

### 测试

- 新增 `test/test-extensions.lisp`，包含 **19 个测试用例**，覆盖全部新增函数
- 原有 **815 个测试用例** 全部通过，无回归
- 总计 **834 个测试用例**，全部通过 ✅

---

## 2026-08-01 — MiMo (Xiaomi AI) 系统性测试与修复

### Bug 修复

| Bug | 严重程度 | 修复方式 |
|-----|----------|----------|
| `vt-copy-into` 慢速路径对非连续视图产生错误结果 | 🔴 严重 | 递归→显式迭代器，修复 SBCL 编译器对递归闭包的优化问题 |
| `vt-qr` Householder QR 分解崩溃 | 🔴 严重 | einsum→直接循环，分离 w 计算与 R 更新 |
| `vt-diagonal` 3D 张量数组越界 | 🔴 严重 | 重写为迭代实现，修复 out-ptr 不传播问题 |
| `vt-eig` Jacobi 特征值分解错误 | 🔴 严重 | 修正旋转角计算 (atan)、V 更新公式、off-diagonal 更新顺序 |
| `vt-convolve` 崩溃 | 🟡 中等 | 添加 `vt-contiguous` 保护负步长视图 |
| `vt-reduce` 空张量 prod 返回 0 | 🟡 中等 | 空张量防御改为返回 `init-val` |

### 性能优化

- `vt-copy-into` 慢速路径：递归→显式迭代器，性能持平或略优（大张量快 4%），修复正确性 bug
- `vt-qr`：用直接循环替代 einsum 调用，消除非连续视图的 stride 处理问题

### 新增功能

- `vt-cholesky`：Cholesky 分解（正定矩阵 → 下/上三角）
- `vt-eig`：对称矩阵特征值分解（Jacobi 旋转法 + atan）
- `vt-pinv`：Moore-Penrose 伪逆（基于 SVD）
- `vt-lstsq`：最小二乘解（基于 SVD）

### 补充导出

以下函数在原代码中已实现但未导出，本次补充到 package.lisp：

- `vt-asinh` / `vt-acosh` / `vt-atanh`：反双曲函数
- `vt-reciprocal` / `vt-negative` / `vt-lerp` / `vt-cbrt`：补充数学函数
- `vt-bit-and` / `vt-bit-ior` / `vt-bit-xor` / `vt-bit-not` / `vt-left-shift` / `vt-right-shift`：位运算
- `vt-fmax` / `vt-fmin`：NaN 忽略的逐元素极值
- `vt-nansum` / `vt-nanmean` / `vt-nanstd` / `vt-nanvar` / `vt-nanmax` / `vt-nanmin`：NaN 感知统计
- `vt-fill` / `vt-interp` / `vt-kron` / `vt-meshgrid`：填充、插值、克罗内克积、网格生成
- `vt-append` / `vt-insert` / `vt-delete`：追加、插入、删除

### 测试体系

构建了全面的自动化测试体系，共 **815 个测试用例**，全部通过：

| 测试套件 | 数量 | 覆盖范围 |
|----------|------|----------|
| `run_all_tests.lisp` | 143 | 基础函数 + NumPy 对比 |
| `run_param_tests.lisp` | 155 | 3D+ 参数化测试（多 axis、keepdims） |
| `nested-test.lisp` | 60 | AI/ML 函数组合（Linear、Attention、BatchNorm、MLP 等） |
| `robustness-test.lisp` | 178 | 边界条件（标量、空张量、NaN/Inf、数值稳定性） |
| `coverage-gap-test.lisp` | 97 | numpy/pytorch 覆盖差距（布尔索引、einsum 高级、torch.nn 模式） |
| `comprehensive-test.lisp` | 119 | 综合功能测试 |
| `auto-compare-test.lisp` | 63 | JSON 驱动自动对比 |

测试可通过 shell 脚本一键运行：
```bash
bash test/run-tests.sh          # 运行所有测试
bash test/run-tests.sh --list   # 列出所有测试套件
bash test/run-tests.sh --suite run_all_tests  # 运行指定套件
```

---

## 2026-07-30 — 初始版本

由'智谱清言'AI(GLM5+)和'DeepSeek'(v4pro) AI 共同编写。

### 核心架构

- 三大核心函数：`vt-einsum`、`vt-map`、`vt-reduce`
- 所有函数以 `vt-` 前缀命名
- 支持 4 种数据类型：`:int32`、`:int64`、`:float32`、`:float64`
- 完美的打印输出功能
- 零拷贝视图操作

### 已实现功能

- 张量创建（arange、linspace、zeros、ones、eye 等）
- 形状操作（reshape、transpose、squeeze、concatenate、stack 等）
- 索引与切片（ref、slice、where、nonzero 等）
- 算术运算（四则运算、三角函数、指数对数等）
- 比较与逻辑
- 归约与统计（sum、mean、std、var、median、percentile 等）
- 线性代数（matmul、solve、inv、det、qr、svd 等）
- einsum 爱因斯坦求和
- 激活函数（sigmoid、relu、tanh、gelu 等）
- 损失函数（softmax、cross-entropy、mse 等）
- 集合操作
- 随机数生成
- NaN/Inf 处理
