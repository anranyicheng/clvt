#!/usr/bin/env bash
# ============================================================
# run-tests.sh — clvt 自动化测试运行器（唯一入口）
# 用法: bash test/run-tests.sh [选项]
#   无参数: 运行所有默认套件（跳过性能基准）
#   --suite NAME: 只运行指定测试套件
#   --list: 列出所有可用测试套件
#   --all: 连性能基准一起运行
#
# 说明: test/ 下每个测试套件都在本脚本登记（SUITES + DEFAULT_ORDER/
#       BENCH_SUITES）。启动时自检运行列表覆盖全部已注册套件，任何
#       「新增了测试文件却忘记接线」都会直接报错退出。
# ============================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
SBCL="${SBCL:-sbcl}"
TIMEOUT="${TIMEOUT:-300}"
# SBCL 动态空间大小（MB）。冷编译整个库（新克隆目录首次运行、CI 干净容器）
# 在 SBCL 默认动态空间下可能直接 "Heap exhausted, game over" 崩溃，进程带着
# 空输出退出 —— 运行器只能报一句无法追查的 "?/?"。统一给足内存。
SBCL_DYNAMIC_SPACE_SIZE="${SBCL_DYNAMIC_SPACE_SIZE:-4096}"

# 颜色
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

# 测试套件定义
declare -A SUITES
SUITES[run_all_tests]="143 基础函数测试"
SUITES[run_param_tests]="155 3D+ 参数化测试"
SUITES[nested-test]="60 AI/ML 函数组合测试"
SUITES[robustness-test]="178 鲁棒性边界测试"
SUITES[coverage-gap-test]="97 numpy/pytorch 覆盖差距测试"
SUITES[comprehensive-test]="119 综合功能测试"
SUITES[auto-compare-test]="63 JSON 自动对比测试"
SUITES[legacy-coverage-test]="284 静态JSON时代未接线用例盘活 (gen_all/gen_param 遗产 281 例 + narrow 语义回归 3 例)"
SUITES[benchmark-copy]="性能基准测试"
SUITES[numpy-compare-test]="69 numpy/pytorch 实时对比测试"
SUITES[extensions2-test]="37 新增函数测试 (fliplr/geomspace/one-hot/layer-norm等)"
SUITES[nan-random-test]="89 NaN 随机数测试"
SUITES[property-test]="性质测试"
SUITES[test-extensions]="19 扩展函数回归测试"
SUITES[simd-test]="SIMD 路径测试"
SUITES[test-overlap]="6 重叠拷贝回归测试"
SUITES[svd-fast-path-test]="SVD 双路径测试 (差分参考/选路不变量/数学性质/numpy对比/退化形状)"
SUITES[linalg-fast-path-test]="LINALG 双路径测试 (lu/qr/solve/chol/eig: 差分参考/选路不变量/数学性质/numpy对比/边界)"
SUITES[test-copy-into]="35 vt-copy-into 正确性测试"
SUITES[io-interop-test]="77 vt-load/vt-save 持久化与 numpy .npy 互操作测试（含 NaN payload 逐位/视图字节等价/不变性/覆盖语义）"
SUITES[test-simd-batch-matmul]="5 SIMD 批量矩阵乘测试"
SUITES[test-bug0]="37 已知 bug 回归测试"
SUITES[test-ai-edge-cases]="40 AI 主流函数语义回归测试 (matmul 1-D/广播 0-size/BCE NaN/vstack)"
SUITES[memsafety-tests]="6 :out 快路径内存安全测试"
SUITES[out-contig-tests]="23 out 连续/非连续写入测试"
SUITES[shape-degenerate-tests]="70 空形状/退化形状表驱动测试 (C1/C2 类目, 见 TEST-PLAN.md)"
SUITES[property-strided-tests]="67 选路不变量/代数恒等式测试 (C8 类目, strided≡contiguous)"
SUITES[error-contract-tests]="19 错误路径契约测试 (C7 类目, L2 底线)"
SUITES[refactor-bugfix-tests]="55 v0.3.5 重构回归测试 (floor族NaN/insert numpy语义/solve校验/tensordot负轴等)"
SUITES[contract-regression-test]="136 numpy/pytorch 契约对齐回归 (广播写穿拦截/roll拷贝语义/diagonal轴参数/均分整除/where单参形态/标量choice/浮点索引报错但空索引合法/digitize形状·bins/insert moveaxis；期望来自 numpy 实测)"
SUITES[numpy-convention-tests]="63 numpy 对齐语义测试 (空归约/:out解耦/mod零除/true_divide/random校验/clip·where 签名)"
SUITES[out-contract-v2-test]="56 :out 契约 v2 严格相等测试 (形状/dtype/可写/别名/非连续 out)"
SUITES[nan-broadcast-test]="31 NaN/Inf 广播对齐 numpy 测试 (算术/极值/比较/归约/dtype 提升)"
SUITES[uncovered-coverage-test]="136 覆盖缺口测试 (此前无任何测试/example 调用的公开函数 + identity/vander/atanh 回归)"
SUITES[extensions3-test]="261 补充 NumPy 缺失函数测试 + 审查修复回归 (extensions3 + 审查报告 P0/P1/P2 + R-3 整数提升回归)"
SUITES[fastmap-invariant-test]="24 数学函数快路径不变量测试 (宏展开结构 + 计时上界, 防 vt-fast-map→vt-map 性能回归)"
SUITES[complex-linalg-test]="216 复数线性代数测试 (46 项 numpy 2.4.6 差分对照: det/inv/solve/cholesky/QR/SVD/厄米特特征值/pinv/lstsq/rank/cond/matrix-power/einsum + 重构&正交&酉恒等式 + MP 四条条件 + 随机/退化/SPD/lstsq 数值压测(108 条, σ²=eig(A^H A) 交叉验证) + :out 与奇异/非正定报错契约 + 实数回归网; 期望由 test/gen_complex_linalg_expected.py 生成)"
SUITES[setops-axis-trace-test]="56 按轴去重/迹 测试 (37 项 numpy 2.4.6 差分对照: vt-unique :axis 含 return-index/inverse/counts、NaN 行不折叠、非连续视图、退化形状 + vt-trace :offset/:axis1/:axis2 含高维收缩 + 契约断言 + 实数回归网; 期望由 test/gen_setops_axis_expected.py 生成)"
SUITES[einsum-gemm-layout-test]="36 einsum GEMM 布局归一化 (§16.26: NN/NT/C-转置/兜底 四条路径各档尺寸 + 批量含广播与批量NT + f32/i32/i64/c128/c64 + 转置输入/切片/非连续 :out/小矩阵 + SIMD关与单线程不变; 主断言 = 与 vt-matmul 逐位相等)"
SUITES[complex-ops-test]="128 复数数据搬运层测试 (97 项 numpy 2.4.6 差分对照: 转换/拼接/剪切/集合/nan 族/缩并/卷积 + dtype&shape 契约 + :out 硬契约 + 未支持组合报错 + 实数回归网; 期望由 test/gen_complex_ops_expected.py 生成)"
SUITES[complex-dispatch-test]="15 复数分派结构回归（第十二轮 §16.32: 92 个已迁移符号必须是具名 DEFUN 而非 setf 装的匿名包装（重载安全的充分必要条件）· list 入参走复数路径 · 拒绝型守卫 · 实数路径不回归 · 复数语义抽样 · :out 契约）"
SUITES[dtype-weak-scalar-test]="14 弱标量提升回归（第十一轮 §16.31: Lisp 标量按 numpy NEP 50 只贡献种类不贡献宽度 —— f32/c64/i32/i8/u8 的「+标量」不再静默升宽; 覆盖 vt-+ / vt-/ / vt-pow / vt-maximum 与值正确性）"
SUITES[complex-nan-inf-test]="77 复数 NaN/Inf 与 median 契约（第十轮 §16.30: median 在 axis=nil/空 fiber/NaN 传播三处与实数路径分叉 · 前缀扫描首元素直拷（-0.0 符号位、Inf 元素）· nan* 家族含 ±Inf 与全 NaN 切片 · isnan/isinf/isfinite/nan_to_num/含 NaN 排序 · axis 元组归约 · :out 非连续/转置/原地 · einsum 九种模式）"
SUITES[complex-scan-axis-test]="26 复数沿轴扫描回归（第九轮 §16.29: cumsum/cumprod/nancumsum/nancumprod 在 axis<rank-1 时只算第一条 fiber 的静默错值 · 2D 两轴 + 3D 三轴 + 负轴 + 转置视图 + c64 + 退化/空轴 + :out 连续与非连续 · 期望值取自 numpy 2.3.5 逐位相等 · 实数路径回归）"
SUITES[complex-gap-test]="50 复数缺口回归（第八轮 §16.28: from-array/from-sequence/full 的 dtype 推断保住虚部 · float_power 复数主值分支与 NaN 边界 · cbrt 对齐 numpy 的确定性拒绝 · layer-norm 复数分支（复数均值+实数方差）· 已支持函数抽查）"
SUITES[complex-matmul-perf-test]="53 复数 matmul 内核回归 (正确性: 朴素三重循环逐位对照/尾路径 n%4/退化与零尺寸/切片视图 offset/批量与 batch 广播/einsum NN·NT·转置输出逐位一致/c64·c128/:out; 性能护栏: 新内核 ≥1.3× 旧形状、≤3.5× 实数标量行内核、并行不改结果; 见 CONVENTIONS §16.27)"
SUITES[complex-test]="136 复数语义测试 (83 项 numpy 2.4.6 差分对照 + dtype 提升 100 格 + 分支切割/NaN/字典序/视图/out 契约/未支持报错 + 实数回归, 期望由 test/gen_complex_expected.py 生成)"
SUITES[test-all]="641 绝大多数函数的测试"
SUITES[core-test]="282 基础契约回归 (1 结构/199 函数语义/返回值形状/错误契约, 参照 CONVENTIONS)"
SUITES[differential-probes-test]="496 差分探针 vs numpy (期望内嵌逐项判定, 覆盖算术/超越/归约/索引/操纵/随机/错误路径)"
SUITES[differential-probes-test2]="405 第2阶段差分探针 vs numpy (导出符号 #200-#377: NaN统计/linalg分解/NN/随机/契约, 期望内嵌逐项判定)"
SUITES[manip-interop-test]="195 形状/视图/拼接/切片 跨语言互操作测试 (Lisp vt-save 落盘 + NumPy 逐位校验, 1000x1000 基准)"
SUITES[performance-bench]="性能基准测试 (1M 元素 / 1k×1k / 100×100 三档)"
SUITES[complex-bench]="复数层性能基准 (1M 元素: 算术/超越/归约/排序/比较/线代, 对照 float64 同规模; 见 CONVENTIONS §16.14)"

# 需要「Lisp 生成数据 + Python/NumPy 校验」两阶段流水线的套件
# 格式: PIPELINE_SUITES[name]="<阶段1 .lisp 文件> <阶段2 .py 文件>"
declare -A PIPELINE_SUITES
PIPELINE_SUITES[manip-interop-test]="vt-manip-interop-test.lisp verify_manip_numpy.py"

# ============================================================
# 运行列表 —— 测试的唯一入口
#   DEFAULT_ORDER : 默认执行的套件及其顺序
#   BENCH_SUITES  : 性能基准类，默认跳过，--all 时执行
# 启动时自检二者之并必须恰好覆盖 SUITES 的全部条目，防止新增测试文件后
# 忘记接线（历史上 legacy-coverage-test 曾被注册却从未纳入运行列表）。
# 新增测试的步骤：写 <name>.lisp（或流水线套件）→ 登记 SUITES →
# 加入下面两个列表之一。
# ============================================================
DEFAULT_ORDER=(
    core-test
    run_all_tests
    run_param_tests
    nested-test
    robustness-test
    coverage-gap-test
    comprehensive-test
    auto-compare-test
    numpy-compare-test
    extensions2-test
    nan-random-test
    property-test
    test-extensions
    simd-test
    test-overlap
    test-copy-into
    test-simd-batch-matmul
    test-bug0
    test-ai-edge-cases
    memsafety-tests
    out-contig-tests
    shape-degenerate-tests
    property-strided-tests
    svd-fast-path-test
    linalg-fast-path-test
    error-contract-tests
    refactor-bugfix-tests
    contract-regression-test
    numpy-convention-tests
    out-contract-v2-test
    nan-broadcast-test
    uncovered-coverage-test
    extensions3-test
    fastmap-invariant-test
    complex-dispatch-test
    dtype-weak-scalar-test
    complex-nan-inf-test
    complex-scan-axis-test
    complex-gap-test
    complex-matmul-perf-test
    complex-test
    complex-ops-test
    complex-linalg-test
    setops-axis-trace-test
    einsum-gemm-layout-test
    legacy-coverage-test
    differential-probes-test
    differential-probes-test2
    io-interop-test
    manip-interop-test
    test-all
)

BENCH_SUITES=(
    benchmark-copy
    performance-bench
    complex-bench
)

# ============================================================
# 辅助函数
# ============================================================

log() { echo -e "${CYAN}[INFO]${NC} $*"; }
ok()  { echo -e "${GREEN}[PASS]${NC} $*"; }
fail(){ echo -e "${RED}[FAIL]${NC} $*"; }
warn(){ echo -e "${YELLOW}[WARN]${NC} $*"; }
header() { echo -e "\n${BOLD}=== $* ===${NC}"; }

usage() {
    cat <<EOF
clvt 自动化测试运行器

用法: bash test/run-tests.sh [选项]

选项:
  无参数          运行所有默认套件 (跳过性能基准)
  --suite NAME    只运行指定测试套件
  --list          列出所有可用测试套件 (标注默认跳过项与流水线套件)
  --quick         跳过性能基准和慢速测试 (等同无参数)
  --all           运行所有测试包括性能基准 (benchmark-copy / performance-bench)
  --verbose       显示完整 SBCL 输出
  --help          显示此帮助信息

环境变量:
  SBCL            SBCL 可执行文件路径 (默认: sbcl)
  SBCL_DYNAMIC_SPACE_SIZE
                  SBCL 动态空间 MB (默认: 4096)。冷编译整个库时若用 SBCL
                  默认值可能 "Heap exhausted" 崩溃并报出无法追查的 ?/?
  TIMEOUT         每个测试超时秒数 (默认: 300)
  CLVT_TEST_OUT   流水线套件的产物目录 (默认: 临时目录，跑完自动清理)

本脚本是 clvt 全部测试的唯一入口：test/ 下每个测试套件（含「Lisp 生成 +
NumPy 校验」两阶段流水线套件）都在此注册，启动时会自检运行列表是否覆盖
全部已注册套件。

示例:
  bash test/run-tests.sh
  bash test/run-tests.sh --list
  bash test/run-tests.sh --suite manip-interop-test
  bash test/run-tests.sh --suite contract-regression-test
  bash test/run-tests.sh --all --verbose
EOF
}

check_sbcl() {
    if ! command -v "$SBCL" &>/dev/null; then
        fail "SBCL 未找到。请安装 SBCL 或设置 SBCL 环境变量。"
        exit 1
    fi
    log "SBCL: $($SBCL --version 2>&1 | head -1)"
}

check_asdf() {
    log "检查 ASDF 注册..."
    "$SBCL" --noinform --non-interactive \
        --eval '(require :asdf)' \
        --eval '(format t "ASDF ~a~%" (asdf:asdf-version))' 2>/dev/null \
        || { fail "ASDF 加载失败"; exit 1; }
}

check_python() {
    if command -v python3 &>/dev/null; then
        if python3 -c 'import numpy' &>/dev/null; then
            log "python3: numpy $(python3 -c 'import numpy;print(numpy.__version__)')"
            log "(numpy-compare-test 不再要求 torch，使用纯 numpy 实现)"
            return 0
        else
            warn "未找到 numpy，跳过 numpy-compare-test"
            return 1
        fi
    else
        warn "未找到 python3，跳过 numpy-compare-test"
        return 1
    fi
}

# ------------------------------------------------------------
# 启动自检：运行列表必须恰好覆盖已注册套件
#   防止「文件写了、SUITES 登记了、却没人运行」的静默漏接线。
# ------------------------------------------------------------
check_suite_wiring() {
    local missing=() unknown=() name s found
    for name in "${!SUITES[@]}"; do
        found=0
        for s in "${DEFAULT_ORDER[@]}" "${BENCH_SUITES[@]}"; do
            [[ "$s" == "$name" ]] && { found=1; break; }
        done
        [[ $found -eq 1 ]] || missing+=("$name")
    done
    for s in "${DEFAULT_ORDER[@]}" "${BENCH_SUITES[@]}"; do
        [[ -n "${SUITES[$s]:-}" ]] || unknown+=("$s")
    done

    if [[ ${#missing[@]} -gt 0 ]]; then
        fail "以下套件已注册但未接入运行列表，请补进 DEFAULT_ORDER 或 BENCH_SUITES:"
        printf '  - %s\n' "${missing[@]}"
        return 1
    fi
    if [[ ${#unknown[@]} -gt 0 ]]; then
        fail "运行列表中出现了未注册到 SUITES 的套件:"
        printf '  - %s\n' "${unknown[@]}"
        return 1
    fi
    # 流水线套件的两个阶段文件必须存在，且阶段 1 文件不应与套件名同名（避免走错分支）
    for name in "${!PIPELINE_SUITES[@]}"; do
        local spec="${PIPELINE_SUITES[$name]}" lisp_rel py_rel
        read -r lisp_rel py_rel <<<"$spec"
        if [[ -z "${SUITES[$name]:-}" ]]; then
            fail "PIPELINE_SUITES 中出现未注册套件: $name"
            return 1
        fi
        [[ -f "${SCRIPT_DIR}/${lisp_rel}" ]] || { fail "流水线套件 $name 缺少阶段 1 文件: $lisp_rel"; return 1; }
        [[ -f "${SCRIPT_DIR}/${py_rel}"   ]] || { fail "流水线套件 $name 缺少阶段 2 文件: $py_rel";   return 1; }
    done
    return 0
}

# ------------------------------------------------------------
# 两阶段流水线套件
#   阶段 1: SBCL 加载 clvt 跑用例，把结果用 vt-save 落盘（.npy / .lisp），
#           并做 Lisp 端 oracle / 错误契约 / 落盘往返自检；
#   阶段 2: NumPy 读取落盘文件，逐用例比对 shape / dtype / 数值。
#   两个阶段都必须退出码为 0 才算通过。
# ------------------------------------------------------------
run_pipeline_suite() {
    local name="$1"
    local desc="${SUITES[$name]:-未知测试}"
    local spec="${PIPELINE_SUITES[$name]}"
    local lisp_rel py_rel
    read -r lisp_rel py_rel <<<"$spec"
    local lisp_file="${SCRIPT_DIR}/${lisp_rel}"
    local py_file="${SCRIPT_DIR}/${py_rel}"

    [[ -f "$lisp_file" ]] || { fail "测试文件不存在: $lisp_file"; return 1; }
    [[ -f "$py_file"   ]] || { fail "校验脚本不存在: $py_file";   return 1; }

    header "$name — $desc"

    if ! python3 -c 'import numpy' &>/dev/null; then
        warn "$name: 未找到 python3/numpy，跳过本轮"
        return 0
    fi

    # 输出目录：默认用临时目录，跑完即清理；用 CLVT_TEST_OUT 可指定以保留产物
    local out_dir own_out=0
    if [[ -n "${CLVT_TEST_OUT:-}" ]]; then
        out_dir="$CLVT_TEST_OUT"
        mkdir -p "$out_dir"
    else
        out_dir="$(mktemp -d "${TMPDIR:-/tmp}/clvt-interop.XXXXXX")"
        own_out=1
    fi

    local start_time end_time rc=0
    start_time=$(date +%s)

    log "阶段 1/2: Lisp 生成数据 (vt-save) + Lisp 端自检"
    log "  阶段 1 文件: $lisp_file"
    log "  输出目录   : $out_dir"
    local lisp_out="" lisp_code=0
    # 注意：--dynamic-space-size 是 C 运行时选项，必须排在 Lisp 选项
    # （--non-interactive 之后即报 "in the middle of Lisp options"）之前。
    #
    # 另外先钉死 *read-default-float-format* = SINGLE-FLOAT：测试文件里的
    # 期望值含裸浮点字面量，用户 .sbclrc 若改成 double-float，会把
    # 「期望值变 double、实得 single」误报成 FAIL（CONVENTIONS §16.18.6）。
    lisp_out=$(CLVT_TEST_OUT="$out_dir" "$SBCL" \
        --dynamic-space-size "$SBCL_DYNAMIC_SPACE_SIZE" \
        --noinform --non-interactive \
        --eval "(setf *read-default-float-format* 'single-float)" \
        --eval "(require :asdf)" \
        --eval "(push #p\"${PROJECT_DIR}/\" asdf:*central-registry*)" \
        --eval "(asdf:load-system :clvt)" \
        --eval "(load \"$lisp_file\")" \
        2>&1) || lisp_code=$?

    local n_gen="" lisp_pass="" lisp_fail=""
    n_gen=$(echo "$lisp_out"     | grep -oP '生成成功用例:\s*\K[0-9]+'        | tail -1 || true)
    lisp_pass=$(echo "$lisp_out" | grep -oP 'Lisp 自检：PASS=\K[0-9]+'        | tail -1 || true)
    lisp_fail=$(echo "$lisp_out" | grep -oP 'Lisp 自检：PASS=[0-9]+ FAIL=\K[0-9]+' | tail -1 || true)

    if [[ $lisp_code -ne 0 ]]; then
        fail "阶段 1/2 失败 (exit=$lisp_code): 生成 ${n_gen:-?} 个用例, Lisp 自检 PASS=${lisp_pass:-?} FAIL=${lisp_fail:-?}"
        echo "$lisp_out" | grep -E "FAIL|错误|Error" | head -15
        warn "阶段 1 输出末尾 20 行（含 Lisp 报错/回溯）："
        echo "$lisp_out" | tail -20 | sed 's/^/    /'
        rc=1
    else
        ok "阶段 1/2 通过: 生成 ${n_gen:-?} 个用例, Lisp 自检 ${lisp_pass:-?} 通过 / ${lisp_fail:-0} 失败"
    fi

    if [[ $rc -eq 0 ]]; then
        log "阶段 2/2: NumPy 逐位校验 (shape / dtype / 数值)"
        log "  阶段 2 文件: $py_file"
        local py_out="" py_code=0
        py_out=$(CLVT_TEST_OUT="$out_dir" python3 "$py_file" 2>&1) || py_code=$?

        local py_total="" py_pass="" py_fail="" py_dev="" py_missing=""
        py_total=$(echo "$py_out"   | grep -oP '总计:\s*\K[0-9]+'                    | tail -1 || true)
        py_pass=$(echo "$py_out"    | grep -oP '通过:\s*\K[0-9]+'                    | tail -1 || true)
        py_fail=$(echo "$py_out"    | grep -oP '失败:\s*\K[0-9]+'                    | tail -1 || true)
        py_dev=$(echo "$py_out"     | grep -oP '已知差异:\s*\K[0-9]+'                | tail -1 || true)
        py_missing=$(echo "$py_out" | grep -oP '缺参考:\s*\K[0-9]+'                  | tail -1 || true)

        if [[ $py_code -eq 0 ]]; then
            ok "阶段 2/2 通过: ${py_pass:-?}/${py_total:-?} 逐位一致, 失败 ${py_fail:-0}, 已知差异 ${py_dev:-0}, 缺参考 ${py_missing:-0}"
        else
            fail "阶段 2/2 失败 (exit=$py_code): ${py_pass:-?}/${py_total:-?} 一致, 失败 ${py_fail:-?}, 缺参考 ${py_missing:-?}"
            echo "$py_out" | grep -E "^\[FAIL|^ *\[FAIL" | head -15
            rc=1
        fi
    fi

    end_time=$(date +%s)

    if [[ $own_out -eq 1 ]]; then
        rm -rf "$out_dir"
    else
        log "保留输出目录（CLVT_TEST_OUT=$out_dir）"
    fi

    if [[ $rc -eq 0 ]]; then
        ok "$name: 用例 ${n_gen:-?} 个 + Lisp 自检 ${lisp_pass:-?} 项 + NumPy 逐位校验 ${py_pass:-?}/${py_total:-?}, $((end_time - start_time))s"
    else
        fail "$name: 失败, $((end_time - start_time))s"
    fi
    return $rc
}

# 套件调度：流水线套件走 run_pipeline_suite，其余走 run_lisp_suite
run_suite() {
    if [[ -n "${PIPELINE_SUITES[$1]:-}" ]]; then
        run_pipeline_suite "$1"
    else
        run_lisp_suite "$1"
    fi
}

# 运行单个 lisp 测试套件（单文件形态）
run_lisp_suite() {
    local name="$1"
    local file="${SCRIPT_DIR}/${name}.lisp"
    local desc="${SUITES[$name]:-未知测试}"

    if [[ ! -f "$file" ]]; then
        fail "测试文件不存在: $file"
        return 1
    fi

    header "$name — $desc"
    log "文件: $file"
    log "运行中..."

    local start_time
    start_time=$(date +%s)

    local output
    local exit_code=0
    output=$("$SBCL" \
        --dynamic-space-size "$SBCL_DYNAMIC_SPACE_SIZE" \
        --noinform --non-interactive \
        --eval "(setf *read-default-float-format* 'single-float)" \
        --eval "(require :asdf)" \
        --eval "(push #p\"${PROJECT_DIR}/\" asdf:*central-registry*)" \
        --eval "(asdf:load-system :clvt)" \
        --eval "(load \"$file\")" \
        2>&1) || exit_code=$?

    local end_time
    end_time=$(date +%s)
    local duration=$((end_time - start_time))

    # 提取测试结果
    # 兼容多种输出格式：
    #   1) Total: N / Pass: N / Fail: N   (大多数套件)
    #   2) 通过: N  失败: M               (out-contract-v2-test)
    #   3) 汇总：通过 N / 失败 M          (nan-broadcast-test)
    local total pass fail_count skip
    total=$(echo "$output" | grep -oP 'Total:\s*\K[0-9]+' | tail -1 || echo "?")
    pass=$(echo "$output" | grep -oP 'Pass:\s*\K[0-9]+' | tail -1 || echo "?")
    fail_count=$(echo "$output" | grep -oP 'Fail:\s*\K[0-9]+' | tail -1 || echo "?")
    # 格式 2/3 兜底：通过 N（可带全角冒号） / 失败 M
    if [[ "$pass" == "?" ]]; then
        pass=$(echo "$output" | grep -oP '通过[：:]?\s*\K[0-9]+' | tail -1 || echo "?")
    fi
    if [[ "$fail_count" == "?" ]]; then
        fail_count=$(echo "$output" | grep -oP '失败[：:]?\s*\K[0-9]+' | tail -1 || echo "?")
    fi
    # 格式 4：PASS=N / FAIL=M（core-test 测试汇总 / differential-probes-test 差分探针汇总）
    if [[ "$pass" == "?" ]]; then
        pass=$(echo "$output" | grep -oP 'PASS=\K[0-9]+' | tail -1 || echo "?")
    fi
    if [[ "$fail_count" == "?" ]]; then
        fail_count=$(echo "$output" | grep -oP 'FAIL=\K[0-9]+' | tail -1 || echo "?")
    fi
    # 由 pass/fail 反推 total
    if [[ "$total" == "?" && "$pass" != "?" && "$fail_count" != "?" ]]; then
        total=$((pass + fail_count))
    fi
    skip=$(echo "$output" | grep -oP 'Skip:\s*\K[0-9]+' | tail -1 || echo "0")

    # 判断结果
    if [[ $exit_code -eq 0 ]] && [[ "$fail_count" == "0" || "$fail_count" == "?" ]]; then
        if [[ "$total" != "?" ]]; then
            ok "$name: $pass/$total 通过 (${duration}s)"
        else
            ok "$name: 完成 (${duration}s)"
        fi
        return 0
    else
        fail "$name: $pass/$total 通过, $fail_count 失败 (exit=$exit_code, ${duration}s)"

        # 显示失败详情
        if [[ -n "$VERBOSE" ]]; then
            echo "$output" | grep -E "❌|FAIL|Failed|error" | head -20
        else
            echo "$output" | grep -E "❌|Failed" | head -10
        fi
        # 计数完全解析不出（通常意味着进程异常退出/被信号终止）时，直接给出
        # 原始输出尾部——否则失败会变成一句无法追查的 "?/?"。
        if [[ "$total" == "?" && "$pass" == "?" && "$fail_count" == "?" ]]; then
            warn "无法从输出解析测试计数（exit=$exit_code），以下为输出末尾 20 行："
            echo "$output" | tail -20 | sed 's/^/    /'
        fi
        return 1
    fi
}

# ============================================================
# 主流程
# ============================================================

main() {
    local suites_to_run=()
    local run_benchmarks=false
    local specific_suite=""

    # 解析参数
    while [[ $# -gt 0 ]]; do
        case $1 in
            --suite)
                specific_suite="$2"
                shift 2
                ;;
            --list)
                header "可用测试套件（共 $((${#DEFAULT_ORDER[@]} + ${#BENCH_SUITES[@]})) 个）"
                echo -e "  ${BOLD}[默认执行]${NC}"
                for name in "${DEFAULT_ORDER[@]}"; do
                    local mark=""
                    [[ -n "${PIPELINE_SUITES[$name]:-}" ]] && mark=" ${CYAN}[两阶段: Lisp+NumPy]${NC}"
                    echo -e "    $name — ${SUITES[$name]:-未知}$mark"
                done
                echo -e "  ${BOLD}[性能基准，默认跳过；--all 执行]${NC}"
                for name in "${BENCH_SUITES[@]}"; do
                    echo "    $name — ${SUITES[$name]:-未知}"
                done
                exit 0
                ;;
            --quick)
                run_benchmarks=false
                shift
                ;;
            --all)
                run_benchmarks=true
                shift
                ;;
            --verbose)
                VERBOSE=1
                shift
                ;;
            --help|-h)
                usage
                exit 0
                ;;
            *)
                warn "未知选项: $1"
                usage
                exit 1
                ;;
        esac
    done

    # 检查环境
    check_sbcl
    PY_AVAILABLE=0
    check_python && PY_AVAILABLE=1 || true

    # 防漂移门禁（静态，秒级）：ref_compute key 约定 + 探针 codegen 同步。
    # 历史上 key 漂移曾两次发生（gen_all_tests/gen_param_tests 旧架构遗留），
    # 该检查确保任何漂移在跑套件前即被拦截并给出明确报因。
    if command -v python3 &>/dev/null; then
        if ! python3 "$SCRIPT_DIR/check_key_sync.py"; then
            fail "测试工具链 key 同步检查未通过（先修复漂移再运行套件，详见上方输出）"
            exit 1
        fi
    fi

    # 运行列表自检：每个已注册套件都必须被接线（防「写了却没人跑」的漏接线）
    if ! check_suite_wiring; then
        fail "套件接线自检未通过（见上方输出）"
        exit 1
    fi

    header "clvt 自动化测试"
    log "项目目录: $PROJECT_DIR"
    log "测试目录: $SCRIPT_DIR"
    log "时间: $(date '+%Y-%m-%d %H:%M:%S')"

    # 确定要运行的测试
    if [[ -n "$specific_suite" ]]; then
        if [[ -z "${SUITES[$specific_suite]+x}" ]]; then
            fail "未知测试套件: $specific_suite"
            echo "可用套件: ${!SUITES[*]}"
            exit 1
        fi
        suites_to_run=("$specific_suite")
    else
        # 唯一入口：按 DEFAULT_ORDER 顺序运行；--all 追加性能基准套件
        suites_to_run=("${DEFAULT_ORDER[@]}")
        if [[ "$run_benchmarks" == true ]]; then
            suites_to_run+=("${BENCH_SUITES[@]}")
        fi
    fi

    # python/numpy 不可用时移除依赖 NumPy 的套件
    if [[ "$PY_AVAILABLE" == "0" ]]; then
        local filtered=()
        for s in "${suites_to_run[@]}"; do
            case "$s" in
                numpy-compare-test|manip-interop-test) ;;
                *) filtered+=("$s") ;;
            esac
        done
        suites_to_run=("${filtered[@]}")
    fi

    log "将运行 ${#suites_to_run[@]} 个测试套件"
    echo ""

    # 运行测试
    local total_suites=0
    local passed_suites=0
    local failed_suites=0
    local failed_names=()
    local overall_start
    overall_start=$(date +%s)

    for suite in "${suites_to_run[@]}"; do
        total_suites=$((total_suites + 1))
        if run_suite "$suite"; then
            passed_suites=$((passed_suites + 1))
        else
            failed_suites=$((failed_suites + 1))
            failed_names+=("$suite")
        fi
        echo ""
    done

    local overall_end
    overall_end=$(date +%s)
    local overall_duration=$((overall_end - overall_start))

    # 总结
    header "测试总结"
    echo ""
    echo -e "  运行套件: ${BOLD}$total_suites${NC}"
    echo -e "  通过:     ${GREEN}$passed_suites${NC}"
    echo -e "  失败:     ${RED}$failed_suites${NC}"
    echo -e "  总耗时:   ${BOLD}${overall_duration}s${NC}"
    echo ""

    if [[ $failed_suites -gt 0 ]]; then
        fail "失败的套件:"
        for name in "${failed_names[@]}"; do
            # 注意: set -u 下对不存在的 key 取值会报 unbound variable，必须给默认值
            echo -e "  ${RED}✗${NC} $name — ${SUITES[$name]:-未知测试}"
        done
        echo ""
        exit 1
    else
        ok "所有测试通过! ✅"
        exit 0
    fi
}

# ============================================================
# 入口
# ============================================================
VERBOSE="${VERBOSE:-}"
main "$@"
