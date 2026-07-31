#!/usr/bin/env bash
# ============================================================
# run-tests.sh — clvt 自动化测试运行器
# 用法: bash test/run-tests.sh [选项]
#   无参数: 运行所有测试
#   --suite NAME: 只运行指定测试套件
#   --list: 列出所有可用测试套件
#   --quick: 跳过性能基准测试
# ============================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
SBCL="${SBCL:-sbcl}"
TIMEOUT="${TIMEOUT:-300}"

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
SUITES[benchmark-copy]="性能基准测试"

# 排除列表 (默认跳过)
SKIP_BY_DEFAULT="benchmark-copy"

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
  无参数          运行所有测试 (跳过 benchmark)
  --suite NAME    只运行指定测试套件
  --list          列出所有可用测试套件
  --quick         跳过性能基准和慢速测试
  --all           运行所有测试包括 benchmark
  --verbose       显示完整 SBCL 输出
  --help          显示此帮助信息

环境变量:
  SBCL            SBCL 可执行文件路径 (默认: sbcl)
  TIMEOUT         每个测试超时秒数 (默认: 300)

示例:
  bash test/run-tests.sh
  bash test/run-tests.sh --suite run_all_tests
  bash test/run-tests.sh --quick
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

# 运行单个测试套件
run_suite() {
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
    output=$("$SBCL" --noinform --non-interactive \
        --eval "(require :asdf)" \
        --eval "(push #p\"${PROJECT_DIR}/\" asdf:*central-registry*)" \
        --eval "(asdf:load-system :clvt)" \
        --eval "(load \"$file\")" \
        2>&1) || exit_code=$?

    local end_time
    end_time=$(date +%s)
    local duration=$((end_time - start_time))

    # 提取测试结果
    local total pass fail_count skip
    total=$(echo "$output" | grep -oP 'Total:\s*\K[0-9]+' | tail -1 || echo "?")
    pass=$(echo "$output" | grep -oP 'Pass:\s*\K[0-9]+' | tail -1 || echo "?")
    fail_count=$(echo "$output" | grep -oP 'Fail:\s*\K[0-9]+' | tail -1 || echo "?")
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
        fail "$name: $pass/$total 通过, $fail_count 失败 (${duration}s)"

        # 显示失败详情
        if [[ -n "$VERBOSE" ]]; then
            echo "$output" | grep -E "❌|FAIL|Failed|error" | head -20
        else
            echo "$output" | grep -E "❌|Failed" | head -10
        fi
        return 1
    fi
}

# ============================================================
# 主流程
# ============================================================

main() {
    local suites_to_run=()
    local skip_benchmark=true
    local specific_suite=""

    # 解析参数
    while [[ $# -gt 0 ]]; do
        case $1 in
            --suite)
                specific_suite="$2"
                shift 2
                ;;
            --list)
                header "可用测试套件"
                for name in "${!SUITES[@]}"; do
                    echo "  $name — ${SUITES[$name]}"
                done
                exit 0
                ;;
            --quick)
                skip_benchmark=true
                shift
                ;;
            --all)
                skip_benchmark=false
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
        # 按顺序运行所有测试
        suites_to_run=(
            run_all_tests
            run_param_tests
            nested-test
            robustness-test
            coverage-gap-test
            comprehensive-test
            auto-compare-test
        )
        if [[ "$skip_benchmark" == false ]]; then
            suites_to_run+=(benchmark-copy)
        fi
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
            echo -e "  ${RED}✗${NC} $name — ${SUITES[$name]}"
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
