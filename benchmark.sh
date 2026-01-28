#!/bin/bash
#
# Benchmark Script for Energy Storms Parallel Implementations
# ============================================================
# Compares Sequential, MPI+OpenMP, and CUDA implementations
#
# Usage: ./benchmark.sh [OPTIONS]
#
# Author: Benchmark Suite for Multicore Programming 2025/2026
#

set -e

# ============================================================================
# CONFIGURATION
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
TEST_DIR="${SCRIPT_DIR}/test_files"
RESULTS_DIR="${SCRIPT_DIR}/benchmark_results"
GROUND_TRUTH_FILE="${RESULTS_DIR}/ground_truth.txt"

# Default settings
NUM_RUNS=5
WARMUP_RUNS=1
MPI_PROCS="1 2 4"
OMP_THREADS="1 2 4"
TIMEOUT_SECONDS=300

# Implementation flags
RUN_SEQ=false
RUN_MPI_OMP=false
RUN_CUDA=false

# Test suite selection
TEST_SUITES="small"  # small, medium, large, all, custom

# Colors for output (check if terminal supports colors)
if [ -t 1 ]; then
    RED='\033[0;31m'
    GREEN='\033[0;32m'
    YELLOW='\033[1;33m'
    BLUE='\033[0;34m'
    CYAN='\033[0;36m'
    BOLD='\033[1m'
    NC='\033[0m'
else
    RED=''
    GREEN=''
    YELLOW=''
    BLUE=''
    CYAN=''
    BOLD=''
    NC=''
fi

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

print_header() {
    printf "\n${BOLD}${BLUE}═══════════════════════════════════════════════════════════════════${NC}\n"
    printf "${BOLD}${BLUE}  %s${NC}\n" "$1"
    printf "${BOLD}${BLUE}═══════════════════════════════════════════════════════════════════${NC}\n\n"
}

print_section() {
    printf "\n${CYAN}───────────────────────────────────────────────────────────────────${NC}\n"
    printf "${CYAN}  %s${NC}\n" "$1"
    printf "${CYAN}───────────────────────────────────────────────────────────────────${NC}\n"
}

print_success() {
    printf "${GREEN}✓ %s${NC}\n" "$1"
}

print_warning() {
    printf "${YELLOW}⚠ %s${NC}\n" "$1"
}

print_error() {
    printf "${RED}✗ %s${NC}\n" "$1"
}

print_info() {
    printf "${BLUE}ℹ %s${NC}\n" "$1"
}

usage() {
    cat << EOF
${BOLD}Energy Storms Benchmark Suite${NC}

${BOLD}USAGE:${NC}
    ./benchmark.sh [OPTIONS]

${BOLD}IMPLEMENTATION SELECTION:${NC}
    -s, --seq           Include sequential implementation
    -m, --mpi-omp       Include MPI+OpenMP implementation
    -c, --cuda          Include CUDA implementation
    -a, --all           Benchmark all implementations

${BOLD}CONFIGURATION:${NC}
    -n, --runs N        Number of benchmark runs (default: ${NUM_RUNS})
    -w, --warmup N      Number of warmup runs (default: ${WARMUP_RUNS})
    -p, --procs LIST    MPI process counts, comma-separated (default: "1,2,4")
    -t, --threads LIST  OpenMP thread counts, comma-separated (default: "1,2,4")
    --timeout N         Timeout in seconds per run (default: ${TIMEOUT_SECONDS})

${BOLD}TEST SELECTION:${NC}
    --small             Run small test cases (test_01, test_03-06, test_09, medium_s)
    --medium            Run medium test cases (test_02)
    --large             Run large test cases (test_07, test_08)
    --all-tests         Run all test cases
    --test NAME         Run specific test (e.g., test_01, test_07)

${BOLD}OUTPUT:${NC}
    -o, --output DIR    Output directory (default: benchmark_results)
    --csv               Also generate CSV output for analysis
    --no-compile        Skip compilation step

${BOLD}CORRECTNESS:${NC}
    --update-truth      Run sequential and update ground truth file
    --verify-only       Only verify correctness (1 run, no timing stats)

${BOLD}EXAMPLES:${NC}
    # Generate ground truth (run once after code changes)
    ./benchmark.sh -s --update-truth --all-tests
    
    # Benchmark MPI+OMP using stored ground truth for verification
    ./benchmark.sh -m --medium -n 10 -p 1,2,4 -t 1,2,4
    
    # Quick correctness check (1 run, no warmup)
    ./benchmark.sh -m --verify-only --all-tests
    
    # Full benchmark with all implementations
    ./benchmark.sh -s -m --small -n 10 --csv
    
    # Full benchmark suite with CSV export
    ./benchmark.sh --all --all-tests -n 5 --csv

EOF
    exit 0
}

# ============================================================================
# ARGUMENT PARSING
# ============================================================================

GENERATE_CSV=false
SKIP_COMPILE=false
SPECIFIC_TEST=""
UPDATE_GROUND_TRUTH=false
VERIFY_ONLY=false

# Correctness tracking
TOTAL_TESTS=0
PASSED_TESTS=0
FAILED_TESTS=0
declare -a FAILED_TEST_NAMES

while [ $# -gt 0 ]; do
    case $1 in
        -s|--seq)
            RUN_SEQ=true
            shift
            ;;
        -m|--mpi-omp)
            RUN_MPI_OMP=true
            shift
            ;;
        -c|--cuda)
            RUN_CUDA=true
            shift
            ;;
        -a|--all)
            RUN_SEQ=true
            RUN_MPI_OMP=true
            RUN_CUDA=true
            shift
            ;;
        -n|--runs)
            NUM_RUNS="$2"
            shift 2
            ;;
        -w|--warmup)
            WARMUP_RUNS="$2"
            shift 2
            ;;
        -p|--procs)
            MPI_PROCS=$(echo "$2" | tr ',' ' ')
            shift 2
            ;;
        -t|--threads)
            OMP_THREADS=$(echo "$2" | tr ',' ' ')
            shift 2
            ;;
        --timeout)
            TIMEOUT_SECONDS="$2"
            shift 2
            ;;
        --small)
            TEST_SUITES="small"
            shift
            ;;
        --medium)
            TEST_SUITES="medium"
            shift
            ;;
        --large)
            TEST_SUITES="large"
            shift
            ;;
        --all-tests)
            TEST_SUITES="all"
            shift
            ;;
        --test)
            SPECIFIC_TEST="$2"
            TEST_SUITES="custom"
            shift 2
            ;;
        -o|--output)
            RESULTS_DIR="$2"
            shift 2
            ;;
        --csv)
            GENERATE_CSV=true
            shift
            ;;
        --no-compile)
            SKIP_COMPILE=true
            shift
            ;;
        --update-truth)
            UPDATE_GROUND_TRUTH=true
            RUN_SEQ=true
            shift
            ;;
        --verify-only)
            VERIFY_ONLY=true
            NUM_RUNS=1
            WARMUP_RUNS=0
            shift
            ;;
        -h|--help)
            usage
            ;;
        *)
            print_error "Unknown option: $1"
            usage
            ;;
    esac
done

# Validate at least one implementation selected
if ! $RUN_SEQ && ! $RUN_MPI_OMP && ! $RUN_CUDA; then
    print_error "No implementation selected. Use -s, -m, -c, or -a"
    usage
fi

# ============================================================================
# TEST CASE DEFINITIONS (using function instead of associative array)
# ============================================================================

get_test_args() {
    local test_name="$1"
    case "$test_name" in
        # Small tests
        test_01)
            echo "35 test_files/test_01_a35_p5_w3 test_files/test_01_a35_p7_w2 test_files/test_01_a35_p8_w1 test_files/test_01_a35_p8_w4"
            ;;
        test_03)
            echo "20 test_files/test_03_a20_p4_w1"
            ;;
        test_04)
            echo "20 test_files/test_04_a20_p4_w1"
            ;;
        test_05)
            echo "20 test_files/test_05_a20_p4_w1"
            ;;
        test_06)
            echo "20 test_files/test_06_a20_p4_w1"
            ;;
        test_09_16)
            echo "16 test_files/test_09_a16-17_p3_w1"
            ;;
        test_09_17)
            echo "17 test_files/test_09_a16-17_p3_w1"
            ;;
        # Small version of test_02 (only first 2 waves)
        medium_s)
            echo "20000 test_files/test_02_a30k_p20k_w1"
            ;;
        # Medium tests
        test_02)
            echo "20000 test_files/test_02_a30k_p20k_w1 test_files/test_02_a30k_p20k_w2 test_files/test_02_a30k_p20k_w3 test_files/test_02_a30k_p20k_w4 test_files/test_02_a30k_p20k_w5 test_files/test_02_a30k_p20k_w6"
            ;;
        # Large tests
        test_07)
            echo "1000000 test_files/test_07_a1M_p5k_w1 test_files/test_07_a1M_p5k_w2 test_files/test_07_a1M_p5k_w3 test_files/test_07_a1M_p5k_w4"
            ;;
        test_08)
            echo "100000000 test_files/test_08_a100M_p1_w1 test_files/test_08_a100M_p1_w2 test_files/test_08_a100M_p1_w3"
            ;;
        *)
            echo ""
            ;;
    esac
}

# Test suite groupings
SMALL_TESTS="test_01 test_03 test_04 test_05 test_06 test_09_16 test_09_17 medium_s"
MEDIUM_TESTS="test_02"
LARGE_TESTS="test_07 test_08"

get_test_list() {
    case "$TEST_SUITES" in
        small)
            echo "$SMALL_TESTS"
            ;;
        medium)
            echo "$MEDIUM_TESTS"
            ;;
        large)
            echo "$LARGE_TESTS"
            ;;
        all)
            echo "$SMALL_TESTS $MEDIUM_TESTS $LARGE_TESTS"
            ;;
        custom)
            echo "$SPECIFIC_TEST"
            ;;
    esac
}

# ============================================================================
# GROUND TRUTH FUNCTIONS
# ============================================================================

# Save ground truth result for a test
save_ground_truth() {
    local test_name="$1"
    local result="$2"
    
    mkdir -p "$RESULTS_DIR"
    
    # Remove old entry if exists
    if [ -f "$GROUND_TRUTH_FILE" ]; then
        grep -v "^${test_name}|" "$GROUND_TRUTH_FILE" > "${GROUND_TRUTH_FILE}.tmp" 2>/dev/null || true
        mv "${GROUND_TRUTH_FILE}.tmp" "$GROUND_TRUTH_FILE"
    fi
    
    # Add new entry
    echo "${test_name}|${result}" >> "$GROUND_TRUTH_FILE"
    print_success "Ground truth saved for $test_name"
}

# Load ground truth result for a test
load_ground_truth() {
    local test_name="$1"
    
    if [ ! -f "$GROUND_TRUTH_FILE" ]; then
        echo ""
        return
    fi
    
    grep "^${test_name}|" "$GROUND_TRUTH_FILE" 2>/dev/null | cut -d'|' -f2- || echo ""
}

# Check if ground truth exists for a test
has_ground_truth() {
    local test_name="$1"
    
    if [ ! -f "$GROUND_TRUTH_FILE" ]; then
        return 1
    fi
    
    grep -q "^${test_name}|" "$GROUND_TRUTH_FILE" 2>/dev/null
}

# Show ground truth status
show_ground_truth_status() {
    print_section "Ground Truth Status"
    
    if [ ! -f "$GROUND_TRUTH_FILE" ]; then
        print_warning "No ground truth file found. Run with --update-truth -s to generate."
        return
    fi
    
    echo "  Ground truth file: $GROUND_TRUTH_FILE"
    echo "  Last updated: $(stat -f '%Sm' "$GROUND_TRUTH_FILE" 2>/dev/null || stat -c '%y' "$GROUND_TRUTH_FILE" 2>/dev/null || echo 'Unknown')"
    echo ""
    echo "  Available ground truth:"
    
    local tests=$(get_test_list)
    for test_name in $tests; do
        if has_ground_truth "$test_name"; then
            printf "    ${GREEN}✓${NC} %s\n" "$test_name"
        else
            printf "    ${RED}✗${NC} %s (missing)\n" "$test_name"
        fi
    done
}

# ============================================================================
# STATISTICS FUNCTIONS
# ============================================================================

calculate_stats() {
    # Read times from stdin, one per line
    local times_file=$(mktemp)
    cat > "$times_file"
    
    local n=$(wc -l < "$times_file" | tr -d ' ')
    
    if [ "$n" -eq 0 ]; then
        echo "0 0 0 0 0"
        rm -f "$times_file"
        return
    fi
    
    # Calculate using awk for portability
    awk '
    BEGIN { sum = 0; min = 999999; max = 0; n = 0 }
    {
        sum += $1
        if ($1 < min) min = $1
        if ($1 > max) max = $1
        times[n] = $1
        n++
    }
    END {
        if (n == 0) {
            print "0 0 0 0 0"
            exit
        }
        mean = sum / n
        
        # Calculate stddev
        sq_diff_sum = 0
        for (i = 0; i < n; i++) {
            diff = times[i] - mean
            sq_diff_sum += diff * diff
        }
        variance = sq_diff_sum / n
        stddev = sqrt(variance)
        
        printf "%.6f %.6f %.6f %.6f %d\n", mean, stddev, min, max, n
    }
    ' "$times_file"
    
    rm -f "$times_file"
}

# ============================================================================
# BENCHMARK RUNNER
# ============================================================================

# Global variables for result passing
LAST_RESULT=""
LAST_TIME=""

run_benchmark() {
    local impl="$1"
    local test_name="$2"
    local config="$3"  # For MPI: "procs_threads", for others: empty
    
    local test_args=$(get_test_args "$test_name")
    if [ -z "$test_args" ]; then
        print_error "Unknown test case: $test_name"
        return 1
    fi
    
    local times_file=$(mktemp)
    local results_file=$(mktemp)
    local failed=0
    
    # Determine executable and command
    local cmd=""
    local exe=""
    local config_desc=""
    
    case "$impl" in
        seq)
            exe="./energy_storms_seq"
            cmd="$exe $test_args"
            config_desc="Sequential"
            ;;
        mpi_omp)
            local procs=$(echo "$config" | cut -d'_' -f1)
            local threads=$(echo "$config" | cut -d'_' -f2)
            exe="./energy_storms_mpi_omp"
            cmd="mpirun --oversubscribe -np $procs $exe $test_args"
            config_desc="MPI+OMP (P=$procs, T=$threads)"
            export OMP_NUM_THREADS=$threads
            ;;
        cuda)
            exe="./energy_storms_cuda"
            cmd="$exe $test_args"
            config_desc="CUDA"
            ;;
    esac
    
    # Check if executable exists
    if [ ! -x "$exe" ]; then
        print_error "Executable not found: $exe"
        rm -f "$times_file" "$results_file"
        return 1
    fi
    
    printf "  Running %s on %s: " "$config_desc" "$test_name"
    
    # Warmup runs
    local i
    for i in $(seq 1 $WARMUP_RUNS); do
        timeout "${TIMEOUT_SECONDS}s" $cmd > /dev/null 2>&1 || true
    done
    
    # Benchmark runs
    for i in $(seq 1 $NUM_RUNS); do
        printf "."
        
        local output
        # Capture stdout and stderr separately, ignore exit code for MPI (may complain about MPI_Finalize)
        output=$(timeout "${TIMEOUT_SECONDS}s" $cmd 2>/dev/null) || true
        
        # If no output, try again capturing stderr too (some implementations print to stderr)
        if [ -z "$output" ]; then
            output=$(timeout "${TIMEOUT_SECONDS}s" $cmd 2>&1) || true
        fi
        
        # Extract time from output
        local time=$(echo "$output" | grep "^Time:" | awk '{print $2}')
        local result=$(echo "$output" | grep "^Result:" | cut -d':' -f2-)
        
        if [ -n "$time" ]; then
            echo "$time" >> "$times_file"
            echo "$result" >> "$results_file"
        else
            failed=$((failed + 1))
        fi
    done
    
    echo ""
    
    # Calculate statistics
    local successful_runs=$(wc -l < "$times_file" | tr -d ' ')
    
    if [ "$successful_runs" -gt 0 ]; then
        local stats=$(calculate_stats < "$times_file")
        local mean=$(echo "$stats" | awk '{print $1}')
        local stddev=$(echo "$stats" | awk '{print $2}')
        local min=$(echo "$stats" | awk '{print $3}')
        local max=$(echo "$stats" | awk '{print $4}')
        local n=$(echo "$stats" | awk '{print $5}')
        
        printf "    ${GREEN}Mean: %.6f s${NC} (±%.6f) | Min: %.6f | Max: %.6f | Runs: %d/%d\n" \
            "$mean" "$stddev" "$min" "$max" "$n" "$NUM_RUNS"
        
        # Get first result for verification
        local first_result=$(head -1 "$results_file")
        
        # Store result in CSV
        echo "$impl,$test_name,$config,$mean,$stddev,$min,$max,$n,$first_result" >> "$CURRENT_RESULTS_FILE"
        
        # Set globals for verification
        LAST_RESULT="$first_result"
        LAST_TIME="$mean"
    else
        print_error "    All runs failed!"
        echo "$impl,$test_name,$config,FAILED,,,,,," >> "$CURRENT_RESULTS_FILE"
        LAST_RESULT=""
        LAST_TIME=""
    fi
    
    rm -f "$times_file" "$results_file"
}

verify_results() {
    local baseline_result="$1"
    local test_result="$2"
    
    # Normalize results for comparison (trim and squeeze whitespace)
    # (Avoid `xargs` here for maximum portability and to prevent noisy failures.)
    baseline_result=$(printf "%s\n" "$baseline_result" | awk '{$1=$1; print}')
    test_result=$(printf "%s\n" "$test_result" | awk '{$1=$1; print}')
    
    # Allow small floating-point deviations in maxima while keeping positions exact.
    # Result format is expected to be: "<pos1> <max1> <pos2> <max2> ..."
    # Tolerance: absolute error up to 1e-3 on the float fields.
    local tol="0.001"
    
    awk -v tol="$tol" -v a="$baseline_result" -v b="$test_result" '
        function isnum(x) {
            return (x ~ /^[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?$/)
        }
        BEGIN {
            gsub(/[[:space:]]+/, " ", a)
            gsub(/[[:space:]]+/, " ", b)
            
            n = split(a, A, " ")
            m = split(b, B, " ")
            
            if (n != m) exit 1
            
            for (i = 1; i <= n; i++) {
                # Odd fields: positions (must match exactly)
                if (i % 2 == 1) {
                    if (A[i] != B[i]) exit 1
                } else {
                    # Even fields: maxima (compare numerically within tolerance)
                    if (!isnum(A[i]) || !isnum(B[i])) {
                        if (A[i] != B[i]) exit 1
                        continue
                    }
                    diff = A[i] - B[i]
                    if (diff < 0) diff = -diff
                    if (diff > tol) exit 1
                }
            }
            exit 0
        }
    ' && return 0 || return 1
}

# ============================================================================
# COMPILATION
# ============================================================================

compile_implementations() {
    print_section "Compiling Implementations"
    
    cd "$SCRIPT_DIR"
    
    # Clean first
    make clean > /dev/null 2>&1 || true
    
    local compile_failed=false
    
    if $RUN_SEQ; then
        printf "  Compiling sequential... "
        if make energy_storms_seq > /dev/null 2>&1; then
            print_success "OK"
        else
            print_error "FAILED"
            compile_failed=true
        fi
    fi
    
    if $RUN_MPI_OMP; then
        printf "  Compiling MPI+OpenMP... "
        if make energy_storms_mpi_omp > /dev/null 2>&1; then
            print_success "OK"
        else
            print_error "FAILED"
            compile_failed=true
        fi
    fi
    
    if $RUN_CUDA; then
        printf "  Compiling CUDA... "
        if make energy_storms_cuda > /dev/null 2>&1; then
            print_success "OK"
        else
            print_error "FAILED (CUDA may not be available)"
            RUN_CUDA=false
        fi
    fi
    
    if $compile_failed; then
        print_error "Some compilations failed. Check your environment."
        exit 1
    fi
}

# ============================================================================
# MAIN BENCHMARK LOOP
# ============================================================================

run_benchmarks() {
    print_section "Running Benchmarks"
    
    local tests=$(get_test_list)
    
    for test_name in $tests; do
        echo ""
        print_info "Test Case: $test_name"
        echo "  Args: $(get_test_args "$test_name")"
        echo ""
        
        local baseline_result=""
        
        # Try to load ground truth first
        baseline_result=$(load_ground_truth "$test_name")
        if [ -n "$baseline_result" ]; then
            echo "  Ground truth loaded from file"
        fi
        
        # Sequential baseline
        if $RUN_SEQ; then
            run_benchmark "seq" "$test_name" ""
            
            # If we got a result, use it as baseline
            if [ -n "$LAST_RESULT" ]; then
                # Save to ground truth if updating
                if $UPDATE_GROUND_TRUTH; then
                    save_ground_truth "$test_name" "$LAST_RESULT"
                fi
                
                # Use fresh sequential result as baseline (overrides loaded truth)
                baseline_result="$LAST_RESULT"
            fi
        fi
        
        # Check if we have baseline for verification
        if [ -z "$baseline_result" ]; then
            print_warning "No ground truth available for $test_name - run with -s or --update-truth first"
        fi
        
        # MPI+OpenMP variants
        if $RUN_MPI_OMP; then
            for procs in $MPI_PROCS; do
                for threads in $OMP_THREADS; do
                    run_benchmark "mpi_omp" "$test_name" "${procs}_${threads}"
                    
                    # Verify against baseline if available
                    if [ -n "$baseline_result" ] && [ -n "$LAST_RESULT" ]; then
                        TOTAL_TESTS=$((TOTAL_TESTS + 1))
                        if verify_results "$baseline_result" "$LAST_RESULT"; then
                            PASSED_TESTS=$((PASSED_TESTS + 1))
                            printf "    ${GREEN}✓ CORRECT${NC} - Result matches ground truth\n"
                        else
                            FAILED_TESTS=$((FAILED_TESTS + 1))
                            FAILED_TEST_NAMES+=("${test_name}:mpi_omp:P${procs}_T${threads}")
                            printf "    ${RED}✗ INCORRECT${NC} - Result differs from ground truth!\n"
                            echo "      Expected:$baseline_result"
                            echo "      Got:     $LAST_RESULT"
                        fi
                    elif [ -n "$LAST_RESULT" ]; then
                        print_warning "    Cannot verify - no ground truth"
                    fi
                done
            done
        fi
        
        # CUDA
        if $RUN_CUDA; then
            run_benchmark "cuda" "$test_name" ""
            
            if [ -n "$baseline_result" ] && [ -n "$LAST_RESULT" ]; then
                TOTAL_TESTS=$((TOTAL_TESTS + 1))
                if verify_results "$baseline_result" "$LAST_RESULT"; then
                    PASSED_TESTS=$((PASSED_TESTS + 1))
                    printf "    ${GREEN}✓ CORRECT${NC} - Result matches ground truth\n"
                else
                    FAILED_TESTS=$((FAILED_TESTS + 1))
                    FAILED_TEST_NAMES+=("${test_name}:cuda")
                    printf "    ${RED}✗ INCORRECT${NC} - Result differs from ground truth!\n"
                    echo "      Expected:$baseline_result"
                    echo "      Got:     $LAST_RESULT"
                fi
            elif [ -n "$LAST_RESULT" ]; then
                print_warning "    Cannot verify - no ground truth"
            fi
        fi
    done
}

# ============================================================================
# REPORT GENERATION
# ============================================================================

generate_report() {
    print_section "Generating Report"
    
    local report_file="${RESULTS_DIR}/benchmark_report_${TIMESTAMP}.txt"
    
    {
        echo "═══════════════════════════════════════════════════════════════════"
        echo "  ENERGY STORMS BENCHMARK REPORT"
        echo "  Generated: $(date)"
        echo "═══════════════════════════════════════════════════════════════════"
        echo ""
        echo "CONFIGURATION"
        echo "─────────────"
        echo "  Runs per test:    $NUM_RUNS"
        echo "  Warmup runs:      $WARMUP_RUNS"
        echo "  Test suite:       $TEST_SUITES"
        printf "  Implementations:  "
        $RUN_SEQ && printf "SEQ "
        $RUN_MPI_OMP && printf "MPI+OMP "
        $RUN_CUDA && printf "CUDA"
        echo ""
        if $RUN_MPI_OMP; then
            echo "  MPI processes:    $MPI_PROCS"
            echo "  OMP threads:      $OMP_THREADS"
        fi
        echo ""
        echo "SYSTEM INFO"
        echo "───────────"
        echo "  Hostname:         $(hostname)"
        echo "  OS:               $(uname -s) $(uname -r)"
        
        # Get CPU info (cross-platform)
        local cpu_info
        if [ "$(uname -s)" = "Darwin" ]; then
            cpu_info=$(sysctl -n machdep.cpu.brand_string 2>/dev/null || echo "Unknown")
            local cores=$(sysctl -n hw.ncpu 2>/dev/null || echo "Unknown")
        else
            cpu_info=$(cat /proc/cpuinfo 2>/dev/null | grep 'model name' | head -1 | cut -d':' -f2 | xargs || echo "Unknown")
            local cores=$(nproc 2>/dev/null || echo "Unknown")
        fi
        echo "  CPU:              $cpu_info"
        echo "  Cores:            $cores"
        echo ""
        echo "═══════════════════════════════════════════════════════════════════"
        echo "  RESULTS SUMMARY"
        echo "═══════════════════════════════════════════════════════════════════"
        echo ""
        printf "%-12s %-12s %-15s %12s %12s %12s %12s\n" \
            "IMPL" "TEST" "CONFIG" "MEAN(s)" "STDDEV" "MIN" "MAX"
        echo "─────────────────────────────────────────────────────────────────────────────────────────"
        
        # Parse CSV and format
        tail -n +2 "$CURRENT_RESULTS_FILE" | while IFS=',' read -r impl test config mean stddev min max n result; do
            if [ "$mean" != "FAILED" ]; then
                local config_str=""
                if [ "$impl" = "mpi_omp" ]; then
                    local p=$(echo "$config" | cut -d'_' -f1)
                    local t=$(echo "$config" | cut -d'_' -f2)
                    config_str="P=${p},T=${t}"
                else
                    config_str="-"
                fi
                printf "%-12s %-12s %-15s %12.6f %12.6f %12.6f %12.6f\n" \
                    "$impl" "$test" "$config_str" "$mean" "$stddev" "$min" "$max"
            else
                printf "%-12s %-12s %-15s %12s\n" "$impl" "$test" "-" "FAILED"
            fi
        done
        
        echo ""
        echo "═══════════════════════════════════════════════════════════════════"
        echo "  SPEEDUP ANALYSIS"
        echo "═══════════════════════════════════════════════════════════════════"
        echo ""
        
        # Calculate speedups relative to sequential
        if $RUN_SEQ; then
            echo "Speedup relative to sequential baseline:"
            echo ""
            printf "%-12s %-15s %12s %12s\n" "TEST" "CONFIG" "SEQ TIME" "SPEEDUP"
            echo "─────────────────────────────────────────────────────────────"
            
            # Get unique tests
            local tests=$(tail -n +2 "$CURRENT_RESULTS_FILE" | cut -d',' -f2 | sort -u)
            
            for test in $tests; do
                local seq_time=$(grep "^seq,$test," "$CURRENT_RESULTS_FILE" | cut -d',' -f4)
                
                if [ -n "$seq_time" ] && [ "$seq_time" != "FAILED" ]; then
                    # Print sequential baseline
                    printf "%-12s %-15s %12.6f %12s\n" "$test" "Sequential" "$seq_time" "1.00x"
                    
                    # Calculate speedups for parallel versions
                    grep "^mpi_omp,$test," "$CURRENT_RESULTS_FILE" 2>/dev/null | while IFS=',' read -r impl t config mean rest; do
                        if [ "$mean" != "FAILED" ]; then
                            local speedup=$(awk "BEGIN {printf \"%.2f\", $seq_time / $mean}")
                            local p=$(echo "$config" | cut -d'_' -f1)
                            local th=$(echo "$config" | cut -d'_' -f2)
                            printf "%-12s %-15s %12.6f %12sx\n" "" "P=$p,T=$th" "$mean" "$speedup"
                        fi
                    done
                    
                    grep "^cuda,$test," "$CURRENT_RESULTS_FILE" 2>/dev/null | while IFS=',' read -r impl t config mean rest; do
                        if [ "$mean" != "FAILED" ]; then
                            local speedup=$(awk "BEGIN {printf \"%.2f\", $seq_time / $mean}")
                            printf "%-12s %-15s %12.6f %12sx\n" "" "CUDA" "$mean" "$speedup"
                        fi
                    done
                    
                    echo ""
                fi
            done
        fi
        
        echo ""
        echo "═══════════════════════════════════════════════════════════════════"
        echo "  CORRECTNESS VERIFICATION"
        echo "═══════════════════════════════════════════════════════════════════"
        echo ""
        if [ "$TOTAL_TESTS" -gt 0 ]; then
            echo "  Total tests:  $TOTAL_TESTS"
            echo "  Passed:       $PASSED_TESTS"
            echo "  Failed:       $FAILED_TESTS"
            echo ""
            if [ "$FAILED_TESTS" -eq 0 ]; then
                echo "  Status:       ✓ ALL TESTS PASSED"
            else
                echo "  Status:       ✗ SOME TESTS FAILED"
                echo ""
                echo "  Failed tests:"
                for failed in "${FAILED_TEST_NAMES[@]}"; do
                    echo "    - $failed"
                done
            fi
        else
            echo "  No correctness tests performed."
            echo "  Run with -s to generate ground truth, or ensure ground_truth.txt exists."
        fi
        echo ""
        echo "═══════════════════════════════════════════════════════════════════"
        echo "  END OF REPORT"
        echo "═══════════════════════════════════════════════════════════════════"
        
    } > "$report_file"
    
    print_success "Report saved to: $report_file"
    
    # Generate CSV if requested
    if $GENERATE_CSV; then
        local csv_file="${RESULTS_DIR}/benchmark_data_${TIMESTAMP}.csv"
        cp "$CURRENT_RESULTS_FILE" "$csv_file"
        print_success "CSV data saved to: $csv_file"
    fi
    
    # Display summary
    echo ""
    cat "$report_file"
}

# ============================================================================
# MAIN
# ============================================================================

main() {
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    
    print_header "Energy Storms Benchmark Suite"
    
    echo "Configuration:"
    printf "  • Implementations: "
    $RUN_SEQ && printf "Sequential "
    $RUN_MPI_OMP && printf "MPI+OMP "
    $RUN_CUDA && printf "CUDA"
    echo ""
    echo "  • Test suite:      $TEST_SUITES"
    echo "  • Runs per test:   $NUM_RUNS (+ $WARMUP_RUNS warmup)"
    if $RUN_MPI_OMP; then
        echo "  • MPI processes:   $MPI_PROCS"
        echo "  • OMP threads:     $OMP_THREADS"
    fi
    
    # Create results directory
    mkdir -p "$RESULTS_DIR"
    CURRENT_RESULTS_FILE="${RESULTS_DIR}/raw_results_${TIMESTAMP}.csv"
    
    # Initialize CSV
    echo "implementation,test,config,mean,stddev,min,max,runs,result" > "$CURRENT_RESULTS_FILE"
    
    # Compile
    if ! $SKIP_COMPILE; then
        compile_implementations
    else
        print_warning "Skipping compilation (--no-compile)"
    fi
    
    # Show ground truth status if not running sequential
    if ! $RUN_SEQ && ($RUN_MPI_OMP || $RUN_CUDA); then
        show_ground_truth_status
    fi
    
    # Run benchmarks
    cd "$SCRIPT_DIR"
    run_benchmarks
    
    # Generate report
    generate_report
    
    # Print correctness summary
    print_section "Correctness Summary"
    if [ "$TOTAL_TESTS" -gt 0 ]; then
        echo ""
        if [ "$FAILED_TESTS" -eq 0 ]; then
            printf "  ${GREEN}${BOLD}✓ ALL TESTS PASSED${NC} (%d/%d)\n" "$PASSED_TESTS" "$TOTAL_TESTS"
        else
            printf "  ${RED}${BOLD}✗ TESTS FAILED${NC} (%d/%d passed)\n" "$PASSED_TESTS" "$TOTAL_TESTS"
            echo ""
            echo "  Failed tests:"
            for failed in "${FAILED_TEST_NAMES[@]}"; do
                printf "    ${RED}✗${NC} %s\n" "$failed"
            done
        fi
        echo ""
    else
        print_warning "No correctness verification performed"
        echo "  To verify correctness:"
        echo "    1. Run: ./benchmark.sh -s --update-truth [--small|--medium|--large|--all-tests]"
        echo "    2. Then: ./benchmark.sh -m [--small|--medium|--large|--all-tests]"
    fi
    
    print_header "Benchmark Complete"
    echo "Results saved in: $RESULTS_DIR"
    
    # Exit with error if tests failed
    if [ "$FAILED_TESTS" -gt 0 ]; then
        exit 1
    fi
}

main "$@"
