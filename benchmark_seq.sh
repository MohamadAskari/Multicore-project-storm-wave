#!/bin/bash
#
# Sequential Benchmark Script for Energy Storms
# ==============================================
# Runs all test cases for the sequential implementation and outputs CSV report.
#
# Usage: ./benchmark_seq.sh [OPTIONS]
#
# Options:
#   -n, --runs N     Number of runs per test (default: 10)
#   -o, --output FILE  Output CSV file (default: benchmark_seq_TIMESTAMP.csv)
#   -h, --help       Show this help message
#

set -e

# ============================================================================
# CONFIGURATION
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
TEST_DIR="${SCRIPT_DIR}/test_files"
RESULTS_DIR="${SCRIPT_DIR}/benchmark_results"

# Default settings
NUM_RUNS=10
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_FILE=""

# Colors for output
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
${BOLD}Energy Storms Sequential Benchmark${NC}

${BOLD}USAGE:${NC}
    ./benchmark_seq.sh [OPTIONS]

${BOLD}OPTIONS:${NC}
    -n, --runs N       Number of runs per test case (default: ${NUM_RUNS})
    -o, --output FILE  Output CSV file (default: benchmark_seq_TIMESTAMP.csv)
    -h, --help         Show this help message

${BOLD}EXAMPLES:${NC}
    # Run with default settings (10 runs per test)
    ./benchmark_seq.sh

    # Run with 20 runs per test
    ./benchmark_seq.sh -n 20

    # Specify output file
    ./benchmark_seq.sh -o my_results.csv

EOF
    exit 0
}

# ============================================================================
# ARGUMENT PARSING
# ============================================================================

while [ $# -gt 0 ]; do
    case $1 in
        -n|--runs)
            NUM_RUNS="$2"
            shift 2
            ;;
        -o|--output)
            OUTPUT_FILE="$2"
            shift 2
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

# Set default output file if not specified
if [ -z "$OUTPUT_FILE" ]; then
    mkdir -p "$RESULTS_DIR"
    OUTPUT_FILE="${RESULTS_DIR}/benchmark_seq_${TIMESTAMP}.csv"
fi

# ============================================================================
# TEST CASE DEFINITIONS
# ============================================================================

declare -a TEST_NAMES=(
    "test_1"
    "test_2"
    "test_3"
    "test_4"
    "test_5"
    "test_6"
    "test_7"
    "test_8"
    "test_9"
)

get_test_args() {
    local test_name="$1"
    case "$test_name" in
        test_1)
            echo "35 test_files/test_01_a35_p5_w3 test_files/test_01_a35_p7_w2 test_files/test_01_a35_p8_w1 test_files/test_01_a35_p8_w4"
            ;;
        test_2)
            echo "20000 test_files/test_02_a30k_p20k_w1 test_files/test_02_a30k_p20k_w2 test_files/test_02_a30k_p20k_w3 test_files/test_02_a30k_p20k_w4 test_files/test_02_a30k_p20k_w5 test_files/test_02_a30k_p20k_w6"
            ;;
        test_3)
            echo "20 test_files/test_03_a20_p4_w1"
            ;;
        test_4)
            echo "20 test_files/test_04_a20_p4_w1"
            ;;
        test_5)
            echo "20 test_files/test_05_a20_p4_w1"
            ;;
        test_6)
            echo "20 test_files/test_06_a20_p4_w1"
            ;;
        test_7)
            echo "1000000 test_files/test_07_a1M_p5k_w1 test_files/test_07_a1M_p5k_w2 test_files/test_07_a1M_p5k_w3 test_files/test_07_a1M_p5k_w4"
            ;;
        test_8)
            echo "100000000 test_files/test_08_a100M_p1_w1 test_files/test_08_a100M_p1_w2 test_files/test_08_a100M_p1_w3"
            ;;
        test_9)
            echo "17 test_files/test_09_a16-17_p3_w1"
            ;;
        *)
            echo ""
            ;;
    esac
}

# ============================================================================
# STATISTICS FUNCTIONS
# ============================================================================

calculate_average() {
    # Read times from stdin, one per line, and calculate average
    awk '
    BEGIN { sum = 0; n = 0 }
    {
        sum += $1
        n++
    }
    END {
        if (n > 0) printf "%.6f", sum / n
        else print "0"
    }
    '
}

calculate_stats() {
    # Read times from stdin, output: mean stddev min max n
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
    '
}

# ============================================================================
# BENCHMARK FUNCTIONS
# ============================================================================

run_single_test() {
    local test_name="$1"
    local test_args=$(get_test_args "$test_name")
    
    if [ -z "$test_args" ]; then
        print_error "Unknown test case: $test_name"
        return 1
    fi
    
    local times_file=$(mktemp)
    local failed=0
    
    printf "  Running %s: " "$test_name"
    
    for i in $(seq 1 $NUM_RUNS); do
        printf "."
        
        # Run the benchmark and capture output
        local output
        output=$(./energy_storms_seq $test_args 2>&1) || true
        
        # Extract time from output
        local time=$(echo "$output" | grep "^Time:" | awk '{print $2}')
        
        if [ -n "$time" ]; then
            echo "$time" >> "$times_file"
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
        
        # Write to CSV
        echo "$test_name,$mean,$stddev,$min,$max,$n" >> "$OUTPUT_FILE"
    else
        print_error "    All runs failed!"
        echo "$test_name,FAILED,,,," >> "$OUTPUT_FILE"
    fi
    
    rm -f "$times_file"
}

# ============================================================================
# MAIN
# ============================================================================

main() {
    print_header "Energy Storms Sequential Benchmark"
    
    echo "Configuration:"
    echo "  • Implementation:  Sequential"
    echo "  • Runs per test:   $NUM_RUNS"
    echo "  • Output file:     $OUTPUT_FILE"
    echo ""
    
    # Compile
    print_section "Compiling Sequential Implementation"
    cd "$SCRIPT_DIR"
    
    printf "  Compiling... "
    if make energy_storms_seq > /dev/null 2>&1; then
        print_success "OK"
    else
        print_error "FAILED"
        exit 1
    fi
    
    # Check executable exists
    if [ ! -x "./energy_storms_seq" ]; then
        print_error "Executable not found: ./energy_storms_seq"
        exit 1
    fi
    
    # Initialize CSV with header
    echo "test_name,avg_time,stddev,min_time,max_time,num_runs" > "$OUTPUT_FILE"
    
    # Run benchmarks
    print_section "Running Benchmarks"
    
    for test_name in "${TEST_NAMES[@]}"; do
        run_single_test "$test_name"
    done
    
    # Summary
    print_section "Summary"
    echo ""
    printf "%-12s %12s %12s %12s %12s %8s\n" "TEST" "AVG(s)" "STDDEV" "MIN" "MAX" "RUNS"
    echo "─────────────────────────────────────────────────────────────────────────"
    
    # Read and display CSV (skip header)
    tail -n +2 "$OUTPUT_FILE" | while IFS=',' read -r test mean stddev min max runs; do
        if [ "$mean" != "FAILED" ]; then
            printf "%-12s %12s %12s %12s %12s %8s\n" "$test" "$mean" "$stddev" "$min" "$max" "$runs"
        else
            printf "%-12s %12s\n" "$test" "FAILED"
        fi
    done
    
    echo ""
    print_header "Benchmark Complete"
    print_success "Results saved to: $OUTPUT_FILE"
}

main "$@"
