#!/bin/bash

# Configuration
EXECUTABLE="./energy_storms_mpi_omp"
TEST_FILE="test_files/test_02_a30k_p20k_w1 test_files/test_02_a30k_p20k_w2 test_files/test_02_a30k_p20k_w3 test_files/test_02_a30k_p20k_w4 test_files/test_02_a30k_p20k_w5 test_files/test_02_a30k_p20k_w6"
SIZE=20000
MAX_THREADS=8  # Change to your core count

echo "==================================="
echo "Performance Test - Strong Scaling"
echo "==================================="
echo ""

# Get sequential baseline
echo "Running sequential version..."
SEQ_TIME=$(./energy_storms_seq $SIZE $TEST_FILE | grep "Time:" | awk '{print $2}')
echo "Sequential time: $SEQ_TIME seconds"
echo ""

echo "Thread,Time(s),Speedup,Efficiency(%)"
echo "--------------------------------------"

# Test with different thread counts
for threads in 1 2 4 8 16; do
    if [ $threads -gt $MAX_THREADS ]; then
        break
    fi
    
    export OMP_NUM_THREADS=$threads
    
    # Run 3 times and take average (more reliable)
    total=0
    for run in 1 2 3; do
        time=$(mpirun -np 1 $EXECUTABLE $SIZE $TEST_FILE 2>/dev/null | grep "Time:" | awk '{print $2}')
        total=$(echo "$total + $time" | bc)
    done
    avg_time=$(echo "scale=6; $total / 3" | bc)
    
    # Calculate metrics
    speedup=$(echo "scale=2; $SEQ_TIME / $avg_time" | bc)
    efficiency=$(echo "scale=1; ($speedup / $threads) * 100" | bc)
    
    echo "$threads,$avg_time,$speedup,$efficiency"
done

echo ""
echo "==================================="