#!/bin/bash
# Benchmark script for serving_gemma_3_4b_it_tp1_random_in1k_out2k

# Log the vLLM server output to a file
LOG_FILE="gemma_server.log"
BENCHMARK_LOG_FILE="gemma_benchmark.log"
# The sentinel message that indicates the server is ready
export READY_MESSAGE="Application startup complete."
# After how long we should timeout if the server doesn't start
export TIMEOUT_SECONDS=3600

# Model and benchmark configuration
MODEL="google/gemma-3-4b-it"
TENSOR_PARALLEL_SIZE=1
SWAP_SPACE=16
MAX_MODEL_LEN=8192
LOAD_FORMAT="dummy"

# Client/Benchmark parameters
BACKEND="vllm"
DATASET_NAME="random"
NUM_PROMPTS=200
RANDOM_INPUT_LEN=1024
RANDOM_OUTPUT_LEN=2048
REQUEST_RATE=10  # QPS from qps_list[0]

# Get the directory of this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Access shared benchmarking functionality
if [ -f "$SCRIPT_DIR/tests/e2e/benchmarking/bench_utils.sh" ]; then
    # shellcheck disable=SC1091
    source "$SCRIPT_DIR/tests/e2e/benchmarking/bench_utils.sh"
else
    echo "Error: bench_utils.sh not found"
    exit 1
fi

echo "==================================================="
echo "Gemma 3 4B IT Benchmarking Setup"
echo "==================================================="
echo "Model: $MODEL"
echo "Tensor Parallel Size: $TENSOR_PARALLEL_SIZE"
echo "Swap Space: $SWAP_SPACE GB"
echo "Max Model Length: $MAX_MODEL_LEN"
echo "Load Format: $LOAD_FORMAT"
echo ""
echo "Client Parameters:"
echo "  Backend: $BACKEND"
echo "  Dataset: $DATASET_NAME"
echo "  Number of Prompts: $NUM_PROMPTS"
echo "  Random Input Length: $RANDOM_INPUT_LEN"
echo "  Random Output Length: $RANDOM_OUTPUT_LEN"
echo "  Request Rate (QPS): $REQUEST_RATE"
echo "==================================================="

# Clean up any existing log files
rm -f "$LOG_FILE" "$BENCHMARK_LOG_FILE"

# Start the vLLM server
echo ""
echo "Starting vLLM server..."
(vllm serve "$MODEL" \
    --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
    --swap-space "$SWAP_SPACE" \
    --disable-log-stats \
    --disable-log-requests \
    --load-format "$LOAD_FORMAT" \
    --max-model-len "$MAX_MODEL_LEN" \
    2>&1 | tee -a "$LOG_FILE") &

# Set trap to ensure cleanup happens even on immediate or normal exit
trap 'cleanUp "$MODEL"' EXIT

# Wait for server to be ready
waitForServerReady

# Run the benchmark
echo ""
echo "Starting benchmark..."
python "$SCRIPT_DIR/scripts/vllm/benchmarking/benchmark_serving.py" \
    --backend "$BACKEND" \
    --model "$MODEL" \
    --dataset-name "$DATASET_NAME" \
    --num-prompts "$NUM_PROMPTS" \
    --random-input-len "$RANDOM_INPUT_LEN" \
    --random-output-len "$RANDOM_OUTPUT_LEN" \
    --request-rate "$REQUEST_RATE" \
    2>&1 | tee -a "$BENCHMARK_LOG_FILE"

echo ""
echo "==================================================="
echo "Benchmark completed!"
echo "==================================================="
echo "Server log: $LOG_FILE"
echo "Benchmark log: $BENCHMARK_LOG_FILE"
echo "==================================================="
