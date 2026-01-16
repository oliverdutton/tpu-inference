# Gemma 3 4B IT Benchmark Commands

## Step 1: Start vLLM Server

```bash
vllm serve google/gemma-3-4b-it \
  --tensor-parallel-size 1 \
  --swap-space 16 \
  --disable-log-stats \
  --disable-log-requests \
  --load-format dummy \
  --max-model-len 8192
```

## Step 2: Run Benchmark (in another terminal)

```bash
python scripts/vllm/benchmarking/benchmark_serving.py \
  --backend vllm \
  --model google/gemma-3-4b-it \
  --dataset-name random \
  --num-prompts 200 \
  --random-input-len 1024 \
  --random-output-len 2048 \
  --request-rate 10
```

## Alternative: Higher QPS to Find Saturation Point

### Try 50 QPS:
```bash
python scripts/vllm/benchmarking/benchmark_serving.py \
  --backend vllm \
  --model google/gemma-3-4b-it \
  --dataset-name random \
  --num-prompts 200 \
  --random-input-len 1024 \
  --random-output-len 2048 \
  --request-rate 50
```

### Try 100 QPS:
```bash
python scripts/vllm/benchmarking/benchmark_serving.py \
  --backend vllm \
  --model google/gemma-3-4b-it \
  --dataset-name random \
  --num-prompts 200 \
  --random-input-len 1024 \
  --random-output-len 2048 \
  --request-rate 100
```

### Try Infinite QPS (send all at once, let server batch):
```bash
python scripts/vllm/benchmarking/benchmark_serving.py \
  --backend vllm \
  --model google/gemma-3-4b-it \
  --dataset-name random \
  --num-prompts 200 \
  --random-input-len 1024 \
  --random-output-len 2048 \
  --request-rate inf
```

## Expected Token Counts

- **Input tokens**: 200 × 1,024 = 204,800 tokens
- **Output tokens**: 200 × 2,048 = 409,600 tokens
- **Total tokens**: 614,400 tokens

## QPS Analysis

- **10 QPS**: 10 requests/sec → takes 20 seconds to send all 200 requests
- **50 QPS**: 50 requests/sec → takes 4 seconds to send all requests
- **100 QPS**: 100 requests/sec → takes 2 seconds to send all requests
- **inf QPS**: All 200 requests sent immediately (stress test)

## What to Watch For

The benchmark will report:
- **Request throughput (req/s)**: How many requests completed per second
- **Output token throughput (tok/s)**: Tokens generated per second (target: 1000-3000+ for A100/TPU)
- **Mean TTFT (ms)**: Time to first token
- **Mean TPOT (ms)**: Time per output token

When throughput stops increasing with higher QPS, you've found saturation.
