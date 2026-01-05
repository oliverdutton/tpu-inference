# Configuration and Environment Variables

This page documents the environment variables available for configuring vLLM TPU behavior.

## Performance Optimization

### `FLASH_SAMPLING_TOPK_THRESHOLD`

Controls the optimized Pallas sampling kernel for large vocabulary models.

**Type:** Integer
**Default:** `0` (disabled)
**Recommended:** `64` or `128`

When set to a positive value, enables a high-performance sampling path that is significantly faster than the standard implementation. The fast path is used when **all** requests in a batch have `top_k ≤ FLASH_SAMPLING_TOPK_THRESHOLD`.

**Performance Impact:**

| Batch Size | Speedup (avg) | Speedup (worst) | Use Case |
|------------|---------------|-----------------|----------|
| 16         | 15×           | 10×             | Small batch inference |
| 128        | 75×           | 45×             | Large batch serving |

**Example:**
```bash
export FLASH_SAMPLING_TOPK_THRESHOLD=128
vllm serve meta-llama/Llama-3.1-8B \
    --tensor_parallel_size=1 \
    --max-model-len=2048
```

**When to use:**
- ✅ Large vocabulary models (100K+ tokens like Gemini, Qwen)
- ✅ Consistent top_k sampling parameters across requests
- ✅ Batch sizes of 16 or more
- ✅ Latency-sensitive applications

**When NOT to use:**
- ❌ Highly variable top_k values per request (causes fallback to slow path)
- ❌ Small vocabularies (< 10K tokens, limited benefit)
- ❌ Greedy decoding only (no sampling parameters)

**Implementation Details:**
- Uses divide-and-filter top-k algorithm with probabilistic early stopping
- Falls back to standard path if any request exceeds threshold
- Adds minimal compilation overhead during startup
- See [sampling kernel documentation](../tpu_inference/kernels/sampling/README.md) for algorithm details

---

## JAX and TPU Configuration

### `JAX_PLATFORMS`

Specifies which JAX backend to use.

**Type:** String
**Default:** `""` (auto-detect)
**Options:** `"tpu"`, `"cpu"`, `"proxy"`

**Example:**
```bash
export JAX_PLATFORMS=tpu
```

### `TPU_ACCELERATOR_TYPE`

TPU accelerator type identifier.

**Type:** String
**Default:** `None` (auto-detect from VM metadata)
**Example:** `"v5litepod-16"`, `"v6e-8"`

### `TPU_NAME`

Name of the TPU resource.

**Type:** String
**Default:** `None`

### `TPU_WORKER_ID`

Worker ID for multi-host TPU setups.

**Type:** String
**Default:** `None`

---

## Multi-Host Configuration

### `TPU_MULTIHOST_BACKEND`

Backend for multi-host communication.

**Type:** String
**Default:** `""` (disabled)
**Options:** `"ray"`

**Example:**
```bash
export TPU_MULTIHOST_BACKEND=ray
```

### `NUM_SLICES`

Number of TPU slices for multi-slice mesh configurations.

**Type:** Integer
**Default:** `1`

---

## Disaggregated Serving

### `PREFILL_SLICES`

Slice configuration for disaggregated prefill workers.

**Type:** String (comma-separated slice IDs)
**Default:** `""`

**Example:**
```bash
export PREFILL_SLICES="0,1"
```

### `DECODE_SLICES`

Slice configuration for disaggregated decode workers.

**Type:** String (comma-separated slice IDs)
**Default:** `""`

**Example:**
```bash
export DECODE_SLICES="2,3"
```

---

## Compilation and Debugging

### `SKIP_JAX_PRECOMPILE`

Skip JAX precompilation step during initialization.

**Type:** Boolean (0 or 1)
**Default:** `0` (precompile enabled)

**Example:**
```bash
export SKIP_JAX_PRECOMPILE=1
```

**Use cases:**
- Faster startup for development/testing
- When models are already compiled and cached
- **Warning:** First inference will be slow

### `VLLM_XLA_CHECK_RECOMPILATION`

Check for XLA recompilation during execution and log warnings.

**Type:** Boolean (0 or 1)
**Default:** `0` (disabled)

**Example:**
```bash
export VLLM_XLA_CHECK_RECOMPILATION=1
```

**Use cases:**
- Debugging unexpected recompilations
- Performance analysis
- **Warning:** Adds overhead, not for production

---

## Model Implementation

### `MODEL_IMPL_TYPE`

Selects model implementation framework.

**Type:** String
**Default:** `"flax_nnx"`
**Options:** `"vllm"`, `"flax_nnx"`, `"jetpack"`

**Example:**
```bash
export MODEL_IMPL_TYPE=flax_nnx
```

**Options:**
- `"flax_nnx"` - Recommended for JAX-optimized models
- `"vllm"` - PyTorch models via torchax
- `"jetpack"` - Experimental JAX models

### `NEW_MODEL_DESIGN`

Enable experimental model design features.

**Type:** Boolean (0 or 1)
**Default:** `0` (disabled)

---

## Profiling

### `PHASED_PROFILING_DIR`

Directory to store phased profiling output.

**Type:** String (path)
**Default:** `""` (disabled)

**Example:**
```bash
export PHASED_PROFILING_DIR=/tmp/profiles
```

See [profiling documentation](profiling.md) for details.

### `PYTHON_TRACER_LEVEL`

Python tracer level for profiling.

**Type:** Integer
**Default:** `1`

---

## Advanced Features

### `USE_MOE_EP_KERNEL`

Use custom expert-parallel kernel for Mixture of Experts models.

**Type:** Boolean (0 or 1)
**Default:** `0` (disabled)

**Example:**
```bash
export USE_MOE_EP_KERNEL=1
```

### `ENABLE_QUANTIZED_MATMUL_KERNEL`

Enable experimental quantized matrix multiplication kernels.

**Type:** Boolean (0 or 1)
**Default:** `0` (disabled)

---

## Ray Configuration

### `RAY_USAGE_STATS_ENABLED`

Enable/disable Ray usage statistics collection.

**Type:** String
**Default:** `"0"` (disabled)

### `VLLM_USE_RAY_COMPILED_DAG_CHANNEL_TYPE`

Ray compiled DAG channel type for TPU.

**Type:** String
**Default:** `"shm"` (shared memory)
**Options:** `"shm"`

---

## Configuration Precedence

Environment variables are read once at startup and cached for performance. To change configuration:

1. Set environment variables **before** starting vLLM
2. Restart the vLLM server process
3. Variables cannot be changed at runtime

**Example startup script:**
```bash
#!/bin/bash
export FLASH_SAMPLING_TOPK_THRESHOLD=128
export MODEL_IMPL_TYPE=flax_nnx
export SKIP_JAX_PRECOMPILE=0

vllm serve meta-llama/Llama-3.1-8B \
    --tensor_parallel_size=1 \
    --max-model-len=2048 \
    --download_dir=/tmp/models
```

---

## See Also

- [Sampling Kernel Documentation](../tpu_inference/kernels/sampling/README.md) - Deep dive into sampling algorithms
- [Profiling Guide](profiling.md) - Performance analysis and optimization
- [Quickstart Guide](getting_started/quickstart.md) - Basic setup and usage
