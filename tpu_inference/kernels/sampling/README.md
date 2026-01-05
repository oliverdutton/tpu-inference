# TPU Sampling Kernels

High-performance sampling and top-k operations for TPU inference, extracted and adapted from [tallax](https://github.com/oliverdutton/tallax).

## Overview

This package provides optimized Pallas kernels for sampling operations commonly used in large language model inference:

- **Divide-and-Filter Top-K**: Fast exact top-k selection with probabilistic early stopping
- **Bitonic Top-K**: Hardware-optimized sorting using compressed transpose format
- **Top-P Filtering**: Nucleus sampling with cumulative probability masking
- **Combined Sampling**: Fused top-k, top-p, and categorical sampling in a single kernel

## Performance Benchmarks

### Logit Sampling (Large Vocabulary Decoding)

Performance with top-k=64, top-p=0.95, and 262K vocabulary (Gemini 3 Pro):

| Batch Size | Average Speedup | Worst-Case Speedup | Latency (tallax) | Latency (baseline) |
|------------|----------------|-------------------|------------------|-------------------|
| 16         | **15×**        | **10×**           | 25μs             | 390μs             |
| 128        | **75×**        | **45×**           | 150μs            | 11,800μs          |

### Speculative Decoding Top-K

For top-5 operations with batch=16 and 32K draft vocabulary:

- **15× speedup** versus XLA (5.5μs vs 85μs)

### Comparison with JAX approx_max_k

The implementation provides **exact guarantees** (unlike the approximation algorithm) and achieves:

- **Up to 5× faster** across various shapes and k values
- **1.5× or better speedup** in most configurations

## Algorithms

### Divide-and-Filter Top-K

The core algorithm partitions the input vocabulary into bins, computes top-m per partition in parallel, identifies unconverged partitions where additional top-k values could exist, then executes final top-k only on relevant elements.

#### Probabilistic Early Stopping

With 256 bins, **bins-top-4 has a >95% probability of containing the entire top-128**. By checking convergence with the (m+1)th value's maximum across bins, the algorithm performs minimal-element top-k operations in most cases.

**Key Benefits:**
- Reduces memory bandwidth by filtering irrelevant elements early
- Guarantees exact top-k results (not approximate)
- Adapts to different vocabulary distributions (random vs worst-case)

### Bitonic Top-K Optimization

Uses a **Compressed Transpose Format** that distributes sort dimensions across multiple tiles:

- Reduces lane permutations from 128 to 4 operations at batch size 8
- Approaches zero cross-lane operations by batch 128
- Maximizes hardware utilization by keeping data in fast on-chip memory

### Sparse Random Sampling

Efficient categorical sampling from sparse logits distributions:

- Generates random samples only for non-zero probability positions
- Avoids full vocabulary materialization
- Maintains statistical correctness with dense random number generation

## Usage

### Basic Top-K

```python
from tpu_inference.kernels.sampling import top_k

# Select top-k elements
topk_values, topk_indices = top_k(
    logits,  # shape: (batch_size, vocab_size)
    k=64,
    num_bins=256,  # Number of partitions for divide-and-filter
)
```

### Combined Top-K + Top-P + Sampling

```python
from tpu_inference.kernels.sampling import topk_topp_and_sample
from tpu_inference.layers.jax.sample.sampling_metadata import TPUSupportedSamplingMetadata

# Create sampling metadata
metadata = TPUSupportedSamplingMetadata(
    top_k=jnp.array([64] * batch_size, dtype=jnp.int32),
    top_p=jnp.array([0.95] * batch_size, dtype=jnp.float32),
    temperature=jnp.array([0.7] * batch_size, dtype=jnp.float32),
    do_sampling=True,
    logprobs=False,
    use_pallas_kernel=True,
)

# Perform fused sampling
sampled_tokens = topk_topp_and_sample(
    rng_key,
    logits,
    metadata,
    max_k=128,  # Maximum k value supported
    sampling_eps=1e-5,
    replace_val=-1e12,
)
```

## Environment Variables

### `FLASH_SAMPLING_TOPK_THRESHOLD`

Controls when the optimized Pallas sampling kernel is used:

- **Default:** `0` (disabled)
- **Recommended:** `64` or `128` for large vocabulary models
- **Behavior:** Enables fast path when all top_k values in a batch are ≤ threshold

**Example:**
```bash
export FLASH_SAMPLING_TOPK_THRESHOLD=128
vllm serve meta-llama/Llama-3.1-8B --tensor_parallel_size=1
```

**Performance Impact:**
- 15-75× faster sampling for large batches with constrained top-k
- Most effective with vocabulary sizes > 32K
- Falls back to standard path if any request exceeds threshold

**When to use:**
- ✅ Large vocabulary models (100K+ tokens)
- ✅ Consistent top-k values across requests
- ✅ Batch sizes > 16
- ❌ Highly variable top-k values per request
- ❌ Small vocabularies (< 10K tokens)

## Module Structure

```
tpu_inference/kernels/sampling/
├── README.md                      # This file
├── __init__.py                    # Public API exports
├── divide_and_filter_topk.py     # Main top-k algorithm
├── bitonic_topk.py               # Bitonic sort-based top-k
├── top_p_and_sample.py           # Top-p filtering and sampling
├── sampling.py                   # Combined top-k/top-p/sample
├── cumsum.py                     # Cumulative sum operation
├── gather.py                     # Take-along-axis operation
├── sparse_random.py              # Sparse random sampling
├── utils.py                      # Shared utilities
└── topk_convergence_theory.py   # Convergence threshold computation
```

## Testing

Run the test suite:

```bash
# Run all sampling kernel tests
pytest tests/kernels/sampling/

# Run specific test
pytest tests/kernels/sampling/topk_topp_and_sample_test.py

# Run with different platforms
JAX_PLATFORMS=cpu pytest tests/kernels/sampling/  # CPU (interpret mode)
JAX_PLATFORMS=tpu pytest tests/kernels/sampling/  # TPU (production)
```

## Implementation Details

### Compressed Transpose Format

The bitonic top-k implementation uses a custom data layout that:

1. Splits arrays into (NUM_SUBLANES, NUM_LANES) tiles
2. Transposes dimensions to align with TPU memory hierarchy
3. Minimizes cross-tile communication during sorting
4. Automatically handles padding to hardware alignment requirements

### Convergence Guarantees

The divide-and-filter algorithm provides exact top-k results by:

1. Computing theoretical convergence thresholds based on bin count and target k
2. Checking if maximum values outside top-m exceed top-m minimums
3. Merging unconverged bins and re-running top-k only on filtered elements
4. Guaranteeing all top-k elements are found after at most log(vocab_size) iterations

### Memory Efficiency

Key optimizations:

- **VMEM usage:** Limited to ~115MB (90% of 128MB) via compiler directives
- **Scratch allocation:** Pre-allocated buffers reused across iterations
- **Block speculation:** Custom BlockSpec functions for dynamic access patterns
- **Type packing:** BF16 values + U16 indices packed into I32 for reduced bandwidth

## References

- Original tallax implementation: https://github.com/oliverdutton/tallax
- Pallas documentation: https://jax.readthedocs.io/en/latest/pallas.html
- TPU architecture: https://cloud.google.com/tpu/docs/system-architecture

## Contributing

When adding new sampling kernels:

1. Follow the `pallas_` naming convention for test imports
2. Add VMEM compiler limits for Pallas kernels in tests
3. Include both CPU (interpret=True) and TPU execution paths
4. Provide convergence proofs for approximate algorithms
5. Benchmark against JAX baseline implementations
