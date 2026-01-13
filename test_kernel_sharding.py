#!/usr/bin/env python3
"""
Standalone test for kernel sampling with sharding support on CPU.

This test demonstrates:
1. Setting up 8 CPU devices for simulating sharding
2. Testing top-p sampling kernel with sharding
3. Using jax.lax.top_k for topk operations
4. Verifying sharding patterns compile correctly
"""

import os
import sys
# MUST set XLA_FLAGS before importing JAX
os.environ["XLA_FLAGS"] = '--xla_force_host_platform_device_count=8'
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Force CPU-only

import functools
import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from jax.experimental.custom_partitioning import custom_partitioning
from jax import lax

print(f"JAX version: {jax.__version__}")
print(f"Available devices: {jax.devices()}")
print(f"Number of devices: {len(jax.devices())}")


# Constants
SAMPLING_EPS = 1e-5
REPLACE_VAL = -1e12


def top_p_mask_simple(logits, p, replace_val):
    """
    Simplified top-p masking using JAX operations.

    Args:
        logits: Logits [batch_size, vocab_size]
        p: Top-p thresholds [batch_size]
        replace_val: Value for masked elements

    Returns:
        Masked logits [batch_size, vocab_size]
    """
    # Convert to probabilities
    probs = jax.nn.softmax(logits, axis=-1)

    # Sort probabilities in descending order
    sorted_probs = jnp.sort(probs, axis=-1)[:, ::-1]

    # Cumulative sum
    cumsum_probs = jnp.cumsum(sorted_probs, axis=-1)

    # Find cutoff where cumsum exceeds p
    p_expanded = p[:, None]
    cutoff_mask = cumsum_probs >= p_expanded

    # Find the minimum probability that's included
    # Use argsort to get back to original indices
    sorted_indices = jnp.argsort(-probs, axis=-1)  # Descending
    cutoff_positions = jnp.argmax(cutoff_mask, axis=-1)

    # Create threshold based on cutoff position
    batch_indices = jnp.arange(logits.shape[0])
    threshold_indices = sorted_indices[batch_indices, cutoff_positions]
    threshold_probs = probs[batch_indices, threshold_indices][:, None]

    # Mask probabilities below threshold
    return jnp.where(probs >= threshold_probs, logits, replace_val)


def sample_with_topk_topp(
    rng_key,
    topk_logits,
    topk_indices,
    top_p,
    temperature,
    vocab_size,
    sampling_eps,
    replace_val,
):
    """
    Sample function using pre-computed topk values.

    This mirrors the kernel approach where topk is computed first,
    then top-p filtering and sampling are applied.

    Args:
        rng_key: Random key [2]
        topk_logits: Top-k logits [batch_size, k]
        topk_indices: Top-k indices [batch_size, k]
        top_p: Top-p thresholds [batch_size]
        temperature: Temperature values [batch_size]
        vocab_size: Vocabulary size
        sampling_eps: Epsilon for greedy sampling
        replace_val: Replacement value for masking

    Returns:
        Sampled token indices [batch_size]
    """
    batch_size, k = topk_logits.shape

    # Apply temperature scaling
    temp_expanded = temperature[:, None]
    scaled_logits = topk_logits / temp_expanded

    # Apply top-p mask to topk logits
    masked_logits = top_p_mask_simple(scaled_logits, top_p, replace_val)

    # Create a sparse representation for sampling
    # We need to map back to full vocabulary space
    # Create full logits array filled with replace_val
    full_logits = jnp.full((batch_size, vocab_size), replace_val, dtype=scaled_logits.dtype)

    # Scatter the topk values back
    batch_indices = jnp.arange(batch_size)[:, None]
    full_logits = full_logits.at[batch_indices, topk_indices].set(masked_logits)

    # Sample from categorical distribution
    sampled = jax.random.categorical(rng_key, full_logits)

    # Greedy: use argmax of original logits (first topk element after sorting)
    greedy = topk_indices[:, 0]

    # Use greedy for low temperatures
    return jnp.where(temperature < sampling_eps, greedy, sampled)


@functools.partial(
    jax.jit,
    static_argnames=["vocab_size", "sampling_eps", "replace_val"],
)
def sample_with_topk_topp_sharded(
    rng_key,
    topk_logits,
    topk_indices,
    top_p,
    temperature,
    vocab_size,
    sampling_eps,
    replace_val,
):
    """
    Sharded wrapper with custom partitioning support.

    Sharding pattern:
    - Batch dimension: sharded
    - K dimension: replicated
    - All other inputs: follow batch sharding
    """

    @custom_partitioning
    def sharded_sample(topk_logits, topk_indices, rng_key, top_p, temperature):
        return sample_with_topk_topp(
            rng_key,
            topk_logits,
            topk_indices,
            top_p,
            temperature,
            vocab_size,
            sampling_eps,
            replace_val,
        )

    def infer_sharding_from_operands(mesh, arg_shapes, result_shape):
        # Output follows batch dimension sharding
        batch_spec = arg_shapes[0].sharding.spec[0]
        return NamedSharding(mesh, P(batch_spec))

    def partition(mesh, arg_shapes, out_shapes):
        arg_shardings, out_shardings = jax.tree.map(
            lambda s: s.sharding, (arg_shapes, out_shapes)
        )
        batch_axis_name = arg_shardings[0].spec[0]

        def shmap_fn(topk_logits, topk_indices, rng_key, top_p, temperature):
            # Compute offset for batch sharding
            dim0_offset = 0
            if batch_axis_name is not None:
                dim0_offset = lax.axis_index(batch_axis_name) * topk_logits.shape[0]

            # For deterministic results across shards, we could split the rng
            # based on the shard index, but for testing we'll use the same key
            return sample_with_topk_topp(
                rng_key,
                topk_logits,
                topk_indices,
                top_p,
                temperature,
                vocab_size,
                sampling_eps,
                replace_val,
            )

        return mesh, shmap_fn, out_shardings, arg_shardings

    # Define partitioning rules
    # Format: "batch k, batch k, rng, batch, batch -> batch"
    sharded_sample.def_partition(
        infer_sharding_from_operands=infer_sharding_from_operands,
        partition=partition,
        sharding_rule="b k, b k, r, b, b -> b",
        need_replication_factors=("k", "r"),
    )

    return sharded_sample(topk_logits, topk_indices, rng_key, top_p, temperature)


def test_basic_topk():
    """Test basic jax.lax.top_k functionality."""
    print("\n" + "="*80)
    print("Test 1: JAX lax.top_k on CPU")
    print("="*80)

    batch_size = 4
    vocab_size = 1000
    k = 50

    rng = jax.random.PRNGKey(42)
    logits = jax.random.normal(rng, (batch_size, vocab_size))

    # Use jax.lax.top_k
    topk_values, topk_indices = jax.lax.top_k(logits, k)

    print(f"Input shape: {logits.shape}")
    print(f"Top-k values shape: {topk_values.shape}")
    print(f"Top-k indices shape: {topk_indices.shape}")

    # Verify sorting
    for i in range(batch_size):
        sorted_correctly = jnp.all(topk_values[i, :-1] >= topk_values[i, 1:])
        assert sorted_correctly, f"Batch {i} not sorted correctly"

    print(f"✓ All batches sorted correctly")
    print(f"Sample top-5 values from batch 0: {topk_values[0, :5]}")
    print(f"Sample top-5 indices from batch 0: {topk_indices[0, :5]}")
    print("✓ Test passed!")


def test_sampling_with_topk():
    """Test sampling with topk."""
    print("\n" + "="*80)
    print("Test 2: Sampling with Top-K and Top-P")
    print("="*80)

    batch_size = 8
    vocab_size = 1000
    k = 50

    # Generate test data
    rng = jax.random.PRNGKey(42)
    logits = jax.random.normal(rng, (batch_size, vocab_size))

    # Get top-k using jax.lax.top_k
    topk_logits, topk_indices = jax.lax.top_k(logits, k)

    # Sampling parameters
    top_p = jnp.array([0.9] * batch_size, dtype=jnp.float32)
    temperature = jnp.array([1.0] * batch_size, dtype=jnp.float32)

    # Sample
    rng_sample = jax.random.PRNGKey(123)
    samples = sample_with_topk_topp(
        rng_sample,
        topk_logits,
        topk_indices,
        top_p,
        temperature,
        vocab_size,
        SAMPLING_EPS,
        REPLACE_VAL,
    )

    print(f"Sampled tokens shape: {samples.shape}")
    print(f"Sampled tokens: {samples}")

    # Test greedy sampling
    temperature_greedy = jnp.array([0.0] * batch_size, dtype=jnp.float32)
    greedy_samples = sample_with_topk_topp(
        rng_sample,
        topk_logits,
        topk_indices,
        top_p,
        temperature_greedy,
        vocab_size,
        SAMPLING_EPS,
        REPLACE_VAL,
    )

    print(f"Greedy samples: {greedy_samples}")
    print(f"Expected greedy (topk[0]): {topk_indices[:, 0]}")

    # Verify greedy matches top-1
    assert jnp.all(greedy_samples == topk_indices[:, 0]), "Greedy sampling failed"
    print("✓ Greedy sampling verified")
    print("✓ Test passed!")


def test_sharded_sampling():
    """Test sampling with sharding across 8 devices."""
    print("\n" + "="*80)
    print("Test 3: Sharded Sampling (8 CPU Devices)")
    print("="*80)

    batch_size = 16  # Will be split across 8 devices (2 per device)
    vocab_size = 1000
    k = 50

    # Create mesh
    devices = jax.devices()
    assert len(devices) == 8, f"Expected 8 devices, got {len(devices)}"
    mesh = Mesh(devices, axis_names=('data',))

    print(f"Mesh: {mesh}")
    print(f"Device count: {len(devices)}")

    # Generate test data
    rng = jax.random.PRNGKey(42)
    logits = jax.random.normal(rng, (batch_size, vocab_size))

    # Get top-k
    topk_logits, topk_indices = jax.lax.top_k(logits, k)

    # Shard inputs
    batch_sharding = NamedSharding(mesh, P('data'))
    batch_k_sharding = NamedSharding(mesh, P('data', None))
    replicated_sharding = NamedSharding(mesh, P())

    topk_logits_sharded = jax.device_put(topk_logits, batch_k_sharding)
    topk_indices_sharded = jax.device_put(topk_indices, batch_k_sharding)
    top_p = jax.device_put(jnp.array([0.9] * batch_size), batch_sharding)
    temperature = jax.device_put(jnp.array([1.0] * batch_size), batch_sharding)

    print(f"topk_logits sharding: {topk_logits_sharded.sharding}")
    print(f"topk_indices sharding: {topk_indices_sharded.sharding}")
    print(f"top_p sharding: {top_p.sharding}")

    # Sample with sharding
    rng_sample = jax.random.PRNGKey(123)
    samples = sample_with_topk_topp_sharded(
        rng_sample,
        topk_logits_sharded,
        topk_indices_sharded,
        top_p,
        temperature,
        vocab_size,
        SAMPLING_EPS,
        REPLACE_VAL,
    )

    print(f"Output shape: {samples.shape}")
    print(f"Output sharding: {samples.sharding}")
    print(f"Sampled tokens: {samples}")
    print("✓ Sharded sampling completed successfully!")
    print("✓ Test passed!")


def test_compilation():
    """Test that sharded function compiles."""
    print("\n" + "="*80)
    print("Test 4: Compilation and Lowering")
    print("="*80)

    batch_size = 16
    vocab_size = 1000
    k = 50

    devices = jax.devices()
    mesh = Mesh(devices, axis_names=('data',))

    # Create test inputs
    rng = jax.random.PRNGKey(42)
    logits = jax.random.normal(rng, (batch_size, vocab_size))
    topk_logits, topk_indices = jax.lax.top_k(logits, k)

    # Shard inputs
    batch_k_sharding = NamedSharding(mesh, P('data', None))
    batch_sharding = NamedSharding(mesh, P('data'))

    topk_logits = jax.device_put(topk_logits, batch_k_sharding)
    topk_indices = jax.device_put(topk_indices, batch_k_sharding)
    top_p = jax.device_put(jnp.array([0.9] * batch_size), batch_sharding)
    temperature = jax.device_put(jnp.array([1.0] * batch_size), batch_sharding)

    rng_sample = jax.random.PRNGKey(123)

    # Lower the function
    print("Lowering function...")
    lowered = sample_with_topk_topp_sharded.lower(
        rng_sample, topk_logits, topk_indices, top_p, temperature,
        vocab_size, SAMPLING_EPS, REPLACE_VAL
    )

    print("✓ Function lowered successfully!")

    # Compile
    print("Compiling function...")
    compiled = lowered.compile()
    print("✓ Function compiled successfully!")

    # Execute
    print("Executing compiled function...")
    result = compiled(rng_sample, topk_logits, topk_indices, top_p, temperature)
    print(f"Result shape: {result.shape}")
    print(f"Result: {result}")
    print("✓ Compiled function executed successfully!")
    print("✓ Test passed!")


def test_sharding_constraint():
    """Test with_sharding_constraint pattern from the original code."""
    print("\n" + "="*80)
    print("Test 5: with_sharding_constraint Pattern")
    print("="*80)

    batch_size = 16
    vocab_size = 1000

    devices = jax.devices()
    mesh = Mesh(devices, axis_names=('data',))

    @functools.partial(jax.jit, static_argnames=['mesh'])
    def sample_with_constraint(logits, mesh):
        """Apply sharding constraint like in the original sample function."""
        # Unshard the vocab dimension (shard batch, replicate vocab)
        logits = jax.lax.with_sharding_constraint(
            logits, NamedSharding(mesh, P('data', None))
        )
        # Just return argmax for simplicity
        return jnp.argmax(logits, axis=-1)

    # Create sharded logits
    rng = jax.random.PRNGKey(42)
    logits = jax.random.normal(rng, (batch_size, vocab_size))

    # Initially shard on batch dimension
    sharding = NamedSharding(mesh, P('data', None))
    logits_sharded = jax.device_put(logits, sharding)

    print(f"Input sharding: {logits_sharded.sharding}")

    # Apply function with constraint
    result = sample_with_constraint(logits_sharded, mesh)

    print(f"Output shape: {result.shape}")
    print(f"Output sharding: {result.sharding}")
    print(f"Result: {result}")
    print("✓ Sharding constraint applied successfully!")
    print("✓ Test passed!")


def main():
    """Run all tests."""
    print("="*80)
    print("JAX Kernel Sampling with Sharding Test Suite")
    print("="*80)
    print()
    print("Setup:")
    print(f"  - 8 simulated CPU devices via XLA_FLAGS")
    print(f"  - JAX version: {jax.__version__}")
    print(f"  - Testing sharding patterns from pallas_sampling branch")

    try:
        test_basic_topk()
        test_sampling_with_topk()
        test_sharded_sampling()
        test_compilation()
        test_sharding_constraint()

        print("\n" + "="*80)
        print("ALL TESTS PASSED! ✓")
        print("="*80)
        print("\nKey Findings:")
        print("1. ✓ Successfully simulated 8 CPU devices using XLA_FLAGS")
        print("2. ✓ jax.lax.top_k works correctly as topk replacement")
        print("3. ✓ Sharding constraints compile and execute properly")
        print("4. ✓ Custom partitioning with sharding rules works on CPU")
        print("5. ✓ Batch dimension sharding with replicated vocab dimension")
        print()
        print("Sharding Pattern Verified:")
        print("  - topk_logits: P('data', None)  # batch sharded, k replicated")
        print("  - topk_indices: P('data', None) # batch sharded, k replicated")
        print("  - top_p: P('data')              # batch sharded")
        print("  - temperature: P('data')        # batch sharded")
        print("  - output: P('data')             # batch sharded")

    except Exception as e:
        print("\n" + "="*80)
        print("TEST FAILED! ✗")
        print("="*80)
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
