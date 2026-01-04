"""Test comparing top_p_mask implementations.

This test compares:
1. top_p_mask from tpu_inference.kernels.sampling.top_p_and_sample (kernel version)
2. topp_mask from tpu_inference.layers.common.binary_search (layers version)

The kernel version expects pre-sorted logits while the layers version works on
unsorted logits, so we need to adapt the inputs/outputs appropriately.
"""

import pytest
import jax
import jax.numpy as jnp
import numpy as np

from tpu_inference.kernels.sampling.top_p_and_sample import top_p_mask
from tpu_inference.layers.common.binary_search import topp_mask


def test_top_p_mask_vs_topp_mask_basic():
  """Test that kernel top_p_mask matches layers topp_mask on basic example."""
  batch_size = 8
  vocab_size = 128
  replace_val = -1e12

  # Create random logits
  key = jax.random.PRNGKey(42)
  logits = jax.random.normal(key, (batch_size, vocab_size), dtype=jnp.float32)

  # Test with p=0.9
  p = 0.9

  # Run layers version (works on unsorted logits)
  layers_result = topp_mask(logits, p, replace_val)

  # For kernel version, we need to:
  # 1. Sort logits in descending order along vocab axis
  # 2. Transpose to (vocab_size, batch_size) since kernel expects axis=0
  # 3. Create p array matching batch dimension

  # Sort logits descending along vocab axis (axis=-1)
  sorted_indices = jnp.argsort(logits, axis=-1)[:, ::-1]  # Descending
  sorted_logits = jnp.take_along_axis(logits, sorted_indices, axis=-1)

  # Transpose to (vocab_size, batch_size) for kernel
  sorted_logits_t = sorted_logits.T

  # Create p array
  p_array = jnp.full((batch_size,), p, dtype=jnp.float32)

  # Run kernel version
  kernel_result_t = top_p_mask(
    topk_logits=sorted_logits_t,
    p=p_array,
    replace_val=replace_val,
    axis=0
  )

  # Transpose back to (batch_size, vocab_size)
  kernel_result_sorted = kernel_result_t.T

  # Unsort the kernel result to match original logits order
  unsort_indices = jnp.argsort(sorted_indices, axis=-1)
  kernel_result = jnp.take_along_axis(kernel_result_sorted, unsort_indices, axis=-1)

  # Both should produce the same masking pattern
  # Check which elements are masked (set to replace_val)
  layers_masked = layers_result == replace_val
  kernel_masked = kernel_result == replace_val

  # The masking patterns should match
  np.testing.assert_array_equal(
    layers_masked,
    kernel_masked,
    err_msg="Kernel top_p_mask should produce same masking as layers topp_mask"
  )

  # For unmasked elements, values should match (within floating point tolerance)
  # Note: We only check unmasked elements since replace_val might differ in precision
  unmasked_indices = ~layers_masked
  np.testing.assert_allclose(
    layers_result[unmasked_indices],
    kernel_result[unmasked_indices],
    rtol=1e-5,
    atol=1e-6,
    err_msg="Unmasked logits should match between implementations"
  )


@pytest.mark.parametrize("p", [0.5, 0.8, 0.9, 0.95, 0.99, 1.0])
@pytest.mark.parametrize("batch_size", [1, 4, 16])
@pytest.mark.parametrize("vocab_size", [32, 128, 256])
def test_top_p_mask_vs_topp_mask_varied_params(p, batch_size, vocab_size):
  """Test top_p_mask vs topp_mask with varied parameters."""
  replace_val = -1e12

  # Create random logits
  key = jax.random.PRNGKey(42 + int(p * 100) + batch_size + vocab_size)
  logits = jax.random.normal(key, (batch_size, vocab_size), dtype=jnp.float32)

  # Run layers version
  layers_result = topp_mask(logits, p, replace_val)

  # Prepare for kernel version
  sorted_indices = jnp.argsort(logits, axis=-1)[:, ::-1]
  sorted_logits = jnp.take_along_axis(logits, sorted_indices, axis=-1)
  sorted_logits_t = sorted_logits.T
  p_array = jnp.full((batch_size,), p, dtype=jnp.float32)

  # Run kernel version
  kernel_result_t = top_p_mask(
    topk_logits=sorted_logits_t,
    p=p_array,
    replace_val=replace_val,
    axis=0
  )

  # Unsort kernel result
  kernel_result_sorted = kernel_result_t.T
  unsort_indices = jnp.argsort(sorted_indices, axis=-1)
  kernel_result = jnp.take_along_axis(kernel_result_sorted, unsort_indices, axis=-1)

  # Check masking patterns match
  layers_masked = layers_result == replace_val
  kernel_masked = kernel_result == replace_val

  np.testing.assert_array_equal(
    layers_masked,
    kernel_masked,
    err_msg=f"Masking patterns should match for p={p}, batch_size={batch_size}, vocab_size={vocab_size}"
  )


def test_top_p_mask_edge_case_p_equals_1():
  """Test that p=1.0 keeps all logits (no masking)."""
  batch_size = 4
  vocab_size = 64
  replace_val = -1e12
  p = 1.0

  key = jax.random.PRNGKey(123)
  logits = jax.random.normal(key, (batch_size, vocab_size), dtype=jnp.float32)

  # Layers version
  layers_result = topp_mask(logits, p, replace_val)

  # Kernel version
  sorted_indices = jnp.argsort(logits, axis=-1)[:, ::-1]
  sorted_logits = jnp.take_along_axis(logits, sorted_indices, axis=-1)
  sorted_logits_t = sorted_logits.T
  p_array = jnp.full((batch_size,), p, dtype=jnp.float32)

  kernel_result_t = top_p_mask(
    topk_logits=sorted_logits_t,
    p=p_array,
    replace_val=replace_val,
    axis=0
  )

  kernel_result_sorted = kernel_result_t.T
  unsort_indices = jnp.argsort(sorted_indices, axis=-1)
  kernel_result = jnp.take_along_axis(kernel_result_sorted, unsort_indices, axis=-1)

  # With p=1.0, no elements should be masked
  layers_masked = layers_result == replace_val
  kernel_masked = kernel_result == replace_val

  assert not jnp.any(layers_masked), "p=1.0 should not mask any elements (layers)"
  assert not jnp.any(kernel_masked), "p=1.0 should not mask any elements (kernel)"

  # All logits should remain unchanged
  np.testing.assert_allclose(layers_result, logits, rtol=1e-5)
  np.testing.assert_allclose(kernel_result, logits, rtol=1e-5)


def test_top_p_mask_edge_case_small_p():
  """Test that very small p masks most logits."""
  batch_size = 4
  vocab_size = 64
  replace_val = -1e12
  p = 0.01  # Very small p

  key = jax.random.PRNGKey(456)
  logits = jax.random.normal(key, (batch_size, vocab_size), dtype=jnp.float32)

  # Layers version
  layers_result = topp_mask(logits, p, replace_val)

  # Kernel version
  sorted_indices = jnp.argsort(logits, axis=-1)[:, ::-1]
  sorted_logits = jnp.take_along_axis(logits, sorted_indices, axis=-1)
  sorted_logits_t = sorted_logits.T
  p_array = jnp.full((batch_size,), p, dtype=jnp.float32)

  kernel_result_t = top_p_mask(
    topk_logits=sorted_logits_t,
    p=p_array,
    replace_val=replace_val,
    axis=0
  )

  kernel_result_sorted = kernel_result_t.T
  unsort_indices = jnp.argsort(sorted_indices, axis=-1)
  kernel_result = jnp.take_along_axis(kernel_result_sorted, unsort_indices, axis=-1)

  # With small p, most elements should be masked
  layers_masked = layers_result == replace_val
  kernel_masked = kernel_result == replace_val

  # Should have many masked elements
  layers_mask_ratio = jnp.mean(layers_masked.astype(jnp.float32))
  kernel_mask_ratio = jnp.mean(kernel_masked.astype(jnp.float32))

  assert layers_mask_ratio > 0.8, f"Small p should mask >80% of logits (layers: {layers_mask_ratio:.2f})"
  assert kernel_mask_ratio > 0.8, f"Small p should mask >80% of logits (kernel: {kernel_mask_ratio:.2f})"

  # Masking patterns should match
  np.testing.assert_array_equal(layers_masked, kernel_masked)


if __name__ == "__main__":
  print("Running top_p_mask comparison tests...")

  print("\n1. Basic test...")
  test_top_p_mask_vs_topp_mask_basic()
  print("   ✓ Passed")

  print("\n2. Edge case: p=1.0...")
  test_top_p_mask_edge_case_p_equals_1()
  print("   ✓ Passed")

  print("\n3. Edge case: small p...")
  test_top_p_mask_edge_case_small_p()
  print("   ✓ Passed")

  print("\n4. Varied parameters...")
  test_cases = [
    (0.5, 4, 128),
    (0.9, 16, 256),
    (0.95, 1, 32),
  ]
  for p, batch_size, vocab_size in test_cases:
    print(f"   Testing p={p}, batch={batch_size}, vocab={vocab_size}...")
    test_top_p_mask_vs_topp_mask_varied_params(p, batch_size, vocab_size)
    print("   ✓ Passed")

  print("\nAll top_p_mask comparison tests passed!")
