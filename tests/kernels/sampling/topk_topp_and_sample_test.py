import pytest
import jax
import jax.numpy as jnp
import numpy as np
from tpu_inference.kernels.sampling.sampling import topk_topp_and_sample
# NOTE: tpu_inference_sampling_as_standalone_file was not copied over (excluded per requirements)
# from tpu_inference.kernels.sampling.vllm.tpu_inference_sampling_as_standalone_file import (
#   TPUSupportedSamplingMetadata,
#   sample as vllm_sample,
#   ShardingAxisName2D,
#   Mesh,
# )
from tpu_inference.kernels.sampling.utils import is_cpu_platform


def uniquely_define_topk(logits, k):
  """Ensure topk is well-defined by handling ties at the k-th boundary.

  If more than k values are >= the k-th largest value, set extras to -inf.
  This ensures topk is deterministic.
  """
  boundary_val = jax.lax.sort(logits)[-k]
  mask = logits >= boundary_val
  # if more than k values gt k-th largest value, set them to -inf
  mask = mask & (mask.cumsum() > k)
  return jnp.where(mask, float("-inf"), logits)


# shapes on either side of the shape[1] pure bitonic vs divide and filter implementations
@pytest.mark.parametrize(
  "shape",
  [
    (16, 16384),
    (13, 11792),
    (256, 2048),
    (256, 8192),
    (279, 3570),
    (279, 7593),
  ],
)
@pytest.mark.parametrize("dtype", [jnp.bfloat16, jnp.float32])
@pytest.mark.parametrize("case", ["random", "worst_case"])
@pytest.mark.parametrize("seed", [42, 123, 456])
@pytest.mark.skip(
  reason="Test requires tpu_inference_sampling_as_standalone_file which was not copied over"
)
def test_topk_topp_and_sample(shape, dtype, case, seed):
  """Test topk_topp_and_sample implementation against vLLM reference.

  Tests both random and worst-case logits distributions.
  Validates that pallas implementation matches vLLM sampling behavior exactly.
  NOTE: Disabled - requires tpu_inference_sampling_as_standalone_file (not copied per requirements).
  """
  num_tokens, vocab_size = shape

  # Create mesh for vLLM sample function
  mesh = Mesh(
    np.array([jax.devices()[0]]), axis_names=(ShardingAxisName2D.ATTN_DATA,)
  )

  # Split main seed into all needed keys
  key = jax.random.PRNGKey(seed)
  key, topk_key, topp_key, temp_key, logits_key, sample_key = jax.random.split(
    key, 6
  )

  # Create sampling metadata with varying top_k, top_p, and temperature
  # We use varying k and temperatures of 10**normal(0,1) so that sometimes random gumbel noise dominates,
  # sometimes logits values dominates. Similarly, varying p threshold in top-p
  tpu_sampling_metadata = TPUSupportedSamplingMetadata(
    top_k=jax.random.randint(topk_key, (num_tokens,), 1, 128, dtype=jnp.int32),
    top_p=jax.random.uniform(topp_key, (num_tokens,), dtype=jnp.float32),
    temperature=10
    ** jax.random.normal(temp_key, (num_tokens,), dtype=jnp.float32),
    do_sampling=True,
  )

  # Generate logits based on case
  logits = jax.random.normal(logits_key, shape).astype(dtype)
  if case == "worst_case":
    logits = logits.at[:, 13::256].add(100)

  logits = jax.vmap(uniquely_define_topk)(logits, tpu_sampling_metadata.top_k)

  # Run both implementations
  pallas_result = topk_topp_and_sample(
    sample_key, logits, tpu_sampling_metadata
  )

  vllm_result = vllm_sample(sample_key, mesh, logits, tpu_sampling_metadata)

  # Compare results - expect exact match
  np.testing.assert_array_equal(
    pallas_result,
    vllm_result,
    err_msg=f"Pallas sampling should exactly match vLLM sampling for "
    f"shape={shape}, dtype={dtype}, case={case}, seed={seed}",
  )


if __name__ == "__main__":
  print("Running topk_topp_and_sample tests...")

  shapes = [(16, 16384), (13, 11792)]
  dtypes = [jnp.bfloat16, jnp.float32]
  cases = ["random", "worst_case"]
  seeds = [42, 123, 456]

  for shape in shapes:
    for dtype in dtypes:
      for case in cases:
        for seed in seeds:
          print(
            f"\nTesting shape={shape}, dtype={dtype}, case={case}, seed={seed}..."
          )
          test_topk_topp_and_sample(shape, dtype, case, seed)
          print("  ✓ Passed")

  print("\nAll topk_topp_and_sample tests passed!")
