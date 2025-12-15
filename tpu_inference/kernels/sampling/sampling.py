# Copyright 2025 The TpuInference Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""High-level sampling interface combining top-k and top-p sampling.

Extracted from tallax: https://github.com/oliverdutton/tallax
"""

import jax
import jax.numpy as jnp
from jax.experimental.custom_partitioning import custom_partitioning
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from tpu_inference.kernels.sampling.bitonic_topk import bitonic_topk
from tpu_inference.kernels.sampling.divide_and_filter_topk import top_dynamic_k
from tpu_inference.kernels.sampling.top_p_and_sample import top_p_and_sample
from tpu_inference.kernels.sampling.utils import NUM_LANES


def _topk_with_sharding(logits: jax.Array, k: jax.Array, replace_val):
    def _topk_arrays(logits: jax.Array, k: jax.Array):
      if logits.shape[-1] <= 4096:
        # for small sizes just do direct top-k. Constant runtime
        idxs = jax.lax.broadcasted_iota(jnp.int32, logits.shape, 1)
        topk_logits, topk_idxs = bitonic_topk([logits, idxs], NUM_LANES)
        topk_logits = jnp.where(
          jnp.arange(NUM_LANES)[None, :] < k[:,None],
          topk_logits,
          replace_val
        )
        return topk_logits, topk_idxs
      return top_dynamic_k(
        logits,
        k=k,
        max_k=NUM_LANES,
        guarantee_convergence=True,
        num_bins=256,
        bins_topm_schedule=(5,9),
        replace_val=replace_val)

    @custom_partitioning
    def sharded_topk(logits, k):
      return _topk_arrays(logits, k)

    def infer_sharding_from_operands(mesh, arg_shapes, result_shape):
      logits_spec = arg_shapes[0].sharding.spec
      return (NamedSharding(mesh, P(logits_spec[0], None)),) * 2

    def partition(mesh, arg_shapes, out_shapes):
      arg_shardings, out_shardings = jax.tree.map(lambda s: s.sharding,
        (arg_shapes, out_shapes))
      axis_name = arg_shardings[0].spec[1]

      def shmap_fn(logits, k):
        topk_logits, topk_idxs = _topk_arrays(logits, NUM_LANES)
        if axis_name is None:
          return topk_logits, topk_idxs
        # convert idxs to global frame
        i = jax.lax.axis_index(axis_name)
        topk_idxs += i * logits.shape[1]
        # all-gather and top-k
        operands = [jax.lax.all_gather(x, axis_name, axis=1) for x in (topk_logits, topk_idxs)]
        topk_logits, topk_idxs = bitonic_topk(operands, k=NUM_LANES)
        topk_logits = jnp.where(
          jax.lax.broadcasted_iota(jnp.int32, topk_logits.shape, 1) < k[:,None],
          topk_logits,
          replace_val
        )
        return topk_logits, topk_idxs
      return mesh, shmap_fn, out_shardings, arg_shardings

    sharded_topk.def_partition(
      infer_sharding_from_operands=infer_sharding_from_operands,
      partition=partition,
      sharding_rule='b v, b -> b k, b k',
    )
    return sharded_topk(logits, k)


def sample(rng_key, logits, tpu_sampling_metadata):
  """Main sampling function combining top-k and top-p filtering.

  Args:
      rng_key: Random key for sampling
      logits: Logits of shape (batch_size, vocab_size)
      tpu_sampling_metadata: Metadata containing top_k, top_p, and temperature

  Returns:
      Sampled token indices of shape (batch_size,)
  """
  vocab_size = logits.shape[1]
  topk_logits, topk_idxs = _topk_with_sharding(
    logits,
    k=tpu_sampling_metadata.top_k,
    replace_val=-1e12)
  return top_p_and_sample(
    topk_logits, topk_idxs,
    rng_key,
    top_p=tpu_sampling_metadata.top_p,
    temperature=tpu_sampling_metadata.temperature,
    vocab_size=vocab_size,
    replace_val=-1e12)
