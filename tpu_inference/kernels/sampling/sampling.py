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

"""Fused TPU sampling kernel implementing top-p filtering, temperature scaling,
and categorical sampling in a single Pallas kernel.

Extracted from tallax: https://github.com/oliverdutton/tallax
"""

import functools
import jax
import jax.numpy as jnp
from jax import jit, lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from jax.experimental.custom_partitioning import custom_partitioning
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from tpu_inference.kernels.sampling.bitonic_topk import bitonic_topk_arrays, bitonic_topk, max_arrays
from tpu_inference.kernels.sampling.gather import take_along_axis_arrays
from tpu_inference.kernels.sampling.sparse_random import sparse_random_categorical
from tpu_inference.kernels.sampling.cumsum import cumsum_arrays
from tpu_inference.kernels.sampling.topk import top_dynamic_k
from tpu_inference.kernels.sampling.utils import NUM_LANES, NUM_SUBLANES, pad, log2, iota_tile, transpose_list_of_lists

_SAMPLING_EPS = 1e-5


def top_p_mask(*, topk_logits, p, replace_val, axis):
    """
    Apply top-p filtering mask to sorted logits.

    Args:
        topk_logits: Sorted logits (descending order)
        p: Top-p threshold(s)
        replace_val: Value to replace filtered logits with
        axis: Axis along which to apply filtering (must be 0)

    Returns:
        Masked logits with values outside top-p set to replace_val
    """
    if axis != 0:
        raise NotImplementedError("topp_mask only supports axis=0")

    shape = topk_logits.shape

    # Compute softmax probabilities
    # For numerical stability, subtract max (pre-sorted so its the first element)
    exp_logits = jnp.exp(topk_logits - topk_logits[:1,:])
    probs = exp_logits / exp_logits.sum(axis=0, keepdims=True)

    # Top-p filtering using cumsum on sorted probabilities
    cumsum_probs = cumsum_arrays(probs, axis=0)

    # Find last idx where top-p probability mass is not covered
    threshold_idx = (cumsum_probs < p[None,:]).sum(0, keepdims=True)
    # vLLM current implementation uses binary search, computing a threshold.
    # so ties at the threshold are all included
    # we replicate that behavior here
    thresholds = take_along_axis_arrays(
        topk_logits, jnp.broadcast_to(threshold_idx, shape), 0)
    topp_logits = jnp.where(
        topk_logits >= thresholds,
        topk_logits, replace_val)

    return topp_logits


def top_p_and_sample_arrays(*, topk_logits, topk_idx, rng_key, top_p, temperature, vocab_size, replace_val, dim0_offset: int = 0):
    """
    Implements top-p filtering, temperature scaling, and sampling.

    Args:
        topk_logits: Sorted logits of shape (batch_size, k)
        topk_idx: Indices corresponding to sorted logits of shape (batch_size, k)
        rng_key: RNG key for sampling, shape (1, 2)
        top_p: Top-p threshold values, shape (batch_size,)
        temperature: Temperature values, shape (batch_size,)
        vocab_size: Vocabulary size for sampling
        replace_val: Value to replace filtered logits with
        dim0_offset: Offset for dim0 (batch) axis, used for sharding (default: 0)

    Returns:
        Sampled tokens of shape (batch_size,)
    """
    topk_logits = topk_logits.astype(jnp.float32)

    # To do reductions and broadcast across sublanes rather than lanes (which are slow)
    # we shift sampling to dim 0
    topk_logits = topk_logits.T
    topk_idx = topk_idx.T
    shape = topk_logits.shape

    topp_logits = top_p_mask(
        topk_logits=topk_logits,
        p=top_p,
        replace_val=replace_val,
        axis=0
    )

    topp_logits_scaled = topp_logits / temperature[None,:].astype(topp_logits.dtype)

    # random key splitting is based on idx in ravelled array
    # we pass in (batch_idx.T, token_idx.T) and sample across axis 0, taking the token_idx
    batch_idx = lax.broadcasted_iota(jnp.int32, shape, 1) + dim0_offset
    next_tokens = sparse_random_categorical(
        rng_key,
        topp_logits_scaled,
        # these are both transposed, (token, batch) shape
        (batch_idx, topk_idx),
        dim1_size=vocab_size,
        axis=0,
        dtype=jnp.float32
        # take sampled_indices[1], the token idx
    )[1]
    greedy_sampled = topk_idx[0,:]
    return jnp.where(
      temperature < _SAMPLING_EPS,
      greedy_sampled, next_tokens)


def top_p_and_sample_refs(
    topk_logits_ref,
    topk_idx_ref,
    rng_key_ref,
    top_p_ref,
    temperature_ref,
    dim0_offset_ref,
    sampled_tokens_ref,
    *,
    vocab_size: int,
    replace_val: float,
):
    """
    Fused kernel implementing top-p filtering, temperature scaling, and sampling.

    Args:
        topk_logits_ref: Reference to sorted logits
        topk_idx_ref: Reference to sorted indices
        rng_key_ref: Reference to RNG key (SMEM)
        top_p_ref: Reference to top-p values
        temperature_ref: Reference to temperature values
        dim0_offset_ref: Reference to dim0 offset for sharding (SMEM, shape (1,))
        sampled_tokens_ref: Reference to output sampled tokens
        vocab_size: Vocabulary size
        replace_val: Value to replace filtered logits with
    """
    sampled_tokens_ref[...] = top_p_and_sample_arrays(
      topk_logits=topk_logits_ref[...],
      topk_idx=topk_idx_ref[...],
      rng_key=rng_key_ref, # SMEM, so keep as ref
      top_p=top_p_ref[...],
      temperature=temperature_ref[...],
      vocab_size=vocab_size,
      replace_val=replace_val,
      dim0_offset=dim0_offset_ref[0], # Extract scalar from SMEM array
    )

def _top_p_and_sample(
    topk_logits: jax.Array,
    topk_idx: jax.Array,
    rng_key: jax.Array, # threefry2x32 key
    top_p: jax.Array,
    temperature: jax.Array,
    *,
    vocab_size: int,
    replace_val: float,
    interpret: bool = False,
    dim0_offset: int = 0,
) -> jax.Array:
    """
    Fused TPU kernel for sampling with top-p filtering and temperature scaling.

    Args:
        topk_logits: Sorted logits of shape (batch_size, k)
        topk_idx: Indices corresponding to sorted logits of shape (batch_size, k)
        rng_key: RNG key for sampling, shape (2,)
        top_p: Top-p threshold values, scalar or shape (batch_size,)
        temperature: Temperature values, scalar or shape (batch_size,)
        vocab_size: Vocabulary size for sampling
        replace_val: Value to replace filtered logits with
        interpret: If True, run in CPU interpret mode (default: False)
        dim0_offset: Offset for dim0 (batch) axis, used for sharding (default: 0)
                     Must be computed outside pallas_call using lax.axis_index

    Returns:
        next_tokens: Sampled tokens of shape (batch_size,)
    """
    return pl.pallas_call(
        functools.partial(
          top_p_and_sample_refs,
          vocab_size=vocab_size,
          replace_val=replace_val,
        ),
        in_specs=(
          pl.BlockSpec(),
          pl.BlockSpec(),
          pl.BlockSpec(memory_space=pltpu.SMEM),
          pl.BlockSpec(),
          pl.BlockSpec(),
          pl.BlockSpec(memory_space=pltpu.SMEM),
        ),
        out_shape=jax.ShapeDtypeStruct(topk_logits.shape[:1], jnp.int32),
        interpret=interpret,
    )(
        topk_logits,
        topk_idx,
        rng_key.reshape(1,2),
        top_p,
        temperature,
        jnp.array(dim0_offset, jnp.int32)[None],
    )

@functools.partial(
    jit,
    static_argnames=("vocab_size", "replace_val", "interpret",),
)
def top_p_and_sample(
    topk_logits: jax.Array,
    topk_idx: jax.Array,
    rng_key: jax.Array,
    top_p: jax.Array,
    temperature: jax.Array,
    *,
    vocab_size: int,
    replace_val: float,
    interpret: bool = False,
) -> jax.Array:
    """
    Sharded wrapper for top-p sampling with custom partitioning.

    Requires all axes except batch dim to be replicated. Batch dim can be sharded.
    """
    @custom_partitioning
    def sharded_top_p_and_sample(topk_logits, topk_idx, rng_key, top_p, temperature):
        return _top_p_and_sample(
            topk_logits, topk_idx, rng_key, top_p, temperature,
            vocab_size=vocab_size, replace_val=replace_val, interpret=interpret
        )

    def infer_sharding_from_operands(mesh, arg_shapes, result_shape):
        # Output follows batch dimension of first input (replicated on other dims)
        batch_spec = arg_shapes[0].sharding.spec[0]
        return NamedSharding(mesh, P(batch_spec))

    def partition(mesh, arg_shapes, out_shapes):
        arg_shardings, out_shardings = jax.tree.map(
            lambda s: s.sharding, (arg_shapes, out_shapes))
        batch_axis_name = arg_shardings[0].spec[0]

        def shmap_fn(topk_logits, topk_idx, rng_key, top_p, temperature):
            # Pass global sharded axis offset to maintain jax.random.categorical sampled values
            dim0_offset = 0
            if batch_axis_name is not None:
                dim0_offset = jax.lax.axis_index(batch_axis_name) * topk_logits.shape[0]
            return _top_p_and_sample(
                topk_logits, topk_idx, rng_key, top_p, temperature,
                vocab_size=vocab_size,
                replace_val=replace_val,
                interpret=interpret,
                dim0_offset=dim0_offset,
            )

        return mesh, shmap_fn, out_shardings, arg_shardings

    sharded_top_p_and_sample.def_partition(
        infer_sharding_from_operands=infer_sharding_from_operands,
        partition=partition,
        sharding_rule='b k, b k, r, b, b -> b',
        need_replication_factors=('k', 'r'),
    )

    return sharded_top_p_and_sample(topk_logits, topk_idx, rng_key, top_p, temperature)

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
