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

"""Top-K selection with dynamic convergence.

Extracted from tallax: https://github.com/oliverdutton/tallax
"""

import functools
import jax
import jax.numpy as jnp
from jax import jit
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from tpu_inference.kernels.sampling.bitonic_topk import bitonic_topk_refs, bitonic_topk_arrays
from tpu_inference.kernels.sampling.topk_convergence_theory import calculate_depth_thresholds
from tpu_inference.kernels.sampling.utils import unrolled_fori_loop, NUM_LANES, NUM_SUBLANES, pad, log2, get_dtype_info, iota_tile, to_32bit_dtype

def binned_topk(
    logits,
    k: int,
    bins_topk_vals,
    bins_topk_idxs,
    completed_k: int = 0,
    num_bins: int = NUM_LANES,
    unroll: int = 32,
):
  """
  Compute binned top-k using a sinking sort approach.

  Processes the vocabulary in num_bins-sized chunks, maintaining the top-k elements
  across all processed bins using a sinking sort algorithm. Values "sink" through
  the maintained top-k list if they are smaller than existing elements.

  Args:
      logits: Input logits of shape [num_tokens, vocab_size].
      k: Number of top elements to find.
      bins_topk_vals: List of k arrays, each of shape [num_tokens, num_bins],
          containing current top-k values per bin.
      bins_topk_idxs: List of k arrays, each of shape [num_tokens, num_bins],
          containing current top-k indices per bin.
      completed_k: Number of top-k positions already finalized (default: 0).
      num_bins: Number of bins/lanes to process simultaneously (default: 128).
      unroll: Loop unroll factor for the vocabulary scan (default: 32).

  Returns:
      Tuple of (bins_topk_vals, bins_topk_idxs) with updated top-k values and indices.
  """
  num_tokens, vocab_size = logits.shape

  def update_bins_topk(bubble_vals, bubble_idxs, bins_topk_vals, bins_topk_idxs):
    """
    Update bins topk with bubble vals/idxs using sinking sort.

    Compares new values against existing top-k, swapping when new values are larger.
    Already-completed positions are invalidated to prevent re-selection.
    """
    # Sinking sort: compare and swap
    for i in range(completed_k):
      # Invalidate already-found elements
      # We use the idxs list to check identity
      bubble_vals = jnp.where(
          bubble_idxs == bins_topk_idxs[i],
          get_dtype_info(bubble_vals).min,
          bubble_vals
      )
    for i in range(completed_k, k):
      # Exchange with stored top-k
      # Only perform the swap if the value is larger
      mask = bubble_vals > bins_topk_vals[i]
      bins_topk_vals[i], bubble_vals = (
          jnp.where(m, bubble_vals, bins_topk_vals[i])
          for m in (mask, ~mask)
      )
      bins_topk_idxs[i], bubble_idxs = (
          jnp.where(m, bubble_idxs, bins_topk_idxs[i])
          for m in (mask, ~mask)
      )
    return (bins_topk_vals, bins_topk_idxs)

  def compute_idxs(i):
    """Compute global vocabulary indices for bin slice i."""
    shape = (num_tokens, num_bins)
    return (
      jnp.full(shape, i * num_bins, jnp.int32) +
      jax.lax.broadcasted_iota(jnp.int32, shape, 1))

  def loop_body(i, bins_topk_outs):
    vals = logits[..., pl.dslice(num_bins * i, num_bins)]
    idxs = compute_idxs(i)
    return update_bins_topk(vals, idxs, *bins_topk_outs)

  num_full_slices = vocab_size // num_bins
  bins_topk_outs = unrolled_fori_loop(
      num_full_slices,
      loop_body,
      (bins_topk_vals, bins_topk_idxs),
      unroll=unroll,
  )

  # Handle remaining elements if vocab_size doesn't divide num_bins
  remainder = vocab_size % num_bins
  if remainder > 0:
    # Load the final boundary segment
    final_vals = logits[..., pl.dslice(num_full_slices * num_bins, remainder)]
    # Pad to num_bins with f32 min
    final_vals = pad(final_vals, (1, num_bins), val='min')
    # Create idxs for the final segment
    final_idxs = compute_idxs(num_full_slices)
    # Update bins topk with the overspill
    bins_topk_outs = update_bins_topk(final_vals, final_idxs, *bins_topk_outs)
  return bins_topk_outs

def _merge_unconverged_bins_topk(
    logits_ref,
    bins_topm_vals_ref,
    bins_topm_idxs_ref,
    *,
    num_bins: int,
    m: int,
    max_k: int,
):
  """Compute top-k from most active bins and merge with unconverged bins."""

  # Derive block_token from logits_ref shape
  block_token = logits_ref.shape[0]

  # Derive num_packed_bins from max_k and m
  # Compute smallest power of 2 >= ceil(max_k / (m - 1))
  num_packed_bins = 2**log2(pl.cdiv(max_k, m - 1))

  # Count contribution of each bin to top-k
  # bins_topm_vals has shape (block_token, m * num_bins)
  # We want to count how many values in each bin are >= pivot
  pivot = bins_topm_vals_ref[:, pl.dslice((m - 1) * num_bins, num_bins)].max(-1, keepdims=True)

  # Count contributions per bin across the m-1 top bins
  # Shape: (block_token, num_bins)
  num_gt_k = jnp.zeros((block_token, num_bins), dtype=jnp.int32)
  for i in range(m - 1):
    bin_vals = bins_topm_vals_ref[:, pl.dslice(i * num_bins, num_bins)]
    num_gt_k += (bin_vals >= pivot).astype(jnp.int32)

  # Use bitonic_topk_arrays descending to get bin indices ordered by contribution count
  bin_indices = jax.lax.broadcasted_iota(jnp.int32, (block_token, num_bins), 1)
  # Sort descending by num_gt_k to get top NUM_LANES bin indices
  _, sorted_bin_indices = bitonic_topk_arrays([num_gt_k, bin_indices], k=NUM_LANES, num_keys=1)
  # Repeat first num_packed_bins values across NUM_LANES positions to create packing permutation
  packing_perm = jnp.take_along_axis(sorted_bin_indices, iota_tile(1) % num_packed_bins, axis=1)

  # produce the (block_token, num_bins) mask
  # index[t, b] = b (the bin index in the second dimension)
  index = jax.lax.broadcasted_iota(jnp.int32, (block_token, num_bins), 1)
  indicator = jnp.zeros((block_token, num_bins), dtype=jnp.bool_)
  for i in range(num_packed_bins):
    # Mark positions where bin index matches the i-th active bin
    indicator |= (index == packing_perm[:, i:i+1])

  bins_topm_vals_ref[...] = jnp.concat([
      jnp.where(
          indicator, get_dtype_info(bins_topm_vals_ref).min,
          bins_topm_vals_ref[:, i * num_bins:(i+1) * num_bins])
      for i in range(bins_topm_vals_ref.shape[1] // num_bins)], axis=1)

  # Loop over blocks and pack data from active bins
  vocab_size = logits_ref.shape[1]
  packed_vals = [jnp.full(
      (block_token, NUM_LANES),
      get_dtype_info(logits_ref).min, dtype=logits_ref.dtype
  ) for _ in range(pl.cdiv(vocab_size, NUM_LANES * (num_bins // num_packed_bins)))]

  for offset in range(0, num_bins, NUM_LANES):
    local_perm = (packing_perm - offset) % NUM_LANES
    in_range_mask = (packing_perm >= offset) & (packing_perm < (offset + NUM_LANES))

    # Extract values from all full bins at this offset
    vals = [logits_ref[:, pl.dslice(start_idx, NUM_LANES)].astype(to_32bit_dtype(logits_ref.dtype)) for start_idx in range(offset, vocab_size, num_bins)]

    # apply permutation
    vals = [jnp.take_along_axis(tile, local_perm, axis=1) for tile in vals]
    # Pack into positions based on active bin index
    index = iota_tile(1)
    for i in range(NUM_LANES // num_packed_bins):
      pack_mask = (
          (index >= i * num_packed_bins) &
          (index < (i + 1) * num_packed_bins) &
          in_range_mask
      )
      # Pack every num_packed_bins-th chunk starting from i
      for j, v in enumerate(vals[i::NUM_LANES//num_packed_bins]):
        packed_vals[j] = jnp.where(pack_mask, v, packed_vals[j])

  packed_vals = jnp.concat(packed_vals, axis=1)
  n = packed_vals.shape[1]

  packed_idxs = (jax.lax.broadcasted_iota(jnp.int32, packed_vals.shape, 1) // num_packed_bins) * num_bins + jnp.concat(
    (packing_perm,)*(n//NUM_LANES), axis=1)

  # we calculate the top 128 vals from the packed bins and a piece of bins_topm_(val/idx)s we overwrite
  # Build input arrays by concatenating packed vals and the top NUM_LANES values
  val_input = jnp.concat([packed_vals, bins_topm_vals_ref[:, :NUM_LANES]], axis=1)
  idx_input = jnp.concat([packed_idxs, bins_topm_idxs_ref[:, :NUM_LANES]], axis=1)
  (
    bins_topm_vals_ref[:, :NUM_LANES],
    bins_topm_idxs_ref[:, :NUM_LANES]
  ) = bitonic_topk_arrays([val_input, idx_input], k=NUM_LANES, num_keys=1)


def dynamic_topk_refs(
    logits_ref,
    k_smem_ref,
    k_vmem_ref,
    topk_vals_ref,
    topk_idxs_ref,
    valid_ref,
    max_depth_ref,
    cutoff_vals_ref,
    # scratch
    bins_topm_vals_ref,
    bins_topm_idxs_ref,
    termination_flag_ref,
    *,
    max_k: int,
    num_bins: int,
    bins_topm_unroll: int,
    bins_topm_schedule: tuple[int, ...],
    guarantee_convergence: bool,
    replace_val: float | int | None,
):
  """
  Pallas kernel for computing binned top-k supersets until global top-k is guaranteed.

  Incrementally computes top-m supersets (m increasing per schedule) until the top-k
  is provably contained within the top-(m-1) bins. Supports dynamic k per token while
  using static max_k for compilation and scheduling.

  The termination criterion checks if the top-(m-1) bins collectively contain at least
  k values larger than the largest m-th largest value across all bins.
  """
  # Initialize buffers
  block_token = logits_ref.shape[0]
  shape = (block_token, bins_topm_vals_ref.shape[1])

  pid = pl.program_id(0)
  token_slice = pl.dslice(pid * block_token, block_token)

  bins_topm_vals_ref[token_slice] = jnp.full(
      shape, get_dtype_info(logits_ref).min, dtype=bins_topm_vals_ref.dtype
  )

  for i in range(block_token):
    max_depth_ref[pid * block_token + i] = max_k
  termination_flag_ref[0] = 0

  # Incremental binned top-k computation
  for completed_m, m in zip(bins_topm_schedule, bins_topm_schedule[1:]):
    @pl.when(termination_flag_ref[0] == 0)
    def _():
      # Compute binned top-m
      bins_topm_vals, bins_topm_idxs = binned_topk(
          logits_ref,
          k=m,
          bins_topk_vals=[
              bins_topm_vals_ref[
                  token_slice, pl.dslice(i * num_bins, num_bins)
              ].astype(to_32bit_dtype(logits_ref.dtype))
              for i in range(m)
          ],
          bins_topk_idxs=[
              bins_topm_idxs_ref[
                  token_slice, pl.dslice(i * num_bins, num_bins)
              ]
              for i in range(m)
          ],
          num_bins=num_bins,
          completed_k=completed_m,
          unroll=bins_topm_unroll,
      )

      # Store results
      for i in range(completed_m, m):
        bins_topm_vals_ref[
            token_slice, pl.dslice(i * num_bins, num_bins)
        ] = bins_topm_vals[i].astype(bins_topm_vals_ref.dtype)
        bins_topm_idxs_ref[
            token_slice, pl.dslice(i * num_bins, num_bins)
        ] = bins_topm_idxs[i].astype(bins_topm_idxs_ref.dtype)

      # Termination criterion:
      # If top-(m-1) bins contain >= k vals larger than
      # the largest m-th largest value, then top-k is guaranteed to be in bins
      # top-(m-1) collated
      pivot = bins_topm_vals[m - 1].max(-1, keepdims=True)
      num_larger = (
          sum([(v >= pivot) for v in bins_topm_vals[:m - 1]])
          .astype(to_32bit_dtype(logits_ref.dtype))
          .sum(-1)
      )

      termination_flag_ref[0] = 0
      for i in range(block_token):
        token_idx = pid * block_token + i
        # Dynamic check against k
        contains_topk = num_larger[i] >= k_smem_ref[token_idx]
        termination_flag_ref[0] += contains_topk

        # Record depth when criterion was met
        current_max = max_depth_ref[token_idx]
        max_depth_ref[token_idx] = jnp.where(
            contains_topk & (current_max == max_k),
            m - 1,
            current_max
        )
        # Record largest m-th largest value
        # Useful for bounds checking if running sharded topk
        cutoff_vals_ref[token_idx] = pivot.squeeze(1)[i]

      # Check if all tokens converged
      @pl.when(termination_flag_ref[0] != block_token)
      def _():
        termination_flag_ref[0] = 0

  # Bin packing optimization for non-convergence cases
  m_final = bins_topm_schedule[-1]
  @pl.when(guarantee_convergence & (m_final != max_k) & (termination_flag_ref[0] == 0))
  def _():
    # This optimization applies when guarantee_convergence is enabled but
    # we haven't fully converged (m_final != max_k) and termination criterion not met.
    # Packs the most active bins to help converge.
    _merge_unconverged_bins_topk(
        logits_ref,
        bins_topm_vals_ref.at[token_slice],
        bins_topm_idxs_ref.at[token_slice],
        num_bins=num_bins,
        m=m_final,
        max_k=max_k
    )

  global_topk_schedule = tuple(sorted(set(2**log2(x - 1) if x >1 else x for x in bins_topm_schedule)))

  # Final top-k extraction (done by last program)
  @pl.when(pl.program_id(0) == (pl.num_programs(0) - 1))
  def _():
    # Find maximum depth across all tokens
    global_max_depth = jnp.array(0)
    for i in range(max_depth_ref.shape[0]):
      global_max_depth = jnp.maximum(global_max_depth, max_depth_ref[i])

    valid_ref[0] = ((
    global_max_depth < bins_topm_schedule[-1]
    ) | (bins_topm_schedule[-1] == max_k)
    ).astype(jnp.int32)

    # Use appropriate sorting depth based on global_max_depth
    for depth_lower, depth_upper in zip(global_topk_schedule, global_topk_schedule[1:]):
      @pl.when((
      (global_max_depth > depth_lower) & (global_max_depth <= depth_upper)
      ) | (
      # Sort to give approx topk if not fully converged
      (depth_upper == global_topk_schedule[-1]) & (global_max_depth > depth_upper)
      ))
      def _():
        # Sort the binned superset
        bitonic_topk_refs(
          [ref.at[:, :depth_upper * num_bins]
            for ref in (bins_topm_vals_ref, bins_topm_idxs_ref)],
          [topk_vals_ref, topk_idxs_ref],
          num_keys=1,
          descending=True,
        )
        if replace_val is not None:
          idx = jax.lax.broadcasted_iota(jnp.int32, topk_vals_ref.shape, 1)
          topk_vals_ref[...] = jnp.where(
            idx < k_vmem_ref[...][:, None],
            topk_vals_ref[...], replace_val)


@functools.partial(
    jit,
    static_argnames=(
        "max_k",
        "block_token",
        "num_bins",
        "bins_topm_unroll",
        "bins_topm_schedule",
        "guarantee_convergence",
        "replace_val",
        "interpret"
    ),
)
def top_dynamic_k(
    logits,
    k,
    max_k: int,
    block_token: int = 8,
    num_bins: int = NUM_LANES,
    bins_topm_unroll: int = 32,
    bins_topm_schedule: tuple[int, ...] | None = None,
    guarantee_convergence: bool = False,
    replace_val: float | int | None = None,
    interpret: bool = False,
):
  """
  High-level interface for adaptive binned top-k computation on TPU.

  Supports dynamic k per token (each token can have a different k value) while
  maintaining efficient TPU execution through static compilation based on max_k.
  Automatically computes optimal search schedules if not provided.

  Args:
      logits: Input logits of shape [num_tokens, vocab_size].
      k: Per-token k values. Can be scalar (broadcast to all tokens) or array
          of shape [num_tokens].
      max_k: Static maximum k across all tokens. Used for buffer sizing and
          compilation. Must be >= all values in k.
      block_token: Number of tokens processed per program block (default: 8).
          Must evenly divide num_tokens.
      num_bins: Number of bins for parallel binned operations (default: 128).
      bins_topm_unroll: Loop unroll factor for binned top-m inner loop (default: 32).
      bins_topm_schedule: Increasing sequence of m values for incremental top-m search.
          If None, automatically computed based on convergence probability thresholds.
      guarantee_convergence: If True, adds max_k to schedule to ensure full convergence
          and enables bin packing optimization for rare non-convergence cases (default: False).
      interpret: If True, run in CPU interpret mode instead of TPU compilation (default: False).

  Returns:
      When guarantee_convergence=False:
          Tuple of (topk_vals, topk_idxs, valid, depths, cutoff_vals):
          - topk_vals: Top-k values of shape [num_tokens, max_k].
          - topk_idxs: Top-k indices of shape [num_tokens, max_k].
          - valid: Boolean indicating if algorithm fully converged.
          - depths: Per-token convergence depth of shape [num_tokens].
          - cutoff_vals: Per-token pivot values of shape [num_tokens].
      When guarantee_convergence=True:
          Tuple of (topk_vals, topk_idxs):
          - topk_vals: Top-k values of shape [num_tokens, max_k].
          - topk_idxs: Top-k indices of shape [num_tokens, max_k].
  """
  num_tokens, vocab_size = logits.shape

  if num_tokens % block_token != 0:
    raise ValueError("num_tokens must be divisible by block_token")
    
  if max_k > NUM_LANES:
    raise NotImplementedError

  k = jnp.broadcast_to(k, (num_tokens,))

  # Auto-compute schedules if not provided
  if bins_topm_schedule is None:
    thresholds = calculate_depth_thresholds(max_k, num_bins, block_token, target_yields=(0.8, 0.98, 0.9999))
    bins_topm_schedule = tuple(t + 1 for t in thresholds)
    print(f"Auto-computed schedules for max_k={max_k}, num_bins={num_bins}:")
    print(f"  bins_topm_schedule: {bins_topm_schedule}")
  bins_topm_schedule = tuple(sorted(set(bins_topm_schedule)))
  bins_topm_schedule = (0,) + bins_topm_schedule

  # binned topk / sort pad len
  max_m = bins_topm_schedule[-1]
  buffer_size = max(max_m, 2**log2(max_m - 1)) * num_bins

  # Updated padded size calculation using num_bins
  padded_max_k = pl.cdiv(max_k, NUM_LANES) * NUM_LANES

  output_shapes = (
      jax.ShapeDtypeStruct((num_tokens, padded_max_k), logits.dtype),
      jax.ShapeDtypeStruct((num_tokens, padded_max_k), jnp.int32),
      jax.ShapeDtypeStruct((1,), jnp.int32),
      jax.ShapeDtypeStruct((num_tokens,), jnp.int32),
      jax.ShapeDtypeStruct((num_tokens,), to_32bit_dtype(logits.dtype)),
  )

  output_specs = (
      pl.BlockSpec(),
      pl.BlockSpec(),
      pl.BlockSpec(memory_space=pltpu.SMEM),
      pl.BlockSpec(memory_space=pltpu.SMEM),
      pl.BlockSpec(memory_space=pltpu.SMEM),
  )

  # Add scratch shapes
  scratch_shapes = [
      pltpu.VMEM((num_tokens, buffer_size), to_32bit_dtype(logits.dtype)),
      pltpu.VMEM((num_tokens, buffer_size), jnp.int32),
      pltpu.SMEM((1,), jnp.int32),
  ]

  outputs = pl.pallas_call(
      functools.partial(
          dynamic_topk_refs,
          max_k=max_k,
          num_bins=num_bins,
          bins_topm_unroll=bins_topm_unroll,
          bins_topm_schedule=bins_topm_schedule,
          guarantee_convergence=guarantee_convergence,
          replace_val=replace_val,
      ),
      in_specs=(
          pl.BlockSpec((block_token, vocab_size), lambda i: (i, 0)),
          # for TPU Pallas lowering reasons it's convenient to have both SMEM and VMEM k
          pl.BlockSpec(memory_space=pltpu.SMEM),
          pl.BlockSpec(memory_space=pltpu.VMEM),
      ),
      out_shape=output_shapes,
      scratch_shapes=tuple(scratch_shapes),
      grid=(num_tokens // block_token,),
      out_specs=output_specs,
      compiler_params=pltpu.CompilerParams(
        vmem_limit_bytes=int(0.9 * 2**27)
      ),
      interpret=interpret,
  )(logits, k, k)
  topk_vals, topk_idxs, valid, depths, cutoff_vals = outputs

  topk_vals, topk_idxs = (x[:,:max_k] for x in (topk_vals, topk_idxs))
  valid = valid.squeeze().astype(bool)

  if guarantee_convergence:
    return topk_vals, topk_idxs
  return topk_vals, topk_idxs, valid, depths, cutoff_vals

@functools.partial(
    jit,
    static_argnames=(
        "k",
        "block_token",
        "num_bins",
        "bins_topm_unroll",
        "bins_topm_schedule",
        "interpret"
    ),
)
def topk(
    logits,
    k: int,
    block_token: int = NUM_SUBLANES,
    num_bins: int = NUM_LANES,
    bins_topm_unroll: int = 32,
    bins_topm_schedule: tuple[int, ...] | None = None,
    interpret: bool = False,
):
  """
  Compute top-k elements with guaranteed convergence.

  Simplified interface for uniform k across all tokens. Automatically ensures
  convergence by setting guarantee_convergence=True internally.

  Args:
      logits: Input logits of shape [num_tokens, vocab_size].
      k: Number of top elements to find (uniform across all tokens).
      block_token: Number of tokens processed per program block (default: 8).
      num_bins: Number of bins for parallel operations (default: 128).
      bins_topm_unroll: Loop unroll factor for inner loop (default: 32).
      bins_topm_schedule: Optional custom search schedule. If None, automatically
          computed.
      interpret: If True, run in CPU interpret mode (default: False).

  Returns:
      Tuple of (topk_vals, topk_idxs):
          - topk_vals: Top-k values of shape [num_tokens, k].
          - topk_idxs: Top-k indices of shape [num_tokens, k].
  """
  return top_dynamic_k(
    logits,
    k=k,
    max_k=k,
    block_token=block_token,
    num_bins=num_bins,
    bins_topm_unroll=bins_topm_unroll,
    bins_topm_schedule=bins_topm_schedule,
    guarantee_convergence=True,
    interpret=interpret,
  )