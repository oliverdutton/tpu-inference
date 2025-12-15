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

"""Bitonic Top-K for k=NUM_LANES=128 using compressed transpose format.

This implementation is optimized for TPU with k=128 and works entirely in
compressed transpose format to maximize efficiency of permutation operations.

Algorithm:
- Convert input to compressed transpose format: (num_tokens, vocab) -> (NUM_LANES, num_tokens*chunks)
- Build bitonic sequences using stages 1-6 (so sorted in 64 length chunks)
- Cross-tile merge with max selection, reducing tile count
- Progressive sublane permute merging with decreasing distances
- Convert back to original format

Extracted from tallax: https://github.com/oliverdutton/tallax
"""

import functools
from collections.abc import Sequence

import jax
import jax.numpy as jnp
from jax import jit
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from tpu_inference.kernels.sampling.utils import (
    NUM_LANES,
    NUM_SUBLANES,
    log2,
    flatten,
    ceil_multiple,
    iota_tile,
    pad,
    canonicalize_operand,
    transpose_list_of_lists,
    to_compressed_transpose_format,
    from_compressed_transpose_format,
    to_32bit_dtype,
)
from tpu_inference.kernels.sampling.sort import (
    run_compressed_transpose_format_substages_on_tiles,
    compare_and_swap,
)

def max_arrays(operands, num_keys, axis):
  """Compute max over several operands, sorting using num_keys.

  This function computes the maximum element along the specified axis for multiple
  operands (e.g., values and indices). When comparing elements, it uses the first
  num_keys operands as sort keys to determine which element is "larger".

  Args:
    operands: List of JAX arrays of the same shape
    num_keys: Number of operands to use as sort keys for comparison
    axis: Axis along which to find the maximum (0 or 1)

  Returns:
    List of 1D arrays containing the maximum element for each operand
  """
  if axis == 1:
    # transpose and run on axis 0
    operands = jax.tree.map(lambda x: x.T, operands)
    axis = 0
  assert axis == 0
  unpadded_shape = operands[0].shape
  padded_dim0 = max(2**log2(unpadded_shape[0]), NUM_SUBLANES)
  operands = [pad(x, (padded_dim0, NUM_LANES), val='min') for x in operands]

  shape = operands[0].shape
  for _ in range(log2(shape[0] // NUM_SUBLANES)):
    lefts, rights = transpose_list_of_lists([jnp.split(arr,2,axis=0) for arr in operands])
    operands = transpose_list_of_lists(compare_and_swap(lefts, rights, num_keys=num_keys, is_descending=True))[0]
  assert operands[0].shape[0] == NUM_SUBLANES
  assert shape[1] % NUM_LANES == 0

  arrs_tiles = [jnp.split(x, shape[1] // NUM_LANES, axis=1) for x in operands]
  for stage in range(log2(NUM_SUBLANES))[::-1]:
    permutation = jnp.bitwise_xor(iota_tile(0), 2**stage)

    # Apply permutation to all tiles
    arrs_tiles_permuted = jax.tree.map(
      lambda tile: jnp.take_along_axis(tile, permutation, axis=0),
      arrs_tiles
    )

    # Compare and merge with permuted values
    outs_tiles = [[] for _ in arrs_tiles]
    for _, (lefts, rights) in enumerate(zip(
          *map(transpose_list_of_lists, (arrs_tiles, arrs_tiles_permuted)),
          strict=True
      )):
        for j, (o, _) in enumerate(compare_and_swap(
            lefts, rights,
            is_descending=True,
            num_keys=num_keys
        )):
          outs_tiles[j].append(o)
    arrs_tiles = outs_tiles
  return [jnp.concatenate(tiles, axis=1)[0,:unpadded_shape[1]] for tiles in arrs_tiles]


def _split_rows(tiles):
  num_rows = NUM_LANES // NUM_SUBLANES
  num_cols = len(tiles) // num_rows
  return [tiles[row*num_cols:(row+1)*num_cols] for row in range(num_rows)]


def _split_actives(tiles):
  num_rows = NUM_LANES // NUM_SUBLANES
  num_cols = len(tiles) // num_rows
  num_active_cols = 2 * (num_cols // 2)
  active = flatten((
    x[:num_active_cols] for x in _split_rows(tiles)
  ))
  remainder = flatten((
    x[num_active_cols:] for x in _split_rows(tiles)
  ))
  return [active, remainder]

def _merge_remainder(merged, remainder):
  return flatten(map(flatten, zip(*map(_split_rows, (merged, remainder)))))


def _compute_padded_shape(unpadded_dim0: int, unpadded_dim1: int) -> tuple[int, int]:
  """Compute padded shape compatible with compressed transpose format requirements.

  The compressed transpose format requires the total number of elements (dim0 * dim1)
  to be a multiple of NUM_LANES^2 (128^2 = 16384). This function finds the minimal
  padded shape that satisfies this constraint while keeping dim0 as a power of 2 between NUM_SUBLANES and NUM_LANES.

  Args:
    unpadded_dim0: Original first dimension size
    unpadded_dim1: Original second dimension size

  Returns:
    Tuple of (padded_dim0, padded_dim1) compatible with compressed transpose format
  """
  if unpadded_dim0 >= NUM_LANES:
    dim0 = ceil_multiple(unpadded_dim0, NUM_LANES)
    dim1 = ceil_multiple(unpadded_dim1, NUM_LANES)
    return (dim0, dim1)

  dim0s = [2**i for i in range(log2(NUM_SUBLANES), log2(NUM_LANES)+1)
    if 2**i >= unpadded_dim0]
  shapes = [
    (dim0, ceil_multiple(unpadded_dim1, (NUM_LANES ** 2) // dim0))
    for dim0 in dim0s]
  # take minimal num elements, larger dim0 on ties
  return sorted(shapes, key=lambda x: (x[0] * x[1], -x[0]))[0]

def _merge_max_crosstile(
    arrs_tiles, dim0, num_keys: int = 1
):
  """Perform crosstile comparison keeping max values.

  Args:
    arrs_tiles: Tuple of lists of tile arrays
    dim0: First dimension size (padded)
    num_keys: Number of sort keys

  Returns:
    Tuple of lists with half the tiles (max halves only), plus remainder if odd
  """
  num_tiles = len(arrs_tiles[0])
  outs_tiles = [[] for t in arrs_tiles]
  for idx in range(0, num_tiles, 2):
    lefts, rights = (
        transpose_list_of_lists(arrs_tiles)[j]
        for j in (idx, idx + 1)
    )
    # Keep only max (left) values, discard min (right)
    for j, (o_left, _) in enumerate(compare_and_swap(
        lefts, rights, is_descending=True, num_keys=num_keys
    )):
      outs_tiles[j].append(o_left)
  return outs_tiles


def bitonic_topk_arrays(operands: list[jax.Array], k: int = NUM_LANES, num_keys: int = 1):
    """
    Progressive bitonic merge for top-k selection.

    Strategy:
    1. Build bitonic sequences (stages 1-6) within tiles
    2. Cross-tile bitonic merge until we reach target tile count
    3. Final progressive merge with lane permutations
    4. Sort final bitonic sequence to descending order

    Args:
        operands: List of JAX arrays of shape (dim0, dim1)
        k: Number of top elements to return (default: NUM_LANES)
        num_keys: Number of sort keys (default: 1)

    Returns:
        List of JAX arrays of shape (original_dim0, k) with top-k elements
    """
    if k > NUM_LANES:
      raise NotImplementedError
    # Compute padded shape that satisfies alignment requirements
    shape = operands[0].shape
    padded_shape = _compute_padded_shape(*shape)
    # Pad both dimensions if needed
    arrs = [pad(op, block_shape=padded_shape, val='min') for op in operands]
    arrs = [x.astype(to_32bit_dtype(x.dtype)) for x in arrs]

    def _topk_arrays(arrs):
      # Convert to compressed transpose format
      arrs_tiles = [to_compressed_transpose_format(arr) for arr in arrs]

      dim0 = arrs[0].shape[0]
      assert dim0 <= NUM_LANES
      log_lanes = log2(NUM_LANES)
      num_merges = log2(shape[1]) - log_lanes
      num_intra_merges = min(
      log2(pl.cdiv(NUM_LANES, dim0)), num_merges)
      # are intra permutations

      # Build bitonic sequences up to length 64 (stage 6)
      for stage in range(1, log_lanes):  # stages 1-6 inclusive
        arrs_tiles = run_compressed_transpose_format_substages_on_tiles(
          arrs_tiles,
          num_substages=stage,
          stage=stage,
          dim0=dim0,
          num_keys=num_keys,
        )

      # Cross-tile merging: reduce tile count by half each iteration
      # Keep merging until we hit target tile count
      for _ in range(num_merges - num_intra_merges):
        # Run substages sorting NUM_LANES but with stage for merging bitonic sequences
        # so different tile sets have different orders.
        has_remainder = ((len(arrs_tiles[0][::16])%2) != 0)
        if has_remainder:
          remainder_arrs_tiles = [
          _split_actives(x)[1] for x in arrs_tiles]
          arrs_tiles = [
          _split_actives(x)[0] for x in arrs_tiles]
        arrs_tiles = run_compressed_transpose_format_substages_on_tiles(
          arrs_tiles,
          num_substages=log_lanes,
          # tile i is different order to tile i+1, so they can be max merged
          stage=log2(NUM_LANES * NUM_LANES // dim0),
          dim0=dim0,
          num_keys=num_keys,
        )

        # Cross-tile comparison: keep max half, discard min half
        arrs_tiles = _merge_max_crosstile(
            arrs_tiles,
            dim0=dim0,
            num_keys=num_keys
        )
        if has_remainder:
          arrs_tiles = [_merge_remainder(*vs) for vs in zip(arrs_tiles, remainder_arrs_tiles)]

      # Progressive intra-tile merging with lane permute
      for i in range(num_intra_merges)[::-1]:
        distance = dim0 * (2**i)
        # Calculate stage based on current merge size
        # Stage = log2(2 * distance * dim0 / NUM_LANES * NUM_LANES) = log2(2 * distance)
        arrs_tiles = run_compressed_transpose_format_substages_on_tiles(
          arrs_tiles,
          num_substages=log_lanes,
          stage=log_lanes+i,
          dim0=dim0,
          num_keys=num_keys,
        )

        # Create permutation indices for tiles using iota_tile
        permutation = jnp.bitwise_xor(iota_tile(1), distance)

        # Apply permutation to all tiles
        arrs_tiles_permuted = jax.tree.map(
          lambda tile: jnp.take_along_axis(tile, permutation, axis=1),
          arrs_tiles
        )

        # Compare and merge with permuted values
        outs_tiles = [[] for _ in arrs_tiles]
        for _, (lefts, rights) in enumerate(zip(
              *map(transpose_list_of_lists, (arrs_tiles, arrs_tiles_permuted)),
              strict=True
          )):
            for j, (o, _) in enumerate(compare_and_swap(
                lefts, rights,
                is_descending=True,
                num_keys=num_keys
            )):
              outs_tiles[j].append(o)
        arrs_tiles = outs_tiles

      # Final sort: convert bitonic sequence to fully descending order
      # Use dim1_offset=2**7 to ensure descending direction
      arrs_tiles = run_compressed_transpose_format_substages_on_tiles(
        arrs_tiles,
        num_substages=log_lanes,
        stage=log_lanes,
        dim1_offset=NUM_LANES,
        dim0=dim0,
        num_keys=num_keys,
      )

      return [from_compressed_transpose_format(
        tiles, dim0=dim0) for tiles in arrs_tiles]
    # wrapping to act on dim0 <= NUM_LANES in the kernel
    return [
      jnp.concatenate(arr_slices, axis=0)[:shape[0],:k]
      for arr_slices in transpose_list_of_lists(
        [_topk_arrays(arrs)
        for arrs in transpose_list_of_lists([
        jnp.split(arr, pl.cdiv(padded_shape[0], NUM_LANES), axis=0) for arr in arrs])
    ])]

def bitonic_topk_refs(
    in_refs,
    out_refs,
    *,
    num_keys: int,
    descending: bool,
):
    """
    Pallas kernel for bitonic top-k with k=128 in compressed transpose format.

    Algorithm:
    1. Pad input to satisfy alignment requirements
    2. Convert to compressed transpose format: (num_tokens, vocab) -> (128, num_tokens*chunks)
    3. Run bitonic top-k stages to select top 128 values per token
    4. Convert back from compressed transpose format
    5. Unpad and extract top-128 per token
    """
    if not descending:
      raise NotImplementedError
    outs = bitonic_topk_arrays(
      [ref[...] for ref in in_refs], k=out_refs[0].shape[1],
      num_keys=num_keys)
    for out, out_ref in zip(outs, out_refs, strict=True):
      out_ref[...] = out.astype(out_ref.dtype)


@functools.partial(
    jit,
    static_argnames=("k", "num_keys", "descending", "interpret"),
)
def bitonic_topk(
    operand: jax.Array | Sequence[jax.Array],
    k: int = NUM_LANES,
    num_keys: int = 1,
    descending: bool = True,
    interpret: bool = False,
) -> tuple[jax.Array, ...]:
    """
    Compute top-k using bitonic sort in compressed transpose format.

    Optimized for k=NUM_LANES=128 only. Works entirely in compressed transpose
    format for maximum TPU efficiency. Supports multiple operands like sort().

    Supports arbitrary input shapes - padding is handled automatically:
    - For small inputs (prod < NUM_LANES2): pads dim0 to make prod = NUM_LANES2
    - For larger inputs: pads both dims minimally to satisfy alignment

    Args:
        operand: Input array(s) of shape [num_tokens, vocab_size].
                Can be a single array or sequence of arrays.
                Any vocab_size is supported (will be padded automatically).
        k: Number of top elements (must be NUM_LANES=128).
        num_keys: Number of arrays to use as sort keys.
        descending: If True, sort in descending order (default for top-k).
        interpret: If True, run in CPU interpret mode.

    Returns:
        Tuple of arrays (same length as input operands):
            - Each array has shape [num_tokens, k]

    Raises:
        ValueError: If k != NUM_LANES
    """
    if k > NUM_LANES:
      raise ValueError(
          f"bitonic_topk only supports k<=NUM_LANES={NUM_LANES}, got k={k}"
      )

    operands, unpadded_shape = canonicalize_operand(operand)
    operands = [pad(x, (NUM_SUBLANES, NUM_LANES),
      val='min' if descending else 'max') for x in operands]
    num_tokens, vocab_size = operands[0].shape
    # Define output shapes
    output_shapes = [
        jax.ShapeDtypeStruct((num_tokens, NUM_LANES), op.dtype)
        for op in operands
    ]
    outputs = pl.pallas_call(
        functools.partial(
            bitonic_topk_refs,
            num_keys=num_keys,
            descending=descending,
        ),
        out_shape=(output_shapes,),
        compiler_params=pltpu.CompilerParams(
            vmem_limit_bytes=int(0.9 * 2**27)
        ),
        interpret=interpret,
    )(operands)[0]
    return tuple(x[:unpadded_shape[0], :k] for x in outputs)
