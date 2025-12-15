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

"""Bitonic sort helper functions for sampling.

Extracted from tallax: https://github.com/oliverdutton/tallax
Only includes functions needed for sampling operations.
"""

import jax
import jax.numpy as jnp

from tpu_inference.kernels.sampling.utils import (
    log2,
    iota_tile,
    create_bit_indicator,
    transpose_list_of_lists,
    NUM_LANES,
    NUM_SUBLANES,
)


### Bitonic Sort Core Operations

def compare_and_swap(lefts, rights, num_keys: int, is_descending: jax.Array | None, is_right_half=None,
             has_unique_key=False):
  """Compare and conditionally swap array pairs.

  Args:
    lefts: Tuple of left arrays to compare
    rights: Tuple of right arrays to compare
    num_keys: Number of arrays to use as sort keys
    is_descending: Boolean mask for sort direction
    is_right_half: Mask for subtile comparisons
    has_unique_key: Whether first key is guaranteed unique

  Returns:
    Tuple of (sorted_lefts, sorted_rights) or sorted values for subtile.
  """
  num_arrs = len(lefts)

  def _compare_pair(i, left, right):
    handle_subtile_ties = (
        is_right_half is not None
        and not has_unique_key and num_arrs != num_keys and i == num_keys - 1
    )

    if handle_subtile_ties:
      left, right = (
          jnp.where(is_right_half, right, left),
          jnp.where(is_right_half, left, right)
      )

    mask = (left > right if type(is_descending) == bool and is_descending
            else right > left)
    mask = mask.astype(jnp.int32)

    if is_right_half is not None and not handle_subtile_ties:
      mask = jnp.bitwise_xor(mask, is_right_half.astype(jnp.int32))
    return mask

  masks = tuple(
      _compare_pair(i, left, right)
      for i, (left, right) in enumerate(zip(lefts, rights, strict=True))
  )

  ties = [(left == right) for left, right in zip(lefts, rights, strict=True)]

  mask = masks[0]
  for k in range(1, num_keys):
    # Break ties in primary key with secondary key comparison
    mask = jnp.where(ties[k - 1], masks[k], mask)
    ties[k] &= ties[k - 1]

  if is_descending is not None and type(is_descending) != bool:
    # Dynamic descending mask
    mask = mask.astype(bool)
    is_descending = is_descending.astype(bool)
    mask = mask ^ is_descending

  return jax.tree.map(
      lambda left, right: (
          (jnp.where(mask, left, right), jnp.where(mask, right, left))
          if is_right_half is None else
          jnp.where(mask, left, right)
      ),
      lefts, rights
  )


### Within-Tile Substages

def _run_compressed_transpose_format_substage_on_tiles(arrs_tiles, substage, dim0, num_keys: int, dim1_offset=0, stage=None):
  """Perform substage using sublane permutation or cross-tile comparison."""

  def _compute_pair_slice_start_index(i, separation, slice_length=1):
    """Compute start index for pair-wise array slicing."""
    if slice_length > separation:
      raise ValueError(
          f'Separation must be at least slice length, {separation=} {slice_length=}'
      )
    slices_per_pair = separation // slice_length
    pair_idx = i // slices_per_pair
    slice_idx = i % slices_per_pair
    return pair_idx * 2 * separation + slice_idx * slice_length

  assert dim0 <= NUM_LANES
  global_base_index = iota_tile(0) + (((iota_tile(1) // dim0) * NUM_LANES))
  num_tiles = len(arrs_tiles[0])
  tile_rows = NUM_LANES // NUM_SUBLANES
  tile_cols = num_tiles // tile_rows

  def compute_is_descending(idx):
    tile_offset = ((idx // tile_cols) * NUM_SUBLANES +
                   (idx % tile_cols) * (NUM_LANES * (NUM_LANES // dim0)))
    is_desc = create_bit_indicator(stage, dim1_offset + tile_offset + global_base_index)
    if type(stage) == int:
      if stage < log2(NUM_SUBLANES):
        return create_bit_indicator(stage, global_base_index)
      elif stage < log2(NUM_LANES):
        return create_bit_indicator(stage, tile_offset)
    return is_desc

  outs_tiles = [[None for _ in t] for t in arrs_tiles]

  if substage < log2(NUM_SUBLANES):
    # Sublane permutation
    permutation = jnp.bitwise_xor(iota_tile(0), 1 << substage)
    arrs_tiles_permuted = jax.tree.map(
        lambda tile: jnp.take_along_axis(tile, permutation, axis=0), arrs_tiles
    )
    is_right_half = create_bit_indicator(substage, iota_tile(0))
    for idx, (lefts, rights) in enumerate(zip(
        *map(transpose_list_of_lists, (arrs_tiles, arrs_tiles_permuted)), strict=True
    )):
      for arr_idx, out in enumerate(compare_and_swap(
          lefts, rights, is_descending=compute_is_descending(idx),
          is_right_half=is_right_half, num_keys=num_keys
      )):
        outs_tiles[arr_idx][idx] = out
  else:
    # Compare tiles
    separation = (2**substage // NUM_SUBLANES) * tile_cols
    for i in range(num_tiles // 2):
      idx = _compute_pair_slice_start_index(i, separation=separation)
      lefts, rights = (transpose_list_of_lists(arrs_tiles)[j] for j in (idx, idx + separation))
      for arr_idx, (out_left, out_right) in enumerate(compare_and_swap(
          lefts, rights, is_descending=compute_is_descending(idx), num_keys=num_keys
      )):
        outs_tiles[arr_idx][idx] = out_left
        outs_tiles[arr_idx][idx + separation] = out_right

  assert all(not any([v is None for v in out_tiles]) for out_tiles in outs_tiles)
  return outs_tiles


def run_compressed_transpose_format_substages_on_tiles(
    arrs_tiles,
    num_substages: int,
    stage: int,
    dim0: int,
    num_keys: int,
    dim1_offset: int = 0,
):
  """Execute multiple substages within tiles."""
  assert num_substages <= log2(NUM_LANES)

  def _sort_tile_stage(arrs_tiles, stage, num_substages):
    for substage in range(num_substages)[::-1]:
      arrs_tiles = _run_compressed_transpose_format_substage_on_tiles(
          arrs_tiles, substage=substage, dim0=dim0, dim1_offset=dim1_offset,
          stage=stage, num_keys=num_keys
      )
    return arrs_tiles

  if stage is not None:
    # Run single stage
    arrs_tiles = _sort_tile_stage(
        arrs_tiles,
        num_substages=num_substages,
        stage=stage,
    )
  else:
    # Run all stages 1 to num_substages (allows compiler fusion)
    num_stages = num_substages
    for stage_ in range(1, num_stages + 1):
      arrs_tiles = _sort_tile_stage(
          arrs_tiles,
          num_substages=stage_,
          stage=stage_,
      )

  return arrs_tiles
