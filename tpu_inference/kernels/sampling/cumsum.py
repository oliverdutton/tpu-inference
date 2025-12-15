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

"""Cumulative sum operations for sampling.

Extracted from tallax: https://github.com/oliverdutton/tallax
"""

import functools
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from tpu_inference.kernels.sampling.utils import (
    iota_tile,
    NUM_LANES,
    NUM_SUBLANES,
    log2,
    pad
)

def reverse_tiles(tiles, axis):
  tile_shape = tiles[0].shape
  reverse_perm = tile_shape[axis] - 1 - iota_tile(axis)
  return [jnp.take_along_axis(tile, reverse_perm, axis=axis) for tile in tiles[::-1]]

def cumsum_tile(tile, axis):
  n = tile.shape[axis]
  idx = iota_tile(axis)
  for stage in range(log2(n)):
    permutation = idx - 2**stage
    tile += jnp.where(
      permutation>=0,
      jnp.take_along_axis(tile, permutation % n, axis=axis),
      0)
  return tile

def cumsum_arrays(arr, axis, reverse=False):
  '''
  TPU Pallas lowerable array based implementation of jax.lax.cumsum

  Note: most TPU versions do not allow lane sums in bfloat16, so suggest  casting to jnp.float32 before passing in
  '''
  assert arr.ndim==2
  shape = arr.shape
  tile_shape = (NUM_SUBLANES, NUM_LANES)
  arr = pad(arr, tile_shape, val=0)
  def _cumsum_arrays(arr):
    n = arr.shape[axis] // tile_shape[axis]
    tiles = jnp.split(arr, n, axis=axis)
    if reverse:
      tiles = reverse_tiles(tiles, axis=axis)
    outs = [cumsum_tile(tile, axis) for tile in tiles]
    tile_sums = [tile.sum(axis, keepdims=True) for tile in tiles]
    for i in range(1, n):
      outs[i] += tile_sums[i-1]
      tile_sums[i] += tile_sums[i-1]
    if reverse:
      outs = reverse_tiles(outs, axis=axis)
    return jnp.concatenate(outs, axis=axis)

  batch_axis = 1 - axis
  return jnp.concatenate(
    [_cumsum_arrays(x)
      for x in jnp.split(
        arr, arr.shape[batch_axis] // tile_shape[batch_axis], axis=batch_axis)
    ],
    axis=batch_axis
  )[:shape[0], :shape[1]]


def cumsum_refs(input_ref, output_ref, *, axis: int, reverse: bool):
  """Cumulative sum kernel.

  Computes the cumulative sum of the input array along the specified axis.
  """
  output_ref[...] = cumsum_arrays(input_ref[...], axis=axis, reverse=reverse)


@functools.partial(jax.jit, static_argnames=("axis", "reverse", "interpret"))
def cumsum(
    arr,
    axis,
    reverse: bool = False,
    interpret: bool = False,
):
  """
  Cumulative sum using Pallas.

  Args:
      arr: Input array.
      axis: Axis along which to compute cumsum.
      reverse: If True, compute cumsum in reverse order.
      interpret: Run in interpreter mode (CPU compatible).

  Returns:
      Cumulative sum array.
  """
  return pl.pallas_call(
      functools.partial(
        cumsum_refs,
        axis=axis,
        reverse=reverse,
      ),
      out_shape=jax.ShapeDtypeStruct(arr.shape, arr.dtype),
      interpret=interpret
  )(arr)
