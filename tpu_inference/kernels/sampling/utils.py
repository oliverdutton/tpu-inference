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

"""Utility functions for sampling kernels.

Extracted from tallax: https://github.com/oliverdutton/tallax
"""

import math
import warnings
from itertools import chain
import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl


# TPU hardware constants
NUM_SUBLANES = 8
NUM_LANES = 128

def is_cpu_platform():
  is_cpu = jax.default_backend() == "cpu"
  if is_cpu:
    warnings.warn("Running on CPU, interpret=True will be used.")
  return is_cpu

def log2(x: int) -> int:
  """Returns ceiling of log2(x)."""
  return math.ceil(math.log2(x))


def flatten(xs):
  """Flatten a nested list by one level."""
  return list(chain.from_iterable(xs))


def ceil_multiple(i, n):
  """Round up i to the nearest multiple of n."""
  return pl.cdiv(i, n) * n


def max_int(a, b):
  """Max of two values, accepts both static and dynamic ints."""
  if not all(map(lambda v: type(v) == int, (a, b))):
    return jnp.maximum(a, b)
  return max(a, b)


def all_concrete_ints(*args):
  """Check if all arguments are concrete Python integers."""
  return all(map(lambda v: type(v) == int, args))


def get_dtype_info(x):
  """Get finfo or iinfo for array dtype."""
  dtype = x.dtype
  if jnp.issubdtype(dtype, jnp.floating):
    return jnp.finfo(dtype)
  elif jnp.issubdtype(dtype, jnp.integer):
    return jnp.iinfo(dtype)
  else:
    raise ValueError('Only int and float supported')


def pad(
    arr: jax.Array,
    block_shape: tuple[int | str, ...] = None,
    prepend: bool | tuple[bool, ...] = False,
    val = 'max_nan',
) -> jax.Array:
  """Pad array to satisfy alignment requirements.

  Args:
    arr: Input array to pad.
    block_shape: Target block shape for each dimension. Can be:
      - int: Pad to be multiple of this value
      - 'power_of_2_lanes': Pad to next power of 2 (at least NUM_LANES)
      Defaults to (NUM_SUBLANES, NUM_LANES).
    prepend: Whether to prepend (True) or append (False) padding.
      Can be a single bool or tuple of bools for each dimension.
    val: Padding value. If None, uses max value (or nan) for sorting.

  Returns:
    Padded array.
  """
  # Handle default block_shape
  if block_shape is None:
    block_shape = (NUM_SUBLANES, NUM_LANES)

  if len(block_shape) != arr.ndim:
    raise ValueError(
        f"block_shape length {len(block_shape)} must match array ndim {arr.ndim}"
    )

  # Normalize prepend to tuple
  if isinstance(prepend, bool):
    prepend = (prepend,) * arr.ndim

  if len(prepend) != arr.ndim:
    raise ValueError(
        f"prepend length {len(prepend)} must match array ndim {arr.ndim}"
    )

  # Calculate padding for each dimension
  pad_widths = []
  for i, (dim_size, block_spec) in enumerate(zip(arr.shape, block_shape)):
    if block_spec == 'power_of_2_lanes':
      target_size = max(2**log2(dim_size), NUM_LANES)
    elif isinstance(block_spec, int):
      target_size = pl.cdiv(dim_size, block_spec) * block_spec
    else:
      raise ValueError(f"Invalid block_shape element: {block_spec}")

    pad_size = target_size - dim_size
    if prepend[i]:
      pad_widths.append((pad_size, 0))
    else:
      pad_widths.append((0, pad_size))

  # Determine padding value
  if isinstance(val, str):
    info = get_dtype_info(arr)
    if val == 'min':
      pad_val = info.min
    elif val == 'max':
      pad_val = info.max
    elif val == 'max_nan':
      pad_val = info.max
      if jnp.issubdtype(arr.dtype, jnp.floating):
        pad_val = jnp.nan
    else:
      raise ValueError
  else:
    pad_val = val

  # Return early if no padding needed
  if all(w == (0, 0) for w in pad_widths):
    return arr

  return jnp.pad(arr, pad_widths, mode='constant', constant_values=pad_val)




def standardize(x):
  """Standardize float values for sorting.

  Converts NaNs to a specific value and normalizes +/-0.
  """
  nan_val = sortable_int_to_float(jnp.iinfo(jnp.int32).max - 1)
  x = jnp.where(jnp.isnan(x), nan_val, x)
  x = jnp.where(x == 0, 0, x)
  return x


def is_32bit(x):
  """Check if array has 32-bit dtype."""
  return x.dtype.itemsize == 4


def to_32bit_dtype(operand_dtype):
  """Convert dtype to corresponding 32-bit dtype."""
  for dtype_class, dtype_32bit in {
      jnp.floating: jnp.float32,
      jnp.integer: jnp.int32,
      jnp.bool_: jnp.int32
  }.items():
    if jnp.issubdtype(operand_dtype, dtype_class):
      return dtype_32bit
  raise ValueError('dtype not recognized')


def same_shape_dtype(ref1, ref2):
  """Check if two refs have same shape and dtype."""
  return (ref1.dtype == ref2.dtype) and (ref1.shape == ref2.shape)


def canonicalize_operand(operand):
  """Convert operand to list of arrays and validate shapes."""
  operands = jax.tree.leaves(operand)
  shapes = [x.shape for x in operands]
  if len(set(shapes)) != 1:
    raise ValueError(f'Inputs must all have the same shape, but found {shapes=}')
  shape = shapes[0]
  if len(shape) != 2:
    raise ValueError('Only 2D inputs supported')
  return operands, shape


### Float-Int Conversion for Sortable Representation

def float_to_sortable_int(x: jnp.ndarray, standardize_input=True) -> jnp.ndarray:
  """Transform float32 bits into sortable int32 representation.

  Negative floats map to [INT_MIN, -1] with reversed order.
  Positive floats map to [0, INT_MAX].
  """
  if standardize_input:
    x = standardize(x)
  i = x.view(jnp.int32)
  return jnp.where(i < 0, i ^ 0x7FFFFFFF, i)


def sortable_int_to_float(i: jnp.ndarray) -> jnp.ndarray:
  """Inverse transformation from sortable int32 back to float32."""
  return jnp.where(i < 0, i ^ 0x7FFFFFFF, i).view(jnp.float32)


### BF16-U16 Packing for Optimization

def pack_bf16_u16_to_i32(val, index):
  """Pack bfloat16 value and uint16 index into single int32.

  BF16 in F32 has empty lower 16 bits where we pack the index.
  This allows sorting while preserving original indices.
  """
  assert index.dtype == jnp.int32
  val_f32 = standardize(val.astype(jnp.float32))
  index = jnp.where(val_f32 < 0, index.shape[1] - 1 - index, index)
  return float_to_sortable_int(
      ((val_f32.view(jnp.int32) & ~0xFFFF) | index).view(jnp.float32),
      standardize_input=False
  )


def unpack_bf16_u16_from_i32(packed):
  """Extract original bfloat16 value and uint16 index from packed int32."""
  assert packed.dtype == jnp.int32, f'found {packed.dtype}'
  packed = sortable_int_to_float(packed)
  val = (packed.view(jnp.int32) & ~0xFFFF).view(jnp.float32).astype(jnp.bfloat16)
  index = packed.view(jnp.int32) & 0xFFFF
  index = jnp.where(val < 0, index.shape[1] - 1 - index, index)
  return val, index


### Tile Operations

def split_array_to_tiles(arr):
  """Split 2D array into flat list of (NUM_SUBLANES, NUM_LANES) tiles."""
  num_rows, num_cols = arr.shape
  tile_rows = num_rows // NUM_SUBLANES
  tile_cols = num_cols // NUM_LANES

  tiles = []
  for row in range(tile_rows):
    for col in range(tile_cols):
      tile = arr[
          row * NUM_SUBLANES: (row + 1) * NUM_SUBLANES,
          col * NUM_LANES: (col + 1) * NUM_LANES,
      ]
      tiles.append(tile)
  return tiles


def join_tiles_to_array(target_shape, tiles):
  """Reconstruct 2D array from flat list of tiles."""
  num_rows, num_cols = target_shape
  tile_rows, tile_cols = tiles[0].shape
  grid_cols = num_cols // tile_cols

  rows = []
  for i in range(len(tiles) // grid_cols):
    row_tiles = tiles[i * grid_cols: (i + 1) * grid_cols]
    rows.append(jnp.concatenate(row_tiles, axis=-1))

  return jnp.concatenate(rows, axis=-2)


def iota_tile(dim):
  """Create iota array with tile shape."""
  return lax.broadcasted_iota(jnp.int32, (NUM_SUBLANES, NUM_LANES), dim)


def create_bit_indicator(bit_position: int, index=None):
  """Create mask indicating which elements have specific bit set.

  Returns bool when bit_position is static int, int32 (0 or 1) when dynamic.
  """
  if index is None:
    index = iota_tile(1)
  if type(bit_position) == int:
    bit = (index & (1 << bit_position))
    return bit > 0
  return (index >> bit_position) & 1


def to_compressed_transpose_format(arr):
  """Convert array to sublane-oriented format for faster permutes."""
  nelems = arr.shape[0] * arr.shape[1]
  assert (nelems % NUM_LANES**2) == 0
  arrs = [
      arr[:, i * NUM_LANES:(i + 1) * NUM_LANES]
      for i in range(pl.cdiv(arr.shape[1], NUM_LANES))
  ]
  arr = jnp.concatenate(arrs, axis=0).T # (128, n*b)
  tiles = split_array_to_tiles(arr)
  return tiles


def from_compressed_transpose_format(tiles, dim0):
  """Convert from compressed transpose format back to original layout."""
  dim1 = (len(tiles) * NUM_SUBLANES * NUM_LANES) // dim0
  arr = join_tiles_to_array(
      (NUM_LANES, (dim0 * dim1) // NUM_LANES),
      tiles) # (128, n*b)
  arr = arr.T
  return jnp.concatenate(
      [arr[i * dim0:(i + 1) * dim0] for i in range(arr.shape[0] // dim0)],
      axis=1
  )


### Loop Utilities

def unrolled_fori_loop(length: int, body_fn, init_val, unroll: int):
  """Execute for loop with manual unrolling for better performance."""
  if length <= 0:
    return init_val
  unroll = min(length, unroll)

  def unrolled_body(i, carry):
    i *= unroll
    for j in range(unroll):
      carry = body_fn(i + j, carry)
    return carry

  carry = jax.lax.fori_loop(0, length // unroll, unrolled_body, init_val)
  for j in range(length % unroll):
    carry = body_fn((length // unroll) * unroll + j, carry)
  return carry


def transpose_list_of_lists(tree):
  """Transpose nested list structure."""
  outer = jax.tree.structure(type(tree)('*') * len(tree))
  inner = jax.tree.structure(type(tree[0])('*') * len(tree[0]))
  return jax.tree.transpose(outer, inner, tree)
