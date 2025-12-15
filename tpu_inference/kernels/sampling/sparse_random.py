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

"""Sparse random number generation for sampling.

Extracted from tallax: https://github.com/oliverdutton/tallax
"""

import jax
import jax.numpy as jnp
from jax.extend.random import threefry2x32_p

from tpu_inference.kernels.sampling.bitonic_topk import max_arrays

def _bits_to_uniform(bits, dtype):
    """
    Convert random uint32 bits to uniform float in [0, 1).

    This matches the conversion in jax._src.random._uniform().

    Args:
        bits: uint32 array of random bits
        dtype: Target float dtype

    Returns:
        Array of uniform random floats in [0, 1)
    """
    # Get dtype properties
    finfo = jnp.finfo(dtype)
    nbits = finfo.bits
    nmant = finfo.nmant

    # Right-shift to keep only mantissa bits
    # For float32: keep 23 bits, shift right by (32 - 23) = 9
    float_bits = jax.lax.shift_right_logical(
        bits,
        jnp.uint32(nbits - nmant)
    )

    # Create bit pattern for 1.0 in the target dtype
    # For float32: 0x3F800000 (sign=0, exp=127, mantissa=0)
    one_bits = jnp.ones((), dtype=dtype).view(jnp.uint32)

    # OR with 1.0 bit pattern to set exponent
    float_bits = jax.lax.bitwise_or(float_bits, one_bits)

    # Bitcast to float and subtract 1.0 to get [0, 1)
    floats = jax.lax.bitcast_convert_type(float_bits, dtype)
    return floats - jnp.ones((), dtype=dtype)

def sparse_random_uniform(key_ref, indices, dim1_size, dtype=jnp.float32, minval=0., maxval=1.):
  assert len(indices)==2
  # Handle JAX key format - if scalar key, extract data; if already (1,2), use as-is
  if key_ref.ndim == 0:
    # Scalar JAX key - extract data and reshape
    key_ref = jnp.reshape(jax.random.key_data(key_ref), (1, 2))
  counts_lo = indices[0] * dim1_size + indices[1]
  counts_lo = counts_lo.astype(jnp.uint32)
  counts_hi = jnp.zeros_like(counts_lo)
  k1 = jnp.reshape(key_ref[0, 0], (1, 1))
  k2 = jnp.reshape(key_ref[0, 1], (1, 1))
  bits1, bits2 = threefry2x32_p.bind(
      k1, k2, counts_hi, counts_lo)
  bits = bits1 ^ bits2
  floats = _bits_to_uniform(bits, dtype)
  # Scale to [minval, maxval) following JAX's implementation
  minval = jax.lax.convert_element_type(minval, dtype)
  maxval = jax.lax.convert_element_type(maxval, dtype)

  # Scale and shift: floats * (maxval - minval) + minval
  # Use lax.max to ensure values are at least minval
  return jax.lax.max(minval, floats * (maxval - minval) + minval)



def sparse_random_categorical(key_ref, logits, indices, dim1_size, axis=-1, dtype=jnp.float32):
    if dtype != jnp.float32:
        raise NotImplementedError

    # Canonicalize axis to positive
    axis = axis if axis >= 0 else logits.ndim + axis

    u = sparse_random_uniform(
        key_ref,
        indices,
        dim1_size=dim1_size,
        dtype=jnp.float32,
        minval=jnp.finfo(jnp.float32).tiny,
        maxval=1.0
    )
    # Compute Gumbel noise: -log(-log(u))
    gumbel = -jnp.log(-jnp.log(u))
    # Add Gumbel noise to scaled logits
    gumbel_logits = logits + gumbel
    # Find argmax of Gumbel-perturbed logits
    sampled_token_indices = max_arrays(
        [gumbel_logits, *indices],
        num_keys=1,
        axis=axis,
    )[1:]

    return sampled_token_indices
