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

"""Test utilities for sampling kernels.

Extracted from tallax: https://github.com/oliverdutton/tallax
"""

import functools
import jax
import jax.numpy as jnp


@jax.jit
def exact_match(xs, ys):
  """Check if two pytrees match exactly (including NaN positions)."""
  return jnp.array(jax.tree.leaves(
      jax.tree.map(lambda x, y: jnp.array_equal(x, y, equal_nan=True), xs, ys)
  )).all()


def verify_topk_output(x, outs, axis=1):
    """Validate top-k outputs for correctness.

    Args:
        x: Input array (must be 2D)
        outs: Tuple of (values, indices) from top-k (both must be 2D)
        axis: Axis along which top-k was computed (0 or 1, default 1)

    Returns:
        Boolean array indicating if the top-k output is valid for each batch element

    Raises:
        ValueError: If x or outputs are not 2D
    """
    if x.ndim != 2:
        raise ValueError(f"verify_topk_output only supports 2D inputs, got {x.ndim}D")

    out_vals, out_indexs = outs

    if out_vals.ndim != 2 or out_indexs.ndim != 2:
        raise ValueError(f"verify_topk_output requires 2D outputs, got values.ndim={out_vals.ndim}, indices.ndim={out_indexs.ndim}")

    batch_axis = 1 - axis

    @functools.partial(jax.vmap, in_axes=batch_axis)
    def verify_slice(x_slice, vals_slice, idxs_slice):
        """Verify a single slice."""
        x_sorted = jnp.sort(x_slice, descending=True)

        k = len(vals_slice)
        n = len(x_slice)
        valid = True

        # actual values must match
        valid &= (vals_slice == x_sorted[:k]).all()

        # indices map to values correctly
        valid &= (x_slice[idxs_slice] == vals_slice).all()

        # indices are all in bounds and unique
        i = jnp.unique(idxs_slice, size=k, fill_value=-1)
        valid &= ((i >= 0) & (i < n)).all()
        return valid

    return verify_slice(x, out_vals, out_indexs)
