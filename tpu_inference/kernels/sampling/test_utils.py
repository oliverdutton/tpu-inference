import functools
import gzip
import json
import os
from glob import glob
import jax
import jax.numpy as jnp
import pandas as pd

from tpu_inference.kernels.sampling.utils import is_cpu_platform


@jax.jit
def exact_match(xs, ys):
  """Check if two pytrees match exactly (including NaN positions)."""
  return jnp.array(
    jax.tree.leaves(
      jax.tree.map(lambda x, y: jnp.array_equal(x, y, equal_nan=True), xs, ys)
    )
  ).all()


def verify_sort_output(
  operands,
  outputs,
  num_keys: int,
  return_argsort: bool = False,
  descending: bool = False,
  is_stable: bool = False,
  interpret: bool | None = None,
):
  """Validate sort outputs against XLA reference implementation.

  Args:
      operands: Input operand(s) that were sorted
      outputs: Output from sort function to validate
      num_keys: Number of arrays to use as sort keys
      return_argsort: Whether argsort indices were returned
      descending: Sort in descending order
      is_stable: Whether sort should be stable
      interpret: Run in interpret mode

  Returns:
      Boolean indicating if outputs are valid
  """
  if interpret is None:
    interpret = is_cpu_platform()

  kwargs = dict(
    return_argsort=return_argsort,
    descending=descending,
    num_keys=num_keys,
    is_stable=is_stable,
  )

  if is_stable:
    # Exact match required for stable sort
    out_xla = xla_equivalent_sort(operands, **kwargs)
    valid = bool(exact_match(outputs, out_xla))

    if not valid:
      m = jnp.zeros(out_xla[0].shape, dtype=bool)
      for ox, op in zip(out_xla, outputs):
        m |= ~((ox == op) | (jnp.isnan(ox) & jnp.isnan(op)))
      debug_msg = []
      for ox, op in zip(out_xla, outputs):
        debug_msg.append(f"xla {ox[m]}\noutput {op[m]}")
      debug_output = "\n".join(debug_msg)
      print(
        f"Output does not match XLA output for stable sort:\n{debug_output}"
      )

    return valid

  else:
    # Check output is valid permutation with correct relative order
    outputs_stable_sorted = xla_equivalent_sort(
      outputs,
      num_keys=num_keys,
      is_stable=True,
      descending=descending,
    )
    valid = bool(exact_match(outputs, outputs_stable_sorted))
    if not valid:
      m = jnp.zeros(outputs_stable_sorted[0].shape, dtype=bool)
      for ox, op in zip(outputs_stable_sorted, outputs):
        m |= ~((ox == op) | (jnp.isnan(ox) & jnp.isnan(op)))
      debug_msg = []
      for ox, op in zip(outputs_stable_sorted, outputs):
        debug_msg.append(f"sorted {ox[m]}\noutput {op[m]}")
      debug_output = "\n".join(debug_msg)
      print(f"Output is not sorted:\n{debug_output}")
      return False

    narrs = len(outputs)
    operands_fully_sorted = xla_equivalent_sort(
      operands, **{**kwargs, "num_keys": narrs}
    )
    outputs_fully_sorted = xla_equivalent_sort(
      outputs, **{**kwargs, "num_keys": narrs, "return_argsort": False}
    )
    valid_permute = bool(
      exact_match(operands_fully_sorted, outputs_fully_sorted)
    )
    if not valid_permute:
      print("Output is not a valid permutation of input")

    return valid and valid_permute


def uniquely_define_topk(logits, k):
  """Ensure topk is well-defined by handling ties at the k-th boundary.

  If more than k values are >= the k-th largest value, set extras at the boundary to -inf.
  """
  boundary_val = jax.lax.sort(logits)[-k]
  mask = logits == boundary_val
  # if more than k values gt k-th largest value, set them to -inf
  k_covered = (logits > boundary_val).sum()
  mask = mask & (mask.cumsum() > k - k_covered)
  logits = jnp.where(mask, float("-inf"), logits)
  # jax.debug.print('k>=threshold {} for k={}', (logits >= boundary_val).sum(), k)
  return logits


def verify_topk_output(x, outs, axis=1, approximate=False):
  """Validate top-k outputs for correctness.

  Args:
      x: Input array (must be 2D)
      outs: Tuple of (values, indices) from top-k (both must be 2D)
      axis: Axis along which top-k was computed (0 or 1, default 1)
      approximate: If True, return float % of vals >= threshold (0.0 if indices invalid).
                   If False, return boolean exact match (default False)

  Returns:
      If approximate=False: Boolean array indicating validity for each batch element
      If approximate=True: Float array with % of values of top-k present (0.0 if indices fail)
  """
  if x.ndim != 2:
    raise ValueError(
      f"verify_topk_output only supports 2D inputs, got {x.ndim}D"
    )

  out_vals, out_indexs = outs

  if out_vals.ndim != 2 or out_indexs.ndim != 2:
    raise ValueError(
      f"verify_topk_output requires 2D outputs, got values.ndim={out_vals.ndim}, indices.ndim={out_indexs.ndim}"
    )

  batch_axis = 1 - axis

  @functools.partial(jax.vmap, in_axes=batch_axis)
  def verify_slice(x_slice, vals_slice, idxs_slice):
    k = len(vals_slice)
    n = len(x_slice)

    true_topk_vals = jax.lax.top_k(x_slice, k)[0]

    indices_mapping_valid = (x_slice[idxs_slice] == vals_slice).all()
    i = jnp.unique(idxs_slice, size=k, fill_value=-1)
    indices_bounds_valid = ((i >= 0) & (i < n)).all()
    indices_valid = indices_mapping_valid & indices_bounds_valid

    if approximate:
      threshold = true_topk_vals[-1]
      # due to ties at the topk boundary we have to be careful here
      vals_recall = (
        # how many values definitely in topk, with a max topk inclusion number at the threshold
        (vals_slice > threshold).sum()
        + jnp.minimum(
          (true_topk_vals == threshold).sum(), (vals_slice == threshold).sum()
        )
      ) / k
      return jnp.where(indices_valid, vals_recall, 0.0)
    else:
      vals_valid = (vals_slice == true_topk_vals).all()
      return vals_valid & indices_valid

  return verify_slice(x, out_vals, out_indexs)


def benchmark(_run):
  """Benchmark function and print timing from profiler trace."""

  def run():
    return jax.block_until_ready(_run())

  # Warmup
  run()

  tmpdir = "."
  with jax.profiler.trace(tmpdir):
    run()

  # Find trace file
  files = glob(f"{tmpdir}/plugins/profile/*/**.json.gz", recursive=True)
  if not files:
    print("No trace file generated.")
    return

  path = sorted(files, key=os.path.getmtime)[-1]
  try:
    with gzip.open(path, "rb") as f:
      trace = json.load(f)
  except Exception as e:
    print(f"Failed to load trace: {e}")
    return

  if "traceEvents" not in trace:
    print("No traceEvents in trace.")
    return

  df = pd.DataFrame(trace["traceEvents"])
  if df.empty or "name" not in df.columns:
    print("Trace dataframe empty or no name column.")
    return

  df = df[~df.name.isna()]
  df["name"] = df.name.apply(lambda s: s.split("(")[0])

  # Look for JIT compiled functions
  mask = df.name.str.startswith("jit_")
  res = df[mask][["name", "dur"]]

  if not res.empty:
    print(res.to_string(index=False))
  else:
    print("No jit functions found in trace.")


@functools.partial(
  jax.jit,
  static_argnames=("descending", "return_argsort", "is_stable", "num_keys"),
)
def xla_equivalent_sort(
  operand,
  num_keys: int,
  is_stable: bool = False,
  return_argsort: bool = False,
  descending: bool = False,
) -> tuple[jax.Array, ...]:
  """Reference implementation using XLA sort for correctness testing.

  Args:
    operand: Input array(s) to sort
    num_keys: Number of sort keys
    is_stable: Whether to perform stable sort
    return_argsort: Whether to return argsort indices
    descending: Sort in descending order

  Returns:
    Tuple of sorted arrays (and optionally argsort indices)
  """
  operands = jax.tree.leaves(operand)

  if return_argsort:
    operands.append(jax.lax.broadcasted_iota(jnp.int32, operands[0].shape, 1))
  if descending and is_stable:
    operands.insert(
      num_keys, -jax.lax.broadcasted_iota(jnp.int32, operands[0].shape, 1)
    )
    num_keys += 1

  outs = jax.lax.sort(operands, num_keys=num_keys, is_stable=is_stable)

  if descending and is_stable:
    outs = list(outs)
    outs.pop(num_keys - 1)
  if descending:
    outs = tuple(x[..., ::-1] for x in outs)

  return tuple(outs)
