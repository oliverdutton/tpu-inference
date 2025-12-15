
import pytest
import jax
import jax.numpy as jnp

from tpu_inference.kernels.sampling.utils import is_cpu_platform
from tpu_inference.kernels.sampling.test_utils import verify_sort_output


def _should_skip_on_cpu(size):
    """Skip tests on CPU for large sizes (> 256) to avoid slow tests."""
    return is_cpu_platform() and size > 256


@pytest.mark.parametrize("dtype", [jnp.bfloat16, jnp.float32])
@pytest.mark.parametrize("size", [128, 256, 2048, 131072])
@pytest.mark.parametrize("variant", [
    "standard",
    "return_argsort",
    "return_argsort_stable",
    "descending",
    "descending_argsort",
    "descending_stable"
])
@pytest.mark.parametrize("num_arrays,num_keys", [
    (1, 1),
    (2, 1),
    (2, 2)
])
def test_sort_comprehensive(dtype, size, variant, num_arrays, num_keys):
    """Comprehensive sort tests with various configurations."""
    # Skip large sizes on CPU
    if _should_skip_on_cpu(size):
        pytest.skip("Skipping large size on CPU - interpret mode is too slow")

    shape = (16, size)
    key = jax.random.key(0)

    # Generate operands using split instead of fold_in
    keys = jax.random.split(key, num_arrays)
    operands = []
    for i in range(num_arrays):
        arr = jax.random.normal(keys[i], shape, dtype=jnp.float32).astype(dtype)
        operands.append(arr)

    # Parse variant
    return_argsort = "return_argsort" in variant
    is_stable = "stable" in variant
    descending = "descending" in variant

    # Use interpret mode on CPU
    interpret = is_cpu_platform()

    verify_sort_output(
        operands,
        num_keys=num_keys,
        return_argsort=return_argsort,
        is_stable=is_stable,
        descending=descending,
        interpret=interpret
    )
