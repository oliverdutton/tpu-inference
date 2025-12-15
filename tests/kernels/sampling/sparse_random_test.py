
import pytest
import jax
import jax.numpy as jnp
import numpy as np
from tpu_inference.kernels.sampling.sparse_random import sparse_random_uniform, sparse_random_categorical


@pytest.mark.parametrize("seed", [42, 123, 456])
@pytest.mark.parametrize("minval,maxval", [(0.0, 1.0), (-1.0, 1.0), (5.0, 10.0)])
def test_sparse_random_uniform(seed, minval, maxval):
    """Test sparse_random_uniform by comparing against indexed dense array."""
    key = jax.random.key(seed)
    key, subkey1, subkey2 = jax.random.split(key, 3)

    # Generate dense random array
    dense_shape = (16, 256)
    dense_uniform = jax.random.uniform(
        key,
        shape=dense_shape,
        dtype=jnp.float32,
        minval=minval,
        maxval=maxval
    )

    # Generate random sparse indices
    sparse_shape = (8, 128)
    indices_0 = jax.random.randint(subkey1, sparse_shape, 0, dense_shape[0])
    indices_1 = jax.random.randint(subkey2, sparse_shape, 0, dense_shape[1])

    # Generate sparse random values
    sparse_uniform = sparse_random_uniform(
        key,
        [indices_0, indices_1],
        dim1_size=dense_shape[1],
        dtype=jnp.float32,
        minval=minval,
        maxval=maxval
    )

    # Index into dense array at the sparse positions
    expected = dense_uniform[indices_0, indices_1]

    # Should match exactly (or within FP32 epsilon for scaled ranges)
    if minval == 0.0 and maxval == 1.0:
        # For [0, 1) range, should be exact
        np.testing.assert_array_equal(sparse_uniform, expected,
            err_msg="sparse_random_uniform should match indexed dense array exactly")
    else:
        # For scaled ranges, allow 1 ULP difference due to FP arithmetic
        np.testing.assert_allclose(sparse_uniform, expected, rtol=0, atol=1e-6,
            err_msg="sparse_random_uniform should match indexed dense array")


@pytest.mark.parametrize("seed", [789, 321, 654])
@pytest.mark.parametrize("axis", [0, 1])
def test_sparse_random_categorical(seed, axis):
    """Test sparse_random_categorical by comparing against masked dense array.

    Uses batch=16, dense=256, sparse=128 for both axes.
    - axis=1: sample along columns, shape (batch, dense) = (16, 256)
    - axis=0: sample along rows, shape (dense, batch) = (256, 16)
    """
    key = jax.random.key(seed)
    key, logits_key, indices_key = jax.random.split(key, 3)

    # Parameterized dimensions
    batch_dim = 16      # Number of independent samples
    dense_dim = 256     # Size of dimension we're sampling from
    sparse_dim = 128    # Number of logits we provide (< dense_dim)

    # Shape depends on axis
    if axis == 1:
        dense_shape = (batch_dim, dense_dim)
        sparse_shape = (batch_dim, sparse_dim)
    else:  # axis == 0
        dense_shape = (dense_dim, batch_dim)
        sparse_shape = (sparse_dim, batch_dim)

    # Generate sparse logits
    sparse_logits = jax.random.normal(logits_key, sparse_shape)

    # Random choice without replacement for dense dimension (one choice per batch element)
    dense_iota = jax.lax.broadcasted_iota(jnp.int32, (batch_dim, dense_dim), 1)
    dense_choices = jax.vmap(lambda k, iota: jax.random.choice(k, iota, shape=(sparse_dim,), replace=False))(
        jax.random.split(indices_key, batch_dim),
        dense_iota
    )

    # Create indices mapping sparse to dense positions
    if axis == 1:
        # Sampling along axis 1: batch dimension maps to itself, dense dimension is random
        indices_0 = jax.lax.broadcasted_iota(jnp.int32, sparse_shape, 0)
        indices_1 = dense_choices
    else:  # axis == 0
        # Sampling along axis 0: dense dimension is random, batch dimension maps to itself
        indices_0 = dense_choices.T
        indices_1 = jax.lax.broadcasted_iota(jnp.int32, sparse_shape, 1)

    # Create dense masked array: all -1e12 except at sparse indices
    dense_masked = jnp.full(dense_shape, -1e12)
    dense_masked = dense_masked.at[indices_0, indices_1].set(sparse_logits)

    # Sample from dense using jax.random.categorical
    dense_result = jax.random.categorical(key, dense_masked, axis=axis)

    # Sample from sparse using sparse_random_categorical
    sparse_result = sparse_random_categorical(
        key,
        sparse_logits,
        [indices_0, indices_1],
        dim1_size=dense_shape[1],
        axis=axis
    )

    # Extract results: sparse_result contains selected indices from the sampled axis
    # Result is already 1D with shape (batch_dim,), no squeeze needed
    if axis == 0:
        mapped_result = sparse_result[0]
    else:  # axis == 1
        mapped_result = sparse_result[1]

    # Compare with dense result
    expected_result = dense_result[:batch_dim] if axis == 0 else dense_result

    # Should match exactly (categorical returns int indices)
    np.testing.assert_array_equal(mapped_result, expected_result,
        err_msg=f"sparse_random_categorical should match dense categorical for axis={axis}")


if __name__ == "__main__":
    print("Running sparse_random_uniform tests...")
    test_sparse_random_uniform(42, 0.0, 1.0)
    test_sparse_random_uniform(123, -1.0, 1.0)
    print("sparse_random_uniform tests passed!")

    print("\nRunning sparse_random_categorical tests...")
    test_sparse_random_categorical(789, 0)
    test_sparse_random_categorical(321, 1)
    print("sparse_random_categorical tests passed!")

    print("\nAll tests passed!")
