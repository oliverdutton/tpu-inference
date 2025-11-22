import functools

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from vllm.v1.outputs import LogprobsTensors

from tpu_inference.layers.common.binary_search import topk_mask, topp_mask
from tpu_inference.layers.common.sharding import ShardingAxisName
from tpu_inference.layers.jax.sample.sampling_metadata import \
    TPUSupportedSamplingMetadata
from tallax import tax

_SAMPLING_EPS = 1e-5


def fallback_sample(
    rng: jax.Array,
    logits: jax.Array,
    tpu_sampling_metadata: TPUSupportedSamplingMetadata,
) -> jax.Array:
    # (B, vocab_size)
    greedy_sampled = jnp.argmax(logits, axis=-1)
    if not tpu_sampling_metadata.do_sampling:
        return greedy_sampled

    logits = logits.astype(jnp.float32)
    logits = topk_mask(logits, tpu_sampling_metadata.top_k, replace_val=-1e12)
    logits = topp_mask(logits, tpu_sampling_metadata.top_p, replace_val=-1e12)

    temperatures = tpu_sampling_metadata.temperature.astype(logits.dtype)
    temperatures = jnp.expand_dims(temperatures, axis=-1)
    logits /= temperatures

    # (batch_size,)
    next_tokens = jax.random.categorical(rng, logits)
    return greedy_sampled, next_tokens


def fast_sample(
    rng: jax.Array,
    logits: jax.Array,
    tpu_sampling_metadata: TPUSupportedSamplingMetadata,
) -> jax.Array:
    # (B, vocab_size)
    input_logits = logits
        
    logits = logits.astype(jnp.float32)
    logits, logits_global_index, is_valid = tax.top_dynamic_k(
      logits,
      tpu_sampling_metadata.top_k,
      max_k = 128,
      block_size = 8,
      block_topk_schedule = (5, 7, 9, 12, 16),
      topk_schedule = (8, 16),
      interpret = False,
    )    
    logits = topp_mask(logits, tpu_sampling_metadata.top_p, replace_val=-1e12)
    temperatures = tpu_sampling_metadata.temperature.astype(logits.dtype)
    temperatures = jnp.expand_dims(temperatures, axis=-1)
    logits /= temperatures

    # (batch_size,)
    next_tokens = jax.vmap(lambda x, y: x[y])(
      logits_global_index,
      jax.random.categorical(rng, logits),
    )
    greedy_sampled = logits_global_index[:,0]
    return jax.lax.cond(
      is_valid.all(),
      lambda: (greedy_sampled, next_tokens),
      lambda: fallback_sample(rng, input_logits, tpu_sampling_metadata)
    )


@functools.partial(
    jax.jit,
    static_argnames=["mesh"],
)
def sample(
    rng: jax.Array,
    mesh: Mesh,
    logits: jax.Array,
    tpu_sampling_metadata: TPUSupportedSamplingMetadata,
) -> jax.Array:
    # (B, vocab_size)
    if tpu_sampling_metadata.do_sampling:
        # Unshard the logits explicity to avoid latency increase.
        logits = jax.lax.with_sharding_constraint(
            logits, NamedSharding(mesh, P(ShardingAxisName.ATTN_DATA, None)))
    
    greedy_sampled, next_tokens = jax.lax.cond(
      (tpu_sampling_metadata.top_k < 128).all(), 
      lambda: fast_sample(rng, input_logits, tpu_sampling_metadata) 
      lambda: fallback_sample(rng, input_logits, tpu_sampling_metadata)
    )
     
    # Note: avoid using the sample result when temperature < _SAMPLING_EPS
    # If temperature < 0, logits /= temperatures will flip the result, causing error.
    return jnp.where(tpu_sampling_metadata.temperature < _SAMPLING_EPS,
                     greedy_sampled, next_tokens)
    

def compute_logprobs(logits: jax.Array) -> jax.Array:
    return jax.nn.log_softmax(logits, axis=-1)


def gather_logprobs(
    logprobs: jax.Array,
    token_ids: jax.Array,
    num_logprobs: int,
) -> LogprobsTensors:
    """
    Gather logprobs for topk and sampled/prompt token.

    Args:
        logprobs: (num tokens) x (vocab) tensor
        token_ids: prompt tokens (if prompt logprobs)
                    or sampled tokens (if sampled
                    logprobs); 1D token ID tensor
                    with (num tokens) elements
        num_logprobs: minimum number of logprobs to
                    retain per token


    Returns:
        Top-k int indices tensor, (num tokens) x (num_logprobs + 1)
        Top-k float logprobs tensor, (num tokens) x (num_logprobs + 1)
        Sampled token rank tensor, (num tokens)
    """
    # Find the topK values.
    topk_logprobs, topk_indices = jax.lax.top_k(logprobs, k=num_logprobs)

    # Get with the logprob of the prompt or sampled token.
    token_ids = jnp.expand_dims(token_ids, axis=-1)
    token_logprobs = jnp.take_along_axis(logprobs, token_ids, axis=-1)

    # Compute the ranks of the actual token.
    token_ranks = jnp.sum(logprobs >= token_logprobs, axis=-1)

    # Concatenate together with the topk.
    indices = jnp.concatenate((token_ids, topk_indices), axis=1)
    logprobs = jnp.concatenate((token_logprobs, topk_logprobs), axis=1)

    # Use int32 to reduce the tensor size.
    indices = jnp.int32(indices)

    return LogprobsTensors(indices, logprobs, token_ranks)
