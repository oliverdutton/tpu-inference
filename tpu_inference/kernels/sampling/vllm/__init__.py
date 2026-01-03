"""TPU inference vLLM integration module.

Public API for vLLM-compatible sampling operations.
"""

from tpu_inference.kernels.sampling.vllm.sampling import topk_topp_and_sample
from tpu_inference.kernels.sampling.vllm.top_p_and_sample import top_p_and_sample

__all__ = [
  "topk_topp_and_sample",
  "top_p_and_sample",
]
