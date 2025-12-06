#!/usr/bin/env python3
"""
Script to extract the sample function from sampling.py and all its internal dependencies.

This script reads the source files, extracts relevant sections, and generates a standalone
script with all tpu_inference imports inlined.
"""

from pathlib import Path


def read_lines(file_path: str) -> list[str]:
    """Read file and return lines."""
    with open(file_path, 'r') as f:
        return f.readlines()


def write_lines(file_path: str, lines: list[str]):
    """Write lines to file."""
    with open(file_path, 'w') as f:
        f.writelines(lines)


def extract_lines_range(lines: list[str], start: int, end: int) -> list[str]:
    """Extract a range of lines (1-indexed line numbers)."""
    return lines[start-1:end]


def generate_standalone_script():
    """Generate the standalone sample script using direct line extraction."""

    base_path = Path('.')

    # File paths
    sampling_file = base_path / 'tpu_inference' / 'layers' / 'jax' / 'sample' / 'sampling.py'
    binary_search_file = base_path / 'tpu_inference' / 'layers' / 'common' / 'binary_search.py'
    sharding_file = base_path / 'tpu_inference' / 'layers' / 'common' / 'sharding.py'
    sampling_metadata_file = base_path / 'tpu_inference' / 'layers' / 'jax' / 'sample' / 'sampling_metadata.py'

    # Read all files
    sampling_lines = read_lines(sampling_file)
    binary_search_lines = read_lines(binary_search_file)
    sharding_lines = read_lines(sharding_file)
    metadata_lines = read_lines(sampling_metadata_file)

    output = []

    # Header
    output.append('"""\n')
    output.append('Standalone script containing the sample function from tpu_inference.layers.jax.sample.sampling\n')
    output.append('\n')
    output.append('This script extracts the sample function along with all necessary internal dependencies,\n')
    output.append('inlining code from tpu_inference modules while preserving external library imports.\n')
    output.append('"""\n')
    output.append('\n')

    # Imports
    output.append('import functools\n')
    output.append('from dataclasses import dataclass\n')
    output.append('from typing import Callable, Optional, Sequence\n')
    output.append('\n')
    output.append('import jax\n')
    output.append('import jax.numpy as jnp\n')
    output.append('from jax import lax\n')
    output.append('from jax.sharding import Mesh, NamedSharding\n')
    output.append('from jax.sharding import PartitionSpec as P\n')
    output.append('from vllm.v1.outputs import LogprobsTensors\n')
    output.append('\n')
    output.append('\n')

    # Section 1: Sharding
    output.append('# ' + '=' * 76 + '\n')
    output.append('# From tpu_inference.layers.common.sharding\n')
    output.append('# ' + '=' * 76 + '\n')
    output.append('\n')
    # Extract ShardingAxisName2D class (lines 33-48 in sharding.py)
    output.extend(extract_lines_range(sharding_lines, 33, 48))
    output.append('\n')
    output.append('\n')
    output.append('ShardingAxisName = ShardingAxisName2D\n')
    output.append('\n')
    output.append('\n')

    # Section 2: Binary search
    output.append('# ' + '=' * 76 + '\n')
    output.append('# From tpu_inference.layers.common.binary_search\n')
    output.append('# ' + '=' * 76 + '\n')
    output.append('\n')
    # Copyright and docstring (lines 1-18)
    output.extend(extract_lines_range(binary_search_lines, 1, 18))
    output.append('\n')
    # int32_bsearch (lines 27-67)
    output.extend(extract_lines_range(binary_search_lines, 27, 67))
    output.append('\n')
    output.append('\n')
    # _monotonic_int32_to_float32_bit_pattern (lines 70-97)
    output.extend(extract_lines_range(binary_search_lines, 70, 97))
    output.append('\n')
    output.append('\n')
    # _monotonic_int32_to_float32 (lines 100-114)
    output.extend(extract_lines_range(binary_search_lines, 100, 114))
    output.append('\n')
    output.append('\n')
    # float32_bsearch (lines 117-161)
    output.extend(extract_lines_range(binary_search_lines, 117, 161))
    output.append('\n')
    output.append('\n')
    # topk_mask (lines 164-224)
    output.extend(extract_lines_range(binary_search_lines, 164, 224))
    output.append('\n')
    output.append('\n')
    # topp_mask (lines 227-295)
    output.extend(extract_lines_range(binary_search_lines, 227, 295))
    output.append('\n')
    output.append('\n')

    # Section 3: Sampling metadata
    output.append('# ' + '=' * 76 + '\n')
    output.append('# From tpu_inference.layers.jax.sample.sampling_metadata\n')
    output.append('# ' + '=' * 76 + '\n')
    output.append('\n')
    # TPUSupportedSamplingMetadata (lines 20-36, just the dataclass definition)
    output.extend(extract_lines_range(metadata_lines, 20, 36))
    output.append('\n')
    output.append('\n')

    # Section 4: Sampling
    output.append('# ' + '=' * 76 + '\n')
    output.append('# From tpu_inference.layers.jax.sample.sampling\n')
    output.append('# ' + '=' * 76 + '\n')
    output.append('\n')
    # _SAMPLING_EPS (line 14)
    output.extend(extract_lines_range(sampling_lines, 14, 14))
    output.append('\n')
    output.append('\n')
    # sample function (lines 17-49)
    output.extend(extract_lines_range(sampling_lines, 17, 49))
    output.append('\n')
    output.append('\n')
    # compute_logprobs (lines 52-53)
    output.extend(extract_lines_range(sampling_lines, 52, 53))
    output.append('\n')
    output.append('\n')
    # gather_logprobs (lines 56-96)
    output.extend(extract_lines_range(sampling_lines, 56, 96))

    return output


if __name__ == '__main__':
    output_lines = generate_standalone_script()

    # Write to file
    output_file = 'standalone_sample_generated.py'
    write_lines(output_file, output_lines)

    print(f"Generated standalone script: {output_file}")
    print(f"Total lines: {len(output_lines)}")
