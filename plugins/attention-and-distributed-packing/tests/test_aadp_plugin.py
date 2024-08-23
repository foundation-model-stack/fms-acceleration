# Copyright The FMS HF Tuning Authors
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

# Standard
from random import randint
import os

# Third Party
from datasets import Dataset  # pylint: disable=import-error
from fms_acceleration.utils import instantiate_framework, read_configuration
import numpy as np
import torch

# First Party
from fms_acceleration_aadp import (
    MultipackDataloaderAccelerationPlugin,
    PaddingFreeAccelerationPlugin,
)
from fms_acceleration_aadp.aadp_utils import calculate_token_lengths
from fms_acceleration_aadp.multipack_sampler import MultipackDistributedBatchSampler

# configuration
DIRNAME = os.path.dirname(__file__)
CONFIG_PATH_PADDINGFREE = os.path.join(DIRNAME, "../configs/padding_free.yaml")
CONFIG_PATH_MULTIPACK = os.path.join(DIRNAME, "../configs/multipack.yaml")


def test_framework_installs_aadp_padding_free_plugin():
    """
    Test framework successfully installs paddingfree plugin
    """
    with instantiate_framework(
        read_configuration(CONFIG_PATH_PADDINGFREE), require_packages_check=False
    ) as framework:
        for plugin in framework.active_plugins:
            assert isinstance(plugin[1], PaddingFreeAccelerationPlugin)


def test_framework_installs_aadp_multipack_and_paddingfree_plugins():
    """
    Test framework installs both multipack and paddingfree plugins
    """
    pf_config = read_configuration(CONFIG_PATH_PADDINGFREE)
    mp_config = read_configuration(CONFIG_PATH_MULTIPACK)
    config = {"training": {**pf_config["training"], **mp_config["training"]}}
    with instantiate_framework(config, require_packages_check=False) as framework:
        assert len(framework.active_plugins) == 2
        for plugin in framework.active_plugins:
            assert isinstance(
                plugin[1],
                (MultipackDataloaderAccelerationPlugin, PaddingFreeAccelerationPlugin),
            )


def test_multipack_sampler_assigns_balanced_tokens():
    """
    Ensure that the multipack sampler load balances the tokens amongst the GPUS
    """
    num_gpus = 8
    batch_size_per_device = 32
    num_samples = 10000
    seed = 42
    num_processes = 4

    # 1. Build a test dataset
    dataset = Dataset.from_list(
        [
            {"input_ids": torch.randint(0, 1000, (randint(256, 1024),))}
            for _ in range(num_samples)
        ]
    )
    lengths = calculate_token_lengths(dataset, num_workers=num_workers)

    # 2.  generate a multipack subset of indices
    max_batch_len = batch_size_per_device * lengths.mean()
    tokens_across_rank_multipack = []
    for rank in range(num_gpus):
        sampler = MultipackDistributedBatchSampler(
            batch_max_length=max_batch_len,
            lengths=lengths,
            num_replicas=num_gpus,
            rank=rank,
            seed=seed,
            padding=False,
        )
        batches = sampler.generate_batches()
        tokens_across_batches = []
        for batch in batches:
            # count all the tokens in the batch
            num_tokens_across_one_batch = sum(lengths[idx] for idx in batch)
            tokens_across_batches.append(num_tokens_across_one_batch)
        # take average number of tokens across the batches
        average_tokens_across_batches = np.ceil(np.mean(tokens_across_batches))
        tokens_across_rank_multipack.append(average_tokens_across_batches)

    # 3. generate a random sampled subset of indices
    tokens_across_rank_random = []
    perm_indices = torch.randperm(len(dataset)).numpy()
    # bin indices to respective ranks
    split_indices_to_ranks = np.array_split(perm_indices, num_gpus)
    # bin indices in each rank to respective batches
    split_indices_to_batches = [
        np.array_split(split, batch_size_per_device) for split in split_indices_to_ranks
    ]
    for rank in split_indices_to_batches:
        # count all the tokens in the batch
        token_length_in_batch = [sum(lengths[idx] for idx in batch) for batch in rank]
        # take average number of tokens across the batches
        tokens_across_rank_random.append(np.ceil(np.mean(token_length_in_batch)))

    # expect std from multipack to be smaller
    assert np.std(tokens_across_rank_multipack) < np.std(tokens_across_rank_random)
