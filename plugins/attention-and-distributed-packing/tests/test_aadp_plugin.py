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
import os

# Third Party
from fms_acceleration.utils import instantiate_framework, read_configuration

# First Party
from fms_acceleration_aadp import (
    PaddingFreeAccelerationPlugin,
    MultipackDataloaderAccelerationPlugin
)
from fms_acceleration_aadp.multipack_sampler import MultipackDistributedBatchSampler
from datasets import Dataset
from random import randint
import torch
import numpy as np

# configuration
DIRNAME = os.path.dirname(__file__)
CONFIG_PATH_PADDINGFREE = os.path.join(DIRNAME, "../configs/padding_free.yaml")
CONFIG_PATH_MULTIPACK = os.path.join(DIRNAME, "../configs/multipack.yaml")
NUM_GPUS = 8
BATCH_SIZE_PER_DEVICE = 32
NUM_SAMPLES = 10000

def test_framework_installs_aadp_padding_free_plugin():
    with instantiate_framework(
        read_configuration(CONFIG_PATH_PADDINGFREE), require_packages_check=False
    ) as framework:
        for plugin in framework.active_plugins:
            assert isinstance(plugin[1], PaddingFreeAccelerationPlugin)

def test_framework_installs_aadp_multipack_and_paddingfree_plugins():
    """
    Test framework installs and both multipack and paddingfree are registered
    """
    pf_config = read_configuration(CONFIG_PATH_PADDINGFREE)
    mp_config = read_configuration(CONFIG_PATH_MULTIPACK)
    config = {"training": {**pf_config['training'], **mp_config['training']}}
    with instantiate_framework(
        config, require_packages_check=False
    ) as framework:
        assert len(framework.active_plugins) == 2
        for plugin in framework.active_plugins:
            assert (
                isinstance(plugin[1], PaddingFreeAccelerationPlugin)
                or
                isinstance(plugin[1], MultipackDataloaderAccelerationPlugin)   
            )

def test_multipack_sampler_assigns_balanced_tokens():
    """
    Ensure that the multipack sampler produces even batches
    1. 
    """
    # Build a test dataset
    dataset = Dataset.from_list(
        [
            {"input_ids": torch.randint(0, 1000, (randint(256, 1024),))}
            for _ in range(NUM_SAMPLES)
        ]
    )
    lengths = np.array(
        dataset.map(
            lambda x: {"len": len(x["input_ids"])},
        )["len"]
    )
    
    # 2.  generate a multipack subset of indices
    max_batch_len = BATCH_SIZE_PER_DEVICE * lengths.mean()
    tokens_across_rank_multipack = []
    for rank in range(NUM_GPUS):
        sampler = MultipackDistributedBatchSampler(
                    batch_max_length=max_batch_len,
                    lengths=lengths,
                    num_replicas=NUM_GPUS,
                    rank=rank,
                    seed=42,
                    padding=False,
                )
        batches = sampler.generate_batches()
        tokens_across_batches = []
        for batch in batches:
            # count all the tokens in the batch
            num_tokens_across_one_batch = sum([lengths[idx] for idx in batch])
            tokens_across_batches.append(num_tokens_across_one_batch)
        # take average number of tokens across the batches
        average_tokens_across_batches = np.ceil(np.mean(tokens_across_batches))
        tokens_across_rank_multipack.append(average_tokens_across_batches)

    # 3. generate a random sampled subset of indices
    tokens_across_rank_random = []
    perm_indices = torch.randperm(len(dataset)).numpy()
    # bin indices to respective ranks
    split_indices_to_ranks = np.array_split(perm_indices, NUM_GPUS)
    # bin indices in each rank to respective batches
    split_indices_to_batches = [
        np.array_split(split, BATCH_SIZE_PER_DEVICE) 
        for split in split_indices_to_ranks
    ]
    for rank in split_indices_to_batches:
        # count all the tokens in the batch
        token_length_in_batch = [sum([lengths[idx] for idx in batch]) for batch in rank]
        # take average number of tokens across the batches
        tokens_across_rank_random.append(np.ceil(np.mean(token_length_in_batch)))

    # expect std from multipack to be smaller   
    assert np.std(tokens_across_rank_multipack) < np.std(tokens_across_rank_random)
