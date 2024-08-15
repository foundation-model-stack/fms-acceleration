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
from types import MethodType
from typing import Dict, Tuple

# Third Party
from accelerate import Accelerator
from fms_acceleration import AccelerationPlugin
from peft import LoraConfig
from torch.utils.data import DataLoader
from transformers import TrainingArguments
import numpy as np

# from accelerate.data_loader import DataLoaderShard
import torch


class MultipackDataloaderAccelerationPlugin(AccelerationPlugin):

    require_packages = {"numba"}

    def __init__(
        self,
        configurations: Dict[str, Dict],
        seed: int = 42,
    ):
        super().__init__(configurations)

        multipack = self._check_config_and_maybe_check_values(
            key="training.dataloader.multipack",
        )

        # multipack settings
        self._effective_batch_size = multipack["effective_batch_size"]
        self._max_batch_len = multipack["max_number_tokens"]

        # see about the collator
        attention = self._check_config_and_maybe_check_values(
            key="training.attention",
        )

        # internal flags
        self._seed = seed
        self._collate_fn = None
        self._padding_free = False
        self._pad_token_id = None

        if "padding_free" in attention:
            # for padding free the multipack preparation will ignore the padding tokens
            self._padding_free = True
        else:
            # NOTE: need to get this from somewhere
            assert self._pad_token_id is not None, "need to get pad token id"

    @property
    def requires_agumentation(self):
        return True

    def augmentation(
        self,
        model,
        train_args: TrainingArguments,
        modifiable_args: Tuple[LoraConfig],
    ):

        # guarded because multipack has numba dependencies
        # Third Party
        from fms_acceleration.accelerator_patcher import (
            AcceleratorPatcher,
            AcceleratorPatcherComponent,
        )

        # Local
        from .multipack_sampler import (
            MultipackDistributedBatchSampler,
            find_packing_max_batch_len_and_grad_accum,
        )

        rank, num_bins = 0, 1
        if torch.distributed.is_initialized():
            num_bins = torch.distributed.get_world_size()
            rank = torch.distributed.get_rank()
        else:
            # NOTE: or should we do a silent fallback
            raise AssertionError(
                "Multipack dataloader only works for distributed training."
            )

        # some checks
        def _prereq(dataloader: DataLoader):
            return hasattr(dataloader, "dataset")

        def _build_multipack_dataloader(
            dataloader: DataLoader, accelerator: Accelerator
        ):

            # NOTE: for now we disable support for deepspeed, but can be added in
            # future if needed
            assert (
                not accelerator.state.deepspeed_plugin
            ), "Currently, multipack not supported for deepspeed"

            # 1. get the dataset
            dataset = dataloader.dataset
            # TODO: make this line neater
            # or do the computation only on the main process if we are sure it can be cached
            # by dataset.map
            lengths = np.array(
                dataset.map(lambda x: {"len": len(x["input_ids"])}, num_proc=16)["len"]
            )

            # 2. compute packing
            packing_max_batch_len, grad_accum = (
                find_packing_max_batch_len_and_grad_accum(
                    num_gpus=num_bins,
                    avg_sample_len=lengths.mean(),
                    effective_batch_size=self._effective_batch_size,
                    max_batch_len_per_gpu=self._max_batch_len,
                    is_padding=not self._padding_free,
                    dataset=dataset,
                    pad_id=self._pad_token_id,  # should be used only in padding
                    seed=self._seed,
                )
            )

            # unfortunately this update is late, so the following will not
            # be properly updated. But maybe it will have little effect
            # - trainer._train_batch_size
            # - trainer.state.train_batch_size
            # NOTE: as such I think it does not work with max_steps > 0 anymore

            # 3. update the train args
            # train_args is a dataclass, so needs to be updated this way
            train_args.__dict__["gradient_accumulation_steps"] = grad_accum
            batch_size_per_device = self._effective_batch_size // grad_accum // num_bins
            train_args.__dict__["per_gpu_train_batch_size"] = batch_size_per_device

            # 4. have to update the accelerator gradient state also
            # - because the train args update is late
            accelerator.gradient_state.plugin_kwargs["num_steps"] = grad_accum

            # 5. prepare the multipack distributed batch sampler
            sampler = MultipackDistributedBatchSampler(
                batch_max_length=packing_max_batch_len,
                lengths=lengths,
                num_replicas=num_bins,
                rank=rank,
                seed=self._seed,
                padding=not self._padding_free,
            )

            # NOTE: also handle the printouts later

            # wanted to use this but its abit annoying,
            # from accelerate.data_loader import DataLoaderShard
            # - so will just patch for now, but lets have a better
            #   solution later

            dataloader = DataLoader(
                dataset,
                batch_sampler=sampler,
                num_workers=dataloader.num_workers,
                collate_fn=dataloader.collate_fn,
            )

            # patch a set epoch function to delegate the call to the
            # batch_sampler
            def _set_epoch(self, epoch: int):
                self.batch_sampler.set_epoch(epoch)

            dataloader.set_epoch = MethodType(_set_epoch, dataloader)

            return dataloader

        AcceleratorPatcher.replace(
            "multipack-{ebs}-{mbl}".format(
                ebs=self._effective_batch_size,
                mbl=self._max_batch_len,
            ),
            AcceleratorPatcherComponent.data_loader,
            replacement_builder=_build_multipack_dataloader,
            pre_requisite_check=_prereq,
            skip_prepare=True,
        )

        # take a pointer to train args
        self._train_args = train_args
        return model, modifiable_args


# register
AccelerationPlugin.register_plugin(
    MultipackDataloaderAccelerationPlugin,
    configuration_and_paths=[
        "training.dataloader.multipack",  # activate if multipack config
        "training.attention",  # currently we require multipack to work with padding free
    ],
)
