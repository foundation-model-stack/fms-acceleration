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
    ):
        super().__init__(configurations)

        # This is the "bin" size for the multipack
        self._max_number_tokens = self._check_config_and_maybe_check_values(
            key="training.dataloader.multipack.max_number_tokens",
            default=60000,
        )

        # this is for in case we support the padding case
        self._pad_token_id = self._check_config_and_maybe_check_values(
            key="training.dataloader.multipack.pad_token_id",
            default=0,
        )

        # see about the collator
        attention = self._check_config_and_maybe_check_values(
            key="training.attention",
        )

        # internal flags
        self._padding_free = False

        if "padding_free" in attention:
            # for padding free the multipack preparation will ignore the padding tokens
            self._padding_free = True
        else:
            raise NotImplementedError(
                "Currently, multipack plugin only supports padding free."
            )

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
        # pylint: disable=import-outside-toplevel
        from fms_acceleration.accelerator_patcher import (
            AcceleratorPatcher,
            AcceleratorPatcherComponent,
        )

        # Local
        # pylint: disable=import-outside-toplevel
        from .multipack_sampler import (
            MultipackDistributedBatchSampler,
        )

        rank, num_bins = 0, 1
        if torch.distributed.is_initialized():
            num_bins = torch.distributed.get_world_size()
            rank = torch.distributed.get_rank()
        else:
            raise NotImplementedError(
                "Multipack dataloader only works for distributed training."
            )

        # some checks
        def _prereq(dataloader: DataLoader):
            return hasattr(dataloader, "dataset")

        def _build_multipack_dataloader(
            dataloader: DataLoader, accelerator: Accelerator
        ):

            # 1. get the dataset
            dataset = dataloader.dataset
            # TODO: make this line neater
            # or do the computation only on the main process if we are sure it can be cached
            # by dataset.map
            lengths = np.array(
                dataset.map(lambda x: {"len": len(x["input_ids"])}, num_proc=16)["len"]
            )

            # prepare the multipack distributed batch sampler
            # - this is just an estimate using the average length
            batch_size_per_device = int(lengths.mean() / self._max_number_tokens)
            train_args.__dict__["per_gpu_train_batch_size"] = batch_size_per_device
            sampler = MultipackDistributedBatchSampler(
                batch_max_length=self._max_number_tokens,
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
