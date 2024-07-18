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
from typing import Dict, Tuple

# Third Party
from fms_acceleration import AccelerationPlugin
from peft import LoraConfig
from transformers import TrainingArguments
from accelerate import Accelerator
# from accelerate.data_loader import DataLoaderShard
import torch
from transformers.utils import logging

from torch.utils.data import DataLoader

from types import MethodType

from .framework_plugin_loss import KEY_ACROSS_GPUS

from .multipack import (
    build_hugginface_padding_free_collator,
    find_packing_max_batch_len_and_grad_accum, 
    MultipackDistributedBatchSampler, TokenDataset
)

# want to use the transformers logger, but a bit of pain
logger = logging.get_logger(__name__)  # pylint: disable=invalid-name
logger.setLevel(logging._get_default_logging_level())
logger.addHandler(logging._default_handler)

class MultipackDataloaderAccelerationPlugin(AccelerationPlugin):

    def __init__(
        self, configurations: Dict[str, Dict], 
        seed: int = 42,
    ):
        super().__init__(configurations)

        multipack = self._check_config_and_maybe_check_values(
            key="training.dataloader.multipack",
        )
    
        # multipack settings
        self._effective_batch_size = multipack["effective_batch_size"]
        self._max_batch_len = multipack["max_number_tokens"]

        # see if we need special loss settings
        loss = self._check_config_and_maybe_check_values(
            key="training.loss",
        )
        if KEY_ACROSS_GPUS in loss:
            loss = loss[KEY_ACROSS_GPUS]
            assert loss['reduction'] == 'mean', "only support mean reduction"
            assert loss['resolution'] == 'token', "only support token resolution"
            self._per_token_loss = True

        # see about the collator
        attention = self._check_config_and_maybe_check_values(
            key="training.attention",
        )

        self._seed = seed 
        self._collate_fn = None
        self._padding_free = False
        self._pad_token_id = None
        if "padding_free" in attention:
            self._padding_free = True

            attention = attention["padding_free"]
            assert attention["method"] == "huggingface-injected", \
                "only supported HF injected padding free"

            self._collate_fn = build_hugginface_padding_free_collator(
                per_token_loss=self._per_token_loss,
                MAX_BATCH_LEN=self._max_batch_len
            )
        else:
            # NOTE: need to get this from somewhere
            assert self._pad_token_id is not None, \
                "need to get pad token id"

    @property
    def requires_agumentation(self):
        return True

    def augmentation(
        self,
        model,
        train_args: TrainingArguments,
        modifiable_args: Tuple[LoraConfig],
    ):

        # take a pointer to train args
        self._train_args = train_args
        return model, modifiable_args

    def get_callbacks_and_ready_for_train(
        self, model: torch.nn.Module = None, accelerator: Accelerator=None
    ):
        # patch the multipack dataloader on the accelerator
        self._patch_multipack_dataloader(accelerator)
        return []

    def _patch_multipack_dataloader(
        self,
        accelerator: Accelerator,
        num_workers: int = 8,
    ):

        rank, num_bins = 0, 1
        if torch.distributed.is_initialized():
            num_bins = torch.distributed.get_world_size()
            rank = torch.distributed.get_rank()

        collate_fn = self._collate_fn
        train_args = self._train_args
        seed = self._seed
        effective_batch_size = self._effective_batch_size
        max_batch_len = self._max_batch_len
        is_padding = not self._padding_free
        pad_token_id = self._pad_token_id
        _old_prepare = accelerator.prepare

        def prepare(self, *args, device_placement=None):

            if len(args) > 1 or not isinstance(args[0], DataLoader):
                return _old_prepare(*args, device_placement=device_placement)

            old_dataloader = args[0]

            # get dataset and compute lengths
            # NOTE: better to let one process do it
            dataset = TokenDataset(old_dataloader.dataset)

            # compute packing
            packing_max_batch_len, grad_accum = find_packing_max_batch_len_and_grad_accum(
                num_gpus=num_bins,
                avg_sample_len=dataset.get_lengths().mean(),
                effective_batch_size=effective_batch_size,
                max_batch_len_per_gpu=max_batch_len,
                is_padding=is_padding,
                dataset=dataset,
                pad_id=pad_token_id, # should be used only in padding
                seed=seed,
            )

            # unfortunately this update is late, so the following will not
            # be properly updated. But maybe it will have little effect
            # - trainer._train_batch_size 
            # - trainer.state.train_batch_size
            # NOTE: as such I think it does not work with max_steps > 0 anymore

            # update the train args
            # train_args is a dataclass, so needs to be updated this way
            train_args.__dict__['gradient_accumulation_steps'] = grad_accum
            batch_size_per_device = effective_batch_size // grad_accum // num_bins
            train_args.__dict__['per_gpu_train_batch_size'] = batch_size_per_device

            # some logging
            if rank == 0:
                logger.info("********************* Multipack Dataloader **********************")
                logger.info( "  Training with multipack so batch size has been adjusted.")
                logger.info(f"  effective_batch_size   : {effective_batch_size}")
                logger.info(f"  max_batch_len (target) : {max_batch_len}")
                logger.info(f"  max_batch_len (actual) : {packing_max_batch_len}")
                logger.info(f"  padding_free           : {not is_padding}")
                logger.info(f"  grad_accum             : {grad_accum}")
                logger.info(f"  batch_size (per device): {batch_size_per_device}")
                if is_padding:
                    logger.info(f"  pad_token_id        : {pad_token_id}")

            # have to update the accelerator gradient state also
            # - because the train args update is late
            accelerator.gradient_state.plugin_kwargs['num_steps'] = grad_accum
            if accelerator.state.deepspeed_plugin:
                ds_config = accelerator.state.deepspeed_plugin.hf_ds_config.config
                ds_config['gradient_accumulation_steps'] = grad_accum
                ds_config['train_micro_batch_size_per_gpu'] = batch_size_per_device
                ds_config['train_batch_size'] = batch_size_per_device * grad_accum * train_args.world_size

            sampler = MultipackDistributedBatchSampler(
                batch_max_length=packing_max_batch_len,
                lengths=dataset.get_lengths(),
                num_replicas=num_bins,
                rank=rank,
                seed=seed,
                padding=is_padding,
            )

            # wanted to use this but its abit annoying, 
            # from accelerate.data_loader import DataLoaderShard
            # - so will just patch for now, but lets have a better
            #   solution later

            dataloader = DataLoader(
                dataset,
                batch_sampler=sampler,
                num_workers=num_workers,
                collate_fn=collate_fn,
            )

            # patch a set epoch function to delegate the call to the 
            # batch_sampler
            def _set_epoch(self, epoch: int):
                self.batch_sampler.set_epoch(epoch)

            dataloader.set_epoch = MethodType(_set_epoch, dataloader)

            return dataloader

        # FIXME: move this somewhere
        accelerator.even_batches = False
        accelerator.prepare = MethodType(prepare, accelerator)


# register
AccelerationPlugin.register_plugin(
    MultipackDataloaderAccelerationPlugin,
    configuration_and_paths=[
        "training.dataloader.multipack", # activate if multipack config
        "training.loss", # certain methods require special handling
        "training.attention", # affects collator
    ],
)
