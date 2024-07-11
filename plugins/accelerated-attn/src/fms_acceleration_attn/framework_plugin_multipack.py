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
import torch

from .dataloader import patch_multipack_dataloader, get_multipack_dataloader
from .framework_plugin_loss import KEY_ACROSS_GPUS

class MultipackDataloaderAccelerationPlugin(AccelerationPlugin):

    def __init__(self, configurations: Dict[str, Dict]):
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

        if "padding_free" in attention:
            attention = attention["padding_free"]
            assert attention["method"] == "huggingface-injected", \
                "only supported HF injected padding free"
            self._method = "huggingface"

    @property
    def requires_agumentation(self):
        return True

    def augmentation(
        self,
        model,
        train_args: TrainingArguments,
        modifiable_args: Tuple[LoraConfig],
    ):

        num_bins = 1
        if torch.distributed.is_initialized():
            num_bins = torch.distributed.get_world_size()

        # FIXME: assume I can get it from here
        data_path = train_args.data_path
        self._train_loader, self._grad_accum = get_multipack_dataloader(
            data_path, num_bins,
            self._effective_batch_size, self._max_batch_len, 
        )

        # train_args is a dataclass, so needs to be updated this way
        train_args.__dict__['gradient_accumulation_steps'] = self._grad_accum
        train_args.__dict__['per_gpu_train_batch_size'] = (
            self._effective_batch_size // self._grad_accum // num_bins
        )

        return model, modifiable_args

    def get_callbacks_and_ready_for_train(
        self, model: torch.nn.Module = None, accelerator: Accelerator=None
    ):

        # patch the multipack dataloader on the accelerator
        patch_multipack_dataloader(
            accelerator, self._train_loader, 
            format=self._method,
            per_token_loss=self._per_token_loss,
            max_batch_len=self._max_batch_len,
        )

        return []

# register
AccelerationPlugin.register_plugin(
    MultipackDataloaderAccelerationPlugin,
    configuration_and_paths=[
        "training.dataloader.multipack", # activate if multipack config
        "training.loss", # certain methods require special handling
        "training.attention", # affects collator
    ],
)
