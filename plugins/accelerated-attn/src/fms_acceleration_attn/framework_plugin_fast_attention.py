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
from typing import Callable, Dict, Tuple

# Third Party
from fms_acceleration import AccelerationPlugin
from peft import LoraConfig
from transformers import TrainingArguments
from accelerate import Accelerator
from transformers.utils import logging
import torch

from .dataloader import patch_multipack_dataloader, get_multipack_dataloader
from .loss import patch_loss_via_accmulate

# want to use the transformers logger, but a bit of pain
logger = logging.get_logger(__name__)  # pylint: disable=invalid-name
logger.setLevel(logging._get_default_logging_level())
logger.addHandler(logging._default_handler)

def log_patch_summary(
    logging_func: Callable = None,
):
    if logging_func is None:
        logging_func = print

    # this is a guarded import, because the model rule registration
    # does not need to be loaded unless patch_model is required
    # Local
    from .model_patcher import (  # pylint: disable=import-outside-toplevel
        patch_model_summary,
    )

    for line in patch_model_summary().split("\n"):
        logging_func(line)


class FastAttentionAccelerationPlugin(AccelerationPlugin):


    def __init__(self, configurations: Dict[str, Dict]):
        super().__init__(configurations)

        self._method = self._check_config_and_maybe_check_values(
            key="training.fast_attention.padding_free.method",
            values=["huggingface"],
        )
        self._loss = self._check_config_and_maybe_check_values(
            key="training.fast_attention.loss",
        )
        self._multipack = self._check_config_and_maybe_check_values(
            key="training.fast_attention.multipack",
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
        # gaurded import 

        effective_batch_size = self._multipack["effective_batch_size"]
        max_batch_len = self._multipack["max_number_tokens"]
        num_bins = 1
        if torch.distributed.is_initialized():
            num_bins = torch.distributed.get_world_size()

        # FIXME: assume I can get it from here
        data_path = train_args.data_path
        self._train_loader, self._grad_accum = get_multipack_dataloader(
            data_path, num_bins,
            effective_batch_size, max_batch_len, 
        )

        train_args.__dict__['gradient_accumulation_steps'] = self._grad_accum
        train_args.__dict__['per_gpu_train_batch_size'] = (
            effective_batch_size // self._grad_accum // num_bins
        )

        from .model_patcher import (  # pylint: disable=import-outside-toplevel
            patch_model,
        )
        model = patch_model(model)

        return model, modifiable_args

    def get_callbacks_and_ready_for_train(
        self, model: torch.nn.Module = None, accelerator: Accelerator=None
    ):
        # if this is moved to framework, it can be handled as the same way as
        # log_initialization_message
        # log the patch summary
        if accelerator is not None and accelerator.is_main_process:
            log_patch_summary(logging_func=logger.info)
            # log_patch_summary(logging_func=print)

        max_batch_len = self._multipack["max_number_tokens"]
        per_token_loss = self._loss["token_averaged_loss"]

        # FIXME: 
        if torch.distributed.is_initialized():
            from torch.distributed.fsdp import MixedPrecision
            accelerator.state.fsdp_plugin.set_auto_wrap_policy(model)
            accelerator.state.fsdp_plugin.mixed_precision_policy = MixedPrecision(
                param_dtype=torch.bfloat16,
                reduce_dtype=torch.bfloat16,
                buffer_dtype=torch.bfloat16,
            )

        # use FSDP checkpointing
        # accelerator.state.fsdp_plugin.activation_checkpointing = train_args.gradient_checkpointing
        accelerator.native_amp = False # defer to FSDP AMP

        patch_multipack_dataloader(
            accelerator, self._train_loader, 
            format=self._method,
            per_token_loss=per_token_loss,
            max_batch_len=max_batch_len,
        )

        if per_token_loss:
            patch_loss_via_accmulate(accelerator)
        return []

# register
AccelerationPlugin.register_plugin(
    FastAttentionAccelerationPlugin,
    configuration_and_paths=[
        "training.fast_attention",
    ],
)
