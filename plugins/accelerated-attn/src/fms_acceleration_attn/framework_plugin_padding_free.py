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
import logging
import torch

# want to use the transformers logger, but a bit of pain
logger = logging.getLogger(__name__)  # pylint: disable=invalid-name
# logger.setLevel(logging._get_default_logging_level())
logger.setLevel(logging.DEBUG)
# logger.addHandler(logging._default_handler)

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


class PaddingFreeAccelerationPlugin(AccelerationPlugin):


    def __init__(self, configurations: Dict[str, Dict]):
        super().__init__(configurations)

        # the fast attention requires knowledge about the 
        # data collator.
        # - currently we do not have a data collator specific plugin
        # - so it requires knowledge about the dataloader
        self._method = self._check_config_and_maybe_check_values(
            key="training.attention.padding_free.method",
            values=["huggingface-injected"],
        )

        self._dropout_method = self._check_config_and_maybe_check_values(
            key="training.attention.padding_free.dropout_method",
            values=["none", "residual"],
            default="none"
        )

        self._dropout_p = self._check_config_and_maybe_check_values(
            key="training.attention.padding_free.dropout_value",
            default=.0,
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

        # guarded
        from .model_patcher import (
            ModelPatcher,
            ModelPatcherRule,
            ModelPatcherTrigger,
        )
        from .flash_attn import build_fa_forward
        from transformers.models.llama.modeling_llama import LlamaFlashAttention2
        from functools import partial

        # TODO: have a generic version of this rule
        # - do regex on RMSNorm class name
        # - check on the tensors required for fast_rms_layernorm
        ModelPatcher.register(
            ModelPatcherRule(
                rule_id=(
                    "llama-pad-free" if self._dropout_method == 'none'
                    else f"llama-pad-free-dropout-{self._dropout_method}-{self._dropout_p}"
                ),
                trigger=ModelPatcherTrigger(check=LlamaFlashAttention2),
                forward_builder=partial(
                    build_fa_forward, causal=True,
                    dropout=(
                        self._dropout_p if self._dropout_method != 'none'
                        else None
                    )
                ),
            ),
        )

        return model, modifiable_args

    def get_callbacks_and_ready_for_train(
        self, model: torch.nn.Module = None, accelerator: Accelerator=None
    ):

        # NOTE: this needs to be moved to framework
        # NOTE: currently self._method is not used
        # NOTE: in future, self._method should be passed into the patcher
        from .model_patcher import (  # pylint: disable=import-outside-toplevel
            patch_model,
        )
        model = patch_model(model)

        # if this is moved to framework, it can be handled as the same way as
        # log_initialization_message
        # log the patch summary
        if accelerator is not None and accelerator.is_main_process:
            log_patch_summary(logging_func=logger.info)

        return []

# register
AccelerationPlugin.register_plugin(
    PaddingFreeAccelerationPlugin,
    configuration_and_paths=[
        "training.attention.padding_free",
    ],
)
