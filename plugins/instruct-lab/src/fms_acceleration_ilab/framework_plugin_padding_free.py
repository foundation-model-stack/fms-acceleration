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
from packaging import version
import warnings

# Third Party
from fms_acceleration import AccelerationPlugin
from peft import LoraConfig
from transformers import TrainingArguments, __version__ as transformers_version, DataCollatorForSeq2Seq
from accelerate import Accelerator
import torch
from types import MethodType
from torch.utils.data import DataLoader

SUPPORTED_TRANSFORMERS_VERSION = "4.43"

class PaddingFreeAccelerationPlugin(AccelerationPlugin):
    
    require_packages = ["flash_attn"]

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

    @property
    def requires_agumentation(self):
        return True

    def augmentation(
        self,
        model,
        train_args: TrainingArguments,
        modifiable_args: Tuple[LoraConfig],
    ):

        # This check is done here to only patch the attention forward
        # if below a specific transformer version (4.43.3) that already
        # addresses padding free
        # https://github.com/huggingface/transformers/pull/31629
        # Subsequently, when additional features are added to the patch
        # such as attention dropout, the version check should be shifted
        # into `build_fa_forward` to manage the forward replacement inside
        # the function.
        if version.parse(transformers_version) < version.parse(SUPPORTED_TRANSFORMERS_VERSION):
            # guarded
            from fms_acceleration.model_patcher import ( # pylint: disable=import-outside-toplevel
                ModelPatcher,
                ModelPatcherRule,
                ModelPatcherTrigger,
            )
            from .flash_attn import build_fa_forward # pylint: disable=import-outside-toplevel
            from functools import partial # pylint: disable=import-outside-toplevel

            # TODO: have a generic version of this rule
            # - do regex on RMSNorm class name
            # - check on the tensors required for fast_rms_layernorm
            model_type = model.config.model_type
            def is_flash_attn_2(module):
                if (
                    module.__class__.__name__.endswith("FlashAttention2")
                ):
                    return True
                return False

            ModelPatcher.register(
                ModelPatcherRule(
                    rule_id=f"{model_type}-pad-free",
                    trigger=ModelPatcherTrigger(check=is_flash_attn_2),
                    forward_builder=partial(
                        build_fa_forward,
                        causal=True,
                    ),
                ),
            )
        else:
            warnings.warn(f"transformers version is equal or later \
                than {SUPPORTED_TRANSFORMERS_VERSION}, attention forward will not be replaced.")

        return model, modifiable_args

    def get_callbacks_and_ready_for_train(
        self, model: torch.nn.Module = None, accelerator: Accelerator=None
    ):
        # patch the dataloader on the accelerator
        self._patch_dataloader(accelerator)
        return []

    def _patch_dataloader(
            self,
            accelerator: Accelerator,
        ):
            """
            Hijacks the accelorator prepare inside `Trainer.train`
            - If it is a single argument. it is assumed to be the prepare call on the dataloader
            - we replace the collate function in the dataloader to flatten the batch into a long
            sequence with special tokens to define the attention computation boundaries
            """
            # Check if transformers already supports a collator that flattens the batch
            # Otherwise, use the locally implemented DataCollatorWithFlattening
            if version.parse(transformers_version) < version.parse(SUPPORTED_TRANSFORMERS_VERSION):
                from .ilab_utils import DataCollatorWithFlattening
            else:
                from transformers import DataCollatorWithFlattening

            # hijack the dataloader in accelerator.prepare to replace the collate_fn
            _old_prepare = accelerator.prepare
            def prepare(self, *args, device_placement=None):
                if len(args) > 1 or not isinstance(args[0], DataLoader):
                    return _old_prepare(*args, device_placement=device_placement)
                dataloader = args[0]

                # Ensure that the current dataloader.collate_fn is Seq2Seq as FMS
                # currently only supports pre-tokenized inputs with Seq2Seq 
                if isinstance(dataloader.collate_fn, DataCollatorForSeq2Seq):
                    raise Exception("The padding-free plugin only works on a Seq2Seq data collator, \
                        otherwise the it can be unreliable")

                # Replace the collate_fn in dataloader
                dataloader.collate_fn = DataCollatorWithFlattening()

                return dataloader

            accelerator.prepare = MethodType(prepare, accelerator)

# register
AccelerationPlugin.register_plugin(
    PaddingFreeAccelerationPlugin,
    configuration_and_paths=[
        "training.attention.padding_free",
    ],
)
