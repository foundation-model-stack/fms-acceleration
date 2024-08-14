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
import warnings

# Third Party
from fms_acceleration import AccelerationPlugin
from peft import LoraConfig
from transformers import (
    TrainingArguments,
    DataCollatorForSeq2Seq,
)
from accelerate import Accelerator
import torch


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
            values=["huggingface"],
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
        from fms_acceleration.model_patcher import (  # pylint: disable=import-outside-toplevel
            ModelPatcher,
            ModelPatcherRule,
            ModelPatcherTrigger,
        )
        from functools import partial  # pylint: disable=import-outside-toplevel

        # TODO: to be moved to framework
        from .accelerator_patcher import AcceleratorPatcher, AcceleratorPatcherComponent

        def _collator_check_seq2seq(collate_fn):
            # supposed to just return bool, but we raise here
            if not isinstance(collate_fn, DataCollatorForSeq2Seq):
                raise TypeError(
                    "The padding-free plugin currently only works with a \
                    `DataCollatorForSeq2Seq` collate_fn, \
                    otherwise the collation can be unreliable"
                )

            return True

        # This check is done here to only patch the attention forward
        # the PR was merged here
        # https://github.com/huggingface/transformers/pull/31629

        _native = False
        try:
            # if this is importable, it means the PR
            # has been merged, and there is no more need to
            # pylint: disable=import-outside-toplevel,no-name-in-module,unused-import
            from transformers import (
                DataCollatorWithFlattening,
            )
            _native = True

        except ImportError:

            # Otherwise, use the locally implemented DataCollatorWithFlattening
            # pylint: disable=import-outside-toplevel
            from .aadp_utils import (
                DataCollatorWithFlattening,
            )

        # setup the collator
        AcceleratorPatcher.replace(
            "flattening-collator", AcceleratorPatcherComponent.data_collator,
            replacement=DataCollatorWithFlattening(), 
            pre_requisite_check=_collator_check_seq2seq,
        )

        if _native:
            # - if natively supported, then no more need for patch the model
            # - so print and return
            warnings.warn(
                "transformers version supports padding free natively in various models."
            )
            return model, modifiable_args

        # Otherwise patching is required:
        # 1. a custom forward has to be registered on the backbone
        #    to intercept the position ids
        def _is_backbone(module: torch.nn.Module):
            return any(isinstance(mod, torch.nn.Embedding) for mod in module.children())

        # - patch backbone
        model_type = model.config.model_type
        # pylint: disable=import-outside-toplevel
        from .flash_attn import build_backbone_forward

        ModelPatcher.register(
            ModelPatcherRule(
                rule_id=f"{model_type}-backbone-pad-free",
                trigger=ModelPatcherTrigger(check=_is_backbone),
                forward_builder=partial(
                    build_backbone_forward,
                    model_id=id(model),
                ),
            ),
        )

        # Next, the flash attention function needs to be patched
        # how it is patched depends on the transformers version
        try:
            # Case I:
            # if transformers.modeling_flash_attention_utils
            # can be imported, then we patch the flash attention function
            # here. This is required because
            # - this is an old version that does not have logic to handle the flattened batch

            # pylint: disable=import-outside-toplevel
            from transformers.modeling_flash_attention_utils import (
                _flash_attention_forward,
            )

            from .flash_attn import _flash_attention_forward_with_posids

            ModelPatcher.register(
                ModelPatcherRule(
                    rule_id="flash_attn_forward",
                    import_and_maybe_reload=(
                        "transformers.modeling_flash_attention_utils._flash_attention_forward",
                        partial(_flash_attention_forward_with_posids, id(model)),
                        model.__module__,
                    ),
                ),
            )
        except ImportError:
            # Case II: the flash attention functions are methods
            # attached to the model classes
            # - for similar reasons as Case I, they need to be patched on the
            #   FA2 modules
            from .flash_attn import (
                build_fa_forward,
            )  # pylint: disable=import-outside-toplevel

            def is_flash_attn_2(module):
                if module.__class__.__name__.endswith("FlashAttention2"):
                    return True
                return False

            ModelPatcher.register(
                ModelPatcherRule(
                    rule_id=f"{model_type}-pad-free",
                    trigger=ModelPatcherTrigger(check=is_flash_attn_2),
                    forward_builder=partial(
                        build_fa_forward,
                        model_id=id(model),
                    ),
                ),
            )


        return model, modifiable_args

    def get_callbacks_and_ready_for_train(
        self, model: torch.nn.Module = None, accelerator: Accelerator = None
    ):

        # TODO: to be moved to framework
        from .accelerator_patcher import patch_accelerator, AcceleratorPatcher
        patch_accelerator(accelerator)
        print (AcceleratorPatcher.summary())

        return []


# register
AccelerationPlugin.register_plugin(
    PaddingFreeAccelerationPlugin,
    configuration_and_paths=[
        "training.attention.padding_free",
    ],
)
