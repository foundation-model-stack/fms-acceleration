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
import torch
from types import MethodType

# this will produce a dropout on the MLP forward
def build_mlp_dropout_forward(
    mlp: torch.nn.Module, 
    dropout: float = 0.1,
):
    mlp.dropout = torch.nn.Dropout(p=dropout)
    old_forward = mlp.forward

    def forward(self, *args, **kwargs):
        out = old_forward(*args, **kwargs)
        out = self.dropout(out)
        return out 
        
    # do this replace
    return forward

class MLPDropoutAccelerationPlugin(AccelerationPlugin):


    def __init__(self, configurations: Dict[str, Dict]):
        super().__init__(configurations)

        # the fast attention requires knowledge about the 
        # data collator.
        # - currently we do not have a data collator specific plugin
        # - so it requires knowledge about the dataloader
        self._method = self._check_config_and_maybe_check_values(
            key="training.mlp.dropout.method",
            values=["residual"],
        )

        self._p = self._check_config_and_maybe_check_values(
            key="training.mlp.dropout.value",
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
        from transformers.models.llama.modeling_llama import LlamaMLP

        # TODO: have a generic version of this rule
        # - do regex on RMSNorm class name
        # - check on the tensors required for fast_rms_layernorm
        ModelPatcher.register(
            ModelPatcherRule(
                rule_id="llama-mlp-dropout",
                trigger=ModelPatcherTrigger(check=LlamaMLP),
                forward_builder_args=['dropout'],
                forward_builder=build_mlp_dropout_forward,
            ),
        )

        return model, modifiable_args

# register
AccelerationPlugin.register_plugin(
    MLPDropoutAccelerationPlugin,
    configuration_and_paths=[
        "training.mlp.dropout",
    ],
)
