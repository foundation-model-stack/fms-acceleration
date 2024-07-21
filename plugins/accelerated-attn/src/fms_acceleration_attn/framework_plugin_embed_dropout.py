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

from .framework_plugin_mlp_dropout import build_dropout_forward

class EmbeddingDropoutAccelerationPlugin(AccelerationPlugin):


    def __init__(self, configurations: Dict[str, Dict]):
        super().__init__(configurations)

        self._method = self._check_config_and_maybe_check_values(
            key="training.embedding.dropout.method",
            values=["inputs"],
        )

        self._p = self._check_config_and_maybe_check_values(
            key="training.embedding.dropout.value",
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
        from torch.nn import Embedding

        # TODO: have a generic version of this rule
        # - do regex on RMSNorm class name
        # - check on the tensors required for fast_rms_layernorm
        ModelPatcher.register(
            ModelPatcherRule(
                rule_id=f"embedding-dropout-{self._p}",
                trigger=ModelPatcherTrigger(check=Embedding),
                forward_builder_args=['dropout'],
                forward_builder=build_dropout_forward,
            ),
        )

        return model, modifiable_args

# register
AccelerationPlugin.register_plugin(
    EmbeddingDropoutAccelerationPlugin,
    configuration_and_paths=[
        "training.embedding.dropout",
    ],
)
