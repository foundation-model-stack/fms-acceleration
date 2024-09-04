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
from typing import Dict, Tuple, Set

# Third Party
from fms_acceleration import AccelerationPlugin, AccelerationPluginConfigError
from peft import LoraConfig
from peft.tuners.lora.layer import LoraLayer
from transformers import TrainingArguments
import torch

# consider rewriting register_foak_model_patch_rules into something 
# like this also
def register_foak_model_patch_rules2(base_type: str, filter_endswith: Set[str] = None):

    # Third Party
    from fms_acceleration.model_patcher import (  # pylint: disable=import-outside-toplevel
        ModelPatcher,
    )

    # Local
    from .models import (  # pylint: disable=import-outside-toplevel
        llama,
        mistral,
        mixtral,
    )
    rules = [
        *llama.get_mp_rules(base_type),
        *mistral.get_mp_rules(base_type),
        *mixtral.get_mp_rules(base_type),
    ]

    if filter_endswith is not None:
        # filter rules
        rules = [
            r for r in rules if 
            any(r.rule_id.endswith(x) for x in filter_endswith)
        ]

    for _rule in rules:
        ModelPatcher.register(_rule)

# maybe this we should define envvars
FILTER_MAP = {
    "fused_lora": {"qkvo", "mlp"},
    "fast_loss": "cross-ent",
    "fast_rsm_layernorm": "rms",
    "fast_rope_embeddings": "rope",
}

class FastKernelsAccelerationPlugin(AccelerationPlugin):

    # NOTE: may remove this when we have generic model rules
    restricted_model_archs = [
        "MixtralForCausalLM",
        "LlamaForCausalLM",
        "MistralForCausalLM",
    ]

    def __init__(self, configurations: Dict[str, Dict]):
        super().__init__(configurations)

        # NOTE: unfortunately we have to do this now, there is no good way to specify mutiple 
        # keys
        try:
            self.configurations = self._check_config_and_maybe_check_values(
                key="training.fused_ops_and_kernels",
            )
        except AccelerationPluginConfigError:
            self.configurations = self._check_config_and_maybe_check_values(
                key="peft.quantization.fused_ops_and_kernels",
            )

        self._check_config_and_maybe_check_values(
            key="base_layer",
            values=["auto_gptq", "bitsandbytes"],
            default="auto_gptq"
        )
        self._check_config_and_maybe_check_values(
            key="fused_lora", values=[False,True], default=True
        )
        self._check_config_and_maybe_check_values(
            key="fast_loss", values=[False,True], default=True
        )
        self._check_config_and_maybe_check_values(
            key="fast_rsm_layernorm", values=[False,True], default=True
        )
        self._check_config_and_maybe_check_values(
            key="fast_rope_embeddings", values=[False,True], default=True
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

        terms = set()
        for k, v in self.configurations.items():
            if v:
                ts = FILTER_MAP[k]
                if isinstance(ts, str):
                    ts = {ts}
                terms.update(ts)

        # wrapper function to register foak patches
        # NOTE: we never take the lora modules so just set arbitrarily
        # to "auto_gptq"
        register_foak_model_patch_rules2(
            base_type="auto_gptq", filter_endswith=terms
        )
        return model, modifiable_args

# register
AccelerationPlugin.register_plugin(
    FastKernelsAccelerationPlugin,
    configuration_or_paths=[
        "training.fused_ops_and_kernels",
        "peft.quantization.fused_ops_and_kernels",
    ],
)
