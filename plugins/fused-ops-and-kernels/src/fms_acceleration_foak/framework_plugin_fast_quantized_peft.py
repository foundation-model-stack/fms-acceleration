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
from accelerate.utils import set_module_tensor_to_device
from fms_acceleration import AccelerationPlugin
from peft import LoraConfig
from peft.tuners.lora.layer import LoraLayer
from transformers import TrainingArguments
from transformers.modeling_utils import is_fsdp_enabled
import torch
import torch.distributed as dist

# consider moving this somewhere else later
def lora_adapters_switch_ddp_from_fsdp(modules, fsdp_plugin):
    """
    This function installs hooks on the target adapter parameters and
    reduces the accumulated gradients across devices
    """

    # NOTE: assuming lora has no bias
    fsdp_plugin.ignored_modules = []
    for mod in modules:
        fsdp_plugin.ignored_modules.append(mod.lora_A)
        fsdp_plugin.ignored_modules.append(mod.lora_B)

    def _all_reduce_hook(grad):
        if grad is not None:
            grad = grad.contiguous()
            dist.all_reduce(grad, op=dist.ReduceOp.AVG, group=None)
        return grad

    for mod in modules:
        A = mod.lora_A.default
        B = mod.lora_B.default

        # because we will ignore these from FSDP, we need to manually
        # move them to gpu if they are already not on them
        # - if the adapters are on meta, we assume that this is for FSDP
        #   low_cpu_mem_mode purposes, and that the values will be synced over
        # - So just initialize them to empty.
        if not A.weight.is_cuda:
            value = A.weight

            if is_fsdp_enabled() and value.device == torch.device('meta'):
                # if low_cpu_mem_mode
                value = torch.empty(*value.size(), dtype=value.dtype)

            set_module_tensor_to_device(A, "weight", "cuda", value)

            if is_fsdp_enabled():
                dist.broadcast(A.weight, src=0)

        if not B.weight.is_cuda:
            value = B.weight

            if is_fsdp_enabled() and value.device == torch.device('meta'):
                value = torch.empty(*value.size(), dtype=value.dtype)

            set_module_tensor_to_device(B, "weight", "cuda", value)

            if is_fsdp_enabled():
                dist.broadcast(B.weight, src=0)

        # install hooks on the adapters
        # - this has to be done after all weight replacement happens
        A.weight.register_hook(_all_reduce_hook)
        B.weight.register_hook(_all_reduce_hook)

def register_foak_model_patch_rules(base_type):
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
    for _rule in rules:
        ModelPatcher.register(_rule)


class FastQuantizedPeftAccelerationPlugin(AccelerationPlugin):

    # NOTE: may remove this when we have generic model rules
    restricted_model_archs = [
        "MixtralForCausalLM",
        "LlamaForCausalLM",
        "MistralForCausalLM",
    ]

    def __init__(self, configurations: Dict[str, Dict]):
        super().__init__(configurations)

        self._base_layer = self._check_config_and_maybe_check_values(
            key="peft.quantization.fused_ops_and_kernels.base_layer",
            values=["auto_gptq", "bitsandbytes"],
        )

        # only support these at the moment
        self._check_config_equal(
            key="peft.quantization.fused_ops_and_kernels.fused_lora", value=True
        )
        self._check_config_equal(
            key="peft.quantization.fused_ops_and_kernels.fast_loss", value=True
        )
        self._check_config_equal(
            key="peft.quantization.fused_ops_and_kernels.fast_rsm_layernorm",
            value=True,
        )
        self._check_config_equal(
            key="peft.quantization.fused_ops_and_kernels.fast_rope_embeddings",
            value=True,
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
        # NOTE: how do I check this now that the modifiable args are missing
        # assert peft_config.lora_dropout == 0, \
        # "Fused Attention requires lora_dropout argument to be set to 0"

        # need to check why this is needed
        assert (
            model.dtype == torch.float16 and train_args.fp16
        ), "need to run in fp16 mixed precision or load model in fp16"

        # wrapper function to register foak patches
        register_foak_model_patch_rules(base_type=self._base_layer)
        return model, modifiable_args

    def get_callbacks_and_ready_for_train(
        self, model: torch.nn.Module = None, accelerator=None
    ):

        callbacks = []
        if (
            accelerator is not None
            and getattr(accelerator.state, "fsdp_plugin", None) is not None
        ):
            # This function installs grad reduction hooks on adapters if
            # FSDP is detected. Because of incompatibility between FSDP and
            # fused modules, adapters are not sharded - instead
            # accumulated gradients from adapters in each device are reduced
            # in these grad reduce hooks
            # This function might be removed in future if the incompatiblity
            # is resolved
            lora_adapters_switch_ddp_from_fsdp(
                [mod for mod in model.modules() if isinstance(mod, LoraLayer)],
                accelerator.state.fsdp_plugin,
            )
        return callbacks


# This plugin is currently deregistered in favour of framework_plugin_fast_kernels.py
# to additionally support both full-FT and standard PEFT
# AccelerationPlugin.register_plugin(
#     FastQuantizedPeftAccelerationPlugin,
#     configuration_and_paths=["peft.quantization.fused_ops_and_kernels"],
# )
