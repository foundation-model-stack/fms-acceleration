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
from accelerate.utils import set_module_tensor_to_device
from fms_acceleration import AccelerationPlugin
from peft import LoraConfig
from peft.tuners.lora.layer import LoraLayer
from transformers import TrainingArguments
from transformers.utils import logging
import torch
import torch.distributed as dist
from .fused_ops.unsloth_lora.utils import register_quant_state

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
    from .models.model_patcher import (  # pylint: disable=import-outside-toplevel
        patch_model_summary,
    )

    for line in patch_model_summary().split("\n"):
        logging_func(line)


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
        register_quant_state(mod)

    def _all_reduce_hook(grad):
        if grad is not None:
            grad = grad.contiguous()
            dist.all_reduce(grad, op=dist.ReduceOp.AVG, group=None)
        return grad

    for mod in modules:
        A = mod.lora_A.default
        B = mod.lora_B.default

        # install hooks on the adapters
        A.weight.register_hook(_all_reduce_hook)
        B.weight.register_hook(_all_reduce_hook)

        # because we will ignore these from FSDP, we need to manually
        # move them to gpu if they are already not on them
        if not A.weight.is_cuda:
            set_module_tensor_to_device(A, "weight", "cuda")
        if not B.weight.is_cuda:
            set_module_tensor_to_device(B, "weight", "cuda")


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

        # this is a guarded import, because the model rule registration
        # does not need to be loaded unless patch_model is required
        # Local
        from .models.model_patcher import (  # pylint: disable=import-outside-toplevel
            patch_model,
        )

        model = patch_model(model, base_type=self._base_layer)
        return model, modifiable_args

    def get_callbacks_and_ready_for_train(
        self, model: torch.nn.Module = None, accelerator=None
    ):

        # if this is moved to framework, it can be handled as the same way as
        # log_initialization_message
        # log the patch summary
        if accelerator is not None and accelerator.is_main_process:
            log_patch_summary(logging_func=logger.info)

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


# register
AccelerationPlugin.register_plugin(
    FastQuantizedPeftAccelerationPlugin,
    configuration_and_paths=["peft.quantization.fused_ops_and_kernels"],
)
