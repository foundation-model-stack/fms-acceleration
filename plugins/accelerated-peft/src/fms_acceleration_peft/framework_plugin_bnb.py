# Copyright The IBM Tuning Team
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

# SPDX-License-Identifier: Apache-2.0
# https://spdx.dev/learn/handling-license-info/

# Standard
from typing import Dict, Tuple
import inspect
import os
import warnings

# Third Party
from fms_acceleration import AccelerationPlugin
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments
from transformers.utils.import_utils import _is_package_available
import torch


# this is a modified copy of the function from peft.utils.other, that we
# will instead use
# - in the original version, all non-INIT8 params (e.g., fp16, bf16) are upcast
#   to full precision.
# - this will cause problems in the LoraLayers, because the activations will then
#   be constantly downcasted, resulting in greatly reduced throughput.
def _prepare_model_for_kbit_training(
    model, use_gradient_checkpointing=True, gradient_checkpointing_kwargs=None
):

    if gradient_checkpointing_kwargs is None:
        gradient_checkpointing_kwargs = {}

    for _, param in model.named_parameters():
        # freeze base model's layers
        param.requires_grad = False

    if use_gradient_checkpointing:
        # When having `use_reentrant=False` + gradient_checkpointing, there is no need for this hack
        if (
            "use_reentrant" not in gradient_checkpointing_kwargs
            or gradient_checkpointing_kwargs["use_reentrant"]
        ):
            # For backward compatibility
            if hasattr(model, "enable_input_require_grads"):
                model.enable_input_require_grads()
            else:

                def make_inputs_require_grad(_module, _input, output):
                    output.requires_grad_(True)

                model.get_input_embeddings().register_forward_hook(
                    make_inputs_require_grad
                )

        # To support older transformers versions,
        # check if the model supports gradient_checkpointing_kwargs
        _supports_gc_kwargs = "gradient_checkpointing_kwargs" in list(
            inspect.signature(model.gradient_checkpointing_enable).parameters
        )

        if not _supports_gc_kwargs and len(gradient_checkpointing_kwargs) > 0:
            warnings.warn(
                "gradient_checkpointing_kwargs is not supported in this version of transformers.",
                "The passed kwargs will be ignored. if you want to use that feature,",
                "please upgrade to the latest version of transformers.",
                FutureWarning,
            )

        gc_enable_kwargs = (
            {}
            if not _supports_gc_kwargs
            else {"gradient_checkpointing_kwargs": gradient_checkpointing_kwargs}
        )

        # enable gradient checkpointing for memory efficiency
        model.gradient_checkpointing_enable(**gc_enable_kwargs)
    return model


class BNBAccelerationPlugin(AccelerationPlugin):

    require_packages = ["bitsandbytes"]

    def __init__(self, configurations: Dict[str, Dict]):
        super().__init__(configurations)

        self._quant_type = self._check_config_and_maybe_check_values(
            key="peft.quantization.bitsandbytes.quant_type", values=["fp4", "nf4"]
        )
        self._no_peft_model = self._check_config_and_maybe_check_values(
            key="peft.quantization.bitsandbytes.no_peft_model", values=[True, False]
        )

    def model_loader(self, model_name: str, **kwargs):

        # get additional parameters
        torch_dtype = kwargs.get("torch_dtype", torch.float32)
        low_cpu_mem_usage = kwargs.get("low_cpu_mem_usage")
        attn_implementation = kwargs.get("attn_implementation")

        config_kwargs = {}
        try:
            world_size = torch.distributed.get_world_size()
        except ValueError:
            world_size = 1  # pg not init

        patched_is_local_dist_rank_0 = None
        if (
            world_size > 1
            and os.environ.get("ACCELERATE_USE_FSDP", "false").lower() == "true"
        ):
            config_kwargs["bnb_4bit_quant_storage"] = torch_dtype

            # - of course assume that this package must exist, simply need the version
            _, _transformers_version = _is_package_available(
                "transformers", return_version=True
            )

            # this is a workaround that disables low_cpu_mem_mode for quant QLORA
            # - this issue was introduced in https://github.com/huggingface/transformers/pull/33154
            #   whereby the low_cpu_mem_mode was actually fixed.
            # - However fixing it causes some problems with the current impl.
            # 1. For lora fused ops, the adapters cannot be managed by FSDP, as
            #  forwards are not called. This causes issue 2) in
            #  https://github.com/foundation-model-stack/fms-acceleration/issues/83
            #  where the adapters are still sharded when passed in the fused-ops.
            #  However, if low_cpu_mem_mode=True, then we NEED FSDP to intialize
            # their state, which contradicts the above point.
            #
            # 2. We have observed,
            # see https://github.com/foundation-model-stack/fms-acceleration/pull/86
            # that low_cpu_mem_mode=True can cause torch distributed primitives
            # to hang.

            if _transformers_version >= "4.45":

                # pylint: disable=import-outside-toplevel
                # Third Party
                from fms_acceleration.model_patcher import patch_target_module
                import transformers.modeling_utils

                def _truthy():
                    return (
                        True  # use this to always return True to is_local_dist_rank_0
                    )

                # - we cannot use the model patcher and this needs to be called immediately below
                #   at the model_loader
                # - but we immediately revert the patch after loading
                patched_is_local_dist_rank_0 = (
                    transformers.modeling_utils.is_local_dist_rank_0
                )
                patch_target_module(
                    "transformers.modeling_utils.is_local_dist_rank_0",
                    _truthy,
                )

                warnings.warn(
                    "Disabling low_cpu_mem_mode in the BNBAccelerationPlugin as this may "
                    "potentiall cause problems with: "
                    "1. the fused-ops-and-kernels package, and, "
                    "2. the syncing of FSDP modules across devices."
                )

        elif world_size > 1:
            warnings.warn(
                "Running in distributed mode but bnb_4bit_quant_storage is not set. "
                "If running in FSDP, this is probably because accelerate is not used. "
                "This will most probably result in error."
            )
        elif world_size == 1 and self._no_peft_model is True:
            warnings.warn(
                """Running on single device and setting plugin config `no_peft_model` as `True`
                PEFT preparation will be managed by SFTTrainer and
                will cause a slowdown in training speed due to
                extraneous dtype casting when SFTTrainer prepares the model using
                https://github.com/huggingface/trl/blob/e90e8d91d2265e484f229c45a5eb8982f94a2936/trl/trainer/sft_trainer.py#L210"""
            )

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type=self._quant_type,
            bnb_4bit_compute_dtype=torch_dtype,
            **config_kwargs,
        )

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            quantization_config=bnb_config,
            token=None,
            low_cpu_mem_usage=low_cpu_mem_usage,
            attn_implementation=attn_implementation,
        )

        if patched_is_local_dist_rank_0 is not None:
            # replace it
            patch_target_module(
                "transformers.modeling_utils.is_local_dist_rank_0",
                patched_is_local_dist_rank_0,
            )

        return model

    @property
    def requires_custom_loading(self):
        return True

    @property
    def requires_agumentation(self):
        # will skip the augmentation if _no_peft_model == True
        return not self._no_peft_model

    def augmentation(
        self,
        model,
        train_args: TrainingArguments,
        modifiable_args: Tuple[LoraConfig],
    ):
        (peft_config,) = modifiable_args  # unpack modifiable args

        # some assertions
        assert peft_config is not None, "need peft_config to install PEFT adapters"

        # requires a custom prepare because the stock one in peft will introduce
        # extraneous casting
        model = _prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=train_args.gradient_checkpointing,
            gradient_checkpointing_kwargs=train_args.gradient_checkpointing_kwargs,
        )

        model = get_peft_model(model, peft_config)
        modifiable_args = (None,)  # return a None
        return model, modifiable_args


# register
AccelerationPlugin.register_plugin(
    BNBAccelerationPlugin,
    configuration_and_paths=["peft.quantization.bitsandbytes"],
)
