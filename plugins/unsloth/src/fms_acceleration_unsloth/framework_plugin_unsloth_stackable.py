from transformers import TrainingArguments
from peft import LoraConfig
from typing import Tuple, Dict

from fms_acceleration import AccelerationPlugin
from fms_acceleration.utils import ignore_modules_in_fsdp
import torch

class UnslothStackableAccelerationPlugin(AccelerationPlugin):

    require_packages = ['xformers']
    restricted_model_archs = ['MixtralForCausalLM', 'LlamaForCausalLM', 'MistralForCausalLM']

    def __init__(self, configurations: Dict[str, Dict]):
        super().__init__(configurations)

        self._base_layer = self._check_config_and_maybe_check_values(
            key=f"peft.quantization.unsloth.base_layer", 
            values=['auto_gptq', 'bitsandbytes']
        )
        
        # only support these at the moment
        self._check_config_equal(key=f"peft.quantization.unsloth.fused_lora", value=True)
        self._check_config_equal(key=f"peft.quantization.unsloth.fast_loss", value=True)
        self._check_config_equal(key=f"peft.quantization.unsloth.fast_rsm_layernorm", value=True)
        self._check_config_equal(key=f"peft.quantization.unsloth.fast_rope_embeddings", value=True)

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
        # assert peft_config.lora_dropout == 0, "Unsloth Fused Attention requires lora_dropout argument to be set to 0"

        # need to check why this is needed
        assert model.dtype == torch.float16 and train_args.fp16,\
             "need to run in fp16 mixed precision or load model in fp16"

        # guarded imports
        # - currently this function works only for auto_gptq
        from .unsloth_utils import (
            add_unsloth_improvements
        )
        model = add_unsloth_improvements(model, stacked_over=self._base_layer)
        return model, modifiable_args

    def callbacks_and_ready_for_train(self, model, accelerator):
        callbacks = []
        if getattr(accelerator.state, "fsdp_plugin", None) is not None:
            # This function installs grad reduction hooks on adapters if FSDP is detected.
            # Because of incompatibility between FSDP and fused modules, adapters are not sharded - instead
            # accumulated gradients from adapters in each device are reduced in these grad reduce hooks
            # This function might be removed in future if the incompatiblity is resolved
            from peft.tuners.lora.layer import LoraLayer
            ignore_modules_in_fsdp([mod for mod in model.modules() if isinstance(mod, LoraLayer)], accelerator.state.fsdp_plugin)
        return callbacks

# register
AccelerationPlugin.register_plugin(
    UnslothStackableAccelerationPlugin,
    configuration_and_paths=["peft.quantization.unsloth"], 
)