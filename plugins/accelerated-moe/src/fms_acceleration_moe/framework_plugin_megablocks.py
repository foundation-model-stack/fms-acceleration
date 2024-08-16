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
import torch
import warnings

# Third Party
from fms_acceleration import AccelerationPlugin
from peft import LoraConfig
from transformers import TrainingArguments, AutoModelForCausalLM


class MegablocksMoEAccelerationPlugin(AccelerationPlugin):

    restricted_model_archs = {"MixtralForCausalLM"}
    require_packages = {"megablocks"}

    def __init__(self, configurations: Dict[str, Dict]):
        super().__init__(configurations)

        # args
        self._dummy = self._check_config_and_maybe_check_values(
            key="training.moe.megablocks",
            values=["dummy"],
        )

    @property
    def requires_custom_loading(self):
        return True

    def model_loader(self, model_name: str, **kwargs):
        # guarded
        from .megablocks_utils.config_utils import update_mlp_registry
        from megablocks_utils.shard_moe_utils import shard_moe, get_moe_kwargs

        # this one does a forward patching on MLP, but needs to be fixed
        # properly as the load balancing loss is currently not properly
        # handled
        update_mlp_registry()

        # get additional parameters
        torch_dtype = kwargs.get("torch_dtype", torch.float32)

        # load the model
        model = AutoModelForCausalLM.from_pretrained(
            model_name, **kwargs
        )

        rank, world_size = 0, 1
        if torch.distributed.is_initialized():
            world_size = torch.distributed.get_world_size()
            rank = torch.distributed.get_rank()
        else:
            # NOTE: or should we do a silent fallback
            raise AssertionError(
                "Megablocks expert parallel only works for distributed training."
            )

        # FIXME: have some way to search out the MOE block
        from transformers.models.mixtral.modeling_mixtral import MixtralSparseMoeBlock

        # FIXME: the dtype checks below are too brittle
        dp_mesh = shard_moe(
            model, 
            MixtralSparseMoeBlock, 
            checkpoint_name_or_path=model_name,
            rank=rank,
            world_size=world_size,
            ep_size=world_size, # FIXME: this can be passed in?
            moe_kwargs=get_moe_kwargs(
                model.config, 
                has_bias=False, # FIXME: is this true in general?
                fp16=torch_dtype == torch.float16,
                bf16=torch_dtype == torch.bfloat16,
            ),
        )

    def get_callbacks_and_ready_for_train(
        self, model: torch.nn.Module = None, accelerator=None
    ):

        callbacks = []
        if (
            accelerator is not None
            and getattr(accelerator.state, "fsdp_plugin", None) is not None
        ):
            # lora_adapters_switch_ddp_from_fsdp(
            #     [mod for mod in model.modules() if isinstance(mod, LoraLayer)],
            #     accelerator.state.fsdp_plugin,
            # )
            # FIXME: should be 
            accelerator.state.fsdp_plugin.ignored_modules = [
                layer.block_sparse_moe for layer in model.model.layers
            ]

        return callbacks

# register
AccelerationPlugin.register_plugin(
    MegablocksMoEAccelerationPlugin,
    configuration_and_paths=[
        "training.moe.megablocks",
    ],
)
