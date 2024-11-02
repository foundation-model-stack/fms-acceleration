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
from typing import Dict
import warnings

# Third Party
from fms_acceleration import AccelerationPlugin
from transformers import AutoConfig, AutoModelForCausalLM
import torch


# pylint: disable=too-many-instance-attributes
class ScatterMoEAccelerationPlugin(AccelerationPlugin):

    # NOTE: its not packaged properly so, "importlib.util.find_spec('khd')"
    # returns but "importlib.metadata.version('kernel-hyperdrive') is needed"
    # require_packages = {"khd"}

    restricted_model_archs = [
        'GraniteMoeForCausalLM', 'MixtralForCausalLM'
    ]

    def __init__(self, configurations: Dict[str, Dict]):
        super().__init__(configurations)

        # arguments for configuring the mixture-of-experts model with defaults
        # shown below for Mixtral 7x8b
        # - 1. component class
        # self._moe_component_cls = self._check_config_and_maybe_check_values(
        #     key="training.moe.scattermoe.moe_component_class",
        #     # default="MixtralSparseMoeBlock",
        #     default="GraniteMoeMoE",
        # )
        # - 2. gate_module_name
        # self._gate_module_name = self._check_config_and_maybe_check_values(
        #     key="training.moe.scattermoe.moe_gate_module_name", default="gate"
        # )
        # # - 3. experts_module_name
        # self._experts_module_name = self._check_config_and_maybe_check_values(
        #     key="training.moe.scattermoe.moe_experts_module_name", default="experts"
        # )
        # # - 4. mlp version
        # self._mlp_version = self._check_config_and_maybe_check_values(
        #     key="training.moe.scattermoe.moe_mlp_impl",
        #     values=["v1", "v2"],
        #     default="v2",
        # )

        # for controlling the type of sharding
        # self._shard_along_dp = self._check_config_and_maybe_check_values(
        #     key="training.moe.scattermoe.shard_along_dp",
        #     values=[True, False],
        #     default=True,
        # )

        # ep_size determines the expert parallel sharding
        # - ep_size is ignored if _shard_along_dp=True
        self._ep_degree = self._check_config_and_maybe_check_values(
            key="training.moe.scattermoe.ep_degree",
            default=1,
        )

        # for the moe_implementation, currently we only use the megablocks
        # dropless sparse implementation
        # self._moe_implementation = self._check_config_and_maybe_check_values(
        #     key="training.moe.scattermoe.moe_implementation",
        #     values=["dropless_sparse"],
        #     default="dropless_sparse",
        # )
        # self._moe_implementation = self._moe_implementation.split("_")[1]

        # self._load_balancing_loss = self._check_config_and_maybe_check_values(
        #     key="training.moe.scattermoe.load_balancing_loss",
        #     values=[True, False],
        #     default=False,
        # )

    @property
    def requires_custom_loading(self):
        return True

    def model_loader(self, model_name: str, **kwargs):
        # guarded
        # Local
        # pylint: disable=import-outside-toplevel
        # from .megablocks_utils.config_utils import update_mlp_registry
        # from .megablocks_utils.shard_moe_utils import get_moe_kwargs, shard_moe

        # # - check the config
        # if self._load_balancing_loss and not hasattr(
        #     AutoConfig.from_pretrained(model_name), "output_router_logits"
        # ):
        #     warnings.warn(
        #         "load_balancing_loss=True but "
        #         "the model '{model_name}' config not have 'output_router_logits' "
        #         "in its config, hence it might not support load balancing and "
        #         "fallback to load_balancing_loss=False."
        #     )
        #     self._load_balancing_loss = False

        # this one does a forward patching on MLP, but needs to be fixed
        # properly as the load balancing loss is currently not properly
        # handled
        # update_mlp_registry(
        #     self._moe_implementation, self._mlp_version, self._load_balancing_loss
        # )
        from .utils import prepare_scattemoe

        # get additional parameters
        # torch_dtype = kwargs.get("torch_dtype", torch.float32)

        # load the model
        model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)

        # set this in the config, which will be picked up by the forward
        # function to go into the load_balancing loss
        # model.config.output_router_logits = self._load_balancing_loss

        rank, world_size = 0, 1
        if torch.distributed.is_initialized():
            world_size = torch.distributed.get_world_size()
            rank = torch.distributed.get_rank()
        # else:
        #     # NOTE: or should we do a silent fallback
        #     raise AssertionError(
        #         "Megablocks expert parallel only works for distributed training."
        #     )

        # shard the MOE, and store products required for
        # FSDP configuration
        # pylint: disable=unused-variable
        self._moe_component_module_names = prepare_scattemoe(
            model,
            # self._moe_component_cls,
            checkpoint_name_or_path=model_name,
            rank=rank,
            world_size=world_size,
            ep_degree=self._ep_degree,
            # shared_mesh_dim=self._shard_along_dp,
            # router_name=self._gate_module_name,
            # expert_name=self._experts_module_name,
            mixed_precision=False,  # Currently this is hardcoded to OFF
        )

        # NOTE: there is currently no good way to get the mixed precision
        # flag from train_args. It will be better to handle this if
        # when we move the sharding to augmentation.

        # NOTE: Currently, it is a bit troublesome to pass the device_mesh to
        #  the FSDP constructor, so we do not do that.
        # - therefore FSDP will always shard on world_size over the default process
        #   group

        return model

    def get_callbacks_and_ready_for_train(
        self, model: torch.nn.Module = None, accelerator=None
    ):

        callbacks = []
        if (
            accelerator is not None
            and getattr(accelerator.state, "fsdp_plugin", None) is not None
        ):
            # TODO: refactor
            # for newer torch that enables foreach for Dtensors we need to remove it
            from torch.optim.optimizer import _foreach_supported_types

            i = 0
            while i < len(_foreach_supported_types):
                x = _foreach_supported_types[i]
                if x.__name__ == 'DTensor':
                    _foreach_supported_types.pop(i)
                else:
                    i += 1 

            # - use an internal function call to get the no split
            # module names, which are typically layers
            _layers = model._get_no_split_modules("")
            accelerator.state.fsdp_plugin.ignored_modules = [
                getattr(layer, name)
                for name in self._moe_component_module_names
                for layer in model.modules()
                if layer.__class__.__name__ in _layers
            ]

            # Third Party
            # TODO: REFACTOR
            from fms_acceleration.model_patcher import patch_target_module

            # Local
            from .utils.checkpoint_utils import (
                load_fsdp_model,
                load_fsdp_optimizer,
                save_fsdp_model,
                save_fsdp_optimizer,
            )

            patch_target_module("transformers.trainer.save_fsdp_model", save_fsdp_model)
            patch_target_module("transformers.trainer.save_fsdp_optimizer", save_fsdp_optimizer)
            patch_target_module("transformers.trainer.load_fsdp_model", load_fsdp_model)
            patch_target_module("transformers.trainer.load_fsdp_optimizer", load_fsdp_optimizer)

        return callbacks


# register
AccelerationPlugin.register_plugin(
    ScatterMoEAccelerationPlugin,
    configuration_and_paths=[
        "training.moe.scattermoe",
    ],
)
