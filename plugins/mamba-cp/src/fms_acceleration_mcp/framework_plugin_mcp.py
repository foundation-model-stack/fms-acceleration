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

# Local
from .utils import patch_mamba_layers_with_cp_head


# pylint: disable=too-many-instance-attributes
class MCPAccelerationPlugin(AccelerationPlugin):

    def __init__(self, configurations: Dict[str, Dict]):
        super().__init__(configurations)
        self._mamba_cp_degree = self._check_config_and_maybe_check_values(
            key="training.mamba.cp.degree",
            default=None,
        )
        self._cp_mamba_impl = self._check_config_and_maybe_check_values(
            key="training.mamba.cp.mamba_impl",
            default="allgather",
        )
        self._cp_mamba_recompute = self._check_config_and_maybe_check_values(
            key="training.mamba.cp.mamba_recompute",
            default=False,
        )

    # data_config file should be there
    @property
    def requires_augmentation(self):
        return True

    def augmentation(
        self,
        model,
        train_args: TrainingArguments,
        modifiable_args: Tuple[LoraConfig],
    ):
        if self._mamba_cp_degree is not None:
            rank = 0
            if torch.distributed.is_initialized():
                rank = torch.distributed.get_node_local_rank()
                world_size = torch.distributed.get_world_size()
            model_name = model.config.name_or_path
            patch_mamba_layers_with_cp_head(
                model=model,
                checkpoint_name_or_path=model_name,
                rank=rank,
                cp_degree=self._mamba_cp_degree,
                world_size=world_size,
                cp_mamba_impl=self._cp_mamba_impl,
                cp_mamba_recompute=self._cp_mamba_recompute,
            )
        return model, modifiable_args

    def get_callbacks_and_ready_for_train(
        self, model: torch.nn.Module = None, accelerator=None
    ):
        callbacks = []
        return callbacks


# register
AccelerationPlugin.register_plugin(
    MCPAccelerationPlugin,
    configuration_and_paths=[
        "training.mamba.cp",
    ],
)
