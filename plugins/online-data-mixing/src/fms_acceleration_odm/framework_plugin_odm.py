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
from .patch import patch_hf_trainer_evaluate


# pylint: disable=too-many-instance-attributes
class OnlineDataMixingAccelerationPlugin(AccelerationPlugin):

    def __init__(self, configurations: Dict[str, Dict]):
        super().__init__(configurations)

        self._update_interval = self._check_config_and_maybe_check_values(
            key="training.odm.odm.update_interval",
            default=1,
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
        # original user intended eval steps is preserved in the model object
        # while we overwrite the training args eval_steps and strategy to 1 and steps
        # since that way eval pipeline is always triggered and is patched for controlled
        # usage for ODM dataloader update action
        model.ta_eval_steps = train_args.eval_steps
        train_args.eval_steps = 1
        train_args.eval_strategy = "steps"

        # update_interval information has to be made available in the evaluate HF patch
        # function and this seems to be the only reasonable way to do so
        model.ta_update_interval = self._update_interval
        return model, modifiable_args

    def get_callbacks_and_ready_for_train(
        self, model: torch.nn.Module = None, accelerator=None
    ):
        callbacks = []
        patch_hf_trainer_evaluate()
        return callbacks


# register
AccelerationPlugin.register_plugin(
    OnlineDataMixingAccelerationPlugin,
    configuration_and_paths=[
        "training.odm.odm",
    ],
)
