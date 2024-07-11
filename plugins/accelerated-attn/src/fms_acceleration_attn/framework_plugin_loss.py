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

# Third Party
from fms_acceleration import AccelerationPlugin
from accelerate import Accelerator
import torch

# loss patcher
import torch
from torch.distributed import all_reduce, ReduceOp
from accelerate import Accelerator
from types import MethodType
from contextlib import contextmanager

DATA_LOADERS_SUPPORTED = {"multipack"}
KEY_ACROSS_GPUS = 'across_gpus'

# ---------------------- helpers ----------------------

# this patches the loss to be token averaged across GPUS
# - it accomplishes it via accelerate.accmulate
def patch_token_avg_loss_via_accmulate(
    accelerator: Accelerator, 
):

    assert torch.distributed.is_initialized(), "only works for distribtued"
    n = torch.distributed.get_world_size()
    r = torch.distributed.get_rank()

    @contextmanager
    def accumulate(self, *models):

        assert len(models) == 1, "only supports single model"
        model = models[0]
        _old_forward = model.forward

        # with per token loss
        def forward(self, **kwargs):
            num_loss_counted_tokens = kwargs.pop("num_loss_counted_tokens")
            outputs = _old_forward(**kwargs)

            # create tensor for aggregation
            aggregated = torch.zeros(1, dtype=torch.float32).to(r)

            # really only need just to aggregate this
            aggregated[0] = num_loss_counted_tokens

            # reduce
            all_reduce(aggregated, op=ReduceOp.SUM)

            # compute the scaling
            scaling = n * num_loss_counted_tokens / aggregated[0]

            # delete off the tensor created for aggregation
            del aggregated

            # scale the loss
            outputs.loss = outputs.loss * scaling

            # return the output
            return outputs

        # patch
        model.forward = MethodType(forward, model)
        yield old_accumulate(model)
        model.forward = _old_forward

    # patch accumulate
    old_accumulate = accelerator.accumulate
    accelerator.accumulate = MethodType(accumulate, accelerator)

# ---------------------- class ----------------------

class LossAccelerationPlugin(AccelerationPlugin):


    def __init__(self, configurations: Dict[str, Dict]):
        super().__init__(configurations)

        # loss configs
        self._loss_reduction = self._check_config_and_maybe_check_values(
            key=f"training.loss.{KEY_ACROSS_GPUS}.reduction",
            values=["mean"]
        )
        self._loss_resolution = self._check_config_and_maybe_check_values(
            key=f"training.loss.{KEY_ACROSS_GPUS}.resolution",
            values=["token"]
        )

        # this is required as the certain loss reductions have dependencies
        self._dataloader = self._check_config_and_maybe_check_values(
            key="training.dataloader",
        )

        assert any(x in self._dataloader for x in DATA_LOADERS_SUPPORTED), \
            "not using a supported dataloader"


    def get_callbacks_and_ready_for_train(
        self, model: torch.nn.Module = None, accelerator: Accelerator=None
    ):
        if self._loss_resolution == 'token':
            patch_token_avg_loss_via_accmulate(accelerator)
        return []

# register
AccelerationPlugin.register_plugin(
    LossAccelerationPlugin,
    configuration_and_paths=[
        f"training.loss.{KEY_ACROSS_GPUS}", 
        "training.dataloader",
    ],
)
