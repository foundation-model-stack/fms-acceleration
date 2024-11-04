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
import os

# Third Party
from accelerate.logging import get_logger
from accelerate.utils.constants import FSDP_MODEL_NAME, OPTIMIZER_NAME
from torch.distributed.checkpoint.default_planner import (
    DefaultLoadPlanner,
    DefaultSavePlanner,
)
from torch.distributed.checkpoint.state_dict import get_state_dict, set_state_dict
from torch.distributed.fsdp.fully_sharded_data_parallel import StateDictType
import torch.distributed.checkpoint as dcp

logger = get_logger(__name__)

# - variable to capture the model variable
#   in the save/load model calls
MODEL_INDEX = None

# Below are rewrite of functions to be able to handle dtensors


# rewrite of func from accelerate.utils.fsdp_utils.py
# - empty function, as main logic is in the optimizer call
#  save_fsdp_optimizer (see below).
def save_fsdp_model(
    fsdp_plugin, accelerator, model, output_dir, model_index=0, adapter_only=False
):
    # pylint: disable=global-statement
    global MODEL_INDEX
    MODEL_INDEX = model_index


# rewrite of func from accelerate.utils.fsdp_utils.py
# - saves both model and optimizer
def save_fsdp_optimizer(
    fsdp_plugin, accelerator, optimizer, model, output_dir, optimizer_index=0
):

    if fsdp_plugin.state_dict_type != StateDictType.SHARDED_STATE_DICT:
        raise NotImplementedError(
            "Checkpointing for megablocks only enabled for sharded state dict."
        )

    # get the state dicts for model and optimize
    (model_state_dict, optimizer_state_dict) = get_state_dict(model, optimizer)

    # - save model
    ckpt_model = os.path.join(output_dir, f"{FSDP_MODEL_NAME}_{MODEL_INDEX}")
    os.makedirs(ckpt_model, exist_ok=True)
    logger.info(f"Saving model to {ckpt_model}")
    dcp.save(
        state_dict={"model": model_state_dict},
        storage_writer=dcp.FileSystemWriter(ckpt_model),
        planner=DefaultSavePlanner(),
    )
    logger.info(f"Model saved to {ckpt_model}")

    # - save optimizer
    ckpt_opt = os.path.join(output_dir, f"{OPTIMIZER_NAME}_{optimizer_index}")
    os.makedirs(ckpt_opt, exist_ok=True)
    logger.info(f"Saving Optimizer state to {ckpt_opt}")
    dcp.save(
        state_dict={"optimizer": optimizer_state_dict},
        storage_writer=dcp.FileSystemWriter(ckpt_opt),
        planner=DefaultSavePlanner(),
    )
    logger.info(f"Optimizer state saved in {ckpt_opt}")


# rewrite of func from accelerate.utils.fsdp_utils.py
# - empty function, as main logic is in the optimizer call
#  load_fsdp_optimizer (see below).
def load_fsdp_model(
    fsdp_plugin, accelerator, model, input_dir, model_index=0, adapter_only=False
):
    # pylint: disable=global-statement
    global MODEL_INDEX
    MODEL_INDEX = model_index


# rewrite of func from accelerate.utils.fsdp_utils.py
# - loads both model and optimizer
def load_fsdp_optimizer(
    fsdp_plugin,
    accelerator,
    optimizer,
    model,
    input_dir,
    optimizer_index=0,
    adapter_only=False,
):

    accelerator.wait_for_everyone()
    if fsdp_plugin.state_dict_type != StateDictType.SHARDED_STATE_DICT:
        raise NotImplementedError(
            "Checkpointing for megablocks only enabled for sharded state dict."
        )

    # - get the state dicts
    model_state_dict, optimizer_state_dict = get_state_dict(model, optimizer)

    # - load the model state dict
    ckpt_model = os.path.join(input_dir, f"{FSDP_MODEL_NAME}_{MODEL_INDEX}")
    dcp.load(
        state_dict={"model": model_state_dict},
        storage_reader=dcp.FileSystemReader(ckpt_model),
        planner=DefaultLoadPlanner(),
    )

    # - load the optimizer state dict
    ckpt_opt = os.path.join(input_dir, f"{OPTIMIZER_NAME}_{optimizer_index}")
    dcp.load(
        state_dict={"optimizer": optimizer_state_dict},
        storage_reader=dcp.FileSystemReader(ckpt_opt),
        planner=DefaultLoadPlanner(),
    )

    # - set the state dicts
    set_state_dict(
        model,
        optimizer,
        model_state_dict=model_state_dict,
        optim_state_dict=optimizer_state_dict,
    )

    # HACK for now
    # - if seems that if params is empty, then the loading has someo
    #    problems
    # - so for now, we just dump some random defaults
    for group in optimizer.param_groups:
        if len(group["params"]) == 0:
            group["betas"] = (0.9, 0.999)
            group["lr"] = 0.0
            group["initial_lr"] = 0.0
            group["eps"] = 1e-8
            group["weight_decay"] = 0.0


# function to replace various trainer functions in HF with the ones
# above
def patch_huggingface_save_and_load_for_dtensors():
    # Third Party
    # NOTE: this is really a global replacement, which we use the patcher
    # to do
    # pylint: disable=import-outside-toplevel
    from fms_acceleration.model_patcher import patch_target_module

    patch_target_module("transformers.trainer.save_fsdp_model", save_fsdp_model)
    patch_target_module("transformers.trainer.save_fsdp_optimizer", save_fsdp_optimizer)
    patch_target_module("transformers.trainer.load_fsdp_model", load_fsdp_model)
    patch_target_module("transformers.trainer.load_fsdp_optimizer", load_fsdp_optimizer)
