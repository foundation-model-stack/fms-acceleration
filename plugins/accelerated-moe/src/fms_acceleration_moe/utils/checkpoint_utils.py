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
from collections import defaultdict
from typing import Dict, List, Union
import json
import math
import os
import re
import shutil
import types

# Third Party
from accelerate.logging import get_logger
from accelerate.utils.constants import FSDP_MODEL_NAME, OPTIMIZER_NAME
from huggingface_hub import split_torch_state_dict_into_shards
from safetensors.torch import load_file, safe_open, save_file
from torch.distributed.checkpoint.default_planner import (
    DefaultLoadPlanner,
    DefaultSavePlanner,
)
from torch.distributed.checkpoint.state_dict import get_state_dict, set_state_dict
from torch.distributed.fsdp.fully_sharded_data_parallel import StateDictType
from torch.distributed.tensor import DTensor
from transformers import PretrainedConfig
from transformers.utils import CONFIG_NAME, SAFE_WEIGHTS_INDEX_NAME, SAFE_WEIGHTS_NAME
import torch
import torch.distributed.checkpoint as dcp

# Local
from .scattermoe_constants import (
    FILE_SAFETENSOR_INDEX,
    KEY_EXPERT_PARALLEL,
    PARAM_NAME_ROUTER_SCATTERMOE,
    PARAM_NAME_WEIGHT_SCATTERMOE,
    get_scattermoe_conv_spec_from_archs,
)
from .scattermoe_state_dict import get_checkpoint_meta_from_sharded_safetensor

logger = get_logger(__name__)

# - variable to capture the model variable
#   in the save/load model calls
MODEL_INDEX = None
KEY_MODEL = "model"
KEY_OPTIMIZER = "optimizer"

ADAPTER_SAFE_WEIGHTS_NAME = "adapter_model.safetensors"

# Below are rewrite of HF FSDP model saving functions to be able to handle
# that the parameters are now a mixture of regular and Dtensors.
# - these functions are found in accelerate.utils.fsdp_utils.py
# - save_fsdp_model, save_fsdp_optimizer, load_fsdp_model, load_fsdp_optimizer
# NOTE: we will observe warnings such as
# /torch/distributed/checkpoint/state_dict.py:520:
# FutureWarning: Please use DTensor instead and we are deprecating ShardedTensor.


# Load weight map either with index file or manually in single-shard state
def load_weight_map(loc, file_safetensor, file_safetensor_index):
    index_path = os.path.join(loc, file_safetensor_index)
    safetensor_path = os.path.join(loc, file_safetensor)

    try:
        if os.path.exists(index_path):
            # Load weight map from index file
            with open(index_path, encoding="utf-8") as f:
                index = json.load(f)
            weight_map = index["weight_map"]
        else:
            # If no index file, assume single shard
            weight_map = {}
            with safe_open(safetensor_path, framework="pt") as f:
                weight_map = {key: file_safetensor for key in f.keys()}
    except (FileNotFoundError, json.JSONDecodeError, KeyError, IOError) as e:
        raise ValueError(
            f"Failed to load weight map from {file_safetensor} or {file_safetensor_index}: {e}"
        ) from e

    return weight_map


# rewrite of func from accelerate.utils.fsdp_utils.py
# - empty function, the main logic will be in save_fsdp_optimizer (see below).
def save_fsdp_model(
    fsdp_plugin, accelerator, model, output_dir, model_index=0, adapter_only=False
):
    # pylint: disable=global-statement
    global MODEL_INDEX
    MODEL_INDEX = model_index


# rewrite of func from accelerate.utils.fsdp_utils.py
# - saves both model and optimizer at the same time
def save_fsdp_optimizer(
    fsdp_plugin, accelerator, optimizer, model, output_dir, optimizer_index=0
):

    if fsdp_plugin.state_dict_type != StateDictType.SHARDED_STATE_DICT:
        raise NotImplementedError(
            "Checkpointing for megablocks only enabled for sharded state dict."
        )

    # get the state dicts for model and optimize
    (model_state_dict, optimizer_state_dict) = get_state_dict(model, optimizer)

    # filter out lora state dict
    # TODO: Once expert layers are supported for LoRA tuning
    # remove the "router" filtering
    lora_state_dict = {
        k: v
        for k, v in model_state_dict.items()
        if ("lora_A" in k or "lora_B" in k) and "router" not in k
    }

    # - save model
    if lora_state_dict:
        ckpt_model = os.path.join(output_dir, f"{FSDP_MODEL_NAME}_{MODEL_INDEX}")
        os.makedirs(ckpt_model, exist_ok=True)
        logger.info(f"Saving lora model to {ckpt_model}")
        dcp.save(
            state_dict={KEY_MODEL: lora_state_dict},
            storage_writer=dcp.FileSystemWriter(ckpt_model),
            planner=DefaultSavePlanner(),
        )
    else:
        ckpt_model = os.path.join(output_dir, f"{FSDP_MODEL_NAME}_{MODEL_INDEX}")
        os.makedirs(ckpt_model, exist_ok=True)
        logger.info(f"Saving ft model to {ckpt_model}")
        dcp.save(
            state_dict={KEY_MODEL: model_state_dict},
            storage_writer=dcp.FileSystemWriter(ckpt_model),
            planner=DefaultSavePlanner(),
        )

    # - save optimizer
    ckpt_opt = os.path.join(output_dir, f"{OPTIMIZER_NAME}_{optimizer_index}")
    os.makedirs(ckpt_opt, exist_ok=True)
    logger.info(f"Saving Optimizer state to {ckpt_opt}")
    dcp.save(
        state_dict={KEY_OPTIMIZER: optimizer_state_dict},
        storage_writer=dcp.FileSystemWriter(ckpt_opt),
        planner=DefaultSavePlanner(),
    )
    logger.info(f"Optimizer state saved in {ckpt_opt}")


# rewrite of func from accelerate.utils.fsdp_utils.py
# - empty function, main logic in load_fsdp_optimizer (see below).
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
        state_dict={KEY_MODEL: model_state_dict},
        storage_reader=dcp.FileSystemReader(ckpt_model),
        planner=DefaultLoadPlanner(),
    )

    # - load the optimizer state dict
    ckpt_opt = os.path.join(input_dir, f"{OPTIMIZER_NAME}_{optimizer_index}")
    dcp.load(
        state_dict={KEY_OPTIMIZER: optimizer_state_dict},
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

    # FIXME:
    # - We see errors that occur in optimizer.step()
    # - torch/optim/optimizer.py", line 89, in _use_grad
    # - torch/optim/adamw.py", line 214, in step beta1,
    #   beta2 = cast(Tuple[float, float], group["betas"])
    # - KeyError: 'betas'
    # - Fortunately, this seems to be limited to the empty groups case, where
    #   it seems that it is just the params are not initialized. Since we suppose
    #   these groups are never used, we simply initialize the empty groups with
    #   random values so the errors do not throw.
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


# function to monkey patch accelerator clip grad_norm
def patch_huggingface_clip_grad_norm_fsdp2(accelerator):
    accelerator.clip_grad_norm_ = types.MethodType(clip_grad_norm_, accelerator)


def patch_huggingface_fsdp2_load_full_state_dict():
    # Third Party
    # pylint: disable=import-outside-toplevel
    from fms_acceleration.model_patcher import patch_target_module

    patch_target_module(
        "accelerate.utils.fsdp_utils.fsdp2_load_full_state_dict",
        fsdp2_load_full_state_dict,
    )
    patch_target_module(
        "accelerate.utils.fsdp_utils.fsdp2_prepare_model", fsdp2_prepare_model
    )


# this function implements a trick to get the resolved cache file to acccess the safetensor
# - NOTE: does not work if _dict_from_json_file is not called, such as in the case of GGUF files.
def get_resolved_checkpoint_location(model_name_or_path: str):

    result = None
    _old_func = PretrainedConfig._dict_from_json_file

    def _dict_from_json_file(resolved_config_file):
        nonlocal result
        result = resolved_config_file
        return _old_func(resolved_config_file)

    # make a hook and restrive
    PretrainedConfig._dict_from_json_file = _dict_from_json_file
    PretrainedConfig.from_pretrained(model_name_or_path)
    PretrainedConfig._dict_from_json_file = _old_func
    return os.path.dirname(result)


# function to get the state dict from dcp_checkpoint
def get_state_dict_from_dcp_checkpoint(
    dcp_checkpoint_dir: str,
):
    # guarded, load some internal functions
    # pylint: disable=import-outside-toplevel
    # Third Party
    from torch.distributed.checkpoint.default_planner import _EmptyStateDictLoadPlanner
    from torch.distributed.checkpoint.metadata import STATE_DICT_TYPE
    from torch.distributed.checkpoint.state_dict_loader import _load_state_dict

    sd: STATE_DICT_TYPE = {}
    _load_state_dict(
        sd,
        storage_reader=dcp.FileSystemReader(dcp_checkpoint_dir),
        planner=_EmptyStateDictLoadPlanner(),
        no_dist=True,
    )
    return sd[KEY_MODEL]


# function to get state dict from regular checkpoint
def get_state_dict_from_safe_checkpoint(safe_checkpoint_dir: str):
    safe_index_file = os.path.join(safe_checkpoint_dir, SAFE_WEIGHTS_INDEX_NAME)
    sd = {}
    if os.path.exists(safe_index_file):
        # Load the index for sharded checkpoints
        with open(safe_index_file, "r", encoding="utf-8") as f:
            index = json.load(f)
        shard_files = list(set(index["weight_map"].values()))
        for shard_file in shard_files:
            for key, v in load_file(
                os.path.join(safe_checkpoint_dir, shard_file)
            ).items():
                sd[key] = v

        return sd
    # No index file found, so assume the checkpoint is not sharded.
    checkpoint_file = os.path.join(safe_checkpoint_dir, "model.safetensors")
    if os.path.exists(checkpoint_file):
        for key, v in load_file(checkpoint_file).items():
            sd[key] = v

        return sd
    files = [
        f for f in os.listdir(safe_checkpoint_dir) if f.endswith("model.safetensors")
    ]
    if len(files) == 1:
        checkpoint_file = os.path.join(safe_checkpoint_dir, files[0])
        for key, v in load_file(checkpoint_file).items():
            sd[key] = v

        return sd
    raise FileNotFoundError("No valid safetensors checkpoint found in directory.")


# function to get the ScatterMoE state dict from its DCP checkpoint
# - if the original pretrained_model_name_or_path is specified, will use the checkpoint as hints
#   to map the ScatterMoE checkpoint to that of the original model. This is useful so that we
#   can restore the checkpoint to be loaded by the original architecture.
def recover_original_state_dict_from_checkpoint(
    sd: Dict,
    pretrained_model_name_or_path: str = None,
):
    """
    Parameters:
        dcp_checkpoint_dir (str): the DCP to be converted.
        pretrained_model_name_or_path (str): Optional, if provided we will
            use the hints to remap the
    """

    # reference dcp_to_torch_save from torch.distributed.checkpoint.format_utils.py
    # - strategy is to use _EmptyStateDictLoadPlanner to populate the state dict, then we remap

    # now do the remap
    loc = get_resolved_checkpoint_location(pretrained_model_name_or_path)

    weight_map = load_weight_map(loc, "model.safetensors", FILE_SAFETENSOR_INDEX)

    # config
    config = PretrainedConfig.from_pretrained(pretrained_model_name_or_path)

    (
        _,
        router_name,
        expert_name,
        __,
        sharded_expert_ckpt,
    ) = get_scattermoe_conv_spec_from_archs(config.architectures)

    # the sd from the module swap must have keys like
    # 'model.layers.0.block_sparse_moe.w1.weight'
    # 'model.layers.0.block_sparse_moe.w2.weight'
    # 'model.layers.0.block_sparse_moe.router.weight'
    # so we use this fact to infer that
    # prefix = model.layers.0 and module_name = block_sparse_moe

    def _infer_prefixes_and_module_names(
        sd_keys: List[str],
        min_count: int = 3,
    ):
        _name = "|".join([PARAM_NAME_ROUTER_SCATTERMOE, *PARAM_NAME_WEIGHT_SCATTERMOE])
        # pylint: disable=anomalous-backslash-in-string
        _reg = re.compile(f"(.*)\.({_name})\.weight")
        found = {}

        for k in sd_keys:
            m = _reg.match(k)
            if m is None:
                continue

            prefix, _ = m.groups()
            found[prefix] = 1 + found.get(prefix, 0)

        results = []
        for prefix, cnt in found.items():
            # if at least router, w1 and w2 are found, take it
            # otherwise we delete
            if cnt >= min_count:
                results.append(prefix)

        return results

    for prefix in _infer_prefixes_and_module_names(sd.keys()):
        prefix = prefix.split(".")
        prefix, module_name = ".".join(prefix[:-1]), prefix[-1]

        # checkpoint metadata is will be a  map
        # key -> list of tuples
        # where each in the list is (param_name, stfile)
        # - if the list is larger than one, it means that the
        #   actual model has a sharded checkpoint

        # defaultdict(list,
        #     {'w1.weight': [('model.layers.0.block_sparse_moe.input_linear.weight',
        #        'model-00001-of-00002.safetensors')],
        #      'w3.weight': [('model.layers.0.block_sparse_moe.input_linear.weight',
        #        'model-00001-of-00002.safetensors')],
        #      'w2.weight': [('model.layers.0.block_sparse_moe.output_linear.weight',
        #        'model-00001-of-00002.safetensors')],
        #      'router.weight': [('model.layers.0.block_sparse_moe.router.layer.weight',
        #        'model-00001-of-00002.safetensors')]})

        checkpoint_metadata = get_checkpoint_meta_from_sharded_safetensor(
            weight_map,
            prefix,
            module_name,
            router_name,
            expert_name,
        )

        model2scatter = defaultdict(dict)
        # construct a map of model_key -> {scatter_key: [params, ...]}
        # - if the param list > 1, that means many scatter keys map to 1
        #   model param and they need to be cat
        for scatter_key, list_of_params in checkpoint_metadata.items():
            scatter_key_fqdn = ".".join([prefix, module_name, scatter_key])
            scatter_param = sd[scatter_key_fqdn]

            # remove from state dict
            del sd[scatter_key_fqdn]

            n = len(list_of_params)
            if scatter_key.startswith(PARAM_NAME_ROUTER_SCATTERMOE):
                assert n == 1, "Router parameters should not be sharded."
            elif not sharded_expert_ckpt:
                assert n == 1, "Expert weights expected to be non-sharded."
            else:
                # if sharded, we just assume that there should be 1 expert
                # per shard
                assert (
                    n == scatter_param.shape[0]
                ), "Sharded expert weights should be 1 expert per shard."

            if any(scatter_key.startswith(k) for k in PARAM_NAME_WEIGHT_SCATTERMOE):
                scatter_param = scatter_param.permute(0, 2, 1)

            # go through all the model keys

            for i, (model_key, _) in enumerate(list_of_params):
                if n == 1:
                    # handles routers and non-sharded experts case
                    _param = scatter_param
                else:
                    # then it needs to be sharded
                    _param = scatter_param[i]

                model2scatter[model_key][scatter_key] = _param

        # replace them back in the sd
        for model_key in list(model2scatter.keys()):

            scatter_params = model2scatter[model_key]

            # - there is an assumption that the ifthere is a cat, then
            #  it will go by order of scatter keys
            scatter_keys = sorted(scatter_params.keys())

            assert (
                len(scatter_keys) > 0
            ), f"Obtained zero scatter keys for model_key '{model_key}'"

            if len(scatter_keys) == 1:
                sd[model_key] = scatter_params[scatter_keys[0]]
            else:
                # unfortunately, there this is a in
                # scattermoe_state_dict._maybe_reshape_scattermoe_expert_weights
                # that we split on the dim=1, so we cat back on that
                sd[model_key] = torch.cat(
                    [scatter_params[k] for k in scatter_keys], dim=1
                )

            # remove from this intemediate mapping
            del model2scatter[model_key]

        rem_keys = ",".join(list(model2scatter))
        assert len(rem_keys) == 0, f"Did not handle model parameters '{rem_keys}'"

    return sd


def save_sharded_safetensors(
    input_state_dict: Dict,
    save_directory: str,
    metadata: Dict,
    max_shard_size: Union[int, str] = "5GB",
    lora: bool = False,
):
    if not lora:
        filename_pattern = SAFE_WEIGHTS_NAME.replace(".bin", "{suffix}.bin").replace(
            ".safetensors", "{suffix}.safetensors"
        )
        state_dict_split = split_torch_state_dict_into_shards(
            input_state_dict,
            filename_pattern=filename_pattern,
            max_shard_size=max_shard_size,
        )

        index = {
            "metadata": state_dict_split.metadata,
            "weight_map": state_dict_split.tensor_to_filename,
        }
        # Save the index
        with open(
            os.path.join(save_directory, SAFE_WEIGHTS_INDEX_NAME), "w", encoding="utf-8"
        ) as f:
            content = json.dumps(index, indent=2, sort_keys=True) + "\n"
            f.write(content)

        filename_to_tensors = state_dict_split.filename_to_tensors.items()
        for shard_file, tensors in filename_to_tensors:
            shard = {
                tensor: input_state_dict[tensor].contiguous() for tensor in tensors
            }
            save_file(
                shard, os.path.join(save_directory, shard_file), metadata=metadata
            )
    else:
        filename_pattern = ADAPTER_SAFE_WEIGHTS_NAME.replace(
            ".bin", "{suffix}.bin"
        ).replace(".safetensors", "{suffix}.safetensors")
        state_dict_split = split_torch_state_dict_into_shards(
            input_state_dict,
            filename_pattern=filename_pattern,
            max_shard_size=max_shard_size,
        )
        filename_to_tensors = state_dict_split.filename_to_tensors.items()
        for shard_file, tensors in filename_to_tensors:
            shard = {
                tensor: input_state_dict[tensor].contiguous() for tensor in tensors
            }
            save_file(
                shard, os.path.join(save_directory, shard_file), metadata=metadata
            )


# --------------------------- SCRIPT -------------------------


def recover_safetensors_from_dcp(
    checkpoint_dir, pretrained_model_name_or_path, output_dir
):
    if checkpoint_dir.startswith(FSDP_MODEL_NAME):
        loader = get_state_dict_from_dcp_checkpoint
    else:
        fsdp_checkpoint_dirs = [
            x
            for x in os.listdir(checkpoint_dir)
            if os.path.isdir(os.path.join(checkpoint_dir, x))
            and x.startswith(FSDP_MODEL_NAME)
        ]
        if len(fsdp_checkpoint_dirs) == 1:
            checkpoint_dir = os.path.join(checkpoint_dir, fsdp_checkpoint_dirs[0])
            loader = get_state_dict_from_dcp_checkpoint
        elif len(fsdp_checkpoint_dirs) > 1:
            raise ValueError(
                f"Found > 1 dirs in dcp checkpoint dir {checkpoint_dir} "
                f"that starts with {FSDP_MODEL_NAME}. Please spectify the exact dir."
            )
        else:
            # then take it as a safetensors checkpoint
            # - do not support .bin checkpoints
            loader = get_state_dict_from_safe_checkpoint

    # - pretrained model name
    _name_or_path = pretrained_model_name_or_path

    # assume output directory exists, we do not create it
    # - copy the config file if exists
    config_file = os.path.join(checkpoint_dir, CONFIG_NAME)
    target_config_file = os.path.join(output_dir, CONFIG_NAME)
    if os.path.exists(config_file):
        shutil.copyfile(config_file, target_config_file)

        # try to populate pretrained_model_name_or_path from the config path
        # if it was None
        if not _name_or_path:
            with open(target_config_file, "r", encoding="utf-8") as file:
                _name_or_path = json.load(file).get("_name_or_path")

    # get the state_dict
    state_dict = loader(checkpoint_dir)

    # filter out additional names created by lora tuning
    # create switch based on state dict for future use
    new_state_dict = {}
    lora = False
    for name, param in state_dict.items():
        # if lora weight, set lora switch to true
        if "lora_A" in name or "lora_B" in name:
            lora = True
        # if lora naming convention, convert to traditional
        if "base_model.model." in name:
            name = name.replace("base_model.model.", "", 1)
        if "default." in name:
            name = name.replace("default.", "", 1)
        new_state_dict[name] = param

    # recover the original state dict
    state_dict = recover_original_state_dict_from_checkpoint(
        new_state_dict, _name_or_path
    )

    # save it as a safetensors file
    save_sharded_safetensors(
        {k: v.contiguous() for k, v in state_dict.items()},
        output_dir,
        metadata={"format": "pt"},
        lora=lora,
    )


def clip_grad_norm_(self, parameters, max_norm, norm_type=2):
    """grad norm patch when EP is enabled"""
    # code inspired from
    # https://github.com/pytorch/torchtitan/blob/72b16b13abc88ba08f3e1796e5caee09abd94554/torchtitan/distributed/utils.py#L398
    ep_params = []
    non_ep_params = []
    ep_grads = []
    non_ep_grads = []

    for p in parameters:
        if p.grad is None:
            continue
        if (
            p.device_mesh.mesh_dim_names
            and KEY_EXPERT_PARALLEL in p.device_mesh.mesh_dim_names
        ):
            ep_params.append(p)
            ep_grads.append(p.grad)
        else:
            non_ep_params.append(p)
            non_ep_grads.append(p.grad)
    ep_grads_total_norm = torch.nn.utils.get_total_norm(
        ep_grads, norm_type, False, True
    )

    if isinstance(ep_grads_total_norm, DTensor):
        ep_grads_total_norm = ep_grads_total_norm.full_tensor()

    non_ep_grads_total_norm = torch.nn.utils.get_total_norm(
        non_ep_grads, norm_type, False, True
    ).full_tensor()

    if math.isinf(norm_type):
        total_norm = torch.maximum(ep_grads_total_norm, non_ep_grads_total_norm)
    else:
        total_norm = ep_grads_total_norm**norm_type + non_ep_grads_total_norm**norm_type
        total_norm **= 1.0 / norm_type

    torch.nn.utils.clip_grads_with_norm_(ep_params, max_norm, total_norm, True)
    torch.nn.utils.clip_grads_with_norm_(non_ep_params, max_norm, total_norm, True)

    return total_norm


# have it serve as a conversion script
if __name__ == "__main__":
    # Standard
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Utility for converting ScatterMoE checkpoint back to the "
            "orginal state dict format. "
            "The ScatterMoE checkpoint was saved after the pretrained model "
            "had been converted by a module swap, hence the state dict will "
            "no longer resemble the original. This utility creaes"
        )
    )

    parser.add_argument(
        "checkpoint_dir",
        help="Path to the checkpoint.",
    )

    parser.add_argument(
        "output_dir", help="Path to the location to write the converted checkpoint."
    )

    parser.add_argument(
        "pretrained_model_name_or_path",
        help=(
            "In order to reconstruct the state dict, we requre hints from "
            "the original pretrained model checkpoint (from which this "
            "checkpoint is obtained)."
        ),
        default=None,
    )

    args = parser.parse_args()
    recover_safetensors_from_dcp(
        args.checkpoint_dir, args.pretrained_model_name_or_path, args.output_dir
    )


# code taken from HF accelerate and modified
def fsdp2_load_full_state_dict(accelerator, model: torch.nn.Module, full_sd: dict):
    """
    Loads the full state dict (could be only on rank 0) into the sharded model. 
    This is done by broadcasting the parameters from rank 0 to all other ranks. 
    This function modifies the model in-place.

    Args:
        accelerator (`Accelerator`): The accelerator instance
        model (`torch.nn.Module`):
            The model to load the state dict into, expected to be on meta device 
            or a VRAM spike can occur
        full_sd (`dict`): The full state dict to load, can only be on rank 0
    """
    # Third Party
    # pylint: disable=import-outside-toplevel
    from torch.distributed.tensor import distribute_tensor
    import torch.distributed as dist

    # Model was previously copied to meta device
    meta_sharded_sd = model.state_dict()
    sharded_sd = {}

    # Rank 0 distributes the full state dict to other ranks
    def _infer_parameter_dtype(model, param_name, empty_param):
        try:
            old_param = model.get_parameter_or_buffer(param_name)
        except AttributeError:
            # Need this for LORA, as there some params are not *parameters* of sorts
            base_param_name, local_param_name = param_name.rsplit(".", 1)
            submodule = model.get_submodule(base_param_name)
            old_param = getattr(submodule, local_param_name)

        is_torch_e4m3fn_available = hasattr(torch, "float8_e4m3fn")
        casting_dtype = None
        is_param_float8_e4m3fn = (
            is_torch_e4m3fn_available and empty_param.dtype == torch.float8_e4m3fn
        )

        if empty_param.dtype.is_floating_point and not is_param_float8_e4m3fn:
            casting_dtype = old_param.dtype

        return old_param is not None and old_param.is_contiguous(), casting_dtype

    def _cast_and_contiguous(tensor, to_contiguous, dtype):
        if dtype is not None:
            tensor = tensor.to(dtype=dtype)
        if to_contiguous:
            tensor = tensor.contiguous()
        return tensor

    ignored_params = {
        p.detach()
        # pylint: disable=undefined-variable
        for p in get_parameters_from_modules(
            accelerator.state.fsdp_plugin.ignored_modules, model, accelerator.device
        )
    }
    if accelerator.is_main_process:
        for (param_name, full_param), sharded_param in zip(
            full_sd.items(), meta_sharded_sd.values()
        ):
            # ignored params will not be on meta device
            # and not handled by FSDP
            if sharded_param.device != torch.device("meta"):
                sharded_sd[param_name] = sharded_param
            else:
                device_mesh = sharded_param.device_mesh
                full_param = full_param.detach().to(device_mesh.device_type)
                dist.broadcast(full_param, src=0, group=dist.group.WORLD)
                sharded_tensor = distribute_tensor(
                    full_param, device_mesh, sharded_param.placements
                )
                to_contiguous, casting_dtype = _infer_parameter_dtype(
                    model,
                    param_name,
                    full_param,
                )
                sharded_tensor = _cast_and_contiguous(
                    sharded_tensor, to_contiguous, casting_dtype
                )
                sharded_sd[param_name] = sharded_tensor
    # We need this else to have a matching `broadcast` for all of the ranks, else we deadlock
    else:
        for param_name, sharded_param in meta_sharded_sd.items():
            # ignored params will not be on meta device
            # and not handled by FSDP
            if sharded_param.device != torch.device("meta"):
                sharded_sd[param_name] = sharded_param
            else:
                device_mesh = sharded_param.device_mesh
                full_tensor = torch.empty(
                    sharded_param.size(),
                    device=device_mesh.device_type,
                    dtype=sharded_param.dtype,
                )
                dist.broadcast(full_tensor, src=0, group=dist.group.WORLD)
                sharded_tensor = distribute_tensor(
                    full_tensor, device_mesh, sharded_param.placements
                )
                to_contiguous, casting_dtype = _infer_parameter_dtype(
                    model,
                    param_name,
                    full_tensor,
                )
                sharded_tensor = _cast_and_contiguous(
                    sharded_tensor, to_contiguous, casting_dtype
                )
                sharded_sd[param_name] = sharded_tensor

    # we set `assign=True` because our params are on meta device
    model.load_state_dict(sharded_sd, assign=True)
    return model


# code taken from HF accelerate and modified
def fsdp2_prepare_model(accelerator, model: torch.nn.Module) -> torch.nn.Module:
    """Prepares the model for FSDP2 in-place. Also returns the model to avoid 
    misuse of the original model.

    Args:
        accelerator (`Accelerator`): The accelerator instance
        model (`torch.nn.Module`): The model to prepare

    Returns:
        `torch.nn.Module`: Prepared model
    """
    # Third Party
    # pylint: disable=import-outside-toplevel
    from torch.distributed.fsdp import FSDPModule, MixedPrecisionPolicy, fully_shard

    is_type_fsdp = isinstance(model, FSDPModule) or (
        # pylint: disable=undefined-variable
        is_compiled_module(model) and isinstance(model._orig_mod, FSDPModule)
    )
    if is_type_fsdp:
        return model

    fsdp2_plugin = accelerator.state.fsdp_plugin

    fsdp2_plugin.set_auto_wrap_policy(model)

    original_sd = model.state_dict()
    mesh = getattr(accelerator, "torch_device_mesh", None)

    fsdp2_kwargs = {
        "reshard_after_forward": fsdp2_plugin.reshard_after_forward,
        "offload_policy": fsdp2_plugin.cpu_offload,
        # `fully_shard` doesn't accept `None` in case of `MixedPrecisionPolicy`
        "mp_policy": fsdp2_plugin.mixed_precision_policy or MixedPrecisionPolicy(),
        "mesh": (
            mesh[tuple(accelerator.parallelism_config.fsdp_dim_names)]
            if mesh is not None
            else None
        ),
        # pylint: disable=undefined-variable
        "ignored_params": get_parameters_from_modules(
            fsdp2_plugin.ignored_modules, model, accelerator.device
        ),
    }

    model_has_params4bit = False
    for _, param in model.named_parameters():
        # this is a temporary fix whereby loading models with bnb params
        # cannot be moved from GPU to a meta device due with FSDP2 because
        # torch operations don't return the original class type bypassing the
        # move to meta will still cause the VRAM spike, but at least it still will load
        if param.__class__.__name__ == "Params4bit":
            model_has_params4bit = True
            break

    if fsdp2_plugin.cpu_ram_efficient_loading and not model_has_params4bit:
        # pylint: disable=undefined-variable
        non_persistent_buffer_fqns = get_non_persistent_buffers(
            model, recurse=True, fqns=True
        )
        # pylint: disable=undefined-variable
        original_non_persistent_buffers = copy.deepcopy(
            {k: v for k, v in model.named_buffers() if k in non_persistent_buffer_fqns}
        )
        # We move the model parameters to meta device that are managed by FSDPv2,
        # as then sharding happens on meta device
        with torch.no_grad():
            for _, module in model.named_modules():
                for param_name, param in list(module.named_parameters(recurse=False)):
                    if param not in fsdp2_kwargs["ignored_params"]:
                        # Create new parameter on meta device
                        meta_param = torch.nn.Parameter(
                            torch.empty(param.shape, dtype=param.dtype, device="meta"),
                            requires_grad=param.requires_grad,
                        )
                        setattr(module, param_name, meta_param)
        # model = model.to(torch.device("meta"))
        # We need to re-tie the weights, not exactly sure why, but if we don't do this,
        # reference to `lm_head/embed_tokens` stay hanging -> more VRAM usage
        # We assume `transformers` models have a `tie_weights` method if they support it
        if hasattr(model, "tie_weights"):
            model.tie_weights()

    # pylint: disable=undefined-variable
    auto_wrap_policy_func = fsdp2_prepare_auto_wrap_policy(fsdp2_plugin, model)
    if auto_wrap_policy_func is not None:
        # We skip the model itself, as that one is always wrapped
        # pylint: disable=undefined-variable
        for module in get_module_children_bottom_up(model)[:-1]:
            if auto_wrap_policy_func(module) and not isinstance(module, FSDPModule):
                fully_shard(module, **fsdp2_kwargs)

    if not isinstance(model, FSDPModule):
        fully_shard(model, **fsdp2_kwargs)

    if fsdp2_plugin.cpu_ram_efficient_loading:
        # If `cpu_ram_efficient_loading` is enabled, only rank 0 loads the weights
        # Other ranks have an empty model on `meta` device, so we need to distribute
        # the weights properly
        fsdp2_load_full_state_dict(accelerator, model, original_sd)

    if fsdp2_plugin.cpu_ram_efficient_loading and not model_has_params4bit:
        # We re-register the buffers, as they may not be in the state_dict
        for fqn, buffer_tensor in original_non_persistent_buffers.items():
            buffer_tensor = buffer_tensor.to(accelerator.device)

            if "." in fqn:
                parent_fqn, local_buffer_name = fqn.rsplit(".", 1)
                parent_module = model.get_submodule(parent_fqn)
            else:
                local_buffer_name = fqn
                parent_module = model

            parent_module.register_buffer(
                local_buffer_name, buffer_tensor, persistent=False
            )

        # We need to tie the weights again, as call to `load_full_state_dict` breaks the tie
        # Needs to be called both here and above
        # removing this call makes the have slightly different loss
        # removing the call above leads to extra memory usage as explained in the comment above
        if hasattr(model, "tie_weights"):
            model.tie_weights()

    # There is no `dtype` attribution for nn.Module
    # Set it to None if it doesn't exist and do the upcast always
    model_dtype = getattr(model, "dtype", None)
    if accelerator.mixed_precision != "no" and (
        model_dtype is None or model_dtype != torch.float32
    ):
        # We upcast the model according to `deepspeed`'s implementation
        # More info about this can be found in `accelerator.py:prepare_model`s
        # FSDP1 section
        model = model.to(torch.float32)
        if accelerator.is_main_process:
            # TODO(siro1): Add a warning for each parameter that was upcasted
            # pylint: disable=undefined-variable
            warnings.warn(
                "FSDP upcast of low precision parameters to fp32 (since mixed_precision != 'no')"
                "may affect the precision of model checkpoints."
            )
    return model
