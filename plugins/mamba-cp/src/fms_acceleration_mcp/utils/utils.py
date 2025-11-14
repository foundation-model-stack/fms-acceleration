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
from typing import Dict
# pylint: disable=import-error
from torch.distributed._tensor.device_mesh import DeviceMesh, init_device_mesh
import torch
from transformers.modeling_utils import is_fsdp_enabled, is_local_dist_rank_0
from tqdm import tqdm

key_ep = "cp"
key_rep = "dp_shard"

def hf_config_ssm_config(hf_config) -> Dict:
    config_ssm = {}
    config_ssm["d_model"] = hf_config.hidden_size
    config_ssm["n_layer"] = hf_config.num_hidden_layers
    config_ssm["tie_embeddings"] = hf_config.tie_word_embeddings
    config_ssm["d_state"] = 128
    config_ssm["ngroups"] = hf_config.mamba_n_groups
    config_ssm["rmsnorm"] = True
    config_ssm["chunk_size"] = hf_config.mamba_chunk_size
    config_ssm["conv_bias"] = hf_config.mamba_conv_bias
    config_ssm["d_conv"] = hf_config.mamba_d_conv
    return config_ssm


def patch_mamba_layers_with_cp_head(
    model,
    checkpoint_name_or_path,
    rank,
    cp_degree,
    world_size,
    cp_mamba_impl,
    cp_attn_impl,
    cp_mamba_recompute
):
    config_ssm = hf_config_ssm_config(model.config)
    device = torch.device(f"cuda:{rank}")
    if is_fsdp_enabled():
        device = torch.device("cpu")
    try:
        from mamba_ssm.modules.mamba2_cp import Mamba2CP
    except ImportError:
        ValueError(
            "Mamba2CP is required to enable context parallelism for mamba layers"
        )
    rep_size = world_size // cp_degree

    if cp_degree == 1:
        raise ValueError("CP degree can't be one")
    elif rep_size == 1:
        device_mesh = init_device_mesh(
            "cuda",
            (cp_degree,),
            mesh_dim_names=(key_ep,),
        )
    else:
        device_mesh = init_device_mesh(
            "cuda",
            (rep_size, cp_degree),
            mesh_dim_names=(key_rep, key_ep),
        )

    cp_args = {
        "cp_mesh": device_mesh[key_ep],
        "cp_mamba_impl": cp_mamba_impl,
        "cp_attn_impl": cp_attn_impl,
        "cp_mamba_recompute": cp_mamba_recompute,
    }
    
    with torch.no_grad():
        for layer in tqdm(model.model.layers, desc="Swapping mamba layers"):
            mamba_layer = Mamba2CP(**config_ssm, **cp_args)
            mamba_layer.load_state_dict(layer.mamba.state_dict())
            setattr(layer, "mamba", mamba_layer)
            layer.to(device)

    if hasattr(model, "tie_weights"):
        model.tie_weights()
