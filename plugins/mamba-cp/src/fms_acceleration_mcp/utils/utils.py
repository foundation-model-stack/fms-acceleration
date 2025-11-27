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
from mamba_ssm.modules.mamba2_cp import Mamba2CP

# pylint: disable=import-error
from torch.distributed._tensor.device_mesh import init_device_mesh
from tqdm import tqdm
from transformers.modeling_utils import is_fsdp_enabled
import torch

# to avoid rechunking/sharding of the buffers
# ideally this is not optimal
from torch.distributed.tensor.experimental._attention import _cp_options
_cp_options.enable_load_balance = False


key_cp = "cp"
key_rep = "dp_shard"


def hf_config_ssm_config(hf_config) -> Dict:
    config_ssm = {}
    config_ssm["d_model"] = hf_config.hidden_size
    config_ssm["d_state"] = 128
    config_ssm["ngroups"] = hf_config.mamba_n_groups
    config_ssm["rmsnorm"] = True
    config_ssm["chunk_size"] = hf_config.mamba_chunk_size
    config_ssm["conv_bias"] = hf_config.mamba_conv_bias
    config_ssm["d_conv"] = hf_config.mamba_d_conv
    return config_ssm


class Mamba2CPHF(Mamba2CP):
    def forward(
        self,
        hidden_states,
        cache_params=None,
        cache_position=None,
        attention_mask=None,
        seq_idx=None,
        **kwargs,
    ):
        return super().forward(
            u=hidden_states,
            seqlen=None,
            seq_idx=None,
            cu_seqlens=None,
            inference_params=None,
        )


def patch_mamba_layers_with_cp_head(
    model,
    checkpoint_name_or_path,
    rank,
    cp_degree,
    world_size,
    cp_mamba_impl,
    cp_mamba_recompute,
):

    config_ssm = hf_config_ssm_config(model.config)
    device = torch.device(f"cuda:{rank}")
    if is_fsdp_enabled():
        device = torch.device("cpu")
    rep_size = world_size // cp_degree

    if cp_degree == 1:
        raise ValueError("CP degree can't be one")
    if rep_size == 1:
        device_mesh = init_device_mesh(
            "cuda",
            (cp_degree,),
            mesh_dim_names=(key_cp,),
        )
    else:
        device_mesh = init_device_mesh(
            "cuda",
            (rep_size, cp_degree),
            mesh_dim_names=(key_rep, key_cp),
        )

    cp_args = {
        "cp_mesh": device_mesh[key_cp],
        "cp_mamba_impl": cp_mamba_impl,
        "cp_mamba_recompute": cp_mamba_recompute,
    }

    with torch.no_grad():
        dtype = model.dtype
        device = model.device
        for layer in tqdm(model.model.layers, desc="Swapping mamba layers"):
            if hasattr(layer, "mamba") and layer.mamba is not None:
                mamba_layer = Mamba2CPHF(**config_ssm, **cp_args)
                mamba_layer.load_state_dict(layer.mamba.state_dict())
                setattr(layer, "mamba", mamba_layer)
                layer.to(dtype).to(device)

    if hasattr(model, "tie_weights"):
        model.tie_weights()
