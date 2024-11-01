
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
from collections import defaultdict, OrderedDict
from typing import Dict, List, Tuple, Type, Union
import re, os
import torch
from contextlib import ExitStack
from safetensors import safe_open
from .scattermoe_constants import KEY_SCATTERMOE_ROUTER, DIM_EXPERT

# This function creates a dictionary of keys and paths into the the sharded
# safetensors checkpoint file, that are relevant to the "prefix" and "instance_name"
# being pased in.
# - the keys point to modules found in megablocks.layers.dmoe.dMoE, the distributed
#   expert module provided by megablocks.
# - the values are tuples pointing to the keys within the checkpoint file.
#
# Example: if prefix="module.layers.0" and instance_name="block_sparse_moe", then a dictionary
# of the following will be returned:
# {
#   'w1.weight': [
#      (
#        'model.layers.0.block_sparse_moe.experts.0.w1.weight',
#        'model-00001-of-00019.safetensors'
#      ),
#      (
#         'model.layers.0.block_sparse_moe.experts.1.w1.weight',
#         'model-00001-of-00019.safetensors'
#      ),
#      ...
#    ]
#    'w2.weight': [...],
#    'w3.weight': [...],
#    'router.weight': [
#       (
#          'model.layers.0.block_sparse_moe.gate.weight',
#          'model-00001-of-00019.safetensors'
#       )
#     ]
# }
# 
# or the non-sharded case (and possibly fused case)
# {
#   'w1.weight': [
#      (
#        'model.layers.0.block_sparse_moe.input_linear.layer.weight',
#        'model-00001-of-00001.safetensors'
#      ),
#    ],
#    ...
#   'w3.weight': [
#      (
#        'model.layers.0.block_sparse_moe.input_linear.layer.weight',
#        'model-00001-of-00001.safetensors'
#      ),
#    ]
# }


def get_checkpoint_meta_from_sharded_safetensor(
    weight_map: Dict,
    prefix: str,  # e.g., 'model.layers.0,
    instance_name: str,  # e.g., block_sparse_moe
    router_name: str = "gate",  # e.g., named "gate" within block_sparse_moe
    expert_name: str = "experts",  # e.g., named "experts" within block_sparse_moe
    expert_map: Dict = None, # map -> [w1,w2,w3]
) -> Dict[str, List[Tuple]]:
    # insert in order
    def _insert(L: List, i: int, v):
        n = len(L)
        if i < n:
            L[i] = v
            return

        n = i - n + 1
        while n > 0:
            L.append(None)
            n -= 1
        L[i] = v

    # if expert_name = input_linear|output_linear|input_linear
    # - in this case will map 
    # - input_linear: [w1, w3], output_linear: {w2}
    # - will assume the latter has double the size and can
    #   be split.
    if expert_map is None:
        if '|' in expert_name:
            expert_map = {}
            _names = expert_name.split('|')
            assert len(_names) in {2,3}, "expert name map has to be length 2/3"
            
            for i, n in enumerate(_names):
                if n not in expert_map:
                    expert_map[n] = []
                expert_map[n].append(f'w{i+1}')
        else:
            expert_map = {x: [x] for x in ['w1', 'w2', 'w3']}

    # state dict -> weights
    # 'router.weight': [(k, file),...]
    # `w1.weight`: [...]
    _map = defaultdict(list)
    prefix = f"{prefix}.{instance_name}."
    for k, stfile in weight_map.items():
        if not k.startswith(prefix):
            continue

        # e.g. after replacement we get
        # - gate.weight
        # - experts.0.w1.weight
        rel_k = k.replace(prefix, "")
        # pylint: disable=anomalous-backslash-in-string
        m = re.match(f"({router_name}|{expert_name})\.?(\d+)?\.?(\w+)?\.weight", rel_k)
        if m is None:
            raise ValueError(
                f"Unable to handle key '{k}' with provided router_name "
                f"'{router_name}' or expert_name '{expert_name}'"
            )
        if m.group(1) == router_name:
            _map[KEY_SCATTERMOE_ROUTER].append((k, stfile))
        elif m.group(1) in expert_name:
            index = m.group(2)
            index = 0 if index is None else int(index)
            mod = None
            for mod in expert_map.get(m.group(1), expert_map.get(m.group(3))):
                _insert(_map[f"{mod}.weight"], index, (k, stfile))

            assert mod is not None, f"cannot map \'{rel_k}\'"

    if len(_map) == 0:
        raise ValueError(
            f"Could not get safetensor map for '{prefix}' and '{instance_name}'"
        )

    return _map

# if the weight is a scattermoe expert weight, need some reshaping
def _maybe_reshape_scattermoe_expert_weights(
    scatter_key: str, param: torch.Tensor,
    num_experts: int,
    intermediate_size: int,
):
    (
        _is_w1, _is_w2, _is_w3
    ) = [ f'{x}.weight' in scatter_key for x in ['w1', 'w2', 'w3']] 

    if _is_w1 or _is_w2 or _is_w3:
        if len(param.shape) == 2:
            param = param.view(num_experts, -1, param.shape[-1])

        if _is_w1 or _is_w3:
            if param.shape[-2] == (2 * intermediate_size):
                # cut it 
                if _is_w1:
                    param = param[..., :intermediate_size, :]
                else:
                    param = param[..., intermediate_size:, :]

            # asumme these are linears
            # assert param.shape[-2] == intermediate_size, "wrong intermediate size"
            # assert param.shape[-1] == hidden_size, "wrong hidden size"

        # have to transpose for weights since scattermoe accepts the differen
        # order
        param = param.permute(0, 2, 1)

    return param

def convert_state_dict(
    prefix: str,
    metadata: Dict,
    state_dict: Dict,
    num_experts: int,
    intermediate_size: int,
    dtype: torch.dtype = None,
):
    target = OrderedDict()

    for scatter_key, vs in metadata.items():
        for state_key, _ in vs:
            state_key = state_key.replace(prefix, "")
            param = state_dict[state_key]
            param = _maybe_reshape_scattermoe_expert_weights(
                scatter_key, param, num_experts, intermediate_size
            )
            if dtype is not None:
                param = param.to(dtype)
            target[scatter_key] = param

    return target

def get_state_dict_from_checkpoint_metadata(
    directory: str,
    checkpoint_metadata: Dict[str, List[Tuple]],
    num_experts: int,
    intermediate_size: int, 
    expert_shards: List[int] = None,
    dtype: torch.dtype = None,
):
    target = OrderedDict()

    # typically they all should be same file, but to play safe, load the checkpoint file onto
    # cpu first since we may not need all weights in that file.
    with ExitStack() as stack:
        files = {}
        for _, vs in checkpoint_metadata.items():
            for _, fi in vs:
                if fi not in files:
                    files[fi] = stack.enter_context(
                        safe_open(
                            os.path.join(directory, fi), framework="pt", device="cpu"
                        )
                    )

        # go by one weight at a time.
        for scatter_key, vs in checkpoint_metadata.items():

            if KEY_SCATTERMOE_ROUTER in scatter_key:
                k, fi = vs[0] # only one item
                param = files[fi].get_tensor(k)

            elif len(vs) == 1:
                k, fi = vs[0] # only one item
                # if its a non-router weight and its non-sharded
                param = files[fi].get_tensor(k)
                assert len(param.shape) == 3, \
                    f"Expected 3D tensor for checkpoints with non-sharded experts, but got shape {param.shape}."

            else:
                # handle sharding if the checkpoint shards experts
                # - 
                data = []
                if expert_shards is not None:
                    vs = [vs[i] for i in expert_shards]

                for k, fi in vs:
                    T = files[fi].get_tensor(k)
                    assert len(T.shape) == 2, \
                        f"Expected 2D tensor for checkpoints with sharded experts, but got shape {T.shape}."

                    T = T.unsqueeze(0)
                    data.append(T)

                param = torch.concat(data, dim=DIM_EXPERT)

            param = _maybe_reshape_scattermoe_expert_weights(
                scatter_key, param, num_experts, intermediate_size
            )
            if dtype is not None:
                param = param.to(dtype)
            
            target[scatter_key] = param

    return target