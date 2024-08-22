
import torch
from transformers import PretrainedConfig
from transformers.activations import ACT2FN
from typing import Tuple, Dict, List, Type, Union
from torch.distributed._tensor import Placement, Replicate, Shard, distribute_tensor
from torch.distributed._tensor.device_mesh import init_device_mesh, DeviceMesh
import os
from copy import copy
from tqdm import tqdm

from safetensors import safe_open
import json, re
from collections import defaultdict

from accelerate import init_empty_weights

from contextlib import ExitStack

FILE_SAFETENSOR_INDEX = 'model.safetensors.index.json'
KEY_DATA_PARALLEL = 'data_parallel'
KEY_EXPERT_PARALLEL = 'expert_parallel'
DIM_EXPERT = 0

# these depend on the namings in the dMOE
KEY_DMOE_ROUTER = 'router.layer.weight'
KEY_DMOE_EXPERTS = 'experts.mlp'

def get_moe_kwargs(
    config: PretrainedConfig,
    fp16: bool = False,
    bf16: bool = False,
):
    return {
        "hidden_size": config.hidden_size,
        "ffn_hidden_size": config.intermediate_size,
        "moe_num_experts": config.num_local_experts,
        "moe_top_k": config.num_experts_per_tok,
        "moe_expert_model_parallelism": True,
        "memory_optimized_mlp": False,
        "activation_fn": ACT2FN[config.hidden_act],
        "moe_normalize_expert_weights": True,
        "return_bias": False,
        "fp16": fp16,
        "bf16": bf16,
    }

# trick to get the resolved cache file to acccess the safetensor
# NOTE: this does not work if _dict_from_json_file, like GGUF files
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
#   'experts.mlp.w1': [
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
#    'experts.mlp.w2': [...], 
#    'experts.mlp.w3': [...],
#    'router.layer.weight': [
#       (
#          'model.layers.0.block_sparse_moe.gate.weight', 
#          'model-00001-of-00019.safetensors'
#       )
#     ]
# }
def get_checkpoint_meta_from_sharded_safetensor(
    weight_map: Dict,
    prefix: str, # e.g., 'model.layers.0,
    instance_name: str, # e.g., block_sparse_moe
    router_name: str = 'gate', # e.g., named "gate" within block_sparse_moe
    expert_name: str = 'experts' # e.g., named "experts" within block_sparse_moe
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

    # state dict -> weights
    # 'router.layer.weight': [(k, file),...]
    # `experts.mlp.w1`: [...]
    _map = defaultdict(list)
    prefix = f"{prefix}.{instance_name}."
    for k, stfile in weight_map.items():
        if not k.startswith(prefix):
            continue

        # e.g. after replacement we get
        # - gate.weight
        # - experts.0.w1.weight
        rel_k = k.replace(prefix, "")
        m = re.match(
            f'({router_name}|{expert_name})\.?(\d+)?\.?(\w+)?\.weight',
            rel_k
        )
        if m is None:
            raise ValueError(
                f"Unable to handle key '{k}' with provided router_name "
                f"'{router_name}' or expert_name '{expert_name}'"
            )
        if m.group(1) == router_name:
            _map[KEY_DMOE_ROUTER].append((k, stfile))
        elif m.group(1) == expert_name:
            index = int(m.group(2))
            mod = m.group(3)
            _insert(_map[f'{KEY_DMOE_EXPERTS}.{mod}'], index, (k, stfile))

    if len(_map) == 0:
        raise ValueError(
            f"Could not get safetensor map for '{prefix}' and '{instance_name}'"
        )

    return _map

# this function will load the sharded experts onto the device. 
# - this assumes that the "dmoe" module is the megablocks.layers.dmoe.dMoE distributed
#   implementation of the mixture of experts.
def load_sharded_experts_onto_device(
    dmoe: torch.nn.Module,
    directory: str,
    checkpoint_metadata: Dict[str, List[Tuple]],
    device_mesh: DeviceMesh, 
    placements: Placement,
    expert_name: str = 'experts' # e.g., named "experts" within block_sparse_moe
):
    # typically they all should be same file, but to play safe, load the checkpoint file onto
    # cpu first since we may not need all weights in that file.
    with ExitStack() as stack:
        files = {}
        for _, vs in checkpoint_metadata.items():
            for _, fi in vs:
                if fi not in files:
                    files[fi] = stack.enter_context(
                        safe_open(os.path.join(directory, fi), framework='pt', device='cpu')
                    )
            
        # go by one weight at a time.
        # - weight_name: points to megablocks.dmoe
        for weight_name, vs in checkpoint_metadata.items():
            data = []
            for k, fi in vs:
                T = files[fi].get_tensor(k)
                if expert_name in k and k.endswith("weight"):
                    if T.shape[1] > T.shape[0]:
                        T = T.t()
                data.append(T)

            # get the module we want to shard
            name = weight_name.split('.')
            path, name = ".".join(name[:-1]), name[-1]
            mod = dmoe.get_submodule(path)
            mod_dtype = getattr(mod, name).dtype

            # the megablocks dmoe experts the expert features to be on DIM_EXPERT.
            # - concat on dim 0 and distribute
            # - cast to the correct dtype for the module
            param = torch.concat(data, dim=DIM_EXPERT).to(mod_dtype)
            if KEY_DMOE_ROUTER not in weight_name:
                param = torch.nn.Parameter(
                    distribute_tensor(param, device_mesh, placements)
                )
            else:
                # - do not shard the router but load onto device as well
                param = torch.nn.Parameter(
                    param.to(torch.cuda.current_device())
                )
                
            # register the sharded parameter onto the megablocks.dmoe
            mod.register_parameter(name, param)

def shard_moe(
    model: torch.nn.Module,
    moe_cls: Union[str,Type],
    checkpoint_name_or_path: str,
    rank: int, 
    world_size: int,
    moe_kwargs: Dict,
    device_type: str = 'cuda',
    key_dp: str = KEY_DATA_PARALLEL,
    key_ep: str = KEY_EXPERT_PARALLEL,
    router_name: str = 'gate',
    expert_name: str = 'experts',
    shared_mesh_dim: bool = True,
    ep_size: int = 1,
):
    """shard_moe takes a mixture-of-experts huggingface model and shards the experts
    on the current device. All layers layers that have a MoE module will be sharded.

    The function requires "checkpoint_name_or_path" to point to the checkpoint that
    the model has been loaded from, because model could have been loaded on the meta
    device, and in which case would be missing the weights. This function will
    instialize the sharded weights onto the device.

    The sharding has two modes, and depends on world_size and number_of_experts the model
    has. This depends on the setting "shared_mesh_dim" to True or False:
    - if True: then dp and ep will happen on the same device_mesh dimension. This is only possible
        if world_size divides number_of_experts (which requires world_size < num_of_experts).
    - if False: then dp and ep will be seperate device_mesh dimensions. The ep_size will be determined
        by the argument passed in (which needs to be properly set ep_size > 1; the default
        value will raise an assertion).

    Parameters:

        model (module): A valid mixture-of-experts Huggingface model.
        moe_cls (str,type): A module class used to identify the MoE components.
        checkpoint_name_or_path (str): name or path pointing to the weight checkpoint.
        rank (int): rank of the current device.
        world_size (int): total world size.
        moe_kwargs (dict): kwargs to be passed to construct megablocks.layers.arguments for 
            constructing the megablocks.layer.dmoe.dMOE.
        device_type (str): the current device to load the sharded model into.
        key_dp (str): name of the data parallel mesh
        key_ep (str): name of the expert parallel mesh (if initialized).
        router_name (str): module name of the router in moe_cls (e.g., "gate").
        expert_name (str): module name of the experts in moe_cls (e.g., "experts").
        shared_mesh_dim (bool): for the sharding mode, see explanation above.
        ep_size (int): for shard_mesh_dim=False only, see explanation above.

    """
    # guarded import
    from megablocks.layers import dmoe, arguments

    if shared_mesh_dim:
        # if sharing mesh with dp, then the ep_size must be the world_size
        # - in this case ep_shard_factor is ignored
        ep_size = world_size
    else:

        # - moe_kwargs is the constructed by get_moe_kwargs above
        _num_experts = moe_kwargs['moe_num_experts']
        assert _num_experts % ep_size == 0, (
            f"ep_shard factor '{ep_size}' does not divide "
            f"number of experts '{_num_experts}'."
        )

    assert ep_size > 1, "expert_parallel dimension must be set larger than 1" 
    assert world_size % ep_size == 0, (
        f"world_size ({world_size}) not divisible by ep_size ({ep_size})."
    )

    # this function will shard the MOE on this rank
    device = torch.device(f'cuda:{rank}')

    if shared_mesh_dim:
        # in this case we will have a 1D mesh and collapse the 
        # expert parallel with data_parallel

        device_mesh = init_device_mesh(
            device_type,
            (ep_size,),
            mesh_dim_names=(key_dp,),
        )
        key_ep = key_dp
        placements: List[Placement] = [Shard(DIM_EXPERT)]
    else:
        # in this case it will distribute experts on a different
        # mesh dimension than dp. 
        # - this will achieve the effect that the expert sharding can be
        #   hierachical (e.g., can be over a slower network plane since
        #   the communication overhead is less
        dp_size = world_size // ep_size
        device_mesh = init_device_mesh(
            device_type,
            (dp_size, ep_size),
            mesh_dim_names=(key_dp, key_ep),
        )
        # - experts will replicate over the first dimension
        placements: List[Placement] = [Replicate(), Shard(DIM_EXPERT)]

    mp_dmoe_args = arguments.Arguments(
        **moe_kwargs, device=device,
        expert_parallel_group=device_mesh[key_ep].get_group(0)
    )

    assert mp_dmoe_args.moe_num_experts % ep_size == 0, (
        f"number of moe experts ({mp_dmoe_args.moe_num_experts}) "
        f"not divisible by ep_size ({ep_size})."
    )

    # for all the MoE related params, e.g., gate, experts
    # get a dictionary
    # parent_mod: (child_instance_name, [list of fqdn keys])
    found = {}
    for name, mod in model.named_modules():
        name = name.split('.')
        parent, child = ".".join(name[:-1]), name[-1]

        # check the module depending if moe_cls is a str or class
        if (
            mod.__class__.__name__ == moe_cls if
            isinstance(moe_cls, str) else
            isinstance(mod, moe_cls)
        ):
            fqdn_keys = [ # all params, including childs'
                f'{parent}.{child}.{n}'
                for n, _ in mod.named_parameters()
            ]

            # check if there are any biases in any of the experts
            # if there are biases
            # Assumption: assume that if one expert has bias,then the others
            # will have it to
            has_bias = any(
                expert_name in k and k.endswith('bias') 
                for k in fqdn_keys
            )

            found[parent] = (child, fqdn_keys, has_bias)

    moe_module_names = set()

    # NOTE: for now we only support sharded safetensors
    # - most MOE models should be used using this checkpoint format
    try:
        loc = get_resolved_checkpoint_location(checkpoint_name_or_path)
        with open(os.path.join(loc, FILE_SAFETENSOR_INDEX)) as f:
            index = json.load(f)

        # e.g., prefix: 'model.layers.0',
        #       module_name: 'block_sparse_moe'
        for prefix, (module_name, relevant_keys, has_bias) in tqdm(
            found.items(), disable=(rank > 0), desc='Sharding MoE'
        ):
            checkpoint_metadata = get_checkpoint_meta_from_sharded_safetensor(
                index['weight_map'], prefix, module_name,
                router_name, expert_name
            )

            _args = copy(mp_dmoe_args)
            _args.bias = has_bias

            # - will replace the MoE module with the megablocks sharded dMoE
            with init_empty_weights():
                mp_dmoe = dmoe.dMoE(_args) # drop in replacement for now

            load_sharded_experts_onto_device(
                mp_dmoe, loc, checkpoint_metadata, 
                device_mesh, placements, expert_name
            )
            parent = model.get_submodule(prefix)
            setattr(parent, module_name, mp_dmoe)

            # - keep track of the name for returning
            moe_module_names.add(module_name)

    except ValueError as e:
        raise ValueError(
            f"Unable to load checkpoint_path '{checkpoint_name_or_path}'. "
            "Currently only support non-GGUF safetensor checkpoints. "
            f": {e}"
        )


    return device_mesh[key_dp], moe_module_names