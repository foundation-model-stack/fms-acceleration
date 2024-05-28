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

import torch
from typing import Callable, List, Type

# GPTQ imports
from ..fused_ops.unsloth_lora.gptq.fast_lora import (
    apply_lora_qkv as fused_op_qkv_gptq
)
from ..fused_ops.unsloth_lora.gptq.fast_lora import (
    get_lora_parameters as get_lora_parameters_gptq, 
    unpack_gptqstate, 
    LoRA_W as LoRA_W_gptq
)

from .model_patcher import ModelPatcherTrigger

# simple utility function to guess if its lora layer
def _is_loralayer(
    module: torch.nn.Module, names = ['lora_A', 'lora_B', 'base_layer']
):
    return all([hasattr(module, x) for x in names])

# builds a triple of forward functions, that each can be attached
# on a series of QKV's, where if the first one is called, will call the 
# fused op
# NOTE: this is not thread-safe (issue warning?)
# NOTE: the unsloth fused_operation "apply_lora_qkv" assumes that the 
#       modules are called q_proj, k_proj, and v_proj, respectively.
# the fused operation can be changed, depending on what the base layer is 
# i.e. gptq or bnb
def _build_qkv_forwards(
    attn: torch.nn.Module,
    fused_operation: Callable = fused_op_qkv_gptq,
    module_names: List[str] = ['q_proj', 'k_proj', 'v_proj'],
):

    Q = K = V = None

    # the fused operation will be called on first one that passes in the 
    # input X. 
    # - populates the triple Q, K, V
    # - subsequent calls will be a no-op until ALL Q, K, V get reset to None
    def _fused_op(X):
        nonlocal Q, K, V
        if Q is None and K is None and V is None:
            Q, K, V = fused_operation(attn, X)

    # each of these functions 
    # - calls the fused op
    # - 
    error_msg = "QKV fused_op needs to be first reset with sequential calls to each of them"
    def _forward_q(self, X):
        nonlocal Q
        _fused_op(X)
        assert Q is not None, error_msg
        out, Q = Q, None # unload
        return out

    def _forward_k(self, X):
        nonlocal K
        _fused_op(X)
        assert K is not None, error_msg
        out, K = K, None # unload
        return out

    def _forward_v(self, X):
        nonlocal V
        _fused_op(X)
        assert V is not None, error_msg
        out, V = V, None # unload
        return out

    return zip(
        module_names,
        [_forward_q, _forward_k, _forward_v]
    )
    
# fused ops for outputs for GPTQ
def fused_op_o_gptq(self, X):
    Oqstate, OA, OB, OS = get_lora_parameters_gptq(self)
    O = LoRA_W_gptq.apply(X, *unpack_gptqstate(Oqstate), OA, OB, OS)
    return O

# TODO: add the MLP
def build_lora_fused_ops(
    attn: torch.nn.Module, 
    base_type: str = 'auto_gptq', 
    qkv_module_names: List[str] = ['q_proj', 'k_proj', 'v_proj'],
    o_module_name: str = 'o_proj',
):

    # handle the QKVs
    if base_type == 'auto_gptq':
        _qkv_fused_op = fused_op_qkv_gptq
        _o_fused_op = fused_op_o_gptq
    else:
        raise NotImplementedError(
            f"Cannot build fused ops for base type '{base_type}'."
        )

    trigger_and_forwards = [
        (
            ModelPatcherTrigger(check=_is_loralayer, module_name=name), 
            forward
        )
        for name, forward in _build_qkv_forwards(
            attn, fused_operation=_qkv_fused_op,
            module_names=qkv_module_names,
        )
    ] 
    
    # handle the self-attn output
    _output_module = getattr(attn, o_module_name)
    if _is_loralayer(_output_module):
        trigger_and_forwards.append(
            (
                ModelPatcherTrigger(
                    check=_is_loralayer, module_name=o_module_name
                ), _o_fused_op
            )
        )

    # return
    return trigger_and_forwards

# trigger if either of the conditions are met
# 1. qkv all have LoRA adapters for a fused op
# 2. o has a lora adapter for the fused op
def trigger_fused_ops(
    module: torch.nn.Module, attn_cls: Type,
    qkv_module_names: List[str] = ['q_proj', 'k_proj', 'v_proj'],
    o_module_name: str = 'o_proj',
):
    _o = getattr(module, o_module_name)
    _qkv = [getattr(module, x) for x in qkv_module_names]

    # trigger on the attention layer
    return (
        isinstance(module, attn_cls) and
        (
            all([_is_loralayer(x) for x in _qkv])
            or _is_loralayer(_o)
        )
    )


