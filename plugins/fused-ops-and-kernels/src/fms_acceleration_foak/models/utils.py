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
from typing import Callable, List, Type

# Third Party
import torch
import os

# Local
# GPTQ imports
from ..fused_ops.unsloth_lora.gptq.fast_lora import (
    apply_lora_qkv as fused_op_qkv_gptq,
    apply_lora_o as fused_op_o_gptq,
    apply_lora_mlp as fused_op_mlp_gptq,
)
from .model_patcher import ModelPatcherTrigger


# simple utility function to guess if its lora layer
def _is_loralayer(module: torch.nn.Module, names: List[str] = None):
    if names is None:
        names = ["lora_A", "lora_B", "base_layer"]
    return all(hasattr(module, x) for x in names)


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
    module_names: List[str] = None,
):
    if module_names is None:
        module_names = ["q_proj", "k_proj", "v_proj"]

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
    error_msg = (
        "QKV fused_op needs to be first reset with sequential calls to each of them"
    )

    def _forward_q(self, X):
        nonlocal Q
        _fused_op(X)
        assert Q is not None, error_msg
        out, Q = Q, None  # unload
        return out

    def _forward_k(self, X):
        nonlocal K
        _fused_op(X)
        assert K is not None, error_msg
        out, K = K, None  # unload
        return out

    def _forward_v(self, X):
        nonlocal V
        _fused_op(X)
        assert V is not None, error_msg
        out, V = V, None  # unload
        return out

    return zip(module_names, [_forward_q, _forward_k, _forward_v])

# TODO: add the MLP
def build_lora_fused_ops(
    attn: torch.nn.Module,
    base_type: str = "auto_gptq",
    qkv_module_names: List[str] = None,
    o_module_name: str = "o_proj",
):
    if qkv_module_names is None:
        qkv_module_names = ["q_proj", "k_proj", "v_proj"]

    # handle the QKVs
    if base_type == "auto_gptq":
        _qkv_fused_op = fused_op_qkv_gptq
        _o_fused_op = fused_op_o_gptq

        # this is required due to this FSDP fix
        # https://github.com/foundation-model-stack/fms-acceleration/pull/15
        try:
            world_size = torch.distributed.get_world_size()
        except ValueError:
            world_size = 1  # pg not init

        if (
            world_size > 1
            and os.environ.get("ACCELERATE_USE_FSDP", "false").lower() == "true"
        ):

            # guarded import
            from fms_acceleration_peft.autogptq_utils import ( #pylint: disable=import-outside-toplevel
                patch_forward_to_view_attributes_before_call, 
                PATCH_FOR_FSDP_TRITON_V2
            )

            # patch each of the fused ops to view the attributes
            # back into torch.int32
            # TODO: add the MLP fused op also
            _qkv_fused_op = patch_forward_to_view_attributes_before_call(
                _qkv_fused_op, 
                PATCH_FOR_FSDP_TRITON_V2, torch.int32,
                submodule_names=[
                    n + '.base_layer' for n in qkv_module_names
                ],
                is_method_forward=False,
            )
            _o_fused_op = patch_forward_to_view_attributes_before_call(
                _o_fused_op,
                PATCH_FOR_FSDP_TRITON_V2, torch.int32,
                submodule_names='base_layer',
                is_method_forward=False,
            )

    else:
        raise NotImplementedError(
            f"Cannot build fused ops for base type '{base_type}'."
        )

    trigger_and_forwards = [
        (ModelPatcherTrigger(check=_is_loralayer, module_name=name), forward)
        for name, forward in _build_qkv_forwards(
            attn,
            fused_operation=_qkv_fused_op,
            module_names=qkv_module_names,
        )
    ]

    # handle the self-attn output
    _output_module = getattr(attn, o_module_name)
    if _is_loralayer(_output_module):
        trigger_and_forwards.append(
            (
                ModelPatcherTrigger(check=_is_loralayer, module_name=o_module_name),
                _o_fused_op,
            )
        )

    # return
    return trigger_and_forwards


# trigger if either of the conditions are met
# 1. qkv all have LoRA adapters for a fused op
# 2. o has a lora adapter for the fused op
def trigger_fused_ops(
    module: torch.nn.Module,
    attn_cls: Type,
    qkv_module_names: List[str] = None,
    o_module_name: str = "o_proj",
):
    if qkv_module_names is None:
        qkv_module_names = ["q_proj", "k_proj", "v_proj"]

    _o = getattr(module, o_module_name)
    _qkv = [getattr(module, x) for x in qkv_module_names]

    # trigger on the attention layer
    return isinstance(module, attn_cls) and (
        all(_is_loralayer(x) for x in _qkv) or _is_loralayer(_o)
    )
