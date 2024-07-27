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
from functools import partial

# Third Party
from transformers.models.llama.modeling_llama import (
    LlamaAttention,
    LlamaMLP,
    LlamaRMSNorm,
)
from fms_acceleration.model_patcher import (
    ModelPatcher,
    ModelPatcherRule,
    ModelPatcherTrigger,
    combine_functions,
    combine_triggers,
)

# Local
from ..kernels.unsloth.cross_entropy_loss import FastCrossEntropyLoss
from ..kernels.unsloth.rms_layernorm import fast_rms_layernorm
from ..kernels.unsloth.rope_embedding import fast_rope_embedding
from .utils import KEY_MLP, KEY_O, KEY_QKV, build_lora_fused_ops, trigger_fused_ops

# TODO: have a generic version of this rule
# - do regex on RMSNorm class name
# - check on the tensors required for fast_rms_layernorm
RULE_LLAMA_RMS = ModelPatcherRule(
    rule_id="llama-rms",
    trigger=ModelPatcherTrigger(check=LlamaRMSNorm),
    forward=fast_rms_layernorm,
)

# TODO: have a generic version of this rule
# - do regex on Attention class name
# - have a set of qkv / o module names and check on that
RULE_LLAMA_QKVO = ModelPatcherRule(
    rule_id="llama-qkvo",
    trigger=combine_triggers(
        ModelPatcherTrigger(
            check=partial(
                trigger_fused_ops,
                attn_cls=LlamaAttention,
                submodule_names=["q_proj", "k_proj", "v_proj"],
            )
        ),
        ModelPatcherTrigger(
            check=partial(
                trigger_fused_ops,
                attn_cls=LlamaAttention,
                submodule_names=["o_proj"],
            )
        ),
        logic="OR",
    ),
    forward_builder=combine_functions(
        partial(
            build_lora_fused_ops,
            submodule_names=["q_proj", "k_proj", "v_proj"],
            fused_op=KEY_QKV,
        ),
        partial(
            build_lora_fused_ops,
            submodule_names=["o_proj"],
            fused_op=KEY_O,
        ),
        logic="APPEND",
    ),
    forward_builder_args=["base_type"],
)

RULE_LLAMA_MLP = ModelPatcherRule(
    rule_id="llama-mlp",
    trigger=ModelPatcherTrigger(
        check=partial(
            trigger_fused_ops,
            attn_cls=LlamaMLP,
            submodule_names=["up_proj", "down_proj", "gate_proj"],
        )
    ),
    forward_builder=partial(
        build_lora_fused_ops,
        submodule_names=["up_proj", "down_proj", "gate_proj"],
        fused_op=KEY_MLP,
    ),
    forward_builder_args=["base_type"],
)

# TODO: have a generic version of this rule
# - get the module_name and reload on that
RULE_LLAMA_CE = ModelPatcherRule(
    rule_id="llama-cross-ent",
    import_and_maybe_reload=(
        "torch.nn.CrossEntropyLoss",
        FastCrossEntropyLoss,
        "transformers.models.llama.modeling_llama",
    ),
)

# TODO: have a generic version of this rule
# - get the module name
# - check if "apply_rotary_pos_emb" exists
# - patch
RULE_LLAMA_ROPE = ModelPatcherRule(
    rule_id="llama-rope",
    import_and_maybe_reload=(
        "transformers.models.llama.modeling_llama.apply_rotary_pos_emb",
        fast_rope_embedding,
        None,
    ),
)

LLAMA_MP_RULES = [
    RULE_LLAMA_RMS,
    RULE_LLAMA_QKVO,
    RULE_LLAMA_MLP,
    RULE_LLAMA_CE,
    RULE_LLAMA_ROPE,
]

def get_mp_rules(base_type):
    """
    Function to access all patch rules in this module.
    If it is a forward_builder rule with `base_type` in
    its forward builder argument, wrap the forward_builder
    function as a partial function with the base_type argument
    """
    for rule in LLAMA_MP_RULES:
        if (
            rule.forward_builder is not None
            and "base_type" in  rule.forward_builder_args
        ):
            rule.forward_builder = partial(
                rule.forward_builder,
                base_type=base_type,
            )
    return LLAMA_MP_RULES
