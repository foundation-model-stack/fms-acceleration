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
from fms_acceleration.model_patcher import (
    ModelPatcherRule,
    ModelPatcherTrigger,
    combine_functions,
    combine_triggers,
)
from transformers.models.llama.modeling_llama import (
    LlamaAttention,
    LlamaMLP,
    LlamaRMSNorm,
)

# Local
from ..kernels.unsloth.cross_entropy_loss import FastCrossEntropyLoss
from ..kernels.unsloth.rms_layernorm import fast_rms_layernorm
from ..kernels.unsloth.rope_embedding import fast_rope_embedding
from .utils import KEY_MLP, KEY_O, KEY_QKV, build_lora_fused_ops, trigger_fused_ops


def get_mp_rules(base_type: str, use_fused_linear_cross_entropy:bool=True):
    """
    Function to access all patch rules in this module.
    If it is a forward_builder rule with `base_type` in
    its forward builder argument, wrap the forward_builder
    function as a partial function with the base_type argument
    """

    def is_lm_head(module):
        return module.__name__ == "lm_head"

    import torch
    from ..kernels.liger.fused_linear_cross_entropy import LigerFusedLinearCrossEntropyFunction
    def build_new_lm_head_forward(lm_head: torch.nn.Module):        
        def lm_head_forward(self, hidden_states):
            # TODO keep the hidden_state somewhere for FusedLCE
            return torch.rand((-1, self.weight.size(0))) 
        return lm_head_forward

    custom_rules = []
    if use_fused_linear_cross_entropy:
        custom_rules.extend([
            ModelPatcherRule(
                rule_id="llama-fused-linear-cross-ent",
                trigger=is_lm_head,
                forward_builder=build_new_lm_head_forward,
            ),
            ModelPatcherRule(
                rule_id="llama-fused-linear-cross-ent",
                import_and_maybe_reload=(
                    "torch.nn.CrossEntropyLoss",
                    LigerFusedLinearCrossEntropyFunction,
                    "transformers.models.llama.modeling_llama",
                ),
            ),
        ])
    else:
        custom_rules.extend([
            ModelPatcherRule(
                rule_id="llama-cross-ent",
                import_and_maybe_reload=(
                    "torch.nn.CrossEntropyLoss",
                    FastCrossEntropyLoss,
                    "transformers.models.llama.modeling_llama",
                ),
            ),
        ])

    return [
        *custom_rules,
        # TODO: have a generic version of this rule
        # - do regex on RMSNorm class name
        # - check on the tensors required for fast_rms_layernorm
        ModelPatcherRule(
            rule_id="llama-rms",
            trigger=ModelPatcherTrigger(check=LlamaRMSNorm),
            forward=fast_rms_layernorm,
        ),
        # TODO: have a generic version of this rule
        # - do regex on Attention class name
        # - have a set of qkv / o module names and check on that
        ModelPatcherRule(
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
                    base_type=base_type,
                ),
                partial(
                    build_lora_fused_ops,
                    submodule_names=["o_proj"],
                    fused_op=KEY_O,
                    base_type=base_type,
                ),
                logic="APPEND",
            ),
        ),
        ModelPatcherRule(
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
                base_type=base_type,
            ),
        ),
        # TODO: have a generic version of this rule
        # - get the module_name and reload on that
        # ModelPatcherRule(
        #     rule_id="llama-cross-ent",
        #     import_and_maybe_reload=(
        #         "torch.nn.CrossEntropyLoss",
        #         FastCrossEntropyLoss,
        #         "transformers.models.llama.modeling_llama",
        #     ),
        # ),
        # TODO: have a generic version of this rule
        # - get the module name
        # - check if "apply_rotary_pos_emb" exists
        # - patch
        ModelPatcherRule(
            rule_id="llama-rope",
            import_and_maybe_reload=(
                "transformers.models.llama.modeling_llama.apply_rotary_pos_emb",
                fast_rope_embedding,
                None,
            ),
        ),
    ]
