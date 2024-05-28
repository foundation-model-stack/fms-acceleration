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

from transformers.models.llama.modeling_llama import LlamaAttention
from transformers.models.llama.modeling_llama import LlamaRMSNorm

from .model_patcher import ModelPatcher, ModelPatcherRule, ModelPatcherTrigger
from ..kernels.unsloth.rms_layernorm import fast_rms_layernorm
from ..kernels.unsloth.rope_embedding import fast_rope_embedding
from ..kernels.unsloth.cross_entropy_loss import FastCrossEntropyLoss

from .utils import trigger_fused_ops, build_lora_fused_ops
from functools import partial

# TODO: have a generic version of this rule
# - do regex on RMSNorm class name
# - check on the tensors required for fast_rms_layernorm
ModelPatcher.register(
    ModelPatcherRule(
        rule_id='llama-rms', 
        trigger=ModelPatcherTrigger(check=LlamaRMSNorm),
        forward=fast_rms_layernorm
    ),
)

# TODO: have a generic version of this rule
# - do regex on Attention class name
# - have a set of qkv / o module names and check on that
ModelPatcher.register(
    ModelPatcherRule(
        rule_id='llama-qkvo', 
        trigger=ModelPatcherTrigger(
            check=partial(
                trigger_fused_ops, attn_cls=LlamaAttention,
                qkv_module_names=['q_proj', 'k_proj', 'v_proj'],
                o_module_name='o_proj',
            )
        ),
        forward_builder=partial(
            build_lora_fused_ops, 
            qkv_module_names=['q_proj', 'k_proj', 'v_proj'],
            o_module_name='o_proj',
        ),
        forward_builder_args=['base_type'],
    )
)

# TODO: have a generic version of this rule
# - get the module_name and reload on that
ModelPatcher.register(
    ModelPatcherRule(
        rule_id='llama-cross-ent',
        import_and_maybe_reload=(
            'torch.nn.CrossEntropyLoss', FastCrossEntropyLoss,
            'transformers.models.llama.modeling_llama'
        )
    )
)

# TODO: have a generic version of this rule
# - get the module name
# - check if "apply_rotary_pos_emb" exists
# - patch
ModelPatcher.register(
    ModelPatcherRule(
        rule_id='llama-rope',
        import_and_maybe_reload=(
            'transformers.models.llama.modeling_llama.apply_rotary_pos_emb',
            fast_rope_embedding,
            None
        )
    )
)