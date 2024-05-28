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
from transformers.models.mistral.modeling_mistral import (
    MistralAttention,
    MistralRMSNorm,
)

# Local
from ..kernels.unsloth.cross_entropy_loss import FastCrossEntropyLoss
from ..kernels.unsloth.rms_layernorm import fast_rms_layernorm
from ..kernels.unsloth.rope_embedding import fast_rope_embedding as _fast_rope_embedding
from .model_patcher import ModelPatcher, ModelPatcherRule, ModelPatcherTrigger
from .utils import build_lora_fused_ops, trigger_fused_ops


# NOTE: fast_rope_embedding does not work with position_ids
# currently they are ignored
def fast_rope_embedding(Q, K, cos, sin, position_ids=None):
    return _fast_rope_embedding(Q, K, cos, sin)


# - do regex on RMSNorm class name
# - check on the tensors required for fast_rms_layernorm
ModelPatcher.register(
    ModelPatcherRule(
        rule_id="mistral-rms",
        trigger=ModelPatcherTrigger(check=MistralRMSNorm),
        forward=fast_rms_layernorm,
    ),
)

# - do regex on Attention class name
# - have a set of qkv / o module names and check on that
ModelPatcher.register(
    ModelPatcherRule(
        rule_id="mistral-qkvo",
        trigger=ModelPatcherTrigger(
            check=partial(
                trigger_fused_ops,
                attn_cls=MistralAttention,
                qkv_module_names=["q_proj", "k_proj", "v_proj"],
                o_module_name="o_proj",
            )
        ),
        forward_builder=partial(
            build_lora_fused_ops,
            qkv_module_names=["q_proj", "k_proj", "v_proj"],
            o_module_name="o_proj",
        ),
        forward_builder_args=["base_type"],
    )
)

# - get the module_name and reload on that
ModelPatcher.register(
    ModelPatcherRule(
        rule_id="mistral-cross-ent",
        import_and_maybe_reload=(
            "torch.nn.CrossEntropyLoss",
            FastCrossEntropyLoss,
            "transformers.models.mistral.modeling_mistral",
        ),
    )
)

# - get the module name
# - check if "apply_rotary_pos_emb" exists
# - patch
ModelPatcher.register(
    ModelPatcherRule(
        rule_id="mistral-rope",
        import_and_maybe_reload=(
            "transformers.models.mistral.modeling_mistral.apply_rotary_pos_emb",
            fast_rope_embedding,
            None,
        ),
    )
)
