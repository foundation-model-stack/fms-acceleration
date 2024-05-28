from transformers.models.mistral.modeling_mistral import MistralAttention
from transformers.models.mistral.modeling_mistral import MistralRMSNorm

from .model_patcher import ModelPatcher, ModelPatcherRule, ModelPatcherTrigger
from ..kernels.unsloth.rms_layernorm import fast_rms_layernorm 
from ..kernels.unsloth.rope_embedding import fast_rope_embedding as _fast_rope_embedding
from ..kernels.unsloth.cross_entropy_loss import FastCrossEntropyLoss

from .utils import trigger_fused_ops, build_lora_fused_ops
from functools import partial

# NOTE: fast_rope_embedding does not work with position_ids
# currently they are ignored
def fast_rope_embedding(Q, K, cos, sin, position_ids=None):
    return _fast_rope_embedding(Q, K, cos, sin)

# TODO: have a generic version of this rule
# - do regex on RMSNorm class name
# - check on the tensors required for fast_rms_layernorm
ModelPatcher.register(
    ModelPatcherRule(
        rule_id='mistral-rms', 
        trigger=ModelPatcherTrigger(check=MistralRMSNorm),
        forward=fast_rms_layernorm
    ),
)

# TODO: have a generic version of this rule
# - do regex on Attention class name
# - have a set of qkv / o module names and check on that
ModelPatcher.register(
    ModelPatcherRule(
        rule_id='mistral-qkvo', 
        trigger=ModelPatcherTrigger(
            check=partial(
                trigger_fused_ops, attn_cls=MistralAttention,
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
        rule_id='mistral-cross-ent',
        import_and_maybe_reload=(
            'torch.nn.CrossEntropyLoss', FastCrossEntropyLoss,
            'transformers.models.mistral.modeling_mistral'
        )
    )
)

# TODO: have a generic version of this rule
# - get the module name
# - check if "apply_rotary_pos_emb" exists
# - patch
ModelPatcher.register(
    ModelPatcherRule(
        rule_id='mistral-rope',
        import_and_maybe_reload=(
            'transformers.models.mistral.modeling_mistral.apply_rotary_pos_emb',
            fast_rope_embedding,
            None
        )
    )
)