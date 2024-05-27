from transformers.models.llama.modeling_llama import LlamaAttention
from transformers.models.llama.modeling_llama import LlamaRMSNorm

from .model_patcher import ModelPatcher, ModelPatcherRule, ModelPatcherTrigger
from ..kernels.unsloth.rms_layernorm import fast_rms_layernorm
from ..kernels.unsloth.rope_embedding import fast_rope_embedding
from ..kernels.unsloth.cross_entropy_loss import FastCrossEntropyLoss

from .utils import trigger_fused_ops, build_lora_fused_ops
from functools import partial

ModelPatcher.register(
    ModelPatcherRule(
        rule_id='llama-rms', 
        trigger=ModelPatcherTrigger(check=LlamaRMSNorm),
        forward=fast_rms_layernorm
    ),
)

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
ModelPatcher.register(
    ModelPatcherRule(
        rule_id='llama-cross-ent',
        import_and_maybe_reload=(
            'torch.nn.CrossEntropyLoss', FastCrossEntropyLoss,
            'transformers.models.llama.modeling_llama'
        )
    )
)

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