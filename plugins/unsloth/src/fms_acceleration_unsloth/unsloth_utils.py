from types import MethodType
from transformers.utils.import_utils import _is_package_available

from typing import List, Set, Tuple, Any

from peft.peft_model import PeftModelForCausalLM
import torch

has_xformers = _is_package_available("xformers")

# simple utility function to guess if its lora layer
def _is_loralayer(
    module: torch.nn.Module, names = ['lora_A', 'lora_B', 'base_layer']
):
    return all([hasattr(module, x) for x in names])

# simple function to check if architectures in the list
def _find_arch(architectures: List[str], artifacts: List[Tuple[Set, Any]]):
    for keys, _artifacts in artifacts:
        if any([arch in keys for arch in architectures]):
            return _artifacts

    return None

from .models.llama import original_apply_o, original_apply_qkv

# unsloth llama imports
from .models.llama import (
    LlamaModel_fast_forward,
    LlamaDecoderLayer_fast_forward, 
    LlamaForCausalLM_fast_forward,
    LlamaModel_fast_forward,
    LlamaAttention_fast_forward,
)

# unsloth mistral imports
from .models.mistral import (
    MistralAttention_fast_forward, MistralForCausalLM_fast_forward
)

# unsloth mixtral imports
from .models.mixtral import (
    MistralAttention_fast_forward, 
    MixtralForCausalLM_fast_forward,
    MixtralModel_fast_forward,
    MixtralDecoderLayer_fast_forward,
)

TAG_GPTQ = 'auto_gptq'
TAG_BNB = 'bitsandbytes'

TAG_QKV = 'qkv'
TAG_O = 'o'
TAG_MLP = 'mlp'

# unsloth gptq fast_lora imports
from .gptq.fast_lora import (
    apply_lora_qkv as _qkv_gptq, 
    apply_lora_o as _o_gptq, 
    apply_lora_mlp as _mlp_gptq
)
from .kernels.fast_lora import (
    apply_lora_qkv as _qkv_bnb, 
    apply_lora_o as _o_bnb, 
    apply_lora_mlp_swiglu as _mlp_bnb
)

# kernels
UNSLOTH_FAST_LORA = {
    TAG_GPTQ: {
        TAG_QKV: _qkv_gptq, TAG_O: _o_gptq, TAG_MLP: _mlp_gptq,
    },
    TAG_BNB: {
        TAG_QKV: _qkv_bnb, TAG_O: _o_bnb, TAG_MLP: _mlp_bnb,
    }
}

# fast forwards
# - key: Set of architectures
# - values:
#     * causallm_fastforward
#     * (backbone_name, backbone_fastforward)
#     * decoder_fastfoward
#     * (attn, attn_fastforward)
#     * (q_name, k_name, v_name), o_name
#     * mlp_name

UNSLOTH_FAST_FORWARDS = [
    (
        {'MistralForCausalLM'}, 
        (
            MistralForCausalLM_fast_forward, 
            ('model', LlamaModel_fast_forward), 
            LlamaDecoderLayer_fast_forward,
            ('self_attn', MistralAttention_fast_forward),
            (('q_proj', 'k_proj', 'v_proj'), 'o_proj'),
            'mlp', 
        )
    ),
    (
        {'MixtralForCausalLM'}, 
        (
            MixtralForCausalLM_fast_forward, 
            ('model', MixtralModel_fast_forward), 
            MixtralDecoderLayer_fast_forward,
            ('self_attn', MistralAttention_fast_forward),
            (('q_proj', 'k_proj', 'v_proj'), 'o_proj'),
            None, # for mixtral we do not apply fused ops on the mlp
        )
    ),
    (
        {'LlamaForCausalLM'}, 
        (
            LlamaForCausalLM_fast_forward, 
            ('model', LlamaModel_fast_forward), 
            LlamaDecoderLayer_fast_forward,
            ('self_attn', LlamaAttention_fast_forward),
            (('q_proj', 'k_proj', 'v_proj'), 'o_proj'),
            'mlp',
        )
    ),        
]

# add improvements to a PeftModelForCausalLM
# - fused ops
# - rms layer norm
# - RoPE embeddings
# - causallm cross-entropy loss
def add_unsloth_improvements(
    model: PeftModelForCausalLM, 
    adapter_name: str = 'default',
    stacked_over: str = 'auto_gptq',
):

    # some checks
    _is_lora_peft = (
        hasattr(model, "peft_config") and 
        model.peft_config[adapter_name].peft_type.value == 'LORA'
    )

    base_model = model.get_base_model()

    # config
    config = base_model.config
    base_model.max_seq_length = config.max_position_embeddings # the forward needs it

    # fetch artifacts
    artifacts = _find_arch(config.architectures, UNSLOTH_FAST_FORWARDS)
    if artifacts is None:
        raise ValueError(f"No unsloth improvements for any architectures in \'{config.architectures}\'")

    if not hasattr(base_model, '_no_split_modules') or base_model._no_split_modules is None:
        raise ValueError(
            "Only can install unsloth improvements in PreTrainedModels with _no_split_modules"
        )

    # get fast lora layers
    _fast_loras = UNSLOTH_FAST_LORA[stacked_over]
    _lqkv = _fast_loras[TAG_QKV]
    _lo = _fast_loras[TAG_O]
    _lmlp = _fast_loras[TAG_MLP]
    
    (
        _causal_f, (_bb_name, _bb_f), _decoder_f, 
        (_attn_name, _attn_f), (_qkv_names, _o_name), _mlp_name,
    ) = artifacts
    _no_split_modules = base_model._no_split_modules

    # for layer in base_model._get_no_split_modules():
    for layer in base_model.modules():
        if layer.__class__.__name__ not in _no_split_modules:
            continue

        self_attn = getattr(layer, _attn_name)
        mlp = None if _mlp_name is None else getattr(layer, _mlp_name)
        self_attn.forward = MethodType(_attn_f, self_attn)
        # NOTE: these are not set using MethodType as they called called as
        # func(self, X) in the fast forwards. To be changed later
        self_attn.apply_qkv = original_apply_qkv
        self_attn.apply_o = original_apply_o
        if _is_lora_peft:
            if all([_is_loralayer(getattr(self_attn, x)) for x in _qkv_names]):
                self_attn.apply_qkv = _lqkv
            if _is_loralayer(getattr(self_attn, _o_name)):
                self_attn.apply_o = _lo
            if mlp is not None and _is_loralayer(mlp):
                mlp.forward = MethodType(_lmlp, mlp) 
        layer.forward = MethodType(_decoder_f, layer)

    backbone = getattr(base_model, _bb_name)
    backbone.forward = MethodType(_bb_f, backbone)
    base_model.forward = MethodType(_causal_f, base_model)
    return model
