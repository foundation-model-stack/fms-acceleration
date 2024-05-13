# Copyright 2023-present Daniel Han-Chen & the Unsloth team. All rights reserved.
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
from typing import Optional, Tuple, List, Union
from torch.nn.functional import scaled_dot_product_attention
from transformers.models.llama.modeling_llama import (
    logger,
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)
from transformers.utils.import_utils import _is_package_available
HAS_FLASH_ATTENTION = _is_package_available("flash_attn")
from ..kernels import *
if HAS_FLASH_ATTENTION:
    from flash_attn import flash_attn_func

import xformers.ops.fmha as xformers
xformers_attention = xformers.memory_efficient_attention
# Final patching code
from transformers.models.llama.modeling_llama import LlamaAttention

# For Pytorch 2.1.1
try:
    from transformers.models.llama.modeling_llama import (
        LlamaSdpaAttention,
        LlamaFlashAttention2,
    )
except:
    LlamaSdpaAttention   = LlamaAttention
    LlamaFlashAttention2 = LlamaAttention

def original_apply_qkv(self, X):
    Q = self.q_proj(X)
    K = self.k_proj(X)
    V = self.v_proj(X)
    return Q, K, V



def original_apply_o(self, X):
    O = self.o_proj(X)
    return O


from math import sqrt as math_sqrt
def LlamaAttention_fast_forward_inference(
    self,
    hidden_states:  torch.Tensor,
    past_key_value: Optional[Tuple[torch.Tensor]],
    position_ids,
):
    """
        https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py#L406
        Fast inference using KV cache.
        QK^T can be computed in 4 chunks

        [Q, q] @ [K, k].T where q, k are the new tokens.
        [QK^T, Qk^T]
        [qK^T, qk^T]

        Since the attention mask wipes Qk^T, we just get
        [QK^T,    0]
        [qK^T, qk^T]

        Since softmax is row-wise, we get
        softmax([QK^T,    0])
        softmax([qK^T, qk^T])

        We then multiply by   [V]
                              [v]
        softmax([QK^T,    0]) [softmax(QK^T)V] *
        softmax([qK^T, qk^T]) [softmax([qK^T, qk^T]) @ [V, v]]

        But notice * [softmax(QK^T)V] is just the last attention.
        We just need to compute the last final row.

        This means we can pass in a row of Q, but we need to
        remember K and V, which are called the KV cache.
    """
    n_heads    = self.num_heads
    n_groups   = self.num_key_value_groups
    n_kv_heads = self.num_key_value_heads
    head_dim   = self.head_dim
    # assert(n_kv_heads * n_groups == n_heads)

    Xn = hidden_states.view(self.hidden_size)
    K1, V1 = past_key_value
    seq_len = K1.shape[-2]
    K1 = K1.view(n_kv_heads, seq_len, head_dim)
    V1 = V1.view(n_kv_heads, seq_len, head_dim)

    # LoRA or general matrix multiplication
    dtype = Xn.dtype
    Qn = fast_linear_forward(self.q_proj, Xn)
    Kn = fast_linear_forward(self.k_proj, Xn)
    Vn = fast_linear_forward(self.v_proj, Xn)

    Qn = Qn.view(n_heads,    1, head_dim)
    Kn = Kn.view(n_kv_heads, 1, head_dim)
    Vn = Vn.view(n_kv_heads, 1, head_dim)

    cos = self.rotary_emb.cos_cached[seq_len]
    sin = self.rotary_emb.sin_cached[seq_len]
    h = head_dim // 2

    RH_Q = torch.empty((n_heads, 1, head_dim), dtype = dtype, device = "cuda")
    RH_Q[:, :, :h] = Qn[:, :, h:]; RH_Q[:, :, h:] = Qn[:, :, :h]; torch.neg(RH_Q[:, :, :h], out = RH_Q[:, :, :h]);
    Qn *= cos; Qn.addcmul_(RH_Q, sin);

    RH_K = RH_Q[:n_kv_heads, :, :] # torch.empty((n_kv_heads, 1, head_dim), dtype = dtype, device = "cuda")
    RH_K[:, :, :h] = Kn[:, :, h:]; RH_K[:, :, h:] = Kn[:, :, :h]; torch.neg(RH_K[:, :, :h], out = RH_K[:, :, :h]);
    Kn *= cos; Kn.addcmul_(RH_K, sin);
    
    # New KV cache
    Kn = torch.cat([K1, Kn], dim = 1)
    Vn = torch.cat([V1, Vn], dim = 1)

    # Grouped query attention
    if n_groups != 1:
        new_seq_len = seq_len + 1
        Knn = Kn[:, None, :, :].expand(n_kv_heads, n_groups, new_seq_len, head_dim)
        Vnn = Vn[:, None, :, :].expand(n_kv_heads, n_groups, new_seq_len, head_dim)
        Knn = Knn.reshape(n_heads, new_seq_len, head_dim)
        Vnn = Vnn.reshape(n_heads, new_seq_len, head_dim)
    else:
        Knn, Vnn = Kn, Vn

    # Attention
    A = torch.matmul(Qn, Knn.transpose(1, 2))
    A *= 1.0 / math_sqrt(self.head_dim)
    A[:] = torch.nn.functional.softmax(A, dim = -1, dtype = torch.float32)#.to(A.dtype)
    A = torch.matmul(A, Vnn, out = Qn)
    A = A.view(self.hidden_size)

    A = fast_linear_forward(self.o_proj, A)
    A = A.reshape(1, 1, self.hidden_size)

    return A, (Kn.unsqueeze(0), Vn.unsqueeze(0))


torch_silu = torch.nn.functional.silu
def fast_mlp_inference(self, X):
    hidden_size = self.hidden_size
    X = X.view(hidden_size)

    gate = fast_linear_forward(self.gate_proj, X)
    up   = fast_linear_forward(self.  up_proj, X)
    gate = torch_silu(gate, inplace = True)
    gate *= up

    down = fast_linear_forward(self.down_proj, gate, out = up[:hidden_size])
    X = down.view(1, 1, hidden_size)

    return X

def fast_rms_layernorm_inference(self, X):
    old_dtype = X.dtype
    X = X.to(torch.float32)
    variance = X.square().mean(-1, keepdim = True)
    variance += self.variance_epsilon
    X *= variance.rsqrt_()
    X = X.to(old_dtype)
    X *= self.weight
    return X


# https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py#L320
def LlamaAttention_fast_forward(
    self,
    hidden_states: torch.Tensor,
    causal_mask: Optional[xformers.attn_bias.BlockDiagonalCausalMask] = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    padding_mask: Optional[torch.LongTensor] = None,
    *args,
    **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    
    # Clear inference
    if hasattr(self, "paged_attention"):
        del self.paged_attention_K
        del self.paged_attention_V
        del self.paged_attention
        del self.temp_QA
        del self.temp_KV
        del self.RH_Q
        del self.attention

    bsz, q_len, _ = hidden_states.size()

    n_heads    = self.num_heads
    n_groups   = self.num_key_value_groups
    n_kv_heads = self.num_key_value_heads
    head_dim = self.head_dim
    assert n_kv_heads * n_groups == n_heads

    Q, K, V = self.apply_qkv(self, hidden_states)
    Q = Q.view(bsz, q_len, n_heads, head_dim).transpose(1, 2)
    K = K.view(bsz, q_len, n_kv_heads, head_dim).transpose(1, 2)
    V = V.view(bsz, q_len, n_kv_heads, head_dim).transpose(1, 2)

    kv_seq_len = K.shape[-2]
    if past_key_value is not None:
        kv_seq_len += past_key_value[0].shape[-2]

    if position_ids is None:
        cos = self.rotary_emb.cos_cached
        sin = self.rotary_emb.sin_cached
        Q, K = fast_rope_embedding(Q, K, cos, sin)
    else:
        cos, sin = self.rotary_emb(V, seq_len=kv_seq_len)
        Q, K = inplace_rope_embedding(Q, K, cos, sin, position_ids)

    if past_key_value is not None:
        K = torch.cat([past_key_value[0], K], dim=2)
        V = torch.cat([past_key_value[1], V], dim=2)
    past_key_value = (K, V) if use_cache else None

    # Attention module
    if (not HAS_FLASH_ATTENTION and attention_mask is None):
        # Xformers memory efficient attention
        # Also has Flash Attention v2 dispatching
        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)

        # Group query attention
        if n_groups != 1:
            K = K.view(bsz, kv_seq_len, n_kv_heads, 1, head_dim)
            V = V.view(bsz, kv_seq_len, n_kv_heads, 1, head_dim)
            K = K.expand(bsz, kv_seq_len, n_kv_heads, n_groups, head_dim)
            V = V.expand(bsz, kv_seq_len, n_kv_heads, n_groups, head_dim)
            if hidden_states.requires_grad:
                K = K.reshape(bsz, kv_seq_len, n_heads, head_dim)
                V = V.reshape(bsz, kv_seq_len, n_heads, head_dim)
            else:
                Q = Q.view(bsz, q_len, n_kv_heads, n_groups, head_dim)
        A = xformers_attention(Q, K, V, attn_bias=causal_mask)
        A = A.view(bsz, q_len, n_heads, head_dim)

    elif HAS_FLASH_ATTENTION and attention_mask is None:
        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)
        A = flash_attn_func(Q, K, V, causal=True)
    else:
        # Grouped query attention
        if n_groups != 1:
            K = K[:, :, None, :, :].expand(
                bsz, n_kv_heads, n_groups, kv_seq_len, head_dim
            )
            V = V[:, :, None, :, :].expand(
                bsz, n_kv_heads, n_groups, kv_seq_len, head_dim
            )
            K = K.reshape(bsz, n_heads, kv_seq_len, head_dim)
            V = V.reshape(bsz, n_heads, kv_seq_len, head_dim)
        # Must be contiguous or else results are False!
        # https://github.com/pytorch/pytorch/issues/112577
        Q, K, V = Q.contiguous(), K.contiguous(), V.contiguous()
        # Needs (batch_size, n_heads, seq_len, head_dim)
        # is_casual and attention_mask must not be both set!
        A = scaled_dot_product_attention(
            Q, K, V, attn_mask=attention_mask, is_causal=False
        )
        # Go back to (batch_size, seq_len, n_heads, head_dim)
        A = A.transpose(1, 2).contiguous()

    attn_output = A.reshape(bsz, q_len, n_heads*head_dim)
    attn_output = self.apply_o(self, attn_output)
    attn_weights = None
    return attn_output, attn_weights, past_key_value


# https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py#L590
def LlamaDecoderLayer_fast_forward(
    self,
    hidden_states:        torch.Tensor,
    causal_mask:          Optional[xformers.attn_bias.BlockDiagonalCausalMask] = None,
    attention_mask:       Optional[torch.Tensor] = None,
    position_ids:         Optional[torch.LongTensor] = None,
    past_key_value:       Optional[Tuple[torch.Tensor]] = None,
    output_attentions:    Optional[bool] = False,
    use_cache:            Optional[bool] = False,
    padding_mask:         Optional[torch.LongTensor] = None,
    *args, **kwargs,
) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
    """
    Args:
        hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
        attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
            `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under
            returned tensors for more detail.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
            (see `past_key_values`).
        past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
    """
    residual = hidden_states
    hidden_states = fast_rms_layernorm(self.input_layernorm, hidden_states)
    hidden_states, self_attn_weights, present_key_value = self.self_attn(
        hidden_states=hidden_states,
        causal_mask=causal_mask,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_value=past_key_value,
        output_attentions=output_attentions,
        use_cache=use_cache,
        padding_mask=padding_mask,
    )
    hidden_states = residual + hidden_states

    # Fully Connected
    residual = hidden_states
    hidden_states = fast_rms_layernorm(self.post_attention_layernorm, hidden_states)
    hidden_states = self.mlp(hidden_states)
    hidden_states = residual + hidden_states

    outputs = (hidden_states,)

    if output_attentions:
        outputs += (self_attn_weights,)

    if use_cache:
        outputs += (present_key_value,)

    return outputs

# https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py#L825
def LlamaModel_fast_forward(
    self,
    input_ids:            torch.LongTensor,
    causal_mask:          Optional[xformers.attn_bias.BlockDiagonalCausalMask] = None,
    attention_mask:       Optional[torch.Tensor] = None,
    position_ids:         Optional[torch.LongTensor] = None,
    past_key_values:      Optional[List[torch.FloatTensor]] = None,
    inputs_embeds:        Optional[torch.FloatTensor] = None,
    use_cache:            Optional[bool] = None,
    output_attentions:    Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict:          Optional[bool] = None,
    *args, **kwargs,
) -> Union[Tuple, BaseModelOutputWithPast]:

    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    assert(output_attentions is False)
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    use_cache = use_cache if use_cache is not None else self.config.use_cache

    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    # retrieve input_ids and inputs_embeds
    if input_ids is not None and inputs_embeds is not None:
        raise ValueError("Unsloth: You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
    elif input_ids is not None:
        batch_size, seq_length = input_ids.shape
    elif inputs_embeds is not None:
        batch_size, seq_length, _ = inputs_embeds.shape
    else:
        raise ValueError("Unsloth: You have to specify either decoder_input_ids or decoder_inputs_embeds")

    seq_length_with_past = seq_length

    # Fix out of bounds tokenization
    if hasattr(self, "max_seq_length"):
        if seq_length > self.max_seq_length:
            logger.warning_once(
                f"Unsloth: Input IDs of length {seq_length} > the model's max sequence length of {self.max_seq_length}.\n"\
                "We shall truncate it ourselves. It's imperative if you correct this issue first."
            )
        if input_ids is not None:
            input_ids = input_ids[:,:self.max_seq_length]
        elif inputs_embeds is not None:
            inputs_embeds = inputs_embeds[:,:self.max_seq_length,:]
    
    past_key_values_length = 0

    if past_key_values is not None:
        past_key_values_length = past_key_values[0][0].shape[2]
        seq_length_with_past = seq_length_with_past + past_key_values_length

    # We already handle KV cache position_ids ourselves.
    if (past_key_values_length != 0):
        position_ids = torch.arange(
            past_key_values_length, seq_length + past_key_values_length,
            dtype  = torch.int32,
            device = "cuda",
        )
        position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
    elif position_ids is not None:
        position_ids = position_ids.view(-1, seq_length).to(torch.int32)#.long()
    else:
        position_ids = None

    if position_ids is not None:
        if position_ids.shape[0] != batch_size:
            position_ids = position_ids.repeat((batch_size, 1))

    # embed positions
    if inputs_embeds is None:
        inputs_embeds = self.embed_tokens(input_ids)

    # Ignore attention_mask
    if attention_mask is None:
        padding_mask = None
    elif self.training:
        attention_mask = None
        padding_mask = None
    else:
        if 0 in attention_mask:
            padding_mask = attention_mask
        else:
            padding_mask = None

        from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask
        attention_mask = _prepare_4d_causal_attention_mask(
            attention_mask,
            (batch_size, seq_length),
            inputs_embeds,
            past_key_values_length,
            sliding_window = getattr(self.config, "sliding_window", None),
        )

    hidden_states = inputs_embeds

    if self.gradient_checkpointing and self.training:
        if use_cache:
            logger.warning_once(
                "Unsloth: `use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`"
            )
            use_cache = False

    # decoder layers
    all_hidden_states = () if output_hidden_states else None
    all_self_attns = () if output_attentions else None
    next_decoder_cache = () if use_cache else None

    for idx, decoder_layer in enumerate(self.layers):
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        past_key_value = past_key_values[idx] if past_key_values is not None else None

        if self.gradient_checkpointing and self.training:

            def create_custom_forward(module):
                def custom_forward(*inputs):
                    # None for past_key_value
                    return module(*inputs, past_key_value, output_attentions, padding_mask=padding_mask)

                return custom_forward

            layer_outputs = torch.utils.checkpoint.checkpoint(
                create_custom_forward(decoder_layer),
                hidden_states,
                causal_mask,
                attention_mask,
                position_ids,
                use_reentrant=True,
                preserve_rng_state=False,
            )
        else:
            layer_outputs = decoder_layer(
                hidden_states,
                causal_mask=causal_mask,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                padding_mask=padding_mask,
            )

        hidden_states = layer_outputs[0]

        if use_cache:
            next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)

        if output_attentions:
            all_self_attns += (layer_outputs[1],)

    bsz, q_len, hd = hidden_states.size()
    if (past_key_value is not None and q_len == 1):
        hidden_states = fast_rms_layernorm_inference(self.norm, hidden_states)
    else:
        hidden_states = fast_rms_layernorm(self.norm, hidden_states)

    # add hidden states from the last decoder layer
    if output_hidden_states:
        all_hidden_states += (hidden_states,)

    next_cache = next_decoder_cache if use_cache else None
    if not return_dict:
        return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
    return BaseModelOutputWithPast(
        last_hidden_state=hidden_states,
        past_key_values=next_cache,
        hidden_states=all_hidden_states,
        attentions=all_self_attns,
    )

def LlamaForCausalLM_fast_forward(
    self,
    input_ids: torch.LongTensor = None,
    causal_mask: Optional[xformers.attn_bias.BlockDiagonalCausalMask] = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[List[torch.FloatTensor]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    labels: Optional[torch.LongTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    *args, **kwargs,
) -> Union[Tuple, CausalLMOutputWithPast]:

    if causal_mask is None:
        causal_mask = xformers.attn_bias.LowerTriangularMask()

    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
    outputs = self.model(
        input_ids=input_ids,
        causal_mask=causal_mask,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
    )

    hidden_states = outputs[0]
    logits = self.lm_head(hidden_states)

    loss = None
    if labels is not None:
        shift_logits = logits
        if not hasattr(self, "extra_ignored_labels"):
            # Fixes https://github.com/unslothai/unsloth/issues/10
            self.extra_ignored_labels = torch.full((self.max_seq_length, 1), -100, device = "cuda")
        
        shift_labels = torch.hstack((labels[..., 1:], self.extra_ignored_labels[:labels.shape[0]]))
        loss = fast_cross_entropy_loss(
            logits = shift_logits,
            labels = shift_labels,
        )

    if not return_dict:
        output = (logits,) + outputs[1:]
        return (loss,) + output if loss is not None else output

    return CausalLMOutputWithPast(
        loss=loss,
        logits=logits,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
    )


def PeftModelForCausalLM_fast_forward(
    self,
    input_ids=None,
    causal_mask=None,
    attention_mask=None,
    inputs_embeds=None,
    labels=None,
    output_attentions=None,
    output_hidden_states=None,
    return_dict=None,
    task_ids=None,
    **kwargs,
):
    return self.base_model(
        input_ids=input_ids,
        causal_mask=causal_mask,
        attention_mask=attention_mask,
        inputs_embeds=inputs_embeds,
        labels=labels,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
        **kwargs,
    )


