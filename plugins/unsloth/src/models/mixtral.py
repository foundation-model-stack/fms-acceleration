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

from .mistral import *
from transformers.models.mixtral.modeling_mixtral import (
    MixtralAttention,
    MoeModelOutputWithPast,
    MoeCausalLMOutputWithPast,
    load_balancing_loss_func
)

# For Pytorch 2.1.1
try:
    from transformers.models.mixtral.modeling_mixtral import (
        MixtralSdpaAttention,
        MixtralFlashAttention2,
    )
except:
    MixtralSdpaAttention   = MixtralAttention
    MixtralFlashAttention2 = MixtralAttention
pass

def MixtralDecoderLayer_fast_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    output_attentions: Optional[bool] = False,
    output_router_logits: Optional[bool] = False,
    use_cache: Optional[bool] = False,
    *args, **kwargs,
) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
    residual = hidden_states
    hidden_states = fast_rms_layernorm(self.input_layernorm, hidden_states)

    # Self Attention
    hidden_states, self_attn_weights, present_key_value = self.self_attn(
        hidden_states=hidden_states,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_value=past_key_value,
        output_attentions=output_attentions,
        use_cache=use_cache,
    )
    hidden_states = residual + hidden_states

    # Fully Connected
    residual = hidden_states
    hidden_states = fast_rms_layernorm(self.post_attention_layernorm, hidden_states)
    hidden_states, router_logits = self.block_sparse_moe(hidden_states)
    hidden_states = residual + hidden_states

    outputs = (hidden_states,)

    if output_attentions:
        outputs += (self_attn_weights,)

    if use_cache:
        outputs += (present_key_value,)

    if output_router_logits:
        outputs += (router_logits,)

    return outputs
pass

def MixtralModel_fast_forward(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[List[torch.FloatTensor]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    output_router_logits: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    *args, **kwargs,
) -> Union[Tuple, MoeModelOutputWithPast]:
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_router_logits = (
        output_router_logits if output_router_logits is not None else self.config.output_router_logits
    )
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    use_cache = use_cache if use_cache is not None else self.config.use_cache

    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    # retrieve input_ids and inputs_embeds
    if input_ids is not None and inputs_embeds is not None:
        raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
    elif input_ids is not None:
        batch_size, seq_length = input_ids.shape
    elif inputs_embeds is not None:
        batch_size, seq_length, _ = inputs_embeds.shape
    else:
        raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")

    past_key_values_length = 0

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
        pass
    
    if self.gradient_checkpointing and self.training:
        if use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
            )
            use_cache = False

    # if use_cache:
    #     past_key_values_length = past_key_values.get_usable_length(seq_length)

    if position_ids is None:
        device = input_ids.device if input_ids is not None else inputs_embeds.device
        position_ids = torch.arange(
            past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
        )
        position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
    else:
        position_ids = position_ids.view(-1, seq_length).long()

    if inputs_embeds is None:
        inputs_embeds = self.embed_tokens(input_ids)

    if attention_mask is not None and self._attn_implementation == "flash_attention_2" and use_cache:
        is_padding_right = attention_mask[:, -1].sum().item() != batch_size
        if is_padding_right:
            raise ValueError(
                "You are attempting to perform batched generation with padding_side='right'"
                " this may lead to unexpected behaviour for Flash Attention version of Mixtral. Make sure to "
                " call `tokenizer.padding_side  = 'left'` before tokenizing the input. "
            )

    if self._attn_implementation == "flash_attention_2":
        # 2d mask is passed through the layers
        attention_mask = attention_mask if (attention_mask is not None and 0 in attention_mask) else None
    elif self._attn_implementation == "sdpa" and not output_attentions:
        # output_attentions=True can not be supported when using SDPA, and we fall back on
        # the manual implementation that requires a 4D causal mask in all cases.
        from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask_for_sdpa
        attention_mask = _prepare_4d_causal_attention_mask_for_sdpa(
            attention_mask,
            (batch_size, seq_length),
            inputs_embeds,
            past_key_values_length,
        )
    else:
        # 4d mask is passed through the layers
        from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask
        attention_mask = _prepare_4d_causal_attention_mask(
            attention_mask,
            (batch_size, seq_length),
            inputs_embeds,
            past_key_values_length,
            sliding_window=self.config.sliding_window,
        )

    hidden_states = inputs_embeds

    # decoder layers
    all_hidden_states = () if output_hidden_states else None
    all_self_attns = () if output_attentions else None
    all_router_logits = () if output_router_logits else None
    next_decoder_cache = None

    for decoder_layer in self.layers:
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        if self.gradient_checkpointing and self.training:
            layer_outputs = self._gradient_checkpointing_func(
                decoder_layer.__call__,
                hidden_states,
                attention_mask,
                position_ids,
                past_key_values,
                output_attentions,
                output_router_logits,
                use_cache,
            )
        else:
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                output_attentions=output_attentions,
                output_router_logits=output_router_logits,
                use_cache=use_cache,
            )

        hidden_states = layer_outputs[0]

        if use_cache:
            next_decoder_cache = layer_outputs[2 if output_attentions else 1]

        if output_attentions:
            all_self_attns += (layer_outputs[1],)

        if output_router_logits:
            all_router_logits += (layer_outputs[-1],)

    hidden_states = self.norm(hidden_states)

    # add hidden states from the last decoder layer
    if output_hidden_states:
        all_hidden_states += (hidden_states,)

    if not return_dict:
        return tuple(
            v
            for v in [hidden_states, next_decoder_cache, all_hidden_states, all_self_attns, all_router_logits]
            if v is not None
        )
    return MoeModelOutputWithPast(
        last_hidden_state=hidden_states,
        past_key_values=next_decoder_cache,
        hidden_states=all_hidden_states,
        attentions=all_self_attns,
        router_logits=all_router_logits,
    )

def MixtralForCausalLM_fast_forward(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[List[torch.FloatTensor]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    labels: Optional[torch.LongTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    output_router_logits: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    *args, **kwargs,
) -> Union[Tuple, MoeCausalLMOutputWithPast]:
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_router_logits = (
        output_router_logits if output_router_logits is not None else self.config.output_router_logits
    )

    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
    outputs = self.model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        output_router_logits=output_router_logits,
        return_dict=return_dict,
    )

    hidden_states = outputs[0]
    logits = self.lm_head(hidden_states)
    logits = logits.float()

    loss = None
    if labels is not None:
        # fixed by aaron.chew1@sg.ibm.com
        shift_logits = logits
        if not hasattr(self, "extra_ignored_labels"):
            # Fixes https://github.com/unslothai/unsloth/issues/10
            self.extra_ignored_labels = torch.full((self.max_seq_length, 1), -100, device = "cuda")

        shift_labels = torch.hstack((labels[..., 1:], self.extra_ignored_labels[:labels.shape[0]]))
        loss = fast_cross_entropy_loss(
            logits = shift_logits,
            labels = shift_labels,
        )

    aux_loss = None
    if output_router_logits:
        aux_loss = load_balancing_loss_func(
            outputs.router_logits if return_dict else outputs[-1],
            self.num_experts,
            self.num_experts_per_tok,
            attention_mask,
        )
        if labels is not None:
            loss += self.router_aux_loss_coef * aux_loss.to(loss.device)  # make sure to reside in the same device

    if not return_dict:
        output = (logits,) + outputs[1:]
        if output_router_logits:
            output = (aux_loss,) + output
        return (loss,) + output if loss is not None else output

    return MoeCausalLMOutputWithPast(
        loss=loss,
        aux_loss=aux_loss,
        logits=logits,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
        router_logits=outputs.router_logits,
    )

def MixtralPeftModelForCausalLM_fast_forward(
    self,
    input_ids=None,
    causal_mask=None,
    attention_mask=None,
    inputs_embeds=None,
    labels=None,
    output_attentions=None,
    output_hidden_states=None,
    output_router_logits=None,
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
        output_router_logits=output_router_logits,
        return_dict=return_dict,
        **kwargs,
    )
