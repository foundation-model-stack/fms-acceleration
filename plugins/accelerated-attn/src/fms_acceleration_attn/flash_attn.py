import torch
from transformers.utils import is_flash_attn_2_available
from types import MethodType

if is_flash_attn_2_available():
    from flash_attn import flash_attn_varlen_func

def prepare_fa2_from_position_ids(query, key, value, position_ids, query_length):
    query = query.view(-1, query.size(-2), query.size(-1))
    key = key.view(-1, key.size(-2), key.size(-1))
    value = value.view(-1, value.size(-2), value.size(-1))
    position_ids = position_ids.flatten()
    indices_q = torch.arange(position_ids.size(0), device=position_ids.device, dtype=torch.int32)
    cu_seq_lens = torch.cat((
        indices_q[position_ids==0],
        torch.tensor(position_ids.size(), device=position_ids.device, dtype=torch.int32)
        ))
    max_length = position_ids.max()+1
    return (query, key, value, indices_q, (cu_seq_lens, cu_seq_lens), (max_length, max_length))

# we can replace with the model patcher eventually
from transformers.models.llama.modeling_llama import LlamaFlashAttention2
def build_fa_forward(
    attention: torch.nn.Module, causal: bool = True,
):
    # assert not hasattr(self, '_position_ids'), "cannot patch fa attention"

    position_ids: torch.Tensor = None
    old_forward = attention.forward

    def forward(self, *args, **kwargs):
        nonlocal position_ids
        position_ids = kwargs['position_ids']
        return old_forward(*args, **kwargs)

    def _flash_attention_forward(
        self, query_states, key_states, value_states, attention_mask, query_length, dropout=0.0, softmax_scale=None
    ):
        # if not self._flash_attn_uses_top_left_mask:
        #     causal = self.is_causal
        # else:
        #     # TODO: Remove the `query_length != 1` check once Flash Attention for RoCm is bumped to 2.1. For details, please see the comment in LlamaFlashAttention2 __init__.
        #     causal = self.is_causal and query_length != 1

        assert attention_mask is None, "should not be using attention mask"
        assert position_ids is not None, "should be expecting position ids"
        batch_size = query_states.size(0)
        query_states, key_states, value_states, indices_q, cu_seq_lens, max_seq_lens = prepare_fa2_from_position_ids(
            query_states, key_states, value_states, position_ids, query_length
        )

        cu_seqlens_q, cu_seqlens_k = cu_seq_lens
        max_seqlen_in_batch_q, max_seqlen_in_batch_k = max_seq_lens

        attn_output = flash_attn_varlen_func(
            query_states,
            key_states,
            value_states,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_in_batch_q,
            max_seqlen_k=max_seqlen_in_batch_k,
            dropout_p=dropout,
            softmax_scale=softmax_scale,
            causal=causal,
        )

        return attn_output.view(batch_size, -1, attn_output.size(-2), attn_output.size(-1))
    
    # do this replace
    attention._flash_attention_forward = MethodType(_flash_attention_forward, attention)

    # return the forward
    return forward
