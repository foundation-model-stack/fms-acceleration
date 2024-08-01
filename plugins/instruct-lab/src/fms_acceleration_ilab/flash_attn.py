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

import os
import inspect
from functools import partial
import torch
from transformers.utils import is_flash_attn_2_available, is_flash_attn_greater_or_equal
from typing import Optional

if is_flash_attn_2_available():
    from flash_attn import flash_attn_func, flash_attn_varlen_func # pylint: disable=import-error
    _flash_supports_window_size = "window_size" in list(inspect.signature(flash_attn_func).parameters)

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

# model id -> position_ids
POSITION_IDS_CACHE = {}

# - needed to store position ids when first come into model
# will pass these to the flash attention function
def build_backbone_forward(
    model: torch.nn.Module, model_id: str,
):
    # forward
    old_forward = model.forward

    # the model will get out the position
    def forward(self, *args, **kwargs):
        # store position ids
        POSITION_IDS_CACHE[model_id] = kwargs['position_ids']
        return old_forward(*args, **kwargs)

    return forward
  
def build_fa_forward(
    attention: torch.nn.Module, model_id: str,
):

    # this is really a dummpy replace 
    old_forward = attention.forward
    def forward(self, *args, **kwargs):
        out, *others = old_forward(*args, **kwargs)
        return out, *others

    _flash_attn = partial(
        _flash_attention_forward_with_posids, model_id
    )

    # do this replace of a method with a static
    attention._flash_attention_forward = _flash_attn

    # return the forward
    return forward

# FIXME: it is difficult to keep up with all the different versions
# - this is a generic version that accepts 
def _flash_attention_forward_with_posids(
    model_id: str,
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    attention_mask: torch.Tensor,
    query_length: int,
    is_causal: bool = True, # make this optional to support < 4.43
    dropout: float = 0.0,
    position_ids: Optional[torch.Tensor] = None,
    softmax_scale: Optional[float] = None,
    sliding_window: Optional[int] = None,
    use_top_left_mask: bool = False,
    softcap: Optional[float] = None,
    deterministic: bool = None,
    **kwargs,
):
    """
    Calls the forward method of Flash Attention - if the input hidden states contain at least one padding token
    first unpad the input, then computes the attention scores and pad the final attention scores.
    """
    position_ids = POSITION_IDS_CACHE[model_id]
    
    if not use_top_left_mask:
        causal = is_causal
    else:
        # TODO: Remove the `query_length != 1` check once Flash Attention for RoCm is bumped to 2.1. For details, please see the comment in transformers.models.llama.modeling_llama.LlamaFlashAttention2.__init__.
        causal = is_causal and query_length != 1

    # for supporting < 4.43
    use_sliding_windows = kwargs.get("use_sliding_windows")
    if use_sliding_windows is None:
        # Assuming 4D tensors, key_states.shape[1] is the key/value sequence length (source length).
        use_sliding_windows = (
            _flash_supports_window_size and sliding_window is not None and key_states.shape[1] > sliding_window
        )
    flash_kwargs = {"window_size": (sliding_window, sliding_window)} if use_sliding_windows else {}

    try:
        if is_flash_attn_greater_or_equal("2.4.1"):
            if deterministic is None:
                deterministic = os.environ.get("FLASH_ATTENTION_DETERMINISTIC", "0") == "1"
            flash_kwargs["deterministic"] = deterministic
    except:
        # FIXME: is_flash_attn_greater_or_equal expects a packaging.version
        # object for < 4.43
        # - we just assume that this deterministic flag is not impt
        pass

    if softcap is not None:
        flash_kwargs["softcap"] = softcap

    assert attention_mask is None, "should not be using attention mask"
    assert position_ids is not None, "should be expecting position ids"
    batch_size = query_states.size(0)
    (
        query_states,
        key_states,
        value_states,
        _,
        cu_seq_lens,
        max_seq_lens,
    ) = prepare_fa2_from_position_ids(
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
        **flash_kwargs,
    )

    attn_output = attn_output.view(batch_size, -1, attn_output.size(-2), attn_output.size(-1))

    return attn_output