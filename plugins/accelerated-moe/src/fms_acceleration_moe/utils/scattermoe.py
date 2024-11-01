# Third Party
from transformers.activations import ACT2FN
import torch
import torch.nn.functional as F
from torch.distributed._tensor import DTensor
from torch.distributed._tensor.device_mesh import DeviceMesh
from peft import LoraConfig
from peft.utils import INCLUDE_LINEAR_LAYERS_SHORTHAND

try:
    from khd.kernels.scattermoe.triton_implementation.ops import (
        scattered_experts, padded_block_indices
    )
except ImportError:
    pass
 
from .scattermoe_constants import SCATTERMOE_HAS_GATE_WEIGHT_SPEC
from .scattermoe_utils import all_to_all_gather_inputs, scatter_with_routing_weights

def resolve_dtensor(weight):
    if isinstance(weight, DTensor):
        return weight.to_local()
    return weight

class ScatteredExperts(torch.nn.Module):
    def __init__(
        self, 
        in_features: int,
        out_features: int,
        num_experts: int,
        fan_out: int,
        grouped_in: bool = False,
        grouped_out: bool = False,
        dtype: torch.dtype = torch.bfloat16,
        device: torch.device = torch.device('cpu'),
        lora_config: LoraConfig = None,
    ):
        super().__init__()
        self.weight = torch.nn.Parameter(
            torch.empty(
                num_experts, in_features, out_features, 
                dtype=dtype, device=device,
            ),
            requires_grad=True,
        )
        self.lora_A, self.lora_B = None, None
        self.lora_r = 0
        if lora_config is not None:
            # no gradient for base layer
            self.weight.requires_grad = False

            # NOTE : - for now adapter takes same dtype as base
            self.lora_A = torch.nn.Parameter(
                torch.empty(
                    num_experts, in_features, lora_config.r,
                    dtype=dtype, device=device,
                ),
                requires_grad=True,
            )
            self.lora_B = torch.nn.Parameter(
                torch.empty(
                    num_experts, lora_config.r, out_features,
                    dtype=dtype, device=device,
                ),
                requires_grad=True,
            )
            self.lora_r = lora_config.r
            
            # NOTE: call init_lora to initialize the adapters
            # - not called during initialization

        self.fan_out = fan_out
        self.grouped_in = grouped_in
        self.grouped_out = grouped_out

    def forward(
        self, x, bin_ids, indices, padded_block_idxs, 
        expert_offsets, gates=None,
    ):
        weight = resolve_dtensor(self.weight)
        lora_A, lora_B = None, None
        if self.lora_r > 0:
            lora_A, lora_B = (
                resolve_dtensor(self.lora_A), resolve_dtensor(self.lora_B)
            )
        return scattered_experts(
            x,
            weight,
            self.fan_out,
            bin_ids, # sorted_expert_idxs,
            indices, # sorted_scattered_idxs,
            padded_block_idxs,
            expert_offsets,
            gates=gates, # we dont have router weights
            grouped_in=self.grouped_in,
            grouped_out=self.grouped_out,
            expert_lora_A=lora_A,
            expert_lora_B=lora_B,
            lora_alp=self.lora_r,
        )

# similar to of MoE_Triton from https://github.com/mayank31398/kernel-hyperdrive
# and ParameterizedScatteredExperts from https://github.com/IBM/dolomite-engine/blob/main/dolomite_engine/hf_models/models/moe_dolomite/moe/scatter.py
# - support expert parallel where the data is communicated via all_to_all 
class ScatterMoE(torch.nn.Module):

    def __init__(
        self, 
        hidden_size: int,
        hidden_act: str,
        intermediate_size: int,
        num_experts: int,
        has_bias: bool = False,
        mlp_arch: str = None,
        top_k: int = 2,
        dtype: torch.dtype = torch.bfloat16,
        device: str = torch.device('cpu'),
        device_mesh: DeviceMesh = None,
        key_ep: str = None,
        lora_config: LoraConfig = None,
    ):
        assert has_bias == False, \
            "ScatterMoE currently unable to handle bias in both gates and experts."

        if lora_config is not None:
            # since this is self implemented, we really only support basic lora funcs
            assert lora_config.bias == 'none', \
                "ScatterMoE currently unable to handle bias in the lora adapters"
            assert (
                lora_config.target_modules == INCLUDE_LINEAR_LAYERS_SHORTHAND or 
                INCLUDE_LINEAR_LAYERS_SHORTHAND in lora_config.target_modules
            ), \
                "ScatterMoe currently only handles lora adapters on all linears."

            assert lora_config.init_lora_weights in {True, 'gaussian'}, \
                "ScatterMoe currently only handles gaussian initialization."

        super().__init__()

        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_experts = num_experts
        self.hidden_act = hidden_act
        self.activation = ACT2FN[hidden_act]
        self.top_k = top_k
        self.all_to_all = (
            device_mesh[key_ep].size() > 1 
            if device_mesh is not None
            else False
        )

        # NOTE: we should then use this to distribute inside
        # and not do the distribution outside
        self.expert_parallel_group = (
            device_mesh[key_ep].get_group(0) 
            if device_mesh is not None
            else None
        )

        # build the router
        self.router = torch.nn.Linear(
            in_features=hidden_size,
            out_features=num_experts,
            bias=has_bias,
            dtype=dtype,
            device=device,
        )

        # currently we only handle MLP's which have the "up_proj", "activate"
        # "down_proj" architecture, with the option of a "gate_project"
        # NOTE: in the future we handle this by passing into 
        # this class a spec on how many to create
        self.w1 = ScatteredExperts(
            in_features=self.hidden_size,
            out_features=self.intermediate_size,
            num_experts=self.num_experts,
            fan_out=self.top_k if not self.all_to_all else 1,
            grouped_out=True,
            dtype=dtype,
            device=device,
            lora_config=lora_config,
        )
        self.w2 = ScatteredExperts(
            in_features=self.intermediate_size,
            out_features=self.hidden_size,
            num_experts=self.num_experts,
            fan_out=1,
            grouped_in=True,
            dtype=dtype,
            device=device,
            lora_config=lora_config,
        )
        if mlp_arch == SCATTERMOE_HAS_GATE_WEIGHT_SPEC:
            self.w3 = ScatteredExperts(
                in_features=self.hidden_size,
                out_features=self.intermediate_size,
                num_experts=self.num_experts,
                fan_out=self.top_k if not self.all_to_all else 1,
                grouped_out=True,
                dtype=dtype,
                device=device,
                lora_config=lora_config,
            )

    # def add_expert(self, key, 
    # dolomite, MoE_Torch
    def _compute_routing_weights(self, hidden_states):

        # router_logits: (batch * sequence_length, n_experts)
        weight = resolve_dtensor(self.router.weight)
        bias = self.router.bias
        if bias: bias = resolve_dtensor(bias)
        router_logits = F.linear(hidden_states, weight, bias)

        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
        routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
        # we cast back to the input dtype
        routing_weights = routing_weights.to(hidden_states.dtype)
        return router_logits, routing_weights, selected_experts

    def _maybe_gather(self, hidden_states, selected_experts):
        # can replace with megablocks version of _indices_and_bins
        # this is if there is no 
        sorted_expert_idxs, sorted_scattered_idxs = torch.sort(selected_experts.flatten())
        if not self.all_to_all:
            # hidden states pass through
            return hidden_states, sorted_expert_idxs, sorted_scattered_idxs

        # needed for scattering later (if required)
        local_gather_products = (
            sorted_expert_idxs,
            sorted_scattered_idxs
        )

        # outputs will be parallel_x, parallel_bin_ids, parallel_ind
        # and followed by 
        # send_counts, recv_counts, bins (local)
        outputs = all_to_all_gather_inputs(
            hidden_states, selected_experts, 
            sorted_expert_idxs, sorted_scattered_idxs,
            self.expert_parallel_group, 
            self.top_k, 
            self.num_experts, 
        )

        return outputs + local_gather_products

    def _maybe_scatter(
        self, hidden_states, 
        routing_weights, original_shape, local_gather_products
    ):

        if not self.all_to_all:
            # in this case scattering is already handled by 
            # scattermoe when computing w2
            return hidden_states.view(original_shape)

        (
            send_counts, recv_counts,
            bins,
            sorted_expert_idxs,
            sorted_scattered_idxs
        ) = local_gather_products

        hidden_states = scatter_with_routing_weights(
            hidden_states,
            routing_weights.flatten(),
            send_counts, recv_counts,
            bins, original_shape, # local
            sorted_expert_idxs, sorted_scattered_idxs,
            self.expert_parallel_group, 
            self.top_k
        )
        return hidden_states

    def forward(self, hidden_states):

        original_shape = hidden_states.shape
        hidden_states = hidden_states.view(-1, self.hidden_size)

        # _, batch_index, batch_gates, expert_size, router_logits = self.router(layer_input)
        router_logits, routing_weights, selected_experts = self._compute_routing_weights(
            hidden_states
        )

        # maybe gather
        # - local_gather_products may or may not be non-empty
        (
            hidden_states, 
            sorted_expert_idxs, 
            sorted_scattered_idxs,
            *local_gather_products
        ) = self._maybe_gather(
            hidden_states, selected_experts
        )

        # padded indicies need to be computed for scattermoe
        with torch.no_grad():
            padded_block_idxs, expert_offsets = padded_block_indices(
                sorted_expert_idxs, self.num_experts
            )
        
        # the up projection
        out = self.w1(
            hidden_states,
            sorted_expert_idxs, sorted_scattered_idxs,
            padded_block_idxs, expert_offsets
        )
        out = self.activation(out)

        # - if defined, a seperate up projection
        if self.w3:
            out *= self.w3(
                hidden_states,
                sorted_expert_idxs, sorted_scattered_idxs,
                padded_block_idxs, expert_offsets
            ) 

        # the down projection
        hidden_states = self.w2(
            out,
            sorted_expert_idxs, sorted_scattered_idxs,
            padded_block_idxs, expert_offsets,
            gates=(
                None if self.all_to_all else
                routing_weights 
            )
        ) 

        # maybe scatter
        hidden_states = self._maybe_scatter(
            hidden_states, routing_weights, original_shape,
            local_gather_products,
        ) 

        return hidden_states, router_logits
