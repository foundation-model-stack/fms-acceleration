import torch
from megablocks.layers import common
from megablocks.layers.arguments import Arguments
from megablocks.layers import mpu
from megablocks.layers.mlp import (
    create_dmoe_expert_weights, scale_gradient,
    resolve_dtensor
)
import torch.nn.functional as F
import stk

class SparseMLPv2(torch.nn.Module):

    def __init__(self, args : Arguments):
        super().__init__()
        self.args = args
        self._num_rows_per_rank = mpu.experts_per_rank(args) * mpu.features_per_rank(args)


        self.w1 = torch.nn.Parameter(torch.empty(
            self._num_rows_per_rank,
            args.hidden_size,
            device=args.device,
            dtype=common.dtype(args)))
        self.w2 = torch.nn.Parameter(torch.empty(
            self._num_rows_per_rank,
            args.hidden_size,
            device=args.device,
            dtype=common.dtype(args)))
        self.w3 = torch.nn.Parameter(torch.empty(
            self._num_rows_per_rank,
            args.hidden_size,
            device=args.device,
            dtype=common.dtype(args)))

        with torch.no_grad():
            self.w1.copy_(create_dmoe_expert_weights(
                args, args.moe_num_experts, args.ffn_hidden_size,
                args.hidden_size, args.init_method))
            self.w2.copy_(create_dmoe_expert_weights(
                args, args.moe_num_experts, args.ffn_hidden_size,
                args.hidden_size, args.output_layer_init_method))
            self.w3.copy_(create_dmoe_expert_weights(
                args, args.moe_num_experts, args.ffn_hidden_size,
                args.hidden_size, args.output_layer_init_method))

        self._should_set_parallelism_attribute = args.moe_expert_model_parallelism
        mpu.set_expert_model_parallel_attributes(
            self.w1, self._should_set_parallelism_attribute)
        mpu.set_expert_model_parallel_attributes(
            self.w2, self._should_set_parallelism_attribute)
        mpu.set_expert_model_parallel_attributes(
            self.w3, self._should_set_parallelism_attribute)

        self.gradient_scale = None
        if self.args.moe_expert_model_parallelism:
            self.gradient_scale = 1 / mpu.get_expert_parallel_world_size(self.args)

    def scale_grad(self, w):
        if self.gradient_scale is None:
            return w
        return scale_gradient(w, self.gradient_scale)

    def forward(self, hidden_states, topo):
        w1, w2, w3 = (
            self.scale_grad(self.w1), self.scale_grad(self.w2), 
            self.scale_grad(self.w3)
        )
        w1, w2, w3 = (
            resolve_dtensor(w1), resolve_dtensor(w2), resolve_dtensor(w3)
        )

        # Perform the expert computation
        hidden_states = stk.Matrix(  # type: ignore
            topo.size(),
            F.silu(stk.ops.sdd(hidden_states, w1.t(), topo).data)
            * stk.ops.sdd(hidden_states, w3.t(), topo).data,
            topo.row_indices,
            topo.column_indices,
            topo.offsets,
            topo.column_indices_t,
            topo.offsets_t,
            topo.block_offsets_t,
        )
        return stk.ops.dsd(hidden_states, w2)