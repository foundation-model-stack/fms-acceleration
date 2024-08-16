# utilities to update megablocks to register the MLP_v2 that 
# handles gate, up, down projections

from megablocks.layers.dmlp_registry import _REGISTRY


# from megablocks.layers import mlp
from .sparse_mlp2 import SparseMLPv2
from megablocks.layers.moe import ParallelMLP
import torch

def update_mlp_registry():
    # patch the registry to point to our v2
    _REGISTRY['mlp']['sparse'] = SparseMLPv2

    def forward(self, x, scores, expert_weights, top_experts):
        in_shape = x.size()

        # Compute the experts.
        x, _ = self.forward_fn(x, expert_weights, top_experts)

        x = x.view(in_shape)
        if self.bias is not None:
            if self.args.return_bias:
                return x, self.bias
            return x + self.bias

        # in this case we should be returning the router
        # logits out of the MoeE forward. However, since
        # the way the code is written now, it si difficult 
        # to extract these logits out, so at the moment,
        # we return None as the placeholder.
        return x, None

    # patch the forward function
    ParallelMLP.forward = forward
    