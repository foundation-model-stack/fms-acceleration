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

# utilities to update megablocks to register various things
# e.g, the MLP_v2 that handles gate, up, down projections

# Third Party
import torch
import torch.nn.functional as F


# this function ensures that the megablocks packaged is configured to use
# the correct SparseMLP implementation
# - at the moment not considering the GroupedMLP implementations
def update_mlp_registry(
    mlp_type: str = "sparse",
    mlp_version: str = "v2",
    load_balancing_loss: bool = False,
):
    # guarded
    # Third Party
    # pylint: disable=import-error,import-outside-toplevel
    from megablocks.layers.dmlp_registry import _REGISTRY
    from megablocks.layers.mlp import SparseMLP, resolve_dtensor
    from megablocks.layers.moe import ParallelMLP
    from megablocks.layers.router import LearnedRouter, _uniform_expert_assignment

    # Local
    from .sparse_mlp2 import SparseMLPv2

    SPARSE_MLP_IMPL = {
        "v1": SparseMLP,
        "v2": SparseMLPv2,
    }

    # replace the registry to point to the the correct sparse implementation
    if mlp_type == "sparse":
        assert (
            mlp_version in SPARSE_MLP_IMPL
        ), f"Megablocks only support sparse mlp versions: {','.join(SPARSE_MLP_IMPL.keys())}"
        _REGISTRY["mlp"]["sparse"] = SPARSE_MLP_IMPL[mlp_version]
    else:
        raise NotImplementedError("Currently only supports sparse MLP implementations.")

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
        # logits out of the MoE forward.
        if load_balancing_loss:
            return x, torch.log(scores)

        # otherwise just return None
        return x, None

    # replace the forward function. Willing to do this because ParallelMLP
    # is only used here and not anywhere else, hence:
    # 1. we do not care about reversing the patch
    # 2. we have control on where this is called, and we know to call it
    #    before our code accesses this function. Hence, we view this as
    #    a hardcoded modification to the megablocks package more than a
    #    patch.
    ParallelMLP.forward = forward

    # for the router
    # - need to resolve the dtensor since we had replicated the router
    #   weights
    def forward_router(self, x):
        if self.training and self.args.moe_jitter_eps is not None:
            x = x * self.jitter(x)

        _weight = resolve_dtensor(self.layer.weight)
        _bias = None if self.layer.bias is None else resolve_dtensor(self.layer.bias)
        # pylint: disable=not-callable
        scores = F.linear(x.view(-1, x.shape[-1]), _weight, _bias).softmax(dim=-1)
        expert_weights, expert_indices = self._top_k(scores)
        if self.args.moe_normalize_expert_weights:
            expert_weights = expert_weights / torch.norm(
                expert_weights,
                p=self.args.moe_normalize_expert_weights,
                dim=-1,
                keepdim=True,
            )

        expert_indices = (
            _uniform_expert_assignment(
                expert_indices,
                self.args.moe_num_experts,
            )
            if self.args.uniform_expert_assignment
            else expert_indices
        )
        return scores, expert_weights, expert_indices

    # replace the forward function in the router
    # - same as above
    LearnedRouter.forward = forward_router

    # Third Party
    from fms_acceleration.model_patcher import patch_target_module

    # Local
    from .checkpoint_utils import (
        load_fsdp_model,
        load_fsdp_optimizer,
        save_fsdp_model,
        save_fsdp_optimizer,
    )

    patch_target_module("transformers.trainer.save_fsdp_model", save_fsdp_model)
    patch_target_module("transformers.trainer.save_fsdp_optimizer", save_fsdp_optimizer)
    patch_target_module("transformers.trainer.load_fsdp_model", load_fsdp_model)
    patch_target_module("transformers.trainer.load_fsdp_optimizer", load_fsdp_optimizer)
