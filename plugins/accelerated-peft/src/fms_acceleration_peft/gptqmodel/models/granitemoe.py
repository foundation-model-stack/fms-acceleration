###############################################################################
# Adapted from https://github.com/ModelCloud/GPTQModel
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
###############################################################################
# Local
from .base import BaseGPTQModel


class GraniteMoeGPTQ(BaseGPTQModel):
    base_modules = ["model.embed_tokens", "model.norm"]
    # convert_3d_modulelist = [
    #     "block_sparse_moe.input_linear",
    #     "block_sparse_moe.output_linear",
    # ]

    layers_node = "model.layers"
    layer_type = "GraniteMoeDecoderLayer"

    # NOTE: we should look at dynamic_expert_index so we dont have
    # to write out experts
    layer_modules = [
        ["self_attn.k_proj", "self_attn.v_proj", "self_attn.q_proj"],
        ["self_attn.o_proj"],
        [f"block_sparse_moe.input_linear.weight.{i}" for i in range(40)],
        [f"block_sparse_moe.output_linear.weight.{i}" for i in range(40)],
    ]