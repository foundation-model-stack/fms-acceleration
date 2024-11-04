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

FILE_SAFETENSOR_INDEX = "model.safetensors.index.json"
KEY_REPLICATE = "replicate"
KEY_EXPERT_PARALLEL = "expert_parallel"
DIM_EXPERT = 0

KEY_SCATTERMOE_ROUTER = "router.weight"

# Currently out ScatterMoE drop supports an up/down proj, and
# and optional gate_proj.
# - When new architectures are supported this list will update
SCATTERMOE_HAS_GATE_WEIGHT_SPEC = "has_gate_proj"

# - moe_cls
# - router_name
# - expert_name
# - weight_spec
# - sharded experts

# NOTE: it is quite challenging to perform the module swap
# when the incoming MoE model can have quite varied impls.
# - hence the adopted strategy is to infer the weights from the
#   state dict of the incoming model, and map them to the ScatterMoE
# - the SPEC is a description of some hints to help perfom this state_dict
#   mapping.

# NOTE: there is an expert_map logic which is currently not exposed
# in the SPEC. the expert_map allows us to map the the parameter names
# if they are different. But so far we do not need to use it.

# when adding new models, follow the following convention:
# - class name of moe module to be replaced with ScatterMoE.
# - module_name of the router.
# - module_name of the experts; this can be specified as a plain
#   name or a regex.
#   (str): name of the module
#   (regex): w1_name|w2_name|w3_name specificy the names if they are different.
# - boolean flag indicating if the experts are sharded in the state dict.
#   i.e., meaning the experts exist in seperate 2D Linear modules
#   or all "combined" into a single 3D linear module.
SCATTERMOE_CONVERSION_SPEC = {
    "MixtralForCausalLM": (
        "MixtralSparseMoeBlock",
        "gate",
        "experts",
        SCATTERMOE_HAS_GATE_WEIGHT_SPEC,
        True,
    ),
    "GraniteMoeForCausalLM": (
        "GraniteMoeMoE",
        "router",
        "input_linear|output_linear|input_linear",
        SCATTERMOE_HAS_GATE_WEIGHT_SPEC,
        False,
    ),
}
