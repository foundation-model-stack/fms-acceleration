
FILE_SAFETENSOR_INDEX = "model.safetensors.index.json"
KEY_REPLICATE = "replicate"
KEY_EXPERT_PARALLEL = "expert_parallel"
DIM_EXPERT = 0

KEY_SCATTERMOE_ROUTER = 'router.weight'

# - moe_cls
# - router_name
# - expert_name
# - weight_spec
# - sharded experts
SCATTERMOE_HAS_GATE_WEIGHT_SPEC = 'has_gate_proj'
SCATTERMOE_CONVERSION_SPEC = {
    "MixtralForCausalLM": (
        'MixtralSparseMoeBlock', 'gate', 'experts', SCATTERMOE_HAS_GATE_WEIGHT_SPEC, True
    ),
    'GraniteMoeForCausalLM': (
        'GraniteMoeMoE', 'router', 'input_linear|output_linear|input_linear', SCATTERMOE_HAS_GATE_WEIGHT_SPEC, False
    )
}