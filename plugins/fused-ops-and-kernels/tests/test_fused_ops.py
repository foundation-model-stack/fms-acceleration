import torch
from bitsandbytes.nn.modules import Linear4bit
from auto_gptq.nn_modules.qlinear.qlinear_tritonv2 import QuantLinear
from peft import get_peft_model, LoraConfig
from peft.tuners.lora.bnb import Linear4bit as LoraBNBLinear4bit
from peft.tuners.lora.gptq import QuantLinear as LoraGPTQLinear4bit
from peft.tuners import LoraModel
from copy import deepcopy
from transformers import AutoConfig
from transformers.models.llama.modeling_llama import LlamaAttention
from peft import LoraConfig, get_peft_model
from fms_acceleration_foak.models.model_patcher import patch_model, patch_model_summary
from transformers import AutoConfig, AutoModelForCausalLM, BitsAndBytesConfig

BNB = "bitsandbytes"
GPTQ = "auto_gptq"

# set a fixed dropout to match outputs between runs
class DummyDropout(torch.nn.Module):
    def __init__(self, p: float = 0.5):
        super(DummyDropout, self).__init__()
        if p < 0 or p > 1:
            raise ValueError("dropout probability has to be between 0 and 1, " "but got {}".format(p))
        self.p = p

    def forward(self, X):
        if self.training:            
            return torch.tril(X, diagonal=0)
        return X

def wrap_peft(module, peft_cls, r, lora_alpha, lora_dropout):
    module.q_proj = peft_cls(module.q_proj, "default", r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout)
    module.k_proj = peft_cls(module.k_proj, "default", r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout)
    module.v_proj = peft_cls(module.v_proj, "default", r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout)
    module.o_proj = peft_cls(module.o_proj, "default", r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout)
    return module

def build_model(model_name, base_type, r, lora_alpha, lora_dropout):
    def get_general_kwargs(mod):
        return {
            "input_features": mod.in_features,
            "output_features": mod.out_features,
            "bias": mod.bias is not None,
        }
    if base_type == BNB:
        config = AutoConfig.from_pretrained(model_name)
        quant_cls = Linear4bit
        peft_cls = LoraBNBLinear4bit
        base_type_kwargs = {
            "compute_dtype": torch.float16,
            "quant_type": 'nf4',
            "quant_storage": torch.float16,
        }
    elif base_type == GPTQ:
        config = AutoConfig.from_pretrained(model_name)
        quant_cls = QuantLinear
        peft_cls = LoraGPTQLinear4bit
        base_type_kwargs = {
            "bits": 4,
            "group_size": -1,
        }
    else:
        raise NotImplementedError

    test_module = LlamaAttention(config, 0)
    test_module.q_proj = quant_cls(
                    **get_general_kwargs(test_module.q_proj),
                    **base_type_kwargs
                )

    test_module.k_proj = quant_cls(
                    **get_general_kwargs(test_module.k_proj),
                    **base_type_kwargs
                )
    test_module.v_proj = quant_cls(
                    **get_general_kwargs(test_module.v_proj),
                    **base_type_kwargs
                )
    test_module.o_proj = quant_cls(
                    **get_general_kwargs(test_module.o_proj),
                    **base_type_kwargs
                )

    return wrap_peft(test_module, peft_cls, r, lora_alpha, lora_dropout)

def prepare_attn_inputs(bs, seq_len, dim_size):
    x = torch.rand((bs, seq_len, dim_size), dtype=torch.float16, requires_grad=True, device="cuda")
    position_ids = torch.arange(seq_len, device=x.device).view(1, -1)
    return x, position_ids

def prepare_model(base_peft_model, base_type, foak_patch=False):
    # TODO fix autogptq weights

    _model = base_peft_model.to("cuda")

    # Inject Dummy Dropout Here
    _model.q_proj.lora_dropout = torch.nn.ModuleDict(
        [
            ["default", DummyDropout(p=_model.q_proj.lora_dropout["default"].p)]
        ]
    )
    _model.k_proj.lora_dropout = torch.nn.ModuleDict(
        [
            ["default", DummyDropout(p=_model.k_proj.lora_dropout["default"].p)]
        ]
    )
    _model.v_proj.lora_dropout = torch.nn.ModuleDict(
        [
            ["default", DummyDropout(p=_model.v_proj.lora_dropout["default"].p)]
        ]
    )
    _model.o_proj.lora_dropout = torch.nn.ModuleDict(
        [
            ["default", DummyDropout(p=_model.o_proj.lora_dropout["default"].p)]
        ]
    )

    if foak_patch:
        if base_type == BNB:
            # TODO sets quant_state dtype to float16 here 1st to avoid type mismatch error
            # will revisit later to find out how to initialize quant_state on casting to device
            base_peft_model.q_proj.base_layer.weight.quant_state.dtype = torch.float16
            base_peft_model.k_proj.base_layer.weight.quant_state.dtype = torch.float16
            base_peft_model.v_proj.base_layer.weight.quant_state.dtype = torch.float16
            base_peft_model.o_proj.base_layer.weight.quant_state.dtype = torch.float16

        _model = patch_model(base_peft_model, base_type=base_type)
        print(patch_model_summary())

    return _model

def run_model(inputs, model):
    x, position_ids = inputs
    with torch.autocast(dtype=torch.float16, device_type="cuda"):
        out, _, _ = model(x, position_ids=position_ids)
    loss = out.norm()
    loss.backward()
    grad_norms_A = [
        model.q_proj.lora_B.default.weight.grad,
        model.k_proj.lora_B.default.weight.grad,
        model.v_proj.lora_B.default.weight.grad,
        model.o_proj.lora_B.default.weight.grad,
    ]
    grad_norms_B = [
        model.q_proj.lora_B.default.weight.grad,
        model.k_proj.lora_B.default.weight.grad,
        model.v_proj.lora_B.default.weight.grad,
        model.o_proj.lora_B.default.weight.grad,
    ]

    return loss, (torch.vstack(grad_norms_A), torch.vstack(grad_norms_B))

TEST_MODELS = {
    BNB: "TinyLlama/TinyLlama-1.1B-Chat-v0.3",
    GPTQ: "TheBloke/TinyLlama-1.1B-Chat-v0.3-GPTQ",
}

# this loads a dummy attention module
def test_adapter_gradients_match():
    for base_type in [BNB, GPTQ]:
        inputs = prepare_attn_inputs(bs=1, seq_len=128, dim_size=2048)
        base_peft_model = build_model(
            TEST_MODELS[base_type], base_type, r=8, lora_alpha=1, lora_dropout=0.1
            )
        unpatched = prepare_model(deepcopy(base_peft_model), base_type, foak_patch=False)
        patched = prepare_model(deepcopy(base_peft_model), base_type, foak_patch=True)
        loss_unpatched, grads_unpatched = run_model(inputs, unpatched)
        loss_patched, grads_patched = run_model(inputs, patched)
        assert (round(loss_unpatched.item(),5) == round(loss_patched.item(),5)), "Loss between unpatched and foak-patched don't match"
        A1, B2 = grads_unpatched
        A2, B2 = grads_patched
        import pdb; pdb.set_trace()        
        assert torch.isclose(A1, A2, rtol=1e-05, atol=1e-05).long()