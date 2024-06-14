import torch
from bitsandbytes.nn.modules import Linear4bit
from auto_gptq.nn_modules.qlinear.qlinear_tritonv2 import QuantLinear
from peft import get_peft_model, LoraConfig
from peft.tuners.lora.bnb import Linear4bit as LoraBNBLinear4bit
from peft.tuners.lora.gptq import QuantLinear as LoraGPTQLinear4bit
from copy import deepcopy
from transformers import AutoConfig
from transformers.models.llama.modeling_llama import LlamaAttention
from peft import LoraConfig, get_peft_model
from fms_acceleration_foak.models.model_patcher import patch_model, patch_model_summary
from transformers import AutoConfig, AutoModelForCausalLM, BitsAndBytesConfig
from types import MethodType
from functools import partial

BNB = "bitsandbytes"
GPTQ = "auto_gptq"

# set a fixed dropout to match outputs between runs
class DummyDropout(torch.nn.Module):
    def __init__(self, dropout_mask=None):
        super().__init__()
        self.dropout_mask = dropout_mask

    def forward(self, X):
        if self.training:
            return X * self.dropout_mask
        return X

def wrap_peft(module, peft_cls, r, lora_alpha, lora_dropout):
    module.q_proj = peft_cls(module.q_proj, "default", r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout)
    module.k_proj = peft_cls(module.k_proj, "default", r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout)
    module.v_proj = peft_cls(module.v_proj, "default", r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout)
    module.o_proj = peft_cls(module.o_proj, "default", r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout)
    return module

def load_gptq_model_weights(model_name, peft_config):
    from peft.tuners import LoraModel

    from auto_gptq import (
        AutoGPTQForCausalLM,
        BaseQuantizeConfig,
    )
    from auto_gptq.utils.peft_utils import (  # pylint: disable=import-outside-toplevel,import-error
        GPTQLoraModel,
        get_gptq_peft_model,
    )
    from fms_acceleration_peft.autogptq_utils import (  # pylint: disable=import-outside-toplevel
        create_new_module_peft,
        replace_module_peft,
    )
    quantize_config = BaseQuantizeConfig.from_pretrained(model_name)
    model = AutoGPTQForCausalLM.from_quantized(
                model_name,
                quantize_config=quantize_config,
                use_tritonv2=True,
                trainable=True,
            )
    model.model.model.layers = model.model.model.layers[:1]

    _old_create_new_module = LoraModel._create_new_module
    _old_replace_module = GPTQLoraModel._replace_module
    _create_new_module = partial(create_new_module_peft, target_cls=QuantLinear)
    LoraModel._create_new_module = staticmethod(_create_new_module)
    GPTQLoraModel._replace_module = MethodType(replace_module_peft, GPTQLoraModel)

    # Install GPTQ adapters using the AutoGPTQ package (with the above patches)
    model = get_gptq_peft_model(
        model,
        peft_config=peft_config,
        auto_find_all_linears=peft_config.target_modules is None,
        train_mode=True,  # install adapaters for training
    )

    LoraModel._create_new_module = staticmethod(_old_create_new_module)
    GPTQLoraModel._replace_module = MethodType(_old_replace_module, GPTQLoraModel)

    return model.model.model.layers[0].self_attn

def build_model(model_name, base_type, r, lora_alpha, lora_dropout, dummy=True):
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

    if dummy:
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
    else:
        peft_config = LoraConfig(
            r=r, 
            lora_alpha=lora_alpha, 
            lora_dropout=lora_dropout, 
            target_modules=["q_proj","k_proj","v_proj","o_proj"]
        )
        if base_type == BNB:
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type=base_type_kwargs["quant_type"],
                bnb_4bit_compute_dtype=base_type_kwargs["compute_dtype"],
                bnb_4bit_use_double_quant=True,
            )
            config.num_hidden_layers = 1
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                config = config,
                torch_dtype = torch.float16,
                quantization_config=quant_config,
            )
            model = get_peft_model(model, peft_config)
            return model.model.model.layers[0].self_attn
        elif base_type == GPTQ:
            return load_gptq_model_weights(model_name, peft_config)
        else:
            raise NotImplementedError("Invalid base type")

def prepare_attn_inputs(bs, seq_len, dim_size):
    x = torch.rand((bs, seq_len, dim_size), dtype=torch.float16, requires_grad=True, device="cuda")
    position_ids = torch.arange(seq_len, device=x.device).view(1, -1)
    return x, position_ids

def prepare_model(base_peft_model, base_type, dropout_mask, foak_patch=False):
    _model = base_peft_model.to("cuda")

    # Inject Dummy Dropout Here
    _model.q_proj.lora_dropout = torch.nn.ModuleDict([["default", DummyDropout(dropout_mask)]])
    _model.k_proj.lora_dropout = torch.nn.ModuleDict([["default", DummyDropout(dropout_mask)]])
    _model.v_proj.lora_dropout = torch.nn.ModuleDict([["default", DummyDropout(dropout_mask)]])
    _model.o_proj.lora_dropout = torch.nn.ModuleDict([["default", DummyDropout(dropout_mask)]])

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

def run_model(model, X, position_ids):
    with torch.autocast(dtype=torch.float16, device_type="cuda"):
        out, _, _ = model(X, position_ids=position_ids)
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

# small
def test_adapter_gradients_match_with_dummy_module():
    '''
    this function loads a dummy attention module, 
    For GPTQ it isn't straightforward to initialize a dummy baselayer, 
    so this function is only supports BNB for now
    '''
    for base_type in [BNB]:
        print(base_type)
        for adapter_dropout in [0., 0.1, 0.5]:
            # Prepare fixed test inputs
            X, position_ids = prepare_attn_inputs(bs=1, seq_len=128, dim_size=2048)
            # build base model
            base_peft_model = build_model(
                TEST_MODELS[base_type], base_type, r=8, lora_alpha=1, lora_dropout=adapter_dropout, dummy=True,
                )
            binomial = torch.distributions.binomial.Binomial(probs=1-adapter_dropout)            
            dropout_mask = binomial.sample(X.size()).to(X.device) * (1.0/(1-adapter_dropout))

            # prepare 2 model variants to compare before and after foak patching
            unpatched = prepare_model(deepcopy(base_peft_model), base_type, dropout_mask, foak_patch=False)
            patched = prepare_model(deepcopy(base_peft_model), base_type, dropout_mask, foak_patch=True)
            # run the models and get the loss and gradients
            loss_unpatched, grads_unpatched = run_model(unpatched, X, position_ids)
            loss_patched, grads_patched = run_model(patched, X, position_ids)
            assert (round(loss_unpatched.item(),5) == round(loss_patched.item(),5)), "Loss after foak patch do not match"
            A1, B1 = grads_unpatched
            A2, B2 = grads_patched
            # check the gradients on the adapters match with some tolerance before patching and after
            match_gradients_A = (~torch.isclose(A1, A2, rtol=1e-05, atol=1e-05)).sum() == 0
            match_gradients_B = (~torch.isclose(B1, B2, rtol=1e-05, atol=1e-05)).sum() == 0
            assert match_gradients_A, "Gradients on lora A don't match after foak patch"
            assert match_gradients_B, "Gradients on lora B don't match after foak patch"

# large
def test_adapter_gradients_match_with_model_weights():
    '''
    this function initializes a full model with actual model weights
    and runs the attention module from the 1st layer
    '''
    for base_type in [BNB, GPTQ]:
        print(base_type)
        for adapter_dropout in [0.1, 0.5]:
            # Prepare fixed test inputs
            X, position_ids = prepare_attn_inputs(bs=1, seq_len=128, dim_size=2048)
            base_peft_model = build_model(
                TEST_MODELS[base_type], base_type, r=8, lora_alpha=1, lora_dropout=adapter_dropout, dummy=False
                )
            # prepare 2 model variants to compare before and after foak patching
            binomial = torch.distributions.binomial.Binomial(probs=1-adapter_dropout)            
            dropout_mask = binomial.sample((X.size(1), X.size(2))).to(X.device) * (1.0/(1-adapter_dropout))
            
            unpatched = prepare_model(deepcopy(base_peft_model), base_type, dropout_mask, foak_patch=False)
            patched = prepare_model(deepcopy(base_peft_model), base_type, dropout_mask, foak_patch=True)
            # run the models and get the loss and gradients
            loss_unpatched, grads_unpatched = run_model(unpatched, X, position_ids)
            loss_patched, grads_patched = run_model(patched, X, position_ids)
            assert (round(loss_unpatched.item(),5) == round(loss_patched.item(),5)), "Loss after foak patch don't match"
            A1, B1 = grads_unpatched
            A2, B2 = grads_patched
            # check the gradients on the adapters match with some tolerance before patching and after
            match_gradients_A = (~torch.isclose(A1, A2, rtol=1e-03, atol=1e-03)).sum() == 0
            match_gradients_B = (~torch.isclose(B1, B2, rtol=1e-03, atol=1e-03)).sum() == 0

            assert match_gradients_A, "Gradients on lora A do not match after foak patch"
            assert match_gradients_B, "Gradients on lora B do not match after foak patch"