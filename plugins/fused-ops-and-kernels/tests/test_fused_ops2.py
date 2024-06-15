import torch
from copy import deepcopy
from transformers import AutoConfig
from transformers.models.llama.modeling_llama import LlamaAttention

from peft.tuners.lora.bnb import Linear4bit as LoraBNBLinear4bit
from peft.tuners.lora.gptq import QuantLinear as LoraGPTQLinear4bit

from fms_acceleration_foak.models.model_patcher import patch_model, patch_model_summary

from itertools import product

import pytest

BNB = "bitsandbytes"
GPTQ = "auto_gptq"

TEST_MODELS = {
    BNB: (
        "TinyLlama/TinyLlama-1.1B-Chat-v0.3", 
        ['q_proj', 'k_proj', 'v_proj', 'o_proj']
    ),
    GPTQ: (
        "TheBloke/TinyLlama-1.1B-Chat-v0.3-GPTQ",
        ['q_proj', 'k_proj', 'v_proj', 'o_proj']
    ),
}

PEFT_CLS = {
    BNB: LoraBNBLinear4bit,
    GPTQ: LoraGPTQLinear4bit,
}

ADAPTER_NAME = 'default'

LOSS_TOL = 1e-5
ALLCLOSE_RTOL = 1e-3
ALLCLOSE_ATOL = 1e-4

FLOAT16 = 'float16'
DTYPES = [FLOAT16]

DROPOUTS = [0, 0.1, 0.5]

# r, lora_alpha
LORA_PARAMS = [(8, 1.0)]

# bs, seqlen, hiddim
SIZES = [(1, 128, 2048)]

# set a fixed dropout to match outputs between runs
class DummyDropout(torch.nn.Module):

    dropout_mask: torch.tensor = None

    def __init__(self):
        super().__init__()

    def forward(self, X):

        # may raise if dropout mask not set
        if self.training:
            return X * self.dropout_mask
        return X

@pytest.fixture()
def test_inputs(seed: int=42, device: torch.device='cuda'):
    torch.manual_seed(seed)

    for dtype in DTYPES:
        inputs = {
            (bs, seq_len, dim_size, dtype): (
                torch.rand((bs, seq_len, dim_size), dtype=getattr(torch, dtype)),
                torch.arange(seq_len).repeat(bs,1)
            )
            for bs, seq_len, dim_size in SIZES
        }
    yield {k: (v1.to(device) for v1 in v)for k,v in inputs.items()}

@pytest.fixture()
def dropout_masks(seed: int=42, device: torch.device='cuda'):
    torch.manual_seed(seed)

    masks = {}
    for d in DROPOUTS:
        binomial = torch.distributions.binomial.Binomial(probs=1-d)            
        for _, sl, hid in SIZES:
            if (sl, hid) not in masks:
                masks[(sl, hid, d)] = binomial.sample((sl, hid)).to(device)

    yield masks

@pytest.fixture()
def test_attention_layers(device: torch.device = 'cuda'):
    from bitsandbytes.nn.modules import Linear4bit
    from auto_gptq.nn_modules.qlinear.qlinear_tritonv2 import QuantLinear

    QUANT_CLS = {
        BNB: Linear4bit
    }

    # this is only done for BNB
    layers = {}
    base_type = BNB
    for dtype in DTYPES:
        for r, lora_alpha in LORA_PARAMS:
            model_name, target_modules = TEST_MODELS[base_type]
            quant_cls = QUANT_CLS[base_type]
            peft_cls = PEFT_CLS[base_type]

            if base_type == BNB:
                base_type_kwargs = {
                    "compute_dtype": getattr(torch, dtype),
                    "quant_type": 'nf4', # TODO how aboute fp4 also?
                    "quant_storage": getattr(torch, dtype),
                }
            elif base_type == GPTQ:
                base_type_kwargs = {"bits": 4,"group_size": -1}
            
            # use the llama model
            config = AutoConfig.from_pretrained(model_name)
            attn_module = LlamaAttention(config, layer_idx=0)
            for tm in target_modules:
                mod = getattr(attn_module, tm)
                quant_base_layer = quant_cls(
                    input_features=mod.in_features,
                    output_features=mod.out_features,
                    bias=mod.bias is not None,
                    **base_type_kwargs,
                )

                if base_type == BNB:
                    quant_base_layer = quant_base_layer.to(device)
                    # TODO sets quant_state dtype to float16 here 1st to avoid type mismatch error
                    # will revisit later to find out how to initialize quant_state on casting to device
                    quant_base_layer.weight.quant_state.dtype = getattr(torch, dtype)

                lora_linear_layer = peft_cls(
                    quant_base_layer, ADAPTER_NAME, 
                    r=r, lora_alpha=lora_alpha, 
                    lora_dropout=0. # will override the dropout anyway
                )

                # this means all target modules get the same dropout
                lora_linear_layer.lora_dropout = torch.nn.ModuleDict(
                    [[ADAPTER_NAME, DummyDropout()]]
                )
                setattr(attn_module, tm, lora_linear_layer)

            layers[(base_type, r, lora_alpha, dtype)] = attn_module.to(device)

    yield layers


def prepare_foak(attn_module, base_type):

    _model = patch_model(attn_module, base_type=base_type)
    print(patch_model_summary())

    return _model

def run_model(
    model, target_modules, dtype, X, 
    device: torch.device = 'cuda', **kwargs, 
):
    with torch.autocast(dtype=getattr(torch, dtype), device_type=device):
        out, _, _ = model(X, **kwargs)
    loss = out.norm()
    loss.backward()

    # comparing the grads on the adapters
    adapter_grads = []
    for tm in target_modules:
        for n in ['lora_A', 'lora_B']:
            mod = model.get_submodule(f"{tm}.{n}.{ADAPTER_NAME}")
            adapter_grads.append(mod.weight.grad)

    return loss, adapter_grads

# small
def test_adapter_gradients_match_with_attention_layer(
    test_inputs, test_attention_layers, dropout_masks
):
    '''
    this function loads a dummy attention module, 
    For GPTQ it isn't straightforward to initialize a dummy baselayer, 
    so this function is only supports BNB for now
    '''
    for (bs, sl, hd), dtype in product(SIZES, DTYPES):
        # only running on one size for now
        X, position_ids = test_inputs[(bs, sl, hd, dtype)]
        _kwargs = {'position_ids': position_ids}
        for base_type in [BNB]:
            for r, lora_alpha in LORA_PARAMS:
                for d in DROPOUTS:

                    # attn layer + mask
                    attn = test_attention_layers[(base_type, r, lora_alpha, dtype)]
                    DummyDropout.dropout_mask = dropout_masks[(sl, hd, d)]
                    _, target_modules = TEST_MODELS[base_type]

                    # prepare models
                    without_foak = deepcopy(attn)
                    with_foak = prepare_foak(deepcopy(attn), base_type)

                    # run the models and get the loss and gradients
                    loss_unpatched, grads_unpatched = run_model(
                        without_foak, target_modules, dtype, X, **_kwargs
                    )
                    loss_patched, grads_patched = run_model(
                        with_foak, target_modules, dtype, X, **_kwargs
                    )

                    # compute outputs
                    assert (loss_unpatched - loss_patched).abs() < LOSS_TOL,\
                        "Loss after foak patch do not match"

                    for x, y in zip(grads_unpatched, grads_patched):
                        assert torch.allclose(x, y, atol=ALLCLOSE_ATOL, rtol=ALLCLOSE_RTOL), "Gradients don't match after foak patch"
