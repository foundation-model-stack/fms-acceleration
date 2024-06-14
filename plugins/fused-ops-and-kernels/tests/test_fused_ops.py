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

torch.nn.Dropout = DummyDropout
torch.nn.functional.dropout = lambda x, p, training: torch.tril(x, diagonal=0)

class Test:
    def __init__(self, base_type) -> None:
        # attention input
        self.x = torch.rand((1, 128, 2048), dtype=torch.float16, requires_grad=True, device="cuda")
        # quantize layer type
        self.base_type = base_type
        # model name
        if self.base_type == BNB:
            self.test_model_name = "TinyLlama/TinyLlama-1.1B-Chat-v0.3"
        elif self.base_type == GPTQ:
            self.test_model_name = "TheBloke/TinyLlama-1.1B-Chat-v0.3-GPTQ"
        else:
            raise NotImplementedError("Invalid base type") 

        self.r = 8
        self.lora_alpha = 1.0
        self.lora_dropout = 0.2

    def load_model(self):
        raise NotImplementedError

    def run_model(self, attention_mod, foak_patch):
        position_ids = torch.arange(self.x.size(1), device=self.x.device).view(1, -1)
        if foak_patch:
            attention_mod = patch_model(attention_mod, base_type=self.base_type)
            print(patch_model_summary())
        out, _, _ = attention_mod(self.x, position_ids=position_ids)
        loss = out.norm()
        loss.backward()
        grad_norms = [
            loss.item(),
            attention_mod.q_proj.lora_A.default.weight.grad.norm().item(),
            attention_mod.k_proj.lora_A.default.weight.grad.norm().item(),
            attention_mod.v_proj.lora_A.default.weight.grad.norm().item(),
            attention_mod.o_proj.lora_A.default.weight.grad.norm().item(),
            attention_mod.q_proj.lora_B.default.weight.grad.norm().item(),
            attention_mod.k_proj.lora_B.default.weight.grad.norm().item(),
            attention_mod.v_proj.lora_B.default.weight.grad.norm().item(),
            attention_mod.o_proj.lora_B.default.weight.grad.norm().item(),
        ]
        return grad_norms

class Test1(Test):
    def __init__(self, base_type) -> None:
        super().__init__(base_type)
        self.model = self.load_model().to(self.x.device)

    def load_model(self):
        config = AutoConfig.from_pretrained(self.test_model_name)
        config.num_hidden_layers = 1
        if self.base_type == BNB:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
            test_model = AutoModelForCausalLM.from_pretrained(
                self.test_model_name,
                config = config,
                torch_dtype = torch.float16,
                quantization_config=bnb_config,
            )
        elif self.base_type == GPTQ:
            raise NotImplementedError

        peft_config = LoraConfig(
            r=self.r, 
            lora_alpha=self.lora_alpha, 
            lora_dropout=self.lora_dropout, 
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                ]
        )
        return get_peft_model(test_model, peft_config)

    def run_model(self, foak_patch = False):
        with torch.autocast(dtype=torch.float16, device_type="cuda"):
            attention_mod = deepcopy(self.model.base_model.model.model.layers[0].self_attn)
            return super().run_model(attention_mod, foak_patch)

class Test2(Test):
    def __init__(self, base_type) -> None:
        super().__init__(base_type)
        self.model = self.load_model().to(self.x.device)

    def load_model(self):
        config = AutoConfig.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v0.3")
        test_module = LlamaAttention(config, 0)
        if self.base_type == BNB:
            quant_cls = Linear4bit
            peft_cls = LoraBNBLinear4bit
            target_kwargs = {
                "compute_dtype": torch.float16,
                "quant_type": 'nf4',
                "quant_storage": torch.float16,
            }
        elif self.base_type == GPTQ:
            quant_cls = QuantLinear
            peft_cls = LoraGPTQLinear4bit
            target_kwargs = {}
        else:
            raise NotImplementedError

        test_module = Test2.replace_linear(
                                    test_module, 
                                    linear_replacement=quant_cls,
                                    **target_kwargs,
        )
        return self.wrap_peft(test_module, peft_cls)

    def run_model(self, foak_patch = False):
        if self.base_type == BNB:
            self.model.q_proj.base_layer.quant_state.dtype = torch.float16
            self.model.k_proj.base_layer.quant_state.dtype = torch.float16
            self.model.v_proj.base_layer.quant_state.dtype = torch.float16
            self.model.o_proj.base_layer.quant_state.dtype = torch.float16
        with torch.autocast(dtype=torch.float16, device_type="cuda"):
            attention_mod = deepcopy(self.model)
            return super().run_model(attention_mod, foak_patch)

    def wrap_peft(self, module, peft_cls):
        module.q_proj = peft_cls(module.q_proj, "default", r=self.r, lora_alpha=self.lora_alpha, lora_dropout = self.lora_dropout)
        module.k_proj = peft_cls(module.k_proj, "default", r=self.r, lora_alpha=self.lora_alpha, lora_dropout = self.lora_dropout)
        module.v_proj = peft_cls(module.v_proj, "default", r=self.r, lora_alpha=self.lora_alpha, lora_dropout = self.lora_dropout)
        module.o_proj = peft_cls(module.o_proj, "default", r=self.r, lora_alpha=self.lora_alpha, lora_dropout = self.lora_dropout)
        return module

    @staticmethod
    def replace_linear(model:torch.nn.Module, linear_replacement:torch.nn.Module, quant_config=None,
                   skip_modules=["lm_head"], **kwargs):
        """
        Replace linear modules with a new Linear module.
        Parameters:
            model (`torch.nn.Module`):
                Input model or `torch.nn.Module` as the function is run recursively.
            linear_replacement (`torch.nn.Module`):
                The linear module that replaces the old one. Only expects standard arguments.
                If other arguments need to be passed, use a lambda.
            skip_modules (`List[str]`, *optional*, defaults to `lm_head`):
                List of modules names not to convert. Defaults to `lm_head`.
        """
        for name, module in model.named_children():
            if name in skip_modules:
                continue

            if len(list(module.children())) > 0:
                Test2.replace_linear(module, linear_replacement, quant_config, skip_modules, **kwargs)

            if isinstance(module, torch.nn.Linear):
                if issubclass(linear_replacement, Linear4bit):
                    model._modules[name] = linear_replacement(
                        module.in_features,
                        module.out_features,
                        module.bias is not None,
                        **kwargs
                    )
                elif issubclass(linear_replacement, QuantLinear):
                    model._modules[name] = linear_replacement(
                        4,
                        -1,
                        module.in_features,
                        module.out_features,
                        module.bias is not None,
                        **kwargs
                    )
        return model

# This loads a single layer tiny llama model
def test_foak_dropout_1():
    for base_type in [BNB, GPTQ]:
        t1 = Test1(base_type = base_type)
        assert all([round(a,5)==round(b,5) for a,b in zip(t1.run_model(), t1.run_model(foak_patch=True))]), "Loss and gradients don't match"

# this loads a dummy attention module
def test_foak_dropout_2():
    for base_type in [BNB, GPTQ]:
        t2 = Test2(base_type = base_type)
        assert all([round(a,5)==round(b,5) for a,b in zip(t2.run_model(), t2.run_model(foak_patch=True))]), "Loss and gradients don't match"