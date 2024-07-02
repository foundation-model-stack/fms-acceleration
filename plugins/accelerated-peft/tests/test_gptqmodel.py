from transformers.utils.import_utils import _is_package_available
import pytest  # pylint: disable=import-error
import torch
from typing import List
from types import MethodType
from functools import partial
from peft import LoraConfig, prepare_model_for_kbit_training
from peft.tuners.lora.gptq import QuantLinear as LoraLinearGPTQ
from peft.tuners.lora.model import LoraModel

GPTQ = "gptq"
# r, lora_alpha
FLOAT16 = "float16"
LORA_r = 8
LORA_alpha = 1.0
BS = 1
SEQLEN = 128

ALLCLOSE_RTOL = 1e-3
ALLCLOSE_ATOL = 1e-4

VANILLA_MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v0.3"
QUANTIZED_MODEL_NAME = "TheBloke/TinyLlama-1.1B-Chat-v0.3-GPTQ"
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]

def replace_module_peft(self, parent_module, child_name, new_module, old_module):
    setattr(parent_module, child_name, new_module)
    for name, module in new_module.named_modules():
        if "lora_" in name:
            device = (list(old_module.parameters()) + list(old_module.buffers()))[0].device
            module.to(device)

def create_new_module_peft(
    lora_config: LoraConfig,
    adapter_name: str,
    target: torch.nn.Module,
    target_cls,
    **kwargs,
):
    new_module = None
    if isinstance(target, target_cls):
        new_module = LoraLinearGPTQ(
            target, adapter_name, lora_config=lora_config, **kwargs
        )
    return new_module            


def get_autogptq_peft_model(model, peft_config):
    from auto_gptq.nn_modules.qlinear.qlinear_tritonv2 import QuantLinear
    from auto_gptq.utils.peft_utils import GPTQLoraModel, get_gptq_peft_model

    model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=False,
            gradient_checkpointing_kwargs={},
        )

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

    # undo the patching for hygine
    LoraModel._create_new_module = staticmethod(_old_create_new_module)
    GPTQLoraModel._replace_module = MethodType(_old_replace_module, GPTQLoraModel)
    return model

def get_autogptq_lib_quantized_model(model_name:str, target_modules:List, torch_dtype:str):
    from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
    quantize_config = BaseQuantizeConfig.from_pretrained(model_name)

    device_map = {
        "": (
            torch.cuda.current_device()
            if torch.cuda.is_available()
            else None
        )
    }
    model = AutoGPTQForCausalLM.from_quantized(
        model_name,
        quantize_config=quantize_config,
        torch_dtype=getattr(torch, torch_dtype),
        low_cpu_mem_usage=False,
        use_marlin=False,
        disable_exllama=True,
        warmup_triton=False,
        use_tritonv2=True,
        trainable=True,
        device_map=device_map,
    )

    peft_config = LoraConfig(
        r=LORA_r,
        lora_alpha=LORA_alpha,
        lora_dropout=0.0,  # anyway we are going to override it
        target_modules=target_modules,
    )
    # model = get_autogptq_peft_model(model, peft_config)
    return model
                
def load_autogptq_plugin_model(model_name:str, target_modules:List, torch_dtype:str):
    from fms_acceleration_peft.framework_plugin_autogptq import (
        AutoGPTQAccelerationPlugin,
    )
    plugins = {
        GPTQ: AutoGPTQAccelerationPlugin(
            {
                "peft": {
                    "quantization": {
                        "auto_gptq": {"kernel": "triton_v2", "from_quantized": True}
                    }
                }
            }
        ),
    }

    class TrainArgs:
        gradient_checkpointing = False
        gradient_checkpointing_kwargs = {}

    args = TrainArgs()
    peft_config = LoraConfig(
        r=LORA_r,
        lora_alpha=LORA_alpha,
        lora_dropout=0.0,  # anyway we are going to override it
        target_modules=target_modules,
    )

    _plugin = plugins[GPTQ]
    model = _plugin.model_loader(
        model_name, torch_dtype=getattr(torch, FLOAT16)
    )
    # model, _ = _plugin.augmentation(model, args, (peft_config,))
    return model

@pytest.fixture()
def input_ids(seed: int = 42, device: torch.device = "cuda"):
    torch.manual_seed(seed)
    yield torch.randint(0, 10000, (BS, SEQLEN))    

@pytest.mark.skipif(
    not _is_package_available("auto_gptq"),
    reason="Only runs if auto_gptq is installed",
)
def test_already_quantized_outputs_match(
    input_ids, seed: int = 42,
):
    torch.manual_seed(seed)
    original_model = get_autogptq_lib_quantized_model(QUANTIZED_MODEL_NAME, TARGET_MODULES, FLOAT16)
    refactored_model = load_autogptq_plugin_model(QUANTIZED_MODEL_NAME, TARGET_MODULES, FLOAT16)
    with torch.autocast(device_type='cuda', dtype=torch.float32):
        original_model.eval()
        original_logits = original_model(input_ids.to(original_model.device)).logits
        refactored_model.eval()
        refactored_logits = refactored_model(input_ids.to(refactored_model.device)).logits

    assert torch.allclose(
        original_logits, refactored_logits, atol=ALLCLOSE_ATOL, rtol=ALLCLOSE_RTOL
    ), "Logits don't match between refactored quantized model and original library"


@pytest.mark.skipif(
    not _is_package_available("auto_gptq"),
    reason="Only runs if auto_gptq is installed",
)
def test_pretrained_to_quantized_outputs_match(
    input_ids, seed: int = 42,
):
    torch.manual_seed(seed)
    from transformers import AutoTokenizer
    from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
    from gptqmodel import GPTQModel, QuantizeConfig
    device = torch.device("cuda")

    tokenizer = AutoTokenizer.from_pretrained(VANILLA_MODEL_NAME, use_fast=True)
    calibration_dataset = [
        tokenizer(
            "The world is a wonderful place full of beauty and love."
        )
    ]

    original_quantize_config = BaseQuantizeConfig(
        bits=4,
        group_size=-1,
        desc_act=False,
    )
    # load un-quantized model, by default, the model will always be loaded into CPU memory
    original_model = AutoGPTQForCausalLM.from_pretrained(
        VANILLA_MODEL_NAME, 
        original_quantize_config
    ).to(device)
    # quantize model, the examples should be list of dict whose keys can only be "input_ids" and "attention_mask"
    original_model.quantize(calibration_dataset, use_triton=True)

    refactored_quant_config = QuantizeConfig(
        bits=4,
        group_size=-1,
        desc_act=False,
    )
    # load un-quantized model, by default, the model will always be loaded into CPU memory
    refactored_model = GPTQModel.from_pretrained(VANILLA_MODEL_NAME, refactored_quant_config).to(device)
    # quantize model, the calibration_dataset should be list of dict whose keys can only be "input_ids" and "attention_mask"
    refactored_model.quantize(calibration_dataset)

    with torch.autocast(device_type='cuda', dtype=torch.float16):
        with torch.no_grad():
            original_model.eval()
            original_logits = original_model(input_ids.to(original_model.device)).logits
            refactored_model.eval()
            refactored_logits = refactored_model(input_ids.to(refactored_model.device)).logits

    assert torch.allclose(
        original_logits, refactored_logits, atol=ALLCLOSE_ATOL, rtol=ALLCLOSE_RTOL
    ), "Logits don't match between refactored quantized model and original library"