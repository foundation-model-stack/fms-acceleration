import torch
from typing import Dict, Any

def create_dummy_module_with_output_functions(
    class_name: str,
    outputs: Dict[str, Any] = None,
    instantiate = False,
):
    cls = type(class_name, (torch.nn.Module,), outputs)
    if instantiate:
        return cls()
    return cls
