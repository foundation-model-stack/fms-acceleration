import torch
from typing import Dict, Any, Type

def create_dummy_module(
    class_name: str,
    namespaces: Dict[str, Any] = {},
    parent_class: Type = torch.nn.Module
):
    cls = type(class_name, (parent_class,), namespaces)
    return cls
