import torch
import os
import sys
import importlib
from contextlib import contextmanager
from typing import Dict, Any, Type

ROOT = 'tests/model_patcher_fixtures'
PATHS = []
for root, dirs, files in os.walk(ROOT):
    for f in files:
        filename, ext = os.path.splitext(f)
        if ext != ".py":
            continue
        if filename != '__init__':
            PATHS.append(os.path.join(root, f.replace(".py", "")))
        else:
            PATHS.append(root)

@contextmanager
def manipulate_test_module_fixures():
    old_mod = {
        k: sys.modules[k.replace("/", ".")] for k in PATHS
    }
    try:
        yield
    finally:
        # sort keys of descending length, to load children 1st
        sorted_keys = sorted(old_mod.keys(), key=len, reverse=True)
        for key in sorted_keys:
            # Unclear why but needs a reload, to be investigated later
            importlib.reload(old_mod[key])


def create_module_class(
    class_name: str,
    namespaces: Dict[str, Any] = {},
    parent_class: Type = torch.nn.Module
):
    cls = type(class_name, (parent_class,), namespaces)
    return cls