import torch

UNPATCHED_RESPONSE = 0
PATCHED_RESPONSE = 1

class DummyAttribute(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, *args, **kwargs):
        return UNPATCHED_RESPONSE

class PatchedAttribute(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, *args, **kwargs):
        return PATCHED_RESPONSE
