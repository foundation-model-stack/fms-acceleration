from .module4_1 import mod_4_function
from .module5 import Module5Class, mod_5_function
import torch

class Module4Class(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.attribute = Module5Class()
