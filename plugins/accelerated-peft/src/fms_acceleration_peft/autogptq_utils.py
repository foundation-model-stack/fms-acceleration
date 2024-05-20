# Copyright The IBM Tuning Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# SPDX-License-Identifier: Apache-2.0
# https://spdx.dev/learn/handling-license-info/

# Third Party
from peft import LoraConfig
from peft.tuners.lora.gptq import QuantLinear as LoraLinearGPTQ
from typing import List, Callable
import torch


def replace_module_peft(self, parent_module, child_name, new_module, old_module):

    # replace the lora linear
    setattr(parent_module, child_name, new_module)

    # dispatch to correct device
    # FIXME: refactor
    for name, module in new_module.named_modules():
        if "lora_" in name:
            device = (list(old_module.parameters()) + list(old_module.buffers()))[
                0
            ].device
            module.to(device)


def create_new_module_peft(
    lora_config: LoraConfig,
    adapter_name: str,
    target: torch.nn.Module,
    target_cls,
    **kwargs,
):
    # if the base layer module matches a supported class, dispatch the lora linear
    # to be installed
    new_module = None
    if isinstance(target, target_cls):
        new_module = LoraLinearGPTQ(
            target, adapter_name, lora_config=lora_config, **kwargs
        )

    # if module cannot be found, return None which results in a raise in the call-stack
    return new_module

# consider to move this somewhere more general
def patch_forward_to_view_attributes_before_call(
    old_forward: Callable,
    attribute_names: List[str], torch_dtype,
):
    # patch old_forward to view attribtues to torch_dype
    # before call

    def _forward(self, *args, **kwargs):
        # perform a view on all these attributes
        for attr_name in attribute_names:

            # the view should be a passthrough 
            # if attr.dtype == torch_dtype
            attr = getattr(self, attr_name)

            # perform view
            attr = attr.view(torch_dtype)

            try:
                setattr(self, attr_name, attr)
            except TypeError:
                # this means already have attr_name as a parameter, then
                # just assign this way
                self.__dict__[attr_name] = attr
        
        return old_forward(*args, **kwargs)
    return _forward
