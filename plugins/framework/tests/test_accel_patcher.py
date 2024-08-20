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

import pytest

from fms_acceleration.accelerator_patcher import (
    AcceleratorRuleReplace,
    AcceleratorPatcher,
    AcceleratorPatcherComponent,
)

from accelerate import Accelerator

RULE_ID = "test"

def test_AP_rule_raises_correct_errors():
    AcceleratorRuleReplace(
        rule_id = RULE_ID,
        component = AcceleratorPatcherComponent.data_loader,
        replacement = None,
        replacement_builder = None,
    )
    pass

def test_AP_patch_correctly_with_simple_replacement():
    pass

def test_AP_patch_correctly_with_replacement_builder():
    pass