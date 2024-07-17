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
import pytest  # pylint: disable=(import-error
import torch
from fms_acceleration.model_patcher import ModelPatcher, ModelPatcherRule, ModelPatcherTrigger

DUMMY_RULE_ID = "test_patch"
DUMMY_HIDDEN_DIM = 32

class DummyModule(torch.nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.linear = torch.nn.Linear(DUMMY_HIDDEN_DIM, DUMMY_HIDDEN_DIM)

    def forward(self, X):
        return self.linear(X)

@pytest.fixture()
def model_inputs(seed: int = 42):
    torch.manual_seed(seed)
    inputs = torch.rand(1, DUMMY_HIDDEN_DIM)
    return inputs

def test_rule_registration_and_model_patching(model_inputs):
    "Test model patcher registers rule and patches correctly"
    dummy_forward_to_patch = lambda self, X: X*2
    rule = ModelPatcherRule(
        rule_id=DUMMY_RULE_ID,
        trigger=ModelPatcherTrigger(check=torch.nn.Linear),
        forward=dummy_forward_to_patch,
    )
    ModelPatcher.register(rule)
    assert DUMMY_RULE_ID in ModelPatcher.rules.keys(), "Rule Registration Failed"

    model = DummyModule()
    ModelPatcher.patch(model)

    assert torch.allclose(model(model_inputs), model_inputs*2), "Failed to patch foward function "



