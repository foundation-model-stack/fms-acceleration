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

# First Party
from fms_acceleration.model_patcher import (
    ModelPatcher,
    ModelPatcherRule,
    ModelPatcherTrigger,
    patch_target_module,
)
from fms_acceleration.utils.test_utils import DummyModule

DUMMY_RULE_ID = "test_patch"
DUMMY_HIDDEN_DIM = 32


class DummyCrossEntropyLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, inputs, targets):
        return -1


@pytest.fixture()
def model():
    return DummyModule(DUMMY_HIDDEN_DIM)

@pytest.fixture()
def model_inputs(seed: int = 42):
    torch.manual_seed(seed)
    return torch.rand(1, DUMMY_HIDDEN_DIM)

def test_rule_registration_and_simple_forward_patching(model_inputs, model): # pylint: disable=redefined-outer-name
    "Test model patcher replaces the forward function with a dummy forward"
    # 1. Register rule and specify a trigger on target module for the rule to be applied
    # 2. Patch model
    # 3. check target module's forward function and dummy patch produces similar outputs
    dummy_forward_to_patch = lambda self, X: X * 2 # pylint: disable=unnecessary-lambda-assignment
    ModelPatcher.rules.pop(DUMMY_RULE_ID, None)
    rule = ModelPatcherRule(
        rule_id=DUMMY_RULE_ID,
        trigger=ModelPatcherTrigger(check=DummyModule),
        forward=dummy_forward_to_patch,
    )
    ModelPatcher.register(rule)
    assert DUMMY_RULE_ID in ModelPatcher.rules.keys(), "Rule Registration Failed" # pylint: disable=consider-iterating-dictionary
    ModelPatcher.patch(model)
    assert torch.allclose(
        model(model_inputs), model_inputs * 2
    ), "Failed to patch forward function"


# Test patching of model attribute
def test_patching_downstream_module(model): # pylint: disable=redefined-outer-name
    "Test patching an imported module indirectly managed by other modules using import_and_reload"
    # 1. Register rule targeting downstream module and specify target to reload with patch applied
    # 2. Patch model
    # 3. check patched module now exist in model
    ModelPatcher.rules.pop(DUMMY_RULE_ID, None)

    # Reload because we only want to patch CrossEntropyLoss for this target module
    ModelPatcher.register(
        ModelPatcherRule(
            rule_id=DUMMY_RULE_ID,
            import_and_maybe_reload=(
                "torch.nn.CrossEntropyLoss",
                DummyCrossEntropyLoss,
                "fms_acceleration.utils.test_utils",
            ),
        )
    )
    ModelPatcher.patch(model)
    assert isinstance(
        DummyModule(DUMMY_HIDDEN_DIM).loss_fn, DummyCrossEntropyLoss
    ), "Failed to patch attribute with import and reload"


# Test patching standalone functions
def test_patching_standalone_function(model_inputs): # pylint: disable=redefined-outer-name
    "Test patching of standalone file functions"
    # 1. Take an arbitrary function
    # 2. replace with a dummy function
    # 3. check that the arbitrary function and dummy functions produces similar outputs
    dummy_function_to_patch = lambda X: X # pylint: disable=unnecessary-lambda-assignment
    patch_target_module(
        "fms_acceleration.utils.test_utils.read_configuration",
        dummy_function_to_patch,
    )
    # First Party
    from fms_acceleration.utils.test_utils import read_configuration # pylint: disable=import-outside-toplevel

    assert torch.allclose(
        read_configuration(model_inputs), model_inputs
    ), "Failed to patch standalone function"


def test_forward_patching_with_forward_builder(model_inputs, model): # pylint: disable=redefined-outer-name
    "Test model patcher replaces forward using a dummy forward building function"

    def dummy_forward_builder(module, multiplier):
        # can apply modifications to module here
        setattr(module, "dummy_attribute", True)
        return lambda self, X: X * multiplier

    ModelPatcher.rules.pop(DUMMY_RULE_ID, None)
    ModelPatcher.register(
        ModelPatcherRule(
            rule_id=DUMMY_RULE_ID,
            trigger=ModelPatcherTrigger(check=DummyModule),
            forward_builder=dummy_forward_builder,
            forward_builder_args=["multiplier"],
        )
    )
    ModelPatcher.patch(model, multiplier=4)
    assert hasattr(model, "dummy_attribute") and torch.allclose(
        model(model_inputs), model_inputs * 4
    ), "Failed to patch forward function with forward building feature"
