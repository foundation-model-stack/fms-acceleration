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
    ModelPatcherTriggerType,
)
from .model_patcher_test_utils import (
    PatchedAttribute,
    DummyAttribute,
    PATCHED_RESPONSE,
    UNPATCHED_RESPONSE,
)
from fms_acceleration.utils.test_utils import instantiate_model_patcher, read_configuration

class DummyModule(torch.nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.dummy_attribute = DummyAttribute()

    def forward(self, *args, **kwargs):
        return UNPATCHED_RESPONSE

DUMMY_RULE_ID = "test_patch"

@pytest.fixture()
def model_inputs(seed: int = 42):
    torch.manual_seed(seed)
    return torch.rand(1, 32)

# | ------------------ Test ModelPatcherTrigger ----------------------- |

def test_mp_trigger_constructs_successfully():
    "Test that model patcher trigger constructs correctly"
    # Test Module Argument Construction
    trigger = ModelPatcherTrigger(check=DummyAttribute)
    assert trigger and trigger.type == ModelPatcherTriggerType.module, \
        "Trigger Construction with Module Failed"

    # Test Callable Argument Constructionn
    trigger = ModelPatcherTrigger(check=lambda module: True)
    assert trigger and trigger.type == ModelPatcherTriggerType.callable, \
        "Trigger Construction with Callable Failed"

def test_mp_trigger_returns_correct_response():
    "Test for correctnness of trigger behaviour"

    # Scenario 1: MP Trigger must follow callable response output
    for result in [True, False]:
        assert ModelPatcherTrigger(check=lambda module: result).is_triggered(
            DummyAttribute(),
            DummyAttribute.__name__
        ) == result, "Trigger output doesn't match with Callable output"

    # Scenario 2: MP Trigger checks module correctly
    trigger = ModelPatcherTrigger(check=DummyAttribute)
    # test that it returns false if not an instance
    for mod, result in [(DummyAttribute, True), (PatchedAttribute, False)]:
        assert trigger.is_triggered(mod(), mod.__name__) == result, \
        "Trigger output must be instance of check argument"

    # Scenario 3: MP Trigger checks for module name correctly
    for _check in [DummyAttribute, lambda module: True]:
        # set same `module_name` in Trigger regardless of check argument
        module_name = ModelPatcherTrigger(check=_check, module_name=DummyAttribute.__name__)
        # ensure that output is false as long as module name doesn't match
        assert module_name.is_triggered(DummyAttribute(), PatchedAttribute.__name__) == False, \
            "`Trigger.module_name` exist, module name passed should not match"

# | ------------------ Test ModelPatcherRule ----------------------- |

# Test patching standalone functions
def test_standalone_import_and_reload_function_replaces_indirect_module(model_inputs): # pylint: disable=redefined-outer-name
    "Test patching of standalone file functions"
    # 1. Take an arbitrary downstream module
    # 2. replace with a patched module
    # 3. check that the downstream module produces a patched response
    patch_target_module(
        "tests.model_patcher_test_utils.DummyAttribute",
        PatchedAttribute,
        "tests.test_model_patcher"
    )
    # First Party
    assert DummyModule().dummy_attribute(model_inputs) == PATCHED_RESPONSE, "Failed to patch standalone function"

def test_mp_registers_only_one_unique_rule():
    "Test MP register method raises assertion when 2 rules with same id are registered"
    with pytest.raises(
        AssertionError
    ):   
        with instantiate_model_patcher():
            for i in range(2):
                ModelPatcher.register(
                    ModelPatcherRule(rule_id=DUMMY_RULE_ID)
                )

def test_mp_rule_constructs_successfully():
    "Ensure MP rule is throws appropriate error when wrong argument combinations are passed"
    # Test empty mp rule construction
    assert ModelPatcherRule(rule_id=DUMMY_RULE_ID), "Empty MP Rule construction failed"

    # Test mp rule construction raises with multiple arguments
    with pytest.raises(
        ValueError, 
        match="must only have only one of forward, " \
        "foward builder, or import_and_maybe_reload, specified."
    ):
        ModelPatcherRule(
            rule_id=DUMMY_RULE_ID,
            forward=lambda self, X: X,
            import_and_maybe_reload=(),
            forward_builder=lambda self, X: X,
        )

    # Test mp rule construction raises with trigger and import_and_reload
    with pytest.raises(
        ValueError, 
        match="has import_and_maybe_reload specified, " \
        "and trigger must be None."
    ):
        ModelPatcherRule(
            rule_id=DUMMY_RULE_ID,
            trigger=ModelPatcherTrigger(check=None),
            import_and_maybe_reload=(),
        )

    with pytest.raises(
        ValueError, 
        match="has forward_builder_args but no " \
        "forward_builder."
    ):
        ModelPatcherRule(
            rule_id=DUMMY_RULE_ID,
            forward_builder_args=[]
        )

def test_mp_rule_patches_forward(model_inputs): # pylint: disable=redefined-outer-name
    "Test model patcher replaces the forward function with a dummy forward"
    # 1. Register rule and specify a trigger on target module for the rule to be applied
    # 2. Patch model
    # 3. check target module's forward function and dummy patch produces similar outputs
    with instantiate_model_patcher():
        model = DummyModule()
        rule = ModelPatcherRule(
            rule_id=DUMMY_RULE_ID,
            trigger=ModelPatcherTrigger(check=DummyModule),
            forward=lambda self, X: PATCHED_RESPONSE,
        )
        ModelPatcher.register(rule)
        ModelPatcher.patch(model)
        assert model(model_inputs) == PATCHED_RESPONSE, "Failed to patch forward function"

# Test patching of model attribute
def test_mp_rule_import_and_reload_patches_downstream_module(): # pylint: disable=redefined-outer-name
    "Test patching an imported module indirectly managed by other modules using import_and_reload"
    # 1. Register rule targeting downstream module and specify target to reload with patch applied
    # 2. Patch model
    # 3. check patched module now exist in model
    with instantiate_model_patcher():
        model = DummyModule()
        ModelPatcher.register(
            ModelPatcherRule(
                rule_id=DUMMY_RULE_ID,
                import_and_maybe_reload=(
                    "tests.model_patcher_test_utils.DummyAttribute",
                    PatchedAttribute,
                    "tests.test_model_patcher",
                ),
            )
        )
        ModelPatcher.patch(model)

        assert isinstance(
            DummyModule().dummy_attribute, PatchedAttribute
        ), "Failed to patch attribute with import and reload"

    # Test that assertion thrown when multiple rules try to reload the same target
    with pytest.raises(
        AssertionError,
        match="can only have at most one rule with reload"
    ):
        with instantiate_model_patcher():
            model = DummyModule()
            for i in range(2):
                ModelPatcher.register(
                    ModelPatcherRule(
                        rule_id=DUMMY_RULE_ID+str(i),
                        import_and_maybe_reload=(
                            "tests.model_patcher_test_utils.DummyAttribute",
                            PatchedAttribute,
                            "tests.test_model_patcher",
                        ),
                    )
                )
            ModelPatcher.patch(model)

            assert isinstance(
                DummyModule().dummy_attribute, PatchedAttribute
            ), "Failed to patch attribute with import and reload"


def test_mp_rule_patches_forward_with_builder_and_args(model_inputs): # pylint: disable=redefined-outer-name
    "Test model patcher replaces forward using forward builder"

    dummy_builder_args = {
        "dummy_arg_1": 1,
        "dummy_arg_2": 2,
        "dummy_arg_3": 3,
    }

    spy = {
        "forward_builder_calls": 0
    }

    model = DummyModule()
    def dummy_forward_builder(module, **kwargs):
        assert kwargs == dummy_builder_args, "Builder arguments not passed to forward builder"
        # spy on function here
        spy["forward_builder_calls"] += 1
        return lambda self, X: PATCHED_RESPONSE

    with instantiate_model_patcher():
        ModelPatcher.register(
            ModelPatcherRule(
                rule_id=DUMMY_RULE_ID,
                trigger=ModelPatcherTrigger(check=DummyModule),
                forward_builder=dummy_forward_builder,
                forward_builder_args=[*dummy_builder_args.keys()],
            )
        )
        ModelPatcher.patch(model, **dummy_builder_args)
        assert spy["forward_builder_calls"] > 0 and model(model_inputs) == PATCHED_RESPONSE, \
            "Failed to patch forward function with forward building feature"

# |----------------------- MP Summary ------------------- |

def test_mp_summary_prints_log_message_of_patches_applied():
    "test summary function prints"
    pass



# |----------------------- MP load_patches ------------------- |

def test_load_patches_import_libraries():
    "Test the function that imports packages."
    "Used in MP when package imports trigger MP Rule registration"
    ModelPatcher.load_patches([])
    pass
