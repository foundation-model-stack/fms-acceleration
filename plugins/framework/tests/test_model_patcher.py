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
    ModelPatcherHistory,
    combine_functions,
    combine_triggers,
)
from .model_patcher_test_utils import (    
    create_dummy_module_with_output_functions,
)
from fms_acceleration.utils.test_utils import instantiate_model_patcher

def returns_false(*args, **kwargs):
    "falsy function"
    return False

def returns_true(*args, **kwargs):
    "truthy function"
    return True

DUMMY_RULE_ID = "test_patch"

@pytest.fixture()
def model_inputs(seed: int = 42):
    torch.manual_seed(seed)
    return torch.rand(1, 32)

# | ------------------ Test ModelPatcherHistory ----------------------- |
def test_mp_history_constructs_successfully():
    "Test that model patcher trigger constructs correctly"
    model = torch.nn.Module()
    assert ModelPatcherHistory(
        instance=id(model),
        cls=model.__class__.__name__,
        parent_cls="",
        module_name="",
        parent_module_name="",
        rule_id=DUMMY_RULE_ID,
    )

# | ------------------ Test ModelPatcherTrigger ----------------------- |

def test_mp_trigger_constructs_with_check_arg_only():
    "Test construction of trigger with check argument"
    # Test that error is raised when check is not of accepted type
    with pytest.raises(
        AssertionError,
        match = "`check` arg type needs to be torch.nn.Module or Callable"
    ):
        ModelPatcherTrigger(check=None)

    # Test module trigger type is correctly inferred from check
    trigger = ModelPatcherTrigger(check=torch.nn.Module)
    assert trigger and trigger.type == ModelPatcherTriggerType.module, \
        "Trigger Construction with Module Failed"

    # Test callable trigger type is correctly inferred from check
    trigger = ModelPatcherTrigger(check=returns_true)
    assert trigger and trigger.type == ModelPatcherTriggerType.callable, \
        "Trigger Construction with Callable Failed"

def test_mp_trigger_constructs_with_check_and_trigger_type_args():
    "Test construction of trigger with check and type arguments"
    # check that trigger constructs successfully as check conforms to specified type
    assert ModelPatcherTrigger(
        check=torch.nn.Module,
        type=ModelPatcherTriggerType.module,
    )

    assert ModelPatcherTrigger(
        check=returns_true,
        type=ModelPatcherTriggerType.callable,
    )

    # Ensure an error is raised when check does not conform to type
    with pytest.raises(
        AssertionError,
        match = "type argument passed but `check` argument does not match type specified",
    ):
        ModelPatcherTrigger(
            check=returns_true,
            type=ModelPatcherTriggerType.module,
        )

    # Ensure an error is raised when check does not conform to type
    with pytest.raises(
        AssertionError,
        match = "type argument passed but `check` argument does not match type specified",
    ):
        ModelPatcherTrigger(
            check=torch.nn.Module,
            type=ModelPatcherTriggerType.callable,
        )

    # Ensure error is raised when improper trigger type is passed
    with pytest.raises(
        NotImplementedError,
        match = "Invalid ModelPatcherTriggerType",
    ):
        ModelPatcherTrigger(
            check=torch.nn.Module,
            type=int,
        )

def test_mp_trigger_constructs_with_all_specified_args():
    "Test construction of trigger with check, type and module_name arguments"
    # check that trigger constructs
    assert ModelPatcherTrigger(
        check=torch.nn.Module,
        type=ModelPatcherTriggerType.module,
        module_name = "UnpatchedSubmodule"
    )
    assert ModelPatcherTrigger(
        check= returns_true,
        type=ModelPatcherTriggerType.callable,
        module_name = "UnpatchedSubmodule"
    )

def test_mp_trigger_returns_correct_response():
    "Test for correctnness of trigger behaviour"

    # Scenario 1: 
    # if check is a Callable, is_triggered result must be equal to the boolean output of check
    assert ModelPatcherTrigger(check=returns_true).is_triggered(
        torch.nn.Module(),
    ) == returns_true()

    assert ModelPatcherTrigger(check=returns_false).is_triggered(
        torch.nn.Module(),
    ) == returns_false()

    # Scenario 2:
    # Ensure return True, if the module passed in `is_triggered` is an instance
    # of ModelPatcherTrigger.check
    assert ModelPatcherTrigger(check=torch.nn.Module).is_triggered(
        torch.nn.Module(),
    ) is True

    # Ensure returns False, if the module passed in `is_triggered` is not an instance
    # of ModelPatcherTrigger.check
    assert ModelPatcherTrigger(check=torch.nn.Module).is_triggered(
        returns_true,
    ) is False

    # Scenario 3:
    # Static check to ensure additional module_name constraint is checked when
    # ModelPatcherTrigger.module_name is specified
    trigger = ModelPatcherTrigger(check=torch.nn.Module, module_name="dummy.module.class")
    # ensure returns false if additional constraint fails
    assert trigger.is_triggered(torch.nn.Module(), "wrong.module.class") is False
    # ensure returns `ModelPatcherTrigger.check` if additional constraint passes
    assert trigger.is_triggered(torch.nn.Module(), "dummy.module.class") is True

def test_correct_output_combine_mp_triggers():
    # test OR case should pass if one trigger is true
    combined_trigger = combine_triggers(
        ModelPatcherTrigger(check=returns_false),
        ModelPatcherTrigger(check=returns_true),
        logic = "OR",
    )
    assert combined_trigger.is_triggered(torch.nn.Module()) is True, \
        "OR logic test should pass if there is a single True"
    combined_trigger = combine_triggers(
        ModelPatcherTrigger(check=returns_false),
        ModelPatcherTrigger(check=returns_false),
        logic = "OR",
    )
    assert combined_trigger.is_triggered(torch.nn.Module()) is False, \
        "OR logic test should fail if there is no True trigger"

    # test AND case should fail if one trigger is false
    combined_trigger = combine_triggers(
            ModelPatcherTrigger(check=returns_false),
            ModelPatcherTrigger(check=returns_true),
            logic = "AND",
        )
    assert combined_trigger.is_triggered(torch.nn.Module()) is False, \
        "AND logic test should fail if there is a single False"
    combined_trigger = combine_triggers(
            ModelPatcherTrigger(check=returns_true),
            ModelPatcherTrigger(check=returns_true),
            logic = "AND",
        )
    assert combined_trigger.is_triggered(torch.nn.Module()) is True, \
        "AND logic test should pass if there is no False trigger"

    with pytest.raises(
        AssertionError,
        match = "Only `AND`, `OR` logic implemented for combining triggers",
    ):
        combine_triggers(
            ModelPatcherTrigger(check=returns_false),
            ModelPatcherTrigger(check=returns_true),
            logic = "NOR",
        )
