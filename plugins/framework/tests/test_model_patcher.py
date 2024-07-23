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
    create_dummy_module,
)
from fms_acceleration.utils.test_utils import instantiate_model_patcher

MODULE_A = create_dummy_module("A")
MODULE_SUB_A = create_dummy_module("sub_A", parent_class=MODULE_A)
MODULE_B = create_dummy_module("B")

def returns_false(*args, **kwargs):
    "falsy function"
    return False

def returns_true(*args, **kwargs):
    "truthy function"
    return True

DUMMY_RULE_ID = "test_patch"

# | ------------------ Test ModelPatcherTrigger ----------------------- |

def test_mp_trigger_constructs_with_check_arg_only():
    "Test construction of trigger with check argument"
    # Test that error is raised when check is not of accepted type
    with pytest.raises(
        TypeError,
        match = "check argument needs to be torch.nn.Module or Callable"
    ):
        ModelPatcherTrigger(check=None)

    # Test module trigger type is correctly inferred from check
    trigger = ModelPatcherTrigger(check=torch.nn.Module)
    assert trigger.type == ModelPatcherTriggerType.module

    # Test callable trigger type is correctly inferred from check
    trigger = ModelPatcherTrigger(check=returns_true)
    assert trigger.type == ModelPatcherTriggerType.callable

def test_mp_trigger_constructs_with_check_and_trigger_type_args():
    "Test construction of trigger with check and type arguments"
    # check that trigger constructs successfully as check conforms to specified type
    ModelPatcherTrigger(
        check=torch.nn.Module,
        type=ModelPatcherTriggerType.module,
    )

    ModelPatcherTrigger(
        check=returns_true,
        type=ModelPatcherTriggerType.callable,
    )

    # Ensure an error is raised when check is callable but type is module
    with pytest.raises(
        AssertionError,
        match = "type argument passed but `check` argument does not match type specified",
    ):
        ModelPatcherTrigger(
            check=returns_true,
            type=ModelPatcherTriggerType.module,
        )

    # Ensure an error is raised when check is module but type is callable
    with pytest.raises(
        AssertionError,
        match = "type argument passed but `check` argument does not match type specified",
    ):
        ModelPatcherTrigger(
            check=torch.nn.Module,
            type=ModelPatcherTriggerType.callable,
        )

# def test_mp_trigger_constructs_with_all_specified_args():
#     "Test construction of trigger with check, type and module_name arguments"
#     # check that trigger constructs 
#     ModelPatcherTrigger(
#         check=MODULE_A,
#         type=ModelPatcherTriggerType.module,
#         module_name = MODULE_A
#     )
#     # raises error if module_name is incorrect type
#     with pytest.raises(
#         AssertionError,
#         match = "module_name has to be type `str`"
#     ):
#         ModelPatcherTrigger(
#             check=torch.nn.Module,
#             type=ModelPatcherTriggerType.module,
#             module_name = int
#         )

def test_mp_trigger_correctly_triggers():
    "Test for correctnness of trigger behaviour"

    module_A = create_dummy_module(
        "module_A",
    )

    module_B = create_dummy_module(
        "module_B",
    )

    subclass_A = create_dummy_module(
        "subclass_A",
        parent_class=module_A,
    )

    # Scenario 1:
    # if check is a Callable, is_triggered result must be equal to the boolean output of check
    # 1. create function to check that returns true is is instance of module_A, otherwise return False
    # 2. create trigger that checks using above function
    # 3. create a subclass of module_A and ensure is_triggered returns True
    # 4. create a module_B and ensure is_triggered returns False
    def check_module(module):
        if isinstance(module, module_A):
            return True
        return False

    assert ModelPatcherTrigger(check=check_module).is_triggered(
        subclass_A(),
    ) is True

    assert ModelPatcherTrigger(check=check_module).is_triggered(
        module_B(),
    ) is False

    # Scenario 2:
    # Ensure return True, if is not an instance of ModelPatcherTrigger.check
    # 1. create trigger that checks for module_A
    # 2. create a subclass of module_A and check is_triggered returns True
    # 3. create a module_B and check is_triggered returns False
    assert ModelPatcherTrigger(check=module_A).is_triggered(
        subclass_A(),
    ) is True

    # Ensure returns False, if is not an instance of ModelPatcherTrigger.check
    assert ModelPatcherTrigger(check=module_A).is_triggered(
        module_B(),
    ) is False

    # Scenario 3:
    # Static check to ensure additional constraint is checked
    # 1. create an instance of module_B as model
    # 2. register 2 submodules that inherit from module_B, submodule_1 and submodule_2
    # 2. create a trigger that checks for an instance of module_B and `submodule_1` module name
    # 3. for each module in model, ensure returns true if trigger detects module,
    # otherwise it should return false
    # Create model
    model = module_A()
    # register submodules
    submodule_A = create_dummy_module(
        "submodule_1",
        parent_class=module_B,
    )
    submodule_B = create_dummy_module(
        "submodule_2",
        parent_class=module_B,
    )
    model.add_module("submodule_1", submodule_A())
    model.add_module("submodule_2", submodule_B())
    # create trigger with search criteria
    trigger = ModelPatcherTrigger(check=module_B, module_name="submodule_1")
    # iterate through modules in model
    for name, module in model.named_modules():
        if name == "submodule_1":
            # assert that is_triggered returns true when module is found 
            assert trigger.is_triggered(module, name) is True
        else:
            # assert that is_triggered otherwise returns false 
            assert trigger.is_triggered(module, name) is False


# Each test instance has
#  - target_module,
#  - tuple of trigger check arguments
#  - a logic operator string
#  - expected result as either a boolean or an error tuple
# 1. Instantiate list of triggers from tuple of trigger check arguments
# 2. construct a combined trigger given list of triggers and logic
# 3. if expected_result is a tuple, ensure an error is raised upon constructing the trigger
# 4. Otherwise, ensure that the combined_trigger returns the expected result on the target module
@pytest.mark.parametrize(
    "target_module,trigger_checks,logic,expected_result", [
    [MODULE_SUB_A(), (returns_true, MODULE_B), "OR", True], # True False
    [MODULE_SUB_A(), (MODULE_B, returns_false), "OR", False], # False False
    [MODULE_SUB_A(), (MODULE_A, returns_true), "OR", True], # True True
    [MODULE_SUB_A(), (returns_true, MODULE_B), "AND", False], # True False
    [MODULE_SUB_A(), (MODULE_B, returns_false), "AND", False], # False False
    [MODULE_SUB_A(), (MODULE_A, returns_true), "AND", True], # True True
    [
        MODULE_SUB_A(), (MODULE_B, MODULE_A), "NOR",
        (AssertionError, "Only `AND`, `OR` logic implemented for combining triggers")
    ],
])
def test_correct_output_combine_mp_triggers(target_module, trigger_checks, logic, expected_result):
    triggers = [ModelPatcherTrigger(check=check) for check in trigger_checks]

    # if expected_result is a tuple of (Exception, Exception_message) 
    if isinstance(expected_result, tuple):
        with pytest.raises(
            expected_result[0],
            match=expected_result[1],
        ):
            combine_triggers(
                *triggers,
                logic=logic,
            )
    else: # otherwise ensure is_triggered output returns the expected_result
        assert combine_triggers(
            *triggers,
            logic=logic,
        ).is_triggered(target_module) is expected_result

