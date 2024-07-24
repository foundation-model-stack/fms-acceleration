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

from .model_patcher_test_utils import create_module_class, isolate_test_module_fixtures
from .model_patcher_fixtures import module1, module2, module4
from fms_acceleration.utils.test_utils import instantiate_model_patcher

from .test_model_patcher import DUMMY_RULE_ID

#Test patching of model attribute
def test_simple_forward_rule_with_mp_replaces_old_forward(): # pylint: disable=redefined-outer-name
    """
    model_patcher_fixtures:
        - module1:
            - module1_1:
                - Module2Class:
                    - attribute: Module2Class
                - mod_1_function
            - module3:
                - module3_1
                    - Module3Class:
                        - attribute: mod_1_function
        - module2:
            - Module2Class:

        - module4:
            - Module4Class(torch.nn.Module):
                - attribute: mod_1_function
    """

    def patched_forward_function(X):
        return "patched_forward_function"

    # 1. Create an instance of Module4Class as model
    # 2. Add a submodule to Module4Class
    # 3. Create and register rule to patch forward of submodule class
    # 4. Patch model
    # 5. Ensure that model's submodule forward is replaced
    with isolate_test_module_fixtures():
        with instantiate_model_patcher():
            model = module4.Module4Class()
            SubModule1 = create_module_class(
                "SubModule1", 
                namespaces={"forward": lambda self: "unpatched_forward_function"}
            )
            model.add_module("submodule_1", SubModule1())
            rule = ModelPatcherRule(
                rule_id=DUMMY_RULE_ID,
                trigger=ModelPatcherTrigger(check=SubModule1),
                forward=patched_forward_function,
            )
            ModelPatcher.register(rule)
            ModelPatcher.patch(model)

            assert model.submodule_1.forward() == "patched_forward_function"

def test_import_and_maybe_reload_rule_with_mp_replaces_old_attribute():
    # 1. Register rule replacing module5.module5_1.Module5Class with a patched_mod_function
    #    reload_target is test.model_patcher.fixtures.module4
    # 2. Patch module4.Module4Class with ModelPatcher
    # 3. check patched module exist in module4.Module4Class.attribute
    PatchedModuleClass = create_module_class(
        "PatchedModClass",
    )


    with isolate_test_module_fixtures():
        with instantiate_model_patcher():
            model = module4.Module4Class()
            ModelPatcher.register(
                ModelPatcherRule(
                    rule_id=DUMMY_RULE_ID,
                    import_and_maybe_reload=(
                        "tests.model_patcher_fixtures.module4.module5.Module5Class",
                        PatchedModuleClass,
                        "tests.model_patcher_fixtures.module4",
                    ),
                )
            )
            ModelPatcher.patch(model)
            assert isinstance(module4.Module4Class().attribute, PatchedModuleClass)

# TODO forward builder test


def test_mp_throws_error_with_multiple_reloads_on_same_target():
    """
    Simulate a case where two rules attempt to reload on the same target prefix

    example:
        - Rule 1 target path 1: x.y.z
        - Rule 2 target path 2: x.y

    this might reverse the patch on Rule 1 and needs to be caught

    model_patcher_fixtures:
        - module1:
            - module1_1:
                - Module2Class:
                    - attribute: Module2Class
                - mod_1_function
            - module3:
                - module3_1
                    - Module3Class:
                        - attribute: mod_1_function
        - module2:
            - Module2Class:

        - module4:
            - Module4Class(torch.nn.Module):
                - attribute: mod_1_function
            - module4_1
                - mod_4_function
            - module5:
                - module5_1
                    - Module5Class
                    - module_5_function

    """

    PatchedModuleClass = create_module_class(
        "PatchedModuleClass",
    )

    def patched_mod_function():
        return "patched_mod_function"

    # Demonstrate that the 2nd patch overwrites the 1st patch if the reload module paths are the same
    with isolate_test_module_fixtures():
        # 1st patch on a function
        patch_target_module(
            "tests.model_patcher_fixtures.module4.module5.module5_1.mod_5_function",
            patched_mod_function,
            "tests.model_patcher_fixtures.module4.module5",
        )

        assert module4.module5.mod_5_function() == "patched_mod_function"

        # 2nd patch on a class that has a target path that reloads module5 as well
        patch_target_module(
            "tests.model_patcher_fixtures.module4.module5.module5_1.Module5Class",
            PatchedModuleClass,
            "tests.model_patcher_fixtures.module4.module5"
        )

        assert isinstance(module4.module5.Module5Class(), PatchedModuleClass)
        assert module4.module5.mod_5_function() == "unpatched_mod_function"

    # Ensure that an assertion is raised if target paths share the same root path
    with pytest.raises(
        AssertionError,
    ):
        with isolate_test_module_fixtures():
            with instantiate_model_patcher():
                # 1. Initialize a model with module path tests.model_patcher_fixtures.module4
                model = module4.Module4Class()

                # 2. Simulate patching a function in module4.module5.module5_1
                ModelPatcher.register(
                    ModelPatcherRule(
                        rule_id=f"{DUMMY_RULE_ID}.2",
                        import_and_maybe_reload=(
                            "tests.model_patcher_fixtures.module4.module5.module5_1.mod_5_function",
                            patched_mod_function,
                            "tests.model_patcher_fixtures.module4.module5.module5_1",
                        ),
                    )
                )

                # 3. Simulate patching a class in module4.module5.module5_1
                ModelPatcher.register(
                    ModelPatcherRule(
                        rule_id=f"{DUMMY_RULE_ID}.1",
                        import_and_maybe_reload=(
                            "tests.model_patcher_fixtures.module4.module5.module5_1.Module5Class",
                            PatchedModuleClass,
                            "tests.model_patcher_fixtures.module4",
                        ),
                    )
                )

                # while ModelPatcher is patching different objects, repeated reloads on same path is risky
                # since module4 is a parent of module5, reloading module4 again might affect the previous patch.
                # To prevent this we throw an exception if the shorter target path is a prefix of the
                # longer target path
                ModelPatcher.patch(model)


def test_mp_throws_warning_with_multiple_patches():
    """
    Ensure for each module, only one forward patch is implemented on it.
    The patch implementation checks if there are multiple forward patch rules that are applied to the module,
    only the 1st forward patch rule is applied, the others will be ignored and a warning will be raised

    In the case of a list of new rules generated by `forwardbuilder`, it will be handled similarly since
    it decomposes to multiple single forward patch rules downstream.
    """
    with pytest.warns(
        UserWarning,
    ):
        with isolate_test_module_fixtures():
            with instantiate_model_patcher():
                # 1. Create a model
                # 2. Create a submodule to patch on
                # 3. Create 1st rule to patch submodule forward function
                # 4. Create 2nd rule to patch submodule forward function again
                # 5. Throws warning that any subsequent forward patches after the 1st patch is ignored

                model = module4.Module4Class()
                SubModule1 = create_module_class(
                    "SubModule1", 
                    namespaces={"forward": lambda self: "unpatched_forward_function"}
                )
                model.add_module("submodule_1", SubModule1())

                ModelPatcher.register(
                    ModelPatcherRule(
                        rule_id=DUMMY_RULE_ID+".1",
                        trigger=ModelPatcherTrigger(check=SubModule1),
                        forward=lambda self: "patched_forward_function",
                    )
                )
                ModelPatcher.register(
                    ModelPatcherRule(
                        rule_id=DUMMY_RULE_ID+".2",
                        trigger=ModelPatcherTrigger(check=SubModule1),
                        forward=lambda self: "patched_forward_function_2",
                    )
                )
                ModelPatcher.patch(model)

    # TODO test on forward builder cases
