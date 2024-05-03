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

# Standard
from contextlib import contextmanager
from tempfile import NamedTemporaryFile
from typing import Callable, Dict, List, Set, Tuple, Type

# Third Party
import pytest
import torch
import yaml

# First Party
from fms_acceleration.framework import KEY_PLUGINS, AccelerationFramework
from fms_acceleration.framework_plugin import PLUGIN_REGISTRATIONS, AccelerationPlugin

# ----------------------------- HELPER -------------------------------------


@contextmanager
def build_framework_and_instantiate(
    configuration_contents: Dict,
    plugins_to_be_registered: List[
        Tuple[List[str], Type[AccelerationPlugin]]  # and_paths, plugin_class
    ],
):
    "helper function to instantiate an acceleration framework for testing"

    # empty out
    old_registrations = []
    old_registrations.extend(PLUGIN_REGISTRATIONS)
    PLUGIN_REGISTRATIONS.clear()
    old_active_plugins = AccelerationFramework.active_plugins
    old_custom_loading_plugins = AccelerationFramework.plugins_require_custom_loading
    AccelerationFramework.active_plugins = {}
    AccelerationFramework.plugins_require_custom_loading = []

    for path, plugin in plugins_to_be_registered:
        AccelerationPlugin.register_plugin(
            plugin,
            configuration_and_paths=path,
        )

    with NamedTemporaryFile("w") as f:
        yaml.dump({KEY_PLUGINS: configuration_contents}, f)
        yield AccelerationFramework(f.name)

    # put back
    PLUGIN_REGISTRATIONS.clear()
    PLUGIN_REGISTRATIONS.extend(old_registrations)
    AccelerationFramework.active_plugins = old_active_plugins
    AccelerationFramework.plugins_require_custom_loading = old_custom_loading_plugins


def create_noop_model_with_archs(class_name: str = "ModelNoop", archs: List[str] = []):
    "helper function to create a dummy model with mocked architectures"

    config = type("Config", (object,), {"architectures": archs})
    return type(class_name, (torch.nn.Module,), {"config": config})


def create_plugin_cls(
    class_name: str = "PluginNoop",
    restricted_models: Set = {},
    require_pkgs: Set = {},
    requires_custom_loading: bool = False,
    requires_agumentation: bool = False,
    agumentation: Callable = None,
    model_loader: Callable = None,
):
    "helper function to create plugin class"

    attributes = {
        "restricted_model_archs": restricted_models,
        "require_packages": require_pkgs,
        "requires_custom_loading": requires_custom_loading,
        "requires_agumentation": requires_agumentation,
    }

    if agumentation is not None:
        attributes["augmentation"] = agumentation

    if model_loader is not None:
        attributes["model_loader"] = model_loader

    return type(class_name, (AccelerationPlugin,), attributes)


def dummy_augmentation(self, model, train_args, modifiable_args):
    "dummy augmentation implementation"
    return model, modifiable_args


def dummy_custom_loader(self, model_name, **kwargs):
    "dummy custom loader returning dummy model"
    return create_noop_model_with_archs(archs=["DummyModel"])


# ----------------------------- TESTS -------------------------------------


def test_config_with_empty_body_raises():
    "test that configuration path with empty body will raise"

    # create empty plugin, since plugin not relevant for these tests
    empty_plugin = create_plugin_cls()

    # register plugin, and try to activate with config with empty body
    # - raise since activation will fail
    with pytest.raises(ValueError) as e:
        with build_framework_and_instantiate(
            configuration_contents={"dummy": None},
            plugins_to_be_registered=[(["dummy"], empty_plugin)],
        ):
            pass

    e.match(
        "No plugins could be configured. Please check the acceleration framework configuration file"
    )

    # register plugin to activate if two paths are specified
    # - raise because one path has an empty body
    with pytest.raises(ValueError) as e:
        with build_framework_and_instantiate(
            configuration_contents={"dummy": None, "dummy2": {"key1": 1}},
            plugins_to_be_registered=[(["dummy", "dummy2"], empty_plugin)],
        ):
            pass

    e.match(
        "No plugins could be configured. Please check the acceleration framework configuration file"
    )


def test_model_with_no_config_raises():
    "augmenting a model with no model.config will raise"

    # create model and (incomplete) plugin with requires_augmentation = True
    model_no_config = torch.nn.Module()  # empty model
    incomplete_plugin = create_plugin_cls(requires_agumentation=True)

    # register and activate 1 incomplete plugin, and:
    # 1. test correct plugin registration and activation.
    # 2. raise because no model_loader implementation but model_loader is called.
    # 3. raise because model does not have config attribute
    with build_framework_and_instantiate(
        configuration_contents={"dummy": {"key1": 1}},
        plugins_to_be_registered=[(["dummy"], incomplete_plugin)],
    ) as framework:
        # check 1.
        assert len(PLUGIN_REGISTRATIONS) == 1
        assert len(framework.active_plugins) == 1
        assert len(framework.plugins_require_custom_loading) == 0

        # check 2.
        with pytest.raises(NotImplementedError) as e:
            framework.model_loader(None)
        e.match("Attempted model loading, but none of activated plugins")

        # check 3.
        with pytest.raises(AttributeError):
            framework.augmentation(model_no_config, None, None)


def test_single_plugin():
    "test various cases given a single registered plugin"

    # create models
    model = create_noop_model_with_archs(archs=["CausalLM"])
    model_diff_arch = create_noop_model_with_archs(archs=["CNN"])

    # create plugins
    empty_plugin = create_plugin_cls()
    incomplete_plugin = create_plugin_cls(
        restricted_models={"CausalLM"},
        requires_agumentation=True,
    )
    plugin = create_plugin_cls(
        restricted_models={"CausalLM"},
        requires_agumentation=True,
        requires_custom_loading=True,
        agumentation=dummy_augmentation,
        model_loader=dummy_custom_loader,
    )
    train_args = None  # dummy for now

    # register and activate 1 incomplete plugin, and:
    # 1. test correct plugin registration and activation.
    # 2. raise because no model_loader implementation but model_loader is called.
    # 3. raise because requires_augmentation = True but no augmentation impl.
    # 4. raise when called on model with incompatible arch.
    with build_framework_and_instantiate(
        configuration_contents={"dummy": {"key1": 1}},
        plugins_to_be_registered=[(["dummy"], incomplete_plugin)],
    ) as framework:

        # check 1.
        assert len(PLUGIN_REGISTRATIONS) == 1
        assert len(framework.active_plugins) == 1
        assert len(framework.plugins_require_custom_loading) == 0

        # check 2.
        with pytest.raises(NotImplementedError) as e:
            framework.model_loader(None)
        e.match("Attempted model loading, but none of activated plugins")

        # check 3.
        with pytest.raises(NotImplementedError):
            framework.augmentation(model, None, None)  # because augmentation not impl

        # check 4.
        with pytest.raises(ValueError) as e:
            framework.augmentation(model_diff_arch, None, None)
        e.match("Model architectures in")  # not supported

    # register 1 empty plugin, and try to activate it with config that has a wrong path
    # - raise because wrong path results in zero active plugins
    with pytest.raises(ValueError) as e:
        with build_framework_and_instantiate(
            configuration_contents={"dummy2": {"key1": 1}},  # config with wrong path
            plugins_to_be_registered=[(["dummy"], empty_plugin)],
        ):
            pass

    e.match("No plugins could be configured. Please check the acceleration")

    # register and activate 1 plugin, and:
    # 1. test correct plugin registration and activation.
    # 2. test augmentation should run
    with build_framework_and_instantiate(
        configuration_contents={"dummy": {"key1": 1}},
        plugins_to_be_registered=[(["dummy"], plugin)],
    ) as framework:
        # check 1.
        assert len(PLUGIN_REGISTRATIONS) == 1
        assert len(framework.active_plugins) == 1
        assert len(framework.plugins_require_custom_loading) == 1

        # check 2.
        framework.augmentation(model, train_args, (None,))


def test_two_plugins():
    "test various cases given two registered plugins"

    model = create_noop_model_with_archs(archs=["CausalLM"])
    incomp_plugin1 = create_plugin_cls(
        restricted_models={"CausalLM"}, requires_agumentation=True
    )
    incomp_plugin2 = create_plugin_cls(requires_agumentation=True)
    incomp_plugin3 = create_plugin_cls(
        class_name="PluginNoop2", requires_agumentation=True
    )
    plugin1 = create_plugin_cls(
        restricted_models={"CausalLM"},
        requires_agumentation=True,
        requires_custom_loading=True,
        agumentation=dummy_augmentation,
        model_loader=dummy_custom_loader,
    )
    plugin2 = create_plugin_cls(
        class_name="PluginNoop2",
        restricted_models={"CausalLM"},
        requires_agumentation=True,
        requires_custom_loading=True,
        agumentation=dummy_augmentation,
        model_loader=dummy_custom_loader,
    )
    plugin3_no_loader = create_plugin_cls(
        class_name="PluginNoop2",
        restricted_models={"CausalLM"},
        requires_agumentation=True,
        agumentation=dummy_augmentation,
    )
    train_args = None  # dummy for now

    # register and activate 2 incomplete plugins, both of the same class
    # 1. test that two plugins will be registered, because not checking for duplications
    #    during registration.
    # 2. test that only 1 plugin will be activated, because duplication is checked here.
    with build_framework_and_instantiate(
        configuration_contents={"dummy": {"key1": 1}, "dummy2": {"key1": 1}},
        plugins_to_be_registered=[
            (["dummy"], incomp_plugin1),
            (["dummy2"], incomp_plugin2),
        ],
    ) as framework:

        # check 1.
        assert len(PLUGIN_REGISTRATIONS) == 2

        # check 2.
        assert len(framework.active_plugins) == 1  # because both plugins of same class

    # register and activate 2 incomplete plugins, both of the different class
    # 1. test that two plugins are registered and activated
    with build_framework_and_instantiate(
        configuration_contents={"dummy": {"key1": 1}, "dummy2": {"key1": 1}},
        plugins_to_be_registered=[
            (["dummy"], incomp_plugin1),
            (["dummy2"], incomp_plugin3),
        ],
    ) as framework:
        # check 1.
        assert len(PLUGIN_REGISTRATIONS) == 2
        assert len(framework.active_plugins) == 2  # because both are of different class

    # register and activate 2 plugins, both of the different class and both have model loading
    # 1. raise because cannot activate two plugins with model loading
    with pytest.raises(AssertionError) as e:
        with build_framework_and_instantiate(
            configuration_contents={"dummy": {"key1": 1}, "dummy2": {"key1": 1}},
            plugins_to_be_registered=[(["dummy"], plugin1), (["dummy2"], plugin2)],
        ) as framework:
            pass

    e.match("Can load at most 1 plugin with custom model loading, but tried")

    # register and activate 2 plugins, both of the different class
    # 1. test correct plugin registration and activation.
    # 2. test model loading works.
    # 3. test model augmentation works.
    with build_framework_and_instantiate(
        configuration_contents={"dummy": {"key1": 1}, "dummy2": {"key1": 1}},
        plugins_to_be_registered=[
            (["dummy"], plugin1),
            (["dummy2"], plugin3_no_loader),
        ],
    ) as framework:
        # check 1.
        assert len(PLUGIN_REGISTRATIONS) == 2
        assert len(framework.active_plugins) == 2  # because both are of different class

        # check 2
        framework.model_loader(None)

        # check 3
        framework.augmentation(model, train_args, None)
