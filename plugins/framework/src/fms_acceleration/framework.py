# Copyright The FMS HF Tuning Authors
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

# Standard
from typing import Callable, Dict, List, Optional, Set, Tuple

# Third Party
from accelerate import Accelerator
from transformers import PreTrainedModel, TrainingArguments
from transformers.utils import logging
from transformers.utils.import_utils import _is_package_available
import torch
import yaml

# want to use the transformers logger, but a bit of pain
logger = logging.get_logger(__name__)  # pylint: disable=invalid-name
logger.setLevel(logging._get_default_logging_level())
logger.addHandler(logging._default_handler)

# First Party
from .framework_plugin import (
    PLUGIN_REGISTRATIONS,
    AccelerationPlugin,
    PluginRegistration,
    get_relevant_configuration_sections,
)
from .constants import KEY_PLUGINS

def check_plugin_packages(plugin: AccelerationPlugin):
    if plugin.require_packages is None:
        return True, []  # passthrough

    missing_packages = []
    for package_name in plugin.require_packages:
        if not _is_package_available(package_name):
            missing_packages.append(package_name)
    return len(missing_packages) == 0, missing_packages

def log_initialization_message(
    active_class_names: Set[str],
    registered_plugins: List[PluginRegistration],  # list of regs
    logger: Callable = None,
):
    if logger is None:
        logger = print

    def _registration_display(reg: PluginRegistration):
        return (
            f"Active Plugin: {reg.plugin.__name__}. "
            f"Python package: {reg.package_name}. "
            f"Version: {reg.package_version}."
        )

    logger("***** FMS AccelerationFramework *****")
    for reg in registered_plugins:
        if reg.plugin.__name__ in active_class_names:
            logger(_registration_display(reg))


class AccelerationFramework:

    active_plugins: List[Tuple[str, AccelerationPlugin]] = list()
    plugins_require_custom_loading: List = list()

    def __init__(
        self, configuration_file: Optional[str], require_packages_check: bool = True
    ):

        with open(configuration_file, "r") as f:
            contents = yaml.safe_load(f)

        if KEY_PLUGINS not in contents or contents[KEY_PLUGINS] is None:
            raise ValueError(f"Configuration file must contain a '{KEY_PLUGINS}' body")

        # pepare the plugin configurations
        plugin_configs = {k: v for k, v in contents[KEY_PLUGINS].items()}

        # relevant sections are returned following plugin precedence, i.e.,
        # they follow the registration order.
        for selected_configs, cls in get_relevant_configuration_sections(
            plugin_configs
        ):

            # then the model is to be installed
            # get the plugin
            plugin_name = str(cls.__name__)
            plugin = cls(selected_configs)

            # check plugin
            has_packages, missing_packages = check_plugin_packages(plugin)
            if not has_packages and require_packages_check:
                missing_packages = ", ".join(missing_packages)
                raise ValueError(
                    f"Packages '{missing_packages}' required by activated plugin '{plugin_name}' "
                    "is missing. Please install it."
                )

            # check if already activated, if so, will not reactivate again
            # maintain uniqueness of activated plugins
            if any([x == plugin_name for x, _ in self.active_plugins]):
                continue

            # activate plugin
            # - activation order will not contradict registration order
            self.active_plugins.append((plugin_name, plugin))
            if plugin.requires_custom_loading:
                self.plugins_require_custom_loading.append(plugin_name)

        if len(self.active_plugins) == 0:
            raise ValueError(
                "No plugins could be configured. Please check the acceleration "
                "framework configuration file."
            )

        assert (
            len(self.plugins_require_custom_loading) <= 1
        ), f"Can load at most 1 plugin with custom model loading, but tried to '{self.plugins_require_custom_loading}'."

    def model_loader(self, model_name: str, **kwargs):

        if len(self.plugins_require_custom_loading) == 0:
            raise NotImplementedError(
                f"Attempted model loading, but none of activated plugins '{list(self.active_plugins)}' "
                "require custom loading."
            )

        # otherwise there should be exactly 1
        plugin_name = self.plugins_require_custom_loading[0]
        plugin = [
            plugin for name, plugin in self.active_plugins if name == plugin_name
        ][0]
        return plugin.model_loader(model_name, **kwargs)

    def augmentation(
        self,
        model: PreTrainedModel,
        train_args: TrainingArguments,
        modifiable_args: Tuple,
    ):
        model_archs = set(model.config.architectures)  # get the config

        # NOTE: this assumes that augmentation order does not matter
        for plugin_name, plugin in self.active_plugins:

            # check the model arcs at augmentation
            if plugin.restricted_model_archs and not any(
                [x in model_archs for x in plugin.restricted_model_archs]
            ):
                raise ValueError(
                    f"Model architectures in '{model_archs}' are supported for '{plugin_name}'."
                )

            if plugin.requires_agumentation:
                model, modifiable_args = plugin.augmentation(
                    model, train_args, modifiable_args=modifiable_args
                )

        return model, modifiable_args

    @property
    def requires_custom_loading(self):
        return len(self.plugins_require_custom_loading) > 0

    @property
    def requires_agumentation(self):
        return any([x.requires_agumentation for _, x in self.active_plugins])

    def get_callbacks_and_ready_for_train(
        self, model: torch.nn.Module = None, accelerator: Accelerator = None
    ):
        # show the initialized message
        log_initialization_message(
            set([x for x, _ in self.active_plugins]),
            PLUGIN_REGISTRATIONS,
            logger=logger.info,
        )

        cbks = []
        for _, plugin in self.active_plugins:
            cbks.extend(plugin.get_callbacks_and_ready_for_train(model, accelerator))
        return cbks
