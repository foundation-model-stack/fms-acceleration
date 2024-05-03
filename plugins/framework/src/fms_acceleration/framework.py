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
from typing import Dict, List, Optional, Tuple

# Third Party
from transformers import PreTrainedModel, TrainingArguments
from transformers.utils.import_utils import _is_package_available
import yaml

# First Party
from fms_acceleration.framework_plugin import (
    AccelerationPlugin,
    get_relevant_configuration_sections,
)


def check_plugin_packages(plugin: AccelerationPlugin):
    if plugin.require_packages is None:
        return True  # passthrough

    for package_name in plugin.require_packages:
        _is_package_available(package_name)


KEY_PLUGINS = "plugins"


class AccelerationFramework:

    active_plugins: Dict[str, AccelerationPlugin] = dict()
    plugins_require_custom_loading: List = list()

    def __init__(self, configuration_file: Optional[str] = None):

        with open(configuration_file, "r") as f:
            contents = yaml.safe_load(f)

        # pepare the plugin configurations
        plugin_configs = {k: v for k, v in contents[KEY_PLUGINS].items()}

        for selected_configs, cls in get_relevant_configuration_sections(
            plugin_configs
        ):

            # then the model is to be installed
            # get the plugin
            plugin_name = str(cls.__name__)
            plugin = cls(selected_configs)

            # check plugin
            check_plugin_packages(plugin)

            # install plugin
            self.active_plugins[plugin_name] = plugin
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
                f"Attempted model loading, but none of activated plugins '{list(self.active_plugins.keys())}' "
                "require custom loading."
            )

        # otherwise there should be exactly 1
        plugin_name = self.plugins_require_custom_loading[0]
        return self.active_plugins[plugin_name].model_loader(model_name, **kwargs)

    def augmentation(
        self,
        model: PreTrainedModel,
        train_args: TrainingArguments,
        modifiable_args: Tuple,
    ):
        model_archs = set(model.config.architectures)  # get the config

        # NOTE: this assumes that augmentation order does not matter
        for plugin_name, plugin in self.active_plugins.items():

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
        return any([x.requires_agumentation for x in self.active_plugins.values()])
