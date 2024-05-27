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
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple
import importlib
import sys

# Third Party
from accelerate import Accelerator
from peft import LoraConfig
from transformers import TrainingArguments
import torch


@dataclass
class PluginRegistration:
    plugin: "AccelerationPlugin"
    AND: List[str] = None
    # OR: List[str] = None # not implemented yet

    # package metadata
    package_name: str = None
    package_version: str = None


PLUGIN_REGISTRATIONS: List[PluginRegistration] = list()


def _trace_key_path(configuration: Dict, key: str):
    t = configuration

    try:
        for k in key.split("."):
            t = t[k]
    except KeyError:
        return None  # None will mean not found
    return t


def get_relevant_configuration_sections(configuration: Dict) -> Dict:
    results = []

    # assume the registrations are all done with at least some default key
    for registration in PLUGIN_REGISTRATIONS:
        relevant_config = {}
        # OR is not implemented yet
        reject = False
        for key in registration.AND:
            content = _trace_key_path(configuration, key)
            if content is None:
                reject = True
                break

            path = key.split(".")
            n = len(path)
            _cfg = relevant_config
            while n > 1:
                p = path.pop(0)
                _cfg[p] = {}
                _cfg = _cfg[p]
                n -= 1

            _cfg[path[0]] = content

        if reject:
            continue

        if len(relevant_config) > 0:
            results.append((relevant_config, registration.plugin))

    return results


class AccelerationPlugin:

    # will be triggered if the configuration_paths are found in the
    # acceleration framework configuration file (under KEY_PLUGINS)
    @staticmethod
    def register_plugin(
        plugin: "AccelerationPlugin",
        configuration_and_paths: List[str],
        **kwargs,
    ):
        global PLUGIN_REGISTRATIONS

        # get the package metadata
        pkg_name = sys.modules[plugin.__module__].__package__
        try:
            package_version = importlib.metadata.version(pkg_name)
        except importlib.metadata.PackageNotFoundError: 
            package_version = None

        PLUGIN_REGISTRATIONS.append(
            PluginRegistration(
                plugin=plugin,
                AND=configuration_and_paths,
                package_name=pkg_name,
                package_version=package_version,
            )
        )

    restricted_model_archs: Optional[Set] = None
    require_packages: Optional[Set] = None

    def __init__(self, configurations: Dict[str, Dict]):

        # will pass in a list of dictionaries keyed by "configuration_keys"
        # to be used for initialization
        self.configurations = configurations

    @property
    def requires_custom_loading(self):
        return False

    @property
    def requires_agumentation(self):
        return False

    def model_loader(self, model_name: str, **kwargs):
        raise NotImplementedError

    def augmentation(
        self,
        model: torch.nn.Module,
        train_args: TrainingArguments,
        modifiable_args: Tuple[LoraConfig],
    ):
        raise NotImplementedError

    def get_callbacks_and_ready_for_train(
        self, model: torch.nn.Module = None, accelerator: Accelerator = None
    ):
        return []

    def _check_config_and_maybe_check_values(self, key: str, values: List[Any] = None):
        t = _trace_key_path(self.configurations, key)

        if values is not None:  # if there is something to check against
            if isinstance(t, dict):
                # if the tree is a dict
                if len(t.keys()) > 1:
                    raise AccelerationPluginConfigError(
                        f"{self.__class__.__name__}: '{key}' found but amongst multiple '{t.keys()}' exist. Ambiguous check in expected set '{values}'."
                    )
                t = list(t.keys())[0]  # otherwise take the first value

            if t not in values:
                raise AccelerationPluginConfigError(
                    f"{self.__class__.__name__}: Value at '{key}' was '{t}'. Not found in expected set '{values}'."
                )
        else:
            # if nothing to check against, we still want to ensure its a valid
            # configuration key
            if t is None:
                raise AccelerationPluginConfigError(
                    f"{self.__class__.__name__}: '{key}' was not a valid configuration config"
                )

        return t

    def _check_config_equal(self, key: str, value: Any, **kwargs):
        return self._check_config_and_maybe_check_values(key, [value], **kwargs)


class AccelerationPluginConfigError(Exception):
    pass
