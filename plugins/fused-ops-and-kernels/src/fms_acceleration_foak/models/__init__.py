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

# Local
from fms_acceleration.model_patcher import ModelPatcher
import importlib

PATCHES = [".models.llama", ".models.mistral", ".models.mixtral"]
PLUGIN_PREFIX = "fms_acceleration_foak"

# TODO: remove the need for the prefix
def register_foak_model_patch_rules(base_type):
    for postfix in PATCHES:
        # define the patch module path to import
        # if it exist, import the module
        patch_path = f"{PLUGIN_PREFIX}{postfix}"
        if importlib.util.find_spec(patch_path):
            m = importlib.import_module(patch_path)
            # get all model patcher rules from the module
            # register every rule in the module
            rules = m.get_mp_rules(base_type)
            for _rule in rules:
                ModelPatcher.register(_rule)
