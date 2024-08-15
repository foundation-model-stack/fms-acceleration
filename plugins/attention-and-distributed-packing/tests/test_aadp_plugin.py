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

import os
from fms_acceleration.utils import (
    instantiate_framework,
    read_configuration,
)
from fms_acceleration_aadp import PaddingFreeAccelerationPlugin

# configuration
DIRNAME = os.path.dirname(__file__)
CONFIG_PATH_ILAB = os.path.join(DIRNAME, "../configs/padding_free.yaml")

def test_framework_installs_aadp_padding_free_plugin():
    with instantiate_framework(
        read_configuration(CONFIG_PATH_ILAB), require_packages_check=False
    ) as framework:
        for plugin in framework.active_plugins:
            assert isinstance(plugin[1], PaddingFreeAccelerationPlugin)
