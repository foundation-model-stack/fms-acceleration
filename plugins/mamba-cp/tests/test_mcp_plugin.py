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
import os

# Third Party
import pytest

# configuration
DIRNAME = os.path.dirname(__file__)
CONFIG_PATH = os.path.join(DIRNAME, "../configs/mcp.yaml")


@pytest.mark.skipif(
    not pytest.importorskip("mamba_ssm", reason="mamba_ssm is not installed"),
    reason="mamba_ssm is not installed",
)
def test_framework_installs_mcp_plugin():
    # Third Party
    # pylint: disable=import-outside-toplevel
    from fms_acceleration.utils import instantiate_framework, read_configuration

    # First Party
    # pylint: disable=import-outside-toplevel
    from fms_acceleration_mcp import MCPAccelerationPlugin

    with instantiate_framework(
        read_configuration(CONFIG_PATH), require_packages_check=False
    ) as framework:
        for plugin in framework.active_plugins:
            assert isinstance(plugin[1], MCPAccelerationPlugin)
