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
import os

# Third Party
from fms_acceleration import AccelerationPluginConfigError
from fms_acceleration.utils import (
    instantiate_framework,
    read_configuration,
    update_configuration_contents,
)
import pytest

# instantiate_fromwork will handle registering and activating AutoGPTQAccelerationPlugin

# configuration
DIRNAME = os.path.dirname(os.path.join(__file__))
CONFIG_PATH_AUTO_GPTQ = os.path.join(DIRNAME, "../configs/autogptq.yaml")
CONFIG_PATH_BNB = os.path.join(DIRNAME, "../configs/bnb.yaml")


def test_configure_gptq_plugin():
    "test auto_gptq plugin loads correctly"

    # test that provided configuration correct correct instantiates plugin
    with instantiate_framework(read_configuration(CONFIG_PATH_AUTO_GPTQ)) as framework:

        # check flags and callbacks
        assert framework.requires_custom_loading
        assert framework.requires_agumentation
        assert len(framework.callbacks()) == 0

    # attempt to activate plugin with configuration pointing to wrong path
    # - raise with message that no plugins can be configured
    with pytest.raises(ValueError) as e:
        with instantiate_framework(
            update_configuration_contents(
                read_configuration(CONFIG_PATH_AUTO_GPTQ),
                "peft.quantization.auto_gptq",
                "something",
            )
        ):
            pass

    e.match("No plugins could be configured")

    # attempt to actiavte plugin with unsupported settings
    # - raise with appropriate message complaining about wrong setting
    for key, wrong_value in [
        ("peft.quantization.auto_gptq.kernel", "triton"),
        ("peft.quantization.auto_gptq.from_quantized", False),
    ]:
        with pytest.raises(AccelerationPluginConfigError) as e:
            with instantiate_framework(
                update_configuration_contents(
                    read_configuration(CONFIG_PATH_AUTO_GPTQ), key, wrong_value
                )
            ):
                pass

        e.match(f"AutoGPTQAccelerationPlugin: Value at '{key}'")


def test_configure_bnb_plugin():
    "test bnb plugin loads correctly"

    # test that provided configuration correct correct instantiates plugin
    with instantiate_framework(read_configuration(CONFIG_PATH_BNB)) as framework:

        # check flags and callbacks
        assert framework.requires_custom_loading
        assert framework.requires_agumentation
        assert len(framework.callbacks()) == 0

    # test valid combinatinos
    for key, correct_value in [
        ("peft.quantization.bitsandbytes.quant_type", "nf4"),
        ("peft.quantization.bitsandbytes.quant_type", "fp4"),
    ]:
        with instantiate_framework(
            update_configuration_contents(
                read_configuration(CONFIG_PATH_BNB), key, correct_value
            )
        ):
            # check flags and callbacks
            assert framework.requires_custom_loading
            assert framework.requires_agumentation
            assert len(framework.callbacks()) == 0

    # attempt to activate plugin with configuration pointing to wrong path
    # - raise with message that no plugins can be configured
    with pytest.raises(ValueError) as e:
        with instantiate_framework(
            update_configuration_contents(
                read_configuration(CONFIG_PATH_BNB),
                "peft.quantization.bitsandbytes",
                "something",
            )
        ):
            pass

    e.match("No plugins could be configured")

    # attempt to actiavte plugin with unsupported settings
    # - raise with appropriate message complaining about wrong setting
    for key, correct_value in [
        ("peft.quantization.bitsandbytes.quant_type", "wrong_type"),
    ]:
        with pytest.raises(AccelerationPluginConfigError) as e:
            with instantiate_framework(
                update_configuration_contents(
                    read_configuration(CONFIG_PATH_BNB), key, correct_value
                )
            ):
                pass

        e.match(f"BNBAccelerationPlugin: Value at '{key}'")
