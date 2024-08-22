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
from .framework_plugin_loss import LossAccelerationPlugin
from .framework_plugin_multipack import MultipackDataloaderAccelerationPlugin
from .framework_plugin_padding_free import PaddingFreeAccelerationPlugin
from .framework_plugin_mlp_dropout import MLPDropoutAccelerationPlugin
from .framework_plugin_embed_dropout import EmbeddingDropoutAccelerationPlugin

from .model_patcher import ModelPatcher

PATCHES = [".flash_attn"]
PLUGIN_PREFIX = "fms_acceleration_attn"

# TODO: remove the need for the prefix
ModelPatcher.load_patches(
    [f"{PLUGIN_PREFIX}{postfix}" for postfix in PATCHES],
)