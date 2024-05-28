from .model_patcher import ModelPatcher

PATCHES = [
    '.models.llama',
    '.models.mistral'
]
PLUGIN_PREFIX = 'fms_acceleration_foak'

# TODO: remove the need for the prefix
ModelPatcher.load_patches(
    [f"{PLUGIN_PREFIX}{postfix}" for postfix in PATCHES],
)