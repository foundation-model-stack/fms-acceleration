# FMS Acceleration for UnSloth

This library contains fused operations and custom kernels from the unsloth library

**This library is undergoing massive refactoring to extract code from [unsloth](https://github.com/unslothai/unsloth).**


Licensing
- While [unsloth](https://github.com/unslothai/unsloth) is released under Apache 2.0, there are [exceptions to the permissive licenses scattered in the code base](https://github.com/unslothai/unsloth/blob/ec19e61c854dcf9104386fa63fc6c4f2944d4f35/unsloth/models/llama.py#L1140-L1143).
    ```
    it would require a commercial license if used to run on more than 4 GPUs, see 
    https://github.com/unslothai/unsloth/blob/d215fd902cf28feb8abcfde2d25281d0fbf9d28c/unsloth/models/llama.py#L1140-L1143
    ```
- these exceptions appear around [Feb 2024 Release](https://github.com/unslothai/unsloth/commit/3e4c5a323c16bbda2c92212b790073c4e99c2a55), around the model files (namely `llama.py`, `mistral.py`, etc).
- in light of this, all code extracted here are taken before the Feb 2024 Release, see dates below.

## Plugins

Plugin | Description | Depends | Loading | Augmentation | Callbacks
--|--|--|--|--|--
[unsloth](./src/fms_accelerate_unsloth/framework_plugin_unsloth_stackable.py) | Loads fused lora, fast cross-entropy, fast rms, fast RoPE | UnSloth |  | ✅

### Unsloth Plugin

Path | Description | Extracted From  | Modifications | Extracted Only | Date
--|--|--|--|--|--
[kernels/](./src/fms_accelerate_unsloth/kernels/) | BNB / CrossEnt / RoPE | `unsloth/main` @ [1ecc0185](https://github.com/unslothai/unsloth/commit/1ecc0185a5759c7a0c95dfc96aceea5023cebdfc) |  | | 28 Jan 2024
[gptq/](./src/fms_accelerate_unsloth/gptq/) | GPTQ / triton_v2 | `jeromeku/main` @ [2839d39](https://github.com/jeromeku/unsloth/commit/2839d390ef3bb318904289bfb9a7751a782c4e44) | `fast_lora.py`<br>`layers.py` | | 6 Feb 2024
[models/](./src/fms_accelerate_unsloth/models/) | Model Forwards | `unsloth/main` @ [1ecc0185](https://github.com/unslothai/unsloth/commit/1ecc0185a5759c7a0c95dfc96aceea5023cebdfc)<br><br>`tohrnii/mixtral` @ [a55b7400](https://github.com/tohrnii/unsloth/commit/a55b740062b4fc8ce8f5196bfabe3cf860020ca7)   | `llama.py`<br>`mistral.py`<br>`mixtral.py`| `llama.py`<br>`mistral.py`<br>`mixtral.py` | 6 Feb 2024<br><br> 22 Feb 2024


## Known Issues

- MixedPrecision `--fp16` should be used `fast_lora`. Also consider loading the model in `torch.float16`.
- [Unsloth](https://github.com/unslothai/unsloth) does not natively support FSDP. We may plan to support it with our refactored versions.