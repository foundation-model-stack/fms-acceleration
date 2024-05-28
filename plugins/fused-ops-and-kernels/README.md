# FMS Acceleration for Fused Operations and Kernels

This library contains fused operations and custom kernels, to be expanded over time. Currently it contains the following:


1. Fused operations and kernels are extracted from [unsloth](#extracted-code-from-unsloth). 
    - Low-Rank Adapter Fused Operations
    - Fast RoPE Triton Kernels
    - Fast RMS LayerNorm Triton Kernels
    - Fast Cross Entropy Triton Kernels

## Plugins

Plugin | Description | Depends | Loading | Augmentation | Callbacks
--|--|--|--|--|--
[fast_quantized_peft](./src/fms_accelerate_foak/framework_plugin_fast_quantized_peft.py) | Loads fused lora, fast cross-entropy, fast rms, fast RoPE |  |  | ✅

### Code Extracted from Unsloth

<!--
NOTE: the 
- fused_ops/unsloth_lora -> unsloth main 
    * utils (fast_dequant, fast_gemv, fast_linear_forward, matmul_lora)
    * geglu, swiglu (this can be reused across other models, but currently used inside MLP fused ops only)
    * bnb (fast_lora.py)
    * gtqp (fast_lora, triton) -> jeromeku
- kernels
    *  cross_ent, rms, rope -> unsloth main
-->

Notes on the extraction of code from [unsloth](https://github.com/unslothai/unsloth):
- while unsloth is released under Apache 2.0, there are [exceptions to the permissive licenses scattered in the code base](https://github.com/unslothai/unsloth/blob/ec19e61c854dcf9104386fa63fc6c4f2944d4f35/unsloth/models/llama.py#L1140-L1143).
    ```
    it would require a commercial license if used to run on more than 4 GPUs, see 
    https://github.com/unslothai/unsloth/blob/d215fd902cf28feb8abcfde2d25281d0fbf9d28c/unsloth/models/llama.py#L1140-L1143
    ```
- these exceptions appear around [Feb 2024 Release](https://github.com/unslothai/unsloth/commit/3e4c5a323c16bbda2c92212b790073c4e99c2a55), around the model files (namely `llama.py`, `mistral.py`, etc). 
    * These model files are **not extracted**.
- All code extracted here before the Feb 2024 Release, see table below.

Path | Description | Extracted From  | Modifications | Date
--|--|--|--|--
[fused_ops/unsloth_lora](./src/fms_acceleration_foak/fused_ops/unsloth_lora) | QLoRA fast dequant, activation kernels | `unsloth/main` @ [1ecc0185](https://github.com/unslothai/unsloth/commit/1ecc0185a5759c7a0c95dfc96aceea5023cebdfc) |  | 28 Jan 2024
[fused_ops/unsloth_lora/bnb](./src/fms_acceleration_foak/fused_ops/unsloth_lora/bnb) | BNB fast lora | `unsloth/main` @ [1ecc0185](https://github.com/unslothai/unsloth/commit/1ecc0185a5759c7a0c95dfc96aceea5023cebdfc) |  | 28 Jan 2024
[fused_ops/unsloth_lora/gptq](./src/fms_acceleration_foak/fused_ops/unsloth_lora/gptq) | GPTQ fast dequant (triton_v2) | `jeromeku/main` @ [2839d39](https://github.com/jeromeku/unsloth/commit/2839d390ef3bb318904289bfb9a7751a782c4e44) | `fast_lora.py`<br>`triton/layers.py` | 6 Feb 2024
[kernels/unsloth](./src/fms_acceleration_foak/kernels/unsloth) | Fast RMS, RoPE, CrossEnt kernels | `unsloth/main` @ [1ecc0185](https://github.com/unslothai/unsloth/commit/1ecc0185a5759c7a0c95dfc96aceea5023cebdfc) | `cross_entropy_loss.py` | 28 Jan 2024

<!--
[models/](./src/fms_accelerate_unsloth/models/) | Model Forwards | `unsloth/main` @ [1ecc0185](https://github.com/unslothai/unsloth/commit/1ecc0185a5759c7a0c95dfc96aceea5023cebdfc)<br><br>`tohrnii/mixtral` @ [a55b7400](https://github.com/tohrnii/unsloth/commit/a55b740062b4fc8ce8f5196bfabe3cf860020ca7)   | `llama.py`<br>`mistral.py`<br>`mixtral.py`| `llama.py`<br>`mistral.py`<br>`mixtral.py` | 6 Feb 2024<br><br> 22 Feb 2024
-->


## Known Issues

- MixedPrecision `--fp16` should be used `fast_lora`. Also consider loading the model in `torch.float16`.
- `fast_lora` has issues with FSDP with the `peft` style of FSDP wrapping. 
    * This is because the adapter's forward functions are bypassed in the fused ops.
    * For AutoGPTQ this is addressed by distributing the adapters using DDP so they will be unsharded in time for the fused ops.
    * However for QLoRA this is not yet done https://github.com/foundation-model-stack/fms-acceleration/issues/3.
- `fast_rope_embeddings` does not work with position_ids. Currently `position_ids` are ignored and could give wrong results.