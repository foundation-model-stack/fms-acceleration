# FMS Acceleration for Accelerated PeFT Techniques

Currently only supports LoRA-related techniques, but more are in the pipeline to be added:

## Plugins

Plugin | Description | Depends | Loading | Augmentation | Callbacks
--|--|--|--|--|--
[autogptq](./src/fms_acceleration_peft/framework_plugin_autogptq.py) | Loads 4bit GPTQ-LoRA with quantized GPTQ as base | AutoGPTQ | ✅ | ✅
[bnb](./src/fms_acceleration_peft/framework_plugin_bnb.py) | Loads 4bit QLoRA with quantized bitsandbytes Linear4 | Huggingface<br>bitsandbytes | ✅ | ✅


### Key Points
- fix upcasting (resulting in slowdown) issue for `bnb` plugin, originally discovered by inventors of [Unsloth](https://unsloth.ai/blog/mistral-benchmark).
- `bnb` properly configured to work with FSDP following [this guide](https://huggingface.co/docs/bitsandbytes/main/en/fsdp_qlora). 
- `triton_v2` kernels are not yet properly integrated into huggingface optimum.
- `triton_v2` kernels are [the only 4bit kernels that work for training](https://github.com/AutoGPTQ/AutoGPTQ/issues/633).

## Known Issues

- Models with sliding windows (e.g., Mistral, Mixtral) will have [memory and throughout issues](https://github.com/huggingface/transformers/issues/30461).
- GPTQ-LORA sometimes observed to have `nan` grad norms in the begining of training, but training proceeds well otherwise.
- `low_cpu_mem_usage` temporarily disabled for AutoGPTQ until bug with `make_sure_no_tensor_in_meta_device` is resolved.
- Requires nightly [AutoGPTQ](https://github.com/AutoGPTQ/AutoGPTQ) until package `> 0.7.1` becomes available
    ```
    pip install git+https://github.com/AutoGPTQ/AutoGPTQ.git
    ```