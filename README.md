# FMS Acceleration

This [monorepo](https://github.com/tweag/python-monorepo-example) collects libraries of packages that accelerate fine-tuning / training of large models, 
intended to be part of the [fms-hf-tuning](https://github.com/foundation-model-stack/fms-hf-tuning) suite.

**This package is in BETA under extensive development. Expect breaking changes!**

## Packages

Package | Description | Depends | License | Status
--|--|--|--|--
[framework](./plugins/framework/README.md) | This acceleration framework for integration with huggingface trainers | | | Beta
[accelerated-peft](./plugins/accelerated-peft/README.md) | For PEFT-training, e.g., 4bit QLoRA. | Huggingface<br>AutoGPTQ | Apache 2.0<br>MIT | Beta
 TBA | Unsloth-inspired. Fused LoRA and triton kernels (e.g., fast cross-entropy, rms, rope) | Xformers | Apache 2.0 with exclusions. | Under Development


## Usage with FMS HF Tuning

This is intended to be a collection of many acceleration routines (including PeFT and other techniques). The below demonstrates a concrete example to show how to accelerate your tuning experience with [tuning/sft_trainer.py](https://github.com/foundation-model-stack/fms-hf-tuning/blob/main/tuning/sft_trainer.py) from `fms-hf-tuning`.

### Accelerated GPTQ-LoRA Training

The below instructions performs accelerated PeFT fine-tuning using this package, in particular GPTQ-LoRA tuning using the AutoGPTQ `triton_v2` kernel; this kernel is state-of-the-art [provided by `jeromeku` on Mar 2024](https://github.com/AutoGPTQ/AutoGPTQ/pull/596):
1. Checkout [fms-hf-tuning](https://github.com/foundation-model-stack/fms-hf-tuning).
2. Ensure that our [framework library](./plugins/framework) is also installed:
    ```
    pip install git+https://github.com/foundation-model-stack/fms-acceleration.git#subdirectory=plugins/framework
    ```
3. For GPTQ-LoRA we require our [accelerated-peft](./plugins/peft/README.md) plugin:
    ```
    pip install git+https://github.com/foundation-model-stack/fms-acceleration.git#subdirectory=plugins/accelerated-peft
    ```
4. Prepare a YAML configuration for the acceleration framework plugins, see our [acccelerated-peft sample configuration](sample-configurations/accelerated-peft-sample-config.yaml) that configures triton V2 kernels.
    * The plugin framework handles automatically the plugin configuration for whatever plugins that are installed; for more details [see framework/README.md](./plugins/framework/README.md#configuration-of-plugins).
5. Run `sft_trainer.py` passing `--acceleration_framework_config_file` pointing to the acceleration framework configuration YAML:
    ```
    python sft_trainer.py \
    	--acceleration_framework_config_file fixtures/acceleration_framework_debug.yaml \
        ...
    ```

**[Packages](#packages) will be updated over times more become available!**.

### Code Architecture

For deeper dive into details see [framework/README.md](./plugins/framework/README.md).

## Reproducibility

TODO: to include section on benchmark scripts using `sft_trainer.py`.


## Maintainers

IBM Research, Singapore
- Fabian Lim flim@sg.ibm.com
- Aaron Chew aaron.chew1@sg.ibm.com
- Laura Wynter lwynter@sg.ibm.com