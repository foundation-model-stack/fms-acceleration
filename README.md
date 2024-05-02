# FMS Acceleration

This [monorepo](https://github.com/tweag/python-monorepo-example) collects libraries of packages that accelerate fine-tuning / training of large models.

It is currently being integrated into [fms-hf-tuning](https://github.com/foundation-model-stack/fms-hf-tuning).

**This package is in BETA under extensive development. Expect breaking changes!**

## Packages

Package | Description | Depends | License | Status
--|--|--|--|--
[framework](./libs/framework/README.md) | This acceleration framework for integration with huggingface trainers | | | Beta
[peft](./libs/peft/README.md) | For PEFT-training, e.g., 4bit QLoRA. | Huggingface<br>AutoGPTQ | Apache 2.0<br>MIT | Beta
unsloth | For fast triton kernels (e.g., fused LoRA, fast cross-entropy, rms, rope) | Xformers | Apache 2.0 with exclusions. | Under Development


## Usage with FMS HF Tuning

The below instructions show how to accelerate your QLoRA tuning experience with [tuning/sft_trainer.py](https://github.com/foundation-model-stack/fms-hf-tuning/blob/main/tuning/sft_trainer.py) from `fms-hf-tuning`. As an example, we show how to use AutoGPTQ triton V2 kernel with QLoRA tuning:
1. Checkout [fms-hf-tuning](https://github.com/foundation-model-stack/fms-hf-tuning) following their README instructions.
2. Ensure that [framework library](./libs/framework) is also installed:
    ```
    pip install git+https://github.com/foundation-model-stack/fms-acceleration.git#subdirectory=libs/framework
    ```
3. For QLoRA we use the Triton V2 GPTQ kernels integrated into [AutoGPTQ](https://github.com/AutoGPTQ/AutoGPTQ). For that, we require our [peft](./libs/peft/README.md) plugin:
    ```
    pip install git+https://github.com/foundation-model-stack/fms-acceleration.git#subdirectory=libs/peft
    ```
4. Prepare a YAML configuration for our acceleration framework plugins. In particular, this [sample configuration](sample-configurations/qlora-sample-config.yaml) will configure the AutoGPTQ triton V2 kernels for QLoRA PeFT.
    * Plugins automatically configured based on configuration; for more details on how plugins are configured, [see framework/README.md](./libs/framework/README.md#configuration-of-plugins).
5. Run `sft_trainer.py` passing `--acceleration_framework_config_file` pointing to the acceleration framework configuration YAML:
    ```
    python sft_trainer.py \
    	--acceleration_framework_config_file fixtures/acceleration_framework_debug.yaml \
        ...
    ```

For further code level details see [framework/README.md](./libs/framework/README.md).

## Reproducibility

TODO: to include section on benchmark scripts using `sft_trainer.py`.


## Maintainers

IBM Research, Singapore
- Fabian Lim flim@sg.ibm.com
- Aaron Chew aaron.chew1@sg.ibm.com
- Laura Wynter lwynter@sg.ibm.com