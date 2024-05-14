# FMS Acceleration

This [monorepo](https://github.com/tweag/python-monorepo-example) collects libraries of packages that accelerate fine-tuning / training of large models, 
intended to be part of the [fms-hf-tuning](https://github.com/foundation-model-stack/fms-hf-tuning) suite.

**This package is in BETA under extensive development. Expect breaking changes!**

## Plugins

Plugin | Description | Depends | License | Status
--|--|--|--|--
[framework](./plugins/framework/README.md) | This acceleration framework for integration with huggingface trainers | | | Beta
[accelerated-peft](./plugins/accelerated-peft/README.md) | For PEFT-training, e.g., 4bit QLoRA. | Huggingface<br>AutoGPTQ | Apache 2.0<br>MIT | Beta
 TBA | Unsloth-inspired. Fused LoRA and triton kernels (e.g., fast cross-entropy, rms, rope) | Xformers | Apache 2.0 with exclusions. | Under Development
 TBA | [MegaBlocks](https://github.com/databricks/megablocks) inspired triton Kernels and acclerations for Mixture-of-Expert models |  | Apache 2.0 | Under Development

## Usage with FMS HF Tuning

This is intended to be a collection of many acceleration routines (including accelerated peft and other techniques). Below demonstrates a concrete example to show how to accelerate your tuning experience with [tuning/sft_trainer.py](https://github.com/foundation-model-stack/fms-hf-tuning/blob/main/tuning/sft_trainer.py) from `fms-hf-tuning`.

### Example: Accelerated GPTQ-LoRA Training

Below instructions for accelerated peft fine-tuning. In particular GPTQ-LoRA tuning with the AutoGPTQ `triton_v2` kernel; this kernel is state-of-the-art [provided by `jeromeku` on Mar 2024](https://github.com/AutoGPTQ/AutoGPTQ/pull/596):
1. Checkout [fms-hf-tuning](https://github.com/foundation-model-stack/fms-hf-tuning) and install the [framework library](./plugins/framework):
    ```
    $ pip install -e .[fms-accel]
    ```
    or alternatively install the framework directly:
    ```
    $ pip install git+https://github.com/foundation-model-stack/fms-acceleration.git#subdirectory=plugins/framework
    ```

    The above installs the command line utility `fms_acceleration.cli`, which can then be used to install plugins and view sample configurations. 

2. Prepare a YAML configuration for the acceleration framework plugins. To help with this, `fms_acceleration.cli` provides a `configs` utility to search for sample configs by entering the following:
    ```
    $ python -m fms_acceleration.cli configs

    1. accelerated-peft-autogptq (accelerated-peft-autogptq-sample-configuration.yaml) - plugins: ['accelerated-peft']
    2. accelerated-peft-bnb (accelerated-peft-bnb-nf4-sample-configuration.yaml) - plugins: ['accelerated-peft']
    ```

    or alternatively search the configurations manually:
    * [Full sample configuration list](./sample-configurations/CONTENTS.yaml) shows the `plugins` required for the configs.
    * E.g., [Accelerated GPTQ-LoRA configuration here](sample-configurations/accelerated-peft-autogptq-sample-configuration.yaml). 


3. Install the required `plugins`. Use `list` to view available plugins; this list updates [as more plugins get developed](#plugins). Recall that `configs` list the required `plugins` for the sample configurations; make sure all of them are installed.
    ```
    $ python -m fms_acceleration.cli plugins

    Choose from the list of plugin shortnames, and do:
    * 'python -m fms_acceleration.cli install <pip-install-flags> PLUGIN_NAME'.

    List of PLUGIN_NAME [PLUGIN_SHORTNAME]:

    1. fms_acceleration_peft [peft]
    ```
    and then `install` the plugin. We install the `fms-acceleration-peft` plugin for GPTQ-LoRA tuning with triton v2 as:
    ```
    python -m fms_acceleration.cli install fms_acceleration_peft
    ```
    The above is the equivalent of:
    ```
    pip install git+https://github.com/foundation-model-stack/fms-acceleration.git#subdirectory=plugins/accelerated-peft
    ```

4. Run `sft_trainer.py` while providing the correct arguments: 
    * `--acceleration_framework_config_file` pointing to framework configuration YAML.  The framework activates relevant plugins given the framework configuration; for more details [see framework/README.md](./plugins/framework/README.md#configuration-of-plugins).
    * arguments required for correct operation (e.g., if using accelerated peft, then `peft_method` is required).

        * Use `arguments` along with the [sample configuration `shortname`](./sample-configurations/CONTENTS.yaml) to display the relevant *critical arguments*; these arguments can be manually referred from [scenarios.yaml](./scripts/benchmarks/scenarios.yaml):
        ```
        $ python -m fms_acceleration.cli arguments accelerated-peft-autogptq

        Searching for configuration shortnames: ['accelerated-peft-autogptq']
        1. scenario: accelerated-peft-gptq
        configs: accelerated-peft-autogptq
        arguments:
            --learning_rate 2e-4 \
            --fp16 True \
            --torch_dtype float16 \
            --peft_method lora \
            --r 16 \
            --lora_alpha 16 \
            --lora_dropout 0.0 \
            --target_modules ['q_proj', 'k_proj', 'v_proj', 'o_proj']
        ```

    * More info on `defaults.yaml` and `scenarios.yaml` [found here](./scripts/benchmarks/README.md#benchmark-scenarios).
        * Arguments *not critical to the plugins* found in [defaults.yaml](./scripts/benchmarks/defaults.yaml). These can be taken with liberty.
        * Arguments *critcal to plugins* found in [scenarios.yaml](./scripts/benchmarks/scenarios.yaml). The relevant section of [scenarios.yaml](./scripts/benchmarks/scenarios.yaml), is the one whose `framework_config` entries, match the `shortname` of the sample configuration of [interest](./sample-configurations/CONTENTS.yaml).

5. Run `sft_trainer.py` providing the acceleration configuration and arguments:
    ```
    # when using sample-configurations, arguments can be referred from
    # defaults.yaml and scenarios.yaml
    python sft_trainer.py \
        --acceleration_framework_config_file framework.yaml \
        ...  # arguments 
    ```

    Activate `TRANSFORMERS_VERBOSITY=info` to see the huggingface trainer printouts and verify that `AccelerationFramework` is activated!

    ```
    # this printout will be seen in huggingface trainer logs if acceleration is activated
    ***** FMS AccelerationFramework *****
    Active Plugin: AutoGPTQAccelerationPlugin. Python package: fms_acceleration_peft. Version: 0.0.1.
    ***** Running training *****
    Num examples = 1,549
    Num Epochs = 1
    Instantaneous batch size per device = 4
    Total train batch size (w. parallel, distributed & accumulation) = 4
    Gradient Accumulation steps = 1
    Total optimization steps = 200
    Number of trainable parameters = 13,631,488
    ```

**Over time, more [plugins](#plugins) will be updated, so please check here for the latest accelerations!**.

### CUDA Dependencies

This repo requires CUDA to compute the kernels, and it is convinient to use [NVidia Pytorch Containers](https://docs.nvidia.com/deeplearning/frameworks/support-matrix/index.html) that already comets with CUDA installed. We have tested with the following versions:
- `pytorch:24.03-py3`

### Benchmarks

The benchmarks can be reproduced [with the provided scripts](./scripts/benchmarks). 
- includes baseline benches (e.g., standard fine-tuning, standard peft).
- benches for various [acceleration sample configs](./sample-configurations/CONTENTS.yaml).

See below CSV files for various results:
- [A100-80GB](./scripts/benchmarks/refs/a100_80gb.csv).
- [L40-40GB](./scripts/benchmarks/refs/l40_40gb.csv).

### Code Architecture

For deeper dive into details see [framework/README.md](./plugins/framework/README.md).


## Maintainers

IBM Research, Singapore
- Fabian Lim flim@sg.ibm.com
- Aaron Chew aaron.chew1@sg.ibm.com
- Laura Wynter lwynter@sg.ibm.com