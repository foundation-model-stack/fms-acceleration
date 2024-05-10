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
 TBA | Triton Kernels for Mixture-of-Experts. | [MegaBlocks](https://github.com/databricks/megablocks) | Apache 2.0 | Under Development

## Usage with FMS HF Tuning

This is intended to be a collection of many acceleration routines (including PeFT and other techniques). Below demonstrates a concrete example to show how to accelerate your tuning experience with [tuning/sft_trainer.py](https://github.com/foundation-model-stack/fms-hf-tuning/blob/main/tuning/sft_trainer.py) from `fms-hf-tuning`.

### Accelerated GPTQ-LoRA Training

These instructions perfor accelerated PeFT fine-tuning using this package, in particular GPTQ-LoRA tuning with the AutoGPTQ `triton_v2` kernel; this kernel is state-of-the-art [provided by `jeromeku` on Mar 2024](https://github.com/AutoGPTQ/AutoGPTQ/pull/596):
1. Checkout [fms-hf-tuning](https://github.com/foundation-model-stack/fms-hf-tuning) and install the [framework library](./plugins/framework):
    ```
    pip install -e .[fms-accel]
    ```
    or alternatively install the framework directly:
    ```
    pip install git+https://github.com/foundation-model-stack/fms-acceleration.git#subdirectory=plugins/framework
    ```
3. The above installs the command line utility `fms_acceleration.cli`, which can then be used to install plugins. Use `list` to view available plugins; this list will be updated [as more plugins get developed](#plugins):
    ```
    $ python -m fms_acceleration.cli list

    Choose from the list of plugin shortnames, and do:
    * 'python -m fms_acceleration.cli install <pip-install-flags> PLUGIN_NAME'.

    Alternatively, specify a local path <PATH> and do:
    * 'python -m fms_acceleration.cli install <pip-install-flags> <PATH>'.

    List of PLUGIN_NAME [PLUGIN_SHORTNAME]:

    1. fms_acceleration_peft [peft]
    ```
    and then `install` the plugin:
    ```
    python -m fms_acceleration.cli install fms_acceleration_peft
    ```
    The above example command installs the plugin for GPTQ-LoRA tuning with triton v2, and is the equivalent of:
    ```
    pip install git+https://github.com/foundation-model-stack/fms-acceleration.git#subdirectory=plugins/accelerated-peft
    ```
4. Prepare a YAML configuration for the acceleration framework plugins, see our [acccelerated-peft sample configuration](sample-configurations/accelerated-peft-autogptq-sample-configuration.yaml) that configures triton v2 kernels.
    * The framework configures the *installed* plugins based on this framework; for more details [see framework/README.md](./plugins/framework/README.md#configuration-of-plugins).
5. Run `sft_trainer.py` passing `--acceleration_framework_config_file` pointing to the acceleration framework configuration YAML:
    ```
    python sft_trainer.py \
    	--acceleration_framework_config_file fixtures/acceleration_framework_debug.yaml \
        ...
    ```
    Also, ensure that the appropriate arguments are passed, e.g., when using `fms_acceleration_peft` pass the appropriate `peft` arguments. 
      * For samples of these arguments consult the relevant discussion in the [section on benchmarks](#benchmarks).

**Over time, more [plugins](#plugins) will be updated, so please check here for the latest accelerations!**.

### CUDA Dependencies

This repo requires CUDA to compute the kernels. We have tested with the following [NVidia Pytorch Containers](https://docs.nvidia.com/deeplearning/frameworks/support-matrix/index.html).
- `pytorch:24.03-py3`

### Benchmarks

Our [benchmark code](./scripts/benchmarks) generates throughputs for different acceleration scenarios. To reproduce (requires `tox`):

```
tox -e run-benches
```
which runs a small set of benches using [benchmark.py](./scripts/benchmarks/benchmark.py).

We have some (draft) benches here:
- [A100-80GB (CSV file)](./scripts/benchmarks/summary.csv).

### Code Architecture

For deeper dive into details see [framework/README.md](./plugins/framework/README.md).


## Maintainers

IBM Research, Singapore
- Fabian Lim flim@sg.ibm.com
- Aaron Chew aaron.chew1@sg.ibm.com
- Laura Wynter lwynter@sg.ibm.com