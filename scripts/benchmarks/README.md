# FMS Benchmark Utility

This utility used to measure throughput and other improvements obtained when using `fms-acceleration` plugins.
- [benchmark.py](./benchmark.py): Main benchmark script.
- [scenarios.yaml](./scenarios.yaml): `sft_trainer.py` arguments organized different *scenarios*.
  * Each `scenario` may apply to one ore more `AccelerationFramework` [sample configuration](../../sample-configurations). These are the *critical* arguments needed for correct operation.
  * See [section on benchmark scenarios](#benchmark-scenarios) for more details.
- [defaults.yaml](./defaults.yaml): `sft_trainer.py` arguments that may be used in addition to [scenarios.yaml](./scenarios.yaml). These are the *non-critical* arguments that will not affect plugin operation.
- [accelerate.yaml](./accelerate.yaml): configurations required by[`accelerate launch`](https://huggingface.co/docs/accelerate/en/package_reference/cli) for multi-gpu benchmarks.


## Benchmark Scenarios

An example of a `scenario` for `accelerated-peft-gptq` given as follows:
```yaml
scenarios:

  # benchmark scenario for accelerated peft using AutoGPTQ triton v2
  - name: accelerated-peft-gptq
    framework_config: 
      # one ore more framework configurations that fall within the scenario group.
      # - each entry points to a shortname in CONTENTS.yaml
      - accelerated-peft-autogptq

    # sft_trainer.py arguments critical for correct plugin operation
    arguments:
      fp16: True
      learning_rate: 2e-4
      torch_dtype: float16
      peft_method: lora
      r: 16
      lora_alpha: 16
      lora_dropout: 0.0
      target_modules: "q_proj k_proj v_proj o_proj"
      model_name_or_path: 
        - 'mistralai/Mistral-7B-v0.1'
        - 'mistralai/Mixtral-8x7B-Instruct-v0.1'
        - 'NousResearch/Llama-2-70b-hf'
```

A `scenario` has the following key components:
- `framework_config`: points to one or more [acceleration configurations](#sample-acceleration-configurations). 
  * list of [sample config `shortname`](../../sample-configurations/CONTENTS.yaml).
  * for each `shortname` is a different bench.
- `arguments`: the *critical* `sft_trainer.py` arguments that need to be passed in alongiside `framework_config` to ensure correct operation.
  * `model_name_or_path` is a list, and the bench will enumerate all of them.
  * **NOTE**: a `plugin` **may not work with arbitrary models**. This depends on the plugin's setting of [`AccelerationPlugin.restricted_model_archs`](../../plugins/framework/src/fms_acceleration/framework_plugin.py).


## Usage

The best way is via `tox` which manages the dependencies, including installing the correct version [fms-hf-tuning](https://github.com/foundation-model-stack/fms-hf-tuning).

- run a *small* representative set of benches:
    ```
    tox -e run-benches
    ```
- run the *full* set of benches on for both 1 and 2 GPU cases:
    ```
    tox -e run-benches -- "1 2" 
    ```

Note:
- `tox` command above accepts environment variables `DRY_RUN, NO_DATA_PROCESSING, NO_OVERWRITE`. See `scripts/run_benchmarks.sh`

## Running Benchmarks

The convinience script [`run_benchmarks.sh`](../run_benchmarks.sh) configures and runs `benchmark.py`; the command is:
```
bash run_benchmarks.sh NUM_GPUS_MATRIX RESULT_DIR SCENARIOS_CONFIG SCENARIOS_FILTER
```
where:
- `NUM_GPUS_MATRIX`: list of `num_gpu` settings to bench for, e.g. `"1 2"` will bench for 1 and 2 gpus.
- `RESULT_DIR`: where the benchmark results will be placed.
- `SCENARIOS_CONFIG`: the `scenarios.yaml` file.
- `SCENARIOS_CONFIG`: specify to run only a specific `scenario` by providing the specific `scenario` name.

The recommended way to run `benchmarks.sh` is using `tox` which handles the dependencies:
```
tox -e run-benches -- NUM_GPUS_MATRIX RESULT_DIR SCENARIOS_CONFIG SCENARIOS_FILTER
```

Alternatively run [`benchmark.py`](./benchmark.py) directly. To see the help do:
```
python benchmark.py --help
```

Note:
- in `run_benchmarks.sh` we will clear the `RESULT_DIR` if it exists, to avoid contaimination with old results. To protect against overwrite, then always run with `NO_OVERWRITE=true`.
