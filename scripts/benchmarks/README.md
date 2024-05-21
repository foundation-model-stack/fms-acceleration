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

- install the `setup_requirements.txt` to get `tox`:
    ```
    pip install -r setup_requirements.txt
    ```

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

## Logging Memory

There are 2 ways to benchmark memory in `benchmark.py`:
- With Nvidia `nvidia-smi`'s API by passing argument `--log_nvidia_smi`
- With HuggingFace `HFTrainer`'s API by passing argument `--log_memory_hf`

Both approaches will print out the memory value to the benchmark report

### Nvidia `nvidia-smi`
`nvidia-smi` is a command line utility (CLI) based on the Nvidia Manage Library (NVML)`. A separate process call is used to start, log and finally terminate the CLI for every experiment.  

The keyword `memory.used` is passed to `--query-gpu` argument to log the memory usage at some interval. The list of keywords that can be logged can be referenced from `nvidia-smi --help-query-gpu`

Since it runs on a separate process, it is a less invasive logging approach and less likely to affect the training. However, it is a coarser approach than HF as NVML's definition of used memory in its [documentation](https://docs.nvidia.com/deploy/nvml-api/structnvmlMemory__t.html#structnvmlMemory__t:~:text=Sum%20of%20Reserved%20and%20Allocated%20device%20memory%20(in%20bytes).%20Note%20that%20the%20driver/GPU%20always%20sets%20aside%20a%20small%20amount%20of%20memory%20for%20bookkeeping) takes the sum of (memory allocated + memory reserved).

After every experiment, 
  - the logged values are calibrated to remove any existing foreign memory values
  - the peak values for each gpu device are taken
  - the values are finally averaged across all devices.

### HuggingFace `HFTrainer`
HFTrainer has a feature to log memory through the `skip_memory_metrics=False` training argument. In their [documentation](https://huggingface.co/docs/transformers/en/main_classes/trainer#transformers.TrainingArguments.skip_memory_metrics), it is mentioned that setting this argument to `False` will affect training speed. In our tests so far (below), we do not see significant difference in throughput (tokens/sec) when using this argument.

The HFTrainer API is more granular than `nvidia-smi` as 
  - It reports the allocated memory by calling `torch.cuda.memory_allocated()` and `torch.cuda.max_memory_allocated()` inside its probes
  - It has memory logging probes at different stages of the Trainer - `init`, `train`, `evaluate`, `predict` 

##### NOTE:
When in distributed mode, the Trainer will only log the rank 0 memory.

#### Deciphering the Memory Metrics
When the memory metrics are logged, additional metrics are included in the output of `Trainer.train / Trainer.evaluate / Trainer.predict`. 

The additional metrics consist of:
A. absolute memory before trainer initialization
B. delta of allocated memory of a stage
C. peaked_delta of a stage (extra mem consumed and freed in between) 

B & C are taken before/after each of the following stages:
- <init>: `HFTrainer.__init__`
- <train>: `HFTrainer.train` (only if called)
- <evaluate>: `HFTrainer.evaluate` (only if called)
- <predict>: `HFTrainer.predict` (only if called)

Example Output:

```
output_metrics = {
    'train_runtime': 143.6704, 
    'train_samples_per_second': 0.835, 
    'train_steps_per_second': 0.209, 
    'train_tokens_per_second': 3421.164, 
    'train_loss': 1.292790667215983, 
    'init_mem_cpu_alloc_delta': 8192, 
    'init_mem_gpu_alloc_delta': 0, 
    'init_mem_cpu_peaked_delta': 0, 
    'init_mem_gpu_peaked_delta': 0, 
    'train_mem_cpu_alloc_delta': 531410944, 
    'train_mem_gpu_alloc_delta': 126487040, 
    'train_mem_cpu_peaked_delta': 0, 
    'train_mem_gpu_peaked_delta': 11016635392, 
    'before_init_mem_cpu': 5831098368, 
    'before_init_mem_gpu': 4747206656, 
    'epoch': 0.04
}
```

To compute the total GPU memory allocated for training
- Total memory delta for train stage = 126487040 bytes ('train_mem_gpu_alloc_delta') + 11016635392 bytes ('train_mem_gpu_peaked_delta')
- Memory consumption for training = 4747206656 bytes ('before_init_mem_gpu') + 0 bytes ('init_mem_gpu_alloc_delta') + 0 bytes ('init_mem_gpu_peaked_delta') + 126487040 bytes ('train_mem_gpu_alloc_delta') + 11016635392 bytes ('train_mem_gpu_peaked_delta')

#### No Significant Slowdown Using HF Memory Probes 
| acceleration type         | model_name_or_path                       | num_gpus | batch size | throughput without mem probs (toks/s) | throughput with mem probs (toks/s) | allocated gpu memory (GiB) |
| ------------------------- | ---------------------------------------- | -------- | ---------- | -------------------------- | ---------------------------------- | -------------------------- |
| accelerated-peft-bnb      | mistralai/Mistral-7B-v0.1                | 1        | 4          | 3385                       | 3451                               | 15.9                       |
| accelerated-peft-bnb      | mistralai/Mistral-7B-v0.1                | 1        | 8          | 3433                       | 3508                               | 26.9                       |
| accelerated-peft-bnb      | mistralai/Mistral-7B-v0.1                | 2        | 2          | 3022                       | 2941                               | 10.0                       |
| accelerated-peft-bnb      | mistralai/Mistral-7B-v0.1                | 2        | 4          | 3315                       | 3319                               | 16.6                       |
| accelerated-peft-bnb      | mistralai/Mixtral-8x7B-Instruct-v0.1     | 1        | 4          | 1793                       | 1781                               | 36.2                       |
| accelerated-peft-bnb      | mistralai/Mixtral-8x7B-Instruct-v0.1     | 1        | 8          | 1900                       | 1917                               | 47.2                       |
| accelerated-peft-bnb      | mistralai/Mixtral-8x7B-Instruct-v0.1     | 2        | 2          | 1500                       | 1454                               | 21.9                       |
| accelerated-peft-bnb      | mistralai/Mixtral-8x7B-Instruct-v0.1     | 2        | 4          | 1731                       | 1726                               | 29.4                       |
| accelerated-peft-bnb      | NousResearch/Llama-2-70b-hf              | 1        | 4          | 445                        | 458                                | 68.2                       |
| accelerated-peft-bnb      | NousResearch/Llama-2-70b-hf              | 1        | 8          | OOM                        | NaN                                | 0.0                        |
| accelerated-peft-bnb      | NousResearch/Llama-2-70b-hf              | 2        | 2          | 422                        | 425                                | 46.5                       |
| accelerated-peft-bnb      | NousResearch/Llama-2-70b-hf              | 2        | 4          | OOM                        | NaN                                | 0.0                        |
| accelerated-peft-autogptq | TheBloke/Mistral-7B-v0.1-GPTQ            | 1        | 4          | 3386                       | 3422                               | 15.9                       |
| accelerated-peft-autogptq | TheBloke/Mistral-7B-v0.1-GPTQ            | 1        | 8          | 3442                       | 3494                               | 26.9                       |
| accelerated-peft-autogptq | TheBloke/Mistral-7B-v0.1-GPTQ            | 2        | 2          | 2988                       | 2780                               | 11.4                       |
| accelerated-peft-autogptq | TheBloke/Mistral-7B-v0.1-GPTQ            | 2        | 4          | 3286                       | 3259                               | 17.7                       |
| accelerated-peft-autogptq | TheBloke/Mixtral-8x7B-Instruct-v0.1-GPTQ | 1        | 4          | 1854                       | 1866                               | 35.5                       |
| accelerated-peft-autogptq | TheBloke/Mixtral-8x7B-Instruct-v0.1-GPTQ | 1        | 8          | 1949                       | 1969                               | 46.5                       |
| accelerated-peft-autogptq | TheBloke/Mixtral-8x7B-Instruct-v0.1-GPTQ | 2        | 2          | 1619                       | 1556                               | 31.6                       |
| accelerated-peft-autogptq | TheBloke/Mixtral-8x7B-Instruct-v0.1-GPTQ | 2        | 4          | 1821                       | 1812                               | 38.4                       |
| accelerated-peft-autogptq | TheBloke/Llama-2-70b-GPTQ                | 1        | 4          | 451                        | 465                                | 65.9                       |
| accelerated-peft-autogptq | TheBloke/Llama-2-70b-GPTQ                | 1        | 8          | OOM                        | NaN                                | 0.0                        |
| accelerated-peft-autogptq | TheBloke/Llama-2-70b-GPTQ                | 2        | 2          | 438                        | 437                                | 61.8                       |
| accelerated-peft-autogptq | TheBloke/Llama-2-70b-GPTQ                | 2        | 4          | OOM                        | NaN                                | 0.0                        |
