# Sample FMS Acceleration Configurations

This directory contains the following sample configurations for different [plugin](../README.md#plugins).

## Sample Acceleration Configurations

The configurations are given short names as follows; these names follow the names under `framework_config` in the 
[benchmark scenarios](../scripts/benchmarks/scenarios.yaml).
- [accelerated-peft-autogptq](../sample-configurations/accelerated-peft-autogptq-sample-configuration.yaml)
- [accelerated-peft-bnb](../sample-configurations/accelerated-peft-bnb-nf4-sample-configuration.yaml)


Configuration | Plugins
--|--
accelerated-peft-autogptq | accelerated_peft
accelerated-peft-bnb | accelerated_peft

## FMS SFT_Trainer Arguments

Each configuration also requires the appropriate args passed to [fms-hf-tuning](https://github.com/foundation-model-stack/fms-hf-tuning).
Our benchmark suite also serves to document these arguments in the [YAML here](../scripts/benchmarks/scenarios.yaml).

Each scenario is documented for a particular
- `framework_config`: points to some [acceleration configuration](#sample-acceleration-configurations).
- `arguments`: contains a dictionary of arguments to be passed to `sft_trainer.py`.

The `arguments` map to arguments used in `sft_trainer.py`; note however `model_name_or_path` is an array, because that is our way of documenting some models we test for. Any of these models (and also others) should work. But do note that not all plugins work with all models; it depends on the `AccelerationFramework.restricted_model_archs` settings.

An example for `accelerated-peft-gptq`.

```yaml
scenarios:

  - name: accelerated-peft-gptq
    framework_config: 
      - accelerated-peft-autogptq
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