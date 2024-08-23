# FMS Acceleration for Attention And Distributed Packing Plugin

This library contains plugins to accelerate finetuning with the following optimizations:

1. Padding-Free Flash Attention Computation
2. Multipack Distributed Sampling


## Plugins

Plugin | Description | Depends | Loading | Augmentation | Callbacks
--|--|--|--|--|--
[padding_free](./src/fms_acceleration_aadp/framework_plugin_padding_free.py) | Padding-Free Flash Attention Computation | flash_attn | | ✅ | 
[multipack sampler](./src/fms_acceleration_aadp/framework_plugin_multipack.py) | Multipack Distributed Sampling | numba | | ✅ | 


## Native Transformers Support from v4.44.0
Transformers natively supports padding-free from v4.44.0 [see here](https://github.com/huggingface/transformers/pull/31629). The padding-free plugin will use the transformers library if compatible, 
otherwise if `transformers < v4.44.0` the plugin will use an internal implementation instead.

## Running Benchmarks

To reproduce the benchmarks, simply run the following commands,

Reproduce [Padding Free on A100 80GB](scripts/benchmarks/refs_orca/a100_80gb_pf.csv)
`bash scripts/run_benchmarks.sh "1 2" "4 8" benchmark_outputs scenarios-orca.yaml "none"`

Reproduce [MultiPack on A100 80GB](scripts/benchmarks/refs_orca/a100_80gb_mp.csv)
`bash scripts/run_benchmarks.sh "2 4 8" "16 32 64" benchmark_outputs scenarios-orca.yaml "padding-free"`

## Known Issues

### Currently Only Supports Pre-Tokenized Dataset

The padding-free plugin currently only works with pre-tokenized datasets, this is because it is currently designed to replace 
the data collator from `SFTTrainer` with a custom data collator to manipulate the input to the modified flash attention forward. 

There are some cases, the data collator for SFTTrainer will handle the formatting and tokenization from raw text datasets. The plugin
is currently unable to both handle the original data collation and apply its custom data collator over it as the same time. This issue 
will be addressed in a future commit to support this case. 

In the meantime, the plugin expects the user to provide a pretokenized dataset that
- is formatted with a template for instruct-tuning cases
- is tokenized
- has template labels that are masked to exclude from loss computation
- has eos token appended

### Currenly Only Supports Multipack with Padding-Free

The multipack plugin currently also requires the padding-free plugin to work.
This may change in the future if there is demand for multipack to work standalone without padding free.

