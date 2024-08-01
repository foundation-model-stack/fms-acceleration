# FMS Acceleration for Instruct Lab

This library contains plugins to accelerate finetuning with the following optimizations:

1. Padding-Free Flash Attention Computation


## Plugins

Plugin | Description | Depends | Loading | Augmentation | Callbacks
--|--|--|--|--|--
[padding_free](./src/fms_acceleration_ilab/framework_plugin_padding_free.py) | Padding-Free Flash Attention Computation | flash_attn | ✅ | ✅


## Native Transformers Support from V4.43.0
Transformers natively supports padding-free from v4.43.0. The padding-free plugin will use the transformers library if compatible, 
otherwise if `transformers < V4.43.0` the plugin will use an internal implementation instead.

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
