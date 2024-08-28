#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export ACCELERATION_FRAMEWORK_CONFIG_FILE=/workspace/fms-acceleration/scripts/benchmarks/../../sample-configurations/moe-megablocks-sample-configuration.yaml
# accelerate launch --config_file scripts/benchmarks/accelerate.yaml --num_processes=8 --main_process_port=29500 -m tuning.sft_trainer --model_name_or_path mistralai/Mixtral-8x7B-Instruct-v0.1 --packing False --max_seq_len 4096 --training_data_path benchmark_outputs/data/cache_all.json --use_flash_attn True --response_template '
# ### Response:' --dataset_text_field output --include_tokens_per_second True --num_train_epochs 1 --gradient_checkpointing True --evaluation_strategy no --save_strategy no --weight_decay 0.01 --warmup_steps 10 --lr_scheduler_type linear --logging_strategy steps --max_steps 100 --learning_rate 5e-5 --torch_dtype bfloat16 --accelerator_config scripts/benchmarks/accelerator-config.json --gradient_accumulation_steps 16 --logging_steps 1 --adam_epsilon 1e-8 --per_device_train_batch_size 1 --output_dir benchmark_outputs/exp_1/hf --skip_memory_metrics True


# deepspeed
# - need to turn on MP or the forward datatype will be wrong
TRANSFORMERS_VERBOSITY=info \
accelerate launch --config_file scripts/benchmarks/accelerate-ds.yaml \
    --num_processes=8 --main_process_port=29500 -m tuning.sft_trainer --model_name_or_path mistralai/Mixtral-8x7B-Instruct-v0.1 --packing False --max_seq_len 4096 --training_data_path benchmark_outputs/data/cache_all.json --use_flash_attn True --response_template '
### Response:' --dataset_text_field output --include_tokens_per_second True --num_train_epochs 1 --gradient_checkpointing True --evaluation_strategy no --save_strategy no --weight_decay 0.01 --warmup_steps 10 --lr_scheduler_type linear --logging_strategy steps --max_steps 100 --learning_rate 5e-5 --torch_dtype bfloat16 --accelerator_config scripts/benchmarks/accelerator-config.json --gradient_accumulation_steps 16 --logging_steps 1 --adam_epsilon 1e-8 --per_device_train_batch_size 1 --output_dir benchmark_outputs/exp_1/hf --skip_memory_metrics True --bf16