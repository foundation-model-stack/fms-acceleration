#!/usr/bin/env bash

echo "FMS Acceleration Benchmarking Script"
echo "Please run this script as "
echo "bash scripts/run_benchmarks.sh ... from the root of the repo"

# inputs
NUM_GPUS_MATRIX=${1-"1 2"}
RESULT_DIR=${2:-"benchmark_outputs"}

echo "NUM_GPUS_MATRIX: $NUM_GPUS_MATRIX"
echo "RESULT_DIR: $RESULT_DIR"

# TODO: this can be improved. For now we assume we always run from 
# root of repo
WORKING_DIR=scripts/benchmarks

# pointing to the configuration directory of the repo
CONFIG_DIR=sample-configurations

# ------------- MAIN CONFIGS -----------------
SCENARIOS_CONFIG=$WORKING_DIR/scenarios.yaml
DEFAULTS_CONFIG=$WORKING_DIR/defaults.yaml
ACCELERATE_CONFIG=$WORKING_DIR/accelerate.yaml

# ------------- FRAMEWORK CONFIGS -----------------

# list down the framework configuration files
CONFIG_PEFT_AUTOGPTQ=$CONFIG_DIR/accelerated-peft-autogptq-sample-configuration.yaml

# list down the framework config tags inside scenarios.yaml
CONFIGTAG_PEFT_AUTOGPTQ=accelerated-peft-autogptq

# ------------- SCENARIO CONFIGS -----------------
SCNTAG_PEFT_AUTOGPTQ=accelerated-peft-gptq

# ------------- OTHER CONFIGS -----------------

# data will be cached in here
DATA_CACHE=$RESULT_DIR/data/cache.json

# run the bench
python $WORKING_DIR/benchmark.py \
   --num_gpus $NUM_GPUS_MATRIX \
   --acceleration_framework_config_keypairs \
       $CONFIGTAG_PEFT_AUTOGPTQ $CONFIG_PEFT_AUTOGPTQ \
   --scenarios_config_path $SCENARIOS_CONFIG \
   --accelerate_config $ACCELERATE_CONFIG \
   --run_only_scenarios $SCNTAG_PEFT_AUTOGPTQ \
   --defaults_config_path $DEFAULTS_CONFIG \
   --dataset_save_path $DATA_CACHE \
   --results_output_path $RESULT_DIR

