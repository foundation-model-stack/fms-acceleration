#!/usr/bin/env bash

set -x

echo "FMS Acceleration Benchmarking Script"
echo "Please run this script as "
echo "bash scripts/run_benchmarks.sh ... from the root of the repo"

# TODO: this can be improved. For now we assume we always run from 
# root of repo
WORKING_DIR=scripts/benchmarks

# pointing to the configuration directory of the repo
CONFIG_DIR=sample-configurations

# ------------- MAIN CONFIGS -----------------
SCENARIOS_CONFIG=scenarios.yaml
DEFAULTS_CONFIG=defaults.yaml
ACCELERATE_CONFIG=accelerate.yaml

# ------------- FRAMEWORK CONFIGS -----------------

# list down the framework configuration files
CONFIG_PEFT_AUTOGPTQ=$CONFIG_DIR/accelerated-peft-autogptq-sample-configuration.yaml

# list down the framework config tags inside scenarios.yaml
CONFIGTAG_PEFT_AUTOGPTQ=accelerated-peft-autogptq

# ------------- SCENARIO CONFIGS -----------------
SCNTAG_PEFT_AUTOGPTQ=accelerated-peft-gptq

# ------------- OTHER CONFIGS -----------------

# data will be cached in here
DATA_CACHE=data/cache.json

# env inputs
DRY_RUN=${DRY_RUN:-"false"}
NO_DATA_PROCESSING=${NO_DATA_PROCESSING:-"false"}

# inputs
NUM_GPUS_MATRIX=${1-"1 2"}
RESULT_DIR=${2:-"benchmark_outputs"}
SCENARIOS_CONFIG=${3:-$SCENARIOS_CONFIG}
SCENARIOS_FILTER=${4-$SCNTAG_PEFT_AUTOGPTQ}

echo "NUM_GPUS_MATRIX: $NUM_GPUS_MATRIX"
echo "RESULT_DIR: $RESULT_DIR"
echo "SCENARIOS_CONFIG: $SCENARIOS_CONFIG"
echo "SCENARIOS_FILTER: $SCENARIOS_FILTER"

# tag on the directories
SCENARIOS_CONFIG=$WORKING_DIR/$SCENARIOS_CONFIG
DEFAULTS_CONFIG=$WORKING_DIR/$DEFAULTS_CONFIG
ACCELERATE_CONFIG=$WORKING_DIR/$ACCELERATE_CONFIG
DATA_CACHE=$RESULT_DIR/$DATA_CACHE

# ------------- EXTRA ARGS -----------------

EXTRA_ARGS=""

if [ ! -z "$SCENARIOS_FILTER" ]; then 
    EXTRA_ARGS="$EXTRA_ARGS --run_only_scenarios $SCENARIOS_FILTER"
fi

if [ "$DRY_RUN" = "true" ]; then 
    EXTRA_ARGS="$EXTRA_ARGS --dry_run"
fi

if [ "$NO_DATA_PROCESSING" = "true" ]; then 
    EXTRA_ARGS="$EXTRA_ARGS --no_data_processing"
fi

# run the bench
python $WORKING_DIR/benchmark.py \
   --num_gpus $NUM_GPUS_MATRIX \
   --acceleration_framework_config_keypairs \
       $CONFIGTAG_PEFT_AUTOGPTQ $CONFIG_PEFT_AUTOGPTQ \
   --scenarios_config_path $SCENARIOS_CONFIG \
   --accelerate_config $ACCELERATE_CONFIG \
   --defaults_config_path $DEFAULTS_CONFIG \
   --dataset_save_path $DATA_CACHE \
   --results_output_path $RESULT_DIR $EXTRA_ARGS
