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

# ------------- SCENARIO CONFIGS -----------------
# this determines which is the default subset
SCNTAG_PEFT_AUTOGPTQ=accelerated-peft-gptq

# ------------- OTHER CONFIGS -----------------

# data will be cached in here
DATA_CACHE=data/cache.json

# final result placed here
BENCH_RESULT_FILE=benchmarks.csv

# env inputs
DRY_RUN=${DRY_RUN:-"false"}
NO_DATA_PROCESSING=${NO_DATA_PROCESSING:-"false"}
NO_OVERWRITE=${NO_OVERWRITE:-"false"}

# inputs
NUM_GPUS_MATRIX=${1-"1 2"}
RESULT_DIR=${2:-"benchmark_outputs"}
SCENARIOS_CONFIG=${3:-$SCENARIOS_CONFIG}
SCENARIOS_FILTER=${4-$SCNTAG_PEFT_AUTOGPTQ}

echo "NUM_GPUS_MATRIX: $NUM_GPUS_MATRIX"
echo "RESULT_DIR: $RESULT_DIR"
echo "SCENARIOS_CONFIG: $SCENARIOS_CONFIG"
echo "SCENARIOS_FILTER: $SCENARIOS_FILTER"

if [ -n "$RESULT_DIR" ]; then
    echo "The results directory is not empty. "
    if [ "$NO_OVERWRITE" = "true" ]; then 
        echo "Results dir $RESULT_DIR is not empty, but NO_OVERWRITE=true"
        echo "If intending to overwrite please delete the folder manually"
        echo "or do not set NO_OVERWRITE"
        exit 1
    fi
    echo "Deleting $RESULT_DIR"
    rm -rf $RESULT_DIR
fi

# tag on the directories
SCENARIOS_CONFIG=$WORKING_DIR/$SCENARIOS_CONFIG
DEFAULTS_CONFIG=$WORKING_DIR/$DEFAULTS_CONFIG
ACCELERATE_CONFIG=$WORKING_DIR/$ACCELERATE_CONFIG
DATA_CACHE=$RESULT_DIR/$DATA_CACHE
BENCH_RESULT_FILE=$RESULT_DIR/$BENCH_RESULT_FILE

# ------------- EXTRA ARGS -----------------

# preload models by default
EXTRA_ARGS="--preload_models"

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
   --scenarios_config_path $SCENARIOS_CONFIG \
   --accelerate_config $ACCELERATE_CONFIG \
   --defaults_config_path $DEFAULTS_CONFIG \
   --dataset_save_path $DATA_CACHE \
   --results_output_path $RESULT_DIR $EXTRA_ARGS

# produce the final CSV for checkin
# need to set PYTHONPATH because there is an import inside
# this will write to the BENCH_RESULT_FILE
PYTHONPATH=. \
    python $WORKING_DIR/display_bench_results.py benchmark_outputs \
    --result_file $BENCH_RESULT_FILE
