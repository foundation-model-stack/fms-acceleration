# FMS Acceleration for Mixture-of-Experts

This library contains plugins to accelerate finetuning with the following optimizations:
1. Expert-Parallel MoE with Triton Kernels from ScatterMoe (& Megablocks)

## Plugins

Plugin | Description | Depends | Loading | Augmentation | Callbacks
--|--|--|--|--|--
[scattermoe](./src/fms_acceleration_moe/framework_plugin_scattermoe.py) | MoE Expert Parallel with Triton Kernels from scattermoe (& megablocks) | scattermoe / megablocks | ✅ | |  ✅


## Adding New Models

Our `ScatterMoe` implementation is a module-swap; to add new models we need to update the specifications in [scattermoe_constants.py](./src/fms_acceleration_moe/utils/scattermoe_constants.py).
- See the code documentation within to understand how to add new models.

## Running Benchmarks


Run the below in the top-level directory of this repo:
- the `scattermoe` dep is not included by default, so the `-x` switch installs it.

```
tox -e run-benches \
    -x testenv:run-benches.deps+="-r plugins/accelerated-moe/requirements-khd.txt" \
    -- \
    "1 2 4" 128 benchmark_outputs scenarios-moe.yaml accelerated-moe-scatter
```
or run the larger `Mixtral-8x7B` bench:
```
tox ... \
    8 128 benchmark_outputs scenarios-moe.yaml accelerated-moe-scatter-mixtral
```

NOTE: if `FileNotFoundError` is observed on the *triton cache*, similar to issues like these:
- https://github.com/triton-lang/triton/issues/2688

then somehow `tox` is causing problems with triton and multiprocessing (there is some race condition).
But the workaound is to first *activate the tox env* and 
running in `bash`:
```
# if FileNotFoundError in the triton cache is observed
# - then activate the env and run the script manually

source .tox/run-benches/bin/activate
bash scripts/run_benchmarks.sh \
    ....
```


### Triton Kernel Dependencies

Currently we do not copy the `scattermoe` kernels into this respository, to this is an additional manual install:

```
# this will install the kernel-hyperdrive fork with the scattermoe triton kernels
pip install -r requirements-khd.txt
```

### Known Issues

These are currently some known issues not yet resolved:
- The design currently does a swap for the mixture-of-expert module with [ScatterMoE](./src/fms_acceleration_moe/utils/scattermoe.py). This affects the `state_dict` of the model, so any saved checkpoint may need to be converted back to original.
- should eventually remove the dependency on an external `kernel-hyperdrive` repository.
- now support only loading *sharded* `safetensor` non-GGUF MoE checkpoints. This is a reasonable assumption since MoE checkpoints are typically above the size limit that prevents it being saved into a single checkpoint filed.
- currently only supports `StateDictType.SHARDED_STATE_DICT` because the implementation uses `DTensors` which have limited support for full state dicts. However for efficiency considerations, sharded state dicts are the most efficient. 


