# FMS Acceleration for Mixture-of-Experts

This library contains plugins to accelerate finetuning with the following optimizations:
1. Expert-Parallel MoE with Triton Kernels from ScatterMoe (& Megablocks)

## Plugins

Plugin | Description | Depends | Loading | Augmentation | Callbacks
--|--|--|--|--|--
[scattermoe](./src/fms_acceleration_moe/framework_plugin_scattermoe.py) | MoE Expert Parallel with Triton Kernels from scattermoe (& megablocks) | scattermoe / megablocks | ✅ | |  ✅


## Running Benchmarks


Run the below in the top-level directory of this repo:
- the `scattermoe` dep is not included by default, so the `-x` switch installs it.

```
tox -e run-benches \
    -x testenv:run-benches.deps+="-r plugins/accelerated-moe/requirements-khd.txt" \
    -- \
    "1 2 4 8" 128 benchmark_outputs scenarios-granite.yaml accelerated-moe-scatter
```
or run the larger `Mixtral-8x7B` bench:
```
tox ... \
    8 128 benchmark_outputs scenarios-granite.yaml accelerated-moe-scatter
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
