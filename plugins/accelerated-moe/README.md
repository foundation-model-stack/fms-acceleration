# FMS Acceleration for Mixture-of-Experts

This library contains plugins to accelerate finetuning with the following optimizations:
1. Expert-Parallel MoE with Megablocks

## Plugins

Plugin | Description | Depends | Loading | Augmentation | Callbacks
--|--|--|--|--|--
[megablocks](./src/fms_acceleration_moe/framework_plugin_megablocks.py) | MoE Expert Parallel with megablocks | megablocks | ✅ | |  ✅


## Running Benchmarks

See the benchmarks [a100_80gb_mb.csv](../../scripts/benchmarks/refs/a100_80gb_mb.csv)


Run the below in the top-level directory of this repo:
- the `megablocks` dep is not included by default, so the `-x` switch installs it.

```
tox -e run-benches \
    -x testenv:run-benches.deps+="-r plugins/accelerated-moe/requirements-mb.txt" \
    -- \
    8 8 benchmark_outputs scenarios.yaml accelerated-moe-megablocks

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
    8 8 benchmark_outputs scenarios.yaml accelerated-moe-megablocks
```


## Expert-Parallel MoE with Megablocks

Not all of the features of `megablocks` are being incorporated; listing down some of the restrictions of the current integration:
- currently not passing the data parallel `dp_mesh` to the `FSDP` constructor, so `FSDP` will always shard over the default process group (over world_size).
- now support only loading *sharded* `safetensor` non-GGUF MoE checkpoints. This is a reasonable assumption since MoE checkpoints are typically above the size limit that prevents it being saved into a single checkpoint filed.
- only supports the *dropless sparse* MLPs in the megablocks package; the other variations like non-dropless and grouped computes are not currently integrated.
- the `shard_moe` may not scale well with larger models as the current implementation `torch.concat` all the expert weights together before passing to `torch.distributed` to be sharded. This is redundently done in all devices, so it is inefficient.
- currently only supports `StateDictType.SHARDED_STATE_DICT` because the implementation uses `DTensors` which have limited support for full state dicts. However for efficiency considerations, sharded state dicts are the most efficient. 

### Megablocks Dependencies

Currently databricks megablocks does not have a PyPi repository and no proper release, so we have to install directly from Github, refer to instructions below. 
- This has to be a manual install as PyPI will complain if included as an official plugin dependency.
- Since this is not a binary install, please note that CUDA Toolkit will be required to build some of the kernels used by megablocks.

```
# this will install the megablocks from Github
# megablocks requires CUDA Toolkit to build.
pip install -r requirements-mb.txt
```