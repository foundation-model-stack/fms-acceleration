# FMS Acceleration for Mixture-of-Experts

This library contains plugins to accelerate finetuning with the following optimizations:
1. Expert-Parallel MoE with Megablocks

## Plugins

Plugin | Description | Depends | Loading | Augmentation | Callbacks
--|--|--|--|--|--
[megablocks](./src/fms_acceleration_moe/framework_plugin_megablocks.py) | MoE Expert Parallel with megablocks | megablocks | ✅ | |  ✅


## Running Benchmarks

```
tox -e run-benches -- 8 8 scenarios.yaml accelerated-moe-megablocks
```

## Expert-Parallel MoE with Megablocks

Not all of the features of `megablocks` are being incorporated; listing down some of the restrictions of the current integration:
- curretnly not passing the data parallel `dp_mesh` to the `FSDP` constructor, so `FSDP` will always shard over the default process group (over world_size).
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
pip install -r requirements_mb.txt
```