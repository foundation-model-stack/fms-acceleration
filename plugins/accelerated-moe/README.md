# FMS Acceleration for Mixture-of-Experts

This library contains plugins to accelerate finetuning with the following optimizations:
1. Expert-Parallel MoE with Megablocks

## Known Issues with Megablocks Plugin Implementation

Known Issues
- Currently we do not pass the data parallel `dp_mesh` to the `FSDP` constructor, so `FSDP` will always shard over the default process group (over world_size).
- Currently only supports loading `safetensor` MoE checkpoints.


## Megablocks Dependencies

Currently databricks megablocks does not have a PyPi repository and does not have a proper release, so we have to install from the github repository as below. Please note that installing from github will require CUDA Toolkit to build.

```
pip install -r requirements_mb.txt
```