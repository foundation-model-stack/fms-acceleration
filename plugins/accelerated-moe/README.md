# FMS Acceleration for Mixture-of-Experts

This library contains plugins to accelerate finetuning with the following optimizations:
1. Expert-Parallel MoE with Megablocks


## Megablocks Dependencies

Currently databricks megablocks does not have a PyPi repository and does not have a proper release, so we have to install from the github repository as below. Please note that installing from github will require CUDA Toolkit to build.

```
pip install git+https://github.com/databricks/megablocks.git@bce5d7b2aaf5038bc93b36f76c2baf51c2939bd2
```