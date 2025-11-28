# Context Parallel for Mamba Kernels

This library contains plugin for applying context parallelism for mamba module (mamba_ssm).

## Plugins

Plugin | Description | Depends | Loading | Augmentation | Callbacks
--|--|--|--|--|--
[mcp](./src/fms_acceleration_mcp/framework_plugin_mcp.py) | context parallel for mamba | [custom mamba cp implementation](https://github.com/garrett361/mamba/tree/mamba-cp) | ✅ | ✅ | ✅

## Mamba CP Implementation

Context parallel implementation is taken from a custom [mamba_ssm repo](https://github.com/garrett361/mamba/tree/mamba-cp) with cp implemenation. Thus, its required this repo is installed to use this plugin.

## Known Issues
1. load balancing is removed given limited support on mamba cp implementation. This could lead to potential throughput drops for trainings using causal mask.
