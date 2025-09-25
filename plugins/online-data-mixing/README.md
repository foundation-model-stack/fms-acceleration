# Online Data Mixing

This library contains plugin for online dynamic reward (learnable) based data mixing framework that operates on dynamically mixing datasets online during training while being adapted based on the signals (e.g. training loss, gradnorm etc) from training.

## Plugins

Plugin | Description | Depends | Loading | Augmentation | Callbacks
--|--|--|--|--|--
[odm](./src/fms_acceleration_odm/framework_plugin_odm.py) | OnlineMixingDataset PyTorch IterableDataset and custom rewards | | ✅ | ✅ | ✅

## Design
![](./artifacts/Design.png)

## Usage in Custom Training Loop


### Planned TODOs
Please see issue [#153](https://github.com/foundation-model-stack/fms-acceleration/issues/153).



