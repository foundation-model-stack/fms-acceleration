# FMS Acceleration Framework Library

This contains the library code for
- `AccelerationFramework`
- `AccelerationPlugin`

The purpose of this library is to:
- provide a single integration point into [Huggingface](https://github.com/huggingface/transformers).
- manage `AccelerationPlugin` in a flexible manner. 
- read and load plugins from a single configuration YAML. Enforce compatiblity rules on how plugins can be combined.

See the following resources
- Instructions on [how to run with `fms-hf-tuning`](https://github.com/foundation-model-stack/fms-acceleration/README.md)
- Sample [plugin configurations](https://github.com/foundation-model-stack/fms-acceleration/sample-configurations).

## Using AccelerationFramework with HF Trainer

First instantiate an `AccelerationFramework` object by passing a YAML configuration, say via a `path_to_config`:
```python
from fms_acceleration import AccelerationFramework
framework = AccelerationFramework(path_to_config)
```

Plugins automatically configured based on configuration; for more details on how plugins are configured, [see below](#configuration-of-plugins).

Some plugins may require custom model loaders (in replacement of the typical `AutoModel.from_pretrained`). In this case, call `framework.model_loader`:

```python
model = framework.model_loader(model_name_or_path, ...)
```
E.g., in the GPTQ example, see [plugin configurations](https://github.com/foundation-model-stack/fms-acceleration/sample-configurations), we require `model_name_or_path` to be custom loaded from a quantized checkpoint.

We provide a flag `framework.requires_custom_loading` to check if plugins require custom loading.

Also some plugins will require the model to be augmented, e.g., replacing layers with plugin-compliant PEFT adapters.  In this case:

```python
# will also take in some other configs that may affect augmentation
# some of these args may be modified due to the augmentation
# e.g., peft_config will be consumed in augmentation, and returned as None 
#       to prevent SFTTrainer from doing extraneous PEFT logic
model, (peft_config,) = framework.augmentation(
    model, 
    train_args, modifiable_args=(peft_config,),
)
```

We also provide `framework.requires_agumentation` to check if augumentation is required by the plugins.

Finally pass the model to train:

```python
# e.g. using transformers.Trainer. Pass in model (with training enchancements)
trainer = Trainer(model, ...)

# call train
trainer.train()
```

Thats all! the model will not be reap all acceleration speedups based on the plugins that were installed!

## Configuration of Plugins

Each [package](#packages) in this monorepo:
- can be *independently installed*. Install only the libraries you need:
   ```shell
   pip install fms-acceleration/libs/peft
   pip install fms-acceleration/libs/unsloth
   ```
- can be *independently configured*. Each plugin is registed under a particular configuration path. E.g., the [autogptq plugin](libs/peft/src/fms_accelerate_peft/framework_plugin_autogptq.py) is reqistered under the config path `peft.quantization.auto_gptq`.
    ```python
    AccelerationPlugin.register_plugin(
        AutoGPTQAccelerationPlugin,
        configuration_and_paths=["peft.quantization.auto_gptq"], 
    )
    ```

    This means that it will be configured under theat exact stanza:
    ```yaml
    plugins:
        peft:
            quantization:
                auto_gptq:
                    # everything under hear will be passed to plugin 
                    # when instantiating
                    ...
    ```

- When instantiating `fms_acceleration.AccelerationFramework`, it internally parses through the configuration stanzas. For plugins that are installed, it will instantiate them; for those that are not, it will simply *passthrough*.
- `AccelerationFramework` will manage plugins transparently for user. User only needs to call `AccelerationFramework.model_loader` and `AccelerationFramework.augmentation`.
