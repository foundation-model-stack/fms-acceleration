def ignore_modules_in_fsdp(modules, fsdp_plugin):
    '''
    This function installs hooks on the target adapter parameters and 
    reduces the accumulated gradients across devices
    '''
    import torch.distributed as dist    

    fsdp_plugin.ignored_modules = modules 

    def _all_reduce_hook(grad):
        if grad is not None:
            grad = grad.contiguous()
            dist.all_reduce(grad, op=dist.ReduceOp.AVG, group=None)
        return grad

    for mod in modules: 
        # install hooks on the adapters
        mod.lora_A.default.weight.register_hook(_all_reduce_hook)
        mod.lora_B.default.weight.register_hook(_all_reduce_hook)
