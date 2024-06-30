# loss patcher
import torch
from torch.distributed import all_reduce, ReduceOp
from accelerate import Accelerator
from types import MethodType
from contextlib import contextmanager

# patch the accumulate to provide scaling of loss
def patch_loss_via_accmulate(
    accelerator: Accelerator, 
):

    assert torch.distributed.is_initialized(), "only works for distribtued"
    n = torch.distributed.get_world_size()
    r = torch.distributed.get_rank()

    @contextmanager
    def accumulate(self, *models):

        assert len(models) == 1, "only supports single model"
        model = models[0]
        _old_forward = model.forward

        # with per token loss
        def forward(self, **kwargs):
            num_loss_counted_tokens = kwargs.pop("num_loss_counted_tokens")
            outputs = _old_forward(**kwargs)

            V = torch.zeros(1, dtype=torch.float32).to(r)
            V[0] = num_loss_counted_tokens
            all_reduce(V, op=ReduceOp.SUM)
            scaling = n * num_loss_counted_tokens / V[0]
            del V
            outputs.loss = outputs.loss * scaling
            return outputs

        # patch
        model.forward = MethodType(forward, model)
        yield old_accumulate(model)
        model.forward = _old_forward

    # patch accumulate
    old_accumulate = accelerator.accumulate
    accelerator.accumulate = MethodType(accumulate, accelerator)
