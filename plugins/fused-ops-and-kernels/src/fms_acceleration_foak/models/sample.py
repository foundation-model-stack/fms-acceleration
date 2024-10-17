import torch


FIFO = []

def build_lm_head_forward(vocab_size: int):
    def lm_head_forward(self, hidden_states: torch.Tensor):
        shift_hidden_states = hidden_states[..., :-1, :].contiguous()
        shift_hidden_states = shift_hidden_states.view(-1, shift_hidden_states.shape[-1])
        logits = torch.empty(1, 2, vocab_size, device='cpu')
        FIFO.append(shift_hidden_states)
        return logits
    return lm_head_forward

from ..kernels.liger.fused_linear_cross_entropy_loss import LigerFusedLinearCrossEntropyFunction
    

from torch.nn import CrossEntropyLoss

def build_fused_cross_entropy_class(model: torch.nn.Module):

    lm_head = model.lm_head
    logits_scaling = (
        None if not hasattr(model.config, "logits_scaling")
        else model.config.logits_scaling
    )

    class LigerFusedLinearCrossEntropyLoss(CrossEntropyLoss):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

        def forward(self, logits, target, bias=None):

            hidden_states = FIFO.pop(0)

            return LigerFusedLinearCrossEntropyFunction.apply(
                (
                    hidden_states / logits_scaling if 
                    logits_scaling is not None else
                    hidden_states
                ),
                lm_head.weight,
                target.to(hidden_states.device),
                bias,
                self.ignore_index,
                self.label_smoothing,
                self.reduction,
            )

    return LigerFusedLinearCrossEntropyLoss