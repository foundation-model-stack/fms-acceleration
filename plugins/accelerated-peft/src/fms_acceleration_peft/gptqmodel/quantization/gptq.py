# License: GPTQModel/licenses/LICENSE.mit
# adapted from @qwopqwop200 's [GPTQ-for-LLaMa](https://github.com/qwopqwop200/GPTQ-for-LLaMa/tree/cuda), which itself is based on [gptq](https://github.com/IST-DASLab/gptq)

# Standard
from logging import getLogger
import math
import os
import time

# Third Party
import torch
import torch.nn as nn
import transformers
import transformers.models.granitemoe.modeling_granitemoe as MOE

# Local
from .quantizer import Quantizer

logger = getLogger(__name__)

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


class GPTQ:
    def __init__(self, layer):
        self.layer = layer
        self.dev = self.layer.weight.device
        self.quantizer = Quantizer()
        # print("GPTQ INIT: layer, type(layer)", layer, type(layer))
        # print("GPTQ INIT: layer.weight.data", layer.weight.data)


        if isinstance(layer, MOE.GraniteMoeParallelExperts):
            self.is_moe = True
            self.num_experts = layer.num_experts
            self.out_features = layer.output_size
            self.in_features = layer.input_size
            # print("GPTQ INIT: self.num_experts, self.out_features, self.in_features", self.num_experts, self.out_features, self.in_features)
            
            # Separate W for each expert
            self.W_list, self.H_list, self.nsamples_list = [], [], []
            for i in range(self.num_experts):
                # Each expert slice is of shape [out_features, in_features]
                self.W_list.append(layer.weight.data[i].clone())

                # For each expert param, we have a Hessian and sample count
                self.H_list.append(torch.zeros((self.in_features, self.in_features), device=self.dev))
                self.nsamples_list.append(0)
            
            # print("GPTQ INIT: self.W_list, len(self.W_list), type(self.W_list[0]), self.W_list[0].size(), self.W_list[0].dim()", self.W_list, len(self.W_list), type(self.W_list[0]), self.W_list[0].size(), self.W_list[0].dim())
            # print("GPTQ INIT: self.H_list, len(self.H_list), type(self.H_list[0]), self.H_list[0].size(), self.H_list[0].dim()", self.H_list, len(self.H_list), type(self.H_list[0]), self.H_list[0].size(), self.H_list[0].dim())

        else:
            # For 2D layer (linear, conv, etc.), we have a single Hessian and sample count
            self.is_moe = False
            W = layer.weight.data.clone()
            if isinstance(layer, nn.Conv2d):
                W = W.flatten(1)
            if isinstance(layer, transformers.pytorch_utils.Conv1D):
                W = W.t()
            # print("GPTQ INIT: W, type(W), W.size(), W.dim()", W, type(W), W.size(), W.dim())

            self.rows = W.shape[0]
            self.columns = W.shape[1]
            self.H = torch.zeros((self.columns, self.columns), device=self.dev)
            self.nsamples = 0
            # print("GPTQ INIT: self.H, type(self.H), self.H.size(), self.H.dim()", self.H, type(self.H), self.H.size(), self.H.dim())
            # print("GPTQ INIT: self.rows, self.columns", self.rows, self.columns)
        

    def add_batch(self, inp, out):
        if os.environ.get("DEBUG"):
            self.inp1 = inp
            self.out1 = out
        # Update entire H_list and nsamples_list
        if self.is_moe:
            # print("INSIDE ADD_BATCH FOR MOE: inp, type(inp), inp.shape", inp, type(inp), inp.shape)
            for expert_idx in range(self.num_experts):    
                H = self.H_list[expert_idx]
                nsamples = self.nsamples_list[expert_idx]
                
                # if len(inp.shape) == 2:
                #     inp = inp.unsqueeze(0)
                # tmp = inp.shape[0]
                # print("INSIDE ADD_BATCH FOR MOE 2: inp, inp.shape, tmp", inp, inp.shape, tmp)

                # Below is doing reverse of above
                # if len(inp.shape) == 3:
                #     inp = inp.reshape((-1, inp.shape[-1]))
                #     print("INSIDE ADD_BATCH FOR MOE 3: inp, inp.shape, tmp", inp, inp.shape, tmp)

                tmp = 1
                # len(inp.shape) == 2 in this case
                mod_inp = inp.t()
                # print("INSIDE ADD_BATCH FOR MOE 4: inp, inp.shape, tmp", inp, inp.shape, tmp)

                H *= nsamples / (nsamples + tmp)
                nsamples += tmp
                mod_inp = math.sqrt(2 / nsamples) * mod_inp.float()
                H += mod_inp.matmul(mod_inp.t())
            
                self.H_list[expert_idx] = H
                self.nsamples_list[expert_idx] = nsamples
        else:
            # print("INSIDE ADD_BATCH FOR 2D")
            # print("INSIDE ADD_BATCH 1: inp", inp, type(inp), inp.shape)
            if len(inp.shape) == 2:
                inp = inp.unsqueeze(0)
            tmp = inp.shape[0]
            if isinstance(self.layer, nn.Linear) or isinstance(
                self.layer, transformers.Conv1D
            ):
                if len(inp.shape) == 3:
                    inp = inp.reshape((-1, inp.shape[-1]))
                inp = inp.t()
            if isinstance(self.layer, nn.Conv2d):
                unfold = nn.Unfold(
                    self.layer.kernel_size,
                    dilation=self.layer.dilation,
                    padding=self.layer.padding,
                    stride=self.layer.stride,
                )
                inp = unfold(inp)
                inp = inp.permute([1, 0, 2])
                inp = inp.flatten(1)
            # print("INSIDE ADD_BATCH 2: BEFORE tmp, self.H, self.nsamples", tmp, self.H, self.nsamples)
            self.H *= self.nsamples / (self.nsamples + tmp)
            self.nsamples += tmp
            # print("INSIDE ADD_BATCH 3: AFTER tmp, self.H, self.nsamples", tmp, self.H, self.nsamples)
            # inp = inp.float()
            inp = math.sqrt(2 / self.nsamples) * inp.float()
            # self.H += 2 / self.nsamples * inp.matmul(inp.t())
            self.H += inp.matmul(inp.t())

    def fasterquant(
        self,
        blocksize=128,
        percdamp=0.01,
        group_size=-1,
        actorder=False,
        static_groups=False,
    ):
        if self.is_moe:
            # For MoE model
            # Loop over each expert param and quantize it separately 
            t_start = time.time()
            scale_list = []
            zero_list = []
            gidx_list = []
            loss_list = []

            for i in range(self.num_experts):
                W = self.W_list[i]  # shape [out_features, in_features]
                H = self.H_list[i]
                nsamples = self.nsamples_list[i]
                
                ##### TODO: QUANTIZATION FOR MOE LAYER #####

            duration = time.time() - t_start
            final_scale = torch.cat(scale_list, dim=1) if scale_list else torch.tensor([], device=self.dev)
            final_zero = torch.cat(zero_list, dim=1) if zero_list else torch.tensor([], device=self.dev)
            final_gidx = torch.cat(gidx_list, dim=0)  if gidx_list else torch.tensor([], device=self.dev)
            avg_loss = sum(loss_list) / (len(loss_list) + 1e-9)

            return final_scale, final_zero, final_gidx, duration, avg_loss
        else:
            W = self.layer.weight.data.clone()
            if isinstance(self.layer, nn.Conv2d):
                W = W.flatten(1)
            if isinstance(self.layer, transformers.Conv1D):
                W = W.t()
            W = W.float()

            tick = time.time()

            if not self.quantizer.ready():
                self.quantizer.find_params(W, weight=True)

            H = self.H
            del self.H
            dead = torch.diag(H) == 0
            H[dead, dead] = 1
            W[:, dead] = 0

            g_idx = []
            scale = []
            zero = []
            now_idx = 1

            if static_groups:
                # Standard
                import copy

                groups = []
                for i in range(0, self.columns, group_size):
                    quantizer = copy.deepcopy(self.quantizer)
                    quantizer.find_params(W[:, i : (i + group_size)], weight=True)
                    scale.append(quantizer.scale)
                    zero.append(quantizer.zero)
                    groups.append(quantizer)

            if actorder:
                perm = torch.argsort(torch.diag(H), descending=True)
                W = W[:, perm]
                H = H[perm][:, perm]
                invperm = torch.argsort(perm)

            Losses = torch.zeros_like(W)
            Q = torch.zeros_like(W)

            damp = percdamp * torch.mean(torch.diag(H))
            diag = torch.arange(self.columns, device=self.dev)
            H[diag, diag] += damp
            H = torch.linalg.cholesky(H)
            H = torch.cholesky_inverse(H)
            H = torch.linalg.cholesky(H, upper=True)
            Hinv = H

            for i1 in range(0, self.columns, blocksize):
                i2 = min(i1 + blocksize, self.columns)
                count = i2 - i1

                W1 = W[:, i1:i2].clone()
                Q1 = torch.zeros_like(W1)
                Err1 = torch.zeros_like(W1)
                Losses1 = torch.zeros_like(W1)
                Hinv1 = Hinv[i1:i2, i1:i2]

                for i in range(count):
                    w = W1[:, i]
                    d = Hinv1[i, i]

                    if group_size != -1:
                        if not static_groups:
                            if (i1 + i) % group_size == 0:
                                self.quantizer.find_params(
                                    W[:, (i1 + i) : (i1 + i + group_size)], weight=True
                                )

                            if ((i1 + i) // group_size) - now_idx == -1:
                                scale.append(self.quantizer.scale)
                                zero.append(self.quantizer.zero)
                                now_idx += 1
                        else:
                            idx = i1 + i
                            if actorder:
                                idx = perm[idx]
                            self.quantizer = groups[idx // group_size]

                    q = self.quantizer.quantize(w.unsqueeze(1)).flatten()
                    Q1[:, i] = q
                    Losses1[:, i] = (w - q) ** 2 / d**2

                    err1 = (w - q) / d
                    W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
                    Err1[:, i] = err1

                Q[:, i1:i2] = Q1
                Losses[:, i1:i2] = Losses1 / 2

                W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])

                if os.environ.get("DEBUG"):
                    self.layer.weight.data[:, :i2] = Q[:, :i2]
                    self.layer.weight.data[:, i2:] = W[:, i2:]
                    logger.debug(torch.sum((self.layer(self.inp1) - self.out1) ** 2))
                    logger.debug(torch.sum(Losses))

            torch.cuda.synchronize()

            duration = time.time() - tick
            avg_loss = torch.sum(Losses).item() / self.nsamples

            group_size = group_size if group_size != -1 else self.columns
            if static_groups and actorder:
                g_idx = [perm[i] // group_size for i in range(self.columns)]
            else:
                g_idx = [i // group_size for i in range(self.columns)]
            g_idx = torch.tensor(g_idx, dtype=torch.int32, device=Q.device)
            if actorder:
                Q = Q[:, invperm]
                g_idx = g_idx[invperm]

            if isinstance(self.layer, transformers.Conv1D):
                Q = Q.t()
            self.layer.weight.data = Q.reshape(self.layer.weight.shape).type_as(
                self.layer.weight.data
            )

            if os.environ.get("DEBUG"):
                logger.debug(torch.sum((self.layer(self.inp1) - self.out1) ** 2))

            if scale == []:
                scale.append(self.quantizer.scale)
                zero.append(self.quantizer.zero)
            scale = torch.cat(scale, dim=1)
            zero = torch.cat(zero, dim=1)
            return scale, zero, g_idx, duration, avg_loss

    def free(self):
        if os.environ.get("DEBUG"):
            self.inp1 = None
            self.out1 = None
        self.H = None
        self.Losses = None
        self.Trace = None
        torch.cuda.empty_cache()


__all__ = ["GPTQ"]
