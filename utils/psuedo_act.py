import torch
import torch.nn as nn
import torch.nn.functional as F

class ActFun(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, thresh=0.5, lens=0.5/3):
        ctx.thresh = thresh
        ctx.lens = lens

        ctx.save_for_backward(input)
        return input.gt(thresh).float()

    @staticmethod
    def backward(ctx, grad_output):
        thresh = ctx.thresh
        lens = ctx.lens

        input, = ctx.saved_tensors
        grad_input = grad_output.clone()

        # temp = abs(input - thresh) < lens
        temp = torch.exp(-(input - thresh) ** 2 / (2 * lens ** 2)) / ((2 * lens * torch.pi) ** 0.5)

        return grad_input * temp.float()

act_fun = ActFun.apply

def mem_update(ops, x, spike, mem=None, decay=0.8):
    curr_volts = ops(x)

    if mem is None:
        mem = torch.zeros_like(curr_volts)
    mem = mem * decay + curr_volts

    spike = act_fun(mem)
    return mem, spike
