import torch
import torch.nn as nn

from utils.kernels import LiftingKernel, GroupKernel

class LiftingConvolution(torch.nn.Module):

    def __init__(self, group, in_channels, out_channels, kernel_size, padding):
        super().__init__()

        self.kernel = LiftingKernel(
            group=group,
            kernel_size=kernel_size,
            in_channels=in_channels,
            out_channels=out_channels
        )
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.padding = padding

    def forward(self, x):
        batch_mode = True if len(x.shape) == 4 else False

        lifting_kernel_weights = self.kernel.lift_kernel()

        lifting_kernel_weights = lifting_kernel_weights.reshape(
            self.kernel.group.elements().numel() * self.kernel.out_channels,
            self.kernel.in_channels,
            self.kernel.kernel_size,
            self.kernel.kernel_size
        )

        x = torch.nn.functional.conv2d(x, 
                                       weight=lifting_kernel_weights, 
                                       padding=self.padding)

        x = x.view(
            -1,
            self.kernel.out_channels,
            self.kernel.group.elements().numel(),
            x.shape[-1],
            x.shape[-2]
        )
        if not batch_mode:
            x = x.squeeze(0)

        return x
    
class GroupConvolution(torch.nn.Module):

    def __init__(self, group, in_channels, out_channels, kernel_size, padding):
        super().__init__()

        self.kernel = GroupKernel(
            group=group,
            kernel_size=kernel_size,
            in_channels=in_channels,
            out_channels=out_channels
        )
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.padding = padding
        

    def forward(self, x):
        batch_mode = True if len(x.shape) == 5 else False

        if batch_mode:
            x = x.flatten(1,2)
        else:
            x = x.flatten(0,1)

        lifting_kernel_weights = self.kernel.lift_kernel()

        lifting_kernel_weights = lifting_kernel_weights.reshape(
            self.kernel.group.elements().numel() * self.kernel.out_channels,
            self.kernel.group.elements().numel() * self.kernel.in_channels,
            self.kernel.kernel_size,
            self.kernel.kernel_size
        )

        x = torch.nn.functional.conv2d(x, 
                                       weight=lifting_kernel_weights,
                                       padding=self.padding)

        x = x.view(
            -1,
            self.kernel.out_channels,
            self.kernel.group.elements().numel(),
            x.shape[-1],
            x.shape[-2]
        )
        if not batch_mode:
            x = x.squeeze(0)

        return x