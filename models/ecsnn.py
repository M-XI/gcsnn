import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.psuedo_act import mem_update
from utils.convolutions import LiftingConvolution, GroupConvolution

# Based on: https://github.com/aa-samad/conv_snn/blob/master/Tests_1-3_and_5/SNN/spiking_model.py

class ECSNN(nn.Module):
    def __init__(self, group, in_channels, out_channels, kernel_size, num_hidden, hidden_channels,
                 event_converter):
        super().__init__()

        self.event_converter = event_converter

        self.lifting_conv = LiftingConvolution(
            group, 
            in_channels=in_channels,
            out_channels=hidden_channels,
            kernel_size=kernel_size,
            padding=0
        )

        self.convs = torch.nn.ModuleList()
        for i in range(num_hidden):
            self.convs.append(
                GroupConvolution(
                    group, 
                    in_channels=hidden_channels,
                    out_channels=hidden_channels,
                    kernel_size=kernel_size,
                    padding=0
                )
            )
        
        self.projection_layer = nn.AdaptiveAvgPool3d(1)

        self.head = torch.nn.Linear(hidden_channels, out_channels)

    def forward(self, input):
        # convolutional layers membrane potential and spike memory
        c1_mem = c1_spike = None
        c_mem = c_spike = [None] * len(self.convs)

        h_mem = h_spike = None
        h_sumspike = 0

        for event in self.event_converter(input):
            c1_mem, c1_spike = mem_update(self.lifting_conv, event.float(), c1_spike, c1_mem)
            x = c1_spike
            # x = F.avg_pool2d(c1_spike, 2)

            for i, conv in enumerate(self.convs):
                c_mem[i], c_spike[i] = mem_update(conv, x, c_spike[i], c_mem[i])
                x = c_spike[i]
                # x = F.avg_pool2d(c_spike[i], 2)

            x = self.projection_layer(x).squeeze()

            h_mem, h_spike = mem_update(self.head, x, h_mem, h_spike)
            h_sumspike += h_spike

        outputs = h_sumspike / len(self.convs)
        return outputs
    

