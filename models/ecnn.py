import torch
import torch.nn as nn

from utils.convolutions import LiftingConvolution, GroupConvolution
    
class GroupEquivariantCNN(torch.nn.Module):

    def __init__(self, group, in_channels, out_channels, kernel_size, num_hidden, hidden_channels):
        super().__init__()

        self.padding = 0

        self.lifting_conv = LiftingConvolution(group, 
                                               in_channels=in_channels,
                                               out_channels=hidden_channels,
                                               kernel_size=kernel_size,
                                               padding=self.padding)

        self.gconvs = torch.nn.ModuleList()

        for i in range(num_hidden):
            self.gconvs.append(GroupConvolution(group, 
                                                in_channels=hidden_channels,
                                                out_channels=hidden_channels,
                                                kernel_size=kernel_size,
                                                padding=self.padding))

        self.projection_layer = nn.AdaptiveAvgPool3d(1)

        self.final_linear = torch.nn.Linear(hidden_channels, out_channels)
    
    def forward(self, x):
        x = self.lifting_conv(x)
        # x = torch.nn.functional.layer_norm(x, x.shape[-4:])
        x = torch.nn.functional.relu(x)

        for gconv in self.gconvs:
            x = gconv(x)
            # x = torch.nn.functional.layer_norm(x, x.shape[-4:])
            x = torch.nn.functional.relu(x)

        x = self.projection_layer(x).squeeze()

        x = self.final_linear(x)
        return x