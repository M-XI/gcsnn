import torch
import torch.nn as nn
import math

from utils.interpolation import bilinear_interpolation, trilinear_interpolation

class LiftingKernel(torch.nn.Module):
    
    def __init__(self, group, kernel_size, in_channels, out_channels):
        super().__init__()
        self.group = group

        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.register_buffer("grid_R2", torch.stack(torch.meshgrid(
            torch.linspace(-1., 1., self.kernel_size),
            torch.linspace(-1., 1., self.kernel_size),
            indexing='ij'
        )).to(self.group.identity.device))

        self.register_buffer("transformed_grid_R2", self.create_transformed_grid_R2())

        self.weight = torch.nn.Parameter(torch.zeros((
            self.out_channels,
            self.in_channels,
            self.kernel_size,
            self.kernel_size
        ), device=self.group.identity.device))

        torch.nn.init.kaiming_uniform_(self.weight.data, a=math.sqrt(5))

    def create_transformed_grid_R2(self):
        group_elements = self.group.elements()

        transformed_grid = []

        for element in group_elements:
            transformed_grid.append(self.group.left_action_on_R2(self.group.inverse(element), self.grid_R2))

        transformed_grid = torch.stack(transformed_grid, dim = 1)

        assert transformed_grid.shape == torch.Size(
            [2, group_elements.numel(), self.kernel_size, self.kernel_size])

        return transformed_grid

    def lift_kernel(self):
        weight = self.weight.view(
            self.out_channels * self.in_channels,
            self.kernel_size,
            self.kernel_size
        )

        group_elements = self.group.elements()

        transformed_weight = []
        for element in group_elements:
            transformed_weight.append(bilinear_interpolation(weight, self.group.left_action_on_R2(self.group.inverse(element), self.grid_R2)))

        transformed_weight = torch.stack(transformed_weight, dim = 0)

        assert transformed_weight.shape == torch.Size(
            [self.group.elements().numel(), \
            self.out_channels * self.in_channels, \
            self.kernel_size, \
            self.kernel_size]), transformed_weight.shape
            
        transformed_weight = transformed_weight.view(
            self.group.elements().numel(),
            self.out_channels,
            self.in_channels,
            self.kernel_size,
            self.kernel_size
        )
        transformed_weight = transformed_weight.transpose(0, 1)

        return transformed_weight
    
class GroupKernel(torch.nn.Module):

    def __init__(self, group, kernel_size, in_channels, out_channels):
        super().__init__()
        self.group = group

        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.register_buffer("grid_R2", torch.stack(torch.meshgrid(
            torch.linspace(-1., 1., self.kernel_size),
            torch.linspace(-1., 1., self.kernel_size),
            indexing='ij'
        )).to(self.group.identity.device))

        self.register_buffer("grid_H", self.group.elements())
        self.register_buffer("transformed_grid_HxR2", self.create_transformed_grid_HxR2())

        self.weight = torch.nn.Parameter(torch.zeros((
            self.out_channels,
            self.in_channels,
            self.group.elements().numel(),
            self.kernel_size,
            self.kernel_size
        ), device=self.group.identity.device))

        torch.nn.init.kaiming_uniform_(self.weight.data, a=math.sqrt(5))

    def create_transformed_grid_HxR2(self):
        group_elements = self.group.elements()

        transformed_grid_R2 = []

        for element in group_elements:
            transformed_grid_R2.append(self.group.left_action_on_R2(self.group.inverse(element), self.grid_R2))

        transformed_grid_R2 = torch.stack(transformed_grid_R2, dim = 1)

        transformed_grid_H = []

        for element in group_elements:
            transformed_grid_H.append(self.group.product(self.group.inverse(element), self.group.elements()))

        transformed_grid_H = torch.stack(transformed_grid_H, dim = 0)

        transformed_grid_H = self.group.normalize_group_elements(transformed_grid_H)

        transformed_grid = torch.cat(
            (
                transformed_grid_H.view(
                    1,
                    group_elements.numel(),
                    group_elements.numel(),
                    1,
                    1,
                ).repeat(1, 1, 1, self.kernel_size, self.kernel_size),
                transformed_grid_R2.view(
                    2,
                    group_elements.numel(),
                    1,
                    self.kernel_size,
                    self.kernel_size,
                ).repeat(1, 1, group_elements.numel(), 1, 1)
            ),
            dim=0
        )
        return transformed_grid


    def lift_kernel(self):
        weight = self.weight.view(
            self.out_channels * self.in_channels,
            self.group.elements().numel(),
            self.kernel_size,
            self.kernel_size
        )

        group_elements = self.group.elements()

        transformed_weight = []
        for i in range(len(group_elements)):
            transformed_weight.append(trilinear_interpolation(weight, self.transformed_grid_HxR2[:, i, ...]))

        transformed_weight = torch.stack(transformed_weight, dim = 0)

        transformed_weight = transformed_weight.view(
            self.group.elements().numel(),
            self.out_channels,
            self.in_channels,
            self.group.elements().numel(),
            self.kernel_size,
            self.kernel_size
        )

        transformed_weight = transformed_weight.transpose(0, 1)
        
        return transformed_weight