## PyTorch
import torch
import torch.nn as nn
import numpy as np

from utils.interpolation import bilinear_interpolation

class CyclicGroup(torch.nn.Module):

    def __init__(self, order):
        super().__init__()
        
        self.dimension = 1
        self.register_buffer('identity', torch.Tensor([0.]))

        assert order > 1
        self.order = torch.tensor(order)

    def elements(self):
        return torch.linspace(
            start=0,
            end=2 * np.pi * float(self.order - 1) / float(self.order),
            steps=self.order,
            device=self.identity.device
        )
    
    def product(self, h, h_prime):
        product = (h + h_prime) % (2 * torch.pi)

        return product

    def inverse(self, h):
        inverse = (-h) % (2 * torch.pi)

        return inverse
    
    def left_action_on_R2(self, h, x):
        transformed_x = torch.tensordot(self.matrix_representation(h), x, dims=1)       
        return transformed_x

    def matrix_representation(self, h):
        h_sin = torch.sin(h)
        h_cos = torch.cos(h)
        representation = torch.Tensor([[h_cos, -h_sin], 
                                       [h_sin, h_cos]]).to(device=h.device)

        return representation
    
    def normalize_group_elements(self, h):
        largest_elem = 2 * np.pi * (self.order - 1) / self.order
        normalized_h = (2*h / largest_elem) - 1.
        return normalized_h
    
def transformed_img_function(img_tensor, img_grid, order=4):
    cn = CyclicGroup(order=order)
    _, g1, _, _ = cn.elements()
    
    transformed_img_grid = cn.left_action_on_R2(cn.inverse(g1), img_grid)
    transformed_img = bilinear_interpolation(img_tensor, transformed_img_grid)
    
    return transformed_img