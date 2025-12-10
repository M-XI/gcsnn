import sys
import os
import numpy as np
import torch


def bilinear_interpolation(signal, grid):
    """ Obtain signal values for a set of gridpoints through bilinear interpolation.
    
    @param signal: Tensor containing pixel values [C, H, W] or [N, C, H, W]
    @param grid: Tensor containing coordinate values [2, H, W] or [2, N, H, W]
    """
    # If signal or grid is a 3D array, add a dimension to support grid_sample.
    assert (len(signal.shape) == 3 and len(grid.shape) == 3) or \
        (len(signal.shape) == 4 and len(grid.shape) == 4), \
        "Both signal and grid should be either 3D or 4D arrays."
    
    batch_mode = True
    if len(signal.shape) == 3:
        signal = signal.unsqueeze(0)
        grid = grid.unsqueeze(1)
        batch_mode = False
    
    # Grid_sample expects [N, H, W, 2] instead of [2, N, H, W]
    grid = grid.permute(1, 2, 3, 0)
    
    # Grid sample expects YX instead of XY.
    grid = torch.roll(grid, shifts=1, dims=-1)
    
    sampled = torch.nn.functional.grid_sample(
        signal,
        grid,
        padding_mode='zeros',
        align_corners=True,
        mode="bilinear"
    )
    if not batch_mode:
        sampled = sampled.squeeze(0)
    return sampled

def trilinear_interpolation(signal, grid):
    """ 
    
    @param signal: Tensor containing pixel values [C, D, H, W] or [N, C, D, H, W]
    @param grid: Tensor containing coordinate values [3, D, H, W] or [3, N, D, H, W]
    """
    # If signal or grid is a 4D array, add a dimension to support grid_sample.
    assert (len(signal.shape) == 4 and len(grid.shape) == 4) or \
        (len(signal.shape) == 5 and len(grid.shape) == 5), \
        "Both signal and grid should be either 4D or 5D arrays."
    
    batch_mode = True
    if len(signal.shape) == 4:
        signal = signal.unsqueeze(0)
        grid = grid.unsqueeze(1)
        batch_mode = False

    # Grid_sample expects [N, D, H, W, 3] instead of [3, N, D, H, W]
    grid = grid.permute(1, 2, 3, 4, 0)
    
    # Grid sample expects ZYX instead of XYZ.
    # grid = torch.roll(grid, shifts=1, dims=-1)
    grid = grid.flip(-1)
    
    sampled = torch.nn.functional.grid_sample(
        signal, 
        grid,
        padding_mode='zeros',
        align_corners=True,
        mode="bilinear" # actually trilinear in this case...
    )
    if not batch_mode:
        sampled = sampled.squeeze(0)
    return sampled