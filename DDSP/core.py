import torch
import torch.nn as nn
import numpy as np
import math

def resample(signal, factor):
    if len(np.shape(signal)) == 2:
        x = signal[:,:,np.newaxis].permute(0, 2, 1)
    else:
        x = signal.permute(0, 2, 1)
    x = nn.functional.interpolate(x, scale_factor=factor)
    if np.shape(x)[2] == 1:
        x = x[:,:,0]
    return x.permute(0, 2, 1)

def scale_function(x):
    return (2 * torch.sigmoid(x)**(math.log(10)) + 1e-7)