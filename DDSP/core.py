import numpy as np
import torch.nn as nn

def resample(signal, factor):
    if len(np.shape(signal)) == 2:
        x = signal[:,:,np.newaxis].permute(0, 2, 1)
    else:
        x = signal.permute(0, 2, 1)
    x = nn.functional.interpolate(x, scale_factor=factor)
    if np.shape(x)[2] == 1:
        x = x[:,:,0]
    return x.permute(0, 2, 1)