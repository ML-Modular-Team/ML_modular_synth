import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
import numpy as np
import torch.nn.utils.prune as prune
import matplotlib.pyplot as plt


class Pruning_tool:
    def __init__(self):
        pass

    def stats_pruning(self,m):
        global_null_weights = 0
        global_total_weights = 0
        print("Model :")
        i = 0
        for module in m.children():
            print(module)
            null_weights = float(torch.sum(module.weight == 0))
            layer_weights = float(module.weight.nelement())
            print("Sparsity in Layer {}: {:.2f}%".format(i,100. * null_weights / layer_weights ))
            global_total_weights += layer_weights
            global_null_weights += null_weights
            i += 1

        print("Global Sparsity : {:.2f}%".format( 100. * global_null_weights / global_total_weights))


    def get_mask(self,layer, amount):
        weights = layer.weight.clone().flatten()
        w, _ = torch.sort(weights.abs())
        cutting_index = int(amount * w.shape[0])
        cutting_value = w[cutting_index]

        mask = layer.weight.abs() > cutting_value
        return mask




