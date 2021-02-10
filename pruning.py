import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
import numpy as np
import torch.nn.utils.prune as prune
import matplotlib.pyplot as plt


class PruningTool:
    def __init__(self):
        pass

    def compute_global_criterion(self, m, amount):
        weights = torch.empty(0).cuda() if torch.cuda.is_available() else torch.empty(0)
        for module in m.children():
            if isinstance(module, nn.Linear):
                weights = torch.cat((weights,module.weight.clone().flatten()),0)
            elif isinstance(module, nn.GRU):
                weights = torch.cat((weights, module.weight_ih_l0.clone().flatten()),0)
                weights = torch.cat((weights, module.weight_hh_l0.clone().flatten()),0)
        w, _ = torch.sort(weights.abs())
        cutting_index = int(amount * w.shape[0])
        cutting_value = w[cutting_index]
        return cutting_value

    def prune(self,model, amount, is_global=True):
        if is_global:
            cutting_value = self.compute_global_criterion(model, amount) 
        for name, module in model.named_modules():
            if len(list(module.children())) > 0:
                continue
            if isinstance(module, torch.nn.Linear):
                pr = PruningLinear(module)
            elif isinstance(module, torch.nn.GRU):
                pr = PruningGRU(module)
            elif isinstance(module, torch.nn.Conv2D):
                pr = PruningConv2D(module)
            else:
                continue
            if is_global:
                pr.set_mask_globally(cutting_value)
            else:
                pr.set_mask_locally(amount)
            module.register_forward_pre_hook(pr)

    def stats_pruning(self, m, verbose=True):
        global_null_weights = 0
        global_total_weights = 0
        if verbose:
            print("Model :")
        i = 0
        for name, module in m.named_modules():
            if isinstance(module, torch.nn.Linear):
                print(module)
                null_weights = float(torch.sum(module.weight == 0))
                layer_weights = float(module.weight.nelement())
                print("Sparsity in Layer {}: {:.2f}%".format(name, 100. * null_weights / layer_weights))
                global_total_weights += layer_weights
                global_null_weights += null_weights
                i += 1
            elif isinstance(module, torch.nn.Conv2d):
                null_weights = 0
                layer_weights = 0
                for idx, channel in enumerate(module.weight):
                    null_weights += float(torch.sum(channel == 0))
                    layer_weights += float(channel.nelement())
                print("Sparsity in Layer {}: {:.2f}%".format(name, 100. * null_weights / layer_weights))
                global_total_weights += layer_weights
                global_null_weights += null_weights
                i += 1

            else:
                if verbose:
                    print(name, module)
        global_sparsity = 100. * global_null_weights / global_total_weights
        print("Global Sparsity : {:.2f}%".format(global_sparsity))
        return global_sparsity


class PruningLinear:
    def __init__(self,module):
        self.mask = None
        self.module = module

    def set_mask_locally(self, amount):
        weights = self.module.weight.clone().flatten()
        w, _ = torch.sort(weights.abs())
        cutting_index = int(amount * w.shape[0])
        cutting_value = w[cutting_index]
        self.mask = self.module.weight.abs() > cutting_value

    def set_mask_globally(self, cutting_value):
        self.mask = self.module.weight.abs() > cutting_value
  
    def __call__(self, module, inputs):
        module.weight.data = module.weight.data * self.mask
    

class PruningGRU:
    def __init__(self,module):
        self.mask_ih = None
        self.mask_hh = None
        self.module = module

    def set_mask_locally(self, amount):
        # reset mask for concatenation of sub masks
        self.mask_ih = torch.empty((0, self.module.weight_ih_l0.shape[1])).cuda()
        self.mask_hh = torch.empty((0, self.module.weight_hh_l0.shape[1])).cuda()

        # weights are chunked into a tuple (w_r,w_z,w_n)
        weights_ih = self.module.weight_ih_l0.clone().chunk(3, 0)
        weights_hh = self.module.weight_hh_l0.clone().chunk(3, 0)

        # computing each submask and concatenate it to the mask
        for k in range(3):
            w_ih, _ = torch.sort(weights_ih[k].flatten().abs())
            w_hh, _ = torch.sort(weights_hh[k].flatten().abs())
            cutting_index_ih = int(amount * w_ih.shape[0])
            cutting_index_hh = int(amount * w_hh.shape[0])
            cutting_value_ih = w_ih[cutting_index_ih]
            cutting_value_hh = w_hh[cutting_index_hh]
            self.mask_ih = torch.cat((self.mask_ih, weights_ih[k].abs() > cutting_value_ih), 0)
            self.mask_hh = torch.cat((self.mask_hh, weights_hh[k].abs() > cutting_value_hh), 0)
    
    def set_mask_globally(self, cutting_value):
        self.mask_ih = self.module.weight_ih_l0.abs() > cutting_value
        self.mask_hh = self.module.weight_hh_l0.abs() > cutting_value
    
    def __call__(self,module,inputs):
        module.weight_ih_l0.data = module.weight_ih_l0.data * self.mask_ih
        module.weight_hh_l0.data = module.weight_hh_l0.data * self.mask_hh
        module.flatten_parameters()


class PruningConv2D:
    def __init__(self, module):
        self.module = module
        self.list_index_to_remove = []
        self.mask = None

    def set_mask_locally(self, amount):
        list_norms = []
        for idx, channel in enumerate(self.module.weight):
            n = torch.norm(channel)
            list_norms.append((n, idx))
        list_norms.sort()  # sort norms, first elt of tuple
        cutting_index = int(amount * len(list_norms))
        self.list_index_to_remove = [elt[1] for elt in list_norms[:cutting_index]]
        self.set_mask()

    def set_mask_globally(self, cutting_value):
        self.list_index_to_remove = []
        for idx, channel in enumerate(self.module.weight):
            n = torch.norm(channel)
            if n < cutting_value:
                self.list_index_to_remove.append(idx)
        self.set_mask()

    def set_mask(self):
        self.mask = torch.ones_like(self.module.weight)
        for idx in self.list_index_to_remove:
            self.mask[idx] = torch.zeros_like(self.mask[idx])

    def __call__(self, module, inputs):
        module.weight.data = module.weight.data * self.mask
