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
    
    def prune(model, amount, is_global=True):
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

    def stats_pruning(self,m):
        global_null_weights = 0
        global_total_weights = 0
        print("Model :")
        i = 0
        for module in model.children():
            print(module)
            if isinstance(module, nn.Linear):
                null_weights = float(torch.sum(module.weight == 0))
                layer_weights = float(module.weight.nelement())
            elif isinstance(module, nn.GRU):
                null_weights = float(torch.sum(module.weight_ih_l0 == 0)) + float(torch.sum(module.weight_hh_l0 == 0))
                layer_weights = float(module.weight_ih_l0.nelement()) + float(module.weight_hh_l0.nelement())
            print("Sparsity in Layer {}: {:.2f}%".format(i,100. * null_weights / layer_weights ))
            global_total_weights += layer_weights
            global_null_weights += null_weights
            i += 1           
        print("Global Sparsity : {:.2f}%".format( 100. * global_null_weights / global_total_weights))
        
    def compute_global_criterion(self, m, amount):
        weights = torch.empty(0).cuda() if torch.cuda.is_available() else torch.empty(0)

        for module in m.children():
            if isinstance(module, nn.Linear):
                weights = torch.cat((weights,module.weight.clone().flatten()),0)
            elif isinstance(module, nn.GRU):
            weights = torch.cat((weight, module.weight_ih_l0.clone().flatten()),0)
            weights = torch.cat((weight, module.weight_hh_l0.clone().flatten()),0)
        w, _ = torch.sort(weights.abs())
        cutting_index = int(amount * w.shape[0])
        cutting_value = w[cutting_index]
        return cutting_value

    
    
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
  
    def __call__(self,module,inputs):
        module.weight.data = module.weight.data * self.mask
    

class PruningGRU:
    def __init__(self,module):
        self.mask_ih = None
        self.mask_hh = None
        self.module = module


    def set_mask_locally(self, amount):
        # reset mask for concatenation of submasks
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
    def __init__(self,module):
        self.module = module
        self.list_index_to_remove = []

    def set_mask_locally(self, amount):
        list_norms=[]
        for idx, channel in enumerate(self.module.weight):
            n = torch.norm(channel)
            list_norms.append((n,idx))
        list_norms.sort() # sort norms, first elt of tuple
        cutting_index = int(amount * len(list_norms))
        self.list_index_to_remove = [elt[1] for elt in list_norms[:cutting_index]]

    def set_mask_globally(self, cutting_value):
        self.list_index_to_remove=[]
        for idx, channel in enumerate(self.module.weight):
            n = torch.norm(channel)
            if n<cutting_value:
                self.list_index_to_remove.append(idx)
        
    
    def __call__(self,module,inputs):
        for idx, channel in enumerate(module.weight):
            if idx in self.list_index_to_remove:
                channel = torch.zeros_like(channel)
                
        for idx, channel in enumerate(module.weight):
            print("Channel ",idx)
            print(channel)