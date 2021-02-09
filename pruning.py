import torch
import torch.utils.data


class PruningTool:
    def __init__(self):
        pass

    def stats_pruning(self, m):
        global_null_weights = 0
        global_total_weights = 0
        print("Model :")
        i = 0
        for name, module in m.named_modules():
            if isinstance(module,torch.nn.Linear):
                print(module)
                null_weights = float(torch.sum(module.weight == 0))
                layer_weights = float(module.weight.nelement())
                print("Sparsity in Layer {}: {:.2f}%".format(i, 100. * null_weights / layer_weights))
                global_total_weights += layer_weights
                global_null_weights += null_weights
                i += 1
            elif isinstance(module,torch.nn.Conv2d):
                null_weights = 0
                layer_weights = 0
                for idx, channel in enumerate(module.weight):
                    null_weights += float(torch.sum(channel == 0))
                    layer_weights += float(channel.nelement())
                    print("null_weights = ", null_weights)
                    print("layer_weights = ", layer_weights)
                print("Sparsity in Layer {}: {:.2f}%".format(i, 100. * null_weights / layer_weights))
                global_total_weights += layer_weights
                global_null_weights += null_weights
                i += 1

            else:
                print(name,module)
        print("Global Sparsity : {:.2f}%".format(100. * global_null_weights / global_total_weights))

    def compute_global_criterion(self, m, amount):
        weights = torch.empty(0).cuda() if torch.cuda.is_available() else torch.empty(0)

        for module in m.children():
            weights = torch.cat((weights, module.weight.clone().flatten()), 0)
        w, _ = torch.sort(weights.abs())
        cutting_index = int(amount * w.shape[0])
        cutting_value = w[cutting_index]
        return cutting_value


class PruningLinear:
    def __init__(self, module):
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


class PruningConv2D:
    def __init__(self, module):
        self.module = module
        self.list_index_to_remove = []
        self.mask = None

    def set_layers_list_locally(self, amount):
        list_norms = []
        for idx, channel in enumerate(self.module.weight):
            n = torch.norm(channel)
            list_norms.append((n, idx))
        list_norms.sort()  # sort norms, first elt of tuple
        cutting_index = int(amount * len(list_norms))
        self.list_index_to_remove = [elt[1] for elt in list_norms[:cutting_index]]
        self.set_mask()

    def set_layers_list_globally(self, cutting_value):
        self.list_index_to_remove = []
        for idx, channel in enumerate(self.module.weight):
            n = torch.norm(channel)
            if n < cutting_value:
                self.list_index_to_remove.append(idx)
        self.set_mask()

    def set_mask(self):
        self.mask = torch.ones_like(self.module.weight)
        for idx in self.list_index_to_remove:
            self.mask[idx]=torch.zeros_like(self.mask[idx])

    def __call__(self, module, inputs):
        module.weight.data = module.weight.data * self.mask
