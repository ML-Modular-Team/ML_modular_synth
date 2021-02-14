import torch
import torch.nn as nn

# Define MLP module
class MLP(nn.Module):
    def __init__(self, in_size, hidden_size, n_layers):
        super(MLP, self).__init__()
        mlp_net = []
        for i in range(n_layers):
            if (i == 0):
                mlp_net.append(nn.Linear(in_size, hidden_size)) # equivalent to Dense in keras
            else:
                mlp_net.append(nn.Linear(hidden_size, hidden_size)) # equivalent to Dense in keras
            mlp_net.append(nn.LayerNorm(hidden_size))
            mlp_net.append(nn.ReLU())
        self.layers = nn.Sequential(*mlp_net)
        
    def forward(self, x):
        x = self.layers(x)
        return x

# Define Decoder
class Decoder(nn.Module):
    def __init__(self, hidden_size, n_harmonics, n_bands, n_layers, sampling_rate, block_size):
        super(Decoder, self).__init__()
        #self.register_buffer("sampling_rate", torch.tensor(sampling_rate))
        #self.register_buffer("block_size", torch.tensor(block_size))
        
        self.MLP_f0 = MLP(1, hidden_size, n_layers)
        self.MLP_loudness = MLP(1, hidden_size, n_layers)
        self.GRU = nn.GRU(2 * hidden_size, hidden_size)
        self.MLP_out = MLP(3 * hidden_size, hidden_size, n_layers)
        
        self.harm_linear = nn.Linear(hidden_size, n_harmonics + 1)
        self.noise_linear = nn.Linear(hidden_size, n_bands)

    def forward(self, f0, loudness):
        x1 = self.MLP_fo(f0)
        x2 = self.MLP_loudness(loudness)
        x = torch.cat([x1, x2], -1)
        x = self.GRU(x)
        x = torch.cat([x, x1, x2], -1)
        x = self.MLP_out(x)
        
        harm_param = self.harm_linear(x)
        noise_param = self.noise_linear(x)
        
        return harm_param, noise_param