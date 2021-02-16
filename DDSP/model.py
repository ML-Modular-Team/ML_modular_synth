import torch
import torch.nn as nn
import torch.nn.functional as f
import math

from ddsp_components import harmonic_synth
from core import resample
from core import scale_function

# Define MLP module
class MLP(nn.Module):
    def __init__(self, in_size, hidden_size, n_layers):
        super(MLP, self).__init__()
        mlp_net = []
        for i in range(0, n_layers):
            if (i == 0):
                mlp_net.append(nn.Linear(in_size, hidden_size)) # equivalent to Dense in keras
            else:
                mlp_net.append(nn.Linear(hidden_size, hidden_size)) # equivalent to Dense in keras
            mlp_net.append(nn.LayerNorm(hidden_size))
            mlp_net.append(nn.LeakyReLU())
        self.layers = nn.Sequential(*mlp_net)
        
    def forward(self, x):
        x = self.layers(x)
        return x

# Define Decoder
class DDSP(nn.Module):
    def __init__(self, hidden_size, n_harmonics, n_bands, n_layers, sampling_rate, block_size):
        super(DDSP, self).__init__()
        self.register_buffer("sampling_rate", torch.tensor(sampling_rate))
        self.register_buffer("block_size", torch.tensor(block_size))
        
        self.MLP_f0 = MLP(1, hidden_size, n_layers)
        self.MLP_loudness = MLP(1, hidden_size, n_layers)
        self.GRU = nn.GRU(2 * hidden_size, hidden_size, batch_first=True)
        self.MLP_out = MLP(hidden_size + 2, hidden_size, n_layers)
        
        self.harm_linear = nn.Linear(hidden_size, n_harmonics + 1)
        #self.noise_linear = nn.Linear(hidden_size, n_bands)

    def forward(self, f0, loudness):
        
        x1 = self.MLP_f0(f0)
        x2 = self.MLP_loudness(loudness)
        x = torch.cat([x1, x2], -1)
        x = self.GRU(x)[0]
        x = torch.cat([x, f0, loudness], -1)
        x = self.MLP_out(x)
        
        harm_param = scale_function(self.harm_linear(x)) # additive synth parameters
        #noise_param = scale_function(self.noise_linear(x)) # filtered noise synth parameters
        
        # Resampling to get audio of the same size
        harm_param = resample(harm_param, self.block_size.item())
        f0_resampled = resample(f0, self.block_size.item())[:,:,0]

        # Extract amplitude total and harmonic distribution from additive synth parameters
        total_amp = harm_param[..., 0]
        harm_amps = harm_param[..., 1:]
        harm_amps = f.normalize(harm_amps, p=1, dim=2) # normalize harmonic distribution

        # Synthesize signal with harmonic additive synthesizer and filtered noise
        audio_harmonic = harmonic_synth(f0_resampled, total_amp, harm_amps, self.sampling_rate)
        #audio_noise = filtered_noise(noise_param, self.samplerate)

        # Combine both signals to get final signal
        #audio = audio_harmonic + audio_noise
        audio = audio_harmonic

        return audio