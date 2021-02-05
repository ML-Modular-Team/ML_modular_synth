import math
import numpy as np
import torch
import torch.nn.functional as f

class HarmonicSynth():
    
    def __init__(self, samplerate=16000):
        self.samplerate = samplerate

    def get_signal(self, f0, amp, harm_distrib):
        
        # Init parameters
        n_batch, n_samples, n_harmonics = np.shape(harm_distrib)

        # Create the harmonic frequencies envelopes
        ratio_harms = torch.linspace(1.0, float(n_harmonics), n_harmonics)
        freqs = f0[:, :, np.newaxis] * ratio_harms # [n_batch, n_samples, n_harmonics]
        harm_distrib_n = f.normalize(harm_distrib, p=1, dim=2) # normalize harmonic distribution
        
        # Compute instant phases
        omegas = 2 * math.pi * freqs / float(self.samplerate)
        phases = torch.cumsum(omegas, axis=1)
        
        # Compute data from phases
        data = harm_distrib_n * torch.sin(phases) # [n_batch, n_samples, n_harmonics]
        
        # Synthesize audio data with the contribution of every harmonic
        audio = amp * torch.sum(data, 2) # [n_batch, n_samples]
        audio /= np.abs(audio).max() # normalize audio

        return audio