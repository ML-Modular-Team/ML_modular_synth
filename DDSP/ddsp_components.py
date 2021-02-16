import math
import numpy as np
import torch
import torch.nn as nn
import torch.fft as fft

# Additive Harmonic Synthesizer
def harmonic_synth(f0, total_amp, harm_amps, samplerate):
    
    # Init parameters
    batch_size, n_samples, n_harmonics = np.shape(harm_amps)

    # Create the harmonic frequencies envelopes
    ratio_harms = torch.linspace(1.0, float(n_harmonics), n_harmonics)
    freqs = f0[:, :, np.newaxis] * ratio_harms.to(f0) # [batch_size, n_samples, n_harmonics]
    freqs = torch.where(freqs > (samplerate/2), torch.zeros_like(freqs), freqs)
    
    # Compute instant phases
    omegas = 2 * math.pi * freqs / float(samplerate)
    phases = torch.cumsum(omegas, axis=1)
    
    # Compute data from phases
    data = harm_amps * torch.sin(phases) # [batch_size, n_samples, n_harmonics]
    
    # Synthesize audio data with the contribution of every harmonic
    audio = total_amp * torch.sum(data, 2) # [batch_size, n_samples]

    return audio


# Subtractive Synthesizer with Noise Filtering
def filtered_noise(noise_mag, samplerate):

    # Init parameters
    batch_size, n_samples, n_filter_banks = np.shape(noise_mag)

    # Create noise signal
    noise = torch.rand([batch_size, n_samples]) * 2 - 1 # noise values between -1 and 1

    # Compute inpulse response from magnitude
    noise_mag_compl = torch.complex(noise_mag, torch.zeros([batch_size, n_samples, n_filter_banks]))
    impulse_response = torch.fft.irfft(noise_mag_compl)

    # Filter noise signal
    audio = nn.Conv1d(noise, impulse_response)

    return audio