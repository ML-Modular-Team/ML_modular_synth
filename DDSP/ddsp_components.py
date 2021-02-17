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
def filtered_noise(magnitudes, block_size):

    # Init parameters
    batch_size, n_samples, n_bands = np.shape(magnitudes)
    
    # Create noise signal
    noise = (torch.rand(batch_size, n_samples, block_size)).to(magnitudes) * 2 - 1 # noise values between -1 and 1

    # Compute frequency impulse response
    magnitudes = torch.stack([magnitudes, torch.zeros_like(magnitudes)], -1) # add a dimension
    magnitudes = torch.view_as_complex(magnitudes) # transform in complex before inverse FFT
    impulse_response = torch.fft.irfft(magnitudes)
    
    # Window impulse response and make it causal
    filter_size = impulse_response.shape[-1] # [2 * n_bands - 2]
    impulse_response = torch.roll(impulse_response, filter_size // 2, -1)
    win = torch.hann_window(filter_size, dtype=impulse_response.dtype, device=impulse_response.device)
    impulse_response = impulse_response * win

    # Zero padding before convolution
    impulse_response = nn.functional.pad(impulse_response, (0, block_size - filter_size))
    impulse_response = torch.roll(impulse_response, -filter_size // 2, -1)

    # Filter noise signal with frequency impulse response
    noise = nn.functional.pad(noise, (0, noise.shape[-1]))
    impulse_response = nn.functional.pad(impulse_response, (impulse_response.shape[-1], 0))
    audio = fft.irfft(fft.rfft(noise) * fft.rfft(impulse_response))
    audio = audio[..., audio.shape[-1] // 2:].contiguous()
    audio = audio.reshape(audio.shape[0], -1, 1)

    return audio[:,:,0] # [batch_size, n_samples * block_size]