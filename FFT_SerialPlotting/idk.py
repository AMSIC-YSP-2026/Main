import numpy as np
from scipy.fft import fft, fftfreq

# 1. Generate a dummy signal
sampling_rate = 1000.0  # Hz
time_step = 1.0 / sampling_rate
t = np.arange(0, 1.0, time_step)
signal = np.sin(2 * np.pi * 50 * t)  # 50 Hz sine wave

# 2. Compute the FFT
fft_output = fft(signal)

# 3. Calculate true frequencies
frequencies = fftfreq(len(signal), d=time_step)

# 4. Get the real magnitude (ignore negative mirror frequencies)
magnitude = np.abs(fft_output)[:len(signal)//2]
positive_frequencies = frequencies[:len(signal)//2]