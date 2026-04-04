import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve
import AlgoNLMS as algo
import synth1  # Assuming the functions above are saved in synth1.py

# 1. Load the far-end signal from the FLAC file
# (Replace 'far_end_audio.flac' with your actual file path)
x_signal, fs = synth1.read_flac_signal('clean_audio.flac')

# 2. Generate the room environment characteristics
N_taps = 2048  # Should match the filter length in AlgoNLMS
h_rir = synth1.generate_synthetic_rir(N_taps, decayRate=0.005)

# 3. Simulate the echo by convolving the far-end signal with the RIR
# Mode='full' creates a tail, but we slice it to match the original signal length
echo_signal = convolve(x_signal, h_rir, mode='full')[:len(x_signal)]

# 4. Generate near-end speech and background noise to match the flac file's length
near_end_signal = synth1.generate_speech_like_signal(len(x_signal), fs=fs)
noise_signal = synth1.generate_background_noise(len(x_signal), fs=fs, std=0.02)

# 5. Create the microphone signal: d(n) = echo + near_end + noise
# We delay the near-end speech slightly so the filter has time to converge on the echo first
near_end_signal[:int(fs * 0.5)] = 0.0 
d_signal = echo_signal + near_end_signal + noise_signal

# 6. Initialize the AEC algorithm
aec = algo.AlgoNLMS()
aec.fs = fs      # Overwrite the placeholder sample rate with the FLAC's actual rate
aec.N = N_taps   # Ensure filter lengths match

# Arrays to store our outputs for plotting
error_signal = np.zeros(len(x_signal))
estimated_echo = np.zeros(len(x_signal))
misalignment_out = np.zeros(len(x_signal)) # Array to track filter misalignment

h_rir_power = np.sum(h_rir**2) + 1e-10 # Pre-compute true RIR power for misalignment math

# 7. Run the sample-by-sample NLMS processing loop
print("Processing audio with NLMS...")
for i in range(len(x_signal)):
    xn = x_signal[i]
    dn = d_signal[i]
    
    # Push new far-end sample into the buffer
    aec.updateBuffer(xn)
    
    # Estimate the current echo
    yEst = aec.estEcho()
    estimated_echo[i] = yEst
    
    # Calculate the error (which is our output audio)
    aec.calcError(dn, yEst)
    error_signal[i] = aec.en
    
    # Update weights only if double-talk is NOT detected
    if aec.checkState(dn):
        aec.updateWeights()
        
    # Calculate System Misalignment (how close the filter w is to the true RIR h)
    misalignment_out[i] = 10 * np.log10(np.sum((h_rir - aec.w)**2) / h_rir_power)

print("Processing complete. Generating plots...")

# =====================================================================
# FIGURE 1: SIGNAL WAVEFORMS
# =====================================================================
plt.figure(figsize=(14, 10))

plt.subplot(4, 1, 1)
plt.plot(x_signal, label='Far-End Signal (x)', color='royalblue')
plt.title('Far-End Signal (From .flac)')
plt.ylabel('Amplitude')
plt.legend(loc='upper right')

plt.subplot(4, 1, 2)
plt.plot(d_signal, label='Microphone Signal (d = echo + near + noise)', color='darkorange')
plt.title('Microphone Signal')
plt.ylabel('Amplitude')
plt.legend(loc='upper right')

plt.subplot(4, 1, 3)
plt.plot(near_end_signal, label='True Near-End Speech', color='forestgreen')
plt.title('Desired Near-End Signal (What we want to isolate)')
plt.ylabel('Amplitude')
plt.legend(loc='upper right')

plt.subplot(4, 1, 4)
plt.plot(error_signal, label='AEC Output / Error Signal (e)', color='crimson')
plt.title('AEC Output (Echo Cancelled)')
plt.xlabel('Samples')
plt.ylabel('Amplitude')
plt.legend(loc='upper right')

plt.tight_layout()

# =====================================================================
# FIGURE 2: AEC PERFORMANCE METRICS (ERLE & Misalignment)
# =====================================================================

# Setup variables for the metrics
time_axis = np.arange(len(x_signal)) / fs
window_time = 0.1 
window_samples = int(window_time * fs)
window = np.ones(window_samples)

# The residual echo is what the filter failed to cancel
residual_echo = echo_signal - estimated_echo

# Compute moving average power of true echo and residual echo
echo_power = np.convolve(echo_signal**2, window, mode='same')
residual_power = np.convolve(residual_echo**2, window, mode='same')

epsilon = 1e-10
erle_out = 10 * np.log10(echo_power / (residual_power + epsilon))
erle_out = np.clip(erle_out, 0, None) # Clip negative ERLE values that happen during initial divergence

plt.figure(figsize=(14, 8))

plt.subplot(2, 1, 1)
plt.title("Echo Return Loss Enhancement (ERLE)")
plt.plot(time_axis, erle_out, color='purple')
plt.ylabel("ERLE (dB)")
plt.axhline(y=15, color='r', linestyle='--', label='Acceptable Target (15 dB)')
plt.axhline(y=25, color='g', linestyle='--', label='Excellent Target (25 dB)')
plt.legend()
plt.grid(True)

plt.subplot(2, 1, 2)
plt.title("Filter Misalignment")
plt.plot(time_axis, misalignment_out, color='brown')
plt.xlabel("Time (seconds)")
plt.ylabel("Misalignment (dB)")
plt.grid(True)

plt.tight_layout()
plt.show()
