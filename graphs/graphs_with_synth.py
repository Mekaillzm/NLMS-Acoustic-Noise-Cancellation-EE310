import matplotlib.pyplot as plt
import numpy as np
import AlgoNLMS as algo
import synth1 as synth

# Simulation Parameters
fs = 1600.0
duration = 5.0 # seconds
num_samples = int(fs * duration)

# 1. Generate Synthetic RIR
N_taps = 512
h_rir = synth.generate_synthetic_rir(N_taps, decayRate=0.5)
norm_h = np.linalg.norm(h_rir) #Pre-calculate norm of true RIR for misalignment

# 2. Generate Far-End Signal
x_far = synth.generate_speech_like_signal(num_samples, fs=fs, base_freq=150.0, seed=42)

# 3. Generate Near-End Signal 
s_near = synth.generate_speech_like_signal(num_samples, fs=fs, base_freq=250.0, seed=99)
# Silence the near-end for the first 2.5 seconds to allow the filter to converge purely on echo
s_near[:int(2.5 * fs)] = 0.0

# 4. Generate Background Noise
n_noise = synth.generate_background_noise(num_samples, fs=fs, std=0.02)

# 5. Create the Microphone Signal d[n]
# Convolve far-end with RIR to create the echo
y_echo = np.convolve(x_far, h_rir, mode='full')[:num_samples]

# Combine signals: d[n] = (x * h)[n] + s_near[n] + n_noise[n]
d_mic = y_echo + s_near + n_noise

# 6. Initialize NLMS Algorithm
aec = algo.AlgoNLMS()
aec.N = N_taps
aec.w = np.zeros(aec.N)
aec.x = np.zeros(aec.N)
aec.fs = fs
aec.stepSize = 0.3
aec.c = 2

# Arrays to store output for plotting
error_out = np.zeros(num_samples)
y_est_out = np.zeros(num_samples)
misalignment_out = np.zeros(num_samples) # Added for tracking filter accuracy
u = 0

# 7. Process sample by sample
print("Running NLMS simulation...")
for n in range(num_samples):
    # Update buffer with current far-end sample
    aec.updateBuffer(x_far[n])
    
    # Estimate the echo
    y_est = aec.estEcho()
    y_est_out[n] = y_est
    
    # Calculate error (which is our cleaned output signal)
    aec.calcError(d_mic[n], y_est)
    error_out[n] = aec.en
    
    # Update weights if Double-Talk Detector allows it
    if aec.checkState(d_mic[n]) == True:
        aec.updateWeights()
        u += 1
        
    # Calculate filter misalignment (in dB)
    # Using 1e-10 to prevent division by zero just in case
    misalignment_out[n] = 20 * np.log10(np.linalg.norm(h_rir - aec.w) / (norm_h + 1e-10))

print("Simulation complete. Calculating ERLE and plotting results...")

# --- NEW METRIC CALCULATION ---
# Calculate ERLE (Echo Return Loss Enhancement) over a sliding 100ms window
window_time = 0.1 
window_samples = int(window_time * fs)
window = np.ones(window_samples)

# The residual echo is what the filter failed to cancel
residual_echo = y_echo - y_est_out

# Compute moving average power of true echo and residual echo
echo_power = np.convolve(y_echo**2, window, mode='same')
residual_power = np.convolve(residual_echo**2, window, mode='same')

epsilon = 1e-10
erle_out = 10 * np.log10(echo_power / (residual_power + epsilon))
erle_out = np.clip(erle_out, 0, None) # Clip negative ERLE values that happen during initial divergence

# 8. Plotting Results
time_axis = np.arange(num_samples) / fs

# ----- FIGURE 1: Time-Domain Signals -----
plt.figure(figsize=(12, 10))

plt.subplot(4, 1, 1)
plt.title("Microphone Signal (d[n] = Echo + Near-End + Noise)")
plt.plot(time_axis, d_mic, color='red', alpha=0.7)
plt.ylabel("Amplitude")
plt.grid(True)

plt.subplot(4, 1, 2)
plt.title("True Echo vs Estimated Echo")
plt.plot(time_axis, y_echo, label='True Echo', color='orange', alpha=0.7)
plt.plot(time_axis, y_est_out, label='Estimated Echo', color='black', alpha=0.7, linestyle='--')
plt.ylabel("Amplitude")
plt.legend()
plt.grid(True)

plt.subplot(4, 1, 3)
plt.title("Clean Near-End Target (s_near[n])")
plt.plot(time_axis, s_near, color='green', alpha=0.7)
plt.ylabel("Amplitude")
plt.grid(True)

plt.subplot(4, 1, 4)
plt.title("AEC Output / Error Signal (e[n])")
plt.plot(time_axis, error_out, color='blue', alpha=0.7)
plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude")
plt.grid(True)

plt.tight_layout()

# ----- FIGURE 2: Analytical Metrics -----
plt.figure(figsize=(12, 8))

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

print(f"Total weight updates performed: {u}")
