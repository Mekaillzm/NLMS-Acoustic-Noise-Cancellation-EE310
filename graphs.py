import matplotlib.pyplot as plt
import numpy as np
import AlgoNLMS as algo
import synth1 as synth




if __name__ == "__main__":
    # Simulation Parameters
    fs = 16000.0

    duration = 5.0 # seconds
    num_samples = int(fs * duration)
    
    # 1. Generate Synthetic RIR
    N_taps = 1023
    h_rir = synth.generate_synthetic_rir(N_taps, decayRate=0.5)
    
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
    
    aec.c = 2
    # Arrays to store output for plotting
    error_out = np.zeros(num_samples)
    y_est_out = np.zeros(num_samples)
    u=0

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
            #print("UPDATING WEIGHTS\n")
            aec.updateWeights()
            u= u + 1

    print("Simulation complete. Plotting results...")

    # 8. Plotting Results
    time_axis = np.arange(num_samples) / fs


    a = -d_mic + error_out
    plt.figure()
    plt.plot(time_axis, a)

    plt.figure(figsize=(12, 8))
    
    plt.subplot(3, 1, 1)
    plt.title("Microphone Signal (d[n] = Echo + Near-End + Noise)")
    plt.plot(time_axis, d_mic, color='red', alpha=0.7)
    plt.ylabel("Amplitude")
    plt.grid(True)
    
    plt.subplot(3, 1, 2)
    plt.title("Clean Near-End Target (s_near[n])")
    plt.plot(time_axis, s_near, color='green', alpha=0.7)
    plt.ylabel("Amplitude")
    plt.grid(True)
    
    plt.subplot(3, 1, 3)
    plt.title("AEC Output / Error Signal (e[n])")
    plt.plot(time_axis, error_out, color='blue', alpha=0.7)
    plt.xlabel("Time (seconds)")
    plt.ylabel("Amplitude")
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    print(u)
