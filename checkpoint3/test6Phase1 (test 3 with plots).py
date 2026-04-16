import numpy as np
import wave
import pyaudio
import AlgoNLMS as algo
import tkinter as tk
from tkinter import filedialog
from mutagen.flac import FLAC
from mutagen.wave import WAVE
import collections 
import time
import matplotlib.pyplot as plt # Added for plotting

# ==========================================
# Real-Time Demo Environment Implementation
# ==========================================
def checkFlac(path):
    audio = FLAC(path)
    sample_rate = audio.info.sample_rate
    print(f"Sampling Rate: {sample_rate} Hz")
    return sample_rate

def checkWave(path):
    audio = WAVE(path)
    sample_rate = audio.info.sample_rate
    print(f"Sampling Rate: {sample_rate} Hz")
    return sample_rate

def auto_calibrate_delay(stream, FS, chunk_size=512):
    print("\n--- AUTO-CALIBRATION ---")
    print("Playing white noise burst to measure hardware delay...")
    print("Please remain silent for 1 second.")
    
    # 1. Generate 1 second of white noise (kept at a low volume of 0.1)
    burst_length = FS 
    white_noise = np.random.normal(0, 0.1, burst_length).astype(np.float32)
    
    recorded_audio = []
    
    # 2. Play the burst and record the microphone simultaneously
    for i in range(0, burst_length, chunk_size):
        block = white_noise[i:i+chunk_size]
        
        # Pad the last block if it's shorter than CHUNK
        if len(block) < chunk_size:
            block = np.pad(block, (0, chunk_size - len(block)), 'constant')
            
        # Convert float32 to int16 for playing
        play_data = (block * 32768.0).astype(np.int16).tobytes()
        
        stream.write(play_data)
        mic_data = stream.read(chunk_size, exception_on_overflow=False)
        
        # Store the recorded float data
        mic_block = np.frombuffer(mic_data, dtype=np.int16).astype(np.float32) / 32768.0
        recorded_audio.extend(mic_block)
        
    recorded_audio = np.array(recorded_audio[:burst_length])
    
    # 3. Calculate cross-correlation
    print("Calculating cross-correlation...")
    correlation = np.correlate(recorded_audio, white_noise, mode='full')
    
    # Slice the second half where delay > 0 (audio arrives *after* it is played)
    correlation_positive = correlation[len(white_noise)-1:] 
    
    # 4. Find the index of the highest peak
    delay_samples = np.argmax(correlation_positive)
    delay_ms = (delay_samples / FS) * 1000
    
    # 5. Sanity Check
    if delay_samples <= 0 or delay_ms > 500:
        print("Warning: Calibration yielded an unrealistic delay. Defaulting to 60ms.")
        delay_samples = int((60 / 1000.0) * FS)
    else:
        print(f"Calibration complete! Hardware delay detected: {delay_samples} samples ({delay_ms:.2f} ms)\n")
    
    # Pause briefly to ensure room echoes die down before starting the real filter
    time.sleep(0.5) 
    
    return delay_samples


def run_phase_1_demo(farEndPath, FS):
    # --- Configuration ---
    CHUNK = 512  
    FAR_END_FILE = farEndPath 
    OUTPUT_FILE = "Output/phase 1.wav"
    
    # --- Algorithm Initialization ---
    nlms = algo.AlgoNLMS()
    nlms.fs = FS 
    nlms.stepSize = 0.1 # Slower step size to prevent divergence
    
    print("Initializing audio streams...")
    try:
        wf = wave.open(FAR_END_FILE, 'rb')
    except FileNotFoundError:
        print(f"Error: Could not find '{FAR_END_FILE}'. Please update the path.")
        return

    if wf.getframerate() != FS:
        print(f"Warning: The .wav file framerate is {wf.getframerate()}Hz, but expected {FS}Hz.")
        
    p = pyaudio.PyAudio()
    
    # Open full-duplex stream
    stream = p.open(format=pyaudio.paInt16,
                    channels=1,
                    rate=FS,
                    input=True,
                    output=True,
                    frames_per_buffer=CHUNK)
                    
    # --- Run Auto-Calibration ---
    DELAY_SAMPLES = int(auto_calibrate_delay(stream, FS, CHUNK))
    
    # Initialize the accurately sized ring buffer
    delay_buffer = collections.deque(np.zeros(DELAY_SAMPLES), maxlen=DELAY_SAMPLES)
                    
    # Open file to save the error output
    out_wf = wave.open(OUTPUT_FILE, 'wb')
    out_wf.setnchannels(1)
    out_wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
    out_wf.setframerate(FS)

    # Tracking arrays for plotting
    full_mic_signal = []
    full_error_signal = []

    print("==================================================")
    print("Starting Phase 1: Convergence (Adaptation Enabled)")
    print("Playing far-end, recording near-end. REMAIN SILENT!")
    print("Watch/listen as the echo fades. Press Ctrl+C to stop.")
    print("==================================================")
    
    data = wf.readframes(CHUNK)
    
    try:
        while len(data) == CHUNK * 2: 
            
            stream.write(data)
            mic_data = stream.read(CHUNK, exception_on_overflow=False)
            
            far_end_block = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
            mic_block = np.frombuffer(mic_data, dtype=np.int16).astype(np.float32) / 32768.0
            
            error_block = np.zeros(CHUNK, dtype=np.float32)
            
            for i in range(CHUNK):
                
                # Push the current far-end sample into the queue
                delay_buffer.append(far_end_block[i])
                # Pop the oldest sample (now perfectly aligned)
                x_n_aligned = delay_buffer[0] 
                
                d_n = mic_block[i]
                
                nlms.updateBuffer(x_n_aligned)
                y_est = nlms.estEcho()
                nlms.calcError(d_n, y_est)
                
                if nlms.checkState(d_n):
                    nlms.updateWeights()
                
                error_block[i] = nlms.en
                
            # Store blocks for later plotting
            full_mic_signal.extend(mic_block)
            full_error_signal.extend(error_block)
                
            error_out_int16 = (error_block * 32768.0).astype(np.int16)
            out_wf.writeframes(error_out_int16.tobytes())
            
            data = wf.readframes(CHUNK)

    except KeyboardInterrupt:
        print("\nDemo interrupted by user.")
        
    print("Demo complete. Cleaning up streams...")
    
    stream.stop_stream()
    stream.close()
    p.terminate()
    wf.close()
    out_wf.close()
    
    print(f"Success! Listen to '{OUTPUT_FILE}' to hear the echo fade.")

    # ==========================================
    # Plotting Waveforms and ERLE
    # ==========================================
    print("Generating plots...")
    
    mic_array = np.array(full_mic_signal)
    error_array = np.array(full_error_signal)
    
    # Calculate power using a sliding window (e.g., 50ms)
    window_size = int(0.05 * FS)
    eps = 1e-10  # Prevent divide-by-zero or log(0)
    
    def moving_average(a, n=window_size):
        ret = np.cumsum(a, dtype=float)
        ret[n:] = ret[n:] - ret[:-n]
        return ret[n - 1:] / n

    # Power calculations
    mic_power = moving_average(mic_array**2, window_size)
    error_power = moving_average(error_array**2, window_size)
    
    # ERLE Calculation in dB
    erle = 10 * np.log10((mic_power + eps) / (error_power + eps))
    
    # Time axes
    t_audio = np.arange(len(mic_array)) / FS
    t_erle = np.arange(len(erle)) / FS + (window_size / (2 * FS))
    
    plt.figure(figsize=(10, 8))
    
    # Plot 1: Input Microphone Signal
    plt.subplot(3, 1, 1)
    plt.plot(t_audio, mic_array, color='blue', alpha=0.7)
    plt.title("Microphone Signal (Input with Echo)")
    plt.ylabel("Amplitude")
    plt.grid(True)
    
    # Plot 2: Output Error Signal
    plt.subplot(3, 1, 2)
    plt.plot(t_audio, error_array, color='green', alpha=0.7)
    plt.title("Error Signal (Output cancelled)")
    plt.ylabel("Amplitude")
    plt.grid(True)
    
    # Plot 3: ERLE
    plt.subplot(3, 1, 3)
    plt.plot(t_erle, erle, color='red', linewidth=1.5)
    plt.axhline(y=15, color='orange', linestyle='--', label='Acceptable (15 dB)')
    plt.axhline(y=25, color='green', linestyle='--', label='Excellent (25 dB)')
    plt.title("Echo Return Loss Enhancement (ERLE)")
    plt.xlabel("Time (seconds)")
    plt.ylabel("ERLE (dB)")
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    root = tk.Tk()
    root.withdraw()

    farEndPath = filedialog.askopenfilename(title="Select a file path for the far end audio")

    if farEndPath:
        print(f"You selected: {farEndPath}")
        fs = checkWave(farEndPath) 
        run_phase_1_demo(farEndPath, fs)
    else:
        print("No file selected.")