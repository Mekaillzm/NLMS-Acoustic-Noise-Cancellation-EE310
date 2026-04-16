import os
import itertools
import numpy as np
import wave
import pyaudio
import AlgoNLMS as algo
import tkinter as tk
from tkinter import filedialog
from mutagen.flac import FLAC
from mutagen.wave import WAVE
import collections 
import matplotlib.pyplot as plt 

# ==========================================
# Real-Time Demo Environment Implementation
# ==========================================
def checkFlac(path):
    # Load the FLAC file
    audio = FLAC(path)
    # Get the sample rate
    sample_rate = audio.info.sample_rate
    print(f"Sampling Rate: {sample_rate} Hz")
    return sample_rate

def checkWave(path):
    # Load the WAV file
    audio = WAVE(path)
    # Get the sample rate
    sample_rate = audio.info.sample_rate
    print(f"Sampling Rate: {sample_rate} Hz")
    return sample_rate

def run_phase_1_demo(farEndPath, FS, N, stepSize, regParam, c, halt):
    # --- Configuration ---
    CHUNK = 512  # Number of samples per block
    
    # Path to the pre-recorded far-end file
    FAR_END_FILE = farEndPath 
    OUTPUT_FILE = "Output/phase 1.wav"
    
    # --- ALIGNMENT / DELAY CALIBRATION ---
    # Heuristically set this to match your hardware/OS latency.
    # Start around 50ms-100ms for typical PyAudio blocking setups.
    HEURISTIC_DELAY_MS = 200
    DELAY_SAMPLES = int((HEURISTIC_DELAY_MS / 1000.0) * FS)
    
    # Create a ring buffer pre-filled with zeros
    # When we append a new sample, the oldest sample is automatically pushed out
    delay_buffer = collections.deque(np.zeros(DELAY_SAMPLES), maxlen=DELAY_SAMPLES)
    
    # --- Algorithm Initialization ---
    nlms = algo.AlgoNLMS()
    nlms.fs = FS 
    
    # Inject Iterative Parameters
    nlms.N = N
    nlms.stepSize = stepSize 
    nlms.regParam = regParam
    nlms.c = c
    nlms.halt = halt
    
    # Resize arrays dynamically to match the current N
    nlms.w = np.zeros(N)
    nlms.x = np.zeros(N)
    
    print("Initializing audio streams...")
    try:
        wf = wave.open(FAR_END_FILE, 'rb')
    except FileNotFoundError:
        print(f"Error: Could not find '{FAR_END_FILE}'. Please update the path.")
        return

    if wf.getframerate() != FS:
        print(f"Warning: The .wav file framerate is {wf.getframerate()}Hz, but expected {FS}Hz.")
        
    p = pyaudio.PyAudio()
    
    # Open half-duplex stream (Read from mic only, NO speaker output)
    stream = p.open(format=pyaudio.paInt16,
                    channels=1,
                    rate=FS,
                    input=True,
                    output=False,
                    frames_per_buffer=CHUNK)
                    
    # Open file to save the error output for review
    # Ensure Output directory exists before writing to it
    os.makedirs("Output", exist_ok=True)
    out_wf = wave.open(OUTPUT_FILE, 'wb')
    out_wf.setnchannels(1)
    out_wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
    out_wf.setframerate(FS)

    print("==================================================")
    print(f"Running Phase 1: N={N}, step={stepSize}, reg={regParam}, c={c}")
    print(f"Hardware Delay Buffer: {HEURISTIC_DELAY_MS}ms ({DELAY_SAMPLES} samples)")
    print("Listening for externally played far-end. Recording near-end. REMAIN SILENT!")
    print("==================================================")
    
    # Read the first block from the file
    data = wf.readframes(CHUNK)
    full_mic_signal = []
    full_error_signal = []
    try:
        while len(data) == CHUNK * 2: 
            
            # 1. Read the block from the microphone (near-end + acoustic echo)
            # This read call will block for exactly (CHUNK/FS) seconds, acting as our real-time pacer
            mic_data = stream.read(CHUNK, exception_on_overflow=False)
            
            # Convert binary data to floating point arrays (-1.0 to 1.0)
            far_end_block = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
            mic_block = np.frombuffer(mic_data, dtype=np.int16).astype(np.float32) / 32768.0
            
            error_block = np.zeros(CHUNK, dtype=np.float32)
            
            # 2. Process the audio sample-by-sample within the loop iteration
            for i in range(CHUNK):
                
                # Push the current far-end file sample into the queue (reference signal)
                delay_buffer.append(far_end_block[i])
                
                # Pop the oldest sample from the queue to use as our aligned reference
                x_n_aligned = delay_buffer[0] 
                
                d_n = mic_block[i]
                
                # --- NLMS Pipeline ---
                # Feed the delayed/aligned sample to the filter
                nlms.updateBuffer(x_n_aligned)
                y_est = nlms.estEcho()
                nlms.calcError(d_n, y_est)
                
                # Phase 1: Adaptation Enabled
                if nlms.checkState(d_n):
                    nlms.updateWeights()
                
                # Capture the error signal
                error_block[i] = nlms.en
                
            full_mic_signal.extend(mic_block)
            full_error_signal.extend(error_block)
            
            # 3. Save error output to block/file (convert float32 back to int16)
            error_out_int16 = (error_block * 32768.0).astype(np.int16)
            out_wf.writeframes(error_out_int16.tobytes())
            
            # Read the next block from the file
            data = wf.readframes(CHUNK)

    except KeyboardInterrupt:
        print("\nDemo interrupted by user.")
        
    print("Demo complete. Cleaning up streams...")
    
    # Teardown
    stream.stop_stream()
    stream.close()
    p.terminate()
    wf.close()
    out_wf.close()
    
    # Plots
    print("Generating and saving plot...")
    
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
    plt.title(f"Microphone Signal (N={N}, stepSize={stepSize}, reg={regParam}, c={c})")
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
    
    # Save graph dynamically
    graph_filename = f"test graphs/NLMS_N-{N}_step-{stepSize}_reg-{regParam}_c-{c}_halt-{halt}.png"
    plt.savefig(graph_filename)
    plt.close() # Close plot to prevent memory leaks during iterations
    print(f"Saved: {graph_filename}\n")


if __name__ == "__main__":
    # Ensure the directory for graphs exists
    os.makedirs("test graphs", exist_ok=True)

    root = tk.Tk()
    root.withdraw()

    farEndPath = filedialog.askopenfilename(title="Select a file path for the far end audio")

    if farEndPath:
        print(f"You selected: {farEndPath}")
        fs = checkWave(farEndPath) 
        
        # --- Parameter Combinations Setup ---
        N_values = [511, 1023, 2047]
        stepSize_values = [0.05, 0.35, 0.7]
        regParam_values = [0.000001, 0.00001, 0.0001]
        c_values = [0.5, 0.6, 0.7]
        halt_values = [1600]
        
        # Create all unique combinations
        all_combinations = list(itertools.product(N_values, stepSize_values, regParam_values, c_values, halt_values))
        print(f"Total iterations to run: {len(all_combinations)}")
        
        # Iterate over combinations
        for idx, params in enumerate(all_combinations):
            N_val, step_val, reg_val, c_val, halt_val = params
            print(f"--- Iteration {idx + 1}/{len(all_combinations)} ---")
            run_phase_1_demo(farEndPath, fs, N_val, step_val, reg_val, c_val, halt_val)
            
    else:
        print("No file selected.")