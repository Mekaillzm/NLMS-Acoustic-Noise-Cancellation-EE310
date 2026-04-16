import numpy as np
import wave
import pyaudio
import AlgoNLMS as algo
from matplotlib import pyplot as plt
import tkinter as tk
from tkinter import filedialog
from mutagen.flac import FLAC
from mutagen.wave import WAVE

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

def run_phase_1_demo(farEndPath, FS):
    # --- Configuration ---
    CHUNK = 512  # Number of samples per block
    
    # Path to the pre-recorded far-end file
    FAR_END_FILE = farEndPath 
    OUTPUT_FILE = "Output/phase 1.wav"
    
    # --- Algorithm Initialization ---
    nlms = algo.AlgoNLMS()
    nlms.fs = FS # Overwrite placeholder sample rate to match the file/mic
    
    print("Initializing audio streams...")
    try:
        wf = wave.open(FAR_END_FILE, 'rb')
    except FileNotFoundError:
        print(f"Error: Could not find '{FAR_END_FILE}'. Please update the path.")
        return

    if wf.getframerate() != FS:
        print(f"Warning: The .wav file framerate is {wf.getframerate()}Hz, but expected {FS}Hz.")
        
    p = pyaudio.PyAudio()
    
    # Open full-duplex stream (Read from mic, write to speakers)
    stream = p.open(format=pyaudio.paInt16,
                    channels=1,
                    rate=FS,
                    input=True,
                    output=True,
                    frames_per_buffer=CHUNK)
                    
    # Open file to save the error output for review
    out_wf = wave.open(OUTPUT_FILE, 'wb')
    out_wf.setnchannels(1)
    out_wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
    out_wf.setframerate(FS)

    print("==================================================")
    print("Starting Phase 1: Convergence (Adaptation Enabled)")
    print("Playing far-end, recording near-end. REMAIN SILENT!")
    print("Watch/listen as the echo fades. Press Ctrl+C to stop.")
    print("==================================================")
    
    # Read the first block from the file
    data = wf.readframes(CHUNK)
    full_mic_signal = []
    full_error_signal = []
    
    try:
        # Loop continues as long as we have full chunks of data
        while len(data) == CHUNK * 2: 
            
            # 1. Play the block through the speakers (far-end)
            stream.write(data)
            
            # 2. Read the block from the microphone (near-end + acoustic echo)
            mic_data = stream.read(CHUNK, exception_on_overflow=False)
            
            # Convert binary data to floating point arrays (-1.0 to 1.0)
            far_end_block = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
            mic_block = np.frombuffer(mic_data, dtype=np.int16).astype(np.float32) / 32768.0
            
            error_block = np.zeros(CHUNK, dtype=np.float32)
            
            # 3. Process the audio sample-by-sample within the loop iteration
            for i in range(CHUNK):
                x_n = far_end_block[i]
                d_n = mic_block[i]
                
                # --- NLMS Pipeline ---
                nlms.updateBuffer(x_n)
                y_est = nlms.estEcho()
                nlms.calcError(d_n, y_est)
                
                # PHASE 1 SPECIFIC: Adaptation Enabled
                # We check the Geigel DTD state. Since you are silent, this should 
                # consistently return True, allowing the weights to update.
                if nlms.checkState(d_n):
                    nlms.updateWeights()
                
                # Capture the error signal
                error_block[i] = nlms.en
            
            full_mic_signal.extend(mic_block)
            full_error_signal.extend(error_block)
            # 4. Save error output to block/file (convert float32 back to int16)
            error_out_int16 = (error_block * 32768.0).astype(np.int16)
            out_wf.writeframes(error_out_int16.tobytes())
            
            # Read the next block
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
    
    print(f"Success! Listen to '{OUTPUT_FILE}' to hear the echo fade as the filter converges.")


    #plots
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

    # Hide the main GUI window so only the file dialog appears
    root.withdraw()

    # Open the file dialog and capture the path
    farEndPath = filedialog.askopenfilename(title="Select a file path for the far end audio")

    # Check if a file was selected (it returns an empty string if cancelled)
    if farEndPath:
        print(f"You selected: {farEndPath}")
        # Assuming WAV files for this demo based on your comments
        fs = checkWave(farEndPath) 
        
        # Run Phase 1
        run_phase_1_demo(farEndPath, fs)
    else:
        print("No file selected.")