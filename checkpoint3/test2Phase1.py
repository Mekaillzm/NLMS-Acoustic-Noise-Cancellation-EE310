import numpy as np
import wave
import pyaudio
import AlgoNLMS as algo
import tkinter as tk
from tkinter import filedialog
from mutagen.flac import FLAC
from mutagen.wave import WAVE
import collections # Added for the delay buffer

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
    
    # --- ALIGNMENT / DELAY CALIBRATION ---
    # Heuristically set this to match your hardware/OS latency.
    # Start around 50ms-100ms for typical PyAudio blocking setups.
    HEURISTIC_DELAY_MS = 60 
    DELAY_SAMPLES = int((HEURISTIC_DELAY_MS / 1000.0) * FS)
    
    # Create a ring buffer pre-filled with zeros
    # When we append a new sample, the oldest sample is automatically pushed out
    delay_buffer = collections.deque(np.zeros(DELAY_SAMPLES), maxlen=DELAY_SAMPLES)
    
    # --- Algorithm Initialization ---
    nlms = algo.AlgoNLMS()
    nlms.fs = FS 
    
    # Lowering step size slightly is recommended when testing alignment
    nlms.stepSize = 0.1 
    
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
    print(f"Hardware Delay Buffer: {HEURISTIC_DELAY_MS}ms ({DELAY_SAMPLES} samples)")
    print("Playing far-end, recording near-end. REMAIN SILENT!")
    print("Watch/listen as the echo fades. Press Ctrl+C to stop.")
    print("==================================================")
    
    # Read the first block from the file
    data = wf.readframes(CHUNK)
    
    try:
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
                
                # Push the current far-end sample into the queue
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
    
    print(f"Success! Listen to '{OUTPUT_FILE}' to hear the echo fade.")

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