import numpy as np
import wave
import pyaudio
import AlgoNLMS as algo
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
    from mutagen.wave import WAVE

    # Load the WAV file
    audio = WAVE(path)

    # Get the sample rate
    sample_rate = audio.info.sample_rate

    print(f"Sampling Rate: {sample_rate} Hz")
    return sample_rate

def run_phase_0_demo(farEndPath, FS):
    # --- Configuration ---
    CHUNK = 512  # Number of samples per block
    
    # PLACEHOLDER PATH - Update this to point to your 48kHz WAV file
    FAR_END_FILE = farEndPath 
    OUTPUT_FILE = "Output/phase 0.wav"
    
    # --- Algorithm Initialization ---
    nlms = algo.AlgoNLMS()
    nlms.fs = FS # Overwrite placeholder sample rate to match the file/mic
    
    print("Initializing audio streams...")
    try:
        wf = wave.open(FAR_END_FILE, 'rb')
    except FileNotFoundError:
        print(f"Error: Could not find '{FAR_END_FILE}'. Please update the placeholder path.")
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
    print("Starting Phase 0: Baseline (Adaptation Disabled)")
    print("Playing far-end, recording near-end. Speak into the mic!")
    print("Press Ctrl+C to stop early.")
    print("==================================================")
    
    # Read the first block from the file
    data = wf.readframes(CHUNK)
    
    try:
        # Loop continues as long as we have full chunks of data
        while len(data) == CHUNK * 2: # 2 bytes per sample (int16)
            
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
                
                # PHASE 0 SPECIFIC: Adaptation Disabled
                # We purposefully skip calling nlms.checkState(d_n) 
                # We purposefully skip calling nlms.updateWeights()
                # The weights (self.w) remain at np.zeros(self.N).
                
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
    
    print(f"Success! Listen to '{OUTPUT_FILE}' to hear the baseline raw echo.")

if __name__ == "__main__":
    root = tk.Tk()

    # 2. Hide the main GUI window so only the file dialog appears
    root.withdraw()

    # 3. Open the file dialog and capture the path
    farEndPath = filedialog.askopenfilename(title = "Select a file path for the far end audio")

    # 4. Check if a file was selected (it returns an empty string if cancelled)
    if farEndPath:
        print(f"You selected: {farEndPath}")
        fs = checkWave(farEndPath)
        run_phase_0_demo(farEndPath, fs)

    else:
        print("No file selected.")
        
