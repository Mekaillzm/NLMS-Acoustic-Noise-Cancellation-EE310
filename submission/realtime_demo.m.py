import numpy as np
import wave
import pyaudio
import collections  
import time
import os
import matplotlib.pyplot as plt
from IPython.display import Audio, display
from mutagen.wave import WAVE
import tkinter as tk
from tkinter import filedialog
from mutagen.flac import FLAC
from mutagen.wave import WAVE
import main_algorithm as algo

FAR_END_FILE = "audio sample 1.wav"
OUTPUT_DIR = "Output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

CHUNK = 512  # number of samples per block
FS = 16000  # sampling rate


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

        
def calibrate_geigel_threshold(duration_sec=5):

    print("Playing audio. REMAIN COMPLETELY SILENT.")
    time.sleep(1.5)

    #algorithm initialization
    nlms_cal = algo.AlgoNLMS()
    nlms_cal.fs = FS  # overwrite placeholder sample rate to match the file

    p   = pyaudio.PyAudio()
    wf  = wave.open(FAR_END_FILE, 'rb')

    # open full-duplex stream (Read from mic, write to speakers)
    stream = p.open(format=pyaudio.paInt16,
                    channels=1,
                    rate=FS,
                    input=True,
                    output=True,
                    frames_per_buffer=CHUNK)

    ratios = []
    samples_needed = duration_sec * FS
    samples_read   = 0

    # Read the first block from the file
    data = wf.readframes(CHUNK)

    while samples_read < samples_needed and len(data) == CHUNK * 2:  # 2 bytes per sample (int16)

        # 1. Play the block through the speakers (far-end)
        stream.write(data)

        # 2. Read the block from the microphone (near-end + acoustic echo)
        mic_data = stream.read(CHUNK, exception_on_overflow=False)

        # Convert binary data to floating point arrays (-1.0 to 1.0)
        far_end_block = np.frombuffer(data,     dtype=np.int16).astype(np.float32) / 32768.0
        mic_block     = np.frombuffer(mic_data, dtype=np.int16).astype(np.float32) / 32768.0

        # 3. Process sample-by-sample to fill the x(n) buffer
        for i in range(CHUNK):
            nlms_cal.updateBuffer(far_end_block[i])
            maxX = np.max(np.abs(nlms_cal.x))  # same logic, max magnitude of x(n)
            if maxX > 1e-6:  # avoid division by zero at the very start before buffer fills
                ratios.append(np.abs(mic_block[i]) / maxX)

        samples_read += CHUNK

        # Read the next block
        data = wf.readframes(CHUNK)

    # Teardown
    stream.stop_stream()
    stream.close()
    p.terminate()
    wf.close()

    if not ratios:
        print("Warning: no valid ratios measured. Defaulting to c=0.7")
        return 0.7

    baseline      = float(np.percentile(ratios, 95))   # 95th percentile of far-end-only ratios
    recommended_c = float(np.clip(baseline * 1.3, 0.5, 0.95))  # 30% margin above baseline

    print(f"\nCalibration complete:")
    print(f"  Baseline energy ratio (95th percentile) : {baseline:.4f}")
    print(f"  Recommended threshold c                 : {recommended_c:.4f}")

    # Plots
    plt.figure(figsize=(8, 3))
    plt.hist(ratios, bins=80, color='steelblue', edgecolor='none', alpha=0.8)
    plt.axvline(baseline,       color='orange', linestyle='--', label=f'95th pct  = {baseline:.3f}')
    plt.axvline(recommended_c,  color='green',  linestyle='--', label=f'Recommended c = {recommended_c:.3f}')
    plt.xlabel("|d(n)| / max|x(n)|")
    plt.ylabel("Count")
    plt.title("Geigel Calibration — Energy Ratio Distribution (Far-End Only, You Silent)")
    plt.legend()
    plt.tight_layout()
    plt.show()

    return recommended_c

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



def run_phase_1_demo(farEndPath, FS):
    # --- Configuration ---
    CHUNK = 512  
    FAR_END_FILE = farEndPath 
    OUTPUT_FILE = "Output/phase 1.wav"
    
    # --- Algorithm Initialization ---
    nlms = algo.AlgoNLMS()
    nlms.fs = FS 
    
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
    return nlms, delay_buffer


def run_phase_2_demo(farEndPath, FS, nlms, delay_buffer):
    
    CHUNK = 512  # Number of samples per block
    FAR_END_FILE = farEndPath
    OUTPUT_FILE = "Output/phase 2.wav"

    try:
        wf = wave.open(FAR_END_FILE, 'rb')
    except FileNotFoundError:
        return nlms, delay_buffer

    p = pyaudio.PyAudio()

    # Open full-duplex stream read from mic, write to speakers
    stream = p.open(format=pyaudio.paInt16,
                    channels=1,
                    rate=FS,
                    input=True,
                    output=True,
                    frames_per_buffer=CHUNK)

    # save the error output
    out_wf = wave.open(OUTPUT_FILE, 'wb')
    out_wf.setnchannels(1)
    out_wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
    out_wf.setframerate(FS)

    print("==================================================")
    print("Starting Phase 2: Double-Talk")
    print("SPEAK into the microphone.")
    print("Your voice should be heard. Echo should stay suppressed.")
    print("==================================================")

    data = wf.readframes(CHUNK)

    dtd_log = []   # track fraction of samples where DTD fired

    try:
        # Loop continues as long as we have full chunks of data
        while len(data) == CHUNK * 2:  # 2 bytes per sample 

            # 1. Play the block through the speakers 
            stream.write(data)

            # 2. Read the block from the microphone (near-end + acoustic echo)
            mic_data = stream.read(CHUNK, exception_on_overflow=False)

            # Binary to floating point arrays (-1.0 to 1.0)
            far_end_block = np.frombuffer(data,     dtype=np.int16).astype(np.float32) / 32768.0
            mic_block     = np.frombuffer(mic_data, dtype=np.int16).astype(np.float32) / 32768.0

            error_block   = np.zeros(CHUNK, dtype=np.float32)
            dtd_block_sum = 0

            # 3. Process the audio sample by sample within the loop iteration
            for i in range(CHUNK):

                # Push the current sample into the queue
                delay_buffer.append(far_end_block[i])
                # Pop the oldest sample (now perfectly aligned)
                x_n_aligned = delay_buffer[0]

                d_n = mic_block[i]

                # NLMS Pipeline
                nlms.updateBuffer(x_n_aligned)
                y_est = nlms.estEcho()
                nlms.calcError(d_n, y_est)

                # checkState returns False when double talk or near end onky is detected, pauses adaptation to prevent the filter from diverging on ur voice
                state = nlms.checkState(d_n)
                dtd_block_sum += (0 if state else 1)

                if state:
                    nlms.updateWeights()

                # Capture the error signal
                error_block[i] = nlms.en

            # Save output
            out_wf.writeframes((error_block * 32768.0).astype(np.int16).tobytes())

            dtd_log.append(dtd_block_sum / CHUNK)  # fraction of samples DTD fired this block

            # Read the next block
            data = wf.readframes(CHUNK)

    except KeyboardInterrupt:
        print("\nDemo interrupted by user.")

    print("Demo complete")

    # Teardown
    stream.stop_stream()
    stream.close()
    p.terminate()
    wf.close()
    out_wf.close()

    print(f"Success! Listen to 'Output/phase 2.wav' to hear your voice with suppressed echo.")

    # Plot DTD activity
    if dtd_log:
        t = np.arange(len(dtd_log)) * CHUNK / FS
        plt.figure(figsize=(9, 3))
        plt.fill_between(t, dtd_log, color='tomato', alpha=0.7,
                         label='DTD active (adaptation frozen)')
        plt.xlabel("Time (s)")
        plt.ylabel("Fraction frozen")
        plt.title("Phase 2 — Geigel DTD Activity (1 = fully frozen, 0 = adapting freely)")
        plt.ylim(0, 1)
        plt.legend()
        plt.tight_layout()
        plt.show()

    # Return filter state, passed forward to Phase 3
    return nlms, delay_buffer




def run_phase_3_demo(farEndPath, FS, nlms, delay_buffer):

    CHUNK = 512  # Number of samples per block
    FAR_END_FILE = farEndPath
    OUTPUT_FILE = "Output/phase 3.wav"

    try:
        wf = wave.open(FAR_END_FILE, 'rb')
    except FileNotFoundError:
        return

    p = pyaudio.PyAudio()

    # Open full-duplex stream (read from mic, write to speakers)
    stream = p.open(format=pyaudio.paInt16,
                    channels=1,
                    rate=FS,
                    input=True,
                    output=True,
                    frames_per_buffer=CHUNK)

    # Open file to save output
    out_wf = wave.open(OUTPUT_FILE, 'wb')
    out_wf.setnchannels(1)
    out_wf.setsampwidth(p.get_sample_size(pyaudio.paInt16))
    out_wf.setframerate(FS)

    print("==================================================")
    print("Starting Phase 3: Acoustic Path Change")
    print("Change the acoustic environment now")
    print("Listen for the brief echo burst, then re-convergence.")
    print("==================================================")

    data = wf.readframes(CHUNK)

    # Tracking ERLE
    erle_log = []
    window_d = collections.deque(maxlen=int(0.3 * FS))  # 300ms sliding window
    window_e = collections.deque(maxlen=int(0.3 * FS))
    sample_count = 0

    full_mic_signal=[]
    full_error_signal=[]
    try:
        # Loop continues
        while len(data) == CHUNK * 2:  # 2 bytes per sample

            # 1. Play the block through the speakers 
            stream.write(data)

            # 2. Read the block from the microphone (near-end + acoustic echo)
            mic_data = stream.read(CHUNK, exception_on_overflow=False)

            # binary to floating point arrays 
            far_end_block = np.frombuffer(data,     dtype=np.int16).astype(np.float32) / 32768.0
            mic_block     = np.frombuffer(mic_data, dtype=np.int16).astype(np.float32) / 32768.0

            error_block = np.zeros(CHUNK, dtype=np.float32)

            # 3. Process the audio sample bysample within the loop iteration
            for i in range(CHUNK):

                # Push the current far end sample into the queue
                delay_buffer.append(far_end_block[i])
                # Pop the oldest sample (now perfectly aligned)
                x_n_aligned = delay_buffer[0]

                d_n = mic_block[i]

                # NLMS Pipeline
                nlms.updateBuffer(x_n_aligned)
                y_est = nlms.estEcho()
                nlms.calcError(d_n, y_est)

                if nlms.checkState(d_n):
                    nlms.updateWeights()

                # Capture error signal
                error_block[i] = nlms.en
                window_d.append(d_n ** 2)
                window_e.append(nlms.en ** 2)
                sample_count += 1
            full_mic_signal.extend(mic_block)
            full_error_signal.extend(error_block)
            # Save output
            out_wf.writeframes((error_block * 32768.0).astype(np.int16).tobytes())

            # Log ERLE every half seconds
            if sample_count % (FS // 2) < CHUNK:
                pd = np.mean(window_d) + 1e-10
                pe = np.mean(window_e) + 1e-10
                erle_log.append(10 * np.log10(pd / pe))

            data = wf.readframes(CHUNK)

    except KeyboardInterrupt:
        print("\nDemo ended.")

    print("Demo complete.")

    # Teardown
    stream.stop_stream()
    stream.close()
    p.terminate()
    wf.close()
    out_wf.close()

    print(f"Success! Listen to 'Output/phase 3.wav' to hear the echo burst then re-convergence.")

    # Plot 
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
    
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    root = tk.Tk()
    root.withdraw()

    farEndPath = filedialog.askopenfilename(title="Select a file path for the far end audio")


    if farEndPath:
        print(f"You selected: {farEndPath}")
        fs = checkWave(farEndPath) 
        
        run_phase_0_demo(farEndPath, fs)

        converged_nlms, converged_delay_buf = run_phase_1_demo(farEndPath, fs)
        
        converged_nlms, converged_delay_buf = run_phase_2_demo(FAR_END_FILE, FS, converged_nlms, converged_delay_buf)
        run_phase_3_demo(FAR_END_FILE, FS, converged_nlms, converged_delay_buf)


    else:
        print("No file selected.")


