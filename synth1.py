import matplotlib.pyplot as plt
import numpy as np
import AlgoNLMS as algo

aec = algo.AlgoNLMS()

def generate_synthetic_rir(N, decayRate=0.005):
        """
        Generates a synthetic exponentially decaying room impulse response.
        
        Parameters:
        N (int): Length of the filter (e.g., 1023 taps)
        decayRate (float): Controls how fast the echo dies out. 
                            Lower = more echo, Higher = less echo.
        """
        # Create an array of sample indices [0, 1, 2, ..., N-1]
        n = np.arange(N)
        
        #Generate the exponential decay 
        e = np.exp(-decayRate * n)
        
        #Generate random gaussian noise to simulate chaotic reflections
        reflections = np.random.normal(0, 1, N)
        
        # Multiply them together
        h = e * reflections
        
        # Normalize the filter so it doesn't artificially amplify the volume
        h = h / np.max(np.abs(h))# h/max(|h|)
        
        return h
def generate_speech_like_signal(length: int, fs: float = 16000.0, base_freq: float = 400.0, seed: int = None) -> np.ndarray:
    """
    Synthetic speech-like signal (far-end or near-end).
    Uses multiple sinusoids (formants) + slow amplitude envelope + tiny unvoiced noise.
    Different base_freq gives different "voices".
    """
    if seed is not None:
        np.random.seed(seed)
    
    t = np.arange(length) / fs
    signal = np.zeros(length, dtype=np.float64)
    
    # Voiced part: 6–8 formant-like sinusoids
    for i in range(7):
        freq = base_freq * (i + 1) + np.random.uniform(-80, 80)   # slight pitch jitter
        amp = np.random.uniform(0.15, 0.7)
        phase = np.random.uniform(0, 2 * np.pi)
        signal += amp * np.sin(2 * np.pi * freq * t + phase)
    
    # Tiny unvoiced (fricative) noise
    signal += np.random.normal(0, 0.08, length)
    
    # Syllabic-rate amplitude envelope (2–4 Hz modulation)
    envelope = 1.0 + 0.7 * np.sin(2 * np.pi * 3.0 * t)          # ~3 syllables/sec
    envelope = np.maximum(envelope, 0.25)                        # avoid silence gaps
    signal *= envelope
    
    # Normalise to [-0.9, 0.9] range (headroom for echo + noise)
    signal = signal / np.max(np.abs(signal)) * 0.9
    
    return signal


def generate_background_noise(length: int, fs: float = 16000.0, std: float = 0.05) -> np.ndarray:
    """
    White Gaussian noise. Scale std to control SNR.
    """
    noise = np.random.normal(0.0, std, length)
    noise = noise / np.max(np.abs(noise)) * 0.9   # keep consistent range
    return noise
