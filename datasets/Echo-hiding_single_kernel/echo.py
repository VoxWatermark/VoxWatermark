import numpy as np
import os
import soundfile as sf
from scipy.io import wavfile
from tqdm import tqdm
from scipy.signal import lfilter

def text_to_bits(text):
    return [int(bit) for char in text for bit in format(ord(char), '08b')]

def echo_enc_single(signal, bits, d0=150, d1=200, alpha=0.5, L=4*1024):
    """
    Echo Hiding Technique with single echo kernel
    
    Args:
        signal: Audio signal (1D array)
        text_or_bits: Either text string or bit list
        d0: Delay rate for bit0 (default 150)
        d1: Delay rate for bit1 (default 200)
        alpha: Echo amplitude (default 0.5)
        L: Length of frames (default 4*1024)
    
    Returns:
        out: Stego signal
    """
    
    signal = np.atleast_2d(signal).T if signal.ndim == 1 else signal
    s_len, s_ch = signal.shape
    
    # Calculate number of complete frames
    nframe = s_len // L
    N = nframe - (nframe % 8)
    
    # Handle message length
    if len(bits) > N:
        bits = bits[:N]
        print(f'Message cropped to {N} bits')
    else:
        bits = bits.ljust(N, '0')
        print(f'Message padded to {N} bits')
    
    # Create echo kernels
    k0 = np.concatenate([np.zeros(d0), [alpha]])
    k1 = np.concatenate([np.zeros(d1), [alpha]])
    
    # Apply filtering
    echo_zro = lfilter(k0, [1], signal, axis=0)
    echo_one = lfilter(k1, [1], signal, axis=0)
    
    # Create mixer window
    window = mixer(L, bits[:N], 0, 1, 256)  
    required_length = N * L
    window = window[:required_length]  
    
    if len(window) < required_length:
        window = np.pad(window, (0, required_length - len(window)))
    mix = np.tile(window[:, np.newaxis], (1, s_ch))
    
    # Embed message
    watermarked = signal.copy()
    watermarked[:required_length] += (echo_zro[:required_length] * (1 - mix) + 
                                    echo_one[:required_length] * mix)
    
    return watermarked.squeeze()

def mixer(L, bits, lower=0, upper=1, K=256):
    """
    Mixer is the mixer signal to smooth data and spread it easier.
    
    Args:
        L: Length of segment
        bits: Binary sequence string
        lower: Lower bound of mixer signal (default 0)
        upper: Upper bound of mixer signal (default 1)
        K: Length to be smoothed (default 256)
    
    Returns:
        w_sig: Smoothed mixer signal
    """
    K = min(K, L//4 - (L//4)%4)  # Ensure valid K size
    N = len(bits)
    
    encbit = np.array([int(b) for b in bits])
    m_sig = np.repeat(encbit, L)[:N*L]  # Force exact length
    
    # Apply smoothing with proper edge handling
    hann = hanning(K)
    c = np.convolve(m_sig, hann, mode='same')
    
    # Normalize 
    valid_length = N * L - K + 1
    if valid_length <= 0:
        valid_length = N * L
    
    start_idx = K // 2
    end_idx = start_idx + valid_length
    c_valid = c[start_idx:end_idx]
    
    if len(c_valid) == 0:
        c_valid = c
    
    wnorm = c_valid / (np.max(np.abs(c_valid)) + 1e-10)  # Avoid division by zero
    return wnorm * (upper - lower) + lower

def hanning(L):
    """
    Manual implementation of hanning window to be used without SciPy.
    
    Args:
        L: Window length (must be greater than zero)
    
    Returns:
        out: Hanning window
    """
    L = int(round(L))
    if L == 1:
        return np.array([1.0])
    elif L > 1 or L == 0:
        n = np.arange(0, L)
        return 0.5 * (1 - np.cos(2 * np.pi * n / (L - 1)))
    else:
        raise ValueError('Input must be greater than zero!')

# --- Dataset Processing Pipeline ---
INPUT_DIR = "path to your unwatermarked dataset"
OUTPUT_DIR = "/echo"
os.makedirs(OUTPUT_DIR, exist_ok=True)

for fname in tqdm(os.listdir(INPUT_DIR), desc="Embedding Echo Watermarks"):
    if fname.lower().endswith(".wav"):
        try:
            sr, signal = wavfile.read(os.path.join(INPUT_DIR, fname))
            if signal.dtype == np.int16:
                signal = signal.astype(np.float32) / 32768.0

            # Convert to mono if needed
            if signal.ndim > 1:
                signal = signal[:, 0]

            # Generate random 16-bit payload
            payload = np.random.choice([0, 1], size=16)
            payload_str = ''.join(map(str, payload))
            
            # Apply echo hiding
            watermarked = echo_enc_single(signal, payload_str)

            # Save with payload in filename
            output_path = os.path.join(OUTPUT_DIR, f"echo_{payload_str}_{fname}")

            if np.max(np.abs(watermarked)) > 1.0:
                watermarked = watermarked / np.max(np.abs(watermarked))

            sf.write(output_path, watermarked, sr, subtype='PCM_16')

        except Exception as e:
            print(f"Failed on {fname}: {str(e)}")
