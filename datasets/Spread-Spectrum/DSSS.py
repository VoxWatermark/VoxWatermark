import numpy as np
import warnings
import soundfile as sf
from scipy.io import wavfile
from tqdm import tqdm
import os

def dsss_enc(signal, bit, L_min=4*1024):

    s_len, s_ch = signal.shape
     
           
    L2 = int(np.floor(s_len / len(bit)))  # length of segments
    L = max(L_min, L2)                    # keep segments large enough
    nframe = int(np.floor(s_len / L))
    N = nframe - (nframe % 8)  
    
    if len(bit) > N:
        warnings.warn("Message is too long, is being cropped...")
        bits = bit[:N]
    else:
        bits = bit + '0' * (N - len(bit))

    r = np.ones(L)
    pr = np.tile(r, N)                    # extend r to N*L
    alpha = 0.005                         # embedding strength

    mix, datasig = mixer(L, bits, -1, 1, 256)
    out = np.copy(signal)
    stego = signal[:N * L, 0] + alpha * mix * pr
    out[:N * L, 0] = stego

    
    return out



def mixer(L, bits, lower=0, upper=1, K=256):
    if 2 * K > L:
        K = (L // 4) - (L // 4) % 4
    else:
        K = K - (K % 4)

    N = len(bits)
    encbit = np.array([int(b) for b in bits])
    m_sig = np.tile(encbit, (L, 1)).T.flatten()
    c = np.convolve(m_sig, hanning(K), mode='full')
    wnorm = c[(K // 2):(len(c) - K // 2 + 1)]
    wnorm = wnorm / np.max(np.abs(wnorm))
    w_sig = wnorm * (upper - lower) + lower
    m_sig = m_sig * (upper - lower) + lower
    return w_sig, m_sig


def hanning(L):
    L = int(round(L))
    if L == 1:
        return np.array([1.0])
    elif L > 1:
        n = np.arange(L)
        return 0.5 * (1 - np.cos((2 * np.pi * n) / (L - 1)))
    else:
        raise ValueError("Input must be greater than zero!")



INPUT_DIR = "path to your unwatermarked dataset"
OUTPUT_DIR = "/dsss"
os.makedirs(OUTPUT_DIR, exist_ok=True)

for fname in tqdm(os.listdir(INPUT_DIR), desc="Embedding Watermarks"):
    if fname.lower().endswith(".wav"):
        try:
            # Read and normalize audio
            sr, signal = wavfile.read(os.path.join(INPUT_DIR, fname))
            if signal.dtype == np.int16:
                signal = signal.astype(np.float32) / 32768.0

            # Ensure 2D shape (samples, channels)
            if signal.ndim == 1:
                signal = np.expand_dims(signal, axis=1)
            elif signal.ndim > 1:
                signal = signal[:, :1]  # use only the first channel

            # Generate random 16-bit payload
            payload_bits = np.random.choice([0, 1], size=16).tolist()
            payload_str = ''.join(map(str, payload_bits))
      

            # Apply DSSS encoding
            watermarked = dsss_enc(signal, payload_str)

            # Normalize and save
            watermarked = np.clip(watermarked, -1, 1)
            output_path = os.path.join(OUTPUT_DIR, f"dsss_{payload_str}_{fname}")
            sf.write(output_path, watermarked.squeeze(), sr, subtype='PCM_16')

        except Exception as e:
            print(f"Failed on {fname}: {str(e)}")

