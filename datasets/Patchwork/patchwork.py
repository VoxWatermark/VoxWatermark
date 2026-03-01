import numpy as np
import os
import soundfile as sf
from scipy.io import wavfile
from scipy.fftpack import dct, idct
from tqdm import tqdm

# Hyperparameters 
fs = 3000  # starting frequency for watermark embedding
fe = 7000  # ending frequency for watermark embedding
k1 = 0.195
k2 = 0.08

def patchwork_watermark_embed(signal, watermark, sr=16000):
    """
    Embeds watermark using Patchwork multilayer technique
    Args:
        signal: 1D numpy array of audio samples
        watermark: 1D binary array (0s and 1s)
        sr: sampling rate (default 16000)
    Returns:
        watermarked_signal: 1D numpy array
    """
    L = len(signal)

    # Convert frequency range to DCT indices
    si = int(fs/(sr/L))
    ei = int(fe/(sr/L))

    # Apply DCT
    X = dct(signal, type=2, norm='ortho')
    Xs = X[si:(ei+1)]
    Ls = len(Xs)

    # Adjust length to be divisible by 2*watermark_length
    if Ls % (len(watermark)*2) != 0:
        Ls -= Ls % (len(watermark)*2)
        Xs = Xs[:Ls]

    # Create paired segments
    Xsp = np.dstack((Xs[:Ls//2], Xs[:(Ls//2-1):-1])).flatten()

    segments = np.array_split(Xsp, len(watermark)*2)
    watermarked_segments = []

    for i in range(0, len(segments), 2):
        j = i//2 + 1
        rj = k1 * np.exp(-k2*j)

        m1j = np.mean(np.abs(segments[i]))
        m2j = np.mean(np.abs(segments[i+1]))
        mj = (m1j + m2j)/2
        mmj = min(m1j, m2j)

        # Apply watermarking
        if watermark[j-1] == 0 and (m1j - m2j) < rj * mmj:
            m1j = mj + (rj*mmj/2)
            m2j = mj - (rj*mmj/2)
        elif watermark[j-1] == 1 and (m2j - m1j) < rj * mmj:
            m1j = mj - (rj*mmj/2)
            m2j = mj + (rj*mmj/2)

        # Scale segments
        watermarked_segments.append(segments[i] * (m1j/np.mean(np.abs(segments[i]))))
        watermarked_segments.append(segments[i+1] * (m2j/np.mean(np.abs(segments[i+1]))))

    # Reconstruct watermarked signal
    Ysp = np.hstack(watermarked_segments)
    Ys = np.hstack([Ysp[::2], Ysp[-1::-2]])

    Y = X.copy()
    Y[si:(si+Ls)] = Ys
    return idct(Y, type=2, norm='ortho')

# Dataset processing
INPUT_DIR = "path to your unwatermarked dataset"
OUTPUT_DIR = "/patchwork_watermarked"
os.makedirs(OUTPUT_DIR, exist_ok=True)

for fname in tqdm(os.listdir(INPUT_DIR), desc="Embedding Patchwork Watermarks"):
    if fname.lower().endswith(".wav"):
        try:
            # Read and normalize audio
            sr, signal = wavfile.read(os.path.join(INPUT_DIR, fname))
            if signal.dtype == np.int16:
                signal = signal.astype(np.float32) / 32768.0

            # Convert to mono if needed
            if signal.ndim > 1:
                signal = signal[:, 0]

            # Generate random 16-bit payload
            payload_bits = np.random.randint(0, 2, size=16)

            # Apply watermarking
            watermarked = patchwork_watermark_embed(signal, payload_bits, sr=sr)

            # Save with payload in filename
            payload_str = ''.join(map(str, payload_bits))
            output_path = os.path.join(OUTPUT_DIR, f"patchwork_{payload_str}_{fname}")

            # Ensure proper scaling before saving
            if np.max(np.abs(watermarked)) > 1.0:
                watermarked = watermarked / np.max(np.abs(watermarked))

            sf.write(output_path, watermarked, sr, subtype='PCM_16')

        except Exception as e:
            print(f"Failed on {fname}: {str(e)}")
