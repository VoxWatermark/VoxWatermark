import numpy as np
import os
import soundfile as sf
from scipy.io import wavfile
from tqdm import tqdm

class QIM:
    def __init__(self, delta=0.01, step=100):
        self.delta = delta  # Quantization step size
        self.step = step    # Sample interval between embedded bits

    def embed(self, x, m):
        x = x.astype(float)
        d = self.delta
        # Repeat message to match selected samples
        repeated_m = np.tile(m, len(x) // (self.step * len(m)) + 1)[:len(x)//self.step]
        # Apply QIM to selected samples
        for i, bit in enumerate(repeated_m):
            idx = i * self.step
            x[idx] = np.round(x[idx]/d) * d + (-1)**(bit+1) * d/4
        return x

    def random_msg(self, l):
        return np.random.randint(0, 2, l)

# === Paths ===
INPUT_DIR = "path to your unwatermarked dataset"
OUTPUT_DIR = "qim_watermarked"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Adjust the parameters 
qim = QIM(delta=0.01, step=100)  

for fname in tqdm(os.listdir(INPUT_DIR), desc="Embedding QIM"):
    if not fname.lower().endswith(".wav"):
        continue
    
    try:
        # Read and normalize
        sr, signal = wavfile.read(os.path.join(INPUT_DIR, fname))
        if signal.dtype == np.int16:
            signal = signal.astype(np.float32) / 32768.0
        if signal.ndim > 1:
            signal = signal[:, 0]  # mono

        # Generate and embed 16-bit message
        msg = qim.random_msg(16)
        watermarked = qim.embed(signal, msg)

        # Save
        payload_str = ''.join(map(str, msg))
        out_path = os.path.join(OUTPUT_DIR, f"qim_{payload_str}_{fname}")
        sf.write(out_path, watermarked, sr, subtype='PCM_16')

    except Exception as e:
        print(f"Failed {fname}: {str(e)}")
