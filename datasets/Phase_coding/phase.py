import numpy as np
import os
import soundfile as sf
from scipy.io import wavfile
from tqdm import tqdm

def getBits(text):
    return ''.join(format(ord(c), '08b') for c in text)

def phase_enc(signal, bits, L=1024):
  
    plain = signal[:, 0] if signal.ndim > 1 else signal
    plain = plain.reshape(-1, 1)
    
    I = len(plain)
    m = len(bits)
    N = I // L
    
    # Reshape with column-major order
    s = plain[:N*L].reshape(L, N, order='F')
    
    # FFT processing
    w = np.fft.fft(s, axis=0)
    Phi = np.angle(w)
    A = np.abs(w)
    
    # Phase differences
    DeltaPhi = np.zeros((L, N))
    for k in range(1, N):
        DeltaPhi[:, k] = Phi[:, k] - Phi[:, k-1]
    
    # Convert bits to phase shifts
    PhiData = np.zeros(m)
    for k in range(m):
        PhiData[k] = np.pi/2 if bits[k] == '0' else -np.pi/2
    
    # Apply phase modifications
    Phi_new = np.zeros((L, N), dtype=np.complex128)
    Phi_new[:, 0] = Phi[:, 0]
    
    mid = L // 2
    Phi_new[mid-m:mid, 0] = PhiData
    Phi_new[mid+1:mid+1+m, 0] = -np.flip(PhiData)
    
    # Reconstruct phases
    for k in range(1, N):
        Phi_new[:, k] = Phi_new[:, k-1] + DeltaPhi[:, k]
    
    # Reconstruct signal
    z = np.real(np.fft.ifft(A * np.exp(1j * Phi_new), axis=0))
    snew = z.reshape(N*L, 1, order='F')
    
    return np.vstack([snew, plain[N*L:]]).flatten()

INPUT_DIR = "path to your unwtermarked dataset"
OUTPUT_DIR = "/phase"
os.makedirs(OUTPUT_DIR, exist_ok=True)

for fname in tqdm(os.listdir(INPUT_DIR), desc="Embedding Watermarks"):
    if fname.lower().endswith(".wav"):
        try:
            sr, signal = wavfile.read(os.path.join(INPUT_DIR, fname))
            if signal.dtype == np.int16:
                signal = signal.astype(np.float32) / 32768.0
            
            if signal.ndim > 1:
                signal = signal[:, 0]
            
            # Generate random 16-bit payload
            payload_bits = np.random.choice(['0', '1'], size=16)
            payload_str = ''.join(payload_bits)
            
            # Apply phase coding
            watermarked = phase_enc(signal, payload_str, L=1024)
            
            watermarked = np.clip(watermarked, -1, 1)
            output_path = os.path.join(OUTPUT_DIR, f"phase_{payload_str}_{fname}")
            sf.write(output_path, watermarked, sr, subtype='PCM_16')
            
        except Exception as e:
            print(f"Failed on {fname}: {str(e)}")
