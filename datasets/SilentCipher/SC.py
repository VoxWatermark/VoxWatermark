import os
import librosa
import soundfile as sf
import silentcipher
import numpy as np
from tqdm import tqdm
import torchaudio  # For reliable resampling

# Constants
INPUT_DIR = "path to your unwatermarked dataset"
OUTPUT_DIR = "/silentcipher"
TARGET_SR = 16000  # SilentCipher works best at 16kHz
MIN_DURATION = 5.0  # Minimum 5 seconds required

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load model
model = silentcipher.get_model(model_type='44.1k', device='cuda')

wav_files = [f for f in os.listdir(INPUT_DIR) if f.endswith(".wav")]

for filename in tqdm(wav_files, desc="Watermarking"):
    try:
        filepath = os.path.join(INPUT_DIR, filename)
        
        # Load and force mono
        y, sr = librosa.load(filepath, sr=None, mono=True)
        
        # Resample to 16kHz if needed
        if sr != TARGET_SR:
            y = librosa.resample(y, orig_sr=sr, target_sr=TARGET_SR)
            sr = TARGET_SR
        
        required_samples = int(MIN_DURATION * sr)
        if len(y) < required_samples:
            padding = np.zeros(required_samples - len(y))
            y = np.concatenate([y, padding])
        
        
        if len(y) % 2 != 0:
            y = y[:-1]
        
       
        msg_16bit = np.random.randint(0, 256, size=2, dtype=np.uint8)
        msg_padded = np.concatenate([msg_16bit, [0, 0, 0]])
        
        # Apply watermark
        encoded_audio, sdr = model.encode_wav(y, sr, msg_padded.tolist())
        
      
        msg_str = ''.join([f"{x:08b}" for x in msg_16bit])
        save_path = os.path.join(OUTPUT_DIR, f"wm_{msg_str}_{filename}")
        sf.write(save_path, encoded_audio, sr)
        
    except Exception as e:
        print(f"\nFailed {filename}: {str(e)}")
        continue
