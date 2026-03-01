import os
import torch
import torchaudio
from audioseal import AudioSeal
from pathlib import Path
from tqdm import tqdm

# Constants
INPUT_DIR = "/path to your unwatermarked dataset"
OUTPUT_DIR = "/audioseal_16bit"
SAMPLE_RATE = 16000
start_index = 0  # <--- Start from this index

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load model
model = AudioSeal.load_generator("audioseal_wm_16bits").to(device)
n_bits = model.msg_processor.nbits

# List all files
wav_files = [f for f in os.listdir(INPUT_DIR) if f.endswith(".wav")]

# Iterate starting from start_index
for i, filename in enumerate(tqdm(wav_files[start_index:], desc="Watermarking audio", initial=start_index, total=len(wav_files))):
    try:
        filepath = os.path.join(INPUT_DIR, filename)
        waveform, sr = torchaudio.load(filepath)

        # Resample to 16kHz if needed
        if sr != SAMPLE_RATE:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=SAMPLE_RATE)
            waveform = resampler(waveform)

        # Convert to mono
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        wav_batch = waveform.unsqueeze(0).to(device)

        # Generate random 16-bit message
        msg = torch.randint(0, 2, (1, n_bits), device=device)

        with torch.no_grad():
            watermark = model.get_watermark(wav_batch, message=msg)
            watermarked = wav_batch + watermark

        msg_str = ''.join(map(str, msg.squeeze().cpu().numpy().astype(int).tolist()))
        save_path = os.path.join(OUTPUT_DIR, f"wm_{msg_str}_{filename}")
        torchaudio.save(save_path, watermarked.squeeze(0).cpu(), SAMPLE_RATE)

        # Free memory
        del waveform, wav_batch, watermarked, watermark, msg
        torch.cuda.empty_cache()

    except Exception as e:
        print(f"Failed to process {filename}: {e}")
