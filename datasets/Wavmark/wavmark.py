import numpy as np
import os
import torch
import soundfile as sf
from tqdm import tqdm
import wavmark
from wavmark.utils import file_reader
from pathlib import Path


# Load wavmark model
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = wavmark.load_model().to(device)

# 4. Define input/output directories
INPUT_DIR = "path to your unwatermarked dataset"
OUTPUT_DIR = "/wavmark_16bit"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# List input audio files
wav_files = [f for f in os.listdir(INPUT_DIR) if f.endswith(".wav")]

# Watermark each file
start_index = 0
for i, fname in enumerate(tqdm(wav_files[start_index:], desc="Embedding with wavmark", initial = start_index, total = len(wav_files))):
    try:

        in_path = os.path.join(INPUT_DIR, fname)

        signal = file_reader.read_as_single_channel(in_path, aim_sr=16000)

        # Generate random 16-bit payload
        payload = np.random.choice([0, 1], size=16)
        payload_str = ''.join(map(str, payload.tolist()))

        # Embed watermark
        watermarked_signal, _ = wavmark.encode_watermark(model, signal, payload, show_progress=False)

        # Save output file with watermark in filename
        out_path = os.path.join(OUTPUT_DIR, f"wm_{payload_str}_{fname}")
        sf.write(out_path, watermarked_signal, 16000)

        # Clean up
        del signal, watermarked_signal, payload
        torch.cuda.empty_cache()

    except Exception as e:
        print(f"Failed on {fname}: {e}")
