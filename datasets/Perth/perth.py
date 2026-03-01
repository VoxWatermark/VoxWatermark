import os
import soundfile as sf
import perth
from tqdm import tqdm

INPUT_DIR = "path to your unwatermarked dataset"
OUTPUT_DIR = "/perth"
os.makedirs(OUTPUT_DIR, exist_ok=True)

watermarker = perth.PerthImplicitWatermarker()
for fname in tqdm(os.listdir(INPUT_DIR)):
    if not fname.endswith(".wav"):
        continue
    in_path = os.path.join(INPUT_DIR, fname)
    wav, sr = sf.read(in_path)
    watermarked_audio = watermarker.apply_watermark(wav, watermark=None, sample_rate=sr)
    out_path = os.path.join(OUTPUT_DIR, f"perth_{fname}")
    sf.write(out_path, watermarked_audio, sr)
