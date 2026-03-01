import os
import wave
import struct
import numpy as np
from tqdm import tqdm


def watermark_to_bits(watermark, nbits=8):
    watermark_bits = []
    for byte in watermark:
        for i in range(nbits):
            watermark_bits.append((byte >> i) & 1)
    return watermark_bits

def bits_to_bytes(bit_list):
    assert len(bit_list) % 8 == 0, "Bit list length must be divisible by 8"
    byte_list = []
    for i in range(0, len(bit_list), 8):
        byte = 0
        for j in range(8):
            byte |= (bit_list[i + j] << j)
        byte_list.append(byte)
    return byte_list

def lsb_watermark(cover_filepath, bit_payload, watermarked_output_path):

    byte_payload = bits_to_bytes(bit_payload)
    
    watermark_size = len(byte_payload)
    watermark_bits = watermark_to_bits((watermark_size,), 32)
    watermark_bits.extend(watermark_to_bits(byte_payload))
    print(watermark_bits)

    cover_audio = wave.open(cover_filepath, 'rb')
    (nchannels, sampwidth, framerate, nframes, comptype, compname) = cover_audio.getparams()
    frames = cover_audio.readframes(nframes * nchannels)
    samples = list(struct.unpack_from("%dh" % (nframes * nchannels), frames))

    if len(samples) < len(watermark_bits):
        raise OverflowError(f"Message too large! Needs {len(watermark_bits)} samples, but only {len(samples)} available.")

    encoded_samples = []
    watermark_position = 0
    for sample in samples:
        encoded_sample = sample
        if watermark_position < len(watermark_bits):
            encode_bit = watermark_bits[watermark_position]
            if encode_bit == 1:
                encoded_sample = sample | 1
            else:
                encoded_sample = sample & ~1
            watermark_position += 1
        encoded_samples.append(encoded_sample)

    encoded_audio = wave.open(watermarked_output_path, 'wb')
    encoded_audio.setparams((nchannels, sampwidth, framerate, nframes, comptype, compname))
    encoded_audio.writeframes(struct.pack("%dh" % len(encoded_samples), *encoded_samples))
    encoded_audio.close()


INPUT_DIR = "path to your unwatermarked dataset"
OUTPUT_DIR = "/content/lsb_watermarked"
os.makedirs(OUTPUT_DIR, exist_ok=True)

for fname in tqdm(os.listdir(INPUT_DIR), desc="Embedding 16-bit LSB"):
    if not fname.lower().endswith(".wav"):
        continue

    try:
        # Generate 16-bit random binary message as list of bits
        bits = np.random.randint(0, 2, 16).tolist()  
        bit_str = ''.join(map(str, bits))
        input_path = os.path.join(INPUT_DIR, fname)
        output_path = os.path.join(OUTPUT_DIR, f"lsb_{bit_str}_{fname}")
        lsb_watermark(input_path, bits, output_path)

    except Exception as e:
        print(f"Failed {fname}: {e}")
