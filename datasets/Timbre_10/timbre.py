import os
import sys
import argparse
import numpy as np
import torch
import torchaudio
from tqdm import tqdm
from torch.nn import functional as F

import librosa
import julius
import typing as tp

import tempfile
import uuid
import random
from numpy.typing import NDArray
from audiomentations.core.transforms_interface import BaseWaveformTransform
from audiomentations.core.utils import (
    convert_float_samples_to_int16, get_max_abs_amplitude,
)


np.random.seed(42)
torch.manual_seed(42)


device = torch.device('cpu')

import yaml
from model.conv2_mel_modules import Encoder, Decoder
process_config = yaml.load(open("config/process.yaml", "r"), Loader=yaml.FullLoader)

model_config = yaml.load(open("config/model.yaml", "r"), Loader=yaml.FullLoader)
train_config = yaml.load(open("config/train.yaml", "r"), Loader=yaml.FullLoader)
win_dim = process_config["audio"]["win_len"]
embedding_dim = model_config["dim"]["embedding"]
nlayers_encoder = model_config["layer"]["nlayers_encoder"]
nlayers_decoder = model_config["layer"]["nlayers_decoder"]
attention_heads_encoder = model_config["layer"]["attention_heads_encoder"]
attention_heads_decoder = model_config["layer"]["attention_heads_decoder"]
generator = Encoder(process_config, model_config, 10, win_dim, embedding_dim, nlayers_encoder=nlayers_encoder, attention_heads=attention_heads_encoder).to(device)

checkpoint = torch.load('results/ckpt/pth/compressed_none-conv2_ep_20_2023-01-17_23_01_01.pth.tar')
generator.load_state_dict(checkpoint['encoder'])

generator.eval()

data_dir = "/content/original3"

output_dir = f'/content/timbre10'





file_list = [file for file in os.listdir(data_dir) if file.endswith('.wav')]
for file in tqdm(file_list, desc="Encoding Watermarks"):
    file_path = os.path.join(data_dir, file)
    waveform, sample_rate = torchaudio.load(file_path)
    if waveform.size(0) > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    '''resample to 16k'''
    if sample_rate != 16000:
        waveform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)(waveform)
        sample_rate = 16000

    # Truncate or pad the waveform to max_length


    signal = waveform.unsqueeze(0).to(device=next(generator.parameters()).device)

    msg = torch.randint(0, 2, (signal.shape[0], 10), device=signal.device,dtype=torch.float32)
    '''rescale the message to -1 and 1 according to the watermark function'''
    msg = msg.unsqueeze(0)
    msg_rescaled = msg * 2 - 1
    '''watermark function call'''
    watermarked_signal = generator.test_forward(signal, msg_rescaled)[0]

    # Compute SNR
    signal_power = torch.mean(signal ** 2)
    noise_power = torch.mean((watermarked_signal - signal) ** 2)
    snr = 10 * torch.log10(signal_power / noise_power)

    msg_str =  ''.join([''.join(map(str, map(int, msg.squeeze().tolist())))])
    file_name = f"{file[:-4]}_{msg_str}.wav"
    watermarked_signal = watermarked_signal.cpu().squeeze(0)
    
    os.makedirs(output_dir, exist_ok=True)
    torchaudio.save(os.path.join(output_dir, file_name), watermarked_signal, sample_rate)




