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

import io
import re
import subprocess
import tempfile
import torch
import torchaudio
from typing import Literal, Optional, Tuple

from pydub import AudioSegment 
from pydub.effects import compress_dynamic_range
from pydub.utils import audioop, db_to_float, ratio_to_db 
from torchinterp1d.interp1d import interp1d as tinterp

import tempfile
import uuid
import random
from numpy.typing import NDArray
from audiomentations.core.transforms_interface import BaseWaveformTransform
from audiomentations.core.utils import (
    convert_float_samples_to_int16, get_max_abs_amplitude,
)
import warnings
warnings.filterwarnings(
    action='ignore',
    message='.*MessageFactory class is deprecated.*'
)

def extract_id(filename):
    file_name = filename.split('/')[-1]
    parts = file_name.split('_')
    id_part = '_'.join(parts[:4])
    return id_part

def pert_time_stretch(waveform, sample_rate, speed_factor):
    waveform_np = waveform.numpy()
    if waveform_np.shape[0] == 1:
        waveform_np = waveform_np.squeeze()

    waveform_stretched = librosa.effects.time_stretch(waveform_np, rate=speed_factor)
    time_stretched_waveform = torch.from_numpy(waveform_stretched).unsqueeze(0).float()
    if time_stretched_waveform.shape[1] < waveform.shape[1]:
        time_stretched_waveform = F.pad(time_stretched_waveform, (0, waveform.shape[1] - time_stretched_waveform.shape[1]))
    elif time_stretched_waveform.shape[1] > waveform.shape[1]:
        time_stretched_waveform = time_stretched_waveform[:, :waveform.shape[1]]

    return time_stretched_waveform, sample_rate


def compute_snr(signal, noisy_signal):
    
    signal_power = torch.mean(signal ** 2)

    noise = noisy_signal - signal
    noise_power = torch.mean(noise ** 2)
    # print(f'mean_noise_power: {noise_power}')

    snr = 10 * torch.log10(signal_power / noise_power)

    return snr.item()



def inverse_polarity(audio: torch.Tensor):
    """Invert the polarity of an audio signal.

    Parameters
    ----------
    audio: torch.Tensor
        The input audio signal.

    Returns
    -------
    inverted_audio: torch.Tensor
        The audio signal with inverted polarity.
    """
    inverted_audio = -audio

    return inverted_audio


def time_jitter(audio: torch.Tensor,
                scale: float = 0.1) -> torch.Tensor:
    """Apply time jitter to audio signal
    Sampling jitter: "https://www.peak-studios.de/en/audio-jitter/#:~:text=Audio%20jitter%20is%20a%20variance,problems%20with%20the%20audio%20hardware."

    Parameters
    ----------
    audio: torch.Tensor
        Audio signal (Expected shape: channels, length)

    scale: float (default=0.1)
        Scale of jitter

    # sr: int (default=44100) # Not used in this implementation
    #     Sample rate

    Returns
    -------
    jittered_audio: torch.Tensor
        Jittered audio signal
    """
    
    audio_flat = audio 

    x = torch.arange(audio_flat.shape[1], dtype=torch.float, device=audio_flat.device).unsqueeze(0).repeat(audio_flat.shape[0], 1) # (channels, length)

    jitter_offsets = torch.normal(mean=0.0, std=scale, size=x.shape, device=audio_flat.device) # (channels, length)

    x_new = x + jitter_offsets 

    
    jittered_audio_flat = tinterp(x, audio_flat, x_new) 

    return jittered_audio_flat


def phase_shift(audio: torch.Tensor,
                shift: int) -> torch.Tensor:
    """Shift the phase of an audio tensor by `phase_shift` samples.

    Parameters
    ----------
    audio: torch.Tensor
        The input audio tensor. (Expected shape: channels, length)
    shift: int
        The number of samples to shift the phase by.

    Returns
    -------
    shifted_audio: torch.Tensor
        The audio tensor with the phase shifted.
    """
    
    channels, length = audio.shape

    if shift == 0:
        return audio

    
    padding = torch.zeros((channels, abs(shift)), dtype=audio.dtype, device=audio.device)

    if shift > 0:
        
        shifted_part = audio[:, shift:] if shift < length else torch.empty((channels, 0), dtype=audio.dtype, device=audio.device)
        shifted_audio = torch.cat([padding, shifted_part], dim=1)
    else: 
        shifted_part = audio[:, :shift] if shift != 0 else audio 
        shifted_audio = torch.cat([shifted_part, padding], dim=1)

    
    if shifted_audio.shape[1] < length:
       
        shifted_audio = F.pad(shifted_audio, (0, length - shifted_audio.shape[1]))
    elif shifted_audio.shape[1] > length:
        
        shifted_audio = shifted_audio[:, :length]

    return shifted_audio




def dynamic_range_compression(audio: torch.Tensor,
                              threshold: float = -20,
                              ratio: float = 4.0,
                              sr: int = 16000,
                              **kwargs) -> torch.Tensor:
    
    init_device = audio.device


    max_val = torch.max(torch.abs(audio))
    audio_norm = audio / max_val
    audio_seg = convert_torch_to_pydub(audio=audio_norm, sr=sr)

    compressed_audio_seg = compress_dynamic_range(audio_seg,
                                                  ratio=ratio,
                                                  threshold=threshold,
                                                  **kwargs)

    compressed_audio = convert_pydub_to_torch(compressed_audio_seg).to(init_device)

    
    compressed_audio *= max_val

    return compressed_audio


def dynamic_range_expansion(audio: torch.Tensor,
                            threshold: float = -20,
                            ratio: float = 4.0,
                            sr: int = 16000,
                            **kwargs) -> torch.Tensor:
    init_device = audio.device

   
    max_val = torch.max(torch.abs(audio))
    audio_norm = audio / max_val

    audio_seg = convert_torch_to_pydub(audio=audio_norm,
                                       sr=sr)

    expanded_audio_seg = expand_dynamic_range(audio_seg,
                                              ratio=ratio,
                                              threshold=threshold,
                                              **kwargs)
    expanded_audio = convert_pydub_to_torch(expanded_audio_seg).to(init_device)

    
    expanded_audio *= max_val

    return expanded_audio


AVERAGE_ENERGY_VCTK = 0.002837200844477648

def convert_pydub_to_torch(
    x: AudioSegment
) -> torch.Tensor:
    
    samples = np.array(x.get_array_of_samples(), dtype=np.int16)
    if samples.dtype != np.float32:
        samples = samples.astype(np.float32) / np.iinfo(samples.dtype).max
    
    x = torch.from_numpy(samples).unsqueeze(0) 

    return x


def convert_torch_to_pydub(
    audio: torch.Tensor,
    sr: int = 16000
) -> AudioSegment:
    
    waveform_np = audio.detach().cpu().numpy()

    
    if waveform_np.shape[0] > 1:
       
        waveform_np = np.transpose(waveform_np).flatten() 
    else: 
        waveform_np = waveform_np[0]

    try:
       
        waveform_int16 = (waveform_np * 32767).astype(np.int16)
    except:
       
        waveform_energy = np.mean(waveform_np**2)
        if waveform_energy == 0:
             waveform_energy = AVERAGE_ENERGY_VCTK 
        scaling_factor = np.sqrt(AVERAGE_ENERGY_VCTK / waveform_energy)
        waveform_scaled = waveform_np * scaling_factor
        waveform_int16 = (waveform_scaled * 32767).astype(np.int16)


    
    num_channels = audio.shape[0]
    audio_segment = AudioSegment(
        data=waveform_int16.tobytes(),
        sample_width=2,  
        frame_rate=sr,
        channels=num_channels
    )

    return audio_segment


def expand_dynamic_range(seg,
                         threshold=-20.0,
                         ratio=4.0,
                         attack=5.0,
                         release=50.0):
    """
    Keyword Arguments:

        threshold - default: -20.0
            Threshold in dBFS. default of -20.0 means -20dB relative to the
            maximum possible volume. 0dBFS is the maximum possible value so
            all values for this argument sould be negative.

        ratio - default: 4.0
            Expansion ratio. Audio quieter than the threshold will be
            expanded to ratio the volume. A ratio of 4.0 means for every 4 dB
            the input signal is below the threshold, the output signal will be
            1 dB below the threshold.

        attack - default: 5.0
            Attack in milliseconds. How long it should take for the expansion
            to kick in once the audio has fallen below the threshold.

        release - default: 50.0
            Release in milliseconds. How long it should take for the expansion
            to stop after the audio has risen above the threshold.


    For an overview of Dynamic Range Expansion, and more detailed explanation
    of the related terminology, see:

        http://en.wikipedia.org/wiki/Dynamic_range_compression
    """

    thresh_rms = seg.max_possible_amplitude * db_to_float(threshold)

    look_frames = int(seg.frame_count(ms=attack))
    def rms_at(frame_i):
        # Calculate RMS over a window around the current frame
        start_frame = max(0, frame_i - look_frames)
        # Use the slice method provided by pydub
        slice_obj = seg.get_sample_slice(start_frame, frame_i)
        if slice_obj.frame_count() > 0:
            return slice_obj.rms
        else:
            return 0 # Return 0 if slice is empty (e.g., at start)
    def db_under_threshold(rms):
        # Calculate how much below the threshold the RMS is
        if thresh_rms == 0 or rms >= thresh_rms:
            return 0.0
        # --- Check for division by zero ---
        if rms == 0:
            # If the current RMS is 0 (silent), treat it as infinitely below the threshold.
            # Returning a large value might be appropriate, or just 0.0 if we don't want
            # to amplify absolute silence. Returning 0.0 means no expansion is applied
            # to this silent segment, which is often the safest.
            return 0.0
        # --- End of check ---
        db = ratio_to_db(thresh_rms / rms) # This line was causing the error
        return max(db, 0)

    output = []

    # Amount to reduce the volume of the audio by (in dB for expansion)
    attenuation = 0.0

    attack_frames = seg.frame_count(ms=attack)
    release_frames = seg.frame_count(ms=release)
    for i in range(int(seg.frame_count())):
        rms_now = rms_at(i)

        # Calculate target attenuation based on how far below threshold the signal is
        # With ratio 4.0, if signal is 12dB below threshold, target_attenuation is 3dB
        max_attenuation  = (ratio - 1) * db_under_threshold(rms_now) / ratio # Simplified calculation

        attenuation_inc = max_attenuation / attack_frames if attack_frames > 0 else max_attenuation
        attenuation_dec = attenuation / release_frames if release_frames > 0 else attenuation

        if rms_now < thresh_rms: # If signal is below threshold
            # Increase attenuation towards the target
            attenuation += attenuation_inc
            attenuation = min(attenuation, max_attenuation)
        else: # If signal is at or above threshold
            # Decrease attenuation back towards 0
            attenuation -= attenuation_dec
            attenuation = max(attenuation, 0)

        frame = seg.get_frame(i)
        if attenuation > 0.001: # Apply attenuation if significant
            frame = audioop.mul(frame,
                                seg.sample_width,
                                db_to_float(-attenuation))

        output.append(frame)

    return seg._spawn(data=b''.join(output))



def get_aac(
    wav_tensor: torch.Tensor,
    sr: int,
    bitrate: str = "128k",
    lowpass_freq: Optional[int] = None,
    ffmpeg4codecs: Optional[str] = None,
) -> torch.Tensor:
    """Converts a batch of audio tensors to AAC format and then back to tensors.

    This function first saves the input tensor batch as WAV files, then uses FFmpeg to convert
    these WAV files to AAC format. Finally, it loads the AAC files back into tensors.

    Args:
        wav_tensor (torch.Tensor): A batch of audio files represented as a tensor.
                                   Shape should be (batch_size, channels, length).
        sr (int): Sampling rate of the audio.
        bitrate (str): Bitrate for AAC conversion, default is '128k'.
        lowpass_freq (Optional[int]): Frequency for a low-pass filter. If None, no filter is applied.
        ffmpeg4codecs: (Optional[str]) = If none, use a defulat ffmpeg. Otherwise, use a specific ffmpeg.

    Returns:
        torch.Tensor: Batch of audio files converted to AAC and back, with the same
                      shape as the input tensor.
    """
    device = wav_tensor.device
    batch_size, channels, original_length = wav_tensor.shape

    # Parse the bitrate value from the string
    match = re.search(r"\d+(\.\d+)?", bitrate)
    if match:
        parsed_bitrate = (
            match.group()
        )  # Default to 128 if parsing fails
    else:
        raise ValueError(f"Invalid bitrate specified (got {bitrate})")

    # Flatten tensor for conversion and move to CPU
    wav_tensor_flat = wav_tensor.view(1, -1).cpu() # one vary large audio file...

    with tempfile.NamedTemporaryFile(
        suffix=".wav"
    ) as f_in, tempfile.NamedTemporaryFile(suffix=".aac") as f_out:
        input_path, output_path = f_in.name, f_out.name

        # Save the tensor as a WAV file
        torchaudio.save(input_path, wav_tensor_flat, sr)

        # Prepare FFmpeg command for AAC conversion
        ffmpeg = "ffmpeg" if ffmpeg4codecs is None else ffmpeg4codecs
        command = [
            ffmpeg,
            "-y",
            "-i",
            input_path,
            "-ar",
            str(sr),
            "-b:a",
            f"{parsed_bitrate}k",
            "-c:a",
            "aac",
        ]
        if lowpass_freq is not None:
            command += ["-cutoff", str(lowpass_freq)]
        command.append(output_path)

        # Run FFmpeg and suppress output
        subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        # Load the AAC audio back into a tensor
        aac_tensor, _ = torchaudio.load(output_path)

    original_length_flat = batch_size * channels * original_length
    compressed_length_flat = aac_tensor.shape[-1]

    # # Trim excess frames
    if compressed_length_flat > original_length_flat:
        min_distance = float('inf')
        min_index = -1
        length = wav_tensor.shape[-1]

        # Iterate over possible indices
        for index in range(aac_tensor.shape[-1] - length + 1):  # Sliding window
            # Extract the window from aac_tensor
            aac_window = aac_tensor[..., index:index+length].squeeze()

            # Compute L1 distance
            l1_distance = torch.sum(torch.abs(aac_window - wav_tensor.cpu().squeeze()))

            # Update minimum distance and index
            if l1_distance < min_distance:
                min_distance = l1_distance
                min_index = index

        aac_tensor = aac_tensor[:, min_index:min_index+length]

    # Pad the shortened frames
    elif compressed_length_flat < original_length_flat:
        padding = torch.zeros(
            1, original_length_flat - compressed_length_flat, device=device
        )
        aac_tensor = torch.cat((aac_tensor, padding), dim=-1)

    # Reshape and adjust length to match original tensor
    wav_tensor = aac_tensor.view(batch_size, channels, -1)
    compressed_length = wav_tensor.shape[-1]

    assert compressed_length == original_length, (
        "AAC-compressed audio does not have the same frames as original one. "
        "One reason can be ffmpeg is not  installed and used as proper backed "
        "for torchaudio, or the AAC encoder is not correct. Run "
        "`torchaudio.utils.ffmpeg_utils.get_audio_encoders()` and make sure we see entry for"
        "AAC in the output."
    )
    return wav_tensor.to(device)



def aac_wrapper(wav_tensor: torch.Tensor,
                sr: int,
                bitrate = '64k', 
                ffmpeg4codecs: str = None):
    return get_aac(wav_tensor, sr, bitrate=bitrate, ffmpeg4codecs=ffmpeg4codecs)




def pert_Gaussian_noise(waveform, snr_db):
    
    signal_power = torch.mean(waveform**2).to(device=waveform.device)

    snr_linear = 10 ** (snr_db / 10)
    noise_power = signal_power / snr_linear

    noise = torch.randn(waveform.size()) * torch.sqrt(noise_power)
    waveform_noisy = waveform + noise
    snr = compute_snr(waveform, waveform_noisy)
   
    return waveform_noisy

import torchaudio
import librosa
import soundfile as sf
import os

def convert_mp3_to_wav(mp3_file_path, output_wav_file_path):
    audio, sr = librosa.load(mp3_file_path, sr=16000)
    audio_mono = librosa.to_mono(audio)
    sf.write(output_wav_file_path, audio_mono, 16000)
    print(f"MP3 converted to WAV and saved as: {output_wav_file_path}")
    return output_wav_file_path

def pert_background_noise(waveform, snr_db, noise_file_path):

    if noise_file_path.endswith('.mp3'):
        # Convert MP3 to WAV and resample to 16kHz
        wav_noise_file_path = '/path to your noise file.wav'
        noise_file_path = convert_mp3_to_wav(noise_file_path, wav_noise_file_path)

    noise, _ = torchaudio.load(noise_file_path)

    if noise.size(0) > 1:
        noise = noise.mean(dim=0, keepdim=True)  # Convert to mono

    if noise.size(1) > waveform.size(1):
        noise = noise[:, :waveform.size(1)]
    else:
        repeat_times = waveform.size(1) // noise.size(1) + 1
        noise = noise.repeat(1, repeat_times)
        noise = noise[:, :waveform.size(1)]

    signal_power = torch.mean(waveform**2)
    noise_power = torch.mean(noise**2)

    snr_linear = 10 ** (snr_db / 10)
    scaling_factor = torch.sqrt(signal_power / (snr_linear * noise_power))

    noisy_waveform = waveform + noise * scaling_factor
    return noisy_waveform


import opuspy
def pert_opus(waveform: torch.tensor, bitrate: int, quality: int, cache: str) -> None:
    waveform_scaled = waveform * 32768
    waveform_scaled = waveform_scaled.reshape(-1,1).numpy()
    cache_file = os.path.join(cache, "temp.opus")
    opuspy.write(cache_file, waveform_scaled, sample_rate = 16000,
                bitrate = bitrate, signal_type = 0, encoder_complexity = quality)
    pert_waveform, sampling_rate = opuspy.read(cache_file)
    os.remove(cache_file)
    resampler = torchaudio.transforms.Resample(orig_freq=48000, new_freq=16000)
    pert_waveform = torch.tensor(pert_waveform, dtype=torch.float32).reshape(1,-1)
    pert_waveform /= 32768
    return resampler(pert_waveform)


def pert_encodec(waveform, sample_rate, bandwidth, model_encodec, processor):
   
    model_encodec = model_encodec.to('cpu')
    waveform = waveform.squeeze().numpy()  # Assuming waveform is a PyTorch tensor
    inputs = processor(raw_audio=waveform, sampling_rate=sample_rate, return_tensors="pt").to('cpu')

    encoder_outputs = model_encodec.encode(inputs["input_values"], inputs["padding_mask"], bandwidth)
    audio_values = model_encodec.decode(encoder_outputs.audio_codes, encoder_outputs.audio_scales, inputs["padding_mask"])[0]
    # The output 'audio_values' is the perturbed waveform
    return torch.tensor(audio_values.detach().cpu()).squeeze().unsqueeze(0)


def pert_quantization(waveform, quantization_bit):
    # Normalize the waveform to the range of the quantization levels
    min_val, max_val = waveform.min(), waveform.max()
    normalized_waveform = (waveform - min_val) / (max_val - min_val)

    # Quantize the normalized waveform
    quantized_waveform = torch.round(normalized_waveform * (quantization_bit - 1))

    # Rescale the quantized waveform back to the original range
    rescaled_waveform = (quantized_waveform / (quantization_bit - 1)) * (max_val - min_val) + min_val

    return rescaled_waveform


def audio_effect_return(
    tensor: torch.Tensor, mask: tp.Optional[torch.Tensor]
) -> tp.Union[tp.Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
    """Return the mask if it was in the input otherwise only the output tensor"""
    if mask is None:
        return tensor
    else:
        return tensor, mask


def pert_highpass(
    waveform: torch.Tensor,
    cutoff_ratio: float,
    sample_rate: int = 16000,
    mask: tp.Optional[torch.Tensor] = None,
) -> tp.Union[tp.Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:

    return audio_effect_return(
        tensor=julius.highpass_filter(waveform, cutoff=cutoff_ratio),
        mask=mask,
    )

def pert_lowpass(
    waveform: torch.Tensor,
    cutoff_ratio: float,
    sample_rate: int = 16000,
    mask: tp.Optional[torch.Tensor] = None,
) -> tp.Union[tp.Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:

    return audio_effect_return(
        tensor=julius.lowpass_filter(waveform, cutoff=cutoff_ratio),
        mask=mask,
    )


def pert_smooth(
    waveform: torch.Tensor,
    window_size: int = 5,
    mask: tp.Optional[torch.Tensor] = None,
) -> tp.Union[tp.Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:

    waveform = waveform.unsqueeze(0)
    window_size = int(window_size)
    # Create a uniform smoothing kernel
    kernel = torch.ones(1, 1, window_size).type(waveform.type()) / window_size
    kernel = kernel.to(waveform.device)

    smoothed = julius.fft_conv1d(waveform, kernel)
    # Ensure tensor size is not changed
    tmp = torch.zeros_like(waveform)
    tmp[..., : smoothed.shape[-1]] = smoothed
    smoothed = tmp

    return audio_effect_return(tensor=smoothed, mask=mask).squeeze().unsqueeze(0)


def pert_echo(
    tensor: torch.Tensor,
    # volume_range: tuple = (0.1, 0.5),
    # duration_range: tuple = (0.1, 0.5),
    volume: float = 0.4,
    duration: float = 0.1,
    sample_rate: int = 16000,
    mask: tp.Optional[torch.Tensor] = None,
) -> tp.Union[tp.Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
    """
    Attenuating the audio volume by a factor of 0.4, delaying it by 100ms,
    and then overlaying it with the original.

    :param tensor: 3D Tensor representing the audio signal [bsz, channels, frames]
    :param echo_volume: volume of the echo signal
    :param sample_rate: Sample rate of the audio signal.
    :return: Audio signal with reverb.
    """
    tensor = tensor.unsqueeze(0)

    duration = torch.Tensor([duration])
    volume = torch.Tensor([volume])
    n_samples = int(sample_rate * duration)
    impulse_response = torch.zeros(n_samples).type(tensor.type()).to(tensor.device)

    impulse_response[0] = 1.0  # Direct sound

    impulse_response[
        int(sample_rate * duration) - 1
    ] = volume  # First reflection after 100ms

    # Add batch and channel dimensions to the impulse response
    impulse_response = impulse_response.unsqueeze(0).unsqueeze(0)

    # Convolve the audio signal with the impulse response
    reverbed_signal = julius.fft_conv1d(tensor, impulse_response)

    # Normalize to the original amplitude range for stability
    reverbed_signal = (
        reverbed_signal
        / torch.max(torch.abs(reverbed_signal))
        * torch.max(torch.abs(tensor))
    )

    # Ensure tensor size is not changed
    tmp = torch.zeros_like(tensor)
    tmp[..., : reverbed_signal.shape[-1]] = reverbed_signal
    reverbed_signal = tmp
    reverbed_signal = reverbed_signal.squeeze(0)
    return audio_effect_return(tensor=reverbed_signal, mask=mask)



class Mp3Compression(BaseWaveformTransform):
    """Compress the audio using an MP3 encoder to lower the audio quality.
    This may help machine learning models deal with compressed, low-quality audio.

    This transform depends on either lameenc or pydub/ffmpeg.

    Note that bitrates below 32 kbps are only supported for low sample rates (up to 24000 Hz).

    Note: When using the lameenc backend, the output may be slightly longer than the input due
    to the fact that the LAME encoder inserts some silence at the beginning of the audio.

    Warning: This transform writes to disk, so it may be slow. Ideally, the work should be done
    in memory. Contributions are welcome.
    """

    supports_multichannel = True

    SUPPORTED_BITRATES = [
        8,
        16,
        24,
        32,
        40,
        48,
        56,
        64,
        80,
        96,
        112,
        128,
        144,
        160,
        192,
        224,
        256,
        320,
    ]

    def __init__(
        self,
        min_bitrate: int = 8,
        max_bitrate: int = 64,
        backend: str = "pydub",
        p: float = 0.5,
    ):
        """
        :param min_bitrate: Minimum bitrate in kbps
        :param max_bitrate: Maximum bitrate in kbps
        :param backend: "pydub" or "lameenc".
            Pydub may use ffmpeg under the hood.
                Pros: Seems to avoid introducing latency in the output.
                Cons: Slower than lameenc.
            lameenc:
                Pros: You can set the quality parameter in addition to bitrate.
                Cons: Seems to introduce some silence at the start of the audio.
        :param p: The probability of applying this transform
        """
        super().__init__(p)
        if min_bitrate < self.SUPPORTED_BITRATES[0]:
            raise ValueError(
                "min_bitrate must be greater than or equal to"
                f" {self.SUPPORTED_BITRATES[0]}"
            )
        if max_bitrate > self.SUPPORTED_BITRATES[-1]:
            raise ValueError(
                "max_bitrate must be less than or equal to"
                f" {self.SUPPORTED_BITRATES[-1]}"
            )
        if max_bitrate < min_bitrate:
            raise ValueError("max_bitrate must be >= min_bitrate")

        is_any_supported_bitrate_in_range = any(
            min_bitrate <= bitrate <= max_bitrate for bitrate in self.SUPPORTED_BITRATES
        )
        if not is_any_supported_bitrate_in_range:
            raise ValueError(
                "There is no supported bitrate in the range between the specified"
                " min_bitrate and max_bitrate. The supported bitrates are:"
                f" {self.SUPPORTED_BITRATES}"
            )

        self.min_bitrate = min_bitrate
        self.max_bitrate = max_bitrate
        if backend not in ("pydub", "lameenc"):
            raise ValueError('backend must be set to either "pydub" or "lameenc"')
        self.backend = backend
        self.post_gain_factor = None

    def randomize_parameters(self, samples: NDArray[np.float32], sample_rate: int):
        super().randomize_parameters(samples, sample_rate)
        if self.parameters["should_apply"]:
            bitrate_choices = [
                bitrate
                for bitrate in self.SUPPORTED_BITRATES
                if self.min_bitrate <= bitrate <= self.max_bitrate
            ]
            self.parameters["bitrate"] = random.choice(bitrate_choices)

    def apply(self, samples: NDArray[np.float32], sample_rate: int):
        if self.backend == "lameenc":
            return self.apply_lameenc(samples, sample_rate)
        elif self.backend == "pydub":
            return self.apply_pydub(samples, sample_rate)
        else:
            raise Exception("Backend {} not recognized".format(self.backend))

    def maybe_pre_gain(self, samples):
        """
        If the audio is too loud, gain it down to avoid distortion in the audio file to
        be encoded.
        """
        greatest_abs_sample = get_max_abs_amplitude(samples)
        if greatest_abs_sample > 1.0:
            self.post_gain_factor = greatest_abs_sample
            samples = samples * (1.0 / greatest_abs_sample)
        else:
            self.post_gain_factor = None
        return samples

    def maybe_post_gain(self, samples):
        """If the audio was pre-gained down earlier, post-gain it up to compensate here."""
        if self.post_gain_factor is not None:
            samples = samples * self.post_gain_factor
        return samples

    def apply_lameenc(self, samples: NDArray[np.float32], sample_rate: int):
        try:
            import lameenc
        except ImportError:
            print(
                (
                    "Failed to import the lame encoder. Maybe it is not installed? "
                    "To install the optional lameenc dependency of audiomentations,"
                    " do `pip install audiomentations[extras]` or simply"
                    " `pip install lameenc`"
                ),
                file=sys.stderr,
            )
            raise

        assert samples.dtype == np.float32

        samples = self.maybe_pre_gain(samples)

        int_samples = convert_float_samples_to_int16(samples).T

        num_channels = 1 if samples.ndim == 1 else samples.shape[0]

        encoder = lameenc.Encoder()
        encoder.set_bit_rate(self.parameters["bitrate"])
        encoder.set_in_sample_rate(sample_rate)
        encoder.set_channels(num_channels)
        encoder.set_quality(7)  # 2 = highest, 7 = fastest
        encoder.silence()

        mp3_data = encoder.encode(int_samples.tobytes())
        mp3_data += encoder.flush()

        # Write a temporary MP3 file that will then be decoded
        tmp_dir = tempfile.gettempdir()
        tmp_file_path = os.path.join(
            tmp_dir, "tmp_compressed_{}.mp3".format(str(uuid.uuid4())[0:12])
        )
        with open(tmp_file_path, "wb") as f:
            f.write(mp3_data)

        degraded_samples, _ = librosa.load(tmp_file_path, sr=sample_rate, mono=False)

        os.unlink(tmp_file_path)

        degraded_samples = self.maybe_post_gain(degraded_samples)

        if num_channels == 1:
            if int_samples.ndim == 1 and degraded_samples.ndim == 2:
                degraded_samples = degraded_samples.flatten()
            elif int_samples.ndim == 2 and degraded_samples.ndim == 1:
                degraded_samples = degraded_samples.reshape((1, -1))

        return degraded_samples

    def apply_pydub(self, samples: NDArray[np.float32], sample_rate: int):
        try:
            import pydub
        except ImportError:
            print(
                (
                    "Failed to import pydub. Maybe it is not installed? "
                    "To install the optional pydub dependency of audiomentations,"
                    " do `pip install audiomentations[extras]` or simply"
                    " `pip install pydub`"
                ),
                file=sys.stderr,
            )
            raise

        assert samples.dtype == np.float32

        samples = self.maybe_pre_gain(samples)

        int_samples = convert_float_samples_to_int16(samples).T
        num_channels = 1 if samples.ndim == 1 else samples.shape[0]
        audio_segment = pydub.AudioSegment(
            int_samples.tobytes(),
            frame_rate=sample_rate,
            sample_width=int_samples.dtype.itemsize,
            channels=num_channels,
        )

        tmp_dir = tempfile.gettempdir()
        tmp_file_path = os.path.join(
            tmp_dir, "tmp_compressed_{}.mp3".format(str(uuid.uuid4())[0:12])
        )

        bitrate_string = "{}k".format(self.parameters["bitrate"])
        file_handle = audio_segment.export(tmp_file_path, bitrate=bitrate_string)
        file_handle.close()

        degraded_samples, _ = librosa.load(tmp_file_path, sr=sample_rate, mono=False)

        os.unlink(tmp_file_path)

        degraded_samples = self.maybe_post_gain(degraded_samples)

        if num_channels == 1:
            if int_samples.ndim == 1 and degraded_samples.ndim == 2:
                degraded_samples = degraded_samples.flatten()
            elif int_samples.ndim == 2 and degraded_samples.ndim == 1:
                degraded_samples = degraded_samples.reshape((1, -1))

        return degraded_samples

def pert_mp3(waveform, bitrate, sample_rate=16000):
    mp3_compressor = Mp3Compression(
    min_bitrate=bitrate,  # Set the minimum bitrate
    max_bitrate=bitrate,  # Set the maximum bitrate
    backend="pydub",  # Choose the backend
    p=1.0)  # Set the probability to 1 to always apply the effect
    waveform = waveform.detach().cpu().numpy()
    mp3_compressor.randomize_parameters(waveform, sample_rate)
    waveform_pert = mp3_compressor.apply(waveform, sample_rate)
    return torch.tensor(waveform_pert)


def apply_no_box_pert(input_wav, output_dir, common_perturbation):
    output_path=""
    filename = os.path.basename(input_wav)
    if common_perturbation == "time_stretch":
        speed_factor_list = [0.7, 0.9, 1.1, 1.3, 1.5]
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        for speed_factor in speed_factor_list:
            output_dir_pert = os.path.join(output_dir, f'time_stretch_speed_{speed_factor}')
            os.makedirs(output_dir_pert, exist_ok=True)

            #file_list = [file for file in os.listdir(output_dir) if file.endswith('.wav')]
            #progress_bar = tqdm(input_wav, desc=f"Applying time stretch (speed {speed_factor})")

            #for file in progress_bar:
            try:
                
                waveform, sample_rate = torchaudio.load(input_wav)

                # Apply time stretch
                waveform_pert, new_sample_rate = pert_time_stretch(waveform, sample_rate, speed_factor)

               
                output_path = os.path.join(output_dir_pert, filename)
                if waveform_pert.ndim == 1:
                    waveform_pert = waveform_pert.unsqueeze(0)
                elif waveform_pert.shape[0] != 1:
                    waveform_pert = waveform_pert[0:1]  

                torchaudio.save(output_path, waveform_pert.cpu(), new_sample_rate)

            except Exception as e:
                print("Failed")

    elif common_perturbation in ["gaussian_noise", "background_noise"]:
        snr_values = [40,30, 20,10, 5]
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        for snr in snr_values:

            output_dir_pert = os.path.join(output_dir, f'{common_perturbation}_{snr}')
            #file_list = [file for file in os.listdir(output_dir) if file.endswith('.wav')]
            #progress_bar = tqdm(input_wav, desc=f"Applying {common_perturbation} (snr {snr})")

            if common_perturbation == "gaussian_noise":

                os.makedirs(output_dir_pert, exist_ok=True)


                
                try:
                    
                    waveform, sample_rate = torchaudio.load(input_wav)
                    waveform_pert = pert_Gaussian_noise(waveform, snr)
                    output_path = os.path.join(output_dir_pert, filename)
                    if waveform_pert.ndim == 1:
                        waveform_pert = waveform_pert.unsqueeze(0)
                    elif waveform_pert.shape[0] != 1:
                        waveform_pert = waveform_pert[0:1]

                    torchaudio.save(output_path, waveform_pert.cpu(), sample_rate)
                except Exception as e:
                    print("Failed")

            elif common_perturbation == "background_noise":
                    noise_dir = "noises_wav"
                    noise_files = [f for f in os.listdir(noise_dir) if f.endswith('.wav')]
                    for noise_file in noise_files:
                        noise_file_path = os.path.join(noise_dir, noise_file)
                        output_dir_pert_noise = os.path.join(output_dir_pert, noise_file.split('.')[0])

                        os.makedirs(output_dir_pert_noise, exist_ok=True)


                        
                        try:
                            
                            waveform, sample_rate = torchaudio.load(input_wav)
                            waveform_pert = pert_background_noise(waveform, snr, noise_file_path)
                            output_path = os.path.join(output_dir_pert_noise, filename)
                            if waveform_pert.ndim == 1:
                                waveform_pert = waveform_pert.unsqueeze(0)
                            elif waveform_pert.shape[0] != 1:
                                waveform_pert = waveform_pert[0:1]

                            torchaudio.save(output_path, waveform_pert.cpu(), sample_rate)
                        except Exception as e:
                            print("Failed")


    elif common_perturbation in ["opus"]:
        bitrate_list = [1, 2, 4, 8 ,16, 31]
        for bitrate in bitrate_list:

            output_dir_pert = os.path.join(output_dir, f'{common_perturbation}_bitrate_{bitrate*16}k')
            os.makedirs(output_dir_pert, exist_ok=True)

            #file_list = [file for file in os.listdir(output_dir) if file.endswith('.wav')]
            #progress_bar = tqdm(file_list, desc=f"Applying {common_perturbation} (bitrate {bitrate*16})k")


           
            try:
                    

                    waveform, sample_rate = torchaudio.load(input_wav)
                    waveform_pert = pert_opus(waveform, bitrate = 1000 * bitrate, quality = 1, cache = output_dir_pert)


                    output_path = os.path.join(output_dir_pert, filename)
                    if waveform_pert.ndim == 1:
                        waveform_pert = waveform_pert.unsqueeze(0)
                    elif waveform_pert.shape[0] != 1:
                        waveform_pert = waveform_pert[0:1]

                    torchaudio.save(output_path, waveform_pert.cpu(), sample_rate)

            except Exception as e:
                print("Failed")

    elif common_perturbation in ["encodec"]:
        from transformers import EncodecModel, AutoProcessor
        import warnings
        warnings.filterwarnings("ignore", message=".*Could not find image processor class.*feature_extractor_type.*")
        # Load the ENCODeC model and processor
        model_encodec = EncodecModel.from_pretrained("facebook/encodec_24khz")
        processor_encodec = AutoProcessor.from_pretrained("facebook/encodec_24khz")
        bandwidth_values = [1.5, 3.0, 6.0, 12.0, 24.0]



        # bandwidth_values = [24.0]
        for bandwidth in bandwidth_values:

            output_dir_pert = os.path.join(output_dir, f'{common_perturbation}_bandwidth_{bandwidth}')
            os.makedirs(output_dir_pert, exist_ok=True)


            #file_list = [file for file in os.listdir(output_dir) if file.endswith('.wav')]
            #progress_bar = tqdm(file_list, desc=f"Applying {common_perturbation} (bandwidth {bandwidth})")


            
            try:
                
                waveform, sample_rate = torchaudio.load(input_wav)
                waveform_pert = pert_encodec(waveform, 24000, bandwidth, model_encodec, processor_encodec)
                output_path = os.path.join(output_dir_pert, filename)
                if waveform_pert.ndim == 1:
                    waveform_pert = waveform_pert.unsqueeze(0)
                elif waveform_pert.shape[0] != 1:
                    waveform_pert = waveform_pert[0:1]

                torchaudio.save(output_path, waveform_pert.cpu(), sample_rate)

            except Exception as e:
                print("Failed")

    elif common_perturbation in ["quantization"]:
      import math
      quantization_levels = [2**2, 2**3, 2**4, 2**5, 2**6]
      for quantization_bit in quantization_levels:

          output_dir_pert = os.path.join(output_dir, f'{common_perturbation}_quantization_bit_{quantization_bit}')
          os.makedirs(output_dir_pert, exist_ok=True)


          #file_list = [file for file in os.listdir(output_dir) if file.endswith('.wav')]
          #progress_bar = tqdm(file_list, desc=f"Applying {common_perturbation} (quantization_bit {quantization_bit})")


          
          try:

            

            waveform, sample_rate = torchaudio.load(input_wav)
            waveform_pert = pert_quantization(waveform,quantization_bit)


            output_path = os.path.join(output_dir_pert, filename)
            if waveform_pert.ndim == 1:
                waveform_pert = waveform_pert.unsqueeze(0)
            elif waveform_pert.shape[0] != 1:
                waveform_pert = waveform_pert[0:1]

            torchaudio.save(output_path, waveform_pert.cpu(), sample_rate)
          except Exception as e:
            print("Failed")

    elif common_perturbation in ["highpass", "lowpass"]:
        ratio_list = [0.1, 0.2, 0.3, 0.4, 0.5]
        for ratio in ratio_list:

            output_dir_pert = os.path.join(output_dir, f'{common_perturbation}_ratio_{ratio}')
            os.makedirs(output_dir_pert, exist_ok=True)


            #file_list = [file for file in os.listdir(output_dir) if file.endswith('.wav')]
            #progress_bar = tqdm(file_list, desc=f"Applying {common_perturbation} (ratio {ratio})")

            
            try:

                

                waveform, sample_rate = torchaudio.load(input_wav)
                if common_perturbation == 'highpass':
                    waveform_pert = pert_highpass(waveform,ratio,sample_rate)
                elif common_perturbation == 'lowpass':
                    waveform_pert = pert_lowpass(waveform,ratio,sample_rate)
                output_path = os.path.join(output_dir_pert, filename)
                if waveform_pert.ndim == 1:
                    waveform_pert = waveform_pert.unsqueeze(0)
                elif waveform_pert.shape[0] != 1:
                    waveform_pert = waveform_pert[0:1]

                torchaudio.save(output_path, waveform_pert.cpu(), sample_rate)

            except Exception as e:
                print("Failed")

    elif common_perturbation in ["smooth"]:
        window_list = [6, 10, 14, 18, 22]
        for window_size in window_list:

            output_dir_pert = os.path.join(output_dir, f'{common_perturbation}_window_size_{window_size}')
            os.makedirs(output_dir_pert, exist_ok=True)


            #file_list = [file for file in os.listdir(output_dir) if file.endswith('.wav')]
            #progress_bar = tqdm(file_list, desc=f"Applying {common_perturbation} (window_size {window_size})")


            
            try:


                waveform, sample_rate = torchaudio.load(input_wav)
                waveform_pert = pert_smooth(waveform, window_size)

                output_path = os.path.join(output_dir_pert, filename)
                if waveform_pert.ndim == 1:
                    waveform_pert = waveform_pert.unsqueeze(0)
                elif waveform_pert.shape[0] != 1:
                    waveform_pert = waveform_pert[0:1]

                torchaudio.save(output_path, waveform_pert.cpu(), sample_rate)

            except Exception as e:
                print("Failed")

    elif common_perturbation in ["echo"]:
        decay_list = [0.1, 0.3, 0.5, 0.7, 0.9]
        for decay in decay_list:

            output_dir_pert = os.path.join(output_dir, f'{common_perturbation}_decay{decay}')
            os.makedirs(output_dir_pert, exist_ok=True)


            #file_list = [file for file in os.listdir(output_dir) if file.endswith('.wav')]
            #progress_bar = tqdm(file_list, desc=f"Applying {common_perturbation} (decay {decay})")



           
            try:


                
                waveform, sample_rate = torchaudio.load(input_wav)
                waveform_pert = pert_echo(waveform, duration=decay)
                output_path = os.path.join(output_dir_pert, filename)
                if waveform_pert.ndim == 1:
                    waveform_pert = waveform_pert.unsqueeze(0)
                elif waveform_pert.shape[0] != 1:
                    waveform_pert = waveform_pert[0:1]

                torchaudio.save(output_path, waveform_pert.cpu(), sample_rate)

            except Exception as e:
                print("Failed")

    elif common_perturbation in ["mp3"]:
        bitrate_list = [8, 16, 24, 32, 40]
        for bitrate in bitrate_list:

            output_dir_pert = os.path.join(output_dir, f'{common_perturbation}_bitrate_{bitrate}')
            os.makedirs(output_dir_pert, exist_ok=True)


            #file_list = [file for file in os.listdir(output_dir) if file.endswith('.wav')]
            #progress_bar = tqdm(file_list, desc=f"Applying {common_perturbation} (bitrate {bitrate})")



           
            try:

                
                waveform, sample_rate = torchaudio.load(input_wav)
                waveform_pert = pert_mp3(waveform, bitrate)
                output_path = os.path.join(output_dir_pert, filename)
                if waveform_pert.ndim == 1:
                    waveform_pert = waveform_pert.unsqueeze(0)
                elif waveform_pert.shape[0] != 1:
                    waveform_pert = waveform_pert[0:1]

                torchaudio.save(output_path, waveform_pert.cpu(), sample_rate)

            except Exception as e:
                print("Failed")


    elif common_perturbation in ["aac"]:
        bitrate_list = ['8k',  '40k']
        for bitrate in bitrate_list:
            if common_perturbation == "aac":
                output_dir_pert = os.path.join(output_dir, f'{common_perturbation}_bitrate_{bitrate}')
                pert_func = aac_wrapper
            

            os.makedirs(output_dir_pert, exist_ok=True)

            #file_list = [file for file in os.listdir(output_dir) if file.endswith('.wav')]
            #progress_bar = tqdm(file_list, desc=f"Applying {common_perturbation} (bitrate {bitrate})")

           
            try:
                
                waveform, sample_rate = torchaudio.load(input_wav)

                if waveform.dim() == 1:
                    waveform = waveform.unsqueeze(0).unsqueeze(0) 
                elif waveform.dim() == 2:
                    waveform = waveform.unsqueeze(0) 

                waveform_pert = pert_func(waveform, sr=sample_rate, bitrate=bitrate)

                if waveform_pert.dim() == 3 and waveform_pert.shape[0] == 1:
                     waveform_pert = waveform_pert.squeeze(0) 

                output_path = os.path.join(output_dir_pert, filename)

                torchaudio.save(output_path, waveform_pert.cpu(), sample_rate)

            except Exception as e:
                print("Failed")
    

    elif common_perturbation in ["dynamic_compression", "dynamic_expansion"]:
        
        if common_perturbation == "dynamic_compression":
            params_list = [
                {"threshold": -10, "ratio": 2.0},
                
                {"threshold": -30, "ratio": 8.0},
            ]
            pert_func = dynamic_range_compression
        elif common_perturbation == "dynamic_expansion":
            params_list = [
                {"threshold": -10, "ratio": 2.0},
                
                {"threshold": -30, "ratio": 8.0},
            ]
            pert_func = dynamic_range_expansion 


        for params in params_list:
            threshold = params["threshold"]
            ratio = params["ratio"]
            output_dir_pert = os.path.join(output_dir, '{common_perturbation}_thresh_{threshold}_ratio_{ratio}')
            os.makedirs(output_dir_pert, exist_ok=True)

            #file_list = [file for file in os.listdir(output_dir) if file.endswith('.wav')]
            #progress_bar = tqdm(file_list, desc=f"Applying {common_perturbation} (thresh {threshold}, ratio {ratio})")

           
            try:
                
                waveform, sample_rate = torchaudio.load(input_wav)

                if waveform.dim() == 1: 
                    waveform = waveform.unsqueeze(0)

                waveform_pert = pert_func(audio=waveform, threshold=threshold, ratio=ratio, sr=sample_rate)

                output_path = os.path.join(output_dir_pert, filename)

                torchaudio.save(output_path, waveform_pert.cpu(), sample_rate)

            except Exception as e:
                print("Failed")

    

    elif common_perturbation == "inverse_polarity":
       
        output_dir_pert = os.path.join(output_dir, f'{common_perturbation}')
        os.makedirs(output_dir_pert, exist_ok=True)

        #file_list = [file for file in os.listdir(output_dir) if file.endswith('.wav')]
        #progress_bar = tqdm(file_list, desc=f"Applying {common_perturbation}")

        
        try:
            
            waveform, sample_rate = torchaudio.load(input_wav)

            waveform_pert = inverse_polarity(waveform)

            output_path = os.path.join(output_dir_pert, filename)

            torchaudio.save(output_path, waveform_pert.cpu(), sample_rate)

        except Exception as e:
            print("Failed")

    elif common_perturbation == "time_jitter":
        
        scale_list = [0.01,  0.5] 

        for scale in scale_list:
            output_dir_pert = os.path.join(output_dir, f'{common_perturbation}_scale_{scale}')
            os.makedirs(output_dir_pert, exist_ok=True)

            #file_list = [file for file in os.listdir(output_dir) if file.endswith('.wav')]
            #progress_bar = tqdm(file_list, desc=f"Applying {common_perturbation} (scale {scale})")

            
            try:
                
                waveform, sample_rate = torchaudio.load(input_wav)

                waveform_pert = time_jitter(waveform, scale=scale)

                output_path = os.path.join(output_dir_pert, filename)

                torchaudio.save(output_path, waveform_pert.cpu(), sample_rate)

            except Exception as e:
                print("Failed")

    elif common_perturbation == "phase_shift":
        
        shift_list = [1, -1000]

        for shift in shift_list:
            output_dir_pert = os.path.join(output_dir, f'{common_perturbation}_shift_{shift}')
            os.makedirs(output_dir_pert, exist_ok=True)

            #file_list = [file for file in os.listdir(output_dir) if file.endswith('.wav')]
            #progress_bar = tqdm(file_list, desc=f"Applying {common_perturbation} (shift {shift})")

          
            try:
                
                waveform, sample_rate = torchaudio.load(input_wav)

                
                waveform_pert = phase_shift(waveform, shift=shift)

                output_path = os.path.join(output_dir_pert, filename)

                torchaudio.save(output_path, waveform_pert.cpu(), sample_rate)

            except Exception as e:
                print("Failed")

    return output_path

    

   




"""def no_box_perturbation(input_dir, output_root, perturbation):
    dataset_dir = "path to your watermarked or unwatermarked dataset"

    perturbations= ['time_stretch','gaussian_noise','background_noise', 'opus', 
                    'encodec', 'quantization', 'highpass', 'lowpass', 'smooth', 
                    'echo', 'mp3', 'aac', 'dynamic_compression', 'dynamic_expansion',
                    'inverse_polarity', 'time_jitter', 'phase_shift']

    #for pert in perturbations:
    apply_no_box_pert(dataset_dir, pert)"""
